from typing import Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..chemical_properties import ChemicalDatabase
else:
    from chemical_properties import ChemicalDatabase

from .common import ThermodynamicsError, _solve_bubble_point_temperature
from .henry import AqueousEquilibriumContext
from .unifac_models import UNIFACThermodynamics
from .nrtl_uniquac import NRTLThermodynamics, UNIQUACThermodynamics

class GammaPhiVaporBackendMixin:
    """Shared gamma-phi vapor-fugacity backend glue for activity models."""

    vapor_backend_model = 'RK'
    vapor_backend_label = 'RK'

    def _initialize_gamma_phi_backend(
        self,
        components: list[str],
        db: Optional[ChemicalDatabase],
        interaction_overrides: Optional[list[dict]] = None,
    ) -> None:
        model = self.vapor_backend_model.upper()
        if model == 'RK':
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..rk_eos import RedlichKwong, RKError
            else:
                from rk_eos import RedlichKwong, RKError

            try:
                self.vapor_eos = RedlichKwong(components, db)
            except RKError as e:
                raise ThermodynamicsError(f"Failed to initialize RK vapor correction: {e}") from e
        elif model == 'PR':
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..cubic_eos import CubicEOS, CubicEOSError
            else:
                from cubic_eos import CubicEOS, CubicEOSError

            try:
                self.vapor_eos = CubicEOS(components, 'PR', db, interaction_overrides)
            except CubicEOSError as e:
                raise ThermodynamicsError(f"Failed to initialize PR vapor correction: {e}") from e
        else:
            raise ThermodynamicsError(f"Unsupported gamma-phi vapor backend: {model}")
        self.extend_warnings(getattr(self.vapor_eos, 'warnings', []))

    def fugacity_coefficients(self, T: float, P: float,
                               composition: dict[str, float],
                               phase: str = 'vapor') -> dict[str, float]:
        return self.vapor_eos.fugacity_coefficients(T, P, composition, phase)

    def K_values(self, T: float, P: float,
                 composition: dict[str, float]) -> dict[str, float]:
        return self._gamma_phi_K_values(T, P, composition)

    def aqueous_K_values(self, T: float, P: float,
                         composition: dict[str, float],
                         context: AqueousEquilibriumContext) -> dict[str, float]:
        """Gamma-phi K-values with frozen Henry liquid standard states."""
        self._warn_aqueous_henry_pressure(P, context)
        x = self._normalized_aqueous_composition(composition, self.components)
        cache_key = self._k_values_cache_key('aqueous_gamma_phi', T, P, x) + (
            self.vapor_backend_model,
            context.cache_key(),
        )
        cached = self._get_cached_k_values(cache_key)
        if cached is not None:
            return cached

        gamma = self.activity_coefficients(
            T,
            self._aqueous_bulk_activity_composition(x, context),
        )
        reference_K = {}
        for comp in self.components:
            if comp in context.component_data:
                value = self._henry_ideal_vapor_K_value(comp, T, P, context)
            else:
                Psat = self.Psat(comp, T)
                phi_sat = self._gamma_phi_phi_sat(comp, T, Psat)
                poynting = self._gamma_phi_poynting_factor(comp, T, P, Psat)
                value = (
                    gamma.get(comp, 1.0) * phi_sat * Psat * poynting
                    / max(P, 1e-12)
                )
            reference_K[comp] = max(1e-12, min(1e12, float(value)))

        y = {comp: x.get(comp, 0.0) * reference_K[comp] for comp in self.components}
        y_total = sum(y.values())
        y = (
            {comp: value / y_total for comp, value in y.items()}
            if y_total > 0.0 else dict(x)
        )
        K = dict(reference_K)
        for _ in range(12):
            try:
                phi_v = self.vapor_eos.fugacity_coefficients(T, P, y, 'vapor')
            except Exception:
                phi_v = {comp: 1.0 for comp in self.components}
            K = {
                comp: max(
                    1e-12,
                    min(1e12, reference_K[comp] / max(phi_v.get(comp, 1.0), 1e-12)),
                )
                for comp in self.components
            }
            y_new = {comp: x.get(comp, 0.0) * K[comp] for comp in self.components}
            y_total = sum(y_new.values())
            if y_total <= 0.0:
                break
            y_new = {comp: value / y_total for comp, value in y_new.items()}
            if max(abs(y_new[comp] - y.get(comp, 0.0)) for comp in self.components) < 1e-9:
                break
            y = y_new
        return self._set_cached_k_values(cache_key, K)

    def K_value(self, comp: str, T: float, P: float,
                x: Optional[dict[str, float]] = None) -> float:
        composition = x if x is not None else {comp: 1.0}
        return self.K_values(T, P, composition).get(comp, 1.0)


class UNIQUACRKThermodynamics(GammaPhiVaporBackendMixin, UNIQUACThermodynamics):
    """UNIQUAC-RK gamma-phi model using RK vapor fugacity coefficients."""

    vapor_backend_model = 'RK'

    def __init__(
        self,
        components: list[str],
        db: Optional[ChemicalDatabase] = None,
        interaction_overrides: Optional[list[dict]] = None,
        interaction_estimation: Optional[list[dict]] = None,
        estimation_unifac_groups: Optional[dict] = None,
        activity_interaction_max_psat_bar: Optional[float] = 10.0,
        activity_interaction_max_temperature_K: Optional[float] = None,
    ):
        super().__init__(components, db, interaction_overrides, interaction_estimation, estimation_unifac_groups, activity_interaction_max_psat_bar, activity_interaction_max_temperature_K)
        self._initialize_gamma_phi_backend(components, db, interaction_overrides)


class UNIQUACPRThermodynamics(GammaPhiVaporBackendMixin, UNIQUACThermodynamics):
    """UNIQUAC-PR gamma-phi model using PR vapor fugacity coefficients."""

    vapor_backend_model = 'PR'

    def __init__(
        self,
        components: list[str],
        db: Optional[ChemicalDatabase] = None,
        interaction_overrides: Optional[list[dict]] = None,
        interaction_estimation: Optional[list[dict]] = None,
        estimation_unifac_groups: Optional[dict] = None,
        activity_interaction_max_psat_bar: Optional[float] = 10.0,
        activity_interaction_max_temperature_K: Optional[float] = None,
    ):
        super().__init__(components, db, interaction_overrides, interaction_estimation, estimation_unifac_groups, activity_interaction_max_psat_bar, activity_interaction_max_temperature_K)
        self._initialize_gamma_phi_backend(components, db, interaction_overrides)


class NRTLRKThermodynamics(GammaPhiVaporBackendMixin, NRTLThermodynamics):
    """NRTL-RK gamma-phi model using RK vapor fugacity coefficients."""

    vapor_backend_model = 'RK'

    def __init__(
        self,
        components: list[str],
        db: Optional[ChemicalDatabase] = None,
        interaction_overrides: Optional[list[dict]] = None,
        interaction_estimation: Optional[list[dict]] = None,
        estimation_unifac_groups: Optional[dict] = None,
        activity_interaction_max_psat_bar: Optional[float] = 10.0,
        activity_interaction_max_temperature_K: Optional[float] = None,
    ):
        super().__init__(components, db, interaction_overrides, interaction_estimation, estimation_unifac_groups, activity_interaction_max_psat_bar, activity_interaction_max_temperature_K)
        self._initialize_gamma_phi_backend(components, db, interaction_overrides)


class NRTLPRThermodynamics(GammaPhiVaporBackendMixin, NRTLThermodynamics):
    """NRTL-PR gamma-phi model using PR vapor fugacity coefficients."""

    vapor_backend_model = 'PR'

    def __init__(
        self,
        components: list[str],
        db: Optional[ChemicalDatabase] = None,
        interaction_overrides: Optional[list[dict]] = None,
        interaction_estimation: Optional[list[dict]] = None,
        estimation_unifac_groups: Optional[dict] = None,
        activity_interaction_max_psat_bar: Optional[float] = 10.0,
        activity_interaction_max_temperature_K: Optional[float] = None,
    ):
        super().__init__(components, db, interaction_overrides, interaction_estimation, estimation_unifac_groups, activity_interaction_max_psat_bar, activity_interaction_max_temperature_K)
        self._initialize_gamma_phi_backend(components, db, interaction_overrides)


class UNIFACRKThermodynamics(GammaPhiVaporBackendMixin, UNIFACThermodynamics):
    """
    Gamma-phi VLE model with UNIFAC liquid activity coefficients and RK vapor
    fugacity coefficients.

    Uses K_i = gamma_i * Psat_i / (phi_i^V * P). The vapor fugacity
    coefficients are evaluated with the RK EOS at the estimated equilibrium
    vapor composition.
    """

    vapor_backend_model = 'RK'

    def __init__(self, components: list[str],
                 db: Optional[ChemicalDatabase] = None,
                 unifac_groups: Optional[dict[str, dict]] = None,
                 interaction_overrides: Optional[list[dict]] = None):
        super().__init__(
            components,
            db,
            unifac_groups,
            interaction_overrides=interaction_overrides,
        )

        missing_tc = []
        for comp in components:
            props = self.props[comp]
            if props.Tc is None or props.Pc is None:
                missing_tc.append(comp)

        if missing_tc:
            raise ThermodynamicsError(
                f"{self.__class__.__name__} requires critical properties (Tc, Pc) for all components. "
                f"Missing for: {', '.join(missing_tc)}"
            )

        self._initialize_gamma_phi_backend(components, db, interaction_overrides)

    def bubble_point_T(self, composition: dict[str, float], P: float,
                       T_guess: float = 350.0) -> float:
        return _solve_bubble_point_temperature(self, composition, P, T_guess)


class UNIFACPRThermodynamics(UNIFACRKThermodynamics):
    """Gamma-phi model with UNIFAC liquid activity and PR vapor fugacity."""

    vapor_backend_model = 'PR'


class UNIFDMDRKThermodynamics(UNIFACRKThermodynamics):
    """Gamma-phi model with Dortmund modified UNIFAC liquid activity."""

    unifac_variant = 'UNIFDMD'
    default_unifac_data_filename = 'unifac_dmd.txt'


class UNIFDMDPRThermodynamics(UNIFACPRThermodynamics):
    """Gamma-phi model with Dortmund modified UNIFAC liquid activity and PR vapor fugacity."""

    unifac_variant = 'UNIFDMD'
    default_unifac_data_filename = 'unifac_dmd.txt'


class UNIFNISTRKThermodynamics(UNIFACRKThermodynamics):
    """Gamma-phi model with NIST-modified UNIFAC liquid activity."""

    unifac_variant = 'UNIFNIST'
    default_unifac_data_filename = 'nist_modified_unifac_params.json'


class UNIFNISTPRThermodynamics(UNIFACPRThermodynamics):
    """Gamma-phi model with NIST-modified UNIFAC liquid activity and PR vapor fugacity."""

    unifac_variant = 'UNIFNIST'
    default_unifac_data_filename = 'nist_modified_unifac_params.json'
