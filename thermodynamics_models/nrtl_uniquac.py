import math
from typing import Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..chemical_properties import ChemicalDatabase
else:
    from chemical_properties import ChemicalDatabase

from .common import ThermodynamicsError
from .activity import ActivityCoefficientThermodynamics, VaporDimerizationActivityMixin
from .base import IdealThermodynamics


def _anchored_log_temperature_term(T: float, T_ref: float) -> float:
    return (T_ref - T) / T + math.log(T / T_ref)

def _interaction_override_map(
    interaction_overrides: Optional[list[dict]],
    model: str,
) -> dict[tuple[str, str], dict]:
    records = {}
    model_key = model.upper()
    for record in interaction_overrides or []:
        if str(record.get("model", "")).upper() != model_key:
            continue
        comp1 = record.get("component1")
        comp2 = record.get("component2")
        if not comp1 or not comp2 or comp1 == comp2:
            continue
        key = tuple(sorted((comp1, comp2)))
        records[key] = dict(record)
    return records


def _oriented_component_override(
    overrides: dict[tuple[str, str], dict],
    comp_i: str,
    comp_j: str,
    orienter,
) -> Optional[dict]:
    record = overrides.get(tuple(sorted((comp_i, comp_j))))
    if record is None:
        return None
    return orienter(record, reverse=record.get("component1") != comp_i)


class NRTLThermodynamics(ActivityCoefficientThermodynamics):
    """NRTL liquid activity coefficient model with Raoult-law vapor phase."""

    R_CAL = 1.98720425864083
    T_REF = 298.15

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
        super().__init__(components, db, interaction_overrides)
        self.configure_activity_interaction_limits(
            max_psat_bar=activity_interaction_max_psat_bar,
            max_temperature_K=activity_interaction_max_temperature_K,
        )
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..interaction_parameters import cas_for_component
        else:
            from interaction_parameters import cas_for_component

        self.component_cas = {
            comp: cas_for_component(comp, self.props.get(comp))
            for comp in components
        }
        self._nrtl_interaction_overrides = _interaction_override_map(
            interaction_overrides,
            "NRTL",
        )
        self._nrtl_estimated_interactions: dict[tuple[str, str], dict] = {}
        self.estimated_interaction_metadata: dict[tuple[str, str], dict] = {}
        self._activity_cache: dict[tuple, dict[str, float]] = {}
        self._nrtl_matrix_cache: dict[
            float,
            tuple[list[list[float]], list[list[float]], list[list[float]]],
        ] = {}
        self._nrtl_parameter_cache: Optional[dict[str, list[list[float]]]] = None
        self._compiled_activity_cache: dict[str, object] = {}
        self._compiled_lle_cache: dict[str, object] = {}
        self._compiled_vlle = None
        self._compiled_vlle_initialized = False
        if interaction_estimation:
            from .interaction_estimation import estimate_missing_interactions
            estimated, metadata = estimate_missing_interactions(
                self,
                'NRTL',
                interaction_estimation,
                self._nrtl_interaction_for_components,
                estimation_unifac_groups,
                set(self._nrtl_interaction_overrides),
            )
            self._nrtl_estimated_interactions = _interaction_override_map(
                estimated, 'NRTL'
            )
            self.estimated_interaction_metadata = metadata
        self._warn_missing_nrtl_interactions()

    def prepare_compiled_backends(
        self,
        *,
        need_lle: bool = False,
        need_vlle: bool = False,
    ) -> None:
        activity = self._compiled_activity_backend(self.T_REF)
        if activity is not None:
            activity.compile_kernels()
        if need_lle:
            backend = self._compiled_lle_backend(self.T_REF)
            if backend is not None:
                backend.compile_kernels()
        if need_vlle:
            backend = self.compiled_vlle_backend()
            if backend is not None:
                backend.compile_kernels()

    def _nrtl_interaction_for_components(self, comp_i: str, comp_j: str) -> Optional[dict]:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..interaction_parameters import nrtl_binary_interaction, orient_nrtl_interaction
        else:
            from interaction_parameters import nrtl_binary_interaction, orient_nrtl_interaction

        override = _oriented_component_override(
            self._nrtl_interaction_overrides,
            comp_i,
            comp_j,
            orient_nrtl_interaction,
        )
        if override is not None:
            return override
        estimated = _oriented_component_override(
            self._nrtl_estimated_interactions,
            comp_i,
            comp_j,
            orient_nrtl_interaction,
        )
        if estimated is not None:
            return estimated
        database = nrtl_binary_interaction(
            self.component_cas.get(comp_i),
            self.component_cas.get(comp_j),
        )
        if database is not None:
            return database
        return None

    def _warn_missing_nrtl_interactions(self) -> None:
        for i, comp_i in enumerate(self.components):
            for comp_j in self.components[i + 1:]:
                if self._nrtl_interaction_for_components(comp_i, comp_j) is not None:
                    continue
                cas_i = self.component_cas.get(comp_i)
                cas_j = self.component_cas.get(comp_j)
                if not cas_i or not cas_j:
                    missing = [
                        comp for comp, comp_cas in ((comp_i, cas_i), (comp_j, cas_j))
                        if not comp_cas
                    ]
                    self.add_warning(
                        f"NRTL binary interaction parameters unavailable for "
                        f"{comp_i}/{comp_j}: missing CAS for {', '.join(missing)}; "
                        "using tau_ij=tau_ji=0 with alpha_ij=0.3."
                    )
                else:
                    self.add_warning(
                        f"NRTL binary interaction parameters missing for "
                        f"{comp_i}/{comp_j}; using tau_ij=tau_ji=0 with alpha_ij=0.3."
                    )

    def _compiled_activity_backend(self, T: float):
        cache_key = "all_temperatures"
        if cache_key in self._compiled_activity_cache:
            return self._compiled_activity_cache[cache_key]
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..compiled_activity import CompiledNRTLBackend
            else:
                from compiled_activity import CompiledNRTLBackend

            backend = CompiledNRTLBackend.from_thermo(self)
        except Exception:
            backend = None
        self._compiled_activity_cache[cache_key] = backend
        return backend

    def _compiled_lle_backend(self, T: float):
        cache_key = "all_temperatures"
        if cache_key in self._compiled_lle_cache:
            return self._compiled_lle_cache[cache_key]
        backend = None
        activity_backend = self._compiled_activity_backend(T)
        if activity_backend is not None:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compiled_lle import CompiledNRTLLLEBackend
                else:
                    from compiled_lle import CompiledNRTLLLEBackend

                backend = CompiledNRTLLLEBackend.from_activity_backend(activity_backend)
            except Exception:
                backend = None
        self._compiled_lle_cache[cache_key] = backend
        return backend

    def liquid_liquid_equilibrium(self, composition: dict[str, float], T: float,
                                  max_iter: int = 100, tol: float = 1e-6) -> tuple[bool, dict, dict, float]:
        backend = self._compiled_lle_backend(T)
        if backend is not None:
            split = backend.split(composition, T, max_iter=max_iter, tol=tol)
            if split is not None:
                return split
        return super().liquid_liquid_equilibrium(composition, T, max_iter, tol)

    def _nrtl_cached_matrices(
        self,
        T: float,
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        T_key = float(T)
        cached = self._nrtl_matrix_cache.get(T_key)
        if cached is not None:
            return cached

        n = len(self.components)
        tau = [[0.0 for _ in range(n)] for _ in range(n)]
        alpha = [[0.3 for _ in range(n)] for _ in range(n)]
        for i, comp_i in enumerate(self.components):
            for j, comp_j in enumerate(self.components):
                if i == j:
                    alpha[i][j] = 0.0
                    continue
                data = self._nrtl_interaction_for_components(comp_i, comp_j)
                if data is None:
                    continue
                interaction_T = self.activity_interaction_temperature(
                    comp_i,
                    comp_j,
                    T,
                )
                if "tau12_c" in data:
                    tref = data.get("tau_tref", self.T_REF)
                    tau[i][j] = (
                        data["tau12_c"]
                        + data.get("tau12_d", 0.0) / interaction_T
                        + data.get("tau12_e", 0.0)
                        * _anchored_log_temperature_term(interaction_T, tref)
                        + data.get("tau12_f", 0.0) * interaction_T
                        + data.get("tau12_g", 0.0) * interaction_T * interaction_T
                    )
                else:
                    tau[i][j] = (
                        data["a12_cal_per_mol"]
                        / (self.R_CAL * interaction_T)
                    )
                alpha[i][j] = data["alpha12"]

        G = [
            [math.exp(-alpha[i][j] * tau[i][j]) for j in range(n)]
            for i in range(n)
        ]
        if len(self._nrtl_matrix_cache) > 256:
            self._nrtl_matrix_cache.clear()
        self._nrtl_matrix_cache[T_key] = (tau, alpha, G)
        return tau, alpha, G

    def _nrtl_parameter_matrices(self) -> dict[str, list[list[float]]]:
        if self._nrtl_parameter_cache is not None:
            return self._nrtl_parameter_cache
        n = len(self.components)
        tau_mode = [[0 for _ in range(n)] for _ in range(n)]
        tau_c = [[0.0 for _ in range(n)] for _ in range(n)]
        tau_d = [[0.0 for _ in range(n)] for _ in range(n)]
        tau_e = [[0.0 for _ in range(n)] for _ in range(n)]
        tau_f = [[0.0 for _ in range(n)] for _ in range(n)]
        tau_g = [[0.0 for _ in range(n)] for _ in range(n)]
        tau_tref = [[self.T_REF for _ in range(n)] for _ in range(n)]
        tau_energy = [[0.0 for _ in range(n)] for _ in range(n)]
        alpha = [[0.3 for _ in range(n)] for _ in range(n)]

        for i, comp_i in enumerate(self.components):
            for j, comp_j in enumerate(self.components):
                if i == j:
                    alpha[i][j] = 0.0
                    continue
                data = self._nrtl_interaction_for_components(comp_i, comp_j)
                if data is None:
                    continue
                alpha[i][j] = data["alpha12"]
                if "tau12_c" in data:
                    tau_mode[i][j] = 1
                    tau_c[i][j] = data["tau12_c"]
                    tau_d[i][j] = data.get("tau12_d", 0.0)
                    tau_e[i][j] = data.get("tau12_e", 0.0)
                    tau_f[i][j] = data.get("tau12_f", 0.0)
                    tau_g[i][j] = data.get("tau12_g", 0.0)
                    tau_tref[i][j] = data.get("tau_tref", self.T_REF)
                else:
                    tau_mode[i][j] = 2
                    tau_energy[i][j] = data["a12_cal_per_mol"] / self.R_CAL

        self._nrtl_parameter_cache = {
            "tau_mode": tau_mode,
            "tau_c": tau_c,
            "tau_d": tau_d,
            "tau_e": tau_e,
            "tau_f": tau_f,
            "tau_g": tau_g,
            "tau_tref": tau_tref,
            "tau_energy": tau_energy,
            "alpha": alpha,
        }
        return self._nrtl_parameter_cache

    def _nrtl_matrices(self, T: float) -> tuple[list[list[float]], list[list[float]]]:
        tau, alpha, _ = self._nrtl_cached_matrices(T)
        return tau, alpha

    def activity_coefficients(self, T: float, composition: dict[str, float]) -> dict[str, float]:
        cache_key = (
            float(T),
            tuple((comp, float(composition.get(comp, 0.0))) for comp in self.components),
        )
        cached = self._activity_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        x = [max(float(composition.get(comp, 0.0)), 0.0) for comp in self.components]
        total = sum(x)
        if total <= 0.0:
            return {comp: 1.0 for comp in self.components}
        x = [value / total for value in x]

        compiled = self._compiled_activity_backend(T)
        if compiled is not None:
            gamma_list = compiled.activity_coefficients(x, T)
            gamma = {
                comp: float(gamma_list[i])
                for i, comp in enumerate(self.components)
            }
            if len(self._activity_cache) > 20000:
                self._activity_cache.clear()
            self._activity_cache[cache_key] = dict(gamma)
            return gamma

        tau, _, G = self._nrtl_cached_matrices(T)
        n = len(self.components)
        denom_col = [
            sum(x[k] * G[k][j] for k in range(n)) or 1e-30
            for j in range(n)
        ]
        weighted_tau_col = [
            sum(x[m] * tau[m][j] * G[m][j] for m in range(n)) / denom_col[j]
            for j in range(n)
        ]

        gamma = {}
        for i, comp_i in enumerate(self.components):
            ln_gamma = 0.0
            for j in range(n):
                ln_gamma += x[j] * tau[j][i] * G[j][i] / denom_col[i]
            for j in range(n):
                ln_gamma += (
                    x[j]
                    * G[i][j]
                    / denom_col[j]
                    * (tau[i][j] - weighted_tau_col[j])
                )
            gamma[comp_i] = max(math.exp(max(min(ln_gamma, 50.0), -50.0)), 1e-12)

        if len(self._activity_cache) > 20000:
            self._activity_cache.clear()
        self._activity_cache[cache_key] = dict(gamma)
        return gamma

    def excess_enthalpy(self, composition: dict[str, float], T: float) -> float:
        backend = self._compiled_activity_backend(T)
        if backend is None:
            return super().excess_enthalpy(composition, T)
        x = [
            max(float(composition.get(comp, 0.0)), 0.0)
            for comp in self.components
        ]
        if sum(x) <= 0.0:
            return 0.0
        return backend.excess_enthalpy(x, T)


class UNIQUACThermodynamics(ActivityCoefficientThermodynamics):
    """UNIQUAC liquid activity coefficient model with Raoult-law vapor phase."""

    R_CAL = 1.98720425864083
    T_REF = 298.15
    Z = 10.0

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
        super().__init__(components, db, interaction_overrides)
        self.configure_activity_interaction_limits(
            max_psat_bar=activity_interaction_max_psat_bar,
            max_temperature_K=activity_interaction_max_temperature_K,
        )
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..interaction_parameters import cas_for_component
        else:
            from interaction_parameters import cas_for_component

        self.component_cas = {
            comp: cas_for_component(comp, self.props.get(comp))
            for comp in components
        }
        self._uniquac_interaction_overrides = _interaction_override_map(
            interaction_overrides,
            "UNIQUAC",
        )
        self._uniquac_estimated_interactions: dict[tuple[str, str], dict] = {}
        self.estimated_interaction_metadata: dict[tuple[str, str], dict] = {}
        self.r: dict[str, float] = {}
        self.q: dict[str, float] = {}
        self.q_prime: dict[str, float] = {}
        self._activity_cache: dict[tuple, dict[str, float]] = {}
        self._uniquac_tau_cache: dict[float, list[list[float]]] = {}
        self._uniquac_parameter_cache: Optional[dict[str, list[list[float]]]] = None
        self._compiled_activity_cache: dict[str, object] = {}
        self._compiled_lle_cache: dict[str, object] = {}
        self._compiled_vlle = None
        self._compiled_vlle_initialized = False
        self._uniquac_rq_warnings: set[str] = set()
        self._deferred_uniquac_rq_errors: dict[str, str] = {}

        for comp in components:
            try:
                data = self._uniquac_rq_data(comp)
            except ThermodynamicsError as error:
                if self.henry_component_data(comp) is None:
                    raise
                self._deferred_uniquac_rq_errors[comp] = str(error)
                continue
            r, q = float(data["r"]), float(data["q"])
            self.r[comp] = r
            self.q[comp] = q
            extended = data.get("extended_uniquac", {}) if isinstance(data, dict) else {}
            self.q_prime[comp] = float(extended.get("q_prime", q))

        if interaction_estimation:
            from .interaction_estimation import estimate_missing_interactions
            estimated, metadata = estimate_missing_interactions(
                self,
                'UNIQUAC',
                interaction_estimation,
                self._uniquac_interaction_for_components,
                estimation_unifac_groups,
                set(self._uniquac_interaction_overrides),
            )
            self._uniquac_estimated_interactions = _interaction_override_map(
                estimated, 'UNIQUAC'
            )
            self.estimated_interaction_metadata = metadata
        self._warn_missing_uniquac_interactions()

    def initialize(self) -> 'UNIQUACThermodynamics':
        """Initialize vapor/property providers without forcing deferred liquid data."""
        if not self._deferred_uniquac_rq_errors:
            return super().initialize()
        IdealThermodynamics.initialize(self)
        self._activity_runtime_initialized = True
        return self

    def prepare_compiled_backends(
        self,
        *,
        need_lle: bool = False,
        need_vlle: bool = False,
    ) -> None:
        activity = self._compiled_activity_backend(298.15)
        if activity is not None:
            activity.compile_kernels()
        if need_lle:
            backend = self._compiled_lle_backend(298.15)
            if backend is not None:
                backend.compile_kernels()
        if need_vlle:
            backend = self.compiled_vlle_backend()
            if backend is not None:
                backend.compile_kernels()

    def _uniquac_interaction_for_components(self, comp_i: str, comp_j: str) -> Optional[dict]:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..interaction_parameters import uniquac_binary_interaction, orient_uniquac_interaction
        else:
            from interaction_parameters import uniquac_binary_interaction, orient_uniquac_interaction

        override = _oriented_component_override(
            self._uniquac_interaction_overrides,
            comp_i,
            comp_j,
            orient_uniquac_interaction,
        )
        if override is not None:
            return override
        estimated = _oriented_component_override(
            self._uniquac_estimated_interactions,
            comp_i,
            comp_j,
            orient_uniquac_interaction,
        )
        if estimated is not None:
            return estimated
        database = uniquac_binary_interaction(
            self.component_cas.get(comp_i),
            self.component_cas.get(comp_j),
        )
        if database is not None:
            return database
        return None

    def _warn_missing_uniquac_interactions(self) -> None:
        for i, comp_i in enumerate(self.components):
            for comp_j in self.components[i + 1:]:
                if (
                    comp_i in self._deferred_uniquac_rq_errors
                    or comp_j in self._deferred_uniquac_rq_errors
                ):
                    continue
                if self._uniquac_interaction_for_components(comp_i, comp_j) is not None:
                    continue
                cas_i = self.component_cas.get(comp_i)
                cas_j = self.component_cas.get(comp_j)
                if not cas_i or not cas_j:
                    missing = [
                        comp for comp, comp_cas in ((comp_i, cas_i), (comp_j, cas_j))
                        if not comp_cas
                    ]
                    self.add_warning(
                        f"UNIQUAC binary interaction parameters unavailable for "
                        f"{comp_i}/{comp_j}: missing CAS for {', '.join(missing)}; "
                        "using tau_ij=tau_ji=1."
                    )
                else:
                    self.add_warning(
                        f"UNIQUAC binary interaction parameters missing for "
                        f"{comp_i}/{comp_j}; using tau_ij=tau_ji=1."
                    )

    def _compiled_activity_backend(self, T: float):
        active_components = tuple(
            comp for comp in self.components
            if comp not in self._deferred_uniquac_rq_errors
        )
        cache_key = active_components
        if cache_key in self._compiled_activity_cache:
            return self._compiled_activity_cache[cache_key]
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..compiled_activity import CompiledUNIQUACBackend
            else:
                from compiled_activity import CompiledUNIQUACBackend

            backend = CompiledUNIQUACBackend.from_thermo(
                self,
                active_components,
            )
        except Exception:
            backend = None
        self._compiled_activity_cache[cache_key] = backend
        return backend

    def _compiled_lle_backend(self, T: float):
        cache_key = "all_temperatures"
        if cache_key in self._compiled_lle_cache:
            return self._compiled_lle_cache[cache_key]
        backend = None
        activity_backend = self._compiled_activity_backend(T)
        if activity_backend is not None:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compiled_lle import CompiledUNIQUACLLEBackend
                else:
                    from compiled_lle import CompiledUNIQUACLLEBackend

                backend = CompiledUNIQUACLLEBackend.from_activity_backend(activity_backend)
            except Exception:
                backend = None
        self._compiled_lle_cache[cache_key] = backend
        return backend

    def liquid_liquid_equilibrium(self, composition: dict[str, float], T: float,
                                  max_iter: int = 100, tol: float = 1e-6) -> tuple[bool, dict, dict, float]:
        backend = self._compiled_lle_backend(T)
        if backend is not None:
            split = backend.split(composition, T, max_iter=max_iter, tol=tol)
            if split is not None:
                return split
        return super().liquid_liquid_equilibrium(composition, T, max_iter, tol)

    def _uniquac_rq(self, comp: str) -> tuple[float, float]:
        data = self._uniquac_rq_data(comp)
        return float(data["r"]), float(data["q"])

    def _uniquac_rq_data(self, comp: str) -> dict:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..interaction_parameters import uniquac_rq_for_component
        else:
            from interaction_parameters import uniquac_rq_for_component

        props = self.props.get(comp)
        data = uniquac_rq_for_component(comp, props)
        if data is not None:
            return dict(data)

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..unifac import UNIFACModel, get_unifac_groups
            else:
                from unifac import UNIFACModel, get_unifac_groups

            identifiers = []
            if props is not None:
                # Process-local component symbols are not chemical identity
                # inputs. Use only the identity resolved from the PFD lookup
                # field and its resulting property record.
                identifiers.extend([props.name, props.CAS, props.formula])
            expected_mw = None
            try:
                expected_mw = float(getattr(props, 'MW', None))
            except (TypeError, ValueError):
                pass
            last_error = None
            for identifier in identifiers:
                if not identifier:
                    continue
                try:
                    groups = get_unifac_groups(
                        identifier,
                        variant='UNIFAC',
                        expected_mw=expected_mw,
                    )
                    r, q = UNIFACModel().calculate_r_q(groups)
                    if comp not in self._uniquac_rq_warnings:
                        self.add_warning(
                            f"UNIQUAC r/q parameters missing for '{comp}'; "
                            "estimated from UNIFAC group volume/area parameters."
                        )
                        self._uniquac_rq_warnings.add(comp)
                    return {"r": r, "q": q}
                except ValueError as exc:
                    last_error = exc

            smiles = getattr(props, 'smiles', None)
            if not smiles and hasattr(self.db, 'resolve_smiles_info'):
                for identifier in identifiers:
                    if not identifier:
                        continue
                    result = self.db.resolve_smiles_info(
                        identifier,
                        fetch_online=True,
                        props=props,
                    )
                    smiles = result.smiles if result else None
                    if smiles:
                        break
            if smiles:
                try:
                    groups = get_unifac_groups(
                        getattr(props, 'name', None) or 'resolved component',
                        smiles=smiles,
                        variant='UNIFAC',
                    )
                    r, q = UNIFACModel().calculate_r_q(groups)
                    if comp not in self._uniquac_rq_warnings:
                        self.add_warning(
                            f"UNIQUAC r/q parameters missing for '{comp}'; "
                            "estimated from UNIFAC group volume/area parameters."
                        )
                        self._uniquac_rq_warnings.add(comp)
                    return {"r": r, "q": q}
                except ValueError as exc:
                    last_error = exc
            if last_error:
                raise last_error
        except ValueError as exc:
            raise ThermodynamicsError(
                f"UNIQUAC r/q parameters not found for '{comp}', and UNIFAC-based "
                f"estimation failed: {exc}"
            ) from exc

        raise ThermodynamicsError(
            f"UNIQUAC r/q parameters not found for '{comp}', and UNIFAC-based "
            "estimation is unavailable."
        )

    def _uniquac_tau_matrix(self, T: float) -> list[list[float]]:
        T_key = float(T)
        cached = self._uniquac_tau_cache.get(T_key)
        if cached is not None:
            return cached

        n = len(self.components)
        tau = [[1.0 for _ in range(n)] for _ in range(n)]
        for i, comp_i in enumerate(self.components):
            for j, comp_j in enumerate(self.components):
                if i == j:
                    continue
                data = self._uniquac_interaction_for_components(comp_i, comp_j)
                if data is None:
                    continue
                interaction_T = self.activity_interaction_temperature(
                    comp_i,
                    comp_j,
                    T,
                )
                if "tau12_a" in data:
                    tref = data.get("tau_tref", self.T_REF)
                    exponent = (
                        data["tau12_a"]
                        + data.get("tau12_b", 0.0) / interaction_T
                        + data.get("tau12_c", 0.0)
                        * _anchored_log_temperature_term(interaction_T, tref)
                        + data.get("tau12_d", 0.0) * interaction_T
                        + data.get("tau12_e", 0.0) * interaction_T * interaction_T
                    )
                else:
                    exponent = (
                        -data["a12_cal_per_mol"]
                        / (self.R_CAL * interaction_T)
                    )
                tau[i][j] = math.exp(max(min(exponent, 50.0), -50.0))
        if len(self._uniquac_tau_cache) > 256:
            self._uniquac_tau_cache.clear()
        self._uniquac_tau_cache[T_key] = tau
        return tau

    def _uniquac_parameter_matrices(self) -> dict[str, list[list[float]]]:
        if self._uniquac_parameter_cache is not None:
            return self._uniquac_parameter_cache

        n = len(self.components)
        tau_mode = [[0 for _ in range(n)] for _ in range(n)]
        tau_a = [[0.0 for _ in range(n)] for _ in range(n)]
        tau_b = [[0.0 for _ in range(n)] for _ in range(n)]
        tau_c = [[0.0 for _ in range(n)] for _ in range(n)]
        tau_d = [[0.0 for _ in range(n)] for _ in range(n)]
        tau_e = [[0.0 for _ in range(n)] for _ in range(n)]
        tau_tref = [[self.T_REF for _ in range(n)] for _ in range(n)]
        use_q_prime = [[False for _ in range(n)] for _ in range(n)]

        for i, comp_i in enumerate(self.components):
            for j, comp_j in enumerate(self.components):
                if i == j:
                    continue
                data = self._uniquac_interaction_for_components(comp_i, comp_j)
                if data is None:
                    continue
                if "tau12_a" in data:
                    tau_mode[i][j] = 1
                    tau_a[i][j] = data["tau12_a"]
                    tau_b[i][j] = data.get("tau12_b", 0.0)
                    tau_c[i][j] = data.get("tau12_c", 0.0)
                    tau_d[i][j] = data.get("tau12_d", 0.0)
                    tau_e[i][j] = data.get("tau12_e", 0.0)
                    tau_tref[i][j] = data.get("tau_tref", self.T_REF)
                else:
                    tau_mode[i][j] = 2
                    tau_a[i][j] = data["a12_cal_per_mol"] / self.R_CAL
                use_q_prime[i][j] = bool(data.get("use_q_prime", False))

        q_residual = []
        for i, comp in enumerate(self.components):
            if comp in self._deferred_uniquac_rq_errors:
                q_residual.append(1.0)
                continue
            if any(use_q_prime[i][j] or use_q_prime[j][i] for j in range(n)):
                q_residual.append(self.q_prime.get(comp, self.q[comp]))
            else:
                q_residual.append(self.q[comp])

        self._uniquac_parameter_cache = {
            "tau_mode": tau_mode,
            "tau_a": tau_a,
            "tau_b": tau_b,
            "tau_c": tau_c,
            "tau_d": tau_d,
            "tau_e": tau_e,
            "tau_tref": tau_tref,
            "use_q_prime": use_q_prime,
            "q_residual": q_residual,
        }
        return self._uniquac_parameter_cache

    def activity_coefficients(self, T: float, composition: dict[str, float]) -> dict[str, float]:
        cache_key = (
            float(T),
            tuple((comp, float(composition.get(comp, 0.0))) for comp in self.components),
        )
        cached = self._activity_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        active_components = [
            comp for comp in self.components
            if comp not in self._deferred_uniquac_rq_errors
        ]
        if self._deferred_uniquac_rq_errors:
            gamma = self._activity_coefficients_for_components(
                T,
                composition,
                active_components,
            )
            gamma.update({comp: 1.0 for comp in self._deferred_uniquac_rq_errors})
            if len(self._activity_cache) > 20000:
                self._activity_cache.clear()
            self._activity_cache[cache_key] = dict(gamma)
            return gamma

        x = [max(float(composition.get(comp, 0.0)), 0.0) for comp in self.components]
        total = sum(x)
        if total <= 0.0:
            return {comp: 1.0 for comp in self.components}
        x = [value / total for value in x]

        compiled = self._compiled_activity_backend(T)
        if compiled is not None:
            gamma_list = compiled.activity_coefficients(x, T)
            gamma = {
                comp: float(gamma_list[i])
                for i, comp in enumerate(self.components)
            }
            if len(self._activity_cache) > 20000:
                self._activity_cache.clear()
            self._activity_cache[cache_key] = dict(gamma)
            return gamma

        n = len(self.components)
        r = [self.r[comp] for comp in self.components]
        q = [self.q[comp] for comp in self.components]
        rx = sum(r[i] * x[i] for i in range(n)) or 1e-30
        qx = sum(q[i] * x[i] for i in range(n)) or 1e-30
        l = [
            0.5 * self.Z * (r[i] - q[i]) - (r[i] - 1.0)
            for i in range(n)
        ]
        xl_sum = sum(x[i] * l[i] for i in range(n))

        tau = self._uniquac_tau_matrix(T)
        q_residual = self._uniquac_parameter_matrices()["q_residual"]
        # Extended (Anderson-Prausnitz) form: the residual term uses area
        # fractions built from q' throughout, not just as a prefactor.
        # Reduces to ordinary UNIQUAC when q' == q.
        qx_residual = sum(q_residual[i] * x[i] for i in range(n)) or 1e-30
        theta_residual = [q_residual[i] * x[i] / qx_residual for i in range(n)]
        theta_tau_col = [
            sum(theta_residual[j] * tau[j][i] for j in range(n)) or 1e-30
            for i in range(n)
        ]

        gamma = {}
        for i, comp_i in enumerate(self.components):
            phi_over_x = max(r[i] / rx, 1e-30)
            theta_over_phi = max(q[i] * rx / max(r[i] * qx, 1e-30), 1e-30)
            ln_gamma_c = (
                math.log(phi_over_x)
                + 0.5 * self.Z * q[i] * math.log(theta_over_phi)
                + l[i]
                - phi_over_x * xl_sum
            )
            residual_sum = sum(
                theta_residual[j] * tau[i][j] / theta_tau_col[j]
                for j in range(n)
            )
            ln_gamma_r = q_residual[i] * (
                1.0
                - math.log(theta_tau_col[i])
                - residual_sum
            )
            ln_gamma = ln_gamma_c + ln_gamma_r
            gamma[comp_i] = max(math.exp(max(min(ln_gamma, 50.0), -50.0)), 1e-12)

        if len(self._activity_cache) > 20000:
            self._activity_cache.clear()
        self._activity_cache[cache_key] = dict(gamma)
        return gamma

    def excess_enthalpy(self, composition: dict[str, float], T: float) -> float:
        backend = self._compiled_activity_backend(T)
        if backend is None:
            return super().excess_enthalpy(composition, T)
        total = sum(
            max(float(composition.get(comp, 0.0)), 0.0)
            for comp in self.components
        )
        if total <= 0.0:
            return 0.0
        x_active = [
            max(float(composition.get(comp, 0.0)), 0.0) / total
            for comp in backend.components
        ]
        active_fraction = sum(x_active)
        if active_fraction <= 0.0:
            return 0.0
        x_active = [value / active_fraction for value in x_active]
        return active_fraction * backend.excess_enthalpy(x_active, T)

    def _activity_coefficients_for_components(
        self,
        T: float,
        composition: dict[str, float],
        components: list[str],
    ) -> dict[str, float]:
        """Reference UNIQUAC calculation for a resolved liquid submixture."""
        if not components:
            raise ThermodynamicsError(
                "UNIQUAC liquid activity calculation has no non-Henry bulk components"
            )
        x = [max(float(composition.get(comp, 0.0)), 0.0) for comp in components]
        total = sum(x)
        if total <= 0.0:
            return {comp: 1.0 for comp in components}
        x = [value / total for value in x]
        compiled = self._compiled_activity_backend(T)
        if compiled is not None and tuple(components) == tuple(compiled.components):
            values = compiled.activity_coefficients(x, T)
            return {
                component: float(values[index])
                for index, component in enumerate(components)
            }
        n = len(components)
        r = [self.r[comp] for comp in components]
        q = [self.q[comp] for comp in components]
        rx = sum(r[i] * x[i] for i in range(n)) or 1e-30
        qx = sum(q[i] * x[i] for i in range(n)) or 1e-30
        l = [
            0.5 * self.Z * (r[i] - q[i]) - (r[i] - 1.0)
            for i in range(n)
        ]
        xl_sum = sum(x[i] * l[i] for i in range(n))

        tau = [[1.0 for _ in range(n)] for _ in range(n)]
        use_q_prime = [[False for _ in range(n)] for _ in range(n)]
        for i, comp_i in enumerate(components):
            for j, comp_j in enumerate(components):
                if i == j:
                    continue
                data = self._uniquac_interaction_for_components(comp_i, comp_j)
                if data is None:
                    continue
                interaction_T = self.activity_interaction_temperature(
                    comp_i,
                    comp_j,
                    T,
                )
                if "tau12_a" in data:
                    tref = data.get("tau_tref", self.T_REF)
                    exponent = (
                        data["tau12_a"]
                        + data.get("tau12_b", 0.0) / interaction_T
                        + data.get("tau12_c", 0.0)
                        * _anchored_log_temperature_term(interaction_T, tref)
                        + data.get("tau12_d", 0.0) * interaction_T
                        + data.get("tau12_e", 0.0) * interaction_T * interaction_T
                    )
                else:
                    exponent = (
                        -data["a12_cal_per_mol"]
                        / (self.R_CAL * interaction_T)
                    )
                tau[i][j] = math.exp(max(min(exponent, 50.0), -50.0))
                use_q_prime[i][j] = bool(data.get("use_q_prime", False))

        q_residual = [
            (
                self.q_prime.get(comp, self.q[comp])
                if any(use_q_prime[i][j] or use_q_prime[j][i] for j in range(n))
                else self.q[comp]
            )
            for i, comp in enumerate(components)
        ]
        qx_residual = sum(q_residual[i] * x[i] for i in range(n)) or 1e-30
        theta_residual = [q_residual[i] * x[i] / qx_residual for i in range(n)]
        theta_tau_col = [
            sum(theta_residual[j] * tau[j][i] for j in range(n)) or 1e-30
            for i in range(n)
        ]

        gamma = {}
        for i, comp_i in enumerate(components):
            phi_over_x = max(r[i] / rx, 1e-30)
            theta_over_phi = max(q[i] * rx / max(r[i] * qx, 1e-30), 1e-30)
            ln_gamma_c = (
                math.log(phi_over_x)
                + 0.5 * self.Z * q[i] * math.log(theta_over_phi)
                + l[i]
                - phi_over_x * xl_sum
            )
            residual_sum = sum(
                theta_residual[j] * tau[i][j] / theta_tau_col[j]
                for j in range(n)
            )
            ln_gamma_r = q_residual[i] * (
                1.0 - math.log(theta_tau_col[i]) - residual_sum
            )
            gamma[comp_i] = max(
                math.exp(max(min(ln_gamma_c + ln_gamma_r, 50.0), -50.0)),
                1e-12,
            )
        return gamma


class UNIQUACVDMThermodynamics(VaporDimerizationActivityMixin, UNIQUACThermodynamics):
    """UNIQUAC liquid activity model with vapor dimerization correction."""

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
        super().__init__(
            components,
            db,
            interaction_overrides,
            interaction_estimation,
            estimation_unifac_groups,
            activity_interaction_max_psat_bar,
            activity_interaction_max_temperature_K,
        )
        self._initialize_vdm()


class NRTLVDMThermodynamics(VaporDimerizationActivityMixin, NRTLThermodynamics):
    """NRTL liquid activity model with vapor dimerization correction."""

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
        super().__init__(
            components,
            db,
            interaction_overrides,
            interaction_estimation,
            estimation_unifac_groups,
            activity_interaction_max_psat_bar,
            activity_interaction_max_temperature_K,
        )
        self._initialize_vdm()
