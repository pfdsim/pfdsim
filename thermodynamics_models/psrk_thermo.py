"""Simulator-facing integration layer for the published PSRK backend."""

from __future__ import annotations

from typing import Iterable, Optional, Union

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..chemical_properties import ChemicalDatabase
else:
    from chemical_properties import ChemicalDatabase
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..interaction_parameters import cas_for_component
else:
    from interaction_parameters import cas_for_component

from .base import StreamState
from .common import (
    ThermodynamicsError,
)
from .eos import PhiPhiThermodynamics, _phi_henry_aqueous_K_values
from .henry import AqueousEquilibriumContext
from .psrk import PSRK, PSRKError, R_BAR_CM3_PER_MOL_K


class PSRKThermodynamics(PhiPhiThermodynamics):
    """First-class phi-phi thermodynamics wrapper around :class:`PSRK`."""

    model = "PSRK"

    def __init__(
        self,
        components: list[str],
        db: Optional[ChemicalDatabase] = None,
        interaction_overrides: Optional[list[dict]] = None,
    ) -> None:
        super().__init__(components, db, interaction_overrides)
        self.component_cas: dict[str, str] = {}
        self.cas_component: dict[str, str] = {}
        fallbacks: dict[str, dict] = {}

        for component in components:
            props = self.props[component]
            cas = cas_for_component(component, props)
            if not cas:
                raise ThermodynamicsError(
                    f"PSRK requires a CAS identity for '{component}'"
                )
            if cas in self.cas_component:
                raise ThermodynamicsError(
                    f"PSRK components '{self.cas_component[cas]}' and "
                    f"'{component}' resolve to the same CAS {cas}"
                )
            self.component_cas[component] = cas
            self.cas_component[cas] = component

            source_metadata = props.property_sources or {}
            explicit_fields = {
                field
                for field in ("Tc", "Pc", "omega", "mc_c1", "mc_c2", "mc_c3")
                if self._is_explicit_pfd_source(source_metadata.get(field))
            }
            fallbacks[cas] = {
                "Tc_K": props.Tc,
                "Pc_bar": props.Pc,
                "omega": props.omega,
                "mc_c1": props.mc_c1 if "mc_c1" in explicit_fields else None,
                "mc_c2": props.mc_c2 if "mc_c2" in explicit_fields else None,
                "mc_c3": props.mc_c3 if "mc_c3" in explicit_fields else None,
                "override_bundle": bool(explicit_fields),
                "source": (
                    "PFD component override"
                    if explicit_fields
                    else "normal property resolution"
                ),
            }

        try:
            self.psrk = PSRK(
                tuple(self.component_cas[component] for component in components),
                component_fallbacks=fallbacks,
            )
        except PSRKError as error:
            raise ThermodynamicsError(str(error)) from error

        self.extend_warnings(self.psrk.warnings)
        for component in components:
            cas = self.component_cas[component]
            if not self.psrk.component_parameter_sources[cas].startswith("PSRK 2005"):
                self.mark_property_source_context(
                    component,
                    ("Tc", "Pc", "omega"),
                    kind="thermo_model",
                    phase="psrk_fallback_parameters",
                    description="PSRK resolved critical-property fallback",
                    affects_result=True,
                )

    @staticmethod
    def _is_explicit_pfd_source(metadata) -> bool:
        if not isinstance(metadata, dict):
            return False
        source = str(metadata.get("source") or "").strip().lower()
        method = str(metadata.get("method") or "").strip().lower()
        return source in {"provided", "user"} or method == "pfd_component_override"

    def _cas_composition(self, composition: dict[str, float]) -> dict[str, float]:
        return {
            self.component_cas[component]: float(composition.get(component, 0.0))
            for component in self.components
        }

    def _component_values(self, values: dict[str, float]) -> dict[str, float]:
        return {
            self.cas_component[cas]: value
            for cas, value in values.items()
        }

    def compressibility_factor(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> float:
        return self.psrk.compressibility_factor(
            T, P, self._cas_composition(composition), phase
        )

    def _eos_reduced_parameters(self, T, composition):
        temperature, _ = self.psrk._validate_temperature_pressure(T, 1.)
        x = self.psrk._normalize_composition(self._cas_composition(composition))
        mixture = self.psrk._mixture_parameters(temperature, x)
        return mixture.D, R_BAR_CM3_PER_MOL_K * temperature / mixture.b, 1., 0.

    def fugacity_coefficients(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> dict[str, float]:
        values = self.psrk.fugacity_coefficients(
            T, P, self._cas_composition(composition), phase
        )
        return self._component_values(values)

    def molar_volume(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> float:
        """Return un-translated PSRK volume [m3/kmol]."""
        return self.psrk.molar_volume(
            T, P, self._cas_composition(composition), phase
        ) / 1000.0

    def molar_volume_vapor(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
    ) -> float:
        return self.molar_volume(T, P, composition, "vapor")

    def vapor_molar_volume_for_density(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
    ) -> float:
        return self.molar_volume_vapor(T, P, composition)

    def molar_volume_liquid(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
    ) -> float:
        """Return resolver-backed liquid volume; PSRK volume translation is absent."""
        return self.mixture_liquid_molar_volume(composition, T)

    def departure_enthalpy(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> float:
        return self.psrk.departure_enthalpy(
            T, P, self._cas_composition(composition), phase
        )

    def departure_entropy(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> float:
        return self.psrk.departure_entropy(
            T, P, self._cas_composition(composition), phase
        )

    def departure_gibbs(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> float:
        return self.psrk.departure_gibbs(
            T, P, self._cas_composition(composition), phase
        )

    def departure_heat_capacity(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> float:
        return self.psrk.departure_heat_capacity(
            T, P, self._cas_composition(composition), phase
        )

    def K_values(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
    ) -> dict[str, float]:
        values = self.psrk.phi_phi_K_values(
            T, P, self._cas_composition(composition)
        )
        return self._component_values(values)

    def aqueous_K_values(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        context: AqueousEquilibriumContext,
    ) -> dict[str, float]:
        return _phi_henry_aqueous_K_values(
            self, T, P, composition, context, "aqueous_psrk_phi_henry"
        )

    def calculate_state(
        self,
        T: float,
        P: float,
        F: float,
        composition: dict[str, float],
        phase: Optional[str] = None,
        flash: bool = True,
        include: Optional[Union[str, Iterable[str]]] = None,
    ) -> StreamState:
        return super().calculate_state(
            T, P, F, composition, phase, flash, include=include
        )
