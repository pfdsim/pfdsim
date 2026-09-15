"""Pure-component liquid and vapor thermal-conductivity resolution."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from .common import PropertyResolutionError, PropertyResolutionResult

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from ..physical_constants import R_J_MOL_K
else:
    from physical_constants import R_J_MOL_K


PERRY_CONDUCTIVITY_EQUATIONS = {
    100: "dippr_eq100",
    102: "dippr_eq102",
}

# Compound-held-out corrected class MAPEs from the selected model artifact.
# The inorganic value is the uncorrected geometry relation's class MAPE.
GROUP_CORRECTED_MAPE_PERCENT = {
    "acid_straight_long_C7_plus": 5.497542921864029,
    "acid_straight_medium_C4_C6": 7.6100262486692,
    "acid_straight_short_C1_C3": 23.984574311246515,
    "acid_unsaturated_or_aromatic_monocarboxylic": 0.7577505940908267,
    "alcohol": 8.06847230156654,
    "aldehyde": 9.39163262841765,
    "aliphatic_ring": 9.463222888757764,
    "alkene": 6.329494755246616,
    "alkyne": 4.640142995926627,
    "aromatic_ring": 4.867386875368716,
    "chlorine": 8.379103928802005,
    "hydrocarbon_only": 4.825355745550674,
    "ketone": 10.391523181072,
    "thiol": 1.717594116088247,
}
UNGROUPED_ORGANIC_MAPE_PERCENT = 7.45787113549919
INORGANIC_MAPE_PERCENT = 8.88
ESTIMATED_VAPOR_CONDUCTIVITY_QUALITY_CAP = 0.88
GOVENDER_LIQUID_BASE_QUALITY = 0.80
GOVENDER_LIQUID_SMALL_MOLECULE_PENALTY = 0.08
GOVENDER_LIQUID_SINGLE_COMPONENT_GROUP_PENALTY = 0.08
GOVENDER_LIQUID_OTHER_CAUTION_GROUP_PENALTY = 0.04


class ThermalConductivityMixin:
    """Resolve pure-component conductivity in W/(m*K)."""

    def resolve_thermal_conductivity(
        self,
        identifier: str,
        T: float,
        phase: str,
        props: Optional[dict[str, Any]] = None,
        *,
        allow_online: bool = True,
    ) -> PropertyResolutionResult:
        """Resolve liquid or dilute-vapor conductivity, preferring in-range data."""
        T = self._normalize_thermal_conductivity_temperature(T)
        phase = self._normalize_thermal_conductivity_phase(phase)
        props = self._coerce_props(identifier, props, allow_online=allow_online)
        allow_online = self._props_allow_online(props, allow_online)
        correlation_key = "kg" if phase == "vapor" else "kl"

        provided = self._evaluate_provided_correlation(
            props,
            correlation_key,
            T,
        )
        if provided is not None:
            value, correlation = provided
            if self._positive_number(value) is not None:
                equation = str(correlation.get("equation") or "correlation")
                return self._provided_correlation_result(
                    value,
                    correlation,
                    f"provided_{phase}_thermal_conductivity_{equation}",
                    "thermal conductivity in W/(m*K)",
                    default_quality=0.97,
                )

        library = self._get_perry_library()
        if library is not None:
            candidates = self._identifier_candidates(identifier, props)
            for candidate in candidates:
                row = library.thermal_conductivity_correlation(candidate, T, phase)
                if row is None:
                    continue
                correlation = self._normalized_perry_conductivity_correlation(row)
                evaluated = self._evaluate_correlation(
                    correlation,
                    T,
                    props=props,
                )
                if evaluated is None:
                    continue
                value, _ = evaluated
                if self._positive_number(value) is None:
                    continue
                equation_id = int(row["equation_id"])
                return PropertyResolutionResult(
                    value=value,
                    source="local",
                    method=f"perry_{phase}_thermal_conductivity_eq{equation_id}",
                    quality=0.97,
                    notes=(
                        f"{library._source(row['source_table'])}; valid from "
                        f"{row['T_min_K']:g} to {row['T_max_K']:g} K; "
                        "units W/(m*K)"
                    ),
                )

            if phase == "liquid":
                for candidate in candidates:
                    tabulated = library.saturated_liquid_thermal_conductivity_W_per_m_K(
                        candidate,
                        T,
                    )
                    if tabulated is None:
                        continue
                    interpolated = tabulated.method.endswith("linear_interpolation")
                    correlation = tabulated.correlation
                    if interpolated:
                        detail = (
                            "linearly interpolated between "
                            f"{correlation['T_min_K']:g} and "
                            f"{correlation['T_max_K']:g} K"
                        )
                    else:
                        detail = f"tabulated at {correlation['temperature_K']:g} K"
                    return PropertyResolutionResult(
                        value=tabulated.value,
                        source="local",
                        method=tabulated.method,
                        quality=0.93 if interpolated else 0.95,
                        notes=f"{tabulated.source}; {detail}; units {tabulated.units}",
                    )

        if phase == "liquid":
            estimated = self._estimated_liquid_thermal_conductivity(
                identifier, T, props, allow_online=allow_online
            )
            if estimated is not None:
                return estimated
        else:
            estimated = self._estimated_vapor_thermal_conductivity(
                identifier, T, props, allow_online=allow_online
            )
            if estimated is not None:
                return estimated

        raise PropertyResolutionError(
            f"Cannot determine {phase} thermal conductivity for {identifier!r} "
            f"at T={T:g} K."
        )

    def _estimated_liquid_thermal_conductivity(
        self,
        identifier: str,
        T: float,
        props: dict[str, Any],
        *,
        allow_online: bool,
    ) -> Optional[PropertyResolutionResult]:
        """Use Govender after all supplied and Perry liquid data are exhausted."""
        if __package__ and __package__.split(".", 1)[0] == "pfdsim":
            from ..govender_method import (
                GovenderError,
                PAPER_CAUTION_GROUPS,
                PAPER_SINGLE_COMPONENT_GROUPS,
                estimate as estimate_govender,
            )
        else:
            from govender_method import (
                GovenderError,
                PAPER_CAUTION_GROUPS,
                PAPER_SINGLE_COMPONENT_GROUPS,
                estimate as estimate_govender,
            )
        from rdkit import Chem

        smiles_result = self._resolve_smiles_result(
            identifier, props, allow_online=allow_online
        )
        smiles = str(smiles_result.value).strip() if smiles_result else ""
        if not smiles:
            return None
        try:
            molecule = Chem.MolFromSmiles(smiles)
        except Exception:
            return None
        if molecule is None:
            return None

        tb_props = dict(props)
        if not (tb_props.get("smiles") or tb_props.get("SMILES")):
            tb_props["smiles"] = smiles
            property_sources = dict(tb_props.get("property_sources") or {})
            property_sources["smiles"] = {
                "source": smiles_result.source,
                "method": smiles_result.method,
                "quality": self._result_quality(smiles_result),
                "notes": smiles_result.notes,
            }
            tb_props["property_sources"] = property_sources
        try:
            tb_result = self.resolve_boiling_point(
                identifier,
                tb_props,
                allow_online=allow_online,
                allow_estimation=True,
            )
        except (PropertyResolutionError, ValueError, TypeError, OverflowError):
            return None
        if tb_result is None or self._positive_number(tb_result.value) is None:
            return None

        try:
            evaluation = estimate_govender(
                smiles,
                tb=float(tb_result.value),
                local_refits=True,
            )
            value = evaluation.conductivity_W_m_K(T)
        except (GovenderError, ValueError, TypeError, OverflowError):
            return None

        carbon_atoms = sum(
            atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()
        )
        single_component_groups = sorted(
            set(evaluation.groups) & PAPER_SINGLE_COMPONENT_GROUPS
        )
        other_caution_groups = sorted(
            set(evaluation.groups)
            & (PAPER_CAUTION_GROUPS - PAPER_SINGLE_COMPONENT_GROUPS)
        )
        model_quality = GOVENDER_LIQUID_BASE_QUALITY
        penalties = []
        if carbon_atoms < 3:
            model_quality -= GOVENDER_LIQUID_SMALL_MOLECULE_PENALTY
            penalties.append(
                f"-{GOVENDER_LIQUID_SMALL_MOLECULE_PENALTY:.2f} for "
                f"fewer than 3 carbon atoms ({carbon_atoms})"
            )
        if single_component_groups:
            model_quality -= GOVENDER_LIQUID_SINGLE_COMPONENT_GROUP_PENALTY
            penalties.append(
                f"-{GOVENDER_LIQUID_SINGLE_COMPONENT_GROUP_PENALTY:.2f} for "
                "paper single-component group(s) "
                f"{single_component_groups}"
            )
        if other_caution_groups:
            model_quality -= GOVENDER_LIQUID_OTHER_CAUTION_GROUP_PENALTY
            penalties.append(
                f"-{GOVENDER_LIQUID_OTHER_CAUTION_GROUP_PENALTY:.2f} for "
                f"other paper caution group(s) {other_caution_groups}"
            )
        model_quality = round(max(0.0, model_quality), 2)
        tb_quality = self._result_quality(tb_result)
        tb_quality_factor = 1.0 - (1.0 - tb_quality) ** 2
        scaled_model_quality = model_quality * tb_quality_factor
        quality = min(
            round(scaled_model_quality, 2),
            math.floor(
                100.0 * self._result_quality(smiles_result) + 1.0e-9
            ) / 100.0,
        )
        penalty_note = "; ".join(penalties) if penalties else "none"
        warnings = "; ".join(evaluation.warnings) or "none"
        return PropertyResolutionResult(
            value=value,
            source="calculated",
            method="govender_saturated_liquid_thermal_conductivity",
            quality=quality,
            notes=(
                "saturated-liquid prediction in W/(m*K); Govender local "
                f"refits enabled; groups {evaluation.groups}; carbon atoms "
                f"{carbon_atoms}; baseline quality "
                f"{GOVENDER_LIQUID_BASE_QUALITY:.2f}; penalties {penalty_note}; "
                f"structural model quality {model_quality:.2f}; Tb quality "
                f"factor 1-(1-{tb_quality:.3g})^2={tb_quality_factor:.4f}; "
                f"scaled model quality {scaled_model_quality:.3f}, final "
                f"quality {quality:.2f}; "
                f"Tb={float(tb_result.value):g} K from {tb_result.method} "
                f"({tb_result.source}, quality {self._result_quality(tb_result):.3g}); "
                f"SMILES from {smiles_result.method} "
                f"({smiles_result.source}, quality "
                f"{self._result_quality(smiles_result):.3g}); warnings {warnings}"
            ),
        )

    def _estimated_vapor_thermal_conductivity(
        self,
        identifier: str,
        T: float,
        props: dict[str, Any],
        *,
        allow_online: bool,
    ) -> Optional[PropertyResolutionResult]:
        """Use the geometry/group model only when all dilute-gas inputs resolve."""
        if __package__ and __package__.split(".", 1)[0] == "pfdsim":
            from ..pfdsim_modified_stiel_thodos import (
                PFDSimModifiedStielThodosError,
                classify_molecular_geometry,
                evaluate_pfdsim_modified_stiel_thodos,
            )
        else:
            from pfdsim_modified_stiel_thodos import (
                PFDSimModifiedStielThodosError,
                classify_molecular_geometry,
                evaluate_pfdsim_modified_stiel_thodos,
            )

        formula_result = self._source_result_for_value(props, "formula")
        perry_identity: dict[str, Any] = {}
        if formula_result is None:
            perry_identity = self._get_perry_identity(identifier)
            formula = perry_identity.get("formula")
            if not formula:
                return None
            formula_result = PropertyResolutionResult(
                value=formula,
                source="local",
                method="perry_molecular_identity",
                quality=0.97,
            )
        formula = str(formula_result.value).strip()
        if not formula:
            return None

        try:
            geometry, geometry_source = classify_molecular_geometry(formula)
        except PFDSimModifiedStielThodosError:
            smiles_result = self._resolve_smiles_result(
                identifier, props, allow_online=allow_online
            )
            smiles = str(smiles_result.value).strip() if smiles_result else ""
            try:
                geometry, geometry_source = classify_molecular_geometry(formula, smiles)
            except PFDSimModifiedStielThodosError:
                return None
        else:
            smiles_result = None
            smiles = ""

        cas = str(
            props.get("CAS") or props.get("cas") or perry_identity.get("CAS") or ""
        ).strip()
        if not cas and self._looks_like_cas(identifier):
            cas = str(identifier).strip()

        try:
            tc_result = self._source_result_for_value(props, "Tc", units="K")
            if tc_result is None or self._positive_number(tc_result.value) is None:
                critical = self.resolve_critical_properties(
                    identifier,
                    props,
                    allow_online=allow_online,
                    allow_estimation=True,
                )
                tc_result = critical.get("Tc") if critical else None
            if tc_result is None or self._positive_number(tc_result.value) is None:
                return None
            mw_result = self.resolve_molecular_weight(
                identifier, props, allow_online=allow_online
            )
            if self._positive_number(mw_result.value) is None:
                return None
            cp_result = self.resolve_heat_capacity(
                identifier, T, "ideal_gas", props, allow_online=allow_online
            )
            if self._positive_number(cp_result.value) is None:
                return None
            viscosity_result = self.resolve_viscosity(
                identifier, T, "vapor", props, allow_online=allow_online
            )
            if self._positive_number(viscosity_result.value) is None:
                return None
            cv = float(cp_result.value) - R_J_MOL_K
            evaluation = evaluate_pfdsim_modified_stiel_thodos(
                temperature_K=T,
                viscosity_Pa_s=float(viscosity_result.value),
                cv_J_per_mol_K=cv,
                molecular_weight_g_per_mol=float(mw_result.value),
                critical_temperature_K=float(tc_result.value),
                geometry=geometry,
                smiles=smiles,
                formula=formula,
                cas=cas,
            )
        except (
            PropertyResolutionError,
            PFDSimModifiedStielThodosError,
            ValueError,
            TypeError,
            OverflowError,
        ):
            return None

        if not evaluation.organic_correction_eligible:
            group, mape_percent = "inorganic", INORGANIC_MAPE_PERCENT
        elif evaluation.active_groups:
            group = max(
                evaluation.active_groups,
                key=GROUP_CORRECTED_MAPE_PERCENT.__getitem__,
            )
            mape_percent = GROUP_CORRECTED_MAPE_PERCENT[group]
        else:
            group, mape_percent = "ungrouped organic", UNGROUPED_ORGANIC_MAPE_PERCENT

        input_results = {
            "formula": formula_result,
            "SMILES": smiles_result,
            "viscosity": viscosity_result,
            "ideal-gas Cp": cp_result,
            "molecular weight": mw_result,
            "critical temperature": tc_result,
        }
        model_quality = min(
            ESTIMATED_VAPOR_CONDUCTIVITY_QUALITY_CAP,
            max(0.0, 1.0 - 3.0 * mape_percent / 100.0),
        )
        rounded_model_quality = round(model_quality, 2)
        weakest_input_quality = min(
            self._result_quality(result)
            for result in input_results.values()
            if result is not None
        )
        quality = min(
            rounded_model_quality,
            math.floor(100.0 * weakest_input_quality + 1.0e-9) / 100.0,
        )
        provenance = "; ".join(
            f"{name}: {result.value} from {result.method} "
            f"({result.source}, quality {self._result_quality(result):.3g})"
            for name, result in input_results.items()
            if result is not None
        )
        return PropertyResolutionResult(
            value=evaluation.value_W_per_m_K,
            source="calculated",
            method="pfdsim_modified_stiel_thodos_dilute_vapor",
            quality=quality,
            notes=(
                f"dilute-vapor prediction in W/(m*K); geometry {geometry} "
                f"({geometry_source}); active groups {evaluation.active_groups or 'none'}; "
                f"quality group {group} with held-out relative MAE {mape_percent:.2f}%; "
                f"model quality {rounded_model_quality:.2f}, final quality {quality:.2f}; "
                f"base {evaluation.base_value_W_per_m_K:.6g} W/(m*K), "
                f"correction factor {evaluation.correction_factor:.6g}; "
                f"Cv=Cp-R={cv:.6g} J/(mol*K); {provenance}"
            ),
        )

    def _normalize_thermal_conductivity_temperature(self, T: float) -> float:
        value = self._positive_number(T)
        if value is None:
            raise PropertyResolutionError(
                "Thermal-conductivity temperature must be a positive finite value in K."
            )
        return value

    @staticmethod
    def _normalize_thermal_conductivity_phase(phase: str) -> str:
        phase_key = str(phase).strip().lower().replace("-", "_")
        if phase_key in {"gas", "vapor", "vapour", "ideal_gas", "ideal"}:
            return "vapor"
        if phase_key in {"liquid", "l"}:
            return "liquid"
        raise PropertyResolutionError(
            f"Unsupported thermal-conductivity phase {phase!r}; "
            "expected liquid or vapor."
        )

    @staticmethod
    def _normalized_perry_conductivity_correlation(
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        equation_id = int(row["equation_id"])
        equation = PERRY_CONDUCTIVITY_EQUATIONS.get(equation_id)
        if equation is None:
            raise PropertyResolutionError(
                f"Unsupported Perry thermal-conductivity equation {equation_id}."
            )
        coefficients = {
            name: float(value)
            for name, value in zip("ABCDE", row["coefficients"], strict=False)
        }
        return {
            "equation": equation,
            "coefficients": coefficients,
            "Tmin_K": float(row["T_min_K"]),
            "Tmax_K": float(row["T_max_K"]),
        }


__all__ = [
    "GOVENDER_LIQUID_BASE_QUALITY",
    "GOVENDER_LIQUID_OTHER_CAUTION_GROUP_PENALTY",
    "GOVENDER_LIQUID_SINGLE_COMPONENT_GROUP_PENALTY",
    "GOVENDER_LIQUID_SMALL_MOLECULE_PENALTY",
    "PERRY_CONDUCTIVITY_EQUATIONS",
    "ThermalConductivityMixin",
]
