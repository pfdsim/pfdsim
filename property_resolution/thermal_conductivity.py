"""Pure-component liquid and vapor thermal-conductivity resolution."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .common import PropertyResolutionError, PropertyResolutionResult


PERRY_CONDUCTIVITY_EQUATIONS = {
    100: "dippr_eq100",
    102: "dippr_eq102",
}


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
        """Resolve an in-range liquid or dilute-vapor conductivity."""
        T = self._normalize_thermal_conductivity_temperature(T)
        phase = self._normalize_thermal_conductivity_phase(phase)
        props = self._coerce_props(identifier, props, allow_online=allow_online)
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

        raise PropertyResolutionError(
            f"Cannot determine {phase} thermal conductivity for {identifier!r} "
            f"at T={T:g} K."
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


__all__ = ["PERRY_CONDUCTIVITY_EQUATIONS", "ThermalConductivityMixin"]
