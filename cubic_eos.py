"""General cubic EOS utilities for phi-phi property methods."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .physical_constants import R_BAR_CM3_MOL_K
    from .lyngby_parameters import canonical_lyngby_method
else:
    from physical_constants import R_BAR_CM3_MOL_K
    from lyngby_parameters import canonical_lyngby_method

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .chemical_properties import ChemicalDatabase, ChemicalProperties, get_database
else:
    from chemical_properties import ChemicalDatabase, ChemicalProperties, get_database
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .interaction_parameters import (
        EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K,
        cas_for_component,
        canonical_eos_model_key,
        eos_record_kij,
        eos_binary_interaction_records,
    )
else:
    from interaction_parameters import (
        EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K,
        cas_for_component,
        canonical_eos_model_key,
        eos_record_kij,
        eos_binary_interaction_records,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .psrk_parameters import compatible_psrk_mathias_copeman
else:
    from psrk_parameters import compatible_psrk_mathias_copeman

R_CM3 = R_BAR_CM3_MOL_K  # bar-cm3/mol-K


class CubicEOSError(Exception):
    """Error in cubic EOS calculations."""


@dataclass(frozen=True)
class CubicPureParams:
    comp: str
    Tc: float
    Pc: float
    omega: float
    a0: float
    b: float
    mc_c1: Optional[float] = None
    mc_c2: Optional[float] = None
    mc_c3: Optional[float] = None
    mc_Tmin: Optional[float] = None
    mc_Tmax: Optional[float] = None
    mc_source: Optional[str] = None
    kappa1: Optional[float] = None
    kappa2: Optional[float] = None
    kappa3: Optional[float] = None
    twu_l: Optional[float] = None
    twu_m: Optional[float] = None
    twu_n: Optional[float] = None


class CubicEOS:
    """
    SRK/RKS and Peng-Robinson cubic EOS with optional alpha variants.

    Uses van der Waals one-fluid mixing with ChemSep binary interaction
    parameters where available.
    """

    def __init__(
        self,
        components: list[str],
        model: str,
        db: Optional[ChemicalDatabase] = None,
        interaction_overrides: Optional[list[dict]] = None,
        unifac_groups: Optional[dict] = None,
    ):
        self.db = db or get_database()
        self.components = components
        self.model = self._canonical_model(model)
        self._ge_provider = None
        self._ge_cache = {}
        self.family = "PR" if self.model.startswith("PR") else "SRK"
        self.use_boston_mathias = self.model.endswith("-BM")
        self.use_mathias_copeman = self.model.endswith("-MC")
        self.use_prsv1 = self.model == "PRSV1"
        self.use_prsv2 = self.model == "PRSV2"
        self.use_twu = self.model.endswith("-TWU")
        self.props: dict[str, ChemicalProperties] = {}
        self.params: dict[str, CubicPureParams] = {}
        self.component_cas: dict[str, Optional[str]] = {}
        self._static_kij: dict[tuple[str, str], float] = {}
        self._temperature_kij: dict[tuple[str, str], list[tuple[float, float | None, float | None]]] = {}
        self._override_static_kij: dict[tuple[str, str], list[dict]] = {}
        self._override_temperature_kij: dict[tuple[str, str], list[dict]] = {}
        self._kij_cache: dict[tuple[str, str, float | None], float] = {}
        self._pair_keys: list[tuple[str, str]] = []
        self._pair_static_kij: tuple[float, ...] = ()
        self._pair_dynamic_indices: tuple[int, ...] = ()
        self._pair_dkij_indices: tuple[int, ...] = ()
        self._pair_kij_values_cache: dict[float | None, tuple[float, ...]] = {}
        self._pair_dkij_values_cache: dict[float | None, tuple[float, ...]] = {}
        self._phi_phi_k_cache: dict[tuple, dict[str, float]] = {}
        self.warnings: list[str] = []
        self._warning_keys: set[str] = set()

        if self.model == 'RKSMHV2' and (not components or len(set(components)) != len(components)):
            raise CubicEOSError('RKSMHV2 requires a nonempty list of unique components')

        for comp in components:
            if hasattr(self.db, 'get_user_component'):
                props = self.db.get_user_component(comp)
            else:
                props = self.db.get(comp)
            if props is None:
                raise CubicEOSError(f"Component '{comp}' not found in database")
            if props.Tc is None or props.Pc is None or props.omega is None:
                raise CubicEOSError(
                    f"{self.model} requires Tc, Pc, and omega for '{comp}'"
                )
            if self.model == 'RKSMHV2' and (
                not all(math.isfinite(v) for v in (props.Tc, props.Pc, props.omega))
                or props.Tc <= 0.0 or props.Pc <= 0.0
            ):
                raise CubicEOSError(f'RKSMHV2 requires finite omega and positive finite Tc/Pc for {comp}')
            self.props[comp] = props
            self.component_cas[comp] = cas_for_component(comp, props)
            omega_a, omega_b = self._omega_constants()
            a0 = omega_a * R_CM3**2 * props.Tc**2 / props.Pc
            b = omega_b * R_CM3 * props.Tc / props.Pc
            mc_parameters = self._mathias_copeman_parameters(comp, props)
            prsv_constants = self._prsv_constants(comp, props)
            missing_prsv_constants = (self.use_prsv1 or self.use_prsv2) and prsv_constants is None
            if missing_prsv_constants:
                prsv_constants = (0.0, 0.0, 0.0)
            twu_constants, twu_partial_override = self._twu_constants(comp, props)
            self.params[comp] = CubicPureParams(
                comp=comp,
                Tc=props.Tc,
                Pc=props.Pc,
                omega=props.omega,
                a0=a0,
                b=b,
                mc_c1=mc_parameters['c1'] if mc_parameters is not None else None,
                mc_c2=mc_parameters['c2'] if mc_parameters is not None else None,
                mc_c3=mc_parameters['c3'] if mc_parameters is not None else None,
                mc_Tmin=mc_parameters.get('Tmin_K') if mc_parameters is not None else None,
                mc_Tmax=mc_parameters.get('Tmax_K') if mc_parameters is not None else None,
                mc_source=mc_parameters.get('source') if mc_parameters is not None else None,
                kappa1=prsv_constants[0] if prsv_constants is not None else None,
                kappa2=prsv_constants[1] if prsv_constants is not None else None,
                kappa3=prsv_constants[2] if prsv_constants is not None else None,
                twu_l=twu_constants[0] if twu_constants is not None else None,
                twu_m=twu_constants[1] if twu_constants is not None else None,
                twu_n=twu_constants[2] if twu_constants is not None else None,
            )
            if self.use_mathias_copeman and mc_parameters is None:
                self.add_warning(
                    f"{self.model} Mathias-Copeman alpha parameters unavailable for "
                    f"{comp}; falling back to {self._parent_bm_model()} alpha for this component."
                )
            if missing_prsv_constants:
                self.add_warning(
                    f"{self.model} PRSV alpha parameters unavailable for "
                    f"{comp}; using PRSV kappa0-only alpha for this component."
                )
            if self.use_twu and twu_constants is None:
                if twu_partial_override:
                    self.add_warning(
                        f"{self.model} Twu alpha parameters incomplete for "
                        f"{comp}; falling back to {self._parent_base_model()} alpha for this component."
                    )
                else:
                    self.add_warning(
                        f"{self.model} Twu alpha parameters unavailable for "
                        f"{comp}; falling back to {self._parent_base_model()} alpha for this component."
                    )
        if self.model == 'RKSMHV2':
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .thermodynamics_models.mhv2 import MHV2ExcessGibbs
                from .thermodynamics_models.ge_eos import ModifiedHuronVidalSecondOrderMixingRule
            else:
                from thermodynamics_models.mhv2 import MHV2ExcessGibbs
                from thermodynamics_models.ge_eos import ModifiedHuronVidalSecondOrderMixingRule
            try:
                self._ge_provider = MHV2ExcessGibbs(
                    components, self.props, self.db, self.component_cas, unifac_groups)
            except ValueError as error:
                raise CubicEOSError(str(error)) from error
            self._ge_mixing = ModifiedHuronVidalSecondOrderMixingRule()
            self._compiled_backend = None
            return
        self._initialize_kij_tables()
        self._initialize_kij_overrides(interaction_overrides or [])
        self._rebuild_kij_pair_cache()
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compiled_cubic_eos import CompiledCubicEOSBackend
            else:
                from compiled_cubic_eos import CompiledCubicEOSBackend

            self._compiled_backend = CompiledCubicEOSBackend.from_eos(self)
        except Exception:
            self._compiled_backend = None

    def add_warning(self, message: str) -> None:
        text = str(message).strip()
        if not text or text in self._warning_keys:
            return
        self._warning_keys.add(text)
        self.warnings.append(text)

    def _initialize_kij_tables(self) -> None:
        for i, comp_i in enumerate(self.components):
            for j, comp_j in enumerate(self.components):
                key = (comp_i, comp_j)
                cas_i = self.component_cas.get(comp_i)
                cas_j = self.component_cas.get(comp_j)
                records = eos_binary_interaction_records(
                    self.model,
                    cas_i,
                    cas_j,
                )
                if not records:
                    self._static_kij[key] = 0.0
                    if i < j:
                        if not cas_i or not cas_j:
                            missing = [
                                comp for comp, comp_cas in ((comp_i, cas_i), (comp_j, cas_j))
                                if not comp_cas
                            ]
                            self.add_warning(
                                f"{self.model} EOS binary interaction parameters unavailable for "
                                f"{comp_i}/{comp_j}: missing CAS for {', '.join(missing)}; "
                                "using k_ij=0."
                            )
                        else:
                            self.add_warning(
                                f"{self.model} EOS binary interaction parameters missing for "
                                f"{comp_i}/{comp_j}; using k_ij=0."
                            )
                    continue
                ranged = []
                for record in records:
                    temperature_range = record.get("temperature_range")
                    if temperature_range is not None:
                        low, high = temperature_range
                        ranged.append((float(record["kij"]), low, high))
                static = [
                    float(record["kij"])
                    for record in records
                    if record.get("temperature_range") is None
                ]
                if ranged:
                    self._temperature_kij[key] = ranged
                if static:
                    self._static_kij[key] = median(static)
                elif not ranged:
                    self._static_kij[key] = median(float(record["kij"]) for record in records)

    def _initialize_kij_overrides(self, interaction_overrides: list[dict]) -> None:
        model_key = canonical_eos_model_key(self.model)
        for record in interaction_overrides:
            if str(record.get("model", "")).upper() != model_key:
                continue
            comp1 = record.get("component1")
            comp2 = record.get("component2")
            if comp1 not in self.components or comp2 not in self.components or comp1 == comp2:
                continue
            normalized = dict(record)
            has_temperature_range = (
                normalized.get("temperature_range") is not None
                or (
                    normalized.get("Tmin_K") is not None
                    and normalized.get("Tmax_K") is not None
                )
            )
            if normalized.get("temperature_range") is None and has_temperature_range:
                normalized["temperature_range"] = (
                    float(normalized["Tmin_K"]),
                    float(normalized["Tmax_K"]),
                )
            keys = ((comp1, comp2), (comp2, comp1))
            if has_temperature_range:
                for key in keys:
                    self._override_temperature_kij.setdefault(key, []).append(normalized)
            else:
                for key in keys:
                    self._override_static_kij.setdefault(key, []).append(normalized)

    @staticmethod
    def _records_have_temperature_formula(records: list[dict]) -> bool:
        return any(
            any(field in record for field in ("kij_a", "kij_b", "kij_c"))
            for record in records
        )

    def _rebuild_kij_pair_cache(self) -> None:
        pair_keys: list[tuple[str, str]] = []
        static_values: list[float] = []
        dynamic_indices: list[int] = []
        dkij_indices: list[int] = []

        for comp_i in self.components:
            for comp_j in self.components:
                key = (comp_i, comp_j)
                pair_index = len(pair_keys)
                pair_keys.append(key)
                static_values.append(self._kij(comp_i, comp_j, None))

                has_temperature_selection = bool(
                    self._temperature_kij.get(key)
                    or self._override_temperature_kij.get(key)
                )
                static_overrides = self._override_static_kij.get(key, [])
                has_temperature_formula = self._records_have_temperature_formula(static_overrides)
                if has_temperature_selection or has_temperature_formula:
                    dynamic_indices.append(pair_index)
                if (
                    has_temperature_formula
                    or self._records_have_temperature_formula(self._override_temperature_kij.get(key, []))
                ):
                    dkij_indices.append(pair_index)

        self._pair_keys = pair_keys
        self._pair_static_kij = tuple(static_values)
        self._pair_dynamic_indices = tuple(dynamic_indices)
        self._pair_dkij_indices = tuple(dkij_indices)
        self._pair_kij_values_cache.clear()
        self._pair_dkij_values_cache.clear()

    def _pair_kij_values(self, T: Optional[float]) -> tuple[float, ...]:
        if not self._pair_dynamic_indices:
            return self._pair_static_kij
        cache_key = None if T is None else float(T)
        cached = self._pair_kij_values_cache.get(cache_key)
        if cached is not None:
            return cached
        values = list(self._pair_static_kij)
        for index in self._pair_dynamic_indices:
            comp_i, comp_j = self._pair_keys[index]
            values[index] = self._kij(comp_i, comp_j, T)
        result = tuple(values)
        if len(self._pair_kij_values_cache) > 20000:
            self._pair_kij_values_cache.clear()
        self._pair_kij_values_cache[cache_key] = result
        return result

    def _pair_dkij_dT_values(self, T: Optional[float]) -> tuple[float, ...]:
        cache_key = None if T is None else float(T)
        cached = self._pair_dkij_values_cache.get(cache_key)
        if cached is not None:
            return cached
        values = [0.0] * len(self._pair_keys)
        for index in self._pair_dkij_indices:
            comp_i, comp_j = self._pair_keys[index]
            values[index] = self._dkij_dT(comp_i, comp_j, T)
        result = tuple(values)
        if len(self._pair_dkij_values_cache) > 20000:
            self._pair_dkij_values_cache.clear()
        self._pair_dkij_values_cache[cache_key] = result
        return result

    def _compiled_kij_values(self, T: float) -> tuple[float, ...]:
        """Preserve runtime alpha warnings before entering compiled code."""
        if self.use_mathias_copeman:
            for params in self.params.values():
                if self._has_mc_constants(params):
                    self._warn_mc_temperature_range(params, T)
        return self._pair_kij_values(T)

    @staticmethod
    def _canonical_model(model: str) -> str:
        text = canonical_lyngby_method(model.upper().replace("_", "-"))
        if text == 'RKSMHV2':
            return 'RKSMHV2'
        if text in ("RKS", "RK-SOAVE", "SOAVE-REDLICH-KWONG"):
            return "SRK"
        if text in ("SRK-BM", "RKS-BM", "RK-SOAVE-BM"):
            return "RKS-BM"
        if text in ("SRK-MC", "RKS-MC", "RK-SOAVE-MC"):
            return "SRK-MC"
        if text in ("SRK-TWU", "RKS-TWU", "RK-SOAVE-TWU"):
            return "SRK-TWU"
        if text in ("PENG-ROBINSON",):
            return "PR"
        if text in ("PENG-ROBINSON-BM",):
            return "PR-BM"
        if text in ("PENG-ROBINSON-MC",):
            return "PR-MC"
        if text in ("PR-TWU", "PENG-ROBINSON-TWU"):
            return "PR-TWU"
        if text in (
            "PRSV",
            "PRSV1",
            "PR-SV",
            "PR-SV1",
            "PENG-ROBINSON-SV",
            "PENG-ROBINSON-SV1",
            "PENG-ROBINSON-STRYJEK-VERA",
        ):
            return "PRSV1"
        if text in ("PRSV2", "PR-SV2", "PENG-ROBINSON-SV2"):
            return "PRSV2"
        if text in (
            "SRK", "PR", "RKS-BM", "PR-BM", "SRK-MC", "PR-MC",
            "PRSV1", "PRSV2", "SRK-TWU", "PR-TWU",
        ):
            return text
        raise CubicEOSError(f"Unsupported cubic EOS model '{model}'")

    def _parent_bm_model(self) -> str:
        return "PR-BM" if self.family == "PR" else "RKS-BM"

    def _parent_base_model(self) -> str:
        return "PR" if self.family == "PR" else "SRK"

    def _omega_constants(self) -> tuple[float, float]:
        if self.family == "PR":
            return 0.45724, 0.07780
        return 0.42747, 0.08664

    def _delta_roots(self) -> tuple[float, float]:
        if self.family == "PR":
            root2 = math.sqrt(2.0)
            return 1.0 + root2, 1.0 - root2
        return 1.0, 0.0

    def _m_soave(self, omega: float) -> float:
        if self.family == "PR":
            return 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        return 0.480 + 1.574 * omega - 0.176 * omega**2

    @staticmethod
    @lru_cache(maxsize=1)
    def _srk_mathias_copeman_table() -> dict:
        path = Path(__file__).parent / "data" / "chemsep_srk_mc.json"
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {}
        data = payload.get("data", payload)
        return data if isinstance(data, dict) else {}

    @staticmethod
    @lru_cache(maxsize=1)
    def _prsv_parameter_table() -> dict:
        path = Path(__file__).parent / "data" / "prsv_parameters_cas.json"
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {}
        data = payload.get("components", payload)
        return data if isinstance(data, dict) else {}

    @staticmethod
    @lru_cache(maxsize=2)
    def _twu_parameter_table(family: str) -> dict:
        filename = "PRTwu.json" if family == "PR" else "SRKTwu.json"
        path = Path(__file__).parent / "data" / filename
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {}
        data = payload.get("data", payload)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _complete_mc_constants_from_props(props: ChemicalProperties) -> Optional[tuple[float, float, float]]:
        c1 = getattr(props, "mc_c1", None)
        if c1 is None:
            return None
        try:
            return (
                float(c1),
                float(getattr(props, "mc_c2", 0.0) or 0.0),
                float(getattr(props, "mc_c3", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _complete_prsv_constants_from_props(props: ChemicalProperties) -> Optional[tuple[float, float, float]]:
        kappa1 = getattr(props, "kappa1", None)
        if kappa1 is None:
            return None
        try:
            return (
                float(kappa1),
                float(getattr(props, "kappa2", 0.0) or 0.0),
                float(getattr(props, "kappa3", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _complete_twu_constants_from_props(props: ChemicalProperties) -> Optional[tuple[float, float, float]]:
        values = (getattr(props, "twu_l", None), getattr(props, "twu_m", None), getattr(props, "twu_n", None))
        if any(value is None for value in values):
            return None
        try:
            return tuple(float(value) for value in values)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _has_any_twu_alpha_props(props: ChemicalProperties) -> bool:
        return any(getattr(props, attr, None) is not None for attr in ("twu_l", "twu_m", "twu_n"))

    def _mathias_copeman_parameters(
        self,
        comp: str,
        props: ChemicalProperties,
    ) -> Optional[dict]:
        if self.model == 'RKSMHV2':
            values = [getattr(props, f'mc_c{i}', None) for i in (1, 2, 3)]
            if any(v is not None for v in values):
                if values[0] is None or any(v is not None and not math.isfinite(float(v)) for v in values):
                    raise CubicEOSError(f'RKSMHV2 requires mc_c1 and finite alpha coefficients for {comp}')
        provided = self._complete_mc_constants_from_props(props)
        if provided is not None:
            return {
                'c1': provided[0],
                'c2': provided[1],
                'c3': provided[2],
                'Tmin_K': None,
                'Tmax_K': None,
                'source': 'component property override',
            }
        if self.model == 'RKSMHV2':
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .lyngby_parameters import parameter_table
            else:
                from lyngby_parameters import parameter_table
            values = parameter_table(True)['alpha'].get(self.component_cas[comp])
            if values is not None:
                return {**values, 'source': 'Dahl et al. (1991), Table I'}
            self.add_warning(f'RKSMHV2 Mathias-Copeman parameters unavailable for {comp}; using Soave alpha.')
            return None
        if not self.use_mathias_copeman or self.family != "SRK":
            return None
        cas = self.component_cas.get(comp) or getattr(props, "CAS", None)
        if not cas:
            return None
        psrk = compatible_psrk_mathias_copeman(cas, props.Tc, props.Pc)
        if psrk is not None:
            return psrk
        entry = self._srk_mathias_copeman_table().get(cas)
        if not isinstance(entry, dict):
            return None
        try:
            return {
                'c1': float(entry["MCSRKC1"]),
                'c2': float(entry["MCSRKC2"]),
                'c3': float(entry["MCSRKC3"]),
                'Tmin_K': None,
                'Tmax_K': None,
                'source': 'ChemSep SRK-MC parameter table',
            }
        except (KeyError, TypeError, ValueError):
            return None

    def _prsv_constants(self, comp: str, props: ChemicalProperties) -> Optional[tuple[float, float, float]]:
        provided = self._complete_prsv_constants_from_props(props)
        if provided is not None:
            return provided
        if not (self.use_prsv1 or self.use_prsv2):
            return None
        cas = self.component_cas.get(comp) or getattr(props, "CAS", None)
        if not cas:
            return None
        entry = self._prsv_parameter_table().get(cas)
        if not isinstance(entry, dict):
            return None
        variant = "prsv2" if self.use_prsv2 else "prsv1"
        values = entry.get(variant)
        if not isinstance(values, dict):
            return None
        try:
            return (
                float(values["kappa1"]),
                float(values.get("kappa2", 0.0) or 0.0),
                float(values.get("kappa3", 0.0) or 0.0),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _twu_constants(self, comp: str, props: ChemicalProperties) -> tuple[Optional[tuple[float, float, float]], bool]:
        provided = self._complete_twu_constants_from_props(props)
        if provided is not None:
            return provided, False
        partial_override = self._has_any_twu_alpha_props(props)
        if partial_override or not self.use_twu:
            return None, partial_override
        cas = self.component_cas.get(comp) or getattr(props, "CAS", None)
        if not cas:
            return None, False
        entry = self._twu_parameter_table(self.family).get(cas)
        if not isinstance(entry, dict):
            return None, False
        prefix = "TwuPR" if self.family == "PR" else "TwuSRK"
        try:
            values = (
                entry[f"{prefix}L"],
                entry[f"{prefix}M"],
                entry[f"{prefix}N"],
            )
            if any(value is None for value in values):
                return None, False
            return tuple(float(value) for value in values), False
        except (KeyError, TypeError, ValueError):
            return None, False

    @staticmethod
    def _has_mc_constants(params: CubicPureParams) -> bool:
        return (
            params.mc_c1 is not None
            and params.mc_c2 is not None
            and params.mc_c3 is not None
        )

    def _warn_mc_temperature_range(
        self,
        params: CubicPureParams,
        T: float,
    ) -> None:
        if not params.mc_source or (
            params.mc_Tmin is None and params.mc_Tmax is None
        ):
            return
        below = params.mc_Tmin is not None and T < params.mc_Tmin
        above = params.mc_Tmax is not None and T > params.mc_Tmax
        if not (below or above):
            return
        if params.mc_Tmin is not None and params.mc_Tmax is not None:
            interval = f"{params.mc_Tmin:g}-{params.mc_Tmax:g} K"
        elif params.mc_Tmin is not None:
            interval = f"T >= {params.mc_Tmin:g} K"
        else:
            interval = f"T <= {params.mc_Tmax:g} K"
        self.add_warning(
            f"{self.model} uses {params.mc_source} for {params.comp} outside "
            f"its published temperature range ({interval})."
        )

    @staticmethod
    def _has_prsv_constants(params: CubicPureParams) -> bool:
        return params.kappa1 is not None

    @staticmethod
    def _has_twu_constants(params: CubicPureParams) -> bool:
        return (
            params.twu_l is not None
            and params.twu_m is not None
            and params.twu_n is not None
        )

    @staticmethod
    def _alpha_mathias_copeman(params: CubicPureParams, Tr: float) -> float:
        theta = 1.0 - math.sqrt(Tr)
        f = 1.0 + params.mc_c1 * theta + params.mc_c2 * theta**2 + params.mc_c3 * theta**3
        return f * f

    @staticmethod
    def _dalpha_mathias_copeman_dT(params: CubicPureParams, Tr: float) -> float:
        sqrt_Tr = math.sqrt(Tr)
        theta = 1.0 - sqrt_Tr
        f = 1.0 + params.mc_c1 * theta + params.mc_c2 * theta**2 + params.mc_c3 * theta**3
        df_dtheta = params.mc_c1 + 2.0 * params.mc_c2 * theta + 3.0 * params.mc_c3 * theta**2
        dtheta_dT = -1.0 / (2.0 * params.Tc * sqrt_Tr)
        return 2.0 * f * df_dtheta * dtheta_dT

    @staticmethod
    def _prsv_kappa0(omega: float) -> float:
        return 0.378893 + 1.4897153 * omega - 0.17131848 * omega**2 + 0.0196554 * omega**3

    @staticmethod
    def _prsv1_taper(Tr: float) -> tuple[float, float]:
        if Tr <= 0.7:
            return 1.0, 0.0
        if Tr >= 1.0:
            return 0.0, 0.0
        s = (Tr - 0.7) / 0.3
        return 1.0 - s**2 * (3.0 - 2.0 * s), -6.0 * s * (1.0 - s) / 0.3

    @staticmethod
    def _prsv_g(Tr: float) -> tuple[float, float]:
        sqrt_Tr = math.sqrt(Tr)
        g = (1.0 + sqrt_Tr) * (0.7 - Tr)
        dg_dTr = (0.7 - Tr) / (2.0 * sqrt_Tr) - (1.0 + sqrt_Tr)
        return g, dg_dTr

    def _prsv_kappa(self, params: CubicPureParams, Tr: float) -> tuple[float, float]:
        kappa0 = self._prsv_kappa0(params.omega)
        g, dg_dTr = self._prsv_g(Tr)
        if self.use_prsv2:
            sqrt_Tr = math.sqrt(Tr)
            h = params.kappa1 + params.kappa2 * (params.kappa3 - Tr) * (1.0 - sqrt_Tr)
            dh_dTr = params.kappa2 * (-(1.0 - sqrt_Tr) - (params.kappa3 - Tr) / (2.0 * sqrt_Tr))
            return kappa0 + h * g, dh_dTr * g + h * dg_dTr
        taper, dtaper_dTr = self._prsv1_taper(Tr)
        correction = params.kappa1 * taper * g
        dcorrection_dTr = params.kappa1 * (dtaper_dTr * g + taper * dg_dTr)
        return kappa0 + correction, dcorrection_dTr

    def _alpha_prsv(self, params: CubicPureParams, Tr: float) -> float:
        sqrt_Tr = math.sqrt(Tr)
        kappa, _ = self._prsv_kappa(params, Tr)
        f = 1.0 + kappa * (1.0 - sqrt_Tr)
        return f * f

    def _dalpha_prsv_dT(self, params: CubicPureParams, Tr: float) -> float:
        sqrt_Tr = math.sqrt(Tr)
        theta = 1.0 - sqrt_Tr
        kappa, dkappa_dTr = self._prsv_kappa(params, Tr)
        f = 1.0 + kappa * theta
        dtheta_dTr = -1.0 / (2.0 * sqrt_Tr)
        df_dTr = dkappa_dTr * theta + kappa * dtheta_dTr
        return 2.0 * f * df_dTr / params.Tc

    @staticmethod
    def _alpha_twu(params: CubicPureParams, Tr: float) -> float:
        l_value = params.twu_l
        m_value = params.twu_m
        n_value = params.twu_n
        return Tr ** (n_value * (m_value - 1.0)) * math.exp(
            l_value * (1.0 - Tr ** (n_value * m_value))
        )

    @staticmethod
    def _dalpha_twu_dT(params: CubicPureParams, Tr: float) -> float:
        l_value = params.twu_l
        m_value = params.twu_m
        n_value = params.twu_n
        alpha = CubicEOS._alpha_twu(params, Tr)
        exponent_a = n_value * (m_value - 1.0)
        exponent_b = n_value * m_value
        dln_alpha_dTr = exponent_a / Tr - l_value * exponent_b * Tr ** (exponent_b - 1.0)
        return alpha * dln_alpha_dTr / params.Tc

    def _alpha_boston_mathias(self, params: CubicPureParams, Tr: float, m: float) -> float:
        exponent = (1.0 - 2.0 / (2.0 + m)) * (1.0 - Tr ** (1.0 + m / 2.0))
        return math.exp(2.0 * exponent)

    def _dalpha_boston_mathias_dT(self, params: CubicPureParams, Tr: float, m: float) -> float:
        c = 1.0 - 2.0 / (2.0 + m)
        power = 1.0 + m / 2.0
        exponent = c * (1.0 - Tr**power)
        alpha = math.exp(2.0 * exponent)
        dexponent_dT = -c * power * Tr ** (power - 1.0) / params.Tc
        return alpha * 2.0 * dexponent_dT

    def _alpha(self, params: CubicPureParams, T: float) -> float:
        Tr = max(T / params.Tc, 1e-12)
        m = self._m_soave(params.omega)
        if self.model == 'RKSMHV2' and self._has_mc_constants(params):
            if Tr <= 1.0:
                return self._alpha_mathias_copeman(params, Tr)
            return (1.0 + params.mc_c1 * (1.0 - math.sqrt(Tr))) ** 2
        if self.use_mathias_copeman and self._has_mc_constants(params):
            self._warn_mc_temperature_range(params, T)
        if self.use_twu and self._has_twu_constants(params):
            return self._alpha_twu(params, Tr)
        if (self.use_prsv1 or self.use_prsv2) and self._has_prsv_constants(params):
            return self._alpha_prsv(params, Tr)
        if self.use_mathias_copeman and self._has_mc_constants(params) and Tr <= 1.0:
            return self._alpha_mathias_copeman(params, Tr)
        if (self.use_boston_mathias or self.use_mathias_copeman) and Tr > 1.0:
            if self.use_mathias_copeman and self._has_mc_constants(params):
                m = params.mc_c1
            return self._alpha_boston_mathias(params, Tr, m)
        return (1.0 + m * (1.0 - math.sqrt(Tr))) ** 2

    def _dalpha_dT(self, params: CubicPureParams, T: float) -> float:
        Tr = max(T / params.Tc, 1e-12)
        m = self._m_soave(params.omega)
        if self.model == 'RKSMHV2' and self._has_mc_constants(params):
            if Tr <= 1.0:
                return self._dalpha_mathias_copeman_dT(params, Tr)
            return -(1.0 + params.mc_c1 * (1.0 - math.sqrt(Tr))) * params.mc_c1 / math.sqrt(T * params.Tc)
        if self.use_mathias_copeman and self._has_mc_constants(params):
            self._warn_mc_temperature_range(params, T)
        if self.use_twu and self._has_twu_constants(params):
            return self._dalpha_twu_dT(params, Tr)
        if (self.use_prsv1 or self.use_prsv2) and self._has_prsv_constants(params):
            return self._dalpha_prsv_dT(params, Tr)
        if self.use_mathias_copeman and self._has_mc_constants(params) and Tr <= 1.0:
            return self._dalpha_mathias_copeman_dT(params, Tr)
        if (self.use_boston_mathias or self.use_mathias_copeman) and Tr > 1.0:
            if self.use_mathias_copeman and self._has_mc_constants(params):
                m = params.mc_c1
            return self._dalpha_boston_mathias_dT(params, Tr, m)
        f = 1.0 + m * (1.0 - math.sqrt(Tr))
        return -f * m / (params.Tc * math.sqrt(Tr))

    def pure_a(self, comp: str, T: float) -> float:
        params = self.params[comp]
        return params.a0 * self._alpha(params, T)

    def pure_da_dT(self, comp: str, T: float) -> float:
        params = self.params[comp]
        return params.a0 * self._dalpha_dT(params, T)

    def _kij(self, comp_i: str, comp_j: str, T: Optional[float] = None) -> float:
        key = (comp_i, comp_j)
        cache_key = (comp_i, comp_j, T)
        if cache_key in self._kij_cache:
            return self._kij_cache[cache_key]
        override_value = self._override_kij(key, T)
        if override_value is not None:
            self._kij_cache[cache_key] = override_value
            return override_value
        records = self._temperature_kij.get(key)
        if records and T is not None:
            in_range = [
                record for record in records
                if self._temperature_kij_score(record, T)[0] is False
            ]
            if in_range:
                value = min(in_range, key=lambda record: self._temperature_kij_score(record, T))[0]
            elif key in self._static_kij:
                nearest = min(records, key=lambda record: self._temperature_kij_score(record, T))[0]
                value = 0.5 * (self._static_kij[key] + nearest)
            else:
                value = min(records, key=lambda record: self._temperature_kij_score(record, T))[0]
        elif key in self._static_kij:
            value = self._static_kij[key]
        elif records:
            value = median(record[0] for record in records)
        else:
            value = 0.0
        self._kij_cache[cache_key] = value
        return value

    def _override_kij(self, key: tuple[str, str], T: Optional[float]) -> Optional[float]:
        ranged = getattr(self, "_override_temperature_kij", {}).get(key, [])
        if ranged and T is not None:
            in_range = [
                record for record in ranged
                if self._override_temperature_score(record, T)[0] is False
            ]
            if in_range:
                selected = min(in_range, key=lambda record: self._override_temperature_score(record, T))
                return eos_record_kij(selected, T)
        static = getattr(self, "_override_static_kij", {}).get(key, [])
        if static:
            return median(eos_record_kij(record, T) for record in static)
        if ranged and T is None:
            return median(eos_record_kij(record, record.get("T_ref_K", 298.15)) for record in ranged)
        return None

    @staticmethod
    def _record_dkij_dT(record: dict, T: Optional[float]) -> float:
        if not any(key in record for key in ("kij_a", "kij_b", "kij_c")):
            return 0.0
        T_eval = float(T if T is not None else record.get("T_ref_K", record.get("Tref_K", 298.15)))
        return -float(record.get("kij_b", 0.0)) / (T_eval * T_eval) + float(record.get("kij_c", 0.0))

    def _override_dkij_dT(self, key: tuple[str, str], T: Optional[float]) -> Optional[float]:
        ranged = getattr(self, "_override_temperature_kij", {}).get(key, [])
        if ranged and T is not None:
            in_range = [
                record for record in ranged
                if self._override_temperature_score(record, T)[0] is False
            ]
            if in_range:
                selected = min(in_range, key=lambda record: self._override_temperature_score(record, T))
                return self._record_dkij_dT(selected, T)
        static = getattr(self, "_override_static_kij", {}).get(key, [])
        if static:
            return median(self._record_dkij_dT(record, T) for record in static)
        if ranged and T is None:
            return median(
                self._record_dkij_dT(record, record.get("T_ref_K", 298.15))
                for record in ranged
            )
        return None

    def _dkij_dT(self, comp_i: str, comp_j: str, T: Optional[float]) -> float:
        override_value = self._override_dkij_dT((comp_i, comp_j), T)
        if override_value is not None:
            return override_value
        return 0.0

    @staticmethod
    def _temperature_kij_score(record: tuple[float, float | None, float | None], T: float) -> tuple[bool, float, float]:
        _, low, high = record
        if low is None or high is None:
            return (True, float("inf"), float("inf"))
        if low == high:
            low -= EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K
            high += EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K
        if low <= T <= high:
            return (False, 0.0, high - low)
        return (True, min(abs(T - low), abs(T - high)), high - low)

    @staticmethod
    def _override_temperature_score(record: dict, T: float) -> tuple[bool, float, float]:
        low, high = record.get("temperature_range", (None, None))
        if low is None or high is None:
            return (True, float("inf"), float("inf"))
        low = float(low)
        high = float(high)
        if low == high:
            low -= EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K
            high += EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K
        if low <= T <= high:
            return (False, 0.0, high - low)
        return (True, min(abs(T - low), abs(T - high)), high - low)

    def mixture_params(
        self,
        T: float,
        composition: dict[str, float],
    ) -> tuple[float, float, dict[str, float], dict[str, float]]:
        x = self._normalized_composition(composition)
        comps = self.components
        x_values = [x[comp] for comp in comps]
        if self._ge_provider is not None:
            mixing = self._ge_state(T, x)
            return (mixing.D * mixing.b * R_CM3 * T, mixing.b,
                    {comp: self.pure_a(comp, T) for comp in comps},
                    {comp: self.params[comp].b for comp in comps})
        if self._compiled_backend is not None:
            a_mix, b_mix, a_values, _, _ = (
                self._compiled_backend.mixture_parameters(
                    T,
                    x_values,
                    self._compiled_kij_values(T),
                )
            )
            return (
                float(a_mix),
                float(b_mix),
                dict(zip(comps, (float(value) for value in a_values))),
                {comp: self.params[comp].b for comp in comps},
            )
        a_values = [self.pure_a(comp, T) for comp in comps]
        b_values = [self.params[comp].b for comp in comps]
        a_i = dict(zip(comps, a_values))
        b_i = dict(zip(comps, b_values))
        kij_values = self._pair_kij_values(T)

        a_mix = 0.0
        pair_index = 0
        for i, x_i in enumerate(x_values):
            a_i_value = a_values[i]
            for j, x_j in enumerate(x_values):
                a_ij = math.sqrt(a_i_value * a_values[j]) * (1.0 - kij_values[pair_index])
                a_mix += x_i * x_j * a_ij
                pair_index += 1

        b_mix = sum(x_value * b_value for x_value, b_value in zip(x_values, b_values))
        return a_mix, b_mix, a_i, b_i

    def mixture_da_dT(self, T: float, composition: dict[str, float]) -> float:
        x = self._normalized_composition(composition)
        if self._ge_provider is not None:
            mixing = self._ge_state(T, x, temperature_derivative=True)
            return mixing.b * R_CM3 * (mixing.D + T * mixing.dD_dT)
        comps = self.components
        x_values = [x[comp] for comp in comps]
        if self._compiled_backend is not None:
            return float(self._compiled_backend.mixture_parameters(
                T,
                x_values,
                self._compiled_kij_values(T),
                self._pair_dkij_dT_values(T),
            )[-1])
        a_values = [self.pure_a(comp, T) for comp in comps]
        da_values = [self.pure_da_dT(comp, T) for comp in comps]
        kij_values = self._pair_kij_values(T)
        dkij_values = self._pair_dkij_dT_values(T)

        da_mix = 0.0
        pair_index = 0
        for i, x_i in enumerate(x_values):
            a_i_value = a_values[i]
            da_i_value = da_values[i]
            for j, x_j in enumerate(x_values):
                a_j_value = a_values[j]
                a_ij_base = math.sqrt(a_i_value * a_j_value)
                kij = kij_values[pair_index]
                if a_i_value <= 0.0 or a_j_value <= 0.0:
                    da_ij = 0.0
                else:
                    da_ij_base = 0.5 * a_ij_base * (
                        da_i_value / a_i_value + da_values[j] / a_j_value
                    )
                    da_ij = (
                        da_ij_base * (1.0 - kij)
                        - a_ij_base * dkij_values[pair_index]
                    )
                da_mix += x_i * x_j * da_ij
                pair_index += 1
        return da_mix

    def _ge_state(self, T, composition, *, temperature_derivative=False):
        """Cache mechanical and caloric mixing states separately.

        Fugacity/root iteration never consumes dD/dT, so avoid calculating
        excess-enthalpy derivatives on that hot path.
        """
        if not math.isfinite(T) or T <= 0.0:
            raise CubicEOSError('RKSMHV2 temperature must be positive and finite')
        x = tuple(composition[comp] for comp in self.components)
        key = (T, x, temperature_derivative)
        if key not in self._ge_cache:
            b = tuple(self.params[comp].b for comp in self.components)
            d = tuple(self.pure_a(comp, T) / (bi * R_CM3 * T)
                      for comp, bi in zip(self.components, b))
            dd = tuple(self.pure_da_dT(comp, T) / (bi * R_CM3 * T) - di / T
                       for comp, bi, di in zip(self.components, b, d))
            try:
                state = self._ge_mixing.mix(x, b, d, dd,
                    self._ge_provider.excess_gibbs_state(
                        x, T, temperature_derivative=temperature_derivative))
            except (ValueError, OverflowError) as error:
                raise CubicEOSError(f'Invalid RKSMHV2 mixing state: {error}') from error
            if len(self._ge_cache) >= 20000:
                self._ge_cache.clear()
            self._ge_cache[key] = state
        return self._ge_cache[key]

    def _normalized_composition(self, composition: dict[str, float]) -> dict[str, float]:
        if self._ge_provider is not None:
            if set(composition) - set(self.components):
                raise CubicEOSError('RKSMHV2 composition contains unknown components')
            if any(not math.isfinite(float(v)) or float(v) < 0.0 for v in composition.values()):
                raise CubicEOSError('RKSMHV2 mole fractions must be nonnegative and finite')
            if sum(composition.values()) <= 0.0:
                raise CubicEOSError('RKSMHV2 composition must have positive total')
        values = {comp: max(float(composition.get(comp, 0.0)), 0.0) for comp in self.components}
        total = sum(values.values())
        if total <= 0.0:
            return {comp: 1.0 / len(self.components) for comp in self.components}
        return {comp: value / total for comp, value in values.items()}

    def _composition_cache_key(self, composition: dict[str, float]) -> tuple[tuple[str, float], ...]:
        return tuple(
            (comp, float(composition.get(comp, 0.0)))
            for comp in self.components
        )

    @staticmethod
    def _cbrt(value: float) -> float:
        return math.copysign(abs(value) ** (1.0 / 3.0), value)

    @classmethod
    def _solve_monic_cubic(cls, a: float, b: float, c: float) -> list[float]:
        """Return the real roots of z^3 + a*z^2 + b*z + c = 0."""
        p = b - a * a / 3.0
        q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c
        shift = a / 3.0
        discriminant = (0.5 * q) ** 2 + (p / 3.0) ** 3
        tolerance = 1e-14

        if discriminant > tolerance:
            sqrt_discriminant = math.sqrt(discriminant)
            y = (
                cls._cbrt(-0.5 * q + sqrt_discriminant)
                + cls._cbrt(-0.5 * q - sqrt_discriminant)
            )
            return [y - shift]

        if abs(p) < tolerance:
            return [-shift]

        argument = (3.0 * q / (2.0 * p)) * math.sqrt(-3.0 / p)
        argument = max(-1.0, min(1.0, argument))
        theta = math.acos(argument)
        radius = 2.0 * math.sqrt(-p / 3.0)
        return [
            radius * math.cos((theta + 2.0 * math.pi * k) / 3.0) - shift
            for k in range(3)
        ]

    def compressibility_roots(self, T: float, P: float, composition: dict[str, float]) -> list[float]:
        if self._ge_provider is not None and (not math.isfinite(P) or P <= 0.0):
            raise CubicEOSError('RKSMHV2 pressure must be positive and finite')
        if self._compiled_backend is not None:
            x = self._normalized_composition(composition)
            return [
                float(value)
                for value in self._compiled_backend.compressibility_roots(
                    T,
                    P,
                    [x[component] for component in self.components],
                    self._compiled_kij_values(T),
                )
            ]
        a_mix, b_mix, _, _ = self.mixture_params(T, composition)
        A = a_mix * P / (R_CM3**2 * T**2)
        B = b_mix * P / (R_CM3 * T)
        delta1, delta2 = self._delta_roots()
        u = delta1 + delta2
        w = delta1 * delta2

        roots = self._solve_monic_cubic(
            -(1.0 + B - u * B),
            A + w * B**2 - u * B - u * B**2,
            -(A * B + w * B**2 + w * B**3),
        )
        return sorted(float(root) for root in roots if root > B + 1e-12)

    def compressibility_factor(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> float:
        if self._ge_provider is not None and phase not in ('vapor', 'liquid'):
            raise CubicEOSError('RKSMHV2 phase must be vapor or liquid')
        roots = self.compressibility_roots(T, P, composition)
        if not roots:
            if self._ge_provider is not None:
                raise CubicEOSError('RKSMHV2 has no physical compressibility root')
            return 1.0
        return max(roots) if phase == "vapor" else min(roots)

    def molar_volume(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> float:
        """Molar volume [cm3/mol]."""
        Z = self.compressibility_factor(T, P, composition, phase)
        return Z * R_CM3 * T / max(P, 1e-12)

    def departure_enthalpy(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> float:
        """EOS residual/departure enthalpy H-H(ideal gas) [kJ/kmol]."""
        if self._compiled_backend is not None:
            x = self._normalized_composition(composition)
            return self._compiled_backend.departure_enthalpy(
                T,
                P,
                [x[component] for component in self.components],
                phase,
                self._compiled_kij_values(T),
                self._pair_dkij_dT_values(T),
            )
        x = self._normalized_composition(composition)
        a_mix, b_mix, _, _ = self.mixture_params(T, x)
        da_mix = self.mixture_da_dT(T, x)
        if b_mix <= 0.0:
            return 0.0
        B = b_mix * P / (R_CM3 * T)
        Z = self.compressibility_factor(T, P, x, phase)
        delta1, delta2 = self._delta_roots()
        delta_diff = delta1 - delta2
        log_arg = (Z + delta1 * B) / max(Z + delta2 * B, 1e-16)
        attraction_log = math.log(max(log_arg, 1e-16))
        h_bar_cm3_per_mol = R_CM3 * T * (Z - 1.0)
        h_bar_cm3_per_mol += (
            (T * da_mix - a_mix)
            / max(b_mix * delta_diff, 1e-16)
            * attraction_log
        )
        return 0.1 * h_bar_cm3_per_mol

    def departure_entropy(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> float:
        """EOS residual/departure entropy S-S(ideal gas) [kJ/kmol-K]."""
        if self._compiled_backend is not None:
            x = self._normalized_composition(composition)
            return self._compiled_backend.departure_entropy(
                T,
                P,
                [x[component] for component in self.components],
                phase,
                self._compiled_kij_values(T),
                self._pair_dkij_dT_values(T),
            )
        x = self._normalized_composition(composition)
        _, b_mix, _, _ = self.mixture_params(T, x)
        da_mix = self.mixture_da_dT(T, x)
        if b_mix <= 0.0:
            return 0.0
        B = b_mix * P / (R_CM3 * T)
        Z = self.compressibility_factor(T, P, x, phase)
        delta1, delta2 = self._delta_roots()
        delta_diff = delta1 - delta2
        log_arg = (Z + delta1 * B) / max(Z + delta2 * B, 1e-16)
        attraction_log = math.log(max(log_arg, 1e-16))
        s_bar_cm3_per_mol_K = R_CM3 * math.log(max(Z - B, 1e-16))
        s_bar_cm3_per_mol_K += (
            da_mix
            / max(b_mix * delta_diff, 1e-16)
            * attraction_log
        )
        return 0.1 * s_bar_cm3_per_mol_K

    def departure_gibbs(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> float:
        """EOS residual/departure Gibbs energy G-G(ideal gas) [kJ/kmol]."""
        return self.departure_enthalpy(T, P, composition, phase) - T * self.departure_entropy(
            T, P, composition, phase
        )

    def fugacity_coefficients(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str = "vapor",
    ) -> dict[str, float]:
        x = self._normalized_composition(composition)
        if self._compiled_backend is not None:
            values = self._compiled_backend.fugacity_coefficients(
                T,
                P,
                [x[component] for component in self.components],
                phase,
                self._compiled_kij_values(T),
            )
            return {
                component: float(value)
                for component, value in zip(self.components, values)
            }
        a_mix, b_mix, a_i, b_i = self.mixture_params(T, x)
        if (a_mix <= 0.0 and self._ge_provider is None) or b_mix <= 0.0:
            return {comp: 1.0 for comp in self.components}

        A = a_mix * P / (R_CM3**2 * T**2)
        B = b_mix * P / (R_CM3 * T)
        Z = self.compressibility_factor(T, P, x, phase)
        delta1, delta2 = self._delta_roots()
        delta_diff = delta1 - delta2
        log_arg = (Z + delta1 * B) / max(Z + delta2 * B, 1e-16)
        attraction_log = math.log(max(log_arg, 1e-16))

        phi: dict[str, float] = {}
        comps = self.components
        x_values = [x[comp] for comp in comps]
        a_values = [a_i[comp] for comp in comps]
        kij_values = self._pair_kij_values(T) if self._ge_provider is None else ()
        mixing = self._ge_state(T, x) if self._ge_provider is not None else None
        pair_index = 0
        for i, comp_i in enumerate(comps):
            bi_over_b = b_i[comp_i] / b_mix
            if mixing is not None:
                ln_phi = (bi_over_b * (Z - 1.0) - math.log(Z - B)
                          - mixing.composition_derivatives[i] * attraction_log / delta_diff)
                phi[comp_i] = math.exp(ln_phi)
                continue
            sum_aij = 0.0
            a_i_value = a_values[i]
            for j, x_j in enumerate(x_values):
                a_ij = math.sqrt(a_i_value * a_values[j]) * (1.0 - kij_values[pair_index])
                sum_aij += x_j * a_ij
                pair_index += 1
            attraction = (2.0 * sum_aij / a_mix) - bi_over_b
            ln_phi = bi_over_b * (Z - 1.0) - math.log(max(Z - B, 1e-16))
            ln_phi -= (A / max(B * delta_diff, 1e-16)) * attraction * attraction_log
            phi[comp_i] = max(math.exp(max(min(ln_phi, 50.0), -50.0)), 1e-12)
        return phi

    def phi_phi_K_values(
        self,
        T: float,
        P: float,
        liquid_composition: dict[str, float],
        max_iter: int = 80,
    ) -> dict[str, float]:
        x = self._normalized_composition(liquid_composition)
        cache_key = (
            float(T),
            float(P),
            self._composition_cache_key(x),
            int(max_iter),
        )
        cached = self._phi_phi_k_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        if self._compiled_backend is not None:
            values = self._compiled_backend.phi_phi_K_values(
                T,
                P,
                [x[component] for component in self.components],
                max_iter,
                self._compiled_kij_values(T),
            )
            result = {
                component: float(value)
                for component, value in zip(self.components, values)
            }
            return self._set_cached_phi_phi_K(cache_key, result)

        single_root = len(self.compressibility_roots(T, P, x)) < 2
        phi_l = self.fugacity_coefficients(T, P, x, "liquid")
        K = self._wilson_K(T, P)
        initial_K = dict(K)
        y = self._normalized_composition({comp: x[comp] * K[comp] for comp in self.components})

        for _ in range(max_iter):
            phi_v = self.fugacity_coefficients(T, P, y, "vapor")
            K_new = {
                comp: max(1e-8, min(1e8, phi_l.get(comp, 1.0) / max(phi_v.get(comp, 1.0), 1e-12)))
                for comp in self.components
            }
            y_new = self._normalized_composition({comp: x[comp] * K_new[comp] for comp in self.components})
            if max(abs(y_new[comp] - y[comp]) for comp in self.components) < 1e-9:
                # A single root is valid for either phase. Only reject the
                # homogeneous K=1 fixed point, which cannot locate a phase
                # boundary; retain Wilson's single-phase extrapolation there.
                if single_root and all(abs(K_new[c] - 1.0) < 1e-6 for c in self.components if x[c] > 0.0):
                    return self._set_cached_phi_phi_K(cache_key, initial_K)
                return self._set_cached_phi_phi_K(cache_key, K_new)
            K = {comp: 0.5 * K[comp] + 0.5 * K_new[comp] for comp in self.components}
            y = y_new
        return self._set_cached_phi_phi_K(cache_key, K)

    def _set_cached_phi_phi_K(self, key: tuple, values: dict[str, float]) -> dict[str, float]:
        if len(self._phi_phi_k_cache) > 20000:
            self._phi_phi_k_cache.clear()
        self._phi_phi_k_cache[key] = dict(values)
        return values

    def _wilson_K(self, T: float, P: float) -> dict[str, float]:
        values = {}
        for comp in self.components:
            props = self.props[comp]
            exponent = 5.373 * (1.0 + props.omega) * (1.0 - props.Tc / T)
            values[comp] = max(1e-8, min(1e8, props.Pc / max(P, 1e-12) * math.exp(exponent)))
        return values
