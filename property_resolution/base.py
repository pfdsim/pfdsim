from .common import *

from contextlib import contextmanager


class PropertyResolverBase:
        CACHE_DIR = Path(__file__).resolve().parent.parent / '.property_cache'
        RUNTIME_CACHE_PATH = (
            Path(__file__).resolve().parent.parent
            / 'data' / 'runtime' / 'property_cache.sqlite'
        )
        LIQUID_VOLUME_ZRA_CACHE_PATH = RUNTIME_CACHE_PATH
        PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        NIST_WEBBOOK = "https://webbook.nist.gov/cgi/cbook.cgi"

        def __init__(self):
            """Initialize property resolver"""
            self._online_cache = {}
            self._identifier_candidates_cache = {}
            self._formula_identity_match_cache = {}
            self._vapor_pressure_table_cache = {}
            self._database_props_cache = {}
            self._element_entropy_cache = None
            self._liquid_volume_zra_cache = None
            self._runtime_json_cache_state = None
            self._ideal_gas_cp_derived_cache_state = None
            self._xtb_rrho_artifact_cache_state = None
            self._liquid_cp_derived_cache_state = None
            self._solid_cp_derived_cache_state = None
            self._ideal_gas_cp_kernel_cache = {}
            self._liquid_cp_kernel_cache = {}
            self._solid_cp_kernel_cache = {}
            self._viscosity_kernel_cache = {}
            self._reichenberg_viscosity_input_cache = {}
            self._yoon_thodos_viscosity_input_cache = {}
            self._hvap_carboxylic_acid_cache = {}
            self._online_attempt_trackers = []


        @contextmanager
        def _online_attempt_scope(self, allow_online: bool):
            """Track whether an online-enabled selected result is complete."""
            tracker = OnlineAttemptTracker(bool(allow_online))
            self._online_attempt_trackers.append(tracker)
            try:
                yield tracker
            finally:
                if self._online_attempt_trackers:
                    if self._online_attempt_trackers[-1] is tracker:
                        self._online_attempt_trackers.pop()
                    else:
                        self._online_attempt_trackers.remove(tracker)


        def _record_online_attempt_state(
            self,
            state: OnlineAttemptState | str,
        ) -> None:
            """Propagate provider completeness to every enclosing cache build."""
            for tracker in tuple(self._online_attempt_trackers):
                tracker.record(state)


        @staticmethod
        def _online_attempt_is_persistable(
            allow_online: bool,
            state: OnlineAttemptState | str,
        ) -> bool:
            """Return whether a selected result is safe for memory/disk reuse."""
            state = OnlineAttemptState(state)
            if state is OnlineAttemptState.TRANSIENT_FAILURE:
                return False
            if not allow_online:
                return True
            return state in {
                OnlineAttemptState.NOT_NEEDED,
                OnlineAttemptState.COMPLETE_WITH_DATA,
                OnlineAttemptState.COMPLETE_NO_DATA,
            }


        def _runtime_json_cache(self):
            """Return the shared persistent cache, honoring custom test dirs."""
            from .runtime_cache import (
                LEGACY_PROPERTY_CACHE_DIR,
                SQLiteJSONCache,
                runtime_cache_path_for_legacy_directory,
            )

            path = runtime_cache_path_for_legacy_directory(
                self.CACHE_DIR,
                default_legacy_directory=LEGACY_PROPERTY_CACHE_DIR,
            )
            state = self._runtime_json_cache_state
            if state is not None and state[0] == path:
                return state[1]
            cache = SQLiteJSONCache(path, 'property_resolver')
            cache.migrate_json_directory(
                self.CACHE_DIR,
                migration_name='property-resolver-cache',
            )
            self._runtime_json_cache_state = path, cache
            return cache


        def _ideal_gas_cp_derived_cache(self):
            """Return the 30-day cache for normalized online/estimated Cp kernels."""
            from .runtime_cache import SQLiteJSONCache

            path = self._runtime_json_cache().path
            state = self._ideal_gas_cp_derived_cache_state
            if state is not None and state[0] == path:
                return state[1]
            cache = SQLiteJSONCache(path, 'ideal_gas_cp_derived_v1')
            self._ideal_gas_cp_derived_cache_state = path, cache
            return cache


        def _liquid_cp_derived_cache(self):
            """Return the 30-day cache for normalized online liquid-Cp kernels."""
            from .runtime_cache import SQLiteJSONCache

            path = self._runtime_json_cache().path
            state = self._liquid_cp_derived_cache_state
            if state is not None and state[0] == path:
                return state[1]
            cache = SQLiteJSONCache(path, 'liquid_cp_derived_v1')
            self._liquid_cp_derived_cache_state = path, cache
            return cache


        def _solid_cp_derived_cache(self):
            """Return the persistent cache for normalized online solid-Cp kernels."""
            from .runtime_cache import SQLiteJSONCache

            path = self._runtime_json_cache().path
            state = self._solid_cp_derived_cache_state
            if state is not None and state[0] == path:
                return state[1]
            cache = SQLiteJSONCache(path, 'solid_cp_derived_v1')
            self._solid_cp_derived_cache_state = path, cache
            return cache


        def _coerce_props(
            self,
            identifier: str,
            props: Optional[Dict[str, Any]],
            allow_online: bool = True,
        ) -> Dict[str, Any]:
            """Use hydrated database properties when a direct resolver call omits props."""
            if props is not None:
                return props

            cache_key = (str(identifier), bool(allow_online))
            cached = self._database_props_cache.get(cache_key)
            if cached is not None:
                return dict(cached)

            result = {}
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..chemical_properties import ChemicalDatabase
                else:
                    from chemical_properties import ChemicalDatabase
                database = ChemicalDatabase(enable_online=allow_online)
                chemical = database.get(identifier, fetch_online=allow_online)
                if chemical is not None:
                    result = chemical.to_dict()
            except Exception:
                result = {}

            if not (result.get('CAS') or result.get('cas')):
                identity = self._get_perry_identity(identifier)
                if identity:
                    for key, value in identity.items():
                        if value and not result.get(key):
                            result[key] = value

            self._database_props_cache[cache_key] = dict(result)
            return result


        def _get_perry_identity(self, identifier: str) -> Dict[str, str]:
            """Return identity-only metadata from an unambiguous Perry match."""
            library = self._get_perry_library()
            if library is None:
                return {}
            try:
                entry = library.get(identifier, expand_identity=False)
            except Exception:
                return {}
            if not entry:
                return {}
            formula = entry.get('formula') or next(
                (value for value in entry.get('formulas', []) if value),
                '',
            )
            return {
                'CAS': entry.get('cas') or '',
                'name': entry.get('name') or '',
                'formula': formula,
                'symbol': formula or entry.get('name') or '',
            }


        def _resolve_smiles_result(
            self,
            identifier: str,
            props: Optional[Dict[str, Any]],
            allow_online: bool = True,
        ) -> Optional[PropertyResolutionResult]:
            """Resolve SMILES through the shared database structure pipeline."""
            allow_online = self._props_allow_online(props, allow_online)
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..chemical_properties import ChemicalDatabase
                else:
                    from chemical_properties import ChemicalDatabase
                database = ChemicalDatabase(enable_online=allow_online)
                info = database.resolve_smiles_info(
                    identifier,
                    fetch_online=allow_online,
                    props=props,
                    candidates=self._identifier_candidates(identifier, props),
                )
            except Exception:
                return None
            if info is None:
                return None
            return PropertyResolutionResult(
                value=info.smiles,
                source=info.source,
                method=info.method,
                quality=info.quality,
                notes=info.notes,
            )


        def resolve_molecular_weight(
            self,
            identifier: str,
            props: Optional[Dict[str, Any]] = None,
            allow_online: bool = True,
        ) -> PropertyResolutionResult:
            """Resolve MW from a supplied value, resolved structure, or known formula."""
            props = self._coerce_props(identifier, props, allow_online=allow_online)
            provided = self._source_result_for_value(props, 'MW', units='g/mol')
            try:
                provided_mw = float(provided.value) if provided else 0.0
            except (TypeError, ValueError):
                provided_mw = 0.0
            if math.isfinite(provided_mw) and provided_mw > 0.0:
                return provided

            # Missing table values use zero; they must not constrain structure
            # matching or win over the mass calculated from a resolved graph.
            props = dict(props)
            props.pop('MW', None)

            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..compound_identity import looks_like_formula, parse_formula_counts
            else:
                from compound_identity import looks_like_formula, parse_formula_counts
            formula_only = looks_like_formula(identifier) and not any(
                props.get(key) for key in ('name', 'CAS', 'cas', 'smiles')
            )
            # Formula-only requests calculate mass without choosing or caching
            # a representative isomer's structure.
            smiles = None if formula_only else self._resolve_smiles_result(
                identifier, props, allow_online=allow_online,
            )
            if not smiles or not smiles.value:
                # A formula determines mass even when it cannot identify an
                # isomer. This is also the PFD custom-component calculation.
                formula = props.get('formula') or identifier
                counts = parse_formula_counts(formula)
                if counts:
                    from chemicals.elements import periodic_table
                    molecular_weight = sum(
                        float(periodic_table[element].MW) * count
                        for element, count in counts.items()
                    )
                    if math.isfinite(molecular_weight) and molecular_weight > 0.0:
                        return PropertyResolutionResult(
                            value=molecular_weight,
                            source='calculated',
                            method='molecular_weight_from_formula',
                            quality=1.0,
                            notes=f"Calculated formula mass for {formula}; no compound identity inferred",
                        )
                raise PropertyResolutionError(f"Cannot resolve molecular weight for {identifier!r}")
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                mol = Chem.MolFromSmiles(str(smiles.value))
            except Exception as exc:
                raise PropertyResolutionError(
                    f"Cannot parse resolved SMILES for molecular weight of {identifier!r}"
                ) from exc
            if mol is None:
                raise PropertyResolutionError(
                    f"Cannot parse resolved SMILES for molecular weight of {identifier!r}"
                )
            molecular_weight = float(Descriptors.MolWt(mol))
            if not math.isfinite(molecular_weight) or molecular_weight <= 0.0:
                raise PropertyResolutionError(
                    f"Cannot resolve a positive molecular weight for {identifier!r}"
                )
            return PropertyResolutionResult(
                value=molecular_weight,
                source='calculated',
                method='rdkit_molwt_from_smiles',
                quality=smiles.quality,
                notes=f"Calculated from resolved SMILES; SMILES source: {smiles.method}",
            )


        def _get_textbook_entry(self, identifier: str) -> Optional[Dict[str, Any]]:
            """Return extracted textbook properties for a name/formula/symbol."""
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..textbook_properties import get_textbook_property_library
                else:
                    from textbook_properties import get_textbook_property_library
            except ImportError:
                return None
            return get_textbook_property_library().get(identifier)


        def _get_perry_library(self):
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..perry_properties import get_perry_property_library
                else:
                    from perry_properties import get_perry_property_library
            except ImportError:
                return None
            return get_perry_property_library()


        def _get_perry_evaluation(
            self,
            identifier: str,
            props: Optional[Dict[str, Any]],
            evaluator_name: str,
            *args,
            prepared=None,
        ):
            if prepared is not None:
                library, entries = prepared
                evaluator = getattr(library, f'{evaluator_name}_from_entry')
                for entry in entries:
                    result = evaluator(entry, *args)
                    if result is not None:
                        return result
                return None
            library = self._get_perry_library()
            if library is None:
                return None
            for candidate in self._identifier_candidates(identifier, props):
                evaluator = getattr(library, evaluator_name)
                result = evaluator(candidate, *args)
                if result is not None:
                    return result
            return None


        @staticmethod
        def _perry_row_endpoint(row: Dict[str, Any], T: float) -> Optional[float]:
            Tmin = row.get('T_min_K')
            Tmax = row.get('T_max_K')
            if Tmin is None or Tmax is None:
                return None
            try:
                Tmin = float(Tmin)
                Tmax = float(Tmax)
            except (TypeError, ValueError):
                return None
            if T < Tmin - 1e-9:
                return Tmin
            if T > Tmax + 1e-9:
                return Tmax
            return None


        def _get_perry_heat_capacity_bounded(
            self,
            identifier: str,
            props: Optional[Dict[str, Any]],
            T: float,
            phase: str,
            *,
            extrapolation_limit: float = CP_EXTRAPOLATION_LIMIT_K,
        ) -> Optional[tuple[float, Any, str, float]]:
            library = self._get_perry_library()
            if library is None:
                return None
            phase_key = phase.strip().lower().replace('-', '_')
            if phase_key in {'liquid', 'l'}:
                table_keys = (('liquid_heat_capacity', {100}),)
            elif phase_key in {'gas', 'vapor', 'vapour', 'ideal_gas', 'ideal'}:
                table_keys = (
                    ('ideal_gas_heat_capacity_polynomial', None),
                    ('ideal_gas_heat_capacity_hyperbolic', None),
                )
            else:
                return None

            candidates = []
            for candidate in self._identifier_candidates(identifier, props):
                entry = library.get(candidate)
                if not entry:
                    continue
                for table_key, supported in table_keys:
                    for row in entry.get(table_key, []) or []:
                        equation_id = row.get('equation_id')
                        if supported is not None and equation_id not in supported:
                            continue
                        boundary_T = self._perry_row_endpoint(row, T)
                        if boundary_T is None:
                            continue
                        if T < boundary_T:
                            evaluation_T = max(float(T), boundary_T - extrapolation_limit)
                        else:
                            evaluation_T = min(float(T), boundary_T + extrapolation_limit)
                        try:
                            width = float(row.get('T_max_K', boundary_T)) - float(row.get('T_min_K', boundary_T))
                        except (TypeError, ValueError):
                            width = math.inf
                        candidates.append((abs(boundary_T - T), width, boundary_T, evaluation_T, table_key, row))

            candidates.sort(key=lambda item: item[:3])
            for _, _, _, evaluation_T, table_key, row in candidates:
                if table_key == 'liquid_heat_capacity':
                    value = library._eval_liquid_heat_capacity_J_per_kmol_K(row, evaluation_T)
                    method = f"perry_liquid_cp_eq{row.get('equation_id')}"
                elif table_key == 'ideal_gas_heat_capacity_polynomial':
                    value = library._eval_ideal_gas_polynomial_J_per_kmol_K(row, evaluation_T)
                    method = 'perry_ideal_gas_cp_polynomial'
                else:
                    value = library._eval_ideal_gas_hyperbolic_J_per_kmol_K(row, evaluation_T)
                    method = 'perry_ideal_gas_cp_hyperbolic'
                if value is None or value <= 0:
                    continue
                return value / 1000.0, row, method, evaluation_T
            return None


        def _get_perry_viscosity_bounded(
            self,
            identifier: str,
            props: Optional[Dict[str, Any]],
            T: float,
            phase: str,
            *,
            extrapolation_limit: float = 20.0,
            prepared=None,
        ) -> Optional[tuple[float, Any, str, float, float]]:
            """Return a nearby Perry viscosity extrapolation before weaker fallbacks.

            The final tuple is ``value, row, method, distance_K, quality_penalty``.
            """
            if prepared is None:
                prepared = self._prepare_perry_viscosity(identifier, props)
            if prepared is None:
                return None
            library, entries = prepared
            phase_key = phase.strip().lower().replace('-', '_')
            if phase_key in {'liquid', 'l'}:
                table_key = 'liquid_viscosity'
                supported = {100, 101}
                evaluator = library._eval_liquid_viscosity_Pa_s
                method_prefix = 'perry_liquid_viscosity'
            elif phase_key in {'gas', 'vapor', 'vapour', 'ideal_gas', 'ideal'}:
                table_key = 'vapor_viscosity'
                supported = None
                evaluator = library._eval_vapor_viscosity_Pa_s
                method_prefix = 'perry_vapor_viscosity'
            else:
                return None

            candidates = []
            for entry in entries:
                for row in entry.get(table_key, []) or []:
                    equation_id = row.get('equation_id')
                    if supported is not None and equation_id not in supported:
                        continue
                    boundary_T = self._perry_row_endpoint(row, T)
                    if boundary_T is None:
                        continue
                    distance = abs(float(boundary_T) - float(T))
                    if distance > extrapolation_limit + 1e-9:
                        continue
                    try:
                        width = (
                            float(row.get('T_max_K', boundary_T))
                            - float(row.get('T_min_K', boundary_T))
                        )
                    except (TypeError, ValueError):
                        width = math.inf
                    candidates.append((distance, width, boundary_T, row))

            candidates.sort(key=lambda item: item[:3])
            for distance, _, _, row in candidates:
                value = evaluator(row, float(T))
                if value is None or value <= 0:
                    continue
                if phase_key in {'liquid', 'l'}:
                    method = f"{method_prefix}_eq{row.get('equation_id')}"
                else:
                    method = method_prefix
                penalty = 0.03 if distance <= 10.0 + 1e-9 else 0.10
                return value, row, method, distance, penalty
            return None


        def _get_perry_critical_properties(
            self,
            identifier: str,
            props: Optional[Dict[str, Any]] = None,
        ) -> Optional[Dict[str, Any]]:
            library = self._get_perry_library()
            if library is None:
                return None
            for candidate in self._identifier_candidates(identifier, props):
                result = library.critical_properties(candidate)
                if result:
                    return result
            return None


        def _get_perry_formation_properties(
            self,
            identifier: str,
            props: Optional[Dict[str, Any]] = None,
        ) -> Optional[Dict[str, Any]]:
            library = self._get_perry_library()
            if library is None:
                return None
            for candidate in self._identifier_candidates(identifier, props):
                result = library.formation_properties(candidate)
                if result:
                    return result
            return None


        def _get_textbook_antoine(self, identifier: str) -> Optional[AntoineCoefficients]:
            entry = self._get_textbook_entry(identifier)
            if not entry or entry.get('antoine_A') is None:
                return None
            return AntoineCoefficients(
                A=entry['antoine_A'],
                B=entry['antoine_B'],
                C=entry['antoine_C'],
                T_min=entry.get('antoine_Tmin', 200.0),
                T_max=entry.get('antoine_Tmax', 500.0),
                source=entry.get('source_table', 'Smith8 Appendix B'),
                P_units='bar',
            )


        def _get_table_antoine(
            self,
            identifier: str,
            T: Optional[float] = None,
        ) -> Optional[AntoineCoefficients]:
            """Return Antoine coefficients from data/antoine.txt if available."""
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..antoine_properties import get_antoine_table
                else:
                    from antoine_properties import get_antoine_table
            except ImportError:
                return None

            entry = get_antoine_table().get(identifier, T)
            if not entry:
                return None
            return AntoineCoefficients(
                A=entry.A,
                B=entry.B,
                C=entry.C,
                T_min=entry.T_min,
                T_max=entry.T_max,
                source=entry.source,
                P_units='bar',
            )


        @staticmethod
        def _get_provided_antoine(props: Optional[Dict[str, Any]]) -> Optional[AntoineCoefficients]:
            """Return Antoine coefficients carried by a ChemicalProperties dict."""
            props = props or {}
            if (
                props.get('antoine_A') is None
                or props.get('antoine_B') is None
                or props.get('antoine_C') is None
            ):
                return None
            return AntoineCoefficients(
                A=props['antoine_A'],
                B=props['antoine_B'],
                C=props['antoine_C'],
                T_min=props.get('antoine_Tmin', 200.0),
                T_max=props.get('antoine_Tmax', 500.0),
                source=props.get('antoine_source', 'provided properties'),
                P_units='bar',
            )


        @staticmethod
        def _correlation_for(props: Optional[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
            correlations = (props or {}).get('property_correlations') or {}
            if not isinstance(correlations, dict):
                return None
            return correlations.get(key)


        @staticmethod
        def _is_pfd_component_override(props: Optional[Dict[str, Any]], key: str) -> bool:
            source = ((props or {}).get('property_sources') or {}).get(key) or {}
            return source.get('method') == 'pfd_component_override'


        @staticmethod
        def _is_pfd_correlation_override(props: Optional[Dict[str, Any]], key: str) -> bool:
            correlations = (props or {}).get('property_correlations') or {}
            if not isinstance(correlations, dict):
                return False
            correlation = correlations.get(key)
            return isinstance(correlation, dict) and bool(correlation.get('_pfd_override'))


        @staticmethod
        def _props_allow_online(props: Optional[Dict[str, Any]], allow_online: bool = True) -> bool:
            if props is not None and props.get('_allow_online_lookup') is False:
                return False
            return bool(allow_online)


        @staticmethod
        def _props_allow_computation(
            props: Optional[Dict[str, Any]],
            allow_computation: bool = True,
        ) -> bool:
            if props is not None and props.get('_allow_computation') is False:
                return False
            return bool(allow_computation)


        @staticmethod
        def _correlation_coefficients(correlation: Dict[str, Any]) -> Dict[str, float]:
            coefficients = correlation.get('coefficients') or {}
            if isinstance(coefficients, dict):
                return {
                    str(key): float(value)
                    for key, value in coefficients.items()
                    if value is not None
                }
            return {}


        @staticmethod
        def _correlation_in_range(correlation: Dict[str, Any], T: float) -> bool:
            Tmin = correlation.get('Tmin_K')
            Tmax = correlation.get('Tmax_K')
            if Tmin is not None and T < float(Tmin) - 1e-9:
                return False
            if Tmax is not None and T > float(Tmax) + 1e-9:
                return False
            return True


        def _evaluate_correlation(
            self,
            correlation: Dict[str, Any],
            T: float,
            *,
            props: Optional[Dict[str, Any]] = None,
            P: Optional[float] = None,
            enforce_range: bool = True,
            rho_r: Optional[float] = None,
        ) -> Optional[tuple[float, Dict[str, Any]]]:
            """Evaluate one normalized correlation through the shared equation path."""
            props = props or {}
            if enforce_range and not self._correlation_in_range(correlation, T):
                return None

            equation = str(correlation.get('equation', '')).lower()
            coeffs = self._correlation_coefficients(correlation)
            x = (T - 298.15) / 100.0

            try:
                if equation == 'reduced_vapor_pressure':
                    Tc = float(correlation.get('Tc_K') or props.get('Tc'))
                    Pc_pa = float(
                        correlation.get('Pc_Pa')
                        or (
                            float(correlation['Pc_bar']) * 100000.0
                            if correlation.get('Pc_bar') is not None
                            else props.get('Pc') * 100000.0
                        )
                    )
                    Tr = T / Tc
                    tau = 1.0 - Tr
                    exponent = (
                        coeffs.get('A', 0.0) * tau
                        + coeffs.get('B', 0.0) * tau**1.5
                        + coeffs.get('C', 0.0) * tau**3
                        + coeffs.get('D', 0.0) * tau**6
                    ) / Tr
                    return Pc_pa * math.exp(exponent) / 100000.0, correlation

                if equation == 'psat_mercury':
                    # Huber-Laesecke-Friend (2006) mercury saturation pressure:
                    # ln(P/Pc) = (Tc/T) * sum(a_i * tau^b_i), tau = 1 - T/Tc,
                    # with exponents b = (1, 1.89, 2, 8, 8.5, 9). Mercury is the
                    # only species that uses this stiff high-order form.
                    Tc = float(correlation.get('Tc_K') or props.get('Tc'))
                    Pc_pa = float(
                        correlation.get('Pc_Pa')
                        or (
                            float(correlation['Pc_bar']) * 100000.0
                            if correlation.get('Pc_bar') is not None
                            else props.get('Pc') * 100000.0
                        )
                    )
                    Tr = T / Tc
                    tau = 1.0 - Tr
                    if tau <= 0.0:
                        return None
                    series = (
                        coeffs.get('A', 0.0) * tau
                        + coeffs.get('B', 0.0) * tau**1.89
                        + coeffs.get('C', 0.0) * tau**2.0
                        + coeffs.get('D', 0.0) * tau**8.0
                        + coeffs.get('E', 0.0) * tau**8.5
                        + coeffs.get('F', 0.0) * tau**9.0
                    )
                    return Pc_pa * math.exp(series / Tr) / 100000.0, correlation

                if equation in {
                    'canonical_psat',
                    'canonical_psat_af',
                    'canonical_psat_ag',
                    'canonical_psat_ah',
                }:
                    Tc = float(correlation.get('Tc_K') or props.get('Tc'))
                    inverse_power = correlation.get('inverse_power')
                    value = (
                        coeffs.get('A', 0.0)
                        + coeffs.get('B', 0.0) / T
                        + coeffs.get('C', 0.0) * math.log(T)
                        + coeffs.get('D', 0.0) * T
                        + coeffs.get('E', 0.0) * T**2
                        + coeffs.get('F', 0.0) * T**5
                    )
                    if equation in {'canonical_psat_ag', 'canonical_psat_ah'}:
                        value += coeffs.get('G', 0.0) * T**3
                    if equation == 'canonical_psat_ah':
                        value += coeffs.get('H', 0.0) * (
                            (T / Tc) ** int(inverse_power) - 1.0
                        )
                    return math.exp(value), correlation

                if equation == 'reduced_hvap_log':
                    Tc = float(correlation.get('Tc_K') or props.get('Tc'))
                    tau = 1.0 - T / Tc
                    if tau <= 0.0:
                        return None
                    value = math.exp(
                        coeffs.get('A', 0.0)
                        + coeffs.get('B', 0.0) * math.log(tau)
                        + coeffs.get('C', 0.0) * tau
                        + coeffs.get('D', 0.0) * tau * tau
                    )
                    return value, correlation

                if equation == 'dippr_eq101':
                    # DIPPR equation 101: value = exp(A + B/T + C*ln(T) + D*T**E).
                    # The exponent E defaults to 1 (a plain D*T term) and D
                    # defaults to 0, so a three-coefficient A/B/C fit reduces to
                    # the Arrhenius-with-log form. General log-property correlation
                    # used for e.g. liquid viscosity ('mul', Pa*s) or vapor
                    # pressure; returns the property in its stored units.
                    exponent_E = coeffs.get('E', 1.0)
                    value = math.exp(
                        coeffs.get('A', 0.0)
                        + coeffs.get('B', 0.0) / T
                        + coeffs.get('C', 0.0) * math.log(T)
                        + coeffs.get('D', 0.0) * T ** exponent_E
                    )
                    return value, correlation

                if equation == 'dippr_eq100':
                    # DIPPR equation 100: an ordinary polynomial in absolute
                    # temperature, distinct from pfdsim's centered poly_x form.
                    value = 0.0
                    for power, name in enumerate(('A', 'B', 'C', 'D', 'E')):
                        value += coeffs.get(name, 0.0) * T**power
                    return value, correlation

                if equation == 'dippr_eq102':
                    # DIPPR equation 102: dilute-gas transport correlation.
                    value = coeffs.get('A', 0.0) * T ** coeffs.get('B', 0.0)
                    value /= (
                        1.0
                        + coeffs.get('C', 0.0) / T
                        + coeffs.get('D', 0.0) / T**2
                    )
                    return value, correlation

                if equation in {'dippr_eq106', 'eq106'}:
                    Tc = float(
                        correlation.get('Tc_K')
                        or coeffs.get('Tc')
                        or props.get('Tc')
                    )
                    tau = 1.0 - T / Tc
                    if tau <= 0.0:
                        return None
                    Tr = T / Tc
                    exponent = (
                        coeffs.get('B', 0.0)
                        + coeffs.get('C', 0.0) * Tr
                        + coeffs.get('D', 0.0) * Tr * Tr
                        + coeffs.get('E', 0.0) * Tr**3
                    )
                    return coeffs.get('A', 0.0) * tau**exponent, correlation

                if equation == 'viscosity_exp_rhor':
                    if rho_r is None or rho_r < 0.0:
                        return None
                    dilute = math.exp(
                        coeffs.get('A', 0.0)
                        + coeffs.get('B', 0.0) / T
                        + coeffs.get('C', 0.0) * math.log(T)
                        + coeffs.get('D', 0.0) * T
                    )
                    density_polynomial = (
                        rho_r
                        + (coeffs.get('x', coeffs.get('X', 0.0))) * rho_r**2
                        + (coeffs.get('y', coeffs.get('Y', 0.0))) * rho_r**3
                    )
                    pressure_increment = math.exp(
                        coeffs.get('E', 0.0) + coeffs.get('F', 0.0) / T
                    ) * density_polynomial
                    return dilute + pressure_increment, correlation

                if equation == 'poly_x':
                    value = 0.0
                    for power, name in enumerate(('A', 'B', 'C', 'D', 'E', 'F')):
                        value += coeffs.get(name, 0.0) * x**power
                    return value, correlation

                if equation == 'exp_poly_x':
                    value = 0.0
                    for power, name in enumerate(('A', 'B', 'C', 'D', 'E', 'F')):
                        value += coeffs.get(name, 0.0) * x**power
                    return math.exp(value), correlation

                if equation == 'poly_tp':
                    if P is None:
                        return None
                    pressure = float(P)
                    Pmin = correlation.get('Pmin_bar')
                    Pmax = correlation.get('Pmax_bar')
                    if Pmin is not None and pressure < float(Pmin):
                        return None
                    if Pmax is not None and pressure > float(Pmax):
                        return None
                    p_ref = float(correlation.get('P_ref_bar', 0.0))
                    p = pressure - p_ref
                    return (
                        coeffs.get('A', 0.0)
                        + coeffs.get('B', 0.0) * x
                        + coeffs.get('C', 0.0) * p
                        + coeffs.get('D', 0.0) * x * x
                        + coeffs.get('E', 0.0) * x * p
                        + coeffs.get('F', 0.0) * p * p
                    ), correlation

                if equation == 'exp_poly_tp':
                    if P is None:
                        return None
                    pressure = float(P)
                    Pmin = correlation.get('Pmin_bar')
                    Pmax = correlation.get('Pmax_bar')
                    if Pmin is not None and pressure < float(Pmin):
                        return None
                    if Pmax is not None and pressure > float(Pmax):
                        return None
                    p_ref = float(correlation.get('P_ref_bar', 0.0))
                    p = pressure - p_ref
                    value = (
                        coeffs.get('A', 0.0)
                        + coeffs.get('B', 0.0) * x
                        + coeffs.get('C', 0.0) * p
                        + coeffs.get('D', 0.0) * x * x
                        + coeffs.get('E', 0.0) * x * p
                        + coeffs.get('F', 0.0) * p * p
                    )
                    return math.exp(value), correlation

                if equation == 'shomate':
                    t = T / 1000.0
                    value = (
                        coeffs.get('A', 0.0)
                        + coeffs.get('B', 0.0) * t
                        + coeffs.get('C', 0.0) * t * t
                        + coeffs.get('D', 0.0) * t**3
                        + coeffs.get('E', 0.0) / (t * t)
                    )
                    return value, correlation
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None
            return None


        def _evaluate_provided_correlation(
            self,
            props: Dict[str, Any],
            key: str,
            T: float,
            *,
            P: Optional[float] = None,
            enforce_range: bool = True,
            rho_r: Optional[float] = None,
        ) -> Optional[tuple[float, Dict[str, Any]]]:
            correlation = self._correlation_for(props, key)
            if not correlation:
                return None
            return self._evaluate_correlation(
                correlation,
                T,
                props=props,
                P=P,
                enforce_range=enforce_range,
                rho_r=rho_r,
            )


        def _evaluate_provided_correlation_bounded(
            self,
            props: Dict[str, Any],
            key: str,
            T: float,
            *,
            extrapolation_limit: float = CP_EXTRAPOLATION_LIMIT_K,
        ) -> Optional[tuple[float, Dict[str, Any], float]]:
            correlation = self._correlation_for(props, key)
            if not correlation or self._correlation_in_range(correlation, T):
                return None
            Tmin = correlation.get('Tmin_K')
            Tmax = correlation.get('Tmax_K')
            if Tmin is None or Tmax is None:
                return None
            try:
                Tmin = float(Tmin)
                Tmax = float(Tmax)
            except (TypeError, ValueError):
                return None
            if T < Tmin:
                evaluation_T = max(float(T), Tmin - extrapolation_limit)
            else:
                evaluation_T = min(float(T), Tmax + extrapolation_limit)
            evaluated = self._evaluate_provided_correlation(
                props,
                key,
                evaluation_T,
                enforce_range=False,
            )
            if not evaluated:
                return None
            value, used_correlation = evaluated
            return value, used_correlation, evaluation_T


        @staticmethod
        def _provided_correlation_result(
            value: float,
            correlation: Dict[str, Any],
            method: str,
            notes_prefix: str = '',
            default_quality: float = 0.96,
        ) -> PropertyResolutionResult:
            notes = notes_prefix
            Tmin = correlation.get('Tmin_K')
            Tmax = correlation.get('Tmax_K')
            if Tmin is not None and Tmax is not None:
                notes = f"{notes}; " if notes else ''
                notes += f"range {Tmin:g}-{Tmax:g} K"
            quality_note = correlation.get('quality_note')
            legacy_quality = correlation.get('quality')
            if quality_note is None and isinstance(legacy_quality, str):
                quality_note = legacy_quality
            if quality_note:
                notes += f"; {quality_note}" if notes else str(quality_note)
            try:
                quality = float(correlation.get('quality', default_quality))
            except (TypeError, ValueError):
                quality = default_quality
            return PropertyResolutionResult(
                value=value,
                source='provided',
                method=method,
                quality=quality,
                notes=notes,
            )


        @staticmethod
        def _positive_number(value: Any) -> Optional[float]:
            """Return a positive finite float, or ``None`` for invalid input."""
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            return result if math.isfinite(result) and result > 0.0 else None


        @staticmethod
        def _clamp_quality(value: Optional[float], default: float = 1.0) -> float:
            try:
                quality = default if value is None else float(value)
            except (TypeError, ValueError):
                quality = default
            return max(0.0, min(1.0, quality))


        @classmethod
        def _meta_quality(cls, source: Optional[Dict[str, Any]], default: float = 1.0) -> float:
            source = source or {}
            return cls._clamp_quality(source.get('quality'), default)


        @classmethod
        def _result_quality(cls, result: Optional[PropertyResolutionResult], default: float = 0.0) -> float:
            if result is None:
                return default
            return cls._clamp_quality(getattr(result, 'quality', None), default)


        @staticmethod
        def _source_meta_is_soft(source: Optional[Dict[str, Any]]) -> bool:
            source = source or {}
            if source.get('replaceable') is False:
                return False
            source_name = str(source.get('source') or '').lower()
            if source_name in {'missing', 'estimated'}:
                return True
            quality = PropertyResolverBase._meta_quality(source, 1.0)
            if quality < SOFT_PROPERTY_QUALITY_THRESHOLD:
                return True
            if source_name == 'exact':
                return False
            method = str(source.get('method') or '').lower()
            return method in {
                'mw_boiling_point',
                'mw_correlation',
                'formula_hbd_boiling_point',
                'formula_no_hbd_boiling_point',
                'nannoolal_tb',
                'nannoolal_tc',
                'nannoolal_pc',
                'nannoolal_vc',
                'nannoolal_hvap_pr',
                'corresponding_states_hvap',
                'guldberg_rule',
                'lydersen_style',
                'atom_count_ring_tb_pc',
                'atom_count_large_ring_vc',
                'trouton',
                'trouton_watson',
                'liquid_gf_plus_standard_vaporization_gibbs',
            }


        @classmethod
        def _result_is_soft(cls, result: Optional[PropertyResolutionResult]) -> bool:
            if not result or result.value is None:
                return True
            source_name = str(result.source or '').lower()
            if source_name in {'missing', 'estimated'}:
                return True
            if cls._result_quality(result, 0.0) < SOFT_PROPERTY_QUALITY_THRESHOLD:
                return True
            if source_name == 'exact':
                return False
            method = str(result.method or '').lower()
            return method in {
                'mw_boiling_point',
                'mw_correlation',
                'formula_hbd_boiling_point',
                'formula_no_hbd_boiling_point',
                'nannoolal_tb',
                'nannoolal_tc',
                'nannoolal_pc',
                'nannoolal_vc',
                'nannoolal_hvap_pr',
                'corresponding_states_hvap',
                'guldberg_rule',
                'lydersen_style',
                'atom_count_ring_tb_pc',
                'atom_count_large_ring_vc',
                'trouton',
                'trouton_watson',
                'liquid_gf_plus_standard_vaporization_gibbs',
            }


        @classmethod
        def _result_is_real(cls, result: Optional[PropertyResolutionResult]) -> bool:
            return bool(result and result.value is not None and not cls._result_is_soft(result))


        @classmethod
        def _combine_quality(
            cls,
            inputs: List[Optional[PropertyResolutionResult]],
            method_factor: float = 1.0,
            exact_formula: bool = False,
        ) -> float:
            usable = [item for item in inputs if item is not None and item.value is not None]
            if not usable:
                return cls._clamp_quality(method_factor)
            qualities = [cls._result_quality(item, 0.0) for item in usable]
            if exact_formula and all(cls._result_is_real(item) for item in usable):
                return min(qualities)
            return cls._clamp_quality(min(qualities) * cls._clamp_quality(method_factor))


        @classmethod
        def _derived_source(
            cls,
            inputs: List[Optional[PropertyResolutionResult]],
            exact_formula: bool = False,
        ) -> str:
            if exact_formula and inputs and all(cls._result_is_real(item) for item in inputs):
                return 'exact'
            return 'calculated'


        @classmethod
        def _source_result_for_value(
            cls,
            props: Dict[str, Any],
            key: str,
            value: Any = None,
            units: str = "",
            default_source: str = 'provided',
            default_method: str = 'direct',
            default_quality: float = 1.0,
        ) -> Optional[PropertyResolutionResult]:
            if value is None:
                value = (props or {}).get(key)
            if value is None:
                return None
            source = ((props or {}).get('property_sources') or {}).get(key) or {}
            source_name = source.get('source') or default_source
            method = source.get('method') or default_method
            notes = source.get('notes') or (f"units {units}" if units else "")
            return PropertyResolutionResult(
                value=value,
                source=source_name,
                method=method,
                quality=cls._meta_quality(source, default_quality),
                notes=notes,
            )
