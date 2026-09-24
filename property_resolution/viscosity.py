from .common import *
from .coolprop import coolprop_module
from .online_viscosity import (
    PubChemViscosityFetchError,
    PubChemViscosityFetcher,
    PubChemViscosityResult,
)
from .organic_classification import hydrogen_bond_donor_profile
from .viscosity_kernel import ViscosityKernel
from collections import Counter
from copy import deepcopy


COOLPROP_VISCOSITY_QUALITY = 0.99
PROVIDED_VISCOSITY_QUALITY = 0.98
COOLPROP_STANDARD_PRESSURE_PA = 101325.0
YOON_THODOS_HYDROCARBON_FACTOR = 0.88
YOON_THODOS_SPARSE_HETEROATOM_FACTOR = 0.88
YOON_THODOS_HETEROATOM_FACTOR = 0.75
YOON_THODOS_SMALL_MOLECULE_FACTOR = 0.50
REICHENBERG_SPARSE_HETEROATOM_FACTOR = 0.91
REICHENBERG_TERMINAL_ALKYNE_FACTOR = 0.91
REICHENBERG_POLAR_ORGANIC_FACTOR = 0.86
REICHENBERG_INORGANIC_FACTOR = 0.80
REICHENBERG_MINIMUM_HEAVY_ATOMS = 3
JOSSI_UNIFAC_CLASSIFICATION_FACTOR = 0.95
JOSSI_HBOND_CLASSIFICATION_FACTOR = 0.80
JOSSI_DEFAULT_POLAR_CLASSIFICATION_FACTOR = 0.75
JOSSI_MAX_REDUCED_DENSITY = 2.6
ONLINE_VISCOSITY_CACHE_VERSION = 2
ONLINE_VISCOSITY_MAXIMUM_EXTRAPOLATION_K = 100.0
ONLINE_VISCOSITY_MAXIMUM_ANCHOR_PRESSURE_BAR = 5.0
ONLINE_VISCOSITY_HIGH_QUALITY_DENSITY = 0.95
ONLINE_VISCOSITY_SAME_TEMPERATURE_TOLERANCE_K = 0.5
ONLINE_VISCOSITY_MINIMUM_FIT_SPAN_K = 10.0
ONLINE_VISCOSITY_FULL_QUALITY_FIT_SPAN_K = 20.0


class ViscosityMixin:
        @staticmethod
        def _viscosity_kernel_props_fingerprint(props: Optional[Dict[str, Any]]) -> str:
            if props is None:
                return '<database-properties>'
            encoded = json.dumps(
                props,
                sort_keys=True,
                separators=(',', ':'),
                ensure_ascii=False,
                default=str,
            )
            return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


        def resolve_viscosity(
            self,
            symbol: str,
            T: float,
            phase: str,
            props: Dict[str, Any] = None,
            *,
            P: Optional[float] = None,
            rho_molar: Optional[float] = None,
            allow_online: bool = True,
        ) -> PropertyResolutionResult:
            """Resolve viscosity through a retained executable kernel."""
            kernel = self.resolve_viscosity_kernel(
                symbol,
                phase,
                props,
                allow_online=allow_online,
            )
            return kernel.evaluate(T, P=P, rho_molar=rho_molar)


        def resolve_viscosity_kernel(
            self,
            symbol: str,
            phase: str,
            props: Dict[str, Any] = None,
            *,
            allow_online: bool = True,
        ) -> ViscosityKernel:
            """Resolve and retain one executable viscosity plan."""
            phase_key = phase.strip().lower().replace('-', '_')
            phase_key = (
                'vapor'
                if phase_key in {'gas', 'vapor', 'vapour', 'ideal_gas', 'ideal'}
                else 'liquid'
                if phase_key in {'liquid', 'l'}
                else phase_key
            )
            cache_key = (
                str(symbol),
                phase_key,
                self._viscosity_kernel_props_fingerprint(props),
                bool(allow_online),
            )
            cached = self._viscosity_kernel_cache.get(cache_key)
            if cached is not None:
                return cached

            prepared_props = self._coerce_props(
                symbol,
                props,
                allow_online=allow_online,
            )
            allow_online = self._props_allow_online(prepared_props, allow_online)
            prepared_props = {
                **deepcopy(prepared_props),
                '_allow_online_lookup': allow_online,
            }
            correlation_key = (
                'mug'
                if phase_key == 'vapor'
                else 'mul'
            )
            coolprop_reference = None
            use_dynamic_coolprop = (
                getattr(self._coolprop_viscosity, '__func__', None)
                is not ViscosityMixin._coolprop_viscosity
            )
            if (
                not use_dynamic_coolprop
                and not self._is_pfd_correlation_override(
                    prepared_props,
                    correlation_key,
                )
            ):
                coolprop_reference = self._coolprop_reference(
                    symbol,
                    prepared_props,
                )
            perry_prepared = self._prepare_perry_viscosity(symbol, prepared_props)

            def evaluate(T, P, rho_molar):
                return self._resolve_prepared_viscosity(
                    symbol,
                    prepared_props,
                    T,
                    phase_key,
                    P,
                    rho_molar,
                    allow_online,
                    coolprop_reference,
                    perry_prepared,
                    use_dynamic_coolprop,
                )

            kernel = ViscosityKernel(phase=phase_key, evaluator=evaluate)
            self._viscosity_kernel_cache[cache_key] = kernel
            return kernel


        def _resolve_prepared_viscosity(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            phase_key: str,
            P: Optional[float],
            rho_molar: Optional[float],
            allow_online: bool,
            coolprop_reference,
            perry_prepared,
            use_dynamic_coolprop: bool,
        ) -> PropertyResolutionResult:
            """Evaluate an already-normalized viscosity provider hierarchy."""
            correlation_key = 'mug' if phase_key == 'vapor' else 'mul'

            if use_dynamic_coolprop and not self._is_pfd_correlation_override(
                props, correlation_key,
            ):
                coolprop_viscosity = self._coolprop_viscosity(
                    symbol,
                    props,
                    T,
                    phase_key,
                    P=P,
                )
                if coolprop_viscosity:
                    return coolprop_viscosity
            elif coolprop_reference is not None:
                coolprop_viscosity = self._coolprop_viscosity_from_reference(
                    coolprop_reference,
                    T,
                    phase_key,
                    P=P,
                )
                if coolprop_viscosity:
                    return coolprop_viscosity

            correlation = self._correlation_for(props, correlation_key)
            equation = str((correlation or {}).get('equation', '')).lower()
            pressure_specific_fit = equation in {'poly_tp', 'exp_poly_tp'}
            pressure_fit = bool(
                correlation
                and equation == 'viscosity_exp_rhor'
            )
            rho_r = None
            critical_density_result = None
            critical_density_note = ''
            if pressure_fit:
                if P is None:
                    rho_r = 0.0
                elif rho_molar is not None:
                    density_state = self._viscosity_critical_density(symbol, props)
                    if density_state is not None:
                        critical_density_result, _, critical_density_note = density_state
                        rho_r = rho_molar / float(critical_density_result.value)
            provided_fit = None
            if not pressure_fit or rho_r is not None:
                provided_fit = self._evaluate_provided_correlation(
                    props,
                    correlation_key,
                    T,
                    P=P,
                    rho_r=rho_r,
                )
            if provided_fit:
                value, correlation = provided_fit
                if correlation_key == 'mug':
                    value *= 1.0e-6
                if value > 0:
                    baseline = self._provided_correlation_result(
                        value,
                        correlation,
                        (
                            'provided_pressure_viscosity_fit'
                            if pressure_fit
                            else 'provided_viscosity_fit'
                        ),
                        (
                            f'dynamic viscosity in Pa*s; reduced-density fit at rho_r={rho_r:.4g}'
                            if pressure_fit and P is not None
                            else 'dynamic viscosity in Pa*s; dilute-density term with P unspecified'
                            if pressure_fit
                            else 'dynamic viscosity in Pa*s'
                        ),
                        default_quality=PROVIDED_VISCOSITY_QUALITY,
                    )
                    if pressure_fit:
                        if critical_density_result is not None:
                            baseline = PropertyResolutionResult(
                                value=baseline.value,
                                source=baseline.source,
                                method=baseline.method,
                                quality=self._combine_quality(
                                    [baseline, critical_density_result],
                                ),
                                notes=f'{baseline.notes}; {critical_density_note}',
                            )
                        return baseline
                    if pressure_specific_fit:
                        return baseline
                    return self._apply_viscosity_pressure_correction(
                        symbol, props, T, phase_key, baseline, P, rho_molar,
                    )

            perry_result = self._evaluate_prepared_perry_viscosity(
                symbol, props, T, phase_key, perry_prepared,
            )
            if perry_result is not None:
                return self._apply_viscosity_pressure_correction(
                    symbol, props, T, phase_key, perry_result, P, rho_molar,
                )

            estimated_vapor_viscosity = self._estimated_vapor_viscosity(
                symbol,
                props,
                T,
                phase_key,
            )
            if estimated_vapor_viscosity:
                return self._apply_viscosity_pressure_correction(
                    symbol, props, T, phase_key, estimated_vapor_viscosity, P, rho_molar,
                )

            online_viscosity = self._online_liquid_viscosity(
                symbol,
                props,
                T,
                phase_key,
                allow_online=allow_online,
            )
            if online_viscosity:
                return self._apply_viscosity_pressure_correction(
                    symbol, props, T, phase_key, online_viscosity, P, rho_molar,
                )

            hsu_viscosity = self._hsu_liquid_viscosity(symbol, props, T, phase_key)
            if hsu_viscosity:
                return self._apply_viscosity_pressure_correction(
                    symbol, props, T, phase_key, hsu_viscosity, P, rho_molar,
                )

            nannoolal_viscosity = self._nannoolal_predictive_liquid_viscosity(
                symbol,
                props,
                T,
                phase_key,
                allow_online=allow_online,
            )
            if nannoolal_viscosity:
                return self._apply_viscosity_pressure_correction(
                    symbol, props, T, phase_key, nannoolal_viscosity, P, rho_molar,
                )

            raise PropertyResolutionError(
                f"Cannot determine {phase_key} viscosity for '{symbol}' at T={T:.1f}K."
            )


        def _prepare_perry_viscosity(
            self,
            symbol: str,
            props: Dict[str, Any],
        ):
            library = self._get_perry_library()
            if library is None:
                return None
            entries = []
            seen = set()
            for candidate in self._identifier_candidates(symbol, props):
                entry = library.get(candidate)
                if entry is None or id(entry) in seen:
                    continue
                seen.add(id(entry))
                entries.append(entry)
            return library, tuple(entries)


        def _evaluate_prepared_perry_viscosity(
            self, symbol, props, T, phase_key, prepared,
        ):
            perry = self._get_perry_evaluation(
                symbol,
                props,
                'viscosity_Pa_s',
                T,
                phase_key,
                prepared=prepared,
            )
            if perry is not None:
                return PropertyResolutionResult(
                    value=perry.value,
                    source='local',
                    method=perry.method,
                    quality=0.97,
                    notes=(
                        f"{perry.source}; 1 atm correlation basis; "
                        f"units {perry.units}"
                    ),
                )
            bounded = self._get_perry_viscosity_bounded(
                symbol,
                props,
                T,
                phase_key,
                prepared=prepared,
            )
            if bounded is None:
                return None
            value, row, method, distance, penalty = bounded
            return PropertyResolutionResult(
                value=value,
                source='local',
                method=f'{method}_bounded_extrapolation',
                quality=max(0.0, 0.97 - penalty),
                notes=(
                    f"Perry 9th Table {row.get('source_table')}; "
                    "1 atm correlation basis; units Pa*s; "
                    f"extrapolated {distance:.3g} K beyond tabulated range; "
                    f"quality penalty {penalty:.2f}"
                ),
            )

        def _viscosity_critical_density(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[tuple[PropertyResolutionResult, Dict[str, PropertyResolutionResult], str]]:
            try:
                critical = self.resolve_critical_properties(
                    symbol,
                    props,
                    allow_online=self._props_allow_online(props),
                    allow_estimation=True,
                )
            except Exception:
                return None
            tc_result = critical.get('Tc') if critical else None
            pc_result = critical.get('Pc') if critical else None
            zc_result = critical.get('Zc') if critical else None
            vc_result = critical.get('Vc') if critical else None

            if (
                tc_result is not None and tc_result.value is not None
                and pc_result is not None and pc_result.value is not None
                and zc_result is not None and zc_result.value is not None
            ):
                try:
                    tc = float(tc_result.value)
                    pc_bar = float(pc_result.value)
                    zc = float(zc_result.value)
                    rho_c = pc_bar / (zc * (R / 100.0) * tc)
                except (TypeError, ValueError, ZeroDivisionError):
                    rho_c = None
                if rho_c is not None and rho_c > 0.0 and math.isfinite(rho_c):
                    inputs = [tc_result, pc_result, zc_result]
                    result = PropertyResolutionResult(
                        value=rho_c,
                        source=self._derived_source(inputs, exact_formula=True),
                        method='critical_density_from_zc',
                        quality=self._combine_quality(inputs, exact_formula=True),
                        notes=f'Critical molar density from Zc={zc:g}; units kmol/m^3',
                    )
                    return result, critical, f'critical density from Zc={zc:g}'

            if vc_result is not None and vc_result.value is not None:
                try:
                    vc = float(vc_result.value)
                    rho_c = 1000.0 / vc
                except (TypeError, ValueError, ZeroDivisionError):
                    rho_c = None
                if rho_c is not None and rho_c > 0.0 and math.isfinite(rho_c):
                    result = PropertyResolutionResult(
                        value=rho_c,
                        source=self._derived_source([vc_result], exact_formula=True),
                        method='critical_density_from_vc',
                        quality=self._combine_quality([vc_result], exact_formula=True),
                        notes=f'Critical molar density from Vc={vc:g} cm^3/mol; units kmol/m^3',
                    )
                    return result, critical, f'critical density from Vc={vc:g} cm^3/mol'
            return None


        @staticmethod
        def _viscosity_smiles(props: Dict[str, Any]) -> Optional[str]:
            value = next(
                (
                    props.get(key) for key in (
                        'smiles', 'SMILES', 'canonical_smiles', 'CanonicalSMILES',
                        'connectivity_smiles', 'ConnectivitySMILES',
                    )
                    if props.get(key)
                ),
                None,
            )
            return str(value).strip() if value else None


        def _fetch_pubchem_viscosity(
            self,
            identifier: str,
        ) -> Optional[PubChemViscosityResult]:
            cache_key = (
                f'viscosity_pubchem_v{ONLINE_VISCOSITY_CACHE_VERSION}_'
                f'{identifier}'
            )
            cached = self._get_cache(cache_key)
            if cached:
                if self._is_missing_cache(cached):
                    return None
                try:
                    return PubChemViscosityResult.from_dict(cached)
                except (KeyError, TypeError, ValueError):
                    pass
            try:
                cid = self._get_pubchem_cid(identifier)
                if not cid:
                    self._set_missing_cache(cache_key)
                    return None
                result = PubChemViscosityFetcher().fetch(cid)
            except PubChemViscosityFetchError as exc:
                raise LookupError(
                    f"Transient PubChem viscosity lookup failure for {identifier!r}"
                ) from exc
            if result is None:
                self._set_missing_cache(cache_key)
                return None
            self._set_cache(cache_key, result.to_dict())
            return result


        def _fetch_viscosity_online(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[PubChemViscosityResult]:
            transient_failure = False
            for identifier in self._identifier_candidates(symbol, props):
                try:
                    result = self._fetch_pubchem_viscosity(str(identifier))
                except LookupError:
                    transient_failure = True
                    continue
                if result is not None:
                    return result
            if transient_failure:
                raise LookupError(
                    f"Transient online viscosity lookup failure for {symbol!r}"
                )
            return None


        def _viscosity_phenol_status(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> tuple[bool, Optional[PropertyResolutionResult]]:
            smiles_result = None
            smiles = self._viscosity_smiles(props)
            if not smiles:
                smiles_result = self._resolve_smiles_result(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
                smiles = (
                    str(smiles_result.value).strip()
                    if smiles_result and smiles_result.value else None
                )
            profile = hydrogen_bond_donor_profile(smiles) if smiles else None
            if profile is not None:
                return bool(profile.phenol_oh), smiles_result
            identity = ' '.join(
                str(value).lower()
                for value in (
                    symbol,
                    props.get('name'),
                )
                if value
            )
            return bool(re.search(
                r'\b(?:phenol|cresol|xylenol|naphthol|hydroxybenzene)\b',
                identity,
            )), smiles_result


        @staticmethod
        def _online_viscosity_point_is_phase_compatible(
            point,
            props: Dict[str, Any],
        ) -> bool:
            temperature = float(point.temperature_K)
            pressure = point.pressure_bar
            if (
                pressure is not None
                and pressure > ONLINE_VISCOSITY_MAXIMUM_ANCHOR_PRESSURE_BAR
            ):
                return False

            def finite_property(name: str) -> Optional[float]:
                try:
                    value = float(props.get(name))
                except (TypeError, ValueError):
                    return None
                return value if value > 0.0 and math.isfinite(value) else None

            melting = finite_property('Tm')
            critical = finite_property('Tc')
            boiling = finite_property('Tb')
            if melting is not None and temperature <= melting:
                return False
            if critical is not None and temperature >= critical:
                return False
            return not (
                point.phase_basis != 'liquid'
                and pressure is None
                and boiling is not None
                and temperature > boiling + 1.0
            )


        def _kinematic_viscosity_dynamic_point(
            self,
            symbol: str,
            props: Dict[str, Any],
            point,
            *,
            allow_online: bool,
        ) -> Optional[Dict[str, Any]]:
            try:
                density = self.resolve_liquid_molar_density(
                    symbol,
                    float(point.temperature_K),
                    props,
                )
                mw = self.resolve_molecular_weight(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
                mass_density = float(density.value) * float(mw.value)
                viscosity = (
                    float(point.kinematic_viscosity_m2_s) * mass_density
                )
            except Exception:
                return None
            if viscosity <= 0.0 or not math.isfinite(viscosity):
                return None
            density_quality = self._combine_quality(
                [density, mw],
                exact_formula=True,
            )
            density_penalty = max(
                0.0,
                ONLINE_VISCOSITY_HIGH_QUALITY_DENSITY - density_quality,
            )
            return {
                'T_K': float(point.temperature_K),
                'mu_Pa_s': viscosity,
                'density_quality': density_quality,
                'density_penalty': density_penalty,
                'kind': 'kinematic_converted',
                'reference': point.reference,
                'raw': point.raw,
                'notes': (
                    f'kinematic viscosity converted with liquid density '
                    f'{mass_density:g} kg/m^3 from {density.source}/'
                    f'{density.method} and MW from {mw.source}/{mw.method}; '
                    f'density quality {density_quality:.3f}'
                ),
            }


        @classmethod
        def _consolidate_online_viscosity_points(
            cls,
            points: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            clusters: List[List[Dict[str, Any]]] = []
            for point in sorted(points, key=lambda item: float(item['T_K'])):
                if (
                    not clusters
                    or abs(
                        float(point['T_K'])
                        - sum(float(item['T_K']) for item in clusters[-1])
                        / len(clusters[-1])
                    ) > ONLINE_VISCOSITY_SAME_TEMPERATURE_TOLERANCE_K
                ):
                    clusters.append([point])
                else:
                    clusters[-1].append(point)

            consolidated = []
            for cluster in clusters:
                logarithms = sorted(
                    math.log(float(point['mu_Pa_s'])) for point in cluster
                )
                if (
                    len(logarithms) == 2
                    and math.exp(logarithms[-1] - logarithms[0]) > 1.25
                ):
                    continue
                median_log = logarithms[len(logarithms) // 2]
                if len(logarithms) % 2 == 0:
                    median_log = 0.5 * (
                        logarithms[len(logarithms) // 2 - 1]
                        + logarithms[len(logarithms) // 2]
                    )
                retained = (
                    [
                        point for point in cluster
                        if abs(math.log(float(point['mu_Pa_s'])) - median_log)
                        <= math.log(1.25)
                    ]
                    if len(cluster) >= 3 else cluster
                )
                retained = retained or cluster
                consolidated.append(cls._average_online_viscosity_points(retained))
            return consolidated


        @staticmethod
        def _average_online_viscosity_points(
            retained: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
            weights = [
                max(0.05, 1.0 - float(point['density_penalty'])) ** 2
                for point in retained
            ]
            weight_sum = sum(weights)
            mean_temperature = sum(
                weight * float(point['T_K'])
                for point, weight in zip(retained, weights)
            ) / weight_sum
            mean_log = sum(
                weight * math.log(float(point['mu_Pa_s']))
                for point, weight in zip(retained, weights)
            ) / weight_sum
            return {
                'T_K': mean_temperature,
                'mu_Pa_s': math.exp(mean_log),
                'density_quality': max(
                    float(point['density_quality']) for point in retained
                ),
                'density_penalty': min(
                    float(point['density_penalty']) for point in retained
                ),
                'kind': '+'.join(sorted({
                    str(point['kind']) for point in retained
                })),
                'reference': '; '.join(dict.fromkeys(
                    str(point.get('reference') or '')
                    for point in retained
                    if point.get('reference')
                )),
                'raw': ' | '.join(dict.fromkeys(
                    str(point.get('raw') or '') for point in retained
                )),
                'notes': '; '.join(dict.fromkeys(
                    str(point.get('notes') or '')
                    for point in retained
                    if point.get('notes')
                )),
            }


        def _online_liquid_viscosity_points(
            self,
            symbol: str,
            props: Dict[str, Any],
            payload: PubChemViscosityResult,
            *,
            allow_online: bool,
        ) -> List[Dict[str, Any]]:
            dynamic = [
                {
                    'T_K': float(point.temperature_K),
                    'mu_Pa_s': float(point.viscosity_Pa_s),
                    'density_quality': 1.0,
                    'density_penalty': 0.0,
                    'kind': 'dynamic',
                    'reference': point.reference,
                    'raw': point.raw,
                    'notes': 'direct dynamic-viscosity observation',
                }
                for point in payload.points
                if point.viscosity_Pa_s is not None
                and self._online_viscosity_point_is_phase_compatible(point, props)
            ]
            dynamic = self._consolidate_online_viscosity_points(dynamic)
            converted = []
            for point in payload.kinematic_points:
                if not self._online_viscosity_point_is_phase_compatible(point, props):
                    continue
                candidate = self._kinematic_viscosity_dynamic_point(
                    symbol,
                    props,
                    point,
                    allow_online=allow_online,
                )
                if candidate is not None:
                    converted.append(candidate)

            high_quality = [
                point for point in converted
                if float(point['density_quality'])
                >= ONLINE_VISCOSITY_HIGH_QUALITY_DENSITY
            ]
            lower_quality = [
                point for point in converted
                if float(point['density_quality'])
                < ONLINE_VISCOSITY_HIGH_QUALITY_DENSITY
            ]
            if len(dynamic) >= 2:
                selected_converted = high_quality
            elif len(dynamic) == 1:
                established_temperatures = [
                    float(point['T_K'])
                    for point in dynamic + high_quality
                ]
                has_shape = any(
                    abs(temperature - established_temperatures[0])
                    > ONLINE_VISCOSITY_SAME_TEMPERATURE_TOLERANCE_K
                    for temperature in established_temperatures[1:]
                )
                second = None
                if not has_shape:
                    distinct = [
                        point for point in lower_quality
                        if all(
                            abs(
                                float(point['T_K']) - established_temperature
                            ) > ONLINE_VISCOSITY_SAME_TEMPERATURE_TOLERANCE_K
                            for established_temperature in established_temperatures
                        )
                    ]
                    second = max(
                        distinct,
                        key=lambda point: (
                            float(point['density_quality']),
                            abs(
                                float(point['T_K'])
                                - float(dynamic[0]['T_K'])
                            ),
                        ),
                        default=None,
                    )
                selected_converted = high_quality + ([second] if second else [])
            else:
                selected_converted = high_quality + lower_quality
            return self._consolidate_online_viscosity_points(
                dynamic + selected_converted
            )


        @staticmethod
        def _online_viscosity_fit_inliers(
            points: List[Dict[str, Any]],
        ) -> tuple[List[Dict[str, Any]], int]:
            retained = list(points)
            rejected = 0
            if len(retained) >= 4:
                slopes = []
                for index, left in enumerate(retained):
                    x_left = 1.0 / float(left['T_K'])
                    y_left = math.log(float(left['mu_Pa_s']))
                    for right in retained[index + 1:]:
                        x_right = 1.0 / float(right['T_K'])
                        if math.isclose(x_left, x_right, abs_tol=1.0e-15):
                            continue
                        y_right = math.log(float(right['mu_Pa_s']))
                        slopes.append((y_right - y_left) / (x_right - x_left))
                if slopes:
                    slope = sorted(slopes)[len(slopes) // 2]
                    intercepts = sorted(
                        math.log(float(point['mu_Pa_s']))
                        - slope / float(point['T_K'])
                        for point in retained
                    )
                    intercept = intercepts[len(intercepts) // 2]
                    residuals = [
                        math.log(float(point['mu_Pa_s']))
                        - intercept - slope / float(point['T_K'])
                        for point in retained
                    ]
                    center = sorted(residuals)[len(residuals) // 2]
                    deviations = sorted(abs(value - center) for value in residuals)
                    mad = deviations[len(deviations) // 2]
                    threshold = max(math.log(1.15), 3.5 * 1.4826 * mad)
                    inliers = [
                        point for point, residual in zip(retained, residuals)
                        if abs(residual - center) <= threshold
                    ]
                    if len(inliers) >= 2:
                        rejected = len(retained) - len(inliers)
                        retained = inliers

            return retained, rejected


        @staticmethod
        def _weighted_online_viscosity_fit(
            retained: List[Dict[str, Any]],
        ) -> Optional[tuple[float, float]]:
            if len(retained) < 2:
                return None
            temperatures = [float(point['T_K']) for point in retained]
            if max(temperatures) - min(temperatures) < ONLINE_VISCOSITY_MINIMUM_FIT_SPAN_K:
                return None
            xs = [1.0 / temperature for temperature in temperatures]
            ys = [math.log(float(point['mu_Pa_s'])) for point in retained]
            weights = [
                max(0.05, 1.0 - float(point['density_penalty'])) ** 2
                for point in retained
            ]
            weight_sum = sum(weights)
            x_mean = sum(x * weight for x, weight in zip(xs, weights)) / weight_sum
            y_mean = sum(y * weight for y, weight in zip(ys, weights)) / weight_sum
            denominator = sum(
                weight * (x - x_mean) ** 2
                for x, weight in zip(xs, weights)
            )
            if denominator <= 0.0:
                return None
            slope = sum(
                weight * (x - x_mean) * (y - y_mean)
                for x, y, weight in zip(xs, ys, weights)
            ) / denominator
            intercept = y_mean - slope * x_mean
            if slope <= 0.0 or not all(math.isfinite(x) for x in (intercept, slope)):
                return None
            return intercept, slope


        @staticmethod
        def _online_viscosity_extrapolation_penalty(
            distance_K: float,
        ) -> Optional[float]:
            if distance_K <= 0.0:
                return 0.0
            if distance_K <= 10.0:
                return 0.01
            if distance_K <= 25.0:
                return 0.04
            if distance_K <= 50.0:
                return 0.09
            if distance_K <= ONLINE_VISCOSITY_MAXIMUM_EXTRAPOLATION_K:
                return 0.18
            return None


        def _online_liquid_viscosity(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            phase_key: str,
            *,
            allow_online: bool,
        ) -> Optional[PropertyResolutionResult]:
            if (
                phase_key not in {'liquid', 'l'}
                or not self._props_allow_online(props, allow_online)
            ):
                return None
            try:
                payload = self._fetch_viscosity_online(symbol, props)
            except LookupError:
                return None
            if payload is None:
                return None
            points = self._online_liquid_viscosity_points(
                symbol,
                props,
                payload,
                allow_online=allow_online,
            )
            if not points:
                return None
            phenol, _ = self._viscosity_phenol_status(
                symbol,
                props,
                allow_online=allow_online,
            )
            retained, outlier_count = self._online_viscosity_fit_inliers(points)
            lower = min(float(point['T_K']) for point in retained)
            upper = max(float(point['T_K']) for point in retained)
            span = upper - lower
            if span >= ONLINE_VISCOSITY_MINIMUM_FIT_SPAN_K:
                fit = self._weighted_online_viscosity_fit(retained)
                if fit is None:
                    return None
                intercept, slope = fit
                span_penalty = (
                    0.04 if span < ONLINE_VISCOSITY_FULL_QUALITY_FIT_SPAN_K else 0.0
                )
                distance = max(lower - T, T - upper, 0.0)
                if phenol and distance > 10.0:
                    return None
                extrapolation_penalty = (
                    self._online_viscosity_extrapolation_penalty(distance)
                )
                if extrapolation_penalty is None:
                    return None
                try:
                    value = math.exp(intercept + slope / float(T))
                except (OverflowError, ValueError, ZeroDivisionError):
                    return None
                if value <= 0.0 or not math.isfinite(value):
                    return None
                base_quality = min(0.93, 0.85 + 0.01 * len(retained))
                density_penalty = max(
                    float(point['density_penalty']) for point in retained
                )
                quality = self._clamp_quality(
                    base_quality
                    - extrapolation_penalty
                    - density_penalty
                    - span_penalty
                    - (0.05 if phenol else 0.0)
                )
                region = (
                    f'interpolation over {lower:g}-{upper:g} K'
                    if distance <= 0.0 else
                    f'extrapolated {distance:g} K beyond {lower:g}-{upper:g} K'
                )
                converted_count = sum(
                    'kinematic_converted' in str(point['kind'])
                    for point in retained
                )
                references = '; '.join(dict.fromkeys(
                    str(point.get('reference') or '')
                    for point in retained
                    if point.get('reference')
                ))
                return PropertyResolutionResult(
                    value=value,
                    source='online',
                    method='pubchem_liquid_viscosity_arrhenius_fit',
                    quality=quality,
                    notes=(
                        f'Weighted ln(mu)=A+B/T fit to {len(retained)} PubChem '
                        f'viscosity temperature point(s), including '
                        f'{converted_count} density-converted kinematic point(s); '
                        f'{region}; A={intercept:.8g}, B={slope:.8g} K; '
                        f'base quality {base_quality:.2f}, extrapolation penalty '
                        f'{extrapolation_penalty:.2f}, density penalty '
                        f'{density_penalty:.2f}, span penalty {span_penalty:.2f}'
                        + ('; phenol penalty 0.05' if phenol else '')
                        + (
                            f'; rejected {outlier_count} robust-fit outlier(s)'
                            if outlier_count else ''
                        )
                        + (f'; references: {references}' if references else '')
                        + '; units Pa*s'
                    ),
                )

            if phenol:
                return None
            point = self._average_online_viscosity_points(retained)
            point['density_penalty'] = max(
                float(item['density_penalty']) for item in retained
            )
            if len(retained) > 1:
                point['notes'] += (
                    f'; combined {len(retained)} points spanning {span:g} K '
                    'into a density-quality-weighted geometric-mean anchor'
                )
            if outlier_count:
                point['notes'] += f'; rejected {outlier_count} robust-fit outlier(s)'
            distance = abs(float(T) - float(point['T_K']))
            if distance <= 25.0:
                quality = 0.85
            elif distance <= 50.0:
                quality = 0.80
            elif distance <= 75.0:
                quality = 0.75
            elif distance <= 100.0:
                quality = 0.70
            else:
                return None
            quality = self._clamp_quality(
                quality - float(point['density_penalty'])
            )
            return self._nannoolal_liquid_viscosity(
                symbol,
                props,
                T,
                phase_key,
                allow_online=allow_online,
                anchor=point,
                quality=quality,
                method='nannoolal_anchored_liquid_viscosity',
            )


        def _nannoolal_liquid_viscosity(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            phase_key: str,
            *,
            allow_online: bool,
            anchor: Optional[Dict[str, Any]],
            quality: float,
            method: str,
        ) -> Optional[PropertyResolutionResult]:
            if phase_key not in {'liquid', 'l'}:
                return None
            phenol, resolved_smiles = self._viscosity_phenol_status(
                symbol,
                props,
                allow_online=allow_online,
            )
            if phenol:
                return None
            smiles = self._viscosity_smiles(props)
            if not smiles and resolved_smiles and resolved_smiles.value:
                smiles = str(resolved_smiles.value).strip()
            if not smiles:
                return None
            allow_online = self._props_allow_online(props, allow_online)
            tc_result = self._source_result_for_value(props, 'Tc', units='K')
            if tc_result is None:
                try:
                    critical = self.resolve_critical_properties(
                        symbol,
                        props,
                        allow_online=allow_online,
                        allow_estimation=True,
                    )
                    tc_result = critical.get('Tc') if critical else None
                except Exception:
                    return None
            try:
                tc = float(tc_result.value) if tc_result is not None else None
                temperature = float(T)
            except (TypeError, ValueError):
                return None
            if (
                tc is None or not math.isfinite(tc) or tc <= 0.0
                or not math.isfinite(temperature) or temperature <= 0.0
            ):
                return None
            tr_limit = 0.8 if self._result_is_real(tc_result) else 0.75
            if temperature > tr_limit * tc:
                return None
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from .. import nannoolal_method
                else:
                    import nannoolal_method
            except ImportError:
                return None

            tb_result = None
            viscosity_point = None
            if anchor is not None:
                viscosity_point = (
                    float(anchor['T_K']),
                    float(anchor['mu_Pa_s']) * 1000.0,
                )
            else:
                tb_result = self._source_result_for_value(props, 'Tb', units='K')
                if tb_result is None:
                    try:
                        tb_result = self.resolve_boiling_point(
                            symbol,
                            props,
                            allow_online=allow_online,
                            allow_estimation=True,
                        )
                    except Exception:
                        tb_result = None
            try:
                estimate = nannoolal_method.estimate_viscosity(
                    smiles,
                    tb=(
                        float(tb_result.value)
                        if tb_result is not None and tb_result.value is not None
                        else None
                    ),
                    visc_point=viscosity_point,
                )
                value = estimate.viscosity_Pa_s(float(T))
            except (
                nannoolal_method.NannoolalError,
                OverflowError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                return None
            if (
                estimate.dbv is None
                or estimate.tv_K is None
                or value is None
                or value <= 0.0
                or not math.isfinite(value)
            ):
                return None
            if anchor is not None and estimate.tv_source != 'from viscosity point':
                return None
            groups = ', '.join(
                f'{count} {name}'
                for name, count in sorted(
                    estimate.groups.items(),
                    key=lambda item: str(item[0]),
                )
            )
            warnings_note = '; '.join(str(item) for item in estimate.warnings)
            if anchor is not None:
                anchor_note = (
                    f"anchored at T={float(anchor['T_K']):g} K, "
                    f"mu={float(anchor['mu_Pa_s']):g} Pa*s from "
                    f"{anchor['kind']}; distance {abs(float(T) - float(anchor['T_K'])):g} K; "
                    f"{anchor.get('notes') or ''}"
                )
                source = 'calculated'
            else:
                anchor_note = (
                    f'Tv {estimate.tv_source}'
                    + (
                        f' using Tb from {tb_result.source}/{tb_result.method}'
                        if tb_result is not None else ''
                    )
                )
                source = 'estimated'
            return PropertyResolutionResult(
                value=value,
                source=source,
                method=method,
                quality=self._clamp_quality(quality),
                notes=(
                    f'Nannoolal Part-4 saturated-liquid viscosity; {anchor_note}; '
                    f'dBv={estimate.dbv:g}, Tv={estimate.tv_K:g} K; '
                    f'T/Tc={temperature / tc:g}, limit {tr_limit:g} '
                    f'using Tc from {tc_result.source}/{tc_result.method}; '
                    f'groups: {groups}'
                    + (f'; {warnings_note}' if warnings_note else '')
                    + '; units Pa*s'
                ),
            )


        def _nannoolal_predictive_liquid_viscosity(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            phase_key: str,
            *,
            allow_online: bool,
        ) -> Optional[PropertyResolutionResult]:
            return self._nannoolal_liquid_viscosity(
                symbol,
                props,
                T,
                phase_key,
                allow_online=allow_online,
                anchor=None,
                quality=0.65,
                method='nannoolal_predictive_liquid_viscosity',
            )


        @classmethod
        def _coolprop_module(cls):
            return coolprop_module()


        def _hsu_liquid_viscosity(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            phase_key: str,
        ) -> Optional[PropertyResolutionResult]:
            """Hsu GC liquid viscosity via the standalone hsu_method engine.

            Fragmentation, misprint corrections, refit alkyne units, domain
            gates and the Sum(d) Pc-sensitivity guard all live in
            hsu_method.py (see its docstring for provenance).
            """
            if phase_key not in {'liquid', 'l'}:
                return None
            try:
                T = float(T)
            except (TypeError, ValueError):
                return None
            if T <= 0.0:
                return None
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from .. import hsu_method
                else:
                    import hsu_method
            except ImportError:
                return None

            resolved_smiles = None
            smiles = self._viscosity_smiles(props)
            if not smiles:
                resolved_smiles = self._resolve_smiles_result(
                    symbol, props, allow_online=self._props_allow_online(props))
                smiles = (str(resolved_smiles.value).strip()
                          if resolved_smiles and resolved_smiles.value else None)
            if not smiles:
                return None

            cache = getattr(self, '_hsu_fragmentation_cache', None)
            if cache is None:
                cache = {}
                self._hsu_fragmentation_cache = cache
            fragmentation = cache.get(smiles)
            if fragmentation is None:
                try:
                    fragmentation = hsu_method.fragment(smiles)
                except hsu_method.HsuFragmentationError as exc:
                    cache[smiles] = exc
                    return None
                cache[smiles] = fragmentation
            elif isinstance(fragmentation, hsu_method.HsuFragmentationError):
                return None

            try:
                critical = self.resolve_critical_properties(
                    symbol,
                    props,
                    allow_online=self._props_allow_online(props),
                    allow_estimation=True,
                )
            except Exception:
                return None
            tc_result = critical.get('Tc') if critical else None
            pc_result = critical.get('Pc') if critical else None
            if (
                tc_result is None or tc_result.value is None
                or pc_result is None or pc_result.value is None
            ):
                return None
            try:
                tc = float(tc_result.value)
                pc_bar = float(pc_result.value)
            except (TypeError, ValueError):
                return None
            if tc <= 0.0 or pc_bar <= 0.0:
                return None
            tr = T / tc
            if not (fragmentation.tr_min - 1e-9 <= tr
                    <= fragmentation.tr_max + 1e-9):
                return None

            pc_quality = float(pc_result.quality or 0.0)
            estimate = hsu_method.HsuViscosityResult(
                fragmentation=fragmentation,
                pc_kPa=pc_bar * 100.0,
                tc_K=tc,
                pc_quality=pc_quality,
            )
            value = estimate.viscosity_Pa_s(T)
            if value is None or value <= 0.0 or not math.isfinite(value):
                return None

            structure_result = resolved_smiles or self._source_result_for_value(
                props,
                'smiles',
                value=smiles,
                default_source='provided',
                default_method='hsu_native_fragmentation',
                default_quality=1.0,
            )
            # Tc only gates the Tr validity window, and Pc uncertainty enters
            # through the Sum(d)-aware guard inside estimate.quality -- so a
            # low-Sum(d) molecule is deliberately NOT penalized for an
            # estimated Pc (hexane Sum(d) ~ -0.05 vs cyclohexane -9.1).
            quality = self._combine_quality(
                [structure_result],
                method_factor=estimate.quality,
            )
            group_note = ', '.join(
                f'{count} {name}'
                for name, count in sorted(fragmentation.groups.items()))
            extra = '; '.join(note for note in fragmentation.notes
                              if not note.startswith('__'))
            return PropertyResolutionResult(
                value=value,
                source='calculated',
                method='hsu_liquid_viscosity',
                quality=quality,
                notes=(
                    f'Hsu group-contribution estimate via hsu_method '
                    f'(Tr={tr:.3f}, valid {fragmentation.tr_min:.2f}-'
                    f'{fragmentation.tr_max:.2f}; '
                    f'Sum(d)={fragmentation.d_sum:+.2f}); '
                    f'groups: {group_note}; '
                    + (f'{extra}; ' if extra else '')
                    + f'Tc from {tc_result.source}/{tc_result.method}, '
                    f'Pc from {pc_result.source}/{pc_result.method} '
                    f'(quality {pc_quality:.2f}); units Pa*s'
                ),
            )


        @staticmethod
        def _coolprop_viscosity_attempts(
            phase_key: str,
            T: float,
            P: Optional[float] = None,
            psat_pa: Optional[float] = None,
            impose_phase: bool = True,
        ):
            # Backends that cannot impose a phase (IF97) pick the region from
            # (T, P) themselves and refuse states on the saturation line, so
            # they get plain 'T' inputs and a direct saturated-state query
            # whenever the default pressure lands exactly on Psat.
            if phase_key in {'liquid', 'l'}:
                input_tag = 'T|liquid' if impose_phase else 'T'
                if P is not None:
                    return ((
                        input_tag, T, 'P', P * 1.0e5,
                        f'liquid at P={P:g} bar',
                    ),)
                pressure_pa = max(COOLPROP_STANDARD_PRESSURE_PA, psat_pa or 0.0)
                basis = (
                    f'max(1 atm, Psat={psat_pa / 1.0e5:g} bar)'
                    if psat_pa is not None
                    else '1 atm; Psat unavailable'
                )
                if (
                    not impose_phase
                    and psat_pa is not None
                    and pressure_pa <= psat_pa * (1.0 + 1.0e-9)
                ):
                    return ((
                        'T', T, 'Q', 0.0,
                        f'saturated liquid ({basis})',
                    ),)
                return ((
                    input_tag, T, 'P', pressure_pa,
                    f'default liquid P={pressure_pa / 1.0e5:g} bar ({basis})',
                ),)
            if phase_key in {'gas', 'vapor', 'vapour', 'ideal_gas', 'ideal'}:
                input_tag = 'T|gas' if impose_phase else 'T'
                if P is not None:
                    return ((
                        input_tag, T, 'P', P * 1.0e5,
                        f'vapor at P={P:g} bar',
                    ),)
                pressure_pa = (
                    min(COOLPROP_STANDARD_PRESSURE_PA, psat_pa)
                    if psat_pa is not None
                    else COOLPROP_STANDARD_PRESSURE_PA
                )
                basis = (
                    f'min(1 atm, Psat={psat_pa / 1.0e5:g} bar)'
                    if psat_pa is not None
                    else '1 atm; Psat unavailable'
                )
                if (
                    not impose_phase
                    and psat_pa is not None
                    and pressure_pa >= psat_pa * (1.0 - 1.0e-9)
                ):
                    return ((
                        'T', T, 'Q', 1.0,
                        f'saturated vapor ({basis})',
                    ),)
                return ((
                    input_tag, T, 'P', pressure_pa,
                    f'default vapor P={pressure_pa / 1.0e5:g} bar ({basis})',
                ),)
            return ()


        def _coolprop_viscosity(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            phase_key: str,
            P: Optional[float] = None,
        ) -> Optional[PropertyResolutionResult]:
            reference = self._coolprop_reference(symbol, props)
            if reference is None:
                return None
            return self._coolprop_viscosity_from_reference(
                reference,
                T,
                phase_key,
                P=P,
            )


        def _coolprop_viscosity_from_reference(
            self,
            reference,
            T: float,
            phase_key: str,
            P: Optional[float] = None,
        ) -> Optional[PropertyResolutionResult]:
            CP = self._coolprop_module()
            if CP is None:
                return None

            psat_pa = None
            if P is None:
                try:
                    candidate = float(CP.PropsSI(
                        'P', 'T', T, 'Q', 0.0, reference.qualified_name,
                    ))
                except Exception:
                    candidate = None
                if candidate is not None and candidate > 0.0 and math.isfinite(candidate):
                    psat_pa = candidate

            for input1, value1, input2, value2, state_note in self._coolprop_viscosity_attempts(
                phase_key,
                T,
                P,
                psat_pa,
                impose_phase=reference.backend != 'IF97',
            ):
                try:
                    value = CP.PropsSI(
                        'V', input1, value1, input2, value2, reference.qualified_name,
                    )
                except Exception:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if value <= 0.0 or not math.isfinite(value):
                    continue
                return PropertyResolutionResult(
                    value=value,
                    source='local',
                    method=f'{reference.method_prefix}_viscosity',
                    quality=COOLPROP_VISCOSITY_QUALITY,
                    notes=(
                        f"CoolProp {reference.backend} transport model for "
                        f"{reference.fluid} ({state_note}); units Pa*s"
                    ),
                )
            return None


        @staticmethod
        def _vapor_viscosity_composition_class(
            counts: Optional[Dict[str, int]],
        ) -> tuple[str, Optional[int]]:
            if not counts:
                return 'unknown', None
            carbon = int(counts.get('C', 0))
            heavy_atoms = sum(
                int(count) for element, count in counts.items()
                if element != 'H'
            )
            heteroatoms = heavy_atoms - carbon
            from .organic_classification import is_strict_organic_formula_counts
            if carbon > 0 and not is_strict_organic_formula_counts(counts):
                return 'inorganic', heavy_atoms
            if carbon > 0 and heteroatoms == 0 and set(counts).issubset({'C', 'H'}):
                return 'hydrocarbon', heavy_atoms
            if carbon == 0:
                return 'inorganic', heavy_atoms
            if heteroatoms <= carbon / 4.0:
                return 'sparse_heteroatom', heavy_atoms
            return 'polar_organic', heavy_atoms


        def _reichenberg_structure(
            self,
            symbol: str,
            props: Dict[str, Any],
        ):
            native_smiles = self._viscosity_smiles(props)
            cache_key = (symbol, native_smiles)
            cache = getattr(self, '_reichenberg_structure_cache', None)
            if cache is None:
                cache = {}
                self._reichenberg_structure_cache = cache
            if cache_key in cache:
                return cache[cache_key]

            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from .. import reichenberg_method
                else:
                    import reichenberg_method
            except ImportError:
                result = (None, None)
                cache[cache_key] = result
                return result

            resolved_smiles = None
            smiles = native_smiles
            if not smiles:
                resolved_smiles = self._resolve_smiles_result(
                    symbol,
                    props,
                    allow_online=self._props_allow_online(props),
                )
                smiles = (
                    str(resolved_smiles.value).strip()
                    if resolved_smiles and resolved_smiles.value else None
                )
            if not smiles:
                result = (None, None)
                cache[cache_key] = result
                return result
            structure_result = resolved_smiles or self._source_result_for_value(
                props,
                'smiles',
                value=smiles,
                default_source='provided',
                default_method='reichenberg_native_fragmentation',
                default_quality=1.0,
            )
            try:
                profile = reichenberg_method.structure_profile(smiles)
            except reichenberg_method.ReichenbergFragmentationError:
                result = (structure_result, None)
                cache[cache_key] = result
                return result
            result = (structure_result, profile)
            cache[cache_key] = result
            return result


        def _reichenberg_fragmentation(
            self,
            structure_result: Optional[PropertyResolutionResult],
        ):
            if structure_result is None or not structure_result.value:
                return None
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from .. import reichenberg_method
                else:
                    import reichenberg_method
            except ImportError:
                return None
            smiles = str(structure_result.value)
            cache = getattr(self, '_reichenberg_fragmentation_cache', None)
            if cache is None:
                cache = {}
                self._reichenberg_fragmentation_cache = cache
            fragmentation = cache.get(smiles)
            if fragmentation is None:
                try:
                    fragmentation = reichenberg_method.fragment(smiles)
                except reichenberg_method.ReichenbergFragmentationError as exc:
                    cache[smiles] = exc
                    return None
                cache[smiles] = fragmentation
            elif isinstance(
                fragmentation,
                reichenberg_method.ReichenbergFragmentationError,
            ):
                return None
            return fragmentation


        @staticmethod
        def _is_unbranched_terminal_1_alkyne(fragmentation) -> bool:
            groups = fragmentation.groups
            return bool(
                groups.get('alkyne_ch') == 1
                and groups.get('alkyne_c') == 1
                and groups.get('ch2', 0) >= 1
                and groups.get('ch3') == 1
                and set(groups) <= {'alkyne_ch', 'alkyne_c', 'ch2', 'ch3'}
            )


        @staticmethod
        def _is_unbranched_c3_plus_aldehyde(fragmentation) -> bool:
            groups = fragmentation.groups
            return bool(
                groups.get('aldehyde') == 1
                and groups.get('ch2', 0) >= 1
                and groups.get('ch3') == 1
                and set(groups) <= {'aldehyde', 'ch2', 'ch3'}
            )


        def _reichenberg_vapor_viscosity(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            *,
            composition_class: str,
            heavy_atoms: int,
            structure_result: Optional[PropertyResolutionResult] = None,
            fragmentation=None,
            inorganic: bool = False,
        ) -> Optional[PropertyResolutionResult]:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from .. import reichenberg_method
                else:
                    import reichenberg_method
            except ImportError:
                return None

            if inorganic:
                fragmentation = None
                method_factor = REICHENBERG_INORGANIC_FACTOR
                method = 'reichenberg_zero_dipole_inorganic_gas_viscosity'
                branch_note = 'inorganic prefactor'
            else:
                if fragmentation is None:
                    fragmentation = self._reichenberg_fragmentation(structure_result)
                if fragmentation is None:
                    return None
                if composition_class == 'hydrocarbon':
                    method_factor = REICHENBERG_TERMINAL_ALKYNE_FACTOR
                    class_note = 'unbranched terminal C4+ 1-alkyne'
                elif composition_class == 'sparse_heteroatom':
                    method_factor = REICHENBERG_SPARSE_HETEROATOM_FACTOR
                    class_note = 'slightly polar organic'
                else:
                    method_factor = REICHENBERG_POLAR_ORGANIC_FACTOR
                    class_note = 'polar organic'
                method = 'reichenberg_zero_dipole_organic_gas_viscosity'
                groups = ', '.join(
                    f'{count} {name}'
                    for name, count in sorted(fragmentation.groups.items())
                )
                branch_note = (
                    f'{class_note}; group sum={fragmentation.contribution_sum:.3f}; '
                    f'groups: {groups}'
                )

            cache_key = (
                str(symbol),
                self._viscosity_kernel_props_fingerprint(props),
                composition_class,
                int(heavy_atoms),
                bool(inorganic),
                branch_note,
                repr(structure_result),
            )
            prepared = self._reichenberg_viscosity_input_cache.get(cache_key)
            if prepared is None:
                mw_result = self._source_result_for_value(
                    props,
                    'MW',
                    units='g/mol',
                )
                if mw_result is None or mw_result.value is None:
                    return None
                try:
                    critical = self.resolve_critical_properties(
                        symbol,
                        props,
                        allow_online=self._props_allow_online(props),
                        allow_estimation=True,
                    )
                except Exception:
                    return None
                tc_result = critical.get('Tc') if critical else None
                pc_result = critical.get('Pc') if critical else None
                if (
                    tc_result is None or tc_result.value is None
                    or pc_result is None or pc_result.value is None
                ):
                    return None
                try:
                    mw = float(mw_result.value)
                    tc = float(tc_result.value)
                    pc_bar = float(pc_result.value)
                except (TypeError, ValueError):
                    return None
                if min(mw, tc, pc_bar) <= 0.0:
                    return None
                inputs = [mw_result, tc_result, pc_result]
                if structure_result is not None and not inorganic:
                    inputs.append(structure_result)
                quality = self._combine_quality(
                    inputs,
                    method_factor=method_factor,
                )
                notes = (
                    f'Reichenberg low-pressure vapor viscosity estimate with zero '
                    f'dipole; {branch_note}; {heavy_atoms} heavy atoms; '
                    f'MW from {mw_result.source}/{mw_result.method}, '
                    f'Tc from {tc_result.source}/{tc_result.method}, '
                    f'Pc from {pc_result.source}/{pc_result.method}; units Pa*s'
                )
                prepared = (mw, tc, pc_bar, quality, notes)
                self._reichenberg_viscosity_input_cache[cache_key] = prepared
            mw, tc, pc_bar, quality, notes = prepared
            try:
                T = float(T)
            except (TypeError, ValueError):
                return None
            if T <= 0.0:
                return None

            try:
                value = reichenberg_method.viscosity_Pa_s(
                    T,
                    mw,
                    tc,
                    pc_bar,
                    dipole_D=0.0,
                    fragmentation=fragmentation,
                    inorganic=inorganic,
                )
            except (TypeError, ValueError, OverflowError):
                return None
            return PropertyResolutionResult(
                value=value,
                source='estimated',
                method=method,
                quality=quality,
                notes=notes,
            )


        def _estimated_vapor_viscosity(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            phase_key: str,
        ) -> Optional[PropertyResolutionResult]:
            if phase_key not in {'gas', 'vapor', 'vapour', 'ideal_gas', 'ideal'}:
                return None

            counts = self._viscosity_formula_counts(symbol, props)
            composition_class, heavy_atoms = self._vapor_viscosity_composition_class(
                counts,
            )
            formula_has_no_hydrogen = bool(counts) and int(counts.get('H', 0)) == 0

            structure_result = None
            structure_profile = None
            fragmentation = None
            if composition_class not in {'inorganic'}:
                structure_result, structure_profile = self._reichenberg_structure(
                    symbol,
                    props,
                )
                if composition_class == 'unknown' and structure_profile is not None:
                    heavy_atoms = structure_profile.heavy_atoms
                    if structure_profile.carbon_atoms == 0:
                        composition_class = 'inorganic'
                    elif structure_profile.heteroatoms == 0:
                        composition_class = 'hydrocarbon'
                    elif (
                        structure_profile.heteroatoms
                        <= structure_profile.carbon_atoms / 4.0
                    ):
                        composition_class = 'sparse_heteroatom'
                    else:
                        composition_class = 'polar_organic'
                fragmentation = self._reichenberg_fragmentation(structure_result)

            if composition_class == 'hydrocarbon':
                # scripts/vapor_viscosity/benchmark_reichenberg_resolved_dipole.py
                # found that all seven unbranched terminal C4+ 1-alkynes favored
                # zero-dipole Reichenberg by 4.34 curve-MAPE points on average.
                # The two-family rule lowered mean MAPE from 3.089% to 2.753%,
                # outperforming the resolved-dipole classifier without QM.
                if (
                    fragmentation is not None
                    and self._is_unbranched_terminal_1_alkyne(fragmentation)
                ):
                    reichenberg = self._reichenberg_vapor_viscosity(
                        symbol,
                        props,
                        T,
                        composition_class=composition_class,
                        heavy_atoms=heavy_atoms,
                        structure_result=structure_result,
                        fragmentation=fragmentation,
                    )
                    if reichenberg is not None:
                        return reichenberg
                return self._yoon_thodos_viscosity(
                    symbol,
                    props,
                    T,
                    phase_key,
                    method_factor=YOON_THODOS_HYDROCARBON_FACTOR,
                    composition_note='hydrocarbon multiplier 0.88',
                )

            # scripts/vapor_viscosity/fit_reichenberg_aldehyde_group.py found
            # no transferable replacement for Perry's published 14.02 group:
            # the shared refit worsened LOO mean MAPE (3.969% -> 4.425%), and
            # the affine refit still lost to Yoon-Thodos on 8/9 aldehydes.
            # Unbranched C3+ homologs favor Yoon or practically tie; the C2
            # member acetaldehyde remains eligible for Reichenberg.
            if (
                fragmentation is not None
                and self._is_unbranched_c3_plus_aldehyde(fragmentation)
            ):
                factor = (
                    YOON_THODOS_SPARSE_HETEROATOM_FACTOR
                    if composition_class == 'sparse_heteroatom'
                    else YOON_THODOS_HETEROATOM_FACTOR
                )
                return self._yoon_thodos_viscosity(
                    symbol,
                    props,
                    T,
                    phase_key,
                    method_factor=factor,
                    composition_note=(
                        'unbranched C3+ aldehyde selected for Yoon-Thodos by '
                        'the Perry aldehyde-family benchmark'
                    ),
                )

            if heavy_atoms is None:
                return self._yoon_thodos_viscosity(
                    symbol,
                    props,
                    T,
                    phase_key,
                    method_factor=YOON_THODOS_HETEROATOM_FACTOR,
                    composition_note=(
                        'formula and usable structure unavailable; '
                        'conservative polar-organic fallback multiplier 0.75'
                    ),
                )
            if heavy_atoms < REICHENBERG_MINIMUM_HEAVY_ATOMS:
                return self._yoon_thodos_viscosity(
                    symbol,
                    props,
                    T,
                    phase_key,
                    method_factor=YOON_THODOS_SMALL_MOLECULE_FACTOR,
                    composition_note=(
                        f'{heavy_atoms} heavy atoms; below Reichenberg minimum of '
                        f'{REICHENBERG_MINIMUM_HEAVY_ATOMS}; multiplier 0.50'
                    ),
                )

            if composition_class == 'inorganic':
                reichenberg = self._reichenberg_vapor_viscosity(
                    symbol,
                    props,
                    T,
                    composition_class=composition_class,
                    heavy_atoms=heavy_atoms,
                    inorganic=True,
                )
                if reichenberg is not None:
                    return reichenberg
            else:
                reichenberg = self._reichenberg_vapor_viscosity(
                    symbol,
                    props,
                    T,
                    composition_class=composition_class,
                    heavy_atoms=heavy_atoms,
                    structure_result=structure_result,
                    fragmentation=fragmentation,
                )
                if reichenberg is not None:
                    return reichenberg
                if (
                    (
                        structure_profile is not None
                        and not structure_profile.has_carbon_hydrogen_bond
                    )
                    or formula_has_no_hydrogen
                ):
                    reichenberg = self._reichenberg_vapor_viscosity(
                        symbol,
                        props,
                        T,
                        composition_class='inorganic',
                        heavy_atoms=heavy_atoms,
                        inorganic=True,
                    )
                    if reichenberg is not None:
                        return reichenberg

            if composition_class == 'sparse_heteroatom':
                factor = YOON_THODOS_SPARSE_HETEROATOM_FACTOR
                note = 'non-fragmentable slightly polar organic fallback multiplier 0.88'
            else:
                factor = YOON_THODOS_HETEROATOM_FACTOR
                note = 'non-fragmentable polar fallback multiplier 0.75'
            return self._yoon_thodos_viscosity(
                symbol,
                props,
                T,
                phase_key,
                method_factor=factor,
                composition_note=note,
            )


        def _yoon_thodos_viscosity(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            phase_key: str,
            *,
            method_factor: Optional[float] = None,
            composition_note: Optional[str] = None,
        ) -> Optional[PropertyResolutionResult]:
            if phase_key not in {'gas', 'vapor', 'vapour', 'ideal_gas', 'ideal'}:
                return None
            try:
                T = float(T)
            except (TypeError, ValueError):
                return None
            if T <= 0.0:
                return None

            if method_factor is None or composition_note is None:
                default_factor, default_note = self._yoon_thodos_composition_factor(
                    symbol,
                    props,
                )
                if method_factor is None:
                    method_factor = default_factor
                if composition_note is None:
                    composition_note = default_note

            cache_key = (
                str(symbol),
                self._viscosity_kernel_props_fingerprint(props),
                float(method_factor),
                str(composition_note),
            )
            prepared = self._yoon_thodos_viscosity_input_cache.get(cache_key)
            if prepared is None:
                mw_result = self._source_result_for_value(
                    props,
                    'MW',
                    units='g/mol',
                )
                if mw_result is None or mw_result.value is None:
                    return None
                try:
                    mw = float(mw_result.value)
                except (TypeError, ValueError):
                    return None
                if mw <= 0.0:
                    return None
                try:
                    critical = self.resolve_critical_properties(
                        symbol,
                        props,
                        allow_online=self._props_allow_online(props),
                        allow_estimation=True,
                    )
                except Exception:
                    return None
                tc_result = critical.get('Tc') if critical else None
                pc_result = critical.get('Pc') if critical else None
                if (
                    tc_result is None or tc_result.value is None
                    or pc_result is None or pc_result.value is None
                ):
                    return None
                try:
                    tc = float(tc_result.value)
                    pc_bar = float(pc_result.value)
                except (TypeError, ValueError):
                    return None
                if tc <= 0.0 or pc_bar <= 0.0:
                    return None
                quality = self._combine_quality(
                    [tc_result, pc_result],
                    method_factor=method_factor,
                )
                notes = (
                    "Yoon-Thodos low-pressure vapor viscosity estimate; "
                    f"{composition_note}; Tc from {tc_result.source}/{tc_result.method}, "
                    f"Pc from {pc_result.source}/{pc_result.method}; units Pa*s"
                )
                prepared = (mw, tc, pc_bar, quality, notes)
                self._yoon_thodos_viscosity_input_cache[cache_key] = prepared
            mw, tc, pc_bar, quality, notes = prepared

            tr = T / tc
            pc_pa = pc_bar * 1.0e5
            try:
                numerator = (
                    46.1 * tr ** 0.618
                    - 20.4 * math.exp(-0.449 * tr)
                    + 19.4 * math.exp(-4.058 * tr)
                    + 1.0
                )
                denominator = (
                    2.173424e11
                    * tc ** (1.0 / 6.0)
                    * mw ** -0.5
                    * pc_pa ** (-2.0 / 3.0)
                )
                value = numerator / denominator
            except (OverflowError, ValueError, ZeroDivisionError):
                return None
            if value <= 0.0 or not math.isfinite(value):
                return None

            return PropertyResolutionResult(
                value=value,
                source='calculated',
                method='yoon_thodos_gas_viscosity',
                quality=quality,
                notes=notes,
            )


        def _yoon_thodos_composition_factor(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> tuple[float, str]:
            counts = self._viscosity_formula_counts(symbol, props)
            if not counts:
                return (
                    YOON_THODOS_HETEROATOM_FACTOR,
                    'formula unavailable; conservative polar-organic multiplier 0.75',
                )

            carbon = counts.get('C', 0)
            heteroatoms = sum(
                count for element, count in counts.items()
                if element not in {'C', 'H'}
            )
            if carbon > 0 and heteroatoms == 0 and set(counts).issubset({'C', 'H'}):
                return YOON_THODOS_HYDROCARBON_FACTOR, 'hydrocarbon multiplier 0.88'
            if carbon > 0 and heteroatoms <= carbon / 4.0:
                return (
                    YOON_THODOS_SPARSE_HETEROATOM_FACTOR,
                    'sparse-heteroatom multiplier 0.88',
                )
            return YOON_THODOS_HETEROATOM_FACTOR, 'heteroatom-rich multiplier 0.75'


        def _apply_viscosity_pressure_correction(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            phase_key: str,
            baseline: PropertyResolutionResult,
            P: Optional[float],
            rho_molar: Optional[float],
        ) -> PropertyResolutionResult:
            if phase_key in {'liquid', 'l'}:
                return self._apply_lucas_pressure_correction(
                    symbol,
                    props,
                    T,
                    baseline,
                    P,
                )
            return self._apply_jossi_pressure_correction(
                symbol,
                props,
                T,
                phase_key,
                baseline,
                P,
                rho_molar,
            )


        @staticmethod
        def _lucas_pressure_quality_factor(P_bar: float) -> float:
            if P_bar <= 5.0:
                return 1.00
            if P_bar <= 10.0:
                return 0.99
            if P_bar <= 20.0:
                return 0.97
            if P_bar <= 50.0:
                return 0.95
            if P_bar <= 100.0:
                return 0.89
            if P_bar <= 250.0:
                return 0.85
            if P_bar <= 500.0:
                return 0.70
            return 0.55


        @staticmethod
        def _lucas_uncorrected_pressure_quality_factor(P_bar: float) -> float:
            if P_bar <= 2.0:
                return 1.00
            if P_bar <= 10.0:
                return 0.96
            if P_bar <= 20.0:
                return 0.91
            if P_bar <= 50.0:
                return 0.85
            if P_bar <= 100.0:
                return 0.73
            if P_bar <= 250.0:
                return 0.55
            return 0.45


        def _lucas_uncorrected_result(
            self,
            baseline: PropertyResolutionResult,
            P_bar: float,
            note: str,
        ) -> PropertyResolutionResult:
            pressure_factor = self._lucas_uncorrected_pressure_quality_factor(P_bar)
            quality = self._clamp_quality(
                self._result_quality(baseline, 0.0) * pressure_factor
            )
            return PropertyResolutionResult(
                value=baseline.value,
                source=baseline.source,
                method=baseline.method,
                quality=quality,
                notes=(
                    f'{baseline.notes}; {note}; uncorrected pressure-quality '
                    f'multiplier {pressure_factor:.2f}'
                    if baseline.notes else
                    f'{note}; uncorrected pressure-quality multiplier {pressure_factor:.2f}'
                ),
            )


        def _lucas_corrected_quality(
            self,
            baseline: PropertyResolutionResult,
            P_bar: float,
            critical_inputs: tuple[PropertyResolutionResult, ...],
        ) -> tuple[float, float, float, float, float]:
            baseline_quality = self._result_quality(baseline, 0.0)
            pressure_factor = self._lucas_pressure_quality_factor(P_bar)
            uncorrected_factor = self._lucas_uncorrected_pressure_quality_factor(P_bar)
            critical_quality = min(
                self._result_quality(result, 0.0)
                for result in critical_inputs
            )
            uncertainty_factor = (
                1.0
                - 11.5
                * (1.0 - critical_quality) ** 1.8
                * (1.0 - uncorrected_factor)
            )
            corrected_quality = self._clamp_quality(
                baseline_quality * pressure_factor * uncertainty_factor
            )
            uncorrected_quality = self._clamp_quality(
                baseline_quality * uncorrected_factor
            )
            return (
                corrected_quality,
                uncorrected_quality,
                pressure_factor,
                uncorrected_factor,
                critical_quality,
            )


        @staticmethod
        def _lucas_liquid_pressure_factor(
            T: float,
            P_bar: float,
            tc: float,
            pc_bar: float,
            omega: float,
            psat_bar: float,
        ) -> Optional[float]:
            if (
                T <= 0.0 or P_bar <= 0.0 or tc <= 0.0 or pc_bar <= 0.0
                or T >= tc
            ):
                return None
            try:
                tr = T / tc
                c_value = tr * (
                    tr * (
                        tr * (
                            tr * (
                                tr * (
                                    tr * (15.6719 * tr - 59.8127) + 96.1209
                                ) - 84.8291
                            ) + 44.1706
                        ) - 13.4040
                    ) + 2.1616
                ) - 0.07921
                d_value = 0.3257 * (1.0039 - tr ** 2.573) ** -0.2906 - 0.2086
                a_value = 0.9991 - 4.674e-4 / (
                    1.0523 * tr ** -0.03877 - 1.0513
                )
                delta_pr = max((P_bar - psat_bar) / pc_bar, 0.0)
                factor = (
                    1.0 + d_value * (delta_pr / 2.118) ** a_value
                ) / (
                    1.0 + c_value * omega * delta_pr
                )
            except (OverflowError, ValueError, ZeroDivisionError):
                return None
            if factor <= 0.0 or not math.isfinite(factor):
                return None
            return factor


        def _apply_lucas_pressure_correction(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            baseline: PropertyResolutionResult,
            P: Optional[float],
        ) -> PropertyResolutionResult:
            reference_pressure_bar = COOLPROP_STANDARD_PRESSURE_PA / 1.0e5
            if P is None or P <= reference_pressure_bar:
                return baseline

            try:
                critical = self.resolve_critical_properties(
                    symbol,
                    props,
                    allow_online=self._props_allow_online(props),
                    allow_estimation=True,
                )
            except Exception:
                critical = None
            tc_result = critical.get('Tc') if critical else None
            pc_result = critical.get('Pc') if critical else None
            omega_result = critical.get('omega') if critical else None
            if not tc_result or not pc_result or not omega_result:
                return self._lucas_uncorrected_result(
                    baseline,
                    P,
                    'Lucas compressed-liquid correction skipped: Tc, Pc, or omega unavailable',
                )

            try:
                tc = float(tc_result.value)
                pc_bar = float(pc_result.value)
                omega = float(omega_result.value)
            except (TypeError, ValueError):
                return self._lucas_uncorrected_result(
                    baseline,
                    P,
                    'Lucas compressed-liquid correction skipped: invalid critical inputs',
                )
            if T >= tc:
                return self._lucas_uncorrected_result(
                    baseline,
                    P,
                    'Lucas compressed-liquid correction skipped at or above Tc',
                )

            try:
                psat_result = self.resolve_vapor_pressure(
                    symbol,
                    T,
                    props,
                    allow_online=self._props_allow_online(props),
                )
                psat_bar = float(psat_result.value)
            except Exception:
                return self._lucas_uncorrected_result(
                    baseline,
                    P,
                    'Lucas compressed-liquid correction skipped: Psat unavailable',
                )
            if psat_bar < 0.0 or not math.isfinite(psat_bar):
                return self._lucas_uncorrected_result(
                    baseline,
                    P,
                    'Lucas compressed-liquid correction skipped: invalid Psat',
                )

            target_factor = self._lucas_liquid_pressure_factor(
                T, P, tc, pc_bar, omega, psat_bar,
            )
            reference_factor = self._lucas_liquid_pressure_factor(
                T,
                reference_pressure_bar,
                tc,
                pc_bar,
                omega,
                psat_bar,
            )
            if target_factor is None or reference_factor is None:
                return self._lucas_uncorrected_result(
                    baseline,
                    P,
                    'Lucas compressed-liquid correction skipped: correlation outside valid state',
                )

            ratio = target_factor / reference_factor
            value = baseline.value * ratio
            if value <= 0.0 or not math.isfinite(value):
                return self._lucas_uncorrected_result(
                    baseline,
                    P,
                    'Lucas compressed-liquid correction skipped: nonphysical result',
                )
            if math.isclose(ratio, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
                return self._viscosity_result_with_note(
                    baseline,
                    f'Lucas pressure ratio is unity at P={P:g} bar',
                )

            (
                quality,
                uncorrected_quality,
                method_factor,
                uncorrected_factor,
                critical_quality,
            ) = self._lucas_corrected_quality(
                baseline,
                P,
                (tc_result, pc_result, omega_result, psat_result),
            )
            if quality < uncorrected_quality:
                return self._lucas_uncorrected_result(
                    baseline,
                    P,
                    'Lucas compressed-liquid correction skipped: corrected '
                    f'quality {quality:.3f} is below uncorrected quality '
                    f'{uncorrected_quality:.3f}, due to critical-input '
                    f'quality {critical_quality:.3f}',
                )
            return PropertyResolutionResult(
                value=value,
                source='calculated',
                method='lucas_compressed_liquid_viscosity',
                quality=quality,
                notes=(
                    f'Lucas compressed-liquid correction at P={P:g} bar from '
                    f'1 atm baseline {baseline.source}/{baseline.method}; '
                    f'back-calculated saturation viscosity using Psat={psat_bar:g} bar '
                    f'from {psat_result.source}/{psat_result.method}; '
                    f'F(P)/F(1 atm)={ratio:.6g}; pressure-quality multiplier '
                    f'{method_factor:.2f}; minimum critical-input quality '
                    f'{critical_quality:.3f}; uncorrected pressure-quality '
                    f'multiplier {uncorrected_factor:.2f}; units Pa*s'
                ),
            )


        @staticmethod
        def _viscosity_result_with_note(
            result: PropertyResolutionResult,
            note: str,
        ) -> PropertyResolutionResult:
            notes = f'{result.notes}; {note}' if result.notes else note
            return PropertyResolutionResult(
                value=result.value,
                source=result.source,
                method=result.method,
                quality=result.quality,
                notes=notes,
            )


        def _jossi_unifac_groups(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[Counter]:
            supplied = props.get('unifac_groups') if props else None
            if isinstance(supplied, dict) and supplied:
                return Counter({str(key).upper(): int(value) for key, value in supplied.items()})

            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..unifac import get_unifac_groups
                else:
                    from unifac import get_unifac_groups
            except Exception:
                return None
            smiles = self._viscosity_smiles(props)
            if not smiles:
                resolved = self._resolve_smiles_result(
                    symbol, props, allow_online=self._props_allow_online(props),
                )
                smiles = str(resolved.value).strip() if resolved and resolved.value else None
            try:
                expected_mw = float((props or {}).get('MW'))
            except (TypeError, ValueError):
                expected_mw = None
            for candidate in self._identifier_candidates(symbol, props):
                try:
                    groups = get_unifac_groups(
                        str(candidate),
                        smiles=smiles,
                        variant='UNIFDMD',
                        expected_mw=expected_mw,
                    )
                except Exception:
                    continue
                if groups:
                    return Counter({str(key).upper(): int(value) for key, value in groups.items()})
            return None


        def _jossi_polarity(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> tuple[Optional[str], float, str]:
            counts = self._viscosity_formula_counts(symbol, props) or {}
            carbon = int(counts.get('C', 0))
            hydrogen = int(counts.get('H', 0))
            if carbon > 0 and hydrogen > 0 and set(counts).issubset({'C', 'H'}):
                return 'nonpolar', 1.0, 'hydrocarbon formula classified as nonpolar'

            symmetric_nonpolar = (
                {'H': 2}, {'N': 2}, {'O': 2}, {'F': 2}, {'Cl': 2},
                {'Br': 2}, {'I': 2}, {'C': 1, 'O': 2}, {'C': 1, 'S': 2},
                {'C': 1, 'F': 4}, {'C': 1, 'Cl': 4}, {'S': 1, 'F': 6},
                {'He': 1}, {'Ne': 1}, {'Ar': 1}, {'Kr': 1}, {'Xe': 1},
            )
            if any(counts == formula for formula in symmetric_nonpolar):
                return 'nonpolar', 1.0, 'symmetric molecule override classified as nonpolar'
            if carbon > 0 and set(counts).issubset({'C', 'F'}):
                return 'nonpolar', 1.0, 'perfluorocarbon formula classified as nonpolar'

            groups = self._jossi_unifac_groups(symbol, props)
            label = ' '.join(
                str(value).lower()
                for value in (
                    symbol,
                    props.get('name') if props else None,
                    props.get('formula') if props else None,
                )
                if value
            )
            identity_label = ' '.join(
                str(value).lower()
                for value in (symbol, props.get('name') if props else None)
                if value
            )
            if (
                counts in ({'H': 2, 'O': 1}, {'H': 3, 'N': 1}, {'H': 1, 'F': 1}, {'H': 4, 'N': 2})
                or any(name in label for name in ('water', 'ammonia', 'hydrazine', 'hydrogen fluoride'))
            ):
                return None, 0.0, 'small strongly associating gas'

            hydrocarbon_groups = {
                'CH3', 'CH2', 'CH', 'C', 'ACH', 'AC',
                'ACCH3', 'ACCH2', 'ACCH', 'CY-CH2', 'CY-CH', 'CY-C',
                'CH2=CH', 'CH=CH', 'CH2=C', 'CH=C', 'C=C', 'CH=-C', 'C=-C',
                '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '65', '66', '70', '78', '79', '80',
            }
            if groups and set(groups).issubset(hydrocarbon_groups):
                return (
                    'nonpolar',
                    JOSSI_UNIFAC_CLASSIFICATION_FACTOR,
                    'Dortmund fragmentation contains only hydrocarbon groups',
                )

            # _jossi_unifac_groups explicitly requests Dortmund, so numeric
            # IDs here follow Dortmund semantics rather than NIST collisions.
            acid_groups = {'COOH', 'HCOOH', '42', '43'}
            if groups and any(key in acid_groups or 'COOH' in key for key in groups):
                return None, 0.0, 'carboxylic-acid vapor association'
            if re.search(r'\bacid\b', identity_label):
                return None, 0.0, 'acid vapor association'

            donor_groups = {
                '14', '81', '82', 'OH', 'OH (P)', 'OH (S)', 'OH (T)',
                'CH3OH', 'ACOH', 'DOH', 'GLYCEROL', 'ACNH2',
                'CH3NH2', 'CH2NH2', 'CHNH2', 'CNH2',
                'CH3NH', 'CH2NH', 'CHNH',
                '15', '17', '28', '29', '30', '31', '32', '33', '36',
                '62', '85', '91', '92', '93', '94', '100',
            }
            donor_count = sum(
                count for key, count in (groups or {}).items()
                if key in donor_groups or 'CONH' in key
            )
            amide_donor_groups = {'91', '92', '93', '94', '100'}
            if groups and any(
                'CONH' in key or key in amide_donor_groups
                for key in groups
            ):
                return None, 0.0, 'hydrogen-bonding amide vapor association'
            if donor_count > 1:
                return None, 0.0, 'multiple hydrogen-bond donor groups'
            if donor_count and carbon <= 2:
                return None, 0.0, 'small alcohol or amine vapor association'
            if donor_count:
                return (
                    'polar',
                    JOSSI_HBOND_CLASSIFICATION_FACTOR,
                    'Dortmund groups indicate a larger alcohol or amine; treated as polar cautiously',
                )
            if groups:
                group_note = ', '.join(f'{count} {key}' for key, count in sorted(groups.items()))
                return (
                    'polar',
                    JOSSI_UNIFAC_CLASSIFICATION_FACTOR,
                    f'Dortmund groups classified as polar ({group_note})',
                )
            heteroatoms = {
                element: count
                for element, count in counts.items()
                if element not in {'C', 'H'} and count
            }
            if heteroatoms:
                formula_note = ', '.join(
                    f'{element}{count}' for element, count in sorted(heteroatoms.items())
                )
                return (
                    'polar',
                    JOSSI_DEFAULT_POLAR_CLASSIFICATION_FACTOR,
                    f'unresolved structure contains heteroatoms ({formula_note}); classified as polar',
                )
            return (
                'polar',
                JOSSI_DEFAULT_POLAR_CLASSIFICATION_FACTOR,
                'structure polarity unresolved; defaulted to polar',
            )


        @staticmethod
        def _jossi_pressure_quality_factor(branch: str, reduced_density: float) -> float:
            rho_r = float(reduced_density)
            quality_points = (
                (
                    (0.10, 0.940),
                    (0.25, 0.915),
                    (0.50, 0.842),
                    (0.75, 0.785),
                    (1.00, 0.750),
                    (1.25, 0.730),
                    (1.50, 0.721),
                    (2.00, 0.542),
                    (2.60, 0.403),
                )
                if branch == 'nonpolar'
                else (
                    (0.10, 0.957),
                    (0.25, 0.921),
                    (0.50, 0.838),
                    (0.75, 0.688),
                    (1.00, 0.526),
                    (1.25, 0.429),
                    (1.50, 0.389),
                    (2.00, 0.314),
                    (2.60, 0.061),
                )
            )
            _, quality = min(
                quality_points,
                key=lambda item: abs(rho_r - item[0]),
            )
            return max(0.50, quality)


        @staticmethod
        def _jossi_dimensionless_increment(
            branch: str,
            reduced_density: float,
        ) -> Optional[float]:
            rho_r = float(reduced_density)
            if branch == 'nonpolar':
                if not 0.1 <= rho_r <= JOSSI_MAX_REDUCED_DENSITY:
                    return None
                polynomial = (
                    1.0230
                    + 0.23364 * rho_r
                    + 0.58533 * rho_r ** 2
                    - 0.40758 * rho_r ** 3
                    + 0.093324 * rho_r ** 4
                )
                return polynomial ** 4 - 1.0
            if branch != 'polar' or not 0.0 < rho_r <= JOSSI_MAX_REDUCED_DENSITY:
                return None
            if rho_r <= 0.1:
                return 1.656 * rho_r ** 1.111
            if rho_r <= 0.9:
                return 0.0607 * (9.045 * rho_r + 0.63) ** 1.739
            if rho_r <= 2.2:
                exponent = 0.6439 - 0.1005 * rho_r
            else:
                exponent = (
                    0.6439
                    - 0.1005 * rho_r
                    - 0.000475 * (rho_r ** 3 - 10.65) ** 2
                )
            return 10.0 ** (4.0 - 10.0 ** exponent)


        def _apply_jossi_pressure_correction(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            phase_key: str,
            baseline: PropertyResolutionResult,
            P: Optional[float],
            rho_molar: Optional[float],
        ) -> PropertyResolutionResult:
            if (
                P is None
                or rho_molar is None
                or phase_key not in {'gas', 'vapor', 'vapour', 'ideal_gas', 'ideal'}
                or baseline.method.startswith('coolprop')
            ):
                return baseline

            mw_result = self._source_result_for_value(props, 'MW', units='g/mol')
            if mw_result is None or mw_result.value is None:
                return baseline
            try:
                mw = float(mw_result.value)
                T = float(T)
            except (TypeError, ValueError):
                return baseline
            if mw <= 0.0 or T <= 0.0:
                return baseline

            density_state = self._viscosity_critical_density(symbol, props)
            if density_state is None:
                return baseline
            critical_density_result, critical, density_note = density_state
            tc_result = critical.get('Tc') if critical else None
            pc_result = critical.get('Pc') if critical else None
            if (
                tc_result is None or tc_result.value is None
                or pc_result is None or pc_result.value is None
            ):
                return baseline
            try:
                tc = float(tc_result.value)
                pc_bar = float(pc_result.value)
            except (TypeError, ValueError):
                return baseline
            if tc <= 0.0 or pc_bar <= 0.0:
                return baseline

            rho_r = float(rho_molar) / float(critical_density_result.value)
            screened_branch, classification_factor, classification_note = (
                self._jossi_polarity(symbol, props)
            )
            if screened_branch is None:
                return self._viscosity_result_with_note(
                    baseline,
                    f'Jossi pressure correction skipped: {classification_note}',
                )
            # scripts/vapor_viscosity/benchmark_jossi_polarity_coolprop.py
            # compared both equations over 56 fluids and a 4x9 (Tr, rho_r)
            # grid. Always-nonpolar reduced mean MAPE from 12.179% for the
            # structural split to 10.788% and raised decisive branch accuracy
            # from 44.6% to 78.6%; raw and reduced dipole thresholds both
            # independently collapsed to this same always-nonpolar policy.
            # Keep the structural screen only for association exclusions.
            branch = 'nonpolar'
            classification_note = (
                'nonpolar equation selected for every non-association-guarded '
                f'gas; structural screen: {classification_note}'
            )
            increment = self._jossi_dimensionless_increment(branch, rho_r)
            if increment is None or increment < 0.0 or not math.isfinite(increment):
                return self._viscosity_result_with_note(
                    baseline,
                    f'Jossi pressure correction outside {branch} reduced-density range (rho_r={rho_r:.3f})',
                )

            pc_mpa = pc_bar / 10.0
            try:
                xi = 2173.4 * tc ** (1.0 / 6.0) * mw ** -0.5 * pc_mpa ** (-2.0 / 3.0)
                value = float(baseline.value) + increment / xi / 1000.0
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return baseline
            if value <= 0.0 or not math.isfinite(value):
                return baseline

            branch_factor = self._jossi_pressure_quality_factor(branch, rho_r)
            quality = self._combine_quality(
                [baseline, mw_result, tc_result, pc_result, critical_density_result],
                method_factor=branch_factor * classification_factor,
            )
            return PropertyResolutionResult(
                value=value,
                source='calculated',
                method=f'jossi_stiel_thodos_{branch}',
                quality=quality,
                notes=(
                    f'Jossi-Stiel-Thodos {branch} dense-gas correction '
                    f'(rho={rho_molar:g} kmol/m^3, rho_r={rho_r:.3f}, xi={xi:.3f}); '
                    f'pressure-quality multiplier {branch_factor:.2f}; '
                    f'{classification_note}; {density_note}; low-pressure baseline from '
                    f'{baseline.source}/{baseline.method}; units Pa*s'
                ),
            )


        def _viscosity_formula_counts(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[Dict[str, int]]:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import get_compound_identity_resolver, parse_formula_counts
                else:
                    from compound_identity import get_compound_identity_resolver, parse_formula_counts
            except Exception:
                get_compound_identity_resolver = None
                parse_formula_counts = None
            if parse_formula_counts is None:
                return None

            candidates = [
                props.get('formula') if props else None,
                props.get('Formula') if props else None,
                symbol,
                props.get('symbol') if props else None,
                props.get('name') if props else None,
                props.get('CAS') if props else None,
                props.get('cas') if props else None,
            ]
            for candidate in candidates:
                if not candidate:
                    continue
                counts = parse_formula_counts(str(candidate))
                if counts:
                    return counts

            if get_compound_identity_resolver is None:
                return None
            try:
                resolver = get_compound_identity_resolver()
                for candidate in self._identifier_candidates(symbol, props):
                    identity = resolver.resolve(candidate, allow_formula=False)
                    if identity and identity.formula:
                        counts = parse_formula_counts(identity.formula)
                        if counts:
                            return counts
            except Exception:
                return None
            return None
