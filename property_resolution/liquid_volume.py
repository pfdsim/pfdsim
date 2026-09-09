from .common import *
from collections.abc import Iterable, Mapping
import sqlite3

from .coolprop import (
    COOLPROP_PROPERTY_QUALITY,
    coolprop_module,
)


if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..physical_constants import R_BAR_CM3_MOL_K
else:
    from physical_constants import R_BAR_CM3_MOL_K

class LiquidVolumeMixin:
        PUBCHEM_DENSITY_CACHE_VERSION = 2
        ASSUMED_BARE_DENSITY_TEMPERATURE_K = 293.15
        DENSITY_PHASE_MARGIN_K = 2.0
        DENSITY_PHASE_MARGIN_REL = 0.01
        SOLID_DENSITY_HEURISTIC_AUTOMORPHISM_LIMIT = 10
        SOLID_DENSITY_HEURISTIC_MIN_REDUCED_TEMPERATURE = 0.30
        SOLID_DENSITY_HEURISTIC_LIQUID_MIN_QUALITY = 0.55
        # Organic-crystal volumetric expansion heuristic.  van der Lee and
        # Dumitrescu, "Thermal expansion properties of organic crystals: a
        # CSD study", Chem. Sci. 2021, 12, 8537-8547,
        # DOI 10.1039/D1SC01076J.  The high-quality 745-observation subset
        # reports mean alpha_V = 168.8e-6 K^-1 (rounded here to 1.7e-4) and
        # sigma = 72.5e-6 K^-1.  This broad distribution is why extrapolation
        # carries an explicit distance-based quality penalty.
        SOLID_ORGANIC_VOLUMETRIC_EXPANSION_K_INV = 1.7e-4
        SOLID_ORGANIC_EXPANSION_QUALITY_PENALTY_PER_5K = 0.01
        SOLID_OTHER_QUALITY_PENALTY_PER_5K = 0.02

        @staticmethod
        def _positive_liquid_volume(value: Optional[float]) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            if 0.005 <= value_f <= 10.0:
                return value_f
            return None


        def _coolprop_saturated_liquid_volume(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            """Saturated-liquid molar volume from a CoolProp reference EOS."""
            if self._is_pfd_correlation_override(props, 'rhol'):
                # PFD components may deliberately impersonate a real chemical
                # (shared CAS) while overriding its behavior; their explicit
                # correlations outrank the reference EOS.
                return None
            reference = self._coolprop_reference(symbol, props)
            if reference is None:
                return None
            CP = coolprop_module()
            if CP is None:
                return None
            try:
                rhomolar = float(CP.PropsSI(
                    'Dmolar', 'T', float(T), 'Q', 0.0, reference.qualified_name,
                ))
            except Exception:
                return None
            if not math.isfinite(rhomolar) or rhomolar <= 0.0:
                return None
            volume = self._positive_liquid_volume(1000.0 / rhomolar)
            if volume is None:
                return None
            return PropertyResolutionResult(
                value=volume,
                source='local',
                method=f'{reference.method_prefix}_saturated_liquid_volume',
                quality=COOLPROP_PROPERTY_QUALITY,
                notes=(
                    f'CoolProp {reference.backend} saturated-liquid molar volume '
                    f'for {reference.fluid}; units m^3/kmol'
                ),
            )


        def _critical_results_for_liquid_volume(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Dict[str, PropertyResolutionResult]:
            try:
                return self.resolve_critical_properties(
                    symbol,
                    props,
                    allow_online=False,
                    allow_estimation=True,
                )
            except Exception:
                return {}


        def _critical_value_result(
            self,
            critical: Dict[str, PropertyResolutionResult],
            key: str,
        ) -> Optional[PropertyResolutionResult]:
            result = critical.get(key)
            if result and result.value is not None:
                return result
            return None


        def _rackett_volume_m3_per_kmol(
            self,
            T: float,
            Tc: float,
            Pc_bar: float,
            Zra: float,
        ) -> Optional[float]:
            try:
                T = float(T)
                Tc = float(Tc)
                Pc_bar = float(Pc_bar)
                Zra = float(Zra)
                if T <= 0.0 or Tc <= 0.0 or Pc_bar <= 0.0 or not (0.05 <= Zra <= 0.45):
                    return None
                Tr = min(T / Tc, 0.999999)
                exponent = 1.0 + max(0.0, 1.0 - Tr) ** (2.0 / 7.0)
                volume_cm3_mol = R_BAR_CM3_MOL_K * Tc / Pc_bar * (Zra ** exponent)
                return self._positive_liquid_volume(volume_cm3_mol / 1000.0)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None


        @staticmethod
        def _yamada_gunn_zra(omega: float) -> Optional[float]:
            try:
                value = 0.29056 - 0.08775 * float(omega)
            except (TypeError, ValueError):
                return None
            if 0.05 <= value <= 0.45:
                return value
            return None


        def _provided_rhol_volume_at(
            self,
            props: Dict[str, Any],
            T: float,
            enforce_range: bool = True,
        ) -> Optional[tuple[float, Dict[str, Any]]]:
            evaluated = self._evaluate_provided_correlation(
                props,
                'rhol',
                T,
                enforce_range=enforce_range,
            )
            if not evaluated:
                return None
            rho_kg_m3, correlation = evaluated
            mw = props.get('MW')
            try:
                volume = float(mw) / float(rho_kg_m3)
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            volume = self._positive_liquid_volume(volume)
            if volume is None:
                return None
            return volume, correlation


        def _provided_rhol_small_extrapolation(
            self,
            props: Dict[str, Any],
            T: float,
            limit: float = LIQUID_VOLUME_EXTRAPOLATION_LIMIT_K,
        ) -> Optional[tuple[float, Dict[str, Any], float]]:
            correlation = self._correlation_for(props, 'rhol')
            if not correlation or self._correlation_in_range(correlation, T):
                return None
            Tmin = correlation.get('Tmin_K')
            Tmax = correlation.get('Tmax_K')
            if Tmin is None or Tmax is None:
                return None
            try:
                Tmin = float(Tmin)
                Tmax = float(Tmax)
                T = float(T)
            except (TypeError, ValueError):
                return None
            distance = Tmin - T if T < Tmin else T - Tmax
            if distance < -1e-9 or distance > limit + 1e-9:
                return None
            evaluated = self._provided_rhol_volume_at(props, T, enforce_range=False)
            if not evaluated:
                return None
            volume, used_correlation = evaluated
            return volume, used_correlation, T


        def _perry_liquid_volume_small_extrapolation(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            limit: float = LIQUID_VOLUME_EXTRAPOLATION_LIMIT_K,
        ) -> Optional[tuple[float, Any, float]]:
            library = self._get_perry_library()
            if library is None:
                return None
            candidates = []
            for candidate in self._identifier_candidates(symbol, props):
                entry = library.get(candidate)
                if not entry:
                    continue
                for row in entry.get('liquid_density', []) or []:
                    if row.get('equation_id') not in {100, 105}:
                        continue
                    boundary_T = self._perry_row_endpoint(row, T)
                    if boundary_T is None:
                        continue
                    distance = abs(float(boundary_T) - float(T))
                    if distance > limit + 1e-9:
                        continue
                    try:
                        width = float(row.get('T_max_K', boundary_T)) - float(row.get('T_min_K', boundary_T))
                    except (TypeError, ValueError):
                        width = math.inf
                    candidates.append((distance, width, row))
            candidates.sort(key=lambda item: item[:2])
            for _, _, row in candidates:
                value = library._eval_liquid_density_mol_per_dm3(row, T)
                if value is None or value <= 0.0:
                    continue
                volume = self._positive_liquid_volume(1.0 / value)
                if volume is not None:
                    return volume, row, float(T)
            return None


        def _load_liquid_volume_zra_cache(self) -> Dict[str, Any]:
            if self._liquid_volume_zra_cache is not None:
                return self._liquid_volume_zra_cache
            store = self._liquid_volume_zra_store()
            self._migrate_liquid_volume_zra_json(store)
            payload = {
                'version': 1,
                'fits': dict(store.items()),
            }
            self._liquid_volume_zra_cache = payload
            return payload


        def _save_liquid_volume_zra_cache(self) -> None:
            payload = self._liquid_volume_zra_cache
            if payload is None:
                return
            try:
                store = self._liquid_volume_zra_store()
                for key, fit in (payload.get('fits') or {}).items():
                    if isinstance(fit, dict):
                        store.set(str(key), fit)
            except (OSError, sqlite3.Error, TypeError, ValueError):
                pass


        def _liquid_volume_zra_store(self):
            from .runtime_cache import (
                SQLiteJSONCache,
                runtime_cache_path_for_legacy_file,
            )

            default_legacy_path = (
                Path(__file__).resolve().parent.parent
                / 'data' / 'liquid_volume_zra_cache.json'
            )
            path = runtime_cache_path_for_legacy_file(
                self.LIQUID_VOLUME_ZRA_CACHE_PATH,
                default_legacy_file=default_legacy_path,
            )
            return SQLiteJSONCache(path, 'liquid_volume_zra_v1')


        def _migrate_liquid_volume_zra_json(self, store) -> None:
            path = Path(self.LIQUID_VOLUME_ZRA_CACHE_PATH)
            if path.suffix.lower() != '.json' or not path.is_file():
                return
            try:
                payload = json.loads(path.read_text())
                fits = payload.get('fits') if isinstance(payload, dict) else None
            except (OSError, TypeError, ValueError):
                return
            if not isinstance(fits, dict):
                return
            for key, fit in fits.items():
                if not isinstance(fit, dict) or store.get(str(key)) is not None:
                    continue
                store.set(str(key), fit)


        @staticmethod
        def _liquid_volume_cache_key(
            symbol: str,
            kind: str,
            correlation: Dict[str, Any],
            Tc: float,
            Pc: float,
        ) -> str:
            signature = {
                'symbol': str(symbol),
                'kind': kind,
                'correlation': correlation,
                'Tc': round(float(Tc), 8),
                'Pc': round(float(Pc), 8),
            }
            payload = json.dumps(signature, sort_keys=True, default=str)
            digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]
            return f"{kind}:{digest}"


        def _fit_zra_from_samples(
            self,
            samples: list[tuple[float, float]],
            Tc: float,
            Pc: float,
        ) -> Optional[float]:
            log_values = []
            base = R_BAR_CM3_MOL_K * float(Tc) / float(Pc)
            if base <= 0.0:
                return None
            for sample_T, volume_m3_kmol in samples:
                volume = self._positive_liquid_volume(volume_m3_kmol)
                if volume is None:
                    continue
                Tr = min(float(sample_T) / float(Tc), 0.999999)
                exponent = 1.0 + max(0.0, 1.0 - Tr) ** (2.0 / 7.0)
                ratio = volume * 1000.0 / base
                if ratio <= 0.0 or exponent <= 0.0:
                    continue
                zra = ratio ** (1.0 / exponent)
                if 0.05 <= zra <= 0.45:
                    log_values.append(math.log(zra))
            if not log_values:
                return None
            return math.exp(sum(log_values) / len(log_values))


        @staticmethod
        def _sample_temperatures(Tmin: float, Tmax: float, count: int = 15) -> list[float]:
            if not math.isfinite(Tmin) or not math.isfinite(Tmax) or Tmax < Tmin:
                return []
            if abs(Tmax - Tmin) < 1e-9:
                return [Tmin]
            count = max(2, count)
            return [Tmin + (Tmax - Tmin) * i / (count - 1) for i in range(count)]


        def _fitted_zra_result(
            self,
            symbol: str,
            props: Dict[str, Any],
            critical: Dict[str, PropertyResolutionResult],
        ) -> Optional[PropertyResolutionResult]:
            Tc_result = self._critical_value_result(critical, 'Tc')
            Pc_result = self._critical_value_result(critical, 'Pc')
            if not Tc_result or not Pc_result:
                return None
            Tc = float(Tc_result.value)
            Pc = float(Pc_result.value)
            fit_candidates = []

            correlation = self._correlation_for(props, 'rhol')
            if correlation and props.get('MW'):
                if str(correlation.get('equation', '')).lower() == 'density_reference':
                    try:
                        rho_kg_m3 = float(correlation.get('rho_kg_m3'))
                        T_ref = float(correlation.get('T_ref_K', REFERENCE_TEMPERATURE_K))
                        volume = self._positive_liquid_volume(float(props.get('MW')) / rho_kg_m3)
                    except (TypeError, ValueError, ZeroDivisionError):
                        volume = None
                    if volume is not None:
                        key = self._liquid_volume_cache_key(symbol, 'provided_rho_reference', correlation, Tc, Pc)
                        fit_candidates.append((
                            key,
                            [(T_ref, volume)],
                            'provided_rho_reference',
                            correlation,
                            self._clamp_quality(correlation.get('quality'), 1.0),
                        ))
                else:
                    Tmin = correlation.get('Tmin_K')
                    Tmax = correlation.get('Tmax_K')
                    try:
                        Tmin_f = float(Tmin)
                        Tmax_f = float(Tmax)
                    except (TypeError, ValueError):
                        Tmin_f = Tmax_f = None
                    if Tmin_f is not None and Tmax_f is not None:
                        key = self._liquid_volume_cache_key(symbol, 'provided_rhol', correlation, Tc, Pc)
                        samples = []
                        for sample_T in self._sample_temperatures(Tmin_f, Tmax_f):
                            evaluated = self._provided_rhol_volume_at(props, sample_T, enforce_range=False)
                            if evaluated:
                                samples.append((sample_T, evaluated[0]))
                        fit_candidates.append((key, samples, 'provided_rhol', correlation, 0.98))

            library = self._get_perry_library()
            if library is not None:
                for candidate in self._identifier_candidates(symbol, props):
                    entry = library.get(candidate)
                    if not entry:
                        continue
                    for row in entry.get('liquid_density', []) or []:
                        if row.get('equation_id') not in {100, 105}:
                            continue
                        Tmin = row.get('T_min_K')
                        Tmax = row.get('T_max_K')
                        try:
                            Tmin_f = float(Tmin)
                            Tmax_f = float(Tmax)
                        except (TypeError, ValueError):
                            continue
                        key = self._liquid_volume_cache_key(candidate, 'perry_density', row, Tc, Pc)
                        samples = []
                        for sample_T in self._sample_temperatures(Tmin_f, Tmax_f):
                            density = library._eval_liquid_density_mol_per_dm3(row, sample_T)
                            if density and density > 0.0:
                                volume = self._positive_liquid_volume(1.0 / density)
                                if volume is not None:
                                    samples.append((sample_T, volume))
                        fit_candidates.append((key, samples, 'perry_density', row, 0.98))

            cache = self._load_liquid_volume_zra_cache()
            fits = cache.setdefault('fits', {})
            override_influenced = (
                self._is_pfd_correlation_override(props, 'rhol')
                or self._is_pfd_component_override(props, 'Tc')
                or self._is_pfd_component_override(props, 'Pc')
            )
            session_fits = getattr(
                self,
                '_override_liquid_volume_zra_fits',
                None,
            )
            if session_fits is None:
                session_fits = {}
                self._override_liquid_volume_zra_fits = session_fits
            for key, samples, kind, source_correlation, source_quality in fit_candidates:
                fit_store = session_fits if override_influenced else fits
                cached = fit_store.get(key)
                zra = None
                if isinstance(cached, dict):
                    try:
                        zra = float(cached.get('Z_RA'))
                    except (TypeError, ValueError):
                        zra = None
                if zra is None or not (0.05 <= zra <= 0.45):
                    zra = self._fit_zra_from_samples(samples, Tc, Pc)
                    if zra is None:
                        continue
                    fit_store[key] = {
                        'Z_RA': zra,
                        'kind': kind,
                        'sample_count': len(samples),
                        'source': source_correlation.get('source') if isinstance(source_correlation, dict) else '',
                    }
                    if not override_influenced:
                        self._save_liquid_volume_zra_cache()
                source_result = PropertyResolutionResult(
                    value=zra,
                    source='local' if kind == 'perry_density' else 'provided',
                    method=f'{kind}_zra_fit_source',
                    quality=source_quality,
                    notes=f"Z_RA fit source from {kind}",
                )
                inputs = [source_result, Tc_result, Pc_result]
                return PropertyResolutionResult(
                    value=zra,
                    source='calculated',
                    method=f'fitted_rackett_zra_from_{kind}',
                    quality=self._combine_quality(inputs, method_factor=0.94),
                    notes=f"Z_RA fitted once from {kind} liquid-density correlation; cache key {key}",
                )
            return None


        def _rackett_fitted_volume(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any],
            critical: Dict[str, PropertyResolutionResult],
        ) -> Optional[PropertyResolutionResult]:
            Tc_result = self._critical_value_result(critical, 'Tc')
            Pc_result = self._critical_value_result(critical, 'Pc')
            zra_result = self._fitted_zra_result(symbol, props, critical)
            if not Tc_result or not Pc_result or not zra_result:
                return None
            value = self._rackett_volume_m3_per_kmol(T, Tc_result.value, Pc_result.value, zra_result.value)
            if value is None:
                return None
            inputs = [Tc_result, Pc_result, zra_result]
            return PropertyResolutionResult(
                value=value,
                source='calculated',
                method='rackett_fitted_zra',
                quality=self._combine_quality(inputs, exact_formula=True),
                notes=f"Rackett liquid volume using {zra_result.method}; {zra_result.notes}",
            )


        @staticmethod
        def _normalize_density_text(text: Any) -> str:
            normalized = (
                html_module.unescape(str(text or ''))
                .replace('−', '-')
                .replace('–', '-')
                .replace('—', '-')
                .replace('Â°', '°')
                .replace('\xa0', ' ')
                .replace(',', '')
            )
            normalized = (
                normalized
                .replace('⁻³', '-3')
                .replace('⁻¹', '-1')
                .replace('−³', '-3')
                .replace('−¹', '-1')
                .replace('³', '^3')
                .replace('²', '^2')
                .replace('·', ' ')
            )
            scientific = re.compile(
                r'(?<![\w.])((?:\d+(?:\.\d*)?|\.\d+))\s*'
                r'(?:x|×)\s*10\s*(?:\^\s*)?([+-]?\d+)',
                re.I,
            )
            return scientific.sub(
                lambda match: f'{float(match.group(1)) * 10.0 ** int(match.group(2)):g}',
                normalized,
            )


        @classmethod
        def _density_temperature_from_text(cls, text: str) -> tuple[Optional[float], bool]:
            """Return the sample temperature, not a specific-gravity water reference."""
            text = cls._normalize_density_text(text)
            number = r'-?(?:\d+(?:\.\d*)?|\.\d+)'
            unit = r'(?:°\s*|deg(?:ree)?s?\s*)?([CFK])\b'
            # ``20 °C/4 °C`` and ``20/4 °C`` mean sample at 20 °C with
            # water at 4 °C as the reference; the first number is the sample.
            temperature_range = re.search(
                rf'(?:\bat\b|@|\btemperature\b\s*(?:of|=|:)?)\s*'
                rf'({number})\s*(?:-|to)\s*({number})\s*{unit}',
                text,
                re.I,
            )
            temperature_uncertainty = re.search(
                rf'(?:\bat\b|@|\btemperature\b\s*(?:of|=|:)?)\s*'
                rf'({number})\s*(?:±|\+/-)\s*{number}\s*{unit}',
                text,
                re.I,
            )
            temperature_parenthetical = re.search(
                rf'(?:\bat\b|@|\btemperature\b\s*(?:of|=|:)?)\s*'
                rf'({number})\s*\(\d+\)\s*{unit}',
                text,
                re.I,
            )
            ratio = re.search(
                rf'({number})\s*(?:°\s*)?C\s*/\s*{number}\s*(?:°\s*)?C\b',
                text,
                re.I,
            ) or re.search(
                rf'({number})\s*/\s*{number}\s*(?:°\s*)?C\b',
                text,
                re.I,
            )
            if temperature_range:
                value = 0.5 * (
                    float(temperature_range.group(1))
                    + float(temperature_range.group(2))
                )
                unit_name = temperature_range.group(3).upper()
                ratio = None
                density_notation = None
            elif temperature_uncertainty:
                value = float(temperature_uncertainty.group(1))
                unit_name = temperature_uncertainty.group(2).upper()
                ratio = None
                density_notation = None
            elif temperature_parenthetical:
                value = float(temperature_parenthetical.group(1))
                unit_name = temperature_parenthetical.group(2).upper()
                ratio = None
                density_notation = None
            elif ratio:
                value = float(ratio.group(1))
                unit_name = 'C'
            else:
                density_notation = re.search(
                    rf'\bd\s*\(?\s*({number})(?:\s*/\s*{number})?\s*\)?',
                    text,
                    re.I,
                )
                if density_notation and -273.15 < float(density_notation.group(1)) < 200.0:
                    value = float(density_notation.group(1))
                    unit_name = 'C'
                else:
                    density_notation = None
            if (
                not temperature_range
                and not temperature_uncertainty
                and not temperature_parenthetical
                and not ratio
                and not density_notation
            ):
                match = re.search(
                    rf'(?:\bat\b|@|\btemperature\b\s*(?:of|=|:)?|[(:])?\s*'
                    rf'({number})\s*{unit}',
                    text,
                    re.I,
                )
                if not match:
                    return None, False
                value = float(match.group(1))
                unit_name = match.group(2).upper()
            if unit_name == 'K':
                temperature = value
            elif unit_name == 'C':
                temperature = value + 273.15
            else:
                temperature = (value - 32.0) * 5.0 / 9.0 + 273.15
            if not (0.0 < temperature < 5000.0):
                return None, True
            return temperature, False


        @staticmethod
        def _relative_density_number(text: str) -> Optional[float]:
            text = re.sub(r'water\s*=\s*1', 'water_reference', text, flags=re.I)
            text = re.sub(
                r'\bd\s*\(?\s*-?\d+(?:\.\d+)?'
                r'(?:\s*/\s*-?\d+(?:\.\d+)?)?\s*\)?',
                ' density_temperature_reference ',
                text,
                flags=re.I,
            )
            tail = text.split(':', 1)[1] if ':' in text else text
            values = [
                float(match)
                for match in re.findall(
                    r'(?<![\w.])-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',
                    tail,
                )
            ]
            values = [value for value in values if 0.05 <= value <= 25.0]
            return values[0] if values else None


        @staticmethod
        def _pubchem_density_reference_quality(info: Dict[str, Any]) -> float:
            quality = 0.0
            if info.get('Reference'):
                quality += 0.01
            description = str(info.get('Description') or '').lower()
            if 'peer reviewed' in description:
                quality += 0.02
            return quality


        @staticmethod
        def _pubchem_density_reference(info: Dict[str, Any]) -> str:
            references = info.get('Reference') or []
            if isinstance(references, str):
                return references.strip()
            if isinstance(references, dict):
                return str(references.get('SourceName') or references.get('Name') or '').strip()
            if isinstance(references, (list, tuple)) and references:
                first = references[0]
                if isinstance(first, dict):
                    return str(first.get('SourceName') or first.get('Name') or '').strip()
                return str(first).strip()
            return ''


        @staticmethod
        def _density_explicit_phase(text: str) -> str:
            lower = str(text or '').lower()
            solid = bool(re.search(
                r'\b(?:solid(?:\s+phase)?|crystals?|crystalline|ice)\b',
                lower,
            ))
            liquid = bool(re.search(
                r'\b(?:liquid(?:\s+phase)?|molten|melt\s+phase)\b',
                lower,
            ))
            if solid == liquid:
                return 'ambiguous'
            return 'solid' if solid else 'liquid'


        @staticmethod
        def _density_crystal_form(text: str) -> str:
            lower = (
                str(text or '').lower()
                .replace('β', 'beta')
                .replace('α', 'alpha')
                .replace('γ', 'gamma')
                .replace('δ', 'delta')
            )
            match = re.search(
                r'\b(alpha|beta|gamma|delta|orthorhombic|rhombic|monoclinic|'
                r'triclinic|tetragonal|hexagonal|cubic|amorphous)\b',
                lower,
            )
            return match.group(1) if match else ''


        @staticmethod
        def _density_report_is_intrinsic(text: str) -> bool:
            lower = str(text or '').lower()
            rejected = (
                'table#', 'conversion', 'ppm', 'specific heat', 'latent heat',
                'viscosity', 'specific volume', 'concn', 'azeotrope', 'wt/wt',
                'wt %', 'wt%', 'vol %', 'vol%', '% by', 'percent', 'aqueous',
                'solution', 'sea water', 'commercial formulation', 'spirit',
                'dynamite', '/usp', '/commercial', 'polymer', 'slurry',
                'suspension', 'dust', 'powder', 'bulk density', 'bulk specific',
                'tapped density', 'tap density', 'apparent density',
                'loose density', 'packed density', 'vapor density',
                'vapour density', 'gas density', 'relative to air', 'air = 1',
            )
            if any(token in lower for token in rejected):
                return False
            if re.search(r'\b(?:estimated|estimate|predicted|prediction)\b', lower):
                return False
            if re.search(r'\b(?:vapor|vapour|gas)\b', lower) and 'liquid' not in lower:
                return False
            return True


        @classmethod
        def _density_value_from_clause(
            cls,
            clause: str,
        ) -> tuple[Optional[float], str, float]:
            """Return intrinsic density in g/cm^3, parse kind, and base quality."""
            number = r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
            unit_patterns = (
                (
                    r'g\s*(?:/\s*(?:cu\s*cm|cm3|cm\^3|cm³|m[lL]|cc)'
                    r'|(?:cm|cc|m[lL])\s*(?:-3|-1|\^-?3))\b',
                    1.0,
                    0.90,
                    'g/cm3',
                ),
                (
                    r'kg\s*(?:/\s*(?:cu\s*m|m3|m\^3|m³)'
                    r'|m\s*(?:-3|\^-?3))\b',
                    0.001,
                    0.90,
                    'kg/m3',
                ),
                (r'kg\s*/\s*(?:d[mM]3|d[mM]\^3|[lL])\b', 1.0, 0.90, 'kg/L'),
                (r'g\s*/\s*[lL]\b', 0.001, 0.88, 'g/L'),
                (r'lb\s*/\s*(?:u\.?s\.?\s*)?gal\b', 0.119826427, 0.84, 'lb/gal'),
                (r'lb\s*/\s*(?:cu\s*ft|ft3|ft\^3|ft³)\b', 0.016018463, 0.84, 'lb/ft3'),
            )
            for unit_pattern, scale, quality, kind in unit_patterns:
                uncertainty_match = re.search(
                    rf'(?<![\w.])({number})\s*(?:±|\+/-)\s*{number}\s*{unit_pattern}',
                    clause,
                    re.I,
                )
                if uncertainty_match:
                    value = float(uncertainty_match.group(1)) * scale
                    if 0.05 <= value <= 25.0:
                        return value, f'{kind}_uncertainty', quality - 0.01
                parenthetical_match = re.search(
                    rf'(?<![\w.])({number})\s*\(\d+\)\s*{unit_pattern}',
                    clause,
                    re.I,
                )
                if parenthetical_match:
                    value = float(parenthetical_match.group(1)) * scale
                    if 0.05 <= value <= 25.0:
                        return value, f'{kind}_uncertainty', quality
                range_match = re.search(
                    rf'(?<![\w.])({number})\s*(?:-|to)\s*({number})\s*{unit_pattern}',
                    clause,
                    re.I,
                )
                if range_match:
                    lower = float(range_match.group(1)) * scale
                    upper = float(range_match.group(2)) * scale
                    if 0.05 <= lower <= 25.0 and 0.05 <= upper <= 25.0:
                        return 0.5 * (lower + upper), f'{kind}_range', quality - 0.02
                match = re.search(
                    rf'(?<![\w.])({number})\s*{unit_pattern}',
                    clause,
                    re.I,
                )
                if match:
                    value = float(match.group(1)) * scale
                    if 0.05 <= value <= 25.0:
                        return value, kind, quality

            lower_clause = clause.lower()
            if (
                'relative density' in lower_clause
                or 'specific gravity' in lower_clause
                or re.search(r'water\s*=\s*1', lower_clause)
            ):
                value = cls._relative_density_number(clause)
                if value is not None:
                    return value, 'relative', 0.84

            range_match = re.search(
                rf'(?<![\w.])({number})\s*(?:-|to)\s*({number})(?![\w.])',
                clause,
                re.I,
            )
            if range_match:
                lower = float(range_match.group(1))
                upper = float(range_match.group(2))
                if 0.05 <= lower <= 25.0 and 0.05 <= upper <= 25.0:
                    return 0.5 * (lower + upper), 'range', 0.82

            # Remove temperatures and common reference notations before using
            # a bare number, so 20 °C/4 °C cannot become a density value.
            stripped = re.sub(
                rf'-?{number}\s*(?:°\s*|deg(?:ree)?s?\s*)?[CFK]\b',
                ' ',
                clause,
                flags=re.I,
            )
            stripped = re.sub(r'water\s*=\s*1', 'water_reference', stripped, flags=re.I)
            values = [
                float(match.group(0))
                for match in re.finditer(rf'(?<![\w.])-?{number}(?![\w.])', stripped)
                if 0.05 <= float(match.group(0)) <= 25.0
            ]
            if values and ('density' in lower_clause or re.match(r'^\s*\d', clause)):
                return values[0], 'bare', 0.84
            return None, '', 0.0


        def _parse_pubchem_density_text(
            self,
            text: str,
            info: Optional[Dict[str, Any]] = None,
            *,
            identity_text: str = '',
        ) -> list[Dict[str, Any]]:
            """Return phase-neutral, normalized PubChem intrinsic densities."""
            text = self._normalize_density_text(text)
            info = info or {}
            reference = self._pubchem_density_reference(info)
            comment = str(info.get('Description') or info.get('Name') or '').strip()
            reference_quality = self._pubchem_density_reference_quality(info)
            records = []
            for clause in [part.strip() for part in re.split(r';|\|', text) if part.strip()]:
                combined = f'{comment} {clause}'.strip()
                if not self._density_report_is_intrinsic(combined):
                    continue
                temperature, invalid_temperature = self._density_temperature_from_text(clause)
                if invalid_temperature:
                    continue
                value, kind, quality = self._density_value_from_clause(clause)
                if value is None:
                    continue
                if temperature is None:
                    quality -= 0.02
                quality = self._clamp_quality(quality + reference_quality, 0.80)
                calculated = bool(re.search(r'\b(?:calculated|computed)\b', combined, re.I))
                if calculated:
                    quality *= 0.95
                form = classify_fusion_material_form(combined, identity_text)
                polymorph = form[2] or self._density_crystal_form(combined)
                labels = [label for label in form[1].split('/') if label]
                if polymorph and polymorph not in labels:
                    labels.append(polymorph)
                records.append({
                    'mass_density_kg_m3': float(value) * 1000.0,
                    'temperature_K': temperature,
                    'phase_hint': self._density_explicit_phase(combined),
                    'material_form': form[0],
                    'form_label': '/'.join(labels),
                    'polymorph': polymorph,
                    'stereochemistry': form[3],
                    'is_identity_form': form[4],
                    'source': 'pubchem',
                    'method': (
                        'pubchem_calculated_crystal_density'
                        if calculated and self._density_explicit_phase(combined) == 'solid'
                        else 'pubchem_reported_density'
                    ),
                    'quality': max(0.76, min(0.94, quality)),
                    'reference': reference,
                    'comment': comment,
                    'raw': clause,
                    'metadata': {
                        'parse_kind': kind,
                        'source_value_kind': 'calculated' if calculated else 'reported',
                    },
                })
            return records


        @classmethod
        def _trusted_density_phase_point(
            cls,
            props: Mapping[str, Any],
            key: str,
        ) -> Optional[float]:
            value = props.get(key)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value) or value <= 0.0:
                return None
            source = (props.get('property_sources') or {}).get(key) or {}
            if key == 'Tb':
                if isinstance(source, Mapping) and cls._meta_quality(dict(source), 1.0) < 0.80:
                    return None
                return value
            if isinstance(source, Mapping) and cls._source_meta_is_soft(dict(source)):
                return None
            text = ' '.join(
                str(source.get(field) or '').lower()
                for field in ('source', 'method', 'notes')
            ) if isinstance(source, Mapping) else ''
            if any(token in text for token in (
                'estimated', 'estimate', 'provisional', 'nannoolal',
                'joback', 'guldberg', 'correlation fallback',
            )):
                return None
            return value


        @classmethod
        def _classify_density_phase(
            cls,
            record: Mapping[str, Any],
            props: Mapping[str, Any],
        ) -> tuple[str, str, float]:
            hint = str(record.get('phase_hint') or 'ambiguous').lower()
            if hint in {'solid', 'liquid'}:
                return hint, f'explicit_{hint}_wording', 1.0

            temperature = record.get('temperature_K')
            try:
                temperature = float(temperature) if temperature is not None else None
            except (TypeError, ValueError):
                temperature = None
            tm = cls._trusted_density_phase_point(props, 'Tm')
            tb = cls._trusted_density_phase_point(props, 'Tb')
            if temperature is not None and tm is not None:
                margin = max(cls.DENSITY_PHASE_MARGIN_K, cls.DENSITY_PHASE_MARGIN_REL * tm)
                if temperature < tm - margin:
                    return 'solid', f'T={temperature:g} K below hard Tm={tm:g} K', 0.98
                if temperature > tm + margin:
                    if tb is not None:
                        boiling_margin = max(
                            cls.DENSITY_PHASE_MARGIN_K,
                            cls.DENSITY_PHASE_MARGIN_REL * tb,
                        )
                        if temperature < tb - boiling_margin:
                            return 'liquid', (
                                f'T={temperature:g} K between hard Tm={tm:g} K '
                                f'and Tb={tb:g} K'
                            ), 0.98
                        return 'ambiguous', (
                            f'T={temperature:g} K is not safely inside the condensed-liquid '
                            f'range bounded by Tm={tm:g} K and Tb={tb:g} K'
                        ), 1.0
                    phase_at_stp = str(
                        props.get('phase_at_STP') or props.get('phase') or ''
                    ).strip().lower()
                    if (
                        phase_at_stp == 'liquid'
                        and abs(temperature - REFERENCE_TEMPERATURE_K) <= 25.0
                    ):
                        return 'liquid', (
                            f'T={temperature:g} K above hard Tm={tm:g} K; '
                            'near-ambient phase_at_STP=liquid'
                        ), 0.96
                    return 'ambiguous', (
                        f'T={temperature:g} K is above hard Tm={tm:g} K but '
                        'no trustworthy Tb bounds the liquid range'
                    ), 1.0
                return 'ambiguous', (
                    f'T={temperature:g} K lies within {margin:g} K of hard Tm={tm:g} K'
                ), 1.0

            phase_at_stp = str(
                props.get('phase_at_STP') or props.get('phase') or ''
            ).strip().lower()
            near_reference = (
                temperature is None
                or abs(temperature - REFERENCE_TEMPERATURE_K) <= 25.0
            )
            if near_reference and phase_at_stp in {'solid', 'liquid'}:
                basis = 'temperatureless' if temperature is None else f'T={temperature:g} K'
                return phase_at_stp, f'{basis}; phase_at_STP={phase_at_stp}', 0.96
            return 'ambiguous', 'no explicit phase or trustworthy phase boundary', 1.0


        def _density_observation_from_record(
            self,
            record: Mapping[str, Any],
            props: Mapping[str, Any],
            identity_text: str,
        ) -> Optional[DensityObservation]:
            try:
                phase, phase_basis, phase_factor = self._classify_density_phase(record, props)
                form = classify_fusion_material_form(
                    f"{record.get('comment') or ''} {record.get('raw') or ''}",
                    identity_text,
                )
                identity_form = classify_fusion_material_form(
                    identity_text,
                    identity_text,
                )
                material_form = form[0]
                is_identity_form = form[4]
                if (
                    material_form == 'unspecified'
                    and identity_form[0] in {'hydrate', 'solvate'}
                ):
                    material_form = identity_form[0]
                    is_identity_form = True
                polymorph = (
                    form[2]
                    or str(record.get('polymorph') or '')
                    or self._density_crystal_form(
                        f"{record.get('comment') or ''} {record.get('raw') or ''}"
                    )
                )
                stereochemistry = form[3] or str(record.get('stereochemistry') or '')
                labels = []
                if material_form != 'unspecified':
                    labels.append(material_form)
                if polymorph:
                    labels.append(polymorph)
                if stereochemistry:
                    labels.append(stereochemistry)
                metadata = dict(record.get('metadata') or {})
                metadata['phase_hint'] = str(record.get('phase_hint') or 'ambiguous')
                if material_form != form[0]:
                    metadata['material_form_inherited_from_identity'] = True
                return DensityObservation(
                    mass_density_kg_m3=record.get('mass_density_kg_m3'),
                    temperature_K=record.get('temperature_K'),
                    phase=phase,
                    phase_basis=phase_basis,
                    material_form=material_form,
                    form_label='/'.join(labels),
                    polymorph=polymorph,
                    stereochemistry=stereochemistry,
                    is_identity_form=is_identity_form,
                    source=str(record.get('source') or 'pubchem'),
                    method=str(record.get('method') or 'pubchem_reported_density'),
                    quality=self._clamp_quality(record.get('quality'), 0.80) * phase_factor,
                    reference=str(record.get('reference') or ''),
                    comment=str(record.get('comment') or ''),
                    raw=str(record.get('raw') or ''),
                    metadata=metadata,
                )
            except (TypeError, ValueError):
                return None


        def _parse_pubchem_liquid_density_text(
            self,
            text: str,
            info: Optional[Dict[str, Any]] = None,
            props: Optional[Mapping[str, Any]] = None,
        ) -> list[Dict[str, Any]]:
            """Compatibility adapter returning only phase-qualified liquid points."""
            props = dict(props or {})
            identity_text = ' '.join(str(props.get(key) or '') for key in ('name', 'CAS', 'formula'))
            points = []
            for record in self._parse_pubchem_density_text(
                text,
                info,
                identity_text=identity_text,
            ):
                observation = self._density_observation_from_record(record, props, identity_text)
                if observation is None or observation.phase != 'liquid':
                    continue
                points.append(self._density_observation_to_liquid_point(observation))
            return points


        @staticmethod
        def _pubchem_density_at_reference(point: Dict[str, Any]) -> float:
            return float(point['rho_g_cm3']) / (1.0 - 9.0e-4 * (float(point['T_K']) - REFERENCE_TEMPERATURE_K))


        def _select_pubchem_density_cluster(self, points: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
            unique = []
            seen = set()
            for point in sorted(points, key=lambda item: (-float(item.get('quality', 0.0)), abs(float(item['T_K']) - REFERENCE_TEMPERATURE_K))):
                key = (round(float(point['rho_g_cm3']), 4), round(float(point['T_K']), 1), str(point.get('kind')))
                if key in seen:
                    continue
                seen.add(key)
                unique.append(point)
            if len(unique) < 2:
                return unique

            best: list[Dict[str, Any]] = []
            best_score = -1.0
            for center_point in unique:
                center = self._pubchem_density_at_reference(center_point)
                cluster = [
                    point
                    for point in unique
                    if abs(self._pubchem_density_at_reference(point) / center - 1.0) <= 0.08
                ]
                score = sum(float(point.get('quality', 0.0)) for point in cluster) + 0.03 * len(cluster)
                if score > best_score:
                    best = cluster
                    best_score = score
            return sorted(best, key=lambda item: (-float(item.get('quality', 0.0)), abs(float(item['T_K']) - REFERENCE_TEMPERATURE_K)))


        @staticmethod
        def _pubchem_density_cluster_quality(points: list[Dict[str, Any]]) -> float:
            if not points:
                return 0.0
            best_quality = max(float(point.get('quality', 0.80)) for point in points)
            if len(points) >= 4:
                best_quality += 0.04
            elif len(points) >= 2:
                best_quality += 0.02
            kinds = {str(point.get('kind')) for point in points}
            if kinds <= {'relative', 'range'}:
                best_quality -= 0.02
            return max(0.80, min(0.92, best_quality))


        def _pubchem_density_infos(self, node: Dict[str, Any]) -> list[tuple[str, Dict[str, Any]]]:
            infos = []

            def walk(current: Dict[str, Any]) -> None:
                if not isinstance(current, dict):
                    return
                if current.get('TOCHeading') == 'Density':
                    for info in current.get('Information', []) or []:
                        value = info.get('Value', {}) or {}
                        texts = []
                        for marked in value.get('StringWithMarkup', []) or []:
                            text = marked.get('String')
                            if text:
                                texts.append(text)
                        if 'Number' in value:
                            unit = value.get('Unit', '')
                            for number in value.get('Number') or []:
                                texts.append(f"{number} {unit}".strip())
                        if texts:
                            infos.append((' | '.join(texts), info))
                for child in current.get('Section', []) or []:
                    walk(child)
                record = current.get('Record')
                if isinstance(record, dict):
                    walk(record)

            walk(node)
            return infos


        @classmethod
        def _density_observation_to_liquid_point(
            cls,
            observation: DensityObservation,
        ) -> Dict[str, Any]:
            temperature_assumed = observation.temperature_K is None
            temperature = (
                cls.ASSUMED_BARE_DENSITY_TEMPERATURE_K
                if temperature_assumed
                else float(observation.temperature_K)
            )
            quality = float(observation.quality) * (0.96 if temperature_assumed else 1.0)
            return {
                'rho_g_cm3': float(observation.mass_density_kg_m3) / 1000.0,
                'T_K': temperature,
                'quality': quality,
                'kind': str(observation.metadata.get('parse_kind') or 'bare'),
                'text': observation.raw,
                'reference': observation.reference,
                'phase_basis': observation.phase_basis,
                'temperature_assumed': temperature_assumed,
            }


        def _fetch_density_pubchem(self, identifier: str) -> Optional[Dict[str, Any]]:
            cache_key = (
                f"density_pubchem_v{self.PUBCHEM_DENSITY_CACHE_VERSION}_{identifier}"
            )
            cached = self._get_cache(cache_key)
            if cached:
                if self._is_missing_cache(cached):
                    return None
                return cached

            try:
                cid = self._get_pubchem_cid(identifier)
                if not cid:
                    self._set_missing_cache(cache_key)
                    return None

                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Density"
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'PFD-Editor/1.0')
                with urllib.request.urlopen(req, timeout=15) as response:
                    data = json.loads(response.read().decode('utf-8'))

                raw_infos = self._pubchem_density_infos(data)
                records = []
                for text, info in raw_infos:
                    records.extend(self._parse_pubchem_density_text(text, info))
                if not records:
                    self._set_missing_cache(cache_key)
                    return None
                result = {
                    'cid': cid,
                    'raw_rows': len(raw_infos),
                    'parsed_records': len(records),
                    'records': records,
                }
                self._set_cache(cache_key, result)
                return result
            except Exception as e:
                if self._is_transient_lookup_error(e) or isinstance(e, LookupError):
                    raise LookupError(
                        f"Transient PubChem density lookup failure for '{identifier}'"
                    ) from e

            self._set_missing_cache(cache_key)
            return None


        @staticmethod
        def _density_identity_text(symbol: str, props: Mapping[str, Any]) -> str:
            return ' '.join(
                str(value).strip()
                for value in (
                    symbol,
                    props.get('name'),
                    props.get('CAS') or props.get('cas'),
                    props.get('formula') or props.get('Formula'),
                )
                if value
            )


        def _fetch_density_online(
            self,
            symbol: str,
            props: Mapping[str, Any],
        ) -> Optional[Dict[str, Any]]:
            transient_failure = False
            for candidate in self._identifier_candidates(symbol, dict(props)):
                try:
                    result = self._fetch_density_pubchem(candidate)
                except LookupError:
                    transient_failure = True
                    continue
                if result:
                    return result
            if transient_failure:
                raise LookupError(f"Transient online density lookup failure for '{symbol}'")
            return None


        def resolve_density_observations(
            self,
            symbol: str,
            props: Dict[str, Any] = None,
            *,
            allow_online: bool = True,
            phase: Optional[str] = None,
            material_form: Optional[str] = None,
        ) -> tuple[DensityObservation, ...]:
            """Return normalized intrinsic density observations for all phases."""
            props = self._coerce_props(symbol, props, allow_online=allow_online)
            if not self._props_allow_online(props, allow_online):
                return ()
            requested_phase = str(phase or '').strip().lower()
            if requested_phase and requested_phase not in {'solid', 'liquid', 'ambiguous'}:
                raise ValueError(f"Unsupported density phase {phase!r}")
            requested_form = str(material_form or '').strip().lower()
            if requested_form and requested_form not in {
                'anhydrous', 'hydrate', 'solvate', 'unspecified',
            }:
                raise ValueError(f"Unsupported density material form {material_form!r}")
            try:
                payload = self._fetch_density_online(symbol, props)
            except LookupError:
                return ()
            if not payload:
                return ()
            identity_text = self._density_identity_text(symbol, props)
            observations = []
            seen = set()
            for raw_record in payload.get('records') or []:
                record = dict(raw_record)
                metadata = dict(record.get('metadata') or {})
                metadata['pubchem_cid'] = payload.get('cid')
                record['metadata'] = metadata
                observation = self._density_observation_from_record(
                    record,
                    props,
                    identity_text,
                )
                if observation is None:
                    continue
                if requested_phase and observation.phase != requested_phase:
                    continue
                if requested_form and observation.material_form != requested_form:
                    continue
                key = (
                    round(observation.mass_density_kg_m3, 6),
                    round(observation.temperature_K, 6)
                    if observation.temperature_K is not None else None,
                    observation.phase,
                    observation.material_form,
                    observation.polymorph,
                    observation.stereochemistry,
                    observation.reference,
                    observation.raw,
                )
                if key in seen:
                    continue
                seen.add(key)
                observations.append(observation)
            return tuple(sorted(
                observations,
                key=lambda item: (
                    {'solid': 0, 'liquid': 1, 'ambiguous': 2}[item.phase],
                    item.temperature_K is None,
                    item.temperature_K or self.ASSUMED_BARE_DENSITY_TEMPERATURE_K,
                    -item.quality,
                    item.mass_density_kg_m3,
                ),
            ))


        def _liquid_density_payload(
            self,
            source_payload: Mapping[str, Any],
            observations: Iterable[DensityObservation],
        ) -> Optional[Dict[str, Any]]:
            points = [
                self._density_observation_to_liquid_point(observation)
                for observation in observations
                if observation.phase == 'liquid'
                and (
                    observation.material_form not in {'hydrate', 'solvate'}
                    or observation.is_identity_form
                )
            ]
            selected = self._select_pubchem_density_cluster(points)
            if not selected:
                return None
            return {
                'cid': source_payload.get('cid'),
                'raw_rows': source_payload.get('raw_rows'),
                'parsed_points': len(points),
                'points': selected,
                'quality': self._pubchem_density_cluster_quality(selected),
            }


        def _fetch_liquid_density_pubchem(
            self,
            identifier: str,
            props: Optional[Dict[str, Any]] = None,
        ) -> Optional[Dict[str, Any]]:
            """Compatibility adapter over the shared phase-neutral source cache."""
            props = self._coerce_props(identifier, props, allow_online=False)
            source_payload = self._fetch_density_pubchem(identifier)
            if not source_payload:
                return None
            identity_text = self._density_identity_text(identifier, props)
            observations = tuple(
                observation
                for observation in (
                    self._density_observation_from_record(record, props, identity_text)
                    for record in source_payload.get('records') or []
                )
                if observation is not None
            )
            return self._liquid_density_payload(source_payload, observations)


        def _fetch_liquid_density_online(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[Dict[str, Any]]:
            source_payload = self._fetch_density_online(symbol, props)
            if not source_payload:
                return None
            identity_text = self._density_identity_text(symbol, props)
            observations = tuple(
                observation
                for observation in (
                    self._density_observation_from_record(record, props, identity_text)
                    for record in source_payload.get('records') or []
                )
                if observation is not None
            )
            return self._liquid_density_payload(source_payload, observations)


        def _pubchem_density_source_result(self, online: Dict[str, Any]) -> PropertyResolutionResult:
            points = online.get('points') or []
            snippets = []
            references = []
            for point in points[:3]:
                snippets.append(str(point.get('text') or ''))
                reference = str(point.get('reference') or '')
                if reference:
                    references.append(reference)
            note = (
                f"PubChem CID {online.get('cid')}; selected {len(points)} density point(s) "
                f"from {online.get('parsed_points')} parsed point(s)"
            )
            if snippets:
                note += "; examples: " + " | ".join(snippets)
            if references:
                note += "; refs: " + " | ".join(references[:2])
            assumed = sum(bool(point.get('temperature_assumed')) for point in points)
            if assumed:
                note += (
                    f"; {assumed} temperatureless report(s) assigned "
                    f"{self.ASSUMED_BARE_DENSITY_TEMPERATURE_K:g} K with a quality penalty"
                )
            return PropertyResolutionResult(
                value=self._clamp_quality(online.get('quality'), 0.84),
                source='online',
                method='pubchem_liquid_density_cluster',
                quality=self._clamp_quality(online.get('quality'), 0.84),
                notes=note,
            )


        @staticmethod
        def _density_point_to_volume(point: Dict[str, Any], mw: float) -> Optional[float]:
            try:
                density_kg_m3 = float(point['rho_g_cm3']) * 1000.0
                volume = float(mw) / density_kg_m3
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            return LiquidVolumeMixin._positive_liquid_volume(volume)


        def _pubchem_density_rackett_volume(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any],
            critical: Dict[str, PropertyResolutionResult],
            online: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            mw = props.get('MW')
            Tc_result = self._critical_value_result(critical, 'Tc')
            Pc_result = self._critical_value_result(critical, 'Pc')
            if not (mw and Tc_result and Pc_result):
                return None
            if self._result_is_soft(Tc_result) or self._result_is_soft(Pc_result):
                return None

            samples = []
            for point in online.get('points') or []:
                volume = self._density_point_to_volume(point, float(mw))
                if volume is not None:
                    samples.append((float(point['T_K']), volume))
            zra = self._fit_zra_from_samples(samples, Tc_result.value, Pc_result.value)
            if zra is None:
                return None
            value = self._rackett_volume_m3_per_kmol(T, Tc_result.value, Pc_result.value, zra)
            if value is None:
                return None
            source_result = self._pubchem_density_source_result(online)
            zra_result = PropertyResolutionResult(
                value=zra,
                source='online',
                method='pubchem_density_zra_fit_source',
                quality=source_result.quality,
                notes=source_result.notes,
            )
            inputs = [source_result, Tc_result, Pc_result]
            return PropertyResolutionResult(
                value=value,
                source='online',
                method='pubchem_rackett_fitted_zra_liquid_volume',
                quality=self._combine_quality(inputs, method_factor=0.98),
                notes=(
                    f"Rackett liquid volume using Z_RA={zra:.5g} fitted from PubChem density; "
                    f"{source_result.notes}"
                ),
            )


        def _pubchem_density_thermal_volume(
            self,
            T: float,
            props: Dict[str, Any],
            online: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            mw = props.get('MW')
            if not mw:
                return None
            candidates = []
            for point in online.get('points') or []:
                volume = self._density_point_to_volume(point, float(mw))
                if volume is None:
                    continue
                candidates.append((abs(float(T) - float(point['T_K'])), point, volume))
            if not candidates:
                return None
            _, point, reference_volume = min(candidates, key=lambda item: (item[0], -float(item[1].get('quality', 0.0))))
            delta_T = float(T) - float(point['T_K'])
            try:
                expansion = math.exp(9.0e-4 * delta_T)
            except OverflowError:
                return None
            expansion = max(0.70, min(1.50, expansion))
            value = self._positive_liquid_volume(reference_volume * expansion)
            if value is None:
                return None
            distance = abs(delta_T)
            if distance <= 25.0:
                factor = 1.0
            elif distance <= 100.0:
                factor = 0.96
            elif distance <= 200.0:
                factor = 0.90
            else:
                factor = 0.84
            source_result = self._pubchem_density_source_result(online)
            point_note = (
                f"nearest PubChem point {float(point['rho_g_cm3']):g} g/cm^3 at "
                f"{float(point['T_K']):g} K; thermal expansion beta=9e-4 1/K"
            )
            if point.get('temperature_assumed'):
                point_note += (
                    f'; {self.ASSUMED_BARE_DENSITY_TEMPERATURE_K:g} K assumed '
                    'for a temperatureless source report'
                )
            return PropertyResolutionResult(
                value=value,
                source='online',
                method='pubchem_liquid_density_thermal_expansion',
                quality=self._combine_quality([source_result], method_factor=factor),
                notes=f"{point_note}; {source_result.notes}",
            )


        def _pubchem_liquid_volume(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any],
            critical: Dict[str, PropertyResolutionResult],
        ) -> Optional[PropertyResolutionResult]:
            if not self._props_allow_online(props):
                return None
            try:
                online = self._fetch_liquid_density_online(symbol, props)
            except LookupError:
                return None
            if not online:
                return None
            rackett = self._pubchem_density_rackett_volume(symbol, T, props, critical, online)
            if rackett:
                return rackett
            return self._pubchem_density_thermal_volume(T, props, online)


        def _rackett_yamada_gunn_volume(
            self,
            T: float,
            critical: Dict[str, PropertyResolutionResult],
        ) -> Optional[PropertyResolutionResult]:
            Tc_result = self._critical_value_result(critical, 'Tc')
            Pc_result = self._critical_value_result(critical, 'Pc')
            omega_result = self._critical_value_result(critical, 'omega')
            if not Tc_result or not Pc_result or not omega_result:
                return None
            zra = self._yamada_gunn_zra(omega_result.value)
            if zra is None:
                return None
            value = self._rackett_volume_m3_per_kmol(T, Tc_result.value, Pc_result.value, zra)
            if value is None:
                return None
            inputs = [Tc_result, Pc_result, omega_result]
            return PropertyResolutionResult(
                value=value,
                source='calculated',
                method='rackett_yamada_gunn',
                quality=self._combine_quality(inputs, method_factor=0.78),
                notes=f"Rackett liquid volume using Yamada-Gunn Z_RA={zra:.5g}",
            )


        @staticmethod
        def _solve_cubic_real_roots(a: float, b: float, c: float) -> list[float]:
            p = b - a * a / 3.0
            q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c
            shift = a / 3.0
            discriminant = (0.5 * q) ** 2 + (p / 3.0) ** 3
            if discriminant > 1.0e-14:
                sqrt_disc = math.sqrt(discriminant)
                u = math.copysign(abs(-0.5 * q + sqrt_disc) ** (1.0 / 3.0), -0.5 * q + sqrt_disc)
                v = math.copysign(abs(-0.5 * q - sqrt_disc) ** (1.0 / 3.0), -0.5 * q - sqrt_disc)
                return [u + v - shift]
            if abs(p) < 1.0e-14:
                return [-shift]
            argument = (3.0 * q / (2.0 * p)) * math.sqrt(-3.0 / p)
            argument = max(-1.0, min(1.0, argument))
            theta = math.acos(argument)
            radius = 2.0 * math.sqrt(-p / 3.0)
            return sorted(radius * math.cos((theta + 2.0 * math.pi * k) / 3.0) - shift for k in range(3))


        def _pure_pr_liquid_volume(
            self,
            T: float,
            P_bar: float,
            Tc: float,
            Pc_bar: float,
            omega: float,
        ) -> Optional[float]:
            try:
                T = float(T)
                P_bar = max(float(P_bar), MIN_EOS_PRESSURE_BAR)
                Tc = float(Tc)
                Pc_bar = float(Pc_bar)
                omega = float(omega)
                Tr = max(T / Tc, 1.0e-12)
                m = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
                alpha = (1.0 + m * (1.0 - math.sqrt(Tr))) ** 2
                a = 0.45724 * R_BAR_CM3_MOL_K**2 * Tc**2 / Pc_bar * alpha
                b = 0.07780 * R_BAR_CM3_MOL_K * Tc / Pc_bar
                A = a * P_bar / (R_BAR_CM3_MOL_K**2 * T**2)
                B = b * P_bar / (R_BAR_CM3_MOL_K * T)
                roots = self._solve_cubic_real_roots(
                    -(1.0 - B),
                    A - 3.0 * B * B - 2.0 * B,
                    -(A * B - B * B - B**3),
                )
                roots = [root for root in roots if root > B + 1.0e-12]
                if not roots:
                    return None
                volume_cm3_mol = min(roots) * R_BAR_CM3_MOL_K * T / P_bar
                return self._positive_liquid_volume(volume_cm3_mol / 1000.0)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None


        def _pure_ptv_liquid_volume(
            self,
            T: float,
            P_bar: float,
            Tc: float,
            Pc_bar: float,
            omega: float,
            Zc: float,
        ) -> Optional[float]:
            try:
                T = float(T)
                P_bar = max(float(P_bar), MIN_EOS_PRESSURE_BAR)
                Tc = float(Tc)
                Pc_bar = float(Pc_bar)
                omega = float(omega)
                Zc = float(Zc)
                if not (0.15 <= Zc <= 0.35):
                    return None
                omega_a = 0.66121 - 0.76105 * Zc
                omega_b = 0.02207 + 0.20868 * Zc
                omega_c = 0.57765 - 1.87080 * Zc
                if omega_a <= 0.0 or omega_b <= 0.0:
                    return None
                m = 0.452413 + 1.30982 * omega - 0.295937 * omega**2
                alpha = (1.0 + m * (1.0 - math.sqrt(max(T / Tc, 1.0e-12)))) ** 2
                a = omega_a * R_BAR_CM3_MOL_K**2 * Tc**2 / Pc_bar * alpha
                b = omega_b * R_BAR_CM3_MOL_K * Tc / Pc_bar
                c = omega_c * R_BAR_CM3_MOL_K * Tc / Pc_bar
                A = a * P_bar / (R_BAR_CM3_MOL_K**2 * T**2)
                B = b * P_bar / (R_BAR_CM3_MOL_K * T)
                C = c * P_bar / (R_BAR_CM3_MOL_K * T)
                roots = self._solve_cubic_real_roots(
                    C - 1.0,
                    A - 2.0 * B * C - B * B - B - C,
                    B * B * C + B * C - A * B,
                )
                roots = [root for root in roots if root > B + 1.0e-12]
                if not roots:
                    return None
                volume_cm3_mol = min(roots) * R_BAR_CM3_MOL_K * T / P_bar
                return self._positive_liquid_volume(volume_cm3_mol / 1000.0)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None


        def _eos_pressure_for_liquid_volume(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any],
        ) -> tuple[float, str]:
            try:
                psat = self.resolve_vapor_pressure(symbol, T, props)
                if psat and psat.value is not None and psat.value > 0.0:
                    return max(float(psat.value), MIN_EOS_PRESSURE_BAR), f"Psat from {psat.source}/{psat.method}"
            except Exception:
                pass
            return NORMAL_BOILING_PRESSURE_BAR, '1 atm normal-boiling fallback pressure'


        def _ptv_liquid_volume(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any],
            critical: Dict[str, PropertyResolutionResult],
        ) -> Optional[PropertyResolutionResult]:
            Tc_result = self._critical_value_result(critical, 'Tc')
            Pc_result = self._critical_value_result(critical, 'Pc')
            omega_result = self._critical_value_result(critical, 'omega')
            zc_result = self._critical_value_result(critical, 'Zc')
            if not (Tc_result and Pc_result and omega_result and zc_result):
                return None
            if self._result_quality(zc_result, 0.0) < SOFT_PROPERTY_QUALITY_THRESHOLD:
                return None
            if self._result_is_soft(zc_result):
                return None
            P_bar, pressure_note = self._eos_pressure_for_liquid_volume(symbol, T, props)
            value = self._pure_ptv_liquid_volume(
                T,
                P_bar,
                Tc_result.value,
                Pc_result.value,
                omega_result.value,
                zc_result.value,
            )
            if value is None:
                return None
            inputs = [Tc_result, Pc_result, omega_result, zc_result]
            return PropertyResolutionResult(
                value=value,
                source='calculated',
                method='ptv_eos_liquid_volume',
                quality=self._combine_quality(inputs, method_factor=0.84),
                notes=f"Pure-fluid PTV EOS liquid root at P={P_bar:g} bar ({pressure_note}; minimum 1 Pa)",
            )


        def _pr_liquid_volume(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any],
            critical: Dict[str, PropertyResolutionResult],
        ) -> Optional[PropertyResolutionResult]:
            Tc_result = self._critical_value_result(critical, 'Tc')
            Pc_result = self._critical_value_result(critical, 'Pc')
            omega_result = self._critical_value_result(critical, 'omega')
            if not (Tc_result and Pc_result and omega_result):
                return None
            P_bar, pressure_note = self._eos_pressure_for_liquid_volume(symbol, T, props)
            value = self._pure_pr_liquid_volume(T, P_bar, Tc_result.value, Pc_result.value, omega_result.value)
            if value is None:
                return None
            inputs = [Tc_result, Pc_result, omega_result]
            return PropertyResolutionResult(
                value=value,
                source='calculated',
                method='pr_eos_liquid_volume',
                quality=self._combine_quality(inputs, method_factor=0.75),
                notes=f"Pure-fluid Peng-Robinson liquid root at P={P_bar:g} bar ({pressure_note}; minimum 1 Pa)",
            )


        def _formula_liquid_volume_estimate(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import parse_formula_counts
                else:
                    from compound_identity import parse_formula_counts
            except Exception:
                parse_formula_counts = None
            formula = props.get('formula') or props.get('Formula') or symbol
            counts = parse_formula_counts(str(formula)) if parse_formula_counts else None
            increments = {
                'C': 16.5,
                'H': 3.7,
                'O': 6.0,
                'N': 5.5,
                'S': 17.0,
                'F': 8.0,
                'Cl': 22.0,
                'Br': 27.0,
                'I': 37.0,
                'Si': 20.0,
                'P': 24.0,
            }
            if counts:
                volume_cm3_mol = 0.0
                unsupported_elements = []
                for element, count in counts.items():
                    increment = increments.get(element)
                    if increment is None:
                        unsupported_elements.append(element)
                        continue
                    volume_cm3_mol += increment * count
                if not unsupported_elements and volume_cm3_mol > 0.0:
                    compactness_factor, compactness_note = self._liquid_volume_atom_density_multiplier(
                        symbol,
                        props,
                        counts,
                    )
                    expansion = max(0.75, min(1.35, 1.0 + 9.0e-4 * (float(T) - REFERENCE_TEMPERATURE_K)))
                    value = self._positive_liquid_volume(volume_cm3_mol * expansion / compactness_factor / 1000.0)
                    if value is not None:
                        correction_note = ''
                        if compactness_factor > 1.0:
                            correction_note = f"; {compactness_note} density multiplier {compactness_factor:g}"
                        return PropertyResolutionResult(
                            value=value,
                            source='estimated',
                            method='atomic_increment_liquid_volume',
                            quality=0.60,
                            notes=f"Rough liquid-volume estimate from formula {formula} atom increments{correction_note}",
                        )

            mw = props.get('MW')
            try:
                mw_f = float(mw)
            except (TypeError, ValueError):
                return None
            if mw_f <= 0.0:
                return None
            density = 650.0 + 250.0 * math.exp(-mw_f / 100.0)
            value = self._positive_liquid_volume(mw_f / density)
            if value is None:
                return None
            return PropertyResolutionResult(
                value=value,
                source='estimated',
                method='mw_liquid_volume_heuristic',
                quality=0.40,
                notes='Very rough MW-based liquid-volume estimate with size-dependent density',
            )


        @classmethod
        def _liquid_volume_atom_density_multiplier(
            cls,
            symbol: str,
            props: Dict[str, Any],
            counts: Optional[Dict[str, int]],
        ) -> tuple[float, str]:
            ring_factor = cls._liquid_volume_ring_compactness_factor(symbol, props)
            if ring_factor > 1.0:
                return ring_factor, 'ring compactness'

            carbon_count = (counts or {}).get('C', 0)
            chain_factor = cls._liquid_volume_chain_density_multiplier(carbon_count)
            acid_factor = 1.10 if cls._looks_like_carboxylic_acid(symbol, props, counts) else 1.0
            factor = chain_factor * acid_factor
            if acid_factor > 1.0:
                factor = min(factor, 1.30)

            if factor <= 1.0:
                return 1.0, ''
            notes = []
            if chain_factor > 1.0:
                notes.append('carbon-count chain correction')
            if acid_factor > 1.0:
                notes.append('carboxylic-acid correction')
            return factor, ' + '.join(notes)


        @staticmethod
        def _liquid_volume_chain_density_multiplier(carbon_count: int) -> float:
            try:
                carbon_count = int(carbon_count)
            except (TypeError, ValueError):
                return 1.0
            if 4 <= carbon_count <= 5:
                return 1.05
            if 6 <= carbon_count <= 8:
                return 1.15
            if 9 <= carbon_count <= 12:
                return 1.25
            if carbon_count >= 13:
                return 1.30
            return 1.0


        @staticmethod
        def _looks_like_carboxylic_acid(
            symbol: str,
            props: Dict[str, Any],
            counts: Optional[Dict[str, int]],
        ) -> bool:
            if (counts or {}).get('O', 0) < 2:
                return False
            names = [
                symbol,
                props.get('name'),
                props.get('Name'),
                *(props.get('identifiers') or []),
            ]
            compact = ' '.join(str(name).lower() for name in names if name)
            compact = re.sub(r'[^a-z0-9]+', '', compact)
            return any(
                hint in compact
                for hint in (
                    'acid',
                    'formic',
                    'acetic',
                    'propionic',
                    'butyric',
                    'pentanoic',
                    'hexanoic',
                    'heptanoic',
                    'octanoic',
                    'nonanoic',
                    'decanoic',
                )
            )


        @classmethod
        def _liquid_volume_ring_compactness_factor(
            cls,
            symbol: str,
            props: Dict[str, Any],
        ) -> float:
            smiles_values = [
                props.get('smiles'),
                props.get('SMILES'),
                props.get('canonical_smiles'),
                props.get('CanonicalSMILES'),
            ]
            saw_smiles = False
            for smiles in smiles_values:
                if smiles:
                    saw_smiles = True
                ring_sizes = cls._smiles_ring_sizes(smiles)
                if ring_sizes:
                    if any(size == 6 for size in ring_sizes):
                        return 1.35
                    if any(size == 5 for size in ring_sizes):
                        return 1.25
                    return 1.0
            if saw_smiles:
                return 1.0

            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..chemical_properties import ChemicalDatabase
                else:
                    from chemical_properties import ChemicalDatabase
                resolved = ChemicalDatabase(enable_online=False).resolve_smiles_info(
                    symbol,
                    fetch_online=False,
                    props=props,
                )
            except Exception:
                resolved = None
            if resolved and resolved.smiles:
                ring_sizes = cls._smiles_ring_sizes(resolved.smiles)
                if ring_sizes:
                    if any(size == 6 for size in ring_sizes):
                        return 1.35
                    if any(size == 5 for size in ring_sizes):
                        return 1.25
                    return 1.0

            names = [
                symbol,
                props.get('name'),
                props.get('Name'),
                *(props.get('identifiers') or []),
            ]
            compact = ' '.join(str(name).lower() for name in names if name)
            compact = re.sub(r'[^a-z0-9]+', '', compact)
            six_member_hints = (
                'benz',
                'phenyl',
                'phenol',
                'anilin',
                'anisole',
                'cresol',
                'styrene',
                'methylstyrene',
                'cumene',
                'naphthalene',
                'tetralin',
                'tetrahydronaphthalene',
                'toluene',
                'xylene',
                'xylenol',
                'pyridin',
                'picoline',
                'cyclohex',
            )
            five_member_hints = (
                'cyclopent',
                'furan',
                'tetrahydrofuran',
                'thf',
                'oxolane',
                'thiophene',
                'pyrrole',
            )
            if any(hint in compact for hint in six_member_hints):
                return 1.35
            if any(hint in compact for hint in five_member_hints):
                return 1.25
            return 1.0


        @classmethod
        def _smiles_ring_sizes(cls, smiles: Any) -> list[int]:
            if not smiles:
                return []
            text = str(smiles).strip()
            atoms: list[str] = []
            adjacency: dict[int, set[int]] = {}
            branch_stack: list[Optional[int]] = []
            ring_open: dict[str, int] = {}
            current: Optional[int] = None
            ring_sizes: list[int] = []
            index = 0

            while index < len(text):
                char = text[index]
                if char == '[':
                    end = text.find(']', index + 1)
                    if end < 0:
                        break
                    token = text[index + 1:end]
                    match = re.search(r'[A-Z][a-z]?|[bcnops]', token)
                    if match:
                        current = cls._smiles_add_atom(atoms, adjacency, current, match.group(0))
                    index = end + 1
                    continue
                if char.isalpha():
                    token = char
                    if char.isupper() and index + 1 < len(text) and text[index + 1].islower():
                        candidate = text[index:index + 2]
                        if candidate in {'Cl', 'Br', 'Si'}:
                            token = candidate
                            index += 1
                    current = cls._smiles_add_atom(atoms, adjacency, current, token)
                elif char == '(':
                    branch_stack.append(current)
                elif char == ')':
                    current = branch_stack.pop() if branch_stack else current
                elif char.isdigit() or char == '%':
                    if char == '%' and index + 2 < len(text) and text[index + 1:index + 3].isdigit():
                        ring_id = text[index + 1:index + 3]
                        index += 2
                    else:
                        ring_id = char
                    if current is not None:
                        other = ring_open.pop(ring_id, None)
                        if other is None:
                            ring_open[ring_id] = current
                        else:
                            size = cls._smiles_shortest_path_length(adjacency, other, current)
                            if size is not None:
                                ring_sizes.append(size + 1)
                            adjacency.setdefault(other, set()).add(current)
                            adjacency.setdefault(current, set()).add(other)
                index += 1
            return ring_sizes


        @staticmethod
        def _smiles_add_atom(
            atoms: list[str],
            adjacency: dict[int, set[int]],
            current: Optional[int],
            token: str,
        ) -> int:
            atom_index = len(atoms)
            atoms.append(token)
            adjacency.setdefault(atom_index, set())
            if current is not None:
                adjacency.setdefault(current, set()).add(atom_index)
                adjacency[atom_index].add(current)
            return atom_index


        @staticmethod
        def _smiles_shortest_path_length(
            adjacency: dict[int, set[int]],
            start: int,
            target: int,
        ) -> Optional[int]:
            queue = [(start, 0)]
            seen = {start}
            for node, distance in queue:
                if node == target:
                    return distance
                for neighbor in adjacency.get(node, ()):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))
            return None


        @staticmethod
        def _bare_solid_density_observations(
            observations: Iterable[DensityObservation],
            identity_material_form: str = 'unspecified',
        ) -> tuple[DensityObservation, ...]:
            solid = [item for item in observations if item.phase == 'solid']
            if identity_material_form in {'hydrate', 'solvate'}:
                solid = [
                    item
                    for item in solid
                    if item.is_identity_form
                    and item.material_form == identity_material_form
                ]
            else:
                solid = [
                    item
                    for item in solid
                    if item.material_form not in {'hydrate', 'solvate'}
                ]
            if not solid:
                return ()
            unlabeled = [item for item in solid if not item.polymorph]
            if not unlabeled:
                # PubChem does not provide a reliable "common polymorph" flag.
                # Preserve labeled forms through resolve_density_observations,
                # but do not guess which one a bare solid-density query means.
                return ()
            return tuple(unlabeled)


        @staticmethod
        def _select_solid_density_cluster(
            observations: Iterable[DensityObservation],
            T: float,
        ) -> tuple[DensityObservation, ...]:
            unique = []
            seen = set()
            for observation in observations:
                key = (
                    round(observation.mass_density_kg_m3, 3),
                    round(observation.temperature_K, 2)
                    if observation.temperature_K is not None else None,
                    observation.material_form,
                    observation.polymorph,
                    observation.reference,
                )
                if key in seen:
                    continue
                seen.add(key)
                unique.append(observation)
            if not unique:
                return ()
            centers = [item for item in unique if not item.polymorph] or unique
            best = []
            best_score = -math.inf
            for center in centers:
                cluster = [
                    item
                    for item in unique
                    if abs(item.mass_density_kg_m3 / center.mass_density_kg_m3 - 1.0)
                    <= 0.08
                ]
                distance = min(
                    abs(
                        float(T)
                        - (
                            item.temperature_K
                            if item.temperature_K is not None
                            else LiquidVolumeMixin.ASSUMED_BARE_DENSITY_TEMPERATURE_K
                        )
                    )
                    for item in cluster
                )
                score = (
                    sum(item.quality for item in cluster)
                    + 0.03 * len(cluster)
                    - 1.0e-4 * distance
                )
                if score > best_score:
                    best = cluster
                    best_score = score
            return tuple(sorted(
                best,
                key=lambda item: (
                    abs(
                        float(T)
                        - (
                            item.temperature_K
                            if item.temperature_K is not None
                            else LiquidVolumeMixin.ASSUMED_BARE_DENSITY_TEMPERATURE_K
                        )
                    ),
                    -item.quality,
                    item.mass_density_kg_m3,
                ),
            ))


        @classmethod
        def _solid_density_cluster_quality(
            cls,
            observations: tuple[DensityObservation, ...],
        ) -> float:
            quality = max(item.quality for item in observations)
            if len(observations) >= 4:
                quality += 0.04
            elif len(observations) >= 2:
                quality += 0.02
            return max(0.0, min(0.94, quality))


        def _solid_density_expansion_policy(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> tuple[float, float, str]:
            identity_text = self._density_identity_text(symbol, props)
            identity_form = classify_fusion_material_form(
                identity_text,
                identity_text,
            )[0]
            if identity_form in {'hydrate', 'solvate'}:
                return (
                    0.0,
                    self.SOLID_OTHER_QUALITY_PENALTY_PER_5K,
                    f'{identity_form} identity: constant-density policy',
                )

            smiles_result = self._resolve_smiles_result(
                symbol,
                props,
                allow_online=allow_online,
            )
            if smiles_result and smiles_result.value:
                try:
                    from rdkit import Chem
                    molecule = Chem.MolFromSmiles(str(smiles_result.value))
                except Exception:
                    molecule = None
                if molecule is not None:
                    fragments = Chem.GetMolFrags(molecule)
                    neutral = (
                        len(fragments) == 1
                        and all(
                            atom.GetFormalCharge() == 0
                            for atom in molecule.GetAtoms()
                        )
                    )
                    organic_props = dict(props)
                    organic_props['smiles'] = str(smiles_result.value)
                    molecular_organic = self._critical_formula_is_organic(organic_props)
                    allowed_atomic_numbers = {
                        5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53,
                    }
                    nonmetal = all(
                        atom.GetAtomicNum() == 1
                        or atom.GetAtomicNum() in allowed_atomic_numbers
                        for atom in molecule.GetAtoms()
                    )
                    if neutral and molecular_organic and nonmetal:
                        return (
                            self.SOLID_ORGANIC_VOLUMETRIC_EXPANSION_K_INV,
                            self.SOLID_ORGANIC_EXPANSION_QUALITY_PENALTY_PER_5K,
                            'neutral molecular organic structure',
                        )
                    return (
                        0.0,
                        self.SOLID_OTHER_QUALITY_PENALTY_PER_5K,
                        'structure is not a neutral nonmetal molecular organic: '
                        'constant-density policy',
                    )

            organic = self._critical_formula_is_organic(props)
            metal_or_salt = self._critical_formula_is_metal_or_salt(props)
            formula = self._critical_identity_formula(props)
            charged = self._formula_reports_net_charge(formula)
            if organic and not metal_or_salt and not charged:
                return (
                    self.SOLID_ORGANIC_VOLUMETRIC_EXPANSION_K_INV,
                    self.SOLID_ORGANIC_EXPANSION_QUALITY_PENALTY_PER_5K,
                    'neutral molecular organic formula fallback',
                )
            return (
                0.0,
                self.SOLID_OTHER_QUALITY_PENALTY_PER_5K,
                'inorganic, ionic, metallic, or uncertain identity: '
                'constant-density policy',
            )


        @staticmethod
        def _solid_density_temperature_steps(T: float, T_ref: float) -> int:
            return int(math.floor((abs(float(T) - float(T_ref)) + 1.0e-9) / 5.0))


        @classmethod
        def _solid_density_at_temperature(
            cls,
            density_kg_m3: float,
            T: float,
            T_ref: float,
            alpha_v_K_inv: float,
        ) -> Optional[float]:
            denominator = 1.0 + float(alpha_v_K_inv) * (float(T) - float(T_ref))
            if denominator <= 0.0 or not math.isfinite(denominator):
                return None
            density = float(density_kg_m3) / denominator
            return density if density > 0.0 and math.isfinite(density) else None


        @classmethod
        def _solid_density_transition_is_usable(
            cls,
            result: Optional[PropertyResolutionResult],
        ) -> bool:
            if (
                result is None
                or result.value is None
                or cls._result_quality(result, 0.0) < SOFT_PROPERTY_QUALITY_THRESHOLD
                or cls._result_is_soft(result)
            ):
                return False
            try:
                value = float(result.value)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(value) or value <= 0.0:
                return False
            text = ' '.join((result.source, result.method, result.notes)).lower()
            return not any(token in text for token in (
                'provisional', 'decompos', 'dehydrat', 'desolvat',
                'sublim', 'glass transition', 'softening', 'solid-solid',
            ))


        def _solid_density_transition_temperature(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
            requested_temperature: Optional[float] = None,
        ) -> tuple[Optional[PropertyResolutionResult], str]:
            def applicable(result: Optional[PropertyResolutionResult]) -> bool:
                if not self._solid_density_transition_is_usable(result):
                    return False
                return (
                    requested_temperature is None
                    or float(result.value) + 1.0e-9 >= float(requested_temperature)
                )

            direct_tt = self._source_result_for_value(props, 'Tt', units='K')
            if applicable(direct_tt):
                return direct_tt, 'Tt'
            try:
                triple = self.resolve_triple_point(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
            except Exception:
                triple = {}
            resolved_tt = triple.get('Tt') if isinstance(triple, Mapping) else None
            if applicable(resolved_tt):
                return resolved_tt, 'Tt'

            direct_tm = self._source_result_for_value(props, 'Tm', units='K')
            if applicable(direct_tm):
                return direct_tm, 'Tm'
            try:
                resolved_tm = self.resolve_melting_point(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
            except Exception:
                resolved_tm = None
            if applicable(resolved_tm):
                return resolved_tm, 'Tm'
            return None, ''


        def _solid_density_heuristic_structure(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> tuple[Optional[PropertyResolutionResult], str]:
            identity_text = self._density_identity_text(symbol, props)
            identity_form = classify_fusion_material_form(
                identity_text,
                identity_text,
            )[0]
            if identity_form in {'hydrate', 'solvate'}:
                return None, f'{identity_form} identities are outside the heuristic domain'
            smiles_result = self._resolve_smiles_result(
                symbol,
                props,
                allow_online=allow_online,
            )
            if not smiles_result or not smiles_result.value:
                return None, 'a molecular structure is required to screen packing eligibility'
            try:
                from rdkit import Chem
            except Exception:
                return None, 'RDKit is required to screen packing eligibility'
            molecule = Chem.MolFromSmiles(str(smiles_result.value))
            if molecule is None:
                return None, 'the resolved molecular structure is invalid'
            molecule = Chem.RemoveHs(molecule)
            if len(Chem.GetMolFrags(molecule)) != 1:
                return None, 'multi-fragment structures are outside the neutral molecular domain'
            if any(atom.GetFormalCharge() != 0 for atom in molecule.GetAtoms()):
                return None, 'formally charged structures are outside the heuristic domain'
            organic_props = dict(props)
            organic_props['smiles'] = str(smiles_result.value)
            if not self._critical_formula_is_organic(organic_props):
                return None, 'the structure is not a molecular organic compound'
            allowed_atomic_numbers = {5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
            heavy_atoms = [
                atom
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() > 1
            ]
            if any(atom.GetAtomicNum() not in allowed_atomic_numbers for atom in heavy_atoms):
                return None, 'metal-containing and nonstandard-element structures are excluded'
            heavy_count = len(heavy_atoms)
            automorphism_count = len(molecule.GetSubstructMatches(
                molecule,
                uniquify=False,
                useChirality=True,
                maxMatches=self.SOLID_DENSITY_HEURISTIC_AUTOMORPHISM_LIMIT,
            ))
            if (
                automorphism_count
                >= self.SOLID_DENSITY_HEURISTIC_AUTOMORPHISM_LIMIT
            ):
                return None, (
                    'highly symmetric molecular graph: at least '
                    f'{self.SOLID_DENSITY_HEURISTIC_AUTOMORPHISM_LIMIT} '
                    'heavy-atom graph automorphisms'
                )
            return smiles_result, (
                f'neutral single-fragment organic; {heavy_count} heavy atom(s); '
                f'{automorphism_count} heavy-atom graph automorphism(s)'
            )


        def _solid_density_intervening_transition(
            self,
            symbol: str,
            T: float,
            transition_temperature: float,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> Optional[float]:
            """Return the first known solid transition crossed by the estimate."""
            try:
                kernel = self.resolve_solid_cp_kernel(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
            except Exception:
                return None
            if kernel is None:
                return None
            low, high = sorted((float(T), float(transition_temperature)))
            crossed = sorted(
                float(value)
                for value in kernel.transition_temperatures
                if low + 1.0e-9 < float(value) < high - 1.0e-9
            )
            return crossed[0] if crossed else None


        def _solid_density_organic_fallback(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> tuple[Optional[PropertyResolutionResult], str]:
            structure_result, eligibility_note = self._solid_density_heuristic_structure(
                symbol,
                props,
                allow_online=allow_online,
            )
            if structure_result is None:
                return None, eligibility_note
            transition, transition_kind = self._solid_density_transition_temperature(
                symbol, props,
                allow_online=allow_online,
                requested_temperature=T,
            )
            if transition is None:
                return None, 'no trustworthy Tt or Tm is available'
            transition_temperature = float(transition.value)
            minimum_temperature = (
                self.SOLID_DENSITY_HEURISTIC_MIN_REDUCED_TEMPERATURE
                * transition_temperature
            )
            if T < minimum_temperature:
                return None, (
                    f'T={T:g} K is below 0.3*{transition_kind}='
                    f'{minimum_temperature:g} K'
                )
            if T > transition_temperature:
                return None, (
                    f'T={T:g} K is above {transition_kind}='
                    f'{transition_temperature:g} K'
                )
            intervening = self._solid_density_intervening_transition(
                symbol,
                T,
                transition_temperature,
                props,
                allow_online=allow_online,
            )
            if intervening is not None:
                return None, (
                    f'a known solid transition at {intervening:g} K lies between '
                    f'T={T:g} K and {transition_kind}={transition_temperature:g} K'
                )
            liquid_props = dict(props)
            if not allow_online:
                liquid_props['_allow_online_lookup'] = False
            try:
                liquid_molar_density = self.resolve_liquid_molar_density(
                    symbol,
                    transition_temperature,
                    liquid_props,
                )
            except Exception:
                liquid_molar_density = None
            if (
                liquid_molar_density is None
                or liquid_molar_density.value is None
                or self._result_quality(liquid_molar_density, 0.0)
                < self.SOLID_DENSITY_HEURISTIC_LIQUID_MIN_QUALITY
            ):
                return None, (
                    'liquid molar volume at the transition is unavailable or '
                    f'below quality {self.SOLID_DENSITY_HEURISTIC_LIQUID_MIN_QUALITY:g}'
                )
            mw_result = self._source_result_for_value(props, 'MW', units='g/mol')
            if mw_result is None or mw_result.value is None:
                return None, 'molecular weight is unavailable'
            try:
                liquid_density = (
                    float(mw_result.value) * float(liquid_molar_density.value)
                )
                # Temperature-general organic volume-of-fusion correlation:
                # J. Chem. Eng. Data (2004) 49 (6): 1512–1514.
                ratio = 1.28 - 0.16 * float(T) / transition_temperature
                density = ratio * liquid_density
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None, 'the heuristic density calculation was not finite'
            if (
                not math.isfinite(liquid_density)
                or liquid_density <= 0.0
                or not math.isfinite(ratio)
                or ratio <= 0.0
                or not math.isfinite(density)
                or density <= 0.0
            ):
                return None, 'the heuristic density calculation was not positive and finite'
            base_quality = 0.70 * min(
                self._result_quality(transition, 0.0),
                self._result_quality(liquid_molar_density, 0.0),
            )
            notes = (
                f'rho_s(T)=(1.28-0.16*T/{transition_kind})*rho_l({transition_kind}); '
                f'T={T:g} K; '
                f'{transition_kind}={transition_temperature:g} K from '
                f'{transition.source}/{transition.method}; rho_l={liquid_density:g} kg/m^3 '
                f'from {liquid_molar_density.source}/{liquid_molar_density.method}; '
                f'multiplier={ratio:g}; reported MAPE at multiple temperatures is '
                '5.6% (not an uncertainty interval); source: J. Chem. Eng. Data '
                '(2004) 49 (6): 1512–1514; no known solid transition lies between '
                f'T and {transition_kind}; '
                f'eligibility: {eligibility_note}; units kg/m^3'
            )
            return PropertyResolutionResult(
                value=density,
                source='estimated',
                method='organic_volume_of_fusion_solid_density',
                quality=base_quality,
                notes=notes,
            ), ''


        def _resolve_observed_or_estimated_solid_mass_density(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any] = None,
            *,
            allow_online: bool = True,
        ) -> PropertyResolutionResult:
            """Resolve intrinsic solid mass density in kg/m^3."""
            try:
                T = float(T)
            except (TypeError, ValueError) as exc:
                raise ValueError('Solid-density temperature must be numeric') from exc
            if not math.isfinite(T) or T <= 0.0:
                raise ValueError('Solid-density temperature must be positive and finite')
            props = self._coerce_props(symbol, props, allow_online=allow_online)
            observations = self.resolve_density_observations(
                symbol,
                props,
                allow_online=allow_online,
                phase='solid',
            )
            identity_text = self._density_identity_text(symbol, props)
            identity_material_form = classify_fusion_material_form(
                identity_text,
                identity_text,
            )[0]
            bare = self._bare_solid_density_observations(
                observations,
                identity_material_form,
            )
            selected = self._select_solid_density_cluster(bare, T)
            if not selected:
                if observations:
                    if any(item.polymorph for item in observations):
                        reason = 'only form-labeled polymorph densities are available'
                    else:
                        reason = 'only incompatible hydrate/solvate solid densities are available'
                    heuristic = None
                else:
                    heuristic, reason = self._solid_density_organic_fallback(
                        symbol,
                        T,
                        props,
                        allow_online=allow_online,
                    )
                if heuristic is not None:
                    return heuristic
                raise PropertyResolutionError(
                    f"Cannot determine solid mass density for '{symbol}': {reason}."
                )
            alpha_v, penalty_per_step, expansion_note = (
                self._solid_density_expansion_policy(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
            )
            adjusted = []
            for item in selected:
                reference_T = (
                    item.temperature_K
                    if item.temperature_K is not None
                    else self.ASSUMED_BARE_DENSITY_TEMPERATURE_K
                )
                value = self._solid_density_at_temperature(
                    item.mass_density_kg_m3,
                    T,
                    reference_T,
                    alpha_v,
                )
                if value is not None:
                    adjusted.append((value, reference_T, item))
            if not adjusted:
                raise PropertyResolutionError(
                    f"Cannot determine solid mass density for '{symbol}': "
                    'thermal-expansion evaluation failed.'
                )
            density = median(item[0] for item in adjusted)
            nearest_steps = min(
                self._solid_density_temperature_steps(T, item[1])
                for item in adjusted
            )
            quality_penalty = penalty_per_step * nearest_steps
            quality = max(
                0.0,
                self._solid_density_cluster_quality(selected) - quality_penalty,
            )
            temperature_notes = [
                (
                    f'{value:g} K'
                    if value is not None
                    else f'assumed {self.ASSUMED_BARE_DENSITY_TEMPERATURE_K:g} K '
                         f'for a temperatureless report'
                )
                for value in (item.temperature_K for item in selected)
            ]
            references = []
            for item in selected:
                if item.reference and item.reference not in references:
                    references.append(item.reference)
            note = (
                f'PubChem intrinsic solid-density cluster from {len(selected)} of '
                f'{len(observations)} phase-qualified solid observation(s); '
                f"observation temperatures: {', '.join(temperature_notes)}; "
                f"phase bases: {' | '.join(dict.fromkeys(item.phase_basis for item in selected))}; "
                f'expansion policy: {expansion_note}; alpha_v={alpha_v:g} 1/K; '
                f'requested T={T:g} K; quality penalty={quality_penalty:g} from '
                f'{nearest_steps} complete 5 K interval(s) relative to the nearest '
                f'observation; units kg/m^3'
            )
            if references:
                note += '; refs: ' + ' | '.join(references[:3])
            return PropertyResolutionResult(
                value=density,
                source='online',
                method='pubchem_solid_density_cluster',
                quality=quality,
                notes=note,
            )


        def _resolve_observed_or_estimated_solid_molar_volume(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any] = None,
            *,
            allow_online: bool = True,
        ) -> PropertyResolutionResult:
            """Resolve solid molar volume in m^3/kmol."""
            props = self._coerce_props(symbol, props, allow_online=allow_online)
            density = self._resolve_observed_or_estimated_solid_mass_density(
                symbol,
                T,
                props,
                allow_online=allow_online,
            )
            try:
                mw = float(props.get('MW'))
                value = mw / float(density.value)
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                raise PropertyResolutionError(
                    f"Cannot convert solid density for '{symbol}' without molecular weight."
                ) from exc
            if not math.isfinite(value) or value <= 0.0:
                raise PropertyResolutionError(
                    f"Cannot determine solid molar volume for '{symbol}'."
                )
            mw_result = self._source_result_for_value(props, 'MW', units='g/mol')
            quality = self._combine_quality([density, mw_result], exact_formula=True)
            return PropertyResolutionResult(
                value=value,
                source=self._derived_source([density, mw_result], exact_formula=True),
                method='solid_molar_volume_from_mass_density',
                quality=quality,
                notes=f'MW/rho_s; units m^3/kmol; {density.notes}',
            )


        def _resolve_observed_or_estimated_solid_molar_density(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any] = None,
            *,
            allow_online: bool = True,
        ) -> PropertyResolutionResult:
            """Resolve solid molar density in mol/dm^3."""
            volume = self._resolve_observed_or_estimated_solid_molar_volume(
                symbol,
                T,
                props,
                allow_online=allow_online,
            )
            return PropertyResolutionResult(
                value=1.0 / float(volume.value),
                source=volume.source,
                method='solid_molar_density_from_molar_volume',
                quality=volume.quality,
                notes=f'Inverse solid molar volume; units mol/dm^3; {volume.notes}',
            )


        def resolve_liquid_molar_density(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any] = None,
        ) -> PropertyResolutionResult:
            """Resolve pure liquid molar density in mol/dm^3."""
            props = self._coerce_props(symbol, props)
            coolprop_volume = self._coolprop_saturated_liquid_volume(symbol, T, props)
            if coolprop_volume and coolprop_volume.value:
                return PropertyResolutionResult(
                    value=1.0 / float(coolprop_volume.value),
                    source=coolprop_volume.source,
                    method=coolprop_volume.method.replace('liquid_volume', 'liquid_density'),
                    quality=coolprop_volume.quality,
                    notes=f'Converted from liquid molar volume; {coolprop_volume.notes}',
                )

            provided_fit = self._evaluate_provided_correlation(props, 'rhol', T)
            if provided_fit:
                rho_kg_m3, correlation = provided_fit
                mw = props.get('MW')
                if rho_kg_m3 > 0 and mw:
                    return self._provided_correlation_result(
                        rho_kg_m3 / mw,
                        correlation,
                        'provided_liquid_density_fit',
                        'mass density fit converted to mol/dm^3',
                        default_quality=0.98,
                    )

            if self._is_pfd_correlation_override(props, 'rhol'):
                try:
                    volume = self.resolve_liquid_molar_volume(symbol, T, props)
                except PropertyResolutionError:
                    volume = None
                if volume and volume.value is not None and volume.value > 0.0:
                    return PropertyResolutionResult(
                        value=1.0 / float(volume.value),
                        source=volume.source,
                        method=volume.method.replace('molar_volume', 'density').replace('liquid_volume', 'liquid_density'),
                        quality=volume.quality,
                        notes=f"Converted from liquid molar volume; {volume.notes}",
                    )

            perry_density = self._get_perry_evaluation(symbol, props, "liquid_molar_density_mol_per_dm3", T)
            if perry_density:
                return PropertyResolutionResult(
                    value=perry_density.value,
                    source='local',
                    method=perry_density.method,
                    quality=0.98,
                    notes=f"{perry_density.source}; units {perry_density.units}"
                )

            volume = self.resolve_liquid_molar_volume(symbol, T, props)
            if volume.value is not None and volume.value > 0.0:
                return PropertyResolutionResult(
                    value=1.0 / float(volume.value),
                    source=volume.source,
                    method=volume.method.replace('molar_volume', 'density').replace('liquid_volume', 'liquid_density'),
                    quality=volume.quality,
                    notes=f"Converted from liquid molar volume; {volume.notes}",
                )
            raise PropertyResolutionError(
                f"Cannot determine liquid molar density for '{symbol}' at T={T:.1f}K."
            )


        def resolve_liquid_molar_volume(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any] = None,
        ) -> PropertyResolutionResult:
            """Resolve pure liquid molar volume in m^3/kmol."""
            props = self._coerce_props(symbol, props)
            coolprop_volume = self._coolprop_saturated_liquid_volume(symbol, T, props)
            if coolprop_volume:
                return coolprop_volume

            provided_fit = self._provided_rhol_volume_at(props, T)
            if provided_fit:
                volume, correlation = provided_fit
                return self._provided_correlation_result(
                    volume,
                    correlation,
                    'provided_liquid_molar_volume_fit',
                    'mass density fit converted to m^3/kmol',
                    default_quality=0.98,
                )

            pfd_critical = None
            if self._is_pfd_correlation_override(props, 'rhol'):
                pfd_critical = self._critical_results_for_liquid_volume(symbol, props)
                pfd_rackett = self._rackett_fitted_volume(symbol, T, props, pfd_critical)
                if pfd_rackett and pfd_rackett.value is not None:
                    return pfd_rackett

            perry_volume = self._get_perry_evaluation(symbol, props, "liquid_molar_volume_m3_per_kmol", T)
            if perry_volume:
                return PropertyResolutionResult(
                    value=perry_volume.value,
                    source='local',
                    method=perry_volume.method,
                    quality=0.98,
                    notes=f"{perry_volume.source}; units {perry_volume.units}"
                )

            provided_extrapolated = self._provided_rhol_small_extrapolation(props, T)
            if provided_extrapolated:
                volume, correlation, evaluation_T = provided_extrapolated
                source_result = self._provided_correlation_result(
                    volume,
                    correlation,
                    'provided_liquid_molar_volume_fit',
                    'mass density fit converted to m^3/kmol',
                    default_quality=0.98,
                )
                return PropertyResolutionResult(
                    value=volume,
                    source=source_result.source,
                    method='provided_liquid_molar_volume_fit_extrapolated',
                    quality=self._combine_quality([source_result], method_factor=0.96),
                    notes=(
                        f"{source_result.notes}; extrapolated liquid-density correlation "
                        f"to T={evaluation_T:g} K within {LIQUID_VOLUME_EXTRAPOLATION_LIMIT_K:g} K"
                    ),
                )

            perry_extrapolated = self._perry_liquid_volume_small_extrapolation(symbol, props, T)
            if perry_extrapolated:
                volume, row, evaluation_T = perry_extrapolated
                source_result = PropertyResolutionResult(
                    value=volume,
                    source='local',
                    method=f"perry_molar_volume_eq{row.get('equation_id')}",
                    quality=0.98,
                    notes=f"Perry 9th Table {row.get('source_table')}; units m^3/kmol",
                )
                return PropertyResolutionResult(
                    value=volume,
                    source='local',
                    method=f"perry_molar_volume_eq{row.get('equation_id')}_extrapolated",
                    quality=self._combine_quality([source_result], method_factor=0.96),
                    notes=(
                        f"{source_result.notes}; extrapolated Perry liquid-density correlation "
                        f"to T={evaluation_T:g} K within {LIQUID_VOLUME_EXTRAPOLATION_LIMIT_K:g} K"
                    ),
                )

            critical = pfd_critical or self._critical_results_for_liquid_volume(symbol, props)
            Tc_result = self._critical_value_result(critical, 'Tc')
            Pc_result = self._critical_value_result(critical, 'Pc')
            omega_result = self._critical_value_result(critical, 'omega')
            zc_result = self._critical_value_result(critical, 'Zc')

            def estimated_quality(keys: tuple[str, ...], method_factor: float) -> float:
                inputs = [self._critical_value_result(critical, key) for key in keys]
                if any(item is None for item in inputs):
                    return 0.0
                return self._combine_quality(inputs, method_factor=method_factor)

            ptv_max_quality = (
                estimated_quality(('Tc', 'Pc', 'omega', 'Zc'), 0.84)
                if zc_result
                and self._result_quality(zc_result, 0.0) >= SOFT_PROPERTY_QUALITY_THRESHOLD
                and not self._result_is_soft(zc_result)
                else 0.0
            )
            yamada_gunn_max_quality = (
                estimated_quality(('Tc', 'Pc', 'omega'), 0.78)
                if Tc_result and Pc_result and omega_result
                else 0.0
            )
            pr_max_quality = (
                estimated_quality(('Tc', 'Pc', 'omega'), 0.75)
                if Tc_result and Pc_result and omega_result
                else 0.0
            )
            formula_max_quality = 0.60 if (props.get('formula') or props.get('Formula') or symbol) else 0.40

            prioritized_fallbacks = [
                (0, 1.00, lambda: self._rackett_fitted_volume(symbol, T, props, critical)),
                (1, 0.92, lambda: self._pubchem_liquid_volume(symbol, T, props, critical)),
            ]
            prioritized_fallbacks.extend(sorted(
                (
                    (2, ptv_max_quality, lambda: self._ptv_liquid_volume(symbol, T, props, critical)),
                    (3, yamada_gunn_max_quality, lambda: self._rackett_yamada_gunn_volume(T, critical)),
                    (4, pr_max_quality, lambda: self._pr_liquid_volume(symbol, T, props, critical)),
                    (5, formula_max_quality, lambda: self._formula_liquid_volume_estimate(symbol, T, props)),
                ),
                key=lambda item: (-item[1], item[0]),
            ))

            best_result = None
            best_priority = math.inf
            for index, (priority, max_quality, resolver) in enumerate(prioritized_fallbacks):
                best_quality = self._result_quality(best_result, 0.0)
                remaining_max_quality = max(
                    candidate[1] for candidate in prioritized_fallbacks[index:]
                )
                if best_result and best_quality >= remaining_max_quality:
                    break
                if best_result and best_quality >= max_quality:
                    continue
                result = resolver()
                if result and result.value is not None:
                    result_quality = self._result_quality(result, 0.0)
                    if (
                        best_result is None
                        or result_quality > best_quality
                        or (
                            math.isclose(result_quality, best_quality)
                            and priority < best_priority
                        )
                    ):
                        best_result = result
                        best_priority = priority
            if best_result:
                return best_result

            raise PropertyResolutionError(
                f"Cannot determine liquid molar volume for '{symbol}' at T={T:.1f}K."
            )


        def _liquid_molar_volume_boundary_temperatures(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any],
        ) -> list[float]:
            """Return supported liquid-volume endpoint temperatures nearest to T."""
            endpoints = []

            def add_endpoint(Tmin, Tmax):
                if Tmin is None or Tmax is None:
                    return
                try:
                    Tmin_f = float(Tmin)
                    Tmax_f = float(Tmax)
                except (TypeError, ValueError):
                    return
                if T < Tmin_f - 1e-9:
                    endpoints.append(Tmin_f)
                elif T > Tmax_f + 1e-9:
                    endpoints.append(Tmax_f)

            correlation = self._correlation_for(props, 'rhol')
            if correlation:
                add_endpoint(correlation.get('Tmin_K'), correlation.get('Tmax_K'))

            library = self._get_perry_library()
            if library is not None:
                for candidate in self._identifier_candidates(symbol, props):
                    entry = library.get(candidate)
                    if not entry:
                        continue
                    for row in entry.get('liquid_density', []):
                        if row.get('equation_id') not in {100, 105}:
                            continue
                        add_endpoint(row.get('T_min_K'), row.get('T_max_K'))

            unique = sorted({round(value, 9): value for value in endpoints}.values())
            return sorted(unique, key=lambda endpoint: abs(endpoint - T))


        def resolve_liquid_molar_volume_nearest(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any] = None,
        ) -> PropertyResolutionResult:
            """
            Resolve pure liquid molar volume in m^3/kmol, clamping to the nearest
            supported endpoint when T is outside available liquid-density fits.
            """
            props = self._coerce_props(symbol, props)
            try:
                return self.resolve_liquid_molar_volume(symbol, T, props)
            except PropertyResolutionError as original_error:
                last_error = original_error

            for boundary_T in self._liquid_molar_volume_boundary_temperatures(symbol, T, props):
                try:
                    result = self.resolve_liquid_molar_volume(symbol, boundary_T, props)
                except PropertyResolutionError as exc:
                    last_error = exc
                    continue
                notes = result.notes
                notes = f"{notes}; " if notes else ''
                notes += (
                    f"requested T={T:.1f} K outside liquid-volume correlation range; "
                    f"used nearest endpoint T={boundary_T:.1f} K"
                )
                return PropertyResolutionResult(
                    value=result.value,
                    source=result.source,
                    method=f"{result.method}_nearest_temperature",
                    quality=self._combine_quality([result], method_factor=0.80),
                    notes=notes,
                )

            raise last_error
