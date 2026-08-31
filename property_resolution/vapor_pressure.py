from .common import *

from contextlib import closing
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import sqlite3
from typing import Callable, Mapping

import numpy as np
from scipy.optimize import brentq

from .cache_expiration import runtime_cache_row_is_fresh

from .vapor_pressure_adapter import (
    DEFAULT_PSAT_MINIMUM_PRESSURE_BAR,
    PSAT_MINIMUM_BOILING_POINT_QUALITY,
    PsatCanonicalizationAdapter,
)
from .vapor_pressure_canonical import (
    CanonicalPsatFitDiagnostics,
    CanonicalPsatForm,
    CanonicalPsatCurve,
    CanonicalPsatFitPolicy,
    CanonicalPsatFitter,
    PsatAssembly,
    PsatAnchorRegistry,
    PsatCanonicalizationError,
    PsatCompletionCoordinator,
    PsatSegmentAssembler,
    PsatSegmentProvenance,
    PsatSegmentSlice,
    PsatSegmentType,
)


# Bump this whenever canonical fitting policy, source arbitration, or the
# persistent row contract changes in a way that can alter a fitted curve.
CANONICAL_PSAT_CACHE_VERSION = 14
CANONICAL_PSAT_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "runtime"
    / "canonical_psat_cache.sqlite"
)


class _CanonicalVaporPressureRuntime:
        """Cached canonical curve, checked evaluator, and tight-loop payload."""

        def __init__(
            self,
            curve: CanonicalPsatCurve,
            evaluator: Callable[[float], PropertyResolutionResult],
        ):
            self.curve = curve
            self.evaluator = evaluator
            self.coefficients = np.asarray(
                (
                    curve.A,
                    curve.B,
                    curve.C,
                    curve.D,
                    curve.E,
                    curve.F,
                    curve.G,
                    curve.H,
                    curve.T_critical,
                    0 if curve.inverse_power is None else curve.inverse_power,
                    curve.supercritical_slope,
                    curve.T_min,
                    curve.lower_continuation_slope,
                ),
                dtype=float,
            )


class VaporPressureMixin:
        CANONICAL_PSAT_CACHE_VERSION = CANONICAL_PSAT_CACHE_VERSION
        CANONICAL_PSAT_CACHE_PATH = CANONICAL_PSAT_CACHE_PATH

        @staticmethod
        def _select_antoine(
            candidates: list[AntoineCoefficients],
            T: Optional[float],
            require_in_range: bool = False,
        ) -> Optional[AntoineCoefficients]:
            if not candidates:
                return None
            if T is None:
                return candidates[0]

            in_range = [candidate for candidate in candidates if candidate.covers_temperature(T)]
            if in_range:
                return in_range[0]
            if require_in_range:
                return None
            return min(candidates, key=lambda candidate: candidate.distance_to_range(T))


        def _local_antoine_candidates(
            self,
            identifier: str,
            T: Optional[float] = None,
            props: Optional[Dict[str, Any]] = None,
        ) -> list[AntoineCoefficients]:
            """Gather local Antoine candidates without applying source shadowing."""
            result = []

            def add(candidate: Optional[AntoineCoefficients]):
                if candidate is None:
                    return
                key = (candidate.A, candidate.B, candidate.C, candidate.T_min, candidate.T_max, candidate.source)
                if all(
                    key != (item.A, item.B, item.C, item.T_min, item.T_max, item.source)
                    for item in result
                ):
                    result.append(candidate)

            add(self._get_provided_antoine(props))
            for candidate in self._identifier_candidates(identifier, props):
                add(self._get_textbook_antoine(candidate))
                add(self._get_table_antoine(candidate, T))
            return result


        @staticmethod
        def _antoine_cache_key(identifier: str, T: Optional[float] = None) -> str:
            if T is None:
                return f"antoine_{identifier}"
            return f"antoine_{identifier}_{T:.2f}K"


        @staticmethod
        def _antoine_missing_cache_key(identifier: str, T: Optional[float] = None) -> str:
            if T is None:
                return f"antoine_missing_{identifier}"
            band_low = 10.0 * math.floor(float(T) / 10.0)
            band_high = band_low + 10.0
            return f"antoine_missing_{identifier}_{band_low:.0f}_{band_high:.0f}K"


        def _get_cache(self, key: str) -> Optional[Dict]:
            """Get cached online data from memory or shared SQLite storage."""
            if key in self._online_cache:
                return self._online_cache[key]

            try:
                data = self._runtime_json_cache().get(key)
            except (OSError, sqlite3.Error, TypeError, ValueError):
                return None
            if data is not None:
                self._online_cache[key] = data
            return data


        def _set_cache(self, key: str, data: Dict):
            """Atomically persist online data in shared SQLite storage."""
            self._online_cache[key] = data
            try:
                self._runtime_json_cache().set(key, data)
            except (OSError, sqlite3.Error, TypeError, ValueError):
                pass


        def _set_missing_cache(self, key: str, reason: str = 'not_found'):
            """Remember a failed online lookup so it is not repeated."""
            self._set_cache(key, {'_missing': True, 'reason': reason})


        @staticmethod
        def _is_missing_cache(data: Optional[Dict]) -> bool:
            return bool(data and data.get('_missing'))


        @staticmethod
        def _is_transient_lookup_error(error: Exception) -> bool:
            """Return True for retryable network errors that should not be cached."""
            if isinstance(error, urllib.error.HTTPError):
                return error.code in (408, 429) or error.code >= 500
            return isinstance(error, (urllib.error.URLError, TimeoutError, OSError))


        @staticmethod
        def _strip_html_text(fragment: str) -> str:
            text = re.sub(r'<script.*?</script>', ' ', fragment, flags=re.S | re.I)
            text = re.sub(r'<style.*?</style>', ' ', text, flags=re.S | re.I)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = html_module.unescape(text).replace('\xa0', ' ')
            return re.sub(r'\s+', ' ', text).strip()


        @staticmethod
        def _first_number(text: str) -> Optional[float]:
            text = html_module.unescape(str(text)).replace('−', '-').replace(',', '')
            match = re.search(r'[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', text)
            if not match:
                return None
            try:
                return float(match.group(0))
            except ValueError:
                return None


        def _html_tables(self, html: str) -> list[tuple[str, list[list[str]]]]:
            tables = []
            for match in re.finditer(r'<table\b[^>]*>.*?</table>', html, flags=re.S | re.I):
                table_html = match.group(0)
                label_match = re.search(r'aria-label="([^"]+)"', table_html, flags=re.I)
                label = html_module.unescape(label_match.group(1)) if label_match else ''
                rows = []
                for row_html in re.findall(r'<tr\b[^>]*>(.*?)</tr>', table_html, flags=re.S | re.I):
                    cells = [
                        self._strip_html_text(cell)
                        for cell in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>', row_html, flags=re.S | re.I)
                    ]
                    if cells:
                        rows.append(cells)
                tables.append((label, rows))
            return tables


        @staticmethod
        def _median(values: list[float]) -> float:
            return float(median(values))


        @staticmethod
        def _linear_fit(x_values: list[float], y_values: list[float]) -> Optional[tuple[float, float]]:
            n = len(x_values)
            if n < 2:
                return None
            x_mean = sum(x_values) / n
            y_mean = sum(y_values) / n
            ss_xx = sum((x - x_mean) ** 2 for x in x_values)
            if ss_xx <= 1e-30:
                return None
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / ss_xx
            intercept = y_mean - slope * x_mean
            return slope, intercept


        def resolve_vapor_pressure(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any] = None,
            allow_online: bool = True,
            *,
            minimum_pressure_bar: Optional[float] = None,
        ) -> PropertyResolutionResult:
            """Resolve Psat from one cached, bounds-checked canonical curve."""
            allow_online = self._props_allow_online(props, allow_online)
            explicit_props = props is not None
            props = self._coerce_props(
                symbol,
                props,
                allow_online=explicit_props and allow_online,
            )
            minimum_pressure_bar = self._canonical_minimum_pressure_bar(
                props,
                minimum_pressure_bar,
            )
            runtime = self._canonical_vapor_pressure_runtime(
                symbol,
                props,
                allow_online=allow_online,
                minimum_pressure_bar=minimum_pressure_bar,
            )
            return runtime.evaluator(float(T))


        def resolve_vapor_pressure_coefficients(
            self,
            symbol: str,
            props: Dict[str, Any] = None,
            allow_online: bool = True,
            *,
            minimum_pressure_bar: Optional[float] = None,
        ) -> np.ndarray:
            """Return canonical coefficients and continuation parameters.

            ``G`` and ``H`` are zero when their terms are inactive, and
            ``inverse_power`` is zero when ``H`` is inactive. The final slope
            above ``Tc`` is ``dP/dT`` in bar/K. The final two entries are
            ``T_min`` and ``dlnP/dT`` there for the lower ``1/T`` continuation.
            The returned array is a copy so callers cannot mutate the cache.
            """
            allow_online = self._props_allow_online(props, allow_online)
            explicit_props = props is not None
            props = self._coerce_props(
                symbol,
                props,
                allow_online=explicit_props and allow_online,
            )
            minimum_pressure_bar = self._canonical_minimum_pressure_bar(
                props,
                minimum_pressure_bar,
            )
            runtime = self._canonical_vapor_pressure_runtime(
                symbol,
                props,
                allow_online=allow_online,
                minimum_pressure_bar=minimum_pressure_bar,
            )
            return runtime.coefficients.copy()


        def _canonical_vapor_pressure_runtime(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
            minimum_pressure_bar: float,
        ) -> _CanonicalVaporPressureRuntime:
            cache = getattr(self, "_canonical_vapor_pressure_curves", None)
            if cache is None:
                cache = {}
                self._canonical_vapor_pressure_curves = cache
            cache_key = self._canonical_vapor_pressure_cache_key(
                symbol,
                props,
                allow_online,
                minimum_pressure_bar,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                self._maybe_backfill_triple_pressure(
                    symbol,
                    props,
                    cached,
                    cache_key=cache_key,
                    allow_online=allow_online,
                )
                return cached
            persistent = self._load_persistent_canonical_vapor_pressure_runtime(
                symbol,
                props,
                cache_key,
            )
            if persistent is not None:
                cache[cache_key] = persistent
                self._maybe_backfill_triple_pressure(
                    symbol,
                    props,
                    persistent,
                    cache_key=cache_key,
                    allow_online=allow_online,
                )
                return persistent
            with self._online_attempt_scope(allow_online) as online_attempt:
                runtime = self._build_canonical_vapor_pressure_runtime(
                    symbol,
                    props,
                    allow_online=allow_online,
                    minimum_pressure_bar=minimum_pressure_bar,
                )
            has_direct_pin = any(
                item.segment_type in {
                    PsatSegmentType.CANONICAL_OVERRIDE.value,
                    PsatSegmentType.PINNED.value,
                }
                for item in runtime.curve.provenance
            )
            pending_tb_validation = any(
                item.metadata.get('tb_validation_required') is True
                and item.metadata.get('tb_validation_status')
                == 'validation_unavailable'
                and item.metadata.get('quality_basis')
                == 'standalone_unvalidated'
                for item in runtime.curve.provenance
            )
            if pending_tb_validation:
                online_attempt.record(OnlineAttemptState.NOT_ATTEMPTED)
            elif not allow_online and has_direct_pin:
                online_attempt.record(OnlineAttemptState.NOT_NEEDED)
            if self._online_attempt_is_persistable(
                allow_online,
                online_attempt.state,
            ):
                has_pfd_override = (
                    self._is_pfd_correlation_override(props, 'Psat')
                    or any(
                        self._is_pfd_component_override(props, name)
                        for name in (
                            'Tb', 'Tm', 'Tt', 'Pt',
                            'Tc', 'Pc', 'Vc', 'Zc', 'omega',
                        )
                    )
                )
                if not has_pfd_override:
                    self._store_persistent_canonical_vapor_pressure_runtime(
                        symbol,
                        props,
                        cache_key,
                        runtime,
                        online_attempt_state=online_attempt.state,
                    )
                cache[cache_key] = runtime
                self._maybe_backfill_triple_pressure(
                    symbol,
                    props,
                    runtime,
                    cache_key=cache_key,
                    allow_online=allow_online,
                )
            return runtime


        def _maybe_backfill_triple_pressure(
            self,
            symbol: str,
            props: Mapping[str, Any],
            runtime: _CanonicalVaporPressureRuntime,
            *,
            cache_key: tuple[str, str, bool, float],
            allow_online: bool,
        ) -> Optional[PropertyResolutionResult]:
            if self._canonical_positive_value(props.get('Pt')) is not None:
                return None
            if (
                self._is_pfd_component_override(props, 'Tt')
                or self._is_pfd_component_override(props, 'Pt')
                or self._is_pfd_correlation_override(props, 'Psat')
            ):
                return None
            triple_temperature = self._source_result_for_value(
                dict(props),
                'Tt',
                units='K',
            )
            if triple_temperature is None or triple_temperature.value is None:
                return None
            try:
                temperature = float(triple_temperature.value)
            except (TypeError, ValueError):
                return None
            if (
                not math.isfinite(temperature)
                or temperature <= 0.0
                or not runtime.curve.covers_temperature(temperature)
            ):
                return None
            canonical_fingerprint = (
                self._canonical_persistent_input_fingerprint(
                    props,
                    cache_key,
                )
            )
            memory_key = (
                self._canonical_persistent_component_identity(
                    symbol,
                    props,
                )[0],
                self._triple_temperature_fingerprint(triple_temperature),
                bool(allow_online),
                canonical_fingerprint,
            )
            completed = getattr(
                self,
                '_completed_triple_pressure_backfills',
                None,
            )
            if completed is None:
                completed = set()
                self._completed_triple_pressure_backfills = completed
            if memory_key in completed:
                return None
            try:
                psat_result = runtime.evaluator(temperature)
                pressure = float(psat_result.value)
            except (
                ArithmeticError,
                PropertyResolutionError,
                TypeError,
                ValueError,
            ):
                return None
            if (
                not math.isfinite(pressure)
                or pressure <= 0.0
                or pressure >= runtime.curve.P_critical_bar
            ):
                return None
            result = self._store_canonical_triple_pressure_backfill(
                symbol,
                props,
                triple_temperature=triple_temperature,
                psat_result=psat_result,
                allow_online=allow_online,
                canonical_input_fingerprint=canonical_fingerprint,
            )
            if result is not None:
                completed.add(memory_key)
            return result


        @staticmethod
        def _canonical_persistent_component_identity(
            symbol: str,
            props: Mapping[str, Any],
        ) -> tuple[str, str, str]:
            cas = str(props.get("CAS") or props.get("cas") or "").strip()
            name = str(props.get("name") or symbol).strip()
            if cas:
                return f"cas:{cas.lower()}", cas, name
            normalized = "".join(
                character.lower()
                for character in str(symbol)
                if character.isalnum()
            )
            return f"symbol:{normalized or str(symbol).lower()}", "", name


        @classmethod
        def _canonical_persistent_input_fingerprint(
            cls,
            props: Mapping[str, Any],
            cache_key: tuple[str, str, bool, float],
        ) -> str:
            relevant_fields = (
                "CAS", "cas", "formula", "smiles", "SMILES", "source",
                "Tc", "Pc", "Vc", "Zc", "omega", "Tb", "Tm", "Tt",
                "Pt", "Hvap", "Hsub", "MW", "critical_properties_unavailable",
                "antoine_A", "antoine_B", "antoine_C", "antoine_Tmin",
                "antoine_Tmax", "antoine_source", "property_correlations",
            )
            relevant_props = {
                field_name: props.get(field_name)
                for field_name in relevant_fields
                if field_name in props
            }
            if not (props.get("CAS") or props.get("cas")):
                relevant_props["symbol"] = props.get("symbol")
                relevant_props["name"] = props.get("name")
            source_fields = {
                "Tc", "Pc", "Vc", "Zc", "omega", "Tb", "Tm", "Tt",
                "Pt", "Hvap", "Hsub", "MW", "Psat", "antoine",
            }
            property_sources = props.get("property_sources")
            if isinstance(property_sources, Mapping):
                relevant_props["property_sources"] = {
                    str(field_name): metadata
                    for field_name, metadata in property_sources.items()
                    if str(field_name) in source_fields
                }
            payload = json.dumps(
                {
                    "properties": relevant_props,
                    "allow_online": cache_key[2],
                    "minimum_pressure_bar": cache_key[3],
                },
                sort_keys=True,
                separators=(",", ":"),
                default=cls._canonical_cache_json_default,
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()


        @staticmethod
        def _canonical_cache_json_default(value: Any):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, np.generic):
                return value.item()
            if is_dataclass(value):
                return asdict(value)
            if isinstance(value, (set, frozenset)):
                return sorted(value, key=repr)
            return repr(value)


        @classmethod
        def _canonical_cache_json(cls, value: Any) -> str:
            return json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                default=cls._canonical_cache_json_default,
            )


        def _ensure_canonical_psat_cache_schema(self, connection) -> None:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS canonical_psat_cache (
                    cache_version INTEGER NOT NULL,
                    component_key TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    cas TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    allow_online INTEGER NOT NULL,
                    online_attempt_state TEXT NOT NULL,
                    minimum_pressure_bar REAL NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    A REAL NOT NULL,
                    B REAL NOT NULL,
                    C REAL NOT NULL,
                    D REAL NOT NULL,
                    E REAL NOT NULL,
                    F REAL NOT NULL,
                    G REAL NOT NULL,
                    H REAL NOT NULL,
                    T_min REAL NOT NULL,
                    T_critical REAL NOT NULL,
                    P_critical_bar REAL NOT NULL,
                    inverse_power INTEGER,
                    form TEXT NOT NULL,
                    T_boiling REAL,
                    quality REAL NOT NULL,
                    supercritical_slope REAL NOT NULL,
                    lower_continuation_slope REAL NOT NULL,
                    provenance_json TEXT NOT NULL,
                    diagnostics_json TEXT,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (
                        cache_version,
                        component_key,
                        input_fingerprint
                    )
                )
                """
            )
            existing_columns = {
                str(row[1])
                for row in connection.execute(
                    'PRAGMA table_info(canonical_psat_cache)'
                )
            }
            if 'online_attempt_state' not in existing_columns:
                connection.execute(
                    'ALTER TABLE canonical_psat_cache '
                    "ADD COLUMN online_attempt_state TEXT NOT NULL "
                    "DEFAULT 'not_attempted'"
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS canonical_psat_cache_cas_idx
                ON canonical_psat_cache (cache_version, cas)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS canonical_psat_cache_name_idx
                ON canonical_psat_cache (cache_version, component_name)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS canonical_psat_cache_symbol_idx
                ON canonical_psat_cache (cache_version, symbol)
                """
            )


        def initialize_canonical_vapor_pressure_disk_cache(self) -> Path:
            """Create the persistent canonical Psat cache and return its path."""
            path = Path(self.CANONICAL_PSAT_CACHE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                self._ensure_canonical_psat_cache_schema(connection)
                connection.execute(
                    f"PRAGMA user_version = "
                    f"{int(self.CANONICAL_PSAT_CACHE_VERSION)}"
                )
                connection.commit()
            return path


        def _load_persistent_canonical_vapor_pressure_runtime(
            self,
            symbol: str,
            props: Mapping[str, Any],
            cache_key: tuple[str, str, bool, float],
        ) -> Optional[_CanonicalVaporPressureRuntime]:
            path = Path(self.CANONICAL_PSAT_CACHE_PATH)
            if not path.exists():
                return None
            component_key, _cas, _name = (
                self._canonical_persistent_component_identity(symbol, props)
            )
            fingerprint = self._canonical_persistent_input_fingerprint(
                props,
                cache_key,
            )
            try:
                with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                    connection.row_factory = sqlite3.Row
                    self._ensure_canonical_psat_cache_schema(connection)
                    row = connection.execute(
                        """
                        SELECT * FROM canonical_psat_cache
                        WHERE cache_version = ?
                          AND component_key = ?
                          AND input_fingerprint = ?
                        """,
                        (
                            int(self.CANONICAL_PSAT_CACHE_VERSION),
                            component_key,
                            fingerprint,
                        ),
                    ).fetchone()
                if row is None:
                    return None
                if not runtime_cache_row_is_fresh(
                    path,
                    row['updated_at_utc'],
                ):
                    return None
                if not self._online_attempt_is_persistable(
                    bool(row['allow_online']),
                    str(row['online_attempt_state']),
                ):
                    return None
                provenance_payload = json.loads(row["provenance_json"])
                provenance = tuple(
                    PsatSegmentProvenance(
                        source=str(item["source"]),
                        method=str(item["method"]),
                        segment_type=str(item["segment_type"]),
                        priority=int(item["priority"]),
                        quality=float(item["quality"]),
                        T_min=float(item["T_min"]),
                        T_max=float(item["T_max"]),
                        allow_junction_slope_mismatch=bool(
                            item.get("allow_junction_slope_mismatch", False)
                        ),
                        context=dict(item.get("context") or {}),
                        metadata=dict(item.get("metadata") or {}),
                    )
                    for item in provenance_payload
                )
                diagnostics_payload = (
                    json.loads(row["diagnostics_json"])
                    if row["diagnostics_json"]
                    else None
                )
                diagnostics = (
                    CanonicalPsatFitDiagnostics(**diagnostics_payload)
                    if diagnostics_payload is not None
                    else None
                )
                metadata = json.loads(row["metadata_json"])
                if not isinstance(metadata, dict):
                    return None
                curve = CanonicalPsatCurve(
                    A=float(row["A"]),
                    B=float(row["B"]),
                    C=float(row["C"]),
                    D=float(row["D"]),
                    E=float(row["E"]),
                    F=float(row["F"]),
                    G=float(row["G"]),
                    H=float(row["H"]),
                    T_min=float(row["T_min"]),
                    T_critical=float(row["T_critical"]),
                    P_critical_bar=float(row["P_critical_bar"]),
                    inverse_power=(
                        None
                        if row["inverse_power"] is None
                        else int(row["inverse_power"])
                    ),
                    form=CanonicalPsatForm(str(row["form"])),
                    T_boiling=(
                        None
                        if row["T_boiling"] is None
                        else float(row["T_boiling"])
                    ),
                    quality=float(row["quality"]),
                    provenance=provenance,
                    diagnostics=diagnostics,
                    metadata=metadata,
                )
                for stored, derived in (
                    (row["supercritical_slope"], curve.supercritical_slope),
                    (
                        row["lower_continuation_slope"],
                        curve.lower_continuation_slope,
                    ),
                ):
                    if not math.isclose(
                        float(stored),
                        float(derived),
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-12,
                    ):
                        return None
                evaluator = self._canonical_checked_evaluator(curve, None)
                return _CanonicalVaporPressureRuntime(curve, evaluator)
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                OSError,
                sqlite3.Error,
            ):
                return None


        def _store_persistent_canonical_vapor_pressure_runtime(
            self,
            symbol: str,
            props: Mapping[str, Any],
            cache_key: tuple[str, str, bool, float],
            runtime: _CanonicalVaporPressureRuntime,
            *,
            online_attempt_state: OnlineAttemptState | str,
        ) -> None:
            curve = runtime.curve
            component_key, cas, component_name = (
                self._canonical_persistent_component_identity(symbol, props)
            )
            fingerprint = self._canonical_persistent_input_fingerprint(
                props,
                cache_key,
            )
            provenance_json = self._canonical_cache_json([
                asdict(item)
                for item in curve.provenance
            ])
            diagnostics_json = (
                None
                if curve.diagnostics is None
                else self._canonical_cache_json(asdict(curve.diagnostics))
            )
            metadata_json = self._canonical_cache_json(dict(curve.metadata))
            path = Path(self.CANONICAL_PSAT_CACHE_PATH)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                    self._ensure_canonical_psat_cache_schema(connection)
                    connection.execute(
                        f"PRAGMA user_version = "
                        f"{int(self.CANONICAL_PSAT_CACHE_VERSION)}"
                    )
                    connection.execute(
                        """
                        INSERT INTO canonical_psat_cache (
                            cache_version, component_key, input_fingerprint,
                            symbol, cas, component_name, allow_online,
                            online_attempt_state,
                            minimum_pressure_bar, created_at_utc, updated_at_utc,
                            A, B, C, D, E, F, G, H,
                            T_min, T_critical, P_critical_bar,
                            inverse_power, form, T_boiling, quality,
                            supercritical_slope, lower_continuation_slope,
                            provenance_json, diagnostics_json, metadata_json
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                            strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        ON CONFLICT (
                            cache_version, component_key, input_fingerprint
                        ) DO UPDATE SET
                            updated_at_utc = excluded.updated_at_utc,
                            symbol = excluded.symbol,
                            cas = excluded.cas,
                            component_name = excluded.component_name,
                            allow_online = excluded.allow_online,
                            online_attempt_state = excluded.online_attempt_state,
                            minimum_pressure_bar = excluded.minimum_pressure_bar,
                            A = excluded.A,
                            B = excluded.B,
                            C = excluded.C,
                            D = excluded.D,
                            E = excluded.E,
                            F = excluded.F,
                            G = excluded.G,
                            H = excluded.H,
                            T_min = excluded.T_min,
                            T_critical = excluded.T_critical,
                            P_critical_bar = excluded.P_critical_bar,
                            inverse_power = excluded.inverse_power,
                            form = excluded.form,
                            T_boiling = excluded.T_boiling,
                            quality = excluded.quality,
                            supercritical_slope = excluded.supercritical_slope,
                            lower_continuation_slope = excluded.lower_continuation_slope,
                            provenance_json = excluded.provenance_json,
                            diagnostics_json = excluded.diagnostics_json,
                            metadata_json = excluded.metadata_json
                        """,
                        (
                            int(self.CANONICAL_PSAT_CACHE_VERSION),
                            component_key,
                            fingerprint,
                            str(symbol),
                            cas,
                            component_name,
                            int(bool(cache_key[2])),
                            OnlineAttemptState(online_attempt_state).value,
                            float(cache_key[3]),
                            curve.A,
                            curve.B,
                            curve.C,
                            curve.D,
                            curve.E,
                            curve.F,
                            curve.G,
                            curve.H,
                            curve.T_min,
                            curve.T_critical,
                            curve.P_critical_bar,
                            curve.inverse_power,
                            curve.form.value,
                            curve.T_boiling,
                            curve.quality,
                            curve.supercritical_slope,
                            curve.lower_continuation_slope,
                            provenance_json,
                            diagnostics_json,
                            metadata_json,
                        ),
                    )
                    connection.commit()
            except (OSError, sqlite3.Error):
                return


        @staticmethod
        def _canonical_minimum_pressure_bar(
            props: Mapping[str, Any],
            explicit_value: Optional[float],
        ) -> float:
            value = (
                props.get("_psat_minimum_pressure_bar")
                if explicit_value is None
                else explicit_value
            )
            if value is None:
                value = DEFAULT_PSAT_MINIMUM_PRESSURE_BAR
            try:
                pressure_bar = float(value)
            except (TypeError, ValueError) as error:
                raise PropertyResolutionError(
                    "Canonical Psat minimum pressure must be positive"
                ) from error
            if not math.isfinite(pressure_bar) or pressure_bar <= 0.0:
                raise PropertyResolutionError(
                    "Canonical Psat minimum pressure must be positive"
                )
            return pressure_bar


        @staticmethod
        def _canonical_vapor_pressure_cache_key(
            symbol: str,
            props: Mapping[str, Any],
            allow_online: bool,
            minimum_pressure_bar: float,
        ) -> tuple[str, str, bool, float]:
            try:
                payload = json.dumps(
                    props,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=repr,
                )
            except (TypeError, ValueError):
                payload = repr(sorted(props.items(), key=lambda item: str(item[0])))
            return (
                str(symbol),
                hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                bool(allow_online),
                round(float(minimum_pressure_bar), 15),
            )


        @staticmethod
        def _canonical_quality(candidate: Optional[PropertyResolutionResult]) -> float:
            if candidate is None:
                return 0.0
            try:
                quality = float(candidate.quality)
            except (TypeError, ValueError):
                return 0.0
            return quality if math.isfinite(quality) and 0.0 <= quality <= 1.0 else 0.0


        @staticmethod
        def _canonical_positive_value(value: Any) -> Optional[float]:
            if isinstance(value, PropertyResolutionResult):
                value = value.value
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            return result if math.isfinite(result) and result > 0.0 else None


        @staticmethod
        def _canonical_finite_value(value: Any) -> Optional[float]:
            if isinstance(value, PropertyResolutionResult):
                value = value.value
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            return result if math.isfinite(result) else None


        def _build_canonical_vapor_pressure_runtime(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
            minimum_pressure_bar: float,
        ) -> _CanonicalVaporPressureRuntime:
            try:
                minimum_pressure_bar = float(minimum_pressure_bar)
            except (TypeError, ValueError) as error:
                raise PropertyResolutionError(
                    "Canonical Psat minimum pressure must be positive"
                ) from error
            if (
                not math.isfinite(minimum_pressure_bar)
                or minimum_pressure_bar <= 0.0
            ):
                raise PropertyResolutionError(
                    "Canonical Psat minimum pressure must be positive"
                )
            critical = self._canonical_critical_properties(
                symbol,
                props,
                allow_online=allow_online,
            )
            T_critical = self._canonical_positive_value(critical.get("Tc"))
            P_critical_bar = self._canonical_positive_value(critical.get("Pc"))
            if T_critical is None or P_critical_bar is None:
                raise PropertyResolutionError(
                    f"Canonical vapor pressure requires Tc and Pc for {symbol!r}"
                )
            critical_inputs = tuple(
                item
                for item in (
                    critical.get("_Tc_result"),
                    critical.get("_Pc_result"),
                )
                if isinstance(item, PropertyResolutionResult)
            )
            critical_quality = min(
                (self._canonical_quality(item) for item in critical_inputs),
                default=0.0,
            )

            try:
                boiling = self.resolve_boiling_point(
                    symbol,
                    props,
                    allow_online=allow_online,
                    allow_estimation=False,
                )
            except (ArithmeticError, LookupError, TypeError, ValueError):
                boiling = None
            T_boiling = (
                self._canonical_positive_value(boiling)
                if isinstance(boiling, PropertyResolutionResult)
                else None
            )
            boiling_quality = (
                self._canonical_quality(boiling)
                if isinstance(boiling, PropertyResolutionResult)
                else 0.0
            )
            component = dict(props)
            component.setdefault("symbol", str(symbol))
            component.setdefault("name", str(symbol))
            component["Tc"] = T_critical
            component["Pc"] = P_critical_bar
            omega = self._canonical_finite_value(critical.get("omega"))
            if omega is not None:
                component["omega"] = omega
            if T_boiling is not None:
                component["Tb"] = T_boiling
            source_metadata = dict(component.get("property_sources") or {})
            for field_name, candidate in (
                ("Tc", critical.get("_Tc_result")),
                ("Pc", critical.get("_Pc_result")),
                ("Tb", boiling),
            ):
                if not isinstance(candidate, PropertyResolutionResult):
                    continue
                source_metadata[field_name] = {
                    "source": candidate.source,
                    "method": candidate.method,
                    "quality": candidate.quality,
                    "notes": candidate.notes,
                }
            component["property_sources"] = source_metadata

            pressure_floor_T_min = self._canonical_broad_T_min(
                T_critical,
                P_critical_bar,
                omega,
                minimum_pressure_bar,
            )
            def resolved_hvap(_component: Any, temperature: float):
                return self.resolve_hvap(
                    symbol,
                    component,
                    T=float(temperature),
                    allow_online=allow_online,
                    allow_estimation=False,
                )

            adapter = PsatCanonicalizationAdapter(
                component,
                tsat_at_pressure=(
                    lambda _component, _pressure: pressure_floor_T_min
                ),
                hvap_at_temperature=resolved_hvap,
                allow_online_hvap=allow_online,
                minimum_pressure_bar=minimum_pressure_bar,
            )
            domain = adapter.resolve_domain(T_critical=T_critical)
            target_T_min = domain.T_min
            phase_change_anchor = adapter.select_phase_change_anchor(
                T_boiling=T_boiling,
                boiling_quality=boiling_quality,
            )
            T_boiling = phase_change_anchor.T_boiling
            boiling_quality = phase_change_anchor.boiling_quality
            if phase_change_anchor.uses_sublimation_anchor:
                component.pop("Tb", None)
                source_metadata.pop("Tb", None)
            elif (
                T_boiling is None
                or T_boiling < target_T_min
                or T_boiling >= T_critical
                or boiling_quality < PSAT_MINIMUM_BOILING_POINT_QUALITY
                or self._result_is_soft(boiling)
            ):
                T_boiling = None
                boiling_quality = 0.0
                component.pop("Tb", None)
                source_metadata.pop("Tb", None)
            if (
                allow_online
                and not self._is_pfd_correlation_override(component, 'Psat')
            ):
                self._prime_nist_antoine_cache(symbol, component)
            inputs = adapter.collect_inputs(
                T_min=target_T_min,
                T_critical=T_critical,
                P_critical_bar=P_critical_bar,
                T_boiling=T_boiling,
            )
            if inputs.canonical_override is not None:
                curve = inputs.canonical_override
                if domain.basis == "pressure_floor":
                    curve = self._canonical_curve_at_pressure_floor(
                        curve,
                        adapter.minimum_pressure_bar,
                        component,
                    )
                curve = replace(curve, metadata={
                    **curve.metadata,
                    "domain_selection": {
                        "basis": domain.basis,
                        "T_min": curve.T_min,
                        "source": domain.source,
                        "method": domain.method,
                        "quality": domain.quality,
                        "notes": domain.notes,
                        "pressure_bar": domain.pressure_bar,
                        "rejected_candidates": domain.rejected_candidates,
                    },
                })
                evaluator = self._canonical_checked_evaluator(curve, None)
                return _CanonicalVaporPressureRuntime(curve, evaluator)

            assembler = PsatSegmentAssembler(target_T_min, T_critical)
            assembler.extend(inputs.segments)
            context = {
                key: component[key]
                for key in ("symbol", "name", "CAS")
                if component.get(key) not in (None, "")
            }
            anchors = PsatAnchorRegistry.from_tb_tc(
                T_critical=T_critical,
                P_critical_bar=P_critical_bar,
                critical_quality=critical_quality,
                T_boiling=T_boiling,
                boiling_quality=boiling_quality if T_boiling is not None else None,
                context=context,
            )
            if phase_change_anchor.triple_anchor is not None:
                anchors.add(phase_change_anchor.triple_anchor)
            for anchor in inputs.additional_anchors:
                anchors.add(anchor)
            try:
                completion = PsatCompletionCoordinator(
                    assembler,
                    inputs.relations,
                    anchors,
                ).complete(require_complete=False)
                assembly = completion.assembly
                if domain.basis == "pressure_floor":
                    assembly = self._canonical_assembly_at_pressure_floor(
                        assembly,
                        adapter.minimum_pressure_bar,
                        component,
                    )
                metadata = dict(inputs.metadata)
                metadata["domain_selection"] = {
                    "basis": domain.basis,
                    "T_min": assembly.target_T_min,
                    "source": domain.source,
                    "method": domain.method,
                    "quality": domain.quality,
                    "notes": domain.notes,
                    "pressure_bar": domain.pressure_bar,
                    "rejected_candidates": domain.rejected_candidates,
                }
                if inputs.warnings:
                    metadata["input_warnings"] = inputs.warnings
                if completion.junction_warnings:
                    metadata["junction_warnings"] = completion.junction_warnings
                lower_ln_pressure = assembly.ln_pressure(
                    assembly.target_T_min
                )
                inverse_retry_powers = (
                    (-3, -5, -7)
                    if (
                        lower_ln_pressure
                        < math.log(DEFAULT_PSAT_MINIMUM_PRESSURE_BAR)
                        or minimum_pressure_bar
                        < DEFAULT_PSAT_MINIMUM_PRESSURE_BAR
                    )
                    else ()
                )
                fitted_lower_pressure_bar = (
                    math.exp(phase_change_anchor.triple_anchor.ln_pressure)
                    if (
                        phase_change_anchor.triple_anchor is not None
                        and abs(
                            assembly.target_T_min
                            - phase_change_anchor.triple_anchor.temperature
                        ) <= 1.0e-7
                    )
                    else None
                )
                curve = CanonicalPsatFitter(CanonicalPsatFitPolicy(
                    inverse_retry_powers=inverse_retry_powers,
                )).fit_assembly(
                    assembly,
                    np.linspace(
                        assembly.target_T_min,
                        assembly.target_T_max,
                        401,
                    ),
                    P_critical_bar=P_critical_bar,
                    P_min_bar=fitted_lower_pressure_bar,
                    T_boiling=T_boiling,
                    metadata=metadata,
                )
                if domain.basis == "pressure_floor":
                    curve, assembly = (
                        self._canonical_fitted_domain_at_pressure_floor(
                            curve,
                            assembly,
                            minimum_pressure_bar,
                        )
                    )
            except PsatCanonicalizationError as error:
                raise PropertyResolutionError(
                    f"Cannot canonicalize vapor pressure for {symbol!r}: {error}"
                ) from error
            evaluator = self._canonical_checked_evaluator(curve, assembly)
            return _CanonicalVaporPressureRuntime(curve, evaluator)


        @staticmethod
        def _canonical_broad_T_min(
            T_critical: float,
            P_critical_bar: float,
            omega: Optional[float],
            pressure_floor_bar: float,
        ) -> float:
            broad_omega = (
                -0.5
                if omega is None
                else max(-0.5, min(2.0, omega - 0.2))
            )
            target = math.log(pressure_floor_bar)

            def aw_ln_pressure(temperature: float) -> float:
                reduced_temperature = temperature / T_critical
                tau = 1.0 - reduced_temperature
                f0 = (
                    -5.97616 * tau
                    + 1.29874 * tau**1.5
                    - 0.60394 * tau**2.5
                    - 1.06841 * tau**5
                ) / reduced_temperature
                f1 = (
                    -5.03365 * tau
                    + 1.11505 * tau**1.5
                    - 5.41217 * tau**2.5
                    - 7.46628 * tau**5
                ) / reduced_temperature
                f2 = (
                    -0.64771 * tau
                    + 2.41539 * tau**1.5
                    - 4.26979 * tau**2.5
                    + 3.25259 * tau**5
                ) / reduced_temperature
                return (
                    math.log(P_critical_bar)
                    + f0
                    + broad_omega * f1
                    + broad_omega**2 * f2
                )

            lower = max(1.0e-3, 1.0e-4 * T_critical)
            upper = T_critical * (1.0 - 1.0e-10)
            try:
                if aw_ln_pressure(lower) >= target:
                    return lower
                return brentq(
                    lambda temperature: aw_ln_pressure(temperature) - target,
                    lower,
                    upper,
                    xtol=1.0e-10,
                    rtol=1.0e-12,
                )
            except (ArithmeticError, ValueError):
                return max(1.0e-3, 0.05 * T_critical)


        @staticmethod
        def _canonical_curve_at_pressure_floor(
            curve: CanonicalPsatCurve,
            pressure_floor_bar: float,
            component: Mapping[str, Any],
        ) -> CanonicalPsatCurve:
            target = math.log(pressure_floor_bar)
            lower = curve.ln_pressure(curve.T_min)
            upper = curve.ln_pressure(curve.T_critical)
            if upper < target - 1.0e-10:
                raise PropertyResolutionError(
                    "Direct canonical Psat curve does not reach the pressure floor"
                )
            if lower >= target - 1.0e-10:
                floor_T_min = curve.T_min
            else:
                floor_T_min = brentq(
                    lambda temperature: curve.ln_pressure(temperature) - target,
                    curve.T_min,
                    curve.T_critical,
                    xtol=1.0e-10,
                    rtol=1.0e-12,
                )
            selected_T_min = VaporPressureMixin._canonical_preferred_T_min(
                component,
                floor_T_min,
                curve.T_critical,
                pressure_floor_bar,
                lambda temperature: curve.pressure_bar(temperature),
            )
            return replace(curve, T_min=selected_T_min)


        @staticmethod
        def _canonical_assembly_at_pressure_floor(
            assembly: PsatAssembly,
            pressure_floor_bar: float,
            component: Mapping[str, Any],
        ) -> PsatAssembly:
            if not math.isfinite(pressure_floor_bar) or pressure_floor_bar <= 0.0:
                raise PsatCanonicalizationError(
                    "Canonical Psat pressure floor must be positive"
                )
            target = math.log(pressure_floor_bar)
            selected_T_min = None
            for item in assembly.slices:
                try:
                    lower = item.ln_pressure(item.T_min)
                    upper = item.ln_pressure(item.T_max)
                except PsatCanonicalizationError:
                    continue
                if max(lower, upper) < target - 1.0e-10:
                    continue
                if lower >= target - 1.0e-10:
                    selected_T_min = item.T_min
                else:
                    selected_T_min = brentq(
                        lambda value: item.ln_pressure(value) - target,
                        item.T_min,
                        item.T_max,
                        xtol=1.0e-10,
                        rtol=1.0e-12,
                    )
                break
            if selected_T_min is None:
                raise PsatCanonicalizationError(
                    "Canonical Psat assembly does not reach the pressure floor"
                )
            selected_T_min = VaporPressureMixin._canonical_preferred_T_min(
                component,
                selected_T_min,
                assembly.target_T_max,
                pressure_floor_bar,
                lambda temperature: math.exp(assembly.ln_pressure(temperature)),
            )
            slices = []
            for item in assembly.slices:
                lower = max(selected_T_min, item.T_min)
                upper = min(assembly.target_T_max, item.T_max)
                if upper <= lower + 1.0e-9:
                    continue
                slices.append(PsatSegmentSlice(item.segment, lower, upper))
            trimmed = PsatAssembly(
                target_T_min=selected_T_min,
                target_T_max=assembly.target_T_max,
                slices=tuple(slices),
            )
            if not trimmed.covers_target():
                gaps = ", ".join(
                    f"{gap.T_min:g}-{gap.T_max:g} K"
                    for gap in trimmed.coverage_gaps()
                )
                raise PsatCanonicalizationError(
                    "Canonical Psat sources are incomplete above the pressure "
                    f"floor; uncovered gaps: {gaps}"
                )
            trimmed.require_compatible_junctions()
            return trimmed


        @staticmethod
        def _canonical_fitted_domain_at_pressure_floor(
            curve: CanonicalPsatCurve,
            assembly: PsatAssembly,
            pressure_floor_bar: float,
        ) -> tuple[CanonicalPsatCurve, PsatAssembly]:
            fitted_pressure = curve.pressure_bar(curve.T_min)
            if fitted_pressure >= pressure_floor_bar * (1.0 - 1.0e-12):
                return curve, assembly
            target = math.log(pressure_floor_bar)
            T_min = brentq(
                lambda temperature: curve.ln_pressure(temperature) - target,
                curve.T_min,
                curve.T_critical,
                xtol=1.0e-10,
                rtol=1.0e-12,
            )
            slices = tuple(
                PsatSegmentSlice(
                    item.segment,
                    max(T_min, item.T_min),
                    item.T_max,
                )
                for item in assembly.slices
                if item.T_max > T_min + 1.0e-9
            )
            return (
                replace(curve, T_min=T_min),
                PsatAssembly(
                    target_T_min=T_min,
                    target_T_max=assembly.target_T_max,
                    slices=slices,
                ),
            )


        @staticmethod
        def _canonical_preferred_T_min(
            component: Mapping[str, Any],
            floor_T_min: float,
            T_critical: float,
            pressure_floor_bar: float,
            pressure_at_temperature: Callable[[float], float],
        ) -> float:
            for field_name in ("Tt", "Tm"):
                try:
                    temperature = float(component.get(field_name))
                except (TypeError, ValueError):
                    continue
                if (
                    not math.isfinite(temperature)
                    or temperature < floor_T_min - 1.0e-9
                    or temperature >= T_critical
                ):
                    continue
                pressure_bar = None
                if field_name == "Tt":
                    try:
                        pressure_bar = float(component.get("Pt"))
                    except (TypeError, ValueError):
                        pressure_bar = None
                    if (
                        pressure_bar is not None
                        and (
                            not math.isfinite(pressure_bar)
                            or pressure_bar <= 0.0
                        )
                    ):
                        pressure_bar = None
                if pressure_bar is None:
                    try:
                        pressure_bar = float(
                            pressure_at_temperature(temperature)
                        )
                    except (
                        ArithmeticError,
                        PsatCanonicalizationError,
                        TypeError,
                        ValueError,
                    ):
                        continue
                if (
                    math.isfinite(pressure_bar)
                    and pressure_bar >= pressure_floor_bar
                ):
                    return temperature
            return floor_T_min


        @staticmethod
        def _canonical_checked_evaluator(
            curve: CanonicalPsatCurve,
            assembly: Optional[PsatAssembly],
        ) -> Callable[
            [float],
            PropertyResolutionResult,
        ]:
            def evaluate(temperature: float) -> PropertyResolutionResult:
                if not math.isfinite(temperature) or not curve.covers_temperature(
                    temperature
                ):
                    raise PropertyResolutionError(
                        f"Temperature {temperature!r} K is outside canonical Psat "
                        f"range {curve.T_min:g}-{curve.T_critical:g} K"
                    )
                pressure_bar = curve.pressure_bar(temperature)
                if assembly is not None:
                    item = assembly.slice_at(temperature)
                else:
                    item = None
                if item is not None:
                    source = item.segment.source
                    method = item.segment.method
                    base_quality = item.segment.quality
                else:
                    candidates = [
                        provenance
                        for provenance in curve.provenance
                        if provenance.T_min - 1.0e-9
                        <= temperature
                        <= provenance.T_max + 1.0e-9
                    ]
                    provenance = (
                        max(
                            candidates,
                            key=lambda item: (item.priority, item.quality),
                        )
                        if candidates
                        else None
                    )
                    source = provenance.source if provenance else "calculated"
                    method = provenance.method if provenance else "canonical_psat"
                    base_quality = (
                        provenance.quality
                        if provenance
                        else float(
                            curve.metadata.get(
                                "fit_original_overall_quality",
                                curve.quality,
                            )
                        )
                    )
                quality_penalty = float(
                    curve.metadata.get("fit_quality_penalty", 0.0)
                )
                quality = max(0.0, base_quality - quality_penalty)
                diagnostics = curve.diagnostics
                fit_note = (
                    ""
                    if diagnostics is None
                    else (
                        f"; canonical MARD={diagnostics.mard_percent:.6g}%, "
                        f"max error={diagnostics.max_absolute_relative_error_percent:.6g}%"
                    )
                )
                penalty_note = (
                    ""
                    if quality_penalty <= 0.0
                    else f"; fit quality penalty={quality_penalty:.6g}"
                )
                return PropertyResolutionResult(
                    value=pressure_bar,
                    source=source,
                    method=method,
                    quality=quality,
                    notes=(
                        f"Canonical {curve.form.value} Psat; qualified range "
                        f"{curve.T_min:g}-{curve.T_critical:g} K"
                        f"{fit_note}{penalty_note}"
                    ),
                )

            return evaluate


        def _canonical_critical_properties(
            self,
            symbol: str,
            props: Dict[str, Any],
            allow_online: bool = True,
        ) -> Dict[str, Any]:
            """Resolve the Tc/Pc/omega inputs used during canonicalization."""
            result = {
                'Tc': props.get('Tc'),
                'Pc': props.get('Pc'),
                'omega': props.get('omega'),
                '_Tc_result': self._source_result_for_value(props, 'Tc', units='K'),
                '_Pc_result': self._source_result_for_value(props, 'Pc', units='bar'),
                '_omega_result': self._source_result_for_value(props, 'omega'),
                '_critical_source': 'provided',
            }
            if result['Tc'] is not None and result['Pc'] is not None and result['omega'] is not None:
                return result

            try:
                resolved = self.resolve_critical_properties(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
            except Exception:
                return result

            sources = []
            for name in ('Tc', 'Pc', 'omega'):
                if result[name] is not None:
                    continue
                item = resolved.get(name)
                if item and item.value is not None:
                    result[name] = item.value
                    result[f'_{name}_result'] = item
                    sources.append(item.source)
            if sources:
                result['_critical_source'] = '/'.join(dict.fromkeys(sources))
            return result


        def get_antoine_local(
            self,
            symbol: str,
            T: Optional[float] = None,
            props: Optional[Dict[str, Any]] = None,
            require_in_range: bool = False,
        ) -> Optional[AntoineCoefficients]:
            """Get Antoine coefficients from local database"""
            return self._select_antoine(
                self._local_antoine_candidates(symbol, T, props),
                T,
                require_in_range=require_in_range,
            )


        def get_antoine_online(
            self,
            symbol: str,
            T: Optional[float] = None,
            props: Optional[Dict[str, Any]] = None,
            require_in_range: bool = False,
        ) -> Optional[AntoineCoefficients]:
            """
            Fetch Antoine coefficients from online databases.

            Tries PubChem and NIST WebBook.
            """
            identifiers = self._identifier_candidates(symbol, props)
            if not identifiers:
                self._record_online_attempt_state(
                    OnlineAttemptState.NOT_ATTEMPTED
                )
                return None
            cache_keys = [
                self._antoine_cache_key(candidate, T)
                for candidate in identifiers
            ]
            missing_cache_keys = [
                self._antoine_missing_cache_key(candidate, T)
                for candidate in identifiers
            ] if require_in_range and T is not None else cache_keys
            missing_cache_hits = 0
            for cache_key in missing_cache_keys:
                cached = self._get_cache(cache_key)
                if self._is_missing_cache(cached):
                    missing_cache_hits += 1
            if missing_cache_keys and missing_cache_hits == len(missing_cache_keys):
                self._record_online_attempt_state(
                    OnlineAttemptState.COMPLETE_NO_DATA
                )
                return None

            for cache_key in cache_keys:
                cached = self._get_cache(cache_key)
                if not cached:
                    continue
                if self._is_missing_cache(cached):
                    continue
                antoine = AntoineCoefficients(**cached)
                if not require_in_range or T is None or antoine.covers_temperature(T):
                    self._record_online_attempt_state(
                        OnlineAttemptState.COMPLETE_WITH_DATA
                    )
                    return antoine

            transient_failure = False

            # Try NIST first; PubChem does not expose Antoine coefficients through
            # its simple property endpoint.
            antoine = None
            for candidate in identifiers:
                try:
                    antoine = self._fetch_antoine_nist(candidate, T)
                except LookupError:
                    transient_failure = True
                    continue
                if antoine:
                    break

            if antoine:
                cache_payload = {
                    'A': antoine.A, 'B': antoine.B, 'C': antoine.C,
                    'T_min': antoine.T_min, 'T_max': antoine.T_max,
                    'source': antoine.source, 'P_units': antoine.P_units
                }
                for cache_key in cache_keys:
                    self._set_cache(cache_key, cache_payload)
                self._record_online_attempt_state(
                    OnlineAttemptState.COMPLETE_WITH_DATA
                )
                if not require_in_range or T is None or antoine.covers_temperature(T):
                    return antoine

            # Try PubChem as a last online source. This usually returns no Antoine
            # coefficients, but keeping the hook is useful if the parser improves.
            antoine = None
            for candidate in identifiers:
                try:
                    antoine = self._fetch_antoine_pubchem(candidate)
                except LookupError:
                    transient_failure = True
                    continue
                if antoine:
                    break

            if antoine:
                # Cache the result
                cache_payload = {
                    'A': antoine.A, 'B': antoine.B, 'C': antoine.C,
                    'T_min': antoine.T_min, 'T_max': antoine.T_max,
                    'source': antoine.source, 'P_units': antoine.P_units
                }
                for cache_key in cache_keys:
                    self._set_cache(cache_key, cache_payload)
                self._record_online_attempt_state(
                    OnlineAttemptState.COMPLETE_WITH_DATA
                )
                if not require_in_range or T is None or antoine.covers_temperature(T):
                    return antoine

            if not transient_failure:
                for cache_key in missing_cache_keys:
                    self._set_missing_cache(cache_key)
                self._record_online_attempt_state(
                    OnlineAttemptState.COMPLETE_NO_DATA
                )
            else:
                self._record_online_attempt_state(
                    OnlineAttemptState.TRANSIENT_FAILURE
                )
            return None


        def _prime_nist_antoine_cache(
            self,
            symbol: str,
            props: Mapping[str, Any],
        ) -> None:
            """Fetch and persist every NIST Antoine row once at initialization."""
            identifiers = self._identifier_candidates(symbol, props)
            if not identifiers:
                self._record_online_attempt_state(
                    OnlineAttemptState.NOT_ATTEMPTED
                )
                return
            completed_without_data = False
            for identifier in identifiers:
                bundle_key = f"nist_antoine_rows_v1_{identifier}"
                cached = self._get_cache(bundle_key)
                if self._is_missing_cache(cached):
                    completed_without_data = True
                    continue
                if cached and isinstance(cached.get("rows"), list):
                    self._cache_nist_antoine_rows(
                        identifier,
                        cached["rows"],
                    )
                    self._record_online_attempt_state(
                        OnlineAttemptState.COMPLETE_WITH_DATA
                    )
                    return
                try:
                    rows = self._fetch_antoine_nist_rows(identifier)
                except LookupError:
                    self._record_online_attempt_state(
                        OnlineAttemptState.TRANSIENT_FAILURE
                    )
                    continue
                completed_without_data = True
                if not rows:
                    self._set_missing_cache(bundle_key)
                    continue
                payloads = [
                    {
                        "A": row.A,
                        "B": row.B,
                        "C": row.C,
                        "T_min": row.T_min,
                        "T_max": row.T_max,
                        "source": row.source,
                        "P_units": row.P_units,
                    }
                    for row in rows
                ]
                self._set_cache(bundle_key, {"rows": payloads})
                self._cache_nist_antoine_rows(identifier, payloads)
                self._record_online_attempt_state(
                    OnlineAttemptState.COMPLETE_WITH_DATA
                )
                return
            if completed_without_data:
                self._record_online_attempt_state(
                    OnlineAttemptState.COMPLETE_NO_DATA
                )


        def _cache_nist_antoine_rows(
            self,
            identifier: str,
            rows: list[Mapping[str, Any]],
        ) -> None:
            for index, row in enumerate(rows):
                payload = dict(row)
                payload["source"] = "NIST WebBook"
                key = f"antoine_{identifier}_canonical_{index}"
                self._set_cache(key, payload)


        def _fetch_antoine_nist_rows(
            self,
            identifier: str,
        ) -> list[AntoineCoefficients]:
            transient_failure = False
            for query_field in ("Name", "Formula"):
                try:
                    params = urllib.parse.urlencode({
                        query_field: identifier,
                        "Units": "SI",
                        "Mask": "4",
                        "Type": "ANTOINE",
                    })
                    url = f"{self.NIST_WEBBOOK}?{params}"
                    request = urllib.request.Request(url)
                    request.add_header("User-Agent", "PFD-Editor/1.0")
                    with urllib.request.urlopen(request, timeout=15) as response:
                        html = response.read().decode("utf-8", errors="ignore")
                    rows = self._parse_nist_antoine_rows(html)
                    if rows:
                        return rows
                except Exception as error:
                    transient_failure = (
                        transient_failure
                        or self._is_transient_lookup_error(error)
                    )
            if transient_failure:
                raise LookupError(
                    f"Transient NIST Antoine lookup failure for {identifier!r}"
                )
            return []


        def _fetch_antoine_nist(
            self,
            identifier: str,
            T: Optional[float] = None,
        ) -> Optional[AntoineCoefficients]:
            """Fetch Antoine coefficients from NIST Chemistry WebBook."""
            transient_failure = False
            for query_field in ('Name', 'Formula'):
                try:
                    params = urllib.parse.urlencode({
                        query_field: identifier,
                        'Units': 'SI',
                        'Mask': '4',
                        'Type': 'ANTOINE',
                    })
                    url = f"{self.NIST_WEBBOOK}?{params}"

                    req = urllib.request.Request(url)
                    req.add_header('User-Agent', 'PFD-Editor/1.0')

                    with urllib.request.urlopen(req, timeout=15) as response:
                        html = response.read().decode('utf-8', errors='ignore')

                    antoine = self._parse_nist_antoine(html, T)
                    if antoine:
                        return antoine
                except Exception as e:
                    transient_failure = transient_failure or self._is_transient_lookup_error(e)

            if transient_failure:
                raise LookupError(f"Transient NIST Antoine lookup failure for '{identifier}'")
            return None


        def _parse_nist_antoine(
            self,
            html: str,
            T: Optional[float] = None,
        ) -> Optional[AntoineCoefficients]:
            """Parse NIST Antoine rows and convert them to local conventions."""
            rows = self._parse_nist_antoine_rows(html)
            if not rows:
                return None

            if T is not None:
                in_range = [
                    row for row in rows
                    if row.T_min <= T <= row.T_max
                ]
                if in_range:
                    return min(
                        in_range,
                        key=lambda row: (
                            row.T_max - row.T_min,
                            abs((row.T_min + row.T_max) / 2.0 - T),
                        ),
                    )
                return min(
                    rows,
                    key=lambda row: min(
                        abs(row.T_min - T),
                        abs(row.T_max - T),
                    ),
                )

            # Prefer ranges that cover ordinary processing temperatures;
            # otherwise choose the highest-temperature range.
            return max(
                rows,
                key=lambda row: (
                    row.T_min <= 298.15 <= row.T_max,
                    row.T_max,
                    row.T_max - row.T_min,
                ),
            )


        @staticmethod
        def _parse_nist_antoine_rows(
            html: str,
        ) -> list[AntoineCoefficients]:
            if "Antoine Equation Parameters" not in html:
                return []
            text = html_module.unescape(html).replace("\u2212", "-")
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            pattern = re.compile(
                r"(?P<Tmin>\d+(?:\.\d+)?)\s+to\s+"
                r"(?P<Tmax>\d+(?:\.\d+)?)\s+"
                r"(?P<A>[-+]?\d+(?:\.\d+)?)\s+"
                r"(?P<B>[-+]?\d+(?:\.\d+)?)\s*"
                r"(?P<C>[-+]?\d+(?:\.\d+)?)"
            )
            rows = []
            seen = set()
            for match in pattern.finditer(text):
                try:
                    values = tuple(
                        float(match.group(name))
                        for name in ("Tmin", "Tmax", "A", "B", "C")
                    )
                except ValueError:
                    continue
                T_min, T_max, A, B, C_kelvin = values
                if T_min <= 0.0 or T_max <= T_min or B <= 0.0:
                    continue
                key = (T_min, T_max, A, B, C_kelvin)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(AntoineCoefficients(
                    A=A,
                    B=B,
                    C=C_kelvin + 273.15,
                    T_min=T_min,
                    T_max=T_max,
                    source="NIST WebBook",
                    P_units="bar",
                ))
            return rows


        def _fetch_antoine_pubchem(self, symbol: str) -> Optional[AntoineCoefficients]:
            """Fetch Antoine coefficients from PubChem"""
            try:
                # First, get CID from symbol/name
                cid = self._get_pubchem_cid(symbol)
                if not cid:
                    return None

                # Get experimental properties
                url = f"{self.PUBCHEM_API}/compound/cid/{cid}/property/MolecularFormula/JSON"

                # PubChem doesn't directly provide Antoine coefficients,
                # but we can get vapor pressure data points and fit

                # Try to get vapor pressure annotations
                url = f"{self.PUBCHEM_API}/compound/cid/{cid}/JSON"

                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'PFD-Editor/1.0')

                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode('utf-8'))

                # Look for vapor pressure in experimental data
                # This is complex because PubChem structure varies
                # For now, return None and rely on other methods

            except Exception as e:
                if self._is_transient_lookup_error(e) or isinstance(e, LookupError):
                    raise LookupError(f"Transient PubChem Antoine lookup failure for '{symbol}'") from e

            return None


        def _fetch_formation_online(
            self,
            symbol: str,
            props: Optional[Dict[str, Any]] = None,
        ) -> Optional[Dict[str, Any]]:
            """Fetch ideal-gas formation properties from online sources."""
            for candidate in self._identifier_candidates(symbol, props):
                try:
                    lookup = self._fetch_formation_nist(candidate)
                except LookupError:
                    continue
                if lookup:
                    return lookup
            return None


        def _fetch_formation_nist(self, identifier: str) -> Optional[Dict[str, Any]]:
            """Fetch NIST gas and liquid thermochemistry rows."""
            cache_key = f"formation_nist_{identifier}"
            cached = self._get_cache(cache_key)
            if cached:
                if self._is_missing_cache(cached):
                    return None
                return cached

            transient_failure = False
            for query_field in ('Name', 'Formula'):
                try:
                    params = urllib.parse.urlencode({
                        query_field: identifier,
                        'Units': 'SI',
                        'Mask': '3',
                    })
                    req = urllib.request.Request(f"{self.NIST_WEBBOOK}?{params}")
                    req.add_header('User-Agent', 'PFD-Editor/1.0')
                    with urllib.request.urlopen(req, timeout=15) as response:
                        html = response.read().decode('utf-8', errors='ignore')

                    result = self._parse_nist_formation_properties(html)
                    if result:
                        self._set_cache(cache_key, result)
                        return result
                except Exception as e:
                    transient_failure = transient_failure or self._is_transient_lookup_error(e)

            if transient_failure:
                raise LookupError(f"Transient NIST formation-property lookup failure for '{identifier}'")
            self._set_missing_cache(cache_key)
            return None


        def _parse_nist_formation_properties(self, html: str) -> Dict[str, Any]:
            """Parse NIST gas-phase thermochemistry values at 298.15 K."""
            candidates: Dict[str, list[Dict[str, Any]]] = {
                'Hf': [],
                'Hf_liquid': [],
                'Gf': [],
                'Gf_liquid': [],
                'S': [],
                'S_liquid': [],
                'Hcomb': [],
            }

            for label, rows in self._html_tables(html):
                if label.strip().lower() != 'one dimensional data':
                    continue
                for row in rows:
                    if len(row) < 3 or row[0].strip().lower() == 'quantity':
                        continue
                    prop = self._nist_formation_property_key(row[0])
                    if prop is None:
                        continue
                    value = self._first_number(row[1])
                    if value is None or not self._nist_formation_units_match(prop, row[2]):
                        continue
                    if prop == 'S' and not (0.0 < value < 2000.0):
                        continue
                    if prop != 'S' and not (-50000.0 < value < 50000.0):
                        continue
                    candidates[prop].append({
                        'value': value,
                        'quantity': row[0],
                        'method': row[3] if len(row) > 3 else '',
                        'reference': row[4] if len(row) > 4 else '',
                        'comment': row[5] if len(row) > 5 else '',
                    })

            result: Dict[str, Any] = {}
            sources: Dict[str, str] = {}
            notes: Dict[str, str] = {}
            for prop, rows in candidates.items():
                selected = self._select_nist_formation_row(prop, rows)
                if selected is None:
                    continue
                value, method, note = selected
                result[prop] = value
                sources[prop] = method
                notes[prop] = note

            if result:
                result['_sources'] = sources
                result['_notes'] = notes
            return result


        @staticmethod
        def _nist_formation_property_key(quantity: str) -> Optional[str]:
            normalized = quantity.replace('Δ', 'delta').replace('δ', 'delta')
            normalized = normalized.replace('°', '').replace('º', '')
            normalized = re.sub(r'[^a-z0-9]+', ' ', normalized.lower()).strip()
            if 'delta f h gas' in normalized:
                return 'Hf'
            if 'delta f h liquid' in normalized:
                return 'Hf_liquid'
            if 'delta f g gas' in normalized:
                return 'Gf'
            if 'delta f g liquid' in normalized:
                return 'Gf_liquid'
            if 'delta c h gas' in normalized:
                return 'Hcomb'
            if normalized.startswith('s gas') or normalized == 's gas':
                return 'S'
            if normalized.startswith('s liquid') or normalized == 's liquid':
                return 'S_liquid'
            return None


        @staticmethod
        def _nist_formation_units_match(prop: str, units: str) -> bool:
            normalized = units.lower().replace(' ', '')
            if prop in {'S', 'S_liquid'}:
                return 'j/mol' in normalized and 'k' in normalized
            return 'kj/mol' in normalized


        def _select_nist_formation_row(
            self,
            prop: str,
            rows: list[Dict[str, Any]],
        ) -> Optional[tuple[float, str, str]]:
            if not rows:
                return None

            def row_rank(row: Dict[str, Any]) -> tuple[int, int, int]:
                method = str(row.get('method') or '').strip().lower()
                quantity = str(row.get('quantity') or '').lower()
                preferred_method = 1 if method in {'avg', 'review'} else 0
                one_bar_entropy = 1 if prop == 'S' and '1 bar' in quantity else 0
                has_reference = 1 if str(row.get('reference') or '').strip().upper() != 'N/A' else 0
                return (one_bar_entropy, preferred_method, has_reference)

            best_rank = max(row_rank(row) for row in rows)
            preferred = [row for row in rows if row_rank(row) == best_rank]
            if best_rank[1] or (prop == 'S' and best_rank[0]):
                row = preferred[0]
                method = str(row.get('method') or 'direct').strip() or 'direct'
                reference = str(row.get('reference') or '').strip()
                phase = 'liquid' if prop.endswith('_liquid') else 'gas'
                note = f"NIST WebBook {phase} thermochemistry {method} row"
                if reference and reference.upper() != 'N/A':
                    note += f"; {reference}"
                return float(row['value']), f'nist_{phase}_thermochemistry', note

            values = [float(row['value']) for row in rows]
            value = self._median(values)
            phase = 'liquid' if prop.endswith('_liquid') else 'gas'
            return (
                value,
                f'nist_{phase}_thermochemistry_median',
                f"NIST WebBook {phase} thermochemistry median of {len(values)} direct row(s)",
            )


        def _get_pubchem_cid(self, identifier: str) -> Optional[int]:
            """Get PubChem CID from identifier"""
            cache_key = f"pubchem_cid_{identifier}"
            cached = self._get_cache(cache_key)
            if cached:
                if self._is_missing_cache(cached):
                    return None
                return cached.get('cid')

            try:
                # Try by name
                url = f"{self.PUBCHEM_API}/compound/name/{urllib.parse.quote(identifier)}/cids/JSON"

                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'PFD-Editor/1.0')

                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                        cid = data['IdentifierList']['CID'][0]
                        self._set_cache(cache_key, {'cid': cid, 'source': 'pubchem'})
                        return cid
            except Exception as e:
                if self._is_transient_lookup_error(e):
                    raise LookupError(f"Transient PubChem CID lookup failure for '{identifier}'") from e

            self._set_missing_cache(cache_key)
            return None
