from .common import *
from contextlib import closing
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import sqlite3
from typing import Callable, Mapping

from .cache_paths import SATURATION_PROPERTIES_CACHE_PATH
from .coolprop import (
    COOLPROP_PROPERTY_QUALITY,
    coolprop_melting_line_domain,
    coolprop_melting_pressure,
    coolprop_melting_temperature,
    coolprop_props_si,
    coolprop_reference_for,
    coolprop_saturation_temperature,
)
from .base import PropertyResolverBase
from .cache_expiration import runtime_cache_row_is_fresh
from .organic_classification import hydrogen_bond_donor_profile

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..compound_identity import parse_formula_counts
else:
    from compound_identity import parse_formula_counts


class PhaseChangeMixin:
        PHASE_POINT_CACHE_VERSION = 16
        TRIPLE_PRESSURE_BACKFILL_CACHE_VERSION = 1
        FUSION_TRANSITION_CACHE_VERSION = 6
        COOLPROP_UNCORROBORATED_MELTING_QUALITY = 0.98
        COOLPROP_PROVISIONAL_TRIPLE_QUALITY = 0.89
        COOLPROP_TRIPLE_CLOSURE_REL_TOL = 0.10
        PHASE_TEMPERATURE_ABS_TOL_K = 0.5
        PHASE_TEMPERATURE_REL_TOL = 0.03
        FUSION_TRANSITION_CLUSTER_TOLERANCE_K = 2.0
        FUSION_TRANSITION_CLUSTER_REL_TOL = 0.01
        SATURATION_PROPERTIES_CACHE_PATH = SATURATION_PROPERTIES_CACHE_PATH

        @staticmethod
        def _copy_phase_point_results(
            results: Mapping[str, PropertyResolutionResult],
        ) -> Dict[str, PropertyResolutionResult]:
            return {
                str(name): PropertyResolutionResult(
                    value=result.value,
                    source=str(result.source),
                    method=str(result.method),
                    quality=float(result.quality),
                    notes=str(result.notes or ''),
                )
                for name, result in results.items()
            }


        @staticmethod
        def _phase_point_component_identity(
            symbol: str,
            props: Mapping[str, Any],
        ) -> tuple[str, str, str]:
            cas = str(props.get('CAS') or props.get('cas') or '').strip()
            name = str(props.get('name') or symbol).strip()
            if cas:
                return f"cas:{cas.lower()}", cas, name
            normalized = ''.join(
                character.lower()
                for character in str(symbol)
                if character.isalnum()
            )
            return f"symbol:{normalized or str(symbol).lower()}", '', name


        @staticmethod
        def _phase_point_cache_json_default(value: Any):
            if isinstance(value, Enum):
                return value.value
            if is_dataclass(value):
                return asdict(value)
            if isinstance(value, (set, frozenset)):
                return sorted(value, key=repr)
            item = getattr(value, 'item', None)
            if callable(item):
                try:
                    return item()
                except Exception:
                    pass
            return repr(value)


        @classmethod
        def _phase_point_cache_json(cls, value: Any) -> str:
            return json.dumps(
                value,
                sort_keys=True,
                separators=(',', ':'),
                default=cls._phase_point_cache_json_default,
            )


        @classmethod
        def _phase_point_input_metadata(
            cls,
            resolution_kind: str,
            symbol: str,
            props: Mapping[str, Any],
            *,
            allow_online: bool,
            allow_estimation: bool,
        ) -> Dict[str, Any]:
            relevant_fields = (
                'CAS', 'cas', 'name', 'symbol', 'formula', 'Formula',
                'smiles', 'SMILES', 'source', 'MW',
                'Tb', 'Tm', 'Tt', 'Pt', 'Tc', 'Pc', 'Vc', 'Zc', 'omega',
                'Hvap', 'Hfus', 'Hsub', 'critical_properties_unavailable',
                'fusion_transitions', 'melting_transitions',
                '_phase_independent_tm_validator',
            )
            relevant_props = {
                field_name: props.get(field_name)
                for field_name in relevant_fields
                if field_name in props
            }
            property_sources = props.get('property_sources')
            if isinstance(property_sources, Mapping):
                source_fields = {
                    'formula', 'smiles', 'MW',
                    'Tb', 'Tm', 'Tt', 'Pt', 'Tc', 'Pc', 'Vc', 'Zc',
                    'omega', 'Hvap', 'Hfus', 'Hsub',
                    'fusion_transitions', 'melting_transitions',
                }
                relevant_props['property_sources'] = {
                    str(field_name): metadata
                    for field_name, metadata in property_sources.items()
                    if str(field_name) in source_fields
                }
            if resolution_kind == 'triple_point':
                provider_order = [
                    'PFD component override',
                    'CoolProp',
                    'provided or hydrated value',
                    'NIST/PubChem online consensus',
                    'canonical Psat evaluated at selected Tt',
                ]
            elif resolution_kind == 'melting_point':
                provider_order = [
                    'PFD component override',
                    'CoolProp HEOS fusion line at 1 atm',
                    'provided or curated value',
                    'Perry 9th',
                    'Perry Table 2-10',
                    'hydrated value',
                    'NIST/PubChem online consensus',
                ]
            else:
                provider_order = [
                    'PFD component override',
                    'CoolProp',
                    'provided value',
                    'Smith8 textbook',
                    'Perry 9th',
                    'Perry Table 2-10',
                    'hydrated value',
                    'online phase change',
                    'estimation fallback',
                ]
            return {
                'resolver_contract': f'resolved_{resolution_kind}_v1',
                'resolution_kind': resolution_kind,
                'symbol_argument': str(symbol),
                'allow_online': bool(allow_online),
                'allow_estimation': bool(allow_estimation),
                'properties': relevant_props,
                'units': {
                    'Tb': 'K',
                    'Tm': 'K',
                    'Tt': 'K',
                    'Pt': 'bar',
                },
                'provider_order': provider_order,
                'future_triple_point_estimation_inputs': [
                    'Tm', 'Tb', 'Tc', 'Hfus',
                ],
            }


        @classmethod
        def _phase_point_input_fingerprint(
            cls,
            resolution_kind: str,
            symbol: str,
            props: Mapping[str, Any],
            *,
            allow_online: bool,
            allow_estimation: bool,
        ) -> str:
            payload = cls._phase_point_cache_json(
                cls._phase_point_input_metadata(
                    resolution_kind,
                    symbol,
                    props,
                    allow_online=allow_online,
                    allow_estimation=allow_estimation,
                )
            )
            return hashlib.sha256(payload.encode('utf-8')).hexdigest()


        def _ensure_phase_point_cache_schema(self, connection) -> None:
            property_columns = []
            for property_name in ('Tb', 'Tm', 'Tt', 'Pt'):
                property_columns.extend([
                    f'{property_name}_value REAL',
                    f'{property_name}_source TEXT NOT NULL',
                    f'{property_name}_method TEXT NOT NULL',
                    f'{property_name}_quality REAL NOT NULL',
                    f'{property_name}_notes TEXT NOT NULL',
                ])
            connection.execute('PRAGMA busy_timeout = 30000')
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS resolved_phase_point_cache (
                    cache_version INTEGER NOT NULL,
                    resolution_kind TEXT NOT NULL,
                    component_key TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    cas TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    allow_online INTEGER NOT NULL,
                    allow_estimation INTEGER NOT NULL,
                    online_attempt_state TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    {', '.join(property_columns)},
                    results_json TEXT NOT NULL,
                    input_metadata_json TEXT NOT NULL,
                    PRIMARY KEY (
                        cache_version,
                        resolution_kind,
                        component_key,
                        input_fingerprint
                    )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS canonical_triple_pressure_backfill (
                    cache_version INTEGER NOT NULL,
                    canonical_psat_cache_version INTEGER NOT NULL,
                    component_key TEXT NOT NULL,
                    triple_temperature_fingerprint TEXT NOT NULL,
                    allow_online INTEGER NOT NULL,
                    canonical_input_fingerprint TEXT NOT NULL,
                    Tt_value REAL NOT NULL,
                    Tt_source TEXT NOT NULL,
                    Tt_method TEXT NOT NULL,
                    Tt_quality REAL NOT NULL,
                    Pt_value REAL NOT NULL,
                    Pt_source TEXT NOT NULL,
                    Pt_method TEXT NOT NULL,
                    Pt_quality REAL NOT NULL,
                    Pt_notes TEXT NOT NULL,
                    psat_source TEXT NOT NULL,
                    psat_method TEXT NOT NULL,
                    psat_quality REAL NOT NULL,
                    dependency_metadata_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    PRIMARY KEY (
                        cache_version,
                        canonical_psat_cache_version,
                        component_key,
                        triple_temperature_fingerprint,
                        allow_online,
                        canonical_input_fingerprint
                    )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS resolved_fusion_transition_cache (
                    cache_version INTEGER NOT NULL,
                    online_phase_contract_version INTEGER NOT NULL,
                    component_key TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    cas TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    allow_online INTEGER NOT NULL,
                    online_attempt_state TEXT NOT NULL,
                    selected_Hfus_value REAL,
                    selected_Hfus_source TEXT NOT NULL,
                    selected_Hfus_method TEXT NOT NULL,
                    selected_Hfus_quality REAL NOT NULL,
                    selected_Hfus_notes TEXT NOT NULL,
                    records_json TEXT NOT NULL,
                    selected_records_json TEXT NOT NULL,
                    input_metadata_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    PRIMARY KEY (
                        cache_version,
                        online_phase_contract_version,
                        component_key,
                        input_fingerprint
                    )
                )
                """
            )
            existing_columns = {
                str(row[1])
                for row in connection.execute(
                    'PRAGMA table_info(resolved_phase_point_cache)'
                )
            }
            migration_columns = {
                'online_attempt_state': (
                    "TEXT NOT NULL DEFAULT 'not_attempted'"
                ),
                'Tm_value': 'REAL',
                'Tm_source': "TEXT NOT NULL DEFAULT ''",
                'Tm_method': "TEXT NOT NULL DEFAULT ''",
                'Tm_quality': 'REAL NOT NULL DEFAULT 0.0',
                'Tm_notes': "TEXT NOT NULL DEFAULT ''",
            }
            for column_name, column_contract in migration_columns.items():
                if column_name in existing_columns:
                    continue
                try:
                    connection.execute(
                        f'ALTER TABLE resolved_phase_point_cache '
                        f'ADD COLUMN {column_name} {column_contract}'
                    )
                except sqlite3.OperationalError as error:
                    if 'duplicate column name' not in str(error).lower():
                        raise
            for suffix, column in (
                ('cas', 'cas'),
                ('name', 'component_name'),
                ('symbol', 'symbol'),
                ('updated', 'updated_at_utc'),
            ):
                connection.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS resolved_phase_point_cache_{suffix}_idx
                    ON resolved_phase_point_cache
                        (cache_version, resolution_kind, {column})
                    """
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    canonical_triple_pressure_backfill_lookup_idx
                ON canonical_triple_pressure_backfill (
                    cache_version,
                    canonical_psat_cache_version,
                    component_key,
                    triple_temperature_fingerprint,
                    allow_online,
                    created_at_utc
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS resolved_fusion_transition_cache_cas_idx
                ON resolved_fusion_transition_cache (
                    cache_version,
                    online_phase_contract_version,
                    cas,
                    updated_at_utc
                )
                """
            )


        @classmethod
        def _triple_temperature_fingerprint(
            cls,
            result: PropertyResolutionResult,
        ) -> str:
            payload = cls._phase_point_cache_json({
                'value': result.value,
                'source': result.source,
                'method': result.method,
                'quality': result.quality,
                'notes': result.notes,
            })
            return hashlib.sha256(payload.encode('utf-8')).hexdigest()


        def _load_canonical_triple_pressure_backfill(
            self,
            *,
            component_key: str,
            triple_temperature: PropertyResolutionResult,
            allow_online: bool,
        ) -> Optional[PropertyResolutionResult]:
            if triple_temperature.value is None:
                return None
            path = Path(self.SATURATION_PROPERTIES_CACHE_PATH)
            if not path.exists():
                return None
            canonical_version = int(
                getattr(self, 'CANONICAL_PSAT_CACHE_VERSION', 0)
            )
            fingerprint = self._triple_temperature_fingerprint(
                triple_temperature
            )
            try:
                with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                    connection.row_factory = sqlite3.Row
                    self._ensure_phase_point_cache_schema(connection)
                    row = connection.execute(
                        """
                        SELECT Pt_value, Pt_source, Pt_method,
                               Pt_quality, Pt_notes, created_at_utc
                        FROM canonical_triple_pressure_backfill
                        WHERE cache_version = ?
                          AND canonical_psat_cache_version = ?
                          AND component_key = ?
                          AND triple_temperature_fingerprint = ?
                          AND allow_online = ?
                        ORDER BY created_at_utc DESC
                        LIMIT 1
                        """,
                        (
                            int(self.TRIPLE_PRESSURE_BACKFILL_CACHE_VERSION),
                            canonical_version,
                            component_key,
                            fingerprint,
                            int(bool(allow_online)),
                        ),
                    ).fetchone()
                if row is None:
                    return None
                if not runtime_cache_row_is_fresh(
                    path,
                    row['created_at_utc'],
                ):
                    return None
                value = float(row['Pt_value'])
                quality = float(row['Pt_quality'])
                if (
                    not math.isfinite(value)
                    or value <= 0.0
                    or not math.isfinite(quality)
                    or not 0.0 <= quality <= 1.0
                ):
                    return None
                return PropertyResolutionResult(
                    value=value,
                    source=str(row['Pt_source']),
                    method=str(row['Pt_method']),
                    quality=quality,
                    notes=str(row['Pt_notes']),
                )
            except (OSError, sqlite3.Error, TypeError, ValueError):
                return None


        def _store_canonical_triple_pressure_backfill(
            self,
            symbol: str,
            props: Mapping[str, Any],
            *,
            triple_temperature: PropertyResolutionResult,
            psat_result: PropertyResolutionResult,
            allow_online: bool,
            canonical_input_fingerprint: str,
        ) -> Optional[PropertyResolutionResult]:
            try:
                Tt_value = float(triple_temperature.value)
                Pt_value = float(psat_result.value)
                quality = min(
                    float(triple_temperature.quality),
                    float(psat_result.quality),
                )
            except (TypeError, ValueError):
                return None
            if (
                not math.isfinite(Tt_value)
                or Tt_value <= 0.0
                or not math.isfinite(Pt_value)
                or Pt_value <= 0.0
                or not math.isfinite(quality)
                or not 0.0 <= quality <= 1.0
                or not canonical_input_fingerprint
                or self._is_pfd_component_override(props, 'Tt')
                or self._is_pfd_component_override(props, 'Pt')
                or self._is_pfd_correlation_override(props, 'Psat')
            ):
                return None
            component_key, _cas, _name = self._phase_point_component_identity(
                symbol,
                props,
            )
            canonical_version = int(
                getattr(self, 'CANONICAL_PSAT_CACHE_VERSION', 0)
            )
            notes = (
                f'Pt = canonical Psat(Tt={Tt_value:g} K) = '
                f'{Pt_value:g} bar; Tt from '
                f'{triple_temperature.source}/{triple_temperature.method}; '
                f'local Psat quality={float(psat_result.quality):g} from '
                f'{psat_result.source}/{psat_result.method}; '
                f'canonical Psat cache version={canonical_version}; '
                f'canonical input fingerprint={canonical_input_fingerprint}'
            )
            result = PropertyResolutionResult(
                value=Pt_value,
                source='calculated',
                method='canonical_psat_at_triple_temperature',
                quality=quality,
                notes=notes,
            )
            dependency_metadata = {
                'equation': 'Pt = Psat(Tt)',
                'canonical_psat_cache_version': canonical_version,
                'canonical_input_fingerprint': canonical_input_fingerprint,
                'Tt': asdict(triple_temperature),
                'Psat_at_Tt': asdict(psat_result),
                'units': {'Tt': 'K', 'Pt': 'bar'},
            }
            timestamp = datetime.now(timezone.utc).isoformat(
                timespec='milliseconds'
            ).replace('+00:00', 'Z')
            path = Path(self.SATURATION_PROPERTIES_CACHE_PATH)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                    self._ensure_phase_point_cache_schema(connection)
                    connection.execute(
                        """
                        INSERT INTO canonical_triple_pressure_backfill (
                            cache_version, canonical_psat_cache_version,
                            component_key, triple_temperature_fingerprint,
                            allow_online, canonical_input_fingerprint,
                            Tt_value, Tt_source, Tt_method, Tt_quality,
                            Pt_value, Pt_source, Pt_method, Pt_quality,
                            Pt_notes, psat_source, psat_method, psat_quality,
                            dependency_metadata_json, created_at_utc
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        ON CONFLICT (
                            cache_version, canonical_psat_cache_version,
                            component_key, triple_temperature_fingerprint,
                            allow_online, canonical_input_fingerprint
                        ) DO NOTHING
                        """,
                        (
                            int(self.TRIPLE_PRESSURE_BACKFILL_CACHE_VERSION),
                            canonical_version,
                            component_key,
                            self._triple_temperature_fingerprint(
                                triple_temperature
                            ),
                            int(bool(allow_online)),
                            canonical_input_fingerprint,
                            Tt_value,
                            str(triple_temperature.source),
                            str(triple_temperature.method),
                            float(triple_temperature.quality),
                            Pt_value,
                            result.source,
                            result.method,
                            result.quality,
                            result.notes,
                            str(psat_result.source),
                            str(psat_result.method),
                            float(psat_result.quality),
                            self._phase_point_cache_json(dependency_metadata),
                            timestamp,
                        ),
                    )
                    connection.commit()
            except (OSError, sqlite3.Error, TypeError, ValueError):
                return None

            memory_cache = getattr(self, '_resolved_phase_point_cache', None)
            if isinstance(memory_cache, dict):
                stale_keys = [
                    key
                    for key, cached in memory_cache.items()
                    if (
                        len(key) >= 3
                        and key[1] == 'triple_point'
                        and key[2] == component_key
                        and cached.get('Pt') is not None
                        and cached['Pt'].value is None
                    )
                ]
                for key in stale_keys:
                    memory_cache.pop(key, None)
            return result


        def initialize_phase_point_disk_cache(self) -> Path:
            """Create the persistent phase-point cache and return its path."""
            path = Path(self.SATURATION_PROPERTIES_CACHE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                connection.execute('PRAGMA journal_mode = WAL')
                self._ensure_phase_point_cache_schema(connection)
                connection.commit()
            return path


        def _load_persistent_phase_point_results(
            self,
            resolution_kind: str,
            *,
            component_key: str,
            fingerprint: str,
        ) -> Optional[Dict[str, PropertyResolutionResult]]:
            path = Path(self.SATURATION_PROPERTIES_CACHE_PATH)
            if not path.exists():
                return None
            try:
                with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                    connection.row_factory = sqlite3.Row
                    self._ensure_phase_point_cache_schema(connection)
                    row = connection.execute(
                        """
                        SELECT * FROM resolved_phase_point_cache
                        WHERE cache_version = ?
                          AND resolution_kind = ?
                          AND component_key = ?
                          AND input_fingerprint = ?
                        """,
                        (
                            int(self.PHASE_POINT_CACHE_VERSION),
                            resolution_kind,
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
                results_payload = json.loads(row['results_json'])
                input_metadata = json.loads(row['input_metadata_json'])
                if not isinstance(results_payload, dict) or not isinstance(input_metadata, dict):
                    return None
                property_names = {
                    'boiling_point': ('Tb',),
                    'melting_point': ('Tm',),
                    'triple_point': ('Tt', 'Pt'),
                }.get(resolution_kind)
                if property_names is None:
                    return None
                results = {
                    name: PropertyResolutionResult(
                        value=(
                            None
                            if row[f'{name}_value'] is None
                            else float(row[f'{name}_value'])
                        ),
                        source=str(row[f'{name}_source']),
                        method=str(row[f'{name}_method']),
                        quality=float(row[f'{name}_quality']),
                        notes=str(row[f'{name}_notes']),
                    )
                    for name in property_names
                }
                if (
                    resolution_kind == 'triple_point'
                    and results['Tt'].value is not None
                    and results['Pt'].value is None
                ):
                    derived = self._load_canonical_triple_pressure_backfill(
                        component_key=component_key,
                        triple_temperature=results['Tt'],
                        allow_online=bool(row['allow_online']),
                    )
                    if derived is not None:
                        results['Pt'] = derived
                return results
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                OSError,
                sqlite3.Error,
            ):
                return None


        def _store_persistent_phase_point_results(
            self,
            resolution_kind: str,
            symbol: str,
            props: Mapping[str, Any],
            results: Mapping[str, PropertyResolutionResult],
            *,
            allow_online: bool,
            allow_estimation: bool,
            online_attempt_state: OnlineAttemptState | str,
            component_key: str,
            fingerprint: str,
        ) -> None:
            component_key_check, cas, component_name = (
                self._phase_point_component_identity(symbol, props)
            )
            if component_key_check != component_key:
                return
            property_names = ('Tb', 'Tm', 'Tt', 'Pt')
            normalized = {
                name: results.get(name) or PropertyResolutionResult(
                    value=None,
                    source='missing',
                    method='none',
                    quality=0.0,
                    notes=f'{name} not resolved by {resolution_kind}',
                )
                for name in property_names
            }
            selected_names = {
                'boiling_point': ('Tb',),
                'melting_point': ('Tm',),
                'triple_point': ('Tt', 'Pt'),
            }.get(resolution_kind)
            if selected_names is None:
                return
            selected_results = {
                name: normalized[name]
                for name in selected_names
            }
            results_json = self._phase_point_cache_json({
                name: asdict(result)
                for name, result in selected_results.items()
            })
            input_metadata_json = self._phase_point_cache_json(
                self._phase_point_input_metadata(
                    resolution_kind,
                    symbol,
                    props,
                    allow_online=allow_online,
                    allow_estimation=allow_estimation,
                )
            )
            timestamp = datetime.now(timezone.utc).isoformat(
                timespec='milliseconds'
            ).replace('+00:00', 'Z')
            columns = [
                'cache_version', 'resolution_kind', 'component_key',
                'input_fingerprint', 'symbol', 'cas', 'component_name',
                'allow_online', 'allow_estimation',
                'online_attempt_state',
                'created_at_utc', 'updated_at_utc',
            ]
            values = [
                int(self.PHASE_POINT_CACHE_VERSION),
                resolution_kind,
                component_key,
                fingerprint,
                str(symbol),
                cas,
                component_name,
                int(bool(allow_online)),
                int(bool(allow_estimation)),
                OnlineAttemptState(online_attempt_state).value,
                timestamp,
                timestamp,
            ]
            update_columns = [
                'symbol', 'cas', 'component_name', 'allow_online',
                'allow_estimation', 'online_attempt_state', 'updated_at_utc',
            ]
            for name in property_names:
                result = normalized[name]
                columns.extend([
                    f'{name}_value', f'{name}_source', f'{name}_method',
                    f'{name}_quality', f'{name}_notes',
                ])
                values.extend([
                    result.value,
                    str(result.source),
                    str(result.method),
                    float(result.quality),
                    str(result.notes or ''),
                ])
                update_columns.extend([
                    f'{name}_value', f'{name}_source', f'{name}_method',
                    f'{name}_quality', f'{name}_notes',
                ])
            columns.extend(['results_json', 'input_metadata_json'])
            values.extend([results_json, input_metadata_json])
            update_columns.extend(['results_json', 'input_metadata_json'])
            placeholders = ', '.join('?' for _ in columns)
            updates = ', '.join(
                f'{column} = excluded.{column}'
                for column in update_columns
            )
            path = Path(self.SATURATION_PROPERTIES_CACHE_PATH)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                    self._ensure_phase_point_cache_schema(connection)
                    connection.execute(
                        f"""
                        INSERT INTO resolved_phase_point_cache
                            ({', '.join(columns)})
                        VALUES ({placeholders})
                        ON CONFLICT (
                            cache_version, resolution_kind,
                            component_key, input_fingerprint
                        ) DO UPDATE SET {updates}
                        """,
                        values,
                    )
                    connection.commit()
            except (OSError, sqlite3.Error, TypeError, ValueError):
                return


        def _cached_phase_point_results(
            self,
            resolution_kind: str,
            symbol: str,
            props: Mapping[str, Any],
            *,
            allow_online: bool,
            allow_estimation: bool,
            build: Callable[[], Mapping[str, PropertyResolutionResult]],
        ) -> Dict[str, PropertyResolutionResult]:
            component_key, _cas, _name = self._phase_point_component_identity(
                symbol,
                props,
            )
            fingerprint = self._phase_point_input_fingerprint(
                resolution_kind,
                symbol,
                props,
                allow_online=allow_online,
                allow_estimation=allow_estimation,
            )
            cache_key = (
                int(self.PHASE_POINT_CACHE_VERSION),
                resolution_kind,
                component_key,
                fingerprint,
            )
            memory_cache = getattr(self, '_resolved_phase_point_cache', None)
            if memory_cache is None:
                memory_cache = {}
                self._resolved_phase_point_cache = memory_cache
            cached = memory_cache.get(cache_key)
            if cached is not None:
                return self._copy_phase_point_results(cached)
            persistent = self._load_persistent_phase_point_results(
                resolution_kind,
                component_key=component_key,
                fingerprint=fingerprint,
            )
            if persistent is not None:
                memory_cache[cache_key] = persistent
                return self._copy_phase_point_results(persistent)
            selected_names = {
                'boiling_point': ('Tb',),
                'melting_point': ('Tm',),
                'triple_point': ('Tt', 'Pt'),
            }.get(resolution_kind, ())
            with self._online_attempt_scope(allow_online) as online_attempt:
                results = dict(build())
            if not allow_online:
                selected = [results.get(name) for name in selected_names]
                if (
                    selected
                    and all(
                        result is not None
                        and result.value is not None
                        and str(result.source).lower()
                        not in {'estimated', 'missing'}
                        for result in selected
                    )
                ) or (
                    resolution_kind == 'boiling_point'
                    and selected
                    and selected[0] is not None
                    and selected[0].method in {
                        'no_normal_boiling_point_at_1atm',
                        'invalid_normal_boiling_point_below_triple_point',
                    }
                ):
                    online_attempt.record(OnlineAttemptState.NOT_NEEDED)
            if self._online_attempt_is_persistable(
                allow_online,
                online_attempt.state,
            ):
                has_pfd_override = (
                    any(
                        self._is_pfd_component_override(props, name)
                        for name in selected_names
                    )
                    or self._is_pfd_correlation_override(props, 'Psat')
                )
                if not has_pfd_override:
                    self._store_persistent_phase_point_results(
                        resolution_kind,
                        symbol,
                        props,
                        results,
                        allow_online=allow_online,
                        allow_estimation=allow_estimation,
                        online_attempt_state=online_attempt.state,
                        component_key=component_key,
                        fingerprint=fingerprint,
                    )
                memory_cache[cache_key] = self._copy_phase_point_results(results)
            return results


        def get_hvap(self, symbol: str) -> Optional[float]:
            """Get heat of vaporization from local database [kJ/mol]"""
            result = self.resolve_hvap(symbol, allow_online=False, allow_estimation=False)
            return result.value


        def get_hvap_online(
            self,
            symbol: str,
            props: Optional[Dict[str, Any]] = None,
        ) -> Optional[float]:
            """
            Fetch online heat of vaporization at the normal boiling point.

            Returns:
                Hvap(Tb) in kJ/mol or None
            """
            identifiers = self._identifier_candidates(symbol, props)
            cache_key = f"hvap_tb_v2_{'|'.join(identifiers or [str(symbol)])}"
            cached = self._get_cache(cache_key)
            if cached and self._is_missing_cache(cached):
                return None
            if cached and 'Hvap' in cached:
                return cached['Hvap']

            try:
                online = self._fetch_phase_change_online(symbol, props or {})
            except LookupError:
                return None

            if online and online.get('Hvap') is not None:
                payload = {
                    'Hvap': online['Hvap'],
                    'Tb': online.get('Tb'),
                    'source': online.get('_sources', {}).get('Hvap', 'online_phase_change'),
                    'quality': online.get('_qualities', {}).get('Hvap'),
                    'notes': online.get('_notes', {}).get('Hvap', ''),
                }
                self._set_cache(cache_key, payload)
                return float(online['Hvap'])

            if (
                online
                and online.get('_online_attempt_state')
                == OnlineAttemptState.TRANSIENT_FAILURE.value
            ):
                return None
            self._set_missing_cache(cache_key)
            return None


        def _fetch_hvap_pubchem(self, symbol: str) -> Optional[float]:
            """Compatibility wrapper returning strict PubChem-derived Hvap(Tb)."""
            lookup = self._fetch_phase_change_pubchem(symbol)
            if not lookup:
                return None
            lookup = dict(lookup)
            self._finalize_online_hvap(lookup)
            value = lookup.get('Hvap')
            return float(value) if value is not None else None


        def _provided_scalar(
            self,
            props: Dict[str, Any],
            key: str,
            units: str = "",
        ) -> Optional[PropertyResolutionResult]:
            if props.get(key) is None:
                return None
            if self._provided_scalar_is_estimated(props, key):
                return None
            return self._source_result_for_value(props, key, units=units)


        def _provided_hvap_scalar(
            self,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            """Return source-backed Hvap(Tb) with the shared default quality."""
            if props.get('Hvap') is None or self._provided_scalar_is_estimated(
                props,
                'Hvap',
            ):
                return None
            return self._source_result_for_value(
                props,
                'Hvap',
                units='kJ/mol',
                default_quality=HVAP_PROVIDED_QUALITY,
            )


        @staticmethod
        def _provided_scalar_is_estimated(props: Dict[str, Any], key: str) -> bool:
            source = ((props or {}).get('property_sources') or {}).get(key) or {}
            return PropertyResolverBase._source_meta_is_soft(source)


        def _missing_scalar(self, key: str) -> PropertyResolutionResult:
            return PropertyResolutionResult(
                value=None,
                source='missing',
                method='none',
                quality=0.0,
                notes=f'{key} not available',
            )


        @staticmethod
        def _watson_hvap_value(
            hvap_ref: float,
            T_ref: float,
            T: float,
            Tc: float,
        ) -> Optional[float]:
            try:
                if hvap_ref <= 0.0 or T_ref >= Tc:
                    return None
                if T >= Tc:
                    return 0.0
                Tr = T / Tc
                Trr = T_ref / Tc
                if not (0.0 < Tr < 1.0 and 0.0 < Trr < 1.0):
                    return None
                value = hvap_ref * ((1.0 - Tr) / (1.0 - Trr)) ** 0.38
                if value < 0.0:
                    return None
                return value
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None


        def _textbook_hvap_result(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            for candidate in self._identifier_candidates(symbol, props):
                entry = self._get_textbook_entry(candidate)
                if entry and entry.get('Hvap') is not None:
                    return PropertyResolutionResult(
                        value=entry['Hvap'],
                        source='textbook',
                        method='Smith8 Appendix B',
                        quality=HVAP_SMITH_QUALITY,
                        notes='units kJ/mol',
                    )
            return None


        def _online_hvap_scalar_result(
            self,
            symbol: str,
            online: Optional[Dict[str, Any]],
        ) -> Optional[PropertyResolutionResult]:
            if online and online.get('Hvap') is not None:
                method = online.get('_sources', {}).get('Hvap', 'online_phase_change')
                return PropertyResolutionResult(
                    value=online['Hvap'],
                    source='online',
                    method=method,
                    quality=float(online.get('_qualities', {}).get('Hvap', 0.81)),
                    notes=online.get('_notes', {}).get(
                        'Hvap',
                        'Hvap at the normal boiling point; units kJ/mol',
                    ),
                )
            return None


        def _hvap_temperature_reference_at_tb(
            self,
            symbol: str,
            props: Dict[str, Any],
            Tb: float,
            online: Optional[Dict[str, Any]],
        ) -> Optional[PropertyResolutionResult]:
            provided_fit = self._evaluate_provided_correlation(props, 'Hvap', Tb)
            if provided_fit:
                value, correlation = provided_fit
                if value > 0:
                    return self._provided_correlation_result(
                        value,
                        correlation,
                        'provided_hvap_fit',
                        'heat of vaporization in kJ/mol',
                        default_quality=HVAP_PROVIDED_QUALITY,
                    )

            perry_hvap = self._get_perry_evaluation(
                symbol,
                props,
                'heat_of_vaporization_kJ_per_mol',
                Tb,
            )
            if perry_hvap:
                return PropertyResolutionResult(
                    value=perry_hvap.value,
                    source='local',
                    method=perry_hvap.method,
                    quality=HVAP_PERRY_QUALITY,
                    notes=f"{perry_hvap.source}; units {perry_hvap.units}",
                )

            # A genuine T-dependent online Hvap fit is preferred as the Watson
            # scaling basis, but a bare online scalar at Tb is NOT: a curated
            # local scalar (tried by the caller after this helper) must outrank
            # an online scalar. The online scalar therefore stays out of this
            # helper and is deferred to the caller's final fallback.
            fit = online.get('Hvap_fit') if online else None
            if isinstance(fit, HvapTemperatureFit):
                fit_value = fit.value_at(float(Tb))
                if fit_value is not None and fit_value > 0.0:
                    quality, range_note = self._hvap_fit_quality(fit, float(Tb))
                    return PropertyResolutionResult(
                        value=fit_value,
                        source='online',
                        method='nist_hvap_watson_fit',
                        quality=quality,
                        notes=(
                            f"fit range {fit.T_min:g}-{fit.T_max:g} K; "
                            f"MAPE {fit.mape_percent:.2f}%; "
                            f"kept {fit.kept_points}/{fit.total_points} points; "
                            f"{range_note}"
                        ),
                    )
            return None


        def _resolve_hvap_watson(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: Optional[float],
            online: Optional[Dict[str, Any]],
            allow_online: bool,
        ) -> Optional[PropertyResolutionResult]:
            if T is None:
                return None
            Tb_result = self._source_result_for_value(props, 'Tb', units='K')
            Tc_result = self._hvap_tc_result(props, online)
            Tb = (Tb_result.value if Tb_result else None) or (online or {}).get('Tb')
            Tc = (Tc_result.value if Tc_result else None) or (online or {}).get('Tc')
            if not Tb or not Tc:
                return None
            if Tb_result is None and (online or {}).get('Tb') is not None:
                Tb_result = PropertyResolutionResult(
                    value=(online or {}).get('Tb'),
                    source='online',
                    method=(online or {}).get('_sources', {}).get('Tb', 'online_phase_change'),
                    quality=0.93,
                    notes='Online phase-change data; units K',
                )
            try:
                Tb = float(Tb)
                Tc = float(Tc)
                T = float(T)
            except (TypeError, ValueError):
                return None

            reference = self._hvap_temperature_reference_at_tb(symbol, props, Tb, online)
            if reference is None:
                reference = self._provided_hvap_scalar(props)
            if reference is None:
                reference = self._textbook_hvap_result(symbol, props)
            if reference is None and allow_online:
                reference = self._online_hvap_scalar_result(symbol, online)
            if reference is None or reference.value is None:
                return None

            value = self._watson_hvap_value(float(reference.value), Tb, T, Tc)
            if value is None:
                return None
            return PropertyResolutionResult(
                value=value,
                source='calculated',
                method='watson_hvap',
                quality=temperature_scaled_hvap_quality(
                    HVAP_WATSON_QUALITY_FACTOR,
                    [
                        self._result_quality(item, 0.0)
                        for item in (reference, Tb_result)
                        if item is not None
                    ],
                    tc_quality=self._result_quality(Tc_result, 0.0),
                    reduced_temperature=T / Tc,
                ),
                notes=(
                    f"Watson scaling from {reference.source}/{reference.method} "
                    f"at Tb={Tb:g} K using Tc={Tc:g} K"
                ),
            )


        def _hvap_tc_result(
            self,
            props: Dict[str, Any],
            online: Optional[Dict[str, Any]] = None,
            *,
            prefer_online: bool = False,
        ) -> Optional[PropertyResolutionResult]:
            online_tc = (online or {}).get('Tc')
            if prefer_online and online_tc is not None:
                result = None
            else:
                result = self._source_result_for_value(props, 'Tc', units='K')
            if result is not None and result.value is not None:
                return result
            if online_tc is not None:
                return PropertyResolutionResult(
                    value=online_tc,
                    source='online',
                    method=(online or {}).get('_sources', {}).get(
                        'Tc',
                        'online_phase_change',
                    ),
                    quality=self._clamp_quality(
                        (online or {}).get('_qualities', {}).get('Tc'),
                        0.93,
                    ),
                    notes='Online critical data; units K',
                )
            return self._source_result_for_value(props, 'Tc', units='K')


        def _resolve_corresponding_states_hvap(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: Optional[float],
            *,
            allow_online: bool,
        ) -> Optional[PropertyResolutionResult]:
            if T is None:
                return None
            if self._hvap_is_carboxylic_acid(
                symbol,
                props,
                allow_online=allow_online,
            ):
                return self._carboxylic_acid_hvap_refusal()
            try:
                target_temperature = float(T)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(target_temperature) or target_temperature <= 0.0:
                return None

            try:
                critical = self.resolve_critical_properties(
                    symbol,
                    props,
                    allow_online=allow_online,
                    allow_estimation=True,
                )
            except Exception:
                return None
            tc_result = critical.get('Tc')
            omega_result = critical.get('omega')
            if (
                tc_result is None
                or tc_result.value is None
                or omega_result is None
                or omega_result.value is None
            ):
                return None
            try:
                tc = float(tc_result.value)
                omega = float(omega_result.value)
            except (TypeError, ValueError):
                return None
            if not (
                math.isfinite(tc)
                and tc > 0.0
                and math.isfinite(omega)
            ):
                return None
            reduced_temperature = target_temperature / tc
            if not (
                NANNOOLAL_HVAP_MAXIMUM_REDUCED_TEMPERATURE
                < reduced_temperature
                <= 1.0
            ):
                return None

            tau = 1.0 - reduced_temperature
            value_J_mol = R * tc * (
                7.08 * tau**0.354
                + 10.95 * omega * tau**0.456
            )
            if not math.isfinite(value_J_mol) or value_J_mol < 0.0:
                return None
            quality = temperature_scaled_hvap_quality(
                CORRESPONDING_STATES_HVAP_BASE_QUALITY,
                [self._result_quality(omega_result, 0.0)],
                tc_quality=self._result_quality(tc_result, 0.0),
                reduced_temperature=reduced_temperature,
            )
            return PropertyResolutionResult(
                value=value_J_mol / 1000.0,
                source='estimated',
                method='corresponding_states_hvap',
                quality=quality,
                notes=(
                    'Corresponding-states Hvap/(R*Tc) relation '
                    f'at T={target_temperature:g} K, '
                    f'Tr={reduced_temperature:g}; '
                    'quality=0.75*min(Tc quality, omega quality), with '
                    'enhanced Tc sensitivity above Tr=0.9; units kJ/mol'
                ),
            )


        def _hvap_smiles_result(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> Optional[PropertyResolutionResult]:
            smiles = self._smiles_result_for_boiling_point(props)
            if smiles is not None and smiles.value:
                return smiles
            return self._resolve_smiles_result(
                symbol,
                props,
                allow_online=allow_online,
            )


        def _hvap_is_carboxylic_acid(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
            smiles_result: Optional[PropertyResolutionResult] = None,
        ) -> bool:
            smiles_result = (
                smiles_result
                or self._smiles_result_for_boiling_point(props)
            )
            cache = self._hvap_carboxylic_acid_cache
            explicit_smiles = (
                str(smiles_result.value).strip()
                if smiles_result is not None and smiles_result.value
                else ''
            )
            if explicit_smiles:
                raw_smiles_key = ('smiles_raw', explicit_smiles)
                if raw_smiles_key in cache:
                    return cache[raw_smiles_key]
                canonical_smiles = None
                try:
                    from rdkit import Chem

                    molecule = Chem.MolFromSmiles(explicit_smiles)
                    if molecule is not None:
                        canonical_smiles = Chem.MolToSmiles(
                            molecule,
                            isomericSmiles=True,
                        )
                except Exception:
                    canonical_smiles = None
                if canonical_smiles is not None:
                    canonical_smiles_key = ('smiles', canonical_smiles)
                    if canonical_smiles_key in cache:
                        result = cache[canonical_smiles_key]
                        cache[raw_smiles_key] = result
                        return result
                    profile = hydrogen_bond_donor_profile(explicit_smiles)
                    if profile is not None:
                        result = bool(profile.carboxylic_acid_oh)
                        cache[canonical_smiles_key] = result
                        cache[raw_smiles_key] = result
                        return result

            formula = str(
                props.get('formula')
                or props.get('Formula')
                or symbol
                or ''
            ).strip()
            formula_counts = parse_formula_counts(formula) if formula else None
            formula_key = (
                (
                    'formula_nonacid',
                    tuple(sorted(formula_counts.items())),
                )
                if formula_counts
                else None
            )
            if formula_key is not None and formula_key in cache:
                return cache[formula_key]
            if formula_counts and (
                formula_counts.get('C', 0) < 1
                or formula_counts.get('H', 0) < 1
                or formula_counts.get('O', 0) < 2
            ):
                cache[formula_key] = False
                return False

            cas = str(props.get('CAS') or props.get('cas') or '').strip()
            candidates = tuple(
                str(identifier)
                for identifier in self._identifier_candidates(symbol, props)
            )
            identity_key = (
                ('cas', cas.lower())
                if cas
                else ('identifiers', candidates, explicit_smiles)
            )
            if identity_key in cache:
                return cache[identity_key]
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..vapor_dimerization import is_monocarboxylic_acid
                else:
                    from vapor_dimerization import is_monocarboxylic_acid
            except ImportError:
                return False
            for identifier in candidates:
                try:
                    if is_monocarboxylic_acid(
                        str(identifier),
                        smiles=explicit_smiles or None,
                    ):
                        cache[identity_key] = True
                        return True
                except (ArithmeticError, LookupError, TypeError, ValueError):
                    continue
            cache[identity_key] = False
            return False


        @staticmethod
        def _carboxylic_acid_hvap_refusal() -> PropertyResolutionResult:
            return PropertyResolutionResult(
                value=None,
                source='missing',
                method='carboxylic_acid_hvap_estimation_refused',
                quality=0.0,
                notes=(
                    'Nannoolal, corresponding-states, and Trouton Hvap '
                    'estimates are refused for '
                    'carboxylic acids because vapor association makes their '
                    'Psat-derived apparent enthalpy incompatible with the '
                    'ordinary calorimetric Hvap target'
                ),
            )


        def _resolve_nannoolal_hvap(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: Optional[float],
            *,
            allow_online: bool,
        ) -> Optional[PropertyResolutionResult]:
            tb_result = self._source_result_for_value(props, 'Tb', units='K')
            if tb_result is None or tb_result.value is None:
                return None
            smiles_result = self._hvap_smiles_result(
                symbol,
                props,
                allow_online=allow_online,
            )
            if smiles_result is None or not smiles_result.value:
                return None
            if self._hvap_is_carboxylic_acid(
                symbol,
                props,
                allow_online=allow_online,
                smiles_result=smiles_result,
            ):
                return self._carboxylic_acid_hvap_refusal()
            try:
                tb = float(tb_result.value)
                target_temperature = tb if T is None else float(T)
            except (TypeError, ValueError):
                return None
            if not (
                math.isfinite(tb)
                and tb > 0.0
                and math.isfinite(target_temperature)
                and target_temperature > 0.0
            ):
                return None

            try:
                critical = self.resolve_critical_properties(
                    symbol,
                    props,
                    allow_online=allow_online,
                    allow_estimation=True,
                )
            except Exception:
                return None
            critical_results = [
                critical.get(name) for name in ('Tc', 'Pc', 'omega')
            ]
            if any(
                result is None or result.value is None
                for result in critical_results
            ):
                return None
            tc_result, pc_result, omega_result = critical_results
            try:
                tc = float(tc_result.value)
                pc_bar = float(pc_result.value)
                omega = float(omega_result.value)
            except (TypeError, ValueError):
                return None
            reduced_temperature = target_temperature / tc
            if not (
                math.isfinite(reduced_temperature)
                and 0.0 < reduced_temperature
                <= NANNOOLAL_HVAP_MAXIMUM_REDUCED_TEMPERATURE
                and math.isfinite(pc_bar)
                and pc_bar > 0.0
                and math.isfinite(omega)
            ):
                return None

            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..nannoolal_method import estimate_psat, NannoolalError
                else:
                    from nannoolal_method import estimate_psat, NannoolalError
            except ImportError:
                return None
            try:
                estimate = estimate_psat(str(smiles_result.value), tb=tb)
            except (ArithmeticError, TypeError, ValueError, NannoolalError):
                return None
            if estimate.groups.get(44):
                return self._carboxylic_acid_hvap_refusal()
            if estimate.db is None:
                return None
            pressure_kPa = estimate.psat_kPa(target_temperature)
            if (
                pressure_kPa is None
                or not math.isfinite(pressure_kPa)
                or pressure_kPa <= 0.0
            ):
                return None
            try:
                from .vapor_pressure_adapter import (
                    _peng_robinson_delta_z_or_ideal,
                )
                delta_z = _peng_robinson_delta_z_or_ideal(
                    target_temperature,
                    pressure_kPa / 100.0,
                    tc,
                    pc_bar,
                    omega,
                )
            except (ImportError, ArithmeticError, TypeError, ValueError):
                return None
            if delta_z == 1.0:
                return None
            value_J_mol = estimate.dhvap_J_mol(
                target_temperature,
                dz_vap=delta_z,
            )
            if (
                value_J_mol is None
                or not math.isfinite(value_J_mol)
                or value_J_mol <= 0.0
            ):
                return None

            critical_quality = min(
                self._result_quality(result, 0.0)
                for result in critical_results
            )
            critical_factor = 1.0 - (1.0 - critical_quality) / 5.0
            quality = self._clamp_quality(
                NANNOOLAL_HVAP_BASE_QUALITY
                * self._result_quality(tb_result, 0.0)
                * critical_factor
            )
            return PropertyResolutionResult(
                value=value_J_mol / 1000.0,
                source='estimated',
                method='nannoolal_hvap_pr',
                quality=quality,
                notes=(
                    f'Nannoolal Part-3 Psat slope with Peng-Robinson delta Z '
                    f'at T={target_temperature:g} K, Tr={reduced_temperature:g}; '
                    f'Tb from {tb_result.source}/{tb_result.method}; '
                    f'SMILES from {smiles_result.source}/{smiles_result.method}; '
                    f'critical quality minimum={critical_quality:g}; '
                    f'quality=0.8*Tb quality*(1-(1-critical quality)/5); '
                    f'units kJ/mol'
                ),
            )


        def _resolve_trouton_hvap(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: Optional[float],
            *,
            allow_online: bool,
        ) -> Optional[PropertyResolutionResult]:
            """Return the shared Trouton estimate, Watson-scaled when possible."""
            if self._hvap_is_carboxylic_acid(
                symbol,
                props,
                allow_online=allow_online,
            ):
                return self._carboxylic_acid_hvap_refusal()
            tb_result = self._source_result_for_value(props, 'Tb', units='K')
            if tb_result is None or tb_result.value is None:
                return None
            try:
                tb = float(tb_result.value)
                target_temperature = tb if T is None else float(T)
                reference_hvap = trouton_hvap_at_tb_kj_mol(tb)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(tb) or tb <= 0.0:
                return None

            tc_result = self._source_result_for_value(props, 'Tc', units='K')
            if tc_result is not None and tc_result.value is not None:
                try:
                    tc = float(tc_result.value)
                except (TypeError, ValueError):
                    tc = None
                if tc is not None and math.isfinite(tc) and tb < tc:
                    value = self._watson_hvap_value(
                        reference_hvap,
                        tb,
                        target_temperature,
                        tc,
                    )
                    if value is not None:
                        quality = trouton_hvap_quality(
                            self._result_quality(tb_result, 0.0),
                            self._result_quality(tc_result, 0.0),
                            reduced_temperature=target_temperature / tc,
                        )
                        return PropertyResolutionResult(
                            value=value,
                            source='estimated',
                            method='trouton_watson',
                            quality=quality,
                            notes=(
                                f'Trouton estimate at Tb={tb:g} K, '
                                f'Watson-scaled to T={target_temperature:g} K '
                                f'using Tc={tc:g} K; units kJ/mol; '
                                f'quality=0.72*min(Tb quality, Tc quality), with '
                                f'enhanced Tc sensitivity above Tr=0.9'
                            ),
                        )

            quality = trouton_hvap_quality(
                self._result_quality(tb_result, 0.0),
            )
            return PropertyResolutionResult(
                value=reference_hvap,
                source='estimated',
                method='trouton',
                quality=quality,
                notes=(
                    f'Unscaled Trouton estimate at Tb={tb:g} K; '
                    f'unable to construct Watson curve; units kJ/mol; '
                    f'quality=0.55*Tb quality'
                ),
            )


        def resolve_boiling_point(
            self,
            symbol: str,
            props: Dict[str, Any] = None,
            allow_online: bool = True,
            allow_estimation: bool = True,
        ) -> PropertyResolutionResult:
            """Resolve normal boiling point in K."""
            explicit_props = props is not None
            props = self._coerce_props(
                symbol,
                props,
                allow_online=allow_online if explicit_props else False,
            )
            results = self._cached_phase_point_results(
                'boiling_point',
                symbol,
                props,
                allow_online=allow_online,
                allow_estimation=allow_estimation,
                build=lambda: {
                    'Tb': self._resolve_boiling_point_uncached(
                        symbol,
                        props,
                        allow_online=allow_online,
                        allow_estimation=allow_estimation,
                    )
                },
            )
            return results['Tb']


        def _resolve_boiling_point_uncached(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
            allow_estimation: bool,
        ) -> PropertyResolutionResult:
            """Run the established boiling-point source order unchanged."""
            provided = self._provided_scalar(props, 'Tb', 'K')
            if provided and self._is_pfd_component_override(props, 'Tb'):
                return provided

            phase_constraint = self._normal_boiling_point_phase_constraint(props)
            if phase_constraint is not None:
                return phase_constraint

            rejected_below_triple = []
            rejected_by_triple_pressure = []
            Tt_result = self._source_result_for_value(props, 'Tt', units='K')
            Pt_result = self._source_result_for_value(props, 'Pt', units='bar')

            def positive_value(result):
                if result is None or result.value is None:
                    return None
                try:
                    value = float(result.value)
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(value) or value <= 0.0:
                    return None
                return value

            triple_temperature = positive_value(Tt_result)
            triple_pressure = positive_value(Pt_result)
            Tt_quality = self._result_quality(Tt_result, default=0.0)
            Pt_quality = self._result_quality(Pt_result, default=0.0)
            provisional_topology_quality = min(Tt_quality, Pt_quality)
            hard_Tt = (
                Tt_result is not None
                and not self._result_is_soft(Tt_result)
            )

            def accepted(candidate):
                if candidate is None:
                    return None
                try:
                    temperature = float(candidate.value)
                except (TypeError, ValueError):
                    return candidate
                candidate_quality = self._result_quality(
                    candidate,
                    default=0.0,
                )
                if (
                    triple_temperature is not None
                    and math.isfinite(temperature)
                    and temperature < triple_temperature - 1.0e-7
                    and (
                        hard_Tt
                        or Tt_quality > candidate_quality + 1.0e-12
                    )
                ):
                    rejected_below_triple.append(
                        f'{candidate.method} returned Tb={temperature:g} K below '
                        f'Tt={triple_temperature:g} K; '
                        f'Tt quality={Tt_quality:g}, '
                        f'Tb quality={candidate_quality:g}'
                    )
                    return None
                if (
                    triple_temperature is not None
                    and triple_pressure is not None
                    and triple_pressure
                    >= NORMAL_BOILING_PRESSURE_BAR - 1.0e-9
                    and provisional_topology_quality
                    > candidate_quality + 1.0e-12
                ):
                    rejected_by_triple_pressure.append(
                        f'{candidate.method} returned Tb={temperature:g} K but '
                        f'Pt={triple_pressure:g} bar is not below '
                        f'{NORMAL_BOILING_PRESSURE_BAR:g} bar; '
                        f'triple topology quality='
                        f'{provisional_topology_quality:g}, '
                        f'Tb quality={candidate_quality:g}'
                    )
                    return None
                return candidate

            hydrated_tb = self._source_result_for_value(props, 'Tb', units='K')

            coolprop_tb = accepted(self._coolprop_boiling_point(symbol, props))
            if coolprop_tb:
                return coolprop_tb

            accepted_provided = accepted(provided)
            if accepted_provided:
                return accepted_provided

            for candidate in self._identifier_candidates(symbol, props):
                entry = self._get_textbook_entry(candidate)
                if entry and entry.get('Tb') is not None:
                    textbook_tb = PropertyResolutionResult(
                        value=entry['Tb'],
                        source='textbook',
                        method='Smith8 Appendix B',
                        quality=0.98,
                        notes='units K',
                    )
                    textbook_tb = accepted(textbook_tb)
                    if textbook_tb:
                        return textbook_tb

            perry_tb = self._get_perry_evaluation(symbol, props, 'normal_boiling_point_K')
            if perry_tb:
                perry_result = accepted(PropertyResolutionResult(
                    value=perry_tb.value,
                    source='local',
                    method=perry_tb.method,
                    quality=0.97,
                    notes=f"{perry_tb.source}; units {perry_tb.units}",
                ))
                if perry_result:
                    return perry_result

            perry_table_tb = self._get_perry_evaluation(symbol, props, 'table_2_10_normal_boiling_point_K')
            if perry_table_tb:
                perry_table_result = accepted(PropertyResolutionResult(
                    value=perry_table_tb.value,
                    source='local',
                    method=perry_table_tb.method,
                    quality=0.94,
                    notes=f"{perry_table_tb.source}; units {perry_table_tb.units}",
                ))
                if perry_table_result:
                    return perry_table_result

            accepted_hydrated = accepted(hydrated_tb)
            if accepted_hydrated:
                return accepted_hydrated

            if allow_online:
                try:
                    online = self._fetch_phase_change_online(symbol, props)
                except LookupError:
                    online = None
                if online and online.get('Tb') is not None:
                    method = online.get('_sources', {}).get('Tb', 'online_phase_change')
                    online_result = accepted(PropertyResolutionResult(
                        value=online['Tb'],
                        source='online',
                        method=method,
                        quality=0.95 if method == 'pubchem' else 0.93,
                        notes='units K',
                    ))
                    if online_result:
                        return online_result

            if allow_estimation:
                estimated = accepted(self._estimate_boiling_point_fallback(props))
                if estimated:
                    return estimated
            if rejected_by_triple_pressure:
                return PropertyResolutionResult(
                    value=None,
                    source='missing',
                    method='no_normal_boiling_point_at_1atm',
                    quality=0.0,
                    notes='; '.join(rejected_by_triple_pressure),
                )
            if rejected_below_triple:
                return PropertyResolutionResult(
                    value=None,
                    source='missing',
                    method='invalid_normal_boiling_point_below_triple_point',
                    quality=0.0,
                    notes='; '.join(rejected_below_triple),
                )
            return self._missing_scalar('Tb')


        def _normal_boiling_point_phase_constraint(
            self,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            """Return definitive unavailability when no liquid exists at 1 atm."""
            Tt = self._source_result_for_value(props, 'Tt', units='K')
            Pt = self._source_result_for_value(props, 'Pt', units='bar')
            if (
                Tt is None
                or Pt is None
                or Tt.value is None
                or Pt.value is None
                or self._result_is_soft(Tt)
                or self._result_is_soft(Pt)
            ):
                return None
            try:
                triple_temperature = float(Tt.value)
                triple_pressure = float(Pt.value)
            except (TypeError, ValueError):
                return None
            if (
                not math.isfinite(triple_temperature)
                or triple_temperature <= 0.0
                or not math.isfinite(triple_pressure)
                or triple_pressure < NORMAL_BOILING_PRESSURE_BAR - 1.0e-9
            ):
                return None
            return PropertyResolutionResult(
                value=None,
                source='missing',
                method='no_normal_boiling_point_at_1atm',
                quality=0.0,
                notes=(
                    f'No stable liquid phase at 1 atm because triple pressure '
                    f'Pt={triple_pressure:g} bar is not below '
                    f'{NORMAL_BOILING_PRESSURE_BAR:g} bar; '
                    f'Tt={triple_temperature:g} K'
                ),
            )


        def _coolprop_phase_reference(self, symbol: str, props: Dict[str, Any]):
            cached = getattr(self, '_coolprop_reference_cache', None)
            if cached is None:
                cached = {}
                self._coolprop_reference_cache = cached
            cache_key = (
                str(symbol),
                str(props.get('CAS') or ''),
                str(props.get('cas') or ''),
            )
            if cache_key not in cached:
                cache_key_value = coolprop_reference_for(symbol, props)
                cached[cache_key] = cache_key_value
            return cached[cache_key]


        def _coolprop_boiling_point(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            reference = self._coolprop_phase_reference(symbol, props)
            if reference is None:
                return None
            triple_temperature = coolprop_props_si('Ttriple', reference)
            triple_pressure_pa = coolprop_props_si('ptriple', reference)
            normal_pressure_pa = NORMAL_BOILING_PRESSURE_BAR * 100000.0
            if (
                triple_pressure_pa is not None
                and triple_pressure_pa >= normal_pressure_pa - 1.0e-4
            ):
                return None
            temperature = coolprop_saturation_temperature(
                reference,
                normal_pressure_pa,
            )
            if temperature is None:
                return None
            if (
                triple_temperature is not None
                and temperature < triple_temperature - 1.0e-7
            ):
                return None
            return PropertyResolutionResult(
                value=temperature,
                source='local',
                method=f'{reference.method_prefix}_boiling_point',
                quality=COOLPROP_PROPERTY_QUALITY,
                notes=(
                    f'CoolProp {reference.backend} pure-fluid saturation '
                    f'temperature for {reference.fluid} at 1 atm; units K'
                ),
            )


        def resolve_triple_point(
            self,
            symbol: str,
            props: Dict[str, Any] = None,
            allow_online: bool = True,
        ) -> Dict[str, PropertyResolutionResult]:
            """Resolve triple-point temperature in K and pressure in bar."""
            explicit_props = props is not None
            props = self._coerce_props(
                symbol,
                props,
                allow_online=allow_online if explicit_props else False,
            )
            has_complete_pfd_pair = all(
                self._is_pfd_component_override(props, key)
                and props.get(key) is not None
                for key in ('Tt', 'Pt')
            )
            if not has_complete_pfd_pair:
                independent_tm = self._independent_melting_validator(
                    symbol,
                    props,
                )
                melting = self.resolve_melting_point(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
                if melting.value is not None:
                    props = dict(props)
                    props['Tm'] = melting.value
                    sources = dict(props.get('property_sources') or {})
                    sources['Tm'] = {
                        'source': melting.source,
                        'method': melting.method,
                        'quality': melting.quality,
                        'notes': melting.notes,
                    }
                    if independent_tm is not None:
                        props['_phase_independent_tm_validator'] = asdict(
                            independent_tm
                        )
                    props['property_sources'] = sources
            return self._cached_phase_point_results(
                'triple_point',
                symbol,
                props,
                allow_online=allow_online,
                allow_estimation=False,
                build=lambda: self._resolve_triple_point_uncached(
                    symbol,
                    props,
                    allow_online=allow_online,
                ),
            )


        def _resolve_triple_point_uncached(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> Dict[str, PropertyResolutionResult]:
            """Resolve triple-point values after PFD/CoolProp/local/online arbitration."""
            results = {}
            for key, units in (('Tt', 'K'), ('Pt', 'bar')):
                provided = self._provided_scalar(props, key, units)
                if provided and self._is_pfd_component_override(props, key):
                    results[key] = provided

            coolprop_rejection = None
            if not results:
                reference = self._coolprop_phase_reference(symbol, props)
                coolprop_results = {}
                coolprop_confirmed = False
                coolprop_confirmation = ''
                if reference is not None:
                    coolprop_values = {
                        'Tt': coolprop_props_si('Ttriple', reference),
                        'Pt': None,
                    }
                    pressure_pa = coolprop_props_si('ptriple', reference)
                    if pressure_pa is not None and pressure_pa > 0.0:
                        coolprop_values['Pt'] = pressure_pa / 100000.0
                    coolprop_rejection = self._coolprop_triple_point_rejection(
                        props,
                        coolprop_values,
                    )
                    if coolprop_rejection is None:
                        (
                            coolprop_confirmed,
                            coolprop_confirmation,
                        ) = self._coolprop_triple_point_confirmation(
                            props,
                            reference,
                            coolprop_values,
                        )
                        coolprop_results = self._triple_results_from_values(
                            coolprop_values,
                            source='local',
                            methods={
                                key: f'{reference.method_prefix}_triple_point'
                                for key in ('Tt', 'Pt')
                            },
                            qualities={
                                key: (
                                    COOLPROP_PROPERTY_QUALITY
                                    if coolprop_confirmed
                                    else self.COOLPROP_PROVISIONAL_TRIPLE_QUALITY
                                )
                                for key in ('Tt', 'Pt')
                            },
                            notes={
                                key: (
                                    f'CoolProp {reference.backend} pure-fluid '
                                    f'triple point for {reference.fluid}; '
                                    f"units {'K' if key == 'Tt' else 'bar'}; "
                                    + (
                                        f'confirmed by {coolprop_confirmation}'
                                        if coolprop_confirmed
                                        else (
                                            'provisional: no independent fusion '
                                            'or matching solid-liquid/VLE closure'
                                        )
                                    )
                                )
                                for key in ('Tt', 'Pt')
                            },
                        )

                hydrated_results = {}
                for key in ('Tt', 'Pt'):
                    hydrated = self._source_result_for_value(
                        props,
                        key,
                        units='K' if key == 'Tt' else 'bar',
                    )
                    if hydrated and not self._phase_result_is_coolprop(hydrated):
                        hydrated_results[key] = hydrated
                hydrated_rejection = self._triple_temperature_rejection(
                    props,
                    {
                        key: item.value
                        for key, item in hydrated_results.items()
                    },
                    provider='Hydrated triple point',
                )
                if hydrated_rejection is not None:
                    hydrated_results = {}

                online_results = {}
                online_tm = None
                need_online = allow_online and not coolprop_confirmed and (
                    (
                        coolprop_results
                        and 'Tt' not in hydrated_results
                    )
                    or (
                        not coolprop_results
                        and any(
                            key not in hydrated_results
                            for key in ('Tt', 'Pt')
                        )
                    )
                )
                if need_online:
                    try:
                        online = self._fetch_phase_change_online(symbol, props)
                    except LookupError:
                        online = None
                    if online:
                        online_tm = self._triple_online_melting_result(online)
                        online_values = {
                            key: online.get(key)
                            for key in ('Tt', 'Pt')
                        }
                        online_rejection = self._triple_temperature_rejection(
                            props,
                            online_values,
                            provider='Online consensus',
                        )
                        if online_rejection is not None:
                            online_values = {}
                        online_results = self._triple_results_from_values(
                            online_values,
                            source='online',
                            methods={
                                key: (online.get('_sources') or {}).get(
                                    key,
                                    'online_phase_change',
                                )
                                for key in ('Tt', 'Pt')
                            },
                            qualities={
                                key: float(
                                    (online.get('_qualities') or {}).get(
                                        key,
                                        0.88,
                                    )
                                )
                                for key in ('Tt', 'Pt')
                            },
                            notes={
                                key: (online.get('_notes') or {}).get(
                                    key,
                                    f"units {'K' if key == 'Tt' else 'bar'}",
                                )
                                for key in ('Tt', 'Pt')
                            },
                        )

                corroborator = (
                    hydrated_results
                    if 'Tt' in hydrated_results
                    else online_results
                )
                if coolprop_results and coolprop_confirmed:
                    results = coolprop_results
                elif coolprop_results and 'Tt' in corroborator:
                    coolprop_agrees = self._phase_temperatures_consistent(
                        coolprop_results['Tt'].value,
                        corroborator['Tt'].value,
                    )
                    online_tm_supports_coolprop = (
                        online_tm is not None
                        and self._phase_temperatures_consistent(
                            coolprop_results['Tt'].value,
                            online_tm.value,
                        )
                    )
                    online_tm_supports_external = (
                        online_tm is not None
                        and self._phase_temperatures_consistent(
                            corroborator['Tt'].value,
                            online_tm.value,
                        )
                    )
                    if coolprop_agrees or (
                        online_tm_supports_coolprop
                        and not online_tm_supports_external
                    ):
                        label = (
                            f'{corroborator["Tt"].source}/'
                            f'{corroborator["Tt"].method}'
                        )
                        if not coolprop_agrees and online_tm is not None:
                            label = (
                                f'{online_tm.source}/{online_tm.method} Tm='
                                f'{float(online_tm.value):g} K over conflicting '
                                f'{label}'
                            )
                        results = self._confirmed_coolprop_triple_results(
                            coolprop_results,
                            f'corroborating {label} Tt='
                            f'{float(corroborator["Tt"].value):g} K'
                            if coolprop_agrees
                            else label,
                        )
                    else:
                        results = corroborator
                        coolprop_rejection = (
                            f'Provisional CoolProp Tt='
                            f'{float(coolprop_results["Tt"].value):g} K '
                            f'conflicts with {corroborator["Tt"].source}/'
                            f'{corroborator["Tt"].method} Tt='
                            f'{float(corroborator["Tt"].value):g} K; '
                            'external result selected'
                        )
                elif coolprop_results and online_tm is not None:
                    if self._phase_temperatures_consistent(
                        coolprop_results['Tt'].value,
                        online_tm.value,
                    ):
                        results = self._confirmed_coolprop_triple_results(
                            coolprop_results,
                            f'corroborating {online_tm.source}/'
                            f'{online_tm.method} Tm='
                            f'{float(online_tm.value):g} K',
                        )
                    else:
                        coolprop_rejection = (
                            f'Provisional CoolProp Tt='
                            f'{float(coolprop_results["Tt"].value):g} K '
                            f'conflicts with independent '
                            f'{online_tm.source}/{online_tm.method} Tm='
                            f'{float(online_tm.value):g} K; CoolProp pair rejected'
                        )
                        online_results = {}
                elif coolprop_results:
                    results = coolprop_results
                elif hydrated_results:
                    results = dict(hydrated_results)
                    for key, item in online_results.items():
                        results.setdefault(key, item)
                elif online_results:
                    results = online_results

                if results and coolprop_rejection is not None:
                    results = self._triple_results_with_note(
                        results,
                        coolprop_rejection,
                    )

            if (
                'Pt' not in results
                and results.get('Tt') is not None
                and results['Tt'].value is not None
                and not self._is_pfd_component_override(props, 'Tt')
                and not self._is_pfd_component_override(props, 'Pt')
                and not self._is_pfd_correlation_override(props, 'Psat')
            ):
                component_key, _cas, _name = (
                    self._phase_point_component_identity(symbol, props)
                )
                derived = self._load_canonical_triple_pressure_backfill(
                    component_key=component_key,
                    triple_temperature=results['Tt'],
                    allow_online=allow_online,
                )
                if derived is not None:
                    results['Pt'] = derived

            has_selected_values = bool(results)
            for key in ('Tt', 'Pt'):
                if key in results:
                    continue
                if coolprop_rejection is None or has_selected_values:
                    missing = self._missing_scalar(key)
                    if coolprop_rejection is None:
                        results[key] = missing
                    else:
                        results[key] = PropertyResolutionResult(
                            value=None,
                            source=missing.source,
                            method=missing.method,
                            quality=missing.quality,
                            notes='; '.join(filter(None, (
                                missing.notes,
                                coolprop_rejection,
                            ))),
                        )
                else:
                    results[key] = PropertyResolutionResult(
                        value=None,
                        source='missing',
                        method='coolprop_triple_point_inconsistent_with_melting_point',
                        quality=0.0,
                        notes=coolprop_rejection,
                    )
            return results


        @staticmethod
        def _triple_results_from_values(
            values: Mapping[str, Any],
            *,
            source: str,
            methods: Mapping[str, str],
            qualities: Mapping[str, float],
            notes: Mapping[str, str],
        ) -> Dict[str, PropertyResolutionResult]:
            results = {}
            for key in ('Tt', 'Pt'):
                value = values.get(key)
                try:
                    value = float(value)
                    quality = float(qualities.get(key, 0.0))
                except (TypeError, ValueError):
                    continue
                if (
                    not math.isfinite(value)
                    or value <= 0.0
                    or not math.isfinite(quality)
                    or not 0.0 <= quality <= 1.0
                ):
                    continue
                results[key] = PropertyResolutionResult(
                    value=value,
                    source=source,
                    method=str(methods.get(key) or 'triple_point'),
                    quality=quality,
                    notes=str(notes.get(key) or ''),
                )
            return results


        def _triple_online_melting_result(
            self,
            online: Mapping[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            try:
                value = float(online.get('Tm'))
                quality = float(
                    (online.get('_qualities') or {}).get('Tm', 0.88)
                )
            except (TypeError, ValueError):
                return None
            result = PropertyResolutionResult(
                value=value,
                source='online',
                method=str(
                    (online.get('_sources') or {}).get(
                        'Tm',
                        'online_phase_change',
                    )
                ),
                quality=quality,
                notes=str(
                    (online.get('_notes') or {}).get('Tm', 'units K')
                ),
            )
            if (
                not math.isfinite(value)
                or value <= 0.0
                or not math.isfinite(quality)
                or not 0.0 <= quality <= 1.0
                or self._result_is_soft(result)
            ):
                return None
            return result


        def _phase_temperatures_consistent(
            self,
            first: Any,
            second: Any,
        ) -> bool:
            try:
                first_value = float(first)
                second_value = float(second)
            except (TypeError, ValueError):
                return False
            if (
                not math.isfinite(first_value)
                or first_value <= 0.0
                or not math.isfinite(second_value)
                or second_value <= 0.0
            ):
                return False
            tolerance = max(
                self.PHASE_TEMPERATURE_ABS_TOL_K,
                self.PHASE_TEMPERATURE_REL_TOL * second_value,
            )
            return abs(first_value - second_value) <= tolerance


        def _independent_hard_melting_result(
            self,
            props: Mapping[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            source_metadata = props.get('property_sources')
            selected_tm_metadata = (
                source_metadata.get('Tm')
                if isinstance(source_metadata, Mapping)
                else None
            )
            preserved = (
                selected_tm_metadata.get('independent_validator')
                if isinstance(selected_tm_metadata, Mapping)
                else None
            )
            for internal in (
                props.get('_phase_independent_tm_validator'),
                preserved,
            ):
                if not isinstance(internal, Mapping):
                    continue
                try:
                    result = PropertyResolutionResult(
                        value=internal.get('value'),
                        source=str(internal.get('source') or ''),
                        method=str(internal.get('method') or ''),
                        quality=float(internal.get('quality', 0.0)),
                        notes=str(internal.get('notes') or ''),
                    )
                except (TypeError, ValueError):
                    result = None
                if (
                    result is not None
                    and result.value is not None
                    and not self._result_is_soft(result)
                    and not self._phase_result_is_coolprop(result)
                ):
                    return result
            result = self._source_result_for_value(
                dict(props),
                'Tm',
                units='K',
            )
            if (
                result is None
                or result.value is None
                or self._result_is_soft(result)
                or self._phase_result_is_coolprop(result)
            ):
                return None
            return result


        def _independent_melting_validator(
            self,
            symbol: str,
            props: Mapping[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            provided = self._independent_hard_melting_result(props)
            if provided is not None:
                return provided
            perry = self._perry_melting_result(symbol, dict(props))
            if (
                perry is not None
                and not self._result_is_soft(perry)
                and not self._phase_result_is_coolprop(perry)
            ):
                return perry
            return None


        def _coolprop_triple_point_confirmation(
            self,
            props: Mapping[str, Any],
            reference,
            values: Mapping[str, Any],
        ) -> tuple[bool, str]:
            Tt = values.get('Tt')
            Pt = values.get('Pt')
            independent_tm = self._independent_hard_melting_result(props)
            if (
                independent_tm is not None
                and Tt is not None
                and self._phase_temperatures_consistent(
                    Tt,
                    independent_tm.value,
                )
            ):
                return True, (
                    f'independent {independent_tm.source}/'
                    f'{independent_tm.method} Tm='
                    f'{float(independent_tm.value):g} K'
                )
            try:
                Tt_value = float(Tt)
                Pt_pa = float(Pt) * 100000.0
            except (TypeError, ValueError):
                return False, ''
            melting_pressure = coolprop_melting_pressure(
                reference,
                Tt_value,
            )
            if melting_pressure is None or not math.isfinite(Pt_pa) or Pt_pa <= 0.0:
                return False, ''
            relative_difference = abs(melting_pressure - Pt_pa) / Pt_pa
            if relative_difference > self.COOLPROP_TRIPLE_CLOSURE_REL_TOL:
                return False, ''
            return True, (
                f'internal solid-liquid/VLE pressure closure '
                f'Pmelt(Tt)={melting_pressure / 100000.0:g} bar versus '
                f'Pt={Pt_pa / 100000.0:g} bar '
                f'({100.0 * relative_difference:.4g}% difference)'
            )


        @staticmethod
        def _confirmed_coolprop_triple_results(
            results: Mapping[str, PropertyResolutionResult],
            confirmation: str,
        ) -> Dict[str, PropertyResolutionResult]:
            return {
                key: PropertyResolutionResult(
                    value=result.value,
                    source=result.source,
                    method=result.method,
                    quality=COOLPROP_PROPERTY_QUALITY,
                    notes='; '.join(filter(None, (
                        result.notes,
                        f'confirmed by {confirmation}',
                    ))),
                )
                for key, result in results.items()
            }


        @staticmethod
        def _triple_results_with_note(
            results: Mapping[str, PropertyResolutionResult],
            note: str,
        ) -> Dict[str, PropertyResolutionResult]:
            return {
                key: PropertyResolutionResult(
                    value=result.value,
                    source=result.source,
                    method=result.method,
                    quality=result.quality,
                    notes='; '.join(filter(None, (result.notes, note))),
                )
                for key, result in results.items()
            }


        def _coolprop_triple_point_rejection(
            self,
            props: Dict[str, Any],
            values: Mapping[str, Any],
        ) -> Optional[str]:
            """Reject CoolProp lower-limit placeholders using a hard Tm anchor."""
            return self._triple_temperature_rejection(
                props,
                values,
                provider='CoolProp',
            )


        def _triple_temperature_rejection(
            self,
            props: Dict[str, Any],
            values: Mapping[str, Any],
            *,
            provider: str,
        ) -> Optional[str]:
            Tm = self._independent_hard_melting_result(props)
            if (
                Tm is None
                or Tm.value is None
                or values.get('Tt') is None
            ):
                return None
            try:
                melting_temperature = float(Tm.value)
                triple_temperature = float(values['Tt'])
            except (TypeError, ValueError):
                return None
            if (
                not math.isfinite(melting_temperature)
                or melting_temperature <= 0.0
                or not math.isfinite(triple_temperature)
                or triple_temperature <= 0.0
            ):
                return None
            tolerance = max(
                self.PHASE_TEMPERATURE_ABS_TOL_K,
                self.PHASE_TEMPERATURE_REL_TOL * melting_temperature,
            )
            difference = abs(triple_temperature - melting_temperature)
            if difference <= tolerance:
                return None
            return (
                f'{provider} Tt={triple_temperature:g} K differs from hard '
                f'Tm={melting_temperature:g} K by {difference:g} K, exceeding '
                f'the {tolerance:g} K fusion-line consistency tolerance'
            )


        def resolve_melting_point(
            self,
            symbol: str,
            props: Dict[str, Any] = None,
            allow_online: bool = True,
        ) -> PropertyResolutionResult:
            """Resolve normal melting point in K."""
            explicit_props = props is not None
            props = self._coerce_props(
                symbol,
                props,
                allow_online=allow_online if explicit_props else False,
            )
            results = self._cached_phase_point_results(
                'melting_point',
                symbol,
                props,
                allow_online=allow_online,
                allow_estimation=False,
                build=lambda: {
                    'Tm': self._resolve_melting_point_uncached(
                        symbol,
                        props,
                        allow_online=allow_online,
                    )
                },
            )
            return results['Tm']


        def _perry_melting_result(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            evaluation = self._get_perry_evaluation(
                symbol,
                props,
                'normal_melting_point_K',
            )
            if evaluation is None:
                return None
            quality = 0.95 if evaluation.method in {
                'perry_heat_of_fusion_melting_point',
                'perry_table_2_10_normal_melting_point',
            } else 0.97
            return PropertyResolutionResult(
                value=evaluation.value,
                source='local',
                method=evaluation.method,
                quality=quality,
                notes=f'{evaluation.source}; units {evaluation.units}',
            )


        def _provided_melting_transition_result(
            self,
            props: Mapping[str, Any],
            identity_text: str,
        ) -> Optional[PropertyResolutionResult]:
            records = []
            for raw in props.get('melting_transitions') or []:
                normalized = self._melting_record_from_mapping(
                    raw,
                    identity_text=identity_text,
                )
                if normalized is not None:
                    records.append(normalized)
            eligible = [
                record for record in records
                if record.material_form not in {'hydrate', 'solvate'}
                or record.is_identity_form
            ]
            if not eligible:
                return None
            identity_records = [
                record for record in eligible if record.is_identity_form
            ]
            if identity_records:
                eligible = identity_records
            else:
                anhydrous = [
                    record for record in eligible
                    if record.material_form == 'anhydrous'
                ]
                if anhydrous:
                    eligible = anhydrous
            defaults = [record for record in eligible if record.is_source_default]
            if defaults:
                eligible = defaults
            anchor = max(eligible, key=lambda record: record.quality)
            cluster = [
                record for record in eligible
                if (
                    record.material_form == anchor.material_form
                    and record.polymorph.lower() == anchor.polymorph.lower()
                    and record.stereochemistry.lower() == anchor.stereochemistry.lower()
                    and abs(record.temperature_K - anchor.temperature_K)
                    <= self._fusion_temperature_tolerance(anchor.temperature_K)
                )
            ] or [anchor]
            return PropertyResolutionResult(
                value=median(record.temperature_K for record in cluster),
                source=anchor.source,
                method=anchor.method,
                quality=min(record.quality for record in cluster),
                notes=(
                    f'units K; selected {len(cluster)} compatible structured '
                    f'melting record(s); form='
                    f'{anchor.form_label or anchor.material_form}'
                ),
            )


        def _resolve_melting_point_uncached(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> PropertyResolutionResult:
            """Resolve Tm without consulting the phase-point result cache."""
            provided = self._provided_scalar(props, 'Tm', 'K')
            if provided and self._is_pfd_component_override(props, 'Tm'):
                return provided

            identity_text = ' '.join(str(value or '') for value in (
                symbol, props.get('name'), props.get('formula'),
            ))
            structured_tm = self._provided_melting_transition_result(
                props,
                identity_text,
            )

            hydrated_tm = self._source_result_for_value(props, 'Tm', units='K')
            perry_result = self._perry_melting_result(symbol, props)

            coolprop_tm, coolprop_domain_rejection = (
                self._coolprop_melting_point(symbol, props)
            )
            validation_tm = next(
                (
                    candidate
                    for candidate in (
                        provided, structured_tm, hydrated_tm, perry_result,
                    )
                    if (
                        candidate is not None
                        and not self._result_is_soft(candidate)
                        and not self._phase_result_is_coolprop(candidate)
                    )
                ),
                None,
            )
            coolprop_rejection = self._coolprop_melting_point_rejection(
                coolprop_tm,
                validation_tm,
            )
            if coolprop_rejection is None:
                coolprop_rejection = coolprop_domain_rejection
            if coolprop_rejection is not None:
                coolprop_tm = None
            if coolprop_tm:
                if validation_tm is not None:
                    coolprop_tm = PropertyResolutionResult(
                        value=coolprop_tm.value,
                        source=coolprop_tm.source,
                        method=coolprop_tm.method,
                        quality=COOLPROP_PROPERTY_QUALITY,
                        notes='; '.join(filter(None, (
                            coolprop_tm.notes,
                            f'Corroborated by independent '
                            f'{validation_tm.source}/{validation_tm.method} '
                            f'Tm={float(validation_tm.value):g} K',
                        ))),
                    )
                return coolprop_tm

            if provided:
                return self._melting_result_with_rejection_note(
                    provided,
                    coolprop_rejection,
                )

            if structured_tm:
                return self._melting_result_with_rejection_note(
                    structured_tm,
                    coolprop_rejection,
                )

            if hydrated_tm and hydrated_tm.method == 'curated_tm_override':
                return self._melting_result_with_rejection_note(
                    hydrated_tm,
                    coolprop_rejection,
                )

            if perry_result:
                return self._melting_result_with_rejection_note(
                    perry_result,
                    coolprop_rejection,
                )
            if hydrated_tm:
                return self._melting_result_with_rejection_note(
                    hydrated_tm,
                    coolprop_rejection,
                )
            if allow_online:
                try:
                    online = self._fetch_phase_change_online(symbol, props)
                except LookupError:
                    online = None
                if online and online.get('Tm') is not None:
                    method = online.get('_sources', {}).get('Tm', 'online_phase_change')
                    return PropertyResolutionResult(
                        value=online['Tm'],
                        source='online',
                        method=method,
                        quality=float(
                            online.get('_qualities', {}).get('Tm', 0.88)
                        ),
                        notes=online.get('_notes', {}).get('Tm', 'units K'),
                    )
            if coolprop_rejection is not None:
                return PropertyResolutionResult(
                    value=None,
                    source='missing',
                    method='coolprop_melting_point_outside_validity',
                    quality=0.0,
                    notes=coolprop_rejection,
                )
            return self._missing_scalar('Tm')


        @staticmethod
        def _phase_result_is_coolprop(
            result: Optional[PropertyResolutionResult],
        ) -> bool:
            if result is None:
                return False
            return (
                str(result.source or '').strip().lower() == 'coolprop'
                or str(result.method or '').strip().lower().startswith(
                    'coolprop_'
                )
            )


        def _coolprop_melting_point_rejection(
            self,
            coolprop: Optional[PropertyResolutionResult],
            validator: Optional[PropertyResolutionResult],
        ) -> Optional[str]:
            if (
                coolprop is None
                or validator is None
                or coolprop.value is None
                or validator.value is None
                or self._result_is_soft(validator)
            ):
                return None
            try:
                coolprop_temperature = float(coolprop.value)
                validation_temperature = float(validator.value)
            except (TypeError, ValueError):
                return None
            if (
                not math.isfinite(coolprop_temperature)
                or coolprop_temperature <= 0.0
                or not math.isfinite(validation_temperature)
                or validation_temperature <= 0.0
            ):
                return None
            tolerance = max(
                self.PHASE_TEMPERATURE_ABS_TOL_K,
                self.PHASE_TEMPERATURE_REL_TOL * validation_temperature,
            )
            difference = abs(coolprop_temperature - validation_temperature)
            if difference <= tolerance:
                return None
            return (
                f'CoolProp fusion temperature {coolprop_temperature:g} K '
                f'differs from hard {validator.method} Tm='
                f'{validation_temperature:g} K by {difference:g} K, exceeding '
                f'the {tolerance:g} K consistency tolerance'
            )


        @staticmethod
        def _melting_result_with_rejection_note(
            result: PropertyResolutionResult,
            rejection: Optional[str],
        ) -> PropertyResolutionResult:
            if rejection is None:
                return result
            return PropertyResolutionResult(
                value=result.value,
                source=result.source,
                method=result.method,
                quality=result.quality,
                notes='; '.join(note for note in (result.notes, rejection) if note),
            )


        def _coolprop_melting_point(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> tuple[Optional[PropertyResolutionResult], Optional[str]]:
            reference = self._coolprop_phase_reference(symbol, props)
            if reference is None:
                return None, None
            domain = coolprop_melting_line_domain(reference)
            if domain is None:
                return None, None
            normal_pressure_pa = NORMAL_BOILING_PRESSURE_BAR * 100000.0
            triple_pressure_pa = coolprop_props_si('ptriple', reference)
            fusion_pressure_pa = normal_pressure_pa
            method = 'coolprop_HEOS_melting_point'
            pressure_note = 'at 1 atm'
            if (
                triple_pressure_pa is not None
                and triple_pressure_pa > normal_pressure_pa
            ):
                fusion_pressure_pa = triple_pressure_pa
                method = 'coolprop_HEOS_fusion_point_at_triple_pressure'
                pressure_note = (
                    f'at the triple pressure {triple_pressure_pa / 100000.0:g} bar; '
                    f'no stable fusion equilibrium exists at 1 atm'
                )
            if not domain.contains_pressure(fusion_pressure_pa):
                return None, (
                    f'CoolProp fusion-line pressure '
                    f'{fusion_pressure_pa / 100000.0:g} bar is outside the '
                    f'declared domain {domain.P_min_pa / 100000.0:g}-'
                    f'{domain.P_max_pa / 100000.0:g} bar for '
                    f'{reference.fluid}; extrapolation rejected'
                )
            temperature = coolprop_melting_temperature(
                reference,
                fusion_pressure_pa,
            )
            if temperature is None:
                return None, None
            return PropertyResolutionResult(
                value=temperature,
                source='local',
                method=method,
                quality=self.COOLPROP_UNCORROBORATED_MELTING_QUALITY,
                notes=(
                    f'CoolProp HEOS pure-fluid fusion-line temperature '
                    f'for {reference.fluid} {pressure_note}; units K; '
                    f'declared pressure domain '
                    f'{domain.P_min_pa / 100000.0:g}-'
                    f'{domain.P_max_pa / 100000.0:g} bar'
                ),
            ), None


        def resolve_hvap(
            self,
            symbol: str,
            props: Dict[str, Any] = None,
            T: Optional[float] = None,
            allow_online: bool = True,
            allow_estimation: bool = True,
        ) -> PropertyResolutionResult:
            """Resolve heat of vaporization in kJ/mol, at T when supplied."""
            props = self._coerce_props(symbol, props, allow_online=allow_online)
            T_hvap = float(T) if T is not None else props.get('Tb')

            if T_hvap:
                provided_fit = self._evaluate_provided_correlation(props, 'Hvap', float(T_hvap))
                if provided_fit:
                    value, correlation = provided_fit
                    if value > 0:
                        return self._provided_correlation_result(
                            value,
                            correlation,
                            'provided_hvap_fit',
                            'heat of vaporization in kJ/mol',
                            default_quality=HVAP_PROVIDED_QUALITY,
                        )

            pfd_hvap = (
                self._provided_hvap_scalar(props)
                if self._is_pfd_component_override(props, 'Hvap')
                else None
            )
            if pfd_hvap:
                if T is None:
                    return pfd_hvap
                Tb_result = self._source_result_for_value(props, 'Tb', units='K')
                Tc_result = self._source_result_for_value(props, 'Tc', units='K')
                Tb = Tb_result.value if Tb_result else props.get('Tb')
                Tc = Tc_result.value if Tc_result else props.get('Tc')
                value = self._watson_hvap_value(float(pfd_hvap.value), float(Tb), float(T), float(Tc)) if Tb and Tc else None
                if value is not None:
                    reduced_temperature = float(T) / float(Tc)
                    return PropertyResolutionResult(
                        value=value,
                        source='calculated',
                        method='watson_hvap',
                        quality=temperature_scaled_hvap_quality(
                            HVAP_WATSON_QUALITY_FACTOR,
                            [
                                self._result_quality(item, 0.0)
                                for item in (pfd_hvap, Tb_result)
                                if item is not None
                            ],
                            tc_quality=self._result_quality(Tc_result, 0.0),
                            reduced_temperature=reduced_temperature,
                        ),
                        notes=(
                            f"Watson scaling from {pfd_hvap.source}/{pfd_hvap.method} "
                            f"at Tb={float(Tb):g} K using Tc={float(Tc):g} K"
                        ),
                    )
                return pfd_hvap

            perry_hvap = self._get_perry_evaluation(
                symbol,
                props,
                'heat_of_vaporization_kJ_per_mol',
                T,
            )
            if perry_hvap:
                return PropertyResolutionResult(
                    value=perry_hvap.value,
                    source='local',
                    method=perry_hvap.method,
                    quality=HVAP_PERRY_QUALITY,
                    notes=f"{perry_hvap.source}; units {perry_hvap.units}",
                )

            online = None
            if allow_online:
                try:
                    online = self._fetch_phase_change_online(symbol, props)
                except LookupError:
                    online = None
                # An online scalar Hvap must not preempt the local curated
                # scalar or Watson scaling; it is deferred to the last-resort
                # fallback below. A genuine T-dependent online fit is still
                # preferred here.
                fit = online.get('Hvap_fit') if online else None
                if isinstance(fit, HvapTemperatureFit):
                    fit_T = T_hvap or online.get('Tb') or props.get('Tb')
                    if fit_T:
                        fit_value = fit.value_at(float(fit_T))
                        if fit_value is not None and fit_value > 0.0:
                            quality, range_note = self._hvap_fit_quality(fit, float(fit_T))
                            fit_reduced_temperature = float(fit_T) / float(fit.Tc)
                            if (
                                fit_reduced_temperature
                                > HVAP_TC_SENSITIVITY_REDUCED_TEMPERATURE
                            ):
                                fit_tc_result = self._hvap_tc_result(
                                    props,
                                    online,
                                    prefer_online=True,
                                )
                                if fit_tc_result is not None:
                                    quality = min(
                                        quality,
                                        hvap_effective_tc_quality(
                                            self._result_quality(fit_tc_result, 0.0),
                                            fit_reduced_temperature,
                                        ),
                                    )
                            return PropertyResolutionResult(
                                value=fit_value,
                                source='online',
                                method='nist_hvap_watson_fit',
                                quality=quality,
                                notes=(
                                    f"fit range {fit.T_min:g}-{fit.T_max:g} K; "
                                    f"MAPE {fit.mape_percent:.2f}%; "
                                    f"kept {fit.kept_points}/{fit.total_points} points; "
                                    f"{range_note}"
                                ),
                            )

            watson = self._resolve_hvap_watson(symbol, props, T, online, allow_online)
            if watson:
                return watson

            provided = self._provided_hvap_scalar(props)
            if provided:
                return provided

            textbook = self._textbook_hvap_result(symbol, props)
            if textbook:
                return textbook

            if allow_online:
                online_scalar = self._online_hvap_scalar_result(symbol, online)
                if online_scalar:
                    return online_scalar

            if allow_estimation:
                nannoolal = self._resolve_nannoolal_hvap(
                    symbol,
                    props,
                    T_hvap,
                    allow_online=allow_online,
                )
                if nannoolal is not None:
                    return nannoolal
                corresponding_states = self._resolve_corresponding_states_hvap(
                    symbol,
                    props,
                    T_hvap,
                    allow_online=allow_online,
                )
                if corresponding_states is not None:
                    return corresponding_states
                trouton = self._resolve_trouton_hvap(
                    symbol,
                    props,
                    T,
                    allow_online=allow_online,
                )
                if trouton is not None:
                    return trouton
            return self._missing_scalar('Hvap')


        def _perry_fusion_dataset_fingerprint(self) -> str:
            library = self._get_perry_library()
            path = getattr(library, 'heat_of_fusion_path', None)
            if path is None:
                return 'unavailable'
            path = Path(path)
            try:
                stat = path.stat()
                signature = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            except OSError:
                return 'unavailable'
            cached = getattr(self, '_perry_fusion_fingerprint_cache', None)
            if cached and cached[0] == signature:
                return cached[1]
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                return 'unavailable'
            self._perry_fusion_fingerprint_cache = (signature, digest)
            return digest


        def _fusion_transition_input_metadata(
            self,
            symbol: str,
            props: Mapping[str, Any],
            *,
            allow_online: bool,
        ) -> Dict[str, Any]:
            relevant_fields = (
                'CAS', 'cas', 'name', 'symbol', 'formula', 'Formula',
                'source', 'MW', 'Tm', 'Hfus',
                'fusion_transitions', 'melting_transitions',
            )
            properties = {
                field_name: props.get(field_name)
                for field_name in relevant_fields
                if field_name in props
            }
            property_sources = props.get('property_sources')
            if isinstance(property_sources, Mapping):
                properties['property_sources'] = {
                    str(field_name): metadata
                    for field_name, metadata in property_sources.items()
                    if str(field_name) in {'Tm', 'Hfus', 'fusion_transitions'}
                }
            return {
                'resolver_contract': 'resolved_fusion_transitions_v1',
                'symbol_argument': str(symbol),
                'allow_online': bool(allow_online),
                'online_phase_contract_version': int(
                    self.ONLINE_PHASE_CHANGE_CACHE_VERSION
                ),
                'properties': properties,
                'perry_heat_of_fusion_sha256': (
                    self._perry_fusion_dataset_fingerprint()
                ),
                'units': {
                    'Hfus': 'kJ/mol',
                    'transition_temperature': 'K',
                },
                'bare_selection_policy': [
                    'identity hydrate/solvate when the requested identity is that form',
                    'otherwise exclude hydrate and solvate records',
                    'match selected Tm when available',
                    'prefer anhydrous or explicit source-default ordinary form',
                    'median same-form same-transition replicate measurements',
                ],
                'provider_order': [
                    'PFD/provided transition records',
                    'Perry Table 2-68',
                    'NIST structured fusion rows',
                    'PubChem reported fusion rows',
                ],
            }


        def _fusion_transition_input_fingerprint(
            self,
            symbol: str,
            props: Mapping[str, Any],
            *,
            allow_online: bool,
        ) -> str:
            payload = self._phase_point_cache_json(
                self._fusion_transition_input_metadata(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
            )
            return hashlib.sha256(payload.encode('utf-8')).hexdigest()


        @staticmethod
        def _copy_fusion_records(
            records: tuple[FusionTransitionRecord, ...],
        ) -> tuple[FusionTransitionRecord, ...]:
            return tuple(
                FusionTransitionRecord(**record.to_dict())
                for record in records
            )


        def _load_persistent_fusion_transitions(
            self,
            *,
            component_key: str,
            fingerprint: str,
        ) -> Optional[tuple[FusionTransitionRecord, ...]]:
            path = Path(self.SATURATION_PROPERTIES_CACHE_PATH)
            if not path.exists():
                return None
            try:
                with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                    connection.row_factory = sqlite3.Row
                    self._ensure_phase_point_cache_schema(connection)
                    row = connection.execute(
                        """
                        SELECT * FROM resolved_fusion_transition_cache
                        WHERE cache_version = ?
                          AND online_phase_contract_version = ?
                          AND component_key = ?
                          AND input_fingerprint = ?
                        """,
                        (
                            int(self.FUSION_TRANSITION_CACHE_VERSION),
                            int(self.ONLINE_PHASE_CHANGE_CACHE_VERSION),
                            component_key,
                            fingerprint,
                        ),
                    ).fetchone()
                if row is None or not self._online_attempt_is_persistable(
                    bool(row['allow_online']),
                    str(row['online_attempt_state']),
                ):
                    return None
                if not runtime_cache_row_is_fresh(
                    path,
                    row['updated_at_utc'],
                ):
                    return None
                payload = json.loads(row['records_json'])
                metadata = json.loads(row['input_metadata_json'])
                if not isinstance(payload, list) or not isinstance(metadata, dict):
                    return None
                return tuple(
                    FusionTransitionRecord(**record)
                    for record in payload
                    if isinstance(record, dict)
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                OSError,
                sqlite3.Error,
            ):
                return None


        def _store_persistent_fusion_transitions(
            self,
            symbol: str,
            props: Mapping[str, Any],
            records: tuple[FusionTransitionRecord, ...],
            *,
            allow_online: bool,
            online_attempt_state: OnlineAttemptState | str,
            component_key: str,
            fingerprint: str,
        ) -> None:
            checked_key, cas, component_name = self._phase_point_component_identity(
                symbol, props,
            )
            if checked_key != component_key:
                return
            selected_records = self._select_bare_fusion_records(
                records,
                props.get('Tm'),
            )
            selected_result = self._fusion_result_from_selected_records(
                selected_records,
            )
            timestamp = datetime.now(timezone.utc).isoformat(
                timespec='milliseconds'
            ).replace('+00:00', 'Z')
            records_json = self._phase_point_cache_json([
                record.to_dict() for record in records
            ])
            selected_records_json = self._phase_point_cache_json([
                record.to_dict() for record in selected_records
            ])
            metadata_json = self._phase_point_cache_json(
                self._fusion_transition_input_metadata(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
            )
            path = Path(self.SATURATION_PROPERTIES_CACHE_PATH)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                    self._ensure_phase_point_cache_schema(connection)
                    connection.execute(
                        """
                        INSERT INTO resolved_fusion_transition_cache (
                            cache_version,
                            online_phase_contract_version,
                            component_key,
                            input_fingerprint,
                            symbol,
                            cas,
                            component_name,
                            allow_online,
                            online_attempt_state,
                            selected_Hfus_value,
                            selected_Hfus_source,
                            selected_Hfus_method,
                            selected_Hfus_quality,
                            selected_Hfus_notes,
                            records_json,
                            selected_records_json,
                            input_metadata_json,
                            created_at_utc,
                            updated_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (
                            cache_version,
                            online_phase_contract_version,
                            component_key,
                            input_fingerprint
                        ) DO UPDATE SET
                            symbol = excluded.symbol,
                            cas = excluded.cas,
                            component_name = excluded.component_name,
                            allow_online = excluded.allow_online,
                            online_attempt_state = excluded.online_attempt_state,
                            selected_Hfus_value = excluded.selected_Hfus_value,
                            selected_Hfus_source = excluded.selected_Hfus_source,
                            selected_Hfus_method = excluded.selected_Hfus_method,
                            selected_Hfus_quality = excluded.selected_Hfus_quality,
                            selected_Hfus_notes = excluded.selected_Hfus_notes,
                            records_json = excluded.records_json,
                            selected_records_json = excluded.selected_records_json,
                            input_metadata_json = excluded.input_metadata_json,
                            updated_at_utc = excluded.updated_at_utc
                        """,
                        (
                            int(self.FUSION_TRANSITION_CACHE_VERSION),
                            int(self.ONLINE_PHASE_CHANGE_CACHE_VERSION),
                            component_key,
                            fingerprint,
                            str(symbol),
                            cas,
                            component_name,
                            int(bool(allow_online)),
                            OnlineAttemptState(online_attempt_state).value,
                            selected_result.value,
                            str(selected_result.source),
                            str(selected_result.method),
                            float(selected_result.quality),
                            str(selected_result.notes or ''),
                            records_json,
                            selected_records_json,
                            metadata_json,
                            timestamp,
                            timestamp,
                        ),
                    )
                    connection.commit()
            except (OSError, sqlite3.Error, TypeError, ValueError):
                return


        def _cached_fusion_transitions(
            self,
            symbol: str,
            props: Mapping[str, Any],
            *,
            allow_online: bool,
            build: Callable[[], tuple[FusionTransitionRecord, ...]],
        ) -> tuple[FusionTransitionRecord, ...]:
            if self._fusion_transitions_have_pfd_override(props):
                return tuple(build())
            component_key, _cas, _name = self._phase_point_component_identity(
                symbol, props,
            )
            fingerprint = self._fusion_transition_input_fingerprint(
                symbol,
                props,
                allow_online=allow_online,
            )
            cache_key = (
                int(self.FUSION_TRANSITION_CACHE_VERSION),
                int(self.ONLINE_PHASE_CHANGE_CACHE_VERSION),
                component_key,
                fingerprint,
            )
            memory = getattr(self, '_resolved_fusion_transition_cache', None)
            if memory is None:
                memory = {}
                self._resolved_fusion_transition_cache = memory
            cached = memory.get(cache_key)
            if cached is not None:
                return self._copy_fusion_records(cached)
            persistent = self._load_persistent_fusion_transitions(
                component_key=component_key,
                fingerprint=fingerprint,
            )
            if persistent is not None:
                memory[cache_key] = persistent
                return self._copy_fusion_records(persistent)

            with self._online_attempt_scope(allow_online) as online_attempt:
                records = tuple(build())
            if not allow_online:
                online_attempt.record(OnlineAttemptState.NOT_NEEDED)
            if self._online_attempt_is_persistable(
                allow_online,
                online_attempt.state,
            ):
                self._store_persistent_fusion_transitions(
                    symbol,
                    props,
                    records,
                    allow_online=allow_online,
                    online_attempt_state=online_attempt.state,
                    component_key=component_key,
                    fingerprint=fingerprint,
                )
                memory[cache_key] = self._copy_fusion_records(records)
            return records


        def _fusion_transitions_have_pfd_override(
            self,
            props: Mapping[str, Any],
        ) -> bool:
            if (
                self._is_pfd_component_override(dict(props), 'Hfus')
                or self._is_pfd_component_override(dict(props), 'Tm')
                or self._is_pfd_component_override(
                    dict(props), 'fusion_transitions',
                )
                or self._is_pfd_component_override(
                    dict(props), 'melting_transitions',
                )
            ):
                return True
            for collection_name in (
                'fusion_transitions',
                'melting_transitions',
            ):
                for record in props.get(collection_name) or []:
                    if not isinstance(record, Mapping):
                        continue
                    method = str(record.get('method') or '').strip().lower()
                    source = str(record.get('source') or '').strip().lower()
                    if (
                        method == 'pfd_component_override'
                        or source == 'pfd'
                        or bool(record.get('_pfd_override'))
                    ):
                        return True
            return False


        @classmethod
        def _fusion_record_from_mapping(
            cls,
            record: Mapping[str, Any],
            *,
            identity_text: str = '',
            default_source: str = 'unknown',
            default_method: str = 'unknown',
            default_quality: float = 1.0,
        ) -> Optional[FusionTransitionRecord]:
            if isinstance(record, FusionTransitionRecord):
                return record
            value = (
                record.get('enthalpy_kJ_mol')
                if record.get('enthalpy_kJ_mol') is not None
                else record.get('Hfus_kJ_per_mol')
            )
            if value is None:
                value = record.get('Hfus')
            if value is None:
                value = record.get('value')
            temperature = (
                record.get('temperature_K')
                if record.get('temperature_K') is not None
                else record.get('Tm_K')
            )
            if temperature is None:
                temperature = record.get('Tfus')
            descriptive_text = ' '.join(str(record.get(key) or '') for key in (
                'form_label', 'table_name', 'name', 'comment',
            ))
            inferred = classify_fusion_material_form(descriptive_text, identity_text)
            material_form = str(record.get('material_form') or inferred[0])
            form_label = str(record.get('form_label') or inferred[1])
            polymorph = str(record.get('polymorph') or inferred[2])
            stereochemistry = str(record.get('stereochemistry') or inferred[3])
            if record.get('suppress_polymorph'):
                polymorph = ''
                if not record.get('form_label'):
                    labels = []
                    if material_form != 'unspecified':
                        labels.append(material_form)
                    if stereochemistry:
                        labels.append(stereochemistry)
                    form_label = '/'.join(labels)
            identity_form = bool(record.get('is_identity_form', inferred[4]))
            identity_lower = identity_text.lower()
            if 'is_identity_form' not in record:
                identity_form = identity_form or (
                    material_form == 'hydrate' and 'hydrate' in identity_lower
                ) or (
                    material_form == 'solvate' and 'solvate' in identity_lower
                )
            try:
                return FusionTransitionRecord(
                    enthalpy_kJ_mol=float(value),
                    temperature_K=(
                        float(temperature) if temperature is not None else None
                    ),
                    material_form=material_form,
                    form_label=form_label,
                    polymorph=polymorph,
                    stereochemistry=stereochemistry,
                    is_identity_form=identity_form,
                    is_source_default=bool(record.get('is_source_default', False)),
                    source=str(record.get('source') or default_source),
                    method=str(record.get('method') or default_method),
                    quality=float(record.get('quality', default_quality)),
                    reference=str(record.get('reference') or ''),
                    comment=str(record.get('comment') or ''),
                    raw=str(record.get('raw') or record.get('source_line') or ''),
                    metadata=dict(record.get('metadata') or {}),
                )
            except (TypeError, ValueError):
                return None


        @classmethod
        def _melting_record_from_mapping(
            cls,
            record: Mapping[str, Any] | MeltingTransitionRecord,
            *,
            identity_text: str = '',
            default_source: str = 'provided',
            default_method: str = 'provided_melting_transition',
            default_quality: float = 1.0,
        ) -> Optional[MeltingTransitionRecord]:
            if isinstance(record, MeltingTransitionRecord):
                return record
            if not isinstance(record, Mapping):
                return None
            temperature = record.get('temperature_K')
            if temperature is None:
                temperature = record.get('Tm_K')
            if temperature is None:
                temperature = record.get('Tm')
            descriptive = ' '.join(str(record.get(key) or '') for key in (
                'form_label', 'name', 'comment', 'raw',
            ))
            form = classify_fusion_material_form(descriptive, identity_text)
            material_form = str(record.get('material_form') or form[0])
            identity_form = bool(record.get('is_identity_form', form[4]))
            if 'is_identity_form' not in record:
                identity_lower = identity_text.lower()
                identity_form = identity_form or (
                    material_form == 'hydrate' and 'hydrate' in identity_lower
                ) or (
                    material_form == 'solvate' and 'solvate' in identity_lower
                )
            try:
                return MeltingTransitionRecord(
                    temperature_K=float(temperature),
                    enthalpy_kJ_mol=record.get('enthalpy_kJ_mol'),
                    material_form=material_form,
                    form_label=str(record.get('form_label') or form[1]),
                    polymorph=str(record.get('polymorph') or form[2]),
                    stereochemistry=str(record.get('stereochemistry') or form[3]),
                    is_identity_form=identity_form,
                    is_source_default=bool(record.get('is_source_default', False)),
                    source=str(record.get('source') or default_source),
                    method=str(record.get('method') or default_method),
                    quality=float(record.get('quality', default_quality)),
                    reference=str(record.get('reference') or ''),
                    comment=str(record.get('comment') or ''),
                    raw=str(record.get('raw') or ''),
                    metadata=dict(record.get('metadata') or {}),
                )
            except (TypeError, ValueError):
                return None


        def _provided_fusion_transition_records(
            self,
            props: Mapping[str, Any],
            identity_text: str,
        ) -> list[FusionTransitionRecord]:
            records = []
            for raw in props.get('fusion_transitions') or []:
                if not isinstance(raw, (Mapping, FusionTransitionRecord)):
                    continue
                normalized = self._fusion_record_from_mapping(
                    raw,
                    identity_text=identity_text,
                    default_source='provided',
                    default_method='provided_fusion_transition',
                    default_quality=1.0,
                )
                if normalized is not None:
                    records.append(normalized)

            provided = self._provided_scalar(dict(props), 'Hfus', 'kJ/mol')
            if provided is not None:
                if self._fusion_scalar_is_resolver_owned(provided):
                    return records
                already_represented = any(
                    math.isclose(
                        record.enthalpy_kJ_mol,
                        float(provided.value),
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-12,
                    )
                    for record in records
                )
                if not already_represented:
                    tm = props.get('Tm')
                    records.append(FusionTransitionRecord(
                        enthalpy_kJ_mol=float(provided.value),
                        temperature_K=float(tm) if tm is not None else None,
                        material_form='unspecified',
                        is_source_default=True,
                        source=provided.source,
                        method=provided.method,
                        quality=provided.quality,
                        comment=provided.notes,
                    ))
            return records


        @staticmethod
        def _fusion_scalar_is_resolver_owned(
            result: PropertyResolutionResult,
        ) -> bool:
            source = str(result.source or '').strip().lower()
            method = str(result.method or '').strip().lower()
            return (
                source in {
                    'online', 'nist', 'nist_phase_change', 'pubchem',
                    'perry 9th',
                }
                or 'perry_heat_of_fusion' in method
                or method == 'perry_local_fetch'
                or method.startswith('nist_')
                or method.startswith('pubchem_')
            )


        def _perry_fusion_transition_records(
            self,
            symbol: str,
            props: Mapping[str, Any],
            identity_text: str,
        ) -> list[FusionTransitionRecord]:
            library = self._get_perry_library()
            if library is None:
                return []
            for candidate in self._identifier_candidates(symbol, dict(props)):
                evaluations = library.heat_of_fusion_records(candidate)
                if not evaluations:
                    continue
                records = []
                for index, evaluation in enumerate(evaluations):
                    row = evaluation.correlation
                    qualifiers = [
                        str(value) for value in row.get('source_name_qualifiers', [])
                    ]
                    resolved_query = str(row.get('resolved_query') or '').lower()
                    identity_polymorphs = {
                        label
                        for label in ('alpha', 'beta', 'gamma', 'delta')
                        if re.search(rf'\b{label}\b', resolved_query)
                    }
                    actual_qualifiers = {
                        value.lower().rstrip('-')
                        for value in qualifiers
                    } - identity_polymorphs
                    mapping = {
                        'enthalpy_kJ_mol': evaluation.value,
                        'temperature_K': row.get('Tm_K'),
                        'table_name': row.get('table_name'),
                        'comment': '; '.join(qualifiers),
                        'raw': row.get('source_line', ''),
                        'source': 'local',
                        'method': evaluation.method,
                        'quality': 0.96,
                        'reference': evaluation.source,
                        'suppress_polymorph': bool(identity_polymorphs),
                        'is_source_default': (
                            index == 0
                            and not any(
                                token in {
                                    'alpha', 'beta', 'gamma', 'delta',
                                    'hydrate', 'solvate',
                                }
                                for token in actual_qualifiers
                            )
                        ),
                        'metadata': {
                            'cas': row.get('cas'),
                            'formula': row.get('formula'),
                            'table_name': row.get('table_name'),
                            'source_name_qualifiers': qualifiers,
                            'source_page': row.get('source_page'),
                            'source_line_number': row.get('source_line_number'),
                            'resolution_source': row.get('resolution_source'),
                            'identity_form_qualifiers': sorted(identity_polymorphs),
                        },
                    }
                    normalized = self._fusion_record_from_mapping(
                        mapping,
                        identity_text=identity_text,
                    )
                    if normalized is not None:
                        records.append(normalized)
                return records
            return []


        @staticmethod
        def _fusion_record_key(record: FusionTransitionRecord) -> tuple:
            return (
                round(record.enthalpy_kJ_mol, 10),
                None if record.temperature_K is None else round(record.temperature_K, 8),
                record.material_form,
                record.form_label.lower(),
                record.source.lower(),
                record.method.lower(),
                record.reference.lower(),
            )


        def _resolve_fusion_transitions_uncached(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> tuple[FusionTransitionRecord, ...]:
            """Build all known fusion records without flattening forms."""
            identity_text = ' '.join(str(value or '') for value in (
                symbol, props.get('name'), props.get('formula'),
            ))
            records = self._provided_fusion_transition_records(props, identity_text)
            records.extend(self._perry_fusion_transition_records(
                symbol, props, identity_text,
            ))
            if allow_online:
                try:
                    online = self._fetch_phase_change_online(symbol, props)
                except LookupError:
                    online = None
                if online:
                    raw_records = list(online.get('Hfus_records') or [])
                    if not raw_records and online.get('Hfus') is not None:
                        raw_records.append({
                            'enthalpy_kJ_mol': online['Hfus'],
                            'temperature_K': online.get('Tm'),
                            'source': 'online',
                            'method': (online.get('_sources') or {}).get(
                                'Hfus', 'online_phase_change',
                            ),
                            'quality': (online.get('_qualities') or {}).get(
                                'Hfus', 0.90,
                            ),
                            'comment': (online.get('_notes') or {}).get('Hfus', ''),
                        })
                    for raw in raw_records:
                        normalized = self._fusion_record_from_mapping(
                            raw,
                            identity_text=identity_text,
                            default_source='online',
                            default_method='online_phase_change',
                            default_quality=0.90,
                        )
                        if normalized is not None:
                            records.append(normalized)

            unique = {}
            for record in records:
                unique.setdefault(self._fusion_record_key(record), record)
            return tuple(unique.values())


        def resolve_fusion_transitions(
            self,
            symbol: str,
            props: Dict[str, Any] = None,
            allow_online: bool = True,
            material_form: Optional[str] = None,
        ) -> tuple[FusionTransitionRecord, ...]:
            """Return all known fusion records without flattening forms."""
            props = self._coerce_props(symbol, props, allow_online=allow_online)
            resolved = self._cached_fusion_transitions(
                symbol,
                props,
                allow_online=allow_online,
                build=lambda: self._resolve_fusion_transitions_uncached(
                    symbol,
                    props,
                    allow_online=allow_online,
                ),
            )
            if material_form is None:
                return resolved
            requested_form = str(material_form).strip().lower()
            return tuple(
                record for record in resolved
                if record.material_form == requested_form
            )


        @staticmethod
        def _melting_record_key(record: MeltingTransitionRecord) -> tuple:
            return (
                round(record.temperature_K, 8),
                None if record.enthalpy_kJ_mol is None
                else round(record.enthalpy_kJ_mol, 10),
                record.material_form,
                record.form_label.lower(),
                record.source.lower(),
                record.method.lower(),
                record.reference.lower(),
            )


        def resolve_melting_transitions(
            self,
            symbol: str,
            props: Dict[str, Any] = None,
            allow_online: bool = True,
            material_form: Optional[str] = None,
        ) -> tuple[MeltingTransitionRecord, ...]:
            """Return every form-specific melting observation and pairing."""
            props = self._coerce_props(symbol, props, allow_online=allow_online)
            identity_text = ' '.join(str(value or '') for value in (
                symbol, props.get('name'), props.get('formula'),
            ))
            records: list[MeltingTransitionRecord] = []

            for raw in props.get('melting_transitions') or []:
                normalized = self._melting_record_from_mapping(
                    raw,
                    identity_text=identity_text,
                )
                if normalized is not None:
                    records.append(normalized)

            fusion_records = self.resolve_fusion_transitions(
                symbol,
                props,
                allow_online=allow_online,
            )
            for record in fusion_records:
                if record.temperature_K is None:
                    continue
                records.append(MeltingTransitionRecord(
                    temperature_K=record.temperature_K,
                    enthalpy_kJ_mol=record.enthalpy_kJ_mol,
                    material_form=record.material_form,
                    form_label=record.form_label,
                    polymorph=record.polymorph,
                    stereochemistry=record.stereochemistry,
                    is_identity_form=record.is_identity_form,
                    is_source_default=record.is_source_default,
                    source=record.source,
                    method=record.method,
                    quality=record.quality,
                    reference=record.reference,
                    comment=record.comment,
                    raw=record.raw,
                    metadata=record.metadata,
                ))

            if allow_online:
                try:
                    online = self._fetch_phase_change_online(symbol, props)
                except LookupError:
                    online = None
                for candidate in (
                    ((online or {}).get('_phase_candidates') or {}).get('Tm') or []
                ):
                    try:
                        records.append(MeltingTransitionRecord(
                            temperature_K=float(candidate['value_K']),
                            material_form=str(
                                candidate.get('material_form') or 'unspecified'
                            ),
                            form_label=str(candidate.get('form_label') or ''),
                            polymorph=str(candidate.get('polymorph') or ''),
                            stereochemistry=str(
                                candidate.get('stereochemistry') or ''
                            ),
                            is_identity_form=bool(
                                candidate.get('is_identity_form', False)
                            ),
                            source=str(candidate.get('source') or 'online'),
                            method=str(candidate.get('method') or 'online_melting'),
                            quality=0.97 if candidate.get('source') == 'pubchem' else 0.96,
                            reference=str(candidate.get('reference') or ''),
                            comment=str(candidate.get('comment') or ''),
                            raw=str(candidate.get('raw') or ''),
                            metadata={
                                'low_K': candidate.get('low_K'),
                                'high_K': candidate.get('high_K'),
                                'uncertainty_K': candidate.get('uncertainty_K'),
                                'qualifiers': dict(candidate.get('qualifiers') or {}),
                            },
                        ))
                    except (KeyError, TypeError, ValueError):
                        continue

            selected = self.resolve_melting_point(
                symbol,
                props,
                allow_online=allow_online,
            )
            selected_is_represented = (
                selected.value is not None
                and any(
                    math.isclose(
                        record.temperature_K,
                        float(selected.value),
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    )
                    for record in records
                )
            )
            if selected.value is not None and not selected_is_represented:
                records.append(MeltingTransitionRecord(
                    temperature_K=float(selected.value),
                    material_form='unspecified',
                    is_source_default=True,
                    source=selected.source,
                    method=selected.method,
                    quality=selected.quality,
                    comment=selected.notes,
                ))

            unique = {}
            for record in records:
                unique.setdefault(self._melting_record_key(record), record)
            resolved = tuple(unique.values())
            if material_form is None:
                return resolved
            requested_form = str(material_form).strip().lower()
            return tuple(
                record for record in resolved
                if record.material_form == requested_form
            )


        @classmethod
        def _fusion_temperature_tolerance(cls, temperature_K: float) -> float:
            return max(
                cls.FUSION_TRANSITION_CLUSTER_TOLERANCE_K,
                cls.FUSION_TRANSITION_CLUSTER_REL_TOL * abs(float(temperature_K)),
            )


        @classmethod
        def _select_bare_fusion_records(
            cls,
            records: tuple[FusionTransitionRecord, ...],
            target_temperature_K: Optional[float],
        ) -> tuple[FusionTransitionRecord, ...]:
            eligible = [
                record for record in records
                if record.material_form not in {'hydrate', 'solvate'}
                or record.is_identity_form
            ]
            if not eligible:
                return ()

            target = None
            try:
                target = float(target_temperature_K)
                if not math.isfinite(target) or target <= 0.0:
                    target = None
            except (TypeError, ValueError):
                target = None

            if target is not None:
                with_temperature = [
                    record for record in eligible
                    if record.temperature_K is not None
                ]
                if with_temperature:
                    tolerance = cls._fusion_temperature_tolerance(target)
                    near = [
                        record for record in with_temperature
                        if abs(record.temperature_K - target) <= tolerance
                    ]
                    if near:
                        eligible = near
                    else:
                        defaults = [
                            record for record in eligible
                            if record.is_source_default
                        ]
                        unlabeled = [
                            record for record in eligible
                            if not record.polymorph
                        ]
                        if defaults:
                            eligible = defaults
                        elif unlabeled:
                            eligible = unlabeled
                        elif len({
                            record.polymorph.lower()
                            for record in eligible
                            if record.polymorph
                        }) > 1:
                            return ()
            else:
                identity_records = [
                    record for record in eligible
                    if record.is_identity_form
                ]
                if identity_records:
                    eligible = identity_records
                else:
                    anhydrous = [
                        record for record in eligible
                        if record.material_form == 'anhydrous'
                    ]
                    if anhydrous:
                        eligible = anhydrous
                defaults = [record for record in eligible if record.is_source_default]
                if defaults:
                    eligible = defaults
                else:
                    unlabeled = [
                        record for record in eligible
                        if not record.polymorph
                    ]
                    if unlabeled:
                        eligible = unlabeled
                    elif len({
                        record.polymorph.lower()
                        for record in eligible
                        if record.polymorph
                    }) > 1:
                        return ()
                paired = [record for record in eligible if record.temperature_K is not None]
                if paired:
                    eligible = paired

            source_priority = {
                'pfd': 100,
                'provided': 100,
                'user': 100,
                'local': 90,
                'nist': 80,
                'nist_phase_change': 80,
                'pubchem': 70,
                'online': 60,
            }
            best_priority = max(
                source_priority.get(record.source.lower(), 50)
                for record in eligible
            )
            eligible = [
                record for record in eligible
                if source_priority.get(record.source.lower(), 50) == best_priority
            ]
            anchor = max(eligible, key=lambda record: (
                record.quality,
                record.is_identity_form,
                record.is_source_default,
                record.temperature_K is not None,
                -record.enthalpy_kJ_mol,
            ))
            cluster = []
            for record in eligible:
                if (
                    record.material_form != anchor.material_form
                    or record.polymorph.lower() != anchor.polymorph.lower()
                    or record.stereochemistry.lower() != anchor.stereochemistry.lower()
                ):
                    continue
                if anchor.temperature_K is None or record.temperature_K is None:
                    if anchor.temperature_K != record.temperature_K:
                        continue
                elif abs(record.temperature_K - anchor.temperature_K) > cls._fusion_temperature_tolerance(anchor.temperature_K):
                    continue
                cluster.append(record)
            return tuple(cluster or [anchor])


        @staticmethod
        def _fusion_result_from_selected_records(
            selected: tuple[FusionTransitionRecord, ...],
        ) -> PropertyResolutionResult:
            if not selected:
                return PropertyResolutionResult(
                    value=None,
                    source='missing',
                    method='none',
                    quality=0.0,
                    notes='Hfus not available',
                )
            values = [record.enthalpy_kJ_mol for record in selected]
            value = median(values)
            quality = min(record.quality for record in selected)
            anchor = max(selected, key=lambda record: record.quality)
            temperatures = [
                record.temperature_K for record in selected
                if record.temperature_K is not None
            ]
            form = anchor.form_label or anchor.material_form
            notes = (
                f"units kJ/mol; selected {len(selected)} compatible "
                f"fusion record(s); form={form or 'unspecified'}"
            )
            if temperatures:
                notes += (
                    f"; transition temperature range "
                    f"{min(temperatures):g}-{max(temperatures):g} K"
                )
            if len(selected) > 1:
                notes += '; scalar is median of same-transition measurements'
            return PropertyResolutionResult(
                value=value,
                source=anchor.source,
                method=anchor.method,
                quality=quality,
                notes=notes,
            )


        def resolve_hfus(
            self,
            symbol: str,
            props: Dict[str, Any] = None,
            allow_online: bool = True,
        ) -> PropertyResolutionResult:
            """Resolve the bare/default-form heat of fusion in kJ/mol."""
            props = self._coerce_props(symbol, props, allow_online=allow_online)
            provided = self._provided_scalar(props, 'Hfus', 'kJ/mol')
            if provided and not self._fusion_scalar_is_resolver_owned(provided):
                return provided
            target_temperature = props.get('Tm')
            if target_temperature is None:
                melting = self.resolve_melting_point(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
                target_temperature = melting.value
            fusion_props = props
            if target_temperature is not None and props.get('Tm') is None:
                fusion_props = dict(props)
                fusion_props['Tm'] = target_temperature
                fusion_props.setdefault('property_sources', {})
                fusion_props['property_sources'] = dict(
                    fusion_props['property_sources']
                )
                fusion_props['property_sources']['Tm'] = {
                    'source': melting.source,
                    'method': melting.method,
                    'quality': melting.quality,
                    'notes': melting.notes,
                }
            records = self.resolve_fusion_transitions(
                symbol,
                fusion_props,
                allow_online=allow_online,
            )
            selected = self._select_bare_fusion_records(
                records,
                target_temperature,
            )
            return self._fusion_result_from_selected_records(selected)
