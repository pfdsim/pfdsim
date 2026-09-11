"""Online phase-change source arbitration shared by scalar phase resolvers."""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from .common import (
    classify_fusion_material_form,
    HvapTemperatureFit,
    NIST_HVAP_FIT_MINIMUM_QUALITY,
    NIST_HVAP_FIT_QUALITY,
    OnlineAttemptState,
    REFERENCE_TEMPERATURE_K,
)
from .phase_point_candidates import (
    legacy_temperature_candidate,
    nist_temperature_candidate,
    parse_reported_temperature_candidates,
    select_pressure_consensus,
    select_temperature_consensus,
)


class OnlinePhaseChangeMixin:
    """Merge NIST/PubChem phase records before property-specific resolution."""

    ONLINE_PHASE_CHANGE_CACHE_VERSION = 9

    def _fetch_phase_change_online(
        self,
        symbol: str,
        props: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        identifiers = self._identifier_candidates(symbol, props)
        if not identifiers:
            self._record_online_attempt_state(
                OnlineAttemptState.NOT_ATTEMPTED
            )
            return None

        pubchem_payload = None
        nist_payload = None
        transient_failure = False
        for candidate in identifiers:
            try:
                pubchem = self._fetch_phase_change_pubchem(candidate)
            except LookupError:
                transient_failure = True
                continue
            if pubchem:
                pubchem_payload = pubchem
                break

        for candidate in identifiers:
            try:
                nist = self._fetch_phase_change_nist(candidate)
            except LookupError:
                transient_failure = True
                continue
            if nist:
                nist_payload = nist
                break

        merged: Dict[str, Any] = {}
        if pubchem_payload:
            merged.update(pubchem_payload)
            for metadata_key in ('_sources', '_qualities', '_notes'):
                if metadata_key in pubchem_payload:
                    merged[metadata_key] = dict(pubchem_payload[metadata_key])

        if nist_payload:
            metadata_maps = {
                metadata_key: merged.setdefault(metadata_key, {})
                for metadata_key in ('_sources', '_qualities', '_notes')
            }
            nist_metadata = {
                metadata_key: nist_payload.get(metadata_key, {})
                for metadata_key in metadata_maps
            }
            for key, value in nist_payload.items():
                if key.startswith('_'):
                    continue
                if key in {'Hvap_records', 'Hfus_records'}:
                    merged.setdefault(key, []).extend(value or [])
                    continue
                if key in {'Tm', 'Tt', 'Pt'}:
                    continue
                replace = (
                    key in {'Tc', 'Pc', 'Vc'}
                    or key not in merged
                    or merged[key] is None
                )
                if not replace:
                    continue
                merged[key] = value
                metadata_maps['_sources'][key] = nist_metadata['_sources'].get(
                    key,
                    'nist_phase_change',
                )
                if key in nist_metadata['_qualities']:
                    metadata_maps['_qualities'][key] = nist_metadata['_qualities'][key]
                else:
                    metadata_maps['_qualities'].pop(key, None)
                if key in nist_metadata['_notes']:
                    metadata_maps['_notes'][key] = nist_metadata['_notes'][key]
                else:
                    metadata_maps['_notes'].pop(key, None)

        phase_candidates: Dict[str, list[Dict[str, Any]]] = {}
        for payload, source_name in (
            (pubchem_payload, 'pubchem'),
            (nist_payload, 'nist'),
        ):
            if not payload:
                continue
            payload_candidates = payload.get('_phase_candidates') or {}
            for key in ('Tm', 'Tt', 'Pt'):
                candidates = list(payload_candidates.get(key) or [])
                if not candidates and key in payload and payload.get(key) is not None:
                    if key in {'Tm', 'Tt'}:
                        legacy = legacy_temperature_candidate(
                            payload[key],
                            source=source_name,
                            method=(payload.get('_sources') or {}).get(key, source_name),
                            notes=(payload.get('_notes') or {}).get(key, ''),
                        )
                        if legacy is not None:
                            candidates.append(legacy)
                    else:
                        try:
                            value_bar = float(payload[key])
                        except (TypeError, ValueError):
                            value_bar = None
                        if value_bar is not None and value_bar > 0.0:
                            candidates.append({
                                'value_bar': value_bar,
                                'source': source_name,
                                'method': (payload.get('_sources') or {}).get(
                                    key,
                                    source_name,
                                ),
                                'reference': '',
                                'comment': (payload.get('_notes') or {}).get(key, ''),
                                'raw': f'legacy selected value {value_bar:g} bar',
                            })
                if candidates:
                    phase_candidates.setdefault(key, []).extend(candidates)
        if phase_candidates:
            merged['_phase_candidates'] = phase_candidates
            self._finalize_online_phase_point_candidates(merged)

        if (
            merged.get('Pc') is not None
            and props
            and hasattr(self, '_critical_pressure_is_admissible')
            and not self._critical_pressure_is_admissible(props, merged['Pc'])
        ):
            rejected_pressure = merged.pop('Pc')
            for metadata_key in ('_sources', '_qualities', '_notes'):
                metadata = merged.get(metadata_key)
                if isinstance(metadata, dict):
                    metadata.pop('Pc', None)
            merged.setdefault('_rejections', []).append(
                f'Online Pc={rejected_pressure:g} bar exceeds the '
                f'{"organic" if self._critical_formula_is_organic(props) else "inorganic"} '
                'critical-pressure plausibility ceiling'
            )

        if merged:
            self._finalize_online_fusion_records(merged)
            self._finalize_online_hvap(merged)
            if transient_failure:
                merged['_online_attempt_state'] = (
                    OnlineAttemptState.TRANSIENT_FAILURE.value
                )
                self._record_online_attempt_state(
                    OnlineAttemptState.TRANSIENT_FAILURE
                )
            else:
                merged['_online_attempt_state'] = (
                    OnlineAttemptState.COMPLETE_WITH_DATA.value
                )
            self._record_online_attempt_state(
                OnlineAttemptState.COMPLETE_WITH_DATA
            )
            return merged
        if transient_failure:
            self._record_online_attempt_state(
                OnlineAttemptState.TRANSIENT_FAILURE
            )
            raise LookupError(
                f"Transient online phase-change lookup failure for '{symbol}'"
            )
        self._record_online_attempt_state(
            OnlineAttemptState.COMPLETE_NO_DATA
        )
        return None

    def _finalize_online_phase_point_candidates(
        self,
        result: Dict[str, Any],
    ) -> None:
        candidates = result.get('_phase_candidates') or {}
        qualities = result.setdefault('_qualities', {})
        notes = result.setdefault('_notes', {})
        sources = result.setdefault('_sources', {})
        selected_payload = result.setdefault('_selected_phase_candidates', {})
        for key, property_name in (
            ('Tm', 'melting_point'),
            ('Tt', 'triple_temperature'),
        ):
            property_candidates = list(candidates.get(key) or [])
            if key == 'Tm':
                bare_candidates = []
                for candidate in property_candidates:
                    material_form = str(
                        candidate.get('material_form') or 'unspecified'
                    ).lower()
                    if (
                        material_form in {'hydrate', 'solvate'}
                        and not candidate.get('is_identity_form')
                    ):
                        continue
                    bare_candidates.append(candidate)
                identity_candidates = [
                    candidate for candidate in bare_candidates
                    if candidate.get('is_identity_form')
                ]
                if identity_candidates:
                    bare_candidates = identity_candidates
                else:
                    anhydrous_candidates = [
                        candidate for candidate in bare_candidates
                        if str(candidate.get('material_form') or '').lower()
                        == 'anhydrous'
                    ]
                    if anhydrous_candidates:
                        bare_candidates = anhydrous_candidates
                    else:
                        unlabeled_candidates = [
                            candidate for candidate in bare_candidates
                            if not candidate.get('polymorph')
                        ]
                        if unlabeled_candidates:
                            corroborating = []
                            for candidate in bare_candidates:
                                if not candidate.get('polymorph'):
                                    corroborating.append(candidate)
                                    continue
                                if any(
                                    abs(
                                        float(candidate['value_K'])
                                        - float(default['value_K'])
                                    )
                                    <= max(
                                        2.0,
                                        0.01 * abs(float(default['value_K'])),
                                    )
                                    for default in unlabeled_candidates
                                ):
                                    corroborating.append(candidate)
                            bare_candidates = corroborating
                        elif len({
                            str(candidate.get('polymorph') or '').lower()
                            for candidate in bare_candidates
                            if candidate.get('polymorph')
                        }) > 1:
                            bare_candidates = []
                property_candidates = bare_candidates
            selection = select_temperature_consensus(
                property_name,
                property_candidates,
            )
            if selection is None:
                result.pop(key, None)
                qualities.pop(key, None)
                notes.pop(key, None)
                sources.pop(key, None)
                selected_payload.pop(key, None)
                continue
            result[key] = selection['value']
            qualities[key] = selection['quality']
            notes[key] = selection['notes']
            sources[key] = selection['method']
            selected_payload[key] = {
                name: selection[name]
                for name in (
                    'selected_candidates',
                    'broad_candidates',
                    'outlier_candidates',
                )
            }

        pressure_selection = select_pressure_consensus(
            'triple_pressure',
            candidates.get('Pt') or [],
        )
        if pressure_selection is None:
            result.pop('Pt', None)
            qualities.pop('Pt', None)
            notes.pop('Pt', None)
            sources.pop('Pt', None)
            selected_payload.pop('Pt', None)
        else:
            result['Pt'] = pressure_selection['value']
            qualities['Pt'] = pressure_selection['quality']
            notes['Pt'] = pressure_selection['notes']
            sources['Pt'] = pressure_selection['method']
            selected_payload['Pt'] = {
                'selected_candidates': pressure_selection['selected_candidates'],
            }
        if not selected_payload:
            result.pop('_selected_phase_candidates', None)
        if not qualities:
            result.pop('_qualities', None)
        if not notes:
            result.pop('_notes', None)
        if not sources:
            result.pop('_sources', None)

    def _finalize_online_fusion_records(self, result: Dict[str, Any]) -> None:
        """Select a backward-compatible scalar without dropping raw records."""
        raw_records = [
            dict(record) for record in result.get('Hfus_records') or []
            if isinstance(record, dict)
        ]
        unique = {}
        for record in raw_records:
            try:
                value = float(record.get('enthalpy_kJ_mol'))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or not (0.0 < value < 200.0):
                continue
            temperature = record.get('temperature_K')
            try:
                temperature = float(temperature) if temperature is not None else None
            except (TypeError, ValueError):
                temperature = None
            if temperature is not None and (
                not math.isfinite(temperature) or temperature <= 0.0
            ):
                temperature = None
            record['enthalpy_kJ_mol'] = value
            record['temperature_K'] = temperature
            key = (
                round(value, 10),
                None if temperature is None else round(temperature, 8),
                str(record.get('material_form') or 'unspecified').lower(),
                str(record.get('form_label') or '').lower(),
                str(record.get('source') or '').lower(),
                str(record.get('reference') or '').lower(),
                str(record.get('raw') or ''),
            )
            unique.setdefault(key, record)
        records = list(unique.values())
        if not records:
            result.pop('Hfus_records', None)
            return
        result['Hfus_records'] = records

        eligible = [
            record for record in records
            if str(record.get('material_form') or 'unspecified').lower()
            not in {'hydrate', 'solvate'}
            or record.get('is_identity_form')
        ]
        if not eligible:
            result.pop('Hfus', None)
            return
        identity_records = [
            record for record in eligible if record.get('is_identity_form')
        ]
        if identity_records:
            eligible = identity_records
        else:
            anhydrous_records = [
                record for record in eligible
                if str(record.get('material_form') or '').lower() == 'anhydrous'
            ]
            if anhydrous_records:
                eligible = anhydrous_records
        source_priority = {
            'nist': 80,
            'nist_phase_change': 80,
            'pubchem': 70,
        }
        priority = max(
            source_priority.get(str(record.get('source') or '').lower(), 50)
            for record in eligible
        )
        eligible = [
            record for record in eligible
            if source_priority.get(str(record.get('source') or '').lower(), 50)
            == priority
        ]
        target = result.get('Tm')
        try:
            target = float(target)
        except (TypeError, ValueError):
            target = None
        with_temperature = [
            record for record in eligible
            if record.get('temperature_K') is not None
        ]
        matched_target = False
        if target is not None and with_temperature:
            tolerance = max(2.0, 0.01 * abs(target))
            near = [
                record for record in with_temperature
                if abs(record['temperature_K'] - target) <= tolerance
            ]
            if near:
                eligible = near
                matched_target = True
        if not matched_target:
            default_records = [
                record for record in eligible
                if record.get('is_source_default')
            ]
            if default_records:
                eligible = default_records
            else:
                unlabeled_records = [
                    record for record in eligible
                    if not record.get('polymorph')
                ]
                if unlabeled_records:
                    eligible = unlabeled_records
                elif len({
                    str(record.get('polymorph') or '').lower()
                    for record in eligible
                    if record.get('polymorph')
                }) > 1:
                    result.pop('Hfus', None)
                    return
        anchor = max(eligible, key=lambda record: float(record.get('quality', 0.0)))
        cluster = []
        for record in eligible:
            if (
                str(record.get('material_form') or 'unspecified').lower()
                != str(anchor.get('material_form') or 'unspecified').lower()
                or str(record.get('polymorph') or '').lower()
                != str(anchor.get('polymorph') or '').lower()
                or str(record.get('stereochemistry') or '').lower()
                != str(anchor.get('stereochemistry') or '').lower()
            ):
                continue
            anchor_temperature = anchor.get('temperature_K')
            record_temperature = record.get('temperature_K')
            if anchor_temperature is None or record_temperature is None:
                if anchor_temperature != record_temperature:
                    continue
            elif abs(record_temperature - anchor_temperature) > max(
                2.0, 0.01 * abs(anchor_temperature),
            ):
                continue
            cluster.append(record)
        cluster = cluster or [anchor]
        result['Hfus'] = self._median([
            record['enthalpy_kJ_mol'] for record in cluster
        ])
        result.setdefault('_sources', {})['Hfus'] = str(
            anchor.get('method') or anchor.get('source') or 'online_phase_change'
        )
        result.setdefault('_qualities', {})['Hfus'] = min(
            float(record.get('quality', 0.90)) for record in cluster
        )
        result.setdefault('_notes', {})['Hfus'] = (
            f"Selected {len(cluster)} compatible fusion record(s); "
            "scalar is a same-transition median"
        )

    def _fetch_phase_change_pubchem(self, identifier: str) -> Optional[Dict[str, Any]]:
        cache_key = (
            f"phase_pubchem_v{self.ONLINE_PHASE_CHANGE_CACHE_VERSION}_{identifier}"
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
            url = (
                'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/'
                f'compound/{cid}/JSON'
            )
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'PFD-Editor/1.0')
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
            result: Dict[str, Any] = {}
            self._extract_phase_change_from_pubchem_node(
                data.get('Record', {}),
                result,
                identity_text=identifier,
            )
            self._finalize_pubchem_critical_candidates(result)
            self._finalize_online_phase_point_candidates(result)
            self._finalize_online_fusion_records(result)
            if result:
                sources = result.setdefault('_sources', {})
                sources.update({
                    key: sources.get(key, 'pubchem')
                    for key in result
                    if not key.startswith('_')
                    and key not in {'Hvap_records', 'Hfus_records'}
                })
                self._set_cache(cache_key, result)
                return result
        except Exception as error:
            if self._is_transient_lookup_error(error) or isinstance(error, LookupError):
                raise LookupError(
                    f"Transient PubChem phase-change lookup failure for '{identifier}'"
                ) from error
        self._set_missing_cache(cache_key)
        return None

    def _extract_phase_change_from_pubchem_node(
        self,
        node: Dict[str, Any],
        result: Dict[str, Any],
        identity_text: str = '',
    ) -> None:
        if not isinstance(node, dict):
            return
        heading = str(node.get('TOCHeading') or '').lower()
        texts = self._pubchem_section_texts(node)
        if heading == 'molecular formula' and '_formula' not in result:
            for text in texts:
                formula = re.sub(r'\s+', '', str(text))
                if re.fullmatch(r'(?:[A-Z][a-z]?\d*)+', formula):
                    result['_formula'] = formula
                    break
        if 'boiling point' in heading and 'Tb' not in result:
            for text in texts:
                value = self._parse_temperature_K(text)
                if value is not None:
                    result['Tb'] = value
                    break
        if 'melting point' in heading or 'freezing point' in heading:
            candidates = result.setdefault('_phase_candidates', {}).setdefault('Tm', [])
            self._collect_pubchem_temperature_candidates(
                node,
                candidates,
                method='pubchem_reported_melting',
                identity_text=identity_text,
            )
        if 'triple point' in heading:
            phase_candidates = result.setdefault('_phase_candidates', {})
            self._collect_pubchem_temperature_candidates(
                node,
                phase_candidates.setdefault('Tt', []),
                method='pubchem_reported_triple_temperature',
                identity_text=identity_text,
            )
            pressure_candidates = phase_candidates.setdefault('Pt', [])
            for information in node.get('Information', []) or []:
                if self._pubchem_information_is_predictive(information):
                    continue
                reference = self._pubchem_phase_reference(information)
                comment = str(
                    information.get('Description')
                    or information.get('Name')
                    or ''
                )
                for text in self._pubchem_information_texts(information):
                    pressure = self._parse_pressure_bar(text)
                    if pressure is not None and pressure > 0.0:
                        pressure_candidates.append({
                            'value_bar': float(pressure),
                            'source': 'pubchem',
                            'method': 'pubchem_reported_triple_pressure',
                            'reference': reference,
                            'comment': comment,
                            'raw': str(text),
                        })
        if 'critical' in heading and 'temperature' in heading:
            self._collect_pubchem_critical_candidates(
                node, result, 'Tc', self._parse_temperature_K,
            )
        if 'critical' in heading and 'pressure' in heading:
            self._collect_pubchem_critical_candidates(
                node, result, 'Pc', self._parse_pressure_bar,
            )
        if 'critical' in heading and 'volume' in heading:
            self._collect_pubchem_critical_candidates(
                node, result, 'Vc', self._parse_volume_cm3_per_mol,
            )
        if 'vaporization' in heading and ('heat' in heading or 'enthalpy' in heading):
            self._collect_pubchem_hvap_records(node, result)
        if 'heat of fusion' in heading:
            self._collect_pubchem_hfus_records(
                node,
                result,
                identity_text=identity_text,
            )
        for child in node.get('Section', []) or []:
            self._extract_phase_change_from_pubchem_node(
                child,
                result,
                identity_text=identity_text,
            )

    def _collect_pubchem_hfus_records(
        self,
        node: Dict[str, Any],
        result: Dict[str, Any],
        *,
        identity_text: str = '',
    ) -> None:
        records = result.setdefault('Hfus_records', [])
        information_rows = list(node.get('Information', []) or [])
        if not information_rows:
            information_rows = [{'Value': {'StringWithMarkup': [
                {'String': text} for text in self._pubchem_section_texts(node)
            ]}}]
        for information in information_rows:
            if self._pubchem_information_is_predictive(information):
                continue
            reference = self._pubchem_phase_reference(information)
            comment = str(
                information.get('Description')
                or information.get('Name')
                or ''
            )
            for text in self._pubchem_information_texts(information):
                value = self._parse_energy_kj_per_mol(text)
                if value is None or not (0.0 < value < 200.0):
                    continue
                temperature = self._parse_temperature_K(text)
                form = classify_fusion_material_form(
                    f'{comment} {text}',
                    identity_text,
                )
                records.append({
                    'enthalpy_kJ_mol': float(value),
                    'temperature_K': temperature,
                    'material_form': form[0],
                    'form_label': form[1],
                    'polymorph': form[2],
                    'stereochemistry': form[3],
                    'is_identity_form': form[4],
                    'source': 'pubchem',
                    'method': 'pubchem_reported_fusion',
                    'quality': 0.90,
                    'reference': reference,
                    'comment': comment,
                    'raw': str(text),
                })
        if not records:
            result.pop('Hfus_records', None)

    def _collect_pubchem_temperature_candidates(
        self,
        node: Dict[str, Any],
        candidates: list[Dict[str, Any]],
        *,
        method: str,
        identity_text: str = '',
    ) -> None:
        for information in node.get('Information', []) or []:
            if self._pubchem_information_is_predictive(information):
                continue
            reference = self._pubchem_phase_reference(information)
            comment = str(
                information.get('Description')
                or information.get('Name')
                or ''
            )
            for text in self._pubchem_information_texts(information):
                parsed = parse_reported_temperature_candidates(
                    text,
                    source='pubchem',
                    method=method,
                    reference=reference,
                    comment=comment,
                )
                for candidate in parsed:
                    form = classify_fusion_material_form(
                        f"{comment} {candidate.get('raw', '')}",
                        identity_text,
                    )
                    candidate.update({
                        'material_form': form[0],
                        'form_label': form[1],
                        'polymorph': form[2],
                        'stereochemistry': form[3],
                        'is_identity_form': form[4],
                    })
                candidates.extend(parsed)

    @staticmethod
    def _pubchem_phase_reference(information: Dict[str, Any]) -> str:
        for key in ('ReferenceNumber', 'Reference'):
            value = information.get(key)
            if value not in (None, '', []):
                if isinstance(value, (list, tuple)):
                    return '; '.join(str(item) for item in value)
                return str(value)
        return ''

    @classmethod
    def _collect_pubchem_critical_candidates(
        cls,
        node: Dict[str, Any],
        result: Dict[str, Any],
        key: str,
        parser,
    ) -> None:
        candidates = result.setdefault('_critical_candidates', {}).setdefault(key, [])
        for information in node.get('Information', []) or []:
            if cls._pubchem_information_is_predictive(information):
                continue
            for text in cls._pubchem_information_texts(information):
                value = parser(text)
                if value is not None and math.isfinite(value) and value > 0.0:
                    candidates.append(float(value))
                    break

    @staticmethod
    def _pubchem_information_texts(information: Dict[str, Any]) -> list[str]:
        value = information.get('Value', {}) if isinstance(information, dict) else {}
        unit = value.get('Unit', '')
        texts = [
            marked.get('String')
            for marked in value.get('StringWithMarkup', []) or []
            if marked.get('String')
        ]
        texts.extend(
            f"{number} {unit}".strip()
            for number in value.get('Number', []) or []
        )
        return texts

    @staticmethod
    def _pubchem_information_is_predictive(information: Dict[str, Any]) -> bool:
        if not isinstance(information, dict):
            return True
        text = json.dumps(information, sort_keys=True).lower()
        return bool(re.search(
            r'\b(?:estimated|estimate|predicted|prediction|calculated|'
            r'model(?:ed|led|ing)?|qspr|q-spr|epi\s*suite)\b',
            text,
        ))

    def _finalize_pubchem_critical_candidates(self, result: Dict[str, Any]) -> None:
        candidates_by_key = result.pop('_critical_candidates', {})
        formula = result.pop('_formula', None)
        tolerances = {'Tc': 0.03, 'Pc': 0.08, 'Vc': 0.10}
        qualities = result.setdefault('_qualities', {})
        notes = result.setdefault('_notes', {})
        for key, candidates in candidates_by_key.items():
            if not candidates:
                continue
            values = sorted(set(round(float(value), 10) for value in candidates))
            selected = self._median(values)
            relative_spread = (
                0.0 if len(values) == 1
                else (values[-1] - values[0]) / selected
            )
            if selected <= 0.0 or relative_spread > tolerances[key]:
                continue
            if key == 'Tc':
                try:
                    boiling_temperature = float(result.get('Tb'))
                except (TypeError, ValueError):
                    boiling_temperature = None
                if (
                    boiling_temperature is not None
                    and math.isfinite(boiling_temperature)
                    and boiling_temperature > 0.0
                    and selected <= boiling_temperature
                ):
                    result.setdefault('_rejections', []).append(
                        f'PubChem Tc={selected:g} K does not exceed reported '
                        f'Tb={boiling_temperature:g} K'
                    )
                    continue
            if (
                key == 'Pc'
                and formula
                and hasattr(self, '_critical_pressure_is_admissible')
                and not self._critical_pressure_is_admissible(
                    {'formula': formula},
                    selected,
                )
            ):
                result.setdefault('_rejections', []).append(
                    f'PubChem Pc={selected:g} bar exceeds the '
                    f'{"organic" if self._critical_formula_is_organic({"formula": formula}) else "inorganic"} '
                    'critical-pressure plausibility ceiling'
                )
                continue
            result[key] = selected
            qualities[key] = 0.88 if len(values) == 1 else 0.89
            notes[key] = (
                f"PubChem unit-qualified critical value from {len(values)} "
                f"consistent reported value(s); predictive entries excluded"
            )
        if not qualities:
            result.pop('_qualities', None)
        if not notes:
            result.pop('_notes', None)

    def _fetch_phase_change_nist(self, identifier: str) -> Optional[Dict[str, Any]]:
        cache_key = (
            f"phase_nist_v{self.ONLINE_PHASE_CHANGE_CACHE_VERSION}_{identifier}"
        )
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
                    'Mask': '4',
                })
                request = urllib.request.Request(f"{self.NIST_WEBBOOK}?{params}")
                request.add_header('User-Agent', 'PFD-Editor/1.0')
                with urllib.request.urlopen(request, timeout=15) as response:
                    html = response.read().decode('utf-8', errors='ignore')
                result = self._parse_nist_phase_change(
                    html,
                    identity_text=identifier,
                )
                if result:
                    self._set_cache(cache_key, result)
                    return result
            except Exception as error:
                transient_failure = (
                    transient_failure or self._is_transient_lookup_error(error)
                )
        if transient_failure:
            raise LookupError(
                f"Transient NIST phase-change lookup failure for '{identifier}'"
            )
        self._set_missing_cache(cache_key)
        return None

    def _parse_nist_phase_change(
        self,
        html: str,
        identity_text: str = '',
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        phase_candidates: Dict[str, list[Dict[str, Any]]] = {
            'Tm': [], 'Tt': [], 'Pt': [],
        }
        hvap_records: list[Dict[str, Any]] = []
        hfus_records: list[Dict[str, Any]] = []
        critical_candidates: Dict[str, list[tuple[float, str, str]]] = {
            'Tc': [], 'Pc': [], 'Vc': [],
        }
        for label, rows in self._html_tables(html):
            label_lower = label.strip().lower()
            if label_lower == 'enthalpy of vaporization':
                if not rows:
                    continue
                header = [cell.lower().replace(' ', '') for cell in rows[0]]
                if (
                    len(header) < 2
                    or 'kj/mol' not in header[0]
                    or 'temperature' not in header[1]
                ):
                    continue
                for row in rows[1:]:
                    if len(row) < 2:
                        continue
                    hvap = self._first_number(row[0])
                    temperature = self._first_number(row[1])
                    if (
                        hvap is not None
                        and temperature is not None
                        and 0.0 < hvap < 250.0
                        and temperature > 0.0
                    ):
                        hvap_records.append({
                            'value': float(hvap),
                            'T_ref': float(temperature),
                            'basis': 'saturation',
                            'source': 'nist_phase_change',
                            'method': row[2].strip() if len(row) > 2 else '',
                            'quality': 0.94,
                            'reference': row[3].strip() if len(row) > 3 else '',
                            'comment': row[4].strip() if len(row) > 4 else '',
                            'raw': '',
                        })
                continue
            if label_lower == 'enthalpy of fusion':
                header = [
                    re.sub(r'[^a-z0-9]+', '', cell.lower())
                    for cell in (rows[0] if rows else [])
                ]
                temperature_index = next(
                    (index for index, cell in enumerate(header) if 'temperature' in cell),
                    1,
                )
                method_index = next(
                    (index for index, cell in enumerate(header) if cell == 'method'),
                    None,
                )
                reference_index = next(
                    (index for index, cell in enumerate(header) if 'reference' in cell),
                    None,
                )
                comment_index = next(
                    (index for index, cell in enumerate(header) if 'comment' in cell),
                    None,
                )
                for row in rows[1:]:
                    if not row:
                        continue
                    value = self._first_number(row[0])
                    if value is None or not (0.0 < value < 200.0):
                        continue
                    temperature = (
                        self._first_number(row[temperature_index])
                        if temperature_index < len(row)
                        else None
                    )
                    method = (
                        row[method_index].strip()
                        if method_index is not None and method_index < len(row)
                        else ''
                    )
                    reference = (
                        row[reference_index].strip()
                        if reference_index is not None and reference_index < len(row)
                        else ''
                    )
                    comment = (
                        row[comment_index].strip()
                        if comment_index is not None and comment_index < len(row)
                        else ''
                    )
                    form = classify_fusion_material_form(
                        f'{method} {reference} {comment}',
                        identity_text,
                    )
                    hfus_records.append({
                        'enthalpy_kJ_mol': float(value),
                        'temperature_K': (
                            float(temperature) if temperature is not None else None
                        ),
                        'material_form': form[0],
                        'form_label': form[1],
                        'polymorph': form[2],
                        'stereochemistry': form[3],
                        'is_identity_form': form[4],
                        'source': 'nist_phase_change',
                        'method': method or 'nist_reported_fusion',
                        'quality': 0.94,
                        'reference': reference,
                        'comment': comment,
                        'raw': ' | '.join(row),
                    })
                continue

            for row in rows:
                if len(row) < 3 or row[0].strip().lower() == 'quantity':
                    continue
                key = re.sub(r'[^a-z0-9]+', '', row[0].lower())
                value = self._first_number(row[1])
                unit = row[2].lower()
                normalized_unit = (
                    re.sub(r'\s+', '', unit)
                    .replace('³', '3')
                    .replace('^', '')
                )
                method = row[3].strip() if len(row) > 3 else ''
                reference = row[4].strip() if len(row) > 4 else ''
                comment = row[5].strip() if len(row) > 5 else ''
                if key in {'tfus', 'ttriple'}:
                    candidate = nist_temperature_candidate(
                        row[1], row[2],
                        method=method,
                        reference=reference,
                        comment=comment,
                    )
                    if candidate is not None:
                        if key == 'tfus':
                            form = classify_fusion_material_form(
                                f'{method} {reference} {comment} {row[1]}',
                                identity_text,
                            )
                            candidate.update({
                                'material_form': form[0],
                                'form_label': form[1],
                                'polymorph': form[2],
                                'stereochemistry': form[3],
                                'is_identity_form': form[4],
                            })
                        phase_candidates[
                            'Tm' if key == 'tfus' else 'Tt'
                        ].append(candidate)
                    continue
                if key in {'ptriple', 'triplepointpressure'}:
                    pressure = self._parse_pressure_bar(f'{row[1]} {row[2]}')
                    if pressure is not None and pressure > 0.0:
                        phase_candidates['Pt'].append({
                            'value_bar': float(pressure),
                            'source': 'nist',
                            'method': (
                                'nist_avg' if method.lower() in {'avg', 'average'}
                                else 'nist_reported'
                            ),
                            'reference': reference,
                            'comment': comment,
                            'raw': str(row[1]),
                        })
                    continue
                if value is None:
                    continue
                if key == 'tboil' and normalized_unit in {'k', 'kelvin'}:
                    result['Tb'] = value
                elif key == 'tc' and normalized_unit in {'k', 'kelvin'}:
                    critical_candidates['Tc'].append((value, method, reference))
                elif key == 'pc':
                    if normalized_unit == 'bar':
                        critical_candidates['Pc'].append((value, method, reference))
                    elif normalized_unit == 'mpa':
                        critical_candidates['Pc'].append((value * 10.0, method, reference))
                elif key == 'vc':
                    if normalized_unit in {'l/mol', 'liter/mol', 'litre/mol'}:
                        critical_candidates['Vc'].append((value * 1000.0, method, reference))
                    elif normalized_unit in {'cm3/mol', 'ml/mol'}:
                        critical_candidates['Vc'].append((value, method, reference))
                elif 'vap' in key and 'h' in key and normalized_unit == 'kj/mol':
                    hvap_records.append({
                        'value': float(value),
                        'T_ref': 298.15,
                        'basis': 'standard_298',
                        'source': 'nist_phase_change',
                        'method': method,
                        'quality': 0.94 if method.lower() in {'avg', 'average'} else 0.91,
                        'reference': reference,
                        'comment': comment,
                        'raw': '',
                    })
                elif 'fus' in key and 'h' in key and 'kj/mol' in unit:
                    form = classify_fusion_material_form(
                        f'{method} {reference} {comment}',
                        identity_text,
                    )
                    hfus_records.append({
                        'enthalpy_kJ_mol': float(value),
                        'temperature_K': None,
                        'material_form': form[0],
                        'form_label': form[1],
                        'polymorph': form[2],
                        'stereochemistry': form[3],
                        'is_identity_form': form[4],
                        'source': 'nist_phase_change',
                        'method': method or 'nist_reported_fusion',
                        'quality': 0.93 if method.lower() in {'avg', 'average'} else 0.90,
                        'reference': reference,
                        'comment': comment,
                        'raw': ' | '.join(row),
                    })

        critical_qualities = {}
        critical_notes = {}
        for key, candidates in critical_candidates.items():
            selected = self._select_nist_critical_candidate(key, candidates)
            if selected is None:
                continue
            value, quality, note = selected
            result[key] = value
            critical_qualities[key] = quality
            critical_notes[key] = note
        if hvap_records:
            result['Hvap_records'] = hvap_records
        if hfus_records:
            result['Hfus_records'] = hfus_records
        nonempty = {key: values for key, values in phase_candidates.items() if values}
        if nonempty:
            result['_phase_candidates'] = nonempty
        if result:
            result['_sources'] = {
                key: 'nist_phase_change'
                for key in result
                if not key.startswith('_')
                and key not in {'Hvap_records', 'Hfus_records'}
            }
            if critical_qualities:
                result.setdefault('_qualities', {}).update(critical_qualities)
                result.setdefault('_notes', {}).update(critical_notes)
            self._finalize_online_phase_point_candidates(result)
            self._finalize_online_fusion_records(result)
        return result

    def _select_nist_critical_candidate(
        self,
        key: str,
        candidates: list[tuple[float, str, str]],
    ) -> Optional[tuple[float, float, str]]:
        if not candidates:
            return None
        averages = [
            candidate for candidate in candidates
            if candidate[1].strip().lower() in {'avg', 'average'}
        ]
        selected_candidates = averages or candidates
        values = sorted(set(
            round(float(value), 10)
            for value, _method, _reference in selected_candidates
        ))
        selected = self._median(values)
        tolerances = {'Tc': 0.03, 'Pc': 0.08, 'Vc': 0.10}
        relative_spread = (
            0.0 if len(values) == 1
            else (values[-1] - values[0]) / selected
        )
        if selected <= 0.0 or relative_spread > tolerances[key]:
            return None
        if averages:
            quality = 0.96
            selection = 'NIST AVG row'
        elif len(values) == 1:
            quality = 0.94
            selection = 'single NIST reported row'
        else:
            quality = 0.95
            selection = f'median of {len(values)} consistent NIST reported rows'
        return selected, quality, f'{selection}; explicit SI-compatible units'
    def _fit_hvap_watson(
        self,
        points: list[tuple[float, float]],
        Tc: Optional[float],
    ) -> Optional[HvapTemperatureFit]:
        if Tc is None or Tc <= 0.0:
            return None

        binned: dict[float, list[float]] = {}
        for T, hvap in points:
            if not (0.35 * Tc <= T < 0.995 * Tc and 0.0 < hvap < 150.0):
                continue
            binned.setdefault(round(float(T), 1), []).append(float(hvap))

        data = [(T, self._median(values)) for T, values in sorted(binned.items())]
        if len(data) < 4:
            return None

        kept = [True] * len(data)
        slope = intercept = None
        for _ in range(8):
            subset = [point for point, include in zip(data, kept) if include]
            if len(subset) < 4:
                return None
            x_values = [math.log(1.0 - T / Tc) for T, _ in subset]
            y_values = [math.log(hvap) for _, hvap in subset]
            fit = self._linear_fit(x_values, y_values)
            if fit is None:
                return None
            slope, intercept = fit

            all_residuals = [
                math.log(hvap) - (intercept + slope * math.log(1.0 - T / Tc))
                for T, hvap in data
            ]
            kept_abs = [abs(residual) for residual, include in zip(all_residuals, kept) if include]
            residual_median = self._median(kept_abs)
            mad = self._median([abs(value - residual_median) for value in kept_abs]) or 1e-12
            threshold = max(0.10, residual_median + 3.0 * 1.4826 * mad)
            new_kept = [abs(residual) <= threshold for residual in all_residuals]
            if sum(new_kept) < 4:
                break
            if new_kept == kept:
                break
            kept = new_kept

        subset = [point for point, include in zip(data, kept) if include]
        if len(subset) < 4:
            return None
        x_values = [math.log(1.0 - T / Tc) for T, _ in subset]
        y_values = [math.log(hvap) for _, hvap in subset]
        fit = self._linear_fit(x_values, y_values)
        if fit is None:
            return None
        slope, intercept = fit
        A = math.exp(intercept)

        predicted = [A * (1.0 - T / Tc) ** slope for T, _ in subset]
        mape = 100.0 * sum(
            abs(pred / hvap - 1.0)
            for pred, (_, hvap) in zip(predicted, subset)
        ) / len(subset)

        if not (0.10 <= slope <= 0.80):
            return None
        if mape > 8.0:
            return None

        return HvapTemperatureFit(
            A=A,
            n=slope,
            Tc=Tc,
            T_min=min(T for T, _ in subset),
            T_max=max(T for T, _ in subset),
            mape_percent=mape,
            kept_points=len(subset),
            total_points=len(data),
            source='NIST WebBook',
        )


    @staticmethod
    def _parse_hvap_energy_kj_per_mol(text: str) -> Optional[float]:
        normalized = str(text).replace('−', '-').replace(',', '').replace('X10+', 'e')
        match = re.search(
            r'(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*'
            r'(kJ/(?:mol|mole)|J/(?:mol|mole)|kcal/(?:mol|mole)|'
            r'cal/(?:mol|mole)|gcal/gmole)\b',
            normalized,
            re.IGNORECASE,
        )
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2).lower().replace('mole', 'mol')
        if unit == 'kj/mol':
            return value
        if unit == 'j/mol':
            return value / 1000.0
        if unit == 'kcal/mol':
            return value * 4.184
        if unit in {'cal/mol', 'gcal/gmol'}:
            return value * 4.184 / 1000.0
        return None


    def _collect_pubchem_hvap_records(
        self,
        node: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        records = result.setdefault('Hvap_records', [])
        for information in node.get('Information', []) or []:
            if self._pubchem_hvap_information_is_predictive(information):
                continue
            texts = self._pubchem_information_texts(information)
            for text in texts:
                hvap = self._parse_hvap_energy_kj_per_mol(text)
                if hvap is None or not (0.0 < hvap < 250.0):
                    continue
                context = ' '.join((
                    text,
                    str(information.get('Name') or ''),
                    str(information.get('Description') or ''),
                ))
                T_ref = self._parse_temperature_K(context)
                lowered = context.lower()
                if T_ref is not None:
                    basis = 'explicit_temperature'
                elif 'standard condition' in lowered:
                    T_ref = REFERENCE_TEMPERATURE_K
                    basis = 'standard_298'
                elif 'boiling point' in lowered:
                    basis = 'normal_boiling_point'
                else:
                    basis = 'unspecified'
                references = information.get('Reference', []) or []
                description = str(information.get('Description') or '')
                records.append({
                    'value': float(hvap),
                    'T_ref': float(T_ref) if T_ref is not None else None,
                    'basis': basis,
                    'source': 'pubchem',
                    'method': description,
                    'quality': (
                        0.89
                        if T_ref is not None
                        else 0.82
                        if basis == 'normal_boiling_point'
                        else 0.76
                    ),
                    'reference': '; '.join(str(item) for item in references),
                    'comment': '',
                    'raw': text,
                })
                break


    @staticmethod
    def _pubchem_hvap_information_is_predictive(information: Dict[str, Any]) -> bool:
        if not isinstance(information, dict):
            return True
        text = json.dumps(information, sort_keys=True).lower()
        return bool(re.search(
            r'\b(?:estimated|estimate|predicted|prediction|qspr|q-spr|'
            r'epi\s*suite|machine[- ]learning|group[- ]contribution|'
            r'model(?:ed|led|ing)?\s+value)\b',
            text,
        ))


    @staticmethod
    def _normalized_hvap_records(records: Any) -> list[Dict[str, Any]]:
        normalized = []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            try:
                value = float(record.get('value'))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(value) and 0.0 < value < 250.0):
                continue
            T_ref = record.get('T_ref')
            try:
                T_ref = float(T_ref) if T_ref is not None else None
            except (TypeError, ValueError):
                T_ref = None
            if T_ref is not None and not (math.isfinite(T_ref) and T_ref > 0.0):
                T_ref = None
            item = dict(record)
            item['value'] = value
            item['T_ref'] = T_ref
            normalized.append(item)
        return normalized


    def _collapse_hvap_records(
        self,
        records: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        grouped: Dict[tuple[str, str, Optional[float]], list[Dict[str, Any]]] = {}
        for record in records:
            T_ref = record.get('T_ref')
            key = (
                str(record.get('source') or ''),
                str(record.get('basis') or ''),
                round(float(T_ref), 1) if T_ref is not None else None,
            )
            grouped.setdefault(key, []).append(record)

        collapsed = []
        for candidates in grouped.values():
            averages = [
                item for item in candidates
                if str(item.get('method') or '').strip().lower() in {'avg', 'average'}
            ]
            selected = averages or candidates
            values = [float(item['value']) for item in selected]
            center = self._median(values)
            consistent = [value for value in values if abs(value / center - 1.0) <= 0.15]
            if not consistent:
                continue
            representative = dict(selected[0])
            representative['value'] = self._median(consistent)
            representative['reported_values'] = len(values)
            if averages:
                representative['quality'] = max(
                    float(representative.get('quality') or 0.0),
                    0.94,
                )
            collapsed.append(representative)
        return collapsed


    @staticmethod
    def _hvap_fit_quality(fit: HvapTemperatureFit, T: float) -> tuple[float, str]:
        if fit.T_min <= T <= fit.T_max:
            return NIST_HVAP_FIT_QUALITY, 'inside fitted temperature range'
        distance = fit.T_min - T if T < fit.T_min else T - fit.T_max
        width = max(fit.T_max - fit.T_min, 1.0)
        quality = max(
            NIST_HVAP_FIT_MINIMUM_QUALITY,
            NIST_HVAP_FIT_QUALITY - 0.08 * min(distance / width, 1.5),
        )
        return quality, f'extrapolated {distance:g} K outside fitted temperature range'


    def _finalize_online_hvap(self, result: Dict[str, Any]) -> None:
        result.pop('Hvap', None)
        result.pop('Hvap_fit', None)
        for metadata_key in ('_sources', '_qualities', '_notes'):
            result.setdefault(metadata_key, {}).pop('Hvap', None)
            result[metadata_key].pop('Hvap_fit', None)

        records = self._normalized_hvap_records(result.get('Hvap_records'))
        if not records:
            result.pop('Hvap_records', None)
            return
        result['Hvap_records'] = records
        collapsed = self._collapse_hvap_records(records)

        Tb = result.get('Tb')
        Tc = result.get('Tc')
        try:
            Tb = float(Tb) if Tb is not None else None
        except (TypeError, ValueError):
            Tb = None
        try:
            Tc = float(Tc) if Tc is not None else None
        except (TypeError, ValueError):
            Tc = None

        nist_points = [
            (float(record['T_ref']), float(record['value']))
            for record in collapsed
            if record.get('source') == 'nist_phase_change'
            and record.get('T_ref') is not None
            and record.get('basis') in {'saturation', 'standard_298'}
        ]
        fit = self._fit_hvap_watson(nist_points, Tc)
        if fit is not None:
            result['Hvap_fit'] = fit
            result['_sources']['Hvap_fit'] = 'nist_phase_change'
            fit_quality, fit_note = (
                self._hvap_fit_quality(fit, Tb)
                if Tb
                else (NIST_HVAP_FIT_QUALITY, '')
            )
            result['_qualities']['Hvap_fit'] = fit_quality
            result['_notes']['Hvap_fit'] = (
                f'NIST Watson fit from {fit.kept_points}/{fit.total_points} '
                f'temperature-qualified points; {fit_note}'
            ).rstrip('; ')

        if Tb is None or not (math.isfinite(Tb) and Tb > 0.0):
            return

        candidates = []
        for record in collapsed:
            T_ref = record.get('T_ref')
            basis = str(record.get('basis') or '')
            source = str(record.get('source') or '')
            value = float(record['value'])
            if basis == 'normal_boiling_point' and T_ref is None:
                converted = value
                conversion = 'reported at the normal boiling point'
                distance = 0.0
            elif T_ref is not None and abs(float(T_ref) - Tb) <= 1.0:
                if Tc and 0.0 < float(T_ref) < Tc and Tb < Tc:
                    converted = self._watson_hvap_value(value, float(T_ref), Tb, Tc)
                else:
                    converted = value
                conversion = f'reported at {float(T_ref):g} K near Tb'
                distance = abs(float(T_ref) - Tb)
            elif T_ref is not None and Tc and 0.0 < float(T_ref) < Tc and Tb < Tc:
                converted = self._watson_hvap_value(value, float(T_ref), Tb, Tc)
                conversion = f'Watson-scaled from {float(T_ref):g} K to Tb'
                distance = abs(float(T_ref) - Tb)
            else:
                continue
            if converted is None or converted <= 0.0:
                continue

            if source == 'nist_phase_change' and basis in {'saturation', 'normal_boiling_point'} and distance <= 1.0:
                rank = 0
            elif source == 'pubchem' and distance <= 1.0:
                rank = 1
            elif source == 'nist_phase_change' and basis == 'saturation':
                rank = 3
            elif source == 'nist_phase_change' and basis == 'standard_298':
                rank = 4
            elif source == 'pubchem' and T_ref is not None:
                rank = 5
            else:
                rank = 6
            base_quality = float(record.get('quality') or 0.81)
            if distance <= 1.0:
                distance_factor = 1.0
            else:
                near_distance = 0.05 * Tc if Tc is not None else 20.0
                distance_factor = 0.95 if distance < near_distance else 0.90
            quality = base_quality * distance_factor
            candidates.append((rank, distance, -quality, converted, quality, record, conversion))

        if fit is not None:
            fit_value = fit.value_at(Tb)
            if fit_value is not None and fit_value > 0.0:
                fit_quality, fit_note = self._hvap_fit_quality(fit, Tb)
                candidates.append((
                    2,
                    0.0,
                    -fit_quality,
                    fit_value,
                    fit_quality,
                    {
                        'source': 'nist_phase_change',
                        'method': 'nist_hvap_watson_fit',
                        'reference': '',
                    },
                    fit_note,
                ))

        if not candidates:
            return
        _rank, _distance, _neg_quality, value, quality, record, conversion = min(candidates)
        source = str(record.get('source') or 'online_phase_change')
        method = str(record.get('method') or '')
        if method == 'nist_hvap_watson_fit':
            selected_method = method
        elif source == 'nist_phase_change':
            selected_method = 'nist_hvap_at_tb'
        else:
            selected_method = 'pubchem_hvap_at_tb'
        reference = str(record.get('reference') or '').strip()
        result['Hvap'] = float(value)
        result['_sources']['Hvap'] = selected_method
        result['_qualities']['Hvap'] = float(quality)
        result['_notes']['Hvap'] = (
            f'Hvap(Tb={Tb:g} K); {conversion}; source {source}'
            + (f'; {reference}' if reference else '')
        )
