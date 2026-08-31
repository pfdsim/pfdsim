from .common import *


DOMALSKI_HEARING_HF_QUALITY = 0.85
DOMALSKI_HEARING_S_QUALITY = 0.89
DOMALSKI_HEARING_SS_S_QUALITY = 0.75


class FormationPropertiesMixin:
        def _derive_hf_gas_from_liquid(
            self,
            symbol: str,
            props: Dict[str, Any],
            online: Optional[Dict[str, Any]],
            allow_online: bool,
        ) -> Optional[PropertyResolutionResult]:
            liquid_hf = self._phase_formation_value(
                'Hf_liquid',
                props,
                online,
                allow_online,
                'kJ/mol',
            )
            if liquid_hf is None or liquid_hf.value is None:
                return None

            try:
                hvap = self.resolve_hvap(
                    symbol,
                    props,
                    T=298.15,
                    allow_online=allow_online,
                    allow_estimation=False,
                )
            except Exception:
                return None
            if hvap.value is None:
                return None

            inputs = [liquid_hf, hvap]
            return PropertyResolutionResult(
                value=float(liquid_hf.value) + float(hvap.value),
                source=self._derived_source(inputs, exact_formula=True),
                method='liquid_hf_plus_hvap',
                quality=self._combine_quality(inputs, exact_formula=True),
                notes=(
                    f"gas Hf derived from {liquid_hf.source}/{liquid_hf.method} "
                    f"({liquid_hf.notes}) plus Hvap(298.15 K) from "
                    f"{hvap.source}/{hvap.method}; units kJ/mol"
                ),
            )


        def _load_element_standard_entropies(self) -> Dict[str, Dict[str, Any]]:
            if self._element_entropy_cache is not None:
                return self._element_entropy_cache

            table: Dict[str, Dict[str, Any]] = {}
            path = Path(__file__).resolve().parent.parent / 'data' / 'S_table_elements.txt'
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import parse_formula_counts
                else:
                    from compound_identity import parse_formula_counts
            except Exception:
                self._element_entropy_cache = table
                return table

            try:
                text = path.read_text()
            except OSError:
                self._element_entropy_cache = table
                return table

            for line in text.splitlines():
                match = re.match(
                    r'\|\s*\d+\s*\|\s*([A-Z][a-z]?)\s*\|\s*([^|]+?)\s*\|\s*([0-9.]+)\*?\s*\|',
                    line,
                )
                if not match:
                    continue
                element, reference_form, entropy_text = match.groups()
                reference_formula = reference_form.strip().split('(', 1)[0].strip()
                counts = parse_formula_counts(reference_formula)
                if not counts or counts.get(element) is None or len(counts) != 1:
                    continue
                table[element] = {
                    'entropy': float(entropy_text),
                    'reference_form': reference_form.strip(),
                    'atoms_per_reference': counts[element],
                }

            self._element_entropy_cache = table
            return table


        def _element_entropy_sum(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[tuple[float, str]]:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import parse_formula_counts
                else:
                    from compound_identity import parse_formula_counts
            except Exception:
                return None

            formula = props.get('formula') or props.get('Formula') or symbol
            counts = parse_formula_counts(str(formula))
            if not counts:
                return None

            element_table = self._load_element_standard_entropies()
            total = 0.0
            for element, atom_count in counts.items():
                entry = element_table.get(element)
                if not entry:
                    return None
                total += atom_count / entry['atoms_per_reference'] * entry['entropy']
            return total, str(formula)


        @staticmethod
        def _formation_result_value(result: Optional[PropertyResolutionResult]) -> Optional[float]:
            if result is None or result.value is None:
                return None
            return float(result.value)


        def _derive_gf_gas_from_hf_and_s(
            self,
            symbol: str,
            props: Dict[str, Any],
            hf: PropertyResolutionResult,
            entropy: PropertyResolutionResult,
        ) -> Optional[PropertyResolutionResult]:
            element_entropy = self._element_entropy_sum(symbol, props)
            if element_entropy is None:
                return None
            element_s, formula = element_entropy
            hf_value = self._formation_result_value(hf)
            s_value = self._formation_result_value(entropy)
            if hf_value is None or s_value is None:
                return None

            T_ref = 298.15
            delta_s = s_value - element_s
            value = hf_value - T_ref * delta_s / 1000.0
            inputs = [hf, entropy]
            return PropertyResolutionResult(
                value=value,
                source=self._derived_source(inputs, exact_formula=True),
                method='hf_absolute_entropy_to_gf',
                quality=self._combine_quality(inputs, exact_formula=True),
                notes=(
                    f"gas Gf derived from gas Hf and absolute gas S at {T_ref:g} K; "
                    f"formula {formula}, elemental reference entropy sum {element_s:.3g} J/mol/K; "
                    "units kJ/mol"
                ),
            )


        def _resolve_domalski_hearing_formation(
            self,
            symbol: str,
            props: Dict[str, Any],
            allow_online: bool,
        ) -> Dict[str, PropertyResolutionResult]:
            """Return final-rung Domalski--Hearing gas Hf/S estimates."""
            smiles_result = self._resolve_smiles_result(
                symbol,
                props,
                allow_online=allow_online,
            )
            if not smiles_result or not smiles_result.value:
                return {}
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from .. import domalski_hearing_method as domalski_hearing
                else:
                    import domalski_hearing_method as domalski_hearing
            except ImportError:
                return {}
            try:
                estimate = domalski_hearing.estimate(str(smiles_result.value))
            except domalski_hearing.DomalskiHearingError:
                return {}

            # A few structures can be fully claimed without receiving any
            # published group. Their zero sums are not physical estimates.
            if not estimate.groups and not estimate.corrections:
                return {}

            resolved: Dict[str, PropertyResolutionResult] = {}
            gas_hf = estimate.gas.enthalpy_formation_kJ_mol
            if gas_hf is not None:
                applicability = domalski_hearing.assess_hf_applicability(estimate)
                if applicability.applicable:
                    resolved['Hf'] = PropertyResolutionResult(
                        value=float(gas_hf),
                        source='estimated',
                        method='domalski_hearing_gas_hf',
                        quality=DOMALSKI_HEARING_HF_QUALITY,
                        notes=(
                            'Domalski-Hearing group-additivity gas Hf at 298.15 K; '
                            f'SMILES source: {smiles_result.method}; units kJ/mol'
                        ),
                    )
                else:
                    resolved['Hf'] = PropertyResolutionResult(
                        value=None,
                        source='missing',
                        method='domalski_hearing_hf_not_applicable',
                        quality=0.0,
                        notes=(
                            'Domalski-Hearing gas Hf excluded as structurally '
                            'non-applicable: ' + '; '.join(applicability.reasons)
                        ),
                    )

            gas_entropy = estimate.gas.entropy_J_mol_K
            if gas_entropy is not None:
                has_ss_bond = domalski_hearing.has_sulfur_sulfur_bond(estimate)
                entropy_quality = (
                    DOMALSKI_HEARING_SS_S_QUALITY
                    if has_ss_bond
                    else DOMALSKI_HEARING_S_QUALITY
                )
                target_pressure_pa = (
                    THERMOCHEMICAL_STANDARD_PRESSURE_BAR * 100000.0
                )
                pressure_correction = domalski_hearing.R_J_MOL_K * math.log(
                    domalski_hearing.PRESSURE_PA / target_pressure_pa
                )
                resolved['S'] = PropertyResolutionResult(
                    value=float(gas_entropy) + pressure_correction,
                    source='estimated',
                    method='domalski_hearing_gas_entropy_1bar',
                    quality=entropy_quality,
                    notes=(
                        'Domalski-Hearing absolute ideal-gas entropy at 298.15 K, '
                        f'converted from {domalski_hearing.PRESSURE_PA:g} Pa to '
                        f'{THERMOCHEMICAL_STANDARD_PRESSURE_BAR:g} bar '
                        f'({pressure_correction:+.6g} J/(mol*K)); '
                        + (
                            'quality reduced for an S-S bonded compound; '
                            if has_ss_bond else ''
                        )
                        + f'SMILES source: {smiles_result.method}; units J/(mol*K)'
                    ),
                )
            return resolved


        def _derive_s_gas_from_hf_and_gf(
            self,
            symbol: str,
            props: Dict[str, Any],
            hf: PropertyResolutionResult,
            gf: PropertyResolutionResult,
        ) -> Optional[PropertyResolutionResult]:
            element_entropy = self._element_entropy_sum(symbol, props)
            if element_entropy is None:
                return None
            element_s, formula = element_entropy
            hf_value = self._formation_result_value(hf)
            gf_value = self._formation_result_value(gf)
            if hf_value is None or gf_value is None:
                return None

            T_ref = 298.15
            value = (hf_value - gf_value) * 1000.0 / T_ref + element_s
            if value <= 0.0:
                return None
            inputs = [hf, gf]
            return PropertyResolutionResult(
                value=value,
                source=self._derived_source(inputs, exact_formula=True),
                method='hf_gf_to_absolute_entropy',
                quality=self._combine_quality(inputs, exact_formula=True),
                notes=(
                    f"absolute gas S derived from gas Hf and gas Gf at {T_ref:g} K; "
                    f"formula {formula}, elemental reference entropy sum {element_s:.3g} J/mol/K; "
                    "units J/(mol*K)"
                ),
            )


        def _phase_formation_value(
            self,
            key: str,
            props: Dict[str, Any],
            online: Optional[Dict[str, Any]],
            allow_online: bool,
            units: str,
        ) -> Optional[PropertyResolutionResult]:
            if props.get(key) is not None:
                return self._source_result_for_value(
                    props,
                    key,
                    value=float(props[key]),
                    units=units,
                    default_quality=1.0,
                )
            if allow_online and online and online.get(key) is not None:
                sources = online.get('_sources', {})
                notes = online.get('_notes', {})
                phase = 'liquid' if key.endswith('_liquid') else 'gas'
                return PropertyResolutionResult(
                    value=float(online[key]),
                    source='online',
                    method=sources.get(key, f'nist_{phase}_thermochemistry'),
                    quality=0.92,
                    notes=f"{notes.get(key, f'NIST WebBook {phase} thermochemistry')}; units {units}",
                )
            return None


        def _derive_gf_gas_from_liquid_gf(
            self,
            symbol: str,
            props: Dict[str, Any],
            online: Optional[Dict[str, Any]],
            allow_online: bool,
            liquid_gf: Optional[PropertyResolutionResult] = None,
        ) -> Optional[PropertyResolutionResult]:
            liquid_gf = liquid_gf or self._phase_formation_value(
                'Gf_liquid',
                props,
                online,
                allow_online,
                'kJ/mol',
            )
            if liquid_gf is None or liquid_gf.value is None:
                return None

            T_ref = 298.15
            try:
                psat = self.resolve_vapor_pressure(symbol, T_ref, props)
            except Exception:
                return None
            if psat.value is None or psat.value <= 0.0:
                return None

            delta_g_vap = (
                -R * T_ref
                * math.log(float(psat.value) / THERMOCHEMICAL_STANDARD_PRESSURE_BAR)
                / 1000.0
            )
            inputs = [liquid_gf, psat]
            return PropertyResolutionResult(
                value=float(liquid_gf.value) + delta_g_vap,
                source='calculated',
                method='liquid_gf_plus_standard_vaporization_gibbs',
                quality=self._combine_quality(inputs, method_factor=0.88),
                notes=(
                    f"gas Gf derived from {liquid_gf.source}/{liquid_gf.method} plus "
                    f"-RT ln(Psat/{THERMOCHEMICAL_STANDARD_PRESSURE_BAR:g} bar) at {T_ref:g} K using Psat from "
                    f"{psat.source}/{psat.method}; units kJ/mol"
                ),
            )


        def _derive_gf_liquid_from_liquid_s(
            self,
            symbol: str,
            props: Dict[str, Any],
            online: Optional[Dict[str, Any]],
            allow_online: bool,
            hf_gas: PropertyResolutionResult,
        ) -> Optional[PropertyResolutionResult]:
            s_liquid = self._phase_formation_value(
                'S_liquid',
                props,
                online,
                allow_online,
                'J/(mol*K)',
            )
            if s_liquid is None or s_liquid.value is None:
                return None

            hf_value = self._formation_result_value(hf_gas)
            if hf_value is None:
                return None

            try:
                hvap = self.resolve_hvap(
                    symbol,
                    props,
                    T=298.15,
                    allow_online=allow_online,
                    allow_estimation=False,
                )
            except Exception:
                return None
            if hvap.value is None:
                return None

            element_entropy = self._element_entropy_sum(symbol, props)
            if element_entropy is None:
                return None
            element_s, formula = element_entropy
            T_ref = 298.15
            hf_liquid = hf_value - float(hvap.value)
            delta_s_liquid = float(s_liquid.value) - element_s
            gf_liquid = hf_liquid - T_ref * delta_s_liquid / 1000.0
            inputs = [hf_gas, s_liquid, hvap]
            return PropertyResolutionResult(
                value=gf_liquid,
                source=self._derived_source(inputs, exact_formula=True),
                method='hf_liquid_absolute_entropy_to_gf_liquid',
                quality=self._combine_quality(inputs, exact_formula=True),
                notes=(
                    f"liquid Gf derived from gas Hf minus Hvap(298.15 K) and absolute liquid S; "
                    f"formula {formula}, elemental reference entropy sum {element_s:.3g} J/mol/K; "
                    f"Hvap from {hvap.source}/{hvap.method}; units kJ/mol"
                ),
            )


        def _formula_counts(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[tuple[Dict[str, int], str]]:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import parse_formula_counts
                else:
                    from compound_identity import parse_formula_counts
            except Exception:
                return None
            formula = props.get('formula') or props.get('Formula') or symbol
            counts = parse_formula_counts(str(formula))
            if not counts:
                return None
            return counts, str(formula)


        def _derive_net_hcomb_from_gross(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            gross = self._source_result_for_value(
                props,
                'Hcomb_gross',
                units='kJ/mol',
                default_quality=1.0,
            )
            if gross is None or gross.value is None:
                return None
            parsed = self._formula_counts(symbol, props)
            if parsed is None:
                return None
            counts, formula = parsed
            water_moles = counts.get('H', 0) / 2.0
            if water_moles <= 0.0:
                return None
            inputs = [gross]
            return PropertyResolutionResult(
                value=float(gross.value) + water_moles * WATER_HVAP_298_KJ_PER_MOL,
                source=self._derived_source(inputs, exact_formula=True),
                method='gross_to_net_hcomb',
                quality=self._combine_quality(inputs, exact_formula=True),
                notes=(
                    f"net Hcomb derived from provided gross Hcomb using formula {formula}; "
                    f"added {water_moles:g} mol water vaporization at 298.15 K; units kJ/mol"
                ),
            )


        @staticmethod
        def _phase_at_298(props: Dict[str, Any]) -> str:
            phase = str(props.get('phase_at_STP') or props.get('phase') or '').strip().lower()
            if phase in {'gas', 'vapor', 'vapour'}:
                return 'gas'
            if phase == 'liquid':
                return 'liquid'
            if phase == 'solid':
                return 'solid'

            Tm = props.get('Tm')
            Tb = props.get('Tb')
            try:
                if Tm is not None and float(Tm) > REFERENCE_TEMPERATURE_K:
                    return 'solid'
                if Tb is not None and float(Tb) <= REFERENCE_TEMPERATURE_K:
                    return 'gas'
                if (
                    Tm is not None and float(Tm) <= REFERENCE_TEMPERATURE_K
                    and Tb is not None and float(Tb) > REFERENCE_TEMPERATURE_K
                ):
                    return 'liquid'
            except (TypeError, ValueError):
                pass
            return 'unknown'


        def _reactant_hf_for_combustion(
            self,
            symbol: str,
            props: Dict[str, Any],
            online: Optional[Dict[str, Any]],
            allow_online: bool,
            hf_gas: Optional[PropertyResolutionResult],
        ) -> Optional[tuple[float, str, PropertyResolutionResult]]:
            phase = self._phase_at_298(props)
            if phase == 'gas':
                value = self._formation_result_value(hf_gas)
                if value is None:
                    return None
                return value, 'gas Hf', hf_gas

            if phase == 'liquid':
                liquid_hf = self._phase_formation_value(
                    'Hf_liquid',
                    props,
                    online,
                    allow_online,
                    'kJ/mol',
                )
                if liquid_hf is not None and liquid_hf.value is not None:
                    return float(liquid_hf.value), f"{liquid_hf.source}/{liquid_hf.method} Hf_liquid", liquid_hf
                hf_value = self._formation_result_value(hf_gas)
                if hf_value is None:
                    return None
                try:
                    hvap = self.resolve_hvap(
                        symbol,
                        props,
                        T=REFERENCE_TEMPERATURE_K,
                        allow_online=allow_online,
                        allow_estimation=False,
                    )
                except Exception:
                    return None
                if hvap.value is None:
                    return None
                liquid_from_gas = PropertyResolutionResult(
                    value=hf_value - float(hvap.value),
                    source=self._derived_source([hf_gas, hvap], exact_formula=True),
                    method='gas_hf_minus_hvap',
                    quality=self._combine_quality([hf_gas, hvap], exact_formula=True),
                    notes=f"gas Hf minus Hvap from {hvap.source}/{hvap.method}",
                )
                return (
                    liquid_from_gas.value,
                    f"gas Hf minus Hvap from {hvap.source}/{hvap.method}",
                    liquid_from_gas,
                )

            if phase == 'solid':
                solid_hf = self._phase_formation_value(
                    'Hf_solid',
                    props,
                    online,
                    allow_online,
                    'kJ/mol',
                )
                if solid_hf is not None and solid_hf.value is not None:
                    return float(solid_hf.value), f"{solid_hf.source}/{solid_hf.method} Hf_solid", solid_hf
                return None

            return None


        def _derive_net_hcomb_from_formula(
            self,
            symbol: str,
            props: Dict[str, Any],
            online: Optional[Dict[str, Any]],
            allow_online: bool,
            hf_gas: Optional[PropertyResolutionResult],
        ) -> Optional[PropertyResolutionResult]:
            parsed = self._formula_counts(symbol, props)
            if parsed is None:
                return None
            counts, formula = parsed
            supported = {'C', 'H', 'O', 'N', 'S', 'Si', 'F', 'Cl', 'Br', 'I'}
            unsupported = set(counts) - supported
            if unsupported:
                return None
            if not (
                counts.get('C', 0)
                or counts.get('S', 0)
                or counts.get('Si', 0)
                or (counts.get('H', 0) and not ({'F', 'Cl', 'Br', 'I'} & set(counts)))
            ):
                return None

            reactant = self._reactant_hf_for_combustion(symbol, props, online, allow_online, hf_gas)
            if reactant is None:
                return None
            reactant_hf, reactant_note, reactant_result = reactant

            c = counts.get('C', 0)
            h = counts.get('H', 0)
            s = counts.get('S', 0)
            si = counts.get('Si', 0)
            o = counts.get('O', 0)
            oxygen_atoms_required = 2.0 * c + 0.5 * h + 2.0 * s + 2.0 * si - o
            if oxygen_atoms_required < -1e-9:
                return None

            product_hf = (
                c * NET_COMBUSTION_PRODUCT_HF_KJ_PER_MOL['CO2']
                + (h / 2.0) * NET_COMBUSTION_PRODUCT_HF_KJ_PER_MOL['H2O']
                + s * NET_COMBUSTION_PRODUCT_HF_KJ_PER_MOL['SO2']
                + si * NET_COMBUSTION_PRODUCT_HF_KJ_PER_MOL['SiO2']
            )
            value = product_hf - reactant_hf
            inputs = [reactant_result]
            return PropertyResolutionResult(
                value=value,
                source=self._derived_source(inputs, exact_formula=True),
                method='formula_net_hcomb',
                quality=self._combine_quality(inputs, exact_formula=True),
                notes=(
                    f"net Hcomb calculated from formula {formula} and {reactant_note}; "
                    "products CO2(g), H2O(g), SO2(g), SiO2(s), N2(g), and elemental halogen standard states; "
                    "units kJ/mol"
                ),
            )


        def resolve_formation_properties(
            self,
            symbol: str,
            props: Dict[str, Any] = None,
            allow_online: bool = True,
        ) -> Dict[str, PropertyResolutionResult]:
            """Resolve ideal-gas formation properties at 298.15 K."""
            props = self._coerce_props(symbol, props, allow_online=allow_online)
            results = {}
            provided_keys = {
                'Hf': ('Hf', 'kJ/mol'),
                'Gf': ('Gf', 'kJ/mol'),
                'S': ('S', 'J/(mol*K)'),
                'Hcomb': ('Hcomb', 'kJ/mol'),
            }
            for key, (_, units) in provided_keys.items():
                if props.get(key) is not None:
                    results[key] = self._source_result_for_value(props, key, units=units)

            perry_props = self._get_perry_formation_properties(symbol, props)
            if perry_props:
                for key, item in perry_props.items():
                    if key in results:
                        continue
                    results[key] = PropertyResolutionResult(
                        value=item.value,
                        source='local',
                        method=item.method,
                        quality=0.96,
                        notes=f"{item.source}; units {item.units}"
                    )

            online = None
            if allow_online:
                online = self._fetch_formation_online(symbol, props)
                if online:
                    sources = online.get('_sources', {})
                    notes = online.get('_notes', {})
                    for key, (_, units) in provided_keys.items():
                        if key in results or online.get(key) is None:
                            continue
                        results[key] = PropertyResolutionResult(
                            value=online[key],
                            source='online',
                            method=sources.get(key, 'nist_gas_thermochemistry'),
                            quality=0.86,
                            notes=f"{notes.get(key, 'NIST WebBook gas thermochemistry')}; units {units}"
                        )

            if 'Hf' not in results:
                derived_hf = self._derive_hf_gas_from_liquid(symbol, props, online, allow_online)
                if derived_hf:
                    results['Hf'] = derived_hf

            hf_result = results.get('Hf')
            if hf_result is not None and hf_result.value is not None:
                if 'Gf' not in results and 'S' in results:
                    derived_gf = self._derive_gf_gas_from_hf_and_s(
                        symbol,
                        props,
                        hf_result,
                        results['S'],
                    )
                    if derived_gf:
                        results['Gf'] = derived_gf

                if 'S' not in results and 'Gf' in results:
                    derived_s = self._derive_s_gas_from_hf_and_gf(
                        symbol,
                        props,
                        hf_result,
                        results['Gf'],
                    )
                    if derived_s:
                        results['S'] = derived_s

                if 'Gf' not in results:
                    derived_gf = self._derive_gf_gas_from_liquid_gf(
                        symbol,
                        props,
                        online,
                        allow_online,
                    )
                    if derived_gf:
                        results['Gf'] = derived_gf

                if 'S' not in results and 'Gf' in results:
                    derived_s = self._derive_s_gas_from_hf_and_gf(
                        symbol,
                        props,
                        hf_result,
                        results['Gf'],
                    )
                    if derived_s:
                        results['S'] = derived_s

                if 'Gf' not in results:
                    derived_liquid_gf = self._derive_gf_liquid_from_liquid_s(
                        symbol,
                        props,
                        online,
                        allow_online,
                        hf_result,
                    )
                    if derived_liquid_gf:
                        derived_gf = self._derive_gf_gas_from_liquid_gf(
                            symbol,
                            props,
                            online,
                            allow_online,
                            liquid_gf=derived_liquid_gf,
                        )
                        if derived_gf:
                            results['Gf'] = derived_gf

                if 'S' not in results and 'Gf' in results:
                    derived_s = self._derive_s_gas_from_hf_and_gf(
                        symbol,
                        props,
                        hf_result,
                        results['Gf'],
                    )
                    if derived_s:
                        results['S'] = derived_s

            # Final Hf/S estimation rung. Higher-quality direct, Perry, online,
            # and exact thermodynamic derivations above always take precedence.
            domalski_hearing = {}
            if 'Hf' not in results or 'S' not in results:
                domalski_hearing = self._resolve_domalski_hearing_formation(
                    symbol,
                    props,
                    allow_online,
                )

            if 'Hf' not in results and 'Hf' in domalski_hearing:
                results['Hf'] = domalski_hearing['Hf']

            hf_result = results.get('Hf')
            if (
                'S' not in results
                and hf_result is not None
                and hf_result.value is not None
                and results.get('Gf') is not None
                and results['Gf'].value is not None
            ):
                derived_s = self._derive_s_gas_from_hf_and_gf(
                    symbol,
                    props,
                    hf_result,
                    results['Gf'],
                )
                if derived_s:
                    results['S'] = derived_s

            if 'S' not in results and 'S' in domalski_hearing:
                results['S'] = domalski_hearing['S']

            if (
                'Gf' not in results
                and results.get('Hf') is not None
                and results['Hf'].value is not None
                and results.get('S') is not None
                and results['S'].value is not None
            ):
                derived_gf = self._derive_gf_gas_from_hf_and_s(
                    symbol,
                    props,
                    results['Hf'],
                    results['S'],
                )
                if derived_gf:
                    results['Gf'] = derived_gf

            if 'Hcomb' not in results:
                derived_hcomb = self._derive_net_hcomb_from_gross(symbol, props)
                if derived_hcomb:
                    results['Hcomb'] = derived_hcomb

            if 'Hcomb' not in results:
                derived_hcomb = self._derive_net_hcomb_from_formula(
                    symbol,
                    props,
                    online,
                    allow_online,
                    results.get('Hf'),
                )
                if derived_hcomb:
                    results['Hcomb'] = derived_hcomb

            for key in ('Hf', 'Gf', 'S', 'Hcomb'):
                results.setdefault(
                    key,
                    PropertyResolutionResult(
                        value=None,
                        source='missing',
                        method='none',
                        quality=0.0,
                        notes=f'{key} not available'
                    )
                )
            return results
