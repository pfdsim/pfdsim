from .common import *
from collections import Counter


if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..physical_constants import R_BAR_CM3_MOL_K
else:
    from physical_constants import R_BAR_CM3_MOL_K

SURFACE_TENSION_CORRELATIONS_PATH = (
    Path(__file__).resolve().parent.parent
    / 'data'
    / 'surface_tension_correlations_cas.json'
)
KNOTTS_PARACHOR_GROUPS_PATH = (
    Path(__file__).resolve().parent.parent
    / 'data'
    / 'perry_surface_tension_parachor_groups.json'
)
KNOTTS_FRAGMENTATION_CACHE_VERSION = 7
KNOTTS_PARACHOR_BASE_QUALITY = 0.80
KNOTTS_PARACHOR_LIQUID_DENSITY_TARGET_QUALITY = 0.92
KNOTTS_PARACHOR_PR_VAPOR_PENALTY = 0.02
KNOTTS_PARACHOR_ZERO_VAPOR_PENALTY = 0.10
KNOTTS_PARACHOR_MISSING_TC_QUALITY = 0.50
KNOTTS_MAX_REDUCED_TEMPERATURE = 0.90
KNOTTS_EOS_MIN_TC_PC_QUALITY = 0.80
KNOTTS_EOS_MIN_OMEGA_QUALITY = 0.70
KNOTTS_ZERO_VAPOR_DENSITY_QUALITY = 0.60

SURFACE_TENSION_METHOD_PRIORITY = (
    'mulero_cachadina_refprop',
    'somayajulu_revised',
    'vdi_ppds_11_dippr_eq106',
    'jasper_lange',
)

SURFACE_TENSION_METHOD_QUALITY = {
    'mulero_cachadina_refprop': 0.98,
    'somayajulu_revised': 0.96,
    'vdi_ppds_11_dippr_eq106': 0.95,
    'jasper_lange': 0.90,
    'somayajulu': 0.90,
}

_SURFACE_TENSION_CORRELATIONS = None
_KNOTTS_PARACHOR_GROUPS = None
_KNOTTS_FRAGMENTATION_CACHE_MISS = object()
_KNOTTS_FRAGMENTATION_REJECTED = object()


class SurfaceTensionMixin:
        def resolve_surface_tension(
            self,
            symbol: str,
            T: float,
            props: Dict[str, Any] = None,
        ) -> PropertyResolutionResult:
            """Resolve pure-component liquid-vapor/air surface tension in N/m."""
            T = self._normalize_surface_tension_temperature(T)
            props = self._coerce_props(symbol, props)

            provided_fit = self._provided_surface_tension(symbol, props, T)
            if provided_fit:
                return provided_fit

            local_fit = self._surface_tension_from_local_correlations(symbol, props, T)
            if local_fit:
                return local_fit

            parachor = self._knotts_parachor_surface_tension(symbol, props, T)
            if parachor:
                return parachor

            raise PropertyResolutionError(
                f"Cannot determine surface tension for '{symbol}' at T={T:.1f}K."
            )


        @staticmethod
        def _normalize_surface_tension_temperature(T: float) -> float:
            try:
                value = float(T)
            except (TypeError, ValueError) as exc:
                raise PropertyResolutionError(
                    "Surface-tension temperature must be a positive finite value in K."
                ) from exc
            if value <= 0.0 or not math.isfinite(value):
                raise PropertyResolutionError(
                    "Surface-tension temperature must be a positive finite value in K."
                )
            return value


        @staticmethod
        def _positive_surface_tension(value: Optional[float]) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            if 0.0 < value_f <= 0.25 and math.isfinite(value_f):
                return value_f
            return None


        @staticmethod
        def _surface_tension_data() -> Dict[str, Any]:
            global _SURFACE_TENSION_CORRELATIONS
            if _SURFACE_TENSION_CORRELATIONS is not None:
                return _SURFACE_TENSION_CORRELATIONS
            try:
                payload = json.loads(SURFACE_TENSION_CORRELATIONS_PATH.read_text())
            except (OSError, ValueError, TypeError):
                payload = {}
            chemicals = payload.get('chemicals') if isinstance(payload, dict) else None
            if not isinstance(chemicals, dict):
                chemicals = {}
            _SURFACE_TENSION_CORRELATIONS = chemicals
            return _SURFACE_TENSION_CORRELATIONS


        @staticmethod
        def _knotts_group_contributions() -> Optional[Dict[str, float]]:
            global _KNOTTS_PARACHOR_GROUPS
            if _KNOTTS_PARACHOR_GROUPS is not None:
                return _KNOTTS_PARACHOR_GROUPS
            try:
                payload = json.loads(KNOTTS_PARACHOR_GROUPS_PATH.read_text())
                groups = payload.get('groups')
                if not isinstance(groups, dict):
                    return None
                _KNOTTS_PARACHOR_GROUPS = {
                    str(name): float(value) for name, value in groups.items()
                }
            except (OSError, ValueError, TypeError):
                return None
            return _KNOTTS_PARACHOR_GROUPS


        @staticmethod
        def _surface_tension_explicit_smiles(props: Dict[str, Any]) -> Optional[str]:
            for key in (
                'smiles', 'SMILES', 'canonical_smiles', 'CanonicalSMILES',
                'connectivity_smiles', 'ConnectivitySMILES',
            ):
                value = props.get(key)
                if value:
                    return str(value).strip()
            return None


        def _surface_tension_structure(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[tuple[str, PropertyResolutionResult]]:
            explicit = self._surface_tension_explicit_smiles(props)
            if explicit:
                result = self._source_result_for_value(
                    props,
                    'smiles',
                    value=explicit,
                    default_source='provided',
                    default_method='provided_smiles',
                    default_quality=1.0,
                )
                return explicit, result

            for candidate in self._identifier_candidates(symbol, props):
                resolved = self._resolve_smiles_result(str(candidate), props, allow_online=True)
                if resolved and resolved.value:
                    return str(resolved.value), resolved
            return None


        @staticmethod
        def _knotts_fragmentation_cache_key(smiles: str) -> str:
            digest = hashlib.sha256(str(smiles).strip().encode('utf-8')).hexdigest()
            return (
                f'knotts_parachor_fragmentation_v'
                f'{KNOTTS_FRAGMENTATION_CACHE_VERSION}_{digest}'
            )


        def _load_knotts_fragmentation_cache(self, smiles: str):
            cache = getattr(self, '_knotts_fragmentation_cache', None)
            if cache is None:
                cache = {}
                self._knotts_fragmentation_cache = cache
            key = str(smiles).strip()
            if key in cache:
                return cache[key]
            try:
                payload = self._get_cache(
                    self._knotts_fragmentation_cache_key(key)
                )
            except (OSError, ValueError, TypeError):
                return _KNOTTS_FRAGMENTATION_CACHE_MISS
            if (
                not isinstance(payload, dict)
                or payload.get('version') != KNOTTS_FRAGMENTATION_CACHE_VERSION
                or payload.get('smiles') != key
            ):
                return _KNOTTS_FRAGMENTATION_CACHE_MISS
            if payload.get('status') == 'rejected':
                cache[key] = _KNOTTS_FRAGMENTATION_REJECTED
                return _KNOTTS_FRAGMENTATION_REJECTED
            if payload.get('status') != 'mapped':
                return _KNOTTS_FRAGMENTATION_CACHE_MISS
            try:
                groups = Counter({
                    str(name): int(count)
                    for name, count in payload.get('groups', {}).items()
                    if int(count) > 0
                })
                note = str(payload['mapping_note'])
            except (KeyError, TypeError, ValueError):
                return _KNOTTS_FRAGMENTATION_CACHE_MISS
            if not groups:
                return _KNOTTS_FRAGMENTATION_CACHE_MISS
            result = groups, note
            cache[key] = result
            return result


        def _save_knotts_fragmentation_cache(
            self,
            smiles: str,
            mapped: Optional[tuple[Counter, str]],
        ) -> None:
            cache = getattr(self, '_knotts_fragmentation_cache', None)
            if cache is None:
                cache = {}
                self._knotts_fragmentation_cache = cache
            key = str(smiles).strip()
            if mapped is None:
                cache[key] = _KNOTTS_FRAGMENTATION_REJECTED
                payload = {
                    'version': KNOTTS_FRAGMENTATION_CACHE_VERSION,
                    'smiles': key,
                    'status': 'rejected',
                }
            else:
                groups, note = mapped
                stored = Counter({str(name): int(count) for name, count in groups.items()})
                cache[key] = stored, str(note)
                payload = {
                    'version': KNOTTS_FRAGMENTATION_CACHE_VERSION,
                    'smiles': key,
                    'status': 'mapped',
                    'groups': dict(sorted(stored.items())),
                    'mapping_note': str(note),
                }
            try:
                self._set_cache(
                    self._knotts_fragmentation_cache_key(key),
                    payload,
                )
            except (OSError, ValueError, TypeError):
                pass


        @staticmethod
        def _knotts_carbon_neighbors(atom) -> list[Any]:
            return [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C']


        @staticmethod
        def _knotts_carbonyl_oxygen(atom):
            for bond in atom.GetBonds():
                neighbor = bond.GetOtherAtom(atom)
                if neighbor.GetSymbol() == 'O' and bond.GetBondTypeAsDouble() == 2.0:
                    return neighbor
            return None


        @staticmethod
        def _knotts_add(groups: Counter, name: str, count: int = 1) -> None:
            groups[name] += int(count)


        def _knotts_fragment_molecule(
            self,
            mol,
        ) -> Optional[tuple[Counter, str]]:
            if mol is None:
                return None
            groups = Counter()
            consumed = set()
            notes = []

            atoms = list(mol.GetAtoms())
            # Knotts' functional-group contributions describe carbon-based
            # compounds. Without a carbon skeleton, labels such as nitroso or
            # halogen would misclassify elemental and inorganic molecules as
            # organic functional groups.
            if not any(atom.GetSymbol() == 'C' for atom in atoms):
                return None
            halogens = {'F', 'Cl', 'Br', 'I'}
            if any(
                sum(neighbor.GetSymbol() in halogens for neighbor in atom.GetNeighbors()) >= 2
                for atom in atoms
            ):
                return None
            ring_info = mol.GetRingInfo()
            atom_rings = tuple(tuple(ring) for ring in ring_info.AtomRings())

            def consume(*selected) -> None:
                consumed.update(atom.GetIdx() for atom in selected if atom is not None)

            def carbonyl_atoms(carbon):
                oxygen = self._knotts_carbonyl_oxygen(carbon)
                single_hetero = [
                    bond.GetOtherAtom(carbon)
                    for bond in carbon.GetBonds()
                    if bond.GetBondTypeAsDouble() == 1.0
                    and bond.GetOtherAtom(carbon).GetSymbol() in {'O', 'N'}
                ]
                return oxygen, single_hetero

            # Acid anhydrides must be claimed before ordinary ester/carboxyl groups.
            for oxygen in atoms:
                if oxygen.GetSymbol() != 'O' or oxygen.GetIdx() in consumed:
                    continue
                carbonyls = [
                    neighbor for neighbor in oxygen.GetNeighbors()
                    if neighbor.GetSymbol() == 'C'
                    and self._knotts_carbonyl_oxygen(neighbor) is not None
                ]
                if len(carbonyls) != 2:
                    continue
                terminal_oxygens = [self._knotts_carbonyl_oxygen(carbon) for carbon in carbonyls]
                self._knotts_add(groups, 'acid_anhydride')
                consume(oxygen, *carbonyls, *terminal_oxygens)

            # Combined carbonyl/heteroatom groups.
            for carbon in atoms:
                if carbon.GetSymbol() != 'C' or carbon.GetIdx() in consumed:
                    continue
                oxygen, single_hetero = carbonyl_atoms(carbon)
                if oxygen is None:
                    continue
                nitrogen = next((atom for atom in single_hetero if atom.GetSymbol() == 'N'), None)
                if nitrogen is not None:
                    hydrogens = nitrogen.GetTotalNumHs()
                    formyl = carbon.GetTotalNumHs() > 0
                    if formyl and hydrogens == 1:
                        name = 'formamide_secondary'
                    elif formyl and hydrogens == 0:
                        name = 'formamide_tertiary'
                    elif hydrogens >= 2:
                        name = 'amide_primary'
                    elif hydrogens == 1:
                        name = 'amide_secondary'
                    else:
                        name = 'amide_tertiary'
                    self._knotts_add(groups, name)
                    consume(carbon, oxygen, nitrogen)
                    continue

                single_oxygen = next((atom for atom in single_hetero if atom.GetSymbol() == 'O'), None)
                if single_oxygen is not None:
                    if single_oxygen.GetTotalNumHs() > 0:
                        name = 'formic_acid' if carbon.GetTotalNumHs() > 0 else 'carboxylic_acid'
                    else:
                        same_ring = any(
                            carbon.GetIdx() in ring and single_oxygen.GetIdx() in ring
                            for ring in atom_rings
                        )
                        if same_ring:
                            name = 'lactone'
                        elif carbon.GetTotalNumHs() > 0:
                            name = 'formate'
                        else:
                            name = 'ester'
                    self._knotts_add(groups, name)
                    consume(carbon, oxygen, single_oxygen)
                    continue

                if carbon.GetTotalNumHs() > 0:
                    name = 'aldehyde'
                else:
                    name = 'ketone_ring' if carbon.IsInRing() else 'ketone_nonring'
                self._knotts_add(groups, name)
                consume(carbon, oxygen)

            # Nitrile and nitro/nitroso groups.
            for nitrogen in atoms:
                if nitrogen.GetSymbol() != 'N' or nitrogen.GetIdx() in consumed:
                    continue
                triple_carbon = next(
                    (
                        bond.GetOtherAtom(nitrogen)
                        for bond in nitrogen.GetBonds()
                        if bond.GetBondTypeAsDouble() == 3.0
                        and bond.GetOtherAtom(nitrogen).GetSymbol() == 'C'
                    ),
                    None,
                )
                if triple_carbon is not None:
                    carbon_neighbors = [
                        neighbor for neighbor in triple_carbon.GetNeighbors()
                        if neighbor.GetIdx() != nitrogen.GetIdx()
                    ]
                    if not carbon_neighbors and triple_carbon.GetTotalNumHs() > 0:
                        name = 'hydrogen_cyanide'
                    elif any(neighbor.GetIsAromatic() for neighbor in carbon_neighbors):
                        name = 'aromatic_nitrile'
                    else:
                        name = 'nitrile'
                    self._knotts_add(groups, name)
                    consume(nitrogen, triple_carbon)
                    continue

                oxygen_neighbors = [
                    neighbor for neighbor in nitrogen.GetNeighbors()
                    if neighbor.GetSymbol() == 'O'
                ]
                if oxygen_neighbors:
                    carbon_neighbors = self._knotts_carbon_neighbors(nitrogen)
                    aromatic = any(atom.GetIsAromatic() for atom in carbon_neighbors)
                    if len(oxygen_neighbors) >= 2:
                        name = 'aromatic_nitro' if aromatic else 'nitro'
                    elif len(oxygen_neighbors) == 1:
                        name = 'nitroso'
                    else:
                        return None
                    self._knotts_add(groups, name)
                    consume(nitrogen, *oxygen_neighbors)

            # Oxygen groups not consumed as part of a carbonyl function.
            alcohol_count = 0
            for oxygen in atoms:
                if oxygen.GetSymbol() != 'O' or oxygen.GetIdx() in consumed:
                    continue
                if any(
                    neighbor.GetSymbol() in {'S', 'P', 'Cl'}
                    for neighbor in oxygen.GetNeighbors()
                ):
                    continue
                carbon_neighbors = self._knotts_carbon_neighbors(oxygen)
                if oxygen.GetTotalNumHs() > 0 and len(carbon_neighbors) == 1:
                    carbon = carbon_neighbors[0]
                    if carbon.GetIsAromatic():
                        name = 'phenol_oh'
                    else:
                        carbon_degree = len(self._knotts_carbon_neighbors(carbon))
                        if carbon_degree <= 1:
                            name = 'alcohol_primary_oh'
                        elif carbon_degree == 2:
                            name = 'alcohol_secondary_oh'
                        else:
                            name = 'alcohol_tertiary_oh'
                    alcohol_count += 1
                elif len(carbon_neighbors) == 2:
                    if oxygen.IsInRing():
                        name = 'ether_o_ring'
                    elif any(carbon.GetIsAromatic() for carbon in carbon_neighbors):
                        name = 'ether_o_aromatic'
                    else:
                        name = 'ether_o_nonring'
                else:
                    return None
                self._knotts_add(groups, name)
                consume(oxygen)
            if alcohol_count > 1:
                notes.append('polyfunctional alcohol')

            # Remaining nitrogen atoms are amines, imines, or aromatic nitrogen.
            for nitrogen in atoms:
                if nitrogen.GetSymbol() != 'N' or nitrogen.GetIdx() in consumed:
                    continue
                if nitrogen.GetIsAromatic():
                    name = (
                        'amine_secondary_aromatic_ring'
                        if nitrogen.GetTotalNumHs() > 0 else 'aromatic_n'
                    )
                elif any(bond.GetBondTypeAsDouble() == 2.0 for bond in nitrogen.GetBonds()):
                    name = 'imine_n'
                else:
                    hydrogens = nitrogen.GetTotalNumHs()
                    carbon_neighbors = self._knotts_carbon_neighbors(nitrogen)
                    if hydrogens >= 2 and len(carbon_neighbors) == 1:
                        carbon = carbon_neighbors[0]
                        if carbon.GetIsAromatic():
                            name = 'amine_primary_aromatic'
                        else:
                            degree = len(self._knotts_carbon_neighbors(carbon))
                            if degree <= 1:
                                name = 'amine_primary_primary_r'
                            elif degree == 2:
                                name = 'amine_primary_secondary_r'
                            else:
                                name = 'amine_primary_tertiary_r'
                    elif hydrogens == 1:
                        name = (
                            'amine_secondary_ring'
                            if nitrogen.IsInRing() else 'amine_secondary_nonring'
                        )
                    elif hydrogens == 0:
                        name = (
                            'amine_tertiary_ring'
                            if nitrogen.IsInRing() else 'amine_tertiary_nonring'
                        )
                    else:
                        return None
                self._knotts_add(groups, name)
                consume(nitrogen)

            # Sulfur functions.
            for sulfur in atoms:
                if sulfur.GetSymbol() != 'S' or sulfur.GetIdx() in consumed:
                    continue
                oxygen_neighbors = [atom for atom in sulfur.GetNeighbors() if atom.GetSymbol() == 'O']
                carbon_neighbors = self._knotts_carbon_neighbors(sulfur)
                if sulfur.GetTotalNumHs() > 0 and len(carbon_neighbors) == 1:
                    carbon = carbon_neighbors[0]
                    if carbon.GetIsAromatic():
                        name = 'thiol_aromatic'
                    else:
                        degree = len(self._knotts_carbon_neighbors(carbon))
                        if degree <= 1:
                            name = 'thiol_primary_r'
                        elif degree == 2:
                            name = 'thiol_secondary_r'
                        else:
                            name = 'thiol_tertiary_r'
                elif len(oxygen_neighbors) >= 2:
                    name = 'sulfone_ring' if sulfur.IsInRing() else 'sulfone_nonring'
                elif len(oxygen_neighbors) == 1:
                    name = 'sulfoxide_nonring'
                elif len(carbon_neighbors) == 2:
                    if sulfur.IsInRing():
                        name = 'thioether_ring'
                    elif any(carbon.GetIsAromatic() for carbon in carbon_neighbors):
                        name = 'thioether_aromatic'
                    else:
                        name = 'thioether_nonring'
                else:
                    return None
                self._knotts_add(groups, name)
                consume(sulfur, *oxygen_neighbors)

            # Halogens and the remaining supported inorganic centers.
            halogen_names = {
                'F': ('fluorine', 'aromatic_fluorine'),
                'Cl': ('chlorine', 'aromatic_chlorine'),
                'Br': ('bromine', 'aromatic_bromine'),
                'I': ('iodine', 'aromatic_iodine'),
            }
            for atom in atoms:
                if atom.GetIdx() in consumed:
                    continue
                symbol = atom.GetSymbol()
                if symbol in halogen_names:
                    if symbol == 'Cl' and sum(
                        neighbor.GetSymbol() == 'O' for neighbor in atom.GetNeighbors()
                    ) >= 3:
                        name = 'chlorate'
                        consume(*[neighbor for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O'])
                    else:
                        attached = next(iter(atom.GetNeighbors()), None)
                        aromatic = bool(attached is not None and attached.GetIsAromatic())
                        name = halogen_names[symbol][1 if aromatic else 0]
                    self._knotts_add(groups, name)
                    consume(atom)
                elif symbol == 'Si':
                    hydrogens = atom.GetTotalNumHs()
                    if hydrogens == 4:
                        name = 'silane'
                    elif hydrogens > 0:
                        name = 'silicon_h'
                    else:
                        name = 'silicon_ring' if atom.IsInRing() else 'silicon'
                    self._knotts_add(groups, name)
                    consume(atom)
                elif symbol == 'P':
                    oxygen_neighbors = [
                        neighbor for neighbor in atom.GetNeighbors()
                        if neighbor.GetSymbol() == 'O'
                    ]
                    name = (
                        'phosphate'
                        if len(oxygen_neighbors) >= 4
                        else 'phosphorus'
                    )
                    self._knotts_add(groups, name)
                    consume(atom, *oxygen_neighbors)
                elif symbol == 'B':
                    self._knotts_add(groups, 'boron')
                    consume(atom)
                elif symbol == 'Al':
                    self._knotts_add(groups, 'aluminum')
                    consume(atom)

            # Carbon atom groups after functional carbon atoms have been consumed.
            remaining_ch2 = [
                atom for atom in atoms
                if atom.GetSymbol() == 'C'
                and atom.GetIdx() not in consumed
                and not atom.IsInRing()
                and not atom.GetIsAromatic()
                and atom.GetTotalNumHs() == 2
                and all(bond.GetBondTypeAsDouble() == 1.0 for bond in atom.GetBonds())
            ]
            if len(remaining_ch2) <= 11:
                ch2_name = 'nonring_ch2_1_11'
            elif len(remaining_ch2) <= 20:
                ch2_name = 'nonring_ch2_12_20'
            else:
                ch2_name = 'nonring_ch2_gt20'

            for carbon in atoms:
                if carbon.GetSymbol() != 'C' or carbon.GetIdx() in consumed:
                    continue
                hydrogens = carbon.GetTotalNumHs()
                carbon_bonds = [
                    bond for bond in carbon.GetBonds()
                    if bond.GetOtherAtom(carbon).GetSymbol() == 'C'
                ]
                if carbon.GetIsAromatic():
                    containing_rings = [ring for ring in atom_rings if carbon.GetIdx() in ring]
                    if len(containing_rings) > 1:
                        all_aromatic = all(
                            all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
                            for ring in containing_rings
                        )
                        name = (
                            'fused_aromatic_aromatic_c'
                            if all_aromatic else 'fused_aromatic_aliphatic_c'
                        )
                        notes.append('fused aromatic classification')
                    else:
                        name = 'aromatic_ch' if hydrogens else 'aromatic_c'
                elif carbon.IsInRing():
                    containing_rings = [ring for ring in atom_rings if carbon.GetIdx() in ring]
                    if len(containing_rings) > 1:
                        if hydrogens != 1:
                            return None
                        name = 'fused_ring_ch'
                        notes.append('fused aliphatic ring classification')
                    elif any(bond.GetBondTypeAsDouble() == 2.0 for bond in carbon_bonds):
                        name = 'ring_alkene_ch' if hydrogens else 'ring_alkene_c'
                    else:
                        name = {2: 'ring_ch2', 1: 'ring_ch', 0: 'ring_c'}.get(hydrogens)
                else:
                    triple_bonds = [bond for bond in carbon_bonds if bond.GetBondTypeAsDouble() == 3.0]
                    double_bonds = [bond for bond in carbon_bonds if bond.GetBondTypeAsDouble() == 2.0]
                    if triple_bonds:
                        name = 'alkyne_ch' if hydrogens else 'alkyne_c'
                    elif len(double_bonds) >= 2:
                        name = 'cumulated_c'
                    elif len(double_bonds) == 1:
                        name = {2: 'alkene_ch2', 1: 'alkene_ch', 0: 'alkene_c'}.get(hydrogens)
                    else:
                        name = {
                            3: 'nonring_ch3',
                            2: ch2_name,
                            1: 'nonring_ch',
                            0: 'nonring_c',
                        }.get(hydrogens)
                if name is None:
                    return None
                self._knotts_add(groups, name)
                consume(carbon)

            # One correction for each isolated nonaromatic carbon ring.
            for ring in atom_rings:
                if not 3 <= len(ring) <= 7:
                    continue
                ring_atoms = [mol.GetAtomWithIdx(index) for index in ring]
                if all(atom.GetSymbol() == 'C' and not atom.GetIsAromatic() for atom in ring_atoms):
                    self._knotts_add(groups, f'ring_{len(ring)}_correction')

            # A degree-three branch point is secondary branching and contributes
            # one branch; degree four is tertiary branching and contributes two.
            branch_centers = {}
            for atom in atoms:
                if atom.GetSymbol() != 'C' or atom.IsInRing():
                    continue
                carbon_degree = len(self._knotts_carbon_neighbors(atom))
                if carbon_degree == 3:
                    branch_centers[atom.GetIdx()] = 'secondary'
                elif carbon_degree == 4:
                    branch_centers[atom.GetIdx()] = 'tertiary'
            branch_count = sum(
                1 if kind == 'secondary' else 2
                for kind in branch_centers.values()
            )
            if branch_count:
                self._knotts_add(groups, 'branch', branch_count)
                adjacency_counts = Counter()
                for bond in mol.GetBonds():
                    begin = bond.GetBeginAtomIdx()
                    end = bond.GetEndAtomIdx()
                    if begin not in branch_centers or end not in branch_centers:
                        continue
                    pair = tuple(sorted((branch_centers[begin], branch_centers[end])))
                    adjacency_name = {
                        ('secondary', 'secondary'): 'secondary_secondary_adjacency',
                        ('secondary', 'tertiary'): 'secondary_tertiary_adjacency',
                        ('tertiary', 'tertiary'): 'tertiary_tertiary_adjacency',
                    }[pair]
                    adjacency_counts[adjacency_name] += 1
                for adjacency_name, count in adjacency_counts.items():
                    self._knotts_add(groups, adjacency_name, count)
                if adjacency_counts:
                    notes.append('branch and branch-adjacency corrections')
                else:
                    notes.append('branch correction')

            # Ortho/meta/para correction for a simple disubstituted benzene ring.
            for ring in atom_rings:
                if len(ring) != 6:
                    continue
                ring_set = set(ring)
                ring_atoms = [mol.GetAtomWithIdx(index) for index in ring]
                if not all(atom.GetSymbol() == 'C' and atom.GetIsAromatic() for atom in ring_atoms):
                    continue
                if any(
                    sum(atom.GetIdx() in candidate for candidate in atom_rings) > 1
                    for atom in ring_atoms
                ):
                    continue
                substituted = [
                    offset for offset, atom in enumerate(ring_atoms)
                    if any(neighbor.GetIdx() not in ring_set for neighbor in atom.GetNeighbors())
                ]
                if len(substituted) == 2:
                    distance = abs(substituted[0] - substituted[1])
                    distance = min(distance, 6 - distance)
                    correction = {
                        1: 'aromatic_ortho_correction',
                        2: 'aromatic_meta_correction',
                        3: 'aromatic_para_correction',
                    }.get(distance)
                    if correction:
                        self._knotts_add(groups, correction)

            aromatic_carbons = {
                atom.GetIdx() for atom in atoms
                if atom.GetSymbol() == 'C' and atom.GetIsAromatic()
            }
            aromatic_components = []
            remaining_aromatic = set(aromatic_carbons)
            while remaining_aromatic:
                seed = remaining_aromatic.pop()
                component = {seed}
                stack = [seed]
                while stack:
                    index = stack.pop()
                    atom = mol.GetAtomWithIdx(index)
                    for neighbor in atom.GetNeighbors():
                        neighbor_index = neighbor.GetIdx()
                        if neighbor_index in remaining_aromatic:
                            remaining_aromatic.remove(neighbor_index)
                            component.add(neighbor_index)
                            stack.append(neighbor_index)
                aromatic_components.append(component)
            for component in aromatic_components:
                if len(component) != 10:
                    continue
                ring_count = sum(set(ring).issubset(component) for ring in atom_rings)
                substituted = any(
                    any(neighbor.GetIdx() not in component for neighbor in mol.GetAtomWithIdx(index).GetNeighbors())
                    for index in component
                )
                if ring_count >= 2 and substituted:
                    self._knotts_add(groups, 'substituted_naphthalene_correction')

            if any(atom.GetIdx() not in consumed and atom.GetAtomicNum() > 1 for atom in atoms):
                return None
            if not groups:
                return None
            note = ', '.join(sorted(set(notes))) or 'direct RDKit Knotts fragmentation'
            return groups, note


        def _knotts_fragmentation(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[tuple[Counter, PropertyResolutionResult, str]]:
            structure = self._surface_tension_structure(symbol, props)
            if structure is None:
                return None
            smiles, structure_result = structure
            mapped = self._load_knotts_fragmentation_cache(smiles)
            if mapped is _KNOTTS_FRAGMENTATION_REJECTED:
                return None
            if mapped is _KNOTTS_FRAGMENTATION_CACHE_MISS:
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(smiles)
                except Exception:
                    return None
                mapped = self._knotts_fragment_molecule(mol)
                self._save_knotts_fragmentation_cache(smiles, mapped)
                if mapped is None:
                    return None
            groups, note = mapped
            return groups, structure_result, note


        def _knotts_pr_vapor_density(
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
                volume_m3_kmol = max(roots) * R_BAR_CM3_MOL_K * T / P_bar / 1000.0
                if volume_m3_kmol <= 0.0 or not math.isfinite(volume_m3_kmol):
                    return None
                density = 1.0 / volume_m3_kmol
                return density if density > 0.0 and math.isfinite(density) else None
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None


        def _knotts_ptv_vapor_density(
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
                volume_m3_kmol = max(roots) * R_BAR_CM3_MOL_K * T / P_bar / 1000.0
                if volume_m3_kmol <= 0.0 or not math.isfinite(volume_m3_kmol):
                    return None
                density = 1.0 / volume_m3_kmol
                return density if density > 0.0 and math.isfinite(density) else None
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None


        def _knotts_saturated_vapor_density(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
            critical: Dict[str, PropertyResolutionResult],
        ) -> PropertyResolutionResult:
            def zero_fallback(reason: str) -> PropertyResolutionResult:
                return PropertyResolutionResult(
                    value=0.0,
                    source='estimated',
                    method='zero_saturated_vapor_density_fallback',
                    quality=KNOTTS_ZERO_VAPOR_DENSITY_QUALITY,
                    notes=(
                        'Saturated vapor molar density assumed negligible for the '
                        f'Knotts Parachor estimate because {reason}; units kmol/m^3'
                    ),
                )

            tc_result = self._critical_value_result(critical, 'Tc')
            pc_result = self._critical_value_result(critical, 'Pc')
            omega_result = self._critical_value_result(critical, 'omega')
            zc_result = self._critical_value_result(critical, 'Zc')
            eos_inputs_usable = bool(
                tc_result
                and pc_result
                and omega_result
                and self._result_quality(tc_result, 0.0) >= KNOTTS_EOS_MIN_TC_PC_QUALITY
                and self._result_quality(pc_result, 0.0) >= KNOTTS_EOS_MIN_TC_PC_QUALITY
                and self._result_quality(omega_result, 0.0) >= KNOTTS_EOS_MIN_OMEGA_QUALITY
            )
            if not eos_inputs_usable:
                return zero_fallback(
                    'Tc/Pc are unavailable or below quality 0.80, or omega is '
                    'unavailable or below quality 0.70'
                )
            try:
                psat_result = self.resolve_vapor_pressure(symbol, T, props)
            except Exception:
                return zero_fallback('saturation pressure could not be resolved')
            if psat_result is None or psat_result.value is None:
                return zero_fallback('saturation pressure could not be resolved')
            try:
                P_bar = max(float(psat_result.value), MIN_EOS_PRESSURE_BAR)
            except (TypeError, ValueError):
                return zero_fallback('saturation pressure was invalid')

            use_ptv = bool(
                zc_result
                and self._result_quality(zc_result, 0.0) >= SOFT_PROPERTY_QUALITY_THRESHOLD
                and not self._result_is_soft(zc_result)
            )
            density = None
            method = ''
            inputs = [tc_result, pc_result, omega_result, psat_result]
            factor = 0.75
            if use_ptv:
                density = self._knotts_ptv_vapor_density(
                    T,
                    P_bar,
                    tc_result.value,
                    pc_result.value,
                    omega_result.value,
                    zc_result.value,
                )
                if density is not None:
                    method = 'ptv_eos_saturated_vapor_density'
                    inputs.append(zc_result)
                    factor = 0.84
            if density is None:
                density = self._knotts_pr_vapor_density(
                    T,
                    P_bar,
                    tc_result.value,
                    pc_result.value,
                    omega_result.value,
                )
                method = 'pr_eos_saturated_vapor_density'
            if density is None:
                return zero_fallback('both the PTV and PR vapor roots were unavailable')
            return PropertyResolutionResult(
                value=density,
                source='calculated',
                method=method,
                quality=self._combine_quality(inputs, method_factor=factor),
                notes=(
                    f"Saturated vapor molar density at Psat={P_bar:g} bar from "
                    f"{psat_result.source}/{psat_result.method}; units kmol/m^3"
                ),
            )


        def _knotts_parachor_surface_tension(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
        ) -> Optional[PropertyResolutionResult]:
            try:
                critical = self.resolve_critical_properties(
                    symbol,
                    props,
                    allow_online=True,
                    allow_estimation=True,
                )
            except Exception:
                critical = {}
            tc_result = self._critical_value_result(critical, 'Tc')
            reduced_temperature = None
            if tc_result is not None:
                try:
                    reduced_temperature = float(T) / float(tc_result.value)
                except (TypeError, ValueError, ZeroDivisionError):
                    reduced_temperature = None
                if reduced_temperature is not None and not (
                    0.0 < reduced_temperature < KNOTTS_MAX_REDUCED_TEMPERATURE
                ):
                    return None

            try:
                liquid_density = self.resolve_liquid_molar_density(symbol, T, props)
            except Exception:
                return None
            vapor_density = self._knotts_saturated_vapor_density(
                symbol,
                props,
                T,
                critical,
            )
            fragmented = self._knotts_fragmentation(symbol, props)
            contributions = self._knotts_group_contributions()
            if vapor_density is None or fragmented is None or contributions is None:
                return None
            groups, _structure_result, mapping_note = fragmented
            try:
                rho_l = float(liquid_density.value)
                rho_v = float(vapor_density.value)
            except (TypeError, ValueError):
                return None
            density_difference = rho_l - rho_v
            if density_difference <= 0.0 or not math.isfinite(density_difference):
                return None

            parachor = 0.0
            for group, count in groups.items():
                contribution = contributions.get(group)
                if contribution is None:
                    return None
                parachor += int(count) * float(contribution)
            if parachor <= 0.0 or not math.isfinite(parachor):
                return None
            try:
                value = (parachor * density_difference / 1000.0) ** 4 / 1000.0
            except (OverflowError, ValueError):
                return None
            value = self._positive_surface_tension(value)
            if value is None:
                return None

            quality = self._knotts_parachor_quality(
                liquid_density,
                vapor_density,
                tc_result,
            )
            group_note = ', '.join(
                f'{count} {name}' for name, count in sorted(groups.items())
            )
            reduced_temperature_note = (
                f'Tr={reduced_temperature:.3f}'
                if reduced_temperature is not None
                else 'Tr unavailable'
            )
            vapor_density_note = f'vapor density from {vapor_density.method}'
            if vapor_density.method == 'zero_saturated_vapor_density_fallback':
                vapor_density_note += f' ({vapor_density.notes})'
            return PropertyResolutionResult(
                value=value,
                source='estimated',
                method='knotts_parachor_group_contribution',
                quality=quality,
                notes=(
                    f"Knotts Parachor group-contribution estimate ({reduced_temperature_note}, "
                    f"P={parachor:.4g}, rhoL={rho_l:.6g}, rhoV={rho_v:.6g} kmol/m^3); "
                    f"{mapping_note}; {vapor_density_note}; "
                    f"liquid density from {liquid_density.source}/{liquid_density.method}; "
                    f"groups: {group_note}; units N/m"
                ),
            )


        def _knotts_parachor_quality(
            self,
            liquid_density: PropertyResolutionResult,
            vapor_density: PropertyResolutionResult,
            tc_result: Optional[PropertyResolutionResult],
        ) -> float:
            if tc_result is None:
                return KNOTTS_PARACHOR_MISSING_TC_QUALITY

            liquid_quality = self._result_quality(liquid_density, 0.0)
            quality = KNOTTS_PARACHOR_BASE_QUALITY - max(
                0.0,
                KNOTTS_PARACHOR_LIQUID_DENSITY_TARGET_QUALITY - liquid_quality,
            )
            if vapor_density.method == 'pr_eos_saturated_vapor_density':
                quality -= KNOTTS_PARACHOR_PR_VAPOR_PENALTY
            elif vapor_density.method == 'zero_saturated_vapor_density_fallback':
                quality -= KNOTTS_PARACHOR_ZERO_VAPOR_PENALTY
            return self._clamp_quality(quality)


        @staticmethod
        def _looks_like_cas_identifier(identifier: Any) -> bool:
            return bool(re.fullmatch(r'\d{2,7}-\d{2}-\d', str(identifier or '').strip()))


        def _surface_tension_cas_candidates(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> list[str]:
            candidates = []
            seen = set()

            def add(value: Any) -> None:
                if not value:
                    return
                text = str(value).strip()
                if self._looks_like_cas_identifier(text) and text not in seen:
                    seen.add(text)
                    candidates.append(text)

            for key in ('CAS', 'cas'):
                add(props.get(key))
            for candidate in self._identifier_candidates(symbol, props):
                add(candidate)

            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import get_compound_identity_resolver
                else:
                    from compound_identity import get_compound_identity_resolver
                identity_resolver = get_compound_identity_resolver()
                for candidate in self._identifier_candidates(symbol, props):
                    add(identity_resolver.resolve_cas(str(candidate), allow_formula=False))
            except Exception:
                pass
            return candidates


        def _provided_surface_tension(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
        ) -> Optional[PropertyResolutionResult]:
            for key in ('sigma', 'surface_tension'):
                correlation = self._correlation_for(props, key)
                if not correlation:
                    continue
                if not self._correlation_in_range(correlation, T):
                    continue
                shared_evaluation = self._evaluate_provided_correlation(props, key, T)
                if shared_evaluation:
                    value, correlation = shared_evaluation
                else:
                    value = self._evaluate_surface_tension_correlation(correlation, T)
                value = self._positive_surface_tension(value)
                if value is None:
                    continue
                return self._provided_correlation_result(
                    value,
                    correlation,
                    'provided_surface_tension_fit',
                    'surface tension in N/m',
                    default_quality=0.96,
                )

            for key in ('sigma', 'surface_tension', 'surface_tension_N_per_m'):
                result = self._source_result_for_value(
                    props,
                    key,
                    units='N/m',
                    default_method='constant_surface_tension',
                    default_quality=0.90,
                )
                if result is None:
                    continue
                value = self._positive_surface_tension(result.value)
                if value is None:
                    continue
                return PropertyResolutionResult(
                    value=value,
                    source=result.source,
                    method=result.method,
                    quality=result.quality,
                    notes=result.notes,
                )
            return None


        def _surface_tension_from_local_correlations(
            self,
            symbol: str,
            props: Dict[str, Any],
            T: float,
        ) -> Optional[PropertyResolutionResult]:
            data = self._surface_tension_data()
            for cas in self._surface_tension_cas_candidates(symbol, props):
                entry = data.get(cas)
                if not isinstance(entry, dict):
                    continue
                correlations = entry.get('correlations') or {}
                if not isinstance(correlations, dict):
                    continue
                for method in SURFACE_TENSION_METHOD_PRIORITY:
                    correlation = correlations.get(method)
                    result = self._local_surface_tension_result(
                        cas,
                        entry,
                        method,
                        correlation,
                        T,
                    )
                    if result:
                        return result
            return None


        def _local_surface_tension_result(
            self,
            cas: str,
            entry: Dict[str, Any],
            method: str,
            correlation: Optional[Dict[str, Any]],
            T: float,
        ) -> Optional[PropertyResolutionResult]:
            if not isinstance(correlation, dict):
                return None
            if not self._correlation_in_range(correlation, T):
                return None
            value = self._evaluate_surface_tension_correlation(correlation, T)
            value = self._positive_surface_tension(value)
            if value is None:
                return None

            Tmin = correlation.get('Tmin_K')
            Tmax = correlation.get('Tmax_K')
            notes = (
                f"{correlation.get('source_table', 'surface_tension_correlations_cas.json')} "
                f"for {entry.get('name') or cas} ({cas}); units N/m"
            )
            if Tmin is not None and Tmax is not None:
                notes += f"; range {float(Tmin):g}-{float(Tmax):g} K"
            elif Tmin is not None:
                notes += f"; Tmin {float(Tmin):g} K"
            elif Tmax is not None:
                notes += f"; Tmax {float(Tmax):g} K"

            return PropertyResolutionResult(
                value=value,
                source='local',
                method=f'{method}_surface_tension',
                quality=SURFACE_TENSION_METHOD_QUALITY.get(method, 0.88),
                notes=notes,
            )


        @classmethod
        def _evaluate_surface_tension_correlation(
            cls,
            correlation: Dict[str, Any],
            T: float,
        ) -> Optional[float]:
            equation = str(correlation.get('equation') or '').strip().lower()
            coeffs = cls._correlation_coefficients(correlation)
            try:
                if equation in {'constant', 'constant_surface_tension', 'surface_tension_reference'}:
                    return (
                        correlation.get('sigma_N_per_m')
                        or correlation.get('value_N_per_m')
                        or coeffs.get('sigma')
                        or coeffs.get('value')
                    )

                if equation in {'jasper', 'jasper_lange'}:
                    sigma = (coeffs['a'] - coeffs['b'] * (T - 273.15)) * 1.0e-3
                    return max(0.0, sigma)

                if equation in {'somayajulu', 'somayajulu_revised'}:
                    Tc = coeffs['Tc']
                    if T >= Tc:
                        return 0.0
                    X = (Tc - T) / Tc
                    if X <= 0.0:
                        return None
                    return X * math.sqrt(math.sqrt(X)) * (
                        coeffs.get('A', 0.0)
                        + X * (coeffs.get('B', 0.0) + coeffs.get('C', 0.0) * X)
                    ) * 1.0e-3

                if equation in {'refprop_sigma', 'refprop', 'refprop_surface_tension'}:
                    Tc = coeffs['Tc']
                    tau = 1.0 - T / Tc
                    if tau <= 0.0:
                        return 0.0
                    value = (
                        coeffs.get('sigma0', 0.0) * tau ** coeffs.get('n0', 0.0)
                        + coeffs.get('sigma1', 0.0) * tau ** coeffs.get('n1', 0.0)
                        + coeffs.get('sigma2', 0.0) * tau ** coeffs.get('n2', 0.0)
                    )
                    return value

                if equation in {'dippr_eq106', 'eq106', 'vdi_ppds_11'}:
                    Tc = coeffs['Tc']
                    tau = 1.0 - T / Tc
                    if tau <= 0.0:
                        return 0.0
                    Tr = T / Tc
                    exponent = (
                        coeffs.get('B', 0.0)
                        + coeffs.get('C', 0.0) * Tr
                        + coeffs.get('D', 0.0) * Tr * Tr
                        + coeffs.get('E', 0.0) * Tr**3
                    )
                    return coeffs.get('A', 0.0) * tau**exponent
            except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None
            return None
