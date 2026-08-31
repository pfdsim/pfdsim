"""Hsu-Sheu-Tu liquid viscosity by group contributions -- standalone module.

Model (Hsu, Sheu & Tu, Chem. Eng. J. 88 (2002) 27-35; Perry 9th Table 2-174):

    ln(eta / mPa.s) = sum_i N_i * (a_i + b_i*T + c_i/T^2 + d_i*ln(Pc/bar))

valid for saturated liquids at ~1 atm and Tr < 0.75 (Tr < 0.65 when a
phenolic OH is present -- the positive b of that group makes the curve turn
upward above ~430 K).

Fragmentation is a native RDKit engine with full atom accounting: every
heavy atom must be consumed by exactly one group or the molecule is
rejected.  This replaces the earlier ugropy/Dortmund mapping route.

Corrections applied to the printed tables (evidence: paper's own Table 6
replay + Perry 2-313 / VDI / Viswanath-Natarajan benchmarks, 2026-07-12):

  Q7  =CH-      a = -1.3365, not printed +1.3365 (sign misprint; 1-butene
                +1380% as printed, ~2% flipped, against the paper's own
                validation values).
  Q56 -NH-      a, b, d divided by 10 (PFDSim's correction; benchmark
                8.8-19.1% on four secondary amines vs 75-84% printed;
                an unconstrained refit LOO'd worse at 24%).
  Q88 (-Br)AC   a = -0.81919, not printed -8.1919 (bromobenzene 2.2% vs
                Perry after the decimal shift; -99.9% as printed).
  Q9+Q10 pairs  The printed values are a collinear pair fit on terminal
                alkynes (a ~ +/-90, d ~ -/+25 lie on the regression's null
                direction).  Replaced by pfdsim-refit alkyne UNITS with
                d = 0, fit to Perry 2-313 curves, leave-one-out validated:
                  HC#C- unit: a=-0.0210, b=-2.269e-3, c=+14889.3 (LOO 12.0%,
                              1-pentyne..1-decyne)
                  -C#C- unit: a=+1.1634, b=-1.518e-3, c=+14661.5 (LOO 10.4%,
                              2-butyne/2-pentyne/2-hexyne/3-hexyne)
  Q90 (-I)AC    Gauge-fixed: d absorbed at the implied anchor (45.3 bar),
                a' = 70.9918 - 18.9106*ln(45.3) = -1.1201, d' = 0
                (iodobenzene 7.2% vs VDI; as printed Sum(d) = -18 makes a
                5% Pc error a 90% viscosity error).

Groups deliberately NOT supported (benchmarked unusable or unverifiable):
polyhydric OH (vicinal chelation breaks additivity; refit with a vicinal
penalty still LOO'd 38%), ring ethers Q30, ring ketones Q34, tetralin Q20 /
turpentine Q19 / spirocyclane Q14, N,N-disubstituted amides Q63 (no decimal
or sign repair works), mixed F/Cl carbons Q82-85 (contradictory anchors;
CoolProp covers the freons), secondary bromide Q87 (Sum(d)=74), diaryl
O/N/S bridges (Ph-O-Ph +63%, Ph-NH-Ph +513%), fused aromatic/saturated
rings, aromatic heterocycles, lactones/lactams, amino acids' territory.

Quality model: each group carries a method factor (0.45-0.75, calibrated
against the 2026-07-12 probe benchmarks).  The molecule factor is the
minimum over its groups, then a Pc-sensitivity guard is applied:

    penalty = max(0, |Sum_i N_i d_i| - 1) * (1 - pc_quality)
    quality = factor * max(0, 1 - penalty)

with a hard floor of 0.3 when |Sum d| >= 3 and pc_quality < 0.9 (ring-heavy
molecules amplify Pc errors ~Sum(d)-fold; cyclohexane's Sum d = -9.1).

CLI:  python hsu_method.py "CCCCCC" --pc 3025 --tc 507.6 --visc 298.15
      (pc in kPa, matching nannoolal_method conventions)
"""
from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

_P_BAR_PER_KPA = 0.01
_DEFAULT_TR_MAX = 0.75
_PHENOLIC_TR_MAX = 0.65

# ---------------------------------------------------------------------------
# Group table: name -> (a, b, c, d, quality_factor, source_gid)
# b and c are stored EVALUATED (paper columns are b*1e2 and c*1e-4).
# ---------------------------------------------------------------------------
GROUPS: Dict[str, Tuple[float, float, float, float, float, str]] = {
    # carbon skeleton
    'ch3':              (0.0570, -0.002383, 7556.0, -0.1765, 0.75, 'Q2'),
    'ch2':              (-0.1497, 6.0e-05, 14157.0, 0.0751, 0.75, 'Q3'),
    'ch':               (-2.2942, 0.004028, 45094.0, 0.6679, 0.75, 'Q4'),
    'c':                (1.0031, -0.003677, -60316.0, 1.1972, 0.75, 'Q5'),
    'alkene_ch2':       (0.9256, -0.002656, 9860.0, -0.4417, 0.75, 'Q6'),
    'alkene_ch':        (-1.3365, 0.001612, 19408.0, 0.2507, 0.75, 'Q7 sign-fixed'),
    'alkene_c':         (-3.5020, 0.004305, 31287.0, 1.0465, 0.70, 'Q8'),
    'alkyne_terminal':  (-0.0210, -0.002269, 14889.3, 0.0, 0.62, 'Q9+Q10 refit LOO 12.0%'),
    'alkyne_internal':  (1.1634, -0.001518, 14661.5, 0.0, 0.65, '2xQ10 refit LOO 10.4%'),
    'ring_ch2':         (6.0416, -0.001778, 8437.0, -1.5184, 0.75, 'Q11'),
    'ring_ch':          (-33.8745, 0.007637, 72433.0, 8.5951, 0.75, 'Q12'),
    'ring_alkene_ch':   (1.2028, -0.00012, 20143.0, -0.3677, 0.70, 'Q13'),
    'aromatic_ch':      (-0.8570, -9.8e-05, 24376.0, 0.1311, 0.75, 'Q15'),
    'aromatic_c_simple': (0.7896, -0.000231, -9222.0, 0.1928, 0.75, 'Q16'),
    'aromatic_c_biphenyl': (2.0973, 0.000444, 81690.0, -0.4351, 0.70, 'Q17'),
    'aromatic_c_naphthalene': (0.4392, 0.000683, 88426.0, -0.1685, 0.70, 'Q18'),
    # oxygen
    'oh_primary_lt3':   (5.7852, -0.00531, 95499.0, -1.0300, 0.75, 'Q21'),
    'oh_primary_gt2':   (1.4351, -0.01001, 138366.0, 0.3418, 0.75, 'Q22'),
    'oh_secondary':     (-2.6895, -0.003645, 298404.0, 0.4246, 0.75, 'Q23'),
    'oh_tertiary':      (-18.5630, 0.024275, 785417.0, 0.9650, 0.70, 'Q24'),
    'oh_ring':          (16.7808, 0.008509, 771759.0, -6.9285, 0.55,
                         'Q25; cyclohexanol 52% full-window'),
    'oh_aromatic':      (-2.0856, 0.006362, 500840.0, -1.0539, 0.55, 'Q27; Tr<=0.65, ±25-30%'),
    'ether_o':          (-0.7185, 0.000985, 29405.0, 0.1149, 0.75, 'Q29'),
    'aromatic_o':       (-2.3454, 0.000872, 64296.0, 0.5389, 0.65, 'Q31; mono-aryl only'),
    'aldehyde':         (-0.8288, -0.002612, 37241.0, 0.2386, 0.75, 'Q32'),
    'ketone':           (-2.6622, 0.001142, 67008.0, 0.7348, 0.75, 'Q33'),
    'formic_acid':      (-2.7291, 0.000413, 274079.0, 0.0002, 0.70, 'Q35'),
    'carboxylic_acid_lt7': (-4.0451, -0.001841, 126878.0, 1.1139, 0.75, 'Q36'),
    'carboxylic_acid_gt6': (-0.6721, -0.001693, 200309.0, 0.0279, 0.75, 'Q37'),
    'formate':          (-3.3731, -0.000113, 94694.0, 0.6071, 0.75, 'Q38'),
    'ester_lt8':        (-0.0635, -0.002162, 19325.0, 0.4686, 0.75, 'Q39'),
    'ester_gt7':        (-2.5390, 6.0e-06, 54231.0, 0.8717, 0.75, 'Q40'),
    'anhydride':        (-11.8236, 0.000111, 72831.0, 3.6587, 0.70, 'Q42'),
    'carbonate':        (-8.0314, 0.002848, 93746.0, 2.1486, 0.50, 'Q43; DEC 9%/DMC +34%'),
    # sulfur
    'thioether_s':      (-3.2767, 0.000779, 44123.0, 0.9549, 0.75, 'Q48'),
    'thiol_primary':    (-2.1030, -0.000965, 60066.0, 0.3464, 0.75, 'Q49'),
    'thiol_secondary':  (-0.2481, -0.003285, 19387.0, 0.1148, 0.75, 'Q50'),
    'thiol_tertiary':   (-12.3498, 0.012621, 231473.0, 1.3950, 0.70, 'Q51'),
    'sulfoxide':        (-32.8607, 0.006232, 275184.0, 7.7525, 0.60, 'Q54; DMSO 13.7%'),
    # nitrogen
    'amine_primary':    (-1.1345, -0.002126, 70544.0, 0.1336, 0.75, 'Q55'),
    'amine_secondary':  (-0.69489, -0.0001723, 57805.0, 0.16467, 0.70,
                         'Q56 PFDSim a,b,d/10; 9-19% on 4 amines'),
    'amine_tertiary':   (-2.1403, 0.004842, 61893.0, 0.4718, 0.72, 'Q57; Et3N 4.7%'),
    'aromatic_amine_primary': (-6.3646, -0.00018, 232752.0, 1.0653, 0.75, 'Q58'),
    'aromatic_amine_secondary': (-1.7592, 0.002208, 149707.0, 0.1171, 0.70,
                                 'Q59; N-methylaniline 8.8%'),
    'aromatic_amine_tertiary': (-1.2982, 0.005975, 140415.0, -0.0031, 0.70,
                                'Q60; N,N-dimethylaniline 10.8%'),
    'formamide_molecule': (-1.5435, -0.002774, 318007.0, 0.0001, 0.45,
                           'Q61; whole molecule, ±20-30%'),
    'nitro':            (-13.0333, 0.001801, 129392.0, 2.8987, 0.75, 'Q45'),
    'aromatic_nitro':   (-1.2954, 0.000427, 121837.0, -0.0948, 0.75, 'Q47'),
    'nitrile':          (-2.7194, -0.001324, 77955.0, 0.6293, 0.75, 'Q70'),
    'aromatic_nitrile': (0.9435, -8.6e-05, 86310.0, -0.6443, 0.75, 'Q71'),
    # halogens
    'cl_primary':       (-1.7997, -0.003851, 30118.0, 0.5524, 0.75, 'Q72'),
    'vinyl_chcl':       (1.5851, -0.001934, 37798.0, -0.4748, 0.70, 'Q73'),
    'cl2':              (-3.0561, -0.01077, 1882.0, 1.2223, 0.75, 'Q74; +carbon'),
    'cl3':              (-1.3357, -0.00322, 88683.0, 0.1702, 0.75, 'Q75; subsumes CHx'),
    'cl4':              (4.2070, -0.00413, 133194.0, -1.1972, 0.75, 'Q76; subsumes C'),
    'aromatic_cl':      (-0.3083, -0.000623, 41382.0, -0.2644, 0.75, 'Q77'),
    'f_primary':        (-9.4982, 0.002607, 113406.0, 1.8461, 0.60, 'Q78; unvalidated'),
    'f2':               (-10.3980, -0.011189, 13134.0, 2.6681, 0.70, 'Q79; +carbons'),
    'f3':               (1.5394, 0.008465, 178121.0, -2.9915, 0.70, 'Q80; aromatic-attached'),
    'aromatic_f':       (0.4079, -0.002352, -1505.0, -0.2893, 0.75, 'Q81'),
    'br_primary':       (-0.7586, -0.006623, -24228.0, 0.7385, 0.75, 'Q86'),
    'aromatic_br':      (-0.81919, -0.001635, 30150.0, 0.0621, 0.75,
                         'Q88 PFDSim a/10; bromobenzene 2.2%'),
    'i_primary':        (-1.4672, -0.002787, 43362.0, 0.5635, 0.75, 'Q89'),
    'aromatic_i':       (-1.1201, -0.000245, 72061.0, 0.0, 0.65,
                         'Q90 gauge@45.3 bar; iodobenzene 7.2%'),
    'acid_chloride':    (-2.3300, -0.00047, 82815.0, 0.4485, 0.65,
                         'Q91; propionyl 4.7%, acetyl -22% (first member)'),
}

_ALLOWED_ELEMENTS = {'C', 'H', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I'}


class HsuFragmentationError(ValueError):
    """Molecule outside the supported Hsu domain (reason in str)."""


@dataclass
class HsuFragmentation:
    smiles: str
    groups: Counter = field(default_factory=Counter)
    notes: List[str] = field(default_factory=list)
    tr_max: float = _DEFAULT_TR_MAX
    tr_min: float = 0.0

    @property
    def method_factor(self) -> float:
        factor = min(GROUPS[g][4] for g in self.groups)
        for note in self.notes:
            if note.startswith('__cap_'):
                factor = min(factor, float(note.strip('_').split('_')[1]))
            elif 'first member' in note:
                factor = min(factor, 0.55)
            elif 'approximated as primary' in note:
                factor = min(factor, 0.60)
        return factor

    @property
    def d_sum(self) -> float:
        return sum(n * GROUPS[g][3] for g, n in self.groups.items())

    def coefficient_sums(self) -> Tuple[float, float, float, float]:
        a = b = c = d = 0.0
        for g, n in self.groups.items():
            ga, gb, gc, gd, _, _ = GROUPS[g]
            a += n * ga
            b += n * gb
            c += n * gc
            d += n * gd
        return a, b, c, d


@dataclass
class HsuViscosityResult:
    fragmentation: HsuFragmentation
    pc_kPa: float
    tc_K: Optional[float] = None
    pc_quality: float = 1.0

    def ln_eta_mPas(self, t_K: float) -> float:
        a, b, c, d = self.fragmentation.coefficient_sums()
        return a + b * t_K + c / (t_K * t_K) + d * math.log(self.pc_kPa * _P_BAR_PER_KPA)

    def viscosity_mPa_s(self, t_K: float) -> Optional[float]:
        if self.tc_K is not None:
            tr = t_K / self.tc_K
            if not (self.fragmentation.tr_min - 1e-9 <= tr
                    <= self.fragmentation.tr_max + 1e-9):
                warnings.warn(
                    f'Hsu viscosity requested at Tr={tr:.3f} outside validity '
                    f'[{self.fragmentation.tr_min:.2f}, '
                    f'{self.fragmentation.tr_max:.2f}]', stacklevel=2)
                return None
        return math.exp(self.ln_eta_mPas(t_K))

    def viscosity_Pa_s(self, t_K: float) -> Optional[float]:
        v = self.viscosity_mPa_s(t_K)
        return None if v is None else v * 1e-3

    @property
    def quality(self) -> float:
        # Pc-sensitivity penalty: extra ln-eta error ~ |Sum d| * sigma(ln Pc),
        # with sigma ~ (1 - pc_quality)/3 on PFDSim's 1-3*MAPE scale.  The
        # 0.2 floor keeps a saturated penalty at "last resort" rather than 0.
        dsum = abs(self.fragmentation.d_sum)
        penalty = max(0.0, dsum - 1.0) * (1.0 - self.pc_quality)
        q = self.fragmentation.method_factor * max(0.2, 1.0 - penalty)
        if dsum >= 3.0 and self.pc_quality < 0.9:
            q = min(q, 0.3)
        return q


# ---------------------------------------------------------------------------
# fragmentation engine
# ---------------------------------------------------------------------------

def _reject(reason: str):
    raise HsuFragmentationError(reason)


def _halogen_counts(atom) -> Counter:
    return Counter(nb.GetSymbol() for nb in atom.GetNeighbors()
                   if nb.GetSymbol() in ('F', 'Cl', 'Br', 'I'))


def _carbon_group_name(atom, in_ring: bool) -> Optional[str]:
    h = atom.GetTotalNumHs()
    if in_ring:
        return {2: 'ring_ch2', 1: 'ring_ch'}.get(h)
    return {3: 'ch3', 2: 'ch2', 1: 'ch', 0: 'c'}.get(h)


def fragment(smiles: str) -> HsuFragmentation:
    """Fragment a molecule into Hsu groups with full atom accounting.

    Raises HsuFragmentationError with the gating reason if the molecule is
    outside the supported domain.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        _reject('unparseable SMILES')
    frag = HsuFragmentation(smiles=smiles)
    groups, notes = frag.groups, frag.notes
    n_atoms = mol.GetNumAtoms()
    assigned = [False] * n_atoms
    total_c = sum(a.GetSymbol() == 'C' for a in mol.GetAtoms())

    def take(*idxs):
        for i in idxs:
            if assigned[i]:
                _reject(f'atom {i} claimed twice (fragmenter bug)')
            assigned[i] = True

    def match(smarts):
        patt = Chem.MolFromSmarts(smarts)
        return [m for m in mol.GetSubstructMatches(patt)
                if not any(assigned[i] for i in m)]

    # ---- pass 0: global vetoes -------------------------------------------
    # nitro groups may carry RDKit's charge-separated form; exempt them
    nitro_matches = []
    nitro_atom_idx = set()
    for patt in ('[NX3](=[OX1])=[OX1]', '[NX3+](=[OX1])[OX1-]'):
        for m in mol.GetSubstructMatches(Chem.MolFromSmarts(patt)):
            if m[0] not in nitro_atom_idx:
                nitro_matches.append(m)
                nitro_atom_idx.update(m)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in _ALLOWED_ELEMENTS:
            _reject(f'element {atom.GetSymbol()} unsupported')
        if ((atom.GetFormalCharge() or atom.GetNumRadicalElectrons())
                and atom.GetIdx() not in nitro_atom_idx):
            _reject('charged or radical species')
        if atom.GetIsAromatic() and atom.GetSymbol() != 'C':
            _reject('aromatic heteroatom (heterocycle)')
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        arom = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring]
        if any(arom) and not all(arom):
            _reject('fused aromatic/saturated ring (tetralin-type)')
        if not any(arom) and len(ring) < 5:
            _reject('strained ring <5 atoms (cyclopropane 63% measured)')
    triple = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE
              and b.GetBeginAtom().GetSymbol() == 'C'
              and b.GetEndAtom().GetSymbol() == 'C']
    if len(triple) > 1:
        _reject('more than one C#C bond')
    for bond in mol.GetBonds():
        b0, b1 = bond.GetBeginAtom(), bond.GetEndAtom()
        if b0.GetSymbol() == 'N' and b1.GetSymbol() == 'N':
            _reject('N-N bond (hydrazine/azo: no Hsu family, '
                    'phenylhydrazine +293% measured)')
        if (b0.GetSymbol() == 'C' and b1.GetSymbol() == 'C'
                and all(sum(_halogen_counts(a).values()) >= 2
                        for a in (b0, b1))):
            _reject('adjacent polyhalogenated carbons (1,1,2,2-'
                    'tetrachloroethane-type: 69-82% measured; use Nannoolal)')
    # ortho-chelating aromatics (intramolecular H-bond kills the association
    # the OH(a) group assumes): methyl salicylate +1207%, salicylaldehyde
    # +819% measured
    for patt in ('[OX2H1][c]:[c][CX3]=[OX1]',
                 '[OX2H1][c]:[c][NX3](=[OX1])=[OX1]',
                 '[OX2H1][c]:[c][NX3+](=[OX1])[OX1-]'):
        if mol.HasSubstructMatch(Chem.MolFromSmarts(patt)):
            _reject('ortho-chelating aromatic (salicylate/o-nitrophenol '
                    'type: intramolecular H-bond, +800-1200% measured)')
    # halogenated carboxylic acids: electron withdrawal strengthens
    # dimerization (chloroacetic 32%, trichloroacetic reference-class case)
    if (mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[OX2H1]'))
            and any(a.GetSymbol() in ('F', 'Cl', 'Br', 'I')
                    for a in mol.GetAtoms())):
        _reject('halogenated carboxylic acid (enhanced dimerization)')
    def _is_carboxyl_oh(o_atom):
        c = o_atom.GetNeighbors()[0]
        return any(b.GetBondType() == Chem.BondType.DOUBLE
                   and b.GetOtherAtom(c).GetSymbol() == 'O'
                   for b in c.GetBonds())

    oh_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == 'O'
                and a.GetTotalNumHs() == 1 and a.GetDegree() == 1
                and a.GetNeighbors()[0].GetSymbol() == 'C']
    n_alcohol = sum(not _is_carboxyl_oh(a) for a in oh_atoms)
    n_carboxyl = len(oh_atoms) - n_alcohol
    if n_alcohol > 1:
        _reject('polyhydric alcohol (vicinal chelation breaks additivity)')
    if n_carboxyl > 1:
        _reject('polycarboxylic acid (multifunctional, unvalidated)')
    if n_alcohol and n_carboxyl:
        _reject('hydroxy-acid (multifunctional, unvalidated)')
    if total_c == 0:
        _reject('no carbon (inorganic species out of scope)')
    if total_c == 1 and mol.GetNumAtoms() == 1:
        _reject('methane (cryogenic; out of scope)')
    # organic requirement: at least one C-H bond (cyanogen, oxalyl chloride
    # and friends are not meaningfully organic).  CCl4 is the one validated
    # H-free exception and is exempted after fragmentation.
    has_ch = any(a.GetSymbol() == 'C' and a.GetTotalNumHs() > 0
                 for a in mol.GetAtoms())

    # ---- pass 1: multi-atom functional groups, priority order -------------
    # formamide: whole molecule
    if Chem.MolToSmiles(mol) == 'NC=O':
        groups['formamide_molecule'] += 1
        return frag
    if match('[CX3](=O)[NX3]'):
        _reject('amide (only formamide itself is supported)')

    for m in match('[CX3](=[OX1])[Cl]'):
        take(*m)
        groups['acid_chloride'] += 1
        if total_c <= 2:
            notes.append('acetyl chloride: first member of series, -22% bias')

    anh = match('[CX3](=[OX1])[OX2][CX3](=[OX1])')
    for m in anh:
        if any(mol.GetAtomWithIdx(i).IsInRing() for i in m):
            _reject('cyclic anhydride unsupported')
        take(*m)
        groups['anhydride'] += 1

    for m in match('[OX2;!R]([#6])[CX3;!R](=[OX1])[OX2;!R][#6]'):
        take(m[0], m[2], m[3], m[4])          # O, C, =O, O (not the alkyls)
        groups['carbonate'] += 1

    for m in match('[CX3H1](=[OX1])[OX2][#6]'):       # formate H-COO-
        if any(mol.GetAtomWithIdx(i).IsInRing() for i in m[:3]):
            _reject('cyclic formate unsupported')
        take(*m[:3])
        groups['formate'] += 1
    for m in match('[CX3;!$([CX3H1])](=[OX1])[OX2][#6]'):   # ester -COO-
        if any(mol.GetAtomWithIdx(i).IsInRing() for i in m[:3]):
            _reject('lactone unsupported')
        take(*m[:3])
        groups['ester_lt8' if total_c < 8 else 'ester_gt7'] += 1

    if Chem.MolToSmiles(mol) == 'O=CO':
        groups['formic_acid'] += 1
        return frag
    for m in match('[CX3](=[OX1])[OX2H1]'):
        take(*m)
        groups['carboxylic_acid_lt7' if total_c < 7 else 'carboxylic_acid_gt6'] += 1

    for m in match('[CX3H1]=[OX1]'):                  # aldehyde
        take(*m)
        groups['aldehyde'] += 1
    for m in match('[#6][CX3](=[OX1])[#6]'):          # ketone (carbonyl only)
        cidx = m[1]
        if mol.GetAtomWithIdx(cidx).IsInRing():
            _reject('ring ketone (Q34 unusable: -73% as printed)')
        take(cidx, m[2])
        groups['ketone'] += 1

    # sulfur
    if match('[SX4]') or match('[SX3](=O)(=O)'):
        _reject('sulfone/sulfate/sulfonamide unsupported')
    for m in match('[SX3](=[OX1])([#6])[#6]'):        # sulfoxide
        take(m[0], m[1])
        groups['sulfoxide'] += 1
    for m in match('[SX2H1][#6]'):                    # thiol
        cn = mol.GetAtomWithIdx(m[1])
        if cn.GetIsAromatic():
            _reject('aromatic thiol unsupported')
        h = cn.GetTotalNumHs()
        take(m[0])
        groups['thiol_primary' if h >= 2 else
               'thiol_secondary' if h == 1 else 'thiol_tertiary'] += 1
    for m in match('[SX2]([#6])[#6]'):                # thioether
        s = mol.GetAtomWithIdx(m[0])
        if any(nb.GetSymbol() == 'S' for nb in s.GetNeighbors()):
            _reject('disulfide unsupported')
        if s.IsInRing():
            _reject('ring sulfide unsupported')
        aryl = sum(mol.GetAtomWithIdx(i).GetIsAromatic() for i in m[1:])
        if aryl >= 2:
            _reject('diaryl sulfide unsupported')
        take(m[0])
        groups['thioether_s'] += 1

    # nitrogen
    for m in match('[NX1]#[CX2]'):                    # nitrile
        c_nb = [nb for nb in mol.GetAtomWithIdx(m[1]).GetNeighbors()
                if nb.GetIdx() != m[0] and nb.GetSymbol() == 'C']
        if not c_nb:
            _reject('nitrile without carbon substituent (HCN/cyanogen)')
        take(*m)
        groups['aromatic_nitrile' if c_nb[0].GetIsAromatic() else 'nitrile'] += 1
    for m in nitro_matches:
        if any(assigned[i] for i in m):
            continue
        n_at = mol.GetAtomWithIdx(m[0])
        carbon = next((nb for nb in n_at.GetNeighbors()
                       if nb.GetSymbol() == 'C'), None)
        if carbon is None:
            _reject('nitrite/nitrate ester unsupported')
        take(*m)
        groups['aromatic_nitro' if carbon.GetIsAromatic() else 'nitro'] += 1

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'N' or assigned[atom.GetIdx()]:
            continue
        if atom.IsInRing():
            _reject('ring amine unsupported')
        aryl = sum(nb.GetIsAromatic() for nb in atom.GetNeighbors())
        h = atom.GetTotalNumHs()
        if aryl >= 2:
            _reject('diaryl amine unsupported (+513% measured)')
        take(atom.GetIdx())
        if h == 2:
            groups['aromatic_amine_primary' if aryl else 'amine_primary'] += 1
        elif h == 1:
            groups['aromatic_amine_secondary' if aryl else 'amine_secondary'] += 1
        else:
            groups['aromatic_amine_tertiary' if aryl else 'amine_tertiary'] += 1

    # oxygen: ethers then OH
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'O' or assigned[atom.GetIdx()]:
            continue
        nbs = atom.GetNeighbors()
        if atom.GetTotalNumHs() == 0 and len(nbs) == 2:
            if any(nb.GetSymbol() == 'O' for nb in nbs):
                _reject('peroxide unsupported')
            if atom.IsInRing():
                _reject('ring ether unsupported (THF +95% as printed)')
            aryl = sum(nb.GetIsAromatic() for nb in nbs)
            if aryl >= 2:
                _reject('diaryl ether unsupported (+63% measured)')
            take(atom.GetIdx())
            groups['aromatic_o' if aryl else 'ether_o'] += 1
        elif atom.GetTotalNumHs() == 1 and len(nbs) == 1:
            carbon = nbs[0]
            take(atom.GetIdx())
            if carbon.GetIsAromatic():
                groups['oh_aromatic'] += 1
                frag.tr_max = _PHENOLIC_TR_MAX
                notes.append('phenolic OH: validity capped at Tr 0.65')
            elif carbon.IsInRing():
                groups['oh_ring'] += 1
            elif carbon.GetTotalNumHs() == 0:
                groups['oh_tertiary'] += 1
            elif carbon.GetTotalNumHs() == 1 and carbon.GetDegree() >= 2 and \
                    sum(nb.GetSymbol() == 'C' for nb in carbon.GetNeighbors()) >= 2:
                groups['oh_secondary'] += 1
            else:
                groups['oh_primary_lt3' if total_c < 3 else 'oh_primary_gt2'] += 1
        else:
            _reject('unsupported oxygen environment')

    # alkynes: consume both carbons as a refit unit
    for bond in triple:
        b0, b1 = bond.GetBeginAtom(), bond.GetEndAtom()
        if assigned[b0.GetIdx()] or assigned[b1.GetIdx()]:
            continue
        unit = {b0.GetIdx(), b1.GetIdx()}
        if not any(nb.GetSymbol() == 'C' and nb.GetIdx() not in unit
                   for at in (b0, b1) for nb in at.GetNeighbors()):
            _reject('bare C#C without carbon substituent (acetylene)')
        terminal = b0.GetTotalNumHs() == 1 or b1.GetTotalNumHs() == 1
        take(b0.GetIdx(), b1.GetIdx())
        groups['alkyne_terminal' if terminal else 'alkyne_internal'] += 1

    # halogen clusters per carbon
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C' or assigned[atom.GetIdx()]:
            continue
        hal = _halogen_counts(atom)
        if not hal:
            continue
        if atom.GetIsAromatic():
            if sum(hal.values()) > 1:
                _reject('polyhalogenated aromatic carbon unsupported')
            sym = next(iter(hal))
            hidx = next(nb.GetIdx() for nb in atom.GetNeighbors()
                        if nb.GetSymbol() == sym)
            take(hidx)
            groups[{'F': 'aromatic_f', 'Cl': 'aromatic_cl',
                    'Br': 'aromatic_br', 'I': 'aromatic_i'}[sym]] += 1
            continue
        if len(hal) > 1:
            _reject('mixed halogens on one carbon (Q82-85 unusable)')
        sym, n = next(iter(hal.items()))
        hidxs = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetSymbol() == sym]
        vinylic = any(b.GetBondType() == Chem.BondType.DOUBLE
                      and b.GetOtherAtom(atom).GetSymbol() == 'C'
                      for b in atom.GetBonds())
        if sym == 'Cl':
            if vinylic:
                if n == 1 and atom.GetTotalNumHs() == 1:
                    take(atom.GetIdx(), *hidxs)
                    groups['vinyl_chcl'] += 1
                    continue
                _reject('polychlorinated vinyl carbon unsupported')
            if n == 1:
                take(*hidxs)
                groups['cl_primary'] += 1
                if atom.GetTotalNumHs() < 2:
                    notes.append('secondary/tertiary Cl approximated as primary '
                                 '(+20% measured)')
            elif n == 2:
                take(*hidxs)          # carbon itself counted separately (validated)
                groups['cl2'] += 1
            elif n == 3:
                if any(nb.GetSymbol() == 'C' for nb in atom.GetNeighbors()):
                    _reject('substituted CCl3 (convention validated only for '
                            'chloroform; 1,1,1-trichloroethane +81% measured; '
                            'use Nannoolal)')
                take(atom.GetIdx(), *hidxs)   # subsumes the carbon (validated)
                groups['cl3'] += 1
            else:
                take(atom.GetIdx(), *hidxs)
                groups['cl4'] += 1
        elif sym == 'F':
            if vinylic:
                _reject('vinyl fluoride unsupported')
            if n == 1:
                _reject('aliphatic monofluoride unsupported '
                        '(Q78 measured 2000%+ on fluoromethane/-ethane)')
            elif n == 2:
                take(*hidxs)
                groups['f2'] += 1
            elif n == 3:
                if not any(nb.GetIsAromatic() for nb in atom.GetNeighbors()):
                    _reject('aliphatic CF3 unvalidated')
                take(*hidxs)
                groups['f3'] += 1
            else:
                _reject('CF4-type carbon unsupported')
        elif sym == 'Br':
            if n > 1 or vinylic:
                _reject('polybrominated/vinylic carbon unsupported')
            if atom.GetTotalNumHs() < 2:
                _reject('secondary bromide unsupported (Q87 Sum(d)=74)')
            take(*hidxs)
            groups['br_primary'] += 1
        else:  # I
            if n > 1 or vinylic:
                _reject('polyiodinated/vinylic carbon unsupported')
            if atom.GetTotalNumHs() < 2:
                _reject('secondary iodide unsupported')
            take(*hidxs)
            groups['i_primary'] += 1

    # alkenes (acyclic; at most one C=C).  A bond may have one atom already
    # consumed by vinyl_chcl -- the partner still needs its alkene group.
    dbl = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
           and b.GetBeginAtom().GetSymbol() == 'C'
           and b.GetEndAtom().GetSymbol() == 'C'
           and not b.GetBeginAtom().GetIsAromatic()]
    acyclic_dbl = [b for b in dbl if not b.GetBeginAtom().IsInRing()
                   and not b.GetEndAtom().IsInRing()]
    ring_dbl = [b for b in dbl if b not in acyclic_dbl]
    if len(acyclic_dbl) > 1:
        _reject('polyene unsupported')
    for b in acyclic_dbl:
        for at in (b.GetBeginAtom(), b.GetEndAtom()):
            if assigned[at.GetIdx()]:
                continue
            name = {2: 'alkene_ch2', 1: 'alkene_ch', 0: 'alkene_c'}.get(
                at.GetTotalNumHs())
            if name is None:
                _reject('alkene carbon with unexpected H count')
            take(at.GetIdx())
            groups[name] += 1
    for b in ring_dbl:
        for at in (b.GetBeginAtom(), b.GetEndAtom()):
            if assigned[at.GetIdx()]:
                continue
            if at.GetTotalNumHs() != 1:
                _reject('substituted ring alkene unsupported')
            take(at.GetIdx())
            groups['ring_alkene_ch'] += 1

    # ---- pass 2: remaining carbons ----------------------------------------
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if assigned[idx] or atom.GetSymbol() != 'C':
            continue
        if atom.GetIsAromatic():
            if atom.GetTotalNumHs():
                groups['aromatic_ch'] += 1
            else:
                inter_ring = any(nb.GetIsAromatic() and not b.GetIsAromatic()
                                 for b in atom.GetBonds()
                                 for nb in (b.GetOtherAtom(atom),))
                n_arom_bonds = sum(b.GetIsAromatic() for b in atom.GetBonds())
                if inter_ring:
                    groups['aromatic_c_biphenyl'] += 1
                elif n_arom_bonds == 3:
                    groups['aromatic_c_naphthalene'] += 1
                else:
                    groups['aromatic_c_simple'] += 1
            take(idx)
            continue
        name = _carbon_group_name(atom, atom.IsInRing())
        if name is None:
            _reject('ring quaternary/CH0 carbon unsupported (Q14 cut)')
        take(idx)
        groups[name] += 1

    # ---- pass 3: leftovers & structure-level gates -------------------------
    for atom in mol.GetAtoms():
        if not assigned[atom.GetIdx()]:
            _reject(f'unassigned atom {atom.GetSymbol()}{atom.GetIdx()}')
    if not groups:
        _reject('no groups found')
    # skeleton-or-singleton rule: pure functional-group assemblies with no
    # carbon skeleton (cyanogen = 2x nitrile, 924% measured) are rejected;
    # the validated whole-molecule singletons are exempt.
    _SKELETON = {'ch3', 'ch2', 'ch', 'c', 'ring_ch2', 'ring_ch',
                 'aromatic_ch', 'aromatic_c_simple', 'aromatic_c_biphenyl',
                 'aromatic_c_naphthalene', 'alkene_ch2', 'alkene_ch',
                 'alkene_c', 'ring_alkene_ch', 'alkyne_terminal',
                 'alkyne_internal', 'vinyl_chcl'}
    _SINGLETONS = {'cl3', 'cl4', 'formic_acid', 'formamide_molecule'}
    if not (set(groups) & _SKELETON) and set(groups) - _SINGLETONS:
        _reject('no carbon skeleton (functional groups only)')
    if not has_ch and set(groups) != {'cl4'}:
        _reject('no C-H bond (not meaningfully organic)')

    # Hsu's own caveat: multifunctional compounds are less reliable.
    # TNT 68%, 1,2-dimethoxypropane 54% measured -> cap 0.60.
    _FUNCTIONAL = {'oh_primary_lt3', 'oh_primary_gt2', 'oh_secondary',
                   'oh_tertiary', 'oh_ring', 'oh_aromatic', 'ether_o',
                   'aromatic_o', 'aldehyde', 'ketone', 'carboxylic_acid_lt7',
                   'carboxylic_acid_gt6', 'formate', 'ester_lt8', 'ester_gt7',
                   'nitro', 'aromatic_nitro', 'nitrile', 'aromatic_nitrile',
                   'amine_primary', 'amine_secondary', 'amine_tertiary',
                   'aromatic_amine_primary', 'aromatic_amine_secondary',
                   'aromatic_amine_tertiary', 'thiol_primary',
                   'thiol_secondary', 'thiol_tertiary', 'thioether_s',
                   'sulfoxide', 'acid_chloride'}
    n_functional = sum(n for g, n in groups.items() if g in _FUNCTIONAL)
    if n_functional >= 2:
        notes.append('multifunctional compound: reliability reduced '
                     '(Hsu caveat; ~50-70% measured on nitro/ether cases)')
        notes.append('__cap_0.60__')

    # benzylic polar substituent (benzyl alcohol 83%, benzyl mercaptan 100%)
    for atom in mol.GetAtoms():
        if (atom.GetSymbol() == 'C' and not atom.GetIsAromatic()
                and not atom.IsInRing()
                and any(nb.GetIsAromatic() for nb in atom.GetNeighbors())
                and any(nb.GetSymbol() in ('O', 'N', 'S')
                        for nb in atom.GetNeighbors())):
            notes.append('benzylic polar substituent (~80-100% measured)')
            notes.append('__cap_0.55__')
            break

    # low-T divergence: ethers over-predict hugely below Tr 0.45 (DME +100%,
    # MIPE +194% at Tr 0.35) and secondary/tertiary alcohols diverge toward
    # Tm (2-propanol +176% at Tr 0.40, -1% at 0.60) -> validity floor.
    if any(g in groups for g in ('ether_o', 'aromatic_o', 'oh_secondary',
                                 'oh_tertiary', 'oh_ring')):
        frag.tr_min = 0.45
        notes.append('ether/branched-OH: validity floored at Tr 0.45 '
                     '(low-T divergence measured)')
    # branch-adjacent ether oxygen keeps a +30-70% floor even in-range
    if groups.get('ether_o'):
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() == 0 and \
                    any(nb.GetSymbol() == 'C' and not nb.GetIsAromatic()
                        and nb.GetTotalNumHs() <= 1 for nb in atom.GetNeighbors()):
                notes.append('branched carbon adjacent to ether O '
                             '(MIPE +30-70% measured)')
                notes.append('__cap_0.55__')
                break

    if groups.get('ring_alkene_ch') and groups.get('ring_ch'):
        _reject('substituted cycloalkene (ring alkene + ring CH: +268% measured)')
    n_ring_ch = groups.get('ring_ch', 0)
    if n_ring_ch > 2:
        _reject('>2 ring CH (bridged/polycyclic beyond decalin-type)')
    if n_ring_ch == 2:
        # only the FUSED pair (decalin pattern) is validated (core ~16%);
        # 1,2-dimethylcyclohexane-type substitution measured ~97%
        ring_ch_atoms = [a for a in mol.GetAtoms()
                         if a.GetSymbol() == 'C' and a.IsInRing()
                         and not a.GetIsAromatic() and a.GetTotalNumHs() == 1
                         and all(nb.GetSymbol() == 'C' for nb in a.GetNeighbors())]
        fused = (len(ring_ch_atoms) == 2
                 and mol.GetBondBetweenAtoms(ring_ch_atoms[0].GetIdx(),
                                             ring_ch_atoms[1].GetIdx()) is not None
                 and all(ri.NumAtomRings(a.GetIdx()) >= 2 for a in ring_ch_atoms))
        if not fused:
            _reject('two non-fused ring CH (dimethylcyclohexane-type: '
                    '~97% measured)')
        notes.append('decalin-type bicyclic: core-range ~16%, quality capped')
        frag.notes.append('__cap_0.50__')
    return frag


def estimate_viscosity(smiles: str, pc_kPa: float, tc_K: Optional[float] = None,
                       pc_quality: float = 1.0) -> HsuViscosityResult:
    """Fragment and wrap into a viscosity estimator.

    pc_kPa: critical pressure in kPa (converted to bar internally).
    pc_quality: quality of the Pc value on PFDSim's scale (1.0 experimental);
        drives the Sum(d) sensitivity penalty and the >=3 hard guard.
    """
    if pc_kPa is None or pc_kPa <= 0:
        raise ValueError('pc_kPa required (Hsu is a Pc-anchored model)')
    frag = fragment(smiles)
    return HsuViscosityResult(fragmentation=frag, pc_kPa=pc_kPa,
                              tc_K=tc_K, pc_quality=pc_quality)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='Hsu liquid viscosity GC')
    ap.add_argument('smiles')
    ap.add_argument('--pc', type=float, required=True, help='Pc in kPa')
    ap.add_argument('--tc', type=float, default=None, help='Tc in K (validity)')
    ap.add_argument('--pcq', type=float, default=1.0, help='Pc quality (0-1)')
    ap.add_argument('--visc', type=float, default=None, metavar='T_K')
    args = ap.parse_args()

    try:
        res = estimate_viscosity(args.smiles, args.pc, args.tc, args.pcq)
    except HsuFragmentationError as exc:
        print(f'rejected: {exc}')
        raise SystemExit(1)
    f = res.fragmentation
    print(f'groups: {dict(sorted(f.groups.items()))}')
    print(f'Sum(d) = {f.d_sum:+.3f}   method factor = {f.method_factor:.2f}   '
          f'quality = {res.quality:.3f}   Tr_max = {f.tr_max}')
    for note in f.notes:
        if not note.startswith('__'):
            print(f'note: {note}')
    if args.visc:
        v = res.viscosity_mPa_s(args.visc)
        print(f'eta({args.visc} K) = {v:.4f} mPa.s' if v else 'outside validity')
