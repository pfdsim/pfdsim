"""
UNIFAC Activity Coefficient Model

Implements the UNIFAC (Universal Functional Activity Coefficient) group contribution
method for calculating liquid phase activity coefficients.

The model calculates activity coefficients using:
    ln(γᵢ) = ln(γᵢᶜ) + ln(γᵢᴿ)

where:
    - γᵢᶜ is the combinatorial contribution (size/shape effects)
    - γᵢᴿ is the residual contribution (group interaction energies)

References:
- Fredenslund, Jones, Prausnitz (1975) - Original UNIFAC
- Poling, Prausnitz, O'Connell - Properties of Gases and Liquids, 5th Ed.
"""

import math
import csv
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json


CLASSIC_UNIFAC_VARIANTS = ('UNIFAC', 'UNIFAC2')
MODIFIED_UNIFAC_VARIANTS = ('UNIFDMD', 'UNIFM2', 'UNIFNIST')

# RDKit is the molecular-graph backend for native UNIFAC fragmentation.
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    RDKIT_AVAILABLE = False


@dataclass
class UNIFACSubgroup:
    """UNIFAC subgroup parameters"""
    number: int           # Subgroup number
    name: str            # Subgroup name (e.g., 'CH3', 'OH')
    main_group: int      # Main group number
    main_group_name: str # Main group name (e.g., '[1]CH2')
    R: float             # Volume parameter (van der Waals volume)
    Q: float             # Surface area parameter (van der Waals surface)


@dataclass
class UNIFACComponent:
    """Component defined by its UNIFAC groups"""
    name: str
    groups: dict[int, int]  # subgroup_number -> count
    r: float = 0.0          # Molecular volume parameter (calculated)
    q: float = 0.0          # Molecular surface area parameter (calculated)
    
    def __post_init__(self):
        """Calculate r and q from groups (if database available)"""
        pass  # Will be calculated by UNIFACModel


class UNIFACModel:
    """
    UNIFAC activity coefficient model
    
    Usage:
        model = UNIFACModel()
        
        # Define components by their UNIFAC groups
        ethanol = {'CH3': 1, 'CH2': 1, 'OH': 1}
        water = {'H2O': 1}
        
        # Calculate activity coefficients
        gamma = model.activity_coefficients(
            components=[ethanol, water],
            x=[0.4, 0.6],
            T=298.15
        )
    """
    
    # Coordination number (standard value)
    Z = 10.0
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize UNIFAC model with group parameters
        
        Args:
            data_path: Path to UNIFAC data file (JSON or Excel)
                      If None, uses built-in parameters
        """
        self.subgroups: dict[int, UNIFACSubgroup] = {}
        self.subgroup_by_name: dict[str, UNIFACSubgroup] = {}
        self.interactions: dict[tuple[int, int], float] = {}  # (main_i, main_j) -> a_ij
        self.interaction_coefficients: dict[tuple[int, int], tuple[float, float, float]] = {}
        self.variant = 'UNIFAC'
        
        # Load parameters
        if data_path:
            self._load_from_file(data_path)
        else:
            self._load_builtin_parameters()
    
    def _load_from_file(self, path: str):
        """Load parameters from Excel or JSON file"""
        path = Path(path)

        if path.name == 'UNIFAC2rq.csv':
            self._load_unifac2(path)
            return
        if path.name == 'UNIFM2rq.xlsx':
            self._load_unifm2(path)
            return
        
        if path.suffix in ('.xlsx', '.xls'):
            self._load_from_excel(path)
        elif path.suffix == '.json':
            self._load_from_json(path)
        elif path.suffix == '.txt':
            self._load_dortmund_from_text(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_unifac2(self, rq_path: Path):
        """Load the UNIFAC 2.0 subgroup table and completed interaction matrix."""
        self.variant = 'UNIFAC2'
        with rq_path.open(newline='') as handle:
            for row in csv.DictReader(handle, delimiter=';'):
                main_group = int(row['Main Group No.'])
                main_group_name = row['Main Group Name'].strip()
                subgroup = UNIFACSubgroup(
                    number=int(row['Subgroup No.']),
                    name=row['Subgroup Name'].strip(),
                    main_group=main_group,
                    main_group_name=f"[{main_group}]{main_group_name}",
                    R=float(row['R']),
                    Q=float(row['Q']),
                )
                self._register_subgroup(subgroup)

        interaction_path = rq_path.with_name('UNIFAC2int.csv')
        with interaction_path.open(newline='') as handle:
            rows = csv.reader(handle, delimiter=';')
            main_groups = [int(value) for value in next(rows)[1:]]
            seen_rows = []
            for row in rows:
                main_i = int(row[0])
                seen_rows.append(main_i)
                if len(row) - 1 != len(main_groups):
                    raise ValueError(
                        f"UNIFAC2 interaction row {main_i} has the wrong length"
                    )
                for main_j, value in zip(main_groups, row[1:]):
                    self.interactions[(main_i, main_j)] = float(value)
        if seen_rows != main_groups:
            raise ValueError("UNIFAC2 interaction row and column groups do not match")

    def _load_unifm2(self, rq_path: Path):
        """Load the modified UNIFAC 2.0 subgroups and completed A/B matrices."""
        import pandas as pd

        self.variant = 'UNIFM2'
        groups_df = pd.read_excel(rq_path)
        for _, row in groups_df.iterrows():
            main_group = int(row['Main Group No.'])
            main_group_name = str(row['Main Group Name']).strip()
            subgroup = UNIFACSubgroup(
                number=int(row['Subgroup No.']),
                name=str(row['Subgroup Name']).strip(),
                main_group=main_group,
                main_group_name=f"[{main_group}]{main_group_name}",
                R=float(row['R']),
                Q=float(row['Q']),
            )
            self._register_subgroup(subgroup)

        a_matrix = pd.read_excel(
            rq_path.with_name('UNIFM2intA.xlsx'), index_col=0
        )
        b_matrix = pd.read_excel(
            rq_path.with_name('UNIFM2intB.xlsx'), index_col=0
        )
        a_groups = [int(value) for value in a_matrix.columns]
        b_groups = [int(value) for value in b_matrix.columns]
        a_rows = [int(value) for value in a_matrix.index]
        b_rows = [int(value) for value in b_matrix.index]
        if a_rows != a_groups or b_rows != b_groups or a_groups != b_groups:
            raise ValueError("UNIFM2 interaction matrix row and column groups do not match")

        for row_index, main_i in enumerate(a_groups):
            for column_index, main_j in enumerate(a_groups):
                a_ij = float(a_matrix.iloc[row_index, column_index])
                b_ij = float(b_matrix.iloc[row_index, column_index])
                self.interactions[(main_i, main_j)] = a_ij
                self.interaction_coefficients[(main_i, main_j)] = (
                    a_ij,
                    b_ij,
                    0.0,
                )

    @staticmethod
    def _parse_float_or_zero(value) -> float:
        """Parse an optional numeric parameter, treating blank cells as zero."""
        if value is None:
            return 0.0
        value = str(value).strip()
        if not value:
            return 0.0
        return float(value)

    @staticmethod
    def _canonical_subgroup_name(name: str, main_group: int,
                                 main_group_name: str) -> str:
        """Normalize historical UNIFAC display names that collide."""
        if name.upper() == 'CHO' and main_group == 13:
            return 'CH-O'
        return name

    def _register_subgroup(self, subgroup: UNIFACSubgroup,
                           aliases: Optional[list[str]] = None):
        subgroup.name = self._canonical_subgroup_name(
            subgroup.name,
            subgroup.main_group,
            subgroup.main_group_name,
        )
        self.subgroups[subgroup.number] = subgroup
        keys = [subgroup.name.upper()]
        if aliases:
            keys.extend(alias.upper() for alias in aliases)

        for key in keys:
            self.subgroup_by_name[key] = subgroup

    def _load_dortmund_from_text(self, path: Path):
        """Load Dortmund modified UNIFAC parameters from the published matrix text."""
        self.variant = 'UNIFDMD'
        lines = path.read_text().splitlines()
        section = 'interactions'

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('i\tj\t'):
                continue
            if stripped.startswith('No.\tSubgroup Name'):
                section = 'subgroups'
                continue

            parts = line.split('\t')
            if section == 'interactions':
                if len(parts) < 8:
                    continue
                i = int(parts[0])
                j = int(parts[1])
                a_ij = self._parse_float_or_zero(parts[2])
                b_ij = self._parse_float_or_zero(parts[3])
                c_ij = self._parse_float_or_zero(parts[4])
                a_ji = self._parse_float_or_zero(parts[5])
                b_ji = self._parse_float_or_zero(parts[6])
                c_ji = self._parse_float_or_zero(parts[7])
                self.interaction_coefficients[(i, j)] = (a_ij, b_ij, c_ij)
                self.interaction_coefficients[(j, i)] = (a_ji, b_ji, c_ji)
                self.interactions[(i, j)] = a_ij
                self.interactions[(j, i)] = a_ji
                continue

            if len(parts) < 6:
                continue
            try:
                number = int(parts[0])
                main_group = int(parts[2])
                r_value = float(parts[4])
                q_value = float(parts[5])
            except ValueError:
                continue

            subgroup = UNIFACSubgroup(
                number=number,
                name=parts[1].strip(),
                main_group=main_group,
                main_group_name=f"[{main_group}]{parts[3].strip()}",
                R=r_value,
                Q=q_value,
            )
            self._register_subgroup(subgroup)

        for mg in set(sg.main_group for sg in self.subgroups.values()):
            self.interaction_coefficients[(mg, mg)] = (0.0, 0.0, 0.0)
            self.interactions[(mg, mg)] = 0.0

    def _load_from_excel(self, path: Path):
        """Load parameters from Excel file"""
        import pandas as pd
        
        # Read groups sheet
        groups_df = pd.read_excel(path, sheet_name='Groups')
        
        for _, row in groups_df.iterrows():
            # Parse main group number from format like '[1]CH2'
            main_group_str = str(row['Maingroup'])
            main_group_num = int(main_group_str.split('[')[1].split(']')[0])
            
            subgroup = UNIFACSubgroup(
                number=int(row['No.']),
                name=str(row['Subgroup Name']).strip(),
                main_group=main_group_num,
                main_group_name=main_group_str,
                R=float(row['R']),
                Q=float(row['Q'])
            )
            self._register_subgroup(subgroup)
        
        # Read interactions sheet
        interactions_df = pd.read_excel(path, sheet_name='Interactions')
        
        for _, row in interactions_df.iterrows():
            i = int(row['i'])
            j = int(row['j'])
            a_ij = float(row['Aij'])
            a_ji = float(row['Aji'])
            
            self.interactions[(i, j)] = a_ij
            self.interactions[(j, i)] = a_ji
        
        # Self-interactions are zero
        for mg in set(sg.main_group for sg in self.subgroups.values()):
            self.interactions[(mg, mg)] = 0.0
    
    def _load_builtin_parameters(self):
        """Load built-in UNIFAC parameters for common groups"""
        # Common subgroups
        builtin_subgroups = [
            # Main group 1: CH2
            (1, 'CH3', 1, '[1]CH2', 0.9011, 0.848),
            (2, 'CH2', 1, '[1]CH2', 0.6744, 0.540),
            (3, 'CH', 1, '[1]CH2', 0.4469, 0.228),
            (4, 'C', 1, '[1]CH2', 0.2195, 0.000),
            # Main group 2: C=C
            (5, 'CH2=CH', 2, '[2]C=C', 1.3454, 1.176),
            (6, 'CH=CH', 2, '[2]C=C', 1.1167, 0.867),
            # Main group 3: ACH (aromatic)
            (9, 'ACH', 3, '[3]ACH', 0.5313, 0.400),
            (10, 'AC', 3, '[3]ACH', 0.3652, 0.120),
            # Main group 4: ACCH2
            (11, 'ACCH3', 4, '[4]ACCH2', 1.2663, 0.968),
            (12, 'ACCH2', 4, '[4]ACCH2', 1.0396, 0.660),
            # Main group 5: OH
            (14, 'OH', 5, '[5]OH', 1.0000, 1.200),
            # Main group 6: CH3OH
            (15, 'CH3OH', 6, '[6]CH3OH', 1.4311, 1.432),
            # Main group 7: H2O
            (16, 'H2O', 7, '[7]H2O', 0.9200, 1.400),
            # Main group 9: CH2CO (ketones)
            (18, 'CH3CO', 9, '[9]CH2CO', 1.6724, 1.488),
            (19, 'CH2CO', 9, '[9]CH2CO', 1.4457, 1.180),
            # Main group 10: CHO (aldehydes)
            (20, 'CHO', 10, '[10]CHO', 0.9980, 0.948),
            # Main group 11: CCOO (esters)
            (21, 'CH3COO', 11, '[11]CCOO', 1.9031, 1.728),
            (22, 'CH2COO', 11, '[11]CCOO', 1.6764, 1.420),
            # Main group 13: CH2O (ethers)
            (24, 'CH3O', 13, '[13]CH2O', 1.1450, 1.088),
            (25, 'CH2O', 13, '[13]CH2O', 0.9183, 0.780),
            (26, 'CH-O', 13, '[13]CH2O', 0.6908, 0.468),
            # Main group 20: COOH (carboxylic acids)
            (42, 'COOH', 20, '[20]COOH', 1.3013, 1.224),
            (43, 'HCOOH', 20, '[20]COOH', 1.5280, 1.532),
        ]
        
        for num, name, main_group, main_name, R, Q in builtin_subgroups:
            subgroup = UNIFACSubgroup(num, name, main_group, main_name, R, Q)
            self._register_subgroup(subgroup)
        
        # Common interaction parameters (a_ij in K)
        # Format: (main_i, main_j, a_ij, a_ji)
        builtin_interactions = [
            # CH2 interactions
            (1, 2, 86.02, -35.36),    # CH2-C=C
            (1, 3, 61.13, -11.12),    # CH2-ACH
            (1, 4, 76.50, -69.70),    # CH2-ACCH2
            (1, 5, 986.50, 156.40),   # CH2-OH
            (1, 6, 697.20, 16.51),    # CH2-CH3OH
            (1, 7, 1318.00, 300.00),  # CH2-H2O
            (1, 9, 476.40, 26.76),    # CH2-CH2CO
            (1, 10, 677.00, 505.70),  # CH2-CHO
            (1, 11, 232.10, 114.80),  # CH2-CCOO
            (1, 13, 251.50, 83.36),   # CH2-CH2O
            (1, 20, 663.50, 315.30),  # CH2-COOH
            # C=C interactions
            (2, 5, 524.10, 457.00),   # C=C-OH
            (2, 7, 270.60, 496.10),   # C=C-H2O
            # ACH interactions
            (3, 5, 636.10, 89.60),    # ACH-OH
            (3, 7, 903.80, 362.30),   # ACH-H2O
            (3, 9, 25.77, -52.10),    # ACH-CH2CO
            # ACCH2 interactions
            (4, 5, 803.20, 25.82),    # ACCH2-OH
            (4, 7, 5695.00, 377.60),  # ACCH2-H2O
            # OH interactions
            (5, 6, -137.10, 249.10),  # OH-CH3OH
            (5, 7, 353.50, -229.10),  # OH-H2O
            (5, 9, 164.50, -356.10),  # OH-CH2CO
            (5, 10, -203.60, 267.80), # OH-CHO
            (5, 11, 101.10, -117.60), # OH-CCOO
            (5, 13, 28.06, -128.60),  # OH-CH2O
            (5, 20, 199.00, -151.00), # OH-COOH
            # CH3OH interactions
            (6, 7, -181.00, 289.60),  # CH3OH-H2O
            (6, 9, -7.838, -186.70),  # CH3OH-CH2CO
            (6, 11, -10.72, 185.10),  # CH3OH-CCOO
            (6, 13, -128.60, -17.50), # CH3OH-CH2O
            (6, 20, 339.80, -178.50), # CH3OH-COOH
            # H2O interactions
            (7, 9, 472.50, -195.40),  # H2O-CH2CO
            (7, 10, -116.00, 304.10), # H2O-CHO
            (7, 11, 200.80, -36.72),  # H2O-CCOO
            (7, 13, 540.50, -314.70), # H2O-CH2O
            (7, 20, -14.09, -66.17),  # H2O-COOH
            # CH2CO interactions
            (9, 11, -213.70, 372.20), # CH2CO-CCOO
            (9, 13, -103.60, 191.10), # CH2CO-CH2O
            (9, 20, 669.40, 0.0),     # CH2CO-COOH
            # CHO interactions
            (10, 20, 497.50, -18.80), # CHO-COOH
            # CCOO interactions
            (11, 13, -235.70, 461.30), # CCOO-CH2O
            (11, 20, 660.20, 664.60),  # CCOO-COOH
            # CH2O interactions
            (13, 20, 664.60, 94.92),   # CH2O-COOH
        ]
        
        for i, j, a_ij, a_ji in builtin_interactions:
            self.interactions[(i, j)] = a_ij
            self.interactions[(j, i)] = a_ji
        
        # Self-interactions are zero
        for mg in set(sg.main_group for sg in self.subgroups.values()):
            self.interactions[(mg, mg)] = 0.0
    
    def save_to_json(self, path: str):
        """Save parameters to JSON file"""
        # Collect unique pairs with both a_ij and a_ji
        interaction_pairs = {}
        for (i, j), a_val in self.interactions.items():
            if i == j:
                continue  # Skip self-interactions (always 0)
            key = (min(i, j), max(i, j))
            if key not in interaction_pairs:
                interaction_pairs[key] = {}
            if i < j:
                interaction_pairs[key]['a_ij'] = a_val
            else:
                interaction_pairs[key]['a_ji'] = a_val
        
        data = {
            'subgroups': [
                {
                    'number': sg.number,
                    'name': sg.name,
                    'main_group': sg.main_group,
                    'main_group_name': sg.main_group_name,
                    'R': sg.R,
                    'Q': sg.Q
                }
                for sg in self.subgroups.values()
            ],
            'interactions': [
                {
                    'i': key[0], 
                    'j': key[1], 
                    'a_ij': vals.get('a_ij', 0.0),
                    'a_ji': vals.get('a_ji', 0.0)
                }
                for key, vals in interaction_pairs.items()
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_from_json(self, path: Path):
        """Load parameters from JSON file"""
        with open(path) as f:
            data = json.load(f)

        metadata = data.get('metadata', {})
        if metadata.get('model') == 'NIST-modified UNIFAC':
            self.variant = 'UNIFNIST'
        elif metadata.get('model') == 'Lyngby modified UNIFAC':
            self.variant = 'UNIFLBY'
        
        for sg_data in data['subgroups']:
            subgroup = UNIFACSubgroup(**{
                key: sg_data[key]
                for key in ('number', 'name', 'main_group', 'main_group_name', 'R', 'Q')
            })
            self._register_subgroup(subgroup)
        
        for inter in data['interactions']:
            i, j = inter['i'], inter['j']
            if 'a1' in inter:
                coeffs = (
                    float(inter.get('a1', 0.0)),
                    float(inter.get('a2', 0.0)),
                    float(inter.get('a3', 0.0)),
                )
                self.interaction_coefficients[(i, j)] = coeffs
                self.interactions[(i, j)] = coeffs[0]
                continue
            a_ij = inter.get('a_ij', 0.0)
            a_ji = inter.get('a_ji', a_ij)  # Fallback for old format
            self.interactions[(i, j)] = a_ij
            self.interactions[(j, i)] = a_ji
        
        # Self-interactions are zero
        for mg in set(sg.main_group for sg in self.subgroups.values()):
            self.interactions[(mg, mg)] = 0.0
            self.interaction_coefficients[(mg, mg)] = (0.0, 0.0, 0.0)
    
    def _resolve_groups(self, groups: dict) -> dict[int, int]:
        """
        Convert group names/numbers to subgroup numbers
        
        Args:
            groups: Dict of group names or numbers to counts
                   e.g., {'CH3': 2, 'OH': 1} or {1: 2, 14: 1}
        
        Returns:
            Dict of subgroup numbers to counts
        """
        resolved = {}
        if self.variant == 'UNIFLBY' and (not groups or any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in groups.values())):
            raise ValueError('Lyngby UNIFAC requires positive finite group counts')
        
        for group_id, count in groups.items():
            if isinstance(group_id, int):
                # Already a subgroup number
                if group_id not in self.subgroups:
                    raise ValueError(f"Unknown subgroup number: {group_id}")
                resolved[group_id] = count
            else:
                # Group name - look up
                name_upper = str(group_id).upper().strip()
                if name_upper.isdigit():
                    subgroup_number = int(name_upper)
                    if subgroup_number not in self.subgroups:
                        raise ValueError(f"Unknown subgroup number: {subgroup_number}")
                    resolved[subgroup_number] = count
                    continue
                if name_upper not in self.subgroup_by_name:
                    raise ValueError(f"Unknown subgroup name: {group_id}")
                sg = self.subgroup_by_name[name_upper]
                resolved[sg.number] = count
        
        return resolved
    
    def calculate_r_q(self, groups: dict) -> tuple[float, float]:
        """
        Calculate molecular r and q parameters from groups
        
        Args:
            groups: Dict of subgroup names/numbers to counts
        
        Returns:
            (r, q) tuple
        """
        resolved = self._resolve_groups(groups)
        
        r = sum(self.subgroups[sg_num].R * count 
                for sg_num, count in resolved.items())
        q = sum(self.subgroups[sg_num].Q * count 
                for sg_num, count in resolved.items())
        
        return r, q
    
    def get_interaction(self, main_i: int, main_j: int) -> float:
        """Get interaction parameter a_ij (K)"""
        key = (main_i, main_j)
        if key in self.interactions:
            return self.interactions[key]
        if self.variant == 'UNIFLBY':
            raise ValueError(f'Lyngby UNIFAC interaction {main_i}->{main_j} is unavailable')
        # If not found, return 0 (same as self-interaction)
        return 0.0
    
    def activity_coefficients(self, 
                             components: list[dict],
                             x: list[float],
                             T: float) -> list[float]:
        """
        Calculate activity coefficients for all components
        
        Args:
            components: List of dicts specifying groups for each component
                       e.g., [{'CH3': 1, 'CH2': 1, 'OH': 1}, {'H2O': 1}]
            x: Mole fractions (must sum to 1.0)
            T: Temperature [K]
        
        Returns:
            List of activity coefficients for each component
        """
        n_comp = len(components)
        
        if len(x) != n_comp:
            raise ValueError("Number of mole fractions must match number of components")
        
        if abs(sum(x) - 1.0) > 0.01:
            raise ValueError(f"Mole fractions must sum to 1.0, got {sum(x)}")
        
        # Resolve all groups to subgroup numbers
        comp_groups = [self._resolve_groups(comp) for comp in components]
        
        # Calculate r and q for each component
        r = []
        q = []
        for groups in comp_groups:
            ri, qi = self.calculate_r_q(groups)
            r.append(ri)
            q.append(qi)
        
        # Combinatorial contribution
        ln_gamma_c = self._combinatorial(x, r, q)
        
        # Residual contribution
        ln_gamma_r = self._residual(comp_groups, x, T)
        
        # Total activity coefficient
        gamma = [math.exp(ln_gamma_c[i] + ln_gamma_r[i]) for i in range(n_comp)]
        
        return gamma
    
    def _combinatorial(self, x: list[float], r: list[float], q: list[float]) -> list[float]:
        """
        Calculate combinatorial contribution to ln(γ)
        
        Uses the Staverman-Guggenheim combinatorial term:
        ln(γᵢᶜ) = ln(Φᵢ/xᵢ) + (z/2)qᵢ·ln(θᵢ/Φᵢ) + lᵢ - (Φᵢ/xᵢ)·Σⱼxⱼlⱼ
        
        Modified UNIFAC uses a simpler form without the Staverman-Guggenheim term.
        """
        if self.variant in MODIFIED_UNIFAC_VARIANTS:
            return self._combinatorial_dortmund(x, r, q)
        if self.variant == 'UNIFLBY':
            volumes = [value ** (2.0 / 3.0) for value in r]
            total = sum(xi * vi for xi, vi in zip(x, volumes))
            return [1.0 - vi / total + math.log(vi / total) for vi in volumes]

        n_comp = len(x)
        z = self.Z
        
        # Calculate l parameter: lᵢ = (z/2)(rᵢ - qᵢ) - (rᵢ - 1)
        l = [(z/2) * (r[i] - q[i]) - (r[i] - 1) for i in range(n_comp)]
        
        # Sum of x*r and x*q
        sum_xr = sum(x[i] * r[i] for i in range(n_comp))
        sum_xq = sum(x[i] * q[i] for i in range(n_comp))
        
        # Avoid division by zero for pure components
        if sum_xr < 1e-10:
            sum_xr = 1e-10
        if sum_xq < 1e-10:
            sum_xq = 1e-10
        
        # Sum of x*l
        sum_xl = sum(x[i] * l[i] for i in range(n_comp))
        
        # Calculate ln(γᶜ) for each component
        ln_gamma_c = []
        for i in range(n_comp):
            phi_over_x = r[i] / sum_xr
            theta_over_phi = q[i] * sum_xr / (r[i] * sum_xq)
            
            if phi_over_x <= 0:
                phi_over_x = 1e-10
            if theta_over_phi <= 0:
                theta_over_phi = 1e-10
            
            term1 = math.log(phi_over_x)
            term2 = (z/2) * q[i] * math.log(theta_over_phi)
            term3 = l[i]
            term4 = -phi_over_x * sum_xl
            
            ln_gamma_c.append(term1 + term2 + term3 + term4)
        
        return ln_gamma_c

    def _combinatorial_dortmund(self, x: list[float], r: list[float],
                                q: list[float]) -> list[float]:
        """
        Dortmund modified UNIFAC combinatorial contribution.

        ln(gamma_i^C) = 1 - V'_i + ln(V'_i)
            - 5 q_i [1 - V_i/F_i + ln(V_i/F_i)]
        where V'_i uses r_i^(3/4).
        """
        sum_xr = sum(x[i] * r[i] for i in range(len(x))) or 1e-10
        sum_xq = sum(x[i] * q[i] for i in range(len(x))) or 1e-10
        sum_xr34 = sum(x[i] * (r[i] ** 0.75) for i in range(len(x))) or 1e-10

        ln_gamma_c = []
        for i in range(len(x)):
            v_prime = (r[i] ** 0.75) / sum_xr34
            v_value = r[i] / sum_xr
            f_value = q[i] / sum_xq
            ratio = v_value / f_value if f_value > 1e-10 else 1.0

            v_prime = max(v_prime, 1e-10)
            ratio = max(ratio, 1e-10)

            ln_gamma_c.append(
                1.0
                - v_prime
                + math.log(v_prime)
                - 5.0 * q[i] * (1.0 - ratio + math.log(ratio))
            )

        return ln_gamma_c
    
    def _residual(self, comp_groups: list[dict], x: list[float], T: float) -> list[float]:
        """
        Calculate residual contribution to ln(γ)
        
        ln(γᵢᴿ) = Σₖνₖ⁽ⁱ⁾[ln(Γₖ) - ln(Γₖ⁽ⁱ⁾)]
        
        where Γₖ is the group activity coefficient
        """
        n_comp = len(comp_groups)
        
        # Get all unique subgroups
        all_subgroups = set()
        for groups in comp_groups:
            all_subgroups.update(groups.keys())
        
        # Calculate group mole fractions in mixture
        X_mix = self._group_mole_fractions(comp_groups, x)
        
        # Calculate group activity coefficients in mixture
        ln_Gamma_mix = self._group_activity_coefficients(X_mix, T)
        
        # Calculate residual contribution for each component
        ln_gamma_r = []
        
        for i in range(n_comp):
            # Group mole fractions in pure component i
            X_pure = self._group_mole_fractions([comp_groups[i]], [1.0])
            
            # Group activity coefficients in pure component i
            ln_Gamma_pure = self._group_activity_coefficients(X_pure, T)
            
            # Sum over all groups in component i
            ln_gamma_ri = 0.0
            for sg_num, count in comp_groups[i].items():
                ln_Gamma_k_mix = ln_Gamma_mix.get(sg_num, 0.0)
                ln_Gamma_k_pure = ln_Gamma_pure.get(sg_num, 0.0)
                ln_gamma_ri += count * (ln_Gamma_k_mix - ln_Gamma_k_pure)
            
            ln_gamma_r.append(ln_gamma_ri)
        
        return ln_gamma_r
    
    def _group_mole_fractions(self, comp_groups: list[dict], x: list[float]) -> dict[int, float]:
        """
        Calculate group mole fractions
        
        Xₘ = Σᵢxᵢνₘ⁽ⁱ⁾ / Σᵢxᵢ Σₖνₖ⁽ⁱ⁾
        """
        # Numerator: sum over all components of x*nu for each group
        numerator = {}
        
        # Denominator: sum over all components of x * (total groups in component)
        denominator = 0.0
        
        for i, groups in enumerate(comp_groups):
            total_groups_i = sum(groups.values())
            denominator += x[i] * total_groups_i
            
            for sg_num, count in groups.items():
                if sg_num not in numerator:
                    numerator[sg_num] = 0.0
                numerator[sg_num] += x[i] * count
        
        if denominator < 1e-10:
            denominator = 1e-10
        
        X = {sg_num: num / denominator for sg_num, num in numerator.items()}
        
        return X
    
    def _group_activity_coefficients(self, X: dict[int, float], T: float) -> dict[int, float]:
        """
        Calculate group activity coefficients
        
        ln(Γₖ) = Qₖ[1 - ln(Σₘθₘψₘₖ) - Σₘ(θₘψₖₘ / Σₙθₙψₙₘ)]
        """
        # Calculate group surface area fractions
        # θₘ = QₘXₘ / ΣₙQₙXₙ
        sum_QX = sum(self.subgroups[sg].Q * X[sg] for sg in X)
        
        if sum_QX < 1e-10:
            sum_QX = 1e-10
        
        theta = {sg: self.subgroups[sg].Q * X[sg] / sum_QX for sg in X}
        
        # Calculate temperature-dependent interaction parameters
        # ψₘₙ = exp(-aₘₙ/T)
        psi = {}
        for sg_m in X:
            main_m = self.subgroups[sg_m].main_group
            for sg_n in X:
                main_n = self.subgroups[sg_n].main_group
                psi[(sg_m, sg_n)] = self._interaction_psi(main_m, main_n, T)
        
        # Calculate ln(Γₖ) for each group
        ln_Gamma = {}
        
        for sg_k in X:
            Q_k = self.subgroups[sg_k].Q
            
            # Term 1: Σₘθₘψₘₖ
            sum_theta_psi_mk = sum(theta[sg_m] * psi.get((sg_m, sg_k), 1.0) 
                                   for sg_m in X)
            
            if sum_theta_psi_mk < 1e-10:
                sum_theta_psi_mk = 1e-10
            
            # Term 2: Σₘ(θₘψₖₘ / Σₙθₙψₙₘ)
            term2 = 0.0
            for sg_m in X:
                sum_theta_psi_nm = sum(theta[sg_n] * psi.get((sg_n, sg_m), 1.0) 
                                       for sg_n in X)
                if sum_theta_psi_nm < 1e-10:
                    sum_theta_psi_nm = 1e-10
                
                term2 += theta[sg_m] * psi.get((sg_k, sg_m), 1.0) / sum_theta_psi_nm
            
            ln_Gamma[sg_k] = Q_k * (1 - math.log(sum_theta_psi_mk) - term2)
        
        return ln_Gamma

    def _interaction_psi(self, main_m: int, main_n: int, T: float) -> float:
        """Calculate the UNIFAC group interaction factor psi_mn."""
        coeffs = self.interaction_coefficients.get((main_m, main_n))
        if coeffs is not None:
            a_mn, b_mn, c_mn = coeffs
            if self.variant == 'UNIFLBY':
                return math.exp(-(
                    a_mn + b_mn * (T - 298.15)
                    + c_mn * (T * math.log(298.15 / T) + T - 298.15)
                ) / T)
            return math.exp(-(a_mn + b_mn * T + c_mn * T * T) / T)
        a_mn = self.get_interaction(main_m, main_n)
        return math.exp(-a_mn / T)


# Native, parameter-set-specific group fragmentation


def parse_smiles_to_unifac(smiles: str, variant: str = 'UNIFAC') -> dict[str | int, int]:
    """Parse a SMILES string with pfdsim's strict native fragmenter."""
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .unifac_fragmenter import fragment
    else:
        from unifac_fragmenter import fragment
    return fragment(smiles, variant)


def smiles_to_unifac_safe(smiles: str, variant: str = 'UNIFAC') -> Optional[dict[str | int, int]]:
    """Safely parse SMILES to UNIFAC groups, returning None on failure."""
    try:
        return parse_smiles_to_unifac(smiles, variant)
    except Exception as e:
        print(f"Warning: Could not parse SMILES '{smiles}': {e}")
        return None


def _resolve_smiles_locally(
    identifier: str,
    expected_mw: Optional[float] = None,
) -> Optional[str]:
    """Return a local/OPSIN SMILES string through the shared resolver."""
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .chemical_properties import ChemicalDatabase
        else:
            from chemical_properties import ChemicalDatabase
    except Exception:
        return None
    try:
        result = ChemicalDatabase(enable_online=False).resolve_smiles_info(
            identifier,
            fetch_online=False,
            expected_mw=expected_mw,
        )
        return result.smiles if result else None
    except Exception:
        return None


# Common molecules with manual UNIFAC group assignments
KNOWN_MOLECULES = {
    # Simple organics
    'methanol': {'CH3OH': 1},
    'CH3OH': {'CH3OH': 1},
    'ethanol': {'CH3': 1, 'CH2': 1, 'OH': 1},
    'C2H5OH': {'CH3': 1, 'CH2': 1, 'OH': 1},
    'propanol': {'CH3': 1, 'CH2': 2, 'OH': 1},
    '1-propanol': {'CH3': 1, 'CH2': 2, 'OH': 1},
    '2-propanol': {'CH3': 2, 'CH': 1, 'OH': 1},
    'isopropanol': {'CH3': 2, 'CH': 1, 'OH': 1},
    'butanol': {'CH3': 1, 'CH2': 3, 'OH': 1},
    '1-butanol': {'CH3': 1, 'CH2': 3, 'OH': 1},
    
    # Alkanes
    'methane': {'CH3': 1},  # Approximate - CH4 doesn't fit well
    'ethane': {'CH3': 2},
    'propane': {'CH3': 2, 'CH2': 1},
    'butane': {'CH3': 2, 'CH2': 2},
    'n-butane': {'CH3': 2, 'CH2': 2},
    'pentane': {'CH3': 2, 'CH2': 3},
    'n-pentane': {'CH3': 2, 'CH2': 3},
    'C5H12': {'CH3': 2, 'CH2': 3},
    'hexane': {'CH3': 2, 'CH2': 4},
    'n-hexane': {'CH3': 2, 'CH2': 4},
    'C6H14': {'CH3': 2, 'CH2': 4},
    '2-methylpentane': {'CH3': 3, 'CH2': 2, 'CH': 1},
    '3-methylpentane': {'CH3': 3, 'CH2': 2, 'CH': 1},
    'heptane': {'CH3': 2, 'CH2': 5},
    'n-heptane': {'CH3': 2, 'CH2': 5},
    'C7H16': {'CH3': 2, 'CH2': 5},
    'octane': {'CH3': 2, 'CH2': 6},
    'n-octane': {'CH3': 2, 'CH2': 6},
    'C8H18': {'CH3': 2, 'CH2': 6},
    'nonane': {'CH3': 2, 'CH2': 7},
    'decane': {'CH3': 2, 'CH2': 8},
    'undecane': {'CH3': 2, 'CH2': 9},
    'n-undecane': {'CH3': 2, 'CH2': 9},
    'C11H24': {'CH3': 2, 'CH2': 9},
    'cyclohexane': {'CH2': 6},
    'cyclopentane': {'CH2': 5},
    
    # Aromatics
    'benzene': {'ACH': 6},
    'toluene': {'ACH': 5, 'ACCH3': 1},
    'ethylbenzene': {'ACH': 5, 'ACCH2': 1, 'CH3': 1},
    'xylene': {'ACH': 4, 'ACCH3': 2},
    
    # Water
    'water': {'H2O': 1},
    'H2O': {'H2O': 1},
    
    # Ketones
    'acetone': {'CH3': 1, 'CH3CO': 1},
    'CH3COCH3': {'CH3': 1, 'CH3CO': 1},
    '2-butanone': {'CH3': 1, 'CH2': 1, 'CH3CO': 1},
    'MEK': {'CH3': 1, 'CH2': 1, 'CH3CO': 1},
    
    # Aldehydes
    'acetaldehyde': {'CH3': 1, 'CHO': 1},
    'formaldehyde': {'CHO': 1},  # Approximate
    
    # Acids
    'acetic acid': {'CH3': 1, 'COOH': 1},
    'acrylic acid': {'CH2=CH': 1, 'COOH': 1},
    'formic acid': {'HCOOH': 1},
    'propionic acid': {'CH3': 1, 'CH2': 1, 'COOH': 1},
    
    # Esters
    'ethyl acetate': {'CH3': 1, 'CH2': 1, 'CH3COO': 1},
    'methyl acetate': {'CH3': 1, 'CH3COO': 1},
    
    # Ethers
    'diethyl ether': {'CH3': 2, 'CH2': 1, 'CH2O': 1},
    'DME': {'CH3': 1, 'CH3O': 1},
    
    # Chlorinated
    'chloroform': {'CHCL3': 1},
    'dichloromethane': {'CH2CL2': 1},
    'carbon tetrachloride': {'CCL4': 1},
    
    # Amines
    'methylamine': {'CH3NH2': 1},
    'ethylamine': {'CH3': 1, 'CH2NH2': 1},
    'dimethylamine': {'CH3': 1, 'CH3NH': 1},
    
    # Nitriles
    'acetonitrile': {'CH3CN': 1},
}


DORTMUND_PRIMARY_ALCOHOLS = {
    'ethanol', 'C2H5OH',
    '1-propanol',
    '1-butanol',
}
DORTMUND_SECONDARY_ALCOHOLS = {'2-propanol', 'isopropanol'}
DORTMUND_TERTIARY_ALCOHOLS: set[str] = set()

MODIFIED_UNIFAC_KNOWN_OVERRIDES = {
    # Modified UNIFAC distinguishes cyclic alkanes from acyclic CH2.
    # Subgroup 78 is CY-CH2 in Dortmund and c-CH2 in NIST.
    'cyclohexane': {78: 6},
    'cyclopentane': {78: 5},
}

DORTMUND_UNIFAC_KNOWN_OVERRIDES = {
    # Dortmund subgroup 119 represents the complete unsubstituted oxirane
    # ring.  Generic ring fragmentation incorrectly decomposes it into
    # CY-CH2 and CY-CH2O groups.
    '75-21-8': {119: 1},
    'ethylene oxide': {119: 1},
    'oxirane': {119: 1},
    # Dortmund documents propylene oxide as one methyl group plus the
    # substituted epoxide-ring subgroup H2COCH.
    'C3H6O_PO': {1: 1, 107: 1},
    '75-56-9': {1: 1, 107: 1},
    'propylene oxide': {1: 1, 107: 1},
    '1,2-propylene oxide': {1: 1, 107: 1},
    '1,2-epoxypropane': {1: 1, 107: 1},
    'methyloxirane': {1: 1, 107: 1},
}

NIST_UNIFAC_KNOWN_OVERRIDES = {
    # Keep common aliases independent of structure-name resolution.  The
    # native fragmenter assigns the same dedicated groups from SMILES.
    'acetic anhydride': {1: 2, 121: 1},
    # Dedicated NIST modified UNIFAC diol subgroup.
    'ethylene glycol': {62: 1},
    '1,2-ethanediol': {62: 1},
    # No dedicated benzyl alcohol subgroup in the NIST table; use standard
    # aromatic, benzylic methylene, and primary alcohol groups.
    'benzyl alcohol': {9: 5, 12: 1, 14: 1},
}


def _groups_for_modified_unifac(identifier: str, groups: dict[str, int]) -> dict:
    """
    Convert built-in molecule group assignments to modified-UNIFAC groups.

    Only built-in molecules with clear primary/secondary/tertiary OH subgroup
    assignments are converted here. Caller-provided group dictionaries are
    resolved directly against the selected modified-UNIFAC table.
    """
    result = {}
    key = identifier.strip()
    name_lower = key.lower()
    if key in MODIFIED_UNIFAC_KNOWN_OVERRIDES:
        return dict(MODIFIED_UNIFAC_KNOWN_OVERRIDES[key])
    if name_lower in MODIFIED_UNIFAC_KNOWN_OVERRIDES:
        return dict(MODIFIED_UNIFAC_KNOWN_OVERRIDES[name_lower])

    for group, count in groups.items():
        if isinstance(group, int):
            result[group] = count
            continue

        group_key = str(group)
        group_upper = group_key.upper()

        if group_upper == 'OH':
            if key in DORTMUND_PRIMARY_ALCOHOLS or name_lower in DORTMUND_PRIMARY_ALCOHOLS:
                result[14] = count
            elif key in DORTMUND_SECONDARY_ALCOHOLS or name_lower in DORTMUND_SECONDARY_ALCOHOLS:
                result[81] = count
            elif key in DORTMUND_TERTIARY_ALCOHOLS or name_lower in DORTMUND_TERTIARY_ALCOHOLS:
                result[82] = count
            else:
                raise ValueError(
                    f"Modified UNIFAC needs a primary/secondary/tertiary OH subgroup "
                    f"for '{identifier}'. Specify subgroup 14, 81, or 82 explicitly."
                )
        else:
            result[group] = count

    return result


def get_unifac_groups(identifier: str, smiles: Optional[str] = None,
                      variant: str = 'UNIFAC',
                      expected_mw: Optional[float] = None) -> dict:
    """
    Get UNIFAC groups for a molecule.

    Tries:
    1. Curated local molecule assignments
    2. Native fragmentation from caller-provided SMILES
    3. Local SMILES lookup, then native fragmentation

    Args:
        identifier: Molecule name or formula
        smiles: Optional SMILES string
        expected_mw: Optional molecular weight used to reject synonym-search
            mismatches during local SMILES resolution

    Returns:
        Dict of group names to counts
    """
    variant = variant.upper()

    if variant == 'UNIFLBY':
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .lyngby_parameters import convert_classic_groups
        else:
            from lyngby_parameters import convert_classic_groups
        # Reuse the structural assignments, translating only groups with an
        # exact Lyngby counterpart or a defined decomposition.
        if identifier.strip().lower() in ('methane', 'ch4', 'formaldehyde', 'ch2o'):
            raise ValueError(f'No exact base Lyngby groups for {identifier}')
        groups = get_unifac_groups(identifier, smiles, 'UNIFAC', expected_mw)
        return convert_classic_groups(groups)

    # Check known molecules
    name_lower = identifier.lower().strip()
    if variant in ('UNIFDMD', 'UNIFM2'):
        key = identifier.strip()
        if key in DORTMUND_UNIFAC_KNOWN_OVERRIDES:
            return dict(DORTMUND_UNIFAC_KNOWN_OVERRIDES[key])
        if name_lower in DORTMUND_UNIFAC_KNOWN_OVERRIDES:
            return dict(DORTMUND_UNIFAC_KNOWN_OVERRIDES[name_lower])
    if variant == 'UNIFNIST':
        key = identifier.strip()
        if key in NIST_UNIFAC_KNOWN_OVERRIDES:
            return dict(NIST_UNIFAC_KNOWN_OVERRIDES[key])
        if name_lower in NIST_UNIFAC_KNOWN_OVERRIDES:
            return dict(NIST_UNIFAC_KNOWN_OVERRIDES[name_lower])

    if name_lower in KNOWN_MOLECULES:
        groups = KNOWN_MOLECULES[name_lower]
        if variant in MODIFIED_UNIFAC_VARIANTS:
            return _groups_for_modified_unifac(identifier, groups)
        return groups

    # Check by formula (case-sensitive)
    if identifier in KNOWN_MOLECULES:
        groups = KNOWN_MOLECULES[identifier]
        if variant in MODIFIED_UNIFAC_VARIANTS:
            return _groups_for_modified_unifac(identifier, groups)
        return groups
    
    if smiles:
        try:
            return parse_smiles_to_unifac(smiles, variant)
        except Exception as exc:
            raise ValueError(
                f"Cannot determine UNIFAC groups for SMILES '{smiles}' "
                f"({identifier}): {exc}"
            ) from exc

    resolved_smiles = _resolve_smiles_locally(identifier, expected_mw)
    if resolved_smiles:
        try:
            return parse_smiles_to_unifac(resolved_smiles, variant)
        except Exception:
            pass

    raise ValueError(f"Cannot determine UNIFAC groups for '{identifier}'. "
                     f"Please specify groups manually or provide resolver-cached SMILES.")


# Convenience function for thermodynamics integration
def calculate_activity_coefficients(components: list[str],
                                   x: list[float],
                                   T: float,
                                   groups: Optional[list[dict]] = None,
                                   smiles: Optional[list[str]] = None,
                                   data_path: Optional[str] = None) -> list[float]:
    """
    Calculate UNIFAC activity coefficients
    
    Args:
        components: List of component names
        x: Mole fractions
        T: Temperature [K]
        groups: Optional list of group dicts (if not provided, will try to look up)
        smiles: Optional list of SMILES strings
        data_path: Optional path to UNIFAC parameter file
    
    Returns:
        List of activity coefficients
    """
    model = UNIFACModel(data_path)
    
    # Get groups for each component
    if groups is None:
        groups = []
        for i, comp in enumerate(components):
            smi = smiles[i] if smiles and i < len(smiles) else None
            comp_groups = get_unifac_groups(comp, smi)
            groups.append(comp_groups)
    
    return model.activity_coefficients(groups, x, T)


if __name__ == '__main__':
    # Test the model
    print("=== UNIFAC Activity Coefficient Model Test ===\n")
    
    # Create model with built-in parameters
    model = UNIFACModel()
    
    # Test 1: Ethanol-Water system at 25°C
    print("Test 1: Ethanol-Water at 25°C")
    ethanol = {'CH3': 1, 'CH2': 1, 'OH': 1}
    water = {'H2O': 1}
    
    # Calculate r and q
    r_eth, q_eth = model.calculate_r_q(ethanol)
    r_h2o, q_h2o = model.calculate_r_q(water)
    print(f"  Ethanol: r={r_eth:.4f}, q={q_eth:.4f}")
    print(f"  Water:   r={r_h2o:.4f}, q={q_h2o:.4f}")
    
    # Activity coefficients at different compositions
    print("\n  x_EtOH    γ_EtOH    γ_H2O")
    for x_eth in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        x_h2o = 1.0 - x_eth
        if x_eth < 0.001:
            x_eth = 0.001
            x_h2o = 0.999
        if x_h2o < 0.001:
            x_h2o = 0.001
            x_eth = 0.999
        
        gamma = model.activity_coefficients([ethanol, water], [x_eth, x_h2o], 298.15)
        print(f"  {x_eth:.1f}       {gamma[0]:.4f}    {gamma[1]:.4f}")
    
    # Test 2: Methanol-Water
    print("\n\nTest 2: Methanol-Water at 25°C")
    methanol = {'CH3OH': 1}
    
    print("\n  x_MeOH    γ_MeOH    γ_H2O")
    for x_meoh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        x_h2o = 1.0 - x_meoh
        gamma = model.activity_coefficients([methanol, water], [x_meoh, x_h2o], 298.15)
        print(f"  {x_meoh:.1f}       {gamma[0]:.4f}    {gamma[1]:.4f}")
    
    # Test 3: Known molecule lookup
    print("\n\nTest 3: Known molecule lookup")
    for name in ['ethanol', 'water', 'acetone', 'benzene']:
        groups = get_unifac_groups(name)
        print(f"  {name}: {groups}")
    
    print("\n=== Tests Complete ===")
