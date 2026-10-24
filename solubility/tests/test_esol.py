"""Check the pinned frozen equations and descriptor conventions."""

from pathlib import Path
import sys

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_esol import REFERENCE, reference_calculator


def test_reference_descriptors_use_average_mass_and_aromatic_atom_fraction():
    calculator = reference_calculator(REFERENCE / 'esol.py')
    ethanol = calculator.calc_esol_descriptors(Chem.MolFromSmiles('CCO'))
    assert np.isclose(ethanol.mw, 46.069)
    assert np.isclose(ethanol.logp, -.0014)
    assert ethanol.rotors == 0 and ethanol.ap == 0
    toluene = calculator.calc_esol_descriptors(Chem.MolFromSmiles('Cc1ccccc1'))
    assert np.isclose(toluene.ap, 6 / 7)


def test_both_frozen_equations_match_independent_ethanol_calculation():
    calculator = reference_calculator(REFERENCE / 'esol.py')
    molecule = Chem.MolFromSmiles('CCO')
    expected_rdkit = .26121066137801696 - .0066138847738667125 * 46.069 + .7416739523408995 * .0014
    expected_original = .16 - .0062 * 46.069 + .63 * .0014
    assert np.isclose(calculator.calc_esol(molecule), expected_rdkit, atol=1e-12)
    assert np.isclose(calculator.calc_esol_orig(molecule), expected_original, atol=1e-12)
