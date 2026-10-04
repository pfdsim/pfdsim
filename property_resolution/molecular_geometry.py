"""Deterministic force-field starting geometry shared by molecular properties."""

import hashlib


def lowest_energy_forcefield_conformer(smiles, molecule=None):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = molecule if molecule is not None else Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid molecular structure for geometry estimation.")
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(hashlib.sha256(smiles.encode()).hexdigest()[:7], 16)
    parameters.pruneRmsThresh = 0.25
    conformers = list(
        AllChem.EmbedMultipleConfs(molecule, numConfs=10, params=parameters)
    )
    if not conformers:
        raise RuntimeError("RDKit could not generate a conformer")
    energies = []
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        properties = AllChem.MMFFGetMoleculeProperties(molecule)
        for identifier in conformers:
            field = AllChem.MMFFGetMoleculeForceField(
                molecule, properties, confId=identifier
            )
            if field is not None:
                field.Minimize(maxIts=500)
                energies.append((field.CalcEnergy(), identifier))
    else:
        for identifier in conformers:
            field = AllChem.UFFGetMoleculeForceField(molecule, confId=identifier)
            if field is not None:
                field.Minimize(maxIts=500)
                energies.append((field.CalcEnergy(), identifier))
    if not energies:
        raise RuntimeError("RDKit could not rank generated conformers")
    return molecule, molecule.GetConformer(min(energies)[1])
