"""Gas-phase molecular dipole-moment resolution."""

from __future__ import annotations

import hashlib
import math
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

import numpy as np

from .common import PropertyResolutionError, PropertyResolutionResult


DIPOLE_CACHE_VERSION = 4
DIPOLE_CACHE_NAMESPACE = "dipole_derived_v1"
GFN2_XTB_QUALITY = 0.75
PBE0_PVDZ_QUALITY = 0.90
HEURISTIC_QUALITY = 0.40
AU_DIPOLE_TO_DEBYE = 2.541746473

# Dortmund subgroup IDs are used only to classify a molecule. Molecular
# dipoles are vectors, so subgroup values must not be added together.
_UNIFAC_DIPOLE_CLASSES = (
    ("sulfone", frozenset({110, 111}), 4.5),
    ("sulfoxide", frozenset({67}), 4.0),
    ("nitrile", frozenset({40, 41, 68}), 3.9),
    ("amide", frozenset({72, 73, 86, 87, 88, 89, 91, 92, 93, 94, 100, 101, 102, 103}), 3.7),
    ("nitro", frozenset({54, 55, 56, 57}), 3.5),
    ("aldehyde_or_ketone", frozenset({18, 19, 20, 61}), 2.7),
    ("pyridine", frozenset({37, 38, 39}), 2.2),
    ("epoxide", frozenset({107, 108, 109, 119, 153}), 1.9),
    ("ester_or_carbonate", frozenset({21, 22, 23, 77, 112, 113, 114}), 1.8),
    ("organic_halide", frozenset({44, 45, 46, 47, 48, 49, 50, 51, 53, 63, 64, 69, 71, 74, 75, 76}), 1.8),
    ("alcohol", frozenset({14, 15, 62, 81, 82}), 1.7),
    ("carboxylic_acid", frozenset({42, 43}), 1.6),
    ("aromatic_amine", frozenset({36}), 1.5),
    ("thiol_or_sulfide", frozenset({59, 60, 104, 105, 106, 122, 123, 124, 201}), 1.5),
    ("aliphatic_amine", frozenset({28, 29, 30, 31, 32, 33, 34, 35, 85}), 1.2),
    ("phenol", frozenset({17}), 1.2),
    ("ether", frozenset({24, 25, 26, 27, 83, 84}), 1.15),
)
_HYDROCARBON_UNIFAC_GROUPS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 65, 66, 70, 78, 79, 80})
_SMARTS_DIPOLE_CLASSES = (
    ("sulfone", "[SX4](=O)(=O)([#6])[#6]", 4.5),
    ("sulfoxide", "[SX3](=O)([#6])[#6]", 4.0),
    ("nitrile", "[#6]#[NX1]", 3.9),
    ("amide", "[NX3][CX3](=[OX1])", 3.7),
    ("nitro", "[NX3+](=[OX1])[OX1-]", 3.5),
    ("carboxylic_acid", "[CX3](=[OX1])[OX2H1]", 1.6),
    ("ester_or_carbonate", "[CX3](=[OX1])[OX2][#6]", 1.8),
    ("aldehyde_or_ketone", "[CX3H1](=[OX1])", 2.7),
    ("aldehyde_or_ketone", "[#6][CX3](=[OX1])[#6]", 2.7),
    ("pyridine", "[nH0]", 2.2),
    ("epoxide", "[OX2r3]1[#6r3][#6r3]1", 1.9),
    ("phenol", "[OX2H1]-[c]", 1.2),
    ("alcohol", "[OX2H1;!$([O]-[C,S,P]=O)]", 1.7),
    ("aromatic_amine", "[NX3;H1,H2]-[c]", 1.5),
    ("aliphatic_amine", "[NX3;H0,H1,H2;!$(N-[CX3]=O)]", 1.2),
    ("thiol_or_sulfide", "[SX2H1]", 1.5),
    ("thiol_or_sulfide", "[SX2]([#6])[#6]", 1.5),
    ("ether", "[OX2]([#6])[#6]", 1.15),
    ("organic_halide", "[#6]-[F,Cl,Br,I]", 1.8),
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


class DipoleMomentMixin:
    """Resolve permanent gas-phase dipole magnitudes in Debye."""

    def resolve_dipole_moment(
        self,
        identifier: str,
        props: Optional[dict[str, Any]] = None,
        *,
        use_pvdz: bool = False,
        allow_online: bool = True,
    ) -> PropertyResolutionResult:
        """Resolve a permanent molecular dipole magnitude in Debye.

        An explicit component-property override has priority, followed by
        CCCBDB experimental data bundled by :mod:`chemicals`. Missing
        experimental data is followed by GFN2-xTB, or by a PBE0/aug-cc-pVDZ
        single point on the xTB geometry when ``use_pvdz`` is true.
        Optional-backend absence or failure falls back to a deliberately
        low-quality functional-class estimate.
        """
        props = self._coerce_props(identifier, props, allow_online=allow_online)

        for key in ("dipole_moment", "dipole_D", "dipole"):
            provided = self._source_result_for_value(props, key, units="Debye")
            if provided is None:
                continue
            try:
                value = float(provided.value)
            except (TypeError, ValueError) as exc:
                raise PropertyResolutionError(
                    f"Invalid provided dipole moment for {identifier!r}"
                ) from exc
            if not math.isfinite(value) or value < 0.0:
                raise PropertyResolutionError(
                    f"Invalid provided dipole moment for {identifier!r}: "
                    "expected a finite nonnegative value in Debye"
                )
            notes = str(provided.notes or "")
            if "debye" not in notes.lower():
                notes = f"{notes}; units Debye" if notes else "units Debye"
            return PropertyResolutionResult(
                value=value,
                source=provided.source,
                method=provided.method,
                quality=provided.quality,
                notes=notes,
            )

        cas = self._dipole_cas(identifier, props)
        experimental = self._nist_dipole(identifier, props, cas=cas)
        if experimental is not None:
            return experimental

        smiles_result = self._resolve_smiles_result(
            identifier,
            props,
            allow_online=allow_online,
        )
        smiles = str(smiles_result.value).strip() if smiles_result and smiles_result.value else ""
        if smiles:
            try:
                self._validated_dipole_molecule(smiles)
            except ValueError as exc:
                raise PropertyResolutionError(
                    f"Cannot determine dipole moment for {identifier!r}: {exc}"
                ) from exc
        backend = "pvdz" if use_pvdz else "xtb"
        identity = f"smiles:{smiles}" if smiles else f"cas:{cas}" if cas else f"identifier:{identifier}"
        pvdz_payload = self._load_dipole_artifact(identity, "result_pvdz")
        xtb_payload = self._load_dipole_artifact(identity, "result_xtb")
        pvdz_result = self._dipole_result_from_artifact(pvdz_payload)
        xtb_result = self._dipole_result_from_artifact(xtb_payload)

        # Cached results are monotonic by quality. Once pVDZ exists it also
        # satisfies ordinary requests that would otherwise select xTB.
        if pvdz_result is not None:
            return pvdz_result
        if not use_pvdz and xtb_result is not None:
            return xtb_result

        dependencies = self._dipole_dependency_state()
        signature = self._backend_signature(backend, dependencies)
        attempts = getattr(self, "_dipole_attempt_cache", None)
        if attempts is None:
            attempts = {}
            self._dipole_attempt_cache = attempts
        attempt_key = (str(self.CACHE_DIR), identity, backend)
        previous_attempt = attempts.get(attempt_key) or {}
        backend_available = self._backend_is_available(backend, dependencies)
        should_attempt = (
            bool(smiles)
            and backend_available
            and previous_attempt.get("signature") != signature
        )
        failure_note = ""
        if should_attempt:
            try:
                geometry_record = self._load_dipole_artifact(identity, "geometry_xtb")
                geometry_payload = (
                    geometry_record.get("geometry")
                    if isinstance(geometry_record, dict)
                    else None
                )
                if isinstance(geometry_payload, dict):
                    try:
                        atoms, charge, multiplicity = self._atoms_from_payload(
                            geometry_payload
                        )
                    except (KeyError, TypeError, ValueError):
                        self._delete_dipole_artifact(identity, "geometry_xtb")
                        atoms, charge, multiplicity = self._xtb_optimized_geometry(smiles)
                        self._save_dipole_artifact(
                            identity,
                            "geometry_xtb",
                            {
                                "geometry": self._atoms_payload(
                                    atoms,
                                    charge,
                                    multiplicity,
                                ),
                                "dependencies": dependencies,
                            },
                        )
                else:
                    atoms, charge, multiplicity = self._xtb_optimized_geometry(smiles)
                    self._save_dipole_artifact(
                        identity,
                        "geometry_xtb",
                        {
                            "geometry": self._atoms_payload(atoms, charge, multiplicity),
                            "dependencies": dependencies,
                        },
                    )
                value = (
                    self._pvdz_dipole_at_geometry(atoms, charge, multiplicity)
                    if use_pvdz
                    else self._gfn2_xtb_dipole_at_geometry(atoms, charge, multiplicity)
                )
                value = float(value)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError("computed dipole was not finite and nonnegative")
                if use_pvdz:
                    result = PropertyResolutionResult(
                        value=value,
                        source="calculated",
                        method="pbe0_aug_cc_pvdz_dipole",
                        quality=PBE0_PVDZ_QUALITY,
                        notes=(
                            "Gas-phase PBE0/aug-cc-pVDZ dipole at a "
                            "GFN2-xTB-optimized geometry; units Debye"
                        ),
                    )
                else:
                    result = PropertyResolutionResult(
                        value=value,
                        source="calculated",
                        method="gfn2_xtb_dipole",
                        quality=GFN2_XTB_QUALITY,
                        notes="Gas-phase GFN2-xTB optimized-geometry dipole; units Debye",
                    )
                self._save_dipole_artifact(
                    identity,
                    f"result_{backend}",
                    {
                        "result": self._result_payload(result),
                        "dependencies": dependencies,
                    },
                )
                attempts.pop(attempt_key, None)
                return result
            except Exception as exc:
                failure_note = f"{backend} backend unavailable or failed ({type(exc).__name__}: {exc})"
                # Runtime and SCF failures may be transient, so suppress repeats
                # only for this resolver instance rather than persisting them.
                attempts[attempt_key] = {"signature": signature, "error": failure_note}
        elif not backend_available:
            failure_note = f"{backend} optional dependencies are not installed"
        elif previous_attempt.get("signature") == signature:
            failure_note = str(previous_attempt.get("error") or f"{backend} previously failed")
        elif not smiles:
            failure_note = "molecular structure unavailable for QM calculation"

        # A previously computed xTB value remains preferable to a heuristic
        # while a requested pVDZ upgrade is unavailable or failed.
        if xtb_result is not None:
            return PropertyResolutionResult(
                value=xtb_result.value,
                source=xtb_result.source,
                method=xtb_result.method,
                quality=xtb_result.quality,
                notes=f"{xtb_result.notes}; pVDZ upgrade pending; {failure_note}",
            )

        heuristic = self._heuristic_dipole(identifier, props, smiles)
        if heuristic is None:
            detail = f"; {failure_note}" if failure_note else ""
            raise PropertyResolutionError(
                f"Cannot determine dipole moment for {identifier!r}{detail}"
            )
        class_name, value, class_note = heuristic
        notes = f"Low-quality dominant-functional-class estimate ({class_note}); units Debye"
        if failure_note:
            notes += f"; {failure_note}"
        elif not smiles:
            notes += "; molecular structure unavailable"
        result = PropertyResolutionResult(
            value=value,
            source="estimated",
            method=f"{class_name}_dipole_heuristic",
            quality=HEURISTIC_QUALITY,
            notes=notes,
        )
        return result

    def _nist_dipole(
        self,
        identifier: str,
        props: dict[str, Any],
        *,
        cas: str = "",
    ) -> Optional[PropertyResolutionResult]:
        try:
            from chemicals.dipole import dipole_moment
        except ImportError:
            return None

        cas = cas or self._dipole_cas(identifier, props)
        if not cas:
            return None
        try:
            value = dipole_moment(cas, method="CCCBDB")
            value = float(value) if value is not None else None
        except (KeyError, TypeError, ValueError):
            return None
        if value is None or not math.isfinite(value) or value < 0.0:
            return None
        result = PropertyResolutionResult(
            value=value,
            source="NIST CCCBDB",
            method="cccbdb_experimental_dipole",
            quality=0.98,
            notes=f"Experimental gas-phase dipole for CAS {cas}; units Debye",
        )
        return result

    def _dipole_cas(self, identifier: str, props: dict[str, Any]) -> str:
        candidates = [props.get("CAS"), props.get("cas")]
        candidates.extend(self._identifier_candidates(identifier, props))
        for candidate in candidates:
            text = str(candidate or "").strip()
            if self._looks_like_dipole_cas(text):
                return text

        try:
            from chemicals.identifiers import int_to_CAS, search_chemical
        except ImportError:
            return ""
        for candidate in candidates:
            if not candidate:
                continue
            try:
                metadata = search_chemical(str(candidate))
                cas = int_to_CAS(int(metadata.CAS))
            except (LookupError, TypeError, ValueError):
                continue
            if self._looks_like_dipole_cas(cas):
                return cas
        return ""

    @staticmethod
    def _looks_like_dipole_cas(value: str) -> bool:
        parts = value.split("-")
        return (
            len(parts) == 3
            and 2 <= len(parts[0]) <= 7
            and len(parts[1]) == 2
            and len(parts[2]) == 1
            and all(part.isdigit() for part in parts)
        )

    @staticmethod
    def _result_payload(result: PropertyResolutionResult) -> dict[str, Any]:
        return {
            "value": float(result.value),
            "source": result.source,
            "method": result.method,
            "quality": float(result.quality),
            "notes": result.notes,
        }

    @staticmethod
    def _result_from_payload(payload: dict[str, Any]) -> PropertyResolutionResult:
        return PropertyResolutionResult(
            value=float(payload["value"]),
            source=str(payload["source"]),
            method=str(payload["method"]),
            quality=float(payload["quality"]),
            notes=str(payload.get("notes") or ""),
        )

    @classmethod
    def _dipole_result_from_artifact(
        cls,
        artifact: Optional[dict[str, Any]],
    ) -> Optional[PropertyResolutionResult]:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("result"), dict):
            return None
        try:
            result = cls._result_from_payload(artifact["result"])
            value = float(result.value)
            quality = float(result.quality)
        except (KeyError, TypeError, ValueError):
            return None
        if (
            not math.isfinite(value)
            or value < 0.0
            or not math.isfinite(quality)
            or not 0.0 <= quality <= 1.0
        ):
            return None
        return result

    @staticmethod
    def _dipole_dependency_state() -> dict[str, str]:
        return {
            name: _package_version(name)
            for name in ("tblite", "ase", "pyscf")
        }

    @staticmethod
    def _backend_signature(backend: str, dependencies: dict[str, str]) -> str:
        required = ("tblite", "ase", "pyscf") if backend == "pvdz" else ("tblite", "ase")
        return ";".join(f"{name}={dependencies[name]}" for name in required)

    @staticmethod
    def _backend_is_available(backend: str, dependencies: dict[str, str]) -> bool:
        required = ("tblite", "ase", "pyscf") if backend == "pvdz" else ("tblite", "ase")
        return all(dependencies.get(name) != "missing" for name in required)

    def _dipole_artifact_key(self, identity: str, artifact: str) -> str:
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"{artifact}_v{DIPOLE_CACHE_VERSION}_{digest}"

    def _dipole_cache(self):
        from .runtime_cache import SQLiteJSONCache

        path = self._runtime_json_cache().path
        state = getattr(self, "_dipole_cache_state", None)
        if state is not None and state[0] == path:
            return state[1]
        cache = SQLiteJSONCache(path, DIPOLE_CACHE_NAMESPACE)
        self._dipole_cache_state = path, cache
        return cache

    def _load_dipole_artifact(
        self,
        identity: str,
        artifact: str,
    ) -> Optional[dict[str, Any]]:
        key = self._dipole_artifact_key(identity, artifact)
        try:
            cached = self._dipole_cache().get(key)
        except Exception:
            return None
        if (
            not isinstance(cached, dict)
            or cached.get("version") != DIPOLE_CACHE_VERSION
            or cached.get("identity") != identity
            or cached.get("artifact") != artifact
        ):
            return None
        return cached

    def _save_dipole_artifact(
        self,
        identity: str,
        artifact: str,
        payload: dict[str, Any],
    ) -> None:
        key = self._dipole_artifact_key(identity, artifact)
        payload = {
            **payload,
            "version": DIPOLE_CACHE_VERSION,
            "identity": identity,
            "artifact": artifact,
        }
        try:
            self._dipole_cache().set(key, payload)
        except Exception:
            pass

    def _delete_dipole_artifact(self, identity: str, artifact: str) -> None:
        key = self._dipole_artifact_key(identity, artifact)
        try:
            self._dipole_cache().delete(key)
        except Exception:
            pass

    @classmethod
    def _calculate_gfn2_xtb_dipole(cls, smiles: str) -> float:
        atoms, charge, multiplicity = cls._xtb_optimized_geometry(smiles)
        return cls._gfn2_xtb_dipole_at_geometry(atoms, charge, multiplicity)

    @staticmethod
    def _gfn2_xtb_dipole_at_geometry(atoms, charge: int, multiplicity: int) -> float:
        from ase.units import Bohr
        from tblite.interface import Calculator

        calculator = Calculator(
            "GFN2-xTB",
            atoms.numbers,
            atoms.positions / Bohr,
            charge=charge,
            uhf=multiplicity - 1,
            color=False,
            logger=lambda _message: None,
        )
        result = calculator.singlepoint()
        return float(np.linalg.norm(result.get("dipole")) * AU_DIPOLE_TO_DEBYE)

    @classmethod
    def _calculate_pvdz_dipole(cls, smiles: str) -> float:
        atoms, charge, multiplicity = cls._xtb_optimized_geometry(smiles)
        return cls._pvdz_dipole_at_geometry(atoms, charge, multiplicity)

    @staticmethod
    def _pvdz_dipole_at_geometry(atoms, charge: int, multiplicity: int) -> float:
        from pyscf import dft, gto

        xyz = ";".join(
            f"{symbol} {x:.12f} {y:.12f} {z:.12f}"
            for symbol, (x, y, z) in zip(
                atoms.get_chemical_symbols(),
                atoms.positions,
                strict=True,
            )
        )
        molecule = gto.M(
            atom=xyz,
            unit="Angstrom",
            charge=charge,
            spin=multiplicity - 1,
            basis="aug-cc-pvdz",
            verbose=0,
        )
        mean_field = (
            dft.RKS(molecule, xc="pbe0")
            if multiplicity == 1
            else dft.UKS(molecule, xc="pbe0")
        ).density_fit()
        mean_field.grids.level = 3
        mean_field.conv_tol = 1.0e-10
        mean_field.max_cycle = 100
        mean_field.kernel()
        if not mean_field.converged:
            raise RuntimeError("PySCF PBE0 SCF did not converge")
        vector = mean_field.dip_moment(unit="Debye", verbose=0)
        return float(np.linalg.norm(vector))

    @staticmethod
    def _atoms_payload(atoms, charge: int, multiplicity: int) -> dict[str, Any]:
        return {
            "symbols": list(atoms.get_chemical_symbols()),
            "positions_angstrom": np.asarray(atoms.positions, dtype=float).tolist(),
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "method": "GFN2-xTB",
        }

    @staticmethod
    def _atoms_from_payload(payload: dict[str, Any]):
        from ase import Atoms

        symbols = list(payload["symbols"])
        positions = np.asarray(payload["positions_angstrom"], dtype=float)
        if positions.shape != (len(symbols), 3) or not np.all(np.isfinite(positions)):
            raise ValueError("cached xTB geometry is malformed")
        return (
            Atoms(symbols, positions=positions),
            int(payload["charge"]),
            int(payload["multiplicity"]),
        )

    @staticmethod
    def _xtb_optimized_geometry(smiles: str):
        from ase import Atoms
        from ase.optimize import BFGS
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from tblite.ase import TBLite

        molecule = DipoleMomentMixin._validated_dipole_molecule(smiles)
        charge = int(Chem.GetFormalCharge(molecule))
        unpaired = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
        multiplicity = int(unpaired) + 1
        molecule = Chem.AddHs(molecule)
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = int(hashlib.sha256(smiles.encode()).hexdigest()[:7], 16)
        parameters.pruneRmsThresh = 0.25
        conformer_ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=10, params=parameters))
        if not conformer_ids:
            raise RuntimeError("RDKit could not generate a conformer")

        energies = []
        if AllChem.MMFFHasAllMoleculeParams(molecule):
            properties = AllChem.MMFFGetMoleculeProperties(molecule)
            for conformer_id in conformer_ids:
                forcefield = AllChem.MMFFGetMoleculeForceField(
                    molecule,
                    properties,
                    confId=conformer_id,
                )
                if forcefield is None:
                    continue
                forcefield.Minimize(maxIts=500)
                energies.append((forcefield.CalcEnergy(), conformer_id))
        else:
            for conformer_id in conformer_ids:
                forcefield = AllChem.UFFGetMoleculeForceField(molecule, confId=conformer_id)
                if forcefield is None:
                    continue
                forcefield.Minimize(maxIts=500)
                energies.append((forcefield.CalcEnergy(), conformer_id))
        if not energies:
            raise RuntimeError("RDKit could not rank generated conformers")

        conformer = molecule.GetConformer(min(energies)[1])
        atoms = Atoms(
            [atom.GetSymbol() for atom in molecule.GetAtoms()],
            positions=np.asarray(conformer.GetPositions()),
        )
        atoms.calc = TBLite(
            method="GFN2-xTB",
            charge=charge,
            multiplicity=multiplicity,
            verbosity=0,
        )
        if not BFGS(atoms, logfile=None).run(fmax=0.01, steps=300):
            raise RuntimeError("GFN2-xTB geometry optimization did not converge")
        return atoms, charge, multiplicity

    @staticmethod
    def _validated_dipole_molecule(smiles: str):
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("resolved SMILES could not be parsed")
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError(
                "a permanent molecular dipole requires one connected structure"
            )
        if Chem.GetFormalCharge(molecule) != 0:
            raise ValueError(
                "a charged structure has no origin-independent permanent dipole"
            )
        return molecule

    @staticmethod
    def _heuristic_dipole(
        identifier: str,
        props: dict[str, Any],
        smiles: str,
    ) -> Optional[tuple[str, float, str]]:
        groups = None
        if smiles:
            try:
                if __package__ and __package__.split(".", 1)[0] == "pfdsim":
                    from ..unifac_fragmenter import fragment_safe
                else:
                    from unifac_fragmenter import fragment_safe
                groups = fragment_safe(smiles, variant="UNIFDMD")
            except Exception:
                groups = None
        if groups:
            group_ids = frozenset(int(group) for group in groups)
            if group_ids <= _HYDROCARBON_UNIFAC_GROUPS:
                return "hydrocarbon", 0.0, "UNIFAC hydrocarbon groups"
            if group_ids == {52}:
                return "symmetric_nonpolar", 0.0, "UNIFAC CCl4 whole-molecule group"
            if group_ids == {58}:
                return "symmetric_nonpolar", 0.0, "UNIFAC CS2 whole-molecule group"
            for class_name, class_groups, value in _UNIFAC_DIPOLE_CLASSES:
                matched = sorted(group_ids & class_groups)
                if matched:
                    return class_name, value, f"dominant UNIFAC subgroup(s) {matched}"

        if smiles:
            try:
                from rdkit import Chem

                molecule = Chem.MolFromSmiles(smiles)
            except Exception:
                molecule = None
            if molecule is not None:
                elements = {atom.GetSymbol() for atom in molecule.GetAtoms()}
                if "C" in elements and elements <= {"C", "H"}:
                    return "hydrocarbon", 0.0, "RDKit hydrocarbon structure"
                for class_name, smarts, value in _SMARTS_DIPOLE_CLASSES:
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern is not None and molecule.HasSubstructMatch(pattern):
                        return class_name, value, f"RDKit SMARTS matched {class_name}"
                if "C" in elements and elements - {"C", "H"}:
                    return (
                        "generic_polar_organic",
                        2.0,
                        "heteroatom-containing organic outside recognized classes",
                    )

        label = " ".join(
            str(value).lower()
            for value in (identifier, props.get("name"), props.get("formula"))
            if value
        )
        named_fallbacks = (
            (("nitrile", "cyanide"), "nitrile", 3.9),
            (("nitro",), "nitro", 3.5),
            (("sulfoxide",), "sulfoxide", 4.0),
            (("sulfone",), "sulfone", 4.5),
            (("amide",), "amide", 3.7),
            (("aldehyde", "ketone"), "aldehyde_or_ketone", 2.7),
            (("carboxylic acid",), "carboxylic_acid", 1.6),
            (("ester",), "ester", 1.8),
            (("alcohol",), "alcohol", 1.7),
            (("ether",), "ether", 1.15),
        )
        for tokens, class_name, value in named_fallbacks:
            if any(token in label for token in tokens):
                return class_name, value, f"identity label matched {class_name}"
        formula = str(props.get("formula") or "")
        if formula and "C" in formula and not any(element in formula for element in ("N", "O", "F", "Cl", "Br", "I", "S", "P", "Si")):
            return "hydrocarbon", 0.0, "hydrocarbon molecular formula"
        return None


__all__ = ["DipoleMomentMixin"]
