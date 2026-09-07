import math
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

from property_resolution.dipole_moment import DipoleMomentMixin
from property_resolver import (
    PropertyResolutionError,
    PropertyResolver,
    resolve_dipole_moment,
)


class DipoleResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.resolver = PropertyResolver()
        self.resolver.CACHE_DIR = Path(self.temporary_directory.name)

    def test_resolver_construction_does_not_eagerly_resolve_dipole(self):
        with patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            side_effect=AssertionError("dipole backend must remain lazy"),
        ), patch.object(
            DipoleMomentMixin,
            "_nist_dipole",
            side_effect=AssertionError("dipole lookup must remain lazy"),
        ):
            PropertyResolver()

    def test_cccbdb_experiment_precedes_computation(self):
        props = {"CAS": "64-17-5", "name": "ethanol", "smiles": "CCO"}
        with patch.object(
            DipoleMomentMixin,
            "_calculate_gfn2_xtb_dipole",
            side_effect=AssertionError("computation should not run"),
        ):
            result = self.resolver.resolve_dipole_moment(
                "ethanol",
                props,
                allow_online=False,
            )

        self.assertTrue(math.isclose(result.value, 1.44))
        self.assertEqual(result.source, "NIST CCCBDB")
        self.assertEqual(result.method, "cccbdb_experimental_dipole")
        self.assertEqual(result.quality, 0.98)

    def test_provided_dipole_precedes_experimental_and_computed_sources(self):
        props = {
            "CAS": "64-17-5",
            "name": "ethanol",
            "smiles": "CCO",
            "dipole_moment": 9.25,
            "property_sources": {
                "dipole_moment": {
                    "source": "provided",
                    "method": "pfd_component_override",
                    "quality": 1.0,
                    "notes": "User-specified in .pfd component definition",
                }
            },
        }
        with patch.object(
            DipoleMomentMixin,
            "_nist_dipole",
            side_effect=AssertionError("PFD override must take precedence"),
        ), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            side_effect=AssertionError("PFD override must avoid computation"),
        ):
            result = self.resolver.resolve_dipole_moment(
                "ethanol",
                props,
                allow_online=False,
            )

        self.assertEqual(result.value, 9.25)
        self.assertEqual(result.source, "provided")
        self.assertEqual(result.method, "pfd_component_override")
        self.assertEqual(result.quality, 1.0)
        self.assertIn("Debye", result.notes)

    def test_default_computational_backend_is_gfn2_xtb(self):
        props = {"CAS": "999-99-9", "name": "test ether", "smiles": "CCOCC"}
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            return_value=(object(), 0, 1),
        ), patch.object(
            DipoleMomentMixin,
            "_atoms_payload",
            return_value={"geometry": "test"},
        ), patch.object(
            DipoleMomentMixin,
            "_gfn2_xtb_dipole_at_geometry",
            return_value=1.234,
        ) as compute:
            result = self.resolver.resolve_dipole_moment(
                "test ether",
                props,
                allow_online=False,
            )

        compute.assert_called_once_with(ANY, 0, 1)
        self.assertEqual(result.value, 1.234)
        self.assertEqual(result.method, "gfn2_xtb_dipole")
        self.assertEqual(result.quality, 0.75)

    def test_use_pvdz_selects_pbe0_backend(self):
        props = {"CAS": "999-99-9", "name": "test ketone", "smiles": "CCC(=O)C"}
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            return_value=(object(), 0, 1),
        ), patch.object(
            DipoleMomentMixin,
            "_atoms_payload",
            return_value={"geometry": "test"},
        ), patch.object(
            DipoleMomentMixin,
            "_pvdz_dipole_at_geometry",
            return_value=2.71,
        ) as compute:
            result = self.resolver.resolve_dipole_moment(
                "test ketone",
                props,
                use_pvdz=True,
                allow_online=False,
            )

        compute.assert_called_once_with(ANY, 0, 1)
        self.assertEqual(result.value, 2.71)
        self.assertEqual(result.method, "pbe0_aug_cc_pvdz_dipole")
        self.assertEqual(result.quality, 0.90)

    def test_backend_failure_falls_back_to_unifac_heuristic(self):
        cases = (
            ("hydrocarbon", "CCCCC", 0.0, "hydrocarbon_dipole_heuristic"),
            ("ether", "CCOCC", 1.15, "ether_dipole_heuristic"),
            ("alcohol", "CCCCO", 1.7, "alcohol_dipole_heuristic"),
            ("ketone", "CCC(=O)C", 2.7, "aldehyde_or_ketone_dipole_heuristic"),
            ("acid", "CCC(=O)O", 1.6, "carboxylic_acid_dipole_heuristic"),
            ("ester", "CCOC(=O)C", 1.8, "ester_or_carbonate_dipole_heuristic"),
            ("nitrile", "CCCC#N", 3.9, "nitrile_dipole_heuristic"),
            ("nitro", "CCC[N+](=O)[O-]", 3.5, "nitro_dipole_heuristic"),
            ("sulfoxide", "CCS(=O)CC", 4.0, "sulfoxide_dipole_heuristic"),
        )
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            side_effect=ImportError("optional backend absent"),
        ):
            for name, smiles, expected, method in cases:
                with self.subTest(name=name):
                    result = self.resolver.resolve_dipole_moment(
                        name,
                        {"CAS": "999-99-9", "name": name, "smiles": smiles},
                        allow_online=False,
                    )
                    self.assertTrue(math.isclose(result.value, expected))
                    self.assertEqual(result.method, method)
                    self.assertEqual(result.source, "estimated")
                    self.assertEqual(result.quality, 0.40)
                    self.assertIn("optional backend absent", result.notes)

    def test_cached_heuristic_upgrades_when_dependencies_appear(self):
        props = {"CAS": "999-99-9", "name": "test ether", "smiles": "CCOCC"}
        missing = {"tblite": "missing", "ase": "missing", "pyscf": "missing"}
        installed = {"tblite": "0.7.0", "ase": "3.29.0", "pyscf": "missing"}
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_dipole_dependency_state",
            return_value=missing,
        ):
            fallback = self.resolver.resolve_dipole_moment(
                "test ether",
                props,
                allow_online=False,
            )

        self.assertEqual(fallback.method, "ether_dipole_heuristic")
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_dipole_dependency_state",
            return_value=installed,
        ), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            return_value=(object(), 0, 1),
        ), patch.object(
            DipoleMomentMixin,
            "_atoms_payload",
            return_value={"geometry": "test"},
        ), patch.object(
            DipoleMomentMixin,
            "_gfn2_xtb_dipole_at_geometry",
            return_value=1.31,
        ) as calculate:
            upgraded = self.resolver.resolve_dipole_moment(
                "test ether",
                props,
                allow_online=False,
            )

        calculate.assert_called_once_with(ANY, 0, 1)
        self.assertEqual(upgraded.method, "gfn2_xtb_dipole")
        self.assertEqual(upgraded.value, 1.31)

    def test_heuristic_is_recomputed_while_dependencies_remain_absent(self):
        props = {"CAS": "999-99-9", "name": "test ether", "smiles": "CCOCC"}
        missing = {"tblite": "missing", "ase": "missing", "pyscf": "missing"}
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_dipole_dependency_state",
            return_value=missing,
        ):
            first = self.resolver.resolve_dipole_moment(
                "test ether",
                props,
                allow_online=False,
            )
            with patch.object(
                DipoleMomentMixin,
                "_heuristic_dipole",
                wraps=DipoleMomentMixin._heuristic_dipole,
            ) as heuristic:
                second = self.resolver.resolve_dipole_moment(
                    "test ether",
                    props,
                    allow_online=False,
                )

        heuristic.assert_called_once_with("test ether", props, "CCOCC")
        self.assertEqual(first.value, 1.15)
        self.assertEqual(second.value, 1.15)

    def test_backend_failure_is_not_retried_at_same_dependency_versions(self):
        props = {"CAS": "999-99-9", "name": "test ether", "smiles": "CCOCC"}
        installed = {"tblite": "0.7.0", "ase": "3.29.0", "pyscf": "missing"}
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_dipole_dependency_state",
            return_value=installed,
        ), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            side_effect=RuntimeError("SCF failed"),
        ) as calculate:
            first = self.resolver.resolve_dipole_moment(
                "test ether",
                props,
                allow_online=False,
            )
            second = self.resolver.resolve_dipole_moment(
                "test ether",
                props,
                allow_online=False,
            )

        calculate.assert_called_once_with("CCOCC")
        self.assertEqual(first.method, "ether_dipole_heuristic")
        self.assertEqual(second.method, "ether_dipole_heuristic")

    def test_cached_xtb_is_retained_while_pvdz_upgrade_is_unavailable(self):
        props = {"CAS": "999-99-9", "name": "test ether", "smiles": "CCOCC"}
        installed = {"tblite": "0.7.0", "ase": "3.29.0", "pyscf": "missing"}
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_dipole_dependency_state",
            return_value=installed,
        ), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            return_value=(object(), 0, 1),
        ), patch.object(
            DipoleMomentMixin,
            "_atoms_payload",
            return_value={"geometry": "test"},
        ), patch.object(
            DipoleMomentMixin,
            "_gfn2_xtb_dipole_at_geometry",
            return_value=1.31,
        ):
            first = self.resolver.resolve_dipole_moment(
                "test ether",
                props,
                allow_online=False,
            )
            with patch.object(
                DipoleMomentMixin,
                "_pvdz_dipole_at_geometry",
                side_effect=AssertionError("unavailable pVDZ must not run"),
            ):
                second = self.resolver.resolve_dipole_moment(
                    "test ether",
                    props,
                    use_pvdz=True,
                    allow_online=False,
                )

        self.assertEqual(first.method, "gfn2_xtb_dipole")
        self.assertEqual(second.method, "gfn2_xtb_dipole")
        self.assertIn("pVDZ upgrade pending", second.notes)
        xtb = self.resolver._load_dipole_artifact("smiles:CCOCC", "result_xtb")
        self.assertEqual(xtb["dependencies"], installed)
        self.assertEqual(xtb["result"]["value"], 1.31)
        self.assertIsNone(
            self.resolver._load_dipole_artifact("smiles:CCOCC", "result_pvdz")
        )

    def test_cached_pvdz_result_overrides_later_default_xtb_request(self):
        props = {"CAS": "999-99-9", "name": "test ketone", "smiles": "CCC(=O)C"}
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            return_value=(object(), 0, 1),
        ), patch.object(
            DipoleMomentMixin,
            "_atoms_payload",
            return_value={"geometry": "test"},
        ), patch.object(
            DipoleMomentMixin,
            "_pvdz_dipole_at_geometry",
            return_value=2.71,
        ):
            high_quality = self.resolver.resolve_dipole_moment(
                "test ketone",
                props,
                use_pvdz=True,
                allow_online=False,
            )

        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_gfn2_xtb_dipole_at_geometry",
            side_effect=AssertionError("lower-quality backend should not run"),
        ):
            ordinary = self.resolver.resolve_dipole_moment(
                "test ketone",
                props,
                allow_online=False,
            )

        self.assertEqual(high_quality.method, "pbe0_aug_cc_pvdz_dipole")
        self.assertEqual(ordinary.method, "pbe0_aug_cc_pvdz_dipole")
        self.assertEqual(ordinary.value, 2.71)

    def test_pvdz_upgrade_reuses_cached_xtb_geometry(self):
        props = {"CAS": "999-99-9", "name": "test ether", "smiles": "CCOCC"}
        installed = {"tblite": "0.7.0", "ase": "3.29.0", "pyscf": "2.14.0"}
        first_atoms = object()
        second_atoms = object()
        geometry_payload = {"geometry": "serialized"}
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_dipole_dependency_state",
            return_value=installed,
        ), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            return_value=(first_atoms, 0, 1),
        ) as optimize, patch.object(
            DipoleMomentMixin,
            "_atoms_payload",
            return_value=geometry_payload,
        ), patch.object(
            DipoleMomentMixin,
            "_gfn2_xtb_dipole_at_geometry",
            return_value=1.31,
        ):
            self.resolver.resolve_dipole_moment(
                "test ether",
                props,
                allow_online=False,
            )

        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_dipole_dependency_state",
            return_value=installed,
        ), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            side_effect=AssertionError("cached xTB geometry should be reused"),
        ), patch.object(
            DipoleMomentMixin,
            "_atoms_from_payload",
            return_value=(second_atoms, 0, 1),
        ) as deserialize, patch.object(
            DipoleMomentMixin,
            "_pvdz_dipole_at_geometry",
            return_value=1.20,
        ) as pvdz:
            upgraded = self.resolver.resolve_dipole_moment(
                "test ether",
                props,
                use_pvdz=True,
                allow_online=False,
            )

        optimize.assert_called_once_with("CCOCC")
        deserialize.assert_called_once_with(geometry_payload)
        pvdz.assert_called_once_with(second_atoms, 0, 1)
        self.assertEqual(upgraded.method, "pbe0_aug_cc_pvdz_dipole")
        self.assertEqual(upgraded.value, 1.20)
        self.assertIsNotNone(
            self.resolver._load_dipole_artifact("smiles:CCOCC", "result_xtb")
        )
        self.assertIsNotNone(
            self.resolver._load_dipole_artifact("smiles:CCOCC", "result_pvdz")
        )

    def test_malformed_cached_geometry_is_replaced(self):
        props = {"CAS": "999-99-9", "name": "test ether", "smiles": "CCOCC"}
        installed = {"tblite": "0.7.0", "ase": "3.29.0", "pyscf": "missing"}
        self.resolver._save_dipole_artifact(
            "smiles:CCOCC",
            "geometry_xtb",
            {"geometry": {"symbols": ["C"]}},
        )

        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_dipole_dependency_state",
            return_value=installed,
        ), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            return_value=(object(), 0, 1),
        ) as optimize, patch.object(
            DipoleMomentMixin,
            "_atoms_payload",
            return_value={"geometry": "replacement"},
        ), patch.object(
            DipoleMomentMixin,
            "_gfn2_xtb_dipole_at_geometry",
            return_value=1.31,
        ):
            result = self.resolver.resolve_dipole_moment(
                "test ether",
                props,
                allow_online=False,
            )

        optimize.assert_called_once_with("CCOCC")
        self.assertEqual(result.value, 1.31)
        geometry = self.resolver._load_dipole_artifact(
            "smiles:CCOCC",
            "geometry_xtb",
        )
        self.assertEqual(geometry["geometry"], {"geometry": "replacement"})

    def test_nist_hit_uses_local_dataset_without_derived_cache(self):
        props = {"CAS": "64-17-5", "name": "ethanol", "smiles": "CCO"}
        first = self.resolver.resolve_dipole_moment(
            "ethanol",
            props,
            allow_online=False,
        )
        second_resolver = PropertyResolver()
        second_resolver.CACHE_DIR = Path(self.temporary_directory.name)
        from chemicals.dipole import dipole_moment

        with patch(
            "chemicals.dipole.dipole_moment",
            wraps=dipole_moment,
        ) as lookup:
            second = second_resolver.resolve_dipole_moment(
                "ethanol",
                props,
                allow_online=False,
            )

        lookup.assert_called_once_with("64-17-5", method="CCCBDB")
        self.assertEqual(first.value, 1.44)
        self.assertEqual(second.value, 1.44)
        self.assertEqual(second.method, "cccbdb_experimental_dipole")
        self.assertEqual(list(second_resolver._dipole_cache().items()), [])

    def test_computed_result_is_persistent_across_resolvers(self):
        props = {"CAS": "999-99-9", "name": "test ether", "smiles": "CCOCC"}
        first_resolver = self.resolver
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            return_value=(object(), 0, 1),
        ), patch.object(
            DipoleMomentMixin,
            "_atoms_payload",
            return_value={"geometry": "test"},
        ), patch.object(
            DipoleMomentMixin,
            "_gfn2_xtb_dipole_at_geometry",
            return_value=1.31,
        ):
            first = first_resolver.resolve_dipole_moment(
                "test ether",
                props,
                allow_online=False,
            )

        second_resolver = PropertyResolver()
        second_resolver.CACHE_DIR = Path(self.temporary_directory.name)
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_gfn2_xtb_dipole_at_geometry",
            side_effect=AssertionError("persistent xTB result should be reused"),
        ):
            second = second_resolver.resolve_dipole_moment(
                "test ether",
                props,
                allow_online=False,
            )

        self.assertEqual(first.value, 1.31)
        self.assertEqual(second.value, 1.31)
        self.assertEqual(second.method, "gfn2_xtb_dipole")

    def test_charged_and_disconnected_structures_are_rejected(self):
        cases = (
            ("charged", "C[N+](C)(C)C", "origin-independent"),
            ("disconnected", "CCO.O", "one connected structure"),
        )
        with patch.object(DipoleMomentMixin, "_nist_dipole", return_value=None), patch.object(
            DipoleMomentMixin,
            "_xtb_optimized_geometry",
            side_effect=AssertionError("invalid structures must not reach xTB"),
        ):
            for name, smiles, message in cases:
                with self.subTest(name=name), self.assertRaisesRegex(
                    PropertyResolutionError,
                    message,
                ):
                    self.resolver.resolve_dipole_moment(
                        name,
                        {"CAS": "999-99-9", "name": name, "smiles": smiles},
                        allow_online=False,
                    )

    def test_facade_forwards_pvdz_flag(self):
        with patch.object(PropertyResolver, "resolve_dipole_moment", return_value="sentinel") as method:
            result = resolve_dipole_moment(
                "compound",
                {"smiles": "CCO"},
                use_pvdz=True,
                allow_online=False,
            )

        self.assertEqual(result, "sentinel")
        method.assert_called_once_with(
            "compound",
            {"smiles": "CCO"},
            use_pvdz=True,
            allow_online=False,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("tblite") is not None
        and importlib.util.find_spec("ase") is not None
        and importlib.util.find_spec("pyscf") is not None,
        "optional benchmark dependencies are not installed",
    )
    def test_benchmark_metrics_allow_no_successful_predictions(self):
        from scripts.dipole.benchmark_dipole_backends import metrics

        result = metrics(
            [{"experimental_D": 1.0, "prediction": ""}],
            "prediction",
        )

        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["mae_D"])
        self.assertIsNone(result["r2"])

    @unittest.skipUnless(
        importlib.util.find_spec("tblite") is not None
        and importlib.util.find_spec("ase") is not None,
        "optional tblite and ASE dependencies are not installed",
    )
    def test_real_gfn2_xtb_backend_pins_methanol_dipole(self):
        value = self.resolver._calculate_gfn2_xtb_dipole("CO")

        self.assertAlmostEqual(value, 1.91575, delta=5.0e-4)

    @unittest.skipUnless(
        importlib.util.find_spec("tblite") is not None
        and importlib.util.find_spec("ase") is not None
        and importlib.util.find_spec("pyscf") is not None,
        "optional tblite, ASE, and PySCF dependencies are not installed",
    )
    def test_real_pvdz_backend_pins_methanol_dipole(self):
        value = self.resolver._calculate_pvdz_dipole("CO")

        self.assertAlmostEqual(value, 1.61889, delta=1.0e-3)


if __name__ == "__main__":
    unittest.main()
