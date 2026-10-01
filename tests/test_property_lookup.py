import json
import math
import os
import sqlite3
import sys
import tempfile
import types
import urllib.error
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator import Simulator
from pfd_parser import parse_pfd
from thermodynamics import create_thermodynamics
from chemical_properties import OnlinePropertyFetcher
from chemical_properties import ChemicalProperties
from chemical_properties import ChemicalDatabase
from property_resolver import PropertyResolver
from textbook_properties import TextbookPropertyLibrary
from antoine_properties import get_antoine_table
from vapor_pressure_tables import get_vapor_pressure_table_library


def component_moles(streams):
    totals = {}
    for stream in streams:
        for comp, frac in stream.composition.items():
            totals[comp] = totals.get(comp, 0.0) + stream.F * frac
    return totals


def relative_component_balance(inlets, outlets):
    inlet_totals = component_moles(inlets)
    outlet_totals = component_moles(outlets)
    keys = set(inlet_totals) | set(outlet_totals)
    denom = sum(abs(value) for value in inlet_totals.values()) or 1.0
    return (
        sum(abs(outlet_totals.get(k, 0.0) - inlet_totals.get(k, 0.0)) for k in keys)
        / denom
    )


class PropertyLookupCacheTests(unittest.TestCase):
    def test_smiles_cache_schema_version_invalidates_legacy_rows(self):
        cache_path = Path(tempfile.mkdtemp()) / "smiles.sqlite"
        with sqlite3.connect(cache_path) as connection:
            connection.execute(
                """
                CREATE TABLE smiles_cache (
                    identifier TEXT PRIMARY KEY,
                    smiles TEXT NOT NULL,
                    source TEXT NOT NULL,
                    quality REAL NOT NULL,
                    notes TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                """
                INSERT INTO smiles_cache
                    (identifier, smiles, source, quality, notes)
                VALUES ('x', 'CO', 'chemicals', 0.99, 'legacy poisoned row')
                """
            )

        db = ChemicalDatabase(enable_online=False)
        db._smiles_cache_path = cache_path
        db._ensure_smiles_cache()

        with sqlite3.connect(cache_path) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            count = connection.execute(
                "SELECT count(*) FROM smiles_cache WHERE identifier = 'x'"
            ).fetchone()[0]
        self.assertEqual(version, db._SMILES_CACHE_SCHEMA_VERSION)
        self.assertEqual(count, 0)

    def test_valid_formula_is_never_reinterpreted_as_local_smiles_alias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChemicalDatabase(enable_online=False)
            db._smiles_cache_path = Path(tmpdir) / "smiles_cache.sqlite"

            carbon_monoxide = db.get("CO", fetch_online=False)
            resolved = db.resolve_smiles_info(
                "CO",
                fetch_online=False,
                props=carbon_monoxide,
            )

            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.smiles, "[C-]#[O+]")
            with sqlite3.connect(db._smiles_cache_path) as connection:
                rows = connection.execute(
                    "SELECT identifier, smiles FROM smiles_cache"
                ).fetchall()
            self.assertTrue(rows)
            self.assertEqual({smiles for _identifier, smiles in rows}, {"[C-]#[O+]"})

    def test_provided_smiles_override_is_not_persisted(self):
        db = ChemicalDatabase(enable_online=False)
        db._smiles_cache_path = Path(tempfile.mkdtemp()) / "smiles.sqlite"
        props = {
            "name": "temporary PFD identity",
            "symbol": "XPFD",
            "smiles": "CCO",
            "property_sources": {
                "smiles": {
                    "source": "pfd",
                    "method": "pfd_component_override",
                    "quality": 1.0,
                },
            },
        }
        result = db.resolve_smiles_info(
            "temporary PFD identity",
            fetch_online=False,
            props=props,
        )
        self.assertEqual(result.smiles, "CCO")
        with sqlite3.connect(db._smiles_cache_path) as connection:
            count = connection.execute("SELECT count(*) FROM smiles_cache").fetchone()[
                0
            ]
        self.assertEqual(count, 0)

    def test_smiles_resolution_prefers_chemicals_before_opsin_and_pubchem(self):
        db = ChemicalDatabase(enable_online=True)
        db._smiles_cache_path = Path(tempfile.mkdtemp()) / "smiles.sqlite"

        chemicals_result = {
            "smiles": "CCO",
            "source": "chemicals",
            "method": "chemicals_identifier_smiles",
            "quality": 0.99,
        }

        with (
            patch.object(db, "_smiles_from_local_database", return_value=None),
            patch.object(db, "_smiles_from_chemicals_metadata") as chemicals,
            patch.object(db, "_smiles_from_opsin") as opsin,
            patch.object(db, "_smiles_from_pubchem") as pubchem,
        ):
            from chemical_properties import SmilesResolution

            chemicals.return_value = SmilesResolution(**chemicals_result)
            result = db.resolve_smiles_info("ethanol", fetch_online=True)

        self.assertEqual(result.smiles, "CCO")
        self.assertEqual(result.method, "chemicals_identifier_smiles")
        opsin.assert_not_called()
        pubchem.assert_not_called()

    def test_smiles_resolution_lowers_opsin_quality_for_ambiguity_warning(self):
        db = ChemicalDatabase(enable_online=False)
        db._smiles_cache_path = Path(tempfile.mkdtemp()) / "smiles.sqlite"

        def fake_py2opsin(*args, **kwargs):
            warnings.warn(
                "OPSIN parse ambiguity: chose first interpretation", RuntimeWarning
            )
            return "CCO"

        fake_module = types.SimpleNamespace(py2opsin=fake_py2opsin)
        with (
            patch.dict(sys.modules, {"py2opsin": fake_module}),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            result = db._smiles_from_opsin("ethanol")

        self.assertEqual(result.smiles, "CCO")
        self.assertEqual(result.quality, 0.90)
        self.assertTrue(any("ambiguity" in str(item.message) for item in caught))

    def test_smiles_resolution_falls_through_to_pubchem_when_opsin_fails(self):
        db = ChemicalDatabase(enable_online=True)
        db._smiles_cache_path = Path(tempfile.mkdtemp()) / "smiles.sqlite"

        def failing_py2opsin(*args, **kwargs):
            raise RuntimeError("java unavailable")

        fake_module = types.SimpleNamespace(py2opsin=failing_py2opsin)
        with (
            patch.object(db, "_smiles_from_local_database", return_value=None),
            patch.object(db, "_smiles_from_chemicals_metadata", return_value=None),
            patch.dict(sys.modules, {"py2opsin": fake_module}),
            patch.object(
                db,
                "_smiles_from_pubchem",
            ) as pubchem,
        ):
            from chemical_properties import SmilesResolution

            pubchem.return_value = SmilesResolution(
                smiles="CCN",
                source="pubchem",
                method="pubchem_structure",
                quality=0.97,
            )
            result = db.resolve_smiles_info("ethylamine", fetch_online=True)

        self.assertEqual(result.smiles, "CCN")
        self.assertEqual(result.quality, 0.97)
        pubchem.assert_called()

    def test_opsin_negative_cache_persists_only_deterministic_failures(self):
        cache_path = Path(tempfile.mkdtemp()) / "smiles.sqlite"
        calls = []

        def unparsed_py2opsin(*args, **kwargs):
            calls.append(args[0])
            return ""

        fake_module = types.SimpleNamespace(py2opsin=unparsed_py2opsin)
        with patch.dict(sys.modules, {"py2opsin": fake_module}):
            first = ChemicalDatabase(enable_online=False)
            first._smiles_cache_path = cache_path
            self.assertIsNone(first._smiles_from_opsin("unparseable systematic name"))

            second = ChemicalDatabase(enable_online=False)
            second._smiles_cache_path = cache_path
            self.assertIsNone(second._smiles_from_opsin("unparseable systematic name"))

        self.assertEqual(calls, ["unparseable systematic name"])
        with sqlite3.connect(cache_path) as connection:
            row = connection.execute(
                """
                SELECT identifier, opsin_version
                FROM opsin_negative_cache
                WHERE identifier = ?
                """,
                ("unparseable systematic name",),
            ).fetchone()
        self.assertEqual(
            row,
            (
                "unparseable systematic name",
                first._opsin_runtime_version(),
            ),
        )

        with (
            patch.object(
                ChemicalDatabase,
                "_opsin_runtime_version",
                return_value="future-opsin-version",
            ),
            patch.dict(sys.modules, {"py2opsin": fake_module}),
        ):
            upgraded = ChemicalDatabase(enable_online=False)
            upgraded._smiles_cache_path = cache_path
            self.assertIsNone(
                upgraded._smiles_from_opsin("unparseable systematic name")
            )
        self.assertEqual(
            calls,
            ["unparseable systematic name", "unparseable systematic name"],
        )

        transient_cache = Path(tempfile.mkdtemp()) / "smiles.sqlite"
        transient_calls = []

        def failing_py2opsin(*args, **kwargs):
            transient_calls.append(args[0])
            raise RuntimeError("java unavailable")

        failing_module = types.SimpleNamespace(py2opsin=failing_py2opsin)
        with patch.dict(sys.modules, {"py2opsin": failing_module}):
            transient_first = ChemicalDatabase(enable_online=False)
            transient_first._smiles_cache_path = transient_cache
            self.assertIsNone(transient_first._smiles_from_opsin("transient name"))
            self.assertIsNone(transient_first._smiles_from_opsin("transient name"))

            transient_second = ChemicalDatabase(enable_online=False)
            transient_second._smiles_cache_path = transient_cache
            self.assertIsNone(transient_second._smiles_from_opsin("transient name"))

        self.assertEqual(transient_calls, ["transient name", "transient name"])
        with sqlite3.connect(transient_cache) as connection:
            count = connection.execute(
                """
                SELECT count(*) FROM opsin_negative_cache
                WHERE identifier = ?
                """,
                ("transient name",),
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_smiles_resolution_reads_sqlite_cache_before_providers(self):
        db = ChemicalDatabase(enable_online=True)
        db._smiles_cache_path = Path(tempfile.mkdtemp()) / "smiles.sqlite"
        from chemical_properties import SmilesResolution

        db._cache_smiles(
            SmilesResolution(
                smiles="CCO",
                source="opsin",
                method="py2opsin",
                quality=0.97,
                notes="cached test row",
                identifier="cached ethanol",
            ),
            ["cached ethanol"],
        )

        with (
            patch.object(db, "_smiles_from_local_database") as local,
            patch.object(db, "_smiles_from_chemicals_metadata") as chemicals,
            patch.object(db, "_smiles_from_opsin") as opsin,
            patch.object(db, "_smiles_from_pubchem") as pubchem,
        ):
            result = db.resolve_smiles_info("cached ethanol", fetch_online=True)

        self.assertEqual(result.smiles, "CCO")
        self.assertEqual(result.source, "opsin")
        self.assertEqual(result.notes, "cached test row")
        local.assert_not_called()
        chemicals.assert_not_called()
        opsin.assert_not_called()
        pubchem.assert_not_called()

        with sqlite3.connect(db._smiles_cache_path) as conn:
            columns = [
                row[1]
                for row in conn.execute("PRAGMA table_info(smiles_cache)").fetchall()
            ]
        self.assertEqual(
            columns, ["identifier", "smiles", "source", "quality", "notes"]
        )

    def test_refrigerant_alias_precedes_conflicting_chemical_synonym(self):
        db = ChemicalDatabase(enable_online=False)
        db._smiles_cache_path = Path(tempfile.mkdtemp()) / "smiles.sqlite"
        from chemical_properties import SmilesResolution

        wrong_aromatic = "C1=CC(=C(C=C1[N+](=O)[O-])Cl)C#N"
        pentafluoroethane = "C(C(F)(F)F)(F)F"
        db._cache_smiles(
            SmilesResolution(
                smiles=wrong_aromatic,
                source="chemicals",
                method="chemicals_identifier_smiles",
                quality=0.99,
                identifier="r125",
            ),
            ["r125"],
        )
        db._cache_smiles(
            SmilesResolution(
                smiles=pentafluoroethane,
                source="chemicals",
                method="chemicals_identifier_smiles",
                quality=0.99,
                identifier="354-33-6",
            ),
            ["354-33-6"],
        )

        refrigerant = db.resolve_smiles_info("R125", fetch_online=False)
        aromatic = db.resolve_smiles_info("28163-00-0", fetch_online=False)

        self.assertIsNotNone(refrigerant)
        self.assertEqual(refrigerant.smiles, pentafluoroethane)
        self.assertNotEqual(refrigerant.smiles, wrong_aromatic)
        self.assertIsNotNone(aromatic)
        self.assertEqual(aromatic.smiles, wrong_aromatic)
        self.assertEqual(db._cached_smiles("r125").smiles, refrigerant.smiles)
        self.assertEqual(db._cached_smiles("354-33-6").smiles, refrigerant.smiles)

    def test_textbook_property_library_loads_appendix_b_values(self):
        library = TextbookPropertyLibrary()

        water = library.get("water")
        hexane = library.get("hexane")
        heptane = library.get("n-heptane")
        triethylamine = library.get("triethylamine")

        self.assertIsNotNone(water)
        self.assertAlmostEqual(water["Tc"], 647.1, places=1)
        self.assertAlmostEqual(water["Pc"], 220.55, places=2)
        self.assertAlmostEqual(water["omega"], 0.345, places=3)
        self.assertAlmostEqual(water["Hvap"], 40.66, places=2)
        self.assertIsNotNone(hexane)
        self.assertIsNotNone(heptane)
        self.assertGreater(heptane["Tb"], hexane["Tb"])
        self.assertIsNone(triethylamine)

    def test_local_scalar_hvap_values_are_normal_boiling_references(self):
        database = ChemicalDatabase(enable_online=False)
        expected_hvap = {
            "ethanol": 38.56,
            "benzene": 30.72,
            "toluene": 33.18,
            "acetone": 29.10,
            "acetic acid": 23.70,
            "acrylic acid": 28.405,
            "aniline": 44.50,
            "phenol": 46.18,
            "acetonitrile": 30.19,
            "propionic acid": 31.113,
        }

        for identifier, expected in expected_hvap.items():
            with self.subTest(identifier=identifier):
                props = database.get(identifier, fetch_online=False)
                self.assertIsNotNone(props)
                self.assertAlmostEqual(props.Hvap, expected, places=2)

        co2 = database.get("carbon dioxide", fetch_online=False)
        self.assertIsNotNone(co2)
        self.assertIsNone(co2.Hvap)
        self.assertAlmostEqual(co2.Hfus, 9.02, places=2)
        self.assertAlmostEqual(co2.Hsub, 25.2, places=1)
        self.assertEqual(co2.to_dict()["Hsub"], 25.2)

    def test_textbook_property_table_passes_sanity_audit(self):
        payload = json.loads(Path(ROOT, "data", "textbook_properties.json").read_text())
        chemicals = payload["chemicals"]

        self.assertGreaterEqual(len(chemicals), 90)
        self.assertNotIn("† Pseudoparameters for y N", chemicals)
        self.assertIn("Isobutane", chemicals)
        self.assertNotIn("iso-Butane", chemicals)
        self.assertIn("Isooctane", chemicals)
        self.assertNotIn("iso-Octane", chemicals)

        sulfuric = chemicals["Sulfuric acid"]
        self.assertIsNone(sulfuric["omega"])
        self.assertAlmostEqual(sulfuric["Tc"], 924.0)
        self.assertAlmostEqual(sulfuric["Pc"], 64.0)
        self.assertAlmostEqual(sulfuric["Tb"], 610.0)
        self.assertEqual(sulfuric["formula"], "H2SO4")

        self.assertEqual(chemicals["Ethyl acetate"]["formula"], "C4H8O2")
        self.assertEqual(chemicals["Nitric oxide (NO)"]["formula"], "NO")
        self.assertEqual(chemicals["Nitrous oxide (N2O)"]["formula"], "N2O")

        for name, entry in chemicals.items():
            with self.subTest(name=name):
                self.assertGreater(entry.get("MW", 1.0), 0.0)
                if entry.get("Pc") is not None:
                    self.assertGreater(entry["Pc"], 0.0)
                if entry.get("Tc") is not None and entry.get("Tb") is not None:
                    self.assertGreater(entry["Tc"], entry["Tb"])
                if entry.get("omega") is not None:
                    self.assertGreaterEqual(entry["omega"], -0.5)
                    self.assertLessEqual(entry["omega"], 1.5)
                if entry.get("antoine_A") is not None and entry.get("Tb") is not None:
                    P_at_Tb = 10 ** (
                        entry["antoine_A"]
                        - entry["antoine_B"]
                        / (entry["antoine_C"] + entry["Tb"] - 273.15)
                    )
                    self.assertAlmostEqual(P_at_Tb, 1.01325, delta=0.01)

    def test_textbook_antoine_conversion_reproduces_normal_boiling_pressure(self):
        resolver = PropertyResolver()

        for name in ["Water", "Ethanol", "n-Hexane", "n-Heptane", "Acetone"]:
            with self.subTest(name=name):
                entry = TextbookPropertyLibrary().get(name)
                antoine = resolver.get_antoine_local(name)
                self.assertIsNotNone(antoine)
                self.assertIn("Smith8 Appendix B", antoine.source)
                self.assertAlmostEqual(
                    antoine.vapor_pressure(entry["Tb"]),
                    1.01325,
                    delta=0.001,
                )

    def test_antoine_table_converts_mmhg_rows_to_bar_coefficients(self):
        table = get_antoine_table()

        water = table.get("water", 298.15)
        self.assertIsNotNone(water)
        self.assertEqual(water.source, "data/antoine.txt")
        self.assertAlmostEqual(water.A, 8.07131 + math.log10(0.001333223684), places=6)
        self.assertAlmostEqual(water.vapor_pressure(373.15), 1.01336, delta=0.002)

        formula_water = table.get("H2O", 298.15)
        self.assertIsNotNone(formula_water)
        self.assertEqual(formula_water.name, "Water")
        self.assertEqual(len(table.entries("water")), 2)
        self.assertEqual(
            [entry.record_id for entry in table.entries("water")],
            ["999", "998"],
        )

    def test_antoine_table_formula_lookup_rejects_isomer_ambiguity(self):
        table = get_antoine_table()

        self.assertIsNone(table.get("C2H6O", 298.15))
        ethyl_alcohol = table.get("ethyl alcohol", 298.15)
        methyl_ether = table.get("methyl ether", 250.0)

        self.assertIsNotNone(ethyl_alcohol)
        self.assertIsNotNone(methyl_ether)
        self.assertNotEqual(ethyl_alcohol.name, methyl_ether.name)

    def test_database_uses_canonical_identity_resolver_conservatively(self):
        database = ChemicalDatabase(enable_online=False)

        self.assertEqual(
            database.get("ethyl-alcohol", fetch_online=False).symbol, "C2H5OH"
        )
        self.assertEqual(
            database.get("ethyl alcohol", fetch_online=False).symbol, "C2H5OH"
        )
        self.assertEqual(database.get("64-17-5", fetch_online=False).symbol, "C2H5OH")
        self.assertIsNone(database.get("C2H6O", fetch_online=False))

    def test_pfd_components_can_omit_mw_when_lookup_resolves_them(self):
        pfd = (
            "PROCESS: Lookup Component Molecular Weights\n"
            "VERSION: 1.0\n"
            "THERMO_METHOD: IDEAL\n"
            "\n"
            "COMPONENTS:\n"
            "    EtOH | Ethanol\n"
            "    H2O | Water\n"
            "\n"
            "STREAM Feed : FEED -> FLASH-1.in\n"
            "    T = 78 [C]\n"
            "    P = 1 [bar]\n"
            "    F = 100 [kmol/h]\n"
            "    x = EtOH:0.50, H2O:0.50\n"
            "\n"
            "STREAM Vapor : FLASH-1.vap -> PRODUCT\n"
            "STREAM Liquid : FLASH-1.liq -> PRODUCT\n"
            "\n"
            "UNIT FLASH-1\n"
            "    TYPE: Flash\n"
            "    PORTS:\n"
            "        in  : inlet\n"
            "        vap : vapor_outlet\n"
            "        liq : liquid_outlet\n"
            "    PARAMS:\n"
            "        T = 78 [C]\n"
            "        P = 1 [bar]\n"
        )

        parsed = parse_pfd(pfd)
        self.assertIsNone(parsed.components[0].molecular_weight)

        result = Simulator.from_string(pfd).run()

        self.assertTrue(result.converged)
        self.assertAlmostEqual(
            result.streams["Feed"].MW, (46.07 + 18.015) / 2, delta=0.1
        )
        self.assertIn("EtOH", result.streams["Vapor"].composition)

    def test_pfd_component_mw_overrides_lookup_value(self):
        pfd = (
            "PROCESS: Override Component Molecular Weight\n"
            "VERSION: 1.0\n"
            "THERMO_METHOD: IDEAL\n"
            "\n"
            "COMPONENTS:\n"
            "    ETOH50 | Ethanol | MW=50.0\n"
            "\n"
            "STREAM Feed : FEED -> FLASH-1.in\n"
            "    T = 78 [C]\n"
            "    P = 1 [bar]\n"
            "    F = 100 [kmol/h]\n"
            "    x = ETOH50:1.0\n"
            "\n"
            "STREAM Vapor : FLASH-1.vap -> PRODUCT\n"
            "STREAM Liquid : FLASH-1.liq -> PRODUCT\n"
            "\n"
            "UNIT FLASH-1\n"
            "    TYPE: Flash\n"
            "    PORTS:\n"
            "        in  : inlet\n"
            "        vap : vapor_outlet\n"
            "        liq : liquid_outlet\n"
            "    PARAMS:\n"
            "        T = 78 [C]\n"
            "        P = 1 [bar]\n"
        )

        result = Simulator.from_string(pfd).run()

        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.streams["Feed"].MW, 50.0, places=8)

    def test_pfd_scalar_overrides_do_not_block_online_lookup(self):
        pfd = (
            "PROCESS: Scalar Overrides Still Resolve Online\n"
            "VERSION: 1.0\n"
            "THERMO_METHOD: IDEAL\n"
            "\n"
            "COMPONENTS:\n"
            "    MYST | Mystery online | MW=50.0, omega=0.2\n"
            "\n"
            "STREAM Feed : FEED -> PRODUCT\n"
            "    T = 25 [C]\n"
            "    P = 1 [bar]\n"
            "    F = 1 [kmol/h]\n"
            "    x = MYST:1.0\n"
        )
        online_props = ChemicalProperties(
            symbol="MYST",
            name="Mystery online",
            formula="C4H8O",
            MW=88.0,
            Tc=512.0,
            Pc=42.0,
            omega=0.72,
            Tb=350.0,
            source="pubchem",
        )

        with patch(
            "chemical_properties.OnlinePropertyFetcher.fetch_from_pubchem",
            return_value=online_props,
        ) as fetch:
            sim = Simulator.from_string(pfd)
            result = sim.run()

        fetch.assert_called_once()
        self.assertIn(fetch.call_args.args[0], {"MYST", "Mystery online"})
        self.assertTrue(result.converged)
        props = sim.thermo.props["MYST"]
        self.assertAlmostEqual(props.MW, 50.0, places=8)
        self.assertAlmostEqual(props.omega, 0.2, places=8)
        self.assertAlmostEqual(props.Tc, 512.0, places=8)
        self.assertAlmostEqual(props.Pc, 42.0, places=8)
        self.assertEqual(
            props.property_sources["MW"]["method"], "pfd_component_override"
        )
        self.assertEqual(
            props.property_sources["omega"]["method"], "pfd_component_override"
        )

    def test_pfd_online_lookup_false_blocks_online_component_lookup(self):
        pfd = (
            "PROCESS: Offline Component Definition\n"
            "VERSION: 1.0\n"
            "ONLINE_LOOKUP: false\n"
            "THERMO_METHOD: IDEAL\n"
            "\n"
            "COMPONENTS:\n"
            "    MYST | Mystery offline | MW=50.0, Tc=512.0, Pc=42.0, omega=0.2\n"
            "\n"
            "STREAM Feed : FEED -> PRODUCT\n"
            "    T = 25 [C]\n"
            "    P = 1 [bar]\n"
            "    F = 1 [kmol/h]\n"
            "    x = MYST:1.0\n"
        )

        with patch(
            "chemical_properties.OnlinePropertyFetcher.fetch_from_pubchem",
            side_effect=AssertionError("online lookup should be disabled by PFD"),
        ):
            sim = Simulator.from_string(pfd)
            result = sim.run()

        self.assertTrue(result.converged)
        self.assertFalse(sim.pfd.metadata.online_lookup)
        props = sim.thermo.props["MYST"]
        self.assertEqual(props.name, "Mystery offline")
        self.assertAlmostEqual(props.MW, 50.0, places=8)
        self.assertEqual(
            props.property_sources["MW"]["method"], "pfd_component_override"
        )

    def test_database_online_fallback_can_resolve_1_octen_3_one_mw(self):
        database = ChemicalDatabase(enable_online=True)
        expected = ChemicalProperties(
            symbol="1-octen-3-one",
            name="oct-1-en-3-one",
            formula="C8H14O",
            MW=126.2,
            source="pubchem",
        )

        with (
            patch.object(
                database,
                "resolve_smiles_info",
                return_value=None,
            ),
            patch.object(
                database.online_fetcher,
                "fetch_from_pubchem",
                return_value=expected,
            ) as fetch,
        ):
            props = database.get("1-octen-3-one", fetch_online=True)

        fetch.assert_called_once_with("1-octen-3-one")
        self.assertIsNotNone(props)
        self.assertEqual(props.source, "pubchem")
        self.assertAlmostEqual(props.MW, 126.2, places=8)
        self.assertEqual(props.formula, "C8H14O")

    def test_property_resolver_prefers_textbook_before_antoine_table_when_both_are_in_range(
        self,
    ):
        resolver = PropertyResolver()

        water = resolver.get_antoine_local("water", 413.15, require_in_range=True)

        self.assertIsNotNone(water)
        self.assertIn("Smith8 Appendix B", water.source)
        self.assertAlmostEqual(water.B, 1687.5380683314654)

        table_water = get_antoine_table().get("water", 413.15)
        self.assertIsNotNone(table_water)
        self.assertEqual(table_water.source, "data/antoine.txt")
        self.assertAlmostEqual(table_water.B, 1810.94)
        self.assertAlmostEqual(table_water.vapor_pressure(413.15), 3.58965, places=4)

    def test_property_resolver_uses_hydrated_textbook_antoine_without_network(self):
        resolver = PropertyResolver()

        with patch(
            "property_resolver.urllib.request.urlopen",
            side_effect=AssertionError(
                "network should not be used for Perry/local hit"
            ),
        ):
            antoine = resolver.get_antoine_local("carbon disulfide")
            result = resolver.resolve_vapor_pressure("carbon disulfide", 319.0)

        self.assertIsNotNone(antoine)
        self.assertEqual(antoine.source, "data/antoine.txt")
        self.assertEqual(result.method, "perry_2_8_vapor_pressure")
        self.assertEqual(result.source, "Perry 9th Table 2-8")
        self.assertAlmostEqual(result.value, 1.0015559, delta=0.01)

    def test_vapor_pressure_tables_do_not_match_formula_isomers(self):
        library = get_vapor_pressure_table_library()

        self.assertIsNone(library.get("C3H6O2", 400.0))
        self.assertIsNone(library.get("C4H8O2", 400.0))
        self.assertEqual(library.get("propionic acid", 400.0).key, "C2H5COOH")
        self.assertEqual(library.get("79-09-4", 400.0).key, "C2H5COOH")
        self.assertEqual(library.get("n-butyric acid", 400.0).key, "C3H7COOH")
        self.assertEqual(library.get("107-92-6", 400.0).key, "C3H7COOH")

        resolver = PropertyResolver()
        database = ChemicalDatabase(enable_online=False)

        ethyl_acetate = create_thermodynamics(["Ethyl acetate"], "IDEAL", database)
        ethyl_props = ethyl_acetate._resolver_known_props["Ethyl acetate"]
        ethyl_psat = resolver.resolve_vapor_pressure(
            ethyl_acetate.props["Ethyl acetate"].symbol,
            374.15,
            ethyl_props,
        )
        self.assertEqual(ethyl_psat.method, "perry_2_8_vapor_pressure")
        self.assertGreater(ethyl_psat.value, 1.5)

        methyl_acetate = create_thermodynamics(["Methyl acetate"], "IDEAL", database)
        methyl_props = methyl_acetate._resolver_known_props["Methyl acetate"]
        methyl_psat = resolver.resolve_vapor_pressure(
            methyl_acetate.props["Methyl acetate"].symbol,
            330.15,
            methyl_props,
        )
        self.assertEqual(methyl_psat.method, "perry_2_8_vapor_pressure")
        self.assertAlmostEqual(methyl_psat.value, 1.0, delta=0.08)

        dioxane = create_thermodynamics(["1,4-Dioxane"], "IDEAL", database)
        dioxane_props = dioxane._resolver_known_props["1,4-Dioxane"]
        dioxane_psat = resolver.resolve_vapor_pressure(
            dioxane.props["1,4-Dioxane"].symbol,
            393.15,
            dioxane_props,
        )
        self.assertEqual(dioxane_psat.method, "perry_2_8_vapor_pressure")
        self.assertLess(dioxane_psat.value, 2.5)

    def test_formula_symbol_is_retained_when_it_matches_specific_identity(self):
        resolver = PropertyResolver()

        ethanol_props = {
            "symbol": "C2H5OH",
            "name": "Ethanol",
            "CAS": "64-17-5",
        }
        ethanol_candidates = resolver._identifier_candidates("C2H5OH", ethanol_props)
        self.assertIn("C2H5OH", ethanol_candidates[:2])

        methyl_acetate_props = {
            "symbol": "C3H6O2",
            "name": "methyl acetate",
            "CAS": "79-20-9",
        }
        methyl_acetate_candidates = resolver._identifier_candidates(
            "C3H6O2",
            methyl_acetate_props,
        )
        self.assertNotIn("C3H6O2", methyl_acetate_candidates)

    def test_online_fetcher_uses_antoine_table_identifiers_before_online_antoine(self):
        fetcher = OnlinePropertyFetcher()
        props = ChemicalProperties(
            symbol="pubchem-cs2",
            name="carbon disulfide",
            formula="CS2",
            MW=76.14,
            Tb=319.0,
            Tc=552.0,
            Pc=79.0,
            Hvap=27.0,
        )

        with patch(
            "property_resolver.PropertyResolver.get_antoine_online",
            side_effect=AssertionError("online Antoine lookup should not be used"),
        ):
            fetcher._estimate_missing_properties(props)

        self.assertIsNotNone(props.antoine_A)
        self.assertEqual(props.source, "local")
        self.assertIn("data/antoine.txt", props.antoine_source)
        self.assertIn("data/antoine.txt", props.property_sources["Antoine"]["notes"])
        self.assertAlmostEqual(props.Psat(319.0), 1.0, delta=0.06)

    def test_online_fetcher_does_not_use_formula_alias_for_antoine_hydration(self):
        fetcher = OnlinePropertyFetcher()
        props = ChemicalProperties(
            symbol="4-methylpyridine",
            name="4-methylpyridine",
            formula="C6H7N",
            CAS="108-89-4",
            MW=93.13,
            Tb=418.0388888888889,
            Tc=645.8,
            Pc=46.8,
            Hvap=40.1,
        )

        antoine = fetcher._get_resolver_antoine(
            PropertyResolver(),
            props,
            local=True,
        )

        self.assertIsNone(antoine)

    def test_database_uses_textbook_before_online_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "empty_chemicals.json"
            db_path.write_text(json.dumps({"chemicals": {}}))
            database = ChemicalDatabase(db_path=db_path, enable_online=True)

            with patch(
                "chemical_properties.urllib.request.urlopen",
                side_effect=AssertionError(
                    "network should not be used for textbook hit"
                ),
            ):
                props = database.get("methyl ethyl ketone")

        self.assertIsNotNone(props)
        self.assertEqual(props.source, "textbook")
        self.assertAlmostEqual(props.MW, 72.107, places=3)
        self.assertAlmostEqual(props.Tb, 352.8, delta=0.06)
        self.assertAlmostEqual(props.Pc, 41.50, places=2)
        self.assertAlmostEqual(props.omega, 0.323, places=3)

    def test_textbook_hit_keeps_source_and_hydrates_known_cas_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "empty_chemicals.json"
            db_path.write_text(json.dumps({"chemicals": {}}))
            database = ChemicalDatabase(db_path=db_path, enable_online=False)

            props = database.get("isobutanol", fetch_online=False)

        self.assertIsNotNone(props)
        self.assertEqual(props.source, "textbook")
        self.assertEqual(props.name, "iso-Butanol")
        self.assertEqual(props.CAS, "78-83-1")
        self.assertAlmostEqual(props.MW, 74.12, delta=0.005)
        self.assertEqual(props.property_sources["MW"]["method"], "rdkit_molwt_from_smiles")

    def test_pfd_missing_textbook_mw_keeps_isomers_distinct(self):
        components = [
            ("ISO", "isobutanol", "78-83-1"),
            ("NBU", "butanol", "71-36-3"),
            ("SEC", "2-butanol", "78-92-2"),
        ]
        for ordered in (components, list(reversed(components))):
            with self.subTest(order=[c[0] for c in ordered]):
                definitions = "\n".join(f"    {symbol} | {name}" for symbol, name, _ in ordered)
                sim = Simulator.from_string(
                    "PROCESS: Butanol identity and mass\nVERSION: 1.0\n"
                    "ONLINE_LOOKUP: false\nTHERMO_METHOD: NRTL\n"
                    f"COMPONENTS:\n{definitions}\n"
                    "STREAM Feed : FEED -> PRODUCT\n"
                    "    T = 25 [C]\n    P = 1 [bar]\n    F = 1 [kmol/h]\n"
                    "    x = ISO:0.25, NBU:0.50, SEC:0.25\n"
                ).initialize()
                for symbol, _, cas in components:
                    props = sim.thermo.props[symbol]
                    self.assertEqual(props.CAS, cas)
                    self.assertEqual(sim.thermo.component_cas[symbol], cas)
                    self.assertAlmostEqual(props.MW, 74.12, delta=0.005)
                result = sim.run()
                self.assertTrue(result.converged)
                self.assertAlmostEqual(result.streams["Feed"].MW, 74.12, delta=0.005)

    def test_pfd_structure_fallback_preserves_known_specific_cas(self):
        sim = Simulator.from_string(
            "PROCESS: Specific alcohol identities\nVERSION: 1.0\n"
            "ONLINE_LOOKUP: false\nTHERMO_METHOD: NRTL\n"
            "COMPONENTS:\n    PEN | 1-pentanol\n    HEX | 1-hexanol\n"
            "STREAM Feed : FEED -> PRODUCT\n"
            "    T = 25 [C]\n    P = 1 [bar]\n    F = 1 [kmol/h]\n"
            "    x = PEN:0.5, HEX:0.5\n"
        ).initialize()
        for symbol, cas, mw in (("PEN", "71-41-0", 88.15), ("HEX", "111-27-3", 102.177)):
            self.assertEqual(sim.thermo.props[symbol].CAS, cas)
            self.assertEqual(sim.thermo.component_cas[symbol], cas)
            self.assertAlmostEqual(sim.thermo.props[symbol].MW, mw, delta=0.005)

    def test_pfd_valid_mw_override_survives_textbook_hydration(self):
        sim = Simulator.from_string(
            "PROCESS: Explicit molecular weight\nVERSION: 1.0\n"
            "ONLINE_LOOKUP: false\nTHERMO_METHOD: IDEAL\n"
            "COMPONENTS:\n    ISO | isobutanol | MW=80\n"
            "STREAM Feed : FEED -> PRODUCT\n"
            "    T = 25 [C]\n    P = 1 [bar]\n    F = 1 [kmol/h]\n"
            "    x = ISO:1\n"
        ).initialize()
        self.assertEqual(sim.thermo.props["ISO"].MW, 80.0)
        self.assertEqual(sim.thermo.props["ISO"].property_sources["MW"]["method"], "pfd_component_override")

    def test_hydration_does_not_assign_isomer_cas_from_a_table_formula(self):
        resolver = types.SimpleNamespace(
            resolve_melting_point=lambda *a, **kw: None,
            resolve_triple_point=lambda *a, **kw: {},
            resolve_boiling_point=lambda *a, **kw: None,
            resolve_critical_properties=lambda *a, **kw: {},
            resolve_hvap=lambda *a, **kw: None,
            resolve_hfus=lambda *a, **kw: None,
            resolve_formation_properties=lambda *a, **kw: {},
        )
        database = ChemicalDatabase(enable_online=False)
        props = ChemicalProperties(symbol="C6H14", name="unrecognized hexane isomer", formula="C6H14", MW=86.18)
        with patch("property_resolver.get_property_resolver", return_value=resolver):
            database._hydrate_properties(props, allow_online=False)
        self.assertEqual(props.CAS, "")

    def test_database_supplements_textbook_hit_with_antoine_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "empty_chemicals.json"
            db_path.write_text(json.dumps({"chemicals": {}}))
            database = ChemicalDatabase(db_path=db_path, enable_online=True)

            with patch(
                "chemical_properties.urllib.request.urlopen",
                side_effect=AssertionError(
                    "network should not be used for textbook plus Antoine-table hit"
                ),
            ):
                props = database.get("carbon disulfide")

        self.assertIsNotNone(props)
        self.assertIsNotNone(props.Tc)
        self.assertIsNotNone(props.Pc)
        self.assertIsNone(props.antoine_A)
        self.assertEqual(props.source, "textbook")
        self.assertAlmostEqual(props.Psat(props.Tb), 1.01325, delta=0.02)

    def test_property_resolver_uses_hydrated_database_before_network_for_local_properties(
        self,
    ):
        resolver = PropertyResolver()

        with patch(
            "property_resolver.urllib.request.urlopen",
            side_effect=AssertionError(
                "network should not be used for Perry/textbook hit"
            ),
        ):
            antoine = resolver.get_antoine_local("n-Heptane")
            critical = resolver.resolve_critical_properties("n-Heptane")

        self.assertIsNotNone(antoine)
        self.assertAlmostEqual(antoine.vapor_pressure(371.55), 1.01325, delta=0.001)
        self.assertEqual(critical["Tc"].source, "local")
        self.assertEqual(critical["Tc"].method, "coolprop_HEOS_critical")
        self.assertAlmostEqual(critical["Tc"].value, 541.2259150893126)
        self.assertAlmostEqual(critical["Pc"].value, 27.73824280294774)
        self.assertAlmostEqual(critical["omega"].value, 0.349, places=3)

    def test_property_resolver_converts_pubchem_critical_pressure_units(self):
        resolver = PropertyResolver()
        section = {
            "Section": [
                {
                    "TOCHeading": "Critical Temperature",
                    "Information": [
                        {"Value": {"StringWithMarkup": [{"String": "513.9 K"}]}}
                    ],
                },
                {
                    "TOCHeading": "Critical Pressure",
                    "Information": [
                        {"Value": {"StringWithMarkup": [{"String": "6.148 MPa"}]}}
                    ],
                },
            ],
        }
        result = {}

        resolver._extract_critical_from_section(section, result)

        self.assertAlmostEqual(result["Tc"], 513.9)
        self.assertAlmostEqual(result["Pc"], 61.48)

    def test_property_resolver_caches_failed_online_antoine_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolver = PropertyResolver()
            resolver.CACHE_DIR = Path(tmp)
            resolver._online_cache = {}
            not_found = urllib.error.HTTPError("url", 404, "Not Found", None, None)

            with patch(
                "property_resolver.urllib.request.urlopen", side_effect=not_found
            ) as urlopen:
                self.assertIsNone(resolver.get_antoine_online("not-a-real-chemical"))
                first_call_count = urlopen.call_count
                self.assertIsNone(resolver.get_antoine_online("not-a-real-chemical"))

            self.assertGreater(first_call_count, 0)
            self.assertEqual(urlopen.call_count, first_call_count)

            resolver2 = PropertyResolver()
            resolver2.CACHE_DIR = Path(tmp)
            resolver2._online_cache = {}

            with patch(
                "property_resolver.urllib.request.urlopen",
                side_effect=AssertionError("network repeated"),
            ):
                self.assertIsNone(resolver2.get_antoine_online("not-a-real-chemical"))
            self.assertTrue(Path(tmp, "property_cache.sqlite").is_file())
            self.assertEqual(list(Path(tmp).glob("*.json")), [])

    def test_property_resolver_parses_nist_antoine_rows(self):
        resolver = PropertyResolver()
        html = """
        <h2>Antoine Equation Parameters</h2>
        log10(P) = A - (B / (T + C))
        P = vapor pressure (bar)
        T = temperature (K)
        Temperature (K) A B C Reference Comment
        185.29 to 295.60 4.81803 1635.409-27.338 Carruth and Kobayashi, 1973
        299.07 to 372.43 4.02832 1268.636-56.199 Williamham, Taylor, et al., 1945
        """

        antoine = resolver._parse_nist_antoine(html)

        self.assertIsNotNone(antoine)
        self.assertAlmostEqual(antoine.A, 4.02832)
        self.assertAlmostEqual(antoine.B, 1268.636)
        self.assertAlmostEqual(antoine.C, 216.951)
        self.assertAlmostEqual(antoine.vapor_pressure(371.58), 1.01337, places=4)

    def test_property_resolver_parses_nist_antoine_rows_by_temperature(self):
        resolver = PropertyResolver()
        html = """
        <h2>Antoine Equation Parameters</h2>
        log10(P) = A - (B / (T + C))
        P = vapor pressure (bar)
        T = temperature (K)
        Temperature (K) A B C Reference Comment
        185.29 to 295.60 4.81803 1635.409-27.338 Carruth and Kobayashi, 1973
        299.07 to 372.43 4.02832 1268.636-56.199 Williamham, Taylor, et al., 1945
        400.00 to 500.00 4.50000 1400.000-40.000 Example, 2026
        """

        cryogenic = resolver._parse_nist_antoine(html, 290.0)
        normal = resolver._parse_nist_antoine(html, 350.0)
        hot = resolver._parse_nist_antoine(html, 450.0)

        self.assertAlmostEqual(cryogenic.A, 4.81803)
        self.assertAlmostEqual(normal.A, 4.02832)
        self.assertAlmostEqual(hot.A, 4.5)

    def test_thermodynamics_psat_skips_out_of_range_local_antoine_before_estimating(
        self,
    ):
        with patch(
            "property_resolver.urllib.request.urlopen",
            side_effect=OSError("network off"),
        ):
            thermo = create_thermodynamics(["ethanol", "water"], "NRTL-RK")
            T = 418.4109
            psat = thermo.Psat("ethanol", T)

        self.assertIsNone(thermo.props["ethanol"].Psat_antoine(T))
        self.assertIsNone(
            thermo.props["ethanol"].Psat_antoine(T, allow_extrapolation=True)
        )
        self.assertAlmostEqual(psat, 8.69703225800406, delta=0.01)

    def test_online_property_fetcher_caches_failed_pubchem_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            fetcher = OnlinePropertyFetcher(cache_dir=tmp)
            not_found = urllib.error.HTTPError("url", 404, "Not Found", None, None)

            with patch(
                "chemical_properties.urllib.request.urlopen", side_effect=not_found
            ) as urlopen:
                self.assertIsNone(fetcher.fetch_from_pubchem("not-a-real-chemical"))
                self.assertIsNone(fetcher.fetch_from_pubchem("not-a-real-chemical"))

            self.assertEqual(urlopen.call_count, 1)

            fetcher2 = OnlinePropertyFetcher(cache_dir=tmp)
            with patch(
                "chemical_properties.urllib.request.urlopen",
                side_effect=AssertionError("network repeated"),
            ):
                self.assertIsNone(fetcher2.fetch_from_pubchem("not-a-real-chemical"))
            self.assertTrue(Path(tmp, "property_cache.sqlite").is_file())
            self.assertEqual(list(Path(tmp).glob("*.json")), [])

    def test_online_property_fetcher_does_not_cache_transient_network_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            fetcher = OnlinePropertyFetcher(cache_dir=tmp)

            with patch(
                "chemical_properties.urllib.request.urlopen",
                side_effect=OSError("Temporary failure in name resolution"),
            ):
                self.assertIsNone(fetcher.fetch_from_pubchem("acrylic acid"))

            self.assertEqual(list(Path(tmp).glob("*acrylic*")), [])

    def test_online_property_fetcher_ignores_legacy_component_hvap_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            fetcher = OnlinePropertyFetcher(cache_dir=tmp)
            unsafe = ChemicalProperties(
                symbol="ethanol",
                name="ethanol",
                formula="C2H6O",
                MW=46.07,
                Tb=351.5,
                Hvap=42.3,
                source="pubchem",
            )
            fetcher._save_cache("pubchem_ethanol", unsafe.to_dict())
            not_found = urllib.error.HTTPError("url", 404, "Not Found", None, None)

            with patch(
                "chemical_properties.urllib.request.urlopen", side_effect=not_found
            ) as urlopen:
                result = fetcher.fetch_from_pubchem("ethanol")

        self.assertIsNone(result)
        self.assertEqual(urlopen.call_count, 1)

    def test_online_property_fetcher_enriches_identity_without_scalar_properties(self):
        fetcher = OnlinePropertyFetcher()
        self.assertEqual(
            fetcher._pubchem_component_cache_key("acrylic acid"),
            "pubchem_component_v6_acrylic acid",
        )
        props = ChemicalProperties(
            symbol="acrylic acid",
            name="prop-2-enoic acid",
            formula="C3H4O2",
            MW=72.06,
            source="pubchem",
        )
        data = {
            "Record": {
                "Section": [
                    {
                        "TOCHeading": "CAS",
                        "Information": [
                            {"Value": {"StringWithMarkup": [{"String": "79-10-7"}]}}
                        ],
                    },
                    {
                        "TOCHeading": "Boiling Point",
                        "Information": [
                            {"Value": {"StringWithMarkup": [{"String": "286 °F"}]}}
                        ],
                    },
                    {
                        "TOCHeading": "Freezing Point",
                        "Information": [
                            {"Value": {"StringWithMarkup": [{"String": "55 °F"}]}}
                        ],
                    },
                    {
                        "TOCHeading": "Critical Temperature",
                        "Information": [
                            {"Value": {"StringWithMarkup": [{"String": "647.6 °F"}]}}
                        ],
                    },
                    {
                        "TOCHeading": "Heat of Vaporization",
                        "Information": [
                            {
                                "Value": {
                                    "StringWithMarkup": [{"String": "272.7 Btu/lb"}]
                                }
                            }
                        ],
                    },
                ]
            }
        }

        with patch("chemical_properties.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
                data
            ).encode()
            fetcher._apply_pubchem_identity_properties(props, 6581)

        self.assertEqual(props.CAS, "79-10-7")
        self.assertIsNone(props.Tb)
        self.assertIsNone(props.Tm)
        self.assertIsNone(props.Tc)
        self.assertIsNone(props.Pc)
        self.assertIsNone(props.Hvap)
        self.assertEqual(
            props.property_sources["CAS"]["method"],
            "pubchem_pug_view",
        )

    def test_pubchem_component_v2_migration_keeps_identity_not_legacy_scalars(self):
        for legacy_version in (2, 3):
            with (
                self.subTest(legacy_version=legacy_version),
                tempfile.TemporaryDirectory() as directory,
            ):
                fetcher = OnlinePropertyFetcher(cache_dir=directory)
                store = fetcher._sqlite_cache()
                store.set(
                    f"pubchem_component_v{legacy_version}_Cycloheptane",
                    {
                        "symbol": "Cycloheptane",
                        "name": "cycloheptane",
                        "formula": "C7H14",
                        "CAS": "291-64-5",
                        "MW": 98.19,
                        "smiles": "C1CCCCCC1",
                        "Tb": 391.65,
                        "Tm": 261.15,
                        "Tc": 604.302,
                        "phase_at_STP": "liquid",
                        "lookup_warnings": [
                            "Boiling point was estimated from pubchem_pug_view.",
                        ],
                        "property_sources": {
                            "Tb": {
                                "source": "pubchem",
                                "method": "pubchem_pug_view",
                                "quality": 0.85,
                            },
                            "Tm": {
                                "source": "pubchem",
                                "method": "pubchem_pug_view",
                                "quality": 0.85,
                            },
                            "Tc": {
                                "source": "local",
                                "method": "effective_critical",
                                "quality": 0.94,
                            },
                        },
                    },
                )
                fetcher._legacy_component_cache_migrated = False
                fetcher._sqlite_cache_state = None
                migrated = fetcher._sqlite_cache().get(
                    "pubchem_component_v4_Cycloheptane"
                )

                self.assertEqual(migrated["CAS"], "291-64-5")
                self.assertEqual(migrated["MW"], 98.19)
                self.assertEqual(migrated["smiles"], "C1CCCCCC1")
                self.assertIsNone(migrated["Tb"])
                self.assertIsNone(migrated["Tm"])
                self.assertEqual(migrated["Tc"], 604.302)
                self.assertNotIn("Tb", migrated["property_sources"])
                self.assertNotIn("Tm", migrated["property_sources"])
                self.assertEqual(
                    migrated["property_sources"]["Tc"]["method"],
                    "effective_critical",
                )
        self.assertIsNone(migrated["phase_at_STP"])
        self.assertFalse(migrated["lookup_warnings"])

    def test_online_property_fetcher_leaves_combined_critical_to_dedicated_resolver(
        self,
    ):
        fetcher = OnlinePropertyFetcher()
        props = ChemicalProperties(
            symbol="acrylic acid",
            name="prop-2-enoic acid",
            formula="C3H4O2",
            MW=72.06,
            source="pubchem",
        )
        data = {
            "Record": {
                "Section": [
                    {
                        "TOCHeading": "Critical Temperature and Pressure",
                        "Information": [
                            {
                                "Value": {
                                    "StringWithMarkup": [
                                        {
                                            "String": "Critical temperature: 648 °F = 342 °C; critical pressure: 57 atm"
                                        }
                                    ]
                                }
                            }
                        ],
                    },
                    {
                        "TOCHeading": "Heat of Vaporization",
                        "Information": [
                            {
                                "Value": {
                                    "StringWithMarkup": [
                                        {"String": "10,955.1 gcal/gmole"}
                                    ]
                                }
                            }
                        ],
                    },
                ]
            }
        }

        with patch("chemical_properties.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
                data
            ).encode()
            fetcher._apply_pubchem_identity_properties(props, 6581)

        self.assertIsNone(props.Tc)
        self.assertIsNone(props.Pc)
        self.assertIsNone(props.Hvap)

    def test_online_property_fetcher_never_stores_unresolved_hvap_text(self):
        fetcher = OnlinePropertyFetcher()
        props = ChemicalProperties(
            symbol="benzoic acid",
            name="benzoic acid",
            formula="C7H6O2",
            MW=122.12,
            source="pubchem",
        )
        data = {
            "Record": {
                "Section": [
                    {
                        "TOCHeading": "Heat of Vaporization",
                        "Information": [
                            {
                                "Value": {
                                    "StringWithMarkup": [
                                        {"String": "534 KJ/mol at 140 °C"}
                                    ]
                                }
                            }
                        ],
                    },
                ]
            }
        }

        with patch("chemical_properties.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
                data
            ).encode()
            fetcher._apply_pubchem_identity_properties(props, 243)

        self.assertIsNone(props.Hvap)
        self.assertFalse(
            any(
                "heat of vaporization" in warning.lower()
                for warning in props.lookup_warnings
            )
        )

    def test_online_property_fetcher_does_not_parse_unvalidated_critical_text(self):
        fetcher = OnlinePropertyFetcher()
        props = ChemicalProperties(
            symbol="ethyl bromide",
            name="bromoethane",
            formula="C2H5Br",
            MW=108.97,
            source="pubchem",
        )
        data = {
            "Record": {
                "Section": [
                    {
                        "TOCHeading": "Boiling Point",
                        "Information": [
                            {"Value": {"StringWithMarkup": [{"String": "38.4 °C"}]}}
                        ],
                    },
                    {
                        "TOCHeading": "Critical Temperature and Pressure",
                        "Information": [
                            {
                                "Value": {
                                    "StringWithMarkup": [
                                        {
                                            "String": "Critical temperature: 776.8 °C; critical pressure: 6.2315X10+6 Pa"
                                        }
                                    ]
                                }
                            }
                        ],
                    },
                ]
            }
        }

        with patch("chemical_properties.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
                data
            ).encode()
            fetcher._apply_pubchem_identity_properties(props, 6332)

        self.assertIsNone(props.Tc)
        self.assertIsNone(props.Pc)
        self.assertFalse(props.lookup_warnings)

    def test_nonvolatile_ionic_species_do_not_get_vle_estimates(self):
        fetcher = OnlinePropertyFetcher()
        props = ChemicalProperties(
            symbol="sodium acetate",
            name="sodium acetate",
            formula="C2H3NaO2",
            MW=82.03,
            Tm=597.15,
        )

        fetcher._estimate_missing_properties(props)

        self.assertEqual(props.phase_at_STP, "solid")
        self.assertIsNone(props.Tb)
        self.assertIsNone(props.Tc)
        self.assertIsNone(props.Pc)
        self.assertIsNone(props.Hvap)
        self.assertTrue(
            any("ionic or nonvolatile" in warning for warning in props.lookup_warnings)
        )

    def test_estimated_boiling_point_is_stored_as_kelvin(self):
        fetcher = OnlinePropertyFetcher()
        props = ChemicalProperties(
            symbol="unknown",
            name="unknown",
            formula="Xx",
            MW=88.0,
        )

        with (
            patch(
                "property_resolver.PropertyResolver.get_antoine_online",
                return_value=None,
            ),
            patch(
                "property_resolver.PropertyResolver._fetch_phase_change_online",
                return_value=None,
            ),
        ):
            fetcher._estimate_missing_properties(props)

        self.assertGreater(props.Tb, 273.15)
        self.assertGreater(props.Tc, props.Tb)
        self.assertIsNotNone(props.Hvap)
        self.assertEqual(props.property_sources["Hvap"]["method"], "trouton_watson")
        self.assertAlmostEqual(
            props.property_sources["Hvap"]["quality"],
            0.72
            * min(
                props.property_sources["Tb"]["quality"],
                props.property_sources["Tc"]["quality"],
            ),
        )

    def test_lee_kesler_acentric_factor_estimate_uses_psat_over_pc(self):
        fetcher = OnlinePropertyFetcher()
        props = ChemicalProperties(
            symbol="acrylic acid",
            name="prop-2-enoic acid",
            formula="C3H4O2",
            MW=72.06,
            Tb=414.26111111111106,
            Tc=615.3722222222223,
            Pc=57.75525,
            Hvap=45.8361384,
        )

        with patch(
            "property_resolver.PropertyResolver.get_antoine_online", return_value=None
        ):
            fetcher._estimate_missing_properties(props)

        self.assertAlmostEqual(props.omega, 0.538324, places=6)
        self.assertEqual(props.property_sources["omega"]["method"], "perry_critical")

    def test_lookup_warnings_report_missing_online_properties(self):
        fetcher = OnlinePropertyFetcher()
        props = ChemicalProperties(
            symbol="acrylic acid",
            name="prop-2-enoic acid",
            formula="C3H4O2",
            MW=72.06,
            Tb=414.261,
            Hvap=39.355,
        )
        direct_fields = {
            "Tb": True,
            "Hvap": True,
            "Tc": False,
            "Pc": False,
            "omega": False,
            "Antoine": False,
        }

        warnings = fetcher._build_lookup_warnings("acrylic acid", props, direct_fields)

        self.assertTrue(any("Tc" in warning for warning in warnings))
        self.assertTrue(any("Antoine" in warning for warning in warnings))
        self.assertTrue(any("Clausius-Clapeyron" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
