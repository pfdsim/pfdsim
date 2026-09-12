import json
import unittest
import urllib.error
from unittest.mock import Mock

from property_resolution.online_viscosity import (
    PubChemViscosityAnnotation,
    PubChemViscosityFetchError,
    PubChemViscosityFetcher,
    extract_pubchem_viscosity_annotations,
    normalize_pubchem_viscosity_annotations,
    parse_pubchem_viscosity_annotation,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def annotation(text, *, comment="", predictive=False):
    return PubChemViscosityAnnotation(
        text=text,
        references=("Example reference",),
        source_name="HSDB",
        source_url="https://example.invalid/source",
        comment=comment,
        predictive=predictive,
    )


class PubChemViscosityParserTests(unittest.TestCase):
    def normalized(self, text, **kwargs):
        return normalize_pubchem_viscosity_annotations((annotation(text, **kwargs),))

    def test_extracts_attributed_viscosity_only(self):
        payload = {
            "Record": {
                "Section": [
                    {
                        "TOCHeading": "Viscosity",
                        "Information": [
                            {
                                "ReferenceNumber": 42,
                                "Description": "PEER REVIEWED",
                                "Reference": ["Merck Index, p. 593"],
                                "Value": {
                                    "StringWithMarkup": [
                                        {"String": "2.47 cP at 20 °C"},
                                    ],
                                },
                            },
                        ],
                    },
                    {
                        "TOCHeading": "Density",
                        "Information": [
                            {
                                "Value": {
                                    "StringWithMarkup": [
                                        {"String": "1.10 g/cm3 at 20 °C"},
                                    ],
                                },
                            },
                        ],
                    },
                ],
                "Reference": [
                    {
                        "ReferenceNumber": 42,
                        "SourceName": "Hazardous Substances Data Bank (HSDB)",
                        "URL": "https://pubchem.ncbi.nlm.nih.gov/source/hsdb/80",
                    },
                ],
            },
        }

        records = extract_pubchem_viscosity_annotations(payload)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].text, "2.47 cP at 20 °C")
        self.assertEqual(records[0].reference_number, 42)
        self.assertEqual(records[0].references, ("Merck Index, p. 593",))
        self.assertEqual(
            records[0].source_name,
            "Hazardous Substances Data Bank (HSDB)",
        )
        self.assertTrue(records[0].peer_reviewed)

        normalized = normalize_pubchem_viscosity_annotations(records)
        self.assertEqual(normalized.points[0].reference_number, 42)
        self.assertEqual(normalized.points[0].reference, "Merck Index, p. 593")
        self.assertTrue(normalized.points[0].peer_reviewed)

    def test_record_title_participates_in_mixture_screening(self):
        payload = {
            "Record": {
                "RecordTitle": "Example chemical (50% solution)",
                "TOCHeading": "Viscosity",
                "Information": [
                    {
                        "Value": {
                            "StringWithMarkup": [
                                {"String": "2.47 cP at 20 °C"},
                            ],
                        },
                    },
                ],
            },
        }

        records = extract_pubchem_viscosity_annotations(payload)
        normalized = normalize_pubchem_viscosity_annotations(records)

        self.assertEqual(records[0].record_title, "Example chemical (50% solution)")
        self.assertEqual(normalized.points, ())
        self.assertIn("solution", normalized.rejected[0].reason)

        isomer_mixture = normalize_pubchem_viscosity_annotations((
            PubChemViscosityAnnotation(
                text="2.47 cP at 20 °C",
                record_title="Hydroxypropyl acrylate, isomers",
            ),
        ))
        self.assertEqual(isomer_mixture.points, ())
        self.assertIn("mixture", isomer_mixture.rejected[0].reason)

    def test_extracts_number_and_unit_payload(self):
        payload = {
            "Record": {
                "TOCHeading": "Viscosity",
                "Information": [
                    {"Value": {"Number": [2.47], "Unit": "cP"}},
                ],
            },
        }

        records = extract_pubchem_viscosity_annotations(payload)

        self.assertEqual(records[0].text, "2.47 cP")

    def test_extended_reference_is_preserved_when_inline_reference_is_absent(self):
        payload = {
            "Record": {
                "TOCHeading": "Viscosity",
                "Information": [
                    {
                        "ExtendedReference": [
                            {"Citation": "A primary viscosity paper"},
                        ],
                        "Value": {
                            "StringWithMarkup": [
                                {"String": "2.47 cP at 20 °C"},
                            ],
                        },
                    },
                ],
            },
        }

        records = extract_pubchem_viscosity_annotations(payload)
        normalized = normalize_pubchem_viscosity_annotations(records)

        self.assertEqual(records[0].references, ("A primary viscosity paper",))
        self.assertEqual(
            normalized.points[0].reference,
            "A primary viscosity paper",
        )

    def test_normalizes_common_dynamic_units_and_typographical_variants(self):
        cases = (
            ("2.47 cP at 20 °C", 0.00247, 293.15),
            ("0.560 mPa-s at 25 °C", 0.000560, 298.15),
            ("7.58 centapoise @ 25 °C", 0.00758, 298.15),
            ("3.19 centpoise @ 50 °C", 0.00319, 323.15),
            ("0.0247 poise at 293.15 K", 0.00247, 293.15),
            ("2470 microPascal-seconds at 68 °F", 0.00247, 293.15),
            ("2.47 cPs at 20 degrees C", 0.00247, 293.15),
            ("2.47 kg/(m*s) at 20 deg C", 2.47, 293.15),
            ("1 millipoise at 20 °C", 0.0001, 293.15),
            ("1 mP at 20 °C", 0.0001, 293.15),
            (".5 cP at 20 °C", 0.0005, 293.15),
        )
        for text, expected_viscosity, expected_temperature in cases:
            with self.subTest(text=text):
                result = self.normalized(text)
                self.assertEqual(result.rejected, ())
                self.assertEqual(len(result.points), 1)
                self.assertAlmostEqual(
                    result.points[0].viscosity_Pa_s,
                    expected_viscosity,
                    places=12,
                )
                self.assertAlmostEqual(
                    result.points[0].temperature_K,
                    expected_temperature,
                    places=9,
                )

    def test_parses_packed_series_and_temperature_first_form(self):
        packed = self.normalized(
            "1.165 mPa-s at -25 °C; 0.778 mPa-s at 0 °C; "
            "0.560 mPa-s at 25 °C"
        )
        reversed_order = self.normalized("20 °C: 0.32 cP; 40 °C: 0.27 cP")

        self.assertEqual(len(packed.points), 3)
        self.assertEqual(len(reversed_order.points), 2)
        self.assertAlmostEqual(packed.points[0].temperature_K, 248.15)
        self.assertAlmostEqual(reversed_order.points[1].viscosity_Pa_s, 0.00027)

    def test_parses_shared_unit_series_without_dropping_last_point(self):
        result = self.normalized(
            "6.4 at 200 K; 9.4 at 300 K; 12.2 at 400 K "
            "(all values in uPa.s)"
        )

        self.assertEqual(len(result.points), 3)
        self.assertAlmostEqual(result.points[-1].viscosity_Pa_s, 12.2e-6)

    def test_comma_series_does_not_cross_pair_measurements_or_pressures(self):
        for text in (
            "2 cP at 20 C, 3 cP at 30 C and 2 MPa",
            "20 C: 2 cP, 30 C: 3 cP and 2 MPa",
        ):
            with self.subTest(text=text):
                result = self.normalized(text)
                self.assertEqual(len(result.points), 2)
                first, second = result.points
                self.assertAlmostEqual(first.viscosity_Pa_s, 0.002)
                self.assertAlmostEqual(first.temperature_K, 293.15)
                self.assertIsNone(first.pressure_bar)
                self.assertAlmostEqual(second.viscosity_Pa_s, 0.003)
                self.assertAlmostEqual(second.temperature_K, 303.15)
                self.assertAlmostEqual(second.pressure_bar, 20.0)

    def test_handles_equivalent_unit_parenthesis_and_explicit_pressure(self):
        result = self.normalized("Liquid: 53 mPa-s (=cP) at 75 °C and 2 MPa")

        self.assertEqual(len(result.points), 1)
        self.assertAlmostEqual(result.points[0].viscosity_Pa_s, 0.053)
        self.assertAlmostEqual(result.points[0].temperature_K, 348.15)
        self.assertAlmostEqual(result.points[0].pressure_bar, 20.0)
        self.assertEqual(result.points[0].phase_basis, "liquid")

    def test_handles_parenthesized_temperature_and_decimal_or_grouping_comma(self):
        parenthesized = self.normalized("2.47 cP (20 °C)")
        decimal = self.normalized("2,47 cP at 20 °C")
        grouped = self.normalized("1,250 cP at 20 °C")

        self.assertAlmostEqual(parenthesized.points[0].viscosity_Pa_s, 0.00247)
        self.assertAlmostEqual(decimal.points[0].viscosity_Pa_s, 0.00247)
        self.assertAlmostEqual(grouped.points[0].viscosity_Pa_s, 1.25)

    def test_mojibake_kinematic_unit_is_rejected(self):
        result = self.normalized("1.95 mmÂ²/s at 20Â °C")

        self.assertEqual(result.points, ())
        self.assertIn("kinematic", result.rejected[0].reason)

        empirical = self.normalized("45 Saybolt Universal Seconds at 100 °F")
        self.assertEqual(empirical.points, ())
        self.assertIn("kinematic", empirical.rejected[0].reason)

    def test_dynamic_and_kinematic_clauses_are_handled_independently(self):
        result = self.normalized(
            "2.47 cP at 20 °C; 1.95 mm²/s at 20 °C"
        )

        self.assertEqual(len(result.points), 1)
        self.assertEqual(len(result.rejected), 1)
        self.assertIn("kinematic", result.rejected[0].reason)

    def test_rejects_non_pure_and_nonordinary_bases(self):
        cases = (
            ("Viscosity of saturated aqueous solution = 1.93 mPa-s at 20 °C", "solution"),
            ("50 wt% mixture: 3.2 cP at 25 °C", "mixture"),
            ("Intrinsic viscosity 1.2 cP at 25 °C", "not ordinary"),
            ("Gas viscosity 12.2 uPa.s at 400 K", "gas or vapor"),
        )
        for text, reason in cases:
            with self.subTest(text=text):
                result = self.normalized(text)
                self.assertEqual(result.points, ())
                self.assertIn(reason, result.rejected[0].reason)

    def test_rejects_predictions_ranges_limits_and_missing_temperatures(self):
        cases = (
            (annotation("2.4 cP at 25 °C", predictive=True), "predictive"),
            (annotation("Estimated viscosity 2.4 cP at 25 °C"), "predictive"),
            (annotation("1.2-1.4 cP at 25 °C"), "range"),
            (annotation("1.2 cP at 20-25 °C"), "range"),
            (annotation("less than 2.4 cP at 25 °C"), "limit"),
            (annotation("2.47 +/- 0.03 cP at 25 °C"), "limit"),
            (annotation("2.47 cP at 25 ± 0.1 °C"), "limit"),
            (annotation("2.47 cP"), "temperature"),
        )
        for record, reason in cases:
            with self.subTest(text=record.text):
                parsed, rejected = parse_pubchem_viscosity_annotation(record)
                self.assertEqual(parsed, ())
                self.assertIn(reason, rejected[0].reason)

    def test_rejects_nonphysical_values(self):
        zero = self.normalized("0 cP at 20 °C")
        impossible_temperature = self.normalized("2.4 cP at -500 °C")

        self.assertIn("nonpositive", zero.rejected[0].reason)
        self.assertIn("nonphysical temperature", impossible_temperature.rejected[0].reason)


class PubChemViscosityFetcherTests(unittest.TestCase):
    def test_fetches_by_name_and_preserves_rejections(self):
        cid_payload = {"IdentifierList": {"CID": [679]}}
        viscosity_payload = {
            "Record": {
                "TOCHeading": "Viscosity",
                "Information": [
                    {
                        "Value": {
                            "StringWithMarkup": [
                                {"String": "2.47 cP at 20 °C"},
                                {"String": "1.95 mm²/s at 20 °C"},
                            ],
                        },
                    },
                ],
            },
        }
        opener = Mock(side_effect=(
            FakeResponse(cid_payload),
            FakeResponse(viscosity_payload),
        ))

        result = PubChemViscosityFetcher(opener=opener).fetch("dimethyl sulfoxide")

        self.assertIsNotNone(result)
        self.assertEqual(result.cid, 679)
        self.assertEqual(len(result.annotations), 2)
        self.assertEqual(len(result.points), 1)
        self.assertEqual(len(result.rejected), 1)
        requests = [call.args[0] for call in opener.call_args_list]
        self.assertIn("dimethyl%20sulfoxide", requests[0].full_url)
        self.assertTrue(requests[1].full_url.endswith("?heading=Viscosity"))
        self.assertEqual(requests[1].get_header("User-agent"), "PFD-Editor/1.0")

    def test_numeric_identifier_skips_cid_lookup(self):
        opener = Mock(return_value=FakeResponse({"Record": {}}))

        result = PubChemViscosityFetcher(opener=opener).fetch(679)

        self.assertIsNotNone(result)
        self.assertEqual(result.cid, 679)
        self.assertEqual(opener.call_count, 1)

    def test_not_found_returns_none_and_transient_failure_is_distinct(self):
        not_found = urllib.error.HTTPError("url", 404, "not found", None, None)
        unavailable = urllib.error.URLError("temporarily unavailable")

        self.assertIsNone(PubChemViscosityFetcher(opener=Mock(side_effect=not_found)).fetch(679))
        with self.assertRaises(PubChemViscosityFetchError):
            PubChemViscosityFetcher(opener=Mock(side_effect=unavailable)).fetch(679)


if __name__ == "__main__":
    unittest.main()
