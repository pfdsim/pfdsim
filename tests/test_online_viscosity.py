import json
import math
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import Mock, patch

from property_resolution.online_viscosity import (
    PubChemViscosityAnnotation,
    PubChemViscosityFetchError,
    PubChemViscosityFetcher,
    PubChemViscosityResult,
    extract_pubchem_viscosity_annotations,
    normalize_pubchem_viscosity_annotations,
    parse_pubchem_viscosity_annotation,
)
from property_resolution.common import PropertyResolutionError, PropertyResolutionResult
from property_resolution.resolver import PropertyResolver
from property_resolution.runtime_cache import cache_key_metadata


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


def viscosity_payload(*texts):
    annotations = tuple(annotation(text) for text in texts)
    normalized = normalize_pubchem_viscosity_annotations(annotations)
    return type("Payload", (), {
        "cid": 123,
        "points": normalized.points,
        "kinematic_points": normalized.kinematic_points,
        "rejected": normalized.rejected,
    })()


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

    def test_mojibake_kinematic_unit_is_preserved_for_density_conversion(self):
        result = self.normalized("1.95 mmÂ²/s at 20Â °C")

        self.assertEqual(result.points, ())
        self.assertEqual(len(result.kinematic_points), 1)
        self.assertAlmostEqual(
            result.kinematic_points[0].kinematic_viscosity_m2_s,
            1.95e-6,
        )

        empirical = self.normalized("45 Saybolt Universal Seconds at 100 °F")
        self.assertEqual(empirical.points, ())
        self.assertIn("kinematic", empirical.rejected[0].reason)

    def test_normalizes_kinematic_unit_families(self):
        cases = (
            ("1.95 cSt at 20 °C", 1.95e-6),
            ("0.0195 stokes at 20 °C", 1.95e-6),
            ("1.0 sq ft/sec at 20 °C", 0.09290304),
            ("1.0 in²/s at 20 °C", 0.00064516),
            ("1.0 mm2/sec at 300 K", 1.0e-6),
            ("1.0 mm²/second at 300 K", 1.0e-6),
            ("1.0 sq mm/seconds at 300 K", 1.0e-6),
            ("1.0 cm^2/secs at 300 K", 1.0e-4),
            ("1.0 m2/second at 300 K", 1.0),
            ("1.0 ft2/seconds at 300 K", 0.09290304),
            ("1.0 sq in/sec at 300 K", 0.00064516),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                result = self.normalized(text)
                self.assertEqual(len(result.kinematic_points), 1)
                self.assertAlmostEqual(
                    result.kinematic_points[0].kinematic_viscosity_m2_s,
                    expected,
                )

    def test_dynamic_and_kinematic_clauses_are_handled_independently(self):
        result = self.normalized(
            "2.47 cP at 20 °C; 1.95 mm²/s at 20 °C"
        )

        self.assertEqual(len(result.points), 1)
        self.assertEqual(len(result.kinematic_points), 1)
        self.assertEqual(result.rejected, ())

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
        self.assertEqual(len(result.kinematic_points), 1)
        self.assertEqual(len(result.rejected), 0)
        self.assertEqual(
            type(result).from_dict(json.loads(json.dumps(result.to_dict()))),
            result,
        )
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


class OnlineViscosityResolutionTests(unittest.TestCase):
    @staticmethod
    def props(smiles="CCCCCC"):
        return {
            "name": "fixture",
            "smiles": smiles,
            "MW": 100.0,
            "Tb": 500.0,
            "Tc": 600.0,
        }

    @staticmethod
    def curve_text(temperature, *, intercept=-9.0, slope=1800.0, scale=1.0):
        viscosity_Pa_s = math.exp(intercept + slope / temperature) * scale
        return f"{viscosity_Pa_s * 1000.0:.12g} cP at {temperature:g} K"

    @staticmethod
    def fake_result(method, quality=0.5, value=0.001):
        return PropertyResolutionResult(value, "estimated", method, quality, "fixture")

    def resolve(self, payload, temperature, *, props=None, allow_online=True, hsu=None):
        resolver = PropertyResolver()
        with (
            patch.object(resolver, "_coolprop_viscosity", return_value=None),
            patch.object(resolver, "_get_perry_evaluation", return_value=None),
            patch.object(resolver, "_get_perry_viscosity_bounded", return_value=None),
            patch.object(resolver, "_fetch_viscosity_online", return_value=payload),
            patch.object(resolver, "_hsu_liquid_viscosity", return_value=hsu),
        ):
            return resolver.resolve_viscosity(
                "viscosity fixture",
                temperature,
                "liquid",
                props=props or self.props(),
                allow_online=allow_online,
            )

    def test_two_points_fit_arrhenius_curve_before_hsu(self):
        payload = viscosity_payload(
            self.curve_text(300.0),
            self.curve_text(350.0),
        )
        hsu = self.fake_result("hsu_liquid_viscosity")

        result = self.resolve(payload, 325.0, hsu=hsu)

        self.assertEqual(result.method, "pubchem_liquid_viscosity_arrhenius_fit")
        self.assertAlmostEqual(result.value, math.exp(-9.0 + 1800.0 / 325.0), places=10)
        self.assertAlmostEqual(result.quality, 0.87)
        self.assertIn("interpolation", result.notes)

    def test_online_fit_extrapolation_quality_and_cutoff(self):
        payload = viscosity_payload(
            self.curve_text(300.0),
            self.curve_text(350.0),
        )
        expected = (
            (360.0, 0.86),
            (375.0, 0.83),
            (400.0, 0.78),
            (450.0, 0.69),
        )
        for temperature, quality in expected:
            with self.subTest(temperature=temperature):
                result = self.resolve(payload, temperature)
                self.assertEqual(
                    result.method,
                    "pubchem_liquid_viscosity_arrhenius_fit",
                )
                self.assertAlmostEqual(result.quality, quality)

        hsu = self.fake_result("hsu_liquid_viscosity")
        result = self.resolve(payload, 450.0001, hsu=hsu)
        self.assertEqual(result.method, "hsu_liquid_viscosity")

    def test_fit_span_boundaries_and_additive_quality_penalty(self):
        for span, penalty in ((10.0, 0.04), (19.999, 0.04), (20.0, 0.0)):
            with self.subTest(span=span):
                payload = viscosity_payload(
                    self.curve_text(300.0), self.curve_text(300.0 + span),
                )
                result = self.resolve(payload, 325.0 + span)
                self.assertEqual(result.method, "pubchem_liquid_viscosity_arrhenius_fit")
                self.assertAlmostEqual(result.quality, 0.87 - 0.04 - penalty)
                self.assertIn(f"span penalty {penalty:.2f}", result.notes)

    def test_short_span_uses_combined_anchor_even_when_slope_is_nonpositive(self):
        resolver = PropertyResolver()
        for span in (0.6, 9.999):
            for values in ((1.1, 1.0), (1.0, 1.1)):
                with self.subTest(span=span, values=values):
                    payload = viscosity_payload(
                        f"{values[0]} cP at 300 K",
                        f"{values[1]} cP at {300.0 + span} K",
                    )
                    with (
                        patch.object(resolver, "_fetch_viscosity_online", return_value=payload),
                        patch.object(resolver, "_nannoolal_liquid_viscosity") as nannoolal,
                    ):
                        resolver._online_liquid_viscosity(
                            "fixture", self.props(), 300.0, "liquid", allow_online=True,
                        )
                    anchor = nannoolal.call_args.kwargs["anchor"]
                    self.assertAlmostEqual(anchor["T_K"], 300.0 + span / 2.0)
                    self.assertAlmostEqual(anchor["mu_Pa_s"], math.sqrt(1.1) * 0.001)
                    self.assertIn("combined 2 points", anchor["notes"])

    def test_span_is_measured_after_robust_outlier_removal(self):
        for span, expected in ((9.0, "nannoolal_anchored_liquid_viscosity"),
                               (15.0, "pubchem_liquid_viscosity_arrhenius_fit")):
            with self.subTest(span=span):
                payload = viscosity_payload(*(
                    self.curve_text(t, scale=100.0 if t == 350.0 else 1.0)
                    for t in (300.0, 302.0, 304.0, 300.0 + span, 350.0)
                ))
                result = self.resolve(payload, 305.0)
                self.assertEqual(result.method, expected)
                self.assertIn("rejected 1 robust-fit outlier", result.notes)
                if span == 15.0:
                    self.assertAlmostEqual(result.quality, 0.89 - 0.04)

    def test_combined_anchor_preserves_density_penalty_and_weights(self):
        resolver = PropertyResolver()
        payload = viscosity_payload("1.0 cP at 300 K", "1.2 cSt at 306 K")
        density = PropertyResolutionResult(10.0, "estimated", "fixture", 0.80, "")
        with (
            patch.object(resolver, "_fetch_viscosity_online", return_value=payload),
            patch.object(resolver, "resolve_liquid_molar_density", return_value=density),
            patch.object(resolver, "_nannoolal_liquid_viscosity") as nannoolal,
        ):
            resolver._online_liquid_viscosity(
                "fixture", self.props(), 303.0, "liquid", allow_online=True,
            )
        anchor = nannoolal.call_args.kwargs["anchor"]
        weight = 0.85 ** 2
        self.assertAlmostEqual(anchor["T_K"], (300.0 + weight * 306.0) / (1.0 + weight))
        self.assertAlmostEqual(
            anchor["mu_Pa_s"],
            math.exp((math.log(0.001) + weight * math.log(0.0012)) / (1.0 + weight)),
        )
        self.assertAlmostEqual(nannoolal.call_args.kwargs["quality"], 0.85 - 0.15)

    def test_robust_fit_rejects_outlier_with_four_or_more_points(self):
        payload = viscosity_payload(*(
            self.curve_text(temperature, scale=10.0 if temperature == 320.0 else 1.0)
            for temperature in (280.0, 300.0, 320.0, 340.0, 360.0)
        ))

        result = self.resolve(payload, 330.0)

        self.assertEqual(result.method, "pubchem_liquid_viscosity_arrhenius_fit")
        self.assertAlmostEqual(result.value, math.exp(-9.0 + 1800.0 / 330.0), places=9)
        self.assertAlmostEqual(result.quality, 0.89)
        self.assertIn("rejected 1 robust-fit outlier", result.notes)

    def test_low_quality_kinematic_is_only_used_when_needed_for_shape(self):
        resolver = PropertyResolver()
        density = PropertyResolutionResult(
            10.0,
            "estimated",
            "density_fixture",
            0.80,
            "mol/dm3",
        )
        one_dynamic = viscosity_payload(
            self.curve_text(300.0),
            f"{math.exp(-9.0 + 1800.0 / 350.0) * 1000.0:.12g} cSt at 350 K",
        )
        two_dynamic = viscosity_payload(
            self.curve_text(300.0),
            self.curve_text(350.0),
            f"{math.exp(-9.0 + 1800.0 / 325.0) * 1000.0:.12g} cSt at 325 K",
        )
        with patch.object(
            resolver,
            "resolve_liquid_molar_density",
            return_value=density,
        ):
            fitted = resolver._online_liquid_viscosity_points(
                "fixture",
                self.props(),
                one_dynamic,
                allow_online=True,
            )
            discarded = resolver._online_liquid_viscosity_points(
                "fixture",
                self.props(),
                two_dynamic,
                allow_online=True,
            )

        self.assertEqual(len(fitted), 2)
        self.assertTrue(any(
            abs(point["density_penalty"] - 0.15) < 1.0e-12
            for point in fitted
        ))
        self.assertEqual(len(discarded), 2)
        self.assertTrue(all(point["density_penalty"] == 0.0 for point in discarded))

        with (
            patch.object(resolver, "_fetch_viscosity_online", return_value=one_dynamic),
            patch.object(resolver, "resolve_liquid_molar_density", return_value=density),
        ):
            result = resolver._online_liquid_viscosity(
                "fixture",
                self.props(),
                325.0,
                "liquid",
                allow_online=True,
            )
        self.assertAlmostEqual(result.quality, 0.87 - 0.15)
        self.assertIn("1 density-converted", result.notes)

    def test_high_quality_kinematic_is_retained_with_two_dynamic_points(self):
        resolver = PropertyResolver()
        density = PropertyResolutionResult(
            10.0,
            "local",
            "density_fixture",
            0.96,
            "mol/dm3",
        )
        payload = viscosity_payload(
            self.curve_text(300.0),
            self.curve_text(350.0),
            f"{math.exp(-9.0 + 1800.0 / 325.0) * 1000.0:.12g} cSt at 325 K",
        )
        with (
            patch.object(resolver, "_fetch_viscosity_online", return_value=payload),
            patch.object(resolver, "resolve_liquid_molar_density", return_value=density),
        ):
            result = resolver._online_liquid_viscosity(
                "fixture",
                self.props(),
                325.0,
                "liquid",
                allow_online=True,
            )

        self.assertAlmostEqual(result.quality, 0.88)
        self.assertIn("3 PubChem", result.notes)

    def test_high_quality_second_temperature_makes_low_quality_third_unnecessary(self):
        resolver = PropertyResolver()
        payload = viscosity_payload(
            self.curve_text(300.0),
            f"{math.exp(-9.0 + 1800.0 / 325.0) * 1000.0:.12g} cSt at 325 K",
            f"{math.exp(-9.0 + 1800.0 / 350.0) * 1000.0:.12g} cSt at 350 K",
        )
        qualities = iter((0.96, 0.80))

        def density(*args, **kwargs):
            quality = next(qualities)
            return PropertyResolutionResult(
                10.0,
                "local" if quality >= 0.95 else "estimated",
                "density_fixture",
                quality,
                "mol/dm3",
            )

        with patch.object(
            resolver,
            "resolve_liquid_molar_density",
            side_effect=density,
        ):
            points = resolver._online_liquid_viscosity_points(
                "fixture",
                self.props(),
                payload,
                allow_online=True,
            )

        self.assertEqual(len(points), 2)
        self.assertTrue(all(point["density_penalty"] == 0.0 for point in points))

    def test_inconsistent_same_temperature_pair_is_not_averaged(self):
        resolver = PropertyResolver()
        points = resolver._consolidate_online_viscosity_points([
            {
                "T_K": 300.0,
                "mu_Pa_s": 0.001,
                "density_quality": 1.0,
                "density_penalty": 0.0,
                "kind": "dynamic",
                "reference": "first",
                "raw": "first",
                "notes": "",
            },
            {
                "T_K": 300.0,
                "mu_Pa_s": 0.003,
                "density_quality": 1.0,
                "density_penalty": 0.0,
                "kind": "dynamic",
                "reference": "second",
                "raw": "second",
                "notes": "",
            },
        ])

        self.assertEqual(points, [])

    def test_single_point_anchors_nannoolal_with_distance_quality_bands(self):
        resolver = PropertyResolver()
        payload = viscosity_payload(self.curve_text(300.0))

        def nannoolal_result(*args, **kwargs):
            return self.fake_result(kwargs["method"], kwargs["quality"])

        with (
            patch.object(resolver, "_fetch_viscosity_online", return_value=payload),
            patch.object(
                resolver,
                "_nannoolal_liquid_viscosity",
                side_effect=nannoolal_result,
            ),
        ):
            for temperature, quality in (
                (325.0, 0.85),
                (350.0, 0.80),
                (375.0, 0.75),
                (400.0, 0.70),
            ):
                with self.subTest(temperature=temperature):
                    result = resolver._online_liquid_viscosity(
                        "fixture",
                        self.props(),
                        temperature,
                        "liquid",
                        allow_online=True,
                    )
                    self.assertEqual(
                        result.method,
                        "nannoolal_anchored_liquid_viscosity",
                    )
                    self.assertAlmostEqual(result.quality, quality)
            self.assertIsNone(resolver._online_liquid_viscosity(
                "fixture",
                self.props(),
                400.0001,
                "liquid",
                allow_online=True,
            ))

    def test_single_dynamic_point_uses_real_anchored_nannoolal_engine(self):
        payload = viscosity_payload("0.80 cP at 300 K")

        result = self.resolve(payload, 320.0)

        self.assertEqual(result.method, "nannoolal_anchored_liquid_viscosity")
        self.assertEqual(result.source, "calculated")
        self.assertAlmostEqual(result.quality, 0.85)
        self.assertGreater(result.value, 0.0)
        self.assertIn("anchored at T=300 K", result.notes)

    def test_single_kinematic_anchor_gets_additive_density_penalty(self):
        resolver = PropertyResolver()
        payload = viscosity_payload("1.0 cSt at 300 K")
        density = PropertyResolutionResult(
            10.0,
            "estimated",
            "density_fixture",
            0.80,
            "mol/dm3",
        )

        def nannoolal_result(*args, **kwargs):
            self.assertAlmostEqual(kwargs["anchor"]["mu_Pa_s"], 0.001)
            return self.fake_result(kwargs["method"], kwargs["quality"])

        with (
            patch.object(resolver, "_fetch_viscosity_online", return_value=payload),
            patch.object(resolver, "resolve_liquid_molar_density", return_value=density),
            patch.object(
                resolver,
                "_nannoolal_liquid_viscosity",
                side_effect=nannoolal_result,
            ),
        ):
            result = resolver._online_liquid_viscosity(
                "fixture",
                self.props(),
                320.0,
                "liquid",
                allow_online=True,
            )

        self.assertAlmostEqual(result.quality, 0.70)

    def test_phenol_fit_is_penalized_and_limited_to_ten_kelvin(self):
        payload = viscosity_payload(
            self.curve_text(300.0),
            self.curve_text(350.0),
        )
        phenol_props = self.props("Oc1ccccc1")

        in_range = self.resolve(payload, 325.0, props=phenol_props)
        at_limit = self.resolve(payload, 360.0, props=phenol_props)
        hsu = self.fake_result("hsu_liquid_viscosity")
        outside = self.resolve(payload, 360.0001, props=phenol_props, hsu=hsu)

        self.assertAlmostEqual(in_range.quality, 0.82)
        self.assertAlmostEqual(at_limit.quality, 0.81)
        self.assertEqual(outside.method, "hsu_liquid_viscosity")

    def test_phenol_cannot_use_anchored_or_predictive_nannoolal(self):
        resolver = PropertyResolver()
        payload = viscosity_payload(self.curve_text(300.0))
        hsu = self.fake_result("hsu_liquid_viscosity")
        with (
            patch.object(resolver, "_coolprop_viscosity", return_value=None),
            patch.object(resolver, "_get_perry_evaluation", return_value=None),
            patch.object(resolver, "_get_perry_viscosity_bounded", return_value=None),
            patch.object(resolver, "_fetch_viscosity_online", return_value=payload),
            patch.object(resolver, "_hsu_liquid_viscosity", return_value=hsu),
            patch.object(
                resolver,
                "_nannoolal_predictive_liquid_viscosity",
                side_effect=AssertionError("phenol reached predictive Nannoolal"),
            ),
        ):
            result = resolver.resolve_viscosity(
                "phenol",
                320.0,
                "liquid",
                props=self.props("Oc1ccccc1"),
            )
        self.assertEqual(result.method, "hsu_liquid_viscosity")

    def test_online_is_skipped_when_disabled(self):
        resolver = PropertyResolver()
        hsu = self.fake_result("hsu_liquid_viscosity")
        with (
            patch.object(resolver, "_coolprop_viscosity", return_value=None),
            patch.object(resolver, "_get_perry_evaluation", return_value=None),
            patch.object(resolver, "_get_perry_viscosity_bounded", return_value=None),
            patch.object(
                resolver,
                "_fetch_viscosity_online",
                side_effect=AssertionError("online lookup attempted"),
            ),
            patch.object(resolver, "_hsu_liquid_viscosity", return_value=hsu),
        ):
            result = resolver.resolve_viscosity(
                "fixture",
                320.0,
                "liquid",
                props=self.props(),
                allow_online=False,
            )
        self.assertEqual(result.method, "hsu_liquid_viscosity")

    def test_hsu_precedes_predictive_nannoolal_and_nannoolal_fills_rejection(self):
        resolver = PropertyResolver()
        order = []

        def hsu(*args, **kwargs):
            order.append("hsu")
            return None

        def nannoolal(*args, **kwargs):
            order.append("nannoolal")
            return self.fake_result("nannoolal_predictive_liquid_viscosity", 0.65)

        with (
            patch.object(resolver, "_coolprop_viscosity", return_value=None),
            patch.object(resolver, "_get_perry_evaluation", return_value=None),
            patch.object(resolver, "_get_perry_viscosity_bounded", return_value=None),
            patch.object(resolver, "_fetch_viscosity_online", return_value=None),
            patch.object(resolver, "_hsu_liquid_viscosity", side_effect=hsu),
            patch.object(
                resolver,
                "_nannoolal_predictive_liquid_viscosity",
                side_effect=nannoolal,
            ),
        ):
            result = resolver.resolve_viscosity(
                "fixture",
                320.0,
                "liquid",
                props=self.props(),
            )

        self.assertEqual(order, ["hsu", "nannoolal"])
        self.assertEqual(result.method, "nannoolal_predictive_liquid_viscosity")
        self.assertAlmostEqual(result.quality, 0.65)

    def test_real_predictive_nannoolal_fills_hsu_structural_rejection(self):
        resolver = PropertyResolver()
        with (
            patch.object(resolver, "_coolprop_viscosity", return_value=None),
            patch.object(resolver, "_get_perry_evaluation", return_value=None),
            patch.object(resolver, "_get_perry_viscosity_bounded", return_value=None),
        ):
            result = resolver.resolve_viscosity(
                "tetrahydrofuran-like",
                300.0,
                "liquid",
                props={
                    "smiles": "C1CCOC1",
                    "Tb": 339.0,
                    "Tc": 540.0,
                    "Pc": 52.0,
                },
                allow_online=False,
            )

        self.assertEqual(result.method, "nannoolal_predictive_liquid_viscosity")
        self.assertEqual(result.source, "estimated")
        self.assertAlmostEqual(result.quality, 0.65)
        self.assertGreater(result.value, 0.0)

    def test_nannoolal_temperature_gate_uses_tc_provenance_for_both_modes(self):
        resolver = PropertyResolver()
        for source, quality, fraction in (
            ("provided", 1.0, 0.8), ("local", 0.90, 0.8),
            ("local", 0.89, 0.75), ("estimated", 0.99, 0.75),
        ):
            props = self.props("C1CCOC1")
            props["property_sources"] = {"Tc": {"source": source, "quality": quality}}
            limit = fraction * props["Tc"]
            for anchor in (None, {"T_K": limit - 10, "mu_Pa_s": 0.001, "kind": "dynamic"}):
                for temperature in (limit, limit + 0.001):
                    with self.subTest(source=source, quality=quality, anchor=anchor, T=temperature):
                        result = resolver._nannoolal_liquid_viscosity(
                            "fixture", props, temperature, "liquid", allow_online=False,
                            anchor=anchor, quality=0.65, method="fixture",
                        )
                        if temperature == limit:
                            self.assertIsNotNone(result)
                        else:
                            self.assertIsNone(result)

    def test_nannoolal_resolves_missing_tc_and_refuses_unavailable_or_invalid_tc(self):
        resolver = PropertyResolver()
        props = self.props("C1CCOC1")
        props.pop("Tc")
        for tc in (600.0, None, 0.0, -600.0, float("nan"), float("inf")):
            with self.subTest(tc=tc), patch.object(
                resolver, "resolve_critical_properties",
                return_value={"Tc": self.fake_result("estimated_tc", value=tc)},
            ) as critical:
                result = resolver._nannoolal_predictive_liquid_viscosity(
                    "fixture", props, 450.0, "liquid", allow_online=False,
                )
                self.assertEqual(result is not None, tc == 600.0)
                self.assertFalse(critical.call_args.kwargs["allow_online"])

    def test_offline_setting_reaches_real_hsu_and_vapor_dependencies(self):
        for phase in ("liquid", "vapor"):
            for smiles in (None, "CCCCCC"):
                with self.subTest(phase=phase, smiles=smiles):
                    resolver = PropertyResolver()
                    props = self.props()
                    if smiles is None:
                        props.pop("smiles")
                    props.pop("Tc")
                    original = dict(props)
                    with (
                        patch.object(resolver, "_coolprop_viscosity", return_value=None),
                        patch.object(resolver, "_get_perry_evaluation", return_value=None),
                        patch.object(resolver, "_get_perry_viscosity_bounded", return_value=None),
                        patch.object(resolver, "_resolve_smiles_result", return_value=None) as structure,
                        patch.object(resolver, "resolve_critical_properties", return_value={}) as critical,
                        patch.object(resolver, "_fetch_viscosity_online") as online,
                    ):
                        with self.assertRaises(PropertyResolutionError):
                            resolver.resolve_viscosity(
                                "fixture", 300.0, phase, props, allow_online=False,
                            )
                    calls = structure.call_args_list + critical.call_args_list
                    self.assertTrue(calls)
                    self.assertTrue(all(call.kwargs["allow_online"] is False for call in calls))
                    online.assert_not_called()
                    self.assertEqual(props, original)

    def test_pubchem_viscosity_cache_roundtrip_and_metadata(self):
        payload = viscosity_payload(
            self.curve_text(300.0),
            "1.0 cSt at 320 K",
        )
        concrete = PubChemViscosityResult(
            identifier="123",
            cid=123,
            annotations=(),
            points=payload.points,
            kinematic_points=payload.kinematic_points,
            rejected=(),
        )
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.CACHE_DIR = Path(directory)
            with (
                patch.object(resolver, "_get_pubchem_cid", return_value=123),
                patch(
                    "property_resolution.viscosity.PubChemViscosityFetcher.fetch",
                    return_value=concrete,
                ) as fetch,
            ):
                first = resolver._fetch_pubchem_viscosity("fixture")
                second = resolver._fetch_pubchem_viscosity("fixture")

        self.assertEqual(first, second)
        self.assertEqual(fetch.call_count, 1)
        metadata = cache_key_metadata(
            "viscosity_pubchem_v2_fixture",
            concrete.to_dict(),
        )
        self.assertEqual(metadata["cache_family"], "viscosity_pubchem")
        self.assertEqual(metadata["provider"], "pubchem")
        self.assertEqual(metadata["contract_version"], 2)


if __name__ == "__main__":
    unittest.main()
