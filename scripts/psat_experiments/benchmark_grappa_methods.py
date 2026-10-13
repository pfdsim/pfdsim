"""Compare four classical Psat configurations on saved GRAPPA benchmark points.

Use --baseline-dir from benchmark_grappa.py and a new --output-dir. Every
reference temperature/pressure and GRAPPA prediction is reused verbatim, so
there is no resampling or neural-network rerun. The primary matched comparison
keeps complete source/compound curves evaluable by all five methods; pairwise
complete-curve comparisons retain more coverage. All failures remain recorded.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

import CoolProp.CoolProp as CP

from benchmark_grappa import REPO_ROOT, SOURCES, metrics, positive, write_csv

sys.path.insert(0, str(REPO_ROOT))
import nannoolal_method as nm
from perry_properties import PerryPropertyLibrary
from property_resolution.vapor_pressure_adapter import (
    _ambrose_walton_ln_pressure,
    _ambrose_walton_omega_from_pressure,
)

METHODS = (
    "GRAPPA",
    "Nannoolal",
    "Nannoolal, real Tb",
    "AW, real Tb, estimated criticals",
    "AW, real criticals",
)
P_ATM_PA = 101325.0
P_ATM_BAR = P_ATM_PA / 100000.0
ERROR_FIELDS = ("predicted_pressure_Pa", "ln_error", "signed_percent_error", "absolute_percent_error")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class ReferenceProperties:
    """Source-consistent Tb and independently available reference criticals."""

    def __init__(self):
        self.perry = json.loads((REPO_ROOT / "data/perry_properties.json").read_text())["chemicals"]
        self.table = json.loads((REPO_ROOT / "data/perry_table_2_10_vapor_pressure.json").read_text())["chemicals"]
        self.library = PerryPropertyLibrary()
        self.cp_by_cas = {}
        for fluid in sorted(CP.get_global_param_string("FluidsList").split(",")):
            if CP.get_fluid_param_string(fluid, "pure").lower() == "true":
                cas = CP.get_fluid_param_string(fluid, "CAS").strip()
                self.cp_by_cas.setdefault(cas, fluid)

    @lru_cache(maxsize=None)
    def get(self, source, cas, name):
        result = {"Tb_K": None, "Tb_status": "unavailable", "Tc_K": None,
                  "Pc_Pa": None, "omega": None, "critical_source": None}
        if source == "CoolProp HEOS":
            state = CP.AbstractState("HEOS", name)
            result.update(Tc_K=state.T_critical(), Pc_Pa=state.p_critical(),
                          omega=state.acentric_factor(), critical_source=f"CoolProp HEOS::{name}")
            try:
                state.update(CP.PQ_INPUTS, P_ATM_PA, 0.0)
                tb = float(state.T())
                if not state.Ttriple() <= tb < state.T_critical():
                    raise ValueError("1-atm saturation is outside the liquid-vapor domain")
                result.update(Tb_K=tb, Tb_status="source HEOS saturation at 1 atm",
                              Tb_reference_pressure_Pa=float(state.p()))
            except (ValueError, RuntimeError) as error:
                result["Tb_status"] = f"no physical source Tb: {error}"
            return result

        if source == "Perry 2-8":
            boiling = self.library.normal_boiling_point_K(cas)
            if boiling is not None:
                tb = float(boiling.value)
                pressure = self.library.vapor_pressure_bar(cas, tb)
                if pressure is None or abs(pressure.value / P_ATM_BAR - 1) > 1e-10:
                    raise ValueError(f"Source Tb consistency check failed: {cas}")
                result.update(Tb_K=tb, Tb_status="root of source Perry 2-8 correlation at 1 atm",
                              Tb_reference_pressure_Pa=pressure.value * 100000)
            else:
                result["Tb_status"] = "source correlation does not bracket 1 atm in its declared range"
        elif source == "Perry 2-10":
            row = self.table[cas]
            tb = positive(row.get("Tb_K"))
            anchors = [p for p in row.get("vapor_pressure", []) if p.get("P_mmHg") == 760]
            if tb and anchors and all(abs(float(p["T_K"]) - tb) < 1e-8 for p in anchors):
                pressure = float(anchors[0]["P_bar"]) * 100000
                if abs(pressure / P_ATM_PA - 1) > 1e-10:
                    raise ValueError(f"Perry 2-10 760-mmHg pressure mismatch: {cas}")
                result.update(Tb_K=tb, Tb_status="source table 760-mmHg point",
                              Tb_reference_pressure_Pa=pressure)
            else:
                result["Tb_status"] = "source table has no consistent 760-mmHg anchor"
        else:
            raise ValueError(f"Unknown source: {source}")

        tm = positive(self.table.get(cas, {}).get("Tm_K"))
        if result["Tb_K"] is not None and tm and result["Tb_K"] < tm:
            result.update(Tb_K=None, Tb_status="source Tb is below listed melting temperature")

        critical = self.perry.get(cas, {}).get("critical_constants", {})
        tc, pc, omega = positive(critical.get("Tc_K")), positive(critical.get("Pc_MPa")), finite(critical.get("omega"))
        if tc is not None and pc is not None and omega is not None:
            result.update(Tc_K=tc, Pc_Pa=pc * 1e6, omega=omega, critical_source="Perry 2-106")
        elif cas in self.cp_by_cas:
            fluid = self.cp_by_cas[cas]
            state = CP.AbstractState("HEOS", fluid)
            result.update(Tc_K=state.T_critical(), Pc_Pa=state.p_critical(),
                          omega=state.acentric_factor(), critical_source=f"CoolProp HEOS::{fluid}")
        return result


@lru_cache(maxsize=None)
def nannoolal_curve(smiles, tb=None):
    return nm.estimate_psat(smiles, tb=tb)


@lru_cache(maxsize=None)
def nannoolal_criticals(smiles, tb):
    return nm.estimate(smiles, tb=tb)  # Preserve PFDSim's default local refits.


def build_model(method, smiles, reference):
    tb = reference["Tb_K"]
    if method in (METHODS[1], METHODS[2]):
        if method == METHODS[2] and tb is None:
            raise ValueError(reference["Tb_status"])
        model = nannoolal_curve(smiles, tb if method == METHODS[2] else None)
        if model.db is None or positive(model.tb_K) is None:
            raise ValueError("Nannoolal unavailable: " + "; ".join(model.warnings))
        metadata = {"model_Tb_K": model.tb_K, "dB": model.db,
                    "alcohol_correction": model.alcohol_correction,
                    "warnings": "; ".join(model.warnings)}
        if tb is not None:
            anchor = model.psat_kPa(tb)
            if positive(anchor) is not None:
                metadata["predicted_pressure_at_source_Tb_Pa"] = anchor * 1000
                metadata["source_Tb_pressure_residual_percent"] = 100 * (anchor * 1000 / P_ATM_PA - 1)

        def predict_nannoolal(t):
            pressure = model.psat_kPa(t)
            return pressure * 1000 if pressure is not None else None

        return predict_nannoolal, metadata

    if method == METHODS[3]:
        if tb is None:
            raise ValueError(reference["Tb_status"])
        estimate = nannoolal_criticals(smiles, tb)
        tc, pc = positive(estimate.tc_K), positive(estimate.pc_kPa)
        if tc is None or pc is None:
            raise ValueError("Nannoolal criticals unavailable: " + "; ".join(estimate.warnings))
        pc *= 1000.0
        if tb >= tc:
            raise ValueError("source Tb is at/above estimated Tc")
        omega = _ambrose_walton_omega_from_pressure(tb, math.log(P_ATM_BAR), tc, pc / 100000)
        anchor = math.exp(_ambrose_walton_ln_pressure(tb, tc, pc / 100000, omega)) * 100000
        # Near Tr=0.7 the quadratic coefficient is tiny, so the shared
        # closed-form root can lose a few digits through cancellation.
        # A 1e-8 relative pressure tolerance is negligible for this benchmark.
        if abs(anchor / P_ATM_PA - 1) > 1e-8:
            raise ValueError(f"AW source Tb anchoring failed: relative residual {anchor / P_ATM_PA - 1}")
        metadata = {"model_Tc_K": tc, "model_Pc_Pa": pc, "model_omega": omega,
                    "source_Tb_pressure_residual_percent": 100 * (anchor / P_ATM_PA - 1),
                    "warnings": "; ".join(estimate.warnings)}
    elif method == METHODS[4]:
        tc, pc, omega = reference["Tc_K"], reference["Pc_Pa"], reference["omega"]
        if positive(tc) is None or positive(pc) is None or finite(omega) is None:
            raise ValueError("no reference Tc/Pc/omega in Perry 2-106 or exact-CAS CoolProp")
        metadata = {"model_Tc_K": tc, "model_Pc_Pa": pc, "model_omega": omega}
    else:
        raise ValueError(f"Unknown method: {method}")

    def predict(t):
        if t >= tc:
            raise ValueError("sample is at/above model Tc")
        return math.exp(_ambrose_walton_ln_pressure(t, tc, pc / 100000, omega)) * 100000

    return predict, metadata


def scored_row(point, method, prediction=None, failure=None):
    row = {k: v for k, v in point.items() if k not in (*ERROR_FIELDS, "prediction_status", "A", "B", "C")}
    row.update(method=method, prediction_status=failure or "ok")
    if failure:
        return row
    if method == METHODS[0]:
        row.update({k: point[k] for k in ERROR_FIELDS})
        row["prediction_status"] = point["prediction_status"]
    else:
        try:
            if positive(prediction) is None:
                raise ValueError("non-positive/non-finite predicted pressure")
            error = math.log(prediction) - math.log(point["reference_pressure_Pa"])
            signed = 100 * math.expm1(error)
            if not math.isfinite(signed):
                raise ValueError("non-finite percentage error")
            row.update(predicted_pressure_Pa=prediction, ln_error=error,
                       signed_percent_error=signed, absolute_percent_error=abs(signed))
        except (ValueError, OverflowError) as error:
            row["prediction_status"] = str(error)
    return row


def summarize(rows, compound_rows):
    result = {}
    for source in SOURCES:
        selected = [r for r in rows if r["source"] == source]
        cases = {r["cas"] for r in selected}
        complete = {method: {r["cas"] for r in compound_rows if r["source"] == source and r["method"] == method and r["complete_curve"]} for method in METHODS}
        common = set.intersection(*(complete[m] for m in METHODS))
        per_method = {}
        matched = {}
        for method in METHODS:
            method_rows = [r for r in selected if r["method"] == method]
            per_method[method] = {
                "all_reference_points": metrics(method_rows),
                "complete_curves": metrics([r for r in method_rows if r["cas"] in complete[method]]),
                "complete_curve_count": len(complete[method]),
                "incomplete_curve_count": len(cases - complete[method]),
                "failure_reasons": dict(Counter(r["prediction_status"] for r in method_rows if r["prediction_status"] != "ok")),
            }
            matched[method] = metrics([r for r in method_rows if r["cas"] in common])
        pairwise = {}
        for method in METHODS[1:]:
            paired = complete[method] & complete[METHODS[0]]
            grappa = metrics([r for r in selected if r["method"] == METHODS[0] and r["cas"] in paired])
            other = metrics([r for r in selected if r["method"] == method and r["cas"] in paired])
            compound_errors = {(r["method"], r["cas"]): r.get("point_mean_absolute_percent_error") for r in compound_rows if r["source"] == source and r["cas"] in paired}
            pairwise[method] = {"GRAPPA": grappa, "competitor": other,
                                "grappa_compound_wins": sum(compound_errors[METHODS[0],cas] < compound_errors[method,cas] for cas in paired),
                                "competitor_compound_wins": sum(compound_errors[METHODS[0],cas] > compound_errors[method,cas] for cas in paired)}
        strata = {}
        for label, predicate in (
            ("P < 1 kPa", lambda r: r["reference_pressure_Pa"] < 1000),
            ("1 <= P < 100 kPa", lambda r: 1000 <= r["reference_pressure_Pa"] < 100000),
            ("P >= 100 kPa", lambda r: r["reference_pressure_Pa"] >= 100000),
            ("Tr <= 0.8", lambda r: r.get("Tr") is not None and r["Tr"] <= 0.8),
            ("Tr > 0.8", lambda r: r.get("Tr") is not None and r["Tr"] > 0.8),
        ):
            strata[label] = {method: metrics([r for r in selected if r["method"] == method and r["cas"] in common and predicate(r)]) for method in METHODS}
        result[source] = {"reference_compounds": len(cases), "reference_points": len(selected) // len(METHODS),
                          "per_method": per_method, "common_complete_curves": matched,
                          "pairwise_complete_curves": pairwise, "common_point_strata": strata}
    return result


def render_report(summary):
    lines = ["# GRAPPA versus classical vapor-pressure methods", "", summary["protocol"], "",
             "Compound error is mean absolute percentage error over that compound's original reference points. The headline is its median across compounds. Percentage errors use reference pressure as denominator.", "",
             "## Common complete curves: all five methods", "",
             "| Source | Method | Compounds | Points | Median compound error % | Mean compound error % | Point p95 error % |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for source, data in summary["sources"].items():
        for method, m in data["common_complete_curves"].items():
            if "compound_median_absolute_percent_error" in m:
                lines.append(f"| {source} | {method} | {m['compounds']} | {m['points']} | {m['compound_median_absolute_percent_error']:.3f} | {m['compound_mean_absolute_percent_error']:.3f} | {m['point_p95_absolute_percent_error']:.3f} |")
    lines += ["", "## Pairwise complete-curve comparisons", "",
              "| Source | Competitor | Compounds | Points | GRAPPA median % | Competitor median % | GRAPPA mean % | Competitor mean % | GRAPPA compound wins |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for source, data in summary["sources"].items():
        for method, pair in data["pairwise_complete_curves"].items():
            g, m = pair["GRAPPA"], pair["competitor"]
            if "compound_median_absolute_percent_error" in m:
                lines.append(f"| {source} | {method} | {m['compounds']} | {m['points']} | {g['compound_median_absolute_percent_error']:.3f} | {m['compound_median_absolute_percent_error']:.3f} | {g['compound_mean_absolute_percent_error']:.3f} | {m['compound_mean_absolute_percent_error']:.3f} | {pair['grappa_compound_wins']} |")
    lines += ["", "## Coverage", "",
              "| Source | Method | Complete curves | Incomplete/unavailable curves | Failed/unavailable points |",
              "|---|---|---:|---:|---:|"]
    for source, data in summary["sources"].items():
        for method, m in data["per_method"].items():
            lines.append(f"| {source} | {method} | {m['complete_curve_count']} | {m['incomplete_curve_count']} | {m['all_reference_points']['prediction_failures']} |")
    lines += ["", "## Configuration and limitations", "",
              "- Nannoolal uses PFDSim's existing implementation and its default local refits/extensions, including the published alcohol bowing correction. Those extensions are not a pristine reproduction of the original papers; some were fitted to reference properties.",
              "- Real Tb comes only from the source itself: full-range Perry 2-8 inversion, Perry 2-10's 760-mmHg point, or physical CoolProp HEOS saturation at 101325 Pa. No extrapolation beyond a source's declared domain is used to obtain Tb.",
              "- Estimated AW criticals are Nannoolal Tc/Pc estimated using that real Tb. AW omega is solved with PFDSim's existing quadratic inversion, selecting its standard plausible root (-0.5 to 2), to reproduce 1 atm at that Tb.",
              "- Real AW Tc/Pc/omega are from HEOS for CoolProp, and Perry 2-106 for Perry sources, with exact-CAS HEOS fallback if Perry lacks them. Missing properties are not estimated in this configuration.",
              "- Temperatures at/above an AW model's Tc are recorded as failures, never clamped to Pc. Matched comparisons keep whole original curves, rather than dropping difficult individual temperatures or resampling.",
              "- Nannoolal is evaluated on the requested full reference grids, including temperatures beyond its stated approximate upper range Tr=0.75--0.8. Matched Tr<=0.8 and Tr>0.8 diagnostics are saved separately in summary.json.",
              "- The existing Nannoolal alcohol correction is nonzero at supplied Tb, so the real-Tb configuration does not exactly reproduce 1 atm there for mono-functional alcohols. Anchor residuals and warnings are saved in model_parameters.csv; the production method is unchanged.",
              "- The GRAPPA model's training overlap with the references is unknown. Reference disagreements and large outliers from the baseline remain included.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    baseline = args.baseline_dir.resolve()
    original_summary = json.loads((baseline / "summary.json").read_text())
    for name, expected in original_summary["source_file_sha256"].items():
        if file_hash(REPO_ROOT / "data" / name) != expected:
            raise ValueError(f"Reference file changed since baseline: {name}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    reference = ReferenceProperties()
    points = []
    with (baseline / "points.csv").open() as handle:
        for index, raw in enumerate(csv.DictReader(handle)):
            row = dict(raw, point_id=index)
            for key in ("temperature_K", "reference_pressure_Pa", "Tc_K", "Tm_or_Ttriple_K", "Tr", "A", "B", "C", *ERROR_FIELDS):
                row[key] = float(raw[key]) if raw.get(key) else None
            points.append(row)
    grouped = defaultdict(list)
    for row in points:
        grouped[row["source"], row["cas"]].append(row)
    scored, compounds, parameters = [], [], []
    for index, ((source, cas), case_points) in enumerate(grouped.items()):
        first = case_points[0]
        info = reference.get(source, cas, first["name"])
        for method in METHODS:
            predict, metadata, failure = None, {}, None
            if method != METHODS[0]:
                try:
                    predict, metadata = build_model(method, first["smiles"], info)
                except (ValueError, OverflowError, ZeroDivisionError) as error:
                    failure = str(error)
            parameters.append({"source": source, "cas": cas, "name": first["name"],
                               "method": method, "setup_status": failure or "ok", **info, **metadata})
            method_rows = []
            for point in case_points:
                if failure or method == METHODS[0]:
                    row = scored_row(point, method, failure=failure)
                else:
                    try:
                        row = scored_row(point, method, predict(point["temperature_K"]))
                    except (ValueError, OverflowError, ZeroDivisionError) as error:
                        row = scored_row(point, method, failure=str(error))
                method_rows.append(row)
            complete = all(r["prediction_status"] == "ok" for r in method_rows)
            compounds.append({"source": source, "cas": cas, "name": first["name"], "method": method,
                              "complete_curve": complete, **metrics(method_rows)})
            scored.extend(method_rows)
        if (index + 1) % 200 == 0:
            print(f"Compared {index + 1}/{len(grouped)} source/compound curves", flush=True)
    sources = summarize(scored, compounds)
    for source in SOURCES:
        actual = sources[source]["per_method"][METHODS[0]]["all_reference_points"]
        for key, expected in original_summary["sources"][source].items():
            if key in actual and not math.isclose(actual[key], expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Baseline metric changed: {source} {key}")
    summary = {
        "protocol": "Reuse every saved GRAPPA benchmark reference temperature/pressure and GRAPPA prediction exactly. No resampling. Ten uniform temperature points per continuous source curve; original eligible Perry 2-10 rows. Main comparisons use common complete curves, with pairwise complete-curve comparisons and full coverage/failure diagnostics.",
        "methods": list(METHODS), "sources": sources,
        "baseline_points": len(points), "method_point_evaluations": len(scored),
        "elapsed_seconds_excluding_imports": time.perf_counter() - started,
        "versions": {name: version(name) for name in ("numpy", "rdkit", "chemicals", "CoolProp")},
        "baseline_files_sha256": {name: file_hash(baseline / name) for name in ("points.csv", "summary.json")},
        "implementation_files_sha256": {name: file_hash(REPO_ROOT / name) for name in ("nannoolal_method.py", "perry_properties.py", "property_resolution/vapor_pressure_adapter.py", "scripts/psat_experiments/benchmark_grappa.py", "scripts/psat_experiments/benchmark_grappa_methods.py")},
        "source_file_sha256": original_summary["source_file_sha256"],
        "baseline_dir": str(baseline), "argv": sys.argv,
    }
    write_csv(output / "points.csv", scored)
    write_csv(output / "compounds.csv", compounds)
    write_csv(output / "model_parameters.csv", parameters)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(render_report(summary))
    print(json.dumps({"elapsed_seconds": summary["elapsed_seconds_excluding_imports"], "output_dir": str(output),
                      "common_complete_curves": {s: d["common_complete_curves"] for s, d in sources.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
