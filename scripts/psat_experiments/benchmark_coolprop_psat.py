import csv
import json
import math
from dataclasses import dataclass
from statistics import mean, median

import CoolProp.CoolProp as CP
import numpy as np
from CoolProp import AbstractState
from CoolProp.CoolProp import QT_INPUTS, iP, iT

from benchmark_local_tables_perry import (
    perry_ln_pressure,
    perry_row,
    perry_slope,
    relative_percent,
)
from benchmark_perry_aw import percentile
from perry_properties import PerryPropertyLibrary
from property_resolution import PsatCanonicalizationAdapter, PsatCanonicalizer
from vapor_pressure_tables import DATA_PATH, get_vapor_pressure_table_library


TABLE_OUTPUT_PATH = "/tmp/coolprop_local_table_benchmark.csv"
PERRY_OUTPUT_PATH = "/tmp/coolprop_perry_benchmark.csv"
WATER_OUTPUT_PATH = "/tmp/coolprop_water_backend_benchmark.csv"
CANONICAL_OUTPUT_PATH = "/tmp/coolprop_canonical_fit_benchmark.csv"


@dataclass
class CoolPropSaturationCurve:
    fluid: str
    backend: str

    def __post_init__(self):
        self.state = AbstractState(self.backend, self.fluid)
        qualified = f"{self.backend}::{self.fluid}"
        self.T_triple = float(CP.PropsSI("Ttriple", qualified))
        self.T_critical = float(CP.PropsSI("Tcrit", qualified))
        self.P_critical_bar = float(CP.PropsSI("pcrit", qualified)) / 100000.0
        self.T_upper = self.T_critical - max(1.0e-7 * self.T_critical, 1.0e-5)

    @property
    def qualified_name(self):
        return f"{self.backend}::{self.fluid}"

    def ln_pressure(self, temperature):
        self.state.update(QT_INPUTS, 0.0, float(temperature))
        pressure = self.state.p() / 100000.0
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise ValueError("CoolProp returned invalid saturation pressure")
        return math.log(pressure)

    def slope(self, temperature):
        temperature = float(temperature)
        if self.backend == "HEOS":
            self.state.update(QT_INPUTS, 0.0, temperature)
            pressure = self.state.p()
            derivative = self.state.first_saturation_deriv(iP, iT)
            value = derivative / pressure
            if math.isfinite(value) and value > 0.0:
                return float(value)
        return self._numerical_slope(temperature)

    def _numerical_slope(self, temperature):
        width = self.T_upper - self.T_triple
        step = min(max(1.0e-5 * temperature, 1.0e-3), width / 1000.0)
        if temperature - step >= self.T_triple and temperature + step <= self.T_upper:
            return (
                self.ln_pressure(temperature + step)
                - self.ln_pressure(temperature - step)
            ) / (2.0 * step)
        if temperature + 2.0 * step <= self.T_upper:
            return (
                -3.0 * self.ln_pressure(temperature)
                + 4.0 * self.ln_pressure(temperature + step)
                - self.ln_pressure(temperature + 2.0 * step)
            ) / (2.0 * step)
        if temperature - 2.0 * step >= self.T_triple:
            return (
                3.0 * self.ln_pressure(temperature)
                - 4.0 * self.ln_pressure(temperature - step)
                + self.ln_pressure(temperature - 2.0 * step)
            ) / (2.0 * step)
        raise ValueError("Cannot evaluate CoolProp saturation derivative")


def coolprop_fluids_by_cas():
    matches = {}
    duplicates = {}
    for fluid in CP.get_global_param_string("FluidsList").split(","):
        try:
            cas = CP.get_fluid_param_string(fluid, "CAS").strip()
        except Exception:
            continue
        if not cas:
            continue
        if cas in matches:
            duplicates.setdefault(cas, [matches[cas]]).append(fluid)
            continue
        matches[cas] = fluid
    return matches, duplicates


def selected_curve(fluid, cas):
    if cas == "7732-18-5":
        return CoolPropSaturationCurve("Water", "IF97")
    return CoolPropSaturationCurve(fluid, "HEOS")


def metric_summary(errors):
    absolute = [abs(value) for value in errors]
    return {
        "count": len(errors),
        "mard": mean(absolute),
        "median": median(absolute),
        "p95": percentile(absolute, 0.95),
        "max": max(absolute),
        "bias": mean(errors),
    }


def comparison_points(low, high, count=401):
    if high <= low:
        return []
    return [float(value) for value in np.linspace(low, high, count)]


def table_comparisons():
    raw_tables = json.loads(DATA_PATH.read_text()).get("tables", {})
    library = get_vapor_pressure_table_library()
    fluids_by_cas, _duplicates = coolprop_fluids_by_cas()
    reports = []
    detail_rows = []
    unsupported = []

    for table in library.tables.values():
        raw = raw_tables.get(table.key, {})
        cas = raw.get("CAS") or raw.get("cas")
        fluid = fluids_by_cas.get(cas)
        if fluid is None:
            unsupported.append((table.key, table.name, cas))
            continue
        curve = selected_curve(fluid, cas)
        low = max(table.T_min, curve.T_triple)
        high = min(table.T_max, curve.T_upper)
        dense_rows = []
        for temperature in comparison_points(low, high):
            table_ln = float(table._ln_pressure_spline(temperature))
            cp_ln = curve.ln_pressure(temperature)
            table_slope = float(table._ln_pressure_spline.derivative()(temperature))
            cp_slope = curve.slope(temperature)
            row = {
                "key": table.key,
                "name": table.name,
                "cas": cas,
                "backend": curve.backend,
                "fluid": curve.fluid,
                "kind": "dense_spline",
                "temperature_K": temperature,
                "pressure_error_percent": 100.0 * math.expm1(table_ln - cp_ln),
                "slope_error_percent": relative_percent(table_slope, cp_slope),
            }
            dense_rows.append(row)
            detail_rows.append(row)

        source_rows = []
        for temperature, pressure in zip(table.temperatures, table.pressures_bar):
            if not low <= temperature <= high:
                continue
            cp_ln = curve.ln_pressure(temperature)
            row = {
                "key": table.key,
                "name": table.name,
                "cas": cas,
                "backend": curve.backend,
                "fluid": curve.fluid,
                "kind": "source_row",
                "temperature_K": temperature,
                "pressure_error_percent": 100.0 * math.expm1(math.log(pressure) - cp_ln),
                "slope_error_percent": "",
            }
            source_rows.append(row)
            detail_rows.append(row)

        pressure_errors = [row["pressure_error_percent"] for row in dense_rows]
        slope_errors = [row["slope_error_percent"] for row in dense_rows]
        source_errors = [row["pressure_error_percent"] for row in source_rows]
        endpoint_temperature = high
        reports.append({
            "key": table.key,
            "name": table.name,
            "cas": cas,
            "backend": curve.backend,
            "fluid": curve.fluid,
            "table_T_min": table.T_min,
            "table_T_max": table.T_max,
            "cp_T_triple": curve.T_triple,
            "cp_T_critical": curve.T_critical,
            "cp_P_critical_bar": curve.P_critical_bar,
            "dense_pressure": metric_summary(pressure_errors),
            "dense_slope": metric_summary(slope_errors),
            "source_pressure": metric_summary(source_errors),
            "upper_pressure_error_percent": 100.0 * math.expm1(
                float(table._ln_pressure_spline(endpoint_temperature))
                - curve.ln_pressure(endpoint_temperature)
            ),
            "upper_slope_error_percent": relative_percent(
                float(table._ln_pressure_spline.derivative()(endpoint_temperature)),
                curve.slope(endpoint_temperature),
            ),
        })

    with open(TABLE_OUTPUT_PATH, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=detail_rows[0].keys())
        writer.writeheader()
        writer.writerows(detail_rows)
    return reports, detail_rows, unsupported


def perry_comparisons():
    library = PerryPropertyLibrary()
    library._load()
    fluids_by_cas, duplicates = coolprop_fluids_by_cas()
    reports = []
    detail_rows = []
    failures = []

    for cas, entry in library.chemicals.items():
        if not entry.get("vapor_pressure"):
            continue
        fluid = fluids_by_cas.get(cas)
        if fluid is None:
            continue
        try:
            curve = selected_curve(fluid, cas)
        except Exception as error:
            failures.append((cas, entry.get("name"), fluid, str(error)))
            continue
        ranges = [
            (float(row["T_min_K"]), float(row["T_max_K"]))
            for row in entry.get("vapor_pressure", [])
        ]
        low = max(min(item[0] for item in ranges), curve.T_triple)
        high = min(max(item[1] for item in ranges), curve.T_upper)
        rows = []
        for temperature in comparison_points(low, high):
            row = perry_row(library, entry, temperature)
            if row is None:
                continue
            try:
                cp_ln = curve.ln_pressure(temperature)
                cp_slope = curve.slope(temperature)
            except Exception:
                continue
            reference_ln = perry_ln_pressure(row, temperature)
            reference_slope = perry_slope(row, temperature)
            if reference_slope <= 0.0:
                continue
            item = {
                "cas": cas,
                "name": entry.get("name") or fluid,
                "backend": curve.backend,
                "fluid": curve.fluid,
                "temperature_K": temperature,
                "reduced_temperature_cp": temperature / curve.T_critical,
                "pressure_bar": math.exp(cp_ln),
                "pressure_error_percent": 100.0 * math.expm1(cp_ln - reference_ln),
                "slope_error_percent": relative_percent(cp_slope, reference_slope),
            }
            rows.append(item)
            detail_rows.append(item)
        if len(rows) < 20:
            failures.append((cas, entry.get("name"), fluid, "fewer than 20 overlap points"))
            continue
        critical = entry.get("critical_constants") or {}
        perry_tc = critical.get("Tc_K")
        perry_pc = critical.get("Pc_MPa")
        reports.append({
            "cas": cas,
            "name": entry.get("name") or fluid,
            "backend": curve.backend,
            "fluid": curve.fluid,
            "low": rows[0]["temperature_K"],
            "high": rows[-1]["temperature_K"],
            "pressure": metric_summary([item["pressure_error_percent"] for item in rows]),
            "slope": metric_summary([item["slope_error_percent"] for item in rows]),
            "Tc_error_percent": (
                relative_percent(curve.T_critical, float(perry_tc))
                if perry_tc is not None else None
            ),
            "Pc_error_percent": (
                relative_percent(curve.P_critical_bar, float(perry_pc) * 10.0)
                if perry_pc is not None else None
            ),
        })

    with open(PERRY_OUTPUT_PATH, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=detail_rows[0].keys())
        writer.writeheader()
        writer.writerows(detail_rows)
    return reports, detail_rows, failures, duplicates


def water_backend_comparison():
    if97 = CoolPropSaturationCurve("Water", "IF97")
    heos = CoolPropSaturationCurve("Water", "HEOS")
    low = max(if97.T_triple, heos.T_triple)
    high = min(if97.T_upper, heos.T_upper)
    rows = []
    for temperature in comparison_points(low, high):
        rows.append({
            "temperature_K": temperature,
            "pressure_error_percent": 100.0 * math.expm1(
                if97.ln_pressure(temperature) - heos.ln_pressure(temperature)
            ),
            "slope_error_percent": relative_percent(
                if97.slope(temperature),
                heos.slope(temperature),
            ),
        })
    with open(WATER_OUTPUT_PATH, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def canonical_fit_comparisons():
    alias_payload = json.loads(
        (
            DATA_PATH.parent
            / "coolprop_fluid_aliases.json"
        ).read_text()
    )
    fluids = sorted(set(alias_payload.get("aliases", {}).values()))
    reports = []
    failures = []
    for fluid in fluids:
        cas = CP.get_fluid_param_string(fluid, "CAS").strip()
        component = {
            "symbol": fluid,
            "name": fluid,
            "CAS": cas,
        }
        adapter = PsatCanonicalizationAdapter(component)
        inputs = adapter.collect_inputs()
        if not inputs.segments:
            continue
        segment = inputs.segments[0]
        if segment.metadata["fluid"] != fluid:
            failures.append((
                fluid,
                f"exact CAS {cas!r} resolved to {segment.metadata['fluid']!r}",
            ))
            continue
        qualified_name = segment.metadata["qualified_name"]
        domain_component = {
            **component,
            "Tt": segment.T_min,
            "Pt": segment.metadata["reported_triple_pressure_bar"],
        }
        domain = PsatCanonicalizationAdapter(
            domain_component,
            psat_at_temperature=lambda _item, temperature: (
                CP.PropsSI("P", "T", temperature, "Q", 0.0, qualified_name)
                / 100000.0
            ),
            tsat_at_pressure=lambda _item, pressure_bar: CP.PropsSI(
                "T",
                "P",
                pressure_bar * 100000.0,
                "Q",
                0.0,
                qualified_name,
            ),
        ).resolve_domain(T_critical=segment.T_max)
        try:
            T_boiling = float(
                CP.PropsSI(
                    "T",
                    "P",
                    101325.0,
                    "Q",
                    0.0,
                    qualified_name,
                )
            )
        except Exception:
            T_boiling = None
        if T_boiling is not None and not segment.T_min < T_boiling < segment.T_max:
            T_boiling = None

        try:
            result = PsatCanonicalizer(
                T_min=domain.T_min,
                T_critical=segment.T_max,
                P_critical_bar=segment.P_max_bar,
                critical_quality=segment.quality,
                T_boiling=T_boiling,
                boiling_quality=segment.quality if T_boiling is not None else None,
            ).canonicalize(**inputs.canonicalizer_arguments())
        except Exception as error:
            failures.append((fluid, str(error)))
            continue

        diagnostics = result.curve.diagnostics
        reports.append({
            "fluid": fluid,
            "backend": segment.metadata["backend"],
            "T_min_K": domain.T_min,
            "domain_basis": domain.basis,
            "domain_pressure_bar": domain.pressure_bar if domain.pressure_bar is not None else "",
            "T_critical_K": segment.T_max,
            "T_boiling_K": T_boiling if T_boiling is not None else "",
            "sample_count": diagnostics.sample_count,
            "canonical_form": result.curve.form.value,
            "inverse_power": (
                result.curve.inverse_power
                if result.curve.inverse_power is not None else ""
            ),
            "attempted_forms": "/".join(
                result.curve.metadata.get("attempted_forms", ())
            ),
            "mard_percent": diagnostics.mard_percent,
            "p95_error_percent": diagnostics.p95_absolute_relative_error_percent,
            "max_error_percent": diagnostics.max_absolute_relative_error_percent,
            "condition_number": diagnostics.reduced_condition_number,
            "monotonic": diagnostics.monotonic,
        })

    if reports:
        with open(CANONICAL_OUTPUT_PATH, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=reports[0].keys())
            writer.writeheader()
            writer.writerows(reports)
    return reports, failures


def print_metric(label, metrics):
    print(
        f"{label}: n={metrics['count']} MARD={metrics['mard']:.4f}% "
        f"median={metrics['median']:.4f}% p95={metrics['p95']:.4f}% "
        f"max={metrics['max']:.4f}% bias={metrics['bias']:+.4f}%"
    )


def print_perry_region(label, rows, predicate):
    selected = [row for row in rows if predicate(row)]
    by_cas = {}
    for row in selected:
        by_cas.setdefault(row["cas"], []).append(abs(row["pressure_error_percent"]))
    curve_mards = [mean(values) for values in by_cas.values()]
    metrics = metric_summary([row["pressure_error_percent"] for row in selected])
    print_metric(f"  {label}", metrics)
    print(
        f"    curves={len(curve_mards)} per-curve MARD "
        f"median={median(curve_mards):.4f}% "
        f"p95={percentile(curve_mards, 0.95):.4f}%"
    )


def main():
    table_reports, table_rows, unsupported = table_comparisons()
    print("LOCAL PSAT TABLES VERSUS SELECTED COOLPROP BACKEND")
    for report in table_reports:
        print(
            f"\n{report['name']} [{report['key']}] -> "
            f"{report['backend']}::{report['fluid']}"
        )
        print(
            f"  table={report['table_T_min']:.3f}-{report['table_T_max']:.3f} K; "
            f"CoolProp triple={report['cp_T_triple']:.6f} K, "
            f"critical={report['cp_T_critical']:.6f} K / "
            f"{report['cp_P_critical_bar']:.6f} bar"
        )
        print_metric("  source rows", report["source_pressure"])
        print_metric("  dense PCHIP pressure", report["dense_pressure"])
        print_metric("  dense PCHIP slope", report["dense_slope"])
        print(
            f"  upper overlap: P={report['upper_pressure_error_percent']:+.4f}% "
            f"slope={report['upper_slope_error_percent']:+.4f}%"
        )
    for key, name, cas in unsupported:
        print(f"\nunsupported local table: {name} [{key}, {cas}]")

    dense_table_rows = [row for row in table_rows if row["kind"] == "dense_spline"]
    print("\nLOCAL TABLE POOLED")
    print_metric(
        "  pressure",
        metric_summary([row["pressure_error_percent"] for row in dense_table_rows]),
    )
    print_metric(
        "  slope",
        metric_summary([row["slope_error_percent"] for row in dense_table_rows]),
    )

    water_rows = water_backend_comparison()
    print("\nIF97 WATER VERSUS HEOS WATER")
    print_metric(
        "  pressure",
        metric_summary([row["pressure_error_percent"] for row in water_rows]),
    )
    print_metric(
        "  slope",
        metric_summary([row["slope_error_percent"] for row in water_rows]),
    )

    canonical_reports, canonical_failures = canonical_fit_comparisons()
    print("\nCOOLPROP FULL CURVES TO CANONICAL FIT")
    print(
        f"canonicalized={len(canonical_reports)} "
        f"failures={len(canonical_failures)}"
    )
    form_counts = {
        form: sum(report["canonical_form"] == form for report in canonical_reports)
        for form in ("A-F", "A-G", "A-H")
    }
    print(
        "  selected forms: "
        + ", ".join(f"{form}={count}" for form, count in form_counts.items())
    )
    for key, label in (
        ("mard_percent", "per-curve MARD"),
        ("p95_error_percent", "per-curve p95 error"),
        ("max_error_percent", "per-curve maximum error"),
    ):
        values = [report[key] for report in canonical_reports]
        print(
            f"  {label}: median={median(values):.4f}% "
            f"p95={percentile(values, 0.95):.4f}% max={max(values):.4f}%"
        )
    print(
        "  maximum-error counts: "
        f">1%={sum(report['max_error_percent'] > 1.0 for report in canonical_reports)}, "
        f">5%={sum(report['max_error_percent'] > 5.0 for report in canonical_reports)}, "
        f">10%={sum(report['max_error_percent'] > 10.0 for report in canonical_reports)}"
    )
    print("  largest canonical-fit maximum errors:")
    for report in sorted(
        canonical_reports,
        key=lambda item: item["max_error_percent"],
        reverse=True,
    )[:10]:
        print(
            f"    {report['fluid']} [{report['backend']}]: "
            f"MARD={report['mard_percent']:.4f}% "
            f"p95={report['p95_error_percent']:.4f}% "
            f"max={report['max_error_percent']:.4f}%"
        )
    if canonical_failures:
        print("  canonical fit failures:")
        for failure in canonical_failures:
            print("   ", failure)

    perry_reports, perry_rows, failures, duplicates = perry_comparisons()
    print("\nPERRY 2-8 VERSUS SELECTED COOLPROP BACKEND")
    print(
        f"matched curves={len(perry_reports)} points={len(perry_rows)} "
        f"failures={len(failures)} duplicate CoolProp CAS={len(duplicates)}"
    )
    print_metric(
        "  pressure",
        metric_summary([row["pressure_error_percent"] for row in perry_rows]),
    )
    print_metric(
        "  slope",
        metric_summary([row["slope_error_percent"] for row in perry_rows]),
    )
    print_perry_region(
        "P >= 0.001 bar",
        perry_rows,
        lambda row: row["pressure_bar"] >= 0.001,
    )
    print_perry_region(
        "P >= 0.01 bar",
        perry_rows,
        lambda row: row["pressure_bar"] >= 0.01,
    )
    print_perry_region(
        "P >= 0.25 bar",
        perry_rows,
        lambda row: row["pressure_bar"] >= 0.25,
    )
    print_perry_region(
        "Tr >= 0.7",
        perry_rows,
        lambda row: row["reduced_temperature_cp"] >= 0.7,
    )
    curve_pressure_mards = [report["pressure"]["mard"] for report in perry_reports]
    curve_slope_medians = [report["slope"]["median"] for report in perry_reports]
    print(
        f"  per-curve pressure MARD median={median(curve_pressure_mards):.4f}% "
        f"p95={percentile(curve_pressure_mards, 0.95):.4f}%"
    )
    print(
        f"  per-curve slope median-error median={median(curve_slope_medians):.4f}% "
        f"p95={percentile(curve_slope_medians, 0.95):.4f}%"
    )
    critical_tc = [abs(report["Tc_error_percent"]) for report in perry_reports if report["Tc_error_percent"] is not None]
    critical_pc = [abs(report["Pc_error_percent"]) for report in perry_reports if report["Pc_error_percent"] is not None]
    print(
        f"  critical mismatch |Tc| median={median(critical_tc):.4f}% "
        f"p95={percentile(critical_tc, 0.95):.4f}%; "
        f"|Pc| median={median(critical_pc):.4f}% "
        f"p95={percentile(critical_pc, 0.95):.4f}%"
    )
    print("\n  largest Perry pressure disagreements:")
    for report in sorted(perry_reports, key=lambda item: item["pressure"]["mard"], reverse=True)[:15]:
        print(
            f"    {report['name']} [{report['cas']}, {report['backend']}::{report['fluid']}]: "
            f"MARD={report['pressure']['mard']:.3f}% "
            f"p95={report['pressure']['p95']:.3f}% "
            f"max={report['pressure']['max']:.3f}% "
            f"slope median={report['slope']['median']:.3f}% "
            f"Tc={report['Tc_error_percent']:+.3f}% "
            f"Pc={report['Pc_error_percent']:+.3f}%"
        )
    if failures:
        print("\n  failures:")
        for failure in failures:
            print("   ", failure)
    print(f"\nwrote {TABLE_OUTPUT_PATH}")
    print(f"wrote {PERRY_OUTPUT_PATH}")
    print(f"wrote {WATER_OUTPUT_PATH}")
    print(f"wrote {CANONICAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
