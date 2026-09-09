"""Populate the persistent canonical-Psat cache from all local source catalogs.

The candidate union contains every Perry 9th Table 2-8 vapor-pressure
correlation, every Perry Table 2-10 vapor-pressure table, and every CoolProp
fluid having a unique nonempty CAS number. Duplicate CAS numbers are fitted
only once, with all source labels retained in the summary.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import multiprocessing
import os
import sqlite3
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_WORKER_DATABASE = None
_WORKER_RESOLVER = None


def _coolprop_candidates() -> dict[str, dict]:
    try:
        import CoolProp.CoolProp as CP
    except ImportError:
        return {}

    candidates = {}
    for fluid in CP.get_global_param_string("FluidsList").split(","):
        try:
            cas = CP.get_fluid_param_string(fluid, "CAS").strip()
        except Exception:
            continue
        if not cas:
            continue
        candidates.setdefault(cas, {
            "cas": cas,
            "name": fluid,
            "formula": "",
            "Tb": None,
            "Tm": None,
            "sources": set(),
        })["sources"].add("CoolProp")
    return candidates


def source_candidates() -> list[dict]:
    from perry_properties import get_perry_property_library

    library = get_perry_property_library()
    library._load()
    library._load_table_2_10()
    candidates = _coolprop_candidates()

    for cas, entry in library.chemicals.items():
        if not entry.get("vapor_pressure"):
            continue
        item = candidates.setdefault(cas, {
            "cas": cas,
            "name": entry.get("name") or cas,
            "formula": entry.get("formula")
            or (entry.get("formulas") or [""])[0],
            "Tb": None,
            "Tm": None,
            "sources": set(),
        })
        item["name"] = entry.get("name") or item["name"]
        item["formula"] = (
            entry.get("formula")
            or (entry.get("formulas") or [""])[0]
            or item["formula"]
        )
        item["sources"].add("Perry 2-8")

    for cas, entry in library.table_2_10_chemicals.items():
        item = candidates.setdefault(cas, {
            "cas": cas,
            "name": entry.get("name") or cas,
            "formula": entry.get("formula") or "",
            "Tb": entry.get("Tb_K"),
            "Tm": entry.get("Tm_K"),
            "sources": set(),
        })
        item["name"] = item["name"] or entry.get("name") or cas
        item["formula"] = item["formula"] or entry.get("formula") or ""
        if item["Tb"] is None:
            item["Tb"] = entry.get("Tb_K")
        if item["Tm"] is None:
            item["Tm"] = entry.get("Tm_K")
        item["sources"].add("Perry 2-10")

    result = []
    for item in candidates.values():
        result.append({
            **item,
            "sources": tuple(sorted(item["sources"])),
        })
    return sorted(result, key=lambda item: item["cas"])


def _initialize_worker() -> None:
    global _WORKER_DATABASE, _WORKER_RESOLVER
    from chemical_properties import ChemicalDatabase
    from property_resolution.resolver import PropertyResolver

    _WORKER_DATABASE = ChemicalDatabase(enable_online=False)
    _WORKER_RESOLVER = PropertyResolver()


def _minimal_component(candidate: dict):
    from chemical_properties import ChemicalProperties

    formula = str(candidate.get("formula") or "")
    name = str(candidate.get("name") or candidate["cas"])
    component = ChemicalProperties(
        symbol=formula or name,
        name=name,
        formula=formula,
        CAS=str(candidate["cas"]),
        Tb=candidate.get("Tb"),
        Tm=candidate.get("Tm"),
        source="canonical_psat_cache_warm",
    )
    for field_name in ("Tb", "Tm"):
        value = getattr(component, field_name)
        if value is None:
            continue
        component.property_sources[field_name] = {
            "source": "Perry 9th Table 2-10",
            "method": "perry_table_2_10_phase_point",
            "quality": 0.95,
            "notes": "Cache-warm component initialized from Perry Table 2-10",
        }
    return component


def _warm_one(candidate: dict) -> dict:
    started = time.perf_counter()
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            component = _WORKER_DATABASE.get(
                candidate["cas"],
                fetch_online=False,
            )
            if component is None:
                component = _minimal_component(candidate)
                _WORKER_DATABASE._hydrate_properties(
                    component,
                    allow_online=False,
                )
            props = component.to_dict()
            _WORKER_RESOLVER.resolve_vapor_pressure_coefficients(
                component.symbol,
                props,
                allow_online=False,
            )
            minimum_pressure_bar = (
                _WORKER_RESOLVER._canonical_minimum_pressure_bar(props, None)
            )
            cache_key = _WORKER_RESOLVER._canonical_vapor_pressure_cache_key(
                component.symbol,
                props,
                False,
                minimum_pressure_bar,
            )
            runtime = _WORKER_RESOLVER._canonical_vapor_pressure_curves[
                cache_key
            ]
        return {
            "ok": True,
            "cas": candidate["cas"],
            "name": candidate["name"],
            "sources": candidate["sources"],
            "elapsed": time.perf_counter() - started,
            "form": runtime.curve.form.value,
            "T_min": runtime.curve.T_min,
            "T_critical": runtime.curve.T_critical,
            "quality": runtime.curve.quality,
        }
    except Exception as error:
        return {
            "ok": False,
            "cas": candidate["cas"],
            "name": candidate["name"],
            "sources": candidate["sources"],
            "elapsed": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
            "captured_output": output.getvalue()[-2000:],
        }


def _cache_row_count(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        return int(connection.execute(
            "SELECT count(*) FROM canonical_psat_cache"
        ).fetchone()[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=min(6, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/canonical_psat_cache_warm_report.json"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any source component cannot be canonicalized.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    from property_resolution.resolver import PropertyResolver

    resolver = PropertyResolver()
    cache_path = resolver.initialize_canonical_vapor_pressure_disk_cache()
    candidates = source_candidates()
    started = time.perf_counter()
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context('fork'),
        initializer=_initialize_worker,
    ) as pool:
        futures = [pool.submit(_warm_one, item) for item in candidates]
        for index, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            results.append(future.result())
            if index % 50 == 0 or index == len(futures):
                successes = sum(item["ok"] for item in results)
                print(
                    f"{index}/{len(futures)} processed; "
                    f"{successes} cached, {index - successes} unavailable",
                    flush=True,
                )

    results.sort(key=lambda item: item["cas"])
    payload = {
        "cache_path": str(cache_path),
        "cache_version": resolver.CANONICAL_PSAT_CACHE_VERSION,
        "workers": args.workers,
        "elapsed_seconds": time.perf_counter() - started,
        "candidate_count": len(candidates),
        "success_count": sum(item["ok"] for item in results),
        "failure_count": sum(not item["ok"] for item in results),
        "database_row_count": _cache_row_count(cache_path),
        "results": results,
    }
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(
        f"Cache rows: {payload['database_row_count']}; "
        f"report: {args.report}"
    )
    return 1 if args.strict and payload["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
