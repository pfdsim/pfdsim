#!/usr/bin/env python3
"""Timing probe for the unified compiled VLLE backend.

This is intentionally a benchmark script, not a unit test.  It times the
compiled ideal-vapor activity backend across representative mixtures, methods,
phase behavior, and guess quality.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermodynamics import create_thermodynamics  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    method: str
    components: tuple[str, ...]
    z: dict[str, float]
    T: float
    P: float
    description: str


CASES = (
    Case(
        name="unifnist_ternary_vlle",
        method="UNIFNIST",
        components=("water", "ethanol", "cyclohexane"),
        z={"water": 0.30, "ethanol": 0.20, "cyclohexane": 0.50},
        T=337.0,
        P=1.01325,
        description="ternary VLLE",
    ),
    Case(
        name="unifnist_high_alcohol_vle",
        method="UNIFNIST",
        components=("water", "ethanol", "cyclohexane"),
        z={"water": 0.20, "ethanol": 0.60, "cyclohexane": 0.20},
        T=337.0,
        P=1.01325,
        description="high-alcohol ordinary VLE",
    ),
    Case(
        name="unifnist_binary_invariant",
        method="UNIFNIST",
        components=("water", "chloroform"),
        z={"water": 0.50, "chloroform": 0.50},
        T=329.1003613978814,
        P=1.01325,
        description="binary invariant VLLE",
    ),
    Case(
        name="nrtl_ternary_vlle",
        method="NRTL",
        components=("water", "methanol", "benzene"),
        z={"water": 0.20, "methanol": 0.30, "benzene": 0.50},
        T=333.0,
        P=1.01325,
        description="NRTL ternary VLLE",
    ),
    Case(
        name="uniquac_ternary_vlle",
        method="UNIQUAC",
        components=("water", "ethanol", "benzene"),
        z={"water": 0.30, "ethanol": 0.10, "benzene": 0.60},
        T=339.0,
        P=1.01325,
        description="UNIQUAC ternary VLLE",
    ),
)


def format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def time_call(fn: Callable, repeats: int, warmups: int) -> tuple[dict[str, float], object]:
    last = None
    for _ in range(warmups):
        last = fn()
    samples = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            last = fn()
            samples.append((time.perf_counter() - start) * 1000.0)
    finally:
        if was_enabled:
            gc.enable()
    ordered = sorted(samples)
    stats = {
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "p95": ordered[int(0.95 * len(ordered))],
        "min": ordered[0],
        "max": ordered[-1],
    }
    return stats, last


def print_timing(label: str, stats: dict[str, float], detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(
        f"  {label:<28}"
        f" med={stats['median']:8.3f} ms"
        f" mean={stats['mean']:8.3f}"
        f" p95={stats['p95']:8.3f}"
        f" min={stats['min']:8.3f}"
        f" max={stats['max']:8.3f}"
        f"{suffix}"
    )


def phase_detail(result) -> str:
    return (
        f"phase={result.phase_count}"
        f" status={getattr(result, 'status', '?')}"
        f" V={result.vapor_fraction:.9g}"
    )


def safe_timing(label: str, fn: Callable, repeats: int, warmups: int) -> object | None:
    try:
        stats, last = time_call(fn, repeats, warmups)
    except Exception as exc:
        print(f"  {label:<28} ERROR {type(exc).__name__}: {exc}")
        return None
    detail = ""
    if isinstance(last, tuple):
        if len(last) >= 2 and hasattr(last[1], "phase_count"):
            detail = f"value={format_value(last[0])} {phase_detail(last[1])}"
        else:
            detail = "value=" + ", ".join(format_value(item) for item in last[:3])
    elif hasattr(last, "phase_count"):
        detail = phase_detail(last)
    else:
        detail = f"value={format_value(last)}"
    print_timing(label, stats, detail)
    return last


def safe_pv_timing(
    label: str,
    fn: Callable,
    repeats: int,
    warmups: int,
    target_T: float,
    target_vapor_fraction: float,
) -> object | None:
    try:
        stats, last = time_call(fn, repeats, warmups)
    except Exception as exc:
        print(f"  {label:<28} ERROR {type(exc).__name__}: {exc}")
        return None
    T, result = last
    dT = float(T) - float(target_T)
    dV = result.vapor_fraction - float(target_vapor_fraction)
    ok = abs(dT) <= 0.05 and abs(dV) <= 5e-5
    detail = (
        f"T={T:.9g}"
        f" dT={dT:+.3g}"
        f" phase={result.phase_count}"
        f" V={result.vapor_fraction:.9g}"
        f" dV={dV:+.3g}"
        f" ok={ok}"
    )
    print_timing(label, stats, detail)
    return last


def safe_tv_timing(
    label: str,
    fn: Callable,
    repeats: int,
    warmups: int,
    target_P: float,
    target_vapor_fraction: float,
) -> object | None:
    try:
        stats, last = time_call(fn, repeats, warmups)
    except Exception as exc:
        print(f"  {label:<28} ERROR {type(exc).__name__}: {exc}")
        return None
    P, result = last
    dP = float(P) - float(target_P)
    dV = result.vapor_fraction - float(target_vapor_fraction)
    ok = abs(dP) <= 5e-5 and abs(dV) <= 5e-5
    detail = (
        f"P={P:.9g}"
        f" dP={dP:+.3g}"
        f" phase={result.phase_count}"
        f" V={result.vapor_fraction:.9g}"
        f" dV={dV:+.3g}"
        f" ok={ok}"
    )
    print_timing(label, stats, detail)
    return last


def safe_ph_timing(
    label: str,
    fn: Callable,
    repeats: int,
    warmups: int,
    target_T: float,
) -> object | None:
    try:
        stats, last = time_call(fn, repeats, warmups)
    except Exception as exc:
        print(f"  {label:<28} ERROR {type(exc).__name__}: {exc}")
        return None
    T, result, residual = last
    dT = float(T) - float(target_T)
    ok = abs(dT) <= 1e-3 and result.phase_count in (2, 3)
    detail = (
        f"T={T:.9g}"
        f" dT={dT:+.3g}"
        f" phase={result.phase_count}"
        f" Hres={residual:+.3g}"
        f" ok={ok}"
    )
    print_timing(label, stats, detail)
    return last


def disable_compiled_vlle(thermo) -> None:
    thermo._compiled_vlle_initialized = True
    thermo._compiled_vlle = None


def run_case(case: Case, repeats: int, warmups: int, include_reference: bool) -> None:
    print()
    print("=" * 96)
    print(f"{case.name} | {case.method} | {case.description}")
    print(f"components={', '.join(case.components)}")
    print(f"T={case.T:g} K  P={case.P:g} bar  z={case.z}")

    thermo = create_thermodynamics(list(case.components), case.method)
    backend = thermo.compiled_vlle_backend()
    if backend is None:
        print("compiled backend unavailable")
        return
    reference_only = create_thermodynamics(list(case.components), case.method)
    disable_compiled_vlle(reference_only)

    # Trigger Numba compilation and Psat caches outside the timed region where
    # possible.  Individual timed functions still include normal cache lookups.
    compiled_tp = backend.flash_TP(case.z, case.T, case.P, max_iter=200)
    reference_tp = thermo.flash3_TP(case.z, case.T, case.P, max_iter=200)
    reference_enthalpy = thermo._flash3_mixture_enthalpy(case.T, case.P, reference_tp)
    print("reference:", phase_detail(reference_tp))
    print("compiled: ", phase_detail(compiled_tp))

    if include_reference:
        safe_timing(
            "reference TP",
            lambda: thermo.flash3_TP(case.z, case.T, case.P, max_iter=200),
            repeats,
            warmups,
        )
    safe_timing(
        "compiled TP",
        lambda: backend.flash_TP(case.z, case.T, case.P, max_iter=200),
        repeats,
        warmups,
    )

    print("  compiled bubble T           disabled")
    print("  compiled dew T              disabled")

    try:
        bubble_T = thermo.bubble_point_T_vlle(case.z, case.P, T_guess=case.T, max_iter=200)
    except Exception:
        bubble_T = None
    try:
        dew_T = thermo.dew_point_T_vlle(case.z, case.P, T_guess=case.T, max_iter=200)
    except Exception:
        dew_T = None

    if include_reference:
        safe_timing(
            "reference bubble T",
            lambda: thermo.bubble_point_T_vlle(case.z, case.P, T_guess=case.T, max_iter=200),
            repeats,
            warmups,
        )
        safe_timing(
            "reference dew T",
            lambda: thermo.dew_point_T_vlle(reference_tp.y, case.P, T_guess=case.T, max_iter=200),
            repeats,
            warmups,
        )

    target_vapor_fraction = compiled_tp.vapor_fraction
    for label, offset in (
        ("compiled PV exact guess", 0.0),
        ("compiled PV near guess", 5.0),
        ("compiled PV high guess", 20.0),
        ("compiled PV low guess", -20.0),
    ):
        safe_pv_timing(
            label,
            lambda offset=offset: backend.flash_PV(
                case.z,
                case.P,
                target_vapor_fraction,
                T_guess=case.T + offset,
                max_iter=200,
            ),
            repeats,
            warmups,
            case.T,
            target_vapor_fraction,
        )

    if bubble_T is not None and dew_T is not None:
        low, high = sorted((float(bubble_T), float(dew_T)))
        safe_pv_timing(
            "compiled PV supplied bounds",
            lambda: backend.flash_PV(
                case.z,
                case.P,
                target_vapor_fraction,
                T_low=low,
                T_high=high,
                T_guess=case.T + 5.0,
                max_iter=200,
            ),
            repeats,
            warmups,
            case.T,
            target_vapor_fraction,
        )

    if include_reference:
        safe_pv_timing(
            "reference PV near guess",
            lambda: thermo.flash3_PV(
                case.z,
                case.P,
                reference_tp.vapor_fraction,
                T_guess=case.T + 5.0,
                max_iter=200,
            ),
            repeats,
            warmups,
            case.T,
            reference_tp.vapor_fraction,
        )

    safe_tv_timing(
        "accelerated TV",
        lambda: thermo.flash3_TV(
            case.z,
            case.T,
            reference_tp.vapor_fraction,
            P_guess=1.2 * case.P,
            max_iter=200,
        ),
        repeats,
        warmups,
        case.P,
        reference_tp.vapor_fraction,
    )

    if include_reference:
        safe_tv_timing(
            "reference TV",
            lambda: reference_only.flash3_TV(
                case.z,
                case.T,
                reference_tp.vapor_fraction,
                P_guess=1.2 * case.P,
                max_iter=200,
            ),
            repeats,
            warmups,
            case.P,
            reference_tp.vapor_fraction,
        )

    safe_ph_timing(
        "accelerated PH",
        lambda: thermo.flash3_PH(
            case.z,
            case.P,
            reference_enthalpy,
            T_guess=case.T + 8.0,
            max_iter=200,
        ),
        repeats,
        warmups,
        case.T,
    )

    if include_reference:
        safe_ph_timing(
            "reference PH",
            lambda: reference_only.flash3_PH(
                case.z,
                case.P,
                reference_enthalpy,
                T_guess=case.T + 8.0,
                max_iter=200,
            ),
            repeats,
            warmups,
            case.T,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=40,
        help="timed repeats per operation (default: 40)",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=8,
        help="warmup calls per operation before timing (default: 8)",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in CASES],
        help="case name to run; may be supplied multiple times",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="also time reference Python paths where available",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [case for case in CASES if args.case is None or case.name in args.case]
    print("Compiled VLLE timing probe")
    print(f"repeats={args.repeats} warmups={args.warmups} reference={args.reference}")
    for case in selected:
        run_case(case, args.repeats, args.warmups, args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
