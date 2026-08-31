"""Rate-limited online corroboration for provisional CoolProp triple points."""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class ProvisionalTripleTarget:
    component_key: str
    cas: str
    name: str
    Tt_K: float
    Pt_bar: float

    @property
    def priority(self) -> tuple[int, int, int, float, str]:
        topology_sensitive = int(self.Pt_bar < 1.01325)
        engineering_fluid_code = int(bool(re.match(
            r'^(?:R\d|HFE|NOVEC|RC\d|D[456]$|M(?:D\d*M|DM|M)$)',
            self.name.strip().upper(),
        )))
        rounded_temperature = int(
            min(
                abs(self.Tt_K - round(self.Tt_K)),
                abs((self.Tt_K - 273.15) - round(self.Tt_K - 273.15)),
            ) > 0.02
        )
        return (
            topology_sensitive,
            engineering_fluid_code,
            rounded_temperature,
            -self.Pt_bar,
            self.cas,
        )


class RequestBudgetExhausted(OSError):
    pass


class RateLimitedUrlopen:
    """Count actual HTTP attempts and space their start times."""

    def __init__(
        self,
        original: Callable[..., Any],
        *,
        maximum_requests: int,
        minimum_interval_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.original = original
        self.maximum_requests = int(maximum_requests)
        self.minimum_interval_seconds = float(minimum_interval_seconds)
        self.monotonic = monotonic
        self.sleep = sleep
        self.request_count = 0
        self.last_started_at: float | None = None
        self.request_log: list[dict[str, Any]] = []

    @property
    def exhausted(self) -> bool:
        return self.request_count >= self.maximum_requests

    def __call__(self, request, *args, **kwargs):
        if self.exhausted:
            raise RequestBudgetExhausted(
                f'online warm request budget exhausted at '
                f'{self.maximum_requests} requests'
            )
        now = self.monotonic()
        if self.last_started_at is not None:
            delay = (
                self.minimum_interval_seconds
                - (now - self.last_started_at)
            )
            if delay > 0.0:
                self.sleep(delay)
        started_at = self.monotonic()
        self.last_started_at = started_at
        self.request_count += 1
        url = getattr(request, 'full_url', request)
        self.request_log.append({
            'number': self.request_count,
            'started_at_monotonic': started_at,
            'url': str(url),
        })
        return self.original(request, *args, **kwargs)


def provisional_targets(database: Path) -> list[ProvisionalTripleTarget]:
    from property_resolution.phase_change import PhaseChangeMixin

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            WITH ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY component_key
                    ORDER BY updated_at_utc DESC, rowid DESC
                ) AS rn
                FROM resolved_phase_point_cache
                WHERE cache_version = ?
                  AND resolution_kind = 'triple_point'
            )
            SELECT component_key, cas, component_name, Tt_value, Pt_value
            FROM ranked
            WHERE rn = 1
              AND Tt_value IS NOT NULL
              AND Pt_value IS NOT NULL
              AND Tt_quality = ?
              AND Pt_quality = ?
            """,
            (
                int(PhaseChangeMixin.PHASE_POINT_CACHE_VERSION),
                float(PhaseChangeMixin.COOLPROP_PROVISIONAL_TRIPLE_QUALITY),
                float(PhaseChangeMixin.COOLPROP_PROVISIONAL_TRIPLE_QUALITY),
            ),
        ).fetchall()
    targets = []
    for component_key, cas, name, Tt, Pt in rows:
        cas = str(cas or '').strip()
        try:
            Tt = float(Tt)
            Pt = float(Pt)
        except (TypeError, ValueError):
            continue
        if (
            not cas
            or not math.isfinite(Tt)
            or Tt <= 0.0
            or not math.isfinite(Pt)
            or Pt <= 0.0
        ):
            continue
        targets.append(ProvisionalTripleTarget(
            component_key=str(component_key),
            cas=cas,
            name=str(name or cas),
            Tt_K=Tt,
            Pt_bar=Pt,
        ))
    return sorted(targets, key=lambda item: item.priority)


def classify_result(triple: dict[str, Any]) -> str:
    Tt = triple.get('Tt')
    if Tt is None or Tt.value is None:
        return 'rejected_or_missing'
    method = str(Tt.method or '')
    if method.startswith('coolprop_') and float(Tt.quality) >= 0.995:
        return 'coolprop_confirmed'
    if not method.startswith('coolprop_'):
        return 'external_selected'
    return 'still_provisional'


def main() -> int:
    from property_resolution.phase_change import SATURATION_PROPERTIES_CACHE_PATH
    from property_resolution.resolver import PropertyResolver

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-requests', type=int, default=100)
    parser.add_argument('--minimum-interval', type=float, default=1.05)
    parser.add_argument('--max-targets', type=int)
    parser.add_argument(
        '--report',
        type=Path,
        default=Path('/tmp/coolprop_triple_corroboration_warm_report.json'),
    )
    args = parser.parse_args()
    if not 1 <= args.max_requests <= 100:
        parser.error('--max-requests must be between 1 and 100')
    if args.minimum_interval <= 1.0:
        parser.error('--minimum-interval must be greater than 1 second')
    if args.max_targets is not None and args.max_targets <= 0:
        parser.error('--max-targets must be positive')

    targets = provisional_targets(Path(SATURATION_PROPERTIES_CACHE_PATH))
    if args.max_targets is not None:
        targets = targets[:args.max_targets]
    resolver = PropertyResolver()
    original_urlopen = urllib.request.urlopen
    limiter = RateLimitedUrlopen(
        original_urlopen,
        maximum_requests=args.max_requests,
        minimum_interval_seconds=args.minimum_interval,
    )
    urllib.request.urlopen = limiter
    outcomes = []
    stopped_reason = ''
    started = time.monotonic()
    try:
        for index, target in enumerate(targets, start=1):
            if limiter.exhausted:
                stopped_reason = 'request_budget_exhausted'
                break
            props = {
                'CAS': target.cas,
                'cas': target.cas,
                'name': target.name,
            }
            resolver._identifier_candidates = (
                lambda _identifier, _props=None, cas=target.cas: [cas]
            )
            before = limiter.request_count
            provider_status = {}
            transient = False
            for provider, fetch in (
                ('pubchem', resolver._fetch_phase_change_pubchem),
                ('nist', resolver._fetch_phase_change_nist),
            ):
                try:
                    provider_status[provider] = bool(fetch(target.cas))
                except LookupError as error:
                    provider_status[provider] = 'transient_failure'
                    transient = True
                    if limiter.exhausted:
                        stopped_reason = 'request_budget_exhausted'
                    else:
                        stopped_reason = f'{provider}_transient_failure'
                    provider_status['error'] = str(error)
                    break
            outcome = {
                **asdict(target),
                'index': index,
                'requests_used': limiter.request_count - before,
                'provider_status': provider_status,
            }
            if transient:
                outcome['classification'] = 'transient_failure'
                outcomes.append(outcome)
                break
            triple = resolver.resolve_triple_point(
                target.cas,
                props,
                allow_online=True,
            )
            outcome.update({
                'classification': classify_result(triple),
                'selected_Tt': asdict(triple['Tt']),
                'selected_Pt': asdict(triple['Pt']),
            })
            outcomes.append(outcome)
            print(
                f'{index}/{len(targets)} {target.cas} {target.name}: '
                f'{outcome["classification"]}; '
                f'{outcome["requests_used"]} request(s), '
                f'{limiter.request_count}/{args.max_requests} total',
                flush=True,
            )
    finally:
        urllib.request.urlopen = original_urlopen

    report = {
        'maximum_requests': args.max_requests,
        'minimum_interval_seconds': args.minimum_interval,
        'requests_used': limiter.request_count,
        'targets_available': len(provisional_targets(
            Path(SATURATION_PROPERTIES_CACHE_PATH)
        )),
        'targets_selected': len(targets),
        'targets_processed': len(outcomes),
        'stopped_reason': stopped_reason,
        'elapsed_seconds': time.monotonic() - started,
        'outcomes': outcomes,
        'request_log': limiter.request_log,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        f'Processed {len(outcomes)}/{len(targets)} target(s); '
        f'{limiter.request_count}/{args.max_requests} request(s); '
        f'report: {args.report}',
        flush=True,
    )
    return 0 if not stopped_reason.endswith('transient_failure') else 1


if __name__ == '__main__':
    raise SystemExit(main())
