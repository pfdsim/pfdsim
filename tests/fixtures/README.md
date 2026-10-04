# Safe runtime-cache baseline

`runtime_cache.json` freezes positive PubChem/NIST source records for compounds
identified by the maintained local chemical database. It was captured from the
runtime source cache on 2026-10-04. Each record retains its original capture time,
source notes, candidate evidence, and quality metadata. Alias keys share one
payload in the fixture.

The baseline excludes negative lookups, arbitrary test/user aliases, obsolete
provider contracts, selected property answers, fitted interactions, canonical
pressure curves, and quantum artifacts. Tests still calculate their numerical
answers with the current implementation. No personal runtime cache is consulted
during a test run.

The shared test harness materializes this fixture once per process, with fresh
SQLite insertion timestamps so a frozen fixture does not disappear because of
production cache expiry. Every test receives an independent writable copy;
writes cannot contaminate the fixture or another test. Explicit cache paths
configured by a test remain empty and independent.

Use `@empty_runtime_cache` on a unittest class/method or pytest function, or
`isolated_runtime_caches(seeded=False)`, when testing source priority, cache
misses, expiry, or cold-start behavior. Real provider requests are blocked by
default. Response mocks still exercise provider parsers and routing. Live
integration operations explicitly open the provider scope through
`run_optional_live_provider` or `allow_live_providers`.

Bare `python -m pytest` uses six pytest-xdist workers with dynamic scheduling.
Worker startup is guarded so a worker cannot create another pool. Explicit
pytest arguments preserve the requested worker count and test selection.
Unittest default discovery delegates to the same pytest worker implementation.

To refresh the source fixture deliberately:

```bash
python scripts/snapshot_test_cache.py data/runtime/property_cache.sqlite \
    tests/fixtures/runtime_cache.json
```

Review the source-data diff before accepting it. This command is never invoked
automatically. It exports positive records only; a transport outage must never
be recorded as a fixture declaring that a provider has no data.
