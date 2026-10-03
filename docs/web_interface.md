# Process laboratory

Install the `web` extra and start a local laboratory:

```sh
uv sync --extra web
uv run --extra web python app.py
```

The local server listens on `127.0.0.1:5000`. The title screen is `/`, the
flowsheet workspace is `/editor`, preferences are `/settings`, and the phase
atlas is `/vle-chart`.

For the deployment configuration in this repository:

```sh
uv run --extra web gunicorn -c gunicorn.conf.py wsgi:app
```

There are **two HTTP workers** with four request threads each, plus **two
persistent calculation workers** shared across them. Gunicorn also has its
ordinary supervisor process. Calculation workers start on demand. A new
flowsheet goes to the worker with the shortest queue; subsequent requests for
that canonical flowsheet retain affinity. A worker retains up to two initialized
simulators and retires after 15 idle minutes. Disk/property/compiled caches use
the simulator's normal cache locations and policies.

## Editing and autosave

- The equipment inventory comes from the current unit registry; every canonical
  equipment type has a shaded engineering illustration.
- Click or drag equipment into the laboratory. Click an outlet and then an inlet
  to create a stream. **Connect stream** also creates feeds and products.
- The inspector uses named controls and separates primary operating settings,
  equipment/model details, solver controls, reactions, ports, and layout.
  Stages and reflux ratio belong to the primary distillation panel. Changing a
  unit or port ID updates connections.
  The Reactions tab is hidden for equipment whose runtime model does not
  support reactions, including ordinary distillation columns. Existing reaction
  definitions on unsupported equipment remain preserved when other fields are edited.
- **Process settings** separates metadata, components, thermodynamics,
  interaction data, recycle controls, and named reactions. Components have
  identity, thermodynamic, solid/particle, and advanced property tabs.
- Optional properties use a labeled property picker with units. Composition
  tables use declared component symbols. Reaction controls expose equations,
  named references, conversion, kinetics, rate bases, and explicit units.
  Interaction controls select declared components and model-specific fields;
  particle controls pair diameters with fractions or select a parametric model.
  Correlations have equation choices and coefficient controls. Specialized
  overrides remain available without making custom dictionaries the primary
  workflow. The existing parser and physical validators remain authoritative.
- **PFD source** preserves text until **Apply source** is clicked. Invalid text
  stays recoverable, including after switching views or reloading. Visual edits
  are blocked until a pending text draft is applied or discarded.
- Undo/redo covers applied edits, connections, and equipment movement.
- Source snapshots include their unapplied drafts, so undoing Apply source can
  be redone. Dropping equipment, connecting ports, and keyboard shortcuts also
  protect unapplied inspector changes. Equipment imported without ports offers
  **Add standard ports** in its inspector.
- Example placement and stream routing use the same Graphviz/obstacle router
  as engineering SVG export. **Auto-layout** rearranges a diagram; ordinary
  edits preserve manual locations and reroute streams. Selecting equipment in
  a very small overview focuses it for readable editing.

Results separate convergence/balance summaries, stream tables and compositions,
equipment duties/performance/profiles, and solver diagnostics. Solver logs are
collapsed and full numerical results can be exported as JSON or `.pfr`.

Applied flowsheets, diagram coordinates, and unapplied source drafts are saved
to browser storage after changes. The title screen's **Saved laboratories**
list restores them. `.pfd` exports contain process specifications; diagram
coordinates remain in laboratory saves because the `.pfd` format does not
serialize them. Comments in source survive until a visual edit regenerates
canonical `.pfd` text.

Signed-in accounts also autosave to private server storage after a short
debounce. If the connection fails, the local copy stays available. Versioned
saves reject concurrent overwrites. A conflict exposes the newer account
version in **Saved laboratories**; opening it first archives the local draft
as another laboratory. Settings are browser preferences and do not silently
change existing process metadata.
Each browser tab retains the account revision it opened, so another tab's
updated save metadata cannot authorize overwriting that tab's newer work.

## Accounts and compute allowance

Local accounts use usernames and hashed passwords. No email or external
identity provider is required. Account handling is separate from process
execution so email identity can be introduced later.

| Identity | Backend CPU allowance per UTC day |
| --- | ---: |
| Guest | 300 seconds / 5 minutes |
| Signed-in account | 900 seconds / 15 minutes |

Guest quotas and account-attempt throttling use a salted IP identity. Raw IP
addresses are not stored in the account database. Visitors sharing an IP share
the guest allowance. Account computation budgets are keyed by account.

Only computation workers debit the allowance: simulations, thermodynamic
initialization, phase diagrams, and group resolution. HTTP handling, queue
waiting, network waiting, and time between calculations are excluded. CPU time
includes calculation threads and live/reaped subprocesses. Enforcement samples
every 100 ms; a running job is stopped when its budget is exhausted, with a
small possible scheduling/multicore overshoot reported as actual CPU usage.
The allowance resets at **00:00 UTC**. Downloads use stored results and never
rerun a calculation.

Jobs report queued/running/completed/failed/cancelled states, recent progress,
and CPU seconds. The shared queue admits up to eight active/queued jobs. Work
can be cancelled while queued or running. A 30-minute wall-clock deadline also
prevents an indefinitely stalled calculation; this deadline is independent of
the CPU allowance. Completed results expire after 24 hours when new work is
submitted; flowsheet autosaves do not expire with them.

## Phase atlas

Select the thermodynamic method, then choose a diagram. The water-only STEAM
model is disabled for these mixture diagrams:

- Binary T–x–y, P–x–y, and x–y curves.
- Binary VLLE T–x–y envelopes and liquid binodals at fixed pressure, with
  variable temperature and a configurable lower binodal temperature.
- Ternary liquid–liquid equilibrium tie lines at specified temperature.
- Ternary VLLE phase maps, tie lines, and three-phase triangles at specified
  temperature and pressure.

Enter any chemical identifier supported by `.pfd`, or select symbols from a
saved/imported laboratory. Component inputs are plain textboxes. **Browse
reference chemicals** is an optional, separate picker containing Perry and
`chemicals.json` identities; it never binds a native suggestion list to the
textbox. References load only when Browse is opened. Independent mixtures
allow automatic online resolution by default, consistent with `.pfd`, and
the lookup policy can be switched off in advanced options.
Saved laboratories supply their applied definitions; unapplied source drafts
are excluded. Imported files remain selectable after switching sources. The
lookup checkbox displays the selected definition's actual policy. Diagrams do
not require the imported process's equipment and feeds to be fully specified.
Changing a model or recycle method resets its additional options to defaults.
Named thermodynamic scopes use their runtime vapor-correlation defaults;
additional correlation options are available only for the global scope.
The same `Simulator.initialize()` path resolves
names, formulas, CAS identifiers, SMILES, custom component properties,
interaction overrides/estimation, thermodynamic scopes, and lookup/computation
policy. The laboratory's method can be selected per scope. Physically
inapplicable model/system combinations return descriptive errors.

Binary generators share the model's pressure/temperature and K-value solvers.
Unconverged points are gaps, with diagnostics; failed dew points are never
replaced with a fabricated bubble result. Ternary diagrams sample a triangular
composition grid and show calculated tie-line endpoints, rather than claiming
an interpolated binodal. “No split found” does not claim a globally proven
single-liquid state. Ternary LLE intentionally excludes vapor; choose the VLLE
map to include it. Splits must satisfy material balance and chemical-potential
equality before they are plotted. All data and an SVG figure can be exported.
EOS VLE curves also audit actual phase fugacities and exclude homogeneous
single-phase roots; single-phase K-value extrapolations are not phase boundaries.
Binary VLLE uses the existing LLE-aware bubble solver. A heteroazeotropic
section joins the two coexisting-liquid endpoints at one temperature; its
vapor composition lies on the dew envelope at that same validated equilibrium.
Curve resolution is distributed over both stable VLE branches, including
narrow water-rich branches. Binodal phases are ordered by component-1 mole
fraction and sampled below the boiling boundary. A missing sampled split is
not a proof that no miscibility gap exists.
SVG exports include the mixture, model, conditions, scope, legend, and the
selected theme's background, so the downloaded figure remains interpretable.

## Storage and proxy configuration

`PFDSIM_WEB_DATA` overrides the private web-data directory, which otherwise
lives under the current account's `.local/share/pfdsim/web`. It contains the
SQLite account/autosave/job database, a persistent session secret, and worker
logs. Keep that directory when restarting or upgrading to retain accounts,
quotas, and account saves. `SECRET_KEY` can supply a deployment-managed secret
instead of the automatically generated one.

Behind a trusted reverse proxy, set `PFDSIM_PROXY_HOPS` to the number of trusted
proxy hops (normally `1` for a single local nginx proxy), and have the proxy
set/append `X-Forwarded-For` and set `X-Forwarded-Proto`. The default `0` ignores
forwarded headers. For HTTPS deployment, set `PFDSIM_SECURE_COOKIES=1`. This keeps
IP quota identity and secure session cookies consistent with the deployment.

## Verification

Focused automated checks:

```sh
uv run --extra web python -m pytest tests/test_web.py tests/test_web_accounts.py tests/test_phase_diagrams.py
```

Opt-in browser coverage needs Playwright and an installed Chrome/Chromium in
the Python environment used to run the script:

```sh
python tests/browser_web.py --gunicorn
python tests/browser_configuration.py --gunicorn
python tests/browser_review.py
python tests/browser_persistence.py
python tests/browser_binary_vlle.py
```

The browser check owns a private preview and stops its processes afterward.
Its reports and screenshots are retained under `/tmp`.
