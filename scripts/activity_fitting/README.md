# NRTL and UNIQUAC fitting

This folder contains binary-parameter regressions, source-data preparation,
structural-parameter collection, and diagnostics used to assess the fits.
Source tables and fit reports live in `data/source/activity_fitting/`.

Run scripts from the repository root, for example:

```sh
python scripts/activity_fitting/refit_uniquac_common_basis.py --help
python scripts/activity_fitting/build_uniquac_rq_parameters.py
python scripts/build_cas_interaction_parameters.py
```

The shared CAS interaction builder remains in `scripts/` because it also builds
EOS interactions. These two builders own the generated runtime JSON tables.
The fitting scripts write source collections only when their CLI requests it.

Historical source-basis parameters remain available for reproducing published
fits. Runtime UNIQUAC uses the ordinary database `r/q` basis; it does not switch
to a published UNIFAC structural basis. The common-basis refits replace the
five corresponding historical UNIQUAC entries during the interaction build.
