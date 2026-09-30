# Activity-model source data

This folder contains the NRTL/UNIQUAC literature collections, regression
observations and reports, structural parameters, and raw interaction tables.
`legacy/` retains the ChemSep-ID source collections used by the CAS builder.

The maintained fitting and preparation scripts are in
`scripts/activity_fitting/`. The shared interaction builder is
`scripts/build_cas_interaction_parameters.py`.

Historical fit collections retain their published structural-basis evidence.
The builder activates the reviewed ordinary-basis UNIQUAC replacements from
`uniquac_common_basis_refits.json`; the generated runtime tables remain in
`data/`.
