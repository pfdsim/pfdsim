# Pinned public ESOL reference

Files `esol.py`, `delaney.csv`, and `LICENSE` were downloaded without modification
from Patrick Walters's [solubility repository](https://github.com/PatWalters/solubility)
at revision `9e0e55fd7040575e8427079c44759d51dc7c2216`, on 2026-10-23.
`source.json` records the repository, revision, and original paper DOI.

The source implementation is distributed under the included MIT license.
The public Delaney table accompanies the repository's RDKit coefficient-refit
routine; it is retained to audit calibration/evaluation structure overlap.
It is not used to refit a model during this investigation.

The maintained [benchmark](../../scripts/benchmark_esol.py) loads only the
`ESOLCalculator` class from the pinned source using Python's syntax tree.
This preserves the exact equations and descriptor implementation without
executing the upstream training/demo code or importing its unused scikit-learn
dependency. The benchmark is offline once these reference files are present.

The RDKit-refitted fixed coefficients are the main comparison. The original
paper's coefficients with RDKit descriptors are a separately labeled sensitivity
check: they are not an exact recreation of Delaney's Daylight-based model.

Reference: Delaney, J. S. (2004), *ESOL: Estimating Aqueous Solubility Directly
from Molecular Structure*, J. Chem. Inf. Comput. Sci. 44, 1000–1005,
[doi:10.1021/ci034243x](https://doi.org/10.1021/ci034243x).
