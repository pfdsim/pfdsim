# Held-out solvent-bias correction versus solvent-only transfer

## Protocol

- Main correction: log10(x_corrected) = log10(x_UNIFAC) − (b_target − b_reference). Fit one constant log bias per solvent using training molecule prediction errors. Solvent offsets have an arbitrary gauge; only differences are used.
- Sensitivity: subtract the target solvent’s mean signed error only. Average within each training molecule/target first, then over molecules.
- Fit every identifiable solvent coefficient within each training fold. Do not select coefficients using the previous full-dataset significance analysis.
- Reuse the previous five-fold chemical-identity splits, repeated ten times. All temperatures, publications and solvents of each held-out molecule are excluded from fitting.
- Domain-matched baseline: fit log10(x_target/x_reference) using the SAME training transfers, chemical coverage, equal-molecule weights and solvent design matrix as the relative-bias correction. Thus it is a direct test of whether UNIFAC adds useful molecule-dependent information beyond fitted solvent averages.
- Also report the original broader-population mean-multiplier baseline, with its saved held-out predictions. That baseline used equal molecule/group/pair weights and all eligible training compounds; it is not the principal controlled comparison.
- Corrections and matched baseline use least squares with equal total weight per molecule. Each available directed transfer within a molecule has equal weight. No ridge penalty or hyperparameter search is used.
- Use saved reported UNIFAC predictions, including the physical x≤1 bound. After correction, cap x again at 1. This is statistical prediction calibration; activity coefficients and phase-equilibrium models are not modified or re-solved.
- Infinite-dilution and finite dilute scenarios use both observed mole fractions <0.05 or <0.01. This is retrospective selection and cannot use an unknown target’s measured solubility prospectively.
- Unseen solvents or disconnected training solvent components are unavailable. Comparisons require all ten repeats to be supported and use identical cases within each table row.
- Primary metric: mean absolute log10 error, first averaged within each molecule and then equally over molecules. Score each split’s prediction before averaging losses; do not ensemble predictions before scoring.
- Resample whole test molecules and split repetitions 20,000 times, seed 20261014. Simultaneous intervals account for 60 primary corrected-versus-matched-baseline comparisons (six variants × five prespecified scenarios × two correction forms). Negative differences favor corrected UNIFAC.
- Shared-case cohorts and comparisons to the broad baseline/uncorrected model are additional diagnostics; their displayed intervals use the same conservative family size. They are not an additional globally adjusted family of all exploratory comparisons.
- Confidence intervals condition on the fitted cross-validation predictions and do not prove parameter-training disjointness or independence between chemical scaffolds.

## finite_all: relative solvent correction, full usable coverage

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 39,680 | 254 | 0.5430 | 0.5412 | 0.5389 | 0.5320 | +0.0023 | -0.0772 to +0.0919 |
| UNIFAC2 | 73,476 | 345 | 0.5996 | 0.6058 | 0.6298 | 0.6279 | -0.0240 | -0.1064 to +0.0598 |
| UNIFDMD | 32,299 | 235 | 0.5303 | 0.5471 | 0.5103 | 0.5239 | +0.0368 | -0.0695 to +0.1574 |
| UNIFM2 | 73,079 | 350 | 0.6299 | 0.6208 | 0.6314 | 0.6265 | -0.0106 | -0.0985 to +0.0786 |
| UNIFNIST | 23,476 | 179 | 0.5749 | 0.5889 | 0.5663 | 0.5494 | +0.0226 | -0.0900 to +0.1667 |
| UNIFLBY | 13,514 | 118 | 0.4360 | 0.4653 | 0.4945 | 0.5071 | -0.0292 | -0.1370 to +0.0909 |

## finite_all: relative solvent correction, common_all_six

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 7,075 | 85 | 0.3756 | 0.4669 | 0.5227 | 0.5212 | -0.0558 | -0.1841 to +0.0650 |
| UNIFAC2 | 7,075 | 85 | 0.3782 | 0.4900 | 0.5108 | 0.5212 | -0.0208 | -0.1459 to +0.1099 |
| UNIFDMD | 7,075 | 85 | 0.3994 | 0.4712 | 0.4880 | 0.5212 | -0.0168 | -0.1518 to +0.1384 |
| UNIFM2 | 7,075 | 85 | 0.3836 | 0.4922 | 0.5020 | 0.5212 | -0.0098 | -0.1411 to +0.1387 |
| UNIFNIST | 7,075 | 85 | 0.4697 | 0.4721 | 0.5201 | 0.5212 | -0.0480 | -0.1698 to +0.0934 |
| UNIFLBY | 7,075 | 85 | 0.3698 | 0.4139 | 0.4868 | 0.5212 | -0.0729 | -0.2058 to +0.0806 |

## finite_all: relative solvent correction, common_five

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 16,249 | 144 | 0.3671 | 0.4315 | 0.5421 | 0.5370 | -0.1106 | -0.2482 to -0.0065 |
| UNIFAC2 | 16,249 | 144 | 0.3692 | 0.4528 | 0.5425 | 0.5370 | -0.0897 | -0.2363 to +0.0173 |
| UNIFDMD | 16,249 | 144 | 0.4378 | 0.4882 | 0.5223 | 0.5370 | -0.0341 | -0.1970 to +0.1288 |
| UNIFM2 | 16,249 | 144 | 0.3830 | 0.4671 | 0.5349 | 0.5370 | -0.0678 | -0.2126 to +0.0455 |
| UNIFNIST | 16,249 | 144 | 0.5136 | 0.5205 | 0.5501 | 0.5370 | -0.0295 | -0.1956 to +0.1482 |

## finite_all: target-only correction sensitivity

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 39,680 | 254 | 0.5430 | 0.5419 | 0.5389 | 0.5320 | +0.0030 | -0.0861 to +0.1028 |
| UNIFAC2 | 73,476 | 345 | 0.5996 | 0.5963 | 0.6298 | 0.6279 | -0.0335 | -0.1112 to +0.0445 |
| UNIFDMD | 32,299 | 235 | 0.5303 | 0.5344 | 0.5103 | 0.5239 | +0.0242 | -0.0799 to +0.1412 |
| UNIFM2 | 73,079 | 350 | 0.6299 | 0.6178 | 0.6314 | 0.6265 | -0.0136 | -0.0962 to +0.0761 |
| UNIFNIST | 23,476 | 179 | 0.5749 | 0.5633 | 0.5663 | 0.5494 | -0.0031 | -0.1155 to +0.1441 |
| UNIFLBY | 13,514 | 118 | 0.4360 | 0.4535 | 0.4945 | 0.5071 | -0.0410 | -0.1394 to +0.0727 |

## finite_0.05: relative solvent correction, full usable coverage

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 21,025 | 192 | 0.6005 | 0.6264 | 0.5164 | 0.5072 | +0.1100 | +0.0011 to +0.2444 |
| UNIFAC2 | 41,724 | 281 | 0.6559 | 0.6376 | 0.5913 | 0.5935 | +0.0463 | -0.0434 to +0.1430 |
| UNIFDMD | 17,141 | 182 | 0.6045 | 0.6360 | 0.4675 | 0.4863 | +0.1686 | +0.0402 to +0.3209 |
| UNIFM2 | 41,098 | 282 | 0.6937 | 0.6645 | 0.5816 | 0.5833 | +0.0830 | -0.0109 to +0.1775 |
| UNIFNIST | 10,818 | 131 | 0.6748 | 0.7065 | 0.5479 | 0.5259 | +0.1586 | +0.0145 to +0.3509 |
| UNIFLBY | 7,424 | 92 | 0.4979 | 0.5165 | 0.4962 | 0.5000 | +0.0203 | -0.1189 to +0.1677 |

## finite_0.05: relative solvent correction, common_all_six

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 2,979 | 60 | 0.4312 | 0.5360 | 0.5286 | 0.5510 | +0.0075 | -0.1483 to +0.1698 |
| UNIFAC2 | 2,979 | 60 | 0.4421 | 0.5363 | 0.5193 | 0.5510 | +0.0170 | -0.1421 to +0.1735 |
| UNIFDMD | 2,979 | 60 | 0.4762 | 0.5586 | 0.5123 | 0.5510 | +0.0463 | -0.1470 to +0.2413 |
| UNIFM2 | 2,979 | 60 | 0.4569 | 0.5699 | 0.5155 | 0.5510 | +0.0544 | -0.1168 to +0.2472 |
| UNIFNIST | 2,979 | 60 | 0.5504 | 0.5441 | 0.5395 | 0.5510 | +0.0046 | -0.1626 to +0.1804 |
| UNIFLBY | 2,979 | 60 | 0.4482 | 0.4575 | 0.5176 | 0.5510 | -0.0602 | -0.2229 to +0.1232 |

## finite_0.05: relative solvent correction, common_five

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 7,071 | 101 | 0.4048 | 0.4855 | 0.4947 | 0.5051 | -0.0092 | -0.1234 to +0.1022 |
| UNIFAC2 | 7,071 | 101 | 0.4209 | 0.4835 | 0.4987 | 0.5051 | -0.0152 | -0.1243 to +0.0870 |
| UNIFDMD | 7,071 | 101 | 0.5206 | 0.5714 | 0.4865 | 0.5051 | +0.0849 | -0.0674 to +0.2951 |
| UNIFM2 | 7,071 | 101 | 0.4463 | 0.5299 | 0.4924 | 0.5051 | +0.0375 | -0.0843 to +0.1632 |
| UNIFNIST | 7,071 | 101 | 0.5927 | 0.6047 | 0.5100 | 0.5051 | +0.0948 | -0.0559 to +0.3233 |

## finite_0.05: target-only correction sensitivity

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 21,025 | 192 | 0.6005 | 0.6145 | 0.5164 | 0.5072 | +0.0981 | -0.0097 to +0.2402 |
| UNIFAC2 | 41,724 | 281 | 0.6559 | 0.6440 | 0.5913 | 0.5935 | +0.0527 | -0.0353 to +0.1507 |
| UNIFDMD | 17,141 | 182 | 0.6045 | 0.6133 | 0.4675 | 0.4863 | +0.1459 | +0.0266 to +0.2937 |
| UNIFM2 | 41,098 | 282 | 0.6937 | 0.6795 | 0.5816 | 0.5833 | +0.0979 | +0.0077 to +0.1948 |
| UNIFNIST | 10,818 | 131 | 0.6748 | 0.6627 | 0.5479 | 0.5259 | +0.1149 | -0.0219 to +0.3009 |
| UNIFLBY | 7,424 | 92 | 0.4979 | 0.5296 | 0.4962 | 0.5000 | +0.0334 | -0.0870 to +0.1631 |

## infinite_0.05: relative solvent correction, full usable coverage

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 21,025 | 192 | 0.6108 | 0.6419 | 0.5164 | 0.5072 | +0.1255 | +0.0129 to +0.2615 |
| UNIFAC2 | 41,724 | 281 | 0.6687 | 0.6474 | 0.5913 | 0.5935 | +0.0562 | -0.0362 to +0.1563 |
| UNIFDMD | 17,141 | 182 | 0.5925 | 0.6274 | 0.4675 | 0.4863 | +0.1599 | +0.0323 to +0.3120 |
| UNIFM2 | 41,098 | 282 | 0.6942 | 0.6631 | 0.5816 | 0.5833 | +0.0815 | -0.0105 to +0.1753 |
| UNIFNIST | 10,818 | 131 | 0.7272 | 0.7559 | 0.5479 | 0.5259 | +0.2081 | +0.0660 to +0.3960 |
| UNIFLBY | 7,424 | 92 | 0.4763 | 0.4960 | 0.4962 | 0.5000 | -0.0003 | -0.1335 to +0.1342 |

## infinite_0.05: relative solvent correction, common_all_six

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 2,979 | 60 | 0.4158 | 0.5310 | 0.5286 | 0.5510 | +0.0024 | -0.1495 to +0.1556 |
| UNIFAC2 | 2,979 | 60 | 0.4414 | 0.5364 | 0.5193 | 0.5510 | +0.0171 | -0.1385 to +0.1717 |
| UNIFDMD | 2,979 | 60 | 0.4557 | 0.5395 | 0.5123 | 0.5510 | +0.0272 | -0.1719 to +0.2190 |
| UNIFM2 | 2,979 | 60 | 0.4459 | 0.5515 | 0.5155 | 0.5510 | +0.0360 | -0.1351 to +0.2207 |
| UNIFNIST | 2,979 | 60 | 0.5864 | 0.5672 | 0.5395 | 0.5510 | +0.0277 | -0.1385 to +0.1930 |
| UNIFLBY | 2,979 | 60 | 0.4302 | 0.4389 | 0.5176 | 0.5510 | -0.0787 | -0.2386 to +0.1025 |

## infinite_0.05: relative solvent correction, common_five

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 7,071 | 101 | 0.4005 | 0.4889 | 0.4947 | 0.5051 | -0.0059 | -0.1230 to +0.1019 |
| UNIFAC2 | 7,071 | 101 | 0.4262 | 0.4855 | 0.4987 | 0.5051 | -0.0131 | -0.1228 to +0.0929 |
| UNIFDMD | 7,071 | 101 | 0.5196 | 0.5675 | 0.4865 | 0.5051 | +0.0809 | -0.0753 to +0.2844 |
| UNIFM2 | 7,071 | 101 | 0.4464 | 0.5221 | 0.4924 | 0.5051 | +0.0296 | -0.0892 to +0.1552 |
| UNIFNIST | 7,071 | 101 | 0.6333 | 0.6309 | 0.5100 | 0.5051 | +0.1209 | -0.0301 to +0.3428 |

## infinite_0.05: target-only correction sensitivity

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 21,025 | 192 | 0.6108 | 0.6268 | 0.5164 | 0.5072 | +0.1105 | -0.0008 to +0.2535 |
| UNIFAC2 | 41,724 | 281 | 0.6687 | 0.6578 | 0.5913 | 0.5935 | +0.0665 | -0.0229 to +0.1648 |
| UNIFDMD | 17,141 | 182 | 0.5925 | 0.6026 | 0.4675 | 0.4863 | +0.1352 | +0.0143 to +0.2832 |
| UNIFM2 | 41,098 | 282 | 0.6942 | 0.6820 | 0.5816 | 0.5833 | +0.1004 | +0.0092 to +0.2010 |
| UNIFNIST | 10,818 | 131 | 0.7272 | 0.7224 | 0.5479 | 0.5259 | +0.1746 | +0.0336 to +0.3612 |
| UNIFLBY | 7,424 | 92 | 0.4763 | 0.5127 | 0.4962 | 0.5000 | +0.0165 | -0.0978 to +0.1347 |

## finite_0.01: relative solvent correction, full usable coverage

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 9,357 | 126 | 0.7557 | 0.8264 | 0.5620 | 0.5469 | +0.2644 | +0.0987 to +0.4588 |
| UNIFAC2 | 20,194 | 199 | 0.8469 | 0.8638 | 0.6113 | 0.6020 | +0.2525 | +0.1044 to +0.4160 |
| UNIFDMD | 7,522 | 111 | 0.7713 | 0.8247 | 0.5032 | 0.5357 | +0.3215 | +0.1252 to +0.5668 |
| UNIFM2 | 19,458 | 199 | 0.8552 | 0.8311 | 0.6032 | 0.5925 | +0.2279 | +0.0872 to +0.3679 |
| UNIFNIST | 4,661 | 81 | 0.7935 | 0.8699 | 0.5820 | 0.5694 | +0.2880 | +0.0658 to +0.5867 |
| UNIFLBY | 3,033 | 55 | 0.6768 | 0.7410 | 0.5900 | 0.5689 | +0.1510 | -0.0792 to +0.3842 |

## finite_0.01: relative solvent correction, common_all_six

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 1,238 | 32 | 0.5312 | 0.7898 | 0.6113 | 0.6709 | +0.1785 | -0.0732 to +0.4370 |
| UNIFAC2 | 1,238 | 32 | 0.6164 | 0.8405 | 0.6108 | 0.6709 | +0.2296 | -0.0741 to +0.5486 |
| UNIFDMD | 1,238 | 32 | 0.5735 | 0.7461 | 0.5688 | 0.6709 | +0.1774 | -0.1415 to +0.5404 |
| UNIFM2 | 1,238 | 32 | 0.6147 | 0.7554 | 0.6048 | 0.6709 | +0.1507 | -0.1311 to +0.4522 |
| UNIFNIST | 1,238 | 32 | 0.6027 | 0.6459 | 0.5916 | 0.6709 | +0.0543 | -0.1842 to +0.3250 |
| UNIFLBY | 1,238 | 32 | 0.6285 | 0.6694 | 0.6406 | 0.6709 | +0.0288 | -0.3506 to +0.3869 |

## finite_0.01: relative solvent correction, common_five

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 2,919 | 58 | 0.5001 | 0.6564 | 0.5575 | 0.5887 | +0.0990 | -0.0852 to +0.3043 |
| UNIFAC2 | 2,919 | 58 | 0.5732 | 0.6941 | 0.5678 | 0.5887 | +0.1263 | -0.0722 to +0.3444 |
| UNIFDMD | 2,919 | 58 | 0.6521 | 0.7896 | 0.5255 | 0.5887 | +0.2641 | -0.0061 to +0.6371 |
| UNIFM2 | 2,919 | 58 | 0.6040 | 0.7131 | 0.5617 | 0.5887 | +0.1514 | -0.0370 to +0.3574 |
| UNIFNIST | 2,919 | 58 | 0.7128 | 0.7706 | 0.5618 | 0.5887 | +0.2088 | -0.0532 to +0.5945 |

## finite_0.01: target-only correction sensitivity

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 9,357 | 126 | 0.7557 | 0.8022 | 0.5620 | 0.5469 | +0.2403 | +0.0766 to +0.4343 |
| UNIFAC2 | 20,194 | 199 | 0.8469 | 0.8231 | 0.6113 | 0.6020 | +0.2118 | +0.0789 to +0.3585 |
| UNIFDMD | 7,522 | 111 | 0.7713 | 0.8067 | 0.5032 | 0.5357 | +0.3035 | +0.1215 to +0.5406 |
| UNIFM2 | 19,458 | 199 | 0.8552 | 0.8317 | 0.6032 | 0.5925 | +0.2284 | +0.0965 to +0.3765 |
| UNIFNIST | 4,661 | 81 | 0.7935 | 0.8090 | 0.5820 | 0.5694 | +0.2270 | +0.0045 to +0.5148 |
| UNIFLBY | 3,033 | 55 | 0.6768 | 0.7600 | 0.5900 | 0.5689 | +0.1701 | -0.0295 to +0.3813 |

## infinite_0.01: relative solvent correction, full usable coverage

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 9,357 | 126 | 0.7490 | 0.8242 | 0.5620 | 0.5469 | +0.2622 | +0.0991 to +0.4577 |
| UNIFAC2 | 20,194 | 199 | 0.8395 | 0.8466 | 0.6113 | 0.6020 | +0.2353 | +0.0965 to +0.3910 |
| UNIFDMD | 7,522 | 111 | 0.7508 | 0.8057 | 0.5032 | 0.5357 | +0.3025 | +0.1064 to +0.5496 |
| UNIFM2 | 19,458 | 199 | 0.8503 | 0.8239 | 0.6032 | 0.5925 | +0.2207 | +0.0813 to +0.3600 |
| UNIFNIST | 4,661 | 81 | 0.7917 | 0.8660 | 0.5820 | 0.5694 | +0.2841 | +0.0642 to +0.5845 |
| UNIFLBY | 3,033 | 55 | 0.6547 | 0.7151 | 0.5900 | 0.5689 | +0.1251 | -0.1034 to +0.3436 |

## infinite_0.01: relative solvent correction, common_all_six

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 1,238 | 32 | 0.5294 | 0.7925 | 0.6113 | 0.6709 | +0.1812 | -0.0775 to +0.4450 |
| UNIFAC2 | 1,238 | 32 | 0.6204 | 0.8373 | 0.6108 | 0.6709 | +0.2265 | -0.0905 to +0.5719 |
| UNIFDMD | 1,238 | 32 | 0.5551 | 0.7157 | 0.5688 | 0.6709 | +0.1469 | -0.1705 to +0.4886 |
| UNIFM2 | 1,238 | 32 | 0.6017 | 0.7287 | 0.6048 | 0.6709 | +0.1240 | -0.1514 to +0.4267 |
| UNIFNIST | 1,238 | 32 | 0.5962 | 0.6351 | 0.5916 | 0.6709 | +0.0435 | -0.1923 to +0.3079 |
| UNIFLBY | 1,238 | 32 | 0.6077 | 0.6486 | 0.6406 | 0.6709 | +0.0080 | -0.3735 to +0.3630 |

## infinite_0.01: relative solvent correction, common_five

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 2,919 | 58 | 0.4999 | 0.6592 | 0.5575 | 0.5887 | +0.1018 | -0.0833 to +0.3135 |
| UNIFAC2 | 2,919 | 58 | 0.5816 | 0.6976 | 0.5678 | 0.5887 | +0.1297 | -0.0797 to +0.3585 |
| UNIFDMD | 2,919 | 58 | 0.6419 | 0.7698 | 0.5255 | 0.5887 | +0.2443 | -0.0193 to +0.6232 |
| UNIFM2 | 2,919 | 58 | 0.5977 | 0.6958 | 0.5617 | 0.5887 | +0.1341 | -0.0516 to +0.3454 |
| UNIFNIST | 2,919 | 58 | 0.7027 | 0.7534 | 0.5618 | 0.5887 | +0.1916 | -0.0662 to +0.5742 |

## infinite_0.01: target-only correction sensitivity

| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UNIFAC | 9,357 | 126 | 0.7490 | 0.7981 | 0.5620 | 0.5469 | +0.2361 | +0.0760 to +0.4263 |
| UNIFAC2 | 20,194 | 199 | 0.8395 | 0.8096 | 0.6113 | 0.6020 | +0.1983 | +0.0756 to +0.3373 |
| UNIFDMD | 7,522 | 111 | 0.7508 | 0.7865 | 0.5032 | 0.5357 | +0.2833 | +0.1049 to +0.5192 |
| UNIFM2 | 19,458 | 199 | 0.8503 | 0.8259 | 0.6032 | 0.5925 | +0.2227 | +0.0908 to +0.3712 |
| UNIFNIST | 4,661 | 81 | 0.7917 | 0.8125 | 0.5820 | 0.5694 | +0.2305 | +0.0071 to +0.5072 |
| UNIFLBY | 3,033 | 55 | 0.6547 | 0.7369 | 0.5900 | 0.5689 | +0.1469 | -0.0441 to +0.3480 |

## Numerical checks

```json
{
  "solvent_only_UNIFAC_reduces_to_matched_baseline": true,
  "held_out_error_perturbation_does_not_change_fit": true,
  "known_bias_recovered": true,
  "unseen_solvent_unavailable": true,
  "maximum_synthetic_residual": 3.885780586188048e-16
}
```

## Interpretation limits

Coverage remains variant-dependent. Own-coverage rows use different compounds across variants; shared-case tables isolate that difference. Training-domain matching is necessary because the original broad baseline can be miscalibrated on a narrower UNIFAC-supported chemical population. Unknown pH, apparent versus intrinsic aqueous solubility, solid forms and possible overlap with historical UNIFAC parameter-training data remain limitations. Calibration is evaluated on other molecules from the same dataset, not on a separately collected external study.

## Files

- One .npz per scenario/variant stores every held-out corrected and matched-baseline error, with case IDs.
- *_coefficients.csv stores every fold’s relative correction, target-only correction and matched-baseline solvent factors.
- summary.json contains MAE, RMSE, signed bias, fold error, factor-2/10 accuracy and paired intervals.
- audit.json records fold training/support counts and normal-equation residuals.
- manifest.json stores portable source/input fingerprints. Previous results, datasets and production code are unchanged.
