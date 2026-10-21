# Infinite-dilution solubility transfer

Predict x_target = x_reference gamma_infinity_reference / gamma_infinity_target, with both gamma values evaluated at zero solute mole fraction and the measured temperature. No activity-equilibrium root is solved.

## Protocol

- Interpret 0.05 and 0.01 as mole fractions (5 mol% and 1 mol%), consistent with the preceding assessment.
- Apply the strict cutoff to BOTH observed reference and observed target solubilities. This is a retrospective dilute-subset test; the unknown target’s solubility cannot be used to select cases prospectively.
- Do not filter out predictions above the cutoff. Cap only at the physical upper bound x=1; retain uncapped error metrics as a sensitivity.
- Reuse the previous exact-temperature/publication pairing, deduplication, strict fragmentation and complete interaction requirement.
- Main comparisons use identical cases where both finite and infinite predictions are available. Infinite availability before requiring finite success is also recorded in summary.json.
- Use pfdsim CompiledUNIFACBackend at [0,1], rather than approximating the limit by an arbitrary small concentration. Verify all six variants against the readable implementation and near-zero concentration.
- Average errors equally over molecules, as before. Positive infinite-minus-finite MAE differences mean the infinite approximation is worse.
- Bootstrap whole molecules 10,000 times with seed 20261014. Ordinary paired 95% intervals and simultaneous 95% intervals across six variants at each cutoff are saved.
- Reuse the existing 10-repeat, five-fold molecule-held-out multiplier errors. No baseline is refitted on the test subsets.
- The fixed cutoff selects different compounds, so compare finite/infinite within each cutoff, not across cutoffs. The preceding solid-form, pH/speciation and parameter-training-overlap limitations remain.

## Both measured mole fractions < 0.05: own_coverage

| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Difference | Paired 95% CI | Infinite RMSE | Finite RMSE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| UNIFAC | 22,636 | 196 | 0.6057 | 0.6157 | +0.0101 | -0.0042 to +0.0265 | 1.1287 | 1.1181 |
| UNIFAC2 | 43,758 | 287 | 0.6515 | 0.6629 | +0.0115 | -0.0007 to +0.0228 | 1.1247 | 1.1140 |
| UNIFDMD | 18,587 | 184 | 0.6031 | 0.5915 | -0.0117 | -0.0309 to +0.0065 | 1.0619 | 1.0726 |
| UNIFM2 | 43,138 | 288 | 0.6839 | 0.6831 | -0.0008 | -0.0111 to +0.0099 | 1.1398 | 1.1300 |
| UNIFNIST | 12,540 | 133 | 0.6904 | 0.7406 | +0.0503 | +0.0157 to +0.0888 | 1.2219 | 1.1420 |
| UNIFLBY | 7,897 | 93 | 0.4958 | 0.4743 | -0.0215 | -0.0356 to -0.0101 | 0.8624 | 0.8947 |

## Both measured mole fractions < 0.05: common_all_six

| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Difference | Paired 95% CI | Infinite RMSE | Finite RMSE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| UNIFAC | 3,357 | 60 | 0.4515 | 0.4364 | -0.0151 | -0.0274 to -0.0048 | 0.7059 | 0.7247 |
| UNIFAC2 | 3,357 | 60 | 0.4500 | 0.4487 | -0.0013 | -0.0068 to +0.0037 | 0.7145 | 0.7122 |
| UNIFDMD | 3,357 | 60 | 0.4766 | 0.4554 | -0.0212 | -0.0434 to -0.0036 | 0.7862 | 0.8265 |
| UNIFM2 | 3,357 | 60 | 0.4676 | 0.4553 | -0.0122 | -0.0243 to -0.0020 | 0.7719 | 0.7927 |
| UNIFNIST | 3,357 | 60 | 0.5545 | 0.5879 | +0.0333 | -0.0025 to +0.0720 | 0.9118 | 0.8706 |
| UNIFLBY | 3,357 | 60 | 0.4501 | 0.4338 | -0.0162 | -0.0275 to -0.0063 | 0.7610 | 0.7837 |

## Both measured mole fractions < 0.05: common_five

| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Difference | Paired 95% CI | Infinite RMSE | Finite RMSE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| UNIFAC | 8,276 | 101 | 0.4173 | 0.4146 | -0.0027 | -0.0128 to +0.0072 | 0.6761 | 0.6814 |
| UNIFAC2 | 8,276 | 101 | 0.4278 | 0.4330 | +0.0051 | -0.0012 to +0.0130 | 0.7076 | 0.6955 |
| UNIFDMD | 8,276 | 101 | 0.5175 | 0.5171 | -0.0004 | -0.0192 to +0.0188 | 0.9426 | 0.9450 |
| UNIFM2 | 8,276 | 101 | 0.4539 | 0.4534 | -0.0005 | -0.0114 to +0.0104 | 0.7696 | 0.7758 |
| UNIFNIST | 8,276 | 101 | 0.5949 | 0.6370 | +0.0421 | +0.0098 to +0.0770 | 1.0881 | 1.0457 |

## Cutoff 0.05: directional breakdown

| Variant | Direction | Transfers | Molecules | Finite MAE | Infinite MAE | Difference |
|---|---|---:|---:|---:|---:|---:|
| UNIFAC | water_to_nonwater | 684 | 47 | 1.1196 | 1.1291 | +0.0095 |
| UNIFAC | nonwater_to_water | 871 | 61 | 1.9406 | 1.8873 | -0.0533 |
| UNIFAC | nonwater_to_nonwater | 21,081 | 187 | 0.5033 | 0.5168 | +0.0135 |
| UNIFAC2 | water_to_nonwater | 1,579 | 89 | 1.1457 | 1.1477 | +0.0020 |
| UNIFAC2 | nonwater_to_water | 1,975 | 111 | 1.8198 | 1.7906 | -0.0292 |
| UNIFAC2 | nonwater_to_nonwater | 40,204 | 274 | 0.5403 | 0.5580 | +0.0176 |
| UNIFDMD | water_to_nonwater | 476 | 32 | 0.9001 | 0.8997 | -0.0004 |
| UNIFDMD | nonwater_to_water | 697 | 49 | 1.6303 | 1.6243 | -0.0061 |
| UNIFDMD | nonwater_to_nonwater | 17,414 | 179 | 0.5327 | 0.5220 | -0.0107 |
| UNIFM2 | water_to_nonwater | 1,451 | 79 | 1.0900 | 1.1025 | +0.0125 |
| UNIFM2 | nonwater_to_water | 1,974 | 108 | 1.8277 | 1.8218 | -0.0059 |
| UNIFM2 | nonwater_to_nonwater | 39,713 | 276 | 0.5828 | 0.5815 | -0.0012 |
| UNIFNIST | water_to_nonwater | 325 | 24 | 1.0843 | 1.1730 | +0.0887 |
| UNIFNIST | nonwater_to_water | 452 | 33 | 1.5985 | 1.6217 | +0.0232 |
| UNIFNIST | nonwater_to_nonwater | 11,763 | 128 | 0.6171 | 0.6748 | +0.0576 |
| UNIFLBY | water_to_nonwater | 172 | 13 | 0.7308 | 0.7444 | +0.0136 |
| UNIFLBY | nonwater_to_water | 262 | 21 | 1.7792 | 1.7725 | -0.0067 |
| UNIFLBY | nonwater_to_nonwater | 7,463 | 90 | 0.4462 | 0.4247 | -0.0215 |

## Cutoff 0.05: multiplier baseline on each variant’s paired cases

| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Multiplier MAE |
|---|---:|---:|---:|---:|---:|
| UNIFAC | 21,792 | 195 | 0.6057 | 0.6152 | 0.5096 |
| UNIFAC2 | 42,663 | 285 | 0.6540 | 0.6651 | 0.5928 |
| UNIFDMD | 17,846 | 182 | 0.6037 | 0.5915 | 0.4905 |
| UNIFM2 | 42,086 | 286 | 0.6884 | 0.6877 | 0.5833 |
| UNIFNIST | 12,042 | 133 | 0.6905 | 0.7404 | 0.5297 |
| UNIFLBY | 7,677 | 92 | 0.4940 | 0.4723 | 0.5008 |

## Cutoff 0.05: pooled_publications, own coverage

| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Difference | Paired 95% CI | Infinite RMSE | Finite RMSE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| UNIFAC | 24,975 | 197 | 0.6027 | 0.6106 | +0.0078 | -0.0069 to +0.0248 | 1.0835 | 1.0751 |
| UNIFAC2 | 47,074 | 288 | 0.6529 | 0.6640 | +0.0111 | -0.0014 to +0.0227 | 1.1199 | 1.1090 |
| UNIFDMD | 20,439 | 186 | 0.6125 | 0.5979 | -0.0146 | -0.0336 to +0.0031 | 1.0621 | 1.0757 |
| UNIFM2 | 46,406 | 290 | 0.6918 | 0.6910 | -0.0007 | -0.0114 to +0.0100 | 1.1473 | 1.1379 |
| UNIFNIST | 13,875 | 135 | 0.6954 | 0.7440 | +0.0486 | +0.0151 to +0.0877 | 1.2186 | 1.1407 |
| UNIFLBY | 9,478 | 94 | 0.4964 | 0.4737 | -0.0227 | -0.0372 to -0.0106 | 0.8514 | 0.8853 |


## Cutoff 0.05: AqSolDB_approximate_water_anchor, own coverage

| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Difference | Paired 95% CI | Infinite RMSE | Finite RMSE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| UNIFAC | 367 | 56 | 1.0243 | 0.9955 | -0.0288 | -0.0512 to -0.0103 | 1.2877 | 1.3008 |
| UNIFAC2 | 528 | 73 | 0.8696 | 0.8631 | -0.0065 | -0.0322 to +0.0178 | 1.2761 | 1.2610 |
| UNIFDMD | 275 | 42 | 1.0089 | 0.9116 | -0.0973 | -0.1671 to -0.0374 | 1.1889 | 1.3271 |
| UNIFM2 | 508 | 72 | 0.9169 | 0.8810 | -0.0358 | -0.0771 to +0.0086 | 1.2547 | 1.2528 |
| UNIFNIST | 199 | 27 | 1.0372 | 0.9993 | -0.0379 | -0.1433 to +0.0518 | 1.1840 | 1.2100 |
| UNIFLBY | 169 | 27 | 0.5966 | 0.5671 | -0.0295 | -0.0825 to +0.0033 | 0.7410 | 0.7854 |

## Both measured mole fractions < 0.01: own_coverage

| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Difference | Paired 95% CI | Infinite RMSE | Finite RMSE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| UNIFAC | 10,095 | 128 | 0.7612 | 0.7556 | -0.0055 | -0.0191 to +0.0052 | 1.3741 | 1.3763 |
| UNIFAC2 | 21,129 | 203 | 0.8312 | 0.8231 | -0.0082 | -0.0296 to +0.0083 | 1.3485 | 1.3637 |
| UNIFDMD | 8,173 | 115 | 0.7597 | 0.7403 | -0.0194 | -0.0336 to -0.0074 | 1.3251 | 1.3455 |
| UNIFM2 | 20,412 | 204 | 0.8443 | 0.8387 | -0.0056 | -0.0154 to +0.0041 | 1.3529 | 1.3493 |
| UNIFNIST | 5,618 | 82 | 0.7910 | 0.7886 | -0.0024 | -0.0325 to +0.0328 | 1.3153 | 1.3094 |
| UNIFLBY | 3,397 | 57 | 0.6603 | 0.6393 | -0.0209 | -0.0413 to -0.0067 | 1.0880 | 1.1252 |

## Both measured mole fractions < 0.01: common_all_six

| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Difference | Paired 95% CI | Infinite RMSE | Finite RMSE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| UNIFAC | 1,508 | 33 | 0.5414 | 0.5409 | -0.0005 | -0.0052 to +0.0046 | 0.8524 | 0.8505 |
| UNIFAC2 | 1,508 | 33 | 0.6041 | 0.6080 | +0.0040 | +0.0003 to +0.0082 | 0.8990 | 0.8944 |
| UNIFDMD | 1,508 | 33 | 0.5614 | 0.5432 | -0.0182 | -0.0433 to -0.0013 | 0.9220 | 0.9560 |
| UNIFM2 | 1,508 | 33 | 0.6065 | 0.5934 | -0.0131 | -0.0314 to -0.0009 | 0.9901 | 1.0117 |
| UNIFNIST | 1,508 | 33 | 0.5900 | 0.5810 | -0.0090 | -0.0307 to +0.0108 | 0.9032 | 0.9206 |
| UNIFLBY | 1,508 | 33 | 0.6262 | 0.6060 | -0.0202 | -0.0397 to -0.0047 | 0.9799 | 1.0128 |

## Both measured mole fractions < 0.01: common_five

| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Difference | Paired 95% CI | Infinite RMSE | Finite RMSE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| UNIFAC | 3,509 | 58 | 0.5173 | 0.5186 | +0.0013 | -0.0044 to +0.0069 | 0.8697 | 0.8650 |
| UNIFAC2 | 3,509 | 58 | 0.5741 | 0.5825 | +0.0084 | +0.0009 to +0.0206 | 0.9330 | 0.9156 |
| UNIFDMD | 3,509 | 58 | 0.6551 | 0.6438 | -0.0112 | -0.0262 to +0.0004 | 1.1945 | 1.2110 |
| UNIFM2 | 3,509 | 58 | 0.6101 | 0.6029 | -0.0072 | -0.0181 to +0.0011 | 1.0421 | 1.0510 |
| UNIFNIST | 3,509 | 58 | 0.7106 | 0.7004 | -0.0103 | -0.0339 to +0.0118 | 1.2568 | 1.2734 |

## Cutoff 0.01: directional breakdown

| Variant | Direction | Transfers | Molecules | Finite MAE | Infinite MAE | Difference |
|---|---|---:|---:|---:|---:|---:|
| UNIFAC | water_to_nonwater | 346 | 31 | 1.1090 | 1.1178 | +0.0087 |
| UNIFAC | nonwater_to_water | 491 | 43 | 2.0701 | 2.0508 | -0.0193 |
| UNIFAC | nonwater_to_nonwater | 9,258 | 117 | 0.6377 | 0.6343 | -0.0034 |
| UNIFAC2 | water_to_nonwater | 890 | 65 | 1.2691 | 1.2588 | -0.0103 |
| UNIFAC2 | nonwater_to_water | 1,130 | 82 | 2.0030 | 1.9425 | -0.0606 |
| UNIFAC2 | nonwater_to_nonwater | 19,109 | 185 | 0.6429 | 0.6519 | +0.0091 |
| UNIFDMD | water_to_nonwater | 262 | 18 | 0.9308 | 0.9266 | -0.0043 |
| UNIFDMD | nonwater_to_water | 412 | 33 | 1.9436 | 1.9508 | +0.0071 |
| UNIFDMD | nonwater_to_nonwater | 7,499 | 108 | 0.6676 | 0.6496 | -0.0179 |
| UNIFM2 | water_to_nonwater | 752 | 55 | 1.2760 | 1.2760 | -0.0001 |
| UNIFM2 | nonwater_to_water | 1,116 | 80 | 2.0451 | 2.0464 | +0.0013 |
| UNIFM2 | nonwater_to_nonwater | 18,544 | 187 | 0.6832 | 0.6759 | -0.0073 |
| UNIFNIST | water_to_nonwater | 169 | 13 | 1.0691 | 1.0325 | -0.0366 |
| UNIFNIST | nonwater_to_water | 244 | 20 | 1.7921 | 1.7901 | -0.0020 |
| UNIFNIST | nonwater_to_nonwater | 5,205 | 77 | 0.7336 | 0.7316 | -0.0020 |
| UNIFLBY | water_to_nonwater | 39 | 7 | 0.7049 | 0.6998 | -0.0051 |
| UNIFLBY | nonwater_to_water | 125 | 14 | 2.2738 | 2.2637 | -0.0101 |
| UNIFLBY | nonwater_to_nonwater | 3,233 | 53 | 0.6003 | 0.5785 | -0.0218 |

## Cutoff 0.01: multiplier baseline on each variant’s paired cases

| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Multiplier MAE |
|---|---:|---:|---:|---:|---:|
| UNIFAC | 9,813 | 128 | 0.7558 | 0.7500 | 0.5459 |
| UNIFAC2 | 20,709 | 201 | 0.8325 | 0.8243 | 0.6010 |
| UNIFDMD | 7,960 | 114 | 0.7573 | 0.7378 | 0.5361 |
| UNIFM2 | 20,020 | 202 | 0.8430 | 0.8372 | 0.5978 |
| UNIFNIST | 5,491 | 82 | 0.7898 | 0.7880 | 0.5652 |
| UNIFLBY | 3,317 | 57 | 0.6490 | 0.6281 | 0.5628 |

## Cutoff 0.01: pooled_publications, own coverage

| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Difference | Paired 95% CI | Infinite RMSE | Finite RMSE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| UNIFAC | 11,475 | 130 | 0.7428 | 0.7361 | -0.0066 | -0.0203 to +0.0042 | 1.3012 | 1.3049 |
| UNIFAC2 | 23,148 | 205 | 0.8203 | 0.8118 | -0.0085 | -0.0297 to +0.0082 | 1.3259 | 1.3424 |
| UNIFDMD | 9,252 | 118 | 0.7728 | 0.7524 | -0.0205 | -0.0354 to -0.0082 | 1.3204 | 1.3416 |
| UNIFM2 | 22,379 | 207 | 0.8402 | 0.8344 | -0.0058 | -0.0158 to +0.0038 | 1.3391 | 1.3358 |
| UNIFNIST | 6,553 | 85 | 0.8021 | 0.8010 | -0.0011 | -0.0297 to +0.0336 | 1.3152 | 1.3080 |
| UNIFLBY | 4,258 | 59 | 0.6385 | 0.6171 | -0.0213 | -0.0411 to -0.0077 | 1.0529 | 1.0910 |


## Cutoff 0.01: AqSolDB_approximate_water_anchor, own coverage

| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Difference | Paired 95% CI | Infinite RMSE | Finite RMSE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| UNIFAC | 158 | 31 | 1.0732 | 1.0296 | -0.0436 | -0.0740 to -0.0192 | 1.3097 | 1.3346 |
| UNIFAC2 | 252 | 48 | 1.0805 | 1.0823 | +0.0018 | -0.0326 to +0.0326 | 1.5435 | 1.5247 |
| UNIFDMD | 133 | 24 | 1.2915 | 1.1545 | -0.1370 | -0.2310 to -0.0624 | 1.4336 | 1.6183 |
| UNIFM2 | 232 | 47 | 1.1591 | 1.1321 | -0.0270 | -0.0796 to +0.0324 | 1.5344 | 1.5308 |
| UNIFNIST | 93 | 17 | 1.1911 | 1.2089 | +0.0178 | -0.0446 to +0.1065 | 1.3495 | 1.3577 |
| UNIFLBY | 88 | 18 | 0.7672 | 0.6796 | -0.0876 | -0.2100 to -0.0136 | 0.8538 | 0.9756 |

## Numerical audit

```json
{
  "checks": {
    "endpoint_matches_readable_all_six": true,
    "endpoint_matches_near_dilute_all_six": true,
    "maximum_ln_gamma_difference_at_1e_minus_12": 1.1143974631915378e-11,
    "identical_solvent_transfer": true
  },
  "required_solutes": 992,
  "endpoint_evaluations": 114744,
  "maximum_near_endpoint_ln_gamma_difference": 0.00017636623886115846
}
```

## Files

- gamma_infinity.csv: endpoint activities, availability and near-endpoint checks for every required binary/temperature/variant.
- infinite_errors.npz: case IDs and capped/uncapped infinite prediction errors.
- summary.json: all scores, uncertainties, prediction differences and baseline comparisons.
- manifest.json: source/parameter/code fingerprints.
- Existing finite predictions, held-out baseline results, source data and production code were not modified.
