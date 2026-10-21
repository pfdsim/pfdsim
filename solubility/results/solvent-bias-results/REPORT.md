# Solvent-dependent UNIFAC prediction bias

Signed error is log10(predicted solubility / measured solubility). Negative bias means underprediction; positive bias means overprediction. 10**bias is the molecule-balanced geometric prediction/measurement ratio, not the mean arithmetic relative error.

## Main protocol

- Reuse the saved same-publication, exact-temperature 20–30 °C predictions, deduplication and failure/coverage screens. No models are rerun.
- For each variant and target solvent, average signed errors within each molecule first, over all its reference solvents, temperatures and publications. Then average those molecule means equally. Reference-solvent effects are calculated separately using the same rule.
- Require at least 20 distinct molecules for significance tests. Sparse solvent results are retained descriptively.
- Test the molecule-level mean against zero using a two-sided one-sample t test. Bootstrap whole molecule means for descriptive 95% intervals. The t test assumes independent molecules and uses an estimated standard error; small or heavy-tailed samples should be interpreted cautiously.
- Apply Holm familywise correction across ALL tested solvents, both roles, and all six variants, separately for each prespecified scenario. A significant result requires corrected p<0.05. Bootstrap intervals are descriptive ordinary 95% intervals and are not multiplicity-adjusted.
- Also report the median molecule error, fraction of molecules overpredicted, and exact sign-test p values with their own Holm correction. These distinguish a mean shift driven by large errors from consistent directional bias across molecules.
- Marginal solvent bias depends on the tested molecules and reference-solvent mixture. It does not prove an intrinsic error in one UNIFAC interaction parameter.

## Significant target-solvent biases, finite concentration

| Variant | Target solvent | Molecules | Bias log10 | Geometric prediction ratio | Bootstrap 95% CI | Holm p |
|---|---|---:|---:|---:|---|---:|
| UNIFAC | THF | 35 | -0.907 | 0.124 | -1.196 to -0.663 | 3.94e-05 |
| UNIFAC | 1,4-dioxane | 42 | -0.898 | 0.127 | -1.218 to -0.637 | 0.000132 |
| UNIFAC | n-propanol | 142 | -0.103 | 0.789 | -0.144 to -0.063 | 0.000479 |
| UNIFAC | methanol | 161 | +0.183 | 1.524 | +0.089 to +0.273 | 0.0435 |
| UNIFAC | acetonitrile | 76 | +0.394 | 2.477 | +0.289 to +0.498 | 2.68e-08 |
| UNIFAC2 | water | 136 | -1.069 | 0.085 | -1.452 to -0.712 | 1.7e-05 |
| UNIFAC2 | THF | 46 | -0.705 | 0.197 | -0.936 to -0.497 | 4.09e-05 |
| UNIFAC2 | 1,4-dioxane | 77 | -0.286 | 0.517 | -0.384 to -0.190 | 3.99e-05 |
| UNIFAC2 | n-pentanol | 47 | -0.135 | 0.732 | -0.201 to -0.070 | 0.0435 |
| UNIFAC2 | ethanol | 285 | -0.130 | 0.741 | -0.182 to -0.079 | 0.00026 |
| UNIFAC2 | acetonitrile | 168 | +0.286 | 1.934 | +0.211 to +0.360 | 7.27e-10 |
| UNIFAC2 | methanol | 236 | +0.304 | 2.015 | +0.233 to +0.377 | 2.74e-12 |
| UNIFDMD | water | 67 | -1.077 | 0.084 | -1.484 to -0.683 | 0.000466 |
| UNIFDMD | ethyl acetate | 130 | -0.219 | 0.604 | -0.336 to -0.114 | 0.0398 |
| UNIFM2 | water | 134 | -1.340 | 0.046 | -1.699 to -0.985 | 4.34e-09 |
| UNIFM2 | THF | 46 | -0.748 | 0.179 | -1.020 to -0.511 | 0.000259 |
| UNIFM2 | ethyl acetate | 231 | -0.448 | 0.357 | -0.534 to -0.365 | 2.51e-18 |
| UNIFM2 | methyl acetate | 81 | -0.421 | 0.379 | -0.545 to -0.306 | 3.26e-07 |
| UNIFM2 | acetonitrile | 171 | +0.204 | 1.599 | +0.133 to +0.275 | 3.17e-05 |
| UNIFM2 | methanol | 239 | +0.363 | 2.308 | +0.286 to +0.442 | 4.26e-15 |
| UNIFNIST | water | 45 | -1.118 | 0.076 | -1.542 to -0.726 | 0.000978 |

## Reference-adjusted relative solvent effects

- Use only solvent pairs where both prediction directions succeeded. Retain each unordered pair once and use d=(error_reference_to_target-error_target_to_reference)/2, removing the direction-independent part of finite-solver error.
- Fit d≈b_target-b_reference by least squares, weighting each molecule equally. Retain core solvents with sufficient distinct molecules and anchor ethanol to zero.
- These coefficients describe relative solvent effects. Absolute solvent errors are unidentifiable from transfer measurements. The projection adjusts for reference-solvent mixture through the shared comparison network; it does not eliminate chemical-class imbalance or solvent–solute interactions.
- Test coefficients using compound-cluster robust sandwich covariance and t reference distribution. Apply Holm correction across all solvent coefficients and all six variants. Molecule-bootstrap refits provide descriptive 95% intervals.
- Coefficients are diagnostic in-sample summaries. They are not used to correct predictions or establish held-out accuracy after correction.

| Variant | Solvent | Bias relative to ethanol | Geometric relative ratio | Bootstrap 95% CI | Holm p |
|---|---|---:|---:|---|---:|
| UNIFAC | methyl acetate | +0.181 | 1.517 | +0.083 to +0.282 | 0.0353 |
| UNIFAC | methanol | +0.280 | 1.905 | +0.178 to +0.384 | 2.54e-05 |
| UNIFAC | acetonitrile | +0.473 | 2.973 | +0.345 to +0.602 | 1.84e-09 |
| UNIFAC2 | methyl acetate | +0.212 | 1.630 | +0.121 to +0.301 | 0.00145 |
| UNIFAC2 | acetonitrile | +0.322 | 2.099 | +0.214 to +0.436 | 2.14e-06 |
| UNIFAC2 | chloroform | +0.407 | 2.555 | +0.182 to +0.650 | 0.0386 |
| UNIFAC2 | cyclohexane | +0.427 | 2.670 | +0.208 to +0.630 | 0.00707 |
| UNIFAC2 | methanol | +0.445 | 2.788 | +0.336 to +0.560 | 1.18e-11 |
| UNIFDMD | methanol | +0.273 | 1.875 | +0.135 to +0.404 | 0.00886 |
| UNIFDMD | toluene | +0.400 | 2.514 | +0.222 to +0.581 | 0.000838 |
| UNIFM2 | ethyl acetate | -0.358 | 0.439 | -0.471 to -0.246 | 1.09e-06 |
| UNIFM2 | methyl acetate | -0.335 | 0.462 | -0.451 to -0.223 | 3.17e-06 |
| UNIFM2 | acetonitrile | +0.203 | 1.597 | +0.106 to +0.297 | 0.00773 |
| UNIFM2 | methanol | +0.363 | 2.307 | +0.245 to +0.483 | 2.85e-07 |
| UNIFM2 | NMP | +0.626 | 4.228 | +0.322 to +0.985 | 0.0195 |
| UNIFM2 | DMF | +0.789 | 6.147 | +0.341 to +1.173 | 0.0253 |
| UNIFNIST | methyl acetate | -0.334 | 0.463 | -0.512 to -0.156 | 0.0345 |

## Dilution sensitivity

finite_0.05 and infinite_0.05 use identical finite-success cases with both measured mole fractions <0.05. finite_0.01 and infinite_0.01 similarly use <0.01. These subsets have different molecule populations. All per-solvent signed means, sample sizes and tests are in solvent_bias.csv.

```json
{
  "finite_0.01": {
    "UNIFAC": {
      "target": {
        "tested": 12,
        "significant": 1
      },
      "reference": {
        "tested": 12,
        "significant": 2
      }
    },
    "UNIFAC2": {
      "target": {
        "tested": 17,
        "significant": 3
      },
      "reference": {
        "tested": 17,
        "significant": 2
      }
    },
    "UNIFDMD": {
      "target": {
        "tested": 9,
        "significant": 1
      },
      "reference": {
        "tested": 8,
        "significant": 1
      }
    },
    "UNIFM2": {
      "target": {
        "tested": 18,
        "significant": 4
      },
      "reference": {
        "tested": 17,
        "significant": 8
      }
    },
    "UNIFNIST": {
      "target": {
        "tested": 8,
        "significant": 0
      },
      "reference": {
        "tested": 7,
        "significant": 0
      }
    },
    "UNIFLBY": {
      "target": {
        "tested": 6,
        "significant": 0
      },
      "reference": {
        "tested": 6,
        "significant": 0
      }
    }
  },
  "finite_0.05": {
    "UNIFAC": {
      "target": {
        "tested": 16,
        "significant": 4
      },
      "reference": {
        "tested": 15,
        "significant": 2
      }
    },
    "UNIFAC2": {
      "target": {
        "tested": 23,
        "significant": 5
      },
      "reference": {
        "tested": 22,
        "significant": 2
      }
    },
    "UNIFDMD": {
      "target": {
        "tested": 16,
        "significant": 1
      },
      "reference": {
        "tested": 16,
        "significant": 2
      }
    },
    "UNIFM2": {
      "target": {
        "tested": 23,
        "significant": 6
      },
      "reference": {
        "tested": 23,
        "significant": 5
      }
    },
    "UNIFNIST": {
      "target": {
        "tested": 13,
        "significant": 1
      },
      "reference": {
        "tested": 12,
        "significant": 0
      }
    },
    "UNIFLBY": {
      "target": {
        "tested": 8,
        "significant": 0
      },
      "reference": {
        "tested": 7,
        "significant": 0
      }
    }
  },
  "finite_all": {
    "UNIFAC": {
      "target": {
        "tested": 23,
        "significant": 5
      },
      "reference": {
        "tested": 22,
        "significant": 4
      }
    },
    "UNIFAC2": {
      "target": {
        "tested": 31,
        "significant": 7
      },
      "reference": {
        "tested": 30,
        "significant": 4
      }
    },
    "UNIFDMD": {
      "target": {
        "tested": 21,
        "significant": 2
      },
      "reference": {
        "tested": 20,
        "significant": 2
      }
    },
    "UNIFM2": {
      "target": {
        "tested": 31,
        "significant": 6
      },
      "reference": {
        "tested": 30,
        "significant": 9
      }
    },
    "UNIFNIST": {
      "target": {
        "tested": 16,
        "significant": 1
      },
      "reference": {
        "tested": 15,
        "significant": 2
      }
    },
    "UNIFLBY": {
      "target": {
        "tested": 12,
        "significant": 0
      },
      "reference": {
        "tested": 11,
        "significant": 0
      }
    }
  },
  "infinite_0.01": {
    "UNIFAC": {
      "target": {
        "tested": 12,
        "significant": 1
      },
      "reference": {
        "tested": 12,
        "significant": 2
      }
    },
    "UNIFAC2": {
      "target": {
        "tested": 17,
        "significant": 4
      },
      "reference": {
        "tested": 17,
        "significant": 2
      }
    },
    "UNIFDMD": {
      "target": {
        "tested": 9,
        "significant": 1
      },
      "reference": {
        "tested": 8,
        "significant": 1
      }
    },
    "UNIFM2": {
      "target": {
        "tested": 18,
        "significant": 4
      },
      "reference": {
        "tested": 17,
        "significant": 8
      }
    },
    "UNIFNIST": {
      "target": {
        "tested": 8,
        "significant": 0
      },
      "reference": {
        "tested": 7,
        "significant": 0
      }
    },
    "UNIFLBY": {
      "target": {
        "tested": 6,
        "significant": 0
      },
      "reference": {
        "tested": 6,
        "significant": 0
      }
    }
  },
  "infinite_0.05": {
    "UNIFAC": {
      "target": {
        "tested": 16,
        "significant": 2
      },
      "reference": {
        "tested": 15,
        "significant": 2
      }
    },
    "UNIFAC2": {
      "target": {
        "tested": 23,
        "significant": 5
      },
      "reference": {
        "tested": 22,
        "significant": 2
      }
    },
    "UNIFDMD": {
      "target": {
        "tested": 16,
        "significant": 1
      },
      "reference": {
        "tested": 16,
        "significant": 2
      }
    },
    "UNIFM2": {
      "target": {
        "tested": 23,
        "significant": 6
      },
      "reference": {
        "tested": 23,
        "significant": 5
      }
    },
    "UNIFNIST": {
      "target": {
        "tested": 13,
        "significant": 1
      },
      "reference": {
        "tested": 12,
        "significant": 0
      }
    },
    "UNIFLBY": {
      "target": {
        "tested": 8,
        "significant": 0
      },
      "reference": {
        "tested": 7,
        "significant": 0
      }
    }
  }
}
```

## Limits

Molecules can share chemical scaffolds and data sources, so molecule-level inference does not guarantee independence between chemical families. Aqueous measurements may be apparent rather than intrinsic neutral-species solubilities; pH is unavailable. Solid forms and some temperatures/pressures may differ despite the matching protocol. UNIFAC parameter-training overlap with underlying publications is unverified. Model-dependent availability and stability rejection select the evaluated populations. Solvent biases are therefore diagnostic of this dataset and protocol, not universal solvent constants.

## Reproducibility and files

- solvent_bias.csv: all target/reference results, including sparse descriptive entries.
- molecule_solvent_bias.csv: molecule-level means underlying the marginal tests.
- relative_solvent_bias.csv: reference-adjusted network coefficients.
- summary.json: counts, largest effects and numerical audit.
- manifest.json: exact arguments and source/result fingerprints.
- Production code, source datasets, previous results and UNIFAC parameters are unchanged.
