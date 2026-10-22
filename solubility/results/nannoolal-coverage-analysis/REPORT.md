# Uncovered Nannoolal aqueous-solubility molecules

The original experiment tested 9,044 neutral, connected, carbon-containing structures. 7,278 were representable and 1,766 were uncovered. This report concerns structural coverage, not invalid regression predictions.

## Native failure mechanism

Run the original native fragmenter on saved uncovered structures, and inspect its Python exception frame to recover ALL unassigned atoms and prior assignments. No matching priorities, patterns, parameters, or production implementation are changed. The native message lists at most eight atoms, so reading only its printed examples would miss some environments.

A second read-only pass checks existing native group patterns at those unassigned atoms, including their original predicates. It distinguishes atoms with no matching native rule from atoms whose matching group overlaps atoms claimed earlier. A blocked pattern is evidence of an assignment conflict, not proof that changing priority alone gives a valid complete fragmentation or that the published method supports every compound.

```json
{
  "no_matching_rules_only": 1012,
  "blocked_rules_only": 628,
  "mixed": 125,
  "single_heavy_atom": 1
}
```

## Structural motifs at actual unassigned atoms

These categories overlap. A motif elsewhere in the molecule is not counted as the cause: the motif atom set must intersect the native unassigned atom set. Phosphorus and unusual-element environments additionally include directly bonded neighbors.

| Motif | Uncovered molecules with unassigned atoms in motif | % of uncovered | Covered molecules containing motif | Uncovered molecules containing motif |
|---|---:|---:|---:|---:|
| carbonyl_other | 494 | 28.0% | 4044 | 1031 |
| amide_or_urea_carbonyl | 460 | 26.0% | 1211 | 583 |
| nitrogen_nitrogen_bond | 350 | 19.8% | 221 | 429 |
| sulfonamide | 336 | 19.0% | 54 | 340 |
| phosphorus_environment | 215 | 12.2% | 90 | 216 |
| nitrogen_oxygen_bond | 163 | 9.2% | 469 | 289 |
| thiocarbonyl | 121 | 6.9% | 32 | 121 |
| sulfonic_acid_or_sulfonate_ester | 115 | 6.5% | 10 | 115 |
| aromatic_nitrogen | 105 | 5.9% | 1010 | 587 |
| formal_charge_environment | 56 | 3.2% | 408 | 178 |
| imine | 39 | 2.2% | 227 | 180 |
| unusual_element_environment | 29 | 1.6% | 8 | 30 |
| nitrile | 10 | 0.6% | 167 | 64 |
| radical_environment | 4 | 0.2% | 3 | 4 |

## Examples

- carbonyl_other: formaldehyde; (1-hydroxy-1-phosphonoethyl)phosphonic acid; carbon dioxide; peroxyacetic acid; hydantoin
- amide_or_urea_carbonyl: hydantoin; 1-Acetylurea; 4(3h)-pyrimidone; succinimide; (E)-(carbamoylimino)urea; (Z)-(carbamoylimino)urea
- nitrogen_nitrogen_bond: methylhydrazine; 1,2-dimethylhydrazine; 1,1-dimethylhydrazine; n-nitrosodimethylamine; ethanol, 2-hydrazino-
- sulfonamide: methanesulfonamide, n-(methylsulfonyl)-; 5-Chlorothiophene-2-sulfonamide; benzenesulfonamide; ethanesulfonamide, n-(ethylsulfonyl)-; Sulfanilamide
- phosphorus_environment: methylphosphonic acid; ethephon; methamidophos; dimethyl methylphosphonate; sarin
- nitrogen_oxygen_bond: n-nitrosodimethylamine; hydroxyurea; n-nitrosomethylvinylamine; n-methyl-n-nitrosoethamine; N,N-diethylhydroxylamine
- thiocarbonyl: dithioxomethane; ethanethioamide; thiourea; aminothiourea; thiourea, cyano-
- sulfonic_acid_or_sulfonate_ester: methanesulfonic acid; methyl methanesulfonate; isethionic acid; taurine; 1,2-oxathiolane 2,2-dioxide
- aromatic_nitrogen: pyridine oxide; N-nitropyrazole; 4-methyl-n-oxidepyridine; 3-methyl-n-oxidepyridine; 3,5-dimethylpyridine-n-oxide
- formal_charge_environment: methanidylidyneoxidanium; pyridine oxide; 1-nitroguanidine; N-nitropyrazole; Betaine
- imine: allopurinol; 5-methyl-[1,2,4]triazolo[1,5-a]pyrimidin-7-ol; 4(1h)-pteridinone, hydrazone; 2(3h)-pteridinone, 3-methyl-; methazolamide
- unusual_element_environment: chipcote; dimethyltin dichloride; cacodylic acid; Tetra(10B)boron carbide; methane arsonate
- nitrile: hydrocyanic acid; iodine cyanide; chipcote; 2-cyanoguanidine; tabun
- radical_environment: 2-methylpropane; tempone; 4-hydroxy-2,2,6,6-tetramethylpiperidinoxyl; Octadecan-1-ol, ethoxylated

## Overlap-aware cumulative coverage

Categories are added in descending individual count, counting each molecule once. Broad motifs such as carbonyls are descriptive; they are not new proposed group definitions.

```json
[
  {
    "motif": "carbonyl_other",
    "new_molecules": 494,
    "cumulative_molecules": 494,
    "cumulative_fraction": 0.2797281993204983
  },
  {
    "motif": "amide_or_urea_carbonyl",
    "new_molecules": 8,
    "cumulative_molecules": 502,
    "cumulative_fraction": 0.2842582106455266
  },
  {
    "motif": "nitrogen_nitrogen_bond",
    "new_molecules": 323,
    "cumulative_molecules": 825,
    "cumulative_fraction": 0.46715741789354476
  },
  {
    "motif": "sulfonamide",
    "new_molecules": 322,
    "cumulative_molecules": 1147,
    "cumulative_fraction": 0.6494903737259343
  },
  {
    "motif": "phosphorus_environment",
    "new_molecules": 195,
    "cumulative_molecules": 1342,
    "cumulative_fraction": 0.7599093997734995
  },
  {
    "motif": "nitrogen_oxygen_bond",
    "new_molecules": 112,
    "cumulative_molecules": 1454,
    "cumulative_fraction": 0.8233295583238958
  },
  {
    "motif": "thiocarbonyl",
    "new_molecules": 103,
    "cumulative_molecules": 1557,
    "cumulative_fraction": 0.8816534541336354
  },
  {
    "motif": "sulfonic_acid_or_sulfonate_ester",
    "new_molecules": 106,
    "cumulative_molecules": 1663,
    "cumulative_fraction": 0.9416761041902605
  },
  {
    "motif": "aromatic_nitrogen",
    "new_molecules": 15,
    "cumulative_molecules": 1678,
    "cumulative_fraction": 0.9501698754246886
  },
  {
    "motif": "formal_charge_environment",
    "new_molecules": 3,
    "cumulative_molecules": 1681,
    "cumulative_fraction": 0.9518686296715742
  },
  {
    "motif": "imine",
    "new_molecules": 4,
    "cumulative_molecules": 1685,
    "cumulative_fraction": 0.9541336353340883
  },
  {
    "motif": "unusual_element_environment",
    "new_molecules": 21,
    "cumulative_molecules": 1706,
    "cumulative_fraction": 0.9660249150622876
  },
  {
    "motif": "nitrile",
    "new_molecules": 8,
    "cumulative_molecules": 1714,
    "cumulative_fraction": 0.970554926387316
  },
  {
    "motif": "radical_environment",
    "new_molecules": 2,
    "cumulative_molecules": 1716,
    "cumulative_fraction": 0.9716874292185731
  }
]
```

## Existing-rule blocking

| Candidate native group | Earlier assigned groups | Molecules | Atom/rule candidates |
|---|---|---:|---:|
| 49 | 100 | 182 | 516 |
| 105 | 82 | 330 | 378 |
| 48 | 100 | 132 | 286 |
| 48 | 48 | 63 | 150 |
| 49 | 49 | 39 | 82 |
| 48 | 99 | 19 | 40 |
| 73 | 73 | 4 | 18 |
| 100 | 100,48 | 9 | 18 |
| 100 | 49,99 | 8 | 16 |
| 100 | 100,49 | 8 | 16 |
| 105 | 105 | 5 | 5 |
| 48 | 109 | 3 | 3 |
| 105 | 100 | 1 | 3 |
| 48 | 105 | 1 | 2 |
| 94 | 79 | 2 | 2 |
| 100 | 100 | 1 | 2 |
| 77 | 118 | 1 | 2 |
| 118 | 118 | 1 | 2 |
| 49 | 99 | 1 | 2 |
| 96 | 16,17,99 | 1 | 2 |
| 76 | 99 | 1 | 2 |
| 47 | 99 | 1 | 2 |
| 48 | 118 | 1 | 1 |
| 91 | 91 | 1 | 1 |
| 49 | 118 | 1 | 1 |

## Scope and files

No fragmenter repair or extension was performed. Missing features, structural assignment conflicts and broad chemistry-domain limitations must be distinguished before proposing new group definitions. The prior neutral-organic filter permits organometallic structures and internally charge-separated molecules if their total charge is zero; those are included in the audit.

- molecules.csv: all covered/uncovered structures, motif flags and native failure classes.
- unassigned_atoms.csv: exact native atom indices and environments.
- blocked_rules.csv: matching native patterns blocked by prior assignments.
- motif_summary.csv, environments.csv, summary.json: counts, examples and overlap-aware summaries.
- manifest.json: reproducible source/results fingerprints. Prior predictions, source data and production code are unchanged.
