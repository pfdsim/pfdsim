# Direct Psat / Normal-Boiling-Point Validation Refactor Plan

Status: design only.

Scope: shared normal-boiling-point validation for direct Psat sources, initially applied to Perry Table 2-10 and the existing Antoine layers. This plan also covers the resulting confidence, online-completeness, omega-recovery, cache-version, testing, audit, and regeneration behavior.

## 1. Why this work exists

Thiodiglycol exposed two independent problems:

1. A transiently incomplete online resolution could be persisted as though the online-enabled resolution were complete. That cache-completeness defect has a separate shared fix.
2. Perry Table 2-10 contains a thiodiglycol vapor-pressure row that is grossly inconsistent with the independently resolved normal boiling point:

   ```text
   hard Tb:                  555.15 K at 1.01325 bar
   Perry 2-10 Psat at Tb:   approximately 0.129 bar
   relative pressure error: approximately 87.25%
   ```

The second problem lets an unreliable direct Psat segment enter canonical Psat construction and omega recovery. Antoine already has related `Tb` validation logic, but that logic is provider-specific and its confidence rules are not yet a coherent shared policy.

The refactor must solve the class of problem rather than adding a thiodiglycol exception.

## 2. Requirements

### 2.1 Required behavior

1. Use one shared normal-boiling-point validation mechanism for eligible direct Psat sources.
2. Apply it to:
   - Smith/high-quality Antoine, priority 550.
   - Perry Table 2-10, priority 500.
   - Ordinary and cached-NIST Antoine, priority 400.
3. Exempt every direct source at priority 600 or higher:
   - Perry Table 2-8, priority 600.
   - Curated/local Antoine, priority 700.
   - Local direct Psat correlations, priority 800.
   - CoolProp, priority 900.
   - PFD overrides, priority 1000.
4. A hard, qualified `Tb` may validate or reject an eligible segment only when `Tb` lies within the segment's declared temperature range, allowing the existing small endpoint tolerance.
5. An estimated, soft, missing, or unprovenanced-inadmissible `Tb` must not reject or directly validate an eligible segment.
6. A segment that has not been hard-`Tb` validated may still be retained at lower confidence.
7. A strictly higher-priority surviving direct source may raise an unvalidated segment to an intermediate confidence tier when the sources agree over a real overlap.
8. A same-priority source must never promote another source's confidence.
9. A segment rejected by the shared admission policy must be unavailable to both:
   - canonical Psat construction; and
   - omega recovery at `Tr = 0.7`.
10. If an eligible selected segment still lacks `Tb` or higher-priority overlap validation, an offline canonical result must remain `not_attempted`, not `not_needed`.
11. If an online-enabled resolution actually completes and finds no qualified `Tb`, the baseline unvalidated curve may persist as `complete_no_data`.
12. PFD and user overrides remain memory-only and must never enter persistent property-result or derived-parameter caches.

### 2.2 Explicit non-goals

1. Do not validate or reject Perry Table 2-8 against `Tb`.
2. Do not validate or reject curated/local Antoine against `Tb`.
3. Do not change the Perry Table 2-10 source data or add a CAS-specific blacklist.
4. Do not derive a missing `Tb` from Psat as part of this refactor.
5. Do not change the 2% normal-point pressure tolerance without a separate benchmark and decision.
6. Do not change the Pitzer omega definition or its existing flat `0.02` Psat-domain penalty.
7. Do not treat a lower confidence tier as proof that the source is wrong; it represents missing independent validation.

## 3. Core invariants

The implementation must preserve all of the following:

```text
priority >= 600
    => exempt from the shared Tb gate

priority < 600 and direct hard-pinned source
    => eligible for the shared Tb gate

qualified hard Tb + in-range agreement
    => hard-Tb validated confidence

qualified hard Tb + in-range conflict
    => reject segment

qualified hard Tb outside declared source range
    => retain at unvalidated confidence; do not reject

no qualified hard Tb
    => retain at unvalidated confidence; validation remains unavailable

strictly higher-priority agreeing overlap
    => intermediate overlap-validated confidence

same-priority overlap
    => no confidence promotion

rejected source
    => absent from canonical Psat and omega recovery

PFD/user override influence
    => no persistent selected or derived cache entry
```

## 4. Terminology

### 4.1 Qualified `Tb`

A `Tb` candidate is qualified when it satisfies the existing shared hard-property admission rule:

```text
finite and positive
quality >= 0.90
not estimated/soft/missing
not rejected by triple-point/sublimator rules
```

Direct unprovenanced programmatic values retain their already-established legacy treatment. Resolver-produced values always carry provenance and follow the explicit quality rule.

### 4.2 Direct Psat segment

A direct segment evaluates source vapor-pressure data without depending on a canonical completion. Examples include Antoine, Perry 2-8, Perry 2-10, CoolProp native saturation, and curated direct correlations.

Ambrose-Walton, Clapeyron, bridges, and other completion relations are not direct source segments and are outside this validation policy.

### 4.3 Intrinsic confidence profile

The confidence ladder must not be hard-coded inside Perry-specific or Antoine-specific collection functions. Each eligible source class uses a centrally declared immutable confidence profile:

```text
standalone_quality
higher_priority_overlap_quality
hard_tb_quality
```

Collection and handoff code consume the profile generically.

## 5. Shared data model

### 5.1 Validation result

Create or retain a provider-neutral result type such as:

```python
PsatBoilingPointValidation(
    T_boiling,
    covers_boiling_point,
    pressure_bar,
    relative_error,
    accepted,
)
```

The old Antoine-specific result name may remain only as a compatibility alias if required by public imports or tests. New implementation code should use the provider-neutral name.

### 5.2 Confidence profile

Define a central immutable type such as:

```python
DirectPsatConfidenceProfile(
    standalone_quality,
    higher_priority_overlap_quality,
    hard_tb_quality,
)
```

Validation in `__post_init__` must enforce:

```text
0 <= standalone <= overlap <= hard_tb <= 1
```

Central profiles:

```text
STANDARD_DIRECT_TABLE_PROFILE
    standalone = 0.90
    overlap    = 0.93
    hard Tb    = 0.95

CACHED_NIST_ANTOINE_PROFILE
    standalone = 0.87
    overlap    = 0.90
    hard Tb    = 0.95

HIGH_QUALITY_ANTOINE_PROFILE
    preserve the intended Smith/high-quality maximum while applying
    the same ordered three-tier structure; exact values must be declared
    once beside the other profiles and covered by pinned tests
```

Perry 2-10 and ordinary Antoine use `STANDARD_DIRECT_TABLE_PROFILE`. No quality literal may appear in the Perry 2-10 admission branch.

Exempt sources do not require a confidence profile for `Tb` gating; they keep their existing intrinsic quality.

### 5.3 Admission decision

Use a provider-neutral result such as:

```python
DirectPsatAdmissionDecision(
    admitted,
    validation_status,
    selected_quality,
    handoff_requirement,
    pressure_bar,
    relative_error,
    warning,
)
```

Allowed statuses:

```text
exempt_priority
hard_tb_validated
hard_tb_conflict
hard_tb_outside_range
validation_unavailable
```

Do not create a separate Boolean `tb_validation_pending` if the same fact can be derived without ambiguity from:

```text
tb_validation_required = true
validation_status = validation_unavailable
quality_basis = standalone_unvalidated
```

Avoid redundant state fields that can disagree.

## 6. Shared algorithms

### 6.1 Provider-neutral numerical validation

Implement one function:

```python
validate_psat_boiling_point(
    pressure_at_temperature,
    T_min,
    T_max,
    T_boiling,
    *,
    range_tolerance_K,
    relative_tolerance,
    target_pressure_bar=1.01325,
)
```

It must:

1. Validate finite inputs and non-reversed bounds.
2. Determine whether `Tb` is within the declared source range using the shared endpoint tolerance.
3. If outside range, return `covers_boiling_point=False` without evaluating/extrapolating the source.
4. If in range, evaluate the source exactly once.
5. Reject nonfinite or nonpositive pressures as invalid in-range results.
6. Compute:

   ```text
   relative_error = abs(Psat(Tb) - 1.01325 bar) / 1.01325 bar
   ```

7. Accept when relative error is at most 2%.

The Antoine helper becomes a thin wrapper around this provider-neutral evaluator.

### 6.2 Provider-neutral admission policy

Implement one function that receives:

```text
segment priority
direct-source confidence profile, or none for exempt sources
qualified Tb candidate, or none
pressure evaluator and source range
source label for diagnostics
```

Decision table:

| Condition | Admission | Initial quality | Handoff requirement | Status |
|---|---:|---:|---|---|
| Priority >= 600 | yes | existing intrinsic quality | existing exempt-source rule | `exempt_priority` |
| Priority < 600, no qualified Tb | yes | profile standalone | overlap-if-available | `validation_unavailable` |
| Priority < 600, hard Tb outside range | yes | profile standalone | overlap-if-available | `hard_tb_outside_range` |
| Priority < 600, hard Tb agrees | yes | profile hard-Tb | none | `hard_tb_validated` |
| Priority < 600, hard Tb conflicts/invalid | no | n/a | n/a | `hard_tb_conflict` |

The warning for rejection must include:

```text
source/provider
Tb
Psat(Tb), when finite
relative error, when finite
allowed tolerance
```

### 6.3 Overlap promotion

Retain the existing central handoff promotion mechanism, but make it consume the generic confidence metadata emitted by the admission policy.

Promotion is permitted only if:

1. The candidate is admitted.
2. Its current quality basis is unvalidated standalone.
3. A surviving source with strictly greater numerical priority overlaps it.
4. The two sources satisfy the existing overlap consistency test.

On promotion:

```text
quality = profile higher-priority-overlap quality
quality_basis = higher_preference_overlap
record corroborating source, method, priority, and original quality
```

Same-priority sources must remain at standalone quality.

### 6.4 Canonical completeness

Canonical completeness must be derived from selected provenance plus the outer online-attempt state.

Offline classification:

```text
selected eligible segment with:
    validation_status = validation_unavailable
    and quality_basis = standalone_unvalidated
        => canonical state not_attempted

selected eligible segment promoted by higher-priority overlap
        => no pending Tb validation remains for that selected interval

selected segment hard-Tb validated
        => complete with respect to Tb validation

exempt selected segment
        => complete with respect to Tb validation
```

If any selected interval remains standalone and validation-unavailable, the entire offline canonical result is `not_attempted`, even if another interval uses an exempt or validated source.

Online-enabled classification continues to use the shared outer attempt tracker:

```text
transient_failure   => do not persist
not_attempted       => do not persist online-enabled result
complete_no_data    => persist baseline curve as complete_no_data
complete_with_data  => persist selected result
not_needed          => persist only when all selected source requirements are complete
```

Do not duplicate online state inside segment metadata.

## 7. Provider integration

### 7.1 Perry Table 2-10

1. Construct its direct pressure evaluator and declared range.
2. Obtain the shared qualified `Tb` candidate.
3. Call the shared admission policy using:

   ```text
   priority: PERRY_2_10 = 500
   profile: STANDARD_DIRECT_TABLE_PROFILE
   ```

4. If rejected, return no segment and emit the shared rejection warning.
5. If admitted, create the segment using the decision's quality, handoff requirement, and metadata.
6. Preserve `property_dependencies=()`.

### 7.2 Antoine layers below priority 600

Route Smith/high-quality, ordinary table, and cached-NIST Antoine through the same admission policy.

Provider code remains responsible only for:

```text
coefficients/evaluator
declared temperature range
priority
confidence profile
source metadata
```

It must not reimplement coverage, residual, rejection, or quality-basis logic.

### 7.3 Exempt sources

Perry 2-8 and every priority >= 600 source bypass the shared gate. Their metadata should state:

```text
tb_validation_required = false
tb_validation_status = exempt_priority
```

Do not retain old curated-Antoine `Tb` rejection code after the exemption policy is implemented. Remove it rather than leaving unreachable or contradictory branches.

## 8. Omega recovery

Omega recovery must consume the same admitted and handoff-coordinated direct segment set as canonical source selection.

The existing formula remains:

```text
Tomega = 0.7 Tc
omega = -log10(Psat(Tomega) / Pc) - 1
```

The existing quality rule remains:

```text
omega quality = min(Tc quality, Pc quality, selected Psat segment quality) - 0.02
```

Consequences of the shared confidence policy:

```text
Perry 2-10 hard-Tb validated:
    Psat input quality 0.95 before omega penalty

Perry 2-10 higher-priority-overlap validated:
    Psat input quality 0.93 before omega penalty

Perry 2-10 standalone/unavailable:
    Psat input quality 0.90 before omega penalty

Perry 2-10 hard-Tb conflict:
    no Psat-derived omega candidate from that segment
```

All existing circularity guards remain:

- no omega-dependent segment;
- no untagged segment;
- no PFD contract;
- no EOS-effective critical bundle;
- segment must cover `0.7 Tc`;
- derived omega must remain in the physical admission interval;
- candidate replaces the existing omega only when its penalized quality is strictly higher.

For thiodiglycol with hard PubChem `Tb`, Perry 2-10 must be rejected before omega selection, leaving the lower-quality Lee-Kesler estimate rather than the previous `omega approximately 1.76` candidate.

## 9. Cache contracts

### 9.1 Required version changes

The implementation changes both selected critical properties and canonical curves:

```text
critical-property cache: bump from version 5 to version 6
canonical Psat cache:    bump from version 12 to version 13
phase-point cache:       remains version 7
```

Do not bump the shared source-response cache contract; no online response payload format changes.

### 9.2 Persistent cache restrictions

Retain the global invariant:

```text
PFD/user override influenced result => memory-only, never persistent
```

Apply it to phase points, critical bundles, canonical Psat, fitted `Z_RA`, and identifier-to-SMILES mappings.

### 9.3 Regeneration order

After implementation and tests pass:

1. Back up saturation and canonical SQLite databases to `/tmp`.
2. Clear all current and obsolete critical-property rows.
3. Clear all current and obsolete canonical Psat rows.
4. Do not clear phase-point version-7 rows merely for this refactor.
5. Run the offline saturation-property warmer to regenerate critical version 6.
6. Run the offline canonical-Psat warmer to regenerate version 13.
7. Check that only the current critical and canonical versions remain.
8. Run WAL checkpoint, `PRAGMA optimize`, and `PRAGMA integrity_check`.

The canonical warmer's expected unavailable set is not a test failure. Record the final coverage count and compare it with the prior `1007 resolved / 97 unavailable` baseline.

## 10. Test plan

### 10.1 Shared numerical validator

Tests must cover:

1. Exact normal-pressure reproduction.
2. In-range accepted value just inside 2%.
3. In-range rejected value just outside 2%.
4. Gross conflict.
5. `Tb` below range.
6. `Tb` above range.
7. Endpoint tolerance.
8. Nonfinite pressure.
9. Nonpositive pressure.
10. Reversed/invalid source range.

### 10.2 Priority policy

Pin the boundary explicitly:

```text
400 => gated
500 => gated
550 => gated
599 => gated
600 => exempt
700 => exempt
800 => exempt
900 => exempt
1000 => exempt
```

### 10.3 Confidence profiles

For each central profile, verify:

1. Standalone quality.
2. Strictly higher-priority overlap quality.
3. Hard-`Tb` quality.
4. Monotonic ordering.
5. Same-priority overlap does not promote.

Perry 2-10 must be tested through the shared standard profile, not against repeated numeric literals inside provider code.

### 10.4 Perry Table 2-10

Tests must cover:

1. A representative accepted compound with hard `Tb`.
2. Thiodiglycol rejection with `Tb = 555.15 K`.
3. Soft/estimated `Tb` cannot reject thiodiglycol.
4. Missing `Tb` retains Perry 2-10 at standalone confidence.
5. Hard `Tb` outside the Perry range retains the segment at standalone confidence.
6. Strictly higher-priority overlap promotes to intermediate confidence.
7. Same-priority Perry 2-10 records do not mutually promote.
8. Rejection warning includes diagnostics.
9. Segment metadata records one coherent status and quality basis.

### 10.5 Exemptions

Tests must prove:

1. Perry 2-8 survives a deliberately conflicting hard `Tb`.
2. Curated/local Antoine survives a deliberately conflicting hard `Tb`.
3. PFD Psat remains authoritative and memory-only.
4. Exempt segment metadata says `exempt_priority` and does not carry contradictory pending state.

### 10.6 Omega

Tests must prove:

1. Rejected thiodiglycol Perry 2-10 cannot supply omega.
2. Standalone Perry 2-10 omega uses the standalone Psat quality before the `0.02` penalty.
3. Higher-priority-overlap Perry 2-10 omega uses the overlap quality.
4. Hard-`Tb` validated Perry 2-10 omega uses the hard-`Tb` quality.
5. Existing higher-quality omega still wins.
6. Equal quality retains the existing omega.

### 10.7 Online completeness and canonical persistence

Tests must prove:

1. Offline canonical using a standalone validation-unavailable eligible segment is stored as `not_attempted`.
2. A mixed curve with one safe segment and one standalone validation-unavailable selected segment remains `not_attempted`.
3. Higher-priority overlap validation removes the selected segment's pending requirement.
4. Online transient `Tb` lookup prevents persistent canonical caching.
5. Online completed-no-data lookup may persist the baseline curve as `complete_no_data`.
6. Online successful hard `Tb` either validates or rejects the segment before persistence.

### 10.8 Regression suites

At minimum run:

```text
test_vapor_pressure_adapter.py
test_vapor_pressure_resolution.py
test_critical_properties_cache.py
test_perry_property_lookup.py
test_property_resolution_system.py
test_pfd_component_properties.py
```

Then run the repository's complete suite exactly as:

```text
python -m pytest
```

Finally run all 32 example flowsheets.

## 11. Perry Table 2-10 rejection audit

Generate a machine-readable report at:

```text
/tmp/perry_2_10_tb_validation_report.json
```

For every Perry 2-10 row, record:

```text
CAS
name
qualified Tb value
Tb source/method/quality
whether cached online Tb was used
source temperature range
Psat(Tb)
relative error
status: accepted / rejected / outside_range / unavailable
```

Report aggregate statistics:

```text
total rows
rows with qualified Tb
qualified Tb inside source range
accepted count
rejected count
outside-range count
unavailable count
rejection rate among in-range qualified cases
rejection rate across all rows
counts above 2%, 3%, 5%, 10%, 20%, and 50% error
```

Also compare canonical and omega populations before and after:

```text
canonical coverage
source-method counts
number of Perry 2-10 selected intervals
number of Perry 2-10 standalone / overlap / hard-Tb quality bases
number of Psat-derived omega results
largest omega changes
any newly unavailable canonical curves
```

## 12. Sanity audit after implementation

Before cache regeneration, inspect the implementation for:

1. Duplicate tolerance or confidence literals.
2. Provider-specific copies of shared validation logic.
3. Contradictory metadata fields.
4. Dead curated-Antoine validation branches after priority exemption.
5. Dead compatibility names that are no longer imported.
6. Unused pending-state fields.
7. Any path where omega sees a segment canonical admission rejected.
8. Any path where same-priority overlap promotes confidence.
9. Any path where an online transient result enters memory or disk selected caches.
10. Any path where PFD/user override influence enters persistent caches.

After regeneration, query the databases for:

```text
only current cache versions
SQLite integrity = ok
zero persisted PFD/override results
zero canonical not_needed rows containing selected standalone validation-unavailable eligible segments
zero rejected Perry 2-10 provenance entries
all exact-key lookups using primary indexes
```

Representative runtime checks:

```text
thiodiglycol with PubChem Tb:
    Perry 2-10 rejected
    no omega = 1.76 from Perry 2-10
    canonical provenance excludes Perry 2-10

ordinary accepted Perry 2-10 compound:
    hard-Tb status
    quality = standard profile hard-Tb quality

Perry 2-8 compound:
    unchanged selection and quality

curated local Antoine compound:
    unchanged authoritative selection
```

## 13. Implementation order

The implementation must follow this order to avoid another incoherent partial refactor:

1. Implement and test the provider-neutral numerical validator.
2. Implement and test the central confidence profiles.
3. Implement and test the provider-neutral admission decision.
4. Route existing gated Antoine layers through the shared policy.
5. Remove obsolete curated/local Antoine validation because priority 700 is exempt.
6. Route Perry 2-10 through the same policy.
7. Verify Perry 2-8 remains untouched and exempt.
8. Integrate generic overlap promotion metadata.
9. Integrate canonical completeness using derived status rather than redundant Booleans.
10. Verify omega recovery uses the exact admitted/coordinated segment set.
11. Run focused tests.
12. Generate and inspect the 921-row rejection audit.
13. Run affected suites.
14. Run the complete suite.
15. Run all examples.
16. Back up, clear, and regenerate critical and canonical SQLite caches offline.
17. Perform read-only database and population audits.
18. Report behavior, statistics, cache versions, tests, remaining known failures, and every non-trivial design decision.

No cache regeneration should occur until steps 1 through 17 are complete.

## 14. Acceptance criteria

This refactor is complete only when all are true:

1. There is one provider-neutral numerical `Tb` validator.
2. There is one provider-neutral direct-source admission policy.
3. There is one central confidence-profile definition per source class.
4. No Perry-specific quality literals implement the confidence ladder.
5. Priority 600 is the exact exemption boundary.
6. Perry 2-8 and curated local Antoine are demonstrably exempt.
7. Thiodiglycol Perry 2-10 is rejected with hard PubChem `Tb`.
8. Thiodiglycol no longer receives the Perry-derived `omega approximately 1.76` candidate when hard `Tb` is present.
9. Pending/unavailable validation cannot create an online-authoritative canonical cache entry.
10. Same-priority overlap cannot promote confidence.
11. PFD/user override results remain absent from persistent caches.
12. Final Perry 2-10 rejection statistics are reported.
13. Critical version 6 and canonical version 13 caches are cleanly regenerated.
14. All affected tests pass.
15. The complete suite has no new failures beyond the two already-known rigorous-distillation errors.
16. All 32 example flowsheets converge.

## 15. Design decisions fixed by this plan

1. **Priority-based exemption, not a provider blacklist.** The exact boundary is priority 600. This keeps policy aligned with source authority and automatically covers current high-authority direct sources.
2. **Shared confidence profiles, not scattered constants or a single universal formula.** This preserves the intentionally lower cached-NIST baseline while allowing Perry 2-10 and ordinary Antoine to share one profile.
3. **Binary rejection only for in-range hard-`Tb` conflicts.** Missing or out-of-range validation lowers confidence but does not erase direct empirical data.
4. **Higher-priority overlap is independent corroboration; same-priority overlap is not.** This prevents mutually reinforcing records from manufacturing confidence.
5. **One admitted/coordinated source set feeds canonical Psat and omega.** This eliminates divergent source acceptance between the two consumers.
6. **Canonical completeness is derived from validation status, quality basis, and the outer online-attempt state.** Redundant pending flags are avoided.
7. **Perry 2-8 and curated/local Antoine are exempt.** Their authority and historical accuracy outweigh the risk addressed by this gate.
8. **Critical and canonical caches are versioned and rebuilt only after full verification.** This prevents stale omega and curve results from surviving the policy change.
