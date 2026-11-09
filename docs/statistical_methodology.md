# Statistical Methodology (v0.4.0 Stage 1)

This document describes the statistical primitives in `openbench.stats`. They are
standard-library only and operate on per-item values that the caller has already aligned on
stable record IDs, filtered and sorted by ID.

> **Scope of Stage 1.** These primitives are not yet wired into any CLI command, evaluator,
> report or run artifact. Existing commands, outputs and scores are unchanged.

Every procedure below states its assumptions and whether its inference is **exact** or
**approximate**. Nothing here is a claim of scientific validation or of absolute model
quality: results describe these models, prompts, settings and reference answers on the
analysed items.

---

## 1. Unit of analysis and estimands

- The unit is the **dataset item**. With repeated runs, an item's value is the mean of its
  metric over that model's valid repetitions. The effective sample size is the number of
  items, never items × repetitions.
- **Paired difference:** d_i = y_B,i − y_A,i, with A the baseline and B the challenger. The
  estimand is the population mean difference Δ = E[d], treating the analysed items as a
  random sample from a larger population of similar items.

## 2. Metric kinds (`score_kind`)

`BaseMetric.score_kind` is a class attribute: `"binary"`, `"bounded_continuous"` or
`"unknown"` (the default for custom metrics). It only selects statistical procedures and
never changes scores.

| Metrics | `score_kind` |
| --- | --- |
| `exact_match`, `accuracy` | `binary` |
| `keyword_coverage` (`keyword_match`), `precision`, `recall`, `f1`, `rouge`, `rouge_1`, `rouge_l`, `bleu`, `semantic_similarity`, `lexical_similarity`, `structured_output` | `bounded_continuous` |

`select_difference_test(score_kind, repetitions_a, repetitions_b)` returns
`"mcnemar_exact"` (binary, one repetition per arm), `"sign_flip"` (bounded-continuous, or
binary with more than one repetition in an arm), or `None` (unknown: interval only).

## 3. Confidence intervals: paired percentile bootstrap

`bootstrap_mean_ci` / `paired_bootstrap_ci`:

- **Defaults:** B = 10,000 resamples (a warning below 1,000); confidence level 0.95, which
  must lie in [0.5, 0.999].
- **Algorithm:** each resample draws n indices with replacement from the `bootstrap` random
  stream and computes the mean with `math.fsum`. The percentiles use the linear-interpolation
  convention of `compute_percentile`. The reported standard error is the SD of the
  resampled means.
- **Optional stratified resampling:** within each stratum, keeping each stratum's size, with
  one stream per stratum (`stratum:<label>`).
- **Edge cases:** n < 2 gives no interval (`insufficient_items`). If all values are equal,
  the interval is a single point with warning `degenerate_zero_variance`.

**Interpretation.** The interval describes uncertainty about the mean over a population of
similar items, for these fixed models, prompts, settings and references. It does not cover
run-to-run generation noise, other datasets, or errors in references. It is never evidence of
equivalence on its own, and the percentile bootstrap tends to under-cover at very small n.

## 4. Exact McNemar test (binary, one repetition per arm)

`mcnemar_exact(b, c)`, with b = #(A wrong, B right) and c = #(A right, B wrong).

- **Hypothesis:** among discordant items, either direction is equally likely (P = ½).
- **p-value:** min(1, 2·P(X ≤ min(b, c))) with X ~ Binomial(b + c, ½), computed in exact
  rational arithmetic. The doubling is conservative.
- **Assumptions:** independent items; one binary outcome per item per arm; the same metric
  definition in both arms.
- **Edge cases:** b + c = 0 gives p = 1 and warning `no_discordant_pairs`; b + c < 10 gives
  warning `few_discordant_pairs`.

## 5. Paired sign-flip permutation test (difference test)

`sign_flip_test(differences)` tests the null of no difference, two-sided, for
bounded-continuous metrics and for per-item means of repeated runs.

**Statistic.** The studentized mean t = d̄/(s/√n) is reported. Under any sign flip
Σd_i² is unchanged, so t is a strictly increasing function of the mean. **Studentization
therefore gives exactly the same p-values as the plain mean: it does not improve, and does
not change, the test's validity.** The implementation compares exact sums of signed
differences: floats are converted to integers over a common power-of-two denominator, so
there is no rounding drift.

**What the p-value means:**

| Null hypothesis | Status |
| --- | --- |
| **Sharp exchangeability:** the two arms are interchangeable on every item, so each d_i is symmetric about 0 | **Exact.** All 2ᵏ sign patterns are equally likely given the \|d_i\|. |
| **Zero mean difference** only | **Approximate.** First-order asymptotically valid for independent items with finite variance; not exact under skew or ties. |

**Permutation distribution:**

- **Exact enumeration** (Gray-code order) when k ≤ 20 non-zero differences (limit decided
  from the Stage 1 timing: about 0.5 s at k = 20).
- **Monte Carlo** otherwise: P = 9,999 sign vectors from the `permutation` stream, with
  p = (1 + count)/(P + 1). The p-value is never zero, and its Monte Carlo standard error is
  reported.

**Labels.** `permutation_distribution` (`exact_enumeration` or `monte_carlo`) describes how
the null distribution was computed, not whether the inference is exact. `validity` states
`{"sharp_exchangeability_null": "exact", "mean_difference_null": "approximate"}`. Reports
never say "exact test" without that qualifier.

**Edge cases:** n < 2 gives `insufficient_items`; all differences zero gives p = 1 and
warning `all_differences_zero`.

**Power depends on k, not n.** When only 0 < k < 10 differences are non-zero, the warning
`few_nonzero_differences` is emitted, and `details.minimum_attainable_p_two_sided` reports
2/2ᵏ (1/(P + 1) on the Monte Carlo path). For example, with k ≤ 5 a two-sided test can never
reach α = 0.05, whatever n is.

## 6. Multiple comparisons

`adjust_p_values(p, method)`:

- **Holm** (default; controls the familywise error rate under arbitrary dependence).
- **Benjamini–Hochberg** (`"bh"`, warning `controls_fdr_not_fwer`).
- **None** (warning `no_multiplicity_correction`).

Raw and adjusted values are both reported. Equivalence tests form a separate Holm family
(section 9).

## 7. Significance, practical importance and equivalence are separate

1. **Significance:** a p-value compared with α. A non-significant result is never described
   as "equivalent", "the same" or "no difference".
2. **Practical importance:** `classify_practical_importance(lower, upper, δp)` classifies a CI
   as `exceeds_practical_threshold`, `below_practical_threshold` or
   `inconclusive_relative_to_threshold`. This is descriptive and makes no equivalence claim.
3. **Equivalence:** only through a formal TOST with a margin δe fixed before the analysis
   (sections 8–9).

**Effect sizes.** `paired_effects` reports:
- the mean difference (primary, in metric units);
- the SD of differences;
- d_z (descriptive);
- the probability of superiority, (wins + ½·ties)/n, with ties defined as |d| ≤ 1e-9;
- win, loss and tie counts.

`binary_paired_counts` reports the 2 × 2 table and the risk difference.

**Sample-size warnings:**
- n < 10: `very_small_sample`.
- n < 30: `low_power_wide_intervals`.
- Binary with b + c < 10: `few_discordant_pairs`.
- Sign-flip with 0 < k < 10 non-zero differences: `few_nonzero_differences`.

**Input validation.** Every exported function in `openbench.stats`, including the
low-level helpers `studentized_mean`, `score_statistic`, `binary_exact_p_upper`,
`binary_exact_p_lower` and `binary_exact_interval`, validates direct calls:

- NaN and infinite data values, margins, tolerances and `delta0` raise `ValueError`.
- Integer parameters (resamples, permutations, limits, counts, repetitions, seeds) must be
  integers, so booleans and floats raise `TypeError`. `alpha`, `margin` and
  `decision_alpha` must be real numbers (not booleans).
- Margins must be positive and finite (`tost_bootstrap_t`), or lie in (0, 1) for the
  binary procedures. `delta0` must lie in (-1, 1); `tolerance` must be positive and
  finite; `alpha` and `decision_alpha` must lie in (0, 0.5).
- `binary_paired_counts` accepts the numbers 0 and 1 (int or float) and rejects booleans.

Internal tight loops call an unchecked core (`_score_statistic`) after the public entry
point has validated its inputs, so validation adds no per-cell cost.

## 8. Equivalence for binary metrics: exact unconditional score TOST

`tost_paired_binary_exact(b, c, n, margin, alpha)` applies to binary metrics with one
repetition per arm.

**Model and estimand.**
- The counts (b, c, n − b − c) are trinomial, with q₀₁ = P(A wrong, B right) and
  q₁₀ = P(A right, B wrong).
- The estimand is Δ = q₀₁ − q₁₀, estimated by Δ̂ = (b − c)/n.
- **Nuisance parameter:** at a boundary value Δ₀, q₀₁ = θ + Δ₀ and q₁₀ = θ, with θ in
  [max(0, −Δ₀), (1 − Δ₀)/2].

**Hypotheses.**
- Lower: H01 Δ ≤ −δe, against Δ > −δe.
- Upper: H02 Δ ≥ +δe, against Δ < +δe.
- Equivalence is established (unadjusted) iff p_TOST = max(p₁, p₂) ≤ α. This is the
  intersection–union principle.

**Statistic (Tango 1998).**
- Restricted MLE: θ̃ is the admissible root of 2nθ² + Bθ + C = 0, with
  B = −(b + c) + Δ₀(2n − b + c) and C = −cΔ₀(1 − Δ₀). It is computed in a cancellation-free
  form: (−B + √D)/(4n) if B ≤ 0, otherwise 2C/(−B − √D).
- Score statistic: Z = (b − c − nΔ₀)/√V, with V = n(2θ̃ + Δ₀(1 − Δ₀)).

**Exact unconditional p-value.**
- **Upper:** p₂ = sup over θ of P_θ{Z(+δe) ≤ z_obs + 1e-12}. Ties count as at least as
  extreme.
- **Lower:** p₁ is the same calculation on the swapped counts (c, b).
- **Validity:** for every θ on the boundary, P_θ(p ≤ u) ≤ u, because p ≥ F_θ(z_obs) and the
  probability integral transform holds for discrete Z. So the size is ≤ α at **every** n.
  There is **no minimum sample size** for validity; small n simply gives large p-values, and
  sample-size warnings are reported.
- **Asymptotic p-value:** the normal p-value Φ(Z) is reported as a diagnostic only. It is
  never used for the decision, and never as a fallback.

**Certified numerical supremum.** The tail probability is a polynomial f(θ). The reported
p-value is a **certified upper bound** on its supremum:

1. **Rejection region.** For every m = b + c, the rejection region in b is checked to be a
   prefix {0..k_m} by scanning all outcomes. If it is not, the result is `not_established`
   with reason `non_monotone_rejection_region`. Then f(θ) = Σ_m Bin(m; n, ψ)·BinCDF(k_m; m, π),
   with ψ = 2θ + Δ₀ and π = (θ + Δ₀)/ψ, is evaluated in O(n) by walking the binomial CDF.
2. **Zeroth-order enclosure on a cell [θ₁, θ₂].**
   - Bin(m; n, ψ) is unimodal in ψ, so it is at most its value at ψ = m/n clipped to
     [ψ(θ₁), ψ(θ₂)].
   - BinCDF(k; m, π) is decreasing in π, so it is at most its value at the smaller of
     π(θ₁) and π(θ₂).
   - The product of these per-term maxima bounds every term, so it bounds f on the cell.
3. **First-order enclosure** (cells narrower than 10⁻⁴).
   - f′(θ) = n[G₁ + G₂] − 2n·G₃, where G₁–G₃ are probabilities of three shifted prefix sets
     at size n − 1.
   - Each Gᵢ is enclosed by evaluating the cell probabilities at the favourable cell
     endpoints.
   - This gives f(θ) ≤ f(mid) + (h/2)·max(0, U′, −L′) on the cell.
4. **Branch and bound.** Start with 32 cells. Repeatedly split the cell with the largest
   bound until the gap between the largest bound and the best evaluated value is at most
   10⁻⁹ (above the floating-point allowance below). The bound remains valid if the refinement
   cap is reached, and this is flagged.
5. **Floating-point allowance and upward rounding.** Each evaluation carries an explicit
   absolute allowance A = S·(64ε(log n! + 1) + 8ε) + 16nε, with ε the machine epsilon.
   - S is the **actual** number of probability terms in the walks for the threshold arrays
     involved. It is counted, not assumed, so the bound does not rely on k_m being
     monotone in m.
   - The 64ε(log n! + 1) per term assumes `math.lgamma` is within 8 ulp; the test suite
     checks this for k ≤ 1,001 against 60-digit `decimal` arithmetic (measured worst
     case: 3.21 ulp).
   - It also uses pmf·|k log π| ≤ log m! + 1, so terms with tiny π are covered.
   - The 16nε covers evaluation points whose (ψ, π) are rounded off the exact boundary.
   - **Outward widening:** the cell-enclosure parameters (ψ, π and the q₀₁, q₁₀, R used
     in the derivative enclosure) are widened by 8ε, so their own rounding cannot shrink
     an enclosure.
   - **Rounding up:** the allowance is added to every bound, and the result is rounded up
     to the next float. Early-exit decisions compare exactly that rounded value with α, so
     a decision and a full computation can never disagree.

The reported p-value is therefore never below the true supremum, under that floating-point
model. The model is an engineering bound with safety factors, not a machine-checked
interval-arithmetic proof. Against exact rational evaluation (n ≤ 30), the measured error stays
below 1% of the allowance. The gap is reported (`numerical.*.gap`); typically it is below 2·10⁻⁸.

- **Independent checks** in the test suite:
  - exact-rational polynomial bounds for small n, where p lies between a dense-grid lower
    bound and a Lipschitz upper bound;
  - a NumPy full-enumeration reference;
  - the analytic case n = 20, b = c = 0, which gives 0.9²⁰;
  - the published Tango interval for b = 1, c = 7, n = 21 ([−0.5172, −0.0260], from the
    `contingencytables` package as printed in the `ratesci` vignette);
  - exact enumeration of the production rule's size.
- **Interval.** The (1 − 2α) interval inverts the **same conservative** one-sided tests. It
  uses a 21-point scan (the margin points are added) and bisection to 10⁻⁵, deciding each
  point with the certified bound. It is reported as an open interval (L, U) whose endpoints
  are certified-rejected values, so every non-rejected Δ₀ lies strictly inside. A
  non-monotone scan is flagged `interval_non_monotone`. The decision always comes from p₁ and
  p₂. The interval agrees with it on every reference case and on every table with n ≤ 6.
- **Boundary rule at Δ₀ = ±1.** These points are decided exactly, not computed.
  - H: Δ ≥ −1 and H: Δ ≤ +1 are the whole parameter space, so they are never rejected.
  - H: Δ ≥ +1 forces every item to be (A wrong, B right), so it is rejected exactly when
    b < n. This is also the limit of the test as Δ₀ → 1.
  - H: Δ ≤ −1 is rejected exactly when c < n.
  - So tables where every item is discordant in one direction (b = n or c = n) have an
    interval ending at ±1. For b = n the lower end is 2α^(1/n) − 1, which the test suite
    checks analytically.
- **Computation limit.** n ≤ 500 (decided from the Stage 1 timing: about 40 s per
  comparison with its interval at n = 500). Above this limit the result is
  `not_established` with reason `exact_computation_limit` and no p-value. It **never**
  falls back to the asymptotic test. Callers may pass a larger `max_n` explicitly.

**Assumptions:**
- independent items forming a random sample from the item population;
- one repetition per arm, with the same metric definition in both arms;
- a margin fixed in advance.

For a fixed set of heterogeneous items, the test is expected to be conservative but is not
claimed exact.

## 9. Holm adjustment and intervals for equivalence

`equivalence_family_holm(results, alpha)` adjusts p_TOST across one equivalence family.
Members are the results that have a p_TOST and are not descriptive-only.

**Fields reported per comparison:**
- the margin and α;
- the unadjusted p-values and `result_unadjusted`;
- `adjusted_p_tost` and `familywise_result` (**the headline decision**);
- the family's ID, size and correction;
- the per-comparison interval, with `multiplicity_adjusted: false`.

**Interval policy.**
- **No Holm-related or simultaneous interval is reported.** The per-comparison interval
  agrees with the **unadjusted** decision only.
- **Fixed label:** "90% interval for this comparison alone (not adjusted for multiple
  comparisons). The familywise equivalence decision uses the Holm-adjusted p-value."

## 10. Continuous equivalence: disabled in v0.4.0

`tost_bootstrap_t(differences, margin, alpha)` implements the null-imposed bootstrap-t TOST:
- translated null;
- plug-in resample standard error;
- B = 9,999 resamples (α(B + 1) is an integer);
- (1 + count)/(B + 1) p-values;
- degenerate resamples kept;
- an order-statistic dual interval.

**It never produces an equivalence claim.** The pre-specified calibration did not validate
any minimum sample size: six of 150 cells failed the unchanged criterion, including cells at
n = 500. So every result is `not_established` with reason `method_not_validated` (or an
earlier gate: `insufficient_items`, `degenerate_zero_variance`). Results are marked
`descriptive_only`, and they are excluded from the Holm family. Enabling the method would
need a new, pre-specified calibration protocol and separate approval.

## 11. Determinism

- **Random streams:** each procedure uses its own stream, `random.Random(seed)` with
  seed = first 8 bytes of SHA-256(f"{analysis_seed}|{purpose}|{metric}|{arm_a}|{arm_b}"). The
  purposes are `bootstrap`, `permutation`, `equivalence` and `stratum:<label>`.
- **Same inputs, same results:** identical inputs, parameters and seed give identical
  results, regardless of the order in which analyses run.

## 12. Literature

- **Tango (1998), Statistics in Medicine 17:891–908:** the source of the score statistic.
- **Davidson & MacKinnon (2000):** Monte Carlo test sizing.
- **Efron & Tibshirani (1993, ch. 16):** the translated-data bootstrap test.
- **Hall & Wilson (1991):** bootstrap-test guidelines.
- **Hall (1992):** Edgeworth theory for bootstrap-t.

These citations still need to be checked against the original papers. None of the validity
arguments above depends on them.

## 13. Known limitations

- **Binary:** the size guarantee depends on the trinomial (random-sample) model and on the
  stated floating-point allowance, which is an engineering bound with safety factors, not a
  machine-checked proof.
- **Continuous:** the method is asymptotic and not validated, so it is disabled.
- **Sign-flip:** approximate for the zero-mean-difference null.
- **Holm:** gives familywise decisions only, with no simultaneous intervals.
- **Small datasets:** the built-in 20-item datasets have low power. Equivalence there is
  typically not established, which is not evidence of a difference.
