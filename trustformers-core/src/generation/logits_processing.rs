//! Logits post-processing primitives shared by every decoding strategy.
//!
//! Everything in this module operates on a plain `&[f32]` slice of *next-token*
//! logits (one entry per vocabulary item).  The functions here are the single
//! source of truth for temperature scaling, repetition penalties, top-k / top-p
//! truncation and multinomial sampling; the decoding strategies in
//! [`super::core`], [`super::beam_search`] and [`super::diverse_sampling`] all
//! delegate to them so that the semantics cannot drift apart.
//!
//! Two conventions are used consistently throughout:
//!
//! * **Masking is done with `f32::NEG_INFINITY`.**  A masked logit maps to a
//!   probability of exactly zero under [`softmax`], and
//!   [`sample_index_from_cumulative`] can never return an index whose
//!   probability is zero, so a truncated distribution is genuinely truncated.
//! * **Ties break towards the lowest index.**  [`argmax`] returns the *first*
//!   maximal element and [`top_k_filter`] keeps exactly `k` entries, breaking
//!   ties by index.  This makes `top_k = 1` provably identical to greedy
//!   decoding.

use crate::errors::{Result, TrustformersError};
use scirs2_core::random::{RngExt, StdRng};

/// Numerically stable softmax over a slice of logits.
///
/// Returns an error when `logits` is empty or when every entry is masked
/// (i.e. `-inf`), because no probability distribution exists in that case.
pub fn softmax(logits: &[f32]) -> Result<Vec<f32>> {
    if logits.is_empty() {
        return Err(TrustformersError::invalid_input(
            "softmax received an empty logits slice".to_string(),
        ));
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return Err(TrustformersError::tensor_op_error(
            "all logits are masked or non-finite; no distribution exists",
            "softmax",
        ));
    }

    let exp_logits: Vec<f32> = logits
        .iter()
        .map(
            |&x| {
                if x == f32::NEG_INFINITY {
                    0.0
                } else {
                    (x - max_logit).exp()
                }
            },
        )
        .collect();

    let sum_exp: f32 = exp_logits.iter().sum();
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return Err(TrustformersError::tensor_op_error(
            "softmax denominator is not a positive finite number",
            "softmax",
        ));
    }

    Ok(exp_logits.into_iter().map(|x| x / sum_exp).collect())
}

/// Numerically stable log-softmax over a slice of logits.
///
/// Masked entries (`-inf`) stay `-inf`, which is the correct log-probability
/// of an impossible token and composes with additive beam scoring.
pub fn log_softmax(logits: &[f32]) -> Result<Vec<f32>> {
    if logits.is_empty() {
        return Err(TrustformersError::invalid_input(
            "log_softmax received an empty logits slice".to_string(),
        ));
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return Err(TrustformersError::tensor_op_error(
            "all logits are masked or non-finite; no distribution exists",
            "log_softmax",
        ));
    }

    let sum_exp: f32 = logits
        .iter()
        .map(
            |&x| {
                if x == f32::NEG_INFINITY {
                    0.0
                } else {
                    (x - max_logit).exp()
                }
            },
        )
        .sum();

    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return Err(TrustformersError::tensor_op_error(
            "log_softmax denominator is not a positive finite number",
            "log_softmax",
        ));
    }

    let log_sum_exp = max_logit + sum_exp.ln();
    Ok(logits
        .iter()
        .map(
            |&x| {
                if x == f32::NEG_INFINITY {
                    f32::NEG_INFINITY
                } else {
                    x - log_sum_exp
                }
            },
        )
        .collect())
}

/// Index of the first maximal element.
///
/// Masked (`-inf`) and NaN entries are never selectable, mirroring
/// [`sample_index_from_cumulative`]: a token that a filter forbade must not
/// come back through greedy decoding.  Returns an error when the slice is
/// empty or when nothing is selectable, rather than inventing a token.
pub fn argmax(values: &[f32]) -> Result<usize> {
    let mut best_idx: Option<usize> = None;
    let mut best_val = f32::NEG_INFINITY;

    for (idx, &value) in values.iter().enumerate() {
        if value.is_nan() || value == f32::NEG_INFINITY {
            continue;
        }
        if best_idx.is_none() || value > best_val {
            best_idx = Some(idx);
            best_val = value;
        }
    }

    best_idx.ok_or_else(|| {
        TrustformersError::invalid_input(
            "argmax has no selectable candidate: the slice is empty or every entry is \
             masked (-inf) or NaN"
                .to_string(),
        )
    })
}

/// Divide every logit by `temperature`, in place.
///
/// A temperature below 1 sharpens the distribution, above 1 flattens it.
/// Returns an error for non-positive or non-finite temperatures instead of
/// silently producing garbage.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) -> Result<()> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(TrustformersError::invalid_input(format!(
            "temperature must be a positive finite number, got {temperature}"
        )));
    }
    if (temperature - 1.0).abs() <= f32::EPSILON {
        return Ok(());
    }
    for logit in logits.iter_mut() {
        *logit /= temperature;
    }
    Ok(())
}

/// Apply the HuggingFace-style repetition penalty over an iterator of token ids.
///
/// The penalty is *sign aware*: positive logits are divided by `penalty` and
/// negative logits are multiplied by it, so that `penalty > 1` always makes an
/// already-seen token less likely.  Naively dividing would *reward* repetition
/// whenever the logit (or log-probability) is negative.
///
/// A `penalty` that is not a positive finite number, or is exactly 1.0, is a
/// no-op.
pub fn apply_repetition_penalty_indexed(
    logits: &mut [f32],
    previous_tokens: impl IntoIterator<Item = usize>,
    penalty: f32,
) {
    if !penalty.is_finite() || penalty <= 0.0 || (penalty - 1.0).abs() <= f32::EPSILON {
        return;
    }
    for token in previous_tokens {
        if let Some(logit) = logits.get_mut(token) {
            if *logit > 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}

/// Slice-taking convenience wrapper around [`apply_repetition_penalty_indexed`].
pub fn apply_repetition_penalty(logits: &mut [f32], previous_tokens: &[usize], penalty: f32) {
    apply_repetition_penalty_indexed(logits, previous_tokens.iter().copied(), penalty);
}

/// Keep exactly the `k` highest logits and mask the rest with `-inf`.
///
/// Ties are broken towards the lower token id, so the retained set is fully
/// deterministic and `k = 1` selects precisely [`argmax`].  Selection uses
/// `select_nth_unstable_by`, i.e. `O(V)` rather than `O(V log V)`.
pub fn top_k_filter(logits: &mut [f32], k: usize) -> Result<()> {
    if logits.is_empty() {
        return Err(TrustformersError::invalid_input(
            "top_k_filter received an empty logits slice".to_string(),
        ));
    }
    if k == 0 {
        return Err(TrustformersError::invalid_input(
            "top_k_filter requires k >= 1".to_string(),
        ));
    }
    if k >= logits.len() {
        return Ok(());
    }

    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    {
        let view: &[f32] = logits;
        order.select_nth_unstable_by(k - 1, |&a, &b| {
            view[b as usize].total_cmp(&view[a as usize]).then_with(|| a.cmp(&b))
        });
    }

    let mut keep = vec![false; logits.len()];
    for &idx in &order[..k] {
        keep[idx as usize] = true;
    }
    for (idx, logit) in logits.iter_mut().enumerate() {
        if !keep[idx] {
            *logit = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

/// Nucleus (top-p) truncation, in place.
///
/// Keeps the smallest set of tokens, ordered by descending probability, whose
/// cumulative probability mass reaches `p`; everything else is masked with
/// `-inf`.  The token that crosses the threshold is retained (so the kept mass
/// is always `>= p`), and at least one token always survives.
pub fn top_p_filter(logits: &mut [f32], p: f32) -> Result<()> {
    if logits.is_empty() {
        return Err(TrustformersError::invalid_input(
            "top_p_filter received an empty logits slice".to_string(),
        ));
    }
    if !p.is_finite() || p <= 0.0 || p > 1.0 {
        return Err(TrustformersError::invalid_input(format!(
            "top_p requires p in (0, 1], got {p}"
        )));
    }

    let probs = softmax(logits)?;

    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        probs[b as usize].total_cmp(&probs[a as usize]).then_with(|| a.cmp(&b))
    });

    let mut nucleus = vec![false; logits.len()];
    let mut cumulative = 0.0_f32;
    for &idx in &order {
        nucleus[idx as usize] = true;
        cumulative += probs[idx as usize];
        if cumulative >= p {
            break;
        }
    }

    for (idx, logit) in logits.iter_mut().enumerate() {
        if !nucleus[idx] {
            *logit = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

/// Pick an index from `probs` given a uniform draw `u` in `[0, 1)`.
///
/// This is the deterministic core of multinomial sampling, factored out so it
/// can be tested against hand-computed values without an RNG.  `probs` does not
/// have to be normalised; the draw is scaled by the total mass.
///
/// Indices whose probability is exactly zero can never be returned: the
/// comparison is strict (`target < cumulative`), so a zero-width bucket is
/// unreachable even for `u == 0.0`.
pub fn sample_index_from_cumulative(probs: &[f32], u: f32) -> Result<usize> {
    if probs.is_empty() {
        return Err(TrustformersError::invalid_input(
            "sample_index_from_cumulative received an empty distribution".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&u) {
        return Err(TrustformersError::invalid_input(format!(
            "uniform draw must lie in [0, 1), got {u}"
        )));
    }

    let mut total = 0.0_f32;
    for &prob in probs {
        if prob.is_nan() || prob < 0.0 {
            return Err(TrustformersError::invalid_input(format!(
                "probabilities must be non-negative and finite, got {prob}"
            )));
        }
        total += prob;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(TrustformersError::invalid_input(
            "probability distribution has zero total mass".to_string(),
        ));
    }

    let target = u * total;
    let mut cumulative = 0.0_f32;
    let mut last_support: Option<usize> = None;
    for (idx, &prob) in probs.iter().enumerate() {
        if prob <= 0.0 {
            continue;
        }
        last_support = Some(idx);
        cumulative += prob;
        if target < cumulative {
            return Ok(idx);
        }
    }

    // Only reachable through floating-point summation error; fall back to the
    // last index that actually carries probability mass.
    last_support.ok_or_else(|| {
        TrustformersError::invalid_input("probability distribution has no support".to_string())
    })
}

/// Draw a single index from `probs` using `rng`.
pub fn multinomial_sample(probs: &[f32], rng: &mut StdRng) -> Result<usize> {
    let raw: f32 = rng.random::<f32>();
    // `random::<f32>()` already yields [0, 1); the clamp only defends against a
    // backend that returns exactly 1.0.
    let u = if raw >= 1.0 { 1.0 - f32::EPSILON } else { raw };
    sample_index_from_cumulative(probs, u)
}

/// Return the tokens whose emission would repeat an n-gram already present in
/// `tokens`.
///
/// Generic over the token representation so both the `usize` sequences used by
/// [`super::core`] and the `u32` sequences used by [`super::beam_search`] share
/// one implementation.
pub fn forbidden_ngram_tokens<T>(tokens: &[T], ngram_size: usize) -> Vec<T>
where
    T: Copy + Ord,
{
    if ngram_size == 0 || tokens.len() + 1 < ngram_size {
        return Vec::new();
    }

    let window_size = ngram_size - 1;
    let suffix = &tokens[tokens.len() - window_size..];

    let mut forbidden = Vec::new();
    for start in 0..tokens.len().saturating_sub(window_size) {
        if &tokens[start..start + window_size] == suffix {
            forbidden.push(tokens[start + window_size]);
        }
    }

    forbidden.sort_unstable();
    forbidden.dedup();
    forbidden
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::SeedableRng;

    #[test]
    fn test_softmax_matches_hand_computed_values() {
        // logits [0, ln 2, ln 4] -> unnormalised [1, 2, 4], total 7.
        let probs = softmax(&[0.0, 2.0_f32.ln(), 4.0_f32.ln()]).expect("softmax");
        assert!((probs[0] - 1.0 / 7.0).abs() < 1e-6, "{probs:?}");
        assert!((probs[1] - 2.0 / 7.0).abs() < 1e-6, "{probs:?}");
        assert!((probs[2] - 4.0 / 7.0).abs() < 1e-6, "{probs:?}");
    }

    #[test]
    fn test_softmax_masked_entries_have_zero_probability() {
        let probs = softmax(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]).expect("softmax");
        assert_eq!(probs[0], 0.0);
        assert_eq!(probs[2], 0.0);
        assert!((probs[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_all_masked_is_error() {
        assert!(softmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]).is_err());
        assert!(softmax(&[]).is_err());
    }

    #[test]
    fn test_log_softmax_is_log_of_softmax() {
        let logits = [0.5_f32, -1.25, 3.0, 0.0];
        let probs = softmax(&logits).expect("softmax");
        let log_probs = log_softmax(&logits).expect("log_softmax");
        for (lp, p) in log_probs.iter().zip(probs.iter()) {
            assert!((lp - p.ln()).abs() < 1e-5, "{lp} vs {}", p.ln());
        }
    }

    #[test]
    fn test_log_softmax_keeps_masked_at_neg_infinity() {
        let log_probs = log_softmax(&[f32::NEG_INFINITY, 0.0]).expect("log_softmax");
        assert_eq!(log_probs[0], f32::NEG_INFINITY);
        assert!(log_probs[1].abs() < 1e-6);
    }

    #[test]
    fn test_argmax_returns_first_maximum() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]).expect("argmax"), 1);
        assert!(argmax(&[]).is_err());
    }

    #[test]
    fn test_argmax_never_returns_a_masked_token() {
        // Regression: `max_by` picked index 0 when every logit was -inf, so
        // greedy decoding emitted a token that a filter had explicitly
        // forbidden while every sampling path errored out instead.
        assert!(
            argmax(&[f32::NEG_INFINITY; 4]).is_err(),
            "a fully masked distribution has no legal token"
        );
        assert_eq!(
            argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY, -5.0]).expect("argmax"),
            2
        );
        assert!(argmax(&[f32::NAN, f32::NEG_INFINITY]).is_err());
    }

    #[test]
    fn test_apply_temperature_scales_and_validates() {
        let mut logits = [2.0_f32, 4.0];
        apply_temperature(&mut logits, 2.0).expect("temperature");
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!(apply_temperature(&mut logits, 0.0).is_err());
        assert!(apply_temperature(&mut logits, -1.0).is_err());
    }

    #[test]
    fn test_repetition_penalty_is_sign_aware() {
        // Regression: a naive `logit /= penalty` *rewards* repetition when the
        // logit is negative, which is the common case for log-probabilities.
        let mut logits = [-1.0_f32, 2.0, -0.5];
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        assert!(
            (logits[0] - (-2.0)).abs() < 1e-6,
            "negative logit must shrink"
        );
        assert!((logits[1] - 1.0).abs() < 1e-6, "positive logit must shrink");
        assert!(
            (logits[2] - (-0.5)).abs() < 1e-6,
            "untouched token unchanged"
        );
    }

    #[test]
    fn test_top_k_filter_keeps_exactly_k_with_index_tiebreak() {
        let mut logits = [1.0_f32, 5.0, 5.0, 0.0];
        top_k_filter(&mut logits, 2).expect("top_k");
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert!((logits[1] - 5.0).abs() < 1e-6);
        assert!((logits[2] - 5.0).abs() < 1e-6);

        let mut tied = [5.0_f32, 5.0, 5.0];
        top_k_filter(&mut tied, 1).expect("top_k");
        assert!((tied[0] - 5.0).abs() < 1e-6, "lowest index wins ties");
        assert_eq!(tied[1], f32::NEG_INFINITY);
        assert_eq!(tied[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_filter_k_one_equals_argmax() {
        let logits = [0.1_f32, -3.0, 7.5, 2.0, 7.4];
        let mut filtered = logits;
        top_k_filter(&mut filtered, 1).expect("top_k");
        let kept: Vec<usize> = filtered
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(kept, vec![argmax(&logits).expect("argmax")]);
    }

    #[test]
    fn test_top_p_filter_respects_probability_mass() {
        // Probabilities proportional to [1, 2, 4, 8]; total 15.
        // Descending: 8/15 = .533, +4/15 = .800, +2/15 = .933, +1/15 = 1.0
        let base = [0.0_f32, 2.0_f32.ln(), 4.0_f32.ln(), 8.0_f32.ln()];

        let mut p60 = base;
        top_p_filter(&mut p60, 0.6).expect("top_p");
        assert!(p60[3].is_finite(), "0.533 < 0.6 so token 2 is also needed");
        assert!(p60[2].is_finite());
        assert_eq!(p60[1], f32::NEG_INFINITY);
        assert_eq!(p60[0], f32::NEG_INFINITY);

        let mut p50 = base;
        top_p_filter(&mut p50, 0.5).expect("top_p");
        assert!(p50[3].is_finite());
        assert_eq!(p50[2], f32::NEG_INFINITY);
        assert_eq!(p50[1], f32::NEG_INFINITY);
        assert_eq!(p50[0], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_p_filter_rejects_invalid_p() {
        let mut logits = [1.0_f32, 2.0];
        assert!(top_p_filter(&mut logits, 0.0).is_err());
        assert!(top_p_filter(&mut logits, 1.5).is_err());
    }

    #[test]
    fn test_sample_index_from_cumulative_hand_computed_buckets() {
        // Buckets: [0, .25) -> 0, [.25, .75) -> 1, [.75, 1) -> 2
        let probs = [0.25_f32, 0.5, 0.25];
        assert_eq!(sample_index_from_cumulative(&probs, 0.0).expect("s"), 0);
        assert_eq!(sample_index_from_cumulative(&probs, 0.24).expect("s"), 0);
        assert_eq!(sample_index_from_cumulative(&probs, 0.25).expect("s"), 1);
        assert_eq!(sample_index_from_cumulative(&probs, 0.74).expect("s"), 1);
        assert_eq!(sample_index_from_cumulative(&probs, 0.75).expect("s"), 2);
        assert_eq!(sample_index_from_cumulative(&probs, 0.999).expect("s"), 2);
    }

    #[test]
    fn test_sample_index_never_returns_zero_probability_token() {
        // Regression: `u <= cumsum` would return index 0 for u = 0.0 even
        // though token 0 was filtered out.
        let probs = [0.0_f32, 1.0];
        assert_eq!(sample_index_from_cumulative(&probs, 0.0).expect("s"), 1);
        assert_eq!(sample_index_from_cumulative(&probs, 0.5).expect("s"), 1);

        let sparse = [0.0_f32, 0.0, 0.5, 0.0, 0.5];
        for step in 0..100 {
            let u = step as f32 / 100.0;
            let idx = sample_index_from_cumulative(&sparse, u).expect("s");
            assert!(idx == 2 || idx == 4, "u={u} produced {idx}");
        }
    }

    #[test]
    fn test_sample_index_rejects_bad_input() {
        assert!(sample_index_from_cumulative(&[], 0.5).is_err());
        assert!(sample_index_from_cumulative(&[0.0, 0.0], 0.5).is_err());
        assert!(sample_index_from_cumulative(&[1.0], 1.0).is_err());
        assert!(sample_index_from_cumulative(&[-1.0, 1.0], 0.5).is_err());
    }

    #[test]
    fn test_multinomial_sample_is_seed_deterministic_and_in_support() {
        let probs = [0.0_f32, 0.7, 0.0, 0.3];
        let mut rng_a = StdRng::seed_from_u64(20240917);
        let mut rng_b = StdRng::seed_from_u64(20240917);
        let mut counts = [0usize; 4];
        for _ in 0..500 {
            let a = multinomial_sample(&probs, &mut rng_a).expect("sample");
            let b = multinomial_sample(&probs, &mut rng_b).expect("sample");
            assert_eq!(a, b, "same seed must give the same stream");
            assert!(a == 1 || a == 3, "sampled outside the support: {a}");
            counts[a] += 1;
        }
        assert_eq!(counts[0], 0);
        assert_eq!(counts[2], 0);
        assert!(counts[1] > counts[3], "0.7 should dominate 0.3: {counts:?}");
    }

    #[test]
    fn test_forbidden_ngram_tokens_generic() {
        assert!(forbidden_ngram_tokens::<u32>(&[1, 2, 1], 0).is_empty());
        assert_eq!(forbidden_ngram_tokens::<u32>(&[1, 2, 1], 2), vec![2]);
        assert_eq!(
            forbidden_ngram_tokens::<usize>(&[1, 2, 3, 1, 2], 3),
            vec![3]
        );
        assert!(forbidden_ngram_tokens::<usize>(&[1, 2, 3], 2).is_empty());
    }
}
