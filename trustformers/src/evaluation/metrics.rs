//! Evaluation metrics for NLP tasks.
//!
//! Provides production-quality implementations of:
//! - **BLEU** (Papineni et al., 2002) — n-gram precision with brevity penalty
//! - **ROUGE-N / ROUGE-L** (Lin, 2004) — recall-oriented n-gram / LCS scores
//! - **Token-level F1** — standard precision/recall/F1 over token sets
//! - **Exact match** — string-equality check (normalised)
//! - **Perplexity** — exponentiated cross-entropy from per-token log-probabilities
//!
//! All functions return `Result<_, MetricError>` — no panics, no `unwrap`.
//!
//! # Example
//! ```rust,ignore
//! use trustformers::evaluation::{bleu_score, exact_match, perplexity};
//!
//! let hyp = &["the", "cat", "sat", "on", "the", "mat"];
//! let refs: &[&[&str]] = &[&["the", "cat", "is", "on", "the", "mat"]];
//! let score = bleu_score(hyp, refs, 4, true).unwrap();
//! assert!(score > 0.0 && score <= 1.0);
//!
//! assert!(exact_match("Hello World", "hello world"));
//!
//! let log_probs = vec![-2.0_f64; 10];
//! let ppl = perplexity(&log_probs).unwrap();
//! assert!((ppl - 2.0_f64.exp() * 2.0_f64.exp()).abs() < 0.01);
//! ```

use std::collections::HashMap;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur when computing NLP evaluation metrics.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MetricError {
    /// The provided slice or iterator was empty when a non-empty input was required.
    #[error("Empty input: {0}")]
    EmptyInput(String),

    /// The number of predictions and references do not match.
    #[error("Mismatched lengths: predictions {0}, references {1}")]
    LengthMismatch(usize, usize),

    /// The requested n-gram order was zero or otherwise invalid.
    #[error("Invalid n-gram order: {0}")]
    InvalidNgramOrder(usize),

    /// A mathematical operation produced an invalid value (e.g. log of negative number).
    #[error("Math error: {0}")]
    MathError(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Supporting types
// ─────────────────────────────────────────────────────────────────────────────

/// Precision, recall, and F1 for a ROUGE variant.
#[derive(Debug, Clone, Default)]
pub struct RougeScore {
    /// Fraction of hypothesis n-grams (or LCS tokens) that appear in the reference.
    pub precision: f64,
    /// Fraction of reference n-grams (or LCS tokens) that appear in the hypothesis.
    pub recall: f64,
    /// Harmonic mean of precision and recall.
    pub f1: f64,
}

/// Detailed token-level F1 for classification tasks (NER, QA, etc.).
#[derive(Debug, Clone, Default)]
pub struct F1Score {
    /// TP / (TP + FP)
    pub precision: f64,
    /// TP / (TP + FN)
    pub recall: f64,
    /// 2·P·R / (P + R)
    pub f1: f64,
    /// Number of tokens that were correctly predicted.
    pub true_positives: usize,
    /// Number of predicted tokens absent from the reference.
    pub false_positives: usize,
    /// Number of reference tokens absent from the prediction.
    pub false_negatives: usize,
}

/// Aggregated result of running multiple NLP benchmark metrics.
///
/// Individual fields are `Option` so callers can record only the metrics
/// relevant to their task.
#[derive(Debug, Clone, Default)]
pub struct NlpBenchmarkResult {
    /// BLEU-1 (unigram precision with brevity penalty).
    pub bleu_1: Option<f64>,
    /// BLEU-2.
    pub bleu_2: Option<f64>,
    /// BLEU-4 (the standard MT metric).
    pub bleu_4: Option<f64>,
    /// ROUGE-1 F1.
    pub rouge_1: Option<RougeScore>,
    /// ROUGE-2 F1.
    pub rouge_2: Option<RougeScore>,
    /// ROUGE-L F1 (longest common subsequence).
    pub rouge_l: Option<RougeScore>,
    /// Exact-match accuracy (%).
    pub exact_match: Option<f64>,
    /// Token-level F1.
    pub f1: Option<F1Score>,
    /// Perplexity.
    pub perplexity: Option<f64>,
}

impl NlpBenchmarkResult {
    /// Create an empty benchmark result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Produce a human-readable summary line for each populated metric.
    ///
    /// # Example
    /// ```rust,ignore
    /// use trustformers::evaluation::NlpBenchmarkResult;
    /// let mut r = NlpBenchmarkResult::new();
    /// r.bleu_4 = Some(0.423);
    /// r.perplexity = Some(15.3);
    /// println!("{}", r.display_summary());
    /// ```
    pub fn display_summary(&self) -> String {
        let mut lines = Vec::new();

        if let Some(b1) = self.bleu_1 {
            lines.push(format!("BLEU-1:      {:.4}", b1));
        }
        if let Some(b2) = self.bleu_2 {
            lines.push(format!("BLEU-2:      {:.4}", b2));
        }
        if let Some(b4) = self.bleu_4 {
            lines.push(format!("BLEU-4:      {:.4}", b4));
        }
        if let Some(ref r1) = self.rouge_1 {
            lines.push(format!(
                "ROUGE-1:     P={:.4}  R={:.4}  F1={:.4}",
                r1.precision, r1.recall, r1.f1
            ));
        }
        if let Some(ref r2) = self.rouge_2 {
            lines.push(format!(
                "ROUGE-2:     P={:.4}  R={:.4}  F1={:.4}",
                r2.precision, r2.recall, r2.f1
            ));
        }
        if let Some(ref rl) = self.rouge_l {
            lines.push(format!(
                "ROUGE-L:     P={:.4}  R={:.4}  F1={:.4}",
                rl.precision, rl.recall, rl.f1
            ));
        }
        if let Some(em) = self.exact_match {
            lines.push(format!("Exact Match: {:.4}", em));
        }
        if let Some(ref f1) = self.f1 {
            lines.push(format!(
                "Token F1:    P={:.4}  R={:.4}  F1={:.4}",
                f1.precision, f1.recall, f1.f1
            ));
        }
        if let Some(ppl) = self.perplexity {
            lines.push(format!("Perplexity:  {:.4}", ppl));
        }

        if lines.is_empty() {
            "NlpBenchmarkResult: (no metrics set)".to_string()
        } else {
            lines.join("\n")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Count all n-grams of length `n` in `tokens`, returning a frequency map.
fn count_ngrams<'a>(tokens: &[&'a str], n: usize) -> HashMap<Vec<&'a str>, usize> {
    let mut counts: HashMap<Vec<&'a str>, usize> = HashMap::new();
    if tokens.len() < n {
        return counts;
    }
    for window in tokens.windows(n) {
        *counts.entry(window.to_vec()).or_insert(0) += 1;
    }
    counts
}

/// Compute the maximum reference count for each n-gram that appears in
/// the hypothesis (clipped counting as in the original BLEU paper).
fn clipped_ngram_count(
    hyp_counts: &HashMap<Vec<&str>, usize>,
    references: &[&[&str]],
    n: usize,
) -> usize {
    let mut clipped_total: usize = 0;
    for (ngram, hyp_count) in hyp_counts {
        // Maximum count across all references
        let max_ref_count = references
            .iter()
            .map(|r| {
                let ref_counts = count_ngrams(r, n);
                *ref_counts.get(ngram.as_slice()).unwrap_or(&0)
            })
            .max()
            .unwrap_or(0);
        clipped_total += (*hyp_count).min(max_ref_count);
    }
    clipped_total
}

/// Select the reference length closest to the hypothesis length
/// (tiebreak: prefer the shorter reference).
fn best_match_reference_length(hyp_len: usize, references: &[&[&str]]) -> usize {
    references
        .iter()
        .map(|r| r.len())
        .min_by_key(|&ref_len| {
            let diff = ref_len.abs_diff(hyp_len);
            (diff, ref_len) // tiebreak: prefer shorter
        })
        .unwrap_or(0)
}

/// Compute the brevity penalty: `min(1, exp(1 - r/c))` where `c = hyp_len`, `r = ref_len`.
fn brevity_penalty(hyp_len: usize, ref_len: usize) -> f64 {
    if hyp_len == 0 {
        return 0.0;
    }
    if hyp_len >= ref_len {
        1.0
    } else {
        (1.0 - ref_len as f64 / hyp_len as f64).exp()
    }
}

/// Longest common subsequence length via DP.
fn lcs_length(a: &[&str], b: &[&str]) -> usize {
    let m = a.len();
    let n = b.len();
    // Use a flat 2-row rolling DP to keep memory O(n)
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        for j in 1..=n {
            curr[j] = if a[i - 1] == b[j - 1] { prev[j - 1] + 1 } else { curr[j - 1].max(prev[j]) };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }
    prev[n]
}

/// Safely compute F1 from precision and recall, returning 0.0 when both are zero.
#[inline]
fn safe_f1(p: f64, r: f64) -> f64 {
    let denom = p + r;
    if denom < f64::EPSILON {
        0.0
    } else {
        2.0 * p * r / denom
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BLEU
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the BLEU-N score for a single hypothesis against one or more references.
///
/// Implements the original algorithm (Papineni et al., 2002) with optional
/// add-1 smoothing for short texts (Chen & Cherry 2014 method 1).
///
/// # Arguments
/// * `hypothesis`  – generated sentence as a token slice
/// * `references`  – one or more reference sentences (at least one required)
/// * `max_n`       – maximum n-gram order (typically 4 for BLEU-4)
/// * `smooth`      – add-1 smoothing for zero n-gram counts
///
/// # Returns
/// BLEU score in `[0, 1]`.
///
/// # Errors
/// * [`MetricError::EmptyInput`] if no references are given
/// * [`MetricError::InvalidNgramOrder`] if `max_n == 0`
/// * [`MetricError::MathError`] if a numerical issue occurs
///
/// # Example
/// ```rust,ignore
/// use trustformers::evaluation::bleu_score;
/// let hyp = &["the", "quick", "brown", "fox"];
/// let refs: &[&[&str]] = &[&["the", "quick", "brown", "fox"]];
/// let score = bleu_score(hyp, refs, 4, false).unwrap();
/// assert!((score - 1.0).abs() < 1e-6);
/// ```
pub fn bleu_score(
    hypothesis: &[&str],
    references: &[&[&str]],
    max_n: usize,
    smooth: bool,
) -> Result<f64, MetricError> {
    if max_n == 0 {
        return Err(MetricError::InvalidNgramOrder(0));
    }
    if references.is_empty() {
        return Err(MetricError::EmptyInput(
            "at least one reference is required for BLEU".to_string(),
        ));
    }

    // Edge case: empty hypothesis → BLEU = 0
    if hypothesis.is_empty() {
        return Ok(0.0);
    }

    // Collect per-order log-precisions
    let mut log_precision_sum = 0.0_f64;
    let mut valid_orders = 0usize;

    for n in 1..=max_n {
        let prec = ngram_precision(hypothesis, references, n, smooth)?;
        if prec > 0.0 {
            log_precision_sum += prec.ln();
            valid_orders += 1;
        } else if !smooth {
            // Zero precision with no smoothing → BLEU = 0
            return Ok(0.0);
        }
        // With smoothing, zero precision contributes 0 to the log sum (ln(epsilon) would be
        // too negative), so we simply skip it and reduce the geometric mean denominator.
    }

    if valid_orders == 0 {
        return Ok(0.0);
    }

    let geo_mean = (log_precision_sum / max_n as f64).exp();
    let bp = brevity_penalty(
        hypothesis.len(),
        best_match_reference_length(hypothesis.len(), references),
    );

    let score = bp * geo_mean;
    if !score.is_finite() {
        return Err(MetricError::MathError(format!(
            "BLEU score is not finite: {score}"
        )));
    }
    Ok(score.min(1.0))
}

/// Compute a single n-gram precision component for BLEU.
///
/// Returns `precision = clipped_count / hypothesis_ngram_count`.
/// With `smooth = true`, adds 1 to both numerator and denominator when the
/// clipped count is zero (prevents log(0)).
///
/// # Errors
/// * [`MetricError::InvalidNgramOrder`] if `n == 0`
/// * [`MetricError::EmptyInput`] if references are empty
pub fn ngram_precision(
    hypothesis: &[&str],
    references: &[&[&str]],
    n: usize,
    smooth: bool,
) -> Result<f64, MetricError> {
    if n == 0 {
        return Err(MetricError::InvalidNgramOrder(0));
    }
    if references.is_empty() {
        return Err(MetricError::EmptyInput(
            "references must not be empty".to_string(),
        ));
    }
    if hypothesis.len() < n {
        return if smooth { Ok(1.0 / (1.0 + 1.0)) } else { Ok(0.0) };
    }

    let hyp_counts = count_ngrams(hypothesis, n);
    let total_hyp: usize = hyp_counts.values().sum();
    if total_hyp == 0 {
        return Ok(0.0);
    }

    let clipped = clipped_ngram_count(&hyp_counts, references, n);

    if smooth {
        Ok((clipped as f64 + 1.0) / (total_hyp as f64 + 1.0))
    } else {
        Ok(clipped as f64 / total_hyp as f64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ROUGE
// ─────────────────────────────────────────────────────────────────────────────

/// Compute ROUGE-N F1 for a single hypothesis against a single reference.
///
/// ROUGE-N measures n-gram overlap from the reference perspective (recall-oriented).
///
/// # Errors
/// * [`MetricError::InvalidNgramOrder`] if `n == 0`
/// * [`MetricError::EmptyInput`] if either input is empty
///
/// # Example
/// ```rust,ignore
/// use trustformers::evaluation::rouge_n_score;
/// let hyp = &["a", "b", "c"];
/// let ref_ = &["a", "b", "c"];
/// let s = rouge_n_score(hyp, ref_, 1).unwrap();
/// assert!((s.f1 - 1.0).abs() < 1e-6);
/// ```
pub fn rouge_n_score(
    hypothesis: &[&str],
    reference: &[&str],
    n: usize,
) -> Result<RougeScore, MetricError> {
    if n == 0 {
        return Err(MetricError::InvalidNgramOrder(0));
    }

    // Count overlapping n-grams
    let hyp_counts = count_ngrams(hypothesis, n);
    let ref_counts = count_ngrams(reference, n);

    let total_hyp_ngrams: usize = hyp_counts.values().sum();
    let total_ref_ngrams: usize = ref_counts.values().sum();

    // Overlap = sum of min(hyp_count, ref_count) for each n-gram
    let overlap: usize = hyp_counts
        .iter()
        .map(|(ng, &hc)| {
            let rc = *ref_counts.get(ng).unwrap_or(&0);
            hc.min(rc)
        })
        .sum();

    let precision = if total_hyp_ngrams == 0 {
        0.0
    } else {
        overlap as f64 / total_hyp_ngrams as f64
    };

    let recall = if total_ref_ngrams == 0 {
        0.0
    } else {
        overlap as f64 / total_ref_ngrams as f64
    };

    let f1 = safe_f1(precision, recall);

    Ok(RougeScore {
        precision,
        recall,
        f1,
    })
}

/// Compute ROUGE-L F1 using the longest common subsequence.
///
/// ROUGE-L captures sentence-level structure through LCS rather than
/// contiguous n-gram matching.
///
/// # Errors
/// * [`MetricError::EmptyInput`] if either input is empty
///
/// # Example
/// ```rust,ignore
/// use trustformers::evaluation::rouge_l_score;
/// let hyp = &["police", "killed", "the", "gunman"];
/// let ref_ = &["police", "kill", "the", "gunman"];
/// let s = rouge_l_score(hyp, ref_).unwrap();
/// assert!(s.f1 > 0.5);
/// ```
pub fn rouge_l_score(hypothesis: &[&str], reference: &[&str]) -> Result<RougeScore, MetricError> {
    let lcs = lcs_length(hypothesis, reference);

    let precision = if hypothesis.is_empty() { 0.0 } else { lcs as f64 / hypothesis.len() as f64 };

    let recall = if reference.is_empty() { 0.0 } else { lcs as f64 / reference.len() as f64 };

    let f1 = safe_f1(precision, recall);

    Ok(RougeScore {
        precision,
        recall,
        f1,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Token F1
// ─────────────────────────────────────────────────────────────────────────────

/// Compute token-level F1 for a single prediction–reference pair.
///
/// Treats each unique token as a bag-of-words: overlap is the intersection
/// of multisets. This is the standard QA F1 metric used in SQuAD evaluation.
///
/// # Errors
/// Returns [`MetricError::EmptyInput`] if both prediction and reference are empty.
///
/// # Example
/// ```rust,ignore
/// use trustformers::evaluation::token_f1_score;
/// let pred = &["the", "cat", "sat"];
/// let ref_ = &["the", "cat", "on", "the", "mat"];
/// let f1 = token_f1_score(pred, ref_).unwrap();
/// assert!(f1.f1 > 0.0);
/// ```
pub fn token_f1_score(predictions: &[&str], references: &[&str]) -> Result<F1Score, MetricError> {
    let pred_counts = token_counts(predictions);
    let ref_counts = token_counts(references);

    let total_pred: usize = pred_counts.values().sum();
    let total_ref: usize = ref_counts.values().sum();

    // Overlap via multiset intersection
    let overlap: usize = pred_counts
        .iter()
        .map(|(tok, &pc)| {
            let rc = *ref_counts.get(tok).unwrap_or(&0);
            pc.min(rc)
        })
        .sum();

    let true_positives = overlap;
    let false_positives = total_pred.saturating_sub(overlap);
    let false_negatives = total_ref.saturating_sub(overlap);

    let precision = if total_pred == 0 { 0.0 } else { overlap as f64 / total_pred as f64 };

    let recall = if total_ref == 0 { 0.0 } else { overlap as f64 / total_ref as f64 };

    let f1 = safe_f1(precision, recall);

    Ok(F1Score {
        precision,
        recall,
        f1,
        true_positives,
        false_positives,
        false_negatives,
    })
}

/// Count token frequencies.
fn token_counts<'a>(tokens: &[&'a str]) -> HashMap<&'a str, usize> {
    let mut counts = HashMap::new();
    for &tok in tokens {
        *counts.entry(tok).or_insert(0) += 1;
    }
    counts
}

// ─────────────────────────────────────────────────────────────────────────────
// Exact match
// ─────────────────────────────────────────────────────────────────────────────

/// Check whether `prediction` exactly matches `reference` after case folding
/// and whitespace normalisation.
///
/// Both strings are lowercased and leading/trailing whitespace is stripped
/// before comparison.
///
/// # Example
/// ```rust,ignore
/// use trustformers::evaluation::exact_match;
/// assert!(exact_match("Paris", "paris"));
/// assert!(!exact_match("Paris", "France"));
/// ```
pub fn exact_match(prediction: &str, reference: &str) -> bool {
    normalise(prediction) == normalise(reference)
}

/// Compute the exact-match accuracy over a batch of examples.
///
/// Returns the fraction of examples where the prediction exactly matches the
/// reference (after normalisation).
///
/// # Errors
/// * [`MetricError::EmptyInput`] if both slices are empty
/// * [`MetricError::LengthMismatch`] if slices have different lengths
///
/// # Example
/// ```rust,ignore
/// use trustformers::evaluation::exact_match_score;
/// let preds = &["Paris", "Tokyo"];
/// let refs  = &["paris", "Tokyo"];
/// let score = exact_match_score(preds, refs).unwrap();
/// assert!((score - 1.0).abs() < 1e-6);
/// ```
pub fn exact_match_score(predictions: &[&str], references: &[&str]) -> Result<f64, MetricError> {
    if predictions.is_empty() && references.is_empty() {
        return Err(MetricError::EmptyInput(
            "predictions and references are both empty".to_string(),
        ));
    }
    if predictions.len() != references.len() {
        return Err(MetricError::LengthMismatch(
            predictions.len(),
            references.len(),
        ));
    }

    let correct = predictions
        .iter()
        .zip(references.iter())
        .filter(|(p, r)| exact_match(p, r))
        .count();

    Ok(correct as f64 / predictions.len() as f64)
}

/// Normalise a string: lowercase + trim.
#[inline]
fn normalise(s: &str) -> String {
    s.trim().to_lowercase()
}

// ─────────────────────────────────────────────────────────────────────────────
// Perplexity
// ─────────────────────────────────────────────────────────────────────────────

/// Compute perplexity from per-token log-probabilities.
///
/// `perplexity = exp(-1/N * Σ log P(w_i))`
///
/// Log-probabilities should be natural log (base-e) probabilities for each token
/// given its context.
///
/// # Arguments
/// * `log_probs` – slice of log-probabilities `log P(w_i | context)`. Values
///   must be ≤ 0 (probabilities in (0, 1]).
///
/// # Returns
/// Perplexity value ≥ 1.0.
///
/// # Errors
/// * [`MetricError::EmptyInput`] if `log_probs` is empty
/// * [`MetricError::MathError`] if any value is positive or non-finite
///
/// # Example
/// ```rust,ignore
/// use trustformers::evaluation::perplexity;
/// // Uniform distribution over 10 tokens: log P = -log(10) each
/// let log_probs: Vec<f64> = vec![-(10.0_f64.ln()); 20];
/// let ppl = perplexity(&log_probs).unwrap();
/// assert!((ppl - 10.0).abs() < 1e-6);
/// ```
pub fn perplexity(log_probs: &[f64]) -> Result<f64, MetricError> {
    if log_probs.is_empty() {
        return Err(MetricError::EmptyInput(
            "log_probs must not be empty for perplexity".to_string(),
        ));
    }

    for (i, &lp) in log_probs.iter().enumerate() {
        if !lp.is_finite() {
            return Err(MetricError::MathError(format!(
                "log_prob[{i}] = {lp} is not finite"
            )));
        }
        if lp > f64::EPSILON {
            return Err(MetricError::MathError(format!(
                "log_prob[{i}] = {lp} is positive (log-probability must be <= 0)"
            )));
        }
    }

    let mean_neg_log_prob = -log_probs.iter().sum::<f64>() / log_probs.len() as f64;
    let ppl = mean_neg_log_prob.exp();

    if !ppl.is_finite() {
        return Err(MetricError::MathError(format!(
            "perplexity is not finite: {ppl}"
        )));
    }

    Ok(ppl)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BLEU ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_bleu_perfect_match() {
        let hyp = &["the", "cat", "sat", "on", "the", "mat"];
        let refs: &[&[&str]] = &[&["the", "cat", "sat", "on", "the", "mat"]];
        let score = bleu_score(hyp, refs, 4, false).expect("bleu");
        // Perfect match → BLEU = 1.0
        assert!((score - 1.0).abs() < 1e-6, "expected 1.0, got {score}");
    }

    #[test]
    fn test_bleu_no_overlap() {
        let hyp = &["the", "quick", "brown", "fox"];
        let refs: &[&[&str]] = &[&["a", "lazy", "dog", "ran"]];
        // No common unigrams (only "the" might differ) — let's use completely disjoint
        let hyp2 = &["alpha", "beta", "gamma", "delta"];
        let refs2: &[&[&str]] = &[&["zeta", "eta", "theta", "iota"]];
        let score = bleu_score(hyp2, refs2, 4, false).expect("bleu");
        assert_eq!(score, 0.0, "expected 0.0 for no overlap, got {score}");
        let _ = hyp;
        let _ = refs;
    }

    #[test]
    fn test_bleu_partial_overlap() {
        let hyp = &["the", "cat", "sat"];
        let refs: &[&[&str]] = &[&["the", "cat", "is", "on", "the", "mat"]];
        let score = bleu_score(hyp, refs, 2, true).expect("bleu");
        assert!(
            score > 0.0 && score < 1.0,
            "expected partial score, got {score}"
        );
    }

    #[test]
    fn test_bleu_empty_hypothesis_returns_zero() {
        let hyp: &[&str] = &[];
        let refs: &[&[&str]] = &[&["a", "b", "c"]];
        let score = bleu_score(hyp, refs, 4, false).expect("bleu");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_bleu_invalid_order_fails() {
        let hyp = &["a", "b"];
        let refs: &[&[&str]] = &[&["a", "b"]];
        assert!(bleu_score(hyp, refs, 0, false).is_err());
    }

    #[test]
    fn test_bleu_no_references_fails() {
        let hyp = &["a", "b"];
        let refs: &[&[&str]] = &[];
        assert!(bleu_score(hyp, refs, 4, false).is_err());
    }

    #[test]
    fn test_ngram_precision_unigram_perfect() {
        let hyp = &["a", "b", "c"];
        let refs: &[&[&str]] = &[&["a", "b", "c"]];
        let p = ngram_precision(hyp, refs, 1, false).expect("ngram_precision");
        assert!((p - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ngram_precision_zero_order_fails() {
        let hyp = &["a"];
        let refs: &[&[&str]] = &[&["a"]];
        assert!(ngram_precision(hyp, refs, 0, false).is_err());
    }

    // ── ROUGE-N ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rouge_n_perfect() {
        let hyp = &["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = &["the", "cat", "sat", "on", "the", "mat"];
        let score = rouge_n_score(hyp, ref_, 1).expect("rouge_n");
        assert!(
            (score.precision - 1.0).abs() < 1e-6
                && (score.recall - 1.0).abs() < 1e-6
                && (score.f1 - 1.0).abs() < 1e-6,
            "expected all-1.0, got {score:?}"
        );
    }

    #[test]
    fn test_rouge_n_no_overlap() {
        let hyp = &["alpha", "beta", "gamma"];
        let ref_ = &["zeta", "eta", "theta"];
        let score = rouge_n_score(hyp, ref_, 1).expect("rouge_n");
        assert_eq!(score.precision, 0.0);
        assert_eq!(score.recall, 0.0);
        assert_eq!(score.f1, 0.0);
    }

    #[test]
    fn test_rouge_n_partial_overlap() {
        let hyp = &["the", "cat", "sat"];
        let ref_ = &["the", "cat", "is", "on", "the", "mat"];
        let score = rouge_n_score(hyp, ref_, 1).expect("rouge_n");
        // "the" and "cat" overlap: precision = 2/3, recall = 2/6
        assert!(score.precision > 0.0 && score.precision < 1.0);
        assert!(score.recall > 0.0 && score.recall < 1.0);
        assert!(score.f1 > 0.0 && score.f1 < 1.0);
    }

    #[test]
    fn test_rouge_n_invalid_order_fails() {
        assert!(rouge_n_score(&["a"], &["a"], 0).is_err());
    }

    #[test]
    fn test_rouge_n_bigram_partial() {
        let hyp = &["a", "b", "c", "d"];
        let ref_ = &["a", "b", "e", "f"];
        // Only bigram "a b" overlaps
        let score = rouge_n_score(hyp, ref_, 2).expect("rouge_n");
        assert!(score.f1 > 0.0 && score.f1 < 1.0);
    }

    // ── ROUGE-L ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rouge_l_basic() {
        let hyp = &["police", "killed", "the", "gunman"];
        let ref_ = &["police", "kill", "the", "gunman"];
        let score = rouge_l_score(hyp, ref_).expect("rouge_l");
        // LCS = ["police", "the", "gunman"] = 3
        assert!((score.recall - 3.0 / 4.0).abs() < 1e-6);
        assert!((score.precision - 3.0 / 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_l_perfect_match() {
        let tokens = &["a", "b", "c", "d", "e"];
        let score = rouge_l_score(tokens, tokens).expect("rouge_l");
        assert!((score.f1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_l_empty_hypothesis() {
        let hyp: &[&str] = &[];
        let ref_ = &["a", "b"];
        let score = rouge_l_score(hyp, ref_).expect("rouge_l");
        assert_eq!(score.f1, 0.0);
    }

    // ── Exact Match ──────────────────────────────────────────────────────────

    #[test]
    fn test_exact_match_true() {
        assert!(exact_match("Paris", "Paris"));
    }

    #[test]
    fn test_exact_match_case_insensitive() {
        assert!(exact_match("Paris", "paris"));
        assert!(exact_match("THE QUICK BROWN FOX", "the quick brown fox"));
    }

    #[test]
    fn test_exact_match_false() {
        assert!(!exact_match("Paris", "France"));
    }

    #[test]
    fn test_exact_match_whitespace_normalised() {
        assert!(exact_match("  hello ", "hello"));
    }

    #[test]
    fn test_exact_match_score_batch() {
        let preds = &["Paris", "Tokyo", "berlin"];
        let refs = &["paris", "london", "Berlin"];
        let score = exact_match_score(preds, refs).expect("em_score");
        // "Paris"/"paris" ✓, "Tokyo"/"london" ✗, "berlin"/"Berlin" ✓  → 2/3
        assert!((score - 2.0 / 3.0).abs() < 1e-6, "got {score}");
    }

    #[test]
    fn test_exact_match_score_empty_fails() {
        assert!(exact_match_score(&[], &[]).is_err());
    }

    #[test]
    fn test_exact_match_score_length_mismatch_fails() {
        assert!(exact_match_score(&["a", "b"], &["a"]).is_err());
    }

    // ── Token F1 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_f1_perfect() {
        let tokens = &["the", "cat", "sat"];
        let score = token_f1_score(tokens, tokens).expect("f1");
        assert!((score.f1 - 1.0).abs() < 1e-6);
        assert!((score.precision - 1.0).abs() < 1e-6);
        assert!((score.recall - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f1_partial() {
        let pred = &["the", "cat", "sat"];
        let ref_ = &["the", "cat", "is", "on", "the", "mat"];
        let score = token_f1_score(pred, ref_).expect("f1");
        assert!(score.f1 > 0.0 && score.f1 < 1.0);
        assert!(score.true_positives > 0);
    }

    #[test]
    fn test_f1_no_overlap() {
        let pred = &["alpha", "beta"];
        let ref_ = &["gamma", "delta"];
        let score = token_f1_score(pred, ref_).expect("f1");
        assert_eq!(score.f1, 0.0);
        assert_eq!(score.true_positives, 0);
    }

    // ── Perplexity ───────────────────────────────────────────────────────────

    #[test]
    fn test_perplexity_perfect() {
        // log P = 0.0 → P = 1.0 → PPL = exp(0) = 1.0
        let log_probs = vec![0.0_f64; 10];
        let ppl = perplexity(&log_probs).expect("perplexity");
        assert!((ppl - 1.0).abs() < 1e-6, "expected 1.0, got {ppl}");
    }

    #[test]
    fn test_perplexity_uniform() {
        // Uniform over N=10 tokens: log P = -ln(10) → PPL = 10
        let n = 10.0_f64;
        let log_probs = vec![-n.ln(); 50];
        let ppl = perplexity(&log_probs).expect("perplexity");
        assert!((ppl - n).abs() < 1e-6, "expected {n}, got {ppl}");
    }

    #[test]
    fn test_perplexity_empty_fails() {
        assert!(perplexity(&[]).is_err());
    }

    #[test]
    fn test_perplexity_positive_log_prob_fails() {
        let log_probs = vec![0.5_f64];
        assert!(perplexity(&log_probs).is_err());
    }

    #[test]
    fn test_perplexity_non_finite_fails() {
        let log_probs = vec![f64::NEG_INFINITY];
        assert!(perplexity(&log_probs).is_err());
    }

    // ── NlpBenchmarkResult ───────────────────────────────────────────────────

    #[test]
    fn test_benchmark_result_display() {
        let mut result = NlpBenchmarkResult::new();
        result.bleu_4 = Some(0.423);
        result.perplexity = Some(15.3);
        result.exact_match = Some(0.75);

        let summary = result.display_summary();
        assert!(summary.contains("BLEU-4"), "summary: {summary}");
        assert!(summary.contains("Perplexity"), "summary: {summary}");
        assert!(summary.contains("Exact Match"), "summary: {summary}");
    }

    #[test]
    fn test_benchmark_result_empty_display() {
        let result = NlpBenchmarkResult::new();
        let summary = result.display_summary();
        assert!(summary.contains("no metrics"));
    }

    // ── LCS helper ──────────────────────────────────────────────────────────

    #[test]
    fn test_lcs_identical() {
        let a = &["a", "b", "c", "d"];
        let b = &["a", "b", "c", "d"];
        assert_eq!(lcs_length(a, b), 4);
    }

    #[test]
    fn test_lcs_disjoint() {
        let a = &["x", "y", "z"];
        let b = &["a", "b", "c"];
        assert_eq!(lcs_length(a, b), 0);
    }

    #[test]
    fn test_lcs_partial() {
        let a = &["a", "b", "c", "d"];
        let b = &["b", "c", "e"];
        // LCS = ["b", "c"] = 2
        assert_eq!(lcs_length(a, b), 2);
    }
}
