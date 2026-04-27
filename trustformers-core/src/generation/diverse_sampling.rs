//! Diverse sampling strategies for text generation.
//!
//! Implements Greedy, Top-K, Top-P (nucleus), Min-P, η-sampling,
//! Typical sampling, and Mirostat.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by sampling functions.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingError {
    /// The logits slice was empty.
    EmptyLogits,
    /// Top-k parameter k is larger than the vocabulary.
    InvalidK { k: usize, vocab: usize },
    /// An invalid probability parameter was supplied.
    InvalidP(String),
    /// Temperature was zero or negative.
    InvalidTemperature,
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplingError::EmptyLogits => write!(f, "logits slice is empty"),
            SamplingError::InvalidK { k, vocab } => {
                write!(f, "k={k} exceeds vocabulary size {vocab}")
            }
            SamplingError::InvalidP(msg) => write!(f, "invalid probability parameter: {msg}"),
            SamplingError::InvalidTemperature => {
                write!(f, "temperature must be positive")
            }
        }
    }
}

impl std::error::Error for SamplingError {}

// ---------------------------------------------------------------------------
// Sampling method enum
// ---------------------------------------------------------------------------

/// Describes which sampling algorithm to apply.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingMethod {
    /// Argmax — always picks the highest-probability token.
    Greedy,
    /// Top-K sampling — restrict to the K most probable tokens.
    TopK { k: usize },
    /// Nucleus (Top-P) sampling — restrict to the smallest set whose
    /// cumulative probability exceeds `p`.
    TopP { p: f32 },
    /// Minimum-probability sampling (Yu et al., 2023).
    MinP { min_p: f32 },
    /// η-sampling — entropy-based dynamic top-k.
    Eta { eta: f32 },
    /// Typical sampling — keep tokens close to the distribution entropy.
    Typical { tau: f32 },
    /// Mirostat — perplexity-controlled sampling.
    Mirostat { tau: f32, learning_rate: f32 },
}

// ---------------------------------------------------------------------------
// Mirostat state
// ---------------------------------------------------------------------------

/// Persistent state for the Mirostat algorithm.
#[derive(Debug, Clone)]
pub struct MirostatState {
    /// Target surprisal (desired perplexity).
    pub tau: f32,
    /// Learning rate for updating the running estimate `mu`.
    pub learning_rate: f32,
    /// Running estimate of the perplexity budget; initialised to `2 * tau`.
    pub mu: f32,
}

impl MirostatState {
    /// Create a new Mirostat state with the given target and learning rate.
    pub fn new(tau: f32, learning_rate: f32) -> Self {
        Self {
            tau,
            learning_rate,
            mu: 2.0 * tau,
        }
    }

    /// Update `mu` based on the log-probability of the sampled token.
    ///
    /// `token_prob` is the *probability* (not log-probability) of the token
    /// that was just sampled.
    pub fn update(&mut self, token_prob: f32) {
        let safe_prob = token_prob.max(f32::MIN_POSITIVE);
        let observed_surprise = -safe_prob.ln();
        self.mu -= self.learning_rate * (observed_surprise - self.tau);
    }
}

// ---------------------------------------------------------------------------
// Sampling configuration
// ---------------------------------------------------------------------------

/// Full configuration for the unified `sample` entry point.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Which sampling method to use.
    pub method: SamplingMethod,
    /// Temperature applied to logits before sampling (1.0 = no change).
    pub temperature: f32,
    /// Repetition penalty: logits for already-generated tokens are divided
    /// by this value (1.0 = no penalty).
    pub repetition_penalty: f32,
    /// Optional pre-filter: restrict to top-K tokens before the main method.
    pub top_k_before_method: Option<usize>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            method: SamplingMethod::Greedy,
            temperature: 1.0,
            repetition_penalty: 1.0,
            top_k_before_method: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Temperature and repetition-penalty helpers
// ---------------------------------------------------------------------------

/// Divide all logits by `temperature` in-place.
///
/// # Panics
/// Does not panic; callers are expected to validate temperature > 0 beforehand.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    for x in logits.iter_mut() {
        *x /= temperature;
    }
}

/// Divide the logit of each token in `generated` by `penalty` in-place.
pub fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    for &tok in generated {
        let idx = tok as usize;
        if idx < logits.len() {
            logits[idx] /= penalty;
        }
    }
}

// ---------------------------------------------------------------------------
// Softmax helper (local, not exported)
// ---------------------------------------------------------------------------

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Shannon entropy of a probability distribution
// ---------------------------------------------------------------------------

fn entropy(probs: &[f32]) -> f32 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

// ---------------------------------------------------------------------------
// Greedy sampling
// ---------------------------------------------------------------------------

/// Return the index of the maximum logit (argmax).
pub fn greedy_sample(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Top-K sampling
// ---------------------------------------------------------------------------

/// Keep only the top-K logits (set the rest to −∞) and return argmax.
pub fn top_k_sample(logits: &[f32], k: usize) -> Result<u32, SamplingError> {
    let vocab = logits.len();
    if vocab == 0 {
        return Err(SamplingError::EmptyLogits);
    }
    if k == 0 || k > vocab {
        return Err(SamplingError::InvalidK { k, vocab });
    }

    // Find the k-th largest value via partial sort.
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];

    // Build filtered logits
    let filtered: Vec<f32> = logits
        .iter()
        .map(|&x| if x >= threshold { x } else { f32::NEG_INFINITY })
        .collect();

    Ok(greedy_sample(&filtered))
}

// ---------------------------------------------------------------------------
// Top-P (nucleus) sampling
// ---------------------------------------------------------------------------

/// Nucleus sampling: keep the smallest set of tokens whose cumulative
/// probability ≥ p, then return argmax of that set.
pub fn top_p_sample(logits: &[f32], p: f32) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if p <= 0.0 || p > 1.0 {
        return Err(SamplingError::InvalidP(format!(
            "p must be in (0, 1], got {p}"
        )));
    }

    let probs = softmax(logits);

    // Sort by descending probability, track original indices.
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Accumulate until we reach p.
    let mut cumulative = 0.0_f32;
    let mut kept_mask = vec![false; logits.len()];
    for (idx, prob) in &indexed {
        kept_mask[*idx] = true;
        cumulative += prob;
        if cumulative >= p {
            break;
        }
    }

    let filtered: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| if kept_mask[i] { x } else { f32::NEG_INFINITY })
        .collect();

    Ok(greedy_sample(&filtered))
}

// ---------------------------------------------------------------------------
// Min-P sampling
// ---------------------------------------------------------------------------

/// Minimum-probability sampling (Yu et al., 2023).
///
/// Compute `threshold = min_p * max_prob`.  Keep all tokens above the
/// threshold, then return argmax.
pub fn min_p_sample(logits: &[f32], min_p: f32) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if min_p < 0.0 || min_p >= 1.0 {
        return Err(SamplingError::InvalidP(format!(
            "min_p must be in [0, 1), got {min_p}"
        )));
    }

    let probs = softmax(logits);
    let max_prob = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let threshold = min_p * max_prob;

    let filtered: Vec<f32> = logits
        .iter()
        .zip(probs.iter())
        .map(|(&x, &prob)| if prob >= threshold { x } else { f32::NEG_INFINITY })
        .collect();

    Ok(greedy_sample(&filtered))
}

// ---------------------------------------------------------------------------
// Eta (η) sampling
// ---------------------------------------------------------------------------

/// η-sampling: entropy-based dynamic top-k.
///
/// Computes Shannon entropy H of the distribution, then applies top-k with
/// `k = max(1, ceil(e^H * eta))`.
pub fn eta_sample(logits: &[f32], eta: f32) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if eta <= 0.0 {
        return Err(SamplingError::InvalidP(format!(
            "eta must be positive, got {eta}"
        )));
    }

    let probs = softmax(logits);
    let h = entropy(&probs);

    let dynamic_k = ((h.exp() * eta).ceil() as usize).max(1);
    let k = dynamic_k.min(logits.len());

    top_k_sample(logits, k)
}

// ---------------------------------------------------------------------------
// Typical sampling
// ---------------------------------------------------------------------------

/// Typical sampling (Meister et al., 2023).
///
/// Sorts tokens by |H − (−log p(x))|, keeps the tokens in that order until
/// cumulative probability ≥ tau, then returns argmax of those tokens.
pub fn typical_sample(logits: &[f32], tau: f32) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if tau <= 0.0 || tau > 1.0 {
        return Err(SamplingError::InvalidP(format!(
            "tau must be in (0, 1], got {tau}"
        )));
    }

    let probs = softmax(logits);
    let h = entropy(&probs);

    // Compute |H - (-log p(x))| for each token.
    let mut typicality: Vec<(usize, f32, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let neg_log_p = if p > 0.0 { -p.ln() } else { f32::MAX };
            let typ_val = (h - neg_log_p).abs();
            (i, typ_val, p)
        })
        .collect();

    // Sort ascending by typicality (most typical first).
    typicality.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0_f32;
    let mut kept_mask = vec![false; logits.len()];
    for (idx, _typ, prob) in &typicality {
        kept_mask[*idx] = true;
        cumulative += prob;
        if cumulative >= tau {
            break;
        }
    }

    let filtered: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| if kept_mask[i] { x } else { f32::NEG_INFINITY })
        .collect();

    Ok(greedy_sample(&filtered))
}

// ---------------------------------------------------------------------------
// Mirostat sampling
// ---------------------------------------------------------------------------

/// Mirostat sampling: perplexity-controlled generation.
///
/// Reads `mu` from `state.mu`.
/// On each call:
/// 1. Compute `k = max(1, round(exp(-ln(2*mu))))`  — dynamic truncation.
/// 2. Apply top-k to select from the truncated distribution.
/// 3. Return argmax of the truncated logits.
///
/// The caller is responsible for calling `state.update(token_prob)` afterwards
/// if they want to evolve `mu` (the `sample` entry-point does this).
pub fn mirostat_sample(
    logits: &[f32],
    state: &MirostatState,
) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }

    let mu = state.mu;
    // k = max(1, round(exp( -ln(2*mu) )))
    // Equivalently: k = max(1, round( 1 / (2*mu) ) )
    let two_mu = (2.0 * mu).max(f32::MIN_POSITIVE);
    let k_float = (1.0 / two_mu).round();
    let k = (k_float as usize).max(1).min(logits.len());

    top_k_sample(logits, k)
}

// ---------------------------------------------------------------------------
// Unified entry point
// ---------------------------------------------------------------------------

/// Unified sampling function.
///
/// Applies (in order):
/// 1. Repetition penalty
/// 2. Temperature scaling
/// 3. Optional top-K pre-filter
/// 4. The configured sampling method
///
/// For `SamplingMethod::Mirostat`, `state` must be `Some(...)`.
pub fn sample(
    logits: &[f32],
    config: &SamplingConfig,
    state: Option<&mut MirostatState>,
    generated: &[u32],
) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if config.temperature <= 0.0 {
        return Err(SamplingError::InvalidTemperature);
    }

    let mut working = logits.to_vec();

    // Step 1: repetition penalty
    if (config.repetition_penalty - 1.0).abs() > f32::EPSILON {
        apply_repetition_penalty(&mut working, generated, config.repetition_penalty);
    }

    // Step 2: temperature
    if (config.temperature - 1.0).abs() > f32::EPSILON {
        apply_temperature(&mut working, config.temperature);
    }

    // Step 3: optional pre-filter top-k
    if let Some(pre_k) = config.top_k_before_method {
        let vocab = working.len();
        if pre_k == 0 || pre_k > vocab {
            return Err(SamplingError::InvalidK { k: pre_k, vocab });
        }
        let mut sorted: Vec<f32> = working.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[pre_k - 1];
        for x in working.iter_mut() {
            if *x < threshold {
                *x = f32::NEG_INFINITY;
            }
        }
    }

    // Step 4: dispatch to method
    match &config.method {
        SamplingMethod::Greedy => Ok(greedy_sample(&working)),
        SamplingMethod::TopK { k } => top_k_sample(&working, *k),
        SamplingMethod::TopP { p } => top_p_sample(&working, *p),
        SamplingMethod::MinP { min_p } => min_p_sample(&working, *min_p),
        SamplingMethod::Eta { eta } => eta_sample(&working, *eta),
        SamplingMethod::Typical { tau } => typical_sample(&working, *tau),
        SamplingMethod::Mirostat { tau, learning_rate } => {
            let mut local_state;
            let s = match state {
                Some(s) => s,
                None => {
                    local_state = MirostatState::new(*tau, *learning_rate);
                    &mut local_state
                }
            };
            let token = mirostat_sample(&working, s)?;
            // Update mu with the probability of the chosen token.
            let probs = softmax(&working);
            let token_prob = probs.get(token as usize).copied().unwrap_or(0.0);
            s.update(token_prob);
            Ok(token)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // 1. Greedy selects the maximum logit
    // ------------------------------------------------------------------
    #[test]
    fn test_greedy_selects_max() {
        let logits = vec![-1.0_f32, 0.5, 2.0, 1.0];
        assert_eq!(greedy_sample(&logits), 2);
    }

    #[test]
    fn test_greedy_single_element() {
        let logits = vec![42.0_f32];
        assert_eq!(greedy_sample(&logits), 0);
    }

    // ------------------------------------------------------------------
    // 2. Temperature scaling
    // ------------------------------------------------------------------
    #[test]
    fn test_temperature_scaling_divides_logits() {
        let mut logits = vec![1.0_f32, 2.0, 3.0];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[0] - 0.5).abs() < 1e-6);
        assert!((logits[1] - 1.0).abs() < 1e-6);
        assert!((logits[2] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_one_is_identity() {
        let original = vec![1.0_f32, 2.0, 3.0];
        let mut logits = original.clone();
        apply_temperature(&mut logits, 1.0);
        for (a, b) in logits.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ------------------------------------------------------------------
    // 3. Top-K filters to k tokens
    // ------------------------------------------------------------------
    #[test]
    fn test_top_k_restricts_to_k() {
        // k=2 from [0.5, 0.1, 0.9, 0.3] → tokens 0,2 kept, greedy picks token 2.
        let logits = vec![0.5_f32, 0.1, 0.9, 0.3];
        let tok = top_k_sample(&logits, 2).expect("ok");
        assert_eq!(tok, 2, "argmax of top-2 is token 2");
    }

    #[test]
    fn test_top_k_k_equals_vocab() {
        let logits = vec![0.1_f32, 0.2, 0.3];
        let tok = top_k_sample(&logits, 3).expect("ok");
        assert_eq!(tok, 2);
    }

    #[test]
    fn test_top_k_k_exceeds_vocab_error() {
        let logits = vec![0.1_f32, 0.2];
        let err = top_k_sample(&logits, 5).expect_err("should error");
        assert!(matches!(err, SamplingError::InvalidK { .. }));
    }

    // ------------------------------------------------------------------
    // 4. Top-P nucleus sampling
    // ------------------------------------------------------------------
    #[test]
    fn test_top_p_returns_argmax_of_nucleus() {
        // Logits skewed so token 3 dominates.
        let logits = vec![-10.0_f32, -10.0, -10.0, 10.0];
        let tok = top_p_sample(&logits, 0.9).expect("ok");
        assert_eq!(tok, 3);
    }

    #[test]
    fn test_top_p_invalid_p_error() {
        let logits = vec![1.0_f32, 2.0];
        assert!(top_p_sample(&logits, 0.0).is_err());
        assert!(top_p_sample(&logits, 1.5).is_err());
    }

    // ------------------------------------------------------------------
    // 5. Min-P threshold calculation
    // ------------------------------------------------------------------
    #[test]
    fn test_min_p_filters_low_prob_tokens() {
        // Logits: token 3 vastly dominant.  min_p = 0.5 means threshold = 0.5 * max_prob.
        // Tokens with prob < 0.5 * max_prob are eliminated.
        let logits = vec![-10.0_f32, -10.0, -10.0, 10.0];
        let tok = min_p_sample(&logits, 0.5).expect("ok");
        assert_eq!(tok, 3);
    }

    #[test]
    fn test_min_p_zero_keeps_all() {
        // min_p = 0 → threshold = 0, all tokens kept.
        let logits = vec![0.0_f32, 0.0, 1.0];
        let tok = min_p_sample(&logits, 0.0).expect("ok");
        assert_eq!(tok, 2);
    }

    // ------------------------------------------------------------------
    // 6. Eta: dynamic k from entropy
    // ------------------------------------------------------------------
    #[test]
    fn test_eta_low_entropy_gives_small_k() {
        // A very peaked distribution → low entropy → small k.
        // Logits: token 0 = 100, rest = -100.
        let logits: Vec<f32> = std::iter::once(100.0_f32)
            .chain(std::iter::repeat(-100.0_f32).take(99))
            .collect();
        let tok = eta_sample(&logits, 1.0).expect("ok");
        // With such low entropy, k should be 1 and argmax is token 0.
        assert_eq!(tok, 0);
    }

    #[test]
    fn test_eta_uniform_distribution_gives_larger_k() {
        // Uniform distribution → maximum entropy → k ≥ 1.
        let logits = vec![0.0_f32; 10];
        let tok = eta_sample(&logits, 1.0).expect("ok");
        // Should succeed; result is within vocab range.
        assert!((tok as usize) < 10);
    }

    // ------------------------------------------------------------------
    // 7. Typical sampling keeps typical tokens
    // ------------------------------------------------------------------
    #[test]
    fn test_typical_sample_peaked_distribution() {
        // Peaked distribution: token 4 dominates.
        let logits: Vec<f32> = (0..5)
            .map(|i| if i == 4 { 10.0_f32 } else { -10.0_f32 })
            .collect();
        let tok = typical_sample(&logits, 0.9).expect("ok");
        assert_eq!(tok, 4);
    }

    #[test]
    fn test_typical_invalid_tau_error() {
        let logits = vec![1.0_f32, 2.0];
        assert!(typical_sample(&logits, 0.0).is_err());
        assert!(typical_sample(&logits, 1.5).is_err());
    }

    // ------------------------------------------------------------------
    // 8. Mirostat state initialisation
    // ------------------------------------------------------------------
    #[test]
    fn test_mirostat_state_init() {
        let state = MirostatState::new(5.0, 0.1);
        assert!((state.tau - 5.0).abs() < f32::EPSILON);
        assert!((state.learning_rate - 0.1).abs() < f32::EPSILON);
        assert!((state.mu - 10.0).abs() < f32::EPSILON, "mu should be 2*tau=10");
    }

    // ------------------------------------------------------------------
    // 9. Mirostat mu update direction
    // ------------------------------------------------------------------
    #[test]
    fn test_mirostat_mu_update_high_surprise() {
        // If observed surprise > tau, mu should decrease.
        let mut state = MirostatState::new(5.0, 0.1);
        let initial_mu = state.mu;
        // Very low probability → high surprise (−ln(very_small) >> tau).
        state.update(0.001);
        assert!(
            state.mu < initial_mu,
            "mu should decrease when surprise > tau"
        );
    }

    #[test]
    fn test_mirostat_mu_update_low_surprise() {
        // If observed surprise < tau, mu should increase.
        let mut state = MirostatState::new(0.1, 0.1);
        let initial_mu = state.mu;
        // High probability → low surprise (−ln(0.99) ≈ 0.01 << tau=0.1...
        // wait tau=0.1 and surprise=0.01, so surprise < tau → mu increases.
        state.update(0.99);
        assert!(
            state.mu > initial_mu,
            "mu should increase when surprise < tau"
        );
    }

    // ------------------------------------------------------------------
    // 10. Repetition penalty reduces already-seen token score
    // ------------------------------------------------------------------
    #[test]
    fn test_repetition_penalty_modifies_generated_tokens() {
        let mut logits = vec![1.0_f32, 1.0, 1.0, 1.0];
        // Token 0 was generated before; it should be penalised.
        apply_repetition_penalty(&mut logits, &[0], 2.0);
        assert!((logits[0] - 0.5).abs() < 1e-6, "logits[0] should be 0.5");
        assert!((logits[1] - 1.0).abs() < 1e-6, "logits[1] unchanged");
    }

    #[test]
    fn test_repetition_penalty_identity_at_one() {
        let original = vec![1.0_f32, 2.0, 3.0];
        let mut logits = original.clone();
        apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
        for (a, b) in logits.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ------------------------------------------------------------------
    // 11. Unified sample function with all methods
    // ------------------------------------------------------------------
    #[test]
    fn test_unified_sample_greedy() {
        let logits = vec![-1.0_f32, 5.0, 2.0];
        let cfg = SamplingConfig {
            method: SamplingMethod::Greedy,
            ..Default::default()
        };
        let tok = sample(&logits, &cfg, None, &[]).expect("ok");
        assert_eq!(tok, 1);
    }

    #[test]
    fn test_unified_sample_top_k() {
        let logits = vec![0.0_f32, 0.0, 1.0, 0.0];
        let cfg = SamplingConfig {
            method: SamplingMethod::TopK { k: 2 },
            ..Default::default()
        };
        let tok = sample(&logits, &cfg, None, &[]).expect("ok");
        assert_eq!(tok, 2);
    }

    #[test]
    fn test_unified_sample_top_p() {
        let logits = vec![-10.0_f32, 10.0];
        let cfg = SamplingConfig {
            method: SamplingMethod::TopP { p: 0.95 },
            ..Default::default()
        };
        let tok = sample(&logits, &cfg, None, &[]).expect("ok");
        assert_eq!(tok, 1);
    }

    #[test]
    fn test_unified_sample_mirostat() {
        let logits = vec![-10.0_f32, -10.0, 10.0];
        let cfg = SamplingConfig {
            method: SamplingMethod::Mirostat { tau: 3.0, learning_rate: 0.1 },
            ..Default::default()
        };
        let tok = sample(&logits, &cfg, None, &[]).expect("ok");
        assert_eq!(tok, 2);
    }

    // ------------------------------------------------------------------
    // 12. Error cases
    // ------------------------------------------------------------------
    #[test]
    fn test_error_empty_logits() {
        let cfg = SamplingConfig::default();
        let err = sample(&[], &cfg, None, &[]).expect_err("should error");
        assert_eq!(err, SamplingError::EmptyLogits);
    }

    #[test]
    fn test_error_invalid_temperature() {
        let logits = vec![1.0_f32, 2.0];
        let cfg = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let err = sample(&logits, &cfg, None, &[]).expect_err("should error");
        assert_eq!(err, SamplingError::InvalidTemperature);
    }

    #[test]
    fn test_error_display() {
        assert!(!SamplingError::EmptyLogits.to_string().is_empty());
        assert!(!SamplingError::InvalidK { k: 5, vocab: 3 }.to_string().is_empty());
        assert!(!SamplingError::InvalidP("x".to_string()).to_string().is_empty());
        assert!(!SamplingError::InvalidTemperature.to_string().is_empty());
    }
}
