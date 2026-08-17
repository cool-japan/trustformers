//! Diverse sampling strategies for text generation.
//!
//! Implements Greedy, Top-K, Top-P (nucleus), Min-P, η-sampling,
//! Typical sampling, and Mirostat v2.
//!
//! Every `*_sample` function here draws a token *stochastically* from the
//! truncated distribution.  Only [`greedy_sample`] is deterministic.  The
//! truncation and multinomial primitives live in
//! [`super::logits_processing`], so these strategies and
//! [`super::core::TextGenerator`] cannot drift apart.

use std::fmt;

use scirs2_core::random::StdRng;

use super::logits_processing as lp;

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
    /// A numeric failure inside the shared logits primitives.
    Numeric(String),
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplingError::EmptyLogits => write!(f, "logits slice is empty"),
            SamplingError::InvalidK { k, vocab } => {
                write!(f, "k={k} exceeds vocabulary size {vocab}")
            },
            SamplingError::InvalidP(msg) => write!(f, "invalid probability parameter: {msg}"),
            SamplingError::InvalidTemperature => {
                write!(f, "temperature must be positive")
            },
            SamplingError::Numeric(msg) => write!(f, "numeric failure while sampling: {msg}"),
        }
    }
}

impl std::error::Error for SamplingError {}

/// Convert a crate error raised by the shared primitives into a
/// [`SamplingError`].
fn numeric(error: crate::errors::TrustformersError) -> SamplingError {
    SamplingError::Numeric(error.to_string())
}

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
    /// η-sampling — entropy-dependent probability floor (Hewitt et al., 2022).
    Eta { eta: f32 },
    /// Typical sampling — keep tokens close to the distribution entropy.
    Typical { tau: f32 },
    /// Mirostat v2 — perplexity-controlled sampling.
    Mirostat { tau: f32, learning_rate: f32 },
}

// ---------------------------------------------------------------------------
// Mirostat state
// ---------------------------------------------------------------------------

/// Persistent state for the Mirostat v2 algorithm.
///
/// All surprisal values are measured in **bits** (`-log2 p`), matching the
/// Mirostat paper, so `tau` is a target perplexity exponent in bits.
#[derive(Debug, Clone)]
pub struct MirostatState {
    /// Target surprisal in bits.
    pub tau: f32,
    /// Learning rate for updating the running estimate `mu`.
    pub learning_rate: f32,
    /// Running surprisal budget; initialised to `2 * tau`.
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

    /// Update `mu` from the probability of the token that was just emitted.
    pub fn update(&mut self, token_prob: f32) {
        let safe_prob = token_prob.max(f32::MIN_POSITIVE);
        let observed_surprise = -safe_prob.log2();
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
    /// Repetition penalty applied to already-generated tokens (1.0 = no
    /// penalty).  See [`lp::apply_repetition_penalty_indexed`] for the exact,
    /// sign-aware formula.
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

/// Divide all logits by `temperature` in place.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) -> Result<(), SamplingError> {
    lp::apply_temperature(logits, temperature).map_err(|_| SamplingError::InvalidTemperature)
}

/// Apply the sign-aware repetition penalty to every token in `generated`.
pub fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    lp::apply_repetition_penalty_indexed(
        logits,
        generated.iter().map(|&token| token as usize),
        penalty,
    );
}

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

fn softmax(logits: &[f32]) -> Result<Vec<f32>, SamplingError> {
    lp::softmax(logits).map_err(numeric)
}

/// Shannon entropy in nats.
fn entropy(probs: &[f32]) -> f32 {
    probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum()
}

/// Draw from `logits` after masking everything outside `keep`.
fn sample_masked(logits: &[f32], keep: &[bool], rng: &mut StdRng) -> Result<u32, SamplingError> {
    let masked: Vec<f32> = logits
        .iter()
        .zip(keep.iter())
        .map(|(&value, &keep_it)| if keep_it { value } else { f32::NEG_INFINITY })
        .collect();
    let probs = softmax(&masked)?;
    let index = lp::multinomial_sample(&probs, rng).map_err(numeric)?;
    Ok(index as u32)
}

// ---------------------------------------------------------------------------
// Greedy sampling
// ---------------------------------------------------------------------------

/// Return the index of the maximum logit (argmax).
///
/// Ties resolve to the lowest index.
pub fn greedy_sample(logits: &[f32]) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    lp::argmax(logits).map(|index| index as u32).map_err(numeric)
}

// ---------------------------------------------------------------------------
// Top-K sampling
// ---------------------------------------------------------------------------

/// Draw a token from the `k` most probable tokens.
pub fn top_k_sample(logits: &[f32], k: usize, rng: &mut StdRng) -> Result<u32, SamplingError> {
    let vocab = logits.len();
    if vocab == 0 {
        return Err(SamplingError::EmptyLogits);
    }
    if k == 0 || k > vocab {
        return Err(SamplingError::InvalidK { k, vocab });
    }

    let mut filtered = logits.to_vec();
    lp::top_k_filter(&mut filtered, k).map_err(numeric)?;
    let probs = softmax(&filtered)?;
    let index = lp::multinomial_sample(&probs, rng).map_err(numeric)?;
    Ok(index as u32)
}

// ---------------------------------------------------------------------------
// Top-P (nucleus) sampling
// ---------------------------------------------------------------------------

/// Nucleus sampling: draw from the smallest set of tokens whose cumulative
/// probability reaches `p`.
pub fn top_p_sample(logits: &[f32], p: f32, rng: &mut StdRng) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if !p.is_finite() || p <= 0.0 || p > 1.0 {
        return Err(SamplingError::InvalidP(format!(
            "p must be in (0, 1], got {p}"
        )));
    }

    let mut filtered = logits.to_vec();
    lp::top_p_filter(&mut filtered, p).map_err(numeric)?;
    let probs = softmax(&filtered)?;
    let index = lp::multinomial_sample(&probs, rng).map_err(numeric)?;
    Ok(index as u32)
}

// ---------------------------------------------------------------------------
// Min-P sampling
// ---------------------------------------------------------------------------

/// Minimum-probability sampling (Yu et al., 2023).
///
/// Computes `threshold = min_p * max_prob`, keeps every token above it, and
/// draws from the renormalised remainder.
pub fn min_p_sample(logits: &[f32], min_p: f32, rng: &mut StdRng) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if !min_p.is_finite() || !(0.0..1.0).contains(&min_p) {
        return Err(SamplingError::InvalidP(format!(
            "min_p must be in [0, 1), got {min_p}"
        )));
    }

    let probs = softmax(logits)?;
    let max_prob = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let threshold = min_p * max_prob;

    let keep: Vec<bool> = probs.iter().map(|&prob| prob >= threshold).collect();
    sample_masked(logits, &keep, rng)
}

// ---------------------------------------------------------------------------
// Eta (η) sampling
// ---------------------------------------------------------------------------

/// η-sampling (Hewitt et al., 2022, "Truncation Sampling as Language Model
/// Desmoothing").
///
/// Builds an entropy-dependent probability floor
/// `epsilon = min(eta, sqrt(eta) * exp(-H))` and keeps every token whose
/// probability reaches it.  The argmax is always retained so the support is
/// never empty.
pub fn eta_sample(logits: &[f32], eta: f32, rng: &mut StdRng) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if !eta.is_finite() || eta <= 0.0 {
        return Err(SamplingError::InvalidP(format!(
            "eta must be positive, got {eta}"
        )));
    }

    let probs = softmax(logits)?;
    let h = entropy(&probs);
    let epsilon = eta.min(eta.sqrt() * (-h).exp());

    let argmax = lp::argmax(&probs).map_err(numeric)?;
    let mut keep: Vec<bool> = probs.iter().map(|&prob| prob >= epsilon).collect();
    keep[argmax] = true;

    sample_masked(logits, &keep, rng)
}

// ---------------------------------------------------------------------------
// Typical sampling
// ---------------------------------------------------------------------------

/// Typical sampling (Meister et al., 2023).
///
/// Sorts tokens by `|H − (−log p(x))|`, keeps them in that order until the
/// cumulative probability reaches `tau`, and draws from the result.
pub fn typical_sample(logits: &[f32], tau: f32, rng: &mut StdRng) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if !tau.is_finite() || tau <= 0.0 || tau > 1.0 {
        return Err(SamplingError::InvalidP(format!(
            "tau must be in (0, 1], got {tau}"
        )));
    }

    let probs = softmax(logits)?;
    let h = entropy(&probs);

    let mut typicality: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(index, &p)| {
            let neg_log_p = if p > 0.0 { -p.ln() } else { f32::MAX };
            (index, (h - neg_log_p).abs())
        })
        .collect();
    typicality.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut cumulative = 0.0_f32;
    let mut keep = vec![false; logits.len()];
    for (index, _) in &typicality {
        keep[*index] = true;
        cumulative += probs[*index];
        if cumulative >= tau {
            break;
        }
    }

    sample_masked(logits, &keep, rng)
}

// ---------------------------------------------------------------------------
// Mirostat sampling
// ---------------------------------------------------------------------------

/// Mirostat v2 sampling: perplexity-controlled generation.
///
/// Tokens whose surprisal `-log2 p` exceeds the running budget `state.mu` are
/// truncated away; the remainder is renormalised and sampled.  The argmax is
/// always kept so the support cannot become empty.
///
/// The caller is responsible for calling [`MirostatState::update`] with the
/// probability of the emitted token afterwards; [`sample`] does this
/// automatically.
pub fn mirostat_sample(
    logits: &[f32],
    state: &MirostatState,
    rng: &mut StdRng,
) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }

    let probs = softmax(logits)?;
    let argmax = lp::argmax(&probs).map_err(numeric)?;

    let mut keep: Vec<bool> = probs
        .iter()
        .map(
            |&prob| {
                if prob <= 0.0 {
                    false
                } else {
                    -prob.log2() <= state.mu
                }
            },
        )
        .collect();
    keep[argmax] = true;

    sample_masked(logits, &keep, rng)
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
pub fn sample(
    logits: &[f32],
    config: &SamplingConfig,
    state: Option<&mut MirostatState>,
    generated: &[u32],
    rng: &mut StdRng,
) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if !config.temperature.is_finite() || config.temperature <= 0.0 {
        return Err(SamplingError::InvalidTemperature);
    }

    let mut working = logits.to_vec();

    apply_repetition_penalty(&mut working, generated, config.repetition_penalty);
    apply_temperature(&mut working, config.temperature)?;

    if let Some(pre_k) = config.top_k_before_method {
        let vocab = working.len();
        if pre_k == 0 || pre_k > vocab {
            return Err(SamplingError::InvalidK { k: pre_k, vocab });
        }
        lp::top_k_filter(&mut working, pre_k).map_err(numeric)?;
    }

    match &config.method {
        SamplingMethod::Greedy => greedy_sample(&working),
        SamplingMethod::TopK { k } => top_k_sample(&working, *k, rng),
        SamplingMethod::TopP { p } => top_p_sample(&working, *p, rng),
        SamplingMethod::MinP { min_p } => min_p_sample(&working, *min_p, rng),
        SamplingMethod::Eta { eta } => eta_sample(&working, *eta, rng),
        SamplingMethod::Typical { tau } => typical_sample(&working, *tau, rng),
        SamplingMethod::Mirostat { tau, learning_rate } => {
            let mut local_state;
            let mirostat = match state {
                Some(state) => state,
                None => {
                    local_state = MirostatState::new(*tau, *learning_rate);
                    &mut local_state
                },
            };
            let token = mirostat_sample(&working, mirostat, rng)?;
            let probs = softmax(&working)?;
            let token_prob = probs.get(token as usize).copied().unwrap_or(0.0);
            mirostat.update(token_prob);
            Ok(token)
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::SeedableRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(0xC0FFEE)
    }

    /// Collect the set of tokens produced by `draws` repetitions.
    fn support(mut draw: impl FnMut(&mut StdRng) -> u32, draws: usize) -> Vec<u32> {
        let mut rng = test_rng();
        let mut seen: Vec<u32> = Vec::new();
        for _ in 0..draws {
            let token = draw(&mut rng);
            if !seen.contains(&token) {
                seen.push(token);
            }
        }
        seen.sort_unstable();
        seen
    }

    // ------------------------------------------------------------------
    // 1. Greedy selects the maximum logit
    // ------------------------------------------------------------------
    #[test]
    fn test_greedy_selects_max() {
        let logits = vec![-1.0_f32, 0.5, 2.0, 1.0];
        assert_eq!(greedy_sample(&logits).expect("greedy"), 2);
    }

    #[test]
    fn test_greedy_single_element() {
        assert_eq!(greedy_sample(&[42.0_f32]).expect("greedy"), 0);
    }

    #[test]
    fn test_greedy_empty_is_error() {
        assert_eq!(greedy_sample(&[]), Err(SamplingError::EmptyLogits));
    }

    // ------------------------------------------------------------------
    // 2. Temperature scaling
    // ------------------------------------------------------------------
    #[test]
    fn test_temperature_scaling_divides_logits() {
        let mut logits = vec![1.0_f32, 2.0, 3.0];
        apply_temperature(&mut logits, 2.0).expect("temperature");
        assert!((logits[0] - 0.5).abs() < 1e-6);
        assert!((logits[1] - 1.0).abs() < 1e-6);
        assert!((logits[2] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_one_is_identity() {
        let original = vec![1.0_f32, 2.0, 3.0];
        let mut logits = original.clone();
        apply_temperature(&mut logits, 1.0).expect("temperature");
        for (a, b) in logits.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_temperature_zero_is_error() {
        let mut logits = vec![1.0_f32];
        assert_eq!(
            apply_temperature(&mut logits, 0.0),
            Err(SamplingError::InvalidTemperature)
        );
    }

    // ------------------------------------------------------------------
    // 3. Top-K really samples inside the top-K set
    // ------------------------------------------------------------------
    #[test]
    fn test_top_k_draws_only_from_top_k() {
        // Regression: this used to return the argmax for every k, making
        // top-k sampling indistinguishable from greedy decoding.
        let logits = vec![0.5_f32, 0.1, 0.9, 0.3];
        let seen = support(|rng| top_k_sample(&logits, 2, rng).expect("top_k"), 400);
        assert_eq!(seen, vec![0, 2], "only the two best tokens may appear");
    }

    #[test]
    fn test_top_k_one_is_deterministic_argmax() {
        let logits = vec![0.5_f32, 0.1, 0.9, 0.3];
        let seen = support(|rng| top_k_sample(&logits, 1, rng).expect("top_k"), 100);
        assert_eq!(seen, vec![2]);
    }

    #[test]
    fn test_top_k_k_equals_vocab_can_reach_every_token() {
        let logits = vec![0.0_f32, 0.0, 0.0];
        let seen = support(|rng| top_k_sample(&logits, 3, rng).expect("top_k"), 400);
        assert_eq!(seen, vec![0, 1, 2]);
    }

    #[test]
    fn test_top_k_k_exceeds_vocab_error() {
        let logits = vec![0.1_f32, 0.2];
        let mut rng = test_rng();
        let err = top_k_sample(&logits, 5, &mut rng).expect_err("should error");
        assert!(matches!(err, SamplingError::InvalidK { .. }));
    }

    // ------------------------------------------------------------------
    // 4. Top-P nucleus sampling
    // ------------------------------------------------------------------
    #[test]
    fn test_top_p_stays_inside_the_nucleus() {
        // Probabilities proportional to [1, 2, 4, 8]; the 0.8 nucleus is
        // {8/15 = .533, 4/15 -> .800}, i.e. tokens 3 and 2 only.
        let logits = vec![0.0_f32, 2.0_f32.ln(), 4.0_f32.ln(), 8.0_f32.ln()];
        let seen = support(|rng| top_p_sample(&logits, 0.8, rng).expect("top_p"), 500);
        assert_eq!(seen, vec![2, 3], "tokens outside the nucleus were sampled");
    }

    #[test]
    fn test_top_p_dominant_token_is_deterministic() {
        let logits = vec![-10.0_f32, -10.0, -10.0, 10.0];
        let seen = support(|rng| top_p_sample(&logits, 0.9, rng).expect("top_p"), 200);
        assert_eq!(seen, vec![3]);
    }

    #[test]
    fn test_top_p_invalid_p_error() {
        let logits = vec![1.0_f32, 2.0];
        let mut rng = test_rng();
        assert!(top_p_sample(&logits, 0.0, &mut rng).is_err());
        assert!(top_p_sample(&logits, 1.5, &mut rng).is_err());
    }

    // ------------------------------------------------------------------
    // 5. Min-P threshold calculation
    // ------------------------------------------------------------------
    #[test]
    fn test_min_p_filters_low_prob_tokens() {
        let logits = vec![-10.0_f32, -10.0, -10.0, 10.0];
        let seen = support(|rng| min_p_sample(&logits, 0.5, rng).expect("min_p"), 200);
        assert_eq!(seen, vec![3]);
    }

    #[test]
    fn test_min_p_zero_keeps_all() {
        let logits = vec![0.0_f32, 0.0, 0.0];
        let seen = support(|rng| min_p_sample(&logits, 0.0, rng).expect("min_p"), 400);
        assert_eq!(seen, vec![0, 1, 2]);
    }

    // ------------------------------------------------------------------
    // 6. Eta: entropy-dependent probability floor
    // ------------------------------------------------------------------
    #[test]
    fn test_eta_low_entropy_keeps_only_the_peak() {
        let logits: Vec<f32> =
            std::iter::once(100.0_f32).chain(std::iter::repeat_n(-100.0_f32, 99)).collect();
        let seen = support(|rng| eta_sample(&logits, 1.0, rng).expect("eta"), 200);
        assert_eq!(seen, vec![0]);
    }

    #[test]
    fn test_eta_uniform_distribution_keeps_everything() {
        // Uniform over 10: H = ln 10, so epsilon = min(0.5, sqrt(0.5) * 0.1)
        // = 0.0707, comfortably below every token's probability of 0.1.
        let logits = vec![0.0_f32; 10];
        let seen = support(|rng| eta_sample(&logits, 0.5, rng).expect("eta"), 800);
        assert_eq!(seen.len(), 10, "uniform distribution must stay uniform");
    }

    #[test]
    fn test_eta_invalid_error() {
        let mut rng = test_rng();
        assert!(eta_sample(&[1.0_f32, 2.0], 0.0, &mut rng).is_err());
        assert!(eta_sample(&[], 1.0, &mut rng).is_err());
    }

    // ------------------------------------------------------------------
    // 7. Typical sampling keeps typical tokens
    // ------------------------------------------------------------------
    #[test]
    fn test_typical_sample_peaked_distribution() {
        let logits: Vec<f32> = (0..5).map(|i| if i == 4 { 10.0_f32 } else { -10.0_f32 }).collect();
        let seen = support(
            |rng| typical_sample(&logits, 0.9, rng).expect("typical"),
            200,
        );
        assert_eq!(seen, vec![4]);
    }

    #[test]
    fn test_typical_invalid_tau_error() {
        let logits = vec![1.0_f32, 2.0];
        let mut rng = test_rng();
        assert!(typical_sample(&logits, 0.0, &mut rng).is_err());
        assert!(typical_sample(&logits, 1.5, &mut rng).is_err());
    }

    // ------------------------------------------------------------------
    // 8. Mirostat state initialisation
    // ------------------------------------------------------------------
    #[test]
    fn test_mirostat_state_init() {
        let state = MirostatState::new(5.0, 0.1);
        assert!((state.tau - 5.0).abs() < f32::EPSILON);
        assert!((state.learning_rate - 0.1).abs() < f32::EPSILON);
        assert!(
            (state.mu - 10.0).abs() < f32::EPSILON,
            "mu should be 2*tau=10"
        );
    }

    // ------------------------------------------------------------------
    // 9. Mirostat mu update direction
    // ------------------------------------------------------------------
    #[test]
    fn test_mirostat_mu_update_high_surprise() {
        let mut state = MirostatState::new(5.0, 0.1);
        let initial_mu = state.mu;
        state.update(0.001);
        assert!(
            state.mu < initial_mu,
            "mu should decrease when surprise > tau"
        );
    }

    #[test]
    fn test_mirostat_mu_update_low_surprise() {
        let mut state = MirostatState::new(0.1, 0.1);
        let initial_mu = state.mu;
        state.update(0.99);
        assert!(
            state.mu > initial_mu,
            "mu should increase when surprise < tau"
        );
    }

    #[test]
    fn test_mirostat_truncates_high_surprisal_tokens() {
        // mu = 1 bit: only tokens with p >= 0.5 survive (plus the argmax).
        let logits = vec![0.0_f32, 0.0, 0.0, 10.0];
        let state = MirostatState {
            tau: 0.5,
            learning_rate: 0.1,
            mu: 1.0,
        };
        let seen = support(
            |rng| mirostat_sample(&logits, &state, rng).expect("mirostat"),
            200,
        );
        assert_eq!(seen, vec![3]);
    }

    // ------------------------------------------------------------------
    // 10. Repetition penalty reduces already-seen token score
    // ------------------------------------------------------------------
    #[test]
    fn test_repetition_penalty_modifies_generated_tokens() {
        let mut logits = vec![1.0_f32, 1.0, 1.0, 1.0];
        apply_repetition_penalty(&mut logits, &[0], 2.0);
        assert!((logits[0] - 0.5).abs() < 1e-6, "logits[0] should be 0.5");
        assert!((logits[1] - 1.0).abs() < 1e-6, "logits[1] unchanged");
    }

    #[test]
    fn test_repetition_penalty_is_sign_aware() {
        let mut logits = vec![-1.0_f32, -1.0];
        apply_repetition_penalty(&mut logits, &[0], 2.0);
        assert!(
            (logits[0] - (-2.0)).abs() < 1e-6,
            "a negative logit must get MORE negative, got {}",
            logits[0]
        );
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
        let mut rng = test_rng();
        assert_eq!(sample(&logits, &cfg, None, &[], &mut rng).expect("ok"), 1);
    }

    #[test]
    fn test_unified_sample_top_k() {
        let logits = vec![-20.0_f32, -20.0, 20.0, -20.0];
        let cfg = SamplingConfig {
            method: SamplingMethod::TopK { k: 2 },
            ..Default::default()
        };
        let mut rng = test_rng();
        assert_eq!(sample(&logits, &cfg, None, &[], &mut rng).expect("ok"), 2);
    }

    #[test]
    fn test_unified_sample_top_p() {
        let logits = vec![-10.0_f32, 10.0];
        let cfg = SamplingConfig {
            method: SamplingMethod::TopP { p: 0.95 },
            ..Default::default()
        };
        let mut rng = test_rng();
        assert_eq!(sample(&logits, &cfg, None, &[], &mut rng).expect("ok"), 1);
    }

    #[test]
    fn test_unified_sample_mirostat_updates_state() {
        let logits = vec![-10.0_f32, -10.0, 10.0];
        let cfg = SamplingConfig {
            method: SamplingMethod::Mirostat {
                tau: 3.0,
                learning_rate: 0.1,
            },
            ..Default::default()
        };
        let mut state = MirostatState::new(3.0, 0.1);
        let before = state.mu;
        let mut rng = test_rng();
        let token = sample(&logits, &cfg, Some(&mut state), &[], &mut rng).expect("ok");
        assert_eq!(token, 2);
        assert!(
            (state.mu - before).abs() > f32::EPSILON,
            "mu must react to the emitted token"
        );
    }

    #[test]
    fn test_unified_sample_is_seed_reproducible() {
        let logits = vec![0.0_f32, 0.0, 0.0, 0.0];
        let cfg = SamplingConfig {
            method: SamplingMethod::TopK { k: 4 },
            ..Default::default()
        };
        let mut rng_a = StdRng::seed_from_u64(7);
        let mut rng_b = StdRng::seed_from_u64(7);
        let first: Vec<u32> = (0..50)
            .map(|_| sample(&logits, &cfg, None, &[], &mut rng_a).expect("ok"))
            .collect();
        let second: Vec<u32> = (0..50)
            .map(|_| sample(&logits, &cfg, None, &[], &mut rng_b).expect("ok"))
            .collect();
        assert_eq!(first, second);
        assert!(
            first.windows(2).any(|pair| pair[0] != pair[1]),
            "a uniform distribution must not collapse to one token: {first:?}"
        );
    }

    // ------------------------------------------------------------------
    // 12. Error cases
    // ------------------------------------------------------------------
    #[test]
    fn test_error_empty_logits() {
        let cfg = SamplingConfig::default();
        let mut rng = test_rng();
        let err = sample(&[], &cfg, None, &[], &mut rng).expect_err("should error");
        assert_eq!(err, SamplingError::EmptyLogits);
    }

    #[test]
    fn test_error_invalid_temperature() {
        let logits = vec![1.0_f32, 2.0];
        let cfg = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = test_rng();
        let err = sample(&logits, &cfg, None, &[], &mut rng).expect_err("should error");
        assert_eq!(err, SamplingError::InvalidTemperature);
    }

    #[test]
    fn test_error_display() {
        assert!(!SamplingError::EmptyLogits.to_string().is_empty());
        assert!(!SamplingError::InvalidK { k: 5, vocab: 3 }.to_string().is_empty());
        assert!(!SamplingError::InvalidP("x".to_string()).to_string().is_empty());
        assert!(!SamplingError::InvalidTemperature.to_string().is_empty());
        assert!(!SamplingError::Numeric("y".to_string()).to_string().is_empty());
    }
}
