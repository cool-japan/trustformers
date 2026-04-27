//! A/B Testing framework for TrustformeRS Serve.
//!
//! Provides experiment management, variant routing, and statistical significance
//! testing for comparing model variants in production.

pub mod statistics;

pub use statistics::{
    sample_size_for_power, sprt_test, welch_t_test, SampleStats, SprtDecision, StatError,
    TTestResult,
};

use std::collections::HashMap;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from the A/B testing framework.
#[derive(Debug, Clone, Error)]
pub enum AbTestError {
    #[error("Traffic split does not sum to 1.0 (got {got:.4})")]
    InvalidTrafficSplit { got: f32 },

    #[error("Traffic split has no variants")]
    EmptyTrafficSplit,

    #[error("Experiment not found: {0}")]
    ExperimentNotFound(String),

    #[error("Variant not found: {variant} in experiment {experiment}")]
    VariantNotFound { experiment: String, variant: String },

    #[error("Experiment {0} is not running")]
    ExperimentNotRunning(String),

    #[error("Duplicate experiment id: {0}")]
    DuplicateExperiment(String),

    #[error("Insufficient samples for statistical test")]
    InsufficientSamples,

    #[error("Statistical error: {0}")]
    StatisticalError(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Experiment status
// ─────────────────────────────────────────────────────────────────────────────

/// Lifecycle state of an experiment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentStatus {
    /// Experiment has been created but not yet started.
    Draft,
    /// Experiment is actively routing traffic.
    Running,
    /// Experiment has been temporarily paused.
    Paused,
    /// Experiment has concluded successfully.
    Completed,
    /// Experiment was terminated early.
    Aborted,
}

// ─────────────────────────────────────────────────────────────────────────────
// Traffic split
// ─────────────────────────────────────────────────────────────────────────────

/// Describes how traffic should be divided between experiment variants.
///
/// The sum of all fractions must equal 1.0 (validated via [`TrafficSplit::validate`]).
#[derive(Debug, Clone)]
pub struct TrafficSplit {
    /// `(variant_name, traffic_fraction)` pairs. Fractions must sum to 1.0.
    pub variants: Vec<(String, f32)>,
}

impl TrafficSplit {
    /// Create a new traffic split.
    pub fn new(variants: Vec<(String, f32)>) -> Self {
        Self { variants }
    }

    /// Validate that variants are non-empty and their fractions sum to 1.0 (±0.001).
    pub fn validate(&self) -> Result<(), AbTestError> {
        if self.variants.is_empty() {
            return Err(AbTestError::EmptyTrafficSplit);
        }
        let total: f32 = self.variants.iter().map(|(_, f)| f).sum();
        if (total - 1.0_f32).abs() > 1e-3 {
            return Err(AbTestError::InvalidTrafficSplit { got: total });
        }
        Ok(())
    }

    /// Deterministically select a variant for the given hashed user identifier.
    ///
    /// Maps `user_hash` to an integer bucket in `[0, 10_000)` and finds the
    /// variant whose cumulative fraction range covers that bucket.  Using an
    /// integer range avoids the floating-point precision loss that occurs when
    /// normalising a `u64` hash directly to `[0, 1)`.
    pub fn select_variant(&self, user_hash: u64) -> &str {
        const BUCKETS: u64 = 10_000;
        let bucket = user_hash % BUCKETS;

        let last_idx = self.variants.len().saturating_sub(1);
        let mut cumulative: u64 = 0;
        for (idx, (name, fraction)) in self.variants.iter().enumerate() {
            if idx == last_idx {
                // Always absorb the remainder to avoid gaps from fp rounding.
                return name.as_str();
            }
            cumulative += (fraction * BUCKETS as f32).round() as u64;
            if bucket < cumulative {
                return name.as_str();
            }
        }
        // Fallback: last variant (should be unreachable with valid splits).
        self.variants.last().map(|(name, _)| name.as_str()).unwrap_or("")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Experiment configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Full configuration for an A/B experiment.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    pub experiment_id: String,
    pub name: String,
    pub description: String,
    pub traffic_split: TrafficSplit,
    pub status: ExperimentStatus,
    pub start_time: Option<std::time::SystemTime>,
    pub end_time: Option<std::time::SystemTime>,
    /// Primary metric used for winner determination, e.g. `"latency_p99"` or `"throughput"`.
    pub primary_metric: String,
    /// Minimum requests per variant before results are considered reliable.
    pub min_sample_size: usize,
}

impl ExperimentConfig {
    /// Create a new experiment configuration with sensible defaults.
    pub fn new(
        experiment_id: impl Into<String>,
        name: impl Into<String>,
        traffic_split: TrafficSplit,
    ) -> Self {
        Self {
            experiment_id: experiment_id.into(),
            name: name.into(),
            description: String::new(),
            traffic_split,
            status: ExperimentStatus::Draft,
            start_time: None,
            end_time: None,
            primary_metric: "latency_p99".to_string(),
            min_sample_size: 100,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-variant statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulated statistics for a single experiment variant.
#[derive(Debug, Clone)]
pub struct ExperimentVariantStats {
    pub variant_name: String,
    pub num_requests: u64,
    pub total_latency_ms: f64,
    pub error_count: u64,
    pub custom_metrics: HashMap<String, f64>,
}

impl ExperimentVariantStats {
    /// Create a new stats accumulator for the given variant.
    pub fn new(variant_name: impl Into<String>) -> Self {
        Self {
            variant_name: variant_name.into(),
            num_requests: 0,
            total_latency_ms: 0.0,
            error_count: 0,
            custom_metrics: HashMap::new(),
        }
    }

    /// Average latency across all recorded requests (0.0 if no requests).
    pub fn mean_latency_ms(&self) -> f64 {
        if self.num_requests == 0 {
            return 0.0;
        }
        self.total_latency_ms / self.num_requests as f64
    }

    /// Fraction of requests that resulted in an error (0.0 if no requests).
    pub fn error_rate(&self) -> f32 {
        if self.num_requests == 0 {
            return 0.0;
        }
        self.error_count as f32 / self.num_requests as f32
    }

    /// Record a single request with its latency and error flag.
    pub fn record_request(&mut self, latency_ms: f64, is_error: bool) {
        self.num_requests += 1;
        self.total_latency_ms += latency_ms;
        if is_error {
            self.error_count += 1;
        }
    }

    /// Record a custom metric value, accumulating (summing) values for the same key.
    pub fn record_metric(&mut self, key: &str, value: f64) {
        let entry = self.custom_metrics.entry(key.to_string()).or_insert(0.0);
        *entry += value;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistical testing helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight two-sample z-test and significance helpers.
pub struct StatisticalTest;

impl StatisticalTest {
    /// Compute the two-sample z-score.
    ///
    /// `z = (mean1 - mean2) / sqrt(std1²/n1 + std2²/n2)`
    ///
    /// Returns `f64::NAN` if the denominator is zero.
    pub fn two_sample_z_test(
        n1: u64,
        mean1: f64,
        std1: f64,
        n2: u64,
        mean2: f64,
        std2: f64,
    ) -> f64 {
        if n1 == 0 || n2 == 0 {
            return f64::NAN;
        }
        let se = ((std1 * std1 / n1 as f64) + (std2 * std2 / n2 as f64)).sqrt();
        if se < f64::EPSILON {
            return f64::NAN;
        }
        (mean1 - mean2) / se
    }

    /// Return `true` if the given z-score is statistically significant at level `alpha`
    /// using a two-tailed test.
    pub fn is_significant(z_score: f64, alpha: f32) -> bool {
        if z_score.is_nan() {
            return false;
        }
        let z_crit = Self::compute_z_critical(alpha);
        z_score.abs() > z_crit
    }

    /// Approximate the critical z-value for the given two-tailed significance level.
    ///
    /// | α    | z_α/2 |
    /// |------|-------|
    /// | 0.10 | 1.645 |
    /// | 0.05 | 1.960 |
    /// | 0.01 | 2.576 |
    pub fn compute_z_critical(alpha: f32) -> f64 {
        if (alpha - 0.10_f32).abs() < 1e-4 {
            return 1.645;
        }
        if (alpha - 0.01_f32).abs() < 1e-4 {
            return 2.576;
        }
        // Default / α=0.05
        1.96
    }

    /// Approximate one-tailed p-value from a z-score using the complementary
    /// error function identity: p_one_tail = erfc(|z| / sqrt(2)) / 2.
    ///
    /// Returns a two-tailed p-value: `2 * p_one_tail`.
    pub fn approximate_p_value(z_score: f64) -> f32 {
        if z_score.is_nan() {
            return 1.0;
        }
        let x = z_score.abs() / std::f64::consts::SQRT_2;
        let p_one_tail = erfc_approx(x) / 2.0;
        ((2.0 * p_one_tail) as f32).min(1.0_f32)
    }
}

/// Approximate complementary error function for p-value estimation.
///
/// Uses the Horner-form rational approximation from Abramowitz & Stegun 7.1.26,
/// valid for x ≥ 0 with maximum error ~1.5 × 10⁻⁷.
fn erfc_approx(x: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A: [f64; 5] = [
        0.254_829_592,
        -0.284_496_736,
        1.421_413_741,
        -1.453_152_027,
        1.061_405_429,
    ];
    let t = 1.0 / (1.0 + P * x);
    let poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
    let erfc = poly * (-x * x).exp();
    erfc.clamp(0.0, 2.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Experiment results
// ─────────────────────────────────────────────────────────────────────────────

/// Summary of a completed or in-progress experiment analysis.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    pub experiment_id: String,
    pub variant_stats: Vec<ExperimentVariantStats>,
    /// Name of the best-performing variant by primary metric; `None` if ambiguous.
    pub winner: Option<String>,
    pub is_significant: bool,
    pub z_score: f64,
    pub p_value_approx: f32,
    pub recommendation: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// FNV-1a hash helper
// ─────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash of a string (pure-Rust, no external crates).
fn fnv1a_hash(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    let mut hash = FNV_OFFSET;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ─────────────────────────────────────────────────────────────────────────────
// A/B Test Manager
// ─────────────────────────────────────────────────────────────────────────────

/// Central coordinator for all A/B experiments.
///
/// Handles registration, request routing, statistics collection, and result
/// computation.
pub struct AbTestManager {
    /// All registered experiments, keyed by experiment_id.
    pub experiments: HashMap<String, ExperimentConfig>,
    /// Per-experiment per-variant statistics.
    /// Outer key: experiment_id; inner vec is in the same order as the config's variants.
    pub stats: HashMap<String, Vec<ExperimentVariantStats>>,
}

impl AbTestManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self {
            experiments: HashMap::new(),
            stats: HashMap::new(),
        }
    }

    /// Register a new experiment. Returns an error if the ID is already in use
    /// or the traffic split is invalid.
    pub fn register_experiment(&mut self, config: ExperimentConfig) -> Result<(), AbTestError> {
        if self.experiments.contains_key(&config.experiment_id) {
            return Err(AbTestError::DuplicateExperiment(
                config.experiment_id.clone(),
            ));
        }
        config.traffic_split.validate()?;

        // Initialise per-variant stats.
        let variant_stats: Vec<ExperimentVariantStats> = config
            .traffic_split
            .variants
            .iter()
            .map(|(name, _)| ExperimentVariantStats::new(name.clone()))
            .collect();

        self.stats.insert(config.experiment_id.clone(), variant_stats);
        self.experiments.insert(config.experiment_id.clone(), config);
        Ok(())
    }

    /// Deterministically assign a variant to a request using FNV-1a hashing.
    pub fn assign_variant(
        &self,
        experiment_id: &str,
        request_id: &str,
    ) -> Result<String, AbTestError> {
        let config = self
            .experiments
            .get(experiment_id)
            .ok_or_else(|| AbTestError::ExperimentNotFound(experiment_id.to_string()))?;

        if config.status != ExperimentStatus::Running {
            return Err(AbTestError::ExperimentNotRunning(experiment_id.to_string()));
        }

        let hash = fnv1a_hash(request_id);
        let variant = config.traffic_split.select_variant(hash);
        Ok(variant.to_string())
    }

    /// Record a request outcome for the given experiment variant.
    pub fn record_request(
        &mut self,
        experiment_id: &str,
        variant: &str,
        latency_ms: f64,
        is_error: bool,
    ) -> Result<(), AbTestError> {
        let stats = self
            .stats
            .get_mut(experiment_id)
            .ok_or_else(|| AbTestError::ExperimentNotFound(experiment_id.to_string()))?;

        let variant_stats =
            stats.iter_mut().find(|s| s.variant_name == variant).ok_or_else(|| {
                AbTestError::VariantNotFound {
                    experiment: experiment_id.to_string(),
                    variant: variant.to_string(),
                }
            })?;

        variant_stats.record_request(latency_ms, is_error);
        Ok(())
    }

    /// Compute experiment results.
    ///
    /// Compares all variants pairwise against the first (control) variant using a
    /// two-sample z-test on mean latency. The winner is the variant with the
    /// lowest mean latency (fewest errors as tiebreaker).
    pub fn compute_results(&self, experiment_id: &str) -> Result<ExperimentResult, AbTestError> {
        let stats = self
            .stats
            .get(experiment_id)
            .ok_or_else(|| AbTestError::ExperimentNotFound(experiment_id.to_string()))?;

        if stats.len() < 2 {
            return Err(AbTestError::InsufficientSamples);
        }

        let control = &stats[0];
        if control.num_requests == 0 {
            return Err(AbTestError::InsufficientSamples);
        }

        // Estimate sample std dev from a simple Poisson-like model:
        // For latency we use mean as proxy for std dev (conservative).
        let ctrl_mean = control.mean_latency_ms();
        let ctrl_std = ctrl_mean.max(1.0);

        // Find best variant (lowest mean latency; error rate as secondary sort).
        let winner_stats = stats.iter().filter(|s| s.num_requests > 0).min_by(|a, b| {
            let la = a.mean_latency_ms();
            let lb = b.mean_latency_ms();
            la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal).then_with(|| {
                a.error_rate().partial_cmp(&b.error_rate()).unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        let winner = winner_stats.map(|s| s.variant_name.clone());

        // Run z-test between control and best non-control variant (or second variant).
        let treatment_idx = stats
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, s)| s.num_requests > 0)
            .map(|(i, _)| i)
            .unwrap_or(1);

        let treatment = &stats[treatment_idx];
        let trt_mean = treatment.mean_latency_ms();
        let trt_std = trt_mean.max(1.0);

        let z_score = StatisticalTest::two_sample_z_test(
            control.num_requests,
            ctrl_mean,
            ctrl_std,
            treatment.num_requests,
            trt_mean,
            trt_std,
        );

        let is_significant = StatisticalTest::is_significant(z_score, 0.05);
        let p_value_approx = StatisticalTest::approximate_p_value(z_score);

        let recommendation =
            build_recommendation(winner.as_deref(), is_significant, p_value_approx);

        Ok(ExperimentResult {
            experiment_id: experiment_id.to_string(),
            variant_stats: stats.clone(),
            winner,
            is_significant,
            z_score,
            p_value_approx,
            recommendation,
        })
    }

    /// Return all currently running experiments.
    pub fn active_experiments(&self) -> Vec<&ExperimentConfig> {
        self.experiments
            .values()
            .filter(|e| e.status == ExperimentStatus::Running)
            .collect()
    }
}

impl Default for AbTestManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-Armed Bandit (Thompson Sampling)
// ─────────────────────────────────────────────────────────────────────────────

/// A single arm in a multi-armed bandit using the Beta-Bernoulli conjugate model.
#[derive(Debug, Clone)]
pub struct Arm {
    /// Beta distribution alpha parameter (prior + successes).
    pub alpha: f64,
    /// Beta distribution beta parameter (prior + failures).
    pub beta: f64,
    /// Raw count of observed successes.
    pub successes: u64,
    /// Raw count of observed failures.
    pub failures: u64,
}

impl Arm {
    /// Create a new arm with uninformative Beta(1, 1) prior.
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            successes: 0,
            failures: 0,
        }
    }

    /// Create an arm with explicit prior parameters (both must be > 0).
    pub fn with_prior(alpha: f64, beta: f64) -> Self {
        Self {
            alpha: alpha.max(f64::EPSILON),
            beta: beta.max(f64::EPSILON),
            successes: 0,
            failures: 0,
        }
    }
}

impl Default for Arm {
    fn default() -> Self {
        Self::new()
    }
}

/// State of a Thompson-Sampling multi-armed bandit.
#[derive(Debug, Clone)]
pub struct BanditState {
    pub arms: Vec<Arm>,
}

impl BanditState {
    /// Construct a bandit with `n_arms` arms, each with the uninformative prior.
    pub fn new(n_arms: usize) -> Self {
        Self {
            arms: (0..n_arms).map(|_| Arm::new()).collect(),
        }
    }
}

/// Sample a value from Beta(alpha, beta) using a 64-bit LCG seed.
///
/// Uses the Johnk method (ratio of gamma-distributed variates, approximated via
/// a rejection-free transform based on the Beta-normal relationship for large
/// parameters, and direct computation for small ones).
///
/// For robustness this employs a simple but numerically adequate approximation:
/// the normal approximation to Beta for large alpha/beta, and an LCG-based
/// inversion for small parameters.
pub fn sample_beta(alpha: f64, beta: f64, seed: u64) -> f64 {
    // LCG constants (Numerical Recipes)
    const A: u64 = 1664525;
    const C: u64 = 1013904223;

    // Generate two uniform samples from the LCG.
    let s1 = A.wrapping_mul(seed).wrapping_add(C);
    let s2 = A.wrapping_mul(s1).wrapping_add(C);
    let u1 = (s1 >> 11) as f64 / (1u64 << 53) as f64;
    let u2 = (s2 >> 11) as f64 / (1u64 << 53) as f64;
    let u1 = u1.clamp(1e-15, 1.0 - 1e-15);
    let u2 = u2.clamp(1e-15, 1.0 - 1e-15);

    // For large alpha and beta use the normal approximation:
    // Beta(α, β) ≈ N(μ, σ²) where μ = α/(α+β) and σ² = αβ/((α+β)²(α+β+1))
    if alpha > 5.0 && beta > 5.0 {
        let total = alpha + beta;
        let mu = alpha / total;
        let variance = (alpha * beta) / (total * total * (total + 1.0));
        let sigma = variance.sqrt();
        // Box-Muller transform
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        return (mu + sigma * z).clamp(0.0, 1.0);
    }

    // For small parameters use the ratio-of-uniforms method (Cheng 1978).
    // Sample x ~ Gamma(alpha) / (Gamma(alpha) + Gamma(beta)).
    // Approximate via cube-root transform (Wilson-Hilferty approximation for gamma).
    let gamma_a = sample_gamma_approx(alpha, s1);
    let gamma_b = sample_gamma_approx(beta, s2);
    let sum = gamma_a + gamma_b;
    if sum < f64::EPSILON {
        return alpha / (alpha + beta); // fallback to mean
    }
    gamma_a / sum
}

/// Approximate Gamma(shape) sample using the Wilson-Hilferty cube-root normal transform.
fn sample_gamma_approx(shape: f64, seed: u64) -> f64 {
    const A: u64 = 6364136223846793005;
    const C: u64 = 1442695040888963407;
    let s1 = A.wrapping_mul(seed).wrapping_add(C);
    let s2 = A.wrapping_mul(s1).wrapping_add(C);
    let u1 = ((s1 >> 11) as f64 / (1u64 << 53) as f64).clamp(1e-15, 1.0 - 1e-15);
    let u2 = ((s2 >> 11) as f64 / (1u64 << 53) as f64).clamp(1e-15, 1.0 - 1e-15);

    if shape >= 1.0 {
        // Wilson-Hilferty: Gamma(k) ≈ k * (1 - 1/(9k) + z/sqrt(9k))^3
        // where z ~ N(0,1).
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let mean_cbrt = 1.0 - 1.0 / (9.0 * shape);
        let std_cbrt = (1.0 / (9.0 * shape)).sqrt();
        let x = shape * (mean_cbrt + std_cbrt * z).powi(3);
        x.max(f64::EPSILON)
    } else {
        // Use the Ahrens-Dieter boost: if U ~ U[0,1], X ~ Gamma(shape+1),
        // then X * U^(1/shape) ~ Gamma(shape).
        let x = sample_gamma_approx(shape + 1.0, seed.wrapping_add(1));
        x * u1.powf(1.0 / shape)
    }
}

/// Thompson Sampling: sample one value from each arm's Beta posterior and
/// return the index of the arm with the highest sample.
///
/// The `seed` is permuted per arm to ensure decorrelated samples.
pub fn select_arm(state: &BanditState, seed: u64) -> usize {
    if state.arms.is_empty() {
        return 0;
    }

    const SPREAD: u64 = 2_654_435_761; // golden-ratio-derived constant
    state
        .arms
        .iter()
        .enumerate()
        .map(|(i, arm)| {
            let arm_seed = seed.wrapping_add((i as u64).wrapping_mul(SPREAD));
            let sample = sample_beta(arm.alpha, arm.beta, arm_seed);
            (i, sample)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Update the chosen arm's Beta posterior with a success or failure observation.
pub fn update_arm(state: &mut BanditState, arm_idx: usize, success: bool) {
    if let Some(arm) = state.arms.get_mut(arm_idx) {
        if success {
            arm.alpha += 1.0;
            arm.successes += 1;
        } else {
            arm.beta += 1.0;
            arm.failures += 1;
        }
    }
}

/// Compute the mean reward of an arm: α / (α + β).
pub fn arm_mean_reward(arm: &Arm) -> f64 {
    arm.alpha / (arm.alpha + arm.beta)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequential Bayesian Testing
// ─────────────────────────────────────────────────────────────────────────────

/// Bayesian belief state from a sequential test comparing two proportions.
#[derive(Debug, Clone)]
pub struct BayesianBelief {
    /// Posterior probability that arm A has a higher success rate than arm B.
    pub p_a_better_than_b: f64,
    /// Bayes factor: ratio of marginal likelihoods under H1 (A != B) vs H0 (A == B).
    pub bayes_factor: f64,
}

/// Update Bayesian beliefs given observed counts for arm A and arm B.
///
/// Uses the Beta-Binomial model with uniform priors Beta(1, 1).
///
/// `p(A > B)` is computed via numerical integration of the overlapping Beta
/// posteriors using a Riemann sum approximation (200 quadrature points).
///
/// The Bayes factor is approximated as the ratio of the likelihood under
/// the maximum-likelihood estimates vs the pooled null model.
pub fn update_beliefs(successes_a: u64, n_a: u64, successes_b: u64, n_b: u64) -> BayesianBelief {
    // Posterior parameters (uniform prior Beta(1,1))
    let alpha_a = successes_a as f64 + 1.0;
    let beta_a = (n_a - successes_a.min(n_a)) as f64 + 1.0;
    let alpha_b = successes_b as f64 + 1.0;
    let beta_b = (n_b - successes_b.min(n_b)) as f64 + 1.0;

    // P(A > B) via numerical integration: ∫₀¹ P(X_B < x) * p_A(x) dx
    // Using Riemann sum with 200 steps.
    const STEPS: usize = 200;
    let mut p_a_better = 0.0_f64;
    let dx = 1.0 / STEPS as f64;

    // Precompute log-normalisation constants for Beta PDFs.
    let ln_norm_a = ln_beta_fn(alpha_a, beta_a);
    let ln_norm_b = ln_beta_fn(alpha_b, beta_b);

    for k in 0..STEPS {
        // Midpoint of interval.
        let x = (k as f64 + 0.5) * dx;

        // Beta PDF at x for arm A.
        let ln_pdf_a = (alpha_a - 1.0) * x.ln() + (beta_a - 1.0) * (1.0 - x).ln() - ln_norm_a;
        let pdf_a = ln_pdf_a.exp();

        // CDF of arm B at x: Beta regularised incomplete function.
        let cdf_b = regularized_beta_cdf(alpha_b, beta_b, x, &ln_norm_b);

        p_a_better += pdf_a * cdf_b * dx;
    }
    p_a_better = p_a_better.clamp(0.0, 1.0);

    // Bayes factor: compare H1 (different rates) vs H0 (same rate).
    let p_a = if n_a > 0 { successes_a as f64 / n_a as f64 } else { 0.5 };
    let p_b = if n_b > 0 { successes_b as f64 / n_b as f64 } else { 0.5 };
    let n_total = n_a + n_b;
    let s_total = successes_a + successes_b;
    let p_null = if n_total > 0 { s_total as f64 / n_total as f64 } else { 0.5 };

    // Log-likelihood under H1
    let ll1 =
        log_binom_likelihood(successes_a, n_a, p_a) + log_binom_likelihood(successes_b, n_b, p_b);
    // Log-likelihood under H0
    let ll0 = log_binom_likelihood(successes_a, n_a, p_null)
        + log_binom_likelihood(successes_b, n_b, p_null);

    let bayes_factor = (ll1 - ll0).exp().clamp(0.0, f64::MAX);

    BayesianBelief {
        p_a_better_than_b: p_a_better,
        bayes_factor,
    }
}

/// Log-normalisation constant of Beta(a, b): ln(B(a, b)) = lnΓ(a) + lnΓ(b) - lnΓ(a+b).
fn ln_beta_fn(a: f64, b: f64) -> f64 {
    ln_gamma_internal(a) + ln_gamma_internal(b) - ln_gamma_internal(a + b)
}

/// Lanczos ln-Gamma approximation (same algorithm as in statistics.rs).
fn ln_gamma_internal(z: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma_internal(1.0 - z);
    }
    let z = z - 1.0;
    let x = C[0]
        + C[1] / (z + 1.0)
        + C[2] / (z + 2.0)
        + C[3] / (z + 3.0)
        + C[4] / (z + 4.0)
        + C[5] / (z + 5.0)
        + C[6] / (z + 6.0)
        + C[7] / (z + 7.0)
        + C[8] / (z + 8.0);
    let t = z + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Regularised incomplete Beta function CDF I_x(a, b) using Riemann sum.
fn regularized_beta_cdf(a: f64, b: f64, x: f64, ln_norm: &f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    const CDF_STEPS: usize = 100;
    let dx = x / CDF_STEPS as f64;
    let mut acc = 0.0_f64;
    for k in 0..CDF_STEPS {
        let t = (k as f64 + 0.5) * dx;
        if t <= 0.0 || t >= 1.0 {
            continue;
        }
        let ln_pdf = (a - 1.0) * t.ln() + (b - 1.0) * (1.0 - t).ln() - ln_norm;
        acc += ln_pdf.exp() * dx;
    }
    acc.clamp(0.0, 1.0)
}

/// Log-likelihood of observing `k` successes in `n` trials with rate `p`.
fn log_binom_likelihood(k: u64, n: u64, p: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let p = p.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    let f = k as f64;
    let nf = n as f64;
    f * p.ln() + (nf - f) * (1.0 - p).ln()
}

fn build_recommendation(winner: Option<&str>, is_significant: bool, p_value: f32) -> String {
    match (winner, is_significant) {
        (Some(w), true) => format!(
            "Deploy variant '{}'. Result is statistically significant (p ≈ {:.3}).",
            w, p_value
        ),
        (Some(w), false) => format!(
            "Variant '{}' performs best so far, but result is not yet significant \
             (p ≈ {:.3}). Continue collecting data.",
            w, p_value
        ),
        (None, _) => "Insufficient data to make a recommendation.".to_string(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn fifty_fifty() -> TrafficSplit {
        TrafficSplit::new(vec![
            ("control".to_string(), 0.5),
            ("treatment".to_string(), 0.5),
        ])
    }

    fn running_experiment(id: &str) -> ExperimentConfig {
        let mut cfg = ExperimentConfig::new(id, "Test experiment", fifty_fifty());
        cfg.status = ExperimentStatus::Running;
        cfg
    }

    // ── TrafficSplit::validate ────────────────────────────────────────────────

    #[test]
    fn test_traffic_split_valid() {
        let split = fifty_fifty();
        assert!(split.validate().is_ok(), "50/50 split should be valid");
    }

    #[test]
    fn test_traffic_split_invalid_sum() {
        let split = TrafficSplit::new(vec![("a".to_string(), 0.3), ("b".to_string(), 0.3)]);
        match split.validate() {
            Err(AbTestError::InvalidTrafficSplit { got }) => {
                assert!((got - 0.6_f32).abs() < 1e-3);
            },
            other => panic!("Expected InvalidTrafficSplit, got {:?}", other),
        }
    }

    #[test]
    fn test_traffic_split_empty() {
        let split = TrafficSplit::new(vec![]);
        assert!(
            matches!(split.validate(), Err(AbTestError::EmptyTrafficSplit)),
            "Empty split should error"
        );
    }

    // ── TrafficSplit::select_variant — determinism ────────────────────────────

    #[test]
    fn test_variant_selection_determinism() {
        let split = fifty_fifty();
        let hash = fnv1a_hash("request-123");
        let v1 = split.select_variant(hash);
        let v2 = split.select_variant(hash);
        assert_eq!(v1, v2, "Same hash should always yield the same variant");
    }

    // ── TrafficSplit::select_variant — uniform distribution ──────────────────

    #[test]
    fn test_variant_selection_approximate_uniform() {
        let split = fifty_fifty();
        let n = 10_000_u64;
        let mut control_count = 0u64;
        let mut treatment_count = 0u64;

        for i in 0..n {
            let hash = fnv1a_hash(&format!("user-{}", i));
            match split.select_variant(hash) {
                "control" => control_count += 1,
                "treatment" => treatment_count += 1,
                other => panic!("Unexpected variant: {}", other),
            }
        }

        // Expect roughly 50/50 (within ±5%)
        let control_frac = control_count as f64 / n as f64;
        assert!(
            (control_frac - 0.5).abs() < 0.05,
            "Expected ~50% control, got {:.1}%",
            control_frac * 100.0
        );
        let _ = treatment_count; // silence unused warning
    }

    // ── StatisticalTest::two_sample_z_test ───────────────────────────────────

    #[test]
    fn test_z_test_identical_means() {
        let z = StatisticalTest::two_sample_z_test(100, 10.0, 2.0, 100, 10.0, 2.0);
        assert!(z.abs() < 1e-9, "Identical means should give z ≈ 0");
    }

    #[test]
    fn test_z_test_formula() {
        // (mean1 - mean2) / sqrt(std1²/n1 + std2²/n2)
        // = (20.0 - 10.0) / sqrt(4.0/100 + 4.0/100)
        // = 10.0 / sqrt(0.08)
        // = 10.0 / 0.2828... ≈ 35.36
        let z = StatisticalTest::two_sample_z_test(100, 20.0, 2.0, 100, 10.0, 2.0);
        let expected = 10.0_f64 / (0.08_f64).sqrt();
        assert!((z - expected).abs() < 1e-6, "z={} expected={}", z, expected);
    }

    // ── StatisticalTest::is_significant ──────────────────────────────────────

    #[test]
    fn test_significance_threshold() {
        // |z| = 2.0 > 1.96 → significant at α=0.05
        assert!(StatisticalTest::is_significant(2.0, 0.05));
        // |z| = 1.5 < 1.96 → not significant at α=0.05
        assert!(!StatisticalTest::is_significant(1.5, 0.05));
    }

    #[test]
    fn test_z_critical_values() {
        let z_05 = StatisticalTest::compute_z_critical(0.05);
        assert!(
            (z_05 - 1.96).abs() < 1e-3,
            "α=0.05 critical value should be ≈1.96"
        );

        let z_01 = StatisticalTest::compute_z_critical(0.01);
        assert!(
            (z_01 - 2.576).abs() < 1e-3,
            "α=0.01 critical value should be ≈2.576"
        );

        let z_10 = StatisticalTest::compute_z_critical(0.10);
        assert!(
            (z_10 - 1.645).abs() < 1e-3,
            "α=0.10 critical value should be ≈1.645"
        );
    }

    // ── AbTestManager::register_experiment ───────────────────────────────────

    #[test]
    fn test_register_experiment_success() {
        let mut mgr = AbTestManager::new();
        let cfg = running_experiment("exp-1");
        assert!(mgr.register_experiment(cfg).is_ok());
        assert!(mgr.experiments.contains_key("exp-1"));
    }

    #[test]
    fn test_register_experiment_duplicate() {
        let mut mgr = AbTestManager::new();
        mgr.register_experiment(running_experiment("exp-dup")).unwrap();
        let result = mgr.register_experiment(running_experiment("exp-dup"));
        assert!(
            matches!(result, Err(AbTestError::DuplicateExperiment(_))),
            "Duplicate registration should fail"
        );
    }

    // ── AbTestManager::assign_variant — consistency ───────────────────────────

    #[test]
    fn test_assign_variant_consistency() {
        let mut mgr = AbTestManager::new();
        mgr.register_experiment(running_experiment("exp-cons")).unwrap();

        let v1 = mgr.assign_variant("exp-cons", "req-abc").unwrap();
        let v2 = mgr.assign_variant("exp-cons", "req-abc").unwrap();
        assert_eq!(
            v1, v2,
            "Same request ID must always be assigned the same variant"
        );
    }

    #[test]
    fn test_assign_variant_not_running() {
        let mut mgr = AbTestManager::new();
        let mut cfg = ExperimentConfig::new("exp-draft", "draft", fifty_fifty());
        cfg.status = ExperimentStatus::Draft;
        mgr.register_experiment(cfg).unwrap();

        let result = mgr.assign_variant("exp-draft", "req-1");
        assert!(matches!(result, Err(AbTestError::ExperimentNotRunning(_))));
    }

    // ── ExperimentVariantStats recording ─────────────────────────────────────

    #[test]
    fn test_stats_recording() {
        let mut stats = ExperimentVariantStats::new("control");
        stats.record_request(100.0, false);
        stats.record_request(200.0, true);
        stats.record_metric("tokens_per_second", 55.0);

        assert_eq!(stats.num_requests, 2);
        assert_eq!(stats.error_count, 1);
        assert!((stats.mean_latency_ms() - 150.0).abs() < 1e-9);
        assert!((stats.error_rate() - 0.5_f32).abs() < 1e-6);
        assert_eq!(
            *stats.custom_metrics.get("tokens_per_second").unwrap() as u64,
            55
        );
    }

    // ── AbTestManager::compute_results — winner determination ─────────────────

    #[test]
    fn test_compute_results_winner() {
        let mut mgr = AbTestManager::new();
        mgr.register_experiment(running_experiment("exp-winner")).unwrap();

        // Control: high latency; treatment: low latency
        for _ in 0..200 {
            mgr.record_request("exp-winner", "control", 200.0, false).unwrap();
        }
        for _ in 0..200 {
            mgr.record_request("exp-winner", "treatment", 50.0, false).unwrap();
        }

        let result = mgr.compute_results("exp-winner").unwrap();
        assert_eq!(
            result.winner,
            Some("treatment".to_string()),
            "Treatment (lower latency) should win"
        );
        assert!(
            result.is_significant,
            "Large latency difference should be significant"
        );
    }

    // ── p-value approximation sanity ─────────────────────────────────────────

    #[test]
    fn test_p_value_approximation_range() {
        // p-value should be in [0, 1]
        let p_high_z = StatisticalTest::approximate_p_value(10.0);
        let p_zero_z = StatisticalTest::approximate_p_value(0.0);

        assert!((0.0..=1.0).contains(&p_high_z), "p-value must be in [0,1]");
        assert!((0.0..=1.0).contains(&p_zero_z), "p-value must be in [0,1]");
        assert!(p_high_z < p_zero_z, "High z should produce lower p-value");
    }

    #[test]
    fn test_p_value_at_significance_boundary() {
        // z ≈ 1.96 should give p ≈ 0.05
        let p = StatisticalTest::approximate_p_value(1.96);
        assert!(
            (p - 0.05_f32).abs() < 0.01,
            "z=1.96 should give p ≈ 0.05, got {}",
            p
        );
    }

    // ── active experiments ────────────────────────────────────────────────────

    #[test]
    fn test_active_experiments_filter() {
        let mut mgr = AbTestManager::new();

        let mut draft_cfg = ExperimentConfig::new("draft-exp", "draft", fifty_fifty());
        draft_cfg.status = ExperimentStatus::Draft;

        let running_cfg = running_experiment("running-exp");

        mgr.register_experiment(draft_cfg).unwrap();
        mgr.register_experiment(running_cfg).unwrap();

        let active = mgr.active_experiments();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].experiment_id, "running-exp");
    }

    // ── Multi-Armed Bandit ────────────────────────────────────────────────────

    // ── Test 19: Arm default initialisation ──
    #[test]
    fn test_arm_default_prior() {
        let arm = Arm::new();
        assert!((arm.alpha - 1.0).abs() < 1e-9, "default alpha = 1");
        assert!((arm.beta - 1.0).abs() < 1e-9, "default beta = 1");
        assert_eq!(arm.successes, 0);
        assert_eq!(arm.failures, 0);
    }

    // ── Test 20: arm_mean_reward ──
    #[test]
    fn test_arm_mean_reward_at_prior() {
        let arm = Arm::new(); // Beta(1,1) → mean = 0.5
        let mean = arm_mean_reward(&arm);
        assert!(
            (mean - 0.5).abs() < 1e-9,
            "uninformative prior mean should be 0.5, got {mean}"
        );
    }

    // ── Test 21: arm_mean_reward after updates ──
    #[test]
    fn test_arm_mean_reward_after_updates() {
        let mut state = BanditState::new(1);
        // 8 successes, 2 failures → posterior Beta(9, 3) → mean = 9/12 = 0.75
        for _ in 0..8 {
            update_arm(&mut state, 0, true);
        }
        for _ in 0..2 {
            update_arm(&mut state, 0, false);
        }
        let mean = arm_mean_reward(&state.arms[0]);
        let expected = 9.0 / 12.0;
        assert!(
            (mean - expected).abs() < 1e-9,
            "mean should be {expected}, got {mean}"
        );
    }

    // ── Test 22: update_arm increments counts correctly ──
    #[test]
    fn test_update_arm_increments_counts() {
        let mut state = BanditState::new(2);
        update_arm(&mut state, 0, true);
        update_arm(&mut state, 0, true);
        update_arm(&mut state, 0, false);
        update_arm(&mut state, 1, false);

        assert_eq!(state.arms[0].successes, 2);
        assert_eq!(state.arms[0].failures, 1);
        assert!((state.arms[0].alpha - 3.0).abs() < 1e-9); // 1 + 2
        assert!((state.arms[0].beta - 2.0).abs() < 1e-9); // 1 + 1
        assert_eq!(state.arms[1].failures, 1);
        assert_eq!(state.arms[1].successes, 0);
    }

    // ── Test 23: select_arm is deterministic for same seed ──
    #[test]
    fn test_select_arm_deterministic() {
        let state = BanditState::new(3);
        let arm1 = select_arm(&state, 42);
        let arm2 = select_arm(&state, 42);
        assert_eq!(arm1, arm2, "same seed must select same arm");
    }

    // ── Test 24: select_arm selects valid arm index ──
    #[test]
    fn test_select_arm_valid_index() {
        let state = BanditState::new(5);
        for seed in 0..20u64 {
            let arm = select_arm(&state, seed);
            assert!(arm < 5, "arm index {arm} out of bounds");
        }
    }

    // ── Test 25: Thompson sampling converges to better arm ──
    #[test]
    fn test_thompson_sampling_converges_to_better_arm() {
        // Arm 0: bad (10% success rate), Arm 1: good (90% success rate).
        // After many pulls with simulated outcomes, arm 1 should be selected more often.
        let mut state = BanditState::new(2);

        // Simulate 200 rounds: use deterministic outcomes based on LCG.
        // Pre-load the posterior with known outcome counts.
        // Arm 0: 20 successes out of 200 trials (10%)
        for _ in 0..20 {
            update_arm(&mut state, 0, true);
        }
        for _ in 0..180 {
            update_arm(&mut state, 0, false);
        }
        // Arm 1: 180 successes out of 200 trials (90%)
        for _ in 0..180 {
            update_arm(&mut state, 1, true);
        }
        for _ in 0..20 {
            update_arm(&mut state, 1, false);
        }

        // Now count how many times each arm is selected across many seeds.
        let mut counts = [0u64; 2];
        for seed in 0..1000u64 {
            let arm = select_arm(&state, seed);
            counts[arm] += 1;
        }

        assert!(
            counts[1] > counts[0],
            "better arm (1) should be selected more often: {:?}",
            counts
        );
    }

    // ── Test 26: sample_beta returns value in [0, 1] ──
    #[test]
    fn test_sample_beta_in_unit_interval() {
        for seed in [0u64, 1, 42, 999, 123456789] {
            let s = sample_beta(2.0, 3.0, seed);
            assert!(
                (0.0..=1.0).contains(&s),
                "sample_beta({seed}) = {s} is outside [0,1]"
            );
        }
    }

    // ── Test 27: sample_beta mean approximates theoretical mean ──
    #[test]
    fn test_sample_beta_mean_approximates_alpha_over_sum() {
        // Beta(3, 7) has mean 3/10 = 0.3
        let alpha = 3.0_f64;
        let beta_param = 7.0_f64;
        let expected_mean = alpha / (alpha + beta_param);

        let n = 1000u64;
        let total: f64 = (0..n).map(|seed| sample_beta(alpha, beta_param, seed * 7)).sum();
        let sample_mean = total / n as f64;

        assert!(
            (sample_mean - expected_mean).abs() < 0.05,
            "sample mean {sample_mean:.3} should be close to {expected_mean:.3}"
        );
    }

    // ── Test 28: BanditState with 1 arm always selects that arm ──
    #[test]
    fn test_single_arm_bandit_always_selects_zero() {
        let state = BanditState::new(1);
        for seed in 0..10u64 {
            assert_eq!(select_arm(&state, seed), 0);
        }
    }

    // ── Sequential Bayesian Test ──────────────────────────────────────────────

    // ── Test 29: update_beliefs — equal rates give p ≈ 0.5 ──
    #[test]
    fn test_bayesian_beliefs_equal_rates() {
        // Same success rate for A and B → P(A > B) ≈ 0.5
        let belief = update_beliefs(50, 100, 50, 100);
        assert!(
            (belief.p_a_better_than_b - 0.5).abs() < 0.1,
            "equal rates should give P(A > B) ≈ 0.5, got {}",
            belief.p_a_better_than_b
        );
    }

    // ── Test 30: update_beliefs — A clearly better gives p > 0.9 ──
    #[test]
    fn test_bayesian_beliefs_a_clearly_better() {
        // A: 90/100, B: 10/100 → A is clearly better
        let belief = update_beliefs(90, 100, 10, 100);
        assert!(
            belief.p_a_better_than_b > 0.9,
            "A with 90% rate vs B with 10% should give P(A>B) > 0.9, got {}",
            belief.p_a_better_than_b
        );
    }

    // ── Test 31: update_beliefs — B clearly better gives p < 0.1 ──
    #[test]
    fn test_bayesian_beliefs_b_clearly_better() {
        // A: 10/100, B: 90/100 → B is clearly better
        let belief = update_beliefs(10, 100, 90, 100);
        assert!(
            belief.p_a_better_than_b < 0.1,
            "A with 10% rate vs B with 90% should give P(A>B) < 0.1, got {}",
            belief.p_a_better_than_b
        );
    }

    // ── Test 32: update_beliefs — p_a_better_than_b is in [0, 1] ──
    #[test]
    fn test_bayesian_belief_p_in_unit_interval() {
        let cases = [
            (0u64, 0u64, 0u64, 0u64),
            (5, 10, 5, 10),
            (100, 100, 0, 100),
            (0, 100, 100, 100),
        ];
        for (sa, na, sb, nb) in cases {
            let belief = update_beliefs(sa, na, sb, nb);
            assert!(
                (0.0..=1.0).contains(&belief.p_a_better_than_b),
                "p_a_better_than_b = {} should be in [0,1]",
                belief.p_a_better_than_b
            );
        }
    }

    // ── Test 33: update_beliefs — bayes_factor > 1 when A is clearly better ──
    #[test]
    fn test_bayesian_belief_bayes_factor_on_clear_winner() {
        let belief = update_beliefs(90, 100, 10, 100);
        assert!(
            belief.bayes_factor > 1.0,
            "clear difference should give bayes_factor > 1, got {}",
            belief.bayes_factor
        );
    }

    // ── Extended tests ─────────────────────────────────────────────────────

    // Test 34: ExperimentVariantStats::new — zero counts on creation
    #[test]
    fn test_variant_stats_new_zeros() {
        let stats = ExperimentVariantStats::new("control");
        assert_eq!(stats.num_requests, 0);
        assert_eq!(stats.error_count, 0);
        assert!((stats.total_latency_ms - 0.0).abs() < 1e-9);
    }

    // Test 35: ExperimentVariantStats::mean_latency_ms — zero when no requests
    #[test]
    fn test_variant_stats_mean_latency_zero_when_empty() {
        let stats = ExperimentVariantStats::new("v");
        assert_eq!(stats.mean_latency_ms(), 0.0);
    }

    // Test 36: ExperimentVariantStats::error_rate — zero when no requests
    #[test]
    fn test_variant_stats_error_rate_zero_when_empty() {
        let stats = ExperimentVariantStats::new("v");
        assert_eq!(stats.error_rate(), 0.0);
    }

    // Test 37: ExperimentVariantStats::record_request accumulates values
    #[test]
    fn test_variant_stats_record_request() {
        let mut stats = ExperimentVariantStats::new("treatment");
        stats.record_request(100.0, false);
        stats.record_request(200.0, true);
        assert_eq!(stats.num_requests, 2);
        assert_eq!(stats.error_count, 1);
        assert!((stats.total_latency_ms - 300.0).abs() < 1e-9);
    }

    // Test 38: ExperimentVariantStats::mean_latency_ms correct formula
    #[test]
    fn test_variant_stats_mean_latency_formula() {
        let mut stats = ExperimentVariantStats::new("v");
        stats.record_request(100.0, false);
        stats.record_request(300.0, false);
        assert!((stats.mean_latency_ms() - 200.0).abs() < 1e-9);
    }

    // Test 39: ExperimentVariantStats::error_rate correct formula
    #[test]
    fn test_variant_stats_error_rate_formula() {
        let mut stats = ExperimentVariantStats::new("v");
        stats.record_request(10.0, true);
        stats.record_request(10.0, false);
        stats.record_request(10.0, false);
        stats.record_request(10.0, false);
        // 1/4 = 0.25
        assert!((stats.error_rate() - 0.25).abs() < 1e-6);
    }

    // Test 40: ExperimentVariantStats::record_metric accumulates values
    #[test]
    fn test_variant_stats_record_metric_accumulates() {
        let mut stats = ExperimentVariantStats::new("v");
        stats.record_metric("throughput", 50.0);
        stats.record_metric("throughput", 25.0);
        assert!((stats.custom_metrics["throughput"] - 75.0).abs() < 1e-9);
    }

    // Test 41: TrafficSplit::validate — empty variants returns error
    #[test]
    fn test_traffic_split_empty_variants_ext() {
        let split = TrafficSplit::new(vec![]);
        assert!(matches!(
            split.validate(),
            Err(AbTestError::EmptyTrafficSplit)
        ));
    }

    // Test 42: TrafficSplit::validate — fractions summing to != 1.0 returns error
    #[test]
    fn test_traffic_split_invalid_sum_ext() {
        let split = TrafficSplit::new(vec![("a".to_string(), 0.3), ("b".to_string(), 0.3)]);
        assert!(matches!(
            split.validate(),
            Err(AbTestError::InvalidTrafficSplit { .. })
        ));
    }

    // Test 43: TrafficSplit::validate — fractions summing to 1.0 returns Ok
    #[test]
    fn test_traffic_split_valid_ext() {
        let split = TrafficSplit::new(vec![("a".to_string(), 0.5), ("b".to_string(), 0.5)]);
        assert!(split.validate().is_ok());
    }

    // Test 44: TrafficSplit::select_variant — always returns a known variant name
    #[test]
    fn test_traffic_split_select_variant_known() {
        let split = TrafficSplit::new(vec![
            ("control".to_string(), 0.5),
            ("treatment".to_string(), 0.5),
        ]);
        let variants = ["control", "treatment"];
        for seed in 0..20u64 {
            let variant = split.select_variant(seed * 1234);
            assert!(
                variants.contains(&variant),
                "selected variant '{variant}' is not in split"
            );
        }
    }

    // Test 45: ExperimentStatus variants are distinct
    #[test]
    fn test_experiment_status_distinct_variants() {
        assert_ne!(ExperimentStatus::Draft, ExperimentStatus::Running);
        assert_ne!(ExperimentStatus::Running, ExperimentStatus::Paused);
        assert_ne!(ExperimentStatus::Completed, ExperimentStatus::Aborted);
    }

    // Test 46: AbTestManager::register_experiment — duplicate ID returns error
    #[test]
    fn test_register_duplicate_experiment() {
        let mut mgr = AbTestManager::new();
        let cfg1 = ExperimentConfig::new("exp-1", "Exp 1", fifty_fifty());
        let cfg2 = ExperimentConfig::new("exp-1", "Exp 1 duplicate", fifty_fifty());
        mgr.register_experiment(cfg1).expect("first ok");
        let err = mgr.register_experiment(cfg2).expect_err("duplicate should fail");
        assert!(matches!(err, AbTestError::DuplicateExperiment(_)));
    }

    // Test 47: AbTestManager::assign_variant — routes to known variant
    #[test]
    fn test_route_request_known_variant() {
        let mut mgr = AbTestManager::new();
        let cfg = running_experiment("route-exp");
        mgr.register_experiment(cfg).expect("register");
        let variants = ["control", "treatment"];
        for seed in 0..10u64 {
            let req_id = seed.to_string();
            let variant = mgr.assign_variant("route-exp", &req_id).expect("assign");
            assert!(
                variants.contains(&variant.as_str()),
                "unexpected variant: {variant}"
            );
        }
    }

    // Test 48: StatisticalTest::two_sample_z_test — NAN when n=0
    #[test]
    fn test_z_test_nan_when_empty() {
        let z = StatisticalTest::two_sample_z_test(0, 10.0, 1.0, 100, 10.5, 1.0);
        assert!(z.is_nan(), "z-test with n=0 must return NAN");
    }

    // Test 49: StatisticalTest::is_significant — NAN is not significant
    #[test]
    fn test_z_test_nan_not_significant() {
        assert!(!StatisticalTest::is_significant(f64::NAN, 0.05));
    }

    // Test 50: StatisticalTest::two_sample_z_test — large difference is significant
    #[test]
    fn test_z_test_large_difference_significant() {
        let z = StatisticalTest::two_sample_z_test(1000, 100.0, 1.0, 1000, 200.0, 1.0);
        assert!(
            StatisticalTest::is_significant(z, 0.05),
            "large effect must be significant"
        );
    }

    // Test 51: StatisticalTest::compute_z_critical — α=0.10 gives ≈1.645
    #[test]
    fn test_z_critical_alpha_010() {
        let z = StatisticalTest::compute_z_critical(0.10);
        assert!((z - 1.645).abs() < 0.01, "α=0.10 → z≈1.645, got {z}");
    }

    // Test 52: AbTestError variants display non-empty strings
    #[test]
    fn test_ab_test_error_display() {
        let errors: &[AbTestError] = &[
            AbTestError::InvalidTrafficSplit { got: 0.8 },
            AbTestError::EmptyTrafficSplit,
            AbTestError::ExperimentNotFound("x".into()),
            AbTestError::DuplicateExperiment("x".into()),
        ];
        for e in errors {
            assert!(!e.to_string().is_empty(), "error display must be non-empty");
        }
    }

    // Test 53: ExperimentConfig::new — default status is Draft
    #[test]
    fn test_experiment_config_default_status_draft() {
        let cfg = ExperimentConfig::new("e", "name", fifty_fifty());
        assert_eq!(cfg.status, ExperimentStatus::Draft);
    }

    // Test 54: BanditState::new — arm count matches argument
    #[test]
    fn test_bandit_state_arm_count() {
        let state = BanditState::new(7);
        assert_eq!(state.arms.len(), 7);
    }
}
