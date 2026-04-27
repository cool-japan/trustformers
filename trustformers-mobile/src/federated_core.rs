//! Federated Learning Core Infrastructure
//!
//! Pure-Rust FedAvg aggregation with optional differential privacy noise.
//! Uses an LCG (Linear Congruential Generator) for deterministic randomness.
//! No rand/ndarray dependencies.

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for a federated learning session.
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// Total number of communication rounds.
    pub num_rounds: usize,
    /// Number of local training epochs each client runs per round.
    pub local_epochs: usize,
    /// Minimum number of clients required for aggregation.
    pub min_clients: usize,
    /// Fraction of total clients to sample per round (0 < fraction_fit ≤ 1).
    pub fraction_fit: f32,
    /// Optional differential-privacy configuration.
    pub differential_privacy: Option<DpConfig>,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            num_rounds: 10,
            local_epochs: 1,
            min_clients: 2,
            fraction_fit: 1.0,
            differential_privacy: None,
        }
    }
}

/// Differential privacy configuration (Gaussian mechanism).
#[derive(Debug, Clone)]
pub struct DpConfig {
    /// Privacy budget ε (epsilon).
    pub epsilon: f32,
    /// Failure probability δ (delta).
    pub delta: f32,
    /// Multiplier on `max_grad_norm` to set noise standard deviation.
    pub noise_multiplier: f32,
    /// Maximum ℓ₂ norm for gradient clipping.
    pub max_grad_norm: f32,
}

// ─── Client Update ────────────────────────────────────────────────────────────

/// A single client's model update produced during one federated round.
#[derive(Debug, Clone)]
pub struct ClientUpdate {
    /// Unique client identifier.
    pub client_id: String,
    /// Gradient (delta) vector produced by local training.
    pub model_delta: Vec<f32>,
    /// Number of samples the client trained on.
    pub num_samples: usize,
    /// Local training loss reported by the client.
    pub loss: f32,
}

impl ClientUpdate {
    /// Construct a new `ClientUpdate`.
    pub fn new(
        client_id: impl Into<String>,
        model_delta: Vec<f32>,
        num_samples: usize,
        loss: f32,
    ) -> Self {
        Self {
            client_id: client_id.into(),
            model_delta,
            num_samples,
            loss,
        }
    }
}

// ─── Round Metrics ────────────────────────────────────────────────────────────

/// Aggregated metrics for a single federated round.
#[derive(Debug, Clone)]
pub struct FedRoundMetrics {
    /// Unweighted mean loss across participating clients.
    pub mean_loss: f32,
    /// Number of clients that participated.
    pub num_clients: usize,
    /// Total number of samples across all clients.
    pub total_samples: usize,
    /// Sample-weighted mean loss (FedAvg-style).
    pub weighted_loss: f32,
}

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors that can occur during federated learning operations.
#[derive(Debug, PartialEq)]
pub enum FederatedError {
    /// Not enough clients joined the round.
    InsufficientClients { got: usize, need: usize },
    /// No client updates were provided.
    EmptyUpdates,
    /// The configuration is logically invalid.
    InvalidConfig(String),
    /// Client updates have incompatible model-delta dimensions.
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for FederatedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientClients { got, need } => {
                write!(f, "Need {need} clients, got {got}")
            }
            Self::EmptyUpdates => write!(f, "No client updates provided"),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for FederatedError {}

// ─── LCG ──────────────────────────────────────────────────────────────────────

/// Linear Congruential Generator (Knuth / glibc constants).
///
/// Used instead of `rand` crate to comply with the COOLJAPAN no-rand policy.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform float in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        // Use top 53 bits for mantissa precision
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }

    /// Approximate standard normal via Box-Muller transform.
    fn next_normal_f32(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10_f32);
        let u2 = self.next_f32();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0_f32 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

// ─── FedAvgAggregator ─────────────────────────────────────────────────────────

/// FedAvg aggregator implementing the McMahan et al. algorithm.
///
/// Aggregates client updates via sample-weighted averaging, with optional
/// Gaussian differential-privacy noise injection.
#[derive(Debug)]
pub struct FedAvgAggregator {
    /// Federated learning configuration.
    pub config: FederatedConfig,
}

impl FedAvgAggregator {
    /// Create a new aggregator, validating the configuration.
    ///
    /// Returns `Err(FederatedError::InvalidConfig)` if:
    /// - `min_clients == 0`
    /// - `fraction_fit` is not in `(0.0, 1.0]`
    pub fn new(config: FederatedConfig) -> Result<Self, FederatedError> {
        if config.min_clients == 0 {
            return Err(FederatedError::InvalidConfig(
                "min_clients must be >= 1".to_string(),
            ));
        }
        if config.fraction_fit <= 0.0 || config.fraction_fit > 1.0 {
            return Err(FederatedError::InvalidConfig(
                "fraction_fit must be in (0.0, 1.0]".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// FedAvg: compute the sample-weighted average of `model_delta` vectors.
    ///
    /// Fails if:
    /// - `updates` is empty
    /// - fewer updates than `min_clients`
    /// - updates have different `model_delta` lengths
    /// - total samples is zero
    pub fn aggregate(&self, updates: &[ClientUpdate]) -> Result<Vec<f32>, FederatedError> {
        if updates.is_empty() {
            return Err(FederatedError::EmptyUpdates);
        }
        if updates.len() < self.config.min_clients {
            return Err(FederatedError::InsufficientClients {
                got: updates.len(),
                need: self.config.min_clients,
            });
        }

        let dim = updates[0].model_delta.len();
        for update in updates.iter().skip(1) {
            if update.model_delta.len() != dim {
                return Err(FederatedError::DimensionMismatch {
                    expected: dim,
                    got: update.model_delta.len(),
                });
            }
        }

        let total_samples: usize = updates.iter().map(|u| u.num_samples).sum();
        if total_samples == 0 {
            return Err(FederatedError::InvalidConfig(
                "total samples across all clients is 0".to_string(),
            ));
        }

        let mut aggregated = vec![0.0_f32; dim];
        for update in updates {
            let weight = update.num_samples as f32 / total_samples as f32;
            for (a, d) in aggregated.iter_mut().zip(update.model_delta.iter()) {
                *a += weight * d;
            }
        }

        Ok(aggregated)
    }

    /// Sample `ceil(fraction_fit * total_clients)` client indices.
    ///
    /// Uses a partial Fisher-Yates shuffle seeded with `seed` (LCG).
    /// The returned indices are sorted ascending and have no duplicates.
    pub fn select_clients(&self, total_clients: usize, seed: u64) -> Vec<usize> {
        if total_clients == 0 {
            return Vec::new();
        }
        let n = ((self.config.fraction_fit * total_clients as f32).ceil() as usize)
            .min(total_clients);
        if n >= total_clients {
            return (0..total_clients).collect();
        }

        let mut rng = Lcg::new(seed);
        let mut indices: Vec<usize> = (0..total_clients).collect();
        // Partial Fisher-Yates: shuffle first `n` positions
        for i in 0..n {
            let remaining = total_clients - i;
            let j = i + (rng.next_u64() as usize % remaining);
            indices.swap(i, j);
        }

        let mut selected = indices[..n].to_vec();
        selected.sort_unstable();
        selected
    }

    /// Apply Gaussian DP noise to the aggregated update vector in-place.
    ///
    /// Noise standard deviation = `noise_multiplier * max_grad_norm`.
    /// Does nothing if `differential_privacy` is `None`.
    pub fn apply_dp_noise(&self, updates: &mut Vec<f32>, seed: u64) {
        if let Some(dp) = &self.config.differential_privacy {
            let sigma = dp.noise_multiplier * dp.max_grad_norm;
            let mut rng = Lcg::new(seed);
            for v in updates.iter_mut() {
                *v += sigma * rng.next_normal_f32();
            }
        }
    }

    /// Compute per-round aggregated metrics from a slice of client updates.
    pub fn round_metrics(&self, updates: &[ClientUpdate]) -> FedRoundMetrics {
        let num_clients = updates.len();
        if num_clients == 0 {
            return FedRoundMetrics {
                mean_loss: 0.0,
                num_clients: 0,
                total_samples: 0,
                weighted_loss: 0.0,
            };
        }

        let total_samples: usize = updates.iter().map(|u| u.num_samples).sum();
        let mean_loss = updates.iter().map(|u| u.loss).sum::<f32>() / num_clients as f32;
        let weighted_loss = if total_samples == 0 {
            mean_loss
        } else {
            updates
                .iter()
                .map(|u| u.loss * u.num_samples as f32)
                .sum::<f32>()
                / total_samples as f32
        };

        FedRoundMetrics {
            mean_loss,
            num_clients,
            total_samples,
            weighted_loss,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_update(id: &str, delta: Vec<f32>, samples: usize, loss: f32) -> ClientUpdate {
        ClientUpdate::new(id, delta, samples, loss)
    }

    fn default_config() -> FederatedConfig {
        FederatedConfig {
            num_rounds: 5,
            local_epochs: 1,
            min_clients: 2,
            fraction_fit: 1.0,
            differential_privacy: None,
        }
    }

    fn agg() -> FedAvgAggregator {
        FedAvgAggregator::new(default_config()).expect("valid config")
    }

    // ── FedAvg aggregation ────────────────────────────────────────────────────

    // Test 1: 3 clients with equal samples → simple average
    #[test]
    fn test_fedavg_basic() {
        let cfg = FederatedConfig { min_clients: 3, ..default_config() };
        let a = FedAvgAggregator::new(cfg).expect("valid");
        let updates = vec![
            make_update("c1", vec![1.0, 2.0, 3.0], 10, 0.5),
            make_update("c2", vec![3.0, 4.0, 5.0], 10, 0.4),
            make_update("c3", vec![5.0, 6.0, 7.0], 10, 0.3),
        ];
        let result = a.aggregate(&updates).expect("aggregate ok");
        // Equal weights (1/3 each): mean of [1,3,5]=3, [2,4,6]=4, [3,5,7]=5
        assert!((result[0] - 3.0).abs() < 1e-5, "got {}", result[0]);
        assert!((result[1] - 4.0).abs() < 1e-5, "got {}", result[1]);
        assert!((result[2] - 5.0).abs() < 1e-5, "got {}", result[2]);
    }

    // Test 2: 2 clients with different sample counts → weighted
    #[test]
    fn test_fedavg_weighted() {
        let updates = vec![
            make_update("c1", vec![0.0], 100, 1.0),
            make_update("c2", vec![1.0], 900, 1.0),
        ];
        let result = agg().aggregate(&updates).expect("ok");
        // weight c1=0.1, c2=0.9 → 0.1*0.0 + 0.9*1.0 = 0.9
        assert!((result[0] - 0.9).abs() < 1e-5, "got {}", result[0]);
    }

    // Test 3: empty updates → EmptyUpdates error
    #[test]
    fn test_fedavg_empty_updates() {
        let err = agg().aggregate(&[]).expect_err("should fail");
        assert_eq!(err, FederatedError::EmptyUpdates);
    }

    // Test 4: fewer clients than min_clients → InsufficientClients
    #[test]
    fn test_fedavg_insufficient_clients() {
        let cfg = FederatedConfig { min_clients: 3, ..default_config() };
        let a = FedAvgAggregator::new(cfg).expect("valid");
        let updates = vec![
            make_update("c1", vec![1.0], 10, 0.5),
            make_update("c2", vec![2.0], 10, 0.4),
        ];
        let err = a.aggregate(&updates).expect_err("should fail");
        assert!(matches!(err, FederatedError::InsufficientClients { got: 2, need: 3 }));
    }

    // Test 5: updates with different delta lengths → DimensionMismatch
    #[test]
    fn test_fedavg_dimension_mismatch() {
        let updates = vec![
            make_update("c1", vec![1.0, 2.0], 10, 0.5),
            make_update("c2", vec![3.0], 10, 0.4),
        ];
        let err = agg().aggregate(&updates).expect_err("should fail");
        assert!(matches!(err, FederatedError::DimensionMismatch { expected: 2, got: 1 }));
    }

    // Test 6: single client (min_clients=1) returns delta unchanged
    #[test]
    fn test_fedavg_single_client() {
        let cfg = FederatedConfig { min_clients: 1, ..default_config() };
        let a = FedAvgAggregator::new(cfg).expect("valid");
        let delta = vec![1.0_f32, 2.0, 3.0];
        let updates = vec![make_update("c1", delta.clone(), 50, 0.3)];
        let result = a.aggregate(&updates).expect("ok");
        for (r, d) in result.iter().zip(delta.iter()) {
            assert!((r - d).abs() < 1e-6);
        }
    }

    // ── select_clients ────────────────────────────────────────────────────────

    // Test 7: fraction_fit=1.0 returns all indices
    #[test]
    fn test_select_clients_all() {
        let selected = agg().select_clients(10, 42);
        assert_eq!(selected, (0..10).collect::<Vec<_>>());
    }

    // Test 8: fraction_fit=0.5, total=10 returns 5 indices
    #[test]
    fn test_select_clients_half() {
        let cfg = FederatedConfig { fraction_fit: 0.5, ..default_config() };
        let a = FedAvgAggregator::new(cfg).expect("valid");
        let selected = a.select_clients(10, 99);
        assert_eq!(selected.len(), 5);
    }

    // Test 9: returned indices are sorted ascending
    #[test]
    fn test_select_clients_sorted() {
        let cfg = FederatedConfig { fraction_fit: 0.5, ..default_config() };
        let a = FedAvgAggregator::new(cfg).expect("valid");
        let selected = a.select_clients(20, 7);
        for w in selected.windows(2) {
            assert!(w[0] < w[1], "not sorted: {w:?}");
        }
    }

    // Test 10: selected indices are unique
    #[test]
    fn test_select_clients_no_duplicates() {
        let cfg = FederatedConfig { fraction_fit: 0.5, min_clients: 2, ..default_config() };
        let a = FedAvgAggregator::new(cfg).expect("valid");
        let selected = a.select_clients(20, 123);
        let mut deduped = selected.clone();
        deduped.dedup();
        assert_eq!(selected, deduped);
    }

    // Test 11: same seed returns same selection
    #[test]
    fn test_select_clients_deterministic() {
        let cfg = FederatedConfig { fraction_fit: 0.6, ..default_config() };
        let a = FedAvgAggregator::new(cfg).expect("valid");
        let s1 = a.select_clients(15, 777);
        let s2 = a.select_clients(15, 777);
        assert_eq!(s1, s2);
    }

    // Test 12: different seeds return different selections (probabilistic)
    #[test]
    fn test_select_clients_different_seeds() {
        let cfg = FederatedConfig { fraction_fit: 0.5, ..default_config() };
        let a = FedAvgAggregator::new(cfg).expect("valid");
        let s1 = a.select_clients(100, 1);
        let s2 = a.select_clients(100, 9999);
        // With 50/100 selection from different seeds, extremely unlikely to be identical
        assert_ne!(s1, s2, "different seeds should produce different selections");
    }

    // ── apply_dp_noise ────────────────────────────────────────────────────────

    // Test 13: DP noise changes at least one value
    #[test]
    fn test_apply_dp_noise_changes_values() {
        let dp = DpConfig {
            epsilon: 1.0,
            delta: 1e-5,
            noise_multiplier: 1.0,
            max_grad_norm: 1.0,
        };
        let cfg = FederatedConfig {
            differential_privacy: Some(dp),
            ..default_config()
        };
        let a = FedAvgAggregator::new(cfg).expect("valid");
        let original = vec![1.0_f32; 100];
        let mut noisy = original.clone();
        a.apply_dp_noise(&mut noisy, 42);
        let changed = noisy.iter().zip(original.iter()).any(|(n, o)| (n - o).abs() > 1e-9);
        assert!(changed, "DP noise should change at least one value");
    }

    // Test 14: no DP config → values unchanged
    #[test]
    fn test_apply_dp_noise_no_dp_config() {
        let a = agg(); // no DP
        let original = vec![2.5_f32; 10];
        let mut v = original.clone();
        a.apply_dp_noise(&mut v, 42);
        assert_eq!(v, original);
    }

    // Test 15: same seed produces same noise
    #[test]
    fn test_apply_dp_noise_deterministic() {
        let dp = DpConfig {
            epsilon: 1.0,
            delta: 1e-5,
            noise_multiplier: 0.5,
            max_grad_norm: 1.0,
        };
        let cfg = FederatedConfig {
            differential_privacy: Some(dp),
            ..default_config()
        };
        let a = FedAvgAggregator::new(cfg).expect("valid");
        let base = vec![0.0_f32; 20];
        let mut v1 = base.clone();
        let mut v2 = base.clone();
        a.apply_dp_noise(&mut v1, 1234);
        a.apply_dp_noise(&mut v2, 1234);
        assert_eq!(v1, v2, "same seed must produce identical noise");
    }

    // ── round_metrics ─────────────────────────────────────────────────────────

    // Test 16: mean_loss is unweighted average
    #[test]
    fn test_round_metrics_mean_loss() {
        let updates = vec![
            make_update("c1", vec![0.0], 10, 1.0),
            make_update("c2", vec![0.0], 10, 3.0),
        ];
        let metrics = agg().round_metrics(&updates);
        assert!((metrics.mean_loss - 2.0).abs() < 1e-5);
    }

    // Test 17: weighted_loss respects sample counts
    #[test]
    fn test_round_metrics_weighted_loss() {
        let updates = vec![
            make_update("c1", vec![0.0], 100, 1.0),
            make_update("c2", vec![0.0], 900, 3.0),
        ];
        let metrics = agg().round_metrics(&updates);
        // weighted: (100*1.0 + 900*3.0) / 1000 = 2800/1000 = 2.8
        assert!((metrics.weighted_loss - 2.8).abs() < 1e-5, "got {}", metrics.weighted_loss);
    }

    // Test 18: empty updates returns zero metrics
    #[test]
    fn test_round_metrics_empty() {
        let metrics = agg().round_metrics(&[]);
        assert_eq!(metrics.num_clients, 0);
        assert_eq!(metrics.total_samples, 0);
        assert_eq!(metrics.mean_loss, 0.0);
        assert_eq!(metrics.weighted_loss, 0.0);
    }

    // Test 19: total_samples sums correctly
    #[test]
    fn test_round_metrics_total_samples() {
        let updates = vec![
            make_update("c1", vec![0.0], 50, 0.5),
            make_update("c2", vec![0.0], 75, 0.6),
            make_update("c3", vec![0.0], 25, 0.7),
        ];
        let metrics = agg().round_metrics(&updates);
        assert_eq!(metrics.total_samples, 150);
        assert_eq!(metrics.num_clients, 3);
    }

    // ── FedAvgAggregator::new validation ─────────────────────────────────────

    // Test 20: min_clients=0 returns InvalidConfig
    #[test]
    fn test_fedavg_new_invalid_min_clients() {
        let cfg = FederatedConfig { min_clients: 0, ..default_config() };
        let err = FedAvgAggregator::new(cfg).expect_err("should fail");
        assert!(matches!(err, FederatedError::InvalidConfig(_)));
    }

    // Test 21: fraction_fit=0.0 returns error
    #[test]
    fn test_fedavg_new_invalid_fraction_fit_zero() {
        let cfg = FederatedConfig { fraction_fit: 0.0, ..default_config() };
        let err = FedAvgAggregator::new(cfg).expect_err("should fail");
        assert!(matches!(err, FederatedError::InvalidConfig(_)));
    }

    // Test 22: fraction_fit=1.5 returns error
    #[test]
    fn test_fedavg_new_invalid_fraction_fit_over_one() {
        let cfg = FederatedConfig { fraction_fit: 1.5, ..default_config() };
        let err = FedAvgAggregator::new(cfg).expect_err("should fail");
        assert!(matches!(err, FederatedError::InvalidConfig(_)));
    }

    // Test 23: valid config succeeds
    #[test]
    fn test_fedavg_new_valid() {
        let cfg = FederatedConfig {
            num_rounds: 20,
            local_epochs: 3,
            min_clients: 5,
            fraction_fit: 0.8,
            differential_privacy: None,
        };
        assert!(FedAvgAggregator::new(cfg).is_ok());
    }
}
