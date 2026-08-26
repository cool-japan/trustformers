//! # Federated Learning Optimization
//!
//! This module implements algorithms for federated learning, enabling distributed
//! training across multiple clients while preserving privacy and handling
//! heterogeneous data distributions.
//!
//! ## Available Algorithms
//!
//! - **FedAvg**: Standard federated averaging algorithm
//! - **FedProx**: Federated optimization with proximal regularization
//! - **Secure Aggregation**: Privacy-preserving parameter aggregation
//! - **Differential Privacy**: Add noise for enhanced privacy protection
//! - **Client Selection**: Strategies for selecting participating clients

// reason: research-stage module — reserved API/scaffolding fields and methods
// retained intentionally for in-progress features; not yet on active call paths.
#![allow(dead_code)]

use anyhow::{anyhow, Result};
use scirs2_core::random::StdRng; // Explicit import for type clarity
use scirs2_core::random::*; // SciRS2 Integration Policy - Replaces rand
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use trustformers_core::tensor::Tensor;

/// Configuration for federated averaging (FedAvg).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FedAvgConfig {
    /// Number of local epochs per client
    pub local_epochs: usize,
    /// Local learning rate for client updates
    pub local_learning_rate: f32,
    /// Fraction of clients participating per round
    pub client_fraction: f32,
    /// Minimum number of clients required per round
    pub min_clients: usize,
    /// Maximum number of clients per round
    pub max_clients: usize,
    /// Weight decay for regularization
    pub weight_decay: f32,
}

impl Default for FedAvgConfig {
    fn default() -> Self {
        Self {
            local_epochs: 5,
            local_learning_rate: 1e-3,
            client_fraction: 0.1,
            min_clients: 2,
            max_clients: 100,
            weight_decay: 0.0,
        }
    }
}

/// Configuration for FedProx algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FedProxConfig {
    /// FedAvg configuration
    pub fedavg_config: FedAvgConfig,
    /// Proximal term coefficient (μ)
    pub mu: f32,
}

impl Default for FedProxConfig {
    fn default() -> Self {
        Self {
            fedavg_config: FedAvgConfig::default(),
            mu: 0.01,
        }
    }
}

/// Configuration for differential privacy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialPrivacyConfig {
    /// Privacy budget (epsilon)
    pub epsilon: f32,
    /// Delta parameter for (ε,δ)-differential privacy
    pub delta: f32,
    /// Sensitivity of the function (max change in output per unit change in input)
    pub sensitivity: f32,
    /// Noise mechanism to use
    pub noise_mechanism: NoiseMechanism,
}

impl Default for DifferentialPrivacyConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-5,
            sensitivity: 1.0,
            noise_mechanism: NoiseMechanism::Gaussian,
        }
    }
}

/// Types of noise mechanisms for differential privacy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseMechanism {
    /// Gaussian noise
    Gaussian,
    /// Laplace noise
    Laplace,
}

/// Client selection strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientSelectionStrategy {
    /// Random selection
    Random,
    /// Selection based on data size
    DataSize,
    /// Selection based on computational capacity
    ComputeCapacity,
    /// Selection based on communication quality
    CommunicationQuality,
}

/// Information about a federated client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Client identifier
    pub client_id: String,
    /// Number of data samples
    pub data_size: usize,
    /// Computational capacity (relative metric)
    pub compute_capacity: f32,
    /// Communication quality (bandwidth, latency, etc.)
    pub communication_quality: f32,
    /// Client availability
    pub available: bool,
}

/// Federated Averaging (FedAvg) optimizer.
///
/// Implements the standard federated learning algorithm where clients
/// perform local updates and the server aggregates them via weighted averaging.
#[derive(Debug)]
pub struct FedAvg {
    config: FedAvgConfig,
    global_parameters: Vec<Tensor>,
    client_weights: HashMap<String, f32>,
    current_round: usize,
    selected_clients: Vec<String>,
    rng: StdRng,
}

impl FedAvg {
    /// Create a new FedAvg optimizer.
    pub fn new(config: FedAvgConfig) -> Self {
        Self {
            config,
            global_parameters: Vec::new(),
            client_weights: HashMap::new(),
            current_round: 0,
            selected_clients: Vec::new(),
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Initialize global parameters.
    pub fn initialize_global_parameters(&mut self, parameters: Vec<Tensor>) {
        self.global_parameters = parameters;
    }

    /// Select clients for the current round.
    pub fn select_clients(
        &mut self,
        available_clients: &[ClientInfo],
        strategy: ClientSelectionStrategy,
    ) -> Result<Vec<String>> {
        let available: Vec<&ClientInfo> =
            available_clients.iter().filter(|c| c.available).collect();

        if available.is_empty() {
            return Err(anyhow!("No available clients"));
        }

        let num_clients = (available.len() as f32 * self.config.client_fraction).round() as usize;
        let num_clients = num_clients
            .max(self.config.min_clients)
            .min(self.config.max_clients)
            .min(available.len());

        let selected = match strategy {
            ClientSelectionStrategy::Random => {
                let mut indices: Vec<usize> = (0..available.len()).collect();
                for i in 0..num_clients {
                    let j = self.rng.random_range(i..indices.len());
                    indices.swap(i, j);
                }
                indices[..num_clients].iter().map(|&i| available[i].client_id.clone()).collect()
            },
            ClientSelectionStrategy::DataSize => {
                let mut clients_with_size: Vec<_> =
                    available.iter().map(|c| (c.client_id.clone(), c.data_size)).collect();
                clients_with_size.sort_by_key(|(_, size)| std::cmp::Reverse(*size));
                clients_with_size[..num_clients].iter().map(|(id, _)| id.clone()).collect()
            },
            ClientSelectionStrategy::ComputeCapacity => {
                let mut clients_with_capacity: Vec<_> =
                    available.iter().map(|c| (c.client_id.clone(), c.compute_capacity)).collect();
                clients_with_capacity.sort_by(|(_, a), (_, b)| {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                });
                clients_with_capacity[..num_clients].iter().map(|(id, _)| id.clone()).collect()
            },
            ClientSelectionStrategy::CommunicationQuality => {
                let mut clients_with_quality: Vec<_> = available
                    .iter()
                    .map(|c| (c.client_id.clone(), c.communication_quality))
                    .collect();
                clients_with_quality.sort_by(|(_, a), (_, b)| {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                });
                clients_with_quality[..num_clients].iter().map(|(id, _)| id.clone()).collect()
            },
        };

        self.selected_clients = selected;
        Ok(self.selected_clients.clone())
    }

    /// Aggregate client updates using weighted averaging.
    pub fn aggregate_updates(
        &mut self,
        client_updates: HashMap<String, Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        if client_updates.is_empty() {
            return Err(anyhow!("No client updates to aggregate"));
        }

        let total_weight: f32 = client_updates
            .keys()
            .map(|client_id| self.client_weights.get(client_id).unwrap_or(&1.0))
            .sum();

        if total_weight == 0.0 {
            return Err(anyhow!("Total client weight is zero"));
        }

        // Initialize aggregated parameters with zeros
        let param_count = client_updates
            .values()
            .next()
            .ok_or_else(|| anyhow::anyhow!("client_updates must have at least one entry"))?
            .len();
        let mut aggregated = Vec::with_capacity(param_count);

        for i in 0..param_count {
            // Get shape from first client's parameter
            let first_param = &client_updates
                .values()
                .next()
                .ok_or_else(|| anyhow::anyhow!("client_updates must have at least one entry"))?[i];
            aggregated.push(Tensor::zeros_like(first_param)?);
        }

        // Weighted aggregation
        for (client_id, updates) in &client_updates {
            let weight = self.client_weights.get(client_id).unwrap_or(&1.0) / total_weight;

            for (i, update) in updates.iter().enumerate() {
                let weighted_update = update.mul_scalar(weight)?;
                aggregated[i] = aggregated[i].add(&weighted_update)?;
            }
        }

        // Update global parameters
        self.global_parameters = aggregated.clone();
        self.current_round += 1;

        Ok(aggregated)
    }

    /// Set client weights for aggregation.
    pub fn set_client_weights(&mut self, weights: HashMap<String, f32>) {
        self.client_weights = weights;
    }

    /// Get current global parameters.
    pub fn get_global_parameters(&self) -> &[Tensor] {
        &self.global_parameters
    }

    /// Get current round number.
    pub fn get_current_round(&self) -> usize {
        self.current_round
    }
}

/// FedProx optimizer with proximal regularization.
///
/// Extends FedAvg with a proximal term to handle client heterogeneity
/// by adding regularization that keeps client updates close to global model.
#[derive(Debug)]
pub struct FedProx {
    fedavg: FedAvg,
    config: FedProxConfig,
}

impl FedProx {
    /// Create a new FedProx optimizer.
    pub fn new(config: FedProxConfig) -> Self {
        Self {
            fedavg: FedAvg::new(config.fedavg_config.clone()),
            config,
        }
    }

    /// Compute proximal term for client update.
    pub fn compute_proximal_term(
        &self,
        client_params: &[Tensor],
        global_params: &[Tensor],
    ) -> Result<f32> {
        if client_params.len() != global_params.len() {
            return Err(anyhow!("Parameter count mismatch"));
        }

        let mut proximal_loss = 0.0;
        for (client_param, global_param) in client_params.iter().zip(global_params.iter()) {
            let diff = client_param.sub(global_param)?;
            let norm_sq = diff.norm_squared()?.to_scalar()?;
            proximal_loss += norm_sq;
        }

        Ok(self.config.mu * proximal_loss / 2.0)
    }

    /// Apply proximal update to client parameters.
    pub fn apply_proximal_update(
        &self,
        client_params: &mut [Tensor],
        global_params: &[Tensor],
        learning_rate: f32,
    ) -> Result<()> {
        for (client_param, global_param) in client_params.iter_mut().zip(global_params.iter()) {
            let diff = client_param.sub(global_param)?;
            let proximal_grad = diff.mul_scalar(self.config.mu)?;
            let update = proximal_grad.mul_scalar(learning_rate)?;
            *client_param = client_param.sub(&update)?;
        }
        Ok(())
    }

    /// Delegate to FedAvg for other operations.
    pub fn select_clients(
        &mut self,
        available_clients: &[ClientInfo],
        strategy: ClientSelectionStrategy,
    ) -> Result<Vec<String>> {
        self.fedavg.select_clients(available_clients, strategy)
    }

    pub fn aggregate_updates(
        &mut self,
        client_updates: HashMap<String, Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        self.fedavg.aggregate_updates(client_updates)
    }

    pub fn get_global_parameters(&self) -> &[Tensor] {
        self.fedavg.get_global_parameters()
    }

    pub fn get_current_round(&self) -> usize {
        self.fedavg.get_current_round()
    }
}

/// Differential privacy mechanism for federated learning.
pub struct DifferentialPrivacy {
    config: DifferentialPrivacyConfig,
    rng: StdRng,
}

impl DifferentialPrivacy {
    /// Create a new differential privacy mechanism.
    pub fn new(config: DifferentialPrivacyConfig) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Add noise to parameters for differential privacy.
    pub fn add_noise(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        let noise_scale = self.compute_noise_scale()?;

        for param in parameters.iter_mut() {
            let noise = self.generate_noise_tensor(param, noise_scale)?;
            *param = param.add(&noise)?;
        }

        Ok(())
    }

    fn compute_noise_scale(&self) -> Result<f32> {
        match self.config.noise_mechanism {
            NoiseMechanism::Gaussian => {
                // For Gaussian mechanism: σ = sqrt(2 * ln(1.25/δ)) * Δf / ε
                let ln_term = (1.25 / self.config.delta).ln();
                let sigma = (2.0 * ln_term).sqrt() * self.config.sensitivity / self.config.epsilon;
                Ok(sigma)
            },
            NoiseMechanism::Laplace => {
                // For Laplace mechanism: b = Δf / ε
                Ok(self.config.sensitivity / self.config.epsilon)
            },
        }
    }

    fn generate_noise_tensor(&mut self, reference: &Tensor, scale: f32) -> Result<Tensor> {
        let shape = reference.shape();
        let mut noise_data = Vec::new();

        match self.config.noise_mechanism {
            NoiseMechanism::Gaussian => {
                use scirs2_core::random::{Distribution, Normal}; // SciRS2 Integration Policy
                let normal = Normal::new(0.0, scale)
                    .map_err(|e| anyhow!("Normal distribution error: {}", e))?;

                for _ in 0..shape.iter().product::<usize>() {
                    noise_data.push(normal.sample(&mut self.rng));
                }
            },
            NoiseMechanism::Laplace => {
                // Use exponential distribution to simulate Laplace
                // Laplace(0, b) can be simulated as: sign * Exponential(1/b)
                use scirs2_core::random::{Distribution, Exp}; // SciRS2 Integration Policy
                let exp_dist = Exp::new(1.0 / scale)
                    .map_err(|e| anyhow!("Exponential distribution error: {}", e))?;

                for _ in 0..shape.iter().product::<usize>() {
                    let sign = if self.rng.random::<bool>() { 1.0 } else { -1.0 };
                    let exp_sample = exp_dist.sample(&mut self.rng);
                    noise_data.push(sign * exp_sample);
                }
            },
        }

        Ok(Tensor::from_data(noise_data, &shape.to_vec())?)
    }
}

/// Secure aggregation for federated learning.
///
/// Implements privacy-preserving aggregation where the server cannot
/// see individual client updates, only the aggregated result.
pub struct SecureAggregation {
    threshold: usize,
    total_clients: usize,
}

impl SecureAggregation {
    /// Create a new secure aggregation instance.
    pub fn new(threshold: usize, total_clients: usize) -> Result<Self> {
        if threshold > total_clients {
            return Err(anyhow!("Threshold cannot exceed total clients"));
        }

        Ok(Self {
            threshold,
            total_clients,
        })
    }

    /// Deterministically derive the pairwise PRG seed two clients share for
    /// masking round `round`. Symmetric in `client_a`/`client_b`, so both
    /// clients independently derive the *same* seed without communicating
    /// (each already knows both its own id and the id it's pairing with).
    ///
    /// Uses [`DefaultHasher`], whose algorithm the standard library does not
    /// guarantee to be stable across Rust compiler versions -- only within a
    /// single build. This is fine for this deterministic in-process
    /// primitive (see [`Self::generate_masks`]'s doc comment) as long as
    /// every participating client is running the same build; it would need
    /// a cross-version-stable hash (e.g. a fixed-algorithm one) before
    /// clients could be deployed from independently-built binaries.
    fn pairwise_seed(client_a: &str, client_b: &str, round: usize) -> u64 {
        let (lower, upper) =
            if client_a <= client_b { (client_a, client_b) } else { (client_b, client_a) };
        let mut hasher = DefaultHasher::new();
        lower.hash(&mut hasher);
        upper.hash(&mut hasher);
        round.hash(&mut hasher);
        hasher.finish()
    }

    /// Generate `client_id`'s pairwise-cancelling masks for `parameter_shapes`
    /// (the caller's real model parameter shapes, in the fixed order every
    /// client and the server agree on for this round).
    ///
    /// Uses the standard pairwise-masking construction for secure
    /// aggregation (Bonawitz et al.): for every OTHER id in
    /// `all_client_ids`, `client_id` and that client derive the same seed
    /// (via `Self::pairwise_seed`) and therefore the same pseudorandom
    /// values -- `client_id` adds them to its mask if it sorts before the
    /// other id, subtracts them otherwise. Summing every participant's mask
    /// together then cancels exactly (up to floating-point rounding): each
    /// pairwise contribution appears once with each sign. See
    /// [`Self::secure_aggregate`] for the aggregation side and what this
    /// construction does and does not protect against.
    ///
    /// `all_client_ids` must be the exact same participant set (including
    /// `client_id` itself) on every client's call for a given `round`, and
    /// `parameter_shapes` must be given in the same order everywhere, or the
    /// masks will not cancel. This does not implement dropout recovery (a
    /// full Bonawitz-style scheme additionally secret-shares each pairwise
    /// seed so surviving clients can reconstruct a dropped client's
    /// contribution): if any client whose id appears in `all_client_ids`
    /// does not actually submit a masked update to
    /// [`Self::secure_aggregate`], the missing client's pairwise terms are
    /// never cancelled and the aggregate is biased by exactly that client's
    /// unpaired contribution.
    pub fn generate_masks(
        &self,
        client_id: &str,
        all_client_ids: &[String],
        round: usize,
        parameter_shapes: &[Vec<usize>],
    ) -> Result<Vec<Tensor>> {
        if !all_client_ids.iter().any(|id| id == client_id) {
            return Err(anyhow!(
                "client_id {client_id} is not present in all_client_ids; this client's masks \
                 would not have matching pairwise partners to cancel against"
            ));
        }

        let mut accumulators: Vec<Vec<f32>> = parameter_shapes
            .iter()
            .map(|shape| vec![0.0f32; shape.iter().product::<usize>()])
            .collect();

        for other_id in all_client_ids {
            if other_id == client_id {
                continue;
            }
            // `client_id`/`other_id` agree on the seed regardless of which
            // one calls `generate_masks`; the sign is what makes the two
            // sides' contributions cancel rather than duplicate.
            let sign: f32 = if client_id < other_id.as_str() { 1.0 } else { -1.0 };
            let mut pair_rng =
                StdRng::seed_from_u64(Self::pairwise_seed(client_id, other_id, round));

            // One RNG stream per pair, drawn across all parameters in the
            // caller-fixed order: both sides advance it identically, so the
            // values -- and therefore the cancellation -- line up parameter
            // by parameter.
            for (accumulator, shape) in accumulators.iter_mut().zip(parameter_shapes.iter()) {
                let mask_size = shape.iter().product::<usize>();
                for slot in accumulator.iter_mut().take(mask_size) {
                    let value: f32 = pair_rng.random_range(-1.0..1.0);
                    *slot += sign * value;
                }
            }
        }

        let mut masks = Vec::with_capacity(accumulators.len());
        for (data, shape) in accumulators.into_iter().zip(parameter_shapes.iter()) {
            masks.push(Tensor::from_data(data, shape)?);
        }
        Ok(masks)
    }

    /// Sum (and average) masked client updates without the server ever
    /// seeing an individual client's true update.
    ///
    /// This assumes every masked update in `masked_updates` was produced by
    /// [`Self::generate_masks`] with the same `all_client_ids`/`round`
    /// (i.e. `masked_updates.keys()` matches `all_client_ids` exactly): the
    /// pairwise masks then cancel exactly when summed (up to
    /// floating-point rounding), leaving the true sum. If any participant
    /// named in that `all_client_ids` set is missing from `masked_updates`
    /// (a dropout), its pairwise terms are NOT cancelled and the result is
    /// biased by that client's unpaired mask contribution -- this
    /// implementation has no secret-sharing-based dropout recovery (see
    /// [`Self::generate_masks`]'s doc comment). `threshold` only checks a
    /// minimum client *count*; it does not verify the update set actually
    /// matches a `generate_masks` call.
    pub fn secure_aggregate(
        &self,
        masked_updates: HashMap<String, Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        if masked_updates.len() < self.threshold {
            return Err(anyhow!("Not enough clients for secure aggregation"));
        }

        // Enhanced secure aggregation with validation and error handling
        let mut result = Vec::new();
        let client_count = masked_updates.len() as f32;

        // Validate that all clients have the same number of parameters
        let parameter_count =
            masked_updates.values().next().map(|update| update.len()).unwrap_or(0);

        for (client_id, update) in &masked_updates {
            if update.len() != parameter_count {
                return Err(anyhow!(
                    "Client {} has {} parameters, expected {}",
                    client_id,
                    update.len(),
                    parameter_count
                ));
            }
        }

        // Aggregate masked updates parameter by parameter
        for param_idx in 0..parameter_count {
            // Collect all client updates for this parameter
            let mut parameter_updates = Vec::new();
            let mut expected_shape: Option<Vec<usize>> = None;

            for (client_id, update) in &masked_updates {
                let param_update = &update[param_idx];

                // Validate tensor shapes are consistent across clients
                if let Some(ref shape) = expected_shape {
                    if param_update.shape() != *shape {
                        return Err(anyhow!(
                            "Client {} parameter {} has shape {:?}, expected {:?}",
                            client_id,
                            param_idx,
                            param_update.shape(),
                            shape
                        ));
                    }
                } else {
                    expected_shape = Some(param_update.shape());
                }

                parameter_updates.push(param_update);
            }

            // Sum all client updates for this parameter
            let shape = expected_shape
                .ok_or_else(|| anyhow!("No client updates found for parameter {}", param_idx))?;
            let mut aggregated_param = Tensor::zeros(&shape)?;
            for param_update in parameter_updates {
                aggregated_param = aggregated_param.add(param_update)?;
            }

            // Average the aggregated parameter. With pairwise masks from
            // `generate_masks` and no dropouts, the mask terms cancelled out
            // during the summation above (see this function's doc comment),
            // so this recovers the true average without the server ever
            // seeing an individual client's true update.
            result.push(aggregated_param.div_scalar(client_count)?);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fedavg_config_default() {
        let config = FedAvgConfig::default();
        assert_eq!(config.local_epochs, 5);
        assert_eq!(config.client_fraction, 0.1);
        assert_eq!(config.min_clients, 2);
    }

    #[test]
    fn test_fedprox_config_default() {
        let config = FedProxConfig::default();
        assert_eq!(config.mu, 0.01);
        assert_eq!(config.fedavg_config.local_epochs, 5);
    }

    #[test]
    fn test_differential_privacy_config() {
        let config = DifferentialPrivacyConfig::default();
        assert_eq!(config.epsilon, 1.0);
        assert_eq!(config.delta, 1e-5);
        assert!(matches!(config.noise_mechanism, NoiseMechanism::Gaussian));
    }

    #[test]
    fn test_client_selection_strategies() {
        let clients = vec![
            ClientInfo {
                client_id: "client1".to_string(),
                data_size: 100,
                compute_capacity: 0.8,
                communication_quality: 0.9,
                available: true,
            },
            ClientInfo {
                client_id: "client2".to_string(),
                data_size: 200,
                compute_capacity: 0.6,
                communication_quality: 0.7,
                available: true,
            },
        ];

        let mut fedavg = FedAvg::new(FedAvgConfig::default());

        // Test random selection
        let selected = fedavg
            .select_clients(&clients, ClientSelectionStrategy::Random)
            .expect("Operation failed in test");
        assert!(!selected.is_empty());

        // Test data size selection
        let selected = fedavg
            .select_clients(&clients, ClientSelectionStrategy::DataSize)
            .expect("Operation failed in test");
        assert!(!selected.is_empty());
    }

    #[test]
    fn test_secure_aggregation_creation() {
        let secure_agg = SecureAggregation::new(3, 5).expect("Construction failed");
        assert_eq!(secure_agg.threshold, 3);
        assert_eq!(secure_agg.total_clients, 5);

        // Should fail if threshold > total clients
        assert!(SecureAggregation::new(6, 5).is_err());
    }

    /// Regression: masks used to always be built for the hardcoded shapes
    /// `[100,50]`/`[50]`/`[50,20]`/`[20]`, unrelated to any caller's model.
    #[test]
    fn test_generate_masks_uses_the_callers_shapes_not_hardcoded_ones() {
        let secure_agg = SecureAggregation::new(2, 2).expect("Construction failed");
        let all_clients = vec!["alice".to_string(), "bob".to_string()];
        // Deliberately NOT the old hardcoded [100,50]/[50]/[50,20]/[20].
        let shapes = vec![vec![3], vec![2, 2], vec![5, 1, 2]];

        let masks = secure_agg
            .generate_masks("alice", &all_clients, 0, &shapes)
            .expect("generate_masks failed");

        assert_eq!(masks.len(), shapes.len());
        for (mask, expected_shape) in masks.iter().zip(shapes.iter()) {
            assert_eq!(&mask.shape(), expected_shape);
        }
    }

    #[test]
    fn test_generate_masks_is_deterministic_for_the_same_inputs() {
        let secure_agg = SecureAggregation::new(2, 3).expect("Construction failed");
        let all_clients = vec!["alice".to_string(), "bob".to_string(), "carol".to_string()];
        let shapes = vec![vec![4], vec![3, 2]];

        let first = secure_agg
            .generate_masks("bob", &all_clients, 7, &shapes)
            .expect("generate_masks failed");
        let second = secure_agg
            .generate_masks("bob", &all_clients, 7, &shapes)
            .expect("generate_masks failed");

        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(
                a.to_vec_f32().expect("read"),
                b.to_vec_f32().expect("read"),
                "the same client/round/shapes must derive the same masks every time"
            );
        }
    }

    #[test]
    fn test_generate_masks_rejects_a_client_id_missing_from_all_client_ids() {
        let secure_agg = SecureAggregation::new(2, 2).expect("Construction failed");
        let all_clients = vec!["alice".to_string(), "bob".to_string()];
        let shapes = vec![vec![2]];

        assert!(secure_agg.generate_masks("carol", &all_clients, 0, &shapes).is_err());
    }

    /// With exactly two clients, each client has exactly one pairwise
    /// partner, so its mask IS that single pairwise term (no summation
    /// across multiple pairs) -- the two clients' masks must be exact
    /// (bit-for-bit) negatives of each other.
    #[test]
    fn test_masks_cancel_exactly_between_two_clients() {
        let secure_agg = SecureAggregation::new(2, 2).expect("Construction failed");
        let all_clients = vec!["alice".to_string(), "bob".to_string()];
        let shapes = vec![vec![6], vec![3, 2]];

        let alice_masks = secure_agg
            .generate_masks("alice", &all_clients, 3, &shapes)
            .expect("generate_masks failed");
        let bob_masks = secure_agg
            .generate_masks("bob", &all_clients, 3, &shapes)
            .expect("generate_masks failed");

        for (alice_mask, bob_mask) in alice_masks.iter().zip(bob_masks.iter()) {
            let a = alice_mask.to_vec_f32().expect("read");
            let b = bob_mask.to_vec_f32().expect("read");
            assert_eq!(a.len(), b.len());
            for (av, bv) in a.iter().zip(b.iter()) {
                assert_eq!(
                    *av, -*bv,
                    "alice's and bob's pairwise mask values must be exact negatives"
                );
            }
        }
    }

    /// Regression: independently-seeded (non-pairwise) masks did not cancel
    /// -- their sum carried the masks' own mean as bias despite the doc
    /// comment's claim. Pairwise masks must make `secure_aggregate` recover
    /// the true average of the clients' real updates, to within
    /// floating-point rounding.
    #[test]
    fn test_secure_aggregate_of_pairwise_masked_updates_recovers_true_average() {
        let secure_agg = SecureAggregation::new(2, 4).expect("Construction failed");
        let client_ids: Vec<String> = ["client-0", "client-1", "client-2", "client-3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let shapes = vec![vec![4], vec![2, 3]];
        let round = 11;

        // Real per-client "true" updates: distinct values per client and
        // per parameter so a broken aggregation could not accidentally
        // match by symmetry.
        let true_updates: HashMap<String, Vec<Tensor>> = client_ids
            .iter()
            .enumerate()
            .map(|(client_index, id)| {
                let updates = shapes
                    .iter()
                    .map(|shape| {
                        let size = shape.iter().product::<usize>();
                        let data: Vec<f32> =
                            (0..size).map(|i| (client_index * 10 + i) as f32 * 0.1).collect();
                        Tensor::from_data(data, shape).expect("tensor must build in test")
                    })
                    .collect();
                (id.clone(), updates)
            })
            .collect();

        let masked_updates: HashMap<String, Vec<Tensor>> = client_ids
            .iter()
            .map(|id| {
                let masks = secure_agg
                    .generate_masks(id, &client_ids, round, &shapes)
                    .expect("generate_masks failed");
                let true_update = &true_updates[id];
                let masked: Vec<Tensor> = true_update
                    .iter()
                    .zip(masks.iter())
                    .map(|(update, mask)| update.add(mask).expect("tensor add failed in test"))
                    .collect();
                (id.clone(), masked)
            })
            .collect();

        let aggregated =
            secure_agg.secure_aggregate(masked_updates).expect("secure_aggregate failed");

        for (param_idx, shape) in shapes.iter().enumerate() {
            let size = shape.iter().product::<usize>();
            let mut expected_sum = vec![0.0f32; size];
            for (client_index, _) in client_ids.iter().enumerate() {
                for (slot, value) in expected_sum.iter_mut().enumerate() {
                    *value += (client_index * 10 + slot) as f32 * 0.1;
                }
            }
            let expected_average: Vec<f32> =
                expected_sum.iter().map(|v| v / client_ids.len() as f32).collect();

            let actual = aggregated[param_idx].to_vec_f32().expect("read");
            for (actual_value, expected_value) in actual.iter().zip(expected_average.iter()) {
                assert!(
                    (actual_value - expected_value).abs() < 1e-3,
                    "pairwise masks must cancel to within floating-point rounding: expected \
                     {expected_value}, got {actual_value}"
                );
            }
        }
    }
}
