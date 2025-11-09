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

use anyhow::{anyhow, Result};
use scirs2_core::random::StdRng; // Explicit import for type clarity
use scirs2_core::random::*; // SciRS2 Integration Policy - Replaces rand
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
                    let j = self.rng.gen_range(i..indices.len());
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
                clients_with_capacity.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
                clients_with_capacity[..num_clients].iter().map(|(id, _)| id.clone()).collect()
            },
            ClientSelectionStrategy::CommunicationQuality => {
                let mut clients_with_quality: Vec<_> = available
                    .iter()
                    .map(|c| (c.client_id.clone(), c.communication_quality))
                    .collect();
                clients_with_quality.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
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
        let param_count = client_updates.values().next().unwrap().len();
        let mut aggregated = Vec::with_capacity(param_count);

        for i in 0..param_count {
            // Get shape from first client's parameter
            let first_param = &client_updates.values().next().unwrap()[i];
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
    #[allow(dead_code)]
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

    /// Generate random masks for secure aggregation.
    /// In practice, this would use cryptographic protocols.
    pub fn generate_masks(&self, client_id: &str, round: usize) -> Result<Vec<Tensor>> {
        // This is a simplified implementation
        // Real secure aggregation uses secret sharing and cryptographic techniques
        let mut rng = StdRng::from_seed({
            let mut seed = [0u8; 32];
            let client_hash = format!("{}-{}", client_id, round);
            let bytes = client_hash.as_bytes();
            for (i, &byte) in bytes.iter().enumerate().take(32) {
                seed[i] = byte;
            }
            seed
        });

        // Generate cryptographic masks for secure aggregation
        // Each mask is a random tensor that will be used to blind the client's update
        let mut masks = Vec::new();

        // Generate masks based on client's expected parameter shapes
        // In practice, these shapes would be communicated during federated setup
        let parameter_shapes = vec![
            vec![100, 50], // Example: First layer weights
            vec![50],      // Example: First layer bias
            vec![50, 20],  // Example: Second layer weights
            vec![20],      // Example: Second layer bias
        ];

        for shape in parameter_shapes {
            // Generate random mask with same shape as parameter
            let mask_size = shape.iter().product::<usize>();
            let mut mask_data: Vec<f32> = Vec::with_capacity(mask_size);

            for _ in 0..mask_size {
                // Generate random float in range [-1.0, 1.0] for better numerical stability
                mask_data.push(rng.gen_range(-1.0..1.0));
            }

            let mask = Tensor::from_data(mask_data, &shape)?;
            masks.push(mask);
        }

        Ok(masks)
    }

    /// Aggregate masked updates securely.
    pub fn secure_aggregate(
        &self,
        masked_updates: HashMap<String, Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        if masked_updates.len() < self.threshold {
            return Err(anyhow!("Not enough clients for secure aggregation"));
        }

        // In a real implementation, this would:
        // 1. Collect masked updates from clients
        // 2. Aggregate the masks
        // 3. Remove the aggregate mask to reveal the sum
        // 4. Compute the average

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
            let mut aggregated_param = Tensor::zeros(&expected_shape.unwrap())?;
            for param_update in parameter_updates {
                aggregated_param = aggregated_param.add(param_update)?;
            }

            // Average the aggregated parameter
            // In secure aggregation, masks cancel out during summation
            // so we get the true average without revealing individual updates
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
        let selected = fedavg.select_clients(&clients, ClientSelectionStrategy::Random).unwrap();
        assert!(!selected.is_empty());

        // Test data size selection
        let selected = fedavg.select_clients(&clients, ClientSelectionStrategy::DataSize).unwrap();
        assert!(!selected.is_empty());
    }

    #[test]
    fn test_secure_aggregation_creation() {
        let secure_agg = SecureAggregation::new(3, 5).unwrap();
        assert_eq!(secure_agg.threshold, 3);
        assert_eq!(secure_agg.total_clients, 5);

        // Should fail if threshold > total clients
        assert!(SecureAggregation::new(6, 5).is_err());
    }
}
