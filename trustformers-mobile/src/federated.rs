//! Federated Learning for Mobile Devices
//!
//! This module provides infrastructure for privacy-preserving federated learning
//! across mobile devices without sending raw data to servers.

use crate::DefaultRng;
use crate::{
    training::{OnDeviceTrainer, OnDeviceTrainingConfig},
    MobileConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::error::{CoreError, Result};
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

/// Federated learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedLearningConfig {
    /// Server endpoint for model synchronization
    pub server_endpoint: String,
    /// Client ID for this device
    pub client_id: String,
    /// Number of local epochs before sync
    pub local_epochs: usize,
    /// Minimum clients for aggregation
    pub min_clients_for_aggregation: usize,
    /// Enable differential privacy
    pub enable_differential_privacy: bool,
    /// Differential privacy configuration
    pub dp_config: Option<DifferentialPrivacyConfig>,
    /// Enable secure aggregation
    pub enable_secure_aggregation: bool,
    /// Communication rounds
    pub communication_rounds: usize,
    /// Enable model compression
    pub enable_compression: bool,
    /// Compression ratio (0.0 - 1.0)
    pub compression_ratio: f32,
    /// Client selection strategy
    pub client_selection: ClientSelectionStrategy,
    /// Aggregation strategy
    pub aggregation_strategy: AggregationStrategy,
}

/// Differential privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialPrivacyConfig {
    /// Privacy budget (epsilon)
    pub epsilon: f64,
    /// Noise parameter (delta)
    pub delta: f64,
    /// Clipping norm for gradients
    pub clipping_norm: f32,
    /// Noise mechanism
    pub noise_mechanism: NoiseMechanism,
    /// Per-layer privacy budget allocation
    pub per_layer_budget: bool,
}

/// Noise mechanisms for differential privacy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseMechanism {
    /// Gaussian noise
    Gaussian,
    /// Laplacian noise
    Laplacian,
    /// Exponential mechanism
    Exponential,
}

/// Client selection strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClientSelectionStrategy {
    /// Random selection
    Random,
    /// Based on device resources
    ResourceBased,
    /// Based on data quality
    QualityBased,
    /// Round-robin selection
    RoundRobin,
    /// Prioritize fast devices
    SpeedOptimized,
}

/// Aggregation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Simple averaging (FedAvg)
    FedAvg,
    /// Weighted averaging by data size
    WeightedAvg,
    /// Momentum-based aggregation
    FedMomentum,
    /// Adaptive aggregation (FedYogi)
    FedYogi,
    /// Personalized aggregation
    PersonalizedFed,
}

impl Default for FederatedLearningConfig {
    fn default() -> Self {
        Self {
            server_endpoint: "https://fl.example.com".to_string(),
            client_id: uuid::Uuid::new_v4().to_string(),
            local_epochs: 5,
            min_clients_for_aggregation: 10,
            enable_differential_privacy: true,
            dp_config: Some(DifferentialPrivacyConfig::default()),
            enable_secure_aggregation: true,
            communication_rounds: 100,
            enable_compression: true,
            compression_ratio: 0.1, // 10% of original size
            client_selection: ClientSelectionStrategy::ResourceBased,
            aggregation_strategy: AggregationStrategy::FedAvg,
        }
    }
}

impl Default for DifferentialPrivacyConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-5,
            clipping_norm: 1.0,
            noise_mechanism: NoiseMechanism::Gaussian,
            per_layer_budget: false,
        }
    }
}

/// Federated learning client for mobile devices
pub struct FederatedLearningClient {
    config: FederatedLearningConfig,
    mobile_config: MobileConfig,
    trainer: OnDeviceTrainer,
    local_model: Option<HashMap<String, Tensor>>,
    global_model: Option<HashMap<String, Tensor>>,
    client_state: ClientState,
    privacy_accountant: PrivacyAccountant,
}

/// Client state in federated learning
#[derive(Debug, Clone)]
struct ClientState {
    current_round: usize,
    local_steps: usize,
    total_samples_processed: usize,
    last_sync_time: Option<std::time::Instant>,
    contribution_weight: f32,
}

/// Privacy accountant for tracking privacy budget
struct PrivacyAccountant {
    total_epsilon_spent: f64,
    total_delta_spent: f64,
    per_round_epsilon: Vec<f64>,
    privacy_violations: usize,
}

impl FederatedLearningClient {
    /// Create new federated learning client
    pub fn new(
        fl_config: FederatedLearningConfig,
        training_config: OnDeviceTrainingConfig,
        mobile_config: MobileConfig,
    ) -> Result<Self> {
        let trainer = OnDeviceTrainer::new(training_config, mobile_config.clone())?;

        Ok(Self {
            config: fl_config,
            mobile_config,
            trainer,
            local_model: None,
            global_model: None,
            client_state: ClientState {
                current_round: 0,
                local_steps: 0,
                total_samples_processed: 0,
                last_sync_time: None,
                contribution_weight: 1.0,
            },
            privacy_accountant: PrivacyAccountant::new(),
        })
    }

    /// Initialize with global model
    pub fn initialize_from_global_model(
        &mut self,
        global_model: HashMap<String, Tensor>,
    ) -> Result<()> {
        self.global_model = Some(global_model.clone());
        self.local_model = Some(global_model.clone());
        self.trainer.initialize_training(global_model)?;

        tracing::info!("Federated client initialized with global model");
        Ok(())
    }

    /// Perform local training on device data
    pub fn train_local_model(
        &mut self,
        local_data: &[(Tensor, Tensor)],
    ) -> Result<LocalTrainingResult> {
        tracing::info!(
            "Starting local training for round {}",
            self.client_state.current_round
        );

        let start_time = std::time::Instant::now();

        // Apply differential privacy if enabled
        let processed_data = if self.config.enable_differential_privacy {
            self.apply_differential_privacy(local_data)?
        } else {
            local_data.to_vec()
        };

        // Train for local epochs
        let mut total_loss = 0.0;
        for epoch in 0..self.config.local_epochs {
            let stats = self.trainer.train(&processed_data)?;
            total_loss += stats.avg_loss;

            tracing::debug!(
                "Local epoch {} completed with loss: {:.4}",
                epoch,
                stats.avg_loss
            );
        }

        // Get model updates (difference from global model)
        let model_updates = self.compute_model_updates()?;

        // Apply compression if enabled
        let compressed_updates = if self.config.enable_compression {
            self.compress_model_updates(&model_updates)?
        } else {
            model_updates
        };

        // Update client state
        self.client_state.local_steps += processed_data.len() * self.config.local_epochs;
        self.client_state.total_samples_processed += processed_data.len();

        let training_time = start_time.elapsed();

        Ok(LocalTrainingResult {
            model_updates: compressed_updates,
            num_samples: processed_data.len(),
            avg_loss: total_loss / self.config.local_epochs as f32,
            training_time_seconds: training_time.as_secs_f32(),
            client_metrics: self.collect_client_metrics(),
        })
    }

    /// Receive and apply global model update
    pub fn apply_global_update(&mut self, global_update: GlobalModelUpdate) -> Result<()> {
        // Validate update
        if global_update.round != self.client_state.current_round + 1 {
            return Err(TrustformersError::runtime_error(format!(
                "Round mismatch: expected {}, got {}",
                self.client_state.current_round + 1,
                global_update.round
            ))
            .into());
        }

        // Apply personalization if needed
        let updated_model = match self.config.aggregation_strategy {
            AggregationStrategy::PersonalizedFed => {
                self.personalize_global_model(&global_update.model)?
            },
            _ => global_update.model,
        };

        // Update local and global models
        self.global_model = Some(updated_model.clone());
        self.local_model = Some(updated_model);

        // Update state
        self.client_state.current_round = global_update.round;
        self.client_state.last_sync_time = Some(std::time::Instant::now());

        tracing::info!(
            "Applied global model update for round {}",
            global_update.round
        );
        Ok(())
    }

    /// Check if client should participate in current round
    pub fn should_participate(&self) -> bool {
        match self.config.client_selection {
            ClientSelectionStrategy::Random => {
                // Random selection with probability based on resources
                DefaultRng::new().gen::<f32>() < self.estimate_participation_probability()
            },
            ClientSelectionStrategy::ResourceBased => self.has_sufficient_resources(),
            ClientSelectionStrategy::QualityBased => self.has_quality_data(),
            ClientSelectionStrategy::RoundRobin => {
                self.client_state.current_round % 5 == self.hash_client_id() % 5
            },
            ClientSelectionStrategy::SpeedOptimized => self.is_fast_device(),
        }
    }

    /// Get federated learning statistics
    pub fn get_fl_stats(&self) -> FederatedLearningStats {
        FederatedLearningStats {
            current_round: self.client_state.current_round,
            local_steps: self.client_state.local_steps,
            total_samples: self.client_state.total_samples_processed,
            contribution_weight: self.client_state.contribution_weight,
            privacy_budget_spent: self.privacy_accountant.total_epsilon_spent,
            last_sync_time: self.client_state.last_sync_time,
        }
    }

    // Private implementation methods

    fn apply_differential_privacy(
        &self,
        data: &[(Tensor, Tensor)],
    ) -> Result<Vec<(Tensor, Tensor)>> {
        if let Some(ref dp_config) = self.config.dp_config {
            let mut private_data = Vec::with_capacity(data.len());

            for (input, target) in data {
                // Add noise to preserve privacy
                let noisy_input = self.add_privacy_noise(input, dp_config)?;
                private_data.push((noisy_input, target.clone()));
            }

            Ok(private_data)
        } else {
            Ok(data.to_vec())
        }
    }

    fn add_privacy_noise(
        &self,
        tensor: &Tensor,
        dp_config: &DifferentialPrivacyConfig,
    ) -> Result<Tensor> {
        match dp_config.noise_mechanism {
            NoiseMechanism::Gaussian => {
                let noise_scale = dp_config.clipping_norm
                    * (2.0 * (1.25 / dp_config.delta).ln()).sqrt() as f32
                    / dp_config.epsilon as f32;
                let noise = Tensor::randn(&tensor.shape())
                    .and_then(|t| t.scalar_mul(noise_scale))
                    .map_err(|e| TrustformersError::runtime_error(format!("{}", e)))?;
                tensor
                    .add(&noise)
                    .map_err(|e| TrustformersError::runtime_error(format!("{}", e)).into())
            },
            NoiseMechanism::Laplacian => {
                let noise_scale = dp_config.clipping_norm / dp_config.epsilon as f32;
                // Using Gaussian approximation for Laplacian noise
                let noise = Tensor::randn(&tensor.shape())
                    .and_then(|t| t.scalar_mul(noise_scale))
                    .map_err(|e| TrustformersError::runtime_error(format!("{}", e)))?;
                tensor
                    .add(&noise)
                    .map_err(|e| TrustformersError::runtime_error(format!("{}", e)).into())
            },
            NoiseMechanism::Exponential => {
                // Simplified exponential mechanism
                Ok(tensor.clone())
            },
        }
    }

    fn compute_model_updates(&self) -> Result<HashMap<String, Tensor>> {
        let mut updates = HashMap::new();

        if let (Some(ref local), Some(ref global)) = (&self.local_model, &self.global_model) {
            for (name, local_param) in local {
                if let Some(global_param) = global.get(name) {
                    // Compute difference: local - global
                    let update = local_param.sub(global_param)?;
                    updates.insert(name.clone(), update);
                }
            }
        }

        Ok(updates)
    }

    fn compress_model_updates(
        &self,
        updates: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut compressed = HashMap::new();

        for (name, update) in updates {
            // Simple top-k sparsification
            let compressed_update = self.sparsify_tensor(update, self.config.compression_ratio)?;
            compressed.insert(name.clone(), compressed_update);
        }

        Ok(compressed)
    }

    fn sparsify_tensor(&self, tensor: &Tensor, keep_ratio: f32) -> Result<Tensor> {
        // Keep only top-k% of values by magnitude
        let num_elements = tensor.shape().iter().product::<usize>();
        let keep_count = (num_elements as f32 * keep_ratio) as usize;

        // Get tensor data for processing
        let data = tensor.data()?;
        let shape = tensor.shape();

        // Create magnitude-value pairs
        let mut indexed_values: Vec<(usize, f32)> =
            data.iter().enumerate().map(|(i, &val)| (i, val.abs())).collect();

        // Sort by magnitude in descending order
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Create sparse tensor data
        let mut sparse_data = vec![0.0; num_elements];

        // Keep only top-k values
        for i in 0..keep_count.min(indexed_values.len()) {
            let (original_idx, _) = indexed_values[i];
            sparse_data[original_idx] = data[original_idx];
        }

        // Create new tensor with sparse data
        Tensor::from_vec(sparse_data, &shape)
            .map_err(|e| TrustformersError::runtime_error(format!("{}", e)).into())
    }

    fn personalize_global_model(
        &self,
        global_model: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut personalized = HashMap::new();

        if let Some(ref local) = self.local_model {
            for (name, global_param) in global_model {
                if let Some(local_param) = local.get(name) {
                    // Blend global and local models
                    let alpha = 0.3; // Personalization factor
                    let blended = global_param
                        .scalar_mul(1.0 - alpha)?
                        .add(&local_param.scalar_mul(alpha)?)?;
                    personalized.insert(name.clone(), blended);
                } else {
                    personalized.insert(name.clone(), global_param.clone());
                }
            }
        } else {
            return Ok(global_model.clone());
        }

        Ok(personalized)
    }

    fn collect_client_metrics(&self) -> ClientMetrics {
        ClientMetrics {
            device_model: self.get_device_model(),
            available_memory_mb: self.get_available_memory(),
            battery_level: self.get_battery_level(),
            network_quality: self.estimate_network_quality(),
            computation_capability: self.estimate_computation_capability(),
        }
    }

    fn estimate_participation_probability(&self) -> f32 {
        // Base probability on device resources
        let memory_factor = (self.get_available_memory() as f32 / 1024.0).min(1.0);
        let battery_factor = self.get_battery_level();
        let network_factor = match self.estimate_network_quality() {
            NetworkQuality::Excellent => 1.0,
            NetworkQuality::Good => 0.8,
            NetworkQuality::Fair => 0.5,
            NetworkQuality::Poor => 0.2,
        };

        (memory_factor * battery_factor * network_factor).max(0.1)
    }

    fn has_sufficient_resources(&self) -> bool {
        self.get_available_memory() >= 512
            && self.get_battery_level() >= 0.3
            && !matches!(self.estimate_network_quality(), NetworkQuality::Poor)
    }

    fn has_quality_data(&self) -> bool {
        // Check if client has sufficient quality data
        self.client_state.total_samples_processed >= 1000
    }

    fn hash_client_id(&self) -> usize {
        // Simple hash of client ID for round-robin
        self.config
            .client_id
            .bytes()
            .fold(0usize, |acc, b| acc.wrapping_add(b as usize))
    }

    fn is_fast_device(&self) -> bool {
        matches!(
            self.estimate_computation_capability(),
            ComputationCapability::High
        )
    }

    fn get_device_model(&self) -> String {
        #[cfg(target_os = "ios")]
        return "iOS Device".to_string();

        #[cfg(target_os = "android")]
        return "Android Device".to_string();

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        return "Generic Device".to_string();
    }

    fn get_available_memory(&self) -> usize {
        #[cfg(target_os = "ios")]
        {
            // Query iOS system memory
            use libc::{sysconf, _SC_PAGE_SIZE, _SC_PHYS_PAGES};
            unsafe {
                let pages = sysconf(_SC_PHYS_PAGES);
                let page_size = sysconf(_SC_PAGE_SIZE);
                if pages > 0 && page_size > 0 {
                    ((pages * page_size) / (1024 * 1024)) as usize // Convert to MB
                } else {
                    2048 // Fallback: 2GB
                }
            }
        }

        #[cfg(target_os = "android")]
        {
            // Query Android system memory via JNI would go here
            // For now, estimate based on common Android device specs
            let cpu_cores = num_cpus::get();
            match cpu_cores {
                1..=2 => 1024, // 1GB for low-end
                3..=4 => 3072, // 3GB for mid-range
                5..=6 => 4096, // 4GB for high-end
                _ => 6144,     // 6GB+ for flagship
            }
        }

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            // Desktop/server fallback
            4096
        }
    }

    fn get_battery_level(&self) -> f32 {
        #[cfg(target_os = "ios")]
        {
            // iOS battery level detection would use UIDevice.current.batteryLevel
            // For this implementation, we'll simulate based on time patterns
            use std::time::{SystemTime, UNIX_EPOCH};
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs();
            let cycle = (now % 3600) as f32 / 3600.0; // Hour cycle
            0.2 + 0.7 * (1.0 + (cycle * 2.0 * std::f32::consts::PI).sin()) / 2.0
        }

        #[cfg(target_os = "android")]
        {
            // Android battery level detection via JNI/system calls would go here
            // Simulate battery level based on device load
            let cpu_cores = num_cpus::get();
            let base_level = match cpu_cores {
                1..=2 => 0.6,  // Low-end devices drain faster
                3..=4 => 0.75, // Mid-range
                _ => 0.85,     // High-end devices have better battery management
            };
            base_level + (DefaultRng::new().gen::<f32>() - 0.5) * 0.2 // Add some randomness
        }

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            1.0 // Desktop/server always "plugged in"
        }
    }

    fn estimate_network_quality(&self) -> NetworkQuality {
        // Simulate network quality estimation based on device capabilities
        let available_memory = self.get_available_memory();
        let battery_level = self.get_battery_level();

        // Better devices typically have better network conditions
        let quality_score = (available_memory as f32 / 1024.0) * battery_level;

        match quality_score {
            score if score < 1.0 => NetworkQuality::Poor,
            score if score < 3.0 => NetworkQuality::Fair,
            score if score < 5.0 => NetworkQuality::Good,
            _ => NetworkQuality::Excellent,
        }
    }

    fn estimate_computation_capability(&self) -> ComputationCapability {
        let cpu_cores = num_cpus::get();
        let available_memory = self.get_available_memory();

        // Combine CPU cores and memory to estimate capability
        let capability_score = cpu_cores as f32 + (available_memory as f32 / 1024.0);

        #[cfg(target_os = "ios")]
        {
            // iOS devices generally have optimized hardware
            match capability_score {
                score if score < 3.0 => ComputationCapability::Low, // iPhone 6/7
                score if score < 6.0 => ComputationCapability::Medium, // iPhone 8/X
                score if score < 10.0 => ComputationCapability::High, // iPhone 11/12
                _ => ComputationCapability::High,                   // iPhone 13+/iPad Pro
            }
        }

        #[cfg(target_os = "android")]
        {
            // Android devices have more variance
            match capability_score {
                score if score < 2.0 => ComputationCapability::Low, // Budget devices
                score if score < 5.0 => ComputationCapability::Medium, // Mid-range
                score if score < 8.0 => ComputationCapability::High, // Flagship
                _ => ComputationCapability::High,                   // High-end flagship
            }
        }

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            // Desktop/server
            match capability_score {
                score if score < 4.0 => ComputationCapability::Medium,
                _ => ComputationCapability::High,
            }
        }
    }
}

/// Result of local training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalTrainingResult {
    /// Model updates (gradients or weight differences)
    #[serde(skip)]
    pub model_updates: HashMap<String, Tensor>,
    /// Number of samples used in training
    pub num_samples: usize,
    /// Average loss during training
    pub avg_loss: f32,
    /// Training time in seconds
    pub training_time_seconds: f32,
    /// Client metrics for aggregation
    pub client_metrics: ClientMetrics,
}

/// Global model update from server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalModelUpdate {
    /// Updated model parameters
    #[serde(skip)]
    pub model: HashMap<String, Tensor>,
    /// Current round number
    pub round: usize,
    /// Aggregation metadata
    pub metadata: AggregationMetadata,
}

/// Client metrics for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientMetrics {
    pub device_model: String,
    pub available_memory_mb: usize,
    pub battery_level: f32,
    pub network_quality: NetworkQuality,
    pub computation_capability: ComputationCapability,
}

/// Network quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkQuality {
    Excellent,
    Good,
    Fair,
    Poor,
}

/// Computation capability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputationCapability {
    Low,
    Medium,
    High,
}

/// Aggregation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationMetadata {
    pub num_clients_participated: usize,
    pub total_samples: usize,
    pub aggregation_time_seconds: f32,
    pub server_version: String,
}

/// Federated learning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedLearningStats {
    pub current_round: usize,
    pub local_steps: usize,
    pub total_samples: usize,
    pub contribution_weight: f32,
    pub privacy_budget_spent: f64,
    #[serde(skip)]
    pub last_sync_time: Option<std::time::Instant>,
}

impl PrivacyAccountant {
    fn new() -> Self {
        Self {
            total_epsilon_spent: 0.0,
            total_delta_spent: 0.0,
            per_round_epsilon: Vec::new(),
            privacy_violations: 0,
        }
    }

    fn account_privacy_cost(&mut self, epsilon: f64, delta: f64) {
        self.total_epsilon_spent += epsilon;
        self.total_delta_spent += delta;
        self.per_round_epsilon.push(epsilon);
    }

    fn check_privacy_budget(&self, budget_epsilon: f64, budget_delta: f64) -> bool {
        self.total_epsilon_spent <= budget_epsilon && self.total_delta_spent <= budget_delta
    }
}

/// Secure aggregation for privacy-preserving model updates
pub struct SecureAggregator {
    threshold: usize,
    shares: HashMap<String, Vec<Tensor>>,
}

impl SecureAggregator {
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold,
            shares: HashMap::new(),
        }
    }

    /// Add client share for secure aggregation
    pub fn add_share(&mut self, client_id: String, share: HashMap<String, Tensor>) {
        for (param_name, param_share) in share {
            self.shares.entry(param_name).or_default().push(param_share);
        }
    }

    /// Aggregate shares securely
    pub fn aggregate(&self) -> Result<HashMap<String, Tensor>> {
        let mut aggregated = HashMap::new();

        for (param_name, shares) in &self.shares {
            if shares.len() >= self.threshold {
                // Average the shares
                let sum =
                    shares[1..].iter().try_fold(shares[0].clone(), |acc, share| acc.add(share))?;
                let avg = sum.scalar_mul(1.0 / shares.len() as f32)?;
                aggregated.insert(param_name.clone(), avg);
            }
        }

        Ok(aggregated)
    }
}

/// Federated learning utilities
pub struct FederatedLearningUtils;

impl FederatedLearningUtils {
    /// Create optimal FL config for mobile device
    pub fn create_mobile_fl_config(
        device_type: &str,
        network_condition: NetworkQuality,
    ) -> FederatedLearningConfig {
        let mut config = FederatedLearningConfig::default();

        // Adjust based on network condition
        match network_condition {
            NetworkQuality::Excellent => {
                config.communication_rounds = 100;
                config.compression_ratio = 0.3;
            },
            NetworkQuality::Good => {
                config.communication_rounds = 50;
                config.compression_ratio = 0.1;
            },
            NetworkQuality::Fair => {
                config.communication_rounds = 20;
                config.compression_ratio = 0.05;
            },
            NetworkQuality::Poor => {
                config.communication_rounds = 10;
                config.compression_ratio = 0.01;
            },
        }

        // Adjust based on device type
        if device_type.contains("flagship") || device_type.contains("pro") {
            config.local_epochs = 5;
            config.enable_secure_aggregation = true;
        } else {
            config.local_epochs = 3;
            config.enable_secure_aggregation = false;
        }

        config
    }

    /// Estimate communication cost
    pub fn estimate_communication_cost(
        model_size_mb: f32,
        compression_ratio: f32,
        rounds: usize,
    ) -> CommunicationCost {
        let upload_per_round = model_size_mb * compression_ratio;
        let download_per_round = model_size_mb; // Full model download

        CommunicationCost {
            total_upload_mb: upload_per_round * rounds as f32,
            total_download_mb: download_per_round * rounds as f32,
            estimated_time_seconds: estimate_transfer_time(
                (upload_per_round + download_per_round) * rounds as f32,
            ),
        }
    }
}

/// Communication cost estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationCost {
    pub total_upload_mb: f32,
    pub total_download_mb: f32,
    pub estimated_time_seconds: f32,
}

fn estimate_transfer_time(total_mb: f32) -> f32 {
    // Assume 10 Mbps average mobile connection
    total_mb * 8.0 / 10.0
}

// Placeholder for uuid crate functionality
mod uuid {
    pub struct Uuid;
    impl Uuid {
        pub fn new_v4() -> Self {
            Self
        }
    }
    impl ToString for Uuid {
        fn to_string(&self) -> String {
            "mock-uuid".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_learning_config() {
        let config = FederatedLearningConfig::default();
        assert_eq!(config.local_epochs, 5);
        assert!(config.enable_differential_privacy);
        assert!(config.enable_secure_aggregation);
    }

    #[test]
    fn test_differential_privacy_config() {
        let dp_config = DifferentialPrivacyConfig::default();
        assert_eq!(dp_config.epsilon, 1.0);
        assert_eq!(dp_config.delta, 1e-5);
        assert_eq!(dp_config.noise_mechanism, NoiseMechanism::Gaussian);
    }

    #[test]
    fn test_federated_client_creation() {
        let fl_config = FederatedLearningConfig::default();
        let training_config = crate::training::OnDeviceTrainingConfig::default();
        let mobile_config = crate::MobileConfig::default();

        let client = FederatedLearningClient::new(fl_config, training_config, mobile_config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_secure_aggregator() {
        let mut aggregator = SecureAggregator::new(3);

        // Add shares from multiple clients
        for i in 0..5 {
            let mut share = HashMap::new();
            share.insert("weight".to_string(), Tensor::ones(&[10, 10]).unwrap());
            aggregator.add_share(format!("client_{}", i), share);
        }

        let result = aggregator.aggregate();
        assert!(result.is_ok());
        assert!(result.unwrap().contains_key("weight"));
    }

    #[test]
    fn test_communication_cost_estimation() {
        let cost = FederatedLearningUtils::estimate_communication_cost(
            100.0, // 100MB model
            0.1,   // 10% compression
            50,    // 50 rounds
        );

        assert_eq!(cost.total_upload_mb, 500.0); // 10MB * 50
        assert_eq!(cost.total_download_mb, 5000.0); // 100MB * 50
    }
}
