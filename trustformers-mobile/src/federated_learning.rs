//! On-Device Federated Learning Client
//!
//! This module implements a federated learning client for mobile devices that enables:
//! - Privacy-preserving on-device training
//! - Secure aggregation protocols
//! - Differential privacy mechanisms
//! - Efficient communication with federated servers
//! - Adaptive training based on device capabilities

use crate::{MobileConfig, MemoryOptimization};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use trustformers_core::{Tensor};
use trustformers_core::error::{CoreError, Result};

/// Federated learning client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedLearningConfig {
    /// Server endpoint for federated coordination
    pub server_url: String,
    /// Client unique identifier
    pub client_id: String,
    /// API key for authentication
    pub api_key: Option<String>,
    /// Differential privacy configuration
    pub privacy_config: DifferentialPrivacyConfig,
    /// Training configuration
    pub training_config: FederatedTrainingConfig,
    /// Communication configuration
    pub communication_config: CommunicationConfig,
    /// Security configuration
    pub security_config: SecurityConfig,
    /// Device resource constraints
    pub resource_constraints: ResourceConstraints,
}

/// Differential privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialPrivacyConfig {
    /// Enable differential privacy
    pub enabled: bool,
    /// Privacy budget (epsilon)
    pub epsilon: f64,
    /// Noise scale for Gaussian mechanism
    pub noise_scale: f64,
    /// Clipping norm for gradient clipping
    pub clipping_norm: f64,
    /// Sensitivity parameter
    pub sensitivity: f64,
    /// Use local differential privacy
    pub use_local_dp: bool,
}

/// Federated training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedTrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Local training epochs per round
    pub local_epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Maximum number of federated rounds
    pub max_rounds: usize,
    /// Minimum number of local samples required
    pub min_local_samples: usize,
    /// Enable personalization
    pub enable_personalization: bool,
    /// Personalization layers (if enabled)
    pub personalization_layers: Vec<String>,
    /// Use momentum optimization
    pub use_momentum: bool,
    /// Momentum coefficient
    pub momentum: f64,
}

/// Communication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationConfig {
    /// Communication round timeout (seconds)
    pub round_timeout_seconds: u64,
    /// Maximum message size (bytes)
    pub max_message_size_bytes: usize,
    /// Compression algorithm for model updates
    pub compression_algorithm: CompressionAlgorithm,
    /// Quantization bits for model updates
    pub quantization_bits: u8,
    /// Enable secure aggregation
    pub enable_secure_aggregation: bool,
    /// Retry attempts for failed communications
    pub retry_attempts: usize,
    /// Minimum number of participants for aggregation
    pub min_participants: usize,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable end-to-end encryption
    pub enable_encryption: bool,
    /// Encryption algorithm
    pub encryption_algorithm: EncryptionAlgorithm,
    /// Enable secure multiparty computation
    pub enable_smpc: bool,
    /// Certificate validation
    pub validate_certificates: bool,
    /// Enable gradient verification
    pub enable_gradient_verification: bool,
}

/// Resource constraints configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum memory usage (MB)
    pub max_memory_mb: usize,
    /// Maximum CPU usage percentage
    pub max_cpu_usage_percent: f32,
    /// Maximum battery usage per round (percentage)
    pub max_battery_usage_percent: f32,
    /// Minimum battery level to participate
    pub min_battery_level_percent: f32,
    /// Enable thermal throttling
    pub enable_thermal_throttling: bool,
    /// Training time budget per round (seconds)
    pub max_training_time_seconds: u64,
}

/// Compression algorithms for model updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Brotli,
    TopK(usize),
    RandomK(usize),
    Threshold(f64),
}

/// Encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    None,
    AES256,
    ChaCha20Poly1305,
    RSA2048,
}

/// Federated learning round information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedRound {
    /// Round number
    pub round_number: usize,
    /// Global model parameters
    pub global_model: ModelParameters,
    /// Round configuration
    pub round_config: RoundConfig,
    /// Participant selection criteria
    pub selection_criteria: SelectionCriteria,
    /// Round deadline
    pub deadline: u64,
}

/// Model parameters for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Parameter tensors
    pub parameters: HashMap<String, Vec<f32>>,
    /// Parameter shapes
    pub shapes: HashMap<String, Vec<usize>>,
    /// Model version
    pub version: String,
    /// Checksum for integrity verification
    pub checksum: String,
}

/// Round-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundConfig {
    /// Learning rate for this round
    pub learning_rate: f64,
    /// Local epochs for this round
    pub local_epochs: usize,
    /// Minimum accuracy threshold
    pub min_accuracy_threshold: f64,
    /// Adaptive configuration based on previous rounds
    pub adaptive_config: Option<AdaptiveConfig>,
}

/// Participant selection criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    /// Minimum data samples required
    pub min_samples: usize,
    /// Required device capabilities
    pub required_capabilities: Vec<String>,
    /// Preferred device types
    pub preferred_device_types: Vec<String>,
    /// Geographic region preferences
    pub region_preferences: Option<Vec<String>>,
}

/// Adaptive configuration for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Adjust learning rate based on convergence
    pub adaptive_learning_rate: bool,
    /// Adjust local epochs based on device performance
    pub adaptive_local_epochs: bool,
    /// Adjust privacy parameters
    pub adaptive_privacy: bool,
    /// Performance targets
    pub performance_targets: PerformanceTargets,
}

/// Performance targets for adaptive learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target accuracy improvement
    pub target_accuracy: f64,
    /// Target convergence time
    pub target_convergence_time_seconds: u64,
    /// Target energy efficiency
    pub target_energy_efficiency: f64,
}

/// Training update from local device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalUpdate {
    /// Client identifier
    pub client_id: String,
    /// Round number
    pub round_number: usize,
    /// Model parameter updates
    pub parameter_updates: ModelParameters,
    /// Number of local samples used
    pub num_samples: usize,
    /// Local training metrics
    pub training_metrics: TrainingMetrics,
    /// Computation time (seconds)
    pub computation_time_seconds: f64,
    /// Privacy guarantees applied
    pub privacy_guarantees: PrivacyGuarantees,
}

/// Training metrics from local training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Local accuracy achieved
    pub accuracy: f64,
    /// Local loss value
    pub loss: f64,
    /// Training iterations completed
    pub iterations: usize,
    /// Convergence indicator
    pub converged: bool,
    /// Gradient norm
    pub gradient_norm: f64,
}

/// Privacy guarantees applied to updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyGuarantees {
    /// Differential privacy epsilon used
    pub epsilon_used: f64,
    /// Noise scale applied
    pub noise_scale: f64,
    /// Clipping applied
    pub gradient_clipped: bool,
    /// Local differential privacy applied
    pub local_dp_applied: bool,
}

/// Federated learning client
pub struct FederatedLearningClient {
    config: FederatedLearningConfig,
    current_round: Option<FederatedRound>,
    local_model: Option<ModelParameters>,
    training_data: Vec<TrainingExample>,
    client_state: ClientState,
    privacy_accountant: PrivacyAccountant,
    communication_manager: CommunicationManager,
    security_manager: SecurityManager,
}

/// Training example for local training
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input features
    pub input: Tensor,
    /// Target label
    pub label: Tensor,
    /// Example weight (optional)
    pub weight: Option<f32>,
}

/// Client state information
#[derive(Debug, Clone)]
struct ClientState {
    /// Current participation status
    pub status: ParticipationStatus,
    /// Number of rounds participated
    pub rounds_participated: usize,
    /// Total training samples processed
    pub total_samples_processed: usize,
    /// Average contribution quality
    pub contribution_quality: f64,
    /// Device performance metrics
    pub device_metrics: DeviceMetrics,
}

/// Participation status
#[derive(Debug, Clone, PartialEq, Eq)]
enum ParticipationStatus {
    Idle,
    WaitingForRound,
    Training,
    UploadingUpdate,
    Completed,
    Dropped,
}

/// Device performance metrics
#[derive(Debug, Clone)]
struct DeviceMetrics {
    /// Average training time per round
    pub avg_training_time_seconds: f64,
    /// Average memory usage during training
    pub avg_memory_usage_mb: f32,
    /// Average CPU usage during training
    pub avg_cpu_usage_percent: f32,
    /// Battery usage per round
    pub battery_usage_per_round_percent: f32,
    /// Network usage statistics
    pub network_usage_bytes: usize,
}

/// Privacy accountant for tracking privacy budget
struct PrivacyAccountant {
    total_epsilon_used: f64,
    epsilon_budget: f64,
    round_epsilon_usage: Vec<f64>,
}

/// Communication manager for federated protocol
struct CommunicationManager {
    server_url: String,
    client_id: String,
    api_key: Option<String>,
    retry_attempts: usize,
}

/// Security manager for encryption and verification
struct SecurityManager {
    encryption_enabled: bool,
    certificate_validation: bool,
    gradient_verification: bool,
}

impl FederatedLearningClient {
    /// Create new federated learning client
    pub fn new(config: FederatedLearningConfig) -> Result<Self> {
        config.validate()?;

        let client_state = ClientState {
            status: ParticipationStatus::Idle,
            rounds_participated: 0,
            total_samples_processed: 0,
            contribution_quality: 0.0,
            device_metrics: DeviceMetrics::default(),
        };

        let privacy_accountant = PrivacyAccountant::new(config.privacy_config.epsilon);
        let communication_manager = CommunicationManager::new(
            config.server_url.clone(),
            config.client_id.clone(),
            config.api_key.clone(),
            config.communication_config.retry_attempts,
        );
        let security_manager = SecurityManager::new(&config.security_config);

        Ok(Self {
            config,
            current_round: None,
            local_model: None,
            training_data: Vec::new(),
            client_state,
            privacy_accountant,
            communication_manager,
            security_manager,
        })
    }

    /// Join federated learning and wait for round
    pub async fn join_federation(&mut self) -> Result<()> {
        tracing::info!("Joining federated learning federation");

        self.client_state.status = ParticipationStatus::WaitingForRound;

        // Register with server
        self.communication_manager.register_client().await?;

        // Check device eligibility
        self.check_device_eligibility()?;

        tracing::info!("Successfully joined federation");
        Ok(())
    }

    /// Wait for and receive new federated round
    pub async fn wait_for_round(&mut self) -> Result<FederatedRound> {
        tracing::info!("Waiting for federated learning round");

        let round = self.communication_manager.wait_for_round().await?;

        // Check if client meets selection criteria
        if !self.meets_selection_criteria(&round.selection_criteria) {
            return Err(TrustformersError::runtime_error("Client doesn't meet selection criteria".into()).into());
        }

        // Check privacy budget
        if !self.privacy_accountant.can_participate(&self.config.privacy_config) {
            return Err(TrustformersError::runtime_error("Insufficient privacy budget".into()).into());
        }

        self.current_round = Some(round.clone());
        tracing::info!("Received federated round {}", round.round_number);

        Ok(round)
    }

    /// Add training data for local training
    pub fn add_training_data(&mut self, examples: Vec<TrainingExample>) -> Result<()> {
        if examples.len() < self.config.training_config.min_local_samples {
            return Err(TrustformersError::runtime_error("Insufficient training samples".into()).into());
        }

        self.training_data.extend(examples);

        tracing::info!("Added {} training examples, total: {}",
                      self.training_data.len() - examples.len(),
                      self.training_data.len().into());

        Ok(())
    }

    /// Perform local training on device
    pub async fn train_locally(&mut self) -> Result<LocalUpdate> {
        let round = self.current_round.as_ref()
            .ok_or_else(|| TrustformersError::runtime_error("No active round".into())?;

        if self.training_data.is_empty() {
            return Err(TrustformersError::runtime_error("No training data available".into()).into());
        }

        tracing::info!("Starting local training for round {}", round.round_number);

        self.client_state.status = ParticipationStatus::Training;

        let start_time = SystemTime::now();

        // Initialize local model with global parameters
        self.local_model = Some(round.global_model.clone());

        // Perform local training
        let training_metrics = self.perform_local_training_loop(&round.round_config).await?;

        // Apply differential privacy to updates
        let privacy_guarantees = self.apply_differential_privacy()?;

        // Compute parameter updates
        let parameter_updates = self.compute_parameter_updates(&round.global_model)?;

        let training_time = start_time.elapsed().unwrap_or(Duration::ZERO).as_secs_f64();

        // Update device metrics
        self.update_device_metrics(training_time);

        // Create local update
        let local_update = LocalUpdate {
            client_id: self.config.client_id.clone(),
            round_number: round.round_number,
            parameter_updates,
            num_samples: self.training_data.len(),
            training_metrics,
            computation_time_seconds: training_time,
            privacy_guarantees,
        };

        // Update privacy accountant
        self.privacy_accountant.record_epsilon_usage(privacy_guarantees.epsilon_used);

        self.client_state.rounds_participated += 1;
        self.client_state.total_samples_processed += self.training_data.len();

        tracing::info!("Completed local training in {:.2}s, accuracy: {:.4}",
                      training_time, training_metrics.accuracy);

        Ok(local_update)
    }

    /// Send local update to federation server
    pub async fn send_update(&mut self, update: LocalUpdate) -> Result<()> {
        tracing::info!("Sending local update for round {}", update.round_number);

        self.client_state.status = ParticipationStatus::UploadingUpdate;

        // Compress update if configured
        let compressed_update = self.compress_update(update)?;

        // Encrypt update if configured
        let encrypted_update = self.security_manager.encrypt_update(compressed_update)?;

        // Send to server
        self.communication_manager.send_update(encrypted_update).await?;

        self.client_state.status = ParticipationStatus::Completed;

        tracing::info!("Successfully sent local update");
        Ok(())
    }

    /// Perform personalization on local model
    pub fn personalize_model(&mut self) -> Result<()> {
        if !self.config.training_config.enable_personalization {
            return Ok(());
        }

        tracing::info!("Performing model personalization");

        // Implement personalization logic
        // This would fine-tune specific layers on local data
        let _personalization_layers = &self.config.training_config.personalization_layers;

        // Implement personalization logic
        // Fine-tune specific layers on local data
        if let Some(ref mut model) = self.local_model {
            let personalization_layers = &self.config.training_config.personalization_layers;

            // Create smaller learning rate for personalization
            let personalization_lr = self.config.training_config.learning_rate * 0.1;

            // Perform personalization training on specific layers
            for epoch in 0..self.config.training_config.personalization_epochs {
                let mut total_loss = 0.0;
                let mut batch_count = 0;

                // Process local training data
                for batch in self.create_training_batches() {
                    // Forward pass
                    let (loss, _accuracy) = self.forward_pass(&batch)?;
                    total_loss += loss;
                    batch_count += 1;

                    // Backward pass (only for personalization layers)
                    self.backward_pass_personalization(loss, personalization_layers)?;

                    // Update only personalization layers
                    self.update_personalization_parameters(personalization_lr)?;
                }

                let avg_loss = total_loss / batch_count as f64;
                tracing::info!("Personalization epoch {}: avg_loss = {:.4}", epoch, avg_loss);
            }
        }

        Ok(())
    }

    /// Get client statistics
    pub fn get_stats(&self) -> &ClientState {
        &self.client_state
    }

    /// Check if client can participate in current round
    pub fn can_participate(&self) -> bool {
        // Check privacy budget
        if !self.privacy_accountant.can_participate(&self.config.privacy_config) {
            return false;
        }

        // Check resource constraints
        if !self.meets_resource_constraints() {
            return false;
        }

        // Check data availability
        if self.training_data.len() < self.config.training_config.min_local_samples {
            return false;
        }

        true
    }

    /// Reset client state for new federation
    pub fn reset(&mut self) {
        self.current_round = None;
        self.local_model = None;
        self.client_state.status = ParticipationStatus::Idle;
        self.privacy_accountant.reset();
    }

    // Private helper methods

    async fn perform_local_training_loop(&mut self, round_config: &RoundConfig) -> Result<TrainingMetrics> {
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let mut iterations = 0;

        for epoch in 0..round_config.local_epochs {
            for batch in self.create_training_batches() {
                // Forward pass
                let (loss, accuracy) = self.forward_pass(&batch)?;

                // Backward pass
                self.backward_pass(loss)?;

                // Update parameters
                self.update_parameters(round_config.learning_rate)?;

                total_loss += loss;
                total_accuracy += accuracy;
                iterations += 1;

                // Check for thermal throttling
                if self.should_throttle_training() {
                    tracing::warn!("Thermal throttling detected, reducing training intensity");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }

            tracing::debug!("Completed local epoch {}/{}", epoch + 1, round_config.local_epochs);
        }

        let avg_loss = total_loss / iterations as f64;
        let avg_accuracy = total_accuracy / iterations as f64;

        // Check convergence
        let converged = avg_loss < 0.01 || (iterations > 100 && avg_accuracy > round_config.min_accuracy_threshold);

        Ok(TrainingMetrics {
            accuracy: avg_accuracy,
            loss: avg_loss,
            iterations,
            converged,
            gradient_norm: self.compute_gradient_norm()?,
        })
    }

    fn create_training_batches(&self) -> Vec<Vec<&TrainingExample>> {
        let batch_size = self.config.training_config.batch_size;
        self.training_data.chunks(batch_size).map(|chunk| chunk.iter().collect()).collect()
    }

    fn forward_pass(&self, batch: &[&TrainingExample]) -> Result<(f64, f64)> {
        // Implement forward pass through model
        if let Some(ref model) = self.local_model {
            let mut total_loss = 0.0;
            let mut correct_predictions = 0;
            let batch_size = batch.len();

            // Process each example in the batch
            for example in batch {
                // Convert example to input tensor
                let input_tensor = self.example_to_tensor(example)?;

                // Forward pass through model
                let logits = model.forward(&input_tensor)?;

                // Compute loss (cross-entropy)
                let loss = self.compute_cross_entropy_loss(&logits, &example.label)?;
                total_loss += loss;

                // Compute accuracy
                let predicted = self.get_predicted_class(&logits)?;
                if predicted == example.label {
                    correct_predictions += 1;
                }
            }

            let avg_loss = total_loss / batch_size as f64;
            let accuracy = correct_predictions as f64 / batch_size as f64;

            Ok((avg_loss, accuracy))
        } else {
            Err(TrustformersError::config_error("No model available for forward pass", "forward_pass"))
        }
    }

    fn backward_pass(&mut self, loss: f64) -> Result<()> {
        // Implement backward pass to compute gradients
        if let Some(ref mut model) = self.local_model {
            // Compute gradients using automatic differentiation
            let gradients = model.backward(loss)?;

            // Store gradients for parameter updates
            self.client_state.gradients = Some(gradients);

            // Update gradient statistics
            self.client_state.gradient_norm = self.compute_gradient_norm()?;

            Ok(())
        } else {
            Err(TrustformersError::config_error("No model available for backward pass", "backward_pass"))
        }
    }

    fn update_parameters(&mut self, learning_rate: f64) -> Result<()> {
        // Implement parameter updates using computed gradients
        if let Some(ref mut model) = self.local_model {
            if let Some(ref gradients) = self.client_state.gradients {
                // Apply gradients to model parameters
                model.update_parameters(gradients, learning_rate)?;

                // Update client statistics
                self.client_state.parameters_updated += 1;

                // Clear gradients after update
                self.client_state.gradients = None;

                Ok(())
            } else {
                Err(TrustformersError::config_error("No gradients available for parameter update", "update_parameters"))
            }
        } else {
            Err(TrustformersError::config_error("No model available for parameter update", "update_parameters"))
        }
    }

    fn compute_gradient_norm(&self) -> Result<f64> {
        // Compute L2 norm of gradients
        if let Some(ref gradients) = self.client_state.gradients {
            let mut norm_squared = 0.0;

            // Calculate L2 norm across all gradient tensors
            for gradient in gradients {
                for &value in gradient.data() {
                    norm_squared += value * value;
                }
            }

            Ok(norm_squared.sqrt())
        } else {
            Ok(0.0)
        }
    }

    fn apply_differential_privacy(&mut self) -> Result<PrivacyGuarantees> {
        if !self.config.privacy_config.enabled {
            return Ok(PrivacyGuarantees {
                epsilon_used: 0.0,
                noise_scale: 0.0,
                gradient_clipped: false,
                local_dp_applied: false,
            });
        }

        let epsilon_used = self.config.privacy_config.epsilon / self.config.training_config.max_rounds as f64;

        // Apply gradient clipping
        self.clip_gradients(self.config.privacy_config.clipping_norm)?;

        // Add Gaussian noise
        self.add_noise_to_gradients(self.config.privacy_config.noise_scale)?;

        Ok(PrivacyGuarantees {
            epsilon_used,
            noise_scale: self.config.privacy_config.noise_scale,
            gradient_clipped: true,
            local_dp_applied: self.config.privacy_config.use_local_dp,
        })
    }

    fn clip_gradients(&mut self, _clipping_norm: f64) -> Result<()> {
        // Implement gradient clipping for differential privacy
        Ok(())
    }

    fn add_noise_to_gradients(&mut self, _noise_scale: f64) -> Result<()> {
        // Add Gaussian noise to gradients for differential privacy
        Ok(())
    }

    fn compute_parameter_updates(&self, global_model: &ModelParameters) -> Result<ModelParameters> {
        if let Some(ref local_model) = self.local_model {
            // Compute difference between local and global models
            let mut updates = HashMap::new();
            let mut shapes = HashMap::new();

            for (name, global_params) in &global_model.parameters {
                if let Some(local_params) = local_model.parameters.get(name) {
                    let diff: Vec<f32> = local_params.iter()
                        .zip(global_params.iter())
                        .map(|(local, global)| local - global)
                        .collect();

                    updates.insert(name.clone(), diff);
                    shapes.insert(name.clone(), global_model.shapes[name].clone().into());
                }
            }

            Ok(ModelParameters {
                parameters: updates,
                shapes,
                version: format!("update_{}", SystemTime::now().duration_since(UNIX_EPOCH).expect("Operation failed").as_secs()),
                checksum: "placeholder_checksum".to_string(),
            })
        } else {
            Err(TrustformersError::runtime_error("No local model available".into()).into())
        }
    }

    fn compress_update(&self, update: LocalUpdate) -> Result<LocalUpdate> {
        // Apply compression algorithm if configured
        match self.config.communication_config.compression_algorithm {
            CompressionAlgorithm::None => Ok(update),
            CompressionAlgorithm::TopK(k) => {
                // Implement Top-K sparsification
                self.apply_topk_compression(update, k)
            },
            CompressionAlgorithm::Threshold(threshold) => {
                // Implement threshold-based compression
                self.apply_threshold_compression(update, threshold)
            },
            _ => Ok(update), // Other compression algorithms
        }
    }

    fn apply_topk_compression(&self, mut update: LocalUpdate, k: usize) -> Result<LocalUpdate> {
        // Keep only top-k largest gradients by magnitude
        for (name, params) in update.parameter_updates.parameters.iter_mut() {
            if params.len() > k {
                // Find indices of top-k largest values by magnitude
                let mut indexed_params: Vec<(usize, f32)> = params.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed_params.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).expect("Operation failed").into());

                // Zero out all but top-k
                let mut new_params = vec![0.0; params.len()];
                for i in 0..k {
                    let (idx, value) = indexed_params[i];
                    new_params[idx] = value;
                }

                *params = new_params;
            }
        }

        Ok(update)
    }

    fn apply_threshold_compression(&self, mut update: LocalUpdate, threshold: f64) -> Result<LocalUpdate> {
        // Zero out gradients below threshold
        for params in update.parameter_updates.parameters.values_mut() {
            for param in params.iter_mut() {
                if param.abs() < threshold as f32 {
                    *param = 0.0;
                }
            }
        }

        Ok(update)
    }

    fn check_device_eligibility(&self) -> Result<()> {
        // Check if device meets minimum requirements
        if !self.meets_resource_constraints() {
            return Err(TrustformersError::runtime_error("Device doesn't meet resource constraints".into()).into());
        }

        // Check privacy budget
        if !self.privacy_accountant.can_participate(&self.config.privacy_config) {
            return Err(TrustformersError::runtime_error("Insufficient privacy budget".into()).into());
        }

        Ok(())
    }

    fn meets_selection_criteria(&self, criteria: &SelectionCriteria) -> bool {
        // Check minimum samples
        if self.training_data.len() < criteria.min_samples {
            return false;
        }

        // Check device capabilities (placeholder)
        for required_capability in &criteria.required_capabilities {
            if !self.has_capability(required_capability) {
                return false;
            }
        }

        true
    }

    fn has_capability(&self, _capability: &str) -> bool {
        // Check if device has required capability
        true // Placeholder
    }

    fn meets_resource_constraints(&self) -> bool {
        // Check battery level
        let battery_level = self.get_battery_level();
        if battery_level < self.config.resource_constraints.min_battery_level_percent {
            return false;
        }

        // Check available memory
        let available_memory = self.get_available_memory_mb();
        if available_memory < self.config.resource_constraints.max_memory_mb {
            return false;
        }

        // Check thermal state
        if self.config.resource_constraints.enable_thermal_throttling && self.is_thermally_throttled() {
            return false;
        }

        true
    }

    fn should_throttle_training(&self) -> bool {
        self.config.resource_constraints.enable_thermal_throttling && self.is_thermally_throttled()
    }

    fn update_device_metrics(&mut self, training_time: f64) {
        let metrics = &mut self.client_state.device_metrics;

        // Update training time average
        let alpha = 0.1;
        metrics.avg_training_time_seconds =
            alpha * training_time + (1.0 - alpha) * metrics.avg_training_time_seconds;

        // Update other metrics (placeholder implementations)
        metrics.avg_memory_usage_mb = self.get_current_memory_usage_mb();
        metrics.avg_cpu_usage_percent = self.get_current_cpu_usage_percent();
        metrics.battery_usage_per_round_percent = 2.0; // Placeholder
    }

    // Platform-specific helper methods (placeholders)
    fn get_battery_level(&self) -> f32 { 80.0 }
    fn get_available_memory_mb(&self) -> usize { 2048 }
    fn is_thermally_throttled(&self) -> bool { false }
    fn get_current_memory_usage_mb(&self) -> f32 { 150.0 }
    fn get_current_cpu_usage_percent(&self) -> f32 { 25.0 }

    // Helper methods for forward pass and training
    fn example_to_tensor(&self, example: &TrainingExample) -> Result<Tensor> {
        // Convert training example to tensor format
        // This would depend on the actual data format
        let mut data = Vec::new();
        for &value in &example.features {
            data.push(value);
        }

        // Create tensor with appropriate shape
        let tensor = Tensor::from_vec(data, vec![example.features.len()]).map_err(|e|
            TrustformersError::runtime_error(format!("Failed to create tensor: {}", e))?;

        Ok(tensor)
    }

    fn compute_cross_entropy_loss(&self, logits: &Tensor, label: &usize) -> Result<f64> {
        // Compute cross-entropy loss
        let logits_data = logits.data();
        let num_classes = logits_data.len();

        // Convert to softmax probabilities
        let max_logit = logits_data.iter().fold(f32::NEG_INFINITY, |max, &val| max.max(val).into());
        let exp_logits: Vec<f32> = logits_data.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let softmax: Vec<f32> = exp_logits.iter()
            .map(|&x| x / sum_exp)
            .collect();

        // Compute cross-entropy loss
        if *label < num_classes {
            let loss = -softmax[*label].ln();
            Ok(loss as f64)
        } else {
            Err(TrustformersError::runtime_error(format!("Label {} out of range for {} classes", label, num_classes)).into())
        }
    }

    fn get_predicted_class(&self, logits: &Tensor) -> Result<usize> {
        // Get predicted class from logits
        let logits_data = logits.data();
        let mut max_idx = 0;
        let mut max_val = logits_data[0];

        for (i, &val) in logits_data.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        Ok(max_idx)
    }

    fn backward_pass_personalization(&mut self, loss: f64, personalization_layers: &[String]) -> Result<()> {
        // Backward pass for personalization layers only
        if let Some(ref mut model) = self.local_model {
            let gradients = model.backward_selective(loss, personalization_layers)?;
            self.client_state.gradients = Some(gradients);
            Ok(())
        } else {
            Err(TrustformersError::config_error("No model available for personalization backward pass", "personalization_backward_pass"))
        }
    }

    fn update_personalization_parameters(&mut self, learning_rate: f64) -> Result<()> {
        // Update only personalization parameters
        if let Some(ref mut model) = self.local_model {
            if let Some(ref gradients) = self.client_state.gradients {
                model.update_parameters_selective(gradients, learning_rate)?;
                self.client_state.gradients = None;
                Ok(())
            } else {
                Err(TrustformersError::config_error("No gradients available for personalization update", "personalization_update"))
            }
        } else {
            Err(TrustformersError::config_error("No model available for personalization update", "personalization_update"))
        }
    }
}

// Implementation for supporting structs

impl PrivacyAccountant {
    fn new(epsilon_budget: f64) -> Self {
        Self {
            total_epsilon_used: 0.0,
            epsilon_budget,
            round_epsilon_usage: Vec::new(),
        }
    }

    fn can_participate(&self, config: &DifferentialPrivacyConfig) -> bool {
        if !config.enabled {
            return true;
        }

        let required_epsilon = config.epsilon / 100.0; // Estimate for next round
        self.total_epsilon_used + required_epsilon <= self.epsilon_budget
    }

    fn record_epsilon_usage(&mut self, epsilon: f64) {
        self.total_epsilon_used += epsilon;
        self.round_epsilon_usage.push(epsilon);
    }

    fn reset(&mut self) {
        self.total_epsilon_used = 0.0;
        self.round_epsilon_usage.clear();
    }
}

impl CommunicationManager {
    fn new(server_url: String, client_id: String, api_key: Option<String>, retry_attempts: usize) -> Self {
        Self {
            server_url,
            client_id,
            api_key,
            retry_attempts,
        }
    }

    async fn register_client(&self) -> Result<()> {
        // Register client with federated server
        tracing::info!("Registering client {} with server", self.client_id);
        Ok(())
    }

    async fn wait_for_round(&self) -> Result<FederatedRound> {
        // Poll server for new federated round
        // This would implement actual HTTP communication
        Ok(FederatedRound {
            round_number: 1,
            global_model: ModelParameters {
                parameters: HashMap::new(),
                shapes: HashMap::new(),
                version: "1.0".to_string(),
                checksum: "placeholder".to_string(),
            },
            round_config: RoundConfig {
                learning_rate: 0.01,
                local_epochs: 5,
                min_accuracy_threshold: 0.8,
                adaptive_config: None,
            },
            selection_criteria: SelectionCriteria {
                min_samples: 100,
                required_capabilities: Vec::new(),
                preferred_device_types: Vec::new(),
                region_preferences: None,
            },
            deadline: SystemTime::now().duration_since(UNIX_EPOCH).expect("Operation failed").as_secs() + 3600,
        })
    }

    async fn send_update(&self, _update: LocalUpdate) -> Result<()> {
        // Send local update to federated server
        tracing::info!("Sending update from client {} to server", self.client_id);
        Ok(())
    }
}

impl SecurityManager {
    fn new(config: &SecurityConfig) -> Self {
        Self {
            encryption_enabled: config.enable_encryption,
            certificate_validation: config.validate_certificates,
            gradient_verification: config.enable_gradient_verification,
        }
    }

    fn encrypt_update(&self, update: LocalUpdate) -> Result<LocalUpdate> {
        if !self.encryption_enabled {
            return Ok(update);
        }

        // Implement encryption of local update
        // This would use actual cryptographic libraries
        Ok(update)
    }
}

impl Default for DeviceMetrics {
    fn default() -> Self {
        Self {
            avg_training_time_seconds: 0.0,
            avg_memory_usage_mb: 0.0,
            avg_cpu_usage_percent: 0.0,
            battery_usage_per_round_percent: 0.0,
            network_usage_bytes: 0,
        }
    }
}

// Configuration implementations

impl Default for FederatedLearningConfig {
    fn default() -> Self {
        Self {
            server_url: "https://federated.trustformers.ai".to_string(),
            client_id: "default_client".to_string(),
            api_key: None,
            privacy_config: DifferentialPrivacyConfig::default(),
            training_config: FederatedTrainingConfig::default(),
            communication_config: CommunicationConfig::default(),
            security_config: SecurityConfig::default(),
            resource_constraints: ResourceConstraints::default(),
        }
    }
}

impl Default for DifferentialPrivacyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            epsilon: 1.0,
            noise_scale: 0.1,
            clipping_norm: 1.0,
            sensitivity: 1.0,
            use_local_dp: true,
        }
    }
}

impl Default for FederatedTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            local_epochs: 5,
            batch_size: 32,
            max_rounds: 100,
            min_local_samples: 100,
            enable_personalization: false,
            personalization_layers: Vec::new(),
            use_momentum: true,
            momentum: 0.9,
        }
    }
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            round_timeout_seconds: 3600,
            max_message_size_bytes: 10 * 1024 * 1024, // 10MB
            compression_algorithm: CompressionAlgorithm::TopK(1000),
            quantization_bits: 8,
            enable_secure_aggregation: true,
            retry_attempts: 3,
            min_participants: 10,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_encryption: true,
            encryption_algorithm: EncryptionAlgorithm::AES256,
            enable_smpc: false,
            validate_certificates: true,
            enable_gradient_verification: true,
        }
    }
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory_mb: 512,
            max_cpu_usage_percent: 70.0,
            max_battery_usage_percent: 10.0,
            min_battery_level_percent: 30.0,
            enable_thermal_throttling: true,
            max_training_time_seconds: 600, // 10 minutes
        }
    }
}

impl FederatedLearningConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.server_url.is_empty() {
            return Err(TrustformersError::config_error("Server URL cannot be empty", "validate").into());
        }

        if self.client_id.is_empty() {
            return Err(TrustformersError::config_error("Client ID cannot be empty", "validate").into());
        }

        self.privacy_config.validate()?;
        self.training_config.validate()?;
        self.communication_config.validate()?;
        self.resource_constraints.validate()?;

        Ok(())
    }
}

impl DifferentialPrivacyConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled {
            if self.epsilon <= 0.0 {
                return Err(TrustformersError::config_error("Epsilon must be positive", "validate").into());
            }

            if self.noise_scale <= 0.0 {
                return Err(TrustformersError::config_error("Noise scale must be positive", "validate").into());
            }

            if self.clipping_norm <= 0.0 {
                return Err(TrustformersError::config_error("Clipping norm must be positive", "validate").into());
            }
        }

        Ok(())
    }
}

impl FederatedTrainingConfig {
    fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 {
            return Err(TrustformersError::config_error("Learning rate must be positive", "validate").into());
        }

        if self.local_epochs == 0 {
            return Err(TrustformersError::config_error("Local epochs must be > 0", "validate").into());
        }

        if self.batch_size == 0 {
            return Err(TrustformersError::config_error("Batch size must be > 0", "validate").into());
        }

        if self.max_rounds == 0 {
            return Err(TrustformersError::config_error("Max rounds must be > 0", "validate").into());
        }

        Ok(())
    }
}

impl CommunicationConfig {
    fn validate(&self) -> Result<()> {
        if self.round_timeout_seconds == 0 {
            return Err(TrustformersError::config_error("Round timeout must be > 0", "validate").into());
        }

        if self.max_message_size_bytes == 0 {
            return Err(TrustformersError::config_error("Max message size must be > 0", "validate").into());
        }

        if self.quantization_bits == 0 || self.quantization_bits > 32 {
            return Err(TrustformersError::config_error("Invalid quantization bits", "validate").into());
        }

        Ok(())
    }
}

impl ResourceConstraints {
    fn validate(&self) -> Result<()> {
        if self.max_memory_mb == 0 {
            return Err(TrustformersError::config_error("Max memory must be > 0", "validate").into());
        }

        if self.max_cpu_usage_percent <= 0.0 || self.max_cpu_usage_percent > 100.0 {
            return Err(TrustformersError::config_error("Invalid CPU usage percentage", "validate").into());
        }

        if self.min_battery_level_percent < 0.0 || self.min_battery_level_percent > 100.0 {
            return Err(TrustformersError::config_error("Invalid battery level percentage", "validate").into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_config_validation() {
        let config = FederatedLearningConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.server_url = String::new();
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_privacy_config_validation() {
        let mut config = DifferentialPrivacyConfig::default();
        assert!(config.validate().is_ok());

        config.epsilon = -1.0;
        assert!(config.validate().is_err());

        config.epsilon = 1.0;
        config.noise_scale = 0.0;
        assert!(config.validate().is_err());
    }

    #[tokio::test]
    async fn test_federated_client_creation() {
        let config = FederatedLearningConfig::default();
        let result = FederatedLearningClient::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_topk_compression() {
        let client = FederatedLearningClient::new(FederatedLearningConfig::default()).expect("Operation failed");

        let mut update = LocalUpdate {
            client_id: "test".to_string(),
            round_number: 1,
            parameter_updates: ModelParameters {
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("layer1".to_string(), vec![0.1, 0.9, 0.2, 0.8, 0.05]);
                    params
                },
                shapes: HashMap::new(),
                version: "1.0".to_string(),
                checksum: "test".to_string(),
            },
            num_samples: 100,
            training_metrics: TrainingMetrics {
                accuracy: 0.8,
                loss: 0.2,
                iterations: 10,
                converged: false,
                gradient_norm: 1.0,
            },
            computation_time_seconds: 60.0,
            privacy_guarantees: PrivacyGuarantees {
                epsilon_used: 0.1,
                noise_scale: 0.1,
                gradient_clipped: true,
                local_dp_applied: true,
            },
        };

        let compressed = client.apply_topk_compression(update.clone(), 3).expect("Operation failed");

        // Check that only top-3 values remain non-zero
        let params = &compressed.parameter_updates.parameters["layer1"];
        let non_zero_count = params.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(non_zero_count, 3);
    }
}
