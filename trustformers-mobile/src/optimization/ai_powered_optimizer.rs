//! AI-Powered Optimization Pipeline
//!
//! This module implements machine learning-driven optimization that learns from usage patterns
//! and dynamically adapts model architectures for optimal mobile performance.

use crate::scirs2_compat::random::legacy;
use crate::{MobileBackend, MobileConfig, PerformanceTier};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;

// Helper functions for random number generation
fn random_usize(max: usize) -> usize {
    if max == 0 {
        return 0;
    }
    ((legacy::f64() * max as f64) as usize).min(max.saturating_sub(1))
}

fn random_f32() -> f32 {
    legacy::f32()
}

/// Neural Architecture Search for mobile-optimized model variants
#[derive(Debug, Clone)]
pub struct MobileNAS {
    search_config: NASConfig,
    architecture_candidates: Vec<MobileArchitecture>,
    performance_history: Vec<PerformanceRecord>,
    optimization_agent: ReinforcementLearningAgent,
}

/// Configuration for Neural Architecture Search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NASConfig {
    /// Maximum search iterations
    pub max_iterations: usize,
    /// Performance metrics to optimize
    pub optimization_targets: Vec<OptimizationTarget>,
    /// Device constraints
    pub device_constraints: DeviceConstraints,
    /// Search strategy
    pub search_strategy: SearchStrategy,
    /// Early stopping criteria
    pub early_stopping: EarlyStoppingConfig,
}

/// Optimization targets for NAS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationTarget {
    /// Minimize inference latency
    Latency,
    /// Minimize memory usage
    Memory,
    /// Minimize power consumption
    Power,
    /// Maximize accuracy
    Accuracy,
    /// Minimize model size
    ModelSize,
    /// Minimize energy consumption
    Energy,
}

/// Device constraints for architecture search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConstraints {
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Maximum inference latency in ms
    pub max_latency_ms: f32,
    /// Target performance tier
    pub performance_tier: PerformanceTier,
    /// Available backends
    pub available_backends: Vec<MobileBackend>,
    /// Power budget
    pub power_budget_mw: f32,
}

/// Search strategy for architecture exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Random search baseline
    Random,
    /// Evolutionary algorithm
    Evolutionary {
        population_size: usize,
        mutation_rate: f32,
        crossover_rate: f32,
    },
    /// Reinforcement learning-based search
    ReinforcementLearning {
        learning_rate: f32,
        exploration_rate: f32,
        replay_buffer_size: usize,
    },
    /// Differentiable architecture search
    Differentiable {
        temperature: f32,
        gumbel_softmax: bool,
    },
    /// Progressive search with early pruning
    Progressive {
        stages: usize,
        pruning_threshold: f32,
    },
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Patience iterations
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_improvement: f32,
    /// Monitor metric
    pub monitor_metric: OptimizationTarget,
}

/// Mobile architecture representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileArchitecture {
    /// Architecture ID
    pub id: String,
    /// Layer configuration
    pub layers: Vec<LayerConfig>,
    /// Skip connections
    pub skip_connections: Vec<SkipConnection>,
    /// Quantization scheme
    pub quantization: QuantizationConfig,
    /// Estimated metrics
    pub estimated_metrics: Option<ArchitectureMetrics>,
}

/// Layer configuration for mobile architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Layer type
    pub layer_type: LayerType,
    /// Input dimensions
    pub input_dim: Vec<usize>,
    /// Output dimensions
    pub output_dim: Vec<usize>,
    /// Layer-specific parameters
    pub parameters: HashMap<String, f32>,
    /// Activation function
    pub activation: ActivationType,
}

/// Mobile-optimized layer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    /// Depthwise separable convolution
    DepthwiseSeparableConv {
        kernel_size: usize,
        stride: usize,
        dilation: usize,
    },
    /// Mobile inverted bottleneck
    MobileBottleneck {
        expansion_ratio: f32,
        kernel_size: usize,
        squeeze_excitation: bool,
    },
    /// Efficient channel attention
    EfficientChannelAttention {
        reduction_ratio: usize,
        use_gating: bool,
    },
    /// Mobile multi-head attention
    MobileMultiHeadAttention {
        num_heads: usize,
        head_dim: usize,
        sparse_attention: bool,
    },
    /// Group normalization (mobile-friendly)
    GroupNormalization { num_groups: usize },
    /// Mobile-optimized linear layer
    MobileLinear { use_bias: bool, quantized: bool },
}

/// Activation types optimized for mobile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    /// Swish activation (mobile-optimized)
    Swish,
    /// Hard swish (more efficient)
    HardSwish,
    /// ReLU6 (hardware-friendly)
    ReLU6,
    /// GELU approximation
    GeluApprox,
    /// Mish (if supported by hardware)
    Mish,
}

/// Skip connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkipConnection {
    /// Source layer index
    pub from_layer: usize,
    /// Target layer index
    pub to_layer: usize,
    /// Connection type
    pub connection_type: ConnectionType,
}

/// Types of skip connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Direct residual connection
    Residual,
    /// Dense connection
    Dense,
    /// Attention-based connection
    Attention { num_heads: usize },
    /// Channel shuffle connection
    ChannelShuffle,
}

/// Quantization configuration for architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Per-layer quantization schemes
    pub layer_schemes: HashMap<usize, QuantizationScheme>,
    /// Mixed precision strategy
    pub mixed_precision: bool,
    /// Dynamic quantization
    pub dynamic_quantization: bool,
}

/// Quantization schemes for different layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// 4-bit quantization
    Int4 { symmetric: bool },
    /// 8-bit quantization
    Int8 { symmetric: bool },
    /// 16-bit floating point
    FP16,
    /// Block-wise quantization
    BlockWise { block_size: usize },
    /// Full precision (no quantization)
    FP32,
}

/// Architecture performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureMetrics {
    /// Inference latency in milliseconds
    pub latency_ms: f32,
    /// Memory usage in MB
    pub memory_mb: f32,
    /// Power consumption in mW
    pub power_mw: f32,
    /// Model accuracy (if available)
    pub accuracy: Option<f32>,
    /// Model size in MB
    pub model_size_mb: f32,
    /// Energy consumption per inference in mJ
    pub energy_per_inference_mj: f32,
    /// Throughput (inferences per second)
    pub throughput_fps: f32,
}

/// Performance record for learning
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Architecture that was evaluated
    pub architecture: MobileArchitecture,
    /// Measured performance metrics
    pub metrics: ArchitectureMetrics,
    /// Device configuration
    pub device_config: MobileConfig,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// User context (if available)
    pub user_context: Option<UserContext>,
}

/// User context for personalized optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    /// Usage patterns
    pub usage_patterns: Vec<UsagePattern>,
    /// Performance preferences
    pub preferences: UserPreferences,
    /// Device usage environment
    pub environment: DeviceEnvironment,
}

/// Usage pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    /// Task type
    pub task_type: String,
    /// Frequency of use
    pub frequency: f32,
    /// Typical input characteristics
    pub input_characteristics: InputCharacteristics,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
}

/// Input characteristics for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputCharacteristics {
    /// Typical input sizes
    pub input_sizes: Vec<Vec<usize>>,
    /// Batch sizes commonly used
    pub common_batch_sizes: Vec<usize>,
    /// Data types
    pub data_types: Vec<String>,
}

/// Performance requirements from user perspective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Maximum acceptable latency
    pub max_latency_ms: f32,
    /// Battery life importance (0.0-1.0)
    pub battery_importance: f32,
    /// Accuracy importance (0.0-1.0)
    pub accuracy_importance: f32,
}

/// User preferences for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Preferred optimization target
    pub primary_target: OptimizationTarget,
    /// Secondary optimization targets
    pub secondary_targets: Vec<OptimizationTarget>,
    /// Acceptable quality tradeoffs
    pub quality_tradeoffs: QualityTradeoffs,
}

/// Quality tradeoffs user is willing to accept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTradeoffs {
    /// Maximum accuracy loss acceptable (%)
    pub max_accuracy_loss: f32,
    /// Maximum latency increase acceptable (%)
    pub max_latency_increase: f32,
    /// Maximum memory increase acceptable (%)
    pub max_memory_increase: f32,
}

/// Device environment context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceEnvironment {
    /// Typical charging status
    pub charging_status: ChargingPattern,
    /// Network connectivity patterns
    pub network_patterns: NetworkPattern,
    /// Temperature environment
    pub thermal_environment: ThermalEnvironment,
}

/// Charging patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChargingPattern {
    /// Frequently plugged in
    FrequentCharging,
    /// Moderate charging
    ModerateCharging,
    /// Infrequent charging
    InfrequentCharging,
}

/// Network connectivity patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkPattern {
    /// Mostly WiFi
    PrimarilyWiFi,
    /// Mixed WiFi/Cellular
    Mixed,
    /// Mostly Cellular
    PrimarilyCellular,
    /// Frequent offline usage
    FrequentOffline,
}

/// Thermal environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalEnvironment {
    /// Cool environment
    Cool,
    /// Moderate temperature
    Moderate,
    /// Warm environment
    Warm,
    /// Variable temperature
    Variable,
}

/// Reinforcement Learning agent for optimization
#[derive(Debug, Clone)]
pub struct ReinforcementLearningAgent {
    /// Agent configuration
    config: RLConfig,
    /// Q-value network (simplified representation)
    q_network: QNetwork,
    /// Experience replay buffer
    replay_buffer: Vec<Experience>,
    /// Current exploration rate
    exploration_rate: f32,
    /// The state, chosen action index, and chosen action itself from the
    /// most recent [`ReinforcementLearningAgent::select_action`] call, kept
    /// so that the following
    /// [`ReinforcementLearningAgent::update_from_experience`] call -- which
    /// only receives a `reward` -- can still perform a real gradient update
    /// on the *specific* action that produced it, and can record a
    /// fully-populated [`Experience`] rather than dropping the
    /// (state, action) pair on the floor.
    pending_transition: Option<(Vec<f32>, usize, ArchitectureAction)>,
}

/// Upper bound on how many actions [`ReinforcementLearningAgent::select_action`]
/// will ever propose in one call, and thus how many of [`QNetwork::weights`]'s
/// rows are addressable by an action index.
const MAX_CANDIDATE_ACTIONS: usize = 64;

/// Dimensionality of the architecture-state vectors
/// [`MobileNAS::encode_architecture_state`] produces and [`QNetwork::weights`]'s
/// rows are shaped for.
const STATE_DIM: usize = 128;

/// RL configuration
#[derive(Debug, Clone)]
pub struct RLConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Discount factor
    pub discount_factor: f32,
    /// Initial exploration rate
    pub initial_exploration_rate: f32,
    /// Exploration decay rate
    pub exploration_decay: f32,
    /// Minimum exploration rate
    pub min_exploration_rate: f32,
}

/// Q-Network representation (simplified)
#[derive(Debug, Clone)]
pub struct QNetwork {
    /// Network weights (simplified)
    weights: Vec<Vec<f32>>,
    /// Network architecture
    architecture: Vec<usize>,
}

/// Experience for replay buffer
#[derive(Debug, Clone)]
pub struct Experience {
    /// State (architecture features)
    pub state: Vec<f32>,
    /// Action (architecture modification)
    pub action: ArchitectureAction,
    /// Reward (performance improvement)
    pub reward: f32,
    /// Next state
    pub next_state: Vec<f32>,
    /// Done flag
    pub done: bool,
}

/// Actions that can be taken on architectures
#[derive(Debug, Clone)]
pub enum ArchitectureAction {
    /// Add a layer
    AddLayer {
        layer_type: LayerType,
        position: usize,
    },
    /// Remove a layer
    RemoveLayer { position: usize },
    /// Modify layer parameters
    ModifyLayer {
        position: usize,
        parameter: String,
        value: f32,
    },
    /// Change quantization scheme
    ChangeQuantization {
        layer: usize,
        scheme: QuantizationScheme,
    },
    /// Add skip connection
    AddSkipConnection {
        from: usize,
        to: usize,
        connection_type: ConnectionType,
    },
    /// Remove skip connection
    RemoveSkipConnection { from: usize, to: usize },
}

impl MobileNAS {
    /// Create new Neural Architecture Search engine
    pub fn new(config: NASConfig) -> Self {
        let rl_config = RLConfig {
            learning_rate: 0.001,
            discount_factor: 0.99,
            initial_exploration_rate: 1.0,
            exploration_decay: 0.995,
            min_exploration_rate: 0.1,
        };

        Self {
            search_config: config,
            architecture_candidates: Vec::new(),
            performance_history: Vec::new(),
            optimization_agent: ReinforcementLearningAgent::new(rl_config),
        }
    }

    /// Search for optimal mobile architecture
    pub fn search_optimal_architecture(
        &mut self,
        base_architecture: MobileArchitecture,
        user_context: Option<UserContext>,
    ) -> Result<MobileArchitecture> {
        let mut best_architecture = base_architecture.clone();
        let mut best_score = f32::NEG_INFINITY;
        let mut iterations_without_improvement = 0;

        for iteration in 0..self.search_config.max_iterations {
            // Generate candidate architecture
            let candidate = match &self.search_config.search_strategy {
                SearchStrategy::Random => self.generate_random_architecture(&base_architecture)?,
                SearchStrategy::Evolutionary { .. } => {
                    self.evolve_architecture(&best_architecture)?
                },
                SearchStrategy::ReinforcementLearning { .. } => {
                    self.rl_generate_architecture(&best_architecture)?
                },
                SearchStrategy::Differentiable { .. } => {
                    self.differentiable_search(&best_architecture)?
                },
                SearchStrategy::Progressive { .. } => {
                    self.progressive_search(&best_architecture, iteration)?
                },
            };

            // Encoded up front (before `candidate` is potentially moved into
            // a `PerformanceRecord` below) so a real `next_state` -- the
            // architecture the RL agent's last action actually produced --
            // is available for `update_from_experience` further down.
            let candidate_state = self.encode_architecture_state(&candidate)?;

            // Evaluate candidate architecture
            let metrics = self.evaluate_architecture(&candidate)?;
            let score = self.calculate_fitness_score(&metrics, &user_context)?;

            // Update best architecture if improved
            if score > best_score {
                best_score = score;
                best_architecture = candidate.clone();
                iterations_without_improvement = 0;

                // Record performance for learning
                let record = PerformanceRecord {
                    architecture: candidate,
                    metrics,
                    device_config: MobileConfig::default(), // Would use actual device config
                    timestamp: std::time::SystemTime::now(),
                    user_context: user_context.clone(),
                };
                self.performance_history.push(record);
            } else {
                iterations_without_improvement += 1;
            }

            // Check early stopping
            if iterations_without_improvement >= self.search_config.early_stopping.patience {
                println!(
                    "Early stopping at iteration {} due to no improvement",
                    iteration
                );
                break;
            }

            // Update RL agent if using RL strategy
            if matches!(
                self.search_config.search_strategy,
                SearchStrategy::ReinforcementLearning { .. }
            ) {
                self.optimization_agent.update_from_experience(score, &candidate_state)?;
            }
        }

        Ok(best_architecture)
    }

    /// Generate random architecture mutation
    fn generate_random_architecture(
        &self,
        base: &MobileArchitecture,
    ) -> Result<MobileArchitecture> {
        let mut candidate = base.clone();

        // Apply random mutations
        for _ in 0..3 {
            match random_usize(4) {
                0 => self.mutate_layer_params(&mut candidate)?,
                1 => self.mutate_quantization(&mut candidate)?,
                2 => self.mutate_skip_connections(&mut candidate)?,
                _ => self.mutate_architecture_structure(&mut candidate)?,
            }
        }

        Ok(candidate)
    }

    /// Evolutionary algorithm architecture generation
    fn evolve_architecture(&self, parent: &MobileArchitecture) -> Result<MobileArchitecture> {
        // Simple mutation-based evolution
        let mut offspring = parent.clone();

        // Apply mutations with probability
        if random_f32() < 0.3 {
            self.mutate_layer_params(&mut offspring)?;
        }
        if random_f32() < 0.2 {
            self.mutate_quantization(&mut offspring)?;
        }
        if random_f32() < 0.1 {
            self.mutate_skip_connections(&mut offspring)?;
        }

        Ok(offspring)
    }

    /// RL-based architecture generation
    fn rl_generate_architecture(
        &mut self,
        current: &MobileArchitecture,
    ) -> Result<MobileArchitecture> {
        let state = self.encode_architecture_state(current)?;
        let actions = Self::candidate_actions(current);
        let action = self.optimization_agent.select_action(&state, &actions)?;
        let mut new_architecture = current.clone();

        self.apply_architecture_action(&mut new_architecture, action)?;

        Ok(new_architecture)
    }

    /// Build the real, architecture-derived set of candidate actions
    /// [`ReinforcementLearningAgent::select_action`] chooses from: one
    /// `ModifyLayer` candidate per existing layer (nudging its first known
    /// numeric parameter), plus `AddLayer`/`RemoveLayer`/`ChangeQuantization`
    /// candidates when they are structurally valid. The previous
    /// implementation proposed exactly one hardcoded `ModifyLayer{position:
    /// 0, ...}` regardless of the architecture's real shape.
    fn candidate_actions(architecture: &MobileArchitecture) -> Vec<ArchitectureAction> {
        let mut actions = Vec::new();

        for (i, layer) in architecture.layers.iter().enumerate() {
            if actions.len() >= MAX_CANDIDATE_ACTIONS {
                break;
            }
            let parameter =
                layer.parameters.keys().next().cloned().unwrap_or_else(|| "scale".to_string());
            let current_value = layer.parameters.get(&parameter).copied().unwrap_or(1.0);
            actions.push(ArchitectureAction::ModifyLayer {
                position: i,
                parameter,
                // A deterministic +/-10% nudge, alternating direction by
                // position so the candidate set explores both sides of the
                // current value across a multi-layer architecture instead
                // of always proposing the same direction.
                value: if i % 2 == 0 { current_value * 1.1 } else { current_value * 0.9 },
            });
        }

        if actions.len() < MAX_CANDIDATE_ACTIONS {
            actions.push(ArchitectureAction::AddLayer {
                layer_type: LayerType::MobileLinear {
                    use_bias: true,
                    quantized: false,
                },
                position: architecture.layers.len(),
            });
        }
        if actions.len() < MAX_CANDIDATE_ACTIONS && architecture.layers.len() > 1 {
            actions.push(ArchitectureAction::RemoveLayer {
                position: architecture.layers.len() - 1,
            });
        }
        if actions.len() < MAX_CANDIDATE_ACTIONS && !architecture.layers.is_empty() {
            actions.push(ArchitectureAction::ChangeQuantization {
                layer: 0,
                scheme: QuantizationScheme::Int8 { symmetric: true },
            });
        }

        actions
    }

    /// Differentiable architecture search
    fn differentiable_search(&self, base: &MobileArchitecture) -> Result<MobileArchitecture> {
        // Simplified DARTS implementation
        let mut candidate = base.clone();

        // Apply gradual changes based on differentiable approximations
        for layer in &mut candidate.layers {
            // Adjust layer parameters based on gradient estimation
            if let Some(param) = layer.parameters.get_mut("channels") {
                *param *= 1.0 + (random_f32() - 0.5) * 0.1; // Small random adjustment
            }
        }

        Ok(candidate)
    }

    /// Progressive search with early pruning
    fn progressive_search(
        &self,
        base: &MobileArchitecture,
        iteration: usize,
    ) -> Result<MobileArchitecture> {
        let mut candidate = base.clone();

        // Progressive complexity increase
        let stage = iteration / (self.search_config.max_iterations / 4);
        match stage {
            0 => self.mutate_layer_params(&mut candidate)?,
            1 => self.mutate_quantization(&mut candidate)?,
            2 => self.mutate_skip_connections(&mut candidate)?,
            _ => self.mutate_architecture_structure(&mut candidate)?,
        }

        Ok(candidate)
    }

    /// Evaluate architecture performance
    fn evaluate_architecture(
        &self,
        architecture: &MobileArchitecture,
    ) -> Result<ArchitectureMetrics> {
        // Estimate performance metrics based on architecture
        let mut total_params = 0;
        let mut total_flops = 0;
        let mut memory_usage = 0;

        for layer in &architecture.layers {
            let (params, flops, memory) = self.estimate_layer_metrics(layer)?;
            total_params += params;
            total_flops += flops;
            memory_usage += memory;
        }

        // Estimate metrics based on hardware and architecture
        let latency_ms = self.estimate_latency(total_flops, &architecture.quantization)?;
        let memory_mb = memory_usage as f32 / (1024.0 * 1024.0);
        let power_mw = self.estimate_power_consumption(total_flops, latency_ms)?;
        let model_size_mb = (total_params * 4) as f32 / (1024.0 * 1024.0); // Assume FP32
        let energy_per_inference_mj = power_mw * latency_ms;
        let throughput_fps = 1000.0 / latency_ms;

        Ok(ArchitectureMetrics {
            latency_ms,
            memory_mb,
            power_mw,
            accuracy: None, // Would need actual evaluation
            model_size_mb,
            energy_per_inference_mj,
            throughput_fps,
        })
    }

    /// Calculate fitness score for architecture
    fn calculate_fitness_score(
        &self,
        metrics: &ArchitectureMetrics,
        user_context: &Option<UserContext>,
    ) -> Result<f32> {
        let mut score = 0.0;
        let mut total_weight = 0.0;

        // Weight based on optimization targets
        for &target in &self.search_config.optimization_targets {
            let (value, weight) = match target {
                OptimizationTarget::Latency => {
                    let normalized = 1.0 / (1.0 + metrics.latency_ms / 100.0);
                    (normalized, 1.0)
                },
                OptimizationTarget::Memory => {
                    let normalized = 1.0 / (1.0 + metrics.memory_mb / 512.0);
                    (normalized, 1.0)
                },
                OptimizationTarget::Power => {
                    let normalized = 1.0 / (1.0 + metrics.power_mw / 1000.0);
                    (normalized, 1.0)
                },
                OptimizationTarget::ModelSize => {
                    let normalized = 1.0 / (1.0 + metrics.model_size_mb / 100.0);
                    (normalized, 1.0)
                },
                OptimizationTarget::Energy => {
                    let normalized = 1.0 / (1.0 + metrics.energy_per_inference_mj / 10.0);
                    (normalized, 1.0)
                },
                OptimizationTarget::Accuracy => {
                    let normalized = metrics.accuracy.unwrap_or(0.8);
                    (normalized, 2.0) // Higher weight for accuracy
                },
            };

            score += value * weight;
            total_weight += weight;
        }

        // Adjust score based on user context
        if let Some(ref context) = user_context {
            score = self.adjust_score_for_user_context(score, metrics, context)?;
        }

        // Apply device constraints penalties
        score = self.apply_constraint_penalties(score, metrics)?;

        Ok(score / total_weight)
    }

    /// Adjust score based on user context
    fn adjust_score_for_user_context(
        &self,
        base_score: f32,
        metrics: &ArchitectureMetrics,
        context: &UserContext,
    ) -> Result<f32> {
        let mut adjusted_score = base_score;

        // Adjust based on user preferences
        match context.preferences.primary_target {
            OptimizationTarget::Latency if metrics.latency_ms > 50.0 => {
                adjusted_score *= 0.8; // Penalize high latency
            },
            OptimizationTarget::Memory if metrics.memory_mb > 256.0 => {
                adjusted_score *= 0.8; // Penalize high memory usage
            },
            OptimizationTarget::Power if metrics.power_mw > 500.0 => {
                adjusted_score *= 0.8; // Penalize high power consumption
            },
            _ => {},
        }

        // Consider usage patterns
        for pattern in &context.usage_patterns {
            if pattern.frequency > 0.5
                && metrics.latency_ms > pattern.performance_requirements.max_latency_ms
            {
                adjusted_score *= 0.9; // Penalize if doesn't meet frequent use case requirements
            }
        }

        Ok(adjusted_score)
    }

    /// Apply device constraint penalties
    fn apply_constraint_penalties(
        &self,
        base_score: f32,
        metrics: &ArchitectureMetrics,
    ) -> Result<f32> {
        let mut score = base_score;

        // Check memory constraints
        if metrics.memory_mb > self.search_config.device_constraints.max_memory_mb as f32 {
            score *= 0.5; // Heavy penalty for exceeding memory limit
        }

        // Check latency constraints
        if metrics.latency_ms > self.search_config.device_constraints.max_latency_ms {
            score *= 0.5; // Heavy penalty for exceeding latency limit
        }

        // Check power constraints
        if metrics.power_mw > self.search_config.device_constraints.power_budget_mw {
            score *= 0.7; // Moderate penalty for exceeding power budget
        }

        Ok(score)
    }

    /// Helper methods for mutations (simplified implementations)
    fn mutate_layer_params(&self, architecture: &mut MobileArchitecture) -> Result<()> {
        if !architecture.layers.is_empty() {
            let layer_idx = random_usize(architecture.layers.len());
            let layer = &mut architecture.layers[layer_idx];

            // Mutate a random parameter
            if !layer.parameters.is_empty() {
                let keys: Vec<_> = layer.parameters.keys().cloned().collect();
                let param_key = &keys[random_usize(keys.len())];
                if let Some(value) = layer.parameters.get_mut(param_key) {
                    *value *= 1.0 + (random_f32() - 0.5) * 0.2; // ±10% change
                }
            }
        }
        Ok(())
    }

    fn mutate_quantization(&self, architecture: &mut MobileArchitecture) -> Result<()> {
        if !architecture.layers.is_empty() {
            let layer_idx = random_usize(architecture.layers.len());
            let schemes = [
                QuantizationScheme::Int4 { symmetric: true },
                QuantizationScheme::Int8 { symmetric: true },
                QuantizationScheme::FP16,
                QuantizationScheme::FP32,
            ];
            let scheme = schemes[random_usize(schemes.len())].clone();
            architecture.quantization.layer_schemes.insert(layer_idx, scheme);
        }
        Ok(())
    }

    /// Really add or remove one skip connection between two distinct real
    /// layer indices -- previously a no-op, so this mutation kind was
    /// dead weight in [`Self::generate_random_architecture`]'s selection.
    fn mutate_skip_connections(&self, architecture: &mut MobileArchitecture) -> Result<()> {
        if architecture.layers.len() < 2 {
            return Ok(());
        }

        if !architecture.skip_connections.is_empty() && random_f32() < 0.5 {
            let idx = random_usize(architecture.skip_connections.len());
            architecture.skip_connections.remove(idx);
            return Ok(());
        }

        let from = random_usize(architecture.layers.len() - 1);
        let to = from + 1 + random_usize(architecture.layers.len() - from - 1);
        if to > from
            && to < architecture.layers.len()
            && !architecture
                .skip_connections
                .iter()
                .any(|c| c.from_layer == from && c.to_layer == to)
        {
            architecture.skip_connections.push(SkipConnection {
                from_layer: from,
                to_layer: to,
                connection_type: ConnectionType::Residual,
            });
        }

        Ok(())
    }

    /// Really add or remove one layer -- previously a no-op. Delegates to
    /// [`MobileNAS::apply_architecture_action`] so structural mutation goes
    /// through the same real insert/remove logic the RL strategy uses,
    /// rather than a second, divergent implementation.
    fn mutate_architecture_structure(&self, architecture: &mut MobileArchitecture) -> Result<()> {
        let action = if architecture.layers.len() > 1 && random_f32() < 0.5 {
            ArchitectureAction::RemoveLayer {
                position: architecture.layers.len() - 1,
            }
        } else {
            ArchitectureAction::AddLayer {
                layer_type: LayerType::MobileLinear {
                    use_bias: true,
                    quantized: false,
                },
                position: architecture.layers.len(),
            }
        };
        self.apply_architecture_action(architecture, action)
    }

    fn estimate_layer_metrics(&self, layer: &LayerConfig) -> Result<(usize, usize, usize)> {
        // Simplified metric estimation
        let params =
            layer.input_dim.iter().product::<usize>() * layer.output_dim.iter().product::<usize>();
        let flops = params * 2; // Rough estimate
        let memory = params * 4; // Assume FP32
        Ok((params, flops, memory))
    }

    fn estimate_latency(
        &self,
        total_flops: usize,
        _quantization: &QuantizationConfig,
    ) -> Result<f32> {
        // Simplified latency estimation
        let base_latency = total_flops as f32 / 1_000_000.0; // Assume 1M FLOPS per ms
        Ok(base_latency)
    }

    fn estimate_power_consumption(&self, total_flops: usize, latency_ms: f32) -> Result<f32> {
        // Simplified power estimation
        let power = (total_flops as f32 / 1_000_000.0) * 100.0 + latency_ms * 10.0;
        Ok(power)
    }

    /// Encode `architecture`'s real, current shape into a fixed
    /// [`STATE_DIM`]-length feature vector: layer count, skip-connection
    /// count, quantization summary, log-scaled total params/FLOPs/memory
    /// (via [`Self::estimate_layer_metrics`]), a per-layer-type histogram,
    /// and a fold of each layer's own input/output dimensions into the
    /// remaining slots. Deterministic and reproducible for a given
    /// architecture -- never the flat `vec![0.5; 128]` constant this used
    /// to return regardless of input, which made every state the RL agent
    /// ever saw indistinguishable from every other.
    fn encode_architecture_state(&self, architecture: &MobileArchitecture) -> Result<Vec<f32>> {
        let mut state = vec![0.0f32; STATE_DIM];

        state[0] = architecture.layers.len() as f32;
        state[1] = architecture.skip_connections.len() as f32;
        state[2] = architecture.quantization.layer_schemes.len() as f32;
        state[3] = if architecture.quantization.mixed_precision { 1.0 } else { 0.0 };
        state[4] = if architecture.quantization.dynamic_quantization { 1.0 } else { 0.0 };

        let mut total_params = 0usize;
        let mut total_flops = 0usize;
        let mut total_memory = 0usize;
        // One counter per `LayerType` variant.
        let mut type_counts = [0f32; 6];
        for layer in &architecture.layers {
            if let Ok((params, flops, memory)) = self.estimate_layer_metrics(layer) {
                total_params += params;
                total_flops += flops;
                total_memory += memory;
            }
            let idx = match layer.layer_type {
                LayerType::DepthwiseSeparableConv { .. } => 0,
                LayerType::MobileBottleneck { .. } => 1,
                LayerType::EfficientChannelAttention { .. } => 2,
                LayerType::MobileMultiHeadAttention { .. } => 3,
                LayerType::GroupNormalization { .. } => 4,
                LayerType::MobileLinear { .. } => 5,
            };
            type_counts[idx] += 1.0;
        }
        state[5] = (total_params as f32).ln_1p();
        state[6] = (total_flops as f32).ln_1p();
        state[7] = (total_memory as f32).ln_1p();
        for (i, count) in type_counts.iter().enumerate() {
            state[8 + i] = *count;
        }

        // Fold each layer's own dimensions into the remaining slots so
        // architectures that differ only in layer shape (not count/type)
        // still produce distinguishable states.
        let base = 8 + type_counts.len();
        let remaining = STATE_DIM - base;
        if remaining > 0 {
            for (i, layer) in architecture.layers.iter().enumerate() {
                let slot = base + (i % remaining);
                let dim_sum: usize =
                    layer.input_dim.iter().sum::<usize>() + layer.output_dim.iter().sum::<usize>();
                state[slot] += (dim_sum as f32).ln_1p();
            }
        }

        Ok(state)
    }

    /// Really apply `action` to `architecture` -- the previous
    /// implementation discarded every action and returned `Ok(())`, so
    /// [`Self::search_optimal_architecture`]'s RL strategy could never
    /// change the architecture it was supposedly searching over.
    fn apply_architecture_action(
        &self,
        architecture: &mut MobileArchitecture,
        action: ArchitectureAction,
    ) -> Result<()> {
        match action {
            ArchitectureAction::AddLayer {
                layer_type,
                position,
            } => {
                let position = position.min(architecture.layers.len());
                // Inherit dimensions from the neighbouring layer so the new
                // layer is shape-compatible with its surroundings rather
                // than an arbitrary guess.
                let (input_dim, output_dim) =
                    if let Some(prev) = architecture.layers.get(position.saturating_sub(1)) {
                        (prev.output_dim.clone(), prev.output_dim.clone())
                    } else if let Some(next) = architecture.layers.get(position) {
                        (next.input_dim.clone(), next.input_dim.clone())
                    } else {
                        (vec![1], vec![1])
                    };
                architecture.layers.insert(
                    position,
                    LayerConfig {
                        layer_type,
                        input_dim,
                        output_dim,
                        parameters: HashMap::new(),
                        activation: ActivationType::ReLU6,
                    },
                );
            },
            ArchitectureAction::RemoveLayer { position } => {
                // Never remove the last remaining layer: an empty
                // architecture is not a smaller candidate, it is a broken
                // one.
                if architecture.layers.len() > 1 && position < architecture.layers.len() {
                    architecture.layers.remove(position);
                    // Drop skip connections and quantization entries that
                    // referenced the removed index so the architecture
                    // stays internally consistent.
                    architecture
                        .skip_connections
                        .retain(|c| c.from_layer != position && c.to_layer != position);
                    architecture.quantization.layer_schemes.remove(&position);
                }
            },
            ArchitectureAction::ModifyLayer {
                position,
                parameter,
                value,
            } => {
                if let Some(layer) = architecture.layers.get_mut(position) {
                    layer.parameters.insert(parameter, value);
                }
            },
            ArchitectureAction::ChangeQuantization { layer, scheme } => {
                architecture.quantization.layer_schemes.insert(layer, scheme);
            },
            ArchitectureAction::AddSkipConnection {
                from,
                to,
                connection_type,
            } => {
                let already_present = architecture
                    .skip_connections
                    .iter()
                    .any(|c| c.from_layer == from && c.to_layer == to);
                if !already_present
                    && from < architecture.layers.len()
                    && to < architecture.layers.len()
                {
                    architecture.skip_connections.push(SkipConnection {
                        from_layer: from,
                        to_layer: to,
                        connection_type,
                    });
                }
            },
            ArchitectureAction::RemoveSkipConnection { from, to } => {
                architecture
                    .skip_connections
                    .retain(|c| !(c.from_layer == from && c.to_layer == to));
            },
        }

        Ok(())
    }
}

impl ReinforcementLearningAgent {
    fn new(config: RLConfig) -> Self {
        Self {
            exploration_rate: config.initial_exploration_rate,
            config,
            q_network: QNetwork {
                weights: vec![vec![0.0; STATE_DIM]; MAX_CANDIDATE_ACTIONS],
                architecture: vec![STATE_DIM, 64, 32, 16],
            },
            replay_buffer: Vec::new(),
            pending_transition: None,
        }
    }

    /// Choose the next architecture-modification action. `actions` is a
    /// real, architecture-derived candidate list built by
    /// [`MobileNAS::candidate_actions`] -- not a single hardcoded
    /// `ModifyLayer`. On the exploit branch this picks the candidate with
    /// the highest real linear Q-value (`dot(state, weights[i])`), not a
    /// constant `0`; [`Self::update_from_experience`] is what moves those
    /// weights toward whichever action actually earned a high reward.
    fn select_action(
        &mut self,
        state: &[f32],
        actions: &[ArchitectureAction],
    ) -> Result<ArchitectureAction> {
        if actions.is_empty() {
            return Err(trustformers_core::errors::runtime_error(
                "ReinforcementLearningAgent::select_action called with zero candidate actions"
                    .to_string(),
            ));
        }

        let action_idx = if random_f32() < self.exploration_rate {
            // Explore: a uniformly random candidate.
            random_usize(actions.len())
        } else {
            // Exploit: the real argmax over this state's Q-values.
            self.q_network.best_action_index(state, actions.len())
        };

        let chosen = actions[action_idx].clone();
        self.pending_transition = Some((state.to_vec(), action_idx, chosen.clone()));
        Ok(chosen)
    }

    /// Real single-step (contextual-bandit-style) Q-learning update: a
    /// gradient-ascent step on exactly the weight row
    /// [`Self::select_action`] used to choose its last action, pushing that
    /// row toward the state that produced a high reward (and away from it
    /// for a negative one). Also records a fully-populated [`Experience`]
    /// in the replay buffer -- previously never written to despite
    /// existing. A no-op (not a fabricated update) if `select_action` was
    /// never called since the last `update_from_experience`.
    fn update_from_experience(&mut self, reward: f32, next_state: &[f32]) -> Result<()> {
        self.exploration_rate = (self.exploration_rate * self.config.exploration_decay)
            .max(self.config.min_exploration_rate);

        if let Some((state, action_idx, action)) = self.pending_transition.take() {
            if let Some(row) = self.q_network.weights.get_mut(action_idx) {
                let lr = self.config.learning_rate;
                for (w, s) in row.iter_mut().zip(state.iter()) {
                    *w += lr * reward * s;
                }
            }

            self.replay_buffer.push(Experience {
                state,
                action,
                reward,
                next_state: next_state.to_vec(),
                done: false,
            });
            // Bound memory use: a rolling window, not an unbounded log.
            const MAX_REPLAY_BUFFER: usize = 10_000;
            if self.replay_buffer.len() > MAX_REPLAY_BUFFER {
                let overflow = self.replay_buffer.len() - MAX_REPLAY_BUFFER;
                self.replay_buffer.drain(0..overflow);
            }
        }

        Ok(())
    }
}

impl QNetwork {
    /// The candidate index whose weight row has the highest dot-product
    /// score against `state` -- a real (if linear) Q-value evaluation.
    /// Indices beyond `weights.len()` (if `num_candidates` ever exceeds the
    /// number of allocated rows) score `0.0` rather than panicking. Ties
    /// resolve to the *highest* index (`Iterator::max_by`'s documented
    /// behaviour: "if several elements are equally maximum, the last
    /// element is returned").
    fn best_action_index(&self, state: &[f32], num_candidates: usize) -> usize {
        if num_candidates == 0 {
            return 0;
        }
        (0..num_candidates)
            .max_by(|&a, &b| {
                let score_a = Self::dot(state, self.weights.get(a));
                let score_b = Self::dot(state, self.weights.get(b));
                score_a.total_cmp(&score_b)
            })
            .unwrap_or(0)
    }

    fn dot(state: &[f32], weights: Option<&Vec<f32>>) -> f32 {
        match weights {
            Some(w) => state.iter().zip(w.iter()).map(|(s, w)| s * w).sum(),
            None => 0.0,
        }
    }
}

impl Default for NASConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            optimization_targets: vec![
                OptimizationTarget::Latency,
                OptimizationTarget::Memory,
                OptimizationTarget::Power,
            ],
            device_constraints: DeviceConstraints {
                max_memory_mb: 512,
                max_latency_ms: 100.0,
                performance_tier: PerformanceTier::Mid,
                available_backends: vec![MobileBackend::CPU, MobileBackend::GPU],
                power_budget_mw: 1000.0,
            },
            search_strategy: SearchStrategy::Evolutionary {
                population_size: 20,
                mutation_rate: 0.1,
                crossover_rate: 0.7,
            },
            early_stopping: EarlyStoppingConfig {
                patience: 10,
                min_improvement: 0.01,
                monitor_metric: OptimizationTarget::Latency,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_nas_creation() {
        let config = NASConfig::default();
        let nas = MobileNAS::new(config);
        assert_eq!(nas.architecture_candidates.len(), 0);
    }

    #[test]
    fn test_architecture_metrics() {
        let metrics = ArchitectureMetrics {
            latency_ms: 50.0,
            memory_mb: 128.0,
            power_mw: 500.0,
            accuracy: Some(0.9),
            model_size_mb: 25.0,
            energy_per_inference_mj: 25.0,
            throughput_fps: 20.0,
        };

        assert_eq!(metrics.latency_ms, 50.0);
        assert_eq!(metrics.throughput_fps, 20.0);
    }

    #[test]
    fn test_nas_config_default() {
        let config = NASConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert!(config.optimization_targets.contains(&OptimizationTarget::Latency));
    }

    #[test]
    fn test_optimization_target_variants() {
        let targets = vec![
            OptimizationTarget::Latency,
            OptimizationTarget::Memory,
            OptimizationTarget::Power,
            OptimizationTarget::Accuracy,
            OptimizationTarget::ModelSize,
            OptimizationTarget::Energy,
        ];
        assert_eq!(targets.len(), 6);
    }

    #[test]
    fn test_search_strategy_random() {
        let strategy = SearchStrategy::Random;
        assert!(matches!(strategy, SearchStrategy::Random));
    }

    #[test]
    fn test_search_strategy_evolutionary() {
        let strategy = SearchStrategy::Evolutionary {
            population_size: 50,
            mutation_rate: 0.1,
            crossover_rate: 0.5,
        };
        if let SearchStrategy::Evolutionary {
            population_size, ..
        } = strategy
        {
            assert_eq!(population_size, 50);
        }
    }

    #[test]
    fn test_activation_type_variants() {
        let activations = vec![
            ActivationType::Swish,
            ActivationType::HardSwish,
            ActivationType::ReLU6,
            ActivationType::GeluApprox,
            ActivationType::Mish,
        ];
        assert_eq!(activations.len(), 5);
    }

    #[test]
    fn test_connection_type_variants() {
        let connections = vec![
            ConnectionType::Residual,
            ConnectionType::Dense,
            ConnectionType::Attention { num_heads: 4 },
            ConnectionType::ChannelShuffle,
        ];
        assert_eq!(connections.len(), 4);
    }

    #[test]
    fn test_quantization_scheme_variants() {
        let schemes = vec![
            QuantizationScheme::Int4 { symmetric: true },
            QuantizationScheme::Int8 { symmetric: false },
            QuantizationScheme::FP16,
            QuantizationScheme::BlockWise { block_size: 32 },
            QuantizationScheme::FP32,
        ];
        assert_eq!(schemes.len(), 5);
    }

    #[test]
    fn test_layer_type_depthwise_conv() {
        let layer = LayerType::DepthwiseSeparableConv {
            kernel_size: 3,
            stride: 1,
            dilation: 1,
        };
        if let LayerType::DepthwiseSeparableConv { kernel_size, .. } = layer {
            assert_eq!(kernel_size, 3);
        }
    }

    #[test]
    fn test_layer_type_mobile_bottleneck() {
        let layer = LayerType::MobileBottleneck {
            expansion_ratio: 6.0,
            kernel_size: 3,
            squeeze_excitation: true,
        };
        if let LayerType::MobileBottleneck {
            expansion_ratio, ..
        } = layer
        {
            assert_eq!(expansion_ratio, 6.0);
        }
    }

    #[test]
    fn test_layer_config_creation() {
        let config = LayerConfig {
            layer_type: LayerType::MobileLinear {
                use_bias: true,
                quantized: false,
            },
            input_dim: vec![768],
            output_dim: vec![256],
            parameters: HashMap::new(),
            activation: ActivationType::ReLU6,
        };
        assert_eq!(config.input_dim, vec![768]);
        assert_eq!(config.output_dim, vec![256]);
    }

    #[test]
    fn test_skip_connection_creation() {
        let skip = SkipConnection {
            from_layer: 0,
            to_layer: 2,
            connection_type: ConnectionType::Residual,
        };
        assert_eq!(skip.from_layer, 0);
        assert_eq!(skip.to_layer, 2);
    }

    #[test]
    fn test_architecture_metrics_throughput() {
        let metrics = ArchitectureMetrics {
            latency_ms: 10.0,
            memory_mb: 64.0,
            power_mw: 200.0,
            accuracy: Some(0.95),
            model_size_mb: 10.0,
            energy_per_inference_mj: 2.0,
            throughput_fps: 100.0,
        };
        assert!((metrics.throughput_fps - 1000.0 / metrics.latency_ms).abs() < 1e-3);
    }

    #[test]
    fn test_early_stopping_config() {
        let config = EarlyStoppingConfig {
            patience: 10,
            min_improvement: 0.001,
            monitor_metric: OptimizationTarget::Accuracy,
        };
        assert_eq!(config.patience, 10);
        assert!(config.min_improvement > 0.0);
    }

    #[test]
    fn test_device_constraints_creation() {
        let constraints = DeviceConstraints {
            max_memory_mb: 256,
            max_latency_ms: 50.0,
            performance_tier: PerformanceTier::Medium,
            available_backends: vec![MobileBackend::CPU],
            power_budget_mw: 1000.0,
        };
        assert_eq!(constraints.max_memory_mb, 256);
        assert!(!constraints.available_backends.is_empty());
    }

    #[test]
    fn test_quantization_config_creation() {
        let config = QuantizationConfig {
            layer_schemes: HashMap::new(),
            mixed_precision: true,
            dynamic_quantization: false,
        };
        assert!(config.mixed_precision);
        assert!(!config.dynamic_quantization);
        assert!(config.layer_schemes.is_empty());
    }

    #[test]
    fn test_mobile_architecture_creation() {
        let arch = MobileArchitecture {
            id: "arch_001".to_string(),
            layers: vec![],
            skip_connections: vec![],
            quantization: QuantizationConfig {
                layer_schemes: HashMap::new(),
                mixed_precision: false,
                dynamic_quantization: false,
            },
            estimated_metrics: None,
        };
        assert_eq!(arch.id, "arch_001");
        assert!(arch.layers.is_empty());
        assert!(arch.estimated_metrics.is_none());
    }

    #[test]
    fn test_nas_with_added_candidates() {
        let config = NASConfig::default();
        let nas = MobileNAS::new(config);
        assert_eq!(nas.architecture_candidates.len(), 0);
        // Verify performance history starts empty
        assert_eq!(nas.performance_history.len(), 0);
    }

    fn three_layer_architecture() -> MobileArchitecture {
        let mut layer = LayerConfig {
            layer_type: LayerType::MobileLinear {
                use_bias: true,
                quantized: false,
            },
            input_dim: vec![32],
            output_dim: vec![64],
            parameters: HashMap::new(),
            activation: ActivationType::ReLU6,
        };
        layer.parameters.insert("channels".to_string(), 64.0);
        MobileArchitecture {
            id: "arch_test".to_string(),
            layers: vec![layer.clone(), layer.clone(), layer],
            skip_connections: vec![],
            quantization: QuantizationConfig {
                layer_schemes: HashMap::new(),
                mixed_precision: false,
                dynamic_quantization: false,
            },
            estimated_metrics: None,
        }
    }

    /// Regression test for the previous `encode_architecture_state`, which
    /// ignored its argument entirely and always returned `vec![0.5; 128]`.
    /// Two structurally different architectures must now encode to
    /// different states.
    #[test]
    fn test_encode_architecture_state_is_not_a_flat_constant() {
        let nas = MobileNAS::new(NASConfig::default());
        let three_layers = three_layer_architecture();
        let mut one_layer = three_layers.clone();
        one_layer.layers.truncate(1);

        let state_a = nas.encode_architecture_state(&three_layers).expect("encode");
        let state_b = nas.encode_architecture_state(&one_layer).expect("encode");

        assert_eq!(state_a.len(), STATE_DIM);
        assert_ne!(
            state_a, state_b,
            "different architectures must encode to different states"
        );
        assert!(
            state_a.iter().any(|&x| x != 0.5),
            "state must reflect real architecture data, not the old flat 0.5 constant"
        );
    }

    /// Regression test for the previous `select_action`, whose exploit
    /// branch was `0 // Simplified: always pick first action` regardless
    /// of the Q-network's weights. With exploration forced off and one
    /// candidate action's weight row biased strongly positive, exploitation
    /// must pick *that* action, not always index 0.
    #[test]
    fn test_select_action_exploits_real_q_values_not_a_hardcoded_index() {
        let rl_config = RLConfig {
            learning_rate: 0.01,
            discount_factor: 0.99,
            initial_exploration_rate: 0.0, // force pure exploitation
            exploration_decay: 1.0,
            min_exploration_rate: 0.0,
        };
        let mut agent = ReinforcementLearningAgent::new(rl_config);
        let architecture = three_layer_architecture();
        let actions = MobileNAS::candidate_actions(&architecture);
        assert!(
            actions.len() >= 3,
            "a 3-layer architecture must offer more than one candidate"
        );

        let state = vec![1.0f32; STATE_DIM];
        // Bias action index 2's weight row strongly positive so its
        // dot-product score dominates every other candidate's (all other
        // rows start at exactly 0.0).
        agent.q_network.weights[2] = vec![10.0; STATE_DIM];

        let chosen = agent.select_action(&state, &actions).expect("select_action");
        assert_eq!(
            format!("{chosen:?}"),
            format!("{:?}", actions[2]),
            "exploitation must follow the real (biased) Q-values, not always index 0"
        );
    }

    /// Regression test for the previous `update_from_experience`, which
    /// decayed `exploration_rate` and otherwise did nothing -- "In a real
    /// implementation, this would update the Q-network weights" -- so the
    /// weights the exploit branch reads from never changed no matter the
    /// reward. A real update must move the acted-on row and record a real
    /// `Experience`.
    #[test]
    fn test_update_from_experience_really_changes_weights_and_records_experience() {
        let rl_config = RLConfig {
            learning_rate: 0.5,
            discount_factor: 0.99,
            initial_exploration_rate: 0.0,
            exploration_decay: 1.0,
            min_exploration_rate: 0.0,
        };
        let mut agent = ReinforcementLearningAgent::new(rl_config);
        let architecture = three_layer_architecture();
        let actions = MobileNAS::candidate_actions(&architecture);

        let state = vec![1.0f32; STATE_DIM];
        let chosen = agent.select_action(&state, &actions).expect("select_action");
        assert!(
            agent.replay_buffer.is_empty(),
            "no experience recorded before an update"
        );

        let before = agent.q_network.weights.clone();
        agent.update_from_experience(2.5, &state).expect("update");
        let after = &agent.q_network.weights;

        assert_ne!(
            before, *after,
            "update_from_experience must actually change the Q-network"
        );
        // Every row *except* the one just acted on must be untouched.
        let changed_rows: Vec<usize> =
            (0..before.len()).filter(|&i| before[i] != after[i]).collect();
        assert_eq!(
            changed_rows.len(),
            1,
            "exactly one action's weight row should move per update, got {changed_rows:?}"
        );

        assert_eq!(agent.replay_buffer.len(), 1);
        let recorded = &agent.replay_buffer[0];
        assert_eq!(recorded.reward, 2.5);
        assert_eq!(recorded.state, state);
        assert_eq!(format!("{:?}", recorded.action), format!("{chosen:?}"));

        // A second update with no intervening `select_action` must be a
        // genuine no-op, not a fabricated "success".
        let after_first = agent.q_network.weights.clone();
        agent
            .update_from_experience(99.0, &state)
            .expect("update with no pending transition");
        assert_eq!(agent.q_network.weights, after_first);
        assert_eq!(
            agent.replay_buffer.len(),
            1,
            "no pending transition means no new experience"
        );
    }

    /// Regression test for the previous `apply_architecture_action`, which
    /// was `// Simplified action application` / `Ok(())` for every variant
    /// -- the architecture passed in was never mutated at all.
    #[test]
    fn test_apply_architecture_action_really_mutates_the_architecture() {
        let nas = MobileNAS::new(NASConfig::default());
        let mut architecture = three_layer_architecture();

        nas.apply_architecture_action(
            &mut architecture,
            ArchitectureAction::ModifyLayer {
                position: 0,
                parameter: "channels".to_string(),
                value: 128.0,
            },
        )
        .expect("apply ModifyLayer");
        assert_eq!(
            architecture.layers[0].parameters.get("channels"),
            Some(&128.0)
        );

        let layer_count_before = architecture.layers.len();
        nas.apply_architecture_action(
            &mut architecture,
            ArchitectureAction::AddLayer {
                layer_type: LayerType::MobileLinear {
                    use_bias: false,
                    quantized: true,
                },
                position: 1,
            },
        )
        .expect("apply AddLayer");
        assert_eq!(architecture.layers.len(), layer_count_before + 1);

        nas.apply_architecture_action(
            &mut architecture,
            ArchitectureAction::RemoveLayer { position: 0 },
        )
        .expect("apply RemoveLayer");
        assert_eq!(architecture.layers.len(), layer_count_before);

        nas.apply_architecture_action(
            &mut architecture,
            ArchitectureAction::ChangeQuantization {
                layer: 0,
                scheme: QuantizationScheme::Int4 { symmetric: true },
            },
        )
        .expect("apply ChangeQuantization");
        assert!(architecture.quantization.layer_schemes.contains_key(&0));
    }
}
