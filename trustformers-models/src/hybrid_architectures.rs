//! # Hybrid Architectures Framework
//!
//! This module provides a comprehensive framework for creating and managing hybrid neural
//! architectures that combine multiple paradigms (e.g., transformers + CNNs, RNNs + attention,
//! state-space models + transformers) to leverage the strengths of different approaches.
//!
//! ## Features
//!
//! - **Multi-Paradigm Fusion**: Combine transformers, CNNs, RNNs, state-space models, and more
//! - **Flexible Architecture Composition**: Modular design for easy architecture mixing
//! - **Adaptive Switching**: Dynamic selection between different computational paths
//! - **Cross-Modal Integration**: Support for vision, text, audio, and multimodal processing
//! - **Efficiency Optimization**: Hybrid approaches for better speed/accuracy trade-offs
//! - **Custom Fusion Strategies**: Extensible framework for novel combination methods
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_models::hybrid_architectures::{
//!     HybridArchitecture, HybridConfig, ArchitecturalComponent, FusionStrategy,
//!     CNNArchitecture, TransformerVariant,
//! };
//! use trustformers_core::Result;
//!
//! fn main() -> Result<()> {
//!     // Create a hybrid CNN-Transformer architecture
//!     let config = HybridConfig::builder()
//!         .add_component(ArchitecturalComponent::CNN {
//!             layers: 3,
//!             channels: 64,
//!             kernel_size: 3,
//!             architecture: CNNArchitecture::ResNet,
//!         })
//!         .add_component(ArchitecturalComponent::Transformer {
//!             layers: 6,
//!             hidden_size: 512,
//!             num_heads: 8,
//!             variant: TransformerVariant::Standard,
//!         })
//!         .fusion_strategy(FusionStrategy::Sequential)
//!         .build()?;
//!
//!     let hybrid_model = HybridArchitecture::new(config)?;
//!     println!("Created hybrid architecture with {} components", hybrid_model.num_components());
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use trustformers_core::{
    errors::{invalid_input, not_implemented, TrustformersError},
    tensor::Tensor,
    Result,
};

/// Hybrid architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Architectural components to combine
    pub components: Vec<ArchitecturalComponent>,
    /// Strategy for fusing components
    pub fusion_strategy: FusionStrategy,
    /// Adaptive switching configuration
    pub adaptive_config: Option<AdaptiveConfig>,
    /// Cross-modal integration settings
    pub cross_modal_config: Option<CrossModalConfig>,
    /// Global architecture parameters
    pub global_params: GlobalParams,
}

impl HybridConfig {
    pub fn builder() -> HybridConfigBuilder {
        HybridConfigBuilder::new()
    }
}

/// Builder for hybrid architecture configuration
pub struct HybridConfigBuilder {
    config: HybridConfig,
}

impl Default for HybridConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: HybridConfig {
                components: Vec::new(),
                fusion_strategy: FusionStrategy::Sequential,
                adaptive_config: None,
                cross_modal_config: None,
                global_params: GlobalParams::default(),
            },
        }
    }

    pub fn add_component(mut self, component: ArchitecturalComponent) -> Self {
        self.config.components.push(component);
        self
    }

    pub fn fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.config.fusion_strategy = strategy;
        self
    }

    pub fn adaptive_config(mut self, config: AdaptiveConfig) -> Self {
        self.config.adaptive_config = Some(config);
        self
    }

    pub fn cross_modal_config(mut self, config: CrossModalConfig) -> Self {
        self.config.cross_modal_config = Some(config);
        self
    }

    pub fn global_params(mut self, params: GlobalParams) -> Self {
        self.config.global_params = params;
        self
    }

    pub fn build(self) -> Result<HybridConfig> {
        if self.config.components.is_empty() {
            return Err(invalid_input(
                "At least one architectural component is required",
            ));
        }
        Ok(self.config)
    }
}

/// Individual architectural components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitecturalComponent {
    /// Transformer component
    Transformer {
        layers: usize,
        hidden_size: usize,
        num_heads: usize,
        variant: TransformerVariant,
    },
    /// Convolutional Neural Network component
    CNN {
        layers: usize,
        channels: usize,
        kernel_size: usize,
        architecture: CNNArchitecture,
    },
    /// Recurrent Neural Network component
    RNN {
        layers: usize,
        hidden_size: usize,
        cell_type: RNNCellType,
        bidirectional: bool,
    },
    /// State-Space Model component
    StateSpace {
        layers: usize,
        state_size: usize,
        model_type: StateSpaceType,
    },
    /// Graph Neural Network component
    GNN {
        layers: usize,
        hidden_size: usize,
        graph_type: GraphType,
    },
    /// Attention mechanism component
    Attention {
        attention_type: AttentionType,
        num_heads: usize,
        key_dim: usize,
    },
    /// Memory component
    Memory {
        memory_type: MemoryType,
        memory_size: usize,
        addressing: AddressingMode,
    },
    /// Custom component
    Custom {
        name: String,
        parameters: HashMap<String, f32>,
        config: HashMap<String, String>,
    },
}

/// Transformer variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformerVariant {
    Standard,
    GPT,
    BERT,
    T5,
    Switch,
    Vision,
    Sparse,
    Linear,
}

/// CNN architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CNNArchitecture {
    ResNet,
    EfficientNet,
    MobileNet,
    DenseNet,
    VGG,
    Inception,
    RegNet,
    ConvNeXt,
}

/// RNN cell types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RNNCellType {
    LSTM,
    GRU,
    RNN,
    IndRNN,
    ConvLSTM,
}

/// State-space model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateSpaceType {
    S4,
    S5,
    Mamba,
    HIPPO,
    Linear,
    Diagonal,
}

/// Graph neural network types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphType {
    GCN,
    GraphSAGE,
    GAT,
    GIN,
    GraphTransformer,
}

/// Attention types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionType {
    MultiHead,
    SparseAttention,
    LocalAttention,
    GlobalAttention,
    CrossAttention,
    SelfAttention,
}

/// Memory types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    NeuralTuringMachine,
    DifferentiableNeuralComputer,
    MemoryAugmentedNetwork,
    ExternalMemory,
    WorkingMemory,
}

/// Memory addressing modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AddressingMode {
    ContentBased,
    LocationBased,
    Hybrid,
    Learned,
}

/// Fusion strategies for combining components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Sequential composition (pipeline)
    Sequential,
    /// Parallel composition with fusion
    Parallel { fusion_method: ParallelFusionMethod },
    /// Hierarchical composition
    Hierarchical { hierarchy_type: HierarchyType },
    /// Adaptive switching between components
    Adaptive {
        switching_criteria: SwitchingCriteria,
    },
    /// Ensemble of components
    Ensemble { combination_method: EnsembleMethod },
    /// Custom fusion strategy
    Custom {
        name: String,
        parameters: HashMap<String, f32>,
    },
}

/// Methods for parallel fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParallelFusionMethod {
    /// Concatenation of outputs
    Concatenation,
    /// Element-wise addition
    Addition,
    /// Element-wise multiplication
    Multiplication,
    /// Learned gating mechanism
    Gating,
    /// Cross-attention fusion
    CrossAttention,
    /// Multi-modal fusion
    MultiModal,
}

/// Hierarchy types for hierarchical fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HierarchyType {
    /// Bottom-up processing
    BottomUp,
    /// Top-down processing
    TopDown,
    /// Bidirectional processing
    Bidirectional,
    /// Pyramid structure
    Pyramid,
}

/// Criteria for adaptive switching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwitchingCriteria {
    /// Input-dependent switching
    InputDependent,
    /// Performance-based switching
    PerformanceBased,
    /// Confidence-based switching
    ConfidenceBased,
    /// Resource-based switching
    ResourceBased,
    /// Learned switching
    Learned,
}

/// Ensemble combination methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    /// Majority voting
    MajorityVoting,
    /// Weighted averaging
    WeightedAveraging,
    /// Stacking
    Stacking,
    /// Boosting
    Boosting,
    /// Bagging
    Bagging,
    /// Dynamic selection
    DynamicSelection,
}

/// Configuration for adaptive behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Enable input-dependent routing
    pub input_routing: bool,
    /// Performance threshold for switching
    pub performance_threshold: f32,
    /// Confidence threshold for decisions
    pub confidence_threshold: f32,
    /// Resource budget constraints
    pub resource_budget: ResourceBudget,
    /// Learning rate for adaptation
    pub adaptation_rate: f32,
}

/// Resource budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudget {
    /// Maximum computation time (ms)
    pub max_compute_time: f32,
    /// Maximum memory usage (MB)
    pub max_memory_mb: f32,
    /// Maximum energy consumption
    pub max_energy: f32,
}

/// Cross-modal integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalConfig {
    /// Modalities to integrate
    pub modalities: Vec<Modality>,
    /// Fusion points in the architecture
    pub fusion_points: Vec<FusionPoint>,
    /// Alignment strategies between modalities
    pub alignment_strategy: AlignmentStrategy,
    /// Shared representation size
    pub shared_repr_size: usize,
}

/// Supported modalities
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum Modality {
    Text,
    Vision,
    Audio,
    Video,
    Sensor,
    Structured,
}

/// Fusion points in the architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionPoint {
    /// Component indices to fuse
    pub component_indices: Vec<usize>,
    /// Fusion method at this point
    pub fusion_method: ParallelFusionMethod,
    /// Layer depth for fusion
    pub fusion_depth: usize,
}

/// Alignment strategies for cross-modal fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignmentStrategy {
    /// Simple concatenation
    Concatenation,
    /// Canonical correlation analysis
    CCA,
    /// Contrastive learning
    Contrastive,
    /// Mutual information maximization
    MutualInformation,
    /// Adversarial alignment
    Adversarial,
}

/// Global parameters for the hybrid architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalParams {
    /// Global activation function
    pub activation: String,
    /// Global normalization strategy
    pub normalization: String,
    /// Global dropout rate
    pub dropout_rate: f32,
    /// Global initialization strategy
    pub initialization: String,
    /// Global optimization settings
    pub optimization: OptimizationParams,
}

impl Default for GlobalParams {
    fn default() -> Self {
        Self {
            activation: "gelu".to_string(),
            normalization: "layer_norm".to_string(),
            dropout_rate: 0.1,
            initialization: "xavier_uniform".to_string(),
            optimization: OptimizationParams::default(),
        }
    }
}

/// Optimization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationParams {
    /// Gradient clipping threshold
    pub grad_clip_threshold: f32,
    /// Mixed precision training
    pub mixed_precision: bool,
    /// Gradient checkpointing
    pub gradient_checkpointing: bool,
    /// Model parallelism strategy
    pub parallelism_strategy: ParallelismStrategy,
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            grad_clip_threshold: 1.0,
            mixed_precision: true,
            gradient_checkpointing: false,
            parallelism_strategy: ParallelismStrategy::DataParallel,
        }
    }
}

/// Model parallelism strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParallelismStrategy {
    DataParallel,
    ModelParallel,
    PipelineParallel,
    TensorParallel,
    HybridParallel,
}

pub mod components;
#[cfg(test)]
mod components_tests;

pub use components::ComponentModule;

/// Main hybrid architecture implementation
pub struct HybridArchitecture {
    /// Configuration
    pub config: HybridConfig,
    /// Component instances
    pub components: Vec<ComponentInstance>,
    /// Fusion layers
    pub fusion_layers: Vec<FusionLayer>,
    /// Adaptive controller
    pub adaptive_controller: Option<AdaptiveController>,
    /// Cross-modal processor
    pub cross_modal_processor: Option<CrossModalProcessor>,
    /// Learned fusion / ensemble parameters, owned so they persist across calls
    pub fusion_operator: components::FusionOperator,
}

/// Individual component instance
#[derive(Debug)]
pub struct ComponentInstance {
    /// Component type
    pub component_type: ArchitecturalComponent,
    /// Component parameters
    pub parameters: ComponentParameters,
    /// Component state
    pub state: ComponentState,
    /// Performance metrics
    pub metrics: ComponentMetrics,
    /// Executable module, built lazily from the first input's feature width
    pub module: Option<ComponentModule>,
}

impl ComponentInstance {
    /// Run the component, instantiating its module on first use.
    ///
    /// The module is built for the observed feature width so a sequential
    /// pipeline keeps its shape, and an unimplemented component reports that
    /// rather than passing its input through unchanged.
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let width = *shape
            .last()
            .ok_or_else(|| invalid_input("hybrid component input has no dimensions"))?;

        if self.module.is_none() {
            self.module = Some(ComponentModule::build(&self.component_type, width)?);
        }

        let module = self
            .module
            .as_ref()
            .ok_or_else(|| invalid_input("hybrid component module was not instantiated"))?;
        module.forward(input)
    }

    /// Number of trainable scalars, or zero before the module is built.
    pub fn parameter_count(&self) -> usize {
        self.module.as_ref().map(|m| m.parameter_count()).unwrap_or(0)
    }
}

/// Component parameters
#[derive(Debug, Clone)]
pub struct ComponentParameters {
    /// Weight tensors
    pub weights: HashMap<String, Tensor>,
    /// Bias tensors
    pub biases: HashMap<String, Tensor>,
    /// Configuration parameters
    pub config_params: HashMap<String, f32>,
}

/// Component state
#[derive(Debug, Clone)]
pub struct ComponentState {
    /// Hidden states
    pub hidden_states: HashMap<String, Tensor>,
    /// Cell states (for RNNs)
    pub cell_states: HashMap<String, Tensor>,
    /// Attention caches
    pub attention_caches: HashMap<String, Tensor>,
    /// Active status
    pub is_active: bool,
}

/// Component performance metrics
#[derive(Debug, Clone)]
pub struct ComponentMetrics {
    /// Inference time
    pub inference_time: f32,
    /// Memory usage
    pub memory_usage: f32,
    /// Accuracy contribution
    pub accuracy_contribution: f32,
    /// Energy consumption
    pub energy_consumption: f32,
}

/// Fusion layer for combining component outputs
#[derive(Debug, Clone)]
pub struct FusionLayer {
    /// Fusion method
    pub fusion_method: ParallelFusionMethod,
    /// Input components
    pub input_components: Vec<usize>,
    /// Fusion parameters
    pub fusion_params: HashMap<String, Tensor>,
    /// Output dimension
    pub output_dim: usize,
}

/// Adaptive controller for dynamic behavior
pub struct AdaptiveController {
    /// Routing network
    pub routing_network: RoutingNetwork,
    /// Performance monitor
    pub performance_monitor: PerformanceMonitor,
    /// Decision history
    pub decision_history: VecDeque<AdaptiveDecision>,
    /// Learning parameters
    pub learning_params: AdaptiveLearningParams,
}

/// Routing network for adaptive switching
#[derive(Debug, Clone)]
pub struct RoutingNetwork {
    /// Router type
    pub router_type: RouterType,
    /// Router parameters
    pub parameters: HashMap<String, Tensor>,
    /// Gating thresholds
    pub gating_thresholds: Vec<f32>,
}

/// Router types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RouterType {
    /// Simple linear router
    Linear,
    /// Multi-layer perceptron router
    MLP,
    /// Attention-based router
    Attention,
    /// Reinforcement learning router
    RL,
}

/// Performance monitor for tracking component performance
pub struct PerformanceMonitor {
    /// Performance history
    pub performance_history: HashMap<usize, VecDeque<f32>>,
    /// Resource usage tracking
    pub resource_usage: HashMap<usize, ResourceUsage>,
    /// Confidence tracking
    pub confidence_tracking: HashMap<usize, VecDeque<f32>>,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU usage
    pub cpu_usage: f32,
    /// Memory usage
    pub memory_usage: f32,
    /// Energy consumption
    pub energy_consumption: f32,
    /// Latency
    pub latency: f32,
}

/// Adaptive decision record
#[derive(Debug, Clone)]
pub struct AdaptiveDecision {
    /// Input characteristics
    pub input_features: Vec<f32>,
    /// Selected component
    pub selected_component: usize,
    /// Decision confidence
    pub confidence: f32,
    /// Performance outcome
    pub performance: f32,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Learning parameters for adaptive behavior
#[derive(Debug, Clone)]
pub struct AdaptiveLearningParams {
    /// Learning rate
    pub learning_rate: f32,
    /// Exploration rate
    pub exploration_rate: f32,
    /// Decay rate
    pub decay_rate: f32,
    /// Update frequency
    pub update_frequency: usize,
}

/// Cross-modal processor for handling multiple modalities
pub struct CrossModalProcessor {
    /// Modality encoders
    pub modality_encoders: HashMap<Modality, ModalityEncoder>,
    /// Alignment network
    pub alignment_network: AlignmentNetwork,
    /// Fusion network
    pub fusion_network: FusionNetwork,
    /// Shared representation space
    pub shared_space: SharedRepresentationSpace,
}

/// Encoder for specific modalities
#[derive(Debug, Clone)]
pub struct ModalityEncoder {
    /// Encoder type
    pub encoder_type: EncoderType,
    /// Encoder parameters
    pub parameters: HashMap<String, Tensor>,
    /// Output dimension
    pub output_dim: usize,
}

/// Encoder types for different modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncoderType {
    TextEncoder,
    VisionEncoder,
    AudioEncoder,
    VideoEncoder,
    SensorEncoder,
    StructuredEncoder,
}

/// Alignment network for cross-modal alignment
#[derive(Debug, Clone)]
pub struct AlignmentNetwork {
    /// Alignment strategy
    pub strategy: AlignmentStrategy,
    /// Alignment parameters
    pub parameters: HashMap<String, Tensor>,
    /// Learned alignments
    pub learned_alignments: HashMap<String, Tensor>,
}

/// Fusion network for combining aligned representations
#[derive(Debug, Clone)]
pub struct FusionNetwork {
    /// Fusion layers
    pub fusion_layers: Vec<FusionLayer>,
    /// Attention mechanisms
    pub attention_mechanisms: Vec<AttentionMechanism>,
    /// Output projection
    pub output_projection: HashMap<String, Tensor>,
}

/// Attention mechanism for fusion
#[derive(Debug, Clone)]
pub struct AttentionMechanism {
    /// Attention type
    pub attention_type: AttentionType,
    /// Parameters
    pub parameters: HashMap<String, Tensor>,
    /// Number of heads
    pub num_heads: usize,
}

/// Shared representation space
#[derive(Debug, Clone)]
pub struct SharedRepresentationSpace {
    /// Dimension of shared space
    pub dimension: usize,
    /// Projection matrices
    pub projection_matrices: HashMap<Modality, Tensor>,
    /// Inverse projection matrices
    pub inverse_projections: HashMap<Modality, Tensor>,
}

impl HybridArchitecture {
    /// Create a new hybrid architecture
    pub fn new(config: HybridConfig) -> Result<Self> {
        let components = Self::initialize_components(&config)?;
        let fusion_layers = Self::create_fusion_layers(&config)?;
        let adaptive_controller = if config.adaptive_config.is_some() {
            Some(Self::create_adaptive_controller(&config)?)
        } else {
            None
        };
        let cross_modal_processor = if config.cross_modal_config.is_some() {
            Some(Self::create_cross_modal_processor(&config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            components,
            fusion_layers,
            adaptive_controller,
            cross_modal_processor,
            fusion_operator: components::FusionOperator::new(),
        })
    }

    /// Forward pass through the hybrid architecture
    pub fn forward(&mut self, inputs: &[Tensor]) -> Result<Tensor> {
        match self.config.fusion_strategy.clone() {
            FusionStrategy::Sequential => self.forward_sequential(inputs),
            FusionStrategy::Parallel { fusion_method } => {
                self.forward_parallel(inputs, &fusion_method)
            },
            FusionStrategy::Hierarchical { hierarchy_type } => {
                self.forward_hierarchical(inputs, &hierarchy_type)
            },
            FusionStrategy::Adaptive { switching_criteria } => {
                self.forward_adaptive(inputs, &switching_criteria)
            },
            FusionStrategy::Ensemble { combination_method } => {
                self.forward_ensemble(inputs, &combination_method)
            },
            FusionStrategy::Custom { name, parameters } => {
                self.forward_custom(inputs, &name, &parameters)
            },
        }
    }

    /// Sequential forward pass
    fn forward_sequential(&mut self, inputs: &[Tensor]) -> Result<Tensor> {
        let mut current_input = inputs[0].clone();

        for component in &mut self.components {
            if !component.state.is_active {
                continue; // Skip inactive components
            }

            let start_time = std::time::Instant::now();

            current_input = component.forward(&current_input)?;

            // Update metrics
            let inference_time = start_time.elapsed().as_secs_f32() * 1000.0;
            component.metrics.inference_time = inference_time;
        }

        Ok(current_input)
    }

    /// Parallel forward pass with fusion
    fn forward_parallel(
        &mut self,
        inputs: &[Tensor],
        fusion_method: &ParallelFusionMethod,
    ) -> Result<Tensor> {
        let mut component_outputs = Vec::new();

        // Process each component in parallel
        for (i, component) in self.components.iter_mut().enumerate() {
            let input = if i < inputs.len() { &inputs[i] } else { &inputs[0] };

            // Inline forward_component logic to avoid borrowing conflict
            if !component.state.is_active {
                component_outputs.push(input.clone()); // Pass-through if inactive
                continue;
            }

            let start_time = std::time::Instant::now();

            let output = component.forward(input)?;

            // Update metrics
            let inference_time = start_time.elapsed().as_secs_f32() * 1000.0;
            component.metrics.inference_time = inference_time;

            component_outputs.push(output);
        }

        // Fuse outputs
        self.fuse_outputs(&component_outputs, fusion_method)
    }

    /// Hierarchical forward pass
    fn forward_hierarchical(
        &mut self,
        inputs: &[Tensor],
        hierarchy_type: &HierarchyType,
    ) -> Result<Tensor> {
        match hierarchy_type {
            HierarchyType::BottomUp => self.forward_bottom_up(inputs),
            HierarchyType::TopDown => self.forward_top_down(inputs),
            HierarchyType::Bidirectional => self.forward_bidirectional(inputs),
            HierarchyType::Pyramid => self.forward_pyramid(inputs),
        }
    }

    /// Adaptive forward pass with component selection
    fn forward_adaptive(
        &mut self,
        inputs: &[Tensor],
        switching_criteria: &SwitchingCriteria,
    ) -> Result<Tensor> {
        if let Some(ref mut controller) = self.adaptive_controller {
            let selected_component = controller.select_component(inputs, switching_criteria)?;
            let component = &mut self.components[selected_component];

            // Inline forward_component logic to avoid borrowing conflict
            let output = if !component.state.is_active {
                inputs[0].clone() // Pass-through if inactive
            } else {
                let start_time = std::time::Instant::now();

                let result = component.forward(&inputs[0])?;

                // Update metrics
                let inference_time = start_time.elapsed().as_secs_f32() * 1000.0;
                component.metrics.inference_time = inference_time;

                result
            };

            // Update performance tracking
            controller.update_performance(selected_component, &output)?;

            Ok(output)
        } else {
            // Fallback to sequential if no adaptive controller
            self.forward_sequential(inputs)
        }
    }

    /// Ensemble forward pass
    fn forward_ensemble(
        &mut self,
        inputs: &[Tensor],
        combination_method: &EnsembleMethod,
    ) -> Result<Tensor> {
        let mut component_outputs = Vec::new();

        // Get outputs from all components
        for component in &mut self.components {
            // Inline forward_component logic to avoid borrowing conflict
            let output = if !component.state.is_active {
                inputs[0].clone() // Pass-through if inactive
            } else {
                let start_time = std::time::Instant::now();

                let result = component.forward(&inputs[0])?;

                // Update metrics
                let inference_time = start_time.elapsed().as_secs_f32() * 1000.0;
                component.metrics.inference_time = inference_time;

                result
            };

            component_outputs.push(output);
        }

        // Combine using ensemble method
        self.combine_ensemble_outputs(&component_outputs, combination_method)
    }

    /// Custom forward pass
    /// Custom fusion strategy.
    ///
    /// A named custom strategy carries no executable body, so this reports the
    /// missing implementation instead of quietly running the sequential
    /// pipeline under a different name.
    fn forward_custom(
        &mut self,
        _inputs: &[Tensor],
        name: &str,
        _parameters: &HashMap<String, f32>,
    ) -> Result<Tensor> {
        Err(not_implemented(format!(
            "FusionStrategy::Custom '{}' has no implementation; the configuration carries only a \
             name and scalar parameters",
            name
        )))
    }

    /// Forward pass through a single component
    #[allow(dead_code)]
    fn forward_component(
        &self,
        component: &mut ComponentInstance,
        input: &Tensor,
    ) -> Result<Tensor> {
        if !component.state.is_active {
            return Ok(input.clone()); // Pass-through if component is inactive
        }

        let start_time = std::time::Instant::now();

        let output = component.forward(input)?;

        // Update metrics
        let inference_time = start_time.elapsed().as_secs_f32() * 1000.0; // Convert to ms
        component.metrics.inference_time = inference_time;

        Ok(output)
    }

    /// Fuse multiple outputs using specified method
    fn fuse_outputs(
        &mut self,
        outputs: &[Tensor],
        fusion_method: &ParallelFusionMethod,
    ) -> Result<Tensor> {
        if outputs.is_empty() {
            return Err(invalid_input("No outputs to fuse"));
        }

        if outputs.len() == 1 {
            return Ok(outputs[0].clone());
        }

        self.fusion_operator.fuse(outputs, fusion_method)
    }

    /// Bottom-up hierarchical processing
    fn forward_bottom_up(&mut self, inputs: &[Tensor]) -> Result<Tensor> {
        // Process from low-level to high-level components
        let mut current_input = inputs[0].clone();

        for component in &mut self.components {
            // Inline forward_component logic to avoid borrowing conflict
            if !component.state.is_active {
                continue; // Skip inactive components
            }

            let start_time = std::time::Instant::now();

            current_input = component.forward(&current_input)?;

            // Update metrics
            let inference_time = start_time.elapsed().as_secs_f32() * 1000.0;
            component.metrics.inference_time = inference_time;
        }

        Ok(current_input)
    }

    /// Top-down hierarchical processing
    fn forward_top_down(&mut self, inputs: &[Tensor]) -> Result<Tensor> {
        // Process from high-level to low-level components
        let mut current_input = inputs[0].clone();

        for component in self.components.iter_mut().rev() {
            // Inline forward_component logic to avoid borrowing conflict
            if !component.state.is_active {
                continue; // Skip inactive components
            }

            let start_time = std::time::Instant::now();

            current_input = component.forward(&current_input)?;

            // Update metrics
            let inference_time = start_time.elapsed().as_secs_f32() * 1000.0;
            component.metrics.inference_time = inference_time;
        }

        Ok(current_input)
    }

    /// Bidirectional hierarchical processing
    fn forward_bidirectional(&mut self, inputs: &[Tensor]) -> Result<Tensor> {
        // Process both bottom-up and top-down, then combine
        let bottom_up = self.forward_bottom_up(inputs)?;
        let top_down = self.forward_top_down(inputs)?;

        // Simple combination - in practice would use learned fusion
        self.fuse_outputs(&[bottom_up, top_down], &ParallelFusionMethod::Addition)
    }

    /// Pyramid hierarchical processing
    fn forward_pyramid(&mut self, inputs: &[Tensor]) -> Result<Tensor> {
        // Implement pyramid-style processing with multiple scales
        let mut pyramid_outputs = Vec::new();

        for (i, component) in self.components.iter_mut().enumerate() {
            let scale_input = &inputs[i % inputs.len()];

            // Inline forward_component logic to avoid borrowing conflict
            let output = if !component.state.is_active {
                scale_input.clone() // Pass-through if inactive
            } else {
                let start_time = std::time::Instant::now();

                let result = component.forward(scale_input)?;

                // Update metrics
                let inference_time = start_time.elapsed().as_secs_f32() * 1000.0;
                component.metrics.inference_time = inference_time;

                result
            };

            pyramid_outputs.push(output);
        }

        // Combine pyramid outputs
        self.fuse_outputs(&pyramid_outputs, &ParallelFusionMethod::Addition)
    }

    /// Combine ensemble outputs.
    ///
    /// The members' recorded accuracy contributions act as the ensemble
    /// weights, so weighted averaging, boosting and dynamic selection all use
    /// measured performance rather than a fixed constant.
    fn combine_ensemble_outputs(
        &mut self,
        outputs: &[Tensor],
        combination_method: &EnsembleMethod,
    ) -> Result<Tensor> {
        let weights: Vec<f32> = self
            .components
            .iter()
            .filter(|component| component.state.is_active)
            .map(|component| component.metrics.accuracy_contribution)
            .collect();
        self.fusion_operator.combine_ensemble(outputs, &weights, combination_method)
    }

    /// Initialize components from configuration
    fn initialize_components(config: &HybridConfig) -> Result<Vec<ComponentInstance>> {
        let mut components = Vec::new();

        for component_config in &config.components {
            let component = ComponentInstance {
                component_type: component_config.clone(),
                parameters: ComponentParameters {
                    weights: HashMap::new(),
                    biases: HashMap::new(),
                    config_params: HashMap::new(),
                },
                state: ComponentState {
                    hidden_states: HashMap::new(),
                    cell_states: HashMap::new(),
                    attention_caches: HashMap::new(),
                    is_active: true,
                },
                metrics: ComponentMetrics {
                    inference_time: 0.0,
                    memory_usage: 0.0,
                    accuracy_contribution: 0.0,
                    energy_consumption: 0.0,
                },
                module: None,
            };
            components.push(component);
        }

        Ok(components)
    }

    /// Create fusion layers from configuration
    fn create_fusion_layers(config: &HybridConfig) -> Result<Vec<FusionLayer>> {
        let mut fusion_layers = Vec::new();

        // Create fusion layers based on strategy
        match &config.fusion_strategy {
            FusionStrategy::Parallel { fusion_method } => {
                let fusion_layer = FusionLayer {
                    fusion_method: fusion_method.clone(),
                    input_components: (0..config.components.len()).collect(),
                    fusion_params: HashMap::new(),
                    output_dim: 512, // Default output dimension
                };
                fusion_layers.push(fusion_layer);
            },
            _ => {
                // Other strategies may not need explicit fusion layers
            },
        }

        Ok(fusion_layers)
    }

    /// Create adaptive controller
    fn create_adaptive_controller(config: &HybridConfig) -> Result<AdaptiveController> {
        let routing_network = RoutingNetwork {
            router_type: RouterType::Linear,
            parameters: HashMap::new(),
            gating_thresholds: vec![0.5; config.components.len()],
        };

        let performance_monitor = PerformanceMonitor {
            performance_history: HashMap::new(),
            resource_usage: HashMap::new(),
            confidence_tracking: HashMap::new(),
        };

        let learning_params = AdaptiveLearningParams {
            learning_rate: 0.001,
            exploration_rate: 0.1,
            decay_rate: 0.99,
            update_frequency: 100,
        };

        Ok(AdaptiveController {
            routing_network,
            performance_monitor,
            decision_history: VecDeque::new(),
            learning_params,
        })
    }

    /// Create cross-modal processor
    fn create_cross_modal_processor(config: &HybridConfig) -> Result<CrossModalProcessor> {
        let cross_modal_config = config.cross_modal_config.as_ref().ok_or_else(|| {
            TrustformersError::invalid_config(
                "cross_modal_config required for cross-modal processing".to_string(),
            )
        })?;

        let mut modality_encoders = HashMap::new();
        for modality in &cross_modal_config.modalities {
            let encoder = ModalityEncoder {
                encoder_type: match modality {
                    Modality::Text => EncoderType::TextEncoder,
                    Modality::Vision => EncoderType::VisionEncoder,
                    Modality::Audio => EncoderType::AudioEncoder,
                    Modality::Video => EncoderType::VideoEncoder,
                    Modality::Sensor => EncoderType::SensorEncoder,
                    Modality::Structured => EncoderType::StructuredEncoder,
                },
                parameters: HashMap::new(),
                output_dim: cross_modal_config.shared_repr_size,
            };
            modality_encoders.insert(modality.clone(), encoder);
        }

        let alignment_network = AlignmentNetwork {
            strategy: cross_modal_config.alignment_strategy.clone(),
            parameters: HashMap::new(),
            learned_alignments: HashMap::new(),
        };

        let fusion_network = FusionNetwork {
            fusion_layers: Vec::new(),
            attention_mechanisms: Vec::new(),
            output_projection: HashMap::new(),
        };

        let shared_space = SharedRepresentationSpace {
            dimension: cross_modal_config.shared_repr_size,
            projection_matrices: HashMap::new(),
            inverse_projections: HashMap::new(),
        };

        Ok(CrossModalProcessor {
            modality_encoders,
            alignment_network,
            fusion_network,
            shared_space,
        })
    }

    /// Get number of components
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Get component metrics
    pub fn get_component_metrics(&self, component_index: usize) -> Option<&ComponentMetrics> {
        self.components.get(component_index).map(|c| &c.metrics)
    }

    /// Enable/disable a component
    pub fn set_component_active(&mut self, component_index: usize, active: bool) -> Result<()> {
        if let Some(component) = self.components.get_mut(component_index) {
            component.state.is_active = active;
            Ok(())
        } else {
            Err(invalid_input(format!(
                "Invalid component index: {}",
                component_index
            )))
        }
    }

    /// Get architecture summary
    pub fn get_architecture_summary(&self) -> ArchitectureSummary {
        let total_parameters = self.estimate_total_parameters();
        let memory_usage = self.estimate_memory_usage();
        let computational_complexity = self.estimate_computational_complexity();

        ArchitectureSummary {
            num_components: self.components.len(),
            fusion_strategy: format!("{:?}", self.config.fusion_strategy),
            total_parameters,
            memory_usage,
            computational_complexity,
            component_types: self
                .components
                .iter()
                .map(|c| format!("{:?}", c.component_type))
                .collect(),
        }
    }

    /// Total trainable scalars across every instantiated component.
    ///
    /// Components are built lazily, so this counts zero for a component that
    /// has not seen an input yet rather than inventing a figure.
    fn estimate_total_parameters(&self) -> usize {
        self.components.iter().map(|component| component.parameter_count()).sum()
    }

    /// Parameter memory in megabytes, at four bytes per scalar.
    fn estimate_memory_usage(&self) -> f32 {
        self.estimate_total_parameters() as f32 * 4.0 / 1_000_000.0
    }

    /// Multiply-accumulate estimate: two FLOPs per parameter per token.
    fn estimate_computational_complexity(&self) -> f64 {
        self.estimate_total_parameters() as f64 * 2.0
    }
}

/// Architecture summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureSummary {
    pub num_components: usize,
    pub fusion_strategy: String,
    pub total_parameters: usize,
    pub memory_usage: f32,
    pub computational_complexity: f64,
    pub component_types: Vec<String>,
}

impl fmt::Display for ArchitectureSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HybridArchitecture {{ components: {}, strategy: {}, params: {}M, memory: {:.1}MB }}",
            self.num_components,
            self.fusion_strategy,
            self.total_parameters / 1_000_000,
            self.memory_usage
        )
    }
}

impl AdaptiveController {
    /// Select component based on input and criteria
    pub fn select_component(
        &mut self,
        inputs: &[Tensor],
        criteria: &SwitchingCriteria,
    ) -> Result<usize> {
        match criteria {
            SwitchingCriteria::InputDependent => self.select_input_dependent(inputs),
            SwitchingCriteria::PerformanceBased => self.select_performance_based(),
            SwitchingCriteria::ConfidenceBased => self.select_confidence_based(inputs),
            SwitchingCriteria::ResourceBased => self.select_resource_based(),
            SwitchingCriteria::Learned => self.select_learned(inputs),
        }
    }

    /// Route on measured input statistics.
    ///
    /// The component index is chosen from the input's mean activation energy,
    /// so different inputs genuinely reach different components.
    fn select_input_dependent(&self, inputs: &[Tensor]) -> Result<usize> {
        let component_count = self.performance_monitor.performance_history.len().max(1);
        let Some(input) = inputs.first() else {
            return Err(invalid_input(
                "input-dependent routing needs at least one input",
            ));
        };
        let energy = components::activation_energy(input)?;
        // Map the energy onto the component range through a bounded transform.
        let normalised = (energy / (1.0 + energy)).clamp(0.0, 1.0);
        Ok(((normalised * component_count as f32) as usize).min(component_count - 1))
    }

    fn select_performance_based(&self) -> Result<usize> {
        // Select component with best historical performance
        let mut best_component = 0;
        let mut best_performance = 0.0;

        for (component_id, performance_history) in &self.performance_monitor.performance_history {
            if let Some(&last_performance) = performance_history.back() {
                if last_performance > best_performance {
                    best_performance = last_performance;
                    best_component = *component_id;
                }
            }
        }

        Ok(best_component)
    }

    /// Route on the softmax margin of the input.
    ///
    /// A confident input (large top-1/top-2 margin) is sent to the first
    /// component; an ambiguous one is escalated to a later component.
    fn select_confidence_based(&self, inputs: &[Tensor]) -> Result<usize> {
        let component_count = self.performance_monitor.performance_history.len().max(1);
        let Some(input) = inputs.first() else {
            return Err(invalid_input(
                "confidence-based routing needs at least one input",
            ));
        };
        let confidence = components::prediction_confidence(input)?.clamp(0.0, 1.0);
        let index = ((1.0 - confidence) * component_count as f32) as usize;
        Ok(index.min(component_count - 1))
    }

    fn select_resource_based(&self) -> Result<usize> {
        // Select component with lowest resource usage
        let mut best_component = 0;
        let mut lowest_usage = f32::INFINITY;

        for (component_id, resource_usage) in &self.performance_monitor.resource_usage {
            let total_usage = resource_usage.cpu_usage
                + resource_usage.memory_usage
                + resource_usage.energy_consumption;
            if total_usage < lowest_usage {
                lowest_usage = total_usage;
                best_component = *component_id;
            }
        }

        Ok(best_component)
    }

    /// Learned routing.
    ///
    /// A trained routing network is not part of this crate's public surface, so
    /// this reports the missing capability rather than silently always
    /// selecting component 0.
    fn select_learned(&self, _inputs: &[Tensor]) -> Result<usize> {
        Err(not_implemented(
            "SwitchingCriteria::Learned needs a trained routing network, which the hybrid \
             architecture does not own; use InputDependent, ConfidenceBased, PerformanceBased \
             or ResourceBased routing",
        ))
    }

    /// Record a component's measured performance for this step.
    ///
    /// The score is the softmax margin of the component's own output, a
    /// computed confidence in `[0, 1]` — not a fixed constant.
    pub fn update_performance(&mut self, component_id: usize, output: &Tensor) -> Result<()> {
        let performance = components::prediction_confidence(output)?;

        self.performance_monitor
            .performance_history
            .entry(component_id)
            .or_default()
            .push_back(performance);

        // Keep only recent history
        if let Some(history) = self.performance_monitor.performance_history.get_mut(&component_id) {
            while history.len() > 100 {
                history.pop_front();
            }
        }

        Ok(())
    }
}
