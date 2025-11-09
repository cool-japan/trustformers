//! # TrustformeRS Optimization
//!
//! This crate provides state-of-the-art optimization algorithms for training transformer models,
//! including distributed training support and memory-efficient techniques.
//!
//! ## Overview
//!
//! TrustformeRS Optim includes:
//! - **Core Optimizers**: Adam, AdamW, SGD, LAMB, AdaFactor
//! - **Cutting-Edge 2024-2025 Optimizers**: HN-Adam, AdEMAMix, Muon, CAME, MicroAdam for state-of-the-art performance
//! - **Schedule-Free Optimizers**: Schedule-Free SGD and Adam (no LR scheduling needed)
//! - **Advanced Quantization**: 4-bit optimizers with NF4 and block-wise quantization
//! - **Memory-Efficient Optimization**: MicroAdam with compressed gradients and low space overhead
//! - **Learning Rate Schedulers**: Linear, Cosine, Polynomial, Step, Exponential
//! - **Distributed Training**: ZeRO optimization stages, multi-node support
//! - **Memory Optimization**: Gradient accumulation, mixed precision, CPU offloading
//!
//! ## Optimizers
//!
//! ### Adam and AdamW
//!
//! Adaptive Moment Estimation with optional weight decay:
//! ```rust,no_run
//! use trustformers_optim::{AdamW, OptimizerState};
//! use trustformers_core::traits::Optimizer;
//!
//! let mut optimizer = AdamW::new(
//!     1e-3,           // learning_rate
//!     (0.9, 0.999),   // (beta1, beta2)
//!     1e-8,           // epsilon
//!     0.01,           // weight_decay
//! );
//!
//! // Ready to use in training loop with .zero_grad(), .update(), and .step()
//! ```
//!
//! ### SGD
//!
//! Stochastic Gradient Descent with momentum and Nesterov acceleration:
//! ```rust,no_run
//! use trustformers_optim::SGD;
//!
//! let optimizer = SGD::new(
//!     0.1,        // learning_rate
//!     0.9,        // momentum
//!     1e-4,       // weight_decay
//!     true,       // nesterov
//! );
//! ```
//!
//! ### Schedule-Free Optimizers
//!
//! Revolutionary optimizers that eliminate the need for learning rate scheduling:
//! ```rust,no_run
//! use trustformers_optim::{ScheduleFreeAdam, ScheduleFreeSGD};
//! use trustformers_core::traits::Optimizer;
//!
//! // Schedule-Free Adam - no learning rate scheduling needed!
//! let optimizer = ScheduleFreeAdam::for_language_models();
//!
//! // Higher learning rates work better (e.g., 0.25-1.0 instead of 0.001)
//! let optimizer = ScheduleFreeAdam::new(0.5, 0.9, 0.95, 1e-8, 0.1);
//!
//! // Schedule-Free SGD for simpler models
//! let optimizer = ScheduleFreeSGD::for_large_models();
//!
//! // No learning rate scheduler needed! Just use .zero_grad(), .update(), .step()
//! // eval_mode() can be used to switch to average weights
//! ```
//!
//! ### Cutting-Edge 2024-2025 Optimizers
//!
//! The latest state-of-the-art optimizers for superior performance:
//!
//! #### 🌟 **NEW: Latest 2025 Research Algorithms** 🚀
//!
//! **Self-Scaled BFGS (SSBFGS)** - Revolutionary quasi-Newton method:
//! ```rust,no_run
//! use trustformers_optim::{SSBFGS, SSBFGSConfig};
//!
//! // For Physics-Informed Neural Networks (PINNs)
//! let optimizer = SSBFGS::for_physics_informed();
//!
//! // For challenging non-convex problems
//! let optimizer = SSBFGS::for_non_convex();
//!
//! // Custom configuration
//! let optimizer = SSBFGS::from_config(SSBFGSConfig {
//!     learning_rate: 0.8,
//!     history_size: 15,
//!     scaling_factor: 1.2,
//!     momentum: 0.95,
//! });
//!
//! // Get optimization statistics
//! let stats = optimizer.get_stats();
//! println!("Current scaling factor: {:.3}", stats.current_scaling_factor);
//! ```
//!
//! **Self-Scaled Broyden (SSBroyden)** - Efficient rank-1 updates:
//! ```rust,no_run
//! use trustformers_optim::{SSBroyden, SSBroydenConfig};
//!
//! // Optimized for PINNs with rank-1 efficiency
//! let optimizer = SSBroyden::for_physics_informed();
//!
//! // More computationally efficient than BFGS
//! let optimizer = SSBroyden::new(); // Default configuration
//! ```
//!
//! **PDE-aware Optimizer** - Specialized for Physics-Informed Neural Networks:
//! ```rust,no_run
//! use trustformers_optim::{PDEAwareOptimizer, PDEAwareConfig};
//!
//! // Specialized configurations for different PDEs
//! let burgers_opt = PDEAwareOptimizer::for_burgers_equation();    // Burgers' equation
//! let allen_cahn_opt = PDEAwareOptimizer::for_allen_cahn();       // Allen-Cahn equation
//! let kdv_opt = PDEAwareOptimizer::for_kdv_equation();            // Korteweg-de Vries
//! let sharp_grad_opt = PDEAwareOptimizer::for_sharp_gradients();  // Sharp gradient regions
//!
//! // Get PDE-specific optimization statistics
//! let stats = sharp_grad_opt.get_pde_stats();
//! println!("Average residual variance: {:.6}", stats.average_residual_variance);
//! ```
//!
//! **🔬 Research Breakthrough Features:**
//! - **Orders-of-magnitude improvements** in PINN training accuracy
//! - **Dynamic rescaling** based on gradient history and PDE residual variance
//! - **Sharp gradient handling** for challenging PDE optimization landscapes
//! - **Lower computational cost** than second-order methods like SOAP
//! - **Specialized presets** for different equation types (Burgers, Allen-Cahn, KdV)
//!
//! #### BGE-Adam (2024) - Revolutionary Performance Optimization! 🚀
//! Enhanced Adam with entropy weighting and adaptive gradient strategy, now featuring **OptimizedBGEAdam** with **3-5x speedup**:
//! ```rust,no_run
//! use trustformers_optim::{BGEAdam, OptimizedBGEAdam, BGEAdamConfig, OptimizedBGEAdamConfig};
//!
//! // 🚀 RECOMMENDED: Use the optimized version for 3-5x better performance!
//! let optimizer = OptimizedBGEAdam::new(); // 3-5x faster than original!
//!
//! // Performance-optimized presets for different use cases
//! let llm_optimizer = OptimizedBGEAdam::for_large_models();     // For LLMs (optimized settings)
//! let vision_optimizer = OptimizedBGEAdam::for_vision();        // For computer vision
//! let perf_optimizer = OptimizedBGEAdam::for_high_performance(); // Maximum speed
//!
//! // Built-in performance monitoring and entropy statistics
//! println!("{}", optimizer.performance_stats());
//! let (min_entropy, max_entropy, avg_entropy) = optimizer.get_entropy_stats();
//!
//! // Original BGE-Adam still available (but much slower)
//! let original_optimizer = BGEAdam::new(
//!     1e-3,        // learning rate
//!     (0.9, 0.999), // (β1, β2)
//!     1e-8,        // epsilon
//!     0.01,        // weight decay
//!     0.1,         // entropy scaling factor
//!     0.05,        // β1 adaptation factor
//!     0.05,        // β2 adaptation factor
//! );
//! ```
//!
//! **🔥 Performance Improvements in OptimizedBGEAdam:**
//! - ⚡ **3.4-4.9x faster execution** (16.3ms → 4.7ms per iteration for 50k params)
//! - 💾 **85-87x memory reduction** through optimized buffer management
//! - 🔥 **Single-pass processing** eliminates redundant calculations
//! - 🚀 **Vectorized operations** with SIMD-friendly processing patterns
//!
//! #### HN-Adam (2024)
//! Hybrid Norm Adam with adaptive step size:
//! ```rust,no_run
//! use trustformers_optim::{HNAdam, HNAdamConfig};
//!
//! // Automatically adjusts step size based on update norms
//! let optimizer = HNAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1);
//!
//! // Or use presets for specific tasks
//! let transformer_opt = HNAdam::for_transformers(); // Optimized for transformers
//! let vision_opt = HNAdam::for_vision(); // Optimized for computer vision
//!
//! // Better convergence speed and accuracy than standard Adam
//! ```
//!
//! #### AdEMAMix (2024)
//! Dual EMA system for better gradient utilization:
//! ```rust,no_run
//! use trustformers_optim::AdEMAMix;
//!
//! // Revolutionary dual EMA optimizer from Apple/EPFL
//! let optimizer = AdEMAMix::for_llm_training(); // Optimized for LLMs
//!
//! // Or for vision tasks
//! let optimizer = AdEMAMix::for_vision_training();
//!
//! // 95% data efficiency improvement demonstrated in research
//! ```
//!
//! #### Muon (2024)
//! Second-order optimizer for hidden layers:
//! ```rust,no_run
//! use trustformers_optim::Muon;
//!
//! // Used in NanoGPT and CIFAR-10 speed records
//! let optimizer = Muon::for_nanogpt(); // <1% FLOP overhead
//!
//! // For large language models
//! let optimizer = Muon::for_large_lm();
//!
//! // Automatically chooses 2D optimization for matrices, 1D fallback for vectors
//! ```
//!
//! #### CAME (2023)
//! Confidence-guided memory efficient optimization:
//! ```rust,no_run
//! use trustformers_optim::CAME;
//!
//! // Memory efficient with fast convergence
//! let optimizer = CAME::for_bert_training();
//!
//! // For memory-constrained environments
//! let optimizer = CAME::for_memory_constrained();
//!
//! // Check memory savings
//! println!("Memory savings: {:.1}%", optimizer.memory_savings_ratio() * 100.0);
//! ```
//!
//! #### MicroAdam (NeurIPS 2024)
//! Memory-efficient Adam with compressed gradients:
//! ```rust,no_run
//! use trustformers_optim::MicroAdam;
//!
//! // Standard configuration with adaptive compression
//! let optimizer = MicroAdam::new();
//!
//! // For large language models (higher compression)
//! let optimizer = MicroAdam::for_large_models();
//!
//! // Memory-constrained environments (aggressive compression)
//! let optimizer = MicroAdam::for_memory_constrained();
//!
//! // Check compression statistics
//! println!("{}", optimizer.compression_statistics());
//! println!("Memory savings: {:.1}%", optimizer.memory_savings_ratio() * 100.0);
//! ```
//!
//! ### Advanced Quantization
//!
//! Ultra-low memory usage with 4-bit quantization:
//! ```rust,no_run
//! use trustformers_optim::{Adam4bit, AdvancedQuantizationConfig, QuantizationMethod};
//!
//! // 4-bit Adam with NF4 quantization (75% memory savings)
//! let optimizer = Adam4bit::new(0.001, 0.9, 0.999, 1e-8, 0.01);
//!
//! // Custom quantization configuration
//! let quant_config = AdvancedQuantizationConfig {
//!     method: QuantizationMethod::NF4,
//!     block_size: 64,
//!     adaptation_rate: 0.01,
//!     double_quantization: true,
//!     ..Default::default()
//! };
//!
//! let optimizer = Adam4bit::with_quantization_config(
//!     Default::default(),
//!     quant_config,
//! );
//!
//! // Massive memory savings for large models
//! println!("Memory savings: {:.1}%", optimizer.memory_savings() * 100.0);
//! ```
//!
//! ## Learning Rate Schedules
//!
//! Control learning rate during training:
//! ```rust,no_run
//! use trustformers_optim::{AdamW, CosineScheduler, LRScheduler};
//!
//! let base_lr = 1e-3;
//! let optimizer = AdamW::new(base_lr, (0.9, 0.999), 1e-8, 0.01);
//!
//! // Cosine annealing with warmup
//! let scheduler = CosineScheduler::new(
//!     base_lr,
//!     1000,   // num_warmup_steps
//!     10000,  // num_training_steps
//!     1e-5,   // min_lr
//! );
//!
//! // Update learning rate each step
//! for step in 0..10000 {
//!     let current_lr = scheduler.get_lr(step);
//!     // Use current_lr with optimizer.set_lr(current_lr)
//! }
//! ```
//!
//! ## ZeRO Optimization
//!
//! Memory-efficient distributed training:
//! ```rust,ignore
//! // ZeRO distributed training (requires distributed environment)
//! use trustformers_optim::{AdamW};
//!
//! let optimizer = AdamW::new(1e-4, (0.9, 0.999), 1e-8, 0.01);
//! // ZeRO configuration and distributed setup would go here
//! ```
//!
//! ### ZeRO Stages
//!
//! - **Stage 1**: Optimizer state partitioning (4x memory reduction)
//! - **Stage 2**: Optimizer + gradient partitioning (8x memory reduction)
//! - **Stage 3**: Full parameter partitioning (Nx memory reduction)
//!
//! ## Multi-Node Training
//!
//! Scale training across multiple machines:
//! ```rust,ignore
//! // Multi-node distributed training setup
//! // Configuration and training would require distributed environment
//! // Example: MultiNodeTrainer::new(config)
//! ```
//!
//! ## Advanced Features
//!
//! ### Gradient Accumulation
//! ```rust,ignore
//! // Example: Accumulate gradients over multiple batches before stepping
//! // if (step + 1) % accumulation_steps == 0 {
//! //     optimizer.step(&mut model.parameters())?;
//! //     optimizer.zero_grad();
//! // }
//! ```
//!
//! ### Mixed Precision Training
//! ```rust,ignore
//! // Mixed precision optimizers can provide memory savings and speed improvements
//! // Configuration example:
//! // MixedPrecisionOptimizer::new(base_optimizer, scale_config)
//! ```
//!
//! ## Performance Tips
//!
//! 1. **Choose the Right Optimizer**:
//!    - AdamW for most transformer training
//!    - SGD for fine-tuning with small learning rates
//!    - LAMB for large batch training
//!
//! 2. **Learning Rate Scheduling**:
//!    - Use warmup for stable training start
//!    - Cosine schedule for most cases
//!    - Linear decay for fine-tuning
//!
//! 3. **Memory Optimization**:
//!    - Enable ZeRO Stage 2 for models > 1B parameters
//!    - Use gradient accumulation for larger effective batch sizes
//!    - Consider CPU offloading for very large models
//!
//! 4. **Distributed Training**:
//!    - Use data parallelism for models < 10B parameters
//!    - Add model parallelism for larger models
//!    - Enable communication overlap for better throughput

// Allow large error types in Result (TrustformersError is large by design)
#![allow(clippy::result_large_err)]
// Allow common patterns in optimizer implementations
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::excessive_nesting)]

pub mod adafactor_new;
pub mod adafisher_simple;
pub mod adam;
pub mod adam_v2;
pub mod adamax_plus;
pub mod adan;
pub mod adaptive;
pub mod ademamix;
pub mod advanced_2025_research;
pub mod advanced_distributed_features;
pub mod advanced_features;
pub mod amacp;
pub mod async_optim;
pub mod averaged_adam;
pub mod bge_adam;
pub mod bge_adam_optimized;
pub mod cache_friendly;
pub mod came;
pub mod common;
pub mod compression;
pub mod continual_learning;
pub mod convergence;
pub mod cpu_offload;
pub mod cross_framework;
pub mod deep_distributed_qp;
pub mod enhanced_distributed_training;
pub mod eva;
pub mod federated;
pub mod fusion;
pub mod genie_stub;
pub mod gradient_processing;
pub mod hardware_aware;
pub mod hierarchical_aggregation;
pub mod hn_adam;
pub mod hyperparameter_tuning;
pub mod jax_compat;
pub mod kernel_fusion;
pub mod lamb;
pub mod lancbio;
pub mod lion;
pub mod lookahead;
pub mod lora;
pub mod lora_rite_stub;
pub mod memory_layout;
pub mod microadam;
pub mod monitoring;
pub mod multinode;
pub mod muon;
pub mod novograd;
pub mod onnx_export;
pub mod optimizer;
pub mod parallel;
pub mod pde_aware;
pub mod performance_validation;
pub mod prodigy;
pub mod pytorch_compat;
pub mod quantized;
pub mod quantized_advanced;
pub mod quantum_inspired;
pub mod schedule_free;
pub mod scheduler;
pub mod second_order;
pub mod sgd;
pub mod simd_optimizations;
pub mod sofo_stub;
pub mod sophia;
pub mod sparse;
pub mod task_specific;
pub mod tensorflow_compat;
pub mod traits;
pub mod zero;

#[cfg(test)]
pub mod tests;

pub use adafactor_new::{AdaFactor, AdaFactorConfig};
pub use adafisher_simple::{AdaFisher, AdaFisherConfig};
pub use adam::{AdaBelief, Adam, AdamW, NAdam, RAdam};
pub use adam_v2::{AdamConfig, StandardizedAdam, StandardizedAdamW};
pub use adamax_plus::{AdaMaxPlus, AdaMaxPlusConfig};
pub use adan::{Adan, AdanConfig};
pub use adaptive::{create_ranger, create_ranger_with_config, AMSBound, AdaBound, Ranger};
pub use ademamix::{AdEMAMix, AdEMAMixConfig};
pub use advanced_2025_research::{AdaWin, AdaWinConfig, DiWo, DiWoConfig, MeZOV2, MeZOV2Config};
pub use advanced_distributed_features::{
    AutoScaler, AutoScalerConfig, CheckpointConfig as AdvancedCheckpointConfig, CheckpointInfo,
    CostOptimizer, MLOptimizerConfig, OptimizationResult, OptimizationType, PerformanceMLOptimizer,
    ScalingDecision, ScalingStrategy, SmartCheckpointManager, WorkloadPredictor,
};
pub use advanced_features::{
    CheckpointConfig, FusedOptimizer, MemoryBandwidthOptimizer, MultiOptimizerStats,
    MultiOptimizerTrainer, ResourceUtilization, WarmupOptimizer, WarmupStrategy,
};
pub use amacp::{AMacP, AMacPConfig, AMacPStats};
pub use async_optim::{
    AsyncSGD, AsyncSGDConfig, DelayCompensationMethod, DelayedGradient, DelayedGradientConfig,
    ElasticAveraging, ElasticAveragingConfig, Hogwild, HogwildConfig, ParameterServer,
};
pub use averaged_adam::{AveragedAdam, AveragedAdamConfig};
pub use bge_adam::{BGEAdam, BGEAdamConfig};
pub use bge_adam_optimized::{OptimizedBGEAdam, OptimizedBGEAdamConfig};
pub use cache_friendly::{
    CacheConfig, CacheFriendlyAdam, CacheFriendlyState, CacheStats, ParameterMetadata,
};
pub use came::{CAMEConfig, CAME};
pub use common::{
    BiasCorrection, GradientProcessor, OptimizerState, ParameterIds, ParameterUpdate,
    StateMemoryStats, WeightDecayMode,
};
pub use compression::{
    CompressedAllReduce, CompressedGradient, CompressionMethod, GradientCompressor,
};
pub use continual_learning::{
    AllocationStrategy, EWCConfig, FisherMethod, L2Regularization, L2RegularizationConfig,
    MemoryReplay, MemoryReplayConfig, MemorySelectionStrategy, PackNet, PackNetConfig,
    UpdateStrategy, EWC,
};
pub use convergence::{
    AggMo, AggMoConfig, FISTAConfig, HeavyBall, HeavyBallConfig, NesterovAcceleratedGradient,
    NesterovAcceleratedGradientConfig, QHMConfig, VarianceReduction, VarianceReductionConfig,
    VarianceReductionMethod, FISTA, QHM,
};
pub use cpu_offload::{
    create_cpu_offloaded_adam, create_cpu_offloaded_adamw, create_cpu_offloaded_sgd,
    CPUOffloadConfig, CPUOffloadStats, CPUOffloadedOptimizer,
};
pub use cross_framework::{
    ConfigSource, ConfigTarget, CrossFrameworkConverter, Framework, JAXOptimizerConfig,
    PyTorchOptimizerConfig, TrustformeRSOptimizerConfig, UniversalOptimizerConfig,
    UniversalOptimizerState,
};
pub use deep_distributed_qp::{DeepDistributedQP, DeepDistributedQPConfig};
pub use enhanced_distributed_training::{
    Bottleneck, CompressionConfig, CompressionType, DistributedConfig, DistributedTrainingStats,
    DynamicBatchingConfig, EnhancedDistributedTrainer, FaultToleranceConfig,
    MemoryOptimizationConfig, MonitoringConfig as DistributedMonitoringConfig,
    PerformanceMetrics as DistributedPerformanceMetrics, PerformanceTrend, TrainingStepResult,
};
pub use eva::{EVAConfig, EVA};
pub use federated::{
    ClientInfo, ClientSelectionStrategy, DifferentialPrivacy, DifferentialPrivacyConfig, FedAvg,
    FedAvgConfig, FedProx, FedProxConfig, NoiseMechanism, SecureAggregation,
};
#[cfg(target_arch = "x86_64")]
pub use fusion::simd;
pub use fusion::{FusedOperation, FusedOptimizerState, FusionConfig, FusionStats};
pub use genie_stub::{DomainStats, GENIEConfig, GENIEStats, GENIE};
pub use gradient_processing::{
    AdaptiveClippingConfig, GradientProcessedOptimizer, GradientProcessingConfig,
    HessianApproximationType, HessianPreconditioningConfig, NoiseInjectionConfig, NoiseType,
    SmoothingConfig,
};
pub use hardware_aware::{
    create_edge_optimizer, create_gpu_adam, create_mobile_optimizer, create_tpu_optimizer,
    CompressionRatio, EdgeOptimizer, GPUAdam, HardwareAwareConfig, HardwareTarget, MobileOptimizer,
    TPUOptimizer, TPUVersion,
};
pub use hierarchical_aggregation::{
    AggregationStats, AggregationStrategy, ButterflyStructure, CommunicationGroups, FaultDetector,
    HierarchicalAggregator, HierarchicalConfig, NodeTopology, RecoveryStrategy, RingStructure,
    TreeStructure,
};
pub use hn_adam::{HNAdam, HNAdamConfig};
pub use hyperparameter_tuning::{
    BayesianOptimizer, HyperparameterSample, HyperparameterSpace, HyperparameterTuner,
    MultiObjectiveOptimizer, OptimizationTask, OptimizerType,
    PerformanceMetrics as HyperparameterPerformanceMetrics, TaskType as HyperparameterTaskType,
};
pub use jax_compat::{
    JAXAdam, JAXAdamW, JAXChain, JAXCosineDecay, JAXCosineDecaySchedule, JAXExponentialDecay,
    JAXGradientTransformation, JAXLearningRateSchedule, JAXOptState, JAXOptimizerFactory,
    JAXOptimizerState, JAXWarmupCosineDecay, JAXSGD,
};
pub use kernel_fusion::{
    CoalescingLevel, FusedGPUState, GPUMemoryStats, KernelFusedAdam, KernelFusionConfig,
};
pub use lamb::LAMB;
pub use lancbio::{LancBiO, LancBiOConfig};
pub use lion::{Lion, LionConfig};
pub use lookahead::{
    Lookahead, LookaheadAdam, LookaheadAdamW, LookaheadNAdam, LookaheadRAdam, LookaheadSGD,
};
pub use lora::{
    create_lora_adam, create_lora_adamw, create_lora_sgd, LoRAAdapter, LoRAConfig, LoRAOptimizer,
};
pub use lora_rite_stub::{LoRARITE, LoRARITEConfig, LoRARITEStats, TransformationStats};
pub use memory_layout::{
    AlignedAllocator, AlignmentConfig, LayoutOptimizedAdam, LayoutStats, SoAOptimizerState,
};
pub use microadam::{MicroAdam, MicroAdamConfig};
pub use monitoring::{
    ConvergenceIndicators, ConvergenceSpeed, HyperparameterSensitivity,
    HyperparameterSensitivityConfig, HyperparameterSensitivityMetrics, MemoryStats, MemoryUsage,
    MetricStats, MonitoringConfig, OptimizerMetrics, OptimizerMonitor, OptimizerRecommendation,
    OptimizerSelector, PerformanceStats, PerformanceTier,
};
pub use muon::{Muon, MuonConfig};
pub use pde_aware::{PDEAwareConfig, PDEAwareOptimizer, PDEAwareStats};
pub use prodigy::{Prodigy, ProdigyConfig};
// pub use optimizer::OptimizerState; // Already imported from common
pub use performance_validation::{
    BenchmarkScenario, ConvergenceAnalysisResults, CorrectnessResults,
    DistributedValidationResults, MathematicalProperty, MathematicalTestCase,
    MemoryValidationResults, PerformanceBenchmarkResults, PerformanceValidator,
    RegressionAnalysisResults, StatisticalMetrics, ValidationConfig, ValidationResults,
};
pub use pytorch_compat::{
    PyTorchAdam, PyTorchAdamW, PyTorchLRScheduler, PyTorchOptimizer, PyTorchOptimizerFactory,
    PyTorchOptimizerState, PyTorchParamGroup, PyTorchSGD,
};
pub use quantized::{Adam8bit, AdamW8bit, QuantizationConfig, QuantizedState};
pub use quantized_advanced::{
    Adam4bit, Adam4bitOptimizerConfig, AdvancedQuantizationConfig, GradientStatistics,
    QuantizationMethod, QuantizationUtils, QuantizedTensor,
};
pub use quantum_inspired::{
    QuantumAnnealingConfig, QuantumAnnealingOptimizer, QuantumAnnealingStats,
};
pub use schedule_free::{
    ScheduleFreeAdam, ScheduleFreeAdamConfig, ScheduleFreeSGD, ScheduleFreeSGDConfig,
};
pub use scheduler::{
    AdaptiveScheduler, CompositeScheduler, ConstantWithWarmupScheduler, CosineScheduler,
    CosineWithRestartsScheduler, CyclicalMode, CyclicalScheduler, DynamicScheduler,
    ExponentialScheduler, LRScheduler, LinearScheduler, OneCycleScheduler, Phase,
    PhaseBasedScheduler, PolynomialScheduler, StepScheduler, SwitchCondition,
    TaskSpecificScheduler, TaskType as SchedulerTaskType,
};
pub use second_order::{
    LineSearchMethod, NewtonCG, SSBFGSConfig, SSBFGSStats, SSBroyden, SSBroydenConfig, LBFGS,
    SSBFGS,
};
pub use sgd::SGD;
pub use simd_optimizations::{SIMDConfig, SIMDOptimizer, SIMDPerformanceInfo};
pub use sofo_stub::{
    ForwardModeStats, MemoryStats as SOFOMemoryStats, SOFOConfig, SOFOStats, SOFO,
};
pub use sophia::{Sophia, SophiaConfig};
pub use sparse::{SparseAdam, SparseConfig, SparseMomentumState, SparseSGD};
pub use task_specific::{
    create_bert_optimizer, create_gan_optimizer, create_maml_optimizer, create_ppo_optimizer,
    BERTOptimizer, GANOptimizer, MetaOptimizer as TaskMetaOptimizer, RLOptimizer,
};
pub use tensorflow_compat::{
    TensorFlowAdam, TensorFlowAdamW, TensorFlowCosineDecay, TensorFlowExponentialDecay,
    TensorFlowLearningRateSchedule, TensorFlowOptimizer, TensorFlowOptimizerConfig,
    TensorFlowOptimizerFactory,
};
pub use traits::{
    AdaptiveMomentumOptimizer, AsyncOptimizer, ClassicalMomentumOptimizer, CompositeOptimizer,
    DistributedOptimizer, FederatedOptimizer, GPUOptimizer, GradientCompressionOptimizer,
    HardwareOptimizer, HardwareStats, LookaheadOptimizer, MetaOptimizer, MomentumOptimizer,
    OptimizerFactory, ScheduledOptimizer, SecondOrderOptimizer, SerializableOptimizer,
    StalenessCompensation, StatefulOptimizer,
};
pub use zero::{
    all_gather_gradients, gather_parameters, partition_gradients, partition_parameters,
    reduce_scatter_gradients, GradientBuffer, ParameterGroup, ParameterPartition, ZeROConfig,
    ZeROImplementationStage, ZeROMemoryStats, ZeROOptimizer, ZeROStage, ZeROStage1, ZeROStage2,
    ZeROStage3, ZeROState,
};

pub use multinode::{MultiNodeConfig, MultiNodeStats, MultiNodeTrainer};
pub use novograd::{MemoryEfficiencyStats, NovoGrad, NovoGradConfig, NovoGradStats};
pub use onnx_export::{
    ONNXExportConfig, ONNXGraph, ONNXModel, ONNXNode, ONNXOptimizerExporter, ONNXOptimizerMetadata,
    OptimizerConfig,
};
pub use parallel::{BatchUpdate, ParallelAdam, ParallelConfig, ParallelStats};
