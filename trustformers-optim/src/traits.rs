//! Advanced optimizer trait hierarchy for TrustformeRS.
//!
//! This module extends the base `Optimizer` trait from `trustformers-core` with
//! additional specialized traits for different categories of optimizers, providing
//! better organization and extensibility.
//!
//! # Trait Hierarchy
//!
//! ```text
//! Optimizer (from trustformers-core)
//!     │
//!     ├── StatefulOptimizer
//!     │   ├── MomentumOptimizer
//!     │   │   ├── AdaptiveMomentumOptimizer  (Adam, AdamW, etc.)
//!     │   │   └── ClassicalMomentumOptimizer (SGD with momentum)
//!     │   └── SecondOrderOptimizer (L-BFGS, Newton-CG, etc.)
//!     │
//!     ├── DistributedOptimizer
//!     │   ├── GradientCompressionOptimizer
//!     │   ├── FederatedOptimizer
//!     │   └── AsyncOptimizer
//!     │
//!     ├── HardwareOptimizer
//!     │   ├── SIMDOptimizer
//!     │   ├── GPUOptimizer
//!     │   └── EdgeOptimizer
//!     │
//!     └── MetaOptimizer
//!         ├── LookaheadOptimizer
//!         ├── ScheduledOptimizer
//!         └── CompositeOptimizer
//! ```

use crate::common::StateMemoryStats;
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Extended optimizer trait with state management capabilities.
///
/// This trait builds on the base `Optimizer` trait to provide standardized
/// state management, serialization, and configuration access.
pub trait StatefulOptimizer: Optimizer {
    /// The configuration type for this optimizer.
    type Config: Clone + Send + Sync;

    /// The state type used by this optimizer.
    type State: Send + Sync;

    /// Gets a reference to the optimizer's configuration.
    fn config(&self) -> &Self::Config;

    /// Gets a reference to the optimizer's internal state.
    fn state(&self) -> &Self::State;

    /// Gets a mutable reference to the optimizer's internal state.
    fn state_mut(&mut self) -> &mut Self::State;

    /// Saves the optimizer state to a dictionary for checkpointing.
    fn state_dict(&self) -> Result<HashMap<String, Tensor>>;

    /// Loads optimizer state from a dictionary during checkpoint restoration.
    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()>;

    /// Gets memory usage statistics for this optimizer.
    fn memory_usage(&self) -> StateMemoryStats;

    /// Resets the optimizer state (useful for training restarts).
    fn reset_state(&mut self);

    /// Returns the number of parameters being optimized.
    fn num_parameters(&self) -> usize;
}

/// Trait for optimizers that use momentum-based updates.
///
/// This includes both classical momentum (SGD) and adaptive momentum (Adam family).
pub trait MomentumOptimizer: StatefulOptimizer {
    /// Gets the momentum decay coefficient (β1 in Adam, momentum in SGD).
    fn momentum_coeff(&self) -> f32;

    /// Sets the momentum decay coefficient.
    fn set_momentum_coeff(&mut self, coeff: f32);

    /// Gets the current momentum buffers (for debugging/analysis).
    fn momentum_buffers(&self) -> &HashMap<String, Vec<f32>>;

    /// Clears all momentum buffers (useful for fine-tuning).
    fn clear_momentum(&mut self);
}

/// Trait for adaptive momentum optimizers (Adam, AdamW, RAdam, etc.).
///
/// These optimizers maintain both first and second moment estimates.
pub trait AdaptiveMomentumOptimizer: MomentumOptimizer {
    /// Gets the second moment decay coefficient (β2 in Adam).
    fn variance_coeff(&self) -> f32;

    /// Sets the second moment decay coefficient.
    fn set_variance_coeff(&mut self, coeff: f32);

    /// Gets the epsilon value for numerical stability.
    fn epsilon(&self) -> f32;

    /// Sets the epsilon value.
    fn set_epsilon(&mut self, eps: f32);

    /// Gets the current variance buffers (for debugging/analysis).
    fn variance_buffers(&self) -> &HashMap<String, Vec<f32>>;

    /// Clears variance buffers.
    fn clear_variance(&mut self);

    /// Applies bias correction to momentum and variance estimates.
    fn apply_bias_correction(&self, momentum: f32, variance: f32, step: usize) -> (f32, f32);
}

/// Trait for classical momentum optimizers (SGD variants).
pub trait ClassicalMomentumOptimizer: MomentumOptimizer {
    /// Gets the dampening factor.
    fn dampening(&self) -> f32;

    /// Sets the dampening factor.
    fn set_dampening(&mut self, dampening: f32);

    /// Whether Nesterov momentum is enabled.
    fn nesterov(&self) -> bool;

    /// Enables or disables Nesterov momentum.
    fn set_nesterov(&mut self, nesterov: bool);
}

/// Trait for second-order optimization methods.
///
/// These optimizers use curvature information (Hessian approximations).
pub trait SecondOrderOptimizer: StatefulOptimizer {
    /// The type used to represent curvature information.
    type CurvatureInfo;

    /// Updates the curvature approximation with new gradient information.
    fn update_curvature(&mut self, gradients: &[Tensor]) -> Result<()>;

    /// Gets the current curvature approximation.
    fn curvature_info(&self) -> &Self::CurvatureInfo;

    /// Applies the inverse Hessian approximation to compute search direction.
    fn apply_inverse_hessian(&self, gradient: &Tensor) -> Result<Tensor>;

    /// Gets the maximum number of curvature pairs stored (for L-BFGS).
    fn history_size(&self) -> usize;
}

/// Trait for distributed optimization capabilities.
///
/// Provides interfaces for gradient synchronization and distributed training.
pub trait DistributedOptimizer: Optimizer {
    /// The communicator type used for distributed operations.
    type Communicator;

    /// Performs all-reduce operation on gradients.
    fn all_reduce_gradients(&mut self, gradients: &mut [Tensor]) -> Result<()>;

    /// Broadcasts parameters from rank 0 to all other ranks.
    fn broadcast_parameters(&mut self, parameters: &mut [Tensor]) -> Result<()>;

    /// Gets the current rank in the distributed group.
    fn rank(&self) -> usize;

    /// Gets the total number of ranks in the distributed group.
    fn world_size(&self) -> usize;

    /// Synchronizes optimizer state across all ranks.
    fn sync_state(&mut self) -> Result<()>;
}

/// Trait for optimizers with gradient compression capabilities.
pub trait GradientCompressionOptimizer: DistributedOptimizer {
    /// The compression method used.
    type CompressionMethod;

    /// Compresses gradients before communication.
    fn compress_gradients(&self, gradients: &[Tensor]) -> Result<Vec<u8>>;

    /// Decompresses received gradient data.
    fn decompress_gradients(&self, data: &[u8]) -> Result<Vec<Tensor>>;

    /// Gets the compression ratio achieved.
    fn compression_ratio(&self) -> f32;

    /// Sets the compression parameters.
    fn set_compression_config(&mut self, config: Self::CompressionMethod);
}

/// Trait for federated learning optimizers.
pub trait FederatedOptimizer: DistributedOptimizer {
    /// Client information type.
    type ClientInfo;

    /// Aggregates model updates from multiple clients.
    fn aggregate_updates(
        &mut self,
        updates: &[Tensor],
        clients: &[Self::ClientInfo],
    ) -> Result<Tensor>;

    /// Selects clients for the next round of training.
    fn select_clients(
        &self,
        available_clients: &[Self::ClientInfo],
        num_clients: usize,
    ) -> Vec<usize>;

    /// Applies differential privacy to updates.
    fn apply_differential_privacy(&mut self, update: &mut Tensor) -> Result<()>;
}

/// Trait for asynchronous optimization methods.
pub trait AsyncOptimizer: DistributedOptimizer {
    /// Applies delayed gradients with staleness compensation.
    fn apply_delayed_gradients(&mut self, gradients: &[Tensor], staleness: usize) -> Result<()>;

    /// Gets the maximum allowed staleness.
    fn max_staleness(&self) -> usize;

    /// Sets the staleness compensation method.
    fn set_staleness_compensation(&mut self, method: StalenessCompensation);
}

/// Staleness compensation methods for asynchronous optimization.
#[derive(Debug, Clone, Copy)]
pub enum StalenessCompensation {
    /// No compensation for staleness.
    None,
    /// Linear scaling by staleness factor.
    Linear,
    /// Exponential decay based on staleness.
    Exponential,
    /// Polynomial scaling with configurable degree.
    Polynomial(f32),
}

/// Trait for hardware-specific optimizer optimizations.
pub trait HardwareOptimizer: Optimizer {
    /// The target hardware type.
    type HardwareTarget;

    /// Optimizes the optimizer for specific hardware.
    fn optimize_for_hardware(&mut self, target: Self::HardwareTarget) -> Result<()>;

    /// Gets hardware utilization statistics.
    fn hardware_utilization(&self) -> HardwareStats;

    /// Checks if the optimizer is compatible with the current hardware.
    fn is_hardware_compatible(&self) -> bool;
}

/// Hardware utilization statistics.
#[derive(Debug, Clone)]
pub struct HardwareStats {
    /// Memory bandwidth utilization (0.0 to 1.0).
    pub memory_bandwidth_utilization: f32,
    /// Compute utilization (0.0 to 1.0).
    pub compute_utilization: f32,
    /// Cache hit rate (0.0 to 1.0).
    pub cache_hit_rate: f32,
    /// FLOPS per second achieved.
    pub flops_per_second: f64,
}

/// Trait for SIMD-optimized operations.
pub trait SIMDOptimizer: HardwareOptimizer {
    /// The SIMD instruction set being used.
    type SIMDType;

    /// Checks if SIMD operations are available.
    fn simd_available(&self) -> bool;

    /// Gets the SIMD vector width.
    fn vector_width(&self) -> usize;

    /// Applies SIMD-optimized parameter updates.
    fn simd_update(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) -> Result<()>;
}

/// Trait for GPU-accelerated optimizers.
pub trait GPUOptimizer: HardwareOptimizer {
    /// The GPU compute capability.
    type ComputeCapability;

    /// Transfers optimizer state to GPU.
    fn to_gpu(&mut self) -> Result<()>;

    /// Transfers optimizer state to CPU.
    fn to_cpu(&mut self) -> Result<()>;

    /// Launches GPU kernels for parameter updates.
    fn gpu_update(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) -> Result<()>;

    /// Gets GPU memory usage.
    fn gpu_memory_usage(&self) -> GPUMemoryStats;
}

/// GPU memory usage statistics.
#[derive(Debug, Clone)]
pub struct GPUMemoryStats {
    /// Total GPU memory in bytes.
    pub total_memory: usize,
    /// Used GPU memory in bytes.
    pub used_memory: usize,
    /// Available GPU memory in bytes.
    pub available_memory: usize,
    /// Memory usage by optimizer state.
    pub optimizer_memory: usize,
}

/// Trait for edge device optimized optimizers.
pub trait EdgeOptimizer: HardwareOptimizer {
    /// Power consumption statistics.
    type PowerStats;

    /// Optimizes for low power consumption.
    fn optimize_for_power(&mut self) -> Result<()>;

    /// Gets current power consumption statistics.
    fn power_stats(&self) -> Self::PowerStats;

    /// Reduces precision to save memory and power.
    fn reduce_precision(&mut self, bits: u8) -> Result<()>;
}

/// Trait for meta-optimizers that wrap other optimizers.
pub trait MetaOptimizer: Optimizer {
    /// The base optimizer type being wrapped.
    type BaseOptimizer: Optimizer;

    /// Gets a reference to the base optimizer.
    fn base_optimizer(&self) -> &Self::BaseOptimizer;

    /// Gets a mutable reference to the base optimizer.
    fn base_optimizer_mut(&mut self) -> &mut Self::BaseOptimizer;

    /// Applies the meta-optimization strategy.
    fn apply_meta_strategy(
        &mut self,
        parameters: &mut [Tensor],
        gradients: &[Tensor],
    ) -> Result<()>;
}

/// Trait for lookahead meta-optimizers.
pub trait LookaheadOptimizer: MetaOptimizer {
    /// Gets the lookahead step size (α).
    fn lookahead_alpha(&self) -> f32;

    /// Sets the lookahead step size.
    fn set_lookahead_alpha(&mut self, alpha: f32);

    /// Gets the lookahead update frequency (k).
    fn lookahead_k(&self) -> usize;

    /// Sets the lookahead update frequency.
    fn set_lookahead_k(&mut self, k: usize);

    /// Gets the slow weights (for debugging).
    fn slow_weights(&self) -> &HashMap<String, Vec<f32>>;
}

/// Trait for scheduled optimizers with learning rate scheduling.
pub trait ScheduledOptimizer: Optimizer {
    /// The scheduler type.
    type Scheduler;

    /// Gets a reference to the scheduler.
    fn scheduler(&self) -> &Self::Scheduler;

    /// Gets a mutable reference to the scheduler.
    fn scheduler_mut(&mut self) -> &mut Self::Scheduler;

    /// Updates the learning rate based on the scheduler.
    fn update_lr(&mut self) -> Result<()>;

    /// Gets the current scheduled learning rate.
    fn current_lr(&self) -> f32;
}

/// Trait for composite optimizers that combine multiple optimization strategies.
pub trait CompositeOptimizer: Optimizer {
    /// The component optimizer types.
    type Components;

    /// Gets references to all component optimizers.
    fn components(&self) -> &Self::Components;

    /// Gets mutable references to all component optimizers.
    fn components_mut(&mut self) -> &mut Self::Components;

    /// Applies updates from all component optimizers.
    fn apply_composite_update(
        &mut self,
        parameters: &mut [Tensor],
        gradients: &[Tensor],
    ) -> Result<()>;

    /// Gets the weight assigned to each component.
    fn component_weights(&self) -> Vec<f32>;

    /// Sets the weights for each component.
    fn set_component_weights(&mut self, weights: Vec<f32>) -> Result<()>;
}

/// Optimizer factory trait for creating optimizers with different configurations.
pub trait OptimizerFactory {
    /// The optimizer type produced by this factory.
    type Optimizer: Optimizer;

    /// The configuration type for the optimizer.
    type Config;

    /// Creates a new optimizer with the given configuration.
    fn create(&self, config: Self::Config) -> Result<Self::Optimizer>;

    /// Lists all available optimizer variants.
    fn available_variants(&self) -> Vec<&'static str>;

    /// Creates an optimizer by name with default configuration.
    fn create_by_name(&self, name: &str) -> Result<Self::Optimizer>;
}

/// Trait for optimizers that can be serialized and restored.
pub trait SerializableOptimizer: Optimizer {
    /// Serializes the optimizer to bytes.
    fn serialize(&self) -> Result<Vec<u8>>;

    /// Deserializes an optimizer from bytes.
    fn deserialize(data: &[u8]) -> Result<Self>
    where
        Self: Sized;

    /// Gets the serialization format version.
    fn version(&self) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_staleness_compensation() {
        let compensation = StalenessCompensation::Linear;
        match compensation {
            StalenessCompensation::Linear => assert!(true),
            _ => assert!(false),
        }
    }

    #[test]
    fn test_hardware_stats() {
        let stats = HardwareStats {
            memory_bandwidth_utilization: 0.8,
            compute_utilization: 0.9,
            cache_hit_rate: 0.95,
            flops_per_second: 1e12,
        };

        assert_eq!(stats.memory_bandwidth_utilization, 0.8);
        assert_eq!(stats.compute_utilization, 0.9);
        assert_eq!(stats.cache_hit_rate, 0.95);
        assert_eq!(stats.flops_per_second, 1e12);
    }

    #[test]
    fn test_gpu_memory_stats() {
        let stats = GPUMemoryStats {
            total_memory: 16 * 1024 * 1024 * 1024,    // 16 GB
            used_memory: 8 * 1024 * 1024 * 1024,      // 8 GB
            available_memory: 8 * 1024 * 1024 * 1024, // 8 GB
            optimizer_memory: 1 * 1024 * 1024 * 1024, // 1 GB
        };

        assert_eq!(stats.total_memory, 16 * 1024 * 1024 * 1024);
        assert_eq!(
            stats.used_memory + stats.available_memory,
            stats.total_memory
        );
        assert!(stats.optimizer_memory <= stats.used_memory);
    }
}
