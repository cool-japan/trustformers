//! ZeRO (Zero Redundancy Optimizer) Implementation for TrustformeRS
//!
//! ZeRO is a memory-efficient training technique that partitions optimizer states,
//! gradients, and parameters across devices to reduce memory usage while maintaining
//! training efficiency.
//!
//! Implements three stages:
//! - Stage 1: Partition optimizer states
//! - Stage 2: Partition optimizer states + gradients
//! - Stage 3: Partition optimizer states + gradients + parameters

pub mod zero_optimizer;
pub mod zero_stage1;
pub mod zero_stage2;
pub mod zero_stage3;
pub mod zero_utils;

pub use zero_optimizer::{ZeROConfig, ZeROOptimizer, ZeROStage};
pub use zero_stage1::ZeROStage1;
pub use zero_stage2::ZeROStage2;
pub use zero_stage3::ZeROStage3;
pub use zero_utils::{
    all_gather_gradients, gather_parameters, partition_gradients, partition_parameters,
    reduce_scatter_gradients, GradientBuffer, ParameterGroup, ParameterPartition, ZeROState,
};

/// ZeRO optimization stages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeROImplementationStage {
    /// Stage 1: Partition optimizer states only
    Stage1,
    /// Stage 2: Partition optimizer states + gradients
    Stage2,
    /// Stage 3: Partition optimizer states + gradients + parameters
    Stage3,
}

/// Memory statistics for ZeRO optimization
#[derive(Debug, Clone)]
pub struct ZeROMemoryStats {
    /// Memory saved by partitioning optimizer states
    pub optimizer_memory_saved: usize,
    /// Memory saved by partitioning gradients
    pub gradient_memory_saved: usize,
    /// Memory saved by partitioning parameters
    pub parameter_memory_saved: usize,
    /// Total memory saved
    pub total_memory_saved: usize,
    /// Memory overhead from communication buffers
    pub communication_overhead: usize,
}

impl Default for ZeROMemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeROMemoryStats {
    pub fn new() -> Self {
        Self {
            optimizer_memory_saved: 0,
            gradient_memory_saved: 0,
            parameter_memory_saved: 0,
            total_memory_saved: 0,
            communication_overhead: 0,
        }
    }

    pub fn update_totals(&mut self) {
        self.total_memory_saved =
            self.optimizer_memory_saved + self.gradient_memory_saved + self.parameter_memory_saved;
    }
}
