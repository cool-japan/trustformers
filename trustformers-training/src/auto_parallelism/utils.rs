//! Free-standing utilities for automatic parallelism selection.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use trustformers_core::Model;

use super::config::{HardwareConstraints, ModelConstraints, PerformanceRequirements};
use super::strategy_types::ParallelismStrategy;

/// Estimate model memory requirements
pub fn estimate_model_memory(constraints: &ModelConstraints) -> u64 {
    let param_memory = constraints.num_parameters * 4;
    let gradient_memory = param_memory;
    let optimizer_memory = param_memory * 2;
    param_memory + gradient_memory + optimizer_memory
}

/// Check if strategy meets performance requirements
pub fn meets_requirements(
    strategy: &ParallelismStrategy,
    requirements: &PerformanceRequirements,
) -> bool {
    if let Some(max_time) = requirements.max_training_time {
        if strategy.expected_performance.time_per_step > max_time {
            return false;
        }
    }
    if let Some(min_throughput) = requirements.min_throughput {
        if strategy.expected_performance.throughput < min_throughput {
            return false;
        }
    }
    if let Some(max_memory) = requirements.max_memory_per_device {
        if strategy.expected_performance.memory_per_device > max_memory {
            return false;
        }
    }
    if let Some(max_comm_overhead) = requirements.max_communication_overhead {
        if strategy.expected_performance.communication_overhead > max_comm_overhead {
            return false;
        }
    }
    if let Some(min_efficiency) = requirements.min_efficiency {
        if strategy.expected_performance.efficiency < min_efficiency {
            return false;
        }
    }
    true
}

/// Create hardware constraints from system information
pub fn detect_hardware_constraints() -> Result<HardwareConstraints> {
    Ok(HardwareConstraints::default())
}

/// Create model constraints from model architecture
pub fn analyze_model_constraints<M: Model>(_model: &M) -> Result<ModelConstraints> {
    Ok(ModelConstraints::default())
}
