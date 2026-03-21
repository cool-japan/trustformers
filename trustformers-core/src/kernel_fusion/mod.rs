//! Kernel fusion automation system for TrustformeRS
//!
//! This module provides comprehensive kernel fusion capabilities to automatically
//! identify and fuse compatible operations for optimal performance. The system
//! includes pattern matching, constraint verification, and multi-backend kernel
//! generation.
//!
//! # Organization
//!
//! The kernel fusion system is organized into several submodules:
//!
//! - [`operation_types`]: Defines operation types, fusion patterns, and constraints
//! - [`graph`]: Computation graph structures and tensor information
//! - [`kernel`]: Fused kernel representation and implementation types
//! - [`performance`]: Performance tracking and device characteristics
//! - [`engine`]: Main fusion engine with pattern matching and kernel generation
//! - [`memory`]: Memory access pattern analysis for optimization
//!
//! # Usage
//!
//! ```rust
//! use trustformers_core::kernel_fusion::{KernelFusionEngine, ComputationGraph};
//!
//! let engine = KernelFusionEngine::new();
//! let graph = ComputationGraph::new();
//!
//! // Analyze graph for fusion opportunities
//! let opportunities = engine.analyze_graph(&graph)?;
//!
//! // Fuse operations
//! for opportunity in opportunities {
//!     if opportunity.constraints_satisfied {
//!         let fused_kernel = engine.fuse_operations(&graph, &opportunity)?;
//!         println!("Generated fused kernel: {}", fused_kernel.name);
//!     }
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod engine;
pub mod graph;
pub mod kernel;
pub mod memory;
pub mod operation_types;
pub mod performance;

// Re-export main types for convenience
pub use engine::{FusionOpportunity, KernelFusionEngine};
pub use graph::{
    ComputationGraph, DataType, Device, GraphNode, MemoryLayout, NodeMetadata, TensorInfo,
};
pub use kernel::{FusedKernel, KernelImplementation};
pub use memory::MemoryAccessPattern;
pub use operation_types::{FusionConstraint, FusionPattern, OperationType};
pub use performance::{
    DeviceCharacteristics, FusionStatistics, OperationCost, PerformanceDatabase,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_fusion_engine_creation() {
        let engine = KernelFusionEngine::new();
        assert!(!engine.patterns.is_empty());
        assert!(!engine.constraints.is_empty());
    }

    #[test]
    fn test_computation_graph_creation() {
        let graph = ComputationGraph::new();
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
        assert!(graph.execution_order.is_empty());
    }

    #[test]
    fn test_shape_broadcasting() {
        let engine = KernelFusionEngine::new();

        // Compatible shapes
        assert!(engine.shapes_broadcastable(&[1, 4], &[3, 4]));
        assert!(engine.shapes_broadcastable(&[3, 1], &[3, 4]));
        assert!(engine.shapes_broadcastable(&[1], &[3, 4]));

        // Incompatible shapes
        assert!(!engine.shapes_broadcastable(&[2, 4], &[3, 4]));
        assert!(!engine.shapes_broadcastable(&[3, 2], &[3, 4]));
    }

    #[test]
    fn test_fusion_opportunity_creation() {
        let opportunity = FusionOpportunity {
            pattern: FusionPattern::ElementWiseChain(vec![OperationType::Add, OperationType::ReLU]),
            node_ids: vec!["node1".to_string(), "node2".to_string()],
            estimated_benefit: 1.5,
            constraints_satisfied: true,
        };

        assert_eq!(opportunity.node_ids.len(), 2);
        assert!(opportunity.estimated_benefit > 1.0);
        assert!(opportunity.constraints_satisfied);
    }

    #[test]
    fn test_memory_access_pattern() {
        let sequential = MemoryAccessPattern::Sequential;
        assert!(sequential.is_cache_friendly());
        assert!(sequential.supports_vectorization());
        assert_eq!(sequential.bandwidth_utilization(), 1.0);

        let strided = MemoryAccessPattern::Strided {
            strides: vec![2, 1],
        };
        assert!(strided.supports_vectorization()); // Unit stride in last dimension

        let random = MemoryAccessPattern::Random;
        assert!(!random.is_cache_friendly());
        assert!(!random.supports_vectorization());
    }

    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::F32.size_bytes(), 4);
        assert_eq!(DataType::F16.size_bytes(), 2);
        assert_eq!(DataType::BF16.size_bytes(), 2);
        assert_eq!(DataType::I32.size_bytes(), 4);
        assert_eq!(DataType::I8.size_bytes(), 1);
        assert_eq!(DataType::U8.size_bytes(), 1);
        assert_eq!(DataType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_kernel_implementation_platform() {
        let cuda_impl = KernelImplementation::CUDA("kernel code".to_string());
        assert_eq!(cuda_impl.platform(), "CUDA");

        let cpu_impl = KernelImplementation::CPU("kernel code".to_string());
        assert_eq!(cpu_impl.platform(), "CPU");

        assert_eq!(cuda_impl.code(), "kernel code");
    }
}
