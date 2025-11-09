//! CPU/GPU Load Balancer Module
//!
//! This module organizes all components of the CPU/GPU load balancing system.

pub mod balancer;
pub mod config;
pub mod monitoring;
pub mod power;
pub mod scheduler;
pub mod types;

// Re-export main types and interfaces
pub use balancer::CpuGpuLoadBalancer;
pub use config::{
    LoadBalancerConfig, LoadBalancingStrategy, PowerEfficiencyMode, PowerScalingFactors,
};
pub use monitoring::{
    LoadBalancerStats, PerformanceMonitor, PowerEfficiencyReport, ProcessorStats,
};
pub use power::{PowerEfficiencyMetrics, PowerManager, PowerOptimizationStrategy, PowerReport};
pub use scheduler::{TaskQueueManager, TaskScheduler};
pub use types::{
    ComputeTask, ExecutionStatus, LoadBalancerEvent, MemoryPattern, PerformanceHistory,
    ProcessorResource, ProcessorType, TaskExecutionResult, TaskPriority, TaskType,
};
