//! CPU/GPU Load Balancer Configuration
//!
//! This module contains all configuration structures and types for the CPU/GPU
//! load balancing system, including power management, NUMA awareness, and
//! performance optimization settings.

use serde::{Deserialize, Serialize};
use trustformers_core::numa_optimization::NumaStrategy;

/// CPU/GPU load balancer configuration
///
/// Comprehensive configuration for load balancing between CPU and GPU resources
/// with support for power efficiency, NUMA awareness, and adaptive strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Enable load balancing
    pub enabled: bool,

    /// CPU utilization threshold for GPU offloading (0-1)
    pub cpu_threshold: f32,

    /// GPU utilization threshold for CPU fallback (0-1)
    pub gpu_threshold: f32,

    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,

    /// Monitoring interval in seconds
    pub monitoring_interval_seconds: u64,

    /// Task queue capacity
    pub task_queue_capacity: usize,

    /// CPU pool size
    pub cpu_pool_size: usize,

    /// Enable adaptive routing
    pub enable_adaptive_routing: bool,

    /// Performance history window size
    pub performance_history_window: usize,

    /// Minimum task size for GPU (in operations)
    pub min_gpu_task_size: usize,

    /// Maximum CPU memory per task (in bytes)
    pub max_cpu_memory_per_task: usize,

    /// Enable power efficiency mode
    pub enable_power_efficiency: bool,

    /// CPU power efficiency weight (0-1)
    pub cpu_power_weight: f32,

    /// GPU power efficiency weight (0-1)
    pub gpu_power_weight: f32,

    /// Power efficiency mode settings
    pub power_efficiency_mode: PowerEfficiencyMode,

    /// Maximum power consumption limit (watts)
    pub max_power_consumption: f32,

    /// Power scaling factors
    pub power_scaling_factors: PowerScalingFactors,

    /// Enable NUMA-aware allocation
    pub enable_numa_awareness: bool,

    /// NUMA allocation strategy
    pub numa_strategy: NumaStrategy,

    /// Preferred NUMA nodes for CPU tasks
    pub preferred_cpu_numa_nodes: Vec<u32>,

    /// Preferred NUMA nodes for GPU tasks
    pub preferred_gpu_numa_nodes: Vec<u32>,

    /// Enable NUMA memory binding
    pub numa_memory_binding: bool,

    /// NUMA affinity threshold for task assignment
    pub numa_affinity_threshold: f32,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cpu_threshold: 0.8,
            gpu_threshold: 0.9,
            strategy: LoadBalancingStrategy::Adaptive,
            monitoring_interval_seconds: 1,
            task_queue_capacity: 10000,
            cpu_pool_size: num_cpus::get(),
            enable_adaptive_routing: true,
            performance_history_window: 100,
            min_gpu_task_size: 1000,
            max_cpu_memory_per_task: 2 * 1024 * 1024 * 1024, // 2GB
            enable_power_efficiency: false,
            cpu_power_weight: 0.3,
            gpu_power_weight: 0.7,
            power_efficiency_mode: PowerEfficiencyMode::Balanced,
            max_power_consumption: 1000.0, // 1000W default limit
            power_scaling_factors: PowerScalingFactors::default(),
            enable_numa_awareness: false,
            numa_strategy: NumaStrategy::LocalNode,
            preferred_cpu_numa_nodes: Vec::new(),
            preferred_gpu_numa_nodes: Vec::new(),
            numa_memory_binding: false,
            numa_affinity_threshold: 0.8,
        }
    }
}

/// Power efficiency modes
///
/// Different operating modes that balance performance and power consumption.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PowerEfficiencyMode {
    /// Maximum performance, power is secondary
    PerformanceFirst,

    /// Balanced performance and power
    Balanced,

    /// Maximum power efficiency, performance is secondary
    PowerFirst,

    /// Adaptive based on current conditions
    Adaptive,

    /// Custom configuration
    Custom,
}

impl Default for PowerEfficiencyMode {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Power scaling factors for different operating modes
///
/// Fine-grained control over power management parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerScalingFactors {
    /// CPU frequency scaling factor (0.5-1.0)
    pub cpu_frequency_scaling: f32,

    /// GPU power limit scaling factor (0.5-1.0)
    pub gpu_power_limit_scaling: f32,

    /// Memory frequency scaling factor (0.7-1.0)
    pub memory_frequency_scaling: f32,

    /// Voltage scaling factor (0.8-1.0)
    pub voltage_scaling: f32,

    /// Idle power optimization factor
    pub idle_power_optimization: f32,

    /// Dynamic voltage and frequency scaling (DVFS) enabled
    pub dvfs_enabled: bool,

    /// Sleep states utilization
    pub sleep_states_enabled: bool,
}

impl Default for PowerScalingFactors {
    fn default() -> Self {
        Self {
            cpu_frequency_scaling: 1.0,
            gpu_power_limit_scaling: 1.0,
            memory_frequency_scaling: 1.0,
            voltage_scaling: 1.0,
            idle_power_optimization: 0.8,
            dvfs_enabled: true,
            sleep_states_enabled: true,
        }
    }
}

/// Load balancing strategies
///
/// Different algorithms for distributing tasks between CPU and GPU resources.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round robin between CPU and GPU
    RoundRobin,

    /// Least loaded processor
    LeastLoaded,

    /// Adaptive based on task characteristics
    Adaptive,

    /// Performance optimized
    PerformanceOptimized,

    /// Power efficiency optimized
    PowerEfficient,

    /// Memory optimized
    MemoryOptimized,

    /// Latency optimized
    LatencyOptimized,

    /// Throughput optimized
    ThroughputOptimized,

    /// NUMA-aware allocation and processing
    NumaAware,
}

/// Configuration validation
impl LoadBalancerConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.cpu_threshold < 0.0 || self.cpu_threshold > 1.0 {
            return Err("CPU threshold must be between 0.0 and 1.0".to_string());
        }

        if self.gpu_threshold < 0.0 || self.gpu_threshold > 1.0 {
            return Err("GPU threshold must be between 0.0 and 1.0".to_string());
        }

        if self.cpu_power_weight < 0.0 || self.cpu_power_weight > 1.0 {
            return Err("CPU power weight must be between 0.0 and 1.0".to_string());
        }

        if self.gpu_power_weight < 0.0 || self.gpu_power_weight > 1.0 {
            return Err("GPU power weight must be between 0.0 and 1.0".to_string());
        }

        if self.numa_affinity_threshold < 0.0 || self.numa_affinity_threshold > 1.0 {
            return Err("NUMA affinity threshold must be between 0.0 and 1.0".to_string());
        }

        if self.max_power_consumption <= 0.0 {
            return Err("Maximum power consumption must be positive".to_string());
        }

        if self.task_queue_capacity == 0 {
            return Err("Task queue capacity must be greater than 0".to_string());
        }

        if self.cpu_pool_size == 0 {
            return Err("CPU pool size must be greater than 0".to_string());
        }

        Ok(())
    }

    /// Get optimal configuration for performance
    pub fn performance_optimized() -> Self {
        Self {
            strategy: LoadBalancingStrategy::PerformanceOptimized,
            enable_power_efficiency: false,
            power_efficiency_mode: PowerEfficiencyMode::PerformanceFirst,
            enable_adaptive_routing: true,
            cpu_threshold: 0.7,
            gpu_threshold: 0.9,
            ..Self::default()
        }
    }

    /// Get optimal configuration for power efficiency
    pub fn power_optimized() -> Self {
        Self {
            strategy: LoadBalancingStrategy::PowerEfficient,
            enable_power_efficiency: true,
            power_efficiency_mode: PowerEfficiencyMode::PowerFirst,
            max_power_consumption: 500.0, // Lower power limit
            power_scaling_factors: PowerScalingFactors {
                cpu_frequency_scaling: 0.8,
                gpu_power_limit_scaling: 0.7,
                memory_frequency_scaling: 0.9,
                voltage_scaling: 0.9,
                idle_power_optimization: 0.6,
                dvfs_enabled: true,
                sleep_states_enabled: true,
            },
            ..Self::default()
        }
    }

    /// Get optimal configuration for NUMA systems
    pub fn numa_optimized() -> Self {
        Self {
            strategy: LoadBalancingStrategy::NumaAware,
            enable_numa_awareness: true,
            numa_strategy: NumaStrategy::LocalNode,
            numa_memory_binding: true,
            numa_affinity_threshold: 0.9,
            ..Self::default()
        }
    }
}
