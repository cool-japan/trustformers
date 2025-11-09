//! CPU/GPU Load Balancer Core Types
//!
//! This module contains all core data structures, enums, and types used
//! throughout the CPU/GPU load balancing system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

// Helper functions for Instant serialization
pub(crate) fn serialize_instant<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let duration_since_epoch = instant.elapsed();
    serializer.serialize_u64(duration_since_epoch.as_millis() as u64)
}

pub(crate) fn deserialize_instant<'de, D>(deserializer: D) -> Result<Instant, D::Error>
where
    D: Deserializer<'de>,
{
    let millis = u64::deserialize(deserializer)?;
    Ok(Instant::now() - Duration::from_millis(millis))
}

pub(crate) fn serialize_optional_instant<S>(
    instant: &Option<Instant>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match instant {
        Some(instant) => {
            let duration_since_epoch = instant.elapsed();
            serializer.serialize_some(&(duration_since_epoch.as_millis() as u64))
        },
        None => serializer.serialize_none(),
    }
}

pub(crate) fn deserialize_optional_instant<'de, D>(
    deserializer: D,
) -> Result<Option<Instant>, D::Error>
where
    D: Deserializer<'de>,
{
    let millis: Option<u64> = Option::deserialize(deserializer)?;
    Ok(millis.map(|m| Instant::now() - Duration::from_millis(m)))
}

/// Compute task for load balancing
///
/// Represents a computational task with all metadata needed for intelligent
/// routing between CPU and GPU resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeTask {
    /// Task ID
    pub id: String,

    /// Task type
    pub task_type: TaskType,

    /// Memory requirements in bytes
    pub memory_required: usize,

    /// Estimated compute operations
    pub compute_operations: usize,

    /// Task priority
    pub priority: TaskPriority,

    /// Preferred processor type
    pub preferred_processor: Option<ProcessorType>,

    /// Batch size for batched operations
    pub batch_size: Option<usize>,

    /// Input data size in bytes
    pub input_size: usize,

    /// Expected output size in bytes
    pub output_size: usize,

    /// Parallelizability factor (0-1)
    pub parallelizability: f32,

    /// Memory access pattern
    pub memory_pattern: MemoryPattern,

    /// Task deadline
    #[serde(
        serialize_with = "serialize_optional_instant",
        deserialize_with = "deserialize_optional_instant"
    )]
    pub deadline: Option<Instant>,

    /// Client ID
    pub client_id: Option<String>,

    /// Task metadata
    pub metadata: HashMap<String, String>,

    /// Creation timestamp
    #[serde(
        serialize_with = "serialize_instant",
        deserialize_with = "deserialize_instant"
    )]
    pub created_at: Instant,
}

/// Task types
///
/// Different categories of computational tasks with varying characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    /// Matrix operations
    MatrixOps,

    /// Neural network inference
    Inference,

    /// Model training
    Training,

    /// Image processing
    ImageProcessing,

    /// Text processing
    TextProcessing,

    /// Batch processing
    BatchProcessing,

    /// Data processing
    DataProcessing,

    /// Streaming operations
    Streaming,

    /// Custom compute
    Custom(String),
}

/// Task priority levels
///
/// Priority levels for task scheduling and resource allocation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Processor types
///
/// Different types of compute processors available.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ProcessorType {
    CPU,
    GPU,
}

/// Memory access patterns
///
/// Different patterns of memory access that affect performance characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryPattern {
    /// Sequential access
    Sequential,

    /// Random access
    Random,

    /// Strided access
    Strided(usize),

    /// Scatter-gather
    ScatterGather,

    /// Cache-friendly
    CacheFriendly,
}

/// Processor resource information
///
/// Complete state and performance characteristics of a compute resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorResource {
    /// Processor type
    pub processor_type: ProcessorType,

    /// Processor ID
    pub id: usize,

    /// Current utilization (0-1)
    pub utilization: f32,

    /// Available memory in bytes
    pub available_memory: usize,

    /// Total memory in bytes
    pub total_memory: usize,

    /// Temperature in Celsius
    pub temperature: f32,

    /// Power consumption in watts
    pub power_consumption: f32,

    /// Active tasks count
    pub active_tasks: usize,

    /// Processor status
    pub status: ProcessorStatus,

    /// Performance characteristics
    pub performance: PerformanceCharacteristics,

    /// Power management state
    pub power_state: PowerState,

    /// Current power efficiency rating
    pub power_efficiency_rating: f32,

    /// Maximum power limit (watts)
    pub max_power_limit: f32,

    /// Current power scaling factor
    pub current_power_scaling: f32,

    /// Last updated timestamp
    #[serde(
        serialize_with = "serialize_instant",
        deserialize_with = "deserialize_instant"
    )]
    pub last_updated: Instant,
}

/// Processor status
///
/// Current operational status of a processor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessorStatus {
    Available,
    Busy,
    Overloaded,
    Maintenance,
    Error(String),
}

/// Power management state
///
/// Different power states for processor power management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerState {
    /// Full power, maximum performance
    FullPower,

    /// Reduced power, balanced performance
    ReducedPower,

    /// Low power, minimum performance
    LowPower,

    /// Sleep state
    Sleep,

    /// Idle state with power optimization
    IdleOptimized,

    /// Custom power state
    Custom(f32), // Power scaling factor
}

/// Performance characteristics
///
/// Detailed performance metrics for a processor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    /// Operations per second
    pub ops_per_second: f64,

    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f32,

    /// Latency per operation (microseconds)
    pub latency_per_op: f32,

    /// Energy efficiency (ops per watt)
    pub energy_efficiency: f32,

    /// Parallel efficiency factor
    pub parallel_efficiency: f32,

    /// Cache miss rate
    pub cache_miss_rate: f32,
}

/// Task execution result
///
/// Complete results and metrics from task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExecutionResult {
    /// Task ID
    pub task_id: String,

    /// Assigned processor
    pub processor: ProcessorType,

    /// Processor ID
    pub processor_id: usize,

    /// Execution status
    pub status: ExecutionStatus,

    /// Actual execution time
    pub execution_time: Duration,

    /// Memory used
    pub memory_used: usize,

    /// Energy consumed (joules)
    pub energy_consumed: f32,

    /// Throughput (ops/sec)
    pub throughput: f64,

    /// Start time
    #[serde(
        serialize_with = "serialize_instant",
        deserialize_with = "deserialize_instant"
    )]
    pub started_at: Instant,

    /// End time
    #[serde(
        serialize_with = "serialize_instant",
        deserialize_with = "deserialize_instant"
    )]
    pub completed_at: Instant,

    /// Error message if failed
    pub error_message: Option<String>,
}

/// Task execution status
///
/// Current status of task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
}

/// Performance history entry
///
/// Historical performance data for analytics and optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHistory {
    /// Task type
    pub task_type: TaskType,

    /// Processor type
    pub processor_type: ProcessorType,

    /// Execution time
    pub execution_time: Duration,

    /// Memory used
    pub memory_used: usize,

    /// Energy consumed
    pub energy_consumed: f32,

    /// Throughput
    pub throughput: f64,

    /// Task size
    pub task_size: usize,

    /// Timestamp
    #[serde(
        serialize_with = "serialize_instant",
        deserialize_with = "deserialize_instant"
    )]
    pub timestamp: Instant,
}

/// Load balancer events
///
/// Events emitted by the load balancer for monitoring and analytics.
#[derive(Debug, Clone, Serialize)]
pub enum LoadBalancerEvent {
    /// Task submitted
    TaskSubmitted {
        task_id: String,
        task_type: TaskType,
        timestamp: DateTime<Utc>,
    },

    /// Task assigned
    TaskAssigned {
        task_id: String,
        processor_type: ProcessorType,
        processor_id: usize,
        timestamp: DateTime<Utc>,
    },

    /// Task completed
    TaskCompleted {
        task_id: String,
        processor_type: ProcessorType,
        execution_time: Duration,
        timestamp: DateTime<Utc>,
    },

    /// Task failed
    TaskFailed {
        task_id: String,
        processor_type: ProcessorType,
        error: String,
        timestamp: DateTime<Utc>,
    },

    /// Resource status changed
    ResourceStatusChanged {
        processor_type: ProcessorType,
        processor_id: usize,
        old_status: ProcessorStatus,
        new_status: ProcessorStatus,
        timestamp: DateTime<Utc>,
    },

    /// Power state changed
    PowerStateChanged {
        processor_type: ProcessorType,
        processor_id: usize,
        old_state: PowerState,
        new_state: PowerState,
        timestamp: DateTime<Utc>,
    },

    /// Load balancer started
    LoadBalancerStarted { timestamp: DateTime<Utc> },

    /// Load balancer stopped
    LoadBalancerStopped { timestamp: DateTime<Utc> },
}

/// Utility implementations
impl ComputeTask {
    /// Create a new task with default values
    pub fn new(id: String, task_type: TaskType) -> Self {
        Self {
            id,
            task_type,
            memory_required: 0,
            compute_operations: 0,
            priority: TaskPriority::Normal,
            preferred_processor: None,
            batch_size: None,
            input_size: 0,
            output_size: 0,
            parallelizability: 0.5,
            memory_pattern: MemoryPattern::Sequential,
            deadline: None,
            client_id: None,
            metadata: HashMap::new(),
            created_at: Instant::now(),
        }
    }

    /// Check if task is suitable for GPU execution
    pub fn is_gpu_suitable(&self) -> bool {
        matches!(
            self.task_type,
            TaskType::MatrixOps
                | TaskType::Inference
                | TaskType::Training
                | TaskType::ImageProcessing
        ) && self.compute_operations > 1000
            && self.parallelizability > 0.3
    }

    /// Get estimated memory efficiency
    pub fn memory_efficiency(&self) -> f32 {
        if self.input_size == 0 {
            return 1.0;
        }

        let total_io = self.input_size + self.output_size;
        let ratio = self.memory_required as f32 / total_io as f32;

        // Lower ratio means better memory efficiency
        1.0 / (1.0 + ratio)
    }
}

impl ProcessorResource {
    /// Check if processor is available for new tasks
    pub fn is_available(&self) -> bool {
        matches!(self.status, ProcessorStatus::Available) && self.utilization < 0.9
    }

    /// Get current efficiency score
    pub fn efficiency_score(&self) -> f32 {
        let utilization_factor = 1.0 - self.utilization;
        let memory_factor = self.available_memory as f32 / self.total_memory as f32;
        let power_factor = self.power_efficiency_rating;

        (utilization_factor * 0.4 + memory_factor * 0.3 + power_factor * 0.3).min(1.0)
    }
}

impl Default for ProcessorResource {
    fn default() -> Self {
        Self {
            processor_type: ProcessorType::CPU,
            id: 0,
            utilization: 0.0,
            available_memory: 0,
            total_memory: 0,
            temperature: 25.0,
            power_consumption: 0.0,
            active_tasks: 0,
            status: ProcessorStatus::Available,
            performance: PerformanceCharacteristics {
                ops_per_second: 0.0,
                memory_bandwidth: 0.0,
                latency_per_op: 0.0,
                energy_efficiency: 0.0,
                parallel_efficiency: 0.0,
                cache_miss_rate: 0.0,
            },
            power_state: PowerState::FullPower,
            power_efficiency_rating: 1.0,
            max_power_limit: 100.0,
            current_power_scaling: 1.0,
            last_updated: Instant::now(),
        }
    }
}
