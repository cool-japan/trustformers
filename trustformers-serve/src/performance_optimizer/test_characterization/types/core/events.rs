//! Event and operation types for test characterization

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Instant};

use super::super::{
    locking::LockEvent,
    network_io::{IoOperation, NetworkEvent},
    patterns::SynchronizationEvent,
    performance::PerformanceSample,
    quality::TracedOperation,
    resources::{MemoryAllocation, SystemCall},
};
use super::enums::OperationType;

pub struct CleanupEvent {
    pub event_type: String,
    pub items_cleaned: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// Traced operations
    pub operations: Vec<TracedOperation>,
    /// Thread execution timeline
    pub thread_timeline: HashMap<u64, Vec<(Instant, OperationType)>>,
    /// Resource allocation timeline
    pub resource_timeline: Vec<(Instant, String, String)>,
    /// Synchronization events
    pub synchronization_events: Vec<SynchronizationEvent>,
    /// Performance samples
    pub performance_samples: Vec<PerformanceSample>,
    /// Memory allocation patterns
    pub memory_allocations: Vec<MemoryAllocation>,
    /// System call trace
    pub system_calls: Vec<SystemCall>,
    /// Lock acquisition timeline
    pub lock_timeline: Vec<LockEvent>,
    /// I/O operation trace
    pub io_operations: Vec<IoOperation>,
    /// Network activity trace
    pub network_activity: Vec<NetworkEvent>,
    /// Resource identifier
    pub resource: String,
    /// Operation type
    pub operation: String,
    /// Timestamp
    pub timestamp: Instant,
    /// Thread ID
    pub thread_id: u64,
}

impl Default for ExecutionTrace {
    fn default() -> Self {
        Self {
            operations: Vec::new(),
            thread_timeline: HashMap::new(),
            resource_timeline: Vec::new(),
            synchronization_events: Vec::new(),
            performance_samples: Vec::new(),
            memory_allocations: Vec::new(),
            system_calls: Vec::new(),
            lock_timeline: Vec::new(),
            io_operations: Vec::new(),
            network_activity: Vec::new(),
            resource: String::new(),
            operation: String::new(),
            timestamp: Instant::now(),
            thread_id: 0,
        }
    }
}

pub struct AutomaticAction {
    pub action_type: String,
    pub target: String,
    pub executed_at: chrono::DateTime<chrono::Utc>,
    pub success: bool,
}

pub struct LifecycleAction {
    pub action_type: String,
    pub trigger_condition: String,
    pub action_params: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub key: String,
    pub value: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct ActionPriority {
    pub priority_level: u8,
    pub priority_name: String,
}

pub struct ActionType {
    pub action_name: String,
    pub action_category: String,
}

pub struct TestExecutionInfo {
    pub test_id: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub duration: std::time::Duration,
    pub status: String,
    pub result: String,
}

pub struct UpdateData {
    pub update_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub data: HashMap<String, String>,
    pub version: String,
}

pub struct UpdateType {
    pub update_type: String,
    pub category: String,
    pub priority: u32,
}
