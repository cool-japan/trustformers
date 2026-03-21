//! Execution Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionInfo {
    pub test_id: String,
    pub test_name: String,
    pub test_suite: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub status: String,
    pub configuration: std::collections::HashMap<String, String>,
    pub expected_duration: Option<Duration>,
    pub resource_requirements: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct TestFilter {
    pub filter_id: String,
    pub criteria: HashMap<String, String>,
    pub include_pattern: Option<String>,
    pub exclude_pattern: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentialIoResult {
    pub throughput: f64,
    pub latency: Duration,
    pub block_size: usize,
    pub total_bytes: usize,
    pub operation_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomIoResult {
    pub iops: f64,
    pub latency: Duration,
    pub queue_depth: usize,
    pub total_operations: usize,
    pub operation_type: String,
}
