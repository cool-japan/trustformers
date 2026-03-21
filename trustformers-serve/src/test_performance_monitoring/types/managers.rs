//! Managers Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

// Import types from sibling modules
use super::config::{DashboardConfig, TestPerformanceMonitoringConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleExecutor {
    config: TestPerformanceMonitoringConfig,
}

impl RuleExecutor {
    pub fn new(config: &TestPerformanceMonitoringConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleScheduler {
    config: TestPerformanceMonitoringConfig,
}

impl RuleScheduler {
    pub fn new(config: &TestPerformanceMonitoringConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateScheduler {
    config: DashboardConfig,
}

impl UpdateScheduler {
    pub fn new(config: &DashboardConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]

pub struct AggregationManager {
    pub aggregated_metrics: HashMap<String, f64>,
    pub last_aggregation: Option<DateTime<Utc>>,
}

impl Default for AggregationManager {
    fn default() -> Self {
        Self {
            aggregated_metrics: HashMap::new(),
            last_aggregation: None,
        }
    }
}

impl AggregationManager {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Serialize, Deserialize)]

pub struct AdaptiveLearningOrchestrator {
    pub learning_rate: f64,
    pub model_updates: u64,
    pub last_update: Option<DateTime<Utc>>,
}

impl AdaptiveLearningOrchestrator {
    pub async fn new(_config: &TestPerformanceMonitoringConfig) -> anyhow::Result<Self> {
        Ok(Self {
            learning_rate: 0.01,
            model_updates: 0,
            last_update: None,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]

pub struct MonitoringScheduler {
    pub enabled: bool,
    pub interval: Duration,
    pub max_concurrent: usize,
}

#[derive(Debug, Serialize, Deserialize)]

pub struct SuppressionScheduler {
    pub schedule_interval: Duration,
    pub max_suppressions: usize,
}

#[derive(Debug, Serialize, Deserialize)]

pub struct DynamicSuppressionEngine {
    pub enabled: bool,
    pub learning_rate: f64,
    pub adaptation_window: Duration,
}

#[derive(Debug, Serialize, Deserialize)]

pub struct AutoRecoveryEngine {
    pub enabled: bool,
    pub max_retries: u32,
    pub retry_delay: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupScheduler {
    pub schedule_interval: Duration,
    pub cleanup_rules: Vec<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationScheduler {
    pub scheduler_id: String,
    pub schedule: String,
    pub window_size: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceManager {
    pub manager_id: String,
    pub compliance_rules: Vec<String>,
    pub audit_log_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionExecutor {
    pub executor_id: String,
    pub transition_type: String,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]

pub struct SchedulerEngine {
    pub engine_id: String,
    pub max_concurrent: usize,
    pub retry_policy: String,
}
