//! Deployment strategies and rollout control

use super::Variant;
use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Deployment strategy for rolling out winning variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    /// Immediate full deployment
    Immediate,
    /// Gradual percentage-based rollout
    Gradual {
        /// Initial percentage
        initial_percentage: f64,
        /// Increment per step
        increment: f64,
        /// Time between increments
        increment_interval: Duration,
    },
    /// Canary deployment
    Canary {
        /// Canary percentage
        canary_percentage: f64,
        /// Monitoring duration
        monitoring_duration: Duration,
    },
    /// Blue-green deployment
    BlueGreen {
        /// Warm-up duration
        warmup_duration: Duration,
    },
    /// Feature flag based
    FeatureFlag {
        /// Flag name
        flag_name: String,
    },
}

/// Rollout status
#[derive(Debug, Clone, PartialEq)]
pub enum RolloutStatus {
    /// Not started
    NotStarted,
    /// In progress
    InProgress {
        /// Current percentage
        current_percentage: f64,
        /// Start time
        started_at: DateTime<Utc>,
    },
    /// Paused
    Paused {
        /// Percentage when paused
        paused_at_percentage: f64,
    },
    /// Completed
    Completed {
        /// Completion time
        completed_at: DateTime<Utc>,
    },
    /// Rolled back
    RolledBack {
        /// Rollback time
        rolled_back_at: DateTime<Utc>,
        /// Reason for rollback
        reason: String,
    },
}

/// Rollout configuration
#[derive(Debug, Clone)]
pub struct RolloutConfig {
    /// Experiment ID
    pub experiment_id: String,
    /// Winning variant
    pub variant: Variant,
    /// Deployment strategy
    pub strategy: DeploymentStrategy,
    /// Health checks
    pub health_checks: Vec<HealthCheck>,
    /// Rollback conditions
    pub rollback_conditions: Vec<RollbackCondition>,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Check name
    pub name: String,
    /// Check type
    pub check_type: HealthCheckType,
    /// Threshold for failure
    pub threshold: f64,
    /// Number of consecutive failures to trigger
    pub consecutive_failures: u32,
}

/// Types of health checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheckType {
    /// Error rate check
    ErrorRate,
    /// Latency check (p99)
    LatencyP99,
    /// CPU usage
    CpuUsage,
    /// Memory usage
    MemoryUsage,
    /// Custom metric
    Custom(String),
}

/// Conditions that trigger automatic rollback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackCondition {
    /// Condition name
    pub name: String,
    /// Metric to monitor
    pub metric: String,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Threshold value
    pub threshold: f64,
    /// Duration condition must be true
    pub duration: Duration,
}

/// Comparison operators for rollback conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Rollout controller
pub struct RolloutController {
    /// Active rollouts
    rollouts: Arc<RwLock<HashMap<String, ActiveRollout>>>,
    /// Health monitor
    health_monitor: Arc<HealthMonitor>,
}

/// Active rollout state
struct ActiveRollout {
    /// Configuration
    config: RolloutConfig,
    /// Current status
    status: RolloutStatus,
    /// Health check failures
    health_failures: HashMap<String, u32>,
    /// Rollback condition states
    rollback_states: HashMap<String, RollbackState>,
}

/// State for tracking rollback conditions
struct RollbackState {
    /// When condition was first met
    first_triggered: Option<DateTime<Utc>>,
    /// Current value
    current_value: f64,
}

/// Health monitoring service
struct HealthMonitor {
    /// Metrics storage
    metrics: Arc<RwLock<HashMap<String, f64>>>,
}

impl Default for RolloutController {
    fn default() -> Self {
        Self::new()
    }
}

impl RolloutController {
    /// Create a new rollout controller
    pub fn new() -> Self {
        Self {
            rollouts: Arc::new(RwLock::new(HashMap::new())),
            health_monitor: Arc::new(HealthMonitor::new()),
        }
    }

    /// Start a new rollout
    pub fn start_rollout(&self, config: RolloutConfig) -> Result<()> {
        let experiment_id = config.experiment_id.clone();

        let status = match &config.strategy {
            DeploymentStrategy::Immediate => RolloutStatus::InProgress {
                current_percentage: 100.0,
                started_at: Utc::now(),
            },
            DeploymentStrategy::Gradual {
                initial_percentage, ..
            } => RolloutStatus::InProgress {
                current_percentage: *initial_percentage,
                started_at: Utc::now(),
            },
            DeploymentStrategy::Canary {
                canary_percentage, ..
            } => RolloutStatus::InProgress {
                current_percentage: *canary_percentage,
                started_at: Utc::now(),
            },
            DeploymentStrategy::BlueGreen { .. } => RolloutStatus::InProgress {
                current_percentage: 0.0,
                started_at: Utc::now(),
            },
            DeploymentStrategy::FeatureFlag { .. } => RolloutStatus::InProgress {
                current_percentage: 0.0,
                started_at: Utc::now(),
            },
        };

        let rollout = ActiveRollout {
            config,
            status,
            health_failures: HashMap::new(),
            rollback_states: HashMap::new(),
        };

        self.rollouts.write().insert(experiment_id, rollout);
        Ok(())
    }

    /// Update rollout progress
    pub fn update_rollout(&self, experiment_id: &str) -> Result<()> {
        let mut rollouts = self.rollouts.write();
        let rollout = rollouts
            .get_mut(experiment_id)
            .ok_or_else(|| anyhow::anyhow!("Rollout not found"))?;

        // Check health and rollback conditions
        if self.should_rollback(rollout)? {
            rollout.status = RolloutStatus::RolledBack {
                rolled_back_at: Utc::now(),
                reason: "Health check or rollback condition triggered".to_string(),
            };
            return Ok(());
        }

        // Update based on strategy
        match &rollout.config.strategy {
            DeploymentStrategy::Gradual {
                increment,
                increment_interval,
                ..
            } => {
                if let RolloutStatus::InProgress {
                    current_percentage,
                    started_at,
                } = &rollout.status
                {
                    let elapsed = Utc::now() - *started_at;
                    let steps = (elapsed.num_seconds() / increment_interval.num_seconds()) as f64;
                    let new_percentage = (current_percentage + steps * increment).min(100.0);

                    if new_percentage >= 100.0 {
                        rollout.status = RolloutStatus::Completed {
                            completed_at: Utc::now(),
                        };
                    } else {
                        rollout.status = RolloutStatus::InProgress {
                            current_percentage: new_percentage,
                            started_at: *started_at,
                        };
                    }
                }
            },
            DeploymentStrategy::Canary {
                monitoring_duration,
                ..
            } => {
                if let RolloutStatus::InProgress { started_at, .. } = &rollout.status {
                    if Utc::now() - *started_at > *monitoring_duration {
                        rollout.status = RolloutStatus::Completed {
                            completed_at: Utc::now(),
                        };
                    }
                }
            },
            DeploymentStrategy::BlueGreen { warmup_duration } => {
                if let RolloutStatus::InProgress { started_at, .. } = &rollout.status {
                    if Utc::now() - *started_at > *warmup_duration {
                        rollout.status = RolloutStatus::InProgress {
                            current_percentage: 100.0,
                            started_at: *started_at,
                        };
                    }
                }
            },
            _ => {},
        }

        Ok(())
    }

    /// Check if rollback is needed
    fn should_rollback(&self, rollout: &mut ActiveRollout) -> Result<bool> {
        // Check health checks
        for health_check in &rollout.config.health_checks {
            let metric_value = self.health_monitor.get_metric(&health_check.name)?;

            let failed = match health_check.check_type {
                HealthCheckType::ErrorRate => metric_value > health_check.threshold,
                HealthCheckType::LatencyP99 => metric_value > health_check.threshold,
                HealthCheckType::CpuUsage => metric_value > health_check.threshold,
                HealthCheckType::MemoryUsage => metric_value > health_check.threshold,
                HealthCheckType::Custom(_) => false, // Would need custom logic
            };

            if failed {
                let failures =
                    rollout.health_failures.entry(health_check.name.clone()).or_insert(0);
                *failures += 1;

                if *failures >= health_check.consecutive_failures {
                    return Ok(true);
                }
            } else {
                rollout.health_failures.remove(&health_check.name);
            }
        }

        // Check rollback conditions
        for condition in &rollout.config.rollback_conditions {
            let metric_value = self.health_monitor.get_metric(&condition.metric)?;

            let triggered = match condition.operator {
                ComparisonOperator::GreaterThan => metric_value > condition.threshold,
                ComparisonOperator::LessThan => metric_value < condition.threshold,
                ComparisonOperator::GreaterThanOrEqual => metric_value >= condition.threshold,
                ComparisonOperator::LessThanOrEqual => metric_value <= condition.threshold,
            };

            let state =
                rollout.rollback_states.entry(condition.name.clone()).or_insert(RollbackState {
                    first_triggered: None,
                    current_value: metric_value,
                });

            state.current_value = metric_value;

            if !triggered {
                state.first_triggered = None;
                continue;
            }

            if state.first_triggered.is_none() {
                state.first_triggered = Some(Utc::now());
                continue;
            }

            if let Some(first_triggered) = state.first_triggered {
                if Utc::now() - first_triggered >= condition.duration {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Promote a variant (start rollout)
    pub fn promote(&self, experiment_id: &str, variant: &Variant) -> Result<()> {
        let config = RolloutConfig {
            experiment_id: experiment_id.to_string(),
            variant: variant.clone(),
            strategy: DeploymentStrategy::Gradual {
                initial_percentage: 10.0,
                increment: 10.0,
                increment_interval: Duration::hours(1),
            },
            health_checks: vec![
                HealthCheck {
                    name: "error_rate".to_string(),
                    check_type: HealthCheckType::ErrorRate,
                    threshold: 0.05,
                    consecutive_failures: 3,
                },
                HealthCheck {
                    name: "latency_p99".to_string(),
                    check_type: HealthCheckType::LatencyP99,
                    threshold: 1000.0,
                    consecutive_failures: 3,
                },
            ],
            rollback_conditions: vec![RollbackCondition {
                name: "sustained_errors".to_string(),
                metric: "error_rate".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 0.1,
                duration: Duration::minutes(5),
            }],
        };

        self.start_rollout(config)
    }

    /// Rollback to control
    pub fn rollback(&self, experiment_id: &str) -> Result<()> {
        let mut rollouts = self.rollouts.write();
        if let Some(rollout) = rollouts.get_mut(experiment_id) {
            rollout.status = RolloutStatus::RolledBack {
                rolled_back_at: Utc::now(),
                reason: "Manual rollback".to_string(),
            };
        }
        Ok(())
    }

    /// Get rollout status
    pub fn get_status(&self, experiment_id: &str) -> Result<RolloutStatus> {
        let rollouts = self.rollouts.read();
        let rollout = rollouts
            .get(experiment_id)
            .ok_or_else(|| anyhow::anyhow!("Rollout not found"))?;
        Ok(rollout.status.clone())
    }

    /// Pause rollout
    pub fn pause(&self, experiment_id: &str) -> Result<()> {
        let mut rollouts = self.rollouts.write();
        if let Some(rollout) = rollouts.get_mut(experiment_id) {
            if let RolloutStatus::InProgress {
                current_percentage, ..
            } = rollout.status
            {
                rollout.status = RolloutStatus::Paused {
                    paused_at_percentage: current_percentage,
                };
            }
        }
        Ok(())
    }

    /// Resume rollout
    pub fn resume(&self, experiment_id: &str) -> Result<()> {
        let mut rollouts = self.rollouts.write();
        if let Some(rollout) = rollouts.get_mut(experiment_id) {
            if let RolloutStatus::Paused {
                paused_at_percentage,
            } = rollout.status
            {
                rollout.status = RolloutStatus::InProgress {
                    current_percentage: paused_at_percentage,
                    started_at: Utc::now(),
                };
            }
        }
        Ok(())
    }
}

impl HealthMonitor {
    fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn get_metric(&self, name: &str) -> Result<f64> {
        let metrics = self.metrics.read();
        Ok(metrics.get(name).copied().unwrap_or(0.0))
    }

    #[allow(dead_code)]
    pub fn update_metric(&self, name: &str, value: f64) {
        self.metrics.write().insert(name.to_string(), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_immediate_deployment() {
        let controller = RolloutController::new();
        let config = RolloutConfig {
            experiment_id: "exp1".to_string(),
            variant: Variant::new("winner", "model-v2"),
            strategy: DeploymentStrategy::Immediate,
            health_checks: vec![],
            rollback_conditions: vec![],
        };

        controller.start_rollout(config).unwrap();

        match controller.get_status("exp1").unwrap() {
            RolloutStatus::InProgress {
                current_percentage, ..
            } => {
                assert_eq!(current_percentage, 100.0);
            },
            _ => panic!("Expected InProgress status"),
        }
    }

    #[test]
    fn test_gradual_rollout() {
        let controller = RolloutController::new();
        let config = RolloutConfig {
            experiment_id: "exp2".to_string(),
            variant: Variant::new("winner", "model-v2"),
            strategy: DeploymentStrategy::Gradual {
                initial_percentage: 10.0,
                increment: 20.0,
                increment_interval: Duration::seconds(1),
            },
            health_checks: vec![],
            rollback_conditions: vec![],
        };

        controller.start_rollout(config).unwrap();

        // Initial percentage
        match controller.get_status("exp2").unwrap() {
            RolloutStatus::InProgress {
                current_percentage, ..
            } => {
                assert_eq!(current_percentage, 10.0);
            },
            _ => panic!("Expected InProgress status"),
        }

        // Wait and update
        std::thread::sleep(std::time::Duration::from_secs(2));
        controller.update_rollout("exp2").unwrap();

        // Should have increased
        match controller.get_status("exp2").unwrap() {
            RolloutStatus::InProgress {
                current_percentage, ..
            } => {
                assert!(current_percentage > 10.0);
            },
            _ => panic!("Expected InProgress status"),
        }
    }

    #[test]
    fn test_health_check_rollback() {
        let controller = RolloutController::new();
        let config = RolloutConfig {
            experiment_id: "exp3".to_string(),
            variant: Variant::new("winner", "model-v2"),
            strategy: DeploymentStrategy::Canary {
                canary_percentage: 5.0,
                monitoring_duration: Duration::hours(1),
            },
            health_checks: vec![HealthCheck {
                name: "error_rate".to_string(),
                check_type: HealthCheckType::ErrorRate,
                threshold: 0.05,
                consecutive_failures: 1,
            }],
            rollback_conditions: vec![],
        };

        controller.start_rollout(config).unwrap();

        // Simulate high error rate
        controller.health_monitor.update_metric("error_rate", 0.1);
        controller.update_rollout("exp3").unwrap();

        // Should be rolled back
        match controller.get_status("exp3").unwrap() {
            RolloutStatus::RolledBack { reason, .. } => {
                assert!(reason.contains("Health check"));
            },
            _ => panic!("Expected RolledBack status"),
        }
    }
}
