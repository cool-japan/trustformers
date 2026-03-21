//! # ProductionConfig - Trait Implementations
//!
//! This module contains trait implementations for `ProductionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{
    BackupRestoreConfig, BackupSchedule, BackupStorage, BlueGreenConfig, CanaryConfig,
    GracefulShutdownConfig, HealthAlertThresholds, HealthEndpoint, HealthMonitoringConfig,
    MaintenanceModeConfig, ProbeConfig, ProductionConfig, PromotionCriterion,
    ResourceSchedulingConfig, RestoreConfig, RestoreStrategy, RollbackConfig, RollbackStrategy,
    RollbackTrigger, RollingUpdateConfig, SchedulingStrategy, ShutdownAction, ShutdownHook,
    ShutdownHookType, ShutdownStage, SwitchStrategy, UpdateHealthCheckConfig, UpdateStrategy,
};

impl Default for ProductionConfig {
    fn default() -> Self {
        Self {
            graceful_shutdown: GracefulShutdownConfig {
                enabled: true,
                grace_period: Duration::from_secs(30),
                drain_timeout: Duration::from_secs(60),
                force_timeout: Duration::from_secs(300),
                pre_shutdown_hooks: vec![
                    ShutdownHook {
                        name: "save_state".to_string(),
                        hook_type: ShutdownHookType::SaveState {
                            path: "/tmp/app_state.json".to_string(),
                        },
                        timeout: Duration::from_secs(30),
                        critical: false,
                    },
                    ShutdownHook {
                        name: "flush_caches".to_string(),
                        hook_type: ShutdownHookType::FlushCaches,
                        timeout: Duration::from_secs(15),
                        critical: false,
                    },
                ],
                post_shutdown_hooks: Vec::new(),
                shutdown_stages: vec![
                    ShutdownStage {
                        name: "stop_accepting_requests".to_string(),
                        actions: vec![ShutdownAction::StopAcceptingRequests],
                        timeout: Duration::from_secs(10),
                        continue_on_failure: false,
                    },
                    ShutdownStage {
                        name: "drain_connections".to_string(),
                        actions: vec![ShutdownAction::DrainConnections],
                        timeout: Duration::from_secs(60),
                        continue_on_failure: true,
                    },
                    ShutdownStage {
                        name: "cleanup".to_string(),
                        actions: vec![
                            ShutdownAction::SaveModelState,
                            ShutdownAction::CloseDatabaseConnections,
                            ShutdownAction::ReleaseResources,
                        ],
                        timeout: Duration::from_secs(30),
                        continue_on_failure: true,
                    },
                ],
            },
            rolling_updates: RollingUpdateConfig {
                enabled: true,
                strategy: UpdateStrategy::Rolling {
                    max_unavailable: 1,
                    max_surge: 1,
                    batch_size: 1,
                },
                health_check: UpdateHealthCheckConfig {
                    endpoint: "/health".to_string(),
                    interval: Duration::from_secs(10),
                    timeout: Duration::from_secs(5),
                    success_threshold: 3,
                    failure_threshold: 3,
                    readiness_probe: ProbeConfig {
                        initial_delay: Duration::from_secs(10),
                        period: Duration::from_secs(10),
                        timeout: Duration::from_secs(5),
                        success_threshold: 1,
                        failure_threshold: 3,
                    },
                    liveness_probe: ProbeConfig {
                        initial_delay: Duration::from_secs(30),
                        period: Duration::from_secs(30),
                        timeout: Duration::from_secs(10),
                        success_threshold: 1,
                        failure_threshold: 3,
                    },
                },
                rollback: RollbackConfig {
                    enabled: true,
                    triggers: vec![
                        RollbackTrigger::ErrorRate {
                            threshold: 0.1,
                            duration: Duration::from_secs(300),
                        },
                        RollbackTrigger::Latency {
                            threshold: Duration::from_secs(5),
                            duration: Duration::from_secs(300),
                        },
                    ],
                    timeout: Duration::from_secs(600),
                    strategy: RollbackStrategy::Immediate,
                },
                canary: CanaryConfig {
                    initial_percentage: 5,
                    increment_steps: vec![10, 25, 50, 100],
                    step_duration: Duration::from_secs(300),
                    success_criteria: vec![PromotionCriterion::SuccessRate {
                        min_rate: 0.99,
                        duration: Duration::from_secs(300),
                    }],
                },
                blue_green: BlueGreenConfig {
                    blue_environment: "blue".to_string(),
                    green_environment: "green".to_string(),
                    switch_strategy: SwitchStrategy::Gradual {
                        duration: Duration::from_secs(300),
                    },
                    warmup_period: Duration::from_secs(60),
                },
            },
            health_monitoring: HealthMonitoringConfig {
                enabled: true,
                endpoints: vec![
                    HealthEndpoint {
                        name: "readiness".to_string(),
                        url: "/health/readiness".to_string(),
                        expected_status: 200,
                        timeout: Duration::from_secs(5),
                        critical: true,
                    },
                    HealthEndpoint {
                        name: "liveness".to_string(),
                        url: "/health/liveness".to_string(),
                        expected_status: 200,
                        timeout: Duration::from_secs(5),
                        critical: true,
                    },
                ],
                interval: Duration::from_secs(30),
                alert_thresholds: HealthAlertThresholds {
                    error_rate: 0.05,
                    response_time: Duration::from_secs(1),
                    availability: 0.99,
                },
            },
            backup_restore: BackupRestoreConfig {
                enabled: false,
                backup_schedule: BackupSchedule::Manual,
                storage: BackupStorage::Local {
                    path: "/tmp/backups".to_string(),
                },
                restore: RestoreConfig {
                    strategy: RestoreStrategy::Full,
                    validation: Vec::new(),
                    timeout: Duration::from_secs(1800),
                },
            },
            resource_scheduling: ResourceSchedulingConfig {
                enabled: false,
                policies: Vec::new(),
                strategies: vec![SchedulingStrategy::Priority],
            },
            maintenance_mode: MaintenanceModeConfig {
                enabled: false,
                message: "System is under maintenance. Please try again later.".to_string(),
                page_template: None,
                allowed_operations: vec!["health".to_string()],
                schedule: None,
            },
        }
    }
}
