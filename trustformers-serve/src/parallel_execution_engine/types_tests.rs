//! Tests for parallel execution engine types

use super::types::*;
use chrono::Utc;
use std::time::Duration;

#[test]
fn test_alert_level_info() {
    let l = AlertLevel::Info;
    let _ = format!("{:?}", l);
}

#[test]
fn test_alert_level_critical() {
    let l = AlertLevel::Critical;
    let _ = format!("{:?}", l);
}

#[test]
fn test_pool_item_state_available() {
    let s = PoolItemState::Available;
    let _ = format!("{:?}", s);
}

#[test]
fn test_pool_item_state_allocated() {
    let s = PoolItemState::Allocated;
    let _ = format!("{:?}", s);
}

#[test]
fn test_execution_session_state_running() {
    let s = ExecutionSessionState::Running;
    let _ = format!("{:?}", s);
}

#[test]
fn test_execution_session_state_completed() {
    let s = ExecutionSessionState::Completed;
    let _ = format!("{:?}", s);
}

#[test]
fn test_dependency_node_type_test() {
    let t = DependencyNodeType::Test;
    let _ = format!("{:?}", t);
}

#[test]
fn test_cleanup_policy_immediate() {
    let p = CleanupPolicy::Immediate;
    let _ = format!("{:?}", p);
}

#[test]
fn test_cleanup_policy_delayed() {
    let p = CleanupPolicy::Delayed(Duration::from_secs(60));
    let _ = format!("{:?}", p);
}

#[test]
fn test_pool_growth_strategy_fixed() {
    let s = PoolGrowthStrategy::Fixed;
    let _ = format!("{:?}", s);
}

#[test]
fn test_pool_growth_strategy_on_demand() {
    let s = PoolGrowthStrategy::OnDemand;
    let _ = format!("{:?}", s);
}

#[test]
fn test_scheduling_constraint_type_time_window() {
    let t = SchedulingConstraintType::TimeWindow;
    let _ = format!("{:?}", t);
}

#[test]
fn test_scheduling_constraint_type_resource() {
    let t = SchedulingConstraintType::ResourceAvailability;
    let _ = format!("{:?}", t);
}

#[test]
fn test_scheduling_event_type_scheduled() {
    let t = SchedulingEventType::Scheduled;
    let _ = format!("{:?}", t);
}

#[test]
fn test_priority_queue_new() {
    let queue: PriorityQueue<i32> = PriorityQueue::new();
    let _ = format!("{:?}", queue);
}

#[test]
fn test_work_stealing_config_creation() {
    let config = WorkStealingConfig {
        enabled: true,
        steal_threshold: 0.5,
        max_steals_per_interval: 3,
        steal_timeout: Duration::from_millis(100),
    };
    assert!(config.enabled);
}

#[test]
fn test_worker_scaling_config_creation() {
    let config = WorkerScalingConfig {
        enabled: true,
        scale_up_threshold: 0.8,
        scale_down_threshold: 0.2,
        cooldown_period: Duration::from_secs(30),
        scaling_factor: 1.5,
    };
    assert!(config.scale_up_threshold > config.scale_down_threshold);
}

#[test]
fn test_resource_pool_config_creation() {
    let config = ResourcePoolConfig {
        min_size: 5,
        max_size: 50,
        growth_strategy: PoolGrowthStrategy::OnDemand,
        cleanup_interval: Duration::from_secs(300),
        item_timeout: Duration::from_secs(600),
    };
    assert!(config.min_size < config.max_size);
}

#[test]
fn test_scheduling_constraint_creation() {
    let constraint = SchedulingConstraint {
        constraint_type: SchedulingConstraintType::TimeWindow,
        value: "60s".to_string(),
        priority: 0.8,
        deadline: Some(Utc::now()),
    };
    assert!(constraint.priority > 0.0);
}

#[test]
fn test_execution_constraint_creation() {
    let constraint = ExecutionConstraint {
        constraint_type: ExecutionConstraintType::Before,
        value: "test_b".to_string(),
        priority: 0.8,
    };
    assert!(constraint.priority > 0.0);
}

#[test]
fn test_alert_thresholds_creation() {
    let thresholds = AlertThresholds {
        high_error_rate: 0.1,
        high_latency: Duration::from_millis(500),
        resource_exhaustion: 0.9,
        queue_backup: 100,
        worker_failure: 3,
    };
    assert!(thresholds.high_error_rate > 0.0);
}

#[test]
fn test_dependency_graph_default() {
    let graph = DependencyGraph::default();
    let _ = format!("{:?}", graph);
}

#[test]
fn test_queue_statistics_default() {
    let stats = QueueStatistics::default();
    let _ = format!("{:?}", stats);
}

#[test]
fn test_rebalancing_config_creation() {
    let config = RebalancingConfig {
        enabled: true,
        interval: Duration::from_secs(30),
        imbalance_threshold: 0.3,
        aggressiveness: 0.5,
        work_stealing: WorkStealingConfig {
            enabled: true,
            steal_threshold: 0.5,
            max_steals_per_interval: 5,
            steal_timeout: Duration::from_millis(100),
        },
    };
    assert!(config.enabled);
}

#[test]
fn test_scheduling_config_default() {
    let config = SchedulingConfig::default();
    let _ = format!("{:?}", config);
}
