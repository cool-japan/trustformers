//! Auto-generated module structure

pub mod adaptiveschedulingparams_traits;
pub mod dependencytracker_traits;
pub mod executionsessionconfig_traits;
pub mod functions;
pub mod priorityqueue_traits;
pub mod priorityweights_traits;
pub mod queuemanagementconfig_traits;
pub mod types;

// Re-export all types
pub use types::*;

#[cfg(test)]
mod functions_tests;
#[cfg(test)]
mod types_tests;

#[cfg(test)]
mod mod_tests {
    use super::*;
    use std::time::Duration;

    // ── ExecutionConstraintType ───────────────────────────────────────────

    #[test]
    fn test_execution_constraint_type_before_debug() {
        let t = ExecutionConstraintType::Before;
        assert!(format!("{:?}", t).contains("Before"));
    }

    #[test]
    fn test_execution_constraint_type_after_debug() {
        let t = ExecutionConstraintType::After;
        assert!(format!("{:?}", t).contains("After"));
    }

    #[test]
    fn test_execution_constraint_type_cannot_execute_with_debug() {
        let t = ExecutionConstraintType::CannotExecuteWith;
        assert!(format!("{:?}", t).contains("CannotExecuteWith"));
    }

    #[test]
    fn test_execution_constraint_type_requires_resource_debug() {
        let t = ExecutionConstraintType::RequiresResource;
        assert!(format!("{:?}", t).contains("RequiresResource"));
    }

    #[test]
    fn test_execution_constraint_type_custom_carries_name() {
        let t = ExecutionConstraintType::Custom("exclusive_gpu".to_string());
        let dbg = format!("{:?}", t);
        assert!(dbg.contains("Custom"));
    }

    // ── DependencyMetrics ─────────────────────────────────────────────────

    #[test]
    fn test_dependency_metrics_default() {
        let m = DependencyMetrics::default();
        assert_eq!(m.total_analyzed, 0);
        assert_eq!(m.resolution_time, Duration::ZERO);
        assert_eq!(m.cache_hit_rate, 0.0);
    }

    #[test]
    fn test_dependency_metrics_custom_values() {
        let m = DependencyMetrics {
            total_analyzed: 1500,
            resolution_time: Duration::from_millis(80),
            cache_hit_rate: 0.92_f64,
        };
        assert_eq!(m.total_analyzed, 1500);
        assert!((m.cache_hit_rate - 0.92).abs() < 1e-9);
    }

    // ── AvailableResources ────────────────────────────────────────────────

    #[test]
    fn test_available_resources_default() {
        let r = AvailableResources::default();
        assert_eq!(r.cpu_cores, 0.0);
        assert_eq!(r.memory_mb, 0);
        assert!(r.gpu_devices.is_empty());
        assert!(r.network_ports.is_empty());
        assert!(r.temp_directories.is_empty());
        assert_eq!(r.database_connections, 0);
        assert!(r.custom_resources.is_empty());
    }

    #[test]
    fn test_available_resources_custom_values() {
        let mut custom = std::collections::HashMap::new();
        custom.insert("fpga_slots".to_string(), 4.0_f64);
        let r = AvailableResources {
            cpu_cores: 32.0_f32,
            memory_mb: 65536,
            gpu_devices: vec![],
            network_ports: vec![8080, 8443],
            temp_directories: vec![],
            database_connections: 10,
            custom_resources: custom,
        };
        assert_eq!(r.cpu_cores, 32.0);
        assert_eq!(r.memory_mb, 65536);
        assert_eq!(r.network_ports.len(), 2);
        assert_eq!(r.database_connections, 10);
    }

    // ── PoolItemState ─────────────────────────────────────────────────────

    #[test]
    fn test_pool_item_state_maintenance_debug() {
        // All three variants
        let states = vec![
            PoolItemState::Available,
            PoolItemState::Allocated,
            PoolItemState::Maintenance,
        ];
        for s in &states {
            assert!(!format!("{:?}", s).is_empty());
        }
    }

    #[test]
    fn test_pool_item_state_clone() {
        let s = PoolItemState::Available;
        let t = s.clone();
        // Both should produce equal debug output
        assert_eq!(format!("{:?}", s), format!("{:?}", t));
    }

    // ── WorkStealingConfig ────────────────────────────────────────────────

    #[test]
    fn test_work_stealing_config_disabled() {
        let cfg = WorkStealingConfig {
            enabled: false,
            steal_threshold: 0.0,
            max_steals_per_interval: 0,
            steal_timeout: Duration::ZERO,
        };
        assert!(!cfg.enabled);
        assert_eq!(cfg.max_steals_per_interval, 0);
    }

    #[test]
    fn test_work_stealing_config_threshold_range() {
        let cfg = WorkStealingConfig {
            enabled: true,
            steal_threshold: 0.7_f32,
            max_steals_per_interval: 10,
            steal_timeout: Duration::from_millis(50),
        };
        assert!(cfg.steal_threshold >= 0.0 && cfg.steal_threshold <= 1.0);
        assert!(cfg.max_steals_per_interval > 0);
    }

    // ── RebalancingConfig ─────────────────────────────────────────────────

    #[test]
    fn test_rebalancing_config_disabled() {
        let cfg = RebalancingConfig {
            enabled: false,
            interval: Duration::from_secs(60),
            imbalance_threshold: 0.5_f32,
            aggressiveness: 0.3_f32,
            work_stealing: WorkStealingConfig {
                enabled: false,
                steal_threshold: 0.0,
                max_steals_per_interval: 0,
                steal_timeout: Duration::ZERO,
            },
        };
        assert!(!cfg.enabled);
        assert!(!cfg.work_stealing.enabled);
    }

    #[test]
    fn test_rebalancing_config_aggressiveness_bounds() {
        let cfg = RebalancingConfig {
            enabled: true,
            interval: Duration::from_secs(30),
            imbalance_threshold: 0.25_f32,
            aggressiveness: 0.8_f32,
            work_stealing: WorkStealingConfig {
                enabled: true,
                steal_threshold: 0.6,
                max_steals_per_interval: 3,
                steal_timeout: Duration::from_millis(100),
            },
        };
        assert!(cfg.aggressiveness >= 0.0 && cfg.aggressiveness <= 1.0);
        assert!(cfg.imbalance_threshold >= 0.0);
    }

    // ── WorkerScalingConfig ───────────────────────────────────────────────

    #[test]
    fn test_worker_scaling_config_threshold_ordering() {
        let cfg = WorkerScalingConfig {
            enabled: true,
            scale_up_threshold: 0.85_f32,
            scale_down_threshold: 0.25_f32,
            cooldown_period: Duration::from_secs(60),
            scaling_factor: 2.0_f32,
        };
        assert!(cfg.scale_up_threshold > cfg.scale_down_threshold);
        assert!(cfg.scaling_factor > 1.0);
    }

    #[test]
    fn test_worker_scaling_config_disabled() {
        let cfg = WorkerScalingConfig {
            enabled: false,
            scale_up_threshold: 0.9,
            scale_down_threshold: 0.1,
            cooldown_period: Duration::from_secs(120),
            scaling_factor: 1.0,
        };
        assert!(!cfg.enabled);
    }
}
