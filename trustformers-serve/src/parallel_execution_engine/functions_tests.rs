//! Comprehensive tests for parallel_execution_engine functions and configs
//!
//! Tests for ResourceRequirement defaults, AdaptiveSchedulingParams,
//! ExecutionSessionConfig, PriorityWeights, QueueManagementConfig,
//! and DependencyTracker.

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use std::collections::HashMap;
    use std::time::Duration;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        #[allow(dead_code)]
        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    #[test]
    fn test_resource_requirement_default_values() {
        let req = ResourceRequirement::default();
        assert_eq!(req.resource_type, "mixed");
        assert!((req.min_amount - 1.0).abs() < f64::EPSILON);
        assert!((req.cpu_cores - 1.0).abs() < f32::EPSILON);
        assert_eq!(req.memory_mb, 512);
        assert!(req.gpu_devices.is_empty());
        assert_eq!(req.network_ports, 0);
        assert_eq!(req.temp_directories, 0);
        assert_eq!(req.database_connections, 0);
        assert!(req.custom_resources.is_empty());
    }

    #[test]
    fn test_resource_requirement_with_gpu() {
        let req = ResourceRequirement {
            resource_type: "gpu_intensive".to_string(),
            min_amount: 4.0,
            cpu_cores: 8.0,
            memory_mb: 32768,
            gpu_devices: vec![0, 1, 2, 3],
            network_ports: 2,
            temp_directories: 1,
            database_connections: 0,
            custom_resources: {
                let mut map = HashMap::new();
                map.insert("vram_gb".to_string(), 16.0);
                map
            },
        };
        assert_eq!(req.gpu_devices.len(), 4);
        assert_eq!(req.memory_mb, 32768);
        assert!(req.custom_resources.contains_key("vram_gb"));
    }

    #[test]
    fn test_adaptive_scheduling_params_default() {
        let params = AdaptiveSchedulingParams::default();
        assert!((params.learning_rate - 0.1).abs() < f32::EPSILON);
        assert_eq!(params.adaptation_interval, Duration::from_secs(300));
        assert_eq!(params.history_window, 100);
        assert!((params.min_confidence - 0.7).abs() < f32::EPSILON);
        assert!((params.max_adaptation_rate - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adaptive_scheduling_params_custom() {
        let params = AdaptiveSchedulingParams {
            learning_rate: 0.05,
            adaptation_interval: Duration::from_secs(600),
            history_window: 200,
            min_confidence: 0.8,
            max_adaptation_rate: 0.5,
        };
        assert!((params.learning_rate - 0.05).abs() < f32::EPSILON);
        assert!(params.history_window > params.min_confidence as usize);
    }

    #[test]
    fn test_execution_session_config_default() {
        let config = ExecutionSessionConfig::default();
        assert!(config.max_concurrent_tests > 0);
        assert_eq!(config.session_timeout, Duration::from_secs(7200));
    }

    #[test]
    fn test_execution_session_config_custom() {
        let config = ExecutionSessionConfig {
            max_concurrent_tests: 8,
            session_timeout: Duration::from_secs(3600),
            failure_handling: crate::test_parallelization::FailureHandlingStrategy::StopDependent,
            early_termination:
                crate::test_parallelization::EarlyTerminationStrategy::ErrorRateThreshold(0.5),
        };
        assert_eq!(config.max_concurrent_tests, 8);
    }

    #[test]
    fn test_priority_weights_default() {
        let weights = PriorityWeights::default();
        assert!(weights.category_weight > 0.0);
        assert!(weights.duration_weight > 0.0);
        assert!(weights.resource_weight > 0.0);
        assert!(weights.dependency_weight > 0.0);
        assert!(weights.performance_weight > 0.0);
        assert!(weights.failure_rate_weight > 0.0);
    }

    #[test]
    fn test_priority_weights_custom() {
        let weights = PriorityWeights {
            category_weight: 0.3,
            duration_weight: 0.2,
            resource_weight: 0.15,
            dependency_weight: 0.15,
            performance_weight: 0.1,
            failure_rate_weight: 0.1,
        };
        let sum = weights.category_weight
            + weights.duration_weight
            + weights.resource_weight
            + weights.dependency_weight
            + weights.performance_weight
            + weights.failure_rate_weight;
        assert!((sum - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_queue_management_config_default() {
        let config = QueueManagementConfig::default();
        assert!(config.max_queue_size > 0);
        assert!(config.queue_timeout > Duration::ZERO);
        assert!(config.priority_boost_interval > Duration::ZERO);
        assert!(config.compaction_interval > Duration::ZERO);
    }

    #[test]
    fn test_queue_management_config_custom() {
        let config = QueueManagementConfig {
            max_queue_size: 500,
            queue_timeout: Duration::from_secs(120),
            priority_boost_interval: Duration::from_secs(30),
            starvation_prevention: true,
            compaction_interval: Duration::from_secs(60),
        };
        assert_eq!(config.max_queue_size, 500);
        assert!(config.starvation_prevention);
    }

    #[test]
    fn test_dependency_tracker_default() {
        let tracker = DependencyTracker::default();
        let metrics = tracker._metrics.lock();
        assert_eq!(metrics.total_analyzed, 0);
    }

    #[test]
    fn test_scheduling_config_default() {
        let config = SchedulingConfig::default();
        let _ = format!("{:?}", config);
    }

    #[test]
    fn test_execution_queue_metadata_default() {
        let metadata = ExecutionQueueMetadata::default();
        assert_eq!(metadata.total_queued, 0);
        assert_eq!(metadata.total_dequeued, 0);
        assert!((metadata.efficiency - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_resource_pool_config() {
        let config = ResourcePoolConfig {
            min_size: 5,
            max_size: 50,
            growth_strategy: PoolGrowthStrategy::OnDemand,
            cleanup_interval: Duration::from_secs(300),
            item_timeout: Duration::from_secs(600),
        };
        assert!(config.min_size <= config.max_size);
        assert!(matches!(
            config.growth_strategy,
            PoolGrowthStrategy::OnDemand
        ));
    }

    #[test]
    fn test_pool_item_creation() {
        let item = PoolItem {
            id: "item_001".to_string(),
            resource: "network_port_8080".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("type".to_string(), "tcp".to_string());
                m
            },
            state: PoolItemState::Available,
            last_used: None,
        };
        assert_eq!(item.id, "item_001");
        assert!(matches!(item.state, PoolItemState::Available));
        assert!(item.last_used.is_none());
    }

    #[test]
    fn test_pool_item_allocated_state() {
        let item = PoolItem {
            id: "item_002".to_string(),
            resource: "gpu_0".to_string(),
            metadata: HashMap::new(),
            state: PoolItemState::Allocated,
            last_used: Some(chrono::Utc::now()),
        };
        assert!(matches!(item.state, PoolItemState::Allocated));
        assert!(item.last_used.is_some());
    }

    #[test]
    fn test_alert_config_creation() {
        let config = AlertConfig {
            enabled: true,
            cooldown_period: Duration::from_secs(60),
            thresholds: AlertThresholds {
                high_error_rate: 0.1,
                high_latency: Duration::from_secs(5),
                resource_exhaustion: 0.9,
                queue_backup: 100,
                worker_failure: 3,
            },
            destinations: vec![AlertDestination {
                destination_type: AlertDestinationType::Log,
                config: HashMap::new(),
                alert_levels: vec![AlertLevel::Error, AlertLevel::Critical],
            }],
        };
        assert!(config.enabled);
        assert_eq!(config.destinations.len(), 1);
    }

    #[test]
    fn test_scheduling_constraint_creation() {
        let constraint = SchedulingConstraint {
            constraint_type: SchedulingConstraintType::Dependency,
            value: "test_prerequisite".to_string(),
            priority: 0.9,
            deadline: Some(chrono::Utc::now()),
        };
        assert!(matches!(
            constraint.constraint_type,
            SchedulingConstraintType::Dependency
        ));
        assert!(constraint.deadline.is_some());
    }

    #[test]
    fn test_execution_constraint() {
        let constraint = ExecutionConstraint {
            constraint_type: ExecutionConstraintType::Before,
            value: "setup_task".to_string(),
            priority: 1.0,
        };
        assert!(matches!(
            constraint.constraint_type,
            ExecutionConstraintType::Before
        ));
    }

    #[test]
    fn test_resource_allocation_state() {
        let state = ResourceAllocationState {
            allocated: crate::test_parallelization::ResourceAllocation {
                resource_type: "CPU".to_string(),
                resource_id: "alloc_001".to_string(),
                allocated_at: chrono::Utc::now(),
                deallocated_at: None,
                duration: Duration::ZERO,
                utilization: 0.8,
                efficiency: 1.0,
            },
            allocated_at: chrono::Utc::now(),
            expected_deallocation: None,
            efficiency: 0.95,
            metadata: HashMap::new(),
        };
        assert!((state.efficiency - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dependency_node_type_variants() {
        let test_node = DependencyNodeType::Test;
        let setup = DependencyNodeType::Setup;
        let teardown = DependencyNodeType::Teardown;
        let resource_init = DependencyNodeType::ResourceInit;
        let custom = DependencyNodeType::Custom("migration".to_string());
        assert!(matches!(test_node, DependencyNodeType::Test));
        assert!(matches!(setup, DependencyNodeType::Setup));
        assert!(matches!(teardown, DependencyNodeType::Teardown));
        assert!(matches!(resource_init, DependencyNodeType::ResourceInit));
        if let DependencyNodeType::Custom(name) = custom {
            assert_eq!(name, "migration");
        }
    }

    #[test]
    fn test_monitoring_config_creation() {
        let config = MonitoringConfig {
            monitoring_interval: Duration::from_secs(1),
            performance_tracking: PerformanceTrackingConfig {
                detailed_tracking: true,
                collection_interval: Duration::from_secs(10),
                retention_period: Duration::from_secs(3600),
                analysis_interval: Duration::from_secs(60),
                regression_detection: true,
            },
            health_checks: HealthCheckConfig {
                interval: Duration::from_secs(30),
                timeout: Duration::from_secs(10),
                failure_threshold: 3,
                recovery_interval: Duration::from_secs(60),
                deep_checks: false,
            },
            alerts: AlertConfig {
                enabled: true,
                cooldown_period: Duration::from_secs(60),
                thresholds: AlertThresholds {
                    high_error_rate: 0.1,
                    high_latency: Duration::from_secs(5),
                    resource_exhaustion: 0.9,
                    queue_backup: 100,
                    worker_failure: 3,
                },
                destinations: vec![],
            },
        };
        assert!(config.performance_tracking.detailed_tracking);
        assert_eq!(config.health_checks.failure_threshold, 3);
    }

    #[test]
    fn test_lcg_deterministic() {
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(42);
        for _ in 0..20 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_lcg_different_seeds() {
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(43);
        let v1 = rng1.next();
        let v2 = rng2.next();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_worker_specialization_config() {
        let config = WorkerSpecializationConfig {
            enabled: true,
            by_category: true,
            by_resource: false,
            by_performance: true,
        };
        assert!(config.enabled);
        assert!(config.by_category);
        assert!(!config.by_resource);
        assert!(config.by_performance);
    }
}
