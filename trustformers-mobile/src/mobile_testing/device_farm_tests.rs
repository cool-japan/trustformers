#[cfg(test)]
mod tests {
    use crate::mobile_testing::config::*;
    use crate::mobile_testing::device_farm::*;
    use crate::mobile_testing::results::*;
    use std::collections::HashMap;
    use std::time::{Duration, SystemTime};

    // --- SessionStatus Tests ---

    #[test]
    fn test_session_status_equality() {
        assert_eq!(SessionStatus::Pending, SessionStatus::Pending);
        assert_ne!(SessionStatus::Running, SessionStatus::Completed);
        assert_ne!(SessionStatus::Failed, SessionStatus::Cancelled);
    }

    #[test]
    fn test_session_status_serialization() {
        let status = SessionStatus::Running;
        let json = serde_json::to_string(&status).expect("Failed to serialize");
        let deserialized: SessionStatus =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, SessionStatus::Running);
    }

    #[test]
    fn test_session_status_all_variants() {
        let variants = vec![
            SessionStatus::Pending,
            SessionStatus::Running,
            SessionStatus::Completed,
            SessionStatus::Failed,
            SessionStatus::Cancelled,
        ];
        assert_eq!(variants.len(), 5);
        for v in &variants {
            let json = serde_json::to_string(v).expect("Failed to serialize");
            let _: SessionStatus = serde_json::from_str(&json).expect("Failed to deserialize");
        }
    }

    // --- TaskPriority Tests ---

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Low < TaskPriority::Normal);
        assert!(TaskPriority::Normal < TaskPriority::High);
        assert!(TaskPriority::High < TaskPriority::Critical);
        assert!(TaskPriority::Critical < TaskPriority::Urgent);
    }

    #[test]
    fn test_task_priority_equality() {
        assert_eq!(TaskPriority::High, TaskPriority::High);
        assert_ne!(TaskPriority::Low, TaskPriority::Urgent);
    }

    #[test]
    fn test_task_priority_serialization() {
        let priority = TaskPriority::Critical;
        let json = serde_json::to_string(&priority).expect("Failed to serialize");
        let deserialized: TaskPriority =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized, TaskPriority::Critical);
    }

    #[test]
    fn test_task_priority_clone_copy() {
        let p = TaskPriority::Normal;
        let p2 = p;
        assert_eq!(p, p2);
    }

    // --- TestType Tests ---

    #[test]
    fn test_test_type_variants() {
        let types = vec![
            TestType::Benchmark,
            TestType::Battery,
            TestType::Stress,
            TestType::Memory,
            TestType::Compatibility,
            TestType::Performance,
            TestType::FullSuite,
        ];
        assert_eq!(types.len(), 7);
    }

    #[test]
    fn test_test_type_serialization() {
        let test_type = TestType::Benchmark;
        let json = serde_json::to_string(&test_type).expect("Failed to serialize");
        let deserialized: TestType = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(format!("{:?}", deserialized), "Benchmark");
    }

    #[test]
    fn test_test_type_debug() {
        assert_eq!(format!("{:?}", TestType::Memory), "Memory");
        assert_eq!(format!("{:?}", TestType::Stress), "Stress");
    }

    // --- TestTask Tests ---

    #[test]
    fn test_test_task_creation_minimal() {
        let task = TestTask {
            task_id: "task-001".to_string(),
            test_config: TestExecutionConfig {
                test_type: TestType::Benchmark,
                timeout: Duration::from_secs(120),
                retry_attempts: 2,
                resource_requirements: HardwareRequirements {
                    min_ram_mb: 4096,
                    min_storage_gb: 32,
                    min_cpu_cores: 4,
                    required_sensors: vec![],
                    required_connectivity: vec![],
                },
            },
            assigned_device: None,
            priority: TaskPriority::Normal,
            status: SessionStatus::Pending,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
        };
        assert_eq!(task.task_id, "task-001");
        assert!(task.assigned_device.is_none());
        assert!(task.started_at.is_none());
    }

    #[test]
    fn test_test_task_with_device() {
        let task = TestTask {
            task_id: "task-002".to_string(),
            test_config: TestExecutionConfig {
                test_type: TestType::Memory,
                timeout: Duration::from_secs(60),
                retry_attempts: 0,
                resource_requirements: HardwareRequirements {
                    min_ram_mb: 2048,
                    min_storage_gb: 16,
                    min_cpu_cores: 2,
                    required_sensors: vec![],
                    required_connectivity: vec![],
                },
            },
            assigned_device: Some("device-001".to_string()),
            priority: TaskPriority::High,
            status: SessionStatus::Running,
            created_at: SystemTime::now(),
            started_at: Some(SystemTime::now()),
            completed_at: None,
        };
        assert!(task.assigned_device.is_some());
        assert!(task.started_at.is_some());
    }

    #[test]
    fn test_test_task_clone() {
        let task = TestTask {
            task_id: "task-003".to_string(),
            test_config: TestExecutionConfig {
                test_type: TestType::Battery,
                timeout: Duration::from_secs(300),
                retry_attempts: 1,
                resource_requirements: HardwareRequirements {
                    min_ram_mb: 1024,
                    min_storage_gb: 8,
                    min_cpu_cores: 1,
                    required_sensors: vec![],
                    required_connectivity: vec![],
                },
            },
            assigned_device: None,
            priority: TaskPriority::Low,
            status: SessionStatus::Pending,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
        };
        let cloned = task.clone();
        assert_eq!(cloned.task_id, "task-003");
    }

    // --- StatisticalMethod Tests ---

    #[test]
    fn test_statistical_method_variants() {
        assert_eq!(StatisticalMethod::Mean, StatisticalMethod::Mean);
        assert_ne!(StatisticalMethod::Median, StatisticalMethod::Mode);
        let _ = StatisticalMethod::StandardDeviation;
        let _ = StatisticalMethod::Variance;
        let _ = StatisticalMethod::Range;
    }

    #[test]
    fn test_statistical_method_percentile() {
        let p95 = StatisticalMethod::Percentile(95);
        let p99 = StatisticalMethod::Percentile(99);
        assert_ne!(p95, p99);
        assert_eq!(StatisticalMethod::Percentile(95), StatisticalMethod::Percentile(95));
    }

    #[test]
    fn test_statistical_method_clone_copy() {
        let m = StatisticalMethod::Mean;
        let m2 = m;
        assert_eq!(m, m2);
    }

    // --- AggregationRules Tests ---

    #[test]
    fn test_aggregation_rules_creation() {
        let rules = AggregationRules {
            statistical_methods: vec![
                StatisticalMethod::Mean,
                StatisticalMethod::Median,
                StatisticalMethod::Percentile(95),
            ],
            outlier_detection: true,
            confidence_level: 0.95,
            minimum_sample_size: 10,
        };
        assert_eq!(rules.statistical_methods.len(), 3);
        assert!(rules.outlier_detection);
    }

    #[test]
    fn test_aggregation_rules_serialization() {
        let rules = AggregationRules {
            statistical_methods: vec![StatisticalMethod::Mean],
            outlier_detection: false,
            confidence_level: 0.9,
            minimum_sample_size: 5,
        };
        let json = serde_json::to_string(&rules).expect("Failed to serialize");
        let deserialized: AggregationRules =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.minimum_sample_size, 5);
    }

    // --- ResultAggregator Tests ---

    #[test]
    fn test_result_aggregator_creation() {
        let rules = AggregationRules {
            statistical_methods: vec![StatisticalMethod::Mean],
            outlier_detection: false,
            confidence_level: 0.9,
            minimum_sample_size: 5,
        };
        let aggregator = ResultAggregator::new(rules);
        let _ = format!("{:?}", aggregator);
    }

    // --- DeviceInfo Tests ---

    #[test]
    fn test_device_info_creation() {
        let info = DeviceInfo {
            device_name: "test-device".to_string(),
            os_name: "Android".to_string(),
            os_version: "14".to_string(),
            device_type: DeviceType::Phone,
            hardware_model: "Pixel 8".to_string(),
            cpu_architecture: "aarch64".to_string(),
            ram_mb: 8192,
            storage_gb: 256,
            screen_resolution: (1080, 2400),
            sensors: vec!["accelerometer".to_string(), "gyroscope".to_string()],
        };
        assert_eq!(info.device_name, "test-device");
        assert_eq!(info.ram_mb, 8192);
        assert_eq!(info.screen_resolution, (1080, 2400));
    }

    #[test]
    fn test_device_info_clone() {
        let info = DeviceInfo {
            device_name: "clone-device".to_string(),
            os_name: "iOS".to_string(),
            os_version: "17.0".to_string(),
            device_type: DeviceType::Phone,
            hardware_model: "iPhone 15".to_string(),
            cpu_architecture: "arm64".to_string(),
            ram_mb: 6144,
            storage_gb: 128,
            screen_resolution: (1170, 2532),
            sensors: vec!["camera".to_string()],
        };
        let cloned = info.clone();
        assert_eq!(cloned.device_name, "clone-device");
    }

    // --- StatisticalSummary Tests ---

    #[test]
    fn test_statistical_summary_creation() {
        let summary = StatisticalSummary {
            mean: 50.0,
            median: 48.0,
            std_deviation: 10.0,
            min: 30.0,
            max: 80.0,
            percentiles: HashMap::from([
                ("P95".to_string(), 70.0),
                ("P99".to_string(), 75.0),
            ]),
        };
        assert!((summary.mean - 50.0).abs() < 1e-5);
        assert!(summary.min < summary.max);
        assert_eq!(summary.percentiles.len(), 2);
    }

    // --- CrossDeviceAnalysis Tests ---

    #[test]
    fn test_cross_device_analysis_creation() {
        let analysis = CrossDeviceAnalysis {
            performance_variance: 0.15,
            best_device: "iphone-14".to_string(),
            worst_device: "old-device".to_string(),
            compatibility_rate: 0.98,
        };
        assert!((analysis.compatibility_rate - 0.98).abs() < 1e-5);
    }

    // --- HardwareRequirements Tests ---

    #[test]
    fn test_hardware_requirements_creation() {
        let reqs = HardwareRequirements {
            min_ram_mb: 4096,
            min_storage_gb: 64,
            min_cpu_cores: 4,
            required_sensors: vec!["gps".to_string()],
            required_connectivity: vec!["wifi".to_string()],
        };
        assert_eq!(reqs.min_ram_mb, 4096);
        assert_eq!(reqs.required_sensors.len(), 1);
    }

    // --- DeviceType Tests ---

    #[test]
    fn test_device_type_variants() {
        let _ = DeviceType::Phone;
        let _ = DeviceType::Tablet;
        let _ = DeviceType::Generic;
    }
}
