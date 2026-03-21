//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {

    use crate::test_independence_analyzer::{ConflictDetector, ConflictType};
    use crate::test_timeout_optimization::TestCategory;
    use crate::{TestParallelizationMetadata, TestResourceUsage};
    use std::collections::HashMap;
    use std::time::Duration;
    fn create_test_metadata(
        test_id: &str,
        cpu_cores: f32,
        memory_mb: u64,
    ) -> TestParallelizationMetadata {
        use crate::test_parallelization::{IsolationRequirements, ParallelizationHints};
        use crate::test_timeout_optimization::{TestComplexityHints, TestExecutionContext};
        TestParallelizationMetadata {
            base_context: TestExecutionContext {
                test_name: test_id.to_string(),
                category: TestCategory::Unit,
                environment: "test".to_string(),
                complexity_hints: TestComplexityHints::default(),
                expected_duration: Some(Duration::from_secs(10)),
                timeout_override: None,
            },
            dependencies: vec![],
            resource_usage: TestResourceUsage {
                test_id: test_id.to_string(),
                cpu_cores,
                memory_mb,
                gpu_devices: vec![],
                network_ports: vec![],
                temp_directories: vec![],
                database_connections: 0,
                duration: Duration::from_secs(10),
                priority: 1.0,
            },
            isolation_requirements: IsolationRequirements {
                process_isolation: false,
                network_isolation: false,
                filesystem_isolation: false,
                database_isolation: false,
                gpu_isolation: false,
                custom_isolation: HashMap::new(),
            },
            tags: vec![],
            priority: 1.0,
            parallelization_hints: ParallelizationHints {
                parallel_within_category: true,
                parallel_with_any: true,
                sequential_only: false,
                preferred_batch_size: None,
                optimal_concurrency: None,
                resource_sharing: crate::test_parallelization::ResourceSharingCapabilities {
                    cpu_sharing: true,
                    memory_sharing: false,
                    gpu_sharing: false,
                    network_sharing: true,
                    filesystem_sharing: false,
                },
            },
        }
    }
    #[test]
    fn test_detector_creation() {
        let detector = ConflictDetector::new();
        let stats = detector.get_statistics();
        assert_eq!(stats.total_conflicts_detected, 0);
    }
    #[test]
    fn test_cpu_conflict_detection() {
        let detector = ConflictDetector::new();
        let test1 = create_test_metadata("test1", 0.6, 512);
        let test2 = create_test_metadata("test2", 0.5, 256);
        let conflicts = detector
            .detect_conflicts_between_tests(&test1, &test2)
            .expect("test operation should succeed");
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_info.resource_type, "CPU");
        assert!(matches!(
            conflicts[0].conflict_info.conflict_type,
            ConflictType::CapacityLimit
        ));
    }
    #[test]
    fn test_gpu_conflict_detection() {
        let detector = ConflictDetector::new();
        let mut test1 = create_test_metadata("test1", 0.2, 256);
        test1.resource_usage.gpu_devices = vec![0, 1];
        let mut test2 = create_test_metadata("test2", 0.3, 512);
        test2.resource_usage.gpu_devices = vec![1, 2];
        let conflicts = detector
            .detect_conflicts_between_tests(&test1, &test2)
            .expect("test operation should succeed");
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_info.resource_type, "GPU");
        assert!(matches!(
            conflicts[0].conflict_info.conflict_type,
            ConflictType::ExclusiveAccess
        ));
    }
    #[test]
    fn test_no_conflict_detection() {
        let detector = ConflictDetector::new();
        let test1 = create_test_metadata("test1", 0.3, 256);
        let test2 = create_test_metadata("test2", 0.2, 128);
        let conflicts = detector
            .detect_conflicts_between_tests(&test1, &test2)
            .expect("test operation should succeed");
        assert_eq!(conflicts.len(), 0);
    }
    #[test]
    fn test_multiple_conflict_detection() {
        let detector = ConflictDetector::new();
        let mut test1 = create_test_metadata("test1", 0.6, 512);
        test1.resource_usage.gpu_devices = vec![0];
        test1.resource_usage.network_ports = vec![8080];
        let mut test2 = create_test_metadata("test2", 0.5, 256);
        test2.resource_usage.gpu_devices = vec![0];
        test2.resource_usage.network_ports = vec![8080];
        let conflicts = detector
            .detect_conflicts_between_tests(&test1, &test2)
            .expect("test operation should succeed");
        assert!(conflicts.len() >= 2);
        let resource_types: Vec<_> =
            conflicts.iter().map(|c| c.conflict_info.resource_type.as_str()).collect();
        assert!(resource_types.contains(&"CPU"));
        assert!(resource_types.contains(&"GPU"));
        assert!(resource_types.contains(&"Network"));
    }
}
