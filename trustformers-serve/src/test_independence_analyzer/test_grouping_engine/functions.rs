//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::test_independence_analyzer::types::*;
use crate::test_parallelization::TestResourceUsage;

/// Helper implementation for ResourceRequirement
impl ResourceRequirement {
    /// Create from TestResourceUsage with comprehensive resource analysis
    pub fn from_resource_usage(usage: &TestResourceUsage) -> Self {
        use crate::test_independence_analyzer::types::{RequirementFlexibility, UsagePriority};
        let cpu_cost = usage.cpu_cores as f64;
        let memory_cost = (usage.memory_mb as f64) / 1024.0;
        let gpu_cost = (usage.gpu_devices.len() as f64) * 2.0;
        let port_cost = (usage.network_ports.len() as f64) * 0.1;
        let total_cost = cpu_cost + memory_cost + gpu_cost + port_cost;
        let resource_type = if gpu_cost > cpu_cost && gpu_cost > memory_cost {
            "gpu_device"
        } else if memory_cost > cpu_cost {
            "memory"
        } else {
            "cpu"
        };
        let priority = if total_cost > 10.0 {
            UsagePriority::High
        } else if total_cost > 5.0 {
            UsagePriority::Normal
        } else {
            UsagePriority::Low
        };
        let flexibility = if !usage.gpu_devices.is_empty() {
            RequirementFlexibility::Strict
        } else if usage.cpu_cores < 2.0 {
            RequirementFlexibility::Flexible
        } else {
            RequirementFlexibility::Flexible
        };
        Self {
            resource_type: resource_type.to_string(),
            min_amount: total_cost,
            max_amount: total_cost * 1.2,
            priority,
            flexibility,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::super::types::*;
    use super::*;
    use crate::test_parallelization::TestParallelizationMetadata;
    use crate::test_timeout_optimization::{
        TestCategory, TestComplexityHints, TestExecutionContext,
    };
    use std::collections::HashMap;
    use std::time::Duration;
    fn create_test_metadata(test_id: &str, category: TestCategory) -> TestParallelizationMetadata {
        TestParallelizationMetadata {
            base_context: TestExecutionContext {
                test_name: test_id.to_string(),
                category,
                environment: "test".to_string(),
                complexity_hints: TestComplexityHints::default(),
                expected_duration: Some(Duration::from_secs(10)),
                timeout_override: None,
            },
            dependencies: vec![],
            resource_usage: TestResourceUsage {
                test_id: test_id.to_string(),
                cpu_cores: 0.2,
                memory_mb: 256,
                gpu_devices: vec![],
                network_ports: vec![],
                temp_directories: vec![],
                database_connections: 0,
                duration: Duration::from_secs(10),
                priority: 1.0,
            },
            isolation_requirements: crate::test_parallelization::IsolationRequirements {
                process_isolation: false,
                network_isolation: false,
                filesystem_isolation: false,
                database_isolation: false,
                gpu_isolation: false,
                custom_isolation: HashMap::new(),
            },
            tags: vec![],
            priority: 1.0,
            parallelization_hints: crate::test_parallelization::ParallelizationHints {
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
    fn test_grouping_engine_creation() {
        let engine = TestGroupingEngine::new();
        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_groupings, 0);
    }
    #[test]
    fn test_basic_grouping() {
        let engine = TestGroupingEngine::new();
        let tests = vec![
            create_test_metadata("test1", TestCategory::Unit),
            create_test_metadata("test2", TestCategory::Unit),
            create_test_metadata("test3", TestCategory::Integration),
            create_test_metadata("test4", TestCategory::Integration),
        ];
        let groups = engine
            .create_test_groups(&tests, &[], &[])
            .expect("creation should succeed in test");
        assert!(!groups.is_empty());
        assert!(groups.iter().all(|g| !g.tests.is_empty()));
        let total_assigned_tests: usize = groups.iter().map(|g| g.tests.len()).sum();
        assert_eq!(total_assigned_tests, tests.len());
    }
    #[test]
    fn test_conflict_aware_grouping() {
        let engine = TestGroupingEngine::new();
        let tests = vec![
            create_test_metadata("test1", TestCategory::Unit),
            create_test_metadata("test2", TestCategory::Unit),
        ];
        let conflicts = vec![ResourceConflict {
            id: "conflict1".to_string(),
            resource_id: "cpu_0".to_string(),
            test1: "test1".to_string(),
            test2: "test2".to_string(),
            resource_type: "CPU".to_string(),
            conflict_type: ConflictType::ExclusiveAccess,
            severity: ConflictSeverity::High,
            description: "CPU conflict".to_string(),
            resolution_strategies: vec![],
            metadata: ConflictMetadata {
                detected_at: chrono::Utc::now(),
                detection_method: "test".to_string(),
                confidence: 1.0,
                historical_occurrences: 0,
                last_occurrence: None,
            },
        }];
        let groups = engine
            .create_test_groups(&tests, &[], &conflicts)
            .expect("creation should succeed in test");
        let test1_groups: Vec<_> =
            groups.iter().filter(|g| g.tests.contains(&"test1".to_string())).collect();
        let test2_groups: Vec<_> =
            groups.iter().filter(|g| g.tests.contains(&"test2".to_string())).collect();
        assert!(test1_groups.len() > 0 && test2_groups.len() > 0);
    }
    #[test]
    fn test_group_characteristics_calculation() {
        let engine = TestGroupingEngine::new();
        let tests = vec![
            create_test_metadata("test1", TestCategory::Unit),
            create_test_metadata("test2", TestCategory::Unit),
        ];
        let test_names = vec!["test1".to_string(), "test2".to_string()];
        let characteristics = engine.calculate_group_characteristics(&test_names, &tests);
        assert!(characteristics.homogeneity >= 0.0 && characteristics.homogeneity <= 1.0);
        assert!(
            characteristics.resource_compatibility >= 0.0
                && characteristics.resource_compatibility <= 1.0
        );
        assert!(characteristics.overall_quality >= 0.0 && characteristics.overall_quality <= 1.0);
    }
}
