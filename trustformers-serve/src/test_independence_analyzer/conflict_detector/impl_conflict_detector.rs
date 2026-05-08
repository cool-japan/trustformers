//! ConflictDetector method implementations
//!
//! Split from types.rs to keep individual files under 2000 lines.

use super::types::*;
use crate::test_independence_analyzer::types::*;
use crate::test_parallelization::{TestParallelizationMetadata, TestResourceUsage};
use chrono::Utc;
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};
use tracing::debug;

impl ConflictDetector {
    /// Create a new conflict detector with default configuration
    pub fn new() -> Self {
        Self::with_config(ConflictDetectionConfig::default())
    }
    /// Create a new conflict detector with custom configuration
    pub fn with_config(config: ConflictDetectionConfig) -> Self {
        Self {
            config: std::sync::Arc::new(parking_lot::RwLock::new(config)),
            _detection_rules: std::sync::Arc::new(parking_lot::RwLock::new(
                Self::create_default_rules(),
            )),
            _detected_conflicts: std::sync::Arc::new(parking_lot::RwLock::new(HashMap::new())),
            _resolution_strategies: std::sync::Arc::new(parking_lot::RwLock::new(
                Self::create_default_strategies(),
            )),
            statistics: std::sync::Arc::new(parking_lot::RwLock::new(
                ConflictDetectionStatistics::default(),
            )),
            _learned_patterns: std::sync::Arc::new(parking_lot::RwLock::new(Vec::new())),
        }
    }
    /// Detect conflicts between a pair of tests
    pub fn detect_conflicts_between_tests(
        &self,
        test1: &TestParallelizationMetadata,
        test2: &TestParallelizationMetadata,
    ) -> AnalysisResult<Vec<DetectedConflict>> {
        let start_time = Instant::now();
        let config = self.config.read();
        if start_time.elapsed() > config.max_analysis_time {
            return Err(AnalysisError::AnalysisTimeout {
                message: format!(
                    "conflict detection timed out after {:?}",
                    config.max_analysis_time
                ),
            });
        }
        let mut conflicts = Vec::new();
        conflicts.extend(self.detect_resource_conflicts(test1, test2)?);
        conflicts.extend(self.detect_pattern_conflicts(test1, test2)?);
        conflicts.extend(self.detect_dependency_conflicts(test1, test2)?);
        if config.enable_ml_patterns {
            conflicts.extend(self.detect_ml_conflicts(test1, test2)?);
        }
        let filtered_conflicts: Vec<_> = conflicts
            .into_iter()
            .filter(|c| c.confidence >= config.confidence_threshold)
            .collect();
        self.update_detection_statistics(&filtered_conflicts, start_time.elapsed());
        if config.detailed_logging {
            debug!(
                "Detected {} conflicts between {} and {} in {:?}",
                filtered_conflicts.len(),
                test1.base_context.test_name,
                test2.base_context.test_name,
                start_time.elapsed()
            );
        }
        Ok(filtered_conflicts)
    }
    /// Detect conflicts across multiple tests
    pub fn detect_conflicts_in_test_set(
        &self,
        tests: &[TestParallelizationMetadata],
    ) -> AnalysisResult<Vec<DetectedConflict>> {
        let mut all_conflicts = Vec::new();
        for (i, test1) in tests.iter().enumerate() {
            for test2 in tests.iter().skip(i + 1) {
                let conflicts = self.detect_conflicts_between_tests(test1, test2)?;
                all_conflicts.extend(conflicts);
            }
        }
        all_conflicts = self.deduplicate_conflicts(all_conflicts);
        all_conflicts.sort_by(|a, b| {
            b.conflict_info
                .severity
                .partial_cmp(&a.conflict_info.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        Ok(all_conflicts)
    }
    /// Detect resource-based conflicts
    fn detect_resource_conflicts(
        &self,
        test1: &TestParallelizationMetadata,
        test2: &TestParallelizationMetadata,
    ) -> AnalysisResult<Vec<DetectedConflict>> {
        let mut conflicts = Vec::new();
        if let Some(conflict) =
            self.detect_cpu_conflict(&test1.resource_usage, &test2.resource_usage)?
        {
            conflicts.push(conflict);
        }
        if let Some(conflict) =
            self.detect_memory_conflict(&test1.resource_usage, &test2.resource_usage)?
        {
            conflicts.push(conflict);
        }
        if let Some(conflict) =
            self.detect_gpu_conflict(&test1.resource_usage, &test2.resource_usage)?
        {
            conflicts.push(conflict);
        }
        if let Some(conflict) =
            self.detect_network_conflict(&test1.resource_usage, &test2.resource_usage)?
        {
            conflicts.push(conflict);
        }
        if let Some(conflict) =
            self.detect_filesystem_conflict(&test1.resource_usage, &test2.resource_usage)?
        {
            conflicts.push(conflict);
        }
        if let Some(conflict) =
            self.detect_database_conflict(&test1.resource_usage, &test2.resource_usage)?
        {
            conflicts.push(conflict);
        }
        Ok(conflicts)
    }
    /// Detect CPU resource conflicts
    fn detect_cpu_conflict(
        &self,
        usage1: &TestResourceUsage,
        usage2: &TestResourceUsage,
    ) -> AnalysisResult<Option<DetectedConflict>> {
        let config = self.config.read();
        let cpu_threshold = config.resource_thresholds.cpu_threshold;
        let combined_cpu_usage = usage1.cpu_cores + usage2.cpu_cores;
        if combined_cpu_usage > cpu_threshold {
            let conflict = DetectedConflict {
                conflict_id: format!("cpu_conflict_{}_{}", usage1.test_id, usage2.test_id),
                conflict_info: ResourceConflict {
                    id: format!("cpu_{}_{}", usage1.test_id, usage2.test_id),
                    test1: usage1.test_id.clone(),
                    test2: usage2.test_id.clone(),
                    resource_type: "CPU".to_string(),
                    resource_id: format!("cpu_combined_{}_{}", usage1.test_id, usage2.test_id),
                    conflict_type: ConflictType::CapacityLimit,
                    severity: self
                        .calculate_cpu_conflict_severity(combined_cpu_usage, cpu_threshold),
                    description: format!(
                        "CPU usage conflict: combined usage ({:.2}) exceeds threshold ({:.2})",
                        combined_cpu_usage, cpu_threshold
                    ),
                    resolution_strategies: vec![],
                    metadata: ConflictMetadata {
                        detected_at: Utc::now(),
                        detection_method: "rule_based".to_string(),
                        confidence: 0.9,
                        historical_occurrences: 1,
                        last_occurrence: None,
                    },
                },
                detection_details: ConflictDetectionDetails {
                    triggered_rules: vec!["cpu_capacity_rule".to_string()],
                    detection_method: ConflictDetectionMethod::RuleBased,
                    analysis_duration: Duration::from_millis(5),
                    evidence: vec![ConflictEvidence {
                        evidence_type: ConflictEvidenceType::ResourceOverlap,
                        description: "CPU usage exceeds capacity threshold".to_string(),
                        strength: (combined_cpu_usage - cpu_threshold) / cpu_threshold,
                        data: [
                            ("test1_usage".to_string(), usage1.cpu_cores.to_string()),
                            ("test2_usage".to_string(), usage2.cpu_cores.to_string()),
                            ("combined_usage".to_string(), combined_cpu_usage.to_string()),
                            ("threshold".to_string(), cpu_threshold.to_string()),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    }],
                    false_positive_probability: 0.1,
                },
                resolution_options: self.generate_cpu_resolution_options(usage1, usage2),
                impact_analysis: self
                    .analyze_cpu_conflict_impact(combined_cpu_usage, cpu_threshold),
                detected_at: Utc::now(),
                confidence: 0.9,
            };
            return Ok(Some(conflict));
        }
        Ok(None)
    }
    /// Detect memory resource conflicts
    fn detect_memory_conflict(
        &self,
        usage1: &TestResourceUsage,
        usage2: &TestResourceUsage,
    ) -> AnalysisResult<Option<DetectedConflict>> {
        let config = self.config.read();
        let memory_threshold = config.resource_thresholds.memory_threshold;
        let combined_memory_usage = (usage1.memory_mb + usage2.memory_mb) as f32 / 1024.0;
        if combined_memory_usage > memory_threshold {
            let conflict = DetectedConflict {
                conflict_id: format!(
                    "memory_conflict_{}_{}", usage1.test_id, usage2.test_id
                ),
                conflict_info: ResourceConflict {
                    id: format!("memory_{}_{}", usage1.test_id, usage2.test_id),
                    test1: usage1.test_id.clone(),
                    test2: usage2.test_id.clone(),
                    resource_type: "Memory".to_string(),
                    resource_id: format!(
                        "memory_combined_{}_{}", usage1.test_id, usage2.test_id
                    ),
                    conflict_type: ConflictType::CapacityLimit,
                    severity: self.calculate_memory_conflict_severity(
                        combined_memory_usage,
                        memory_threshold,
                    ),
                    description: format!(
                        "Memory usage conflict: combined usage ({:.2} GB) exceeds threshold ({:.2} GB)",
                        combined_memory_usage, memory_threshold
                    ),
                    resolution_strategies: vec![],
                    metadata: ConflictMetadata {
                        detected_at: Utc::now(),
                        detection_method: "rule_based".to_string(),
                        confidence: 0.85,
                        historical_occurrences: 1,
                        last_occurrence: None,
                    },
                },
                detection_details: ConflictDetectionDetails {
                    triggered_rules: vec!["memory_capacity_rule".to_string()],
                    detection_method: ConflictDetectionMethod::RuleBased,
                    analysis_duration: Duration::from_millis(3),
                    evidence: vec![ConflictEvidence {
                        evidence_type: ConflictEvidenceType::ResourceOverlap,
                        description: "Memory usage exceeds capacity threshold".to_string(),
                        strength: (combined_memory_usage - memory_threshold) / memory_threshold,
                        data: [
                            ("test1_usage".to_string(), format!("{} MB", usage1.memory_mb)),
                            ("test2_usage".to_string(), format!("{} MB", usage2.memory_mb)),
                            (
                                "combined_usage".to_string(),
                                format!("{:.2} GB", combined_memory_usage),
                            ),
                            (
                                "threshold".to_string(),
                                format!("{:.2} GB", memory_threshold),
                            ),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    }],
                    false_positive_probability: 0.15,
                },
                resolution_options: self.generate_memory_resolution_options(usage1, usage2),
                impact_analysis: self
                    .analyze_memory_conflict_impact(combined_memory_usage, memory_threshold),
                detected_at: Utc::now(),
                confidence: 0.85,
            };
            return Ok(Some(conflict));
        }
        Ok(None)
    }
    /// Detect GPU resource conflicts
    fn detect_gpu_conflict(
        &self,
        usage1: &TestResourceUsage,
        usage2: &TestResourceUsage,
    ) -> AnalysisResult<Option<DetectedConflict>> {
        let gpu1: HashSet<_> = usage1.gpu_devices.iter().collect();
        let gpu2: HashSet<_> = usage2.gpu_devices.iter().collect();
        let overlap: Vec<_> = gpu1.intersection(&gpu2).cloned().collect();
        if !overlap.is_empty() {
            let conflict = DetectedConflict {
                conflict_id: format!(
                    "gpu_conflict_{}_{}", usage1.test_id, usage2.test_id
                ),
                conflict_info: ResourceConflict {
                    id: format!("gpu_{}_{}", usage1.test_id, usage2.test_id),
                    test1: usage1.test_id.clone(),
                    test2: usage2.test_id.clone(),
                    resource_type: "GPU".to_string(),
                    resource_id: format!("gpu_devices_{:?}", overlap),
                    conflict_type: ConflictType::ExclusiveAccess,
                    severity: ConflictSeverity::High,
                    description: format!(
                        "GPU device conflict: both tests require exclusive access to GPU devices {:?}",
                        overlap
                    ),
                    resolution_strategies: vec![],
                    metadata: ConflictMetadata {
                        detected_at: Utc::now(),
                        detection_method: "rule_based".to_string(),
                        confidence: 0.95,
                        historical_occurrences: 1,
                        last_occurrence: None,
                    },
                },
                detection_details: ConflictDetectionDetails {
                    triggered_rules: vec!["gpu_exclusive_access_rule".to_string()],
                    detection_method: ConflictDetectionMethod::RuleBased,
                    analysis_duration: Duration::from_millis(2),
                    evidence: vec![ConflictEvidence {
                        evidence_type: ConflictEvidenceType::ResourceOverlap,
                        description: "Overlapping GPU device requirements".to_string(),
                        strength: overlap.len() as f32 / gpu1.len().max(gpu2.len()) as f32,
                        data: [
                            ("test1_gpus".to_string(), format!("{:?}", usage1.gpu_devices)),
                            ("test2_gpus".to_string(), format!("{:?}", usage2.gpu_devices)),
                            ("overlapping_gpus".to_string(), format!("{:?}", overlap)),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    }],
                    false_positive_probability: 0.05,
                },
                resolution_options: self.generate_gpu_resolution_options(usage1, usage2),
                impact_analysis: self.analyze_gpu_conflict_impact(&overlap),
                detected_at: Utc::now(),
                confidence: 0.95,
            };
            return Ok(Some(conflict));
        }
        Ok(None)
    }
    /// Detect network resource conflicts
    fn detect_network_conflict(
        &self,
        usage1: &TestResourceUsage,
        usage2: &TestResourceUsage,
    ) -> AnalysisResult<Option<DetectedConflict>> {
        let ports1: HashSet<_> = usage1.network_ports.iter().collect();
        let ports2: HashSet<_> = usage2.network_ports.iter().collect();
        let overlap: Vec<_> = ports1.intersection(&ports2).cloned().collect();
        if !overlap.is_empty() {
            let conflict = DetectedConflict {
                conflict_id: format!("network_conflict_{}_{}", usage1.test_id, usage2.test_id),
                conflict_info: ResourceConflict {
                    id: format!("network_{}_{}", usage1.test_id, usage2.test_id),
                    test1: usage1.test_id.clone(),
                    test2: usage2.test_id.clone(),
                    resource_type: "Network".to_string(),
                    resource_id: format!("network_ports_{:?}", overlap),
                    conflict_type: ConflictType::ExclusiveAccess,
                    severity: ConflictSeverity::Medium,
                    description: format!(
                        "Network port conflict: both tests require ports {:?}",
                        overlap
                    ),
                    resolution_strategies: vec![],
                    metadata: ConflictMetadata {
                        detected_at: Utc::now(),
                        detection_method: "rule_based".to_string(),
                        confidence: 0.9,
                        historical_occurrences: 1,
                        last_occurrence: None,
                    },
                },
                detection_details: ConflictDetectionDetails {
                    triggered_rules: vec!["network_port_rule".to_string()],
                    detection_method: ConflictDetectionMethod::RuleBased,
                    analysis_duration: Duration::from_millis(3),
                    evidence: vec![ConflictEvidence {
                        evidence_type: ConflictEvidenceType::ResourceOverlap,
                        description: "Overlapping network port requirements".to_string(),
                        strength: overlap.len() as f32 / ports1.len().max(ports2.len()) as f32,
                        data: [
                            (
                                "test1_ports".to_string(),
                                format!("{:?}", usage1.network_ports),
                            ),
                            (
                                "test2_ports".to_string(),
                                format!("{:?}", usage2.network_ports),
                            ),
                            ("overlapping_ports".to_string(), format!("{:?}", overlap)),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    }],
                    false_positive_probability: 0.1,
                },
                resolution_options: self.generate_network_resolution_options(usage1, usage2),
                impact_analysis: self.analyze_network_conflict_impact(&overlap),
                detected_at: Utc::now(),
                confidence: 0.9,
            };
            return Ok(Some(conflict));
        }
        Ok(None)
    }
    /// Detect filesystem resource conflicts
    fn detect_filesystem_conflict(
        &self,
        usage1: &TestResourceUsage,
        usage2: &TestResourceUsage,
    ) -> AnalysisResult<Option<DetectedConflict>> {
        let dirs1: HashSet<_> = usage1.temp_directories.iter().collect();
        let dirs2: HashSet<_> = usage2.temp_directories.iter().collect();
        let overlap: Vec<_> = dirs1.intersection(&dirs2).cloned().collect();
        if !overlap.is_empty() {
            let conflict = DetectedConflict {
                conflict_id: format!(
                    "filesystem_conflict_{}_{}", usage1.test_id, usage2.test_id
                ),
                conflict_info: ResourceConflict {
                    id: format!("filesystem_{}_{}", usage1.test_id, usage2.test_id),
                    test1: usage1.test_id.clone(),
                    test2: usage2.test_id.clone(),
                    resource_type: "Filesystem".to_string(),
                    resource_id: format!(
                        "filesystem_dirs_{:?}",
                        overlap.iter().take(3).collect::<Vec<_>>()
                    ),
                    conflict_type: ConflictType::DataCorruption,
                    severity: ConflictSeverity::High,
                    description: format!(
                        "Filesystem conflict: both tests use overlapping directories {:?}",
                        overlap
                    ),
                    resolution_strategies: vec![],
                    metadata: ConflictMetadata {
                        detected_at: Utc::now(),
                        detection_method: "rule_based".to_string(),
                        confidence: 0.9,
                        historical_occurrences: 1,
                        last_occurrence: None,
                    },
                },
                detection_details: ConflictDetectionDetails {
                    triggered_rules: vec!["filesystem_isolation_rule".to_string()],
                    detection_method: ConflictDetectionMethod::RuleBased,
                    analysis_duration: Duration::from_millis(4),
                    evidence: vec![ConflictEvidence {
                        evidence_type: ConflictEvidenceType::ResourceOverlap,
                        description: "Overlapping filesystem directory usage".to_string(),
                        strength: overlap.len() as f32 / dirs1.len().max(dirs2.len()) as f32,
                        data: [
                            (
                                "test1_dirs".to_string(),
                                format!("{:?}", usage1.temp_directories),
                            ),
                            (
                                "test2_dirs".to_string(),
                                format!("{:?}", usage2.temp_directories),
                            ),
                            ("overlapping_dirs".to_string(), format!("{:?}", overlap)),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    }],
                    false_positive_probability: 0.1,
                },
                resolution_options: self.generate_filesystem_resolution_options(usage1, usage2),
                impact_analysis: self.analyze_filesystem_conflict_impact(&overlap),
                detected_at: Utc::now(),
                confidence: 0.9,
            };
            return Ok(Some(conflict));
        }
        Ok(None)
    }
    /// Detect database resource conflicts
    fn detect_database_conflict(
        &self,
        usage1: &TestResourceUsage,
        usage2: &TestResourceUsage,
    ) -> AnalysisResult<Option<DetectedConflict>> {
        if usage1.database_connections > 0 && usage2.database_connections > 0 {
            let conflict = DetectedConflict {
                conflict_id: format!("database_conflict_{}_{}", usage1.test_id, usage2.test_id),
                conflict_info: ResourceConflict {
                    id: format!("database_{}_{}", usage1.test_id, usage2.test_id),
                    test1: usage1.test_id.clone(),
                    test2: usage2.test_id.clone(),
                    resource_type: "Database".to_string(),
                    resource_id: "database_shared".to_string(),
                    conflict_type: ConflictType::DataCorruption,
                    severity: ConflictSeverity::Medium,
                    description: "Potential database conflict: both tests use database connections"
                        .to_string(),
                    resolution_strategies: vec![],
                    metadata: ConflictMetadata {
                        detected_at: Utc::now(),
                        detection_method: "rule_based_heuristic".to_string(),
                        confidence: 0.7,
                        historical_occurrences: 1,
                        last_occurrence: None,
                    },
                },
                detection_details: ConflictDetectionDetails {
                    triggered_rules: vec!["database_isolation_rule".to_string()],
                    detection_method: ConflictDetectionMethod::RuleBased,
                    analysis_duration: Duration::from_millis(3),
                    evidence: vec![ConflictEvidence {
                        evidence_type: ConflictEvidenceType::ResourceOverlap,
                        description: "Both tests require database access".to_string(),
                        strength: 0.5,
                        data: [
                            (
                                "test1_db_connections".to_string(),
                                usage1.database_connections.to_string(),
                            ),
                            (
                                "test2_db_connections".to_string(),
                                usage2.database_connections.to_string(),
                            ),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    }],
                    false_positive_probability: 0.3,
                },
                resolution_options: self.generate_database_resolution_options(usage1, usage2),
                impact_analysis: self.analyze_database_conflict_impact(),
                detected_at: Utc::now(),
                confidence: 0.7,
            };
            return Ok(Some(conflict));
        }
        Ok(None)
    }
    /// Detect pattern-based conflicts between two tests.
    ///
    /// Two kinds of pattern signals are considered:
    ///
    /// 1. **Shared-tag overlap** – if both tests carry at least one identical tag, they
    ///    likely compete for the same logical resource or environment category.
    ///    Severity scales with the number of shared tags.
    ///
    /// 2. **Resource-keyword overlap** in test names – if both test names contain the
    ///    same keyword drawn from a known set of resource indicators ("database", "redis",
    ///    "postgres", "mongo", "kafka", "rabbitmq", "elasticsearch", "port", "socket",
    ///    "lock", "mutex", "queue", "cache", "s3", "minio", "grpc", "http", "smtp"),
    ///    the tests are likely contending for the same singleton or global resource.
    fn detect_pattern_conflicts(
        &self,
        test1: &TestParallelizationMetadata,
        test2: &TestParallelizationMetadata,
    ) -> AnalysisResult<Vec<DetectedConflict>> {
        const RESOURCE_KEYWORDS: &[&str] = &[
            "database",
            "redis",
            "postgres",
            "mysql",
            "mongo",
            "kafka",
            "rabbitmq",
            "elasticsearch",
            "port",
            "socket",
            "lock",
            "mutex",
            "queue",
            "cache",
            "s3",
            "minio",
            "grpc",
            "http",
            "smtp",
        ];

        let mut conflicts = Vec::new();

        let name1 = test1.base_context.test_name.to_lowercase();
        let name2 = test2.base_context.test_name.to_lowercase();

        // Check for shared resource-indicator keywords in test names.
        let shared_keywords: Vec<&str> = RESOURCE_KEYWORDS
            .iter()
            .copied()
            .filter(|kw| name1.contains(kw) && name2.contains(kw))
            .collect();

        if !shared_keywords.is_empty() {
            let severity = if shared_keywords.len() >= 3 {
                ConflictSeverity::High
            } else if shared_keywords.len() == 2 {
                ConflictSeverity::Medium
            } else {
                ConflictSeverity::Low
            };
            let keywords_str = shared_keywords.join(", ");
            conflicts.push(DetectedConflict {
                conflict_id: format!(
                    "pattern_name_{}_{}_{}",
                    test1.base_context.test_name,
                    test2.base_context.test_name,
                    shared_keywords[0],
                ),
                conflict_info: ResourceConflict {
                    id: format!(
                        "pattern_name_{}_{}",
                        test1.base_context.test_name,
                        test2.base_context.test_name
                    ),
                    test1: test1.base_context.test_name.clone(),
                    test2: test2.base_context.test_name.clone(),
                    resource_type: "PatternKeyword".to_string(),
                    resource_id: format!("keywords:[{}]", keywords_str),
                    conflict_type: ConflictType::ExclusiveAccess,
                    severity,
                    description: format!(
                        "Both tests contain resource-indicator keyword(s) [{}] in their names, \
                         suggesting shared singleton-resource usage",
                        keywords_str
                    ),
                    resolution_strategies: vec![],
                    metadata: ConflictMetadata {
                        detected_at: Utc::now(),
                        detection_method: "pattern_based".to_string(),
                        confidence: 0.65,
                        historical_occurrences: 1,
                        last_occurrence: None,
                    },
                },
                detection_details: ConflictDetectionDetails {
                    triggered_rules: vec!["name_keyword_pattern_rule".to_string()],
                    detection_method: ConflictDetectionMethod::PatternRecognition,
                    analysis_duration: Duration::from_millis(1),
                    evidence: vec![ConflictEvidence {
                        evidence_type: ConflictEvidenceType::ResourceOverlap,
                        description: format!(
                            "Test names share resource keywords: [{}]",
                            keywords_str
                        ),
                        strength: (shared_keywords.len() as f32 * 0.3).min(1.0),
                        data: [
                            (
                                "test1_name".to_string(),
                                test1.base_context.test_name.clone(),
                            ),
                            (
                                "test2_name".to_string(),
                                test2.base_context.test_name.clone(),
                            ),
                            ("shared_keywords".to_string(), keywords_str.clone()),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    }],
                    false_positive_probability: 0.35,
                },
                resolution_options: vec![],
                impact_analysis: ConflictImpactAnalysis {
                    performance_impact: PerformanceImpact {
                        cpu_degradation: 0.0,
                        memory_degradation: 0.0,
                        io_degradation: 0.1,
                        network_degradation: 0.1,
                        overall_degradation: 0.1,
                    },
                    reliability_impact: ReliabilityImpact {
                        error_rate_increase: 0.1,
                        timeout_probability_increase: 0.1,
                        flakiness_increase: 0.15,
                        reliability_decrease: 0.1,
                    },
                    efficiency_impact: EfficiencyImpact {
                        utilization_efficiency_loss: 0.1,
                        time_efficiency_loss: 0.1,
                        cost_efficiency_impact: 0.05,
                        overall_efficiency_loss: 0.1,
                    },
                    execution_time_impact: ExecutionTimeImpact {
                        individual_test_time_increase: 0.15,
                        total_suite_time_increase: 0.1,
                        parallelization_efficiency_loss: 0.2,
                    },
                    overall_impact_score: 0.1,
                },
                detected_at: Utc::now(),
                confidence: 0.65,
            });
        }

        // Check for shared tags (logical grouping overlap).
        let tags1: HashSet<&String> = test1.tags.iter().collect();
        let shared_tags: Vec<&String> = test2.tags.iter().filter(|t| tags1.contains(t)).collect();

        if !shared_tags.is_empty() {
            let severity = if shared_tags.len() >= 3 {
                ConflictSeverity::Medium
            } else {
                ConflictSeverity::Low
            };
            let tags_str: String = shared_tags
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            conflicts.push(DetectedConflict {
                conflict_id: format!(
                    "pattern_tag_{}_{}",
                    test1.base_context.test_name,
                    test2.base_context.test_name,
                ),
                conflict_info: ResourceConflict {
                    id: format!(
                        "pattern_tag_{}_{}",
                        test1.base_context.test_name,
                        test2.base_context.test_name
                    ),
                    test1: test1.base_context.test_name.clone(),
                    test2: test2.base_context.test_name.clone(),
                    resource_type: "TagOverlap".to_string(),
                    resource_id: format!("tags:[{}]", tags_str),
                    conflict_type: ConflictType::ExclusiveAccess,
                    severity,
                    description: format!(
                        "Both tests share tag(s) [{}], indicating overlap in resource \
                         category or environment usage",
                        tags_str
                    ),
                    resolution_strategies: vec![],
                    metadata: ConflictMetadata {
                        detected_at: Utc::now(),
                        detection_method: "pattern_based".to_string(),
                        confidence: 0.6,
                        historical_occurrences: 1,
                        last_occurrence: None,
                    },
                },
                detection_details: ConflictDetectionDetails {
                    triggered_rules: vec!["tag_overlap_pattern_rule".to_string()],
                    detection_method: ConflictDetectionMethod::PatternRecognition,
                    analysis_duration: Duration::from_millis(1),
                    evidence: vec![ConflictEvidence {
                        evidence_type: ConflictEvidenceType::ResourceOverlap,
                        description: format!("Tests share tags: [{}]", tags_str),
                        strength: (shared_tags.len() as f32 * 0.2).min(1.0),
                        data: [
                            ("test1_tags".to_string(), format!("{:?}", test1.tags)),
                            ("test2_tags".to_string(), format!("{:?}", test2.tags)),
                            ("shared_tags".to_string(), tags_str),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    }],
                    false_positive_probability: 0.4,
                },
                resolution_options: vec![],
                impact_analysis: ConflictImpactAnalysis {
                    performance_impact: PerformanceImpact {
                        cpu_degradation: 0.0,
                        memory_degradation: 0.0,
                        io_degradation: 0.05,
                        network_degradation: 0.05,
                        overall_degradation: 0.05,
                    },
                    reliability_impact: ReliabilityImpact {
                        error_rate_increase: 0.05,
                        timeout_probability_increase: 0.05,
                        flakiness_increase: 0.1,
                        reliability_decrease: 0.05,
                    },
                    efficiency_impact: EfficiencyImpact {
                        utilization_efficiency_loss: 0.05,
                        time_efficiency_loss: 0.05,
                        cost_efficiency_impact: 0.03,
                        overall_efficiency_loss: 0.05,
                    },
                    execution_time_impact: ExecutionTimeImpact {
                        individual_test_time_increase: 0.1,
                        total_suite_time_increase: 0.08,
                        parallelization_efficiency_loss: 0.15,
                    },
                    overall_impact_score: 0.07,
                },
                detected_at: Utc::now(),
                confidence: 0.6,
            });
        }

        Ok(conflicts)
    }
    /// Detect dependency-based conflicts between two tests.
    ///
    /// Three signals are checked:
    ///
    /// 1. **Direct dependency** – if test1 lists test2 as a dependency (or vice versa),
    ///    running them in parallel risks the dependent test observing partial state.
    ///
    /// 2. **Shared transitive dependency** – if both tests depend on a common third test,
    ///    they may contend for side-effects introduced by that shared prerequisite.
    ///
    /// 3. **Custom-isolation key overlap** – if both tests declare the same key in their
    ///    `isolation_requirements.custom_isolation` map, they require exclusive access to
    ///    the same named environment or resource class.
    fn detect_dependency_conflicts(
        &self,
        test1: &TestParallelizationMetadata,
        test2: &TestParallelizationMetadata,
    ) -> AnalysisResult<Vec<DetectedConflict>> {
        let mut conflicts = Vec::new();

        let name1 = &test1.base_context.test_name;
        let name2 = &test2.base_context.test_name;

        // Build sets of what each test depends on.
        let deps1: HashSet<&str> = test1
            .dependencies
            .iter()
            .map(|d| d.dependency_test.as_str())
            .collect();
        let deps2: HashSet<&str> = test2
            .dependencies
            .iter()
            .map(|d| d.dependency_test.as_str())
            .collect();

        // Signal 1: direct dependency (test1 -> test2 or test2 -> test1).
        let direct_fwd = deps1.contains(name2.as_str());
        let direct_rev = deps2.contains(name1.as_str());
        if direct_fwd || direct_rev {
            let (dependent, dependency) = if direct_fwd {
                (name1, name2)
            } else {
                (name2, name1)
            };
            conflicts.push(DetectedConflict {
                conflict_id: format!("dep_direct_{}_{}", name1, name2),
                conflict_info: ResourceConflict {
                    id: format!("dep_direct_{}_{}", name1, name2),
                    test1: name1.clone(),
                    test2: name2.clone(),
                    resource_type: "Dependency".to_string(),
                    resource_id: format!("direct_dep_{}_on_{}", dependent, dependency),
                    conflict_type: ConflictType::DataCorruption,
                    severity: ConflictSeverity::High,
                    description: format!(
                        "{} directly depends on {}; running them concurrently may \
                         observe intermediate state",
                        dependent, dependency
                    ),
                    resolution_strategies: vec![],
                    metadata: ConflictMetadata {
                        detected_at: Utc::now(),
                        detection_method: "dependency_analysis".to_string(),
                        confidence: 0.9,
                        historical_occurrences: 1,
                        last_occurrence: None,
                    },
                },
                detection_details: ConflictDetectionDetails {
                    triggered_rules: vec!["direct_dependency_rule".to_string()],
                    detection_method: ConflictDetectionMethod::RuleBased,
                    analysis_duration: Duration::from_millis(1),
                    evidence: vec![ConflictEvidence {
                        evidence_type: ConflictEvidenceType::ResourceOverlap,
                        description: format!(
                            "Direct test dependency: {} -> {}",
                            dependent, dependency
                        ),
                        strength: 0.9,
                        data: [
                            ("dependent".to_string(), dependent.clone()),
                            ("dependency".to_string(), dependency.clone()),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    }],
                    false_positive_probability: 0.05,
                },
                resolution_options: vec![],
                impact_analysis: ConflictImpactAnalysis {
                    performance_impact: PerformanceImpact {
                        cpu_degradation: 0.0,
                        memory_degradation: 0.0,
                        io_degradation: 0.2,
                        network_degradation: 0.0,
                        overall_degradation: 0.2,
                    },
                    reliability_impact: ReliabilityImpact {
                        error_rate_increase: 0.4,
                        timeout_probability_increase: 0.2,
                        flakiness_increase: 0.5,
                        reliability_decrease: 0.4,
                    },
                    efficiency_impact: EfficiencyImpact {
                        utilization_efficiency_loss: 0.1,
                        time_efficiency_loss: 0.3,
                        cost_efficiency_impact: 0.1,
                        overall_efficiency_loss: 0.2,
                    },
                    execution_time_impact: ExecutionTimeImpact {
                        individual_test_time_increase: 0.5,
                        total_suite_time_increase: 0.3,
                        parallelization_efficiency_loss: 0.6,
                    },
                    overall_impact_score: 0.4,
                },
                detected_at: Utc::now(),
                confidence: 0.9,
            });
        }

        // Signal 2: shared transitive dependency.
        let shared_deps: Vec<&&str> = deps1.iter().filter(|d| deps2.contains(*d)).collect();
        if !shared_deps.is_empty() {
            let shared_str: String = shared_deps.iter().map(|d| **d).collect::<Vec<_>>().join(", ");
            conflicts.push(DetectedConflict {
                conflict_id: format!("dep_shared_{}_{}", name1, name2),
                conflict_info: ResourceConflict {
                    id: format!("dep_shared_{}_{}", name1, name2),
                    test1: name1.clone(),
                    test2: name2.clone(),
                    resource_type: "SharedDependency".to_string(),
                    resource_id: format!("shared_deps:[{}]", shared_str),
                    conflict_type: ConflictType::DataCorruption,
                    severity: ConflictSeverity::Medium,
                    description: format!(
                        "Both tests share transitive dependency/ies [{}]; \
                         concurrent execution may observe conflicting side-effects \
                         from that shared prerequisite",
                        shared_str
                    ),
                    resolution_strategies: vec![],
                    metadata: ConflictMetadata {
                        detected_at: Utc::now(),
                        detection_method: "dependency_analysis".to_string(),
                        confidence: 0.75,
                        historical_occurrences: 1,
                        last_occurrence: None,
                    },
                },
                detection_details: ConflictDetectionDetails {
                    triggered_rules: vec!["shared_dependency_rule".to_string()],
                    detection_method: ConflictDetectionMethod::RuleBased,
                    analysis_duration: Duration::from_millis(1),
                    evidence: vec![ConflictEvidence {
                        evidence_type: ConflictEvidenceType::ResourceOverlap,
                        description: format!("Shared dependencies: [{}]", shared_str),
                        strength: (shared_deps.len() as f32 * 0.3).min(1.0),
                        data: [
                            ("test1".to_string(), name1.clone()),
                            ("test2".to_string(), name2.clone()),
                            ("shared_dependencies".to_string(), shared_str),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    }],
                    false_positive_probability: 0.25,
                },
                resolution_options: vec![],
                impact_analysis: ConflictImpactAnalysis {
                    performance_impact: PerformanceImpact {
                        cpu_degradation: 0.0,
                        memory_degradation: 0.05,
                        io_degradation: 0.15,
                        network_degradation: 0.05,
                        overall_degradation: 0.1,
                    },
                    reliability_impact: ReliabilityImpact {
                        error_rate_increase: 0.2,
                        timeout_probability_increase: 0.15,
                        flakiness_increase: 0.3,
                        reliability_decrease: 0.2,
                    },
                    efficiency_impact: EfficiencyImpact {
                        utilization_efficiency_loss: 0.1,
                        time_efficiency_loss: 0.15,
                        cost_efficiency_impact: 0.05,
                        overall_efficiency_loss: 0.1,
                    },
                    execution_time_impact: ExecutionTimeImpact {
                        individual_test_time_increase: 0.2,
                        total_suite_time_increase: 0.15,
                        parallelization_efficiency_loss: 0.3,
                    },
                    overall_impact_score: 0.2,
                },
                detected_at: Utc::now(),
                confidence: 0.75,
            });
        }

        // Signal 3: custom isolation key overlap.
        let iso1 = &test1.isolation_requirements.custom_isolation;
        let iso2 = &test2.isolation_requirements.custom_isolation;
        let shared_iso_keys: Vec<&String> =
            iso1.keys().filter(|k| iso2.contains_key(*k)).collect();

        if !shared_iso_keys.is_empty() {
            let keys_str: String = shared_iso_keys
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            conflicts.push(DetectedConflict {
                conflict_id: format!("dep_iso_{}_{}", name1, name2),
                conflict_info: ResourceConflict {
                    id: format!("dep_iso_{}_{}", name1, name2),
                    test1: name1.clone(),
                    test2: name2.clone(),
                    resource_type: "CustomIsolation".to_string(),
                    resource_id: format!("isolation_keys:[{}]", keys_str),
                    conflict_type: ConflictType::ExclusiveAccess,
                    severity: ConflictSeverity::Medium,
                    description: format!(
                        "Both tests declare the same custom-isolation key(s) [{}], \
                         requiring exclusive access to the same environment resource",
                        keys_str
                    ),
                    resolution_strategies: vec![],
                    metadata: ConflictMetadata {
                        detected_at: Utc::now(),
                        detection_method: "isolation_analysis".to_string(),
                        confidence: 0.8,
                        historical_occurrences: 1,
                        last_occurrence: None,
                    },
                },
                detection_details: ConflictDetectionDetails {
                    triggered_rules: vec!["custom_isolation_key_rule".to_string()],
                    detection_method: ConflictDetectionMethod::RuleBased,
                    analysis_duration: Duration::from_millis(1),
                    evidence: vec![ConflictEvidence {
                        evidence_type: ConflictEvidenceType::ResourceOverlap,
                        description: format!("Shared custom-isolation keys: [{}]", keys_str),
                        strength: (shared_iso_keys.len() as f32 * 0.4).min(1.0),
                        data: [
                            (
                                "test1_iso_keys".to_string(),
                                format!("{:?}", iso1.keys().collect::<Vec<_>>()),
                            ),
                            (
                                "test2_iso_keys".to_string(),
                                format!("{:?}", iso2.keys().collect::<Vec<_>>()),
                            ),
                            ("shared_keys".to_string(), keys_str),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    }],
                    false_positive_probability: 0.2,
                },
                resolution_options: vec![],
                impact_analysis: ConflictImpactAnalysis {
                    performance_impact: PerformanceImpact {
                        cpu_degradation: 0.0,
                        memory_degradation: 0.0,
                        io_degradation: 0.1,
                        network_degradation: 0.05,
                        overall_degradation: 0.1,
                    },
                    reliability_impact: ReliabilityImpact {
                        error_rate_increase: 0.15,
                        timeout_probability_increase: 0.1,
                        flakiness_increase: 0.2,
                        reliability_decrease: 0.15,
                    },
                    efficiency_impact: EfficiencyImpact {
                        utilization_efficiency_loss: 0.1,
                        time_efficiency_loss: 0.1,
                        cost_efficiency_impact: 0.05,
                        overall_efficiency_loss: 0.1,
                    },
                    execution_time_impact: ExecutionTimeImpact {
                        individual_test_time_increase: 0.2,
                        total_suite_time_increase: 0.15,
                        parallelization_efficiency_loss: 0.25,
                    },
                    overall_impact_score: 0.15,
                },
                detected_at: Utc::now(),
                confidence: 0.8,
            });
        }

        Ok(conflicts)
    }
    /// Detect ML-based conflicts between two tests.
    ///
    /// ML-based conflict detection requires a trained classifier — returns no conflicts
    /// until model inference is wired up (tracked_external: trustformers ML integration).
    fn detect_ml_conflicts(
        &self,
        _test1: &TestParallelizationMetadata,
        _test2: &TestParallelizationMetadata,
    ) -> AnalysisResult<Vec<DetectedConflict>> {
        Ok(vec![])
    }
    /// Helper methods for conflict severity calculation
    fn calculate_cpu_conflict_severity(
        &self,
        combined_usage: f32,
        threshold: f32,
    ) -> ConflictSeverity {
        let excess_ratio = (combined_usage - threshold) / threshold;
        match excess_ratio {
            r if r > 1.0 => ConflictSeverity::Critical,
            r if r > 0.5 => ConflictSeverity::High,
            r if r > 0.2 => ConflictSeverity::Medium,
            _ => ConflictSeverity::Low,
        }
    }
    fn calculate_memory_conflict_severity(
        &self,
        combined_usage: f32,
        threshold: f32,
    ) -> ConflictSeverity {
        let excess_ratio = (combined_usage - threshold) / threshold;
        match excess_ratio {
            r if r > 1.0 => ConflictSeverity::Critical,
            r if r > 0.5 => ConflictSeverity::High,
            r if r > 0.2 => ConflictSeverity::Medium,
            _ => ConflictSeverity::Low,
        }
    }
    /// Generate resolution options for different conflict types
    fn generate_cpu_resolution_options(
        &self,
        _usage1: &TestResourceUsage,
        _usage2: &TestResourceUsage,
    ) -> Vec<ConflictResolutionOption> {
        vec![]
    }
    fn generate_memory_resolution_options(
        &self,
        _usage1: &TestResourceUsage,
        _usage2: &TestResourceUsage,
    ) -> Vec<ConflictResolutionOption> {
        vec![]
    }
    fn generate_gpu_resolution_options(
        &self,
        _usage1: &TestResourceUsage,
        _usage2: &TestResourceUsage,
    ) -> Vec<ConflictResolutionOption> {
        vec![]
    }
    fn generate_network_resolution_options(
        &self,
        _usage1: &TestResourceUsage,
        _usage2: &TestResourceUsage,
    ) -> Vec<ConflictResolutionOption> {
        vec![]
    }
    fn generate_filesystem_resolution_options(
        &self,
        _usage1: &TestResourceUsage,
        _usage2: &TestResourceUsage,
    ) -> Vec<ConflictResolutionOption> {
        vec![]
    }
    fn generate_database_resolution_options(
        &self,
        _usage1: &TestResourceUsage,
        _usage2: &TestResourceUsage,
    ) -> Vec<ConflictResolutionOption> {
        vec![]
    }
    /// Impact analysis methods
    fn analyze_cpu_conflict_impact(
        &self,
        _combined_usage: f32,
        _threshold: f32,
    ) -> ConflictImpactAnalysis {
        ConflictImpactAnalysis {
            performance_impact: PerformanceImpact {
                cpu_degradation: 0.2,
                memory_degradation: 0.0,
                io_degradation: 0.0,
                network_degradation: 0.0,
                overall_degradation: 0.2,
            },
            reliability_impact: ReliabilityImpact {
                error_rate_increase: 0.1,
                timeout_probability_increase: 0.15,
                flakiness_increase: 0.1,
                reliability_decrease: 0.1,
            },
            efficiency_impact: EfficiencyImpact {
                utilization_efficiency_loss: 0.25,
                time_efficiency_loss: 0.2,
                cost_efficiency_impact: 0.15,
                overall_efficiency_loss: 0.2,
            },
            execution_time_impact: ExecutionTimeImpact {
                individual_test_time_increase: 0.3,
                total_suite_time_increase: 0.25,
                parallelization_efficiency_loss: 0.4,
            },
            overall_impact_score: 0.25,
        }
    }
    fn analyze_memory_conflict_impact(
        &self,
        _combined_usage: f32,
        _threshold: f32,
    ) -> ConflictImpactAnalysis {
        ConflictImpactAnalysis {
            performance_impact: PerformanceImpact {
                cpu_degradation: 0.1,
                memory_degradation: 0.3,
                io_degradation: 0.2,
                network_degradation: 0.0,
                overall_degradation: 0.25,
            },
            reliability_impact: ReliabilityImpact {
                error_rate_increase: 0.15,
                timeout_probability_increase: 0.2,
                flakiness_increase: 0.15,
                reliability_decrease: 0.15,
            },
            efficiency_impact: EfficiencyImpact {
                utilization_efficiency_loss: 0.3,
                time_efficiency_loss: 0.25,
                cost_efficiency_impact: 0.2,
                overall_efficiency_loss: 0.25,
            },
            execution_time_impact: ExecutionTimeImpact {
                individual_test_time_increase: 0.35,
                total_suite_time_increase: 0.3,
                parallelization_efficiency_loss: 0.45,
            },
            overall_impact_score: 0.3,
        }
    }
    fn analyze_gpu_conflict_impact(&self, _overlap: &[&usize]) -> ConflictImpactAnalysis {
        ConflictImpactAnalysis {
            performance_impact: PerformanceImpact {
                cpu_degradation: 0.0,
                memory_degradation: 0.1,
                io_degradation: 0.0,
                network_degradation: 0.0,
                overall_degradation: 0.5,
            },
            reliability_impact: ReliabilityImpact {
                error_rate_increase: 0.3,
                timeout_probability_increase: 0.4,
                flakiness_increase: 0.35,
                reliability_decrease: 0.3,
            },
            efficiency_impact: EfficiencyImpact {
                utilization_efficiency_loss: 0.6,
                time_efficiency_loss: 0.5,
                cost_efficiency_impact: 0.4,
                overall_efficiency_loss: 0.5,
            },
            execution_time_impact: ExecutionTimeImpact {
                individual_test_time_increase: 0.8,
                total_suite_time_increase: 0.7,
                parallelization_efficiency_loss: 0.9,
            },
            overall_impact_score: 0.6,
        }
    }
    fn analyze_network_conflict_impact(&self, _overlap: &[&u16]) -> ConflictImpactAnalysis {
        ConflictImpactAnalysis {
            performance_impact: PerformanceImpact {
                cpu_degradation: 0.0,
                memory_degradation: 0.0,
                io_degradation: 0.0,
                network_degradation: 0.4,
                overall_degradation: 0.3,
            },
            reliability_impact: ReliabilityImpact {
                error_rate_increase: 0.25,
                timeout_probability_increase: 0.3,
                flakiness_increase: 0.2,
                reliability_decrease: 0.25,
            },
            efficiency_impact: EfficiencyImpact {
                utilization_efficiency_loss: 0.2,
                time_efficiency_loss: 0.3,
                cost_efficiency_impact: 0.15,
                overall_efficiency_loss: 0.2,
            },
            execution_time_impact: ExecutionTimeImpact {
                individual_test_time_increase: 0.4,
                total_suite_time_increase: 0.35,
                parallelization_efficiency_loss: 0.5,
            },
            overall_impact_score: 0.3,
        }
    }
    fn analyze_filesystem_conflict_impact(&self, _overlap: &[&String]) -> ConflictImpactAnalysis {
        ConflictImpactAnalysis {
            performance_impact: PerformanceImpact {
                cpu_degradation: 0.0,
                memory_degradation: 0.0,
                io_degradation: 0.3,
                network_degradation: 0.0,
                overall_degradation: 0.25,
            },
            reliability_impact: ReliabilityImpact {
                error_rate_increase: 0.4,
                timeout_probability_increase: 0.2,
                flakiness_increase: 0.5,
                reliability_decrease: 0.4,
            },
            efficiency_impact: EfficiencyImpact {
                utilization_efficiency_loss: 0.15,
                time_efficiency_loss: 0.3,
                cost_efficiency_impact: 0.2,
                overall_efficiency_loss: 0.2,
            },
            execution_time_impact: ExecutionTimeImpact {
                individual_test_time_increase: 0.3,
                total_suite_time_increase: 0.25,
                parallelization_efficiency_loss: 0.6,
            },
            overall_impact_score: 0.35,
        }
    }
    fn analyze_database_conflict_impact(&self) -> ConflictImpactAnalysis {
        ConflictImpactAnalysis {
            performance_impact: PerformanceImpact {
                cpu_degradation: 0.1,
                memory_degradation: 0.1,
                io_degradation: 0.2,
                network_degradation: 0.1,
                overall_degradation: 0.2,
            },
            reliability_impact: ReliabilityImpact {
                error_rate_increase: 0.3,
                timeout_probability_increase: 0.25,
                flakiness_increase: 0.4,
                reliability_decrease: 0.3,
            },
            efficiency_impact: EfficiencyImpact {
                utilization_efficiency_loss: 0.2,
                time_efficiency_loss: 0.25,
                cost_efficiency_impact: 0.15,
                overall_efficiency_loss: 0.2,
            },
            execution_time_impact: ExecutionTimeImpact {
                individual_test_time_increase: 0.35,
                total_suite_time_increase: 0.3,
                parallelization_efficiency_loss: 0.5,
            },
            overall_impact_score: 0.3,
        }
    }
    /// Remove duplicate conflicts
    fn deduplicate_conflicts(&self, mut conflicts: Vec<DetectedConflict>) -> Vec<DetectedConflict> {
        conflicts.sort_by(|a, b| a.conflict_id.cmp(&b.conflict_id));
        conflicts.dedup_by(|a, b| a.conflict_id == b.conflict_id);
        conflicts
    }
    /// Update detection statistics
    fn update_detection_statistics(
        &self,
        conflicts: &[DetectedConflict],
        analysis_duration: Duration,
    ) {
        let mut stats = self.statistics.write();
        stats.total_conflicts_detected += conflicts.len() as u64;
        for conflict in conflicts {
            *stats
                .conflicts_by_type
                .entry(conflict.conflict_info.conflict_type.clone())
                .or_insert(0) += 1;
            *stats
                .conflicts_by_severity
                .entry(conflict.conflict_info.severity.clone())
                .or_insert(0) += 1;
        }
        if analysis_duration > stats.performance_metrics.max_detection_time {
            stats.performance_metrics.max_detection_time = analysis_duration;
        }
        let current_avg = stats.performance_metrics.average_detection_time;
        let total_detections = stats.total_conflicts_detected.max(1);
        stats.performance_metrics.average_detection_time = Duration::from_nanos(
            ((current_avg.as_nanos() * (total_detections - 1) as u128
                + analysis_duration.as_nanos())
                / total_detections as u128) as u64,
        );
    }
    /// Create default detection rules
    fn create_default_rules() -> Vec<ConflictDetectionRule> {
        vec![
            ConflictDetectionRule {
                rule_id: "cpu_capacity_rule".to_string(),
                name: "CPU Capacity Conflict".to_string(),
                description: "Detects conflicts when combined CPU usage exceeds threshold"
                    .to_string(),
                category: ConflictRuleCategory::ResourceBased,
                pattern: ConflictPattern::Custom {
                    pattern_name: "cpu_capacity".to_string(),
                    pattern_data: HashMap::new(),
                },
                action: ConflictDetectionAction::Block,
                confidence: 0.9,
                priority: 100,
                enabled: true,
                conditions: vec![],
            },
            ConflictDetectionRule {
                rule_id: "memory_capacity_rule".to_string(),
                name: "Memory Capacity Conflict".to_string(),
                description: "Detects conflicts when combined memory usage exceeds threshold"
                    .to_string(),
                category: ConflictRuleCategory::ResourceBased,
                pattern: ConflictPattern::Custom {
                    pattern_name: "memory_capacity".to_string(),
                    pattern_data: HashMap::new(),
                },
                action: ConflictDetectionAction::Block,
                confidence: 0.85,
                priority: 95,
                enabled: true,
                conditions: vec![],
            },
            ConflictDetectionRule {
                rule_id: "gpu_exclusive_access_rule".to_string(),
                name: "GPU Exclusive Access Conflict".to_string(),
                description: "Detects conflicts when tests require the same GPU devices"
                    .to_string(),
                category: ConflictRuleCategory::ResourceBased,
                pattern: ConflictPattern::GpuConflict {
                    gpu_ids: vec![],
                    exclusive_access: true,
                },
                action: ConflictDetectionAction::Queue,
                confidence: 0.95,
                priority: 110,
                enabled: true,
                conditions: vec![],
            },
            ConflictDetectionRule {
                rule_id: "network_port_rule".to_string(),
                name: "Network Port Conflict".to_string(),
                description: "Detects conflicts when tests use the same network ports".to_string(),
                category: ConflictRuleCategory::ResourceBased,
                pattern: ConflictPattern::PortRangeOverlap {
                    min_overlap: 1,
                    max_overlap: 65535,
                },
                action: ConflictDetectionAction::AutoResolve,
                confidence: 0.9,
                priority: 80,
                enabled: true,
                conditions: vec![],
            },
            ConflictDetectionRule {
                rule_id: "filesystem_isolation_rule".to_string(),
                name: "Filesystem Isolation Conflict".to_string(),
                description: "Detects conflicts when tests use overlapping file paths".to_string(),
                category: ConflictRuleCategory::ResourceBased,
                pattern: ConflictPattern::FilePathOverlap {
                    path_pattern: "*".to_string(),
                    case_sensitive: true,
                },
                action: ConflictDetectionAction::Block,
                confidence: 0.9,
                priority: 105,
                enabled: true,
                conditions: vec![],
            },
            ConflictDetectionRule {
                rule_id: "database_isolation_rule".to_string(),
                name: "Database Isolation Conflict".to_string(),
                description: "Detects potential database conflicts between tests".to_string(),
                category: ConflictRuleCategory::ResourceBased,
                pattern: ConflictPattern::DatabaseConflict {
                    database_type: "any".to_string(),
                    conflict_scope: DatabaseConflictScope::Database,
                },
                action: ConflictDetectionAction::Warn,
                confidence: 0.7,
                priority: 60,
                enabled: true,
                conditions: vec![],
            },
        ]
    }
    /// Create default resolution strategies
    fn create_default_strategies() -> HashMap<ConflictType, Vec<ConflictResolutionStrategy>> {
        let mut strategies = HashMap::new();
        strategies.insert(
            ConflictType::CapacityLimit,
            vec![
                ConflictResolutionStrategy {
                    strategy_id: "sequential_execution".to_string(),
                    name: "Sequential Execution".to_string(),
                    description: "Run conflicting tests sequentially to avoid resource contention"
                        .to_string(),
                    strategy_type: ConflictResolutionType::Sequential,
                    applicability: vec![],
                    expected_outcomes: vec![],
                    resource_requirements: StrategyResourceRequirements {
                        cpu_overhead: 0.0,
                        memory_overhead: 0.0,
                        network_overhead: 0.0,
                        time_overhead: Duration::from_secs(0),
                        custom_overheads: HashMap::new(),
                    },
                },
                ConflictResolutionStrategy {
                    strategy_id: "resource_provisioning".to_string(),
                    name: "Additional Resource Provisioning".to_string(),
                    description: "Provision additional resources to accommodate both tests"
                        .to_string(),
                    strategy_type: ConflictResolutionType::ResourceProvisioning,
                    applicability: vec![],
                    expected_outcomes: vec![],
                    resource_requirements: StrategyResourceRequirements {
                        cpu_overhead: 0.5,
                        memory_overhead: 0.5,
                        network_overhead: 0.0,
                        time_overhead: Duration::from_secs(30),
                        custom_overheads: HashMap::new(),
                    },
                },
            ],
        );
        strategies.insert(
            ConflictType::ExclusiveAccess,
            vec![
                ConflictResolutionStrategy {
                    strategy_id: "resource_isolation".to_string(),
                    name: "Resource Isolation".to_string(),
                    description: "Isolate tests using separate resource instances".to_string(),
                    strategy_type: ConflictResolutionType::Isolation,
                    applicability: vec![],
                    expected_outcomes: vec![],
                    resource_requirements: StrategyResourceRequirements {
                        cpu_overhead: 0.1,
                        memory_overhead: 0.1,
                        network_overhead: 0.0,
                        time_overhead: Duration::from_secs(10),
                        custom_overheads: HashMap::new(),
                    },
                },
                ConflictResolutionStrategy {
                    strategy_id: "queued_execution".to_string(),
                    name: "Queued Execution".to_string(),
                    description: "Queue tests for exclusive resource access".to_string(),
                    strategy_type: ConflictResolutionType::Sequential,
                    applicability: vec![],
                    expected_outcomes: vec![],
                    resource_requirements: StrategyResourceRequirements {
                        cpu_overhead: 0.0,
                        memory_overhead: 0.0,
                        network_overhead: 0.0,
                        time_overhead: Duration::from_secs(0),
                        custom_overheads: HashMap::new(),
                    },
                },
            ],
        );
        strategies
    }
    /// Get conflict detection statistics
    pub fn get_statistics(&self) -> ConflictDetectionStatistics {
        (*self.statistics.read()).clone()
    }
}
