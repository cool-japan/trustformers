//! Resource Conflict Detector
//!
//! Detects potential conflicts between concurrent test executions and provides
//! sophisticated resolution strategies and mitigation techniques.

use super::super::types::*;
use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, sync::Arc, time::Instant};

pub struct ResourceConflictDetector {
    /// Conflict detection algorithms
    algorithms: Arc<Mutex<Vec<Box<dyn ConflictDetectionAlgorithm + Send + Sync>>>>,

    /// Conflict resolution strategies
    resolution_strategies: Arc<Mutex<Vec<Box<dyn ConflictResolutionStrategy + Send + Sync>>>>,

    /// Resource dependency graph
    dependency_graph: Arc<RwLock<ResourceDependencyGraph>>,

    /// Configuration
    config: ConflictDetectionConfig,
}

impl ResourceConflictDetector {
    /// Creates a new resource conflict detector
    pub async fn new(config: ConflictDetectionConfig) -> Result<Self> {
        let mut algorithms: Vec<Box<dyn ConflictDetectionAlgorithm + Send + Sync>> = Vec::new();
        let mut strategies: Vec<Box<dyn ConflictResolutionStrategy + Send + Sync>> = Vec::new();

        // Initialize detection algorithms.
        //
        // `MLConflictDetectionAlgorithm` used to be pushed here as a fourth
        // entry; it was deleted in 0.2.1 because it named a model that does not
        // exist in this crate and could only answer by inventing. The three
        // that remain each derive their answer from the access patterns they
        // are handed -- see `types::core::conflict_algorithms`.
        algorithms.push(Box::new(StaticConflictDetectionAlgorithm::new(true, 0.8)));
        algorithms.push(Box::new(DynamicConflictDetectionAlgorithm::new(true, 0.1)));
        algorithms.push(Box::new(PredictiveConflictDetectionAlgorithm::new(5, 0.8)));

        // Initialize resolution strategies
        strategies.push(Box::new(AvoidanceResolutionStrategy::new(true, false)));
        strategies.push(Box::new(TimeoutResolutionStrategy::new(1000, true)));
        strategies.push(Box::new(PartitioningResolutionStrategy::new()?));
        strategies.push(Box::new(AdaptiveResolutionStrategy::new()?));

        Ok(Self {
            algorithms: Arc::new(Mutex::new(algorithms)),
            resolution_strategies: Arc::new(Mutex::new(strategies)),
            dependency_graph: Arc::new(RwLock::new(ResourceDependencyGraph::new()?)),
            config,
        })
    }

    /// Detects resource conflicts in test execution data
    pub async fn detect_conflicts(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<ConflictAnalysisResult> {
        let start_time = Utc::now();

        if !self.config.detection_enabled {
            return Ok(ConflictAnalysisResult {
                conflicts: Vec::new(),
                resource_conflicts: Vec::new(),
                resolutions: Vec::new(),
                resource_constraints: HashMap::new(),
                resource_limits: HashMap::new(),
                isolation_requirements: IsolationRequirements {
                    process_isolation: false,
                    thread_isolation: false,
                    memory_isolation: false,
                    network_isolation: false,
                    filesystem_isolation: false,
                    custom_isolation: HashMap::new(),
                },
                detection_results: Vec::new(),
                analysis_duration: std::time::Duration::from_secs(0),
                confidence: 1.0,
            });
        }

        // Update dependency graph
        self.update_dependency_graph(test_data).await?;

        // Run all detection algorithms
        // Extract algorithm results synchronously to avoid lifetime issues
        let detection_results: Vec<_> = {
            let algorithms = self.algorithms.lock();
            algorithms
                .iter()
                .map(|algorithm| {
                    let algorithm_name = algorithm.name().to_string();
                    let detection_start = Instant::now();
                    let result = algorithm.detect_conflicts(&test_data.resource_access_patterns);
                    let detection_duration = detection_start.elapsed();
                    (algorithm_name, result, detection_duration)
                })
                .collect()
        };

        // Collect and merge results
        let mut all_conflicts = Vec::new();
        let mut algorithm_results = Vec::new();

        for (algorithm_name, result, duration) in detection_results {
            match result {
                Ok(mut conflicts) => {
                    algorithm_results.push(ConflictDetectionResult {
                        algorithm: algorithm_name,
                        conflicts: conflicts.clone(),
                        duration,
                        confidence: self.calculate_detection_confidence(&conflicts),
                    });
                    all_conflicts.append(&mut conflicts);
                },
                Err(e) => {
                    log::warn!("Conflict detection algorithm failed: {}", e);
                },
            }
        }

        // Filter conflicts below sensitivity threshold
        let sensitive_conflicts: Vec<ResourceConflict> = all_conflicts
            .into_iter()
            .filter(|c| c.probability >= self.config.sensitivity)
            .collect();

        // Deduplicate and prioritize conflicts; cap depth to max_depth
        let unique_conflicts = self.deduplicate_conflicts(&sensitive_conflicts);
        let prioritized_conflicts: Vec<ResourceConflict> = self
            .prioritize_conflicts(&unique_conflicts)
            .into_iter()
            .take(self.config.max_depth)
            .collect();

        // Generate resolution strategies
        let resolutions = self.generate_resolutions(&prioritized_conflicts).await?;

        // Calculate resource constraints
        let resource_constraints_vec = self.calculate_resource_constraints(&prioritized_conflicts);
        let resource_constraints: HashMap<String, f64> = resource_constraints_vec
            .iter()
            .map(|c| (c.resource_type.clone(), c.max_value))
            .collect();
        let resource_limits_f64 = self.calculate_resource_limits(&prioritized_conflicts);
        // resource_limits_f64 values are severity-based fractions in 0.05..=0.9;
        // the minimum concurrent resource limit is 1 for each resource.
        let resource_limits: HashMap<String, usize> =
            resource_limits_f64.keys().map(|k| (k.clone(), 1_usize)).collect();
        let isolation_requirements = self.generate_isolation_requirements(&prioritized_conflicts);

        let elapsed = Utc::now().signed_duration_since(start_time).to_std().unwrap_or_default();
        if elapsed > self.config.timeout {
            log::warn!(
                "Conflict detection exceeded configured timeout ({:?} > {:?})",
                elapsed,
                self.config.timeout
            );
        }

        Ok(ConflictAnalysisResult {
            conflicts: prioritized_conflicts.clone(),
            resource_conflicts: prioritized_conflicts,
            resolutions,
            resource_constraints,
            resource_limits,
            isolation_requirements,
            detection_results: algorithm_results,
            analysis_duration: elapsed,
            confidence: self.calculate_overall_conflict_confidence(&unique_conflicts),
        })
    }

    /// Updates the resource dependency graph
    async fn update_dependency_graph(&self, test_data: &TestExecutionData) -> Result<()> {
        let mut graph = self.dependency_graph.write();

        for pattern in &test_data.resource_access_patterns {
            graph.add_resource(pattern.resource_id.clone());

            for trace in &test_data.execution_traces {
                if trace.resource == pattern.resource_id {
                    // Convert Instant timestamp to f64 seconds (elapsed time as weight)
                    let timestamp_secs = trace.timestamp.elapsed().as_secs_f64();
                    graph.add_dependency(
                        pattern.resource_id.clone(),
                        trace.operation.clone(),
                        timestamp_secs,
                    );
                }
            }
        }

        Ok(())
    }

    /// Calculates detection confidence based on conflict characteristics
    fn calculate_detection_confidence(&self, conflicts: &[ResourceConflict]) -> f64 {
        if conflicts.is_empty() {
            return 1.0;
        }

        let avg_probability =
            conflicts.iter().map(|c| c.probability).sum::<f64>() / conflicts.len() as f64;
        let severity_factor = conflicts
            .iter()
            .map(|c| match c.severity {
                ConflictSeverity::Fatal => 1.0_f64,
                ConflictSeverity::Blocking => 0.95,
                ConflictSeverity::Critical => 0.9,
                ConflictSeverity::Severe => 0.8,
                ConflictSeverity::Major | ConflictSeverity::High => 0.6,
                ConflictSeverity::Moderate | ConflictSeverity::Medium => 0.4,
                ConflictSeverity::Minor | ConflictSeverity::Low => 0.2,
            })
            .sum::<f64>()
            / conflicts.len() as f64;

        (avg_probability + severity_factor) / 2.0
    }

    /// Deduplicates conflicts based on resource overlap and similarity
    fn deduplicate_conflicts(&self, conflicts: &[ResourceConflict]) -> Vec<ResourceConflict> {
        let mut unique_conflicts = Vec::new();

        for conflict in conflicts {
            let is_duplicate = unique_conflicts
                .iter()
                .any(|existing: &ResourceConflict| self.conflicts_are_similar(existing, conflict));

            if !is_duplicate {
                unique_conflicts.push(conflict.clone());
            }
        }

        unique_conflicts
    }

    /// Checks if two conflicts are similar enough to be considered duplicates
    fn conflicts_are_similar(&self, a: &ResourceConflict, b: &ResourceConflict) -> bool {
        // Check if conflicts involve the same resource
        let resource_overlap = a.resource_id == b.resource_id;

        // Check if conflict types are compatible
        let type_similarity = match (&a.conflict_type, &b.conflict_type) {
            (ConflictType::Data, ConflictType::Data) => true,
            (ConflictType::Lock, ConflictType::Lock) => true,
            (ConflictType::ResourceAccess, ConflictType::ResourceAccess) => true,
            _ => false,
        };

        resource_overlap && type_similarity
    }

    /// Prioritizes conflicts based on severity and impact
    fn prioritize_conflicts(&self, conflicts: &[ResourceConflict]) -> Vec<ResourceConflict> {
        let mut prioritized = conflicts.to_vec();

        prioritized.sort_by(|a, b| {
            // First by severity
            let severity_cmp = match (&a.severity, &b.severity) {
                (s1, s2) if std::mem::discriminant(s1) == std::mem::discriminant(s2) => {
                    std::cmp::Ordering::Equal
                },
                (ConflictSeverity::Fatal, _) => std::cmp::Ordering::Less,
                (_, ConflictSeverity::Fatal) => std::cmp::Ordering::Greater,
                (ConflictSeverity::Blocking, _) => std::cmp::Ordering::Less,
                (_, ConflictSeverity::Blocking) => std::cmp::Ordering::Greater,
                (ConflictSeverity::Critical, _) => std::cmp::Ordering::Less,
                (_, ConflictSeverity::Critical) => std::cmp::Ordering::Greater,
                (ConflictSeverity::Severe, _) => std::cmp::Ordering::Less,
                (_, ConflictSeverity::Severe) => std::cmp::Ordering::Greater,
                (ConflictSeverity::Major | ConflictSeverity::High, _) => std::cmp::Ordering::Less,
                (_, ConflictSeverity::Major | ConflictSeverity::High) => {
                    std::cmp::Ordering::Greater
                },
                (ConflictSeverity::Moderate | ConflictSeverity::Medium, _) => {
                    std::cmp::Ordering::Less
                },
                (_, ConflictSeverity::Moderate | ConflictSeverity::Medium) => {
                    std::cmp::Ordering::Greater
                },
                (
                    ConflictSeverity::Minor | ConflictSeverity::Low,
                    ConflictSeverity::Minor | ConflictSeverity::Low,
                ) => std::cmp::Ordering::Equal,
            };

            if severity_cmp != std::cmp::Ordering::Equal {
                return severity_cmp;
            }

            // Then by probability (higher probability first)
            b.probability.partial_cmp(&a.probability).unwrap_or(std::cmp::Ordering::Equal)
        });

        prioritized
    }

    /// Generates resolution strategies for conflicts
    async fn generate_resolutions(
        &self,
        conflicts: &[ResourceConflict],
    ) -> Result<Vec<ConflictResolution>> {
        let strategies = self.resolution_strategies.lock();
        let mut resolutions = Vec::new();

        for conflict in conflicts {
            for strategy in strategies.iter() {
                if strategy.is_applicable(conflict) {
                    if let Ok(resolution) = strategy.resolve_conflict(conflict) {
                        resolutions.push(resolution);
                    }
                }
            }
        }

        Ok(resolutions)
    }

    /// Calculates resource constraints based on conflicts
    fn calculate_resource_constraints(
        &self,
        conflicts: &[ResourceConflict],
    ) -> Vec<ResourceConstraint> {
        let mut constraints = Vec::new();

        for conflict in conflicts {
            let max_concurrent = match conflict.severity {
                ConflictSeverity::Fatal => 1.0,
                ConflictSeverity::Blocking => 1.0,
                ConflictSeverity::Critical => 2.0,
                ConflictSeverity::Severe => 3.0,
                ConflictSeverity::Major | ConflictSeverity::High => 4.0,
                ConflictSeverity::Moderate | ConflictSeverity::Medium => 6.0,
                ConflictSeverity::Minor | ConflictSeverity::Low => 8.0,
            };

            let constraint = ResourceConstraint {
                constraint_id: format!("constraint_{}", &conflict.resource_id),
                resource_type: conflict.resource_id.clone(),
                min_value: 0.0,
                max_value: max_concurrent,
                constraint_type: match conflict.conflict_type {
                    ConflictType::Data => "ExclusiveAccess".to_string(),
                    ConflictType::Lock => "LimitedConcurrency".to_string(),
                    ConflictType::ResourceAccess => "ResourceQuota".to_string(),
                    ConflictType::Memory => "ResourceQuota".to_string(),
                    ConflictType::Io => "LimitedConcurrency".to_string(),
                    ConflictType::Network => "LimitedConcurrency".to_string(),
                    ConflictType::Database => "OrderedAccess".to_string(),
                    ConflictType::Process => "OrderedAccess".to_string(),
                    ConflictType::Timing => "TemporalConstraint".to_string(),
                    ConflictType::Configuration => "ExclusiveAccess".to_string(),
                    ConflictType::ReadWrite => "LimitedConcurrency".to_string(),
                },
                enforcement_level: "Required".to_string(),
            };

            constraints.push(constraint);
        }

        constraints
    }

    /// Calculates resource limits based on conflicts
    fn calculate_resource_limits(&self, conflicts: &[ResourceConflict]) -> HashMap<String, f64> {
        let mut limits: HashMap<String, f64> = HashMap::new();

        for conflict in conflicts {
            let limit: f64 = match conflict.severity {
                ConflictSeverity::Fatal => 0.05,
                ConflictSeverity::Blocking => 0.08,
                ConflictSeverity::Critical => 0.1,
                ConflictSeverity::Severe => 0.2,
                ConflictSeverity::High => 0.3,
                ConflictSeverity::Major => 0.4,
                ConflictSeverity::Medium => 0.6,
                ConflictSeverity::Moderate => 0.7,
                ConflictSeverity::Low => 0.8,
                ConflictSeverity::Minor => 0.9,
            };

            limits
                .entry(conflict.resource_id.clone())
                .and_modify(|existing| *existing = existing.min(limit))
                .or_insert(limit);
        }

        limits
    }

    /// Generates isolation requirements
    fn generate_isolation_requirements(
        &self,
        conflicts: &[ResourceConflict],
    ) -> IsolationRequirements {
        let mut isolation_requirements = IsolationRequirements {
            process_isolation: false,
            thread_isolation: false,
            memory_isolation: false,
            network_isolation: false,
            filesystem_isolation: false,
            custom_isolation: HashMap::new(),
        };

        for conflict in conflicts {
            match conflict.severity {
                ConflictSeverity::Fatal => {
                    isolation_requirements.process_isolation = true;
                    isolation_requirements.memory_isolation = true;
                    isolation_requirements.network_isolation = true;
                    isolation_requirements.filesystem_isolation = true;
                },
                ConflictSeverity::Blocking => {
                    isolation_requirements.process_isolation = true;
                    isolation_requirements.memory_isolation = true;
                    isolation_requirements.network_isolation = true;
                },
                ConflictSeverity::Critical => {
                    isolation_requirements.process_isolation = true;
                    isolation_requirements.memory_isolation = true;
                },
                ConflictSeverity::Severe => {
                    isolation_requirements.thread_isolation = true;
                    isolation_requirements.memory_isolation = true;
                },
                ConflictSeverity::Major | ConflictSeverity::High => {
                    isolation_requirements.thread_isolation = true;
                    isolation_requirements.memory_isolation = true;
                },
                ConflictSeverity::Moderate | ConflictSeverity::Medium => {
                    isolation_requirements.thread_isolation = true;
                },
                ConflictSeverity::Minor | ConflictSeverity::Low => {
                    // No additional isolation required
                },
            }
        }

        isolation_requirements
    }

    /// Calculates overall conflict confidence
    fn calculate_overall_conflict_confidence(&self, conflicts: &[ResourceConflict]) -> f64 {
        if conflicts.is_empty() {
            return 1.0;
        }

        let avg_probability =
            conflicts.iter().map(|c| c.probability).sum::<f64>() / conflicts.len() as f64;
        let consistency_factor = self.calculate_conflict_consistency(conflicts);

        avg_probability * consistency_factor
    }

    /// Calculates consistency factor for conflicts
    fn calculate_conflict_consistency(&self, conflicts: &[ResourceConflict]) -> f64 {
        if conflicts.len() < 2 {
            return 1.0;
        }

        // Measure how consistent the conflict severities are
        let severities: Vec<f64> = conflicts
            .iter()
            .map(|c| match c.severity {
                ConflictSeverity::Fatal => 7.0,
                ConflictSeverity::Blocking => 6.0,
                ConflictSeverity::Critical => 5.0,
                ConflictSeverity::Severe => 4.0,
                ConflictSeverity::Major | ConflictSeverity::High => 3.0,
                ConflictSeverity::Moderate | ConflictSeverity::Medium => 2.0,
                ConflictSeverity::Minor | ConflictSeverity::Low => 1.0,
            })
            .collect();

        let mean = severities.iter().sum::<f64>() / severities.len() as f64;
        let variance =
            severities.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / severities.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 1.0 };

        (1.0 - coefficient_of_variation.min(1.0)).max(0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflict_history_new_returns_ok() {
        let history = ConflictHistory::new();
        assert!(history.is_ok());
        let h = history.expect("should succeed");
        assert!(h.conflicts.is_empty());
        assert_eq!(h.total_conflicts, 0);
    }

    #[test]
    fn test_resource_dependency_graph_new_and_populate() {
        let graph_result = ResourceDependencyGraph::new();
        assert!(
            graph_result.is_ok(),
            "ResourceDependencyGraph::new() should return Ok"
        );
        let mut graph = graph_result.expect("should succeed");

        graph.add_resource("resource_a".to_string());
        graph.add_resource("resource_b".to_string());
        graph.add_dependency("resource_a".to_string(), "resource_b".to_string(), 1.0);

        assert!(graph.nodes.contains(&"resource_a".to_string()));
        assert!(graph.nodes.contains(&"resource_b".to_string()));
        assert!(!graph.edges.is_empty());
    }

    #[test]
    fn test_resource_conflict_uses_resource_id() {
        // Verify the ResourceConflict type has resource_id field (not resources)
        // This is a compile-time check that our field access is correct
        let conflict = ResourceConflict {
            conflict_id: "test_conflict".to_string(),
            conflict_type: ConflictType::Lock,
            severity: ConflictSeverity::High,
            conflicting_tests: vec!["test_a".to_string(), "test_b".to_string()],
            resource_id: "shared_mutex".to_string(),
            probability: 0.8,
            performance_impact: ConflictImpact {
                performance_degradation: 0.5,
                reliability_impact: Some(0.3),
                resource_impact: std::collections::HashMap::new(),
                user_experience_impact: Some(0.2),
                stability_impact: Some(0.1),
                recovery_time: None,
                cascade_potential: Some(0.4),
                mitigation_effectiveness: Some(0.7),
                long_term_effects: Vec::new(),
                confidence: 0.9,
            },
            resolutions: Vec::new(),
            detected_at: std::time::Instant::now(),
            confidence: 0.9,
            historical_count: 0,
            max_safe_concurrency: 1,
        };

        assert_eq!(conflict.resource_id, "shared_mutex");
        assert!(!conflict.conflicting_tests.is_empty());
    }

    #[tokio::test]
    async fn test_conflict_detector_constructs_without_error() {
        let config = ConflictDetectionConfig::default();
        let detector = ResourceConflictDetector::new(config).await;
        assert!(
            detector.is_ok(),
            "ResourceConflictDetector::new() should succeed"
        );
    }
}
