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

    /// Conflict history for learning
    conflict_history: Arc<Mutex<ConflictHistory>>,

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

        // Initialize detection algorithms
        algorithms.push(Box::new(StaticConflictDetectionAlgorithm::new(true, 3)));
        algorithms.push(Box::new(DynamicConflictDetectionAlgorithm::new(true, 0.1)));
        algorithms.push(Box::new(PredictiveConflictDetectionAlgorithm::new(5, 0.8)));
        algorithms.push(Box::new(MLConflictDetectionAlgorithm::new(
            "default".to_string(),
            0.85,
        )));

        // Initialize resolution strategies
        strategies.push(Box::new(AvoidanceResolutionStrategy::new(true, false)));
        strategies.push(Box::new(TimeoutResolutionStrategy::new(1000, true)));
        strategies.push(Box::new(PartitioningResolutionStrategy::new()?));
        strategies.push(Box::new(AdaptiveResolutionStrategy::new()?));

        Ok(Self {
            algorithms: Arc::new(Mutex::new(algorithms)),
            resolution_strategies: Arc::new(Mutex::new(strategies)),
            // TODO: Changed ConflictHistory::new() to ::default() to fix E0308 - new() returns Result
            conflict_history: Arc::new(Mutex::new(ConflictHistory::default())),
            // TODO: Changed ResourceDependencyGraph::new() to ::default() to fix E0308 - new() returns Result
            dependency_graph: Arc::new(RwLock::new(ResourceDependencyGraph::default())),
            config,
        })
    }

    /// Detects resource conflicts in test execution data
    pub async fn detect_conflicts(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<ConflictAnalysisResult> {
        let start_time = Utc::now();

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
                    // TODO: detect_conflicts takes 1 argument, removed dependency_graph parameter
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
                        confidence: self.calculate_detection_confidence(&conflicts) as f64,
                    });
                    all_conflicts.append(&mut conflicts);
                },
                Err(e) => {
                    log::warn!("Conflict detection algorithm failed: {}", e);
                },
            }
        }

        // Deduplicate and prioritize conflicts
        let unique_conflicts = self.deduplicate_conflicts(&all_conflicts);
        let prioritized_conflicts = self.prioritize_conflicts(&unique_conflicts);

        // Generate resolution strategies
        let resolutions = self.generate_resolutions(&prioritized_conflicts).await?;

        // Calculate resource constraints
        let resource_constraints_vec = self.calculate_resource_constraints(&prioritized_conflicts);
        let resource_constraints: HashMap<String, f64> = resource_constraints_vec
            .iter()
            .map(|c| (c.resource_type.clone(), c.max_value))
            .collect();
        let resource_limits_f32 = self.calculate_resource_limits(&prioritized_conflicts);
        let resource_limits: HashMap<String, usize> = resource_limits_f32
            .iter()
            .map(|(k, v)| (k.clone(), (*v as usize).max(1)))
            .collect();
        let isolation_requirements = self.generate_isolation_requirements(&prioritized_conflicts);

        Ok(ConflictAnalysisResult {
            conflicts: prioritized_conflicts.clone(),
            resource_conflicts: prioritized_conflicts,
            resolutions,
            resource_constraints,
            resource_limits,
            isolation_requirements,
            detection_results: algorithm_results,
            analysis_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence: self.calculate_overall_conflict_confidence(&unique_conflicts) as f64,
        })
    }

    /// Updates the resource dependency graph
    async fn update_dependency_graph(&self, test_data: &TestExecutionData) -> Result<()> {
        let mut graph = self.dependency_graph.write();

        for pattern in &test_data.resource_access_patterns {
            graph.add_resource(pattern.resource_id.clone());

            for trace in &test_data.execution_traces {
                if trace.resource == pattern.resource_id {
                    // Convert Instant timestamp to f64 seconds
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
    fn calculate_detection_confidence(&self, conflicts: &[ResourceConflict]) -> f32 {
        if conflicts.is_empty() {
            return 1.0;
        }

        let avg_probability =
            conflicts.iter().map(|c| c.probability).sum::<f64>() as f32 / conflicts.len() as f32;
        let severity_factor = conflicts
            .iter()
            .map(|c| match c.severity {
                ConflictSeverity::Fatal => 1.0,
                ConflictSeverity::Blocking => 0.95,
                ConflictSeverity::Critical => 0.9,
                ConflictSeverity::Severe => 0.8,
                ConflictSeverity::Major | ConflictSeverity::High => 0.6,
                ConflictSeverity::Moderate | ConflictSeverity::Medium => 0.4,
                ConflictSeverity::Minor | ConflictSeverity::Low => 0.2,
            })
            .sum::<f64>() as f32
            / conflicts.len() as f32;

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
        // Check if conflicts involve the same resources
        // TODO: ResourceConflict no longer has resources field, only resource_id
        let resource_overlap = a.resource_id == b.resource_id;

        // Check if conflict types are compatible
        // TODO: ConflictType enum simplified - using new variants
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
            // TODO: ResourceConflict no longer has resources field, only resource_id
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
                // TODO: ConflictType enum simplified - mapping new variants to constraint types
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
    fn calculate_resource_limits(&self, conflicts: &[ResourceConflict]) -> HashMap<String, f32> {
        let mut limits: HashMap<String, f32> = HashMap::new();

        for conflict in conflicts {
            // TODO: ResourceConflict no longer has resources field, only resource_id
            let limit: f32 = match conflict.severity {
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
    fn calculate_overall_conflict_confidence(&self, conflicts: &[ResourceConflict]) -> f32 {
        if conflicts.is_empty() {
            return 1.0;
        }

        let avg_probability =
            conflicts.iter().map(|c| c.probability).sum::<f64>() as f32 / conflicts.len() as f32;
        let consistency_factor = self.calculate_conflict_consistency(conflicts);

        avg_probability * consistency_factor
    }

    /// Calculates consistency factor for conflicts
    fn calculate_conflict_consistency(&self, conflicts: &[ResourceConflict]) -> f32 {
        if conflicts.len() < 2 {
            return 1.0;
        }

        // Measure how consistent the conflict severities are
        let severities: Vec<f32> = conflicts
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

        let mean =
            severities.iter().map(|&s| s as f64).sum::<f64>() as f32 / severities.len() as f32;
        let variance = severities.iter().map(|&s| (s - mean).powi(2) as f64).sum::<f64>() as f32
            / severities.len() as f32;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 1.0 };

        (1.0 - coefficient_of_variation.min(1.0)).max(0.1)
    }
}
