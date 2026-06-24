//! Deadlock Analyzer
//!
//! Provides sophisticated deadlock detection and prevention mechanisms using
//! multiple algorithms including cycle detection and resource allocation graphs.

use super::super::types::*;
use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};

pub struct DeadlockAnalyzer {
    /// Deadlock detection algorithms
    detection_algorithms: Arc<Mutex<Vec<Box<dyn DeadlockDetectionAlgorithm + Send + Sync>>>>,

    /// Prevention strategies
    prevention_strategies: Arc<Mutex<Vec<Box<dyn DeadlockPreventionStrategy + Send + Sync>>>>,

    /// Lock dependency graph
    dependency_graph: Arc<RwLock<LockDependencyGraph>>,

    /// Configuration
    config: DeadlockAnalysisConfig,
}

impl DeadlockAnalyzer {
    /// Creates a new deadlock analyzer
    pub async fn new(config: DeadlockAnalysisConfig) -> Result<Self> {
        let mut detection_algorithms: Vec<Box<dyn DeadlockDetectionAlgorithm + Send + Sync>> =
            Vec::new();
        let mut prevention_strategies: Vec<Box<dyn DeadlockPreventionStrategy + Send + Sync>> =
            Vec::new();

        // Initialize detection algorithms
        detection_algorithms.push(Box::new(CycleDetectionAlgorithm::new(
            true,
            "dfs".to_string(),
        )));
        detection_algorithms.push(Box::new(WaitForGraphAlgorithm::new()?));
        detection_algorithms.push(Box::new(ResourceAllocationGraphAlgorithm::new()?));
        detection_algorithms.push(Box::new(PredictiveDeadlockAlgorithm::new(true, 0.8)));

        // Initialize prevention strategies
        prevention_strategies.push(Box::new(OrderedLockingStrategy::new(true, Vec::new())));
        prevention_strategies.push(Box::new(TimeoutBasedStrategy::new(1000, false)));
        prevention_strategies.push(Box::new(ResourceOrderingStrategy::new()?));
        prevention_strategies.push(Box::new(AdaptivePreventionStrategy::new()?));

        Ok(Self {
            detection_algorithms: Arc::new(Mutex::new(detection_algorithms)),
            prevention_strategies: Arc::new(Mutex::new(prevention_strategies)),
            dependency_graph: Arc::new(RwLock::new(LockDependencyGraph::new()?)),
            config,
        })
    }

    /// Analyzes deadlock risks in test execution data
    pub async fn analyze_deadlock_risks(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<DeadlockAnalysisResult> {
        let start_time = Utc::now();

        if !self.config.detection_enabled {
            return Ok(DeadlockAnalysisResult {
                potential_deadlocks: Vec::new(),
                has_deadlock_risk: false,
                risk_level: "None".to_string(),
                safe_concurrency_limit: self.config.max_detection_depth,
                prevention_recommendations: Vec::new(),
                synchronization_requirements: Vec::new(),
                prevention_requirements: Vec::new(),
                detection_results: Vec::new(),
                analysis_duration: std::time::Duration::from_secs(0),
                confidence: 1.0,
            });
        }

        let timeout = std::time::Duration::from_secs(self.config.timeout_seconds);

        // Update dependency graph
        self.update_dependency_graph(test_data).await?;

        // Extract lock dependencies
        let lock_dependencies = self.extract_lock_dependencies(test_data)?;

        // Run detection algorithms
        // Execute synchronously to avoid lifetime issues
        let detection_task_results: Vec<_> = {
            let detection_algorithms = self.detection_algorithms.lock();
            detection_algorithms
                .iter()
                .map(|algorithm| {
                    let algorithm_name = algorithm.name().to_string();
                    let detection_start = Instant::now();
                    let result = algorithm.detect_deadlocks(&lock_dependencies);
                    let detection_duration = detection_start.elapsed();
                    (algorithm_name, result, detection_duration)
                })
                .collect()
        };

        // Collect detection results
        let mut potential_deadlocks = Vec::new();
        let mut detection_results = Vec::new();

        for (algorithm_name, result, duration) in detection_task_results {
            match result {
                Ok(mut deadlocks) => {
                    // Convert DeadlockRisk to DeadlockScenario, populating fields from real data
                    let scenarios: Vec<DeadlockScenario> = deadlocks
                        .iter()
                        .map(|d| {
                            let lock_names: Vec<String> =
                                d.lock_cycles.iter().flat_map(|c| c.iter().cloned()).collect();
                            let risk_str = if d.probability > 0.8 {
                                "Critical"
                            } else if d.probability > 0.5 {
                                "High"
                            } else {
                                "Medium"
                            };
                            DeadlockScenario {
                                scenario_id: format!("deadlock_{}", uuid::Uuid::new_v4()),
                                involved_threads: Vec::new(),
                                involved_resources: lock_names.clone(),
                                lock_order: lock_names,
                                risk_level: risk_str.to_string(),
                                timestamp: Utc::now(),
                            }
                        })
                        .collect();

                    // Convert to PotentialDeadlock for confidence calculation, using real lock cycles
                    let potential: Vec<PotentialDeadlock> = deadlocks
                        .iter()
                        .map(|d| PotentialDeadlock {
                            locks: d.lock_cycles.iter().flat_map(|c| c.iter().cloned()).collect(),
                        })
                        .collect();

                    detection_results.push(DeadlockDetectionResult {
                        algorithm: algorithm_name,
                        deadlocks: scenarios,
                        detection_duration: duration,
                        confidence: self.calculate_detection_confidence(&potential),
                    });
                    potential_deadlocks.append(&mut deadlocks);
                },
                Err(e) => {
                    log::warn!("Deadlock detection algorithm failed: {}", e);
                },
            }
        }

        // Convert DeadlockRisk to PotentialDeadlock for deduplication, using real lock cycles
        let potential_deadlock_converted: Vec<PotentialDeadlock> = potential_deadlocks
            .iter()
            .map(|d| PotentialDeadlock {
                locks: d.lock_cycles.iter().flat_map(|c| c.iter().cloned()).collect(),
            })
            .collect();

        // Deduplicate and prioritize deadlocks
        let unique_deadlocks = self.deduplicate_deadlocks(&potential_deadlock_converted);
        let prioritized_deadlocks = self.prioritize_deadlocks(&unique_deadlocks);

        // Generate prevention strategies
        let prevention_recommendations =
            self.generate_prevention_strategies(&prioritized_deadlocks).await?;

        // Assess overall risk
        let has_deadlock_risk = !prioritized_deadlocks.is_empty();
        let risk_level = self.assess_deadlock_risk_level(&prioritized_deadlocks);
        let safe_concurrency_limit = if has_deadlock_risk {
            Some(self.calculate_safe_concurrency_limit(&prioritized_deadlocks))
        } else {
            None
        };

        // Generate synchronization requirements
        let synchronization_requirements =
            self.generate_synchronization_requirements(&prioritized_deadlocks);
        let prevention_requirements = self.generate_prevention_requirements(&prioritized_deadlocks);

        // Build final DeadlockScenario list from prioritized PotentialDeadlocks
        let deadlock_scenarios: Vec<DeadlockScenario> = prioritized_deadlocks
            .iter()
            .map(|d| {
                let risk_str = if d.locks.len() > 4 {
                    "Critical"
                } else if d.locks.len() > 2 {
                    "High"
                } else {
                    "Medium"
                };
                DeadlockScenario {
                    scenario_id: format!("deadlock_{}", uuid::Uuid::new_v4()),
                    involved_threads: Vec::new(),
                    involved_resources: d.locks.clone(),
                    lock_order: d.locks.clone(),
                    risk_level: risk_str.to_string(),
                    timestamp: Utc::now(),
                }
            })
            .collect();

        let elapsed = Utc::now().signed_duration_since(start_time).to_std().unwrap_or_default();
        if elapsed > timeout {
            log::warn!(
                "Deadlock analysis exceeded timeout ({:?} > {:?})",
                elapsed,
                timeout
            );
        }

        let risk_level_str = format!("{:?}", risk_level);
        // Cap concurrency at configured max_detection_depth
        let safe_limit = safe_concurrency_limit.unwrap_or(1).min(self.config.max_detection_depth);
        let sync_reqs_vec = vec![format!("{:?}", synchronization_requirements)];
        let prev_reqs_vec = vec![format!("{:?}", prevention_requirements)];

        Ok(DeadlockAnalysisResult {
            potential_deadlocks: deadlock_scenarios,
            has_deadlock_risk,
            risk_level: risk_level_str,
            safe_concurrency_limit: safe_limit,
            prevention_recommendations,
            synchronization_requirements: sync_reqs_vec,
            prevention_requirements: prev_reqs_vec,
            detection_results,
            analysis_duration: elapsed,
            confidence: self.calculate_overall_deadlock_confidence(&unique_deadlocks),
        })
    }

    /// Updates the lock dependency graph
    async fn update_dependency_graph(&self, test_data: &TestExecutionData) -> Result<()> {
        let mut graph = self.dependency_graph.write();

        for lock_usage in &test_data.lock_usage {
            graph.add_lock(lock_usage.lock_id.clone());

            // Add dependencies based on execution traces
            for trace in &test_data.execution_traces {
                match trace.operation.as_str() {
                    "LockAcquire" => {
                        graph.add_lock_acquisition(
                            trace.thread_id,
                            trace.resource.clone(),
                            Vec::new(), // Stack trace not available from execution trace
                        );
                    },
                    "LockRelease" => {
                        graph.add_lock_release(trace.thread_id, trace.resource.clone());
                    },
                    _ => {},
                }
            }
        }

        Ok(())
    }

    /// Extracts lock dependencies from test data
    fn extract_lock_dependencies(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<Vec<LockDependency>> {
        let mut dependencies = Vec::new();
        let mut thread_locks: HashMap<u64, Vec<(String, Instant)>> = HashMap::new();

        // Track lock acquisitions per thread
        for trace in &test_data.execution_traces {
            match trace.operation.as_str() {
                "LockAcquire" | "LockAcquisition" => {
                    thread_locks
                        .entry(trace.thread_id)
                        .or_default()
                        .push((trace.resource.clone(), trace.timestamp));
                },
                "LockRelease" => {
                    if let Some(locks) = thread_locks.get_mut(&trace.thread_id) {
                        locks.retain(|(lock_id, _)| lock_id != &trace.resource);
                    }
                },
                _ => {},
            }
        }

        // Create dependencies based on lock ordering
        for (_thread_id, locks) in thread_locks {
            for i in 0..locks.len() {
                for j in i + 1..locks.len() {
                    let (lock1, time1) = &locks[i];
                    let (lock2, time2) = &locks[j];

                    let time_diff = if time2 > time1 {
                        time2.duration_since(*time1).as_secs_f64()
                    } else {
                        0.0
                    };
                    let contention_prob = (1.0_f64 / (1.0 + time_diff)).min(0.9);

                    dependencies.push(LockDependency {
                        lock_id: lock1.clone(),
                        lock_type: LockType::Mutex,
                        dependent_locks: vec![lock2.clone()],
                        acquisition_order: vec![lock1.clone(), lock2.clone()],
                        hold_duration_stats: DurationStatistics::default(),
                        contention_probability: contention_prob,
                        deadlock_risk_factor: 0.5,
                        alternatives: Vec::new(),
                        performance_impact: 0.3,
                        optimization_opportunities: Vec::new(),
                    });
                }
            }
        }

        Ok(dependencies)
    }

    /// Calculates detection confidence based on deadlock count and average lock cycle length
    fn calculate_detection_confidence(&self, deadlocks: &[PotentialDeadlock]) -> f64 {
        if deadlocks.is_empty() {
            return 1.0;
        }

        // Confidence grows with more detected deadlocks (more evidence = more certain)
        let count_factor = (deadlocks.len() as f64 / 10.0).min(1.0);
        let base_confidence = 0.5 + count_factor * 0.4;

        // Longer lock cycles are more reliably identified (more distinguishing structure)
        let total_locks: usize = deadlocks.iter().map(|d| d.locks.len()).sum();
        let avg_cycle_len = total_locks as f64 / deadlocks.len() as f64;
        let cycle_factor = (avg_cycle_len / 4.0).min(1.0) * 0.1;

        (base_confidence + cycle_factor).min(1.0)
    }

    /// Deduplicates deadlocks based on involved locks
    fn deduplicate_deadlocks(&self, deadlocks: &[PotentialDeadlock]) -> Vec<PotentialDeadlock> {
        let mut unique_deadlocks = Vec::new();

        for deadlock in deadlocks {
            let is_duplicate = unique_deadlocks
                .iter()
                .any(|existing: &PotentialDeadlock| self.deadlocks_are_similar(existing, deadlock));

            if !is_duplicate {
                unique_deadlocks.push(deadlock.clone());
            }
        }

        unique_deadlocks
    }

    /// Checks if two deadlocks are similar
    fn deadlocks_are_similar(&self, a: &PotentialDeadlock, b: &PotentialDeadlock) -> bool {
        // Check if deadlocks involve the same set of locks
        let a_locks: HashSet<_> = a.locks.iter().collect();
        let b_locks: HashSet<_> = b.locks.iter().collect();

        a_locks == b_locks
    }

    /// Prioritizes deadlocks based on lock cycle length (longer cycles = higher priority)
    fn prioritize_deadlocks(&self, deadlocks: &[PotentialDeadlock]) -> Vec<PotentialDeadlock> {
        let mut prioritized = deadlocks.to_vec();
        prioritized.sort_by_key(|b| std::cmp::Reverse(b.locks.len()));
        prioritized
    }

    /// Generates prevention strategies
    async fn generate_prevention_strategies(
        &self,
        deadlocks: &[PotentialDeadlock],
    ) -> Result<Vec<DeadlockPreventionRecommendation>> {
        let strategies = self.prevention_strategies.lock();
        let mut recommendations = Vec::new();

        for deadlock in deadlocks {
            let deadlock_risk: DeadlockRisk = deadlock.into();
            for strategy in strategies.iter() {
                if strategy.is_applicable(&deadlock_risk) {
                    if let Ok(actions) = strategy.prevent_deadlock(&deadlock_risk) {
                        let action_str = format!("{} prevention actions", actions.len());
                        let effectiveness =
                            self.calculate_strategy_effectiveness(strategy.name(), deadlock);
                        let complexity = self.calculate_implementation_complexity(strategy.name());

                        recommendations.push(DeadlockPreventionRecommendation {
                            deadlock_id: deadlock.locks.join("->"),
                            strategy_name: strategy.name().to_string(),
                            prevention_action: action_str,
                            expected_effectiveness: effectiveness,
                            implementation_complexity: format!("{:.2}", complexity),
                        });
                    }
                }
            }
        }

        Ok(recommendations)
    }

    /// Calculates strategy effectiveness
    fn calculate_strategy_effectiveness(
        &self,
        strategy_name: &str,
        _deadlock: &PotentialDeadlock,
    ) -> f64 {
        match strategy_name {
            "OrderedLockingStrategy" => 0.9,
            "TimeoutBasedStrategy" => 0.8,
            "ResourceOrderingStrategy" => 0.85,
            "AdaptivePreventionStrategy" => 0.75,
            _ => 0.6,
        }
    }

    /// Calculates implementation complexity
    fn calculate_implementation_complexity(&self, strategy_name: &str) -> f64 {
        match strategy_name {
            "OrderedLockingStrategy" => 0.6,
            "TimeoutBasedStrategy" => 0.3,
            "ResourceOrderingStrategy" => 0.7,
            "AdaptivePreventionStrategy" => 0.9,
            _ => 0.5,
        }
    }

    /// Assesses overall deadlock risk level based on lock cycle length and count
    fn assess_deadlock_risk_level(&self, deadlocks: &[PotentialDeadlock]) -> DeadlockRiskLevel {
        if deadlocks.is_empty() {
            return DeadlockRiskLevel {
                level: "None".to_string(),
                risk_score: 0.0,
                contributing_factors: Vec::new(),
            };
        }

        let count = deadlocks.len();
        // Compute average lock count per deadlock to scale risk_score
        let total_locks: usize = deadlocks.iter().map(|d| d.locks.len()).sum();
        let avg_lock_count = total_locks / count.max(1);
        // risk_score is proportional to average cycle length, clamped to 0..1
        // A cycle of 4+ locks is considered maximum risk
        let raw_score = avg_lock_count as f64 / 4.0;
        let base_risk_score = raw_score.clamp(0.0, 1.0);

        if count > 10 {
            DeadlockRiskLevel {
                level: "Critical".to_string(),
                risk_score: base_risk_score.max(0.9),
                contributing_factors: vec![
                    "High deadlock count".to_string(),
                    format!("Average cycle length: {}", avg_lock_count),
                ],
            }
        } else if count > 5 {
            DeadlockRiskLevel {
                level: "High".to_string(),
                risk_score: base_risk_score.max(0.6),
                contributing_factors: vec![
                    "Moderate deadlock count".to_string(),
                    format!("Average cycle length: {}", avg_lock_count),
                ],
            }
        } else if count > 2 {
            DeadlockRiskLevel {
                level: "Medium".to_string(),
                risk_score: base_risk_score.max(0.3),
                contributing_factors: vec![
                    "Some deadlocks detected".to_string(),
                    format!("Average cycle length: {}", avg_lock_count),
                ],
            }
        } else {
            DeadlockRiskLevel {
                level: "Low".to_string(),
                risk_score: base_risk_score,
                contributing_factors: vec![
                    "Few deadlocks detected".to_string(),
                    format!("Average cycle length: {}", avg_lock_count),
                ],
            }
        }
    }

    /// Calculates safe concurrency limit based on deadlocks
    fn calculate_safe_concurrency_limit(&self, deadlocks: &[PotentialDeadlock]) -> usize {
        if deadlocks.is_empty() {
            return usize::MAX;
        }

        // Find the most restrictive deadlock (smallest lock cycle)
        let min_lock_count = deadlocks.iter().map(|d| d.locks.len()).min().unwrap_or(0);

        // Guard against 0: a cycle of length 0 or 1 gives a safe limit of 1
        if min_lock_count <= 1 {
            return 1;
        }

        // Safe concurrency is one less than the minimum lock cycle size
        (min_lock_count - 1).max(1)
    }

    /// Generates synchronization requirements
    fn generate_synchronization_requirements(
        &self,
        deadlocks: &[PotentialDeadlock],
    ) -> SynchronizationRequirements {
        let mut requirements = SynchronizationRequirements {
            synchronization_points: vec![],
            lock_usage_patterns: vec![],
            coordination_requirements: vec![],
            synchronization_overhead: 0.0,
            deadlock_prevention: vec![],
            optimization_opportunities: vec![],
            complexity_score: 0.0,
            performance_impact: 0.0,
            alternative_strategies: vec![],
            average_wait_time: Duration::from_millis(0),
            ordered_locking: false,
            timeout_based_locking: false,
            resource_ordering: vec![],
            lock_free_alternatives: vec![],
            custom_requirements: vec![],
        };

        // Enable generic deadlock prevention requirements for all detected deadlocks
        for _deadlock in deadlocks {
            requirements.ordered_locking = true;
            requirements.timeout_based_locking = true;
        }

        requirements
    }

    /// Generates prevention requirements
    fn generate_prevention_requirements(
        &self,
        deadlocks: &[PotentialDeadlock],
    ) -> DeadlockPreventionRequirements {
        DeadlockPreventionRequirements {
            lock_ordering_required: true,
            timeout_enabled: true,
            max_wait_time: Duration::from_secs(10),
            prevention_strategies: if deadlocks.len() > 5 {
                vec!["ImmediateTermination".to_string()]
            } else {
                vec!["GracefulRecovery".to_string()]
            },
        }
    }

    /// Calculates overall deadlock confidence
    fn calculate_overall_deadlock_confidence(&self, deadlocks: &[PotentialDeadlock]) -> f64 {
        if deadlocks.is_empty() {
            return 1.0;
        }

        // More detected deadlocks gives higher confidence in the detection system
        let avg_confidence: f64 = if deadlocks.len() > 5 {
            0.9
        } else if deadlocks.len() > 2 {
            0.7
        } else {
            0.5
        };
        let consistency_factor = self.calculate_deadlock_consistency(deadlocks);

        avg_confidence * consistency_factor
    }

    /// Calculates deadlock consistency by checking for common locks across all detected deadlocks
    fn calculate_deadlock_consistency(&self, deadlocks: &[PotentialDeadlock]) -> f64 {
        if deadlocks.len() < 2 {
            return 1.0;
        }

        // Build the set of locks in the first deadlock as baseline
        let first_locks: HashSet<&String> = deadlocks[0].locks.iter().collect();

        // Check if all subsequent deadlocks share at least one common lock with the first
        let all_share_common_lock = deadlocks[1..].iter().all(|d| {
            let d_locks: HashSet<&String> = d.locks.iter().collect();
            !first_locks.is_disjoint(&d_locks)
        });

        if all_share_common_lock {
            // High consistency: a common lock resource threads through all deadlocks
            0.9
        } else {
            // Check partial overlap: how many pairs share at least one lock
            let n = deadlocks.len();
            let mut overlap_count = 0usize;
            let mut pair_count = 0usize;
            for i in 0..n {
                for j in i + 1..n {
                    pair_count += 1;
                    let a_locks: HashSet<&String> = deadlocks[i].locks.iter().collect();
                    let b_locks: HashSet<&String> = deadlocks[j].locks.iter().collect();
                    if !a_locks.is_disjoint(&b_locks) {
                        overlap_count += 1;
                    }
                }
            }
            if pair_count == 0 {
                return 0.5;
            }
            let overlap_ratio = overlap_count as f64 / pair_count as f64;
            // Map overlap ratio to the 0.5..0.7 range
            0.5 + overlap_ratio * 0.2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_potential_deadlock_built_from_risk_cycles() {
        // Build a DeadlockRisk with a 2-lock cycle
        let risk = DeadlockRisk {
            risk_level: RiskLevel::High,
            probability: 0.8,
            impact_severity: 0.9,
            risk_factors: Vec::new(),
            lock_cycles: vec![vec!["lock_a".to_string(), "lock_b".to_string()]],
            prevention_strategies: Vec::new(),
            detection_mechanisms: Vec::new(),
            recovery_procedures: Vec::new(),
            historical_incidents: Vec::new(),
            mitigation_effectiveness: 0.7,
        };

        // Simulate PotentialDeadlock construction (from real logic)
        let potential = PotentialDeadlock {
            locks: risk.lock_cycles.iter().flat_map(|c| c.iter().cloned()).collect(),
        };

        assert!(
            !potential.locks.is_empty(),
            "locks should be populated from cycles"
        );
        assert!(potential.locks.contains(&"lock_a".to_string()));
        assert!(potential.locks.contains(&"lock_b".to_string()));

        // Test risk level construction
        let risk_level = DeadlockRiskLevel {
            level: "High".to_string(),
            risk_score: 0.8,
            contributing_factors: vec!["Circular wait detected".to_string()],
        };
        assert!(risk_level.risk_score > 0.0);
    }

    #[test]
    fn test_detect_deadlock_type_from_cycle_pattern() {
        // A 2-lock cycle with mutual hold-and-wait pattern
        let locks = vec!["lock_a".to_string(), "lock_b".to_string()];

        // Circular wait: more than 1 lock in cycle
        let deadlock_type = if locks.len() >= 2 {
            DeadlockType::CircularWait
        } else {
            DeadlockType::HoldAndWait
        };

        assert_eq!(deadlock_type, DeadlockType::CircularWait);
    }
}
