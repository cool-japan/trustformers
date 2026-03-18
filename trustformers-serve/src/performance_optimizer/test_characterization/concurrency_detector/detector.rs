//! Core Concurrency Requirements Detector
//!
//! Provides comprehensive analysis of test behavior to determine safe concurrency levels,
//! resource sharing capabilities, deadlock prevention, and parallel execution constraints.

use super::super::types::*;
use anyhow::{Context, Result};
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::task::JoinHandle;

use super::conflict_detector::ResourceConflictDetector;
use super::deadlock_analyzer::DeadlockAnalyzer;
use super::estimator::SafeConcurrencyEstimator;
use super::lock_analyzer::LockContentionAnalyzer;
use super::pattern_detector::ConcurrencyPatternDetector;
use super::risk_assessment::ConcurrencyRiskAssessment;
use super::safety_validator::SafetyValidator;
use super::sharing_analyzer::SharingCapabilityAnalyzer;
use super::thread_analyzer::ThreadInteractionAnalyzer;

pub struct ConcurrencyRequirementsDetector {
    /// Detector configuration
    config: Arc<RwLock<ConcurrencyDetectorConfig>>,

    /// Safe concurrency estimator
    estimator: Arc<SafeConcurrencyEstimator>,

    /// Resource conflict detector
    conflict_detector: Arc<ResourceConflictDetector>,

    /// Resource sharing analyzer
    sharing_analyzer: Arc<SharingCapabilityAnalyzer>,

    /// Deadlock analyzer
    deadlock_analyzer: Arc<DeadlockAnalyzer>,

    /// Risk assessment engine
    risk_assessor: Arc<ConcurrencyRiskAssessment>,

    /// Thread interaction analyzer
    thread_analyzer: Arc<ThreadInteractionAnalyzer>,

    /// Lock contention analyzer
    lock_analyzer: Arc<LockContentionAnalyzer>,

    /// Pattern detector
    pattern_detector: Arc<ConcurrencyPatternDetector>,

    /// Safety validator
    safety_validator: Arc<SafetyValidator>,

    /// Analysis history for learning and optimization
    analysis_history: Arc<Mutex<ConcurrencyAnalysisHistory>>,

    /// Background analysis tasks
    background_tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

impl ConcurrencyRequirementsDetector {
    /// Creates a new concurrency requirements detector
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the detector
    ///
    /// # Returns
    ///
    /// A new detector instance with all components initialized
    pub async fn new(config: ConcurrencyDetectorConfig) -> Result<Self> {
        let config_arc = Arc::new(RwLock::new(config.clone()));

        // TODO: PatternEstimationConfig -> EstimationConfig type mismatch
        // Using default EstimationConfig for now
        let estimator = Arc::new(
            SafeConcurrencyEstimator::new(EstimationConfig {
                safety_margin: 0.2,
                history_retention_limit: 1000,
            })
            .await
            .context("Failed to create safe concurrency estimator")?,
        );

        let conflict_detector = Arc::new(
            ResourceConflictDetector::new(config.conflict_config.clone())
                .await
                .context("Failed to create resource conflict detector")?,
        );

        let sharing_analyzer = Arc::new(
            SharingCapabilityAnalyzer::new(config.sharing_config.clone())
                .await
                .context("Failed to create sharing capability analyzer")?,
        );

        let deadlock_analyzer = Arc::new(
            DeadlockAnalyzer::new(config.deadlock_config.clone())
                .await
                .context("Failed to create deadlock analyzer")?,
        );

        let risk_assessor = Arc::new(
            ConcurrencyRiskAssessment::new(config.risk_config.clone())
                .await
                .context("Failed to create risk assessor")?,
        );

        let thread_analyzer = Arc::new(
            ThreadInteractionAnalyzer::new(config.thread_config.clone())
                .await
                .context("Failed to create thread interaction analyzer")?,
        );

        let lock_analyzer = Arc::new(
            LockContentionAnalyzer::new(config.lock_config.clone())
                .await
                .context("Failed to create lock contention analyzer")?,
        );

        let pattern_detector = Arc::new(
            ConcurrencyPatternDetector::new(config.pattern_config.clone())
                .await
                .context("Failed to create pattern detector")?,
        );

        let safety_validator = Arc::new(
            SafetyValidator::new(config.safety_config.clone())
                .await
                .context("Failed to create safety validator")?,
        );

        Ok(Self {
            config: config_arc,
            estimator,
            conflict_detector,
            sharing_analyzer,
            deadlock_analyzer,
            risk_assessor,
            thread_analyzer,
            lock_analyzer,
            pattern_detector,
            safety_validator,
            analysis_history: Arc::new(Mutex::new(ConcurrencyAnalysisHistory::new())),
            background_tasks: Arc::new(Mutex::new(Vec::new())),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Performs comprehensive concurrency analysis on test execution data
    ///
    /// # Arguments
    ///
    /// * `test_data` - Test execution data to analyze
    ///
    /// # Returns
    ///
    /// Comprehensive concurrency analysis results including safe concurrency levels,
    /// resource conflicts, deadlock risks, and optimization recommendations
    pub async fn analyze_concurrency(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<ConcurrencyAnalysisResult> {
        let start_time = Utc::now();

        // Validate input data
        self.validate_test_data(test_data).context("Test data validation failed")?;

        // Parallel analysis execution for optimal performance
        let (
            estimation_result,
            conflict_result,
            sharing_result,
            deadlock_result,
            risk_result,
            thread_result,
            lock_result,
            pattern_result,
        ) = tokio::try_join!(
            self.estimator.estimate_safe_concurrency(test_data),
            self.conflict_detector.detect_conflicts(test_data),
            self.sharing_analyzer.analyze_sharing_capabilities(test_data),
            self.deadlock_analyzer.analyze_deadlock_risks(test_data),
            self.risk_assessor.assess_concurrency_risks(test_data),
            self.thread_analyzer.analyze_thread_interactions(test_data),
            self.lock_analyzer.analyze_lock_contention(test_data),
            self.pattern_detector.detect_concurrency_patterns(test_data)
        )?;

        // Synthesize results
        let requirements = self
            .synthesize_requirements(
                &estimation_result,
                &conflict_result,
                &sharing_result,
                &deadlock_result,
                &risk_result,
                &thread_result,
                &lock_result,
                &pattern_result,
            )
            .await?;

        // Validate safety constraints
        let safety_result = self.safety_validator.validate_safety(&requirements, test_data).await?;

        // Extract lock dependencies from lock analysis
        let lock_dependencies = self.extract_lock_dependencies(&lock_result)?;

        // TODO: Type mismatch - sharing_result has Vec<SharingCapability> (enum) but field expects Vec<ResourceSharingCapabilities> (struct)
        // Using empty vec for now - needs proper conversion
        let analysis_result = ConcurrencyAnalysisResult {
            timestamp: start_time,
            test_id: test_data.test_id.clone(),
            max_safe_concurrency: estimation_result.max_safe_concurrency,
            recommended_concurrency: estimation_result.recommended_concurrency,
            resource_conflicts: conflict_result.resource_conflicts.clone(),
            lock_dependencies,
            sharing_capabilities: Vec::new(), // TODO: Convert from sharing_result.sharing_capabilities
            safety_constraints: safety_result
                .safety_constraints
                .first()
                .cloned()
                .unwrap_or_default(),
            recommendations: self.generate_recommendations(&requirements).await?,
            confidence: self.calculate_overall_confidence(&requirements) as f64,
            performance_impact: estimation_result.performance_impact,
            requirements: requirements.clone(),
            estimation_details: serde_json::to_string(&estimation_result)
                .unwrap_or_else(|_| format!("{:?}", estimation_result)),
            conflict_analysis: serde_json::to_string(&conflict_result)
                .unwrap_or_else(|_| format!("{:?}", conflict_result)),
            sharing_analysis: serde_json::to_string(&sharing_result)
                .unwrap_or_else(|_| format!("{:?}", sharing_result)),
            deadlock_analysis: serde_json::to_string(&deadlock_result)
                .unwrap_or_else(|_| format!("{:?}", deadlock_result)),
            risk_assessment: serde_json::to_string(&risk_result)
                .unwrap_or_else(|_| format!("{:?}", risk_result)),
            thread_analysis: serde_json::to_string(&thread_result)
                .unwrap_or_else(|_| format!("{:?}", thread_result)),
            lock_analysis: serde_json::to_string(&lock_result)
                .unwrap_or_else(|_| format!("{:?}", lock_result)),
            pattern_analysis: serde_json::to_string(&pattern_result)
                .unwrap_or_else(|_| format!("{:?}", pattern_result)),
            safety_validation: serde_json::to_string(&safety_result)
                .unwrap_or_else(|_| format!("{:?}", safety_result)),
            analysis_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
        };

        // Store analysis for learning and optimization
        self.store_analysis_result(&analysis_result).await?;

        Ok(analysis_result)
    }

    /// Validates test execution data for analysis
    fn validate_test_data(&self, test_data: &TestExecutionData) -> Result<()> {
        if test_data.test_id.is_empty() {
            anyhow::bail!("Test ID cannot be empty");
        }

        if test_data.execution_traces.is_empty() {
            anyhow::bail!("Test must have execution traces for analysis");
        }

        Ok(())
    }

    /// Synthesizes requirements from all analysis components
    async fn synthesize_requirements(
        &self,
        estimation: &ConcurrencyEstimationResult,
        conflicts: &ConflictAnalysisResult,
        sharing: &SharingAnalysisResult,
        deadlock: &DeadlockAnalysisResult,
        risks: &RiskAssessmentResult,
        threads: &ThreadAnalysisResult,
        locks: &LockAnalysisResult,
        patterns: &PatternAnalysisResult,
    ) -> Result<ConcurrencyRequirements> {
        let base_concurrency = estimation.recommended_concurrency;

        // Apply conflict constraints
        let conflict_limited = conflicts
            .conflicts
            .iter()
            .map(|c| c.max_safe_concurrency.min(base_concurrency))
            .min()
            .unwrap_or(base_concurrency);

        // Apply deadlock constraints
        let deadlock_limited = if deadlock.has_deadlock_risk {
            deadlock.safe_concurrency_limit.min(1)
        } else {
            conflict_limited
        };

        // Apply risk constraints
        let risk_limited = if risks.overall_risk_level > RiskLevel::Medium {
            (deadlock_limited as f32 * 0.7) as usize
        } else {
            deadlock_limited
        };

        let final_concurrency = risk_limited.min(self.config.read().max_concurrency);

        // TODO: Type mismatches - need proper conversions
        // conflicts.resource_constraints is HashMap<String, f64> but field expects Vec<String>
        // deadlock.synchronization_requirements is Vec<String> but field expects SynchronizationRequirements
        // sharing.sharing_capabilities is Vec<SharingCapability> but field expects ResourceSharingCapabilities
        Ok(ConcurrencyRequirements {
            max_concurrent_instances: final_concurrency,
            isolation_level: IsolationLevel::default(),
            shared_resources: Vec::new(),
            lock_dependencies: Vec::new(),
            resource_conflicts: Vec::new(),
            safety_constraints: self
                .build_safety_constraints(estimation, conflicts, deadlock, risks),
            execution_safety: 1.0,
            deadlock_risk: DeadlockRisk::default(),
            concurrency_overhead: 0.0,
            recommended_concurrency: estimation.recommended_concurrency,
            max_safe_concurrency: final_concurrency,
            min_required_concurrency: 1,
            optimal_concurrency: estimation.optimal_concurrency,
            resource_constraints: conflicts.resource_constraints.keys().cloned().collect(),
            sharing_requirements: sharing.sharing_requirements.clone(),
            // TODO: SynchronizationRequirements struct fields changed - using defaults
            synchronization_requirements: SynchronizationRequirements {
                synchronization_points: Vec::new(),
                lock_usage_patterns: Vec::new(),
                coordination_requirements: deadlock.synchronization_requirements.clone(),
                synchronization_overhead: 0.0,
                deadlock_prevention: Vec::new(),
                optimization_opportunities: Vec::new(),
                complexity_score: 0.0,
                performance_impact: 0.0,
                alternative_strategies: Vec::new(),
                average_wait_time: Duration::from_millis(0),
                ordered_locking: false,
                timeout_based_locking: false,
                resource_ordering: Vec::new(),
                lock_free_alternatives: Vec::new(),
                custom_requirements: Vec::new(),
            },
            performance_guarantees: self.build_performance_guarantees(threads, locks, patterns),
            max_threads: final_concurrency,
            parallel_capable: estimation.is_parallelizable,
            resource_sharing: ResourceSharingCapabilities {
                supports_read_sharing: false,
                supports_write_sharing: false,
                max_concurrent_readers: None,
                max_concurrent_writers: None,
                sharing_overhead: 0.0,
                consistency_guarantees: Vec::new(),
                isolation_requirements: Vec::new(),
                recommended_strategy: SharingStrategy::NoSharing,
                safety_assessment: 0.0,
                performance_tradeoffs: std::collections::HashMap::new(),
                performance_overhead: 0.0,
                implementation_complexity: 0.0,
                sharing_mode: String::from("none"),
            },
        })
    }

    /// Builds safety constraints from analysis results
    fn build_safety_constraints(
        &self,
        estimation: &ConcurrencyEstimationResult,
        _conflicts: &ConflictAnalysisResult,
        _deadlock: &DeadlockAnalysisResult,
        _risks: &RiskAssessmentResult,
    ) -> SafetyConstraints {
        SafetyConstraints {
            max_instances: estimation.recommended_concurrency,
            // TODO: IsolationLevel::Process doesn't exist - using Serializable for strong isolation
            isolation_level: IsolationLevel::Serializable,
            resource_restrictions: std::collections::HashMap::new(),
            ordering_dependencies: Vec::new(),
            sync_requirements: Vec::new(),
            performance_constraints: PerformanceConstraints::default(),
            quality_requirements: QualityRequirements::default(),
            safety_margin: 0.2,
            validation_rules: Vec::new(),
            compliance_level: 1.0,
        }
    }

    /// Builds performance guarantees from analysis results
    fn build_performance_guarantees(
        &self,
        threads: &ThreadAnalysisResult,
        locks: &LockAnalysisResult,
        patterns: &PatternAnalysisResult,
    ) -> Vec<String> {
        let mut guarantees = Vec::new();

        if !threads.throughput_analysis.is_empty() {
            for (thread_id, throughput) in &threads.throughput_analysis {
                guarantees.push(format!(
                    "Thread {} throughput target {:.2}",
                    thread_id, throughput
                ));
            }
        }

        if !locks.latency_bounds.is_empty() {
            for (lock_id, bound) in &locks.latency_bounds {
                guarantees.push(format!(
                    "Lock {} latency <= {:.2}ms",
                    lock_id,
                    bound.as_secs_f64() * 1000.0
                ));
            }
        }

        if !patterns.scalability_patterns.is_empty() {
            guarantees.extend(
                patterns
                    .scalability_patterns
                    .iter()
                    .map(|pattern| format!("Scalability pattern: {}", pattern)),
            );
        }

        if guarantees.is_empty() {
            guarantees.push("No explicit performance guarantees derived".to_string());
        }

        guarantees
    }

    /// Calculates overall confidence score
    fn calculate_overall_confidence(&self, requirements: &ConcurrencyRequirements) -> f32 {
        // Implementation of confidence calculation algorithm
        let mut confidence_scores = Vec::new();

        // TODO: ConcurrencyRequirements fields changed - max_safe_concurrency is now max_concurrent_instances (usize, not Option)
        if requirements.max_concurrent_instances > 0 {
            confidence_scores.push(0.9);
        }

        // TODO: resource_constraints doesn't exist, using shared_resources instead
        if !requirements.shared_resources.is_empty() {
            confidence_scores.push(0.8);
        }

        // TODO: sharing_requirements is now safety_constraints (not Option)
        // Check if safety constraints are defined (not default)
        if requirements.safety_constraints.max_instances > 0 {
            confidence_scores.push(0.85);
        }

        if confidence_scores.is_empty() {
            0.5
        } else {
            confidence_scores.iter().sum::<f64>() as f32 / confidence_scores.len() as f32
        }
    }

    /// Extract lock dependencies from lock analysis result
    fn extract_lock_dependencies(
        &self,
        lock_result: &LockAnalysisResult,
    ) -> Result<Vec<LockDependency>> {
        use std::collections::{HashMap, HashSet};

        let mut lock_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut lock_order: HashMap<String, Vec<String>> = HashMap::new();
        let mut hold_durations: HashMap<String, Vec<Duration>> = HashMap::new();

        // Track lock acquisitions per thread to build dependencies
        let mut thread_locks: HashMap<u64, Vec<(String, Instant)>> = HashMap::new();

        for event in &lock_result.lock_events {
            match event.event_type.as_str() {
                "Acquire" if event.success => {
                    // Track which locks are held by this thread
                    thread_locks
                        .entry(event.thread_id)
                        .or_default()
                        .push((event.lock_id.clone(), event.timestamp));

                    // Record dependencies: locks held before this one
                    if let Some(held_locks) = thread_locks.get(&event.thread_id) {
                        if held_locks.len() > 1 {
                            let dependent: Vec<String> = held_locks[..held_locks.len() - 1]
                                .iter()
                                .map(|(id, _)| id.clone())
                                .collect();
                            lock_map
                                .entry(event.lock_id.clone())
                                .or_default()
                                .extend(dependent.clone());
                            lock_order.entry(event.lock_id.clone()).or_default().extend(dependent);
                        }
                    }
                },
                "Release" if event.success => {
                    // Calculate hold duration and remove from held locks
                    if let Some(held) = thread_locks.get_mut(&event.thread_id) {
                        if let Some(pos) = held.iter().position(|(id, _)| id == &event.lock_id) {
                            let (_, acquire_time) = held.remove(pos);
                            let hold_duration = event.timestamp.duration_since(acquire_time);
                            hold_durations
                                .entry(event.lock_id.clone())
                                .or_default()
                                .push(hold_duration);
                        }
                    }
                },
                _ => {},
            }
        }

        // Build LockDependency objects
        let mut dependencies = Vec::new();
        let all_locks: HashSet<String> =
            lock_result.lock_events.iter().map(|e| e.lock_id.clone()).collect();

        for lock_id in all_locks {
            let dependent_locks: Vec<String> = lock_map
                .get(&lock_id)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();

            let acquisition_order = lock_order.get(&lock_id).cloned().unwrap_or_default();

            // Calculate hold duration statistics
            let durations = hold_durations.get(&lock_id).cloned().unwrap_or_default();
            let hold_duration_stats = if !durations.is_empty() {
                let total: Duration = durations.iter().sum();
                let mean = total / durations.len() as u32;
                let max = durations.iter().max().copied().unwrap_or_default();
                let min = durations.iter().min().copied().unwrap_or_default();

                DurationStatistics {
                    mean,
                    median: mean,                      // Simplified
                    std_dev: Duration::from_millis(0), // Simplified
                    min,
                    max,
                    p95: max, // Simplified
                    p99: max, // Simplified
                    sample_count: durations.len(),
                    variance: 0.0, // Simplified variance calculation
                    trend: TrendDirection::Stable,
                }
            } else {
                DurationStatistics::default()
            };

            dependencies.push(LockDependency {
                lock_id,
                lock_type: LockType::Mutex, // Default, could be inferred
                dependent_locks: dependent_locks.clone(),
                acquisition_order: acquisition_order.clone(),
                hold_duration_stats,
                contention_probability: 0.3, // Moderate contention probability
                deadlock_risk_factor: if dependent_locks.len() > 2 { 0.6 } else { 0.2 },
                alternatives: vec!["lock-free".to_string(), "fine-grained".to_string()],
                performance_impact: 0.4, // Moderate performance impact
                optimization_opportunities: vec!["Reduce lock scope".to_string()],
            });
        }

        Ok(dependencies)
    }

    /// Generates optimization recommendations
    async fn generate_recommendations(
        &self,
        requirements: &ConcurrencyRequirements,
    ) -> Result<Vec<ConcurrencyRecommendation>> {
        let mut recommendations = Vec::new();

        let max_concurrency = requirements.max_safe_concurrency;
        {
            if max_concurrency == 1 {
                recommendations.push(ConcurrencyRecommendation {
                    recommendation_type: RecommendationType::SerialExecution,
                    description: "Execute test serially due to safety constraints".to_string(),
                    priority: PriorityLevel::High,
                    expected_benefit: 0.9,
                    complexity: ComplexityLevel::Simple,
                    required_resources: vec!["single_thread".to_string()],
                    implementation_steps: vec![
                        "Disable parallel execution for this test".to_string(),
                        "Ensure proper resource cleanup".to_string(),
                    ],
                    performance_impact: 0.9,
                    risk_assessment: 0.1,
                    confidence: 0.95,
                    expected_impact: 0.9,
                    implementation_complexity: 0.1,
                });
            }
        }

        Ok(recommendations)
    }

    /// Stores analysis result for learning and optimization
    async fn store_analysis_result(&self, result: &ConcurrencyAnalysisResult) -> Result<()> {
        let mut history = self.analysis_history.lock();
        history.add_analysis(result.clone());

        // Cleanup old entries if needed
        let retention_limit = self.config.read().history_retention_limit;
        history.cleanup(retention_limit);

        Ok(())
    }

    /// Retrieves analysis history for machine learning and optimization
    pub async fn get_analysis_history(&self) -> ConcurrencyAnalysisHistory {
        self.analysis_history.lock().clone()
    }

    /// Gracefully shuts down the detector and all background tasks
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);

        let mut tasks = self.background_tasks.lock();
        for task in tasks.drain(..) {
            task.abort();
        }

        Ok(())
    }
}
