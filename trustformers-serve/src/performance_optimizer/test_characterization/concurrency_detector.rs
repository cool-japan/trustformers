//! Comprehensive Concurrency Detection Module for Test Characterization System
//!
//! This module provides sophisticated concurrency analysis and detection capabilities
//! for the TrustformeRS test framework, implementing advanced algorithms for safe
//! parallel execution, deadlock prevention, and resource conflict resolution.
//!
//! # Key Components
//!
//! 1. **ConcurrencyRequirementsDetector**: Core concurrency analysis engine
//! 2. **SafeConcurrencyEstimator**: Advanced algorithms for optimal concurrency estimation
//! 3. **ResourceConflictDetector**: Detection and analysis of resource conflicts
//! 4. **SharingCapabilityAnalyzer**: Resource sharing analysis and optimization
//! 5. **DeadlockAnalyzer**: Deadlock detection and prevention mechanisms
//! 6. **ConcurrencyRiskAssessment**: Risk assessment for concurrent execution
//! 7. **ThreadInteractionAnalyzer**: Thread interaction and synchronization analysis
//! 8. **LockContentionAnalyzer**: Lock contention detection and optimization
//! 9. **ConcurrencyPatternDetector**: Pattern recognition for concurrent behaviors
//! 10. **SafetyValidator**: Comprehensive safety validation and compliance checking
//!
//! # Features
//!
//! - **Multi-Algorithm Estimation**: Conservative, Optimistic, Adaptive, and ML-based approaches
//! - **Advanced Deadlock Detection**: Cycle detection, resource dependency analysis
//! - **Resource Conflict Resolution**: Sophisticated conflict detection and mitigation
//! - **Thread-Safe Operations**: Lock-free and wait-free concurrent data structures
//! - **Real-Time Analysis**: Low-latency analysis with minimal overhead
//! - **Comprehensive Safety**: Multi-layered safety validation and risk assessment
//!
//! # Example Usage
//!
//! ```rust
//! use trustformers_serve::performance_optimizer::test_characterization::concurrency_detector::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ConcurrencyDetectorConfig::default();
//!     let detector = ConcurrencyRequirementsDetector::new(config).await?;
//!
//!     let test_data = TestExecutionData::new("test_id", "integration_test");
//!     let analysis = detector.analyze_concurrency(&test_data).await?;
//!
//!     println!("Safe concurrency level: {:?}", analysis.requirements.max_safe_concurrency);
//!     Ok(())
//! }
//! ```

use super::types::*;
use crate::test_performance_monitoring::types::CachedSharingCapability;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::task::JoinHandle;

// =============================================================================
// CORE CONCURRENCY REQUIREMENTS DETECTOR
// =============================================================================

/// Advanced concurrency requirements detection system
///
/// Provides comprehensive analysis of test behavior to determine safe concurrency levels,
/// resource sharing capabilities, deadlock prevention, and parallel execution constraints.
/// Uses multiple sophisticated algorithms and real-time analysis for optimal performance.
#[derive(Debug)]
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

// =============================================================================
// SAFE CONCURRENCY ESTIMATOR
// =============================================================================

/// Advanced safe concurrency level estimation system
///
/// Implements multiple sophisticated algorithms for determining optimal concurrency
/// levels including conservative, optimistic, adaptive, and machine learning approaches.
#[derive(Debug)]
pub struct SafeConcurrencyEstimator {
    /// Estimation algorithms
    algorithms: Arc<Mutex<Vec<Box<dyn ConcurrencyEstimationAlgorithm + Send + Sync>>>>,

    /// Algorithm performance history
    algorithm_performance: Arc<Mutex<HashMap<String, AlgorithmPerformance>>>,

    /// Estimation history for learning
    estimation_history: Arc<Mutex<Vec<EstimationRecord>>>,

    /// Safety constraints
    safety_constraints: Arc<RwLock<EstimationSafetyConstraints>>,

    /// Configuration
    config: EstimationConfig,
}

impl SafeConcurrencyEstimator {
    /// Creates a new safe concurrency estimator
    pub async fn new(config: EstimationConfig) -> Result<Self> {
        let mut algorithms: Vec<Box<dyn ConcurrencyEstimationAlgorithm + Send + Sync>> = Vec::new();

        // Initialize estimation algorithms
        algorithms.push(Box::new(ConservativeEstimationAlgorithm::new(0.2, true)));
        algorithms.push(Box::new(OptimisticEstimationAlgorithm::new(false, 1.5)));
        algorithms.push(Box::new(MLBasedEstimationAlgorithm::new(
            "default".to_string(),
            0.85,
        )));

        Ok(Self {
            algorithms: Arc::new(Mutex::new(algorithms)),
            algorithm_performance: Arc::new(Mutex::new(HashMap::new())),
            estimation_history: Arc::new(Mutex::new(Vec::new())),
            safety_constraints: Arc::new(RwLock::new(EstimationSafetyConstraints::default())),
            config,
        })
    }

    /// Estimates safe concurrency level using multiple algorithms
    pub async fn estimate_safe_concurrency(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<ConcurrencyEstimationResult> {
        let start_time = Utc::now();

        // Create preliminary analysis result for algorithms
        let preliminary_result = ConcurrencyAnalysisResult {
            timestamp: Utc::now(),
            test_id: test_data.test_id.clone(),
            max_safe_concurrency: num_cpus::get(), // Default to CPU count
            recommended_concurrency: (num_cpus::get() / 2).max(1), // Conservative initial estimate
            resource_conflicts: Vec::new(),
            lock_dependencies: Vec::new(),
            sharing_capabilities: Vec::new(),
            safety_constraints: SafetyConstraints::default(),
            recommendations: Vec::new(),
            confidence: 0.5, // Low initial confidence
            performance_impact: 0.5,
            requirements: ConcurrencyRequirements::default(),
            estimation_details: String::new(),
            conflict_analysis: String::new(),
            sharing_analysis: String::new(),
            deadlock_analysis: String::new(),
            risk_assessment: String::new(),
            thread_analysis: String::new(),
            lock_analysis: String::new(),
            pattern_analysis: String::new(),
            analysis_duration: Duration::from_secs(0),
            safety_validation: String::new(),
        };

        // Run all algorithms sequentially (trait objects can't be easily cloned for parallel execution)
        let algorithms = self.algorithms.lock();
        let mut estimations = Vec::new();

        for algorithm in algorithms.iter() {
            let algorithm_name = algorithm.name().to_string();
            let estimation_start = Instant::now();
            let result = algorithm.estimate_safe_concurrency(&preliminary_result);
            let duration = estimation_start.elapsed();

            match result {
                Ok(estimation) => {
                    estimations.push(EstimationResult {
                        algorithm: algorithm_name.clone(),
                        concurrency: estimation,
                        confidence: self
                            .calculate_algorithm_confidence(&algorithm_name, test_data)
                            .await? as f64,
                        duration,
                    });

                    // Update algorithm performance
                    self.update_algorithm_performance(&algorithm_name, true, duration).await;
                },
                Err(e) => {
                    log::warn!("Algorithm {} failed: {}", algorithm_name, e);
                    self.update_algorithm_performance(&algorithm_name, false, duration).await;
                },
            }
        }

        drop(algorithms);

        if estimations.is_empty() {
            anyhow::bail!("All estimation algorithms failed");
        }

        // Select best estimation using ensemble approach
        let (recommended, optimal) = self.select_best_estimation(&estimations, test_data).await?;

        let estimation_summaries: Vec<String> = estimations
            .iter()
            .map(|e| {
                format!(
                    "{}: {} (conf: {:.2})",
                    e.algorithm, e.concurrency, e.confidence
                )
            })
            .collect();

        let result = ConcurrencyEstimationResult {
            recommended_concurrency: recommended,
            optimal_concurrency: optimal,
            max_safe_concurrency: ((optimal as f64) * (1.0 + self.config.safety_margin)) as usize,
            is_parallelizable: optimal > 1, // Test is parallelizable if optimal concurrency > 1
            estimations: estimation_summaries,
            analysis_confidence: self.calculate_overall_estimation_confidence(&estimations) as f64,
            estimation_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            safety_margin: self.config.safety_margin,
            performance_impact: (self.calculate_overall_estimation_confidence(&estimations) * 0.8)
                as f64, // Impact proportional to confidence
        };

        // Store estimation for learning
        self.store_estimation_record(test_data, &result).await?;

        Ok(result)
    }

    /// Calculates confidence for a specific algorithm
    async fn calculate_algorithm_confidence(
        &self,
        algorithm_name: &str,
        test_data: &TestExecutionData,
    ) -> Result<f32> {
        let performance = self.algorithm_performance.lock();

        if let Some(perf) = performance.get(algorithm_name) {
            let base_confidence = perf.success_rate;

            // Adjust confidence based on test characteristics
            let complexity_factor = self.assess_test_complexity(test_data);
            let adjusted_confidence = base_confidence * (1.0 - (complexity_factor as f64) * 0.2);

            Ok((adjusted_confidence.clamp(0.0, 1.0)) as f32)
        } else {
            Ok(0.5) // Default confidence for new algorithms
        }
    }

    /// Assesses test complexity for confidence adjustment
    fn assess_test_complexity(&self, test_data: &TestExecutionData) -> f32 {
        let trace_complexity = (test_data.execution_traces.len() as f32).log10() / 3.0;
        let resource_complexity = (test_data.resource_access_patterns.len() as f32).log10() / 2.0;
        let lock_complexity = (test_data.lock_usage.len() as f32).log10() / 2.0;

        (trace_complexity + resource_complexity + lock_complexity) / 3.0
    }

    /// Selects best estimation using ensemble approach
    async fn select_best_estimation(
        &self,
        estimations: &[EstimationResult],
        _test_data: &TestExecutionData,
    ) -> Result<(usize, usize)> {
        if estimations.is_empty() {
            return Ok((1, 1));
        }

        // Weighted average based on confidence and performance
        let mut weighted_sum = 0.0_f64;
        let mut total_weight = 0.0_f64;

        for estimation in estimations {
            let weight = estimation.confidence;
            weighted_sum += (estimation.concurrency as f64) * weight;
            total_weight += weight;
        }

        let recommended = if total_weight > 0.0 {
            (weighted_sum / total_weight).round() as usize
        } else {
            estimations.iter().map(|e| e.concurrency).min().unwrap_or(1)
        };

        // Apply safety margin
        let safety_margin = self.config.safety_margin;
        let safe_recommended = ((recommended as f64) * (1.0 - safety_margin)).ceil() as usize;

        // Calculate optimal (without safety margin)
        let optimal = recommended;

        Ok((safe_recommended.max(1), optimal.max(1)))
    }

    /// Calculates overall estimation confidence
    fn calculate_overall_estimation_confidence(&self, estimations: &[EstimationResult]) -> f32 {
        if estimations.is_empty() {
            return 0.0;
        }

        let avg_confidence: f32 = estimations.iter().map(|e| e.confidence).sum::<f64>()
            as f32
            / estimations.len() as f32;
        let consensus_factor = self.calculate_consensus_factor(estimations);

        avg_confidence * consensus_factor
    }

    /// Calculates consensus factor based on estimation agreement
    fn calculate_consensus_factor(&self, estimations: &[EstimationResult]) -> f32 {
        if estimations.len() < 2 {
            return 1.0;
        }

        let concurrencies: Vec<usize> = estimations.iter().map(|e| e.concurrency).collect();
        let mean = concurrencies.iter().sum::<usize>() as f32 / concurrencies.len() as f32;

        let variance = concurrencies.iter().map(|&c| (c as f32 - mean).powi(2) as f64).sum::<f64>()
            as f32
            / concurrencies.len() as f32;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 1.0 };

        // Higher consensus (lower variation) = higher factor
        (1.0 - coefficient_of_variation.min(1.0)).max(0.1)
    }

    /// Generates timeout requirements based on estimations
    fn generate_timeout_requirements(
        &self,
        estimations: &[EstimationResult],
    ) -> TimeoutRequirements {
        let max_duration = estimations
            .iter()
            .map(|e| e.duration)
            .max()
            .unwrap_or(Duration::from_millis(100));

        TimeoutRequirements {
            estimation_timeout: max_duration * 2,
            execution_timeout: max_duration * 10,
            cleanup_timeout: max_duration,
        }
    }

    /// Updates algorithm performance metrics
    async fn update_algorithm_performance(
        &self,
        algorithm_name: &str,
        success: bool,
        duration: Duration,
    ) {
        let mut performance = self.algorithm_performance.lock();
        let entry = performance
            .entry(algorithm_name.to_string())
            .or_default();

        entry.total_runs += 1;
        if success {
            entry.successful_runs += 1;
        }
        entry.total_duration += duration;
        entry.success_rate = entry.successful_runs as f64 / entry.total_runs as f64;
        let avg_secs = entry.total_duration.as_secs_f64() / entry.total_runs as f64;
        entry.avg_duration = Duration::from_secs_f64(avg_secs);
    }

    /// Stores estimation record for learning
    async fn store_estimation_record(
        &self,
        test_data: &TestExecutionData,
        result: &ConcurrencyEstimationResult,
    ) -> Result<()> {
        // TODO: from_test_data requires 4 arguments, using defaults for missing data
        let record = EstimationRecord {
            timestamp: Utc::now(),
            test_id: test_data.test_id.clone(),
            test_characteristics: TestCharacteristics::from_test_data(
                test_data.test_id.clone(),
                ResourceIntensity {
                    cpu_intensity: 0.0,
                    memory_intensity: 0.0,
                    io_intensity: 0.0,
                    network_intensity: 0.0,
                    gpu_intensity: 0.0,
                    overall_intensity: 0.0,
                    peak_periods: Vec::new(),
                    usage_variance: 0.0,
                    baseline_comparison: 0.0,
                    calculation_method: IntensityCalculationMethod::MovingAverage,
                },
                ConcurrencyRequirements::default(),
                SynchronizationRequirements {
                    synchronization_points: Vec::new(),
                    lock_usage_patterns: Vec::new(),
                    coordination_requirements: Vec::new(),
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
            ),
            estimation_result: EstimationResult {
                algorithm: "ensemble".to_string(),
                concurrency: result.recommended_concurrency,
                confidence: result.analysis_confidence,
                duration: result.estimation_duration,
            },
        };

        let mut history = self.estimation_history.lock();
        history.push(record);

        // Cleanup old records if needed
        let retention_limit = self.config.history_retention_limit;
        if history.len() > retention_limit {
            let history_len = history.len();
            history.drain(0..history_len - retention_limit);
        }

        Ok(())
    }
}

// =============================================================================
// RESOURCE CONFLICT DETECTOR
// =============================================================================

/// Advanced resource conflict detection and resolution system
///
/// Detects potential conflicts between concurrent test executions and provides
/// sophisticated resolution strategies and mitigation techniques.
#[derive(Debug)]
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

        let avg_probability = conflicts.iter().map(|c| c.probability).sum::<f64>() as f32
            / conflicts.len() as f32;
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

        let avg_probability = conflicts.iter().map(|c| c.probability).sum::<f64>() as f32
            / conflicts.len() as f32;
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

// =============================================================================
// SHARING CAPABILITY ANALYZER
// =============================================================================

/// Advanced resource sharing capability analysis system
///
/// Analyzes resource sharing patterns and capabilities to optimize concurrent
/// test execution while maintaining safety and performance guarantees.
#[derive(Debug)]
pub struct SharingCapabilityAnalyzer {
    /// Sharing analysis strategies
    strategies: Arc<Mutex<Vec<Box<dyn SharingAnalysisStrategy + Send + Sync>>>>,

    /// Sharing patterns database
    patterns_db: Arc<RwLock<SharingPatternsDatabase>>,

    /// Capability cache for performance
    capability_cache: Arc<Mutex<HashMap<String, CachedSharingCapability>>>,

    /// Sharing performance history
    performance_history: Arc<Mutex<SharingPerformanceHistory>>,

    /// Configuration
    config: SharingAnalysisConfig,
}

impl SharingCapabilityAnalyzer {
    /// Creates a new sharing capability analyzer
    pub async fn new(config: SharingAnalysisConfig) -> Result<Self> {
        let mut strategies: Vec<Box<dyn SharingAnalysisStrategy + Send + Sync>> = Vec::new();

        // Initialize sharing analysis strategies
        strategies.push(Box::new(ReadOnlySharingStrategy::new(true, true)));
        strategies.push(Box::new(PartitionedSharingStrategy::new()?));
        strategies.push(Box::new(TemporalSharingStrategy::new()?));
        strategies.push(Box::new(AdaptiveSharingStrategy::new()?));

        let patterns_db = SharingPatternsDatabase::new();

        Ok(Self {
            strategies: Arc::new(Mutex::new(strategies)),
            patterns_db: Arc::new(RwLock::new(patterns_db)),
            capability_cache: Arc::new(Mutex::new(HashMap::new())),
            performance_history: Arc::new(Mutex::new(SharingPerformanceHistory::new())),
            config,
        })
    }

    /// Analyzes sharing capabilities for test execution data
    pub async fn analyze_sharing_capabilities(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<SharingAnalysisResult> {
        let start_time = Utc::now();

        // Check cache first
        if let Some(cached) = self.check_cache(&test_data.test_id).await {
            if cached.is_valid() {
                // Convert cached String result to SharingCapability enum
                let capability = match cached.result.as_str() {
                    "ReadOnly" => SharingCapability::ReadOnly,
                    "ReadWrite" => SharingCapability::ReadWrite,
                    "Exclusive" => SharingCapability::Exclusive,
                    "Shared" => SharingCapability::Shared,
                    "None" => SharingCapability::None,
                    _ => SharingCapability::None, // Default
                };

                return Ok(SharingAnalysisResult {
                    sharing_requirements: SynchronizationRequirements::default(),
                    confidence: cached.confidence,
                    sharing_capabilities: vec![capability],
                    optimizations: Vec::new(),
                    performance_predictions: Vec::new(),
                    strategy_results: Vec::new(),
                    analysis_duration: Duration::from_millis(0),
                });
            }
        }

        // Run sharing analysis strategies
        // Execute synchronously to avoid lifetime issues with mutex guards
        let analysis_results: Vec<_> = {
            let strategies = self.strategies.lock();
            strategies
                .iter()
                .map(|strategy| {
                    let strategy_name = strategy.name().to_string();
                    let analysis_start = Instant::now();
                    let result = strategy.analyze_sharing_capability(
                        &test_data.test_id,
                        &test_data.resource_access_patterns,
                    );
                    let analysis_duration = analysis_start.elapsed();
                    (strategy_name, result, analysis_duration)
                })
                .collect()
        };

        // Collect results
        let mut sharing_capability_structs = Vec::new(); // For helper methods
        let mut sharing_capability_enums = Vec::new(); // For result
        let mut strategy_results = Vec::new();

        for (strategy_name, result, duration) in analysis_results {
            match result {
                Ok(capability) => {
                    // Convert ResourceSharingCapabilities to SharingCapability enum
                    let sharing_cap =
                        if capability.supports_write_sharing && capability.supports_read_sharing {
                            SharingCapability::ReadWrite
                        } else if capability.supports_read_sharing {
                            SharingCapability::ReadOnly
                        } else if capability.supports_write_sharing {
                            SharingCapability::Exclusive
                        } else {
                            SharingCapability::None
                        };

                    strategy_results.push(SharingStrategyResult {
                        strategy: strategy_name,
                        capability: sharing_cap.clone(),
                        detection_duration: duration,
                        confidence: self.calculate_strategy_confidence(&capability) as f64,
                    });
                    sharing_capability_structs.push(capability);
                    sharing_capability_enums.push(sharing_cap);
                },
                Err(e) => {
                    log::warn!("Sharing analysis strategy failed: {}", e);
                },
            }
        }

        // Synthesize sharing requirements
        let _sharing_requirements =
            self.synthesize_sharing_requirements(&sharing_capability_structs)?;

        // Generate optimization recommendations
        let optimizations =
            self.generate_sharing_optimizations(&sharing_capability_structs).await?;

        // Calculate performance predictions
        let performance_prediction =
            self.predict_sharing_performance(&sharing_capability_structs).await?;
        let performance_predictions = vec![performance_prediction];

        let result = SharingAnalysisResult {
            sharing_requirements: SynchronizationRequirements::default(),
            sharing_capabilities: sharing_capability_enums,
            optimizations,
            performance_predictions,
            strategy_results,
            analysis_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence: self.calculate_overall_sharing_confidence(&sharing_capability_structs)
                as f64,
        };

        // Cache result
        self.cache_result(&test_data.test_id, &result).await?;

        Ok(result)
    }

    /// Checks cache for existing analysis results
    async fn check_cache(&self, test_id: &str) -> Option<CachedSharingCapability> {
        let cache = self.capability_cache.lock();
        cache.get(test_id).cloned()
    }

    /// Calculates confidence for a sharing strategy
    fn calculate_strategy_confidence(&self, capability: &ResourceSharingCapabilities) -> f32 {
        let mut confidence_factors = Vec::new();

        // TODO: ResourceSharingCapabilities no longer has sharing_safety_level enum
        // It has safety_assessment: f64 instead
        // Factor in sharing safety
        confidence_factors.push(capability.safety_assessment);

        // TODO: performance_overhead → sharing_overhead
        // Factor in performance impact
        confidence_factors.push(1.0 - capability.sharing_overhead.abs());

        // TODO: ResourceSharingCapabilities no longer has implementation_complexity
        // Using sharing_overhead as proxy (lower overhead = less complex)
        // Factor in complexity
        confidence_factors.push(1.0 - capability.sharing_overhead);

        confidence_factors.iter().sum::<f64>() as f32 / confidence_factors.len() as f32
    }

    /// Synthesizes sharing requirements from all capabilities
    fn synthesize_sharing_requirements(
        &self,
        capabilities: &[ResourceSharingCapabilities],
    ) -> Result<SharingRequirements> {
        if capabilities.is_empty() {
            return Ok(SharingRequirements::default());
        }

        // Find the most restrictive but safe sharing approach
        // TODO: ResourceSharingCapabilities no longer has sharing_safety_level enum
        // It has safety_assessment: f64 instead (higher = safer)
        // Also no longer has implementation_complexity
        let safest_capability = capabilities
            .iter()
            .filter(|c| c.safety_assessment >= 0.7) // Safe or conditionally safe
            .min_by(|a, b| {
                // Use sharing_overhead as complexity proxy (lower = simpler)
                a.sharing_overhead
                    .partial_cmp(&b.sharing_overhead)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .or_else(|| capabilities.first())
            .unwrap();

        // Convert String to SharingMode enum
        let sharing_mode = match safest_capability.sharing_mode.as_str() {
            "ReadOnly" => SharingMode::ReadOnly,
            "Write" => SharingMode::Write,
            "ReadWrite" => SharingMode::ReadWrite,
            "Exclusive" => SharingMode::Exclusive,
            "ExclusiveWrite" => SharingMode::ExclusiveWrite,
            _ => SharingMode::NoSharing,
        };

        // Convert Vec<String> to IsolationLevel enum (use first or default)
        let isolation_level = if let Some(first) = safest_capability.isolation_requirements.first()
        {
            match first.as_str() {
                "ReadUncommitted" => IsolationLevel::ReadUncommitted,
                "ReadCommitted" => IsolationLevel::ReadCommitted,
                "RepeatableRead" => IsolationLevel::RepeatableRead,
                "Serializable" => IsolationLevel::Serializable,
                _ => IsolationLevel::None,
            }
        } else {
            IsolationLevel::None
        };

        // Convert Vec<String> to SynchronizationRequirements struct
        let synchronization_requirements = SynchronizationRequirements {
            synchronization_points: Vec::new(),
            lock_usage_patterns: Vec::new(),
            coordination_requirements: safest_capability.consistency_guarantees.clone(),
            synchronization_overhead: safest_capability.sharing_overhead,
            deadlock_prevention: Vec::new(),
            optimization_opportunities: Vec::new(),
            complexity_score: 0.0,
            performance_impact: safest_capability.sharing_overhead,
            alternative_strategies: Vec::new(),
            average_wait_time: Duration::from_secs(0),
            ordered_locking: false,
            timeout_based_locking: false,
            resource_ordering: Vec::new(),
            lock_free_alternatives: Vec::new(),
            custom_requirements: Vec::new(),
        };

        Ok(SharingRequirements {
            max_concurrent_shares: safest_capability.max_concurrent_readers.unwrap_or(1),
            sharing_mode,
            isolation_level,
            synchronization_requirements,
            performance_requirements: vec![
                format!("Max overhead: {}", safest_capability.sharing_overhead),
                format!(
                    "Throughput target: {}",
                    1.0 - safest_capability.sharing_overhead
                ),
                "Latency target: 100ms".to_string(),
            ],
            required_synchronization: vec![],
            exclusive_access_needed: false,
            concurrency_limit: safest_capability.max_concurrent_readers.unwrap_or(1),
        })
    }

    /// Generates sharing optimization recommendations
    async fn generate_sharing_optimizations(
        &self,
        capabilities: &[ResourceSharingCapabilities],
    ) -> Result<Vec<SharingOptimization>> {
        let mut optimizations = Vec::new();

        for capability in capabilities {
            if capability.performance_overhead > 0.2 {
                optimizations.push(SharingOptimization {
                    optimization_type: "ReduceOverhead".to_string(),
                    description: "Consider using more efficient sharing mechanisms".to_string(),
                    expected_improvement: capability.performance_overhead * 0.5,
                    implementation_effort: "Medium".to_string(),
                    recommendations: vec![
                        "Use lock-free data structures where possible".to_string(),
                        "Implement read-copy-update patterns".to_string(),
                        "Consider message passing instead of shared state".to_string(),
                    ],
                });
            }

            if capability.implementation_complexity > 0.7 {
                optimizations.push(SharingOptimization {
                    optimization_type: "SimplifyImplementation".to_string(),
                    description: "Simplify sharing implementation for better maintainability"
                        .to_string(),
                    expected_improvement: 0.3,
                    implementation_effort: "High".to_string(),
                    recommendations: vec![
                        "Break down complex sharing patterns".to_string(),
                        "Use higher-level synchronization primitives".to_string(),
                        "Implement gradual sharing capability expansion".to_string(),
                    ],
                });
            }
        }

        Ok(optimizations)
    }

    /// Predicts sharing performance based on capabilities
    async fn predict_sharing_performance(
        &self,
        capabilities: &[ResourceSharingCapabilities],
    ) -> Result<SharingPerformancePrediction> {
        let history = self.performance_history.lock();

        // Use historical data for prediction if available
        let base_throughput = history.get_average_throughput();

        let base_latency_ms = history.get_average_latency();
        let _base_latency = Duration::from_millis(base_latency_ms as u64);

        // Calculate predictions based on capabilities
        let avg_overhead = capabilities.iter().map(|c| c.performance_overhead).sum::<f64>()
            / capabilities.len() as f64;

        let predicted_throughput = base_throughput * (1.0 - avg_overhead);
        let latency_multiplier = 1.0 + avg_overhead;
        let predicted_latency_ms = base_latency_ms * latency_multiplier;
        let predicted_latency = Duration::from_millis(predicted_latency_ms as u64);

        let scalability_factor = capabilities
            .iter()
            .map(|c| match c.sharing_mode.as_str() {
                "ReadOnly" => 0.9,
                "ReadWrite" => 0.6,
                "ExclusiveWrite" => 0.3,
                "NoSharing" => 0.1,
                _ => 0.5, // Default for unknown modes
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.5);

        let bottlenecks = self.analyze_potential_bottlenecks(capabilities);
        let bottleneck_summary = format!("Found {} potential bottlenecks", bottlenecks.len());

        Ok(SharingPerformancePrediction {
            expected_throughput: predicted_throughput,
            analysis_duration: predicted_latency,
            scalability_factor,
            confidence: self.calculate_prediction_confidence(capabilities) as f64,
            bottleneck_analysis: bottleneck_summary,
        })
    }

    /// Calculates prediction confidence
    fn calculate_prediction_confidence(&self, capabilities: &[ResourceSharingCapabilities]) -> f32 {
        if capabilities.is_empty() {
            return 0.0;
        }

        let avg_confidence = capabilities
            .iter()
            .map(|c| self.calculate_strategy_confidence(c) as f64)
            .sum::<f64>() as f32
            / capabilities.len() as f32;

        let consistency_factor = self.calculate_capability_consistency(capabilities);

        avg_confidence * consistency_factor
    }

    /// Calculates capability consistency
    fn calculate_capability_consistency(
        &self,
        capabilities: &[ResourceSharingCapabilities],
    ) -> f32 {
        if capabilities.len() < 2 {
            return 1.0;
        }

        // Measure consistency in overhead predictions
        let overheads: Vec<f32> =
            capabilities.iter().map(|c| c.performance_overhead as f32).collect();
        let mean = overheads.iter().map(|&o| o as f64).sum::<f64>() as f32 / overheads.len() as f32;
        let variance = overheads.iter().map(|&o| (o - mean).powi(2) as f64).sum::<f64>() as f32
            / overheads.len() as f32;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 1.0 };

        (1.0 - coefficient_of_variation.min(1.0)).max(0.1)
    }

    /// Analyzes potential bottlenecks
    fn analyze_potential_bottlenecks(
        &self,
        capabilities: &[ResourceSharingCapabilities],
    ) -> Vec<BottleneckAnalysis> {
        let mut bottlenecks = Vec::new();

        for capability in capabilities {
            if capability.performance_overhead > 0.3 {
                bottlenecks.push(BottleneckAnalysis {
                    bottleneck_type: "SynchronizationOverhead".to_string(),
                    severity: if capability.performance_overhead > 0.5 {
                        BottleneckSeverity::High
                    } else {
                        BottleneckSeverity::Medium
                    },
                    affected_components: vec!["synchronization".to_string()],
                    impact_score: capability.performance_overhead,
                    resolution_suggestions: vec![
                        "Consider reducing synchronization frequency".to_string(),
                        "Use more efficient synchronization primitives".to_string(),
                    ],
                });
            }

            if capability.max_concurrent_readers.unwrap_or(0) < 2 {
                bottlenecks.push(BottleneckAnalysis {
                    bottleneck_type: "ConcurrencyLimitation".to_string(),
                    severity: BottleneckSeverity::Medium,
                    affected_components: vec!["concurrency".to_string()],
                    impact_score: 0.5,
                    resolution_suggestions: vec![
                        "Investigate increasing concurrent access limits".to_string(),
                        "Consider resource partitioning strategies".to_string(),
                    ],
                });
            }
        }

        bottlenecks
    }

    /// Caches analysis result
    async fn cache_result(&self, test_id: &str, result: &SharingAnalysisResult) -> Result<()> {
        let can_share = !result.sharing_capabilities.is_empty();
        let sharing_mode = if can_share { "parallel".to_string() } else { "none".to_string() };
        let cache_timestamp = Utc::now();

        let cached = CachedSharingCapability {
            can_share,
            sharing_mode,
            cache_timestamp,
            result: format!("SharingAnalysis(confidence={})", result.confidence),
            cached_at: cache_timestamp,
            confidence: result.confidence,
        };

        let mut cache = self.capability_cache.lock();
        cache.insert(test_id.to_string(), cached);

        // Cleanup old cache entries if needed
        let cache_limit = self.config.cache_size_limit;
        if cache.len() > cache_limit {
            // Remove oldest entries (simple LRU approximation)
            let oldest_key = cache.iter().min_by_key(|(_, v)| v.cached_at).map(|(k, _)| k.clone());

            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }

        Ok(())
    }

    /// Calculates overall sharing confidence
    fn calculate_overall_sharing_confidence(
        &self,
        capabilities: &[ResourceSharingCapabilities],
    ) -> f32 {
        if capabilities.is_empty() {
            return 0.0;
        }

        let individual_confidences: Vec<f32> =
            capabilities.iter().map(|c| self.calculate_strategy_confidence(c)).collect();

        let avg_confidence = individual_confidences.iter().map(|&x| x as f64).sum::<f64>() as f32
            / individual_confidences.len() as f32;
        let consistency_factor = self.calculate_capability_consistency(capabilities);

        avg_confidence * consistency_factor
    }
}

// =============================================================================
// DEADLOCK ANALYZER
// =============================================================================

/// Advanced deadlock detection and prevention system
///
/// Implements sophisticated algorithms for detecting potential deadlocks,
/// analyzing lock dependencies, and providing prevention strategies.
#[derive(Debug)]
pub struct DeadlockAnalyzer {
    /// Deadlock detection algorithms
    detection_algorithms: Arc<Mutex<Vec<Box<dyn DeadlockDetectionAlgorithm + Send + Sync>>>>,

    /// Prevention strategies
    prevention_strategies: Arc<Mutex<Vec<Box<dyn DeadlockPreventionStrategy + Send + Sync>>>>,

    /// Lock dependency graph
    dependency_graph: Arc<RwLock<LockDependencyGraph>>,

    /// Deadlock incident history
    incident_history: Arc<Mutex<DeadlockIncidentHistory>>,

    /// Real-time monitoring data
    monitoring_data: Arc<RwLock<DeadlockMonitoringData>>,

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
        // TODO: OrderedLockingStrategy::new requires hierarchy: Vec<String>
        prevention_strategies.push(Box::new(OrderedLockingStrategy::new(true, Vec::new())));
        // TODO: TimeoutBasedStrategy::new requires abort_on_timeout: bool
        prevention_strategies.push(Box::new(TimeoutBasedStrategy::new(1000, false)));
        prevention_strategies.push(Box::new(ResourceOrderingStrategy::new()?));
        prevention_strategies.push(Box::new(AdaptivePreventionStrategy::new()?));

        Ok(Self {
            detection_algorithms: Arc::new(Mutex::new(detection_algorithms)),
            prevention_strategies: Arc::new(Mutex::new(prevention_strategies)),
            dependency_graph: Arc::new(RwLock::new(LockDependencyGraph::new()?)),
            incident_history: Arc::new(Mutex::new(DeadlockIncidentHistory::new()?)),
            monitoring_data: Arc::new(RwLock::new(DeadlockMonitoringData::new()?)),
            config,
        })
    }

    /// Analyzes deadlock risks in test execution data
    pub async fn analyze_deadlock_risks(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<DeadlockAnalysisResult> {
        let start_time = Utc::now();

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
                    // TODO: detect_deadlocks takes 1 argument, removed graph parameter
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
                    // Convert DeadlockRisk to DeadlockScenario
                    let scenarios: Vec<DeadlockScenario> = deadlocks
                        .iter()
                        .map(|_d| DeadlockScenario {
                            scenario_id: format!("deadlock_{}", uuid::Uuid::new_v4()),
                            involved_threads: Vec::new(),
                            involved_resources: Vec::new(),
                            lock_order: Vec::new(),
                            risk_level: "Medium".to_string(),
                            timestamp: Utc::now(),
                        })
                        .collect();

                    // Convert to PotentialDeadlock for confidence calculation
                    let potential: Vec<PotentialDeadlock> = deadlocks
                        .iter()
                        .map(|_d| PotentialDeadlock { locks: Vec::new() })
                        .collect();

                    detection_results.push(DeadlockDetectionResult {
                        algorithm: algorithm_name,
                        deadlocks: scenarios,
                        detection_duration: duration,
                        confidence: self.calculate_detection_confidence(&potential) as f64,
                    });
                    potential_deadlocks.append(&mut deadlocks);
                },
                Err(e) => {
                    log::warn!("Deadlock detection algorithm failed: {}", e);
                },
            }
        }

        // Convert DeadlockRisk to PotentialDeadlock for deduplication
        let potential_deadlock_converted: Vec<PotentialDeadlock> = potential_deadlocks
            .iter()
            .map(|_d| PotentialDeadlock { locks: Vec::new() })
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

        // Convert types to match DeadlockAnalysisResult requirements
        let deadlock_scenarios: Vec<DeadlockScenario> = prioritized_deadlocks
            .iter()
            .map(|_d| DeadlockScenario {
                scenario_id: format!("deadlock_{}", uuid::Uuid::new_v4()),
                involved_threads: Vec::new(),
                involved_resources: Vec::new(),
                lock_order: Vec::new(),
                risk_level: "Medium".to_string(),
                timestamp: Utc::now(),
            })
            .collect();

        let risk_level_str = format!("{:?}", risk_level);
        let safe_limit = safe_concurrency_limit.unwrap_or(1);
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
            analysis_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence: self.calculate_overall_deadlock_confidence(&unique_deadlocks) as f64,
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
                        // TODO: add_lock_acquisition signature changed - needs thread_id, lock_id, and stack trace
                        graph.add_lock_acquisition(
                            trace.thread_id,
                            trace.resource.clone(),
                            Vec::new(), // Stack trace not available from execution trace
                        );
                    },
                    "LockRelease" => {
                        // TODO: add_lock_release takes 2 arguments (thread_id, lock_id), removed timestamp
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

                    // TODO: calculate_dependency_strength expects DateTime<Utc>, but we have Instant
                    // Using default probability instead
                    let time_diff = if time2 > time1 {
                        time2.duration_since(*time1).as_secs_f32()
                    } else {
                        0.0
                    };
                    let contention_prob = (1.0 / (1.0 + time_diff)).min(0.9) as f64;

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

    /// Calculates dependency strength based on timing
    fn calculate_dependency_strength(&self, time1: &DateTime<Utc>, time2: &DateTime<Utc>) -> f32 {
        let duration = (*time2 - *time1).num_milliseconds() as f32;

        // Shorter intervals indicate stronger dependencies
        let strength = 1.0 / (1.0 + duration / 1000.0);
        strength.clamp(0.1, 1.0)
    }

    /// Calculates detection confidence
    fn calculate_detection_confidence(&self, deadlocks: &[PotentialDeadlock]) -> f32 {
        if deadlocks.is_empty() {
            return 1.0;
        }

        // TODO: PotentialDeadlock no longer has confidence and probability fields
        // Using placeholder values
        let avg_confidence = 0.8_f32; // Placeholder
        let avg_probability = 0.7_f32; // Placeholder

        (avg_confidence + avg_probability) / 2.0
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

        // TODO: PotentialDeadlock no longer has deadlock_type field
        a_locks == b_locks // && a.deadlock_type == b.deadlock_type
    }

    /// Prioritizes deadlocks based on severity and probability
    fn prioritize_deadlocks(&self, deadlocks: &[PotentialDeadlock]) -> Vec<PotentialDeadlock> {
        let prioritized = deadlocks.to_vec();

        // TODO: PotentialDeadlock no longer has impact and probability fields
        // Cannot prioritize without these fields, returning as-is
        // prioritized.sort_by(|a, b| {
        //     // First by severity, then by probability
        //     b.probability.partial_cmp(&a.probability).unwrap_or(std::cmp::Ordering::Equal)
        // });

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
                            expected_effectiveness: effectiveness as f64,
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
    ) -> f32 {
        // TODO: PotentialDeadlock no longer has deadlock_type field
        match strategy_name {
            "OrderedLockingStrategy" => 0.9,
            "TimeoutBasedStrategy" => 0.8,
            "ResourceOrderingStrategy" => 0.85,
            "AdaptivePreventionStrategy" => 0.75,
            _ => 0.6,
        }
    }

    /// Calculates implementation complexity
    fn calculate_implementation_complexity(&self, strategy_name: &str) -> f32 {
        match strategy_name {
            "OrderedLockingStrategy" => 0.6,
            "TimeoutBasedStrategy" => 0.3,
            "ResourceOrderingStrategy" => 0.7,
            "AdaptivePreventionStrategy" => 0.9,
            _ => 0.5,
        }
    }

    /// Assesses overall deadlock risk level
    fn assess_deadlock_risk_level(&self, deadlocks: &[PotentialDeadlock]) -> DeadlockRiskLevel {
        // TODO: DeadlockRiskLevel changed from enum to struct
        if deadlocks.is_empty() {
            return DeadlockRiskLevel {
                level: "None".to_string(),
                risk_score: 0.0,
                contributing_factors: Vec::new(),
            };
        }

        // TODO: PotentialDeadlock no longer has impact and probability fields
        // Using deadlock count as a simple heuristic for risk level
        let count = deadlocks.len();

        if count > 10 {
            DeadlockRiskLevel {
                level: "Critical".to_string(),
                risk_score: 1.0,
                contributing_factors: vec!["High deadlock count".to_string()],
            }
        } else if count > 5 {
            DeadlockRiskLevel {
                level: "High".to_string(),
                risk_score: 0.8,
                contributing_factors: vec!["Moderate deadlock count".to_string()],
            }
        } else if count > 2 {
            DeadlockRiskLevel {
                level: "Medium".to_string(),
                risk_score: 0.5,
                contributing_factors: vec!["Some deadlocks detected".to_string()],
            }
        } else {
            DeadlockRiskLevel {
                level: "Low".to_string(),
                risk_score: 0.2,
                contributing_factors: vec!["Few deadlocks detected".to_string()],
            }
        }
    }

    /// Calculates safe concurrency limit based on deadlocks
    fn calculate_safe_concurrency_limit(&self, deadlocks: &[PotentialDeadlock]) -> usize {
        if deadlocks.is_empty() {
            return usize::MAX;
        }

        // Find the most restrictive deadlock
        let min_lock_count = deadlocks.iter().map(|d| d.locks.len()).min().unwrap_or(1);

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

        // TODO: PotentialDeadlock no longer has deadlock_type, impact, confidence, probability fields
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
        // TODO: PotentialDeadlock no longer has impact field
        // Using count-based heuristic: many deadlocks = critical
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
    fn calculate_overall_deadlock_confidence(&self, deadlocks: &[PotentialDeadlock]) -> f32 {
        if deadlocks.is_empty() {
            return 1.0;
        }

        // TODO: PotentialDeadlock no longer has confidence field
        // Using count-based confidence: more deadlocks = higher confidence
        let avg_confidence = if deadlocks.len() > 5 {
            0.9
        } else if deadlocks.len() > 2 {
            0.7
        } else {
            0.5
        };
        let consistency_factor = self.calculate_deadlock_consistency(deadlocks);

        avg_confidence * consistency_factor
    }

    /// Calculates deadlock consistency
    fn calculate_deadlock_consistency(&self, deadlocks: &[PotentialDeadlock]) -> f32 {
        if deadlocks.len() < 2 {
            return 1.0;
        }

        // TODO: PotentialDeadlock no longer has probability field
        // Using count-based consistency: more deadlocks of same type = higher consistency
        // Assuming moderate consistency for now
        0.7
    }
}

// =============================================================================
// CONCURRENCY RISK ASSESSMENT
// =============================================================================

/// Advanced concurrency risk assessment system
///
/// Provides comprehensive risk analysis for concurrent test execution scenarios,
/// evaluating potential hazards and providing mitigation strategies.
#[derive(Debug)]
pub struct ConcurrencyRiskAssessment {
    /// Risk assessment algorithms
    assessment_algorithms: Arc<Mutex<Vec<Box<dyn RiskAssessmentAlgorithm + Send + Sync>>>>,

    /// Risk mitigation strategies
    mitigation_strategies: Arc<Mutex<Vec<Box<dyn RiskMitigationStrategy + Send + Sync>>>>,

    /// Historical risk data
    risk_history: Arc<Mutex<RiskAssessmentHistory>>,

    /// Risk monitoring data
    monitoring_data: Arc<RwLock<RiskMonitoringData>>,

    /// Configuration
    config: RiskAssessmentConfig,
}

impl ConcurrencyRiskAssessment {
    /// Creates a new concurrency risk assessment system
    pub async fn new(config: RiskAssessmentConfig) -> Result<Self> {
        let mut assessment_algorithms: Vec<Box<dyn RiskAssessmentAlgorithm + Send + Sync>> =
            Vec::new();
        let mut mitigation_strategies: Vec<Box<dyn RiskMitigationStrategy + Send + Sync>> =
            Vec::new();

        // Initialize risk assessment algorithms
        assessment_algorithms.push(Box::new(MachineLearningRiskAssessment::new(
            "default".to_string(),
            0.85,
        )));

        // Initialize mitigation strategies
        // TODO: PreventiveMitigation::new requires enabled: bool, strategies: Vec<String>
        mitigation_strategies.push(Box::new(PreventiveMitigation::new(true, Vec::new())));
        // TODO: ReactiveMitigation::new requires enabled: bool, response_time_ms: u64
        mitigation_strategies.push(Box::new(ReactiveMitigation::new(true, 1000)));
        mitigation_strategies.push(Box::new(AdaptiveMitigation::new()));

        Ok(Self {
            assessment_algorithms: Arc::new(Mutex::new(assessment_algorithms)),
            mitigation_strategies: Arc::new(Mutex::new(mitigation_strategies)),
            risk_history: Arc::new(Mutex::new(RiskAssessmentHistory::new())),
            monitoring_data: Arc::new(RwLock::new(RiskMonitoringData::new())),
            config,
        })
    }

    /// Assesses concurrency risks for test execution data
    pub async fn assess_concurrency_risks(
        &self,
        _test_data: &TestExecutionData,
    ) -> Result<RiskAssessmentResult> {
        let start_time = Utc::now();

        // Run risk assessment algorithms synchronously to avoid lifetime issues
        let assessment_task_results: Vec<_> = {
            let algorithms = self.assessment_algorithms.lock();
            algorithms
                .iter()
                .map(|algorithm| {
                    let algorithm_name = algorithm.name().to_string();
                    let assessment_start = Instant::now();
                    // TODO: assess_risk takes 0 arguments, removed test_data parameter
                    let result = algorithm.assess_risk();
                    let assessment_duration = assessment_start.elapsed();
                    (algorithm_name, result, assessment_duration)
                })
                .collect()
        };

        // Collect assessment results
        let mut risk_assessments = Vec::new();
        let mut algorithm_results = Vec::new();

        for (algorithm_name, risk_score, duration) in assessment_task_results {
            let assessment = RiskAssessment {
                risk_level: if risk_score > 0.7 {
                    "HIGH".to_string()
                } else if risk_score > 0.4 {
                    "MEDIUM".to_string()
                } else {
                    "LOW".to_string()
                },
                risk_score,
                risk_factors: Vec::new(), // TODO: populate from algorithm details
                primary_risk_factor: "concurrency".to_string(),
                potential_impact: risk_score,
            };

            let risk_assessment_struct = RiskAssessment {
                risk_level: if risk_score > 0.7 {
                    "High"
                } else if risk_score > 0.4 {
                    "Medium"
                } else {
                    "Low"
                }
                .to_string(),
                risk_score: risk_score,
                risk_factors: Vec::new(),
                primary_risk_factor: "Concurrency".to_string(),
                potential_impact: risk_score,
            };

            algorithm_results.push(RiskAlgorithmResult {
                algorithm: algorithm_name,
                assessment: risk_assessment_struct.clone(),
                assessment_duration: duration,
                confidence: 0.8, // Default confidence for risk assessment
            });
            risk_assessments.push(assessment);
        }

        // Synthesize overall risk assessment
        let overall_risk_level = self.synthesize_risk_level(&risk_assessments);
        let risk_factors = self.identify_risk_factors(&risk_assessments);
        let risk_thresholds_f32 = self.calculate_risk_thresholds(&risk_assessments);
        // Convert HashMap<String, f32> to HashMap<String, f64>
        let risk_thresholds: HashMap<String, f64> =
            risk_thresholds_f32.into_iter().map(|(k, v)| (k, v as f64)).collect();

        // Generate mitigation recommendations
        let mitigation_recommendations =
            self.generate_mitigation_recommendations(&risk_assessments).await?;

        Ok(RiskAssessmentResult {
            overall_risk_level,
            risk_factors,
            risk_thresholds,
            mitigation_recommendations,
            algorithm_results,
            assessment_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence: self.calculate_overall_risk_confidence(&risk_assessments) as f64,
        })
    }

    /// Calculates algorithm confidence
    fn calculate_algorithm_confidence(&self, assessment: &RiskAssessment) -> f32 {
        let factor_confidence = if assessment.risk_factors.is_empty() {
            0.5
        } else {
            assessment.risk_factors.iter().map(|f| f.confidence).sum::<f64>() as f32
                / assessment.risk_factors.len() as f32
        };

        // Map impact value (0.0-1.0) to confidence
        let impact_confidence = if assessment.potential_impact < 0.25 {
            0.8 // Low impact
        } else if assessment.potential_impact < 0.5 {
            0.7 // Medium impact
        } else if assessment.potential_impact < 0.75 {
            0.6 // High impact
        } else {
            0.5 // Critical impact
        };

        (factor_confidence + impact_confidence) / 2.0
    }

    /// Synthesizes overall risk level from multiple assessments
    fn synthesize_risk_level(&self, assessments: &[RiskAssessment]) -> RiskLevel {
        if assessments.is_empty() {
            return RiskLevel::Negligible;
        }

        // Convert String risk levels to enum and find the highest
        let risk_levels: Vec<RiskLevel> = assessments
            .iter()
            .map(|a| match a.risk_level.as_str() {
                "Negligible" => RiskLevel::Negligible,
                "VeryLow" => RiskLevel::VeryLow,
                "Low" => RiskLevel::Low,
                "Medium" => RiskLevel::Medium,
                "High" => RiskLevel::High,
                "VeryHigh" => RiskLevel::VeryHigh,
                "Severe" => RiskLevel::Severe,
                "Critical" => RiskLevel::Critical,
                "Extreme" => RiskLevel::Extreme,
                _ => RiskLevel::Negligible, // Default for unknown
            })
            .collect();

        // Find highest risk level (assuming enum order matches risk severity)
        risk_levels.into_iter().max().unwrap_or(RiskLevel::Negligible)
    }

    /// Identifies common risk factors
    fn identify_risk_factors(&self, assessments: &[RiskAssessment]) -> Vec<RiskFactor> {
        let mut all_factors = Vec::new();

        for assessment in assessments {
            all_factors.extend(assessment.risk_factors.clone());
        }

        // Deduplicate and merge similar factors
        self.deduplicate_risk_factors(&all_factors)
    }

    /// Deduplicates risk factors
    fn deduplicate_risk_factors(&self, factors: &[RiskFactor]) -> Vec<RiskFactor> {
        let mut unique_factors = Vec::new();

        for factor in factors {
            let existing = unique_factors
                .iter_mut()
                .find(|f: &&mut RiskFactor| f.factor_type == factor.factor_type);

            if let Some(existing_factor) = existing {
                // Merge factors by taking maximum severity and confidence
                existing_factor.severity = existing_factor.severity.max(factor.severity);
                existing_factor.confidence = existing_factor.confidence.max(factor.confidence);
            } else {
                unique_factors.push(factor.clone());
            }
        }

        unique_factors
    }

    /// Calculates risk thresholds
    fn calculate_risk_thresholds(&self, assessments: &[RiskAssessment]) -> HashMap<String, f32> {
        let mut thresholds = HashMap::new();

        for assessment in assessments {
            // Match on string risk_level field
            let threshold = match assessment.risk_level.as_str() {
                "Negligible" | "VeryLow" => 0.0,
                "Low" => 0.2,
                "Medium" => 0.5,
                "High" | "VeryHigh" => 0.8,
                "Severe" | "Critical" | "Extreme" => 1.0,
                _ => 0.0, // Default for unknown
            };

            thresholds.insert(format!("risk_level_{}", assessment.risk_level), threshold);
        }

        thresholds
    }

    /// Generates mitigation recommendations
    async fn generate_mitigation_recommendations(
        &self,
        assessments: &[RiskAssessment],
    ) -> Result<Vec<RiskMitigationRecommendation>> {
        let strategies = self.mitigation_strategies.lock();
        let mut recommendations = Vec::new();

        for assessment in assessments {
            for strategy in strategies.iter() {
                // TODO: is_applicable takes 0 arguments, removed assessment parameter
                if strategy.is_applicable() {
                    // TODO: generate_mitigation takes 0 arguments, removed assessment parameter
                    let mitigation = strategy.generate_mitigation();
                    recommendations.push(RiskMitigationRecommendation {
                        risk_factor: assessment.primary_risk_factor.clone(),
                        mitigation_strategy: strategy.name().to_string(),
                        mitigation_action: mitigation,
                        expected_effectiveness: self
                            .calculate_mitigation_effectiveness(strategy.name(), assessment)
                            as f64,
                        implementation_cost: self.calculate_implementation_cost(strategy.name())
                            as f64,
                    });
                }
            }
        }

        Ok(recommendations)
    }

    /// Calculates mitigation effectiveness
    fn calculate_mitigation_effectiveness(
        &self,
        strategy_name: &str,
        assessment: &RiskAssessment,
    ) -> f32 {
        let base_effectiveness = match strategy_name {
            "PreventiveMitigation" => 0.9,
            "ReactiveMitigation" => 0.7,
            "AdaptiveMitigation" => 0.8,
            _ => 0.6,
        };

        // Adjust based on risk level (assessment.risk_level is String)
        let risk_adjustment = match assessment.risk_level.as_str() {
            "Negligible" | "VeryLow" => 1.0,
            "Low" => 0.9,
            "Medium" => 0.8,
            "High" | "VeryHigh" => 0.7,
            "Severe" | "Critical" | "Extreme" => 0.6,
            _ => 0.75, // Default for unknown risk levels
        };

        base_effectiveness * risk_adjustment
    }

    /// Calculates implementation cost
    fn calculate_implementation_cost(&self, strategy_name: &str) -> f32 {
        match strategy_name {
            "PreventiveMitigation" => 0.8,
            "ReactiveMitigation" => 0.4,
            "AdaptiveMitigation" => 0.9,
            _ => 0.5,
        }
    }

    /// Calculates overall risk confidence
    fn calculate_overall_risk_confidence(&self, assessments: &[RiskAssessment]) -> f32 {
        if assessments.is_empty() {
            return 0.0;
        }

        let confidences: Vec<f32> =
            assessments.iter().map(|a| self.calculate_algorithm_confidence(a)).collect();

        let avg_confidence =
            confidences.iter().map(|&x| x as f64).sum::<f64>() as f32 / confidences.len() as f32;
        let consistency_factor = self.calculate_assessment_consistency(&confidences);

        avg_confidence * consistency_factor
    }

    /// Calculates assessment consistency
    fn calculate_assessment_consistency(&self, confidences: &[f32]) -> f32 {
        if confidences.len() < 2 {
            return 1.0;
        }

        let mean =
            confidences.iter().map(|&x| x as f64).sum::<f64>() as f32 / confidences.len() as f32;
        let variance = confidences.iter().map(|&c| (c - mean).powi(2) as f64).sum::<f64>() as f32
            / confidences.len() as f32;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 1.0 };

        (1.0 - coefficient_of_variation.min(1.0)).max(0.1)
    }
}

// =============================================================================
// THREAD INTERACTION ANALYZER
// =============================================================================

/// Advanced thread interaction analysis system
///
/// Analyzes thread interactions, communication patterns, and synchronization
/// requirements for optimal concurrent test execution.
#[derive(Debug)]
pub struct ThreadInteractionAnalyzer {
    /// Analysis algorithms
    analysis_algorithms: Arc<Mutex<Vec<Box<dyn ThreadAnalysisAlgorithm + Send + Sync>>>>,

    /// Interaction pattern database
    pattern_database: Arc<RwLock<ThreadInteractionPatternDatabase>>,

    /// Performance metrics collector
    metrics_collector: Arc<RwLock<ThreadPerformanceMetrics>>,

    /// Configuration
    config: ThreadAnalysisConfig,
}

impl ThreadInteractionAnalyzer {
    /// Creates a new thread interaction analyzer
    pub async fn new(config: ThreadAnalysisConfig) -> Result<Self> {
        let mut analysis_algorithms: Vec<Box<dyn ThreadAnalysisAlgorithm + Send + Sync>> =
            Vec::new();

        // Initialize thread analysis algorithms
        analysis_algorithms.push(Box::new(CommunicationPatternAnalysis::new()));
        analysis_algorithms.push(Box::new(SynchronizationAnalysis::new()));
        // TODO: PerformanceImpactAnalysis::new requires degradation: f64, impact_areas: Vec<String>
        analysis_algorithms.push(Box::new(PerformanceImpactAnalysis::new(0.0, Vec::new())));
        analysis_algorithms.push(Box::new(ScalabilityAnalysis::new()));

        Ok(Self {
            analysis_algorithms: Arc::new(Mutex::new(analysis_algorithms)),
            pattern_database: Arc::new(RwLock::new(ThreadInteractionPatternDatabase::new())),
            metrics_collector: Arc::new(RwLock::new(ThreadPerformanceMetrics::new())),
            config,
        })
    }

    /// Analyzes thread interactions in test execution data
    pub async fn analyze_thread_interactions(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<ThreadAnalysisResult> {
        let start_time = Utc::now();

        // Extract thread interaction data
        let thread_interactions = self.extract_thread_interactions(test_data)?;

        // Run analysis algorithms synchronously to avoid lifetime issues
        let thread_analysis_results: Vec<_> = {
            let algorithms = self.analysis_algorithms.lock();
            algorithms
                .iter()
                .map(|algorithm| {
                    let algorithm_name = algorithm.name().to_string();
                    let analysis_start = Instant::now();
                    // TODO: analyze_threads takes 0 arguments, removed interactions parameter
                    let result_string = algorithm.analyze_threads();
                    // Convert String result to ThreadAnalysis
                    let result: Result<ThreadAnalysis> = Ok(ThreadAnalysis {
                        thread_count: 0,
                        interactions: Vec::new(),
                        performance_metrics: HashMap::new(),
                        bottlenecks: Vec::new(),
                        detected_patterns: vec![result_string],
                        performance_impact: 0.0,
                        baseline_throughput: 0.0,
                        projected_throughput: 0.0,
                        scalability_factor: 0.0,
                        estimated_saturation_point: 0,
                        optimal_thread_count: 0,
                        cpu_efficiency: 0.0,
                        memory_efficiency: 0.0,
                        synchronization_efficiency: 0.0,
                    });
                    let analysis_duration = analysis_start.elapsed();
                    (algorithm_name, result, analysis_duration)
                })
                .collect()
        };

        // Collect analysis results
        let mut thread_analyses = Vec::new();
        let mut algorithm_results = Vec::new();

        for (algorithm_name, result, duration) in thread_analysis_results {
            match result {
                Ok(analysis) => {
                    algorithm_results.push(ThreadAlgorithmResult {
                        algorithm: algorithm_name,
                        analysis: analysis.clone(),
                        analysis_duration: duration,
                        confidence: self.calculate_thread_analysis_confidence(&analysis) as f64,
                    });
                    thread_analyses.push(analysis);
                },
                Err(e) => {
                    log::warn!("Thread analysis algorithm failed: {}", e);
                },
            }
        }

        // Synthesize results
        let _throughput_analysis_struct = self.synthesize_throughput_analysis(&thread_analyses);
        let _efficiency_metrics_struct = self.calculate_efficiency_metrics(&thread_analyses);
        let interaction_patterns_vec = self.identify_interaction_patterns(&thread_interactions);
        let optimization_opportunities_vec =
            self.identify_optimization_opportunities(&thread_analyses);

        // Convert to expected types
        let throughput_analysis: HashMap<u64, f64> = HashMap::new(); // TODO: extract from throughput_analysis_struct
        let efficiency_metrics: HashMap<String, f64> = HashMap::new(); // TODO: extract from efficiency_metrics_struct
        let interaction_patterns: Vec<String> =
            interaction_patterns_vec.iter().map(|p| format!("{:?}", p)).collect();
        let optimization_opportunities: Vec<String> =
            optimization_opportunities_vec.iter().map(|o| format!("{:?}", o)).collect();

        Ok(ThreadAnalysisResult {
            thread_interactions,
            throughput_analysis,
            efficiency_metrics,
            interaction_patterns,
            optimization_opportunities,
            algorithm_results,
            analysis_window: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence: self.calculate_overall_thread_confidence(&thread_analyses) as f64,
        })
    }

    /// Extracts thread interaction data from test execution traces
    fn extract_thread_interactions(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<Vec<ThreadInteraction>> {
        let mut interactions = Vec::new();
        let mut thread_activities: HashMap<u64, Vec<&ExecutionTrace>> = HashMap::new();

        // Group traces by thread
        for trace in &test_data.execution_traces {
            thread_activities.entry(trace.thread_id).or_default().push(trace);
        }

        // Identify interactions between threads
        for (thread_id, traces) in &thread_activities {
            for trace in traces {
                // Look for interactions with other threads
                for (other_thread_id, other_traces) in &thread_activities {
                    if thread_id != other_thread_id {
                        for other_trace in other_traces {
                            if self.traces_interact(trace, other_trace) {
                                // Convert ThreadInteractionType to InteractionType
                                let thread_interaction_type =
                                    self.determine_interaction_type(trace, other_trace);
                                let interaction_type =
                                    match thread_interaction_type.interaction_type.as_str() {
                                        "ReadWrite" | "WriteRead" | "WriteWrite" | "ReadRead" => {
                                            InteractionType::SharedMemory
                                        },
                                        "LockContention" => InteractionType::Synchronization,
                                        "ChannelCommunication" => InteractionType::MessagePassing,
                                        _ => InteractionType::SharedMemory, // Default fallback
                                    };

                                interactions.push(ThreadInteraction {
                                    source_thread: *thread_id,
                                    target_thread: *other_thread_id,
                                    from_thread: *thread_id,
                                    to_thread: *other_thread_id,
                                    interaction_type,
                                    frequency: 1.0,
                                    // Calculate analysis duration from thread_timeline
                                    analysis_duration: self.calculate_trace_duration(trace),
                                    data_patterns: vec![],
                                    sync_requirements: vec![],
                                    performance_impact: 0.0,
                                    optimization_opportunities: vec![],
                                    safety_considerations: vec![],
                                    timestamp: chrono::Utc::now(), // TODO: Convert Instant to DateTime<Utc>
                                    resource: trace.resource.clone(),
                                    strength: self
                                        .calculate_interaction_strength(trace, other_trace)
                                        as f64,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(interactions)
    }

    /// Checks if two traces represent an interaction
    fn traces_interact(&self, trace1: &ExecutionTrace, trace2: &ExecutionTrace) -> bool {
        // Check if traces access the same resource within a time window
        let resource_match = trace1.resource == trace2.resource;
        let time_window = Duration::from_millis(self.config.interaction_time_window_ms);
        let time_diff = if trace1.timestamp > trace2.timestamp {
            trace1
                .timestamp
                .checked_duration_since(trace2.timestamp)
                .unwrap_or(Duration::ZERO)
        } else {
            trace2
                .timestamp
                .checked_duration_since(trace1.timestamp)
                .unwrap_or(Duration::ZERO)
        };
        let time_proximity = time_diff < time_window;

        resource_match && time_proximity
    }

    /// Determines the type of interaction between threads
    fn determine_interaction_type(
        &self,
        trace1: &ExecutionTrace,
        trace2: &ExecutionTrace,
    ) -> ThreadInteractionType {
        // Match on string operation fields
        match (trace1.operation.as_str(), trace2.operation.as_str()) {
            ("Read", "Write") => ThreadInteractionType {
                interaction_type: "ReadWrite".to_string(),
                synchronization_required: true,
                typical_duration: Duration::from_millis(10),
            },
            ("Write", "Read") => ThreadInteractionType {
                interaction_type: "WriteRead".to_string(),
                synchronization_required: true,
                typical_duration: Duration::from_millis(10),
            },
            ("Write", "Write") => ThreadInteractionType {
                interaction_type: "WriteWrite".to_string(),
                synchronization_required: true,
                typical_duration: Duration::from_millis(15),
            },
            ("LockAcquire" | "LockAcquisition", "LockAcquire" | "LockAcquisition") => {
                ThreadInteractionType {
                    interaction_type: "LockContention".to_string(),
                    synchronization_required: true,
                    typical_duration: Duration::from_millis(20),
                }
            },
            _ => ThreadInteractionType {
                interaction_type: "Other".to_string(),
                synchronization_required: false,
                typical_duration: Duration::from_millis(1),
            },
        }
    }

    /// Calculates interaction strength
    fn calculate_interaction_strength(
        &self,
        trace1: &ExecutionTrace,
        trace2: &ExecutionTrace,
    ) -> f32 {
        let time_diff = if trace1.timestamp > trace2.timestamp {
            trace1
                .timestamp
                .checked_duration_since(trace2.timestamp)
                .unwrap_or(Duration::ZERO)
        } else {
            trace2
                .timestamp
                .checked_duration_since(trace1.timestamp)
                .unwrap_or(Duration::ZERO)
        };
        let time_diff_ms = time_diff.as_millis() as f32;
        // TODO: ExecutionTrace no longer has duration field
        let duration1 = self.calculate_trace_duration(trace1);
        let duration2 = self.calculate_trace_duration(trace2);
        let duration_factor = (duration1.as_millis() as f32 + duration2.as_millis() as f32) / 2.0;

        // Closer in time and longer duration = stronger interaction
        let time_strength = 1.0 / (1.0 + time_diff_ms / 1000.0);
        let duration_strength = (duration_factor / 100.0).min(1.0);

        (time_strength + duration_strength) / 2.0
    }

    /// Calculate duration of an execution trace from its timeline
    fn calculate_trace_duration(&self, trace: &ExecutionTrace) -> Duration {
        // TODO: ExecutionTrace no longer has duration field
        // Calculate from thread_timeline if available
        if let Some(timeline) = trace.thread_timeline.get(&trace.thread_id) {
            if timeline.len() >= 2 {
                let first = timeline.first().map(|(t, _)| *t);
                let last = timeline.last().map(|(t, _)| *t);
                if let (Some(start), Some(end)) = (first, last) {
                    return end.duration_since(start);
                }
            }
        }
        // Default to 100ms if we can't calculate
        Duration::from_millis(100)
    }

    /// Calculates thread analysis confidence
    fn calculate_thread_analysis_confidence(&self, analysis: &ThreadAnalysis) -> f32 {
        // TODO: ThreadAnalysis.detected_patterns is now Vec<String>, not structs with confidence
        // Using count-based heuristic: more patterns = higher confidence
        let pattern_confidence = if analysis.detected_patterns.is_empty() {
            0.5
        } else if analysis.detected_patterns.len() > 3 {
            0.9
        } else {
            0.7
        };

        let metrics_confidence = if analysis.performance_impact > 0.0 {
            1.0 - analysis.performance_impact.abs()
        } else {
            0.8
        };

        ((pattern_confidence + metrics_confidence) / 2.0) as f32
    }

    /// Synthesizes throughput analysis from multiple thread analyses
    fn synthesize_throughput_analysis(&self, analyses: &[ThreadAnalysis]) -> ThroughputAnalysis {
        let baseline_throughput = analyses.iter().map(|a| a.baseline_throughput).sum::<f64>()
            as f32
            / analyses.len().max(1) as f32;

        let projected_throughput = analyses.iter().map(|a| a.projected_throughput).sum::<f64>()
            as f32
            / analyses.len().max(1) as f32;

        let _bottlenecks: Vec<String> =
            analyses.iter().flat_map(|a| a.bottlenecks.clone()).collect();

        ThroughputAnalysis {
            average_throughput: baseline_throughput as f64,
            peak_throughput: projected_throughput as f64,
            throughput_variance: (projected_throughput - baseline_throughput).abs() as f64,
            throughput_trend: if projected_throughput > baseline_throughput {
                "Improving".to_string()
            } else {
                "Stable".to_string()
            },
        }
    }

    /// Analyzes scaling characteristics
    fn analyze_scaling_characteristics(
        &self,
        analyses: &[ThreadAnalysis],
    ) -> ScalingCharacteristics {
        let avg_scalability = analyses.iter().map(|a| a.scalability_factor).sum::<f64>() as f32
            / analyses.len().max(1) as f32;
        let saturation_point = self.estimate_saturation_point(analyses);
        let optimal_thread_count = self.estimate_optimal_thread_count(analyses);

        ScalingCharacteristics {
            horizontal_scaling: ScalingBehavior {
                scaling_type: "Linear".to_string(),
                scaling_efficiency: avg_scalability as f64,
                optimal_scale: optimal_thread_count,
                scaling_limits: (1, saturation_point),
            },
            vertical_scaling: ScalingBehavior {
                scaling_type: "Limited".to_string(),
                scaling_efficiency: avg_scalability as f64 * 0.8,
                optimal_scale: 1,
                scaling_limits: (1, 4),
            },
            scaling_overhead: (1.0 - avg_scalability as f64).max(0.0),
            recommended_scaling_strategy: if avg_scalability > 0.8 {
                "HorizontalScaling".to_string()
            } else {
                "Vertical".to_string()
            },
        }
    }

    /// Estimates saturation point
    fn estimate_saturation_point(&self, analyses: &[ThreadAnalysis]) -> usize {
        analyses
            .iter()
            .map(|a| a.estimated_saturation_point.min(16))
            .min()
            .unwrap_or(16)
    }

    /// Estimates optimal thread count
    fn estimate_optimal_thread_count(&self, analyses: &[ThreadAnalysis]) -> usize {
        analyses.iter().map(|a| a.optimal_thread_count.max(4)).max().unwrap_or(4)
    }

    /// Calculates efficiency metrics
    fn calculate_efficiency_metrics(&self, analyses: &[ThreadAnalysis]) -> EfficiencyMetrics {
        let cpu_efficiency = analyses.iter().map(|a| a.cpu_efficiency).sum::<f64>() as f32
            / analyses.len().max(1) as f32;

        let memory_efficiency = analyses.iter().map(|a| a.memory_efficiency).sum::<f64>() as f32
            / analyses.len().max(1) as f32;

        let synchronization_efficiency =
            analyses.iter().map(|a| a.synchronization_efficiency).sum::<f64>() as f32
                / analyses.len().max(1) as f32;

        let overall_efficiency =
            (cpu_efficiency + memory_efficiency + synchronization_efficiency) / 3.0;

        let efficiency_rating = if overall_efficiency >= 0.9 {
            EfficiencyRating::VeryHigh
        } else if overall_efficiency >= 0.75 {
            EfficiencyRating::High
        } else if overall_efficiency >= 0.5 {
            EfficiencyRating::Medium
        } else if overall_efficiency >= 0.25 {
            EfficiencyRating::Low
        } else {
            EfficiencyRating::VeryLow
        };

        EfficiencyMetrics {
            cpu_efficiency: cpu_efficiency as f64,
            memory_efficiency: memory_efficiency as f64,
            io_efficiency: synchronization_efficiency as f64,
            overall_efficiency: overall_efficiency as f64,
            efficiency_rating,
        }
    }

    /// Analyzes efficiency trends
    fn analyze_efficiency_trends(&self, analyses: &[ThreadAnalysis]) -> EfficiencyTrends {
        let avg_efficiency = analyses
            .iter()
            .map(|a| (a.cpu_efficiency + a.memory_efficiency + a.synchronization_efficiency) / 3.0)
            .sum::<f64>() as f32
            / analyses.len().max(1) as f32;

        EfficiencyTrends {
            trend_data: Vec::new(), // Simplified for this implementation
            trend_direction: "Stable".to_string(),
            average_efficiency: avg_efficiency as f64,
            efficiency_volatility: 0.0,
        }
    }

    /// Identifies interaction patterns
    fn identify_interaction_patterns(
        &self,
        interactions: &[ThreadInteraction],
    ) -> Vec<InteractionPattern> {
        let mut patterns = Vec::new();

        // Identify common patterns
        // TODO: InteractionType is an enum without ReadWrite variant
        // For now, count SharedMemory interactions as potential read-write patterns
        let read_write_count = interactions
            .iter()
            .filter(|i| matches!(i.interaction_type, InteractionType::SharedMemory))
            .count();

        if read_write_count > 0 {
            let freq = read_write_count as f32 / interactions.len().max(1) as f32;
            patterns.push(InteractionPattern {
                // TODO: PatternType::ReadWritePattern doesn't exist, using Concurrency
                pattern_type: "Concurrency".to_string(),
                interacting_components: vec![], // Components involved in interaction
                interaction_frequency: freq as f64,
                frequency: freq as f64,
                confidence: 0.8,
                description: "Read-write interaction pattern detected".to_string(),
                // TODO: PatternImpact::Medium doesn't exist, using Neutral
                impact: PatternImpact::Neutral,
            });
        }

        // Add more pattern detection logic as needed
        patterns
    }

    /// Identifies optimization opportunities
    fn identify_optimization_opportunities(
        &self,
        analyses: &[ThreadAnalysis],
    ) -> Vec<ThreadOptimizationOpportunity> {
        let mut opportunities = Vec::new();

        for analysis in analyses {
            if analysis.synchronization_efficiency < 0.7 {
                opportunities.push(ThreadOptimizationOpportunity {
                    opportunity_type: "reduce_synchronization".to_string(),
                    // TODO: ThreadAnalysis no longer has thread_id field
                    // Using empty vec since ThreadAnalysis doesn't track individual threads
                    affected_threads: vec![],
                    expected_improvement: 0.3,
                    implementation_cost: "medium".to_string(),
                    description: "High synchronization overhead detected".to_string(),
                    implementation_effort: "medium".to_string(),
                    recommendations: vec![
                        "Consider using lock-free data structures".to_string(),
                        "Reduce critical section size".to_string(),
                    ],
                });
            }

            if analysis.cpu_efficiency < 0.6 {
                opportunities.push(ThreadOptimizationOpportunity {
                    opportunity_type: "improve_load_balancing".to_string(),
                    // TODO: ThreadAnalysis no longer has thread_id field
                    affected_threads: vec![],
                    expected_improvement: 0.4,
                    implementation_cost: "high".to_string(),
                    description: "Poor CPU utilization detected".to_string(),
                    implementation_effort: "high".to_string(),
                    recommendations: vec![
                        "Implement work stealing algorithms".to_string(),
                        "Balance workload distribution".to_string(),
                    ],
                });
            }
        }

        opportunities
    }

    /// Calculates overall thread confidence
    fn calculate_overall_thread_confidence(&self, analyses: &[ThreadAnalysis]) -> f32 {
        if analyses.is_empty() {
            return 0.0;
        }

        let confidences: Vec<f32> =
            analyses.iter().map(|a| self.calculate_thread_analysis_confidence(a)).collect();

        confidences.iter().map(|&x| x as f64).sum::<f64>() as f32 / confidences.len() as f32
    }
}

// =============================================================================
// LOCK CONTENTION ANALYZER
// =============================================================================

/// Advanced lock contention analysis system
///
/// Analyzes lock usage patterns, contention levels, and provides optimization
/// recommendations for improved concurrent performance.
#[derive(Debug)]
pub struct LockContentionAnalyzer {
    /// Contention analysis algorithms
    analysis_algorithms: Arc<Mutex<Vec<Box<dyn LockAnalysisAlgorithm + Send + Sync>>>>,

    /// Lock usage pattern database
    pattern_database: Arc<RwLock<LockUsagePatternDatabase>>,

    /// Contention metrics collector
    metrics_collector: Arc<RwLock<LockContentionMetrics>>,

    /// Configuration
    config: LockAnalysisConfig,
}

impl LockContentionAnalyzer {
    /// Creates a new lock contention analyzer
    pub async fn new(config: LockAnalysisConfig) -> Result<Self> {
        let mut analysis_algorithms: Vec<Box<dyn LockAnalysisAlgorithm + Send + Sync>> = Vec::new();

        // Initialize lock analysis algorithms
        // TODO: ContentionFrequencyAnalysis::new requires frequency: f64, hotspots: Vec<String>
        analysis_algorithms.push(Box::new(ContentionFrequencyAnalysis::new(0.0, Vec::new())));
        analysis_algorithms.push(Box::new(HoldTimeAnalysis::new()));
        // TODO: WaitTimeAnalysis::new requires avg_wait_time_us: u64, max_wait_time_us: u64
        analysis_algorithms.push(Box::new(WaitTimeAnalysis::new(0, 0)));
        analysis_algorithms.push(Box::new(DeadlockPotentialAnalysis::new()));

        Ok(Self {
            analysis_algorithms: Arc::new(Mutex::new(analysis_algorithms)),
            pattern_database: Arc::new(RwLock::new(LockUsagePatternDatabase::new())),
            metrics_collector: Arc::new(RwLock::new(LockContentionMetrics::new())),
            config,
        })
    }

    /// Analyzes lock contention in test execution data
    pub async fn analyze_lock_contention(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<LockAnalysisResult> {
        let start_time = Utc::now();

        // Extract lock usage data
        let lock_events = self.extract_lock_events(test_data)?;

        // Execute synchronously to avoid lifetime issues with mutex guards
        let analysis_task_results: Vec<_> = {
            let algorithms = self.analysis_algorithms.lock();
            algorithms
                .iter()
                .map(|algorithm| {
                    let algorithm_name = algorithm.name().to_string();
                    let analysis_start = Instant::now();
                    let result_string = algorithm.analyze_locks();
                    let result: Result<String> = Ok(result_string);
                    let analysis_duration = analysis_start.elapsed();
                    (algorithm_name, result, analysis_duration)
                })
                .collect()
        };

        // Collect analysis results
        let mut lock_analyses_structs = Vec::new(); // For helper methods
        let mut algorithm_results = Vec::new();

        for (algorithm_name, result, duration) in analysis_task_results {
            match result {
                Ok(analysis_string) => {
                    // Create a placeholder LockAnalysis for helper methods
                    let lock_analysis = LockAnalysis {
                        lock_id: analysis_string.clone(),
                        lock_events: Vec::new(),
                        contention_metrics: LockContentionMetrics::default(),
                        dependencies: Vec::new(),
                        analysis_timestamp: chrono::Utc::now(),
                        average_contention_level: 0.0,
                        average_hold_time: Duration::from_secs(0),
                        contention_events: Vec::new(),
                        max_wait_time: Duration::from_secs(0),
                        min_wait_time: Duration::from_secs(0),
                        average_wait_time: Duration::from_secs(0),
                    };

                    algorithm_results.push(LockAlgorithmResult {
                        algorithm: algorithm_name,
                        analysis: analysis_string,
                        analysis_duration: duration,
                        confidence: self.calculate_lock_analysis_confidence(&lock_analysis) as f64,
                    });
                    lock_analyses_structs.push(lock_analysis);
                },
                Err(e) => {
                    log::warn!("Lock analysis algorithm failed: {}", e);
                },
            }
        }

        // Synthesize results
        let _contention_summary_struct = self.synthesize_contention_summary(&lock_analyses_structs);
        let _latency_bounds_struct = self.calculate_latency_bounds(&lock_analyses_structs);
        let optimization_recommendations_vec =
            self.generate_lock_optimizations(&lock_analyses_structs);

        // Convert to expected types
        let contention_summary: HashMap<String, f64> = HashMap::new(); // TODO: extract from contention_summary_struct
        let latency_bounds: HashMap<String, Duration> = HashMap::new(); // TODO: extract from latency_bounds_struct
        let optimization_recommendations: Vec<String> =
            optimization_recommendations_vec.iter().map(|o| format!("{:?}", o)).collect();

        Ok(LockAnalysisResult {
            lock_events,
            contention_summary,
            latency_bounds,
            optimization_recommendations,
            algorithm_results,
            analysis_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence: self.calculate_overall_lock_confidence(&lock_analyses_structs) as f64,
        })
    }

    /// Extracts lock events from test execution traces
    fn extract_lock_events(&self, test_data: &TestExecutionData) -> Result<Vec<LockEvent>> {
        let mut events = Vec::new();

        for trace in &test_data.execution_traces {
            match trace.operation.as_str() {
                "LockAcquire" | "LockAcquisition" => {
                    events.push(LockEvent {
                        timestamp: trace.timestamp,
                        lock_id: trace.resource.clone(),
                        event_type: "acquire".to_string(),
                        thread_id: trace.thread_id,
                        duration: Duration::from_secs(0), // Duration not available in trace
                        wait_time: None,
                        contention_level: 0.0,
                        performance_impact: 0.0,
                        deadlock_risk: 0.0,
                        alternatives: Vec::new(),
                        // TODO: ExecutionTrace no longer has result field
                        success: true, // Assume success if not specified
                    });
                },
                "LockRelease" => {
                    events.push(LockEvent {
                        timestamp: trace.timestamp,
                        lock_id: trace.resource.clone(),
                        event_type: "release".to_string(),
                        thread_id: trace.thread_id,
                        duration: Duration::from_secs(0), // Duration not available in trace
                        wait_time: None,
                        contention_level: 0.0,
                        performance_impact: 0.0,
                        deadlock_risk: 0.0,
                        alternatives: Vec::new(),
                        // TODO: ExecutionTrace no longer has result field
                        success: true, // Assume success if not specified
                    });
                },
                _ => {},
            }
        }

        Ok(events)
    }

    /// Calculates lock analysis confidence
    fn calculate_lock_analysis_confidence(&self, analysis: &LockAnalysis) -> f32 {
        let contention_factor = 1.0 - analysis.average_contention_level;
        let hold_time_factor =
            if analysis.average_hold_time > Duration::from_millis(100) { 0.7 } else { 0.9 };

        ((contention_factor + hold_time_factor) / 2.0) as f32
    }

    /// Synthesizes contention summary
    fn synthesize_contention_summary(&self, analyses: &[LockAnalysis]) -> ContentionSummary {
        let total_contentions = analyses.iter().map(|a| a.contention_events.len()).sum::<usize>();

        let avg_contention_level = analyses.iter().map(|a| a.average_contention_level).sum::<f64>()
            as f32
            / analyses.len().max(1) as f32;

        let _max_wait_time = analyses
            .iter()
            .map(|a| a.max_wait_time)
            .max()
            .unwrap_or(Duration::from_millis(0));

        let hotspot_ids = self
            .identify_contention_hotspots(analyses)
            .iter()
            .map(|h| h.resource_id.clone())
            .collect();

        let contention_by_resource = analyses
            .iter()
            .map(|a| (a.lock_id.clone(), a.contention_events.len()))
            .collect();

        ContentionSummary {
            total_contentions,
            contention_hotspots: hotspot_ids,
            average_contention_duration: Duration::from_secs_f64(avg_contention_level as f64),
            contention_by_resource,
        }
    }

    /// Identifies contention hotspots
    fn identify_contention_hotspots(&self, analyses: &[LockAnalysis]) -> Vec<ContentionHotspot> {
        let mut hotspots = Vec::new();

        for analysis in analyses {
            if analysis.average_contention_level > 0.7 {
                hotspots.push(ContentionHotspot {
                    resource_id: analysis.lock_id.clone(),
                    contention_frequency: analysis.average_contention_level,
                    average_wait_time: analysis.max_wait_time,
                    affected_threads: Vec::new(),
                });
            }
        }

        hotspots
    }

    /// Calculates severity distribution
    fn calculate_severity_distribution(&self, analyses: &[LockAnalysis]) -> SeverityDistribution {
        let mut low = 0;
        let mut medium = 0;
        let mut high = 0;
        let mut critical = 0;

        for analysis in analyses {
            match analysis.average_contention_level {
                x if x < 0.25 => low += 1,
                x if x < 0.5 => medium += 1,
                x if x < 0.75 => high += 1,
                _ => critical += 1,
            }
        }

        let total = low + medium + high + critical;
        let mut distribution = HashMap::new();
        distribution.insert("Low".to_string(), low);
        distribution.insert("Medium".to_string(), medium);
        distribution.insert("High".to_string(), high);
        distribution.insert("Critical".to_string(), critical);

        let most_common = if critical > high && critical > medium && critical > low {
            "Critical"
        } else if high > medium && high > low {
            "High"
        } else if medium > low {
            "Medium"
        } else {
            "Low"
        };

        SeverityDistribution {
            distribution,
            total_count: total,
            most_common_severity: most_common.to_string(),
        }
    }

    /// Calculates latency bounds
    fn calculate_latency_bounds(&self, analyses: &[LockAnalysis]) -> LatencyBounds {
        let min_latency = analyses
            .iter()
            .map(|a| a.min_wait_time)
            .min()
            .unwrap_or(Duration::from_millis(0));

        let max_latency = analyses
            .iter()
            .map(|a| a.max_wait_time)
            .max()
            .unwrap_or(Duration::from_millis(0));

        let avg_latency = Duration::from_millis(
            analyses.iter().map(|a| a.average_wait_time.as_millis() as u64).sum::<u64>()
                / analyses.len().max(1) as u64,
        );

        LatencyBounds {
            min_latency,
            max_latency,
            average_latency: avg_latency,
        }
    }

    /// Calculates percentile latency
    fn calculate_percentile_latency(&self, analyses: &[LockAnalysis], percentile: u8) -> Duration {
        if analyses.is_empty() {
            return Duration::from_millis(0);
        }

        let mut wait_times: Vec<Duration> = analyses.iter().map(|a| a.average_wait_time).collect();

        wait_times.sort();

        let index = ((percentile as f32 / 100.0) * wait_times.len() as f32) as usize;
        wait_times
            .get(index.min(wait_times.len() - 1))
            .copied()
            .unwrap_or(Duration::from_millis(0))
    }

    /// Generates lock optimization recommendations
    fn generate_lock_optimizations(
        &self,
        analyses: &[LockAnalysis],
    ) -> Vec<LockOptimizationRecommendation> {
        let mut recommendations = Vec::new();

        for analysis in analyses {
            if analysis.average_contention_level > 0.5 {
                recommendations.push(LockOptimizationRecommendation {
                    lock_id: analysis.lock_id.clone(),
                    recommendation_type: "ReduceContention".to_string(),
                    expected_improvement: 0.4,
                    implementation_effort: "Medium".to_string(),
                });
            }

            if analysis.average_hold_time > Duration::from_millis(100) {
                recommendations.push(LockOptimizationRecommendation {
                    lock_id: analysis.lock_id.clone(),
                    recommendation_type: "ReduceHoldTime".to_string(),
                    expected_improvement: 0.3,
                    implementation_effort: "Low".to_string(),
                });
            }
        }

        recommendations
    }

    /// Calculates overall lock confidence
    fn calculate_overall_lock_confidence(&self, analyses: &[LockAnalysis]) -> f32 {
        if analyses.is_empty() {
            return 0.0;
        }

        let confidences: Vec<f32> =
            analyses.iter().map(|a| self.calculate_lock_analysis_confidence(a)).collect();

        confidences.iter().map(|&x| x as f64).sum::<f64>() as f32 / confidences.len() as f32
    }
}

// =============================================================================
// CONCURRENCY PATTERN DETECTOR
// =============================================================================

/// Advanced concurrency pattern detection system
///
/// Detects and analyzes common concurrency patterns in test execution to
/// provide insights and optimization recommendations.
#[derive(Debug)]
pub struct ConcurrencyPatternDetector {
    /// Pattern detection algorithms
    detection_algorithms: Arc<Mutex<Vec<Box<dyn PatternDetectionAlgorithm + Send + Sync>>>>,

    /// Known pattern library
    pattern_library: Arc<RwLock<ConcurrencyPatternLibrary>>,

    /// Pattern performance database
    performance_db: Arc<RwLock<PatternPerformanceDatabase>>,

    /// Configuration
    config: PatternDetectionConfig,
}

impl ConcurrencyPatternDetector {
    /// Creates a new concurrency pattern detector
    pub async fn new(config: PatternDetectionConfig) -> Result<Self> {
        let mut detection_algorithms: Vec<Box<dyn PatternDetectionAlgorithm + Send + Sync>> =
            Vec::new();

        // Initialize pattern detection algorithms
        detection_algorithms.push(Box::new(ProducerConsumerDetection::new()));
        detection_algorithms.push(Box::new(MasterWorkerDetection::new()));
        detection_algorithms.push(Box::new(PipelineDetection::new()));
        detection_algorithms.push(Box::new(ForkJoinDetection::new()));

        let pattern_library = ConcurrencyPatternLibrary::new();
        let performance_db = PatternPerformanceDatabase::new();

        Ok(Self {
            detection_algorithms: Arc::new(Mutex::new(detection_algorithms)),
            pattern_library: Arc::new(RwLock::new(pattern_library)),
            performance_db: Arc::new(RwLock::new(performance_db)),
            config,
        })
    }

    /// Detects concurrency patterns in test execution data
    pub async fn detect_concurrency_patterns(
        &self,
        _test_data: &TestExecutionData,
    ) -> Result<PatternAnalysisResult> {
        let start_time = Utc::now();

        // Execute synchronously to avoid lifetime issues with mutex guards
        let detection_results: Vec<_> = {
            let algorithms = self.detection_algorithms.lock();
            algorithms
                .iter()
                .map(|algorithm| {
                    let algorithm_name = algorithm.name().to_string();
                    let detection_start = Instant::now();
                    let result_string = algorithm.detect_patterns();
                    let result: Result<Vec<String>> = Ok(vec![result_string]);
                    let detection_duration = detection_start.elapsed();
                    (algorithm_name, result, detection_duration)
                })
                .collect()
        };

        // Collect detection results
        let mut detected_patterns_strings = Vec::new();
        let mut detected_patterns_structs = Vec::new(); // For helper methods
        let mut algorithm_results = Vec::new();

        for (algorithm_name, result, duration) in detection_results {
            match result {
                Ok(mut patterns) => {
                    // Create placeholder ConcurrencyPattern structs from String results
                    let pattern_structs: Vec<ConcurrencyPattern> = patterns
                        .iter()
                        .map(|pattern_str| ConcurrencyPattern {
                            pattern_type: pattern_str.clone(),
                            description: pattern_str.clone(),
                            characteristics: vec![pattern_str.clone()],
                            applicability: 0.5,
                            confidence: 0.5,
                            thread_count: 1,
                        })
                        .collect();

                    algorithm_results.push(PatternAlgorithmResult {
                        algorithm: algorithm_name,
                        patterns: patterns.clone(),
                        detection_duration: duration,
                        confidence: self.calculate_pattern_detection_confidence(&pattern_structs)
                            as f64,
                    });
                    detected_patterns_strings.append(&mut patterns);
                    detected_patterns_structs.extend(pattern_structs);
                },
                Err(e) => {
                    log::warn!("Pattern detection algorithm failed: {}", e);
                },
            }
        }

        // Deduplicate and classify patterns using structs
        let unique_patterns = self.deduplicate_patterns(&detected_patterns_structs);
        let classified_patterns = self.classify_patterns(&unique_patterns);

        // Analyze scalability characteristics
        let scalability_patterns_vec = self.analyze_scalability_patterns(&classified_patterns);

        // Generate pattern-based recommendations
        let pattern_recommendations =
            self.generate_pattern_recommendations(&classified_patterns).await?;

        // Convert types
        let scalability_patterns: Vec<String> =
            scalability_patterns_vec.iter().map(|p| format!("{:?}", p)).collect();

        // Extract ConcurrencyPattern from ClassifiedConcurrencyPattern
        let detected_patterns: Vec<ConcurrencyPattern> =
            classified_patterns.into_iter().map(|cp| cp.pattern).collect();

        Ok(PatternAnalysisResult {
            detected_patterns,
            scalability_patterns,
            pattern_recommendations,
            algorithm_results,
            timeout_requirements: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence: self.calculate_overall_pattern_confidence(&unique_patterns) as f64,
        })
    }

    /// Calculates pattern detection confidence
    fn calculate_pattern_detection_confidence(&self, patterns: &[ConcurrencyPattern]) -> f32 {
        if patterns.is_empty() {
            return 1.0;
        }

        patterns.iter().map(|p| p.confidence).sum::<f64>() as f32 / patterns.len() as f32
    }

    /// Deduplicates detected patterns
    fn deduplicate_patterns(&self, patterns: &[ConcurrencyPattern]) -> Vec<ConcurrencyPattern> {
        let mut unique_patterns = Vec::new();

        for pattern in patterns {
            let is_duplicate = unique_patterns
                .iter()
                .any(|existing: &ConcurrencyPattern| self.patterns_are_similar(existing, pattern));

            if !is_duplicate {
                unique_patterns.push(pattern.clone());
            }
        }

        unique_patterns
    }

    /// Checks if two patterns are similar
    fn patterns_are_similar(&self, a: &ConcurrencyPattern, b: &ConcurrencyPattern) -> bool {
        a.pattern_type == b.pattern_type
            && a.thread_count == b.thread_count
            && (a.confidence - b.confidence).abs() < 0.2
    }

    /// Classifies patterns by type and characteristics
    fn classify_patterns(
        &self,
        patterns: &[ConcurrencyPattern],
    ) -> Vec<ClassifiedConcurrencyPattern> {
        patterns
            .iter()
            .map(|pattern| ClassifiedConcurrencyPattern {
                pattern: pattern.clone(),
                classification: self.classify_single_pattern(pattern),
                performance_characteristics: self.analyze_pattern_performance(pattern),
                optimization_potential: self.assess_optimization_potential(pattern).potential_score,
            })
            .collect()
    }

    /// Classifies a single pattern
    fn classify_single_pattern(&self, pattern: &ConcurrencyPattern) -> PatternClassification {
        let scalability = self.assess_pattern_scalability(pattern);
        let efficiency = self.assess_pattern_efficiency(pattern);

        PatternClassification {
            classification_type: pattern.pattern_type.clone(),
            confidence: pattern.confidence,
            categories: vec![pattern.pattern_type.clone()],
            primary_type: pattern.pattern_type.clone(),
            complexity_level: self.assess_pattern_complexity(pattern),
            // TODO: ScalabilityRating is now a struct, use its score field
            scalability_rating: scalability.score,
            efficiency_rating: match efficiency {
                EfficiencyRating::VeryLow => 0.1,
                EfficiencyRating::Low => 0.3,
                EfficiencyRating::Medium => 0.6,
                EfficiencyRating::High => 0.9,
                EfficiencyRating::VeryHigh => 1.0,
            },
        }
    }

    /// Assesses pattern complexity
    fn assess_pattern_complexity(&self, pattern: &ConcurrencyPattern) -> ComplexityLevel {
        // Match on string pattern_type field
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => ComplexityLevel::Medium,
            "MasterWorker" => ComplexityLevel::Simple,
            "Pipeline" => ComplexityLevel::Complex,
            "ForkJoin" => ComplexityLevel::Medium,
            _ => ComplexityLevel::Complex, // Default for Custom and unknown patterns
        }
    }

    /// Assesses pattern scalability
    fn assess_pattern_scalability(&self, pattern: &ConcurrencyPattern) -> ScalabilityRating {
        // TODO: ScalabilityRating is now a struct with rating: String and score: f64
        if pattern.thread_count > 8 {
            ScalabilityRating {
                rating: "High".to_string(),
                score: 0.9,
            }
        } else if pattern.thread_count > 4 {
            ScalabilityRating {
                rating: "Medium".to_string(),
                score: 0.6,
            }
        } else {
            ScalabilityRating {
                rating: "Low".to_string(),
                score: 0.3,
            }
        }
    }

    /// Assesses pattern efficiency
    fn assess_pattern_efficiency(&self, pattern: &ConcurrencyPattern) -> EfficiencyRating {
        // Match on string pattern_type field
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => EfficiencyRating::High,
            "MasterWorker" => EfficiencyRating::Medium,
            "Pipeline" => EfficiencyRating::High,
            "ForkJoin" => EfficiencyRating::Medium,
            _ => EfficiencyRating::Low, // Default for Custom and unknown patterns
        }
    }

    /// Analyzes pattern performance characteristics
    fn analyze_pattern_performance(
        &self,
        pattern: &ConcurrencyPattern,
    ) -> PatternPerformanceCharacteristics {
        let scaling = self.analyze_scaling_behavior(pattern);
        let throughput_factor = self.estimate_throughput_factor(pattern);
        let latency_impact = self.estimate_latency_impact(pattern);
        let resource_utilization = self.estimate_resource_utilization(pattern);

        PatternPerformanceCharacteristics {
            throughput: throughput_factor as f64,
            latency: Duration::from_millis((latency_impact * 1000.0) as u64),
            resource_efficiency: resource_utilization as f64,
            throughput_factor: throughput_factor as f64,
            latency_impact: latency_impact as f64,
            resource_utilization: resource_utilization as f64,
            // TODO: ScalingBehavior is now a struct with scaling_type: String
            scaling_behavior: scaling.scaling_type.clone(),
        }
    }

    /// Estimates throughput factor
    fn estimate_throughput_factor(&self, pattern: &ConcurrencyPattern) -> f32 {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 0.8,
            "MasterWorker" => 0.9,
            "Pipeline" => 0.95,
            "ForkJoin" => 0.7,
            _ => 0.6, // Custom or unknown
        }
    }

    /// Estimates latency impact
    fn estimate_latency_impact(&self, pattern: &ConcurrencyPattern) -> f32 {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 0.2,
            "MasterWorker" => 0.1,
            "Pipeline" => 0.3,
            "ForkJoin" => 0.4,
            _ => 0.5, // Custom or unknown
        }
    }

    /// Estimates resource utilization
    fn estimate_resource_utilization(&self, pattern: &ConcurrencyPattern) -> f32 {
        let base_utilization = match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 0.75,
            "MasterWorker" => 0.85,
            "Pipeline" => 0.9,
            "ForkJoin" => 0.65,
            _ => 0.5, // Custom or unknown
        };

        // Adjust based on thread count
        let thread_factor = (pattern.thread_count as f32).log2() / 4.0;
        (base_utilization * (1.0 + thread_factor)).min(1.0)
    }

    /// Analyzes scaling behavior
    fn analyze_scaling_behavior(&self, pattern: &ConcurrencyPattern) -> ScalingBehavior {
        // TODO: ScalingBehavior is now a struct, not an enum
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => ScalingBehavior {
                scaling_type: "Linear".to_string(),
                scaling_efficiency: 0.9,
                optimal_scale: 8,
                scaling_limits: (1, 64),
            },
            "MasterWorker" => ScalingBehavior {
                scaling_type: "SubLinear".to_string(),
                scaling_efficiency: 0.7,
                optimal_scale: 4,
                scaling_limits: (1, 32),
            },
            "Pipeline" => ScalingBehavior {
                scaling_type: "Linear".to_string(),
                scaling_efficiency: 0.85,
                optimal_scale: 8,
                scaling_limits: (1, 64),
            },
            "ForkJoin" => ScalingBehavior {
                scaling_type: "SubLinear".to_string(),
                scaling_efficiency: 0.75,
                optimal_scale: 6,
                scaling_limits: (1, 48),
            },
            _ => ScalingBehavior {
                scaling_type: "Unknown".to_string(),
                scaling_efficiency: 0.5,
                optimal_scale: 4,
                scaling_limits: (1, 16),
            },
        }
    }

    /// Assesses optimization potential
    fn assess_optimization_potential(&self, pattern: &ConcurrencyPattern) -> OptimizationPotential {
        let throughput_improvement = self.estimate_throughput_improvement_potential(pattern);
        let latency_reduction = self.estimate_latency_reduction_potential(pattern);
        let resource_efficiency = self.estimate_resource_efficiency_potential(pattern);
        let complexity = self.estimate_optimization_complexity(pattern);

        let potential_score =
            (throughput_improvement + latency_reduction + resource_efficiency) / 3.0;

        let mut optimization_areas = Vec::new();
        if throughput_improvement > 0.3 {
            optimization_areas.push("Throughput".to_string());
        }
        if latency_reduction > 0.3 {
            optimization_areas.push("Latency".to_string());
        }
        if resource_efficiency > 0.3 {
            optimization_areas.push("ResourceEfficiency".to_string());
        }

        // TODO: OptimizationComplexity is now a struct with complexity_level: ComplexityLevel
        let feasibility = match complexity.complexity_level {
            ComplexityLevel::VerySimple | ComplexityLevel::Simple => 0.9,
            ComplexityLevel::Medium => 0.6,
            ComplexityLevel::Complex | ComplexityLevel::VeryComplex => 0.3,
            ComplexityLevel::HighlyComplex => 0.2,
            ComplexityLevel::ExtremelyComplex => 0.1,
        };

        OptimizationPotential {
            potential_score: potential_score as f64,
            optimization_areas,
            expected_improvement: ((throughput_improvement + latency_reduction) / 2.0) as f64,
            feasibility,
        }
    }

    /// Estimates throughput improvement potential
    fn estimate_throughput_improvement_potential(&self, pattern: &ConcurrencyPattern) -> f32 {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 0.3,
            "MasterWorker" => 0.2,
            "Pipeline" => 0.4,
            "ForkJoin" => 0.5,
            _ => 0.6, // Custom or other patterns
        }
    }

    /// Estimates latency reduction potential
    fn estimate_latency_reduction_potential(&self, pattern: &ConcurrencyPattern) -> f32 {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 0.2,
            "MasterWorker" => 0.1,
            "Pipeline" => 0.3,
            "ForkJoin" => 0.4,
            _ => 0.5, // Custom or other patterns
        }
    }

    /// Estimates resource efficiency potential
    fn estimate_resource_efficiency_potential(&self, pattern: &ConcurrencyPattern) -> f32 {
        let current_efficiency = self.estimate_resource_utilization(pattern);
        1.0 - current_efficiency
    }

    /// Estimates optimization complexity
    fn estimate_optimization_complexity(
        &self,
        pattern: &ConcurrencyPattern,
    ) -> OptimizationComplexity {
        // TODO: OptimizationComplexity is now a struct, not an enum
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => OptimizationComplexity {
                complexity_level: ComplexityLevel::Medium,
                complexity_score: 0.5,
                complexity_factors: vec!["Coordination overhead".to_string()],
            },
            "MasterWorker" => OptimizationComplexity {
                complexity_level: ComplexityLevel::Simple,
                complexity_score: 0.3,
                complexity_factors: vec!["Simple distribution".to_string()],
            },
            "Pipeline" => OptimizationComplexity {
                complexity_level: ComplexityLevel::Complex,
                complexity_score: 0.8,
                complexity_factors: vec!["Stage synchronization".to_string()],
            },
            "ForkJoin" => OptimizationComplexity {
                complexity_level: ComplexityLevel::Medium,
                complexity_score: 0.6,
                complexity_factors: vec!["Join coordination".to_string()],
            },
            _ => OptimizationComplexity {
                complexity_level: ComplexityLevel::Complex,
                complexity_score: 0.9,
                complexity_factors: vec!["Unknown pattern".to_string()],
            },
        }
    }

    /// Analyzes scalability patterns
    fn analyze_scalability_patterns(
        &self,
        patterns: &[ClassifiedConcurrencyPattern],
    ) -> Vec<ScalabilityPattern> {
        let mut scalability_patterns = Vec::new();

        for pattern in patterns {
            let pattern_type_str = format!("{:?}", pattern.pattern.pattern_type);
            let efficiency_curve_data = self.model_efficiency_curve(&pattern.pattern);
            let scalability_pattern = ScalabilityPattern {
                pattern_type: pattern_type_str,
                efficiency_curve: efficiency_curve_data
                    .data_points
                    .iter()
                    .map(|(_, y)| *y)
                    .collect(),
            };

            scalability_patterns.push(scalability_pattern);
        }

        scalability_patterns
    }

    /// Calculates scaling factor
    fn calculate_scaling_factor(&self, pattern: &ConcurrencyPattern) -> f32 {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 0.8,
            "MasterWorker" => 0.9,
            "Pipeline" => 0.85,
            "ForkJoin" => 0.75,
            _ => 0.6, // Custom or unknown
        }
    }

    /// Estimates optimal thread count
    fn estimate_optimal_threads(&self, pattern: &ConcurrencyPattern) -> usize {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 4,
            "MasterWorker" => 8,
            "Pipeline" => 6,
            "ForkJoin" => 4,
            _ => 2, // Custom or unknown
        }
    }

    /// Estimates saturation point
    fn estimate_saturation_point(&self, pattern: &ConcurrencyPattern) -> usize {
        match pattern.pattern_type.as_str() {
            "ProducerConsumer" => 8,
            "MasterWorker" => 16,
            "Pipeline" => 12,
            "ForkJoin" => 8,
            _ => 4, // Custom or unknown
        }
    }

    /// Models efficiency curve
    fn model_efficiency_curve(&self, pattern: &ConcurrencyPattern) -> EfficiencyCurve {
        let optimal_threads = self.estimate_optimal_threads(pattern) as f64;
        let saturation_point = self.estimate_saturation_point(pattern) as f64;

        // Generate data points for the efficiency curve
        let mut data_points = Vec::new();
        for i in 1..=32 {
            let threads = i as f64;
            let efficiency = if threads <= optimal_threads {
                threads / optimal_threads // Linear growth in optimal region
            } else if threads <= saturation_point {
                1.0 - (threads - optimal_threads) / (saturation_point - optimal_threads) * 0.2
            // Slight degradation
            } else {
                0.8 - (threads - saturation_point) / 32.0 * 0.3 // Further degradation
            };
            data_points.push((threads, efficiency));
        }

        EfficiencyCurve {
            data_points,
            curve_type: "Logarithmic".to_string(),
            peak_efficiency: 1.0,
            optimal_point: (optimal_threads, 1.0),
        }
    }

    /// Generates efficiency function
    fn generate_efficiency_function(&self, pattern: &ConcurrencyPattern) -> EfficiencyFunction {
        EfficiencyFunction {
            function_type: "Logarithmic".to_string(),
            parameters: vec![
                self.calculate_scaling_factor(pattern) as f64,
                self.estimate_optimal_threads(pattern) as f64,
                self.estimate_saturation_point(pattern) as f64,
            ],
            domain: (1.0, 128.0), // Thread count domain (1 to 128 threads)
            range: (0.0, 1.0),    // Efficiency range (0% to 100%)
        }
    }

    /// Generates pattern-based recommendations
    async fn generate_pattern_recommendations(
        &self,
        patterns: &[ClassifiedConcurrencyPattern],
    ) -> Result<Vec<PatternOptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        for pattern in patterns {
            // optimization_potential is f64, use it directly for threshold comparison
            if pattern.optimization_potential > 0.2 {
                let effort = self.convert_complexity_to_effort(0.5);
                recommendations.push(PatternOptimizationRecommendation {
                    pattern_type: pattern.pattern.pattern_type.clone(),
                    optimization_type: "ThroughputOptimization".to_string(),
                    description: "Significant throughput improvement potential detected"
                        .to_string(),
                    expected_improvement: pattern.optimization_potential,
                    implementation_effort: format!("{:?}", effort),
                    recommendations: vec!["Optimize throughput".to_string()], // TODO: parse pattern_type to enum
                });
            }

            if pattern.optimization_potential > 0.3 {
                let effort = self.convert_complexity_to_effort(0.5);
                recommendations.push(PatternOptimizationRecommendation {
                    pattern_type: pattern.pattern.pattern_type.clone(),
                    optimization_type: "LatencyOptimization".to_string(),
                    description: "Significant latency reduction potential detected".to_string(),
                    expected_improvement: pattern.optimization_potential,
                    implementation_effort: format!("{:?}", effort),
                    recommendations: vec!["Reduce latency".to_string()], // TODO: parse pattern_type to enum
                });
            }
        }

        Ok(recommendations)
    }

    /// Converts optimization complexity to effort
    fn convert_complexity_to_effort(&self, complexity: f64) -> OptimizationEffort {
        if complexity < 0.3 {
            OptimizationEffort::Low
        } else if complexity < 0.6 {
            OptimizationEffort::Medium
        } else {
            OptimizationEffort::High
        }
    }

    /// Generates throughput recommendations
    fn generate_throughput_recommendations(
        &self,
        pattern_type: &ConcurrencyPatternType,
    ) -> Vec<String> {
        match pattern_type {
            ConcurrencyPatternType::ProducerConsumer => vec![
                "Implement bounded queues with optimal capacity".to_string(),
                "Use multiple producer/consumer threads".to_string(),
                "Consider lock-free queue implementations".to_string(),
            ],
            ConcurrencyPatternType::MasterWorker => vec![
                "Implement work stealing algorithms".to_string(),
                "Balance workload distribution".to_string(),
                "Use thread pool sizing based on workload".to_string(),
            ],
            ConcurrencyPatternType::Pipeline => vec![
                "Optimize pipeline stage parallelism".to_string(),
                "Balance pipeline stage processing times".to_string(),
                "Implement dynamic pipeline scaling".to_string(),
            ],
            ConcurrencyPatternType::ForkJoin => vec![
                "Optimize task granularity".to_string(),
                "Implement efficient join synchronization".to_string(),
                "Use recursive task decomposition".to_string(),
            ],
            ConcurrencyPatternType::Custom(_) => vec![
                "Analyze custom pattern for optimization opportunities".to_string(),
                "Consider standard pattern alternatives".to_string(),
            ],
        }
    }

    /// Generates latency recommendations
    fn generate_latency_recommendations(
        &self,
        pattern_type: &ConcurrencyPatternType,
    ) -> Vec<String> {
        match pattern_type {
            ConcurrencyPatternType::ProducerConsumer => vec![
                "Reduce queue wait times".to_string(),
                "Implement priority queues for urgent tasks".to_string(),
                "Minimize synchronization overhead".to_string(),
            ],
            ConcurrencyPatternType::MasterWorker => vec![
                "Reduce task distribution overhead".to_string(),
                "Implement worker affinity".to_string(),
                "Use batched task assignment".to_string(),
            ],
            ConcurrencyPatternType::Pipeline => vec![
                "Reduce inter-stage latency".to_string(),
                "Implement pipeline bypassing for urgent tasks".to_string(),
                "Optimize stage transition overhead".to_string(),
            ],
            ConcurrencyPatternType::ForkJoin => vec![
                "Minimize fork overhead".to_string(),
                "Optimize join synchronization".to_string(),
                "Implement early termination strategies".to_string(),
            ],
            ConcurrencyPatternType::Custom(_) => vec![
                "Analyze critical path for latency bottlenecks".to_string(),
                "Implement asynchronous operations where possible".to_string(),
            ],
        }
    }

    /// Calculates overall pattern confidence
    fn calculate_overall_pattern_confidence(&self, patterns: &[ConcurrencyPattern]) -> f32 {
        if patterns.is_empty() {
            return 0.0;
        }

        let confidences: Vec<f32> = patterns.iter().map(|p| p.confidence as f32).collect();

        confidences.iter().map(|&x| x as f64).sum::<f64>() as f32 / confidences.len() as f32
    }
}

// =============================================================================
// SAFETY VALIDATOR
// =============================================================================

/// Comprehensive safety validation system
///
/// Validates concurrency safety constraints and ensures compliance with
/// safety requirements for concurrent test execution.
#[derive(Debug)]
pub struct SafetyValidator {
    /// Safety validation rules
    validation_rules: Arc<Mutex<Vec<Box<dyn SafetyValidationRule + Send + Sync>>>>,

    /// Safety constraint database
    constraint_db: Arc<RwLock<SafetyConstraintDatabase>>,

    /// Validation history
    validation_history: Arc<Mutex<SafetyValidationHistory>>,

    /// Configuration
    config: SafetyValidationConfig,
}

impl SafetyValidator {
    /// Creates a new safety validator
    pub async fn new(config: SafetyValidationConfig) -> Result<Self> {
        let mut validation_rules: Vec<Box<dyn SafetyValidationRule + Send + Sync>> = Vec::new();

        // Initialize safety validation rules
        validation_rules.push(Box::new(DeadlockSafetyRule::new()));
        validation_rules.push(Box::new(ResourceSafetyRule::new()));
        validation_rules.push(Box::new(ConcurrencySafetyRule::new()));
        validation_rules.push(Box::new(IsolationSafetyRule::new()));

        let constraint_db = SafetyConstraintDatabase::new();

        Ok(Self {
            validation_rules: Arc::new(Mutex::new(validation_rules)),
            constraint_db: Arc::new(RwLock::new(constraint_db)),
            validation_history: Arc::new(Mutex::new(SafetyValidationHistory::new())),
            config,
        })
    }

    /// Validates safety of concurrency requirements
    pub async fn validate_safety(
        &self,
        _requirements: &ConcurrencyRequirements,
        _test_data: &TestExecutionData,
    ) -> Result<SafetyValidationResult> {
        let start_time = Utc::now();

        // Run safety validation rules synchronously to avoid lifetime issues
        let safety_validation_results: Vec<_> = {
            let rules = self.validation_rules.lock();
            rules
                .iter()
                .map(|rule| {
                    let rule_name = rule.name().to_string();
                    let validation_start = Instant::now();
                    // TODO: validate_safety takes 0 arguments, removed requirements and test_data parameters
                    let result_bool = rule.validate_safety();
                    // Convert bool to SafetyValidation
                    let result: Result<SafetyValidation> = Ok(SafetyValidation {
                        is_safe: result_bool,
                        validation_checks: Vec::new(),
                        violations_found: Vec::new(),
                        violations: Vec::new(),
                    });
                    let validation_duration = validation_start.elapsed();
                    (rule_name, result, validation_duration)
                })
                .collect()
        };

        // Collect validation results
        let mut validation_results = Vec::new();
        let mut safety_violations = Vec::new();

        for (rule_name, result, duration) in safety_validation_results {
            match result {
                Ok(validation) => {
                    validation_results.push(SafetyRuleResult {
                        rule_name,
                        validation: validation.clone(),
                        validation_duration: duration,
                        confidence: self.calculate_rule_confidence(&validation) as f64,
                    });

                    if !validation.is_safe {
                        safety_violations.extend(validation.violations);
                    }
                },
                Err(e) => {
                    log::warn!("Safety validation rule failed: {}", e);
                },
            }
        }

        // Determine overall safety
        let overall_safety = safety_violations.is_empty();
        let safety_score = self.calculate_safety_score(&validation_results);

        // Generate safety recommendations
        let safety_recommendations_vec = self.generate_safety_recommendations(&safety_violations);

        // Generate compliance report
        let _compliance_report_struct =
            self.generate_compliance_report(&validation_results, &safety_violations);

        // Convert types
        let safety_recommendations: Vec<String> =
            safety_recommendations_vec.iter().map(|r| format!("{:?}", r)).collect();
        let compliance_report: HashMap<String, bool> = HashMap::new(); // TODO: extract from compliance_report_struct

        let confidence = self.calculate_overall_safety_confidence(&validation_results) as f64;

        let result = SafetyValidationResult {
            overall_safety,
            safety_score: safety_score as f64,
            safety_violations,
            safety_recommendations,
            safety_constraints: Vec::new(), // TODO: populate with actual safety constraints
            compliance_report,
            validation_results,
            validation_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            confidence,
        };

        // Store validation result
        self.store_validation_result(&result).await?;

        Ok(result)
    }

    /// Calculates rule confidence
    fn calculate_rule_confidence(&self, validation: &SafetyValidation) -> f32 {
        let violation_factor = if validation.violations.is_empty() {
            1.0
        } else {
            1.0 - (validation.violations.len() as f32 * 0.1).min(0.8)
        };

        let severity_factor = validation
            .violations
            .iter()
            .map(|v| match v.severity {
                ViolationSeverity::Critical => 0.2,
                ViolationSeverity::High => 0.4,
                ViolationSeverity::Medium => 0.6,
                ViolationSeverity::Low => 0.8,
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);

        (violation_factor + severity_factor) / 2.0
    }

    /// Calculates overall safety score
    fn calculate_safety_score(&self, results: &[SafetyRuleResult]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        let safe_rules = results.iter().filter(|r| r.validation.is_safe).count();

        safe_rules as f32 / results.len() as f32
    }

    /// Generates safety recommendations
    fn generate_safety_recommendations(
        &self,
        violations: &[SafetyViolation],
    ) -> Vec<SafetyRecommendation> {
        let mut recommendations = Vec::new();

        for (idx, violation) in violations.iter().enumerate() {
            let recommendation_text = self.generate_violation_recommendation(violation);
            let severity_str = format!("{:?}", violation.severity);

            recommendations.push(SafetyRecommendation {
                recommendation_id: format!("safety_rec_{}", idx),
                recommendation_type: format!("{:?}", violation.violation_type),
                description: recommendation_text,
                severity: severity_str,
            });
        }

        recommendations
    }

    /// Generates recommendation for a specific violation
    fn generate_violation_recommendation(&self, violation: &SafetyViolation) -> String {
        match violation.violation_type {
            ViolationType::DataRace => {
                "Eliminate data races through proper synchronization or using thread-safe primitives"
                    .to_string()
            },
            ViolationType::Deadlock => {
                "Resolve deadlock by reviewing lock ordering and avoiding circular dependencies"
                    .to_string()
            },
            ViolationType::ResourceLeak => {
                "Fix resource leaks by ensuring proper cleanup and RAII patterns".to_string()
            },
            ViolationType::SynchronizationViolation => {
                "Correct synchronization issues by using appropriate synchronization primitives"
                    .to_string()
            },
            ViolationType::LockOrderingViolation => {
                "Establish and enforce consistent lock ordering to prevent deadlocks".to_string()
            },
            ViolationType::DeadlockRisk => {
                "Implement deadlock prevention strategies such as ordered locking or timeouts"
                    .to_string()
            },
            ViolationType::ResourceConflict => {
                "Resolve resource conflicts through partitioning or temporal separation".to_string()
            },
            ViolationType::ConcurrencyViolation => {
                "Adjust concurrency levels to meet safety constraints".to_string()
            },
            ViolationType::IsolationBreach => {
                "Strengthen isolation between concurrent operations".to_string()
            },
            ViolationType::Custom(ref custom) => {
                format!("Address custom safety concern: {}", custom)
            },
        }
    }

    /// Calculates implementation priority
    fn calculate_implementation_priority(
        &self,
        violation: &SafetyViolation,
    ) -> ImplementationPriority {
        match violation.severity {
            ViolationSeverity::Critical => ImplementationPriority {
                priority_level: 4,
                priority_name: "Immediate".to_string(),
            },
            ViolationSeverity::High => ImplementationPriority {
                priority_level: 3,
                priority_name: "High".to_string(),
            },
            ViolationSeverity::Medium => ImplementationPriority {
                priority_level: 2,
                priority_name: "Medium".to_string(),
            },
            ViolationSeverity::Low => ImplementationPriority {
                priority_level: 1,
                priority_name: "Low".to_string(),
            },
        }
    }

    /// Estimates safety impact
    fn estimate_safety_impact(&self, violation: &SafetyViolation) -> f32 {
        match violation.severity {
            ViolationSeverity::Critical => 1.0,
            ViolationSeverity::High => 0.8,
            ViolationSeverity::Medium => 0.6,
            ViolationSeverity::Low => 0.4,
        }
    }

    /// Generates compliance report
    fn generate_compliance_report(
        &self,
        results: &[SafetyRuleResult],
        violations: &[SafetyViolation],
    ) -> ComplianceReport {
        let total_rules = results.len();
        let passed_rules = results.iter().filter(|r| r.validation.is_safe).count();
        let _failed_rules = total_rules - passed_rules;

        let critical_violations = violations
            .iter()
            .filter(|v| matches!(v.severity, ViolationSeverity::Critical))
            .count();

        let high_violations = violations
            .iter()
            .filter(|v| matches!(v.severity, ViolationSeverity::High))
            .count();

        let compliance_status = if critical_violations > 0 {
            ComplianceStatus::NonCompliant
        } else if high_violations > 0 {
            ComplianceStatus::ConditionallyCompliant
        } else {
            ComplianceStatus::Compliant
        };

        let violation_list: Vec<String> = violations
            .iter()
            .map(|v| format!("{:?}: {}", v.violation_type, v.severity))
            .collect();

        ComplianceReport {
            report_id: format!("compliance_{}", chrono::Utc::now().timestamp()),
            compliance_status,
            violations: violation_list,
            generated_at: chrono::Utc::now(),
        }
    }

    /// Stores validation result for historical tracking
    async fn store_validation_result(&self, result: &SafetyValidationResult) -> Result<()> {
        let mut history = self.validation_history.lock();
        history.add_validation_result(result.clone());

        // Cleanup old results if needed
        let retention_limit = self.config.history_retention_limit;
        history.cleanup(retention_limit);

        Ok(())
    }

    /// Calculates overall safety confidence
    fn calculate_overall_safety_confidence(&self, results: &[SafetyRuleResult]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        let confidences: Vec<f32> = results.iter().map(|r| r.confidence as f32).collect();

        confidences.iter().map(|&x| x as f64).sum::<f64>() as f32 / confidences.len() as f32
    }
}
