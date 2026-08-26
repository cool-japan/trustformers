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

/// Converts a SharingCapability enum variant into a ResourceSharingCapabilities struct
fn sharing_capability_to_struct(cap: &SharingCapability) -> ResourceSharingCapabilities {
    use std::collections::HashMap;
    match cap {
        SharingCapability::None => ResourceSharingCapabilities {
            supports_read_sharing: false,
            supports_write_sharing: false,
            max_concurrent_readers: Some(0),
            max_concurrent_writers: Some(0),
            sharing_overhead: 0.0,
            consistency_guarantees: Vec::new(),
            isolation_requirements: vec!["Exclusive access required".to_string()],
            recommended_strategy: SharingStrategy::NoSharing,
            safety_assessment: 1.0,
            performance_tradeoffs: HashMap::new(),
            performance_overhead: 0.0,
            implementation_complexity: 0.1,
            sharing_mode: "none".to_string(),
        },
        SharingCapability::ReadOnly => ResourceSharingCapabilities {
            supports_read_sharing: true,
            supports_write_sharing: false,
            max_concurrent_readers: None,
            max_concurrent_writers: Some(0),
            sharing_overhead: 0.05,
            consistency_guarantees: vec!["Read consistency guaranteed".to_string()],
            isolation_requirements: vec!["No concurrent writers".to_string()],
            recommended_strategy: SharingStrategy::ReadSharing,
            safety_assessment: 0.95,
            performance_tradeoffs: HashMap::new(),
            performance_overhead: 0.05,
            implementation_complexity: 0.2,
            sharing_mode: "read-only".to_string(),
        },
        SharingCapability::ReadWrite => ResourceSharingCapabilities {
            supports_read_sharing: true,
            supports_write_sharing: true,
            max_concurrent_readers: None,
            max_concurrent_writers: Some(1),
            sharing_overhead: 0.15,
            consistency_guarantees: vec!["Sequential consistency for writes".to_string()],
            isolation_requirements: vec!["Write serialization required".to_string()],
            recommended_strategy: SharingStrategy::CopyOnWrite,
            safety_assessment: 0.80,
            performance_tradeoffs: {
                let mut m = HashMap::new();
                m.insert("write_latency".to_string(), 0.15);
                m.insert("read_latency".to_string(), 0.05);
                m
            },
            performance_overhead: 0.15,
            implementation_complexity: 0.6,
            sharing_mode: "read-write".to_string(),
        },
        SharingCapability::Exclusive => ResourceSharingCapabilities {
            supports_read_sharing: false,
            supports_write_sharing: true,
            max_concurrent_readers: Some(1),
            max_concurrent_writers: Some(1),
            sharing_overhead: 0.25,
            consistency_guarantees: vec!["Exclusive access guaranteed".to_string()],
            isolation_requirements: vec!["Single accessor at a time".to_string()],
            recommended_strategy: SharingStrategy::NoSharing,
            safety_assessment: 0.99,
            performance_tradeoffs: {
                let mut m = HashMap::new();
                m.insert("throughput_reduction".to_string(), 0.5);
                m
            },
            performance_overhead: 0.25,
            implementation_complexity: 0.4,
            sharing_mode: "exclusive".to_string(),
        },
        SharingCapability::Shared => ResourceSharingCapabilities {
            supports_read_sharing: true,
            supports_write_sharing: false,
            max_concurrent_readers: Some(8),
            max_concurrent_writers: Some(0),
            sharing_overhead: 0.10,
            consistency_guarantees: vec!["Eventual consistency".to_string()],
            isolation_requirements: Vec::new(),
            recommended_strategy: SharingStrategy::ReadSharing,
            safety_assessment: 0.85,
            performance_tradeoffs: HashMap::new(),
            performance_overhead: 0.10,
            implementation_complexity: 0.35,
            sharing_mode: "shared".to_string(),
        },
    }
}

/// Reports the detector's own observable state.
///
/// The sub-analyzers hold `dyn` algorithm objects that are not `Debug`, so this
/// is written by hand rather than derived; it names what can be read without
/// taking any of their locks.
impl std::fmt::Debug for ConcurrencyRequirementsDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConcurrencyRequirementsDetector")
            .field(
                "shutdown",
                &self.shutdown.load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("background_tasks", &self.background_tasks.lock().len())
            .finish_non_exhaustive()
    }
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

        // Convert PatternEstimationConfig to EstimationConfig:
        // - safety_margin: derived from enabled flag (active = more conservative 0.2, disabled = 0.5)
        //   and bounded below by 1/(timeout_seconds+1) to reflect urgency
        // - history_retention_limit: derived from max_iterations (capped to reasonable range)
        let estimation_config = {
            let pc = &config.estimation_config;
            EstimationConfig {
                safety_margin: if pc.enabled {
                    0.2_f64.max(1.0 / (pc.timeout_seconds as f64 + 1.0))
                } else {
                    0.5
                },
                history_retention_limit: pc.max_iterations.clamp(100, 10_000),
            }
        };
        let estimator = Arc::new(
            SafeConcurrencyEstimator::new(estimation_config)
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

        let analysis_result = ConcurrencyAnalysisResult {
            timestamp: start_time,
            test_id: test_data.test_id.clone(),
            max_safe_concurrency: estimation_result.max_safe_concurrency,
            recommended_concurrency: estimation_result.recommended_concurrency,
            resource_conflicts: conflict_result.resource_conflicts.clone(),
            lock_dependencies,
            sharing_capabilities: sharing_result
                .sharing_capabilities
                .iter()
                .map(sharing_capability_to_struct)
                .collect(),
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
            resource_sharing: sharing
                .sharing_capabilities
                .first()
                .map(sharing_capability_to_struct)
                .unwrap_or_else(|| ResourceSharingCapabilities {
                    supports_read_sharing: false,
                    supports_write_sharing: false,
                    max_concurrent_readers: Some(0),
                    max_concurrent_writers: Some(0),
                    sharing_overhead: 0.0,
                    consistency_guarantees: Vec::new(),
                    isolation_requirements: vec!["No sharing configured".to_string()],
                    recommended_strategy: SharingStrategy::NoSharing,
                    safety_assessment: 1.0,
                    performance_tradeoffs: std::collections::HashMap::new(),
                    performance_overhead: 0.0,
                    implementation_complexity: 0.1,
                    sharing_mode: "none".to_string(),
                }),
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
        let mut confidence_scores = Vec::new();

        // Non-zero concurrency indicates we have a concrete estimate
        if requirements.max_concurrent_instances > 0 {
            confidence_scores.push(0.9_f64);
        }

        // Having identified resource constraints improves confidence
        if !requirements.resource_constraints.is_empty() {
            confidence_scores.push(0.85);
        }

        // Having shared resources information improves confidence
        if !requirements.shared_resources.is_empty() {
            confidence_scores.push(0.8);
        }

        // Safety constraints being defined improves confidence
        if requirements.safety_constraints.max_instances > 0 {
            confidence_scores.push(0.85);
        }

        // Performance guarantees further improve confidence
        if !requirements.performance_guarantees.is_empty() {
            confidence_scores.push(0.8);
        }

        if confidence_scores.is_empty() {
            0.5
        } else {
            confidence_scores.iter().sum::<f64>() as f32 / confidence_scores.len() as f32
        }
    }

    /// Extract lock dependencies from lock analysis result.
    ///
    /// Delegates to [`lock_dependencies_from_events`], which derives every
    /// field it publishes from the trace.
    fn extract_lock_dependencies(
        &self,
        lock_result: &LockAnalysisResult,
    ) -> Result<Vec<LockDependency>> {
        Ok(lock_dependencies_from_events(&lock_result.lock_events))
    }

    /// Generates optimization recommendations.
    ///
    /// One recommendation is produced, and only when the analysis concluded
    /// that the test is safe at a concurrency of exactly one.
    ///
    /// `confidence` is the confidence of the analysis this advice rests on, so
    /// a weakly-supported analysis no longer yields a recommendation stated as
    /// firmly as a well-supported one (it was the constant `0.95`). The
    /// remaining numbers -- `expected_benefit`, `performance_impact`,
    /// `risk_assessment`, `expected_impact` and `implementation_complexity` --
    /// are fixed properties of *this kind of advice* ("run it serially": large
    /// benefit, trivial to implement, little risk), not measurements of the
    /// test under analysis. Their types cannot express that distinction; the
    /// owner of [`ConcurrencyRecommendation`] would have to make them optional
    /// for the record to state it in the data rather than here.
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
                    confidence: self.calculate_overall_confidence(requirements) as f64,
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
// LOCK DEPENDENCY EXTRACTION
// =============================================================================

/// Relative drift below which a hold-duration series counts as flat.
///
/// The fitted line's total rise across the observed holds is compared against
/// their mean; anything smaller than this is reported as
/// [`TrendDirection::Stable`] rather than as a direction, because a series that
/// drifts by a few percent of its own mean carries no usable direction.
const HOLD_DURATION_TREND_THRESHOLD: f64 = 0.10;

/// Share of the traced span a lock must be held for before its scope is
/// reported as an optimization opportunity.
const HOLD_SHARE_REPORTING_THRESHOLD: f64 = 0.25;

/// Observed contention rate above which a lock-free alternative is suggested.
const LOCK_FREE_SUGGESTION_THRESHOLD: f64 = 0.5;

/// True when `event_type` names a lock acquisition.
///
/// The comparison is case-insensitive: [`LockContentionAnalyzer`] emits
/// `"acquire"` while hand-written traces and fixtures use `"Acquire"`. Matching
/// only the capitalised spelling (which is what this module did before 0.2.1)
/// silently dropped every event the in-tree producer emits, so no hold duration
/// was ever measured on the live path.
fn is_acquire_event(event_type: &str) -> bool {
    event_type.eq_ignore_ascii_case("acquire")
        || event_type.eq_ignore_ascii_case("lockacquire")
        || event_type.eq_ignore_ascii_case("lockacquisition")
}

/// True when `event_type` names a lock release.
fn is_release_event(event_type: &str) -> bool {
    event_type.eq_ignore_ascii_case("release") || event_type.eq_ignore_ascii_case("lockrelease")
}

/// Classify a lock by its identifier.
///
/// [`LockEvent`] carries no lock kind, so the identifier is the only evidence a
/// trace offers. Recognised substrings map to the matching [`LockType`];
/// anything else is reported as [`LockType::Custom`] carrying the raw
/// identifier rather than defaulted to `Mutex`, which is what this module did
/// before 0.2.1 ("Default, could be inferred") for every lock in every trace.
fn classify_lock_type(lock_id: &str) -> LockType {
    let lowered = lock_id.to_ascii_lowercase();
    // Most specific spellings first: "rwlock" also contains "lock".
    if lowered.contains("rwlock") || lowered.contains("rw_lock") || lowered.contains("read_write") {
        LockType::RwLock
    } else if lowered.contains("semaphore") {
        LockType::Semaphore
    } else if lowered.contains("condvar")
        || lowered.contains("cond_var")
        || lowered.contains("condition")
    {
        LockType::CondVar
    } else if lowered.contains("barrier") {
        LockType::Barrier
    } else if lowered.contains("atomic") {
        LockType::Atomic
    } else if lowered.contains("spin") {
        LockType::SpinLock
    } else if lowered.contains("file") {
        LockType::FileLock
    } else if lowered.contains("database") || lowered.contains("db_") {
        LockType::DatabaseLock
    } else if lowered.contains("mutex") {
        LockType::Mutex
    } else {
        LockType::Custom(lock_id.to_string())
    }
}

/// Order statistic of an ascending duration slice by the nearest-rank rule.
///
/// `p` is a fraction in `(0, 1]`. The rank is `ceil(p·n)`, so the 95th
/// percentile of fewer than twenty observations is legitimately the largest of
/// them -- that is the definition, not the `p95: max, // Simplified` alias this
/// module published before 0.2.1, which reported the maximum for samples of
/// every size.
fn percentile_of_sorted(sorted: &[Duration], p: f64) -> Option<Duration> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    let rank = (p * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted.get(index).copied()
}

/// Convert seconds to a [`Duration`], reporting zero for values a duration
/// cannot represent (negative or non-finite).
fn duration_from_seconds(seconds: f64) -> Duration {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Duration::ZERO;
    }
    Duration::try_from_secs_f64(seconds).unwrap_or(Duration::MAX)
}

/// Direction of the hold-duration series, fitted by ordinary least squares.
///
/// Returns [`TrendDirection::Unknown`] for fewer than three holds: two points
/// always define a line exactly, so they cannot separate a trend from noise.
/// The fitted total rise is compared against the mean hold duration, and a
/// drift below [`HOLD_DURATION_TREND_THRESHOLD`] of the mean is reported as
/// `Stable`. Before 0.2.1 every lock in every trace was reported as `Stable`.
fn hold_duration_trend(durations: &[Duration]) -> TrendDirection {
    if durations.len() < 3 {
        return TrendDirection::Unknown;
    }
    let n = durations.len() as f64;
    let seconds: Vec<f64> = durations.iter().map(|d| d.as_secs_f64()).collect();
    let mean_index = (n - 1.0) / 2.0;
    let mean_value = seconds.iter().sum::<f64>() / n;
    if mean_value <= 0.0 {
        // Every hold was instantaneous at the clock's resolution; there is no
        // series to fit.
        return TrendDirection::Stable;
    }

    let mut covariance = 0.0;
    let mut index_variance = 0.0;
    for (index, value) in seconds.iter().enumerate() {
        let centred_index = index as f64 - mean_index;
        covariance += centred_index * (value - mean_value);
        index_variance += centred_index * centred_index;
    }
    if index_variance <= 0.0 {
        return TrendDirection::Unknown;
    }

    let slope = covariance / index_variance;
    let relative_drift = slope * (n - 1.0) / mean_value;
    if !relative_drift.is_finite() {
        return TrendDirection::Unknown;
    }
    if relative_drift > HOLD_DURATION_TREND_THRESHOLD {
        TrendDirection::Increasing
    } else if relative_drift < -HOLD_DURATION_TREND_THRESHOLD {
        TrendDirection::Decreasing
    } else {
        TrendDirection::Stable
    }
}

/// Descriptive statistics of the hold durations observed for one lock.
///
/// Every field is computed from `durations`, which is the sequence of
/// acquire-to-release intervals in the order the trace recorded them. Before
/// 0.2.1 this record was built inline with `median: mean`, `std_dev: 0`,
/// `variance: 0.0`, `p95: max`, `p99: max` and `trend: Stable`, each marked
/// "Simplified".
///
/// `variance` is the **population** variance of the observed holds, in
/// **seconds squared** -- the recorded holds are the population of holds that
/// happened, not a sample drawn from a larger one -- and `std_dev` is its
/// square root expressed as a duration. An empty input yields
/// [`DurationStatistics::default`], which reports `sample_count: 0`; that is
/// the honest record for a lock the trace never showed being held.
fn duration_statistics(durations: &[Duration]) -> DurationStatistics {
    if durations.is_empty() {
        return DurationStatistics::default();
    }

    let mut sorted = durations.to_vec();
    sorted.sort_unstable();

    let count = sorted.len();
    let total_nanos: u128 = sorted.iter().map(|d| d.as_nanos()).sum();
    let mean_nanos = total_nanos / count as u128;
    let mean = Duration::from_nanos(u64::try_from(mean_nanos).unwrap_or(u64::MAX));

    let median = if count.is_multiple_of(2) {
        let lower = sorted.get(count / 2 - 1).map(|d| d.as_nanos()).unwrap_or(0);
        let upper = sorted.get(count / 2).map(|d| d.as_nanos()).unwrap_or(0);
        Duration::from_nanos(u64::try_from((lower + upper) / 2).unwrap_or(u64::MAX))
    } else {
        sorted.get(count / 2).copied().unwrap_or_default()
    };

    let mean_seconds = sorted.iter().map(|d| d.as_secs_f64()).sum::<f64>() / count as f64;
    let variance = sorted
        .iter()
        .map(|d| {
            let deviation = d.as_secs_f64() - mean_seconds;
            deviation * deviation
        })
        .sum::<f64>()
        / count as f64;

    DurationStatistics {
        min: sorted.first().copied().unwrap_or_default(),
        max: sorted.last().copied().unwrap_or_default(),
        mean,
        median,
        std_dev: duration_from_seconds(variance.sqrt()),
        p95: percentile_of_sorted(&sorted, 0.95).unwrap_or_default(),
        p99: percentile_of_sorted(&sorted, 0.99).unwrap_or_default(),
        sample_count: count,
        variance,
        trend: hold_duration_trend(durations),
    }
}

/// Share of `lock_id`'s observed acquisition-order pairs that were also
/// observed in the opposite order.
///
/// An ordered pair `(a, b)` means some thread held `a` when it acquired `b`. A
/// pair observed in both directions is a lock-order inversion, which is the
/// precondition for a hold-and-wait deadlock; the fraction of this lock's pairs
/// that are inverted is therefore a measured deadlock-risk contribution in
/// `[0, 1]`. Before 0.2.1 the field was `0.6` for locks with more than two
/// dependents and `0.2` for every other lock, regardless of ordering.
fn ordering_inversion_rate(
    lock_id: &str,
    ordering_pairs: &std::collections::BTreeSet<(String, String)>,
) -> f64 {
    let involved: Vec<&(String, String)> = ordering_pairs
        .iter()
        .filter(|(before, after)| before == lock_id || after == lock_id)
        .collect();
    if involved.is_empty() {
        return 0.0;
    }
    let inverted = involved
        .iter()
        .filter(|(before, after)| ordering_pairs.contains(&(after.clone(), before.clone())))
        .count();
    inverted as f64 / involved.len() as f64
}

/// Build one [`LockDependency`] per lock named by `events`, measuring every
/// field from the trace.
///
/// The walk pairs each successful acquire with the next release of the same
/// lock on the same thread, which yields the hold durations; the locks a thread
/// already holds when it takes another one give the dependency and ordering
/// edges; and an acquire taken while a *different* thread holds the same lock
/// is counted as an observed contention.
///
/// [`LockEvent`]'s own documentation names a third event type, `"contention"`.
/// No producer in this crate emits one, and such an event carries no thread to
/// attribute a wait to, so the walk ignores it: contention is derived from
/// overlapping acquisitions instead, which the traces that do exist can show.
///
/// ## What each published field means
///
/// * `contention_probability` -- observed contentions / observed acquire
///   attempts. `0.0` means the trace showed no contention, which is a
///   measurement; it was the constant `0.3` before 0.2.1.
/// * `deadlock_risk_factor` -- [`ordering_inversion_rate`].
/// * `performance_impact` -- summed hold time / traced span, clamped to one
///   (concurrent holds by several threads can otherwise exceed the span). `0.0`
///   when the trace covers no measurable span. It was the constant `0.4`.
/// * `alternatives` / `optimization_opportunities` -- advice, each entry
///   emitted only when the measurement that motivates it crosses its threshold,
///   and each carrying that measurement in its text. Both were fixed lists
///   attached to every lock.
///
/// Results are ordered by lock id so that repeated analyses of the same trace
/// produce the same report; the previous `HashSet` iteration made the order
/// vary between runs.
pub(crate) fn lock_dependencies_from_events(events: &[LockEvent]) -> Vec<LockDependency> {
    use std::collections::{BTreeMap, BTreeSet};

    let mut all_locks: BTreeSet<String> = BTreeSet::new();
    let mut predecessors: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut hold_durations: BTreeMap<String, Vec<Duration>> = BTreeMap::new();
    let mut acquire_attempts: BTreeMap<String, usize> = BTreeMap::new();
    let mut contended_acquires: BTreeMap<String, usize> = BTreeMap::new();
    let mut ordering_pairs: BTreeSet<(String, String)> = BTreeSet::new();

    // Locks currently held by each thread, with the instant each was taken.
    let mut thread_locks: BTreeMap<u64, Vec<(String, Instant)>> = BTreeMap::new();
    // Threads currently holding each lock, so that an acquire arriving while
    // another thread holds it can be counted as contention.
    let mut holders: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();

    let mut span_start: Option<Instant> = None;
    let mut span_end: Option<Instant> = None;

    for event in events {
        all_locks.insert(event.lock_id.clone());
        span_start = Some(span_start.map_or(event.timestamp, |start| start.min(event.timestamp)));
        span_end = Some(span_end.map_or(event.timestamp, |end| end.max(event.timestamp)));

        if is_acquire_event(&event.event_type) {
            *acquire_attempts.entry(event.lock_id.clone()).or_default() += 1;
            if !event.success {
                // A failed acquire takes nothing and holds nothing; it is
                // still an attempt, so it stays in the denominator.
                continue;
            }

            let holder_set = holders.entry(event.lock_id.clone()).or_default();
            if holder_set.iter().any(|thread| *thread != event.thread_id) {
                *contended_acquires.entry(event.lock_id.clone()).or_default() += 1;
            }
            holder_set.insert(event.thread_id);

            let held = thread_locks.entry(event.thread_id).or_default();
            for (already_held, _) in held.iter() {
                if already_held == &event.lock_id {
                    continue;
                }
                ordering_pairs.insert((already_held.clone(), event.lock_id.clone()));
                let observed = predecessors.entry(event.lock_id.clone()).or_default();
                if !observed.contains(already_held) {
                    observed.push(already_held.clone());
                }
            }
            held.push((event.lock_id.clone(), event.timestamp));
        } else if is_release_event(&event.event_type) {
            if !event.success {
                continue;
            }
            if let Some(held) = thread_locks.get_mut(&event.thread_id) {
                if let Some(position) = held.iter().position(|(id, _)| id == &event.lock_id) {
                    let (_, acquired_at) = held.remove(position);
                    // `saturating_duration_since` keeps a trace whose
                    // timestamps run backwards from panicking.
                    hold_durations
                        .entry(event.lock_id.clone())
                        .or_default()
                        .push(event.timestamp.saturating_duration_since(acquired_at));
                }
            }
            if let Some(holder_set) = holders.get_mut(&event.lock_id) {
                holder_set.remove(&event.thread_id);
            }
        }
    }

    let span = match (span_start, span_end) {
        (Some(start), Some(end)) => end.saturating_duration_since(start),
        _ => Duration::ZERO,
    };

    let mut dependencies = Vec::with_capacity(all_locks.len());
    for lock_id in all_locks {
        let observed_predecessors = predecessors.get(&lock_id).cloned().unwrap_or_default();
        let mut dependent_locks = observed_predecessors.clone();
        dependent_locks.sort();
        dependent_locks.dedup();

        let durations = hold_durations.get(&lock_id).cloned().unwrap_or_default();
        let attempts = acquire_attempts.get(&lock_id).copied().unwrap_or(0);
        let contended = contended_acquires.get(&lock_id).copied().unwrap_or(0);
        let contention_probability =
            if attempts == 0 { 0.0 } else { contended as f64 / attempts as f64 };
        let deadlock_risk_factor = ordering_inversion_rate(&lock_id, &ordering_pairs);

        let held_total: Duration = durations.iter().sum();
        let performance_impact = if span.is_zero() {
            0.0
        } else {
            (held_total.as_secs_f64() / span.as_secs_f64()).clamp(0.0, 1.0)
        };

        let mut alternatives = Vec::new();
        if contention_probability > 0.0 {
            alternatives.push(format!(
                "fine-grained locking ({:.0}% of {} acquisitions were contended)",
                contention_probability * 100.0,
                attempts
            ));
        }
        if contention_probability >= LOCK_FREE_SUGGESTION_THRESHOLD {
            alternatives.push("lock-free data structure".to_string());
        }

        let mut optimization_opportunities = Vec::new();
        if performance_impact >= HOLD_SHARE_REPORTING_THRESHOLD {
            optimization_opportunities.push(format!(
                "Reduce lock scope: held for {:.0}% of the traced span",
                performance_impact * 100.0
            ));
        }
        if deadlock_risk_factor > 0.0 {
            optimization_opportunities.push(format!(
                "Establish a global lock ordering: {:.0}% of this lock's observed orderings were \
                 also observed inverted",
                deadlock_risk_factor * 100.0
            ));
        }

        dependencies.push(LockDependency {
            lock_type: classify_lock_type(&lock_id),
            lock_id,
            dependent_locks,
            acquisition_order: observed_predecessors,
            hold_duration_stats: duration_statistics(&durations),
            contention_probability,
            deadlock_risk_factor,
            alternatives,
            performance_impact,
            optimization_opportunities,
        });
    }

    dependencies
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one lock event; every field the extractor reads is explicit.
    fn lock_event(lock_id: &str, event_type: &str, thread_id: u64, at: Instant) -> LockEvent {
        LockEvent {
            timestamp: at,
            lock_id: lock_id.to_string(),
            event_type: event_type.to_string(),
            thread_id,
            success: true,
            ..LockEvent::default()
        }
    }

    /// One acquire/release pair for `thread_id`, held for `hold`.
    fn hold(
        lock_id: &str,
        thread_id: u64,
        start: Instant,
        hold: Duration,
        events: &mut Vec<LockEvent>,
    ) {
        events.push(lock_event(lock_id, "acquire", thread_id, start));
        events.push(lock_event(lock_id, "release", thread_id, start + hold));
    }

    /// Regression: the hold-duration record published `median: mean`,
    /// `std_dev: 0`, `variance: 0.0`, `p95: max` and `p99: max`, each marked
    /// "Simplified". The order statistics are computed from the holds now, so
    /// the median separates from the mean on a skewed sample.
    #[test]
    fn hold_duration_statistics_are_measured_order_statistics() {
        let base = Instant::now();
        let mut events = Vec::new();
        for (index, millis) in [10u64, 20, 30, 40, 500].into_iter().enumerate() {
            hold(
                "mutex_a",
                1,
                base + Duration::from_millis(1000 * index as u64),
                Duration::from_millis(millis),
                &mut events,
            );
        }

        let dependencies = lock_dependencies_from_events(&events);
        assert_eq!(dependencies.len(), 1, "one lock appears in the trace");
        let stats = &dependencies[0].hold_duration_stats;

        assert_eq!(stats.sample_count, 5, "five holds were observed");
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(500));
        assert_eq!(
            stats.median,
            Duration::from_millis(30),
            "the median is the middle hold, not the mean"
        );
        assert!(
            stats.mean > stats.median,
            "this sample is right-skewed, so mean {:?} must exceed median {:?}",
            stats.mean,
            stats.median
        );
        assert!(
            stats.std_dev > Duration::ZERO && stats.variance > 0.0,
            "a spread sample has non-zero spread, got std_dev {:?} variance {}",
            stats.std_dev,
            stats.variance
        );
        // Population variance in seconds squared, cross-checked independently.
        let seconds = [0.010f64, 0.020, 0.030, 0.040, 0.500];
        let mean = seconds.iter().sum::<f64>() / seconds.len() as f64;
        let expected =
            seconds.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / seconds.len() as f64;
        assert!(
            (stats.variance - expected).abs() < 1e-12,
            "variance {} must equal the population variance {expected} in seconds squared",
            stats.variance
        );
        assert!(
            (stats.std_dev.as_secs_f64() - expected.sqrt()).abs() < 1e-9,
            "std_dev must be the square root of the variance"
        );
    }

    /// Regression: `extract_lock_dependencies` matched `"Acquire"`/`"Release"`
    /// while `LockContentionAnalyzer::extract_lock_events` emits
    /// `"acquire"`/`"release"`, so on the live path no hold was ever paired and
    /// every lock reported `DurationStatistics::default()`.
    #[test]
    fn lowercase_trace_events_are_measured() {
        let base = Instant::now();
        let mut events = Vec::new();
        hold("mutex_a", 7, base, Duration::from_millis(25), &mut events);

        let dependencies = lock_dependencies_from_events(&events);
        assert_eq!(dependencies.len(), 1);
        assert_eq!(
            dependencies[0].hold_duration_stats.sample_count, 1,
            "the lowercase spelling the analyzer emits must be measured"
        );
        assert_eq!(
            dependencies[0].hold_duration_stats.mean,
            Duration::from_millis(25)
        );

        // The capitalised spelling used by hand-written fixtures still works.
        let mut capitalised = Vec::new();
        capitalised.push(lock_event("mutex_a", "Acquire", 7, base));
        capitalised.push(lock_event(
            "mutex_a",
            "Release",
            7,
            base + Duration::from_millis(25),
        ));
        assert_eq!(
            lock_dependencies_from_events(&capitalised)[0].hold_duration_stats.sample_count,
            1
        );
    }

    /// Regression: `contention_probability` was the constant `0.3` for every
    /// lock in every trace. It is the observed contention rate now, so a trace
    /// without overlapping acquisitions reports zero and one with them does
    /// not.
    #[test]
    fn contention_probability_is_the_observed_rate() {
        let base = Instant::now();

        // Thread 1 holds the lock while thread 2 takes it: one contended
        // acquisition out of two attempts.
        let contended = vec![
            lock_event("mutex_a", "acquire", 1, base),
            lock_event("mutex_a", "acquire", 2, base + Duration::from_millis(1)),
            lock_event("mutex_a", "release", 1, base + Duration::from_millis(2)),
            lock_event("mutex_a", "release", 2, base + Duration::from_millis(3)),
        ];
        let measured = lock_dependencies_from_events(&contended);
        assert!(
            (measured[0].contention_probability - 0.5).abs() < 1e-12,
            "one of two acquisitions was contended, got {}",
            measured[0].contention_probability
        );

        // The same two threads, serialised: no acquisition finds the lock held.
        let mut serialised = Vec::new();
        hold(
            "mutex_a",
            1,
            base,
            Duration::from_millis(1),
            &mut serialised,
        );
        hold(
            "mutex_a",
            2,
            base + Duration::from_millis(2),
            Duration::from_millis(1),
            &mut serialised,
        );
        let quiet = lock_dependencies_from_events(&serialised);
        assert_eq!(
            quiet[0].contention_probability, 0.0,
            "an uncontended trace must report no contention"
        );
        assert!(
            quiet[0].alternatives.is_empty(),
            "advice is attached only when a measurement motivates it, got {:?}",
            quiet[0].alternatives
        );
    }

    /// Regression: `deadlock_risk_factor` was `0.6` for locks with more than
    /// two dependents and `0.2` otherwise, ignoring acquisition order. It
    /// reports the observed lock-order inversion rate now.
    #[test]
    fn deadlock_risk_factor_reports_observed_inversion() {
        let base = Instant::now();

        // Thread 1 takes A then B; thread 2 takes B then A -- a full inversion.
        let inverted = vec![
            lock_event("mutex_a", "acquire", 1, base),
            lock_event("mutex_b", "acquire", 1, base + Duration::from_millis(1)),
            lock_event("mutex_b", "acquire", 2, base + Duration::from_millis(2)),
            lock_event("mutex_a", "acquire", 2, base + Duration::from_millis(3)),
        ];
        for dependency in lock_dependencies_from_events(&inverted) {
            assert!(
                (dependency.deadlock_risk_factor - 1.0).abs() < 1e-12,
                "{} took part in an inverted ordering, got {}",
                dependency.lock_id,
                dependency.deadlock_risk_factor
            );
            assert!(
                dependency
                    .optimization_opportunities
                    .iter()
                    .any(|text| text.contains("global lock ordering")),
                "an inversion must be reported, got {:?}",
                dependency.optimization_opportunities
            );
        }

        // Both threads take A then B: consistent, so no inversion.
        let consistent = vec![
            lock_event("mutex_a", "acquire", 1, base),
            lock_event("mutex_b", "acquire", 1, base + Duration::from_millis(1)),
            lock_event("mutex_a", "acquire", 2, base + Duration::from_millis(2)),
            lock_event("mutex_b", "acquire", 2, base + Duration::from_millis(3)),
        ];
        for dependency in lock_dependencies_from_events(&consistent) {
            assert_eq!(
                dependency.deadlock_risk_factor, 0.0,
                "{} was always taken in the same order",
                dependency.lock_id
            );
        }
    }

    /// Regression: `performance_impact` was the constant `0.4`. It is the share
    /// of the traced span the lock was held for now, so it tracks the trace.
    #[test]
    fn performance_impact_is_the_measured_hold_share() {
        let base = Instant::now();

        let mut long_hold = Vec::new();
        hold(
            "mutex_a",
            1,
            base,
            Duration::from_millis(50),
            &mut long_hold,
        );
        // A second event pair pushes the traced span out to 100 ms.
        long_hold.push(lock_event(
            "mutex_b",
            "acquire",
            1,
            base + Duration::from_millis(100),
        ));
        let long_measured = lock_dependencies_from_events(&long_hold);
        let long_share = long_measured
            .iter()
            .find(|d| d.lock_id == "mutex_a")
            .map(|d| d.performance_impact)
            .unwrap_or_default();
        assert!(
            (long_share - 0.5).abs() < 1e-6,
            "50 ms held over a 100 ms span is half the span, got {long_share}"
        );

        let mut short_hold = Vec::new();
        hold(
            "mutex_a",
            1,
            base,
            Duration::from_millis(5),
            &mut short_hold,
        );
        short_hold.push(lock_event(
            "mutex_b",
            "acquire",
            1,
            base + Duration::from_millis(100),
        ));
        let short_share = lock_dependencies_from_events(&short_hold)
            .iter()
            .find(|d| d.lock_id == "mutex_a")
            .map(|d| d.performance_impact)
            .unwrap_or_default();
        assert!(
            short_share < long_share,
            "a shorter hold over the same span must report a smaller share: \
             {short_share} vs {long_share}"
        );
        assert!(
            lock_dependencies_from_events(&short_hold)
                .iter()
                .all(|d| d.optimization_opportunities.is_empty()),
            "a 5% hold share is below the reporting threshold"
        );
    }

    /// Regression: every lock was published as `LockType::Mutex` ("Default,
    /// could be inferred").
    #[test]
    fn lock_type_is_classified_from_the_identifier() {
        assert_eq!(classify_lock_type("global_rwlock"), LockType::RwLock);
        assert_eq!(classify_lock_type("MUTEX_registry"), LockType::Mutex);
        assert_eq!(classify_lock_type("db_row_lock"), LockType::DatabaseLock);
        assert_eq!(classify_lock_type("spin_guard"), LockType::SpinLock);
        assert_eq!(
            classify_lock_type("resource_7"),
            LockType::Custom("resource_7".to_string()),
            "an unrecognised identifier is reported as itself, not defaulted"
        );
    }

    /// Regression: `trend` was `TrendDirection::Stable` for every lock.
    #[test]
    fn hold_duration_trend_follows_the_series() {
        let rising: Vec<Duration> = (1..=6).map(|n| Duration::from_millis(10 * n)).collect();
        assert_eq!(hold_duration_trend(&rising), TrendDirection::Increasing);

        let falling: Vec<Duration> = (1..=6).rev().map(|n| Duration::from_millis(10 * n)).collect();
        assert_eq!(hold_duration_trend(&falling), TrendDirection::Decreasing);

        let flat = vec![Duration::from_millis(20); 6];
        assert_eq!(hold_duration_trend(&flat), TrendDirection::Stable);

        let too_short = vec![Duration::from_millis(10), Duration::from_millis(90)];
        assert_eq!(
            hold_duration_trend(&too_short),
            TrendDirection::Unknown,
            "two holds cannot separate a trend from noise"
        );
    }

    /// A lock the trace never showed being held reports no samples rather than
    /// a fabricated record.
    #[test]
    fn a_lock_never_held_reports_no_samples() {
        let base = Instant::now();
        // An acquire that failed: an attempt, but nothing was ever held.
        let events = vec![LockEvent {
            timestamp: base,
            lock_id: "mutex_a".to_string(),
            event_type: "acquire".to_string(),
            thread_id: 1,
            success: false,
            ..LockEvent::default()
        }];

        let dependencies = lock_dependencies_from_events(&events);
        assert_eq!(dependencies.len(), 1);
        assert_eq!(dependencies[0].hold_duration_stats.sample_count, 0);
        assert_eq!(dependencies[0].hold_duration_stats.mean, Duration::ZERO);
        assert_eq!(dependencies[0].performance_impact, 0.0);
        assert!(dependencies[0].optimization_opportunities.is_empty());
    }

    /// The report is ordered by lock id so repeated analyses of one trace agree
    /// (the previous `HashSet` walk made the order vary between runs).
    #[test]
    fn dependencies_are_reported_in_a_stable_order() {
        let base = Instant::now();
        let events = vec![
            lock_event("zeta", "acquire", 1, base),
            lock_event("alpha", "acquire", 2, base + Duration::from_millis(1)),
            lock_event("mid", "acquire", 3, base + Duration::from_millis(2)),
        ];
        let ids: Vec<String> =
            lock_dependencies_from_events(&events).into_iter().map(|d| d.lock_id).collect();
        assert_eq!(ids, vec!["alpha", "mid", "zeta"]);
    }

    /// The locks a thread already holds when it takes another are the
    /// dependency edges, and they are recorded once each.
    #[test]
    fn dependent_locks_come_from_the_observed_nesting() {
        let base = Instant::now();
        let events = vec![
            lock_event("outer_mutex", "acquire", 1, base),
            lock_event("inner_mutex", "acquire", 1, base + Duration::from_millis(1)),
            lock_event("inner_mutex", "release", 1, base + Duration::from_millis(2)),
            lock_event("inner_mutex", "acquire", 1, base + Duration::from_millis(3)),
            lock_event("inner_mutex", "release", 1, base + Duration::from_millis(4)),
            lock_event("outer_mutex", "release", 1, base + Duration::from_millis(5)),
        ];
        let dependencies = lock_dependencies_from_events(&events);
        let inner = dependencies
            .iter()
            .find(|d| d.lock_id == "inner_mutex")
            .expect("inner lock is reported");
        assert_eq!(
            inner.dependent_locks,
            vec!["outer_mutex".to_string()],
            "the outer lock is recorded once, not once per acquisition"
        );
        let outer = dependencies
            .iter()
            .find(|d| d.lock_id == "outer_mutex")
            .expect("outer lock is reported");
        assert!(
            outer.dependent_locks.is_empty(),
            "nothing was held when the outer lock was taken"
        );
        assert_eq!(
            inner.hold_duration_stats.sample_count, 2,
            "both nested holds are measured"
        );
    }

    #[test]
    fn test_sharing_capability_to_struct_read_only() {
        let cap = sharing_capability_to_struct(&SharingCapability::ReadOnly);
        assert!(
            cap.supports_read_sharing,
            "ReadOnly should support read sharing"
        );
        assert!(
            !cap.supports_write_sharing,
            "ReadOnly should NOT support write sharing"
        );
        assert!(
            cap.safety_assessment > 0.8,
            "ReadOnly should have high safety"
        );
        assert!(
            cap.performance_overhead < 0.2,
            "ReadOnly should have low overhead"
        );
    }

    #[test]
    fn test_sharing_capability_to_struct_read_write() {
        let cap = sharing_capability_to_struct(&SharingCapability::ReadWrite);
        assert!(
            cap.supports_read_sharing,
            "ReadWrite should support read sharing"
        );
        assert!(
            cap.supports_write_sharing,
            "ReadWrite should support write sharing"
        );
        assert!(
            cap.safety_assessment < 0.95,
            "ReadWrite should have lower safety than ReadOnly"
        );
        assert!(
            cap.implementation_complexity > 0.3,
            "ReadWrite should have higher complexity"
        );
    }

    #[test]
    fn test_sharing_capability_to_struct_exclusive() {
        let cap = sharing_capability_to_struct(&SharingCapability::Exclusive);
        assert!(
            !cap.supports_read_sharing,
            "Exclusive should not support read sharing"
        );
        assert!(
            cap.safety_assessment > 0.95,
            "Exclusive access should be very safe"
        );
        assert_eq!(
            cap.max_concurrent_readers,
            Some(1),
            "Exclusive allows only 1 reader"
        );
    }

    #[test]
    fn test_estimation_config_from_pattern_config() {
        let pattern_cfg = PatternEstimationConfig {
            enabled: true,
            timeout_seconds: 30,
            max_iterations: 500,
        };
        let estimation_cfg = EstimationConfig {
            safety_margin: if pattern_cfg.enabled {
                0.2_f64.max(1.0 / (pattern_cfg.timeout_seconds as f64 + 1.0))
            } else {
                0.5
            },
            history_retention_limit: pattern_cfg.max_iterations.clamp(100, 10_000),
        };
        assert!(
            estimation_cfg.safety_margin > 0.0,
            "safety_margin should be positive"
        );
        assert!(
            estimation_cfg.safety_margin <= 0.2,
            "enabled config should use conservative margin"
        );
        assert_eq!(
            estimation_cfg.history_retention_limit, 500,
            "max_iterations maps to history limit"
        );
    }

    #[test]
    fn test_resource_constraints_populated() {
        let reqs = ConcurrencyRequirements {
            resource_constraints: vec!["cpu".to_string(), "memory".to_string()],
            ..ConcurrencyRequirements::default()
        };
        assert_eq!(
            reqs.resource_constraints.len(),
            2,
            "resource_constraints should be populated"
        );
    }
}
