//! Results Synthesizer
//!
//! Synthesizer for integrating results from all analysis modules.

use super::super::types::*;
use super::*;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use tokio::sync::RwLock as TokioRwLock;
use tracing::{debug, error, instrument};

// Explicitly import TestProfile from profiling_pipeline to avoid ambiguity
// Type alias for orchestrator's TestProfile (simpler version)
use super::orchestrator::TestProfile as OrchestratorTestProfile;

#[derive(Debug)]
pub struct ResultsSynthesizer {
    /// Synthesis algorithms configuration
    synthesis_config: Arc<TokioRwLock<SynthesisConfig>>,
    /// Synthesis statistics
    synthesis_stats: Arc<SynthesisStatistics>,
}

/// Configuration for result synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisConfig {
    /// Weighting strategy for different analysis results
    pub weighting_strategy: WeightingStrategy,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolutionStrategy,
    /// Quality thresholds
    pub quality_thresholds: QualityThresholds,
    /// Synthesis algorithms
    pub synthesis_algorithms: SynthesisAlgorithms,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            weighting_strategy: WeightingStrategy::Balanced,
            conflict_resolution: ConflictResolutionStrategy::HighestConfidence,
            quality_thresholds: QualityThresholds::default(),
            synthesis_algorithms: SynthesisAlgorithms::default(),
        }
    }
}

/// Weighting strategies for analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightingStrategy {
    Equal,
    Balanced,
    ConfidenceBased,
    AccuracyBased,
    Custom(HashMap<String, f64>),
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    HighestConfidence,
    MajorityVote,
    WeightedAverage,
    Conservative,
    Aggressive,
}

/// Quality thresholds for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum confidence score
    pub min_confidence: f64,
    /// Minimum data completeness
    pub min_completeness: f64,
    /// Maximum result variance
    pub max_variance: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            min_completeness: 0.8,
            max_variance: 0.3,
        }
    }
}

/// Synthesis algorithms configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisAlgorithms {
    /// Resource intensity synthesis algorithm
    pub resource_intensity_algorithm: ResourceSynthesisAlgorithm,
    /// Concurrency synthesis algorithm
    pub concurrency_algorithm: ConcurrencySynthesisAlgorithm,
    /// Pattern synthesis algorithm
    pub pattern_algorithm: PatternSynthesisAlgorithm,
}

impl Default for SynthesisAlgorithms {
    fn default() -> Self {
        Self {
            resource_intensity_algorithm: ResourceSynthesisAlgorithm::MaxValue,
            concurrency_algorithm: ConcurrencySynthesisAlgorithm::MaxRequirement,
            pattern_algorithm: PatternSynthesisAlgorithm::PatternMerging,
        }
    }
}

/// Resource synthesis algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceSynthesisAlgorithm {
    MaxValue,
    AverageValue,
    WeightedAverage,
    PercentileValue(u8),
}

/// Concurrency synthesis algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConcurrencySynthesisAlgorithm {
    MaxRequirement,
    AverageRequirement,
    MedianRequirement,
    SafetyMargin(f64),
}

/// Pattern synthesis algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternSynthesisAlgorithm {
    PatternMerging,
    PatternIntersection,
    PatternUnion,
    ConfidenceFiltering(f64),
}

/// Statistics for synthesis operations
#[derive(Debug, Default)]
pub struct SynthesisStatistics {
    /// Total synthesis operations
    pub total_syntheses: AtomicU64,
    /// Successful syntheses
    pub successful_syntheses: AtomicU64,
    /// Failed syntheses
    pub failed_syntheses: AtomicU64,
    /// Average synthesis time
    pub average_synthesis_time_ms: AtomicU64,
    /// Conflicts resolved
    pub conflicts_resolved: AtomicU64,
    /// Quality failures
    pub quality_failures: AtomicU64,
}

impl ResultsSynthesizer {
    /// Create a new results synthesizer
    pub async fn new(
        _component_manager: Arc<ComponentManager>,
        _cache_coordinator: Arc<CacheCoordinator>,
    ) -> Result<Self> {
        Ok(Self {
            synthesis_config: Arc::new(TokioRwLock::new(SynthesisConfig::default())),
            synthesis_stats: Arc::new(SynthesisStatistics::default()),
        })
    }

    /// Synthesize comprehensive test characteristics
    ///
    /// # Arguments
    ///
    /// * `phase_results` - Results from all analysis phases
    ///
    /// # Returns
    ///
    /// Synthesized test characteristics
    ///
    /// # Errors
    ///
    /// Returns an error if synthesis fails
    #[instrument(skip(self, phase_results))]
    pub async fn synthesize_results(
        &self,
        phase_results: HashMap<AnalysisPhase, PhaseResult>,
    ) -> Result<TestCharacteristics> {
        let start_time = Instant::now();
        self.synthesis_stats.total_syntheses.fetch_add(1, Ordering::Relaxed);

        debug!(
            "Starting result synthesis for {} phases",
            phase_results.len()
        );

        let config = self.synthesis_config.read().await;
        let result = self.perform_synthesis(&phase_results, &config).await;

        let duration = start_time.elapsed();

        match result {
            Ok(characteristics) => {
                self.synthesis_stats.successful_syntheses.fetch_add(1, Ordering::Relaxed);

                // Update average synthesis time
                let current_avg =
                    self.synthesis_stats.average_synthesis_time_ms.load(Ordering::Relaxed);
                let new_avg = if current_avg == 0 {
                    duration.as_millis() as u64
                } else {
                    (current_avg + duration.as_millis() as u64) / 2
                };
                self.synthesis_stats.average_synthesis_time_ms.store(new_avg, Ordering::Relaxed);

                debug!("Result synthesis completed in {:?}", duration);
                Ok(characteristics)
            },
            Err(e) => {
                self.synthesis_stats.failed_syntheses.fetch_add(1, Ordering::Relaxed);
                error!("Result synthesis failed: {}", e);
                Err(e)
            },
        }
    }

    /// Perform the actual synthesis
    async fn perform_synthesis(
        &self,
        phase_results: &HashMap<AnalysisPhase, PhaseResult>,
        config: &SynthesisConfig,
    ) -> Result<TestCharacteristics> {
        let mut characteristics = TestCharacteristics::default();

        // Extract and synthesize resource intensity
        if let Some(PhaseResult::ResourceAnalysis(resource_analysis)) =
            phase_results.get(&AnalysisPhase::ResourceAnalysis)
        {
            characteristics.resource_intensity = resource_analysis.clone();
        }

        // Extract and synthesize concurrency requirements
        if let Some(PhaseResult::ConcurrencyDetection(concurrency_requirements)) =
            phase_results.get(&AnalysisPhase::ConcurrencyDetection)
        {
            characteristics.concurrency_requirements = (**concurrency_requirements).clone();
        }

        // Extract and synthesize synchronization dependencies
        if let Some(PhaseResult::SynchronizationAnalysis(sync_dependencies)) =
            phase_results.get(&AnalysisPhase::SynchronizationAnalysis)
        {
            characteristics.synchronization_dependencies = sync_dependencies.clone();
        }

        // Extract and synthesize performance patterns
        if let Some(PhaseResult::PatternRecognition(patterns)) =
            phase_results.get(&AnalysisPhase::PatternRecognition)
        {
            characteristics.performance_patterns = patterns.clone();
        }

        // Integrate profiling pipeline results if available
        if let Some(PhaseResult::ProfilingPipeline(profile)) =
            phase_results.get(&AnalysisPhase::ProfilingPipeline)
        {
            self.integrate_profiling_results(&mut characteristics, profile, config).await?;
        }

        // Record real-time profiler measurements if available
        if let Some(PhaseResult::RealTimeProfiler(rt_statistics)) =
            phase_results.get(&AnalysisPhase::RealTimeProfiler)
        {
            characteristics =
                self.record_realtime_statistics(characteristics, rt_statistics.as_ref());
        }

        // Apply quality checks
        self.validate_synthesis_quality(&characteristics, config).await?;

        // Set synthesis metadata
        characteristics.analysis_metadata.timestamp = SystemTime::now();
        characteristics.analysis_metadata.version = "2.0.0".to_string();
        characteristics.analysis_metadata.confidence_score =
            self.calculate_overall_confidence(&characteristics);

        Ok(characteristics)
    }

    /// Integrate profiling pipeline results
    async fn integrate_profiling_results(
        &self,
        characteristics: &mut TestCharacteristics,
        profile: &OrchestratorTestProfile,
        config: &SynthesisConfig,
    ) -> Result<()> {
        // Integrate resource metrics if available
        if let Some(cpu) = profile.resource_metrics.get("cpu_usage_percent") {
            match config.synthesis_algorithms.resource_intensity_algorithm {
                ResourceSynthesisAlgorithm::MaxValue => {
                    characteristics.resource_intensity.cpu_intensity =
                        characteristics.resource_intensity.cpu_intensity.max(*cpu);
                },
                ResourceSynthesisAlgorithm::AverageValue => {
                    characteristics.resource_intensity.cpu_intensity =
                        (characteristics.resource_intensity.cpu_intensity + *cpu) / 2.0;
                },
                ResourceSynthesisAlgorithm::WeightedAverage => {
                    // Use weighted average with current having weight 0.7
                    characteristics.resource_intensity.cpu_intensity =
                        characteristics.resource_intensity.cpu_intensity * 0.7 + *cpu * 0.3;
                },
                ResourceSynthesisAlgorithm::PercentileValue(_) => {
                    // Use max value for percentile
                    characteristics.resource_intensity.cpu_intensity =
                        characteristics.resource_intensity.cpu_intensity.max(*cpu);
                },
            }
        }

        if let Some(mem) = profile.resource_metrics.get("memory_usage_mb") {
            match config.synthesis_algorithms.resource_intensity_algorithm {
                ResourceSynthesisAlgorithm::MaxValue => {
                    characteristics.resource_intensity.memory_intensity =
                        characteristics.resource_intensity.memory_intensity.max(*mem);
                },
                ResourceSynthesisAlgorithm::AverageValue => {
                    characteristics.resource_intensity.memory_intensity =
                        (characteristics.resource_intensity.memory_intensity + *mem) / 2.0;
                },
                ResourceSynthesisAlgorithm::WeightedAverage => {
                    // Use weighted average with current having weight 0.7
                    characteristics.resource_intensity.memory_intensity =
                        characteristics.resource_intensity.memory_intensity * 0.7 + *mem * 0.3;
                },
                ResourceSynthesisAlgorithm::PercentileValue(_) => {
                    // Use max value for percentile
                    characteristics.resource_intensity.memory_intensity =
                        characteristics.resource_intensity.memory_intensity.max(*mem);
                },
            }
        }

        Ok(())
    }

    /// Record what the real-time profiler measured.
    ///
    /// ## Changed in 0.2.1
    ///
    /// This used to take a second `TestCharacteristics` produced by the
    /// real-time-profiler phase and merge it into the synthesised result with
    /// weighted averages and confidence comparisons. That second value was
    /// always `TestCharacteristics::default()` — the phase invented it because
    /// the profiler has no method that derives test characteristics — so every
    /// "merge" mixed zeroed resource intensities and a zero confidence score
    /// into a real analysis. The phase now reports the sampling counters it
    /// genuinely measured, and this records them as notes without touching any
    /// analysed field.
    fn record_realtime_statistics(
        &self,
        mut base: TestCharacteristics,
        statistics: &ProfilingStatistics,
    ) -> TestCharacteristics {
        base.analysis_metadata.notes.push(format!(
            "real-time profiler: {} samples over {:?} (avg interval {:?}, {} active sessions)",
            statistics.total_samples,
            statistics.sampling_duration,
            statistics.average_sample_interval,
            statistics.active_sessions,
        ));
        base.analysis_metadata.notes.push(format!(
            "real-time profiler counters: {} data points processed, {} anomalies, {} insights, \
             buffer {:.1}% full, {:.2} points/s",
            statistics.data_points_processed,
            statistics.anomalies_detected,
            statistics.insights_generated,
            statistics.buffer_utilization * 100.0,
            statistics.processing_rate,
        ));
        base
    }

    /// Validate synthesis quality
    async fn validate_synthesis_quality(
        &self,
        characteristics: &TestCharacteristics,
        config: &SynthesisConfig,
    ) -> Result<()> {
        // Check confidence threshold
        if characteristics.analysis_metadata.confidence_score
            < config.quality_thresholds.min_confidence
        {
            self.synthesis_stats.quality_failures.fetch_add(1, Ordering::Relaxed);
            return Err(anyhow!(
                "Synthesis quality below confidence threshold: {} < {}",
                characteristics.analysis_metadata.confidence_score,
                config.quality_thresholds.min_confidence
            ));
        }

        // Check completeness
        let completeness = self.calculate_completeness(characteristics);
        if completeness < config.quality_thresholds.min_completeness {
            self.synthesis_stats.quality_failures.fetch_add(1, Ordering::Relaxed);
            return Err(anyhow!(
                "Synthesis quality below completeness threshold: {} < {}",
                completeness,
                config.quality_thresholds.min_completeness
            ));
        }

        Ok(())
    }

    /// Calculate completeness score
    fn calculate_completeness(&self, characteristics: &TestCharacteristics) -> f64 {
        let mut completeness_factors = 0;
        let total_factors = 5; // Total possible factors

        if characteristics.resource_intensity.cpu_intensity > 0.0 {
            completeness_factors += 1;
        }

        if characteristics.concurrency_requirements.max_threads > 0 {
            completeness_factors += 1;
        }

        // TODO: synchronization_dependencies is now Vec<String>, not struct
        if !characteristics.synchronization_dependencies.is_empty() {
            completeness_factors += 1;
        }

        // TODO: performance_patterns is now Vec<String>, not struct
        if !characteristics.performance_patterns.is_empty() {
            completeness_factors += 1;
        }

        if characteristics.analysis_metadata.confidence_score > 0.0 {
            completeness_factors += 1;
        }

        completeness_factors as f64 / total_factors as f64
    }

    /// Calculate overall confidence score
    fn calculate_overall_confidence(&self, characteristics: &TestCharacteristics) -> f64 {
        let mut confidence_sum = 0.0;
        let mut factor_count = 0;

        // Resource intensity confidence
        if characteristics.resource_intensity.cpu_intensity > 0.0 {
            confidence_sum += 0.8;
            factor_count += 1;
        }

        // Concurrency requirements confidence
        if characteristics.concurrency_requirements.max_threads > 0 {
            confidence_sum += 0.9;
            factor_count += 1;
        }

        // Synchronization dependencies confidence
        // TODO: synchronization_dependencies is now Vec<String>, not struct
        if !characteristics.synchronization_dependencies.is_empty() {
            confidence_sum += 0.7;
            factor_count += 1;
        }

        // Performance patterns confidence
        // TODO: performance_patterns is now Vec<String>, not struct
        if !characteristics.performance_patterns.is_empty() {
            confidence_sum += 0.6;
            factor_count += 1;
        }

        if factor_count > 0 {
            confidence_sum / factor_count as f64
        } else {
            0.0
        }
    }

    /// Get synthesis statistics
    pub async fn get_statistics(&self) -> SynthesisStatistics {
        SynthesisStatistics {
            total_syntheses: AtomicU64::new(
                self.synthesis_stats.total_syntheses.load(Ordering::Relaxed),
            ),
            successful_syntheses: AtomicU64::new(
                self.synthesis_stats.successful_syntheses.load(Ordering::Relaxed),
            ),
            failed_syntheses: AtomicU64::new(
                self.synthesis_stats.failed_syntheses.load(Ordering::Relaxed),
            ),
            average_synthesis_time_ms: AtomicU64::new(
                self.synthesis_stats.average_synthesis_time_ms.load(Ordering::Relaxed),
            ),
            conflicts_resolved: AtomicU64::new(
                self.synthesis_stats.conflicts_resolved.load(Ordering::Relaxed),
            ),
            quality_failures: AtomicU64::new(
                self.synthesis_stats.quality_failures.load(Ordering::Relaxed),
            ),
        }
    }
}
