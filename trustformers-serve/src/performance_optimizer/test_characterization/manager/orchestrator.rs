//! Analysis Orchestrator
//!
//! Orchestrator for coordinating analysis phases and components.

use super::super::profiling_pipeline::*;
use super::super::types::*;
use super::*;

use anyhow::{anyhow, Result};
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{RwLock as TokioRwLock, Semaphore};
use tokio::task::spawn;
use tracing::{debug, error, instrument, warn};

/// Orchestrator for coordinating analysis phases and components
///
/// The `AnalysisOrchestrator` manages the sequencing and coordination of different analysis
/// phases, ensuring optimal resource utilization and result quality.
#[derive(Debug)]
pub struct AnalysisOrchestrator {
    /// Component manager reference
    component_manager: Arc<ComponentManager>,
    /// Analysis scheduler reference
    analysis_scheduler: Arc<AnalysisScheduler>,
    /// Cache coordinator reference
    cache_coordinator: Arc<CacheCoordinator>,
    /// Error recovery manager reference
    error_recovery_manager: Arc<ErrorRecoveryManager>,
    /// Analysis phases configuration
    phases_config: Arc<TokioRwLock<AnalysisPhasesConfig>>,
    /// Orchestration statistics
    orchestration_stats: Arc<OrchestrationStatistics>,
}

/// Configuration for analysis phases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisPhasesConfig {
    /// Phase execution order
    pub phase_order: Vec<AnalysisPhase>,
    /// Parallel execution configuration
    pub parallel_execution: ParallelExecutionConfig,
    /// Phase dependencies
    pub phase_dependencies: HashMap<AnalysisPhase, Vec<AnalysisPhase>>,
    /// Phase timeouts
    pub phase_timeouts: HashMap<AnalysisPhase, Duration>,
}

impl Default for AnalysisPhasesConfig {
    fn default() -> Self {
        Self {
            phase_order: vec![
                AnalysisPhase::ResourceAnalysis,
                AnalysisPhase::ConcurrencyDetection,
                AnalysisPhase::SynchronizationAnalysis,
                AnalysisPhase::PatternRecognition,
                AnalysisPhase::ProfilingPipeline,
            ],
            parallel_execution: ParallelExecutionConfig::default(),
            phase_dependencies: HashMap::new(),
            phase_timeouts: HashMap::new(),
        }
    }
}

/// Analysis phases
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum AnalysisPhase {
    ResourceAnalysis,
    ConcurrencyDetection,
    SynchronizationAnalysis,
    PatternRecognition,
    ProfilingPipeline,
    RealTimeProfiler,
}

/// Parallel execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelExecutionConfig {
    /// Maximum parallel phases
    pub max_parallel_phases: usize,
    /// Phase grouping strategy
    pub grouping_strategy: PhaseGroupingStrategy,
    /// Resource allocation per phase
    pub resource_allocation: HashMap<AnalysisPhase, ResourceAllocation>,
}

impl Default for ParallelExecutionConfig {
    fn default() -> Self {
        Self {
            max_parallel_phases: 4,
            grouping_strategy: PhaseGroupingStrategy::Independent,
            resource_allocation: HashMap::new(),
        }
    }
}

/// Phase grouping strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhaseGroupingStrategy {
    Sequential,
    Independent,
    ResourceBased,
    DependencyBased,
}

/// Resource allocation for phases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// CPU cores allocated
    pub cpu_cores: usize,
    /// Memory allocation in MB
    pub memory_mb: usize,
    /// I/O priority
    pub io_priority: u8,
}

/// Statistics for orchestration
#[derive(Debug, Default)]
pub struct OrchestrationStatistics {
    /// Total orchestrations
    pub total_orchestrations: AtomicU64,
    /// Successful orchestrations
    pub successful_orchestrations: AtomicU64,
    /// Failed orchestrations
    pub failed_orchestrations: AtomicU64,
    /// Average orchestration duration
    pub average_orchestration_duration_ms: AtomicU64,
    /// Phase execution counts
    pub phase_execution_counts: Arc<Mutex<HashMap<AnalysisPhase, u64>>>,
    /// Phase average durations
    pub phase_average_durations: Arc<Mutex<HashMap<AnalysisPhase, u64>>>,
}

impl AnalysisOrchestrator {
    /// Create a new analysis orchestrator
    pub async fn new(
        component_manager: Arc<ComponentManager>,
        analysis_scheduler: Arc<AnalysisScheduler>,
        cache_coordinator: Arc<CacheCoordinator>,
        error_recovery_manager: Arc<ErrorRecoveryManager>,
    ) -> Result<Self> {
        Ok(Self {
            component_manager,
            analysis_scheduler,
            cache_coordinator,
            error_recovery_manager,
            phases_config: Arc::new(TokioRwLock::new(AnalysisPhasesConfig::default())),
            orchestration_stats: Arc::new(OrchestrationStatistics::default()),
        })
    }

    /// Orchestrate comprehensive analysis
    #[instrument(skip(self, test_data, options))]
    pub async fn orchestrate_analysis(
        &self,
        test_data: &TestExecutionData,
        options: &ProfilingOptions,
    ) -> Result<TestCharacteristics> {
        let start_time = Instant::now();
        self.orchestration_stats.total_orchestrations.fetch_add(1, Ordering::Relaxed);

        debug!(
            "Starting analysis orchestration for test: {}",
            test_data.test_id
        );

        let phases_config = self.phases_config.read().await;
        let result = self.execute_analysis_phases(test_data, options, &phases_config).await;

        let duration = start_time.elapsed();

        match result {
            Ok(characteristics) => {
                self.orchestration_stats
                    .successful_orchestrations
                    .fetch_add(1, Ordering::Relaxed);

                let current_avg = self
                    .orchestration_stats
                    .average_orchestration_duration_ms
                    .load(Ordering::Relaxed);
                let new_avg = if current_avg == 0 {
                    duration.as_millis() as u64
                } else {
                    (current_avg + duration.as_millis() as u64) / 2
                };
                self.orchestration_stats
                    .average_orchestration_duration_ms
                    .store(new_avg, Ordering::Relaxed);

                debug!(
                    "Analysis orchestration completed successfully in {:?}",
                    duration
                );
                Ok(characteristics)
            },
            Err(e) => {
                self.orchestration_stats.failed_orchestrations.fetch_add(1, Ordering::Relaxed);
                error!("Analysis orchestration failed: {}", e);
                Err(e)
            },
        }
    }

    /// Execute analysis phases according to configuration
    async fn execute_analysis_phases(
        &self,
        test_data: &TestExecutionData,
        options: &ProfilingOptions,
        config: &AnalysisPhasesConfig,
    ) -> Result<TestCharacteristics> {
        let mut phase_results = HashMap::new();

        match config.parallel_execution.grouping_strategy {
            PhaseGroupingStrategy::Sequential => {
                self.execute_phases_sequentially(test_data, options, config, &mut phase_results)
                    .await?;
            },
            PhaseGroupingStrategy::Independent => {
                self.execute_phases_parallel(test_data, options, config, &mut phase_results)
                    .await?;
            },
            _ => {
                self.execute_phases_sequentially(test_data, options, config, &mut phase_results)
                    .await?;
            },
        }

        self.synthesize_phase_results(phase_results).await
    }

    /// Execute phases sequentially
    async fn execute_phases_sequentially(
        &self,
        test_data: &TestExecutionData,
        options: &ProfilingOptions,
        config: &AnalysisPhasesConfig,
        phase_results: &mut HashMap<AnalysisPhase, PhaseResult>,
    ) -> Result<()> {
        for phase in &config.phase_order {
            let start_time = Instant::now();

            let result = self.execute_single_phase(phase, test_data, options).await;

            let duration = start_time.elapsed();
            self.update_phase_statistics(phase, duration).await;

            match result {
                Ok(phase_result) => {
                    phase_results.insert(phase.clone(), phase_result);
                },
                Err(e) => {
                    if let Ok(recovered_result) = self
                        .error_recovery_manager
                        .recover_from_phase_error(phase, &e, test_data, options)
                        .await
                    {
                        phase_results.insert(phase.clone(), recovered_result);
                    } else {
                        return Err(e);
                    }
                },
            }
        }

        Ok(())
    }

    /// Execute phases in parallel
    async fn execute_phases_parallel(
        &self,
        test_data: &TestExecutionData,
        options: &ProfilingOptions,
        config: &AnalysisPhasesConfig,
        phase_results: &mut HashMap<AnalysisPhase, PhaseResult>,
    ) -> Result<()> {
        let semaphore = Arc::new(Semaphore::new(
            config.parallel_execution.max_parallel_phases,
        ));
        let mut tasks = Vec::new();

        for phase in &config.phase_order {
            let permit = semaphore.clone().acquire_owned().await?;
            let phase_clone = phase.clone();
            let test_data_clone = test_data.clone();
            let options_clone = options.clone();
            let component_manager = self.component_manager.clone();
            let error_recovery_manager = self.error_recovery_manager.clone();
            let orchestration_stats = self.orchestration_stats.clone();

            let task = spawn(async move {
                let _permit = permit;
                let start_time = Instant::now();

                let result = Self::execute_phase_task(
                    &phase_clone,
                    &test_data_clone,
                    &options_clone,
                    component_manager,
                )
                .await;

                let duration = start_time.elapsed();

                {
                    let mut counts = orchestration_stats
                        .phase_execution_counts
                        .lock()
                        .map_err(|_| anyhow!("Lock poisoned"))?;
                    *counts.entry(phase_clone.clone()).or_insert(0) += 1;

                    let mut durations = orchestration_stats
                        .phase_average_durations
                        .lock()
                        .map_err(|_| anyhow!("Lock poisoned"))?;
                    let current_avg = durations.get(&phase_clone).cloned().unwrap_or(0);
                    let new_avg = if current_avg == 0 {
                        duration.as_millis() as u64
                    } else {
                        (current_avg + duration.as_millis() as u64) / 2
                    };
                    durations.insert(phase_clone.clone(), new_avg);
                }

                match result {
                    Ok(phase_result) => Ok((phase_clone, phase_result)),
                    Err(e) => {
                        if let Ok(recovered_result) = error_recovery_manager
                            .recover_from_phase_error(
                                &phase_clone,
                                &e,
                                &test_data_clone,
                                &options_clone,
                            )
                            .await
                        {
                            Ok((phase_clone, recovered_result))
                        } else {
                            Err(e)
                        }
                    },
                }
            });

            tasks.push(task);
        }

        let results = try_join_all(tasks).await?;

        for result in results {
            let (phase, phase_result) = result?;
            phase_results.insert(phase, phase_result);
        }

        Ok(())
    }

    /// Execute a single phase task
    async fn execute_phase_task(
        phase: &AnalysisPhase,
        test_data: &TestExecutionData,
        options: &ProfilingOptions,
        component_manager: Arc<ComponentManager>,
    ) -> Result<PhaseResult> {
        match phase {
            AnalysisPhase::ResourceAnalysis => {
                let analyzer = component_manager.get_resource_analyzer();
                let analysis = analyzer.analyze_resource_intensity(&test_data.test_id).await?;
                Ok(PhaseResult::ResourceAnalysis(analysis))
            },
            AnalysisPhase::ConcurrencyDetection => {
                let detector = component_manager.get_concurrency_detector();
                let requirements = detector.detect_concurrency_requirements(test_data).await?;
                Ok(PhaseResult::ConcurrencyDetection(Box::new(requirements)))
            },
            AnalysisPhase::SynchronizationAnalysis => {
                let analyzer = component_manager.get_synchronization_analyzer();
                let metadata = TestMetadata {
                    test_id: test_data.test_id.clone(),
                    test_name: test_data.test_id.clone(),
                    test_suite: String::from("default"),
                    tags: Vec::new(),
                    author: String::from("system"),
                    created_at: chrono::Utc::now(),
                    description: String::from("Test synchronization analysis"),
                };
                let analysis = analyzer.analyze_test_synchronization(&metadata).await?;
                let sync_strings = vec![format!("{:?}", analysis)];
                Ok(PhaseResult::SynchronizationAnalysis(sync_strings))
            },
            AnalysisPhase::PatternRecognition => {
                let engine = component_manager.get_pattern_engine();
                let patterns = engine.recognize_test_patterns(test_data)?;
                let pattern_strings: Vec<String> =
                    patterns.iter().map(|p| format!("{:?}", p)).collect();
                Ok(PhaseResult::PatternRecognition(pattern_strings))
            },
            AnalysisPhase::ProfilingPipeline => {
                let pipeline = component_manager.get_profiling_pipeline();
                let request = ProfilingRequest {
                    test_id: test_data.test_id.clone(),
                    test_data: test_data.clone(),
                    profiling_options: options.clone(),
                    priority: ProfilingPriority::Normal,
                    context: HashMap::new(),
                    timestamp: chrono::Utc::now(),
                    stages: Vec::new(),
                };
                let profiling_result =
                    pipeline.execute_profiling_pipeline(test_data.test_id.clone(), request).await?;
                let mut resource_metrics = HashMap::new();
                resource_metrics.insert(
                    "cpu_usage_percent".to_string(),
                    profiling_result.characteristics.resource_intensity.cpu_intensity,
                );
                resource_metrics.insert(
                    "memory_usage_mb".to_string(),
                    profiling_result.characteristics.resource_intensity.memory_intensity,
                );
                let profile = TestProfile { resource_metrics };
                Ok(PhaseResult::ProfilingPipeline(profile))
            },
            AnalysisPhase::RealTimeProfiler => {
                let profiler = component_manager.get_real_time_profiler();
                profiler.start_profiling(&test_data.test_id)?;

                tokio::time::sleep(Duration::from_millis(100)).await;

                profiler.stop_profiling(&test_data.test_id)?;

                let characteristics = TestCharacteristics::default();
                Ok(PhaseResult::RealTimeProfiler(Box::new(characteristics)))
            },
        }
    }

    /// Execute a single phase
    async fn execute_single_phase(
        &self,
        phase: &AnalysisPhase,
        test_data: &TestExecutionData,
        options: &ProfilingOptions,
    ) -> Result<PhaseResult> {
        Self::execute_phase_task(phase, test_data, options, self.component_manager.clone()).await
    }

    /// Update phase statistics
    async fn update_phase_statistics(&self, phase: &AnalysisPhase, duration: Duration) {
        {
            let mut counts =
                self.orchestration_stats.phase_execution_counts.lock().expect("Lock poisoned");
            *counts.entry(phase.clone()).or_insert(0) += 1;
        }

        {
            let mut durations =
                self.orchestration_stats.phase_average_durations.lock().expect("Lock poisoned");
            let current_avg = durations.get(phase).cloned().unwrap_or(0);
            let new_avg = if current_avg == 0 {
                duration.as_millis() as u64
            } else {
                (current_avg + duration.as_millis() as u64) / 2
            };
            durations.insert(phase.clone(), new_avg);
        }
    }

    /// Synthesize results from all phases
    async fn synthesize_phase_results(
        &self,
        phase_results: HashMap<AnalysisPhase, PhaseResult>,
    ) -> Result<TestCharacteristics> {
        let mut characteristics = TestCharacteristics::default();

        for (phase, result) in phase_results {
            match (phase, result) {
                (AnalysisPhase::ResourceAnalysis, PhaseResult::ResourceAnalysis(analysis)) => {
                    characteristics.resource_intensity = analysis;
                },
                (
                    AnalysisPhase::ConcurrencyDetection,
                    PhaseResult::ConcurrencyDetection(requirements),
                ) => {
                    characteristics.concurrency_requirements = *requirements.clone();
                },
                (
                    AnalysisPhase::SynchronizationAnalysis,
                    PhaseResult::SynchronizationAnalysis(dependencies),
                ) => {
                    characteristics.synchronization_dependencies = dependencies;
                },
                (AnalysisPhase::PatternRecognition, PhaseResult::PatternRecognition(patterns)) => {
                    characteristics.performance_patterns = patterns;
                },
                (AnalysisPhase::ProfilingPipeline, PhaseResult::ProfilingPipeline(profile)) => {
                    if let Some(cpu) = profile.resource_metrics.get("cpu_usage_percent") {
                        characteristics.resource_intensity.cpu_intensity = *cpu;
                    }
                    if let Some(mem) = profile.resource_metrics.get("memory_usage_mb") {
                        characteristics.resource_intensity.memory_intensity = *mem;
                    }
                },
                (
                    AnalysisPhase::RealTimeProfiler,
                    PhaseResult::RealTimeProfiler(rt_characteristics),
                ) => {
                    characteristics =
                        self.merge_characteristics(characteristics, *rt_characteristics);
                },
                _ => {
                    warn!("Mismatched phase and result type");
                },
            }
        }

        characteristics.analysis_metadata.timestamp = SystemTime::now();
        characteristics.analysis_metadata.version = "1.0.0".to_string();
        characteristics.analysis_metadata.confidence_score =
            self.calculate_confidence_score(&characteristics);

        Ok(characteristics)
    }

    /// Merge two test characteristics
    fn merge_characteristics(
        &self,
        mut base: TestCharacteristics,
        other: TestCharacteristics,
    ) -> TestCharacteristics {
        base.resource_intensity.cpu_intensity = base
            .resource_intensity
            .cpu_intensity
            .max(other.resource_intensity.cpu_intensity);
        base.resource_intensity.memory_intensity = base
            .resource_intensity
            .memory_intensity
            .max(other.resource_intensity.memory_intensity);
        base.resource_intensity.io_intensity =
            base.resource_intensity.io_intensity.max(other.resource_intensity.io_intensity);
        base.resource_intensity.network_intensity = base
            .resource_intensity
            .network_intensity
            .max(other.resource_intensity.network_intensity);

        base.concurrency_requirements.max_threads = base
            .concurrency_requirements
            .max_threads
            .max(other.concurrency_requirements.max_threads);
        base.concurrency_requirements.max_concurrent_instances = base
            .concurrency_requirements
            .max_concurrent_instances
            .max(other.concurrency_requirements.max_concurrent_instances);

        base.synchronization_dependencies.extend(other.synchronization_dependencies);
        base.performance_patterns.extend(other.performance_patterns);

        base
    }

    /// Calculate confidence score for characteristics
    fn calculate_confidence_score(&self, characteristics: &TestCharacteristics) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;

        if characteristics.resource_intensity.cpu_intensity > 0.0 {
            score += 0.8;
            factors += 1;
        }

        if characteristics.concurrency_requirements.max_threads > 0 {
            score += 0.9;
            factors += 1;
        }

        if !characteristics.synchronization_dependencies.is_empty() {
            score += 0.7;
            factors += 1;
        }

        if !characteristics.performance_patterns.is_empty() {
            score += 0.6;
            factors += 1;
        }

        if factors > 0 {
            score / factors as f64
        } else {
            0.0
        }
    }
}

/// Result from a single analysis phase
#[derive(Debug, Clone)]
pub enum PhaseResult {
    ResourceAnalysis(ResourceIntensity),
    ConcurrencyDetection(Box<ConcurrencyRequirements>),
    SynchronizationAnalysis(Vec<String>),
    PatternRecognition(Vec<String>),
    ProfilingPipeline(TestProfile),
    RealTimeProfiler(Box<TestCharacteristics>),
}

/// Test profile for profiling pipeline results
#[derive(Debug, Clone, Default)]
pub struct TestProfile {
    pub resource_metrics: HashMap<String, f64>,
}
