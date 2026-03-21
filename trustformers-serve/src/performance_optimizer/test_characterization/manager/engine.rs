//! Test Characterization Engine
//!
//! Main orchestrator that coordinates all specialized analysis modules.

use super::super::profiling_pipeline::*;
use super::super::types::*;
use super::*;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock as TokioRwLock;
use tracing::{error, info, instrument};

use crate::parallel_execution_engine::SchedulingConfig;
use crate::performance_optimizer::test_characterization::types::data_management::CacheConfig;
// Explicitly import PatternRecognitionConfig from patterns to avoid ambiguity
use crate::performance_optimizer::test_characterization::types::patterns::PatternRecognitionConfig;

/// Configuration for error recovery behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecoveryConfig {
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Enable automatic recovery
    pub auto_recovery_enabled: bool,
    /// Circuit breaker threshold (number of failures before opening circuit)
    pub circuit_breaker_threshold: usize,
    /// Circuit breaker reset timeout in seconds
    pub circuit_breaker_reset_timeout_seconds: u64,
}

impl Default for ErrorRecoveryConfig {
    fn default() -> Self {
        Self {
            max_retries: ERROR_RECOVERY_MAX_RETRIES,
            retry_delay_ms: 1000,
            auto_recovery_enabled: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_reset_timeout_seconds: 60,
        }
    }
}

/// Main orchestrator that coordinates all specialized analysis modules
///
/// The `TestCharacterizationEngine` serves as the central coordination point for all test
/// characterization activities. It manages the lifecycle of all analysis components,
/// coordinates caching and configuration, schedules analysis tasks intelligently,
/// and synthesizes results from all modules into comprehensive reports.
#[derive(Debug)]
pub struct TestCharacterizationEngine {
    /// Analysis orchestrator for coordinating analysis phases
    analysis_orchestrator: Arc<AnalysisOrchestrator>,
    /// Component manager for lifecycle control
    component_manager: Arc<ComponentManager>,
    /// Results synthesizer for result integration
    results_synthesizer: Arc<ResultsSynthesizer>,
    /// Cache coordinator for coordinated caching
    cache_coordinator: Arc<CacheCoordinator>,
    /// Configuration manager for centralized configuration
    configuration_manager: Arc<ConfigurationManager>,
    /// Analysis scheduler for intelligent scheduling
    analysis_scheduler: Arc<AnalysisScheduler>,
    /// Performance coordinator for monitoring
    performance_coordinator: Arc<PerformanceCoordinator>,
    /// Error recovery manager for centralized error handling
    error_recovery_manager: Arc<ErrorRecoveryManager>,
    /// Reporting coordinator for comprehensive reporting
    reporting_coordinator: Arc<ReportingCoordinator>,
    /// Main configuration
    config: Arc<TokioRwLock<TestCharacterizationConfig>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Engine statistics
    stats: Arc<EngineStatistics>,
}

/// Statistics for the characterization engine
#[derive(Debug, Default)]
pub struct EngineStatistics {
    /// Total analyses performed
    pub total_analyses: AtomicU64,
    /// Total successful analyses
    pub successful_analyses: AtomicU64,
    /// Total failed analyses
    pub failed_analyses: AtomicU64,
    /// Total cache hits
    pub cache_hits: AtomicU64,
    /// Total cache misses
    pub cache_misses: AtomicU64,
    /// Average analysis duration in milliseconds
    pub average_analysis_duration_ms: AtomicU64,
    /// Total errors recovered
    pub errors_recovered: AtomicU64,
    /// Active analysis tasks
    pub active_analyses: AtomicUsize,
}

/// Configuration for the test characterization engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Maximum concurrent analyses
    pub max_concurrent_analyses: usize,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Configuration refresh interval
    pub config_refresh_interval_seconds: u64,
    /// Performance monitoring interval
    pub performance_monitoring_interval_ms: u64,
    /// Error recovery configuration
    pub error_recovery_config: ErrorRecoveryConfig,
    /// Analysis scheduling configuration
    #[serde(skip)]
    pub scheduling_config: SchedulingConfig,
    /// Cache configuration
    #[serde(skip)]
    pub cache_config: CacheConfig,
    /// Component configurations
    pub component_configs: ComponentConfigs,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_concurrent_analyses: MAX_CONCURRENT_ANALYSES,
            cache_ttl_seconds: CACHE_TTL_SECONDS,
            config_refresh_interval_seconds: CONFIG_REFRESH_INTERVAL_SECONDS,
            performance_monitoring_interval_ms: PERFORMANCE_MONITORING_INTERVAL_MS,
            error_recovery_config: ErrorRecoveryConfig::default(),
            scheduling_config: SchedulingConfig::default(),
            cache_config: CacheConfig::default(),
            component_configs: ComponentConfigs::default(),
        }
    }
}

/// Component configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentConfigs {
    /// Resource analyzer configuration
    pub resource_analyzer_config: ResourceAnalyzerConfig,
    /// Concurrency detector configuration
    pub concurrency_detector_config: ConcurrencyDetectorConfig,
    /// Synchronization analyzer configuration
    pub synchronization_analyzer_config:
        super::super::synchronization_analyzer::SynchronizationAnalyzerConfig,
    /// Profiling pipeline configuration
    pub profiling_pipeline_config: ProfilingPipelineConfig,
    /// Pattern engine configuration
    pub pattern_engine_config: PatternRecognitionConfig,
    /// Real-time profiler configuration
    pub real_time_profiler_config: RealTimeProfilerConfig,
    /// Optional profiler configuration
    pub profiler_config: Option<RealTimeProfilerConfig>,
    /// Optional pattern configuration
    pub pattern_config: Option<PatternRecognitionConfig>,
    /// Optional resource configuration
    pub resource_config: Option<ResourceAnalyzerConfig>,
}

impl Default for ComponentConfigs {
    fn default() -> Self {
        Self {
            resource_analyzer_config: ResourceAnalyzerConfig::default(),
            concurrency_detector_config: ConcurrencyDetectorConfig::default(),
            synchronization_analyzer_config:
                super::super::synchronization_analyzer::SynchronizationAnalyzerConfig::default(),
            profiling_pipeline_config: ProfilingPipelineConfig::default(),
            pattern_engine_config: PatternRecognitionConfig::default(),
            real_time_profiler_config: RealTimeProfilerConfig::default(),
            profiler_config: None,
            pattern_config: None,
            resource_config: None,
        }
    }
}

impl TestCharacterizationEngine {
    /// Create a new test characterization engine
    ///
    /// # Arguments
    ///
    /// * `config` - Test characterization configuration
    ///
    /// # Returns
    ///
    /// A new test characterization engine instance
    ///
    /// # Errors
    ///
    /// Returns an error if any component fails to initialize
    #[instrument(skip(config))]
    pub async fn new(config: TestCharacterizationConfig) -> Result<Self> {
        info!("Initializing TestCharacterizationEngine");

        let engine_config = EngineConfig::default();
        let stats = Arc::new(EngineStatistics::default());

        // Create component manager first
        let component_manager =
            Arc::new(ComponentManager::new(engine_config.component_configs.clone()).await?);

        // Create cache coordinator
        let cache_coordinator =
            Arc::new(CacheCoordinator::new(engine_config.cache_config.clone()).await?);

        // Create configuration manager
        let configuration_manager =
            Arc::new(ConfigurationManager::new(config.clone(), engine_config.clone()).await?);

        // Create error recovery manager
        let error_recovery_manager =
            Arc::new(ErrorRecoveryManager::new(engine_config.error_recovery_config.clone()).await?);

        // Create analysis scheduler
        let analysis_scheduler = Arc::new(
            AnalysisScheduler::new(
                engine_config.scheduling_config.clone(),
                component_manager.clone(),
                error_recovery_manager.clone(),
            )
            .await?,
        );

        // Create performance coordinator
        let performance_coordinator = Arc::new(
            PerformanceCoordinator::new(
                engine_config.performance_monitoring_interval_ms,
                stats.clone(),
            )
            .await?,
        );

        // Create analysis orchestrator
        let analysis_orchestrator = Arc::new(
            AnalysisOrchestrator::new(
                component_manager.clone(),
                analysis_scheduler.clone(),
                cache_coordinator.clone(),
                error_recovery_manager.clone(),
            )
            .await?,
        );

        // Create results synthesizer
        let results_synthesizer = Arc::new(
            ResultsSynthesizer::new(component_manager.clone(), cache_coordinator.clone()).await?,
        );

        // Create reporting coordinator
        let reporting_coordinator = Arc::new(
            ReportingCoordinator::new(
                results_synthesizer.clone(),
                performance_coordinator.clone(),
                stats.clone(),
            )
            .await?,
        );

        let engine = Self {
            analysis_orchestrator,
            component_manager,
            results_synthesizer,
            cache_coordinator,
            configuration_manager,
            analysis_scheduler,
            performance_coordinator,
            error_recovery_manager,
            reporting_coordinator,
            config: Arc::new(TokioRwLock::new(config)),
            shutdown: Arc::new(AtomicBool::new(false)),
            stats,
        };

        // Start background tasks
        engine.start_background_tasks().await?;

        info!("TestCharacterizationEngine initialized successfully");
        Ok(engine)
    }

    /// Start background tasks for the engine
    async fn start_background_tasks(&self) -> Result<()> {
        // Start performance monitoring
        self.performance_coordinator.start_monitoring().await?;

        // Start configuration refresh
        self.configuration_manager.start_refresh_task().await?;

        // Start cache maintenance
        self.cache_coordinator.start_maintenance_task().await?;

        // Start error recovery monitoring
        self.error_recovery_manager.start_monitoring().await?;

        Ok(())
    }

    /// Perform comprehensive test characterization
    ///
    /// This is the main entry point for test characterization. It coordinates all analysis
    /// modules to provide comprehensive test characteristics.
    ///
    /// # Arguments
    ///
    /// * `test_data` - Test execution data to analyze
    /// * `options` - Profiling options for the analysis
    ///
    /// # Returns
    ///
    /// Comprehensive test characteristics
    ///
    /// # Errors
    ///
    /// Returns an error if the analysis fails
    #[instrument(skip(self, test_data, options))]
    pub async fn characterize_test(
        &self,
        test_data: &TestExecutionData,
        options: &ProfilingOptions,
    ) -> Result<TestCharacteristics> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(anyhow!("Engine is shutting down"));
        }

        let start_time = Instant::now();
        self.stats.total_analyses.fetch_add(1, Ordering::Relaxed);
        self.stats.active_analyses.fetch_add(1, Ordering::Relaxed);

        info!(
            "Starting test characterization for test: {}",
            test_data.test_id
        );

        // Try to get from cache first
        if let Some(cached_result) =
            self.cache_coordinator.get_test_characteristics(&test_data.test_id).await?
        {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            self.stats.active_analyses.fetch_sub(1, Ordering::Relaxed);
            info!("Cache hit for test characterization: {}", test_data.test_id);
            return Ok(cached_result);
        }

        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);

        // Perform comprehensive analysis
        let result = self.analysis_orchestrator.orchestrate_analysis(test_data, options).await;

        let duration = start_time.elapsed();
        self.stats.active_analyses.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok(characteristics) => {
                self.stats.successful_analyses.fetch_add(1, Ordering::Relaxed);

                // Update average duration
                let current_avg = self.stats.average_analysis_duration_ms.load(Ordering::Relaxed);
                let new_avg = if current_avg == 0 {
                    duration.as_millis() as u64
                } else {
                    (current_avg + duration.as_millis() as u64) / 2
                };
                self.stats.average_analysis_duration_ms.store(new_avg, Ordering::Relaxed);

                // Cache the result
                self.cache_coordinator
                    .cache_test_characteristics(&test_data.test_id, &characteristics)
                    .await?;

                info!(
                    "Test characterization completed successfully for test: {} in {:?}",
                    test_data.test_id, duration
                );

                Ok(characteristics)
            },
            Err(e) => {
                self.stats.failed_analyses.fetch_add(1, Ordering::Relaxed);
                error!(
                    "Test characterization failed for test: {}: {}",
                    test_data.test_id, e
                );
                Err(e)
            },
        }
    }

    /// Start real-time profiling for a test
    ///
    /// # Arguments
    ///
    /// * `test_id` - Identifier of the test to profile
    ///
    /// # Returns
    ///
    /// Profile session ID for tracking
    ///
    /// # Errors
    ///
    /// Returns an error if profiling cannot be started
    #[instrument(skip(self))]
    pub async fn start_real_time_profiling(&self, test_id: &str) -> Result<String> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(anyhow!("Engine is shutting down"));
        }

        info!("Starting real-time profiling for test: {}", test_id);

        self.component_manager.get_real_time_profiler().start_profiling(test_id)?;
        Ok(test_id.to_string())
    }

    /// Stop real-time profiling and get results
    ///
    /// # Arguments
    ///
    /// * `profile_id` - Profile session ID
    ///
    /// # Returns
    ///
    /// Test characteristics from real-time profiling
    ///
    /// # Errors
    ///
    /// Returns an error if profiling cannot be stopped
    #[instrument(skip(self))]
    pub async fn stop_real_time_profiling(&self, profile_id: &str) -> Result<TestCharacteristics> {
        info!("Stopping real-time profiling: {}", profile_id);

        self.component_manager.get_real_time_profiler().stop_profiling(profile_id)?;
        Ok(TestCharacteristics::default())
    }

    /// Generate comprehensive analysis report
    ///
    /// # Arguments
    ///
    /// * `test_id` - Test identifier
    ///
    /// # Returns
    ///
    /// Comprehensive analysis report
    ///
    /// # Errors
    ///
    /// Returns an error if report generation fails
    #[instrument(skip(self))]
    pub async fn generate_comprehensive_report(
        &self,
        test_id: &str,
    ) -> Result<ComprehensiveReport> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(anyhow!("Engine is shutting down"));
        }

        info!("Generating comprehensive report for test: {}", test_id);

        self.reporting_coordinator.generate_comprehensive_report(test_id).await
    }

    /// Get engine statistics
    ///
    /// # Returns
    ///
    /// Current engine statistics
    pub async fn get_statistics(&self) -> EngineStatistics {
        EngineStatistics {
            total_analyses: AtomicU64::new(self.stats.total_analyses.load(Ordering::Relaxed)),
            successful_analyses: AtomicU64::new(
                self.stats.successful_analyses.load(Ordering::Relaxed),
            ),
            failed_analyses: AtomicU64::new(self.stats.failed_analyses.load(Ordering::Relaxed)),
            cache_hits: AtomicU64::new(self.stats.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.stats.cache_misses.load(Ordering::Relaxed)),
            average_analysis_duration_ms: AtomicU64::new(
                self.stats.average_analysis_duration_ms.load(Ordering::Relaxed),
            ),
            errors_recovered: AtomicU64::new(self.stats.errors_recovered.load(Ordering::Relaxed)),
            active_analyses: AtomicUsize::new(self.stats.active_analyses.load(Ordering::Relaxed)),
        }
    }

    /// Update engine configuration
    ///
    /// # Arguments
    ///
    /// * `new_config` - New configuration to apply
    ///
    /// # Errors
    ///
    /// Returns an error if configuration update fails
    #[instrument(skip(self, new_config))]
    pub async fn update_configuration(&self, new_config: TestCharacterizationConfig) -> Result<()> {
        info!("Updating engine configuration");

        self.configuration_manager.update_configuration(new_config).await
    }

    /// Get current configuration
    ///
    /// # Returns
    ///
    /// Current configuration
    pub async fn get_configuration(&self) -> TestCharacterizationConfig {
        self.config.read().await.clone()
    }

    /// Shutdown the engine gracefully
    ///
    /// # Errors
    ///
    /// Returns an error if shutdown fails
    #[instrument(skip(self))]
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down TestCharacterizationEngine");

        self.shutdown.store(true, Ordering::Release);

        // Stop all components
        self.component_manager.shutdown().await?;
        self.cache_coordinator.shutdown().await?;
        self.configuration_manager.shutdown().await?;
        self.analysis_scheduler.shutdown().await?;
        self.performance_coordinator.shutdown().await?;
        self.error_recovery_manager.shutdown().await?;
        self.reporting_coordinator.shutdown().await?;

        info!("TestCharacterizationEngine shutdown completed");
        Ok(())
    }
}
