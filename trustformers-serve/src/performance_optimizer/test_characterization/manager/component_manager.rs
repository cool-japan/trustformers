//! Component Manager
//!
//! Manager for component lifecycle and coordination.

use super::super::types::*;
use super::super::{profiling_pipeline::*, synchronization_analyzer::*};
use super::*;

use anyhow::Result;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::{Mutex as TokioMutex, RwLock as TokioRwLock};
use tracing::{debug, info};

/// Manager for component lifecycle and coordination
///
/// The `ComponentManager` handles the creation, configuration, and lifecycle management
/// of all analysis components, ensuring proper resource allocation and coordination.
#[derive(Debug)]
pub struct ComponentManager {
    /// Resource analyzer instance
    resource_analyzer: Arc<ResourceIntensityAnalyzer>,
    /// Concurrency detector instance
    concurrency_detector: Arc<ConcurrencyRequirementsDetector>,
    /// Synchronization analyzer instance
    synchronization_analyzer: Arc<SynchronizationAnalyzer>,
    /// Profiling pipeline instance
    profiling_pipeline: Arc<TestProfilingPipeline>,
    /// Pattern engine instance
    pattern_engine: Arc<TestPatternRecognitionEngine>,
    /// Real-time profiler instance
    real_time_profiler: Arc<RealTimeTestProfiler>,
    /// Component configurations
    configs: Arc<TokioRwLock<ComponentConfigs>>,
    /// Component health status
    health_status: Arc<TokioMutex<ComponentHealthStatus>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

/// Health status for all components
#[derive(Debug, Clone, Default)]
pub struct ComponentHealthStatus {
    /// Resource analyzer health
    pub resource_analyzer_healthy: bool,
    /// Concurrency detector health
    pub concurrency_detector_healthy: bool,
    /// Synchronization analyzer health
    pub synchronization_analyzer_healthy: bool,
    /// Profiling pipeline health
    pub profiling_pipeline_healthy: bool,
    /// Pattern engine health
    pub pattern_engine_healthy: bool,
    /// Real-time profiler health
    pub real_time_profiler_healthy: bool,
    /// Last health check timestamp
    pub last_health_check: Option<SystemTime>,
}

impl ComponentManager {
    /// Create a new component manager
    pub async fn new(configs: ComponentConfigs) -> Result<Self> {
        info!("Initializing ComponentManager");

        // Initialize all components
        let resource_analyzer = Arc::new(
            ResourceIntensityAnalyzer::new(configs.resource_analyzer_config.clone()).await?,
        );

        let concurrency_detector = Arc::new(
            ConcurrencyRequirementsDetector::new(configs.concurrency_detector_config.clone())
                .await?,
        );

        let synchronization_analyzer = Arc::new(
            SynchronizationAnalyzer::new(configs.synchronization_analyzer_config.clone()).await?,
        );

        let profiling_pipeline =
            Arc::new(TestProfilingPipeline::new(configs.profiling_pipeline_config.clone()).await?);

        let pattern_engine = Arc::new(TestPatternRecognitionEngine::new());

        let real_time_profiler = Arc::new(RealTimeTestProfiler::new(Arc::new(RwLock::new(
            configs.real_time_profiler_config.clone(),
        ))));

        let mut health_status = ComponentHealthStatus::default();
        health_status.resource_analyzer_healthy = true;
        health_status.concurrency_detector_healthy = true;
        health_status.synchronization_analyzer_healthy = true;
        health_status.profiling_pipeline_healthy = true;
        health_status.pattern_engine_healthy = true;
        health_status.real_time_profiler_healthy = true;
        health_status.last_health_check = Some(SystemTime::now());

        Ok(Self {
            resource_analyzer,
            concurrency_detector,
            synchronization_analyzer,
            profiling_pipeline,
            pattern_engine,
            real_time_profiler,
            configs: Arc::new(TokioRwLock::new(configs)),
            health_status: Arc::new(TokioMutex::new(health_status)),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Get resource analyzer reference
    pub fn get_resource_analyzer(&self) -> Arc<ResourceIntensityAnalyzer> {
        self.resource_analyzer.clone()
    }

    /// Get concurrency detector reference
    pub fn get_concurrency_detector(&self) -> Arc<ConcurrencyRequirementsDetector> {
        self.concurrency_detector.clone()
    }

    /// Get synchronization analyzer reference
    pub fn get_synchronization_analyzer(&self) -> Arc<SynchronizationAnalyzer> {
        self.synchronization_analyzer.clone()
    }

    /// Get profiling pipeline reference
    pub fn get_profiling_pipeline(&self) -> Arc<TestProfilingPipeline> {
        self.profiling_pipeline.clone()
    }

    /// Get pattern engine reference
    pub fn get_pattern_engine(&self) -> Arc<TestPatternRecognitionEngine> {
        self.pattern_engine.clone()
    }

    /// Get real-time profiler reference
    pub fn get_real_time_profiler(&self) -> Arc<RealTimeTestProfiler> {
        self.real_time_profiler.clone()
    }

    /// Check health of all components
    pub async fn check_component_health(&self) -> Result<ComponentHealthStatus> {
        let mut health_status = self.health_status.lock().await;

        // Check each component's health
        health_status.resource_analyzer_healthy = self.check_resource_analyzer_health().await;
        health_status.concurrency_detector_healthy = self.check_concurrency_detector_health().await;
        health_status.synchronization_analyzer_healthy =
            self.check_synchronization_analyzer_health().await;
        health_status.profiling_pipeline_healthy = self.check_profiling_pipeline_health().await;
        health_status.pattern_engine_healthy = self.check_pattern_engine_health().await;
        health_status.real_time_profiler_healthy = self.check_real_time_profiler_health().await;
        health_status.last_health_check = Some(SystemTime::now());

        Ok((*health_status).clone())
    }

    /// Check resource analyzer health
    async fn check_resource_analyzer_health(&self) -> bool {
        !self.shutdown.load(Ordering::Acquire)
    }

    /// Check concurrency detector health
    async fn check_concurrency_detector_health(&self) -> bool {
        !self.shutdown.load(Ordering::Acquire)
    }

    /// Check synchronization analyzer health
    async fn check_synchronization_analyzer_health(&self) -> bool {
        !self.shutdown.load(Ordering::Acquire)
    }

    /// Check profiling pipeline health
    async fn check_profiling_pipeline_health(&self) -> bool {
        !self.shutdown.load(Ordering::Acquire)
    }

    /// Check pattern engine health
    async fn check_pattern_engine_health(&self) -> bool {
        !self.shutdown.load(Ordering::Acquire)
    }

    /// Check real-time profiler health
    async fn check_real_time_profiler_health(&self) -> bool {
        !self.shutdown.load(Ordering::Acquire)
    }

    /// Update component configurations
    pub async fn update_configurations(&self, new_configs: ComponentConfigs) -> Result<()> {
        info!("Updating component configurations");

        let mut configs = self.configs.write().await;
        let configs_clone = new_configs.clone();
        *configs = new_configs;

        debug!(
            "Configuration updated with {} components configured",
            [
                configs_clone.profiler_config.is_some(),
                configs_clone.pattern_config.is_some(),
                configs_clone.resource_config.is_some(),
            ]
            .iter()
            .filter(|&&x| x)
            .count()
        );

        Ok(())
    }

    /// Shutdown all components
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down ComponentManager");

        self.shutdown.store(true, Ordering::Release);

        info!("ComponentManager shutdown completed");
        Ok(())
    }
}
