//! Main MobilePerformanceProfiler implementation
//!
//! Core profiler methods for session management, metrics collection,
//! bottleneck detection, optimization suggestions, and data export.

use super::profiler_components::get_platform_capabilities;
use super::profiler_types::*;
use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

use crate::device_info::{MobileDeviceDetector, MobileDeviceInfo, ThermalState};
use crate::mobile_performance_profiler::analysis::*;
use crate::mobile_performance_profiler::collector::{CollectionStatistics, MobileMetricsCollector};
use crate::mobile_performance_profiler::config::MobileProfilerConfig;
use crate::mobile_performance_profiler::export::*;
use crate::mobile_performance_profiler::monitoring::*;
use crate::mobile_performance_profiler::session::*;
use crate::mobile_performance_profiler::types::*;

impl MobilePerformanceProfiler {
    /// Create a new mobile performance profiler instance
    ///
    /// # Arguments
    ///
    /// * `config` - Profiler configuration
    ///
    /// # Returns
    ///
    /// Returns a new profiler instance or an error if initialization fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = fast_test_config();
    /// let profiler = MobilePerformanceProfiler::new(config)?;
    /// ```
    pub fn new(config: MobileProfilerConfig) -> Result<Self> {
        info!("Initializing mobile performance profiler");

        // Detect device capabilities
        let device_info =
            MobileDeviceDetector::detect().context("Failed to detect mobile device information")?;

        debug!("Detected device: {:?}", device_info);

        // Initialize metrics collector
        let metrics_collector = MobileMetricsCollector::new(config.clone())
            .context("Failed to initialize metrics collector")?;

        // Initialize all subsystems
        let session_tracker = ProfilingSession::new(device_info.clone())?;
        let bottleneck_detector = BottleneckDetector::new(config.clone())?;
        let optimization_engine = OptimizationEngine::new(config.clone())?;
        let real_time_monitor = RealTimeMonitor::new(config.clone())?;
        let export_manager = ProfilerExportManager::new(config.clone())?;
        let alert_manager = AlertManager::new(config.clone())?;
        let performance_analyzer = PerformanceAnalyzer::new(config.clone())?;

        let profiling_state = ProfilingState {
            is_active: false,
            current_session_id: None,
            start_time: None,
            total_duration: Duration::ZERO,
            events_recorded: 0,
            snapshots_taken: 0,
            last_error: None,
        };

        info!("Mobile performance profiler initialized successfully");

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            session_tracker: Arc::new(Mutex::new(session_tracker)),
            metrics_collector: Arc::new(Mutex::new(metrics_collector)),
            bottleneck_detector: Arc::new(Mutex::new(bottleneck_detector)),
            optimization_engine: Arc::new(Mutex::new(optimization_engine)),
            real_time_monitor: Arc::new(Mutex::new(real_time_monitor)),
            export_manager: Arc::new(Mutex::new(export_manager)),
            alert_manager: Arc::new(Mutex::new(alert_manager)),
            performance_analyzer: Arc::new(Mutex::new(performance_analyzer)),
            profiling_state: Arc::new(RwLock::new(profiling_state)),
            _background_workers: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Start a new profiling session
    ///
    /// Initializes all profiling subsystems and begins collecting performance data.
    ///
    /// # Returns
    ///
    /// Returns the session ID on success, or an error if profiling cannot be started.
    ///
    /// # Errors
    ///
    /// * `ProfilerError::AlreadyActive` - If profiling is already active
    /// * `ProfilerError::InitializationFailed` - If subsystem initialization fails
    ///
    /// # Example
    ///
    /// ```rust
    /// let session_id = profiler.start_profiling()?;
    /// println!("Started profiling session: {}", session_id);
    /// ```
    pub fn start_profiling(&self) -> Result<String> {
        info!("Starting profiling session");

        // Check if profiling is already active
        {
            let state = self.profiling_state.read().expect("RwLock poisoned");
            if state.is_active {
                warn!("Profiling session already active");
                return Err(anyhow::anyhow!("Profiling is already active"));
            }
        }

        // Check if profiling is enabled
        {
            let config = self.config.read().expect("RwLock poisoned");
            if !config.enabled {
                warn!("Profiling is disabled in configuration");
                return Err(anyhow::anyhow!("Profiling is disabled"));
            }
        }

        // Start session tracking
        let session_id = {
            let mut session = self.session_tracker.lock().expect("Lock poisoned");
            session.start_session().context("Failed to start profiling session")?
        };

        // Start metrics collection
        {
            let collector = self.metrics_collector.lock().expect("Lock poisoned");
            collector.start_collection().context("Failed to start metrics collection")?;
        }

        // Start real-time monitoring if enabled
        {
            let config = self.config.read().expect("RwLock poisoned");
            if config.real_time_monitoring.enabled {
                let mut monitor = self.real_time_monitor.lock().expect("Lock poisoned");
                monitor.start_monitoring().context("Failed to start real-time monitoring")?;
            }
        }

        // Update profiling state
        {
            let mut state = self.profiling_state.write().expect("RwLock poisoned");
            state.is_active = true;
            state.current_session_id = Some(session_id.clone());
            state.start_time = Some(Instant::now());
            state.events_recorded = 0;
            state.snapshots_taken = 0;
            state.last_error = None;
        }

        info!("Profiling session started: {}", session_id);
        Ok(session_id)
    }

    /// Stop the current profiling session
    ///
    /// Stops all profiling subsystems and returns the collected profiling data.
    ///
    /// # Returns
    ///
    /// Returns comprehensive profiling data or an error if stopping fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// let profiling_data = profiler.stop_profiling()?;
    /// println!("Collected {} metrics snapshots", profiling_data.metrics.len());
    /// ```
    pub fn stop_profiling(&self) -> Result<ProfilingData> {
        info!("Stopping profiling session");

        // Check if profiling is active
        let session_id = {
            let state = self.profiling_state.read().expect("RwLock poisoned");
            if !state.is_active {
                warn!("No active profiling session to stop");
                return Err(anyhow::anyhow!("No active profiling session"));
            }
            state.current_session_id.clone()
        };

        // Stop session tracking
        {
            let mut session = self.session_tracker.lock().expect("Lock poisoned");
            session.end_session().context("Failed to end profiling session")?;
        }

        // Stop metrics collection
        {
            let collector = self.metrics_collector.lock().expect("Lock poisoned");
            collector.stop_collection().context("Failed to stop metrics collection")?;
        }

        // Stop real-time monitoring
        {
            let mut monitor = self.real_time_monitor.lock().expect("Lock poisoned");
            monitor.stop_monitoring().context("Failed to stop real-time monitoring")?;
        }

        // Generate comprehensive profiling data
        let profiling_data =
            self.generate_profiling_data().context("Failed to generate profiling data")?;

        // Auto-export if configured
        {
            let config = self.config.read().expect("RwLock poisoned");
            if config.export_config.auto_export {
                let export_manager = self.export_manager.lock().expect("Lock poisoned");
                if let Err(e) = export_manager.export_data(&profiling_data) {
                    warn!("Auto-export failed: {}", e);
                }
            }
        }

        // Update profiling state
        {
            let mut state = self.profiling_state.write().expect("RwLock poisoned");
            state.is_active = false;
            state.current_session_id = None;
            if let Some(start_time) = state.start_time {
                state.total_duration += start_time.elapsed();
            }
            state.start_time = None;
        }

        info!("Profiling session stopped: {:?}", session_id);
        Ok(profiling_data)
    }

    /// Pause the current profiling session
    ///
    /// Temporarily suspends data collection while maintaining session state.
    pub fn pause_profiling(&self) -> Result<()> {
        info!("Pausing profiling session");

        {
            let state = self.profiling_state.read().expect("RwLock poisoned");
            if !state.is_active {
                return Err(anyhow::anyhow!("No active profiling session to pause"));
            }
        }

        // Pause metrics collection
        {
            let collector = self.metrics_collector.lock().expect("Lock poisoned");
            collector.pause_collection()?;
        }

        // Pause real-time monitoring
        {
            let mut monitor = self.real_time_monitor.lock().expect("Lock poisoned");
            monitor.pause_monitoring()?;
        }

        info!("Profiling session paused");
        Ok(())
    }

    /// Resume a paused profiling session
    ///
    /// Resumes data collection from a paused state.
    pub fn resume_profiling(&self) -> Result<()> {
        info!("Resuming profiling session");

        {
            let state = self.profiling_state.read().expect("RwLock poisoned");
            if !state.is_active {
                return Err(anyhow::anyhow!("No active profiling session to resume"));
            }
        }

        // Resume metrics collection
        {
            let collector = self.metrics_collector.lock().expect("Lock poisoned");
            collector.resume_collection()?;
        }

        // Resume real-time monitoring
        {
            let mut monitor = self.real_time_monitor.lock().expect("Lock poisoned");
            monitor.resume_monitoring()?;
        }

        info!("Profiling session resumed");
        Ok(())
    }

    /// Record a profiling event
    ///
    /// Records a significant event during profiling with optional timing information.
    ///
    /// # Arguments
    ///
    /// * `event_type` - Type of event (e.g., "inference_start", "model_load")
    /// * `duration_ms` - Optional duration in milliseconds
    ///
    /// # Example
    ///
    /// ```rust
    /// profiler.record_inference_event("model_load", Some(250.0))?;
    /// profiler.record_inference_event("inference_start", None)?;
    /// ```
    pub fn record_inference_event(&self, event_type: &str, duration_ms: Option<f64>) -> Result<()> {
        debug!(
            "Recording inference event: {} ({:?}ms)",
            event_type, duration_ms
        );

        let event = ProfilingEvent {
            event_id: format!("event_{}", chrono::Utc::now().timestamp_millis()),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64,
            event_type: EventType::InferenceStart, // Default to InferenceStart, should be mapped properly
            category: "inference".to_string(),
            description: format!("Inference event: {}", event_type),
            data: EventData {
                payload: HashMap::new(),
                metrics: None,
            },
            metadata: HashMap::new(),
            tags: vec!["inference".to_string()],
            thread_id: 0, // Thread ID will be set by the collector
            duration_ms,
        };

        {
            let mut session = self.session_tracker.lock().expect("Lock poisoned");
            session.add_event(event);
        }

        // Update event counter
        {
            let mut state = self.profiling_state.write().expect("RwLock poisoned");
            state.events_recorded += 1;
        }

        Ok(())
    }

    /// Get current performance metrics snapshot
    ///
    /// Returns the most recent metrics snapshot including CPU, memory, GPU,
    /// network, thermal, and battery metrics.
    ///
    /// # Returns
    ///
    /// Current metrics snapshot or error if collection is not active.
    pub fn get_current_metrics(&self) -> Result<MobileMetricsSnapshot> {
        debug!("Getting current metrics snapshot");

        let collector = self.metrics_collector.lock().expect("Lock poisoned");
        collector
            .get_current_snapshot()
            .context("Failed to get current metrics snapshot")
    }

    /// Get comprehensive collection statistics
    ///
    /// Returns detailed statistics about the metrics collection process.
    pub fn get_collection_stats(&self) -> Result<CollectionStatistics> {
        let collector = self.metrics_collector.lock().expect("Lock poisoned");
        Ok(collector.get_collection_stats()?)
    }

    /// Detect current performance bottlenecks
    ///
    /// Analyzes recent metrics to identify active performance bottlenecks.
    ///
    /// # Returns
    ///
    /// Vector of detected bottlenecks sorted by severity.
    pub fn detect_bottlenecks(&self) -> Result<Vec<PerformanceBottleneck>> {
        debug!("Detecting performance bottlenecks");

        let detector = self.bottleneck_detector.lock().expect("Lock poisoned");
        let bottlenecks = detector.get_active_bottlenecks();

        debug!("Detected {} bottlenecks", bottlenecks.len());
        Ok(bottlenecks)
    }

    /// Get optimization suggestions
    ///
    /// Returns AI-generated optimization suggestions based on current
    /// performance patterns and detected bottlenecks.
    ///
    /// # Returns
    ///
    /// Vector of optimization suggestions ranked by potential impact.
    pub fn get_optimization_suggestions(&self) -> Result<Vec<OptimizationSuggestion>> {
        debug!("Getting optimization suggestions");

        let engine = self.optimization_engine.lock().expect("Lock poisoned");
        let suggestions = engine.get_active_suggestions();

        debug!("Generated {} optimization suggestions", suggestions.len());
        Ok(suggestions)
    }

    /// Get active performance alerts
    ///
    /// Returns currently active performance alerts that require attention.
    pub fn get_active_alerts(&self) -> Result<Vec<PerformanceAlert>> {
        let manager = self.alert_manager.lock().expect("Lock poisoned");
        Ok(manager.get_active_alerts())
    }

    /// Get comprehensive system health assessment
    ///
    /// Returns overall system health status including component-specific
    /// health scores and recommendations.
    pub fn get_system_health(&self) -> Result<SystemHealth> {
        debug!("Getting system health assessment");

        let analyzer = self.performance_analyzer.lock().expect("Lock poisoned");
        analyzer.get_current_health().context("Failed to get system health assessment")
    }

    /// Export profiling data in specified format
    ///
    /// # Arguments
    ///
    /// * `format` - Export format (JSON, CSV, HTML, etc.)
    ///
    /// # Returns
    ///
    /// Path to exported file or error if export fails.
    pub fn export_data(&self, format: ExportFormat) -> Result<String> {
        info!("Exporting profiling data in format: {:?}", format);

        let profiling_data = self
            .generate_profiling_data()
            .context("Failed to generate profiling data for export")?;

        let manager = self.export_manager.lock().expect("Lock poisoned");
        let export_path = manager
            .export_data(&profiling_data)
            .context("Failed to export profiling data")?;

        info!("Profiling data exported to: {}", export_path);
        Ok(export_path)
    }

    /// Update profiler configuration
    ///
    /// Hot-reloads the profiler configuration without stopping the current session.
    ///
    /// # Arguments
    ///
    /// * `new_config` - New profiler configuration
    pub fn update_config(&self, new_config: MobileProfilerConfig) -> Result<()> {
        info!("Updating profiler configuration");

        // Validate new configuration
        Self::validate_config(&new_config).context("Invalid profiler configuration")?;

        // Update configuration
        {
            let mut config = self.config.write().expect("RwLock poisoned");
            *config = new_config.clone();
        }

        // Propagate configuration updates to subsystems
        {
            let collector = self.metrics_collector.lock().expect("Lock poisoned");
            collector.update_config(new_config.clone())?;
        }

        {
            let mut detector = self.bottleneck_detector.lock().expect("Lock poisoned");
            detector.update_config(new_config.clone())?;
        }

        {
            let mut engine = self.optimization_engine.lock().expect("Lock poisoned");
            engine.update_config(new_config.clone())?;
        }

        {
            let mut monitor = self.real_time_monitor.lock().expect("Lock poisoned");
            monitor.update_config(new_config.clone())?;
        }

        info!("Profiler configuration updated successfully");
        Ok(())
    }

    /// Check if profiling is currently active
    ///
    /// # Returns
    ///
    /// `true` if profiling is active, `false` otherwise.
    pub fn is_profiling_active(&self) -> bool {
        let state = self.profiling_state.read().expect("RwLock poisoned");
        state.is_active
    }

    /// Get current profiling state information
    ///
    /// Returns comprehensive information about the current profiling state.
    pub fn get_profiling_state(&self) -> ProfilingState {
        let state = self.profiling_state.read().expect("RwLock poisoned");
        state.clone()
    }

    /// Generate performance report
    ///
    /// Creates a comprehensive human-readable performance report.
    ///
    /// # Returns
    ///
    /// HTML performance report as a string.
    pub fn generate_performance_report(&self) -> Result<String> {
        info!("Generating performance report");

        let profiling_data = self
            .generate_profiling_data()
            .context("Failed to generate profiling data for report")?;

        let manager = self.export_manager.lock().expect("Lock poisoned");
        manager
            .generate_report(&profiling_data)
            .context("Failed to generate performance report")
    }

    /// Perform health check on the profiler system
    ///
    /// Returns system health status and diagnostic information.
    pub fn health_check(&self) -> Result<SystemHealth> {
        debug!("Performing profiler health check");

        let analyzer = self.performance_analyzer.lock().expect("Lock poisoned");
        analyzer.get_current_health().context("Failed to perform health check")
    }

    /// Get profiler capabilities and supported features
    ///
    /// Returns information about what the profiler can monitor and analyze.
    pub fn get_capabilities(&self) -> Result<ProfilerCapabilities> {
        debug!("Getting profiler capabilities");

        // Create capabilities based on current configuration and platform
        let config = self.config.read().expect("RwLock poisoned");
        Ok(ProfilerCapabilities {
            memory_profiling: config.memory_profiling.enabled,
            cpu_profiling: config.cpu_profiling.enabled,
            gpu_profiling: config.gpu_profiling.enabled,
            network_profiling: config.network_profiling.enabled,
            thermal_monitoring: config.cpu_profiling.thermal_monitoring,
            battery_monitoring: true,
            real_time_monitoring: config.real_time_monitoring.enabled,
            platform_specific: get_platform_capabilities(),
        })
    }

    /// Take a snapshot of current performance metrics
    ///
    /// Captures current system state for analysis.
    pub fn take_snapshot(&self) -> Result<MobileMetricsSnapshot> {
        debug!("Taking performance snapshot");

        let collector = self.metrics_collector.lock().expect("Lock poisoned");
        collector.get_current_snapshot().context("Failed to take performance snapshot")
    }

    /// Assess overall system health
    ///
    /// Provides comprehensive health assessment of the mobile system.
    pub fn assess_system_health(&self) -> Result<SystemHealth> {
        debug!("Assessing system health");

        let analyzer = self.performance_analyzer.lock().expect("Lock poisoned");
        analyzer.get_current_health().context("Failed to assess system health")
    }

    // =============================================================================
    // PRIVATE HELPER METHODS
    // =============================================================================

    /// Generate comprehensive profiling data
    fn generate_profiling_data(&self) -> Result<ProfilingData> {
        debug!("Generating comprehensive profiling data");

        // Collect session information
        let session_info = {
            let session = self.session_tracker.lock().expect("Lock poisoned");
            session.get_session_info()?
        };

        // Collect all metrics snapshots
        let metrics = {
            let collector = self.metrics_collector.lock().expect("Lock poisoned");
            collector.get_all_snapshots()
        };

        // Collect all events
        let events = {
            let session = self.session_tracker.lock().expect("Lock poisoned");
            session.get_all_events()
        };

        // Get detected bottlenecks
        let bottlenecks = {
            let detector = self.bottleneck_detector.lock().expect("Lock poisoned");
            detector.get_all_bottlenecks()
        };

        // Get optimization suggestions
        let suggestions = {
            let engine = self.optimization_engine.lock().expect("Lock poisoned");
            engine.get_all_suggestions()
        };

        // Generate summary statistics
        let summary = self.calculate_profiling_summary(&metrics, &events, &bottlenecks)?;

        // Get system health assessment
        let system_health = {
            let analyzer = self.performance_analyzer.lock().expect("Lock poisoned");
            analyzer.get_current_health()?
        };

        Ok(ProfilingData {
            session_info,
            metrics,
            events,
            bottlenecks,
            suggestions,
            summary,
            system_health,
            export_timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64,
            profiler_version: "1.0.0".to_string(),
        })
    }

    /// Calculate comprehensive profiling summary
    fn calculate_profiling_summary(
        &self,
        metrics: &[MobileMetricsSnapshot],
        events: &[ProfilingEvent],
        bottlenecks: &[PerformanceBottleneck],
    ) -> Result<ProfilingSummary> {
        if metrics.is_empty() {
            return Ok(ProfilingSummary::default());
        }

        // Calculate inference statistics
        let inference_events: Vec<_> =
            events.iter().filter(|e| e.category == "inference").collect();

        let total_inferences = inference_events.len() as u64;
        let avg_inference_time_ms =
            inference_events.iter().filter_map(|e| e.duration_ms).sum::<f64>()
                / total_inferences.max(1) as f64;

        // Calculate resource usage statistics
        let peak_memory_mb = metrics
            .iter()
            .map(|m| m.memory.heap_used_mb + m.memory.native_used_mb)
            .fold(0.0f32, f32::max);

        let avg_cpu_usage =
            metrics.iter().map(|m| m.cpu.usage_percent).sum::<f32>() / metrics.len() as f32;

        let avg_gpu_usage =
            metrics.iter().map(|m| m.gpu.usage_percent).sum::<f32>() / metrics.len() as f32;

        // Calculate battery consumption
        let battery_consumed_mah =
            metrics.iter().map(|m| m.battery.power_consumption_mw).sum::<f32>() / 1000.0; // Convert mW to mAh estimate

        // Count thermal events
        let thermal_events =
            metrics.iter().filter(|m| m.thermal.throttling_level > 0.0).count() as u32;

        // Calculate overall performance score
        let performance_score = self.calculate_performance_score(metrics, bottlenecks)?;

        Ok(ProfilingSummary {
            total_inferences,
            avg_inference_time_ms,
            peak_memory_mb,
            avg_cpu_usage_percent: avg_cpu_usage,
            avg_gpu_usage_percent: avg_gpu_usage,
            battery_consumed_mah,
            thermal_events,
            performance_score,
            total_events: events.len() as u64,
            total_bottlenecks: bottlenecks.len() as u64,
            session_duration_ms: self.get_session_duration_ms(),
        })
    }

    /// Calculate overall performance score
    fn calculate_performance_score(
        &self,
        metrics: &[MobileMetricsSnapshot],
        bottlenecks: &[PerformanceBottleneck],
    ) -> Result<f32> {
        if metrics.is_empty() {
            return Ok(50.0); // Neutral score
        }

        let latest_metrics = &metrics[metrics.len() - 1];

        // Base score from resource utilization (0-100)
        let memory_score = 100.0
            - (latest_metrics.memory.heap_used_mb / latest_metrics.memory.heap_total_mb.max(1.0))
                * 100.0;
        let cpu_score = 100.0 - latest_metrics.cpu.usage_percent;
        let gpu_score = 100.0 - latest_metrics.gpu.usage_percent;
        let thermal_score = match latest_metrics.thermal.thermal_state {
            ThermalState::Nominal => 100.0,
            ThermalState::Fair => 80.0,
            ThermalState::Serious => 60.0,
            ThermalState::Critical => 20.0,
            ThermalState::Emergency => 5.0,
            ThermalState::Shutdown => 0.0,
        };

        // Calculate weighted average
        let base_score =
            (memory_score * 0.3 + cpu_score * 0.3 + gpu_score * 0.2 + thermal_score * 0.2)
                .max(0.0)
                .min(100.0);

        // Apply bottleneck penalties
        let bottleneck_penalty = bottlenecks
            .iter()
            .map(|b| match b.severity {
                BottleneckSeverity::Low => 2.0,
                BottleneckSeverity::Medium => 5.0,
                BottleneckSeverity::High => 10.0,
                BottleneckSeverity::Critical => 20.0,
            })
            .sum::<f32>();

        Ok((base_score - bottleneck_penalty).max(0.0).min(100.0))
    }

    /// Get current session duration in milliseconds
    fn get_session_duration_ms(&self) -> u64 {
        let state = self.profiling_state.read().expect("RwLock poisoned");
        if let Some(start_time) = state.start_time {
            start_time.elapsed().as_millis() as u64
        } else {
            0
        }
    }

    /// Validate profiler configuration
    pub(crate) fn validate_config(config: &MobileProfilerConfig) -> Result<()> {
        // Validate sampling configuration
        if config.sampling.interval_ms == 0 {
            return Err(anyhow::anyhow!("Sampling interval must be greater than 0"));
        }

        if config.sampling.max_samples == 0 {
            return Err(anyhow::anyhow!("Max samples must be greater than 0"));
        }

        // Validate memory profiling configuration
        if config.memory_profiling.stack_trace_depth > 100 {
            warn!(
                "Large stack trace depth may impact performance: {}",
                config.memory_profiling.stack_trace_depth
            );
        }

        // Validate export configuration
        if config.export_config.compression_level > 9 {
            return Err(anyhow::anyhow!("Compression level must be between 0-9"));
        }

        Ok(())
    }
}
