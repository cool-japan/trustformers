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
    /// # fn main() -> anyhow::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::{MobilePerformanceProfiler, MobileProfilerConfig};
    /// let config = MobileProfilerConfig::default();
    /// let profiler = MobilePerformanceProfiler::new(config)?;
    /// # let _ = profiler;
    /// # Ok(())
    /// # }
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
        let optimization_engine =
            crate::mobile_performance_profiler::optimization::OptimizationEngine::new(
                OptimizationEngineConfig::default(),
            )?;
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
    /// # fn main() -> anyhow::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::{MobilePerformanceProfiler, MobileProfilerConfig};
    /// # let config = MobileProfilerConfig::default();
    /// # let profiler = MobilePerformanceProfiler::new(config)?;
    /// let session_id = profiler.start_profiling()?;
    /// println!("Started profiling session: {}", session_id);
    /// # Ok(())
    /// # }
    /// ```
    pub fn start_profiling(&self) -> Result<String> {
        info!("Starting profiling session");

        // Check if profiling is already active
        {
            let state = self.profiling_state.read().unwrap_or_else(|p| p.into_inner());
            if state.is_active {
                warn!("Profiling session already active");
                return Err(anyhow::anyhow!("Profiling is already active"));
            }
        }

        // Check if profiling is enabled
        {
            let config = self.config.read().unwrap_or_else(|p| p.into_inner());
            if !config.enabled {
                warn!("Profiling is disabled in configuration");
                return Err(anyhow::anyhow!("Profiling is disabled"));
            }
        }

        // Start session tracking
        let session_id = {
            let mut session = self.session_tracker.lock().unwrap_or_else(|p| p.into_inner());
            session.start_session().context("Failed to start profiling session")?
        };

        // Start metrics collection
        {
            let collector = self.metrics_collector.lock().unwrap_or_else(|p| p.into_inner());
            collector.start_collection().context("Failed to start metrics collection")?;
        }

        // Start real-time monitoring if enabled
        {
            let config = self.config.read().unwrap_or_else(|p| p.into_inner());
            if config.real_time_monitoring.enabled {
                let mut monitor = self.real_time_monitor.lock().unwrap_or_else(|p| p.into_inner());
                monitor.start_monitoring().context("Failed to start real-time monitoring")?;
            }
        }

        // Update profiling state
        {
            let mut state = self.profiling_state.write().unwrap_or_else(|p| p.into_inner());
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
    /// # fn main() -> anyhow::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::{MobilePerformanceProfiler, MobileProfilerConfig};
    /// # let config = MobileProfilerConfig::default();
    /// # let profiler = MobilePerformanceProfiler::new(config)?;
    /// # profiler.start_profiling()?;
    /// let profiling_data = profiler.stop_profiling()?;
    /// println!("Collected {} metrics snapshots", profiling_data.metrics.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn stop_profiling(&self) -> Result<ProfilingData> {
        info!("Stopping profiling session");

        // Check if profiling is active
        let session_id = {
            let state = self.profiling_state.read().unwrap_or_else(|p| p.into_inner());
            if !state.is_active {
                warn!("No active profiling session to stop");
                return Err(anyhow::anyhow!("No active profiling session"));
            }
            state.current_session_id.clone()
        };

        // Stop session tracking
        {
            let mut session = self.session_tracker.lock().unwrap_or_else(|p| p.into_inner());
            session.end_session().context("Failed to end profiling session")?;
        }

        // Stop metrics collection
        {
            let collector = self.metrics_collector.lock().unwrap_or_else(|p| p.into_inner());
            collector.stop_collection().context("Failed to stop metrics collection")?;
        }

        // Stop real-time monitoring
        {
            let mut monitor = self.real_time_monitor.lock().unwrap_or_else(|p| p.into_inner());
            monitor.stop_monitoring().context("Failed to stop real-time monitoring")?;
        }

        // Generate comprehensive profiling data
        let profiling_data =
            self.generate_profiling_data().context("Failed to generate profiling data")?;

        // Auto-export if configured
        {
            let config = self.config.read().unwrap_or_else(|p| p.into_inner());
            if config.export_config.auto_export {
                let export_manager = self.export_manager.lock().unwrap_or_else(|p| p.into_inner());
                if let Err(e) = export_manager.export_data(&profiling_data) {
                    warn!("Auto-export failed: {}", e);
                }
            }
        }

        // Update profiling state
        {
            let mut state = self.profiling_state.write().unwrap_or_else(|p| p.into_inner());
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
            let state = self.profiling_state.read().unwrap_or_else(|p| p.into_inner());
            if !state.is_active {
                return Err(anyhow::anyhow!("No active profiling session to pause"));
            }
        }

        // Pause metrics collection
        {
            let collector = self.metrics_collector.lock().unwrap_or_else(|p| p.into_inner());
            collector.pause_collection()?;
        }

        // Pause real-time monitoring
        {
            let mut monitor = self.real_time_monitor.lock().unwrap_or_else(|p| p.into_inner());
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
            let state = self.profiling_state.read().unwrap_or_else(|p| p.into_inner());
            if !state.is_active {
                return Err(anyhow::anyhow!("No active profiling session to resume"));
            }
        }

        // Resume metrics collection
        {
            let collector = self.metrics_collector.lock().unwrap_or_else(|p| p.into_inner());
            collector.resume_collection()?;
        }

        // Resume real-time monitoring
        {
            let mut monitor = self.real_time_monitor.lock().unwrap_or_else(|p| p.into_inner());
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
    /// # fn main() -> anyhow::Result<()> {
    /// # use trustformers_mobile::mobile_performance_profiler::{MobilePerformanceProfiler, MobileProfilerConfig};
    /// # let config = MobileProfilerConfig::default();
    /// # let profiler = MobilePerformanceProfiler::new(config)?;
    /// profiler.record_inference_event("model_load", Some(250.0))?;
    /// profiler.record_inference_event("inference_start", None)?;
    /// # Ok(())
    /// # }
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
            let mut session = self.session_tracker.lock().unwrap_or_else(|p| p.into_inner());
            session.add_event(event);
        }

        // Update event counter
        {
            let mut state = self.profiling_state.write().unwrap_or_else(|p| p.into_inner());
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

        let collector = self.metrics_collector.lock().unwrap_or_else(|p| p.into_inner());
        collector
            .get_current_snapshot()
            .context("Failed to get current metrics snapshot")
    }

    /// Get comprehensive collection statistics
    ///
    /// Returns detailed statistics about the metrics collection process.
    pub fn get_collection_stats(&self) -> Result<CollectionStatistics> {
        let collector = self.metrics_collector.lock().unwrap_or_else(|p| p.into_inner());
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

        // Evaluate the detection rules against the collector's current
        // snapshot. This call used to return `get_active_bottlenecks()` from a
        // detector whose rule list was constructed empty and never populated,
        // so it could only ever return nothing -- a clean bill of health that
        // no measurement backed.
        let metrics = self.measure_now()?;

        let mut detector = self.bottleneck_detector.lock().unwrap_or_else(|p| p.into_inner());
        detector.analyze(&metrics).context("Failed to evaluate bottleneck rules")?;
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

        let bottlenecks = self.detect_bottlenecks()?;
        let metrics = self.measure_now()?;

        let mut engine = self.optimization_engine.lock().unwrap_or_else(|p| p.into_inner());
        let suggestions = engine
            .generate_suggestions(&metrics, &bottlenecks)
            .context("Failed to generate optimization suggestions")?;

        debug!("Generated {} optimization suggestions", suggestions.len());
        Ok(suggestions)
    }

    /// Get active performance alerts
    ///
    /// Returns currently active performance alerts that require attention.
    pub fn get_active_alerts(&self) -> Result<Vec<PerformanceAlert>> {
        let metrics = self.measure_now()?;

        let mut manager = self.alert_manager.lock().unwrap_or_else(|p| p.into_inner());
        manager.evaluate(&metrics).context("Failed to evaluate alert rules")?;
        Ok(manager.get_active_alerts())
    }

    /// Get comprehensive system health assessment
    ///
    /// Returns overall system health status including component-specific
    /// health scores and recommendations.
    pub fn get_system_health(&self) -> Result<Option<SystemHealth>> {
        debug!("Getting system health assessment");
        self.assess_health_now().context("Failed to get system health assessment")
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

        let manager = self.export_manager.lock().unwrap_or_else(|p| p.into_inner());
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
            let mut config = self.config.write().unwrap_or_else(|p| p.into_inner());
            *config = new_config.clone();
        }

        // Propagate configuration updates to subsystems
        {
            let collector = self.metrics_collector.lock().unwrap_or_else(|p| p.into_inner());
            collector.update_config(new_config.clone())?;
        }

        {
            let mut detector = self.bottleneck_detector.lock().unwrap_or_else(|p| p.into_inner());
            detector.update_config(new_config.clone())?;
        }

        {
            let mut engine = self.optimization_engine.lock().unwrap_or_else(|p| p.into_inner());
            engine.update_config(new_config.clone())?;
        }

        {
            let mut monitor = self.real_time_monitor.lock().unwrap_or_else(|p| p.into_inner());
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
        let state = self.profiling_state.read().unwrap_or_else(|p| p.into_inner());
        state.is_active
    }

    /// Get current profiling state information
    ///
    /// Returns comprehensive information about the current profiling state.
    pub fn get_profiling_state(&self) -> ProfilingState {
        let state = self.profiling_state.read().unwrap_or_else(|p| p.into_inner());
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

        let manager = self.export_manager.lock().unwrap_or_else(|p| p.into_inner());
        manager
            .generate_report(&profiling_data)
            .context("Failed to generate performance report")
    }

    /// Perform health check on the profiler system
    ///
    /// Returns system health status and diagnostic information.
    pub fn health_check(&self) -> Result<Option<SystemHealth>> {
        debug!("Performing profiler health check");
        self.assess_health_now().context("Failed to perform health check")
    }

    /// Get profiler capabilities and supported features
    ///
    /// Returns information about what the profiler can monitor and analyze.
    pub fn get_capabilities(&self) -> Result<ProfilerCapabilities> {
        debug!("Getting profiler capabilities");

        // Create capabilities based on current configuration and platform
        let config = self.config.read().unwrap_or_else(|p| p.into_inner());
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

        let collector = self.metrics_collector.lock().unwrap_or_else(|p| p.into_inner());
        collector.get_current_snapshot().context("Failed to take performance snapshot")
    }

    /// Assess overall system health
    ///
    /// Provides comprehensive health assessment of the mobile system.
    pub fn assess_system_health(&self) -> Result<Option<SystemHealth>> {
        debug!("Assessing system health");
        self.assess_health_now().context("Failed to assess system health")
    }

    // =============================================================================
    // PRIVATE HELPER METHODS
    // =============================================================================

    /// Take a fresh measurement and return it.
    ///
    /// The query APIs below (`detect_bottlenecks`, `get_active_alerts`,
    /// health assessment) sample at the moment they are called rather than
    /// reading whatever the background sampler last stored, so a profiler that
    /// has not yet completed a sampling interval still answers from real data
    /// instead of from a default-constructed snapshot.
    fn measure_now(&self) -> Result<MobileMetricsSnapshot> {
        let collector = self.metrics_collector.lock().unwrap_or_else(|p| p.into_inner());
        if let Err(error) = collector.collect_metrics() {
            // Collection disabled by configuration, or a family refused: the
            // snapshot below then reports which families are absent.
            debug!(
                "Metrics collection reported an error while sampling: {}",
                error
            );
        }
        collector.get_current_snapshot().context("Failed to read current metrics")
    }

    /// Assess health from a fresh measurement.
    ///
    /// `None` when the running device measured no metric family at all.
    fn assess_health_now(&self) -> Result<Option<SystemHealth>> {
        let metrics = self.measure_now()?;
        let analyzer = self.performance_analyzer.lock().unwrap_or_else(|p| p.into_inner());
        analyzer.assess_health(&metrics)
    }

    /// Generate comprehensive profiling data
    fn generate_profiling_data(&self) -> Result<ProfilingData> {
        debug!("Generating comprehensive profiling data");

        // Collect session information
        let session_info = {
            let session = self.session_tracker.lock().unwrap_or_else(|p| p.into_inner());
            session.get_session_info()?
        };

        // Collect all metrics snapshots
        let metrics = {
            let collector = self.metrics_collector.lock().unwrap_or_else(|p| p.into_inner());
            collector.get_all_snapshots()
        };

        // Collect all events
        let events = {
            let session = self.session_tracker.lock().unwrap_or_else(|p| p.into_inner());
            session.get_all_events()
        };

        // Get detected bottlenecks
        let bottlenecks = {
            let detector = self.bottleneck_detector.lock().unwrap_or_else(|p| p.into_inner());
            detector.get_all_bottlenecks()
        };

        // Get optimization suggestions
        let suggestions = {
            let engine = self.optimization_engine.lock().unwrap_or_else(|p| p.into_inner());
            engine.get_all_suggestions()
        };

        // Generate summary statistics
        let summary = self.calculate_profiling_summary(&metrics, &events, &bottlenecks)?;

        // Get system health assessment
        let system_health = self.assess_health_now()?;

        Ok(ProfilingData {
            session_info,
            metrics,
            events,
            bottlenecks,
            suggestions,
            summary,
            system_health,
            export_timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64,
            profiler_version: env!("CARGO_PKG_VERSION").to_string(),
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

        // Resource usage statistics. Every aggregate below is taken over the
        // snapshots that actually carried the metric: a snapshot with no GPU,
        // battery or thermal telemetry contributes nothing rather than a zero,
        // and an aggregate with no contributing snapshot at all is `None`.
        let peak_memory_mb = metrics
            .iter()
            .filter_map(|m| {
                m.memory
                    .as_ref()
                    .map(|memory| memory.heap_used_mb + memory.native_used_mb.unwrap_or(0.0))
            })
            .fold(None::<f32>, |peak, mb| {
                Some(peak.map_or(mb, |best| best.max(mb)))
            });

        let cpu_samples: Vec<f32> = metrics
            .iter()
            .filter_map(|m| m.cpu.as_ref().map(|cpu| cpu.usage_percent))
            .collect();
        let avg_cpu_usage = if cpu_samples.is_empty() {
            None
        } else {
            Some(cpu_samples.iter().sum::<f32>() / cpu_samples.len() as f32)
        };

        let gpu_samples: Vec<f32> = metrics
            .iter()
            .filter_map(|m| m.gpu.as_ref().map(|gpu| gpu.usage_percent))
            .collect();
        let avg_gpu_usage = if gpu_samples.is_empty() {
            None
        } else {
            Some(gpu_samples.iter().sum::<f32>() / gpu_samples.len() as f32)
        };

        // A genuine mAh figure needs a time integral of current, and current
        // needs a voltage to turn a power reading into. `metrics` is
        // chronological (the collector only ever appends), so consecutive
        // entries with both a power and a voltage reading form the intervals
        // `integrate_battery_consumed_mah` integrates over; a voltage-less
        // power reading contributes to neither endpoint, so it simply isn't
        // part of any interval.
        let battery_series: Vec<(u64, f64, f64)> = metrics
            .iter()
            .filter_map(|m| {
                let battery = m.battery.as_ref()?;
                let power_mw = battery.power_consumption_mw?;
                let voltage_v = battery.voltage_v?;
                (voltage_v > 0.0).then_some((m.timestamp, power_mw as f64, voltage_v as f64))
            })
            .collect();
        let battery_consumed_mah = integrate_battery_consumed_mah(&battery_series);

        let throttling_samples: Vec<f32> = metrics
            .iter()
            .filter_map(|m| m.thermal.as_ref().and_then(|thermal| thermal.throttling_level))
            .collect();
        let thermal_events = if throttling_samples.is_empty() {
            None
        } else {
            Some(throttling_samples.iter().filter(|level| **level > 0.0).count() as u32)
        };

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

    /// Overall performance score in `0.0..=100.0`, or `None` when the session
    /// recorded nothing to score.
    ///
    /// The score is a weighted average over the components the latest snapshot
    /// actually measured, with the remaining weights renormalised. GPU and
    /// thermal drop out on a device that reports neither rather than
    /// contributing a full mark (which the old `100.0 - 0.0` produced from an
    /// unmeasured GPU) or a zero. The old empty-session return was a flat
    /// `50.0` "neutral score" -- a number where there was no measurement.
    fn calculate_performance_score(
        &self,
        metrics: &[MobileMetricsSnapshot],
        bottlenecks: &[PerformanceBottleneck],
    ) -> Result<Option<f32>> {
        let Some(latest_metrics) = metrics.last() else {
            return Ok(None);
        };

        let mut weighted_sum = 0.0f32;
        let mut total_weight = 0.0f32;
        let mut fold = |score: f32, weight: f32| {
            weighted_sum += score.clamp(0.0, 100.0) * weight;
            total_weight += weight;
        };

        if let Some(share) = latest_metrics
            .memory
            .as_ref()
            .and_then(|memory| memory.resident_share_percent())
        {
            fold(100.0 - share, 0.3);
        }
        if let Some(cpu) = latest_metrics.cpu.as_ref() {
            fold(100.0 - cpu.usage_percent, 0.3);
        }
        if let Some(gpu) = latest_metrics.gpu.as_ref() {
            fold(100.0 - gpu.usage_percent, 0.2);
        }
        if let Some(thermal) = latest_metrics.thermal.as_ref() {
            fold(
                match thermal.thermal_state {
                    ThermalState::Nominal => 100.0,
                    ThermalState::Fair => 80.0,
                    ThermalState::Serious => 60.0,
                    ThermalState::Critical => 20.0,
                    ThermalState::Emergency => 5.0,
                    ThermalState::Shutdown => 0.0,
                },
                0.2,
            );
        }

        if total_weight <= 0.0 {
            return Ok(None);
        }
        let base_score = (weighted_sum / total_weight).clamp(0.0, 100.0);

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

        Ok(Some((base_score - bottleneck_penalty).clamp(0.0, 100.0)))
    }

    /// Get current session duration in milliseconds
    fn get_session_duration_ms(&self) -> u64 {
        let state = self.profiling_state.read().unwrap_or_else(|p| p.into_inner());
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

/// Integrate a chronological series of `(timestamp_ms, power_mw, voltage_v)`
/// battery readings into a total charge throughput in mAh.
///
/// Current is computed pointwise at each reading (`I = P / V`, `P = V * I`
/// rearranged) *before* averaging: each consecutive pair of readings is one
/// interval, and the interval's average current in mA is the arithmetic mean
/// of the two endpoints' own `power / voltage` currents (the trapezoidal
/// rule applied to the current samples), integrated over the interval's
/// elapsed time in hours. This is deliberately not the trapezoidal average
/// power divided by the trapezoidal average voltage -- `avg(P) / avg(V)` is
/// not the same quantity as `avg(P / V)` whenever voltage varies within an
/// interval (division is nonlinear), so computing the ratio first and
/// averaging second is required for the result to actually be an average
/// current. `readings` with fewer than two entries yield `None` -- there is
/// no interval to integrate over, so there is nothing to report rather than
/// a number derived from a single instant.
fn integrate_battery_consumed_mah(readings: &[(u64, f64, f64)]) -> Option<f32> {
    if readings.len() < 2 {
        return None;
    }

    let mut milliamp_hours = 0.0f64;
    for pair in readings.windows(2) {
        let (t0, power0_mw, voltage0_v) = pair[0];
        let (t1, power1_mw, voltage1_v) = pair[1];
        let elapsed_hours = t1.saturating_sub(t0) as f64 / 3_600_000.0; // ms -> hours
        let current0_ma = power0_mw / voltage0_v;
        let current1_ma = power1_mw / voltage1_v;
        let avg_current_ma = (current0_ma + current1_ma) / 2.0;
        milliamp_hours += avg_current_ma * elapsed_hours;
    }
    Some(milliamp_hours as f32)
}

#[cfg(test)]
mod battery_integral_tests {
    use super::integrate_battery_consumed_mah;

    /// Fewer than two readings: nothing to integrate over.
    #[test]
    fn empty_and_single_reading_yield_none() {
        assert_eq!(integrate_battery_consumed_mah(&[]), None);
        assert_eq!(integrate_battery_consumed_mah(&[(0, 1000.0, 5.0)]), None);
    }

    /// Constant 1000 mW at 5 V for exactly one hour: I = P / V = 200 mA,
    /// held for 1 h, so the integral is exactly 200 mAh.
    #[test]
    fn constant_power_and_voltage_for_one_hour() {
        let one_hour_ms = 3_600_000u64;
        let readings = [(0u64, 1000.0f64, 5.0f64), (one_hour_ms, 1000.0, 5.0)];
        let mah = integrate_battery_consumed_mah(&readings).expect("two readings");
        assert!((mah - 200.0).abs() < 1e-3, "got {mah}");
    }

    /// A linear power ramp at constant voltage: the trapezoidal average power
    /// over the interval equals the arithmetic mean of the endpoints, so the
    /// result must match the constant-power case at that mean power.
    #[test]
    fn linear_power_ramp_matches_its_average() {
        let one_hour_ms = 3_600_000u64;
        let ramp = [(0u64, 500.0f64, 5.0f64), (one_hour_ms, 1500.0, 5.0)];
        let mah = integrate_battery_consumed_mah(&ramp).expect("two readings");
        // Average power 1000 mW at 5 V -> 200 mA -> 200 mAh over 1 h, same as
        // the constant-power test above.
        assert!((mah - 200.0).abs() < 1e-3, "got {mah}");
    }

    /// Three readings covering two half-hour intervals accumulate rather
    /// than only reflecting the first or last interval.
    #[test]
    fn multiple_intervals_accumulate() {
        let half_hour_ms = 1_800_000u64;
        let readings = [
            (0u64, 1000.0f64, 5.0f64),       // 200 mA
            (half_hour_ms, 1000.0, 5.0),     // 200 mA, 0.5 h -> 100 mAh
            (half_hour_ms * 2, 2000.0, 5.0), // ramps to 400 mA; avg 300 mA, 0.5 h -> 150 mAh
        ];
        let mah = integrate_battery_consumed_mah(&readings).expect("three readings");
        assert!((mah - 250.0).abs() < 1e-3, "got {mah}"); // 100 + 150
    }

    /// A zero-length interval (two readings with the same timestamp)
    /// contributes nothing, and must not divide by a zero elapsed time.
    #[test]
    fn zero_length_interval_contributes_nothing() {
        let readings = [(1_000u64, 1000.0f64, 5.0f64), (1_000u64, 1000.0, 5.0)];
        let mah = integrate_battery_consumed_mah(&readings).expect("two readings");
        assert!((mah - 0.0).abs() < 1e-6, "got {mah}");
    }

    /// Every other test above holds voltage constant across the interval, so
    /// they cannot tell the correct `avg(P / V)` from the wrong `avg(P) /
    /// avg(V)` -- the two formulas coincide when voltage doesn't move. This
    /// test varies voltage within a single interval to discriminate them.
    ///
    /// P0 = 1000 mW at V0 = 5 V  -> I0 = 200 mA
    /// P1 = 1000 mW at V1 = 10 V -> I1 = 100 mA
    /// Correct trapezoidal average current = (200 + 100) / 2 = 150 mA, held
    /// for 1 h -> 150 mAh. The wrong `avg(P) / avg(V)` formula would instead
    /// give avg(P) = 1000 mW, avg(V) = 7.5 V -> 133.33... mA -> ~133.33 mAh,
    /// which this assertion rejects.
    #[test]
    fn varying_voltage_averages_current_not_power_over_voltage() {
        let one_hour_ms = 3_600_000u64;
        let readings = [(0u64, 1000.0f64, 5.0f64), (one_hour_ms, 1000.0, 10.0)];
        let mah = integrate_battery_consumed_mah(&readings).expect("two readings");
        assert!(
            (mah - 150.0).abs() < 1e-3,
            "got {mah}, expected 150.0 (not ~133.33)"
        );
    }
}
