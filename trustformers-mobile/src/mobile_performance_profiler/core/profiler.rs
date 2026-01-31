//! Core Mobile Performance Profiler
//!
//! This is the main orchestrating profiler that integrates all the specialized
//! components (bottleneck detection, optimization, real-time monitoring, export).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};
use trustformers_core::errors::TrustformersError;

use super::super::bottleneck::BottleneckDetector;
use super::super::collector::{CollectionStatistics, MobileMetricsCollector};
use super::super::optimization::OptimizationEngine;
use super::super::realtime::RealTimeMonitor;
use super::super::types::*;
use crate::device_info::{MobileDeviceDetector, MobileDeviceInfo};

/// Main Mobile Performance Profiler
#[derive(Debug)]
pub struct MobilePerformanceProfiler {
    /// Profiler configuration
    config: MobileProfilerConfig,
    /// Current profiling state
    profiling_state: Arc<Mutex<ProfilingState>>,
    /// Metrics collection system
    metrics_collector: MobileMetricsCollector,
    /// Active profiling session
    current_session: Option<ProfilingSession>,
    /// Bottleneck detection engine
    bottleneck_detector: BottleneckDetector,
    /// Optimization suggestion engine
    optimization_engine: OptimizationEngine,
    /// Real-time monitoring system
    real_time_monitor: Option<RealTimeMonitor>,
    /// Device information
    device_info: MobileDeviceInfo,
    /// Profiling statistics
    profiling_stats: ProfilingStats,
}

/// Current profiling state
#[derive(Debug, Clone)]
pub struct ProfilingState {
    /// Whether profiling is currently active
    pub is_active: bool,
    /// Current session ID
    pub session_id: Option<String>,
    /// Profiling start time
    pub start_time: Option<Instant>,
    /// Total number of profiling sessions
    pub total_sessions: u64,
    /// Current profiling mode
    pub profiling_mode: ProfilingMode,
}

/// Profiling session information
#[derive(Debug, Clone)]
pub struct ProfilingSession {
    /// Unique session identifier
    pub session_id: String,
    /// Session start time
    pub start_time: Instant,
    /// Session configuration
    pub config: SessionConfig,
    /// Collected profiling data
    pub profiling_data: ProfilingData,
    /// Session metadata
    pub metadata: SessionMetadata,
    /// Session statistics
    pub session_stats: SessionStats,
}

/// Session statistics
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// Total events recorded
    pub total_events: u64,
    /// Total metrics collected
    pub total_metrics: u64,
    /// Total bottlenecks detected
    pub total_bottlenecks: u64,
    /// Total suggestions generated
    pub total_suggestions: u64,
    /// Session duration
    pub duration: Duration,
}

/// Overall profiling statistics
#[derive(Debug, Clone, Default)]
pub struct ProfilingStats {
    /// Total profiling time across all sessions
    pub total_profiling_time: Duration,
    /// Total sessions completed
    pub total_sessions_completed: u64,
    /// Average session duration
    pub avg_session_duration_ms: f32,
    /// Total performance events recorded
    pub total_events_recorded: u64,
    /// Total optimization suggestions generated
    pub total_suggestions_generated: u64,
    /// Success rate of optimization suggestions
    pub suggestion_success_rate: f32,
}

impl MobilePerformanceProfiler {
    /// Create a new mobile performance profiler
    pub fn new(config: MobileProfilerConfig) -> Result<Self> {
        let device_info = MobileDeviceDetector::detect()
            .context("Failed to detect mobile device information")?;

        let metrics_collector = MobileMetricsCollector::new(config.collector_config.clone())?;
        let bottleneck_detector = BottleneckDetector::new(config.bottleneck_detection_config.clone())?;
        let optimization_engine = OptimizationEngine::new(config.optimization_engine_config.clone())?;

        let real_time_monitor = if config.enable_real_time_monitoring {
            Some(RealTimeMonitor::new(config.real_time_monitoring_config.clone())?)
        } else {
            None
        };

        Ok(Self {
            config,
            profiling_state: Arc::new(Mutex::new(ProfilingState::new())),
            metrics_collector,
            current_session: None,
            bottleneck_detector,
            optimization_engine,
            real_time_monitor,
            device_info,
            profiling_stats: ProfilingStats::default(),
        })
    }

    /// Start a new profiling session
    pub fn start_profiling(&mut self) -> Result<String> {
        let session_id = format!("session_{}", chrono::Utc::now().timestamp());

        info!("Starting profiling session: {}", session_id);

        // Create new session
        let session = ProfilingSession::new(session_id.clone(), SessionConfig::default())?;
        self.current_session = Some(session);

        // Update profiling state
        {
            let mut state = self.profiling_state.lock()
                .expect("profiling_state lock should not be poisoned");
            state.is_active = true;
            state.session_id = Some(session_id.clone());
            state.start_time = Some(Instant::now());
            state.total_sessions += 1;
        }

        // Start metrics collection
        self.metrics_collector.start_collection()?;

        // Start real-time monitoring if enabled
        if let Some(ref mut monitor) = self.real_time_monitor {
            monitor.start_monitoring()?;
        }

        Ok(session_id)
    }

    /// Stop the current profiling session
    pub fn stop_profiling(&mut self) -> Result<ProfilingData> {
        info!("Stopping profiling session");

        let session = self.current_session.take()
            .ok_or_else(|| TrustformersError::runtime_error("No active profiling session", "stop_profiling"))?;

        // Stop metrics collection
        self.metrics_collector.stop_collection()?;

        // Stop real-time monitoring
        if let Some(ref mut monitor) = self.real_time_monitor {
            monitor.stop_monitoring()?;
        }

        // Update profiling state
        {
            let mut state = self.profiling_state.lock()
                .expect("profiling_state lock should not be poisoned");
            state.is_active = false;
            state.session_id = None;

            if let Some(start_time) = state.start_time {
                let session_duration = start_time.elapsed();
                self.profiling_stats.total_profiling_time += session_duration;
                self.profiling_stats.total_sessions_completed += 1;

                // Update average session duration
                let total_time_ms = self.profiling_stats.total_profiling_time.as_millis() as f32;
                self.profiling_stats.avg_session_duration_ms =
                    total_time_ms / self.profiling_stats.total_sessions_completed as f32;
            }
        }

        Ok(session.profiling_data)
    }

    /// Record a profiling event during inference
    pub fn record_inference_event(&mut self, event_name: &str, duration_ms: Option<f32>) -> Result<()> {
        if let Some(ref mut session) = self.current_session {
            let event = ProfilingEvent {
                event_type: EventType::InferenceEvent,
                timestamp: Instant::now(),
                event_data: EventData {
                    event_name: event_name.to_string(),
                    duration_ms,
                    metadata: HashMap::new(),
                },
                metrics: None,
            };

            session.profiling_data.events.push(event);
            session.session_stats.total_events += 1;
            self.profiling_stats.total_events_recorded += 1;

            debug!("Recorded inference event: {}", event_name);
        }

        Ok(())
    }

    /// Get current performance metrics
    pub fn get_current_metrics(&self) -> Result<MobileMetricsSnapshot> {
        self.metrics_collector.get_current_metrics()
    }

    /// Detect current performance bottlenecks
    pub fn detect_bottlenecks(&mut self) -> Result<Vec<PerformanceBottleneck>> {
        let metrics = self.get_current_metrics()?;
        self.bottleneck_detector.detect_bottlenecks(&metrics)
    }

    /// Generate optimization suggestions
    pub fn get_optimization_suggestions(&mut self) -> Result<Vec<OptimizationSuggestion>> {
        let metrics = self.get_current_metrics()?;
        let bottlenecks = self.detect_bottlenecks()?;

        let suggestions = self.optimization_engine.generate_suggestions(&metrics, &bottlenecks)?;
        self.profiling_stats.total_suggestions_generated += suggestions.len() as u64;

        Ok(suggestions)
    }

    /// Get real-time monitoring state
    pub fn get_real_time_state(&self) -> Option<&crate::mobile_performance_profiler::realtime::RealTimeState> {
        self.real_time_monitor.as_ref().map(|m| m.get_current_state())
    }

    /// Export profiling data
    pub fn export_data(&self, format: ExportFormat) -> Result<String> {
        let session = self.current_session.as_ref()
            .ok_or_else(|| TrustformersError::runtime_error("No active profiling session", "export_data"))?;

        // Simplified export - in practice, you'd use ProfilerExportManager
        let export_path = format!("/tmp/claude/profiling_data.{}",
            match format {
                ExportFormat::JSON => "json",
                ExportFormat::CSV => "csv",
                ExportFormat::HTML => "html",
                ExportFormat::XML => "xml",
                ExportFormat::Flamegraph => "svg",
                ExportFormat::ChromeDevTools => "json",
                _ => "txt",
            }
        );

        // Create export directory if it doesn't exist
        std::fs::create_dir_all("/tmp/claude/")?;

        // Simple JSON export for now
        let json_data = serde_json::to_string_pretty(&session.profiling_data)?;
        std::fs::write(&export_path, json_data)?;

        info!("Exported profiling data to: {}", export_path);
        Ok(export_path)
    }

    /// Get profiling statistics
    pub fn get_profiling_stats(&self) -> &ProfilingStats {
        &self.profiling_stats
    }

    /// Get device information
    pub fn get_device_info(&self) -> &MobileDeviceInfo {
        &self.device_info
    }

    /// Check if profiling is currently active
    pub fn is_profiling_active(&self) -> bool {
        let state = self.profiling_state.lock()
            .expect("profiling_state lock should not be poisoned");
        state.is_active
    }
}

impl ProfilingState {
    /// Create a new profiling state
    pub fn new() -> Self {
        Self {
            is_active: false,
            session_id: None,
            start_time: None,
            total_sessions: 0,
            profiling_mode: ProfilingMode::Standard,
        }
    }
}

impl ProfilingSession {
    /// Create a new profiling session
    pub fn new(session_id: String, config: SessionConfig) -> Result<Self> {
        Ok(Self {
            session_id,
            start_time: Instant::now(),
            config,
            profiling_data: ProfilingData::new(),
            metadata: SessionMetadata::new(),
            session_stats: SessionStats::default(),
        })
    }

    /// Add metrics to the session
    pub fn add_metrics(&mut self, metrics: MobileMetricsSnapshot) {
        self.profiling_data.metrics.push(metrics);
        self.session_stats.total_metrics += 1;
    }

    /// Get session duration
    pub fn get_duration(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for MobilePerformanceProfiler {
    fn default() -> Self {
        Self::new(MobileProfilerConfig::default())
            .expect("Failed to create default mobile performance profiler")
    }
}