//! The `MobilePerformanceProfiler` engine: orchestrates collection, analysis and alerting.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::device_info::MobileDeviceInfo;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;
use trustformers_core::error::Result;
use trustformers_core::TrustformersError;

use super::analysis_engines::{AlertSystem, BottleneckDetector, PerformanceAnalyzer};
use super::analysis_types::{BottleneckSeverity, PerformanceBottleneck};
use super::collector::{
    ExportFormat, MetricsCollector, ProfileSnapshot, ProfilerCapability, ProfilingSession,
    SessionStats,
};
use super::config_types::ProfilerConfig;
use super::metrics_types::{MemoryPressureLevel, MetricsSnapshot, PlatformMetrics};
use super::platform_profilers::{AndroidProfiler, GenericProfiler, IOSProfiler, PlatformProfiler};

/// Advanced mobile performance profiler
pub struct MobilePerformanceProfiler {
    pub(super) config: ProfilerConfig,
    pub(super) platform_profiler: Box<dyn PlatformProfiler + Send + Sync>,
    pub(super) metrics_collector: MetricsCollector,
    pub(super) bottleneck_detector: BottleneckDetector,
    pub(super) performance_analyzer: PerformanceAnalyzer,
    pub(super) alert_system: AlertSystem,
    pub(super) profiling_session: Option<ProfilingSession>,
    pub(super) historical_data: VecDeque<ProfileSnapshot>,
}
impl MobilePerformanceProfiler {
    /// Create new performance profiler
    pub fn new(config: ProfilerConfig, device_info: &MobileDeviceInfo) -> Result<Self> {
        let platform_profiler = Self::create_platform_profiler(device_info)?;
        let metrics_collector =
            MetricsCollector::new(config.metrics_config.clone(), device_info.clone());
        let bottleneck_detector = BottleneckDetector::new(config.bottleneck_config.clone());
        let performance_analyzer = PerformanceAnalyzer::new();
        let alert_system = AlertSystem::new(config.alert_thresholds.clone());

        Ok(Self {
            config: config.clone(),
            platform_profiler,
            metrics_collector,
            bottleneck_detector,
            performance_analyzer,
            alert_system,
            profiling_session: None,
            historical_data: VecDeque::with_capacity(config.max_history_size),
        })
    }

    /// Create platform-specific profiler
    pub(super) fn create_platform_profiler(
        device_info: &MobileDeviceInfo,
    ) -> Result<Box<dyn PlatformProfiler + Send + Sync>> {
        match device_info.basic_info.platform {
            crate::MobilePlatform::Ios => Ok(Box::new(IOSProfiler::new()?)),
            crate::MobilePlatform::Android => Ok(Box::new(AndroidProfiler::new()?)),
            crate::MobilePlatform::Generic => Ok(Box::new(GenericProfiler::new()?)),
        }
    }

    /// Start profiling session
    pub fn start_session(&mut self, session_id: String) -> Result<()> {
        if self.profiling_session.is_some() {
            return Err(TrustformersError::config_error(
                "Profiling session already active",
                "start_session",
            )
            .into());
        }

        self.profiling_session = Some(ProfilingSession {
            session_id: session_id.clone(),
            start_time: Instant::now(),
            end_time: None,
            config: self.config.clone(),
            collected_snapshots: 0,
            session_stats: SessionStats {
                duration_ms: 0,
                snapshots_collected: 0,
                avg_sampling_rate_hz: 0.0,
                data_size_bytes: 0,
                bottlenecks_detected: 0,
                alerts_triggered: 0,
            },
        });

        if self.config.enable_platform_integration {
            self.platform_profiler.start_profiling()?;
        }

        self.metrics_collector.start()?;

        Ok(())
    }

    /// Stop profiling session
    pub fn stop_session(&mut self) -> Result<SessionStats> {
        let session = self.profiling_session.take().ok_or_else(|| {
            TrustformersError::config_error("No active profiling session", "stop_session")
        })?;

        if self.config.enable_platform_integration {
            self.platform_profiler.stop_profiling()?;
        }

        self.metrics_collector.stop()?;

        let duration = session.start_time.elapsed();
        let stats = SessionStats {
            duration_ms: duration.as_millis() as u64,
            snapshots_collected: session.collected_snapshots,
            avg_sampling_rate_hz: session.collected_snapshots as f32 / duration.as_secs() as f32,
            data_size_bytes: self.estimate_data_size(),
            bottlenecks_detected: self.bottleneck_detector.detected_bottlenecks.len(),
            alerts_triggered: self.alert_system.alert_history.len(),
        };

        Ok(stats)
    }

    /// Collect performance snapshot
    pub fn collect_snapshot(&mut self) -> Result<ProfileSnapshot> {
        let platform_metrics = if self.config.enable_platform_integration {
            self.platform_profiler.collect_metrics()?
        } else {
            PlatformMetrics::default()
        };

        let metrics_snapshot = self.metrics_collector.collect_snapshot(platform_metrics)?;

        // Detect bottlenecks
        let bottlenecks = self.bottleneck_detector.analyze(&metrics_snapshot)?;

        // Check for alerts
        let alerts = self.alert_system.check_thresholds(&metrics_snapshot)?;

        // Generate optimization suggestions
        let optimizations = self
            .performance_analyzer
            .generate_suggestions(&metrics_snapshot, &bottlenecks)?;

        // Calculate performance score
        let performance_score = self.calculate_performance_score(&metrics_snapshot, &bottlenecks);

        let snapshot = ProfileSnapshot {
            timestamp: Instant::now(),
            performance_score,
            bottlenecks,
            alerts,
            metrics: metrics_snapshot,
            optimization_suggestions: optimizations,
        };

        // Update profiling session
        if let Some(ref mut session) = self.profiling_session {
            session.collected_snapshots += 1;
        }

        // Store in historical data
        self.historical_data.push_back(snapshot.clone());
        if self.historical_data.len() > self.config.max_history_size {
            self.historical_data.pop_front();
        }

        Ok(snapshot)
    }

    /// Calculate overall performance score (0.0-100.0)
    pub(super) fn calculate_performance_score(
        &self,
        metrics: &MetricsSnapshot,
        bottlenecks: &[PerformanceBottleneck],
    ) -> f32 {
        let mut score = 100.0;

        // Penalize based on CPU utilization
        if metrics.platform_metrics.cpu_metrics.utilization_percent > 80.0 {
            score -= (metrics.platform_metrics.cpu_metrics.utilization_percent - 80.0) * 0.5;
        }

        // Penalize based on memory pressure
        match metrics.platform_metrics.memory_metrics.pressure_level {
            MemoryPressureLevel::Medium => score -= 10.0,
            MemoryPressureLevel::High => score -= 25.0,
            MemoryPressureLevel::Critical => score -= 50.0,
            _ => {},
        }

        // Penalize based on inference latency
        if metrics.inference_metrics.latency_ms > 100.0 {
            score -= (metrics.inference_metrics.latency_ms - 100.0) * 0.1;
        }

        // Penalize based on bottlenecks
        for bottleneck in bottlenecks {
            let penalty = match bottleneck.severity {
                BottleneckSeverity::Low => 5.0,
                BottleneckSeverity::Medium => 10.0,
                BottleneckSeverity::High => 20.0,
                BottleneckSeverity::Critical => 40.0,
            };
            score -= penalty;
        }

        score.max(0.0).min(100.0)
    }

    /// Estimate total data size collected
    pub(super) fn estimate_data_size(&self) -> usize {
        // Rough estimate based on historical data size
        self.historical_data.len() * 2048 // ~2KB per snapshot
    }

    /// Export profiling data
    pub fn export_data(&self, format: ExportFormat) -> Result<Vec<u8>> {
        match format {
            ExportFormat::JSON => {
                let data = serde_json::to_vec(&self.historical_data)
                    .map_err(|e| TrustformersError::serialization_error(e.to_string()))?;
                Ok(data)
            },
            _ => {
                // Delegate to platform profiler for specialized formats
                self.platform_profiler.export_data(format)
            },
        }
    }

    /// Get profiler capabilities
    pub fn get_capabilities(&self) -> Vec<ProfilerCapability> {
        self.platform_profiler.get_capabilities()
    }

    /// Get current profiling statistics
    pub fn get_statistics(&self) -> Result<ProfilingStatistics> {
        let current_session = self.profiling_session.as_ref();

        Ok(ProfilingStatistics {
            total_snapshots: self.historical_data.len(),
            average_performance_score: self.calculate_average_performance_score(),
            active_bottlenecks: self.bottleneck_detector.detected_bottlenecks.len(),
            active_alerts: self.alert_system.active_alerts.len(),
            session_duration_ms: current_session
                .map(|s| s.start_time.elapsed().as_millis() as u64)
                .unwrap_or(0),
            data_collection_rate_hz: self.calculate_data_collection_rate(),
        })
    }

    pub(super) fn calculate_average_performance_score(&self) -> f32 {
        if self.historical_data.is_empty() {
            return 0.0;
        }

        let sum: f32 = self.historical_data.iter().map(|s| s.performance_score).sum();
        sum / self.historical_data.len() as f32
    }

    pub(super) fn calculate_data_collection_rate(&self) -> f32 {
        if let Some(session) = &self.profiling_session {
            let duration_secs = session.start_time.elapsed().as_secs_f32();
            if duration_secs > 0.0 {
                return session.collected_snapshots as f32 / duration_secs;
            }
        }
        0.0
    }
}
/// Overall profiling statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingStatistics {
    /// Total snapshots collected
    pub total_snapshots: usize,
    /// Average performance score
    pub average_performance_score: f32,
    /// Number of active bottlenecks
    pub active_bottlenecks: usize,
    /// Number of active alerts
    pub active_alerts: usize,
    /// Current session duration (ms)
    pub session_duration_ms: u64,
    /// Data collection rate (Hz)
    pub data_collection_rate_hz: f32,
}
