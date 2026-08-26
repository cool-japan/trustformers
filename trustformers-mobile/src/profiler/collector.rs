//! Metrics collection, profiling sessions and their snapshots/exports.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::device_info::MobileDeviceInfo;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;
use trustformers_core::error::Result;

use super::analysis_types::{OptimizationSuggestion, PerformanceAlert, PerformanceBottleneck};
use super::config_types::{MetricsConfig, ProfilerConfig};
use super::metrics_types::{InferenceMetrics, MetricsSnapshot, PlatformMetrics};

/// Export formats for profiling data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    JSON,
    CSV,
    Protobuf,
    Trace,
    Chrome,
    Instruments,
    Perfetto,
}
/// Metrics collector
pub(super) struct MetricsCollector {
    pub(super) config: MetricsConfig,
    pub(super) device_info: MobileDeviceInfo,
    pub(super) metrics_history: VecDeque<MetricsSnapshot>,
    pub(super) collection_start_time: Option<Instant>,
}
impl MetricsCollector {
    pub(super) fn new(config: MetricsConfig, device_info: MobileDeviceInfo) -> Self {
        Self {
            config,
            device_info,
            metrics_history: VecDeque::new(),
            collection_start_time: None,
        }
    }

    pub(super) fn start(&mut self) -> Result<()> {
        self.collection_start_time = Some(Instant::now());
        Ok(())
    }

    pub(super) fn stop(&mut self) -> Result<()> {
        self.collection_start_time = None;
        Ok(())
    }

    pub(super) fn collect_snapshot(
        &mut self,
        platform_metrics: PlatformMetrics,
    ) -> Result<MetricsSnapshot> {
        let snapshot = MetricsSnapshot {
            timestamp: Instant::now(),
            platform_metrics,
            inference_metrics: InferenceMetrics::default(),
            thermal_metrics: None,
            battery_metrics: None,
        };

        self.metrics_history.push_back(snapshot.clone());
        Ok(snapshot)
    }
}
/// Profile snapshot for historical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSnapshot {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub performance_score: f32,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub alerts: Vec<PerformanceAlert>,
    pub metrics: MetricsSnapshot,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}
/// Profiler capabilities for different platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfilerCapability {
    CpuProfiling,
    GpuProfiling,
    MemoryProfiling,
    NetworkProfiling,
    ThermalProfiling,
    BatteryProfiling,
    InstrumentsIntegration,
    SystraceIntegration,
    PerfettoIntegration,
    CustomProfiling,
}
/// Profiling session information
#[derive(Debug, Clone)]
pub struct ProfilingSession {
    pub(super) session_id: String,
    pub(super) start_time: Instant,
    pub(super) end_time: Option<Instant>,
    pub(super) config: ProfilerConfig,
    pub(super) collected_snapshots: usize,
    pub(super) session_stats: SessionStats,
}
/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    /// Total profiling duration (ms)
    pub duration_ms: u64,
    /// Total snapshots collected
    pub snapshots_collected: usize,
    /// Average sampling rate (Hz)
    pub avg_sampling_rate_hz: f32,
    /// Data size collected (bytes)
    pub data_size_bytes: usize,
    /// Bottlenecks detected
    pub bottlenecks_detected: usize,
    /// Alerts triggered
    pub alerts_triggered: usize,
}
