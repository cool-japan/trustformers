// Monitoring and debugging tools for TrustformeRS
pub mod activation_stats;
pub mod attention;
pub mod gradient_flow;
pub mod memory;
pub mod metrics;
pub mod profiler;
pub mod tensorboard;

pub use activation_stats::*;
pub use attention::*;
pub use gradient_flow::*;
pub use memory::*;
pub use metrics::*;
pub use profiler::*;
pub use tensorboard::*;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Central monitoring system for tracking model performance and resource usage
#[derive(Debug, Clone)]
pub struct ModelMonitor {
    memory_tracker: MemoryTracker,
    attention_visualizer: AttentionVisualizer,
    profiler: ModelProfiler,
    metrics_collector: MetricsCollector,
    enabled: bool,
}

impl Default for ModelMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelMonitor {
    pub fn new() -> Self {
        Self {
            memory_tracker: MemoryTracker::new(),
            attention_visualizer: AttentionVisualizer::new(),
            profiler: ModelProfiler::new(),
            metrics_collector: MetricsCollector::new(),
            enabled: true,
        }
    }

    pub fn with_config(config: MonitoringConfig) -> Self {
        Self {
            memory_tracker: MemoryTracker::with_config(config.memory_config),
            attention_visualizer: AttentionVisualizer::with_config(config.attention_config),
            profiler: ModelProfiler::with_config(config.profiler_config),
            metrics_collector: MetricsCollector::with_config(config.metrics_config),
            enabled: config.enabled,
        }
    }

    /// Start monitoring a forward pass
    pub fn start_forward_pass(
        &mut self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<MonitoringSession> {
        if !self.enabled {
            return Ok(MonitoringSession::disabled());
        }

        let session_id = uuid::Uuid::new_v4().to_string();
        let start_time = Instant::now();

        self.memory_tracker.start_tracking(&session_id)?;
        self.attention_visualizer.start_tracking(&session_id)?;
        self.profiler.start_profiling(&session_id)?;

        Ok(MonitoringSession {
            id: session_id,
            start_time,
            batch_size,
            sequence_length,
            enabled: true,
        })
    }

    /// Track attention weights for visualization
    pub fn track_attention(
        &mut self,
        session: &MonitoringSession,
        layer_idx: usize,
        attention_weights: &crate::tensor::Tensor,
        input_tokens: Option<&[String]>,
    ) -> Result<()> {
        if !session.enabled {
            return Ok(());
        }

        self.attention_visualizer.track_attention(
            &session.id,
            layer_idx,
            attention_weights,
            input_tokens,
        )
    }

    /// Track memory usage at a specific point
    pub fn track_memory(
        &mut self,
        session: &MonitoringSession,
        checkpoint: &str,
    ) -> Result<MemorySnapshot> {
        if !session.enabled {
            return Ok(MemorySnapshot::default());
        }

        self.memory_tracker.take_snapshot(&session.id, checkpoint)
    }

    /// End monitoring session and collect results
    pub fn end_session(&mut self, session: MonitoringSession) -> Result<MonitoringReport> {
        if !session.enabled {
            return Ok(MonitoringReport::default());
        }

        let duration = session.start_time.elapsed();

        let memory_report = self.memory_tracker.end_tracking(&session.id)?;
        let profiling_report = self.profiler.end_profiling(&session.id)?;
        let attention_report = self.attention_visualizer.get_report(&session.id)?;

        let report = MonitoringReport {
            session_id: session.id,
            duration,
            batch_size: session.batch_size,
            sequence_length: session.sequence_length,
            memory_report,
            profiling_report,
            attention_report,
            metrics: self.metrics_collector.collect_metrics()?,
        };

        Ok(report)
    }

    /// Enable or disable monitoring
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get current monitoring status
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Clear all collected data
    pub fn clear(&mut self) -> Result<()> {
        self.memory_tracker.clear()?;
        self.attention_visualizer.clear()?;
        self.profiler.clear()?;
        self.metrics_collector.clear()?;
        Ok(())
    }
}

/// Configuration for the monitoring system
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub memory_config: MemoryTrackerConfig,
    pub attention_config: AttentionVisualizerConfig,
    pub profiler_config: ProfilerConfig,
    pub metrics_config: MetricsCollectorConfig,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_config: MemoryTrackerConfig::default(),
            attention_config: AttentionVisualizerConfig::default(),
            profiler_config: ProfilerConfig::default(),
            metrics_config: MetricsCollectorConfig::default(),
        }
    }
}

/// Monitoring session for tracking a single forward pass
#[derive(Debug, Clone)]
pub struct MonitoringSession {
    pub id: String,
    pub start_time: Instant,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub enabled: bool,
}

impl MonitoringSession {
    fn disabled() -> Self {
        Self {
            id: String::new(),
            start_time: Instant::now(),
            batch_size: 0,
            sequence_length: 0,
            enabled: false,
        }
    }
}

/// Complete monitoring report for a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringReport {
    pub session_id: String,
    pub duration: Duration,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub memory_report: MemoryReport,
    pub profiling_report: ProfilingReport,
    pub attention_report: AttentionReport,
    pub metrics: HashMap<String, f64>,
}

impl Default for MonitoringReport {
    fn default() -> Self {
        Self {
            session_id: String::new(),
            duration: Duration::from_secs(0),
            batch_size: 0,
            sequence_length: 0,
            memory_report: MemoryReport::default(),
            profiling_report: ProfilingReport::default(),
            attention_report: AttentionReport::default(),
            metrics: HashMap::new(),
        }
    }
}

impl MonitoringReport {
    /// Save report to file
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load report from file
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let report = serde_json::from_str(&content)?;
        Ok(report)
    }

    /// Print a summary of the report
    pub fn print_summary(&self) {
        println!("Monitoring Report Summary");
        println!("========================");
        println!("Session ID: {}", self.session_id);
        println!("Duration: {:.2}ms", self.duration.as_millis());
        println!("Batch Size: {}", self.batch_size);
        println!("Sequence Length: {}", self.sequence_length);
        println!();

        self.memory_report.print_summary();
        println!();

        self.profiling_report.print_summary();
        println!();

        self.attention_report.print_summary();
        println!();

        if !self.metrics.is_empty() {
            println!("Additional Metrics:");
            for (name, value) in &self.metrics {
                println!("  {}: {:.4}", name, value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let monitor = ModelMonitor::new();
        assert!(monitor.is_enabled());
    }

    #[test]
    fn test_monitor_with_config() {
        let config = MonitoringConfig {
            enabled: false,
            ..Default::default()
        };

        let monitor = ModelMonitor::with_config(config);
        assert!(!monitor.is_enabled());
    }

    #[test]
    fn test_monitoring_session() -> Result<()> {
        let mut monitor = ModelMonitor::new();

        let session = monitor.start_forward_pass(4, 128)?;
        assert_eq!(session.batch_size, 4);
        assert_eq!(session.sequence_length, 128);
        assert!(session.enabled);

        let report = monitor.end_session(session)?;
        assert!(report.duration > Duration::from_nanos(0));

        Ok(())
    }

    #[test]
    fn test_disabled_monitoring() -> Result<()> {
        let mut monitor = ModelMonitor::new();
        monitor.set_enabled(false);

        let session = monitor.start_forward_pass(4, 128)?;
        assert!(!session.enabled);

        let report = monitor.end_session(session)?;
        assert_eq!(report.session_id, "");

        Ok(())
    }

    #[test]
    fn test_monitor_clear() -> Result<()> {
        let mut monitor = ModelMonitor::new();

        // Start and end a session to populate some data
        let session = monitor.start_forward_pass(4, 128)?;
        let _report = monitor.end_session(session)?;

        // Clear should not fail
        monitor.clear()?;

        Ok(())
    }

    #[test]
    fn test_monitoring_config_default() {
        let config = MonitoringConfig::default();
        assert!(config.enabled);
    }

    #[test]
    fn test_monitoring_report_serialization() -> Result<()> {
        let report = MonitoringReport::default();

        // Test saving and loading
        let temp_path = "/tmp/test_monitoring_report.json";
        report.save_to_file(temp_path)?;
        let loaded_report = MonitoringReport::load_from_file(temp_path)?;

        assert_eq!(report.session_id, loaded_report.session_id);
        assert_eq!(report.batch_size, loaded_report.batch_size);

        // Clean up
        std::fs::remove_file(temp_path).ok();

        Ok(())
    }
}
