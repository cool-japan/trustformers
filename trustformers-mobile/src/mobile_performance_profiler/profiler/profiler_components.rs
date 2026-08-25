//! Component implementations for the mobile performance profiler
//!
//! Contains implementations for ProfilingSession, BottleneckDetector,
//! OptimizationEngine, RealTimeMonitor, ProfilerExportManager,
//! AlertManager, PerformanceAnalyzer, and helper functions.

use super::profiler_types::*;
use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

use crate::device_info::{MobileDeviceDetector, MobileDeviceInfo, ThermalState};
use crate::mobile_performance_profiler::collector::{CollectionStatistics, MobileMetricsCollector};
use crate::mobile_performance_profiler::config::MobileProfilerConfig;
use crate::mobile_performance_profiler::types::{
    AlertManagerConfig, AlertSeverity, AlertType, AnalysisConfig, BottleneckDetectionConfig,
    BottleneckSeverity, BottleneckType, CpuMetrics, ExportManagerConfig, HealthStatus,
    MobileMetricsSnapshot, OptimizationEngineConfig, OptimizationSuggestion, PerformanceAlert,
    PerformanceBottleneck, PlatformCapabilities, ProfilingData, ProfilingEvent,
    RealTimeMonitoringConfig, SessionInfo, SessionMetadata, SystemHealth, TrendingMetrics,
};

// =============================================================================
// SESSION MANAGEMENT IMPLEMENTATION
// =============================================================================

impl ProfilingSession {
    /// Create a new profiling session
    pub(crate) fn new(device_info: MobileDeviceInfo) -> Result<Self> {
        Ok(Self {
            session_id: None,
            start_time: None,
            end_time: None,
            device_info: Some(device_info),
            metadata: SessionMetadata::default(),
            events: VecDeque::new(),
            config_snapshot: None,
            state: SessionState::Idle,
            max_events: 10000, // Default maximum events
        })
    }

    /// Start a profiling session
    pub(crate) fn start_session(&mut self) -> Result<String> {
        if self.state != SessionState::Idle {
            return Err(anyhow::anyhow!("Session is not in idle state"));
        }

        self.state = SessionState::Starting;

        let session_id = format!("session_{}", chrono::Utc::now().timestamp_millis());
        self.session_id = Some(session_id.clone());
        self.start_time = Some(Instant::now());
        self.end_time = None;
        self.events.clear();

        // Initialize metadata
        self.metadata.session_id = session_id.clone();
        self.metadata.start_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;

        self.state = SessionState::Active;

        Ok(session_id)
    }

    /// End the current session
    pub(crate) fn end_session(&mut self) -> Result<()> {
        if self.state != SessionState::Active && self.state != SessionState::Paused {
            return Err(anyhow::anyhow!("No active session to end"));
        }

        self.state = SessionState::Stopping;
        self.end_time = Some(Instant::now());
        self.metadata.end_time =
            Some(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64);
        self.state = SessionState::Completed;

        Ok(())
    }

    /// Add an event to the session
    pub(crate) fn add_event(&mut self, event: ProfilingEvent) {
        self.events.push_back(event);

        // Limit memory usage by removing old events
        while self.events.len() > self.max_events {
            self.events.pop_front();
        }
    }

    /// Get session information
    pub(crate) fn get_session_info(&self) -> Result<SessionInfo> {
        Ok(SessionInfo {
            id: self.session_id.clone().unwrap_or_default(),
            start_time: self.metadata.start_time,
            end_time: self.metadata.end_time,
            duration_ms: self.calculate_duration_ms(),
            device_info: self.device_info.clone().unwrap_or_default(),
            metadata: self.metadata.clone(),
        })
    }

    /// Get all recorded events
    pub(crate) fn get_all_events(&self) -> Vec<ProfilingEvent> {
        self.events.iter().cloned().collect()
    }

    /// Calculate session duration in milliseconds
    pub(crate) fn calculate_duration_ms(&self) -> Option<u64> {
        if let (Some(start), Some(end)) = (self.start_time, self.end_time) {
            Some(end.duration_since(start).as_millis() as u64)
        } else {
            self.start_time.map(|start| start.elapsed().as_millis() as u64)
        }
    }
}

// =============================================================================
// DETECTION, ALERTING AND ANALYSIS
// =============================================================================

impl BottleneckDetector {
    pub(crate) fn new(_config: MobileProfilerConfig) -> Result<Self> {
        Ok(Self {
            config: BottleneckDetectionConfig::default(),
            active_bottlenecks: HashMap::new(),
            bottleneck_history: VecDeque::new(),
            detection_rules: Self::default_rules(),
            detection_stats: BottleneckDetectionStats::default(),
        })
    }

    /// Threshold rules evaluated on every snapshot.
    ///
    /// Deliberately limited to the three metric families this crate actually
    /// measures -- resident memory, CPU usage and the inference tracker's own
    /// latency and cache figures. Rules keyed on GPU, thermal or battery
    /// telemetry are absent because those are not measured, and a rule that
    /// can never fire is worse than no rule: it reads as a check that passed.
    fn default_rules() -> Vec<BottleneckRule> {
        vec![
            BottleneckRule {
                id: "memory_usage_high".to_string(),
                name: "High Memory Usage".to_string(),
                condition: BottleneckCondition::MemoryUsageHigh {
                    threshold_percent: 85.0,
                    duration_ms: 5_000,
                },
                severity: BottleneckSeverity::High,
                suggestion: "Reduce batch size or enable memory optimization".to_string(),
                confidence: 0.9,
                enabled: true,
            },
            BottleneckRule {
                id: "cpu_usage_high".to_string(),
                name: "High CPU Usage".to_string(),
                condition: BottleneckCondition::CPUUsageHigh {
                    threshold_percent: 90.0,
                    duration_ms: 3_000,
                },
                severity: BottleneckSeverity::High,
                suggestion: "Optimize model operations or reduce thread count".to_string(),
                confidence: 0.85,
                enabled: true,
            },
            BottleneckRule {
                id: "inference_latency_high".to_string(),
                name: "High Inference Latency".to_string(),
                condition: BottleneckCondition::LatencyHigh {
                    threshold_ms: 500.0,
                    sample_count: 10,
                },
                severity: BottleneckSeverity::Medium,
                suggestion: "Enable quantization or hardware acceleration".to_string(),
                confidence: 0.8,
                enabled: true,
            },
            BottleneckRule {
                id: "cache_hit_rate_low".to_string(),
                name: "Low Cache Hit Rate".to_string(),
                condition: BottleneckCondition::CacheHitRateLow {
                    threshold_percent: 50.0,
                    sample_count: 10,
                },
                severity: BottleneckSeverity::Low,
                suggestion: "Increase cache size or review cache key selection".to_string(),
                confidence: 0.75,
                enabled: true,
            },
        ]
    }

    /// Evaluate every enabled rule against one real snapshot, recording each
    /// bottleneck that trips.
    pub(crate) fn analyze(
        &mut self,
        metrics: &MobileMetricsSnapshot,
    ) -> Result<Vec<PerformanceBottleneck>> {
        let mut detected = Vec::new();

        for rule in self.detection_rules.clone() {
            if !rule.enabled {
                continue;
            }
            let Some((measured, threshold)) = Self::evaluate_rule(&rule, metrics) else {
                continue;
            };

            let bottleneck_type = Self::rule_bottleneck_type(&rule);
            let deviation =
                if threshold > 0.0 { ((measured - threshold) / threshold).abs() } else { 0.0 };

            let bottleneck = PerformanceBottleneck {
                bottleneck_type,
                severity: rule.severity,
                description: format!(
                    "{}: measured {:.1}, rule threshold {:.1}",
                    rule.name, measured, threshold
                ),
                affected_component: Self::rule_component(&rule).to_string(),
                impact_score: (deviation * 100.0).clamp(0.0, 100.0),
                suggestions: vec![rule.suggestion.clone()],
                timestamp: metrics.timestamp,
            };

            self.bottleneck_history.push_back(BottleneckDetectionEvent {
                timestamp: Instant::now(),
                bottleneck_type,
                severity: rule.severity,
                detected_value: measured,
                threshold_value: threshold,
                duration_ms: 0,
                rule_id: rule.id.clone(),
                metadata: HashMap::new(),
            });
            while self.bottleneck_history.len() > 256 {
                self.bottleneck_history.pop_front();
            }

            self.detection_stats.total_detections += 1;
            self.active_bottlenecks.insert(rule.id.clone(), bottleneck.clone());
            detected.push(bottleneck);
        }

        // A rule that no longer trips is no longer an active bottleneck.
        let still_active: Vec<String> = self
            .detection_rules
            .iter()
            .filter(|rule| rule.enabled && Self::evaluate_rule(rule, metrics).is_some())
            .map(|rule| rule.id.clone())
            .collect();
        self.active_bottlenecks.retain(|id, _| still_active.contains(id));

        Ok(detected)
    }

    /// `Some((measured, threshold))` when the rule trips on real data,
    /// `None` when it does not trip or when the metric it needs was not
    /// measured on this device.
    fn evaluate_rule(rule: &BottleneckRule, metrics: &MobileMetricsSnapshot) -> Option<(f32, f32)> {
        match &rule.condition {
            BottleneckCondition::MemoryUsageHigh {
                threshold_percent, ..
            } => {
                let share = metrics.memory.as_ref()?.resident_share_percent()?;
                (share > *threshold_percent).then_some((share, *threshold_percent))
            },
            BottleneckCondition::CPUUsageHigh {
                threshold_percent, ..
            } => {
                let usage = metrics.cpu.as_ref()?.usage_percent;
                (usage > *threshold_percent).then_some((usage, *threshold_percent))
            },
            BottleneckCondition::LatencyHigh {
                threshold_ms,
                sample_count,
            } => {
                if metrics.inference.total_inferences < u64::from(*sample_count) {
                    return None;
                }
                let latency = metrics.inference.avg_latency_ms as f32;
                (latency > *threshold_ms).then_some((latency, *threshold_ms))
            },
            BottleneckCondition::CacheHitRateLow {
                threshold_percent,
                sample_count,
            } => {
                if metrics.inference.total_inferences < u64::from(*sample_count) {
                    return None;
                }
                let hit_rate = metrics.inference.cache_hit_rate as f32 * 100.0;
                (hit_rate < *threshold_percent).then_some((hit_rate, *threshold_percent))
            },
            // GPU, thermal, battery and network conditions have no measured
            // input on any target this crate builds for, so they never fire.
            _ => None,
        }
    }

    fn rule_bottleneck_type(rule: &BottleneckRule) -> BottleneckType {
        match &rule.condition {
            BottleneckCondition::MemoryUsageHigh { .. } => BottleneckType::Memory,
            BottleneckCondition::CPUUsageHigh { .. } => BottleneckType::CPU,
            BottleneckCondition::GPUUsageHigh { .. } => BottleneckType::GPU,
            BottleneckCondition::LatencyHigh { .. } => BottleneckType::Latency,
            BottleneckCondition::ThermalThrottling { .. } => BottleneckType::Thermal,
            BottleneckCondition::BatteryDrainHigh { .. } => BottleneckType::Power,
            BottleneckCondition::NetworkLatencyHigh { .. } => BottleneckType::Network,
            BottleneckCondition::CacheHitRateLow { .. } => BottleneckType::Cache,
            _ => BottleneckType::Memory,
        }
    }

    fn rule_component(rule: &BottleneckRule) -> &'static str {
        match &rule.condition {
            BottleneckCondition::MemoryUsageHigh { .. } => "memory",
            BottleneckCondition::CPUUsageHigh { .. } => "cpu",
            BottleneckCondition::LatencyHigh { .. }
            | BottleneckCondition::CacheHitRateLow { .. } => "inference",
            _ => "system",
        }
    }

    pub(crate) fn get_active_bottlenecks(&self) -> Vec<PerformanceBottleneck> {
        self.active_bottlenecks.values().cloned().collect()
    }

    pub(crate) fn get_all_bottlenecks(&self) -> Vec<PerformanceBottleneck> {
        self.active_bottlenecks.values().cloned().collect()
    }

    pub(crate) fn update_config(&mut self, config: MobileProfilerConfig) -> Result<()> {
        // Update bottleneck detection configuration with available fields from main config
        self.config.enabled = config.enabled;
        // Use sampling interval for detection interval
        self.config.detection_interval_ms = config.sampling.interval_ms;

        debug!("Updated bottleneck detector configuration");
        Ok(())
    }
}

impl RealTimeMonitor {
    pub(crate) fn new(_config: MobileProfilerConfig) -> Result<Self> {
        Ok(Self {
            config: RealTimeMonitoringConfig::default(),
            current_state: RealTimeState {
                performance_score: 85.0,
                active_alerts: Vec::new(),
                trending_metrics: TrendingMetrics::default(),
                system_health: SystemHealth::default(),
                last_update: None,
                uptime: Duration::ZERO,
            },
            alert_manager: AlertManager::new(_config.clone())?,
            live_metrics: Arc::new(RwLock::new(VecDeque::new())),
            trending_metrics: TrendingMetrics::default(),
            system_health: SystemHealth::default(),
            monitor_stats: MonitoringStats::default(),
            _monitor_thread: None,
        })
    }

    pub(crate) fn start_monitoring(&mut self) -> Result<()> {
        self.current_state.last_update = Some(Instant::now());
        Ok(())
    }

    pub(crate) fn stop_monitoring(&mut self) -> Result<()> {
        // Clear current monitoring state
        self.current_state.performance_score = 0.0;
        self.current_state.active_alerts.clear();
        self.current_state.last_update = None;

        // Clear metrics buffer
        if let Ok(mut metrics) = self.live_metrics.write() {
            metrics.clear();
        }

        // Stop the background monitoring thread if running
        if let Some(handle) = self._monitor_thread.take() {
            // Thread will naturally stop when monitoring is disabled
            drop(handle);
        }

        info!("Real-time monitoring stopped");
        Ok(())
    }

    pub(crate) fn pause_monitoring(&mut self) -> Result<()> {
        // Pause monitoring by setting last_update to None
        // This signals that monitoring is paused
        self.current_state.last_update = None;

        // Update monitoring statistics (using available fields)
        self.monitor_stats.total_monitor_time =
            self.monitor_stats.total_monitor_time.saturating_sub(Duration::from_millis(100));

        info!("Real-time monitoring paused");
        Ok(())
    }

    pub(crate) fn resume_monitoring(&mut self) -> Result<()> {
        // Resume monitoring by updating last_update timestamp
        self.current_state.last_update = Some(Instant::now());

        // Update monitoring statistics (using available fields)
        self.monitor_stats.total_monitor_time += Duration::from_millis(100);

        info!("Real-time monitoring resumed");
        Ok(())
    }

    pub(crate) fn update_config(&mut self, config: MobileProfilerConfig) -> Result<()> {
        // Update real-time monitoring configuration with available fields
        self.config.enabled = config.real_time_monitoring.enabled;
        self.config.update_frequency_ms = config.real_time_monitoring.update_interval_ms;
        // Set alert interval to same as update interval since alert_interval_ms field doesn't exist
        self.config.alert_interval_ms = config.real_time_monitoring.update_interval_ms;
        // Set max_alerts to max_history_points since max_alerts field doesn't exist in config
        self.config.max_alerts = config.real_time_monitoring.max_history_points.min(100);

        // If monitoring is disabled, stop it
        if !self.config.enabled {
            self.stop_monitoring()?;
        }

        // Trim history points if max_history_points decreased
        // (Note: active_alerts trimming removed since max_alerts field doesn't exist)

        debug!("Updated real-time monitor configuration");
        Ok(())
    }
}

impl ProfilerExportManager {
    pub(crate) fn new(_config: MobileProfilerConfig) -> Result<Self> {
        Ok(Self {
            config: ExportManagerConfig::default(),
            formatters: HashMap::new(),
            export_history: VecDeque::new(),
            pending_exports: VecDeque::new(),
            export_stats: ExportManagerStats::default(),
        })
    }

    pub(crate) fn export_data(&self, data: &ProfilingData) -> Result<String> {
        // Generate timestamp-based filename
        let timestamp = chrono::Utc::now().timestamp();
        let export_path = std::env::temp_dir()
            .join(format!("profiling_export_{}.json", timestamp))
            .to_string_lossy()
            .into_owned();

        // Serialize data to JSON
        let json_data =
            serde_json::to_string_pretty(data).context("Failed to serialize profiling data")?;

        // Write data to file (compression temporarily disabled)
        std::fs::create_dir_all("/tmp/claude").context("Failed to create export directory")?;

        // Write JSON data directly (compression support can be added later with flate2 crate)
        std::fs::write(&export_path, json_data).context("Failed to write export file")?;

        // Update export statistics
        info!("Profiling data exported to: {}", export_path);
        Ok(export_path)
    }

    pub(crate) fn generate_report(&self, data: &ProfilingData) -> Result<String> {
        // Generate comprehensive HTML report
        let session_duration = if let Some(end) = data.session_info.end_time {
            Duration::from_secs(end.saturating_sub(data.session_info.start_time))
        } else {
            Duration::ZERO
        };

        let bottleneck_count = data.bottlenecks.len();
        let suggestion_count = data.suggestions.len();
        let metrics_count = data.metrics.len();
        let events_count = data.events.len();

        // Averages over the snapshots that carried a measurement; the report
        // says so rather than printing 0.0 for "nothing was measured".
        let mean = |samples: Vec<f32>| -> String {
            if samples.is_empty() {
                "not measured".to_string()
            } else {
                format!("{:.1}", samples.iter().sum::<f32>() / samples.len() as f32)
            }
        };
        let avg_cpu_usage = mean(
            data.metrics
                .iter()
                .filter_map(|m| m.cpu.as_ref().map(|c| c.usage_percent))
                .collect(),
        );
        let avg_memory_usage = mean(
            data.metrics
                .iter()
                .filter_map(|m| m.memory.as_ref().map(|memory| memory.heap_used_mb))
                .collect(),
        );

        let report_html = format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformRS Mobile Performance Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px 8px 0 0; }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header .subtitle {{ margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em; }}
        .section {{ padding: 30px; border-bottom: 1px solid #eee; }}
        .section:last-child {{ border-bottom: none; }}
        .section h2 {{ color: #333; margin: 0 0 20px 0; font-size: 1.5em; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #667eea; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; margin: 0; }}
        .metric-label {{ color: #666; margin: 5px 0 0 0; text-transform: uppercase; font-size: 0.8em; letter-spacing: 1px; }}
        .bottlenecks, .suggestions {{ margin: 20px 0; }}
        .bottleneck-item, .suggestion-item {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #ffc107; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.9em; }}
        .timestamp {{ color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Performance Report</h1>
            <div class="subtitle">TrustformRS Mobile Profiler Analysis</div>
            <div class="timestamp">Generated: {}</div>
        </div>

        <div class="section">
            <h2>Session Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{:.1}s</div>
                    <div class="metric-label">Session Duration</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Events Recorded</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Metrics Snapshots</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.1}</div>
                    <div class="metric-label">Overall Health</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{:.1}%</div>
                    <div class="metric-label">Average CPU Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.1} MB</div>
                    <div class="metric-label">Average Memory Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Bottlenecks Detected</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Optimization Suggestions</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Performance Issues</h2>
            <div class="bottlenecks">
                {}
            </div>
        </div>

        <div class="section">
            <h2>Optimization Recommendations</h2>
            <div class="suggestions">
                {}
            </div>
        </div>

        <div class="footer">
            <p>Generated by TrustformRS Mobile Performance Profiler v{}</p>
        </div>
    </div>
</body>
</html>
        "#,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            session_duration.as_secs_f64(),
            events_count,
            metrics_count,
            data.system_health.as_ref().map_or_else(
                || "not measured".to_string(),
                |h| format!("{:.1}", h.overall_score)
            ),
            avg_cpu_usage,
            avg_memory_usage,
            bottleneck_count,
            suggestion_count,
            if data.bottlenecks.is_empty() {
                "<div class=\"bottleneck-item\">No performance bottlenecks detected.</div>"
                    .to_string()
            } else {
                data.bottlenecks.iter().take(5).map(|b|
                    format!("<div class=\"bottleneck-item\"><strong>{}</strong>: {} (Severity: {:?})</div>",
                        b.affected_component, b.description, b.severity)
                ).collect::<Vec<_>>().join("\n")
            },
            if data.suggestions.is_empty() {
                "<div class=\"suggestion-item\">No optimization suggestions available.</div>"
                    .to_string()
            } else {
                data.suggestions.iter().take(5).map(|s|
                    format!("<div class=\"suggestion-item\"><strong>{}</strong>: {} (Priority: {:?})</div>",
                        format!("{:?}", s.suggestion_type), s.description, s.priority)
                ).collect::<Vec<_>>().join("\n")
            },
            data.profiler_version
        );

        Ok(report_html)
    }
}

impl AlertManager {
    pub(crate) fn new(_config: MobileProfilerConfig) -> Result<Self> {
        Ok(Self {
            config: AlertManagerConfig::default(),
            active_alerts: HashMap::new(),
            alert_history: VecDeque::new(),
            alert_rules: Self::default_rules(),
            notification_handlers: Vec::new(),
        })
    }

    /// Alert rules evaluated on every snapshot.
    ///
    /// Like the bottleneck rules, restricted to metrics that are genuinely
    /// measured: an alert that can never fire is indistinguishable from a
    /// system that is fine.
    fn default_rules() -> Vec<AlertRule> {
        vec![
            AlertRule {
                rule_id: "high_memory_usage".to_string(),
                name: "High Memory Usage".to_string(),
                condition: "resident memory share > threshold".to_string(),
                threshold_value: 90.0,
                severity: "High".to_string(),
                enabled: true,
                created_at: Instant::now(),
            },
            AlertRule {
                rule_id: "high_cpu_usage".to_string(),
                name: "High CPU Usage".to_string(),
                condition: "cpu usage > threshold".to_string(),
                threshold_value: 95.0,
                severity: "High".to_string(),
                enabled: true,
                created_at: Instant::now(),
            },
        ]
    }

    /// Evaluate every enabled rule against one real snapshot.
    ///
    /// Alerts whose rule no longer trips are cleared, so
    /// [`Self::get_active_alerts`] reflects the current state rather than
    /// accumulating forever.
    pub(crate) fn evaluate(
        &mut self,
        metrics: &MobileMetricsSnapshot,
    ) -> Result<Vec<PerformanceAlert>> {
        let mut triggered = Vec::new();

        for rule in self.alert_rules.clone() {
            if !rule.enabled {
                continue;
            }
            let Some(measured) = Self::measure_for_rule(&rule, metrics) else {
                self.active_alerts.remove(&rule.rule_id);
                continue;
            };
            if measured <= rule.threshold_value {
                self.active_alerts.remove(&rule.rule_id);
                continue;
            }

            let alert = PerformanceAlert {
                id: rule.rule_id.clone(),
                alert_type: match rule.rule_id.as_str() {
                    "high_memory_usage" => AlertType::MemoryPressure,
                    _ => AlertType::PerformanceDegradation,
                },
                severity: match rule.severity.as_str() {
                    "Info" => AlertSeverity::Info,
                    "Critical" => AlertSeverity::Critical,
                    "High" | "Error" => AlertSeverity::Error,
                    _ => AlertSeverity::Warning,
                },
                message: format!(
                    "{}: measured {:.1}, rule threshold {:.1}",
                    rule.name, measured, rule.threshold_value
                ),
                timestamp: metrics.timestamp,
                suggested_action: match rule.rule_id.as_str() {
                    "high_memory_usage" => {
                        "Reduce batch size or release cached tensors".to_string()
                    },
                    _ => "Reduce concurrent work or enable hardware acceleration".to_string(),
                },
            };

            self.alert_history.push_back(AlertRecord {
                alert_id: rule.rule_id.clone(),
                timestamp: Instant::now(),
                alert_type: rule.name.clone(),
                severity: rule.severity.clone(),
                message: alert.message.clone(),
                resolved: false,
                resolution_time: None,
            });
            while self.alert_history.len() > 256 {
                self.alert_history.pop_front();
            }
            self.active_alerts.insert(rule.rule_id.clone(), alert.clone());
            triggered.push(alert);
        }

        Ok(triggered)
    }

    /// The measured value a rule is compared against, or `None` when that
    /// metric was not measured on this device.
    fn measure_for_rule(rule: &AlertRule, metrics: &MobileMetricsSnapshot) -> Option<f32> {
        match rule.rule_id.as_str() {
            "high_memory_usage" => {
                metrics.memory.as_ref().and_then(|memory| memory.resident_share_percent())
            },
            "high_cpu_usage" => metrics.cpu.as_ref().map(|cpu| cpu.usage_percent),
            _ => None,
        }
    }

    pub(crate) fn get_active_alerts(&self) -> Vec<PerformanceAlert> {
        self.active_alerts.values().cloned().collect()
    }
}

impl PerformanceAnalyzer {
    pub(crate) fn new(_config: MobileProfilerConfig) -> Result<Self> {
        Ok(Self {
            config: AnalysisConfig::default(),
            analysis_cache: HashMap::new(),
            trend_data: VecDeque::new(),
        })
    }

    /// Assess system health from one real metrics snapshot.
    ///
    /// Every component score below is a function of a measured value. The
    /// previous body scored "cpu" from `trend_data.iter().take(10).count()`
    /// (a count of stored data points, not a CPU reading) and "memory" from
    /// `analysis_cache.len()`; because `performance_models` was a
    /// never-populated stub, the CPU branch was unreachable and the score was
    /// always exactly 85.0. Those were constants wearing arithmetic.
    ///
    /// Components the device does not measure are omitted from
    /// `component_scores` and from the average rather than scored at a
    /// default, so `overall_score` is only ever an average over real data.
    /// `Ok(None)` when nothing at all was measurable.
    pub(crate) fn assess_health(
        &self,
        metrics: &MobileMetricsSnapshot,
    ) -> Result<Option<SystemHealth>> {
        let mut component_scores: HashMap<String, f32> = HashMap::new();
        let mut recommendations = Vec::new();

        if let Some(cpu) = metrics.cpu.as_ref() {
            let cpu_score = (100.0 - cpu.usage_percent).clamp(0.0, 100.0);
            component_scores.insert("cpu".to_string(), cpu_score);
            if cpu_score < 70.0 {
                recommendations.push(format!(
                    "CPU at {:.1}% -- reduce concurrent work or enable hardware acceleration",
                    cpu.usage_percent
                ));
            }
        }

        if let Some(share) =
            metrics.memory.as_ref().and_then(|memory| memory.resident_share_percent())
        {
            let memory_score = (100.0 - share).clamp(0.0, 100.0);
            component_scores.insert("memory".to_string(), memory_score);
            if memory_score < 70.0 {
                recommendations.push(format!(
                    "Resident memory at {:.1}% of usable -- reduce batch size or release caches",
                    share
                ));
            }
        }

        if let Some(thermal) = metrics.thermal.as_ref() {
            let thermal_score = match thermal.thermal_state {
                ThermalState::Nominal => Some(100.0),
                ThermalState::Fair => Some(80.0),
                ThermalState::Serious => Some(60.0),
                ThermalState::Critical => Some(20.0),
                ThermalState::Emergency => Some(5.0),
                ThermalState::Shutdown => Some(0.0),
                // `ThermalMetrics.thermal_state` is only ever populated (see
                // `SystemMetricsCollector::collect_thermal_metrics`,
                // collector.rs:404-407) from a genuinely measured
                // `hottest_component_celsius()` reading bucketed by
                // `thermal_state_for`, which never produces `Unknown` -- this
                // arm exists only to satisfy the shared enum's
                // exhaustiveness. Excluded from the component scores for the
                // same reason an absent `metrics.thermal` already is, rather
                // than contributing a fabricated number.
                ThermalState::Unknown => None,
            };
            if let Some(thermal_score) = thermal_score {
                component_scores.insert("thermal".to_string(), thermal_score);
                if thermal_score < 70.0 {
                    recommendations.push(format!(
                        "Device at {:.1} C -- reduce inference frequency to allow cooldown",
                        thermal.temperature_c
                    ));
                }
            }
        }

        if metrics.inference.total_inferences > 0 {
            // Latency scored against a 500 ms ceiling: at or beyond it the
            // component scores zero, at 0 ms it scores 100.
            let latency_score =
                (100.0 - (metrics.inference.avg_latency_ms as f32 / 5.0)).clamp(0.0, 100.0);
            component_scores.insert("inference".to_string(), latency_score);
            if latency_score < 70.0 {
                recommendations.push(format!(
                    "Mean inference latency {:.1} ms over {} inferences -- consider quantization",
                    metrics.inference.avg_latency_ms, metrics.inference.total_inferences
                ));
            }
        }

        if component_scores.is_empty() {
            return Ok(None);
        }

        let overall_score = component_scores.values().sum::<f32>() / component_scores.len() as f32;

        let status = match overall_score {
            score if score >= 90.0 => HealthStatus::Excellent,
            score if score >= 75.0 => HealthStatus::Good,
            score if score >= 60.0 => HealthStatus::Healthy,
            score if score >= 45.0 => HealthStatus::Fair,
            score if score >= 30.0 => HealthStatus::Poor,
            _ => HealthStatus::Critical,
        };

        if recommendations.is_empty() {
            recommendations.push(format!(
                "No component scored below 70; health assessed from {} measured component(s)",
                component_scores.len()
            ));
        }

        Ok(Some(SystemHealth {
            overall_score,
            component_scores,
            status,
            recommendations,
        }))
    }
}

// =============================================================================
// DEFAULT IMPLEMENTATIONS
// =============================================================================

impl Default for ProfilingState {
    fn default() -> Self {
        Self {
            is_active: false,
            current_session_id: None,
            start_time: None,
            total_duration: Duration::ZERO,
            events_recorded: 0,
            snapshots_taken: 0,
            last_error: None,
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Get platform-specific capabilities
pub(crate) fn get_platform_capabilities() -> PlatformCapabilities {
    let mut capabilities = PlatformCapabilities::default();

    #[cfg(target_os = "ios")]
    {
        capabilities.ios_features = vec![
            "Metal".to_string(),
            "CoreML".to_string(),
            "Instruments".to_string(),
            "iOS Memory Pressure".to_string(),
        ];
    }

    #[cfg(target_os = "android")]
    {
        capabilities.android_features = vec![
            "NNAPI".to_string(),
            "GPU Delegate".to_string(),
            "Android Profiler".to_string(),
            "System Trace".to_string(),
        ];
    }

    capabilities.generic_features = vec![
        "CPU Profiling".to_string(),
        "Memory Profiling".to_string(),
        "Network Monitoring".to_string(),
        "Battery Monitoring".to_string(),
        "Thermal Monitoring".to_string(),
    ];

    capabilities
}

// =============================================================================
// COMPREHENSIVE TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mobile_performance_profiler::ExportFormat;
    use std::time::Duration;

    /// Create a fast test config with minimal overhead for tests
    pub(crate) fn fast_test_config() -> MobileProfilerConfig {
        // Start with default config
        let mut config = MobileProfilerConfig::default();

        // Disable all expensive profiling
        config.memory_profiling.enabled = false;
        config.cpu_profiling.enabled = false;
        config.gpu_profiling.enabled = false;
        config.network_profiling.enabled = false;
        config.real_time_monitoring.enabled = false;

        // Slow down sampling
        config.sampling.interval_ms = 10000; // 10 seconds
        config.sampling.max_samples = 10;

        config
    }

    #[test]
    pub(crate) fn test_profiler_creation() {
        let config = fast_test_config();
        let result = MobilePerformanceProfiler::new(config);
        assert!(
            result.is_ok(),
            "Failed to create profiler: {:?}",
            result.err()
        );
    }

    #[test]
    pub(crate) fn test_profiling_lifecycle() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        // Test initial state
        assert!(!profiler.is_profiling_active());

        // Test start profiling
        let session_id = profiler.start_profiling()?;
        assert!(!session_id.is_empty());
        assert!(profiler.is_profiling_active());

        // Test double start (should fail)
        assert!(profiler.start_profiling().is_err());

        // Test stop profiling
        let profiling_data = profiler.stop_profiling()?;
        assert!(!profiler.is_profiling_active());
        assert_eq!(profiling_data.session_info.metadata.session_id, session_id);

        Ok(())
    }

    #[test]
    pub(crate) fn test_event_recording() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;

        // Record various events
        profiler.record_inference_event("model_load", Some(250.0))?;
        profiler.record_inference_event("inference_start", None)?;
        profiler.record_inference_event("inference_end", Some(85.0))?;

        let profiling_data = profiler.stop_profiling()?;
        assert_eq!(profiling_data.events.len(), 3);

        Ok(())
    }

    #[test]
    pub(crate) fn test_metrics_collection() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;

        // Allow some time for metrics collection
        std::thread::sleep(Duration::from_millis(1));

        let metrics = profiler.get_current_metrics()?;
        assert!(metrics.timestamp > 0);

        let _stats = profiler.get_collection_stats()?;

        profiler.stop_profiling()?;
        Ok(())
    }

    /// Was `#[ignore]`d with "60+ second delays (likely thread/deadlock
    /// issue)". Investigated 2026-08-24: `detect_bottlenecks` ->
    /// `measure_now` locks `metrics_collector` once and calls straight-line
    /// methods on it (`collect_metrics`, `get_current_snapshot`); nothing in
    /// that path spawns a thread, waits on a channel, or blocks on a
    /// `Condvar`. `RealTimeMonitor::start_monitoring`/`stop_monitoring`
    /// (called from `start_profiling`/`stop_profiling`) never populate
    /// `_monitor_thread`, so there is no background thread here either.
    /// Reliably completes in well under a second, run standalone or as part
    /// of the full suite; the stale FIXME predates this module's rewrite.
    #[test]
    pub(crate) fn test_bottleneck_detection() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;

        // Give profiler time to collect initial metrics
        std::thread::sleep(std::time::Duration::from_millis(1));

        let _bottlenecks = profiler.detect_bottlenecks()?;
        // Should not crash and return a vector (may be empty)

        profiler.stop_profiling()?;

        // Give background tasks time to clean up
        std::thread::sleep(std::time::Duration::from_millis(1));
        Ok(())
    }

    #[test]
    pub(crate) fn test_optimization_suggestions() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;

        let _suggestions = profiler.get_optimization_suggestions()?;
        // Should not crash and return a vector (may be empty)

        profiler.stop_profiling()?;
        Ok(())
    }

    #[test]
    pub(crate) fn test_pause_resume_profiling() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;
        assert!(profiler.is_profiling_active());

        profiler.pause_profiling()?;
        assert!(profiler.is_profiling_active()); // Still active, just paused

        profiler.resume_profiling()?;
        assert!(profiler.is_profiling_active());

        profiler.stop_profiling()?;
        assert!(!profiler.is_profiling_active());

        Ok(())
    }

    #[test]
    pub(crate) fn test_config_validation() {
        let mut config = MobileProfilerConfig::default();

        // Test valid config
        assert!(MobilePerformanceProfiler::validate_config(&config).is_ok());

        // Test invalid sampling interval
        config.sampling.interval_ms = 0;
        assert!(MobilePerformanceProfiler::validate_config(&config).is_err());

        // Reset and test invalid max samples
        config = MobileProfilerConfig::default();
        config.sampling.max_samples = 0;
        assert!(MobilePerformanceProfiler::validate_config(&config).is_err());

        // Reset and test invalid compression level
        config = MobileProfilerConfig::default();
        config.export_config.compression_level = 10;
        assert!(MobilePerformanceProfiler::validate_config(&config).is_err());
    }

    #[test]
    pub(crate) fn test_config_update() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        let mut new_config = MobileProfilerConfig::default();
        new_config.sampling.interval_ms = 200;
        new_config.memory_profiling.heap_analysis = true;

        profiler.update_config(new_config)?;

        Ok(())
    }

    #[test]
    pub(crate) fn test_export_functionality() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;
        profiler.record_inference_event("test_event", Some(100.0))?;
        profiler.stop_profiling()?;

        let export_path = profiler.export_data(ExportFormat::JSON)?;
        assert!(!export_path.is_empty());

        Ok(())
    }

    #[test]
    pub(crate) fn test_system_health_assessment() -> Result<()> {
        // With memory and CPU profiling switched off, neither may appear as a
        // scored component. The old implementation always reported both, from
        // four hardcoded constants, regardless of what was measured.
        let profiler = MobilePerformanceProfiler::new(fast_test_config())?;
        profiler.start_profiling()?;
        if let Some(health) = profiler.get_system_health()? {
            assert!(!health.component_scores.contains_key("cpu"));
            assert!(!health.component_scores.contains_key("memory"));
        }
        profiler.stop_profiling()?;

        // With CPU profiling enabled the assessment is real, and every score
        // it contains is a percentage derived from a measurement.
        let mut measuring_config = fast_test_config();
        measuring_config.cpu_profiling.enabled = true;
        let profiler = MobilePerformanceProfiler::new(measuring_config)?;
        profiler.start_profiling()?;
        let health = profiler.get_system_health()?.expect("cpu profiling was enabled");
        assert!(health.overall_score >= 0.0 && health.overall_score <= 100.0);
        assert!(health.component_scores.contains_key("cpu"));
        for score in health.component_scores.values() {
            assert!(*score >= 0.0 && *score <= 100.0);
        }
        profiler.stop_profiling()?;
        Ok(())
    }

    #[test]
    pub(crate) fn test_performance_report_generation() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;
        profiler.record_inference_event("test_inference", Some(50.0))?;
        profiler.stop_profiling()?;

        let report = profiler.generate_performance_report()?;
        assert!(!report.is_empty());
        assert!(report.contains("html")); // Should be HTML format

        Ok(())
    }

    #[test]
    pub(crate) fn test_session_state_tracking() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        // Test initial state
        let state = profiler.get_profiling_state();
        assert!(!state.is_active);
        assert_eq!(state.events_recorded, 0);

        // Start profiling and check state
        profiler.start_profiling()?;
        let state = profiler.get_profiling_state();
        assert!(state.is_active);
        assert!(state.current_session_id.is_some());
        assert!(state.start_time.is_some());

        // Record events and check counter
        profiler.record_inference_event("event1", None)?;
        profiler.record_inference_event("event2", None)?;
        let state = profiler.get_profiling_state();
        assert_eq!(state.events_recorded, 2);

        profiler.stop_profiling()?;
        Ok(())
    }

    #[test]
    pub(crate) fn test_error_handling() {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config).expect("Failed to create profiler");

        // Test operations on inactive profiler
        assert!(profiler.stop_profiling().is_err());
        assert!(profiler.pause_profiling().is_err());
        assert!(profiler.resume_profiling().is_err());

        // Test invalid config updates
        let mut invalid_config = MobileProfilerConfig::default();
        invalid_config.sampling.interval_ms = 0;
        assert!(profiler.update_config(invalid_config).is_err());
    }

    #[test]
    pub(crate) fn test_thread_safety() -> Result<()> {
        use std::sync::Arc;
        use std::thread;

        let config = fast_test_config();
        let profiler = Arc::new(MobilePerformanceProfiler::new(config)?);

        profiler.start_profiling()?;

        // Spawn multiple threads to test concurrent access
        let handles: Vec<_> = (0..5)
            .map(|i| {
                let profiler_clone = Arc::clone(&profiler);
                thread::spawn(move || {
                    for j in 0..10 {
                        let event_name = format!("thread_{}_event_{}", i, j);
                        let _ = profiler_clone.record_inference_event(&event_name, Some(10.0));
                        let _ = profiler_clone.get_current_metrics();
                        let _ = profiler_clone.detect_bottlenecks();
                    }
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let profiling_data = profiler.stop_profiling()?;

        // Should have recorded events from multiple threads
        assert!(!profiling_data.events.is_empty());

        Ok(())
    }

    /// Regression: `BottleneckDetector::new` used to build with
    /// `detection_rules: Vec::new()` and nothing ever populated it, so
    /// `detect_bottlenecks()` was structurally incapable of returning
    /// anything -- a permanent clean bill of health.
    #[test]
    pub(crate) fn test_detector_has_real_rules_and_fires_on_real_data() {
        let mut detector =
            BottleneckDetector::new(fast_test_config()).expect("detector constructs");
        assert!(
            !detector.detection_rules.is_empty(),
            "a detector with no rules can never detect anything"
        );

        // A snapshot that genuinely exceeds the CPU and latency thresholds.
        let mut metrics = MobileMetricsSnapshot {
            timestamp: 1,
            ..Default::default()
        };
        metrics.cpu = Some(CpuMetrics {
            usage_percent: 97.0,
            ..Default::default()
        });
        metrics.inference.total_inferences = 50;
        metrics.inference.avg_latency_ms = 900.0;

        let detected = detector.analyze(&metrics).expect("analyze");
        let types: Vec<_> = detected.iter().map(|b| b.bottleneck_type).collect();
        assert!(
            types.contains(&BottleneckType::CPU),
            "high CPU must trip a rule"
        );
        assert!(
            types.contains(&BottleneckType::Latency),
            "high latency must trip a rule"
        );
        assert_eq!(
            detector.detection_stats.total_detections as usize,
            detected.len()
        );

        // A healthy snapshot clears them again rather than latching forever.
        let calm = MobileMetricsSnapshot {
            timestamp: 2,
            ..Default::default()
        };
        let cleared = detector.analyze(&calm).expect("analyze");
        assert!(cleared.is_empty());
        assert!(detector.get_active_bottlenecks().is_empty());
    }

    /// Rules whose metric family is unmeasured must not fire. A snapshot with
    /// no GPU/thermal/battery telemetry is not evidence that those are fine.
    #[test]
    pub(crate) fn test_unmeasured_families_never_trip_a_rule() {
        let mut detector = BottleneckDetector::new(fast_test_config()).expect("detector");
        let metrics = MobileMetricsSnapshot {
            timestamp: 1,
            ..Default::default()
        };
        assert!(metrics.gpu.is_none() && metrics.thermal.is_none() && metrics.battery.is_none());

        let detected = detector.analyze(&metrics).expect("analyze");
        for bottleneck in &detected {
            assert!(
                !matches!(
                    bottleneck.bottleneck_type,
                    BottleneckType::GPU | BottleneckType::Thermal | BottleneckType::Power
                ),
                "unmeasured family reported a bottleneck: {:?}",
                bottleneck.bottleneck_type
            );
        }
    }

    /// Regression: `get_current_health` scored "cpu" from
    /// `trend_data.take(10).count()` behind an always-true
    /// `performance_models.is_empty()` guard, so it returned exactly 85.0
    /// forever. Health must now move with the measurement.
    #[test]
    pub(crate) fn test_health_scores_track_measurements() {
        let analyzer = PerformanceAnalyzer::new(fast_test_config()).expect("analyzer");

        let mut idle = MobileMetricsSnapshot {
            timestamp: 1,
            ..Default::default()
        };
        idle.cpu = Some(CpuMetrics {
            usage_percent: 5.0,
            ..Default::default()
        });
        let mut busy = idle.clone();
        busy.cpu = Some(CpuMetrics {
            usage_percent: 95.0,
            ..Default::default()
        });

        let idle_health = analyzer.assess_health(&idle).expect("assess").expect("cpu measured");
        let busy_health = analyzer.assess_health(&busy).expect("assess").expect("cpu measured");

        let idle_cpu = idle_health.component_scores.get("cpu").copied().expect("cpu score");
        let busy_cpu = busy_health.component_scores.get("cpu").copied().expect("cpu score");
        assert!(
            idle_cpu > busy_cpu,
            "cpu health did not respond to cpu usage: {} vs {}",
            idle_cpu,
            busy_cpu
        );
        assert_ne!(idle_cpu, 85.0);
        assert_ne!(busy_cpu, 85.0);

        // Unmeasured families contribute no score at all.
        assert!(!idle_health.component_scores.contains_key("thermal"));
        assert!(!idle_health.component_scores.contains_key("gpu"));
    }

    /// Regression: `AlertManager::new` built an empty rule list, so
    /// `get_active_alerts()` could only ever be empty.
    #[test]
    pub(crate) fn test_alert_manager_evaluates_real_rules() {
        let mut manager = AlertManager::new(fast_test_config()).expect("manager");
        assert!(!manager.alert_rules.is_empty());

        let mut hot = MobileMetricsSnapshot {
            timestamp: 1,
            ..Default::default()
        };
        hot.cpu = Some(CpuMetrics {
            usage_percent: 99.0,
            ..Default::default()
        });
        let triggered = manager.evaluate(&hot).expect("evaluate");
        assert!(!triggered.is_empty(), "99% CPU must raise the CPU alert");
        assert!(triggered.iter().any(|alert| alert.id == "high_cpu_usage"));

        let mut calm = hot.clone();
        calm.cpu = Some(CpuMetrics {
            usage_percent: 3.0,
            ..Default::default()
        });
        manager.evaluate(&calm).expect("evaluate");
        assert!(
            manager.get_active_alerts().iter().all(|alert| alert.id != "high_cpu_usage"),
            "an alert whose condition cleared must not stay active"
        );
    }
}
