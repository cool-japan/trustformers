//! Automated performance regression detection.
//!
//! Compares current profiling measurements against a historical baseline and
//! produces severity-graded alerts together with actionable recommendations.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// Data types
// ============================================================================

/// A single performance measurement (latency, memory, throughput) at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfMeasurement {
    /// Component or layer name, e.g. `"attention_forward"`.
    pub name: String,
    /// Wall-clock latency in milliseconds.
    pub latency_ms: f64,
    /// Peak memory allocated in megabytes.
    pub memory_mb: f64,
    /// Throughput expressed as tokens/sec or samples/sec.
    pub throughput: f64,
    /// Unix timestamp (seconds) of when this measurement was taken.
    pub timestamp_secs: u64,
}

/// A historical baseline used as the reference for regression comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfBaseline {
    /// All raw measurements that make up this baseline.
    pub measurements: Vec<PerfMeasurement>,
    /// Unix timestamp (seconds) when this baseline was captured.
    pub created_at: u64,
    /// Human-readable description, e.g. `"v0.1.0 release candidate"`.
    pub description: String,
}

impl PerfBaseline {
    /// Compute per-metric statistics for a named component. Returns `None` when
    /// fewer than two data points are present (not enough to compute std-dev).
    pub fn stats_for(&self, name: &str) -> Option<BaselineStats> {
        let latencies: Vec<f64> = self
            .measurements
            .iter()
            .filter(|m| m.name == name)
            .map(|m| m.latency_ms)
            .collect();

        if latencies.len() < 2 {
            return None;
        }

        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let variance = latencies.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (latencies.len() - 1) as f64;
        let std = variance.sqrt();

        let mut sorted = latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        Some(BaselineStats {
            mean_latency_ms: mean,
            std_latency_ms: std,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
        })
    }
}

/// Statistical summary of baseline latency for a single component.
#[derive(Debug, Clone)]
pub struct BaselineStats {
    pub mean_latency_ms: f64,
    pub std_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
}

// ============================================================================
// Regression alerts
// ============================================================================

/// Which performance metric regressed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionMetric {
    Latency,
    Memory,
    Throughput,
}

/// How severe the regression is.
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// 5–10 % regression.
    Minor,
    /// 10–25 % regression.
    Moderate,
    /// 25–50 % regression.
    Severe,
    /// > 50 % regression.
    Critical,
}

impl RegressionSeverity {
    /// Derive severity from an absolute regression percentage value.
    pub fn from_pct(pct: f64) -> Self {
        match pct.abs() as u64 {
            0..=10 => Self::Minor,
            11..=25 => Self::Moderate,
            26..=50 => Self::Severe,
            _ => Self::Critical,
        }
    }
}

/// A single regression alert.
#[derive(Debug, Clone)]
pub struct RegressionAlert {
    /// Component name.
    pub name: String,
    /// Which metric regressed.
    pub metric: RegressionMetric,
    /// Derived severity class.
    pub severity: RegressionSeverity,
    /// Reference value from the baseline.
    pub baseline_value: f64,
    /// Current (measured) value.
    pub current_value: f64,
    /// Signed percentage change: positive = worse for latency/memory, negative = worse for throughput.
    pub regression_pct: f64,
    /// Human-readable summary.
    pub message: String,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration knobs for the regression detector.
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    /// Minimum absolute % change to report (default: 5.0).
    pub min_regression_pct: f64,
    /// Z-score threshold for statistical significance (default: 2.0).
    pub z_score_threshold: f64,
    /// Monitor latency regressions (default: true).
    pub monitor_latency: bool,
    /// Monitor memory regressions (default: true).
    pub monitor_memory: bool,
    /// Monitor throughput regressions (default: true).
    pub monitor_throughput: bool,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            min_regression_pct: 5.0,
            z_score_threshold: 2.0,
            monitor_latency: true,
            monitor_memory: true,
            monitor_throughput: true,
        }
    }
}

// ============================================================================
// Detector
// ============================================================================

/// Compares profiling measurements against a stored baseline and emits alerts.
pub struct RegressionDetector {
    config: RegressionConfig,
}

impl RegressionDetector {
    pub fn new(config: RegressionConfig) -> Self {
        Self { config }
    }

    /// Detect regressions by comparing `current` measurements to the `baseline`.
    ///
    /// For each named component that appears in both sets the detector computes:
    /// * mean current value vs. baseline mean
    /// * optional z-score significance check (when baseline has ≥ 2 points)
    ///
    /// Returns every alert whose absolute regression percentage exceeds
    /// `config.min_regression_pct`.
    pub fn detect(
        &self,
        current: &[PerfMeasurement],
        baseline: &PerfBaseline,
    ) -> Vec<RegressionAlert> {
        // Group current measurements by name.
        let mut current_by_name: HashMap<&str, Vec<&PerfMeasurement>> = HashMap::new();
        for m in current {
            current_by_name.entry(m.name.as_str()).or_default().push(m);
        }

        let mut alerts = Vec::new();

        for (name, measurements) in &current_by_name {
            // Baseline aggregate for this component.
            let baseline_measurements: Vec<&PerfMeasurement> = baseline
                .measurements
                .iter()
                .filter(|m| m.name == *name)
                .collect();

            if baseline_measurements.is_empty() {
                continue;
            }

            let current_mean_latency =
                measurements.iter().map(|m| m.latency_ms).sum::<f64>() / measurements.len() as f64;
            let current_mean_memory =
                measurements.iter().map(|m| m.memory_mb).sum::<f64>() / measurements.len() as f64;
            let current_mean_throughput =
                measurements.iter().map(|m| m.throughput).sum::<f64>() / measurements.len() as f64;

            let baseline_mean_latency = baseline_measurements.iter().map(|m| m.latency_ms).sum::<f64>()
                / baseline_measurements.len() as f64;
            let baseline_mean_memory = baseline_measurements.iter().map(|m| m.memory_mb).sum::<f64>()
                / baseline_measurements.len() as f64;
            let baseline_mean_throughput = baseline_measurements
                .iter()
                .map(|m| m.throughput)
                .sum::<f64>()
                / baseline_measurements.len() as f64;

            let stats = baseline.stats_for(name);

            // --- latency (higher is worse) ---
            if self.config.monitor_latency && baseline_mean_latency > 0.0 {
                let pct =
                    (current_mean_latency - baseline_mean_latency) / baseline_mean_latency * 100.0;
                if pct > self.config.min_regression_pct
                    && self.is_significant(
                        current_mean_latency,
                        baseline_mean_latency,
                        stats.as_ref().map(|s| s.std_latency_ms),
                    )
                {
                    let severity = RegressionSeverity::from_pct(pct);
                    alerts.push(RegressionAlert {
                        name: name.to_string(),
                        metric: RegressionMetric::Latency,
                        severity: severity.clone(),
                        baseline_value: baseline_mean_latency,
                        current_value: current_mean_latency,
                        regression_pct: pct,
                        message: format!(
                            "[{name}] Latency regressed by {pct:.1}% ({baseline_mean_latency:.2}ms → {current_mean_latency:.2}ms) — severity: {severity:?}",
                        ),
                    });
                }
            }

            // --- memory (higher is worse) ---
            if self.config.monitor_memory && baseline_mean_memory > 0.0 {
                let pct =
                    (current_mean_memory - baseline_mean_memory) / baseline_mean_memory * 100.0;
                if pct > self.config.min_regression_pct {
                    let severity = RegressionSeverity::from_pct(pct);
                    alerts.push(RegressionAlert {
                        name: name.to_string(),
                        metric: RegressionMetric::Memory,
                        severity: severity.clone(),
                        baseline_value: baseline_mean_memory,
                        current_value: current_mean_memory,
                        regression_pct: pct,
                        message: format!(
                            "[{name}] Memory regressed by {pct:.1}% ({baseline_mean_memory:.1}MB → {current_mean_memory:.1}MB) — severity: {severity:?}",
                        ),
                    });
                }
            }

            // --- throughput (lower is worse) ---
            if self.config.monitor_throughput && baseline_mean_throughput > 0.0 {
                let pct = (baseline_mean_throughput - current_mean_throughput)
                    / baseline_mean_throughput
                    * 100.0;
                if pct > self.config.min_regression_pct {
                    let severity = RegressionSeverity::from_pct(pct);
                    alerts.push(RegressionAlert {
                        name: name.to_string(),
                        metric: RegressionMetric::Throughput,
                        severity: severity.clone(),
                        baseline_value: baseline_mean_throughput,
                        current_value: current_mean_throughput,
                        regression_pct: pct,
                        message: format!(
                            "[{name}] Throughput dropped by {pct:.1}% ({baseline_mean_throughput:.1} → {current_mean_throughput:.1} tok/s) — severity: {severity:?}",
                        ),
                    });
                }
            }
        }

        // Sort by severity (worst first) so consumers see critical issues first.
        alerts.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
        alerts
    }

    /// Render alerts as a human-readable text report.
    pub fn report(&self, alerts: &[RegressionAlert]) -> String {
        if alerts.is_empty() {
            return "No regressions detected — all metrics within acceptable range.\n".to_string();
        }

        let mut out = String::new();
        out.push_str("=== Performance Regression Report ===\n\n");
        out.push_str(&format!("Total alerts: {}\n\n", alerts.len()));

        for (i, alert) in alerts.iter().enumerate() {
            out.push_str(&format!("{}. {}\n", i + 1, alert.message));
        }

        out.push('\n');
        out.push_str(&format!(
            "Critical: {}  Severe: {}  Moderate: {}  Minor: {}\n",
            alerts.iter().filter(|a| a.severity == RegressionSeverity::Critical).count(),
            alerts.iter().filter(|a| a.severity == RegressionSeverity::Severe).count(),
            alerts.iter().filter(|a| a.severity == RegressionSeverity::Moderate).count(),
            alerts.iter().filter(|a| a.severity == RegressionSeverity::Minor).count(),
        ));
        out
    }

    /// Persist a `PerfBaseline` as JSON to `path`.
    pub fn save_baseline(&self, baseline: &PerfBaseline, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(baseline)
            .context("failed to serialize baseline")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create baseline directory: {}", parent.display()))?;
        }
        std::fs::write(path, json)
            .with_context(|| format!("failed to write baseline: {}", path.display()))?;
        tracing::info!(path = %path.display(), "saved performance baseline");
        Ok(())
    }

    /// Load a `PerfBaseline` from a JSON file.
    pub fn load_baseline(&self, path: &Path) -> Result<PerfBaseline> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read baseline: {}", path.display()))?;
        let baseline: PerfBaseline =
            serde_json::from_str(&content).context("failed to deserialize baseline")?;
        Ok(baseline)
    }

    /// Build a new `PerfBaseline` from a set of measurements.
    pub fn create_baseline(measurements: Vec<PerfMeasurement>, description: &str) -> PerfBaseline {
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        PerfBaseline { measurements, created_at, description: description.to_string() }
    }

    /// Generate actionable recommendations for a list of alerts.
    pub fn recommendations(&self, alerts: &[RegressionAlert]) -> Vec<String> {
        let mut recs: Vec<String> = Vec::new();

        let has_latency = alerts.iter().any(|a| a.metric == RegressionMetric::Latency);
        let has_memory = alerts.iter().any(|a| a.metric == RegressionMetric::Memory);
        let has_throughput = alerts.iter().any(|a| a.metric == RegressionMetric::Throughput);
        let has_critical = alerts.iter().any(|a| a.severity == RegressionSeverity::Critical);
        let has_severe = alerts.iter().any(|a| a.severity == RegressionSeverity::Severe);

        if has_latency {
            recs.push("Profile forward/backward passes to identify new hot-spots (use flame_graph_profiler).".to_string());
            recs.push("Check for inadvertent Python or C FFI call sites introduced in recent commits.".to_string());
            recs.push("Consider operator fusion or kernel-level optimisations for attention / FFN layers.".to_string());
        }
        if has_memory {
            recs.push("Audit tensor lifetime to ensure activations are freed promptly after use.".to_string());
            recs.push("Enable gradient checkpointing or reduce batch size to stay within memory budget.".to_string());
            recs.push("Use memory_profiler to locate the largest allocations.".to_string());
        }
        if has_throughput {
            recs.push("Check data-loading pipeline: a slow DataLoader can mask compute regressions.".to_string());
            recs.push("Investigate whether new ops are preventing Tensor-Core utilisation.".to_string());
        }
        if has_critical || has_severe {
            recs.push("URGENT: consider reverting the most recent change and bisecting to isolate the regression.".to_string());
        }

        if recs.is_empty() {
            recs.push("No actionable recommendations — regressions are within acceptable bounds.".to_string());
        }

        recs
    }

    // ---------- helpers ----------

    /// True when the observed deviation is statistically significant (z-score > threshold).
    /// Falls back to `true` when no std-dev is available (always report the regression).
    fn is_significant(&self, current: f64, baseline_mean: f64, baseline_std: Option<f64>) -> bool {
        match baseline_std {
            Some(std) if std > 0.0 => {
                let z = (current - baseline_mean).abs() / std;
                z > self.config.z_score_threshold
            }
            _ => true,
        }
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn make_measurement(name: &str, latency_ms: f64, memory_mb: f64, throughput: f64) -> PerfMeasurement {
        PerfMeasurement {
            name: name.to_string(),
            latency_ms,
            memory_mb,
            throughput,
            timestamp_secs: 0,
        }
    }

    fn make_baseline(measurements: Vec<PerfMeasurement>) -> PerfBaseline {
        PerfBaseline {
            measurements,
            created_at: 0,
            description: "test baseline".to_string(),
        }
    }

    #[test]
    fn test_no_regression_when_within_threshold() {
        let detector = RegressionDetector::new(RegressionConfig::default());
        let baseline = make_baseline(vec![make_measurement("attn", 10.0, 100.0, 1000.0)]);
        let current = vec![make_measurement("attn", 10.2, 101.0, 998.0)]; // < 5%
        let alerts = detector.detect(&current, &baseline);
        assert!(alerts.is_empty(), "should not alert on < 5% changes");
    }

    #[test]
    fn test_latency_regression_detected() {
        let config = RegressionConfig { monitor_memory: false, monitor_throughput: false, ..Default::default() };
        let detector = RegressionDetector::new(config);
        let baseline = make_baseline(vec![
            make_measurement("attn", 10.0, 0.0, 0.0),
            make_measurement("attn", 10.0, 0.0, 0.0),
        ]);
        let current = vec![make_measurement("attn", 15.0, 0.0, 0.0)]; // 50% regression
        let alerts = detector.detect(&current, &baseline);
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].metric, RegressionMetric::Latency);
        assert_eq!(alerts[0].severity, RegressionSeverity::Severe);
    }

    #[test]
    fn test_memory_regression_detected() {
        let config = RegressionConfig { monitor_latency: false, monitor_throughput: false, ..Default::default() };
        let detector = RegressionDetector::new(config);
        let baseline = make_baseline(vec![make_measurement("ffn", 0.0, 100.0, 0.0)]);
        let current = vec![make_measurement("ffn", 0.0, 150.0, 0.0)]; // 50%
        let alerts = detector.detect(&current, &baseline);
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].metric, RegressionMetric::Memory);
    }

    #[test]
    fn test_throughput_regression_detected() {
        let config = RegressionConfig { monitor_latency: false, monitor_memory: false, ..Default::default() };
        let detector = RegressionDetector::new(config);
        let baseline = make_baseline(vec![make_measurement("decode", 0.0, 0.0, 1000.0)]);
        let current = vec![make_measurement("decode", 0.0, 0.0, 600.0)]; // 40%
        let alerts = detector.detect(&current, &baseline);
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].metric, RegressionMetric::Throughput);
    }

    #[test]
    fn test_severity_from_pct() {
        assert_eq!(RegressionSeverity::from_pct(7.0), RegressionSeverity::Minor);
        assert_eq!(RegressionSeverity::from_pct(15.0), RegressionSeverity::Moderate);
        assert_eq!(RegressionSeverity::from_pct(35.0), RegressionSeverity::Severe);
        assert_eq!(RegressionSeverity::from_pct(75.0), RegressionSeverity::Critical);
    }

    #[test]
    fn test_report_no_alerts() {
        let detector = RegressionDetector::new(RegressionConfig::default());
        let report = detector.report(&[]);
        assert!(report.contains("No regressions detected"));
    }

    #[test]
    fn test_save_and_load_baseline() -> Result<()> {
        let detector = RegressionDetector::new(RegressionConfig::default());
        let baseline = RegressionDetector::create_baseline(
            vec![make_measurement("attn", 10.0, 100.0, 500.0)],
            "test baseline v1",
        );
        let path = temp_dir().join(format!("baseline_{}.json", uuid::Uuid::new_v4()));
        detector.save_baseline(&baseline, &path)?;
        let loaded = detector.load_baseline(&path)?;
        assert_eq!(loaded.description, "test baseline v1");
        assert_eq!(loaded.measurements.len(), 1);
        Ok(())
    }

    #[test]
    fn test_recommendations_populated_for_latency() {
        let config = RegressionConfig { monitor_memory: false, monitor_throughput: false, ..Default::default() };
        let detector = RegressionDetector::new(config);
        let baseline = make_baseline(vec![
            make_measurement("a", 10.0, 0.0, 0.0),
            make_measurement("a", 10.0, 0.0, 0.0),
        ]);
        let current = vec![make_measurement("a", 20.0, 0.0, 0.0)];
        let alerts = detector.detect(&current, &baseline);
        let recs = detector.recommendations(&alerts);
        assert!(!recs.is_empty());
    }

    #[test]
    fn test_baseline_stats_for() {
        let baseline = make_baseline(vec![
            make_measurement("layer_a", 10.0, 0.0, 0.0),
            make_measurement("layer_a", 12.0, 0.0, 0.0),
            make_measurement("layer_a", 11.0, 0.0, 0.0),
        ]);
        let stats = baseline.stats_for("layer_a");
        assert!(stats.is_some());
        let s = stats.expect("should have stats");
        assert!((s.mean_latency_ms - 11.0).abs() < 0.1);
        assert!(s.std_latency_ms > 0.0);
    }

    // ── additional tests ──────────────────────────────────────────────────

    #[test]
    fn test_regression_config_default() {
        let cfg = RegressionConfig::default();
        assert!(cfg.monitor_latency);
        assert!(cfg.monitor_memory);
        assert!(cfg.monitor_throughput);
        assert!(cfg.min_regression_pct > 0.0);
        assert!(cfg.z_score_threshold > 0.0);
    }

    #[test]
    fn test_create_baseline_sets_description() {
        let baseline = RegressionDetector::create_baseline(
            vec![make_measurement("l0", 5.0, 50.0, 200.0)],
            "v1.0 baseline",
        );
        assert_eq!(baseline.description, "v1.0 baseline");
        assert_eq!(baseline.measurements.len(), 1);
    }

    #[test]
    fn test_create_baseline_empty_measurements() {
        let baseline = RegressionDetector::create_baseline(vec![], "empty");
        assert!(baseline.measurements.is_empty());
    }

    #[test]
    fn test_detect_no_baseline_data_no_alerts() {
        let detector = RegressionDetector::new(RegressionConfig::default());
        let baseline = make_baseline(vec![make_measurement("other_layer", 10.0, 100.0, 500.0)]);
        let current = vec![make_measurement("attn", 20.0, 200.0, 100.0)];
        // "attn" not in baseline → no alerts
        let alerts = detector.detect(&current, &baseline);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_detect_alerts_sorted_by_severity() {
        let detector = RegressionDetector::new(RegressionConfig::default());
        let baseline = make_baseline(vec![
            make_measurement("a", 10.0, 0.0, 0.0),
            make_measurement("a", 10.0, 0.0, 0.0),
            make_measurement("b", 10.0, 0.0, 0.0),
            make_measurement("b", 10.0, 0.0, 0.0),
        ]);
        let current = vec![
            make_measurement("a", 16.0, 0.0, 0.0), // ~60% → Critical
            make_measurement("b", 11.5, 0.0, 0.0), // ~15% → Moderate
        ];
        let cfg = RegressionConfig { monitor_memory: false, monitor_throughput: false, ..Default::default() };
        let det2 = RegressionDetector::new(cfg);
        let alerts = det2.detect(&current, &baseline);
        if alerts.len() >= 2 {
            assert!(alerts[0].severity >= alerts[1].severity);
        }
    }

    #[test]
    fn test_regression_metric_variants() {
        let metrics = [RegressionMetric::Latency, RegressionMetric::Memory, RegressionMetric::Throughput];
        for m in &metrics {
            assert!(!format!("{:?}", m).is_empty());
        }
    }

    #[test]
    fn test_regression_severity_variants() {
        let severities = [
            RegressionSeverity::Minor,
            RegressionSeverity::Moderate,
            RegressionSeverity::Severe,
            RegressionSeverity::Critical,
        ];
        for s in &severities {
            assert!(!format!("{:?}", s).is_empty());
        }
    }

    #[test]
    fn test_regression_alert_fields() {
        let alert = RegressionAlert {
            name: "attention".to_string(),
            metric: RegressionMetric::Latency,
            severity: RegressionSeverity::Severe,
            baseline_value: 10.0,
            current_value: 15.0,
            regression_pct: 50.0,
            message: "50% regression".to_string(),
        };
        assert_eq!(alert.name, "attention");
        assert!((alert.regression_pct - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_report_with_alerts_includes_count() {
        let detector = RegressionDetector::new(RegressionConfig::default());
        let alerts = vec![
            RegressionAlert {
                name: "a".to_string(),
                metric: RegressionMetric::Latency,
                severity: RegressionSeverity::Minor,
                baseline_value: 10.0,
                current_value: 10.5,
                regression_pct: 5.0,
                message: "Minor regression".to_string(),
            },
        ];
        let report = detector.report(&alerts);
        assert!(report.contains("Total alerts: 1"));
    }

    #[test]
    fn test_perf_measurement_fields() {
        let m = PerfMeasurement {
            name: "ffn".to_string(),
            latency_ms: 12.5,
            memory_mb: 256.0,
            throughput: 1024.0,
            timestamp_secs: 1000,
        };
        assert_eq!(m.name, "ffn");
        assert!((m.latency_ms - 12.5).abs() < 1e-6);
    }

    #[test]
    fn test_baseline_stats_for_single_measurement_returns_none() {
        // Need at least 2 measurements for stats
        let baseline = make_baseline(vec![make_measurement("solo", 10.0, 0.0, 0.0)]);
        assert!(baseline.stats_for("solo").is_none());
    }

    #[test]
    fn test_baseline_stats_p95_p99() {
        let measurements: Vec<PerfMeasurement> = (1..=10)
            .map(|i| make_measurement("l", i as f64 * 10.0, 0.0, 0.0))
            .collect();
        let baseline = make_baseline(measurements);
        let stats = baseline.stats_for("l").expect("should have stats");
        assert!(stats.p95_latency_ms >= stats.mean_latency_ms);
        assert!(stats.p99_latency_ms >= stats.p95_latency_ms);
    }
}
