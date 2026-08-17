// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Hardware monitoring and health checking components
//!
//! This module provides performance monitoring, anomaly detection, and health checking
//! capabilities for hardware devices in the TrustformeRS ecosystem.

use super::config::OperationStats;
use super::{HardwareMetrics, HardwareResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// Performance monitor for tracking hardware metrics and efficiency
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    /// Performance history for each device
    pub history: HashMap<String, Vec<(SystemTime, HardwareMetrics)>>,
    /// Operation statistics per device
    pub operation_stats: HashMap<String, OperationStats>,
    /// Device efficiency scores
    pub efficiency_scores: HashMap<String, f64>,
    /// Anomaly detector instance
    pub anomaly_detector: AnomalyDetector,
}

/// Anomaly detector for identifying unusual hardware behavior
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Anomaly detection thresholds
    pub thresholds: HashMap<String, f64>,
    /// Detected anomalies history
    pub anomalies: Vec<Anomaly>,
    /// Active detection algorithms
    pub algorithms: Vec<AnomalyAlgorithm>,
}

/// Anomaly information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Anomaly {
    /// Device ID where anomaly was detected
    pub device_id: String,
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Detection timestamp
    pub timestamp: SystemTime,
    /// Anomaly score (0.0 - 1.0)
    pub score: f64,
    /// Additional details
    pub details: String,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
}

/// Types of hardware anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyType {
    /// High latency detected
    HighLatency,
    /// Low throughput detected
    LowThroughput,
    /// High memory usage detected
    HighMemoryUsage,
    /// High temperature detected
    HighTemperature,
    /// High power consumption detected
    HighPowerConsumption,
    /// Frequent errors detected
    FrequentErrors,
    /// Device unavailable
    DeviceUnavailable,
    /// Performance degradation detected
    PerformanceDegradation,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalySeverity {
    /// Low severity - informational
    Low,
    /// Medium severity - attention needed
    Medium,
    /// High severity - action required
    High,
    /// Critical severity - immediate action required
    Critical,
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyAlgorithm {
    /// Statistical outlier detection
    StatisticalOutlier,
    /// Moving average based detection
    MovingAverage,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// Isolation forest algorithm
    IsolationForest,
    /// One-class SVM
    OneClassSVM,
    /// LSTM autoencoder
    LSTMAutoencoder,
}

/// Health checker for device health monitoring
#[derive(Debug, Clone)]
pub struct HealthChecker {
    /// Health check results per device
    pub results: HashMap<String, HealthCheckResult>,
    /// Health check schedules per device
    pub schedule: HashMap<String, Duration>,
    /// Health check policies per device
    pub policies: HashMap<String, HealthCheckPolicy>,
}

/// Health check result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Device ID
    pub device_id: String,
    /// Overall health status
    pub status: HealthStatus,
    /// Check timestamp
    pub timestamp: SystemTime,
    /// Response time in milliseconds
    pub response_time: f64,
    /// Health score (0.0 = unhealthy, 1.0 = perfect health)
    pub health_score: f64,
    /// Issues found during check
    pub issues: Vec<HealthIssue>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Health status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Device is healthy
    Healthy,
    /// Device has warnings
    Warning,
    /// Device has critical issues
    Critical,
    /// Device is unhealthy
    Unhealthy,
    /// Health status unknown
    Unknown,
}

/// Health issue details
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthIssue {
    /// Type of issue
    pub issue_type: HealthIssueType,
    /// Issue severity
    pub severity: AnomalySeverity,
    /// Human-readable description
    pub description: String,
    /// Components affected by this issue
    pub affected_components: Vec<String>,
    /// Potential root causes
    pub potential_causes: Vec<String>,
    /// Suggested fixes
    pub suggested_fixes: Vec<String>,
}

/// Types of health issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthIssueType {
    /// Performance degradation
    PerformanceDegradation,
    /// Memory leak detected
    MemoryLeak,
    /// High temperature
    HighTemperature,
    /// Power-related issues
    PowerIssues,
    /// Communication errors
    CommunicationErrors,
    /// Driver issues
    DriverIssues,
    /// Hardware faults
    HardwareFaults,
    /// Configuration issues
    ConfigurationIssues,
}

/// Health check policy configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthCheckPolicy {
    /// Check interval
    pub interval: Duration,
    /// Timeout for health checks
    pub timeout: Duration,
    /// Number of retries on failure
    pub retry_count: u32,
    /// Failure threshold before marking unhealthy
    pub failure_threshold: u32,
    /// Recovery threshold before marking healthy
    pub recovery_threshold: u32,
    /// Enable detailed diagnostics
    pub detailed_diagnostics: bool,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            history: HashMap::new(),
            operation_stats: HashMap::new(),
            efficiency_scores: HashMap::new(),
            anomaly_detector: AnomalyDetector::new(),
        }
    }

    /// Update metrics for a device
    pub fn update_metrics(&mut self, device_id: &str, metrics: &HardwareMetrics) {
        let entry = self.history.entry(device_id.to_string()).or_default();
        entry.push((SystemTime::now(), metrics.clone()));

        // Keep only last 1000 entries to prevent memory growth
        if entry.len() > 1000 {
            entry.drain(..500);
        }

        // Update anomaly detector
        self.anomaly_detector.check_anomalies(device_id, metrics);
    }

    /// Analyze performance for all devices
    pub fn analyze_performance(&mut self, device_metrics: &HashMap<String, HardwareMetrics>) {
        for (device_id, metrics) in device_metrics {
            let efficiency = self.calculate_efficiency_score(metrics);
            self.efficiency_scores.insert(device_id.clone(), efficiency);
        }
    }

    /// Calculate efficiency score for a device
    pub fn calculate_efficiency_score(&self, metrics: &HardwareMetrics) -> f64 {
        // Calculate utilization score (higher is better)
        let utilization_score = (metrics.utilization / 100.0).min(1.0);

        // Calculate latency score (lower is better)
        let latency_score = (1.0 / (1.0 + metrics.latency / 100.0)).min(1.0);

        // Calculate throughput score (higher is better, normalized)
        let throughput_score = (metrics.throughput / 1000.0).min(1.0);

        // Weighted average
        utilization_score * 0.4 + latency_score * 0.3 + throughput_score * 0.3
    }

    /// Get top performing devices
    pub fn get_top_performers(&self, count: usize) -> Vec<(String, f64)> {
        let mut performers: Vec<_> =
            self.efficiency_scores.iter().map(|(id, score)| (id.clone(), *score)).collect();

        performers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(::std::cmp::Ordering::Equal));
        performers.truncate(count);
        performers
    }
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new() -> Self {
        Self {
            thresholds: [
                ("high_latency".to_string(), 100.0),
                ("low_throughput".to_string(), 10.0),
                ("high_temperature".to_string(), 80.0),
                ("high_power".to_string(), 200.0),
                ("high_utilization".to_string(), 95.0),
            ]
            .into(),
            anomalies: Vec::new(),
            algorithms: vec![
                AnomalyAlgorithm::StatisticalOutlier,
                AnomalyAlgorithm::MovingAverage,
            ],
        }
    }

    /// Check for anomalies in device metrics
    pub fn check_anomalies(&mut self, device_id: &str, metrics: &HardwareMetrics) {
        // Check for high latency
        if metrics.latency > *self.thresholds.get("high_latency").unwrap_or(&100.0) {
            self.add_anomaly(
                device_id,
                AnomalyType::HighLatency,
                AnomalySeverity::High,
                metrics.latency / 100.0,
                "Latency exceeds threshold",
            );
        }

        // Check for low throughput
        if metrics.throughput < *self.thresholds.get("low_throughput").unwrap_or(&10.0) {
            self.add_anomaly(
                device_id,
                AnomalyType::LowThroughput,
                AnomalySeverity::Medium,
                1.0 - (metrics.throughput / 100.0),
                "Throughput below threshold",
            );
        }

        // Check for high temperature
        if let Some(temp) = metrics.temperature {
            if temp > *self.thresholds.get("high_temperature").unwrap_or(&80.0) {
                self.add_anomaly(
                    device_id,
                    AnomalyType::HighTemperature,
                    AnomalySeverity::Critical,
                    temp / 100.0,
                    "Temperature exceeds safe limits",
                );
            }
        }

        // Check for high power consumption
        if metrics.power_consumption > *self.thresholds.get("high_power").unwrap_or(&200.0) {
            self.add_anomaly(
                device_id,
                AnomalyType::HighPowerConsumption,
                AnomalySeverity::Medium,
                metrics.power_consumption / 300.0,
                "Power consumption is high",
            );
        }

        // Check for high utilization
        if metrics.utilization > *self.thresholds.get("high_utilization").unwrap_or(&95.0) {
            self.add_anomaly(
                device_id,
                AnomalyType::HighMemoryUsage,
                AnomalySeverity::Low,
                metrics.utilization / 100.0,
                "Utilization is very high",
            );
        }

        // Trim old anomalies to prevent memory growth
        if self.anomalies.len() > 1000 {
            self.anomalies.drain(..500);
        }
    }

    /// Add a new anomaly
    fn add_anomaly(
        &mut self,
        device_id: &str,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        score: f64,
        details: &str,
    ) {
        let anomaly = Anomaly {
            device_id: device_id.to_string(),
            anomaly_type,
            severity,
            timestamp: SystemTime::now(),
            score,
            details: details.to_string(),
            affected_metrics: vec!["latency".to_string(), "throughput".to_string()],
        };
        self.anomalies.push(anomaly);
    }

    /// Get recent anomalies for a device
    pub fn get_recent_anomalies(&self, device_id: &str, duration: Duration) -> Vec<&Anomaly> {
        let threshold = SystemTime::now() - duration;
        self.anomalies
            .iter()
            .filter(|a| a.device_id == device_id && a.timestamp >= threshold)
            .collect()
    }
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
            schedule: HashMap::new(),
            policies: HashMap::new(),
        }
    }

    /// Perform a real health check on a device from its current metrics.
    ///
    /// `metrics` should be the device's live `HardwareMetrics` (e.g. from
    /// `HardwareManager::get_device_metrics`). This diagnoses the real
    /// thermal, error-rate and utilization signals already carried on
    /// `HardwareMetrics` against fixed safety thresholds, rather than
    /// fabricating a constant 0.95/`Healthy` verdict regardless of device
    /// state. When no live telemetry is available for the device
    /// (`metrics` is `None`), the honest result is `HealthStatus::Unknown`
    /// -- a monitoring API must not report a device it cannot see as
    /// healthy.
    pub async fn check_device(
        &mut self,
        device_id: &str,
        metrics: Option<&HardwareMetrics>,
    ) -> HardwareResult<()> {
        let start_time = Instant::now();

        let (status, health_score, issues, recommendations) = match metrics {
            None => (
                HealthStatus::Unknown,
                0.0,
                Vec::new(),
                vec!["No telemetry available for this device".to_string()],
            ),
            Some(metrics) => Self::diagnose(metrics),
        };

        let result = HealthCheckResult {
            device_id: device_id.to_string(),
            status,
            timestamp: SystemTime::now(),
            response_time: start_time.elapsed().as_millis() as f64,
            health_score,
            issues,
            recommendations,
        };

        self.results.insert(device_id.to_string(), result);
        Ok(())
    }

    /// Threshold-based diagnosis of a real `HardwareMetrics` snapshot.
    /// Fixed, documented thresholds stand in for a device-specific policy
    /// until `HealthCheckPolicy` grows dedicated threshold fields; they are
    /// applied to the same real numbers `PerformanceMonitor` records, not
    /// to invented data.
    fn diagnose(metrics: &HardwareMetrics) -> (HealthStatus, f64, Vec<HealthIssue>, Vec<String>) {
        const TEMP_WARNING_C: f64 = 80.0;
        const TEMP_CRITICAL_C: f64 = 90.0;
        const ERROR_RATE_WARNING: f64 = 0.01;
        const ERROR_RATE_CRITICAL: f64 = 0.05;
        const UTILIZATION_SATURATED: f64 = 95.0;

        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        let mut score = 1.0f64;

        if let Some(temp) = metrics.temperature {
            if temp >= TEMP_CRITICAL_C {
                score -= 0.5;
                issues.push(HealthIssue {
                    issue_type: HealthIssueType::HighTemperature,
                    severity: AnomalySeverity::Critical,
                    description: format!(
                        "Temperature {temp:.1}\u{b0}C is at or above the critical threshold \
                         of {TEMP_CRITICAL_C:.1}\u{b0}C"
                    ),
                    affected_components: vec!["thermal".to_string()],
                    potential_causes: vec![
                        "Inadequate cooling".to_string(),
                        "Sustained heavy load".to_string(),
                    ],
                    suggested_fixes: vec![
                        "Reduce load immediately".to_string(),
                        "Improve cooling".to_string(),
                    ],
                });
                recommendations.push("Reduce load or improve cooling immediately".to_string());
            } else if temp >= TEMP_WARNING_C {
                score -= 0.2;
                issues.push(HealthIssue {
                    issue_type: HealthIssueType::HighTemperature,
                    severity: AnomalySeverity::Medium,
                    description: format!(
                        "Temperature {temp:.1}\u{b0}C is at or above the warning threshold \
                         of {TEMP_WARNING_C:.1}\u{b0}C"
                    ),
                    affected_components: vec!["thermal".to_string()],
                    potential_causes: vec!["Sustained heavy load".to_string()],
                    suggested_fixes: vec!["Monitor temperature".to_string()],
                });
                recommendations.push("Monitor temperature".to_string());
            }
        }

        if metrics.error_rate >= ERROR_RATE_CRITICAL {
            score -= 0.5;
            issues.push(HealthIssue {
                issue_type: HealthIssueType::HardwareFaults,
                severity: AnomalySeverity::Critical,
                description: format!(
                    "Error rate {:.4} is at or above the critical threshold of {ERROR_RATE_CRITICAL:.4}",
                    metrics.error_rate
                ),
                affected_components: vec!["compute".to_string()],
                potential_causes: vec![
                    "Hardware fault".to_string(),
                    "Driver instability".to_string(),
                ],
                suggested_fixes: vec![
                    "Reset the device".to_string(),
                    "Inspect driver logs".to_string(),
                ],
            });
            recommendations.push("Investigate elevated error rate".to_string());
        } else if metrics.error_rate >= ERROR_RATE_WARNING {
            score -= 0.15;
            issues.push(HealthIssue {
                issue_type: HealthIssueType::HardwareFaults,
                severity: AnomalySeverity::Medium,
                description: format!(
                    "Error rate {:.4} is at or above the warning threshold of {ERROR_RATE_WARNING:.4}",
                    metrics.error_rate
                ),
                affected_components: vec!["compute".to_string()],
                potential_causes: vec!["Transient faults".to_string()],
                suggested_fixes: vec!["Continue monitoring error rate".to_string()],
            });
            recommendations.push("Continue monitoring error rate".to_string());
        }

        if metrics.utilization >= UTILIZATION_SATURATED {
            recommendations
                .push("Utilization is near saturation; consider load balancing".to_string());
        }

        let score = score.clamp(0.0, 1.0);
        let status = if issues.iter().any(|i| i.severity == AnomalySeverity::Critical) {
            HealthStatus::Critical
        } else if !issues.is_empty() {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };

        if recommendations.is_empty() {
            recommendations.push("No action required".to_string());
        }

        (status, score, issues, recommendations)
    }

    /// Get health status for a device
    pub fn get_health_status(&self, device_id: &str) -> Option<HealthStatus> {
        self.results.get(device_id).map(|result| result.status)
    }

    /// Set health check policy for a device
    pub fn set_policy(&mut self, device_id: &str, policy: HealthCheckPolicy) {
        self.policies.insert(device_id.to_string(), policy);
    }

    /// Get health check results for all devices
    pub fn get_all_results(&self) -> &HashMap<String, HealthCheckResult> {
        &self.results
    }
}

impl Default for HealthCheckPolicy {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(60),
            timeout: Duration::from_secs(10),
            retry_count: 3,
            failure_threshold: 3,
            recovery_threshold: 2,
            detailed_diagnostics: true,
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_metrics() -> HardwareMetrics {
        HardwareMetrics {
            ops_per_second: 1_000_000.0,
            memory_bandwidth: 10e9,
            utilization: 20.0,
            power_consumption: 65.0,
            temperature: Some(45.0),
            error_rate: 0.0,
            latency: 1.0,
            throughput: 1000.0,
        }
    }

    /// Regression test: `check_device` used to always insert a hardcoded
    /// `health_score = 0.95` / `HealthStatus::Healthy` result regardless of
    /// device state -- a monitoring API that could never detect a fault.
    /// With normal metrics the device really is healthy, so this alone does
    /// not distinguish old from new behavior; the overheating/error-rate
    /// tests below do.
    #[tokio::test]
    async fn test_check_device_reports_healthy_for_normal_metrics() {
        let mut checker = HealthChecker::new();
        checker
            .check_device("dev0", Some(&healthy_metrics()))
            .await
            .expect("check failed");

        let result = checker.get_all_results().get("dev0").expect("missing result");
        assert_eq!(result.status, HealthStatus::Healthy);
        assert!(result.issues.is_empty());
    }

    /// Regression test: an overheating device must be flagged, not silently
    /// reported healthy with a fixed 0.95 score.
    #[tokio::test]
    async fn test_check_device_flags_critical_temperature() {
        let mut checker = HealthChecker::new();
        let mut metrics = healthy_metrics();
        metrics.temperature = Some(95.0); // above the 90C critical threshold

        checker.check_device("hot-dev", Some(&metrics)).await.expect("check failed");

        let result = checker.get_all_results().get("hot-dev").expect("missing result");
        assert_eq!(result.status, HealthStatus::Critical);
        assert!(!result.issues.is_empty());
        assert!(
            result.health_score < 0.95,
            "score must reflect the fault, not stay at 0.95"
        );
        assert_eq!(
            checker.get_health_status("hot-dev"),
            Some(HealthStatus::Critical)
        );
    }

    /// Regression test: an elevated error rate must be flagged as a
    /// warning, not silently reported healthy.
    #[tokio::test]
    async fn test_check_device_flags_elevated_error_rate() {
        let mut checker = HealthChecker::new();
        let mut metrics = healthy_metrics();
        metrics.error_rate = 0.02; // above the 0.01 warning threshold

        checker.check_device("flaky-dev", Some(&metrics)).await.expect("check failed");

        let result = checker.get_all_results().get("flaky-dev").expect("missing result");
        assert_eq!(result.status, HealthStatus::Warning);
        assert!(!result.issues.is_empty());
    }

    /// Regression test: a device with no telemetry must be reported
    /// `Unknown`, never fabricated as `Healthy`.
    #[tokio::test]
    async fn test_check_device_without_metrics_is_unknown() {
        let mut checker = HealthChecker::new();
        checker.check_device("ghost-dev", None).await.expect("check failed");

        assert_eq!(
            checker.get_health_status("ghost-dev"),
            Some(HealthStatus::Unknown)
        );
    }
}
