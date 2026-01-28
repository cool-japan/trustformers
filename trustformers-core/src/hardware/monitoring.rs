// Copyright (c) 2024 TrustformeRS Contributors
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

        performers.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Partial comparison failed"));
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

    /// Perform health check on a device
    pub async fn check_device(&mut self, device_id: &str) -> HardwareResult<()> {
        let start_time = Instant::now();

        // Simulate health check (in practice, this would perform actual diagnostics)
        let health_score = 0.95; // Placeholder
        let issues = vec![]; // Placeholder

        let result = HealthCheckResult {
            device_id: device_id.to_string(),
            status: HealthStatus::Healthy,
            timestamp: SystemTime::now(),
            response_time: start_time.elapsed().as_millis() as f64,
            health_score,
            issues,
            recommendations: vec!["Monitor temperature".to_string()],
        };

        self.results.insert(device_id.to_string(), result);
        Ok(())
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
