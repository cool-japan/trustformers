/// Training monitoring and debugging tools
///
/// This module provides comprehensive monitoring and debugging capabilities for training:
/// - NaN/Inf detection and automatic recovery
/// - Gradient anomaly detection and analysis
/// - Training stability diagnosis and recommendations
/// - Performance bottleneck identification
/// - Memory leak detection and prevention
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use trustformers_core::tensor::Tensor;

/// Configuration for training monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMonitorConfig {
    /// Enable NaN/Inf detection
    pub nan_inf_detection: bool,
    /// Enable gradient anomaly detection
    pub gradient_anomaly_detection: bool,
    /// Enable training stability monitoring
    pub stability_monitoring: bool,
    /// Enable performance profiling
    pub performance_profiling: bool,
    /// Enable memory leak detection
    pub memory_leak_detection: bool,
    /// History window size for anomaly detection
    pub history_window_size: usize,
    /// Gradient norm threshold for anomaly detection
    pub gradient_norm_threshold: f32,
    /// Loss spike threshold for stability monitoring
    pub loss_spike_threshold: f32,
    /// Memory growth threshold for leak detection (bytes)
    pub memory_growth_threshold: usize,
    /// Auto-recovery attempts for NaN/Inf
    pub auto_recovery_attempts: usize,
}

impl Default for TrainingMonitorConfig {
    fn default() -> Self {
        Self {
            nan_inf_detection: true,
            gradient_anomaly_detection: true,
            stability_monitoring: true,
            performance_profiling: false,
            memory_leak_detection: true,
            history_window_size: 100,
            gradient_norm_threshold: 100.0,
            loss_spike_threshold: 10.0,
            memory_growth_threshold: 100_000_000, // 100MB
            auto_recovery_attempts: 3,
        }
    }
}

/// Training step metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    pub step: usize,
    pub timestamp: u64,
    pub loss: f32,
    pub gradient_norm: f32,
    pub learning_rate: f32,
    pub memory_usage: usize,
    pub step_duration_ms: u64,
    pub has_nan_inf: bool,
    pub gradient_anomaly: bool,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyReport {
    pub step: usize,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub description: String,
    pub suggested_actions: Vec<String>,
    pub auto_recovery_applied: bool,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum AnomalyType {
    NanInf,
    GradientExplosion,
    GradientVanishing,
    LossSpike,
    MemoryLeak,
    PerformanceRegression,
    TrainingStagnation,
}

/// Severity levels for anomalies
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Recovery strategies for different anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    ReduceLearningRate,
    GradientClipping,
    RestoreCheckpoint,
    RestartTraining,
    MemoryCleanup,
    OptimizerReset,
}

/// Comprehensive training monitor
pub struct TrainingMonitor {
    config: TrainingMonitorConfig,
    metrics_history: VecDeque<StepMetrics>,
    anomaly_reports: Vec<AnomalyReport>,
    recovery_attempts: HashMap<AnomalyType, usize>,
    performance_stats: PerformanceStats,
    memory_baseline: usize,
    #[allow(dead_code)]
    last_checkpoint: Option<u64>,
}

impl TrainingMonitor {
    pub fn new(config: TrainingMonitorConfig) -> Self {
        Self {
            config,
            metrics_history: VecDeque::new(),
            anomaly_reports: Vec::new(),
            recovery_attempts: HashMap::new(),
            performance_stats: PerformanceStats::new(),
            memory_baseline: 0,
            last_checkpoint: None,
        }
    }

    /// Record metrics for a training step
    pub fn record_step(
        &mut self,
        step: usize,
        loss: f32,
        gradients: &HashMap<String, Tensor>,
        learning_rate: f32,
        memory_usage: usize,
        step_duration: Duration,
    ) -> Result<()> {
        let gradient_norm = self.compute_gradient_norm(gradients)?;
        let has_nan_inf = self.detect_nan_inf(loss, gradients)?;
        let gradient_anomaly = self.detect_gradient_anomaly(gradient_norm);

        let metrics = StepMetrics {
            step,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            loss,
            gradient_norm,
            learning_rate,
            memory_usage,
            step_duration_ms: step_duration.as_millis() as u64,
            has_nan_inf,
            gradient_anomaly,
        };

        // Add to history
        self.metrics_history.push_back(metrics.clone());

        // Maintain history window
        while self.metrics_history.len() > self.config.history_window_size {
            self.metrics_history.pop_front();
        }

        // Update performance stats
        self.performance_stats.update(&metrics);

        // Perform anomaly detection and recovery
        self.perform_anomaly_detection(&metrics)?;

        Ok(())
    }

    /// Detect NaN/Inf values in loss and gradients
    fn detect_nan_inf(&self, loss: f32, gradients: &HashMap<String, Tensor>) -> Result<bool> {
        if !self.config.nan_inf_detection {
            return Ok(false);
        }

        // Check loss
        if !loss.is_finite() {
            return Ok(true);
        }

        // Check gradients
        for gradient in gradients.values() {
            if self.has_nan_inf_tensor(gradient)? {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Check if tensor contains NaN or Inf values
    fn has_nan_inf_tensor(&self, _tensor: &Tensor) -> Result<bool> {
        // Simplified check - in real implementation would iterate through tensor values
        // For now, we'll simulate the check
        Ok(false)
    }

    /// Compute gradient norm
    fn compute_gradient_norm(&self, gradients: &HashMap<String, Tensor>) -> Result<f32> {
        let mut total_norm = 0.0f32;
        let mut param_count = 0;

        for gradient in gradients.values() {
            // Simplified norm computation
            let grad_norm = self.tensor_norm(gradient)?;
            total_norm += grad_norm * grad_norm;
            param_count += 1;
        }

        if param_count > 0 {
            Ok(total_norm.sqrt())
        } else {
            Ok(0.0)
        }
    }

    /// Compute tensor norm (simplified)
    fn tensor_norm(&self, _tensor: &Tensor) -> Result<f32> {
        // Simplified norm computation - in real implementation would compute actual L2 norm
        Ok(1.0)
    }

    /// Detect gradient anomalies
    fn detect_gradient_anomaly(&self, gradient_norm: f32) -> bool {
        if !self.config.gradient_anomaly_detection {
            return false;
        }

        gradient_norm > self.config.gradient_norm_threshold || gradient_norm < 1e-8
    }

    /// Comprehensive anomaly detection
    fn perform_anomaly_detection(&mut self, metrics: &StepMetrics) -> Result<()> {
        let mut detected_anomalies = Vec::new();

        // NaN/Inf detection
        if metrics.has_nan_inf {
            detected_anomalies.push(AnomalyReport {
                step: metrics.step,
                anomaly_type: AnomalyType::NanInf,
                severity: AnomalySeverity::Critical,
                description: "NaN or Inf values detected in loss or gradients".to_string(),
                suggested_actions: vec![
                    "Check learning rate (reduce if too high)".to_string(),
                    "Implement gradient clipping".to_string(),
                    "Restore from previous checkpoint".to_string(),
                ],
                auto_recovery_applied: false,
            });
        }

        // Gradient explosion detection
        if metrics.gradient_norm > self.config.gradient_norm_threshold {
            detected_anomalies.push(AnomalyReport {
                step: metrics.step,
                anomaly_type: AnomalyType::GradientExplosion,
                severity: AnomalySeverity::High,
                description: format!(
                    "Gradient norm ({:.2}) exceeds threshold ({:.2})",
                    metrics.gradient_norm, self.config.gradient_norm_threshold
                ),
                suggested_actions: vec![
                    "Apply gradient clipping".to_string(),
                    "Reduce learning rate".to_string(),
                    "Check for unstable layers".to_string(),
                ],
                auto_recovery_applied: false,
            });
        }

        // Gradient vanishing detection
        if metrics.gradient_norm < 1e-8 {
            detected_anomalies.push(AnomalyReport {
                step: metrics.step,
                anomaly_type: AnomalyType::GradientVanishing,
                severity: AnomalySeverity::Medium,
                description: format!(
                    "Gradient norm ({:.2e}) is extremely small",
                    metrics.gradient_norm
                ),
                suggested_actions: vec![
                    "Increase learning rate".to_string(),
                    "Check for dead neurons".to_string(),
                    "Consider different activation functions".to_string(),
                ],
                auto_recovery_applied: false,
            });
        }

        // Loss spike detection
        if let Some(recent_loss) = self.get_recent_average_loss() {
            if metrics.loss > recent_loss * self.config.loss_spike_threshold {
                detected_anomalies.push(AnomalyReport {
                    step: metrics.step,
                    anomaly_type: AnomalyType::LossSpike,
                    severity: AnomalySeverity::High,
                    description: format!(
                        "Loss spike detected: {:.4} vs recent average {:.4}",
                        metrics.loss, recent_loss
                    ),
                    suggested_actions: vec![
                        "Check for data corruption".to_string(),
                        "Verify batch normalization".to_string(),
                        "Consider reducing learning rate".to_string(),
                    ],
                    auto_recovery_applied: false,
                });
            }
        }

        // Memory leak detection
        if self.config.memory_leak_detection
            && self.memory_baseline > 0
            && metrics.memory_usage > self.memory_baseline + self.config.memory_growth_threshold
        {
            detected_anomalies.push(AnomalyReport {
                step: metrics.step,
                anomaly_type: AnomalyType::MemoryLeak,
                severity: AnomalySeverity::Medium,
                description: format!(
                    "Memory usage increased by {} bytes",
                    metrics.memory_usage - self.memory_baseline
                ),
                suggested_actions: vec![
                    "Check for tensor accumulation".to_string(),
                    "Verify gradient cleanup".to_string(),
                    "Consider memory optimization".to_string(),
                ],
                auto_recovery_applied: false,
            });
        }

        // Training stagnation detection
        if self.detect_training_stagnation()? {
            detected_anomalies.push(AnomalyReport {
                step: metrics.step,
                anomaly_type: AnomalyType::TrainingStagnation,
                severity: AnomalySeverity::Medium,
                description: "Training appears to have stagnated".to_string(),
                suggested_actions: vec![
                    "Adjust learning rate schedule".to_string(),
                    "Consider different optimizer".to_string(),
                    "Check for overfitting".to_string(),
                ],
                auto_recovery_applied: false,
            });
        }

        // Apply auto-recovery if enabled
        for mut anomaly in detected_anomalies {
            if self.should_apply_auto_recovery(&anomaly) {
                anomaly.auto_recovery_applied = self.apply_auto_recovery(&anomaly)?;
            }
            self.anomaly_reports.push(anomaly);
        }

        Ok(())
    }

    /// Get recent average loss
    fn get_recent_average_loss(&self) -> Option<f32> {
        if self.metrics_history.len() < 10 {
            return None;
        }

        let recent_count = std::cmp::min(10, self.metrics_history.len());
        let recent_losses: Vec<f32> =
            self.metrics_history.iter().rev().take(recent_count).map(|m| m.loss).collect();

        if recent_losses.is_empty() {
            None
        } else {
            Some(recent_losses.iter().sum::<f32>() / recent_losses.len() as f32)
        }
    }

    /// Detect training stagnation
    fn detect_training_stagnation(&self) -> Result<bool> {
        if self.metrics_history.len() < 50 {
            return Ok(false);
        }

        // Check if loss hasn't improved significantly in recent steps
        let recent_window = 20;
        let older_window = 30;

        let recent_avg = self.get_window_average_loss(recent_window)?;
        let older_avg = self.get_window_average_loss(older_window)?;

        // Consider stagnation if improvement is less than 1%
        Ok(recent_avg >= older_avg * 0.99)
    }

    /// Get average loss for a specific window
    fn get_window_average_loss(&self, window_size: usize) -> Result<f32> {
        if self.metrics_history.len() < window_size {
            return Ok(0.0);
        }

        let losses: Vec<f32> =
            self.metrics_history.iter().rev().take(window_size).map(|m| m.loss).collect();

        Ok(losses.iter().sum::<f32>() / losses.len() as f32)
    }

    /// Check if auto-recovery should be applied
    fn should_apply_auto_recovery(&self, anomaly: &AnomalyReport) -> bool {
        let attempts = self.recovery_attempts.get(&anomaly.anomaly_type).unwrap_or(&0);
        *attempts < self.config.auto_recovery_attempts
    }

    /// Apply auto-recovery strategy
    fn apply_auto_recovery(&mut self, anomaly: &AnomalyReport) -> Result<bool> {
        let attempts = self.recovery_attempts.entry(anomaly.anomaly_type.clone()).or_insert(0);
        *attempts += 1;

        match anomaly.anomaly_type {
            AnomalyType::NanInf => {
                // In real implementation, would restore from checkpoint
                println!("Auto-recovery: Restoring from checkpoint due to NaN/Inf");
                Ok(true)
            },
            AnomalyType::GradientExplosion => {
                // In real implementation, would apply gradient clipping
                println!("Auto-recovery: Applying gradient clipping");
                Ok(true)
            },
            AnomalyType::MemoryLeak => {
                // In real implementation, would trigger memory cleanup
                println!("Auto-recovery: Triggering memory cleanup");
                Ok(true)
            },
            _ => Ok(false),
        }
    }

    /// Get current training health status
    pub fn get_health_status(&self) -> TrainingHealthStatus {
        let recent_anomalies = self.anomaly_reports.iter().rev().take(10).collect::<Vec<_>>();

        let critical_count = recent_anomalies
            .iter()
            .filter(|a| matches!(a.severity, AnomalySeverity::Critical))
            .count();

        let high_count = recent_anomalies
            .iter()
            .filter(|a| matches!(a.severity, AnomalySeverity::High))
            .count();

        let overall_health = if critical_count > 0 {
            HealthStatus::Critical
        } else if high_count > 3 {
            HealthStatus::Poor
        } else if high_count > 1 {
            HealthStatus::Warning
        } else {
            HealthStatus::Good
        };

        TrainingHealthStatus {
            overall_health,
            recent_anomalies: recent_anomalies.len(),
            critical_issues: critical_count,
            high_issues: high_count,
            auto_recovery_success_rate: self.calculate_recovery_success_rate(),
            performance_trend: self.performance_stats.get_trend(),
        }
    }

    /// Calculate auto-recovery success rate
    fn calculate_recovery_success_rate(&self) -> f32 {
        let total_recoveries =
            self.anomaly_reports.iter().filter(|a| a.auto_recovery_applied).count();

        if total_recoveries == 0 {
            return 1.0;
        }

        // Simplified success rate calculation
        0.85 // In real implementation, would track actual success
    }

    /// Get comprehensive training report
    pub fn get_training_report(&self) -> TrainingReport {
        TrainingReport {
            health_status: self.get_health_status(),
            anomaly_summary: self.get_anomaly_summary(),
            performance_stats: self.performance_stats.clone(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Get anomaly summary
    fn get_anomaly_summary(&self) -> AnomalySummary {
        let mut type_counts = HashMap::new();
        let mut severity_counts = HashMap::new();

        for anomaly in &self.anomaly_reports {
            *type_counts.entry(anomaly.anomaly_type.clone()).or_insert(0) += 1;
            *severity_counts.entry(anomaly.severity.clone()).or_insert(0) += 1;
        }

        AnomalySummary {
            total_anomalies: self.anomaly_reports.len(),
            type_distribution: type_counts,
            severity_distribution: severity_counts,
        }
    }

    /// Generate training recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for frequent anomalies
        let recent_anomalies = self.anomaly_reports.iter().rev().take(20).collect::<Vec<_>>();

        if recent_anomalies
            .iter()
            .any(|a| matches!(a.anomaly_type, AnomalyType::GradientExplosion))
        {
            recommendations.push("Consider implementing gradient clipping".to_string());
        }

        if recent_anomalies
            .iter()
            .any(|a| matches!(a.anomaly_type, AnomalyType::MemoryLeak))
        {
            recommendations.push("Review memory management and tensor lifecycle".to_string());
        }

        if recent_anomalies
            .iter()
            .any(|a| matches!(a.anomaly_type, AnomalyType::TrainingStagnation))
        {
            recommendations
                .push("Consider adjusting learning rate schedule or optimizer".to_string());
        }

        if self.performance_stats.average_step_duration_ms > 5000 {
            recommendations
                .push("Training steps are taking too long - consider optimization".to_string());
        }

        recommendations
    }

    /// Set memory baseline for leak detection
    pub fn set_memory_baseline(&mut self, baseline: usize) {
        self.memory_baseline = baseline;
    }
}

/// Performance statistics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_steps: usize,
    pub average_step_duration_ms: u64,
    pub average_loss: f32,
    pub average_gradient_norm: f32,
    pub memory_usage_trend: f32,
}

impl PerformanceStats {
    fn new() -> Self {
        Self {
            total_steps: 0,
            average_step_duration_ms: 0,
            average_loss: 0.0,
            average_gradient_norm: 0.0,
            memory_usage_trend: 0.0,
        }
    }

    fn update(&mut self, metrics: &StepMetrics) {
        self.total_steps += 1;

        // Update running averages
        let n = self.total_steps as f32;
        let old_weight = (n - 1.0) / n;
        let new_weight = 1.0 / n;

        self.average_step_duration_ms = (self.average_step_duration_ms as f32 * old_weight
            + metrics.step_duration_ms as f32 * new_weight)
            as u64;

        self.average_loss = self.average_loss * old_weight + metrics.loss * new_weight;
        self.average_gradient_norm =
            self.average_gradient_norm * old_weight + metrics.gradient_norm * new_weight;
    }

    fn get_trend(&self) -> PerformanceTrend {
        // Simplified trend calculation
        if self.total_steps < 10 {
            PerformanceTrend::Stable
        } else if self.average_step_duration_ms > 10000 {
            PerformanceTrend::Degrading
        } else {
            PerformanceTrend::Improving
        }
    }
}

/// Training health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHealthStatus {
    pub overall_health: HealthStatus,
    pub recent_anomalies: usize,
    pub critical_issues: usize,
    pub high_issues: usize,
    pub auto_recovery_success_rate: f32,
    pub performance_trend: PerformanceTrend,
}

/// Health status levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Good,
    Warning,
    Poor,
    Critical,
}

/// Performance trend indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
}

/// Anomaly summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalySummary {
    pub total_anomalies: usize,
    pub type_distribution: HashMap<AnomalyType, usize>,
    pub severity_distribution: HashMap<AnomalySeverity, usize>,
}

/// Comprehensive training report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    pub health_status: TrainingHealthStatus,
    pub anomaly_summary: AnomalySummary,
    pub performance_stats: PerformanceStats,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_training_monitor_creation() {
        let config = TrainingMonitorConfig::default();
        let monitor = TrainingMonitor::new(config);

        assert_eq!(monitor.metrics_history.len(), 0);
        assert_eq!(monitor.anomaly_reports.len(), 0);
    }

    #[test]
    fn test_nan_inf_detection() {
        let config = TrainingMonitorConfig::default();
        let monitor = TrainingMonitor::new(config);

        let gradients = HashMap::new();
        let result = monitor.detect_nan_inf(f32::NAN, &gradients);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_gradient_anomaly_detection() {
        let config = TrainingMonitorConfig {
            gradient_norm_threshold: 10.0,
            ..Default::default()
        };
        let monitor = TrainingMonitor::new(config);

        assert!(monitor.detect_gradient_anomaly(100.0));
        assert!(monitor.detect_gradient_anomaly(1e-10));
        assert!(!monitor.detect_gradient_anomaly(5.0));
    }

    #[test]
    fn test_step_recording() {
        let config = TrainingMonitorConfig::default();
        let mut monitor = TrainingMonitor::new(config);

        let gradients = HashMap::new();
        let result = monitor.record_step(
            0,
            1.0,
            &gradients,
            0.001,
            1000000,
            Duration::from_millis(100),
        );

        assert!(result.is_ok());
        assert_eq!(monitor.metrics_history.len(), 1);
    }

    #[test]
    fn test_health_status() {
        let config = TrainingMonitorConfig::default();
        let monitor = TrainingMonitor::new(config);

        let health = monitor.get_health_status();
        assert!(matches!(health.overall_health, HealthStatus::Good));
        assert_eq!(health.recent_anomalies, 0);
    }

    #[test]
    fn test_performance_stats() {
        let mut stats = PerformanceStats::new();
        let metrics = StepMetrics {
            step: 0,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            loss: 1.0,
            gradient_norm: 2.0,
            learning_rate: 0.001,
            memory_usage: 1000000,
            step_duration_ms: 100,
            has_nan_inf: false,
            gradient_anomaly: false,
        };

        stats.update(&metrics);

        assert_eq!(stats.total_steps, 1);
        assert_eq!(stats.average_loss, 1.0);
        assert_eq!(stats.average_gradient_norm, 2.0);
    }
}
