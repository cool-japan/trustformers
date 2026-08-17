//! Anomaly Detection for Model Debugging
//!
//! Detects unusual patterns in model execution, tensor values, and gradients
//! to help identify potential issues during training and inference.
// reason: debug/profiling scaffolding — structs are constructed and their fields/methods
// are retained for the data model, serialization completeness, and future consumers that
// do not yet read every member. Consolidated from many item-level #[allow(dead_code)].
#![allow(dead_code)]

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

use crate::DebugConfig;

/// Anomaly detector for model execution
#[derive(Debug)]
pub struct AnomalyDetector {
    config: AnomalyDetectorConfig,
    detected_anomalies: Vec<Anomaly>,
    start_time: DateTime<Utc>,
    recovery_attempts: Vec<RecoveryAttempt>,
    monitoring_stats: MonitoringStats,
    performance_history: VecDeque<f64>,
    gradient_history: HashMap<String, VecDeque<f64>>,
    loss_history: VecDeque<f64>,
    weight_baseline: HashMap<String, Vec<f32>>,
    /// Live handle into the training loop that automatic recovery actions
    /// are dispatched through. `None` (the default) means recovery actions
    /// are computed and recorded but never actually performed -- see
    /// [`Self::execute_recovery_action`].
    training_control: Option<Box<dyn TrainingControl>>,
}

/// A live handle into the training loop that [`AnomalyDetector`] can drive
/// automatic recovery actions through.
///
/// Implementations perform the real mutation (clip gradients, reduce the
/// learning rate, restart the optimizer, ...) against whatever optimizer or
/// model state the caller owns. Without a `TrainingControl` attached via
/// [`AnomalyDetector::set_training_control`], [`AnomalyDetector`] never
/// claims a recovery action succeeded: every action honestly reports "not
/// performed" instead of the old unconditional `Ok(true)`.
///
/// Each method returns `Err` (propagated into the recorded
/// [`RecoveryAttempt::error_message`], never swallowed) if the underlying
/// mutation could not be carried out.
pub trait TrainingControl: std::fmt::Debug + Send + Sync {
    /// Zero out (or otherwise reset) the current gradients.
    fn reset_gradients(&mut self) -> Result<()>;
    /// Multiply the optimizer's learning rate by `factor`.
    fn reduce_learning_rate(&mut self, factor: f64) -> Result<()>;
    /// Clip gradients to `max_norm` (e.g. global-norm clipping).
    fn clip_gradients(&mut self, max_norm: f64) -> Result<()>;
    /// Reset the optimizer's internal state (momentum buffers, etc.).
    fn restart_optimizer(&mut self) -> Result<()>;
    /// Skip the batch currently being processed.
    fn skip_batch(&mut self) -> Result<()>;
    /// Reset a specific layer's weights to their initialization.
    fn reset_weights(&mut self, layer_name: &str) -> Result<()>;
    /// Apply weight decay at `rate` to the current weights.
    fn apply_weight_decay(&mut self, rate: f64) -> Result<()>;
    /// Immediately halt training.
    fn emergency_stop(&mut self) -> Result<()>;
}
// `TrainingControl: Debug` being a supertrait means `dyn TrainingControl`
// (and so `Box<dyn TrainingControl>` / the `Option` around it on
// `AnomalyDetector`) automatically implements `Debug` -- no manual impl
// needed here.

/// Configuration for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectorConfig {
    pub enable_nan_detection: bool,
    pub enable_inf_detection: bool,
    pub enable_gradient_explosion: bool,
    pub enable_gradient_vanishing: bool,
    pub gradient_threshold: f64,
    pub enable_memory_leak_detection: bool,
    pub enable_numerical_instability_detection: bool,
    pub enable_gradient_conflict_detection: bool,
    pub enable_performance_monitoring: bool,
    pub enable_weight_divergence_detection: bool,
    pub enable_activation_dead_detection: bool,
    pub enable_loss_anomaly_detection: bool,
    pub enable_auto_recovery: bool,
    pub numerical_instability_threshold: f64,
    pub performance_degradation_threshold: f64,
    pub weight_divergence_threshold: f64,
    pub loss_spike_threshold: f64,
    pub monitoring_window_size: usize,
    pub recovery_attempts_limit: usize,
}

impl Default for AnomalyDetectorConfig {
    fn default() -> Self {
        Self {
            enable_nan_detection: true,
            enable_inf_detection: true,
            enable_gradient_explosion: true,
            enable_gradient_vanishing: true,
            gradient_threshold: 1e6,
            enable_memory_leak_detection: true,
            enable_numerical_instability_detection: true,
            enable_gradient_conflict_detection: true,
            enable_performance_monitoring: true,
            enable_weight_divergence_detection: true,
            enable_activation_dead_detection: true,
            enable_loss_anomaly_detection: true,
            enable_auto_recovery: false, // Conservative default
            numerical_instability_threshold: 1e-12,
            performance_degradation_threshold: 0.5, // 50% degradation
            weight_divergence_threshold: 5.0,
            loss_spike_threshold: 10.0, // 10x loss increase
            monitoring_window_size: 100,
            recovery_attempts_limit: 3,
        }
    }
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    NaN,
    Infinity,
    GradientExplosion,
    GradientVanishing,
    MemoryLeak,
    UnusualActivation,
    NumericalInstability,
    GradientConflict,
    PerformanceDegradation,
    WeightDivergence,
    ActivationDead,
    LossAnomalous,
}

/// An detected anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub anomaly_type: AnomalyType,
    pub timestamp: DateTime<Utc>,
    pub location: String,
    pub description: String,
    pub severity: AnomalySeverity,
    pub metadata: HashMap<String, String>,
}

/// Severity level of an anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Auto-recovery action that can be taken
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    None,
    ResetGradients,
    ReduceLearningRate { factor: f64 },
    ClipGradients { max_norm: f64 },
    RestartOptimizer,
    SkipBatch,
    ResetWeights { layer_name: String },
    ApplyWeightDecay { rate: f64 },
    EmergencyStop,
}

/// Recovery attempt record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAttempt {
    pub anomaly_id: String,
    pub action: RecoveryAction,
    pub timestamp: DateTime<Utc>,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Real-time monitoring statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringStats {
    pub total_anomalies: usize,
    pub anomalies_per_type: HashMap<String, usize>,
    pub recovery_attempts: usize,
    pub successful_recoveries: usize,
    pub average_detection_time_ms: f64,
    pub monitoring_window: Vec<AnomalySnapshot>,
}

/// Snapshot of anomaly state for monitoring window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalySnapshot {
    pub timestamp: DateTime<Utc>,
    pub anomaly_count: usize,
    pub severity_distribution: HashMap<String, usize>,
    pub performance_metrics: HashMap<String, f64>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new(_config: &DebugConfig) -> Self {
        let monitoring_window_size = AnomalyDetectorConfig::default().monitoring_window_size;
        Self {
            config: AnomalyDetectorConfig::default(),
            detected_anomalies: Vec::new(),
            start_time: Utc::now(),
            recovery_attempts: Vec::new(),
            monitoring_stats: MonitoringStats {
                total_anomalies: 0,
                anomalies_per_type: HashMap::new(),
                recovery_attempts: 0,
                successful_recoveries: 0,
                average_detection_time_ms: 0.0,
                monitoring_window: Vec::new(),
            },
            performance_history: VecDeque::with_capacity(monitoring_window_size),
            gradient_history: HashMap::new(),
            loss_history: VecDeque::with_capacity(monitoring_window_size),
            weight_baseline: HashMap::new(),
            training_control: None,
        }
    }

    /// Attach a live [`TrainingControl`] handle so that automatic recovery
    /// actions (see [`Self::attempt_recovery`]) are actually performed
    /// against real training state, instead of only being computed and
    /// recorded.
    pub fn set_training_control(&mut self, control: Box<dyn TrainingControl>) {
        self.training_control = Some(control);
    }

    /// Detach the [`TrainingControl`] handle, if any. Future recovery
    /// actions will honestly report "not performed" until a new one is
    /// attached.
    pub fn clear_training_control(&mut self) {
        self.training_control = None;
    }

    /// Whether a [`TrainingControl`] handle is currently attached.
    pub fn has_training_control(&self) -> bool {
        self.training_control.is_some()
    }

    /// Start the anomaly detector
    pub async fn start(&mut self) -> Result<()> {
        self.start_time = Utc::now();
        self.detected_anomalies.clear();
        Ok(())
    }

    /// Check for NaN values in tensors
    pub fn check_nan(&mut self, values: &[f32], location: &str) -> Result<()> {
        if !self.config.enable_nan_detection {
            return Ok(());
        }

        if values.iter().any(|v| v.is_nan()) {
            self.report_anomaly(Anomaly {
                anomaly_type: AnomalyType::NaN,
                timestamp: Utc::now(),
                location: location.to_string(),
                description: "NaN values detected in tensor".to_string(),
                severity: AnomalySeverity::High,
                metadata: HashMap::new(),
            });
        }

        Ok(())
    }

    /// Check for infinite values in tensors
    pub fn check_inf(&mut self, values: &[f32], location: &str) -> Result<()> {
        if !self.config.enable_inf_detection {
            return Ok(());
        }

        if values.iter().any(|v| v.is_infinite()) {
            self.report_anomaly(Anomaly {
                anomaly_type: AnomalyType::Infinity,
                timestamp: Utc::now(),
                location: location.to_string(),
                description: "Infinite values detected in tensor".to_string(),
                severity: AnomalySeverity::High,
                metadata: HashMap::new(),
            });
        }

        Ok(())
    }

    /// Check for gradient explosion
    pub fn check_gradient_explosion(&mut self, gradient_norm: f64, location: &str) -> Result<()> {
        if !self.config.enable_gradient_explosion {
            return Ok(());
        }

        if gradient_norm > self.config.gradient_threshold {
            self.report_anomaly(Anomaly {
                anomaly_type: AnomalyType::GradientExplosion,
                timestamp: Utc::now(),
                location: location.to_string(),
                description: format!("Gradient explosion detected: norm = {}", gradient_norm),
                severity: AnomalySeverity::Critical,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("gradient_norm".to_string(), gradient_norm.to_string());
                    meta
                },
            });
        }

        Ok(())
    }

    /// Check for vanishing gradients
    pub fn check_gradient_vanishing(&mut self, gradient_norm: f64, location: &str) -> Result<()> {
        if !self.config.enable_gradient_vanishing {
            return Ok(());
        }

        let vanishing_threshold = 1e-8;
        if gradient_norm < vanishing_threshold {
            self.report_anomaly(Anomaly {
                anomaly_type: AnomalyType::GradientVanishing,
                timestamp: Utc::now(),
                location: location.to_string(),
                description: format!("Vanishing gradient detected: norm = {}", gradient_norm),
                severity: AnomalySeverity::High,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("gradient_norm".to_string(), gradient_norm.to_string());
                    meta.insert("threshold".to_string(), vanishing_threshold.to_string());
                    meta
                },
            });
        }

        Ok(())
    }

    /// Check for numerical instability
    pub fn check_numerical_instability(&mut self, values: &[f32], location: &str) -> Result<()> {
        let mut metadata = HashMap::new();

        // Check for values close to zero that might cause division problems
        let near_zero_count = values.iter().filter(|&&v| v.abs() < 1e-10 && v != 0.0).count();
        if near_zero_count > values.len() / 10 {
            metadata.insert("near_zero_count".to_string(), near_zero_count.to_string());
            metadata.insert("total_values".to_string(), values.len().to_string());

            self.report_anomaly(Anomaly {
                anomaly_type: AnomalyType::UnusualActivation,
                timestamp: Utc::now(),
                location: location.to_string(),
                description: format!(
                    "Numerical instability: {} values near zero",
                    near_zero_count
                ),
                severity: AnomalySeverity::Medium,
                metadata: metadata.clone(),
            });
        }

        // Check for extreme values that might cause overflow
        let extreme_count = values.iter().filter(|&&v| v.abs() > 1e6).count();
        if extreme_count > 0 {
            metadata.insert("extreme_count".to_string(), extreme_count.to_string());

            self.report_anomaly(Anomaly {
                anomaly_type: AnomalyType::UnusualActivation,
                timestamp: Utc::now(),
                location: location.to_string(),
                description: format!("Numerical instability: {} extreme values", extreme_count),
                severity: AnomalySeverity::High,
                metadata,
            });
        }

        Ok(())
    }

    /// Check for activation saturation
    pub fn check_activation_saturation(
        &mut self,
        activations: &[f32],
        activation_type: &str,
        location: &str,
    ) -> Result<()> {
        let saturation_threshold = match activation_type.to_lowercase().as_str() {
            "sigmoid" | "tanh" => 0.01, // Close to 0 or 1 for sigmoid, -1 or 1 for tanh
            "relu" => 0.0,              // Zero values for ReLU
            _ => 0.01,
        };

        let saturated_count = match activation_type.to_lowercase().as_str() {
            "sigmoid" => activations
                .iter()
                .filter(|&&v| v < saturation_threshold || v > 1.0 - saturation_threshold)
                .count(),
            "tanh" => activations.iter().filter(|&&v| v.abs() > 1.0 - saturation_threshold).count(),
            "relu" => activations.iter().filter(|&&v| v == 0.0).count(),
            _ => activations.iter().filter(|&&v| v.abs() < saturation_threshold).count(),
        };

        let saturation_ratio = saturated_count as f32 / activations.len() as f32;

        if saturation_ratio > 0.9 {
            let mut metadata = HashMap::new();
            metadata.insert("activation_type".to_string(), activation_type.to_string());
            metadata.insert("saturated_count".to_string(), saturated_count.to_string());
            metadata.insert("total_count".to_string(), activations.len().to_string());
            metadata.insert("saturation_ratio".to_string(), saturation_ratio.to_string());

            self.report_anomaly(Anomaly {
                anomaly_type: AnomalyType::UnusualActivation,
                timestamp: Utc::now(),
                location: location.to_string(),
                description: format!(
                    "Activation saturation detected: {:.1}% of {} activations saturated",
                    saturation_ratio * 100.0,
                    activation_type
                ),
                severity: AnomalySeverity::High,
                metadata,
            });
        }

        Ok(())
    }

    /// Check for memory leaks by tracking memory usage patterns
    pub fn check_memory_leak(
        &mut self,
        current_memory_mb: usize,
        expected_memory_mb: Option<usize>,
        location: &str,
    ) -> Result<()> {
        if !self.config.enable_memory_leak_detection {
            return Ok(());
        }

        let mut should_report = false;
        let mut description = String::new();
        let mut metadata = HashMap::new();

        metadata.insert(
            "current_memory_mb".to_string(),
            current_memory_mb.to_string(),
        );

        if let Some(expected) = expected_memory_mb {
            metadata.insert("expected_memory_mb".to_string(), expected.to_string());

            let growth_ratio = current_memory_mb as f64 / expected as f64;
            if growth_ratio > 2.0 {
                should_report = true;
                description = format!(
                    "Memory usage {}MB is {:.1}x expected {}MB",
                    current_memory_mb, growth_ratio, expected
                );
                metadata.insert("growth_ratio".to_string(), growth_ratio.to_string());
            }
        } else {
            // Check for absolute high memory usage
            if current_memory_mb > 8192 {
                // 8GB threshold
                should_report = true;
                description = format!("High memory usage detected: {}MB", current_memory_mb);
            }
        }

        if should_report {
            self.report_anomaly(Anomaly {
                anomaly_type: AnomalyType::MemoryLeak,
                timestamp: Utc::now(),
                location: location.to_string(),
                description,
                severity: if current_memory_mb > 16384 {
                    AnomalySeverity::Critical
                } else {
                    AnomalySeverity::High
                },
                metadata,
            });
        }

        Ok(())
    }

    /// Check for weight explosion in model parameters
    pub fn check_weight_explosion(&mut self, weights: &[f32], layer_name: &str) -> Result<()> {
        let weight_threshold = 10.0;
        let extreme_weights: Vec<f32> =
            weights.iter().filter(|&&w| w.abs() > weight_threshold).cloned().collect();

        if !extreme_weights.is_empty() {
            let mut metadata = HashMap::new();
            metadata.insert("layer_name".to_string(), layer_name.to_string());
            metadata.insert(
                "extreme_weight_count".to_string(),
                extreme_weights.len().to_string(),
            );
            metadata.insert("total_weight_count".to_string(), weights.len().to_string());
            metadata.insert(
                "max_weight".to_string(),
                extreme_weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max).to_string(),
            );

            self.report_anomaly(Anomaly {
                anomaly_type: AnomalyType::UnusualActivation,
                timestamp: Utc::now(),
                location: layer_name.to_string(),
                description: format!(
                    "Weight explosion in {}: {} weights > {}",
                    layer_name,
                    extreme_weights.len(),
                    weight_threshold
                ),
                severity: AnomalySeverity::High,
                metadata,
            });
        }

        Ok(())
    }

    /// Report an anomaly
    fn report_anomaly(&mut self, anomaly: Anomaly) {
        eprintln!(
            "🚨 Anomaly detected: {} at {}",
            anomaly.description, anomaly.location
        );

        // Update monitoring stats
        self.monitoring_stats.total_anomalies += 1;
        let anomaly_type_key = format!("{:?}", anomaly.anomaly_type);
        *self.monitoring_stats.anomalies_per_type.entry(anomaly_type_key).or_insert(0) += 1;

        self.detected_anomalies.push(anomaly);
    }

    /// Get all detected anomalies
    pub fn get_anomalies(&self) -> &[Anomaly] {
        &self.detected_anomalies
    }

    /// Clear all detected anomalies
    pub fn clear_anomalies(&mut self) {
        self.detected_anomalies.clear();
    }

    /// Check for gradient conflicts between layers
    pub fn check_gradient_conflict(
        &mut self,
        layer_gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<()> {
        if !self.config.enable_gradient_conflict_detection {
            return Ok(());
        }

        let layer_names: Vec<_> = layer_gradients.keys().cloned().collect();

        for i in 0..layer_names.len() {
            for j in i + 1..layer_names.len() {
                let layer1 = &layer_names[i];
                let layer2 = &layer_names[j];

                if let (Some(grad1), Some(grad2)) =
                    (layer_gradients.get(layer1), layer_gradients.get(layer2))
                {
                    let conflict_score = self.compute_gradient_conflict(grad1, grad2);

                    if conflict_score > 0.8 {
                        let mut metadata = HashMap::new();
                        metadata.insert("layer1".to_string(), layer1.clone());
                        metadata.insert("layer2".to_string(), layer2.clone());
                        metadata.insert("conflict_score".to_string(), conflict_score.to_string());

                        self.report_anomaly(Anomaly {
                            anomaly_type: AnomalyType::GradientConflict,
                            timestamp: Utc::now(),
                            location: format!("{}↔{}", layer1, layer2),
                            description: format!(
                                "Gradient conflict detected between {} and {} (score: {:.2})",
                                layer1, layer2, conflict_score
                            ),
                            severity: AnomalySeverity::High,
                            metadata,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Check for weight divergence from baseline
    pub fn check_weight_divergence(
        &mut self,
        layer_name: &str,
        current_weights: &[f32],
    ) -> Result<()> {
        if !self.config.enable_weight_divergence_detection {
            return Ok(());
        }

        // Initialize baseline if not exists
        if !self.weight_baseline.contains_key(layer_name) {
            self.weight_baseline.insert(layer_name.to_string(), current_weights.to_vec());
            return Ok(());
        }

        let Some(baseline) = self.weight_baseline.get(layer_name) else {
            return Ok(());
        };
        if baseline.len() != current_weights.len() {
            return Ok(()); // Skip if dimensions don't match
        }

        let divergence = self.compute_weight_divergence(baseline, current_weights);

        if divergence > self.config.weight_divergence_threshold {
            let mut metadata = HashMap::new();
            metadata.insert("layer_name".to_string(), layer_name.to_string());
            metadata.insert("divergence_score".to_string(), divergence.to_string());
            metadata.insert(
                "threshold".to_string(),
                self.config.weight_divergence_threshold.to_string(),
            );

            self.report_anomaly(Anomaly {
                anomaly_type: AnomalyType::WeightDivergence,
                timestamp: Utc::now(),
                location: layer_name.to_string(),
                description: format!(
                    "Weight divergence in {}: {:.2} (threshold: {:.2})",
                    layer_name, divergence, self.config.weight_divergence_threshold
                ),
                severity: if divergence > self.config.weight_divergence_threshold * 2.0 {
                    AnomalySeverity::Critical
                } else {
                    AnomalySeverity::High
                },
                metadata,
            });
        }

        Ok(())
    }

    /// Check for performance degradation
    pub fn check_performance_degradation(
        &mut self,
        current_performance: f64,
        location: &str,
    ) -> Result<()> {
        if !self.config.enable_performance_monitoring {
            return Ok(());
        }

        // Add to history
        if self.performance_history.len() >= self.config.monitoring_window_size {
            self.performance_history.pop_front();
        }
        self.performance_history.push_back(current_performance);

        // Check for degradation if we have enough history
        if self.performance_history.len() >= 10 {
            let recent_avg = self.performance_history.iter().rev().take(5).sum::<f64>() / 5.0;
            let baseline_avg = self.performance_history.iter().take(5).sum::<f64>() / 5.0;

            let degradation_ratio = (baseline_avg - recent_avg) / baseline_avg;

            if degradation_ratio > self.config.performance_degradation_threshold {
                let mut metadata = HashMap::new();
                metadata.insert("baseline_performance".to_string(), baseline_avg.to_string());
                metadata.insert("current_performance".to_string(), recent_avg.to_string());
                metadata.insert(
                    "degradation_ratio".to_string(),
                    degradation_ratio.to_string(),
                );

                self.report_anomaly(Anomaly {
                    anomaly_type: AnomalyType::PerformanceDegradation,
                    timestamp: Utc::now(),
                    location: location.to_string(),
                    description: format!(
                        "Performance degradation detected: {:.1}% drop from baseline",
                        degradation_ratio * 100.0
                    ),
                    severity: if degradation_ratio > 0.8 {
                        AnomalySeverity::Critical
                    } else {
                        AnomalySeverity::High
                    },
                    metadata,
                });
            }
        }

        Ok(())
    }

    /// Check for loss anomalies
    pub fn check_loss_anomaly(&mut self, current_loss: f64, location: &str) -> Result<()> {
        if !self.config.enable_loss_anomaly_detection {
            return Ok(());
        }

        // Add to history
        if self.loss_history.len() >= self.config.monitoring_window_size {
            self.loss_history.pop_front();
        }
        self.loss_history.push_back(current_loss);

        // Check for loss spikes
        if self.loss_history.len() >= 3 {
            let prev_loss = self.loss_history[self.loss_history.len() - 2];
            let loss_ratio = current_loss / prev_loss;

            if loss_ratio > self.config.loss_spike_threshold {
                let mut metadata = HashMap::new();
                metadata.insert("previous_loss".to_string(), prev_loss.to_string());
                metadata.insert("current_loss".to_string(), current_loss.to_string());
                metadata.insert("spike_ratio".to_string(), loss_ratio.to_string());

                self.report_anomaly(Anomaly {
                    anomaly_type: AnomalyType::LossAnomalous,
                    timestamp: Utc::now(),
                    location: location.to_string(),
                    description: format!(
                        "Loss spike detected: {:.2}x increase (from {:.6} to {:.6})",
                        loss_ratio, prev_loss, current_loss
                    ),
                    severity: if loss_ratio > 100.0 {
                        AnomalySeverity::Critical
                    } else {
                        AnomalySeverity::High
                    },
                    metadata,
                });
            }
        }

        Ok(())
    }

    /// Attempt automatic recovery from an anomaly
    pub async fn attempt_recovery(&mut self, anomaly: &Anomaly) -> Result<RecoveryAction> {
        if !self.config.enable_auto_recovery {
            return Ok(RecoveryAction::None);
        }

        let action = self.determine_recovery_action(anomaly);
        let anomaly_id = format!(
            "{:?}_{}",
            anomaly.anomaly_type,
            anomaly.timestamp.timestamp()
        );

        let (success, error_message) = self.execute_recovery_action(&action).await?;

        self.recovery_attempts.push(RecoveryAttempt {
            anomaly_id: anomaly_id.clone(),
            action: action.clone(),
            timestamp: Utc::now(),
            success,
            error_message,
        });

        self.monitoring_stats.recovery_attempts += 1;
        if success {
            self.monitoring_stats.successful_recoveries += 1;
        }

        Ok(action)
    }

    /// Get monitoring statistics
    pub fn get_monitoring_stats(&self) -> &MonitoringStats {
        &self.monitoring_stats
    }

    /// Get recovery attempts history
    pub fn get_recovery_attempts(&self) -> &[RecoveryAttempt] {
        &self.recovery_attempts
    }

    /// Update monitoring window with current state
    pub fn update_monitoring_window(&mut self) -> Result<()> {
        let mut severity_distribution = HashMap::new();
        for anomaly in &self.detected_anomalies {
            let key = format!("{:?}", anomaly.severity);
            *severity_distribution.entry(key).or_insert(0) += 1;
        }

        let mut performance_metrics = HashMap::new();
        if let Some(latest_perf) = self.performance_history.back() {
            performance_metrics.insert("latest_performance".to_string(), *latest_perf);
        }
        if let Some(latest_loss) = self.loss_history.back() {
            performance_metrics.insert("latest_loss".to_string(), *latest_loss);
        }

        let snapshot = AnomalySnapshot {
            timestamp: Utc::now(),
            anomaly_count: self.detected_anomalies.len(),
            severity_distribution,
            performance_metrics,
        };

        self.monitoring_stats.monitoring_window.push(snapshot);

        // Keep only recent snapshots
        if self.monitoring_stats.monitoring_window.len() > self.config.monitoring_window_size {
            self.monitoring_stats.monitoring_window.remove(0);
        }

        Ok(())
    }

    // Private helper methods for new functionality

    fn compute_gradient_conflict(&self, grad1: &[f32], grad2: &[f32]) -> f64 {
        if grad1.len() != grad2.len() {
            return 0.0;
        }

        let dot_product: f64 =
            grad1.iter().zip(grad2.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();

        let norm1: f64 = grad1.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let norm2: f64 = grad2.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        // Cosine similarity - conflicts are indicated by negative correlation
        let cosine_sim = dot_product / (norm1 * norm2);

        // Convert to conflict score (0 = no conflict, 1 = maximum conflict)
        (1.0 - cosine_sim) / 2.0
    }

    fn compute_weight_divergence(&self, baseline: &[f32], current: &[f32]) -> f64 {
        let mse: f64 = baseline
            .iter()
            .zip(current.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / baseline.len() as f64;

        mse.sqrt()
    }

    fn determine_recovery_action(&self, anomaly: &Anomaly) -> RecoveryAction {
        match anomaly.anomaly_type {
            AnomalyType::GradientExplosion => RecoveryAction::ClipGradients { max_norm: 1.0 },
            AnomalyType::GradientVanishing => RecoveryAction::ReduceLearningRate { factor: 0.5 },
            AnomalyType::NaN | AnomalyType::Infinity => RecoveryAction::ResetGradients,
            AnomalyType::WeightDivergence => RecoveryAction::ApplyWeightDecay { rate: 0.01 },
            AnomalyType::LossAnomalous => RecoveryAction::SkipBatch,
            AnomalyType::MemoryLeak => RecoveryAction::RestartOptimizer,
            AnomalyType::PerformanceDegradation => {
                RecoveryAction::ReduceLearningRate { factor: 0.8 }
            },
            _ => RecoveryAction::None,
        }
    }

    /// Dispatch `action` through the attached [`TrainingControl`], if any.
    ///
    /// Returns `(true, None)` when the handle performed the action for
    /// real; `(false, Some(reason))` when it was attempted and the handle
    /// reported an error, or when no handle is attached at all. This used
    /// to unconditionally return `Ok(true)` for every variant except
    /// `EmergencyStop` regardless of whether anything was ever done --
    /// every branch now goes through a real [`TrainingControl`] call.
    async fn execute_recovery_action(
        &mut self,
        action: &RecoveryAction,
    ) -> Result<(bool, Option<String>)> {
        let Some(control) = self.training_control.as_mut() else {
            let reason = format!(
                "no TrainingControl attached to this AnomalyDetector (see \
                 AnomalyDetector::set_training_control); recovery action {action:?} was not \
                 performed"
            );
            tracing::warn!("{reason}");
            return Ok((false, Some(reason)));
        };

        let outcome = match action {
            RecoveryAction::None => Ok(()),
            RecoveryAction::ResetGradients => control.reset_gradients(),
            RecoveryAction::ReduceLearningRate { factor } => control.reduce_learning_rate(*factor),
            RecoveryAction::ClipGradients { max_norm } => control.clip_gradients(*max_norm),
            RecoveryAction::RestartOptimizer => control.restart_optimizer(),
            RecoveryAction::SkipBatch => control.skip_batch(),
            RecoveryAction::ResetWeights { layer_name } => control.reset_weights(layer_name),
            RecoveryAction::ApplyWeightDecay { rate } => control.apply_weight_decay(*rate),
            RecoveryAction::EmergencyStop => control.emergency_stop(),
        };

        match (action, outcome) {
            (RecoveryAction::EmergencyStop, Ok(())) => {
                // Matches the pre-existing contract: EmergencyStop always
                // reports `false` here (it halts training rather than
                // "recovering" it), but it is now a real call, not a no-op.
                tracing::warn!("Executed recovery: Emergency stop");
                Ok((false, None))
            },
            (_, Ok(())) => {
                tracing::info!("Executed recovery action: {action:?}");
                Ok((true, None))
            },
            (_, Err(e)) => {
                let reason = e.to_string();
                tracing::error!("Recovery action {action:?} failed: {reason}");
                Ok((false, Some(reason)))
            },
        }
    }

    /// Quick anomaly check for simplified interface
    pub async fn quick_check(&self) -> Result<crate::QuickAnomalySummary> {
        let anomaly_count = self.detected_anomalies.len();

        let severity_level = match anomaly_count {
            0 => "None",
            1..=3 => "Low",
            4..=10 => "Medium",
            11..=20 => "High",
            _ => "Critical",
        }
        .to_string();

        let mut recommendations = Vec::new();
        if anomaly_count > 0 {
            recommendations.push("Review recent training metrics for instabilities".to_string());
        }
        if anomaly_count > 5 {
            recommendations.push(
                "Consider adjusting learning rate or implementing gradient clipping".to_string(),
            );
        }
        if anomaly_count > 15 {
            recommendations
                .push("Training may need to be restarted with better configuration".to_string());
        }
        if anomaly_count == 0 {
            recommendations.push("No anomalies detected, training appears stable".to_string());
        }

        Ok(crate::QuickAnomalySummary {
            anomaly_count,
            severity_level,
            recommendations,
        })
    }

    /// Generate anomaly detection report
    pub async fn generate_report(&self) -> Result<AnomalyDetectorReport> {
        let mut anomaly_counts = HashMap::new();
        for anomaly in &self.detected_anomalies {
            let count = anomaly_counts.entry(format!("{:?}", anomaly.anomaly_type)).or_insert(0);
            *count += 1;
        }

        Ok(AnomalyDetectorReport {
            session_duration: Utc::now().signed_duration_since(self.start_time),
            total_anomalies: self.detected_anomalies.len(),
            anomaly_counts,
            most_recent_anomalies: self.detected_anomalies.iter().rev().take(10).cloned().collect(),
            config: self.config.clone(),
        })
    }
}

/// Report generated by the anomaly detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectorReport {
    pub session_duration: chrono::Duration,
    pub total_anomalies: usize,
    pub anomaly_counts: HashMap<String, usize>,
    pub most_recent_anomalies: Vec<Anomaly>,
    pub config: AnomalyDetectorConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detector_creation() {
        let config = DebugConfig::default();
        let detector = AnomalyDetector::new(&config);
        assert_eq!(detector.get_anomalies().len(), 0);
    }

    #[test]
    fn test_nan_detection() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        let values = vec![1.0, 2.0, f32::NAN, 4.0];
        detector.check_nan(&values, "test_location").expect("operation failed in test");

        assert_eq!(detector.get_anomalies().len(), 1);
        assert!(matches!(
            detector.get_anomalies()[0].anomaly_type,
            AnomalyType::NaN
        ));
    }

    #[test]
    fn test_inf_detection() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        let values = vec![1.0, 2.0, f32::INFINITY, 4.0];
        detector.check_inf(&values, "test_location").expect("operation failed in test");

        assert_eq!(detector.get_anomalies().len(), 1);
        assert!(matches!(
            detector.get_anomalies()[0].anomaly_type,
            AnomalyType::Infinity
        ));
    }

    #[test]
    fn test_gradient_explosion_detection() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        detector
            .check_gradient_explosion(1e7, "test_layer")
            .expect("operation failed in test");

        assert_eq!(detector.get_anomalies().len(), 1);
        assert!(matches!(
            detector.get_anomalies()[0].anomaly_type,
            AnomalyType::GradientExplosion
        ));
    }

    #[test]
    fn test_gradient_vanishing_detection() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        detector
            .check_gradient_vanishing(1e-10, "test_layer")
            .expect("operation failed in test");

        assert_eq!(detector.get_anomalies().len(), 1);
        assert!(matches!(
            detector.get_anomalies()[0].anomaly_type,
            AnomalyType::GradientVanishing
        ));
    }

    #[test]
    fn test_numerical_instability_detection() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        // Test near-zero values
        let near_zero_values: Vec<f32> =
            (0..100).map(|i| if i < 50 { 1e-12 } else { 1.0 }).collect();
        detector
            .check_numerical_instability(&near_zero_values, "test_location")
            .expect("operation failed in test");
        assert_eq!(detector.get_anomalies().len(), 1);

        detector.clear_anomalies();

        // Test extreme values
        let extreme_values = vec![1.0, 2.0, 1e7, 4.0];
        detector
            .check_numerical_instability(&extreme_values, "test_location")
            .expect("operation failed in test");
        assert_eq!(detector.get_anomalies().len(), 1);
    }

    #[test]
    fn test_activation_saturation_detection() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        // Test ReLU saturation (all zeros)
        let relu_saturated: Vec<f32> = vec![0.0; 100];
        detector
            .check_activation_saturation(&relu_saturated, "relu", "test_layer")
            .expect("operation failed in test");
        assert_eq!(detector.get_anomalies().len(), 1);

        detector.clear_anomalies();

        // Test sigmoid saturation (all ones)
        let sigmoid_saturated: Vec<f32> = vec![0.999; 100];
        detector
            .check_activation_saturation(&sigmoid_saturated, "sigmoid", "test_layer")
            .expect("operation failed in test");
        assert_eq!(detector.get_anomalies().len(), 1);
    }

    #[test]
    fn test_memory_leak_detection() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        // Test memory growth detection (3x growth should trigger)
        detector
            .check_memory_leak(3072, Some(1024), "test_location")
            .expect("operation failed in test");
        assert_eq!(detector.get_anomalies().len(), 1);
        assert!(matches!(
            detector.get_anomalies()[0].anomaly_type,
            AnomalyType::MemoryLeak
        ));

        detector.clear_anomalies();

        // Test absolute high memory
        detector
            .check_memory_leak(10240, None, "test_location")
            .expect("operation failed in test");
        assert_eq!(detector.get_anomalies().len(), 1);
    }

    #[test]
    fn test_weight_explosion_detection() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        let weights = vec![1.0, 2.0, 15.0, 4.0, -20.0]; // Two weights exceed threshold of 10.0
        detector
            .check_weight_explosion(&weights, "test_layer")
            .expect("operation failed in test");

        assert_eq!(detector.get_anomalies().len(), 1);
        assert!(matches!(
            detector.get_anomalies()[0].anomaly_type,
            AnomalyType::UnusualActivation
        ));
    }

    #[test]
    fn test_gradient_conflict_detection() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        let mut layer_gradients = HashMap::new();
        layer_gradients.insert("layer1".to_string(), vec![1.0, 0.0, 0.0]);
        layer_gradients.insert("layer2".to_string(), vec![-1.0, 0.0, 0.0]); // Opposing gradients

        detector
            .check_gradient_conflict(&layer_gradients)
            .expect("operation failed in test");

        assert_eq!(detector.get_anomalies().len(), 1);
        assert!(matches!(
            detector.get_anomalies()[0].anomaly_type,
            AnomalyType::GradientConflict
        ));
    }

    #[test]
    fn test_weight_divergence_detection() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        let baseline_weights = vec![1.0, 2.0, 3.0, 4.0];
        let diverged_weights = vec![10.0, 20.0, 30.0, 40.0]; // Significant divergence

        // First call establishes baseline
        detector
            .check_weight_divergence("test_layer", &baseline_weights)
            .expect("operation failed in test");
        assert_eq!(detector.get_anomalies().len(), 0);

        // Second call detects divergence
        detector
            .check_weight_divergence("test_layer", &diverged_weights)
            .expect("operation failed in test");
        assert_eq!(detector.get_anomalies().len(), 1);
        assert!(matches!(
            detector.get_anomalies()[0].anomaly_type,
            AnomalyType::WeightDivergence
        ));
    }

    #[test]
    fn test_performance_degradation_detection() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        // Add baseline performance metrics
        for _ in 0..10 {
            detector
                .check_performance_degradation(100.0, "training")
                .expect("operation failed in test"); // Good performance
        }
        assert_eq!(detector.get_anomalies().len(), 0);

        // Add degraded performance metrics - just enough to trigger once
        for _ in 0..5 {
            detector
                .check_performance_degradation(20.0, "training")
                .expect("operation failed in test"); // Poor performance
        }

        // Should have at least one degradation anomaly
        assert!(!detector.get_anomalies().is_empty());
        assert!(detector
            .get_anomalies()
            .iter()
            .any(|a| matches!(a.anomaly_type, AnomalyType::PerformanceDegradation)));
    }

    #[test]
    fn test_loss_anomaly_detection() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        // Add normal loss values
        detector.check_loss_anomaly(1.0, "training").expect("operation failed in test");
        detector.check_loss_anomaly(0.9, "training").expect("operation failed in test");
        assert_eq!(detector.get_anomalies().len(), 0);

        // Add loss spike
        detector
            .check_loss_anomaly(100.0, "training")
            .expect("operation failed in test"); // 100x spike
        assert_eq!(detector.get_anomalies().len(), 1);
        assert!(matches!(
            detector.get_anomalies()[0].anomaly_type,
            AnomalyType::LossAnomalous
        ));
    }

    #[tokio::test]
    async fn test_auto_recovery() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);
        detector.config.enable_auto_recovery = true;

        let anomaly = Anomaly {
            anomaly_type: AnomalyType::GradientExplosion,
            timestamp: Utc::now(),
            location: "test_layer".to_string(),
            description: "Test gradient explosion".to_string(),
            severity: AnomalySeverity::High,
            metadata: HashMap::new(),
        };

        let action = detector.attempt_recovery(&anomaly).await.expect("temp file creation failed");
        assert!(matches!(action, RecoveryAction::ClipGradients { .. }));
        assert_eq!(detector.get_recovery_attempts().len(), 1);

        // Regression: without a `TrainingControl` attached, the old
        // implementation still reported every recovery attempt (other than
        // `EmergencyStop`) as a silent, unconditional success. It must now
        // be honestly recorded as *not performed*.
        let attempt = &detector.get_recovery_attempts()[0];
        assert!(
            !attempt.success,
            "must not report success when no TrainingControl was ever attached"
        );
        assert!(
            attempt.error_message.is_some(),
            "must record a real reason, not silently claim success"
        );
    }

    /// A [`TrainingControl`] mock that records every call it receives and
    /// lets a test assert the anomaly detector actually drove it -- rather
    /// than merely logging and claiming success.
    #[derive(Debug, Default)]
    struct RecordingTrainingControl {
        calls: Vec<String>,
        fail_next: bool,
    }

    impl TrainingControl for RecordingTrainingControl {
        fn reset_gradients(&mut self) -> Result<()> {
            self.calls.push("reset_gradients".to_string());
            Ok(())
        }
        fn reduce_learning_rate(&mut self, factor: f64) -> Result<()> {
            self.calls.push(format!("reduce_learning_rate({factor})"));
            Ok(())
        }
        fn clip_gradients(&mut self, max_norm: f64) -> Result<()> {
            self.calls.push(format!("clip_gradients({max_norm})"));
            if self.fail_next {
                anyhow::bail!("simulated clip_gradients failure");
            }
            Ok(())
        }
        fn restart_optimizer(&mut self) -> Result<()> {
            self.calls.push("restart_optimizer".to_string());
            Ok(())
        }
        fn skip_batch(&mut self) -> Result<()> {
            self.calls.push("skip_batch".to_string());
            Ok(())
        }
        fn reset_weights(&mut self, layer_name: &str) -> Result<()> {
            self.calls.push(format!("reset_weights({layer_name})"));
            Ok(())
        }
        fn apply_weight_decay(&mut self, rate: f64) -> Result<()> {
            self.calls.push(format!("apply_weight_decay({rate})"));
            Ok(())
        }
        fn emergency_stop(&mut self) -> Result<()> {
            self.calls.push("emergency_stop".to_string());
            Ok(())
        }
    }

    /// Regression test: with a real `TrainingControl` attached, a recovery
    /// action must actually be dispatched to it (with the real parameters
    /// from `determine_recovery_action`), and the attempt must be recorded
    /// as a real success -- this is the "real work" side of the fix.
    #[tokio::test]
    async fn test_auto_recovery_with_training_control_actually_dispatches() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);
        detector.config.enable_auto_recovery = true;
        assert!(!detector.has_training_control());
        detector.set_training_control(Box::new(RecordingTrainingControl::default()));
        assert!(detector.has_training_control());

        let anomaly = Anomaly {
            anomaly_type: AnomalyType::GradientExplosion,
            timestamp: Utc::now(),
            location: "test_layer".to_string(),
            description: "Test gradient explosion".to_string(),
            severity: AnomalySeverity::High,
            metadata: HashMap::new(),
        };

        let action = detector.attempt_recovery(&anomaly).await.expect("recovery should not error");
        assert!(matches!(action, RecoveryAction::ClipGradients { max_norm } if max_norm == 1.0));

        let attempt = &detector.get_recovery_attempts()[0];
        assert!(
            attempt.success,
            "a real, successful TrainingControl call must report success"
        );
        assert!(attempt.error_message.is_none());
    }

    /// Regression test: when the attached `TrainingControl` itself fails,
    /// that must surface as a real, non-generic error message on the
    /// recorded attempt -- not the old hardcoded `"Recovery failed"`
    /// string, and not a silent `Ok(true)`.
    #[tokio::test]
    async fn test_auto_recovery_reports_real_error_from_training_control() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);
        detector.config.enable_auto_recovery = true;
        detector.set_training_control(Box::new(RecordingTrainingControl {
            calls: Vec::new(),
            fail_next: true,
        }));

        let anomaly = Anomaly {
            anomaly_type: AnomalyType::GradientExplosion,
            timestamp: Utc::now(),
            location: "test_layer".to_string(),
            description: "Test gradient explosion".to_string(),
            severity: AnomalySeverity::High,
            metadata: HashMap::new(),
        };

        detector.attempt_recovery(&anomaly).await.expect("recovery should not error");
        let attempt = &detector.get_recovery_attempts()[0];
        assert!(!attempt.success);
        let message = attempt.error_message.as_deref().unwrap_or_default();
        assert!(
            message.contains("simulated clip_gradients failure"),
            "must surface the real underlying error, not a generic placeholder: {message}"
        );
        assert_ne!(
            message, "Recovery failed",
            "must not be the old generic placeholder string"
        );
    }

    #[test]
    fn test_monitoring_stats() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        // Create some anomalies to generate stats
        detector.check_nan(&[f32::NAN], "test").expect("operation failed in test");
        detector.check_inf(&[f32::INFINITY], "test").expect("operation failed in test");

        let stats = detector.get_monitoring_stats();
        assert_eq!(stats.total_anomalies, 2);
        assert!(stats.anomalies_per_type.contains_key("NaN"));
        assert!(stats.anomalies_per_type.contains_key("Infinity"));
    }

    #[test]
    fn test_monitoring_window_update() {
        let config = DebugConfig::default();
        let mut detector = AnomalyDetector::new(&config);

        detector.check_nan(&[f32::NAN], "test").expect("operation failed in test");
        detector.update_monitoring_window().expect("operation failed in test");

        let stats = detector.get_monitoring_stats();
        assert_eq!(stats.monitoring_window.len(), 1);
        assert_eq!(stats.monitoring_window[0].anomaly_count, 1);
    }
}
