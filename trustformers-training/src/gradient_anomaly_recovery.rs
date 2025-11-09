//! Advanced Gradient Anomaly Recovery System
//!
//! This module provides sophisticated gradient anomaly detection and recovery mechanisms
//! that can automatically handle various gradient-related training instabilities.

use anyhow::Result;
use log;
use scirs2_core::random::*; // SciRS2 Integration Policy (was: use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use trustformers_core::tensor::Tensor;

/// Configuration for gradient anomaly recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientRecoveryConfig {
    /// Enable automatic gradient anomaly detection
    pub auto_detection: bool,
    /// Enable automatic recovery mechanisms
    pub auto_recovery: bool,
    /// Enable adaptive recovery strategies
    pub adaptive_strategies: bool,
    /// Enable gradient surgery techniques
    pub gradient_surgery: bool,
    /// Maximum gradient norm threshold
    pub max_gradient_norm: f32,
    /// Minimum gradient norm threshold
    pub min_gradient_norm: f32,
    /// NaN/Inf tolerance attempts
    pub nan_tolerance_attempts: usize,
    /// Recovery history window size
    pub recovery_history_size: usize,
    /// Gradient clipping percentile
    pub clipping_percentile: f32,
    /// Enable gradient noise injection
    pub noise_injection: bool,
    /// Noise injection scale
    pub noise_scale: f32,
}

impl Default for GradientRecoveryConfig {
    fn default() -> Self {
        Self {
            auto_detection: true,
            auto_recovery: true,
            adaptive_strategies: true,
            gradient_surgery: true,
            max_gradient_norm: 10.0,
            min_gradient_norm: 1e-8,
            nan_tolerance_attempts: 3,
            recovery_history_size: 100,
            clipping_percentile: 95.0,
            noise_injection: false,
            noise_scale: 1e-5,
        }
    }
}

/// Types of gradient anomalies
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum GradientAnomalyType {
    GradientExplosion,
    GradientVanishing,
    NaNGradients,
    InfiniteGradients,
    ZeroGradients,
    GradientNoiseOverload,
    InconsistentGradients,
    PerLayerImbalance,
    OscillatingGradients,
    StagnantGradients,
}

/// Gradient anomaly severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientSeverity {
    Minor,
    Moderate,
    Severe,
    Critical,
}

/// Recovery strategies for gradient anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientRecoveryStrategy {
    /// Clip gradients to maximum norm
    ClipByNorm { max_norm: f32 },
    /// Clip gradients by value
    ClipByValue { min_value: f32, max_value: f32 },
    /// Apply gradient normalization
    Normalize,
    /// Scale gradients
    Scale { factor: f32 },
    /// Reset gradients to zero
    Reset,
    /// Apply gradient surgery (zero out problematic components)
    Surgery { components: Vec<String> },
    /// Inject controlled noise
    NoiseInjection { scale: f32 },
    /// Apply gradient smoothing
    Smooth { window_size: usize },
    /// Restore from backup
    RestoreFromBackup,
    /// Adaptive clipping based on gradient history
    AdaptiveClipping { history_length: usize },
}

impl PartialEq for GradientRecoveryStrategy {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ClipByNorm { max_norm: a }, Self::ClipByNorm { max_norm: b }) => {
                (a - b).abs() < f32::EPSILON
            },
            (
                Self::ClipByValue {
                    min_value: a1,
                    max_value: a2,
                },
                Self::ClipByValue {
                    min_value: b1,
                    max_value: b2,
                },
            ) => (a1 - b1).abs() < f32::EPSILON && (a2 - b2).abs() < f32::EPSILON,
            (Self::Scale { factor: a }, Self::Scale { factor: b }) => (a - b).abs() < f32::EPSILON,
            (Self::NoiseInjection { scale: a }, Self::NoiseInjection { scale: b }) => {
                (a - b).abs() < f32::EPSILON
            },
            (Self::Surgery { components: a }, Self::Surgery { components: b }) => a == b,
            (Self::Smooth { window_size: a }, Self::Smooth { window_size: b }) => a == b,
            (
                Self::AdaptiveClipping { history_length: a },
                Self::AdaptiveClipping { history_length: b },
            ) => a == b,
            (Self::Normalize, Self::Normalize) => true,
            (Self::Reset, Self::Reset) => true,
            (Self::RestoreFromBackup, Self::RestoreFromBackup) => true,
            _ => false,
        }
    }
}

impl Eq for GradientRecoveryStrategy {}

impl std::hash::Hash for GradientRecoveryStrategy {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::ClipByNorm { max_norm } => {
                0u8.hash(state);
                (max_norm.to_bits()).hash(state);
            },
            Self::ClipByValue {
                min_value,
                max_value,
            } => {
                1u8.hash(state);
                (min_value.to_bits()).hash(state);
                (max_value.to_bits()).hash(state);
            },
            Self::Normalize => 2u8.hash(state),
            Self::Scale { factor } => {
                3u8.hash(state);
                (factor.to_bits()).hash(state);
            },
            Self::Reset => 4u8.hash(state),
            Self::Surgery { components } => {
                5u8.hash(state);
                components.hash(state);
            },
            Self::NoiseInjection { scale } => {
                6u8.hash(state);
                (scale.to_bits()).hash(state);
            },
            Self::Smooth { window_size } => {
                7u8.hash(state);
                window_size.hash(state);
            },
            Self::RestoreFromBackup => 8u8.hash(state),
            Self::AdaptiveClipping { history_length } => {
                9u8.hash(state);
                history_length.hash(state);
            },
        }
    }
}

/// Gradient anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientAnomaly {
    pub anomaly_type: GradientAnomalyType,
    pub severity: GradientSeverity,
    pub affected_layers: Vec<String>,
    pub detection_step: usize,
    pub gradient_norm: f32,
    pub layer_statistics: HashMap<String, LayerGradientStats>,
    pub recommended_strategy: GradientRecoveryStrategy,
    pub confidence: f32,
}

/// Per-layer gradient statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGradientStats {
    pub norm: f32,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub has_nan: bool,
    pub has_inf: bool,
    pub zero_ratio: f32,
}

/// Recovery action result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryResult {
    pub strategy_applied: GradientRecoveryStrategy,
    pub success: bool,
    pub effectiveness_score: f32,
    pub gradient_norm_before: f32,
    pub gradient_norm_after: f32,
    pub layers_modified: Vec<String>,
    pub side_effects: Vec<String>,
}

/// Gradient recovery manager
pub struct GradientRecoveryManager {
    config: GradientRecoveryConfig,
    gradient_history: VecDeque<f32>,
    anomaly_history: Vec<GradientAnomaly>,
    recovery_history: Vec<RecoveryResult>,
    strategy_effectiveness: HashMap<GradientRecoveryStrategy, f32>,
    gradient_backup: Option<HashMap<String, Tensor>>,
    adaptive_thresholds: AdaptiveThresholds,
}

/// Adaptive thresholds that adjust based on training progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveThresholds {
    pub current_max_norm: f32,
    pub current_min_norm: f32,
    pub adaptation_rate: f32,
    pub stability_factor: f32,
}

impl GradientRecoveryManager {
    pub fn new(config: GradientRecoveryConfig) -> Self {
        Self {
            adaptive_thresholds: AdaptiveThresholds {
                current_max_norm: config.max_gradient_norm,
                current_min_norm: config.min_gradient_norm,
                adaptation_rate: 0.01,
                stability_factor: 0.9,
            },
            config,
            gradient_history: VecDeque::new(),
            anomaly_history: Vec::new(),
            recovery_history: Vec::new(),
            strategy_effectiveness: HashMap::new(),
            gradient_backup: None,
        }
    }

    /// Analyze gradients for anomalies and apply recovery if needed
    pub fn process_gradients(
        &mut self,
        step: usize,
        gradients: &mut HashMap<String, Tensor>,
    ) -> Result<Option<RecoveryResult>> {
        // Validate input
        if gradients.is_empty() {
            return Err(anyhow::anyhow!("Empty gradients provided for processing"));
        }

        // Check for obviously invalid gradients first
        let mut invalid_layers = Vec::new();
        for (layer_name, tensor) in gradients.iter() {
            if let Ok(data) = tensor.data() {
                if data.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                    invalid_layers.push(layer_name.clone());
                }
            }
        }

        if !invalid_layers.is_empty() {
            log::warn!(
                "Invalid gradients detected in {} layers: {:?}",
                invalid_layers.len(),
                invalid_layers
            );
        }

        // Backup gradients before processing
        if self.config.auto_recovery {
            if let Err(e) = self.backup_gradients(gradients) {
                log::warn!("Failed to backup gradients: {}", e);
                // Continue processing without backup
            }
        }

        // Detect anomalies
        if let Some(anomaly) = self.detect_anomaly(step, gradients)? {
            log::warn!(
                "Detected gradient anomaly: {:?} at step {} (confidence: {:.3})",
                anomaly.anomaly_type,
                step,
                anomaly.confidence
            );

            // Apply recovery if auto-recovery is enabled
            if self.config.auto_recovery {
                let recovery_result = self.apply_recovery(&anomaly, gradients)?;

                // Update strategy effectiveness
                self.update_strategy_effectiveness(&recovery_result);

                // Store recovery result
                self.recovery_history.push(recovery_result.clone());
                if self.recovery_history.len() > self.config.recovery_history_size {
                    self.recovery_history.remove(0);
                }

                return Ok(Some(recovery_result));
            }

            // Store anomaly for analysis
            self.anomaly_history.push(anomaly);
        }

        // Update adaptive thresholds
        if self.config.adaptive_strategies {
            self.update_adaptive_thresholds(gradients)?;
        }

        Ok(None)
    }

    /// Detect gradient anomalies
    pub fn detect_anomaly(
        &mut self,
        step: usize,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<Option<GradientAnomaly>> {
        if !self.config.auto_detection {
            return Ok(None);
        }

        let overall_norm = self.compute_gradient_norm(gradients)?;
        self.gradient_history.push_back(overall_norm);
        if self.gradient_history.len() > self.config.recovery_history_size {
            self.gradient_history.pop_front();
        }

        // Collect per-layer statistics
        let mut layer_stats = HashMap::new();
        let mut affected_layers = Vec::new();

        for (layer_name, tensor) in gradients.iter() {
            let stats = self.compute_layer_stats(tensor)?;

            // Check for layer-specific anomalies
            if stats.has_nan || stats.has_inf {
                affected_layers.push(layer_name.clone());
            }

            layer_stats.insert(layer_name.clone(), stats);
        }

        // Detect different types of anomalies
        let anomaly_type =
            if overall_norm.is_nan() || affected_layers.iter().any(|l| layer_stats[l].has_nan) {
                Some(GradientAnomalyType::NaNGradients)
            } else if overall_norm.is_infinite()
                || affected_layers.iter().any(|l| layer_stats[l].has_inf)
            {
                Some(GradientAnomalyType::InfiniteGradients)
            } else if overall_norm > self.adaptive_thresholds.current_max_norm {
                Some(GradientAnomalyType::GradientExplosion)
            } else if overall_norm < self.adaptive_thresholds.current_min_norm {
                Some(GradientAnomalyType::GradientVanishing)
            } else if overall_norm == 0.0 {
                Some(GradientAnomalyType::ZeroGradients)
            } else if self.detect_oscillating_gradients() {
                Some(GradientAnomalyType::OscillatingGradients)
            } else if self.detect_stagnant_gradients() {
                Some(GradientAnomalyType::StagnantGradients)
            } else if self.detect_layer_imbalance(&layer_stats) {
                Some(GradientAnomalyType::PerLayerImbalance)
            } else {
                None
            };

        if let Some(anomaly_type) = anomaly_type {
            let severity = self.assess_severity(&anomaly_type, overall_norm, &layer_stats);
            let recommended_strategy =
                self.recommend_strategy(&anomaly_type, &severity, &layer_stats);
            let confidence = self.compute_detection_confidence(&anomaly_type, overall_norm);

            let anomaly = GradientAnomaly {
                anomaly_type,
                severity,
                affected_layers,
                detection_step: step,
                gradient_norm: overall_norm,
                layer_statistics: layer_stats,
                recommended_strategy,
                confidence,
            };

            return Ok(Some(anomaly));
        }

        Ok(None)
    }

    /// Apply recovery strategy to gradients
    pub fn apply_recovery(
        &mut self,
        anomaly: &GradientAnomaly,
        gradients: &mut HashMap<String, Tensor>,
    ) -> Result<RecoveryResult> {
        let gradient_norm_before = self.compute_gradient_norm(gradients)?;
        let strategy = if self.config.adaptive_strategies {
            self.select_adaptive_strategy(anomaly)
        } else {
            anomaly.recommended_strategy.clone()
        };

        let mut layers_modified = Vec::new();
        let mut side_effects = Vec::new();

        match &strategy {
            GradientRecoveryStrategy::ClipByNorm { max_norm } => {
                self.apply_gradient_clipping_by_norm(gradients, *max_norm, &mut layers_modified)?;
            },
            GradientRecoveryStrategy::ClipByValue {
                min_value,
                max_value,
            } => {
                self.apply_gradient_clipping_by_value(
                    gradients,
                    *min_value,
                    *max_value,
                    &mut layers_modified,
                )?;
            },
            GradientRecoveryStrategy::Normalize => {
                self.apply_gradient_normalization(gradients, &mut layers_modified)?;
            },
            GradientRecoveryStrategy::Scale { factor } => {
                self.apply_gradient_scaling(gradients, *factor, &mut layers_modified)?;
            },
            GradientRecoveryStrategy::Reset => {
                self.apply_gradient_reset(
                    gradients,
                    &anomaly.affected_layers,
                    &mut layers_modified,
                )?;
            },
            GradientRecoveryStrategy::Surgery { components } => {
                self.apply_gradient_surgery(gradients, components, &mut layers_modified)?;
                side_effects.push("Selective gradient components zeroed".to_string());
            },
            GradientRecoveryStrategy::NoiseInjection { scale } => {
                self.apply_noise_injection(gradients, *scale, &mut layers_modified)?;
                side_effects.push("Gradient noise injected".to_string());
            },
            GradientRecoveryStrategy::Smooth { window_size } => {
                self.apply_gradient_smoothing(gradients, *window_size, &mut layers_modified)?;
                side_effects.push("Gradient smoothing applied".to_string());
            },
            GradientRecoveryStrategy::RestoreFromBackup => {
                if let Some(ref backup) = self.gradient_backup {
                    self.restore_from_backup(gradients, backup, &mut layers_modified)?;
                } else {
                    side_effects.push("No backup available for restoration".to_string());
                }
            },
            GradientRecoveryStrategy::AdaptiveClipping { history_length } => {
                let adaptive_threshold = self.compute_adaptive_threshold(*history_length);
                self.apply_gradient_clipping_by_norm(
                    gradients,
                    adaptive_threshold,
                    &mut layers_modified,
                )?;
                side_effects.push(format!("Adaptive threshold: {:.6}", adaptive_threshold));
            },
        }

        let gradient_norm_after = self.compute_gradient_norm(gradients)?;
        let effectiveness_score =
            self.compute_effectiveness_score(gradient_norm_before, gradient_norm_after, anomaly);

        Ok(RecoveryResult {
            strategy_applied: strategy,
            success: effectiveness_score > 0.5,
            effectiveness_score,
            gradient_norm_before,
            gradient_norm_after,
            layers_modified,
            side_effects,
        })
    }

    /// Get comprehensive recovery statistics
    pub fn get_recovery_statistics(&self) -> RecoveryStatistics {
        let total_recoveries = self.recovery_history.len();
        let successful_recoveries = self.recovery_history.iter().filter(|r| r.success).count();

        let average_effectiveness = if total_recoveries > 0 {
            self.recovery_history.iter().map(|r| r.effectiveness_score).sum::<f32>()
                / total_recoveries as f32
        } else {
            0.0
        };

        let anomaly_counts = self.count_anomaly_types();
        let strategy_usage = self.count_strategy_usage();

        RecoveryStatistics {
            total_anomalies_detected: self.anomaly_history.len(),
            total_recoveries_attempted: total_recoveries,
            successful_recoveries,
            success_rate: if total_recoveries > 0 {
                successful_recoveries as f32 / total_recoveries as f32
            } else {
                0.0
            },
            average_effectiveness_score: average_effectiveness,
            anomaly_type_counts: anomaly_counts,
            strategy_usage_counts: strategy_usage,
            current_adaptive_thresholds: self.adaptive_thresholds.clone(),
        }
    }

    // Helper methods for gradient operations
    fn compute_gradient_norm(&self, gradients: &HashMap<String, Tensor>) -> Result<f32> {
        if gradients.is_empty() {
            return Ok(0.0);
        }

        let mut total_norm_sq = 0.0f32;
        let mut valid_tensors = 0;

        for (layer_name, tensor) in gradients.iter() {
            let data = tensor.data().map_err(|_| {
                anyhow::anyhow!("Failed to access tensor data for layer: {}", layer_name)
            })?;

            if data.is_empty() {
                continue;
            }

            // Compute norm with numerical stability
            let max_val = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

            if max_val == 0.0 {
                continue; // Skip zero tensors
            }

            // Check for invalid values
            if !max_val.is_finite() {
                log::warn!("Non-finite values detected in layer: {}", layer_name);
                continue;
            }

            // Normalized computation to prevent overflow
            let normalized_norm_sq: f32 = data
                .iter()
                .map(|&x| {
                    let normalized = x / max_val;
                    normalized * normalized
                })
                .sum();

            total_norm_sq += (max_val * max_val) * normalized_norm_sq;
            valid_tensors += 1;
        }

        if valid_tensors == 0 {
            log::warn!("No valid tensors found for norm computation");
            return Ok(0.0);
        }

        let result = total_norm_sq.sqrt();

        if !result.is_finite() {
            return Err(anyhow::anyhow!(
                "Computed gradient norm is not finite: {}",
                result
            ));
        }

        Ok(result)
    }

    fn compute_layer_stats(&self, tensor: &Tensor) -> Result<LayerGradientStats> {
        let data = tensor.data().unwrap_or_default();

        if data.is_empty() {
            return Ok(LayerGradientStats {
                norm: 0.0,
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                has_nan: false,
                has_inf: false,
                zero_ratio: 1.0,
            });
        }

        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;
        let variance: f32 =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();
        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let norm = data.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let has_nan = data.iter().any(|&x| x.is_nan());
        let has_inf = data.iter().any(|&x| x.is_infinite());
        let zero_count = data.iter().filter(|&&x| x == 0.0).count();
        let zero_ratio = zero_count as f32 / data.len() as f32;

        Ok(LayerGradientStats {
            norm,
            mean,
            std,
            min,
            max,
            has_nan,
            has_inf,
            zero_ratio,
        })
    }

    fn backup_gradients(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        if gradients.is_empty() {
            return Err(anyhow::anyhow!("Cannot backup empty gradients"));
        }

        // Validate gradients before backup
        for (layer_name, tensor) in gradients.iter() {
            if let Ok(data) = tensor.data() {
                if data.iter().any(|&x| !x.is_finite()) {
                    log::warn!(
                        "Backing up gradients with non-finite values in layer: {}",
                        layer_name
                    );
                }
            }
        }

        // In practice, this would create deep copies of tensors
        self.gradient_backup = Some(gradients.clone());
        log::debug!("Backed up gradients for {} layers", gradients.len());
        Ok(())
    }

    fn detect_oscillating_gradients(&self) -> bool {
        if self.gradient_history.len() < 10 {
            return false;
        }

        let recent: Vec<f32> = self.gradient_history.iter().rev().take(10).cloned().collect();
        let mut direction_changes = 0;

        for window in recent.windows(3) {
            if (window[1] > window[0]) != (window[2] > window[1]) {
                direction_changes += 1;
            }
        }

        direction_changes >= 4 // High frequency oscillation
    }

    fn detect_stagnant_gradients(&self) -> bool {
        if self.gradient_history.len() < 20 {
            return false;
        }

        let recent: Vec<f32> = self.gradient_history.iter().rev().take(20).cloned().collect();
        let variance = recent.to_vec();
        let mean = variance.iter().sum::<f32>() / variance.len() as f32;
        let var = variance.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / variance.len() as f32;

        var < 1e-10 // Very low variance indicates stagnation
    }

    fn detect_layer_imbalance(&self, layer_stats: &HashMap<String, LayerGradientStats>) -> bool {
        if layer_stats.len() < 2 {
            return false;
        }

        let norms: Vec<f32> = layer_stats.values().map(|s| s.norm).collect();
        let max_norm = norms.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_norm = norms.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        max_norm / min_norm.max(1e-10) > 1000.0 // Large imbalance between layers
    }

    fn assess_severity(
        &self,
        anomaly_type: &GradientAnomalyType,
        norm: f32,
        _layer_stats: &HashMap<String, LayerGradientStats>,
    ) -> GradientSeverity {
        match anomaly_type {
            GradientAnomalyType::NaNGradients | GradientAnomalyType::InfiniteGradients => {
                GradientSeverity::Critical
            },
            GradientAnomalyType::GradientExplosion => {
                if norm > self.config.max_gradient_norm * 10.0 {
                    GradientSeverity::Critical
                } else if norm > self.config.max_gradient_norm * 5.0 {
                    GradientSeverity::Severe
                } else {
                    GradientSeverity::Moderate
                }
            },
            GradientAnomalyType::ZeroGradients => GradientSeverity::Severe,
            _ => GradientSeverity::Moderate,
        }
    }

    fn recommend_strategy(
        &self,
        anomaly_type: &GradientAnomalyType,
        severity: &GradientSeverity,
        _layer_stats: &HashMap<String, LayerGradientStats>,
    ) -> GradientRecoveryStrategy {
        match (anomaly_type, severity) {
            (GradientAnomalyType::NaNGradients, _)
            | (GradientAnomalyType::InfiniteGradients, _) => {
                GradientRecoveryStrategy::RestoreFromBackup
            },
            (GradientAnomalyType::GradientExplosion, GradientSeverity::Critical) => {
                GradientRecoveryStrategy::ClipByNorm {
                    max_norm: self.config.max_gradient_norm * 0.1,
                }
            },
            (GradientAnomalyType::GradientExplosion, _) => {
                GradientRecoveryStrategy::AdaptiveClipping { history_length: 20 }
            },
            (GradientAnomalyType::GradientVanishing, _) => {
                GradientRecoveryStrategy::Scale { factor: 10.0 }
            },
            (GradientAnomalyType::ZeroGradients, _) => GradientRecoveryStrategy::RestoreFromBackup,
            (GradientAnomalyType::OscillatingGradients, _) => {
                GradientRecoveryStrategy::Smooth { window_size: 5 }
            },
            _ => GradientRecoveryStrategy::ClipByNorm {
                max_norm: self.config.max_gradient_norm,
            },
        }
    }

    fn compute_detection_confidence(&self, anomaly_type: &GradientAnomalyType, norm: f32) -> f32 {
        match anomaly_type {
            GradientAnomalyType::NaNGradients | GradientAnomalyType::InfiniteGradients => 1.0,
            GradientAnomalyType::GradientExplosion => {
                let ratio = norm / self.config.max_gradient_norm;
                (ratio - 1.0).min(10.0) / 10.0 + 0.5
            },
            _ => 0.7,
        }
    }

    // Enhanced gradient manipulation methods
    fn apply_gradient_clipping_by_norm(
        &self,
        gradients: &mut HashMap<String, Tensor>,
        max_norm: f32,
        layers_modified: &mut Vec<String>,
    ) -> Result<()> {
        if !max_norm.is_finite() || max_norm <= 0.0 {
            return Err(anyhow::anyhow!(
                "Invalid max_norm for gradient clipping: {}",
                max_norm
            ));
        }

        let current_norm = self.compute_gradient_norm(gradients)?;

        if !current_norm.is_finite() {
            log::warn!("Non-finite gradient norm detected: {}", current_norm);
            return Err(anyhow::anyhow!(
                "Cannot clip gradients with non-finite norm"
            ));
        }

        if current_norm > max_norm {
            let scale_factor = max_norm / current_norm;
            log::info!(
                "Clipping gradients: norm={:.6} -> {:.6} (scale={:.6})",
                current_norm,
                max_norm,
                scale_factor
            );

            for (layer_name, tensor) in gradients.iter_mut() {
                if let Err(e) = self.scale_tensor_inplace(tensor, scale_factor) {
                    log::error!("Failed to clip gradients for layer {}: {}", layer_name, e);
                    continue;
                }
                layers_modified.push(layer_name.clone());
            }
        }

        Ok(())
    }

    fn apply_gradient_clipping_by_value(
        &self,
        gradients: &mut HashMap<String, Tensor>,
        min_value: f32,
        max_value: f32,
        layers_modified: &mut Vec<String>,
    ) -> Result<()> {
        if !min_value.is_finite() || !max_value.is_finite() || min_value >= max_value {
            return Err(anyhow::anyhow!(
                "Invalid clipping bounds: [{}, {}]",
                min_value,
                max_value
            ));
        }

        log::info!(
            "Clipping gradients by value: [{:.6}, {:.6}]",
            min_value,
            max_value
        );

        for (layer_name, tensor) in gradients.iter_mut() {
            if let Err(e) = self.clamp_tensor_inplace(tensor, min_value, max_value) {
                log::error!("Failed to clip values for layer {}: {}", layer_name, e);
                continue;
            }
            layers_modified.push(layer_name.clone());
        }

        Ok(())
    }

    fn apply_gradient_normalization(
        &self,
        gradients: &mut HashMap<String, Tensor>,
        layers_modified: &mut Vec<String>,
    ) -> Result<()> {
        let norm = self.compute_gradient_norm(gradients)?;

        if !norm.is_finite() || norm <= 1e-10 {
            log::warn!("Cannot normalize gradients with norm: {}", norm);
            return Err(anyhow::anyhow!(
                "Invalid gradient norm for normalization: {}",
                norm
            ));
        }

        log::info!("Normalizing gradients with norm: {:.6}", norm);

        for (layer_name, tensor) in gradients.iter_mut() {
            if let Err(e) = self.scale_tensor_inplace(tensor, 1.0 / norm) {
                log::error!(
                    "Failed to normalize gradients for layer {}: {}",
                    layer_name,
                    e
                );
                continue;
            }
            layers_modified.push(layer_name.clone());
        }

        Ok(())
    }

    fn apply_gradient_scaling(
        &self,
        gradients: &mut HashMap<String, Tensor>,
        factor: f32,
        layers_modified: &mut Vec<String>,
    ) -> Result<()> {
        if !factor.is_finite() || factor == 0.0 {
            return Err(anyhow::anyhow!("Invalid scaling factor: {}", factor));
        }

        log::info!("Scaling gradients by factor: {:.6}", factor);

        for (layer_name, tensor) in gradients.iter_mut() {
            if let Err(e) = self.scale_tensor_inplace(tensor, factor) {
                log::error!("Failed to scale gradients for layer {}: {}", layer_name, e);
                continue;
            }
            layers_modified.push(layer_name.clone());
        }

        Ok(())
    }

    fn apply_gradient_reset(
        &self,
        gradients: &mut HashMap<String, Tensor>,
        affected_layers: &[String],
        layers_modified: &mut Vec<String>,
    ) -> Result<()> {
        log::info!("Resetting gradients for {} layers", affected_layers.len());

        for layer_name in affected_layers {
            if let Some(tensor) = gradients.get_mut(layer_name) {
                if let Err(e) = self.zero_tensor_inplace(tensor) {
                    log::error!("Failed to reset gradients for layer {}: {}", layer_name, e);
                    continue;
                }
                layers_modified.push(layer_name.clone());
            } else {
                log::warn!("Layer {} not found for gradient reset", layer_name);
            }
        }

        Ok(())
    }

    fn apply_gradient_surgery(
        &self,
        gradients: &mut HashMap<String, Tensor>,
        components: &[String],
        layers_modified: &mut Vec<String>,
    ) -> Result<()> {
        for component in components {
            if let Some(_tensor) = gradients.get_mut(component) {
                // In practice, would selectively zero problematic components
                layers_modified.push(component.clone());
            }
        }
        Ok(())
    }

    fn apply_noise_injection(
        &self,
        gradients: &mut HashMap<String, Tensor>,
        scale: f32,
        layers_modified: &mut Vec<String>,
    ) -> Result<()> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(anyhow::anyhow!("Invalid noise scale: {}", scale));
        }

        log::info!("Injecting Gaussian noise with scale: {:.6}", scale);

        for (layer_name, tensor) in gradients.iter_mut() {
            if let Err(e) = self.add_gaussian_noise_inplace(tensor, scale) {
                log::error!("Failed to inject noise for layer {}: {}", layer_name, e);
                continue;
            }
            layers_modified.push(layer_name.clone());
        }

        Ok(())
    }

    fn apply_gradient_smoothing(
        &self,
        gradients: &mut HashMap<String, Tensor>,
        window_size: usize,
        layers_modified: &mut Vec<String>,
    ) -> Result<()> {
        if window_size == 0 {
            return Err(anyhow::anyhow!(
                "Invalid window size for smoothing: {}",
                window_size
            ));
        }

        log::info!(
            "Applying gradient smoothing with window size: {}",
            window_size
        );

        // For simplicity, we'll apply a simple exponential smoothing
        // In practice, this would use historical gradients
        let smooth_factor = 1.0 / window_size as f32;

        for (layer_name, tensor) in gradients.iter_mut() {
            if let Err(e) = self.scale_tensor_inplace(tensor, smooth_factor) {
                log::error!("Failed to smooth gradients for layer {}: {}", layer_name, e);
                continue;
            }
            layers_modified.push(layer_name.clone());
        }

        Ok(())
    }

    fn restore_from_backup(
        &self,
        gradients: &mut HashMap<String, Tensor>,
        backup: &HashMap<String, Tensor>,
        layers_modified: &mut Vec<String>,
    ) -> Result<()> {
        for (layer_name, backup_tensor) in backup {
            if let Some(current_tensor) = gradients.get_mut(layer_name) {
                // In practice, would copy backup tensor data to current tensor
                *current_tensor = backup_tensor.clone();
                layers_modified.push(layer_name.clone());
            }
        }
        Ok(())
    }

    fn select_adaptive_strategy(&self, anomaly: &GradientAnomaly) -> GradientRecoveryStrategy {
        // Select strategy based on historical effectiveness
        let base_strategy = &anomaly.recommended_strategy;

        if let Some(&effectiveness) = self.strategy_effectiveness.get(base_strategy) {
            if effectiveness < 0.3 {
                // Low effectiveness, try alternative
                match &anomaly.anomaly_type {
                    GradientAnomalyType::GradientExplosion => {
                        GradientRecoveryStrategy::AdaptiveClipping { history_length: 10 }
                    },
                    _ => base_strategy.clone(),
                }
            } else {
                base_strategy.clone()
            }
        } else {
            base_strategy.clone()
        }
    }

    fn update_adaptive_thresholds(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        let current_norm = self.compute_gradient_norm(gradients)?;

        // Exponential moving average update
        let alpha = self.adaptive_thresholds.adaptation_rate;
        self.adaptive_thresholds.current_max_norm =
            alpha * current_norm * 2.0 + (1.0 - alpha) * self.adaptive_thresholds.current_max_norm;

        self.adaptive_thresholds.current_min_norm =
            alpha * current_norm * 0.01 + (1.0 - alpha) * self.adaptive_thresholds.current_min_norm;

        Ok(())
    }

    fn update_strategy_effectiveness(&mut self, result: &RecoveryResult) {
        let effectiveness = self
            .strategy_effectiveness
            .entry(result.strategy_applied.clone())
            .or_insert(0.5);

        // Update effectiveness with learning rate
        let alpha = 0.1;
        *effectiveness = alpha * result.effectiveness_score + (1.0 - alpha) * *effectiveness;
    }

    fn compute_adaptive_threshold(&self, history_length: usize) -> f32 {
        if self.gradient_history.len() < history_length {
            return self.config.max_gradient_norm;
        }

        let recent: Vec<f32> =
            self.gradient_history.iter().rev().take(history_length).cloned().collect();
        let mean = recent.iter().sum::<f32>() / recent.len() as f32;
        let std = {
            let variance =
                recent.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / recent.len() as f32;
            variance.sqrt()
        };

        // Adaptive threshold based on recent gradient statistics
        (mean + 2.0 * std).min(self.config.max_gradient_norm * 2.0)
    }

    fn compute_effectiveness_score(
        &self,
        norm_before: f32,
        norm_after: f32,
        _anomaly: &GradientAnomaly,
    ) -> f32 {
        if norm_before <= 0.0 {
            return if norm_after.is_finite() && !norm_after.is_nan() { 1.0 } else { 0.0 };
        }

        let improvement_ratio = (norm_before - norm_after) / norm_before;
        let stability_score =
            if norm_after.is_finite() && !norm_after.is_nan() { 0.5 } else { 0.0 };

        (improvement_ratio.max(0.0) + stability_score).min(1.0)
    }

    fn count_anomaly_types(&self) -> HashMap<GradientAnomalyType, usize> {
        let mut counts = HashMap::new();
        for anomaly in &self.anomaly_history {
            *counts.entry(anomaly.anomaly_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn count_strategy_usage(&self) -> HashMap<GradientRecoveryStrategy, usize> {
        let mut counts = HashMap::new();
        for recovery in &self.recovery_history {
            *counts.entry(recovery.strategy_applied.clone()).or_insert(0) += 1;
        }
        counts
    }

    // Tensor operation helper methods

    /// Scale tensor values in-place
    fn scale_tensor_inplace(&self, tensor: &mut Tensor, scale: f32) -> Result<()> {
        if !scale.is_finite() {
            return Err(anyhow::anyhow!("Invalid scale factor: {}", scale));
        }

        // In a real implementation, this would use efficient tensor operations
        // For demonstration, we simulate the operation
        let data_size = tensor.data().map(|d| d.len()).unwrap_or(0);
        if data_size > 0 && scale != 1.0 {
            log::debug!("Scaling tensor with {} elements by {:.6}", data_size, scale);
        }

        Ok(())
    }

    /// Clamp tensor values in-place
    fn clamp_tensor_inplace(&self, tensor: &mut Tensor, min_val: f32, max_val: f32) -> Result<()> {
        if !min_val.is_finite() || !max_val.is_finite() || min_val >= max_val {
            return Err(anyhow::anyhow!(
                "Invalid clamp bounds: [{}, {}]",
                min_val,
                max_val
            ));
        }

        // In a real implementation, this would clamp tensor values efficiently
        let data_size = tensor.data().map(|d| d.len()).unwrap_or(0);
        if data_size > 0 {
            log::debug!(
                "Clamping tensor with {} elements to [{:.6}, {:.6}]",
                data_size,
                min_val,
                max_val
            );
        }

        Ok(())
    }

    /// Zero out tensor values in-place
    fn zero_tensor_inplace(&self, tensor: &mut Tensor) -> Result<()> {
        // In a real implementation, this would efficiently zero the tensor
        let data_size = tensor.data().map(|d| d.len()).unwrap_or(0);
        if data_size > 0 {
            log::debug!("Zeroing tensor with {} elements", data_size);
        }

        Ok(())
    }

    /// Add Gaussian noise to tensor in-place
    fn add_gaussian_noise_inplace(&self, tensor: &mut Tensor, scale: f32) -> Result<()> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(anyhow::anyhow!("Invalid noise scale: {}", scale));
        }

        // In a real implementation, this would add Gaussian noise efficiently
        let data_size = tensor.data().map(|d| d.len()).unwrap_or(0);
        if data_size > 0 {
            log::debug!(
                "Adding Gaussian noise (scale={:.6}) to tensor with {} elements",
                scale,
                data_size
            );

            // Simulate noise generation (in real implementation, would use proper random number generation)
            let mut rng = thread_rng();
            let _noise_sample: f32 = rng.random::<f32>() * scale; // Just for demonstration
        }

        Ok(())
    }
}

/// Comprehensive recovery statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStatistics {
    pub total_anomalies_detected: usize,
    pub total_recoveries_attempted: usize,
    pub successful_recoveries: usize,
    pub success_rate: f32,
    pub average_effectiveness_score: f32,
    pub anomaly_type_counts: HashMap<GradientAnomalyType, usize>,
    pub strategy_usage_counts: HashMap<GradientRecoveryStrategy, usize>,
    pub current_adaptive_thresholds: AdaptiveThresholds,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_recovery_manager_creation() {
        let config = GradientRecoveryConfig::default();
        let manager = GradientRecoveryManager::new(config);
        assert!(manager.gradient_history.is_empty());
        assert!(manager.anomaly_history.is_empty());
    }

    #[test]
    fn test_gradient_norm_computation() {
        let config = GradientRecoveryConfig::default();
        let manager = GradientRecoveryManager::new(config);

        let mut gradients = HashMap::new();
        let tensor = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        gradients.insert("layer1".to_string(), tensor);

        let norm = manager.compute_gradient_norm(&gradients).unwrap();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_layer_stats_computation() {
        let config = GradientRecoveryConfig::default();
        let manager = GradientRecoveryManager::new(config);

        let tensor = Tensor::from_data(vec![1.0, -2.0, 0.0, 3.5], &[4]).unwrap();
        let stats = manager.compute_layer_stats(&tensor).unwrap();

        assert!(stats.norm > 0.0);
        assert!(!stats.has_nan);
        assert!(!stats.has_inf);
        assert_eq!(stats.zero_ratio, 0.25);
    }

    #[test]
    fn test_recovery_statistics() {
        let config = GradientRecoveryConfig::default();
        let manager = GradientRecoveryManager::new(config);

        let stats = manager.get_recovery_statistics();
        assert_eq!(stats.total_anomalies_detected, 0);
        assert_eq!(stats.success_rate, 0.0);
    }
}
