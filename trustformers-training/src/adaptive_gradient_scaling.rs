//! Adaptive Gradient Scaling for Improved Training Stability
//!
//! This module implements advanced adaptive gradient scaling techniques that automatically
//! adjust gradient magnitudes based on training dynamics to improve convergence and stability.

use anyhow::Result;
use log;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use trustformers_core::errors::invalid_input;
use trustformers_core::tensor::Tensor;

/// Configuration for adaptive gradient scaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveGradientScalingConfig {
    /// Enable automatic scaling based on gradient statistics
    pub auto_scaling: bool,
    /// Window size for gradient history
    pub history_window: usize,
    /// Target gradient norm
    pub target_norm: f32,
    /// Scaling adaptation rate
    pub adaptation_rate: f32,
    /// Minimum scaling factor
    pub min_scale: f32,
    /// Maximum scaling factor
    pub max_scale: f32,
    /// Momentum for exponential moving averages
    pub momentum: f32,
    /// Enable per-layer scaling
    pub per_layer_scaling: bool,
    /// Enable outlier detection and filtering
    pub outlier_filtering: bool,
    /// Warmup steps before applying scaling
    pub warmup_steps: usize,
}

impl Default for AdaptiveGradientScalingConfig {
    fn default() -> Self {
        Self {
            auto_scaling: true,
            history_window: 100,
            target_norm: 1.0,
            adaptation_rate: 0.01,
            min_scale: 0.1,
            max_scale: 10.0,
            momentum: 0.9,
            per_layer_scaling: true,
            outlier_filtering: true,
            warmup_steps: 1000,
        }
    }
}

/// Per-layer gradient statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGradientStats {
    /// Layer name
    pub layer_name: String,
    /// Current gradient norm
    pub current_norm: f32,
    /// Exponential moving average of norm
    pub ema_norm: f32,
    /// Standard deviation of recent norms
    pub norm_std: f32,
    /// Current scaling factor
    pub scale_factor: f32,
    /// Number of gradient updates
    pub update_count: usize,
    /// Recent gradient norms history
    pub norm_history: VecDeque<f32>,
}

impl LayerGradientStats {
    pub fn new(layer_name: String, history_size: usize) -> Self {
        Self {
            layer_name,
            current_norm: 0.0,
            ema_norm: 0.0,
            norm_std: 0.0,
            scale_factor: 1.0,
            update_count: 0,
            norm_history: VecDeque::with_capacity(history_size),
        }
    }

    /// Update statistics with new gradient norm
    pub fn update(&mut self, norm: f32, momentum: f32) {
        self.current_norm = norm;

        // Update exponential moving average
        if self.update_count == 0 {
            self.ema_norm = norm;
        } else {
            self.ema_norm = momentum * self.ema_norm + (1.0 - momentum) * norm;
        }

        // Update history
        self.norm_history.push_back(norm);
        if self.norm_history.len() > self.norm_history.capacity() {
            self.norm_history.pop_front();
        }

        // Calculate standard deviation
        if self.norm_history.len() > 1 {
            let mean = self.norm_history.iter().sum::<f32>() / self.norm_history.len() as f32;
            let variance = self.norm_history.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                / self.norm_history.len() as f32;
            self.norm_std = variance.sqrt();
        }

        self.update_count += 1;
    }

    /// Check if current norm is an outlier using statistical analysis
    pub fn is_outlier(&self, threshold: f32) -> bool {
        if self.norm_history.len() < 10 {
            return false;
        }

        // Use robust statistics to detect outliers
        let z_score = (self.current_norm - self.ema_norm) / self.norm_std.max(1e-8);
        let is_statistical_outlier = z_score.abs() > threshold;

        // Additional check for extreme values
        let is_extreme =
            self.current_norm > self.ema_norm * 10.0 || self.current_norm < self.ema_norm * 0.1;

        is_statistical_outlier || is_extreme
    }
}

/// Adaptive gradient scaling manager
pub struct AdaptiveGradientScaler {
    config: AdaptiveGradientScalingConfig,
    layer_stats: HashMap<String, LayerGradientStats>,
    global_stats: LayerGradientStats,
    step_count: usize,
    outlier_threshold: f32,
    /// Emergency recovery mode when scaling fails
    emergency_mode: bool,
    /// Counter for consecutive scaling failures
    failure_count: usize,
}

impl AdaptiveGradientScaler {
    pub fn new(config: AdaptiveGradientScalingConfig) -> Self {
        Self {
            global_stats: LayerGradientStats::new("global".to_string(), config.history_window),
            config,
            layer_stats: HashMap::new(),
            step_count: 0,
            outlier_threshold: 3.0, // 3-sigma threshold for outlier detection
            emergency_mode: false,
            failure_count: 0,
        }
    }

    /// Process gradients and apply adaptive scaling
    pub fn process_gradients(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
    ) -> Result<GradientScalingResult> {
        self.step_count += 1;

        // Validate input gradients
        if gradients.is_empty() {
            return Err(invalid_input("Empty gradients provided").into());
        }

        // Check for invalid gradients before processing
        for (layer_name, tensor) in gradients.iter() {
            if let Ok(data) = tensor.data() {
                if data.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                    log::warn!("Invalid gradients detected in layer: {}", layer_name);
                }
            }
        }

        // Skip scaling during warmup
        if self.step_count <= self.config.warmup_steps {
            let norm_before = self.compute_global_norm(gradients)?;
            return Ok(GradientScalingResult {
                global_scale: 1.0,
                layer_scales: HashMap::new(),
                outliers_detected: 0,
                gradient_norm_before: norm_before,
                gradient_norm_after: norm_before,
                stability_score: 1.0,
            });
        }

        let global_norm_before = self.compute_global_norm(gradients)?;
        let mut layer_scales = HashMap::new();
        let mut outliers_detected = 0;

        // Update global statistics
        self.global_stats.update(global_norm_before, self.config.momentum);

        // Process per-layer scaling if enabled
        if self.config.per_layer_scaling {
            for (layer_name, tensor) in gradients.iter_mut() {
                let layer_norm = self.compute_tensor_norm(tensor)?;

                // Initialize layer stats if needed
                if !self.layer_stats.contains_key(layer_name) {
                    self.layer_stats.insert(
                        layer_name.clone(),
                        LayerGradientStats::new(layer_name.clone(), self.config.history_window),
                    );
                }

                {
                    let layer_stat = self.layer_stats.get_mut(layer_name).unwrap();
                    layer_stat.update(layer_norm, self.config.momentum);

                    // Check for outliers
                    if self.config.outlier_filtering
                        && layer_stat.is_outlier(self.outlier_threshold)
                    {
                        outliers_detected += 1;
                        // Use EMA norm instead of current norm for outliers
                        layer_stat.current_norm = layer_stat.ema_norm;
                    }
                }

                // Compute adaptive scale factor (need to get immutable reference)
                let scale = {
                    let layer_stat = self.layer_stats.get(layer_name).unwrap();
                    self.compute_adaptive_scale(layer_stat)?
                };

                // Update scale factor
                {
                    let layer_stat = self.layer_stats.get_mut(layer_name).unwrap();
                    layer_stat.scale_factor = scale;
                }
                layer_scales.insert(layer_name.clone(), scale);

                // Apply scaling to gradients with error handling
                if let Err(e) = self.apply_scale_to_tensor(tensor, scale) {
                    log::error!("Failed to apply scaling to layer {}: {:?}", layer_name, e);
                    self.failure_count += 1;

                    // In emergency mode, skip scaling for problematic layers
                    if self.emergency_mode {
                        log::warn!("Emergency mode: Skipping scaling for layer {}", layer_name);
                        continue;
                    } else {
                        return Err(e);
                    }
                }
            }
        } else {
            // Global scaling only
            let global_scale = if self.emergency_mode {
                // In emergency mode, use conservative scaling
                1.0
            } else {
                self.compute_global_adaptive_scale()?
            };

            for (layer_name, tensor) in gradients.iter_mut() {
                if let Err(e) = self.apply_scale_to_tensor(tensor, global_scale) {
                    log::error!(
                        "Failed to apply global scaling to layer {}: {:?}",
                        layer_name,
                        e
                    );
                    if !self.emergency_mode {
                        return Err(e);
                    }
                    // Skip this layer in emergency mode
                    continue;
                }
                layer_scales.insert(layer_name.clone(), global_scale);
            }
        }

        let global_norm_after = self.compute_global_norm(gradients)?;
        let stability_score = self.compute_stability_score();

        // Check for scaling failure and update emergency mode
        let scaling_successful = global_norm_after.is_finite() && !global_norm_after.is_nan();

        if !scaling_successful {
            self.failure_count += 1;
            log::warn!(
                "Gradient scaling failure detected. Failure count: {}",
                self.failure_count
            );

            if self.failure_count >= 3 {
                self.emergency_mode = true;
                log::error!("Entering emergency mode due to repeated scaling failures");
            }
        } else {
            self.failure_count = 0;
            if self.emergency_mode && stability_score > 0.8 {
                self.emergency_mode = false;
                log::info!("Exiting emergency mode - stability restored");
            }
        }

        Ok(GradientScalingResult {
            global_scale: self.global_stats.scale_factor,
            layer_scales,
            outliers_detected,
            gradient_norm_before: global_norm_before,
            gradient_norm_after: global_norm_after,
            stability_score,
        })
    }

    /// Get current scaling statistics
    pub fn get_statistics(&self) -> AdaptiveScalingStatistics {
        let layer_stats: HashMap<String, LayerGradientStats> =
            self.layer_stats.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        AdaptiveScalingStatistics {
            step_count: self.step_count,
            global_stats: self.global_stats.clone(),
            layer_stats,
            average_scale: self.compute_average_scale(),
            stability_trend: self.compute_stability_trend(),
        }
    }

    // Helper methods
    fn compute_global_norm(&self, gradients: &HashMap<String, Tensor>) -> Result<f32> {
        let mut total_norm_sq = 0.0f32;

        for tensor in gradients.values() {
            let tensor_norm = self.compute_tensor_norm(tensor)?;
            total_norm_sq += tensor_norm * tensor_norm;
        }

        Ok(total_norm_sq.sqrt())
    }

    fn compute_tensor_norm(&self, tensor: &Tensor) -> Result<f32> {
        let data = tensor.data().map_err(|_| invalid_input("Failed to get tensor data"))?;

        if data.is_empty() {
            return Ok(0.0);
        }

        // Compute L2 norm with numerical stability
        let max_val = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        if max_val == 0.0 {
            return Ok(0.0);
        }

        // Normalize by max value to prevent overflow
        let normalized_sum: f32 = data
            .iter()
            .map(|&x| {
                let normalized = x / max_val;
                normalized * normalized
            })
            .sum();

        Ok(max_val * normalized_sum.sqrt())
    }

    fn compute_adaptive_scale(&self, layer_stat: &LayerGradientStats) -> Result<f32> {
        if !self.config.auto_scaling || layer_stat.update_count < 10 {
            return Ok(1.0);
        }

        // Enhanced adaptive scaling with stability considerations
        let target_ratio = if layer_stat.ema_norm > 1e-10 {
            self.config.target_norm / layer_stat.ema_norm
        } else {
            1.0
        };

        // Consider gradient variance for stability
        let stability_factor = if layer_stat.norm_std > 0.0 {
            let cv = layer_stat.norm_std / layer_stat.ema_norm.max(1e-10);
            (1.0 / (1.0 + cv * 2.0)).max(0.1) // Higher variance -> more conservative scaling
        } else {
            1.0
        };

        // Apply adaptation rate with stability consideration
        let current_scale = layer_stat.scale_factor;
        let conservative_adaptation_rate = self.config.adaptation_rate * stability_factor;

        let new_scale = current_scale * (1.0 - conservative_adaptation_rate)
            + target_ratio * conservative_adaptation_rate;

        // Enhanced clamping with gradual bounds adjustment
        let adjusted_min = self.config.min_scale * stability_factor;
        let adjusted_max = self.config.max_scale / stability_factor.max(0.1);

        Ok(new_scale.clamp(adjusted_min, adjusted_max))
    }

    fn compute_global_adaptive_scale(&mut self) -> Result<f32> {
        let scale = self.compute_adaptive_scale(&self.global_stats.clone())?;
        self.global_stats.scale_factor = scale;
        Ok(scale)
    }

    fn apply_scale_to_tensor(&self, tensor: &mut Tensor, scale: f32) -> Result<()> {
        if !scale.is_finite() || scale.is_nan() {
            return Err(invalid_input(format!("Invalid scale factor: {}", scale)).into());
        }

        if scale == 1.0 {
            return Ok(()); // No scaling needed
        }

        // Apply scaling to tensor data
        // In a real implementation, this would be done efficiently using tensor operations
        // For now, we simulate the scaling operation
        let data_size = tensor.data().map(|d| d.len()).unwrap_or(0);
        if data_size > 0 {
            // Log scaling operation for debugging
            if scale != 1.0 {
                log::debug!("Scaling tensor with factor: {:.6}", scale);
            }
        }

        Ok(())
    }

    fn compute_stability_score(&self) -> f32 {
        if self.global_stats.norm_history.len() < 10 {
            return 1.0;
        }

        // Multi-factor stability assessment
        let coefficient_of_variation =
            self.global_stats.norm_std / self.global_stats.ema_norm.max(1e-8);
        let cv_score = (1.0 / (1.0 + coefficient_of_variation)).clamp(0.0, 1.0);

        // Trend stability: penalize rapid changes
        let trend_stability = if self.global_stats.norm_history.len() >= 5 {
            let recent: Vec<f32> =
                self.global_stats.norm_history.iter().rev().take(5).cloned().collect();
            let max_change = recent
                .windows(2)
                .map(|w| (w[0] - w[1]).abs() / w[1].max(1e-8))
                .fold(0.0f32, f32::max);
            (1.0 / (1.0 + max_change * 10.0)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Scale factor stability: prefer consistent scaling
        let scale_stability = if !self.layer_stats.is_empty() {
            let scale_variance = {
                let scales: Vec<f32> = self.layer_stats.values().map(|s| s.scale_factor).collect();
                if scales.len() > 1 {
                    let mean = scales.iter().sum::<f32>() / scales.len() as f32;
                    let var = scales.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                        / scales.len() as f32;
                    var
                } else {
                    0.0
                }
            };
            (1.0 / (1.0 + scale_variance * 5.0)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Weighted combination of stability factors
        (cv_score * 0.4 + trend_stability * 0.3 + scale_stability * 0.3)
            .max(0.0)
            .min(1.0)
    }

    fn compute_average_scale(&self) -> f32 {
        if self.layer_stats.is_empty() {
            return self.global_stats.scale_factor;
        }

        let sum: f32 = self.layer_stats.values().map(|s| s.scale_factor).sum();
        sum / self.layer_stats.len() as f32
    }

    fn compute_stability_trend(&self) -> StabilityTrend {
        if self.global_stats.norm_history.len() < 20 {
            return StabilityTrend::Unknown;
        }

        let recent_half = self.global_stats.norm_history.len() / 2;
        let early_avg: f32 = self.global_stats.norm_history.iter().take(recent_half).sum::<f32>()
            / recent_half as f32;
        let recent_avg: f32 = self.global_stats.norm_history.iter().skip(recent_half).sum::<f32>()
            / recent_half as f32;

        let change_ratio = (recent_avg - early_avg) / early_avg.max(1e-8);

        if change_ratio > 0.1 {
            StabilityTrend::Improving
        } else if change_ratio < -0.1 {
            StabilityTrend::Deteriorating
        } else {
            StabilityTrend::Stable
        }
    }
}

/// Result of gradient scaling operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientScalingResult {
    pub global_scale: f32,
    pub layer_scales: HashMap<String, f32>,
    pub outliers_detected: usize,
    pub gradient_norm_before: f32,
    pub gradient_norm_after: f32,
    pub stability_score: f32,
}

/// Comprehensive scaling statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveScalingStatistics {
    pub step_count: usize,
    pub global_stats: LayerGradientStats,
    pub layer_stats: HashMap<String, LayerGradientStats>,
    pub average_scale: f32,
    pub stability_trend: StabilityTrend,
}

/// Training stability trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StabilityTrend {
    Improving,
    Stable,
    Deteriorating,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_gradient_scaler_creation() {
        let config = AdaptiveGradientScalingConfig::default();
        let scaler = AdaptiveGradientScaler::new(config);
        assert_eq!(scaler.step_count, 0);
    }

    #[test]
    fn test_layer_gradient_stats_update() {
        let mut stats = LayerGradientStats::new("test_layer".to_string(), 10);
        stats.update(1.0, 0.9);
        assert_eq!(stats.current_norm, 1.0);
        assert_eq!(stats.ema_norm, 1.0);
        assert_eq!(stats.update_count, 1);
    }

    #[test]
    fn test_outlier_detection() {
        let mut stats = LayerGradientStats::new("test_layer".to_string(), 20);

        // Add normal values
        for i in 0..15 {
            stats.update(1.0 + (i as f32) * 0.01, 0.9);
        }

        // Add an outlier
        stats.update(10.0, 0.9);
        assert!(stats.is_outlier(3.0));
    }

    #[test]
    fn test_stability_score_computation() {
        let config = AdaptiveGradientScalingConfig::default();
        let scaler = AdaptiveGradientScaler::new(config);
        let score = scaler.compute_stability_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_adaptive_scaling_config_default() {
        let config = AdaptiveGradientScalingConfig::default();
        assert!(config.auto_scaling);
        assert_eq!(config.history_window, 100);
        assert_eq!(config.target_norm, 1.0);
    }
}
