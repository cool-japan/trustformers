//! Training Dynamics Analysis Module
//!
//! This module provides comprehensive analysis of training dynamics including:
//! - Loss landscape visualization
//! - Gradient flow analytics
//! - Weight evolution tracking
//! - Training stability monitoring
//! - Convergence analysis

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use trustformers_core::{Result, Tensor};

/// Configuration for training dynamics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDynamicsConfig {
    /// How often to compute dynamics metrics (in steps)
    pub analysis_interval: usize,
    /// Maximum number of historical records to keep
    pub max_history_length: usize,
    /// Enable loss landscape analysis
    pub enable_loss_landscape: bool,
    /// Enable gradient flow analysis
    pub enable_gradient_flow: bool,
    /// Enable weight evolution tracking
    pub enable_weight_evolution: bool,
    /// Enable convergence analysis
    pub enable_convergence_analysis: bool,
    /// Learning rate history smoothing window
    pub smoothing_window: usize,
    /// Gradient clipping threshold for analysis
    pub gradient_clip_threshold: f32,
}

impl Default for TrainingDynamicsConfig {
    fn default() -> Self {
        Self {
            analysis_interval: 100,
            max_history_length: 1000,
            enable_loss_landscape: true,
            enable_gradient_flow: true,
            enable_weight_evolution: true,
            enable_convergence_analysis: true,
            smoothing_window: 10,
            gradient_clip_threshold: 1.0,
        }
    }
}

/// Gradient flow analytics data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlowMetrics {
    /// Step number
    pub step: usize,
    /// Gradient norms by layer
    pub layer_gradient_norms: HashMap<String, f32>,
    /// Average gradient norm
    pub avg_gradient_norm: f32,
    /// Max gradient norm
    pub max_gradient_norm: f32,
    /// Min gradient norm
    pub min_gradient_norm: f32,
    /// Gradient norm variance
    pub gradient_norm_variance: f32,
    /// Number of vanishing gradients (below threshold)
    pub vanishing_gradients: usize,
    /// Number of exploding gradients (above threshold)
    pub exploding_gradients: usize,
    /// Gradient flow direction consistency
    pub flow_consistency: f32,
}

/// Weight evolution tracking data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightEvolutionMetrics {
    /// Step number
    pub step: usize,
    /// Weight norms by layer
    pub layer_weight_norms: HashMap<String, f32>,
    /// Weight change norms by layer
    pub layer_weight_changes: HashMap<String, f32>,
    /// Average weight norm
    pub avg_weight_norm: f32,
    /// Weight update magnitudes
    pub weight_update_magnitudes: HashMap<String, f32>,
    /// Weight stability score (lower = more stable)
    pub weight_stability_score: f32,
    /// Dead neurons count by layer
    pub dead_neurons: HashMap<String, usize>,
    /// Active neurons count by layer
    pub active_neurons: HashMap<String, usize>,
}

/// Loss landscape analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossLandscapeMetrics {
    /// Step number
    pub step: usize,
    /// Current loss value
    pub current_loss: f32,
    /// Loss sharpness (second derivative approximation)
    pub loss_sharpness: f32,
    /// Loss curvature in random directions
    pub loss_curvature: Vec<f32>,
    /// Local minima indicator
    pub local_minima_score: f32,
    /// Flatness measure
    pub flatness_measure: f32,
    /// Barrier height estimation
    pub barrier_height: f32,
}

/// Convergence analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    /// Step number
    pub step: usize,
    /// Loss moving average
    pub loss_moving_avg: f32,
    /// Loss variance over window
    pub loss_variance: f32,
    /// Convergence rate estimate
    pub convergence_rate: f32,
    /// Plateau detection score
    pub plateau_score: f32,
    /// Divergence risk score
    pub divergence_risk: f32,
    /// Oscillation amplitude
    pub oscillation_amplitude: f32,
    /// Progress score (0 = no progress, 1 = good progress)
    pub progress_score: f32,
}

/// Complete training dynamics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDynamicsSnapshot {
    /// Step number
    pub step: usize,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Gradient flow metrics
    pub gradient_flow: Option<GradientFlowMetrics>,
    /// Weight evolution metrics
    pub weight_evolution: Option<WeightEvolutionMetrics>,
    /// Loss landscape metrics
    pub loss_landscape: Option<LossLandscapeMetrics>,
    /// Convergence metrics
    pub convergence: Option<ConvergenceMetrics>,
}

/// Training dynamics analyzer
pub struct TrainingDynamicsAnalyzer {
    config: TrainingDynamicsConfig,
    snapshots: VecDeque<TrainingDynamicsSnapshot>,
    loss_history: VecDeque<f32>,
    gradient_history: VecDeque<HashMap<String, f32>>,
    weight_history: VecDeque<HashMap<String, f32>>,
    last_weights: HashMap<String, Tensor>,
    current_step: usize,
}

impl TrainingDynamicsAnalyzer {
    /// Create a new training dynamics analyzer
    pub fn new(config: TrainingDynamicsConfig) -> Self {
        Self {
            config,
            snapshots: VecDeque::new(),
            loss_history: VecDeque::new(),
            gradient_history: VecDeque::new(),
            weight_history: VecDeque::new(),
            last_weights: HashMap::new(),
            current_step: 0,
        }
    }

    /// Update the analyzer with new training data
    pub fn update(
        &mut self,
        step: usize,
        loss: f32,
        gradients: &HashMap<String, Tensor>,
        weights: &HashMap<String, Tensor>,
    ) -> Result<()> {
        self.current_step = step;

        // Update loss history
        self.loss_history.push_back(loss);
        if self.loss_history.len() > self.config.max_history_length {
            self.loss_history.pop_front();
        }

        // Update gradient history
        let gradient_norms: HashMap<String, f32> = gradients
            .iter()
            .map(|(name, tensor)| {
                let norm = self.compute_tensor_norm(tensor).unwrap_or(0.0);
                (name.clone(), norm)
            })
            .collect();
        self.gradient_history.push_back(gradient_norms);
        if self.gradient_history.len() > self.config.max_history_length {
            self.gradient_history.pop_front();
        }

        // Update weight history
        let weight_norms: HashMap<String, f32> = weights
            .iter()
            .map(|(name, tensor)| {
                let norm = self.compute_tensor_norm(tensor).unwrap_or(0.0);
                (name.clone(), norm)
            })
            .collect();
        self.weight_history.push_back(weight_norms);
        if self.weight_history.len() > self.config.max_history_length {
            self.weight_history.pop_front();
        }

        // Store current weights for next iteration
        self.last_weights = weights.clone();

        // Perform analysis if it's time
        if step % self.config.analysis_interval == 0 {
            self.analyze_current_state(step, loss, gradients, weights)?;
        }

        Ok(())
    }

    /// Perform comprehensive analysis of current training state
    fn analyze_current_state(
        &mut self,
        step: usize,
        loss: f32,
        gradients: &HashMap<String, Tensor>,
        weights: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let mut snapshot = TrainingDynamicsSnapshot {
            step,
            timestamp: chrono::Utc::now(),
            gradient_flow: None,
            weight_evolution: None,
            loss_landscape: None,
            convergence: None,
        };

        // Gradient flow analysis
        if self.config.enable_gradient_flow {
            snapshot.gradient_flow = Some(self.analyze_gradient_flow(step, gradients)?);
        }

        // Weight evolution analysis
        if self.config.enable_weight_evolution {
            snapshot.weight_evolution = Some(self.analyze_weight_evolution(step, weights)?);
        }

        // Loss landscape analysis
        if self.config.enable_loss_landscape {
            snapshot.loss_landscape = Some(self.analyze_loss_landscape(step, loss)?);
        }

        // Convergence analysis
        if self.config.enable_convergence_analysis {
            snapshot.convergence = Some(self.analyze_convergence(step)?);
        }

        // Store snapshot
        self.snapshots.push_back(snapshot);
        if self.snapshots.len() > self.config.max_history_length {
            self.snapshots.pop_front();
        }

        Ok(())
    }

    /// Analyze gradient flow patterns
    fn analyze_gradient_flow(
        &self,
        step: usize,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<GradientFlowMetrics> {
        let mut layer_gradient_norms = HashMap::new();
        let mut gradient_norms = Vec::new();

        for (layer_name, gradient) in gradients {
            let norm = self.compute_tensor_norm(gradient)?;
            layer_gradient_norms.insert(layer_name.clone(), norm);
            gradient_norms.push(norm);
        }

        let avg_gradient_norm = gradient_norms.iter().sum::<f32>() / gradient_norms.len() as f32;
        let max_gradient_norm = gradient_norms.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_gradient_norm = gradient_norms.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        // Compute variance
        let variance = gradient_norms.iter().map(|&x| (x - avg_gradient_norm).powi(2)).sum::<f32>()
            / gradient_norms.len() as f32;

        // Count vanishing and exploding gradients
        let vanishing_threshold = 1e-6;
        let exploding_threshold = self.config.gradient_clip_threshold;

        let vanishing_gradients =
            gradient_norms.iter().filter(|&&x| x < vanishing_threshold).count();
        let exploding_gradients =
            gradient_norms.iter().filter(|&&x| x > exploding_threshold).count();

        // Compute flow consistency (how similar gradients are across layers)
        let flow_consistency = self.compute_gradient_flow_consistency(gradients)?;

        Ok(GradientFlowMetrics {
            step,
            layer_gradient_norms,
            avg_gradient_norm,
            max_gradient_norm,
            min_gradient_norm,
            gradient_norm_variance: variance,
            vanishing_gradients,
            exploding_gradients,
            flow_consistency,
        })
    }

    /// Analyze weight evolution patterns
    fn analyze_weight_evolution(
        &self,
        step: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<WeightEvolutionMetrics> {
        let mut layer_weight_norms = HashMap::new();
        let mut layer_weight_changes = HashMap::new();
        let mut weight_update_magnitudes = HashMap::new();
        let mut dead_neurons = HashMap::new();
        let mut active_neurons = HashMap::new();

        let mut weight_norms = Vec::new();
        let mut weight_changes = Vec::new();

        for (layer_name, weight) in weights {
            let norm = self.compute_tensor_norm(weight)?;
            layer_weight_norms.insert(layer_name.clone(), norm);
            weight_norms.push(norm);

            // Compute weight changes if we have previous weights
            if let Some(prev_weight) = self.last_weights.get(layer_name) {
                let change = self.compute_weight_change(prev_weight, weight)?;
                layer_weight_changes.insert(layer_name.clone(), change);
                weight_changes.push(change);

                // Compute update magnitude
                let update_magnitude = change / norm.max(1e-8);
                weight_update_magnitudes.insert(layer_name.clone(), update_magnitude);
            }

            // Analyze neuron activity
            let (dead, active) = self.analyze_neuron_activity(weight)?;
            dead_neurons.insert(layer_name.clone(), dead);
            active_neurons.insert(layer_name.clone(), active);
        }

        let avg_weight_norm = weight_norms.iter().sum::<f32>() / weight_norms.len() as f32;

        // Compute weight stability score (lower = more stable)
        let weight_stability_score = if !weight_changes.is_empty() {
            weight_changes.iter().sum::<f32>() / weight_changes.len() as f32
        } else {
            0.0
        };

        Ok(WeightEvolutionMetrics {
            step,
            layer_weight_norms,
            layer_weight_changes,
            avg_weight_norm,
            weight_update_magnitudes,
            weight_stability_score,
            dead_neurons,
            active_neurons,
        })
    }

    /// Analyze loss landscape characteristics
    fn analyze_loss_landscape(
        &self,
        step: usize,
        current_loss: f32,
    ) -> Result<LossLandscapeMetrics> {
        // Compute loss sharpness using finite differences
        let loss_sharpness = self.compute_loss_sharpness()?;

        // Compute loss curvature in random directions
        let loss_curvature = self.compute_loss_curvature()?;

        // Compute local minima score
        let local_minima_score = self.compute_local_minima_score()?;

        // Compute flatness measure
        let flatness_measure = self.compute_flatness_measure()?;

        // Estimate barrier height
        let barrier_height = self.estimate_barrier_height()?;

        Ok(LossLandscapeMetrics {
            step,
            current_loss,
            loss_sharpness,
            loss_curvature,
            local_minima_score,
            flatness_measure,
            barrier_height,
        })
    }

    /// Analyze convergence characteristics
    fn analyze_convergence(&self, step: usize) -> Result<ConvergenceMetrics> {
        let window_size = self.config.smoothing_window.min(self.loss_history.len());

        if window_size == 0 {
            return Ok(ConvergenceMetrics {
                step,
                loss_moving_avg: 0.0,
                loss_variance: 0.0,
                convergence_rate: 0.0,
                plateau_score: 0.0,
                divergence_risk: 0.0,
                oscillation_amplitude: 0.0,
                progress_score: 0.0,
            });
        }

        let recent_losses: Vec<f32> =
            self.loss_history.iter().rev().take(window_size).cloned().collect();

        let loss_moving_avg = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;

        let loss_variance =
            recent_losses.iter().map(|&x| (x - loss_moving_avg).powi(2)).sum::<f32>()
                / recent_losses.len() as f32;

        let convergence_rate = self.compute_convergence_rate(&recent_losses)?;
        let plateau_score = self.compute_plateau_score(&recent_losses)?;
        let divergence_risk = self.compute_divergence_risk(&recent_losses)?;
        let oscillation_amplitude = self.compute_oscillation_amplitude(&recent_losses)?;
        let progress_score = self.compute_progress_score(&recent_losses)?;

        Ok(ConvergenceMetrics {
            step,
            loss_moving_avg,
            loss_variance,
            convergence_rate,
            plateau_score,
            divergence_risk,
            oscillation_amplitude,
            progress_score,
        })
    }

    /// Compute tensor norm
    fn compute_tensor_norm(&self, tensor: &Tensor) -> Result<f32> {
        match tensor {
            Tensor::F32(arr) => {
                let norm = arr.iter().map(|&x| x * x).sum::<f32>().sqrt();
                Ok(norm)
            },
            Tensor::F64(arr) => {
                let norm = arr.iter().map(|&x| x * x).sum::<f64>().sqrt() as f32;
                Ok(norm)
            },
            Tensor::F16(arr) => {
                let norm = arr
                    .iter()
                    .map(|&x| {
                        let f32_val = x.to_f32();
                        f32_val * f32_val
                    })
                    .sum::<f32>()
                    .sqrt();
                Ok(norm)
            },
            Tensor::BF16(arr) => {
                let norm = arr
                    .iter()
                    .map(|&x| {
                        let f32_val = x.to_f32();
                        f32_val * f32_val
                    })
                    .sum::<f32>()
                    .sqrt();
                Ok(norm)
            },
            Tensor::I64(_) => Ok(0.0), // Not applicable for integer tensors
            Tensor::C32(arr) => {
                let norm = arr.iter().map(|&x| x.norm_sqr()).sum::<f32>().sqrt();
                Ok(norm)
            },
            Tensor::C64(arr) => {
                let norm = arr.iter().map(|&x| x.norm_sqr() as f32).sum::<f32>().sqrt();
                Ok(norm)
            },
            Tensor::CF16(arr) => {
                let norm = arr
                    .iter()
                    .map(|&x| {
                        let re = x.re.to_f32();
                        let im = x.im.to_f32();
                        re * re + im * im
                    })
                    .sum::<f32>()
                    .sqrt();
                Ok(norm)
            },
            Tensor::CBF16(arr) => {
                let norm = arr
                    .iter()
                    .map(|&x| {
                        let re = x.re.to_f32();
                        let im = x.im.to_f32();
                        re * re + im * im
                    })
                    .sum::<f32>()
                    .sqrt();
                Ok(norm)
            },
            Tensor::Sparse(_) => Ok(1.0), // Default norm for sparse tensors
            #[cfg(feature = "torch")]
            Tensor::Torch(_) => Ok(1.0), // Default norm for Torch tensors
            #[cfg(feature = "candle")]
            Tensor::Candle(_) => Ok(1.0), // Default norm for Candle tensors
        }
    }

    /// Compute weight change between two tensors
    fn compute_weight_change(&self, prev: &Tensor, curr: &Tensor) -> Result<f32> {
        match (prev, curr) {
            (Tensor::F32(prev_arr), Tensor::F32(curr_arr)) => {
                let change = prev_arr
                    .iter()
                    .zip(curr_arr.iter())
                    .map(|(&p, &c)| (p - c).powi(2))
                    .sum::<f32>()
                    .sqrt();
                Ok(change)
            },
            (Tensor::F64(prev_arr), Tensor::F64(curr_arr)) => {
                let change = prev_arr
                    .iter()
                    .zip(curr_arr.iter())
                    .map(|(&p, &c)| (p - c).powi(2))
                    .sum::<f64>()
                    .sqrt() as f32;
                Ok(change)
            },
            _ => Ok(0.0),
        }
    }

    /// Analyze neuron activity (dead vs active neurons)
    fn analyze_neuron_activity(&self, weights: &Tensor) -> Result<(usize, usize)> {
        match weights {
            Tensor::F32(arr) => {
                let threshold = 1e-6;
                let dead = arr.iter().filter(|&&x| x.abs() < threshold).count();
                let active = arr.len() - dead;
                Ok((dead, active))
            },
            Tensor::F64(arr) => {
                let threshold = 1e-6;
                let dead = arr.iter().filter(|&&x| x.abs() < threshold).count();
                let active = arr.len() - dead;
                Ok((dead, active))
            },
            _ => Ok((0, 0)),
        }
    }

    /// Compute gradient flow consistency
    fn compute_gradient_flow_consistency(
        &self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<f32> {
        // Simplified implementation: measure how similar gradient magnitudes are
        let norms: Vec<f32> = gradients
            .values()
            .map(|tensor| self.compute_tensor_norm(tensor).unwrap_or(0.0))
            .collect();

        if norms.len() < 2 {
            return Ok(1.0);
        }

        let mean = norms.iter().sum::<f32>() / norms.len() as f32;
        let variance = norms.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / norms.len() as f32;

        // Consistency score: higher variance = lower consistency
        let consistency = 1.0 / (1.0 + variance);
        Ok(consistency)
    }

    /// Compute loss sharpness (approximation using recent history)
    fn compute_loss_sharpness(&self) -> Result<f32> {
        if self.loss_history.len() < 3 {
            return Ok(0.0);
        }

        let recent: Vec<f32> = self.loss_history.iter().rev().take(3).cloned().collect();
        let second_derivative = recent[0] - 2.0 * recent[1] + recent[2];
        Ok(second_derivative.abs())
    }

    /// Compute loss curvature in random directions
    fn compute_loss_curvature(&self) -> Result<Vec<f32>> {
        // Simplified implementation: use loss history variations
        let window_size = 5.min(self.loss_history.len());
        if window_size < 2 {
            return Ok(vec![0.0]);
        }

        let recent: Vec<f32> = self.loss_history.iter().rev().take(window_size).cloned().collect();
        let mut curvatures = Vec::new();

        for i in 1..recent.len() {
            let curvature = (recent[i - 1] - recent[i]).abs();
            curvatures.push(curvature);
        }

        Ok(curvatures)
    }

    /// Compute local minima score
    fn compute_local_minima_score(&self) -> Result<f32> {
        if self.loss_history.len() < 5 {
            return Ok(0.0);
        }

        let recent: Vec<f32> = self.loss_history.iter().rev().take(5).cloned().collect();
        let current = recent[0];
        let neighbors = &recent[1..];

        // Check if current loss is lower than neighbors
        let is_local_minimum = neighbors.iter().all(|&x| current <= x);
        Ok(if is_local_minimum { 1.0 } else { 0.0 })
    }

    /// Compute flatness measure
    fn compute_flatness_measure(&self) -> Result<f32> {
        if self.loss_history.len() < 3 {
            return Ok(0.0);
        }

        let recent: Vec<f32> = self.loss_history.iter().rev().take(10).cloned().collect();
        let variance = recent
            .iter()
            .map(|&x| {
                let mean = recent.iter().sum::<f32>() / recent.len() as f32;
                (x - mean).powi(2)
            })
            .sum::<f32>()
            / recent.len() as f32;

        // Flatness: lower variance = flatter
        Ok(1.0 / (1.0 + variance))
    }

    /// Estimate barrier height
    fn estimate_barrier_height(&self) -> Result<f32> {
        if self.loss_history.len() < 10 {
            return Ok(0.0);
        }

        let recent: Vec<f32> = self.loss_history.iter().rev().take(10).cloned().collect();
        let min_loss = recent.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_loss = recent.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        Ok(max_loss - min_loss)
    }

    /// Compute convergence rate
    fn compute_convergence_rate(&self, losses: &[f32]) -> Result<f32> {
        if losses.len() < 2 {
            return Ok(0.0);
        }

        let initial = losses[losses.len() - 1];
        let final_loss = losses[0];
        let rate = (initial - final_loss) / losses.len() as f32;
        Ok(rate)
    }

    /// Compute plateau score
    fn compute_plateau_score(&self, losses: &[f32]) -> Result<f32> {
        if losses.len() < 2 {
            return Ok(0.0);
        }

        let variance = losses
            .iter()
            .map(|&x| {
                let mean = losses.iter().sum::<f32>() / losses.len() as f32;
                (x - mean).powi(2)
            })
            .sum::<f32>()
            / losses.len() as f32;

        // Plateau: very low variance indicates plateau
        Ok(1.0 / (1.0 + variance * 100.0))
    }

    /// Compute divergence risk
    fn compute_divergence_risk(&self, losses: &[f32]) -> Result<f32> {
        if losses.len() < 2 {
            return Ok(0.0);
        }

        let trend = losses[0] - losses[losses.len() - 1];
        let risk = if trend > 0.0 { trend } else { 0.0 };
        Ok(risk.tanh()) // Normalize to [0, 1]
    }

    /// Compute oscillation amplitude
    fn compute_oscillation_amplitude(&self, losses: &[f32]) -> Result<f32> {
        if losses.len() < 2 {
            return Ok(0.0);
        }

        let mean = losses.iter().sum::<f32>() / losses.len() as f32;
        let max_deviation = losses.iter().map(|&x| (x - mean).abs()).fold(0.0f32, |a, b| a.max(b));

        Ok(max_deviation)
    }

    /// Compute progress score
    fn compute_progress_score(&self, losses: &[f32]) -> Result<f32> {
        if losses.len() < 2 {
            return Ok(0.0);
        }

        let initial = losses[losses.len() - 1];
        let final_loss = losses[0];
        let improvement = (initial - final_loss) / initial.max(1e-8);
        Ok(improvement.clamp(0.0, 1.0))
    }

    /// Get the latest snapshot
    pub fn get_latest_snapshot(&self) -> Option<&TrainingDynamicsSnapshot> {
        self.snapshots.back()
    }

    /// Get all snapshots
    pub fn get_snapshots(&self) -> &VecDeque<TrainingDynamicsSnapshot> {
        &self.snapshots
    }

    /// Get loss history
    pub fn get_loss_history(&self) -> &VecDeque<f32> {
        &self.loss_history
    }

    /// Generate a comprehensive dynamics report
    pub fn generate_report(&self) -> TrainingDynamicsReport {
        let latest_snapshot = self.get_latest_snapshot();

        TrainingDynamicsReport {
            total_steps: self.current_step,
            total_snapshots: self.snapshots.len(),
            latest_snapshot: latest_snapshot.cloned(),
            loss_trend: self.compute_loss_trend(),
            gradient_health: self.compute_gradient_health(),
            weight_stability: self.compute_weight_stability(),
            convergence_status: self.compute_convergence_status(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Compute loss trend
    fn compute_loss_trend(&self) -> String {
        if self.loss_history.len() < 2 {
            return "Insufficient data".to_string();
        }

        let recent = self.loss_history.iter().rev().take(10).cloned().collect::<Vec<_>>();
        let slope = (recent[0] - recent[recent.len() - 1]) / recent.len() as f32;

        if slope < -0.01 {
            "Decreasing".to_string()
        } else if slope > 0.01 {
            "Increasing".to_string()
        } else {
            "Stable".to_string()
        }
    }

    /// Compute gradient health
    fn compute_gradient_health(&self) -> String {
        if let Some(latest) = self.get_latest_snapshot() {
            if let Some(ref grad_flow) = latest.gradient_flow {
                if grad_flow.exploding_gradients > 0 {
                    "Exploding gradients detected".to_string()
                } else if grad_flow.vanishing_gradients > grad_flow.layer_gradient_norms.len() / 2 {
                    "Vanishing gradients detected".to_string()
                } else {
                    "Healthy".to_string()
                }
            } else {
                "No gradient data".to_string()
            }
        } else {
            "No data".to_string()
        }
    }

    /// Compute weight stability
    fn compute_weight_stability(&self) -> String {
        if let Some(latest) = self.get_latest_snapshot() {
            if let Some(ref weight_evo) = latest.weight_evolution {
                if weight_evo.weight_stability_score > 1.0 {
                    "Unstable".to_string()
                } else if weight_evo.weight_stability_score > 0.1 {
                    "Moderate".to_string()
                } else {
                    "Stable".to_string()
                }
            } else {
                "No weight data".to_string()
            }
        } else {
            "No data".to_string()
        }
    }

    /// Compute convergence status
    fn compute_convergence_status(&self) -> String {
        if let Some(latest) = self.get_latest_snapshot() {
            if let Some(ref convergence) = latest.convergence {
                if convergence.divergence_risk > 0.5 {
                    "Diverging".to_string()
                } else if convergence.plateau_score > 0.8 {
                    "Plateaued".to_string()
                } else if convergence.progress_score > 0.5 {
                    "Converging".to_string()
                } else {
                    "Slow progress".to_string()
                }
            } else {
                "No convergence data".to_string()
            }
        } else {
            "No data".to_string()
        }
    }

    /// Generate recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if let Some(latest) = self.get_latest_snapshot() {
            // Gradient flow recommendations
            if let Some(ref grad_flow) = latest.gradient_flow {
                if grad_flow.exploding_gradients > 0 {
                    recommendations.push(
                        "Consider reducing learning rate or adding gradient clipping".to_string(),
                    );
                }
                if grad_flow.vanishing_gradients > grad_flow.layer_gradient_norms.len() / 2 {
                    recommendations.push(
                        "Consider increasing learning rate or using residual connections"
                            .to_string(),
                    );
                }
            }

            // Weight evolution recommendations
            if let Some(ref weight_evo) = latest.weight_evolution {
                if weight_evo.weight_stability_score > 1.0 {
                    recommendations.push(
                        "Weights are changing rapidly - consider reducing learning rate"
                            .to_string(),
                    );
                }

                let total_dead = weight_evo.dead_neurons.values().sum::<usize>();
                let total_neurons = weight_evo.dead_neurons.values().sum::<usize>()
                    + weight_evo.active_neurons.values().sum::<usize>();

                if total_dead > total_neurons / 2 {
                    recommendations.push("Many dead neurons detected - consider adjusting initialization or activation functions".to_string());
                }
            }

            // Convergence recommendations
            if let Some(ref convergence) = latest.convergence {
                if convergence.plateau_score > 0.8 {
                    recommendations.push("Training has plateaued - consider learning rate scheduling or early stopping".to_string());
                }
                if convergence.divergence_risk > 0.5 {
                    recommendations.push(
                        "Risk of divergence detected - consider reducing learning rate immediately"
                            .to_string(),
                    );
                }
            }
        }

        if recommendations.is_empty() {
            recommendations.push("Training appears to be progressing normally".to_string());
        }

        recommendations
    }
}

/// Training dynamics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDynamicsReport {
    pub total_steps: usize,
    pub total_snapshots: usize,
    pub latest_snapshot: Option<TrainingDynamicsSnapshot>,
    pub loss_trend: String,
    pub gradient_health: String,
    pub weight_stability: String,
    pub convergence_status: String,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_dynamics_config_default() {
        let config = TrainingDynamicsConfig::default();
        assert_eq!(config.analysis_interval, 100);
        assert_eq!(config.max_history_length, 1000);
        assert!(config.enable_loss_landscape);
        assert!(config.enable_gradient_flow);
        assert!(config.enable_weight_evolution);
        assert!(config.enable_convergence_analysis);
    }

    #[test]
    fn test_analyzer_creation() {
        let config = TrainingDynamicsConfig::default();
        let analyzer = TrainingDynamicsAnalyzer::new(config);
        assert_eq!(analyzer.current_step, 0);
        assert!(analyzer.snapshots.is_empty());
        assert!(analyzer.loss_history.is_empty());
    }

    #[test]
    fn test_loss_history_update() {
        let config = TrainingDynamicsConfig::default();
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        let gradients = HashMap::new();
        let weights = HashMap::new();

        analyzer.update(1, 1.0, &gradients, &weights).unwrap();
        assert_eq!(analyzer.loss_history.len(), 1);
        assert_eq!(analyzer.loss_history[0], 1.0);
    }

    #[test]
    fn test_tensor_norm_computation() {
        let config = TrainingDynamicsConfig::default();
        let analyzer = TrainingDynamicsAnalyzer::new(config);

        let tensor = Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap();
        let norm = analyzer.compute_tensor_norm(&tensor).unwrap();
        assert!((norm - 5.0).abs() < 1e-6); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_neuron_activity_analysis() {
        let config = TrainingDynamicsConfig::default();
        let analyzer = TrainingDynamicsAnalyzer::new(config);

        let weights = Tensor::from_vec(vec![0.0, 1e-8, 0.1, 0.5], &[4]).unwrap();
        let (dead, active) = analyzer.analyze_neuron_activity(&weights).unwrap();
        assert_eq!(dead, 2); // 0.0 and 1e-8 are below threshold
        assert_eq!(active, 2); // 0.1 and 0.5 are above threshold
    }

    #[test]
    fn test_weight_change_computation() {
        let config = TrainingDynamicsConfig::default();
        let analyzer = TrainingDynamicsAnalyzer::new(config);

        let prev = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let curr = Tensor::from_vec(vec![1.1, 2.1], &[2]).unwrap();
        let change = analyzer.compute_weight_change(&prev, &curr).unwrap();

        // sqrt((0.1)^2 + (0.1)^2) = sqrt(0.02) ≈ 0.1414
        assert!((change - 0.1414).abs() < 1e-3);
    }

    #[test]
    fn test_report_generation() {
        let config = TrainingDynamicsConfig::default();
        let analyzer = TrainingDynamicsAnalyzer::new(config);

        let report = analyzer.generate_report();
        assert_eq!(report.total_steps, 0);
        assert_eq!(report.total_snapshots, 0);
        assert!(report.latest_snapshot.is_none());
        assert!(!report.recommendations.is_empty());
    }
}
