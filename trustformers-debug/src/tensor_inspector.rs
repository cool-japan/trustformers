//! Tensor inspection and analysis tools

use anyhow::Result;
use scirs2_core::ndarray::*; // SciRS2 Integration Policy - was: use ndarray::{Array, ArrayD, Axis, IxDyn};
use scirs2_core::random::random; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::Instant;
use uuid::Uuid;

use crate::DebugConfig;

/// Statistics about a tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStats {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub total_elements: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub l1_norm: f64,
    pub l2_norm: f64,
    pub infinity_norm: f64,
    pub nan_count: usize,
    pub inf_count: usize,
    pub zero_count: usize,
    pub memory_usage_bytes: usize,
    pub sparsity: f64,
}

/// Distribution analysis of tensor values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDistribution {
    pub histogram: Vec<(f64, usize)>,
    pub percentiles: HashMap<String, f64>,
    pub outliers: Vec<f64>,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Tensor metadata and tracking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub id: Uuid,
    pub name: String,
    pub layer_name: Option<String>,
    pub operation: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub stats: TensorStats,
    pub distribution: Option<TensorDistribution>,
    pub gradient_stats: Option<TensorStats>,
}

/// Comparison between two tensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorComparison {
    pub tensor1_id: Uuid,
    pub tensor2_id: Uuid,
    pub mse: f64,
    pub mae: f64,
    pub max_diff: f64,
    pub cosine_similarity: f64,
    pub correlation: f64,
    pub shape_match: bool,
    pub dtype_match: bool,
}

/// Real-time tensor monitoring data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorTimeSeries {
    pub tensor_id: Uuid,
    pub timestamps: VecDeque<chrono::DateTime<chrono::Utc>>,
    pub values: VecDeque<TensorStats>,
    pub max_history: usize,
}

/// Tensor dependency tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDependency {
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub operation: String,
    pub weight: f64,
}

/// Tensor lifecycle event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorLifecycleEvent {
    Created { size_bytes: usize },
    Modified { operation: String },
    Accessed { access_type: String },
    Destroyed,
}

/// Tensor lifecycle tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorLifecycle {
    pub tensor_id: Uuid,
    pub events: Vec<(chrono::DateTime<chrono::Utc>, TensorLifecycleEvent)>,
    pub total_accesses: usize,
    pub creation_time: chrono::DateTime<chrono::Utc>,
}

/// Advanced tensor analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedTensorAnalysis {
    pub spectral_analysis: Option<SpectralAnalysis>,
    pub information_content: InformationContent,
    pub stability_metrics: StabilityMetrics,
    pub relationship_analysis: RelationshipAnalysis,
}

/// Spectral analysis of tensor values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalysis {
    pub eigenvalues: Vec<f64>,
    pub condition_number: f64,
    pub rank: usize,
    pub spectral_norm: f64,
}

/// Information content metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationContent {
    pub entropy: f64,
    pub mutual_information: f64,
    pub effective_rank: f64,
    pub compression_ratio: f64,
}

/// Stability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    pub numerical_stability: f64,
    pub gradient_stability: f64,
    pub perturbation_sensitivity: f64,
    pub robustness_score: f64,
}

/// Relationship analysis between tensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipAnalysis {
    pub cross_correlations: HashMap<Uuid, f64>,
    pub dependency_strength: HashMap<Uuid, f64>,
    pub causal_relationships: Vec<TensorDependency>,
}

/// Enhanced tensor inspector for detailed analysis
#[derive(Debug)]
pub struct TensorInspector {
    config: DebugConfig,
    tracked_tensors: HashMap<Uuid, TensorInfo>,
    comparisons: Vec<TensorComparison>,
    alerts: Vec<TensorAlert>,
    // New advanced features
    time_series: HashMap<Uuid, TensorTimeSeries>,
    dependencies: Vec<TensorDependency>,
    lifecycles: HashMap<Uuid, TensorLifecycle>,
    monitoring_enabled: bool,
    last_analysis_time: Option<Instant>,
}

impl TensorInspector {
    /// Create a new tensor inspector
    pub fn new(config: &DebugConfig) -> Self {
        Self {
            config: config.clone(),
            tracked_tensors: HashMap::new(),
            comparisons: Vec::new(),
            alerts: Vec::new(),
            // Initialize new advanced features
            time_series: HashMap::new(),
            dependencies: Vec::new(),
            lifecycles: HashMap::new(),
            monitoring_enabled: false,
            last_analysis_time: None,
        }
    }

    /// Start the tensor inspector
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting tensor inspector");
        Ok(())
    }

    /// Inspect a tensor and return detailed analysis
    pub fn inspect_tensor<T>(
        &mut self,
        tensor: &ArrayD<T>,
        name: &str,
        layer_name: Option<&str>,
        operation: Option<&str>,
    ) -> Result<Uuid>
    where
        T: Clone + Into<f64> + fmt::Debug + 'static,
    {
        let id = Uuid::new_v4();

        // Convert to f64 for analysis
        let values: Vec<f64> = tensor.iter().map(|x| x.clone().into()).collect();
        let shape = tensor.shape().to_vec();

        let stats = self.compute_tensor_stats(&values, &shape, std::mem::size_of::<T>())?;
        let distribution = if self.should_compute_distribution() {
            Some(self.compute_distribution(&values)?)
        } else {
            None
        };

        let tensor_info = TensorInfo {
            id,
            name: name.to_string(),
            layer_name: layer_name.map(|s| s.to_string()),
            operation: operation.map(|s| s.to_string()),
            timestamp: chrono::Utc::now(),
            stats,
            distribution,
            gradient_stats: None,
        };

        // Check for alerts
        self.check_tensor_alerts(&tensor_info)?;

        // Store tensor info if we have space
        if self.tracked_tensors.len() < self.config.max_tracked_tensors {
            self.tracked_tensors.insert(id, tensor_info.clone());
        }

        // Advanced features integration
        self.record_lifecycle_event(
            id,
            TensorLifecycleEvent::Created {
                size_bytes: std::mem::size_of::<T>() * tensor.len(),
            },
        );

        if self.monitoring_enabled {
            if let Some(tensor_info) = self.tracked_tensors.get(&id) {
                self.update_time_series(id, tensor_info.stats.clone());
            }
        }

        Ok(id)
    }

    /// Inspect tensor gradients
    pub fn inspect_gradients<T>(&mut self, tensor_id: Uuid, gradients: &ArrayD<T>) -> Result<()>
    where
        T: Clone + Into<f64> + fmt::Debug + 'static,
    {
        let values: Vec<f64> = gradients.iter().map(|x| x.clone().into()).collect();
        let shape = gradients.shape().to_vec();

        let gradient_stats =
            self.compute_tensor_stats(&values, &shape, std::mem::size_of::<T>())?;

        if let Some(tensor_info) = self.tracked_tensors.get_mut(&tensor_id) {
            tensor_info.gradient_stats = Some(gradient_stats);
        }

        // Check for gradient-specific alerts - get the data we need first
        let tensor_info_for_alerts = self
            .tracked_tensors
            .get(&tensor_id)
            .map(|info| (info.id, info.name.clone(), info.gradient_stats.clone()));

        if let Some((id, name, grad_stats)) = tensor_info_for_alerts {
            self.check_gradient_alerts_with_data(id, &name, grad_stats)?;
        }

        Ok(())
    }

    /// Compare two tensors
    pub fn compare_tensors(&mut self, id1: Uuid, id2: Uuid) -> Result<TensorComparison> {
        let tensor1 = self
            .tracked_tensors
            .get(&id1)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", id1))?;
        let tensor2 = self
            .tracked_tensors
            .get(&id2)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", id2))?;

        let comparison = TensorComparison {
            tensor1_id: id1,
            tensor2_id: id2,
            mse: self.compute_mse(&tensor1.stats, &tensor2.stats),
            mae: self.compute_mae(&tensor1.stats, &tensor2.stats),
            max_diff: (tensor1.stats.max - tensor2.stats.max).abs(),
            cosine_similarity: self.compute_cosine_similarity(&tensor1.stats, &tensor2.stats),
            correlation: self.compute_correlation(&tensor1.stats, &tensor2.stats),
            shape_match: tensor1.stats.shape == tensor2.stats.shape,
            dtype_match: tensor1.stats.dtype == tensor2.stats.dtype,
        };

        self.comparisons.push(comparison.clone());
        Ok(comparison)
    }

    /// Get tensor information by ID
    pub fn get_tensor_info(&self, id: Uuid) -> Option<&TensorInfo> {
        self.tracked_tensors.get(&id)
    }

    /// Get all tracked tensors
    pub fn get_all_tensors(&self) -> Vec<&TensorInfo> {
        self.tracked_tensors.values().collect()
    }

    /// Get tensors by layer name
    pub fn get_tensors_by_layer(&self, layer_name: &str) -> Vec<&TensorInfo> {
        self.tracked_tensors
            .values()
            .filter(|info| info.layer_name.as_ref() == Some(&layer_name.to_string()))
            .collect()
    }

    /// Get all alerts
    pub fn get_alerts(&self) -> &[TensorAlert] {
        &self.alerts
    }

    /// Clear tracking data
    pub fn clear(&mut self) {
        self.tracked_tensors.clear();
        self.comparisons.clear();
        self.alerts.clear();
        // Clear new advanced features
        self.time_series.clear();
        self.dependencies.clear();
        self.lifecycles.clear();
        self.last_analysis_time = None;
    }

    /// Generate inspection report
    pub async fn generate_report(&self) -> Result<TensorInspectionReport> {
        let total_tensors = self.tracked_tensors.len();
        let tensors_with_issues = self
            .tracked_tensors
            .values()
            .filter(|info| info.stats.nan_count > 0 || info.stats.inf_count > 0)
            .count();

        let memory_usage =
            self.tracked_tensors.values().map(|info| info.stats.memory_usage_bytes).sum();

        Ok(TensorInspectionReport {
            total_tensors,
            tensors_with_issues,
            total_memory_usage: memory_usage,
            alerts: self.alerts.clone(),
            comparisons: self.comparisons.clone(),
            summary_stats: self.compute_summary_stats(),
        })
    }

    // Advanced debugging features

    /// Enable real-time tensor monitoring
    pub fn enable_monitoring(&mut self, enable: bool) {
        self.monitoring_enabled = enable;
        if enable {
            tracing::info!("Real-time tensor monitoring enabled");
        } else {
            tracing::info!("Real-time tensor monitoring disabled");
        }
    }

    /// Track tensor dependency
    pub fn track_dependency(
        &mut self,
        source_id: Uuid,
        target_id: Uuid,
        operation: &str,
        weight: f64,
    ) {
        let dependency = TensorDependency {
            source_id,
            target_id,
            operation: operation.to_string(),
            weight,
        };
        self.dependencies.push(dependency);
    }

    /// Record tensor lifecycle event
    pub fn record_lifecycle_event(&mut self, tensor_id: Uuid, event: TensorLifecycleEvent) {
        let lifecycle = self.lifecycles.entry(tensor_id).or_insert_with(|| TensorLifecycle {
            tensor_id,
            events: Vec::new(),
            total_accesses: 0,
            creation_time: chrono::Utc::now(),
        });

        lifecycle.events.push((chrono::Utc::now(), event.clone()));

        if matches!(event, TensorLifecycleEvent::Accessed { .. }) {
            lifecycle.total_accesses += 1;
        }
    }

    /// Update tensor time series data
    pub fn update_time_series(&mut self, tensor_id: Uuid, stats: TensorStats) {
        if !self.monitoring_enabled {
            return;
        }

        let time_series = self.time_series.entry(tensor_id).or_insert_with(|| TensorTimeSeries {
            tensor_id,
            timestamps: VecDeque::new(),
            values: VecDeque::new(),
            max_history: 1000, // Default history size
        });

        time_series.timestamps.push_back(chrono::Utc::now());
        time_series.values.push_back(stats);

        // Maintain maximum history size
        while time_series.timestamps.len() > time_series.max_history {
            time_series.timestamps.pop_front();
            time_series.values.pop_front();
        }
    }

    /// Perform advanced tensor analysis
    pub fn perform_advanced_analysis<T>(&self, tensor: &ArrayD<T>) -> Result<AdvancedTensorAnalysis>
    where
        T: Clone + Into<f64> + fmt::Debug + 'static,
    {
        let values: Vec<f64> = tensor.iter().map(|x| x.clone().into()).collect();

        Ok(AdvancedTensorAnalysis {
            spectral_analysis: self.compute_spectral_analysis(&values, tensor.shape())?,
            information_content: self.compute_information_content(&values)?,
            stability_metrics: self.compute_stability_metrics(&values)?,
            relationship_analysis: self.compute_relationship_analysis(&values)?,
        })
    }

    /// Detect tensor anomalies using advanced algorithms
    pub fn detect_advanced_anomalies(&self, tensor_id: Uuid) -> Result<Vec<TensorAlert>> {
        let mut alerts = Vec::new();

        if let Some(time_series) = self.time_series.get(&tensor_id) {
            // Detect drift in tensor statistics over time
            if time_series.values.len() >= 10 {
                let recent_mean =
                    time_series.values.iter().rev().take(5).map(|stats| stats.mean).sum::<f64>()
                        / 5.0;
                let historical_mean =
                    time_series.values.iter().take(5).map(|stats| stats.mean).sum::<f64>() / 5.0;

                let drift_ratio =
                    (recent_mean - historical_mean).abs() / historical_mean.abs().max(1e-8);

                if drift_ratio > 0.5 {
                    if let Some(tensor_info) = self.tracked_tensors.get(&tensor_id) {
                        alerts.push(TensorAlert {
                            id: Uuid::new_v4(),
                            tensor_id,
                            tensor_name: tensor_info.name.clone(),
                            alert_type: TensorAlertType::ExtremeValues,
                            severity: AlertSeverity::Warning,
                            message: format!(
                                "Detected statistical drift in tensor '{}': {:.2}% change",
                                tensor_info.name,
                                drift_ratio * 100.0
                            ),
                            timestamp: chrono::Utc::now(),
                        });
                    }
                }
            }
        }

        Ok(alerts)
    }

    /// Get tensor dependencies
    pub fn get_dependencies(&self) -> &[TensorDependency] {
        &self.dependencies
    }

    /// Get tensor lifecycle
    pub fn get_lifecycle(&self, tensor_id: Uuid) -> Option<&TensorLifecycle> {
        self.lifecycles.get(&tensor_id)
    }

    /// Get tensor time series
    pub fn get_time_series(&self, tensor_id: Uuid) -> Option<&TensorTimeSeries> {
        self.time_series.get(&tensor_id)
    }

    /// Analyze tensor relationships
    pub fn analyze_tensor_relationships(&self) -> HashMap<Uuid, Vec<Uuid>> {
        let mut relationships = HashMap::new();

        for dependency in &self.dependencies {
            relationships
                .entry(dependency.source_id)
                .or_insert_with(Vec::new)
                .push(dependency.target_id);
        }

        relationships
    }

    /// Get frequently accessed tensors
    pub fn get_frequent_tensors(&self, min_accesses: usize) -> Vec<Uuid> {
        self.lifecycles
            .iter()
            .filter(|(_, lifecycle)| lifecycle.total_accesses >= min_accesses)
            .map(|(id, _)| *id)
            .collect()
    }

    // Private helper methods

    fn compute_tensor_stats(
        &self,
        values: &[f64],
        shape: &[usize],
        element_size: usize,
    ) -> Result<TensorStats> {
        let total_elements = values.len();
        let mean = values.iter().sum::<f64>() / total_elements as f64;

        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / total_elements as f64;
        let std = variance.sqrt();

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut sorted_values = values.to_vec();
        // Filter out NaN values before sorting to avoid panic
        sorted_values.retain(|x| !x.is_nan());
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted_values.is_empty() {
            f64::NAN
        } else if sorted_values.len() % 2 == 0 {
            (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2])
                / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        let l1_norm = values.iter().map(|x| x.abs()).sum::<f64>();
        let l2_norm = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        let infinity_norm = values.iter().map(|x| x.abs()).fold(0.0, f64::max);

        let nan_count = values.iter().filter(|x| x.is_nan()).count();
        let inf_count = values.iter().filter(|x| x.is_infinite()).count();
        let zero_count = values.iter().filter(|x| **x == 0.0).count();

        let memory_usage_bytes = total_elements * element_size;
        let sparsity = zero_count as f64 / total_elements as f64;

        Ok(TensorStats {
            shape: shape.to_vec(),
            dtype: "f64".to_string(), // Simplified for now
            total_elements,
            mean,
            std,
            min,
            max,
            median,
            l1_norm,
            l2_norm,
            infinity_norm,
            nan_count,
            inf_count,
            zero_count,
            memory_usage_bytes,
            sparsity,
        })
    }

    fn compute_distribution(&self, values: &[f64]) -> Result<TensorDistribution> {
        // Simple histogram computation
        let num_bins = 50;
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bin_width = (max - min) / num_bins as f64;

        let mut histogram = vec![(0.0, 0); num_bins];
        for &value in values {
            if !value.is_finite() {
                continue;
            }
            let bin_idx = ((value - min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            histogram[bin_idx].0 = min + bin_idx as f64 * bin_width;
            histogram[bin_idx].1 += 1;
        }

        // Compute percentiles
        let mut sorted_values =
            values.iter().cloned().filter(|x| x.is_finite()).collect::<Vec<_>>();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut percentiles = HashMap::new();
        for &p in &[5.0, 25.0, 50.0, 75.0, 95.0, 99.0] {
            let idx = ((p / 100.0) * (sorted_values.len() - 1) as f64) as usize;
            percentiles.insert(format!("p{}", p as u8), sorted_values[idx]);
        }

        // Identify outliers (simple method using IQR)
        let q1 = percentiles["p25"];
        let q3 = percentiles["p75"];
        let iqr = q3 - q1;
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        let outliers: Vec<f64> = sorted_values
            .iter()
            .cloned()
            .filter(|&x| x < lower_bound || x > upper_bound)
            .take(100) // Limit outliers to avoid memory issues
            .collect();

        // Basic skewness and kurtosis (simplified formulas)
        let mean = sorted_values.iter().sum::<f64>() / sorted_values.len() as f64;
        let variance = sorted_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / sorted_values.len() as f64;
        let std = variance.sqrt();

        let skewness = if std > 0.0 {
            sorted_values.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>()
                / sorted_values.len() as f64
        } else {
            0.0
        };

        let kurtosis = if std > 0.0 {
            sorted_values.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>()
                / sorted_values.len() as f64
                - 3.0
        } else {
            0.0
        };

        Ok(TensorDistribution {
            histogram,
            percentiles,
            outliers,
            skewness,
            kurtosis,
        })
    }

    fn should_compute_distribution(&self) -> bool {
        self.config.sampling_rate >= 1.0
            || (self.config.sampling_rate > 0.0 && random::<f32>() < self.config.sampling_rate)
    }

    fn check_tensor_alerts(&mut self, tensor_info: &TensorInfo) -> Result<()> {
        // Check for NaN values
        if tensor_info.stats.nan_count > 0 {
            self.alerts.push(TensorAlert {
                id: Uuid::new_v4(),
                tensor_id: tensor_info.id,
                tensor_name: tensor_info.name.clone(),
                alert_type: TensorAlertType::NaNValues,
                severity: AlertSeverity::Critical,
                message: format!(
                    "Found {} NaN values in tensor '{}'",
                    tensor_info.stats.nan_count, tensor_info.name
                ),
                timestamp: chrono::Utc::now(),
            });
        }

        // Check for infinite values
        if tensor_info.stats.inf_count > 0 {
            self.alerts.push(TensorAlert {
                id: Uuid::new_v4(),
                tensor_id: tensor_info.id,
                tensor_name: tensor_info.name.clone(),
                alert_type: TensorAlertType::InfiniteValues,
                severity: AlertSeverity::Critical,
                message: format!(
                    "Found {} infinite values in tensor '{}'",
                    tensor_info.stats.inf_count, tensor_info.name
                ),
                timestamp: chrono::Utc::now(),
            });
        }

        // Check for extreme values
        if tensor_info.stats.max.abs() > 1e10 || tensor_info.stats.min.abs() > 1e10 {
            self.alerts.push(TensorAlert {
                id: Uuid::new_v4(),
                tensor_id: tensor_info.id,
                tensor_name: tensor_info.name.clone(),
                alert_type: TensorAlertType::ExtremeValues,
                severity: AlertSeverity::Warning,
                message: format!(
                    "Extreme values detected in tensor '{}': min={:.2e}, max={:.2e}",
                    tensor_info.name, tensor_info.stats.min, tensor_info.stats.max
                ),
                timestamp: chrono::Utc::now(),
            });
        }

        Ok(())
    }

    fn check_gradient_alerts_with_data(
        &mut self,
        tensor_id: Uuid,
        tensor_name: &str,
        grad_stats: Option<TensorStats>,
    ) -> Result<()> {
        if let Some(ref stats) = grad_stats {
            // Check for vanishing gradients
            if stats.l2_norm < 1e-8 {
                self.alerts.push(TensorAlert {
                    id: Uuid::new_v4(),
                    tensor_id,
                    tensor_name: tensor_name.to_string(),
                    alert_type: TensorAlertType::VanishingGradients,
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "Vanishing gradients detected in '{}': L2 norm = {:.2e}",
                        tensor_name, stats.l2_norm
                    ),
                    timestamp: chrono::Utc::now(),
                });
            }

            // Check for exploding gradients
            if stats.l2_norm > 100.0 {
                self.alerts.push(TensorAlert {
                    id: Uuid::new_v4(),
                    tensor_id,
                    tensor_name: tensor_name.to_string(),
                    alert_type: TensorAlertType::ExplodingGradients,
                    severity: AlertSeverity::Critical,
                    message: format!(
                        "Exploding gradients detected in '{}': L2 norm = {:.2e}",
                        tensor_name, stats.l2_norm
                    ),
                    timestamp: chrono::Utc::now(),
                });
            }
        }

        Ok(())
    }

    fn compute_mse(&self, stats1: &TensorStats, stats2: &TensorStats) -> f64 {
        // Simplified MSE using means (would need actual tensor data for real MSE)
        (stats1.mean - stats2.mean).powi(2)
    }

    fn compute_mae(&self, stats1: &TensorStats, stats2: &TensorStats) -> f64 {
        // Simplified MAE using means
        (stats1.mean - stats2.mean).abs()
    }

    fn compute_cosine_similarity(&self, stats1: &TensorStats, stats2: &TensorStats) -> f64 {
        // Simplified using L2 norms (would need actual tensors for real cosine similarity)
        if stats1.l2_norm == 0.0 || stats2.l2_norm == 0.0 {
            0.0
        } else {
            (stats1.mean * stats2.mean) / (stats1.l2_norm * stats2.l2_norm)
        }
    }

    fn compute_correlation(&self, stats1: &TensorStats, stats2: &TensorStats) -> f64 {
        // Simplified correlation (would need actual tensor data for real correlation)
        if stats1.std == 0.0 || stats2.std == 0.0 {
            0.0
        } else {
            0.5 // Placeholder
        }
    }

    // Advanced analysis helper methods

    fn compute_spectral_analysis(
        &self,
        values: &[f64],
        shape: &[usize],
    ) -> Result<Option<SpectralAnalysis>> {
        // Only perform spectral analysis for 2D matrices
        if shape.len() != 2 || values.len() < 4 {
            return Ok(None);
        }

        let rows = shape[0];
        let cols = shape[1];

        // Simple eigenvalue estimation using power iteration for the largest eigenvalue
        let largest_eigenvalue = self.power_iteration(values, rows, cols)?;

        // Estimate condition number using Frobenius norm ratio
        let frobenius_norm = (values.iter().map(|x| x * x).sum::<f64>()).sqrt();
        let condition_number = if frobenius_norm > 1e-12 {
            largest_eigenvalue / frobenius_norm.max(1e-12)
        } else {
            f64::INFINITY
        };

        // Estimate rank by counting non-zero singular values (simplified)
        let rank = values.iter().filter(|&&x| x.abs() > 1e-10).count().min(rows.min(cols));

        Ok(Some(SpectralAnalysis {
            eigenvalues: vec![largest_eigenvalue], // Simplified - only largest
            condition_number,
            rank,
            spectral_norm: largest_eigenvalue,
        }))
    }

    fn power_iteration(&self, matrix: &[f64], rows: usize, cols: usize) -> Result<f64> {
        if rows != cols {
            return Ok(0.0); // Non-square matrices
        }

        let n = rows;
        let mut x = vec![1.0; n];
        let mut lambda = 0.0;

        // Simple power iteration (few iterations for performance)
        for _ in 0..10 {
            let mut ax = vec![0.0; n];

            // Matrix-vector multiplication: Ax
            for i in 0..n {
                for j in 0..n {
                    ax[i] += matrix[i * n + j] * x[j];
                }
            }

            // Compute Rayleigh quotient
            let dot_ax_x: f64 = ax.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let dot_x_x: f64 = x.iter().map(|a| a * a).sum();

            if dot_x_x > 1e-12 {
                lambda = dot_ax_x / dot_x_x;
            }

            // Normalize
            let norm = (ax.iter().map(|a| a * a).sum::<f64>()).sqrt();
            if norm > 1e-12 {
                x = ax.iter().map(|a| a / norm).collect();
            }
        }

        Ok(lambda.abs())
    }

    fn compute_information_content(&self, values: &[f64]) -> Result<InformationContent> {
        // Shannon entropy calculation
        let mut histogram = std::collections::HashMap::new();
        let quantization_levels = 100;
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        if range > 1e-12 {
            for &value in values {
                let bucket = ((value - min_val) / range * quantization_levels as f64) as usize;
                let bucket = bucket.min(quantization_levels - 1);
                *histogram.entry(bucket).or_insert(0) += 1;
            }
        }

        let total_count = values.len() as f64;
        let entropy = if total_count > 0.0 {
            histogram
                .values()
                .map(|&count| {
                    let p = count as f64 / total_count;
                    if p > 0.0 {
                        -p * p.log2()
                    } else {
                        0.0
                    }
                })
                .sum()
        } else {
            0.0
        };

        // Effective rank estimation using entropy
        let effective_rank = if entropy > 0.0 { 2.0_f64.powf(entropy) } else { 1.0 };

        // Simple compression ratio estimation
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_values.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        let unique_values = sorted_values.len();
        let compression_ratio = unique_values as f64 / values.len() as f64;

        Ok(InformationContent {
            entropy,
            mutual_information: 0.0, // Would need multiple tensors to compute
            effective_rank,
            compression_ratio,
        })
    }

    fn compute_stability_metrics(&self, values: &[f64]) -> Result<StabilityMetrics> {
        // Numerical stability based on condition of values
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let numerical_stability = if std_dev > 1e-12 {
            1.0 / (1.0 + std_dev / mean.abs().max(1e-12))
        } else {
            1.0
        };

        // Simple perturbation sensitivity
        let max_abs = values.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let perturbation_sensitivity = if max_abs > 1e-12 { std_dev / max_abs } else { 0.0 };

        // Overall robustness score
        let robustness_score = numerical_stability * (1.0 - perturbation_sensitivity.min(1.0));

        Ok(StabilityMetrics {
            numerical_stability,
            gradient_stability: 0.8, // Placeholder - would need gradient info
            perturbation_sensitivity,
            robustness_score,
        })
    }

    fn compute_relationship_analysis(&self, values: &[f64]) -> Result<RelationshipAnalysis> {
        // Simple relationship analysis within the tensor
        let mut cross_correlations = HashMap::new();
        let mut dependency_strength = HashMap::new();
        let causal_relationships = Vec::new(); // Would need temporal data

        // For now, just compute some basic relationships
        for tensor_info in self.tracked_tensors.values() {
            if tensor_info.stats.total_elements > 0 {
                let correlation =
                    self.compute_simple_correlation(values, &[tensor_info.stats.mean]);
                cross_correlations.insert(tensor_info.id, correlation);
                dependency_strength.insert(tensor_info.id, correlation.abs());
            }
        }

        Ok(RelationshipAnalysis {
            cross_correlations,
            dependency_strength,
            causal_relationships,
        })
    }

    fn compute_simple_correlation(&self, values1: &[f64], values2: &[f64]) -> f64 {
        if values1.is_empty() || values2.is_empty() {
            return 0.0;
        }

        let mean1 = values1.iter().sum::<f64>() / values1.len() as f64;
        let mean2 = values2.iter().sum::<f64>() / values2.len() as f64;

        let min_len = values1.len().min(values2.len());
        let numerator: f64 = values1
            .iter()
            .zip(values2.iter())
            .take(min_len)
            .map(|(x, y)| (x - mean1) * (y - mean2))
            .sum();

        let sum_sq1: f64 = values1.iter().take(min_len).map(|x| (x - mean1).powi(2)).sum();
        let sum_sq2: f64 = values2.iter().take(min_len).map(|y| (y - mean2).powi(2)).sum();

        let denominator = (sum_sq1 * sum_sq2).sqrt();
        if denominator > 1e-12 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn compute_summary_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        if !self.tracked_tensors.is_empty() {
            let values: Vec<f64> = self.tracked_tensors.values().map(|t| t.stats.mean).collect();
            stats.insert(
                "mean_of_means".to_string(),
                values.iter().sum::<f64>() / values.len() as f64,
            );

            let total_memory: usize =
                self.tracked_tensors.values().map(|t| t.stats.memory_usage_bytes).sum();
            stats.insert(
                "total_memory_mb".to_string(),
                total_memory as f64 / (1024.0 * 1024.0),
            );

            let avg_sparsity: f64 =
                self.tracked_tensors.values().map(|t| t.stats.sparsity).sum::<f64>()
                    / self.tracked_tensors.len() as f64;
            stats.insert("avg_sparsity".to_string(), avg_sparsity);

            // Add advanced statistics
            stats.insert(
                "total_dependencies".to_string(),
                self.dependencies.len() as f64,
            );
            stats.insert(
                "monitored_tensors".to_string(),
                self.time_series.len() as f64,
            );
            stats.insert(
                "active_lifecycles".to_string(),
                self.lifecycles.len() as f64,
            );
        }

        stats
    }
}

/// Alert types for tensor issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorAlertType {
    NaNValues,
    InfiniteValues,
    ExtremeValues,
    VanishingGradients,
    ExplodingGradients,
    MemoryUsage,
    ShapeMismatch,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Tensor alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorAlert {
    pub id: Uuid,
    pub tensor_id: Uuid,
    pub tensor_name: String,
    pub alert_type: TensorAlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Tensor inspection report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInspectionReport {
    pub total_tensors: usize,
    pub tensors_with_issues: usize,
    pub total_memory_usage: usize,
    pub alerts: Vec<TensorAlert>,
    pub comparisons: Vec<TensorComparison>,
    pub summary_stats: HashMap<String, f64>,
}

impl TensorInspectionReport {
    pub fn has_nan_values(&self) -> bool {
        self.alerts.iter().any(|a| matches!(a.alert_type, TensorAlertType::NaNValues))
    }

    pub fn has_inf_values(&self) -> bool {
        self.alerts
            .iter()
            .any(|a| matches!(a.alert_type, TensorAlertType::InfiniteValues))
    }

    pub fn total_nan_count(&self) -> usize {
        self.alerts
            .iter()
            .filter(|a| matches!(a.alert_type, TensorAlertType::NaNValues))
            .count()
    }

    pub fn total_inf_count(&self) -> usize {
        self.alerts
            .iter()
            .filter(|a| matches!(a.alert_type, TensorAlertType::InfiniteValues))
            .count()
    }

    pub fn has_critical_alerts(&self) -> bool {
        self.alerts.iter().any(|a| matches!(a.severity, AlertSeverity::Critical))
    }
}
