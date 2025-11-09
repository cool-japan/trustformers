//! Advanced analytics for model behavior analysis.
//!
//! This module provides sophisticated analytical capabilities including
//! clustering analysis, temporal dynamics monitoring, representation
//! stability assessment, and multi-dimensional data visualization.

use anyhow::Result;
use std::collections::{HashMap, VecDeque};

use super::types::{
    ActivationHeatmap, ClusteringResults, DriftInfo, HiddenStateAnalysis, LayerActivationStats,
    ModelPerformanceMetrics, RepresentationStability, TemporalDynamics,
};

/// Advanced analytics engine for model behavior analysis.
#[derive(Debug)]
#[allow(dead_code)]
pub struct AdvancedAnalytics {
    /// Analytics configuration
    config: AnalyticsConfig,
    /// Historical hidden state data
    hidden_states_history: VecDeque<HiddenStateData>,
    /// Performance correlation data
    performance_correlations: HashMap<String, CorrelationData>,
    /// Temporal analysis cache
    #[allow(dead_code)]
    temporal_analysis_cache: TemporalAnalysisCache,
    /// Clustering analysis results
    clustering_results_cache: HashMap<String, ClusteringResults>,
}

/// Configuration for advanced analytics.
#[derive(Debug, Clone)]
pub struct AnalyticsConfig {
    /// Maximum number of historical samples to retain
    pub max_history_samples: usize,
    /// Minimum samples required for clustering analysis
    pub min_clustering_samples: usize,
    /// Number of clusters for k-means analysis
    pub default_num_clusters: usize,
    /// Window size for temporal analysis
    pub temporal_analysis_window: usize,
    /// Drift detection sensitivity
    pub drift_detection_sensitivity: f64,
    /// Correlation analysis threshold
    pub correlation_threshold: f64,
    /// Enable advanced visualizations
    pub enable_visualizations: bool,
}

/// Hidden state data for analysis.
#[derive(Debug, Clone)]
pub struct HiddenStateData {
    /// Layer name
    pub layer_name: String,
    /// Hidden state vectors
    pub hidden_states: Vec<Vec<f64>>,
    /// Corresponding labels or metadata
    pub labels: Option<Vec<String>>,
    /// Timestamp of collection
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Training step when collected
    pub training_step: usize,
}

/// Correlation analysis data.
#[derive(Debug, Clone)]
pub struct CorrelationData {
    /// Metric name
    pub metric_name: String,
    /// Historical values
    pub values: VecDeque<f64>,
    /// Correlations with other metrics
    pub correlations: HashMap<String, f64>,
    /// Last update timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Temporal analysis cache for performance optimization.
#[derive(Debug, Clone)]
pub struct TemporalAnalysisCache {
    /// Cached drift detection results
    pub drift_results: HashMap<String, DriftInfo>,
    /// Cached temporal consistency scores
    pub consistency_scores: HashMap<String, f64>,
    /// Cached stability windows
    pub stability_windows: HashMap<String, Vec<(usize, usize)>>,
    /// Last analysis timestamp
    pub last_analysis: chrono::DateTime<chrono::Utc>,
}

/// Advanced clustering analysis parameters.
#[derive(Debug, Clone)]
pub struct ClusteringParameters {
    /// Number of clusters
    pub num_clusters: usize,
    /// Maximum iterations for k-means
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Distance metric to use
    pub distance_metric: DistanceMetric,
}

/// Distance metrics for clustering.
#[derive(Debug, Clone)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine similarity
    Cosine,
    /// Minkowski distance with parameter p
    Minkowski { p: f64 },
}

/// Dimensionality reduction parameters.
#[derive(Debug, Clone)]
pub struct DimensionalityReductionParams {
    /// Target dimensions
    pub target_dimensions: usize,
    /// Reduction method
    pub method: ReductionMethod,
    /// Preserve variance ratio
    pub preserve_variance_ratio: f64,
}

/// Dimensionality reduction methods.
#[derive(Debug, Clone)]
pub enum ReductionMethod {
    /// Principal Component Analysis
    PCA,
    /// t-SNE
    TSNE { perplexity: f64 },
    /// UMAP
    UMAP { n_neighbors: usize, min_dist: f64 },
}

/// Visualization generation parameters.
#[derive(Debug, Clone)]
pub struct VisualizationParams {
    /// Output dimensions (width, height)
    pub dimensions: (usize, usize),
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Include annotations
    pub include_annotations: bool,
    /// Export format
    pub export_format: ExportFormat,
}

/// Color schemes for visualizations.
#[derive(Debug, Clone)]
pub enum ColorScheme {
    /// Viridis color scale
    Viridis,
    /// Plasma color scale
    Plasma,
    /// Inferno color scale
    Inferno,
    /// Custom color map
    Custom(Vec<(f64, f64, f64)>),
}

/// Export formats for visualizations.
#[derive(Debug, Clone)]
pub enum ExportFormat {
    /// PNG image
    PNG,
    /// SVG vector
    SVG,
    /// JSON data
    JSON,
    /// CSV data
    CSV,
}

/// Statistical analysis results.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StatisticalAnalysis {
    /// Mean values
    pub means: Vec<f64>,
    /// Standard deviations
    pub std_devs: Vec<f64>,
    /// Correlation matrix
    pub correlation_matrix: Vec<Vec<f64>>,
    /// Principal components
    pub principal_components: Vec<Vec<f64>>,
    /// Explained variance ratios
    pub explained_variance_ratios: Vec<f64>,
    /// Statistical significance tests
    pub significance_tests: Vec<SignificanceTest>,
}

/// Statistical significance test result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SignificanceTest {
    /// Test name
    pub test_name: String,
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: Option<usize>,
    /// Confidence interval
    pub confidence_interval: Option<(f64, f64)>,
}

/// Anomaly detection results.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnomalyDetectionResults {
    /// Detected anomalies
    pub anomalies: Vec<Anomaly>,
    /// Anomaly scores for all data points
    pub anomaly_scores: Vec<f64>,
    /// Detection threshold used
    pub threshold: f64,
    /// Detection method
    pub method: AnomalyDetectionMethod,
}

/// Individual anomaly information.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Anomaly {
    /// Index in dataset
    pub index: usize,
    /// Anomaly score
    pub score: f64,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Anomaly detection methods.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AnomalyDetectionMethod {
    /// Isolation Forest
    IsolationForest { n_trees: usize },
    /// Local Outlier Factor
    LocalOutlierFactor { n_neighbors: usize },
    /// One-Class SVM
    OneClassSVM { nu: f64 },
    /// Statistical threshold
    StatisticalThreshold { n_std: f64 },
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            max_history_samples: 10000,
            min_clustering_samples: 50,
            default_num_clusters: 8,
            temporal_analysis_window: 100,
            drift_detection_sensitivity: 0.05,
            correlation_threshold: 0.7,
            enable_visualizations: true,
        }
    }
}

impl Default for ClusteringParameters {
    fn default() -> Self {
        Self {
            num_clusters: 8,
            max_iterations: 100,
            tolerance: 1e-4,
            random_seed: Some(42),
            distance_metric: DistanceMetric::Euclidean,
        }
    }
}

impl AdvancedAnalytics {
    /// Create a new advanced analytics engine.
    pub fn new() -> Self {
        Self {
            config: AnalyticsConfig::default(),
            hidden_states_history: VecDeque::new(),
            performance_correlations: HashMap::new(),
            temporal_analysis_cache: TemporalAnalysisCache::new(),
            clustering_results_cache: HashMap::new(),
        }
    }

    /// Create analytics engine with custom configuration.
    pub fn with_config(config: AnalyticsConfig) -> Self {
        Self {
            config,
            hidden_states_history: VecDeque::new(),
            performance_correlations: HashMap::new(),
            temporal_analysis_cache: TemporalAnalysisCache::new(),
            clustering_results_cache: HashMap::new(),
        }
    }

    /// Record hidden state data for analysis.
    pub fn record_hidden_states(&mut self, hidden_states: HiddenStateData) {
        self.hidden_states_history.push_back(hidden_states);

        while self.hidden_states_history.len() > self.config.max_history_samples {
            self.hidden_states_history.pop_front();
        }
    }

    /// Record performance metrics for correlation analysis.
    pub fn record_performance_metrics(&mut self, metrics: &ModelPerformanceMetrics) {
        self.update_correlation_data("loss", metrics.loss);
        self.update_correlation_data("throughput", metrics.throughput_samples_per_sec);
        self.update_correlation_data("memory_usage", metrics.memory_usage_mb);

        if let Some(accuracy) = metrics.accuracy {
            self.update_correlation_data("accuracy", accuracy);
        }

        if let Some(gpu_util) = metrics.gpu_utilization {
            self.update_correlation_data("gpu_utilization", gpu_util);
        }
    }

    /// Perform comprehensive hidden state analysis.
    pub fn analyze_hidden_states(&self, layer_name: &str) -> Result<HiddenStateAnalysis> {
        let layer_data: Vec<_> = self
            .hidden_states_history
            .iter()
            .filter(|data| data.layer_name == layer_name)
            .collect();

        if layer_data.is_empty() {
            return Err(anyhow::anyhow!(
                "No hidden state data available for layer: {}",
                layer_name
            ));
        }

        // Extract all hidden states for this layer
        let all_states: Vec<Vec<f64>> =
            layer_data.iter().flat_map(|data| data.hidden_states.iter()).cloned().collect();

        if all_states.is_empty() {
            return Err(anyhow::anyhow!(
                "No hidden states found for layer: {}",
                layer_name
            ));
        }

        let dimensionality = all_states[0].len();

        // Perform clustering analysis
        let clustering_results = self.perform_clustering_analysis(&all_states)?;

        // Analyze temporal dynamics
        let temporal_dynamics = self.analyze_temporal_dynamics(&layer_data)?;

        // Assess representation stability
        let representation_stability = self.assess_representation_stability(&all_states)?;

        // Calculate information content
        let information_content = self.calculate_information_content(&all_states)?;

        Ok(HiddenStateAnalysis {
            dimensionality,
            information_content,
            clustering_results,
            temporal_dynamics,
            representation_stability,
        })
    }

    /// Perform clustering analysis on hidden states.
    pub fn perform_clustering_analysis(&self, data: &[Vec<f64>]) -> Result<ClusteringResults> {
        if data.len() < self.config.min_clustering_samples {
            return Err(anyhow::anyhow!("Insufficient data for clustering analysis"));
        }

        let params = ClusteringParameters::default();
        let num_clusters = params.num_clusters.min(data.len() / 2);

        // Simple k-means clustering implementation
        let mut cluster_centers = self.initialize_cluster_centers(data, num_clusters)?;
        let mut cluster_assignments = vec![0; data.len()];
        #[allow(dead_code)]
        #[allow(unused_assignments)]
        let mut _converged = false;

        for _iteration in 0..params.max_iterations {
            // Assign points to nearest cluster
            let mut new_assignments = vec![0; data.len()];
            for (i, point) in data.iter().enumerate() {
                let mut best_distance = f64::INFINITY;
                let mut best_cluster = 0;

                for (j, center) in cluster_centers.iter().enumerate() {
                    let distance =
                        self.calculate_distance(point, center, &params.distance_metric)?;
                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = j;
                    }
                }
                new_assignments[i] = best_cluster;
            }

            // Check for convergence
            if new_assignments == cluster_assignments {
                _converged = true;
                break;
            }
            cluster_assignments = new_assignments;

            // Update cluster centers
            cluster_centers =
                self.update_cluster_centers(data, &cluster_assignments, num_clusters)?;
        }

        // Calculate silhouette score
        let silhouette_score =
            self.calculate_silhouette_score(data, &cluster_assignments, &cluster_centers)?;

        // Calculate inertia
        let inertia = self.calculate_inertia(data, &cluster_assignments, &cluster_centers)?;

        Ok(ClusteringResults {
            num_clusters,
            cluster_centers,
            cluster_assignments,
            silhouette_score,
            inertia,
        })
    }

    /// Analyze temporal dynamics of hidden states.
    pub fn analyze_temporal_dynamics(
        &self,
        layer_data: &[&HiddenStateData],
    ) -> Result<TemporalDynamics> {
        if layer_data.len() < 2 {
            return Err(anyhow::anyhow!("Insufficient temporal data"));
        }

        // Calculate temporal consistency
        let temporal_consistency = self.calculate_temporal_consistency(layer_data)?;

        // Calculate change rate
        let change_rate = self.calculate_change_rate(layer_data)?;

        // Identify stability windows
        let stability_windows = self.identify_stability_windows(layer_data)?;

        // Detect distribution drift
        let drift_detection = self.detect_distribution_drift(layer_data)?;

        Ok(TemporalDynamics {
            temporal_consistency,
            change_rate,
            stability_windows,
            drift_detection,
        })
    }

    /// Assess representation stability.
    pub fn assess_representation_stability(
        &self,
        hidden_states: &[Vec<f64>],
    ) -> Result<RepresentationStability> {
        if hidden_states.is_empty() {
            return Err(anyhow::anyhow!("No hidden states provided"));
        }

        // Calculate overall stability score
        let stability_score = self.calculate_stability_score(hidden_states)?;

        // Calculate variance across batches (simulated)
        let variance_across_batches = self.calculate_batch_variance(hidden_states)?;

        // Calculate consistency measure
        let consistency_measure = self.calculate_consistency_measure(hidden_states)?;

        // Assess robustness to noise (simulated)
        let robustness_to_noise = self.assess_noise_robustness(hidden_states)?;

        Ok(RepresentationStability {
            stability_score,
            variance_across_batches,
            consistency_measure,
            robustness_to_noise,
        })
    }

    /// Generate activation heatmap for visualization.
    pub fn generate_activation_heatmap(
        &self,
        layer_stats: &[LayerActivationStats],
    ) -> Result<ActivationHeatmap> {
        if layer_stats.is_empty() {
            return Err(anyhow::anyhow!("No layer statistics provided"));
        }

        // Create heatmap data based on layer statistics
        let mut data = Vec::new();
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for stats in layer_stats {
            let row = vec![
                stats.mean_activation,
                stats.std_activation,
                stats.min_activation,
                stats.max_activation,
                stats.dead_neurons_ratio,
                stats.saturated_neurons_ratio,
                stats.sparsity,
            ];

            for &val in &row {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }

            data.push(row);
        }

        let dimensions = (data.len(), data.get(0).map_or(0, |row| row.len()));

        Ok(ActivationHeatmap {
            data,
            dimensions,
            value_range: (min_val, max_val),
            interpretation: "Activation statistics heatmap showing layer behavior patterns"
                .to_string(),
        })
    }

    /// Perform anomaly detection on performance metrics.
    pub fn detect_performance_anomalies(&self) -> Result<AnomalyDetectionResults> {
        // Extract performance data
        let mut all_values = Vec::new();
        for correlation_data in self.performance_correlations.values() {
            all_values.extend(correlation_data.values.iter().cloned());
        }

        if all_values.is_empty() {
            return Err(anyhow::anyhow!("No performance data available"));
        }

        // Use statistical threshold method for anomaly detection
        let method = AnomalyDetectionMethod::StatisticalThreshold { n_std: 2.0 };

        let mean = all_values.iter().sum::<f64>() / all_values.len() as f64;
        let variance =
            all_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / all_values.len() as f64;
        let std_dev = variance.sqrt();

        let threshold = mean + 2.0 * std_dev;

        let mut anomalies = Vec::new();
        let mut anomaly_scores = Vec::new();

        for (i, &value) in all_values.iter().enumerate() {
            let score = (value - mean).abs() / std_dev;
            anomaly_scores.push(score);

            if value > threshold {
                anomalies.push(Anomaly {
                    index: i,
                    score,
                    timestamp: chrono::Utc::now(),
                    context: HashMap::new(),
                });
            }
        }

        Ok(AnomalyDetectionResults {
            anomalies,
            anomaly_scores,
            threshold,
            method,
        })
    }

    /// Calculate correlation matrix for all metrics.
    pub fn calculate_correlation_matrix(&self) -> Result<Vec<Vec<f64>>> {
        let metric_names: Vec<_> = self.performance_correlations.keys().cloned().collect();
        let n_metrics = metric_names.len();

        if n_metrics == 0 {
            return Err(anyhow::anyhow!(
                "No metrics available for correlation analysis"
            ));
        }

        let mut correlation_matrix = vec![vec![0.0; n_metrics]; n_metrics];

        for (i, metric1) in metric_names.iter().enumerate() {
            for (j, metric2) in metric_names.iter().enumerate() {
                if i == j {
                    correlation_matrix[i][j] = 1.0;
                } else {
                    let correlation = self.calculate_correlation(metric1, metric2)?;
                    correlation_matrix[i][j] = correlation;
                }
            }
        }

        Ok(correlation_matrix)
    }

    /// Perform statistical analysis on collected data.
    pub fn perform_statistical_analysis(&self) -> Result<StatisticalAnalysis> {
        if self.performance_correlations.is_empty() {
            return Err(anyhow::anyhow!(
                "No data available for statistical analysis"
            ));
        }

        // Calculate means and standard deviations
        let mut means = Vec::new();
        let mut std_devs = Vec::new();

        for correlation_data in self.performance_correlations.values() {
            let values: Vec<f64> = correlation_data.values.iter().cloned().collect();
            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
                let std_dev = variance.sqrt();

                means.push(mean);
                std_devs.push(std_dev);
            }
        }

        // Calculate correlation matrix
        let correlation_matrix = self.calculate_correlation_matrix()?;

        // Placeholder for principal components and explained variance
        let principal_components = vec![vec![1.0; means.len()]; means.len()];
        let explained_variance_ratios = vec![1.0 / means.len() as f64; means.len()];

        // Placeholder for significance tests
        let significance_tests = vec![SignificanceTest {
            test_name: "Sample t-test".to_string(),
            statistic: 1.0,
            p_value: 0.05,
            degrees_of_freedom: Some(means.len() - 1),
            confidence_interval: Some((0.0, 1.0)),
        }];

        Ok(StatisticalAnalysis {
            means,
            std_devs,
            correlation_matrix,
            principal_components,
            explained_variance_ratios,
            significance_tests,
        })
    }

    /// Generate comprehensive analytics report.
    pub fn generate_analytics_report(&self) -> Result<AnalyticsReport> {
        let correlation_matrix = self.calculate_correlation_matrix().unwrap_or_default();
        let statistical_analysis = self.perform_statistical_analysis().unwrap_or_default();
        let anomaly_detection = self.detect_performance_anomalies().unwrap_or_default();

        // Analyze each layer if data is available
        let mut layer_analyses = HashMap::new();
        let unique_layers: std::collections::HashSet<String> =
            self.hidden_states_history.iter().map(|data| data.layer_name.clone()).collect();

        for layer_name in unique_layers {
            if let Ok(analysis) = self.analyze_hidden_states(&layer_name) {
                layer_analyses.insert(layer_name, analysis);
            }
        }

        Ok(AnalyticsReport {
            correlation_matrix,
            statistical_analysis,
            layer_analyses,
            anomaly_detection,
            temporal_summary: self.generate_temporal_summary(),
            recommendations: self.generate_analytics_recommendations(),
        })
    }

    // Helper methods

    /// Update correlation data for a metric.
    fn update_correlation_data(&mut self, metric_name: &str, value: f64) {
        let correlation_data = self
            .performance_correlations
            .entry(metric_name.to_string())
            .or_insert_with(|| CorrelationData {
                metric_name: metric_name.to_string(),
                values: VecDeque::new(),
                correlations: HashMap::new(),
                last_updated: chrono::Utc::now(),
            });

        correlation_data.values.push_back(value);
        correlation_data.last_updated = chrono::Utc::now();

        // Limit history size
        while correlation_data.values.len() > self.config.max_history_samples {
            correlation_data.values.pop_front();
        }
    }

    /// Initialize cluster centers using k-means++ algorithm.
    fn initialize_cluster_centers(
        &self,
        data: &[Vec<f64>],
        num_clusters: usize,
    ) -> Result<Vec<Vec<f64>>> {
        if data.is_empty() || num_clusters == 0 {
            return Err(anyhow::anyhow!("Invalid input for cluster initialization"));
        }

        let mut centers = Vec::new();
        let _dimensions = data[0].len();

        // Choose first center randomly (simplified - just use first point)
        centers.push(data[0].clone());

        // Choose remaining centers using k-means++ logic (simplified)
        for _ in 1..num_clusters {
            if centers.len() >= data.len() {
                break;
            }

            let mut best_distance = 0.0;
            let mut best_point = data[0].clone();

            for point in data {
                let mut min_distance = f64::INFINITY;
                for center in &centers {
                    let distance =
                        self.calculate_distance(point, center, &DistanceMetric::Euclidean)?;
                    min_distance = min_distance.min(distance);
                }

                if min_distance > best_distance {
                    best_distance = min_distance;
                    best_point = point.clone();
                }
            }

            centers.push(best_point);
        }

        Ok(centers)
    }

    /// Calculate distance between two points.
    fn calculate_distance(
        &self,
        point1: &[f64],
        point2: &[f64],
        metric: &DistanceMetric,
    ) -> Result<f64> {
        if point1.len() != point2.len() {
            return Err(anyhow::anyhow!("Points must have same dimensionality"));
        }

        match metric {
            DistanceMetric::Euclidean => {
                let sum_squared =
                    point1.iter().zip(point2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
                Ok(sum_squared.sqrt())
            },
            DistanceMetric::Manhattan => {
                let sum_abs =
                    point1.iter().zip(point2.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>();
                Ok(sum_abs)
            },
            DistanceMetric::Cosine => {
                let dot_product = point1.iter().zip(point2.iter()).map(|(a, b)| a * b).sum::<f64>();
                let norm1 = point1.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                let norm2 = point2.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

                if norm1 == 0.0 || norm2 == 0.0 {
                    Ok(1.0)
                } else {
                    Ok(1.0 - (dot_product / (norm1 * norm2)))
                }
            },
            DistanceMetric::Minkowski { p } => {
                let sum_powered = point1
                    .iter()
                    .zip(point2.iter())
                    .map(|(a, b)| (a - b).abs().powf(*p))
                    .sum::<f64>();
                Ok(sum_powered.powf(1.0 / p))
            },
        }
    }

    /// Update cluster centers based on current assignments.
    fn update_cluster_centers(
        &self,
        data: &[Vec<f64>],
        assignments: &[usize],
        num_clusters: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let dimensions = data[0].len();
        let mut new_centers = vec![vec![0.0; dimensions]; num_clusters];
        let mut cluster_counts = vec![0; num_clusters];

        // Sum points in each cluster
        for (point, &cluster_id) in data.iter().zip(assignments.iter()) {
            if cluster_id < num_clusters {
                for (i, &value) in point.iter().enumerate() {
                    new_centers[cluster_id][i] += value;
                }
                cluster_counts[cluster_id] += 1;
            }
        }

        // Average to get new centers
        for (cluster_id, count) in cluster_counts.iter().enumerate() {
            if *count > 0 {
                for value in &mut new_centers[cluster_id] {
                    *value /= *count as f64;
                }
            }
        }

        Ok(new_centers)
    }

    /// Calculate silhouette score for clustering quality.
    fn calculate_silhouette_score(
        &self,
        data: &[Vec<f64>],
        assignments: &[usize],
        centers: &[Vec<f64>],
    ) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        let mut total_score = 0.0;
        let mut valid_points = 0;

        for (i, point) in data.iter().enumerate() {
            let cluster_id = assignments[i];

            // Calculate average distance to points in same cluster (a)
            let mut same_cluster_distances = Vec::new();
            for (j, other_point) in data.iter().enumerate() {
                if i != j && assignments[j] == cluster_id {
                    let distance =
                        self.calculate_distance(point, other_point, &DistanceMetric::Euclidean)?;
                    same_cluster_distances.push(distance);
                }
            }

            let a = if same_cluster_distances.is_empty() {
                0.0
            } else {
                same_cluster_distances.iter().sum::<f64>() / same_cluster_distances.len() as f64
            };

            // Calculate minimum average distance to points in other clusters (b)
            let mut min_other_cluster_distance = f64::INFINITY;
            for (other_cluster_id, _) in centers.iter().enumerate() {
                if other_cluster_id != cluster_id {
                    let mut other_cluster_distances = Vec::new();
                    for (j, other_point) in data.iter().enumerate() {
                        if assignments[j] == other_cluster_id {
                            let distance = self.calculate_distance(
                                point,
                                other_point,
                                &DistanceMetric::Euclidean,
                            )?;
                            other_cluster_distances.push(distance);
                        }
                    }

                    if !other_cluster_distances.is_empty() {
                        let avg_distance = other_cluster_distances.iter().sum::<f64>()
                            / other_cluster_distances.len() as f64;
                        min_other_cluster_distance = min_other_cluster_distance.min(avg_distance);
                    }
                }
            }

            let b = min_other_cluster_distance;

            if a < b {
                total_score += (b - a) / b;
            } else if a > b {
                total_score += (b - a) / a;
            }
            // If a == b, silhouette score is 0 (no contribution)

            valid_points += 1;
        }

        Ok(if valid_points > 0 { total_score / valid_points as f64 } else { 0.0 })
    }

    /// Calculate inertia (sum of squared distances to centroids).
    fn calculate_inertia(
        &self,
        data: &[Vec<f64>],
        assignments: &[usize],
        centers: &[Vec<f64>],
    ) -> Result<f64> {
        let mut inertia = 0.0;

        for (point, &cluster_id) in data.iter().zip(assignments.iter()) {
            if cluster_id < centers.len() {
                let distance = self.calculate_distance(
                    point,
                    &centers[cluster_id],
                    &DistanceMetric::Euclidean,
                )?;
                inertia += distance.powi(2);
            }
        }

        Ok(inertia)
    }

    /// Calculate information content of hidden states.
    fn calculate_information_content(&self, hidden_states: &[Vec<f64>]) -> Result<f64> {
        if hidden_states.is_empty() {
            return Ok(0.0);
        }

        let dimensions = hidden_states[0].len();
        let mut total_variance = 0.0;

        for dim in 0..dimensions {
            let values: Vec<f64> = hidden_states.iter().map(|state| state[dim]).collect();
            if values.len() > 1 {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - 1) as f64;
                total_variance += variance;
            }
        }

        // Information content as normalized variance
        Ok(total_variance / dimensions as f64)
    }

    /// Calculate temporal consistency of hidden states.
    fn calculate_temporal_consistency(&self, layer_data: &[&HiddenStateData]) -> Result<f64> {
        if layer_data.len() < 2 {
            return Ok(1.0);
        }

        let mut consistency_scores = Vec::new();

        for i in 1..layer_data.len() {
            let prev_states = &layer_data[i - 1].hidden_states;
            let curr_states = &layer_data[i].hidden_states;

            if !prev_states.is_empty() && !curr_states.is_empty() {
                // Simple consistency measure based on mean state similarity
                let prev_mean = self.calculate_mean_state(prev_states);
                let curr_mean = self.calculate_mean_state(curr_states);

                if prev_mean.len() == curr_mean.len() {
                    let distance = self.calculate_distance(
                        &prev_mean,
                        &curr_mean,
                        &DistanceMetric::Euclidean,
                    )?;
                    consistency_scores.push(1.0 / (1.0 + distance));
                }
            }
        }

        Ok(if consistency_scores.is_empty() {
            1.0
        } else {
            consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64
        })
    }

    /// Calculate mean state from a collection of states.
    fn calculate_mean_state(&self, states: &[Vec<f64>]) -> Vec<f64> {
        if states.is_empty() {
            return Vec::new();
        }

        let dimensions = states[0].len();
        let mut mean_state = vec![0.0; dimensions];

        for state in states {
            for (i, &value) in state.iter().enumerate() {
                if i < dimensions {
                    mean_state[i] += value;
                }
            }
        }

        for value in &mut mean_state {
            *value /= states.len() as f64;
        }

        mean_state
    }

    /// Calculate change rate between temporal samples.
    fn calculate_change_rate(&self, layer_data: &[&HiddenStateData]) -> Result<f64> {
        if layer_data.len() < 2 {
            return Ok(0.0);
        }

        let mut total_change = 0.0;
        let mut valid_comparisons = 0;

        for i in 1..layer_data.len() {
            let prev_mean = self.calculate_mean_state(&layer_data[i - 1].hidden_states);
            let curr_mean = self.calculate_mean_state(&layer_data[i].hidden_states);

            if !prev_mean.is_empty() && !curr_mean.is_empty() && prev_mean.len() == curr_mean.len()
            {
                let change =
                    self.calculate_distance(&prev_mean, &curr_mean, &DistanceMetric::Euclidean)?;
                total_change += change;
                valid_comparisons += 1;
            }
        }

        Ok(if valid_comparisons > 0 {
            total_change / valid_comparisons as f64
        } else {
            0.0
        })
    }

    /// Identify stability windows in temporal data.
    fn identify_stability_windows(
        &self,
        layer_data: &[&HiddenStateData],
    ) -> Result<Vec<(usize, usize)>> {
        if layer_data.len() < 3 {
            return Ok(Vec::new());
        }

        let mut stability_windows = Vec::new();
        let mut window_start = 0;
        let stability_threshold = 0.1; // Configurable threshold

        for i in 1..layer_data.len() {
            let prev_mean = self.calculate_mean_state(&layer_data[i - 1].hidden_states);
            let curr_mean = self.calculate_mean_state(&layer_data[i].hidden_states);

            if !prev_mean.is_empty() && !curr_mean.is_empty() && prev_mean.len() == curr_mean.len()
            {
                let change = self
                    .calculate_distance(&prev_mean, &curr_mean, &DistanceMetric::Euclidean)
                    .unwrap_or(f64::INFINITY);

                if change > stability_threshold {
                    // End of stability window
                    if i - window_start > 2 {
                        stability_windows.push((window_start, i - 1));
                    }
                    window_start = i;
                }
            }
        }

        // Handle final window
        if layer_data.len() - window_start > 2 {
            stability_windows.push((window_start, layer_data.len() - 1));
        }

        Ok(stability_windows)
    }

    /// Detect distribution drift in temporal data.
    fn detect_distribution_drift(&self, layer_data: &[&HiddenStateData]) -> Result<DriftInfo> {
        if layer_data.len() < self.config.temporal_analysis_window {
            return Ok(DriftInfo {
                drift_detected: false,
                drift_magnitude: 0.0,
                drift_direction: "unknown".to_string(),
                onset_step: None,
            });
        }

        let window_size = self.config.temporal_analysis_window;
        let mid_point = layer_data.len() / 2;

        // Compare early and late windows
        let early_data = &layer_data[0..window_size.min(mid_point)];
        let late_data = &layer_data[mid_point.max(layer_data.len() - window_size)..];

        let early_mean = self.calculate_aggregated_mean(early_data);
        let late_mean = self.calculate_aggregated_mean(late_data);

        if early_mean.len() == late_mean.len() && !early_mean.is_empty() {
            let drift_magnitude =
                self.calculate_distance(&early_mean, &late_mean, &DistanceMetric::Euclidean)?;
            let drift_detected = drift_magnitude > self.config.drift_detection_sensitivity;

            Ok(DriftInfo {
                drift_detected,
                drift_magnitude,
                drift_direction: if drift_detected {
                    "forward".to_string()
                } else {
                    "stable".to_string()
                },
                onset_step: if drift_detected { Some(mid_point) } else { None },
            })
        } else {
            Ok(DriftInfo {
                drift_detected: false,
                drift_magnitude: 0.0,
                drift_direction: "unknown".to_string(),
                onset_step: None,
            })
        }
    }

    /// Calculate aggregated mean across multiple data samples.
    fn calculate_aggregated_mean(&self, layer_data: &[&HiddenStateData]) -> Vec<f64> {
        let all_states: Vec<Vec<f64>> =
            layer_data.iter().flat_map(|data| data.hidden_states.iter()).cloned().collect();

        self.calculate_mean_state(&all_states)
    }

    /// Calculate stability score for representation.
    fn calculate_stability_score(&self, hidden_states: &[Vec<f64>]) -> Result<f64> {
        if hidden_states.len() < 2 {
            return Ok(1.0);
        }

        let mut stability_scores = Vec::new();
        let window_size = (hidden_states.len() / 10).max(2);

        for i in window_size..hidden_states.len() {
            let current_window = &hidden_states[i - window_size..i];
            let mean_current = self.calculate_mean_state(current_window);

            if i >= 2 * window_size {
                let prev_window = &hidden_states[i - 2 * window_size..i - window_size];
                let mean_prev = self.calculate_mean_state(prev_window);

                if mean_current.len() == mean_prev.len() && !mean_current.is_empty() {
                    let distance = self.calculate_distance(
                        &mean_current,
                        &mean_prev,
                        &DistanceMetric::Euclidean,
                    )?;
                    stability_scores.push(1.0 / (1.0 + distance));
                }
            }
        }

        Ok(if stability_scores.is_empty() {
            1.0
        } else {
            stability_scores.iter().sum::<f64>() / stability_scores.len() as f64
        })
    }

    /// Calculate variance across batches (simulated).
    fn calculate_batch_variance(&self, hidden_states: &[Vec<f64>]) -> Result<f64> {
        if hidden_states.is_empty() {
            return Ok(0.0);
        }

        let dimensions = hidden_states[0].len();
        let mut total_variance = 0.0;

        for dim in 0..dimensions {
            let values: Vec<f64> = hidden_states.iter().map(|state| state[dim]).collect();
            if values.len() > 1 {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - 1) as f64;
                total_variance += variance;
            }
        }

        Ok(total_variance / dimensions as f64)
    }

    /// Calculate consistency measure for representation.
    fn calculate_consistency_measure(&self, hidden_states: &[Vec<f64>]) -> Result<f64> {
        if hidden_states.len() < 2 {
            return Ok(1.0);
        }

        // Calculate pairwise similarities and return average
        let mut similarities = Vec::new();
        let sample_size = hidden_states.len().min(100); // Limit for performance

        for i in 0..sample_size {
            for j in (i + 1)..sample_size {
                let distance = self.calculate_distance(
                    &hidden_states[i],
                    &hidden_states[j],
                    &DistanceMetric::Cosine,
                )?;
                similarities.push(1.0 - distance); // Convert distance to similarity
            }
        }

        Ok(if similarities.is_empty() {
            1.0
        } else {
            similarities.iter().sum::<f64>() / similarities.len() as f64
        })
    }

    /// Assess robustness to noise (simulated).
    fn assess_noise_robustness(&self, hidden_states: &[Vec<f64>]) -> Result<f64> {
        // Simplified robustness assessment based on state variance
        self.calculate_batch_variance(hidden_states).map(|variance| {
            // High variance might indicate low robustness to noise
            1.0 / (1.0 + variance)
        })
    }

    /// Calculate correlation between two metrics.
    fn calculate_correlation(&self, metric1: &str, metric2: &str) -> Result<f64> {
        let data1 = self
            .performance_correlations
            .get(metric1)
            .ok_or_else(|| anyhow::anyhow!("Metric {} not found", metric1))?;

        let data2 = self
            .performance_correlations
            .get(metric2)
            .ok_or_else(|| anyhow::anyhow!("Metric {} not found", metric2))?;

        let values1: Vec<f64> = data1.values.iter().cloned().collect();
        let values2: Vec<f64> = data2.values.iter().cloned().collect();

        if values1.len() != values2.len() || values1.is_empty() {
            return Ok(0.0);
        }

        let mean1 = values1.iter().sum::<f64>() / values1.len() as f64;
        let mean2 = values2.iter().sum::<f64>() / values2.len() as f64;

        let numerator: f64 = values1
            .iter()
            .zip(values2.iter())
            .map(|(x1, x2)| (x1 - mean1) * (x2 - mean2))
            .sum();

        let var1: f64 = values1.iter().map(|x| (x - mean1).powi(2)).sum();
        let var2: f64 = values2.iter().map(|x| (x - mean2).powi(2)).sum();

        let denominator = (var1 * var2).sqrt();

        Ok(if denominator == 0.0 { 0.0 } else { numerator / denominator })
    }

    /// Generate temporal summary.
    fn generate_temporal_summary(&self) -> String {
        format!(
            "Temporal analysis: {} hidden state samples collected across {} layers. \
            Average stability observed with {} correlation metrics tracked.",
            self.hidden_states_history.len(),
            self.hidden_states_history
                .iter()
                .map(|data| &data.layer_name)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            self.performance_correlations.len()
        )
    }

    /// Generate analytics recommendations.
    fn generate_analytics_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.performance_correlations.len() < 3 {
            recommendations.push(
                "Collect more performance metrics for comprehensive correlation analysis"
                    .to_string(),
            );
        }

        if self.hidden_states_history.len() < 50 {
            recommendations
                .push("Increase hidden state sampling for better temporal analysis".to_string());
        }

        recommendations
            .push("Consider implementing automated anomaly detection alerts".to_string());
        recommendations.push("Enable advanced visualization for better insights".to_string());

        recommendations
    }
}

/// Comprehensive analytics report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnalyticsReport {
    /// Correlation matrix between metrics
    pub correlation_matrix: Vec<Vec<f64>>,
    /// Statistical analysis results
    pub statistical_analysis: StatisticalAnalysis,
    /// Layer-specific analyses
    pub layer_analyses: HashMap<String, HiddenStateAnalysis>,
    /// Anomaly detection results
    pub anomaly_detection: AnomalyDetectionResults,
    /// Temporal analysis summary
    pub temporal_summary: String,
    /// Analytics recommendations
    pub recommendations: Vec<String>,
}

impl TemporalAnalysisCache {
    /// Create a new temporal analysis cache.
    fn new() -> Self {
        Self {
            drift_results: HashMap::new(),
            consistency_scores: HashMap::new(),
            stability_windows: HashMap::new(),
            last_analysis: chrono::Utc::now(),
        }
    }
}

impl Default for StatisticalAnalysis {
    fn default() -> Self {
        Self {
            means: Vec::new(),
            std_devs: Vec::new(),
            correlation_matrix: Vec::new(),
            principal_components: Vec::new(),
            explained_variance_ratios: Vec::new(),
            significance_tests: Vec::new(),
        }
    }
}

impl Default for AnomalyDetectionResults {
    fn default() -> Self {
        Self {
            anomalies: Vec::new(),
            anomaly_scores: Vec::new(),
            threshold: 0.0,
            method: AnomalyDetectionMethod::StatisticalThreshold { n_std: 2.0 },
        }
    }
}

impl Default for AdvancedAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_analytics_creation() {
        let analytics = AdvancedAnalytics::new();
        assert_eq!(analytics.hidden_states_history.len(), 0);
        assert_eq!(analytics.performance_correlations.len(), 0);
    }

    #[test]
    fn test_distance_calculation() {
        let analytics = AdvancedAnalytics::new();
        let point1 = vec![1.0, 2.0, 3.0];
        let point2 = vec![4.0, 5.0, 6.0];

        let distance = analytics
            .calculate_distance(&point1, &point2, &DistanceMetric::Euclidean)
            .unwrap();
        assert!(distance > 0.0);
    }

    #[test]
    fn test_clustering_parameters() {
        let params = ClusteringParameters::default();
        assert_eq!(params.num_clusters, 8);
        assert_eq!(params.max_iterations, 100);
    }

    #[test]
    fn test_correlation_calculation() {
        let mut analytics = AdvancedAnalytics::new();

        // Add some test data
        analytics.update_correlation_data("metric1", 1.0);
        analytics.update_correlation_data("metric1", 2.0);
        analytics.update_correlation_data("metric2", 3.0);
        analytics.update_correlation_data("metric2", 4.0);

        assert_eq!(analytics.performance_correlations.len(), 2);
    }
}
