//! Behavior Analysis
//!
//! Advanced analysis tools for understanding neural network behavior including
//! input sensitivity, feature importance, and neuron activation patterns.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for behavior analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorAnalysisConfig {
    /// Enable input sensitivity analysis
    pub enable_input_sensitivity: bool,
    /// Enable feature importance calculations
    pub enable_feature_importance: bool,
    /// Enable neuron activation pattern analysis
    pub enable_activation_patterns: bool,
    /// Enable dead neuron detection
    pub enable_dead_neuron_detection: bool,
    /// Enable correlation analysis
    pub enable_correlation_analysis: bool,
    /// Threshold for dead neuron detection (activation below this value)
    pub dead_neuron_threshold: f32,
    /// Number of samples for sensitivity analysis
    pub sensitivity_samples: usize,
    /// Perturbation magnitude for sensitivity analysis
    pub perturbation_magnitude: f32,
    /// Correlation threshold for significance
    pub correlation_threshold: f32,
}

impl Default for BehaviorAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_input_sensitivity: true,
            enable_feature_importance: true,
            enable_activation_patterns: true,
            enable_dead_neuron_detection: true,
            enable_correlation_analysis: true,
            dead_neuron_threshold: 1e-6,
            sensitivity_samples: 100,
            perturbation_magnitude: 0.01,
            correlation_threshold: 0.5,
        }
    }
}

/// Input sensitivity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSensitivity {
    pub input_dimension: usize,
    pub sensitivity_score: f32,
    pub gradient_magnitude: f32,
    pub perturbation_impact: f32,
    pub rank: usize,
}

/// Feature importance analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    pub feature_id: String,
    pub importance_score: f32,
    pub attribution_method: AttributionMethod,
    pub confidence: f32,
    pub rank: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributionMethod {
    GradientBased,
    PermutationImportance,
    ShapleySampling,
    IntegratedGradients,
    LimeApproximation,
}

/// Neuron activation pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronActivationPattern {
    pub layer_id: String,
    pub neuron_id: usize,
    pub activation_statistics: ActivationStatistics,
    pub pattern_type: ActivationPatternType,
    pub stability_score: f32,
    pub selectivity_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationStatistics {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub percentile_25: f32,
    pub percentile_75: f32,
    pub skewness: f32,
    pub kurtosis: f32,
    pub sparsity: f32, // Fraction of near-zero activations
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationPatternType {
    Normal,
    Saturated,
    Dead,
    Oscillating,
    Sparse,
    Dense,
    Bipolar,
}

/// Dead neuron detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadNeuronInfo {
    pub layer_id: String,
    pub neuron_id: usize,
    pub activation_level: f32,
    pub dead_probability: f32,
    pub suggested_action: NeuronRepairAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuronRepairAction {
    Reinitialize,
    AdjustLearningRate,
    ChangeActivationFunction,
    AddNoise,
    Skip, // Neuron is functioning normally
}

/// Correlation analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysis {
    pub correlation_matrix: Vec<Vec<f32>>,
    pub significant_correlations: Vec<CorrelationPair>,
    pub redundant_features: Vec<FeatureGroup>,
    pub independent_features: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationPair {
    pub feature_a: usize,
    pub feature_b: usize,
    pub correlation: f32,
    pub p_value: f32,
    pub relationship_type: CorrelationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationType {
    Strong,
    Moderate,
    Weak,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGroup {
    pub features: Vec<usize>,
    pub average_correlation: f32,
    pub group_importance: f32,
}

/// Comprehensive behavior analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorAnalysisReport {
    pub input_sensitivities: Vec<InputSensitivity>,
    pub feature_importances: Vec<FeatureImportance>,
    pub activation_patterns: Vec<NeuronActivationPattern>,
    pub dead_neurons: Vec<DeadNeuronInfo>,
    pub correlation_analysis: Option<CorrelationAnalysis>,
    pub behavior_summary: BehaviorSummary,
    pub recommendations: Vec<BehaviorRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorSummary {
    pub total_neurons_analyzed: usize,
    pub dead_neuron_percentage: f32,
    pub average_activation_sparsity: f32,
    pub feature_distribution_entropy: f32,
    pub model_stability_score: f32,
    pub interpretability_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorRecommendation {
    pub category: RecommendationCategory,
    pub priority: Priority,
    pub description: String,
    pub implementation: String,
    pub expected_impact: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Architecture,
    Training,
    Initialization,
    Regularization,
    DataPreprocessing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

/// Behavior analyzer
#[derive(Debug)]
pub struct BehaviorAnalyzer {
    config: BehaviorAnalysisConfig,
    activation_history: HashMap<String, Vec<Vec<f32>>>,
    input_gradients: HashMap<String, Vec<f32>>,
    feature_attributions: HashMap<String, FeatureImportance>,
    analysis_cache: HashMap<String, BehaviorAnalysisReport>,
}

impl BehaviorAnalyzer {
    /// Create a new behavior analyzer
    pub fn new(config: BehaviorAnalysisConfig) -> Self {
        Self {
            config,
            activation_history: HashMap::new(),
            input_gradients: HashMap::new(),
            feature_attributions: HashMap::new(),
            analysis_cache: HashMap::new(),
        }
    }

    /// Record neuron activations for analysis
    pub fn record_activations(&mut self, layer_id: String, activations: Vec<f32>) {
        self.activation_history.entry(layer_id).or_default().push(activations);
    }

    /// Record input gradients for sensitivity analysis
    pub fn record_input_gradients(&mut self, input_id: String, gradients: Vec<f32>) {
        self.input_gradients.insert(input_id, gradients);
    }

    /// Perform comprehensive behavior analysis
    pub async fn analyze(&mut self) -> Result<BehaviorAnalysisReport> {
        let mut report = BehaviorAnalysisReport {
            input_sensitivities: Vec::new(),
            feature_importances: Vec::new(),
            activation_patterns: Vec::new(),
            dead_neurons: Vec::new(),
            correlation_analysis: None,
            behavior_summary: BehaviorSummary {
                total_neurons_analyzed: 0,
                dead_neuron_percentage: 0.0,
                average_activation_sparsity: 0.0,
                feature_distribution_entropy: 0.0,
                model_stability_score: 0.0,
                interpretability_score: 0.0,
            },
            recommendations: Vec::new(),
        };

        if self.config.enable_input_sensitivity {
            report.input_sensitivities = self.analyze_input_sensitivity().await?;
        }

        if self.config.enable_feature_importance {
            report.feature_importances = self.calculate_feature_importance().await?;
        }

        if self.config.enable_activation_patterns {
            report.activation_patterns = self.analyze_activation_patterns().await?;
        }

        if self.config.enable_dead_neuron_detection {
            report.dead_neurons = self.detect_dead_neurons().await?;
        }

        if self.config.enable_correlation_analysis {
            report.correlation_analysis = Some(self.perform_correlation_analysis().await?);
        }

        self.generate_behavior_summary(&mut report);
        self.generate_recommendations(&mut report);

        Ok(report)
    }

    /// Analyze input sensitivity using gradient-based methods
    async fn analyze_input_sensitivity(&self) -> Result<Vec<InputSensitivity>> {
        let mut sensitivities = Vec::new();

        for gradients in self.input_gradients.values() {
            for (dim, &gradient) in gradients.iter().enumerate() {
                let sensitivity_score = gradient.abs();
                let gradient_magnitude = gradient.abs();

                // Simulate perturbation impact (would normally require model re-evaluation)
                let perturbation_impact = self.estimate_perturbation_impact(gradient, dim);

                sensitivities.push(InputSensitivity {
                    input_dimension: dim,
                    sensitivity_score,
                    gradient_magnitude,
                    perturbation_impact,
                    rank: 0, // Will be set after sorting
                });
            }
        }

        // Sort by sensitivity score and assign ranks
        sensitivities
            .sort_by(|a, b| b.sensitivity_score.partial_cmp(&a.sensitivity_score).unwrap());
        for (rank, sensitivity) in sensitivities.iter_mut().enumerate() {
            sensitivity.rank = rank + 1;
        }

        Ok(sensitivities)
    }

    /// Estimate perturbation impact (simplified version)
    fn estimate_perturbation_impact(&self, gradient: f32, _dimension: usize) -> f32 {
        // Simplified estimation: perturbation impact is proportional to gradient magnitude
        gradient.abs() * self.config.perturbation_magnitude
    }

    /// Calculate feature importance using multiple methods
    async fn calculate_feature_importance(&self) -> Result<Vec<FeatureImportance>> {
        let mut importances = Vec::new();

        // Gradient-based importance
        for (input_id, gradients) in &self.input_gradients {
            let total_gradient = gradients.iter().map(|g| g.abs()).sum::<f32>();
            let importance_score = total_gradient / gradients.len() as f32;

            importances.push(FeatureImportance {
                feature_id: input_id.clone(),
                importance_score,
                attribution_method: AttributionMethod::GradientBased,
                confidence: self.calculate_attribution_confidence(importance_score),
                rank: 0,
            });
        }

        // Sort by importance and assign ranks
        importances.sort_by(|a, b| b.importance_score.partial_cmp(&a.importance_score).unwrap());
        for (rank, importance) in importances.iter_mut().enumerate() {
            importance.rank = rank + 1;
        }

        Ok(importances)
    }

    /// Calculate confidence in attribution score
    fn calculate_attribution_confidence(&self, score: f32) -> f32 {
        // Simple confidence based on score magnitude
        (score.tanh() * 0.5 + 0.5).min(1.0)
    }

    /// Analyze neuron activation patterns
    async fn analyze_activation_patterns(&self) -> Result<Vec<NeuronActivationPattern>> {
        let mut patterns = Vec::new();

        for (layer_id, activation_history) in &self.activation_history {
            if activation_history.is_empty() {
                continue;
            }

            let neuron_count = activation_history[0].len();

            for neuron_id in 0..neuron_count {
                let neuron_activations: Vec<f32> = activation_history
                    .iter()
                    .map(|batch| batch.get(neuron_id).copied().unwrap_or(0.0))
                    .collect();

                let statistics = self.compute_activation_statistics(&neuron_activations);
                let pattern_type = self.classify_activation_pattern(&statistics);
                let stability_score = self.compute_stability_score(&neuron_activations);
                let selectivity_score = self.compute_selectivity_score(&neuron_activations);

                patterns.push(NeuronActivationPattern {
                    layer_id: layer_id.clone(),
                    neuron_id,
                    activation_statistics: statistics,
                    pattern_type,
                    stability_score,
                    selectivity_score,
                });
            }
        }

        Ok(patterns)
    }

    /// Compute detailed activation statistics
    fn compute_activation_statistics(&self, activations: &[f32]) -> ActivationStatistics {
        if activations.is_empty() {
            return ActivationStatistics {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                percentile_25: 0.0,
                percentile_75: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
                sparsity: 1.0,
            };
        }

        let mean = activations.iter().sum::<f32>() / activations.len() as f32;
        let variance =
            activations.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / activations.len() as f32;
        let std = variance.sqrt();

        let mut sorted_activations = activations.to_vec();
        sorted_activations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_activations[0];
        let max = sorted_activations[sorted_activations.len() - 1];
        let percentile_25 = sorted_activations[sorted_activations.len() / 4];
        let percentile_75 = sorted_activations[3 * sorted_activations.len() / 4];

        // Calculate skewness and kurtosis
        let skewness = if std > 0.0 {
            activations.iter().map(|&x| ((x - mean) / std).powi(3)).sum::<f32>()
                / activations.len() as f32
        } else {
            0.0
        };

        let kurtosis = if std > 0.0 {
            activations.iter().map(|&x| ((x - mean) / std).powi(4)).sum::<f32>()
                / activations.len() as f32
                - 3.0
        } else {
            0.0
        };

        // Calculate sparsity (fraction of near-zero activations)
        let near_zero_count = activations
            .iter()
            .filter(|&&x| x.abs() < self.config.dead_neuron_threshold)
            .count();
        let sparsity = near_zero_count as f32 / activations.len() as f32;

        ActivationStatistics {
            mean,
            std,
            min,
            max,
            percentile_25,
            percentile_75,
            skewness,
            kurtosis,
            sparsity,
        }
    }

    /// Classify activation pattern type
    fn classify_activation_pattern(&self, stats: &ActivationStatistics) -> ActivationPatternType {
        if stats.sparsity > 0.9 {
            ActivationPatternType::Dead
        } else if stats.sparsity > 0.7 {
            ActivationPatternType::Sparse
        } else if stats.max > 0.95 && stats.mean > 0.8 {
            ActivationPatternType::Saturated
        } else if stats.std / stats.mean.abs().max(1e-8) > 2.0 {
            ActivationPatternType::Oscillating
        } else if stats.mean.abs() > 0.1 && stats.mean * stats.min < 0.0 {
            ActivationPatternType::Bipolar
        } else if stats.sparsity < 0.3 {
            ActivationPatternType::Dense
        } else {
            ActivationPatternType::Normal
        }
    }

    /// Compute stability score for neuron activations
    fn compute_stability_score(&self, activations: &[f32]) -> f32 {
        if activations.len() < 2 {
            return 0.0;
        }

        let mean = activations.iter().sum::<f32>() / activations.len() as f32;
        let variance =
            activations.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / activations.len() as f32;

        // Stability is inverse of coefficient of variation
        if mean.abs() > 1e-8 {
            1.0 / (1.0 + variance.sqrt() / mean.abs())
        } else {
            0.0
        }
    }

    /// Compute selectivity score (how selective the neuron is)
    fn compute_selectivity_score(&self, activations: &[f32]) -> f32 {
        if activations.is_empty() {
            return 0.0;
        }

        // Selectivity based on activation distribution
        let max_activation = activations.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let mean_activation =
            activations.iter().map(|x| x.abs()).sum::<f32>() / activations.len() as f32;

        if max_activation > 1e-8 {
            1.0 - (mean_activation / max_activation)
        } else {
            0.0
        }
    }

    /// Detect dead neurons
    async fn detect_dead_neurons(&self) -> Result<Vec<DeadNeuronInfo>> {
        let mut dead_neurons = Vec::new();

        for (layer_id, activation_history) in &self.activation_history {
            if activation_history.is_empty() {
                continue;
            }

            let neuron_count = activation_history[0].len();

            for neuron_id in 0..neuron_count {
                let neuron_activations: Vec<f32> = activation_history
                    .iter()
                    .map(|batch| batch.get(neuron_id).copied().unwrap_or(0.0))
                    .collect();

                let activation_level = neuron_activations.iter().map(|x| x.abs()).sum::<f32>()
                    / neuron_activations.len() as f32;

                let dead_probability = if activation_level < self.config.dead_neuron_threshold {
                    1.0 - (activation_level / self.config.dead_neuron_threshold)
                } else {
                    0.0
                };

                if dead_probability > 0.5 {
                    let suggested_action =
                        self.suggest_neuron_repair_action(activation_level, &neuron_activations);

                    dead_neurons.push(DeadNeuronInfo {
                        layer_id: layer_id.clone(),
                        neuron_id,
                        activation_level,
                        dead_probability,
                        suggested_action,
                    });
                }
            }
        }

        Ok(dead_neurons)
    }

    /// Suggest repair action for dead neurons
    fn suggest_neuron_repair_action(
        &self,
        activation_level: f32,
        activations: &[f32],
    ) -> NeuronRepairAction {
        if activation_level < self.config.dead_neuron_threshold * 0.1 {
            NeuronRepairAction::Reinitialize
        } else if activation_level < self.config.dead_neuron_threshold * 0.5 {
            let variance =
                activations.iter().map(|&x| x.powi(2)).sum::<f32>() / activations.len() as f32;
            if variance < 1e-10 {
                NeuronRepairAction::AddNoise
            } else {
                NeuronRepairAction::AdjustLearningRate
            }
        } else {
            NeuronRepairAction::ChangeActivationFunction
        }
    }

    /// Perform correlation analysis
    async fn perform_correlation_analysis(&self) -> Result<CorrelationAnalysis> {
        // For simplification, we'll analyze correlations between input gradients
        let gradient_vectors: Vec<&Vec<f32>> = self.input_gradients.values().collect();

        if gradient_vectors.len() < 2 {
            return Ok(CorrelationAnalysis {
                correlation_matrix: Vec::new(),
                significant_correlations: Vec::new(),
                redundant_features: Vec::new(),
                independent_features: Vec::new(),
            });
        }

        let n = gradient_vectors.len();
        let mut correlation_matrix = vec![vec![0.0; n]; n];
        let mut significant_correlations = Vec::new();

        // Compute correlation matrix
        for i in 0..n {
            for j in i..n {
                let correlation =
                    self.compute_correlation(gradient_vectors[i], gradient_vectors[j]);
                correlation_matrix[i][j] = correlation;
                correlation_matrix[j][i] = correlation;

                if i != j && correlation.abs() > self.config.correlation_threshold {
                    let correlation_type = if correlation.abs() > 0.8 {
                        CorrelationType::Strong
                    } else if correlation.abs() > 0.5 {
                        CorrelationType::Moderate
                    } else {
                        CorrelationType::Weak
                    };

                    significant_correlations.push(CorrelationPair {
                        feature_a: i,
                        feature_b: j,
                        correlation,
                        p_value: 0.01, // Simplified p-value
                        relationship_type: correlation_type,
                    });
                }
            }
        }

        // Find redundant features (groups of highly correlated features)
        let redundant_features = self.find_redundant_feature_groups(&correlation_matrix);

        // Find independent features
        let independent_features = self.find_independent_features(&correlation_matrix);

        Ok(CorrelationAnalysis {
            correlation_matrix,
            significant_correlations,
            redundant_features,
            independent_features,
        })
    }

    /// Compute Pearson correlation coefficient
    fn compute_correlation(&self, x: &[f32], y: &[f32]) -> f32 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f32;
        let mean_x = x.iter().sum::<f32>() / n;
        let mean_y = y.iter().sum::<f32>() / n;

        let numerator: f32 =
            x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y)).sum();

        let sum_sq_x: f32 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f32 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator > 1e-8 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Find groups of redundant features
    fn find_redundant_feature_groups(&self, correlation_matrix: &[Vec<f32>]) -> Vec<FeatureGroup> {
        let mut groups = Vec::new();
        let mut visited = HashSet::new();

        for i in 0..correlation_matrix.len() {
            if visited.contains(&i) {
                continue;
            }

            let mut group = vec![i];
            let mut group_correlations = Vec::new();

            for j in (i + 1)..correlation_matrix.len() {
                if correlation_matrix[i][j].abs() > 0.7 {
                    group.push(j);
                    group_correlations.push(correlation_matrix[i][j].abs());
                    visited.insert(j);
                }
            }

            if group.len() > 1 {
                let average_correlation =
                    group_correlations.iter().sum::<f32>() / group_correlations.len() as f32;
                groups.push(FeatureGroup {
                    features: group,
                    average_correlation,
                    group_importance: average_correlation, // Simplified importance
                });
            }

            visited.insert(i);
        }

        groups
    }

    /// Find independent features
    fn find_independent_features(&self, correlation_matrix: &[Vec<f32>]) -> Vec<usize> {
        let mut independent = Vec::new();

        for i in 0..correlation_matrix.len() {
            let max_correlation = correlation_matrix[i]
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &corr)| corr.abs())
                .fold(0.0f32, |a, b| a.max(b));

            if max_correlation < self.config.correlation_threshold {
                independent.push(i);
            }
        }

        independent
    }

    /// Generate behavior summary
    fn generate_behavior_summary(&self, report: &mut BehaviorAnalysisReport) {
        let total_neurons = report.activation_patterns.len();
        let dead_neurons = report.dead_neurons.len();

        report.behavior_summary.total_neurons_analyzed = total_neurons;
        report.behavior_summary.dead_neuron_percentage = if total_neurons > 0 {
            (dead_neurons as f32 / total_neurons as f32) * 100.0
        } else {
            0.0
        };

        if !report.activation_patterns.is_empty() {
            report.behavior_summary.average_activation_sparsity = report
                .activation_patterns
                .iter()
                .map(|p| p.activation_statistics.sparsity)
                .sum::<f32>()
                / report.activation_patterns.len() as f32;

            report.behavior_summary.model_stability_score =
                report.activation_patterns.iter().map(|p| p.stability_score).sum::<f32>()
                    / report.activation_patterns.len() as f32;
        }

        // Simple entropy calculation for feature distribution
        if !report.feature_importances.is_empty() {
            let total_importance: f32 =
                report.feature_importances.iter().map(|f| f.importance_score).sum();

            if total_importance > 0.0 {
                let entropy: f32 = report
                    .feature_importances
                    .iter()
                    .map(|f| {
                        let p = f.importance_score / total_importance;
                        if p > 0.0 {
                            -p * p.log2()
                        } else {
                            0.0
                        }
                    })
                    .sum();
                report.behavior_summary.feature_distribution_entropy = entropy;
            }
        }

        // Overall interpretability score
        report.behavior_summary.interpretability_score =
            (report.behavior_summary.model_stability_score * 0.4
                + (1.0 - report.behavior_summary.dead_neuron_percentage / 100.0) * 0.3
                + (1.0 - report.behavior_summary.average_activation_sparsity) * 0.3)
                .max(0.0)
                .min(1.0);
    }

    /// Generate behavior recommendations
    fn generate_recommendations(&self, report: &mut BehaviorAnalysisReport) {
        // Dead neuron recommendations
        if report.behavior_summary.dead_neuron_percentage > 20.0 {
            report.recommendations.push(BehaviorRecommendation {
                category: RecommendationCategory::Training,
                priority: Priority::Critical,
                description: format!("High percentage of dead neurons detected ({:.1}%)",
                                   report.behavior_summary.dead_neuron_percentage),
                implementation: "Consider reducing learning rate, changing initialization, or adding batch normalization".to_string(),
                expected_impact: 0.8,
            });
        }

        // Sparsity recommendations
        if report.behavior_summary.average_activation_sparsity > 0.8 {
            report.recommendations.push(BehaviorRecommendation {
                category: RecommendationCategory::Architecture,
                priority: Priority::High,
                description: "Very sparse activations detected, model may be under-utilized".to_string(),
                implementation: "Consider reducing model capacity or adjusting activation functions".to_string(),
                expected_impact: 0.6,
            });
        }

        // Stability recommendations
        if report.behavior_summary.model_stability_score < 0.5 {
            report.recommendations.push(BehaviorRecommendation {
                category: RecommendationCategory::Training,
                priority: Priority::High,
                description: "Low model stability detected".to_string(),
                implementation: "Consider adding regularization, reducing learning rate, or using gradient clipping".to_string(),
                expected_impact: 0.7,
            });
        }

        // Feature importance recommendations
        if report.feature_importances.len() > 10 {
            let top_features = &report.feature_importances[..5];
            let bottom_features =
                &report.feature_importances[report.feature_importances.len() - 5..];

            let top_importance: f32 = top_features.iter().map(|f| f.importance_score).sum();
            let bottom_importance: f32 = bottom_features.iter().map(|f| f.importance_score).sum();

            if top_importance > bottom_importance * 10.0 {
                report.recommendations.push(BehaviorRecommendation {
                    category: RecommendationCategory::DataPreprocessing,
                    priority: Priority::Medium,
                    description: "Highly imbalanced feature importance detected".to_string(),
                    implementation: "Consider feature selection or dimensionality reduction"
                        .to_string(),
                    expected_impact: 0.5,
                });
            }
        }
    }

    /// Generate a comprehensive report
    pub async fn generate_report(&self) -> Result<BehaviorAnalysisReport> {
        let mut temp_analyzer = BehaviorAnalyzer {
            config: self.config.clone(),
            activation_history: self.activation_history.clone(),
            input_gradients: self.input_gradients.clone(),
            feature_attributions: self.feature_attributions.clone(),
            analysis_cache: HashMap::new(),
        };

        temp_analyzer.analyze().await
    }

    /// Clear all recorded data
    pub fn clear(&mut self) {
        self.activation_history.clear();
        self.input_gradients.clear();
        self.feature_attributions.clear();
        self.analysis_cache.clear();
    }

    /// Get summary of current analysis state
    pub fn get_analysis_summary(&self) -> AnalysisSummary {
        AnalysisSummary {
            total_layers_tracked: self.activation_history.len(),
            total_activation_samples: self
                .activation_history
                .values()
                .map(|history| history.len())
                .sum(),
            total_inputs_tracked: self.input_gradients.len(),
            analysis_coverage: if self.activation_history.is_empty() {
                0.0
            } else {
                1.0 // Simplified coverage metric
            },
        }
    }
}

/// Summary of analysis state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisSummary {
    pub total_layers_tracked: usize,
    pub total_activation_samples: usize,
    pub total_inputs_tracked: usize,
    pub analysis_coverage: f32,
}
