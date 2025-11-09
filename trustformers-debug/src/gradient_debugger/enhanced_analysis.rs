//! Enhanced Layer Analysis and Network-Level Insights
//!
//! This module provides comprehensive enhanced analysis capabilities including
//! detailed layer-wise analysis, network-level gradient insights, and optimization
//! priority ranking for gradient debugging.

use super::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Enhanced layer-wise gradient analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedLayerGradientAnalysis {
    pub layer_details: HashMap<String, LayerGradientDetails>,
    pub network_level_analysis: NetworkLevelAnalysis,
    pub gradient_hierarchy: GradientHierarchy,
    pub optimization_priorities: Vec<OptimizationPriority>,
}

/// Detailed gradient analysis for a specific layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGradientDetails {
    pub layer_name: String,
    pub gradient_statistics: GradientStatistics,
    pub flow_characteristics: FlowCharacteristics,
    pub health_metrics: LayerHealthMetrics,
    pub optimization_suggestions: Vec<LayerOptimizationSuggestion>,
    pub comparative_analysis: ComparativeAnalysis,
}

/// Network-level gradient analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLevelAnalysis {
    pub overall_gradient_health: LayerHealth,
    pub gradient_distribution: GradientDistribution,
    pub layer_interactions: Vec<LayerInteraction>,
    pub convergence_indicators: ConvergenceIndicators,
    pub training_dynamics: TrainingDynamics,
    pub stability_assessment: StabilityAssessment,
}

/// Distribution of gradients across the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientDistribution {
    pub mean_gradient_norm: f64,
    pub gradient_variance: f64,
    pub gradient_skewness: f64,
    pub gradient_kurtosis: f64,
    pub layer_gradient_ratios: HashMap<String, f64>,
    pub distribution_type: DistributionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    Normal,
    Skewed,
    HeavyTailed,
    Multimodal,
    Degenerate,
}

/// Interaction between layers in gradient flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInteraction {
    pub layer1: String,
    pub layer2: String,
    pub interaction_strength: f64,
    pub interaction_type: InteractionType,
    pub impact_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    Cooperative,
    Competitive,
    Neutral,
    Disruptive,
}

/// Indicators of training convergence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceIndicators {
    pub gradient_convergence_score: f64,
    pub parameter_convergence_score: f64,
    pub loss_convergence_score: f64,
    pub convergence_trend: ConvergenceTrend,
    pub estimated_steps_to_convergence: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceTrend {
    Converging,
    Stable,
    Diverging,
    Oscillating,
    Unknown,
}

/// Training dynamics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDynamics {
    pub learning_phase: LearningPhase,
    pub gradient_momentum: f64,
    pub learning_velocity: f64,
    pub adaptation_rate: f64,
    pub plateau_detection: PlateauDetection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningPhase {
    InitialLearning,
    RapidLearning,
    Refinement,
    Convergence,
    Plateau,
    Overfitting,
}

/// Plateau detection in training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateauDetection {
    pub is_plateau: bool,
    pub plateau_duration: usize,
    pub plateau_severity: PlateauSeverity,
    pub suggested_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlateauSeverity {
    Mild,
    Moderate,
    Severe,
}

/// Network stability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityAssessment {
    pub overall_stability: f64,
    pub stability_trend: StabilityTrend,
    pub instability_sources: Vec<InstabilitySource>,
    pub stability_forecast: StabilityForecast,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StabilityTrend {
    Improving,
    Stable,
    Degrading,
}

/// Source of instability in the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstabilitySource {
    pub source_type: InstabilityType,
    pub affected_layers: Vec<String>,
    pub severity: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstabilityType {
    GradientExplosion,
    GradientVanishing,
    Oscillation,
    Stagnation,
    Chaos,
}

/// Forecast of stability trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityForecast {
    pub short_term_outlook: StabilityOutlook,
    pub long_term_outlook: StabilityOutlook,
    pub confidence_level: f64,
    pub recommended_monitoring: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StabilityOutlook {
    Stable,
    Improving,
    Deteriorating,
    Uncertain,
}

/// Hierarchical organization of gradient information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientHierarchy {
    pub layer_groups: Vec<LayerGroup>,
    pub hierarchy_levels: Vec<HierarchyLevel>,
    pub cross_level_interactions: Vec<CrossLevelInteraction>,
}

/// Group of related layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGroup {
    pub group_name: String,
    pub layers: Vec<String>,
    pub group_characteristics: GroupCharacteristics,
    pub internal_coherence: f64,
}

/// Characteristics of a layer group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupCharacteristics {
    pub average_gradient_norm: f64,
    pub gradient_synchronization: f64,
    pub learning_rate_sensitivity: f64,
    pub optimization_difficulty: OptimizationDifficulty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationDifficulty {
    Easy,
    Moderate,
    Difficult,
    VeryDifficult,
}

/// Level in the gradient hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyLevel {
    pub level_id: usize,
    pub level_name: String,
    pub layer_groups: Vec<String>,
    pub level_importance: f64,
    pub optimization_impact: f64,
}

/// Interaction between hierarchy levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLevelInteraction {
    pub from_level: usize,
    pub to_level: usize,
    pub interaction_strength: f64,
    pub interaction_direction: InteractionDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionDirection {
    TopDown,
    BottomUp,
    Bidirectional,
}

/// Optimization priority for layers or groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPriority {
    pub target_name: String,
    pub target_type: OptimizationTarget,
    pub priority_score: f64,
    pub urgency_level: UrgencyLevel,
    pub optimization_potential: f64,
    pub recommended_actions: Vec<PrioritizedAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTarget {
    IndividualLayer,
    LayerGroup,
    NetworkLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Prioritized optimization action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritizedAction {
    pub action_name: String,
    pub action_type: ActionType,
    pub expected_impact: f64,
    pub implementation_effort: ImplementationEffort,
    pub prerequisites: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    ParameterAdjustment,
    ArchitecturalChange,
    OptimizationTechnique,
    RegularizationMethod,
    LearningRateScheduling,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Minimal,
    Low,
    Moderate,
    High,
    Extensive,
}

/// Layer-specific optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerOptimizationSuggestion {
    pub suggestion_type: SuggestionType,
    pub description: String,
    pub rationale: String,
    pub expected_improvement: f64,
    pub implementation_complexity: ImplementationComplexity,
    pub side_effects: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionType {
    WeightInitialization,
    LearningRateAdjustment,
    RegularizationAdd,
    ArchitecturalModification,
    OptimizationAlgorithm,
    BatchNormalization,
    DropoutAdjustment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationComplexity {
    Simple,
    Moderate,
    Complex,
    RequiresRetraining,
}

/// Enhanced gradient analyzer
#[derive(Debug)]
pub struct EnhancedGradientAnalyzer {
    #[allow(dead_code)]
    analysis_depth: AnalysisDepth,
    convergence_window: usize,
    stability_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum AnalysisDepth {
    Basic,
    Standard,
    Comprehensive,
    Expert,
}

impl Default for EnhancedGradientAnalyzer {
    fn default() -> Self {
        Self {
            analysis_depth: AnalysisDepth::Standard,
            convergence_window: 100,
            stability_threshold: 0.8,
        }
    }
}

impl EnhancedGradientAnalyzer {
    pub fn new(depth: AnalysisDepth, window: usize, threshold: f64) -> Self {
        Self {
            analysis_depth: depth,
            convergence_window: window,
            stability_threshold: threshold,
        }
    }

    pub fn generate_enhanced_analysis(
        &self,
        gradient_histories: &HashMap<String, GradientHistory>,
    ) -> EnhancedLayerGradientAnalysis {
        let layer_details = self.generate_layer_details(gradient_histories);
        let network_level_analysis = self.analyze_network_level_gradients(&layer_details);
        let gradient_hierarchy = self.build_gradient_hierarchy(&layer_details);
        let optimization_priorities =
            self.rank_optimization_priorities(&layer_details, &network_level_analysis);

        EnhancedLayerGradientAnalysis {
            layer_details,
            network_level_analysis,
            gradient_hierarchy,
            optimization_priorities,
        }
    }

    fn generate_layer_details(
        &self,
        gradient_histories: &HashMap<String, GradientHistory>,
    ) -> HashMap<String, LayerGradientDetails> {
        let mut layer_details = HashMap::new();

        for (layer_name, history) in gradient_histories {
            let gradient_statistics = self.compute_detailed_gradient_stats(history);
            let flow_characteristics = self.analyze_flow_characteristics(history);
            let health_metrics = self.compute_layer_health_metrics(history);
            let optimization_suggestions =
                self.generate_layer_optimization_suggestions(layer_name, history);
            let comparative_analysis =
                self.compare_with_other_layers(layer_name, history, gradient_histories);

            let analysis = LayerGradientDetails {
                layer_name: layer_name.clone(),
                gradient_statistics,
                flow_characteristics,
                health_metrics,
                optimization_suggestions,
                comparative_analysis,
            };

            layer_details.insert(layer_name.clone(), analysis);
        }

        layer_details
    }

    fn compute_detailed_gradient_stats(&self, history: &GradientHistory) -> GradientStatistics {
        if history.gradient_norms.is_empty() {
            return GradientStatistics {
                mean: 0.0,
                std: 0.0,
                median: 0.0,
                percentile_95: 0.0,
                percentile_5: 0.0,
                samples: 0,
                variance: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
            };
        }

        let values: Vec<f64> = history.gradient_norms.iter().cloned().collect();
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_idx = values.len() / 2;
        let median = if values.len() % 2 == 0 {
            (sorted_values[median_idx - 1] + sorted_values[median_idx]) / 2.0
        } else {
            sorted_values[median_idx]
        };

        let percentile_5_idx = (values.len() as f64 * 0.05) as usize;
        let percentile_95_idx = (values.len() as f64 * 0.95) as usize;
        let percentile_5 = sorted_values[percentile_5_idx];
        let percentile_95 = sorted_values[percentile_95_idx.min(sorted_values.len() - 1)];

        // Compute skewness and kurtosis
        let skewness = if std > 0.0 {
            values.iter().map(|&x| ((x - mean) / std).powi(3)).sum::<f64>() / n
        } else {
            0.0
        };

        let kurtosis = if std > 0.0 {
            values.iter().map(|&x| ((x - mean) / std).powi(4)).sum::<f64>() / n - 3.0
        } else {
            0.0
        };

        GradientStatistics {
            mean,
            std,
            median,
            percentile_95,
            percentile_5,
            samples: values.len(),
            variance,
            skewness,
            kurtosis,
        }
    }

    fn analyze_flow_characteristics(&self, history: &GradientHistory) -> FlowCharacteristics {
        let consistency_score = self.compute_flow_consistency(history);
        let smoothness_index = self.compute_smoothness_index(history);
        let trend_strength = self.compute_trend_strength(history);
        let oscillation_frequency = self.compute_oscillation_frequency(history);
        let stability_measure = self.compute_stability_measure(history);

        FlowCharacteristics {
            consistency_score,
            smoothness_index,
            trend_strength,
            oscillation_frequency,
            stability_measure,
        }
    }

    fn compute_flow_consistency(&self, history: &GradientHistory) -> f64 {
        if history.gradient_norms.len() < 2 {
            return 1.0;
        }

        let variations: Vec<f64> = history
            .gradient_norms
            .iter()
            .collect::<Vec<&f64>>()
            .windows(2)
            .map(|pair| (*pair[1] - *pair[0]).abs() / (*pair[0] + 1e-8))
            .collect();

        let avg_variation = variations.iter().sum::<f64>() / variations.len() as f64;
        (1.0_f64 / (1.0 + avg_variation)).min(1.0)
    }

    fn compute_smoothness_index(&self, history: &GradientHistory) -> f64 {
        if history.gradient_norms.len() < 3 {
            return 1.0;
        }

        // Compute second derivatives to measure smoothness
        let second_derivatives: Vec<f64> = history
            .gradient_norms
            .iter()
            .collect::<Vec<&f64>>()
            .windows(3)
            .map(|window| *window[2] - 2.0 * *window[1] + *window[0])
            .collect();

        let avg_second_derivative = second_derivatives.iter().map(|&x| x.abs()).sum::<f64>()
            / second_derivatives.len() as f64;
        (1.0_f64 / (1.0 + avg_second_derivative)).min(1.0)
    }

    fn compute_trend_strength(&self, history: &GradientHistory) -> f64 {
        history.get_trend_slope().map(|slope| slope.abs().min(1.0)).unwrap_or(0.0)
    }

    fn compute_oscillation_frequency(&self, history: &GradientHistory) -> f64 {
        if history.gradient_norms.len() < 4 {
            return 0.0;
        }

        let sign_changes = history
            .gradient_norms
            .iter()
            .collect::<Vec<&f64>>()
            .windows(2)
            .map(|pair| *pair[1] - *pair[0])
            .collect::<Vec<f64>>()
            .windows(2)
            .filter(|pair| pair[0] * pair[1] < 0.0)
            .count();

        sign_changes as f64 / history.gradient_norms.len() as f64
    }

    fn compute_stability_measure(&self, history: &GradientHistory) -> f64 {
        if history.gradient_norms.is_empty() {
            return 0.0;
        }

        let mean = history.gradient_norms.iter().sum::<f64>() / history.gradient_norms.len() as f64;
        let variance = history.gradient_norms.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / history.gradient_norms.len() as f64;

        if mean == 0.0 {
            return 0.0;
        }

        let coefficient_of_variation = variance.sqrt() / mean;
        (1.0 / (1.0 + coefficient_of_variation)).min(1.0)
    }

    fn compute_layer_health_metrics(&self, history: &GradientHistory) -> LayerHealthMetrics {
        let gradient_statistics = self.compute_detailed_gradient_stats(history);
        let flow_characteristics = self.analyze_flow_characteristics(history);

        let gradient_stability = flow_characteristics.stability_measure;
        let information_flow_rate = self.compute_information_flow_rate(history);
        let neuron_activity_ratio = self.estimate_neuron_activity_ratio(history);
        let convergence_indicator = self.compute_convergence_indicator(history);

        let mut risk_factors = Vec::new();
        if gradient_statistics.mean < 1e-5 {
            risk_factors.push("Very low gradient magnitude".to_string());
        }
        if gradient_statistics.mean > 100.0 {
            risk_factors.push("Very high gradient magnitude".to_string());
        }
        if gradient_stability < 0.5 {
            risk_factors.push("High gradient instability".to_string());
        }
        if flow_characteristics.oscillation_frequency > 0.5 {
            risk_factors.push("High oscillation frequency".to_string());
        }

        let overall_health = if !risk_factors.is_empty() {
            if risk_factors.len() > 2 {
                LayerHealth::Critical
            } else {
                LayerHealth::Warning
            }
        } else {
            LayerHealth::Healthy
        };

        LayerHealthMetrics {
            overall_health,
            gradient_stability,
            information_flow_rate,
            neuron_activity_ratio,
            convergence_indicator,
            risk_factors,
        }
    }

    fn compute_information_flow_rate(&self, history: &GradientHistory) -> f64 {
        if history.gradient_norms.len() < 2 {
            return 0.0;
        }

        let total_change: f64 = history
            .gradient_norms
            .iter()
            .collect::<Vec<&f64>>()
            .windows(2)
            .map(|pair| (*pair[1] - *pair[0]).abs())
            .sum();

        total_change / history.gradient_norms.len() as f64
    }

    fn estimate_neuron_activity_ratio(&self, history: &GradientHistory) -> f64 {
        // Simplified estimation based on gradient magnitude
        let mean_gradient =
            history.gradient_norms.iter().sum::<f64>() / history.gradient_norms.len() as f64;
        (mean_gradient / (mean_gradient + 1e-5)).min(1.0)
    }

    fn compute_convergence_indicator(&self, history: &GradientHistory) -> f64 {
        if history.gradient_norms.len() < self.convergence_window {
            return 0.5; // Neutral score if insufficient data
        }

        let recent: Vec<f64> = history
            .gradient_norms
            .iter()
            .rev()
            .take(self.convergence_window)
            .cloned()
            .collect();
        let trend_slope = self.compute_trend_for_values(&recent);

        // Negative slope indicates convergence (decreasing gradients)
        if trend_slope < 0.0 {
            (-trend_slope).min(1.0)
        } else {
            0.0
        }
    }

    fn compute_trend_for_values(&self, values: &[f64]) -> f64 {
        if values.len() < 3 {
            return 0.0;
        }

        let n = values.len() as f64;
        let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2))
    }

    fn generate_layer_optimization_suggestions(
        &self,
        _layer_name: &str,
        history: &GradientHistory,
    ) -> Vec<LayerOptimizationSuggestion> {
        let mut suggestions = Vec::new();
        let stats = self.compute_detailed_gradient_stats(history);
        let flow = self.analyze_flow_characteristics(history);

        // Low gradient suggestions
        if stats.mean < 1e-5 {
            suggestions.push(LayerOptimizationSuggestion {
                suggestion_type: SuggestionType::WeightInitialization,
                description: "Consider better weight initialization methods".to_string(),
                rationale: "Very low gradients may indicate poor initialization".to_string(),
                expected_improvement: 0.7,
                implementation_complexity: ImplementationComplexity::Simple,
                side_effects: vec!["May require retraining from scratch".to_string()],
            });
        }

        // High gradient suggestions
        if stats.mean > 10.0 {
            suggestions.push(LayerOptimizationSuggestion {
                suggestion_type: SuggestionType::LearningRateAdjustment,
                description: "Reduce learning rate for this layer".to_string(),
                rationale: "High gradients may indicate learning rate is too large".to_string(),
                expected_improvement: 0.6,
                implementation_complexity: ImplementationComplexity::Simple,
                side_effects: vec!["May slow down convergence".to_string()],
            });
        }

        // High oscillation suggestions
        if flow.oscillation_frequency > 0.5 {
            suggestions.push(LayerOptimizationSuggestion {
                suggestion_type: SuggestionType::RegularizationAdd,
                description: "Add dropout or weight decay".to_string(),
                rationale: "High oscillation may indicate overfitting or instability".to_string(),
                expected_improvement: 0.5,
                implementation_complexity: ImplementationComplexity::Moderate,
                side_effects: vec!["May reduce model capacity".to_string()],
            });
        }

        suggestions
    }

    fn compare_with_other_layers(
        &self,
        layer_name: &str,
        history: &GradientHistory,
        all_histories: &HashMap<String, GradientHistory>,
    ) -> ComparativeAnalysis {
        let current_stats = self.compute_detailed_gradient_stats(history);
        let mut other_means = Vec::new();

        for (other_name, other_history) in all_histories {
            if other_name != layer_name {
                let other_stats = self.compute_detailed_gradient_stats(other_history);
                other_means.push(other_stats.mean);
            }
        }

        if other_means.is_empty() {
            return ComparativeAnalysis {
                relative_performance: 1.0,
                rank_among_layers: 1,
                similar_layers: vec![],
                performance_gap: 0.0,
                optimization_potential: 0.5,
            };
        }

        other_means.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let rank = other_means
            .iter()
            .position(|&x| x <= current_stats.mean)
            .unwrap_or(other_means.len())
            + 1;

        let avg_other_mean = other_means.iter().sum::<f64>() / other_means.len() as f64;
        let relative_performance =
            if avg_other_mean > 0.0 { current_stats.mean / avg_other_mean } else { 1.0 };

        let performance_gap = (current_stats.mean - avg_other_mean).abs();

        // Find similar layers (within 20% of performance)
        let similar_layers: Vec<String> = all_histories
            .iter()
            .filter(|(other_name, other_history)| {
                if *other_name == layer_name {
                    return false;
                }
                let other_stats = self.compute_detailed_gradient_stats(other_history);
                let ratio = (current_stats.mean / (other_stats.mean + 1e-8))
                    .max(other_stats.mean / (current_stats.mean + 1e-8));
                ratio <= 1.2
            })
            .map(|(name, _)| name.clone())
            .collect();

        let optimization_potential = if relative_performance < 0.5 {
            0.8
        } else if relative_performance < 0.8 {
            0.6
        } else {
            0.3
        };

        ComparativeAnalysis {
            relative_performance,
            rank_among_layers: rank,
            similar_layers,
            performance_gap,
            optimization_potential,
        }
    }

    fn analyze_network_level_gradients(
        &self,
        layer_details: &HashMap<String, LayerGradientDetails>,
    ) -> NetworkLevelAnalysis {
        let overall_gradient_health = self.assess_overall_health(layer_details);
        let gradient_distribution = self.analyze_gradient_distribution(layer_details);
        let layer_interactions = self.analyze_layer_interactions(layer_details);
        let convergence_indicators = self.analyze_convergence_indicators(layer_details);
        let training_dynamics = self.analyze_training_dynamics(layer_details);
        let stability_assessment = self.assess_network_stability(layer_details);

        NetworkLevelAnalysis {
            overall_gradient_health,
            gradient_distribution,
            layer_interactions,
            convergence_indicators,
            training_dynamics,
            stability_assessment,
        }
    }

    fn assess_overall_health(
        &self,
        layer_details: &HashMap<String, LayerGradientDetails>,
    ) -> LayerHealth {
        let health_counts = layer_details
            .values()
            .map(|details| &details.health_metrics.overall_health)
            .fold([0, 0, 0], |mut acc, health| {
                match health {
                    LayerHealth::Healthy => acc[0] += 1,
                    LayerHealth::Warning => acc[1] += 1,
                    LayerHealth::Critical => acc[2] += 1,
                    LayerHealth::Unknown => {}, // Ignore unknown health status
                }
                acc
            });

        let total = health_counts.iter().sum::<usize>();
        if total == 0 {
            return LayerHealth::Healthy;
        }

        let critical_ratio = health_counts[2] as f64 / total as f64;
        let warning_ratio = health_counts[1] as f64 / total as f64;

        if critical_ratio > 0.3 {
            LayerHealth::Critical
        } else if critical_ratio > 0.1 || warning_ratio > 0.5 {
            LayerHealth::Warning
        } else {
            LayerHealth::Healthy
        }
    }

    fn analyze_gradient_distribution(
        &self,
        layer_details: &HashMap<String, LayerGradientDetails>,
    ) -> GradientDistribution {
        let gradient_means: Vec<f64> =
            layer_details.values().map(|details| details.gradient_statistics.mean).collect();

        if gradient_means.is_empty() {
            return GradientDistribution {
                mean_gradient_norm: 0.0,
                gradient_variance: 0.0,
                gradient_skewness: 0.0,
                gradient_kurtosis: 0.0,
                layer_gradient_ratios: HashMap::new(),
                distribution_type: DistributionType::Degenerate,
            };
        }

        let n = gradient_means.len() as f64;
        let mean_gradient_norm = gradient_means.iter().sum::<f64>() / n;
        let gradient_variance =
            gradient_means.iter().map(|&x| (x - mean_gradient_norm).powi(2)).sum::<f64>() / n;

        let std_dev = gradient_variance.sqrt();
        let gradient_skewness = if std_dev > 0.0 {
            gradient_means
                .iter()
                .map(|&x| ((x - mean_gradient_norm) / std_dev).powi(3))
                .sum::<f64>()
                / n
        } else {
            0.0
        };

        let gradient_kurtosis = if std_dev > 0.0 {
            gradient_means
                .iter()
                .map(|&x| ((x - mean_gradient_norm) / std_dev).powi(4))
                .sum::<f64>()
                / n
                - 3.0
        } else {
            0.0
        };

        let mut layer_gradient_ratios = HashMap::new();
        for (layer_name, details) in layer_details {
            let ratio = if mean_gradient_norm > 0.0 {
                details.gradient_statistics.mean / mean_gradient_norm
            } else {
                1.0
            };
            layer_gradient_ratios.insert(layer_name.clone(), ratio);
        }

        let distribution_type =
            self.classify_distribution_type(gradient_skewness, gradient_kurtosis);

        GradientDistribution {
            mean_gradient_norm,
            gradient_variance,
            gradient_skewness,
            gradient_kurtosis,
            layer_gradient_ratios,
            distribution_type,
        }
    }

    fn classify_distribution_type(&self, skewness: f64, kurtosis: f64) -> DistributionType {
        if skewness.abs() > 2.0 {
            DistributionType::Skewed
        } else if kurtosis > 3.0 {
            DistributionType::HeavyTailed
        } else if kurtosis < -1.0 {
            DistributionType::Multimodal
        } else {
            DistributionType::Normal
        }
    }

    fn analyze_layer_interactions(
        &self,
        layer_details: &HashMap<String, LayerGradientDetails>,
    ) -> Vec<LayerInteraction> {
        let mut interactions = Vec::new();

        let layer_names: Vec<String> = layer_details.keys().cloned().collect();
        for i in 0..layer_names.len() {
            for j in (i + 1)..layer_names.len() {
                let layer1 = &layer_names[i];
                let layer2 = &layer_names[j];

                if let (Some(details1), Some(details2)) =
                    (layer_details.get(layer1), layer_details.get(layer2))
                {
                    let interaction_strength =
                        self.compute_interaction_strength(details1, details2);
                    let interaction_type = self.classify_interaction_type(details1, details2);
                    let impact_score = interaction_strength * 0.5; // Simplified impact calculation

                    interactions.push(LayerInteraction {
                        layer1: layer1.clone(),
                        layer2: layer2.clone(),
                        interaction_strength,
                        interaction_type,
                        impact_score,
                    });
                }
            }
        }

        interactions
    }

    fn compute_interaction_strength(
        &self,
        details1: &LayerGradientDetails,
        details2: &LayerGradientDetails,
    ) -> f64 {
        let mean_diff =
            (details1.gradient_statistics.mean - details2.gradient_statistics.mean).abs();
        let stability_diff = (details1.flow_characteristics.stability_measure
            - details2.flow_characteristics.stability_measure)
            .abs();

        // Interaction strength is inversely related to differences
        let combined_diff = mean_diff + stability_diff;
        1.0 / (1.0 + combined_diff)
    }

    fn classify_interaction_type(
        &self,
        details1: &LayerGradientDetails,
        details2: &LayerGradientDetails,
    ) -> InteractionType {
        let convergence_diff = (details1.health_metrics.convergence_indicator
            - details2.health_metrics.convergence_indicator)
            .abs();

        if convergence_diff < 0.1 {
            InteractionType::Cooperative
        } else if convergence_diff > 0.5 {
            InteractionType::Competitive
        } else {
            InteractionType::Neutral
        }
    }

    fn analyze_convergence_indicators(
        &self,
        layer_details: &HashMap<String, LayerGradientDetails>,
    ) -> ConvergenceIndicators {
        let convergence_scores: Vec<f64> = layer_details
            .values()
            .map(|details| details.health_metrics.convergence_indicator)
            .collect();

        let gradient_convergence_score =
            convergence_scores.iter().sum::<f64>() / convergence_scores.len().max(1) as f64;
        let parameter_convergence_score = gradient_convergence_score * 0.8; // Simplified
        let loss_convergence_score = gradient_convergence_score * 0.9; // Simplified

        let convergence_trend = if gradient_convergence_score > 0.8 {
            ConvergenceTrend::Converging
        } else if gradient_convergence_score > 0.6 {
            ConvergenceTrend::Stable
        } else if gradient_convergence_score < 0.3 {
            ConvergenceTrend::Diverging
        } else {
            ConvergenceTrend::Unknown
        };

        let estimated_steps_to_convergence = if gradient_convergence_score > 0.1 {
            Some(((1.0 - gradient_convergence_score) * 1000.0) as usize)
        } else {
            None
        };

        ConvergenceIndicators {
            gradient_convergence_score,
            parameter_convergence_score,
            loss_convergence_score,
            convergence_trend,
            estimated_steps_to_convergence,
        }
    }

    fn analyze_training_dynamics(
        &self,
        layer_details: &HashMap<String, LayerGradientDetails>,
    ) -> TrainingDynamics {
        let avg_convergence = layer_details
            .values()
            .map(|details| details.health_metrics.convergence_indicator)
            .sum::<f64>()
            / layer_details.len().max(1) as f64;

        let learning_phase = match avg_convergence {
            x if x < 0.2 => LearningPhase::InitialLearning,
            x if x < 0.4 => LearningPhase::RapidLearning,
            x if x < 0.6 => LearningPhase::Refinement,
            x if x < 0.8 => LearningPhase::Convergence,
            _ => LearningPhase::Plateau,
        };

        let gradient_momentum = avg_convergence * 0.8; // Simplified
        let learning_velocity = avg_convergence * 1.2; // Simplified
        let adaptation_rate = 1.0 - avg_convergence; // Simplified

        let plateau_detection = PlateauDetection {
            is_plateau: avg_convergence > 0.9,
            plateau_duration: if avg_convergence > 0.9 { 10 } else { 0 },
            plateau_severity: if avg_convergence > 0.95 {
                PlateauSeverity::Severe
            } else {
                PlateauSeverity::Mild
            },
            suggested_actions: if avg_convergence > 0.9 {
                vec![
                    "Consider learning rate reduction".to_string(),
                    "Add regularization".to_string(),
                ]
            } else {
                vec![]
            },
        };

        TrainingDynamics {
            learning_phase,
            gradient_momentum,
            learning_velocity,
            adaptation_rate,
            plateau_detection,
        }
    }

    fn assess_network_stability(
        &self,
        layer_details: &HashMap<String, LayerGradientDetails>,
    ) -> StabilityAssessment {
        let stability_scores: Vec<f64> = layer_details
            .values()
            .map(|details| details.flow_characteristics.stability_measure)
            .collect();

        let overall_stability =
            stability_scores.iter().sum::<f64>() / stability_scores.len().max(1) as f64;

        let stability_trend = if overall_stability > self.stability_threshold {
            StabilityTrend::Stable
        } else {
            StabilityTrend::Degrading
        };

        let instability_sources = self.identify_instability_sources(layer_details);

        let stability_forecast = StabilityForecast {
            short_term_outlook: if overall_stability > 0.7 {
                StabilityOutlook::Stable
            } else {
                StabilityOutlook::Deteriorating
            },
            long_term_outlook: if overall_stability > 0.8 {
                StabilityOutlook::Stable
            } else {
                StabilityOutlook::Uncertain
            },
            confidence_level: overall_stability,
            recommended_monitoring: vec![
                "Monitor gradient norms".to_string(),
                "Track convergence indicators".to_string(),
            ],
        };

        StabilityAssessment {
            overall_stability,
            stability_trend,
            instability_sources,
            stability_forecast,
        }
    }

    fn identify_instability_sources(
        &self,
        layer_details: &HashMap<String, LayerGradientDetails>,
    ) -> Vec<InstabilitySource> {
        let mut sources = Vec::new();

        for (layer_name, details) in layer_details {
            if details.gradient_statistics.mean > 100.0 {
                sources.push(InstabilitySource {
                    source_type: InstabilityType::GradientExplosion,
                    affected_layers: vec![layer_name.clone()],
                    severity: details.gradient_statistics.mean / 100.0,
                    description: format!("High gradient magnitude in layer {}", layer_name),
                });
            }

            if details.gradient_statistics.mean < 1e-5 {
                sources.push(InstabilitySource {
                    source_type: InstabilityType::GradientVanishing,
                    affected_layers: vec![layer_name.clone()],
                    severity: 1.0 - (details.gradient_statistics.mean * 1e5),
                    description: format!("Very low gradient magnitude in layer {}", layer_name),
                });
            }

            if details.flow_characteristics.oscillation_frequency > 0.5 {
                sources.push(InstabilitySource {
                    source_type: InstabilityType::Oscillation,
                    affected_layers: vec![layer_name.clone()],
                    severity: details.flow_characteristics.oscillation_frequency,
                    description: format!("High oscillation frequency in layer {}", layer_name),
                });
            }
        }

        sources
    }

    fn build_gradient_hierarchy(
        &self,
        layer_details: &HashMap<String, LayerGradientDetails>,
    ) -> GradientHierarchy {
        // Simplified hierarchy building - group layers by similar characteristics
        let mut layer_groups = Vec::new();
        let mut hierarchy_levels = Vec::new();
        let cross_level_interactions = Vec::new(); // Simplified - would compute actual interactions

        // Group by gradient magnitude ranges
        let high_gradient_layers: Vec<String> = layer_details
            .iter()
            .filter(|(_, details)| details.gradient_statistics.mean > 1.0)
            .map(|(name, _)| name.clone())
            .collect();

        let medium_gradient_layers: Vec<String> = layer_details
            .iter()
            .filter(|(_, details)| {
                details.gradient_statistics.mean >= 0.1 && details.gradient_statistics.mean <= 1.0
            })
            .map(|(name, _)| name.clone())
            .collect();

        let low_gradient_layers: Vec<String> = layer_details
            .iter()
            .filter(|(_, details)| details.gradient_statistics.mean < 0.1)
            .map(|(name, _)| name.clone())
            .collect();

        if !high_gradient_layers.is_empty() {
            layer_groups.push(LayerGroup {
                group_name: "High Gradient Layers".to_string(),
                layers: high_gradient_layers.clone(),
                group_characteristics: GroupCharacteristics {
                    average_gradient_norm: 2.0, // Simplified
                    gradient_synchronization: 0.8,
                    learning_rate_sensitivity: 0.9,
                    optimization_difficulty: OptimizationDifficulty::Difficult,
                },
                internal_coherence: 0.7,
            });

            hierarchy_levels.push(HierarchyLevel {
                level_id: 0,
                level_name: "High Gradient Level".to_string(),
                layer_groups: vec!["High Gradient Layers".to_string()],
                level_importance: 0.9,
                optimization_impact: 0.8,
            });
        }

        if !medium_gradient_layers.is_empty() {
            layer_groups.push(LayerGroup {
                group_name: "Medium Gradient Layers".to_string(),
                layers: medium_gradient_layers,
                group_characteristics: GroupCharacteristics {
                    average_gradient_norm: 0.5,
                    gradient_synchronization: 0.6,
                    learning_rate_sensitivity: 0.5,
                    optimization_difficulty: OptimizationDifficulty::Moderate,
                },
                internal_coherence: 0.8,
            });

            hierarchy_levels.push(HierarchyLevel {
                level_id: 1,
                level_name: "Medium Gradient Level".to_string(),
                layer_groups: vec!["Medium Gradient Layers".to_string()],
                level_importance: 0.7,
                optimization_impact: 0.6,
            });
        }

        if !low_gradient_layers.is_empty() {
            layer_groups.push(LayerGroup {
                group_name: "Low Gradient Layers".to_string(),
                layers: low_gradient_layers,
                group_characteristics: GroupCharacteristics {
                    average_gradient_norm: 0.05,
                    gradient_synchronization: 0.4,
                    learning_rate_sensitivity: 0.3,
                    optimization_difficulty: OptimizationDifficulty::Easy,
                },
                internal_coherence: 0.5,
            });

            hierarchy_levels.push(HierarchyLevel {
                level_id: 2,
                level_name: "Low Gradient Level".to_string(),
                layer_groups: vec!["Low Gradient Layers".to_string()],
                level_importance: 0.5,
                optimization_impact: 0.4,
            });
        }

        GradientHierarchy {
            layer_groups,
            hierarchy_levels,
            cross_level_interactions,
        }
    }

    fn rank_optimization_priorities(
        &self,
        layer_details: &HashMap<String, LayerGradientDetails>,
        network_analysis: &NetworkLevelAnalysis,
    ) -> Vec<OptimizationPriority> {
        let mut priorities = Vec::new();

        for (layer_name, details) in layer_details {
            let priority_score = self.calculate_priority_score(details, network_analysis);
            let urgency_level = self.determine_urgency_level(details);
            let optimization_potential = details.comparative_analysis.optimization_potential;
            let recommended_actions = self.generate_prioritized_actions(details);

            priorities.push(OptimizationPriority {
                target_name: layer_name.clone(),
                target_type: OptimizationTarget::IndividualLayer,
                priority_score,
                urgency_level,
                optimization_potential,
                recommended_actions,
            });
        }

        // Sort by priority score
        priorities.sort_by(|a, b| b.priority_score.partial_cmp(&a.priority_score).unwrap());

        priorities
    }

    fn calculate_priority_score(
        &self,
        details: &LayerGradientDetails,
        network_analysis: &NetworkLevelAnalysis,
    ) -> f64 {
        let health_weight = match details.health_metrics.overall_health {
            LayerHealth::Critical => 1.0,
            LayerHealth::Warning => 0.7,
            LayerHealth::Healthy => 0.3,
            LayerHealth::Unknown => 0.5, // Default moderate weight for unknown health
        };

        let stability_weight = 1.0 - details.flow_characteristics.stability_measure;
        let optimization_weight = details.comparative_analysis.optimization_potential;
        let network_impact_weight = details.health_metrics.information_flow_rate
            / network_analysis.gradient_distribution.mean_gradient_norm.max(1e-8);

        (health_weight * 0.4
            + stability_weight * 0.3
            + optimization_weight * 0.2
            + network_impact_weight * 0.1)
            .min(1.0)
    }

    fn determine_urgency_level(&self, details: &LayerGradientDetails) -> UrgencyLevel {
        match details.health_metrics.overall_health {
            LayerHealth::Critical => UrgencyLevel::Critical,
            LayerHealth::Warning => {
                if details.flow_characteristics.stability_measure < 0.3 {
                    UrgencyLevel::High
                } else {
                    UrgencyLevel::Medium
                }
            },
            LayerHealth::Healthy => UrgencyLevel::Low,
            LayerHealth::Unknown => UrgencyLevel::Medium, // Default moderate urgency for unknown health
        }
    }

    fn generate_prioritized_actions(
        &self,
        details: &LayerGradientDetails,
    ) -> Vec<PrioritizedAction> {
        let mut actions = Vec::new();

        if details.gradient_statistics.mean < 1e-5 {
            actions.push(PrioritizedAction {
                action_name: "Weight Initialization Improvement".to_string(),
                action_type: ActionType::ParameterAdjustment,
                expected_impact: 0.8,
                implementation_effort: ImplementationEffort::Moderate,
                prerequisites: vec!["Model architecture review".to_string()],
            });
        }

        if details.gradient_statistics.mean > 10.0 {
            actions.push(PrioritizedAction {
                action_name: "Learning Rate Reduction".to_string(),
                action_type: ActionType::LearningRateScheduling,
                expected_impact: 0.7,
                implementation_effort: ImplementationEffort::Minimal,
                prerequisites: vec![],
            });
        }

        if details.flow_characteristics.stability_measure < 0.5 {
            actions.push(PrioritizedAction {
                action_name: "Gradient Clipping".to_string(),
                action_type: ActionType::OptimizationTechnique,
                expected_impact: 0.6,
                implementation_effort: ImplementationEffort::Low,
                prerequisites: vec!["Hyperparameter tuning".to_string()],
            });
        }

        actions
    }

    /// Analyze gradients and generate enhanced analysis results
    pub fn analyze_gradients(
        &self,
        gradient_histories: &HashMap<String, GradientHistory>,
    ) -> EnhancedLayerGradientAnalysis {
        // Use existing method to generate the analysis
        self.generate_enhanced_analysis(gradient_histories)
    }
}
