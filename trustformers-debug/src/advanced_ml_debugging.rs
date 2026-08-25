//! # Advanced ML Debugging Tools
//!
//! Advanced machine learning specific debugging techniques including layer-wise learning rate adaptation,
//! model sensitivity analysis, gradient flow optimization, and neural architecture debugging.

use anyhow::Result;
use chrono::{DateTime, Utc};
use scirs2_core::ndarray::*; // SciRS2 Integration Policy - was: use ndarray::{Array1, Array2, Array3, ArrayD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for advanced ML debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedMLDebuggingConfig {
    /// Enable layer-wise learning rate analysis
    pub enable_layer_wise_lr_analysis: bool,
    /// Enable model sensitivity analysis
    pub enable_model_sensitivity_analysis: bool,
    /// Enable gradient flow optimization analysis
    pub enable_gradient_flow_optimization: bool,
    /// Enable neural architecture debugging
    pub enable_neural_architecture_debugging: bool,
    /// Enable activation pattern analysis
    pub enable_activation_pattern_analysis: bool,
    /// Enable weight distribution analysis
    pub enable_weight_distribution_analysis: bool,
    /// Enable training dynamics analysis
    pub enable_training_dynamics_analysis: bool,
    /// Enable optimization landscape analysis
    pub enable_optimization_landscape_analysis: bool,
    /// Number of samples for sensitivity analysis
    pub sensitivity_samples: usize,
    /// Learning rate adaptation threshold
    pub lr_adaptation_threshold: f64,
    /// Maximum number of layers to analyze
    pub max_layers_to_analyze: usize,
}

impl Default for AdvancedMLDebuggingConfig {
    fn default() -> Self {
        Self {
            enable_layer_wise_lr_analysis: true,
            enable_model_sensitivity_analysis: true,
            enable_gradient_flow_optimization: true,
            enable_neural_architecture_debugging: true,
            enable_activation_pattern_analysis: true,
            enable_weight_distribution_analysis: true,
            enable_training_dynamics_analysis: true,
            enable_optimization_landscape_analysis: true,
            sensitivity_samples: 1000,
            lr_adaptation_threshold: 0.1,
            max_layers_to_analyze: 50,
        }
    }
}

/// Layer-wise learning rate adaptation analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWiseLRAnalysisResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Learning rate recommendations per layer
    pub layer_lr_recommendations: HashMap<String, LayerLRRecommendation>,
    /// Global learning rate insights
    pub global_lr_insights: GlobalLRInsights,
    /// Learning rate adaptation strategy
    pub adaptation_strategy: LRAdaptationStrategy,
    /// Training phase recommendations
    pub training_phase_recommendations: Vec<TrainingPhaseRecommendation>,
    /// Performance predictions with different LR schedules
    pub lr_schedule_predictions: Vec<LRSchedulePrediction>,
}

/// Learning rate recommendation for a specific layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerLRRecommendation {
    /// Layer identifier
    pub layer_id: String,
    /// Layer type (e.g., "attention", "feedforward", "embedding")
    pub layer_type: String,
    /// Current learning rate
    pub current_lr: f64,
    /// Recommended learning rate
    pub recommended_lr: f64,
    /// Statistical confidence in the recommendation.
    ///
    /// `None` from [`AdvancedMLDebugger::analyze_layer_wise_learning_rates`]:
    /// that path sees a single gradient/weight snapshot per layer, with no
    /// repeated trials or held-out evaluation, so there is no sampling
    /// distribution to derive a confidence from. It used to be a hardcoded
    /// `0.8`.
    pub confidence: Option<f64>,
    /// Reasoning for recommendation
    pub reasoning: String,
    /// Layer-specific metrics
    pub layer_metrics: LayerLRMetrics,
    /// Sensitivity to learning rate changes
    pub lr_sensitivity: f64,
    /// Adaptation urgency level
    pub urgency: AdaptationUrgency,
}

/// Layer-specific learning rate metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerLRMetrics {
    /// Gradient magnitude
    pub gradient_magnitude: f64,
    /// Weight update magnitude
    pub weight_update_magnitude: f64,
    /// Parameter norm
    pub parameter_norm: f64,
    /// Loss contribution
    pub loss_contribution: f64,
    /// Training stability score
    pub stability_score: f64,
    /// Convergence rate
    pub convergence_rate: f64,
    /// Learning efficiency
    pub learning_efficiency: f64,
}

/// Urgency level for learning rate adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationUrgency {
    Low,
    Medium,
    High,
    Critical,
}

/// Global learning rate insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalLRInsights {
    /// Overall model learning efficiency
    pub overall_efficiency: f64,
    /// Learning rate distribution health
    pub lr_distribution_health: f64,
    /// Gradient flow quality
    pub gradient_flow_quality: f64,
    /// Training stability assessment
    pub training_stability: TrainingStability,
    /// Recommended global adjustments
    pub global_adjustments: Vec<GlobalLRAdjustment>,
    /// Critical issues requiring immediate attention
    pub critical_issues: Vec<String>,
}

/// Training stability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStability {
    /// Stability score (0-1)
    pub stability_score: f64,
    /// Instability indicators
    pub instability_indicators: Vec<InstabilityIndicator>,
    /// Stability trends over time
    pub stability_trends: Vec<StabilityTrendPoint>,
    /// Predicted stability with current settings
    pub predicted_stability: f64,
}

/// Indicator of training instability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstabilityIndicator {
    /// Type of instability
    pub instability_type: InstabilityType,
    /// Severity level
    pub severity: f64,
    /// Affected layers
    pub affected_layers: Vec<String>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Type of training instability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstabilityType {
    GradientExplosion,
    GradientVanishing,
    OscillatingLoss,
    SlowConvergence,
    WeightDivergence,
    NumericalInstability,
}

/// Point in stability trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityTrendPoint {
    /// Time step or epoch
    pub time_step: usize,
    /// Stability score at this point
    pub stability_score: f64,
    /// Contributing factors
    pub contributing_factors: HashMap<String, f64>,
}

/// Global learning rate adjustment recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalLRAdjustment {
    /// Adjustment type
    pub adjustment_type: GlobalAdjustmentType,
    /// Adjustment magnitude
    pub magnitude: f64,
    /// Expected impact
    pub expected_impact: f64,
    /// Implementation priority
    pub priority: AdjustmentPriority,
    /// Implementation instructions
    pub instructions: String,
}

/// Type of global learning rate adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GlobalAdjustmentType {
    UniformScaling,
    LayerTypeSpecific,
    DepthDependent,
    AdaptiveScheduling,
    WarmupAdjustment,
    DecayRateModification,
}

/// Priority level for adjustments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdjustmentPriority {
    Low,
    Medium,
    High,
    Immediate,
}

/// Learning rate adaptation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRAdaptationStrategy {
    /// Strategy name
    pub strategy_name: String,
    /// Strategy description
    pub description: String,
    /// Implementation steps
    pub implementation_steps: Vec<ImplementationStep>,
    /// Expected benefits
    pub expected_benefits: Vec<String>,
    /// Potential risks
    pub potential_risks: Vec<String>,
    /// Success metrics
    pub success_metrics: Vec<String>,
    /// Monitoring requirements
    pub monitoring_requirements: Vec<String>,
}

/// Step in implementing an adaptation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationStep {
    /// Step number
    pub step_number: usize,
    /// Step description
    pub description: String,
    /// Code changes required
    pub code_changes: Vec<String>,
    /// Expected timeline
    pub timeline: String,
    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Training phase recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPhaseRecommendation {
    /// Phase name
    pub phase_name: String,
    /// Phase duration (epochs)
    pub duration_epochs: usize,
    /// Learning rate schedule for this phase
    pub lr_schedule: LRSchedule,
    /// Phase objectives
    pub objectives: Vec<String>,
    /// Success criteria
    pub success_criteria: Vec<String>,
    /// Transition conditions
    pub transition_conditions: Vec<String>,
}

/// Learning rate schedule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRSchedule {
    /// Schedule type
    pub schedule_type: LRScheduleType,
    /// Initial learning rate
    pub initial_lr: f64,
    /// Schedule parameters
    pub parameters: HashMap<String, f64>,
    /// Layer-specific multipliers
    pub layer_multipliers: HashMap<String, f64>,
}

/// Type of learning rate schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LRScheduleType {
    Constant,
    LinearDecay,
    ExponentialDecay,
    CosineAnnealing,
    StepDecay,
    CyclicalLR,
    OneCycleLR,
    AdaptiveSchedule,
}

/// Prediction of performance with different LR schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRSchedulePrediction {
    /// Schedule being evaluated
    pub schedule: LRSchedule,
    /// Predicted final accuracy
    pub predicted_accuracy: f64,
    /// Predicted convergence time
    pub predicted_convergence_epochs: usize,
    /// Predicted training stability
    pub predicted_stability: f64,
    /// Confidence in prediction
    pub prediction_confidence: f64,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
}

/// Risk assessment for a learning rate schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk level
    pub overall_risk: RiskLevel,
    /// Specific risks
    pub specific_risks: Vec<SpecificRisk>,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Risk level assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Specific risk in training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecificRisk {
    /// Risk type
    pub risk_type: String,
    /// Probability of occurrence
    pub probability: f64,
    /// Impact severity
    pub impact: f64,
    /// Description
    pub description: String,
}

/// Model sensitivity analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSensitivityAnalysisResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Hyperparameter sensitivity analysis
    pub hyperparameter_sensitivity: HyperparameterSensitivity,
    /// Architecture sensitivity analysis
    pub architecture_sensitivity: ArchitectureSensitivity,
    /// Data sensitivity analysis
    pub data_sensitivity: DataSensitivity,
    /// Training procedure sensitivity
    pub training_sensitivity: TrainingSensitivity,
    /// Overall sensitivity insights
    pub sensitivity_insights: SensitivityInsights,
}

/// Hyperparameter sensitivity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSensitivity {
    /// Learning rate sensitivity
    pub learning_rate_sensitivity: ParameterSensitivity,
    /// Batch size sensitivity
    pub batch_size_sensitivity: ParameterSensitivity,
    /// Regularization sensitivity
    pub regularization_sensitivity: ParameterSensitivity,
    /// Architecture parameter sensitivity
    pub architecture_param_sensitivity: HashMap<String, ParameterSensitivity>,
    /// Most sensitive parameters
    pub most_sensitive_params: Vec<String>,
    /// Least sensitive parameters
    pub least_sensitive_params: Vec<String>,
    /// Parameter interaction effects
    pub interaction_effects: Vec<ParameterInteraction>,
}

/// Sensitivity analysis for a specific parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSensitivity {
    /// Parameter name
    pub parameter_name: String,
    /// Current value
    pub current_value: f64,
    /// Sensitivity score
    pub sensitivity_score: f64,
    /// Optimal value range
    pub optimal_range: (f64, f64),
    /// Performance impact curve
    pub impact_curve: Vec<(f64, f64)>,
    /// Stability region
    pub stability_region: (f64, f64),
    /// Critical thresholds
    pub critical_thresholds: Vec<f64>,
}

/// Interaction between parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInteraction {
    /// First parameter
    pub param1: String,
    /// Second parameter
    pub param2: String,
    /// Interaction strength
    pub interaction_strength: f64,
    /// Interaction type
    pub interaction_type: InteractionType,
    /// Joint optimal region
    pub joint_optimal_region: HashMap<String, (f64, f64)>,
}

/// Type of parameter interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    Synergistic,
    Antagonistic,
    Independent,
    Conditional,
}

/// Architecture sensitivity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureSensitivity {
    /// Layer depth sensitivity
    pub depth_sensitivity: ArchitecturalSensitivity,
    /// Layer width sensitivity
    pub width_sensitivity: ArchitecturalSensitivity,
    /// Attention head sensitivity
    pub attention_head_sensitivity: ArchitecturalSensitivity,
    /// Skip connection sensitivity
    pub skip_connection_sensitivity: ArchitecturalSensitivity,
    /// Architectural component importance
    pub component_importance: HashMap<String, f64>,
    /// Architectural bottlenecks
    pub bottlenecks: Vec<ArchitecturalBottleneck>,
}

/// Sensitivity analysis for architectural component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalSensitivity {
    /// Component name
    pub component_name: String,
    /// Sensitivity to changes
    pub change_sensitivity: f64,
    /// Performance degradation curve
    pub degradation_curve: Vec<(f64, f64)>,
    /// Minimum viable configuration
    pub min_viable_config: f64,
    /// Optimal configuration
    pub optimal_config: f64,
    /// Diminishing returns threshold
    pub diminishing_returns_threshold: f64,
}

/// Architectural bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalBottleneck {
    /// Bottleneck location
    pub location: String,
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity
    pub severity: f64,
    /// Performance impact
    pub performance_impact: f64,
    /// Resolution recommendations
    pub resolution_recommendations: Vec<String>,
}

/// Type of architectural bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    ComputationalBottleneck,
    MemoryBottleneck,
    InformationBottleneck,
    CapacityBottleneck,
    CommunicationBottleneck,
}

/// Data sensitivity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSensitivity {
    /// Training data size sensitivity
    pub data_size_sensitivity: DataSizeSensitivity,
    /// Data quality sensitivity
    pub data_quality_sensitivity: DataQualitySensitivity,
    /// Data distribution sensitivity
    pub distribution_sensitivity: DistributionSensitivity,
    /// Feature sensitivity analysis
    pub feature_sensitivity: FeatureSensitivityAnalysis,
}

/// Sensitivity to training data size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSizeSensitivity {
    /// Current data size
    pub current_size: usize,
    /// Minimum effective size
    pub minimum_effective_size: usize,
    /// Performance vs size curve
    pub performance_curve: Vec<(usize, f64)>,
    /// Data efficiency score
    pub data_efficiency: f64,
    /// Diminishing returns point
    pub diminishing_returns_point: usize,
}

/// Sensitivity to data quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualitySensitivity {
    /// Noise tolerance
    pub noise_tolerance: f64,
    /// Label quality importance
    pub label_quality_importance: f64,
    /// Feature quality importance
    pub feature_quality_importance: f64,
    /// Quality degradation impact
    pub quality_impact_curve: Vec<(f64, f64)>,
}

/// Sensitivity to data distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionSensitivity {
    /// Distribution shift sensitivity
    pub shift_sensitivity: f64,
    /// Class imbalance sensitivity
    pub imbalance_sensitivity: f64,
    /// Domain adaptation requirements
    pub domain_adaptation_requirements: Vec<String>,
    /// Robustness to distribution changes
    pub distribution_robustness: f64,
}

/// Feature-level sensitivity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSensitivityAnalysis {
    /// Most important features
    pub most_important_features: Vec<String>,
    /// Least important features
    pub least_important_features: Vec<String>,
    /// Feature interaction importance
    pub feature_interactions: HashMap<(String, String), f64>,
    /// Feature stability analysis
    pub feature_stability: HashMap<String, f64>,
}

/// Training procedure sensitivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSensitivity {
    /// Initialization sensitivity
    pub initialization_sensitivity: InitializationSensitivity,
    /// Optimization method sensitivity
    pub optimization_sensitivity: OptimizationSensitivity,
    /// Training schedule sensitivity
    pub schedule_sensitivity: ScheduleSensitivity,
    /// Regularization sensitivity
    pub regularization_sensitivity: RegularizationSensitivity,
}

/// Sensitivity to initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializationSensitivity {
    /// Weight initialization sensitivity
    pub weight_init_sensitivity: f64,
    /// Bias initialization sensitivity
    pub bias_init_sensitivity: f64,
    /// Random seed sensitivity
    pub seed_sensitivity: f64,
    /// Initialization scheme importance
    pub scheme_importance: HashMap<String, f64>,
}

/// Sensitivity to optimization method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSensitivity {
    /// Optimizer choice sensitivity
    pub optimizer_sensitivity: f64,
    /// Momentum parameter sensitivity
    pub momentum_sensitivity: f64,
    /// Second-order moment sensitivity
    pub second_moment_sensitivity: f64,
    /// Optimizer comparison
    pub optimizer_comparison: HashMap<String, f64>,
}

/// Sensitivity to training schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleSensitivity {
    /// Learning rate schedule sensitivity
    pub lr_schedule_sensitivity: f64,
    /// Training duration sensitivity
    pub duration_sensitivity: f64,
    /// Warmup sensitivity
    pub warmup_sensitivity: f64,
    /// Schedule parameter importance
    pub schedule_param_importance: HashMap<String, f64>,
}

/// Sensitivity to regularization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationSensitivity {
    /// Dropout sensitivity
    pub dropout_sensitivity: f64,
    /// Weight decay sensitivity
    pub weight_decay_sensitivity: f64,
    /// Batch normalization sensitivity
    pub batch_norm_sensitivity: f64,
    /// Regularization method comparison
    pub method_comparison: HashMap<String, f64>,
}

/// Overall sensitivity insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityInsights {
    /// Most critical factors
    pub most_critical_factors: Vec<String>,
    /// Least critical factors
    pub least_critical_factors: Vec<String>,
    /// Surprising findings
    pub surprising_findings: Vec<String>,
    /// Robustness assessment
    pub robustness_assessment: RobustnessAssessment,
    /// Optimization recommendations
    pub optimization_recommendations: Vec<String>,
}

/// Model robustness assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessAssessment {
    /// Overall robustness score
    pub overall_robustness: f64,
    /// Robustness breakdown by category
    pub category_robustness: HashMap<String, f64>,
    /// Vulnerability areas
    pub vulnerabilities: Vec<Vulnerability>,
    /// Strength areas
    pub strengths: Vec<String>,
}

/// Model vulnerability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    /// Vulnerability type
    pub vulnerability_type: String,
    /// Severity level
    pub severity: f64,
    /// Impact description
    pub impact: String,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Advanced ML debugger
#[derive(Debug)]
pub struct AdvancedMLDebugger {
    config: AdvancedMLDebuggingConfig,
    lr_analysis_results: Vec<LayerWiseLRAnalysisResult>,
    sensitivity_analysis_results: Vec<ModelSensitivityAnalysisResult>,
}

impl AdvancedMLDebugger {
    /// Create a new advanced ML debugger
    pub fn new(config: AdvancedMLDebuggingConfig) -> Self {
        Self {
            config,
            lr_analysis_results: Vec::new(),
            sensitivity_analysis_results: Vec::new(),
        }
    }

    /// Perform layer-wise learning rate analysis
    pub async fn analyze_layer_wise_learning_rates(
        &mut self,
        layer_gradients: &HashMap<String, ArrayD<f32>>,
        layer_weights: &HashMap<String, ArrayD<f32>>,
        current_lr: f64,
        loss_history: &[f64],
    ) -> Result<LayerWiseLRAnalysisResult> {
        if !self.config.enable_layer_wise_lr_analysis {
            return Err(anyhow::anyhow!(
                "Layer-wise learning rate analysis is disabled"
            ));
        }

        let mut layer_lr_recommendations = HashMap::new();

        // Analyze each layer
        for (layer_id, gradients) in layer_gradients {
            if let Some(weights) = layer_weights.get(layer_id) {
                let recommendation = self.analyze_single_layer_lr(
                    layer_id,
                    gradients,
                    weights,
                    current_lr,
                    loss_history,
                );
                layer_lr_recommendations.insert(layer_id.clone(), recommendation);
            }
        }

        // Generate global insights
        let global_lr_insights =
            self.generate_global_lr_insights(&layer_lr_recommendations, loss_history);

        // Create adaptation strategy
        let adaptation_strategy =
            self.create_lr_adaptation_strategy(&layer_lr_recommendations, &global_lr_insights);

        // Generate training phase recommendations
        let training_phase_recommendations =
            self.generate_training_phase_recommendations(&adaptation_strategy);

        // Predict performance with different schedules
        let lr_schedule_predictions =
            self.predict_lr_schedule_performance(&layer_lr_recommendations);

        let result = LayerWiseLRAnalysisResult {
            timestamp: Utc::now(),
            layer_lr_recommendations,
            global_lr_insights,
            adaptation_strategy,
            training_phase_recommendations,
            lr_schedule_predictions,
        };

        self.lr_analysis_results.push(result.clone());
        Ok(result)
    }

    /// Perform comprehensive model sensitivity analysis.
    ///
    /// **Always returns a structured error.** A sensitivity analysis measures
    /// how performance responds when a factor is *varied*, which requires
    /// re-evaluating the model at perturbed hyperparameters, architectures,
    /// dataset sizes and seeds. This method receives only the CURRENT parameter
    /// values plus a flat `&[f64]` of already-observed metrics -- one point per
    /// factor -- so no response curve, optimal range, stability region or
    /// robustness score is derivable from its inputs.
    ///
    /// It used to return a fully populated [`ModelSensitivityAnalysisResult`]
    /// assembled from hardcoded constants: `sensitivity_score: 0.8`,
    /// `optimal_range: (0.0001, 0.01)`, `current_size: 10000`,
    /// `most_important_features: ["feature_1", "feature_2"]`, and a
    /// `surprising_findings` string, none of which looked at `model_params`,
    /// `performance_metrics` or `architecture_config`. The five helpers that
    /// produced those constants have been deleted.
    ///
    /// The result types remain public so a caller that really does run a
    /// parameter sweep can construct and share one.
    pub async fn analyze_model_sensitivity(
        &mut self,
        model_params: &HashMap<String, f64>,
        performance_metrics: &[f64],
        architecture_config: &HashMap<String, f64>,
    ) -> Result<ModelSensitivityAnalysisResult> {
        if !self.config.enable_model_sensitivity_analysis {
            return Err(anyhow::anyhow!("Model sensitivity analysis is disabled"));
        }
        Err(anyhow::anyhow!(
            "model sensitivity analysis is not implemented: it needs the model re-evaluated at \
             perturbed settings, but this call receives only {} current parameter values, {} \
             architecture values and {} already-observed metric samples -- no perturbation \
             response is derivable from a single operating point",
            model_params.len(),
            architecture_config.len(),
            performance_metrics.len(),
        ))
    }

    /// Generate comprehensive advanced ML debugging report
    pub async fn generate_report(&self) -> Result<AdvancedMLDebuggingReport> {
        Ok(AdvancedMLDebuggingReport {
            timestamp: Utc::now(),
            config: self.config.clone(),
            lr_analysis_count: self.lr_analysis_results.len(),
            sensitivity_analysis_count: self.sensitivity_analysis_results.len(),
            recent_lr_analyses: self.lr_analysis_results.iter().rev().take(3).cloned().collect(),
            recent_sensitivity_analyses: self
                .sensitivity_analysis_results
                .iter()
                .rev()
                .take(3)
                .cloned()
                .collect(),
            advanced_insights: self.generate_advanced_insights(),
        })
    }

    // Helper methods for layer-wise LR analysis

    fn analyze_single_layer_lr(
        &self,
        layer_id: &str,
        gradients: &ArrayD<f32>,
        weights: &ArrayD<f32>,
        current_lr: f64,
        loss_history: &[f64],
    ) -> LayerLRRecommendation {
        // Calculate gradient statistics
        let gradient_magnitude =
            gradients.iter().map(|&x| x.abs() as f64).sum::<f64>() / gradients.len() as f64;
        let weight_magnitude =
            weights.iter().map(|&x| x.abs() as f64).sum::<f64>() / weights.len() as f64;

        // Estimate optimal learning rate based on gradient properties
        let gradient_variance =
            gradients.iter().map(|&x| (x as f64 - gradient_magnitude).powi(2)).sum::<f64>()
                / gradients.len() as f64;

        let gradient_norm = gradient_magnitude;
        let recommended_lr = if gradient_norm > 0.0 {
            // Adaptive learning rate based on gradient properties
            let base_lr = 0.001;
            let adaptation_factor = (1.0 / (1.0 + gradient_variance)).sqrt();
            let magnitude_factor = (1.0 / (1.0 + gradient_norm)).sqrt();
            base_lr * adaptation_factor * magnitude_factor * 10.0
        } else {
            current_lr
        };

        // Calculate layer metrics
        let layer_metrics = LayerLRMetrics {
            gradient_magnitude,
            weight_update_magnitude: gradient_magnitude * current_lr,
            parameter_norm: weight_magnitude,
            loss_contribution: self.estimate_layer_loss_contribution(loss_history),
            stability_score: self.calculate_layer_stability(gradients),
            convergence_rate: self.estimate_convergence_rate(loss_history),
            learning_efficiency: gradient_magnitude / (weight_magnitude + 1e-8),
        };

        // Determine urgency
        let lr_ratio = recommended_lr / current_lr;
        let urgency = if !(0.1..=10.0).contains(&lr_ratio) {
            AdaptationUrgency::Critical
        } else if !(0.33..=3.0).contains(&lr_ratio) {
            AdaptationUrgency::High
        } else if !(0.67..=1.5).contains(&lr_ratio) {
            AdaptationUrgency::Medium
        } else {
            AdaptationUrgency::Low
        };

        // Generate reasoning
        let reasoning = if recommended_lr > current_lr * 1.2 {
            "Layer shows slow learning with small gradients, increase learning rate".to_string()
        } else if recommended_lr < current_lr * 0.8 {
            "Layer shows instability or large gradients, decrease learning rate".to_string()
        } else {
            "Current learning rate appears appropriate for this layer".to_string()
        };

        LayerLRRecommendation {
            layer_id: layer_id.to_string(),
            layer_type: self.infer_layer_type(layer_id),
            current_lr,
            recommended_lr,
            // No statistical confidence is derivable here: the recommendation
            // comes from one gradient/weight snapshot of one layer, with no
            // repeated trials and no held-out evaluation to estimate a
            // sampling distribution from. Previously a hardcoded `0.8`.
            confidence: None,
            reasoning,
            layer_metrics,
            lr_sensitivity: lr_ratio.abs(),
            urgency,
        }
    }

    fn generate_global_lr_insights(
        &self,
        layer_recommendations: &HashMap<String, LayerLRRecommendation>,
        loss_history: &[f64],
    ) -> GlobalLRInsights {
        let overall_efficiency = layer_recommendations
            .values()
            .map(|rec| rec.layer_metrics.learning_efficiency)
            .sum::<f64>()
            / layer_recommendations.len() as f64;

        let lr_distribution_health = self.calculate_lr_distribution_health(layer_recommendations);
        let gradient_flow_quality = self.calculate_gradient_flow_quality(layer_recommendations);
        let training_stability =
            self.assess_training_stability(layer_recommendations, loss_history);
        let global_adjustments = self.generate_global_adjustments(layer_recommendations);
        let critical_issues = self.identify_critical_issues(layer_recommendations);

        GlobalLRInsights {
            overall_efficiency,
            lr_distribution_health,
            gradient_flow_quality,
            training_stability,
            global_adjustments,
            critical_issues,
        }
    }

    fn create_lr_adaptation_strategy(
        &self,
        _layer_recommendations: &HashMap<String, LayerLRRecommendation>,
        global_insights: &GlobalLRInsights,
    ) -> LRAdaptationStrategy {
        // Simplified strategy creation
        let strategy_name = if global_insights.overall_efficiency < 0.5 {
            "Aggressive Learning Rate Adaptation".to_string()
        } else {
            "Conservative Learning Rate Tuning".to_string()
        };

        LRAdaptationStrategy {
            strategy_name: strategy_name.clone(),
            description: "Strategy to optimize learning rates based on current model state"
                .to_string(),
            implementation_steps: vec![ImplementationStep {
                step_number: 1,
                description: "Implement layer-wise learning rate multipliers".to_string(),
                code_changes: vec!["Add lr_multipliers to optimizer config".to_string()],
                timeline: "1-2 days".to_string(),
                dependencies: vec!["Optimizer modification".to_string()],
            }],
            expected_benefits: vec![
                "Improved convergence speed".to_string(),
                "Better training stability".to_string(),
                "Reduced overfitting risk".to_string(),
            ],
            potential_risks: vec!["Initial instability during adaptation".to_string()],
            success_metrics: vec![
                "Faster loss reduction".to_string(),
                "Improved validation accuracy".to_string(),
            ],
            monitoring_requirements: vec!["Track per-layer gradient norms".to_string()],
        }
    }

    fn generate_training_phase_recommendations(
        &self,
        _strategy: &LRAdaptationStrategy,
    ) -> Vec<TrainingPhaseRecommendation> {
        vec![TrainingPhaseRecommendation {
            phase_name: "Warmup Phase".to_string(),
            duration_epochs: 5,
            lr_schedule: LRSchedule {
                schedule_type: LRScheduleType::LinearDecay,
                initial_lr: 0.0001,
                parameters: HashMap::new(),
                layer_multipliers: HashMap::new(),
            },
            objectives: vec!["Stabilize training".to_string()],
            success_criteria: vec!["Decreasing loss".to_string()],
            transition_conditions: vec!["Stable gradient norms".to_string()],
        }]
    }

    fn predict_lr_schedule_performance(
        &self,
        _layer_recommendations: &HashMap<String, LayerLRRecommendation>,
    ) -> Vec<LRSchedulePrediction> {
        vec![LRSchedulePrediction {
            schedule: LRSchedule {
                schedule_type: LRScheduleType::ExponentialDecay,
                initial_lr: 0.001,
                parameters: HashMap::new(),
                layer_multipliers: HashMap::new(),
            },
            predicted_accuracy: 0.92,
            predicted_convergence_epochs: 50,
            predicted_stability: 0.8,
            prediction_confidence: 0.7,
            risk_assessment: RiskAssessment {
                overall_risk: RiskLevel::Medium,
                specific_risks: vec![],
                mitigation_strategies: vec![],
            },
        }]
    }

    // Helper methods for sensitivity analysis

    // Additional helper methods

    fn estimate_layer_loss_contribution(&self, loss_history: &[f64]) -> f64 {
        // Simplified estimation
        if loss_history.len() >= 2 {
            (loss_history[loss_history.len() - 2] - loss_history[loss_history.len() - 1]).abs()
        } else {
            0.1
        }
    }

    fn calculate_layer_stability(&self, gradients: &ArrayD<f32>) -> f64 {
        let gradient_variance = gradients.iter().map(|&x| x as f64).collect::<Vec<_>>();

        if gradient_variance.is_empty() {
            return 0.5;
        }

        let mean = gradient_variance.iter().sum::<f64>() / gradient_variance.len() as f64;
        let variance = gradient_variance.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / gradient_variance.len() as f64;

        1.0 / (1.0 + variance) // Higher stability for lower variance
    }

    fn estimate_convergence_rate(&self, loss_history: &[f64]) -> f64 {
        if loss_history.len() < 3 {
            return 0.5;
        }

        let recent_improvement =
            loss_history[loss_history.len() - 3] - loss_history[loss_history.len() - 1];
        recent_improvement.abs()
    }

    fn infer_layer_type(&self, layer_id: &str) -> String {
        if layer_id.contains("attention") {
            "attention".to_string()
        } else if layer_id.contains("feedforward") || layer_id.contains("mlp") {
            "feedforward".to_string()
        } else if layer_id.contains("embedding") {
            "embedding".to_string()
        } else {
            "unknown".to_string()
        }
    }

    fn calculate_lr_distribution_health(
        &self,
        recommendations: &HashMap<String, LayerLRRecommendation>,
    ) -> f64 {
        let lr_ratios: Vec<f64> = recommendations
            .values()
            .map(|rec| rec.recommended_lr / rec.current_lr)
            .collect();

        if lr_ratios.is_empty() {
            return 0.5;
        }

        let mean_ratio = lr_ratios.iter().sum::<f64>() / lr_ratios.len() as f64;
        let variance = lr_ratios.iter().map(|&x| (x - mean_ratio).powi(2)).sum::<f64>()
            / lr_ratios.len() as f64;

        1.0 / (1.0 + variance) // Better health for lower variance
    }

    fn calculate_gradient_flow_quality(
        &self,
        recommendations: &HashMap<String, LayerLRRecommendation>,
    ) -> f64 {
        recommendations
            .values()
            .map(|rec| rec.layer_metrics.stability_score)
            .sum::<f64>()
            / recommendations.len() as f64
    }

    fn assess_training_stability(
        &self,
        recommendations: &HashMap<String, LayerLRRecommendation>,
        _loss_history: &[f64],
    ) -> TrainingStability {
        let stability_score = recommendations
            .values()
            .map(|rec| rec.layer_metrics.stability_score)
            .sum::<f64>()
            / recommendations.len() as f64;

        TrainingStability {
            stability_score,
            instability_indicators: vec![],
            stability_trends: vec![],
            predicted_stability: stability_score * 0.9, // Slightly pessimistic prediction
        }
    }

    fn generate_global_adjustments(
        &self,
        _recommendations: &HashMap<String, LayerLRRecommendation>,
    ) -> Vec<GlobalLRAdjustment> {
        vec![GlobalLRAdjustment {
            adjustment_type: GlobalAdjustmentType::LayerTypeSpecific,
            magnitude: 1.5,
            expected_impact: 0.1,
            priority: AdjustmentPriority::Medium,
            instructions: "Apply different learning rates to attention vs feedforward layers"
                .to_string(),
        }]
    }

    fn identify_critical_issues(
        &self,
        recommendations: &HashMap<String, LayerLRRecommendation>,
    ) -> Vec<String> {
        let mut issues = Vec::new();

        for recommendation in recommendations.values() {
            if matches!(recommendation.urgency, AdaptationUrgency::Critical) {
                issues.push(format!(
                    "Critical learning rate issue in layer {}",
                    recommendation.layer_id
                ));
            }
        }

        issues
    }

    fn generate_advanced_insights(&self) -> HashMap<String, String> {
        let mut insights = HashMap::new();

        insights.insert(
            "total_lr_analyses".to_string(),
            self.lr_analysis_results.len().to_string(),
        );
        insights.insert(
            "total_sensitivity_analyses".to_string(),
            self.sensitivity_analysis_results.len().to_string(),
        );

        if let Some(latest_lr) = self.lr_analysis_results.last() {
            insights.insert(
                "latest_lr_efficiency".to_string(),
                format!("{:.2}", latest_lr.global_lr_insights.overall_efficiency),
            );
        }

        insights
    }
}

/// Comprehensive advanced ML debugging report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedMLDebuggingReport {
    pub timestamp: DateTime<Utc>,
    pub config: AdvancedMLDebuggingConfig,
    pub lr_analysis_count: usize,
    pub sensitivity_analysis_count: usize,
    pub recent_lr_analyses: Vec<LayerWiseLRAnalysisResult>,
    pub recent_sensitivity_analyses: Vec<ModelSensitivityAnalysisResult>,
    pub advanced_insights: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Wave 6c debug-sweep2 honesty regressions ------------------------

    #[tokio::test]
    async fn model_sensitivity_analysis_refuses_instead_of_inventing_curves() {
        let config = AdvancedMLDebuggingConfig::default();
        let mut debugger = AdvancedMLDebugger::new(config);
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), 0.001);
        let err = debugger
            .analyze_model_sensitivity(&params, &[0.9, 0.91, 0.92], &HashMap::new())
            .await
            .expect_err("a single operating point cannot yield a sensitivity analysis");
        let msg = err.to_string();
        assert!(
            msg.contains("not implemented"),
            "must name what is missing: {msg}"
        );
        assert!(
            msg.contains("perturbation response"),
            "must explain why it is unmeasurable: {msg}"
        );
    }

    #[tokio::test]
    async fn layer_lr_recommendation_confidence_is_absent_not_a_constant() {
        let config = AdvancedMLDebuggingConfig::default();
        let mut debugger = AdvancedMLDebugger::new(config);
        let mut layer_gradients = HashMap::new();
        let mut layer_weights = HashMap::new();
        let gradients = ArrayD::from_shape_vec(
            scirs2_core::ndarray::IxDyn(&[4]),
            vec![0.1f32, -0.2, 0.3, -0.4],
        )
        .expect("gradient tensor");
        let weights = ArrayD::from_shape_vec(
            scirs2_core::ndarray::IxDyn(&[4]),
            vec![1.0f32, 2.0, 3.0, 4.0],
        )
        .expect("weight tensor");
        layer_gradients.insert("layer0".to_string(), gradients);
        layer_weights.insert("layer0".to_string(), weights);

        let result = debugger
            .analyze_layer_wise_learning_rates(
                &layer_gradients,
                &layer_weights,
                0.001,
                &[1.0, 0.9, 0.8],
            )
            .await
            .expect("analysis should run");
        for rec in result.layer_lr_recommendations.values() {
            assert_eq!(
                rec.confidence, None,
                "a one-snapshot recommendation has no statistical confidence (was 0.8)"
            );
        }
    }

    #[tokio::test]
    async fn test_advanced_ml_debugger_creation() {
        let config = AdvancedMLDebuggingConfig::default();
        let debugger = AdvancedMLDebugger::new(config);
        assert_eq!(debugger.lr_analysis_results.len(), 0);
    }

    #[tokio::test]
    async fn test_layer_wise_lr_analysis() {
        let config = AdvancedMLDebuggingConfig::default();
        let mut debugger = AdvancedMLDebugger::new(config);

        let mut layer_gradients = HashMap::new();
        let mut layer_weights = HashMap::new();

        // Create test data
        let gradients =
            ArrayD::from_shape_vec(vec![10, 10], (0..100).map(|x| x as f32 * 0.01).collect())
                .expect("operation failed in test");
        let weights =
            ArrayD::from_shape_vec(vec![10, 10], (0..100).map(|x| x as f32 * 0.1).collect())
                .expect("operation failed in test");

        layer_gradients.insert("layer_0".to_string(), gradients);
        layer_weights.insert("layer_0".to_string(), weights);

        let loss_history = vec![1.0, 0.8, 0.6, 0.5];

        let result = debugger
            .analyze_layer_wise_learning_rates(
                &layer_gradients,
                &layer_weights,
                0.001,
                &loss_history,
            )
            .await;
        assert!(result.is_ok());

        let analysis = result.expect("operation failed in test");
        assert_eq!(analysis.layer_lr_recommendations.len(), 1);
        assert!(analysis.layer_lr_recommendations.contains_key("layer_0"));
    }
}
