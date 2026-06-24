//! # Interpretability Tools
//!
//! Comprehensive model interpretability toolkit including SHAP integration, LIME support,
//! attention analysis, feature attribution, and counterfactual generation for TrustformeRS models.
//!
//! ## Refactoring Summary
//!
//! Previously this was a single 2,803-line file containing all interpretability functionality.
//! It has been split into focused modules:
//!
//! - `interpretability/config.rs` - Configuration structures and enums (77 lines)
//! - `interpretability/shap.rs` - SHAP analysis types and functionality (66 lines)
//! - `interpretability/lime.rs` - LIME analysis types and functionality (78 lines)
//! - `interpretability/attention.rs` - Attention analysis for transformers (426 lines)
//! - `interpretability/attribution.rs` - Feature attribution methods (103 lines)
//! - `interpretability/counterfactual.rs` - Counterfactual generation (191 lines)
//! - `interpretability/analyzer.rs` - Main analyzer implementation (318 lines)
//! - `interpretability/report.rs` - Reporting functionality (23 lines)
//!
//! This refactoring improves:
//! - Code maintainability and readability
//! - Module compilation times
//! - Test isolation
//! - Code reuse through focused modules
//! - Developer experience when working on specific interpretability methods

pub use crate::interpretability::{
    ActionableInsight,
    // Attention analysis
    AttentionAnalysisResult,
    AttentionBottleneck,
    AttentionFlowAnalysis,
    AttentionFlowPath,
    AttentionHeadResult,
    AttentionInsight,
    AttentionLayerResult,
    AttentionPatterns,
    AttentionStatistics,
    AttributionMethod,

    AttributionMethodResult,
    AttributionVisualizationData,
    BlockPattern,
    BottleneckType,
    BoundaryCrossingPoint,
    ChangeDirection,
    Counterfactual,
    CounterfactualQualityMetrics,
    // Counterfactual generation
    CounterfactualResult,
    DecisionBoundaryAnalysis,
    DiagonalPattern,
    FeatureAttribution,
    // Feature attribution
    FeatureAttributionResult,
    FeatureChange,
    FeatureContribution,
    FeatureImportance,
    FeatureInteraction,
    FeatureSensitivityAnalysis,
    FlowEfficiencyMetrics,
    FlowTransformation,
    HeadCluster,
    HeadRedundancyAnalysis,
    HeadSpecializationAnalysis,
    HeadSpecializationType,
    ImplementationDifficulty,
    InsightType,

    InteractionEffect,
    InteractionEffectType,
    InteractionType,

    // Main analyzer
    InterpretabilityAnalyzer,

    // Configuration
    InterpretabilityConfig,
    // Reporting
    InterpretabilityReport,
    LayerAttentionPatterns,
    LayerAttentionStats,
    LayerFlowStats,
    LayerFlowStep,
    // LIME analysis
    LimeAnalysisResult,
    MethodAgreementAnalysis,
    NeighborhoodStats,

    PerturbationAnalysis,
    PerturbationResult,
    PruningImpact,
    PruningRecommendation,
    RedundancyType,
    RedundantHeadPair,
    RepetitivePattern,
    RiskLevel,
    // SHAP analysis
    ShapAnalysisResult,
    ShapSummary,

    SparsityDistribution,
    SpecializationEvolution,
    SpecializationTransition,
    SpecializationTrend,
    ThresholdAnalysis,
    TimeHorizon,

    TimelinePoint,
    TokenAttentionScore,
    TopFeature,
    VerticalPattern,
};

#[cfg(test)]
mod tests {
    use crate::interpretability::{
        AttributionMethod, InterpretabilityAnalyzer, InterpretabilityConfig,
    };
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_interpretability_analyzer_creation() {
        let config = InterpretabilityConfig::default();
        let _analyzer = InterpretabilityAnalyzer::new(config);
    }

    #[tokio::test]
    async fn test_shap_analysis() {
        let config = InterpretabilityConfig::default();
        let mut analyzer = InterpretabilityAnalyzer::new(config);

        let mut instance = HashMap::new();
        instance.insert("feature1".to_string(), 1.0);
        instance.insert("feature2".to_string(), 2.0);

        let model_predictions = vec![0.8, 0.7, 0.9];
        let background_data = vec![{
            let mut bg = HashMap::new();
            bg.insert("feature1".to_string(), 0.5);
            bg.insert("feature2".to_string(), 1.0);
            bg
        }];

        let result = analyzer.analyze_shap(&instance, &model_predictions, &background_data).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_lime_analysis() {
        let config = InterpretabilityConfig::default();
        let mut analyzer = InterpretabilityAnalyzer::new(config);

        let mut instance = HashMap::new();
        instance.insert("feature1".to_string(), 1.0);
        instance.insert("feature2".to_string(), 2.0);

        let model_fn: Box<dyn Fn(&HashMap<String, f64>) -> f64> =
            Box::new(|input: &HashMap<String, f64>| input.values().sum::<f64>() * 0.1);

        let result = analyzer.analyze_lime(&instance, model_fn).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_feature_attribution_integrated_gradients() {
        let config = InterpretabilityConfig {
            enable_feature_attribution: true,
            attribution_methods: vec![AttributionMethod::IntegratedGradients],
            ..InterpretabilityConfig::default()
        };
        let mut analyzer = InterpretabilityAnalyzer::new(config);

        let mut instance = HashMap::new();
        instance.insert("feature1".to_string(), 0.5);
        instance.insert("feature2".to_string(), 1.5);
        instance.insert("feature3".to_string(), 2.0);

        let model_predictions = vec![0.7, 0.6, 0.8];
        let background_data = vec![{
            let mut bg = HashMap::new();
            bg.insert("feature1".to_string(), 0.0);
            bg.insert("feature2".to_string(), 0.0);
            bg.insert("feature3".to_string(), 0.0);
            bg
        }];

        let result = analyzer.analyze_shap(&instance, &model_predictions, &background_data).await;
        assert!(result.is_ok());
        let shap_result = result.expect("SHAP analysis failed");
        assert!(
            !shap_result.feature_contributions.is_empty(),
            "attributions must be non-empty"
        );
    }
}
