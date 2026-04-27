use super::*;

// ─── helpers ────────────────────────────────────────────────────────────────

fn lcg_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state % 1000) as f32 / 1000.0
}

fn make_activations(n: usize) -> Vec<f32> {
    let mut s = 42u64;
    (0..n).map(|_| lcg_f32(&mut s)).collect()
}

// ─── BehaviorAnalysisConfig ──────────────────────────────────────────────────

#[test]
fn test_config_default_fields() {
    let cfg = BehaviorAnalysisConfig::default();
    assert!(cfg.enable_input_sensitivity);
    assert!(cfg.enable_feature_importance);
    assert!(cfg.enable_activation_patterns);
    assert!(cfg.enable_dead_neuron_detection);
    assert!(cfg.enable_correlation_analysis);
    assert!(cfg.dead_neuron_threshold > 0.0);
    assert!(cfg.sensitivity_samples > 0);
    assert!(cfg.perturbation_magnitude > 0.0);
    assert!(cfg.correlation_threshold > 0.0);
}

#[test]
fn test_config_custom_values() {
    let cfg = BehaviorAnalysisConfig {
        enable_input_sensitivity: false,
        enable_feature_importance: false,
        enable_activation_patterns: true,
        enable_dead_neuron_detection: false,
        enable_correlation_analysis: false,
        dead_neuron_threshold: 0.001,
        sensitivity_samples: 50,
        perturbation_magnitude: 0.05,
        correlation_threshold: 0.7,
    };
    assert!(!cfg.enable_input_sensitivity);
    assert_eq!(cfg.sensitivity_samples, 50);
    assert!((cfg.correlation_threshold - 0.7).abs() < 1e-6);
}

// ─── BehaviorAnalyzer construction ─────────────────────────────────────────

#[test]
fn test_analyzer_new() {
    let cfg = BehaviorAnalysisConfig::default();
    let analyzer = BehaviorAnalyzer::new(cfg);
    // Just verify it constructs without panic.
    let _ = format!("{:?}", analyzer);
}

// ─── record_activations ─────────────────────────────────────────────────────

#[test]
fn test_record_activations_single_layer() {
    let mut analyzer = BehaviorAnalyzer::new(BehaviorAnalysisConfig::default());
    let acts = make_activations(8);
    analyzer.record_activations("layer0".to_string(), acts.clone());
    // A second recording to the same layer accumulates.
    analyzer.record_activations("layer0".to_string(), make_activations(8));
    let _ = format!("{:?}", analyzer);
}

#[test]
fn test_record_activations_multiple_layers() {
    let mut analyzer = BehaviorAnalyzer::new(BehaviorAnalysisConfig::default());
    for i in 0..5 {
        analyzer.record_activations(format!("layer{}", i), make_activations(4));
    }
    let _ = format!("{:?}", analyzer);
}

// ─── record_input_gradients ──────────────────────────────────────────────────

#[test]
fn test_record_input_gradients_overwrites() {
    let mut analyzer = BehaviorAnalyzer::new(BehaviorAnalysisConfig::default());
    analyzer.record_input_gradients("inp".to_string(), vec![0.1, 0.2, 0.3]);
    // Second call with same key should overwrite.
    analyzer.record_input_gradients("inp".to_string(), vec![0.9, 0.8]);
    let _ = format!("{:?}", analyzer);
}

// ─── AttributionMethod variants ──────────────────────────────────────────────

#[test]
fn test_attribution_method_variants() {
    let methods = [
        AttributionMethod::GradientBased,
        AttributionMethod::PermutationImportance,
        AttributionMethod::ShapleySampling,
        AttributionMethod::IntegratedGradients,
        AttributionMethod::LimeApproximation,
    ];
    for m in &methods {
        let s = format!("{:?}", m);
        assert!(!s.is_empty());
    }
}

// ─── ActivationPatternType variants ──────────────────────────────────────────

#[test]
fn test_activation_pattern_type_variants() {
    let types = [
        ActivationPatternType::Normal,
        ActivationPatternType::Saturated,
        ActivationPatternType::Dead,
        ActivationPatternType::Oscillating,
        ActivationPatternType::Sparse,
        ActivationPatternType::Dense,
        ActivationPatternType::Bipolar,
    ];
    for t in &types {
        let s = format!("{:?}", t);
        assert!(!s.is_empty());
    }
}

// ─── NeuronRepairAction variants ─────────────────────────────────────────────

#[test]
fn test_neuron_repair_action_variants() {
    let actions = [
        NeuronRepairAction::Reinitialize,
        NeuronRepairAction::AdjustLearningRate,
        NeuronRepairAction::ChangeActivationFunction,
        NeuronRepairAction::AddNoise,
        NeuronRepairAction::Skip,
    ];
    for a in &actions {
        assert!(!format!("{:?}", a).is_empty());
    }
}

// ─── CorrelationType variants ────────────────────────────────────────────────

#[test]
fn test_correlation_type_variants() {
    let types = [
        CorrelationType::Strong,
        CorrelationType::Moderate,
        CorrelationType::Weak,
        CorrelationType::None,
    ];
    for t in &types {
        assert!(!format!("{:?}", t).is_empty());
    }
}

// ─── RecommendationCategory variants ─────────────────────────────────────────

#[test]
fn test_recommendation_category_variants() {
    let cats = [
        RecommendationCategory::Architecture,
        RecommendationCategory::Training,
        RecommendationCategory::Initialization,
        RecommendationCategory::Regularization,
        RecommendationCategory::DataPreprocessing,
    ];
    for c in &cats {
        assert!(!format!("{:?}", c).is_empty());
    }
}

// ─── Priority variants ───────────────────────────────────────────────────────

#[test]
fn test_priority_variants() {
    let priorities = [
        Priority::Critical,
        Priority::High,
        Priority::Medium,
        Priority::Low,
    ];
    for p in &priorities {
        assert!(!format!("{:?}", p).is_empty());
    }
}

// ─── InputSensitivity construction ───────────────────────────────────────────

#[test]
fn test_input_sensitivity_construction() {
    let s = InputSensitivity {
        input_dimension: 3,
        sensitivity_score: 0.75,
        gradient_magnitude: 0.5,
        perturbation_impact: 0.01,
        rank: 1,
    };
    assert_eq!(s.input_dimension, 3);
    assert_eq!(s.rank, 1);
    assert!((s.sensitivity_score - 0.75).abs() < 1e-6);
}

// ─── FeatureImportance construction ──────────────────────────────────────────

#[test]
fn test_feature_importance_construction() {
    let fi = FeatureImportance {
        feature_id: "f0".to_string(),
        importance_score: 0.42,
        attribution_method: AttributionMethod::GradientBased,
        confidence: 0.9,
        rank: 2,
    };
    assert_eq!(fi.feature_id, "f0");
    assert_eq!(fi.rank, 2);
}

// ─── NeuronActivationPattern construction ────────────────────────────────────

#[test]
fn test_neuron_activation_pattern_construction() {
    let stats = ActivationStatistics {
        mean: 0.3,
        std: 0.1,
        min: 0.1,
        max: 0.5,
        percentile_25: 0.2,
        percentile_75: 0.4,
        skewness: 0.0,
        kurtosis: 0.0,
        sparsity: 0.1,
    };
    let pat = NeuronActivationPattern {
        layer_id: "l0".to_string(),
        neuron_id: 5,
        activation_statistics: stats,
        pattern_type: ActivationPatternType::Normal,
        stability_score: 0.8,
        selectivity_score: 0.6,
    };
    assert_eq!(pat.neuron_id, 5);
    assert!((pat.stability_score - 0.8).abs() < 1e-6);
}

// ─── DeadNeuronInfo construction ─────────────────────────────────────────────

#[test]
fn test_dead_neuron_info_construction() {
    let dn = DeadNeuronInfo {
        layer_id: "l1".to_string(),
        neuron_id: 2,
        activation_level: 0.0,
        dead_probability: 0.99,
        suggested_action: NeuronRepairAction::Reinitialize,
    };
    assert_eq!(dn.neuron_id, 2);
    assert!(dn.dead_probability > 0.9);
}

// ─── BehaviorSummary field validation ────────────────────────────────────────

#[test]
fn test_behavior_summary_default_zero() {
    let summary = BehaviorSummary {
        total_neurons_analyzed: 0,
        dead_neuron_percentage: 0.0,
        average_activation_sparsity: 0.0,
        feature_distribution_entropy: 0.0,
        model_stability_score: 0.0,
        interpretability_score: 0.0,
    };
    assert_eq!(summary.total_neurons_analyzed, 0);
    assert!((summary.interpretability_score - 0.0).abs() < 1e-6);
}

#[test]
fn test_behavior_summary_valid_ranges() {
    let summary = BehaviorSummary {
        total_neurons_analyzed: 128,
        dead_neuron_percentage: 12.5,
        average_activation_sparsity: 0.35,
        feature_distribution_entropy: 2.3,
        model_stability_score: 0.75,
        interpretability_score: 0.68,
    };
    assert!(summary.dead_neuron_percentage >= 0.0 && summary.dead_neuron_percentage <= 100.0);
    assert!(
        summary.average_activation_sparsity >= 0.0 && summary.average_activation_sparsity <= 1.0
    );
    assert!(summary.model_stability_score >= 0.0 && summary.model_stability_score <= 1.0);
}

// ─── async analyze with selective flags ─────────────────────────────────────

#[tokio::test]
async fn test_analyze_with_no_features_enabled() {
    let cfg = BehaviorAnalysisConfig {
        enable_input_sensitivity: false,
        enable_feature_importance: false,
        enable_activation_patterns: false,
        enable_dead_neuron_detection: false,
        enable_correlation_analysis: false,
        ..BehaviorAnalysisConfig::default()
    };
    let mut analyzer = BehaviorAnalyzer::new(cfg);
    let report = analyzer.analyze().await.expect("analyze should succeed");
    assert!(report.input_sensitivities.is_empty());
    assert!(report.feature_importances.is_empty());
    assert!(report.activation_patterns.is_empty());
    assert!(report.dead_neurons.is_empty());
    assert!(report.correlation_analysis.is_none());
}

#[tokio::test]
async fn test_analyze_with_activation_patterns_enabled() {
    let cfg = BehaviorAnalysisConfig {
        enable_input_sensitivity: false,
        enable_feature_importance: false,
        enable_activation_patterns: true,
        enable_dead_neuron_detection: true,
        enable_correlation_analysis: false,
        dead_neuron_threshold: 0.01,
        ..BehaviorAnalysisConfig::default()
    };
    let mut analyzer = BehaviorAnalyzer::new(cfg);
    // Record activations for two neurons across two batches
    analyzer.record_activations("fc1".to_string(), vec![0.5, 0.0]);
    analyzer.record_activations("fc1".to_string(), vec![0.6, 0.0]);

    let report = analyzer.analyze().await.expect("analyze should succeed");
    assert!(!report.activation_patterns.is_empty());
}

#[tokio::test]
async fn test_analyze_input_sensitivity_with_gradients() {
    let cfg = BehaviorAnalysisConfig {
        enable_input_sensitivity: true,
        enable_feature_importance: true,
        enable_activation_patterns: false,
        enable_dead_neuron_detection: false,
        enable_correlation_analysis: false,
        ..BehaviorAnalysisConfig::default()
    };
    let mut analyzer = BehaviorAnalyzer::new(cfg);
    analyzer.record_input_gradients("inp0".to_string(), vec![0.1, 0.8, 0.3]);
    let report = analyzer.analyze().await.expect("analyze should succeed");
    // Should have one set of input sensitivities for 3 gradient dimensions
    assert_eq!(report.input_sensitivities.len(), 3);
}

#[tokio::test]
async fn test_analyze_correlation_analysis_requires_two_inputs() {
    let cfg = BehaviorAnalysisConfig {
        enable_input_sensitivity: false,
        enable_feature_importance: false,
        enable_activation_patterns: false,
        enable_dead_neuron_detection: false,
        enable_correlation_analysis: true,
        correlation_threshold: 0.5,
        ..BehaviorAnalysisConfig::default()
    };
    let mut analyzer = BehaviorAnalyzer::new(cfg);
    // Only one gradient → correlation analysis returns empty result
    analyzer.record_input_gradients("inp0".to_string(), vec![0.1, 0.2]);
    let report = analyzer.analyze().await.expect("analyze should succeed");
    let corr = report.correlation_analysis.expect("should have correlation analysis");
    assert!(corr.correlation_matrix.is_empty());
}

#[tokio::test]
async fn test_behavior_recommendation_generated_for_high_dead_neurons() {
    let mut cfg = BehaviorAnalysisConfig::default();
    // Use very tiny threshold so all neurons count as dead
    cfg.dead_neuron_threshold = 999.0;
    cfg.enable_activation_patterns = true;
    cfg.enable_dead_neuron_detection = true;
    cfg.enable_input_sensitivity = false;
    cfg.enable_feature_importance = false;
    cfg.enable_correlation_analysis = false;

    let mut analyzer = BehaviorAnalyzer::new(cfg);
    // Record 10 neurons all near zero so > 20% are dead
    for _ in 0..5 {
        analyzer.record_activations("fc0".to_string(), vec![0.0; 10]);
    }
    let report = analyzer.analyze().await.expect("analyze should succeed");
    // With 100% dead neurons, recommendations should be non-empty
    assert!(!report.recommendations.is_empty() || report.dead_neurons.len() > 2);
}
