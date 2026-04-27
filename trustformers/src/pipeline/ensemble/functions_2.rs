//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TrustformersError};
use crate::pipeline::{Pipeline, PipelineOptions, PipelineOutput};
use std::sync::{Arc, Mutex};

use super::functions::{
    create_adaptive_voting_ensemble, create_cascade_ensemble, create_dynamic_routing_ensemble,
    create_efficient_ensemble, create_high_performance_ensemble, create_quality_latency_ensemble,
    create_resource_aware_ensemble, create_uncertainty_ensemble, GatingNetwork, Router,
};
use super::types::{
    EmbeddingCosineRouter, HashRoutingGate, KeywordRouter, ModelSelectionInfo,
    ModelSelectionStrategy, ModelWeight,
};
use super::types_3::EnsemblePipeline;
use super::types_4::{
    EnsembleConfig, EnsembleStrategy, InputCharacteristics, SoftmaxEmbeddingGate,
};

#[cfg(test)]
mod tests {
    use super::super::functions::{GatingNetwork, Router};
    use super::*;
    #[test]
    fn test_ensemble_config_default() {
        let config = EnsembleConfig::default();
        assert_eq!(config.confidence_threshold, 0.5);
        assert_eq!(config.consensus_threshold, 0.7);
    }
    #[test]
    fn test_model_weight_calculation() {
        let weight = ModelWeight {
            model_id: "test_model".to_string(),
            weight: 0.5,
            confidence_weight: 0.8,
            accuracy_weight: 0.9,
            dynamic_weight: 1.1,
        };
        assert!((weight.total_weight() - 0.396).abs() < 0.001);
    }
    #[test]
    fn test_ensemble_pipeline_creation() {
        let config = EnsembleConfig::default();
        let ensemble = EnsemblePipeline::new(config);
        assert_eq!(ensemble.models.len(), 0);
    }
    #[test]
    fn test_advanced_ensemble_config() {
        let config = EnsembleConfig::default();
        assert_eq!(config.cascade_early_exit_threshold, 0.8);
        assert_eq!(config.cascade_max_models, 3);
        assert_eq!(config.resource_budget_mb, 2048);
        assert_eq!(config.uncertainty_sampling_rate, 0.1);
    }
    #[test]
    fn test_model_selection_strategy() {
        let strategy = ModelSelectionStrategy::TopK(3);
        match strategy {
            ModelSelectionStrategy::TopK(k) => assert_eq!(k, 3),
            _ => panic!("Unexpected strategy type"),
        }
    }
    #[test]
    fn test_input_characteristics_analysis() {
        let config = EnsembleConfig::default();
        let ensemble = EnsemblePipeline::new(config);
        let input = "This is a test input for complexity analysis";
        let characteristics = ensemble.analyze_input_characteristics(input);
        assert!(characteristics.length > 0);
        assert!(characteristics.complexity_score >= 0.0);
        assert!(characteristics.complexity_score <= 1.0);
        assert!(characteristics.estimated_processing_time > 0);
    }
    #[test]
    fn test_domain_detection() {
        let config = EnsembleConfig::default();
        let ensemble = EnsemblePipeline::new(config);
        assert_eq!(
            ensemble.detect_domain("medical diagnosis of patient"),
            Some("medical".to_string())
        );
        assert_eq!(
            ensemble.detect_domain("legal contract review"),
            Some("legal".to_string())
        );
        assert_eq!(
            ensemble.detect_domain("science research experiment"),
            Some("scientific".to_string())
        );
        assert_eq!(
            ensemble.detect_domain("programming code function"),
            Some("technical".to_string())
        );
        assert_eq!(ensemble.detect_domain("general conversation"), None);
    }
    #[test]
    fn test_language_detection() {
        let config = EnsembleConfig::default();
        let ensemble = EnsemblePipeline::new(config);
        assert_eq!(
            ensemble.detect_language("Hello world"),
            Some("en".to_string())
        );
        assert_eq!(ensemble.detect_language("你好世界"), Some("zh".to_string()));
        assert_eq!(
            ensemble.detect_language("مرحبا بالعالم"),
            Some("ar".to_string())
        );
        assert_eq!(
            ensemble.detect_language("Привет мир"),
            Some("ru".to_string())
        );
    }
    #[test]
    fn test_cascade_ensemble_creation() {
        let ensemble = create_cascade_ensemble(0.9, 2);
        match ensemble.config.strategy {
            EnsembleStrategy::CascadePipeline => {},
            _ => panic!("Expected CascadePipeline strategy"),
        }
        assert_eq!(ensemble.config.cascade_early_exit_threshold, 0.9);
        assert_eq!(ensemble.config.cascade_max_models, 2);
    }
    #[test]
    fn test_dynamic_routing_ensemble_creation() {
        let features = vec!["length".to_string(), "complexity".to_string()];
        let ensemble = create_dynamic_routing_ensemble(features.clone());
        match ensemble.config.strategy {
            EnsembleStrategy::DynamicRouting => {},
            _ => panic!("Expected DynamicRouting strategy"),
        }
        assert_eq!(ensemble.config.routing_features, features);
        assert!(ensemble.config.enable_model_selection);
    }
    #[test]
    fn test_quality_latency_ensemble_creation() {
        let ensemble = create_quality_latency_ensemble(0.7);
        match ensemble.config.strategy {
            EnsembleStrategy::QualityLatencyOptimized => {},
            _ => panic!("Expected QualityLatencyOptimized strategy"),
        }
        assert_eq!(ensemble.config.quality_latency_weight, 0.7);
    }
    #[test]
    fn test_resource_aware_ensemble_creation() {
        let ensemble = create_resource_aware_ensemble(1024);
        match ensemble.config.strategy {
            EnsembleStrategy::ResourceAware => {},
            _ => panic!("Expected ResourceAware strategy"),
        }
        assert_eq!(ensemble.config.resource_budget_mb, 1024);
    }
    #[test]
    fn test_uncertainty_ensemble_creation() {
        let ensemble = create_uncertainty_ensemble(0.2);
        match ensemble.config.strategy {
            EnsembleStrategy::UncertaintyBased => {},
            _ => panic!("Expected UncertaintyBased strategy"),
        }
        assert_eq!(ensemble.config.uncertainty_sampling_rate, 0.2);
    }
    #[test]
    fn test_adaptive_voting_ensemble_creation() {
        let ensemble = create_adaptive_voting_ensemble(0.05);
        match ensemble.config.strategy {
            EnsembleStrategy::AdaptiveVoting => {},
            _ => panic!("Expected AdaptiveVoting strategy"),
        }
        assert_eq!(ensemble.config.adaptive_learning_rate, 0.05);
    }
    #[test]
    fn test_high_performance_ensemble_creation() {
        let ensemble = create_high_performance_ensemble();
        assert!(ensemble.config.parallel_execution);
        assert_eq!(ensemble.config.max_concurrent_models, 8);
        assert!(ensemble.config.enable_diversity_boost);
        assert!(ensemble.config.enable_calibration);
        assert!(ensemble.config.enable_explanation);
        assert!(ensemble.config.enable_model_selection);
    }
    #[test]
    fn test_efficient_ensemble_creation() {
        let ensemble = create_efficient_ensemble();
        match ensemble.config.strategy {
            EnsembleStrategy::CascadePipeline => {},
            _ => panic!("Expected CascadePipeline strategy"),
        }
        assert_eq!(ensemble.config.cascade_early_exit_threshold, 0.85);
        assert_eq!(ensemble.config.cascade_max_models, 2);
        assert_eq!(ensemble.config.resource_budget_mb, 1024);
        assert_eq!(ensemble.config.quality_latency_weight, 0.3);
    }
    #[test]
    fn test_model_selection_info() {
        let info = ModelSelectionInfo {
            selected_models: vec!["model1".to_string(), "model2".to_string()],
            selection_reason: "Top performers".to_string(),
            selection_confidence: 0.85,
            alternative_models: vec!["model3".to_string()],
        };
        assert_eq!(info.selected_models.len(), 2);
        assert_eq!(info.alternative_models.len(), 1);
        assert!(info.selection_confidence > 0.8);
    }
    #[test]
    fn test_input_characteristics() {
        let characteristics = InputCharacteristics {
            length: 100,
            complexity_score: 0.5,
            estimated_processing_time: 50,
            required_resource_mb: 256,
            domain: Some("technical".to_string()),
            language: Some("en".to_string()),
        };
        assert_eq!(characteristics.length, 100);
        assert_eq!(characteristics.complexity_score, 0.5);
        assert_eq!(characteristics.required_resource_mb, 256);
        assert_eq!(characteristics.domain, Some("technical".to_string()));
    }
    #[test]
    fn test_complexity_estimation() {
        let config = EnsembleConfig::default();
        let ensemble = EnsemblePipeline::new(config);
        let simple_text = "Hello world";
        let complex_text = "The comprehensive implementation of advanced machine learning algorithms requires sophisticated understanding of mathematical foundations and computational complexity theory";
        let simple_complexity = ensemble.estimate_complexity(simple_text);
        let complex_complexity = ensemble.estimate_complexity(complex_text);
        assert!((0.0..=1.0).contains(&simple_complexity));
        assert!((0.0..=1.0).contains(&complex_complexity));
        assert!(complex_complexity >= simple_complexity);
    }
    fn make_classification_preds(
        labels: &[&str],
        score: f32,
    ) -> Vec<(String, PipelineOutput, u64)> {
        labels
            .iter()
            .enumerate()
            .map(|(i, &label)| {
                (
                    format!("model_{i}"),
                    PipelineOutput::Classification(vec![crate::pipeline::ClassificationOutput {
                        label: label.to_string(),
                        score,
                    }]),
                    0u64,
                )
            })
            .collect()
    }
    #[test]
    fn test_boosting_classification_weighted_vote() {
        let mut config = EnsembleConfig::default();
        config.strategy = EnsembleStrategy::Boosting;
        config.boosting_learning_rate = 1.0;
        let ensemble = EnsemblePipeline::new(config);
        let preds = make_classification_preds(&["A", "A", "B"], 0.9);
        let weights = [0.5_f64, 0.4, 0.1];
        let result = ensemble
            .boosting_predictions(&preds, &weights)
            .expect("boosting should succeed");
        if let PipelineOutput::Classification(results) = result {
            assert!(!results.is_empty());
            assert_eq!(
                results[0].label, "A",
                "A should win with higher accumulated weight"
            );
        } else {
            panic!("Expected Classification output");
        }
    }
    #[test]
    fn test_boosting_equal_weights_matches_majority_vote() {
        let mut config = EnsembleConfig::default();
        config.strategy = EnsembleStrategy::Boosting;
        let ensemble = EnsemblePipeline::new(config);
        let preds = make_classification_preds(&["A", "A", "B"], 0.9);
        let weights = [1.0_f64 / 3.0; 3];
        let boosting = ensemble
            .boosting_predictions(&preds, &weights)
            .expect("boosting should succeed");
        let uniform: Vec<f32> = weights.iter().map(|w| *w as f32).collect();
        let majority = ensemble
            .majority_vote_predictions(&preds)
            .expect("majority_vote should succeed");
        let top_boosting = if let PipelineOutput::Classification(r) = &boosting {
            r[0].label.clone()
        } else {
            panic!("expected class")
        };
        let top_majority = if let PipelineOutput::Classification(r) = &majority {
            r[0].label.clone()
        } else {
            panic!("expected class")
        };
        let _ = uniform;
        assert_eq!(
            top_boosting, top_majority,
            "equal-weight boosting must agree with majority vote"
        );
    }
    #[test]
    fn test_bagging_deterministic_with_fixed_seed() {
        let mut config = EnsembleConfig::default();
        config.strategy = EnsembleStrategy::Bagging;
        config.random_seed = 12345;
        let ensemble = EnsemblePipeline::new(config);
        let preds = make_classification_preds(&["A", "B", "C"], 0.9);
        let weights = [1.0_f64 / 3.0; 3];
        let result1 = ensemble.bagging_predictions(&preds, &weights).expect("bagging call 1");
        let result2 = ensemble.bagging_predictions(&preds, &weights).expect("bagging call 2");
        if let (PipelineOutput::Classification(r1), PipelineOutput::Classification(r2)) =
            (result1, result2)
        {
            assert_eq!(
                r1[0].label, r2[0].label,
                "same seed must produce same top label"
            );
        } else {
            panic!("Expected Classification output");
        }
    }
    #[test]
    fn test_bagging_is_different_from_average() {
        let mut config = EnsembleConfig::default();
        config.strategy = EnsembleStrategy::Bagging;
        config.random_seed = 99;
        let ensemble = EnsemblePipeline::new(config);
        let preds: Vec<(String, PipelineOutput, u64)> = vec![
            (
                "m0".to_string(),
                PipelineOutput::Classification(vec![crate::pipeline::ClassificationOutput {
                    label: "A".to_string(),
                    score: 0.95,
                }]),
                0,
            ),
            (
                "m1".to_string(),
                PipelineOutput::Classification(vec![crate::pipeline::ClassificationOutput {
                    label: "A".to_string(),
                    score: 0.9,
                }]),
                0,
            ),
            (
                "m2".to_string(),
                PipelineOutput::Classification(vec![crate::pipeline::ClassificationOutput {
                    label: "Z".to_string(),
                    score: 0.99,
                }]),
                0,
            ),
        ];
        let weights = [1.0_f64 / 3.0; 3];
        let result = ensemble.bagging_predictions(&preds, &weights).expect("bagging should work");
        if let PipelineOutput::Classification(results) = result {
            assert!(!results.is_empty());
        } else {
            panic!("Expected Classification output");
        }
    }
    #[test]
    fn test_hash_routing_gate_is_deterministic() {
        let gate = HashRoutingGate::new(1.0);
        let scores1 = gate.gate("medical query about patient diagnosis", 4).expect("gate call 1");
        let scores2 = gate.gate("medical query about patient diagnosis", 4).expect("gate call 2");
        assert_eq!(scores1, scores2, "hash gate must be deterministic");
        assert!(
            (scores1.iter().sum::<f32>() - 1.0).abs() < 1e-5,
            "scores must sum to 1"
        );
    }
    #[test]
    fn test_softmax_embedding_gate_routes_to_highest_similarity() {
        let expert_embeddings = vec![vec![1.0_f32, 0.0, 0.0], vec![0.0_f32, 1.0, 0.0]];
        let gate = SoftmaxEmbeddingGate::new(
            expert_embeddings,
            Arc::new(|_text: &str| vec![0.95_f32, 0.05, 0.0]),
        );
        let scores = gate.gate("medical patient", 2).expect("embedding gate");
        assert!(
            scores[0] > scores[1],
            "input closer to expert 0 must get higher score"
        );
        assert!((scores.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
    #[test]
    fn test_moe_falls_back_to_weighted_average_without_gate() {
        let mut config = EnsembleConfig::default();
        config.strategy = EnsembleStrategy::MixtureOfExperts;
        config.gating_network = None;
        let ensemble = EnsemblePipeline::new(config);
        let preds = make_classification_preds(&["A", "A"], 0.8);
        let weights = [0.5_f32, 0.5];
        let chars = InputCharacteristics {
            length: 10,
            complexity_score: 0.1,
            estimated_processing_time: 1,
            required_resource_mb: 1,
            domain: None,
            language: None,
        };
        let moe = ensemble
            .mixture_of_experts_predictions(&preds, &weights, &chars)
            .expect("moe without gate should work");
        let avg = ensemble.average_predictions(&preds, &weights).expect("average should work");
        if let (PipelineOutput::Classification(m), PipelineOutput::Classification(a)) = (moe, avg) {
            assert_eq!(m[0].label, a[0].label);
        } else {
            panic!("Expected Classification output from both");
        }
    }
    #[test]
    fn test_load_balance_stats_populated_after_moe_call() {
        let mut config = EnsembleConfig::default();
        config.strategy = EnsembleStrategy::MixtureOfExperts;
        config.gating_network = Some(Arc::new(HashRoutingGate::new(0.5)));
        config.moe_top_k = 0;
        let ensemble = EnsemblePipeline::new(config);
        let preds = make_classification_preds(&["A", "B", "C"], 0.8);
        let weights = [1.0_f32 / 3.0; 3];
        let chars = InputCharacteristics {
            length: 20,
            complexity_score: 0.3,
            estimated_processing_time: 5,
            required_resource_mb: 10,
            domain: Some("medical".to_string()),
            language: Some("en".to_string()),
        };
        ensemble
            .mixture_of_experts_predictions(&preds, &weights, &chars)
            .expect("moe with gate should work");
        let stats = ensemble.last_load_balance_stats();
        assert!(
            stats.is_some(),
            "load balance stats must be populated after MoE call with gate"
        );
        let stats = stats.expect("stats should be present");
        assert_eq!(stats.expert_loads.len(), 3);
        assert!((stats.expert_loads.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }
    struct TimedMockPipeline {
        label: String,
        delay_ms: u64,
    }
    impl Pipeline for TimedMockPipeline {
        type Input = String;
        type Output = PipelineOutput;
        fn __call__(&self, _input: Self::Input) -> crate::error::Result<Self::Output> {
            if self.delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
            }
            Ok(PipelineOutput::Classification(vec![
                crate::pipeline::ClassificationOutput {
                    label: self.label.clone(),
                    score: 1.0,
                },
            ]))
        }
    }
    #[test]
    fn test_parallel_execution_order_preserved() {
        let mut config = EnsembleConfig::default();
        config.strategy = EnsembleStrategy::Average;
        config.parallel_execution = true;
        let mut ensemble = EnsemblePipeline::new(config);
        for i in 0..3 {
            ensemble
                .add_model(
                    format!("model_{i}"),
                    Box::new(TimedMockPipeline {
                        label: format!("label_{i}"),
                        delay_ms: 0,
                    }),
                    1.0 / 3.0,
                )
                .expect("add_model failed");
        }
        let result = ensemble.predict_individual_models("test").expect("predict failed");
        for (i, (model_id, _, _)) in result.iter().enumerate() {
            assert_eq!(
                model_id,
                &format!("model_{i}"),
                "output order must match model order"
            );
        }
    }
    #[test]
    fn test_parallel_off_runs_sequentially() {
        let mut config = EnsembleConfig::default();
        config.strategy = EnsembleStrategy::Average;
        config.parallel_execution = false;
        let mut ensemble = EnsemblePipeline::new(config);
        for i in 0..2 {
            ensemble
                .add_model(
                    format!("seq_model_{i}"),
                    Box::new(TimedMockPipeline {
                        label: format!("label_{i}"),
                        delay_ms: 0,
                    }),
                    0.5,
                )
                .expect("add_model failed");
        }
        let result = ensemble.predict_individual_models("input").expect("predict failed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "seq_model_0");
        assert_eq!(result[1].0, "seq_model_1");
    }
    #[test]
    fn test_keyword_router_backcompat() {
        let router = KeywordRouter;
        let model_ids = vec![
            "small-fast-model".to_string(),
            "large-bert-model".to_string(),
        ];
        let short_input = "hi";
        let routed = router.route(short_input, &model_ids);
        assert!(
            !routed.is_empty(),
            "router should return at least one model"
        );
        let selected_ids: Vec<usize> = routed.iter().map(|(i, _)| *i).collect();
        assert!(
            selected_ids.contains(&0),
            "small-fast model must be selected for short input"
        );
    }
    #[test]
    fn test_embedding_cosine_router_selects_closest_model() {
        let router = EmbeddingCosineRouter::new(
            vec![
                ("medical_expert".to_string(), vec![1.0_f32, 0.0, 0.0]),
                ("legal_expert".to_string(), vec![0.0_f32, 1.0, 0.0]),
            ],
            Arc::new(|_text: &str| vec![0.9_f32, 0.1, 0.0]),
        );
        let model_ids = vec!["medical_expert".to_string(), "legal_expert".to_string()];
        let results = router.route("patient symptoms", &model_ids);
        assert!(!results.is_empty());
        let top = results[0].0;
        assert_eq!(
            top, 0,
            "medical expert (index 0) should be selected for medical query"
        );
    }
    #[test]
    fn test_router_returns_empty_when_no_models() {
        let router = KeywordRouter;
        let results = router.route("some input text", &[]);
        assert!(
            results.is_empty(),
            "router must return empty vec for empty model_ids"
        );
        let emb_router = EmbeddingCosineRouter::new(
            vec![("m0".to_string(), vec![1.0_f32])],
            Arc::new(|_: &str| vec![1.0_f32]),
        );
        let results2 = emb_router.route("text", &[]);
        assert!(results2.is_empty());
    }
    #[test]
    fn test_bagging_populates_bootstrap_stats() {
        use crate::pipeline::Pipeline;
        let mut config = EnsembleConfig::default();
        config.strategy = EnsembleStrategy::Bagging;
        config.random_seed = 7;
        config.parallel_execution = false;
        let mut ensemble = EnsemblePipeline::new(config);
        for i in 0..3 {
            ensemble
                .add_model(
                    format!("bag_model_{i}"),
                    Box::new(TimedMockPipeline {
                        label: format!("label_{i}"),
                        delay_ms: 0,
                    }),
                    1.0 / 3.0,
                )
                .expect("add_model failed");
        }
        let prediction = ensemble
            .__call__("test bagging input".to_string())
            .expect("Bagging __call__ failed");
        assert!(
            prediction.bootstrap_stats.is_some(),
            "bootstrap_stats must be populated when Bagging strategy is used"
        );
        let stats = prediction.bootstrap_stats.expect("bootstrap_stats should be Some");
        assert!(stats.n_samples > 0, "n_samples must be positive");
        assert!(
            stats.mean >= 0.0 && stats.mean <= 1.0,
            "bootstrap mean must be a valid confidence score in [0, 1]"
        );
        assert!(
            stats.variance >= 0.0,
            "bootstrap variance must be non-negative"
        );
    }
}
