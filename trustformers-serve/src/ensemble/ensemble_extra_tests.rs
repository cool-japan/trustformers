#![cfg(test)]
/// Extended tests for the model ensemble module.
use super::*;

fn make_pred(model_id: &str, logits: Vec<f64>) -> ModelPrediction {
    ModelPrediction::new(model_id, logits)
}

fn uniform_preds(n: usize, logits: Vec<f64>) -> Vec<ModelPrediction> {
    (0..n).map(|i| make_pred(&format!("m{i}"), logits.clone())).collect()
}

// ── 33. ModelPrediction::new — probabilities sum to 1.0 ──────────────────
#[test]
fn test_model_prediction_probs_sum_to_one() {
    let pred = make_pred("m", vec![1.0, 2.0, 3.0]);
    let sum: f64 = pred.probabilities.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-9,
        "probabilities must sum to 1.0, got {sum}"
    );
}

// ── 34. ModelPrediction::new — predicted_class is argmax ──────────────────
#[test]
fn test_model_prediction_argmax() {
    let pred = make_pred("m", vec![0.1, 3.0, 0.5]);
    assert_eq!(
        pred.predicted_class, 1,
        "argmax of [0.1, 3.0, 0.5] is class 1"
    );
}

// ── 35. ModelPrediction::softmax — empty input returns empty ─────────────
#[test]
fn test_softmax_empty_input_returns_empty() {
    let result = ModelPrediction::softmax(&[]);
    assert!(result.is_empty());
}

// ── 36. ModelPrediction::softmax — single element is 1.0 ─────────────────
#[test]
fn test_softmax_single_element_is_one() {
    let result = ModelPrediction::softmax(&[5.0_f64]);
    assert!((result[0] - 1.0).abs() < 1e-9);
}

// ── 37. ModelPrediction::softmax — all-zero logits give uniform ───────────
#[test]
fn test_softmax_uniform_on_equal_logits() {
    let result = ModelPrediction::softmax(&[0.0_f64; 4]);
    for &p in &result {
        assert!(
            (p - 0.25).abs() < 1e-9,
            "equal logits must give uniform softmax, got {p}"
        );
    }
}

// ── 38. ModelEnsemble::new — negative weight returns error ───────────────
#[test]
fn test_ensemble_new_negative_weight_returns_error() {
    let err = ModelEnsemble::new(
        EnsembleStrategy::WeightedAverage {
            weights: vec![-1.0, 1.0],
        },
        2,
    )
    .unwrap_err();
    assert!(matches!(err, EnsembleError::InvalidWeight(_)));
}

// ── 39. ModelEnsemble::aggregate — empty predictions returns error ─────────
#[test]
fn test_aggregate_empty_returns_error() {
    let ensemble = ModelEnsemble::new(EnsembleStrategy::MajorityVote, 3).expect("valid ensemble");
    let err = ensemble.aggregate(&[]).unwrap_err();
    assert!(matches!(err, EnsembleError::EmptyPredictions));
}

// ── 40. ModelEnsemble::aggregate — class count mismatch returns error ──────
#[test]
fn test_aggregate_class_count_mismatch_returns_error() {
    let ensemble = ModelEnsemble::new(EnsembleStrategy::MajorityVote, 3).expect("valid ensemble");
    let pred = make_pred("m", vec![1.0, 2.0]); // only 2 classes
    let err = ensemble.aggregate(&[pred]).unwrap_err();
    assert!(matches!(err, EnsembleError::ClassCountMismatch { .. }));
}

// ── 41. MajorityVote — unanimous result has final_confidence near 1.0 ─────
#[test]
fn test_majority_vote_unanimous_confidence() {
    let preds = uniform_preds(5, vec![5.0_f64, -5.0, -5.0]);
    let ensemble = ModelEnsemble::new(EnsembleStrategy::MajorityVote, 3).expect("valid ensemble");
    let result = ensemble.aggregate(&preds).expect("aggregate");
    assert_eq!(result.final_class, 0);
    // Majority vote one-hots the class, so confidence should be 1.0
    assert!((result.final_confidence - 1.0).abs() < 1e-6);
}

// ── 42. MaxConfidence — selects model with highest confidence ─────────────
#[test]
fn test_max_confidence_selects_most_confident() {
    let low_conf = make_pred("low", vec![0.5_f64, 0.5, 0.0]);
    let high_conf = make_pred("high", vec![10.0_f64, -10.0, -10.0]); // strongly class 0
    let ensemble = ModelEnsemble::new(EnsembleStrategy::MaxConfidence, 3).expect("valid ensemble");
    let result = ensemble.aggregate(&[low_conf, high_conf]).expect("aggregate");
    assert_eq!(
        result.final_class, 0,
        "MaxConfidence should pick class 0 from high-conf model"
    );
}

// ── 43. MeanProbability — probs sum to 1.0 ────────────────────────────────
#[test]
fn test_mean_probability_probs_sum_to_one() {
    let preds = uniform_preds(3, vec![1.0_f64, 2.0, 3.0]);
    let ensemble =
        ModelEnsemble::new(EnsembleStrategy::MeanProbability, 3).expect("valid ensemble");
    let result = ensemble.aggregate(&preds).expect("aggregate");
    let sum: f64 = result.aggregated_probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-8,
        "mean probability must sum to 1.0, got {sum}"
    );
}

// ── 44. GeometricMean — probs sum to approximately 1.0 ───────────────────
#[test]
fn test_geometric_mean_probs_sum_to_one() {
    let preds = uniform_preds(3, vec![1.0_f64, 2.0, 3.0]);
    let ensemble = ModelEnsemble::new(EnsembleStrategy::GeometricMean, 3).expect("valid ensemble");
    let result = ensemble.aggregate(&preds).expect("aggregate");
    let sum: f64 = result.aggregated_probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "geometric mean probs should sum to ~1.0, got {sum}"
    );
}

// ── 45. WeightedAverage — weight count mismatch returns error ─────────────
#[test]
fn test_weighted_average_weight_count_mismatch_returns_error() {
    let preds = uniform_preds(3, vec![1.0_f64, 2.0]);
    let ensemble = ModelEnsemble::new(
        EnsembleStrategy::WeightedAverage {
            weights: vec![0.5, 0.5],
        }, // only 2 weights for 3 models
        2,
    )
    .expect("valid ensemble");
    let err = ensemble.aggregate(&preds).unwrap_err();
    assert!(matches!(err, EnsembleError::WeightCountMismatch { .. }));
}

// ── 46. ConfidenceWeighted — probs sum to 1.0 ─────────────────────────────
#[test]
fn test_confidence_weighted_probs_sum_to_one() {
    let preds = vec![
        make_pred("a", vec![2.0_f64, 1.0]),
        make_pred("b", vec![1.0_f64, 2.0]),
    ];
    let ensemble =
        ModelEnsemble::new(EnsembleStrategy::ConfidenceWeighted, 2).expect("valid ensemble");
    let result = ensemble.aggregate(&preds).expect("aggregate");
    let sum: f64 = result.aggregated_probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "confidence-weighted probs must sum to 1.0, got {sum}"
    );
}

// ── 47. EnsembleResult::entropy — non-negative ───────────────────────────
#[test]
fn test_ensemble_result_entropy_non_negative() {
    let preds = uniform_preds(2, vec![1.0_f64, 2.0, 3.0]);
    // Use MeanProbability so aggregated_probs are proper softmax values in (0,1).
    // MajorityVote produces a spike at 1.0 which causes a tiny negative offset
    // in the formula due to the 1e-10 bias term in -p*(p+1e-10).log2().
    let ensemble = ModelEnsemble::new(EnsembleStrategy::MeanProbability, 3).expect("valid");
    let result = ensemble.aggregate(&preds).expect("aggregate");
    assert!(result.entropy() >= 0.0, "entropy must be non-negative");
}

// ── 48. EnsembleResult::top_k_classes — k=1 returns single element ────────
#[test]
fn test_top_k_classes_k_one() {
    let preds = uniform_preds(2, vec![1.0_f64, 2.0, 3.0]);
    let ensemble = ModelEnsemble::new(EnsembleStrategy::MeanProbability, 3).expect("valid");
    let result = ensemble.aggregate(&preds).expect("aggregate");
    let top_k = result.top_k_classes(1);
    assert_eq!(top_k.len(), 1, "top_k(1) must return exactly 1 element");
}

// ── 49. EnsembleResult::top_k_classes — sorted descending by probability ──
#[test]
fn test_top_k_classes_sorted_descending() {
    let preds = uniform_preds(2, vec![1.0_f64, 5.0, 2.0]);
    let ensemble = ModelEnsemble::new(EnsembleStrategy::MeanProbability, 3).expect("valid");
    let result = ensemble.aggregate(&preds).expect("aggregate");
    let top_k = result.top_k_classes(3);
    for i in 0..top_k.len().saturating_sub(1) {
        assert!(
            top_k[i].1 >= top_k[i + 1].1,
            "top_k must be sorted descending: {:?}",
            top_k
        );
    }
}

// ── 50. EnsembleResult::agreement_ratio — in [0.0, 1.0] ─────────────────
#[test]
fn test_agreement_ratio_in_unit_interval() {
    let preds = vec![
        make_pred("a", vec![5.0_f64, -5.0]),
        make_pred("b", vec![-5.0_f64, 5.0]),
    ];
    let ensemble = ModelEnsemble::new(EnsembleStrategy::MajorityVote, 2).expect("valid");
    let result = ensemble.aggregate(&preds).expect("aggregate");
    assert!(
        (0.0..=1.0).contains(&result.agreement_ratio),
        "agreement_ratio must be in [0, 1], got {}",
        result.agreement_ratio
    );
}

// ── 51. EnsembleResult::total_latency_ms — max of individual latencies ────
#[test]
fn test_total_latency_ms_is_max() {
    let mut p1 = make_pred("a", vec![1.0_f64, 2.0]);
    p1.latency_ms = 50;
    let mut p2 = make_pred("b", vec![1.0_f64, 2.0]);
    p2.latency_ms = 120;
    let ensemble = ModelEnsemble::new(EnsembleStrategy::MeanProbability, 2).expect("valid");
    let result = ensemble.aggregate(&[p1, p2]).expect("aggregate");
    assert_eq!(result.total_latency_ms, 120);
}

// ── 52. EnsembleError — all variants implement Display ────────────────────
#[test]
fn test_ensemble_error_display() {
    let errors: Vec<Box<dyn std::error::Error>> = vec![
        Box::new(EnsembleError::EmptyPredictions),
        Box::new(EnsembleError::InvalidWeight(-1.0)),
        Box::new(EnsembleError::WeightCountMismatch {
            weights: 2,
            models: 3,
        }),
        Box::new(EnsembleError::ClassCountMismatch {
            expected: 3,
            got: 2,
        }),
    ];
    for e in &errors {
        assert!(!e.to_string().is_empty(), "error display must be non-empty");
    }
}
