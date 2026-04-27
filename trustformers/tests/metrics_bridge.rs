// Integration tests for the metrics bridge layer.
// Asserts that NlpAdapter.compute() is numerically identical to direct evaluation::metrics calls.

use scirs2_core::ndarray::{ArrayD, IxDyn};
use trustformers::auto::metrics::{Metric, MetricInput};
use trustformers::evaluation::bridge::{NlpAdapter, TensorAdapter};
use trustformers::evaluation::{bleu_score, exact_match_score, rouge_l_score, token_f1_score};
use trustformers_core::Tensor;

// Integration test: NlpAdapter::bleu matches evaluation::metrics::bleu_score directly
#[test]
fn test_nlp_adapter_bleu_matches_direct_fn() {
    let hyp = "the cat sat on the mat";
    let ref_ = "the cat is on the mat";

    let hyp_tokens: Vec<&str> = hyp.split_whitespace().collect();
    let ref_tokens: Vec<&str> = ref_.split_whitespace().collect();
    let refs_slice: &[&[&str]] = &[ref_tokens.as_slice()];
    let direct_score =
        bleu_score(&hyp_tokens, refs_slice, 4, true).expect("direct bleu_score failed");

    let mut adapter = NlpAdapter::new_bleu(4, true);
    adapter
        .add_batch(
            &MetricInput::Text(vec![hyp.to_string()]),
            &MetricInput::Text(vec![ref_.to_string()]),
        )
        .expect("add_batch failed");
    let adapter_result = adapter.compute().expect("adapter compute failed");

    assert!(
        (adapter_result.value - direct_score).abs() < 1e-9,
        "NlpAdapter BLEU={} differs from direct BLEU={}",
        adapter_result.value,
        direct_score
    );
}

// Integration test: NlpAdapter::rouge_l matches evaluation::metrics::rouge_l_score directly
#[test]
fn test_nlp_adapter_rouge_l_matches_direct_fn() {
    let hyp = "police killed the gunman";
    let ref_ = "police kill the gunman";

    let hyp_tokens: Vec<&str> = hyp.split_whitespace().collect();
    let ref_tokens: Vec<&str> = ref_.split_whitespace().collect();
    let direct_score =
        rouge_l_score(&hyp_tokens, &ref_tokens).expect("direct rouge_l_score failed");

    let mut adapter = NlpAdapter::new_rouge_l();
    adapter
        .add_batch(
            &MetricInput::Text(vec![hyp.to_string()]),
            &MetricInput::Text(vec![ref_.to_string()]),
        )
        .expect("add_batch failed");
    let adapter_result = adapter.compute().expect("adapter compute failed");

    assert!(
        (adapter_result.value - direct_score.f1).abs() < 1e-9,
        "NlpAdapter ROUGE-L={} differs from direct ROUGE-L={}",
        adapter_result.value,
        direct_score.f1
    );
}

// Integration test: NlpAdapter::token_f1 matches evaluation::metrics::token_f1_score directly
#[test]
fn test_nlp_adapter_token_f1_matches_direct_fn() {
    let hyp = "the cat sat";
    let ref_ = "the cat is on the mat";

    let hyp_tokens: Vec<&str> = hyp.split_whitespace().collect();
    let ref_tokens: Vec<&str> = ref_.split_whitespace().collect();
    let direct_score =
        token_f1_score(&hyp_tokens, &ref_tokens).expect("direct token_f1_score failed");

    let mut adapter = NlpAdapter::new_token_f1();
    adapter
        .add_batch(
            &MetricInput::Text(vec![hyp.to_string()]),
            &MetricInput::Text(vec![ref_.to_string()]),
        )
        .expect("add_batch failed");
    let adapter_result = adapter.compute().expect("adapter compute failed");

    assert!(
        (adapter_result.value - direct_score.f1).abs() < 1e-9,
        "NlpAdapter token-F1={} differs from direct token-F1={}",
        adapter_result.value,
        direct_score.f1
    );
}

// Integration test: NlpAdapter::exact_match matches evaluation::metrics::exact_match_score
#[test]
fn test_nlp_adapter_exact_match_matches_direct_fn() {
    let preds = ["Paris", "Tokyo", "berlin"];
    let refs = ["paris", "london", "Berlin"];

    let direct_score = exact_match_score(&preds, &refs).expect("direct exact_match_score failed");

    let mut adapter = NlpAdapter::new_exact_match();
    adapter
        .add_batch(
            &MetricInput::Text(preds.iter().map(|s| s.to_string()).collect()),
            &MetricInput::Text(refs.iter().map(|s| s.to_string()).collect()),
        )
        .expect("add_batch failed");
    let adapter_result = adapter.compute().expect("adapter compute failed");

    assert!(
        (adapter_result.value - direct_score).abs() < 1e-9,
        "NlpAdapter exact-match={} differs from direct exact-match={}",
        adapter_result.value,
        direct_score
    );
}

// Integration test: TensorAdapter::accuracy on a small i64-label fixture
#[test]
fn test_tensor_adapter_accuracy_on_fixture() {
    let pred_data = vec![0i64, 1i64, 2i64, 1i64];
    let target_data = vec![0i64, 1i64, 2i64, 0i64];

    let pred_arr =
        ArrayD::from_shape_vec(IxDyn(&[4]), pred_data).expect("shape vec failed in test");
    let target_arr =
        ArrayD::from_shape_vec(IxDyn(&[4]), target_data).expect("shape vec failed in test");

    let pred_tensor = Tensor::I64(pred_arr);
    let target_tensor = Tensor::I64(target_arr.clone());

    let mut adapter = TensorAdapter::new_accuracy();
    adapter
        .add_batch(
            &MetricInput::Tensors {
                predictions: pred_tensor,
                targets: target_tensor,
            },
            &MetricInput::Classifications(vec![]),
        )
        .expect("add_batch failed");

    let result = adapter.compute().expect("compute failed");
    // 3 correct out of 4 → 0.75
    assert!(
        (result.value - 0.75).abs() < 1e-5,
        "expected accuracy=0.75, got {}",
        result.value
    );
}
