//! Tests for the proxy train-and-evaluate loop.

use super::*;
use std::collections::HashMap;

fn architecture(hidden_size: i32, num_layers: i32, activation: &str) -> Architecture {
    let mut arch = Architecture::new();
    arch.dimensions.insert("hidden_size".to_string(), hidden_size);
    arch.dimensions.insert("num_layers".to_string(), num_layers);
    arch.choices.insert("activation".to_string(), activation.to_string());
    arch
}

#[test]
fn test_training_actually_learns_the_task() {
    let mut evaluator = ProxyTaskEvaluator::new();
    let trained = evaluator
        .evaluate(&architecture(512, 2, "relu"))
        .expect("evaluation must succeed");

    // Chance level is 0.5; a trained two-layer network must beat it clearly.
    assert!(
        trained.accuracy > 0.7,
        "the proxy network must learn the task, got accuracy {}",
        trained.accuracy
    );
    assert!(trained.train_loss.is_finite());
    assert!(trained.trained_parameters > 0);
    assert!(
        trained.inference_seconds > 0.0,
        "inference latency must be measured with a real clock"
    );
}

#[test]
fn test_untrained_network_is_at_chance() {
    // Zero training steps: the same architecture must NOT reach trained accuracy,
    // proving the accuracy comes from training rather than from the architecture.
    let mut untrained = ProxyTaskEvaluator::with_config(ProxyTaskConfig {
        train_steps: 0,
        ..Default::default()
    });
    let untrained_result = untrained
        .evaluate(&architecture(512, 2, "relu"))
        .expect("evaluation must succeed");

    let mut trained = ProxyTaskEvaluator::new();
    let trained_result = trained
        .evaluate(&architecture(512, 2, "relu"))
        .expect("evaluation must succeed");

    assert!(
        trained_result.accuracy > untrained_result.accuracy + 0.1,
        "training must improve accuracy: trained {} vs untrained {}",
        trained_result.accuracy,
        untrained_result.accuracy
    );
}

#[test]
fn test_depth_matters_for_a_non_linear_task() {
    let mut evaluator = ProxyTaskEvaluator::new();

    // A network with no hidden layer is a linear classifier and cannot separate
    // the XOR-like boundary; a deeper one can.
    let linear = evaluator
        .evaluate(&architecture(512, 0, "relu"))
        .expect("evaluation must succeed");
    let deep = evaluator
        .evaluate(&architecture(512, 3, "relu"))
        .expect("evaluation must succeed");

    assert!(
        deep.accuracy > linear.accuracy,
        "a hidden layer must help on a non-linear task: {} vs {}",
        deep.accuracy,
        linear.accuracy
    );
}

#[test]
fn test_accuracy_is_not_a_function_of_parameter_count_alone() {
    let mut evaluator = ProxyTaskEvaluator::new();

    let relu = evaluator
        .evaluate(&architecture(512, 2, "relu"))
        .expect("evaluation must succeed");
    let tanh = evaluator
        .evaluate(&architecture(512, 2, "tanh"))
        .expect("evaluation must succeed");

    assert_eq!(
        relu.trained_parameters, tanh.trained_parameters,
        "the two candidates have the same shape"
    );
    assert!(
        (relu.accuracy - tanh.accuracy).abs() > f32::EPSILON,
        "identical parameter counts must still give different measured accuracies \
         ({} vs {}); an analytic function of the parameter count could not",
        relu.accuracy,
        tanh.accuracy
    );
}

#[test]
fn test_evaluation_is_deterministic() {
    let mut evaluator = ProxyTaskEvaluator::new();
    let first = evaluator.evaluate(&architecture(1024, 2, "gelu")).expect("evaluation");
    let second = evaluator.evaluate(&architecture(1024, 2, "gelu")).expect("evaluation");

    assert_eq!(first.accuracy, second.accuracy);
    assert_eq!(first.trained_parameters, second.trained_parameters);
}

#[test]
fn test_parameter_count_grows_with_width() {
    let mut evaluator = ProxyTaskEvaluator::new();
    let narrow = evaluator.evaluate(&architecture(16, 2, "relu")).expect("evaluation");
    let wide = evaluator.evaluate(&architecture(4096, 2, "relu")).expect("evaluation");

    assert!(
        wide.trained_parameters > narrow.trained_parameters,
        "a wider candidate must build a bigger proxy: {} vs {}",
        wide.trained_parameters,
        narrow.trained_parameters
    );
}

#[test]
fn test_empty_dataset_is_an_error() {
    let mut evaluator = ProxyTaskEvaluator::with_config(ProxyTaskConfig {
        train_samples: 0,
        eval_samples: 0,
        ..Default::default()
    });
    assert!(evaluator.evaluate(&architecture(512, 2, "relu")).is_err());
}

#[test]
fn test_custom_metrics_default_to_empty() {
    let mut evaluator = ProxyTaskEvaluator::new();
    let result = evaluator.evaluate(&architecture(512, 1, "silu")).expect("evaluation");
    assert!(result.custom_metrics.is_empty());
    assert!(evaluator.description().contains("proxy task"));
}

#[test]
fn test_measured_performance_is_comparable() {
    let mut metrics = HashMap::new();
    metrics.insert("custom".to_string(), 1.0);
    let performance = MeasuredPerformance {
        accuracy: 0.9,
        train_loss: 0.2,
        trained_parameters: 100,
        inference_seconds: 0.001,
        custom_metrics: metrics,
    };
    assert_eq!(performance.clone(), performance);
}
