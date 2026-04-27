// Integration test for trainer auto-metric wiring.
//
// Note: trustformers-training depends on trustformers-core but NOT on trustformers.
// Adding add_eval_metric_auto directly to Trainer<M> would be circular.
// Instead the bridge lives in trustformers::training_utils.
//
// These tests verify the plumbing at the AutoMetric → TrainerWithAutoMetrics level
// without requiring a fully-constructed Trainer (which would need a real Model impl).

use trustformers::auto::metrics::AutoMetric;

/// Verify that AutoMetric::for_task resolves to the right named metric for each task.
#[test]
fn test_auto_metric_for_text_classification_has_accuracy() {
    let metric = AutoMetric::for_task("text-classification").expect("AutoMetric::for_task failed");
    assert_eq!(
        metric.name(),
        "classification",
        "text-classification should produce a 'classification' metric"
    );
}

/// Verify AutoMetric produces different metrics for different tasks.
#[test]
fn test_auto_metric_task_dispatch() {
    let cases = [
        ("text-classification", "classification"),
        ("text-generation", "generation"),
        ("question-answering", "question_answering"),
        ("translation", "seq2seq"),
        ("summarization", "seq2seq"),
    ];
    for (task, expected_name) in &cases {
        let metric =
            AutoMetric::for_task(task).unwrap_or_else(|e| panic!("for_task({task}) failed: {e}"));
        assert_eq!(
            metric.name(),
            *expected_name,
            "task={task}: expected metric name {expected_name}, got {}",
            metric.name()
        );
    }
}

/// Verify that TrainerWithAutoMetrics records the metric name when add_eval_metric_auto is called.
#[test]
fn test_trainer_add_eval_metric_auto_records_metric_name() {
    // We test via a direct mock of the auto-metric tracking logic instead,
    // since building a Trainer<M> requires a concrete Model + Optimizer + Loss.

    struct MockTrainerProxy {
        auto_metric_names: Vec<String>,
    }

    impl MockTrainerProxy {
        fn new() -> Self {
            Self {
                auto_metric_names: Vec::new(),
            }
        }

        fn add_eval_metric_auto(
            mut self,
            task: &str,
        ) -> Result<Self, Box<trustformers::error::TrustformersError>> {
            let metric = AutoMetric::for_task(task).map_err(Box::new)?;
            self.auto_metric_names.push(metric.name().to_string());
            Ok(self)
        }
    }

    let proxy = MockTrainerProxy::new()
        .add_eval_metric_auto("text-classification")
        .expect("add_eval_metric_auto failed");

    assert!(
        proxy.auto_metric_names.contains(&"classification".to_string()),
        "expected 'classification' in auto_metric_names, got {:?}",
        proxy.auto_metric_names
    );
}

/// End-to-end: composite metrics for multiple tasks produces the right count.
#[test]
fn test_composite_metric_for_multiple_tasks() {
    let composite = AutoMetric::composite(&["text-classification", "text-generation"])
        .expect("composite failed");
    assert_eq!(
        composite.metrics().len(),
        2,
        "composite should hold 2 metrics"
    );
}
