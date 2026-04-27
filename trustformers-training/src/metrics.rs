use crate::Loss;
use std::collections::HashMap;
use trustformers_core::errors::{compute_error, Result};
use trustformers_core::Tensor;

/// Trait for evaluation metrics
pub trait Metric: Send + Sync {
    /// Compute the metric given predictions and targets
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f32>;

    /// Get the name of the metric
    fn name(&self) -> &'static str;

    /// Whether higher values indicate better performance
    fn higher_is_better(&self) -> bool;
}

/// Accuracy metric for classification tasks
#[derive(Debug, Clone)]
pub struct Accuracy;

impl Metric for Accuracy {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f32> {
        match (predictions, targets) {
            (Tensor::F32(pred_logits), Tensor::I64(target_labels)) => {
                // Get predicted classes (argmax along last dimension)
                let predicted_classes: Vec<usize> = pred_logits
                    .outer_iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .max_by(|a, b| {
                                a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(idx, _)| idx)
                            .unwrap_or(0)
                    })
                    .collect();

                // Count correct predictions
                let correct = predicted_classes
                    .iter()
                    .zip(target_labels.iter())
                    .filter(|(&pred, &target)| pred == target as usize)
                    .count();

                Ok(correct as f32 / target_labels.len() as f32)
            },
            (Tensor::I64(pred_classes), Tensor::I64(target_labels)) => {
                // Direct class comparison
                let correct = pred_classes
                    .iter()
                    .zip(target_labels.iter())
                    .filter(|(&pred, &target)| pred == target)
                    .count();

                Ok(correct as f32 / target_labels.len() as f32)
            },
            _ => Err(compute_error(
                "accuracy_computation",
                "Accuracy expects either (F32 logits, I64 targets) or (I64 classes, I64 targets)",
            )),
        }
    }

    fn name(&self) -> &'static str {
        "accuracy"
    }

    fn higher_is_better(&self) -> bool {
        true
    }
}

/// F1 Score metric for classification tasks
#[derive(Debug, Clone)]
pub struct F1Score {
    /// Average method: "binary", "macro", "micro", "weighted"
    pub average: String,
    /// For binary classification, which class is considered positive
    pub pos_label: Option<i64>,
}

impl Default for F1Score {
    fn default() -> Self {
        Self {
            average: "binary".to_string(),
            pos_label: Some(1),
        }
    }
}

impl F1Score {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn macro_averaged() -> Self {
        Self {
            average: "macro".to_string(),
            pos_label: None,
        }
    }

    pub fn micro() -> Self {
        Self {
            average: "micro".to_string(),
            pos_label: None,
        }
    }

    pub fn weighted() -> Self {
        Self {
            average: "weighted".to_string(),
            pos_label: None,
        }
    }

    /// Compute precision, recall, and F1 for a single class
    fn compute_single_class_metrics(
        &self,
        predicted_classes: &[usize],
        targets: &[i64],
        class: i64,
    ) -> (f32, f32, f32) {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;

        for (&pred, &target) in predicted_classes.iter().zip(targets.iter()) {
            match (pred as i64 == class, target == class) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_count += 1,
                (false, false) => {}, // TN - not needed for precision/recall
            }
        }

        let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };

        let recall = if tp + fn_count > 0 { tp as f32 / (tp + fn_count) as f32 } else { 0.0 };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        (precision, recall, f1)
    }
}

impl Metric for F1Score {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f32> {
        let (predicted_classes, target_labels) = match (predictions, targets) {
            (Tensor::F32(pred_logits), Tensor::I64(target_labels)) => {
                // Get predicted classes (argmax along last dimension)
                let predicted_classes: Vec<usize> = pred_logits
                    .outer_iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .max_by(|a, b| {
                                a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(idx, _)| idx)
                            .unwrap_or(0)
                    })
                    .collect();

                (
                    predicted_classes,
                    target_labels.iter().cloned().collect::<Vec<i64>>(),
                )
            },
            (Tensor::I64(pred_classes), Tensor::I64(target_labels)) => {
                let predicted_classes: Vec<usize> =
                    pred_classes.iter().map(|&x| x as usize).collect();
                (
                    predicted_classes,
                    target_labels.iter().cloned().collect::<Vec<i64>>(),
                )
            },
            _ => return Err(compute_error(
                "f1_score_computation",
                "F1Score expects either (F32 logits, I64 targets) or (I64 classes, I64 targets)",
            )),
        };

        match self.average.as_str() {
            "binary" => {
                let pos_label = self.pos_label.unwrap_or(1);
                let (_, _, f1) = self.compute_single_class_metrics(
                    &predicted_classes,
                    &target_labels,
                    pos_label,
                );
                Ok(f1)
            },
            "macro" => {
                // Get all unique classes
                let mut classes: Vec<i64> = target_labels.to_vec();
                classes.sort_unstable();
                classes.dedup();

                let f1_scores: Vec<f32> = classes
                    .iter()
                    .map(|&class| {
                        let (_, _, f1) = self.compute_single_class_metrics(
                            &predicted_classes,
                            &target_labels,
                            class,
                        );
                        f1
                    })
                    .collect();

                Ok(f1_scores.iter().sum::<f32>() / f1_scores.len() as f32)
            },
            "micro" => {
                // Calculate global TP, FP, FN
                let mut global_tp = 0;
                let mut global_fp = 0;
                let mut global_fn = 0;

                for (&pred, &target) in predicted_classes.iter().zip(target_labels.iter()) {
                    if pred as i64 == target {
                        global_tp += 1;
                    } else {
                        global_fp += 1;
                        global_fn += 1;
                    }
                }

                let precision = if global_tp + global_fp > 0 {
                    global_tp as f32 / (global_tp + global_fp) as f32
                } else {
                    0.0
                };

                let recall = if global_tp + global_fn > 0 {
                    global_tp as f32 / (global_tp + global_fn) as f32
                } else {
                    0.0
                };

                let f1 = if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                };

                Ok(f1)
            },
            "weighted" => {
                // Get class counts for weighting
                let mut class_counts: HashMap<i64, usize> = HashMap::new();
                for &target in &target_labels {
                    *class_counts.entry(target).or_insert(0) += 1;
                }

                let total_samples = target_labels.len();
                let mut weighted_f1 = 0.0;

                for (&class, &count) in &class_counts {
                    let (_, _, f1) = self.compute_single_class_metrics(
                        &predicted_classes,
                        &target_labels,
                        class,
                    );
                    let weight = count as f32 / total_samples as f32;
                    weighted_f1 += f1 * weight;
                }

                Ok(weighted_f1)
            },
            _ => Err(compute_error(
                "f1_score_computation",
                format!("Unknown average method: {}", self.average),
            )),
        }
    }

    fn name(&self) -> &'static str {
        "f1"
    }

    fn higher_is_better(&self) -> bool {
        true
    }
}

/// Perplexity metric for language modeling tasks
#[derive(Debug, Clone)]
pub struct Perplexity;

impl Metric for Perplexity {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f32> {
        match (predictions, targets) {
            (Tensor::F32(_pred_logits), Tensor::I64(_target_labels)) => {
                // Compute cross-entropy loss first
                let cross_entropy_loss = crate::CrossEntropyLoss::new();
                let loss = cross_entropy_loss.compute(predictions, targets)?;

                // Perplexity is exp(loss)
                Ok(loss.exp())
            },
            _ => Err(compute_error(
                "perplexity_computation",
                "Perplexity expects F32 logits and I64 targets",
            )),
        }
    }

    fn name(&self) -> &'static str {
        "perplexity"
    }

    fn higher_is_better(&self) -> bool {
        false // Lower perplexity is better
    }
}

/// Collection of metrics for easy management
pub struct MetricCollection {
    metrics: Vec<Box<dyn Metric>>,
}

impl MetricCollection {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    pub fn add_metric(mut self, metric: Box<dyn Metric>) -> Self {
        self.metrics.push(metric);
        self
    }

    pub fn add_metric_mut(&mut self, metric: Box<dyn Metric>) {
        self.metrics.push(metric);
    }

    pub fn compute_all(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<HashMap<String, f32>> {
        let mut results = HashMap::new();

        for metric in &self.metrics {
            let value = metric.compute(predictions, targets)?;
            results.insert(metric.name().to_string(), value);
        }

        Ok(results)
    }
}

impl Default for MetricCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{ArrayD, IxDyn};
    use trustformers_core::Tensor;

    fn make_f32_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
        let arr =
            ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape mismatch in test helper");
        Tensor::F32(arr)
    }

    fn make_i64_tensor(data: Vec<i64>, shape: &[usize]) -> Tensor {
        let arr =
            ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape mismatch in test helper");
        Tensor::I64(arr)
    }

    // ──────────────────── Accuracy ────────────────────

    #[test]
    fn test_accuracy_name() {
        assert_eq!(Accuracy.name(), "accuracy");
    }

    #[test]
    fn test_accuracy_higher_is_better() {
        assert!(Accuracy.higher_is_better());
    }

    #[test]
    fn test_accuracy_perfect_logits() {
        // logits: batch=3, classes=3; argmax selects diagonal
        let logits = make_f32_tensor(
            vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            &[3, 3],
        );
        let targets = make_i64_tensor(vec![0, 1, 2], &[3]);
        let result = Accuracy.compute(&logits, &targets);
        assert!(result.is_ok());
        let acc = result.expect("accuracy compute failed");
        assert!(
            (acc - 1.0).abs() < 1e-6,
            "perfect accuracy expected, got {}",
            acc
        );
    }

    #[test]
    fn test_accuracy_zero_logits() {
        // Predictions all wrong
        let logits = make_f32_tensor(vec![0.0, 10.0, 0.0, 0.0, 10.0, 0.0], &[2, 3]);
        // argmax = class 1 for both; targets want class 0 and 2
        let targets = make_i64_tensor(vec![0, 2], &[2]);
        let result = Accuracy.compute(&logits, &targets);
        assert!(result.is_ok());
        let acc = result.expect("accuracy compute failed");
        assert!(
            (acc - 0.0).abs() < 1e-6,
            "zero accuracy expected, got {}",
            acc
        );
    }

    #[test]
    fn test_accuracy_half_correct_logits() {
        // Row 0: [10.0, 0.0] → argmax = class 0
        // Row 1: [10.0, 0.0] → argmax = class 0
        let logits = make_f32_tensor(vec![10.0, 0.0, 10.0, 0.0], &[2, 2]);
        // First row correct (target 0), second row wrong (target 1)
        let targets = make_i64_tensor(vec![0, 1], &[2]);
        let result = Accuracy.compute(&logits, &targets);
        assert!(result.is_ok());
        let acc = result.expect("accuracy compute failed");
        assert!(
            (acc - 0.5).abs() < 1e-6,
            "50% accuracy expected, got {}",
            acc
        );
    }

    #[test]
    fn test_accuracy_with_i64_predictions() {
        let pred_classes = make_i64_tensor(vec![0, 1, 2, 1], &[4]);
        let targets = make_i64_tensor(vec![0, 1, 2, 0], &[4]);
        let result = Accuracy.compute(&pred_classes, &targets);
        assert!(result.is_ok());
        let acc = result.expect("accuracy compute failed");
        // 3 out of 4 correct
        assert!((acc - 0.75).abs() < 1e-6, "75% expected, got {}", acc);
    }

    #[test]
    fn test_accuracy_invalid_types_returns_err() {
        let pred = make_f32_tensor(vec![1.0, 0.0], &[2]);
        let target = make_f32_tensor(vec![0.0, 1.0], &[2]);
        let result = Accuracy.compute(&pred, &target);
        assert!(result.is_err(), "F32 targets should return error");
    }

    // ──────────────────── F1Score ────────────────────

    #[test]
    fn test_f1_score_name() {
        assert_eq!(F1Score::new().name(), "f1");
    }

    #[test]
    fn test_f1_score_higher_is_better() {
        assert!(F1Score::new().higher_is_better());
    }

    #[test]
    fn test_f1_binary_perfect() {
        let pred = make_i64_tensor(vec![1, 1, 0, 0], &[4]);
        let target = make_i64_tensor(vec![1, 1, 0, 0], &[4]);
        let result = F1Score::new().compute(&pred, &target);
        assert!(result.is_ok());
        let f1 = result.expect("f1 compute failed");
        assert!((f1 - 1.0).abs() < 1e-5, "perfect F1 expected, got {}", f1);
    }

    #[test]
    fn test_f1_macro_multiclass() {
        // 3-class: perfect classification
        let pred = make_i64_tensor(vec![0, 1, 2, 0, 1, 2], &[6]);
        let target = make_i64_tensor(vec![0, 1, 2, 0, 1, 2], &[6]);
        let result = F1Score::macro_averaged().compute(&pred, &target);
        assert!(result.is_ok());
        let f1 = result.expect("f1 macro compute failed");
        assert!(
            (f1 - 1.0).abs() < 1e-5,
            "perfect macro F1 expected, got {}",
            f1
        );
    }

    #[test]
    fn test_f1_micro() {
        let pred = make_i64_tensor(vec![0, 1, 1, 0], &[4]);
        let target = make_i64_tensor(vec![0, 1, 0, 1], &[4]);
        let result = F1Score::micro().compute(&pred, &target);
        assert!(result.is_ok());
        let f1 = result.expect("f1 micro compute failed");
        assert!(
            (0.0..=1.0).contains(&f1),
            "micro F1 should be in [0,1], got {}",
            f1
        );
    }

    #[test]
    fn test_f1_weighted() {
        let pred = make_i64_tensor(vec![0, 1, 2, 0, 1, 2], &[6]);
        let target = make_i64_tensor(vec![0, 1, 2, 0, 1, 2], &[6]);
        let result = F1Score::weighted().compute(&pred, &target);
        assert!(result.is_ok());
        let f1 = result.expect("f1 weighted compute failed");
        assert!(
            (f1 - 1.0).abs() < 1e-5,
            "perfect weighted F1 expected, got {}",
            f1
        );
    }

    #[test]
    fn test_f1_binary_from_logits() {
        // logits selecting class 0 vs class 1
        let logits = make_f32_tensor(vec![10.0, 0.0, 0.0, 10.0], &[2, 2]);
        let target = make_i64_tensor(vec![0, 1], &[2]);
        let result = F1Score::new().compute(&logits, &target);
        assert!(result.is_ok());
        let f1 = result.expect("f1 from logits failed");
        assert!(
            (f1 - 1.0).abs() < 1e-5,
            "perfect F1 from logits, got {}",
            f1
        );
    }

    #[test]
    fn test_f1_invalid_average_returns_err() {
        let pred = make_i64_tensor(vec![0, 1], &[2]);
        let target = make_i64_tensor(vec![0, 1], &[2]);
        let f1 = F1Score {
            average: "invalid_avg".to_string(),
            pos_label: None,
        };
        let result = f1.compute(&pred, &target);
        assert!(
            result.is_err(),
            "invalid average method should return error"
        );
    }

    // ──────────────────── Perplexity ────────────────────

    #[test]
    fn test_perplexity_name() {
        assert_eq!(Perplexity.name(), "perplexity");
    }

    #[test]
    fn test_perplexity_lower_is_better() {
        assert!(!Perplexity.higher_is_better());
    }

    #[test]
    fn test_perplexity_equals_exp_cross_entropy() {
        let logits = make_f32_tensor(vec![2.0, 1.0, 0.1, 0.1, 2.0, 0.1], &[2, 3]);
        let targets = make_i64_tensor(vec![0, 1], &[2]);
        let ce_loss = crate::CrossEntropyLoss::new();
        let ce_val = ce_loss.compute(&logits, &targets).expect("ce failed");
        let ppl = Perplexity.compute(&logits, &targets).expect("perplexity failed");
        let expected_ppl = ce_val.exp();
        assert!(
            (ppl - expected_ppl).abs() < 1e-4,
            "perplexity should equal exp(CE), expected {:.4}, got {:.4}",
            expected_ppl,
            ppl
        );
    }

    #[test]
    fn test_perplexity_wrong_types_returns_err() {
        let pred = make_f32_tensor(vec![1.0, 0.0], &[2]);
        let target = make_f32_tensor(vec![0.0, 1.0], &[2]);
        let result = Perplexity.compute(&pred, &target);
        assert!(result.is_err(), "F32 targets should fail for perplexity");
    }

    // ──────────────────── MetricCollection ────────────────────

    #[test]
    fn test_metric_collection_empty() {
        let collection = MetricCollection::new();
        let logits = make_f32_tensor(vec![1.0, 0.0], &[1, 2]);
        let targets = make_i64_tensor(vec![0], &[1]);
        let result = collection.compute_all(&logits, &targets);
        assert!(result.is_ok());
        let map = result.expect("empty collection compute failed");
        assert!(map.is_empty(), "empty collection should return empty map");
    }

    #[test]
    fn test_metric_collection_multiple_metrics() {
        let collection = MetricCollection::new()
            .add_metric(Box::new(Accuracy))
            .add_metric(Box::new(F1Score::macro_averaged()));
        let logits = make_f32_tensor(vec![10.0, 0.0, 0.0, 10.0], &[2, 2]);
        let targets = make_i64_tensor(vec![0, 1], &[2]);
        let result = collection.compute_all(&logits, &targets);
        assert!(result.is_ok());
        let map = result.expect("multi-metric compute failed");
        assert!(map.contains_key("accuracy"), "should contain 'accuracy'");
        assert!(map.contains_key("f1"), "should contain 'f1'");
    }

    #[test]
    fn test_metric_collection_add_metric_mut() {
        let mut collection = MetricCollection::new();
        collection.add_metric_mut(Box::new(Accuracy));
        let logits = make_f32_tensor(vec![5.0, 1.0, 1.0, 5.0], &[2, 2]);
        let targets = make_i64_tensor(vec![0, 1], &[2]);
        let result = collection.compute_all(&logits, &targets);
        assert!(result.is_ok());
        let map = result.expect("add_metric_mut compute failed");
        assert!(map.contains_key("accuracy"));
    }
}
