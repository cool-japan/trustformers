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
