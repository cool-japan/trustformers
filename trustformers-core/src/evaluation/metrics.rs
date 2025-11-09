// Evaluation metrics for NLP tasks
use anyhow::Result;
use std::collections::HashMap;

/// Trait for computing evaluation metrics
pub trait Metric {
    fn compute(&self, predictions: &[String], targets: &[String]) -> Result<f64>;
    fn name(&self) -> &str;
}

/// Accuracy metric
pub struct Accuracy;

impl Metric for Accuracy {
    fn compute(&self, predictions: &[String], targets: &[String]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!(
                "Predictions and targets must have the same length"
            ));
        }

        if predictions.is_empty() {
            return Ok(0.0);
        }

        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(pred, target)| pred == target)
            .count();

        Ok(correct as f64 / predictions.len() as f64)
    }

    fn name(&self) -> &str {
        "accuracy"
    }
}

/// F1 Score metric
pub struct F1Score {
    average: F1Average,
    labels: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy)]
pub enum F1Average {
    Binary,
    Macro,
    Micro,
    Weighted,
}

impl F1Score {
    pub fn new(average: F1Average) -> Self {
        Self {
            average,
            labels: None,
        }
    }

    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }

    fn compute_binary_f1(
        &self,
        predictions: &[String],
        targets: &[String],
        positive_label: &str,
    ) -> Result<f64> {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            match (pred == positive_label, target == positive_label) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_count += 1,
                (false, false) => {}, // tn
            }
        }

        let precision = if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 };
        let recall = if tp + fn_count == 0 { 0.0 } else { tp as f64 / (tp + fn_count) as f64 };

        if precision + recall == 0.0 {
            Ok(0.0)
        } else {
            Ok(2.0 * precision * recall / (precision + recall))
        }
    }

    fn get_unique_labels(&self, predictions: &[String], targets: &[String]) -> Vec<String> {
        let mut labels = std::collections::HashSet::new();

        for pred in predictions {
            labels.insert(pred.clone());
        }
        for target in targets {
            labels.insert(target.clone());
        }

        let mut sorted_labels: Vec<String> = labels.into_iter().collect();
        sorted_labels.sort();
        sorted_labels
    }

    fn compute_macro_f1(&self, predictions: &[String], targets: &[String]) -> Result<f64> {
        let labels = if let Some(ref labels) = self.labels {
            labels.clone()
        } else {
            self.get_unique_labels(predictions, targets)
        };

        let mut f1_scores = Vec::new();

        for label in &labels {
            let f1 = self.compute_binary_f1(predictions, targets, label)?;
            f1_scores.push(f1);
        }

        if f1_scores.is_empty() {
            Ok(0.0)
        } else {
            Ok(f1_scores.iter().sum::<f64>() / f1_scores.len() as f64)
        }
    }

    fn compute_micro_f1(&self, predictions: &[String], targets: &[String]) -> Result<f64> {
        let labels = if let Some(ref labels) = self.labels {
            labels.clone()
        } else {
            self.get_unique_labels(predictions, targets)
        };

        let mut total_tp = 0;
        let mut total_fp = 0;
        let mut total_fn = 0;

        for label in &labels {
            for (pred, target) in predictions.iter().zip(targets.iter()) {
                match (pred == label, target == label) {
                    (true, true) => total_tp += 1,
                    (true, false) => total_fp += 1,
                    (false, true) => total_fn += 1,
                    (false, false) => {},
                }
            }
        }

        let precision = if total_tp + total_fp == 0 {
            0.0
        } else {
            total_tp as f64 / (total_tp + total_fp) as f64
        };
        let recall = if total_tp + total_fn == 0 {
            0.0
        } else {
            total_tp as f64 / (total_tp + total_fn) as f64
        };

        if precision + recall == 0.0 {
            Ok(0.0)
        } else {
            Ok(2.0 * precision * recall / (precision + recall))
        }
    }

    fn compute_weighted_f1(&self, predictions: &[String], targets: &[String]) -> Result<f64> {
        let labels = if let Some(ref labels) = self.labels {
            labels.clone()
        } else {
            self.get_unique_labels(predictions, targets)
        };

        // Count label frequencies in targets
        let mut label_counts = HashMap::new();
        for target in targets {
            *label_counts.entry(target.clone()).or_insert(0) += 1;
        }

        let mut weighted_f1 = 0.0;
        let total_samples = targets.len() as f64;

        for label in &labels {
            let f1 = self.compute_binary_f1(predictions, targets, label)?;
            let weight = *label_counts.get(label).unwrap_or(&0) as f64 / total_samples;
            weighted_f1 += f1 * weight;
        }

        Ok(weighted_f1)
    }
}

impl Metric for F1Score {
    fn compute(&self, predictions: &[String], targets: &[String]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!(
                "Predictions and targets must have the same length"
            ));
        }

        if predictions.is_empty() {
            return Ok(0.0);
        }

        match self.average {
            F1Average::Binary => {
                // For binary classification, assume the positive label is the second unique label
                let labels = self.get_unique_labels(predictions, targets);
                if labels.len() != 2 {
                    return Err(anyhow::anyhow!(
                        "Binary F1 requires exactly 2 unique labels, found {}",
                        labels.len()
                    ));
                }
                self.compute_binary_f1(predictions, targets, &labels[1])
            },
            F1Average::Macro => self.compute_macro_f1(predictions, targets),
            F1Average::Micro => self.compute_micro_f1(predictions, targets),
            F1Average::Weighted => self.compute_weighted_f1(predictions, targets),
        }
    }

    fn name(&self) -> &str {
        match self.average {
            F1Average::Binary => "f1_binary",
            F1Average::Macro => "f1_macro",
            F1Average::Micro => "f1_micro",
            F1Average::Weighted => "f1_weighted",
        }
    }
}

/// Exact Match metric (for QA tasks)
pub struct ExactMatch;

impl Metric for ExactMatch {
    fn compute(&self, predictions: &[String], targets: &[String]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!(
                "Predictions and targets must have the same length"
            ));
        }

        if predictions.is_empty() {
            return Ok(0.0);
        }

        let exact_matches = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(pred, target)| {
                // Normalize whitespace and case for fair comparison
                let pred_normalized = pred.trim().to_lowercase();
                let target_normalized = target.trim().to_lowercase();
                pred_normalized == target_normalized
            })
            .count();

        Ok(exact_matches as f64 / predictions.len() as f64)
    }

    fn name(&self) -> &str {
        "exact_match"
    }
}

/// BLEU score metric (simplified implementation)
pub struct BLEU {
    n_grams: usize,
}

impl BLEU {
    pub fn new(n_grams: usize) -> Self {
        Self { n_grams }
    }

    fn get_ngrams<'a>(&self, tokens: &[&'a str], n: usize) -> Vec<Vec<&'a str>> {
        if tokens.len() < n {
            return vec![];
        }

        (0..=tokens.len() - n).map(|i| tokens[i..i + n].to_vec()).collect()
    }

    fn compute_bleu_score(&self, prediction: &str, reference: &str) -> f64 {
        let pred_tokens: Vec<&str> = prediction.split_whitespace().collect();
        let ref_tokens: Vec<&str> = reference.split_whitespace().collect();

        if pred_tokens.is_empty() || ref_tokens.is_empty() {
            return 0.0;
        }

        let mut precisions = Vec::new();

        for n in 1..=self.n_grams {
            let pred_ngrams = self.get_ngrams(&pred_tokens, n);
            let ref_ngrams = self.get_ngrams(&ref_tokens, n);

            if pred_ngrams.is_empty() {
                precisions.push(0.0);
                continue;
            }

            // Count matches
            let mut ref_counts = HashMap::new();
            for ngram in &ref_ngrams {
                *ref_counts.entry(ngram.clone()).or_insert(0) += 1;
            }

            let mut matches = 0;
            for ngram in &pred_ngrams {
                if let Some(count) = ref_counts.get_mut(ngram) {
                    if *count > 0 {
                        matches += 1;
                        *count -= 1;
                    }
                }
            }

            let precision = matches as f64 / pred_ngrams.len() as f64;
            precisions.push(precision);
        }

        // Geometric mean of precisions
        let log_sum: f64 = precisions.iter().map(|p| (p + 1e-10).ln()).sum();
        let geometric_mean = (log_sum / precisions.len() as f64).exp();

        // Brevity penalty
        let pred_len = pred_tokens.len() as f64;
        let ref_len = ref_tokens.len() as f64;
        let brevity_penalty =
            if pred_len > ref_len { 1.0 } else { (1.0 - ref_len / pred_len).exp() };

        geometric_mean * brevity_penalty
    }
}

impl Metric for BLEU {
    fn compute(&self, predictions: &[String], targets: &[String]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!(
                "Predictions and targets must have the same length"
            ));
        }

        if predictions.is_empty() {
            return Ok(0.0);
        }

        let scores: Vec<f64> = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| self.compute_bleu_score(pred, target))
            .collect();

        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    fn name(&self) -> &str {
        // Static string instead of dynamic formatting
        match self.n_grams {
            1 => "bleu_1",
            2 => "bleu_2",
            3 => "bleu_3",
            4 => "bleu_4",
            _ => "bleu_n",
        }
    }
}

/// Perplexity metric for language modeling
pub struct Perplexity;

impl Perplexity {
    pub fn compute_from_logits(&self, logits: &[Vec<f64>], targets: &[usize]) -> Result<f64> {
        if logits.len() != targets.len() {
            return Err(anyhow::anyhow!(
                "Logits and targets must have the same length"
            ));
        }

        if logits.is_empty() {
            return Ok(f64::INFINITY);
        }

        let mut total_log_prob = 0.0;
        let mut count = 0;

        for (logit_vec, &target_idx) in logits.iter().zip(targets.iter()) {
            if target_idx >= logit_vec.len() {
                continue; // Skip invalid targets
            }

            // Convert logits to probabilities (softmax)
            let max_logit = logit_vec.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp_logits: Vec<f64> = logit_vec.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f64 = exp_logits.iter().sum();

            let prob = exp_logits[target_idx] / sum_exp;
            if prob > 0.0 {
                total_log_prob += prob.ln();
                count += 1;
            }
        }

        if count == 0 {
            Ok(f64::INFINITY)
        } else {
            let avg_log_prob = total_log_prob / count as f64;
            Ok((-avg_log_prob).exp())
        }
    }
}

impl Metric for Perplexity {
    fn compute(&self, predictions: &[String], targets: &[String]) -> Result<f64> {
        // For string-based interface, this is a simplified version
        // In practice, perplexity needs access to actual logits
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!(
                "Predictions and targets must have the same length"
            ));
        }

        // Simplified: compute based on token-level accuracy
        let accuracy = Accuracy.compute(predictions, targets)?;

        // Rough approximation: perplexity inversely related to accuracy
        if accuracy > 0.0 {
            Ok(1.0 / accuracy)
        } else {
            Ok(f64::INFINITY)
        }
    }

    fn name(&self) -> &str {
        "perplexity"
    }
}

/// Collection of multiple metrics
pub struct MetricCollection {
    metrics: Vec<Box<dyn Metric>>,
}

impl Default for MetricCollection {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn add_accuracy(self) -> Self {
        self.add_metric(Box::new(Accuracy))
    }

    pub fn add_f1(self, average: F1Average) -> Self {
        self.add_metric(Box::new(F1Score::new(average)))
    }

    pub fn add_exact_match(self) -> Self {
        self.add_metric(Box::new(ExactMatch))
    }

    pub fn add_bleu(self, n_grams: usize) -> Self {
        self.add_metric(Box::new(BLEU::new(n_grams)))
    }

    pub fn add_perplexity(self) -> Self {
        self.add_metric(Box::new(Perplexity))
    }

    pub fn compute_all(
        &self,
        predictions: &[String],
        targets: &[String],
    ) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();

        for metric in &self.metrics {
            let score = metric.compute(predictions, targets)?;
            results.insert(metric.name().to_string(), score);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy() {
        let accuracy = Accuracy;

        let predictions = vec!["pos".to_string(), "neg".to_string(), "pos".to_string()];
        let targets = vec!["pos".to_string(), "neg".to_string(), "neg".to_string()];

        let score = accuracy.compute(&predictions, &targets).unwrap();
        assert!((score - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_f1_binary() {
        let f1 = F1Score::new(F1Average::Binary);

        let predictions = vec![
            "pos".to_string(),
            "neg".to_string(),
            "pos".to_string(),
            "neg".to_string(),
        ];
        let targets = vec![
            "pos".to_string(),
            "neg".to_string(),
            "neg".to_string(),
            "pos".to_string(),
        ];

        let score = f1.compute(&predictions, &targets).unwrap();
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_exact_match() {
        let em = ExactMatch;

        let predictions = vec!["Hello World".to_string(), "goodbye".to_string()];
        let targets = vec!["hello world".to_string(), "goodbye".to_string()];

        let score = em.compute(&predictions, &targets).unwrap();
        assert_eq!(score, 1.0); // Both should match after normalization
    }

    #[test]
    fn test_bleu() {
        let bleu = BLEU::new(4);

        let predictions = vec!["the cat sat on the mat".to_string()];
        let targets = vec!["the cat is on the mat".to_string()];

        let score = bleu.compute(&predictions, &targets).unwrap();
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_metric_collection() {
        let collection = MetricCollection::new()
            .add_accuracy()
            .add_f1(F1Average::Macro)
            .add_exact_match();

        let predictions = vec!["pos".to_string(), "neg".to_string()];
        let targets = vec!["pos".to_string(), "pos".to_string()];

        let results = collection.compute_all(&predictions, &targets).unwrap();

        assert!(results.contains_key("accuracy"));
        assert!(results.contains_key("f1_macro"));
        assert!(results.contains_key("exact_match"));
    }

    #[test]
    fn test_empty_inputs() {
        let accuracy = Accuracy;
        let predictions: Vec<String> = vec![];
        let targets: Vec<String> = vec![];

        let score = accuracy.compute(&predictions, &targets).unwrap();
        assert_eq!(score, 0.0);
    }
}
