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
    /// Perplexity cannot be computed from decoded strings.
    ///
    /// It is `exp(-mean log p(target))`, which needs the model's per-token
    /// log-probabilities. The string-based [`Metric`] interface carries none,
    /// so this reports the mismatch instead of returning `1 / accuracy` under
    /// the name "perplexity". Use [`Perplexity::compute_from_logits`].
    fn compute(&self, predictions: &[String], targets: &[String]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!(
                "Predictions and targets must have the same length"
            ));
        }

        Err(anyhow::anyhow!(
            "perplexity cannot be derived from decoded strings: it needs per-token \
             log-probabilities. Call Perplexity::compute_from_logits(logits, targets) instead of \
             adding Perplexity to a string-based MetricCollection."
        ))
    }

    fn name(&self) -> &str {
        "perplexity"
    }
}

/// Pearson product-moment correlation between numeric predictions and targets.
///
/// Both sides are parsed as `f64`; pairs where either side does not parse are
/// skipped. Used by regression benchmarks such as GLUE STS-B.
pub struct PearsonCorrelation;

/// Spearman rank correlation between numeric predictions and targets.
///
/// Ranks are averaged over ties, then Pearson correlation is taken over the
/// ranks. Used by regression benchmarks such as GLUE STS-B.
pub struct SpearmanCorrelation;

/// Parse the numeric pairs a correlation metric can use.
fn numeric_pairs(predictions: &[String], targets: &[String]) -> Result<(Vec<f64>, Vec<f64>)> {
    if predictions.len() != targets.len() {
        return Err(anyhow::anyhow!(
            "Predictions and targets must have the same length"
        ));
    }

    let mut xs = Vec::with_capacity(predictions.len());
    let mut ys = Vec::with_capacity(targets.len());
    for (prediction, target) in predictions.iter().zip(targets.iter()) {
        if let (Ok(x), Ok(y)) = (
            prediction.trim().parse::<f64>(),
            target.trim().parse::<f64>(),
        ) {
            xs.push(x);
            ys.push(y);
        }
    }

    if xs.len() < 2 {
        return Err(anyhow::anyhow!(
            "correlation needs at least two numeric prediction/target pairs; got {}",
            xs.len()
        ));
    }

    Ok((xs, ys))
}

/// Pearson correlation of two equal-length samples.
fn pearson(xs: &[f64], ys: &[f64]) -> Result<f64> {
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;

    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }

    let denominator = (variance_x * variance_y).sqrt();
    if denominator <= 0.0 {
        return Err(anyhow::anyhow!(
            "correlation is undefined when a sample has zero variance"
        ));
    }
    Ok(covariance / denominator)
}

/// Fractional ranks with ties averaged.
fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].partial_cmp(&values[*b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; values.len()];
    let mut index = 0;
    while index < order.len() {
        let mut end = index + 1;
        while end < order.len() && values[order[end]] == values[order[index]] {
            end += 1;
        }
        // Ranks are 1-based; average over the tied block.
        let average = ((index + 1) as f64 + end as f64) / 2.0;
        for slot in &order[index..end] {
            ranks[*slot] = average;
        }
        index = end;
    }
    ranks
}

impl Metric for PearsonCorrelation {
    fn compute(&self, predictions: &[String], targets: &[String]) -> Result<f64> {
        let (xs, ys) = numeric_pairs(predictions, targets)?;
        pearson(&xs, &ys)
    }

    fn name(&self) -> &str {
        "pearson"
    }
}

impl Metric for SpearmanCorrelation {
    fn compute(&self, predictions: &[String], targets: &[String]) -> Result<f64> {
        let (xs, ys) = numeric_pairs(predictions, targets)?;
        pearson(&average_ranks(&xs), &average_ranks(&ys))
    }

    fn name(&self) -> &str {
        "spearman"
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

    /// Add perplexity to a string-based collection.
    ///
    /// Perplexity needs logits, which this interface does not carry, so the
    /// resulting collection fails at `compute_all` time rather than reporting
    /// something else under the name "perplexity". Prefer
    /// [`Perplexity::compute_from_logits`].
    #[deprecated(
        since = "0.2.1",
        note = "perplexity needs logits; use Perplexity::compute_from_logits"
    )]
    pub fn add_perplexity(self) -> Self {
        self.add_metric(Box::new(Perplexity))
    }

    /// Add Pearson correlation (for regression tasks such as GLUE STS-B).
    pub fn add_pearson(self) -> Self {
        self.add_metric(Box::new(PearsonCorrelation))
    }

    /// Add Spearman rank correlation (for regression tasks such as GLUE STS-B).
    pub fn add_spearman(self) -> Self {
        self.add_metric(Box::new(SpearmanCorrelation))
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

    /// Regression test: `Perplexity`'s string-based `Metric::compute` used to
    /// return `1 / accuracy` under the name "perplexity".
    #[test]
    fn test_perplexity_refuses_the_string_interface() {
        let predictions = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let targets = vec!["a".to_string(), "b".to_string(), "d".to_string()];

        // Accuracy here is 2/3, so the old code returned 1.5.
        let error = Perplexity
            .compute(&predictions, &targets)
            .expect_err("perplexity cannot come from decoded strings");
        assert!(
            error.to_string().contains("compute_from_logits"),
            "unexpected error: {error}"
        );

        // The logits-based path is the real one and still works: a model that
        // is certain and right has perplexity ~1.
        let logits = vec![vec![10.0, 0.0], vec![10.0, 0.0]];
        let perplexity =
            Perplexity.compute_from_logits(&logits, &[0, 0]).expect("logits perplexity");
        assert!(
            (perplexity - 1.0).abs() < 1e-3,
            "confident correct predictions should give perplexity ~1, got {perplexity}"
        );

        // A uniform two-way distribution has perplexity 2.
        let uniform = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let perplexity =
            Perplexity.compute_from_logits(&uniform, &[0, 1]).expect("logits perplexity");
        assert!(
            (perplexity - 2.0).abs() < 1e-6,
            "a uniform binary distribution has perplexity 2, got {perplexity}"
        );
    }

    /// Regression test: STS-B used to be scored with accuracy because no
    /// correlation metric existed.
    #[test]
    fn test_correlation_metrics_known_values() {
        let strings = |values: &[f64]| -> Vec<String> {
            values.iter().map(|value| value.to_string()).collect()
        };

        // Perfect positive linear relationship.
        let x = strings(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = strings(&[2.0, 4.0, 6.0, 8.0, 10.0]);
        let pearson = PearsonCorrelation.compute(&x, &y).expect("pearson");
        assert!((pearson - 1.0).abs() < 1e-9, "got {pearson}");
        let spearman = SpearmanCorrelation.compute(&x, &y).expect("spearman");
        assert!((spearman - 1.0).abs() < 1e-9, "got {spearman}");

        // Perfect anti-correlation.
        let reversed = strings(&[10.0, 8.0, 6.0, 4.0, 2.0]);
        let pearson = PearsonCorrelation.compute(&x, &reversed).expect("pearson");
        assert!((pearson + 1.0).abs() < 1e-9, "got {pearson}");
        let spearman = SpearmanCorrelation.compute(&x, &reversed).expect("spearman");
        assert!((spearman + 1.0).abs() < 1e-9, "got {spearman}");

        // Monotone but non-linear: Spearman is 1.0 while Pearson is not, which
        // is exactly why both are reported for STS-B.
        let exponential = strings(&[1.0, 2.0, 8.0, 64.0, 1024.0]);
        let spearman = SpearmanCorrelation.compute(&x, &exponential).expect("spearman");
        assert!((spearman - 1.0).abs() < 1e-9, "got {spearman}");
        let pearson = PearsonCorrelation.compute(&x, &exponential).expect("pearson");
        assert!(
            pearson < 0.95,
            "Pearson must not equal Spearman here: {pearson}"
        );

        assert_eq!(PearsonCorrelation.name(), "pearson");
        assert_eq!(SpearmanCorrelation.name(), "spearman");
    }

    /// Tied ranks are averaged, so a tied sample still yields a finite
    /// Spearman value rather than a rank-ordering artefact.
    #[test]
    fn test_spearman_averages_tied_ranks() {
        let predictions: Vec<String> =
            ["1", "2", "2", "3"].iter().map(|value| value.to_string()).collect();
        let targets: Vec<String> =
            ["1", "2", "2", "3"].iter().map(|value| value.to_string()).collect();
        let spearman = SpearmanCorrelation.compute(&predictions, &targets).expect("spearman");
        assert!(
            (spearman - 1.0).abs() < 1e-9,
            "identical tied samples must give 1.0"
        );
    }

    /// Correlation is undefined for a constant sample and for fewer than two
    /// numeric pairs; both must be errors, not zeros.
    #[test]
    fn test_correlation_rejects_degenerate_input() {
        let constant: Vec<String> = vec!["1".to_string(); 4];
        let varied: Vec<String> =
            ["1", "2", "3", "4"].iter().map(|value| value.to_string()).collect();
        assert!(PearsonCorrelation.compute(&constant, &varied).is_err());

        let single = vec!["1".to_string()];
        assert!(PearsonCorrelation.compute(&single, &single).is_err());

        let non_numeric = vec!["a".to_string(), "b".to_string()];
        assert!(SpearmanCorrelation.compute(&non_numeric, &non_numeric).is_err());
    }

    #[test]
    fn test_accuracy() {
        let accuracy = Accuracy;

        let predictions = vec!["pos".to_string(), "neg".to_string(), "pos".to_string()];
        let targets = vec!["pos".to_string(), "neg".to_string(), "neg".to_string()];

        let score = accuracy.compute(&predictions, &targets).expect("operation failed in test");
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

        let score = f1.compute(&predictions, &targets).expect("operation failed in test");
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_exact_match() {
        let em = ExactMatch;

        let predictions = vec!["Hello World".to_string(), "goodbye".to_string()];
        let targets = vec!["hello world".to_string(), "goodbye".to_string()];

        let score = em.compute(&predictions, &targets).expect("operation failed in test");
        assert_eq!(score, 1.0); // Both should match after normalization
    }

    #[test]
    fn test_bleu() {
        let bleu = BLEU::new(4);

        let predictions = vec!["the cat sat on the mat".to_string()];
        let targets = vec!["the cat is on the mat".to_string()];

        let score = bleu.compute(&predictions, &targets).expect("operation failed in test");
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

        let results = collection
            .compute_all(&predictions, &targets)
            .expect("operation failed in test");

        assert!(results.contains_key("accuracy"));
        assert!(results.contains_key("f1_macro"));
        assert!(results.contains_key("exact_match"));
    }

    #[test]
    fn test_empty_inputs() {
        let accuracy = Accuracy;
        let predictions: Vec<String> = vec![];
        let targets: Vec<String> = vec![];

        let score = accuracy.compute(&predictions, &targets).expect("operation failed in test");
        assert_eq!(score, 0.0);
    }
}
