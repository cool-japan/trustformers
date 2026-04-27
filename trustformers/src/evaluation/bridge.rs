// Adapters that map the four overlapping metric stacks onto the canonical
// auto::metrics::Metric trait so callers use one API regardless of entry point.
//
// CoreAdapter: deferred — trustformers-core::Metric is defined in trustformers-core and the
// bridge lives in trustformers which already depends on core. Creating a bridge from core's
// evaluation harness metrics back up to auto::metrics::Metric would require trustformers-core
// to know about auto::metrics::Metric, creating a circular dependency. Deferred.

use crate::{
    auto::metrics::{Metric, MetricInput, MetricResult},
    error::{Result, TrustformersError},
    evaluation::metrics::{
        bleu_score, exact_match_score, perplexity, rouge_l_score, rouge_n_score, token_f1_score,
    },
};
use std::collections::HashMap;
use trustformers_core::Tensor;
use trustformers_training::metrics::{
    Accuracy as TrainingAccuracy, F1Score as TrainingF1Score, Metric as TrainingMetric,
    Perplexity as TrainingPerplexity,
};

// ─────────────────────────────────────────────────────────────────────────────
// NlpAdapter — wraps the string-based NLP functions from evaluation::metrics
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NlpMetricKind {
    Bleu { max_n: usize, smooth: bool },
    RougeN { n: usize },
    RougeL,
    TokenF1,
    ExactMatch,
    Perplexity,
}

/// Adapter that lifts the string-based NLP metric functions from
/// `evaluation::metrics` into the `auto::metrics::Metric` trait.
///
/// Text pairs are buffered via `add_batch` and the underlying function is
/// called once per pair at `compute` time; results are averaged.
#[derive(Debug)]
pub struct NlpAdapter {
    kind: NlpMetricKind,
    predictions: Vec<String>,
    references: Vec<String>,
}

impl NlpAdapter {
    pub(crate) fn new(kind: NlpMetricKind) -> Self {
        Self {
            kind,
            predictions: Vec::new(),
            references: Vec::new(),
        }
    }

    /// Construct a concrete NlpAdapter for BLEU (for use in integration tests).
    pub fn new_bleu(max_n: usize, smooth: bool) -> Self {
        Self::new(NlpMetricKind::Bleu { max_n, smooth })
    }

    /// Construct a concrete NlpAdapter for ROUGE-N.
    pub fn new_rouge_n(n: usize) -> Self {
        Self::new(NlpMetricKind::RougeN { n })
    }

    /// Construct a concrete NlpAdapter for ROUGE-L.
    pub fn new_rouge_l() -> Self {
        Self::new(NlpMetricKind::RougeL)
    }

    /// Construct a concrete NlpAdapter for token F1.
    pub fn new_token_f1() -> Self {
        Self::new(NlpMetricKind::TokenF1)
    }

    /// Construct a concrete NlpAdapter for exact match.
    pub fn new_exact_match() -> Self {
        Self::new(NlpMetricKind::ExactMatch)
    }

    /// Construct a concrete NlpAdapter for perplexity.
    pub fn new_perplexity() -> Self {
        Self::new(NlpMetricKind::Perplexity)
    }

    /// Sentence-level BLEU averaged over all buffered pairs.
    pub fn bleu(max_n: usize, smooth: bool) -> Box<dyn Metric> {
        Box::new(Self::new(NlpMetricKind::Bleu { max_n, smooth }))
    }

    /// ROUGE-N (n-gram) F1 averaged over all buffered pairs.
    pub fn rouge_n(n: usize) -> Box<dyn Metric> {
        Box::new(Self::new(NlpMetricKind::RougeN { n }))
    }

    /// ROUGE-L (LCS-based) F1 averaged over all buffered pairs.
    pub fn rouge_l() -> Box<dyn Metric> {
        Box::new(Self::new(NlpMetricKind::RougeL))
    }

    /// Token-level F1 averaged over all buffered pairs.
    pub fn token_f1() -> Box<dyn Metric> {
        Box::new(Self::new(NlpMetricKind::TokenF1))
    }

    /// Exact-match accuracy over all buffered pairs.
    pub fn exact_match() -> Box<dyn Metric> {
        Box::new(Self::new(NlpMetricKind::ExactMatch))
    }

    /// Perplexity from whitespace-delimited log-probability strings.
    ///
    /// Each "prediction" string is treated as space-separated f64 log-probabilities;
    /// references are ignored. The log-probs from all pairs are concatenated before
    /// computing corpus-level perplexity.
    pub fn perplexity() -> Box<dyn Metric> {
        Box::new(Self::new(NlpMetricKind::Perplexity))
    }
}

impl Metric for NlpAdapter {
    fn add_batch(&mut self, predictions: &MetricInput, references: &MetricInput) -> Result<()> {
        match (predictions, references) {
            (MetricInput::Text(pred), MetricInput::Text(refs)) => {
                if pred.len() != refs.len() {
                    return Err(TrustformersError::invalid_input_simple(format!(
                        "NlpAdapter::add_batch: predictions length {} != references length {}",
                        pred.len(),
                        refs.len()
                    )));
                }
                self.predictions.extend(pred.iter().cloned());
                self.references.extend(refs.iter().cloned());
                Ok(())
            },
            _ => Err(TrustformersError::invalid_input_simple(
                "NlpAdapter expects MetricInput::Text for both predictions and references"
                    .to_string(),
            )),
        }
    }

    fn compute(&self) -> Result<MetricResult> {
        if self.predictions.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "NlpAdapter: no data has been added".to_string(),
            ));
        }

        let (value, mut details) = match self.kind {
            NlpMetricKind::Bleu { max_n, smooth } => {
                let scores: Vec<f64> = self
                    .predictions
                    .iter()
                    .zip(self.references.iter())
                    .map(|(pred, ref_)| {
                        let pred_tokens: Vec<&str> = pred.split_whitespace().collect();
                        let ref_tokens: Vec<&str> = ref_.split_whitespace().collect();
                        let refs_slice: &[&[&str]] = &[ref_tokens.as_slice()];
                        bleu_score(&pred_tokens, refs_slice, max_n, smooth).map_err(|e| {
                            TrustformersError::invalid_input_simple(format!(
                                "BLEU computation failed: {e}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                let mut d = HashMap::new();
                d.insert("bleu".to_string(), avg);
                (avg, d)
            },

            NlpMetricKind::RougeN { n } => {
                let scores: Vec<f64> = self
                    .predictions
                    .iter()
                    .zip(self.references.iter())
                    .map(|(pred, ref_)| {
                        let pred_tokens: Vec<&str> = pred.split_whitespace().collect();
                        let ref_tokens: Vec<&str> = ref_.split_whitespace().collect();
                        rouge_n_score(&pred_tokens, &ref_tokens, n).map(|s| s.f1).map_err(|e| {
                            TrustformersError::invalid_input_simple(format!(
                                "ROUGE-N computation failed: {e}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                let mut d = HashMap::new();
                d.insert(format!("rouge_{n}"), avg);
                (avg, d)
            },

            NlpMetricKind::RougeL => {
                let scores: Vec<f64> = self
                    .predictions
                    .iter()
                    .zip(self.references.iter())
                    .map(|(pred, ref_)| {
                        let pred_tokens: Vec<&str> = pred.split_whitespace().collect();
                        let ref_tokens: Vec<&str> = ref_.split_whitespace().collect();
                        rouge_l_score(&pred_tokens, &ref_tokens).map(|s| s.f1).map_err(|e| {
                            TrustformersError::invalid_input_simple(format!(
                                "ROUGE-L computation failed: {e}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                let mut d = HashMap::new();
                d.insert("rouge_l".to_string(), avg);
                (avg, d)
            },

            NlpMetricKind::TokenF1 => {
                let scores: Vec<f64> = self
                    .predictions
                    .iter()
                    .zip(self.references.iter())
                    .map(|(pred, ref_)| {
                        let pred_tokens: Vec<&str> = pred.split_whitespace().collect();
                        let ref_tokens: Vec<&str> = ref_.split_whitespace().collect();
                        token_f1_score(&pred_tokens, &ref_tokens).map(|s| s.f1).map_err(|e| {
                            TrustformersError::invalid_input_simple(format!(
                                "token-F1 computation failed: {e}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                let mut d = HashMap::new();
                d.insert("token_f1".to_string(), avg);
                (avg, d)
            },

            NlpMetricKind::ExactMatch => {
                let pred_refs: Vec<(&str, &str)> = self
                    .predictions
                    .iter()
                    .zip(self.references.iter())
                    .map(|(p, r)| (p.as_str(), r.as_str()))
                    .collect();
                let preds: Vec<&str> = pred_refs.iter().map(|(p, _)| *p).collect();
                let refs: Vec<&str> = pred_refs.iter().map(|(_, r)| *r).collect();
                let score = exact_match_score(&preds, &refs).map_err(|e| {
                    TrustformersError::invalid_input_simple(format!(
                        "exact-match computation failed: {e}"
                    ))
                })?;
                let mut d = HashMap::new();
                d.insert("exact_match".to_string(), score);
                (score, d)
            },

            NlpMetricKind::Perplexity => {
                let log_probs: Vec<f64> = self
                    .predictions
                    .iter()
                    .flat_map(|s| s.split_whitespace().filter_map(|tok| tok.parse::<f64>().ok()))
                    .collect();
                if log_probs.is_empty() {
                    return Err(TrustformersError::invalid_input_simple(
                        "NlpAdapter perplexity: no parseable log-probabilities found in predictions"
                            .to_string(),
                    ));
                }
                let ppl = perplexity(&log_probs).map_err(|e| {
                    TrustformersError::invalid_input_simple(format!(
                        "perplexity computation failed: {e}"
                    ))
                })?;
                let mut d = HashMap::new();
                d.insert("perplexity".to_string(), ppl);
                (ppl, d)
            },
        };

        let metric_name = self.name().to_string();
        details.insert(metric_name.clone(), value);

        Ok(MetricResult {
            name: metric_name,
            value,
            details,
            metadata: HashMap::new(),
        })
    }

    fn reset(&mut self) {
        self.predictions.clear();
        self.references.clear();
    }

    fn name(&self) -> &str {
        match self.kind {
            NlpMetricKind::Bleu { .. } => "nlp_bleu",
            NlpMetricKind::RougeN { .. } => "nlp_rouge_n",
            NlpMetricKind::RougeL => "nlp_rouge_l",
            NlpMetricKind::TokenF1 => "nlp_token_f1",
            NlpMetricKind::ExactMatch => "nlp_exact_match",
            NlpMetricKind::Perplexity => "nlp_perplexity",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorAdapter — wraps trustformers_training::metrics::{Accuracy,F1Score,Perplexity}
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorMetricKind {
    Accuracy,
    F1,
    Perplexity,
}

/// Adapter that lifts tensor-native training metrics (Accuracy, F1Score, Perplexity)
/// into the `auto::metrics::Metric` trait.
///
/// Accepts `MetricInput::Tensors { predictions, targets }` for add_batch.
/// Results are averaged over all buffered pairs.
#[derive(Debug)]
pub struct TensorAdapter {
    kind: TensorMetricKind,
    average: String,
    results: Vec<f32>,
    pair_count: usize,
}

impl TensorAdapter {
    fn new(kind: TensorMetricKind, average: impl Into<String>) -> Self {
        Self {
            kind,
            average: average.into(),
            results: Vec::new(),
            pair_count: 0,
        }
    }

    /// Concrete (non-boxed) accuracy adapter for use in integration tests.
    pub fn new_accuracy() -> Self {
        Self::new(TensorMetricKind::Accuracy, "")
    }

    /// Concrete (non-boxed) F1 adapter.
    pub fn new_f1(average: impl Into<String>) -> Self {
        Self::new(TensorMetricKind::F1, average)
    }

    /// Concrete (non-boxed) perplexity adapter.
    pub fn new_perplexity() -> Self {
        Self::new(TensorMetricKind::Perplexity, "")
    }

    /// Accuracy over buffered (predictions, targets) tensor pairs.
    pub fn accuracy() -> Box<dyn Metric> {
        Box::new(Self::new(TensorMetricKind::Accuracy, ""))
    }

    /// F1Score over buffered (predictions, targets) tensor pairs.
    pub fn f1(average: String) -> Box<dyn Metric> {
        Box::new(Self::new(TensorMetricKind::F1, average))
    }

    /// Perplexity over buffered (predictions, targets) tensor pairs.
    pub fn perplexity() -> Box<dyn Metric> {
        Box::new(Self::new(TensorMetricKind::Perplexity, ""))
    }

    fn compute_single(&self, predictions: &Tensor, targets: &Tensor) -> Result<f32> {
        let core_result = match self.kind {
            TensorMetricKind::Accuracy => TrainingAccuracy.compute(predictions, targets),
            TensorMetricKind::F1 => {
                let scorer = TrainingF1Score {
                    average: self.average.clone(),
                    pos_label: if self.average == "binary" { Some(1) } else { None },
                };
                scorer.compute(predictions, targets)
            },
            TensorMetricKind::Perplexity => TrainingPerplexity.compute(predictions, targets),
        };
        core_result.map_err(|e| {
            TrustformersError::invalid_input_simple(format!("TensorAdapter compute error: {e}"))
        })
    }
}

impl Metric for TensorAdapter {
    fn add_batch(&mut self, predictions: &MetricInput, _references: &MetricInput) -> Result<()> {
        match predictions {
            MetricInput::Tensors {
                predictions: pred,
                targets: tgt,
            } => {
                let val = self.compute_single(pred, tgt)?;
                self.results.push(val);
                self.pair_count += 1;
                Ok(())
            },
            _ => Err(TrustformersError::invalid_input_simple(
                "TensorAdapter expects MetricInput::Tensors for predictions".to_string(),
            )),
        }
    }

    fn compute(&self) -> Result<MetricResult> {
        if self.results.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "TensorAdapter: no data has been added".to_string(),
            ));
        }
        let avg = self.results.iter().sum::<f32>() / self.results.len() as f32;
        let value = avg as f64;
        let mut details = HashMap::new();
        details.insert(self.name().to_string(), value);
        Ok(MetricResult {
            name: self.name().to_string(),
            value,
            details,
            metadata: HashMap::new(),
        })
    }

    fn reset(&mut self) {
        self.results.clear();
        self.pair_count = 0;
    }

    fn name(&self) -> &str {
        match self.kind {
            TensorMetricKind::Accuracy => "tensor_accuracy",
            TensorMetricKind::F1 => "tensor_f1",
            TensorMetricKind::Perplexity => "tensor_perplexity",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nlp_bleu() -> NlpAdapter {
        NlpAdapter::new(NlpMetricKind::Bleu {
            max_n: 4,
            smooth: true,
        })
    }
    fn make_nlp_rouge_l() -> NlpAdapter {
        NlpAdapter::new(NlpMetricKind::RougeL)
    }
    fn make_nlp_token_f1() -> NlpAdapter {
        NlpAdapter::new(NlpMetricKind::TokenF1)
    }
    fn make_nlp_exact_match() -> NlpAdapter {
        NlpAdapter::new(NlpMetricKind::ExactMatch)
    }

    #[test]
    fn test_nlp_adapter_bleu_perfect_match() {
        let mut adapter = make_nlp_bleu();
        let preds = MetricInput::Text(vec!["the cat sat on the mat".to_string()]);
        let refs = MetricInput::Text(vec!["the cat sat on the mat".to_string()]);
        adapter.add_batch(&preds, &refs).expect("add_batch failed");
        let result = adapter.compute().expect("compute failed");
        assert!(
            (result.value - 1.0).abs() < 1e-6,
            "perfect BLEU should be 1.0, got {}",
            result.value
        );
    }

    #[test]
    fn test_nlp_adapter_exact_match_basic() {
        let mut adapter = make_nlp_exact_match();
        let preds = MetricInput::Text(vec!["Paris".to_string(), "Tokyo".to_string()]);
        let refs = MetricInput::Text(vec!["paris".to_string(), "London".to_string()]);
        adapter.add_batch(&preds, &refs).expect("add_batch failed");
        let result = adapter.compute().expect("compute failed");
        assert!(
            (result.value - 0.5).abs() < 1e-6,
            "expected 0.5, got {}",
            result.value
        );
    }

    #[test]
    fn test_nlp_adapter_rouge_l_perfect() {
        let mut adapter = make_nlp_rouge_l();
        let preds = MetricInput::Text(vec!["a b c d".to_string()]);
        let refs = MetricInput::Text(vec!["a b c d".to_string()]);
        adapter.add_batch(&preds, &refs).expect("add_batch failed");
        let result = adapter.compute().expect("compute failed");
        assert!(
            (result.value - 1.0).abs() < 1e-6,
            "perfect ROUGE-L should be 1.0, got {}",
            result.value
        );
    }

    #[test]
    fn test_nlp_adapter_token_f1_no_overlap() {
        let mut adapter = make_nlp_token_f1();
        let preds = MetricInput::Text(vec!["alpha beta".to_string()]);
        let refs = MetricInput::Text(vec!["gamma delta".to_string()]);
        adapter.add_batch(&preds, &refs).expect("add_batch failed");
        let result = adapter.compute().expect("compute failed");
        assert_eq!(result.value, 0.0, "no-overlap token F1 should be 0");
    }

    #[test]
    fn test_nlp_adapter_reset_clears_state() {
        let mut adapter = make_nlp_exact_match();
        let preds = MetricInput::Text(vec!["hello".to_string()]);
        let refs = MetricInput::Text(vec!["hello".to_string()]);
        adapter.add_batch(&preds, &refs).expect("add_batch failed");
        adapter.reset();
        assert!(adapter.compute().is_err(), "should fail after reset");
    }

    #[test]
    fn test_nlp_adapter_length_mismatch_returns_err() {
        let mut adapter = make_nlp_exact_match();
        let preds = MetricInput::Text(vec!["a".to_string(), "b".to_string()]);
        let refs = MetricInput::Text(vec!["a".to_string()]);
        assert!(adapter.add_batch(&preds, &refs).is_err());
    }

    #[test]
    fn test_tensor_adapter_names() {
        assert_eq!(TensorAdapter::accuracy().name(), "tensor_accuracy");
        assert_eq!(TensorAdapter::f1("macro".to_string()).name(), "tensor_f1");
        assert_eq!(TensorAdapter::perplexity().name(), "tensor_perplexity");
    }
}
