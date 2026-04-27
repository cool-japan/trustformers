use crate::error::{Result, TrustformersError};
use crate::pipeline::{BasePipeline, ClassificationOutput, Pipeline, PipelineOutput};
use crate::AutoModel;
use crate::AutoTokenizer;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use trustformers_core::cache::CacheKeyBuilder;
use trustformers_core::traits::{Model, Tokenizer};

// ---------------------------------------------------------------------------
// ClassificationResult — richer result type
// ---------------------------------------------------------------------------

/// A single label prediction with its numeric score.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClassificationResult {
    /// Human-readable label string.
    pub label: String,
    /// Confidence score (0..1 after softmax/sigmoid).
    pub score: f32,
    /// Zero-based index into the label vocabulary.
    pub label_id: usize,
}

// ---------------------------------------------------------------------------
// ClassificationPostprocessor — pure numeric helpers
// ---------------------------------------------------------------------------

/// Stateless post-processing utilities for classification pipelines.
pub struct ClassificationPostprocessor;

impl ClassificationPostprocessor {
    /// Convert logits to `ClassificationResult` objects sorted by score descending.
    ///
    /// `id2label` maps position index → label string.
    pub fn logits_to_labels(logits: &[f32], id2label: &[String]) -> Vec<ClassificationResult> {
        let probs = Self::apply_softmax(logits);
        let mut results: Vec<ClassificationResult> = probs
            .into_iter()
            .enumerate()
            .map(|(i, score)| ClassificationResult {
                label: id2label.get(i).cloned().unwrap_or_else(|| format!("LABEL_{i}")),
                score,
                label_id: i,
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Element-wise sigmoid — use for multi-label classification.
    pub fn apply_sigmoid(logits: &[f32]) -> Vec<f32> {
        logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }

    /// Numerically-stable softmax — use for single-label classification.
    pub fn apply_softmax(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum < f32::EPSILON {
            return vec![1.0 / logits.len() as f32; logits.len()];
        }
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Multi-label prediction with a threshold: apply sigmoid then keep entries ≥ threshold.
    ///
    /// Results are sorted by score descending.
    pub fn threshold_predictions(
        scores: &[f32],
        labels: &[String],
        threshold: f32,
    ) -> Vec<ClassificationResult> {
        let mut results: Vec<ClassificationResult> = scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s >= threshold)
            .map(|(i, &s)| ClassificationResult {
                label: labels.get(i).cloned().unwrap_or_else(|| format!("LABEL_{i}")),
                score: s,
                label_id: i,
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Return the top-`k` results from an already-sorted slice.
    pub fn top_k_labels(results: &[ClassificationResult], k: usize) -> Vec<ClassificationResult> {
        let mut sorted = results.to_vec();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        sorted
    }
}

/// Configuration for text classification pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextClassificationConfig {
    /// Maximum sequence length
    pub max_length: usize,
    /// Labels for classification
    pub labels: Vec<String>,
    /// Return scores for all labels
    pub return_all_scores: bool,
}

impl Default for TextClassificationConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            labels: vec!["NEGATIVE".to_string(), "POSITIVE".to_string()],
            return_all_scores: false,
        }
    }
}

/// Pipeline for text classification tasks
#[derive(Clone)]
pub struct TextClassificationPipeline {
    base: BasePipeline<AutoModel, AutoTokenizer>,
    labels: Arc<Vec<String>>,
}

impl TextClassificationPipeline {
    pub fn new(model: AutoModel, tokenizer: AutoTokenizer) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            labels: Arc::new(vec!["NEGATIVE".to_string(), "POSITIVE".to_string()]), // Default labels
        })
    }

    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Arc::new(labels);
        self
    }

    fn classify(&self, text: &str) -> Result<Vec<ClassificationOutput>> {
        // Check cache if enabled
        if let Some(cache) = &self.base.cache {
            // Build cache key
            let cache_key = CacheKeyBuilder::new("text-classification", "text-classification")
                .with_text(text)
                .with_param("max_length", &self.base.max_length)
                .build();

            // Try to get from cache
            if let Some(cached_data) = cache.get(&cache_key) {
                // Deserialize cached results
                if let Ok(results) =
                    serde_json::from_slice::<Vec<ClassificationOutput>>(&cached_data)
                {
                    return Ok(results);
                }
            }
        }

        // Tokenize input
        let inputs = self.base.tokenizer.encode(text)?;

        // Forward pass
        let results = match &self.base.model.model_type {
            #[cfg(feature = "bert")]
            crate::automodel::AutoModelType::BertForSequenceClassification(model) => {
                let outputs = model.forward(inputs)?;

                // Apply softmax to logits
                let logits = outputs.logits;
                let probs = softmax(&logits)?;

                // Create output
                let mut results = Vec::new();
                for (idx, &score) in probs.iter().enumerate() {
                    if idx < self.labels.len() {
                        results.push(ClassificationOutput {
                            label: self.labels[idx].clone(),
                            score,
                        });
                    }
                }

                // Sort by score descending
                results.sort_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                });

                results
            },
            _ => {
                return Err(TrustformersError::model(
                    "Model does not support sequence classification".to_string(),
                    "unknown",
                ))
            },
        };

        // Cache the results if enabled
        if let Some(cache) = &self.base.cache {
            let cache_key = CacheKeyBuilder::new("text-classification", "text-classification")
                .with_text(text)
                .with_param("max_length", &self.base.max_length)
                .build();

            // Serialize and cache
            if let Ok(serialized) = serde_json::to_vec(&results) {
                cache.insert(cache_key, serialized);
            }
        }

        Ok(results)
    }

    fn classify_batch(&self, texts: &[String]) -> Result<Vec<Vec<ClassificationOutput>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // If only one text, use the single classify method
        if texts.len() == 1 {
            return Ok(vec![self.classify(&texts[0])?]);
        }

        // Tokenize all texts and find the maximum length for padding
        let mut tokenized_inputs = Vec::new();
        let mut max_length = 0;

        for text in texts {
            let inputs = self.base.tokenizer.encode(text)?;
            max_length = max_length.max(inputs.input_ids.len());
            tokenized_inputs.push(inputs);
        }

        // Limit max_length to model's maximum if set
        max_length = max_length.min(self.base.max_length);

        // Pad all sequences to the same length
        let batch_size = texts.len();
        let mut batch_input_ids = Vec::new();
        let mut batch_attention_mask = Vec::new();

        for inputs in tokenized_inputs {
            let mut input_ids = inputs.input_ids;
            let mut attention_mask = inputs.attention_mask;

            // Truncate if necessary
            if input_ids.len() > max_length {
                input_ids.truncate(max_length);
                attention_mask.truncate(max_length);
            }

            // Pad to max_length
            while input_ids.len() < max_length {
                input_ids.push(0); // padding token ID
                attention_mask.push(0); // padding attention
            }

            batch_input_ids.extend(input_ids);
            batch_attention_mask.extend(attention_mask);
        }

        // Create batch TokenizedInput
        let batch_inputs = crate::core::traits::TokenizedInput {
            input_ids: batch_input_ids,
            attention_mask: batch_attention_mask,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        // Forward pass
        let results = match &self.base.model.model_type {
            #[cfg(feature = "bert")]
            crate::automodel::AutoModelType::BertForSequenceClassification(model) => {
                let outputs = model.forward(batch_inputs)?;

                // Process batch logits - shape should be [batch_size, num_labels]
                let logits = outputs.logits;
                let mut batch_results = Vec::new();

                // Extract logits for each sample in the batch
                match &logits {
                    trustformers_core::tensor::Tensor::F32(arr) => {
                        let shape = arr.shape();
                        if shape.len() == 2 && shape[0] == batch_size {
                            let num_labels = shape[1];

                            for batch_idx in 0..batch_size {
                                // Extract logits for this sample
                                let sample_logits: Vec<f32> = (0..num_labels)
                                    .map(|label_idx| arr[[batch_idx, label_idx]])
                                    .collect();

                                // Create tensor from logits and apply softmax
                                let logits_tensor =
                                    crate::Tensor::from_vec(sample_logits, &[num_labels])?;
                                let probs = softmax(&logits_tensor)?;

                                // Create classification output
                                let mut sample_results = Vec::new();
                                for (idx, &score) in probs.iter().enumerate() {
                                    if idx < self.labels.len() {
                                        sample_results.push(ClassificationOutput {
                                            label: self.labels[idx].clone(),
                                            score,
                                        });
                                    }
                                }

                                // Sort by score descending
                                sample_results.sort_by(|a, b| {
                                    b.score
                                        .partial_cmp(&a.score)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });
                                batch_results.push(sample_results);
                            }
                        } else {
                            // Fallback to sequential processing if batch shape is unexpected
                            return texts.iter().map(|text| self.classify(text)).collect();
                        }
                    },
                    _ => {
                        // Fallback to sequential processing for unsupported tensor types
                        return texts.iter().map(|text| self.classify(text)).collect();
                    },
                }

                batch_results
            },
            _ => {
                // For unsupported models, fallback to sequential processing
                return texts.iter().map(|text| self.classify(text)).collect();
            },
        };

        Ok(results)
    }
}

impl Pipeline for TextClassificationPipeline {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let results = self.classify(&input)?;
        Ok(PipelineOutput::Classification(results))
    }

    fn batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        let batch_results = self.classify_batch(&inputs)?;
        Ok(batch_results.into_iter().map(PipelineOutput::Classification).collect())
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl crate::pipeline::AsyncPipeline for TextClassificationPipeline {
    type Input = String;
    type Output = PipelineOutput;

    async fn __call_async__(&self, input: Self::Input) -> Result<Self::Output> {
        // For CPU operations, we can use tokio::task::spawn_blocking
        let pipeline = self.clone();
        tokio::task::spawn_blocking(move || pipeline.__call__(input))
            .await
            .map_err(|e| {
                TrustformersError::runtime_error(format!(
                    "text-classification pipeline error: {}",
                    e
                ))
            })?
    }
}

/// Simple softmax implementation
fn softmax(logits: &crate::Tensor) -> Result<Vec<f32>> {
    let data = logits.data()?;
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let exp_sum: f32 = data.iter().map(|&x| (x - max).exp()).sum();

    let probs: Vec<f32> = data.iter().map(|&x| (x - max).exp() / exp_sum).collect();

    Ok(probs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Vec<String> {
        vec![
            "NEGATIVE".to_string(),
            "POSITIVE".to_string(),
            "NEUTRAL".to_string(),
        ]
    }

    // -----------------------------------------------------------------------
    // ClassificationPostprocessor::apply_softmax
    // -----------------------------------------------------------------------

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let probs = ClassificationPostprocessor::apply_softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_highest_logit_has_highest_prob() {
        let logits = vec![0.0f32, 0.0, 10.0];
        let probs = ClassificationPostprocessor::apply_softmax(&logits);
        assert!(probs[2] > probs[0]);
        assert!(probs[2] > probs[1]);
    }

    #[test]
    fn softmax_equal_logits_equal_probs() {
        let logits = vec![2.0f32; 4];
        let probs = ClassificationPostprocessor::apply_softmax(&logits);
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_empty_returns_empty() {
        let probs = ClassificationPostprocessor::apply_softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn softmax_negative_logits() {
        let logits = vec![-5.0f32, -1.0, -3.0];
        let probs = ClassificationPostprocessor::apply_softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[1] > probs[2]);
        assert!(probs[2] > probs[0]);
    }

    // -----------------------------------------------------------------------
    // ClassificationPostprocessor::apply_sigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn sigmoid_zero_returns_half() {
        let scores = ClassificationPostprocessor::apply_sigmoid(&[0.0]);
        assert!((scores[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn sigmoid_large_positive_approaches_one() {
        let scores = ClassificationPostprocessor::apply_sigmoid(&[100.0]);
        assert!(scores[0] > 0.999);
    }

    #[test]
    fn sigmoid_large_negative_approaches_zero() {
        let scores = ClassificationPostprocessor::apply_sigmoid(&[-100.0]);
        assert!(scores[0] < 0.001);
    }

    #[test]
    fn sigmoid_does_not_sum_to_one() {
        // Unlike softmax, sigmoid outputs are independent and need not sum to 1
        let logits = vec![1.0f32, 1.0, 1.0, 1.0];
        let scores = ClassificationPostprocessor::apply_sigmoid(&logits);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() > 0.1, "sigmoid sum was {sum}");
    }

    #[test]
    fn sigmoid_empty_returns_empty() {
        let scores = ClassificationPostprocessor::apply_sigmoid(&[]);
        assert!(scores.is_empty());
    }

    // -----------------------------------------------------------------------
    // ClassificationPostprocessor::logits_to_labels
    // -----------------------------------------------------------------------

    #[test]
    fn logits_to_labels_sorted_descending() {
        let logits = vec![1.0f32, 3.0, 2.0];
        let lbls = labels();
        let results = ClassificationPostprocessor::logits_to_labels(&logits, &lbls);
        assert_eq!(results.len(), 3);
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[test]
    fn logits_to_labels_top_is_positive() {
        // POSITIVE (index 1) has highest logit
        let logits = vec![0.1f32, 5.0, 0.2];
        let lbls = labels();
        let results = ClassificationPostprocessor::logits_to_labels(&logits, &lbls);
        assert_eq!(results[0].label, "POSITIVE");
        assert_eq!(results[0].label_id, 1);
    }

    #[test]
    fn logits_to_labels_scores_sum_to_one() {
        let logits = vec![1.0f32, 2.0, 0.5];
        let lbls = labels();
        let results = ClassificationPostprocessor::logits_to_labels(&logits, &lbls);
        let sum: f32 = results.iter().map(|r| r.score).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn logits_to_labels_fallback_label_name() {
        // Fewer labels than logits → fallback names
        let logits = vec![1.0f32, 2.0, 0.5];
        let lbls = vec!["A".to_string()]; // only one label for three logits
        let results = ClassificationPostprocessor::logits_to_labels(&logits, &lbls);
        assert_eq!(results.len(), 3);
        // The two fallbacks should use "LABEL_N" pattern
        let has_fallback = results.iter().any(|r| r.label.starts_with("LABEL_"));
        assert!(has_fallback);
    }

    // -----------------------------------------------------------------------
    // ClassificationPostprocessor::threshold_predictions
    // -----------------------------------------------------------------------

    #[test]
    fn threshold_predictions_basic() {
        let scores = vec![0.9f32, 0.3, 0.7, 0.1];
        let lbls = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];
        let results = ClassificationPostprocessor::threshold_predictions(&scores, &lbls, 0.5);
        assert_eq!(results.len(), 2);
        let result_labels: Vec<&str> = results.iter().map(|r| r.label.as_str()).collect();
        assert!(result_labels.contains(&"A"));
        assert!(result_labels.contains(&"C"));
    }

    #[test]
    fn threshold_predictions_none_above_threshold() {
        let scores = vec![0.1f32, 0.2, 0.3];
        let lbls = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let results = ClassificationPostprocessor::threshold_predictions(&scores, &lbls, 0.9);
        assert!(results.is_empty());
    }

    #[test]
    fn threshold_predictions_all_above_threshold() {
        let scores = vec![0.8f32, 0.9, 0.7];
        let lbls = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let results = ClassificationPostprocessor::threshold_predictions(&scores, &lbls, 0.5);
        assert_eq!(results.len(), 3);
        // Sorted descending
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn threshold_predictions_sorted_descending() {
        let scores = vec![0.6f32, 0.9, 0.7];
        let lbls = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let results = ClassificationPostprocessor::threshold_predictions(&scores, &lbls, 0.0);
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    // -----------------------------------------------------------------------
    // ClassificationPostprocessor::top_k_labels
    // -----------------------------------------------------------------------

    #[test]
    fn top_k_labels_returns_k_items() {
        let results = vec![
            ClassificationResult {
                label: "A".to_string(),
                score: 0.5,
                label_id: 0,
            },
            ClassificationResult {
                label: "B".to_string(),
                score: 0.8,
                label_id: 1,
            },
            ClassificationResult {
                label: "C".to_string(),
                score: 0.3,
                label_id: 2,
            },
        ];
        let top = ClassificationPostprocessor::top_k_labels(&results, 2);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn top_k_labels_sorted_descending() {
        let results = vec![
            ClassificationResult {
                label: "A".to_string(),
                score: 0.3,
                label_id: 0,
            },
            ClassificationResult {
                label: "B".to_string(),
                score: 0.9,
                label_id: 1,
            },
            ClassificationResult {
                label: "C".to_string(),
                score: 0.6,
                label_id: 2,
            },
        ];
        let top = ClassificationPostprocessor::top_k_labels(&results, 3);
        assert_eq!(top[0].label, "B");
        assert_eq!(top[1].label, "C");
        assert_eq!(top[2].label, "A");
    }

    #[test]
    fn top_k_larger_than_slice() {
        let results = vec![ClassificationResult {
            label: "A".to_string(),
            score: 0.5,
            label_id: 0,
        }];
        let top = ClassificationPostprocessor::top_k_labels(&results, 10);
        assert_eq!(top.len(), 1);
    }

    // -----------------------------------------------------------------------
    // ClassificationResult struct
    // -----------------------------------------------------------------------

    #[test]
    fn classification_result_serde_roundtrip() {
        let r = ClassificationResult {
            label: "POSITIVE".to_string(),
            score: 0.95,
            label_id: 1,
        };
        let json = serde_json::to_string(&r).expect("serialize");
        let back: ClassificationResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.label, r.label);
        assert_eq!(back.label_id, r.label_id);
        assert!((back.score - r.score).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_vs_softmax_multi_label() {
        // For multi-label: sigmoid is better because classes are independent
        let logits = vec![2.0f32, -1.0, 0.5];
        let sigmoid_scores = ClassificationPostprocessor::apply_sigmoid(&logits);
        let softmax_scores = ClassificationPostprocessor::apply_softmax(&logits);
        // Sigmoid: each independently > 0 and < 1
        for &s in &sigmoid_scores {
            assert!(s > 0.0 && s < 1.0);
        }
        // Softmax sums to 1; sigmoid does not
        let sig_sum: f32 = sigmoid_scores.iter().sum();
        let sft_sum: f32 = softmax_scores.iter().sum();
        assert!((sft_sum - 1.0).abs() < 1e-5);
        assert!((sig_sum - 1.0).abs() > 0.1);
    }
}
