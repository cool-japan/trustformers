use crate::error::{Result, TrustformersError};
use crate::pipeline::{BasePipeline, ClassificationOutput, Pipeline, PipelineOutput};
use crate::AutoModel;
use crate::AutoTokenizer;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use trustformers_core::cache::CacheKeyBuilder;
use trustformers_core::traits::{Model, Tokenizer};

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
                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

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
                                sample_results
                                    .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
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
