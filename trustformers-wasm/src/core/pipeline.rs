//! WebAssembly-compatible NLP pipelines

use crate::core::model::{ModelArchitecture, ModelConfig, WasmModel};
use crate::core::tensor::WasmTensor;
use crate::core::tokenizer::{TokenizerType, WasmTokenizer};
use serde::{Deserialize, Serialize};
use std::string::{String, ToString};
use std::vec::Vec;
use std::{format, vec};
use wasm_bindgen::prelude::*;

/// Pipeline type
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineType {
    TextGeneration,
    TextClassification,
    TokenClassification,
    QuestionAnswering,
    Summarization,
    Translation,
}

/// Generation parameters
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_length: usize,
    pub min_length: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub num_beams: usize,
    pub do_sample: bool,
    pub early_stopping: bool,
    pub repetition_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 50,
            min_length: 1,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            num_beams: 1,
            do_sample: true,
            early_stopping: true,
            repetition_penalty: 1.0,
        }
    }
}

/// Text generation pipeline
#[wasm_bindgen]
pub struct TextGenerationPipeline {
    model: WasmModel,
    tokenizer: WasmTokenizer,
    config: GenerationConfig,
}

#[wasm_bindgen]
impl TextGenerationPipeline {
    /// Create a new text generation pipeline
    #[wasm_bindgen(constructor)]
    pub fn new(model: WasmModel, tokenizer: WasmTokenizer) -> Self {
        Self {
            model,
            tokenizer,
            config: GenerationConfig::default(),
        }
    }

    /// Generate text from a prompt
    pub async fn generate(&self, prompt: &str) -> Result<String, JsValue> {
        // Tokenize input
        let input_ids = self.tokenizer.encode(prompt, true);
        let input_tensor = WasmTensor::new(
            input_ids.iter().map(|&id| id as f32).collect(),
            vec![1, input_ids.len()],
        )?;

        // Generate tokens
        let mut generated_ids = input_ids.clone();
        let _past_key_values: Option<Vec<WasmTensor>> = None;

        for _ in 0..self.config.max_length {
            // Forward pass
            let outputs = self.model.forward(&input_tensor)?;

            // Get next token (simplified - just take argmax of last position)
            let logits = outputs.data();
            let vocab_size = self.model.config().vocab_size;
            let last_logits = &logits[logits.len() - vocab_size..];

            let next_token_id = if self.config.do_sample {
                self.sample_token(last_logits)?
            } else {
                self.argmax(last_logits)
            };

            generated_ids.push(next_token_id);

            // Check stopping conditions
            if self.should_stop(&generated_ids) {
                break;
            }
        }

        // Decode generated tokens
        let generated_text = self.tokenizer.decode(generated_ids, true);
        Ok(generated_text)
    }

    /// Generate text with streaming support - yields tokens incrementally
    pub async fn generate_stream(
        &self,
        prompt: &str,
        callback: &js_sys::Function,
    ) -> Result<String, JsValue> {
        // Tokenize input
        let input_ids = self.tokenizer.encode(prompt, true);
        let input_tensor = WasmTensor::new(
            input_ids.iter().map(|&id| id as f32).collect(),
            vec![1, input_ids.len()],
        )?;

        // Generate tokens
        let mut generated_ids = input_ids.clone();
        let _past_key_values: Option<Vec<WasmTensor>> = None;
        let mut generated_text = String::new();

        for step in 0..self.config.max_length {
            // Forward pass
            let outputs = self.model.forward(&input_tensor)?;

            // Get next token
            let logits = outputs.data();
            let vocab_size = self.model.config().vocab_size;
            let last_logits = &logits[logits.len() - vocab_size..];

            let next_token_id = if self.config.do_sample {
                self.sample_token(last_logits)?
            } else {
                self.argmax(last_logits)
            };

            generated_ids.push(next_token_id);

            // Decode new token
            let new_token_text = self.tokenizer.decode(vec![next_token_id], false);
            generated_text.push_str(&new_token_text);

            // Call callback with progress
            let progress = StreamProgress {
                step,
                total_steps: self.config.max_length,
                token: new_token_text.clone(),
                partial_text: generated_text.clone(),
                is_complete: false,
            };

            let this = JsValue::null();
            let progress_js = serde_wasm_bindgen::to_value(&progress)?;
            callback.call1(&this, &progress_js)?;

            // Check stopping conditions
            if self.should_stop(&generated_ids) {
                break;
            }

            // Yield control to allow UI updates
            wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&JsValue::from(0)))
                .await?;
        }

        // Final callback
        let final_progress = StreamProgress {
            step: self.config.max_length,
            total_steps: self.config.max_length,
            token: String::new(),
            partial_text: generated_text.clone(),
            is_complete: true,
        };

        let this = JsValue::null();
        let progress_js = serde_wasm_bindgen::to_value(&final_progress)?;
        callback.call1(&this, &progress_js)?;

        Ok(generated_text)
    }

    /// Set generation configuration
    pub fn set_config(&mut self, config: GenerationConfig) {
        self.config = config;
    }

    /// Generate multiple sequences
    pub async fn generate_batch(&self, prompts: Vec<String>) -> Result<Vec<String>, JsValue> {
        let mut results = Vec::new();

        for prompt in prompts {
            let generated = self.generate(&prompt).await?;
            results.push(generated);
        }

        Ok(results)
    }

    // Private helper methods

    fn sample_token(&self, logits: &[f32]) -> Result<u32, JsValue> {
        // Apply temperature
        let scaled_logits: Vec<f32> = logits.iter().map(|&l| l / self.config.temperature).collect();

        // Apply top-k filtering
        let mut indexed_logits: Vec<(usize, f32)> =
            scaled_logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed_logits.truncate(self.config.top_k);

        // Apply softmax
        let max_logit = indexed_logits.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = indexed_logits.iter().map(|(_, l)| (l - max_logit).exp()).sum();

        // Sample from distribution
        let mut rng_val = js_sys::Math::random() as f32;

        for &(idx, logit) in &indexed_logits {
            let prob = (logit - max_logit).exp() / exp_sum;
            rng_val -= prob;
            if rng_val <= 0.0 {
                return Ok(idx as u32);
            }
        }

        // Fallback to first token
        Ok(indexed_logits[0].0 as u32)
    }

    fn argmax(&self, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }

    fn should_stop(&self, token_ids: &[u32]) -> bool {
        // Check for EOS token or max length
        if token_ids.len() >= self.config.max_length {
            return true;
        }

        // Check for EOS token (simplified)
        if let Some(&last_id) = token_ids.last() {
            // Common EOS token IDs
            if last_id == 2 || last_id == 50256 {
                return true;
            }
        }

        false
    }
}

/// Text classification pipeline
#[wasm_bindgen]
pub struct TextClassificationPipeline {
    model: WasmModel,
    tokenizer: WasmTokenizer,
    labels: Vec<String>,
}

#[wasm_bindgen]
impl TextClassificationPipeline {
    /// Create a new text classification pipeline
    #[wasm_bindgen(constructor)]
    pub fn new(model: WasmModel, tokenizer: WasmTokenizer) -> Self {
        Self {
            model,
            tokenizer,
            labels: vec!["negative".to_string(), "positive".to_string()],
        }
    }

    /// Set classification labels
    pub fn set_labels(&mut self, labels: Vec<String>) {
        self.labels = labels;
    }

    /// Classify text
    pub async fn classify(&self, text: &str) -> Result<ClassificationResult, JsValue> {
        // Tokenize input
        let input_ids = self.tokenizer.encode(text, true);
        let input_tensor = WasmTensor::new(
            input_ids.iter().map(|&id| id as f32).collect(),
            vec![1, input_ids.len()],
        )?;

        // Forward pass
        let outputs = self.model.forward(&input_tensor)?;

        // Get classification logits (assuming last hidden state -> classification head)
        let logits = outputs.data();
        let num_labels = self.labels.len();
        let classification_logits = &logits[logits.len() - num_labels..];

        // Apply softmax
        let probs = self.softmax(classification_logits);

        // Find best label
        let (label_idx, score) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, &score)| (idx, score))
            .unwrap_or((0, 0.0));

        Ok(ClassificationResult {
            label: self.labels[label_idx].clone(),
            score,
            all_scores: probs,
        })
    }

    /// Classify multiple texts
    pub async fn classify_batch(
        &self,
        texts: Vec<String>,
    ) -> Result<Vec<ClassificationResult>, JsValue> {
        let mut results = Vec::new();

        for text in texts {
            let result = self.classify(&text).await?;
            results.push(result);
        }

        Ok(results)
    }

    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        logits.iter().map(|&l| (l - max_logit).exp() / exp_sum).collect()
    }
}

/// Classification result
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    label: String,
    score: f32,
    all_scores: Vec<f32>,
}

#[wasm_bindgen]
impl ClassificationResult {
    #[wasm_bindgen(getter)]
    pub fn label(&self) -> String {
        self.label.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f32 {
        self.score
    }

    #[wasm_bindgen(getter)]
    pub fn all_scores(&self) -> Vec<f32> {
        self.all_scores.clone()
    }
}

/// Question answering pipeline
#[wasm_bindgen]
pub struct QuestionAnsweringPipeline {
    model: WasmModel,
    tokenizer: WasmTokenizer,
}

#[wasm_bindgen]
impl QuestionAnsweringPipeline {
    /// Create a new question answering pipeline
    #[wasm_bindgen(constructor)]
    pub fn new(model: WasmModel, tokenizer: WasmTokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Answer a question given context
    pub async fn answer(&self, question: &str, context: &str) -> Result<AnswerResult, JsValue> {
        // Tokenize question and context
        let question_tokens = self.tokenizer.encode(question, false);
        let context_tokens = self.tokenizer.encode(context, false);

        // Combine with special tokens
        let mut input_ids = vec![101]; // [CLS]
        input_ids.extend(&question_tokens);
        input_ids.push(102); // [SEP]
        input_ids.extend(&context_tokens);
        input_ids.push(102); // [SEP]

        let input_tensor = WasmTensor::new(
            input_ids.iter().map(|&id| id as f32).collect(),
            vec![1, input_ids.len()],
        )?;

        // Forward pass
        let outputs = self.model.forward(&input_tensor)?;

        // Get start and end logits (simplified)
        let logits = outputs.data();
        let seq_len = input_ids.len();
        let start_logits = &logits[0..seq_len];
        let end_logits = &logits[seq_len..2 * seq_len];

        // Find best span
        let (start_idx, end_idx) =
            self.find_best_span(start_logits, end_logits, question_tokens.len() + 2);

        // Extract answer tokens
        let answer_tokens: Vec<u32> = input_ids[start_idx..=end_idx].to_vec();
        let answer_text = self.tokenizer.decode(answer_tokens, true);

        Ok(AnswerResult {
            answer: answer_text,
            start: start_idx,
            end: end_idx,
            score: (start_logits[start_idx] + end_logits[end_idx]) / 2.0,
        })
    }

    fn find_best_span(
        &self,
        start_logits: &[f32],
        end_logits: &[f32],
        context_start: usize,
    ) -> (usize, usize) {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_start = context_start;
        let mut best_end = context_start;

        for (i, &start_val) in start_logits.iter().enumerate().skip(context_start) {
            for (j, &end_val) in end_logits
                .iter()
                .enumerate()
                .skip(i)
                .take(core::cmp::min(20, end_logits.len() - i))
            {
                // Max answer length of 20
                let score = start_val + end_val;
                if score > best_score {
                    best_score = score;
                    best_start = i;
                    best_end = j + i;
                }
            }
        }

        (best_start, best_end)
    }
}

/// Answer result
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerResult {
    answer: String,
    start: usize,
    end: usize,
    score: f32,
}

#[wasm_bindgen]
impl AnswerResult {
    #[wasm_bindgen(getter)]
    pub fn answer(&self) -> String {
        self.answer.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn start(&self) -> usize {
        self.start
    }

    #[wasm_bindgen(getter)]
    pub fn end(&self) -> usize {
        self.end
    }

    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f32 {
        self.score
    }
}

/// Pipeline factory
#[wasm_bindgen]
pub struct PipelineFactory;

#[wasm_bindgen]
impl PipelineFactory {
    /// Create a pipeline from model name
    pub async fn from_pretrained(
        pipeline_type: PipelineType,
        model_name: &str,
    ) -> Result<JsValue, JsValue> {
        // Determine model architecture from name
        let architecture = if model_name.contains("bert") {
            ModelArchitecture::Bert
        } else if model_name.contains("gpt2") {
            ModelArchitecture::GPT2
        } else if model_name.contains("t5") {
            ModelArchitecture::T5
        } else if model_name.contains("llama") {
            ModelArchitecture::Llama
        } else {
            ModelArchitecture::Bert // Default
        };

        // Create model and tokenizer
        let config = ModelConfig::new(architecture);
        let mut model = WasmModel::new(config);
        model.load_from_url(&format!("https://models.example.com/{model_name}")).await?;

        let tokenizer_type = match architecture {
            ModelArchitecture::Bert => TokenizerType::WordPiece,
            ModelArchitecture::GPT2 => TokenizerType::BPE,
            _ => TokenizerType::WordPiece,
        };
        let tokenizer = WasmTokenizer::new(tokenizer_type);

        // Create appropriate pipeline
        match pipeline_type {
            PipelineType::TextGeneration => {
                let pipeline = TextGenerationPipeline::new(model, tokenizer);
                Ok(JsValue::from(pipeline))
            },
            PipelineType::TextClassification => {
                let pipeline = TextClassificationPipeline::new(model, tokenizer);
                Ok(JsValue::from(pipeline))
            },
            PipelineType::QuestionAnswering => {
                let pipeline = QuestionAnsweringPipeline::new(model, tokenizer);
                Ok(JsValue::from(pipeline))
            },
            _ => Err(JsValue::from_str("Pipeline type not yet implemented")),
        }
    }
}

/// Progress information for streaming generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProgress {
    pub step: usize,
    pub total_steps: usize,
    pub token: String,
    pub partial_text: String,
    pub is_complete: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_length, 50);
        assert_eq!(config.temperature, 1.0);
    }
}
