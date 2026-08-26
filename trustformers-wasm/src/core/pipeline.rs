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
        let input_ids = self.tokenizer.encode(prompt, true)?;
        let generated_ids = self.generate_ids(&input_ids)?;

        // Decode generated tokens
        let generated_text = self.tokenizer.decode(generated_ids, true)?;
        Ok(generated_text)
    }

    /// Autoregressively generate up to `self.config.max_length` new token
    /// ids on top of `prompt_ids`, feeding every previously generated token
    /// back into the model as context for the next step.
    ///
    /// This used to build `input_tensor` once from the prompt and then call
    /// `self.model.forward(&input_tensor)` in a loop *without ever
    /// rebuilding it* — every "generated" token was really just the
    /// argmax/sample of the same first next-token prediction, repeated
    /// `max_length` times. There is intentionally no KV cache here: each
    /// step reruns the full transformer over the whole growing sequence
    /// (`O(generated_len^2)` total), which is correct but not the fastest
    /// possible implementation — a real incremental cache is future work,
    /// not something to fake in the meantime.
    ///
    /// Stops (without erroring) once the sequence would exceed the model's
    /// `max_position_embeddings`, since `WasmModel::forward` rejects
    /// sequences longer than that.
    fn generate_ids(&self, prompt_ids: &[u32]) -> Result<Vec<u32>, JsValue> {
        let mut generated_ids = prompt_ids.to_vec();
        let max_position = self.model.config().max_position_embeddings;

        for _ in 0..self.config.max_length {
            if generated_ids.len() >= max_position {
                break;
            }

            let next_token_id = self.next_token(&generated_ids)?;
            generated_ids.push(next_token_id);

            if self.should_stop(&generated_ids) {
                break;
            }
        }

        Ok(generated_ids)
    }

    /// Run one real forward pass over `context_ids` and return the next
    /// token id (argmax or temperature/top-k sample, per
    /// [`GenerationConfig::do_sample`]). Rebuilds the input tensor from the
    /// full context every call — no KV cache, matching `Self::generate_ids`.
    ///
    /// Exposed publicly (unlike the rest of this pipeline's step-by-step
    /// internals) so external incremental/streaming callers - such as
    /// [`crate::streaming_generation::StreamingGenerator`] - can drive real
    /// model generation one token at a time without duplicating the
    /// forward-pass/logits-extraction/sampling logic, and without ever
    /// having to fabricate placeholder tokens themselves.
    pub fn next_token(&self, context_ids: &[u32]) -> Result<u32, JsValue> {
        self.next_token_with_confidence(context_ids).map(|(id, _confidence)| id)
    }

    /// Like [`Self::next_token`], but also returns the model's real softmax
    /// probability of the chosen token (its own confidence in that
    /// prediction) - not a fabricated placeholder value.
    ///
    /// Not `#[wasm_bindgen]`-exported: `wasm-bindgen` cannot describe a
    /// bare tuple return type across the JS boundary. Crate-internal
    /// callers (currently just
    /// [`crate::streaming_generation::StreamingGenerator`]) use it
    /// directly; JS callers get the same information indirectly via
    /// [`Self::next_token`] plus a separate confidence query if ever
    /// needed.
    pub(crate) fn next_token_with_confidence(
        &self,
        context_ids: &[u32],
    ) -> Result<(u32, f32), JsValue> {
        let vocab_size = self.model.config().vocab_size;

        let current_tensor = WasmTensor::new(
            context_ids.iter().map(|&id| id as f32).collect(),
            vec![1, context_ids.len()],
        )?;
        let outputs = self.model.forward(&current_tensor)?;

        let logits = outputs.data();
        if logits.len() < vocab_size {
            return Err(JsValue::from_str(
                "TextGenerationPipeline: model output is shorter than one vocabulary row",
            ));
        }
        let last_logits = &logits[logits.len() - vocab_size..];

        let token_id = if self.config.do_sample {
            self.sample_token(last_logits)?
        } else {
            self.argmax(last_logits)
        };

        let confidence = softmax_probability(last_logits, token_id as usize);

        Ok((token_id, confidence))
    }

    /// Tokenize `text` the same way [`Self::generate`] does. Errors if the
    /// pipeline's tokenizer has no vocabulary loaded (see
    /// [`crate::core::tokenizer::WasmTokenizer::encode`]).
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, JsValue> {
        self.tokenizer.encode(text, add_special_tokens)
    }

    /// Decode `token_ids` the same way [`Self::generate`] does. Errors if
    /// the pipeline's tokenizer has no vocabulary loaded (see
    /// [`crate::core::tokenizer::WasmTokenizer::decode`]).
    pub fn decode(
        &self,
        token_ids: Vec<u32>,
        skip_special_tokens: bool,
    ) -> Result<String, JsValue> {
        self.tokenizer.decode(token_ids, skip_special_tokens)
    }

    /// The model's maximum context length (`max_position_embeddings`),
    /// beyond which [`Self::next_token`] cannot be called.
    pub fn max_position_embeddings(&self) -> usize {
        self.model.config().max_position_embeddings
    }

    /// Generate text with streaming support - yields tokens incrementally
    pub async fn generate_stream(
        &self,
        prompt: &str,
        callback: &js_sys::Function,
    ) -> Result<String, JsValue> {
        // Tokenize input
        let input_ids = self.tokenizer.encode(prompt, true)?;

        // Generate tokens. As in `generate_ids`, the input tensor is
        // rebuilt from the full running context on every step — the old
        // code built it once from the prompt and reused it for every
        // iteration, so every streamed token was a copy of the same first
        // prediction.
        let mut generated_ids = input_ids.clone();
        let mut generated_text = String::new();
        let max_position = self.model.config().max_position_embeddings;

        for step in 0..self.config.max_length {
            if generated_ids.len() >= max_position {
                break;
            }

            let next_token_id = self.next_token(&generated_ids)?;
            generated_ids.push(next_token_id);

            // Decode new token
            let new_token_text = self.tokenizer.decode(vec![next_token_id], false)?;
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

        Self::is_eos_token(token_ids.last().copied())
    }

    /// Whether `token_id` is one of the recognized end-of-sequence token
    /// ids (simplified: common EOS ids across tokenizer families, not a
    /// per-tokenizer lookup). Exposed publicly, independent of
    /// `Self::should_stop`'s length-based cutoff, so callers that manage
    /// their own generation-length budget (like
    /// [`crate::streaming_generation::StreamingGenerator`]) can check for
    /// a real end-of-text condition without being coupled to this
    /// pipeline's own `config.max_length`.
    pub fn is_eos_token(token_id: Option<u32>) -> bool {
        matches!(token_id, Some(2) | Some(50256))
    }
}

/// The softmax probability of `logits[index]` among all of `logits`.
///
/// Used by [`TextGenerationPipeline::next_token_with_confidence`] to report
/// a token's real model-assigned probability, replacing the fabricated
/// `0.8 + Math::random() * 0.2` "confidence" that
/// `streaming_generation::StreamingGenerator` used to invent for every
/// token regardless of what (fake) token it was attached to.
fn softmax_probability(logits: &[f32], index: usize) -> f32 {
    if logits.is_empty() || index >= logits.len() {
        return 0.0;
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    if exp_sum <= 0.0 {
        return 0.0;
    }
    (logits[index] - max_logit).exp() / exp_sum
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
        let input_ids = self.tokenizer.encode(text, true)?;
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
        let question_tokens = self.tokenizer.encode(question, false)?;
        let context_tokens = self.tokenizer.encode(context, false)?;

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
        let answer_text = self.tokenizer.decode(answer_tokens, true)?;

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

/// Token-level classification result for a single token
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResult {
    token: String,
    label: String,
    score: f32,
}

#[wasm_bindgen]
impl TokenResult {
    #[wasm_bindgen(getter)]
    pub fn token(&self) -> String {
        self.token.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn label(&self) -> String {
        self.label.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f32 {
        self.score
    }
}

/// Token classification pipeline (NER / POS tagging).
///
/// Each token in the input is assigned an independent label from the label set.
/// The model is expected to produce per-token logits of shape `[seq_len * num_labels]`.
#[wasm_bindgen]
pub struct TokenClassificationPipeline {
    model: WasmModel,
    tokenizer: WasmTokenizer,
    labels: Vec<String>,
}

#[wasm_bindgen]
impl TokenClassificationPipeline {
    /// Create a new token classification pipeline with default NER labels (BIO scheme).
    #[wasm_bindgen(constructor)]
    pub fn new(model: WasmModel, tokenizer: WasmTokenizer) -> Self {
        Self {
            model,
            tokenizer,
            labels: vec![
                "O".to_string(),
                "B-PER".to_string(),
                "I-PER".to_string(),
                "B-ORG".to_string(),
                "I-ORG".to_string(),
                "B-LOC".to_string(),
                "I-LOC".to_string(),
                "B-MISC".to_string(),
                "I-MISC".to_string(),
            ],
        }
    }

    /// Override the label set.
    pub fn set_labels(&mut self, labels: Vec<String>) {
        self.labels = labels;
    }

    /// Classify each token in `text` and return one [`TokenResult`] per input token.
    ///
    /// Special tokens (`[CLS]` / `[SEP]`) are stripped from the output.
    pub async fn classify_tokens(&self, text: &str) -> Result<Vec<TokenResult>, JsValue> {
        let num_labels = self.labels.len();
        if num_labels == 0 {
            return Err(JsValue::from_str(
                "TokenClassificationPipeline: label set is empty",
            ));
        }

        // Tokenize with special tokens so the model sees the standard BERT layout.
        let input_ids = self.tokenizer.encode(text, true)?;
        let seq_len = input_ids.len();

        let input_tensor = WasmTensor::new(
            input_ids.iter().map(|&id| id as f32).collect(),
            vec![1, seq_len],
        )?;

        // Forward pass — expect logits of shape [seq_len * num_labels].
        let outputs = self.model.forward(&input_tensor)?;
        let logits = outputs.data();

        // If the model produced fewer logits than expected, fall back gracefully.
        let effective_labels = if logits.len() >= seq_len * num_labels {
            num_labels
        } else if seq_len > 0 && logits.len() >= seq_len {
            logits.len() / seq_len
        } else {
            1
        };

        let mut results = Vec::with_capacity(seq_len);
        for (token_idx, &token_id) in input_ids.iter().enumerate() {
            // Skip special tokens: [CLS]=101, [SEP]=102, [PAD]=0.
            if token_id == 101 || token_id == 102 || token_id == 0 {
                continue;
            }

            let offset = token_idx * effective_labels;
            let token_logits = if offset + effective_labels <= logits.len() {
                &logits[offset..offset + effective_labels]
            } else {
                // Not enough logit data for this position — assign O label.
                &logits[..0]
            };

            let (label_idx, score) = if token_logits.is_empty() {
                (0_usize, 0.0_f32)
            } else {
                // Softmax then argmax.
                let max_l = token_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = token_logits.iter().map(|&l| (l - max_l).exp()).sum();
                token_logits
                    .iter()
                    .enumerate()
                    .map(|(i, &l)| (i, (l - max_l).exp() / exp_sum))
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, 0.0))
            };

            let label = self.labels.get(label_idx).cloned().unwrap_or_else(|| "O".to_string());
            let token_str = self.tokenizer.decode(vec![token_id], false)?;

            results.push(TokenResult {
                token: token_str,
                label,
                score,
            });
        }

        Ok(results)
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
            PipelineType::TokenClassification => {
                let pipeline = TokenClassificationPipeline::new(model, tokenizer);
                Ok(JsValue::from(pipeline))
            },
            PipelineType::Summarization => Err(JsValue::from_str(
                "Summarization pipeline is not yet available in the WASM build. \
                 Sequence-to-sequence models require additional WASM support.",
            )),
            PipelineType::Translation => Err(JsValue::from_str(
                "Translation pipeline is not yet available in the WASM build. \
                 Sequence-to-sequence models require additional WASM support.",
            )),
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
    use crate::core::model::weights::{layer_prefix, NamedWeights};

    #[test]
    fn test_generation_config() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_length, 50);
        assert_eq!(config.temperature, 1.0);
    }

    /// Deterministic pseudo-random f32 generator (no external RNG dependency
    /// needed — just enough spread to make matmuls non-degenerate).
    fn fill(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 8) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn tensor(data: Vec<f32>, shape: Vec<usize>) -> WasmTensor {
        WasmTensor::new(data, shape).expect("valid tensor")
    }

    /// Build a tiny, fully-populated GPT-2-shaped `TextGenerationPipeline`
    /// (real weights, real tokenizer, no network/wasm-bindgen boundary) for
    /// exercising the autoregressive loop directly.
    fn build_test_pipeline() -> TextGenerationPipeline {
        build_test_pipeline_seeded(3)
    }

    fn build_test_pipeline_seeded(seed_base: u32) -> TextGenerationPipeline {
        let config = ModelConfig {
            architecture: ModelArchitecture::GPT2,
            vocab_size: 12,
            hidden_size: 8,
            num_layers: 2,
            num_heads: 2,
            max_position_embeddings: 16,
            intermediate_size: 10,
            hidden_dropout_prob: 0.0,
            attention_dropout_prob: 0.0,
        };
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let mut w = NamedWeights::new();
        let mut seed = seed_base;
        let mut next = |n: usize| {
            seed = seed.wrapping_add(211);
            fill(n, seed)
        };

        w.insert(
            "token_embeddings.weight",
            tensor(next(config.vocab_size * h), vec![config.vocab_size, h]),
        );
        w.insert(
            "position_embeddings.weight",
            tensor(
                next(config.max_position_embeddings * h),
                vec![config.max_position_embeddings, h],
            ),
        );
        for i in 0..config.num_layers {
            let p = layer_prefix(i);
            for name in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"] {
                w.insert(format!("{p}{name}.weight"), tensor(next(h * h), vec![h, h]));
            }
            w.insert(format!("{p}norm1.weight"), tensor(vec![1.0; h], vec![h]));
            w.insert(format!("{p}norm2.weight"), tensor(vec![1.0; h], vec![h]));
            w.insert(
                format!("{p}ffn.fc1.weight"),
                tensor(next(h * inter), vec![h, inter]),
            );
            w.insert(
                format!("{p}ffn.fc2.weight"),
                tensor(next(inter * h), vec![inter, h]),
            );
        }
        w.insert("final_norm.weight", tensor(vec![1.0; h], vec![h]));

        let vocab_size = config.vocab_size;
        let model = WasmModel::with_weights_for_test(config, w);
        let mut tokenizer = WasmTokenizer::new(TokenizerType::BPE);
        // Real (if tiny) vocabulary, loaded through the `JsValue`-free
        // native-test path (`load_vocab_map`, not the `#[wasm_bindgen]`
        // `load_vocab`, which needs a real `JsValue` and would panic
        // natively) - one single-byte-alphabet symbol per id, covering the
        // whole `vocab_size` range so any argmax-selected token id decodes
        // to real text.
        let vocab: std::collections::BTreeMap<String, u32> = (0u32..vocab_size as u32)
            .map(|id| (((b'a' + id as u8) as char).to_string(), id))
            .collect();
        tokenizer.load_vocab_map(vocab).expect("non-empty vocab");
        TextGenerationPipeline::new(model, tokenizer)
    }

    #[test]
    fn test_generate_ids_feeds_generated_tokens_back_into_context() {
        // Regression test for the former bug: `input_tensor` was built once
        // from the prompt and never rebuilt from `generated_ids`, so every
        // step's forward pass saw the identical frozen prompt tensor and
        // every generated token was a copy of the very first prediction.
        //
        // `do_sample: false` (argmax) keeps this deterministic and avoids
        // `js_sys::Math::random()`, which is unavailable on native targets.
        let mut pipeline = build_test_pipeline();
        pipeline.set_config(GenerationConfig {
            max_length: 6,
            do_sample: false,
            early_stopping: false,
            ..GenerationConfig::default()
        });

        let prompt_ids = vec![1u32, 2, 3];
        let generated = pipeline
            .generate_ids(&prompt_ids)
            .expect("generation over real weights should succeed");

        assert!(
            generated.len() > prompt_ids.len(),
            "must generate at least one new token"
        );
        let new_tokens = &generated[prompt_ids.len()..];

        assert!(
            new_tokens.iter().any(|&t| t != new_tokens[0]),
            "generated tokens must not all be identical — the old bug re-predicted the same \
             token every step because the input tensor was never rebuilt: {new_tokens:?}"
        );
    }

    #[test]
    fn test_generate_ids_stops_at_model_context_limit() {
        // max_position_embeddings is 16; start near that limit and ask for
        // far more new tokens than could possibly fit. The old code had no
        // notion of a context limit at all (it never grew the sequence), so
        // this guards the new stopping condition added alongside the fix.
        let mut pipeline = build_test_pipeline();
        pipeline.set_config(GenerationConfig {
            max_length: 100,
            do_sample: false,
            early_stopping: false,
            ..GenerationConfig::default()
        });

        let prompt_ids: Vec<u32> = (0..14).map(|i| i % 5).collect();
        let generated = pipeline
            .generate_ids(&prompt_ids)
            .expect("must stop cleanly at the context limit, not error");
        assert!(
            generated.len() <= 16,
            "must not exceed max_position_embeddings: {}",
            generated.len()
        );
    }

    #[test]
    fn test_generate_produces_nonempty_text_from_real_model() {
        let mut pipeline = build_test_pipeline();
        pipeline.set_config(GenerationConfig {
            max_length: 4,
            do_sample: false,
            early_stopping: false,
            ..GenerationConfig::default()
        });
        // `generate_ids` (not the async `generate`/wasm-bindgen boundary) —
        // native tests have no JS microtask queue to drive
        // `wasm_bindgen_futures`/`JsFuture`, but `generate`'s only `.await`
        // point lives in `generate_stream`, not here; `generate_ids` is the
        // synchronous core shared by both.
        let generated = pipeline.generate_ids(&[1, 2]).expect("generation should succeed");
        assert!(generated.len() >= 2);
    }

    // -----------------------------------------------------------------
    // `next_token`/`next_token_with_confidence`: the incremental,
    // one-token-at-a-time API extracted from `generate_ids` so external
    // callers (`streaming_generation::StreamingGenerator`) can drive real
    // generation without duplicating the forward-pass/sampling logic.
    // -----------------------------------------------------------------

    #[test]
    fn test_next_token_matches_first_step_of_generate_ids() {
        let mut pipeline = build_test_pipeline();
        pipeline.set_config(GenerationConfig {
            max_length: 1,
            do_sample: false,
            early_stopping: false,
            ..GenerationConfig::default()
        });

        let prompt_ids = [1u32, 2, 3];
        let via_next_token = pipeline.next_token(&prompt_ids).expect("next_token should succeed");
        let via_generate_ids =
            pipeline.generate_ids(&prompt_ids).expect("generate_ids should succeed");

        assert_eq!(via_generate_ids.len(), prompt_ids.len() + 1);
        assert_eq!(via_generate_ids[prompt_ids.len()], via_next_token);
    }

    #[test]
    fn test_next_token_with_confidence_reports_a_real_probability() {
        let mut pipeline = build_test_pipeline();
        pipeline.set_config(GenerationConfig {
            do_sample: false,
            ..GenerationConfig::default()
        });

        let (token_id, confidence) =
            pipeline.next_token_with_confidence(&[1, 2, 3]).expect("should succeed");

        assert!(
            (0.0..=1.0).contains(&confidence),
            "confidence must be a valid softmax probability: {confidence}"
        );
        assert!((token_id as usize) < pipeline.model.config().vocab_size);
    }

    #[test]
    fn test_next_token_with_confidence_varies_with_different_contexts() {
        // Regression guard for the property `streaming_generation`'s old
        // fabricated confidence never had: a real softmax probability is a
        // deterministic function of the model's real logits for a given
        // context, so two different contexts over the same (deterministic,
        // argmax) pipeline generally produce different confidences - unlike
        // the old `0.8 + Math::random() * 0.2`, which was independent of
        // any context and merely redrawn from the same fixed range.
        let mut pipeline = build_test_pipeline();
        pipeline.set_config(GenerationConfig {
            do_sample: false,
            ..GenerationConfig::default()
        });

        let (_id_a, confidence_a) =
            pipeline.next_token_with_confidence(&[1, 2, 3]).expect("should succeed");
        let (_id_b, confidence_b) =
            pipeline.next_token_with_confidence(&[4, 5, 6, 7]).expect("should succeed");

        assert_ne!(
            confidence_a, confidence_b,
            "confidence should be a genuine function of context, not a fixed/random constant"
        );
    }

    /// Regression guard: `TextGenerationPipeline::encode`/`decode` used to
    /// be infallible passthroughs to a tokenizer that could never fail
    /// (real fabricated vocabulary always available); now that
    /// `WasmTokenizer` requires a real, loaded vocabulary, these must
    /// propagate that as a `Result` rather than panicking or silently
    /// returning nothing.
    #[test]
    fn test_pipeline_encode_decode_round_trip_with_real_vocab() {
        let pipeline = build_test_pipeline();
        let ids = pipeline.encode("abc", false).expect("real vocab must encode");
        assert!(!ids.is_empty());
        let text = pipeline.decode(ids, false).expect("real vocab must decode");
        assert_eq!(text, "abc");
    }

    #[test]
    fn test_is_eos_token() {
        assert!(TextGenerationPipeline::is_eos_token(Some(2)));
        assert!(TextGenerationPipeline::is_eos_token(Some(50256)));
        assert!(!TextGenerationPipeline::is_eos_token(Some(5)));
        assert!(!TextGenerationPipeline::is_eos_token(None));
    }

    #[test]
    fn test_softmax_probability_sums_to_one_across_all_indices() {
        let logits = [1.0f32, 2.0, 0.5, -1.0];
        let total: f32 = (0..logits.len()).map(|i| softmax_probability(&logits, i)).sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "softmax probabilities must sum to 1, got {total}"
        );
    }

    #[test]
    fn test_softmax_probability_out_of_bounds_index_is_zero_not_panic() {
        let logits = [1.0f32, 2.0, 0.5];
        assert_eq!(softmax_probability(&logits, 10), 0.0);
        assert_eq!(softmax_probability(&[], 0), 0.0);
    }
}
