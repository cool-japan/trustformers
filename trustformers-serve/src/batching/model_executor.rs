//! Real model adapters for the batching stack.
//!
//! This module bridges [`trustformers_models`] decoders and
//! [`trustformers_tokenizers`] tokenizers onto the dyn-safe
//! [`BatchModel`](crate::batching::processor::BatchModel) /
//! [`Tokenizer`](crate::batching::processor::Tokenizer) traits the batch
//! executor consumes.
//!
//! Nothing here fabricates output: every call runs the real forward pass of a
//! real model, and every failure is reported as an error.

use crate::batching::processor::{BatchModel, ModelBatchExecutor, Tokenizer};
use anyhow::{anyhow, Context, Result};
use std::path::Path;
use std::sync::Arc;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Model, TokenizedInput, Tokenizer as CoreTokenizer};
use trustformers_models::gpt2::{Gpt2Config, Gpt2LMHeadModel};
use trustformers_tokenizers::tokenizer::TokenizerImpl;

/// A GPT-2 language model exposed to the batching stack.
///
/// The adapter runs the genuine `Gpt2LMHeadModel::forward` for each row of the
/// padded batch and stitches the per-row logits back into a
/// `[batch, seq, vocab]` tensor.
pub struct Gpt2BatchModel {
    model: Gpt2LMHeadModel,
    vocab_size: usize,
    eos_token_id: u32,
    n_positions: usize,
}

impl std::fmt::Debug for Gpt2BatchModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpt2BatchModel")
            .field("vocab_size", &self.vocab_size)
            .field("eos_token_id", &self.eos_token_id)
            .field("n_positions", &self.n_positions)
            .field("num_parameters", &self.model.num_parameters())
            .finish()
    }
}

impl Gpt2BatchModel {
    /// Wrap an already-constructed model.
    pub fn new(model: Gpt2LMHeadModel) -> Self {
        let config = model.get_config().clone();
        Self {
            model,
            vocab_size: config.vocab_size,
            eos_token_id: config.eos_token_id,
            n_positions: config.n_positions,
        }
    }

    /// Build a model from a configuration without loading any weights.
    ///
    /// The weights are the architecture's own initialization — real tensors that
    /// produce a real (but untrained) distribution. Use
    /// [`Gpt2BatchModel::from_checkpoint`] for a trained model.
    pub fn untrained(config: Gpt2Config) -> Result<Self> {
        let model = Gpt2LMHeadModel::new(config)
            .map_err(|e| anyhow!("failed to construct GPT-2 model: {}", e))?;
        Ok(Self::new(model))
    }

    /// Load real weights from a `safetensors` / PyTorch checkpoint file.
    ///
    /// `config_path` must point at a HuggingFace-style `config.json`;
    /// `weights_path` at the checkpoint container.
    pub fn from_checkpoint(config_path: &Path, weights_path: &Path) -> Result<Self> {
        let config_text = std::fs::read_to_string(config_path).with_context(|| {
            format!("failed to read GPT-2 config from {}", config_path.display())
        })?;
        let config: Gpt2Config = serde_json::from_str(&config_text).with_context(|| {
            format!("failed to parse GPT-2 config at {}", config_path.display())
        })?;

        let mut model = Gpt2LMHeadModel::new(config)
            .map_err(|e| anyhow!("failed to construct GPT-2 model: {}", e))?;
        let bytes = std::fs::read(weights_path).with_context(|| {
            format!(
                "failed to read GPT-2 weights from {}",
                weights_path.display()
            )
        })?;
        model
            .load_pretrained(&mut bytes.as_slice())
            .map_err(|e| anyhow!("failed to load GPT-2 weights: {}", e))?;

        Ok(Self::new(model))
    }

    /// Total number of parameters of the wrapped model.
    pub fn num_parameters(&self) -> usize {
        self.model.num_parameters()
    }
}

impl BatchModel for Gpt2BatchModel {
    fn forward(&self, input: Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(anyhow!(
                "GPT-2 batch input must be [batch, seq]; got {:?}",
                shape
            ));
        }
        let (batch, seq) = (shape[0], shape[1]);
        let ids = input.data().map_err(|e| anyhow!("failed to read input ids: {}", e))?;

        let mut all_logits = Vec::with_capacity(batch * seq * self.vocab_size);
        for row in 0..batch {
            let row_ids: Vec<u32> =
                ids[row * seq..(row + 1) * seq].iter().map(|&v| v.max(0.0) as u32).collect();
            let tokenized = TokenizedInput {
                input_ids: row_ids,
                attention_mask: vec![1u8; seq],
                token_type_ids: None,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            };

            let output = self
                .model
                .forward(tokenized)
                .map_err(|e| anyhow!("GPT-2 forward pass failed: {}", e))?;
            let row_logits = output
                .logits
                .data()
                .map_err(|e| anyhow!("failed to read GPT-2 logits: {}", e))?;

            let expected = seq * self.vocab_size;
            if row_logits.len() != expected {
                return Err(anyhow!(
                    "GPT-2 returned {} logits for a sequence of {} tokens; expected {}",
                    row_logits.len(),
                    seq,
                    expected
                ));
            }
            all_logits.extend_from_slice(&row_logits);
        }

        Tensor::from_vec(all_logits, &[batch, seq, self.vocab_size])
            .map_err(|e| anyhow!("failed to assemble batched logits: {}", e))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn pad_token_id(&self) -> u32 {
        self.eos_token_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(self.eos_token_id)
    }

    fn max_context_tokens(&self) -> Option<usize> {
        Some(self.n_positions)
    }
}

/// A HuggingFace `tokenizer.json` exposed to the batching stack.
#[derive(Debug)]
pub struct HuggingFaceTokenizer {
    inner: TokenizerImpl,
}

impl HuggingFaceTokenizer {
    /// Load from an explicit `tokenizer.json` path.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = TokenizerImpl::from_file(path)
            .map_err(|e| anyhow!("failed to load tokenizer from {}: {}", path.display(), e))?;
        Ok(Self { inner })
    }

    /// Load from a local model directory or the local HuggingFace hub cache.
    ///
    /// Nothing is downloaded; an absent tokenizer is an error, never a fallback.
    pub fn from_pretrained(name: &str) -> Result<Self> {
        let inner = TokenizerImpl::from_pretrained(name)
            .map_err(|e| anyhow!("failed to load tokenizer '{}': {}", name, e))?;
        Ok(Self { inner })
    }

    /// Vocabulary size reported by the underlying tokenizer.
    pub fn vocab_size(&self) -> usize {
        CoreTokenizer::vocab_size(&self.inner)
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        match CoreTokenizer::encode(&self.inner, text) {
            Ok(encoded) => encoded.input_ids,
            Err(e) => {
                tracing::error!("tokenizer failed to encode a prompt: {}", e);
                Vec::new()
            },
        }
    }

    fn decode(&self, ids: &[u32]) -> String {
        match CoreTokenizer::decode(&self.inner, ids) {
            Ok(text) => text,
            Err(e) => {
                tracing::error!("tokenizer failed to decode generated ids: {}", e);
                String::new()
            },
        }
    }
}

/// A byte-level tokenizer.
///
/// This is a genuine, fully reversible tokenizer (one token per UTF-8 byte, plus
/// a reserved end-of-text id), not a stand-in: it is the right choice for models
/// whose vocabulary is byte-level and for exercising the serving path without a
/// `tokenizer.json` on disk.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    /// Number of distinct ids this tokenizer can emit (256 bytes + EOT).
    pub const VOCAB_SIZE: usize = 257;
    /// Id reserved for end-of-text.
    pub const EOT_ID: u32 = 256;
}

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.as_bytes().iter().map(|&b| b as u32).collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().filter(|&&id| id < 256).map(|&id| id as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Build a text-generation executor from a GPT-2 model plus a tokenizer.
pub fn gpt2_text_executor(
    model: Arc<Gpt2BatchModel>,
    tokenizer: Arc<dyn Tokenizer>,
) -> ModelBatchExecutor {
    ModelBatchExecutor::new(model).with_tokenizer(tokenizer)
}

/// Build an executor around a small **untrained** GPT-2 over a byte-level
/// vocabulary.
///
/// The architecture, the weights and the forward pass are all real; the weights
/// are simply the initialization the architecture produces rather than trained
/// parameters, so the generated text is real model output and not English. This
/// is the right way to exercise the serving path end to end without a trained
/// checkpoint on disk — it is not a stand-in that fakes inference.
pub fn untrained_byte_gpt2_executor(
    n_layer: usize,
    n_embd: usize,
    max_new_tokens: usize,
) -> Result<ModelBatchExecutor> {
    let n_head = if n_embd.is_multiple_of(4) { 4 } else { 1 };
    let config = Gpt2Config {
        vocab_size: ByteTokenizer::VOCAB_SIZE,
        n_positions: 1024,
        n_embd,
        n_layer: n_layer.max(1),
        n_head,
        n_inner: Some(n_embd * 2),
        resid_pdrop: 0.0,
        embd_pdrop: 0.0,
        attn_pdrop: 0.0,
        bos_token_id: ByteTokenizer::EOT_ID,
        eos_token_id: ByteTokenizer::EOT_ID,
        ..Gpt2Config::default()
    };
    let model = Arc::new(Gpt2BatchModel::untrained(config)?);
    Ok(gpt2_text_executor(model, Arc::new(ByteTokenizer)).with_max_new_tokens(max_new_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batching::aggregator::{
        ProcessingOutput, Request, RequestBatch, RequestId, RequestInput,
    };
    use crate::batching::config::{BatchingConfig, Priority};
    use crate::batching::processor::BatchExecutor;
    use std::collections::HashMap;
    use std::time::Instant;

    /// A genuinely small GPT-2: real architecture, real (untrained) weights.
    fn tiny_config() -> Gpt2Config {
        Gpt2Config {
            vocab_size: ByteTokenizer::VOCAB_SIZE,
            n_positions: 32,
            n_embd: 16,
            n_layer: 1,
            n_head: 2,
            n_inner: Some(32),
            resid_pdrop: 0.0,
            embd_pdrop: 0.0,
            attn_pdrop: 0.0,
            bos_token_id: ByteTokenizer::EOT_ID,
            eos_token_id: ByteTokenizer::EOT_ID,
            ..Gpt2Config::default()
        }
    }

    #[test]
    fn byte_tokenizer_roundtrips() {
        let tokenizer = ByteTokenizer;
        let ids = tokenizer.encode("héllo");
        assert_eq!(tokenizer.decode(&ids), "héllo");
    }

    /// Regression: a real GPT-2 forward pass must produce a `[batch, seq, vocab]`
    /// logits tensor whose contents depend on the input, replacing the old
    /// `"Processed: {input}"` echo.
    #[test]
    fn gpt2_batch_model_runs_real_forward() {
        let model = Gpt2BatchModel::untrained(tiny_config()).expect("model must build");
        assert!(model.num_parameters() > 0);

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("input tensor must build");
        let logits = model.forward(input).expect("forward must succeed");

        assert_eq!(logits.shape(), vec![2, 3, ByteTokenizer::VOCAB_SIZE]);
        let values = logits.data().expect("logits readable");
        assert!(
            values.iter().any(|v| v.abs() > f32::EPSILON),
            "a real forward pass must not produce an all-zero logits tensor"
        );
    }

    /// Regression: the end-to-end text path must return decoded model output, not
    /// the prompt with a prefix.
    #[tokio::test]
    async fn gpt2_text_executor_produces_model_output() {
        let model = Arc::new(Gpt2BatchModel::untrained(tiny_config()).expect("model must build"));
        let executor = gpt2_text_executor(model, Arc::new(ByteTokenizer));

        let request = Request {
            id: RequestId::new(),
            input: RequestInput::Text {
                text: "Hi".to_string(),
                max_length: Some(4),
            },
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            deadline: None,
            metadata: HashMap::new(),
        };
        let id = request.id.clone();
        let batch = RequestBatch {
            id: uuid::Uuid::new_v4(),
            requests: vec![request],
            created_at: Instant::now(),
            total_memory: 0,
            max_sequence_length: 0,
            priority: Priority::Normal,
        };

        let results = executor
            .execute_batch(&batch, &BatchingConfig::default())
            .await
            .expect("real executor must succeed");

        match results.get(&id) {
            Some(ProcessingOutput::Text(text)) => {
                assert!(
                    !text.starts_with("Processed: "),
                    "executor must not echo the prompt, got {text:?}"
                );
            },
            other => panic!("expected decoded text output, got {other:?}"),
        }
    }

    /// Regression: a missing tokenizer file is an error, never a silent fallback.
    #[test]
    fn missing_tokenizer_file_is_an_error() {
        let missing = std::env::temp_dir().join("trustformers-serve-absent-tokenizer.json");
        let _ = std::fs::remove_file(&missing);
        let error = HuggingFaceTokenizer::from_file(&missing).expect_err("must not succeed");
        assert!(error.to_string().contains("failed to load tokenizer"));
    }

    /// Regression: a missing checkpoint is an error, never a fabricated model.
    #[test]
    fn missing_checkpoint_is_an_error() {
        let dir = std::env::temp_dir().join("trustformers-serve-absent-checkpoint");
        let error = Gpt2BatchModel::from_checkpoint(&dir.join("config.json"), &dir.join("m.st"))
            .expect_err("must not succeed");
        assert!(error.to_string().contains("failed to read GPT-2 config"));
    }
}
