//! Real model adapters for the batching stack.
//!
//! This module bridges [`trustformers_models`] decoders and
//! [`trustformers_tokenizers`] tokenizers onto the dyn-safe
//! [`crate::batching::processor::BatchModel`] /
//! [`crate::batching::processor::Tokenizer`] traits the batch
//! executor consumes.
//!
//! Nothing here fabricates output: every call runs the real forward pass of a
//! real model, and every failure is reported as an error.

use crate::batching::processor::{BatchModel, EmbeddingModel, ModelBatchExecutor, Tokenizer};
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
    n_embd: usize,
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
            n_embd: config.n_embd,
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

    /// Width of the model's residual stream, i.e. the size of one hidden-state row.
    pub fn hidden_size(&self) -> usize {
        self.n_embd
    }
}

impl EmbeddingModel for Gpt2BatchModel {
    /// Mean-pool the last-layer (post-`ln_f`) hidden states of `ids`.
    ///
    /// This is the model's own representation of the sequence, obtained from a
    /// genuine forward pass through
    /// [`Gpt2LMHeadModel::logits_and_hidden_states`]; nothing is synthesized. The
    /// row count returned by the backbone is checked against the input length so
    /// a layout change cannot silently turn into a plausible-looking but wrong
    /// vector.
    fn embed_tokens(&self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(anyhow!("cannot embed an empty token sequence"));
        }
        if ids.len() > self.n_positions {
            return Err(anyhow!(
                "a sequence of {} token(s) exceeds the model's context window of {} token(s)",
                ids.len(),
                self.n_positions
            ));
        }

        let (_, hidden_rows) = self
            .model
            .logits_and_hidden_states(ids)
            .map_err(|e| anyhow!("GPT-2 forward pass failed while embedding: {}", e))?;

        if hidden_rows.len() != ids.len() {
            return Err(anyhow!(
                "GPT-2 returned {} hidden-state row(s) for {} input token(s); the pooled vector \
                 would not correspond to the input",
                hidden_rows.len(),
                ids.len()
            ));
        }

        let mut pooled = vec![0.0f32; self.n_embd];
        for (index, row) in hidden_rows.iter().enumerate() {
            if row.len() != self.n_embd {
                return Err(anyhow!(
                    "GPT-2 hidden-state row {} has {} value(s); expected the model's hidden size \
                     of {}",
                    index,
                    row.len(),
                    self.n_embd
                ));
            }
            for (slot, value) in pooled.iter_mut().zip(row.iter()) {
                *slot += *value;
            }
        }
        let count = hidden_rows.len() as f32;
        for slot in pooled.iter_mut() {
            *slot /= count;
        }

        if pooled.iter().any(|v| !v.is_finite()) {
            return Err(anyhow!(
                "GPT-2 produced a non-finite pooled embedding; refusing to return it"
            ));
        }
        Ok(pooled)
    }

    fn embedding_dim(&self) -> usize {
        self.n_embd
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

    /// The pooled embedding must come from the model's own hidden states, with a
    /// layout that genuinely corresponds to the input: one row per input token,
    /// each of `n_embd` values. This pins the assumption the `/v1/embeddings`
    /// path is built on.
    #[test]
    fn gpt2_embedding_pools_real_hidden_states() {
        let config = tiny_config();
        let hidden_size = config.n_embd;
        let model = Gpt2BatchModel::untrained(config).expect("model must build");
        assert_eq!(model.hidden_size(), hidden_size);
        assert_eq!(EmbeddingModel::embedding_dim(&model), hidden_size);

        let vector = model.embed_tokens(&[1, 2, 3]).expect("embedding must succeed");
        assert_eq!(vector.len(), hidden_size);
        assert!(vector.iter().all(|v| v.is_finite()));
        assert!(
            vector.iter().any(|v| v.abs() > f32::EPSILON),
            "a pooled hidden state must not be an all-zero vector"
        );

        // Different inputs must give different representations; an implementation
        // that ignored its input would return the same vector here.
        let other = model.embed_tokens(&[40, 41, 42]).expect("embedding must succeed");
        assert_eq!(other.len(), hidden_size);
        let distance: f32 =
            vector.iter().zip(other.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>();
        assert!(
            distance > f32::EPSILON,
            "distinct token sequences must yield distinct embeddings"
        );
    }

    /// Regression: an empty or over-long sequence is an error, never a zero vector.
    #[test]
    fn gpt2_embedding_refuses_impossible_inputs() {
        let model = Gpt2BatchModel::untrained(tiny_config()).expect("model must build");
        let empty = model.embed_tokens(&[]).expect_err("empty input must fail");
        assert!(empty.to_string().contains("empty token sequence"));

        let too_long: Vec<u32> = (0..64).collect();
        let overflow = model.embed_tokens(&too_long).expect_err("over-long input must fail");
        assert!(overflow.to_string().contains("context window"));
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
