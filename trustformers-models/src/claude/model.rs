use crate::claude::config::ClaudeConfig;
use crate::weight_loading::binding::{
    bind_embedding, bind_linear, take_norm_weight, DECODER_BUFFER_SUFFIXES,
};
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, UnusedTensors, WeightBinder};
use std::collections::HashMap;
use trustformers_core::{
    errors::{Result, TrustformersError},
    layers::{Embedding, LayerNorm, Linear},
    ops::activations::silu,
    tensor::Tensor,
    traits::{Layer, Model},
};

/// Bind a `[dim]` LayerNorm scale, and its `[dim]` shift when the checkpoint
/// carries one.
///
/// A Claude-family export produced from an RMSNorm holds only `<name>.weight`;
/// one produced from a full LayerNorm holds `<name>.bias` as well. The bias is
/// therefore bound when present and otherwise left at its constructor value,
/// which is exactly zero — the identity shift — rather than an invented vector.
/// The scale is always requested, so a checkpoint that omits it is reported as
/// a missing parameter by [`WeightBinder::finish`] instead of passing silently.
fn bind_layer_norm(
    binder: &mut WeightBinder<'_>,
    name: &str,
    dim: usize,
    norm: &mut LayerNorm,
) -> Result<()> {
    if let Some(weight) = take_norm_weight(binder, name, dim)? {
        norm.set_weight(weight)?;
    }
    let bias_name = format!("{name}.bias");
    if binder.has(&bias_name) {
        if let Some(bias) = binder.take_shaped(&bias_name, &[dim])? {
            norm.set_bias(bias)?;
        }
    }
    Ok(())
}

/// Claude-specific attention mechanism with Constitutional AI principles
pub struct ClaudeAttention {
    config: ClaudeConfig,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    attention_dropout: f32,
    scale: f32,
}

impl ClaudeAttention {
    pub fn new(config: ClaudeConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        let q_proj = Linear::new(hidden_size, num_heads * head_dim, false);
        let k_proj = Linear::new(hidden_size, num_kv_heads * head_dim, false);
        let v_proj = Linear::new(hidden_size, num_kv_heads * head_dim, false);
        let o_proj = Linear::new(num_heads * head_dim, hidden_size, false);

        let rotary_emb =
            RotaryEmbedding::new(head_dim, config.max_position_embeddings, config.rope_theta);

        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            attention_dropout: config.attention_dropout,
            config,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            scale,
        })
    }
}

impl Layer for ClaudeAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, hidden_states: Self::Input) -> Result<Self::Output> {
        let seq_len = hidden_states.shape()[1];
        let batch_size = hidden_states.shape()[0];

        // Project to query, key, value
        let query_states = self.q_proj.forward(hidden_states.clone())?;
        let key_states = self.k_proj.forward(hidden_states.clone())?;
        let value_states = self.v_proj.forward(hidden_states)?;

        // Reshape for multi-head attention
        let query_states = query_states.reshape(&[
            batch_size,
            seq_len,
            self.config.num_attention_heads,
            self.config.head_dim(),
        ])?;
        let key_states = key_states.reshape(&[
            batch_size,
            seq_len,
            self.config.num_kv_heads(),
            self.config.head_dim(),
        ])?;
        let value_states = value_states.reshape(&[
            batch_size,
            seq_len,
            self.config.num_kv_heads(),
            self.config.head_dim(),
        ])?;

        // Apply rotary position embedding
        let position_ids: Vec<usize> = (0..seq_len).collect();
        let (query_states, key_states) =
            self.rotary_emb.apply_rotary_emb(&query_states, &key_states, &position_ids)?;

        // Compute attention scores
        let attn_weights = query_states.matmul(&key_states.transpose(2, 3)?)?;
        let attn_weights = attn_weights.mul_scalar(self.scale)?;

        // Apply causal mask
        let causal_mask = create_causal_mask(seq_len)?;
        // Manually apply masking by adding negative infinity where mask is true
        let mask_value = Tensor::from_vec(vec![f32::NEG_INFINITY], &[1])?;
        let attn_weights = attn_weights.add(&causal_mask.mul(&mask_value)?)?;

        // Apply softmax
        let attn_weights = attn_weights.softmax(3)?;

        // Apply dropout if training
        let attn_weights = if self.attention_dropout > 0.0 {
            attn_weights.dropout(self.attention_dropout)?
        } else {
            attn_weights
        };

        // Apply attention to values
        let attn_output = attn_weights.matmul(&value_states)?;

        // Reshape and project output
        let attn_output = attn_output.reshape(&[batch_size, seq_len, self.config.hidden_size])?;
        let attn_output = self.o_proj.forward(attn_output)?;

        Ok(attn_output)
    }
}

impl ClaudeAttention {
    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.o_proj.parameter_count()
    }
}

/// Claude-specific MLP with SwiGLU activation
pub struct ClaudeMLP {
    #[allow(dead_code)]
    config: ClaudeConfig,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    dropout: f32,
}

impl ClaudeMLP {
    pub fn new(config: ClaudeConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let gate_proj = Linear::new(hidden_size, intermediate_size, false);
        let up_proj = Linear::new(hidden_size, intermediate_size, false);
        let down_proj = Linear::new(intermediate_size, hidden_size, false);

        Ok(Self {
            dropout: config.ffn_dropout,
            config,
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Layer for ClaudeMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, hidden_states: Self::Input) -> Result<Self::Output> {
        let gate_output = self.gate_proj.forward(hidden_states.clone())?;
        let up_output = self.up_proj.forward(hidden_states)?;

        // Apply SwiGLU activation
        let gate_output = silu(&gate_output)?;
        let intermediate = gate_output.mul(&up_output)?;

        // Apply dropout if training
        let intermediate = if self.dropout > 0.0 {
            intermediate.dropout(self.dropout)?
        } else {
            intermediate
        };

        let output = self.down_proj.forward(intermediate)?;
        Ok(output)
    }
}

impl ClaudeMLP {
    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
    }
}

/// Claude decoder layer with Constitutional AI enhancements
pub struct ClaudeDecoderLayer {
    #[allow(dead_code)]
    config: ClaudeConfig,
    self_attn: ClaudeAttention,
    mlp: ClaudeMLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    #[allow(dead_code)]
    constitutional_ai: bool,
}

impl ClaudeDecoderLayer {
    pub fn new(config: ClaudeConfig) -> Result<Self> {
        let self_attn = ClaudeAttention::new(config.clone())?;
        let mlp = ClaudeMLP::new(config.clone())?;
        let input_layernorm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;
        let post_attention_layernorm =
            LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        Ok(Self {
            constitutional_ai: config.constitutional_ai,
            config,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

impl Layer for ClaudeDecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, hidden_states: Self::Input) -> Result<Self::Output> {
        let residual = hidden_states.clone();

        // Self-attention with pre-norm
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(hidden_states)?;
        let hidden_states = residual.add(&attn_output)?;

        let residual = hidden_states.clone();

        // MLP with pre-norm
        let hidden_states = self.post_attention_layernorm.forward(hidden_states)?;
        let mlp_output = self.mlp.forward(hidden_states)?;
        let hidden_states = residual.add(&mlp_output)?;

        Ok(hidden_states)
    }
}

impl ClaudeDecoderLayer {
    pub fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.mlp.parameter_count()
            + self.input_layernorm.parameter_count()
            + self.post_attention_layernorm.parameter_count()
    }
}

/// Main Claude model
pub struct ClaudeModel {
    config: ClaudeConfig,
    embed_tokens: Embedding,
    layers: Vec<ClaudeDecoderLayer>,
    norm: LayerNorm,
    constitutional_weights: Option<HashMap<String, f32>>,
}

impl ClaudeModel {
    pub fn new(config: ClaudeConfig) -> Result<Self> {
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(ClaudeDecoderLayer::new(config.clone())?);
        }

        let norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        let constitutional_weights = if config.constitutional_ai {
            let mut weights = HashMap::new();
            weights.insert("harmlessness".to_string(), config.harmlessness_weight);
            weights.insert("helpfulness".to_string(), config.helpfulness_weight);
            weights.insert("honesty".to_string(), config.honesty_weight);
            Some(weights)
        } else {
            None
        };

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            constitutional_weights,
        })
    }

    /// Apply Constitutional AI principles to the output
    pub fn apply_constitutional_ai(&self, hidden_states: &Tensor) -> Result<Tensor> {
        if let Some(weights) = &self.constitutional_weights {
            // Apply constitutional AI weighting
            // This is a simplified implementation - in practice, this would involve
            // more sophisticated constitutional AI techniques
            let mut result = hidden_states.clone();

            // Apply harmlessness constraint
            if let Some(&harmlessness_weight) = weights.get("harmlessness") {
                result = result.mul_scalar(harmlessness_weight)?;
            }

            // Apply helpfulness boost
            if let Some(&helpfulness_weight) = weights.get("helpfulness") {
                result = result.mul_scalar(helpfulness_weight)?;
            }

            // Apply honesty normalization
            if let Some(&honesty_weight) = weights.get("honesty") {
                result = result.mul_scalar(honesty_weight)?;
            }

            Ok(result)
        } else {
            Ok(hidden_states.clone())
        }
    }
}

impl ClaudeModel {
    /// Checkpoint namespaces the backbone legitimately does not consume.
    ///
    /// The language-model head lives on [`ClaudeForCausalLM`], not here, so a
    /// causal-LM export carries an `lm_head.` entry the backbone must be allowed
    /// to leave behind. Everything else the checkpoint holds must be recognised.
    pub const ALLOWED_UNUSED_PREFIXES: &'static [&'static str] = &["lm_head."];

    /// Bind an already-parsed checkpoint into this backbone.
    ///
    /// Claude's decoder is laid out like the HuggingFace LLaMA export it is
    /// modelled on: the backbone nests under `model.` (the bare layout is
    /// accepted too), attention projections are `self_attn.{q,k,v,o}_proj`,
    /// the gated feed-forward is `mlp.{gate,up,down}_proj`, and the two
    /// per-layer norms are `input_layernorm` / `post_attention_layernorm`.
    /// Grouped-query attention narrows `k_proj`/`v_proj` to
    /// `num_kv_heads * head_dim`, which is checked rather than assumed.
    ///
    /// A parameter the checkpoint does not carry is *recorded* and reported by
    /// [`WeightBinder::finish`], never substituted, so a mismatched checkpoint
    /// cannot leave constructor-initialised tensors in place while the load
    /// returns `Ok`.
    ///
    /// # Errors
    ///
    /// Fails when the stream is not a checkpoint container, when the checkpoint
    /// does not look like a Claude checkpoint, when any tensor has the wrong
    /// shape, when a parameter is missing, or when the checkpoint carries
    /// weights this architecture does not recognise.
    pub fn load_checkpoint(
        &mut self,
        checkpoint: &Checkpoint,
        allowed_unused_prefixes: &[&str],
    ) -> Result<LoadReport> {
        let prefix = checkpoint.detect_prefix(&["model.", ""], "embed_tokens.weight")?;
        let mut binder = checkpoint.binder(&prefix);

        let hidden = self.config.hidden_size;
        let head_dim = self.config.head_dim();
        let q_width = self.config.num_attention_heads * head_dim;
        let kv_width = self.config.num_kv_heads() * head_dim;
        let intermediate = self.config.intermediate_size;

        bind_embedding(
            &mut binder,
            "embed_tokens",
            self.config.vocab_size,
            hidden,
            &mut self.embed_tokens,
        )?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let attn = format!("layers.{i}.self_attn");
            bind_linear(
                &mut binder,
                &format!("{attn}.q_proj"),
                q_width,
                hidden,
                false,
                &mut layer.self_attn.q_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{attn}.k_proj"),
                kv_width,
                hidden,
                false,
                &mut layer.self_attn.k_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{attn}.v_proj"),
                kv_width,
                hidden,
                false,
                &mut layer.self_attn.v_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{attn}.o_proj"),
                hidden,
                q_width,
                false,
                &mut layer.self_attn.o_proj,
            )?;

            let mlp = format!("layers.{i}.mlp");
            bind_linear(
                &mut binder,
                &format!("{mlp}.gate_proj"),
                intermediate,
                hidden,
                false,
                &mut layer.mlp.gate_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{mlp}.up_proj"),
                intermediate,
                hidden,
                false,
                &mut layer.mlp.up_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{mlp}.down_proj"),
                hidden,
                intermediate,
                false,
                &mut layer.mlp.down_proj,
            )?;

            bind_layer_norm(
                &mut binder,
                &format!("layers.{i}.input_layernorm"),
                hidden,
                &mut layer.input_layernorm,
            )?;
            bind_layer_norm(
                &mut binder,
                &format!("layers.{i}.post_attention_layernorm"),
                hidden,
                &mut layer.post_attention_layernorm,
            )?;
        }

        bind_layer_norm(&mut binder, "norm", hidden, &mut self.norm)?;

        binder.finish(UnusedTensors::new(
            allowed_unused_prefixes,
            DECODER_BUFFER_SUFFIXES,
        ))
    }
}

impl Model for ClaudeModel {
    type Config = ClaudeConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        // Convert input_ids tensor to Vec<u32> for embedding layer
        let input_ids_vec: Vec<u32> =
            input_ids.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids_vec)?;

        // Pass through all decoder layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        // Apply final layer norm
        hidden_states = self.norm.forward(hidden_states)?;

        // Apply Constitutional AI if enabled
        hidden_states = self.apply_constitutional_ai(&hidden_states)?;

        Ok(hidden_states)
    }
    /// Load a Claude-family checkpoint (safetensors or `torch.save`) into the
    /// backbone.
    ///
    /// A previous revision read the stream into a buffer, compared its *length*
    /// against a set of size estimates and returned `Ok(())` under the comment
    /// "Success: weights are loaded and validated" — while binding nothing at
    /// all. Every load left the model at its constructor initialisation and
    /// reported success. That is gone: the container is parsed for real and
    /// every parameter is either filled from the checkpoint or named in the
    /// error.
    ///
    /// # Errors
    ///
    /// See [`ClaudeModel::load_checkpoint`].
    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> Result<()> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_checkpoint(&checkpoint, Self::ALLOWED_UNUSED_PREFIXES)?;
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let embed_params = self.embed_tokens.parameter_count();
        let layers_params: usize = self.layers.iter().map(|layer| layer.parameter_count()).sum();
        let norm_params = self.norm.parameter_count();

        embed_params + layers_params + norm_params
    }
}

/// Claude for causal language modeling
pub struct ClaudeForCausalLM {
    model: ClaudeModel,
    lm_head: Linear,
    config: ClaudeConfig,
}

impl ClaudeForCausalLM {
    pub fn new(config: ClaudeConfig) -> Result<Self> {
        let model = ClaudeModel::new(config.clone())?;
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);

        Ok(Self {
            model,
            lm_head,
            config,
        })
    }

    /// Generate text with Constitutional AI constraints
    pub fn generate_with_constitutional_ai(
        &self,
        input_ids: Tensor,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Tensor> {
        // This is a simplified generation implementation
        // In practice, this would involve more sophisticated generation strategies
        let mut current_ids = input_ids;

        for _ in 0..max_new_tokens {
            let hidden_states = self.model.forward(current_ids.clone())?;
            let logits = self.lm_head.forward(hidden_states)?;

            // Apply temperature and top-p sampling
            let logits = logits.div_scalar(temperature)?;
            let probs = logits.softmax(-1)?;

            // Sample next token (simplified)
            let next_token = sample_from_distribution(&probs, top_p)?;

            // Append to sequence
            current_ids = Tensor::concat(&[current_ids, next_token], 0)?;
        }

        Ok(current_ids)
    }
}

impl Model for ClaudeForCausalLM {
    type Config = ClaudeConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.model.forward(input_ids)?;
        let logits = self.lm_head.forward(hidden_states)?;
        Ok(logits)
    }
    /// Load a Claude-family causal-LM checkpoint (safetensors or `torch.save`).
    ///
    /// The backbone is bound first, then the LM head. A checkpoint exported with
    /// tied word embeddings carries no `lm_head.weight`; the input embedding
    /// matrix is reused in that case, which is what the tied configuration
    /// means — not a fallback to something invented.
    ///
    /// Like [`ClaudeModel`]'s loader, this replaces a revision that inspected
    /// the byte length of the stream and returned `Ok(())` without binding a
    /// single tensor.
    ///
    /// # Errors
    ///
    /// See [`ClaudeModel::load_checkpoint`]; additionally fails when the
    /// checkpoint holds neither an LM head nor an embedding matrix to tie it to,
    /// or when the head has the wrong shape.
    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> Result<()> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.model.load_checkpoint(&checkpoint, &["lm_head."])?;

        let expected = [self.config.vocab_size, self.config.hidden_size];
        let head = match checkpoint.take_shaped("lm_head.weight", &expected)? {
            Some(weight) => weight,
            None => {
                let embed_name = if checkpoint.contains("model.embed_tokens.weight") {
                    "model.embed_tokens.weight"
                } else {
                    "embed_tokens.weight"
                };
                checkpoint.take_shaped(embed_name, &expected)?.ok_or_else(|| {
                    TrustformersError::weight_load_error(
                        "checkpoint holds neither lm_head.weight nor an embedding matrix to \
                             tie it to"
                            .to_string(),
                    )
                })?
            },
        };
        self.lm_head.set_weight(head)?;
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.model.num_parameters() + self.lm_head.parameter_count()
    }
}

/// Rotary Position Embedding implementation
pub struct RotaryEmbedding {
    #[allow(dead_code)]
    dim: usize,
    #[allow(dead_code)]
    max_seq_len: usize,
    #[allow(dead_code)]
    base: f32,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32) -> Self {
        Self {
            dim,
            max_seq_len,
            base,
        }
    }

    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        _position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        // Simplified RoPE implementation
        // In practice, this would involve proper complex number rotations
        Ok((q.clone(), k.clone()))
    }
}

// Helper functions

fn create_causal_mask(seq_len: usize) -> Result<Tensor> {
    // Create a causal mask for attention
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = 1.0; // true positions become 1.0
        }
    }

    // Convert to tensor
    Tensor::from_vec(mask_data, &[seq_len, seq_len])
}

fn sample_from_distribution(_probs: &Tensor, _top_p: f32) -> Result<Tensor> {
    // Simplified sampling implementation
    // In practice, this would involve proper probability sampling
    Tensor::zeros(&[1, 1])
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claude::config::ClaudeConfig;
    use trustformers_core::{
        tensor::Tensor,
        traits::{Config, Model},
    };

    // ── Config tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_default_config_vocab_size() {
        let cfg = ClaudeConfig::default();
        assert_eq!(
            cfg.vocab_size, 100352,
            "Claude default vocab_size should be 100352"
        );
    }

    #[test]
    fn test_default_config_constitutional_ai_enabled() {
        let cfg = ClaudeConfig::default();
        assert!(
            cfg.constitutional_ai,
            "Constitutional AI should be enabled by default"
        );
    }

    #[test]
    fn test_default_harmlessness_weight() {
        let cfg = ClaudeConfig::default();
        assert!(
            (cfg.harmlessness_weight - 1.0).abs() < 1e-4,
            "default harmlessness_weight = 1.0"
        );
    }

    #[test]
    fn test_default_helpfulness_weight() {
        let cfg = ClaudeConfig::default();
        assert!(
            (cfg.helpfulness_weight - 1.0).abs() < 1e-4,
            "default helpfulness_weight = 1.0"
        );
    }

    #[test]
    fn test_default_honesty_weight() {
        let cfg = ClaudeConfig::default();
        assert!(
            (cfg.honesty_weight - 1.0).abs() < 1e-4,
            "default honesty_weight = 1.0"
        );
    }

    #[test]
    fn test_small_test_config_valid() {
        ClaudeConfig::small_test_config()
            .validate()
            .expect("small_test_config should be valid");
    }

    #[test]
    fn test_config_validate_negative_weight_fails() {
        let mut cfg = ClaudeConfig::small_test_config();
        cfg.harmlessness_weight = -0.1;
        assert!(
            cfg.validate().is_err(),
            "negative constitutional weight should fail validation"
        );
    }

    #[test]
    fn test_head_dim_computation() {
        let cfg = ClaudeConfig::small_test_config();
        assert_eq!(
            cfg.head_dim(),
            cfg.hidden_size / cfg.num_attention_heads,
            "head_dim = hidden_size / num_attention_heads"
        );
    }

    #[test]
    fn test_num_kv_heads_defaults_to_num_attention_heads() {
        let mut cfg = ClaudeConfig::small_test_config();
        cfg.num_key_value_heads = None;
        assert_eq!(
            cfg.num_kv_heads(),
            cfg.num_attention_heads,
            "when num_kv_heads is None, num_kv_heads() == num_attention_heads"
        );
    }

    #[test]
    fn test_num_query_groups_with_gqa() {
        let cfg = ClaudeConfig::claude_3_haiku(); // has num_kv_heads = Some(8)
        let expected = cfg.num_attention_heads / cfg.num_kv_heads();
        assert_eq!(
            cfg.num_query_groups(),
            expected,
            "num_query_groups = heads / kv_heads"
        );
    }

    // ── Model construction tests ──────────────────────────────────────────────

    #[test]
    fn test_model_creation() {
        let cfg = ClaudeConfig::small_test_config();
        ClaudeModel::new(cfg).expect("ClaudeModel creation should succeed");
    }

    #[test]
    fn test_model_parameter_count_nonzero() {
        let cfg = ClaudeConfig::small_test_config();
        let model = ClaudeModel::new(cfg).expect("model creation should succeed");
        assert!(
            model.num_parameters() > 0,
            "model must have non-zero parameters"
        );
    }

    // ── Constitutional AI tests ───────────────────────────────────────────────

    #[test]
    fn test_constitutional_ai_apply_preserves_shape() {
        let cfg = ClaudeConfig::small_test_config();
        let model = ClaudeModel::new(cfg.clone()).expect("model creation should succeed");
        let hidden = Tensor::from_vec(vec![0.5_f32; cfg.hidden_size], &[cfg.hidden_size])
            .expect("tensor creation should succeed");
        let result = model
            .apply_constitutional_ai(&hidden)
            .expect("constitutional AI should succeed");
        assert_eq!(result.shape(), hidden.shape(), "shape must be preserved");
    }

    #[test]
    fn test_constitutional_ai_disabled_returns_unchanged() {
        let mut cfg = ClaudeConfig::small_test_config();
        cfg.constitutional_ai = false;
        let model = ClaudeModel::new(cfg.clone()).expect("model creation should succeed");
        let hidden = Tensor::from_vec(vec![1.0_f32; cfg.hidden_size], &[cfg.hidden_size])
            .expect("tensor creation should succeed");
        let result = model.apply_constitutional_ai(&hidden).expect("should succeed");
        let orig_vals = hidden.to_vec_f32().expect("to_vec_f32 should succeed");
        let result_vals = result.to_vec_f32().expect("to_vec_f32 should succeed");
        for (o, r) in orig_vals.iter().zip(result_vals.iter()) {
            assert!(
                (o - r).abs() < 1e-5,
                "disabled CAI must return input unchanged"
            );
        }
    }

    // ── Claude variant configs ────────────────────────────────────────────────

    #[test]
    fn test_claude_3_haiku_smaller_than_opus() {
        let haiku = ClaudeConfig::claude_3_haiku();
        let opus = ClaudeConfig::claude_3_opus();
        assert!(
            haiku.hidden_size < opus.hidden_size,
            "Haiku should have smaller hidden_size than Opus"
        );
    }

    #[test]
    fn test_from_pretrained_name_valid() {
        let cfg = ClaudeConfig::from_pretrained_name("claude-2");
        assert!(cfg.is_some(), "claude-2 should be a known pretrained name");
    }

    #[test]
    fn test_from_pretrained_name_unknown_returns_none() {
        let cfg = ClaudeConfig::from_pretrained_name("unknown-model-xyz");
        assert!(cfg.is_none(), "unknown model name should return None");
    }
}

#[cfg(test)]
mod loading_tests {
    use super::*;
    use crate::weight_loading::test_support::{build_safetensors, DecoderFixtureSpec, F32Tensor};

    /// A deliberately tiny Claude configuration: two layers, grouped-query
    /// attention (2 query heads over 1 KV head) so a loader that assumed square
    /// projections cannot pass.
    fn loading_config() -> ClaudeConfig {
        ClaudeConfig {
            vocab_size: 12,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            constitutional_ai: false,
            ..ClaudeConfig::default()
        }
    }

    /// The checkpoint a Claude backbone of `config`'s shape would be exported as.
    ///
    /// `norm_bias` is on because this implementation's per-layer and final norms
    /// are full [`LayerNorm`]s, so a real export of it carries `<norm>.bias`.
    fn loading_fixture(config: &ClaudeConfig) -> DecoderFixtureSpec {
        let head_dim = config.head_dim();
        let mut spec = DecoderFixtureSpec::llama_style(
            "model.",
            config.vocab_size,
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.num_attention_heads * head_dim,
            config.num_kv_heads() * head_dim,
        );
        spec.norm_bias = true;
        spec
    }

    fn fixture_values(tensors: &[F32Tensor], name: &str) -> Vec<f32> {
        tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("fixture must carry {name}"))
            .values
            .clone()
    }

    /// Regression: `load_pretrained` read the stream into a buffer, compared its
    /// *length* against a handful of size estimates and returned `Ok(())` under
    /// the comment "Success: weights are loaded and validated" — without binding
    /// a single tensor. The proof that it now loads for real is that the
    /// checkpoint's exact values arrive in the layers.
    #[test]
    fn load_pretrained_binds_every_parameter_from_the_checkpoint() {
        let config = loading_config();
        let tensors = loading_fixture(&config).tensors();
        let bytes = build_safetensors(&tensors);

        let mut model = ClaudeModel::new(config).expect("model must build");
        let before = model.embed_tokens.weight().data().expect("readable");

        model
            .load_pretrained(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");

        let embed = model.embed_tokens.weight().data().expect("readable");
        assert_ne!(
            embed, before,
            "the embedding matrix must actually change when a checkpoint is loaded"
        );
        assert_eq!(
            embed,
            fixture_values(&tensors, "model.embed_tokens.weight"),
            "the embedding matrix must hold the checkpoint's values"
        );
        assert_eq!(
            model.layers[0].self_attn.q_proj.weight().data().expect("readable"),
            fixture_values(&tensors, "model.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            model.layers[1].mlp.down_proj.weight().data().expect("readable"),
            fixture_values(&tensors, "model.layers.1.mlp.down_proj.weight")
        );
        assert_eq!(
            model.norm.weight().data().expect("readable"),
            fixture_values(&tensors, "model.norm.weight")
        );
        assert_eq!(
            model.norm.bias().data().expect("readable"),
            fixture_values(&tensors, "model.norm.bias"),
            "the LayerNorm shift must be bound too, not left at zero"
        );
    }

    /// An export produced from an RMS norm carries no `<norm>.bias`. That is a
    /// legitimate checkpoint, not a gap: the zero-initialised shift is the
    /// identity, so nothing is invented by leaving it alone.
    #[test]
    fn load_pretrained_accepts_a_checkpoint_whose_norms_carry_no_bias() {
        let config = loading_config();
        let mut spec = loading_fixture(&config);
        spec.norm_bias = false;
        let bytes = spec.safetensors();

        let mut model = ClaudeModel::new(config).expect("model must build");
        model
            .load_pretrained(&mut bytes.as_slice())
            .expect("an RMS-norm-shaped export must load");
        assert_eq!(
            model.norm.bias().data().expect("readable"),
            vec![0.0f32; 8],
            "an absent shift must stay at the identity rather than being invented"
        );
    }

    /// Grouped-query attention: `k_proj`/`v_proj` are narrower than `q_proj`.
    #[test]
    fn load_pretrained_respects_grouped_query_attention_widths() {
        let config = loading_config();
        let head_dim = config.head_dim();
        let bytes = loading_fixture(&config).safetensors();

        let mut model = ClaudeModel::new(config.clone()).expect("model must build");
        model.load_pretrained(&mut bytes.as_slice()).expect("checkpoint must load");

        assert_eq!(
            model.layers[0].self_attn.k_proj.weight().shape(),
            vec![config.num_kv_heads() * head_dim, config.hidden_size]
        );
        assert_eq!(
            model.layers[0].self_attn.q_proj.weight().shape(),
            vec![config.num_attention_heads * head_dim, config.hidden_size]
        );
    }

    #[test]
    fn load_pretrained_reports_a_missing_parameter_instead_of_inventing_it() {
        let config = loading_config();
        let mut tensors = loading_fixture(&config).tensors();
        tensors.retain(|t| t.name != "model.layers.1.mlp.up_proj.weight");
        let bytes = build_safetensors(&tensors);

        let mut model = ClaudeModel::new(config).expect("model must build");
        let before = model.layers[1].mlp.up_proj.weight().data().expect("readable");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an incomplete checkpoint must not load silently");
        assert!(
            err.to_string().contains("layers.1.mlp.up_proj.weight"),
            "the error must name the gap: {err}"
        );
        assert_eq!(
            model.layers[1].mlp.up_proj.weight().data().expect("readable"),
            before,
            "an absent tensor must leave the parameter untouched"
        );
    }

    #[test]
    fn load_pretrained_rejects_a_foreign_tensor() {
        let config = loading_config();
        let mut tensors = loading_fixture(&config).tensors();
        tensors.push(F32Tensor::ramp(
            "model.layers.9.mystery.weight",
            &[4, 4],
            99.0,
        ));
        let bytes = build_safetensors(&tensors);

        let mut model = ClaudeModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an unrecognised weight must fail the load");
        assert!(
            err.to_string().contains("mystery.weight"),
            "unexpected: {err}"
        );
    }

    /// Regression: the old loader accepted *any* buffer that cleared its size
    /// thresholds. 4 KiB of `0xAB` cleared every one of them for this config and
    /// produced `Ok(())`; it is not a checkpoint container at all.
    #[test]
    fn load_pretrained_rejects_bytes_that_are_not_a_checkpoint() {
        let mut model = ClaudeModel::new(loading_config()).expect("model must build");
        let garbage = vec![0xABu8; 4096];
        let err = model
            .load_pretrained(&mut garbage.as_slice())
            .expect_err("garbage must not be accepted as weights");
        assert!(
            err.to_string().contains("unrecognised checkpoint container"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn load_pretrained_rejects_a_checkpoint_for_a_different_configuration() {
        let config = loading_config();
        let wider = ClaudeConfig {
            hidden_size: config.hidden_size * 2,
            intermediate_size: config.intermediate_size * 2,
            ..config.clone()
        };
        let bytes = loading_fixture(&wider).safetensors();

        let mut model = ClaudeModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a mismatched checkpoint must not be reshaped into place");
        assert!(err.to_string().contains("expects"), "unexpected: {err}");
    }

    /// Regression: `ClaudeForCausalLM::load_pretrained` carried the same
    /// size-heuristic stub, so the LM head was never filled either.
    #[test]
    fn causal_lm_load_pretrained_binds_the_language_model_head() {
        let config = loading_config();
        let mut spec = loading_fixture(&config);
        spec.include_lm_head = true;
        let tensors = spec.tensors();
        let bytes = build_safetensors(&tensors);

        let mut model = ClaudeForCausalLM::new(config).expect("model must build");
        model
            .load_pretrained(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");

        assert_eq!(
            model.lm_head.weight().data().expect("readable"),
            fixture_values(&tensors, "lm_head.weight"),
            "the LM head must hold the checkpoint's values"
        );
        assert_eq!(
            model.model.embed_tokens.weight().data().expect("readable"),
            fixture_values(&tensors, "model.embed_tokens.weight"),
            "the backbone must be bound as well as the head"
        );
    }

    /// A tied-embedding export carries no `lm_head.weight`; reusing the input
    /// embedding matrix is what "tied" means, not a fallback to an invention.
    #[test]
    fn causal_lm_ties_the_head_to_the_embeddings_when_the_checkpoint_omits_it() {
        let config = loading_config();
        let tensors = loading_fixture(&config).tensors();
        let bytes = build_safetensors(&tensors);

        let mut model = ClaudeForCausalLM::new(config).expect("model must build");
        model
            .load_pretrained(&mut bytes.as_slice())
            .expect("a tied-embedding checkpoint must load");

        assert_eq!(
            model.lm_head.weight().data().expect("readable"),
            fixture_values(&tensors, "model.embed_tokens.weight"),
            "a tied head must reuse the embedding matrix"
        );
    }

    #[test]
    fn causal_lm_load_pretrained_rejects_a_head_of_the_wrong_width() {
        let config = loading_config();
        let mut tensors = loading_fixture(&config).tensors();
        tensors.push(F32Tensor::ramp(
            "lm_head.weight",
            &[config.vocab_size + 1, config.hidden_size],
            7.0,
        ));
        let bytes = build_safetensors(&tensors);

        let mut model = ClaudeForCausalLM::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a head of the wrong width must not be reshaped into place");
        assert!(
            err.to_string().contains("lm_head.weight"),
            "unexpected: {err}"
        );
    }
}
