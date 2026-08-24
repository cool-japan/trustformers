//! # DeepSeek-V2 Model Implementation
//!
//! Core architecture components:
//! - `DeepSeekV2RmsNorm` — standard RMS normalisation
//! - `DeepSeekV2RotaryEmbedding` — RoPE applied only to the `qk_rope_head_dim` slice
//! - `MlaAttention` — Multi-head Latent Attention with compressed KV cache
//! - `DeepSeekV2MLP` — dense SwiGLU / GELU MLP used in early layers and shared experts
//! - `DeepSeekV2MoELayer` — sparse MoE with shared + top-k routed experts
//! - `DeepSeekV2DecoderLayer` — single transformer layer (dense or MoE FFN)
//! - `DeepSeekV2Model` — full stack of decoder layers

use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::{Embedding, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

use super::config::{ActivationType, DeepSeekV2Config};

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

/// SiLU (Swish): `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU (tanh approximation).
pub fn gelu(x: f32) -> f32 {
    use std::f32::consts::PI;
    let c = (2.0f32 / PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// Apply the configured activation element-wise.
pub fn apply_activation(data: &[f32], act: ActivationType) -> Vec<f32> {
    match act {
        ActivationType::SiLU => data.iter().map(|&x| silu(x)).collect(),
        ActivationType::GeLU => data.iter().map(|&x| gelu(x)).collect(),
    }
}

// ---------------------------------------------------------------------------
// Attention building blocks
// ---------------------------------------------------------------------------

pub use super::attention::{DeepSeekV2RmsNorm, DeepSeekV2RotaryEmbedding, MlaAttention};

// ---------------------------------------------------------------------------
// Dense MLP (used in early layers and as shared experts)
// ---------------------------------------------------------------------------

/// Dense SwiGLU/GELU MLP used in non-MoE layers and as shared experts in MoE layers.
///
/// Architecture: `down_proj(act(gate_proj(x)) * up_proj(x))`
pub struct DeepSeekV2MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: ActivationType,
    device: Device,
}

impl DeepSeekV2MLP {
    pub fn new(
        in_features: usize,
        intermediate: usize,
        activation: ActivationType,
        device: Device,
    ) -> Self {
        let gate_proj = Linear::new_with_device(in_features, intermediate, false, device);
        let up_proj = Linear::new_with_device(in_features, intermediate, false, device);
        let down_proj = Linear::new_with_device(intermediate, in_features, false, device);
        Self {
            gate_proj,
            up_proj,
            down_proj,
            activation,
            device,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Total learnable parameters in this MLP.
    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
    }

    /// The three projections, for the checkpoint binder.
    pub(super) fn projections_mut(&mut self) -> (&mut Linear, &mut Linear, &mut Linear) {
        (&mut self.gate_proj, &mut self.up_proj, &mut self.down_proj)
    }
}

impl Layer for DeepSeekV2MLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate_out = self.gate_proj.forward(input.clone())?;
        let up_out = self.up_proj.forward(input)?;

        let activated = match (&gate_out, &up_out) {
            (Tensor::F32(g), Tensor::F32(u)) => {
                let g_slice = g.as_slice().ok_or_else(|| {
                    tensor_op_error("deepseek_v2_mlp", "gate tensor not contiguous")
                })?;
                let u_slice = u.as_slice().ok_or_else(|| {
                    tensor_op_error("deepseek_v2_mlp", "up tensor not contiguous")
                })?;
                let gated: Vec<f32> = apply_activation(g_slice, self.activation)
                    .into_iter()
                    .zip(u_slice.iter())
                    .map(|(g, &u)| g * u)
                    .collect();
                let shape = g.shape().to_vec();
                Tensor::from_vec(gated, &shape)?
            },
            _ => {
                return Err(tensor_op_error(
                    "deepseek_v2_mlp",
                    "gate and up tensors must be F32",
                ))
            },
        };
        self.down_proj.forward(activated)
    }
}

// ---------------------------------------------------------------------------
// Expert router
// ---------------------------------------------------------------------------

/// Lightweight top-k expert router.
///
/// Computes per-expert affinity scores from a hidden vector and returns the
/// indices of the top-`k` selected experts along with their normalised weights.
pub struct ExpertRouter {
    gate: Linear,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f32,
    #[allow(dead_code)]
    device: Device,
}

impl ExpertRouter {
    pub fn new(config: &DeepSeekV2Config, device: Device) -> Self {
        let gate =
            Linear::new_with_device(config.hidden_size, config.n_routed_experts, false, device);
        Self {
            gate,
            n_routed_experts: config.n_routed_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            n_group: config.n_group,
            topk_group: config.topk_group,
            routed_scaling_factor: config.routed_scaling_factor,
            device,
        }
    }

    /// The routing projection, for the checkpoint binder.
    pub(super) fn gate_mut(&mut self) -> &mut Linear {
        &mut self.gate
    }

    /// Total learnable parameters in the router.
    pub fn parameter_count(&self) -> usize {
        self.gate.parameter_count()
    }

    /// Compute logits and select top-k experts.
    ///
    /// Returns `(selected_expert_indices, normalised_weights)`.
    pub fn route(&self, input: &Tensor) -> Result<(Vec<usize>, Vec<f32>)> {
        let logits_tensor = self.gate.forward(input.clone())?;
        let logits: Vec<f32> = match &logits_tensor {
            Tensor::F32(arr) => arr
                .as_slice()
                .ok_or_else(|| tensor_op_error("expert_router", "logits tensor not contiguous"))?
                .to_vec(),
            _ => return Err(tensor_op_error("expert_router", "logits must be F32")),
        };

        // Softmax over all routed experts
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = if sum_exp > 0.0 {
            exp_logits.iter().map(|&x| x / sum_exp).collect()
        } else {
            vec![1.0 / self.n_routed_experts as f32; self.n_routed_experts]
        };

        // GroupLimitedGreedy: within each group, select top-`topk_group` experts,
        // then take the overall top-`num_experts_per_tok` from those candidates.
        let group_size = self.n_routed_experts.div_ceil(self.n_group);
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for g in 0..self.n_group {
            let start = g * group_size;
            let end = (start + group_size).min(self.n_routed_experts);
            let mut group_probs: Vec<(usize, f32)> =
                (start..end).map(|i| (i, *probs.get(i).unwrap_or(&0.0))).collect();
            group_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.extend(group_probs.into_iter().take(self.topk_group));
        }

        // Final top-k
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected: Vec<(usize, f32)> =
            candidates.into_iter().take(self.num_experts_per_tok).collect();

        // Normalise weights and apply scaling factor
        let weight_sum: f32 = selected.iter().map(|(_, w)| w).sum();
        let norm = if weight_sum > 0.0 { weight_sum } else { 1.0 };

        let indices: Vec<usize> = selected.iter().map(|(i, _)| *i).collect();
        let weights: Vec<f32> =
            selected.iter().map(|(_, w)| w / norm * self.routed_scaling_factor).collect();

        Ok((indices, weights))
    }
}

// ---------------------------------------------------------------------------
// MoE Layer
// ---------------------------------------------------------------------------

/// DeepSeek-V2 Mixture-of-Experts FFN layer.
///
/// Contains:
/// - `n_shared_experts` always-active shared MLP experts (outputs are summed in)
/// - `n_routed_experts` routed expert MLPs, of which `num_experts_per_tok` are selected per token
pub struct DeepSeekV2MoELayer {
    shared_experts: Vec<DeepSeekV2MLP>,
    routed_experts: Vec<DeepSeekV2MLP>,
    router: ExpertRouter,
    device: Device,
}

impl DeepSeekV2MoELayer {
    pub fn new(config: &DeepSeekV2Config, device: Device) -> Result<Self> {
        let act = config.hidden_act;
        let shared_experts = (0..config.n_shared_experts)
            .map(|_| DeepSeekV2MLP::new(config.hidden_size, config.intermediate_size, act, device))
            .collect();
        let routed_experts = (0..config.n_routed_experts)
            .map(|_| DeepSeekV2MLP::new(config.hidden_size, config.intermediate_size, act, device))
            .collect();
        let router = ExpertRouter::new(config, device);
        Ok(Self {
            shared_experts,
            routed_experts,
            router,
            device,
        })
    }

    pub fn num_routed_experts(&self) -> usize {
        self.routed_experts.len()
    }

    pub fn num_shared_experts(&self) -> usize {
        self.shared_experts.len()
    }

    /// The shared experts, for the checkpoint binder.
    pub(super) fn shared_experts_mut(&mut self) -> &mut [DeepSeekV2MLP] {
        &mut self.shared_experts
    }

    /// The routed experts, for the checkpoint binder.
    pub(super) fn routed_experts_mut(&mut self) -> &mut [DeepSeekV2MLP] {
        &mut self.routed_experts
    }

    /// The router, for the checkpoint binder.
    pub(super) fn router_mut(&mut self) -> &mut ExpertRouter {
        &mut self.router
    }

    /// Total learnable parameters across every expert and the router.
    pub fn parameter_count(&self) -> usize {
        self.shared_experts.iter().map(DeepSeekV2MLP::parameter_count).sum::<usize>()
            + self.routed_experts.iter().map(DeepSeekV2MLP::parameter_count).sum::<usize>()
            + self.router.parameter_count()
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for DeepSeekV2MoELayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // --- Shared experts (always active) ---
        let mut output: Option<Vec<f32>> = None;
        let (input_len, input_shape) = match &input {
            Tensor::F32(arr) => (arr.len(), arr.shape().to_vec()),
            _ => return Err(tensor_op_error("deepseek_v2_moe", "input must be F32")),
        };

        for expert in &self.shared_experts {
            let out = expert.forward(input.clone())?;
            let out_slice = match &out {
                Tensor::F32(arr) => arr
                    .as_slice()
                    .ok_or_else(|| {
                        tensor_op_error("deepseek_v2_moe", "shared expert output not contiguous")
                    })?
                    .to_vec(),
                _ => {
                    return Err(tensor_op_error(
                        "deepseek_v2_moe",
                        "shared expert output must be F32",
                    ))
                },
            };
            match &mut output {
                None => output = Some(out_slice),
                Some(acc) => {
                    for (a, b) in acc.iter_mut().zip(out_slice.iter()) {
                        *a += b;
                    }
                },
            }
        }

        // --- Routed experts ---
        let (expert_indices, expert_weights) = self.router.route(&input)?;
        for (idx, weight) in expert_indices.iter().zip(expert_weights.iter()) {
            let expert = self
                .routed_experts
                .get(*idx)
                .ok_or_else(|| tensor_op_error("deepseek_v2_moe", "expert index out of bounds"))?;
            let out = expert.forward(input.clone())?;
            let out_slice = match &out {
                Tensor::F32(arr) => arr
                    .as_slice()
                    .ok_or_else(|| {
                        tensor_op_error("deepseek_v2_moe", "routed expert output not contiguous")
                    })?
                    .to_vec(),
                _ => {
                    return Err(tensor_op_error(
                        "deepseek_v2_moe",
                        "routed expert output must be F32",
                    ))
                },
            };
            match &mut output {
                None => output = Some(out_slice.iter().map(|&x| x * weight).collect()),
                Some(acc) => {
                    for (a, b) in acc.iter_mut().zip(out_slice.iter()) {
                        *a += b * weight;
                    }
                },
            }
        }

        let mut result = output.unwrap_or_else(|| vec![0.0_f32; input_len]);
        result.resize(input_len, 0.0_f32);
        // Preserve original input shape
        let shape: Vec<usize> = if input_shape.is_empty() { vec![input_len] } else { input_shape };
        Tensor::from_vec(result, &shape)
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

/// DeepSeek-V2 transformer decoder layer.
///
/// Early layers (layer_idx < `first_k_dense_replace`) use a dense MLP.
/// All subsequent layers (respecting `moe_layer_freq`) use a MoE FFN.
pub struct DeepSeekV2DecoderLayer {
    self_attn: MlaAttention,
    /// Dense MLP, present when this is a dense layer.
    dense_mlp: Option<DeepSeekV2MLP>,
    /// MoE layer, present when this is a MoE layer.
    moe_layer: Option<DeepSeekV2MoELayer>,
    input_layernorm: DeepSeekV2RmsNorm,
    post_attention_layernorm: DeepSeekV2RmsNorm,
    device: Device,
}

impl DeepSeekV2DecoderLayer {
    pub fn new(config: &DeepSeekV2Config, layer_idx: usize, device: Device) -> Result<Self> {
        let self_attn = MlaAttention::new(config, device)?;
        let input_layernorm =
            DeepSeekV2RmsNorm::new(config.hidden_size, config.rms_norm_eps, device)?;
        let post_attention_layernorm =
            DeepSeekV2RmsNorm::new(config.hidden_size, config.rms_norm_eps, device)?;

        let (dense_mlp, moe_layer) = if config.is_dense_layer(layer_idx) {
            let mlp = DeepSeekV2MLP::new(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                device,
            );
            (Some(mlp), None)
        } else {
            let moe = DeepSeekV2MoELayer::new(config, device)?;
            (None, Some(moe))
        };

        Ok(Self {
            self_attn,
            dense_mlp,
            moe_layer,
            input_layernorm,
            post_attention_layernorm,
            device,
        })
    }

    /// Returns `true` when this layer uses a dense (non-MoE) FFN.
    pub fn is_dense(&self) -> bool {
        self.dense_mlp.is_some()
    }

    /// Total learnable parameters in this layer.
    pub fn parameter_count(&self) -> usize {
        let ffn = match (&self.dense_mlp, &self.moe_layer) {
            (Some(mlp), _) => mlp.parameter_count(),
            (None, Some(moe)) => moe.parameter_count(),
            (None, None) => 0,
        };
        self.self_attn.parameter_count()
            + ffn
            + self.input_layernorm.parameter_count()
            + self.post_attention_layernorm.parameter_count()
    }

    /// The attention block, for the checkpoint binder.
    pub(super) fn self_attn_mut(&mut self) -> &mut MlaAttention {
        &mut self.self_attn
    }

    /// The dense FFN, for the checkpoint binder.
    pub(super) fn dense_mlp_mut(&mut self) -> Option<&mut DeepSeekV2MLP> {
        self.dense_mlp.as_mut()
    }

    /// The MoE FFN, for the checkpoint binder.
    pub(super) fn moe_layer_mut(&mut self) -> Option<&mut DeepSeekV2MoELayer> {
        self.moe_layer.as_mut()
    }

    /// The two per-layer norms, for the checkpoint binder.
    pub(super) fn norms_mut(&mut self) -> (&mut DeepSeekV2RmsNorm, &mut DeepSeekV2RmsNorm) {
        (
            &mut self.input_layernorm,
            &mut self.post_attention_layernorm,
        )
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for DeepSeekV2DecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Pre-norm → attention → residual.
        //
        // `MlaAttention` projects back to `hidden_size`, so both residual adds
        // are real shape-checked additions. A previous revision wrote
        // `input.add(&attn_out).unwrap_or(attn_out)` / `.or(Ok(ff_out))`, which
        // silently *dropped the residual branch* whenever the shapes disagreed —
        // and they always disagreed, because that attention block returned a
        // resized copy of its own query buffer.
        let normed = self.input_layernorm.forward(input.clone())?;
        let attn_out = self.self_attn.forward(normed)?;
        let hidden = input.add(&attn_out)?;

        // Pre-norm → FFN → residual
        let normed_ff = self.post_attention_layernorm.forward(hidden.clone())?;
        let ff_out = if let Some(mlp) = &self.dense_mlp {
            mlp.forward(normed_ff)?
        } else if let Some(moe) = &self.moe_layer {
            moe.forward(normed_ff)?
        } else {
            return Err(tensor_op_error(
                "deepseek_v2_decoder",
                "layer has neither dense_mlp nor moe_layer",
            ));
        };
        hidden.add(&ff_out)
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2Model
// ---------------------------------------------------------------------------

/// DeepSeek-V2 base model: token embedding + decoder layers + final RMSNorm.
pub struct DeepSeekV2Model {
    config: DeepSeekV2Config,
    embed_tokens: Embedding,
    layers: Vec<DeepSeekV2DecoderLayer>,
    norm: DeepSeekV2RmsNorm,
    device: Device,
}

impl DeepSeekV2Model {
    pub fn new(config: DeepSeekV2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: DeepSeekV2Config, device: Device) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(DeepSeekV2DecoderLayer::new(&config, layer_idx, device)?);
        }

        let norm = DeepSeekV2RmsNorm::new(config.hidden_size, config.rms_norm_eps, device)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            device,
        })
    }

    pub fn config(&self) -> &DeepSeekV2Config {
        &self.config
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The token embedding table, for the checkpoint binder.
    pub(super) fn embed_tokens_mut(&mut self) -> &mut Embedding {
        &mut self.embed_tokens
    }

    /// The decoder stack, for the checkpoint binder.
    pub(super) fn layers_mut(&mut self) -> &mut [DeepSeekV2DecoderLayer] {
        &mut self.layers
    }

    /// The final norm, for the checkpoint binder.
    pub(super) fn final_norm_mut(&mut self) -> &mut DeepSeekV2RmsNorm {
        &mut self.norm
    }

    /// The token embedding table.
    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }
}

impl Model for DeepSeekV2Model {
    type Config = DeepSeekV2Config;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        let token_ids: Vec<u32> = match &input_ids {
            Tensor::I64(arr) => arr.as_slice().unwrap_or(&[]).iter().map(|&x| x as u32).collect(),
            Tensor::F32(arr) => {
                arr.as_slice().unwrap_or(&[]).iter().map(|&x| x.round() as u32).collect()
            },
            _ => {
                return Err(tensor_op_error(
                    "deepseek_v2_forward",
                    "input_ids must be I64 or F32",
                ))
            },
        };

        let mut hidden_states = self.embed_tokens.forward(token_ids)?;
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }
        self.norm.forward(hidden_states)
    }

    /// Load a HuggingFace DeepSeek-V2 checkpoint.
    ///
    /// Every tensor is bound by name; the complete map lives in
    /// [`crate::deepseek_v2::loading`]. Nothing is skipped: a parameter the
    /// checkpoint does not carry, and a checkpoint tensor this architecture does
    /// not recognise, both fail the load with the offending names listed.
    ///
    /// Two earlier revisions of this method were both dishonest in their own
    /// way. The first read the stream into a buffer, checked only that the
    /// buffer was non-empty and returned `Ok(())` — binding nothing. The second
    /// replaced that with a `not_implemented` error, correct at the time,
    /// because this file's attention block stored `c_kv`/`k_pe`/`k_nope`/
    /// `v_proj` and modelled neither latent norm, so a real export's tensors had
    /// nowhere to land. The attention block now matches the reference
    /// implementation, so the binder is real.
    ///
    /// # Errors
    ///
    /// Fails when the container cannot be parsed, when the stream does not look
    /// like a DeepSeek-V2 checkpoint, when a tensor has the wrong shape, when a
    /// parameter is missing, or when an unrecognised tensor is present.
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    /// Count the parameters this model actually holds.
    ///
    /// Summed from the live layers rather than re-derived from the config: the
    /// previous formula estimated the MLA block from a projection decomposition
    /// this model no longer uses and charged every layer a dense MLP even when
    /// it was a MoE layer, so the number disagreed with the model in front of it.
    fn num_parameters(&self) -> usize {
        self.embed_tokens.parameter_count()
            + self.layers.iter().map(DeepSeekV2DecoderLayer::parameter_count).sum::<usize>()
            + self.norm.parameter_count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek_v2::config::{ActivationType, DeepSeekV2Config, TopKMethod};
    use trustformers_core::{
        tensor::Tensor,
        traits::{Config, Model},
    };

    /// Minimal config that builds fast in tests.
    fn tiny_config() -> DeepSeekV2Config {
        DeepSeekV2Config {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            kv_lora_rank: 8,
            q_lora_rank: 16,
            qk_rope_head_dim: 4,
            qk_nope_head_dim: 4,
            v_head_dim: 4,
            num_experts_per_tok: 2,
            n_routed_experts: 4,
            n_shared_experts: 1,
            routed_scaling_factor: 1.0,
            topk_method: TopKMethod::Noaux,
            n_group: 2,
            topk_group: 1,
            aux_loss_alpha: 0.001,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            hidden_act: ActivationType::SiLU,
            initializer_range: 0.02,
            first_k_dense_replace: 1,
            moe_layer_freq: 1,
        }
    }

    // ── Config tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_default_kv_lora_rank() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(
            cfg.kv_lora_rank, 512,
            "MLA kv_lora_rank default should be 512"
        );
    }

    #[test]
    fn test_default_q_lora_rank() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(
            cfg.q_lora_rank, 1536,
            "MLA q_lora_rank default should be 1536"
        );
    }

    #[test]
    fn test_default_qk_nope_head_dim() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(
            cfg.qk_nope_head_dim, 128,
            "no-RoPE head_dim default should be 128"
        );
    }

    #[test]
    fn test_default_qk_rope_head_dim() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(
            cfg.qk_rope_head_dim, 64,
            "RoPE head_dim default should be 64"
        );
    }

    #[test]
    fn test_qk_head_dim_sum() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(
            cfg.qk_head_dim(),
            cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            "total head_dim = rope_head_dim + nope_head_dim"
        );
    }

    #[test]
    fn test_default_num_attention_heads() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(
            cfg.num_attention_heads, 128,
            "DeepSeek-V2 has 128 attention heads"
        );
    }

    #[test]
    fn test_config_validate_ok() {
        tiny_config().validate().expect("tiny_config should be valid");
    }

    #[test]
    fn test_config_validate_zero_kv_lora_rank_fails() {
        let mut cfg = tiny_config();
        cfg.kv_lora_rank = 0;
        assert!(
            cfg.validate().is_err(),
            "zero kv_lora_rank must fail validation"
        );
    }

    #[test]
    fn test_config_validate_experts_per_tok_exceeds_total_fails() {
        let mut cfg = tiny_config();
        cfg.num_experts_per_tok = cfg.n_routed_experts + 1;
        assert!(
            cfg.validate().is_err(),
            "experts_per_tok > n_routed_experts must fail"
        );
    }

    #[test]
    fn test_dense_layer_detection_first_k() {
        let cfg = tiny_config(); // first_k_dense_replace = 1
        assert!(
            cfg.is_dense_layer(0),
            "layer 0 should be dense (first_k_dense_replace=1)"
        );
        assert!(
            !cfg.is_dense_layer(1),
            "layer 1 should be MoE (moe_layer_freq=1)"
        );
    }

    // ── Activation function tests ──────────────────────────────────────────────

    #[test]
    fn test_silu_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6, "silu(0) == 0");
    }

    #[test]
    fn test_silu_positive_input_positive_output() {
        assert!(silu(1.0) > 0.0, "silu(1.0) should be positive");
    }

    #[test]
    fn test_gelu_zero() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-4, "gelu(0) ≈ 0");
    }

    #[test]
    fn test_apply_activation_length_preserved() {
        let data = vec![1.0_f32, -1.0, 0.5, 2.0];
        let out_silu = apply_activation(&data, ActivationType::SiLU);
        let out_gelu = apply_activation(&data, ActivationType::GeLU);
        assert_eq!(
            out_silu.len(),
            data.len(),
            "silu activation preserves length"
        );
        assert_eq!(
            out_gelu.len(),
            data.len(),
            "gelu activation preserves length"
        );
    }

    // ── RMSNorm tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_unit_weight_normalizes() {
        let device = trustformers_core::device::Device::CPU;
        let norm =
            DeepSeekV2RmsNorm::new(4, 1e-6, device).expect("rmsnorm creation should succeed");
        let input =
            Tensor::from_vec(vec![2.0_f32; 4], &[4]).expect("tensor creation should succeed");
        let output = norm.forward(input).expect("rmsnorm forward should succeed");
        let vals = output.to_vec_f32().expect("to_vec_f32 should succeed");
        for v in vals {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "unit weights + uniform input → ≈ 1.0, got {v}"
            );
        }
    }

    // ── RoPE tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_rope_apply_preserves_length() {
        let cfg = tiny_config();
        let device = trustformers_core::device::Device::CPU;
        let rope = DeepSeekV2RotaryEmbedding::new(&cfg, device);
        let seq_len = 4;
        let mut data = vec![0.5_f32; seq_len * cfg.qk_rope_head_dim];
        rope.apply(&mut data, seq_len);
        assert_eq!(
            data.len(),
            seq_len * cfg.qk_rope_head_dim,
            "RoPE must preserve data length"
        );
    }

    #[test]
    fn test_rope_position_zero_unchanged() {
        let cfg = tiny_config();
        let device = trustformers_core::device::Device::CPU;
        let rope = DeepSeekV2RotaryEmbedding::new(&cfg, device);
        // At position 0, angle = 0 → cos=1, sin=0 → values unchanged
        let original = vec![1.0_f32, 0.0, 1.0, 0.0];
        let mut data = original.clone();
        rope.apply(&mut data, 1);
        for (orig, got) in original.iter().zip(data.iter()) {
            assert!(
                (orig - got).abs() < 1e-5,
                "pos=0 should leave values unchanged"
            );
        }
    }

    // ── MLA Attention tests ───────────────────────────────────────────────────

    #[test]
    fn test_mla_attention_creation() {
        let cfg = tiny_config();
        let device = trustformers_core::device::Device::CPU;
        MlaAttention::new(&cfg, device).expect("MlaAttention creation should succeed");
    }

    #[test]
    fn test_mla_attention_output_shape() {
        let cfg = tiny_config();
        let hidden_size = cfg.hidden_size;
        let device = trustformers_core::device::Device::CPU;
        let attn = MlaAttention::new(&cfg, device).expect("MlaAttention should be created");
        // Linear requires at least 2D input: [seq_len, hidden_size]
        let input = Tensor::from_vec(vec![0.1_f32; hidden_size], &[1, hidden_size])
            .expect("tensor creation should succeed");
        let output = attn.forward(input).expect("MlaAttention forward should succeed");
        assert_eq!(
            output.shape()[output.shape().len() - 1],
            hidden_size,
            "MLA output must project back to hidden_size"
        );
    }

    // ── Model tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_model_creation() {
        let cfg = tiny_config();
        DeepSeekV2Model::new(cfg).expect("model creation should succeed");
    }

    #[test]
    fn test_model_forward_with_f32_ids() {
        let cfg = tiny_config();
        let hidden_size = cfg.hidden_size;
        let model = DeepSeekV2Model::new(cfg).expect("model creation should succeed");
        let input_ids = Tensor::from_vec(vec![0.0_f32, 1.0, 2.0], &[3])
            .expect("tensor creation should succeed");
        let output = model.forward(input_ids).expect("model forward should succeed");
        let shape = output.shape();
        assert_eq!(
            shape[shape.len() - 1],
            hidden_size,
            "output last dim must be hidden_size"
        );
    }

    #[test]
    fn test_model_parameter_count_nonzero() {
        let cfg = tiny_config();
        let model = DeepSeekV2Model::new(cfg).expect("model creation should succeed");
        assert!(model.num_parameters() > 0, "model must have parameters");
    }

    /// Regression: `load_pretrained` used to read the stream, check only that it
    /// was non-empty and return `Ok(())` without binding anything — so *any*
    /// non-empty byte sequence reported a successful load while the model kept
    /// its constructor initialisation. A later revision refused outright. Now a
    /// real binder runs, so a buffer that is not a checkpoint at all must fail
    /// on the container, and no parameter may move.
    #[test]
    fn load_pretrained_rejects_a_buffer_that_is_not_a_checkpoint() {
        let mut model = DeepSeekV2Model::new(tiny_config()).expect("model must build");
        let before = model.embed_tokens.weight().data().expect("readable");

        let plausible_weights = vec![0x11u8; 4096];
        model
            .load_pretrained(&mut plausible_weights.as_slice())
            .expect_err("a buffer that is not a checkpoint must not be reported as a load");
        assert_eq!(
            model.embed_tokens.weight().data().expect("readable"),
            before,
            "a refused load must leave every parameter untouched"
        );
    }

    /// The stream is consumed, so a caller that reuses the reader sees a defined
    /// state rather than a partially consumed one.
    #[test]
    fn load_pretrained_drains_the_reader_before_refusing() {
        let mut model = DeepSeekV2Model::new(tiny_config()).expect("model must build");
        let bytes = vec![0x22u8; 128];
        let mut cursor = bytes.as_slice();
        let _ = model.load_pretrained(&mut cursor);
        assert!(
            cursor.is_empty(),
            "the reader must be fully consumed even when the load is refused"
        );
    }
}
