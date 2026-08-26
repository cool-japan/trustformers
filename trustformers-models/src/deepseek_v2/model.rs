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

/// One token's routing decision, as produced by [`ExpertRouter::route_all`].
///
/// Routing in a Mixture-of-Experts layer is a *per-token* decision, so this is
/// the unit the router works in. The full probability row is carried alongside
/// the selection deliberately: a caller (and the regression tests) can then
/// check the distribution itself — that it is a distribution at all, over this
/// token's experts — rather than only which experts came out on top.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenRouting {
    /// Softmax over this token's `n_routed_experts` gate logits. Length is
    /// `n_routed_experts` and the entries sum to 1.
    pub probabilities: Vec<f32>,
    /// The selected expert indices, most probable first. At most
    /// `num_experts_per_tok` of them.
    pub experts: Vec<usize>,
    /// The factor each selected expert's output is scaled by before it is summed
    /// into this token's output: the selected probabilities renormalised to sum
    /// to 1, then multiplied by `routed_scaling_factor`. Parallel to `experts`.
    pub weights: Vec<f32>,
}

/// Lightweight top-k expert router.
///
/// Computes per-expert affinity scores from each token's hidden vector and
/// returns, for every token independently, the indices of the top-`k` selected
/// experts along with their normalised weights.
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

    /// Route every token in `input` independently.
    ///
    /// `input` is `[num_tokens, hidden_size]`; the gate produces
    /// `[num_tokens, n_routed_experts]` logits, and **each row is softmaxed and
    /// top-k'd on its own**. That per-row treatment is the whole point of a
    /// Mixture-of-Experts router and it is what an earlier revision of this
    /// method did not do: it flattened the logits, ran one softmax over all
    /// `num_tokens * n_routed_experts` values at once (so no token's
    /// probabilities summed to 1) and then sliced `probs[0..n_routed_experts]`,
    /// which is token 0's row — every token in the sequence was routed to token
    /// 0's experts with token 0's weights, and `num_experts_per_tok` described
    /// the sequence rather than the token.
    ///
    /// Selection is DeepSeek-V2's GroupLimitedGreedy: rank within each of
    /// `n_group` groups, keep `topk_group` per group, then take the global
    /// top-`num_experts_per_tok` of those candidates.
    ///
    /// # Errors
    ///
    /// Fails when the router is configured with zero experts or zero groups,
    /// when the gate's output is not a contiguous `F32` tensor, or when that
    /// output's length is not a whole number of `n_routed_experts` rows.
    pub fn route_all(&self, input: &Tensor) -> Result<Vec<TokenRouting>> {
        if self.n_routed_experts == 0 {
            return Err(tensor_op_error(
                "expert_router",
                "n_routed_experts must be > 0 to route anything",
            ));
        }
        if self.n_group == 0 {
            return Err(tensor_op_error(
                "expert_router",
                "n_group must be > 0: GroupLimitedGreedy needs at least one group",
            ));
        }

        let logits_tensor = self.gate.forward(input.clone())?;
        let logits: Vec<f32> = match &logits_tensor {
            Tensor::F32(arr) => arr
                .as_slice()
                .ok_or_else(|| tensor_op_error("expert_router", "logits tensor not contiguous"))?
                .to_vec(),
            _ => return Err(tensor_op_error("expert_router", "logits must be F32")),
        };

        if !logits.len().is_multiple_of(self.n_routed_experts) {
            return Err(tensor_op_error(
                "expert_router",
                format!(
                    "gate produced {} logits, which is not a whole number of rows of \
                     n_routed_experts = {}",
                    logits.len(),
                    self.n_routed_experts
                ),
            ));
        }

        let num_tokens = logits.len() / self.n_routed_experts;
        let mut routing = Vec::with_capacity(num_tokens);
        for token in 0..num_tokens {
            let start = token * self.n_routed_experts;
            routing.push(self.route_row(&logits[start..start + self.n_routed_experts]));
        }
        Ok(routing)
    }

    /// Route one token from its own `n_routed_experts` gate logits.
    fn route_row(&self, logits: &[f32]) -> TokenRouting {
        // Softmax over this token's routed experts, so the row is a real
        // probability distribution.
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probabilities: Vec<f32> = if sum_exp > 0.0 && sum_exp.is_finite() {
            exp_logits.iter().map(|&x| x / sum_exp).collect()
        } else {
            // Every logit was NaN or the exponentials underflowed to nothing:
            // a uniform row is the only distribution the gate supports here.
            vec![1.0 / self.n_routed_experts as f32; self.n_routed_experts]
        };

        // GroupLimitedGreedy: within each group, select top-`topk_group` experts,
        // then take the overall top-`num_experts_per_tok` from those candidates.
        let group_size = self.n_routed_experts.div_ceil(self.n_group);
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for g in 0..self.n_group {
            let start = g * group_size;
            let end = (start + group_size).min(self.n_routed_experts);
            if start >= end {
                continue;
            }
            let mut group_probs: Vec<(usize, f32)> = (start..end)
                .map(|i| (i, probabilities.get(i).copied().unwrap_or(0.0)))
                .collect();
            group_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.extend(group_probs.into_iter().take(self.topk_group));
        }

        // Final top-k
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected: Vec<(usize, f32)> =
            candidates.into_iter().take(self.num_experts_per_tok).collect();

        // Normalise weights across the selected experts and apply the scaling factor
        let weight_sum: f32 = selected.iter().map(|(_, w)| w).sum();
        let norm = if weight_sum > 0.0 { weight_sum } else { 1.0 };

        let experts: Vec<usize> = selected.iter().map(|(i, _)| *i).collect();
        let weights: Vec<f32> =
            selected.iter().map(|(_, w)| w / norm * self.routed_scaling_factor).collect();

        TokenRouting {
            probabilities,
            experts,
            weights,
        }
    }

    /// Route a single token.
    ///
    /// Returns `(selected_expert_indices, normalised_weights)` for that one
    /// token — the convenience form of [`ExpertRouter::route_all`].
    ///
    /// # Errors
    ///
    /// Everything [`ExpertRouter::route_all`] fails on, plus an input carrying
    /// anything other than exactly one token. Reporting one token's decision for
    /// a whole sequence is precisely the bug this router used to have, so a
    /// multi-token input is refused here rather than silently answered for the
    /// first row: call [`ExpertRouter::route_all`] for a sequence.
    pub fn route(&self, input: &Tensor) -> Result<(Vec<usize>, Vec<f32>)> {
        let mut routing = self.route_all(input)?;
        if routing.len() != 1 {
            return Err(tensor_op_error(
                "expert_router",
                format!(
                    "route() answers for a single token but the input carries {}: use \
                     route_all() to route a sequence",
                    routing.len()
                ),
            ));
        }
        let decision = routing.remove(0);
        Ok((decision.experts, decision.weights))
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

    /// Run the sparse FFN.
    ///
    /// Every token is routed on its own logits (see
    /// [`ExpertRouter::route_all`]) and then **executed on its own row**: the
    /// selected experts see only that token's hidden vector, and their outputs
    /// are summed into that token's output row weighted by that token's gate
    /// probabilities. Shared experts are applied to every token, as they are
    /// always active.
    ///
    /// The previous revision routed once for the whole sequence and then ran
    /// each selected expert over the *entire* input, so a sequence of `n` tokens
    /// got token 0's expert set applied uniformly — the layer was dense in
    /// everything but name.
    ///
    /// # Errors
    ///
    /// Fails when the input is not a contiguous `F32` tensor, when the routed
    /// token count does not divide the input evenly, when a selected expert
    /// index is out of range, or when an expert returns a row of the wrong
    /// width.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (input_data, input_shape) = match &input {
            Tensor::F32(arr) => (
                arr.as_slice()
                    .ok_or_else(|| {
                        tensor_op_error("deepseek_v2_moe", "input tensor not contiguous")
                    })?
                    .to_vec(),
                arr.shape().to_vec(),
            ),
            _ => return Err(tensor_op_error("deepseek_v2_moe", "input must be F32")),
        };
        let input_len = input_data.len();

        // --- Routing: one decision per token, never one for the whole sequence ---
        let routing = self.router.route_all(&input)?;
        let num_tokens = routing.len();
        if num_tokens == 0 {
            // No rows to route: nothing to compute, and nothing to invent.
            return Ok(input);
        }
        if !input_len.is_multiple_of(num_tokens) {
            return Err(tensor_op_error(
                "deepseek_v2_moe",
                format!(
                    "input of {input_len} values does not split evenly into {num_tokens} tokens"
                ),
            ));
        }
        let hidden_size = input_len / num_tokens;

        let mut output = vec![0.0_f32; input_len];

        // --- Shared experts (always active, every token) ---
        for expert in &self.shared_experts {
            let out = expert.forward(input.clone())?;
            let out_values = moe_output_values(&out, "shared expert")?;
            if out_values.len() != input_len {
                return Err(tensor_op_error(
                    "deepseek_v2_moe",
                    format!(
                        "shared expert returned {} values for an input of {input_len}",
                        out_values.len()
                    ),
                ));
            }
            for (accumulator, value) in output.iter_mut().zip(out_values.iter()) {
                *accumulator += value;
            }
        }

        // --- Routed experts (per token) ---
        for (token_index, decision) in routing.iter().enumerate() {
            if decision.experts.is_empty() {
                continue;
            }
            let start = token_index * hidden_size;
            let row = Tensor::from_vec(
                input_data[start..start + hidden_size].to_vec(),
                &[1, hidden_size],
            )?;

            for (expert_index, weight) in decision.experts.iter().zip(decision.weights.iter()) {
                let expert = self.routed_experts.get(*expert_index).ok_or_else(|| {
                    tensor_op_error(
                        "deepseek_v2_moe",
                        format!(
                            "routed expert index {expert_index} is out of bounds for {} experts",
                            self.routed_experts.len()
                        ),
                    )
                })?;
                let out = expert.forward(row.clone())?;
                let out_values = moe_output_values(&out, "routed expert")?;
                if out_values.len() != hidden_size {
                    return Err(tensor_op_error(
                        "deepseek_v2_moe",
                        format!(
                            "routed expert returned {} values for a token of width {hidden_size}",
                            out_values.len()
                        ),
                    ));
                }
                for (accumulator, value) in
                    output[start..start + hidden_size].iter_mut().zip(out_values.iter())
                {
                    *accumulator += value * weight;
                }
            }
        }

        // Preserve original input shape
        let shape: Vec<usize> = if input_shape.is_empty() { vec![input_len] } else { input_shape };
        Tensor::from_vec(output, &shape)
    }
}

/// Read an expert's output as a contiguous `f32` slice.
///
/// `what` names the expert kind so a failure says which of the two paths
/// produced the unusable tensor.
fn moe_output_values(tensor: &Tensor, what: &str) -> Result<Vec<f32>> {
    match tensor {
        Tensor::F32(arr) => Ok(arr
            .as_slice()
            .ok_or_else(|| {
                tensor_op_error("deepseek_v2_moe", format!("{what} output not contiguous"))
            })?
            .to_vec()),
        _ => Err(tensor_op_error(
            "deepseek_v2_moe",
            format!("{what} output must be F32"),
        )),
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

    // ── MoE routing tests ─────────────────────────────────────────────────────

    /// Install `scale * I` as a square `Linear`'s weight, turning it into an
    /// element-wise multiply so an expert's output can be computed by hand.
    fn set_scaled_identity(layer: &mut Linear, size: usize, scale: f32) {
        let mut data = vec![0.0_f32; size * size];
        for row in 0..size {
            data[row * size + row] = scale;
        }
        layer
            .set_weight(Tensor::from_vec(data, &[size, size]).expect("identity weight tensor"))
            .expect("installing a correctly shaped weight succeeds");
    }

    /// A 4-wide MoE layer with two routed experts and no shared experts, small
    /// enough that every expert's output can be written down by hand.
    fn handbuilt_moe_config() -> DeepSeekV2Config {
        DeepSeekV2Config {
            hidden_size: 4,
            intermediate_size: 4,
            n_routed_experts: 2,
            n_shared_experts: 0,
            num_experts_per_tok: 1,
            n_group: 1,
            topk_group: 2,
            routed_scaling_factor: 1.0,
            hidden_act: ActivationType::SiLU,
            ..tiny_config()
        }
    }

    /// Every token gets its own softmax over `n_routed_experts`.
    ///
    /// Regression: the router used to flatten the `[seq_len, n_routed_experts]`
    /// gate logits and run **one** softmax over all `seq_len * n_routed_experts`
    /// values, so no token's row was a probability distribution — the whole
    /// matrix summed to 1 instead. `routed_scaling_factor` is deliberately not
    /// 1.0 here so the "probabilities sum to 1" assertion cannot be satisfied by
    /// the selected weights and vice versa.
    #[test]
    fn moe_router_gives_every_token_its_own_probability_distribution() {
        let cfg = DeepSeekV2Config {
            routed_scaling_factor: 2.5,
            ..tiny_config()
        };
        let router = ExpertRouter::new(&cfg, Device::CPU);
        let seq_len = 3;
        let data: Vec<f32> =
            (0..seq_len * cfg.hidden_size).map(|i| ((i % 9) as f32) * 0.2 - 0.7).collect();
        let input =
            Tensor::from_vec(data, &[seq_len, cfg.hidden_size]).expect("input tensor builds");

        let routing = router.route_all(&input).expect("routing succeeds");
        assert_eq!(routing.len(), seq_len, "one routing decision per token");

        for (token, decision) in routing.iter().enumerate() {
            assert_eq!(
                decision.probabilities.len(),
                cfg.n_routed_experts,
                "token {token}: the probability row covers every routed expert"
            );
            let total: f32 = decision.probabilities.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "token {token}: probabilities sum to {total}, not 1.0 — the old router \
                 softmaxed over seq_len × n_routed_experts values at once"
            );
            assert_eq!(
                decision.experts.len(),
                cfg.num_experts_per_tok,
                "token {token}: exactly num_experts_per_tok experts are selected"
            );
            assert_eq!(
                decision.weights.len(),
                decision.experts.len(),
                "token {token}: one weight per selected expert"
            );
            let weight_total: f32 = decision.weights.iter().sum();
            assert!(
                (weight_total - cfg.routed_scaling_factor).abs() < 1e-5,
                "token {token}: the selected weights renormalise to routed_scaling_factor \
                 ({}), got {weight_total}",
                cfg.routed_scaling_factor
            );
        }
    }

    /// Two tokens whose gate logits point at opposite experts must be routed to
    /// opposite experts.
    ///
    /// The gate is hand-built as the identity, so token `[8,0,0,0]` has its
    /// largest logit at expert 0 and token `[0,0,0,8]` at expert 3. Under the
    /// previous per-sequence routing both tokens received token 0's expert set.
    #[test]
    fn moe_router_sends_two_tokens_with_opposing_logits_to_different_experts() {
        let cfg = DeepSeekV2Config {
            hidden_size: 4,
            n_routed_experts: 4,
            num_experts_per_tok: 1,
            n_group: 1,
            topk_group: 4,
            routed_scaling_factor: 1.0,
            ..tiny_config()
        };
        let mut router = ExpertRouter::new(&cfg, Device::CPU);
        set_scaled_identity(router.gate_mut(), 4, 1.0);

        let input = Tensor::from_vec(vec![8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0], &[2, 4])
            .expect("input tensor builds");
        let routing = router.route_all(&input).expect("routing succeeds");

        assert_eq!(routing.len(), 2, "one routing decision per token");
        assert_eq!(
            routing[0].experts,
            vec![0],
            "the first token's largest logit is expert 0, got {:?} from {:?}",
            routing[0].experts,
            routing[0].probabilities
        );
        assert_eq!(
            routing[1].experts,
            vec![3],
            "the second token's largest logit is expert 3, got {:?} from {:?}",
            routing[1].experts,
            routing[1].probabilities
        );
        assert_ne!(
            routing[0].experts, routing[1].experts,
            "opposing tokens must not share an expert set"
        );
        for (token, decision) in routing.iter().enumerate() {
            let total: f32 = decision.probabilities.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "token {token}: probabilities sum to {total}, not 1.0"
            );
        }
    }

    /// `route()` answers for one token, so it refuses a sequence rather than
    /// reporting the first row's decision for every token — the exact
    /// substitution the previous implementation made silently.
    #[test]
    fn expert_router_route_refuses_a_multi_token_input() {
        let cfg = tiny_config();
        let router = ExpertRouter::new(&cfg, Device::CPU);
        let input = Tensor::from_vec(vec![0.1_f32; 3 * cfg.hidden_size], &[3, cfg.hidden_size])
            .expect("input tensor builds");
        let error = router
            .route(&input)
            .expect_err("a multi-token input must be refused by the single-token entry point");
        let message = error.to_string();
        assert!(
            message.contains("route_all"),
            "the error must point at the per-sequence entry point, got: {message}"
        );
    }

    /// Each token is *executed* by the experts it routed to, not by the first
    /// token's experts.
    ///
    /// Both experts compute `scale × (silu(x) ⊙ x)` with different scales, and
    /// the gate sends token 0 to expert 0 (scale 1) and token 1 to expert 1
    /// (scale 3). Under the previous per-sequence routing token 1 would have
    /// been run through expert 0 and come out three times too small — which the
    /// final assertion pins down explicitly.
    #[test]
    fn moe_layer_runs_each_token_through_its_own_expert() {
        let cfg = handbuilt_moe_config();
        let mut layer = DeepSeekV2MoELayer::new(&cfg, Device::CPU).expect("moe layer builds");

        // Gate: logit 0 reads x[0], logit 1 reads x[1].
        layer
            .router_mut()
            .gate_mut()
            .set_weight(
                Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 4])
                    .expect("gate weight tensor"),
            )
            .expect("installing the gate weight succeeds");

        for (index, scale) in [1.0_f32, 3.0].into_iter().enumerate() {
            let (gate_proj, up_proj, down_proj) =
                layer.routed_experts_mut()[index].projections_mut();
            set_scaled_identity(gate_proj, 4, 1.0);
            set_scaled_identity(up_proj, 4, 1.0);
            set_scaled_identity(down_proj, 4, scale);
        }

        let input = Tensor::from_vec(vec![5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0], &[2, 4])
            .expect("input tensor builds");
        let output = layer.forward(input).expect("moe forward succeeds");
        let values = output.to_vec_f32().expect("output values are readable");
        assert_eq!(values.len(), 8, "the output keeps one row per token");

        let base = silu(5.0) * 5.0;
        assert!(
            (values[0] - base).abs() < 1e-3,
            "token 0 routes to expert 0 (scale 1): got {}, expected {base}",
            values[0]
        );
        assert!(
            (values[5] - 3.0 * base).abs() < 1e-2,
            "token 1 routes to expert 1 (scale 3): got {}, expected {}",
            values[5],
            3.0 * base
        );
        assert!(
            (values[5] - base).abs() > 1.0,
            "token 1 came out with expert 0's scale ({}), which is the per-sequence routing bug",
            values[5]
        );
    }

    /// Shared experts are always active, so their contribution reaches every
    /// token — including tokens whose routed experts contribute nothing.
    #[test]
    fn moe_layer_applies_shared_experts_to_every_token() {
        let cfg = DeepSeekV2Config {
            n_shared_experts: 1,
            ..handbuilt_moe_config()
        };
        let mut layer = DeepSeekV2MoELayer::new(&cfg, Device::CPU).expect("moe layer builds");

        // The one shared expert computes silu(x) ⊙ x.
        {
            let (gate_proj, up_proj, down_proj) = layer.shared_experts_mut()[0].projections_mut();
            set_scaled_identity(gate_proj, 4, 1.0);
            set_scaled_identity(up_proj, 4, 1.0);
            set_scaled_identity(down_proj, 4, 1.0);
        }
        // Both routed experts contribute exactly nothing, isolating the shared path.
        for index in 0..2 {
            let (gate_proj, up_proj, down_proj) =
                layer.routed_experts_mut()[index].projections_mut();
            set_scaled_identity(gate_proj, 4, 1.0);
            set_scaled_identity(up_proj, 4, 1.0);
            set_scaled_identity(down_proj, 4, 0.0);
        }

        let input = Tensor::from_vec(vec![5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0], &[2, 4])
            .expect("input tensor builds");
        let output = layer.forward(input).expect("moe forward succeeds");
        let values = output.to_vec_f32().expect("output values are readable");

        let base = silu(5.0) * 5.0;
        assert!(
            (values[0] - base).abs() < 1e-3,
            "the shared expert must reach token 0: got {}, expected {base}",
            values[0]
        );
        assert!(
            (values[5] - base).abs() < 1e-3,
            "the shared expert must reach token 1 too: got {}, expected {base}",
            values[5]
        );
    }

    /// Shape and finiteness over a multi-token sequence with the ordinary
    /// randomly initialised experts.
    #[test]
    fn moe_layer_forward_preserves_shape_and_stays_finite() {
        let cfg = tiny_config();
        let layer = DeepSeekV2MoELayer::new(&cfg, Device::CPU).expect("moe layer builds");
        let seq_len = 5;
        let data: Vec<f32> =
            (0..seq_len * cfg.hidden_size).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect();
        let input =
            Tensor::from_vec(data, &[seq_len, cfg.hidden_size]).expect("input tensor builds");

        let output = layer.forward(input).expect("moe forward succeeds");
        assert_eq!(
            output.shape().to_vec(),
            vec![seq_len, cfg.hidden_size],
            "the MoE layer preserves the input shape"
        );
        let values = output.to_vec_f32().expect("output values are readable");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "every output value must be finite"
        );
    }
}
