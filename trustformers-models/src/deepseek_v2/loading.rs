//! Checkpoint binding for DeepSeek-V2.
//!
//! # Tensor-name contract
//!
//! The names below are HuggingFace's, as written by `DeepseekV2ForCausalLM`
//! (`modeling_deepseek.py`). `{P}` is the backbone prefix, detected as `model.`
//! for a causal-LM export and `""` for a bare backbone export.
//!
//! | Name | Shape |
//! |------|-------|
//! | `{P}embed_tokens.weight` | `[vocab_size, hidden_size]` |
//! | `{P}layers.{i}.input_layernorm.weight` | `[hidden_size]` |
//! | `{P}layers.{i}.post_attention_layernorm.weight` | `[hidden_size]` |
//! | `{P}layers.{i}.self_attn.q_proj.weight` | `[heads * qk_head_dim, hidden_size]` — only when `q_lora_rank == 0` |
//! | `{P}layers.{i}.self_attn.q_a_proj.weight` | `[q_lora_rank, hidden_size]` — only when `q_lora_rank > 0` |
//! | `{P}layers.{i}.self_attn.q_a_layernorm.weight` | `[q_lora_rank]` — only when `q_lora_rank > 0` |
//! | `{P}layers.{i}.self_attn.q_b_proj.weight` | `[heads * qk_head_dim, q_lora_rank]` — only when `q_lora_rank > 0` |
//! | `{P}layers.{i}.self_attn.kv_a_proj_with_mqa.weight` | `[kv_lora_rank + qk_rope_head_dim, hidden_size]` |
//! | `{P}layers.{i}.self_attn.kv_a_layernorm.weight` | `[kv_lora_rank]` |
//! | `{P}layers.{i}.self_attn.kv_b_proj.weight` | `[heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]` |
//! | `{P}layers.{i}.self_attn.o_proj.weight` | `[hidden_size, heads * v_head_dim]` |
//! | `{P}layers.{i}.mlp.{gate,up,down}_proj.weight` | dense layers only |
//! | `{P}layers.{i}.mlp.gate.weight` | `[n_routed_experts, hidden_size]` — MoE layers |
//! | `{P}layers.{i}.mlp.experts.{j}.{gate,up,down}_proj.weight` | MoE layers |
//! | `{P}layers.{i}.mlp.shared_experts.{gate,up,down}_proj.weight` | MoE layers, fused across the shared experts |
//! | `{P}norm.weight` | `[hidden_size]` |
//! | `lm_head.weight` | `[vocab_size, hidden_size]` — bound by [`DeepSeekV2ForCausalLM`], not by the backbone |
//!
//! # Two places the reference model and this one differ
//!
//! **Shared experts are fused upstream.** `DeepseekV2MoE` builds *one*
//! `DeepseekV2MLP` of width `moe_intermediate_size * n_shared_experts` for the
//! shared branch, while this crate keeps `n_shared_experts` separate MLPs and
//! sums their outputs. The two are exactly equivalent — a SwiGLU MLP over a
//! concatenated intermediate axis is the sum of the per-block MLPs, because the
//! down-projection is linear — so the fused tensors are split here rather than
//! refused: `gate_proj`/`up_proj` by rows, `down_proj` by columns.
//!
//! **One intermediate width.** This crate's config has a single
//! `intermediate_size` where the reference has `intermediate_size` (dense FFN)
//! and `moe_intermediate_size` (experts). An export whose two widths differ
//! therefore fails the shape check for its expert tensors instead of being
//! quietly mis-bound. That is a refusal, by design, not a silent conversion.

use trustformers_core::{
    errors::{Result, TrustformersError},
    layers::Linear,
    tensor::Tensor,
};

use super::attention::{DeepSeekV2RmsNorm, MlaAttention};
use super::config::DeepSeekV2Config;
use super::model::{DeepSeekV2MLP, DeepSeekV2Model};
use crate::weight_loading::binding::{bind_embedding, bind_linear};
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, UnusedTensors, WeightBinder};

/// Checkpoint namespaces a bare DeepSeek-V2 backbone legitimately leaves unbound.
///
/// A causal-LM export ships `lm_head.weight` alongside the backbone;
/// [`super::tasks::DeepSeekV2ForCausalLM`] binds it, the backbone does not.
pub(super) const ALLOWED_UNUSED_PREFIXES: &[&str] = &["lm_head."];

/// Non-parameter buffers a DeepSeek-V2 export stores next to its weights.
///
/// The rotary tables are recomputed from `rope_theta` at construction time, so
/// they are cached values rather than learnable parameters.
pub(super) const ALLOWED_UNUSED_SUFFIXES: &[&str] = &[
    "rotary_emb.inv_freq",
    "rotary_emb.cos_cached",
    "rotary_emb.sin_cached",
];

/// The unused-tensor policy for a bare backbone load.
pub(super) fn backbone_unused_policy() -> UnusedTensors<'static> {
    UnusedTensors::new(ALLOWED_UNUSED_PREFIXES, ALLOWED_UNUSED_SUFFIXES)
}

/// Every tensor name this architecture binds, for a given config.
///
/// Published so a mismatch can be reported as a contract rather than as a
/// one-off message, and so tests can assert the map instead of re-deriving it.
pub fn expected_tensor_names(config: &DeepSeekV2Config, prefix: &str) -> Vec<String> {
    let mut names = vec![format!("{prefix}embed_tokens.weight")];
    for layer in 0..config.num_hidden_layers {
        let attn = format!("{prefix}layers.{layer}.self_attn");
        if config.q_lora_rank == 0 {
            names.push(format!("{attn}.q_proj.weight"));
        } else {
            names.push(format!("{attn}.q_a_proj.weight"));
            names.push(format!("{attn}.q_a_layernorm.weight"));
            names.push(format!("{attn}.q_b_proj.weight"));
        }
        names.push(format!("{attn}.kv_a_proj_with_mqa.weight"));
        names.push(format!("{attn}.kv_a_layernorm.weight"));
        names.push(format!("{attn}.kv_b_proj.weight"));
        names.push(format!("{attn}.o_proj.weight"));

        let mlp = format!("{prefix}layers.{layer}.mlp");
        if config.is_dense_layer(layer) {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                names.push(format!("{mlp}.{projection}.weight"));
            }
        } else {
            names.push(format!("{mlp}.gate.weight"));
            for expert in 0..config.n_routed_experts {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    names.push(format!("{mlp}.experts.{expert}.{projection}.weight"));
                }
            }
            if config.n_shared_experts > 0 {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    names.push(format!("{mlp}.shared_experts.{projection}.weight"));
                }
            }
        }
        names.push(format!("{prefix}layers.{layer}.input_layernorm.weight"));
        names.push(format!(
            "{prefix}layers.{layer}.post_attention_layernorm.weight"
        ));
    }
    names.push(format!("{prefix}norm.weight"));
    names
}

/// Split a `[rows, cols]` tensor into `parts` row blocks of `[rows / parts, cols]`.
fn split_rows(name: &str, tensor: &Tensor, parts: usize) -> Result<Vec<Tensor>> {
    let shape = tensor.shape();
    if shape.len() != 2 || parts == 0 || !shape[0].is_multiple_of(parts) {
        return Err(TrustformersError::shape_error(format!(
            "checkpoint tensor {name} has shape {shape:?}; expected a 2-D tensor whose first \
             dimension divides into {parts} shared-expert block(s)"
        )));
    }
    let values = tensor.to_vec_f32()?;
    let block_rows = shape[0] / parts;
    let block_len = block_rows * shape[1];
    (0..parts)
        .map(|part| {
            Tensor::from_vec(
                values[part * block_len..(part + 1) * block_len].to_vec(),
                &[block_rows, shape[1]],
            )
        })
        .collect()
}

/// Split a `[rows, cols]` tensor into `parts` column blocks of `[rows, cols / parts]`.
fn split_columns(name: &str, tensor: &Tensor, parts: usize) -> Result<Vec<Tensor>> {
    let shape = tensor.shape();
    if shape.len() != 2 || parts == 0 || !shape[1].is_multiple_of(parts) {
        return Err(TrustformersError::shape_error(format!(
            "checkpoint tensor {name} has shape {shape:?}; expected a 2-D tensor whose second \
             dimension divides into {parts} shared-expert block(s)"
        )));
    }
    let values = tensor.to_vec_f32()?;
    let (rows, cols) = (shape[0], shape[1]);
    let block_cols = cols / parts;
    (0..parts)
        .map(|part| {
            let mut block = Vec::with_capacity(rows * block_cols);
            for row in 0..rows {
                let start = row * cols + part * block_cols;
                block.extend_from_slice(&values[start..start + block_cols]);
            }
            Tensor::from_vec(block, &[rows, block_cols])
        })
        .collect()
}

/// Bind one `[dim]` RMS-norm weight.
fn bind_rms_norm(
    binder: &mut WeightBinder<'_>,
    name: &str,
    dim: usize,
    norm: &mut DeepSeekV2RmsNorm,
) -> Result<()> {
    if let Some(weight) = binder.take_shaped(&format!("{name}.weight"), &[dim])? {
        norm.set_weight(weight)?;
    }
    Ok(())
}

/// Bind one SwiGLU MLP's three projections.
fn bind_mlp(
    binder: &mut WeightBinder<'_>,
    name: &str,
    hidden_size: usize,
    intermediate_size: usize,
    mlp: &mut DeepSeekV2MLP,
) -> Result<()> {
    let (gate, up, down) = mlp.projections_mut();
    bind_linear(
        binder,
        &format!("{name}.gate_proj"),
        intermediate_size,
        hidden_size,
        false,
        gate,
    )?;
    bind_linear(
        binder,
        &format!("{name}.up_proj"),
        intermediate_size,
        hidden_size,
        false,
        up,
    )?;
    bind_linear(
        binder,
        &format!("{name}.down_proj"),
        hidden_size,
        intermediate_size,
        false,
        down,
    )
}

/// Bind the fused shared-expert MLP, splitting it across this model's blocks.
fn bind_shared_experts(
    binder: &mut WeightBinder<'_>,
    name: &str,
    hidden_size: usize,
    intermediate_size: usize,
    experts: &mut [DeepSeekV2MLP],
) -> Result<()> {
    let parts = experts.len();
    if parts == 0 {
        return Ok(());
    }
    let fused = parts * intermediate_size;

    let gate_name = format!("{name}.gate_proj.weight");
    if let Some(tensor) = binder.take_shaped(&gate_name, &[fused, hidden_size])? {
        for (expert, block) in experts.iter_mut().zip(split_rows(&gate_name, &tensor, parts)?) {
            expert.projections_mut().0.set_weight(block)?;
        }
    }
    let up_name = format!("{name}.up_proj.weight");
    if let Some(tensor) = binder.take_shaped(&up_name, &[fused, hidden_size])? {
        for (expert, block) in experts.iter_mut().zip(split_rows(&up_name, &tensor, parts)?) {
            expert.projections_mut().1.set_weight(block)?;
        }
    }
    let down_name = format!("{name}.down_proj.weight");
    if let Some(tensor) = binder.take_shaped(&down_name, &[hidden_size, fused])? {
        for (expert, block) in experts.iter_mut().zip(split_columns(&down_name, &tensor, parts)?) {
            expert.projections_mut().2.set_weight(block)?;
        }
    }
    Ok(())
}

/// Bind one layer's multi-head latent attention block.
fn bind_attention(
    binder: &mut WeightBinder<'_>,
    name: &str,
    config: &DeepSeekV2Config,
    attention: &mut MlaAttention,
) -> Result<()> {
    let hidden = config.hidden_size;
    let heads = config.num_attention_heads;
    let qk = config.qk_head_dim();

    match (attention.q_proj_mut(), config.q_lora_rank) {
        (Some(q_proj), _) => {
            bind_linear(
                binder,
                &format!("{name}.q_proj"),
                heads * qk,
                hidden,
                false,
                q_proj,
            )?;
        },
        (None, rank) => {
            let down = attention.q_a_proj_mut().ok_or_else(|| {
                TrustformersError::weight_load_error(
                    "attention has neither q_proj nor q_a_proj".to_string(),
                )
            })?;
            bind_linear(
                binder,
                &format!("{name}.q_a_proj"),
                rank,
                hidden,
                false,
                down,
            )?;
            let norm = attention.q_a_layernorm_mut().ok_or_else(|| {
                TrustformersError::weight_load_error(
                    "a query LoRA path without its q_a_layernorm".to_string(),
                )
            })?;
            bind_rms_norm(binder, &format!("{name}.q_a_layernorm"), rank, norm)?;
            let up = attention.q_b_proj_mut().ok_or_else(|| {
                TrustformersError::weight_load_error(
                    "a query LoRA path without its q_b_proj".to_string(),
                )
            })?;
            bind_linear(
                binder,
                &format!("{name}.q_b_proj"),
                heads * qk,
                rank,
                false,
                up,
            )?;
        },
    }

    bind_linear(
        binder,
        &format!("{name}.kv_a_proj_with_mqa"),
        config.kv_lora_rank + config.qk_rope_head_dim,
        hidden,
        false,
        attention.kv_a_proj_with_mqa_mut(),
    )?;
    bind_rms_norm(
        binder,
        &format!("{name}.kv_a_layernorm"),
        config.kv_lora_rank,
        attention.kv_a_layernorm_mut(),
    )?;
    bind_linear(
        binder,
        &format!("{name}.kv_b_proj"),
        heads * (config.qk_nope_head_dim + config.v_head_dim),
        config.kv_lora_rank,
        false,
        attention.kv_b_proj_mut(),
    )?;
    bind_linear(
        binder,
        &format!("{name}.o_proj"),
        hidden,
        heads * config.v_head_dim,
        false,
        attention.o_proj_mut(),
    )
}

impl DeepSeekV2Model {
    /// Bind an already-parsed checkpoint into this backbone.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint does not carry `embed_tokens.weight` under any
    /// recognised prefix, when a tensor has the wrong shape, when a parameter is
    /// missing, or when the checkpoint holds a tensor this architecture does not
    /// recognise.
    pub fn load_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<LoadReport> {
        let config = self.config().clone();
        let prefix =
            checkpoint
                .detect_prefix(&["", "model."], "embed_tokens.weight")
                .map_err(|error| {
                    TrustformersError::weight_load_error(format!(
                        "{error}. A DeepSeek-V2 checkpoint is expected to carry these tensors: {}",
                        super::loading::expected_tensor_names(&config, "model.").join(", ")
                    ))
                })?;
        let mut binder = checkpoint.binder(&prefix);

        bind_embedding(
            &mut binder,
            "embed_tokens",
            config.vocab_size,
            config.hidden_size,
            self.embed_tokens_mut(),
        )?;

        for (index, layer) in self.layers_mut().iter_mut().enumerate() {
            bind_attention(
                &mut binder,
                &format!("layers.{index}.self_attn"),
                &config,
                layer.self_attn_mut(),
            )?;

            let mlp_name = format!("layers.{index}.mlp");
            if let Some(dense) = layer.dense_mlp_mut() {
                bind_mlp(
                    &mut binder,
                    &mlp_name,
                    config.hidden_size,
                    config.intermediate_size,
                    dense,
                )?;
            } else if let Some(moe) = layer.moe_layer_mut() {
                bind_linear(
                    &mut binder,
                    &format!("{mlp_name}.gate"),
                    config.n_routed_experts,
                    config.hidden_size,
                    false,
                    moe.router_mut().gate_mut(),
                )?;
                for (expert_index, expert) in moe.routed_experts_mut().iter_mut().enumerate() {
                    bind_mlp(
                        &mut binder,
                        &format!("{mlp_name}.experts.{expert_index}"),
                        config.hidden_size,
                        config.intermediate_size,
                        expert,
                    )?;
                }
                bind_shared_experts(
                    &mut binder,
                    &format!("{mlp_name}.shared_experts"),
                    config.hidden_size,
                    config.intermediate_size,
                    moe.shared_experts_mut(),
                )?;
            } else {
                return Err(TrustformersError::weight_load_error(format!(
                    "layer {index} has neither a dense MLP nor a MoE FFN"
                )));
            }

            let (input_norm, post_norm) = layer.norms_mut();
            bind_rms_norm(
                &mut binder,
                &format!("layers.{index}.input_layernorm"),
                config.hidden_size,
                input_norm,
            )?;
            bind_rms_norm(
                &mut binder,
                &format!("layers.{index}.post_attention_layernorm"),
                config.hidden_size,
                post_norm,
            )?;
        }

        bind_rms_norm(
            &mut binder,
            "norm",
            config.hidden_size,
            self.final_norm_mut(),
        )?;
        binder.finish(backbone_unused_policy())
    }

    /// Load pretrained weights and report exactly what was bound.
    ///
    /// # Errors
    ///
    /// See [`DeepSeekV2Model::load_from_checkpoint`], plus any container-parsing
    /// failure.
    pub fn load_pretrained_report(&mut self, reader: &mut dyn std::io::Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_from_checkpoint(&checkpoint)
    }
}

/// Bind a language-modelling head off an already-parsed checkpoint.
///
/// Absent is legitimate — a backbone export carries no head — and is recorded in
/// [`LoadReport::missing`] rather than invented.
///
/// # Errors
///
/// Fails when the tensor is present with the wrong shape.
pub(super) fn bind_lm_head(
    checkpoint: &Checkpoint,
    report: &mut LoadReport,
    vocab_size: usize,
    hidden_size: usize,
    head: &mut Linear,
) -> Result<()> {
    let name = "lm_head.weight";
    match checkpoint.take_shaped(name, &[vocab_size, hidden_size])? {
        Some(weight) => {
            head.set_weight(weight)?;
            report.mark_loaded(name);
        },
        None => report.note_absent(name),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek_v2::config::{ActivationType, TopKMethod};
    use crate::deepseek_v2::tasks::DeepSeekV2ForCausalLM;
    use crate::weight_loading::test_support::{build_safetensors, write_temp_file, F32Tensor};
    use std::fs::File;
    use trustformers_core::traits::Model;

    /// Deterministic small values, distinct per tensor name, that keep a two-layer
    /// forward pass numerically tame.
    fn values(name: &str, count: usize) -> Vec<f32> {
        let mut state = name.bytes().fold(0x2545_f491_4f6c_dd1d_u64, |acc, byte| {
            acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(u64::from(byte))
        }) | 1;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / 16_777_216.0) * 0.2 - 0.1
            })
            .collect()
    }

    fn tensor(name: &str, shape: &[usize]) -> F32Tensor {
        let count: usize = shape.iter().product();
        F32Tensor::new(name, shape, values(name, count))
    }

    fn tiny_config() -> DeepSeekV2Config {
        DeepSeekV2Config {
            vocab_size: 24,
            hidden_size: 16,
            intermediate_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            kv_lora_rank: 6,
            q_lora_rank: 8,
            qk_rope_head_dim: 4,
            qk_nope_head_dim: 4,
            v_head_dim: 4,
            num_experts_per_tok: 2,
            n_routed_experts: 3,
            n_shared_experts: 2,
            routed_scaling_factor: 1.0,
            topk_method: TopKMethod::Noaux,
            n_group: 1,
            topk_group: 3,
            aux_loss_alpha: 0.001,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            hidden_act: ActivationType::SiLU,
            initializer_range: 0.02,
            first_k_dense_replace: 1,
            moe_layer_freq: 1,
        }
    }

    /// Every tensor a DeepSeek-V2 export of this shape carries, HuggingFace names.
    fn fixture(config: &DeepSeekV2Config, prefix: &str, with_head: bool) -> Vec<F32Tensor> {
        let hidden = config.hidden_size;
        let heads = config.num_attention_heads;
        let qk = config.qk_head_dim();
        let mut tensors = vec![tensor(
            &format!("{prefix}embed_tokens.weight"),
            &[config.vocab_size, hidden],
        )];
        for layer in 0..config.num_hidden_layers {
            let attn = format!("{prefix}layers.{layer}.self_attn");
            if config.q_lora_rank == 0 {
                tensors.push(tensor(
                    &format!("{attn}.q_proj.weight"),
                    &[heads * qk, hidden],
                ));
            } else {
                tensors.push(tensor(
                    &format!("{attn}.q_a_proj.weight"),
                    &[config.q_lora_rank, hidden],
                ));
                tensors.push(tensor(
                    &format!("{attn}.q_a_layernorm.weight"),
                    &[config.q_lora_rank],
                ));
                tensors.push(tensor(
                    &format!("{attn}.q_b_proj.weight"),
                    &[heads * qk, config.q_lora_rank],
                ));
            }
            tensors.push(tensor(
                &format!("{attn}.kv_a_proj_with_mqa.weight"),
                &[config.kv_lora_rank + config.qk_rope_head_dim, hidden],
            ));
            tensors.push(tensor(
                &format!("{attn}.kv_a_layernorm.weight"),
                &[config.kv_lora_rank],
            ));
            tensors.push(tensor(
                &format!("{attn}.kv_b_proj.weight"),
                &[
                    heads * (config.qk_nope_head_dim + config.v_head_dim),
                    config.kv_lora_rank,
                ],
            ));
            tensors.push(tensor(
                &format!("{attn}.o_proj.weight"),
                &[hidden, heads * config.v_head_dim],
            ));

            let mlp = format!("{prefix}layers.{layer}.mlp");
            let intermediate = config.intermediate_size;
            if config.is_dense_layer(layer) {
                tensors.push(tensor(
                    &format!("{mlp}.gate_proj.weight"),
                    &[intermediate, hidden],
                ));
                tensors.push(tensor(
                    &format!("{mlp}.up_proj.weight"),
                    &[intermediate, hidden],
                ));
                tensors.push(tensor(
                    &format!("{mlp}.down_proj.weight"),
                    &[hidden, intermediate],
                ));
            } else {
                tensors.push(tensor(
                    &format!("{mlp}.gate.weight"),
                    &[config.n_routed_experts, hidden],
                ));
                for expert in 0..config.n_routed_experts {
                    let name = format!("{mlp}.experts.{expert}");
                    tensors.push(tensor(
                        &format!("{name}.gate_proj.weight"),
                        &[intermediate, hidden],
                    ));
                    tensors.push(tensor(
                        &format!("{name}.up_proj.weight"),
                        &[intermediate, hidden],
                    ));
                    tensors.push(tensor(
                        &format!("{name}.down_proj.weight"),
                        &[hidden, intermediate],
                    ));
                }
                let fused = config.n_shared_experts * intermediate;
                let shared = format!("{mlp}.shared_experts");
                tensors.push(tensor(
                    &format!("{shared}.gate_proj.weight"),
                    &[fused, hidden],
                ));
                tensors.push(tensor(
                    &format!("{shared}.up_proj.weight"),
                    &[fused, hidden],
                ));
                tensors.push(tensor(
                    &format!("{shared}.down_proj.weight"),
                    &[hidden, fused],
                ));
            }
            tensors.push(tensor(
                &format!("{prefix}layers.{layer}.input_layernorm.weight"),
                &[hidden],
            ));
            tensors.push(tensor(
                &format!("{prefix}layers.{layer}.post_attention_layernorm.weight"),
                &[hidden],
            ));
        }
        tensors.push(tensor(&format!("{prefix}norm.weight"), &[hidden]));
        if with_head {
            tensors.push(tensor("lm_head.weight", &[config.vocab_size, hidden]));
        }
        tensors
    }

    fn input_ids() -> Tensor {
        Tensor::from_vec(vec![1.0_f32, 5.0, 9.0, 2.0], &[4]).expect("input tensor must build")
    }

    fn logits(model: &DeepSeekV2ForCausalLM) -> Vec<f32> {
        model
            .forward(input_ids())
            .expect("forward must succeed")
            .to_vec_f32()
            .expect("logits must be F32")
    }

    /// Every tensor in a synthetic HuggingFace-named checkpoint lands in a
    /// parameter, read back from a real file under `std::env::temp_dir()`.
    #[test]
    fn a_synthetic_checkpoint_round_trips_through_a_real_file() {
        let config = tiny_config();
        let tensors = fixture(&config, "model.", true);
        let path = write_temp_file(
            "deepseek_v2_round_trip",
            "safetensors",
            &build_safetensors(&tensors),
        );
        let mut file = File::open(&path).expect("the fixture file must open");

        let mut model = DeepSeekV2ForCausalLM::new(config.clone()).expect("model must build");
        let report = model
            .load_pretrained_report(&mut file)
            .expect("a complete DeepSeek-V2 checkpoint must load");
        let _ = std::fs::remove_file(&path);

        assert!(
            report.is_complete(),
            "every parameter must be filled, missing: {:?}",
            report.missing
        );
        assert!(
            report.unexpected.is_empty(),
            "nothing may be left unrecognised: {:?}",
            report.unexpected
        );
        for name in expected_tensor_names(&config, "model.") {
            assert!(
                report.loaded.iter().any(|loaded| loaded == &name),
                "{name} must be reported as loaded"
            );
        }
        assert!(
            report.loaded.iter().any(|loaded| loaded == "lm_head.weight"),
            "the language-modelling head must be bound too: {:?}",
            report.loaded
        );
        assert_eq!(
            report.loaded.len(),
            tensors.len(),
            "exactly the checkpoint's tensors must be accounted for"
        );

        let produced = logits(&model);
        assert!(
            produced.iter().all(|value| value.is_finite()),
            "a loaded model must produce finite logits"
        );
    }

    /// The fused shared-expert projections are split across this model's blocks
    /// rather than dropped, and the halves land in the right experts.
    #[test]
    fn the_fused_shared_experts_are_split_across_the_blocks() {
        let config = tiny_config();
        let tensors = fixture(&config, "model.", false);
        let checkpoint =
            Checkpoint::from_bytes(&build_safetensors(&tensors)).expect("checkpoint must parse");

        let mut model = DeepSeekV2Model::new(config.clone()).expect("model must build");
        model.load_from_checkpoint(&checkpoint).expect("the backbone must load");

        let fused_name = "model.layers.1.mlp.shared_experts.gate_proj.weight";
        let fused = tensors
            .iter()
            .find(|entry| entry.name == fused_name)
            .expect("the fixture must carry the fused shared-expert gate");
        let block = config.intermediate_size * config.hidden_size;

        let down_name = "model.layers.1.mlp.shared_experts.down_proj.weight";
        let fused_down = tensors
            .iter()
            .find(|entry| entry.name == down_name)
            .expect("the fixture must carry the fused shared-expert down projection");
        let intermediate = config.intermediate_size;
        let parts = config.n_shared_experts;
        let fused_cols = parts * intermediate;

        let layer = &mut model.layers_mut()[1];
        let moe = layer.moe_layer_mut().expect("layer 1 is a MoE layer");
        for (index, expert) in moe.shared_experts_mut().iter_mut().enumerate() {
            let (gate, _, down) = expert.projections_mut();
            let bound = gate.weight().to_vec_f32().expect("readable");
            assert_eq!(
                bound,
                fused.values[index * block..(index + 1) * block].to_vec(),
                "shared expert {index} must hold row block {index} of the fused gate projection"
            );

            // `down_proj` is `[hidden_size, n_shared * intermediate]`, so the
            // split is by *column*, not by row — the one place a strided copy is
            // needed, and the one a wrong index would survive: a misindexed
            // version still produces finite, different-from-baseline output and
            // would pass every other assertion in this module.
            let bound_down = down.weight().to_vec_f32().expect("readable");
            assert_eq!(
                bound_down.len(),
                config.hidden_size * intermediate,
                "each shared expert's down projection is one column block"
            );
            for row in 0..config.hidden_size {
                let want_start = row * fused_cols + index * intermediate;
                assert_eq!(
                    bound_down[row * intermediate..(row + 1) * intermediate].to_vec(),
                    fused_down.values[want_start..want_start + intermediate].to_vec(),
                    "shared expert {index}, row {row}: down_proj must be column block {index} \
                     of the fused tensor"
                );
            }
        }
    }

    /// Changing one checkpoint tensor changes the model's output — proof the
    /// binder installs values rather than validating and discarding them.
    #[test]
    fn a_tampered_tensor_changes_the_forward_output() {
        let config = tiny_config();
        let tensors = fixture(&config, "model.", true);

        let mut baseline = DeepSeekV2ForCausalLM::new(config.clone()).expect("model must build");
        baseline
            .load_pretrained(&mut build_safetensors(&tensors).as_slice())
            .expect("the pristine checkpoint must load");
        let before = logits(&baseline);

        // Every name here is on the unconditional path: layer 0 is a dense FFN
        // layer and the shared experts of layer 1 are always active. A *routed*
        // expert is covered separately, by a config that activates all of them —
        // top-k routing legitimately leaves some experts unused for a given
        // input, so tampering one of those proves nothing either way.
        for target in [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.kv_a_layernorm.weight",
            "model.layers.0.self_attn.q_a_layernorm.weight",
            "model.layers.0.self_attn.kv_b_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.mlp.shared_experts.down_proj.weight",
            "model.layers.1.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ] {
            let mut tampered = tensors.clone();
            let entry = tampered
                .iter_mut()
                .find(|entry| entry.name == target)
                .unwrap_or_else(|| panic!("{target} must be in the fixture"));
            for value in &mut entry.values {
                *value += 0.75;
            }

            let mut model = DeepSeekV2ForCausalLM::new(config.clone()).expect("model must build");
            model
                .load_pretrained(&mut build_safetensors(&tampered).as_slice())
                .expect("the tampered checkpoint must still load");
            assert_ne!(
                logits(&model),
                before,
                "changing {target} must change the model's output"
            );
        }
    }

    /// Each routed expert's weights reach the model, shown with a config whose
    /// top-k covers every expert so none can be skipped by the router.
    #[test]
    fn a_tampered_routed_expert_changes_the_output() {
        let mut config = tiny_config();
        config.num_experts_per_tok = config.n_routed_experts;
        config.topk_group = config.n_routed_experts;
        let tensors = fixture(&config, "model.", true);

        let mut baseline = DeepSeekV2ForCausalLM::new(config.clone()).expect("model must build");
        baseline
            .load_pretrained(&mut build_safetensors(&tensors).as_slice())
            .expect("the pristine checkpoint must load");
        let before = logits(&baseline);

        for expert in 0..config.n_routed_experts {
            let target = format!("model.layers.1.mlp.experts.{expert}.up_proj.weight");
            let mut tampered = tensors.clone();
            let entry = tampered
                .iter_mut()
                .find(|entry| entry.name == target)
                .unwrap_or_else(|| panic!("{target} must be in the fixture"));
            for value in &mut entry.values {
                *value += 0.75;
            }

            let mut model = DeepSeekV2ForCausalLM::new(config.clone()).expect("model must build");
            model
                .load_pretrained(&mut build_safetensors(&tampered).as_slice())
                .expect("the tampered checkpoint must still load");
            assert_ne!(
                logits(&model),
                before,
                "changing {target} must change the model's output"
            );
        }
    }

    /// A tensor this architecture does not recognise fails the load instead of
    /// being silently ignored.
    #[test]
    fn an_unknown_tensor_fails_the_load() {
        let config = tiny_config();
        let mut tensors = fixture(&config, "model.", true);
        tensors.push(tensor(
            "model.layers.0.self_attn.kv_b_prj.weight",
            &[config.hidden_size],
        ));

        let mut model = DeepSeekV2ForCausalLM::new(config).expect("model must build");
        let error = model
            .load_pretrained(&mut build_safetensors(&tensors).as_slice())
            .expect_err("a misspelt tensor name must not be tolerated");
        assert!(
            error.to_string().contains("kv_b_prj"),
            "the offending name must be reported: {error}"
        );
    }

    /// A missing parameter is reported rather than left silently at its
    /// constructor initialisation.
    #[test]
    fn a_missing_parameter_fails_the_load() {
        let config = tiny_config();
        let tensors: Vec<F32Tensor> = fixture(&config, "model.", true)
            .into_iter()
            .filter(|entry| entry.name != "model.layers.1.self_attn.kv_a_layernorm.weight")
            .collect();

        let mut model = DeepSeekV2ForCausalLM::new(config).expect("model must build");
        let error = model
            .load_pretrained(&mut build_safetensors(&tensors).as_slice())
            .expect_err("an absent latent norm must fail the load");
        assert!(
            error.to_string().contains("kv_a_layernorm"),
            "the missing name must be reported: {error}"
        );
    }

    /// A bare-backbone export (no `lm_head.weight`) still loads; the absent head
    /// is reported rather than invented.
    #[test]
    fn a_backbone_only_checkpoint_reports_the_absent_head() {
        let config = tiny_config();
        let tensors = fixture(&config, "model.", false);
        let mut model = DeepSeekV2ForCausalLM::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut build_safetensors(&tensors).as_slice())
            .expect("a backbone-only checkpoint must load");
        assert!(
            report.missing.iter().any(|name| name == "lm_head.weight"),
            "the absent head must be named: {:?}",
            report.missing
        );
    }

    /// A backbone export written without the `model.` prefix loads too.
    #[test]
    fn the_backbone_prefix_is_detected() {
        let config = tiny_config();
        let tensors = fixture(&config, "", false);
        let mut model = DeepSeekV2Model::new(config.clone()).expect("model must build");
        let report = model
            .load_pretrained_report(&mut build_safetensors(&tensors).as_slice())
            .expect("an unprefixed backbone export must load");
        assert!(report.is_complete(), "missing: {:?}", report.missing);
        assert_eq!(
            report.loaded.len(),
            expected_tensor_names(&config, "").len()
        );
    }

    /// A checkpoint that is not DeepSeek-V2 at all is refused with the full
    /// expected-name list, not a bare "not found".
    #[test]
    fn an_unrecognisable_checkpoint_names_the_expected_tensors() {
        let config = tiny_config();
        let bytes = build_safetensors(&[tensor("some.other.model.weight", &[2, 2])]);
        let mut model = DeepSeekV2Model::new(config).expect("model must build");
        let error = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect_err("a foreign checkpoint must be refused");
        let message = error.to_string();
        assert!(
            message.contains("model.embed_tokens.weight") && message.contains("kv_a_proj_with_mqa"),
            "the refusal must enumerate the contract: {message}"
        );
    }

    /// A plain query projection (no LoRA) is bound under `q_proj`, and the LoRA
    /// names are then not part of the contract.
    #[test]
    fn a_config_without_query_lora_binds_q_proj() {
        let mut config = tiny_config();
        config.q_lora_rank = 0;
        let names = expected_tensor_names(&config, "model.");
        assert!(names.iter().any(|name| name == "model.layers.0.self_attn.q_proj.weight"));
        assert!(!names.iter().any(|name| name.contains("q_a_layernorm")));

        let tensors = fixture(&config, "model.", false);
        let mut model = DeepSeekV2Model::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut build_safetensors(&tensors).as_slice())
            .expect("a q_proj-style checkpoint must load");
        assert!(report.is_complete(), "missing: {:?}", report.missing);
    }
}
