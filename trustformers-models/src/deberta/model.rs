use crate::deberta::config::DebertaConfig;
use crate::weight_loading::binding::{
    bind_embedding, bind_linear, take_norm_bias, take_norm_weight,
};
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, UnusedTensors};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, Axis, Ix2, Ix3}; // SciRS2 Integration Policy
use trustformers_core::device::Device;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::layers::{
    embedding::Embedding, feedforward::FeedForward, layernorm::LayerNorm, linear::Linear,
};
use trustformers_core::ops::activations::gelu;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Layer, Model, TokenizedInput};

#[derive(Debug, Clone)]
pub struct DebertaEmbeddings {
    pub word_embeddings: Embedding,
    pub layer_norm: LayerNorm,
    pub dropout: f32,
    device: Device,
}

impl DebertaEmbeddings {
    pub fn new(config: &DebertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &DebertaConfig, device: Device) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new_with_device(
                config.vocab_size,
                config.hidden_size,
                Some(config.pad_token_id as usize),
                device,
            )?,
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, input_ids: &Array1<u32>) -> Result<Array2<f32>> {
        // Word embeddings
        let input_ids_slice = input_ids.as_slice().ok_or_else(|| {
            TrustformersError::tensor_op_error("forward", "input_ids is not contiguous in memory")
        })?;
        let embeddings = self.word_embeddings.forward_ids(input_ids_slice)?;
        let embeddings_2d = match embeddings {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor for word embeddings",
                    "embeddings",
                ))
            },
        };

        // Layer normalization
        let norm_input = Tensor::F32(embeddings_2d.clone().into_dyn());
        let embeddings = self.layer_norm.forward(norm_input)?;
        let embeddings_2d = match embeddings {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor after layer norm",
                    "embeddings",
                ))
            },
        };

        // Apply dropout (simplified - in training mode would be stochastic)
        Ok(embeddings_2d * (1.0 - self.dropout))
    }
}

/// DeBERTa's disentangled self-attention.
///
/// The DeBERTa paper (He et al., 2021, §3.1) decomposes the attention logit
/// between token `i` and token `j` into three additive terms:
///
/// ```text
/// A[i, j] = Q_c[i]·K_c[j]      (content → content)
///         + Q_c[i]·K_r[δ(i,j)] (content → position, "c2p")
///         + K_c[j]·Q_r[δ(j,i)] (position → content, "p2c")
/// ```
///
/// where `δ(i, j)` is the bucketed relative distance and `K_r` / `Q_r` are the
/// key/query projections of a *learned relative-position embedding table*. The
/// sum is scaled by `1/sqrt(3 · d)` rather than `1/sqrt(d)`, because three terms
/// of comparable magnitude are being added.
///
/// # What this replaces
///
/// A previous revision computed `pos_query_proj.forward(hidden_states)` and
/// immediately discarded the result into `let _pos_query_layer = …`, then added
/// `relative_pos[i, j] as f32 * 0.01` to every head's logits — a scalar that is
/// *linear in the signed distance*, identical across heads, and completely
/// independent of the projections it had just computed. Under it, token 5
/// attending to token 0 always got exactly `+0.05`, whatever the content. That
/// is not disentangled attention; it is a hand-written linear position bias with
/// no learned parameters, and it made `p2c` and `c2p` numerically identical.
#[derive(Debug, Clone)]
pub struct DebertaDisentangledSelfAttention {
    pub query_proj: Linear,
    pub key_proj: Linear,
    pub value_proj: Linear,
    pub pos_query_proj: Option<Linear>, // For content-to-position attention
    pub pos_key_proj: Option<Linear>,   // For position-to-content attention
    pub pos_proj: Option<Linear>,       // Position embeddings projection
    /// Learned relative-position embedding table, `[2 * span, hidden_size]`.
    ///
    /// Row `k` holds the embedding of relative distance `k - span`, so the table
    /// covers `[-span, span)`. This is the `P` matrix of the paper; without it
    /// there is nothing for `pos_query_proj` / `pos_key_proj` to project, which
    /// is why the previous revision had no choice but to invent a scalar.
    pub rel_embeddings: Array2<f32>,
    pub dropout: f32,
    pub num_attention_heads: usize,
    pub attention_head_size: usize,
    pub all_head_size: usize,
    pub max_relative_positions: i32,
    /// Half-width of the relative-position window, always > 0.
    pub position_buckets: usize,
    pub pos_att_type: Vec<String>,
    pub share_att_key: bool,
    device: Device,
}

impl DebertaDisentangledSelfAttention {
    pub fn new(config: &DebertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &DebertaConfig, device: Device) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;

        let pos_query_proj = if config.pos_att_type.contains(&"c2p".to_string()) {
            Some(Linear::new_with_device(
                config.hidden_size,
                all_head_size,
                true,
                device,
            ))
        } else {
            None
        };

        let pos_key_proj =
            if config.pos_att_type.contains(&"p2c".to_string()) && !config.share_att_key {
                Some(Linear::new_with_device(
                    config.hidden_size,
                    all_head_size,
                    true,
                    device,
                ))
            } else {
                None
            };

        let pos_proj = if config.max_relative_positions > 0 {
            Some(Linear::new_with_device(
                config.max_relative_positions as usize * 2,
                all_head_size,
                false,
                device,
            ))
        } else {
            None
        };

        // `max_relative_positions = -1` means "no explicit limit", in which case
        // HuggingFace falls back to `max_position_embeddings`. The window is
        // always a concrete positive number here, because the relative-position
        // embedding table needs a row count.
        let position_buckets = if config.max_relative_positions > 0 {
            config.max_relative_positions as usize
        } else {
            config.max_position_embeddings
        }
        .max(1);

        // The relative-position table is a real learned parameter. It is
        // initialised deterministically (a small, distinct value per row and
        // column) so that an un-loaded model is reproducible rather than random,
        // and it is overwritten wholesale by
        // `DebertaDisentangledSelfAttention::set_rel_embeddings` when a
        // checkpoint supplies `rel_embeddings.weight`.
        let rows = position_buckets * 2;
        let scale = config.initializer_range;
        let rel_embeddings = Array2::from_shape_fn((rows, config.hidden_size), |(r, c)| {
            let phase = (r as f32 * 0.7 + c as f32 * 0.13).sin();
            phase * scale
        });

        Ok(Self {
            query_proj: Linear::new_with_device(config.hidden_size, all_head_size, true, device),
            key_proj: Linear::new_with_device(config.hidden_size, all_head_size, true, device),
            value_proj: Linear::new_with_device(config.hidden_size, all_head_size, true, device),
            pos_query_proj,
            pos_key_proj,
            pos_proj,
            rel_embeddings,
            dropout: config.attention_probs_dropout_prob,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
            max_relative_positions: config.max_relative_positions,
            position_buckets,
            pos_att_type: config.pos_att_type.clone(),
            share_att_key: config.share_att_key,
            device,
        })
    }

    /// Install a relative-position embedding table from a checkpoint.
    ///
    /// # Errors
    ///
    /// Fails when `table` does not have the `[2 * position_buckets, hidden]`
    /// shape this attention block was configured for.
    pub fn set_rel_embeddings(&mut self, table: Array2<f32>) -> Result<()> {
        if table.dim() != self.rel_embeddings.dim() {
            return Err(TrustformersError::shape_error(format!(
                "DeBERTa relative-position table must be {:?}, got {:?}",
                self.rel_embeddings.dim(),
                table.dim()
            )));
        }
        self.rel_embeddings = table;
        Ok(())
    }

    /// Bucket index into [`Self::rel_embeddings`] for the distance `query - key`.
    ///
    /// The signed distance is clamped to `[-span, span - 1]` and shifted so that
    /// distance `-span` maps to row 0 and distance `span - 1` to the last row.
    fn relative_bucket(&self, query_index: usize, key_index: usize) -> usize {
        let span = self.position_buckets as i64;
        let distance = query_index as i64 - key_index as i64;
        let clamped = distance.clamp(-span, span - 1);
        (clamped + span) as usize
    }

    /// Project the relative-position table through a `Linear` and split it into
    /// per-head `[rows, head_size]` slices.
    fn project_relative(&self, projection: &Linear) -> Result<Array3<f32>> {
        let rows = self.rel_embeddings.nrows();
        let input = Tensor::F32(self.rel_embeddings.clone().into_dyn());
        let projected = projection.forward(input)?;
        let projected = match projected {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from the relative-position projection",
                    "attention",
                ))
            },
        };
        // [rows, all_head_size] -> [rows, num_heads, head_size]
        projected
            .to_shape((rows, self.num_attention_heads, self.attention_head_size))
            .map(|view| view.to_owned())
            .map_err(|e| {
                TrustformersError::shape_error(format!(
                    "relative-position projection has the wrong width: {e}"
                ))
            })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn transpose_for_scores(&self, x: &Array3<f32>) -> Result<Array4<f32>> {
        let (batch_size, seq_len, _) = x.dim();

        // Reshape to (batch_size, seq_len, num_heads, head_size)
        let reshaped = x
            .to_shape((
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            ))
            .map_err(|e| {
                TrustformersError::shape_error(format!("Failed to reshape tensor: {}", e))
            })?
            .to_owned();

        // Transpose to (batch_size, num_heads, seq_len, head_size)
        Ok(reshaped.permuted_axes([0, 2, 1, 3]))
    }

    /// The signed, clamped relative distance matrix `δ(i, j) = i − j`.
    ///
    /// Exposed for inspection and testing; the attention logits are built from
    /// [`Self::relative_bucket`], which maps the same distances onto rows of the
    /// learned embedding table.
    pub fn build_relative_position(&self, query_size: usize, key_size: usize) -> Array2<i32> {
        let mut relative_positions = Array2::zeros((query_size, key_size));

        for i in 0..query_size {
            for j in 0..key_size {
                let relative_pos = i as i32 - j as i32;

                // Clamp to max_relative_positions range
                let clamped_pos = if self.max_relative_positions > 0 {
                    relative_pos.clamp(-self.max_relative_positions, self.max_relative_positions)
                } else {
                    relative_pos
                };

                relative_positions[[i, j]] = clamped_pos;
            }
        }

        relative_positions
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array3<f32>> {
        let (batch_size, seq_len, _hidden_size) = hidden_states.dim();

        // Content-to-content attention
        let query_input = Tensor::F32(hidden_states.clone().into_dyn());
        let key_input = Tensor::F32(hidden_states.clone().into_dyn());
        let value_input = Tensor::F32(hidden_states.clone().into_dyn());

        let query_layer = self.query_proj.forward(query_input)?;
        let key_layer = self.key_proj.forward(key_input)?;
        let value_layer = self.value_proj.forward(value_input)?;

        let query_layer = match query_layer {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from query projection",
                    "attention",
                ))
            },
        };
        let key_layer = match key_layer {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from key projection",
                    "attention",
                ))
            },
        };
        let value_layer = match value_layer {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from value projection",
                    "attention",
                ))
            },
        };

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        // Compute attention scores
        let mut attention_scores =
            Array4::zeros((batch_size, self.num_attention_heads, seq_len, seq_len));

        // The paper divides by sqrt(3 * d) when all three terms are present,
        // because it sums three dot products of comparable magnitude. Scale by
        // the number of terms this configuration actually enables, so a
        // content-only model still gets the usual 1/sqrt(d).
        let use_c2p = self.pos_att_type.iter().any(|t| t == "c2p");
        let use_p2c = self.pos_att_type.iter().any(|t| t == "p2c");
        let num_terms = 1 + usize::from(use_c2p) + usize::from(use_p2c);
        let scale = 1.0 / ((num_terms * self.attention_head_size) as f32).sqrt();

        // Term 1: content → content, Q_c[i] · K_c[j].
        for b in 0..batch_size {
            for h in 0..self.num_attention_heads {
                let q = query_layer.slice(s![b, h, .., ..]);
                let k = key_layer.slice(s![b, h, .., ..]);

                // Compute dot product attention
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let score: f32 = q
                            .slice(s![i, ..])
                            .iter()
                            .zip(k.slice(s![j, ..]).iter())
                            .map(|(a, b)| a * b)
                            .sum();

                        attention_scores[[b, h, i, j]] = score * scale;
                    }
                }
            }
        }

        // Term 2: content → position, Q_c[i] · K_r[δ(i, j)].
        //
        // `pos_key_proj` is the relative-position *key* projection. DeBERTa's
        // `share_att_key` reuses the content key projection for it, which is why
        // no separate `pos_key_proj` is constructed in that case — a genuine
        // weight sharing, not a stand-in.
        if use_c2p {
            let key_projection =
                match (&self.pos_key_proj, self.share_att_key) {
                    (Some(proj), _) => proj,
                    (None, true) => &self.key_proj,
                    (None, false) => return Err(TrustformersError::model_error(
                        "DeBERTa is configured for c2p attention without share_att_key, but no \
                         relative-position key projection was built"
                            .to_string(),
                    )),
                };
            let rel_keys = self.project_relative(key_projection)?;

            for b in 0..batch_size {
                for h in 0..self.num_attention_heads {
                    let q = query_layer.slice(s![b, h, .., ..]);
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            let bucket = self.relative_bucket(i, j);
                            let mut term = 0.0f32;
                            for d in 0..self.attention_head_size {
                                term += q[[i, d]] * rel_keys[[bucket, h, d]];
                            }
                            attention_scores[[b, h, i, j]] += term * scale;
                        }
                    }
                }
            }
        }

        // Term 3: position → content, K_c[j] · Q_r[δ(j, i)].
        //
        // Note the *reversed* distance: the p2c term asks "how does position j
        // relate to the content at i", so its bucket is `δ(j, i)`, the negation
        // of the c2p bucket. Using the same bucket for both — as the previous
        // scalar bias did — collapses the two terms into one.
        if use_p2c {
            let query_projection =
                match (&self.pos_query_proj, self.share_att_key) {
                    (Some(proj), _) => proj,
                    (None, true) => &self.query_proj,
                    (None, false) => return Err(TrustformersError::model_error(
                        "DeBERTa is configured for p2c attention without share_att_key, but no \
                         relative-position query projection was built"
                            .to_string(),
                    )),
                };
            let rel_queries = self.project_relative(query_projection)?;

            for b in 0..batch_size {
                for h in 0..self.num_attention_heads {
                    let k = key_layer.slice(s![b, h, .., ..]);
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            let bucket = self.relative_bucket(j, i);
                            let mut term = 0.0f32;
                            for d in 0..self.attention_head_size {
                                term += k[[j, d]] * rel_queries[[bucket, h, d]];
                            }
                            attention_scores[[b, h, i, j]] += term * scale;
                        }
                    }
                }
            }
        }

        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            // Expand mask to match attention_scores dimensions
            for b in 0..batch_size {
                for h in 0..self.num_attention_heads {
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            if mask[[b, i, j]] == 0.0 {
                                attention_scores[[b, h, i, j]] = -10000.0; // Large negative value
                            }
                        }
                    }
                }
            }
        }

        // Apply softmax to get attention probabilities
        let mut attention_probs =
            Array4::zeros((batch_size, self.num_attention_heads, seq_len, seq_len));

        for b in 0..batch_size {
            for h in 0..self.num_attention_heads {
                for i in 0..seq_len {
                    // Softmax over the last dimension
                    let mut max_val = f32::NEG_INFINITY;
                    for j in 0..seq_len {
                        max_val = max_val.max(attention_scores[[b, h, i, j]]);
                    }

                    let mut sum_exp = 0.0;
                    for j in 0..seq_len {
                        let exp_val = (attention_scores[[b, h, i, j]] - max_val).exp();
                        attention_probs[[b, h, i, j]] = exp_val;
                        sum_exp += exp_val;
                    }

                    for j in 0..seq_len {
                        attention_probs[[b, h, i, j]] /= sum_exp;
                    }
                }
            }
        }

        // Apply dropout (simplified)
        attention_probs *= 1.0 - self.dropout;

        // Apply attention to values
        let mut context_layer = Array4::zeros((
            batch_size,
            self.num_attention_heads,
            seq_len,
            self.attention_head_size,
        ));

        for b in 0..batch_size {
            for h in 0..self.num_attention_heads {
                for i in 0..seq_len {
                    for d in 0..self.attention_head_size {
                        let mut sum = 0.0;
                        for j in 0..seq_len {
                            sum += attention_probs[[b, h, i, j]] * value_layer[[b, h, j, d]];
                        }
                        context_layer[[b, h, i, d]] = sum;
                    }
                }
            }
        }

        // Transpose back to (batch_size, seq_len, num_heads, head_size)
        let context_layer = context_layer.permuted_axes([0, 2, 1, 3]);

        // Reshape to (batch_size, seq_len, all_head_size).
        //
        // `permuted_axes` leaves the array non-contiguous, so `to_shape` can
        // legitimately fail; reporting that is the contract of this
        // `Result`-returning function, not a reason to panic.
        let context_layer = context_layer
            .to_shape((batch_size, seq_len, self.all_head_size))
            .map_err(|e| {
                TrustformersError::shape_error(format!(
                    "failed to merge DeBERTa attention heads back into \
                     [{batch_size}, {seq_len}, {}]: {e}",
                    self.all_head_size
                ))
            })?
            .to_owned();

        Ok(context_layer)
    }
}

#[derive(Debug, Clone)]
pub struct DebertaSelfOutput {
    pub dense: Linear,
    pub layer_norm: LayerNorm,
    pub dropout: f32,
    device: Device,
}

impl DebertaSelfOutput {
    pub fn new(config: &DebertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &DebertaConfig, device: Device) -> Result<Self> {
        Ok(Self {
            dense: Linear::new_with_device(config.hidden_size, config.hidden_size, true, device),
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        input_tensor: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        let dense_input = Tensor::F32(hidden_states.clone().into_dyn());
        let dense_output = self.dense.forward(dense_input)?;
        let hidden_states = match dense_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from dense layer",
                    "dense_layer",
                ))
            },
        };
        let hidden_states = hidden_states * (1.0 - self.dropout);
        let residual = hidden_states + input_tensor;
        let norm_input = Tensor::F32(residual.into_dyn());
        let output = self.layer_norm.forward(norm_input)?;
        let output = match output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from layer norm",
                    "layer_norm",
                ))
            },
        };
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct DebertaAttention {
    pub self_attention: DebertaDisentangledSelfAttention,
    pub output: DebertaSelfOutput,
    device: Device,
}

impl DebertaAttention {
    pub fn new(config: &DebertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &DebertaConfig, device: Device) -> Result<Self> {
        Ok(Self {
            self_attention: DebertaDisentangledSelfAttention::new_with_device(config, device)?,
            output: DebertaSelfOutput::new_with_device(config, device)?,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array3<f32>> {
        let self_outputs = self.self_attention.forward(hidden_states, attention_mask)?;
        let attention_output = self.output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

#[derive(Debug, Clone)]
pub struct DebertaLayer {
    pub attention: DebertaAttention,
    pub feed_forward: FeedForward,
    pub output_layer_norm: LayerNorm,
    pub dropout: f32,
    device: Device,
}

impl DebertaLayer {
    pub fn new(config: &DebertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &DebertaConfig, device: Device) -> Result<Self> {
        Ok(Self {
            attention: DebertaAttention::new_with_device(config, device)?,
            feed_forward: FeedForward::new_with_device(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_dropout_prob,
                device,
            ),
            output_layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array3<f32>> {
        // Self-attention
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;

        // Feed-forward with residual connection
        let ff_input = Tensor::F32(attention_output.clone().into_dyn());
        let ff_output = self.feed_forward.forward(ff_input)?;
        let ff_output = match ff_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from feed forward",
                    "feed_forward",
                ))
            },
        };
        let ff_output = ff_output * (1.0 - self.dropout);
        let residual = &attention_output + &ff_output;
        let norm_input = Tensor::F32(residual.into_dyn());
        let output = self.output_layer_norm.forward(norm_input)?;
        let output = match output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from layer norm",
                    "layer_norm",
                ))
            },
        };

        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct DebertaEncoder {
    pub layers: Vec<DebertaLayer>,
    device: Device,
}

impl DebertaEncoder {
    pub fn new(config: &DebertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &DebertaConfig, device: Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(DebertaLayer::new_with_device(config, device)?);
        }

        Ok(Self { layers, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        mut hidden_states: Array3<f32>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array3<f32>> {
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        Ok(hidden_states)
    }
}

#[derive(Debug, Clone)]
pub struct DebertaModel {
    pub embeddings: DebertaEmbeddings,
    pub encoder: DebertaEncoder,
    pub config: DebertaConfig,
    device: Device,
}

impl DebertaModel {
    pub fn new(config: DebertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: DebertaConfig, device: Device) -> Result<Self> {
        Ok(Self {
            embeddings: DebertaEmbeddings::new_with_device(&config, device)?,
            encoder: DebertaEncoder::new_with_device(&config, device)?,
            config,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = DebertaConfig::from_pretrained_name(model_name);
        Self::new(config)
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array3<f32>> {
        // Get embeddings
        let embeddings = self.embeddings.forward(input_ids)?;

        // Convert to 3D for encoder (batch_size=1, seq_len, hidden_size)
        let hidden_states = embeddings.insert_axis(Axis(0));

        // Pass through encoder
        let encoder_output = self.encoder.forward(hidden_states, attention_mask)?;

        Ok(encoder_output)
    }
}

impl Model for DebertaModel {
    type Config = DebertaConfig;
    type Input = TokenizedInput;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let ids = Array1::from_vec(input.input_ids);
        let hidden = DebertaModel::forward(self, &ids, None)?;
        Ok(Tensor::F32(hidden.into_dyn()))
    }

    /// Load a HuggingFace DeBERTa checkpoint (safetensors or `torch.save`).
    ///
    /// See [`DebertaModel::load_from_checkpoint`] for the name map.
    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let hidden = self.config.hidden_size;
        let embeddings = self.config.vocab_size * hidden + 2 * hidden;
        let per_layer = 4 * hidden * hidden
            + 4 * hidden
            + 2 * hidden * self.config.intermediate_size
            + hidden
            + self.config.intermediate_size
            + 4 * hidden;
        let relative = self.encoder.layers.first().map_or(0, |layer| {
            layer.attention.self_attention.rel_embeddings.len()
        });
        embeddings + self.config.num_hidden_layers * per_layer + relative
    }
}

impl DebertaModel {
    /// Checkpoint namespaces a DeBERTa encoder legitimately does not consume.
    pub(crate) const ALLOWED_UNUSED_PREFIXES: &'static [&'static str] = &[
        "cls.",
        "classifier.",
        "qa_outputs.",
        "pooler.",
        "lm_predictions.",
    ];

    /// Non-parameter buffers HuggingFace stores alongside DeBERTa's weights.
    pub(crate) const ALLOWED_UNUSED_SUFFIXES: &'static [&'static str] =
        &["embeddings.position_ids", "embeddings.token_type_ids"];

    /// Load pretrained weights and report exactly what was bound.
    ///
    /// # Errors
    ///
    /// See [`DebertaModel::load_from_checkpoint`].
    pub fn load_pretrained_report(&mut self, reader: &mut dyn std::io::Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_from_checkpoint(&checkpoint)
    }

    /// Bind an already-parsed checkpoint into this model.
    ///
    /// DeBERTa's tensor layout is BERT's with two additions that matter here:
    /// the **relative-position embedding table**
    /// (`encoder.rel_embeddings.weight`), which the disentangled attention
    /// projects to form its c2p/p2c terms, and — when `share_att_key` is off —
    /// separate `pos_key_proj` / `pos_query_proj` projections. With
    /// `share_att_key` on (the default) those reuse the content projections and
    /// the checkpoint carries no separate tensors for them, which is a genuine
    /// weight sharing rather than a gap.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint does not look like a DeBERTa checkpoint, when
    /// any tensor has the wrong shape, when a parameter is missing, or when the
    /// checkpoint carries weights this architecture does not recognise.
    pub fn load_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<LoadReport> {
        let prefix =
            checkpoint.detect_prefix(&["", "deberta."], "embeddings.word_embeddings.weight")?;
        let mut binder = checkpoint.binder(&prefix);

        let config = self.config.clone();
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;

        bind_embedding(
            &mut binder,
            "embeddings.word_embeddings",
            config.vocab_size,
            hidden,
            &mut self.embeddings.word_embeddings,
        )?;
        if let Some(w) = take_norm_weight(&mut binder, "embeddings.LayerNorm", hidden)? {
            self.embeddings.layer_norm.set_weight(w)?;
        }
        if let Some(b) = take_norm_bias(&mut binder, "embeddings.LayerNorm", hidden)? {
            self.embeddings.layer_norm.set_bias(b)?;
        }

        // The relative-position table is shared by every layer in a HuggingFace
        // DeBERTa export, so it is bound once and copied into each block.
        let rel_name = binder.qualified("encoder.rel_embeddings.weight");
        let shared_rel = match checkpoint.get(&rel_name) {
            Some(tensor) => {
                binder.mark_consumed(&rel_name);
                let expected = self
                    .encoder
                    .layers
                    .first()
                    .map(|layer| layer.attention.self_attention.rel_embeddings.dim());
                let values = tensor.data().map_err(|e| {
                    TrustformersError::tensor_op_error(
                        "failed to read the relative-position table",
                        &e.to_string(),
                    )
                })?;
                match (expected, tensor.shape().as_slice()) {
                    (Some((rows, cols)), [t_rows, t_cols])
                        if rows == *t_rows && cols == *t_cols =>
                    {
                        Some(
                            Array2::from_shape_vec((rows, cols), values)
                                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                        )
                    },
                    (Some(dim), other) => {
                        return Err(TrustformersError::shape_error(format!(
                            "relative-position table has shape {other:?} but this model expects \
                             {dim:?}"
                        )))
                    },
                    (None, _) => None,
                }
            },
            None => None,
        };

        for (index, layer) in self.encoder.layers.iter_mut().enumerate() {
            let base = format!("encoder.layer.{index}");

            bind_linear(
                &mut binder,
                &format!("{base}.attention.self.query_proj"),
                hidden,
                hidden,
                true,
                &mut layer.attention.self_attention.query_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{base}.attention.self.key_proj"),
                hidden,
                hidden,
                true,
                &mut layer.attention.self_attention.key_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{base}.attention.self.value_proj"),
                hidden,
                hidden,
                true,
                &mut layer.attention.self_attention.value_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{base}.attention.output.dense"),
                hidden,
                hidden,
                true,
                &mut layer.attention.output.dense,
            )?;
            let attn_norm = format!("{base}.attention.output.LayerNorm");
            if let Some(w) = take_norm_weight(&mut binder, &attn_norm, hidden)? {
                layer.attention.output.layer_norm.set_weight(w)?;
            }
            if let Some(b) = take_norm_bias(&mut binder, &attn_norm, hidden)? {
                layer.attention.output.layer_norm.set_bias(b)?;
            }

            // Feed-forward: `intermediate.dense` then `output.dense`.
            if let Some(w) = binder.take_shaped(
                &format!("{base}.intermediate.dense.weight"),
                &[intermediate, hidden],
            )? {
                layer.feed_forward.set_dense_weight(w)?;
            }
            if let Some(b) =
                binder.take_shaped(&format!("{base}.intermediate.dense.bias"), &[intermediate])?
            {
                layer.feed_forward.set_dense_bias(b)?;
            }
            if let Some(w) = binder.take_shaped(
                &format!("{base}.output.dense.weight"),
                &[hidden, intermediate],
            )? {
                layer.feed_forward.set_output_weight(w)?;
            }
            if let Some(b) = binder.take_shaped(&format!("{base}.output.dense.bias"), &[hidden])? {
                layer.feed_forward.set_output_bias(b)?;
            }
            let out_norm = format!("{base}.output.LayerNorm");
            if let Some(w) = take_norm_weight(&mut binder, &out_norm, hidden)? {
                layer.output_layer_norm.set_weight(w)?;
            }
            if let Some(b) = take_norm_bias(&mut binder, &out_norm, hidden)? {
                layer.output_layer_norm.set_bias(b)?;
            }

            // Optional per-layer position projections (only when the key is not
            // shared with the content projection).
            if !config.share_att_key {
                if let Some(projection) = layer.attention.self_attention.pos_key_proj.as_mut() {
                    bind_linear(
                        &mut binder,
                        &format!("{base}.attention.self.pos_key_proj"),
                        hidden,
                        hidden,
                        true,
                        projection,
                    )?;
                }
                if let Some(projection) = layer.attention.self_attention.pos_query_proj.as_mut() {
                    bind_linear(
                        &mut binder,
                        &format!("{base}.attention.self.pos_query_proj"),
                        hidden,
                        hidden,
                        true,
                        projection,
                    )?;
                }
            }

            if let Some(table) = shared_rel.as_ref() {
                layer.attention.self_attention.set_rel_embeddings(table.clone())?;
            }
        }

        binder.finish(UnusedTensors::new(
            Self::ALLOWED_UNUSED_PREFIXES,
            Self::ALLOWED_UNUSED_SUFFIXES,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct DebertaForSequenceClassification {
    pub deberta: DebertaModel,
    pub pooler: Linear,
    pub classifier: Linear,
    pub dropout: f32,
    pub num_labels: usize,
    device: Device,
}

impl DebertaForSequenceClassification {
    pub fn new(config: DebertaConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(
        config: DebertaConfig,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let dropout = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);

        Ok(Self {
            deberta: DebertaModel::new_with_device(config.clone(), device)?,
            pooler: Linear::new_with_device(config.hidden_size, config.hidden_size, true, device),
            classifier: Linear::new_with_device(config.hidden_size, num_labels, true, device),
            dropout,
            num_labels,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn from_pretrained(model_name: &str, num_labels: usize) -> Result<Self> {
        let config = DebertaConfig::from_pretrained_name(model_name);
        Self::new(config, num_labels)
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array2<f32>> {
        let hidden_states = self.deberta.forward(input_ids, attention_mask)?;

        // Use [CLS] token representation (first token)
        let cls_hidden = hidden_states.slice(s![0, 0, ..]).to_owned();

        // Pooler
        let pooler_input = Tensor::F32(cls_hidden.insert_axis(Axis(0)).into_dyn());
        let pooled_output = self.pooler.forward(pooler_input)?;
        let pooled_output = match pooled_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from pooler",
                    "pooler",
                ))
            },
        };
        let pooled_tensor = Tensor::F32(pooled_output.into_dyn());
        let pooled_output = gelu(&pooled_tensor)?;
        let pooled_output = match pooled_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from gelu",
                    "gelu",
                ))
            },
        };

        // Apply dropout
        let pooled_output = pooled_output * (1.0 - self.dropout);

        // Classification head
        let classifier_input = Tensor::F32(pooled_output.into_dyn());
        let logits = self.classifier.forward(classifier_input)?;
        let logits = match logits {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from classifier",
                    "classifier",
                ))
            },
        };

        Ok(logits)
    }
}

#[derive(Debug, Clone)]
pub struct DebertaForMaskedLM {
    pub deberta: DebertaModel,
    pub cls: Linear,
    device: Device,
}

impl DebertaForMaskedLM {
    pub fn new(config: DebertaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: DebertaConfig, device: Device) -> Result<Self> {
        Ok(Self {
            deberta: DebertaModel::new_with_device(config.clone(), device)?,
            cls: Linear::new_with_device(config.hidden_size, config.vocab_size, true, device),
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = DebertaConfig::from_pretrained_name(model_name);
        Self::new(config)
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array3<f32>> {
        let hidden_states = self.deberta.forward(input_ids, attention_mask)?;
        let cls_input = Tensor::F32(hidden_states.clone().into_dyn());
        let prediction_scores = self.cls.forward(cls_input)?;
        let prediction_scores = match prediction_scores {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from cls layer",
                    "cls_layer",
                ))
            },
        };
        Ok(prediction_scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deberta::config::DebertaConfig;
    use scirs2_core::ndarray::Array1;
    use trustformers_core::traits::Config;

    /// Minimal DeBERTa config for fast tests.
    fn mini_config() -> DebertaConfig {
        DebertaConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            intermediate_size: 256,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 32,
            type_vocab_size: 0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-7,
            pad_token_id: 0,
            position_embedding_type: "relative_key_query".to_string(),
            use_cache: true,
            classifier_dropout: None,
            relative_attention: true,
            max_relative_positions: -1,
            pos_att_type: vec!["p2c".to_string(), "c2p".to_string()],
            norm_rel_ebd: "layer_norm".to_string(),
            share_att_key: true,
            model_type: "deberta".to_string(),
        }
    }

    fn sample_ids(len: usize) -> Array1<u32> {
        (0..len as u32).collect()
    }

    // ── DebertaEmbeddings ─────────────────────────────────────────────────

    #[test]
    fn test_deberta_embeddings_new_succeeds() {
        let cfg = mini_config();
        DebertaEmbeddings::new(&cfg).expect("DebertaEmbeddings::new should succeed");
    }

    #[test]
    fn test_deberta_embeddings_forward_shape() {
        let cfg = mini_config();
        let emb = DebertaEmbeddings::new(&cfg).expect("DebertaEmbeddings::new failed");
        let ids: Array1<u32> = sample_ids(6);
        let out = emb.forward(&ids).expect("DebertaEmbeddings::forward failed");
        assert_eq!(out.shape(), &[6, cfg.hidden_size]);
    }

    // ── DebertaDisentangledSelfAttention ──────────────────────────────────

    #[test]
    fn test_deberta_disentangled_attention_new_with_relative() {
        let cfg = mini_config();
        DebertaDisentangledSelfAttention::new(&cfg)
            .expect("DebertaDisentangledSelfAttention::new should succeed");
    }

    #[test]
    fn test_deberta_disentangled_attention_new_without_relative() {
        let mut cfg = mini_config();
        cfg.relative_attention = false;
        DebertaDisentangledSelfAttention::new(&cfg)
            .expect("DebertaDisentangledSelfAttention without relative should succeed");
    }

    #[test]
    fn test_deberta_disentangled_attention_pos_att_type_p2c_and_c2p() {
        let cfg = mini_config();
        let attn = DebertaDisentangledSelfAttention::new(&cfg).expect("attention creation failed");
        assert!(
            cfg.pos_att_type.contains(&"p2c".to_string()),
            "pos_att_type should contain 'p2c'"
        );
        assert!(
            cfg.pos_att_type.contains(&"c2p".to_string()),
            "pos_att_type should contain 'c2p'"
        );
        // When relative_attention is true, pos projections should be populated
        assert!(attn.pos_query_proj.is_some() || attn.pos_key_proj.is_some());
    }

    // ── DebertaSelfOutput ─────────────────────────────────────────────────

    #[test]
    fn test_deberta_self_output_new_succeeds() {
        let cfg = mini_config();
        DebertaSelfOutput::new(&cfg).expect("DebertaSelfOutput::new should succeed");
    }

    // ── DebertaAttention ──────────────────────────────────────────────────

    #[test]
    fn test_deberta_attention_new_succeeds() {
        let cfg = mini_config();
        DebertaAttention::new(&cfg).expect("DebertaAttention::new should succeed");
    }

    // ── DebertaLayer ──────────────────────────────────────────────────────

    #[test]
    fn test_deberta_layer_new_succeeds() {
        let cfg = mini_config();
        DebertaLayer::new(&cfg).expect("DebertaLayer::new should succeed");
    }

    // ── DebertaEncoder ────────────────────────────────────────────────────

    #[test]
    fn test_deberta_encoder_new_single_layer() {
        let cfg = mini_config();
        DebertaEncoder::new(&cfg).expect("DebertaEncoder::new should succeed");
    }

    #[test]
    fn test_deberta_encoder_new_multi_layer() {
        let mut cfg = mini_config();
        cfg.num_hidden_layers = 2;
        DebertaEncoder::new(&cfg).expect("DebertaEncoder with 2 layers should succeed");
    }

    // ── DebertaModel ──────────────────────────────────────────────────────

    #[test]
    fn test_deberta_model_new_with_base_config() {
        let cfg = mini_config();
        DebertaModel::new(cfg).expect("DebertaModel::new should succeed");
    }

    #[test]
    fn test_deberta_model_forward_output_shape() {
        let cfg = mini_config();
        let model = DebertaModel::new(cfg.clone()).expect("DebertaModel::new failed");
        let ids: Array1<u32> = sample_ids(5);
        let out = model.forward(&ids, None).expect("DebertaModel::forward failed");
        // Output should be [1 (batch), seq_len, hidden_size]
        assert_eq!(out.shape(), &[1, 5, cfg.hidden_size]);
    }

    #[test]
    fn test_deberta_model_from_pretrained_deberta_base() {
        // from_pretrained uses config presets — no actual weight loading in tests
        let _model = DebertaModel::from_pretrained("deberta-base")
            .expect("from_pretrained deberta-base should succeed");
    }

    #[test]
    fn test_deberta_model_from_pretrained_deberta_large() {
        let _model = DebertaModel::from_pretrained("deberta-large")
            .expect("from_pretrained deberta-large should succeed");
    }

    // ── DeBERTa-v2 config has vocab_size 128100 ───────────────────────────

    #[test]
    fn test_deberta_v2_xlarge_vocab_size() {
        let cfg = DebertaConfig::xlarge();
        assert_eq!(
            cfg.vocab_size, 128100,
            "DeBERTa-v2 xlarge should have vocab_size=128100"
        );
    }

    #[test]
    fn test_deberta_v3_large_vocab_size() {
        let cfg = DebertaConfig::v3_large();
        assert_eq!(
            cfg.vocab_size, 128100,
            "DeBERTa-v3 large should have vocab_size=128100"
        );
    }

    // ── share_att_key default ─────────────────────────────────────────────

    #[test]
    fn test_deberta_default_share_att_key_true() {
        let cfg = DebertaConfig::default();
        assert!(cfg.share_att_key, "share_att_key should default to true");
    }

    // ── DebertaForSequenceClassification ──────────────────────────────────

    #[test]
    fn test_deberta_seq_class_new_two_labels() {
        let cfg = mini_config();
        DebertaForSequenceClassification::new(cfg, 2)
            .expect("DebertaForSequenceClassification with 2 labels failed");
    }

    #[test]
    fn test_deberta_seq_class_forward_output_shape() {
        let cfg = mini_config();
        let model = DebertaForSequenceClassification::new(cfg, 2).expect("model creation failed");
        let ids: Array1<u32> = sample_ids(4);
        let out = model.forward(&ids, None).expect("forward should succeed");
        assert_eq!(out.shape(), &[1, 2]);
    }

    #[test]
    fn test_deberta_seq_class_three_labels_output_shape() {
        let cfg = mini_config();
        let model =
            DebertaForSequenceClassification::new(cfg, 3).expect("model with 3 labels failed");
        let ids: Array1<u32> = sample_ids(4);
        let out = model.forward(&ids, None).expect("forward should succeed");
        assert_eq!(out.shape(), &[1, 3]);
    }

    // ── DebertaForMaskedLM ────────────────────────────────────────────────

    #[test]
    fn test_deberta_masked_lm_new_succeeds() {
        let cfg = mini_config();
        DebertaForMaskedLM::new(cfg).expect("DebertaForMaskedLM::new should succeed");
    }

    #[test]
    fn test_deberta_masked_lm_forward_output_shape() {
        let cfg = mini_config();
        let model = DebertaForMaskedLM::new(cfg.clone()).expect("model creation failed");
        let ids: Array1<u32> = sample_ids(4);
        let out = model.forward(&ids, None).expect("forward should succeed");
        // shape: [1 (batch), seq_len, vocab_size]
        assert_eq!(out.shape(), &[1, 4, cfg.vocab_size]);
    }

    // ── Config validation ─────────────────────────────────────────────────

    #[test]
    fn test_deberta_mini_config_validates() {
        let cfg = mini_config();
        cfg.validate().expect("mini_config should be valid");
    }

    #[test]
    fn test_deberta_base_config_validates() {
        let cfg = DebertaConfig::base();
        cfg.validate().expect("base config should be valid");
    }

    // ── Disentangled attention (regression for the linear-in-distance fake) ──

    fn attention_config(pos_att_type: Vec<String>) -> DebertaConfig {
        DebertaConfig {
            hidden_size: 8,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_relative_positions: 4,
            pos_att_type,
            share_att_key: true,
            ..mini_config()
        }
    }

    /// Deterministic, non-symmetric hidden states so that the c2p and p2c terms
    /// genuinely differ.
    fn hidden(seq_len: usize, hidden_size: usize) -> Array3<f32> {
        Array3::from_shape_fn((1, seq_len, hidden_size), |(_, t, d)| {
            ((t * 7 + d * 3) % 11) as f32 * 0.1 - 0.5
        })
    }

    /// Regression: the c2p and p2c terms were both `relative_pos[i,j] * 0.01`,
    /// so enabling one or the other produced *identical* logits, and enabling
    /// both simply doubled the same scalar. The real terms contract the content
    /// projections against the relative-position table with mirrored buckets, so
    /// they must differ.
    #[test]
    fn c2p_and_p2c_terms_are_different_functions() {
        let seq_len = 5usize;
        let hidden_size = 8usize;
        let states = hidden(seq_len, hidden_size);

        let c2p = DebertaDisentangledSelfAttention::new(&attention_config(vec!["c2p".to_string()]))
            .expect("c2p attention must build");
        let p2c = DebertaDisentangledSelfAttention::new(&attention_config(vec!["p2c".to_string()]))
            .expect("p2c attention must build");

        let c2p_out = c2p.forward(&states, None).expect("c2p forward must succeed");
        let p2c_out = p2c.forward(&states, None).expect("p2c forward must succeed");

        let max_difference = c2p_out
            .iter()
            .zip(p2c_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_difference > 1e-4,
            "content-to-position and position-to-content must be different terms, \
             largest difference was {max_difference}"
        );
    }

    /// The relative-position table is a real parameter: changing it changes the
    /// output. Under the old scalar bias, the table did not exist and the
    /// projections were discarded, so nothing about the positional pathway was
    /// learnable.
    #[test]
    fn the_relative_position_table_affects_the_output() {
        let seq_len = 4usize;
        let hidden_size = 8usize;
        let states = hidden(seq_len, hidden_size);

        let mut attention = DebertaDisentangledSelfAttention::new(&attention_config(vec![
            "c2p".to_string(),
            "p2c".to_string(),
        ]))
        .expect("attention must build");

        let before = attention.forward(&states, None).expect("forward must succeed");

        let (rows, cols) = attention.rel_embeddings.dim();
        let replacement = Array2::from_shape_fn((rows, cols), |(r, c)| {
            ((r * 5 + c * 2) % 9) as f32 * 0.25 - 1.0
        });
        attention
            .set_rel_embeddings(replacement)
            .expect("a correctly-shaped table must be accepted");

        let after = attention.forward(&states, None).expect("forward must succeed");
        let max_difference = before
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_difference > 1e-4,
            "replacing the relative-position table must change the attention output, \
             largest difference was {max_difference}"
        );
    }

    /// The positional pathway must depend on the *content* as well as the
    /// distance. The old bias added the same number for every pair at a given
    /// distance regardless of what the tokens were, so two sequences with
    /// identical shapes but different values received identical position terms.
    #[test]
    fn the_positional_term_depends_on_content_not_only_distance() {
        let seq_len = 4usize;
        let hidden_size = 8usize;
        let attention = DebertaDisentangledSelfAttention::new(&attention_config(vec![
            "c2p".to_string(),
            "p2c".to_string(),
        ]))
        .expect("attention must build");
        let content_only = DebertaDisentangledSelfAttention {
            pos_att_type: Vec::new(),
            ..attention.clone()
        };

        let states_a = hidden(seq_len, hidden_size);
        let states_b = Array3::from_shape_fn((1, seq_len, hidden_size), |(_, t, d)| {
            ((t * 3 + d * 5) % 7) as f32 * 0.2 - 0.7
        });

        let delta = |states: &Array3<f32>| -> f32 {
            let with_pos = attention.forward(states, None).expect("forward must succeed");
            let without = content_only.forward(states, None).expect("forward must succeed");
            with_pos
                .iter()
                .zip(without.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };

        let delta_a = delta(&states_a);
        let delta_b = delta(&states_b);
        assert!(
            delta_a > 1e-4 && delta_b > 1e-4,
            "the positional pathway must move the logits"
        );
        assert!(
            (delta_a - delta_b).abs() > 1e-5,
            "the positional contribution must depend on the content, but two different \
             sequences produced the same shift ({delta_a} vs {delta_b})"
        );
    }

    #[test]
    fn relative_buckets_are_mirrored_between_c2p_and_p2c() {
        let attention = DebertaDisentangledSelfAttention::new(&attention_config(vec![
            "c2p".to_string(),
            "p2c".to_string(),
        ]))
        .expect("attention must build");
        let span = attention.position_buckets;
        // δ(3, 1) = +2 and δ(1, 3) = -2 must land on opposite sides of the table.
        assert_eq!(attention.relative_bucket(3, 1), span + 2);
        assert_eq!(attention.relative_bucket(1, 3), span - 2);
        // Distances beyond the window clamp rather than index out of bounds.
        assert_eq!(attention.relative_bucket(1000, 0), span * 2 - 1);
        assert_eq!(attention.relative_bucket(0, 1000), 0);
    }

    #[test]
    fn a_wrongly_shaped_relative_position_table_is_rejected() {
        let mut attention =
            DebertaDisentangledSelfAttention::new(&attention_config(vec!["c2p".to_string()]))
                .expect("attention must build");
        let err = attention
            .set_rel_embeddings(Array2::zeros((3, 3)))
            .expect_err("a mismatched table must not be installed");
        assert!(
            err.to_string().contains("relative-position table"),
            "unexpected: {err}"
        );
    }
}
