use crate::deberta::config::DebertaConfig;
use scirs2_core::ndarray::{Array1, Array2, Array3, Array4, Axis}; // SciRS2 Integration Policy
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::layers::{
    embedding::Embedding, feedforward::FeedForward, layernorm::LayerNorm, linear::Linear,
};
use trustformers_core::ops::activations::gelu;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Layer;

#[derive(Debug, Clone)]
pub struct DebertaEmbeddings {
    pub word_embeddings: Embedding,
    pub layer_norm: LayerNorm,
    pub dropout: f32,
}

impl DebertaEmbeddings {
    pub fn new(config: &DebertaConfig) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(
                config.vocab_size,
                config.hidden_size,
                Some(config.pad_token_id as usize),
            )?,
            layer_norm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            dropout: config.hidden_dropout_prob,
        })
    }

    pub fn forward(&self, input_ids: &Array1<u32>) -> Result<Array2<f32>> {
        // Word embeddings
        let embeddings = self.word_embeddings.forward_ids(input_ids.as_slice().unwrap())?;
        let embeddings_2d = match embeddings {
            Tensor::F32(arr) => arr
                .into_dimensionality::<ndarray::Ix2>()
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
                .into_dimensionality::<ndarray::Ix2>()
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

#[derive(Debug, Clone)]
pub struct DebertaDisentangledSelfAttention {
    pub query_proj: Linear,
    pub key_proj: Linear,
    pub value_proj: Linear,
    pub pos_query_proj: Option<Linear>, // For content-to-position attention
    pub pos_key_proj: Option<Linear>,   // For position-to-content attention
    pub pos_proj: Option<Linear>,       // Position embeddings projection
    pub dropout: f32,
    pub num_attention_heads: usize,
    pub attention_head_size: usize,
    pub all_head_size: usize,
    pub max_relative_positions: i32,
    pub pos_att_type: Vec<String>,
    pub share_att_key: bool,
}

impl DebertaDisentangledSelfAttention {
    pub fn new(config: &DebertaConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;

        let pos_query_proj = if config.pos_att_type.contains(&"c2p".to_string()) {
            Some(Linear::new(config.hidden_size, all_head_size, true))
        } else {
            None
        };

        let pos_key_proj =
            if config.pos_att_type.contains(&"p2c".to_string()) && !config.share_att_key {
                Some(Linear::new(config.hidden_size, all_head_size, true))
            } else {
                None
            };

        let pos_proj = if config.max_relative_positions > 0 {
            Some(Linear::new(
                config.max_relative_positions as usize * 2,
                all_head_size,
                false,
            ))
        } else {
            None
        };

        Ok(Self {
            query_proj: Linear::new(config.hidden_size, all_head_size, true),
            key_proj: Linear::new(config.hidden_size, all_head_size, true),
            value_proj: Linear::new(config.hidden_size, all_head_size, true),
            pos_query_proj,
            pos_key_proj,
            pos_proj,
            dropout: config.attention_probs_dropout_prob,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
            max_relative_positions: config.max_relative_positions,
            pos_att_type: config.pos_att_type.clone(),
            share_att_key: config.share_att_key,
        })
    }

    fn transpose_for_scores(&self, x: &Array3<f32>) -> Array4<f32> {
        let (batch_size, seq_len, _) = x.dim();

        // Reshape to (batch_size, seq_len, num_heads, head_size)
        let reshaped = x
            .to_shape((
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            ))
            .unwrap()
            .to_owned();

        // Transpose to (batch_size, num_heads, seq_len, head_size)
        reshaped.permuted_axes([0, 2, 1, 3])
    }

    fn build_relative_position(&self, query_size: usize, key_size: usize) -> Array2<i32> {
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
                .into_dimensionality::<ndarray::Ix3>()
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
                .into_dimensionality::<ndarray::Ix3>()
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
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from value projection",
                    "attention",
                ))
            },
        };

        let query_layer = self.transpose_for_scores(&query_layer);
        let key_layer = self.transpose_for_scores(&key_layer);
        let value_layer = self.transpose_for_scores(&value_layer);

        // Compute attention scores
        let mut attention_scores =
            Array4::zeros((batch_size, self.num_attention_heads, seq_len, seq_len));

        // Content-to-content attention
        for b in 0..batch_size {
            for h in 0..self.num_attention_heads {
                let q = query_layer.slice(ndarray::s![b, h, .., ..]);
                let k = key_layer.slice(ndarray::s![b, h, .., ..]);

                // Compute dot product attention
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let score: f32 = q
                            .slice(ndarray::s![i, ..])
                            .iter()
                            .zip(k.slice(ndarray::s![j, ..]).iter())
                            .map(|(a, b)| a * b)
                            .sum();

                        attention_scores[[b, h, i, j]] =
                            score / (self.attention_head_size as f32).sqrt();
                    }
                }
            }
        }

        // Add position-aware attention if enabled
        if self.pos_att_type.contains(&"c2p".to_string()) {
            // Content-to-position attention
            if let Some(pos_query_proj) = &self.pos_query_proj {
                let pos_query_input = Tensor::F32(hidden_states.clone().into_dyn());
                let pos_query_result = pos_query_proj.forward(pos_query_input)?;
                let pos_query_layer = match pos_query_result {
                    Tensor::F32(arr) => arr
                        .into_dimensionality::<ndarray::Ix3>()
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                    _ => {
                        return Err(TrustformersError::tensor_op_error(
                            "Expected F32 tensor from pos query projection",
                            "attention",
                        ))
                    },
                };
                let _pos_query_layer = self.transpose_for_scores(&pos_query_layer);

                // Build relative position embeddings
                let relative_pos = self.build_relative_position(seq_len, seq_len);

                // Add relative position bias (simplified implementation)
                for b in 0..batch_size {
                    for h in 0..self.num_attention_heads {
                        for i in 0..seq_len {
                            for j in 0..seq_len {
                                let pos_bias = relative_pos[[i, j]] as f32 * 0.01; // Simplified bias
                                attention_scores[[b, h, i, j]] += pos_bias;
                            }
                        }
                    }
                }
            }
        }

        if self.pos_att_type.contains(&"p2c".to_string()) {
            // Position-to-content attention (simplified)
            let relative_pos = self.build_relative_position(seq_len, seq_len);

            for b in 0..batch_size {
                for h in 0..self.num_attention_heads {
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            let pos_bias = relative_pos[[i, j]] as f32 * 0.01; // Simplified bias
                            attention_scores[[b, h, i, j]] += pos_bias;
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

        // Reshape to (batch_size, seq_len, all_head_size)
        let context_layer = context_layer
            .to_shape((batch_size, seq_len, self.all_head_size))
            .unwrap()
            .to_owned();

        Ok(context_layer)
    }
}

#[derive(Debug, Clone)]
pub struct DebertaSelfOutput {
    pub dense: Linear,
    pub layer_norm: LayerNorm,
    pub dropout: f32,
}

impl DebertaSelfOutput {
    pub fn new(config: &DebertaConfig) -> Result<Self> {
        Ok(Self {
            dense: Linear::new(config.hidden_size, config.hidden_size, true),
            layer_norm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            dropout: config.hidden_dropout_prob,
        })
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
                .into_dimensionality::<ndarray::Ix3>()
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
                .into_dimensionality::<ndarray::Ix3>()
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
}

impl DebertaAttention {
    pub fn new(config: &DebertaConfig) -> Result<Self> {
        Ok(Self {
            self_attention: DebertaDisentangledSelfAttention::new(config)?,
            output: DebertaSelfOutput::new(config)?,
        })
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
}

impl DebertaLayer {
    pub fn new(config: &DebertaConfig) -> Result<Self> {
        Ok(Self {
            attention: DebertaAttention::new(config)?,
            feed_forward: FeedForward::new(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_dropout_prob,
            )?,
            output_layer_norm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            dropout: config.hidden_dropout_prob,
        })
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
                .into_dimensionality::<ndarray::Ix3>()
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
                .into_dimensionality::<ndarray::Ix3>()
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
}

impl DebertaEncoder {
    pub fn new(config: &DebertaConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(DebertaLayer::new(config)?);
        }

        Ok(Self { layers })
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
}

impl DebertaModel {
    pub fn new(config: DebertaConfig) -> Result<Self> {
        Ok(Self {
            embeddings: DebertaEmbeddings::new(&config)?,
            encoder: DebertaEncoder::new(&config)?,
            config,
        })
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

#[derive(Debug, Clone)]
pub struct DebertaForSequenceClassification {
    pub deberta: DebertaModel,
    pub pooler: Linear,
    pub classifier: Linear,
    pub dropout: f32,
    pub num_labels: usize,
}

impl DebertaForSequenceClassification {
    pub fn new(config: DebertaConfig, num_labels: usize) -> Result<Self> {
        let dropout = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);

        Ok(Self {
            deberta: DebertaModel::new(config.clone())?,
            pooler: Linear::new(config.hidden_size, config.hidden_size, true),
            classifier: Linear::new(config.hidden_size, num_labels, true),
            dropout,
            num_labels,
        })
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
        let cls_hidden = hidden_states.slice(ndarray::s![0, 0, ..]).to_owned();

        // Pooler
        let pooler_input = Tensor::F32(cls_hidden.insert_axis(Axis(0)).into_dyn());
        let pooled_output = self.pooler.forward(pooler_input)?;
        let pooled_output = match pooled_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<ndarray::Ix2>()
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
                .into_dimensionality::<ndarray::Ix2>()
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
                .into_dimensionality::<ndarray::Ix2>()
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
}

impl DebertaForMaskedLM {
    pub fn new(config: DebertaConfig) -> Result<Self> {
        Ok(Self {
            deberta: DebertaModel::new(config.clone())?,
            cls: Linear::new(config.hidden_size, config.vocab_size, true),
        })
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
                .into_dimensionality::<ndarray::Ix3>()
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
