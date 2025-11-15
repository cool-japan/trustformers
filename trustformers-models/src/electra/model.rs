use crate::electra::config::ElectraConfig;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis, Ix2, Ix3}; // SciRS2 Integration Policy
use trustformers_core::device::Device;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::layers::{
    attention::MultiHeadAttention, embedding::Embedding, feedforward::FeedForward,
    layernorm::LayerNorm, linear::Linear,
};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Layer;

#[derive(Debug, Clone)]
pub struct ElectraEmbeddings {
    pub word_embeddings: Embedding,
    pub position_embeddings: Embedding,
    pub token_type_embeddings: Embedding,
    pub layer_norm: LayerNorm,
    pub dropout: f32,
    device: Device,
}

impl ElectraEmbeddings {
    pub fn new(config: &ElectraConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &ElectraConfig, device: Device) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new_with_device(
                config.vocab_size,
                config.embedding_size,
                Some(config.pad_token_id as usize),
                device,
            )?,
            position_embeddings: Embedding::new_with_device(
                config.max_position_embeddings,
                config.embedding_size,
                None,
                device,
            )?,
            token_type_embeddings: Embedding::new_with_device(
                config.type_vocab_size,
                config.embedding_size,
                None,
                device,
            )?,
            layer_norm: LayerNorm::new_with_device(
                vec![config.embedding_size],
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
        input_ids: &Array1<u32>,
        token_type_ids: Option<&Array1<u32>>,
        position_ids: Option<&Array1<u32>>,
    ) -> Result<Array2<f32>> {
        let seq_len = input_ids.len();

        // Word embeddings
        let word_emb = self.word_embeddings.forward_ids(input_ids.as_slice().unwrap())?;
        let word_emb_2d = match word_emb {
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

        // Position embeddings
        let pos_ids: Array1<u32> = if let Some(pos_ids) = position_ids {
            pos_ids.clone()
        } else {
            (0..seq_len as u32).collect()
        };
        let pos_emb = self.position_embeddings.forward_ids(pos_ids.as_slice().unwrap())?;
        let pos_emb_2d = match pos_emb {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor for position embeddings",
                    "embeddings",
                ))
            },
        };

        // Token type embeddings
        let tt_ids: Array1<u32> = if let Some(tt_ids) = token_type_ids {
            tt_ids.clone()
        } else {
            Array1::zeros(seq_len)
        };
        let tt_emb = self.token_type_embeddings.forward_ids(tt_ids.as_slice().unwrap())?;
        let tt_emb_2d = match tt_emb {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor for token type embeddings",
                    "embeddings",
                ))
            },
        };

        // Sum all embeddings
        let combined_embeddings = word_emb_2d + pos_emb_2d + tt_emb_2d;

        // Layer normalization
        let norm_input = Tensor::F32(combined_embeddings.into_dyn());
        let embeddings = self.layer_norm.forward(norm_input)?;
        let embeddings_2d = match embeddings {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor after layer norm",
                    "layer_norm",
                ))
            },
        };

        // Apply dropout
        Ok(embeddings_2d * (1.0 - self.dropout))
    }
}

#[derive(Debug, Clone)]
pub struct ElectraLayer {
    pub attention: MultiHeadAttention,
    pub feed_forward: FeedForward,
    pub attention_layer_norm: LayerNorm,
    pub output_layer_norm: LayerNorm,
    pub dropout: f32,
    device: Device,
}

impl ElectraLayer {
    pub fn new(
        config: &ElectraConfig,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        Self::new_with_device(
            config,
            hidden_size,
            num_heads,
            intermediate_size,
            Device::CPU,
        )
    }

    pub fn new_with_device(
        config: &ElectraConfig,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new_with_device(
                hidden_size,
                num_heads,
                config.attention_probs_dropout_prob,
                true,
                device,
            )?,
            feed_forward: FeedForward::new_with_device(
                hidden_size,
                intermediate_size,
                config.hidden_dropout_prob,
                device,
            )?,
            attention_layer_norm: LayerNorm::new_with_device(
                vec![hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            output_layer_norm: LayerNorm::new_with_device(
                vec![hidden_size],
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
        _attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array3<f32>> {
        // Self-attention with residual connection
        let hidden_states_tensor = Tensor::F32(hidden_states.clone().into_dyn());
        let attention_output = self.attention.forward(hidden_states_tensor)?;
        let attention_output = match attention_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from attention",
                    "attention",
                ))
            },
        };
        // Apply dropout (element-wise multiplication)
        let attention_output = attention_output.mapv(|x| x * (1.0 - self.dropout));

        // Add residual and apply layer norm
        let attention_residual = hidden_states + &attention_output;
        let attention_norm_input = Tensor::F32(attention_residual.into_dyn());
        let attention_output = self.attention_layer_norm.forward(attention_norm_input)?;
        let attention_output = match attention_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor after attention layer norm",
                    "layer_norm",
                ))
            },
        };

        // Feed-forward with residual connection
        let ff_input = Tensor::F32(attention_output.clone().into_dyn());
        let ff_output = self.feed_forward.forward(ff_input)?;
        let ff_output = match ff_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor after feed forward",
                    "feed_forward",
                ))
            },
        };
        let ff_output = ff_output * (1.0 - self.dropout);

        // Add residual and apply layer norm
        let output_residual = &attention_output + &ff_output;
        let output_norm_input = Tensor::F32(output_residual.into_dyn());
        let output = self.output_layer_norm.forward(output_norm_input)?;
        let output = match output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor after output layer norm",
                    "layer_norm",
                ))
            },
        };

        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct ElectraEncoder {
    pub layers: Vec<ElectraLayer>,
    device: Device,
}

impl ElectraEncoder {
    pub fn new(
        config: &ElectraConfig,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        Self::new_with_device(
            config,
            hidden_size,
            num_layers,
            num_heads,
            intermediate_size,
            Device::CPU,
        )
    }

    pub fn new_with_device(
        config: &ElectraConfig,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_size: usize,
        device: Device,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(ElectraLayer::new_with_device(
                config,
                hidden_size,
                num_heads,
                intermediate_size,
                device,
            )?);
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
pub struct ElectraGenerator {
    pub embeddings: ElectraEmbeddings,
    pub embeddings_project: Option<Linear>,
    pub encoder: ElectraEncoder,
    pub layer_norm: LayerNorm,
    pub lm_head: Linear,
    pub config: ElectraConfig,
    device: Device,
}

impl ElectraGenerator {
    pub fn new(config: &ElectraConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &ElectraConfig, device: Device) -> Result<Self> {
        // Create projection layer if embedding size differs from generator hidden size
        let embeddings_project = if config.embedding_size != config.generator_hidden_size {
            Some(Linear::new_with_device(
                config.embedding_size,
                config.generator_hidden_size,
                true,
                device,
            ))
        } else {
            None
        };

        Ok(Self {
            embeddings: ElectraEmbeddings::new_with_device(config, device)?,
            embeddings_project,
            encoder: ElectraEncoder::new_with_device(
                config,
                config.generator_hidden_size,
                config.generator_num_hidden_layers,
                config.generator_num_attention_heads,
                config.generator_intermediate_size,
                device,
            )?,
            layer_norm: LayerNorm::new_with_device(
                vec![config.generator_hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            lm_head: Linear::new_with_device(
                config.generator_hidden_size,
                config.vocab_size,
                true,
                device,
            ),
            config: config.clone(),
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        token_type_ids: Option<&Array1<u32>>,
        position_ids: Option<&Array1<u32>>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array3<f32>> {
        // Get embeddings
        let mut embeddings = self.embeddings.forward(input_ids, token_type_ids, position_ids)?;

        // Project embeddings if necessary
        if let Some(ref proj) = self.embeddings_project {
            let emb_3d = embeddings.insert_axis(Axis(0));
            let proj_input = Tensor::F32(emb_3d.into_dyn());
            let proj_output = proj.forward(proj_input)?;
            embeddings = match proj_output {
                Tensor::F32(arr) => {
                    let arr_3d = arr
                        .into_dimensionality::<Ix3>()
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    // Remove batch dimension to get back to 2D
                    arr_3d.index_axis_move(Axis(0), 0)
                },
                _ => {
                    return Err(TrustformersError::tensor_op_error(
                        "Expected F32 tensor from projection",
                        "projection",
                    ))
                },
            };
        }

        // Convert to 3D for encoder (batch_size=1, seq_len, hidden_size)
        let hidden_states = embeddings.insert_axis(Axis(0));

        // Pass through encoder
        let encoder_output = self.encoder.forward(hidden_states, attention_mask)?;

        // Layer normalization
        let norm_input = Tensor::F32(encoder_output.into_dyn());
        let normalized_output = self.layer_norm.forward(norm_input)?;
        let normalized_output = match normalized_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor after layer norm",
                    "layer_norm",
                ))
            },
        };

        // Language modeling head
        let lm_input = Tensor::F32(normalized_output.clone().into_dyn());
        let logits = self.lm_head.forward(lm_input)?;
        let logits = match logits {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from LM head",
                    "lm_head",
                ))
            },
        };

        Ok(logits)
    }
}

#[derive(Debug, Clone)]
pub struct ElectraDiscriminator {
    pub embeddings: ElectraEmbeddings,
    pub embeddings_project: Option<Linear>,
    pub encoder: ElectraEncoder,
    pub layer_norm: LayerNorm,
    pub config: ElectraConfig,
    device: Device,
}

impl ElectraDiscriminator {
    pub fn new(config: &ElectraConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &ElectraConfig, device: Device) -> Result<Self> {
        // Create projection layer if embedding size differs from discriminator hidden size
        let embeddings_project = if config.embedding_size != config.discriminator_hidden_size {
            Some(Linear::new_with_device(
                config.embedding_size,
                config.discriminator_hidden_size,
                true,
                device,
            ))
        } else {
            None
        };

        Ok(Self {
            embeddings: ElectraEmbeddings::new_with_device(config, device)?,
            embeddings_project,
            encoder: ElectraEncoder::new_with_device(
                config,
                config.discriminator_hidden_size,
                config.discriminator_num_hidden_layers,
                config.discriminator_num_attention_heads,
                config.discriminator_intermediate_size,
                device,
            )?,
            layer_norm: LayerNorm::new_with_device(
                vec![config.discriminator_hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            config: config.clone(),
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        token_type_ids: Option<&Array1<u32>>,
        position_ids: Option<&Array1<u32>>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array3<f32>> {
        // Get embeddings
        let mut embeddings = self.embeddings.forward(input_ids, token_type_ids, position_ids)?;

        // Project embeddings if necessary
        if let Some(ref proj) = self.embeddings_project {
            let emb_3d = embeddings.insert_axis(Axis(0));
            let proj_input = Tensor::F32(emb_3d.into_dyn());
            let proj_output = proj.forward(proj_input)?;
            embeddings = match proj_output {
                Tensor::F32(arr) => {
                    let arr_3d = arr
                        .into_dimensionality::<Ix3>()
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    // Remove batch dimension to get back to 2D
                    arr_3d.index_axis_move(Axis(0), 0)
                },
                _ => {
                    return Err(TrustformersError::tensor_op_error(
                        "Expected F32 tensor from projection",
                        "projection",
                    ))
                },
            };
        }

        // Convert to 3D for encoder (batch_size=1, seq_len, hidden_size)
        let hidden_states = embeddings.insert_axis(Axis(0));

        // Pass through encoder
        let encoder_output = self.encoder.forward(hidden_states, attention_mask)?;

        // Layer normalization
        let norm_input = Tensor::F32(encoder_output.into_dyn());
        let output = self.layer_norm.forward(norm_input)?;
        let output = match output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor after layer norm",
                    "layer_norm",
                ))
            },
        };

        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct ElectraModel {
    pub generator: ElectraGenerator,
    pub discriminator: ElectraDiscriminator,
    pub config: ElectraConfig,
    device: Device,
}

impl ElectraModel {
    pub fn new(config: ElectraConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: ElectraConfig, device: Device) -> Result<Self> {
        Ok(Self {
            generator: ElectraGenerator::new_with_device(&config, device)?,
            discriminator: ElectraDiscriminator::new_with_device(&config, device)?,
            config,
            device,
        })
    }

    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = ElectraConfig::from_pretrained_name(model_name);
        Self::new(config)
    }

    pub fn from_pretrained_with_device(model_name: &str, device: Device) -> Result<Self> {
        let config = ElectraConfig::from_pretrained_name(model_name);
        Self::new_with_device(config, device)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn get_generator(&self) -> &ElectraGenerator {
        &self.generator
    }

    pub fn get_discriminator(&self) -> &ElectraDiscriminator {
        &self.discriminator
    }
}

// Discriminator head for binary classification (replaced token detection)
#[derive(Debug, Clone)]
pub struct ElectraForPreTraining {
    pub electra: ElectraModel,
    pub discriminator_head: Linear,
    device: Device,
}

impl ElectraForPreTraining {
    pub fn new(config: ElectraConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: ElectraConfig, device: Device) -> Result<Self> {
        Ok(Self {
            electra: ElectraModel::new_with_device(config.clone(), device)?,
            discriminator_head: Linear::new_with_device(
                config.discriminator_hidden_size,
                1,
                true,
                device,
            ),
            device,
        })
    }

    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = ElectraConfig::from_pretrained_name(model_name);
        Self::new(config)
    }

    pub fn from_pretrained_with_device(model_name: &str, device: Device) -> Result<Self> {
        let config = ElectraConfig::from_pretrained_name(model_name);
        Self::new_with_device(config, device)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        token_type_ids: Option<&Array1<u32>>,
        position_ids: Option<&Array1<u32>>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<(Array3<f32>, Array3<f32>)> {
        // Generator predictions
        let generator_logits = self.electra.generator.forward(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
        )?;

        // Discriminator hidden states
        let discriminator_hidden = self.electra.discriminator.forward(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
        )?;

        // Discriminator predictions (token replacement detection)
        let disc_input = Tensor::F32(discriminator_hidden.clone().into_dyn());
        let discriminator_logits = self.discriminator_head.forward(disc_input)?;
        let discriminator_logits = match discriminator_logits {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Expected F32 tensor from discriminator head",
                    "discriminator_head",
                ))
            },
        };

        Ok((generator_logits, discriminator_logits))
    }
}

#[derive(Debug, Clone)]
pub struct ElectraForSequenceClassification {
    pub electra: ElectraDiscriminator,
    pub classifier: Linear,
    pub dropout: f32,
    pub num_labels: usize,
    device: Device,
}

impl ElectraForSequenceClassification {
    pub fn new(config: ElectraConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(
        config: ElectraConfig,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let dropout = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);

        Ok(Self {
            electra: ElectraDiscriminator::new_with_device(&config, device)?,
            classifier: Linear::new_with_device(
                config.discriminator_hidden_size,
                num_labels,
                true,
                device,
            ),
            dropout,
            num_labels,
            device,
        })
    }

    pub fn from_pretrained(model_name: &str, num_labels: usize) -> Result<Self> {
        let config = ElectraConfig::from_pretrained_name(model_name);
        Self::new(config, num_labels)
    }

    pub fn from_pretrained_with_device(
        model_name: &str,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let config = ElectraConfig::from_pretrained_name(model_name);
        Self::new_with_device(config, num_labels, device)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(
        &self,
        input_ids: &Array1<u32>,
        token_type_ids: Option<&Array1<u32>>,
        position_ids: Option<&Array1<u32>>,
        attention_mask: Option<&Array3<f32>>,
    ) -> Result<Array2<f32>> {
        let hidden_states =
            self.electra.forward(input_ids, token_type_ids, position_ids, attention_mask)?;

        // Use [CLS] token representation (first token)
        let cls_hidden = hidden_states.slice(s![0, 0, ..]).to_owned();

        // Apply dropout
        let cls_hidden = cls_hidden * (1.0 - self.dropout);

        // Classification head
        let cls_input = Tensor::F32(cls_hidden.insert_axis(Axis(0)).into_dyn());
        let logits = self.classifier.forward(cls_input)?;
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
