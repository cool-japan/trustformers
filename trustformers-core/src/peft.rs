#![allow(unused_variables)] // PEFT implementation with reserved parameters

use crate::errors::{Result, TrustformersError};
use crate::layers::Linear;
use crate::tensor::Tensor;
use crate::traits::Layer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Parameter-Efficient Fine-Tuning (PEFT) methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeftMethod {
    /// Low-Rank Adaptation (LoRA)
    LoRA,
    /// Quantized LoRA (QLoRA)
    QLoRA,
    /// Adaptive Low-Rank Adaptation (AdaLoRA)
    AdaLoRA,
    /// Prefix Tuning
    PrefixTuning,
    /// P-Tuning v2
    PTuningV2,
    /// Prompt Tuning
    PromptTuning,
    /// Adapter layers
    Adapter,
    /// BitFit (bias-only fine-tuning)
    BitFit,
}

/// Configuration for PEFT methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeftConfig {
    pub method: PeftMethod,
    pub r: Option<usize>,            // Rank for LoRA
    pub alpha: Option<f32>,          // Scaling factor for LoRA
    pub dropout: Option<f32>,        // Dropout rate
    pub target_modules: Vec<String>, // Which modules to apply PEFT to
    pub bias: Option<String>,        // Bias training strategy
    pub task_type: Option<String>,   // Task type for optimization
    pub inference_mode: bool,        // Whether in inference mode
}

impl Default for PeftConfig {
    fn default() -> Self {
        Self {
            method: PeftMethod::LoRA,
            r: Some(8),
            alpha: Some(16.0),
            dropout: Some(0.1),
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            bias: Some("none".to_string()),
            task_type: Some("CAUSAL_LM".to_string()),
            inference_mode: false,
        }
    }
}

/// LoRA (Low-Rank Adaptation) layer
///
/// LoRA approximates weight updates as W' = W + BA where B is r×d and A is d×r
/// This reduces trainable parameters from d×d to 2×d×r where r << d
#[derive(Debug, Clone)]
pub struct LoRALayer {
    pub base_layer: Linear,
    pub lora_a: Linear, // Down-projection: input_dim -> r
    pub lora_b: Linear, // Up-projection: r -> output_dim
    pub alpha: f32,     // Scaling factor
    pub r: usize,       // Rank
    pub dropout: f32,
    pub merged: bool, // Whether LoRA weights are merged into base layer
    pub frozen: bool, // Whether base layer is frozen
}

impl LoRALayer {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        r: usize,
        alpha: f32,
        dropout: f32,
        bias: bool,
    ) -> Result<Self> {
        if r == 0 {
            return Err(TrustformersError::invalid_config(
                "LoRA rank must be greater than 0".into(),
            ));
        }

        Ok(Self {
            base_layer: Linear::new(input_dim, output_dim, bias),
            lora_a: Linear::new(input_dim, r, false), // No bias for LoRA layers
            lora_b: Linear::new(r, output_dim, false),
            alpha,
            r,
            dropout,
            merged: false,
            frozen: true, // Base layer starts frozen
        })
    }

    /// Initialize LoRA weights
    pub fn initialize_weights(&mut self) -> Result<()> {
        // Initialize A with Gaussian noise, B with zeros (standard LoRA initialization)
        // This ensures that initially LoRA contributes nothing: BA = 0

        // Initialize lora_a weights with small random values
        let a_weights = Tensor::randn(&[self.r, self.lora_a.weight().shape()[1]])?;
        let scaled_a = a_weights.scalar_mul(0.01)?; // Small initialization
        self.lora_a.set_weight(scaled_a)?;

        // Initialize lora_b weights to zero
        let b_weights = Tensor::zeros(&[self.lora_b.weight().shape()[0], self.r])?;
        self.lora_b.set_weight(b_weights)?;

        Ok(())
    }

    /// Merge LoRA weights into base layer for inference
    pub fn merge_weights(&mut self) -> Result<()> {
        if self.merged {
            return Ok(()); // Already merged
        }

        // Compute LoRA contribution: (alpha/r) * B @ A
        let lora_weight = self.lora_b.weight().matmul(self.lora_a.weight())?;
        let scaling = self.alpha / self.r as f32;
        let scaled_lora = lora_weight.scalar_mul(scaling)?;

        // Add to base weights: W' = W + (alpha/r) * B @ A
        let new_weight = self.base_layer.weight().add(&scaled_lora)?;
        self.base_layer.set_weight(new_weight)?;
        self.merged = true;

        Ok(())
    }

    /// Unmerge LoRA weights from base layer
    pub fn unmerge_weights(&mut self) -> Result<()> {
        if !self.merged {
            return Ok(()); // Not merged
        }

        // Subtract LoRA contribution
        let lora_weight = self.lora_b.weight().matmul(self.lora_a.weight())?;
        let scaling = self.alpha / self.r as f32;
        let scaled_lora = lora_weight.scalar_mul(scaling)?;

        // Subtract from base weights: W = W' - (alpha/r) * B @ A
        let neg_lora = scaled_lora.scalar_mul(-1.0)?;
        let new_weight = self.base_layer.weight().add(&neg_lora)?;
        self.base_layer.set_weight(new_weight)?;
        self.merged = false;

        Ok(())
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.frozen = false;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.frozen = true;
    }

    /// Get trainable parameters
    pub fn trainable_parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![self.lora_a.weight(), self.lora_b.weight()];

        if !self.frozen {
            params.push(self.base_layer.weight());
            if let Some(bias) = self.base_layer.bias() {
                params.push(bias);
            }
        }

        params
    }
}

impl Layer for LoRALayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        if self.merged {
            // If weights are merged, just use base layer
            self.base_layer.forward(input)
        } else {
            // Compute: h = (W + (alpha/r) * B @ A) @ x
            // = W @ x + (alpha/r) * B @ (A @ x)

            let base_output = self.base_layer.forward(input.clone())?;

            // LoRA path: A @ x
            let lora_a_output = self.lora_a.forward(input)?;

            // Apply dropout to LoRA path
            let lora_a_dropped = if self.dropout > 0.0 {
                lora_a_output.dropout(self.dropout)?
            } else {
                lora_a_output
            };

            // B @ (A @ x)
            let lora_output = self.lora_b.forward(lora_a_dropped)?;

            // Scale and add: output = base_output + (alpha/r) * lora_output
            let scaling = self.alpha / self.r as f32;
            let scaled_lora = lora_output.scalar_mul(scaling)?;

            base_output.add(&scaled_lora)
        }
    }
}

/// QLoRA layer combining LoRA with quantization
#[derive(Debug, Clone)]
pub struct QLoRALayer {
    pub lora_layer: LoRALayer,
    pub quantized_base: Option<crate::quantization::QuantizedTensor>,
}

impl QLoRALayer {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        r: usize,
        alpha: f32,
        dropout: f32,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            lora_layer: LoRALayer::new(input_dim, output_dim, r, alpha, dropout, bias)?,
            quantized_base: None,
        })
    }

    /// Quantize the base layer weights
    pub fn quantize_base(
        &mut self,
        config: &crate::quantization::QuantizationConfig,
    ) -> Result<()> {
        let quantized =
            crate::quantization::Quantizer::quantize(self.lora_layer.base_layer.weight(), config)?;
        self.quantized_base = Some(quantized);
        Ok(())
    }
}

impl Layer for QLoRALayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // If base layer is quantized, dequantize for computation
        if let Some(ref quantized) = self.quantized_base {
            let dequantized_weight = quantized.dequantize()?;

            // Create temporary linear layer with dequantized weights
            let mut temp_base = self.lora_layer.base_layer.clone();
            temp_base.set_weight(dequantized_weight)?;

            // Compute base output
            let base_output = temp_base.forward(input.clone())?;

            // LoRA computation
            let lora_a_output = self.lora_layer.lora_a.forward(input)?;
            let lora_a_dropped = if self.lora_layer.dropout > 0.0 {
                lora_a_output.dropout(self.lora_layer.dropout)?
            } else {
                lora_a_output
            };
            let lora_output = self.lora_layer.lora_b.forward(lora_a_dropped)?;

            let scaling = self.lora_layer.alpha / self.lora_layer.r as f32;
            let scaled_lora = lora_output.scalar_mul(scaling)?;

            base_output.add(&scaled_lora)
        } else {
            // Fall back to regular LoRA
            self.lora_layer.forward(input)
        }
    }
}

/// Adapter layer for parameter-efficient fine-tuning
#[derive(Debug, Clone)]
pub struct AdapterLayer {
    pub down_proj: Linear,
    pub up_proj: Linear,
    pub activation: ActivationType,
    pub bottleneck_size: usize,
    pub dropout: f32,
    pub residual_connection: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    GELU,
    Swish,
    Tanh,
}

impl AdapterLayer {
    pub fn new(
        hidden_size: usize,
        bottleneck_size: usize,
        activation: ActivationType,
        dropout: f32,
    ) -> Self {
        Self {
            down_proj: Linear::new(hidden_size, bottleneck_size, true),
            up_proj: Linear::new(bottleneck_size, hidden_size, true),
            activation,
            bottleneck_size,
            dropout,
            residual_connection: true,
        }
    }

    fn apply_activation(&self, tensor: &Tensor) -> Result<Tensor> {
        match self.activation {
            ActivationType::ReLU => self.relu(tensor),
            ActivationType::GELU => self.gelu(tensor),
            ActivationType::Swish => self.swish(tensor),
            ActivationType::Tanh => self.tanh(tensor),
        }
    }

    fn relu(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let result = arr.mapv(|x| x.max(0.0));
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for ReLU",
                "LoRAActivation::relu",
            )),
        }
    }

    fn gelu(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let result = arr.mapv(|x| {
                    0.5 * x
                        * (1.0
                            + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3)))
                                .tanh())
                });
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for GELU",
                "LoRAActivation::gelu",
            )),
        }
    }

    fn swish(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let result = arr.mapv(|x| x / (1.0 + (-x).exp()));
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for Swish",
                "LoRAActivation::swish",
            )),
        }
    }

    fn tanh(&self, tensor: &Tensor) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let result = arr.mapv(|x| x.tanh());
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for Tanh",
                "LoRAActivation::tanh",
            )),
        }
    }
}

impl Layer for AdapterLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Down-projection
        let down_output = self.down_proj.forward(input.clone())?;

        // Activation
        let activated = self.apply_activation(&down_output)?;

        // Dropout
        let dropped = if self.dropout > 0.0 { activated.dropout(self.dropout)? } else { activated };

        // Up-projection
        let up_output = self.up_proj.forward(dropped)?;

        // Residual connection
        if self.residual_connection {
            input.add(&up_output)
        } else {
            Ok(up_output)
        }
    }
}

/// Prefix tuning layer
#[derive(Debug, Clone)]
pub struct PrefixTuningLayer {
    pub prefix_length: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub prefix_projection: Linear,
    pub prefix_embeddings: Tensor,
}

impl PrefixTuningLayer {
    pub fn new(
        prefix_length: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let projection_dim = hidden_size * 2; // For both key and value
        let total_prefix_dim = num_layers * num_heads * prefix_length * 2; // Key + Value

        Ok(Self {
            prefix_length,
            hidden_size,
            num_layers,
            num_heads,
            prefix_projection: Linear::new(hidden_size, projection_dim, true),
            prefix_embeddings: Tensor::randn(&[prefix_length, hidden_size])?,
        })
    }

    pub fn get_prefix_states(&self) -> Result<Vec<(Tensor, Tensor)>> {
        let mut prefix_states = Vec::new();

        for layer_idx in 0..self.num_layers {
            // Project prefix embeddings to get key and value states
            let projected = self.prefix_projection.forward(self.prefix_embeddings.clone())?;

            // Split into key and value
            let key_value_split = projected.split(1, self.hidden_size)?; // Split along last dimension
            if key_value_split.len() != 2 {
                return Err(TrustformersError::invalid_input(
                    "Projection split failed".into(),
                ));
            }

            let key_states = key_value_split[0].clone();
            let value_states = key_value_split[1].clone();

            prefix_states.push((key_states, value_states));
        }

        Ok(prefix_states)
    }
}

/// Prompt tuning embeddings
#[derive(Debug, Clone)]
pub struct PromptTuningEmbedding {
    pub num_virtual_tokens: usize,
    pub hidden_size: usize,
    pub prompt_embeddings: Tensor,
    pub init_method: PromptInitMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptInitMethod {
    Random,
    Text,
    VocabAverage,
}

impl PromptTuningEmbedding {
    pub fn new(
        num_virtual_tokens: usize,
        hidden_size: usize,
        init_method: PromptInitMethod,
    ) -> Result<Self> {
        let prompt_embeddings = match init_method {
            PromptInitMethod::Random => Tensor::randn(&[num_virtual_tokens, hidden_size])?,
            PromptInitMethod::Text => {
                // Initialize with small random values for text-based initialization
                let embeddings = Tensor::randn(&[num_virtual_tokens, hidden_size])?;
                embeddings.scalar_mul(0.1)?
            },
            PromptInitMethod::VocabAverage => {
                // Initialize with zeros for vocabulary average initialization
                Tensor::zeros(&[num_virtual_tokens, hidden_size])?
            },
        };

        Ok(Self {
            num_virtual_tokens,
            hidden_size,
            prompt_embeddings,
            init_method,
        })
    }

    pub fn get_prompt_embeddings(&self) -> &Tensor {
        &self.prompt_embeddings
    }

    pub fn update_embeddings(&mut self, new_embeddings: Tensor) -> Result<()> {
        if new_embeddings.shape() != self.prompt_embeddings.shape() {
            return Err(TrustformersError::shape_error(format!(
                "Shape mismatch: expected {:?}, got {:?}",
                self.prompt_embeddings.shape(),
                new_embeddings.shape()
            )));
        }

        self.prompt_embeddings = new_embeddings;
        Ok(())
    }
}

impl Layer for PrefixTuningLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Apply prefix projection to input
        let projected = self.prefix_projection.forward(input)?;

        // For prefix tuning, we typically just return the projected input
        // The actual prefix embeddings are used during attention computation
        // which is handled by the attention mechanism that queries this layer
        Ok(projected)
    }
}

impl Layer for PromptTuningEmbedding {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // For prompt tuning, we concatenate the virtual prompt tokens with the input
        // The prompt embeddings are prepended to the input sequence

        // Get batch size from input (assuming input shape is [batch_size, seq_len, hidden_size])
        let input_shape = input.shape();
        if input_shape.len() != 3 {
            return Err(TrustformersError::shape_error(format!(
                "Expected 3D input tensor [batch_size, seq_len, {}], got {:?}",
                self.hidden_size, input_shape
            )));
        }

        let batch_size = input_shape[0];

        // Expand prompt embeddings to match batch size
        // First reshape to add batch dimension: [num_virtual_tokens, hidden_size] -> [1, num_virtual_tokens, hidden_size]
        let prompt_with_batch =
            self.prompt_embeddings
                .reshape(&[1, self.num_virtual_tokens, self.hidden_size])?;

        // Then broadcast to match batch size: [1, num_virtual_tokens, hidden_size] -> [batch_size, num_virtual_tokens, hidden_size]
        let prompt_expanded = prompt_with_batch.broadcast_to(&[
            batch_size,
            self.num_virtual_tokens,
            self.hidden_size,
        ])?;

        // Concatenate prompt embeddings with input along sequence dimension
        let concatenated = Tensor::concat(&[prompt_expanded, input], 1)?;

        Ok(concatenated)
    }
}

/// Serializable representation of PEFT layer data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializableLayerData {
    LoRA {
        base_weight: Vec<f32>,
        base_bias: Option<Vec<f32>>,
        lora_a_weight: Vec<f32>,
        lora_b_weight: Vec<f32>,
        alpha: f32,
        r: usize,
        dropout: f32,
        merged: bool,
        frozen: bool,
        input_dim: usize,
        output_dim: usize,
    },
    Adapter {
        down_proj_weight: Vec<f32>,
        down_proj_bias: Vec<f32>,
        up_proj_weight: Vec<f32>,
        up_proj_bias: Vec<f32>,
        activation: ActivationType,
        bottleneck_size: usize,
        dropout: f32,
        residual_connection: bool,
        hidden_size: usize,
    },
    PrefixTuning {
        prefix_projection_weight: Vec<f32>,
        prefix_projection_bias: Vec<f32>,
        prefix_embeddings: Vec<f32>,
        prefix_length: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
    },
    PromptTuning {
        prompt_embeddings: Vec<f32>,
        num_virtual_tokens: usize,
        hidden_size: usize,
        init_method: PromptInitMethod,
    },
}

/// PEFT model wrapper that applies PEFT methods to a base model
pub struct PeftModel {
    pub config: PeftConfig,
    pub peft_layers: HashMap<String, Box<dyn Layer<Input = Tensor, Output = Tensor>>>,
    pub layer_metadata: HashMap<String, SerializableLayerData>,
    pub active: bool,
}

impl PeftModel {
    pub fn new(config: PeftConfig) -> Self {
        Self {
            config,
            peft_layers: HashMap::new(),
            layer_metadata: HashMap::new(),
            active: true,
        }
    }

    /// Convert a LoRA layer to serializable data
    fn serialize_lora_layer(layer: &LoRALayer) -> Result<SerializableLayerData> {
        let base_weight = layer.base_layer.weight().data()?;
        let base_bias = layer.base_layer.bias().map(|b| b.data()).transpose()?;
        let lora_a_weight = layer.lora_a.weight().data()?;
        let lora_b_weight = layer.lora_b.weight().data()?;

        Ok(SerializableLayerData::LoRA {
            base_weight,
            base_bias,
            lora_a_weight,
            lora_b_weight,
            alpha: layer.alpha,
            r: layer.r,
            dropout: layer.dropout,
            merged: layer.merged,
            frozen: layer.frozen,
            input_dim: layer.base_layer.weight().shape()[1],
            output_dim: layer.base_layer.weight().shape()[0],
        })
    }

    /// Convert serializable data to a LoRA layer
    fn deserialize_lora_layer(data: &SerializableLayerData) -> Result<LoRALayer> {
        if let SerializableLayerData::LoRA {
            base_weight,
            base_bias,
            lora_a_weight,
            lora_b_weight,
            alpha,
            r,
            dropout,
            merged,
            frozen,
            input_dim,
            output_dim,
        } = data
        {
            let mut layer = LoRALayer::new(
                *input_dim,
                *output_dim,
                *r,
                *alpha,
                *dropout,
                base_bias.is_some(),
            )?;

            // Set base layer weights
            let base_weight_tensor =
                Tensor::from_vec(base_weight.clone(), &[*output_dim, *input_dim])?;
            layer.base_layer.set_weight(base_weight_tensor)?;

            if let Some(bias_data) = base_bias {
                let bias_tensor = Tensor::from_vec(bias_data.clone(), &[*output_dim])?;
                layer.base_layer.set_bias(bias_tensor)?;
            }

            // Set LoRA weights
            let lora_a_tensor = Tensor::from_vec(lora_a_weight.clone(), &[*r, *input_dim])?;
            layer.lora_a.set_weight(lora_a_tensor)?;

            let lora_b_tensor = Tensor::from_vec(lora_b_weight.clone(), &[*output_dim, *r])?;
            layer.lora_b.set_weight(lora_b_tensor)?;

            // Set state
            layer.merged = *merged;
            layer.frozen = *frozen;

            Ok(layer)
        } else {
            Err(TrustformersError::invalid_input(
                "Expected LoRA layer data".into(),
            ))
        }
    }

    /// Convert an Adapter layer to serializable data
    fn serialize_adapter_layer(layer: &AdapterLayer) -> Result<SerializableLayerData> {
        let down_proj_weight = layer.down_proj.weight().data()?;
        let down_proj_bias =
            layer.down_proj.bias().map(|b| b.data()).transpose()?.unwrap_or_default();
        let up_proj_weight = layer.up_proj.weight().data()?;
        let up_proj_bias = layer.up_proj.bias().map(|b| b.data()).transpose()?.unwrap_or_default();

        Ok(SerializableLayerData::Adapter {
            down_proj_weight,
            down_proj_bias,
            up_proj_weight,
            up_proj_bias,
            activation: layer.activation,
            bottleneck_size: layer.bottleneck_size,
            dropout: layer.dropout,
            residual_connection: layer.residual_connection,
            hidden_size: layer.up_proj.weight().shape()[1],
        })
    }

    /// Convert a PrefixTuning layer to serializable data
    fn serialize_prefix_tuning_layer(layer: &PrefixTuningLayer) -> Result<SerializableLayerData> {
        let prefix_projection_weight = layer.prefix_projection.weight().data()?;
        let prefix_projection_bias = layer
            .prefix_projection
            .bias()
            .map(|b| b.data())
            .transpose()?
            .unwrap_or_default();
        let prefix_embeddings = layer.prefix_embeddings.data()?;

        Ok(SerializableLayerData::PrefixTuning {
            prefix_projection_weight,
            prefix_projection_bias,
            prefix_embeddings,
            prefix_length: layer.prefix_length,
            hidden_size: layer.hidden_size,
            num_layers: layer.num_layers,
            num_heads: layer.num_heads,
        })
    }

    /// Convert a PromptTuning embedding to serializable data
    fn serialize_prompt_tuning_embedding(
        embedding: &PromptTuningEmbedding,
    ) -> Result<SerializableLayerData> {
        let prompt_embeddings = embedding.prompt_embeddings.data()?;

        Ok(SerializableLayerData::PromptTuning {
            prompt_embeddings,
            num_virtual_tokens: embedding.num_virtual_tokens,
            hidden_size: embedding.hidden_size,
            init_method: embedding.init_method,
        })
    }

    /// Convert serializable data to an Adapter layer
    fn deserialize_adapter_layer(data: &SerializableLayerData) -> Result<AdapterLayer> {
        if let SerializableLayerData::Adapter {
            down_proj_weight,
            down_proj_bias,
            up_proj_weight,
            up_proj_bias,
            activation,
            bottleneck_size,
            dropout,
            residual_connection,
            hidden_size,
        } = data
        {
            let mut layer =
                AdapterLayer::new(*hidden_size, *bottleneck_size, *activation, *dropout);

            // Set down projection weights
            let down_weight_tensor =
                Tensor::from_vec(down_proj_weight.clone(), &[*bottleneck_size, *hidden_size])?;
            layer.down_proj.set_weight(down_weight_tensor)?;

            let down_bias_tensor = Tensor::from_vec(down_proj_bias.clone(), &[*bottleneck_size])?;
            layer.down_proj.set_bias(down_bias_tensor)?;

            // Set up projection weights
            let up_weight_tensor =
                Tensor::from_vec(up_proj_weight.clone(), &[*hidden_size, *bottleneck_size])?;
            layer.up_proj.set_weight(up_weight_tensor)?;

            let up_bias_tensor = Tensor::from_vec(up_proj_bias.clone(), &[*hidden_size])?;
            layer.up_proj.set_bias(up_bias_tensor)?;

            // Set configuration
            layer.residual_connection = *residual_connection;

            Ok(layer)
        } else {
            Err(TrustformersError::invalid_input(
                "Expected Adapter layer data".into(),
            ))
        }
    }

    /// Convert serializable data to a PrefixTuning layer
    fn deserialize_prefix_tuning_layer(data: &SerializableLayerData) -> Result<PrefixTuningLayer> {
        if let SerializableLayerData::PrefixTuning {
            prefix_projection_weight,
            prefix_projection_bias,
            prefix_embeddings,
            prefix_length,
            hidden_size,
            num_layers,
            num_heads,
        } = data
        {
            let mut layer =
                PrefixTuningLayer::new(*prefix_length, *hidden_size, *num_layers, *num_heads)?;

            // Set prefix projection weights
            let proj_weight_tensor = Tensor::from_vec(
                prefix_projection_weight.clone(),
                &[*hidden_size, *prefix_length],
            )?;
            layer.prefix_projection.set_weight(proj_weight_tensor)?;

            let proj_bias_tensor =
                Tensor::from_vec(prefix_projection_bias.clone(), &[*hidden_size])?;
            layer.prefix_projection.set_bias(proj_bias_tensor)?;

            // Set prefix embeddings
            let embeddings_tensor = Tensor::from_vec(
                prefix_embeddings.clone(),
                &[
                    *num_layers,
                    2,
                    *num_heads,
                    *prefix_length,
                    *hidden_size / *num_heads,
                ],
            )?;
            layer.prefix_embeddings = embeddings_tensor;

            Ok(layer)
        } else {
            Err(TrustformersError::invalid_input(
                "Expected PrefixTuning layer data".into(),
            ))
        }
    }

    /// Convert serializable data to a PromptTuning embedding
    fn deserialize_prompt_tuning_embedding(
        data: &SerializableLayerData,
    ) -> Result<PromptTuningEmbedding> {
        if let SerializableLayerData::PromptTuning {
            prompt_embeddings,
            num_virtual_tokens,
            hidden_size,
            init_method,
        } = data
        {
            let mut embedding =
                PromptTuningEmbedding::new(*num_virtual_tokens, *hidden_size, *init_method)?;

            // Set prompt embeddings
            let embeddings_tensor = Tensor::from_vec(
                prompt_embeddings.clone(),
                &[*num_virtual_tokens, *hidden_size],
            )?;
            embedding.prompt_embeddings = embeddings_tensor;

            Ok(embedding)
        } else {
            Err(TrustformersError::invalid_input(
                "Expected PromptTuning embedding data".into(),
            ))
        }
    }

    pub fn add_lora_layer(&mut self, name: String, layer: LoRALayer) {
        // Store serializable metadata
        if let Ok(metadata) = Self::serialize_lora_layer(&layer) {
            self.layer_metadata.insert(name.clone(), metadata);
        }
        self.peft_layers.insert(name, Box::new(layer));
    }

    pub fn add_adapter_layer(&mut self, name: String, layer: AdapterLayer) {
        // Store serializable metadata
        if let Ok(metadata) = Self::serialize_adapter_layer(&layer) {
            self.layer_metadata.insert(name.clone(), metadata);
        }
        self.peft_layers.insert(name, Box::new(layer));
    }

    pub fn add_prefix_tuning_layer(&mut self, name: String, layer: PrefixTuningLayer) {
        // Store serializable metadata
        if let Ok(metadata) = Self::serialize_prefix_tuning_layer(&layer) {
            self.layer_metadata.insert(name.clone(), metadata);
        }
        self.peft_layers.insert(name, Box::new(layer));
    }

    pub fn add_prompt_tuning_embedding(&mut self, name: String, embedding: PromptTuningEmbedding) {
        // Store serializable metadata
        if let Ok(metadata) = Self::serialize_prompt_tuning_embedding(&embedding) {
            self.layer_metadata.insert(name.clone(), metadata);
        }
        self.peft_layers.insert(name, Box::new(embedding));
    }

    pub fn enable_peft(&mut self) {
        self.active = true;
    }

    pub fn disable_peft(&mut self) {
        self.active = false;
    }

    pub fn merge_and_unload(&mut self) -> Result<()> {
        // Merge all LoRA layers
        for (name, layer) in &mut self.peft_layers {
            // This would need to be implemented per layer type
            // For now, just mark as merged
        }

        self.active = false;
        Ok(())
    }

    pub fn get_trainable_parameters(&self) -> Vec<String> {
        if !self.active {
            return Vec::new();
        }

        let mut trainable = Vec::new();
        for name in self.peft_layers.keys() {
            if self.config.target_modules.contains(name) {
                trainable.push(name.clone());
            }
        }

        trainable
    }

    pub fn save_pretrained(&self, path: &str) -> Result<()> {
        // Create directory if it doesn't exist
        std::fs::create_dir_all(path).map_err(|e| TrustformersError::io_error(e.to_string()))?;

        // Save PEFT configuration
        let config_json = serde_json::to_string_pretty(&self.config)
            .map_err(|e| TrustformersError::other(format!("Serialization error: {}", e)))?;
        std::fs::write(format!("{}/peft_config.json", path), config_json)
            .map_err(|e| TrustformersError::io_error(e.to_string()))?;

        // Save adapter weights using stored metadata
        let weights_json = serde_json::to_string_pretty(&self.layer_metadata)
            .map_err(|e| TrustformersError::other(format!("Serialization error: {}", e)))?;
        std::fs::write(format!("{}/adapter_weights.json", path), weights_json)
            .map_err(|e| TrustformersError::io_error(e.to_string()))?;

        Ok(())
    }

    pub fn load_pretrained(path: &str) -> Result<Self> {
        let config_str = std::fs::read_to_string(format!("{}/peft_config.json", path))
            .map_err(|e| TrustformersError::io_error(e.to_string()))?;

        let config: PeftConfig = serde_json::from_str(&config_str)
            .map_err(|e| TrustformersError::other(format!("Serialization error: {}", e)))?;

        let mut model = Self::new(config);

        // Load adapter weights
        let weights_str = std::fs::read_to_string(format!("{}/adapter_weights.json", path))
            .map_err(|e| TrustformersError::io_error(e.to_string()))?;

        let layer_metadata: HashMap<String, SerializableLayerData> =
            serde_json::from_str(&weights_str)
                .map_err(|e| TrustformersError::other(format!("Serialization error: {}", e)))?;

        // Reconstruct layers from metadata
        for (name, data) in layer_metadata {
            match &data {
                SerializableLayerData::LoRA { .. } => {
                    let layer = Self::deserialize_lora_layer(&data)?;
                    model.add_lora_layer(name, layer);
                },
                SerializableLayerData::Adapter { .. } => {
                    let layer = Self::deserialize_adapter_layer(&data)?;
                    model.add_adapter_layer(name, layer);
                },
                SerializableLayerData::PrefixTuning { .. } => {
                    let layer = Self::deserialize_prefix_tuning_layer(&data)?;
                    model.add_prefix_tuning_layer(name, layer);
                },
                SerializableLayerData::PromptTuning { .. } => {
                    let embedding = Self::deserialize_prompt_tuning_embedding(&data)?;
                    model.add_prompt_tuning_embedding(name, embedding);
                },
            }
        }

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_layer_creation() {
        let lora = LoRALayer::new(768, 768, 8, 16.0, 0.1, true).unwrap();
        assert_eq!(lora.r, 8);
        assert_eq!(lora.alpha, 16.0);
        assert!(!lora.merged);
        assert!(lora.frozen);
    }

    #[test]
    fn test_lora_layer_forward() {
        let mut lora = LoRALayer::new(64, 64, 4, 8.0, 0.0, false).unwrap();
        lora.initialize_weights().unwrap();

        let input = Tensor::randn(&[10, 64]).expect("Failed to create random tensor");
        let output = lora.forward(input.clone()).unwrap();

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_lora_merge_unmerge() {
        let mut lora = LoRALayer::new(32, 32, 2, 4.0, 0.0, false).unwrap();
        lora.initialize_weights().unwrap();

        assert!(!lora.merged);

        lora.merge_weights().unwrap();
        assert!(lora.merged);

        lora.unmerge_weights().unwrap();
        assert!(!lora.merged);
    }

    #[test]
    fn test_qlora_layer() {
        let mut qlora = QLoRALayer::new(64, 64, 4, 8.0, 0.1, false).unwrap();

        let quant_config = crate::quantization::QuantizationConfig::default();
        qlora.quantize_base(&quant_config).unwrap();

        let input = Tensor::randn(&[5, 64]).expect("Failed to create random tensor");
        let output = qlora.forward(input.clone()).unwrap();

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_adapter_layer() {
        let adapter = AdapterLayer::new(128, 32, ActivationType::GELU, 0.1);
        assert_eq!(adapter.bottleneck_size, 32);

        let input = Tensor::randn(&[8, 128]).expect("Failed to create random tensor");
        let output = adapter.forward(input.clone()).unwrap();

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_prefix_tuning_layer() {
        let prefix = PrefixTuningLayer::new(10, 64, 12, 8).unwrap();
        assert_eq!(prefix.prefix_length, 10);
        assert_eq!(prefix.num_layers, 12);

        let prefix_states = prefix.get_prefix_states().unwrap();
        assert_eq!(prefix_states.len(), 12);
    }

    #[test]
    fn test_prompt_tuning_embedding() {
        let prompt = PromptTuningEmbedding::new(5, 768, PromptInitMethod::Random).unwrap();
        assert_eq!(prompt.num_virtual_tokens, 5);
        assert_eq!(prompt.hidden_size, 768);

        let embeddings = prompt.get_prompt_embeddings();
        assert_eq!(embeddings.shape(), vec![5, 768]);
    }

    #[test]
    fn test_peft_model() {
        let config = PeftConfig::default();
        let mut peft_model = PeftModel::new(config);

        let lora = LoRALayer::new(64, 64, 4, 8.0, 0.1, false).unwrap();
        peft_model.add_lora_layer("test_layer".to_string(), lora);

        assert_eq!(peft_model.peft_layers.len(), 1);
        assert!(peft_model.active);

        peft_model.disable_peft();
        assert!(!peft_model.active);
    }

    #[test]
    fn test_peft_config_serialization() {
        let config = PeftConfig::default();
        let json = serde_json::to_string(&config).expect("JSON serialization failed");
        let deserialized: PeftConfig =
            serde_json::from_str(&json).expect("JSON deserialization failed");

        assert_eq!(config.method, deserialized.method);
        assert_eq!(config.r, deserialized.r);
        assert_eq!(config.alpha, deserialized.alpha);
    }

    #[test]
    fn test_activation_functions() {
        let adapter = AdapterLayer::new(64, 16, ActivationType::ReLU, 0.0);
        let input =
            Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).expect("Tensor from_vec failed");

        let relu_result = adapter.relu(&input).unwrap();
        let data = relu_result.data().unwrap();
        assert_eq!(data[0], 0.0); // ReLU(-1) = 0
        assert_eq!(data[1], 0.0); // ReLU(0) = 0
        assert_eq!(data[2], 1.0); // ReLU(1) = 1
        assert_eq!(data[3], 2.0); // ReLU(2) = 2
    }
}
