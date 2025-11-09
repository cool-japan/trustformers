use crate::core::tensor::WasmTensor;
use serde::{Deserialize, Serialize};
use std::vec::Vec;
use std::{format, vec};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    weight: WasmTensor,
    bias: Option<WasmTensor>,
    in_features: usize,
    out_features: usize,
}

#[wasm_bindgen]
impl Linear {
    #[wasm_bindgen(constructor)]
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Result<Linear, JsValue> {
        let weight = WasmTensor::randn(vec![out_features, in_features])
            .map_err(|e| JsValue::from_str(&format!("Failed to create weight: {e:?}")))?;

        let bias = if use_bias {
            Some(
                WasmTensor::zeros(vec![out_features])
                    .map_err(|e| JsValue::from_str(&format!("Failed to create bias: {e:?}")))?,
            )
        } else {
            None
        };

        Ok(Linear {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    pub fn forward(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        // Input shape: [batch_size, in_features]
        // Weight shape: [out_features, in_features]
        // Output shape: [batch_size, out_features]

        // Transpose weight for matmul
        let weight_t = self.weight.transpose()?;
        let output = input.matmul(&weight_t)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            // Broadcast bias across batch dimension
            output.add(bias)
        } else {
            Ok(output)
        }
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    weight: WasmTensor,
    bias: WasmTensor,
    eps: f32,
}

#[wasm_bindgen]
impl LayerNorm {
    #[wasm_bindgen(constructor)]
    pub fn new(normalized_shape: Vec<usize>, eps: f32) -> Result<LayerNorm, JsValue> {
        let size: usize = normalized_shape.iter().product();
        let weight = WasmTensor::ones(vec![size])
            .map_err(|e| JsValue::from_str(&format!("Failed to create weight: {e:?}")))?;
        let bias = WasmTensor::zeros(vec![size])
            .map_err(|e| JsValue::from_str(&format!("Failed to create bias: {e:?}")))?;

        Ok(LayerNorm {
            normalized_shape,
            weight,
            bias,
            eps,
        })
    }

    pub fn forward(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        // Simplified layer norm for 2D tensors
        let data = input.data();
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(JsValue::from_str(
                "LayerNorm currently only supports 2D tensors",
            ));
        }

        let batch_size = shape[0];
        let features = shape[1];

        let mut output_data = Vec::with_capacity(data.len());

        for i in 0..batch_size {
            let start = i * features;
            let end = start + features;
            let row = &data[start..end];

            // Compute mean
            let mean: f32 = row.iter().sum::<f32>() / features as f32;

            // Compute variance
            let variance: f32 =
                row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / features as f32;

            // Normalize
            let std = (variance + self.eps).sqrt();

            for ((&val, &weight), &bias) in
                row.iter().zip(self.weight.data().iter()).zip(self.bias.data().iter())
            {
                let normalized = (val - mean) / std;
                let scaled = normalized * weight + bias;
                output_data.push(scaled);
            }
        }

        WasmTensor::new(output_data, shape)
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    weight: WasmTensor,
    num_embeddings: usize,
    embedding_dim: usize,
}

#[wasm_bindgen]
impl Embedding {
    #[wasm_bindgen(constructor)]
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Result<Embedding, JsValue> {
        let weight = WasmTensor::randn(vec![num_embeddings, embedding_dim])?;

        Ok(Embedding {
            weight,
            num_embeddings,
            embedding_dim,
        })
    }

    pub fn forward(&self, input_ids: &[usize]) -> Result<WasmTensor, JsValue> {
        let batch_size = input_ids.len();
        let mut output_data = Vec::with_capacity(batch_size * self.embedding_dim);

        for &idx in input_ids {
            if idx >= self.num_embeddings {
                return Err(JsValue::from_str(&format!(
                    "Index {} out of range for {} embeddings",
                    idx, self.num_embeddings
                )));
            }

            let start = idx * self.embedding_dim;
            let end = start + self.embedding_dim;
            output_data.extend_from_slice(&self.weight.data()[start..end]);
        }

        WasmTensor::new(output_data, vec![batch_size, self.embedding_dim])
    }
}

#[wasm_bindgen]
pub struct Dropout {
    p: f32,
    training: bool,
}

#[wasm_bindgen]
impl Dropout {
    #[wasm_bindgen(constructor)]
    pub fn new(p: f32) -> Dropout {
        Dropout { p, training: true }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    pub fn forward(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        // For now, just return input as dropout requires RNG
        // In a real implementation, we'd apply dropout mask
        Ok(input.clone())
    }
}

// Activation functions
#[wasm_bindgen]
pub fn relu(input: &WasmTensor) -> WasmTensor {
    input.relu()
}

#[wasm_bindgen]
pub fn gelu(input: &WasmTensor) -> WasmTensor {
    input.gelu()
}

#[wasm_bindgen]
pub fn softmax(input: &WasmTensor, dim: i32) -> Result<WasmTensor, JsValue> {
    input.softmax(dim)
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer() {
        let linear = Linear::new(3, 2, true).unwrap();
        let input = WasmTensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let output = linear.forward(&input).unwrap();
        assert_eq!(output.shape(), vec![1, 2]);
    }

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let input =
            WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]).unwrap();
        let output = ln.forward(&input).unwrap();
        assert_eq!(output.shape(), vec![2, 4]);
    }

    #[test]
    fn test_embedding() {
        let emb = Embedding::new(10, 4).unwrap();
        let output = emb.forward(&[0, 2, 5]).unwrap();
        assert_eq!(output.shape(), vec![3, 4]);
    }
}
