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
        // Input shape: [..., in_features]
        // Weight shape: [out_features, in_features]
        // Output shape: [..., out_features]

        // Transpose weight for matmul
        let weight_t = self.weight.transpose()?;
        let output = input.matmul(&weight_t)?;

        // Add bias if present. `WasmTensor::add` requires exact shape
        // equality (no broadcasting), so a naive `output.add(bias)` here
        // always fails with "Shape mismatch for addition" (bias is 1D
        // `[out_features]`, output is `[..., out_features]`) - bias was
        // therefore never actually applied. Broadcast the bias manually
        // over every row of the last dimension instead.
        if let Some(ref bias) = self.bias {
            let out_shape = output.shape();
            if out_shape.last().copied() != Some(self.out_features) {
                return Err(JsValue::from_str(&format!(
                    "Linear forward: output shape {out_shape:?} does not end in out_features {}",
                    self.out_features
                )));
            }

            let bias_data = bias.data();
            let broadcasted: Vec<f32> = output
                .data()
                .chunks(self.out_features)
                .flat_map(|row| row.iter().zip(bias_data.iter()).map(|(&v, &b)| v + b))
                .collect();

            WasmTensor::new(broadcasted, out_shape)
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
    /// Create a new dropout layer.
    ///
    /// Defaults to `training = false` (inert / identity pass-through).
    /// This crate is inference-only: there is no backward pass, optimizer,
    /// or `.train()`/`.eval()` plumbing anywhere, and none of `models.rs`'s
    /// `AttentionHead`/`BertLayer`/`BertEmbeddings` expose a way to reach
    /// this layer's `set_training`. Defaulting to `true` (as an earlier
    /// version of this constructor did) would have made every
    /// `BertModelWasm`/`TextGenerator` forward pass silently
    /// non-deterministic the moment [`Self::forward`] applied a real
    /// dropout mask instead of the previous (fake) `Ok(input.clone())` -
    /// with no way for a caller to disable it. Training mode remains fully
    /// real and available via `set_training(true)` for callers that do
    /// implement their own training loop around this layer.
    #[wasm_bindgen(constructor)]
    pub fn new(p: f32) -> Dropout {
        Dropout { p, training: false }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Real inverted dropout: during training, each element is independently
    /// zeroed with probability `p` (drawn from a genuine entropy source, not
    /// fabricated), and surviving elements are scaled by `1 / (1 - p)` so the
    /// expected activation magnitude is unchanged between train and eval. In
    /// eval mode (`training == false`) or when `p == 0.0`, this is an
    /// identity pass-through, matching standard framework semantics.
    ///
    /// Draws come from [`Self::real_uniform_draws`] rather than
    /// [`WasmTensor::random_uniform`] deliberately: `random_uniform`'s error
    /// path silently substitutes a *fixed*-seed pseudo-random sequence
    /// (the same sequence every call) when the real entropy source is
    /// unavailable, which for dropout would mean applying the same fixed
    /// mask on every forward pass - a static pruning pattern, not dropout,
    /// and a silent-fallback violation of this crate's no-fabrication
    /// policy. A real entropy failure is therefore surfaced as an `Err`
    /// here instead.
    pub fn forward(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let shape = input.shape();

        if self.p >= 1.0 {
            // Every element is dropped.
            return WasmTensor::zeros(shape);
        }

        let data = input.data();
        let draws = Self::real_uniform_draws(data.len())
            .map_err(|e| JsValue::from_str(&format!("Dropout requires real randomness: {e}")))?;
        let keep_scale = 1.0 / (1.0 - self.p);

        let output_data: Vec<f32> = data
            .iter()
            .zip(draws.iter())
            .map(|(&value, &draw)| if draw < self.p { 0.0 } else { value * keep_scale })
            .collect();

        WasmTensor::new(output_data, shape)
    }

    /// Draw `count` independent uniform values in `[0, 1)` from the system's
    /// real entropy source (`getrandom`), returning a structured `Err`
    /// rather than a fixed pseudo-random substitute when entropy is
    /// unavailable. See [`Self::forward`] for why this does not delegate to
    /// [`WasmTensor::random_uniform`].
    fn real_uniform_draws(count: usize) -> Result<Vec<f32>, String> {
        let mut raw_bytes = vec![0u8; count * 4];
        getrandom::fill(&mut raw_bytes).map_err(|e| format!("entropy source failed: {e:?}"))?;

        Ok(raw_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                // Top 24 bits -> a value uniformly distributed in [0, 1).
                (bits >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect())
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

// Note: this module previously ran only `#[cfg(all(test, target_arch =
// "wasm32"))]`, so it was never exercised by native `cargo test`/`cargo
// nextest` runs and a compile error here (missing `.expect(..)` on
// `LayerNorm::new`, which returns `Result`) went unnoticed until a real
// `cargo check --target wasm32-unknown-unknown --all-targets` was run.
// None of `Linear`/`LayerNorm`/`Embedding`/`Dropout` touch `web_sys`/`js_sys`
// browser-only APIs, so the module is safe to run on every target.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer() {
        let linear = Linear::new(3, 2, true).expect("Linear layer creation should succeed");
        let input = WasmTensor::new(vec![1.0, 2.0, 3.0], vec![1, 3])
            .expect("tensor creation should succeed");
        let output = linear.forward(&input).expect("forward pass should succeed");
        assert_eq!(output.shape(), vec![1, 2]);
    }

    #[test]
    fn test_linear_layer_with_bias_actually_broadcasts_and_adds_bias() {
        // Old code called `output.add(bias)` where `output` is
        // `[batch_size, out_features]` and `bias` is `[out_features]` - an
        // exact-shape-only add, so this always returned a shape-mismatch
        // error (which panics when converted to `JsValue` on native
        // targets). Verify bias is now genuinely broadcast-added per row.
        let mut linear = Linear::new(2, 3, true).expect("Linear layer creation should succeed");
        // Force deterministic, known weight/bias so we can hand-check the result.
        linear.weight = WasmTensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2])
            .expect("tensor creation should succeed");
        linear.bias = Some(
            WasmTensor::new(vec![10.0, 20.0, 30.0], vec![3])
                .expect("tensor creation should succeed"),
        );

        let input = WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .expect("tensor creation should succeed");
        let output = linear.forward(&input).expect("forward pass should succeed");

        assert_eq!(output.shape(), vec![2, 3]);
        // Row 0: [1,2] @ W^T = [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3] + bias -> [11, 22, 33]
        // Row 1: [3,4] @ W^T = [3, 4, 7] + bias -> [13, 24, 37]
        let expected = [11.0, 22.0, 33.0, 13.0, 24.0, 37.0];
        for (got, want) in output.data().iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(vec![4], 1e-5).expect("LayerNorm creation should succeed");
        let input = WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4])
            .expect("tensor creation should succeed");
        let output = ln.forward(&input).expect("forward pass should succeed");
        assert_eq!(output.shape(), vec![2, 4]);
    }

    #[test]
    fn test_embedding() {
        let emb = Embedding::new(10, 4).expect("embedding creation should succeed");
        let output = emb.forward(&[0, 2, 5]).expect("forward pass should succeed");
        assert_eq!(output.shape(), vec![3, 4]);
    }

    // --- Dropout: was `Ok(input.clone())` unconditionally (comment: "For
    // now, just return input as dropout requires RNG"), a silent no-op that
    // never dropped anything even in training mode. These tests fail
    // against that old behavior.

    #[test]
    fn test_dropout_eval_mode_is_identity() {
        let mut dropout = Dropout::new(0.9);
        dropout.set_training(false);
        let input = WasmTensor::new((0..64).map(|i| i as f32 + 1.0).collect(), vec![64])
            .expect("tensor creation should succeed");
        let output = dropout.forward(&input).expect("forward pass should succeed");
        assert_eq!(output.data(), input.data());
    }

    #[test]
    fn test_dropout_new_defaults_to_eval_mode() {
        // This crate is inference-only (no backward pass/optimizer/train
        // loop anywhere), and `models.rs`'s `AttentionHead`/`BertLayer`/
        // `BertEmbeddings` construct `Dropout` internally with no way for a
        // caller to reach `set_training`. If the constructor defaulted to
        // `training = true`, making dropout real (instead of the old
        // `Ok(input.clone())` no-op) would have silently turned every
        // `BertModelWasm`/`TextGenerator` forward pass non-deterministic
        // with no way to opt out. The constructor must default to eval
        // mode (identity pass-through) so real dropout is opt-in only.
        let dropout = Dropout::new(0.9);
        let input = WasmTensor::new((0..64).map(|i| i as f32 + 1.0).collect(), vec![64])
            .expect("tensor creation should succeed");
        let output = dropout.forward(&input).expect("forward pass should succeed");
        assert_eq!(output.data(), input.data());
    }

    #[test]
    fn test_dropout_p_zero_is_identity() {
        let mut dropout = Dropout::new(0.0);
        dropout.set_training(true);
        let input = WasmTensor::new((0..64).map(|i| i as f32 + 1.0).collect(), vec![64])
            .expect("tensor creation should succeed");
        let output = dropout.forward(&input).expect("forward pass should succeed");
        assert_eq!(output.data(), input.data());
    }

    #[test]
    fn test_dropout_p_one_zeroes_everything() {
        let mut dropout = Dropout::new(1.0);
        dropout.set_training(true);
        let input =
            WasmTensor::new(vec![1.0; 32], vec![32]).expect("tensor creation should succeed");
        let output = dropout.forward(&input).expect("forward pass should succeed");
        assert!(output.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dropout_training_mode_actually_drops_and_scales() {
        // Old fake behavior always returned the input unchanged; a real
        // inverted-dropout pass over enough elements must produce both an
        // exact zero (a dropped unit) and a scaled survivor, and every
        // element must be one of those two possibilities.
        let mut dropout = Dropout::new(0.5);
        dropout.set_training(true);
        let size = 4096;
        let input =
            WasmTensor::new(vec![1.0; size], vec![size]).expect("tensor creation should succeed");
        let output = dropout.forward(&input).expect("forward pass should succeed");
        let data = output.data();

        let expected_scale = 1.0 / (1.0 - 0.5);
        let mut saw_zero = false;
        let mut saw_scaled_survivor = false;
        for &v in data.iter() {
            if v == 0.0 {
                saw_zero = true;
            } else {
                assert!(
                    (v - expected_scale).abs() < 1e-5,
                    "surviving element should be scaled by 1/(1-p): got {v}"
                );
                saw_scaled_survivor = true;
            }
        }
        assert!(
            saw_zero,
            "expected at least one dropped element out of {size}"
        );
        assert!(
            saw_scaled_survivor,
            "expected at least one surviving element out of {size}"
        );

        // Old code returned the unscaled input verbatim; verify we differ.
        assert_ne!(data, input.data());
    }

    #[test]
    fn test_real_uniform_draws_are_in_unit_range() {
        let draws =
            Dropout::real_uniform_draws(1000).expect("entropy source should be available in tests");
        assert_eq!(draws.len(), 1000);
        for &d in &draws {
            assert!((0.0..1.0).contains(&d), "draw {d} out of [0,1) range");
        }
    }

    #[test]
    fn test_real_uniform_draws_differ_across_calls() {
        // `WasmTensor::random_uniform`'s error-fallback path resets to a
        // fixed seed on every call, so repeated draws would be identical
        // whenever the primary entropy source was unavailable - exactly the
        // silent-fallback failure mode `Dropout` deliberately avoids by not
        // using it. A real entropy source must not repeat like that.
        let first =
            Dropout::real_uniform_draws(64).expect("entropy source should be available in tests");
        let second =
            Dropout::real_uniform_draws(64).expect("entropy source should be available in tests");
        assert_ne!(first, second);
    }
}
