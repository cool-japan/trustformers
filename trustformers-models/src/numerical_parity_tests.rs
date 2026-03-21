//! Numerical parity tests to ensure our implementations match reference outputs
//!
//! This module provides utilities for testing numerical accuracy and consistency
//! of model implementations against known good values.

use anyhow::Result;
use approx::assert_abs_diff_eq;
use trustformers_core::tensor::Tensor;

/// Test utilities for numerical parity validation
pub struct NumericalParityTests;

impl NumericalParityTests {
    /// Compare two tensors for numerical equality within tolerance
    pub fn assert_tensors_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
        match (actual, expected) {
            (Tensor::F32(actual_arr), Tensor::F32(expected_arr)) => {
                if actual_arr.shape() != expected_arr.shape() {
                    return Err(anyhow::anyhow!(
                        "Shape mismatch: {:?} vs {:?}",
                        actual_arr.shape(),
                        expected_arr.shape()
                    ));
                }

                for (a, e) in actual_arr.iter().zip(expected_arr.iter()) {
                    assert_abs_diff_eq!(a, e, epsilon = tolerance);
                }
            },
            _ => {
                return Err(anyhow::anyhow!(
                    "Only F32 tensors are supported for comparison"
                ));
            },
        }

        Ok(())
    }

    /// Test that tensor values are finite (no NaN or infinity)
    pub fn assert_tensor_finite(tensor: &Tensor) -> Result<()> {
        match tensor {
            Tensor::F32(arr) => {
                for &value in arr.iter() {
                    if !value.is_finite() {
                        return Err(anyhow::anyhow!("Non-finite value found: {}", value));
                    }
                }
            },
            _ => {
                return Err(anyhow::anyhow!(
                    "Only F32 tensors are supported for finite check"
                ));
            },
        }

        Ok(())
    }

    /// Test that tensor values are within expected range
    pub fn assert_tensor_range(tensor: &Tensor, min_val: f32, max_val: f32) -> Result<()> {
        match tensor {
            Tensor::F32(arr) => {
                for &value in arr.iter() {
                    if value < min_val || value > max_val {
                        return Err(anyhow::anyhow!(
                            "Value {} outside expected range [{}, {}]",
                            value,
                            min_val,
                            max_val
                        ));
                    }
                }
            },
            _ => {
                return Err(anyhow::anyhow!(
                    "Only F32 tensors are supported for range check"
                ));
            },
        }

        Ok(())
    }

    /// Compute relative error between two tensors
    pub fn compute_relative_error(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        match (actual, expected) {
            (Tensor::F32(actual_arr), Tensor::F32(expected_arr)) => {
                if actual_arr.shape() != expected_arr.shape() {
                    return Err(anyhow::anyhow!(
                        "Shape mismatch: {:?} vs {:?}",
                        actual_arr.shape(),
                        expected_arr.shape()
                    ));
                }

                let mut max_relative_error = 0.0f32;
                for (a, e) in actual_arr.iter().zip(expected_arr.iter()) {
                    if e.abs() > 1e-8 {
                        let relative_error = ((a - e) / e).abs();
                        max_relative_error = max_relative_error.max(relative_error);
                    }
                }

                Ok(max_relative_error)
            },
            _ => Err(anyhow::anyhow!(
                "Only F32 tensors are supported for error computation"
            )),
        }
    }

    /// Generate test data for parity testing
    pub fn generate_test_data(
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<TestData> {
        Ok(TestData {
            input_ids: (0..batch_size * seq_len).map(|i| (i % 1000) as u32).collect(),
            attention_mask: vec![1u32; batch_size * seq_len],
            expected_shape: vec![batch_size, seq_len, hidden_size],
        })
    }
}

/// Test data structure for parity tests
#[derive(Debug, Clone)]
pub struct TestData {
    /// Input token IDs
    pub input_ids: Vec<u32>,
    /// Attention mask
    pub attention_mask: Vec<u32>,
    /// Expected output shape
    pub expected_shape: Vec<usize>,
}

/// Reference values for numerical parity tests
pub struct ReferenceValues;

impl ReferenceValues {
    /// Get BERT base reference values (dummy values for testing framework)
    pub fn bert_base_uncased() -> ReferenceData {
        ReferenceData {
            model_name: "bert-base-uncased".to_string(),
            hidden_size: 768,
            num_layers: 12,
            num_attention_heads: 12,
            vocab_size: 30522,
            max_position_embeddings: 512,
            // These would be actual reference values in practice
            sample_outputs: vec![],
        }
    }

    /// Get GPT-2 reference values (dummy values for testing framework)
    pub fn gpt2() -> ReferenceData {
        ReferenceData {
            model_name: "gpt2".to_string(),
            hidden_size: 768,
            num_layers: 12,
            num_attention_heads: 12,
            vocab_size: 50257,
            max_position_embeddings: 1024,
            sample_outputs: vec![],
        }
    }
}

/// Reference data structure
#[derive(Debug, Clone)]
pub struct ReferenceData {
    /// Model name
    pub model_name: String,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// Sample output tensors for comparison
    pub sample_outputs: Vec<Tensor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_comparison() -> Result<()> {
        let tensor1 = Tensor::ones(&[2, 3])?;
        let tensor2 = Tensor::ones(&[2, 3])?;

        NumericalParityTests::assert_tensors_close(&tensor1, &tensor2, 1e-6)?;
        Ok(())
    }

    #[test]
    fn test_tensor_finite() -> Result<()> {
        let tensor = Tensor::ones(&[2, 3])?;
        NumericalParityTests::assert_tensor_finite(&tensor)?;
        Ok(())
    }

    #[test]
    fn test_tensor_range() -> Result<()> {
        let tensor = Tensor::ones(&[2, 3])?;
        NumericalParityTests::assert_tensor_range(&tensor, 0.0, 2.0)?;
        Ok(())
    }

    #[test]
    fn test_generate_test_data() -> Result<()> {
        let test_data = NumericalParityTests::generate_test_data(2, 10, 768)?;
        assert_eq!(test_data.input_ids.len(), 20);
        assert_eq!(test_data.attention_mask.len(), 20);
        assert_eq!(test_data.expected_shape, vec![2, 10, 768]);
        Ok(())
    }

    #[test]
    fn test_reference_values() {
        let bert_ref = ReferenceValues::bert_base_uncased();
        assert_eq!(bert_ref.hidden_size, 768);
        assert_eq!(bert_ref.vocab_size, 30522);

        let gpt2_ref = ReferenceValues::gpt2();
        assert_eq!(gpt2_ref.hidden_size, 768);
        assert_eq!(gpt2_ref.vocab_size, 50257);
    }
}
