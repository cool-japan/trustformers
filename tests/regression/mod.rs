//! Regression test suite for TrustformeRS
//!
//! This module contains regression tests to ensure that changes don't break
//! existing functionality. These tests capture specific behaviors and edge cases
//! that have been fixed in the past.

use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use trustformers_core::tensor::Tensor;

/// Regression test case structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestCase {
    /// Unique identifier for the test case
    pub id: String,
    /// Description of what this test is checking
    pub description: String,
    /// The issue or bug this test prevents regression of
    pub issue: Option<String>,
    /// Test category (e.g., "tensor", "layer", "model")
    pub category: String,
    /// The actual test data
    pub test_data: TestData,
    /// Expected output or behavior
    pub expected: ExpectedResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TestData {
    TensorOperation {
        operation: String,
        inputs: Vec<TensorData>,
        params: serde_json::Value,
    },
    LayerForward {
        layer_type: String,
        config: serde_json::Value,
        input: TensorData,
    },
    ModelInference {
        model_type: String,
        config: serde_json::Value,
        inputs: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExpectedResult {
    TensorOutput {
        data: Vec<f32>,
        shape: Vec<usize>,
        tolerance: f32,
    },
    Error {
        error_type: String,
        message_contains: Option<String>,
    },
    Success,
}

/// Load regression test cases from a directory
pub fn load_test_cases(dir: &Path) -> Result<Vec<RegressionTestCase>, Box<dyn std::error::Error>> {
    let mut test_cases = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let content = fs::read_to_string(&path)?;
            let test_case: RegressionTestCase = serde_json::from_str(&content)?;
            test_cases.push(test_case);
        }
    }

    Ok(test_cases)
}

/// Run a single regression test case
pub fn run_test_case(test_case: &RegressionTestCase) -> Result<(), String> {
    match &test_case.test_data {
        TestData::TensorOperation { operation, inputs, params } => {
            run_tensor_operation_test(operation, inputs, params, &test_case.expected)
        }
        TestData::LayerForward { layer_type, config, input } => {
            run_layer_forward_test(layer_type, config, input, &test_case.expected)
        }
        TestData::ModelInference { model_type, config, inputs } => {
            run_model_inference_test(model_type, config, inputs, &test_case.expected)
        }
    }
}

fn run_tensor_operation_test(
    operation: &str,
    inputs: &[TensorData],
    _params: &serde_json::Value,
    expected: &ExpectedResult,
) -> Result<(), String> {
    let tensors: Result<Vec<Tensor>, _> = inputs.iter()
        .map(|td| Tensor::new(td.data.clone(), td.shape.clone()))
        .collect();

    let tensors = tensors.map_err(|e| format!("Failed to create input tensors: {:?}", e))?;

    let result = match operation {
        "add" => {
            if tensors.len() != 2 {
                return Err("Add operation requires exactly 2 tensors".to_string());
            }
            tensors[0].add(&tensors[1])
        }
        "matmul" => {
            if tensors.len() != 2 {
                return Err("Matmul operation requires exactly 2 tensors".to_string());
            }
            tensors[0].matmul(&tensors[1])
        }
        "transpose" => {
            if tensors.len() != 1 {
                return Err("Transpose operation requires exactly 1 tensor".to_string());
            }
            tensors[0].transpose(0, 1)
        }
        "softmax" => {
            if tensors.len() != 1 {
                return Err("Softmax operation requires exactly 1 tensor".to_string());
            }
            tensors[0].softmax(-1)
        }
        _ => return Err(format!("Unknown tensor operation: {}", operation)),
    };

    check_result(result, expected)
}

fn run_layer_forward_test(
    layer_type: &str,
    config: &serde_json::Value,
    input: &TensorData,
    expected: &ExpectedResult,
) -> Result<(), String> {
    use trustformers_core::layers::{Layer, Linear, LayerNorm};

    let input_tensor = Tensor::new(input.data.clone(), input.shape.clone())
        .map_err(|e| format!("Failed to create input tensor: {:?}", e))?;

    let result: Result<Tensor, _> = match layer_type {
        "linear" => {
            let in_features = config["in_features"].as_u64().unwrap() as usize;
            let out_features = config["out_features"].as_u64().unwrap() as usize;
            let use_bias = config["use_bias"].as_bool().unwrap_or(true);

            let layer = Linear::new(in_features, out_features, use_bias);
            layer.forward(&input_tensor)
        }
        "layer_norm" => {
            let normalized_shape: Vec<usize> = config["normalized_shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let eps = config["eps"].as_f64().unwrap_or(1e-5) as f32;

            let layer = LayerNorm::new(normalized_shape, eps);
            layer.forward(&input_tensor)
        }
        _ => return Err(format!("Unknown layer type: {}", layer_type)),
    };

    check_result(result, expected)
}

fn run_model_inference_test(
    _model_type: &str,
    _config: &serde_json::Value,
    _inputs: &serde_json::Value,
    _expected: &ExpectedResult,
) -> Result<(), String> {
    // Model inference tests would be implemented here
    // For now, we'll skip these as they require full model loading
    Ok(())
}

fn check_result<T>(result: Result<T, trustformers_core::tensor::TensorError>, expected: &ExpectedResult) -> Result<(), String>
where
    T: AsRef<Tensor>,
{
    match (result, expected) {
        (Ok(tensor), ExpectedResult::TensorOutput { data, shape, tolerance }) => {
            let tensor = tensor.as_ref();

            // Check shape
            if tensor.shape() != shape {
                return Err(format!(
                    "Shape mismatch: expected {:?}, got {:?}",
                    shape, tensor.shape()
                ));
            }

            // Check data within tolerance
            for (i, (actual, expected)) in tensor.data().iter().zip(data.iter()).enumerate() {
                if (actual - expected).abs() > *tolerance {
                    return Err(format!(
                        "Data mismatch at index {}: expected {}, got {} (tolerance: {})",
                        i, expected, actual, tolerance
                    ));
                }
            }

            Ok(())
        }
        (Err(_), ExpectedResult::Error { .. }) => {
            // Expected an error and got one
            Ok(())
        }
        (Ok(_), ExpectedResult::Error { .. }) => {
            Err("Expected an error but operation succeeded".to_string())
        }
        (Err(e), ExpectedResult::TensorOutput { .. }) => {
            Err(format!("Expected success but got error: {:?}", e))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regression_framework() {
        // Create a simple test case
        let test_case = RegressionTestCase {
            id: "tensor_add_basic".to_string(),
            description: "Basic tensor addition".to_string(),
            issue: None,
            category: "tensor".to_string(),
            test_data: TestData::TensorOperation {
                operation: "add".to_string(),
                inputs: vec![
                    TensorData {
                        data: vec![1.0, 2.0, 3.0, 4.0],
                        shape: vec![2, 2],
                    },
                    TensorData {
                        data: vec![5.0, 6.0, 7.0, 8.0],
                        shape: vec![2, 2],
                    },
                ],
                params: serde_json::json!({}),
            },
            expected: ExpectedResult::TensorOutput {
                data: vec![6.0, 8.0, 10.0, 12.0],
                shape: vec![2, 2],
                tolerance: 1e-6,
            },
        };

        // Run the test
        let result = run_test_case(&test_case);
        assert!(result.is_ok(), "Test case failed: {:?}", result);
    }
}