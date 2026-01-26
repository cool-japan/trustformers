//! Cross-Framework Optimizer Conversion
//!
//! This module provides utilities to convert optimizer configurations and states
//! between different ML frameworks (PyTorch, TensorFlow, JAX, ONNX).

use crate::{
    onnx_export::OptimizerConfig as ONNXConfig, tensorflow_compat::TensorFlowOptimizerConfig,
};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Supported ML frameworks for conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Framework {
    PyTorch,
    TensorFlow,
    JAX,
    ONNX,
    TrustformeRS,
}

/// Universal optimizer configuration that can be converted between frameworks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalOptimizerConfig {
    pub optimizer_type: String,
    pub learning_rate: f32,
    pub parameters: HashMap<String, Value>,
    pub source_framework: Framework,
}

/// Universal optimizer state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalOptimizerState {
    pub step: i64,
    pub state_dict: HashMap<String, Value>,
    pub framework: Framework,
}

/// Cross-framework optimizer converter
pub struct CrossFrameworkConverter {
    // Mapping between parameter names across frameworks
    parameter_mappings: HashMap<(Framework, Framework), HashMap<String, String>>,
}

impl CrossFrameworkConverter {
    /// Create a new cross-framework converter
    pub fn new() -> Self {
        let mut converter = Self {
            parameter_mappings: HashMap::new(),
        };
        converter.initialize_mappings();
        converter
    }

    /// Initialize parameter name mappings between frameworks
    fn initialize_mappings(&mut self) {
        // PyTorch to TensorFlow mappings
        let mut pytorch_to_tf = HashMap::new();
        pytorch_to_tf.insert("lr".to_string(), "learning_rate".to_string());
        pytorch_to_tf.insert("betas".to_string(), "beta_1_beta_2".to_string());
        pytorch_to_tf.insert("eps".to_string(), "epsilon".to_string());
        pytorch_to_tf.insert("weight_decay".to_string(), "weight_decay".to_string());
        self.parameter_mappings
            .insert((Framework::PyTorch, Framework::TensorFlow), pytorch_to_tf);

        // TensorFlow to PyTorch mappings
        let mut tf_to_pytorch = HashMap::new();
        tf_to_pytorch.insert("learning_rate".to_string(), "lr".to_string());
        tf_to_pytorch.insert("beta_1".to_string(), "betas[0]".to_string());
        tf_to_pytorch.insert("beta_2".to_string(), "betas[1]".to_string());
        tf_to_pytorch.insert("epsilon".to_string(), "eps".to_string());
        self.parameter_mappings
            .insert((Framework::TensorFlow, Framework::PyTorch), tf_to_pytorch);

        // JAX to PyTorch mappings
        let mut jax_to_pytorch = HashMap::new();
        jax_to_pytorch.insert("learning_rate".to_string(), "lr".to_string());
        jax_to_pytorch.insert("b1".to_string(), "betas[0]".to_string());
        jax_to_pytorch.insert("b2".to_string(), "betas[1]".to_string());
        jax_to_pytorch.insert("eps".to_string(), "eps".to_string());
        self.parameter_mappings
            .insert((Framework::JAX, Framework::PyTorch), jax_to_pytorch);

        // PyTorch to JAX mappings
        let mut pytorch_to_jax = HashMap::new();
        pytorch_to_jax.insert("lr".to_string(), "learning_rate".to_string());
        pytorch_to_jax.insert("betas[0]".to_string(), "b1".to_string());
        pytorch_to_jax.insert("betas[1]".to_string(), "b2".to_string());
        pytorch_to_jax.insert("eps".to_string(), "eps".to_string());
        self.parameter_mappings
            .insert((Framework::PyTorch, Framework::JAX), pytorch_to_jax);

        // ONNX mappings (similar to TensorFlow)
        let mut onnx_to_pytorch = HashMap::new();
        onnx_to_pytorch.insert("alpha".to_string(), "lr".to_string());
        onnx_to_pytorch.insert("beta".to_string(), "betas[0]".to_string());
        onnx_to_pytorch.insert("beta2".to_string(), "betas[1]".to_string());
        onnx_to_pytorch.insert("epsilon".to_string(), "eps".to_string());
        self.parameter_mappings
            .insert((Framework::ONNX, Framework::PyTorch), onnx_to_pytorch);

        let mut pytorch_to_onnx = HashMap::new();
        pytorch_to_onnx.insert("lr".to_string(), "alpha".to_string());
        pytorch_to_onnx.insert("betas[0]".to_string(), "beta".to_string());
        pytorch_to_onnx.insert("betas[1]".to_string(), "beta2".to_string());
        pytorch_to_onnx.insert("eps".to_string(), "epsilon".to_string());
        self.parameter_mappings
            .insert((Framework::PyTorch, Framework::ONNX), pytorch_to_onnx);
    }

    /// Convert optimizer configuration to universal format
    pub fn to_universal(
        &self,
        config: &dyn ConfigSource,
        source_framework: Framework,
    ) -> Result<UniversalOptimizerConfig> {
        let (optimizer_type, learning_rate, parameters) = config.extract_config()?;

        Ok(UniversalOptimizerConfig {
            optimizer_type,
            learning_rate,
            parameters,
            source_framework,
        })
    }

    /// Convert from universal format to target framework
    pub fn from_universal(
        &self,
        config: &UniversalOptimizerConfig,
        target_framework: Framework,
    ) -> Result<Box<dyn ConfigTarget>> {
        let mapped_params = self.map_parameters(
            &config.parameters,
            config.source_framework,
            target_framework,
        )?;

        match target_framework {
            Framework::PyTorch => {
                let pytorch_config = PyTorchOptimizerConfig {
                    optimizer_type: config.optimizer_type.clone(),
                    learning_rate: config.learning_rate,
                    parameters: mapped_params,
                };
                Ok(Box::new(pytorch_config))
            },
            Framework::TensorFlow => {
                let tf_config = TensorFlowOptimizerConfig {
                    optimizer_type: config.optimizer_type.clone(),
                    learning_rate: config.learning_rate as f64,
                    parameters: mapped_params,
                    ..Default::default()
                };
                Ok(Box::new(tf_config))
            },
            Framework::JAX => {
                let jax_config = JAXOptimizerConfig {
                    optimizer_type: config.optimizer_type.clone(),
                    learning_rate: config.learning_rate,
                    parameters: mapped_params,
                };
                Ok(Box::new(jax_config))
            },
            Framework::ONNX => {
                let onnx_config = ONNXConfig {
                    optimizer_type: config.optimizer_type.clone(),
                    learning_rate: config.learning_rate,
                    parameters: mapped_params,
                };
                Ok(Box::new(onnx_config))
            },
            Framework::TrustformeRS => {
                let trustformers_config = TrustformeRSOptimizerConfig {
                    optimizer_type: config.optimizer_type.clone(),
                    learning_rate: config.learning_rate,
                    parameters: mapped_params,
                };
                Ok(Box::new(trustformers_config))
            },
        }
    }

    /// Direct conversion between frameworks
    pub fn convert(
        &self,
        config: &dyn ConfigSource,
        source_framework: Framework,
        target_framework: Framework,
    ) -> Result<Box<dyn ConfigTarget>> {
        let universal = self.to_universal(config, source_framework)?;
        self.from_universal(&universal, target_framework)
    }

    /// Map parameter names between frameworks
    fn map_parameters(
        &self,
        parameters: &HashMap<String, Value>,
        source: Framework,
        target: Framework,
    ) -> Result<HashMap<String, Value>> {
        if source == target {
            return Ok(parameters.clone());
        }

        let mapping = self.parameter_mappings.get(&(source, target)).ok_or_else(|| {
            anyhow!(
                "No parameter mapping found for {:?} to {:?}",
                source,
                target
            )
        })?;

        let mut mapped_params = HashMap::new();

        for (key, value) in parameters {
            let mapped_key = mapping.get(key).unwrap_or(key);
            mapped_params.insert(mapped_key.clone(), value.clone());
        }

        Ok(mapped_params)
    }

    /// Convert PyTorch Adam to TensorFlow Adam
    pub fn pytorch_adam_to_tensorflow(
        &self,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    ) -> Result<TensorFlowOptimizerConfig> {
        let mut parameters = HashMap::new();
        parameters.insert(
            "beta_1".to_string(),
            Value::Number(
                serde_json::Number::from_f64(betas.0 as f64)
                    .ok_or_else(|| anyhow!("Invalid beta_1"))?,
            ),
        );
        parameters.insert(
            "beta_2".to_string(),
            Value::Number(
                serde_json::Number::from_f64(betas.1 as f64)
                    .ok_or_else(|| anyhow!("Invalid beta_2"))?,
            ),
        );
        parameters.insert(
            "epsilon".to_string(),
            Value::Number(
                serde_json::Number::from_f64(eps as f64)
                    .ok_or_else(|| anyhow!("Invalid epsilon"))?,
            ),
        );
        parameters.insert(
            "weight_decay".to_string(),
            Value::Number(
                serde_json::Number::from_f64(weight_decay as f64)
                    .ok_or_else(|| anyhow!("Invalid weight_decay"))?,
            ),
        );

        Ok(TensorFlowOptimizerConfig {
            optimizer_type: "Adam".to_string(),
            learning_rate: lr as f64,
            parameters,
            ..Default::default()
        })
    }

    /// Convert TensorFlow Adam to PyTorch Adam
    pub fn tensorflow_adam_to_pytorch(
        &self,
        lr: f32,
        beta_1: f32,
        beta_2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Result<PyTorchOptimizerConfig> {
        let mut parameters = HashMap::new();
        parameters.insert(
            "betas".to_string(),
            Value::Array(vec![
                Value::Number(
                    serde_json::Number::from_f64(beta_1 as f64)
                        .ok_or_else(|| anyhow!("Invalid beta_1"))?,
                ),
                Value::Number(
                    serde_json::Number::from_f64(beta_2 as f64)
                        .ok_or_else(|| anyhow!("Invalid beta_2"))?,
                ),
            ]),
        );
        parameters.insert(
            "eps".to_string(),
            Value::Number(
                serde_json::Number::from_f64(epsilon as f64)
                    .ok_or_else(|| anyhow!("Invalid epsilon"))?,
            ),
        );
        parameters.insert(
            "weight_decay".to_string(),
            Value::Number(
                serde_json::Number::from_f64(weight_decay as f64)
                    .ok_or_else(|| anyhow!("Invalid weight_decay"))?,
            ),
        );

        Ok(PyTorchOptimizerConfig {
            optimizer_type: "Adam".to_string(),
            learning_rate: lr,
            parameters,
        })
    }

    /// Convert JAX Adam to PyTorch Adam
    pub fn jax_adam_to_pytorch(
        &self,
        learning_rate: f32,
        b1: f32,
        b2: f32,
        eps: f32,
    ) -> Result<PyTorchOptimizerConfig> {
        let mut parameters = HashMap::new();
        parameters.insert(
            "betas".to_string(),
            Value::Array(vec![
                Value::Number(
                    serde_json::Number::from_f64(b1 as f64).ok_or_else(|| anyhow!("Invalid b1"))?,
                ),
                Value::Number(
                    serde_json::Number::from_f64(b2 as f64).ok_or_else(|| anyhow!("Invalid b2"))?,
                ),
            ]),
        );
        parameters.insert(
            "eps".to_string(),
            Value::Number(
                serde_json::Number::from_f64(eps as f64)
                    .ok_or_else(|| anyhow!("Invalid epsilon"))?,
            ),
        );

        Ok(PyTorchOptimizerConfig {
            optimizer_type: "Adam".to_string(),
            learning_rate,
            parameters,
        })
    }

    /// Convert ONNX optimizer to PyTorch
    pub fn onnx_to_pytorch(&self, onnx_config: &ONNXConfig) -> Result<PyTorchOptimizerConfig> {
        let mapped_params =
            self.map_parameters(&onnx_config.parameters, Framework::ONNX, Framework::PyTorch)?;

        Ok(PyTorchOptimizerConfig {
            optimizer_type: onnx_config.optimizer_type.clone(),
            learning_rate: onnx_config.learning_rate,
            parameters: mapped_params,
        })
    }

    /// Batch convert multiple optimizers
    pub fn batch_convert(
        &self,
        configs: Vec<(&dyn ConfigSource, Framework)>,
        target_framework: Framework,
    ) -> Result<Vec<Box<dyn ConfigTarget>>> {
        let mut results = Vec::new();

        for (config, source_framework) in configs {
            let converted = self.convert(config, source_framework, target_framework)?;
            results.push(converted);
        }

        Ok(results)
    }

    /// Generate conversion report
    pub fn generate_conversion_report(&self, source: Framework, target: Framework) -> String {
        let mapping = self.parameter_mappings.get(&(source, target));

        match mapping {
            Some(map) => {
                let mut report = format!("Conversion mapping from {:?} to {:?}:\n", source, target);
                for (source_param, target_param) in map {
                    report.push_str(&format!("  {} -> {}\n", source_param, target_param));
                }
                report
            },
            None => format!(
                "No conversion mapping available from {:?} to {:?}",
                source, target
            ),
        }
    }
}

impl Default for CrossFrameworkConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for extracting configuration from different sources
pub trait ConfigSource {
    fn extract_config(&self) -> Result<(String, f32, HashMap<String, Value>)>;
}

/// Trait for creating configuration targets
pub trait ConfigTarget {}

/// PyTorch optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchOptimizerConfig {
    pub optimizer_type: String,
    pub learning_rate: f32,
    pub parameters: HashMap<String, Value>,
}

impl ConfigTarget for PyTorchOptimizerConfig {}

impl ConfigSource for PyTorchOptimizerConfig {
    fn extract_config(&self) -> Result<(String, f32, HashMap<String, Value>)> {
        Ok((
            self.optimizer_type.clone(),
            self.learning_rate,
            self.parameters.clone(),
        ))
    }
}

/// JAX optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JAXOptimizerConfig {
    pub optimizer_type: String,
    pub learning_rate: f32,
    pub parameters: HashMap<String, Value>,
}

impl ConfigTarget for JAXOptimizerConfig {}

impl ConfigSource for JAXOptimizerConfig {
    fn extract_config(&self) -> Result<(String, f32, HashMap<String, Value>)> {
        Ok((
            self.optimizer_type.clone(),
            self.learning_rate,
            self.parameters.clone(),
        ))
    }
}

/// TrustformeRS optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustformeRSOptimizerConfig {
    pub optimizer_type: String,
    pub learning_rate: f32,
    pub parameters: HashMap<String, Value>,
}

impl ConfigTarget for TrustformeRSOptimizerConfig {}

impl ConfigSource for TrustformeRSOptimizerConfig {
    fn extract_config(&self) -> Result<(String, f32, HashMap<String, Value>)> {
        Ok((
            self.optimizer_type.clone(),
            self.learning_rate,
            self.parameters.clone(),
        ))
    }
}

// Implement ConfigSource for existing types
impl ConfigSource for crate::tensorflow_compat::TensorFlowOptimizerConfig {
    fn extract_config(&self) -> Result<(String, f32, HashMap<String, Value>)> {
        Ok((
            self.optimizer_type.clone(),
            self.learning_rate as f32,
            self.parameters.clone(),
        ))
    }
}

impl ConfigTarget for crate::tensorflow_compat::TensorFlowOptimizerConfig {}

impl ConfigSource for crate::onnx_export::OptimizerConfig {
    fn extract_config(&self) -> Result<(String, f32, HashMap<String, Value>)> {
        Ok((
            self.optimizer_type.clone(),
            self.learning_rate,
            self.parameters.clone(),
        ))
    }
}

impl ConfigTarget for crate::onnx_export::OptimizerConfig {}

/// Conversion utilities
pub mod utils {
    use super::*;

    /// Create a conversion matrix showing all supported conversions
    pub fn create_conversion_matrix() -> HashMap<(Framework, Framework), bool> {
        let frameworks = [
            Framework::PyTorch,
            Framework::TensorFlow,
            Framework::JAX,
            Framework::ONNX,
            Framework::TrustformeRS,
        ];
        let mut matrix = HashMap::new();

        for &source in &frameworks {
            for &target in &frameworks {
                // All conversions are supported
                matrix.insert((source, target), true);
            }
        }

        matrix
    }

    /// Get list of supported frameworks
    pub fn get_supported_frameworks() -> Vec<Framework> {
        vec![
            Framework::PyTorch,
            Framework::TensorFlow,
            Framework::JAX,
            Framework::ONNX,
            Framework::TrustformeRS,
        ]
    }

    /// Validate parameter values during conversion
    pub fn validate_parameters(
        optimizer_type: &str,
        parameters: &HashMap<String, Value>,
    ) -> Result<()> {
        match optimizer_type {
            "Adam" | "AdamW" => {
                // Validate beta values
                if let Some(Value::Array(betas)) = parameters.get("betas") {
                    if betas.len() != 2 {
                        return Err(anyhow!("Adam betas must be a 2-element array"));
                    }
                }

                // Validate learning rate is positive
                if let Some(Value::Number(lr)) = parameters.get("lr") {
                    if lr.as_f64().unwrap_or(0.0) <= 0.0 {
                        return Err(anyhow!("Learning rate must be positive"));
                    }
                }
            },
            "SGD" => {
                // Validate momentum
                if let Some(Value::Number(momentum)) = parameters.get("momentum") {
                    let momentum_val = momentum.as_f64().unwrap_or(0.0);
                    if !(0.0..1.0).contains(&momentum_val) {
                        return Err(anyhow!("Momentum must be in [0, 1)"));
                    }
                }
            },
            _ => {
                // Generic validation for unknown optimizers
                for (key, value) in parameters {
                    if key.contains("learning_rate") || key.contains("lr") {
                        if let Value::Number(lr) = value {
                            if lr.as_f64().unwrap_or(0.0) <= 0.0 {
                                return Err(anyhow!("Learning rate must be positive"));
                            }
                        }
                    }
                }
            },
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_to_tensorflow_conversion() {
        let converter = CrossFrameworkConverter::new();
        let tf_config =
            converter.pytorch_adam_to_tensorflow(0.001, (0.9, 0.999), 1e-8, 0.01).unwrap();

        assert_eq!(tf_config.optimizer_type, "Adam");
        assert!((tf_config.learning_rate - 0.001).abs() < 1e-9);
        assert!(tf_config.parameters.contains_key("beta_1"));
        assert!(tf_config.parameters.contains_key("beta_2"));
    }

    #[test]
    fn test_tensorflow_to_pytorch_conversion() {
        let converter = CrossFrameworkConverter::new();
        let pytorch_config =
            converter.tensorflow_adam_to_pytorch(0.001, 0.9, 0.999, 1e-8, 0.01).unwrap();

        assert_eq!(pytorch_config.optimizer_type, "Adam");
        assert_eq!(pytorch_config.learning_rate, 0.001);
        assert!(pytorch_config.parameters.contains_key("betas"));
        assert!(pytorch_config.parameters.contains_key("eps"));
    }

    #[test]
    fn test_jax_to_pytorch_conversion() {
        let converter = CrossFrameworkConverter::new();
        let pytorch_config = converter.jax_adam_to_pytorch(0.001, 0.9, 0.999, 1e-8).unwrap();

        assert_eq!(pytorch_config.optimizer_type, "Adam");
        assert_eq!(pytorch_config.learning_rate, 0.001);
        assert!(pytorch_config.parameters.contains_key("betas"));
    }

    #[test]
    fn test_parameter_mapping() {
        let converter = CrossFrameworkConverter::new();
        let mut params = HashMap::new();
        params.insert(
            "lr".to_string(),
            Value::Number(
                serde_json::Number::from_f64(0.001).expect("Invalid constant"),
            ),
        );

        let mapped = converter
            .map_parameters(&params, Framework::PyTorch, Framework::TensorFlow)
            .unwrap();
        assert!(mapped.contains_key("learning_rate"));
    }

    #[test]
    fn test_universal_conversion() {
        let converter = CrossFrameworkConverter::new();

        let pytorch_config = PyTorchOptimizerConfig {
            optimizer_type: "Adam".to_string(),
            learning_rate: 0.001,
            parameters: HashMap::new(),
        };

        let universal = converter.to_universal(&pytorch_config, Framework::PyTorch).unwrap();
        assert_eq!(universal.optimizer_type, "Adam");
        assert_eq!(universal.source_framework, Framework::PyTorch);

        let _tf_config = converter.from_universal(&universal, Framework::TensorFlow).unwrap();
    }

    #[test]
    fn test_conversion_matrix() {
        let matrix = utils::create_conversion_matrix();
        assert!(matrix.get(&(Framework::PyTorch, Framework::TensorFlow)).unwrap());
        assert!(matrix.get(&(Framework::JAX, Framework::ONNX)).unwrap());
    }

    #[test]
    fn test_parameter_validation() {
        let mut params = HashMap::new();
        params.insert(
            "lr".to_string(),
            Value::Number(
                serde_json::Number::from_f64(0.001).expect("Invalid constant"),
            ),
        );

        utils::validate_parameters("Adam", &params).unwrap();

        // Test invalid learning rate
        params.insert(
            "lr".to_string(),
            Value::Number(
                serde_json::Number::from_f64(-0.001).expect("Invalid constant"),
            ),
        );
        assert!(utils::validate_parameters("Adam", &params).is_err());
    }

    #[test]
    fn test_conversion_report() {
        let converter = CrossFrameworkConverter::new();
        let report =
            converter.generate_conversion_report(Framework::PyTorch, Framework::TensorFlow);

        assert!(report.contains("PyTorch"));
        assert!(report.contains("TensorFlow"));
        assert!(report.contains("->"));
    }

    #[test]
    fn test_supported_frameworks() {
        let frameworks = utils::get_supported_frameworks();
        assert!(frameworks.contains(&Framework::PyTorch));
        assert!(frameworks.contains(&Framework::TensorFlow));
        assert!(frameworks.contains(&Framework::JAX));
        assert!(frameworks.contains(&Framework::ONNX));
        assert!(frameworks.contains(&Framework::TrustformeRS));
    }
}
