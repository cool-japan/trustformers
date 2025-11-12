//! Main checkpoint converter implementation

use anyhow::{anyhow, Result};
use log;
use scirs2_core::parallel_ops::*; // SciRS2 Policy compliant (replaces rayon)
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::checkpoint::{
    formats::{
        Checkpoint, CheckpointFormat, JaxCheckpoint, PyTorchCheckpoint, TensorFlowCheckpoint,
        TrustformersCheckpoint, WeightTensor,
    },
    mapping::{ConvFormat, ModelType, WeightMapping, WeightTransform},
};

/// Configuration for checkpoint conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionConfig {
    /// Target framework format
    pub target_format: CheckpointFormat,
    /// Model type for specialized mappings
    pub model_type: ModelType,
    /// Whether to validate converted weights
    pub validate: bool,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Custom weight name mappings
    pub custom_mappings: HashMap<String, String>,
    /// Weights to exclude from conversion
    pub exclude_weights: Vec<String>,
    /// Whether to preserve metadata
    pub preserve_metadata: bool,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            target_format: CheckpointFormat::Trustformers,
            model_type: ModelType::Generic,
            validate: true,
            parallel: true,
            custom_mappings: HashMap::new(),
            exclude_weights: Vec::new(),
            preserve_metadata: true,
        }
    }
}

/// Result of checkpoint conversion
#[derive(Debug, Serialize, Deserialize)]
pub struct ConversionResult {
    pub source_format: CheckpointFormat,
    pub target_format: CheckpointFormat,
    pub weights_converted: usize,
    pub weights_skipped: Vec<String>,
    pub warnings: Vec<String>,
    pub conversion_time_ms: u64,
}

/// Main checkpoint converter
pub struct CheckpointConverter {
    config: ConversionConfig,
    weight_mapping: WeightMapping,
}

impl Default for CheckpointConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointConverter {
    pub fn new() -> Self {
        Self::with_config(ConversionConfig::default())
    }

    pub fn with_config(config: ConversionConfig) -> Self {
        let weight_mapping = WeightMapping::new(config.model_type);
        Self {
            config,
            weight_mapping,
        }
    }

    /// Convert checkpoint from source to target format
    pub async fn convert(
        &self,
        source_path: &Path,
        target_path: &Path,
        target_format: CheckpointFormat,
    ) -> Result<ConversionResult> {
        let start_time = std::time::Instant::now();

        // Detect source format
        let source_format = CheckpointFormat::from_path(source_path).ok_or_else(|| {
            anyhow!(
                "Unable to detect source format from path: {:?}",
                source_path
            )
        })?;

        // Load source checkpoint
        let source_checkpoint = self.load_checkpoint(source_path, source_format)?;

        // Create target checkpoint
        let mut target_checkpoint = self.create_checkpoint(target_format)?;

        // Convert weights
        let result = self.convert_weights(
            &*source_checkpoint,
            &mut *target_checkpoint,
            source_format,
            target_format,
        )?;

        // Save target checkpoint
        target_checkpoint.save(target_path)?;

        let conversion_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(ConversionResult {
            source_format,
            target_format,
            weights_converted: result.0,
            weights_skipped: result.1,
            warnings: result.2,
            conversion_time_ms,
        })
    }

    /// Load checkpoint based on format
    fn load_checkpoint(
        &self,
        path: &Path,
        format: CheckpointFormat,
    ) -> Result<Box<dyn Checkpoint>> {
        match format {
            CheckpointFormat::PyTorch => Ok(Box::new(PyTorchCheckpoint::load(path)?)),
            CheckpointFormat::TensorFlow => Ok(Box::new(TensorFlowCheckpoint::load(path)?)),
            CheckpointFormat::JAX => Ok(Box::new(JaxCheckpoint::load(path)?)),
            CheckpointFormat::Trustformers => Ok(Box::new(TrustformersCheckpoint::load(path)?)),
            _ => Err(anyhow!("Unsupported source format: {:?}", format)),
        }
    }

    /// Create new checkpoint based on format
    fn create_checkpoint(&self, format: CheckpointFormat) -> Result<Box<dyn Checkpoint>> {
        match format {
            CheckpointFormat::PyTorch => Ok(Box::new(PyTorchCheckpoint::new())),
            CheckpointFormat::TensorFlow => Ok(Box::new(TensorFlowCheckpoint::new())),
            CheckpointFormat::JAX => Ok(Box::new(JaxCheckpoint::new())),
            CheckpointFormat::Trustformers => Ok(Box::new(TrustformersCheckpoint::new())),
            _ => Err(anyhow!("Unsupported target format: {:?}", format)),
        }
    }

    /// Convert weights between checkpoints
    fn convert_weights(
        &self,
        source: &dyn Checkpoint,
        target: &mut dyn Checkpoint,
        source_format: CheckpointFormat,
        target_format: CheckpointFormat,
    ) -> Result<(usize, Vec<String>, Vec<String>)> {
        let mut weights_converted = 0;
        let mut weights_skipped = Vec::new();
        let mut warnings = Vec::new();

        let weight_names = source.weight_names();

        // Process weights in parallel if enabled
        let conversions: Vec<_> = if self.config.parallel {
            weight_names
                .par_iter()
                .filter_map(|name| {
                    self.convert_single_weight(name, source, source_format, target_format).ok()
                })
                .collect()
        } else {
            weight_names
                .iter()
                .filter_map(|name| {
                    self.convert_single_weight(name, source, source_format, target_format).ok()
                })
                .collect()
        };

        // Apply conversions
        for (target_name, weight, warning) in conversions {
            target.set_weight(&target_name, weight)?;
            weights_converted += 1;
            if let Some(w) = warning {
                warnings.push(w);
            }
        }

        // Track skipped weights
        for name in weight_names {
            if self.config.exclude_weights.contains(&name) {
                weights_skipped.push(name);
            }
        }

        Ok((weights_converted, weights_skipped, warnings))
    }

    /// Convert a single weight
    fn convert_single_weight(
        &self,
        name: &str,
        source: &dyn Checkpoint,
        source_format: CheckpointFormat,
        target_format: CheckpointFormat,
    ) -> Result<(String, WeightTensor, Option<String>)> {
        // Skip excluded weights
        if self.config.exclude_weights.contains(&name.to_string()) {
            return Err(anyhow!("Weight excluded"));
        }

        // Get source weight
        let mut weight = source.get_weight(name)?;

        // Apply custom mapping if exists
        let target_name = if let Some(custom_name) = self.config.custom_mappings.get(name) {
            custom_name.clone()
        } else {
            // Use automatic mapping
            let (mapped_name, transform) =
                self.map_weight_name(name, source_format, target_format)?;

            // Apply transformation if needed
            if let Some(t) = transform {
                self.apply_transform(&mut weight, &t)?;
            }

            mapped_name
        };

        let warning = None; // Could add validation warnings here

        Ok((target_name, weight, warning))
    }

    /// Map weight name between formats
    fn map_weight_name(
        &self,
        name: &str,
        source_format: CheckpointFormat,
        target_format: CheckpointFormat,
    ) -> Result<(String, Option<WeightTransform>)> {
        match (source_format, target_format) {
            (CheckpointFormat::PyTorch, CheckpointFormat::TensorFlow) => {
                self.weight_mapping.pytorch_to_tensorflow(name)
            },
            (CheckpointFormat::TensorFlow, CheckpointFormat::PyTorch) => {
                self.weight_mapping.tensorflow_to_pytorch(name)
            },
            (CheckpointFormat::PyTorch, CheckpointFormat::JAX) => {
                self.weight_mapping.pytorch_to_jax(name)
            },
            (CheckpointFormat::JAX, CheckpointFormat::PyTorch) => {
                self.weight_mapping.jax_to_pytorch(name)
            },
            _ => {
                // Default: keep the same name
                Ok((name.to_string(), None))
            },
        }
    }

    /// Apply transformation to weight tensor
    fn apply_transform(
        &self,
        weight: &mut WeightTensor,
        transform: &WeightTransform,
    ) -> Result<()> {
        match transform {
            WeightTransform::Identity => Ok(()),
            WeightTransform::Transpose(dims) => {
                weight.transpose(dims)?;
                Ok(())
            },
            WeightTransform::Reshape(new_shape) => {
                let shape: Vec<usize> = new_shape
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| {
                        if s == -1 {
                            // Infer dimension
                            let total: usize = weight.shape.iter().product();
                            let other: usize = new_shape
                                .iter()
                                .enumerate()
                                .filter(|(j, &v)| *j != i && v != -1)
                                .map(|(_, &v)| v as usize)
                                .product();
                            total / other
                        } else {
                            s as usize
                        }
                    })
                    .collect();
                weight.reshape(shape)?;
                Ok(())
            },
            WeightTransform::Split { axis, sizes } => {
                // Note: Split transform requires special handling during conversion
                // as it produces multiple output tensors from a single input tensor
                if *axis >= weight.shape.len() {
                    return Err(anyhow!(
                        "Split axis {} out of bounds for shape {:?}",
                        axis,
                        weight.shape
                    ));
                }

                let axis_size = weight.shape[*axis];
                let total_size: usize = sizes.iter().sum();

                if total_size != axis_size {
                    return Err(anyhow!(
                        "Split sizes {:?} don't match axis size {} for axis {}",
                        sizes,
                        axis_size,
                        axis
                    ));
                }

                // For now, we'll store metadata about the split in the tensor
                // The actual splitting should be handled by the conversion pipeline
                log::warn!(
                    "Split transform applied - requires special handling in conversion pipeline"
                );
                Ok(())
            },
            WeightTransform::Merge { axis } => {
                // Note: Merge transform requires special handling during conversion
                // as it requires multiple input tensors to produce a single output tensor
                if *axis >= weight.shape.len() {
                    return Err(anyhow!(
                        "Merge axis {} out of bounds for shape {:?}",
                        axis,
                        weight.shape
                    ));
                }

                // For now, we'll store metadata about the merge in the tensor
                // The actual merging should be handled by the conversion pipeline
                log::warn!(
                    "Merge transform applied - requires special handling in conversion pipeline"
                );
                Ok(())
            },
            WeightTransform::ConvFormat { from, to } => {
                // Convert convolution weight formats between NCHW and NHWC
                if weight.shape.len() != 4 {
                    return Err(anyhow!(
                        "ConvFormat transform requires 4D tensor, got shape {:?}",
                        weight.shape
                    ));
                }

                match (from, to) {
                    (ConvFormat::NCHW, ConvFormat::NHWC) => {
                        // NCHW [N, C, H, W] -> NHWC [N, H, W, C]
                        // Transpose dimensions [0, 1, 2, 3] -> [0, 2, 3, 1]
                        weight.transpose(&[0, 2, 3, 1])?;
                        Ok(())
                    },
                    (ConvFormat::NHWC, ConvFormat::NCHW) => {
                        // NHWC [N, H, W, C] -> NCHW [N, C, H, W]
                        // Transpose dimensions [0, 1, 2, 3] -> [0, 3, 1, 2]
                        weight.transpose(&[0, 3, 1, 2])?;
                        Ok(())
                    },
                    (from_fmt, to_fmt) if from_fmt == to_fmt => {
                        // No transformation needed if formats are the same
                        Ok(())
                    },
                    _ => Err(anyhow!(
                        "Unsupported ConvFormat conversion from {:?} to {:?}",
                        from,
                        to
                    )),
                }
            },
        }
    }
}

/// Builder for checkpoint converter
pub struct CheckpointConverterBuilder {
    config: ConversionConfig,
}

impl Default for CheckpointConverterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointConverterBuilder {
    pub fn new() -> Self {
        Self {
            config: ConversionConfig::default(),
        }
    }

    pub fn model_type(mut self, model_type: ModelType) -> Self {
        self.config.model_type = model_type;
        self
    }

    pub fn validate(mut self, validate: bool) -> Self {
        self.config.validate = validate;
        self
    }

    pub fn parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }

    pub fn add_custom_mapping(mut self, source: &str, target: &str) -> Self {
        self.config.custom_mappings.insert(source.to_string(), target.to_string());
        self
    }

    pub fn exclude_weight(mut self, name: &str) -> Self {
        self.config.exclude_weights.push(name.to_string());
        self
    }

    pub fn build(self) -> CheckpointConverter {
        CheckpointConverter::with_config(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_converter_creation() {
        let converter = CheckpointConverterBuilder::new()
            .model_type(ModelType::BERT)
            .validate(true)
            .parallel(true)
            .build();

        assert_eq!(converter.config.model_type, ModelType::BERT);
        assert!(converter.config.validate);
        assert!(converter.config.parallel);
    }

    #[test]
    fn test_weight_transform() {
        let converter = CheckpointConverter::new();
        let mut weight = WeightTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // Test reshape
        let transform = WeightTransform::Reshape(vec![4, -1]);
        converter.apply_transform(&mut weight, &transform).unwrap();
        assert_eq!(weight.shape, vec![4, 1]);
    }
}
