//! Embedding layer implementation
//!
//! This module provides token embedding functionality with device support.

use crate::device::Device;
use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use crate::traits::Layer;
use scirs2_core::ndarray::{Array2, Axis};

/// Token Embedding Layer
///
/// Maps discrete token IDs to continuous vector representations. This is typically
/// the first layer in transformer models, converting input tokens to dense embeddings.
///
/// # Parameters
///
/// - `weight`: Embedding lookup table of shape `[num_embeddings, embedding_dim]`
/// - `num_embeddings`: Vocabulary size
/// - `embedding_dim`: Dimension of each embedding vector
///
/// # Example
///
/// ```no_run
/// use trustformers_core::layers::Embedding;
/// use trustformers_core::tensor::Tensor;
/// use trustformers_core::traits::Layer;
/// use trustformers_core::device::Device;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create embedding layer: vocab_size=30000, embedding_dim=768
/// let embedding = Embedding::new_with_device(30000, 768, None, Device::CPU)?;
///
/// // Map token IDs to embeddings
/// let token_ids = vec![5, 142, 9876];
/// let embeddings = embedding.forward(token_ids)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Embedding {
    weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
    device: Device,
}

impl Embedding {
    /// Creates a new Embedding layer on CPU
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
    ) -> Result<Self> {
        Self::new_with_device(num_embeddings, embedding_dim, padding_idx, Device::CPU)
    }

    /// Creates a new Embedding layer on specified device
    pub fn new_with_device(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        device: Device,
    ) -> Result<Self> {
        let mut weight = Tensor::randn(&[num_embeddings, embedding_dim])?;

        // Zero out padding embedding if specified
        if let Some(padding_idx) = padding_idx {
            if padding_idx < num_embeddings {
                weight = weight.zero_padding_embedding(padding_idx)?;
            }
        }

        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            device,
        })
    }

    /// Sets the weight tensor (e.g., for loading pretrained embeddings)
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    /// Forward pass with explicit token IDs
    pub fn forward_ids(&self, input_ids: &[u32]) -> Result<Tensor> {
        self.forward(input_ids.to_vec())
    }

    /// Returns the device this layer uses for computations
    pub fn device(&self) -> Device {
        self.device
    }

    /// Moves this layer to a different device
    pub fn to_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Returns the number of parameters in this embedding layer
    pub fn parameter_count(&self) -> usize {
        self.num_embeddings * self.embedding_dim
    }

    /// Upload embedding weights to GPU for GPU-resident inference
    #[cfg(feature = "metal")]
    pub fn weights_to_gpu(&mut self, device: &Device) -> Result<()> {
        if !matches!(device, Device::Metal(_)) {
            return Ok(());
        }
        self.device = *device;
        self.weight = self.weight.to_device_enum(device)?;
        Ok(())
    }
}

impl Layer for Embedding {
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Handle Metal weights: convert to CPU for lookup, return Metal tensor
        #[cfg(feature = "metal")]
        if let Tensor::Metal(_) = &self.weight {
            // Convert Metal weights to CPU for lookup
            let cpu_weight = self.weight.to_device_enum(&Device::CPU)?;

            if let Tensor::F32(weight_arr) = cpu_weight {
                let batch_size = input.len();
                let mut output = Array2::<f32>::zeros((batch_size, self.embedding_dim));

                for (i, &idx) in input.iter().enumerate() {
                    if idx as usize >= self.num_embeddings {
                        return Err(TrustformersError::tensor_op_error(
                            &format!(
                                "Index {} out of range for embedding table of size {}",
                                idx, self.num_embeddings
                            ),
                            "Embedding::forward",
                        ));
                    }
                    let embedding = weight_arr.index_axis(Axis(0), idx as usize);
                    output.row_mut(i).assign(&embedding);
                }

                // Return as Metal tensor if device is Metal
                let result = Tensor::F32(output.into_dyn());
                if matches!(self.device, Device::Metal(_)) {
                    let metal_result = result.to_device_enum(&self.device)?;
                    return Ok(metal_result);
                }
                return Ok(result);
            }
        }

        // Standard F32 path
        match &self.weight {
            Tensor::F32(weight_arr) => {
                let batch_size = input.len();
                let mut output = Array2::<f32>::zeros((batch_size, self.embedding_dim));

                for (i, &idx) in input.iter().enumerate() {
                    if idx as usize >= self.num_embeddings {
                        return Err(TrustformersError::tensor_op_error(
                            &format!(
                                "Index {} out of range for embedding table of size {}",
                                idx, self.num_embeddings
                            ),
                            "Embedding::forward",
                        ));
                    }
                    let embedding = weight_arr.index_axis(Axis(0), idx as usize);
                    output.row_mut(i).assign(&embedding);
                }

                Ok(Tensor::F32(output.into_dyn()))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for embedding",
                "Embedding::forward",
            )),
        }
    }
}
