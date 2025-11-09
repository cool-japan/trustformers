use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use crate::traits::Layer;
use ndarray::Axis;

#[derive(Debug, Clone)]
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: f32) -> Result<Self> {
        let weight = Tensor::ones(&normalized_shape)?;
        let bias = Tensor::zeros(&normalized_shape)?;

        Ok(Self {
            normalized_shape,
            weight,
            bias,
            eps,
        })
    }

    pub fn new_simple(normalized_shape: usize, eps: f32) -> Self {
        Self::new(vec![normalized_shape], eps).unwrap()
    }

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    pub fn set_bias(&mut self, bias: Tensor) -> Result<()> {
        self.bias = bias;
        Ok(())
    }

    /// Returns the total number of learnable parameters in this layer.
    ///
    /// # Returns
    ///
    /// The total parameter count including weight and bias tensors.
    pub fn parameter_count(&self) -> usize {
        let weight_count = self.weight.len();
        let bias_count = self.bias.len();
        weight_count + bias_count
    }
}

impl Layer for LayerNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                let ndim = arr.ndim();
                let norm_ndim = self.normalized_shape.len();

                // For LayerNorm, we normalize over the last norm_ndim dimensions
                // Calculate mean and variance
                let axes: Vec<usize> = ((ndim - norm_ndim)..ndim).collect();

                // Compute mean across the normalized dimensions
                let mut mean = arr.clone();
                for &axis in axes.iter().rev() {
                    mean = mean.mean_axis(Axis(axis)).unwrap().insert_axis(Axis(axis));
                }

                // Compute variance
                let diff = arr - &mean;
                let mut var = (&diff * &diff).to_owned();
                for &axis in axes.iter().rev() {
                    var = var.mean_axis(Axis(axis)).unwrap().insert_axis(Axis(axis));
                }

                // Normalize
                let normalized = &diff / (var + self.eps).mapv(f32::sqrt);

                match (&self.weight, &self.bias) {
                    (Tensor::F32(w), Tensor::F32(b)) => {
                        // Handle broadcasting for weight and bias
                        let mut broadcast_shape = vec![1; ndim];
                        for (i, &dim) in self.normalized_shape.iter().enumerate() {
                            broadcast_shape[ndim - norm_ndim + i] = dim;
                        }

                        use ndarray::IxDyn;
                        let w_broadcast = w
                            .view()
                            .into_shape_with_order(IxDyn(&broadcast_shape))
                            .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to broadcast weight: {}",
                                e
                            ))
                        })?;
                        let b_broadcast = b
                            .view()
                            .into_shape_with_order(IxDyn(&broadcast_shape))
                            .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to broadcast bias: {}",
                                e
                            ))
                        })?;

                        let output = &normalized * &w_broadcast + &b_broadcast;
                        Ok(Tensor::F32(output))
                    },
                    _ => Err(TrustformersError::tensor_op_error(
                        "LayerNorm weight/bias type mismatch",
                        "LayerNorm::forward",
                    )),
                }
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for LayerNorm",
                "LayerNorm::forward",
            )),
        }
    }
}

/// Root Mean Square Layer Normalization
///
/// RMSNorm normalizes the input using only the root mean square (RMS) of the input,
/// without centering by subtracting the mean. This is computationally more efficient
/// than standard LayerNorm and is used in many modern architectures like LLaMA.
#[derive(Debug, Clone)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Result<Self> {
        let weight = Tensor::ones(&[hidden_size])?;
        Ok(Self { weight, eps })
    }

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    /// Returns the total number of learnable parameters in this layer.
    ///
    /// # Returns
    ///
    /// The total parameter count for the weight tensor.
    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

impl Layer for RMSNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                let ndim = arr.ndim();
                let last_dim = ndim - 1;

                // Compute RMS: sqrt(mean(x^2))
                let squares = arr.mapv(|x| x * x);
                let mean_squares =
                    squares.mean_axis(Axis(last_dim)).unwrap().insert_axis(Axis(last_dim));
                let rms = mean_squares.mapv(|x| (x + self.eps).sqrt());

                // Normalize: x / rms
                let normalized = arr / &rms;

                // Apply weight
                match &self.weight {
                    Tensor::F32(w) => {
                        let mut broadcast_shape = vec![1; ndim];
                        broadcast_shape[last_dim] = w.len();

                        use ndarray::IxDyn;
                        let w_broadcast = w
                            .view()
                            .into_shape_with_order(IxDyn(&broadcast_shape))
                            .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to broadcast weight: {}",
                                e
                            ))
                        })?;

                        let output = &normalized * &w_broadcast;
                        Ok(Tensor::F32(output))
                    },
                    _ => Err(TrustformersError::tensor_op_error(
                        "RMSNorm weight type mismatch",
                        "RMSNorm::forward",
                    )),
                }
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for RMSNorm",
                "RMSNorm::forward",
            )),
        }
    }
}
