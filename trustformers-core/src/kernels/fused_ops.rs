// Convenience wrappers that chain common operation sequences (activation,
// bias + activation, linear + activation, ...) behind one call.
//
// HONESTY NOTE: despite the "Fused*" names, none of these run a single
// kernel or reduce memory traffic versus calling the underlying `Tensor`
// ops directly - each step below (matmul, broadcast_add, softmax, the
// activation itself, ...) still materializes its own full intermediate
// tensor, exactly as if the caller had chained those `Tensor` methods by
// hand. "Fused" here means "one function call for a common op sequence",
// not "single-pass/fused-kernel execution". A real fusion would need a
// combined-expression kernel (e.g. one `mapv` pass over the whole
// expression, or a Metal `matmul_bias_gelu`-style kernel like
// `gpu_ops::metal::metalbackend_matmul_gelu_f32_group`) that never
// materializes the intermediate at all.
use crate::tensor::Tensor;
use anyhow::Result;

/// GELU activation (erf-exact or tanh-approximate). See the module-level
/// honesty note: this is a plain, single-op convenience wrapper, not a
/// fused/single-pass kernel.
pub struct FusedGELU {
    approximate: bool,
}

impl FusedGELU {
    pub fn new(approximate: bool) -> Self {
        Self { approximate }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.approximate {
            self.gelu_tanh_approximation(input)
        } else {
            self.gelu_erf(input)
        }
    }

    fn gelu_erf(&self, input: &Tensor) -> Result<Tensor> {
        // Accurate GELU using erf approximation: 0.5 * x * (1 + erf(x / sqrt(2)))
        let input_data = input.data()?;
        let sqrt_2_inv = std::f32::consts::FRAC_1_SQRT_2;

        let output_data: Vec<f32> = input_data
            .iter()
            .map(|&x| {
                let scaled = x * sqrt_2_inv;
                // More accurate erf approximation using rational function
                let erf_val = Self::erf_approximation(scaled);
                0.5 * x * (1.0 + erf_val)
            })
            .collect();

        Ok(Tensor::from_vec(output_data, &input.shape())?)
    }

    /// Accurate erf approximation using Abramowitz and Stegun formula
    fn erf_approximation(x: f32) -> f32 {
        // Constants for the approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();

        // A&S formula 7.1.26
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    fn gelu_tanh_approximation(&self, input: &Tensor) -> Result<Tensor> {
        // GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        let input_data = input.data()?;
        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
        let coefficient = 0.044715f32;

        let output_data: Vec<f32> = input_data
            .iter()
            .map(|&x| {
                let x_cubed = x * x * x;
                let inner = x + coefficient * x_cubed;
                let scaled = inner * sqrt_2_over_pi;
                let tanh_vals = scaled.tanh();
                let one_plus_tanh = 1.0 + tanh_vals;
                0.5 * x * one_plus_tanh
            })
            .collect();

        Ok(Tensor::from_vec(output_data, &input.shape())?)
    }
}

/// Bias-add followed by an activation function, as one call. See the
/// module-level honesty note: `broadcast_add` and the activation each
/// still materialize their own output tensor - no memory bandwidth is
/// saved versus calling them separately.
pub struct FusedBiasActivation {
    activation: ActivationType,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    GELU,
    SiLU,
    Tanh,
    Sigmoid,
}

impl FusedBiasActivation {
    pub fn new(activation: ActivationType) -> Self {
        Self { activation }
    }

    pub fn forward(&self, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
        // Add bias first
        let biased = input.broadcast_add(bias)?;

        // Apply activation
        match self.activation {
            ActivationType::ReLU => Ok(biased.relu()?),
            ActivationType::GELU => {
                let gelu = FusedGELU::new(true);
                gelu.forward(&biased)
            },
            ActivationType::SiLU => {
                // SiLU(x) = x * sigmoid(x)
                let sigmoid_vals = biased.sigmoid()?;
                Ok(biased.mul(&sigmoid_vals)?)
            },
            ActivationType::Tanh => Ok(biased.tanh()?),
            ActivationType::Sigmoid => Ok(biased.sigmoid()?),
        }
    }
}

/// Softmax followed by dropout, as one call. See the module-level honesty
/// note: `softmax` and `dropout` each still materialize their own output
/// tensor - no memory accesses are saved versus calling them separately.
pub struct FusedAttentionDropout {
    dropout_prob: f32,
    training: bool,
}

impl FusedAttentionDropout {
    pub fn new(dropout_prob: f32, training: bool) -> Self {
        Self {
            dropout_prob,
            training,
        }
    }

    pub fn forward(&self, attention_scores: &Tensor) -> Result<Tensor> {
        let softmax_scores = attention_scores.softmax(-1)?;

        if self.training && self.dropout_prob > 0.0 {
            self.apply_dropout(&softmax_scores)
        } else {
            Ok(softmax_scores)
        }
    }

    fn apply_dropout(&self, input: &Tensor) -> Result<Tensor> {
        // Use simplified dropout implementation with the existing Tensor API
        Ok(input.dropout(self.dropout_prob)?)
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Linear transformation (matmul + optional bias) followed by an optional
/// activation, as one call. See the module-level honesty note: the matmul,
/// bias-add, and activation each still materialize their own output
/// tensor - this is not a single fused kernel.
pub struct FusedLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    activation: Option<ActivationType>,
}

impl FusedLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>, activation: Option<ActivationType>) -> Self {
        Self {
            weight,
            bias,
            activation,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Linear transformation
        let output = input.matmul(&self.weight.t()?)?;

        // Add bias if present
        let output = if let Some(ref bias) = self.bias {
            output.broadcast_add(bias)?
        } else {
            output
        };

        // Apply activation if present
        if let Some(activation) = self.activation {
            match activation {
                ActivationType::ReLU => Ok(output.relu()?),
                ActivationType::GELU => {
                    let gelu = FusedGELU::new(true);
                    gelu.forward(&output)
                },
                ActivationType::SiLU => {
                    let sigmoid_vals = output.sigmoid()?;
                    Ok(output.mul(&sigmoid_vals)?)
                },
                ActivationType::Tanh => Ok(output.tanh()?),
                ActivationType::Sigmoid => Ok(output.sigmoid()?),
            }
        } else {
            Ok(output)
        }
    }
}

/// Matrix multiplication followed by scaling, as one call. See the
/// module-level honesty note: `matmul` and the scaling multiply each still
/// materialize their own output tensor - this is not a single fused
/// kernel.
pub struct FusedMatmulScale {
    scale: f32,
}

impl FusedMatmulScale {
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }

    pub fn forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let result = a.matmul(b)?;
        if self.scale != 1.0 {
            Ok(result.mul(&Tensor::full(self.scale, result.shape())?)?)
        } else {
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Remove candle dependency

    #[test]
    fn test_fused_gelu() -> Result<()> {
        let gelu_erf = FusedGELU::new(false);
        let gelu_tanh = FusedGELU::new(true);

        // Create simple input tensor
        let input = Tensor::new(vec![1.0, -1.0, 0.5, -0.5])?;

        let output_erf = gelu_erf.forward(&input)?;
        let output_tanh = gelu_tanh.forward(&input)?;

        assert_eq!(output_erf.shape(), input.shape());
        assert_eq!(output_tanh.shape(), input.shape());

        Ok(())
    }

    #[test]
    fn test_fused_bias_activation() -> Result<()> {
        let fused_relu = FusedBiasActivation::new(ActivationType::ReLU);

        let input = Tensor::new(vec![1.0, -1.0, 0.5, -0.5])?;
        let bias = Tensor::new(vec![0.1, 0.2, 0.3, 0.4])?;

        let output = fused_relu.forward(&input, &bias)?;

        assert_eq!(output.shape(), input.shape());

        Ok(())
    }

    #[test]
    fn test_activation_types() {
        // Test that all activation types can be created
        let activations = [
            ActivationType::ReLU,
            ActivationType::GELU,
            ActivationType::SiLU,
            ActivationType::Tanh,
            ActivationType::Sigmoid,
        ];

        for activation in activations.iter() {
            let _fused = FusedBiasActivation::new(*activation);
            // Just test creation doesn't panic
        }
    }
}
