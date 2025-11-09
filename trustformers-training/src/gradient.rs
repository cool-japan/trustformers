use trustformers_core::errors::{tensor_op_error, Result};
use trustformers_core::Tensor;

/// Utility functions for gradient operations
pub struct GradientUtils;

impl GradientUtils {
    /// Clip gradients by global norm
    ///
    /// This function clips the gradients of a list of tensors by their global norm.
    /// If the global norm exceeds max_norm, all gradients are scaled down proportionally.
    ///
    /// # Arguments
    /// * `gradients` - Mutable reference to a vector of gradients
    /// * `max_norm` - Maximum allowed norm for gradient clipping
    ///
    /// # Returns
    /// The actual norm of the gradients before clipping
    pub fn clip_grad_norm(gradients: &mut Vec<Tensor>, max_norm: f32) -> Result<f32> {
        if gradients.is_empty() {
            return Ok(0.0);
        }

        // Compute global norm
        let mut total_norm_sq = 0.0;
        for grad in gradients.iter() {
            let norm = grad.norm()?;
            total_norm_sq += norm * norm;
        }
        let total_norm = total_norm_sq.sqrt();

        // Clip if necessary
        if total_norm > max_norm {
            let clip_coef = max_norm / total_norm;
            for grad in gradients.iter_mut() {
                *grad = grad.scale(clip_coef)?;
            }
        }

        Ok(total_norm)
    }

    /// Clip gradients by value
    ///
    /// This function clips each gradient tensor element-wise to be within [-clip_value, clip_value]
    ///
    /// # Arguments
    /// * `gradients` - Mutable reference to a vector of gradients
    /// * `clip_value` - Maximum absolute value for each gradient element
    pub fn clip_grad_value(gradients: &mut Vec<Tensor>, clip_value: f32) -> Result<()> {
        for grad in gradients.iter_mut() {
            match grad {
                Tensor::F32(arr) => {
                    arr.mapv_inplace(|x| x.clamp(-clip_value, clip_value));
                },
                Tensor::F64(arr) => {
                    let clip_value_f64 = clip_value as f64;
                    arr.mapv_inplace(|x| x.clamp(-clip_value_f64, clip_value_f64));
                },
                _ => {
                    return Err(tensor_op_error(
                        "gradient_clipping",
                        "Unsupported tensor type for gradient value clipping",
                    ))
                },
            }
        }
        Ok(())
    }

    /// Accumulate gradients by averaging
    ///
    /// This function adds new gradients to accumulated gradients and scales them
    /// by 1/accumulation_steps to maintain proper scaling
    ///
    /// # Arguments
    /// * `accumulated_grads` - Mutable reference to accumulated gradients
    /// * `new_grads` - New gradients to add
    /// * `accumulation_steps` - Number of accumulation steps (for scaling)
    pub fn accumulate_gradients(
        accumulated_grads: &mut Vec<Tensor>,
        new_grads: &[Tensor],
        accumulation_steps: usize,
    ) -> Result<()> {
        if accumulated_grads.len() != new_grads.len() {
            return Err(tensor_op_error(
                "gradient_accumulation",
                "mismatched number of gradients",
            ));
        }

        let scale = 1.0 / accumulation_steps as f32;

        for (acc_grad, new_grad) in accumulated_grads.iter_mut().zip(new_grads.iter()) {
            let scaled_new_grad = new_grad.scale(scale)?;
            *acc_grad = acc_grad.add(&scaled_new_grad)?;
        }

        Ok(())
    }

    /// Zero out accumulated gradients
    pub fn zero_accumulated_gradients(accumulated_grads: &mut Vec<Tensor>) -> Result<()> {
        for grad in accumulated_grads.iter_mut() {
            match grad {
                Tensor::F32(arr) => {
                    arr.fill(0.0);
                },
                Tensor::F64(arr) => {
                    arr.fill(0.0);
                },
                _ => {
                    return Err(tensor_op_error(
                        "gradient_zeroing",
                        "Unsupported tensor type for zeroing gradients",
                    ))
                },
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::Tensor;

    #[test]
    fn test_clip_grad_norm() {
        let mut gradients = vec![
            Tensor::zeros(&[2, 2]).unwrap(),
            Tensor::zeros(&[3, 3]).unwrap(),
        ];

        // Set some values
        if let Tensor::F32(ref mut arr) = gradients[0] {
            arr[[0, 0]] = 3.0;
            arr[[0, 1]] = 4.0; // norm = 5.0
        }

        if let Tensor::F32(ref mut arr) = gradients[1] {
            arr[[0, 0]] = 6.0;
            arr[[0, 1]] = 8.0; // norm = 10.0
        }

        // Total norm should be sqrt(5^2 + 10^2) = sqrt(125) ≈ 11.18
        let norm = GradientUtils::clip_grad_norm(&mut gradients, 5.0).unwrap();
        assert!(norm > 11.0 && norm < 12.0);

        // After clipping, gradients should be scaled down
        if let Tensor::F32(ref arr) = gradients[0] {
            assert!(arr[[0, 0]] < 3.0);
            assert!(arr[[0, 1]] < 4.0);
        }
    }

    #[test]
    fn test_clip_grad_value() {
        let mut gradients = vec![Tensor::zeros(&[2, 2]).unwrap()];

        // Set some values that exceed clip_value
        if let Tensor::F32(ref mut arr) = gradients[0] {
            arr[[0, 0]] = 10.0;
            arr[[0, 1]] = -15.0;
            arr[[1, 0]] = 2.0;
            arr[[1, 1]] = -3.0;
        }

        GradientUtils::clip_grad_value(&mut gradients, 5.0).unwrap();

        // Check that values are clipped
        if let Tensor::F32(ref arr) = gradients[0] {
            assert_eq!(arr[[0, 0]], 5.0); // 10.0 clipped to 5.0
            assert_eq!(arr[[0, 1]], -5.0); // -15.0 clipped to -5.0
            assert_eq!(arr[[1, 0]], 2.0); // 2.0 unchanged
            assert_eq!(arr[[1, 1]], -3.0); // -3.0 unchanged
        }
    }
}
