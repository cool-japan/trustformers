//! Dropout layer implementation for regularization
//!
//! Dropout randomly sets a fraction of the input elements to zero during training,
//! which helps prevent overfitting.

use crate::errors::Result;
use crate::tensor::Tensor;
use crate::traits::Layer;
use rand::Rng;

/// Dropout layer for regularization during training
///
/// During training, randomly sets input elements to zero with probability `p`,
/// and scales remaining elements by `1/(1-p)` to maintain expected output.
/// During evaluation (inference), acts as identity function.
///
/// # Example
///
/// ```no_run
/// use trustformers_core::layers::Dropout;
/// use trustformers_core::tensor::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let dropout = Dropout::new(0.1); // 10% dropout rate
/// let input = Tensor::randn(&[32, 768])?;
/// let output = dropout.forward(&input)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Dropout {
    /// Dropout probability (0.0 to 1.0)
    p: f32,
    /// Whether layer is in training mode
    training: bool,
}

impl Dropout {
    /// Create a new dropout layer
    ///
    /// # Arguments
    ///
    /// * `p` - Dropout probability, should be between 0.0 and 1.0
    ///
    /// # Panics
    ///
    /// Panics if `p` is not between 0.0 and 1.0
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Dropout probability must be between 0.0 and 1.0, got {}",
            p
        );

        Self { p, training: true }
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Get dropout probability
    pub fn dropout_rate(&self) -> f32 {
        self.p
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }
}

impl Layer for Dropout {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // During evaluation, just return input unchanged
        if !self.training || self.p == 0.0 {
            return Ok(input);
        }

        // During training, apply dropout
        let mut rng = rand::rng();
        let data = input.data()?;
        let mut output_data = Vec::with_capacity(data.len());

        let keep_prob = 1.0 - self.p;
        let scale = 1.0 / keep_prob;

        for &value in &data {
            if rng.random::<f32>() < keep_prob {
                // Keep the element, but scale to maintain expected value
                output_data.push(value * scale);
            } else {
                // Drop the element (set to zero)
                output_data.push(0.0);
            }
        }

        Tensor::from_vec(output_data, &input.shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_creation() {
        let dropout = Dropout::new(0.5);
        assert_eq!(dropout.dropout_rate(), 0.5);
        assert!(dropout.is_training());
    }

    #[test]
    fn test_dropout_inference_mode() {
        let mut dropout = Dropout::new(0.5);
        dropout.set_training(false);

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let input_data = input.data().unwrap();
        let output = dropout.forward(input).unwrap();
        let output_data = output.data().unwrap();

        // In inference mode, output should equal input
        assert_eq!(input_data, output_data);
    }

    #[test]
    fn test_dropout_zero_rate() {
        let dropout = Dropout::new(0.0);

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let input_data = input.data().unwrap();
        let input_shape = input.shape().to_vec();
        let output = dropout.forward(input).unwrap();
        let output_data = output.data().unwrap();
        let output_shape = output.shape().to_vec();

        // With 0.0 dropout, output should equal input
        assert_eq!(input_data, output_data);
        assert_eq!(input_shape, output_shape);
    }

    #[test]
    fn test_dropout_full_rate() {
        let dropout = Dropout::new(1.0);

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let input_shape = input.shape().to_vec();
        let output = dropout.forward(input).unwrap();
        let output_data = output.data().unwrap();
        let output_shape = output.shape().to_vec();

        // With 1.0 dropout, all elements should be zero
        assert!(output_data.iter().all(|&x| x == 0.0));
        assert_eq!(input_shape, output_shape);
    }

    #[test]
    fn test_dropout_statistical_properties() {
        let dropout = Dropout::new(0.5);
        let size = 1000;

        // Run dropout multiple times and check statistics
        let mut zero_counts = Vec::new();
        let mut sums = Vec::new();

        for _ in 0..20 {
            let input = Tensor::from_vec(vec![1.0; size], &[size]).unwrap();
            let output = dropout.forward(input).unwrap();
            let output_data = output.data().unwrap();
            let zero_count = output_data.iter().filter(|&&x| x == 0.0).count();
            let sum: f32 = output_data.iter().sum();

            zero_counts.push(zero_count);
            sums.push(sum);
        }

        // Check that approximately 50% of elements are zeroed
        let avg_zero_rate = zero_counts.iter().sum::<usize>() as f32 / (20.0 * size as f32);
        assert!(
            (avg_zero_rate - 0.5).abs() < 0.1,
            "Expected ~50% zeros, got {:.1}%",
            avg_zero_rate * 100.0
        );

        // Check that the sum is preserved on average (due to scaling)
        let avg_sum = sums.iter().sum::<f32>() / 20.0;
        let expected_sum = size as f32; // Input sum
        assert!(
            (avg_sum - expected_sum).abs() < expected_sum * 0.2,
            "Sum not preserved: expected {}, got {}",
            expected_sum,
            avg_sum
        );
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be between 0.0 and 1.0")]
    fn test_invalid_dropout_rate_high() {
        Dropout::new(1.5);
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be between 0.0 and 1.0")]
    fn test_invalid_dropout_rate_negative() {
        Dropout::new(-0.1);
    }
}
