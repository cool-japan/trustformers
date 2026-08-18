//! SciRS2 Compatibility Layer
//!
//! This module provides compatibility types and traits for SciRS2 integration
//! until the full SciRS2 API is available.

use serde::{Deserialize, Serialize};
use trustformers_core::TrustformersError as CoreError;

pub type Result<T> = std::result::Result<T, CoreError>;

/// Compatibility tensor type (placeholder for SciRS2 tensor)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T: Clone> Tensor<T> {
    pub fn from_slice(data: &[T], shape: &[usize]) -> Result<Self> {
        Ok(Self {
            data: data.to_vec(),
            shape: shape.to_vec(),
        })
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Tensor<f32> {
    /// Element-wise greater than scalar comparison (returns boolean tensor as f32)
    pub fn gt_scalar(&self, scalar: f32) -> Result<Tensor<f32>> {
        let result_data: Vec<f32> =
            self.data.iter().map(|x| if *x > scalar { 1.0 } else { 0.0 }).collect();
        Tensor::from_slice(&result_data, &self.shape)
    }

    /// Element-wise greater than scalar comparison (returns boolean values)
    pub fn gt_scalar_bool(&self, scalar: f32) -> Vec<bool> {
        self.data.iter().map(|x| *x > scalar).collect()
    }

    /// Clone data as vector
    pub fn data_cloned(&self) -> Vec<f32> {
        self.data.clone()
    }
}

/// Placeholder SIMD operations trait
pub struct SimdOps {
    vector_width: usize,
}

impl SimdOps {
    pub fn new_with_width(_width: usize) -> Result<Self> {
        Ok(Self { vector_width: 256 })
    }

    pub fn vector_width(&self) -> usize {
        self.vector_width
    }

    /// Real Pearson product-moment correlation coefficient between `a` and
    /// `b`, computed in `f64` for numerical stability and cast back to
    /// `f32`. Previously this ignored both arguments entirely and always
    /// returned the constant `0.5`.
    pub fn correlation(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<f32> {
        if a.data.len() != b.data.len() {
            return Err(CoreError::runtime_error(format!(
                "correlation requires equal-length tensors, got {} and {}",
                a.data.len(),
                b.data.len()
            )));
        }
        if a.data.is_empty() {
            return Err(CoreError::runtime_error(
                "correlation requires non-empty tensors".to_string(),
            ));
        }

        let n = a.data.len() as f64;
        let mean_a = a.data.iter().map(|&x| x as f64).sum::<f64>() / n;
        let mean_b = b.data.iter().map(|&x| x as f64).sum::<f64>() / n;

        let mut covariance = 0.0f64;
        let mut variance_a = 0.0f64;
        let mut variance_b = 0.0f64;
        for (&x, &y) in a.data.iter().zip(b.data.iter()) {
            let dx = x as f64 - mean_a;
            let dy = y as f64 - mean_b;
            covariance += dx * dy;
            variance_a += dx * dx;
            variance_b += dy * dy;
        }

        let denominator = (variance_a * variance_b).sqrt();
        if denominator == 0.0 {
            // At least one series is constant, so Pearson's r is
            // mathematically undefined (0/0 covariance ratio). Reporting 0
            // (no linear relationship measurable) rather than NaN keeps
            // this a total function over all finite inputs.
            return Ok(0.0);
        }

        Ok((covariance / denominator) as f32)
    }

    pub fn abs(&self, tensor: &Tensor<f32>) -> Result<Tensor<f32>> {
        let abs_data: Vec<f32> = tensor.data.iter().map(|x| x.abs()).collect();
        Tensor::from_slice(&abs_data, &tensor.shape)
    }

    pub fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
        let result_data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect();
        Tensor::from_slice(&result_data, &a.shape)
    }

    pub fn sub(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
        let result_data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x - y).collect();
        Tensor::from_slice(&result_data, &a.shape)
    }

    pub fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
        let result_data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).collect();
        Tensor::from_slice(&result_data, &a.shape)
    }

    pub fn sub_scalar(&self, tensor: &Tensor<f32>, scalar: f32) -> Result<Tensor<f32>> {
        let result_data: Vec<f32> = tensor.data.iter().map(|x| x - scalar).collect();
        Tensor::from_slice(&result_data, &tensor.shape)
    }

    pub fn add_scalar(&self, tensor: &Tensor<f32>, scalar: f32) -> Result<Tensor<f32>> {
        let result_data: Vec<f32> = tensor.data.iter().map(|x| x + scalar).collect();
        Tensor::from_slice(&result_data, &tensor.shape)
    }

    pub fn mul_scalar(&self, tensor: &Tensor<f32>, scalar: f32) -> Result<Tensor<f32>> {
        let result_data: Vec<f32> = tensor.data.iter().map(|x| x * scalar).collect();
        Tensor::from_slice(&result_data, &tensor.shape)
    }

    pub fn div_scalar(&self, tensor: &Tensor<f32>, scalar: f32) -> Result<Tensor<f32>> {
        let result_data: Vec<f32> = tensor.data.iter().map(|x| x / scalar).collect();
        Tensor::from_slice(&result_data, &tensor.shape)
    }

    pub fn pow_scalar(&self, tensor: &Tensor<f32>, exponent: f32) -> Result<Tensor<f32>> {
        let result_data: Vec<f32> = tensor.data.iter().map(|x| x.powf(exponent)).collect();
        Tensor::from_slice(&result_data, &tensor.shape)
    }
}

/// Placeholder linear algebra operations trait
pub struct LinalgOps;

impl LinalgOps {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LinalgOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Placeholder statistical operations trait
pub struct StatisticalOps;

impl StatisticalOps {
    pub fn new() -> Self {
        Self
    }

    pub fn simd_mean(&self, tensor: &Tensor<f32>) -> Result<f32> {
        let sum: f32 = tensor.data.iter().sum();
        Ok(sum / tensor.data.len() as f32)
    }

    pub fn simd_variance(&self, tensor: &Tensor<f32>) -> Result<f32> {
        let mean = self.simd_mean(tensor)?;
        let variance =
            tensor.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / tensor.data.len() as f32;
        Ok(variance)
    }

    pub fn simd_std(&self, tensor: &Tensor<f32>) -> Result<f32> {
        Ok(self.simd_variance(tensor)?.sqrt())
    }

    pub fn simd_sum(&self, tensor: &Tensor<f32>) -> Result<f32> {
        Ok(tensor.data.iter().sum())
    }

    pub fn simd_minmax(&self, tensor: &Tensor<f32>) -> Result<(f32, f32)> {
        let min = tensor.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = tensor.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        Ok((min, max))
    }

    pub fn simd_quantile(&self, tensor: &Tensor<f32>, quantile: f32) -> Result<f32> {
        let mut sorted_data = tensor.data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let index = ((sorted_data.len() - 1) as f32 * quantile) as usize;
        Ok(sorted_data[index.min(sorted_data.len() - 1)])
    }
}

impl Default for StatisticalOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Placeholder distribution operations trait
pub struct DistributionOps;

impl DistributionOps {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DistributionOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Placeholder RNG type for compatibility
pub struct DefaultRng;

impl DefaultRng {
    pub fn new() -> Self {
        Self
    }

    /// Generate a random value of the specified type
    pub fn gen<T>(&mut self) -> T
    where
        T: RandomGenerate,
    {
        T::generate()
    }

    /// Generate a random value (rand 0.10 compatible name)
    pub fn random<T>(&mut self) -> T
    where
        T: RandomGenerate,
    {
        T::generate()
    }
}

impl Default for DefaultRng {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can be randomly generated
pub trait RandomGenerate {
    fn generate() -> Self;
}

impl RandomGenerate for f32 {
    fn generate() -> Self {
        fastrand::f32()
    }
}

impl RandomGenerate for f64 {
    fn generate() -> Self {
        fastrand::f64()
    }
}

impl RandomGenerate for u32 {
    fn generate() -> Self {
        fastrand::u32(..)
    }
}

impl RandomGenerate for u64 {
    fn generate() -> Self {
        fastrand::u64(..)
    }
}

impl RandomGenerate for usize {
    fn generate() -> Self {
        fastrand::usize(..)
    }
}

impl<T> Tensor<T> {
    pub fn to_vec(&self) -> &Vec<T> {
        &self.data
    }
}

/// Random number generation module for compatibility
pub mod random {
    /// Legacy random number generation functions
    pub mod legacy {
        /// Generate a random f32 value between 0.0 and 1.0
        pub fn f32() -> f32 {
            fastrand::f32()
        }

        /// Generate a random f64 value between 0.0 and 1.0
        pub fn f64() -> f64 {
            fastrand::f64()
        }

        /// Generate a random usize value
        pub fn usize_range(min: usize, max: usize) -> usize {
            if max <= min {
                min
            } else {
                fastrand::usize(min..max)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the previous `correlation` implementation, which
    /// ignored both tensors entirely and always returned `0.5`.
    #[test]
    fn test_correlation_is_computed_not_a_flat_constant() {
        let ops = SimdOps::new_with_width(256).expect("simd ops");

        // Perfectly positively correlated: b = 2*a.
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("tensor a");
        let b = Tensor::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0], &[5]).expect("tensor b");
        let r = ops.correlation(&a, &b).expect("correlation");
        assert!((r - 1.0).abs() < 1e-5, "expected r ~= 1.0, got {r}");
        assert_ne!(r, 0.5);

        // Perfectly negatively correlated.
        let c = Tensor::from_slice(&[5.0, 4.0, 3.0, 2.0, 1.0], &[5]).expect("tensor c");
        let r_neg = ops.correlation(&a, &c).expect("correlation");
        assert!(
            (r_neg - (-1.0)).abs() < 1e-5,
            "expected r ~= -1.0, got {r_neg}"
        );

        // Unrelated-but-not-degenerate data should not coincidentally land
        // on the old placeholder value either.
        let d = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]).expect("tensor d");
        let r_other = ops.correlation(&a, &d).expect("correlation");
        assert_ne!(r_other, 0.5);
    }

    #[test]
    fn test_correlation_rejects_mismatched_lengths() {
        let ops = SimdOps::new_with_width(256).expect("simd ops");
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).expect("tensor a");
        let b = Tensor::from_slice(&[1.0, 2.0], &[2]).expect("tensor b");
        assert!(ops.correlation(&a, &b).is_err());
    }

    #[test]
    fn test_correlation_of_constant_series_is_zero_not_nan() {
        let ops = SimdOps::new_with_width(256).expect("simd ops");
        let a = Tensor::from_slice(&[7.0, 7.0, 7.0], &[3]).expect("tensor a");
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).expect("tensor b");
        let r = ops.correlation(&a, &b).expect("correlation");
        assert_eq!(r, 0.0);
    }
}
