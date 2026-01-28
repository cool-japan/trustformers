//! Reference value comparison utilities

use anyhow::{Error, Result};
use trustformers_core::tensor::Tensor;

use super::types::NumericalDifferences;

/// Reference value comparison utilities
pub struct ReferenceComparator {
    tolerance: f32,
}

impl ReferenceComparator {
    /// Create a new reference comparator
    pub fn new(tolerance: f32) -> Self {
        Self { tolerance }
    }

    /// Get the tolerance value
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Compare model output with reference values
    pub fn compare_with_reference(
        &self,
        actual: &Tensor,
        expected: &Tensor,
    ) -> Result<NumericalDifferences> {
        match (actual, expected) {
            (Tensor::F32(actual_arr), Tensor::F32(expected_arr)) => {
                if actual_arr.shape() != expected_arr.shape() {
                    return Err(Error::msg("Tensor shapes don't match"));
                }

                let diffs: Vec<f32> = actual_arr
                    .iter()
                    .zip(expected_arr.iter())
                    .map(|(a, e)| (a - e).abs())
                    .collect();

                let max_abs_diff = diffs.iter().cloned().fold(0.0, f32::max);
                let mean_abs_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
                let rms_diff =
                    (diffs.iter().map(|d| d * d).sum::<f32>() / diffs.len() as f32).sqrt();
                let within_tolerance = diffs.iter().filter(|&&d| d <= self.tolerance).count();
                let within_tolerance_percent =
                    (within_tolerance as f32 / diffs.len() as f32) * 100.0;

                Ok(NumericalDifferences {
                    max_abs_diff,
                    mean_abs_diff,
                    rms_diff,
                    within_tolerance_percent,
                })
            },
            _ => Err(Error::msg("Unsupported tensor types for comparison")),
        }
    }

    /// Validate that differences are within acceptable bounds
    pub fn validate_differences(&self, differences: &NumericalDifferences) -> bool {
        differences.max_abs_diff <= self.tolerance && differences.within_tolerance_percent >= 95.0
    }
}
