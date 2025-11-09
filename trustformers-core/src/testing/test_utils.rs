//! Standardized test utilities for trustformers-core
//!
//! This module provides common test utilities, assertions, and patterns
//! to ensure consistency across all test modules.

use crate::errors::Result;
use crate::tensor::Tensor;
use std::time::{Duration, Instant};

/// Standard test result type for consistency
pub type TestResult<T = ()> = Result<T>;

/// Test configuration for consistent test parameters
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub epsilon: f32,
    pub timeout_ms: u64,
    pub max_tensor_size: usize,
    pub enable_detailed_output: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            timeout_ms: 5000,
            max_tensor_size: 1000000,
            enable_detailed_output: false,
        }
    }
}

/// Standard tensor creation utilities for tests
pub struct TensorTestUtils;

impl TensorTestUtils {
    /// Create a test tensor with sequential values for predictable testing
    pub fn sequential_f32(shape: &[usize]) -> TestResult<Tensor> {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        Tensor::from_vec(data, shape)
    }

    /// Create a test tensor with random values in a reasonable range
    pub fn random_f32(shape: &[usize]) -> TestResult<Tensor> {
        Tensor::randn(shape)
    }

    /// Create a test tensor with small positive values (safe for log, sqrt)
    pub fn positive_f32(shape: &[usize]) -> TestResult<Tensor> {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01) + 1.0).collect();
        Tensor::from_vec(data, shape)
    }

    /// Create a test tensor with values in [0, 1] range
    pub fn normalized_f32(shape: &[usize]) -> TestResult<Tensor> {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| (i as f32) / (size as f32)).collect();
        Tensor::from_vec(data, shape)
    }

    /// Create a complex test tensor with known real and imaginary parts
    pub fn complex_test_tensor(shape: &[usize]) -> TestResult<Tensor> {
        let size: usize = shape.iter().product();
        let real: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        let imag: Vec<f32> = (0..size).map(|i| i as f32 * 0.02).collect();
        Tensor::complex(real, imag, shape)
    }
}

/// Standard assertion utilities with consistent patterns
pub struct TestAssertions;

impl TestAssertions {
    /// Assert tensors are approximately equal with standard epsilon
    pub fn assert_tensor_eq(a: &Tensor, b: &Tensor) -> TestResult<()> {
        Self::assert_tensor_eq_with_epsilon(a, b, 1e-4)
    }

    /// Assert tensors are approximately equal with custom epsilon
    pub fn assert_tensor_eq_with_epsilon(a: &Tensor, b: &Tensor, epsilon: f32) -> TestResult<()> {
        assert_eq!(a.shape(), b.shape(), "Tensor shapes must match");

        let a_data = a.to_vec_f32()?;
        let b_data = b.to_vec_f32()?;

        for (i, (val_a, val_b)) in a_data.iter().zip(b_data.iter()).enumerate() {
            // Handle special cases for infinite and NaN values
            if val_a.is_nan() && val_b.is_nan() {
                continue; // Both NaN is considered equal
            }
            if val_a.is_infinite()
                && val_b.is_infinite()
                && val_a.is_sign_positive() == val_b.is_sign_positive()
            {
                continue; // Same infinity is considered equal
            }
            if val_a.is_nan() || val_b.is_nan() || val_a.is_infinite() || val_b.is_infinite() {
                panic!(
                    "Values differ at index {}: {} vs {} (diff: {})",
                    i,
                    val_a,
                    val_b,
                    (val_a - val_b).abs()
                );
            }

            // Use relative tolerance for larger values and absolute tolerance for smaller values
            let max_val = val_a.abs().max(val_b.abs());
            let tolerance = if max_val > 1.0 {
                epsilon * max_val // Relative tolerance
            } else {
                epsilon // Absolute tolerance
            };

            let diff = (val_a - val_b).abs();
            assert!(
                diff <= tolerance,
                "Values differ at index {}: {} vs {} (diff: {}, tolerance: {})",
                i,
                val_a,
                val_b,
                diff,
                tolerance
            );
        }
        Ok(())
    }

    /// Assert tensor has expected shape
    pub fn assert_shape(tensor: &Tensor, expected_shape: &[usize]) -> TestResult<()> {
        assert_eq!(tensor.shape(), expected_shape, "Unexpected tensor shape");
        Ok(())
    }

    /// Assert all tensor values are within bounds
    pub fn assert_values_in_range(tensor: &Tensor, min: f32, max: f32) -> TestResult<()> {
        let data = tensor.to_vec_f32()?;
        for (i, &val) in data.iter().enumerate() {
            assert!(
                val >= min && val <= max,
                "Value {} at index {} is outside range [{}, {}]",
                val,
                i,
                min,
                max
            );
        }
        Ok(())
    }

    /// Assert tensor contains only finite values (no NaN or infinite)
    pub fn assert_finite_values(tensor: &Tensor) -> TestResult<()> {
        let data = tensor.to_vec_f32()?;
        for (i, &val) in data.iter().enumerate() {
            assert!(val.is_finite(), "Non-finite value {} at index {}", val, i);
        }
        Ok(())
    }

    /// Alias for assert_finite_values for backward compatibility
    pub fn assert_all_finite(tensor: &Tensor) -> TestResult<()> {
        Self::assert_finite_values(tensor)
    }

    /// Assert operation preserves tensor properties
    pub fn assert_operation_preserves_shape<F>(input: &Tensor, operation: F) -> TestResult<()>
    where
        F: FnOnce(&Tensor) -> TestResult<Tensor>,
    {
        let input_shape = input.shape().to_vec();
        let result = operation(input)?;
        Self::assert_shape(&result, &input_shape)?;
        Ok(())
    }
}

/// Performance testing utilities
pub struct PerformanceTestUtils;

impl PerformanceTestUtils {
    /// Measure execution time of an operation
    pub fn measure_time<F, T>(operation: F) -> (T, Duration)
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Assert operation completes within timeout
    pub fn assert_timeout<F, T>(operation: F, timeout: Duration) -> TestResult<T>
    where
        F: FnOnce() -> TestResult<T>,
    {
        let (result, duration) = Self::measure_time(operation);
        assert!(
            duration <= timeout,
            "Operation took {:?}, exceeding timeout of {:?}",
            duration,
            timeout
        );
        result
    }

    /// Benchmark operation with multiple iterations
    pub fn benchmark_operation<F>(operation: F, iterations: usize) -> BenchmarkResult
    where
        F: Fn() -> TestResult<()>,
    {
        let mut timings = Vec::with_capacity(iterations);
        let mut errors = 0;

        for _ in 0..iterations {
            let (result, duration) = Self::measure_time(&operation);

            match result {
                Ok(()) => timings.push(duration),
                Err(_) => errors += 1,
            }
        }

        BenchmarkResult::new(timings, errors)
    }
}

/// Benchmark result with statistics
#[derive(Debug)]
pub struct BenchmarkResult {
    pub timings: Vec<Duration>,
    pub errors: usize,
    pub mean: Duration,
    pub median: Duration,
    pub min: Duration,
    pub max: Duration,
    pub std_dev: Duration,
}

impl BenchmarkResult {
    fn new(mut timings: Vec<Duration>, errors: usize) -> Self {
        if timings.is_empty() {
            return Self {
                timings,
                errors,
                mean: Duration::ZERO,
                median: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                std_dev: Duration::ZERO,
            };
        }

        timings.sort();

        let total: Duration = timings.iter().sum();
        let mean = total / timings.len() as u32;
        let median = timings[timings.len() / 2];
        let min = timings[0];
        let max = timings[timings.len() - 1];

        // Calculate standard deviation
        let variance_sum: u128 = timings
            .iter()
            .map(|&d| (d.as_nanos() as i128 - mean.as_nanos() as i128).unsigned_abs())
            .map(|diff| diff.pow(2))
            .sum();
        let variance = variance_sum / timings.len() as u128;
        let std_dev = Duration::from_nanos((variance as f64).sqrt() as u64);

        Self {
            timings,
            errors,
            mean,
            median,
            min,
            max,
            std_dev,
        }
    }

    pub fn success_rate(&self) -> f64 {
        let total = self.timings.len() + self.errors;
        if total == 0 {
            0.0
        } else {
            self.timings.len() as f64 / total as f64
        }
    }
}

/// Property-based testing utilities
#[cfg(test)]
pub mod property_utils {

    use proptest::prelude::*;

    /// Strategy for generating valid tensor shapes (1D to 4D, reasonable sizes)
    pub fn tensor_shapes() -> impl Strategy<Value = Vec<usize>> {
        use proptest::prelude::*;
        prop_oneof![
            // 1D shapes
            (1..100usize).prop_map(|n| vec![n]),
            // 2D shapes
            (1..50usize, 1..50usize).prop_map(|(m, n)| vec![m, n]),
            // 3D shapes
            (1..20usize, 1..20usize, 1..20usize).prop_map(|(a, b, c)| vec![a, b, c]),
            // 4D shapes
            (1..10usize, 1..10usize, 1..10usize, 1..10usize)
                .prop_map(|(a, b, c, d)| vec![a, b, c, d]),
        ]
    }

    /// Strategy for matrix multiplication compatible shapes
    #[cfg(test)]
    pub fn matmul_shapes() -> impl Strategy<Value = (Vec<usize>, Vec<usize>)> {
        use proptest::prelude::*;
        prop_oneof![
            // 2D x 2D: [m, k] x [k, n]
            (2..20usize, 2..20usize, 2..20usize).prop_map(|(m, k, n)| (vec![m, k], vec![k, n])),
            // 3D x 3D: [b, m, k] x [b, k, n]
            (1..10usize, 2..15usize, 2..15usize, 2..15usize)
                .prop_map(|(b, m, k, n)| (vec![b, m, k], vec![b, k, n])),
        ]
    }

    /// Strategy for reasonable floating point values (avoiding extreme values)
    #[cfg(test)]
    pub fn reasonable_f32() -> impl Strategy<Value = f32> {
        use proptest::prelude::*;
        (-1000.0f32..1000.0f32).prop_filter("finite", |x| x.is_finite())
    }

    /// Strategy for positive floating point values
    #[cfg(test)]
    pub fn positive_f32() -> impl Strategy<Value = f32> {
        use proptest::prelude::*;
        (0.001f32..1000.0f32).prop_filter("finite_positive", |x| x.is_finite() && *x > 0.0)
    }
}

/// Error testing utilities
pub struct ErrorTestUtils;

impl ErrorTestUtils {
    /// Assert operation fails with expected error type
    pub fn assert_error_type<T, E, F>(operation: F, expected_error_check: E) -> TestResult<()>
    where
        F: FnOnce() -> Result<T>,
        E: FnOnce(&crate::errors::TrustformersError) -> bool,
    {
        match operation() {
            Ok(_) => panic!("Expected operation to fail, but it succeeded"),
            Err(err) => {
                assert!(
                    expected_error_check(&err),
                    "Unexpected error type: {:?}",
                    err
                );
            },
        }
        Ok(())
    }

    /// Assert incompatible tensor operations fail appropriately
    pub fn assert_incompatible_shapes_fail(a: &Tensor, b: &Tensor) -> TestResult<()> {
        if a.shape() != b.shape() {
            assert!(a.add(b).is_err(), "Add should fail for incompatible shapes");
            assert!(a.sub(b).is_err(), "Sub should fail for incompatible shapes");
            assert!(a.mul(b).is_err(), "Mul should fail for incompatible shapes");
        }
        Ok(())
    }
}

/// Test macros for common patterns
#[macro_export]
macro_rules! test_tensor_property {
    ($name:ident, $property:expr) => {
        #[test]
        fn $name() -> $crate::testing::test_utils::TestResult<()> {
            use $crate::testing::test_utils::*;
            $property
        }
    };
}

#[macro_export]
macro_rules! test_with_shapes {
    ($name:ident, $shapes:expr, $test_fn:expr) => {
        #[test]
        fn $name() -> $crate::testing::test_utils::TestResult<()> {
            use $crate::testing::test_utils::*;
            for shape in $shapes {
                $test_fn(&shape)?;
            }
            Ok(())
        }
    };
}

#[macro_export]
macro_rules! benchmark_tensor_op {
    ($name:ident, $setup:expr, $operation:expr) => {
        #[test]
        fn $name() -> $crate::testing::test_utils::TestResult<()> {
            use $crate::testing::test_utils::*;

            let setup_result = $setup;
            let benchmark_result =
                PerformanceTestUtils::benchmark_operation(|| $operation(&setup_result), 10);

            println!(
                "Benchmark {}: mean={:?}, median={:?}, min={:?}, max={:?}",
                stringify!($name),
                benchmark_result.mean,
                benchmark_result.median,
                benchmark_result.min,
                benchmark_result.max
            );

            // Assert reasonable performance (less than 1 second)
            assert!(benchmark_result.mean < Duration::from_secs(1));
            Ok(())
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_tensor_creation() -> TestResult<()> {
        let tensor = TensorTestUtils::sequential_f32(&[2, 3])?;
        TestAssertions::assert_shape(&tensor, &[2, 3])?;

        let expected_data = vec![0.0, 0.01, 0.02, 0.03, 0.04, 0.05];
        TestAssertions::assert_tensor_eq_with_epsilon(
            &tensor,
            &Tensor::from_vec(expected_data, &[2, 3])?,
            1e-6,
        )?;
        Ok(())
    }

    #[test]
    fn test_performance_measurement() -> TestResult<()> {
        let (result, duration) = PerformanceTestUtils::measure_time(|| {
            std::thread::sleep(Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(10));
        Ok(())
    }

    #[test]
    fn test_benchmark_result() {
        let timings = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(15),
        ];

        let result = BenchmarkResult::new(timings, 0);
        assert_eq!(result.success_rate(), 1.0);
        assert!(result.mean > Duration::ZERO);
        assert!(result.min <= result.median);
        assert!(result.median <= result.max);
    }
}
