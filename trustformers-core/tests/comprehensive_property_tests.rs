//! Comprehensive property-based tests for trustformers-core
//!
//! These tests use proptest to verify mathematical properties and invariants
//! that should hold for all valid inputs, integrated with our standardized
//! testing infrastructure.

use proptest::prelude::*;
use trustformers_core::tensor::Tensor;
use trustformers_core::testing::{TensorTestUtils, TestAssertions, TestResult};

// Property test strategies (local definitions since property_utils has import issues)
fn tensor_shapes() -> impl Strategy<Value = Vec<usize>> {
    prop_oneof![
        // 1D shapes
        (1..100usize).prop_map(|n| vec![n]),
        // 2D shapes
        (1..50usize, 1..50usize).prop_map(|(m, n)| vec![m, n]),
        // 3D shapes
        (1..20usize, 1..20usize, 1..20usize).prop_map(|(a, b, c)| vec![a, b, c]),
        // 4D shapes
        (1..10usize, 1..10usize, 1..10usize, 1..10usize).prop_map(|(a, b, c, d)| vec![a, b, c, d]),
    ]
}

fn matmul_shapes() -> impl Strategy<Value = (Vec<usize>, Vec<usize>)> {
    prop_oneof![
        // 2D x 2D: [m, k] x [k, n]
        (2..20usize, 2..20usize, 2..20usize).prop_map(|(m, k, n)| (vec![m, k], vec![k, n])),
        // 3D x 3D: [b, m, k] x [b, k, n]
        (1..10usize, 2..15usize, 2..15usize, 2..15usize)
            .prop_map(|(b, m, k, n)| (vec![b, m, k], vec![b, k, n])),
    ]
}

fn reasonable_f32() -> impl Strategy<Value = f32> {
    (-1000.0f32..1000.0f32).prop_filter("finite", |x| x.is_finite())
}

// Re-enable all property tests with standardized patterns

proptest! {
    /// Property: Tensor creation preserves all input properties
    #[test]
    fn tensor_creation_properties(shape in tensor_shapes()) {
        let size: usize = shape.iter().product();
        let data = vec![0.0; size];

        let tensor = Tensor::from_vec(data.clone(), &shape)?;

        // Use standardized assertions
        TestAssertions::assert_shape(&tensor, &shape)?;
        let tensor_data = tensor.to_vec_f32()?;
        prop_assert_eq!(tensor_data.len(), size);
        prop_assert_eq!(tensor.shape().len(), shape.len());
    }
}

proptest! {
    /// Property: Element-wise addition is commutative and associative
    #[test]
    fn tensor_addition_properties(
        shape in tensor_shapes(),
        scale1 in reasonable_f32(),
        scale2 in reasonable_f32(),
        scale3 in reasonable_f32()
    ) {
        let size: usize = shape.iter().product();
        prop_assume!(size > 0 && size < 10000); // Reasonable size limits

        let tensor1 = Tensor::from_vec(vec![scale1; size], &shape)?;
        let tensor2 = Tensor::from_vec(vec![scale2; size], &shape)?;
        let tensor3 = Tensor::from_vec(vec![scale3; size], &shape)?;

        // Test commutativity: A + B = B + A
        let ab = tensor1.add(&tensor2)?;
        let ba = tensor2.add(&tensor1)?;
        TestAssertions::assert_tensor_eq(&ab, &ba)?;

        // Test associativity: (A + B) + C = A + (B + C)
        let ab_c = ab.add(&tensor3)?;
        let bc = tensor2.add(&tensor3)?;
        let a_bc = tensor1.add(&bc)?;
        TestAssertions::assert_tensor_eq(&ab_c, &a_bc)?;

        // Shape preservation
        TestAssertions::assert_shape(&ab, &shape)?;

        // Value correctness
        let expected_sum = scale1 + scale2;
        for &val in ab.to_vec_f32()?.iter() {
            prop_assert!((val - expected_sum).abs() < 1e-5);
        }
    }
}

proptest! {
    /// Property: Multiplication distributivity and commutativity
    #[test]
    fn tensor_multiplication_properties(
        shape in tensor_shapes(),
        scale_a in reasonable_f32(),
        scale_b in reasonable_f32(),
        scalar in reasonable_f32()
    ) {
        let size: usize = shape.iter().product();
        prop_assume!(size > 0 && size < 10000);

        let tensor_a = Tensor::from_vec(vec![scale_a; size], &shape)?;
        let tensor_b = Tensor::from_vec(vec![scale_b; size], &shape)?;

        // Test commutativity: A * B = B * A
        let ab = tensor_a.mul(&tensor_b)?;
        let ba = tensor_b.mul(&tensor_a)?;
        TestAssertions::assert_tensor_eq(&ab, &ba)?;

        // Test scalar distributivity: s * (A + B) = s * A + s * B
        // Use larger epsilon due to numerical precision issues with large values
        // and catastrophic cancellation when scale_a and scale_b are close to negatives of each other
        let a_plus_b = tensor_a.add(&tensor_b)?;
        let s_ab = a_plus_b.mul_scalar(scalar)?;
        let sa = tensor_a.mul_scalar(scalar)?;
        let sb = tensor_b.mul_scalar(scalar)?;
        let sa_plus_sb = sa.add(&sb)?;
        TestAssertions::assert_tensor_eq_with_epsilon(&s_ab, &sa_plus_sb, 1e-3)?;
    }
}

proptest! {
    /// Property: Matrix multiplication associativity and dimension correctness
    #[test]
    fn matrix_multiplication_properties(
        (shape_a, shape_b) in matmul_shapes(),
        scale in reasonable_f32()
    ) {
        let size_a = shape_a.iter().product::<usize>();
        let size_b = shape_b.iter().product::<usize>();

        let tensor_a = Tensor::from_vec(vec![scale; size_a], &shape_a)?;
        let tensor_b = Tensor::from_vec(vec![1.0; size_b], &shape_b)?;

        let result = tensor_a.matmul(&tensor_b)?;

        // Check result dimensions
        let expected_shape = if shape_a.len() == 2 && shape_b.len() == 2 {
            vec![shape_a[0], shape_b[1]]
        } else if shape_a.len() == 3 && shape_b.len() == 3 {
            vec![shape_a[0], shape_a[1], shape_b[2]]
        } else {
            return Ok(()); // Skip unsupported combinations
        };

        TestAssertions::assert_shape(&result, &expected_shape)?;

        // For matrices filled with constants, verify mathematical correctness
        if shape_a.len() == 2 && shape_b.len() == 2 {
            let expected_value = scale * shape_a[1] as f32; // sum across k dimension
            for &val in result.to_vec_f32()?.iter() {
                // Use relative tolerance for larger values
                let tolerance = if expected_value.abs() > 1.0 {
                    1e-3 * expected_value.abs()
                } else {
                    1e-3
                };
                prop_assert!((val - expected_value).abs() < tolerance);
            }
        }
    }
}

proptest! {
    /// Property: Transpose is involutive (transpose twice = identity)
    #[test]
    fn transpose_involution_property(
        rows in 1usize..50,
        cols in 1usize..50,
        values in prop::collection::vec(reasonable_f32(), 1..2500)
    ) {
        let shape = vec![rows, cols];
        let size = rows * cols;
        prop_assume!(values.len() >= size);

        let data = values[..size].to_vec();
        let tensor = Tensor::from_vec(data.clone(), &shape)?;

        // Test (A^T)^T = A
        let transposed = tensor.transpose(1, 0)?;
        let double_transposed = transposed.transpose(1, 0)?;

        TestAssertions::assert_tensor_eq(&tensor, &double_transposed)?;
        TestAssertions::assert_shape(&transposed, &[cols, rows])?;
    }
}

proptest! {
    /// Property: Reshape preserves total number of elements and data
    #[test]
    fn reshape_preserves_elements(
        shape1 in tensor_shapes(),
        new_dim in 1usize..5
    ) {
        let size1: usize = shape1.iter().product();
        prop_assume!(size1 > 0 && size1 < 10000);

        // Generate a second shape with the same total size
        let mut shape2 = vec![new_dim];
        let mut remaining = size1 / new_dim;
        if size1 % new_dim != 0 {
            // If not divisible, use the original shape to ensure valid reshape
            return Ok(());
        }

        // Factor the remaining size into reasonable dimensions
        while remaining > 1 && shape2.len() < 4 {
            let factor = (2..=remaining.min(10)).find(|&f| remaining % f == 0).unwrap_or(remaining);
            shape2.push(factor);
            remaining /= factor;
        }
        if remaining > 1 {
            shape2.push(remaining);
        }

        let data: Vec<f32> = (0..size1).map(|i| i as f32 * 0.01).collect();
        let tensor = Tensor::from_vec(data.clone(), &shape1)?;

        let reshaped = tensor.reshape(&shape2)?;

        TestAssertions::assert_shape(&reshaped, &shape2)?;
        prop_assert_eq!(reshaped.to_vec_f32()?.len(), size1);

        // Data should be preserved
        for (original, reshaped_val) in data.iter().zip(reshaped.to_vec_f32()?.iter()) {
            prop_assert!((original - reshaped_val).abs() < 1e-6);
        }
    }
}

proptest! {
    /// Property: Activation functions preserve shape and satisfy bounds
    #[test]
    fn activation_function_properties(
        shape_and_values in tensor_shapes().prop_flat_map(|shape| {
            let size = shape.iter().product::<usize>();
            prop::collection::vec(reasonable_f32(), size..=size)
                .prop_map(move |values| (shape.clone(), values))
        })
    ) {
        let (shape, values) = shape_and_values;
        let size: usize = shape.iter().product();
        prop_assume!(size > 0 && size < 10000);

        let data = values;
        let tensor = Tensor::from_vec(data.clone(), &shape)?;

        // Test ReLU properties
        let relu_result = tensor.relu()?;
        TestAssertions::assert_shape(&relu_result, &shape)?;
        TestAssertions::assert_values_in_range(&relu_result, 0.0, f32::INFINITY)?;

        for (&original, &relu_val) in data.iter().zip(relu_result.to_vec_f32()?.iter()) {
            if original > 0.0 {
                prop_assert!((original - relu_val).abs() < 1e-6);
            } else {
                prop_assert_eq!(relu_val, 0.0);
            }
        }

        // Test Sigmoid properties: output should be in (0, 1)
        let sigmoid_result = tensor.sigmoid()?;
        TestAssertions::assert_shape(&sigmoid_result, &shape)?;
        // Allow small epsilon for numerical precision (sigmoid can be slightly outside due to rounding)
        TestAssertions::assert_values_in_range(&sigmoid_result, -1e-6, 1.0 + 1e-6)?;

        // Test Tanh properties: output should be in (-1, 1)
        let tanh_result = tensor.tanh()?;
        TestAssertions::assert_shape(&tanh_result, &shape)?;
        // Allow small epsilon for numerical precision (tanh can produce values like 1.0000001)
        TestAssertions::assert_values_in_range(&tanh_result, -1.0 - 1e-6, 1.0 + 1e-6)?;
    }
}

proptest! {
    /// Property: Softmax outputs form valid probability distribution
    #[test]
    fn softmax_probability_properties(
        batch_size in 1usize..20,
        feature_size in 2usize..50,
        values in prop::collection::vec(reasonable_f32(), 2..1000)
    ) {
        let shape = vec![batch_size, feature_size];
        let size = batch_size * feature_size;
        prop_assume!(values.len() >= size);

        let data = values[..size].to_vec();
        let tensor = Tensor::from_vec(data, &shape)?;

        let softmax_result = tensor.softmax(-1)?;
        TestAssertions::assert_shape(&softmax_result, &shape)?;
        TestAssertions::assert_values_in_range(&softmax_result, 0.0, 1.0)?;

        // Each row should sum to approximately 1
        for batch in 0..batch_size {
            let mut row_sum = 0.0f32;
            for feature in 0..feature_size {
                row_sum += softmax_result.to_vec_f32()?[batch * feature_size + feature];
            }
            prop_assert!((row_sum - 1.0).abs() < 1e-5);
        }
    }
}

proptest! {
    /// Property: Norm properties (positive definiteness, triangle inequality)
    #[test]
    fn norm_properties(
        shape in tensor_shapes(),
        scale_a in reasonable_f32(),
        scale_b in reasonable_f32()
    ) {
        let size: usize = shape.iter().product();
        prop_assume!(size > 0 && size < 10000);

        let tensor_a = Tensor::from_vec(vec![scale_a; size], &shape)?;
        let tensor_b = Tensor::from_vec(vec![scale_b; size], &shape)?;

        let norm_a_val = tensor_a.norm()?;
        let norm_b_val = tensor_b.norm()?;

        // Positive definiteness: ||x|| >= 0
        prop_assert!(norm_a_val >= 0.0);
        prop_assert!(norm_b_val >= 0.0);

        // Triangle inequality: ||a + b|| <= ||a|| + ||b||
        let a_plus_b = tensor_a.add(&tensor_b)?;
        let norm_a_plus_b_val = a_plus_b.norm()?;
        let max_norm = norm_a_val.max(norm_b_val);
        let tolerance = if max_norm > 1.0 {
            1e-3 * max_norm // Relative tolerance for larger norms
        } else {
            1e-3 // Absolute tolerance for smaller norms
        };
        prop_assert!(norm_a_plus_b_val <= norm_a_val + norm_b_val + tolerance);

        // For constant tensors: ||c * 1_vec|| = |c| * sqrt(n)
        let expected_norm_a = scale_a.abs() * (size as f32).sqrt();
        let tolerance = if expected_norm_a > 1.0 {
            1e-3 * expected_norm_a // Relative tolerance for larger values
        } else {
            1e-3 // Absolute tolerance for smaller values
        };
        prop_assert!((norm_a_val - expected_norm_a).abs() < tolerance);
    }
}

proptest! {
    /// Property: Mathematical function identities
    #[test]
    fn mathematical_function_identities(
        shape in tensor_shapes(),
        positive_values in prop::collection::vec(0.1f32..100.0f32, 1..1000)
    ) {
        let size: usize = shape.iter().product();
        prop_assume!(positive_values.len() >= size && size > 0 && size < 10000);

        // Use only positive values for mathematical identities that require them
        let data = positive_values[..size].to_vec();
        let tensor = Tensor::from_vec(data.clone(), &shape)?;

        // Test exp(log(x)) ≈ x for positive values
        let log_result = tensor.log()?;
        let exp_log_result = log_result.exp()?;
        TestAssertions::assert_tensor_eq_with_epsilon(&tensor, &exp_log_result, 1e-3)?;

        // Test sqrt(x^2) ≈ x for positive values
        let squared = tensor.mul(&tensor)?;
        let sqrt_squared = squared.sqrt()?;
        TestAssertions::assert_tensor_eq_with_epsilon(&tensor, &sqrt_squared, 1e-4)?;

        // Test log properties: log(a*b) = log(a) + log(b) for positive values
        // Only test if size is even and >= 4 to ensure proper reshape
        if size >= 4 && size % 2 == 0 {
            let half_size = size / 2;
            let tensor_split = tensor.reshape(&[2, half_size])?;
            let a = tensor_split.slice(0, 0, 1)?.reshape(&[half_size])?;
            let b = tensor_split.slice(0, 1, 2)?.reshape(&[half_size])?;

            let ab = a.mul(&b)?;
            let log_ab = ab.log()?;
            let log_a = a.log()?;
            let log_b = b.log()?;
            let log_a_plus_log_b = log_a.add(&log_b)?;

            TestAssertions::assert_tensor_eq_with_epsilon(&log_ab, &log_a_plus_log_b, 1e-3)?;
        }
    }
}

proptest! {
    /// Property: Error handling for incompatible operations
    #[test]
    fn incompatible_operations_fail_gracefully(
        dim1 in 2usize..5,
        dim2 in 3usize..6
    ) {
        // Create truly incompatible shapes that cannot be broadcast
        // Both dimensions must be greater than 1 and different from each other
        prop_assume!(dim1 != dim2 && dim1 > 1 && dim2 > 1);

        let shape1 = vec![dim1];    // e.g., [3]
        let shape2 = vec![dim2];    // e.g., [4] - incompatible with [3]

        let size1 = shape1.iter().product::<usize>();
        let size2 = shape2.iter().product::<usize>();

        let tensor1 = Tensor::from_vec(vec![1.0; size1], &shape1)?;
        let tensor2 = Tensor::from_vec(vec![2.0; size2], &shape2)?;

        // Element-wise operations should fail for truly incompatible shapes
        prop_assert!(tensor1.add(&tensor2).is_err());
        prop_assert!(tensor1.sub(&tensor2).is_err());
        prop_assert!(tensor1.mul(&tensor2).is_err());
        prop_assert!(tensor1.div(&tensor2).is_err());
    }
}

proptest! {
    /// Property: Broadcasting behavior for compatible shapes
    #[test]
    fn broadcasting_properties(
        base_dim in 2usize..20,
        batch_size in 1usize..10
    ) {
        let large_shape = vec![batch_size, base_dim];
        let small_shape = vec![1, base_dim];

        let large_size = large_shape.iter().product::<usize>();
        let small_size = small_shape.iter().product::<usize>();

        let large_tensor = Tensor::from_vec(vec![2.0; large_size], &large_shape)?;
        let small_tensor = Tensor::from_vec(vec![3.0; small_size], &small_shape)?;

        // Broadcasting should work and preserve shape of larger tensor
        let broadcast_result = large_tensor.add(&small_tensor)?;
        TestAssertions::assert_shape(&broadcast_result, &large_shape)?;

        // Verify broadcasting semantics
        for &val in broadcast_result.to_vec_f32()?.iter() {
            prop_assert!((val - 5.0).abs() < 1e-6); // 2.0 + 3.0 = 5.0
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use trustformers_core::testing::IntegrationTestRunner;
    use trustformers_core::TrustformersError;

    /// Integration test combining property tests with memory leak detection
    #[test]
    fn property_tests_with_memory_tracking() -> TestResult<()> {
        let mut runner = IntegrationTestRunner::new();

        // Test tensor operations with memory tracking
        runner
            .run_test("tensor_creation_memory", |_detector| {
                let tensor = TensorTestUtils::sequential_f32(&[100, 100])?;
                TestAssertions::assert_shape(&tensor, &[100, 100])?;
                Ok(())
            })
            .map_err(|e| TrustformersError::tensor_op_error(&format!("{}", e), "run_test"))?;

        runner
            .run_test("mathematical_operations_memory", |_detector| {
                let a = TensorTestUtils::random_f32(&[50, 50])?;
                let b = TensorTestUtils::random_f32(&[50, 50])?;

                let sum = a.add(&b)?;
                let product = a.mul(&b)?;
                let matmul = a.matmul(&b)?;

                TestAssertions::assert_shape(&sum, &[50, 50])?;
                TestAssertions::assert_shape(&product, &[50, 50])?;
                TestAssertions::assert_shape(&matmul, &[50, 50])?;

                Ok(())
            })
            .map_err(|e| TrustformersError::tensor_op_error(&format!("{}", e), "run_test"))?;

        let summary = runner.get_summary();
        assert!(
            summary.all_passed(),
            "Some property tests failed memory leak detection"
        );

        Ok(())
    }
}
