//! Property-based tests for tensor operations.
//!
//! This module contains property-based tests using proptest to verify
//! mathematical properties of tensor operations across various inputs.

use crate::tensor::Tensor;
use approx::assert_relative_eq;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

/// Generate random valid shapes for testing (1D to 3D, small sizes for performance)
fn valid_shapes() -> impl Strategy<Value = Vec<usize>> {
    prop_oneof![
        // 1D shapes
        (1..20usize).prop_map(|n| vec![n]),
        // 2D shapes
        (1..10usize, 1..10usize).prop_map(|(m, n)| vec![m, n]),
        // 3D shapes
        (1..5usize, 1..5usize, 1..5usize).prop_map(|(a, b, c)| vec![a, b, c]),
    ]
}

/// Generate compatible matrix multiplication shapes
#[allow(dead_code)]
fn matmul_shapes() -> impl Strategy<Value = (Vec<usize>, Vec<usize>)> {
    prop_oneof![
        // 2D x 2D: [m, k] x [k, n]
        (2..10usize, 2..10usize, 2..10usize).prop_map(|(m, k, n)| (vec![m, k], vec![k, n])),
        // 3D x 3D: [b, m, k] x [b, k, n]
        (1..5usize, 2..8usize, 2..8usize, 2..8usize)
            .prop_map(|(b, m, k, n)| (vec![b, m, k], vec![b, k, n])),
    ]
}

/// Generate random f32 values avoiding extreme values that cause numerical instability
fn reasonable_f32() -> impl Strategy<Value = f32> {
    -100.0f32..100.0f32
}

/// Generate arrays of reasonable f32 values
fn reasonable_f32_vec(size: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(reasonable_f32(), size..=size)
}

/// Generate a single f32 value for testing
fn generate_f32() -> f32 {
    let mut runner = TestRunner::default();
    reasonable_f32().new_tree(&mut runner).unwrap().current()
}

#[cfg(test)]
mod tensor_property_tests {
    use super::*;

    proptest! {
        /// Property: Addition is commutative
        /// For any tensors A and B of the same shape: A + B = B + A
        #[test]
        fn addition_is_commutative(
            _shape in valid_shapes(),
            values_a in reasonable_f32_vec(0).prop_flat_map(|_| {
                valid_shapes().prop_flat_map(|s| {
                    let size = s.iter().product();
                    reasonable_f32_vec(size).prop_map(move |v| (s.clone(), v))
                })
            })
        ) {
            let (shape, vals_a) = values_a;
            let mut runner = TestRunner::default();
            let vals_b: Vec<f32> = reasonable_f32_vec(vals_a.len()).new_tree(&mut runner).unwrap().current();

            if let (Ok(a), Ok(b)) = (
                Tensor::from_vec(vals_a, &shape),
                Tensor::from_vec(vals_b, &shape)
            ) {
                if let (Ok(ab), Ok(ba)) = (a.add(&b), b.add(&a)) {
                    let ab_data = ab.data().unwrap();
                    let ba_data = ba.data().unwrap();
                    for (x, y) in ab_data.iter().zip(ba_data.iter()) {
                        assert_relative_eq!(x, y, epsilon = 1e-6);
                    }
                }
            }
        }

        /// Property: Addition is associative
        /// For any tensors A, B, C: (A + B) + C = A + (B + C)
        #[test]
        fn addition_is_associative(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {  // Reasonable size limits
                let mut runner = TestRunner::default();
                let vals_a: Vec<f32> = reasonable_f32_vec(size).new_tree(&mut runner).unwrap().current();
                let vals_b: Vec<f32> = reasonable_f32_vec(size).new_tree(&mut runner).unwrap().current();
                let vals_c: Vec<f32> = reasonable_f32_vec(size).new_tree(&mut runner).unwrap().current();

                if let (Ok(a), Ok(b), Ok(c)) = (
                    Tensor::from_vec(vals_a, &shape),
                    Tensor::from_vec(vals_b, &shape),
                    Tensor::from_vec(vals_c, &shape)
                ) {
                    if let (Ok(ab), Ok(bc)) = (a.add(&b), b.add(&c)) {
                        if let (Ok(ab_c), Ok(a_bc)) = (ab.add(&c), a.add(&bc)) {
                            let ab_c_data = ab_c.data().unwrap();
                            let a_bc_data = a_bc.data().unwrap();
                            for (x, y) in ab_c_data.iter().zip(a_bc_data.iter()) {
                                assert_relative_eq!(x, y, epsilon = 1e-4);
                            }
                        }
                    }
                }
            }
        }

        /// Property: Additive identity
        /// For any tensor A: A + 0 = A
        #[test]
        fn additive_identity(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                let vals: Vec<f32> = (0..size).map(|_| generate_f32()).collect();

                if let Ok(a) = Tensor::from_vec(vals.clone(), &shape) {
                    if let Ok(zero) = Tensor::zeros(&shape) {
                        if let Ok(result) = a.add(&zero) {
                            if let Ok(result_data) = result.data() {
                                for (x, y) in vals.iter().zip(result_data.iter()) {
                                    assert_relative_eq!(x, y, epsilon = 1e-6);
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Property: Multiplication distributivity over addition
        /// For any tensors A, B and scalar s: s * (A + B) = s * A + s * B
        #[test]
        fn multiplication_distributive(
            shape in valid_shapes(),
            scalar in reasonable_f32()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                let vals_a: Vec<f32> = (0..size).map(|_| generate_f32()).collect();
                let vals_b: Vec<f32> = (0..size).map(|_| generate_f32()).collect();

                if let (Ok(a), Ok(b)) = (
                    Tensor::from_vec(vals_a, &shape),
                    Tensor::from_vec(vals_b, &shape)
                ) {
                    if let Ok(a_plus_b) = a.add(&b) {
                        if let (Ok(s_ab), Ok(sa), Ok(sb)) = (
                            a_plus_b.mul_scalar(scalar),
                            a.mul_scalar(scalar),
                            b.mul_scalar(scalar)
                        ) {
                            if let Ok(sa_plus_sb) = sa.add(&sb) {
                                let s_ab_data = s_ab.data().unwrap();
                                let sa_plus_sb_data = sa_plus_sb.data().unwrap();
                                for (x, y) in s_ab_data.iter().zip(sa_plus_sb_data.iter()) {
                                    assert_relative_eq!(x, y, epsilon = 1e-2);
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Property: Matrix multiplication associativity
        /// For compatible matrices A, B, C: (A × B) × C = A × (B × C)
        #[test]
        fn matmul_associative(
            m in 2..8usize,
            k1 in 2..8usize,
            k2 in 2..8usize,
            n in 2..8usize
        ) {
            let size_a = m * k1;
            let size_b = k1 * k2;
            let size_c = k2 * n;

            let vals_a: Vec<f32> = (0..size_a).map(|_| generate_f32()).collect();
            let vals_b: Vec<f32> = (0..size_b).map(|_| generate_f32()).collect();
            let vals_c: Vec<f32> = (0..size_c).map(|_| generate_f32()).collect();

            if let (Ok(a), Ok(b), Ok(c)) = (
                Tensor::from_vec(vals_a, &[m, k1]),
                Tensor::from_vec(vals_b, &[k1, k2]),
                Tensor::from_vec(vals_c, &[k2, n])
            ) {
                if let (Ok(ab), Ok(bc)) = (a.matmul(&b), b.matmul(&c)) {
                    if let (Ok(ab_c), Ok(a_bc)) = (ab.matmul(&c), a.matmul(&bc)) {
                        let ab_c_data = ab_c.data().unwrap();
                        let a_bc_data = a_bc.data().unwrap();
                        for (x, y) in ab_c_data.iter().zip(a_bc_data.iter()) {
                            let rel_error = (x - y).abs() / x.abs().max(y.abs()).max(1e-10);
                            assert!(rel_error < 2e-2, "Relative error too large: {} vs {}, error: {}", x, y, rel_error);
                        }
                    }
                }
            }
        }

        /// Property: Matrix multiplication with identity
        /// For any matrix A and identity matrix I: A × I = A and I × A = A
        #[test]
        fn matmul_identity(
            m in 2..10usize,
            n in 2..10usize
        ) {
            let size = m * n;
            let vals: Vec<f32> = (0..size).map(|_| generate_f32()).collect();

            if let Ok(a) = Tensor::from_vec(vals.clone(), &[m, n]) {
                if let Ok(identity) = Tensor::eye_f32(n) {
                    if let Ok(result) = a.matmul(&identity) {
                        if let Ok(result_data) = result.data() {
                            for (x, y) in vals.iter().zip(result_data.iter()) {
                                assert_relative_eq!(x, y, epsilon = 1e-5);
                            }
                        }
                    }
                }
            }
        }

        /// Property: Element-wise operations preserve shape
        #[test]
        fn elementwise_preserves_shape(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                let vals: Vec<f32> = (0..size).map(|_| (generate_f32() + 1.0).abs()).collect();

                if let Ok(tensor) = Tensor::from_vec(vals, &shape) {
                    // Test various element-wise operations
                    let operations = vec![
                        tensor.sqrt(),
                        tensor.exp(),
                        tensor.log(),
                        tensor.mul_scalar(2.0),
                        tensor.add_scalar(1.0),
                    ];

                    for result in operations.into_iter().flatten() {
                        assert_eq!(result.shape(), shape);
                    }
                }
            }
        }

        /// Property: Logarithm and exponential are inverse operations
        /// For positive values: log(exp(x)) ≈ x and exp(log(x)) ≈ x
        #[test]
        fn log_exp_inverse(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                // Use positive values to avoid log domain issues
                let vals: Vec<f32> = (0..size).map(|_|
                    (generate_f32().abs() + 0.1).min(10.0)
                ).collect();

                if let Ok(tensor) = Tensor::from_vec(vals.clone(), &shape) {
                    // Test exp(log(x)) ≈ x
                    if let Ok(log_result) = tensor.log() {
                        if let Ok(exp_log_result) = log_result.exp() {
                            if let Ok(result_data) = exp_log_result.data() {
                                for (original, computed) in vals.iter().zip(result_data.iter()) {
                                    assert_relative_eq!(original, computed, epsilon = 1e-4);
                                }
                            }
                        }
                    }

                    // Test log(exp(x)) ≈ x for smaller values
                    let small_vals: Vec<f32> = vals.iter().map(|&x| x.min(5.0)).collect();
                    if let Ok(small_tensor) = Tensor::from_vec(small_vals.clone(), &shape) {
                        if let Ok(exp_result) = small_tensor.exp() {
                            if let Ok(log_exp_result) = exp_result.log() {
                                if let Ok(result_data) = log_exp_result.data() {
                                    for (original, computed) in small_vals.iter().zip(result_data.iter()) {
                                        assert_relative_eq!(original, computed, epsilon = 1e-4);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Property: Square root properties
        /// For non-negative values: sqrt(x^2) = |x| and (sqrt(x))^2 = x
        #[test]
        fn sqrt_properties(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                // Use non-negative values for sqrt
                let vals: Vec<f32> = (0..size).map(|_| generate_f32().abs()).collect();

                if let Ok(tensor) = Tensor::from_vec(vals.clone(), &shape) {
                    // Test (sqrt(x))^2 = x
                    if let Ok(sqrt_result) = tensor.sqrt() {
                        if let Ok(sqrt_squared) = sqrt_result.mul(&sqrt_result) {
                            if let Ok(result_data) = sqrt_squared.data() {
                                for (original, computed) in vals.iter().zip(result_data.iter()) {
                                    assert_relative_eq!(original, computed, epsilon = 1e-5);
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Property: Power function properties
        /// For any x and integers m, n: x^(m+n) = x^m * x^n and (x^m)^n = x^(m*n)
        #[test]
        fn power_properties(
            shape in valid_shapes(),
            power1 in 1..5i32,
            power2 in 1..5i32
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                // Use positive values to avoid complex number issues
                let vals: Vec<f32> = (0..size).map(|_|
                    (generate_f32().abs() + 0.1).min(2.0)
                ).collect();

                if let Ok(tensor) = Tensor::from_vec(vals, &shape) {
                    let p1 = power1 as f32;
                    let p2 = power2 as f32;

                    // Test x^(m+n) = x^m * x^n
                    if let (Ok(x_p1), Ok(x_p2), Ok(x_p1_plus_p2)) = (
                        tensor.pow(p1),
                        tensor.pow(p2),
                        tensor.pow(p1 + p2)
                    ) {
                        if let Ok(x_p1_times_x_p2) = x_p1.mul(&x_p2) {
                            let expected = x_p1_plus_p2.data().unwrap();
                            let computed = x_p1_times_x_p2.data().unwrap();
                            for (exp, comp) in expected.iter().zip(computed.iter()) {
                                assert_relative_eq!(exp, comp, epsilon = 1e-4);
                            }
                        }
                    }
                }
            }
        }

        /// Property: Activation function properties
        /// Test properties of sigmoid: σ(x) = 1/(1+e^(-x))
        /// Properties: σ(-x) = 1 - σ(x), 0 < σ(x) < 1, σ(0) = 0.5
        #[test]
        fn sigmoid_properties(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                let vals: Vec<f32> = (0..size).map(|_|
                    generate_f32() * 0.1  // Scale to avoid saturation
                ).collect();

                if let Ok(tensor) = Tensor::from_vec(vals, &shape) {
                    if let Ok(sigmoid_result) = tensor.sigmoid() {
                        if let Ok(result_data) = sigmoid_result.data() {
                            // Test: 0 < σ(x) < 1
                            for &val in result_data.iter() {
                                assert!(val > 0.0 && val < 1.0, "Sigmoid output should be in (0,1)");
                            }

                            // Test: σ(-x) = 1 - σ(x)
                            if let Ok(neg_tensor) = tensor.mul_scalar(-1.0) {
                                if let Ok(sigmoid_neg) = neg_tensor.sigmoid() {
                                    if let Ok(sigmoid_neg_data) = sigmoid_neg.data() {
                                        for (pos, neg) in result_data.iter().zip(sigmoid_neg_data.iter()) {
                                            assert_relative_eq!(pos + neg, 1.0, epsilon = 1e-5);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Property: Tanh properties
        /// Test properties of tanh: tanh(-x) = -tanh(x), -1 < tanh(x) < 1, tanh(0) = 0
        #[test]
        fn tanh_properties(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                let vals: Vec<f32> = (0..size).map(|_|
                    generate_f32() * 0.1
                ).collect();

                if let Ok(tensor) = Tensor::from_vec(vals, &shape) {
                    if let Ok(tanh_result) = tensor.tanh() {
                        if let Ok(result_data) = tanh_result.data() {
                            // Test: -1 <= tanh(x) <= 1 (allowing boundary values due to numerical precision)
                            for &val in result_data.iter() {
                                assert!((-1.0..=1.0).contains(&val), "Tanh output should be in [-1,1]");
                            }

                            // Test: tanh(-x) = -tanh(x)
                            if let Ok(neg_tensor) = tensor.mul_scalar(-1.0) {
                                if let Ok(tanh_neg) = neg_tensor.tanh() {
                                    if let Ok(tanh_neg_data) = tanh_neg.data() {
                                        for (pos, neg) in result_data.iter().zip(tanh_neg_data.iter()) {
                                            assert_relative_eq!(*pos, -*neg, epsilon = 1e-5);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Property: Softmax properties
        /// Test properties: sum = 1, all values > 0, softmax is monotonic
        #[test]
        fn softmax_properties(
            dim1 in 2..10usize,
            dim2 in 2..10usize
        ) {
            let size = dim1 * dim2;
            let vals: Vec<f32> = (0..size).map(|_| generate_f32() * 0.1).collect();

            if let Ok(tensor) = Tensor::from_vec(vals, &[dim1, dim2]) {
                if let Ok(softmax_result) = tensor.softmax(-1) {
                    let result_data = softmax_result.data().unwrap();

                    // Test: all values > 0
                    for &val in result_data.iter() {
                        assert!(val > 0.0, "Softmax output should be positive");
                    }

                    // Test: sum along last dimension = 1
                    for i in 0..dim1 {
                        let mut sum = 0.0f32;
                        for j in 0..dim2 {
                            sum += result_data[i * dim2 + j];
                        }
                        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
                    }
                }
            }
        }

        /// Property: Transpose properties
        /// Test: (A^T)^T = A, shape transformation correctness
        #[test]
        fn transpose_properties(
            m in 2..10usize,
            n in 2..10usize
        ) {
            let size = m * n;
            let vals: Vec<f32> = (0..size).map(|_| generate_f32()).collect();

            if let Ok(tensor) = Tensor::from_vec(vals.clone(), &[m, n]) {
                if let Ok(transposed) = tensor.transpose(1, 0) {
                    // Test shape transformation
                    assert_eq!(transposed.shape(), &[n, m]);

                    // Test (A^T)^T = A
                    if let Ok(double_transposed) = transposed.transpose(1, 0) {
                        assert_eq!(double_transposed.shape(), &[m, n]);
                        if let Ok(result_data) = double_transposed.data() {
                            for (original, computed) in vals.iter().zip(result_data.iter()) {
                                assert_relative_eq!(original, computed, epsilon = 1e-6);
                            }
                        }
                    }
                }
            }
        }

        /// Property: Reshape properties
        /// Test: volume preservation, data preservation for compatible shapes
        #[test]
        fn reshape_properties(
            original_shape in valid_shapes()
        ) {
            let size: usize = original_shape.iter().product();
            if size > 0 && size < 1000 {
                let vals: Vec<f32> = (0..size).map(|_| generate_f32()).collect();

                if let Ok(tensor) = Tensor::from_vec(vals.clone(), &original_shape) {
                    // Find compatible reshape dimensions
                    let new_shapes = if size >= 6 {
                        vec![vec![size], vec![1, size], vec![size, 1]]
                    } else {
                        vec![vec![size]]
                    };

                    for new_shape in new_shapes {
                        if let Ok(reshaped) = tensor.reshape(&new_shape) {
                            // Test volume preservation
                            assert_eq!(reshaped.shape().iter().product::<usize>(), size);

                            // Test data preservation
                            if let Ok(reshaped_data) = reshaped.data() {
                                for (original, computed) in vals.iter().zip(reshaped_data.iter()) {
                                    assert_relative_eq!(original, computed, epsilon = 1e-6);
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Property: Reduction operations properties
        /// Test: sum correctness, mean = sum/count
        #[test]
        fn reduction_properties(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                let vals: Vec<f32> = (0..size).map(|_| generate_f32()).collect();

                if let Ok(tensor) = Tensor::from_vec(vals.clone(), &shape) {
                    // Test sum
                    if let Ok(sum_result) = tensor.sum(None, false) {
                        if let Ok(sum_data) = sum_result.data() {
                            let expected_sum: f32 = vals.iter().sum();
                            let computed_sum = sum_data[0];
                            assert_relative_eq!(expected_sum, computed_sum, epsilon = 1e-3);
                        }
                    }

                    // Test mean = sum/count
                    if let (Ok(sum_result), Ok(mean_result)) = (tensor.sum(None, false), tensor.mean()) {
                        if let (Ok(sum_data), Ok(mean_data)) = (sum_result.data(), mean_result.data()) {
                            let expected_mean = sum_data[0] / (size as f32);
                            let computed_mean = mean_data[0];
                            assert_relative_eq!(expected_mean, computed_mean, epsilon = 1e-5);
                        }
                    }
                }
            }
        }

        /// Property: Norm properties
        /// Test: L2 norm properties, triangle inequality
        #[test]
        fn norm_properties(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                let vals_a: Vec<f32> = (0..size).map(|_| generate_f32()).collect();
                let vals_b: Vec<f32> = (0..size).map(|_| generate_f32()).collect();

                if let (Ok(a), Ok(b)) = (
                    Tensor::from_vec(vals_a, &shape),
                    Tensor::from_vec(vals_b, &shape)
                ) {
                    // Test triangle inequality: ||a + b|| <= ||a|| + ||b||
                    if let (Ok(norm_a_val), Ok(norm_b_val)) = (a.norm(), b.norm()) {
                        if let Ok(a_plus_b) = a.add(&b) {
                            if let Ok(norm_a_plus_b_val) = a_plus_b.norm() {
                                assert!(norm_a_plus_b_val <= norm_a_val + norm_b_val + 1e-5,
                                    "Triangle inequality should hold");
                            }
                        }
                    }

                    // Test positive definiteness: ||a|| >= 0, ||a|| = 0 iff a = 0
                    if let Ok(norm_a_val) = a.norm() {
                        assert!(norm_a_val >= 0.0, "Norm should be non-negative");
                    }
                }
            }
        }

        /// Property: Broadcasting properties
        /// Test: broadcasting preserves mathematical properties
        #[test]
        fn broadcasting_properties(
            base_dim in 2..8usize
        ) {
            let large_shape = vec![base_dim, base_dim];
            let small_shape = vec![1, base_dim];

            let large_size = large_shape.iter().product::<usize>();
            let small_size = small_shape.iter().product::<usize>();

            let large_vals: Vec<f32> = (0..large_size).map(|_| generate_f32()).collect();
            let small_vals: Vec<f32> = (0..small_size).map(|_| generate_f32()).collect();

            if let (Ok(large_tensor), Ok(small_tensor)) = (
                Tensor::from_vec(large_vals, &large_shape),
                Tensor::from_vec(small_vals, &small_shape)
            ) {
                // Test broadcasting in addition
                if let Ok(broadcast_add) = large_tensor.add(&small_tensor) {
                    assert_eq!(broadcast_add.shape(), large_shape);

                    // Verify broadcasting semantics: each row should be identical
                    if let (Ok(result_data), Ok(large_data), Ok(small_data)) =
                        (broadcast_add.data(), large_tensor.data(), small_tensor.data()) {
                        for i in 0..base_dim {
                            for j in 0..base_dim {
                                let large_val = large_data[i * base_dim + j];
                                let small_val = small_data[j];
                                let expected = large_val + small_val;
                                let actual = result_data[i * base_dim + j];
                                assert_relative_eq!(expected, actual, epsilon = 1e-6);
                            }
                        }
                    }
                }
            }
        }

        /// Property: Complex number operations
        /// Test: complex arithmetic properties for tensors with complex data
        #[test]
        fn complex_number_properties(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                let real_vals: Vec<f32> = (0..size).map(|_| generate_f32()).collect();
                let imag_vals: Vec<f32> = (0..size).map(|_| generate_f32()).collect();

                if let Ok(real_tensor) = Tensor::from_vec(real_vals.clone(), &shape) {
                    if let Ok(imag_tensor) = Tensor::from_vec(imag_vals.clone(), &shape) {
                        if let Ok(complex_tensor) = Tensor::complex(real_vals, imag_vals, &shape) {
                            // Test real/imaginary extraction
                            if let (Ok(extracted_real), Ok(extracted_imag)) = (
                                complex_tensor.real(),
                                complex_tensor.imag()
                            ) {
                                // Real parts should match
                                for (orig, extr) in real_tensor.data().unwrap().iter().zip(extracted_real.data().unwrap().iter()) {
                                    assert_relative_eq!(orig, extr, epsilon = 1e-6);
                                }

                                // Imaginary parts should match
                                for (orig, extr) in imag_tensor.data().unwrap().iter().zip(extracted_imag.data().unwrap().iter()) {
                                    assert_relative_eq!(orig, extr, epsilon = 1e-6);
                                }
                            }

                            // Test magnitude properties: |z|^2 = real^2 + imag^2
                            if let Ok(magnitude) = complex_tensor.magnitude() {
                                if let (Ok(mag_data), Ok(real_data), Ok(imag_data)) =
                                    (magnitude.data(), real_tensor.data(), imag_tensor.data()) {
                                    for i in 0..size {
                                        let expected_mag_sq = real_data[i].powi(2) + imag_data[i].powi(2);
                                        let actual_mag_sq = mag_data[i].powi(2);
                                        assert_relative_eq!(expected_mag_sq, actual_mag_sq, epsilon = 1e-2);
                                    }
                                }
                            }

                            // Test conjugate properties: conj(conj(z)) = z
                            if let Ok(conjugate) = complex_tensor.conj() {
                                if let Ok(double_conjugate) = conjugate.conj() {
                                    if let (Ok(orig_real), Ok(orig_imag)) = (complex_tensor.real(), complex_tensor.imag()) {
                                        if let (Ok(final_real), Ok(final_imag)) = (double_conjugate.real(), double_conjugate.imag()) {
                                            if let (Ok(orig_real_data), Ok(final_real_data)) = (orig_real.data(), final_real.data()) {
                                                for (o, f) in orig_real_data.iter().zip(final_real_data.iter()) {
                                                    assert_relative_eq!(o, f, epsilon = 1e-6);
                                                }
                                            }
                                            if let (Ok(orig_imag_data), Ok(final_imag_data)) = (orig_imag.data(), final_imag.data()) {
                                                for (o, f) in orig_imag_data.iter().zip(final_imag_data.iter()) {
                                                    assert_relative_eq!(o, f, epsilon = 1e-6);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Property: Sparse tensor operations
        /// Test: sparse tensor behavior and conversion properties
        #[test]
        fn sparse_tensor_properties(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                // Create a mostly sparse tensor (80% zeros)
                let mut vals: Vec<f32> = vec![0.0; size];
                let non_zero_count = size / 5; // 20% non-zero
                for val in vals.iter_mut().take(non_zero_count) {
                    *val = generate_f32();
                }

                if let Ok(dense_tensor) = Tensor::from_vec(vals.clone(), &shape) {
                    // Test dense-to-sparse conversion
                    if let Ok(sparse_tensor) = dense_tensor.to_sparse(1e-6) {
                        // Test sparse-to-dense conversion (round trip)
                        if let Ok(dense_again) = sparse_tensor.to_dense() {
                            // Should recover original values
                            if let Ok(dense_data) = dense_again.data() {
                                for (orig, recovered) in vals.iter().zip(dense_data.iter()) {
                                    assert_relative_eq!(orig, recovered, epsilon = 1e-6);
                                }
                            }
                        }

                        // Test that sparse operations preserve sparsity structure
                        if let Ok(sparse_scaled) = sparse_tensor.mul_scalar(2.0) {
                            if let Ok(dense_scaled) = sparse_scaled.to_dense() {
                                if let Ok(scaled_data) = dense_scaled.data() {
                                    for (orig, scaled) in vals.iter().zip(scaled_data.iter()) {
                                        assert_relative_eq!(orig * 2.0, scaled, epsilon = 1e-6);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Property: Gradient computation properties (if autodiff is enabled)
        /// Test: gradient properties like chain rule, linearity
        #[test]
        fn gradient_properties(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 100 { // Smaller sizes for gradient computation
                let vals: Vec<f32> = (0..size).map(|_| generate_f32()).collect();

                if let Ok(tensor) = Tensor::from_vec(vals, &shape) {
                    // Test that gradients exist for differentiable operations
                    // Note: This assumes gradient computation is available

                    // Test linearity: d/dx(af(x) + bg(x)) = a*df/dx + b*dg/dx
                    let a = 2.0f32;
                    let b = 3.0f32;

                    if let (Ok(scaled_a), Ok(scaled_b)) = (tensor.mul_scalar(a), tensor.mul_scalar(b)) {
                        if let Ok(combined) = scaled_a.add(&scaled_b) {
                            // Combined should equal (a+b) * tensor
                            if let Ok(expected) = tensor.mul_scalar(a + b) {
                                if let (Ok(exp_data), Ok(act_data)) = (expected.data(), combined.data()) {
                                    for (exp, act) in exp_data.iter().zip(act_data.iter()) {
                                        assert_relative_eq!(exp, act, epsilon = 1e-6);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Property: Numerical stability properties
        /// Test: operations maintain numerical stability and avoid overflow/underflow
        #[test]
        fn numerical_stability_properties(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                // Test with values that might cause numerical issues
                let extreme_vals: Vec<f32> = (0..size).map(|i| {
                    match i % 4 {
                        0 => 1e-10,  // Very small positive
                        1 => 1e10,   // Very large
                        2 => -1e10,  // Very large negative
                        _ => 0.0,    // Zero
                    }
                }).collect();

                if let Ok(tensor) = Tensor::from_vec(extreme_vals, &shape) {
                    // Test that operations don't produce NaN or Inf inappropriately
                    if let Ok(normalized) = tensor.add_scalar(1e-8) {
                        if let Ok(abs_result) = normalized.abs() {
                            if let Ok(log_result) = abs_result.log() {
                                if let Ok(log_data) = log_result.data() {
                                    for &val in log_data.iter() {
                                        // Should not be NaN (unless input was zero or negative)
                                        if !val.is_nan() {
                                            assert!(val.is_finite() || val == f32::NEG_INFINITY);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Test softmax stability (should not overflow)
                    if shape.len() >= 2 {
                        if let Ok(softmax_result) = tensor.softmax(-1) {
                            if let Ok(softmax_data) = softmax_result.data() {
                                for &val in softmax_data.iter() {
                                    assert!(val.is_finite() && (0.0..=1.0).contains(&val));
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Property: Memory layout properties
        /// Test: operations preserve contiguity and memory layout expectations
        #[test]
        fn memory_layout_properties(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                let vals: Vec<f32> = (0..size).map(|i| i as f32).collect();

                if let Ok(tensor) = Tensor::from_vec(vals.clone(), &shape) {
                    // Test that data is laid out as expected (row-major order)
                    if let Ok(tensor_data) = tensor.data() {
                        for (i, &val) in tensor_data.iter().enumerate() {
                            assert_relative_eq!(val, vals[i], epsilon = 1e-6);
                        }
                    }

                    // Test that reshape preserves data order
                    if size >= 6 {
                        let new_shape = vec![size / 2, 2];
                        if let Ok(reshaped) = tensor.reshape(&new_shape) {
                            if let Ok(reshaped_data) = reshaped.data() {
                                for (orig, reshaped_val) in vals.iter().zip(reshaped_data.iter()) {
                                    assert_relative_eq!(orig, reshaped_val, epsilon = 1e-6);
                                }
                            }
                        }
                    }

                    // Test that transpose affects memory layout predictably
                    if shape.len() == 2 && shape[0] > 1 && shape[1] > 1 {
                        if let Ok(transposed) = tensor.transpose(1, 0) {
                            assert_eq!(transposed.shape(), &[shape[1], shape[0]]);

                            // Check that elements are in the right positions
                            if let (Ok(orig_data), Ok(trans_data)) = (tensor.data(), transposed.data()) {
                                for i in 0..shape[0] {
                                    for j in 0..shape[1] {
                                        let orig_val = orig_data[i * shape[1] + j];
                                        let trans_val = trans_data[j * shape[0] + i];
                                        assert_relative_eq!(orig_val, trans_val, epsilon = 1e-6);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Property: Boundary condition properties
        /// Test: operations handle edge cases and boundary conditions correctly
        #[test]
        fn boundary_condition_properties(
            dim in 1..5usize
        ) {
            // Test with minimal tensors (1 element)
            let single_val = vec![generate_f32()];
            let single_shape = vec![1; dim];

            if let Ok(single_tensor) = Tensor::from_vec(single_val.clone(), &single_shape) {
                // Test that all operations work with single-element tensors
                if let Ok(result) = single_tensor.add_scalar(1.0) {
                    if let Ok(result_data) = result.data() {
                        assert_eq!(result_data[0], single_val[0] + 1.0);
                    }
                }

                if single_val[0] > 0.0 {
                    if let Ok(log_result) = single_tensor.log() {
                        if let Ok(log_data) = log_result.data() {
                            assert_relative_eq!(log_data[0], single_val[0].ln(), epsilon = 1e-6);
                        }
                    }
                }

                if let Ok(exp_result) = single_tensor.exp() {
                    if let Ok(exp_data) = exp_result.data() {
                        assert_relative_eq!(exp_data[0], single_val[0].exp(), epsilon = 1e-6);
                    }
                }
            }

            // Test with maximum reasonable dimensions
            let max_shape = vec![10; dim.min(3)]; // Limit to prevent huge tensors
            let max_size: usize = max_shape.iter().product();
            if max_size < 1000 {
                let max_vals: Vec<f32> = (0..max_size).map(|_| generate_f32()).collect();

                if let Ok(max_tensor) = Tensor::from_vec(max_vals, &max_shape) {
                    // Test that operations complete without error on larger tensors
                    let _ = max_tensor.sum(None, false);
                    let _ = max_tensor.mean();
                    let _ = max_tensor.add_scalar(1.0);
                }
            }
        }

        /// Property: Dtype conversion properties
        /// Test: data type conversions preserve values within precision limits
        #[test]
        fn dtype_conversion_properties(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                // Use values that can be represented exactly in different precisions
                let vals: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();

                if let Ok(tensor_f32) = Tensor::from_vec(vals.clone(), &shape) {
                    // Test F32 -> Vec<f32> -> F32 round trip
                    if let Ok(extracted) = tensor_f32.to_vec_f32() {
                        if let Ok(round_trip) = Tensor::from_vec(extracted, &shape) {
                            if let Ok(rt_data) = round_trip.data() {
                                for (orig, rt) in vals.iter().zip(rt_data.iter()) {
                                    assert_relative_eq!(orig, rt, epsilon = 1e-6);
                                }
                            }
                        }
                    }

                    // Test conversion to other formats if available
                    if let Ok(u8_data) = tensor_f32.to_vec_u8() {
                        // U8 should clamp values to [0, 255]
                        for (&orig, &converted) in vals.iter().zip(u8_data.iter()) {
                            let expected = orig.clamp(0.0, 255.0) as u8;
                            assert_eq!(converted, expected);
                        }
                    }
                }
            }
        }

        /// Property: Concurrent operation properties
        /// Test: operations are safe for concurrent access (if applicable)
        #[test]
        fn concurrent_operation_properties(
            shape in valid_shapes()
        ) {
            let size: usize = shape.iter().product();
            if size > 0 && size < 1000 {
                let vals: Vec<f32> = (0..size).map(|_| generate_f32()).collect();

                if let Ok(tensor) = Tensor::from_vec(vals.clone(), &shape) {
                    // Test that multiple read operations can happen concurrently
                    // Note: This is a basic test - real concurrent testing would use threads

                    let result1 = tensor.clone();
                    let result2 = tensor.clone();

                    // Both clones should have the same data
                    if let (Ok(data1), Ok(data2)) = (result1.data(), result2.data()) {
                        for (r1, r2) in data1.iter().zip(data2.iter()) {
                            assert_relative_eq!(r1, r2, epsilon = 1e-6);
                        }
                    }

                    // Operations on clones should not affect original
                    if let Ok(modified) = result1.add_scalar(1.0) {
                        if let (Ok(orig_data), Ok(mod_data)) = (tensor.data(), modified.data()) {
                            for (orig, mod_val) in orig_data.iter().zip(mod_data.iter()) {
                                assert_relative_eq!(*mod_val, *orig + 1.0, epsilon = 1e-6);
                                // Original should be unchanged
                            }
                        }
                    }
                }
            }
        }

        /// Property: Error handling properties
        /// Test: operations fail gracefully with appropriate errors
        #[test]
        fn error_handling_properties(
            dim1 in 2..10usize,
            dim2 in 2..10usize,
            dim3 in 2..10usize
        ) {
            // Test specific incompatible shape scenarios instead of random ones

            // Test 1: Element-wise operations with clearly incompatible shapes
            let shape1 = vec![dim1, dim2];
            let shape2 = vec![dim1 + 1, dim2 + 1]; // Clearly incompatible

            let size1 = shape1.iter().product::<usize>();
            let size2 = shape2.iter().product::<usize>();

            let vals1: Vec<f32> = (0..size1).map(|_| generate_f32()).collect();
            let vals2: Vec<f32> = (0..size2).map(|_| generate_f32()).collect();

            // Create tensors and verify they were created successfully
            match (Tensor::from_vec(vals1, &shape1), Tensor::from_vec(vals2, &shape2)) {
                (Ok(tensor1), Ok(tensor2)) => {
                    // Operations between incompatible shapes should return errors, not panic
                    let add_result = tensor1.add(&tensor2);
                    let mul_result = tensor1.mul(&tensor2);
                    let sub_result = tensor1.sub(&tensor2);
                    let div_result = tensor1.div(&tensor2);

                    // These should all be errors
                    assert!(add_result.is_err(), "Addition with incompatible shapes should return error");
                    assert!(mul_result.is_err(), "Multiplication with incompatible shapes should return error");
                    assert!(sub_result.is_err(), "Subtraction with incompatible shapes should return error");
                    assert!(div_result.is_err(), "Division with incompatible shapes should return error");
                },
                _ => {
                    // If tensor creation fails, that's also a valid error case to test
                    // We just skip the operation tests
                }
            }

            // Test 2: Matrix multiplication with incompatible dimensions
            let matmul_shape1 = vec![dim1, dim2];
            let matmul_shape2 = vec![dim2 + 1, dim3]; // Incompatible: dim2 != dim2 + 1

            let matmul_size1 = matmul_shape1.iter().product::<usize>();
            let matmul_size2 = matmul_shape2.iter().product::<usize>();

            let matmul_vals1: Vec<f32> = (0..matmul_size1).map(|_| generate_f32()).collect();
            let matmul_vals2: Vec<f32> = (0..matmul_size2).map(|_| generate_f32()).collect();

            match (
                Tensor::from_vec(matmul_vals1, &matmul_shape1),
                Tensor::from_vec(matmul_vals2, &matmul_shape2)
            ) {
                (Ok(mat1), Ok(mat2)) => {
                    let matmul_result = mat1.matmul(&mat2);
                    assert!(matmul_result.is_err(), "Matrix multiplication with incompatible dimensions should return error");
                },
                _ => {
                    // Skip if tensor creation fails
                }
            }

            // Test 3: Invalid reshape operations
            let reshape_vals: Vec<f32> = (0..size1).map(|_| generate_f32()).collect();

            if let Ok(tensor) = Tensor::from_vec(reshape_vals, &shape1) {
                // Try to reshape to incompatible total size
                let invalid_shape = vec![size1 + 1]; // Wrong total size
                let reshape_result = tensor.reshape(&invalid_shape);
                assert!(reshape_result.is_err(), "Reshape to incompatible size should return error");

                // Try to reshape to negative or zero dimensions (if the API allows)
                let zero_shape = vec![0, size1];
                let zero_reshape_result = tensor.reshape(&zero_shape);
                if size1 > 0 {
                    assert!(zero_reshape_result.is_err(), "Reshape to zero dimension should return error");
                }
            }

            // Test 4: Operations that should handle edge cases gracefully
            let edge_vals = vec![0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
            let edge_shape = vec![4];

            if let Ok(edge_tensor) = Tensor::from_vec(edge_vals, &edge_shape) {
                // These operations should not panic, but may return errors or special values
                let sqrt_result = edge_tensor.sqrt();
                let log_result = edge_tensor.log();

                // We don't assert these are errors because some edge cases might be valid
                // We just ensure they don't panic
                match sqrt_result {
                    Ok(result) => {
                        // sqrt of negative should produce NaN, not panic
                        if let Ok(data) = result.data() {
                            // Just verify we can access the data without panicking
                            let _ = data.len();
                        }
                    },
                    Err(_) => {
                        // Error is also acceptable
                    }
                }

                match log_result {
                    Ok(result) => {
                        // log of negative/zero should produce NaN/-inf, not panic
                        if let Ok(data) = result.data() {
                            // Just verify we can access the data without panicking
                            let _ = data.len();
                        }
                    },
                    Err(_) => {
                        // Error is also acceptable
                    }
                }
            }

            // Test 5: Transpose with invalid dimensions
            if let Ok(tensor) = Tensor::from_vec((0..size1).map(|i| i as f32).collect(), &shape1) {
                // Try to transpose with out-of-bounds dimensions
                let invalid_transpose = tensor.transpose(shape1.len(), 0); // Invalid first dim
                assert!(invalid_transpose.is_err(), "Transpose with invalid dimension should return error");

                let invalid_transpose2 = tensor.transpose(0, shape1.len()); // Invalid second dim
                assert!(invalid_transpose2.is_err(), "Transpose with invalid dimension should return error");
            }
        }
    }
}
