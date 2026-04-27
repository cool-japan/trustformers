/// Tests for activation functions: gelu, relu, sigmoid, tanh, silu, swiglu
#[cfg(test)]
mod tests {
    use crate::ops::activations::{gelu, gelu_new, relu, sigmoid, silu, swiglu, tanh};
    use crate::tensor::Tensor;
    use scirs2_core::ndarray::ArrayD;

    fn make_f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let arr = ArrayD::from_shape_vec(shape, data).expect("Valid shape");
        Tensor::F32(arr)
    }

    fn extract_f32(t: &Tensor) -> Vec<f32> {
        match t {
            Tensor::F32(arr) => arr.iter().cloned().collect(),
            _ => panic!("Expected F32 tensor"),
        }
    }

    // ---- ReLU tests ----

    #[test]
    fn test_relu_positive_values_unchanged() {
        let t = make_f32_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        let result = relu(&t).expect("relu should succeed");
        let data = extract_f32(&result);
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_negative_values_become_zero() {
        let t = make_f32_tensor(vec![-1.0, -5.0, -0.001], vec![3]);
        let result = relu(&t).expect("relu should succeed");
        let data = extract_f32(&result);
        for v in data {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_relu_zero_stays_zero() {
        let t = make_f32_tensor(vec![0.0], vec![1]);
        let result = relu(&t).expect("relu should succeed");
        let data = extract_f32(&result);
        assert_eq!(data[0], 0.0);
    }

    #[test]
    fn test_relu_mixed_values() {
        let t = make_f32_tensor(vec![-2.0, 0.0, 3.0, -0.5, 1.0], vec![5]);
        let result = relu(&t).expect("relu should succeed");
        let data = extract_f32(&result);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 0.0);
        assert!((data[2] - 3.0).abs() < 1e-6);
        assert_eq!(data[3], 0.0);
        assert!((data[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_2d_tensor() {
        let t = make_f32_tensor(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]);
        let result = relu(&t).expect("relu should succeed");
        let data = extract_f32(&result);
        assert_eq!(data[0], 0.0);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert_eq!(data[2], 0.0);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    // ---- Sigmoid tests ----

    #[test]
    fn test_sigmoid_zero_gives_half() {
        let t = make_f32_tensor(vec![0.0], vec![1]);
        let result = sigmoid(&t).expect("sigmoid should succeed");
        let data = extract_f32(&result);
        assert!((data[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_large_positive_near_one() {
        let t = make_f32_tensor(vec![20.0], vec![1]);
        let result = sigmoid(&t).expect("sigmoid should succeed");
        let data = extract_f32(&result);
        assert!(data[0] > 0.99);
        assert!(data[0] <= 1.0);
    }

    #[test]
    fn test_sigmoid_large_negative_near_zero() {
        let t = make_f32_tensor(vec![-20.0], vec![1]);
        let result = sigmoid(&t).expect("sigmoid should succeed");
        let data = extract_f32(&result);
        assert!(data[0] < 0.01);
        assert!(data[0] >= 0.0);
    }

    #[test]
    fn test_sigmoid_output_range_all_in_0_1() {
        let mut s = 42u64;
        let mut vals = Vec::new();
        for _ in 0..20 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = (s % 1000) as f32 / 100.0 - 5.0; // [-5, 5)
            vals.push(v);
        }
        let t = make_f32_tensor(vals.clone(), vec![vals.len()]);
        let result = sigmoid(&t).expect("sigmoid should succeed");
        let data = extract_f32(&result);
        for v in data {
            assert!(
                (0.0..=1.0).contains(&v),
                "Sigmoid output {} not in [0, 1]",
                v
            );
        }
    }

    // ---- Tanh tests ----

    #[test]
    fn test_tanh_zero_gives_zero() {
        let t = make_f32_tensor(vec![0.0], vec![1]);
        let result = tanh(&t).expect("tanh should succeed");
        let data = extract_f32(&result);
        assert!((data[0]).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_range_minus1_to_1() {
        let vals: Vec<f32> = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
        let n = vals.len();
        let t = make_f32_tensor(vals, vec![n]);
        let result = tanh(&t).expect("tanh should succeed");
        let data = extract_f32(&result);
        for v in data {
            assert!((-1.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_tanh_odd_function() {
        let t_pos = make_f32_tensor(vec![1.5], vec![1]);
        let t_neg = make_f32_tensor(vec![-1.5], vec![1]);
        let r_pos = tanh(&t_pos).expect("tanh should succeed");
        let r_neg = tanh(&t_neg).expect("tanh should succeed");
        let pos_data = extract_f32(&r_pos);
        let neg_data = extract_f32(&r_neg);
        assert!((pos_data[0] + neg_data[0]).abs() < 1e-5);
    }

    // ---- SiLU (Swish) tests ----

    #[test]
    fn test_silu_zero_gives_zero() {
        let t = make_f32_tensor(vec![0.0], vec![1]);
        let result = silu(&t).expect("silu should succeed");
        let data = extract_f32(&result);
        assert!((data[0]).abs() < 1e-6);
    }

    #[test]
    fn test_silu_large_positive_approaches_identity() {
        // silu(x) = x * sigmoid(x) ≈ x for large x
        let t = make_f32_tensor(vec![10.0], vec![1]);
        let result = silu(&t).expect("silu should succeed");
        let data = extract_f32(&result);
        assert!((data[0] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_silu_negative_input_small_output() {
        // silu(-10) ≈ -10 * sigmoid(-10) ≈ 0
        let t = make_f32_tensor(vec![-10.0], vec![1]);
        let result = silu(&t).expect("silu should succeed");
        let data = extract_f32(&result);
        assert!(data[0].abs() < 0.01);
    }

    // ---- GELU tests ----

    #[test]
    fn test_gelu_large_positive_approaches_identity() {
        let t = make_f32_tensor(vec![10.0], vec![1]);
        let result = gelu(&t).expect("gelu should succeed");
        let data = extract_f32(&result);
        // gelu(10) ≈ 10
        assert!((data[0] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_gelu_large_negative_approaches_zero() {
        let t = make_f32_tensor(vec![-10.0], vec![1]);
        let result = gelu(&t).expect("gelu should succeed");
        let data = extract_f32(&result);
        // gelu(-10) ≈ 0
        assert!(data[0].abs() < 0.01);
    }

    #[test]
    fn test_gelu_no_nan() {
        let vals: Vec<f32> = vec![-5.0, -2.0, 0.0, 0.5, 2.0, 5.0];
        let n = vals.len();
        let t = make_f32_tensor(vals, vec![n]);
        let result = gelu(&t).expect("gelu should succeed");
        let data = extract_f32(&result);
        for v in data {
            assert!(!v.is_nan(), "gelu output was NaN");
            assert!(!v.is_infinite(), "gelu output was infinite");
        }
    }

    #[test]
    fn test_gelu_new_no_nan() {
        let vals: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let n = vals.len();
        let t = make_f32_tensor(vals, vec![n]);
        let result = gelu_new(&t).expect("gelu_new should succeed");
        let data = extract_f32(&result);
        for v in data {
            assert!(!v.is_nan());
        }
    }

    // ---- SwiGLU tests ----

    #[test]
    fn test_swiglu_basic() {
        let x = make_f32_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        let gate = make_f32_tensor(vec![1.0, 1.0, 1.0], vec![3]);
        // SwiGLU(x, gate) = SiLU(gate) * x
        let result = swiglu(&x, &gate).expect("swiglu should succeed");
        let data = extract_f32(&result);
        // silu(1.0) * 1.0 = 1/(1+e^-1) * 1.0 ≈ 0.731
        assert!((data[0] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_swiglu_zero_gate_gives_zero() {
        let x = make_f32_tensor(vec![5.0, 10.0], vec![2]);
        let gate = make_f32_tensor(vec![-100.0, -100.0], vec![2]);
        let result = swiglu(&x, &gate).expect("swiglu should succeed");
        let data = extract_f32(&result);
        // silu(-100) ≈ 0, so SwiGLU ≈ 0
        for v in data {
            assert!(v.abs() < 0.01);
        }
    }
}
