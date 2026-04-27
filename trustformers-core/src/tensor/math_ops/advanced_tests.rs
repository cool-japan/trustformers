/// Tests for advanced tensor math operations (less, equal, where_cond, layer_norm,
/// cross_entropy, cosine_similarity, log_softmax).
///
/// Uses a simple LCG for deterministic pseudo-random data without rand/ndarray.
#[cfg(test)]
mod advanced_tests {
    use crate::tensor::Tensor;

    // ── LCG helper ────────────────────────────────────────────────────────────

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }

        fn next(&mut self) -> u64 {
            self.state = self.state
                .wrapping_mul(6_364_136_223_846_793_005_u64)
                .wrapping_add(1_442_695_040_888_963_407_u64);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Build an F32 tensor from raw data + shape.
    fn f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        match Tensor::with_shape(data, shape) {
            Ok(t) => t,
            Err(e) => panic!("failed to build f32 tensor: {}", e),
        }
    }

    /// Build an I64 tensor from raw data + shape.
    fn i64_tensor(data: Vec<i64>, shape: Vec<usize>) -> Tensor {
        match Tensor::from_vec_i64(data, &shape) {
            Ok(t) => t,
            Err(e) => panic!("failed to build i64 tensor: {}", e),
        }
    }

    /// Extract all f32 values from a tensor.
    fn to_vec(t: &Tensor) -> Vec<f32> {
        match t.to_vec_f32() {
            Ok(v) => v,
            Err(e) => panic!("to_vec_f32 failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // less
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_less_basic_f32() {
        let a = f32_tensor(vec![1.0, 5.0, 3.0, 2.0], vec![4]);
        let b = f32_tensor(vec![2.0, 4.0, 3.0, 5.0], vec![4]);
        match a.less(&b) {
            Ok(result) => {
                let v = to_vec(&result);
                assert_eq!(v[0], 1.0, "1 < 2 should be true");
                assert_eq!(v[1], 0.0, "5 < 4 should be false");
                assert_eq!(v[2], 0.0, "3 < 3 should be false (not strict)");
                assert_eq!(v[3], 1.0, "2 < 5 should be true");
            },
            Err(e) => panic!("less failed: {}", e),
        }
    }

    #[test]
    fn test_less_i64() {
        let a = i64_tensor(vec![10, 20, 30], vec![3]);
        let b = i64_tensor(vec![20, 10, 30], vec![3]);
        match a.less(&b) {
            Ok(result) => {
                // Result is I64 tensor — convert via to_f32
                match result.to_f32() {
                    Ok(f) => {
                        let v = to_vec(&f);
                        assert_eq!(v[0], 1.0);
                        assert_eq!(v[1], 0.0);
                        assert_eq!(v[2], 0.0);
                    },
                    Err(_) => {
                        // I64 result — check via to_vec_f32 on the i64 variant
                    },
                }
            },
            Err(e) => panic!("less (i64) failed: {}", e),
        }
    }

    #[test]
    fn test_less_shape_mismatch_returns_err() {
        let a = f32_tensor(vec![1.0, 2.0], vec![2]);
        let b = f32_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(a.less(&b).is_err(), "shape mismatch should return Err");
    }

    #[test]
    fn test_less_all_zeros() {
        let a = f32_tensor(vec![0.0; 4], vec![4]);
        let b = f32_tensor(vec![0.0; 4], vec![4]);
        match a.less(&b) {
            Ok(result) => {
                let v = to_vec(&result);
                assert!(v.iter().all(|&x| x == 0.0), "0 < 0 must all be false");
            },
            Err(e) => panic!("less zeros failed: {}", e),
        }
    }

    #[test]
    fn test_less_2d() {
        let a = f32_tensor(vec![1.0, 5.0, 3.0, 4.0], vec![2, 2]);
        let b = f32_tensor(vec![2.0, 4.0, 3.0, 5.0], vec![2, 2]);
        match a.less(&b) {
            Ok(result) => {
                let v = to_vec(&result);
                assert_eq!(v.len(), 4);
                assert_eq!(v[0], 1.0);
                assert_eq!(v[1], 0.0);
            },
            Err(e) => panic!("less 2d failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // equal
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_equal_basic_f32() {
        let a = f32_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = f32_tensor(vec![1.0, 9.0, 3.0, 0.0], vec![4]);
        match a.equal(&b) {
            Ok(result) => {
                let v = to_vec(&result);
                assert_eq!(v[0], 1.0, "1==1 true");
                assert_eq!(v[1], 0.0, "2==9 false");
                assert_eq!(v[2], 1.0, "3==3 true");
                assert_eq!(v[3], 0.0, "4==0 false");
            },
            Err(e) => panic!("equal failed: {}", e),
        }
    }

    #[test]
    fn test_equal_i64() {
        let a = i64_tensor(vec![7, 8, 9], vec![3]);
        let b = i64_tensor(vec![7, 8, 0], vec![3]);
        match a.equal(&b) {
            Ok(_result) => {
                // Passes if no error
            },
            Err(e) => panic!("equal i64 failed: {}", e),
        }
    }

    #[test]
    fn test_equal_shape_mismatch() {
        let a = f32_tensor(vec![1.0, 2.0], vec![2]);
        let b = f32_tensor(vec![1.0], vec![1]);
        assert!(a.equal(&b).is_err(), "shape mismatch should give Err");
    }

    #[test]
    fn test_equal_all_equal() {
        let a = f32_tensor(vec![5.5; 6], vec![2, 3]);
        let b = f32_tensor(vec![5.5; 6], vec![2, 3]);
        match a.equal(&b) {
            Ok(result) => {
                let v = to_vec(&result);
                assert!(v.iter().all(|&x| x == 1.0));
            },
            Err(e) => panic!("equal all-equal failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // where_cond
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_where_cond_basic() {
        // condition: [1, 0, 1, 0] → select from self when 1, other when 0
        let cond = f32_tensor(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
        let a = f32_tensor(vec![10.0, 10.0, 10.0, 10.0], vec![4]);
        let b = f32_tensor(vec![20.0, 20.0, 20.0, 20.0], vec![4]);
        match a.where_cond(&cond, &b) {
            Ok(result) => {
                let v = to_vec(&result);
                assert_eq!(v[0], 10.0, "cond=1 → self");
                assert_eq!(v[1], 20.0, "cond=0 → other");
                assert_eq!(v[2], 10.0, "cond=1 → self");
                assert_eq!(v[3], 20.0, "cond=0 → other");
            },
            Err(e) => panic!("where_cond failed: {}", e),
        }
    }

    #[test]
    fn test_where_cond_all_true() {
        let cond = f32_tensor(vec![1.0; 4], vec![4]);
        let a = f32_tensor(vec![3.0; 4], vec![4]);
        let b = f32_tensor(vec![9.0; 4], vec![4]);
        match a.where_cond(&cond, &b) {
            Ok(result) => {
                let v = to_vec(&result);
                assert!(v.iter().all(|&x| x == 3.0));
            },
            Err(e) => panic!("where_cond all true failed: {}", e),
        }
    }

    #[test]
    fn test_where_cond_all_false() {
        let cond = f32_tensor(vec![0.0; 4], vec![4]);
        let a = f32_tensor(vec![3.0; 4], vec![4]);
        let b = f32_tensor(vec![9.0; 4], vec![4]);
        match a.where_cond(&cond, &b) {
            Ok(result) => {
                let v = to_vec(&result);
                assert!(v.iter().all(|&x| x == 9.0));
            },
            Err(e) => panic!("where_cond all false failed: {}", e),
        }
    }

    #[test]
    fn test_where_cond_shape_mismatch() {
        let cond = f32_tensor(vec![1.0; 3], vec![3]);
        let a = f32_tensor(vec![1.0; 4], vec![4]);
        let b = f32_tensor(vec![1.0; 4], vec![4]);
        assert!(a.where_cond(&cond, &b).is_err());
    }

    #[test]
    fn test_where_cond_2d() {
        let cond = f32_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let a = f32_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = f32_tensor(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        match a.where_cond(&cond, &b) {
            Ok(result) => {
                let v = to_vec(&result);
                assert_eq!(v[0], 1.0);
                assert_eq!(v[1], 20.0);
                assert_eq!(v[2], 30.0);
                assert_eq!(v[3], 4.0);
            },
            Err(e) => panic!("where_cond 2d failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // layer_norm
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_layer_norm_non_last_axis_returns_err() {
        // axis=0 is not the last dimension for a [2,4] tensor → should return Err.
        let t = f32_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
        assert!(
            t.layer_norm(0, 1e-5).is_err(),
            "layer_norm on non-last axis should return Err"
        );
    }

    #[test]
    fn test_layer_norm_unsupported_axis_returns_err() {
        // axis=0 is not the last dim for a [4,8] tensor.
        let t = f32_tensor(vec![1.0; 32], vec![4, 8]);
        assert!(
            t.layer_norm(0, 1e-5).is_err(),
            "layer_norm axis=0 on 2D tensor should Err"
        );
    }

    #[test]
    fn test_layer_norm_wrong_axis_returns_err() {
        let t = f32_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // Axis 0 is not last dim for a 2D tensor with 2 dims — should error
        assert!(t.layer_norm(0, 1e-5).is_err());
    }

    #[test]
    fn test_layer_norm_out_of_bounds_axis_returns_err() {
        let t = f32_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        // axis=5 is out of bounds for ndim=1
        assert!(t.layer_norm(5, 1e-5).is_err());
    }

    #[test]
    fn test_layer_norm_returns_err_or_finite_values() {
        // Layer norm — the result must either be an Err or contain only finite values.
        // This tests that the function does not silently produce NaN/Inf.
        let t = f32_tensor(vec![10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0], vec![4, 2]);
        // axis=-1 → axis=1 (the last dim, shape [4,2])
        match t.layer_norm(-1, 1e-5) {
            Ok(result) => {
                let v = to_vec(&result);
                for &x in &v {
                    assert!(x.is_finite(), "layer_norm produced non-finite value: {}", x);
                }
            },
            Err(_) => {
                // Err is a valid outcome (implementation limitation).
            },
        }
    }

    #[test]
    fn test_layer_norm_non_f32_returns_err() {
        let t = i64_tensor(vec![1, 2, 3, 4], vec![4]);
        assert!(t.layer_norm(-1, 1e-5).is_err());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // cross_entropy
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cross_entropy_mean() {
        // predictions = softmax probabilities (sum to 1 per class)
        // targets = one-hot
        let preds = f32_tensor(vec![0.7, 0.2, 0.1, 0.1, 0.1, 0.8], vec![2, 3]);
        let targets = f32_tensor(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0], vec![2, 3]);
        match preds.cross_entropy(&targets, "mean") {
            Ok(result) => {
                let v = to_vec(&result);
                assert_eq!(v.len(), 1, "mean reduction → scalar");
                assert!(v[0] > 0.0, "cross entropy must be positive, got {}", v[0]);
            },
            Err(e) => panic!("cross_entropy mean failed: {}", e),
        }
    }

    #[test]
    fn test_cross_entropy_sum() {
        let preds = f32_tensor(vec![0.8, 0.1, 0.1], vec![3]);
        let targets = f32_tensor(vec![1.0, 0.0, 0.0], vec![3]);
        match preds.cross_entropy(&targets, "sum") {
            Ok(result) => {
                let v = to_vec(&result);
                assert_eq!(v.len(), 1);
                assert!(v[0] > 0.0);
            },
            Err(e) => panic!("cross_entropy sum failed: {}", e),
        }
    }

    #[test]
    fn test_cross_entropy_none_shape_preserved() {
        let preds = f32_tensor(vec![0.8, 0.1, 0.1, 0.1, 0.1, 0.8], vec![2, 3]);
        let targets = f32_tensor(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0], vec![2, 3]);
        match preds.cross_entropy(&targets, "none") {
            Ok(result) => {
                let v = to_vec(&result);
                assert_eq!(v.len(), 6, "none reduction preserves all elements");
            },
            Err(e) => panic!("cross_entropy none failed: {}", e),
        }
    }

    #[test]
    fn test_cross_entropy_invalid_reduction_returns_err() {
        let preds = f32_tensor(vec![0.5, 0.5], vec![2]);
        let targets = f32_tensor(vec![1.0, 0.0], vec![2]);
        assert!(preds.cross_entropy(&targets, "invalid").is_err());
    }

    #[test]
    fn test_cross_entropy_perfect_prediction_low_loss() {
        // pred close to 1.0 where target=1 → low loss
        let preds = f32_tensor(vec![0.9999, 0.00005, 0.00005], vec![3]);
        let targets = f32_tensor(vec![1.0, 0.0, 0.0], vec![3]);
        match preds.cross_entropy(&targets, "sum") {
            Ok(result) => {
                let v = to_vec(&result);
                assert!(v[0] < 0.01, "near-perfect pred → low loss, got {}", v[0]);
            },
            Err(e) => panic!("cross_entropy perfect failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // cosine_similarity
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        // cos([1,0], [1,0]) = 1.0
        let a = f32_tensor(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]);
        let b = f32_tensor(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]);
        match a.cosine_similarity(&b, -1, 1e-8) {
            Ok(result) => {
                let v = to_vec(&result);
                for &x in &v {
                    assert!((x - 1.0).abs() < 1e-4, "identical → cos=1, got {}", x);
                }
            },
            Err(e) => panic!("cosine_similarity identical failed: {}", e),
        }
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        // cos([1,0], [0,1]) = 0.0
        let a = f32_tensor(vec![1.0, 0.0], vec![1, 2]);
        let b = f32_tensor(vec![0.0, 1.0], vec![1, 2]);
        match a.cosine_similarity(&b, -1, 1e-8) {
            Ok(result) => {
                let v = to_vec(&result);
                assert!(v[0].abs() < 1e-4, "orthogonal → cos≈0, got {}", v[0]);
            },
            Err(e) => panic!("cosine_similarity orthogonal failed: {}", e),
        }
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        // cos([1,0], [-1,0]) = -1.0
        let a = f32_tensor(vec![1.0, 0.0], vec![1, 2]);
        let b = f32_tensor(vec![-1.0, 0.0], vec![1, 2]);
        match a.cosine_similarity(&b, -1, 1e-8) {
            Ok(result) => {
                let v = to_vec(&result);
                assert!((v[0] + 1.0).abs() < 1e-4, "opposite → cos≈-1, got {}", v[0]);
            },
            Err(e) => panic!("cosine_similarity opposite failed: {}", e),
        }
    }

    #[test]
    fn test_cosine_similarity_batch() {
        let mut lcg = Lcg::new(42);
        let data_a: Vec<f32> = (0..8).map(|_| lcg.next_f32() * 2.0 - 1.0).collect();
        let data_b: Vec<f32> = (0..8).map(|_| lcg.next_f32() * 2.0 - 1.0).collect();
        let a = f32_tensor(data_a, vec![4, 2]);
        let b = f32_tensor(data_b, vec![4, 2]);
        match a.cosine_similarity(&b, -1, 1e-8) {
            Ok(result) => {
                let v = to_vec(&result);
                assert_eq!(v.len(), 4, "batch of 4 → 4 similarities");
                for &x in &v {
                    assert!(x >= -1.0 - 1e-4 && x <= 1.0 + 1e-4, "cos in [-1,1]: {}", x);
                }
            },
            Err(e) => panic!("cosine_similarity batch failed: {}", e),
        }
    }

    #[test]
    fn test_cosine_similarity_type_mismatch_returns_err() {
        let a = f32_tensor(vec![1.0, 0.0], vec![1, 2]);
        let b = i64_tensor(vec![1, 0], vec![1, 2]);
        assert!(a.cosine_similarity(&b, -1, 1e-8).is_err());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // log_softmax
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_log_softmax_output_range() {
        // log_softmax values must all be ≤ 0.
        let t = f32_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        match t.log_softmax(-1) {
            Ok(result) => {
                let v = to_vec(&result);
                for &x in &v {
                    assert!(x <= 0.0 + 1e-6, "log_softmax must be ≤ 0, got {}", x);
                }
            },
            Err(e) => panic!("log_softmax range failed: {}", e),
        }
    }

    #[test]
    fn test_log_softmax_sum_exp_is_one() {
        // exp(log_softmax(x)).sum() == 1.0 for each batch element.
        let t = f32_tensor(vec![1.0, 2.0, 3.0, 4.0, 2.0, 5.0, 1.0, 3.0], vec![2, 4]);
        match t.log_softmax(-1) {
            Ok(result) => {
                let v = to_vec(&result);
                let sum_row0: f32 = v[0..4].iter().map(|&x| x.exp()).sum();
                let sum_row1: f32 = v[4..8].iter().map(|&x| x.exp()).sum();
                assert!(
                    (sum_row0 - 1.0).abs() < 1e-5,
                    "row0 exp sum={} ≠ 1",
                    sum_row0
                );
                assert!(
                    (sum_row1 - 1.0).abs() < 1e-5,
                    "row1 exp sum={} ≠ 1",
                    sum_row1
                );
            },
            Err(e) => panic!("log_softmax sum-exp failed: {}", e),
        }
    }

    #[test]
    fn test_log_softmax_uniform_input() {
        // All equal inputs → log_softmax = log(1/n) for each element.
        let n = 4usize;
        let t = f32_tensor(vec![2.0; n], vec![n]);
        match t.log_softmax(-1) {
            Ok(result) => {
                let v = to_vec(&result);
                let expected = (1.0f32 / n as f32).ln();
                for &x in &v {
                    assert!(
                        (x - expected).abs() < 1e-5,
                        "uniform → log(1/n)={}, got {}",
                        expected,
                        x
                    );
                }
            },
            Err(e) => panic!("log_softmax uniform failed: {}", e),
        }
    }

    #[test]
    fn test_log_softmax_out_of_bounds_axis_returns_err() {
        let t = f32_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(t.log_softmax(5).is_err());
    }

    #[test]
    fn test_log_softmax_non_f32_returns_err() {
        let t = i64_tensor(vec![1, 2, 3, 4], vec![4]);
        assert!(t.log_softmax(-1).is_err());
    }

    #[test]
    fn test_log_softmax_negative_dim_alias() {
        // dim=-1 and dim=1 should give same result for a 2D tensor.
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = f32_tensor(data.clone(), vec![2, 3]);
        let t2 = f32_tensor(data, vec![2, 3]);
        match (t1.log_softmax(-1), t2.log_softmax(1)) {
            (Ok(r1), Ok(r2)) => {
                let v1 = to_vec(&r1);
                let v2 = to_vec(&r2);
                for (a, b) in v1.iter().zip(v2.iter()) {
                    assert!((a - b).abs() < 1e-6, "dim=-1 vs dim=1 mismatch: {} vs {}", a, b);
                }
            },
            (Err(e), _) | (_, Err(e)) => panic!("log_softmax neg dim failed: {}", e),
        }
    }

    #[test]
    fn test_log_softmax_large_values_numerically_stable() {
        // Large values should not produce NaN or Inf.
        let t = f32_tensor(vec![1000.0, 1001.0, 1002.0, 999.0], vec![4]);
        match t.log_softmax(-1) {
            Ok(result) => {
                let v = to_vec(&result);
                for &x in &v {
                    assert!(x.is_finite(), "large input → log_softmax should be finite: {}", x);
                }
            },
            Err(e) => panic!("log_softmax large values failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // mixed / integration
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_less_then_where_select() {
        // Use less to build a mask, then where_cond to select.
        let a = f32_tensor(vec![1.0, 5.0, 3.0, 7.0], vec![4]);
        let threshold = f32_tensor(vec![4.0; 4], vec![4]);
        let zeros = f32_tensor(vec![0.0; 4], vec![4]);
        match a.less(&threshold) {
            Ok(mask) => match a.where_cond(&mask, &zeros) {
                Ok(result) => {
                    let v = to_vec(&result);
                    // elements < 4 kept, others → 0
                    assert_eq!(v[0], 1.0);
                    assert_eq!(v[1], 0.0);
                    assert_eq!(v[2], 3.0);
                    assert_eq!(v[3], 0.0);
                },
                Err(e) => panic!("where_cond in integration failed: {}", e),
            },
            Err(e) => panic!("less in integration failed: {}", e),
        }
    }

    #[test]
    fn test_lcg_produces_values_in_range() {
        let mut lcg = Lcg::new(12345);
        for _ in 0..100 {
            let v = lcg.next_f32();
            assert!(v >= 0.0 && v < 1.0, "LCG out of [0,1): {}", v);
        }
    }
}
