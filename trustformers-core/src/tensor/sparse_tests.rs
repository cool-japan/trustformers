/// Tests for sparse tensor operations (to_sparse, to_dense, is_sparse, sparsity, nnz,
/// sparse_coo, sparse_csr).
#[cfg(test)]
mod sparse_tests {
    use crate::tensor::Tensor;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        match Tensor::with_shape(data, shape) {
            Ok(t) => t,
            Err(e) => panic!("failed to build f32 tensor: {}", e),
        }
    }

    fn to_vec(t: &Tensor) -> Vec<f32> {
        match t.to_vec_f32() {
            Ok(v) => v,
            Err(e) => panic!("to_vec_f32 failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // is_sparse / to_sparse basics
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dense_tensor_is_not_sparse() {
        let t = f32_tensor(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        assert!(!t.is_sparse(), "a freshly built F32 tensor must not be sparse");
    }

    #[test]
    fn test_to_sparse_produces_sparse_tensor() {
        let t = f32_tensor(vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0], vec![2, 3]);
        match t.to_sparse(0.5) {
            Ok(sparse) => assert!(sparse.is_sparse(), "result of to_sparse must be sparse"),
            Err(e) => panic!("to_sparse failed: {}", e),
        }
    }

    #[test]
    fn test_already_sparse_stays_sparse() {
        let t = f32_tensor(vec![0.0, 1.0, 0.0, 0.0], vec![2, 2]);
        match t.to_sparse(0.5) {
            Ok(sparse) => match sparse.to_sparse(0.5) {
                Ok(again) => assert!(again.is_sparse()),
                Err(e) => panic!("second to_sparse failed: {}", e),
            },
            Err(e) => panic!("first to_sparse failed: {}", e),
        }
    }

    #[test]
    fn test_to_sparse_non_f32_returns_err() {
        let t = match Tensor::from_vec_i64(vec![1, 2, 3, 4], &[4]) {
            Ok(t) => t,
            Err(e) => panic!("build i64 tensor failed: {}", e),
        };
        assert!(
            t.to_sparse(0.5).is_err(),
            "i64 tensor → to_sparse should return Err"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // to_dense
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dense_to_dense_is_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = f32_tensor(data.clone(), vec![2, 2]);
        match t.to_dense() {
            Ok(dense) => {
                let v = to_vec(&dense);
                assert_eq!(v, data, "dense→dense must preserve values");
            },
            Err(e) => panic!("to_dense on dense failed: {}", e),
        }
    }

    #[test]
    fn test_sparse_to_dense_roundtrip() {
        // Create a 3×3 tensor with some zeros, convert to sparse and back.
        let data = vec![1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 9.0];
        let t = f32_tensor(data.clone(), vec![3, 3]);
        match t.to_sparse(0.5) {
            Ok(sparse) => match sparse.to_dense() {
                Ok(dense) => {
                    let v = to_vec(&dense);
                    assert_eq!(v.len(), 9);
                    // Non-zero positions must be preserved
                    assert!((v[0] - 1.0).abs() < 1e-6, "pos 0 must be 1.0, got {}", v[0]);
                    assert!((v[4] - 5.0).abs() < 1e-6, "pos 4 must be 5.0, got {}", v[4]);
                    assert!((v[8] - 9.0).abs() < 1e-6, "pos 8 must be 9.0, got {}", v[8]);
                },
                Err(e) => panic!("to_dense after sparse failed: {}", e),
            },
            Err(e) => panic!("to_sparse in roundtrip failed: {}", e),
        }
    }

    #[test]
    fn test_all_zeros_sparse_to_dense() {
        let t = f32_tensor(vec![0.0; 6], vec![2, 3]);
        match t.to_sparse(0.5) {
            Ok(sparse) => match sparse.to_dense() {
                Ok(dense) => {
                    let v = to_vec(&dense);
                    assert!(v.iter().all(|&x| x == 0.0), "all-zeros roundtrip must stay zero");
                },
                Err(e) => panic!("to_dense all-zeros failed: {}", e),
            },
            Err(e) => panic!("to_sparse all-zeros failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // sparsity
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_sparsity_all_zeros() {
        let t = f32_tensor(vec![0.0; 8], vec![2, 4]);
        match t.sparsity() {
            Ok(s) => assert!((s - 1.0).abs() < 1e-6, "all-zeros → sparsity=1.0, got {}", s),
            Err(e) => panic!("sparsity all-zeros failed: {}", e),
        }
    }

    #[test]
    fn test_sparsity_no_zeros() {
        let t = f32_tensor(vec![1.0; 8], vec![2, 4]);
        match t.sparsity() {
            Ok(s) => assert!((s - 0.0).abs() < 1e-6, "no-zeros → sparsity=0.0, got {}", s),
            Err(e) => panic!("sparsity no-zeros failed: {}", e),
        }
    }

    #[test]
    fn test_sparsity_half() {
        // 4 zeros out of 8 elements.
        let t = f32_tensor(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], vec![2, 4]);
        match t.sparsity() {
            Ok(s) => assert!((s - 0.5).abs() < 1e-6, "half-zeros → sparsity=0.5, got {}", s),
            Err(e) => panic!("sparsity half failed: {}", e),
        }
    }

    #[test]
    fn test_sparsity_of_sparse_tensor() {
        let t = f32_tensor(vec![1.0, 0.0, 0.0, 0.0], vec![2, 2]);
        match t.to_sparse(0.5) {
            Ok(sparse) => match sparse.sparsity() {
                Ok(s) => {
                    // At least 3 out of 4 are zero → sparsity ≥ 0.5
                    assert!(s >= 0.5, "sparse tensor sparsity must be ≥ 0.5, got {}", s);
                },
                Err(e) => panic!("sparsity on sparse tensor failed: {}", e),
            },
            Err(e) => panic!("to_sparse in sparsity test failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // nnz
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_nnz_dense_all_nonzero() {
        let t = f32_tensor(vec![1.0; 6], vec![2, 3]);
        match t.nnz() {
            Ok(n) => assert_eq!(n, 6, "all-nonzero → nnz=6"),
            Err(e) => panic!("nnz all-nonzero failed: {}", e),
        }
    }

    #[test]
    fn test_nnz_dense_mixed() {
        let t = f32_tensor(vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0], vec![2, 3]);
        match t.nnz() {
            Ok(n) => assert_eq!(n, 3, "3 non-zeros expected, got {}", n),
            Err(e) => panic!("nnz mixed failed: {}", e),
        }
    }

    #[test]
    fn test_nnz_sparse_tensor() {
        // Build a sparse tensor with 2 explicit non-zeros.
        match Tensor::sparse_coo(
            vec![vec![0, 1], vec![1, 2]],
            vec![7.0, 8.0],
            vec![3, 4],
        ) {
            Ok(sparse) => match sparse.nnz() {
                Ok(n) => assert_eq!(n, 2, "COO with 2 values → nnz=2, got {}", n),
                Err(e) => panic!("nnz on sparse_coo failed: {}", e),
            },
            Err(e) => panic!("sparse_coo construction failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // sparse_coo
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_sparse_coo_construction() {
        // 3×3 matrix with 3 non-zeros on the diagonal.
        match Tensor::sparse_coo(
            vec![vec![0, 1, 2], vec![0, 1, 2]],
            vec![1.0, 2.0, 3.0],
            vec![3, 3],
        ) {
            Ok(t) => {
                assert!(t.is_sparse());
                match t.nnz() {
                    Ok(n) => assert_eq!(n, 3),
                    Err(e) => panic!("nnz on COO failed: {}", e),
                }
            },
            Err(e) => panic!("sparse_coo diagonal failed: {}", e),
        }
    }

    #[test]
    fn test_sparse_coo_single_element() {
        match Tensor::sparse_coo(vec![vec![0], vec![0]], vec![42.0], vec![5, 5]) {
            Ok(t) => {
                assert!(t.is_sparse());
                match t.nnz() {
                    Ok(n) => assert_eq!(n, 1),
                    Err(e) => panic!("nnz single elem COO: {}", e),
                }
            },
            Err(e) => panic!("sparse_coo single elem failed: {}", e),
        }
    }

    #[test]
    fn test_sparse_coo_to_dense_values() {
        // 2×2 matrix: only (0,1)=5.0 is non-zero.
        match Tensor::sparse_coo(vec![vec![0], vec![1]], vec![5.0], vec![2, 2]) {
            Ok(sparse) => match sparse.to_dense() {
                Ok(dense) => {
                    let v = to_vec(&dense);
                    assert_eq!(v.len(), 4);
                    assert!((v[1] - 5.0).abs() < 1e-6, "pos (0,1) must be 5.0, got {}", v[1]);
                    assert!(v[0].abs() < 1e-6, "pos (0,0) must be 0, got {}", v[0]);
                    assert!(v[2].abs() < 1e-6, "pos (1,0) must be 0, got {}", v[2]);
                    assert!(v[3].abs() < 1e-6, "pos (1,1) must be 0, got {}", v[3]);
                },
                Err(e) => panic!("to_dense from COO failed: {}", e),
            },
            Err(e) => panic!("sparse_coo 2×2 single failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // sparse_csr
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_sparse_csr_construction() {
        // 3×3 identity matrix in CSR format:
        //   row_ptr    = [0, 1, 2, 3]
        //   col_indices = [0, 1, 2]
        //   values      = [1, 1, 1]
        match Tensor::sparse_csr(
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![1.0, 1.0, 1.0],
            vec![3, 3],
        ) {
            Ok(t) => {
                assert!(t.is_sparse());
                match t.nnz() {
                    Ok(n) => assert_eq!(n, 3, "identity CSR → nnz=3, got {}", n),
                    Err(e) => panic!("nnz on CSR failed: {}", e),
                }
            },
            Err(e) => panic!("sparse_csr identity failed: {}", e),
        }
    }

    #[test]
    fn test_sparse_csr_to_dense_roundtrip() {
        // 2×4 matrix with two non-zeros: (0,1)=3.0, (1,3)=7.0.
        //   row_ptr     = [0, 1, 2]
        //   col_indices = [1, 3]
        match Tensor::sparse_csr(vec![0, 1, 2], vec![1, 3], vec![3.0, 7.0], vec![2, 4]) {
            Ok(sparse) => match sparse.to_dense() {
                Ok(dense) => {
                    let v = to_vec(&dense);
                    assert_eq!(v.len(), 8);
                    assert!((v[1] - 3.0).abs() < 1e-6, "pos (0,1) must be 3.0, got {}", v[1]);
                    assert!((v[7] - 7.0).abs() < 1e-6, "pos (1,3) must be 7.0, got {}", v[7]);
                },
                Err(e) => panic!("to_dense CSR failed: {}", e),
            },
            Err(e) => panic!("sparse_csr 2×4 failed: {}", e),
        }
    }

    #[test]
    fn test_sparse_csr_empty_row() {
        // 3×3 matrix with only row 1 having data.
        //   row_ptr = [0, 0, 1, 1]
        //   col_indices = [2]
        //   values = [9.0]
        match Tensor::sparse_csr(vec![0, 0, 1, 1], vec![2], vec![9.0], vec![3, 3]) {
            Ok(t) => match t.nnz() {
                Ok(n) => assert_eq!(n, 1, "single element CSR → nnz=1"),
                Err(e) => panic!("nnz CSR empty row failed: {}", e),
            },
            Err(e) => panic!("sparse_csr empty row failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // threshold behaviour
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_to_sparse_threshold_keeps_large_values() {
        // With threshold=0.9, only 1.0 survives (0.5 < 0.9).
        let t = f32_tensor(vec![0.5, 1.0, 0.3, 0.0], vec![2, 2]);
        match t.to_sparse(0.9) {
            Ok(sparse) => match sparse.nnz() {
                Ok(n) => assert_eq!(n, 1, "only value ≥ 0.9 → nnz=1, got {}", n),
                Err(e) => panic!("nnz after threshold failed: {}", e),
            },
            Err(e) => panic!("to_sparse with threshold 0.9 failed: {}", e),
        }
    }

    #[test]
    fn test_to_sparse_zero_threshold_keeps_all_nonzeros() {
        // threshold=0.0 → every non-zero value is kept.
        let data = vec![0.0, 1.0, 2.0, 0.0, 3.0, 0.0];
        let t = f32_tensor(data, vec![2, 3]);
        match t.to_sparse(0.0) {
            Ok(sparse) => match sparse.nnz() {
                Ok(n) => assert_eq!(n, 3, "3 nonzeros with threshold=0, got {}", n),
                Err(e) => panic!("nnz zero-threshold failed: {}", e),
            },
            Err(e) => panic!("to_sparse zero-threshold failed: {}", e),
        }
    }
}
