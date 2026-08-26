//! Small dense linear-algebra kernels used by the second-order optimizers.
//!
//! These are deliberately *small-matrix* routines: LoRA factors, per-layer Gram
//! matrices and tangent-space Gauss-Newton blocks are at most a few hundred rows,
//! so an O(n³) Jacobi sweep is both fast enough and numerically excellent.
//!
//! Everything here is pure Rust with no BLAS/LAPACK dependency, and every routine
//! returns a real decomposition — none of them approximate a result with a
//! placeholder.

use trustformers_core::errors::{Result, TrustformersError};

/// Maximum number of Jacobi sweeps before the iteration is declared converged.
///
/// Classical Jacobi converges quadratically; 60 sweeps is far beyond what any
/// realistic input needs and exists purely to bound the loop.
const MAX_SWEEPS: usize = 60;

/// Relative off-diagonal threshold below which a Jacobi sweep stops rotating.
const JACOBI_TOL: f64 = 1e-12;

/// A dense row-major matrix of `f64`, used as the working type for the kernels below.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl DenseMatrix {
    /// Builds a matrix from row-major data.
    ///
    /// # Errors
    ///
    /// Returns an error when `data.len() != rows * cols`.
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(TrustformersError::invalid_input(format!(
                "dense matrix expects {} elements for a {rows}x{cols} shape, got {}",
                rows * cols,
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// Builds a matrix from row-major `f32` data.
    ///
    /// # Errors
    ///
    /// See [`DenseMatrix::from_row_major`].
    pub fn from_f32(rows: usize, cols: usize, data: &[f32]) -> Result<Self> {
        Self::from_row_major(rows, cols, data.iter().map(|&v| v as f64).collect())
    }

    /// The `n x n` identity.
    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for (i, slot) in data.iter_mut().step_by(n + 1).enumerate() {
            debug_assert!(i < n);
            *slot = 1.0;
        }
        Self {
            rows: n,
            cols: n,
            data,
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row-major backing storage.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Row-major storage converted to `f32`.
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&v| v as f32).collect()
    }

    /// Element at `(row, col)`; returns `0.0` for out-of-range indices, which cannot
    /// happen for the internally generated indices used by the kernels below.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data.get(row * self.cols + col).copied().unwrap_or(0.0)
    }

    fn set(&mut self, row: usize, col: usize, value: f64) {
        if let Some(slot) = self.data.get_mut(row * self.cols + col) {
            *slot = value;
        }
    }

    /// The main diagonal (length `min(rows, cols)`).
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.rows.min(self.cols)).map(|i| self.get(i, i)).collect()
    }

    /// Matrix product `self * other`.
    ///
    /// # Errors
    ///
    /// Returns an error when the inner dimensions disagree.
    pub fn matmul(&self, other: &DenseMatrix) -> Result<DenseMatrix> {
        if self.cols != other.rows {
            return Err(TrustformersError::invalid_input(format!(
                "cannot multiply {}x{} by {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut out = vec![0.0; self.rows * other.cols];
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    out[i * other.cols + j] += a * other.get(k, j);
                }
            }
        }
        DenseMatrix::from_row_major(self.rows, other.cols, out)
    }

    /// The transpose.
    pub fn transpose(&self) -> DenseMatrix {
        let mut out = vec![0.0; self.data.len()];
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[j * self.rows + i] = self.get(i, j);
            }
        }
        DenseMatrix {
            rows: self.cols,
            cols: self.rows,
            data: out,
        }
    }
}

/// Eigen-decomposition of a real symmetric matrix.
#[derive(Debug, Clone)]
pub struct SymmetricEigen {
    /// Eigenvalues in descending order.
    pub values: Vec<f64>,
    /// Eigenvectors as columns of an orthogonal matrix, matching `values` order.
    pub vectors: DenseMatrix,
}

/// Computes the eigen-decomposition of a real symmetric matrix with the cyclic
/// Jacobi method.
///
/// The input is symmetrised (`(A + Aᵀ)/2`) before the sweeps, so a matrix that is
/// symmetric only up to rounding is handled cleanly.
///
/// # Errors
///
/// Returns an error when the matrix is not square.
pub fn symmetric_eigen(matrix: &DenseMatrix) -> Result<SymmetricEigen> {
    let n = matrix.rows();
    if matrix.cols() != n {
        return Err(TrustformersError::invalid_input(format!(
            "symmetric eigendecomposition needs a square matrix, got {}x{}",
            matrix.rows(),
            matrix.cols()
        )));
    }
    if n == 0 {
        return Ok(SymmetricEigen {
            values: Vec::new(),
            vectors: DenseMatrix::identity(0),
        });
    }

    // Work on a symmetrised copy so tiny asymmetries in the caller's Gram matrix
    // do not break the rotation invariants.
    let mut a = DenseMatrix::identity(n);
    for i in 0..n {
        for j in 0..n {
            a.set(i, j, 0.5 * (matrix.get(i, j) + matrix.get(j, i)));
        }
    }
    let mut v = DenseMatrix::identity(n);

    let frobenius: f64 = a.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
    let threshold = JACOBI_TOL * frobenius.max(1.0);

    for _ in 0..MAX_SWEEPS {
        let mut off_diagonal = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diagonal += a.get(i, j) * a.get(i, j);
            }
        }
        if off_diagonal.sqrt() <= threshold {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a.get(p, q);
                if apq.abs() <= threshold {
                    continue;
                }
                let app = a.get(p, p);
                let aqq = a.get(q, q);
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                for k in 0..n {
                    let akp = a.get(k, p);
                    let akq = a.get(k, q);
                    a.set(k, p, c * akp - s * akq);
                    a.set(k, q, s * akp + c * akq);
                }
                for k in 0..n {
                    let apk = a.get(p, k);
                    let aqk = a.get(q, k);
                    a.set(p, k, c * apk - s * aqk);
                    a.set(q, k, s * apk + c * aqk);
                }
                for k in 0..n {
                    let vkp = v.get(k, p);
                    let vkq = v.get(k, q);
                    v.set(k, p, c * vkp - s * vkq);
                    v.set(k, q, s * vkp + c * vkq);
                }
            }
        }
    }

    // Sort eigenpairs by descending eigenvalue.
    let mut order: Vec<usize> = (0..n).collect();
    let diag: Vec<f64> = (0..n).map(|i| a.get(i, i)).collect();
    order.sort_by(|&x, &y| diag[y].partial_cmp(&diag[x]).unwrap_or(std::cmp::Ordering::Equal));

    let values: Vec<f64> = order.iter().map(|&i| diag[i]).collect();
    let mut vectors = DenseMatrix::identity(n);
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..n {
            vectors.set(row, new_col, v.get(row, old_col));
        }
    }

    Ok(SymmetricEigen { values, vectors })
}

/// A thin singular-value decomposition `M = U · diag(S) · Vᵀ`.
#[derive(Debug, Clone)]
pub struct Svd {
    /// Left singular vectors, `rows x k` where `k = min(rows, cols)`.
    pub u: DenseMatrix,
    /// Singular values in descending order, length `k`.
    pub s: Vec<f64>,
    /// Right singular vectors, `cols x k` (columns are the right vectors).
    pub v: DenseMatrix,
}

/// Computes the thin SVD of a dense matrix with the one-sided Jacobi method.
///
/// One-sided Jacobi orthogonalises the *columns* of `M` by accumulating plane
/// rotations into `V`; the resulting column norms are the singular values and the
/// normalised columns are `U`. It is backward stable and needs no eigen-solver on
/// `MᵀM`, so small singular values keep their relative accuracy.
///
/// # Errors
///
/// Returns an error when the matrix has no rows or columns.
pub fn jacobi_svd(matrix: &DenseMatrix) -> Result<Svd> {
    let (m, n) = (matrix.rows(), matrix.cols());
    if m == 0 || n == 0 {
        return Err(TrustformersError::invalid_input(
            "SVD requires a matrix with at least one row and one column".to_string(),
        ));
    }

    // One-sided Jacobi works on the tall case; transpose and swap U/V otherwise.
    if m < n {
        let transposed = matrix.transpose();
        let svd = jacobi_svd(&transposed)?;
        return Ok(Svd {
            u: svd.v,
            s: svd.s,
            v: svd.u,
        });
    }

    let mut work = matrix.clone();
    let mut v = DenseMatrix::identity(n);

    for _ in 0..MAX_SWEEPS {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for k in 0..m {
                    let wp = work.get(k, p);
                    let wq = work.get(k, q);
                    alpha += wp * wp;
                    beta += wq * wq;
                    gamma += wp * wq;
                }
                if gamma.abs() <= JACOBI_TOL * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;

                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for k in 0..m {
                    let wp = work.get(k, p);
                    let wq = work.get(k, q);
                    work.set(k, p, c * wp - s * wq);
                    work.set(k, q, s * wp + c * wq);
                }
                for k in 0..n {
                    let vp = v.get(k, p);
                    let vq = v.get(k, q);
                    v.set(k, p, c * vp - s * vq);
                    v.set(k, q, s * vp + c * vq);
                }
            }
        }
        if !rotated {
            break;
        }
    }

    // Column norms are the singular values.
    let mut singular: Vec<(f64, usize)> = (0..n)
        .map(|j| {
            let norm = (0..m).map(|i| work.get(i, j) * work.get(i, j)).sum::<f64>().sqrt();
            (norm, j)
        })
        .collect();
    singular.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut u = DenseMatrix::from_row_major(m, n, vec![0.0; m * n])?;
    let mut v_sorted = DenseMatrix::from_row_major(n, n, vec![0.0; n * n])?;
    let mut s = Vec::with_capacity(n);
    for (new_col, &(norm, old_col)) in singular.iter().enumerate() {
        s.push(norm);
        if norm > 0.0 {
            for row in 0..m {
                u.set(row, new_col, work.get(row, old_col) / norm);
            }
        }
        for row in 0..n {
            v_sorted.set(row, new_col, v.get(row, old_col));
        }
    }

    Ok(Svd { u, s, v: v_sorted })
}

/// Computes `A^(-1/2)` for a symmetric positive semi-definite matrix, flooring the
/// eigenvalues at `eps` so the inverse stays finite for rank-deficient input.
///
/// This is the preconditioner shape used by LoRA-RITE and by Kronecker-factored
/// second-order methods.
///
/// # Errors
///
/// Returns an error when the matrix is not square.
pub fn inverse_sqrt_psd(matrix: &DenseMatrix, eps: f64) -> Result<DenseMatrix> {
    let eigen = symmetric_eigen(matrix)?;
    let n = matrix.rows();
    let mut scaled = DenseMatrix::from_row_major(n, n, vec![0.0; n * n])?;
    for (col, &value) in eigen.values.iter().enumerate() {
        let inv_sqrt = 1.0 / value.max(eps).sqrt();
        for row in 0..n {
            scaled.set(row, col, eigen.vectors.get(row, col) * inv_sqrt);
        }
    }
    scaled.matmul(&eigen.vectors.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b}");
    }

    #[test]
    fn symmetric_eigen_matches_hand_computed_2x2() {
        // [[2, 1], [1, 2]] has eigenvalues 3 and 1.
        let m = DenseMatrix::from_row_major(2, 2, vec![2.0, 1.0, 1.0, 2.0]).expect("matrix");
        let eigen = symmetric_eigen(&m).expect("eigen");
        assert_close(eigen.values[0], 3.0, 1e-10, "largest eigenvalue");
        assert_close(eigen.values[1], 1.0, 1e-10, "smallest eigenvalue");
    }

    #[test]
    fn symmetric_eigen_reconstructs_the_matrix() {
        let m =
            DenseMatrix::from_row_major(3, 3, vec![4.0, 1.0, -2.0, 1.0, 3.0, 0.5, -2.0, 0.5, 6.0])
                .expect("matrix");
        let eigen = symmetric_eigen(&m).expect("eigen");
        let mut lambda = DenseMatrix::from_row_major(3, 3, vec![0.0; 9]).expect("lambda");
        for (i, &value) in eigen.values.iter().enumerate() {
            lambda.set(i, i, value);
        }
        let reconstructed = eigen
            .vectors
            .matmul(&lambda)
            .expect("v*lambda")
            .matmul(&eigen.vectors.transpose())
            .expect("*vt");
        for i in 0..3 {
            for j in 0..3 {
                assert_close(reconstructed.get(i, j), m.get(i, j), 1e-9, "reconstruction");
            }
        }
    }

    #[test]
    fn jacobi_svd_matches_hand_computed_diagonal() {
        let m = DenseMatrix::from_row_major(2, 2, vec![3.0, 0.0, 0.0, -4.0]).expect("matrix");
        let svd = jacobi_svd(&m).expect("svd");
        assert_close(svd.s[0], 4.0, 1e-10, "largest singular value");
        assert_close(svd.s[1], 3.0, 1e-10, "smallest singular value");
    }

    #[test]
    fn jacobi_svd_reconstructs_a_rectangular_matrix() {
        let m =
            DenseMatrix::from_row_major(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("matrix");
        let svd = jacobi_svd(&m).expect("svd");
        let mut sigma = DenseMatrix::from_row_major(2, 2, vec![0.0; 4]).expect("sigma");
        for (i, &value) in svd.s.iter().enumerate() {
            sigma.set(i, i, value);
        }
        let reconstructed =
            svd.u.matmul(&sigma).expect("u*s").matmul(&svd.v.transpose()).expect("*vt");
        for i in 0..3 {
            for j in 0..2 {
                assert_close(reconstructed.get(i, j), m.get(i, j), 1e-9, "reconstruction");
            }
        }
        // Frobenius norm identity: sum of squared singular values == ||M||_F^2.
        let frob_sq: f64 = m.as_slice().iter().map(|x| x * x).sum();
        let s_sq: f64 = svd.s.iter().map(|x| x * x).sum();
        assert_close(s_sq, frob_sq, 1e-9, "singular value energy");
    }

    #[test]
    fn jacobi_svd_handles_wide_matrices() {
        let m =
            DenseMatrix::from_row_major(2, 3, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0]).expect("matrix");
        let svd = jacobi_svd(&m).expect("svd");
        assert_eq!(svd.u.rows(), 2);
        assert_close(svd.s[0], 2.0, 1e-10, "largest singular value");
        assert_close(svd.s[1], 1.0, 1e-10, "second singular value");
    }

    #[test]
    fn inverse_sqrt_psd_squares_back_to_the_inverse() {
        // diag(4, 9) -> inverse sqrt is diag(1/2, 1/3).
        let m = DenseMatrix::from_row_major(2, 2, vec![4.0, 0.0, 0.0, 9.0]).expect("matrix");
        let inv_sqrt = inverse_sqrt_psd(&m, 1e-12).expect("inv sqrt");
        assert_close(inv_sqrt.get(0, 0), 0.5, 1e-10, "first");
        assert_close(inv_sqrt.get(1, 1), 1.0 / 3.0, 1e-10, "second");
        assert_close(inv_sqrt.get(0, 1), 0.0, 1e-10, "off diagonal");
    }

    #[test]
    fn inverse_sqrt_psd_floors_a_singular_direction() {
        let m = DenseMatrix::from_row_major(2, 2, vec![4.0, 0.0, 0.0, 0.0]).expect("matrix");
        let inv_sqrt = inverse_sqrt_psd(&m, 1e-4).expect("inv sqrt");
        assert!(
            inv_sqrt.get(1, 1).is_finite(),
            "floored direction must stay finite"
        );
        assert_close(inv_sqrt.get(1, 1), 100.0, 1e-6, "floored direction");
    }

    #[test]
    fn non_square_symmetric_eigen_is_rejected() {
        let m = DenseMatrix::from_row_major(2, 3, vec![0.0; 6]).expect("matrix");
        assert!(symmetric_eigen(&m).is_err());
    }
}
