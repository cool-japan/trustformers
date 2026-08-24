//! HiPPO initialisation and continuous-to-discrete conversion for S4.
//!
//! # What changed and why
//!
//! [`Discretization`] used to advertise four methods and compute one: `ZOH`,
//! `Bilinear` and `BackwardEuler` all returned the *forward Euler* step
//! `Ā = I + Δ·A`, `B̄ = Δ·B`, each behind a comment saying it was "simplified".
//! A caller selecting `bilinear` in its config got Euler and was told it got
//! Tustin. The four are now genuinely different:
//!
//! | Method | `Ā` | `B̄` |
//! |--------|-----|-----|
//! | `Euler` | `I + Δ·A` | `Δ·B` |
//! | `BackwardEuler` | `(I − Δ·A)⁻¹` | `(I − Δ·A)⁻¹ Δ·B` |
//! | `Bilinear` | `(I − Δ/2·A)⁻¹(I + Δ/2·A)` | `(I − Δ/2·A)⁻¹ Δ·B` |
//! | `ZOH` | `exp(Δ·A)` | `A⁻¹(exp(Δ·A) − I)B` |
//!
//! `ZOH` is computed without ever inverting `A`, through the standard augmented
//! exponential `exp(Δ·[[A, B], [0, 0]]) = [[Ā, B̄], [0, 1]]`, so a singular `A`
//! (the HiPPO-LEGS matrix is skew-symmetric and singular for odd `n`) is handled
//! rather than being a hidden precondition.
//!
//! The two implicit methods need a linear solve; it is a Gauss–Jordan inversion
//! with partial pivoting that reports a singular matrix as an error instead of
//! returning garbage.

use scirs2_core::ndarray::{Array1, Array2}; // SciRS2 Integration Policy
use std::f32::consts::PI;
use trustformers_core::errors::{runtime_error, Result};

/// HiPPO matrix initialization methods
/// Reference: "HiPPO: Recurrent Memory with Optimal Polynomial Projections"
#[derive(Debug, Clone)]
pub enum HiPPOMatrix {
    /// Legendre measure (uniform on [-1, 1])
    LEGS,
    /// Laguerre measure (exponential decay on [0, ∞))
    LEGT,
    /// Laguerre (translated)
    LAGT,
    /// Fourier basis
    Fourier,
    /// Random initialization
    Random,
}

impl HiPPOMatrix {
    /// Initialize HiPPO matrix A of shape (N, N)
    pub fn initialize(&self, n: usize) -> Array2<f32> {
        match self {
            HiPPOMatrix::LEGS => self.init_legs(n),
            HiPPOMatrix::LEGT => self.init_legt(n),
            HiPPOMatrix::LAGT => self.init_lagt(n),
            HiPPOMatrix::Fourier => self.init_fourier(n),
            HiPPOMatrix::Random => self.init_random(n),
        }
    }

    fn init_legs(&self, n: usize) -> Array2<f32> {
        // Legendre (LEGS) matrix
        let mut a = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let val = if i == j {
                    0.0
                } else if i > j {
                    (2.0 * i as f32 + 1.0).sqrt() * (2.0 * j as f32 + 1.0).sqrt()
                } else {
                    0.0
                };
                a[[i, j]] = val;
            }
        }
        // Make skew-symmetric
        &a - &a.t()
    }

    fn init_legt(&self, n: usize) -> Array2<f32> {
        // Laguerre (LEGT) matrix
        let mut a = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    a[[i, j]] = 1.0;
                } else if i == j {
                    a[[i, j]] = -(2.0 * i as f32 + 1.0) / 2.0;
                }
            }
        }
        a
    }

    fn init_lagt(&self, n: usize) -> Array2<f32> {
        // Translated Laguerre (LAGT) matrix
        let mut a = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    a[[i, j]] = (-1.0_f32).powi((i - j) as i32);
                } else if i == j {
                    a[[i, j]] = -0.5;
                }
            }
        }
        a
    }

    fn init_fourier(&self, n: usize) -> Array2<f32> {
        // Fourier basis matrix
        let mut a = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                    a[[i, j]] = sign * PI * (i as f32 - j as f32);
                }
            }
        }
        a
    }

    fn init_random(&self, n: usize) -> Array2<f32> {
        // Random skew-symmetric initialization
        use scirs2_core::random::*; // SciRS2 Integration Policy
        let mut rng = thread_rng();
        let mut a = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..i {
                let val = rng.random_range(-1.0..1.0);
                a[[i, j]] = val;
                a[[j, i]] = -val; // Skew-symmetric
            }
        }
        a
    }
}

/// Invert a square matrix by Gauss–Jordan elimination with partial pivoting.
///
/// # Errors
///
/// Fails when the matrix is not square or is numerically singular — reported
/// rather than papered over with a pseudo-inverse the caller did not ask for.
pub fn invert(matrix: &Array2<f32>) -> Result<Array2<f32>> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(runtime_error(format!(
            "matrix inversion needs a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        )));
    }
    // Work in f64: the state matrices are ill-conditioned enough that an f32
    // elimination loses accuracy the discretisation then propagates every step.
    let mut work = vec![0.0_f64; n * 2 * n];
    for row in 0..n {
        for col in 0..n {
            work[row * 2 * n + col] = f64::from(matrix[[row, col]]);
        }
        work[row * 2 * n + n + row] = 1.0;
    }

    for column in 0..n {
        let mut pivot_row = column;
        let mut best = work[column * 2 * n + column].abs();
        for row in (column + 1)..n {
            let candidate = work[row * 2 * n + column].abs();
            if candidate > best {
                best = candidate;
                pivot_row = row;
            }
        }
        if best < 1e-12 {
            return Err(runtime_error(format!(
                "matrix is singular to working precision at column {column}; no discrete-time \
                 equivalent exists for this timestep"
            )));
        }
        if pivot_row != column {
            for col in 0..(2 * n) {
                work.swap(column * 2 * n + col, pivot_row * 2 * n + col);
            }
        }
        let pivot = work[column * 2 * n + column];
        for col in 0..(2 * n) {
            work[column * 2 * n + col] /= pivot;
        }
        for row in 0..n {
            if row == column {
                continue;
            }
            let factor = work[row * 2 * n + column];
            if factor == 0.0 {
                continue;
            }
            for col in 0..(2 * n) {
                work[row * 2 * n + col] -= factor * work[column * 2 * n + col];
            }
        }
    }

    let mut inverse = Array2::<f32>::zeros((n, n));
    for row in 0..n {
        for col in 0..n {
            inverse[[row, col]] = work[row * 2 * n + n + col] as f32;
        }
    }
    Ok(inverse)
}

/// Matrix exponential by scaling and squaring with a truncated Taylor series.
///
/// The argument is scaled down until its infinity norm is below 1/2, where the
/// Taylor series converges quickly, then squared back up.
pub fn matrix_exponential(matrix: &Array2<f32>) -> Result<Array2<f32>> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(runtime_error(format!(
            "matrix exponential needs a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        )));
    }
    let norm = (0..n)
        .map(|row| (0..n).map(|col| f64::from(matrix[[row, col]]).abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    if !norm.is_finite() {
        return Err(runtime_error(
            "matrix exponential requires finite entries".to_string(),
        ));
    }
    let squarings = if norm > 0.5 {
        (norm / 0.5).log2().ceil().max(0.0).min(60.0) as u32
    } else {
        0
    };
    let scale = 2.0_f64.powi(-(squarings as i32));

    let mut scaled = vec![0.0_f64; n * n];
    for row in 0..n {
        for col in 0..n {
            scaled[row * n + col] = f64::from(matrix[[row, col]]) * scale;
        }
    }

    // exp(X) = sum_k X^k / k!
    let mut result = vec![0.0_f64; n * n];
    let mut term = vec![0.0_f64; n * n];
    for index in 0..n {
        result[index * n + index] = 1.0;
        term[index * n + index] = 1.0;
    }
    let mut next = vec![0.0_f64; n * n];
    for k in 1..=18_u32 {
        multiply(&term, &scaled, &mut next, n);
        let inverse_k = 1.0 / f64::from(k);
        for (slot, value) in term.iter_mut().zip(next.iter()) {
            *slot = value * inverse_k;
        }
        let mut magnitude = 0.0_f64;
        for (slot, value) in result.iter_mut().zip(term.iter()) {
            *slot += value;
            magnitude = magnitude.max(value.abs());
        }
        if magnitude < 1e-18 {
            break;
        }
    }

    for _ in 0..squarings {
        multiply(&result, &result, &mut next, n);
        result.copy_from_slice(&next);
    }

    let mut out = Array2::<f32>::zeros((n, n));
    for row in 0..n {
        for col in 0..n {
            out[[row, col]] = result[row * n + col] as f32;
        }
    }
    Ok(out)
}

/// `out = lhs * rhs` for row-major `n x n` matrices.
fn multiply(lhs: &[f64], rhs: &[f64], out: &mut [f64], n: usize) {
    for value in out.iter_mut() {
        *value = 0.0;
    }
    for row in 0..n {
        for inner in 0..n {
            let left = lhs[row * n + inner];
            if left == 0.0 {
                continue;
            }
            for col in 0..n {
                out[row * n + col] += left * rhs[inner * n + col];
            }
        }
    }
}

/// Discretization methods for continuous-time to discrete-time conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Discretization {
    /// Zero-order hold: `Ā = exp(Δ·A)`.
    ZOH,
    /// Bilinear transform (Tustin's method).
    Bilinear,
    /// Forward Euler.
    Euler,
    /// Backward Euler.
    BackwardEuler,
}

impl Discretization {
    /// Parse a config's `discretization` string, defaulting to zero-order hold.
    pub fn from_name(name: &str) -> Self {
        match name {
            "bilinear" => Discretization::Bilinear,
            "euler" => Discretization::Euler,
            "backward_euler" => Discretization::BackwardEuler,
            _ => Discretization::ZOH,
        }
    }

    /// Discretize continuous-time `(A, B)` to discrete-time `(Ā, B̄)`.
    ///
    /// # Errors
    ///
    /// Fails when an implicit method's `(I − cΔ·A)` factor is singular, or when
    /// the matrices are not square.
    pub fn discretize(
        &self,
        a: &Array2<f32>,
        b: &Array1<f32>,
        dt: f32,
    ) -> Result<(Array2<f32>, Array1<f32>)> {
        let n = a.nrows();
        if a.ncols() != n || b.len() != n {
            return Err(runtime_error(format!(
                "discretization needs a square A and a matching B, got A {}x{} and B of length {}",
                a.nrows(),
                a.ncols(),
                b.len()
            )));
        }
        let eye = Array2::<f32>::eye(n);
        match self {
            Discretization::Euler => Ok((&eye + &(a * dt), b * dt)),
            Discretization::BackwardEuler => {
                let resolvent = invert(&(&eye - &(a * dt)))?;
                let b_bar = resolvent.dot(&(b * dt));
                Ok((resolvent, b_bar))
            },
            Discretization::Bilinear => {
                let half = dt / 2.0;
                let resolvent = invert(&(&eye - &(a * half)))?;
                let a_bar = resolvent.dot(&(&eye + &(a * half)));
                let b_bar = resolvent.dot(&(b * dt));
                Ok((a_bar, b_bar))
            },
            Discretization::ZOH => {
                // exp(Δ·[[A, B], [0, 0]]) = [[exp(Δ·A), A⁻¹(exp(Δ·A) − I)B], [0, 1]]
                // — the augmented form avoids inverting a possibly singular A.
                let mut augmented = Array2::<f32>::zeros((n + 1, n + 1));
                for row in 0..n {
                    for col in 0..n {
                        augmented[[row, col]] = a[[row, col]] * dt;
                    }
                    augmented[[row, n]] = b[row] * dt;
                }
                let exponential = matrix_exponential(&augmented)?;
                let mut a_bar = Array2::<f32>::zeros((n, n));
                let mut b_bar = Array1::<f32>::zeros(n);
                for row in 0..n {
                    for col in 0..n {
                        a_bar[[row, col]] = exponential[[row, col]];
                    }
                    b_bar[row] = exponential[[row, n]];
                }
                Ok((a_bar, b_bar))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_recovers_the_identity() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0])
                .expect("matrix must build");
        let inverse = invert(&matrix).expect("a well-conditioned matrix must invert");
        let product = matrix.dot(&inverse);
        for row in 0..3 {
            for col in 0..3 {
                let want = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (product[[row, col]] - want).abs() < 1e-4,
                    "A·A⁻¹ must be the identity, got {product:?}"
                );
            }
        }
    }

    #[test]
    fn invert_reports_a_singular_matrix() {
        let matrix = Array2::<f32>::zeros((2, 2));
        invert(&matrix).expect_err("the zero matrix has no inverse");
    }

    #[test]
    fn matrix_exponential_of_zero_is_the_identity() {
        let result = matrix_exponential(&Array2::<f32>::zeros((4, 4))).expect("must compute");
        for row in 0..4 {
            for col in 0..4 {
                let want = if row == col { 1.0 } else { 0.0 };
                assert!((result[[row, col]] - want).abs() < 1e-6, "{result:?}");
            }
        }
    }

    #[test]
    fn matrix_exponential_of_a_diagonal_is_elementwise() {
        let mut matrix = Array2::<f32>::zeros((3, 3));
        matrix[[0, 0]] = 1.0;
        matrix[[1, 1]] = -2.0;
        matrix[[2, 2]] = 0.5;
        let result = matrix_exponential(&matrix).expect("must compute");
        for (index, value) in [1.0_f32, -2.0, 0.5].into_iter().enumerate() {
            assert!(
                (result[[index, index]] - value.exp()).abs() < 1e-4,
                "diagonal {index}: {} vs {}",
                result[[index, index]],
                value.exp()
            );
        }
    }

    /// exp of a nilpotent Jordan block has a closed form: `[[1, t], [0, 1]]`.
    #[test]
    fn matrix_exponential_handles_a_nilpotent_block() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![0.0, 3.0, 0.0, 0.0]).expect("matrix must build");
        let result = matrix_exponential(&matrix).expect("must compute");
        assert!((result[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((result[[0, 1]] - 3.0).abs() < 1e-5);
        assert!(result[[1, 0]].abs() < 1e-5);
        assert!((result[[1, 1]] - 1.0).abs() < 1e-5);
    }

    /// The four methods are genuinely four methods.
    ///
    /// Regression test: `ZOH`, `Bilinear` and `BackwardEuler` all returned the
    /// forward-Euler step `I + Δ·A`, so selecting `bilinear` in a config silently
    /// got Euler.
    #[test]
    fn the_four_methods_produce_four_different_operators() {
        let a =
            Array2::from_shape_vec((2, 2), vec![-0.5, 1.0, -1.0, -0.5]).expect("matrix must build");
        let b = Array1::from_vec(vec![1.0, -1.0]);
        let dt = 0.4_f32;

        let mut operators = Vec::new();
        for method in [
            Discretization::Euler,
            Discretization::BackwardEuler,
            Discretization::Bilinear,
            Discretization::ZOH,
        ] {
            let (a_bar, b_bar) = method.discretize(&a, &b, dt).expect("must discretize");
            operators.push((format!("{method:?}"), a_bar, b_bar));
        }
        for i in 0..operators.len() {
            for j in (i + 1)..operators.len() {
                let (left, right) = (&operators[i], &operators[j]);
                let difference: f32 =
                    left.1.iter().zip(right.1.iter()).map(|(a, b)| (a - b).abs()).sum();
                assert!(
                    difference > 1e-3,
                    "{} and {} produced the same Ā ({difference}): {:?}",
                    left.0,
                    right.0,
                    left.1
                );
            }
        }
    }

    /// Backward Euler really is the resolvent: `(I − Δ·A)·Ā = I`.
    #[test]
    fn backward_euler_is_the_resolvent() {
        let a =
            Array2::from_shape_vec((2, 2), vec![-1.0, 0.5, 0.25, -2.0]).expect("matrix must build");
        let b = Array1::from_vec(vec![1.0, 1.0]);
        let dt = 0.3_f32;
        let (a_bar, _) =
            Discretization::BackwardEuler.discretize(&a, &b, dt).expect("must discretize");
        let check = (&Array2::<f32>::eye(2) - &(&a * dt)).dot(&a_bar);
        for row in 0..2 {
            for col in 0..2 {
                let want = if row == col { 1.0 } else { 0.0 };
                assert!((check[[row, col]] - want).abs() < 1e-4, "{check:?}");
            }
        }
    }

    /// Zero-order hold really is the matrix exponential, on a case with a closed
    /// form: a diagonal `A` gives `Ā = diag(exp(Δ·aᵢ))`.
    #[test]
    fn zoh_is_the_matrix_exponential() {
        let mut a = Array2::<f32>::zeros((2, 2));
        a[[0, 0]] = -2.0;
        a[[1, 1]] = 0.5;
        let b = Array1::from_vec(vec![1.0, 1.0]);
        let dt = 0.25_f32;
        let (a_bar, b_bar) = Discretization::ZOH.discretize(&a, &b, dt).expect("must discretize");
        assert!(
            (a_bar[[0, 0]] - (-2.0_f32 * dt).exp()).abs() < 1e-5,
            "{a_bar:?}"
        );
        assert!(
            (a_bar[[1, 1]] - (0.5_f32 * dt).exp()).abs() < 1e-5,
            "{a_bar:?}"
        );
        assert!(a_bar[[0, 1]].abs() < 1e-6);
        // For a diagonal A, B̄ᵢ = (exp(Δ·aᵢ) − 1)/aᵢ · bᵢ.
        for (index, value) in [-2.0_f32, 0.5].into_iter().enumerate() {
            let want = ((value * dt).exp() - 1.0) / value;
            assert!(
                (b_bar[index] - want).abs() < 1e-5,
                "B̄[{index}] = {} vs {want}",
                b_bar[index]
            );
        }
    }

    /// A singular `A` is not a precondition of zero-order hold here.
    #[test]
    fn zoh_handles_a_singular_state_matrix() {
        let a = Array2::<f32>::zeros((3, 3));
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let dt = 0.1_f32;
        let (a_bar, b_bar) = Discretization::ZOH.discretize(&a, &b, dt).expect("must discretize");
        // With A = 0: Ā = I and B̄ = Δ·B.
        for index in 0..3 {
            assert!((a_bar[[index, index]] - 1.0).abs() < 1e-6);
            assert!((b_bar[index] - b[index] * dt).abs() < 1e-6);
        }
    }

    #[test]
    fn discretization_names_map_to_methods() {
        assert_eq!(Discretization::from_name("zoh"), Discretization::ZOH);
        assert_eq!(
            Discretization::from_name("bilinear"),
            Discretization::Bilinear
        );
        assert_eq!(Discretization::from_name("euler"), Discretization::Euler);
        assert_eq!(
            Discretization::from_name("backward_euler"),
            Discretization::BackwardEuler
        );
        assert_eq!(
            Discretization::from_name("something else"),
            Discretization::ZOH,
            "an unknown name falls back to the documented default"
        );
    }
}
