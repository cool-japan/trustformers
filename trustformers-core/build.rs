//! Build script for `trustformers-core`.
//!
//! # Pure-Rust policy (COOLJAPAN)
//!
//! This crate deliberately links **no** C/C++/Fortran system libraries in its
//! default configuration. In particular Apple's `Accelerate` framework (a
//! C/Fortran BLAS/LAPACK implementation) is *not* linked: dense linear algebra
//! goes through `oxiblas` (see `layers/linear.rs` and `kernels/simd/matrix_ops.rs`,
//! which call `oxiblas_blas::level3::gemm`), and GPU work on macOS goes through
//! the `metal` feature (`oxicuda-metal` + hand-written MSL kernels), which links
//! only the Metal framework via the `metal` crate's own build configuration.
//!
//! If a future code path genuinely needs Accelerate, gate it behind an
//! off-by-default `accelerate` cargo feature and emit the link directive from
//! inside that gate — never unconditionally.

fn main() {
    // Re-run only when this script itself changes; no link directives are emitted.
    println!("cargo:rerun-if-changed=build.rs");

    // Off-by-default escape hatch: enabling the `accelerate` feature opts a build
    // out of the pure-Rust policy explicitly and audibly. No code in this crate
    // currently requires it (a workspace grep finds no `cblas_`/`LAPACKE_`/vecLib
    // symbol use), so the feature exists purely so downstream consumers that
    // vendor Accelerate-dependent code can re-add the link without patching.
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
