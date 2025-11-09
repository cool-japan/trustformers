/// Second-order optimization methods
///
/// This module provides advanced optimization algorithms that use second-order information
/// (Hessian or Fisher Information Matrix) to achieve faster convergence than first-order methods.
pub mod lbfgs;
pub mod newton_cg;
pub mod self_scaled;

// Re-export for convenience
pub use lbfgs::{LineSearchMethod, LBFGS};
pub use newton_cg::NewtonCG;
pub use self_scaled::{SSBFGSConfig, SSBFGSStats, SSBroyden, SSBroydenConfig, SSBFGS};
