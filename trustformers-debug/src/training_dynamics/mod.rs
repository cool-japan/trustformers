//! Auto-generated module structure

pub mod functions;
pub mod trait_impls;
pub mod types;
pub mod types_3;
pub mod types_4;

// Re-export all types
pub use types::*;
pub use types_3::*;
pub use types_4::*;

#[cfg(test)]
#[path = "../training_dynamics_tests.rs"]
mod training_dynamics_tests;
