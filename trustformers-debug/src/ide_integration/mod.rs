//! Auto-generated module structure

pub mod functions;
pub mod trait_impls;
pub mod types;
pub mod types_3;

// Re-export all types
pub use types::*;
pub use types_3::*;

#[cfg(test)]
#[path = "../ide_integration_tests.rs"]
mod ide_integration_tests;
