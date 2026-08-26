//! Auto-generated module structure

pub mod functions;
pub mod functions_2;
pub mod trait_impls;
pub mod types;
pub mod types_3;
pub mod types_4;

// Re-export all types.
//
// `functions_2` (test-only helpers/fixtures) and `trait_impls` (trait `impl`
// blocks only, nothing nameable to re-export) intentionally have no `pub use`
// here: glob-importing them is a no-op that only produced an unused-import
// warning.
pub use functions::*;
pub use types::*;
pub use types_3::*;
pub use types_4::*;
