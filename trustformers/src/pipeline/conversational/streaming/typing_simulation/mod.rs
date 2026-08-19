//! Auto-generated module structure

pub mod functions;
pub mod trait_impls;
pub mod types;
pub mod types_3;

// Re-export all types.
//
// `functions` (test-only helpers/fixtures) and `trait_impls` (trait `impl`
// blocks only, nothing nameable to re-export) intentionally have no `pub use`
// here: glob-importing them is a no-op that only produced an unused-import
// warning.
pub use types::*;
pub use types_3::*;
