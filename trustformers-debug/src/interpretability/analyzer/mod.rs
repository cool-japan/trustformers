//! Auto-generated module structure

pub mod functions;
pub mod types;

// Re-export all types
// Note: functions.rs is intentionally minimal (split at struct boundary); all impls are in types.rs
pub use types::*;
