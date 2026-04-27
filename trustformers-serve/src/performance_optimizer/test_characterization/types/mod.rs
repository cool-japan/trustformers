//! Type definitions for test characterization

pub mod alerts;
pub mod analysis;
pub mod core;
pub mod data_management;
pub mod gpu;
pub mod locking;
pub mod network_io;
pub mod optimization;
pub mod patterns;
pub mod patterns_extended;
pub mod performance;
pub mod quality;
pub mod reporting;
pub mod resources;

// Re-export all types
pub use alerts::*;
pub use analysis::*;
pub use core::*;
pub use data_management::functions::*;
pub use data_management::types::*;
pub use data_management::types_3::*;
pub use gpu::*;
pub use locking::*;
pub use network_io::*;
pub use optimization::*;
pub use patterns::*;
// patterns_extended is re-exported via patterns.rs
pub use performance::*;
pub use quality::*;
pub use reporting::*;
pub use resources::functions::*;
pub use resources::types::*;
pub use resources::types_3::*;
