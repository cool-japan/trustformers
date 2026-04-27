//! Auto-generated module structure

pub mod portperformancemetrics_traits;
pub mod performanceconfig_traits;
pub mod performancealertthresholds_traits;
pub mod performancetrends_traits;
pub mod types;
pub mod functions;

// Re-export all types
pub use portperformancemetrics_traits::*;
pub use performanceconfig_traits::*;
pub use performancealertthresholds_traits::*;
pub use performancetrends_traits::*;
pub use types::*;
pub use functions::*;

#[cfg(test)]
mod types_tests;
#[cfg(test)]
mod analysis_tests;
