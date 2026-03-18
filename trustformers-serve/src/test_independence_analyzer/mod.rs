//! Auto-generated module structure
//!
//! 🤖 Refactored with [SplitRS](https://github.com/cool-japan/splitrs)

// Existing submodules
pub mod analysis_cache;
pub mod conflict_detector;
pub mod dependency_graph;
pub mod resource_database;
pub mod test_grouping_engine;

// Refactored modules
pub mod analysisconfig_traits;
pub mod analysisqualitythresholds_traits;
pub mod functions;
pub mod testindependenceanalyzer_traits;
pub mod types;

// Re-export all types
pub use analysisconfig_traits::*;
pub use functions::*;
