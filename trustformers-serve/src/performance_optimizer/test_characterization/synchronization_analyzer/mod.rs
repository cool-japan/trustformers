//! Auto-generated module structure

pub mod criticalsectionanalyzerconfig_traits;
pub mod deadlockpreventionconfig_traits;
pub mod deadlockpreventionengine_traits;
pub mod functions;
pub mod lockdependencyanalyzerconfig_traits;
pub mod lockdependencygraph_traits;
pub mod lockorderingconfig_traits;
pub mod metricsengineconfig_traits;
pub mod patternrecognitionconfig_traits;
pub mod prioritybasedorderingalgorithm_traits;
pub mod recommendationengineconfig_traits;
pub mod synchronizationanalysisstats_traits;
pub mod synchronizationanalyzerconfig_traits;
pub mod synchronizationmetricsdatabase_traits;
pub mod synchronizationpatternlibrary_traits;
pub mod synchronizationpointdetectorconfig_traits;
pub mod topologicalorderingalgorithm_traits;
pub mod types;
pub mod types_patterns;
pub mod waittimeanalyzerconfig_traits;

// Re-export all types (types_patterns is re-exported via types.rs)
pub use functions::*;
pub use types::*;
