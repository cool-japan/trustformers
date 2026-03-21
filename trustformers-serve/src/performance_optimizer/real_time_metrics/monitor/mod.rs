//! Auto-generated module structure

pub mod anomalydetectionconfig_traits;
pub mod baselineconfig_traits;
pub mod baselinevalidationengine_traits;
pub mod defaultadaptationalgorithm_traits;
pub mod defaultscalingalgorithm_traits;
pub mod exponentialsmoothingadaptation_traits;
pub mod functions;
pub mod linearregressiontrenddetector_traits;
pub mod movingaveragetrenddetector_traits;
pub mod patternanomalydetector_traits;
pub mod patternrecognitionengine_traits;
pub mod performancebaseline_traits;
pub mod regressionengine_traits;
pub mod statisticalanomalydetector_traits;
pub mod threadconfiguration_traits;
pub mod threadloadbalancer_traits;
pub mod threadpoolconfig_traits;
pub mod thresholdanomalydetector_traits;
pub mod trendanalysisconfig_traits;
pub mod types;
pub mod types_baseline;
pub mod variabilitybounds_traits;

// Re-export all types (types_baseline is re-exported via types.rs)
pub use functions::*;
pub use types::*;
