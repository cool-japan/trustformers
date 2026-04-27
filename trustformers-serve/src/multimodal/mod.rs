//! Auto-generated module structure

pub mod audioprocessingconfig_traits;
pub mod contentvalidationconfig_traits;
pub mod documentprocessingconfig_traits;
pub mod functions;
pub mod imageprocessingconfig_traits;
pub mod multimodalconfig_traits;
pub mod ocrconfig_traits;
pub mod processingoptions_traits;
pub mod serving;
pub mod storageconfig_traits;
pub mod textpreprocessingconfig_traits;
pub mod types;
#[cfg(test)]
mod types_tests;
pub mod videoprocessingconfig_traits;

// Re-export all types
pub use serving::{
    Modality, MultiModalMessage, MultiModalProcessor, MultiModalServingError,
    MultiModalServingInput, MultiModalServingRequest, MultiModalServingResponse, ValidationReport,
};
pub use types::*;
