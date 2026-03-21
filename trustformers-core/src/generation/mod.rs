// Generation module - refactored from generation.rs for better organization

pub mod cache;
// pub mod cfg;  // Temporarily disabled due to compilation errors
pub mod config;
pub mod constraints;
pub mod core;
pub mod streaming;
// pub mod watermarking;  // Missing file - temporarily disabled
// pub mod assisted;  // Missing file - temporarily disabled

// Re-export commonly used types
pub use cache::{Beam, KVCache};
// pub use cfg::CFGGenerator;
pub use config::{
    // AssistedGenerationConfig, CFGConfig,
    GenerationConfig,
    GenerationStrategy,
    // GuidedGenerationConfig, WatermarkingAlgorithm, WatermarkingConfig,
};
pub use constraints::{ConstraintValidator, GrammarValidator, JsonSchemaValidator};
pub use core::TextGenerator;
pub use streaming::{FinishReason, GenerationStream, GenerationStreamTrait, GenerationToken};
// pub use watermarking::{TextWatermarker, WatermarkDetectionResult, WatermarkedGenerator};
// pub use assisted::{AssistedGenerator, SpeculativeDecoder};

// For backward compatibility, re-export everything at the root level
// pub use cfg::*;
pub use config::*;
// pub use watermarking::*;
// pub use assisted::*;
