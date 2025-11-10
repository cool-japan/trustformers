//! Model Export Module
//!
//! This module provides export capabilities to various model formats
//! for deployment on different platforms.

pub mod coreml;

pub use coreml::{
    CoreMLComputeUnit, CoreMLExportConfig, CoreMLExporter, CoreMLLayerType, CoreMLModel,
    CoreMLModelMetadata, CoreMLPrecision, CoreMLVersion,
};
