//! Automatic quantization for efficient web deployment

pub mod algorithms;
pub mod config;
pub mod quantizer;

// Re-export main types and functions
pub use config::*;
pub use quantizer::*;
