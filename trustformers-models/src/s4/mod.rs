//! S4 (Structured State Space) Model Implementation
//!
//! This module implements the S4 model architecture from "Efficiently Modeling Long Sequences
//! with Structured State Spaces" (Gu et al., 2022).
//!
//! S4 models excel at capturing long-range dependencies in sequences through efficient
//! state space mechanisms, making them particularly suitable for:
//! - Long document processing
//! - Time series modeling
//! - Audio and speech processing
//! - Any task requiring efficient handling of very long sequences

pub mod config;
pub mod model;

pub use config::S4Config;
pub use model::{S4Block, S4ForLanguageModeling, S4Layer, S4Model};

#[cfg(test)]
mod tests;
