/// FNet: Mixing Tokens with Fourier Transforms
///
/// Paper: "FNet: Mixing Tokens with Fourier Transforms" (Lee-Thorp et al., 2021)
/// Key innovation: Replaces self-attention with Fourier transforms for O(n log n) complexity
pub mod config;
pub mod model;

pub use config::FNetConfig;
pub use model::{FNetForMaskedLM, FNetForSequenceClassification, FNetModel};
