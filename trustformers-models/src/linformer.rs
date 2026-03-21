/// Linformer: Self-Attention with Linear Complexity
///
/// Paper: "Linformer: Self-Attention with Linear Complexity" (Wang et al., 2020)
/// Key innovation: Projects keys and values to lower-dimensional space to achieve O(n) attention
pub mod config;
pub mod model;

pub use config::LinformerConfig;
pub use model::{LinformerForMaskedLM, LinformerForSequenceClassification, LinformerModel};
