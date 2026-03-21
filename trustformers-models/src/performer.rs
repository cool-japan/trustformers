/// Performer: Fast Attention via Positive Orthogonal Random features (FAVOR+)
///
/// Paper: "Rethinking Attention with Performers" (Choromanski et al., 2020)
/// Key innovation: Approximates softmax attention using random feature maps for O(n) complexity
pub mod config;
pub mod model;

pub use config::PerformerConfig;
pub use model::{PerformerForMaskedLM, PerformerForSequenceClassification, PerformerModel};
