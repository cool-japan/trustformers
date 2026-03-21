/// RetNet: Retentive Network for Large Language Models
///
/// Paper: "Retentive Network: A Successor to Transformer for Large Language Models" (Sun et al., 2023)
/// Key innovation: Replaces attention with retention mechanism for better scaling and efficiency
pub mod config;
pub mod model;

pub use config::RetNetConfig;
pub use model::{RetNetForLanguageModeling, RetNetForSequenceClassification, RetNetModel};
