/// Hyena Hierarchy: Subquadratic Attention via Implicit Long Convolutions
///
/// Paper: "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution" (Nguyen et al., 2023)
/// Key innovation: Replaces attention with implicit long convolutions for subquadratic complexity
pub mod config;
pub mod model;

pub use config::HyenaConfig;
pub use model::{HyenaForLanguageModeling, HyenaForSequenceClassification, HyenaModel};
