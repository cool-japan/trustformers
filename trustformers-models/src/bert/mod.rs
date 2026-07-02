//! # BERT (Bidirectional Encoder Representations from Transformers)
//!
//! BERT is a transformer-based model that uses bidirectional attention to create
//! deep bidirectional representations. It's pre-trained on masked language modeling
//! and next sentence prediction tasks.
//!
//! ## Architecture
//!
//! BERT consists of:
//! - Multi-layer bidirectional Transformer encoder
//! - WordPiece embeddings with positional and segment embeddings
//! - Layer normalization and dropout for regularization
//! - GELU activation functions
//!
//! ## Model Variants
//!
//! This implementation supports:
//! - **BERT-Base**: 12 layers, 768 hidden, 12 heads, 110M parameters
//! - **BERT-Large**: 24 layers, 1024 hidden, 16 heads, 340M parameters
//! - **Custom configurations**: Create your own BERT variant
//!
//! ## Usage Examples
//!
//! ### Text Classification
//! ```rust,no_run
//! use trustformers_models::bert::{BertForSequenceClassification, BertConfig};
//! use trustformers_core::traits::{Model, TokenizedInput};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = BertConfig::default();
//! # let num_labels = 2;
//! let model = BertForSequenceClassification::new(config, num_labels)?;
//!
//! // Perform classification
//! # let input_ids: Vec<u32> = vec![101, 2054, 2003, 102];
//! # let attention_mask: Vec<u8> = vec![1, 1, 1, 1];
//! let outputs = model.forward(TokenizedInput::new(input_ids, attention_mask))?;
//! let predictions = outputs.logits.argmax(-1)?;
//! # let _ = predictions;
//! # Ok(())
//! # }
//! ```
//!
//! ### Masked Language Modeling
//! ```rust,no_run
//! use trustformers_models::bert::{BertForMaskedLM, BertConfig};
//! use trustformers_core::traits::{Model, TokenizedInput};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = BertConfig::default();
//! let model = BertForMaskedLM::new(config)?;
//!
//! // Predict masked tokens
//! # let masked_input_ids: Vec<u32> = vec![101, 103, 2003, 102];
//! # let attention_mask: Vec<u8> = vec![1, 1, 1, 1];
//! let outputs = model.forward(TokenizedInput::new(masked_input_ids, attention_mask))?;
//! let predictions = outputs.logits.argmax(-1)?;
//! # let _ = predictions;
//! # Ok(())
//! # }
//! ```
//!
//! ### Feature Extraction
//! ```rust,no_run
//! use trustformers_models::bert::{BertModel, BertConfig};
//! use trustformers_core::traits::{Model, TokenizedInput};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = BertConfig::default();
//! let model = BertModel::new(config)?;
//!
//! // Extract features
//! # let input_ids: Vec<u32> = vec![101, 2054, 2003, 102];
//! # let attention_mask: Vec<u8> = vec![1, 1, 1, 1];
//! let outputs = model.forward(TokenizedInput::new(input_ids, attention_mask))?;
//! let pooled_output = outputs.pooler_output; // [CLS] token representation
//! let sequence_output = outputs.last_hidden_state; // All token representations
//! # let _ = pooled_output;
//! # let _ = sequence_output;
//! # Ok(())
//! # }
//! ```
//!
//! ## Pre-training Tasks
//!
//! BERT is pre-trained on two tasks:
//!
//! 1. **Masked Language Modeling (MLM)**: Randomly mask 15% of tokens and predict them
//! 2. **Next Sentence Prediction (NSP)**: Predict if sentence B follows sentence A
//!
//! ## Fine-tuning
//!
//! BERT can be fine-tuned for various downstream tasks:
//! - Text classification (sentiment analysis, spam detection)
//! - Named Entity Recognition (NER)
//! - Question Answering
//! - Text similarity
//! - Token classification
//!
//! ## Performance Tips
//!
//! - Use `bert-base` for most tasks (good balance of performance/accuracy)
//! - Enable mixed precision training for faster fine-tuning
//! - Adjust max sequence length based on your data
//! - Use gradient accumulation for larger effective batch sizes

pub mod config;
pub mod layers;
pub mod model;
pub mod tasks;

pub use config::BertConfig;
pub use model::BertModel;
pub use tasks::{
    BertForMaskedLM, BertForQuestionAnswering, BertForSequenceClassification,
    BertForTokenClassification,
};
