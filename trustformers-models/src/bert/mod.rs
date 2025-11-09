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
//!
//! let config = BertConfig::bert_base_uncased();
//! let mut model = BertForSequenceClassification::new(config, num_labels)?;
//! model.load_from_hub("bert-base-uncased")?;
//!
//! // Perform classification
//! let outputs = model.forward(input_ids, attention_mask)?;
//! let predictions = outputs.logits.argmax(-1)?;
//! ```
//!
//! ### Masked Language Modeling
//! ```rust,no_run
//! use trustformers_models::bert::{BertForMaskedLM, BertConfig};
//!
//! let config = BertConfig::bert_base_uncased();
//! let mut model = BertForMaskedLM::new(config)?;
//! model.load_from_hub("bert-base-uncased")?;
//!
//! // Predict masked tokens
//! let outputs = model.forward(masked_input_ids, attention_mask)?;
//! let predictions = outputs.logits.argmax(-1)?;
//! ```
//!
//! ### Feature Extraction
//! ```rust,no_run
//! use trustformers_models::bert::{BertModel, BertConfig};
//!
//! let config = BertConfig::bert_base_uncased();
//! let mut model = BertModel::new(config)?;
//! model.load_from_hub("bert-base-uncased")?;
//!
//! // Extract features
//! let outputs = model.forward(input_ids, attention_mask)?;
//! let pooled_output = outputs.pooler_output; // [CLS] token representation
//! let sequence_output = outputs.last_hidden_state; // All token representations
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
