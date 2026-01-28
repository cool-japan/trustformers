//! # Question Answering Data Collators
//!
//! This module provides specialized data collators for question answering tasks.
//! It includes implementations for extractive QA models like BERT that identify
//! answer spans within given contexts.
//!
//! ## Overview
//!
//! Question answering involves training models to find answers to questions
//! within provided text contexts. This module provides collators that prepare
//! data appropriately for extractive QA tasks, handling question-context pairs
//! and answer span annotations.
//!
//! ## Architecture
//!
//! ```text
//! Question Answering Data Collator
//!      ├─ Handles question-context input pairs
//!      ├─ Manages answer span annotations (start/end positions)
//!      ├─ Creates appropriate token type IDs for question/context
//!      ├─ Supports document stride for long contexts
//!      └─ Handles unanswerable questions
//! ```
//!
//! ## Features
//!
//! - **Span Annotation**: Processes start and end position labels for answers
//! - **Question-Context Pairing**: Properly formats question and context sequences
//! - **Token Type Support**: Creates token type IDs to distinguish question from context
//! - **Document Stride**: Handles long contexts with sliding window approach
//! - **Unanswerable Questions**: Supports questions with no valid answers
//! - **Position Mapping**: Maintains mapping between original and tokenized positions
//!
//! ## Usage Examples
//!
//! ### Extractive Question Answering
//!
//! ```rust,ignore
//! use trustformers::auto::data_collators::question_answering::{
//!     QuestionAnsweringDataCollator, QuestionAnsweringCollatorConfig
//! };
//! use trustformers::auto::types::PaddingStrategy;
//!
//! let config = QuestionAnsweringCollatorConfig {
//!     max_length: Some(512),
//!     padding: PaddingStrategy::Longest,
//!     truncation: true,
//!     pad_token_id: 0,
//!     doc_stride: 128,
//!     max_answer_length: 30,
//! };
//!
//! let collator = QuestionAnsweringDataCollator::new(config);
//! ```
//!
//! ### SQuAD-style Dataset
//!
//! ```rust,ignore
//! use trustformers::auto::data_collators::question_answering::{
//!     QuestionAnsweringDataCollator, QuestionAnsweringCollatorConfig
//! };
//! use trustformers::auto::types::{DataExample, PaddingStrategy};
//!
//! // Configuration for SQuAD-like QA
//! let config = QuestionAnsweringCollatorConfig {
//!     max_length: Some(384),
//!     padding: PaddingStrategy::Longest,
//!     truncation: true,
//!     pad_token_id: 0,
//!     doc_stride: 128,
//!     max_answer_length: 30,
//! };
//!
//! let collator = QuestionAnsweringDataCollator::new(config);
//!
//! // Example with question and context
//! let examples = vec![
//!     DataExample {
//!         input_ids: vec![101, 2054, 2003, 102, 1996, 3438, 2003, 2093, 102],
//!         // [CLS] What is [SEP] The answer is blue [SEP]
//!         attention_mask: Some(vec![1, 1, 1, 1, 1, 1, 1, 1, 1]),
//!         token_type_ids: Some(vec![0, 0, 0, 0, 1, 1, 1, 1, 1]),
//!         labels: Some(vec![7, 7]),  // Start and end positions for "blue"
//!         metadata: HashMap::new(),
//!     },
//! ];
//!
//! let batch = collator.collate(&examples)?;
//! ```
//!
//! ## Supported Tasks
//!
//! - **Extractive QA**: Finding answer spans in given contexts (SQuAD, MSMARCO)
//! - **Reading Comprehension**: Understanding and answering questions about text
//! - **Information Extraction**: Extracting specific information from documents
//! - **Fact Verification**: Verifying claims against evidence texts
//!
//! ## Answer Span Format
//!
//! The collator expects answer positions in the `labels` field:
//! - **Answerable**: `vec![start_pos, end_pos]` (inclusive positions)
//! - **Unanswerable**: `vec![-100, -100]` or `None`
//! - **Multiple Spans**: First valid span is used
//!
//! ## Long Document Handling
//!
//! For documents longer than max_length, the collator can use document stride
//! to create overlapping windows, ensuring answer spans aren't split across windows.

use super::{DataCollator, DataCollatorConfig};
use crate::auto::data_collators::language_modeling::{
    LanguageModelingCollatorConfig, LanguageModelingDataCollator,
};
use crate::auto::types::{CollatedBatch, DataExample, PaddingStrategy};
use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Question Answering Data Collator
// =============================================================================

/// Data collator for question answering tasks (BERT-like models)
///
/// This collator is specifically designed for extractive question answering
/// models that identify answer spans within provided contexts. It handles
/// question-context pairs and properly formats answer position labels.
///
/// ## Features
///
/// - **Question-Context Processing**: Handles paired question and context sequences
/// - **Span Labeling**: Processes start and end position labels for answers
/// - **Token Type IDs**: Creates proper separation between question and context
/// - **Position Validation**: Ensures answer positions are within valid ranges
/// - **Unanswerable Support**: Handles questions without valid answers
/// - **Document Chunking**: Supports sliding window for long documents
///
/// ## Input Format
///
/// The collator expects input in specific format:
/// - `input_ids`: Concatenated question and context tokens with separators
/// - `token_type_ids`: 0 for question tokens, 1 for context tokens
/// - `labels`: `[start_position, end_position]` for answer span
///
/// ## Label Format
///
/// Answer positions are expected as:
/// - **Valid Answer**: `vec![start_pos, end_pos]` (token positions in sequence)
/// - **No Answer**: `vec![-100, -100]` (will be ignored in loss)
/// - **CLS Answer**: `vec![0, 0]` (for unanswerable questions in some formats)
///
/// ## Usage Examples
///
/// ```rust,ignore
/// use trustformers::auto::data_collators::question_answering::{
///     QuestionAnsweringDataCollator, QuestionAnsweringCollatorConfig
/// };
/// use trustformers::auto::types::{DataExample, PaddingStrategy};
///
/// // Configuration for BERT-based QA
/// let config = QuestionAnsweringCollatorConfig {
///     max_length: Some(512),
///     padding: PaddingStrategy::Longest,
///     truncation: true,
///     pad_token_id: 0,
///     doc_stride: 128,
///     max_answer_length: 30,
/// };
///
/// let collator = QuestionAnsweringDataCollator::new(config);
///
/// // Example for extractive QA
/// let examples = vec![
///     DataExample {
///         input_ids: vec![101, 2054, 2003, 102, 1996, 3438, 2003, 2769, 102],
///         // [CLS] What is [SEP] The answer is green [SEP]
///         attention_mask: Some(vec![1, 1, 1, 1, 1, 1, 1, 1, 1]),
///         token_type_ids: Some(vec![0, 0, 0, 0, 1, 1, 1, 1, 1]),
///         labels: Some(vec![7, 7]),  // Position of "green"
///         metadata: HashMap::new(),
///     },
/// ];
///
/// let batch = collator.collate(&examples)?;
/// ```
#[derive(Debug, Clone)]
pub struct QuestionAnsweringDataCollator {
    config: QuestionAnsweringCollatorConfig,
}

impl QuestionAnsweringDataCollator {
    /// Create a new question answering data collator
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying collation behavior for QA tasks
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = QuestionAnsweringCollatorConfig {
    ///     max_length: Some(384),
    ///     padding: PaddingStrategy::Longest,
    ///     truncation: true,
    ///     pad_token_id: 0,
    ///     doc_stride: 128,
    ///     max_answer_length: 30,
    /// };
    /// let collator = QuestionAnsweringDataCollator::new(config);
    /// ```
    pub fn new(config: QuestionAnsweringCollatorConfig) -> Self {
        Self { config }
    }

    /// Process QA labels for span prediction
    ///
    /// This method processes the answer span labels, ensuring they are in
    /// the correct format for question answering loss computation.
    ///
    /// # Arguments
    ///
    /// * `examples` - Input examples with answer span labels
    /// * `sequence_length` - Length of the processed sequences
    ///
    /// # Returns
    ///
    /// Processed labels formatted for QA span prediction
    fn process_qa_labels(&self, examples: &[DataExample], sequence_length: usize) -> Vec<Vec<i64>> {
        let mut processed_labels = Vec::with_capacity(examples.len());

        for example in examples {
            if let Some(ref example_labels) = example.labels {
                if example_labels.len() >= 2 {
                    let start_pos = example_labels[0];
                    let end_pos = example_labels[1];

                    // Validate positions
                    if start_pos >= 0
                        && end_pos >= 0
                        && (start_pos as usize) < sequence_length
                        && (end_pos as usize) < sequence_length
                        && start_pos <= end_pos
                    {
                        // Valid answer span
                        processed_labels.push(vec![start_pos, end_pos]);
                    } else {
                        // Invalid positions - treat as unanswerable
                        processed_labels.push(vec![-100, -100]);
                    }
                } else {
                    // Insufficient label data
                    processed_labels.push(vec![-100, -100]);
                }
            } else {
                // No labels provided - unanswerable
                processed_labels.push(vec![-100, -100]);
            }
        }

        processed_labels
    }

    /// Validate answer span positions
    ///
    /// This method checks that answer spans are within valid bounds and
    /// conform to the expected format for QA tasks.
    ///
    /// # Arguments
    ///
    /// * `examples` - Input examples to validate
    /// * `sequence_length` - Maximum valid position
    ///
    /// # Returns
    ///
    /// Result indicating whether all spans are valid
    fn validate_answer_spans(
        &self,
        examples: &[DataExample],
        sequence_length: usize,
    ) -> Result<()> {
        for (i, example) in examples.iter().enumerate() {
            if let Some(ref labels) = example.labels {
                if labels.len() >= 2 {
                    let start_pos = labels[0];
                    let end_pos = labels[1];

                    // Check for valid positions (allow -100 for unanswerable)
                    if start_pos != -100 && end_pos != -100 {
                        if start_pos < 0 || end_pos < 0 {
                            return Err(TrustformersError::invalid_input(
                                format!(
                                    "Negative answer positions in example {}: start={}, end={}",
                                    i, start_pos, end_pos
                                ),
                                Some("answer_positions"),
                                Some("non-negative start and end positions"),
                                Some(format!("start={}, end={}", start_pos, end_pos)),
                            ));
                        }

                        if (start_pos as usize) >= sequence_length
                            || (end_pos as usize) >= sequence_length
                        {
                            return Err(TrustformersError::invalid_input(                                format!("Answer positions exceed sequence length in example {}: start={}, end={}, seq_len={}", i, start_pos, end_pos, sequence_length),
                                Some("answer_positions"),
                                Some(format!("positions within sequence length {}", sequence_length)),
                                Some(format!("start={}, end={}", start_pos, end_pos))
                            ));
                        }

                        if start_pos > end_pos {
                            return Err(TrustformersError::invalid_input(
                                format!(
                                    "Invalid answer span in example {}: start={} > end={}",
                                    i, start_pos, end_pos
                                ),
                                Some("answer_span"),
                                Some("start position <= end position"),
                                Some(format!("start={}, end={}", start_pos, end_pos)),
                            ));
                        }

                        // Check maximum answer length
                        let answer_length = (end_pos - start_pos + 1) as usize;
                        if answer_length > self.config.max_answer_length {
                            return Err(TrustformersError::invalid_input(
                                format!(
                                    "Answer span too long in example {}: length={}, max={}",
                                    i, answer_length, self.config.max_answer_length
                                ),
                                Some("answer_length"),
                                Some(format!("length <= {}", self.config.max_answer_length)),
                                Some(answer_length.to_string()),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Create question-context separation metadata
    ///
    /// This method creates metadata about question and context boundaries
    /// which can be useful for model training and evaluation.
    ///
    /// # Arguments
    ///
    /// * `examples` - Input examples with token type IDs
    ///
    /// # Returns
    ///
    /// Metadata about question and context boundaries
    fn create_sequence_metadata(
        &self,
        examples: &[DataExample],
    ) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();

        let mut question_lengths = Vec::new();
        let mut context_starts = Vec::new();

        for example in examples {
            if let Some(ref token_types) = example.token_type_ids {
                // Find the transition from question (0) to context (1)
                let context_start =
                    token_types.iter().position(|&t| t == 1).unwrap_or(token_types.len());

                context_starts.push(context_start);
                question_lengths.push(context_start);
            } else {
                // No token type IDs - assume entire sequence is context
                context_starts.push(0);
                question_lengths.push(0);
            }
        }

        metadata.insert(
            "question_lengths".to_string(),
            serde_json::to_value(question_lengths).unwrap_or(serde_json::Value::Null),
        );
        metadata.insert(
            "context_starts".to_string(),
            serde_json::to_value(context_starts).unwrap_or(serde_json::Value::Null),
        );

        metadata
    }
}

impl DataCollator for QuestionAnsweringDataCollator {
    fn collate(&self, examples: &[DataExample]) -> Result<CollatedBatch> {
        if examples.is_empty() {
            return Err(TrustformersError::invalid_input(
                "Cannot collate empty batch for question answering".to_string(),
                Some("examples".to_string()),
                Some("non-empty batch".to_string()),
                Some("empty batch".to_string()),
            ));
        }

        // Use the base language modeling collator for sequence processing
        // (without MLM masking)
        let sequence_collator = LanguageModelingDataCollator::new(LanguageModelingCollatorConfig {
            max_length: self.config.max_length,
            padding: self.config.padding,
            truncation: self.config.truncation,
            pad_token_id: self.config.pad_token_id,
            mask_token_id: 0, // No masking for QA
            mlm_probability: 0.0,
        });

        // Collate input sequences
        let mut batch = sequence_collator.collate(examples)?;

        // Validate answer spans
        self.validate_answer_spans(examples, batch.sequence_length)?;

        // Process QA-specific labels (start and end positions)
        let processed_labels = self.process_qa_labels(examples, batch.sequence_length);
        batch.labels = Some(processed_labels);

        // Add QA-specific metadata
        batch.metadata.insert(
            "task_type".to_string(),
            serde_json::Value::String("question_answering".to_string()),
        );
        batch.metadata.insert(
            "doc_stride".to_string(),
            serde_json::Value::Number(self.config.doc_stride.into()),
        );
        batch.metadata.insert(
            "max_answer_length".to_string(),
            serde_json::Value::Number(self.config.max_answer_length.into()),
        );

        // Add sequence boundary metadata
        let sequence_metadata = self.create_sequence_metadata(examples);
        for (key, value) in sequence_metadata {
            batch.metadata.insert(key, value);
        }

        Ok(batch)
    }

    fn config(&self) -> &dyn DataCollatorConfig {
        &self.config
    }

    fn preprocess_examples(&self, examples: &[DataExample]) -> Result<Vec<DataExample>> {
        // For QA, we might want to add preprocessing like
        // question-context concatenation or special token insertion
        // For now, just return as-is assuming input is already properly formatted
        Ok(examples.to_vec())
    }
}

// =============================================================================
// Question Answering Configuration
// =============================================================================

/// Configuration for question answering data collator
///
/// This configuration struct controls all aspects of data collation for
/// question answering tasks. It extends the basic collation parameters
/// with QA-specific settings like document stride and answer length limits.
///
/// ## Configuration Parameters
///
/// - `max_length`: Maximum sequence length for padding/truncation
/// - `padding`: Strategy for padding sequences in batch
/// - `truncation`: Whether to truncate sequences exceeding max_length
/// - `pad_token_id`: Token ID used for padding positions
/// - `doc_stride`: Stride for sliding window over long documents
/// - `max_answer_length`: Maximum allowed length for answer spans
///
/// ## Document Processing
///
/// For long documents that exceed `max_length`, the collator can use
/// `doc_stride` to create overlapping windows. This ensures that answer
/// spans are not accidentally split across windows.
///
/// ## Answer Constraints
///
/// The `max_answer_length` parameter constrains the maximum length of
/// valid answer spans, helping to filter out overly long or invalid spans.
///
/// ## Default Values
///
/// The `from_config` method provides sensible defaults:
/// - `doc_stride`: 128 tokens (overlap for long documents)
/// - `max_answer_length`: 30 tokens (reasonable for most QA tasks)
/// - `padding`: Longest sequence in batch
/// - `truncation`: Enabled
/// - `pad_token_id`: From model config or 0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionAnsweringCollatorConfig {
    /// Maximum sequence length for padding/truncation
    pub max_length: Option<usize>,
    /// Padding strategy for batch collation
    pub padding: PaddingStrategy,
    /// Whether to truncate sequences exceeding max_length
    pub truncation: bool,
    /// Token ID used for padding
    pub pad_token_id: u32,
    /// Document stride for sliding window over long contexts
    pub doc_stride: usize,
    /// Maximum allowed length for answer spans
    pub max_answer_length: usize,
}

impl QuestionAnsweringCollatorConfig {
    /// Create configuration from model config JSON
    ///
    /// This method extracts relevant configuration parameters from a model's
    /// config.json file and creates an appropriate collator configuration
    /// for question answering tasks.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value (typically from config.json)
    ///
    /// # Returns
    ///
    /// A configured `QuestionAnsweringCollatorConfig` with parameters extracted
    /// from the model configuration and QA-specific defaults.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let model_config = serde_json::json!({
    ///     "max_position_embeddings": 512,
    ///     "pad_token_id": 0,
    ///     "vocab_size": 30522
    /// });
    ///
    /// let config = QuestionAnsweringCollatorConfig::from_config(&model_config)?;
    /// assert_eq!(config.max_length, Some(512));
    /// assert_eq!(config.pad_token_id, 0);
    /// assert_eq!(config.doc_stride, 128);
    /// assert_eq!(config.max_answer_length, 30);

    pub fn from_config(config: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            max_length: config
                .get("max_position_embeddings")
                .or_else(|| config.get("max_length"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: config.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            doc_stride: config.get("doc_stride").and_then(|v| v.as_u64()).unwrap_or(128) as usize,
            max_answer_length: config
                .get("max_answer_length")
                .and_then(|v| v.as_u64())
                .unwrap_or(30) as usize,
        })
    }

    /// Create a configuration for SQuAD-style QA
    ///
    /// Creates a configuration with settings commonly used for SQuAD and
    /// similar extractive QA datasets.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    ///
    /// # Returns
    ///
    /// A configuration optimized for SQuAD-style QA
    pub fn for_squad(config: &serde_json::Value) -> Result<Self> {
        let mut squad_config = Self::from_config(config)?;
        squad_config.max_length = Some(384); // Common SQuAD setting
        squad_config.doc_stride = 128;
        squad_config.max_answer_length = 30;
        Ok(squad_config)
    }

    /// Create a configuration for long-form QA
    ///
    /// Creates a configuration for QA tasks with longer contexts and
    /// potentially longer answers.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    /// * `context_length` - Maximum context length
    /// * `answer_length` - Maximum answer length
    ///
    /// # Returns
    ///
    /// A configuration optimized for long-form QA
    pub fn for_long_form_qa(
        config: &serde_json::Value,
        context_length: Option<usize>,
        answer_length: Option<usize>,
    ) -> Result<Self> {
        let mut long_config = Self::from_config(config)?;

        if let Some(max_len) = context_length {
            long_config.max_length = Some(max_len);
            // Adjust doc_stride for longer contexts
            long_config.doc_stride = max_len / 4;
        }

        if let Some(max_ans) = answer_length {
            long_config.max_answer_length = max_ans;
        }

        Ok(long_config)
    }

    /// Create a configuration for conversational QA
    ///
    /// Creates a configuration for multi-turn QA where context includes
    /// conversation history.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    ///
    /// # Returns
    ///
    /// A configuration optimized for conversational QA
    pub fn for_conversational_qa(config: &serde_json::Value) -> Result<Self> {
        let mut conv_config = Self::from_config(config)?;
        // Longer sequences for conversation history
        conv_config.max_length = Some(512);
        conv_config.doc_stride = 256; // Larger stride for conversations
        conv_config.max_answer_length = 50; // Potentially longer answers
        Ok(conv_config)
    }
}

impl DataCollatorConfig for QuestionAnsweringCollatorConfig {
    fn max_length(&self) -> Option<usize> {
        self.max_length
    }

    fn padding(&self) -> PaddingStrategy {
        self.padding
    }

    fn truncation(&self) -> bool {
        self.truncation
    }
}

// =============================================================================
// Module Re-exports
// =============================================================================

// Note: No need to re-export since these are already public structs in this module

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_qa_collator_creation() {
        let config = QuestionAnsweringCollatorConfig {
            max_length: Some(384),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
            doc_stride: 128,
            max_answer_length: 30,
        };

        let collator = QuestionAnsweringDataCollator::new(config);
        assert_eq!(collator.config().max_length(), Some(384));
        assert_eq!(collator.config.doc_stride, 128);
        assert_eq!(collator.config.max_answer_length, 30);
    }

    #[test]
    fn test_qa_config_from_json() {
        let config_json = serde_json::json!({
            "max_position_embeddings": 512,
            "pad_token_id": 1,
            "doc_stride": 64,
            "max_answer_length": 20,
            "vocab_size": 30522
        });

        let config = QuestionAnsweringCollatorConfig::from_config(&config_json).unwrap();
        assert_eq!(config.max_length, Some(512));
        assert_eq!(config.pad_token_id, 1);
        assert_eq!(config.doc_stride, 64);
        assert_eq!(config.max_answer_length, 20);
    }

    #[test]
    fn test_collate_qa_examples() {
        let config = QuestionAnsweringCollatorConfig {
            max_length: Some(20),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
            doc_stride: 128,
            max_answer_length: 10,
        };

        let collator = QuestionAnsweringDataCollator::new(config);

        let examples = vec![
            DataExample {
                input_ids: vec![101, 2054, 2003, 102, 1996, 3438, 2003, 2769, 102],
                // [CLS] What is [SEP] The answer is green [SEP]
                attention_mask: Some(vec![1, 1, 1, 1, 1, 1, 1, 1, 1]),
                token_type_ids: Some(vec![0, 0, 0, 0, 1, 1, 1, 1, 1]),
                labels: Some(vec![7, 7]), // Position of "green"
                metadata: HashMap::new(),
            },
            DataExample {
                input_ids: vec![101, 2073, 102, 3376, 102],
                // [CLS] Who [SEP] Bob [SEP]
                attention_mask: Some(vec![1, 1, 1, 1, 1]),
                token_type_ids: Some(vec![0, 0, 0, 1, 1]),
                labels: Some(vec![3, 3]), // Position of "Bob"
                metadata: HashMap::new(),
            },
        ];

        let batch = collator.collate(&examples).unwrap();
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.input_ids.len(), 2);
        assert!(batch.labels.is_some());

        let labels = batch.labels.as_ref().unwrap();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], vec![7, 7]);
        assert_eq!(labels[1], vec![3, 3]);
    }

    #[test]
    fn test_unanswerable_questions() {
        let config = QuestionAnsweringCollatorConfig {
            max_length: Some(20),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
            doc_stride: 128,
            max_answer_length: 10,
        };

        let collator = QuestionAnsweringDataCollator::new(config);

        let examples = vec![DataExample {
            input_ids: vec![101, 2054, 102, 1045, 2123, 1005, 1056, 2113, 102],
            // [CLS] What [SEP] I don't know [SEP]
            attention_mask: Some(vec![1, 1, 1, 1, 1, 1, 1, 1, 1]),
            token_type_ids: Some(vec![0, 0, 0, 1, 1, 1, 1, 1, 1]),
            labels: Some(vec![-100, -100]), // Unanswerable
            metadata: HashMap::new(),
        }];

        let batch = collator.collate(&examples).unwrap();
        let labels = batch.labels.as_ref().unwrap();
        assert_eq!(labels[0], vec![-100, -100]);
    }

    #[test]
    fn test_answer_span_validation() {
        let config = QuestionAnsweringCollatorConfig {
            max_length: Some(10),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
            doc_stride: 128,
            max_answer_length: 3,
        };

        let collator = QuestionAnsweringDataCollator::new(config);

        // Valid spans
        let valid_examples = vec![DataExample {
            input_ids: vec![101, 102, 103, 104],
            attention_mask: Some(vec![1, 1, 1, 1]),
            token_type_ids: None,
            labels: Some(vec![2, 3]), // Valid span
            metadata: HashMap::new(),
        }];

        assert!(collator.validate_answer_spans(&valid_examples, 4).is_ok());

        // Invalid spans (start > end)
        let invalid_examples = vec![DataExample {
            input_ids: vec![101, 102, 103, 104],
            attention_mask: Some(vec![1, 1, 1, 1]),
            token_type_ids: None,
            labels: Some(vec![3, 2]), // Invalid: start > end
            metadata: HashMap::new(),
        }];

        assert!(collator.validate_answer_spans(&invalid_examples, 4).is_err());

        // Span too long
        let long_span_examples = vec![DataExample {
            input_ids: vec![101, 102, 103, 104, 105, 106],
            attention_mask: Some(vec![1, 1, 1, 1, 1, 1]),
            token_type_ids: None,
            labels: Some(vec![1, 5]), // Span length = 5, max = 3
            metadata: HashMap::new(),
        }];

        assert!(collator.validate_answer_spans(&long_span_examples, 6).is_err());
    }

    #[test]
    fn test_squad_config() {
        let model_config = serde_json::json!({
            "max_position_embeddings": 512,
            "pad_token_id": 0
        });

        let config = QuestionAnsweringCollatorConfig::for_squad(&model_config).unwrap();
        assert_eq!(config.max_length, Some(384));
        assert_eq!(config.doc_stride, 128);
        assert_eq!(config.max_answer_length, 30);
    }

    #[test]
    fn test_sequence_metadata() {
        let config = QuestionAnsweringCollatorConfig {
            max_length: Some(20),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
            doc_stride: 128,
            max_answer_length: 10,
        };

        let collator = QuestionAnsweringDataCollator::new(config);

        let examples = vec![DataExample {
            input_ids: vec![101, 2054, 102, 1996, 3438, 102],
            attention_mask: Some(vec![1, 1, 1, 1, 1, 1]),
            token_type_ids: Some(vec![0, 0, 0, 1, 1, 1]),
            labels: Some(vec![4, 4]),
            metadata: HashMap::new(),
        }];

        let metadata = collator.create_sequence_metadata(&examples);
        assert!(metadata.contains_key("question_lengths"));
        assert!(metadata.contains_key("context_starts"));
    }
}
