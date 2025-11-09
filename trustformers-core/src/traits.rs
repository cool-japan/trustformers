//! Core traits defining the fundamental abstractions of TrustformeRS.
//!
//! This module contains the essential traits that form the foundation of the TrustformeRS
//! transformer library. These traits define the interfaces for models, layers, configuration,
//! tokenization, optimization, and parameter initialization.
//!
//! # Overview
//!
//! The traits in this module establish a consistent API across all transformer implementations:
//!
//! - [`Model`]: The main trait for transformer models with forward pass and loading capabilities
//! - [`Layer`]: Building blocks for neural network architectures
//! - [`Config`]: Configuration management for models and components
//! - [`WeightReader`]: Interface for loading pretrained model weights
//! - [`Tokenizer`]: Text tokenization and encoding/decoding
//! - [`Optimizer`]: Parameter optimization algorithms
//! - [`ParameterInit`]: Weight initialization strategies
//!
//! # Examples
//!
//! ```no_run
//! use trustformers_core::traits::{Model, Config};
//! use trustformers_core::tensor::Tensor;
//!
//! // Example model implementation
//! struct MyModel {
//!     config: MyConfig,
//!     // ... model layers
//! }
//!
//! impl Model for MyModel {
//!     type Config = MyConfig;
//!     type Input = Tensor;
//!     type Output = Tensor;
//!
//!     fn forward(&self, input: Self::Input) -> Result<Self::Output> {
//!         // Model forward pass implementation
//!         Ok(input)
//!     }
//!
//!     fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
//!         // Load pretrained weights
//!         Ok(())
//!     }
//!
//!     fn get_config(&self) -> &Self::Config {
//!         &self.config
//!     }
//! }
//! ```

use crate::errors::Result;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::io::Read;

/// The main trait for transformer models.
///
/// This trait defines the interface that all transformer models must implement,
/// providing a consistent API for forward passes, weight loading, and configuration access.
///
/// # Type Parameters
///
/// - `Config`: The configuration type for this model, must implement [`Config`]
/// - `Input`: The input type for the model's forward pass
/// - `Output`: The output type produced by the model
///
/// # Thread Safety
///
/// Models must be `Send + Sync` to support multi-threaded inference and training.
///
/// # Example
///
/// ```no_run
/// use trustformers_core::traits::{Model, Config};
/// use trustformers_core::tensor::Tensor;
/// use trustformers_core::error::Result;
/// use std::io::Read;
/// use serde::{Deserialize, Serialize};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// #[derive(Deserialize, Serialize)]
/// struct BertConfig {
///     hidden_size: usize,
///     num_attention_heads: usize,
///     // ... other config fields
/// }
///
/// impl Config for BertConfig {
///     fn architecture(&self) -> &'static str {
///         "bert"
///     }
/// }
///
/// struct BertModel {
///     config: BertConfig,
///     // ... model layers
/// }
///
/// impl Model for BertModel {
///     type Config = BertConfig;
///     type Input = Tensor;
///     type Output = Tensor;
///
///     fn forward(&self, input: Self::Input) -> Result<Self::Output> {
///         // BERT forward pass implementation
///         Ok(input)
///     }
///
///     fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
///         // Load BERT weights from reader
///         Ok(())
///     }
///
///     fn get_config(&self) -> &Self::Config {
///         &self.config
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub trait Model: Send + Sync {
    type Config: Config;
    type Input;
    type Output;

    /// Performs a forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data for the model
    ///
    /// # Returns
    ///
    /// Returns `Ok(output)` on success, or an error if the forward pass fails.
    ///
    /// # Errors
    ///
    /// May return errors for:
    /// - Invalid input dimensions
    /// - Numerical computation errors
    /// - Out of memory conditions
    fn forward(&self, input: Self::Input) -> Result<Self::Output>;

    /// Loads pretrained weights into the model.
    ///
    /// This method reads model weights from a reader (typically a file or network stream)
    /// and updates the model's parameters accordingly.
    ///
    /// # Arguments
    ///
    /// * `reader` - A reader providing access to the pretrained weight data
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful loading, or an error if loading fails.
    ///
    /// # Errors
    ///
    /// May return errors for:
    /// - IO errors while reading
    /// - Incompatible weight formats
    /// - Mismatched tensor shapes
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()>;

    /// Returns a reference to the model's configuration.
    ///
    /// # Returns
    ///
    /// A reference to the model's configuration object.
    fn get_config(&self) -> &Self::Config;

    /// Returns the total number of parameters in the model.
    ///
    /// This method calculates the total number of trainable parameters
    /// across all layers in the model. It's useful for compression metrics,
    /// model size analysis, and memory usage calculations.
    ///
    /// # Returns
    ///
    /// The total number of parameters as a `usize`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::traits::Model;
    ///
    /// fn analyze_model_size<M: Model>(model: &M) {
    ///     let params = model.num_parameters();
    ///     let memory_mb = params * 4 / (1024 * 1024); // Assuming f32 weights
    ///     println!("Model has {} parameters ({} MB)", params, memory_mb);
    /// }
    /// ```
    fn num_parameters(&self) -> usize;
}

/// A building block for neural network architectures.
///
/// The `Layer` trait represents a single computational unit in a neural network,
/// such as a linear transformation, attention mechanism, or normalization layer.
/// Layers can be composed together to build complete models.
///
/// # Type Parameters
///
/// - `Input`: The input type accepted by this layer
/// - `Output`: The output type produced by this layer
///
/// # Thread Safety
///
/// Layers must be `Send + Sync` to support parallel computation.
///
/// # Example
///
/// ```no_run
/// use trustformers_core::traits::Layer;
/// use trustformers_core::tensor::Tensor;
/// use trustformers_core::error::Result;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// struct LinearLayer {
///     weight: Tensor,
///     bias: Option<Tensor>,
/// }
///
/// impl Layer for LinearLayer {
///     type Input = Tensor;
///     type Output = Tensor;
///
///     fn forward(&self, input: Self::Input) -> Result<Self::Output> {
///         // Compute linear transformation: y = xW^T + b
///         let output = input.matmul(&self.weight.transpose()?)?;
///         if let Some(bias) = &self.bias {
///             output.add(bias)
///         } else {
///             Ok(output)
///         }
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub trait Layer: Send + Sync {
    type Input;
    type Output;

    /// Performs the forward computation of this layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data to process
    ///
    /// # Returns
    ///
    /// Returns `Ok(output)` containing the layer's output, or an error if computation fails.
    ///
    /// # Errors
    ///
    /// May return errors for:
    /// - Invalid input dimensions
    /// - Numerical errors during computation
    /// - Resource allocation failures
    fn forward(&self, input: Self::Input) -> Result<Self::Output>;
}

/// Configuration trait for models and components.
///
/// This trait provides a standardized interface for configuration objects
/// that can be serialized, deserialized, and validated. All model configurations
/// must implement this trait to ensure compatibility with the TrustformeRS ecosystem.
///
/// # Requirements
///
/// Implementing types must be serializable and deserializable using serde.
///
/// # Example
///
/// ```no_run
/// use trustformers_core::traits::Config;
/// use trustformers_core::error::Result;
/// use serde::{Deserialize, Serialize};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// #[derive(Debug, Clone, Deserialize, Serialize)]
/// struct GPT2Config {
///     vocab_size: usize,
///     hidden_size: usize,
///     num_layers: usize,
///     num_heads: usize,
/// }
///
/// impl Config for GPT2Config {
///     fn validate(&self) -> Result<()> {
///         if self.hidden_size % self.num_heads != 0 {
///             return Err(anyhow::anyhow!(
///                 "hidden_size must be divisible by num_heads"
///             ));
///         }
///         Ok(())
///     }
///
///     fn architecture(&self) -> &'static str {
///         "gpt2"
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub trait Config: for<'de> Deserialize<'de> + Serialize {
    /// Validates the configuration for correctness.
    ///
    /// This method should check that all configuration parameters are valid
    /// and compatible with each other. The default implementation accepts
    /// all configurations as valid.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the configuration is valid, or an error describing
    /// the validation failure.
    ///
    /// # Example
    ///
    /// Common validations include:
    /// - Checking that dimensions are compatible
    /// - Verifying that values are within acceptable ranges
    /// - Ensuring required fields are properly set
    fn validate(&self) -> Result<()> {
        Ok(())
    }

    /// Returns the architecture name for this configuration.
    ///
    /// This should return a static string identifying the model architecture,
    /// such as "bert", "gpt2", "t5", etc. This is used for model registration
    /// and automatic model selection.
    ///
    /// # Returns
    ///
    /// A static string slice containing the architecture name.
    fn architecture(&self) -> &'static str;
}

/// Interface for reading model weights from various sources.
///
/// `WeightReader` provides an abstraction over different weight storage formats,
/// allowing models to load pretrained parameters from files, network sources,
/// or other storage backends.
///
/// # Supported Formats
///
/// Implementations may support various formats including:
/// - SafeTensors (.safetensors)
/// - PyTorch checkpoints (.pt, .bin)
/// - NumPy arrays (.npz)
/// - Custom formats
///
/// # Example
///
/// ```no_run
/// use trustformers_core::traits::WeightReader;
/// use trustformers_core::tensor::Tensor;
/// use trustformers_core::error::Result;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// struct SafeTensorsReader {
///     // ... implementation details
/// }
///
/// impl WeightReader for SafeTensorsReader {
///     fn read_tensor(&mut self, name: &str) -> Result<Tensor> {
///         // Read tensor from SafeTensors file
///         // ...
///         Tensor::zeros(&[768, 768])
///     }
///
///     fn list_tensors(&self) -> Vec<String> {
///         vec![
///             "bert.embeddings.word_embeddings.weight".to_string(),
///             "bert.encoder.layer.0.attention.self.query.weight".to_string(),
///             // ... more tensor names
///         ]
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub trait WeightReader {
    /// Reads a tensor by name from the weight source.
    ///
    /// # Arguments
    ///
    /// * `name` - The name/key of the tensor to read (e.g., "encoder.layer.0.weight")
    ///
    /// # Returns
    ///
    /// Returns `Ok(tensor)` containing the requested tensor, or an error if the
    /// tensor cannot be found or loaded.
    ///
    /// # Errors
    ///
    /// May return errors for:
    /// - Tensor not found with the given name
    /// - IO errors while reading
    /// - Corrupted or invalid tensor data
    /// - Unsupported tensor format
    fn read_tensor(&mut self, name: &str) -> Result<Tensor>;

    /// Lists all available tensor names in the weight source.
    ///
    /// This method is useful for debugging and for discovering the structure
    /// of saved model weights.
    ///
    /// # Returns
    ///
    /// A vector containing the names of all available tensors.
    fn list_tensors(&self) -> Vec<String>;
}

/// Text tokenization interface for transformer models.
///
/// The `Tokenizer` trait provides methods for converting between text and token IDs,
/// which is essential for preparing input data for transformer models. Implementations
/// may use various tokenization algorithms such as WordPiece, BPE, or SentencePiece.
///
/// # Thread Safety
///
/// Tokenizers must be `Send + Sync` to support concurrent tokenization.
///
/// # Example
///
/// ```no_run
/// use trustformers_core::traits::{Tokenizer, TokenizedInput};
/// use trustformers_core::error::Result;
///
/// struct BertTokenizer {
///     vocab: std::collections::HashMap<String, u32>,
///     // ... other fields
/// }
///
/// impl Tokenizer for BertTokenizer {
///     fn encode(&self, text: &str) -> Result<TokenizedInput> {
///         // Tokenize text into subwords
///         let tokens = vec![101, 2023, 2003, 1037, 3231, 102]; // [CLS] this is a test [SEP]
///         Ok(TokenizedInput {
///             input_ids: tokens,
///             attention_mask: vec![1; 6],
///             token_type_ids: Some(vec![0; 6]),
///         })
///     }
///
///     fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
///         // Encode two texts for tasks like question answering
///         let tokens1 = self.encode(text)?;
///         let tokens2 = self.encode(text2)?;
///         // Combine tokens with separator
///         let mut combined_tokens = tokens1.token_ids;
///         combined_tokens.extend_from_slice(&tokens2.token_ids);
///         Ok(TokenizedInput {
///             token_ids: combined_tokens,
///             attention_mask: vec![1; combined_tokens.len()],
///             token_type_ids: Some(vec![0; tokens1.token_ids.len()].into_iter()
///                 .chain(vec![1; tokens2.token_ids.len()]).collect()),
///         })
///     }
///
///     fn decode(&self, ids: &[u32]) -> Result<String> {
///         // Convert token IDs back to text
///         Ok("this is a test".to_string())
///     }
///
///     fn vocab_size(&self) -> usize {
///         30522 // BERT base vocabulary size
///     }
/// }
/// ```
pub trait Tokenizer: Send + Sync {
    /// Encodes a single text string into tokens.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to tokenize
    ///
    /// # Returns
    ///
    /// Returns a `TokenizedInput` containing:
    /// - `input_ids`: The token IDs
    /// - `attention_mask`: Binary mask indicating real vs padding tokens
    /// - `token_type_ids`: Optional segment IDs for models like BERT
    ///
    /// # Errors
    ///
    /// May return errors for:
    /// - Invalid UTF-8 sequences
    /// - Text exceeding maximum length
    /// - Unknown tokens that cannot be handled
    fn encode(&self, text: &str) -> Result<TokenizedInput>;

    /// Encodes a pair of texts for sequence-pair tasks.
    ///
    /// This method is used for tasks that require two input sequences,
    /// such as question answering, textual entailment, or sequence classification.
    ///
    /// # Arguments
    ///
    /// * `text` - The first text sequence
    /// * `text2` - The second text sequence
    ///
    /// # Returns
    ///
    /// Returns a `TokenizedInput` with both sequences encoded and separated
    /// by appropriate special tokens (e.g., [SEP] for BERT).
    ///
    /// # Errors
    ///
    /// May return errors for the same reasons as `encode()`.
    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput>;

    /// Decodes token IDs back into text.
    ///
    /// # Arguments
    ///
    /// * `ids` - The token IDs to decode
    ///
    /// # Returns
    ///
    /// Returns the decoded text string. Special tokens may be included
    /// or excluded depending on the implementation.
    ///
    /// # Errors
    ///
    /// May return errors for:
    /// - Invalid token IDs
    /// - Decoding errors
    fn decode(&self, ids: &[u32]) -> Result<String>;

    /// Returns the size of the tokenizer's vocabulary.
    ///
    /// # Returns
    ///
    /// The total number of tokens in the vocabulary.
    fn vocab_size(&self) -> usize;

    /// Returns a copy of the vocabulary as a mapping from tokens to IDs.
    ///
    /// # Returns
    ///
    /// A HashMap containing the vocabulary mapping.
    fn get_vocab(&self) -> std::collections::HashMap<String, u32>;

    /// Converts a token string to its corresponding ID.
    ///
    /// # Arguments
    ///
    /// * `token` - The token string to convert
    ///
    /// # Returns
    ///
    /// The token ID if the token exists in the vocabulary, None otherwise.
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// Converts a token ID to its corresponding token string.
    ///
    /// # Arguments
    ///
    /// * `id` - The token ID to convert
    ///
    /// # Returns
    ///
    /// The token string if the ID exists in the vocabulary, None otherwise.
    fn id_to_token(&self, id: u32) -> Option<String>;
}

/// Represents tokenized input ready for model consumption.
///
/// `TokenizedInput` contains all the necessary components for feeding
/// text data into a transformer model after tokenization.
///
/// # Fields
///
/// * `input_ids` - The token IDs representing the input text
/// * `attention_mask` - Binary mask (0 or 1) indicating which tokens are real vs padding
/// * `token_type_ids` - Optional segment IDs for models that use them (e.g., BERT)
///
/// # Example
///
/// ```no_run
/// use trustformers_core::traits::TokenizedInput;
///
/// let input = TokenizedInput {
///     input_ids: vec![101, 2023, 2003, 1037, 3231, 102], // [CLS] this is a test [SEP]
///     attention_mask: vec![1, 1, 1, 1, 1, 1], // All tokens are real (not padding)
///     token_type_ids: Some(vec![0, 0, 0, 0, 0, 0]), // All tokens from first segment
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub struct TokenizedInput {
    /// Token IDs representing the encoded text.
    /// These correspond to entries in the tokenizer's vocabulary.
    pub input_ids: Vec<u32>,

    /// Binary attention mask indicating real tokens (1) vs padding tokens (0).
    /// This prevents the model from attending to padding tokens.
    pub attention_mask: Vec<u8>,

    /// Optional token type IDs for distinguishing between different segments.
    /// Used by models like BERT for tasks involving multiple sequences.
    /// Typically 0 for the first sequence and 1 for the second sequence.
    pub token_type_ids: Option<Vec<u32>>,

    /// Optional special tokens mask indicating special tokens (1) vs regular tokens (0).
    /// Used to identify tokens like [CLS], [SEP], [PAD] etc.
    pub special_tokens_mask: Option<Vec<u8>>,

    /// Optional offset mapping showing character positions of tokens in original text.
    /// Each tuple contains (start_pos, end_pos) character offsets.
    pub offset_mapping: Option<Vec<(usize, usize)>>,

    /// Optional overflowing tokens when text exceeds max length.
    /// Contains tokens that were truncated from the input.
    pub overflowing_tokens: Option<Vec<u32>>,
}

impl TokenizedInput {
    /// Create a new TokenizedInput with minimal required fields
    pub fn new(input_ids: Vec<u32>, attention_mask: Vec<u8>) -> Self {
        Self {
            input_ids,
            attention_mask,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        }
    }

    /// Create a new TokenizedInput with token type IDs
    pub fn with_token_type_ids(
        input_ids: Vec<u32>,
        attention_mask: Vec<u8>,
        token_type_ids: Option<Vec<u32>>,
    ) -> Self {
        Self {
            input_ids,
            attention_mask,
            token_type_ids,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        }
    }
}

/// Parameter optimization algorithms for training neural networks.
///
/// The `Optimizer` trait defines the interface for gradient-based optimization
/// algorithms such as SGD, Adam, AdamW, etc. Optimizers update model parameters
/// based on computed gradients to minimize the loss function.
///
/// # Thread Safety
///
/// Optimizers must be `Send + Sync` to support distributed training.
///
/// # Example
///
/// ```no_run
/// use trustformers_core::traits::Optimizer;
/// use trustformers_core::tensor::Tensor;
/// use trustformers_core::error::Result;
///
/// struct SGD {
///     learning_rate: f32,
///     momentum: f32,
///     velocity: std::collections::HashMap<String, Tensor>,
/// }
///
/// impl Optimizer for SGD {
///     fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
///         // SGD with momentum: v = momentum * v - lr * grad
///         // parameter += v
///         Ok(())
///     }
///
///     fn zero_grad(&mut self) {
///         // Clear accumulated gradients
///     }
///
///     fn step(&mut self) {
///         // Apply updates to all parameters
///     }
///
///     fn get_lr(&self) -> f32 {
///         self.learning_rate
///     }
///
///     fn set_lr(&mut self, lr: f32) {
///         self.learning_rate = lr;
///     }
/// }
/// ```
pub trait Optimizer: Send + Sync {
    /// Updates a parameter based on its gradient.
    ///
    /// # Arguments
    ///
    /// * `parameter` - The parameter tensor to update
    /// * `grad` - The gradient tensor for this parameter
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful update, or an error if the update fails.
    ///
    /// # Errors
    ///
    /// May return errors for:
    /// - Mismatched tensor shapes
    /// - Numerical errors (e.g., NaN or Inf values)
    /// - Memory allocation failures
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()>;

    /// Clears all accumulated gradients.
    ///
    /// This should be called before each backward pass to ensure
    /// gradients don't accumulate across batches (unless gradient
    /// accumulation is intentionally being used).
    fn zero_grad(&mut self);

    /// Performs a single optimization step.
    ///
    /// This method applies all pending parameter updates. It should be
    /// called after gradients have been computed for all parameters.
    fn step(&mut self);

    /// Gets the current learning rate.
    ///
    /// # Returns
    ///
    /// The current learning rate value.
    fn get_lr(&self) -> f32;

    /// Sets a new learning rate.
    ///
    /// # Arguments
    ///
    /// * `lr` - The new learning rate value
    ///
    /// # Note
    ///
    /// This is useful for implementing learning rate schedules.
    fn set_lr(&mut self, lr: f32);

    /// Accumulates gradients for gradient accumulation.
    ///
    /// This method is used when training with gradient accumulation,
    /// where gradients from multiple batches are accumulated before
    /// performing an update step.
    ///
    /// # Arguments
    ///
    /// * `parameter` - The parameter tensor
    /// * `grad` - The gradient to accumulate
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if accumulation fails.
    ///
    /// # Default Implementation
    ///
    /// The default implementation simply calls `update()`. Override this
    /// method for optimizers that need special gradient accumulation logic.
    fn accumulate_grad(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        // Default implementation: just store the gradient for later use
        self.update(parameter, grad)
    }

    /// Applies accumulated gradients after gradient accumulation.
    ///
    /// This method should be called after accumulating gradients from
    /// multiple batches to apply the averaged update.
    ///
    /// # Arguments
    ///
    /// * `accumulation_steps` - The number of accumulation steps performed
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if application fails.
    ///
    /// # Default Implementation
    ///
    /// The default implementation is a no-op. Override this method for
    /// optimizers that implement gradient accumulation.
    fn apply_accumulated_grads(&mut self, accumulation_steps: usize) -> Result<()> {
        // Default implementation: no-op, override if needed
        let _ = accumulation_steps;
        Ok(())
    }
}

/// Weight initialization strategies for neural network parameters.
///
/// The `ParameterInit` trait provides various initialization methods that help
/// ensure proper gradient flow and training stability. Different initialization
/// strategies are optimal for different activation functions and architectures.
///
/// # Example
///
/// ```no_run
/// use trustformers_core::traits::ParameterInit;
/// use trustformers_core::tensor::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut weight = Tensor::zeros(&[768, 768])?;
///
/// // Initialize with Xavier/Glorot uniform for tanh activations
/// weight.xavier_uniform();
///
/// // Or use Kaiming/He initialization for ReLU activations
/// weight.kaiming_normal("fan_in", "relu");
/// # Ok(())
/// # }
/// ```
pub trait ParameterInit {
    /// Initializes the tensor with values from a normal distribution.
    ///
    /// # Arguments
    ///
    /// * `mean` - The mean of the normal distribution
    /// * `std` - The standard deviation of the normal distribution
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use trustformers_core::tensor::Tensor;
    /// # use trustformers_core::traits::ParameterInit;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut tensor = Tensor::zeros(&[100, 100])?;
    /// tensor.normal(0.0, 0.02); // Common for transformer embeddings
    /// # Ok(())
    /// # }
    /// ```
    fn normal(&mut self, mean: f32, std: f32);

    /// Initializes the tensor with values from a uniform distribution.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value (inclusive)
    /// * `max` - The maximum value (exclusive)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use trustformers_core::tensor::Tensor;
    /// # use trustformers_core::traits::ParameterInit;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut tensor = Tensor::zeros(&[100, 100])?;
    /// tensor.uniform(-0.1, 0.1);
    /// # Ok(())
    /// # }
    /// ```
    fn uniform(&mut self, min: f32, max: f32);

    /// Xavier/Glorot uniform initialization.
    ///
    /// Initializes weights to maintain variance across layers, optimal for
    /// tanh and sigmoid activations. The range is [-x, x] where
    /// x = sqrt(6 / (fan_in + fan_out)).
    ///
    /// # References
    ///
    /// Glorot & Bengio (2010): "Understanding the difficulty of training
    /// deep feedforward neural networks"
    fn xavier_uniform(&mut self);

    /// Xavier/Glorot normal initialization.
    ///
    /// Similar to `xavier_uniform` but uses a normal distribution with
    /// std = sqrt(2 / (fan_in + fan_out)).
    fn xavier_normal(&mut self);

    /// Kaiming/He uniform initialization.
    ///
    /// Designed for ReLU and similar activations. Maintains variance when
    /// half of the neurons are zeroed out by ReLU.
    ///
    /// # Arguments
    ///
    /// * `mode` - Either "fan_in" or "fan_out", determines which dimension to use
    /// * `nonlinearity` - The activation function ("relu", "leaky_relu", "linear")
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use trustformers_core::tensor::Tensor;
    /// # use trustformers_core::traits::ParameterInit;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut conv_weight = Tensor::zeros(&[64, 32, 3, 3])?;
    /// conv_weight.kaiming_uniform("fan_in", "relu");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # References
    ///
    /// He et al. (2015): "Delving Deep into Rectifiers: Surpassing
    /// Human-Level Performance on ImageNet Classification"
    fn kaiming_uniform(&mut self, mode: &str, nonlinearity: &str);

    /// Kaiming/He normal initialization.
    ///
    /// Similar to `kaiming_uniform` but uses a normal distribution.
    /// Generally preferred over uniform for deeper networks.
    ///
    /// # Arguments
    ///
    /// * `mode` - Either "fan_in" or "fan_out"
    /// * `nonlinearity` - The activation function ("relu", "leaky_relu", "linear")
    fn kaiming_normal(&mut self, mode: &str, nonlinearity: &str);
}
