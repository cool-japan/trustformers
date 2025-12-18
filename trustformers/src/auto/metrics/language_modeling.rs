//! # Language Modeling Metrics for TrustformeRS
//!
//! This module provides evaluation metrics for language modeling tasks, including
//! masked language modeling, fill-mask, and autoregressive language modeling.
//!
//! ## Overview
//!
//! The `LanguageModelingMetric` implementation focuses on perplexity calculation,
//! which is the standard evaluation metric for language models. Perplexity measures
//! how well a probability model predicts a sample sequence.
//!
//! ## Features
//!
//! - **Perplexity calculation**: Standard language modeling metric
//! - **Log-likelihood accumulation**: Efficient streaming computation
//! - **Token-level evaluation**: Works with probability distributions over vocabulary
//! - **Numerical stability**: Handles edge cases and numerical precision issues
//!
//! ## Usage Examples
//!
//! ### Basic Language Model Evaluation
//!
//! ```rust
//! use trustformers::auto::metrics::{LanguageModelingMetric, MetricInput, Metric};
//!
//! let mut metric = LanguageModelingMetric::new();
//!
//! // Probability distributions for each token position
//! let probabilities = MetricInput::Probabilities(vec![
//!     vec![
//!         vec![0.1, 0.8, 0.1], // Position 0: high prob for token 1
//!         vec![0.3, 0.4, 0.3], // Position 1: moderate uncertainty
//!     ]
//! ]);
//!
//! // Actual token IDs that occurred
//! let tokens = MetricInput::Tokens(vec![vec![1, 1]]);
//!
//! metric.add_batch(&probabilities, &tokens)?;
//!
//! // Compute results
//! let result = metric.compute()?;
//! println!("Perplexity: {:.3}", result.value);
//! println!("Log-likelihood: {:.3}", result.details.get("log_likelihood").unwrap());
//! ```
//!
//! ### Streaming Evaluation
//!
//! ```rust
//! use trustformers::auto::metrics::{LanguageModelingMetric, MetricInput, Metric};
//!
//! let mut metric = LanguageModelingMetric::new();
//!
//! // Process multiple batches
//! for batch_idx in 0..3 {
//!     let probabilities = MetricInput::Probabilities(vec![
//!         vec![vec![0.2, 0.7, 0.1]]
//!     ]);
//!     let tokens = MetricInput::Tokens(vec![vec![1]]);
//!
//!     metric.add_batch(&probabilities, &tokens)?;
//! }
//!
//! let result = metric.compute()?;
//! ```
//!
//! ## Implementation Details
//!
//! ### Perplexity Calculation
//!
//! 1. **Log-likelihood accumulation**: Sum log probabilities of actual tokens
//! 2. **Average log-likelihood**: Divide by total number of tokens
//! 3. **Perplexity**: `exp(-average_log_likelihood)`
//!
//! ### Mathematical Definition
//!
//! Given a sequence of tokens with probabilities:
//! - Log-likelihood: `LL = Σ log(P(token_i))`
//! - Average log-likelihood: `avg_LL = LL / N`
//! - Perplexity: `PPL = exp(-avg_LL)`
//!
//! Lower perplexity indicates better language model performance.
//!
//! ### Performance Characteristics
//!
//! - **Time complexity**: O(N) where N is total number of tokens
//! - **Space complexity**: O(1) for metric state (accumulates scalars)
//! - **Numerical stability**: Uses log probabilities to avoid underflow

use super::{Metric, MetricInput, MetricResult};
use crate::error::{Result, TrustformersError};
use std::collections::HashMap;

/// Language modeling metric implementation
///
/// Provides perplexity calculation for language modeling tasks. Perplexity
/// measures how well a probability model predicts token sequences.
///
/// ## Design Principles
///
/// - **Accumulative**: Collects log-likelihoods over multiple batches
/// - **Streaming**: Processes sequences incrementally for memory efficiency
/// - **Stable**: Uses log probabilities for numerical stability
/// - **Standard**: Implements the widely-accepted perplexity metric
///
/// ## Supported Input Types
///
/// - `Probabilities`: Token probability distributions (predictions)
/// - `Tokens`: Actual token IDs (references)
///
/// ## Perplexity Interpretation
///
/// - **Lower is better**: Lower perplexity indicates better prediction
/// - **Range**: [1, ∞) where 1 is perfect prediction
/// - **Baseline**: Random guessing gives perplexity equal to vocabulary size
#[derive(Debug, Clone)]
pub struct LanguageModelingMetric {
    /// Accumulated log-likelihood across all tokens
    log_likelihood: f64,
    /// Total number of tokens processed
    num_tokens: usize,
}

impl LanguageModelingMetric {
    /// Create a new language modeling metric instance
    ///
    /// Initializes an empty metric ready to accumulate log-likelihoods and token counts.
    ///
    /// # Returns
    ///
    /// New `LanguageModelingMetric` instance with empty state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::LanguageModelingMetric;
    ///
    /// let metric = LanguageModelingMetric::new();
    /// assert_eq!(metric.name(), "language_modeling");
    /// ```
    pub fn new() -> Self {
        Self {
            log_likelihood: 0.0,
            num_tokens: 0,
        }
    }
}

impl Metric for LanguageModelingMetric {
    /// Add a batch of probability distributions and token sequences
    ///
    /// Accumulates log-likelihood data for perplexity computation. Expects
    /// probability distributions over vocabulary and corresponding token IDs.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Probability distributions over vocabulary (Probabilities)
    /// * `references` - Actual token IDs that occurred (Tokens)
    ///
    /// # Input Format Requirements
    ///
    /// - **Probabilities**: Shape `[batch_size, sequence_length, vocab_size]`
    /// - **Tokens**: Shape `[batch_size, sequence_length]`
    /// - Token IDs must be valid indices into the probability distributions
    /// - Probabilities should sum to 1.0 for each position (not enforced)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if input formats are incompatible.
    ///
    /// # Errors
    ///
    /// - `InvalidInput`: If input types are not Probabilities and Tokens
    /// - Silent handling: Invalid token IDs (>= vocab_size) are skipped
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{LanguageModelingMetric, MetricInput, Metric};
    ///
    /// let mut metric = LanguageModelingMetric::new();
    ///
    /// // Probability distributions for 2 sequences
    /// let probabilities = MetricInput::Probabilities(vec![
    ///     vec![vec![0.1, 0.9], vec![0.3, 0.7]], // Sequence 1: 2 positions
    ///     vec![vec![0.5, 0.5]],                  // Sequence 2: 1 position
    /// ]);
    ///
    /// // Corresponding token IDs
    /// let tokens = MetricInput::Tokens(vec![
    ///     vec![1, 1], // Sequence 1: tokens at each position
    ///     vec![0],    // Sequence 2: token at position
    /// ]);
    ///
    /// metric.add_batch(&probabilities, &tokens)?;
    /// ```
    fn add_batch(&mut self, predictions: &MetricInput, references: &MetricInput) -> Result<()> {
        match (predictions, references) {
            (MetricInput::Probabilities(probs), MetricInput::Tokens(tokens)) => {
                // Process each sequence pair
                for (prob_seq, token_seq) in probs.iter().zip(tokens.iter()) {
                    // Process each position in the sequence
                    for (prob_dist, &token) in prob_seq.iter().zip(token_seq.iter()) {
                        // Ensure token ID is valid for this vocabulary
                        if (token as usize) < prob_dist.len() {
                            // Get probability of the actual token
                            let token_prob = prob_dist[token as usize];

                            // Add log probability (handle zero probability gracefully)
                            if token_prob > 0.0 {
                                self.log_likelihood += token_prob.ln() as f64;
                            } else {
                                // Assign very small probability to avoid -inf
                                self.log_likelihood += f64::NEG_INFINITY.max(-50.0); // Cap at -50
                            }

                            self.num_tokens += 1;
                        }
                        // Invalid token IDs are silently skipped
                    }
                }
                Ok(())
            },
            _ => Err(TrustformersError::invalid_input_simple("Invalid input types for language modeling metric: expected Probabilities for predictions and Tokens for references".to_string()
            )),
        }
    }

    /// Compute language modeling metrics
    ///
    /// Calculates perplexity and related metrics based on accumulated log-likelihoods.
    ///
    /// # Returns
    ///
    /// `MetricResult` containing:
    /// - **Primary value**: Perplexity (lower is better)
    /// - **Details**:
    ///   - `perplexity`: The computed perplexity value (same as primary)
    ///   - `log_likelihood`: Total accumulated log-likelihood
    ///   - `num_tokens`: Total number of tokens processed
    ///
    /// # Errors
    ///
    /// - Returns valid results even with no tokens (perplexity = ∞)
    /// - No errors are returned from this method
    ///
    /// # Perplexity Calculation
    ///
    /// ```text
    /// perplexity = exp(-log_likelihood / num_tokens)
    /// ```
    ///
    /// # Special Cases
    ///
    /// - **No tokens**: Returns infinite perplexity
    /// - **Very negative log-likelihood**: Capped to prevent numerical issues
    /// - **Zero probabilities**: Handled as very small probabilities
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{LanguageModelingMetric, MetricInput, Metric};
    ///
    /// let mut metric = LanguageModelingMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Probabilities(vec![vec![vec![0.5, 0.5]]]),
    ///     &MetricInput::Tokens(vec![vec![0]])
    /// )?;
    ///
    /// let result = metric.compute()?;
    /// assert_eq!(result.name, "language_modeling");
    /// assert!(result.value >= 1.0); // Perplexity >= 1
    /// assert!(result.details.contains_key("perplexity"));
    /// assert!(result.details.contains_key("log_likelihood"));
    /// assert!(result.details.contains_key("num_tokens"));
    /// ```
    fn compute(&self) -> Result<MetricResult> {
        // Calculate perplexity
        let perplexity = if self.num_tokens > 0 {
            (-self.log_likelihood / self.num_tokens as f64).exp()
        } else {
            f64::INFINITY
        };

        // Build result details
        let mut details = HashMap::new();
        details.insert("perplexity".to_string(), perplexity);
        details.insert("log_likelihood".to_string(), self.log_likelihood);
        details.insert("num_tokens".to_string(), self.num_tokens as f64);

        Ok(MetricResult {
            name: "language_modeling".to_string(),
            value: perplexity, // Primary metric is perplexity
            details,
            metadata: HashMap::new(),
        })
    }

    /// Reset the metric state
    ///
    /// Clears all accumulated log-likelihood and token count data, preparing
    /// the metric for a new evaluation run.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{LanguageModelingMetric, MetricInput, Metric};
    ///
    /// let mut metric = LanguageModelingMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Probabilities(vec![vec![vec![0.5, 0.5]]]),
    ///     &MetricInput::Tokens(vec![vec![0]])
    /// )?;
    ///
    /// metric.reset();
    ///
    /// let result = metric.compute()?;
    /// assert_eq!(result.value, f64::INFINITY); // No data = infinite perplexity
    /// ```
    fn reset(&mut self) {
        self.log_likelihood = 0.0;
        self.num_tokens = 0;
    }

    /// Get the metric name
    ///
    /// Returns the identifier for this metric type, used in logging and results.
    ///
    /// # Returns
    ///
    /// String slice "language_modeling"
    fn name(&self) -> &str {
        "language_modeling"
    }
}

impl Default for LanguageModelingMetric {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_modeling_metric_basic() {
        let mut metric = LanguageModelingMetric::new();

        // High probability for correct tokens should give low perplexity
        let probabilities = MetricInput::Probabilities(vec![vec![
            vec![0.1, 0.9], // High prob for token 1
            vec![0.2, 0.8], // High prob for token 1
        ]]);
        let tokens = MetricInput::Tokens(vec![vec![1, 1]]);

        metric.add_batch(&probabilities, &tokens).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.name, "language_modeling");
        assert!(result.value >= 1.0); // Perplexity should be >= 1
        assert!(result.details.contains_key("perplexity"));
        assert!(result.details.contains_key("log_likelihood"));
        assert_eq!(result.details.get("num_tokens"), Some(&2.0));
    }

    #[test]
    fn test_language_modeling_metric_perfect_prediction() {
        let mut metric = LanguageModelingMetric::new();

        // Perfect prediction (probability 1.0 for correct token)
        let probabilities = MetricInput::Probabilities(vec![vec![vec![0.0, 1.0]]]);
        let tokens = MetricInput::Tokens(vec![vec![1]]);

        metric.add_batch(&probabilities, &tokens).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.value, 1.0); // Perfect prediction gives perplexity 1
    }

    #[test]
    fn test_language_modeling_metric_random_prediction() {
        let mut metric = LanguageModelingMetric::new();

        // Random prediction (equal probabilities)
        let probabilities = MetricInput::Probabilities(vec![vec![vec![0.5, 0.5]]]);
        let tokens = MetricInput::Tokens(vec![vec![0]]);

        metric.add_batch(&probabilities, &tokens).unwrap();

        let result = metric.compute().unwrap();
        // Random binary prediction gives perplexity ~2 (use tolerance for f64)
        assert!((result.value - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_language_modeling_metric_zero_probability() {
        let mut metric = LanguageModelingMetric::new();

        // Zero probability for actual token
        let probabilities = MetricInput::Probabilities(vec![vec![vec![1.0, 0.0]]]);
        let tokens = MetricInput::Tokens(vec![vec![1]]);

        metric.add_batch(&probabilities, &tokens).unwrap();

        let result = metric.compute().unwrap();
        // Should handle zero probability gracefully
        assert!(result.value > 1.0);
    }

    #[test]
    fn test_language_modeling_metric_invalid_token() {
        let mut metric = LanguageModelingMetric::new();

        // Token ID out of vocabulary range
        let probabilities = MetricInput::Probabilities(vec![vec![vec![0.5, 0.5]]]);
        let tokens = MetricInput::Tokens(vec![vec![5]]); // Invalid: only 2 vocab items

        metric.add_batch(&probabilities, &tokens).unwrap();

        let result = metric.compute().unwrap();
        // Should skip invalid tokens
        assert_eq!(result.value, f64::INFINITY); // No valid tokens processed
        assert_eq!(result.details.get("num_tokens"), Some(&0.0));
    }

    #[test]
    fn test_language_modeling_metric_multiple_batches() {
        let mut metric = LanguageModelingMetric::new();

        // First batch
        metric
            .add_batch(
                &MetricInput::Probabilities(vec![vec![vec![0.0, 1.0]]]),
                &MetricInput::Tokens(vec![vec![1]]),
            )
            .unwrap();

        // Second batch
        metric
            .add_batch(
                &MetricInput::Probabilities(vec![vec![vec![0.5, 0.5]]]),
                &MetricInput::Tokens(vec![vec![0]]),
            )
            .unwrap();

        let result = metric.compute().unwrap();
        // Should combine both batches: (ln(1.0) + ln(0.5)) / 2 = ln(0.5) / 2
        let expected = (-0.5_f64.ln() / 2.0).exp();
        assert!((result.value - expected).abs() < 1e-6);
        assert_eq!(result.details.get("num_tokens"), Some(&2.0));
    }

    #[test]
    fn test_language_modeling_metric_reset() {
        let mut metric = LanguageModelingMetric::new();

        metric
            .add_batch(
                &MetricInput::Probabilities(vec![vec![vec![0.5, 0.5]]]),
                &MetricInput::Tokens(vec![vec![0]]),
            )
            .unwrap();

        metric.reset();

        let result = metric.compute().unwrap();
        assert_eq!(result.value, f64::INFINITY); // No data = infinite perplexity
        assert_eq!(result.details.get("num_tokens"), Some(&0.0));
    }

    #[test]
    fn test_language_modeling_metric_invalid_input() {
        let mut metric = LanguageModelingMetric::new();

        let predictions = MetricInput::Text(vec!["hello".to_string()]);
        let references = MetricInput::Tokens(vec![vec![0]]);

        let result = metric.add_batch(&predictions, &references);
        assert!(result.is_err());
    }

    #[test]
    fn test_language_modeling_metric_empty_sequences() {
        let mut metric = LanguageModelingMetric::new();

        // Empty sequences should be handled gracefully
        let probabilities = MetricInput::Probabilities(vec![vec![]]);
        let tokens = MetricInput::Tokens(vec![vec![]]);

        metric.add_batch(&probabilities, &tokens).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.value, f64::INFINITY); // No tokens = infinite perplexity
    }
}
