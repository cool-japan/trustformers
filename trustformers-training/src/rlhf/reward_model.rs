//! Reward model implementation for RLHF training.

use crate::rlhf::{HumanFeedback, PreferencePair, RewardModelConfig, RewardModelType};
use anyhow::Result;
use scirs2_core::ndarray::{Array1, Array2}; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Reward model for scoring text generations
#[derive(Debug)]
pub struct RewardModel {
    config: RewardModelConfig,
    model_type: RewardModelType,
    parameters: HashMap<String, Array2<f32>>,
    #[allow(dead_code)]
    tokenizer: Option<RewardTokenizer>,
    training_data: Vec<PreferencePair>,
    statistics: RewardModelStatistics,
}

/// Simplified tokenizer for reward model
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RewardTokenizer {
    #[allow(dead_code)]
    vocab_size: usize,
    max_length: usize,
}

/// Training statistics for reward model
#[derive(Debug, Default)]
pub struct RewardModelStatistics {
    /// Training accuracy over time
    pub accuracies: Vec<f32>,
    /// Training losses over time
    pub losses: Vec<f32>,
    /// Validation accuracies
    pub val_accuracies: Vec<f32>,
    /// Validation losses
    pub val_losses: Vec<f32>,
    /// Number of training steps
    pub training_steps: usize,
    /// Average reward scores
    pub avg_reward_scores: Vec<f32>,
}

/// Reward prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardPrediction {
    /// Input text
    pub text: String,
    /// Predicted reward score
    pub score: f32,
    /// Confidence in the prediction (0.0 to 1.0)
    pub confidence: f32,
    /// Individual component scores (if using ensemble)
    pub component_scores: Vec<f32>,
    /// Attention weights (if available)
    pub attention_weights: Option<Array2<f32>>,
}

/// Training batch for reward model
#[derive(Debug, Clone)]
pub struct RewardBatch {
    /// Prompt texts
    pub prompts: Vec<String>,
    /// Response texts (chosen)
    pub chosen_responses: Vec<String>,
    /// Response texts (rejected)
    pub rejected_responses: Vec<String>,
    /// Tokenized chosen sequences
    pub chosen_tokens: Array2<u32>,
    /// Tokenized rejected sequences
    pub rejected_tokens: Array2<u32>,
    /// Preference labels (1.0 for chosen, 0.0 for rejected)
    pub labels: Array1<f32>,
}

/// Reward model training step result
#[derive(Debug, Clone)]
pub struct RewardTrainingResult {
    /// Training loss
    pub loss: f32,
    /// Training accuracy
    pub accuracy: f32,
    /// Average reward for chosen responses
    pub avg_chosen_reward: f32,
    /// Average reward for rejected responses
    pub avg_rejected_reward: f32,
    /// Reward margin (chosen - rejected)
    pub reward_margin: f32,
}

impl RewardModel {
    /// Create a new reward model
    pub fn new(config: RewardModelConfig) -> Result<Self> {
        let tokenizer = RewardTokenizer {
            vocab_size: 50000, // Default vocabulary size
            max_length: config.max_length,
        };

        Ok(Self {
            model_type: config.model_type,
            config,
            parameters: HashMap::new(),
            tokenizer: Some(tokenizer),
            training_data: Vec::new(),
            statistics: RewardModelStatistics::default(),
        })
    }

    /// Initialize model parameters
    pub fn initialize_parameters(&mut self, base_model_size: usize) -> Result<()> {
        match self.model_type {
            RewardModelType::Linear => {
                self.initialize_linear_head(base_model_size)?;
            },
            RewardModelType::MLP => {
                self.initialize_mlp_head(base_model_size)?;
            },
            RewardModelType::Transformer => {
                self.initialize_transformer_head(base_model_size)?;
            },
            RewardModelType::Ensemble => {
                self.initialize_ensemble_heads(base_model_size)?;
            },
        }
        Ok(())
    }

    /// Load training data from preference pairs
    pub fn load_training_data(&mut self, preference_pairs: Vec<PreferencePair>) -> Result<()> {
        self.training_data = preference_pairs;
        Ok(())
    }

    /// Load training data from human feedback
    pub fn load_from_human_feedback(&mut self, feedback: Vec<HumanFeedback>) -> Result<()> {
        // Convert human feedback to preference pairs
        let mut pairs = Vec::new();

        // Group feedback by prompt and create pairs based on ratings
        let mut feedback_by_prompt: HashMap<String, Vec<&HumanFeedback>> = HashMap::new();

        for fb in &feedback {
            feedback_by_prompt.entry(fb.prompt.clone()).or_default().push(fb);
        }

        for (prompt, prompt_feedback) in feedback_by_prompt {
            // Sort by rating descending
            let mut sorted_feedback = prompt_feedback;
            sorted_feedback.sort_by(|a, b| b.rating.partial_cmp(&a.rating).unwrap());

            // Create pairs between high and low rated responses
            for i in 0..sorted_feedback.len() {
                for j in (i + 1)..sorted_feedback.len() {
                    if sorted_feedback[i].rating > sorted_feedback[j].rating {
                        pairs.push(PreferencePair {
                            prompt: prompt.clone(),
                            chosen: sorted_feedback[i].response.clone(),
                            rejected: sorted_feedback[j].response.clone(),
                            confidence: (sorted_feedback[i].rating - sorted_feedback[j].rating)
                                / 5.0,
                            reasoning: None,
                        });
                    }
                }
            }
        }

        self.training_data = pairs;
        Ok(())
    }

    /// Train the reward model on preference data
    pub fn train_step(&mut self, batch: &RewardBatch) -> Result<RewardTrainingResult> {
        // Forward pass for chosen responses
        let chosen_rewards = self.forward(&batch.chosen_tokens)?;

        // Forward pass for rejected responses
        let rejected_rewards = self.forward(&batch.rejected_tokens)?;

        // Calculate ranking loss
        let loss = self.calculate_ranking_loss(&chosen_rewards, &rejected_rewards)?;

        // Calculate accuracy (how often chosen > rejected)
        let accuracy = self.calculate_accuracy(&chosen_rewards, &rejected_rewards)?;

        // Calculate statistics
        let avg_chosen_reward = chosen_rewards.mean().unwrap_or(0.0);
        let avg_rejected_reward = rejected_rewards.mean().unwrap_or(0.0);
        let reward_margin = avg_chosen_reward - avg_rejected_reward;

        // Update statistics
        self.statistics.training_steps += 1;
        self.statistics.losses.push(loss);
        self.statistics.accuracies.push(accuracy);
        self.statistics.avg_reward_scores.push(avg_chosen_reward);

        // Backward pass (simplified - would use actual gradients)
        self.backward_pass(&chosen_rewards, &rejected_rewards)?;

        Ok(RewardTrainingResult {
            loss,
            accuracy,
            avg_chosen_reward,
            avg_rejected_reward,
            reward_margin,
        })
    }

    /// Predict reward for a single text
    pub fn predict_reward(&self, text: &str) -> Result<RewardPrediction> {
        let tokens = self.tokenize(text)?;
        let tokens_array = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?;

        let rewards = self.forward(&tokens_array)?;
        let score = rewards[0];

        // Calculate confidence based on model uncertainty (simplified)
        let confidence = self.calculate_confidence(score)?;

        Ok(RewardPrediction {
            text: text.to_string(),
            score,
            confidence,
            component_scores: vec![score], // Single model for now
            attention_weights: None,
        })
    }

    /// Predict rewards for multiple texts
    pub fn predict_batch(&self, texts: &[String]) -> Result<Vec<RewardPrediction>> {
        let mut predictions = Vec::new();

        for text in texts {
            predictions.push(self.predict_reward(text)?);
        }

        Ok(predictions)
    }

    /// Compare two responses and return preference probability
    pub fn compare_responses(
        &self,
        prompt: &str,
        response_a: &str,
        response_b: &str,
    ) -> Result<f32> {
        let full_text_a = format!("{} {}", prompt, response_a);
        let full_text_b = format!("{} {}", prompt, response_b);

        let pred_a = self.predict_reward(&full_text_a)?;
        let pred_b = self.predict_reward(&full_text_b)?;

        // Convert to probability using sigmoid
        let score_diff = pred_a.score - pred_b.score;
        let preference_prob = 1.0 / (1.0 + (-score_diff).exp());

        Ok(preference_prob)
    }

    /// Get model statistics
    pub fn get_statistics(&self) -> &RewardModelStatistics {
        &self.statistics
    }

    // Private helper methods

    fn initialize_linear_head(&mut self, base_size: usize) -> Result<()> {
        let weight = Array2::zeros((base_size, 1));
        let bias = Array2::zeros((1, 1));

        self.parameters.insert("reward_head.weight".to_string(), weight);
        self.parameters.insert("reward_head.bias".to_string(), bias);
        Ok(())
    }

    fn initialize_mlp_head(&mut self, base_size: usize) -> Result<()> {
        let hidden_size = self.config.reward_head_hidden_size;

        let fc1_weight = Array2::zeros((base_size, hidden_size));
        let fc1_bias = Array2::zeros((1, hidden_size));
        let fc2_weight = Array2::zeros((hidden_size, 1));
        let fc2_bias = Array2::zeros((1, 1));

        self.parameters.insert("reward_head.fc1.weight".to_string(), fc1_weight);
        self.parameters.insert("reward_head.fc1.bias".to_string(), fc1_bias);
        self.parameters.insert("reward_head.fc2.weight".to_string(), fc2_weight);
        self.parameters.insert("reward_head.fc2.bias".to_string(), fc2_bias);
        Ok(())
    }

    fn initialize_transformer_head(&mut self, base_size: usize) -> Result<()> {
        // Simplified transformer head initialization
        let attention_weight = Array2::zeros((base_size, base_size));
        let output_weight = Array2::zeros((base_size, 1));

        self.parameters
            .insert("reward_head.attention.weight".to_string(), attention_weight);
        self.parameters.insert("reward_head.output.weight".to_string(), output_weight);
        Ok(())
    }

    fn initialize_ensemble_heads(&mut self, base_size: usize) -> Result<()> {
        // Initialize multiple heads for ensemble
        for i in 0..3 {
            let weight = Array2::zeros((base_size, 1));
            let bias = Array2::zeros((1, 1));

            self.parameters.insert(format!("reward_head_{}.weight", i), weight);
            self.parameters.insert(format!("reward_head_{}.bias", i), bias);
        }
        Ok(())
    }

    fn forward(&self, tokens: &Array2<u32>) -> Result<Array1<f32>> {
        // Simplified forward pass - would use actual model inference
        let batch_size = tokens.shape()[0];
        let mut rewards = Array1::zeros(batch_size);

        for i in 0..batch_size {
            // Simplified scoring based on sequence length and content
            let seq_len = tokens.row(i).iter().filter(|&&t| t != 0).count();
            let base_score = (seq_len as f32 / 100.0).min(1.0);

            // Add some variation based on content (simplified)
            let content_score =
                tokens.row(i).iter().map(|&t| (t % 10) as f32 / 10.0).sum::<f32>() / seq_len as f32;

            rewards[i] = base_score + content_score - 0.5;
        }

        Ok(rewards)
    }

    fn calculate_ranking_loss(&self, chosen: &Array1<f32>, rejected: &Array1<f32>) -> Result<f32> {
        let mut total_loss = 0.0;
        let batch_size = chosen.len();

        for i in 0..batch_size {
            // Ranking loss: -log(sigmoid(chosen - rejected))
            let score_diff = chosen[i] - rejected[i] - self.config.margin as f32;
            let sigmoid = 1.0 / (1.0 + (-score_diff).exp());
            total_loss -= sigmoid.ln();
        }

        Ok(total_loss / batch_size as f32)
    }

    fn calculate_accuracy(&self, chosen: &Array1<f32>, rejected: &Array1<f32>) -> Result<f32> {
        let mut correct = 0;
        let batch_size = chosen.len();

        for i in 0..batch_size {
            if chosen[i] > rejected[i] {
                correct += 1;
            }
        }

        Ok(correct as f32 / batch_size as f32)
    }

    fn calculate_confidence(&self, score: f32) -> Result<f32> {
        // Simple confidence based on score magnitude
        Ok((score.abs() / 2.0).min(1.0))
    }

    fn backward_pass(&mut self, _chosen: &Array1<f32>, _rejected: &Array1<f32>) -> Result<()> {
        // Simplified backward pass - would compute actual gradients
        // and update parameters
        Ok(())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Simplified tokenization
        Ok(text.chars().map(|c| c as u32).take(self.config.max_length).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_model_creation() {
        let config = RewardModelConfig::default();
        let model = RewardModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_reward_prediction() {
        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).unwrap();
        model.initialize_parameters(768).unwrap();

        let prediction = model.predict_reward("This is a test response");
        assert!(prediction.is_ok());

        let pred = prediction.unwrap();
        assert!(!pred.text.is_empty());
        assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
    }

    #[test]
    fn test_preference_pair_conversion() {
        let feedback = vec![
            HumanFeedback {
                id: "1".to_string(),
                prompt: "What is AI?".to_string(),
                response: "AI is artificial intelligence".to_string(),
                rating: 5.0,
                feedback_text: None,
                timestamp: chrono::Utc::now(),
                annotator_id: None,
                metadata: HashMap::new(),
            },
            HumanFeedback {
                id: "2".to_string(),
                prompt: "What is AI?".to_string(),
                response: "AI is bad".to_string(),
                rating: 2.0,
                feedback_text: None,
                timestamp: chrono::Utc::now(),
                annotator_id: None,
                metadata: HashMap::new(),
            },
        ];

        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).unwrap();
        let result = model.load_from_human_feedback(feedback);
        assert!(result.is_ok());
        assert_eq!(model.training_data.len(), 1);
    }

    #[test]
    fn test_response_comparison() {
        let config = RewardModelConfig::default();
        let mut model = RewardModel::new(config).unwrap();
        model.initialize_parameters(768).unwrap();

        let preference = model.compare_responses(
            "What is the capital of France?",
            "The capital of France is Paris.",
            "I don't know.",
        );

        assert!(preference.is_ok());
        let prob = preference.unwrap();
        assert!(prob >= 0.0 && prob <= 1.0);
    }
}
