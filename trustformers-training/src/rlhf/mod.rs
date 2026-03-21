//! Reinforcement Learning from Human Feedback (RLHF) infrastructure.
//!
//! This module provides implementations for RLHF training including:
//! - PPO (Proximal Policy Optimization) for language models
//! - Reward model training infrastructure
//! - Human feedback collection interface
//! - Constitutional AI training support
//! - Direct Preference Optimization (DPO)

pub mod config;
pub mod dpo;
pub mod feedback;
pub mod ppo;
pub mod reward_model;
pub mod trainer;

pub use config::{DPOConfig as ConfigDPOConfig, *};
pub use dpo::{DPOConfig, *};
pub use feedback::*;
pub use ppo::*;
pub use reward_model::*;
pub use trainer::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// RLHF training phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RLHFPhase {
    /// Supervised fine-tuning phase
    SFT,
    /// Reward model training
    RewardModel,
    /// PPO training with human feedback
    PPO,
    /// Direct Preference Optimization
    DPO,
    /// Constitutional AI training
    Constitutional,
}

/// Training metrics for RLHF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLHFMetrics {
    /// Current training phase
    pub phase: RLHFPhase,
    /// Policy model loss
    pub policy_loss: Option<f32>,
    /// Value model loss
    pub value_loss: Option<f32>,
    /// Reward model accuracy
    pub reward_accuracy: Option<f32>,
    /// KL divergence from reference model
    pub kl_divergence: f32,
    /// Average reward score
    pub avg_reward: f32,
    /// PPO objective
    pub ppo_objective: Option<f32>,
    /// Advantage estimates
    pub advantages: Vec<f32>,
    /// Response lengths
    pub response_lengths: Vec<usize>,
    /// Constitutional AI violations
    pub constitutional_violations: Option<usize>,
}

impl Default for RLHFMetrics {
    fn default() -> Self {
        Self {
            phase: RLHFPhase::SFT,
            policy_loss: None,
            value_loss: None,
            reward_accuracy: None,
            kl_divergence: 0.0,
            avg_reward: 0.0,
            ppo_objective: None,
            advantages: Vec::new(),
            response_lengths: Vec::new(),
            constitutional_violations: None,
        }
    }
}

/// Human feedback data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanFeedback {
    /// Unique identifier for the feedback
    pub id: String,
    /// The prompt that was given to the model
    pub prompt: String,
    /// The model's response
    pub response: String,
    /// Human rating (typically 1-5 or 1-10)
    pub rating: f32,
    /// Optional detailed feedback text
    pub feedback_text: Option<String>,
    /// Timestamp of feedback
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Annotator ID
    pub annotator_id: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Preference pair for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferencePair {
    /// The prompt/context
    pub prompt: String,
    /// Preferred response
    pub chosen: String,
    /// Rejected response
    pub rejected: String,
    /// Confidence in the preference (0.0 to 1.0)
    pub confidence: f32,
    /// Optional reasoning for the preference
    pub reasoning: Option<String>,
}

/// Constitutional AI principle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalPrinciple {
    /// Name of the principle
    pub name: String,
    /// Description of the principle
    pub description: String,
    /// Weight/importance of this principle
    pub weight: f32,
    /// Evaluation function or criteria
    pub criteria: String,
    /// Examples of violations
    pub violation_examples: Vec<String>,
    /// Examples of adherence
    pub adherence_examples: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rlhf_metrics_default() {
        let metrics = RLHFMetrics::default();
        assert_eq!(metrics.phase, RLHFPhase::SFT);
        assert_eq!(metrics.kl_divergence, 0.0);
        assert_eq!(metrics.avg_reward, 0.0);
        assert!(metrics.advantages.is_empty());
    }

    #[test]
    fn test_human_feedback_creation() {
        let feedback = HumanFeedback {
            id: "test_id".to_string(),
            prompt: "Test prompt".to_string(),
            response: "Test response".to_string(),
            rating: 4.5,
            feedback_text: Some("Good response".to_string()),
            timestamp: chrono::Utc::now(),
            annotator_id: Some("annotator_1".to_string()),
            metadata: HashMap::new(),
        };

        assert_eq!(feedback.rating, 4.5);
        assert_eq!(feedback.prompt, "Test prompt");
    }

    #[test]
    fn test_preference_pair() {
        let pair = PreferencePair {
            prompt: "What is AI?".to_string(),
            chosen: "AI is artificial intelligence".to_string(),
            rejected: "AI is nothing".to_string(),
            confidence: 0.9,
            reasoning: Some("First response is more accurate".to_string()),
        };

        assert_eq!(pair.confidence, 0.9);
        assert!(pair.reasoning.is_some());
    }

    #[test]
    fn test_constitutional_principle() {
        let principle = ConstitutionalPrinciple {
            name: "Helpfulness".to_string(),
            description: "The AI should be helpful and informative".to_string(),
            weight: 1.0,
            criteria: "Response provides useful information".to_string(),
            violation_examples: vec!["Refusing to help with reasonable requests".to_string()],
            adherence_examples: vec!["Providing clear, helpful answers".to_string()],
        };

        assert_eq!(principle.weight, 1.0);
        assert_eq!(principle.violation_examples.len(), 1);
    }
}
