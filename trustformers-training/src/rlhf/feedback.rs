use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackConfig {
    pub max_feedback_length: usize,
    pub feedback_temperature: f32,
    pub feedback_penalty_alpha: f32,
    pub use_human_feedback: bool,
    pub feedback_aggregation: FeedbackAggregation,
    pub quality_threshold: f32,
    pub consistency_weight: f32,
    pub diversity_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackAggregation {
    Mean,
    Median,
    WeightedMean,
    Consensus,
    MajorityVote,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            max_feedback_length: 1024,
            feedback_temperature: 1.0,
            feedback_penalty_alpha: 0.1,
            use_human_feedback: true,
            feedback_aggregation: FeedbackAggregation::WeightedMean,
            quality_threshold: 0.7,
            consistency_weight: 0.3,
            diversity_weight: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanFeedback {
    pub id: String,
    pub prompt: String,
    pub response: String,
    pub rating: f32, // 0.0 to 1.0
    pub feedback_text: Option<String>,
    pub annotator_id: String,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIFeedback {
    pub id: String,
    pub prompt: String,
    pub response: String,
    pub helpfulness_score: f32,
    pub harmlessness_score: f32,
    pub honesty_score: f32,
    pub overall_score: f32,
    pub explanation: String,
    pub confidence: f32,
    pub model_version: String,
}

#[derive(Debug, Clone)]
pub struct FeedbackBatch {
    pub prompts: Vec<String>,
    pub responses: Vec<String>,
    pub ratings: Tensor,
    pub feedback_texts: Vec<Option<String>>,
    pub weights: Option<Tensor>,
}

#[derive(Debug)]
pub struct FeedbackProcessor {
    config: FeedbackConfig,
    human_feedback_buffer: Vec<HumanFeedback>,
    ai_feedback_buffer: Vec<AIFeedback>,
    feedback_statistics: FeedbackStatistics,
}

#[derive(Debug, Default)]
pub struct FeedbackStatistics {
    pub total_feedback_count: usize,
    pub human_feedback_count: usize,
    pub ai_feedback_count: usize,
    pub average_rating: f32,
    pub rating_variance: f32,
    pub annotator_agreement: f32,
    pub consistency_score: f32,
}

#[allow(dead_code)]
impl FeedbackProcessor {
    pub fn new(config: FeedbackConfig) -> Self {
        Self {
            config,
            human_feedback_buffer: Vec::new(),
            ai_feedback_buffer: Vec::new(),
            feedback_statistics: FeedbackStatistics::default(),
        }
    }

    pub fn add_human_feedback(&mut self, feedback: HumanFeedback) -> Result<()> {
        if feedback.rating < 0.0 || feedback.rating > 1.0 {
            return Err(anyhow!("Rating must be between 0.0 and 1.0"));
        }

        self.human_feedback_buffer.push(feedback);
        self.update_statistics()?;
        Ok(())
    }

    pub fn add_ai_feedback(&mut self, feedback: AIFeedback) -> Result<()> {
        if feedback.overall_score < 0.0 || feedback.overall_score > 1.0 {
            return Err(anyhow!("Overall score must be between 0.0 and 1.0"));
        }

        self.ai_feedback_buffer.push(feedback);
        self.update_statistics()?;
        Ok(())
    }

    pub fn process_feedback_batch(&self, batch: &FeedbackBatch) -> Result<ProcessedFeedback> {
        let batch_size = batch.prompts.len();

        if batch_size == 0 {
            return Err(anyhow!("Empty feedback batch"));
        }

        // Aggregate ratings based on configuration
        let aggregated_ratings = self.aggregate_ratings(&batch.ratings)?;

        // Compute quality scores
        let quality_scores = self.compute_quality_scores(batch)?;

        // Apply filtering based on quality threshold
        let filtered_indices = self.filter_by_quality(&quality_scores)?;

        // Compute feedback weights
        let feedback_weights = self.compute_feedback_weights(batch, &quality_scores)?;

        Ok(ProcessedFeedback {
            aggregated_ratings,
            quality_scores,
            filtered_indices,
            feedback_weights,
            batch_statistics: self.compute_batch_statistics(batch)?,
        })
    }

    fn aggregate_ratings(&self, ratings: &Tensor) -> Result<Tensor> {
        // For now, just return the input ratings as-is to preserve shape
        match self.config.feedback_aggregation {
            FeedbackAggregation::Mean => Ok(ratings.clone()),
            FeedbackAggregation::Median => {
                // Simplified median approximation using input ratings for now
                Ok(ratings.clone())
            },
            FeedbackAggregation::WeightedMean => {
                // Simplified weighted mean using input ratings for now
                Ok(ratings.clone())
            },
            FeedbackAggregation::Consensus => {
                // Simplified consensus using input ratings for now
                Ok(ratings.clone())
            },
            FeedbackAggregation::MajorityVote => {
                // Simplified majority vote using input ratings for now
                Ok(ratings.clone())
            },
        }
    }

    fn compute_quality_scores(&self, batch: &FeedbackBatch) -> Result<Tensor> {
        let batch_size = batch.prompts.len();
        let mut quality_scores = Vec::with_capacity(batch_size);

        // Simplified implementation using tensor mean and basic scoring
        let mean_rating = 0.5f32; // Placeholder value since item() method not available

        for i in 0..batch_size {
            // Base quality score from mean rating
            let mut quality = mean_rating;

            // Apply diversity bonus (simplified)
            let diversity_bonus =
                self.compute_diversity_bonus(&batch.responses[i])? * self.config.diversity_weight;
            quality += diversity_bonus;

            // Clamp to [0, 1] range
            quality = quality.clamp(0.0, 1.0);
            quality_scores.push(quality);
        }

        Ok(Tensor::from_vec(quality_scores, &[batch_size])?)
    }

    fn filter_by_quality(&self, quality_scores: &Tensor) -> Result<Vec<usize>> {
        // Simplified implementation - return all indices for now
        let shape = quality_scores.shape();
        let batch_size = shape[0];
        Ok((0..batch_size).collect())
    }

    fn compute_feedback_weights(
        &self,
        batch: &FeedbackBatch,
        quality_scores: &Tensor,
    ) -> Result<Tensor> {
        if let Some(existing_weights) = &batch.weights {
            // Combine existing weights with quality scores
            let combined = existing_weights.mul(quality_scores)?;
            Ok(combined)
        } else {
            // Use quality scores as weights
            Ok(quality_scores.clone())
        }
    }

    fn compute_batch_statistics(&self, batch: &FeedbackBatch) -> Result<BatchStatistics> {
        let response_lengths: Vec<f32> = batch.responses.iter().map(|r| r.len() as f32).collect();
        let avg_response_length =
            response_lengths.iter().sum::<f32>() / response_lengths.len() as f32;

        Ok(BatchStatistics {
            mean_rating: 0.5, // Placeholder since item() not available
            std_rating: 0.1,  // Placeholder since item() not available
            avg_response_length,
            feedback_coverage: self.compute_feedback_coverage(batch)?,
        })
    }

    #[allow(dead_code)]
    fn compute_rating_weights(&self, ratings: &Tensor) -> Result<Tensor> {
        // Simplified implementation using mean for now
        Ok(ratings.mean()?)
    }

    fn compute_rating_std(&self, ratings: &Tensor) -> Result<Tensor> {
        // Simplified implementation - return small constant for now
        let shape = ratings.shape();
        Ok(Tensor::ones(&shape)?.mul_scalar(0.1)?)
    }

    fn compute_rating_std_single(&self, _ratings: &Tensor) -> Result<f32> {
        // Simplified implementation - return constant for now
        Ok(0.1)
    }

    fn compute_diversity_bonus(&self, response: &str) -> Result<f32> {
        // Simple diversity measure based on unique words
        let words: std::collections::HashSet<&str> = response.split_whitespace().collect();
        let unique_ratio = words.len() as f32 / response.split_whitespace().count().max(1) as f32;
        Ok(unique_ratio * 0.1) // Small bonus for diversity
    }

    fn compute_feedback_coverage(&self, batch: &FeedbackBatch) -> Result<f32> {
        let feedback_count = batch.feedback_texts.iter().filter(|f| f.is_some()).count();
        Ok(feedback_count as f32 / batch.feedback_texts.len() as f32)
    }

    fn update_statistics(&mut self) -> Result<()> {
        self.feedback_statistics.total_feedback_count =
            self.human_feedback_buffer.len() + self.ai_feedback_buffer.len();
        self.feedback_statistics.human_feedback_count = self.human_feedback_buffer.len();
        self.feedback_statistics.ai_feedback_count = self.ai_feedback_buffer.len();

        // Compute average rating from human feedback
        if !self.human_feedback_buffer.is_empty() {
            let total_rating: f32 = self.human_feedback_buffer.iter().map(|f| f.rating).sum();
            self.feedback_statistics.average_rating =
                total_rating / self.human_feedback_buffer.len() as f32;

            // Compute rating variance
            let mean = self.feedback_statistics.average_rating;
            let variance: f32 = self
                .human_feedback_buffer
                .iter()
                .map(|f| (f.rating - mean).powi(2))
                .sum::<f32>()
                / self.human_feedback_buffer.len() as f32;
            self.feedback_statistics.rating_variance = variance;
        }

        Ok(())
    }

    pub fn get_statistics(&self) -> &FeedbackStatistics {
        &self.feedback_statistics
    }

    pub fn clear_buffers(&mut self) {
        self.human_feedback_buffer.clear();
        self.ai_feedback_buffer.clear();
    }
}

#[derive(Debug)]
pub struct ProcessedFeedback {
    pub aggregated_ratings: Tensor,
    pub quality_scores: Tensor,
    pub filtered_indices: Vec<usize>,
    pub feedback_weights: Tensor,
    pub batch_statistics: BatchStatistics,
}

#[derive(Debug)]
pub struct BatchStatistics {
    pub mean_rating: f32,
    pub std_rating: f32,
    pub avg_response_length: f32,
    pub feedback_coverage: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_config_default() {
        let config = FeedbackConfig::default();
        assert_eq!(config.max_feedback_length, 1024);
        assert_eq!(config.feedback_temperature, 1.0);
        assert!(matches!(
            config.feedback_aggregation,
            FeedbackAggregation::WeightedMean
        ));
    }

    #[test]
    fn test_feedback_processor_creation() {
        let config = FeedbackConfig::default();
        let processor = FeedbackProcessor::new(config);
        assert_eq!(processor.get_statistics().total_feedback_count, 0);
    }

    #[test]
    fn test_human_feedback_addition() -> Result<()> {
        let mut processor = FeedbackProcessor::new(FeedbackConfig::default());

        let feedback = HumanFeedback {
            id: "test_1".to_string(),
            prompt: "Test prompt".to_string(),
            response: "Test response".to_string(),
            rating: 0.8,
            feedback_text: Some("Good response".to_string()),
            annotator_id: "annotator_1".to_string(),
            timestamp: 1234567890,
            metadata: HashMap::new(),
        };

        processor.add_human_feedback(feedback)?;
        assert_eq!(processor.get_statistics().human_feedback_count, 1);
        assert_eq!(processor.get_statistics().average_rating, 0.8);

        Ok(())
    }

    #[test]
    fn test_invalid_rating() {
        let mut processor = FeedbackProcessor::new(FeedbackConfig::default());

        let feedback = HumanFeedback {
            id: "test_1".to_string(),
            prompt: "Test prompt".to_string(),
            response: "Test response".to_string(),
            rating: 1.5, // Invalid rating
            feedback_text: None,
            annotator_id: "annotator_1".to_string(),
            timestamp: 1234567890,
            metadata: HashMap::new(),
        };

        assert!(processor.add_human_feedback(feedback).is_err());
    }

    #[test]
    fn test_feedback_batch_processing() -> Result<()> {
        let processor = FeedbackProcessor::new(FeedbackConfig::default());

        let batch = FeedbackBatch {
            prompts: vec!["Prompt 1".to_string(), "Prompt 2".to_string()],
            responses: vec!["Response 1".to_string(), "Response 2".to_string()],
            ratings: Tensor::from_vec(vec![0.8, 0.9], &[2])?,
            feedback_texts: vec![Some("Good".to_string()), None],
            weights: None,
        };

        let processed = processor.process_feedback_batch(&batch)?;
        assert_eq!(processed.aggregated_ratings.shape(), &[2]);
        assert_eq!(processed.quality_scores.shape(), &[2]);

        Ok(())
    }
}
