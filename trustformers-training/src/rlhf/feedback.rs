use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Number of equal-width bins [`FeedbackAggregation::MajorityVote`] discretises `[0, 1]` into.
///
/// Five matches the Likert scale that human raters are normally given, and is the same scale
/// [`crate::rlhf::trainer`] normalises its ratings against.
pub const MAJORITY_VOTE_BINS: usize = 5;

/// Arithmetic mean of a non-empty slice; `0.0` for an empty one.
fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// Population standard deviation (divisor `n`), used to define the consensus band.
fn population_std(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

/// Median of a slice: the central order statistic, averaging the two central values when the
/// count is even. Non-finite entries sort last so they cannot corrupt the ordering silently.
fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// `Σ wᵢ rᵢ / Σ wᵢ` over one item's annotator group.
///
/// `item` is only used to make the error message locate the offending row.
fn weighted_mean(values: &[f32], weights: &[f32], item: usize) -> Result<f32> {
    if values.len() != weights.len() {
        return Err(anyhow!(
            "item {item}: {} ratings but {} weights",
            values.len(),
            weights.len()
        ));
    }
    if let Some(bad) = weights.iter().find(|w| **w < 0.0 || !w.is_finite()) {
        return Err(anyhow!(
            "item {item}: feedback weight {bad} is negative or not finite; a weighted mean \
             is only defined for non-negative finite weights"
        ));
    }
    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        return Err(anyhow!(
            "item {item}: feedback weights sum to {total}; the weighted mean is undefined"
        ));
    }
    Ok(values.iter().zip(weights).map(|(&v, &w)| v * w).sum::<f32>() / total)
}

/// One-sigma trimmed mean — the "raters that agree" for a continuous rating scale.
///
/// Annotators more than one population standard deviation from the group mean are treated as
/// outliers and dropped; the mean of the survivors is the consensus. A group whose ratings all
/// coincide has zero spread and is returned unchanged.
fn consensus(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let centre = mean(values);
    let spread = population_std(values);
    if spread <= f32::EPSILON {
        return centre;
    }
    let kept: Vec<f32> = values.iter().copied().filter(|v| (v - centre).abs() <= spread).collect();
    if kept.is_empty() {
        // Unreachable for a finite group (the value nearest the mean always survives), but a
        // fallback keeps the function total rather than dividing by zero.
        return centre;
    }
    mean(&kept)
}

/// Modal-bin vote over [`MAJORITY_VOTE_BINS`] equal bins of `[0, 1]`.
///
/// The winning bin is the one holding the most ratings; ties go to the bin containing the
/// group's median, and failing that to the lower bin index. The reported value is the mean of
/// the ratings that fell in the winning bin, so the result stays on the rating scale instead of
/// snapping to a bin centre.
fn majority_vote(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let bin_of = |value: f32| -> usize {
        let scaled = (value.clamp(0.0, 1.0) * MAJORITY_VOTE_BINS as f32) as usize;
        scaled.min(MAJORITY_VOTE_BINS - 1)
    };

    let mut counts = [0usize; MAJORITY_VOTE_BINS];
    for &value in values {
        counts[bin_of(value)] += 1;
    }

    let median_bin = bin_of(median(values));
    let best_count = counts.iter().copied().max().unwrap_or(0);
    let winner = if counts[median_bin] == best_count {
        median_bin
    } else {
        counts.iter().position(|&c| c == best_count).unwrap_or(median_bin)
    };

    let members: Vec<f32> = values.iter().copied().filter(|&v| bin_of(v) == winner).collect();
    mean(&members)
}

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
        let aggregated_ratings =
            self.aggregate_ratings(&batch.ratings, batch_size, batch.weights.as_ref())?;

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

    /// Collapse each item's annotator ratings into a single rating per item.
    ///
    /// # Layout contract
    ///
    /// `ratings` is read as a flat, **item-major** buffer of `batch_size * annotators`
    /// values: entries `[i * annotators, (i + 1) * annotators)` are the ratings that the
    /// annotators gave item `i`. The common single-annotator case (`annotators == 1`) is
    /// exactly the `[batch_size]` tensor callers already pass. The returned tensor always
    /// has shape `[batch_size]`.
    ///
    /// # Aggregators
    ///
    /// Every arm below computes a genuinely different statistic of the group — this used to
    /// be five arms all returning `ratings.clone()`:
    ///
    /// * [`FeedbackAggregation::Mean`] — arithmetic mean.
    /// * [`FeedbackAggregation::Median`] — order statistic; the mean of the two central
    ///   values for an even number of annotators.
    /// * [`FeedbackAggregation::WeightedMean`] — `Σ wᵢ rᵢ / Σ wᵢ`. Per-annotator weights are
    ///   used when `weights.len() == ratings.len()`. When `weights.len() == batch_size` the
    ///   item's single weight is shared by all of its annotators and cancels out, so the
    ///   result *is* the arithmetic mean — that is the exact value of the weighted mean, not
    ///   a substitution. Absent weights mean uniform weights, likewise exactly the mean.
    /// * [`FeedbackAggregation::Consensus`] — one-sigma trimmed mean: annotators further than
    ///   one population standard deviation from the group mean are dropped as outliers and
    ///   the remainder is averaged, which is what "the raters that agree" means for a
    ///   continuous scale.
    /// * [`FeedbackAggregation::MajorityVote`] — the ratings are binned into
    ///   [`MAJORITY_VOTE_BINS`] equal Likert-style bins over `[0, 1]`, the modal bin wins
    ///   (ties broken towards the bin containing the median, then towards the lower bin) and
    ///   the mean of that bin's ratings is returned.
    ///
    /// # Errors
    ///
    /// * `batch_size` is zero, or `ratings` is empty.
    /// * `ratings.len()` is not a whole multiple of `batch_size` (ragged annotator groups
    ///   cannot be sliced unambiguously).
    /// * a supplied weight vector has a length other than `ratings.len()` or `batch_size`.
    /// * the weights of some item sum to zero or are negative, which leaves the weighted
    ///   mean undefined.
    fn aggregate_ratings(
        &self,
        ratings: &Tensor,
        batch_size: usize,
        weights: Option<&Tensor>,
    ) -> Result<Tensor> {
        if batch_size == 0 {
            return Err(anyhow!("cannot aggregate ratings for an empty batch"));
        }
        let values = ratings.to_vec_f32()?;
        if values.is_empty() {
            return Err(anyhow!(
                "cannot aggregate an empty ratings tensor for a batch of {batch_size} items"
            ));
        }
        if !values.len().is_multiple_of(batch_size) {
            return Err(anyhow!(
                "ratings tensor of {} values does not split evenly across {batch_size} items; \
                 every item must carry the same number of annotator ratings",
                values.len()
            ));
        }
        let annotators = values.len() / batch_size;

        // Per-annotator weights, expanded to `values.len()` so the same slicing works for the
        // per-item and per-annotator layouts alike.
        let per_annotator_weights: Option<Vec<f32>> = match weights {
            None => None,
            Some(tensor) => {
                let raw = tensor.to_vec_f32()?;
                if raw.len() == values.len() {
                    Some(raw)
                } else if raw.len() == batch_size {
                    // One weight per item: constant within each group, so it cancels in the
                    // ratio. Expanded rather than special-cased so the arithmetic is uniform.
                    Some(raw.iter().flat_map(|&w| std::iter::repeat_n(w, annotators)).collect())
                } else {
                    return Err(anyhow!(
                        "feedback weights of length {} match neither the ratings length {} \
                         nor the batch size {batch_size}",
                        raw.len(),
                        values.len()
                    ));
                }
            },
        };

        let mut aggregated = Vec::with_capacity(batch_size);
        for item in 0..batch_size {
            let start = item * annotators;
            let group = &values[start..start + annotators];
            let value = match self.config.feedback_aggregation {
                FeedbackAggregation::Mean => mean(group),
                FeedbackAggregation::Median => median(group),
                FeedbackAggregation::WeightedMean => match &per_annotator_weights {
                    None => mean(group),
                    Some(all) => weighted_mean(group, &all[start..start + annotators], item)?,
                },
                FeedbackAggregation::Consensus => consensus(group),
                FeedbackAggregation::MajorityVote => majority_vote(group),
            };
            aggregated.push(value);
        }

        Ok(Tensor::from_vec(aggregated, &[batch_size])?)
    }

    fn compute_quality_scores(&self, batch: &FeedbackBatch) -> Result<Tensor> {
        let batch_size = batch.prompts.len();
        let mut quality_scores = Vec::with_capacity(batch_size);

        // Derive a per-item base rating from the ratings tensor instead of a constant.
        // `ratings` is normally shape `[batch_size]` (one rating per item); if it carries
        // several values per item (e.g. multiple annotators) average each item's group,
        // and fall back to the global mean when the layout does not divide evenly.
        let ratings = batch.ratings.to_vec_f32()?;
        let global_mean = if ratings.is_empty() {
            0.0
        } else {
            ratings.iter().sum::<f32>() / ratings.len() as f32
        };
        let per_item = |index: usize| -> f32 {
            if batch_size == 0 {
                global_mean
            } else if ratings.len() == batch_size {
                ratings[index]
            } else if ratings.len() % batch_size == 0 {
                let group = ratings.len() / batch_size;
                let start = index * group;
                ratings[start..start + group].iter().sum::<f32>() / group as f32
            } else {
                global_mean
            }
        };

        for i in 0..batch_size {
            // Base quality score from this item's own rating.
            let mut quality = per_item(i);

            // Apply diversity bonus.
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
        // Keep only the items whose quality score meets the configured threshold.
        let scores = quality_scores.to_vec_f32()?;
        let threshold = self.config.quality_threshold;
        let filtered = scores
            .iter()
            .enumerate()
            .filter_map(
                |(index, &score)| {
                    if score >= threshold {
                        Some(index)
                    } else {
                        None
                    }
                },
            )
            .collect();
        Ok(filtered)
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

        // Real rating statistics over the batch (was a hardcoded 0.5 / 0.1 placeholder).
        let ratings = batch.ratings.to_vec_f32()?;
        let mean_rating = if ratings.is_empty() {
            0.0
        } else {
            ratings.iter().sum::<f32>() / ratings.len() as f32
        };
        let std_rating = self.compute_rating_std_single(&batch.ratings)?;

        Ok(BatchStatistics {
            mean_rating,
            std_rating,
            avg_response_length,
            feedback_coverage: self.compute_feedback_coverage(batch)?,
        })
    }

    #[allow(dead_code)]
    fn compute_rating_std(&self, ratings: &Tensor) -> Result<Tensor> {
        // Sample standard deviation reduced to a single scalar tensor of shape [1].
        let std = self.compute_rating_std_single(ratings)?;
        Ok(Tensor::from_vec(vec![std], &[1])?)
    }

    fn compute_rating_std_single(&self, ratings: &Tensor) -> Result<f32> {
        // Real sample standard deviation: mean = Sum(x)/n,
        // variance = Sum((x - mean)^2) / (n - 1), std = sqrt(variance).
        let values = ratings.to_vec_f32()?;
        let n = values.len();
        if n <= 1 {
            // Guard against division by zero; a single (or empty) sample has no spread.
            return Ok(0.0);
        }
        let mean = values.iter().sum::<f32>() / n as f32;
        let variance = values
            .iter()
            .map(|&value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f32>()
            / (n as f32 - 1.0);
        Ok(variance.sqrt())
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

    #[test]
    fn test_quality_scores_reflect_real_ratings() -> Result<()> {
        // Quality must track each item's own rating, not a constant 0.5 placeholder.
        let cfg = FeedbackConfig {
            diversity_weight: 0.0, // isolate the rating contribution
            ..FeedbackConfig::default()
        };
        let processor = FeedbackProcessor::new(cfg);

        let batch = FeedbackBatch {
            prompts: vec!["p1".to_string(), "p2".to_string()],
            responses: vec!["a b c".to_string(), "d e f".to_string()],
            ratings: Tensor::from_vec(vec![0.2, 0.9], &[2])?,
            feedback_texts: vec![None, None],
            weights: None,
        };

        let scores = processor.compute_quality_scores(&batch)?.to_vec_f32()?;
        assert_eq!(scores.len(), 2);
        // With diversity disabled, quality == clamped per-item rating.
        assert!(
            (scores[0] - 0.2).abs() < 1e-5,
            "item 0 should reflect its 0.2 rating, got {}",
            scores[0]
        );
        assert!(
            (scores[1] - 0.9).abs() < 1e-5,
            "item 1 should reflect its 0.9 rating, got {}",
            scores[1]
        );
        // Low- and high-rated items must differ (they would be equal under the 0.5 placeholder).
        assert!(scores[1] > scores[0]);
        Ok(())
    }

    #[test]
    fn test_batch_statistics_are_real_not_placeholder() -> Result<()> {
        let processor = FeedbackProcessor::new(FeedbackConfig::default());
        let batch = FeedbackBatch {
            prompts: vec!["p1".to_string(), "p2".to_string(), "p3".to_string()],
            responses: vec!["x".to_string(), "y".to_string(), "z".to_string()],
            ratings: Tensor::from_vec(vec![0.1, 0.2, 0.9], &[3])?,
            feedback_texts: vec![Some("ok".to_string()), None, None],
            weights: None,
        };

        let stats = processor.compute_batch_statistics(&batch)?;
        // mean of {0.1, 0.2, 0.9} = 0.4 — computed, not the 0.5 placeholder.
        assert!(
            (stats.mean_rating - 0.4).abs() < 1e-5,
            "mean should be the real 0.4, got {}",
            stats.mean_rating
        );
        // sample std of {0.1, 0.2, 0.9}: variance = 0.38/2 = 0.19, std = sqrt(0.19) ≈ 0.4359,
        // clearly not the 0.1 placeholder.
        assert!(
            (stats.std_rating - 0.19f32.sqrt()).abs() < 1e-4,
            "std should be the real sqrt(0.19), got {}",
            stats.std_rating
        );
        assert!((stats.feedback_coverage - (1.0 / 3.0)).abs() < 1e-6);
        Ok(())
    }

    // ── Additional tests ──────────────────────────────────────────────────────

    #[test]
    fn test_feedback_config_default_values() {
        let cfg = FeedbackConfig::default();
        assert!(cfg.max_feedback_length > 0);
        assert!(cfg.feedback_temperature > 0.0);
        assert!(cfg.quality_threshold >= 0.0 && cfg.quality_threshold <= 1.0);
        assert!(cfg.consistency_weight >= 0.0);
        assert!(cfg.diversity_weight >= 0.0);
    }

    #[test]
    fn test_feedback_config_use_human_feedback_default() {
        let cfg = FeedbackConfig::default();
        assert!(cfg.use_human_feedback, "default should use human feedback");
    }

    #[test]
    fn test_feedback_statistics_default_zeros() {
        let stats = FeedbackStatistics::default();
        assert_eq!(stats.total_feedback_count, 0);
        assert_eq!(stats.human_feedback_count, 0);
        assert_eq!(stats.ai_feedback_count, 0);
        assert!((stats.average_rating).abs() < 1e-6);
    }

    #[test]
    fn test_feedback_processor_initial_stats() {
        let processor = FeedbackProcessor::new(FeedbackConfig::default());
        let stats = processor.get_statistics();
        assert_eq!(stats.total_feedback_count, 0);
    }

    #[test]
    fn test_human_feedback_valid_rating_accepted() {
        let mut processor = FeedbackProcessor::new(FeedbackConfig::default());
        let feedback = HumanFeedback {
            id: "valid_001".to_string(),
            prompt: "Prompt".to_string(),
            response: "Response".to_string(),
            rating: 0.8,
            feedback_text: Some("Good answer".to_string()),
            annotator_id: "ann_1".to_string(),
            timestamp: 1_000_000,
            metadata: HashMap::new(),
        };
        assert!(processor.add_human_feedback(feedback).is_ok());
    }

    #[test]
    fn test_human_feedback_zero_rating_valid() {
        let mut processor = FeedbackProcessor::new(FeedbackConfig::default());
        let feedback = HumanFeedback {
            id: "zero_001".to_string(),
            prompt: "What?".to_string(),
            response: "Nothing".to_string(),
            rating: 0.0,
            feedback_text: None,
            annotator_id: "ann_2".to_string(),
            timestamp: 2_000_000,
            metadata: HashMap::new(),
        };
        assert!(processor.add_human_feedback(feedback).is_ok());
    }

    #[test]
    fn test_human_feedback_max_rating_valid() {
        let mut processor = FeedbackProcessor::new(FeedbackConfig::default());
        let feedback = HumanFeedback {
            id: "max_001".to_string(),
            prompt: "Perfect?".to_string(),
            response: "Yes".to_string(),
            rating: 1.0,
            feedback_text: None,
            annotator_id: "ann_3".to_string(),
            timestamp: 3_000_000,
            metadata: HashMap::new(),
        };
        assert!(processor.add_human_feedback(feedback).is_ok());
    }

    #[test]
    fn test_ai_feedback_struct_creation() {
        let ai_fb = AIFeedback {
            id: "ai_001".to_string(),
            prompt: "Question?".to_string(),
            response: "Answer.".to_string(),
            helpfulness_score: 0.9,
            harmlessness_score: 0.95,
            honesty_score: 0.85,
            overall_score: 0.9,
            explanation: "Very helpful response".to_string(),
            confidence: 0.8,
            model_version: "gpt-4".to_string(),
        };
        assert!(ai_fb.overall_score >= 0.0 && ai_fb.overall_score <= 1.0);
        assert!(ai_fb.confidence >= 0.0 && ai_fb.confidence <= 1.0);
    }

    #[test]
    fn test_ai_feedback_accepted() {
        let mut processor = FeedbackProcessor::new(FeedbackConfig::default());
        let ai_fb = AIFeedback {
            id: "ai_002".to_string(),
            prompt: "Prompt".to_string(),
            response: "Response".to_string(),
            helpfulness_score: 0.8,
            harmlessness_score: 0.9,
            honesty_score: 0.7,
            overall_score: 0.8,
            explanation: "OK".to_string(),
            confidence: 0.75,
            model_version: "v1".to_string(),
        };
        assert!(processor.add_ai_feedback(ai_fb).is_ok());
    }

    #[test]
    fn test_feedback_aggregation_variants() {
        // Verify all FeedbackAggregation variants are distinct
        let mean = FeedbackAggregation::Mean;
        let median = FeedbackAggregation::Median;
        let weighted = FeedbackAggregation::WeightedMean;
        let consensus = FeedbackAggregation::Consensus;
        let majority = FeedbackAggregation::MajorityVote;
        // Just ensure they can be created without issues
        let _ = (mean, median, weighted, consensus, majority);
    }

    #[test]
    fn test_feedback_processor_clear_buffers() {
        let mut processor = FeedbackProcessor::new(FeedbackConfig::default());
        let feedback = HumanFeedback {
            id: "clear_001".to_string(),
            prompt: "Prompt".to_string(),
            response: "Response".to_string(),
            rating: 0.5,
            feedback_text: None,
            annotator_id: "ann_1".to_string(),
            timestamp: 1_000_000,
            metadata: HashMap::new(),
        };
        processor.add_human_feedback(feedback).unwrap_or(());
        // Verify feedback was added before clearing
        assert_eq!(processor.get_statistics().human_feedback_count, 1);
        processor.clear_buffers();
        // clear_buffers clears internal buffers; a subsequent update_statistics call
        // would reflect 0 — here we verify the method does not panic
        // and the processor remains usable after clearing
        let stats_after = processor.get_statistics();
        let _ = stats_after; // verify we can still call get_statistics without panic
    }

    #[test]
    fn test_feedback_batch_with_weights() -> Result<()> {
        let processor = FeedbackProcessor::new(FeedbackConfig::default());
        let batch = FeedbackBatch {
            prompts: vec!["P1".to_string()],
            responses: vec!["R1".to_string()],
            ratings: Tensor::from_vec(vec![0.7], &[1])?,
            feedback_texts: vec![None],
            weights: Some(Tensor::from_vec(vec![0.5], &[1])?),
        };
        let processed = processor.process_feedback_batch(&batch);
        assert!(processed.is_ok());
        Ok(())
    }

    #[test]
    fn test_feedback_multiple_add_increments_stats() {
        let mut processor = FeedbackProcessor::new(FeedbackConfig::default());
        for i in 0..5u32 {
            let feedback = HumanFeedback {
                id: format!("fb_{}", i),
                prompt: format!("Prompt {}", i),
                response: format!("Response {}", i),
                rating: 0.5 + (i as f32) * 0.08,
                feedback_text: None,
                annotator_id: format!("ann_{}", i),
                timestamp: i as u64 * 1000,
                metadata: HashMap::new(),
            };
            processor.add_human_feedback(feedback).unwrap_or(());
        }
        let stats = processor.get_statistics();
        assert_eq!(stats.human_feedback_count, 5);
    }

    // ── aggregate_ratings: real statistics, not five copies of the input ──────

    fn processor_with(aggregation: FeedbackAggregation) -> FeedbackProcessor {
        FeedbackProcessor::new(FeedbackConfig {
            feedback_aggregation: aggregation,
            ..FeedbackConfig::default()
        })
    }

    /// Two items, three annotators each: `[0.1, 0.2, 0.9]` and `[0.4, 0.5, 0.6]`.
    fn three_annotator_ratings() -> Result<Tensor> {
        Ok(Tensor::from_vec(vec![0.1, 0.2, 0.9, 0.4, 0.5, 0.6], &[6])?)
    }

    #[test]
    fn test_aggregate_ratings_collapses_annotator_groups() -> Result<()> {
        // Regression: every arm returned `ratings.clone()`, so the output kept the raw
        // annotator layout (shape [6]) instead of one rating per item (shape [2]).
        let processor = processor_with(FeedbackAggregation::Mean);
        let aggregated = processor.aggregate_ratings(&three_annotator_ratings()?, 2, None)?;
        assert_eq!(
            aggregated.shape(),
            &[2],
            "aggregation must yield one rating per item"
        );
        Ok(())
    }

    #[test]
    fn test_aggregate_ratings_mean_is_hand_computed() -> Result<()> {
        let processor = processor_with(FeedbackAggregation::Mean);
        let got = processor
            .aggregate_ratings(&three_annotator_ratings()?, 2, None)?
            .to_vec_f32()?;
        // (0.1 + 0.2 + 0.9) / 3 = 0.4 ; (0.4 + 0.5 + 0.6) / 3 = 0.5
        assert!((got[0] - 0.4).abs() < 1e-5, "mean item0 = {}", got[0]);
        assert!((got[1] - 0.5).abs() < 1e-5, "mean item1 = {}", got[1]);
        Ok(())
    }

    #[test]
    fn test_aggregate_ratings_median_is_the_order_statistic() -> Result<()> {
        let processor = processor_with(FeedbackAggregation::Median);
        let got = processor
            .aggregate_ratings(&three_annotator_ratings()?, 2, None)?
            .to_vec_f32()?;
        // median{0.1, 0.2, 0.9} = 0.2 — and crucially NOT the mean 0.4.
        assert!((got[0] - 0.2).abs() < 1e-5, "median item0 = {}", got[0]);
        assert!((got[1] - 0.5).abs() < 1e-5, "median item1 = {}", got[1]);
        Ok(())
    }

    #[test]
    fn test_aggregate_ratings_median_averages_two_central_values() -> Result<()> {
        let processor = processor_with(FeedbackAggregation::Median);
        // One item, four annotators: sorted {0.1, 0.2, 0.6, 0.8} → (0.2 + 0.6) / 2 = 0.4
        let ratings = Tensor::from_vec(vec![0.8, 0.1, 0.6, 0.2], &[4])?;
        let got = processor.aggregate_ratings(&ratings, 1, None)?.to_vec_f32()?;
        assert!(
            (got[0] - 0.4).abs() < 1e-5,
            "even-count median = {}",
            got[0]
        );
        Ok(())
    }

    #[test]
    fn test_aggregate_ratings_consensus_drops_the_outlier() -> Result<()> {
        let processor = processor_with(FeedbackAggregation::Consensus);
        let got = processor
            .aggregate_ratings(&three_annotator_ratings()?, 2, None)?
            .to_vec_f32()?;
        // item0: mean 0.4, population std = sqrt(0.38 / 3) ≈ 0.3559. The 0.9 rater is 0.5 away
        // and is trimmed; consensus = mean{0.1, 0.2} = 0.15.
        assert!((got[0] - 0.15).abs() < 1e-4, "consensus item0 = {}", got[0]);
        // item1: std ≈ 0.0816, so only the 0.5 rater survives.
        assert!((got[1] - 0.5).abs() < 1e-4, "consensus item1 = {}", got[1]);
        Ok(())
    }

    #[test]
    fn test_aggregate_ratings_majority_vote_picks_the_modal_bin() -> Result<()> {
        let processor = processor_with(FeedbackAggregation::MajorityVote);
        // item0 {0.05, 0.15, 0.95}: bins 0, 0, 4 → modal bin 0 → mean{0.05, 0.15} = 0.10
        // item1 {0.45, 0.55, 0.65}: bins 2, 2, 3 → modal bin 2 → mean{0.45, 0.55} = 0.50
        let ratings = Tensor::from_vec(vec![0.05, 0.15, 0.95, 0.45, 0.55, 0.65], &[6])?;
        let got = processor.aggregate_ratings(&ratings, 2, None)?.to_vec_f32()?;
        assert!((got[0] - 0.10).abs() < 1e-4, "majority item0 = {}", got[0]);
        assert!((got[1] - 0.50).abs() < 1e-4, "majority item1 = {}", got[1]);
        Ok(())
    }

    #[test]
    fn test_aggregate_ratings_weighted_mean_uses_per_annotator_weights() -> Result<()> {
        let processor = processor_with(FeedbackAggregation::WeightedMean);
        // Two items, two annotators: ratings {0.0, 1.0} both times, mirrored weights.
        let ratings = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4])?;
        let weights = Tensor::from_vec(vec![3.0, 1.0, 1.0, 3.0], &[4])?;
        let got = processor.aggregate_ratings(&ratings, 2, Some(&weights))?.to_vec_f32()?;
        // (0*3 + 1*1) / 4 = 0.25 and (0*1 + 1*3) / 4 = 0.75 — both differ from the mean 0.5.
        assert!((got[0] - 0.25).abs() < 1e-5, "weighted item0 = {}", got[0]);
        assert!((got[1] - 0.75).abs() < 1e-5, "weighted item1 = {}", got[1]);
        Ok(())
    }

    #[test]
    fn test_aggregate_ratings_per_item_weights_reduce_to_the_mean() -> Result<()> {
        let processor = processor_with(FeedbackAggregation::WeightedMean);
        let ratings = Tensor::from_vec(vec![0.0, 1.0, 0.2, 0.4], &[4])?;
        // One weight per item: constant inside each group, so it cancels exactly.
        let weights = Tensor::from_vec(vec![7.0, 0.5], &[2])?;
        let got = processor.aggregate_ratings(&ratings, 2, Some(&weights))?.to_vec_f32()?;
        assert!((got[0] - 0.5).abs() < 1e-5, "item0 = {}", got[0]);
        assert!((got[1] - 0.3).abs() < 1e-5, "item1 = {}", got[1]);
        Ok(())
    }

    #[test]
    fn test_aggregation_modes_disagree_on_a_skewed_group() -> Result<()> {
        // The core regression: the five arms were byte-identical, so this could not fail.
        let ratings = three_annotator_ratings()?;
        let mean_out = processor_with(FeedbackAggregation::Mean)
            .aggregate_ratings(&ratings, 2, None)?
            .to_vec_f32()?;
        let median_out = processor_with(FeedbackAggregation::Median)
            .aggregate_ratings(&ratings, 2, None)?
            .to_vec_f32()?;
        let consensus_out = processor_with(FeedbackAggregation::Consensus)
            .aggregate_ratings(&ratings, 2, None)?
            .to_vec_f32()?;
        assert!(
            (mean_out[0] - median_out[0]).abs() > 1e-3,
            "mean {} and median {} must differ on a skewed group",
            mean_out[0],
            median_out[0]
        );
        assert!(
            (mean_out[0] - consensus_out[0]).abs() > 1e-3,
            "mean {} and consensus {} must differ on a skewed group",
            mean_out[0],
            consensus_out[0]
        );
        Ok(())
    }

    #[test]
    fn test_aggregate_ratings_rejects_ragged_and_degenerate_input() -> Result<()> {
        let processor = processor_with(FeedbackAggregation::Mean);
        // 5 ratings cannot split evenly across 2 items.
        let ragged = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5], &[5])?;
        assert!(
            processor.aggregate_ratings(&ragged, 2, None).is_err(),
            "ragged annotator groups must be an error, not a silent reshape"
        );
        // Weights that match neither layout.
        let ratings = Tensor::from_vec(vec![0.1, 0.2], &[2])?;
        let bad_weights = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3])?;
        assert!(processor.aggregate_ratings(&ratings, 2, Some(&bad_weights)).is_err());
        Ok(())
    }

    #[test]
    fn test_weighted_mean_rejects_zero_and_negative_weights() -> Result<()> {
        let processor = processor_with(FeedbackAggregation::WeightedMean);
        let ratings = Tensor::from_vec(vec![0.1, 0.2], &[2])?;
        let zeros = Tensor::from_vec(vec![0.0, 0.0], &[2])?;
        assert!(
            processor.aggregate_ratings(&ratings, 1, Some(&zeros)).is_err(),
            "an all-zero weight vector leaves the weighted mean undefined"
        );
        let negative = Tensor::from_vec(vec![-1.0, 2.0], &[2])?;
        assert!(processor.aggregate_ratings(&ratings, 1, Some(&negative)).is_err());
        Ok(())
    }
}
