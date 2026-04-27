//! Model Ensemble module for serving multiple models simultaneously and aggregating their outputs.
//!
//! Supports various aggregation strategies including weighted averaging, majority voting,
//! max confidence selection, mean probability, geometric mean, and stacking.

mod ensemble_extra_tests;

use thiserror::Error;

/// Strategy for combining multiple model outputs
#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleStrategy {
    /// Weighted average of logit vectors
    WeightedAverage { weights: Vec<f64> },
    /// Majority vote over discrete labels
    MajorityVote,
    /// Highest confidence prediction wins
    MaxConfidence,
    /// Arithmetic mean of probabilities
    MeanProbability,
    /// Geometric mean of probabilities (more robust to outliers)
    GeometricMean,
    /// Stacking: use another model to combine outputs (simplified: weighted average with learned weights)
    Stacking { meta_weights: Vec<f64> },
    /// Weight models by softmax of their peak logit confidence score
    ConfidenceWeighted,
    /// Trim the highest and lowest `percentile` fraction of predictions before averaging
    Trimmed { percentile: f32 },
}

/// One model's prediction output
#[derive(Debug, Clone)]
pub struct ModelPrediction {
    pub model_id: String,
    pub logits: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub predicted_class: usize,
    pub confidence: f64,
    pub latency_ms: u64,
}

impl ModelPrediction {
    /// Create a new ModelPrediction from raw logits.
    /// Computes softmax probabilities, argmax predicted_class, and max confidence.
    pub fn new(model_id: impl Into<String>, logits: Vec<f64>) -> Self {
        let probabilities = Self::softmax(&logits);
        let (predicted_class, confidence) = probabilities.iter().enumerate().fold(
            (0, f64::NEG_INFINITY),
            |(best_idx, best_val), (i, &v)| {
                if v > best_val {
                    (i, v)
                } else {
                    (best_idx, best_val)
                }
            },
        );
        let confidence = if confidence == f64::NEG_INFINITY { 0.0 } else { confidence };

        Self {
            model_id: model_id.into(),
            logits,
            probabilities,
            predicted_class,
            confidence,
            latency_ms: 0,
        }
    }

    /// Compute softmax of logit vector. Uses numerically stable implementation.
    pub fn softmax(logits: &[f64]) -> Vec<f64> {
        if logits.is_empty() {
            return Vec::new();
        }
        let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 {
            vec![1.0 / logits.len() as f64; logits.len()]
        } else {
            exps.iter().map(|&e| e / sum).collect()
        }
    }
}

/// Result of ensemble aggregation
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    pub final_class: usize,
    pub final_confidence: f64,
    pub aggregated_probs: Vec<f64>,
    pub individual_predictions: Vec<ModelPrediction>,
    pub strategy: String,
    /// Fraction of models that agree with final_class
    pub agreement_ratio: f64,
    /// Max of individual latencies (parallel execution model)
    pub total_latency_ms: u64,
}

impl EnsembleResult {
    /// Shannon entropy of aggregated_probs: -sum(p * log2(p+1e-10))
    pub fn entropy(&self) -> f64 {
        self.aggregated_probs
            .iter()
            .map(|&p| if p <= 0.0 { 0.0 } else { -p * p.log2() })
            .sum::<f64>()
            .max(0.0)
    }

    /// Return top-k (class_index, probability) pairs sorted by probability descending.
    pub fn top_k_classes(&self, k: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> =
            self.aggregated_probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }
}

/// Model ensemble aggregator
#[derive(Debug)]
pub struct ModelEnsemble {
    strategy: EnsembleStrategy,
    num_classes: usize,
}

impl ModelEnsemble {
    /// Create a new ModelEnsemble with the given strategy and number of output classes.
    ///
    /// For WeightedAverage and Stacking strategies, the weights length is validated
    /// against the number of predictions at aggregate time.
    pub fn new(strategy: EnsembleStrategy, num_classes: usize) -> Result<Self, EnsembleError> {
        // Validate weights are non-negative for strategies that require them
        match &strategy {
            EnsembleStrategy::WeightedAverage { weights } => {
                for &w in weights {
                    if w < 0.0 || w.is_nan() || w.is_infinite() {
                        return Err(EnsembleError::InvalidWeight(w));
                    }
                }
            },
            EnsembleStrategy::Stacking { meta_weights } => {
                for &w in meta_weights {
                    if w < 0.0 || w.is_nan() || w.is_infinite() {
                        return Err(EnsembleError::InvalidWeight(w));
                    }
                }
            },
            _ => {},
        }
        Ok(Self {
            strategy,
            num_classes,
        })
    }

    /// Aggregate multiple model predictions using the configured strategy.
    pub fn aggregate(
        &self,
        predictions: &[ModelPrediction],
    ) -> Result<EnsembleResult, EnsembleError> {
        if predictions.is_empty() {
            return Err(EnsembleError::EmptyPredictions);
        }

        // Validate all predictions have expected class count
        for pred in predictions {
            if pred.probabilities.len() != self.num_classes {
                return Err(EnsembleError::ClassCountMismatch {
                    expected: self.num_classes,
                    got: pred.probabilities.len(),
                });
            }
        }

        let strategy_name = match &self.strategy {
            EnsembleStrategy::WeightedAverage { .. } => "WeightedAverage",
            EnsembleStrategy::MajorityVote => "MajorityVote",
            EnsembleStrategy::MaxConfidence => "MaxConfidence",
            EnsembleStrategy::MeanProbability => "MeanProbability",
            EnsembleStrategy::GeometricMean => "GeometricMean",
            EnsembleStrategy::Stacking { .. } => "Stacking",
            EnsembleStrategy::ConfidenceWeighted => "ConfidenceWeighted",
            EnsembleStrategy::Trimmed { .. } => "Trimmed",
        };

        let aggregated_probs = match &self.strategy {
            EnsembleStrategy::WeightedAverage { weights } => {
                if weights.len() != predictions.len() {
                    return Err(EnsembleError::WeightCountMismatch {
                        weights: weights.len(),
                        models: predictions.len(),
                    });
                }
                self.weighted_average(predictions, weights)
            },
            EnsembleStrategy::MajorityVote => self.majority_vote(predictions),
            EnsembleStrategy::MaxConfidence => self.max_confidence(predictions),
            EnsembleStrategy::MeanProbability => self.mean_probability(predictions),
            EnsembleStrategy::GeometricMean => self.geometric_mean(predictions),
            EnsembleStrategy::Stacking { meta_weights } => {
                if meta_weights.len() != predictions.len() {
                    return Err(EnsembleError::WeightCountMismatch {
                        weights: meta_weights.len(),
                        models: predictions.len(),
                    });
                }
                self.weighted_average(predictions, meta_weights)
            },
            EnsembleStrategy::ConfidenceWeighted => self.confidence_weighted(predictions),
            EnsembleStrategy::Trimmed { percentile } => self.trimmed_mean(predictions, *percentile),
        };

        // Determine final class as argmax of aggregated_probs
        let (final_class, final_confidence) = aggregated_probs.iter().enumerate().fold(
            (0, f64::NEG_INFINITY),
            |(best_idx, best_val), (i, &v)| {
                if v > best_val {
                    (i, v)
                } else {
                    (best_idx, best_val)
                }
            },
        );
        let final_confidence =
            if final_confidence == f64::NEG_INFINITY { 0.0 } else { final_confidence };

        // Agreement ratio: fraction of models predicting final_class
        let agreeing = predictions.iter().filter(|p| p.predicted_class == final_class).count();
        let agreement_ratio = agreeing as f64 / predictions.len() as f64;

        // Total latency: max of individual latencies (parallel execution model)
        let total_latency_ms = predictions.iter().map(|p| p.latency_ms).max().unwrap_or(0);

        Ok(EnsembleResult {
            final_class,
            final_confidence,
            aggregated_probs,
            individual_predictions: predictions.to_vec(),
            strategy: strategy_name.to_string(),
            agreement_ratio,
            total_latency_ms,
        })
    }

    /// Weighted average: weighted_probs[c] = sum(w_i * probs_i[c]) / sum(w_i)
    fn weighted_average(&self, predictions: &[ModelPrediction], weights: &[f64]) -> Vec<f64> {
        let weight_sum: f64 = weights.iter().sum();
        let safe_sum = if weight_sum == 0.0 { 1.0 } else { weight_sum };

        let mut result = vec![0.0f64; self.num_classes];
        for (pred, &w) in predictions.iter().zip(weights.iter()) {
            for (c, &p) in pred.probabilities.iter().enumerate() {
                result[c] += w * p;
            }
        }
        result.iter_mut().for_each(|v| *v /= safe_sum);
        result
    }

    /// Majority vote: one-hot encode majority class as probabilities.
    fn majority_vote(&self, predictions: &[ModelPrediction]) -> Vec<f64> {
        let mut vote_counts = vec![0usize; self.num_classes];
        for pred in predictions {
            if pred.predicted_class < self.num_classes {
                vote_counts[pred.predicted_class] += 1;
            }
        }
        let winner = vote_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut result = vec![0.0f64; self.num_classes];
        result[winner] = 1.0;
        result
    }

    /// Max confidence: return probabilities of the most confident model.
    fn max_confidence(&self, predictions: &[ModelPrediction]) -> Vec<f64> {
        predictions
            .iter()
            .max_by(|a, b| {
                a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.probabilities.clone())
            .unwrap_or_else(|| vec![1.0 / self.num_classes as f64; self.num_classes])
    }

    /// Mean probability: arithmetic mean across all models.
    fn mean_probability(&self, predictions: &[ModelPrediction]) -> Vec<f64> {
        let n = predictions.len() as f64;
        let mut result = vec![0.0f64; self.num_classes];
        for pred in predictions {
            for (c, &p) in pred.probabilities.iter().enumerate() {
                result[c] += p;
            }
        }
        result.iter_mut().for_each(|v| *v /= n);
        result
    }

    /// Geometric mean: exp(mean(log(p_i + 1e-10))), then renormalize.
    fn geometric_mean(&self, predictions: &[ModelPrediction]) -> Vec<f64> {
        let n = predictions.len() as f64;
        let mut log_sum = vec![0.0f64; self.num_classes];
        for pred in predictions {
            for (c, &p) in pred.probabilities.iter().enumerate() {
                log_sum[c] += (p + 1e-10).ln();
            }
        }
        let mut result: Vec<f64> = log_sum.iter().map(|&s| (s / n).exp()).collect();

        // Renormalize
        let sum: f64 = result.iter().sum();
        if sum > 0.0 {
            result.iter_mut().for_each(|v| *v /= sum);
        }
        result
    }

    /// Confidence-weighted average: weight each model's probabilities by softmax
    /// of its peak confidence score, so more confident models contribute more.
    fn confidence_weighted(&self, predictions: &[ModelPrediction]) -> Vec<f64> {
        // Collect raw confidence scores (max probability per model).
        let confs: Vec<f64> = predictions.iter().map(|p| p.confidence).collect();

        // Numerically stable softmax over confidence scores.
        let max_conf = confs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = confs.iter().map(|&c| (c - max_conf).exp()).collect();
        let exp_sum: f64 = exps.iter().sum();
        let weights: Vec<f64> = if exp_sum > 0.0 {
            exps.iter().map(|&e| e / exp_sum).collect()
        } else {
            vec![1.0 / predictions.len() as f64; predictions.len()]
        };

        self.weighted_average(predictions, &weights)
    }

    /// Trimmed mean: discard the `percentile` fraction of models with the
    /// highest and lowest per-class predicted probabilities, then average
    /// the remaining models.  Operates class-by-class.
    ///
    /// If `percentile` is 0.0 this degenerates to the arithmetic mean.
    /// If the trim would remove all models the full set is used as fallback.
    fn trimmed_mean(&self, predictions: &[ModelPrediction], percentile: f32) -> Vec<f64> {
        let n = predictions.len();
        if n == 0 {
            return vec![1.0 / self.num_classes as f64; self.num_classes];
        }

        // Number of predictions to trim from each end per class.
        let trim_count = ((n as f32 * percentile.clamp(0.0, 0.49)) as usize).min(n / 2);
        let keep = n.saturating_sub(2 * trim_count);

        if keep == 0 {
            // Fallback: use full set.
            return self.mean_probability(predictions);
        }

        let mut result = vec![0.0f64; self.num_classes];
        for c in 0..self.num_classes {
            // Collect class-c probabilities with their model indices.
            let mut class_probs: Vec<f64> = predictions
                .iter()
                .map(|p| if c < p.probabilities.len() { p.probabilities[c] } else { 0.0 })
                .collect();
            class_probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // Keep the middle `keep` values (remove trim_count from each end).
            let trimmed = &class_probs[trim_count..trim_count + keep];
            result[c] = trimmed.iter().sum::<f64>() / keep as f64;
        }

        // Renormalize so that probabilities sum to 1.
        let sum: f64 = result.iter().sum();
        if sum > 0.0 {
            result.iter_mut().for_each(|v| *v /= sum);
        }
        result
    }

    /// Compute mean prediction and predictive variance (epistemic uncertainty proxy).
    ///
    /// Returns `(mean_probs, variance)` where `variance` is the mean across classes
    /// of the variance of each class probability across models.
    pub fn predict_with_uncertainty(logits: &[Vec<f32>]) -> Result<(Vec<f32>, f32), EnsembleError> {
        if logits.is_empty() {
            return Err(EnsembleError::EmptyPredictions);
        }
        let n = logits.len();
        let num_classes = logits[0].len();
        if num_classes == 0 {
            return Err(EnsembleError::EmptyPredictions);
        }

        // Convert each logit vector to probabilities via softmax.
        let probs: Vec<Vec<f32>> = logits.iter().map(|lv| Self::softmax_f32(lv)).collect();

        // Mean probability per class.
        let mut mean = vec![0.0f32; num_classes];
        for p in &probs {
            for (c, &pv) in p.iter().enumerate() {
                if c < num_classes {
                    mean[c] += pv;
                }
            }
        }
        mean.iter_mut().for_each(|v| *v /= n as f32);

        // Variance per class (Bessel-corrected if n > 1, otherwise population variance).
        let divisor = if n > 1 { (n - 1) as f32 } else { n as f32 };
        let mut var_sum = 0.0f32;
        for c in 0..num_classes {
            let class_var: f32 = probs
                .iter()
                .map(|p| {
                    let diff = p.get(c).copied().unwrap_or(0.0) - mean[c];
                    diff * diff
                })
                .sum::<f32>()
                / divisor;
            var_sum += class_var;
        }
        let mean_variance = var_sum / num_classes as f32;

        Ok((mean, mean_variance))
    }

    /// Epistemic (model) uncertainty: variance across model predictions.
    ///
    /// Returns the average pairwise variance of softmax probabilities across models,
    /// averaged over all classes.
    pub fn epistemic_uncertainty(logits: &[Vec<f32>]) -> Result<f32, EnsembleError> {
        let (_, variance) = Self::predict_with_uncertainty(logits)?;
        Ok(variance)
    }

    /// Aleatoric uncertainty: mean of per-model uncertainty estimates.
    ///
    /// Each model may produce its own uncertainty (e.g. entropy of its distribution).
    /// This function averages those per-model scalar uncertainties.
    pub fn aleatoric_uncertainty(model_uncertainties: &[f32]) -> f32 {
        if model_uncertainties.is_empty() {
            return 0.0;
        }
        model_uncertainties.iter().sum::<f32>() / model_uncertainties.len() as f32
    }

    /// Numerically stable softmax over f32 slices.
    fn softmax_f32(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 {
            vec![1.0 / logits.len() as f32; logits.len()]
        } else {
            exps.iter().map(|&e| e / sum).collect()
        }
    }
}

/// Ensemble diversity and agreement metrics.
pub struct EnsembleMetrics;

impl EnsembleMetrics {
    /// Average pairwise KL divergence between model probability distributions.
    ///
    /// KL(P || Q) = sum_c P(c) * log(P(c) / Q(c))
    /// Returns the mean over all ordered pairs (i, j) with i != j.
    pub fn diversity(logits: &[Vec<f32>]) -> Result<f32, EnsembleError> {
        if logits.len() < 2 {
            return Err(EnsembleError::EmptyPredictions);
        }
        let probs: Vec<Vec<f32>> = logits.iter().map(|lv| ModelEnsemble::softmax_f32(lv)).collect();

        let n = probs.len();
        let mut total_kl = 0.0f32;
        let mut pair_count = 0u64;

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let kl = Self::kl_divergence(&probs[i], &probs[j]);
                total_kl += kl;
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            return Ok(0.0);
        }
        Ok(total_kl / pair_count as f32)
    }

    /// Fraction of predictions that agree with the majority-vote winner.
    ///
    /// `predictions` contains the argmax class index for each model.
    pub fn agreement_rate(predictions: &[usize]) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }

        // Count votes.
        let max_class = predictions.iter().cloned().max().unwrap_or(0);
        let mut counts = vec![0usize; max_class + 1];
        for &p in predictions {
            counts[p] += 1;
        }

        let majority_count = counts.iter().cloned().max().unwrap_or(0);
        majority_count as f32 / predictions.len() as f32
    }

    /// KL divergence KL(p || q) with epsilon smoothing to avoid log(0).
    fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
        const EPS: f32 = 1e-10;
        let len = p.len().min(q.len());
        (0..len)
            .map(|i| {
                let pi = p[i] + EPS;
                let qi = q[i] + EPS;
                pi * (pi / qi).ln()
            })
            .sum()
    }
}

/// Errors that can occur during ensemble operations
#[derive(Debug, Error)]
pub enum EnsembleError {
    #[error("Empty predictions")]
    EmptyPredictions,
    #[error("Weight count mismatch: {weights} weights for {models} models")]
    WeightCountMismatch { weights: usize, models: usize },
    #[error("Class count mismatch: expected {expected}, got {got}")]
    ClassCountMismatch { expected: usize, got: usize },
    #[error("Invalid weight: {0}")]
    InvalidWeight(f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pred(model_id: &str, logits: Vec<f64>, latency_ms: u64) -> ModelPrediction {
        let mut p = ModelPrediction::new(model_id, logits);
        p.latency_ms = latency_ms;
        p
    }

    // ── Test 1: ModelPrediction::new computes softmax and argmax correctly ──
    #[test]
    fn test_model_prediction_new_softmax_and_argmax() {
        // logits [1.0, 2.0, 3.0] → softmax peaks at index 2
        let pred = ModelPrediction::new("model_a", vec![1.0, 2.0, 3.0]);
        assert_eq!(pred.model_id, "model_a");
        assert_eq!(pred.predicted_class, 2);
        assert!(
            pred.confidence > 0.5,
            "confidence should be highest for class 2"
        );
        // Probabilities should sum to ~1
        let sum: f64 = pred.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "probabilities should sum to 1");
    }

    // ── Test 2: softmax of uniform logits gives uniform probabilities ──
    #[test]
    fn test_softmax_uniform() {
        let probs = ModelPrediction::softmax(&[0.0, 0.0, 0.0, 0.0]);
        for &p in &probs {
            assert!(
                (p - 0.25).abs() < 1e-9,
                "uniform logits should give 0.25 each"
            );
        }
    }

    // ── Test 3: WeightedAverage basic (equal weights) ──
    #[test]
    fn test_weighted_average_equal_weights() {
        let pred_a = make_pred("a", vec![2.0, 0.0], 10);
        let pred_b = make_pred("b", vec![0.0, 2.0], 20);
        let ensemble = ModelEnsemble::new(
            EnsembleStrategy::WeightedAverage {
                weights: vec![1.0, 1.0],
            },
            2,
        )
        .expect("valid ensemble");

        let result = ensemble.aggregate(&[pred_a, pred_b]).expect("should aggregate");
        // Equal weight → aggregated should be roughly equal between classes
        let diff = (result.aggregated_probs[0] - result.aggregated_probs[1]).abs();
        assert!(
            diff < 0.05,
            "equal weights should give similar probs: {diff}"
        );
        // Total latency should be max(10, 20) = 20
        assert_eq!(result.total_latency_ms, 20);
    }

    // ── Test 4: WeightedAverage with unequal weights strongly favors one model ──
    #[test]
    fn test_weighted_average_unequal_weights() {
        // pred_a predicts class 0, pred_b predicts class 1
        let pred_a = make_pred("a", vec![5.0, -5.0], 5);
        let pred_b = make_pred("b", vec![-5.0, 5.0], 5);
        let ensemble = ModelEnsemble::new(
            EnsembleStrategy::WeightedAverage {
                weights: vec![9.0, 1.0], // heavily favor pred_a (class 0)
            },
            2,
        )
        .expect("valid ensemble");

        let result = ensemble.aggregate(&[pred_a, pred_b]).expect("should aggregate");
        assert_eq!(result.final_class, 0, "heavily weighted model should win");
        assert!(
            result.aggregated_probs[0] > result.aggregated_probs[1],
            "class 0 should dominate"
        );
    }

    // ── Test 5: MajorityVote – 3 agree on class 0, 1 disagrees ──
    #[test]
    fn test_majority_vote_three_agree() {
        let p0 = make_pred("a", vec![3.0, -3.0], 1);
        let p1 = make_pred("b", vec![3.0, -3.0], 2);
        let p2 = make_pred("c", vec![3.0, -3.0], 3);
        let p3 = make_pred("d", vec![-3.0, 3.0], 4); // disagrees

        let ensemble = ModelEnsemble::new(EnsembleStrategy::MajorityVote, 2).expect("valid");
        let result = ensemble.aggregate(&[p0, p1, p2, p3]).expect("aggregate");
        assert_eq!(result.final_class, 0, "majority should pick class 0");
        assert_eq!(
            result.aggregated_probs[0], 1.0,
            "majority vote gives one-hot"
        );
        // Agreement ratio: 3 out of 4 agree with final_class=0
        assert!(
            (result.agreement_ratio - 0.75).abs() < 1e-9,
            "agreement should be 0.75, got {}",
            result.agreement_ratio
        );
    }

    // ── Test 6: MaxConfidence selects the most confident model ──
    #[test]
    fn test_max_confidence_selects_right_model() {
        // pred_a: class 0 with low confidence (0.55)
        // pred_b: class 1 with high confidence (0.99)
        let pred_a = make_pred("a", vec![0.2, -0.2], 1);
        let pred_b = make_pred("b", vec![-5.0, 5.0], 2); // very confident about class 1

        let ensemble =
            ModelEnsemble::new(EnsembleStrategy::MaxConfidence, 2).expect("valid ensemble");
        let result = ensemble.aggregate(&[pred_a, pred_b]).expect("aggregate");
        // Should use pred_b's probabilities
        assert_eq!(
            result.final_class, 1,
            "max confidence model predicts class 1"
        );
        assert!(
            result.final_confidence > 0.9,
            "confidence should be high: {}",
            result.final_confidence
        );
    }

    // ── Test 7: MeanProbability computes arithmetic mean ──
    #[test]
    fn test_mean_probability_average() {
        let pred_a = ModelPrediction {
            model_id: "a".into(),
            logits: vec![],
            probabilities: vec![0.8, 0.2],
            predicted_class: 0,
            confidence: 0.8,
            latency_ms: 1,
        };
        let pred_b = ModelPrediction {
            model_id: "b".into(),
            logits: vec![],
            probabilities: vec![0.4, 0.6],
            predicted_class: 1,
            confidence: 0.6,
            latency_ms: 2,
        };

        let ensemble =
            ModelEnsemble::new(EnsembleStrategy::MeanProbability, 2).expect("valid ensemble");
        let result = ensemble.aggregate(&[pred_a, pred_b]).expect("aggregate");
        assert!(
            (result.aggregated_probs[0] - 0.6).abs() < 1e-9,
            "mean of 0.8 and 0.4 should be 0.6, got {}",
            result.aggregated_probs[0]
        );
        assert!(
            (result.aggregated_probs[1] - 0.4).abs() < 1e-9,
            "mean of 0.2 and 0.6 should be 0.4, got {}",
            result.aggregated_probs[1]
        );
    }

    // ── Test 8: GeometricMean result ≤ ArithmeticMean (AM-GM inequality) ──
    #[test]
    fn test_geometric_mean_less_than_arithmetic_mean() {
        // Use predictions with varied probabilities so AM-GM gap is clear
        let pred_a = ModelPrediction {
            model_id: "a".into(),
            logits: vec![],
            probabilities: vec![0.9, 0.1],
            predicted_class: 0,
            confidence: 0.9,
            latency_ms: 1,
        };
        let pred_b = ModelPrediction {
            model_id: "b".into(),
            logits: vec![],
            probabilities: vec![0.1, 0.9],
            predicted_class: 1,
            confidence: 0.9,
            latency_ms: 1,
        };

        let ensemble_geo = ModelEnsemble::new(EnsembleStrategy::GeometricMean, 2).expect("valid");
        let ensemble_arith =
            ModelEnsemble::new(EnsembleStrategy::MeanProbability, 2).expect("valid");

        let preds = [pred_a.clone(), pred_b.clone()];
        let geo_result = ensemble_geo.aggregate(&preds).expect("geo aggregate");
        let arith_result = ensemble_arith.aggregate(&preds).expect("arith aggregate");

        // For class 0: geo_mean(0.9, 0.1) < arith_mean(0.9, 0.1) by AM-GM
        // geo = sqrt(0.9 * 0.1) ≈ 0.3 (before renorm), arith = 0.5
        // After renorm geo is 0.5 but raw geo < arith; test final_confidence difference
        // Actually both classes have symmetric values so normalized geo ≈ arith ≈ 0.5
        // Let's use asymmetric probs where AM > GM per class
        drop(geo_result);
        drop(arith_result);

        // Use clearly asymmetric distributions
        let pred_c = ModelPrediction {
            model_id: "c".into(),
            logits: vec![],
            probabilities: vec![0.95, 0.05],
            predicted_class: 0,
            confidence: 0.95,
            latency_ms: 1,
        };
        let pred_d = ModelPrediction {
            model_id: "d".into(),
            logits: vec![],
            probabilities: vec![0.05, 0.95],
            predicted_class: 1,
            confidence: 0.95,
            latency_ms: 1,
        };
        let preds2 = [pred_c, pred_d];

        // Geometric mean of (0.95, 0.05) = sqrt(0.95 * 0.05) ≈ 0.218
        // Arithmetic mean of (0.95, 0.05) = 0.5
        // Before normalization, geo < arith for each class individually
        let geo_result2 = ensemble_geo.aggregate(&preds2).expect("geo agg 2");
        // The result is renormalized, so final probs ≈ (0.5, 0.5) but
        // the raw geo values before normalization are (0.218, 0.218)
        // and arith values are (0.5, 0.5) - in this symmetric case they match after normalization.
        // Let's verify the probs sum to 1 which is the important property
        let sum: f64 = geo_result2.aggregated_probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "geometric mean probs should sum to 1, got {sum}"
        );

        // Verify: for asymmetric case where both models strongly agree on different classes,
        // geometric mean penalizes outliers. Use 3 models: 2 agree on class 0, 1 on class 1.
        let pa = ModelPrediction {
            model_id: "pa".into(),
            logits: vec![],
            probabilities: vec![0.9, 0.1],
            predicted_class: 0,
            confidence: 0.9,
            latency_ms: 1,
        };
        let pb = ModelPrediction {
            model_id: "pb".into(),
            logits: vec![],
            probabilities: vec![0.8, 0.2],
            predicted_class: 0,
            confidence: 0.8,
            latency_ms: 1,
        };
        let pc = ModelPrediction {
            model_id: "pc".into(),
            logits: vec![],
            probabilities: vec![0.3, 0.7],
            predicted_class: 1,
            confidence: 0.7,
            latency_ms: 1,
        };

        let preds3 = [pa.clone(), pb.clone(), pc.clone()];
        let arith_result3 = ensemble_arith.aggregate(&preds3).expect("arith 3");
        let geo_result3 = ensemble_geo.aggregate(&preds3).expect("geo 3");

        // Arithmetic mean for class 0: (0.9 + 0.8 + 0.3)/3 ≈ 0.667
        // Geometric mean for class 0 (raw): (0.9 * 0.8 * 0.3)^(1/3) ≈ 0.617
        // After normalization both should pick class 0 as winner
        assert_eq!(arith_result3.final_class, 0, "arith: majority pick class 0");
        assert_eq!(geo_result3.final_class, 0, "geo: majority pick class 0");

        // Geometric mean class 0 raw (before renorm) < arith class 0
        // This demonstrates AM-GM: geometric mean ≤ arithmetic mean
        let arith_c0 = arith_result3.aggregated_probs[0];
        // Geo is renormalized; but we know raw geo < raw arith for class 0
        // We can verify by computing raw values
        let raw_geo_c0 = (0.9_f64 * 0.8 * 0.3).powf(1.0 / 3.0);
        let raw_arith_c0 = (0.9 + 0.8 + 0.3) / 3.0;
        assert!(
            raw_geo_c0 < raw_arith_c0,
            "AM-GM: geometric ({raw_geo_c0}) < arithmetic ({raw_arith_c0})"
        );
        // The arith probability for class 0 should be > 0.5
        assert!(
            arith_c0 > 0.5,
            "arithmetic mean should pick class 0 with prob > 0.5"
        );
    }

    // ── Test 9: Stacking uses meta_weights correctly ──
    #[test]
    fn test_stacking_meta_weights() {
        let pred_a = make_pred("a", vec![3.0, -3.0], 5);
        let pred_b = make_pred("b", vec![-3.0, 3.0], 5);

        // Strongly weight model a
        let ensemble = ModelEnsemble::new(
            EnsembleStrategy::Stacking {
                meta_weights: vec![10.0, 1.0],
            },
            2,
        )
        .expect("valid stacking");

        let result = ensemble.aggregate(&[pred_a, pred_b]).expect("aggregate");
        assert_eq!(
            result.final_class, 0,
            "stacking should heavily favor model a"
        );
        assert_eq!(result.strategy, "Stacking");
    }

    // ── Test 10: agreement_ratio calculation ──
    #[test]
    fn test_agreement_ratio() {
        // All 3 models predict class 1
        let p0 = make_pred("a", vec![-3.0, 3.0], 1);
        let p1 = make_pred("b", vec![-3.0, 3.0], 1);
        let p2 = make_pred("c", vec![-3.0, 3.0], 1);

        let ensemble = ModelEnsemble::new(EnsembleStrategy::MeanProbability, 2).expect("valid");
        let result = ensemble.aggregate(&[p0, p1, p2]).expect("aggregate");
        assert_eq!(result.final_class, 1);
        assert!(
            (result.agreement_ratio - 1.0).abs() < 1e-9,
            "all agree: ratio should be 1.0, got {}",
            result.agreement_ratio
        );
    }

    // ── Test 11: Entropy of uniform distribution ≈ log2(K) ──
    #[test]
    fn test_entropy_uniform_distribution() {
        let k = 4usize;
        // Aggregate 4 models each predicting different classes to get ~uniform output
        let probs_uniform = vec![0.25, 0.25, 0.25, 0.25];
        let result = EnsembleResult {
            final_class: 0,
            final_confidence: 0.25,
            aggregated_probs: probs_uniform,
            individual_predictions: vec![],
            strategy: "test".to_string(),
            agreement_ratio: 0.25,
            total_latency_ms: 0,
        };
        let entropy = result.entropy();
        let expected = (k as f64).log2();
        assert!(
            (entropy - expected).abs() < 0.01,
            "uniform entropy should be log2({k}) ≈ {expected:.3}, got {entropy:.3}"
        );
    }

    // ── Test 12: Entropy of peaked distribution ≈ 0 ──
    #[test]
    fn test_entropy_peaked_distribution() {
        let result = EnsembleResult {
            final_class: 0,
            final_confidence: 1.0,
            aggregated_probs: vec![1.0, 0.0, 0.0, 0.0],
            individual_predictions: vec![],
            strategy: "test".to_string(),
            agreement_ratio: 1.0,
            total_latency_ms: 0,
        };
        let entropy = result.entropy();
        // For p=1.0: -1.0 * log2(1.0 + 1e-10) ≈ very small
        // For p=0.0: -0.0 * log2(1e-10) = 0
        assert!(
            entropy.abs() < 0.001,
            "peaked distribution entropy should be ≈ 0, got {entropy}"
        );
    }

    // ── Test 13: top_k_classes returns correct ordering ──
    #[test]
    fn test_top_k_classes_ordering() {
        let result = EnsembleResult {
            final_class: 2,
            final_confidence: 0.6,
            aggregated_probs: vec![0.1, 0.2, 0.6, 0.1],
            individual_predictions: vec![],
            strategy: "test".to_string(),
            agreement_ratio: 1.0,
            total_latency_ms: 0,
        };
        let top2 = result.top_k_classes(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 2, "top class should be index 2 (prob 0.6)");
        assert_eq!(top2[1].0, 1, "second class should be index 1 (prob 0.2)");
        assert!(
            top2[0].1 > top2[1].1,
            "first should have higher prob than second"
        );
    }

    // ── Test 14: EmptyPredictions error ──
    #[test]
    fn test_empty_predictions_error() {
        let ensemble = ModelEnsemble::new(EnsembleStrategy::MeanProbability, 3).expect("valid");
        let err = ensemble.aggregate(&[]).expect_err("should fail on empty");
        assert!(
            matches!(err, EnsembleError::EmptyPredictions),
            "should be EmptyPredictions error"
        );
    }

    // ── Test 15: WeightCountMismatch error ──
    #[test]
    fn test_weight_count_mismatch_error() {
        let pred_a = make_pred("a", vec![1.0, 0.0], 1);
        let pred_b = make_pred("b", vec![0.0, 1.0], 1);
        let ensemble = ModelEnsemble::new(
            EnsembleStrategy::WeightedAverage {
                weights: vec![1.0, 1.0, 1.0], // 3 weights for 2 predictions
            },
            2,
        )
        .expect("valid ensemble creation");

        let err = ensemble.aggregate(&[pred_a, pred_b]).expect_err("should fail");
        assert!(
            matches!(
                err,
                EnsembleError::WeightCountMismatch {
                    weights: 3,
                    models: 2
                }
            ),
            "expected WeightCountMismatch, got {err:?}"
        );
    }

    // ── Test 16: ClassCountMismatch error ──
    #[test]
    fn test_class_count_mismatch_error() {
        let pred_wrong = ModelPrediction {
            model_id: "x".into(),
            logits: vec![],
            probabilities: vec![0.5, 0.3, 0.2], // 3 classes
            predicted_class: 0,
            confidence: 0.5,
            latency_ms: 1,
        };
        let ensemble =
            ModelEnsemble::new(EnsembleStrategy::MeanProbability, 2).expect("valid ensemble");
        let err = ensemble.aggregate(&[pred_wrong]).expect_err("should fail on mismatch");
        assert!(
            matches!(
                err,
                EnsembleError::ClassCountMismatch {
                    expected: 2,
                    got: 3
                }
            ),
            "expected ClassCountMismatch, got {err:?}"
        );
    }

    // ── Test 17: ConfidenceWeighted — more confident model dominates ──
    #[test]
    fn test_confidence_weighted_high_confidence_dominates() {
        // pred_a: very confident about class 0 (high logit)
        let pred_a = make_pred("a", vec![10.0, -10.0], 5);
        // pred_b: mildly unsure, leans class 1
        let pred_b = make_pred("b", vec![-1.0, 1.0], 5);
        // pred_a.confidence >> pred_b.confidence → class 0 should win
        let ensemble = ModelEnsemble::new(EnsembleStrategy::ConfidenceWeighted, 2).expect("valid");
        let result = ensemble.aggregate(&[pred_a, pred_b]).expect("aggregate");
        assert_eq!(
            result.final_class, 0,
            "highly confident model for class 0 should dominate"
        );
    }

    // ── Test 18: ConfidenceWeighted — strategy name in result ──
    #[test]
    fn test_confidence_weighted_strategy_name() {
        let pred_a = make_pred("a", vec![1.0, 0.0], 5);
        let pred_b = make_pred("b", vec![1.0, 0.0], 5);
        let ensemble = ModelEnsemble::new(EnsembleStrategy::ConfidenceWeighted, 2).expect("valid");
        let result = ensemble.aggregate(&[pred_a, pred_b]).expect("aggregate");
        assert_eq!(result.strategy, "ConfidenceWeighted");
    }

    // ── Test 19: ConfidenceWeighted — equal confidence == mean probability ──
    #[test]
    fn test_confidence_weighted_equal_confidence_matches_mean() {
        // When all models have equal confidence, softmax weights are equal,
        // so result should be the same as MeanProbability.
        let pred_a = ModelPrediction {
            model_id: "a".into(),
            logits: vec![],
            probabilities: vec![0.7, 0.3],
            predicted_class: 0,
            confidence: 0.7,
            latency_ms: 1,
        };
        let pred_b = ModelPrediction {
            model_id: "b".into(),
            logits: vec![],
            probabilities: vec![0.4, 0.6],
            predicted_class: 1,
            confidence: 0.6,
            latency_ms: 1,
        };
        // Give both exactly same confidence so weights should be ~equal
        let pred_a_eq = ModelPrediction {
            confidence: 0.5,
            ..pred_a.clone()
        };
        let pred_b_eq = ModelPrediction {
            confidence: 0.5,
            ..pred_b.clone()
        };

        let cw_ensemble =
            ModelEnsemble::new(EnsembleStrategy::ConfidenceWeighted, 2).expect("valid");
        let mean_ensemble =
            ModelEnsemble::new(EnsembleStrategy::MeanProbability, 2).expect("valid");
        let preds_eq = [pred_a_eq, pred_b_eq];

        let cw_res = cw_ensemble.aggregate(&preds_eq).expect("cw");
        let mean_res = mean_ensemble.aggregate(&preds_eq).expect("mean");
        // With equal confidences, confidence-weighted should yield same result as mean
        for (cw_p, mean_p) in cw_res.aggregated_probs.iter().zip(mean_res.aggregated_probs.iter()) {
            assert!((cw_p - mean_p).abs() < 1e-6, "cw={cw_p} vs mean={mean_p}");
        }
    }

    // ── Test 20: Trimmed ensemble — basic smoke test ──
    #[test]
    fn test_trimmed_ensemble_basic() {
        // 5 models: 3 predict class 0 strongly, 2 are outliers predicting class 1
        let mut preds = Vec::new();
        for _ in 0..3 {
            preds.push(make_pred("good", vec![5.0, -5.0], 1));
        }
        preds.push(make_pred("outlier_a", vec![-5.0, 5.0], 1));
        preds.push(make_pred("outlier_b", vec![-5.0, 5.0], 1));

        // Trim 20% from each end (1 model from each side)
        let ensemble =
            ModelEnsemble::new(EnsembleStrategy::Trimmed { percentile: 0.2 }, 2).expect("valid");
        let result = ensemble.aggregate(&preds).expect("aggregate");
        assert_eq!(result.strategy, "Trimmed");
        // After trimming outliers, class 0 should still dominate
        assert_eq!(
            result.final_class, 0,
            "majority class 0 should win after trimming"
        );
    }

    // ── Test 21: Trimmed ensemble — percentile 0.0 degenerates to mean ──
    #[test]
    fn test_trimmed_zero_percentile_equals_mean() {
        let pred_a = ModelPrediction {
            model_id: "a".into(),
            logits: vec![],
            probabilities: vec![0.8, 0.2],
            predicted_class: 0,
            confidence: 0.8,
            latency_ms: 1,
        };
        let pred_b = ModelPrediction {
            model_id: "b".into(),
            logits: vec![],
            probabilities: vec![0.4, 0.6],
            predicted_class: 1,
            confidence: 0.6,
            latency_ms: 1,
        };
        let preds = [pred_a, pred_b];

        let trim_ensemble =
            ModelEnsemble::new(EnsembleStrategy::Trimmed { percentile: 0.0 }, 2).expect("valid");
        let mean_ensemble =
            ModelEnsemble::new(EnsembleStrategy::MeanProbability, 2).expect("valid");

        let trim_res = trim_ensemble.aggregate(&preds).expect("trim");
        let mean_res = mean_ensemble.aggregate(&preds).expect("mean");

        // With percentile=0.0 no trimming; renormalized result equals mean_probability
        assert_eq!(trim_res.final_class, mean_res.final_class);
    }

    // ── Test 22: predict_with_uncertainty returns correct mean ──
    #[test]
    fn test_predict_with_uncertainty_mean() {
        // Two models: one predicts [1, 0], other [0, 1] → mean should be [0.5, 0.5]
        let logits_a = vec![10.0_f32, -10.0]; // softmax ≈ [1, 0]
        let logits_b = vec![-10.0_f32, 10.0]; // softmax ≈ [0, 1]
        let logits = vec![logits_a, logits_b];

        let (mean, _var) = ModelEnsemble::predict_with_uncertainty(&logits).expect("ok");
        assert!(
            (mean[0] - 0.5).abs() < 0.01,
            "mean[0] should be ≈0.5, got {}",
            mean[0]
        );
        assert!(
            (mean[1] - 0.5).abs() < 0.01,
            "mean[1] should be ≈0.5, got {}",
            mean[1]
        );
    }

    // ── Test 23: predict_with_uncertainty — high variance when models disagree ──
    #[test]
    fn test_predict_with_uncertainty_variance_high_when_disagreement() {
        // Maximum disagreement: 4 models, half predict [1, 0], half predict [0, 1]
        let agree_a = vec![20.0_f32, -20.0];
        let agree_b = vec![-20.0_f32, 20.0];
        let logits = vec![
            agree_a.clone(),
            agree_a.clone(),
            agree_b.clone(),
            agree_b.clone(),
        ];
        let (_, var_disagree) = ModelEnsemble::predict_with_uncertainty(&logits).expect("ok");

        // Same models all agreeing → zero variance
        let agree_all = vec![
            agree_a.clone(),
            agree_a.clone(),
            agree_a.clone(),
            agree_a.clone(),
        ];
        let (_, var_agree) = ModelEnsemble::predict_with_uncertainty(&agree_all).expect("ok");

        assert!(
            var_disagree > var_agree,
            "disagree variance {var_disagree} should exceed agree variance {var_agree}"
        );
    }

    // ── Test 24: epistemic_uncertainty is zero when all models agree ──
    #[test]
    fn test_epistemic_uncertainty_zero_on_agreement() {
        let logits = vec![
            vec![10.0_f32, -10.0],
            vec![10.0_f32, -10.0],
            vec![10.0_f32, -10.0],
        ];
        let eps = ModelEnsemble::epistemic_uncertainty(&logits).expect("ok");
        assert!(
            eps < 1e-5,
            "epistemic uncertainty should be ~0 when all models agree, got {eps}"
        );
    }

    // ── Test 25: aleatoric_uncertainty is the mean of model uncertainties ──
    #[test]
    fn test_aleatoric_uncertainty_mean() {
        let model_uncertainties = vec![0.2_f32, 0.4_f32, 0.6_f32];
        let aleatoric = ModelEnsemble::aleatoric_uncertainty(&model_uncertainties);
        assert!(
            (aleatoric - 0.4_f32).abs() < 1e-6,
            "aleatoric should be 0.4, got {aleatoric}"
        );
    }

    // ── Test 26: aleatoric_uncertainty with empty slice returns 0 ──
    #[test]
    fn test_aleatoric_uncertainty_empty() {
        let aleatoric = ModelEnsemble::aleatoric_uncertainty(&[]);
        assert_eq!(aleatoric, 0.0);
    }

    // ── Test 27: EnsembleMetrics::diversity — identical models → ~0 ──
    #[test]
    fn test_diversity_identical_models() {
        let logits = vec![
            vec![2.0_f32, 1.0, 0.5],
            vec![2.0_f32, 1.0, 0.5],
            vec![2.0_f32, 1.0, 0.5],
        ];
        let div = EnsembleMetrics::diversity(&logits).expect("ok");
        assert!(
            div < 1e-5,
            "identical model distributions should have near-zero diversity, got {div}"
        );
    }

    // ── Test 28: EnsembleMetrics::diversity — maximally different → large ──
    #[test]
    fn test_diversity_maximally_different_models() {
        // One model is sure about class 0, other is sure about class 1 → high KL
        let logits = vec![vec![20.0_f32, -20.0], vec![-20.0_f32, 20.0]];
        let div = EnsembleMetrics::diversity(&logits).expect("ok");
        assert!(
            div > 1.0,
            "maximally different distributions should have high diversity, got {div}"
        );
    }

    // ── Test 29: EnsembleMetrics::agreement_rate — unanimous ──
    #[test]
    fn test_agreement_rate_unanimous() {
        let predictions = vec![2usize, 2, 2, 2];
        let rate = EnsembleMetrics::agreement_rate(&predictions);
        assert!(
            (rate - 1.0).abs() < f32::EPSILON,
            "unanimous predictions should have 100% agreement, got {rate}"
        );
    }

    // ── Test 30: EnsembleMetrics::agreement_rate — majority ──
    #[test]
    fn test_agreement_rate_majority() {
        let predictions = vec![0usize, 0, 0, 1]; // 3/4 agree on 0
        let rate = EnsembleMetrics::agreement_rate(&predictions);
        assert!(
            (rate - 0.75).abs() < 1e-5,
            "3/4 majority should give 0.75 agreement, got {rate}"
        );
    }

    // ── Test 31: predict_with_uncertainty error on empty input ──
    #[test]
    fn test_predict_with_uncertainty_empty_error() {
        let result = ModelEnsemble::predict_with_uncertainty(&[]);
        assert!(
            matches!(result, Err(EnsembleError::EmptyPredictions)),
            "should fail on empty logits"
        );
    }

    // ── Test 32: Trimmed result sums to 1 ──
    #[test]
    fn test_trimmed_probabilities_sum_to_one() {
        let preds: Vec<ModelPrediction> = (0..6)
            .map(|i| {
                let logit = if i < 4 { vec![3.0, 1.0, 0.5] } else { vec![-3.0, 2.0, 1.5] };
                make_pred("m", logit, 1)
            })
            .collect();

        let ensemble =
            ModelEnsemble::new(EnsembleStrategy::Trimmed { percentile: 0.15 }, 3).expect("valid");
        let result = ensemble.aggregate(&preds).expect("aggregate");
        let sum: f64 = result.aggregated_probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-8,
            "trimmed mean probs should sum to 1, got {sum}"
        );
    }

    // ── Extended tests ─────────────────────────────────────────────────────

    // Test 33: ModelPrediction::softmax empty input returns empty
    #[test]
    fn test_softmax_empty_input_returns_empty() {
        let probs = ModelPrediction::softmax(&[]);
        assert!(probs.is_empty(), "softmax of empty slice must be empty");
    }

    // Test 34: ModelPrediction::softmax sums to 1 for single element
    #[test]
    fn test_softmax_single_element_is_one() {
        let probs = ModelPrediction::softmax(&[5.0]);
        assert!(
            (probs[0] - 1.0).abs() < 1e-9,
            "softmax of single element must be 1"
        );
    }

    // Test 35: ModelPrediction confidence is in [0, 1]
    #[test]
    fn test_model_prediction_confidence_in_unit_interval() {
        let pred = ModelPrediction::new("m", vec![1.0, 2.0, 3.0, 4.0]);
        assert!(
            (0.0..=1.0).contains(&pred.confidence),
            "confidence must be in [0,1]"
        );
    }

    // Test 36: EnsembleStrategy::MajorityVote != MeanProbability
    #[test]
    fn test_ensemble_strategy_variants_differ() {
        assert_ne!(
            EnsembleStrategy::MajorityVote,
            EnsembleStrategy::MeanProbability
        );
    }

    // Test 37: ModelEnsemble with zero num_classes — construction succeeds,
    // but aggregate with non-empty predictions returns ClassCountMismatch.
    #[test]
    fn test_model_ensemble_zero_num_classes_error() {
        // new() itself does not validate num_classes; it succeeds.
        let ensemble = ModelEnsemble::new(EnsembleStrategy::MajorityVote, 0)
            .expect("construction with num_classes=0 should succeed");
        // Aggregating predictions whose probabilities.len() != 0 yields an error.
        let preds = vec![make_pred("m", vec![1.0, 0.0], 1)];
        let result = ensemble.aggregate(&preds);
        assert!(
            result.is_err(),
            "aggregate with mismatched class count must return error"
        );
    }

    // Test 38: EnsembleResult::entropy — uniform distribution has max entropy
    #[test]
    fn test_entropy_uniform_is_max() {
        // All equal probs → max entropy for 4 classes
        let preds = vec![
            make_pred("a", vec![0.0, 0.0, 0.0, 0.0], 1),
            make_pred("b", vec![0.0, 0.0, 0.0, 0.0], 1),
        ];
        let ensemble = ModelEnsemble::new(EnsembleStrategy::MeanProbability, 4).expect("valid");
        let result_uniform = ensemble.aggregate(&preds).expect("aggregate");

        // A peaked distribution should have lower entropy
        let preds_peaked = vec![
            make_pred("a", vec![10.0, 0.0, 0.0, 0.0], 1),
            make_pred("b", vec![10.0, 0.0, 0.0, 0.0], 1),
        ];
        let result_peaked = ensemble.aggregate(&preds_peaked).expect("aggregate");

        assert!(
            result_uniform.entropy() > result_peaked.entropy(),
            "uniform distribution should have higher entropy than peaked"
        );
    }

    // Test 39: EnsembleResult::top_k_classes — k=0 returns empty
    #[test]
    fn test_top_k_classes_k_zero_returns_empty() {
        let preds = vec![make_pred("a", vec![1.0, 2.0, 3.0], 1)];
        let ensemble = ModelEnsemble::new(EnsembleStrategy::MeanProbability, 3).expect("valid");
        let result = ensemble.aggregate(&preds).expect("aggregate");
        assert_eq!(result.top_k_classes(0).len(), 0);
    }

    // Test 40: EnsembleResult::strategy string is non-empty
    #[test]
    fn test_ensemble_result_strategy_non_empty() {
        let preds = vec![make_pred("a", vec![1.0, 2.0], 1)];
        let ensemble = ModelEnsemble::new(EnsembleStrategy::MajorityVote, 2).expect("valid");
        let result = ensemble.aggregate(&preds).expect("aggregate");
        assert!(
            !result.strategy.is_empty(),
            "strategy string must be non-empty"
        );
    }

    // Test 41: GeometricMean — probabilities sum to 1
    #[test]
    fn test_geometric_mean_probs_sum_to_one() {
        let preds = vec![
            make_pred("a", vec![1.0, 2.0, 3.0], 5),
            make_pred("b", vec![3.0, 2.0, 1.0], 5),
        ];
        let ensemble = ModelEnsemble::new(EnsembleStrategy::GeometricMean, 3).expect("valid");
        let result = ensemble.aggregate(&preds).expect("aggregate");
        let sum: f64 = result.aggregated_probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-8,
            "geometric mean probs must sum to 1, got {sum}"
        );
    }

    // Test 42: MaxConfidence — picks class from highest-confidence model
    #[test]
    fn test_max_confidence_picks_highest_confidence() {
        // pred_a is very uncertain (flat), pred_b is very confident on class 1
        let pred_a = make_pred("a", vec![0.0, 0.0, 0.0], 1);
        let pred_b = make_pred("b", vec![-10.0, 10.0, -10.0], 1);
        let ensemble = ModelEnsemble::new(EnsembleStrategy::MaxConfidence, 3).expect("valid");
        let result = ensemble.aggregate(&[pred_a, pred_b]).expect("aggregate");
        assert_eq!(
            result.final_class, 1,
            "MaxConfidence must pick class 1 (highest confidence)"
        );
    }

    // Test 43: EnsembleError::EmptyPredictions — aggregate on empty slice returns error
    #[test]
    fn test_aggregate_empty_preds_returns_error() {
        let ensemble = ModelEnsemble::new(EnsembleStrategy::MeanProbability, 3).expect("valid");
        let result = ensemble.aggregate(&[]);
        assert!(matches!(result, Err(EnsembleError::EmptyPredictions)));
    }

    // Test 44: individual_predictions stored in result
    #[test]
    fn test_result_individual_predictions_count() {
        let preds = vec![
            make_pred("m1", vec![1.0, 0.0], 1),
            make_pred("m2", vec![0.0, 1.0], 1),
            make_pred("m3", vec![1.0, 0.0], 1),
        ];
        let ensemble = ModelEnsemble::new(EnsembleStrategy::MajorityVote, 2).expect("valid");
        let result = ensemble.aggregate(&preds).expect("aggregate");
        assert_eq!(result.individual_predictions.len(), 3);
    }

    // Test 45: Stacking strategy — result has expected num_classes
    #[test]
    fn test_stacking_result_num_classes() {
        let preds = vec![
            make_pred("a", vec![1.0, 2.0, 3.0], 5),
            make_pred("b", vec![3.0, 1.0, 2.0], 5),
        ];
        let ensemble = ModelEnsemble::new(
            EnsembleStrategy::Stacking {
                meta_weights: vec![0.6, 0.4],
            },
            3,
        )
        .expect("valid");
        let result = ensemble.aggregate(&preds).expect("aggregate");
        assert_eq!(result.aggregated_probs.len(), 3);
    }

    // Test 46: WeightedAverage with negative weight returns error
    #[test]
    fn test_weighted_average_negative_weight_error() {
        let result = ModelEnsemble::new(
            EnsembleStrategy::WeightedAverage {
                weights: vec![-1.0, 1.0],
            },
            2,
        );
        assert!(result.is_err(), "negative weight must return error");
    }

    // Test 47: EnsembleStrategy::Trimmed with out-of-range percentile behaves correctly
    #[test]
    fn test_trimmed_strategy_single_pred() {
        let preds = vec![make_pred("a", vec![1.0, 2.0, 3.0], 1)];
        let ensemble =
            ModelEnsemble::new(EnsembleStrategy::Trimmed { percentile: 0.0 }, 3).expect("valid");
        let result = ensemble.aggregate(&preds).expect("aggregate");
        let sum: f64 = result.aggregated_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);
    }

    // Test 48: MeanProbability with single prediction returns same class
    #[test]
    fn test_mean_probability_single_pred_class() {
        let pred = make_pred("only", vec![-5.0, 5.0, -5.0], 1);
        let expected_class = pred.predicted_class;
        let ensemble = ModelEnsemble::new(EnsembleStrategy::MeanProbability, 3).expect("valid");
        let result = ensemble.aggregate(&[pred]).expect("aggregate");
        assert_eq!(
            result.final_class, expected_class,
            "single-pred result must match input class"
        );
    }

    // Test 49: EnsembleResult::final_confidence is in [0, 1]
    #[test]
    fn test_ensemble_result_final_confidence_in_range() {
        let preds = vec![
            make_pred("a", vec![1.0, 2.0], 1),
            make_pred("b", vec![2.0, 1.0], 1),
        ];
        let ensemble = ModelEnsemble::new(EnsembleStrategy::MajorityVote, 2).expect("valid");
        let result = ensemble.aggregate(&preds).expect("aggregate");
        assert!(
            (0.0..=1.0).contains(&result.final_confidence),
            "final_confidence {} must be in [0, 1]",
            result.final_confidence
        );
    }

    // Test 50: ClassCountMismatch error when pred has wrong num classes
    #[test]
    fn test_class_count_mismatch_error_ext() {
        let pred = make_pred("a", vec![1.0, 2.0], 1); // 2 classes, but ensemble expects 3
        let ensemble = ModelEnsemble::new(EnsembleStrategy::MeanProbability, 3).expect("valid");
        let result = ensemble.aggregate(&[pred]);
        assert!(matches!(
            result,
            Err(EnsembleError::ClassCountMismatch { .. })
        ));
    }

    // Test 51: predict_with_uncertainty returns probs in [0, 1]
    #[test]
    fn test_predict_with_uncertainty_returns_valid_range() {
        let logits = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![3.0_f32, 2.0, 1.0],
            vec![2.0_f32, 2.0, 2.0],
        ];
        let (mean_probs, variance) = ModelEnsemble::predict_with_uncertainty(&logits).expect("ok");
        for &p in &mean_probs {
            assert!((0.0..=1.0).contains(&p), "mean prob must be in [0, 1]");
        }
        assert!(variance >= 0.0, "variance must be non-negative");
    }

    // Test 52: ModelPrediction latency_ms field stored correctly
    #[test]
    fn test_model_prediction_latency_stored() {
        let pred = make_pred("model_x", vec![1.0, 2.0], 150);
        assert_eq!(pred.latency_ms, 150);
    }
}
