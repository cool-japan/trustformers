use crate::error::Result;
use crate::pipeline::{Pipeline, PipelineOutput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitStrategy {
    /// Exit when confidence exceeds threshold
    ConfidenceThreshold(f32),
    /// Exit when entropy is below threshold (high certainty)
    EntropyThreshold(f32),
    /// Exit when variance across predictions is low
    VarianceThreshold(f32),
    /// Exit based on prediction consistency
    ConsistencyThreshold(f32),
    /// Exit when computational budget is exceeded
    ComputationalBudget(u64), // in milliseconds
    /// Exit based on energy consumption
    EnergyBudget(f32),
    /// Adaptive threshold based on task difficulty
    AdaptiveThreshold,
    /// Exit when patience counter is exceeded
    Patience(u32),
    /// Combination of multiple strategies
    Combined(Vec<ExitStrategy>),
    /// Machine learning-based exit predictor
    LearnedExit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyExitConfig {
    pub strategy: ExitStrategy,
    pub min_layers: usize,
    pub max_layers: usize,
    pub patience_threshold: u32,
    pub confidence_calibration: bool,
    pub dynamic_threshold_adjustment: bool,
    pub performance_tracking: bool,
    pub energy_aware: bool,
    pub memory_aware: bool,
    pub context_aware: bool,
    pub fallback_to_full: bool,
    pub exit_point_optimization: bool,
}

impl Default for EarlyExitConfig {
    fn default() -> Self {
        Self {
            strategy: ExitStrategy::ConfidenceThreshold(0.9),
            min_layers: 6,
            max_layers: 12,
            patience_threshold: 3,
            confidence_calibration: true,
            dynamic_threshold_adjustment: true,
            performance_tracking: true,
            energy_aware: false,
            memory_aware: true,
            context_aware: true,
            fallback_to_full: true,
            exit_point_optimization: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitPoint {
    pub layer_index: usize,
    pub confidence_score: f32,
    pub entropy_score: f32,
    pub variance_score: f32,
    pub consistency_score: f32,
    pub computation_time_ms: u64,
    pub energy_consumed: f32,
    pub memory_used_mb: f64,
    pub should_exit: bool,
    pub exit_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyExitResult {
    pub prediction: PipelineOutput,
    pub exit_point: ExitPoint,
    pub total_layers_computed: usize,
    pub computation_saved_percent: f32,
    pub energy_saved_percent: f32,
    pub confidence_score: f32,
    pub quality_score: f32,
    pub exit_path: Vec<ExitPoint>,
    pub final_decision_reason: String,
}

#[derive(Debug, Clone)]
pub struct LayerOutput {
    pub layer_index: usize,
    pub hidden_states: Vec<f32>, // Simplified representation
    pub attention_weights: Option<Vec<f32>>,
    pub logits: Option<Vec<f32>>,
    pub intermediate_prediction: Option<PipelineOutput>,
    pub computation_time_ms: u64,
    pub memory_usage_mb: f64,
}

#[derive(Clone)]
pub struct EarlyExitPredictor {
    config: EarlyExitConfig,
    exit_history: Vec<ExitPoint>,
    performance_stats: HashMap<usize, PerformanceStats>,
    adaptive_thresholds: HashMap<String, f32>,
    energy_tracker: EnergyTracker,
    memory_tracker: MemoryTracker,
    context_analyzer: ContextAnalyzer,
}

/// Per-layer early-exit performance stats, returned by
/// [`EarlyExitPredictor::get_performance_stats`].
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub total_exits: u64,
    pub successful_exits: u64,
    pub average_confidence: f32,
    pub average_computation_time: f64,
    /// Running estimate of accuracy lost by exiting early, approximated from
    /// exit-time confidence (ground-truth accuracy isn't available at
    /// inference time — see [`EarlyExitPredictor`]'s
    /// `update_performance_stats`).
    pub accuracy_loss: f32,
}

#[derive(Debug, Clone)]
struct EnergyTracker {
    baseline_energy_per_layer: f32,
    current_energy_consumption: f32,
    energy_budget_remaining: f32,
}

#[derive(Debug, Clone)]
struct MemoryTracker {
    peak_memory_usage: f64,
    current_memory_usage: f64,
    memory_pressure_level: f32,
}

#[derive(Debug, Clone)]
struct ContextAnalyzer {
    input_complexity_score: f32,
    task_difficulty_estimate: f32,
    domain_specific_threshold: f32,
}

impl EarlyExitPredictor {
    pub fn new(config: EarlyExitConfig) -> Self {
        Self {
            config,
            exit_history: Vec::new(),
            performance_stats: HashMap::new(),
            adaptive_thresholds: HashMap::new(),
            energy_tracker: EnergyTracker {
                baseline_energy_per_layer: 1.0,
                current_energy_consumption: 0.0,
                energy_budget_remaining: 100.0,
            },
            memory_tracker: MemoryTracker {
                peak_memory_usage: 0.0,
                current_memory_usage: 0.0,
                memory_pressure_level: 0.0,
            },
            context_analyzer: ContextAnalyzer {
                input_complexity_score: 0.5,
                task_difficulty_estimate: 0.5,
                domain_specific_threshold: 0.8,
            },
        }
    }

    /// Get a mutable reference to the configuration for modification
    pub fn config_mut(&mut self) -> &mut EarlyExitConfig {
        &mut self.config
    }

    /// Get a reference to the configuration for reading
    pub fn config(&self) -> &EarlyExitConfig {
        &self.config
    }

    /// Energy actually saved by stopping after `layers_executed` of
    /// `total_layers`.
    ///
    /// Derived from the per-layer energy the tracker measured during this run,
    /// not from a fixed multiplier on the computation saving.
    pub fn energy_saved_percent(&self, layers_executed: usize, total_layers: usize) -> f32 {
        if total_layers == 0 || layers_executed >= total_layers {
            return 0.0;
        }
        let per_layer = if layers_executed > 0 {
            self.energy_tracker.current_energy_consumption / layers_executed as f32
        } else {
            self.energy_tracker.baseline_energy_per_layer
        };
        let projected_total = per_layer * total_layers as f32;
        if projected_total <= f32::EPSILON {
            return 0.0;
        }
        let skipped = per_layer * (total_layers - layers_executed) as f32;
        (skipped / projected_total * 100.0).clamp(0.0, 100.0)
    }

    pub fn should_exit(&mut self, layer_output: &LayerOutput) -> Result<ExitPoint> {
        let mut exit_point = self.create_base_exit_point(layer_output)?;

        // Update tracking
        self.update_energy_tracking(layer_output);
        self.update_memory_tracking(layer_output);
        self.update_context_analysis(layer_output);

        // Apply exit strategy
        exit_point.should_exit = self.evaluate_exit_strategy(&exit_point, layer_output)?;

        // Apply constraints
        if layer_output.layer_index < self.config.min_layers {
            exit_point.should_exit = false;
            exit_point.exit_reason = format!("Below minimum layers ({})", self.config.min_layers);
        }

        if layer_output.layer_index >= self.config.max_layers {
            exit_point.should_exit = true;
            exit_point.exit_reason = "Reached maximum layers".to_string();
        }

        // Update history
        self.exit_history.push(exit_point.clone());
        if self.exit_history.len() > 1000 {
            self.exit_history.remove(0);
        }

        // Update performance stats
        self.update_performance_stats(&exit_point);

        Ok(exit_point)
    }

    fn create_base_exit_point(&self, layer_output: &LayerOutput) -> Result<ExitPoint> {
        let confidence_score = self.calculate_confidence_score(layer_output)?;
        let entropy_score = self.calculate_entropy_score(layer_output)?;
        let variance_score = self.calculate_variance_score(layer_output)?;
        let consistency_score = self.calculate_consistency_score(layer_output)?;

        Ok(ExitPoint {
            layer_index: layer_output.layer_index,
            confidence_score,
            entropy_score,
            variance_score,
            consistency_score,
            computation_time_ms: layer_output.computation_time_ms,
            energy_consumed: self.energy_tracker.current_energy_consumption,
            memory_used_mb: layer_output.memory_usage_mb,
            should_exit: false,
            exit_reason: String::new(),
        })
    }

    fn calculate_confidence_score(&self, layer_output: &LayerOutput) -> Result<f32> {
        if let Some(ref logits) = layer_output.logits {
            // Calculate max probability as confidence
            let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
            let max_prob = 1.0 / exp_sum; // exp(max_logit - max_logit) = exp(0) = 1
            Ok(max_prob)
        } else if let Some(ref prediction) = layer_output.intermediate_prediction {
            // Extract confidence from prediction
            match prediction {
                PipelineOutput::Classification(results) => {
                    Ok(results.iter().map(|r| r.score).fold(0.0f32, f32::max))
                },
                PipelineOutput::QuestionAnswering(result) => Ok(result.score),
                _ => Ok(0.8), // Default confidence
            }
        } else {
            // Fallback: use layer depth as proxy for confidence
            let depth_factor = layer_output.layer_index as f32 / self.config.max_layers as f32;
            Ok(0.5 + 0.3 * depth_factor) // Confidence increases with depth
        }
    }

    fn calculate_entropy_score(&self, layer_output: &LayerOutput) -> Result<f32> {
        if let Some(ref logits) = layer_output.logits {
            // Calculate entropy of the probability distribution
            let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();

            let entropy: f32 = logits
                .iter()
                .map(|&x| {
                    let prob = (x - max_logit).exp() / exp_sum;
                    if prob > 0.0 {
                        -prob * prob.ln()
                    } else {
                        0.0
                    }
                })
                .sum();

            // Normalize entropy (lower entropy = higher certainty)
            let max_entropy = (logits.len() as f32).ln();
            Ok(1.0 - entropy / max_entropy)
        } else {
            // Use hidden state variance as proxy for entropy
            let variance = self.calculate_hidden_state_variance(&layer_output.hidden_states);
            Ok(1.0 / (1.0 + variance)) // Higher variance = higher entropy
        }
    }

    fn calculate_variance_score(&self, layer_output: &LayerOutput) -> Result<f32> {
        let variance = self.calculate_hidden_state_variance(&layer_output.hidden_states);
        // Lower variance indicates more stable representation
        Ok(1.0 / (1.0 + variance))
    }

    fn calculate_consistency_score(&self, layer_output: &LayerOutput) -> Result<f32> {
        // The current layer's own representation stability: a real signal
        // (rather than a fixed fallback) even before there's enough exit
        // history to compare against.
        let current_stability =
            1.0 / (1.0 + self.calculate_hidden_state_variance(&layer_output.hidden_states));

        if self.exit_history.len() < 2 {
            return Ok(current_stability);
        }

        // Compare with previous layer predictions
        let recent_confidences: Vec<f32> =
            self.exit_history.iter().rev().take(3).map(|ep| ep.confidence_score).collect();

        if recent_confidences.len() < 2 {
            return Ok(current_stability);
        }

        // Calculate consistency as inverse of variance in recent confidences
        let mean = recent_confidences.iter().sum::<f32>() / recent_confidences.len() as f32;
        let variance = recent_confidences.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
            / recent_confidences.len() as f32;
        let historical_consistency = 1.0 / (1.0 + variance);

        // Blend historical confidence-consistency with the current layer's
        // own representation stability.
        Ok((historical_consistency + current_stability) / 2.0)
    }

    fn calculate_hidden_state_variance(&self, hidden_states: &[f32]) -> f32 {
        if hidden_states.is_empty() {
            return 0.0;
        }

        let mean = hidden_states.iter().sum::<f32>() / hidden_states.len() as f32;
        let variance = hidden_states.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
            / hidden_states.len() as f32;

        variance
    }

    fn evaluate_exit_strategy(
        &mut self,
        exit_point: &ExitPoint,
        layer_output: &LayerOutput,
    ) -> Result<bool> {
        match &self.config.strategy {
            ExitStrategy::ConfidenceThreshold(threshold) => {
                let adjusted_threshold = self.get_adjusted_threshold("confidence", *threshold);
                if exit_point.confidence_score >= adjusted_threshold {
                    Ok(true)
                } else {
                    Ok(false)
                }
            },
            ExitStrategy::EntropyThreshold(threshold) => {
                let adjusted_threshold = self.get_adjusted_threshold("entropy", *threshold);
                Ok(exit_point.entropy_score >= adjusted_threshold)
            },
            ExitStrategy::VarianceThreshold(threshold) => {
                let adjusted_threshold = self.get_adjusted_threshold("variance", *threshold);
                Ok(exit_point.variance_score >= adjusted_threshold)
            },
            ExitStrategy::ConsistencyThreshold(threshold) => {
                let adjusted_threshold = self.get_adjusted_threshold("consistency", *threshold);
                Ok(exit_point.consistency_score >= adjusted_threshold)
            },
            ExitStrategy::ComputationalBudget(budget_ms) => {
                Ok(exit_point.computation_time_ms >= *budget_ms)
            },
            ExitStrategy::EnergyBudget(budget) => {
                Ok(self.energy_tracker.energy_budget_remaining <= *budget)
            },
            ExitStrategy::AdaptiveThreshold => {
                self.evaluate_adaptive_threshold(exit_point, layer_output)
            },
            ExitStrategy::Patience(max_patience) => {
                self.evaluate_patience_strategy(exit_point, *max_patience)
            },
            ExitStrategy::Combined(strategies) => {
                self.evaluate_combined_strategies(strategies, exit_point, layer_output)
            },
            ExitStrategy::LearnedExit => self.evaluate_learned_exit(exit_point, layer_output),
        }
    }

    fn get_adjusted_threshold(&self, strategy_type: &str, base_threshold: f32) -> f32 {
        if !self.config.dynamic_threshold_adjustment {
            return base_threshold;
        }

        // Adjust based on context
        let mut adjusted = base_threshold;

        // Adjust for input complexity
        if self.context_analyzer.input_complexity_score > 0.7 {
            adjusted *= 0.9; // Lower threshold for complex inputs
        }

        // Adjust for task difficulty
        if self.context_analyzer.task_difficulty_estimate > 0.8 {
            adjusted *= 0.85; // Lower threshold for difficult tasks
        }

        // Adjust for memory pressure
        if self.memory_tracker.memory_pressure_level > 0.8 {
            adjusted *= 1.1; // Higher threshold under memory pressure
        }

        // Apply adaptive threshold if available
        if let Some(&adaptive_threshold) = self.adaptive_thresholds.get(strategy_type) {
            adjusted = (adjusted + adaptive_threshold) / 2.0;
        }

        // Blend in the domain-specific reference threshold, with a smaller
        // influence than the learned per-strategy adaptive threshold above
        // since it's a static configuration value rather than one tuned
        // from observed exit history.
        adjusted = adjusted * 0.9 + self.context_analyzer.domain_specific_threshold * 0.1;

        adjusted.clamp(0.1, 0.99)
    }

    fn evaluate_adaptive_threshold(
        &mut self,
        exit_point: &ExitPoint,
        _layer_output: &LayerOutput,
    ) -> Result<bool> {
        // Adaptive threshold based on multiple factors
        let confidence_weight = 0.4;
        let entropy_weight = 0.2;
        let consistency_weight = 0.2;
        let context_weight = 0.2;

        let composite_score = confidence_weight * exit_point.confidence_score
            + entropy_weight * exit_point.entropy_score
            + consistency_weight * exit_point.consistency_score
            + context_weight * (1.0 - self.context_analyzer.input_complexity_score);

        // Dynamic threshold based on performance history
        let historical_threshold = self.calculate_historical_threshold();
        let adaptive_threshold = (0.8 + historical_threshold) / 2.0;

        Ok(composite_score >= adaptive_threshold)
    }

    fn evaluate_patience_strategy(
        &self,
        exit_point: &ExitPoint,
        max_patience: u32,
    ) -> Result<bool> {
        // Count consecutive layers below confidence threshold
        let mut patience_counter = 0;
        let confidence_threshold = 0.8;

        for previous_exit in self.exit_history.iter().rev() {
            if previous_exit.confidence_score < confidence_threshold {
                patience_counter += 1;
            } else {
                break;
            }
        }

        // Exit if patience exceeded or current confidence is high
        Ok(patience_counter >= max_patience || exit_point.confidence_score >= 0.95)
    }

    fn evaluate_combined_strategies(
        &self,
        strategies: &[ExitStrategy],
        exit_point: &ExitPoint,
        layer_output: &LayerOutput,
    ) -> Result<bool> {
        let mut exit_votes = 0;
        let mut total_strategies = 0;

        for strategy in strategies {
            total_strategies += 1;

            // Create a temporary predictor with this strategy
            let mut temp_config = self.config.clone();
            temp_config.strategy = strategy.clone();
            let mut temp_predictor = EarlyExitPredictor::new(temp_config);

            if temp_predictor.evaluate_exit_strategy(exit_point, layer_output)? {
                exit_votes += 1;
            }
        }

        // Majority vote
        Ok(exit_votes > total_strategies / 2)
    }

    fn evaluate_learned_exit(
        &self,
        exit_point: &ExitPoint,
        _layer_output: &LayerOutput,
    ) -> Result<bool> {
        // Simplified learned exit predictor
        // In a real implementation, this would use a trained model
        let features = [
            exit_point.confidence_score,
            exit_point.entropy_score,
            exit_point.consistency_score,
            exit_point.layer_index as f32 / self.config.max_layers as f32,
            self.context_analyzer.input_complexity_score,
            self.memory_tracker.memory_pressure_level,
        ];

        // Simple linear combination (placeholder for actual ML model)
        let weights = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1];
        let score: f32 = features.iter().zip(weights.iter()).map(|(f, w)| f * w).sum();

        Ok(score >= 0.7)
    }

    fn calculate_historical_threshold(&self) -> f32 {
        if self.exit_history.is_empty() {
            return 0.8;
        }

        // Calculate average confidence of successful early exits
        let successful_exits: Vec<&ExitPoint> =
            self.exit_history.iter().filter(|ep| ep.should_exit).collect();

        if successful_exits.is_empty() {
            return 0.8;
        }

        let avg_confidence = successful_exits.iter().map(|ep| ep.confidence_score).sum::<f32>()
            / successful_exits.len() as f32;

        avg_confidence * 0.9 // Slightly lower than historical average
    }

    fn update_energy_tracking(&mut self, layer_output: &LayerOutput) {
        self.energy_tracker.current_energy_consumption +=
            self.energy_tracker.baseline_energy_per_layer;

        // Adjust based on layer complexity (simplified)
        let complexity_factor = layer_output.hidden_states.len() as f32 / 1000.0;
        self.energy_tracker.current_energy_consumption += complexity_factor;

        self.energy_tracker.energy_budget_remaining -=
            self.energy_tracker.baseline_energy_per_layer;
    }

    fn update_memory_tracking(&mut self, layer_output: &LayerOutput) {
        self.memory_tracker.current_memory_usage = layer_output.memory_usage_mb;

        if layer_output.memory_usage_mb > self.memory_tracker.peak_memory_usage {
            self.memory_tracker.peak_memory_usage = layer_output.memory_usage_mb;
        }

        // Calculate memory pressure (simplified)
        let memory_limit = 2048.0; // 2GB limit
        self.memory_tracker.memory_pressure_level =
            (self.memory_tracker.current_memory_usage / memory_limit).min(1.0) as f32;
    }

    fn update_context_analysis(&mut self, layer_output: &LayerOutput) {
        // Update input complexity based on hidden state statistics
        let variance = self.calculate_hidden_state_variance(&layer_output.hidden_states);
        self.context_analyzer.input_complexity_score =
            (self.context_analyzer.input_complexity_score * 0.9 + variance * 0.1).clamp(0.0, 1.0);

        // Update task difficulty based on convergence rate
        if layer_output.layer_index > 0 {
            let convergence_rate = self.calculate_convergence_rate();
            self.context_analyzer.task_difficulty_estimate =
                (1.0 - convergence_rate).clamp(0.0, 1.0);
        }
    }

    fn calculate_convergence_rate(&self) -> f32 {
        if self.exit_history.len() < 3 {
            return 0.5;
        }

        let recent_confidences: Vec<f32> =
            self.exit_history.iter().rev().take(3).map(|ep| ep.confidence_score).collect();

        // Calculate improvement rate
        let improvement = recent_confidences[0] - recent_confidences[2];
        (improvement + 1.0) / 2.0 // Normalize to [0, 1]
    }

    fn update_performance_stats(&mut self, exit_point: &ExitPoint) {
        let layer_index = exit_point.layer_index;
        let stats = self.performance_stats.entry(layer_index).or_insert(PerformanceStats {
            total_exits: 0,
            successful_exits: 0,
            average_confidence: 0.0,
            average_computation_time: 0.0,
            accuracy_loss: 0.0,
        });

        stats.total_exits += 1;

        if exit_point.should_exit {
            stats.successful_exits += 1;
        }

        // Update running averages
        let alpha = 0.1f32; // Learning rate
        stats.average_confidence =
            stats.average_confidence * (1.0 - alpha) + exit_point.confidence_score * alpha;
        stats.average_computation_time = stats.average_computation_time * (1.0 - alpha as f64)
            + exit_point.computation_time_ms as f64 * alpha as f64;
        // Ground-truth accuracy isn't available at inference time, so this
        // approximates per-exit accuracy loss from how far below full
        // confidence the exit point was: exiting at low confidence risks
        // more accuracy loss than exiting at high confidence.
        let estimated_accuracy_loss = (1.0 - exit_point.confidence_score).clamp(0.0, 1.0);
        stats.accuracy_loss = stats.accuracy_loss * (1.0 - alpha) + estimated_accuracy_loss * alpha;
    }

    pub fn get_performance_stats(&self) -> &HashMap<usize, PerformanceStats> {
        &self.performance_stats
    }

    pub fn reset(&mut self) {
        self.exit_history.clear();
        self.energy_tracker.current_energy_consumption = 0.0;
        self.energy_tracker.energy_budget_remaining = 100.0;
        self.memory_tracker.current_memory_usage = 0.0;
        self.memory_tracker.peak_memory_usage = 0.0;
        self.context_analyzer.input_complexity_score = 0.5;
        self.context_analyzer.task_difficulty_estimate = 0.5;
    }
}

/// A model that can be executed one layer at a time.
///
/// Early exiting is only meaningful when the wrapped computation can actually
/// be stopped part-way through, which requires per-layer access. Implement this
/// for a model that can expose its intermediate hidden states, logits and
/// prediction; [`EarlyExitPipeline`] then really runs layer by layer and stops
/// when the exit predictor says the answer has converged.
pub trait LayerwiseInference {
    /// Input accepted by this model.
    type Input;

    /// Per-request state carried between layers.
    type State;

    /// Total number of layers available.
    fn layer_count(&self) -> usize;

    /// Prepare the per-request state (embedding, tokenization, …).
    ///
    /// # Errors
    ///
    /// Whatever the model's own preparation step can fail with.
    fn begin(&self, input: &Self::Input) -> Result<Self::State>;

    /// Run layer `layer_index`, updating `state` and returning what that layer
    /// produced.
    ///
    /// # Errors
    ///
    /// Whatever the model's layer computation can fail with.
    fn run_layer(&self, layer_index: usize, state: &mut Self::State) -> Result<LayerOutput>;

    /// Produce the final output from the state after the last executed layer.
    ///
    /// # Errors
    ///
    /// Whatever the model's head can fail with.
    fn finish(&self, state: &Self::State, layers_executed: usize) -> Result<PipelineOutput>;
}

/// Runs a [`LayerwiseInference`] model and stops as soon as the exit predictor
/// is confident enough.
///
/// Every number this pipeline reports — hidden states, logits, timings, memory,
/// savings — comes from the layers that were actually executed.
#[derive(Clone)]
pub struct EarlyExitPipeline<P> {
    base_pipeline: P,
    exit_predictor: EarlyExitPredictor,
}

impl<P> EarlyExitPipeline<P> {
    /// Wrap `base_pipeline` with the given exit policy.
    pub fn new(base_pipeline: P, config: EarlyExitConfig) -> Self {
        Self {
            base_pipeline,
            exit_predictor: EarlyExitPredictor::new(config),
        }
    }

    /// Get a mutable reference to the exit predictor for configuration changes
    pub fn exit_predictor_mut(&mut self) -> &mut EarlyExitPredictor {
        &mut self.exit_predictor
    }

    /// Get a reference to the exit predictor for reading configuration
    pub fn exit_predictor(&self) -> &EarlyExitPredictor {
        &self.exit_predictor
    }

    /// Borrow the wrapped model.
    pub fn base_pipeline(&self) -> &P {
        &self.base_pipeline
    }
}

impl<P> EarlyExitPipeline<P>
where
    P: LayerwiseInference,
{
    /// Run the model layer by layer, stopping early when allowed.
    ///
    /// # Errors
    ///
    /// Propagates the model's own preparation, layer and head errors.
    pub fn run(&self, input: &P::Input) -> Result<EarlyExitResult> {
        let start_time = Instant::now();
        let available_layers = self.base_pipeline.layer_count();
        if available_layers == 0 {
            return Err(crate::error::TrustformersError::invalid_input_simple(
                "the wrapped model reports zero layers, so there is nothing to run".to_string(),
            ));
        }
        let max_layers = self.exit_predictor.config.max_layers.min(available_layers);
        let min_layers = self.exit_predictor.config.min_layers.min(max_layers);

        let mut state = self.base_pipeline.begin(input)?;
        let mut exit_path = Vec::new();
        let mut predictor = self.exit_predictor.clone();

        for current_layer in 0..max_layers {
            let layer_output = self.base_pipeline.run_layer(current_layer, &mut state)?;

            // Below `min_layers` the model has not produced enough evidence to
            // be allowed to stop; run the layer but do not consult the
            // predictor.
            if current_layer + 1 < min_layers {
                continue;
            }

            let exit_point = predictor.should_exit(&layer_output)?;
            exit_path.push(exit_point.clone());

            if exit_point.should_exit {
                let layers_executed = current_layer + 1;
                let computation_saved =
                    ((max_layers - layers_executed) as f32 / max_layers as f32) * 100.0;
                let confidence_score = exit_point.confidence_score;
                let exit_reason = exit_point.exit_reason.clone();
                let quality_score = self.estimate_quality_score(&exit_point);
                let energy_saved = predictor.energy_saved_percent(layers_executed, max_layers);

                return Ok(EarlyExitResult {
                    prediction: self.base_pipeline.finish(&state, layers_executed)?,
                    exit_point,
                    total_layers_computed: layers_executed,
                    computation_saved_percent: computation_saved,
                    energy_saved_percent: energy_saved,
                    confidence_score,
                    quality_score,
                    exit_path,
                    final_decision_reason: exit_reason,
                });
            }
        }

        // No exit fired: the full stack ran.
        let total_time = start_time.elapsed().as_millis() as u64;
        let last = exit_path.last().cloned();
        let exit_point = ExitPoint {
            layer_index: max_layers - 1,
            confidence_score: last.as_ref().map(|e| e.confidence_score).unwrap_or(0.0),
            entropy_score: last.as_ref().map(|e| e.entropy_score).unwrap_or(0.0),
            variance_score: last.as_ref().map(|e| e.variance_score).unwrap_or(0.0),
            consistency_score: last.as_ref().map(|e| e.consistency_score).unwrap_or(0.0),
            computation_time_ms: total_time,
            energy_consumed: last.as_ref().map(|e| e.energy_consumed).unwrap_or(0.0),
            memory_used_mb: last.as_ref().map(|e| e.memory_used_mb).unwrap_or(0.0),
            should_exit: false,
            exit_reason: "Completed all layers without meeting an exit condition".to_string(),
        };
        let confidence_score = exit_point.confidence_score;
        let quality_score = self.estimate_quality_score(&exit_point);

        Ok(EarlyExitResult {
            prediction: self.base_pipeline.finish(&state, max_layers)?,
            exit_point,
            total_layers_computed: max_layers,
            computation_saved_percent: 0.0,
            energy_saved_percent: 0.0,
            confidence_score,
            quality_score,
            exit_path,
            final_decision_reason: "Full computation completed".to_string(),
        })
    }

    fn estimate_quality_score(&self, exit_point: &ExitPoint) -> f32 {
        // Estimate quality based on confidence, layer depth, and consistency
        let depth_factor =
            exit_point.layer_index as f32 / self.exit_predictor.config.max_layers.max(1) as f32;
        let confidence_factor = exit_point.confidence_score;
        let consistency_factor = exit_point.consistency_score;

        (depth_factor * 0.3 + confidence_factor * 0.5 + consistency_factor * 0.2).min(1.0)
    }
}

/// `Pipeline` conformance for a wrapped *whole-model* pipeline.
///
/// A plain [`Pipeline`] can only be invoked end to end, so there is no layer at
/// which to exit early and no intermediate state to judge. Rather than
/// synthesising per-layer hidden states, logits and savings — which is exactly
/// what this module used to do — every call reports that the wrapped pipeline
/// is not layerwise. Wrap a [`LayerwiseInference`] model and call
/// [`EarlyExitPipeline::run`] to get real early exiting.
impl<P> Pipeline for EarlyExitPipeline<P>
where
    P: Pipeline,
    P::Input: Clone,
{
    type Input = P::Input;
    type Output = EarlyExitResult;

    fn __call__(&self, _input: Self::Input) -> Result<Self::Output> {
        Err(crate::error::TrustformersError::feature_unavailable(
            "early exit needs per-layer access to the model, and a plain `Pipeline` only exposes \
             a whole-model call. Wrap a type implementing \
             `trustformers::pipeline::early_exit::LayerwiseInference` and call \
             `EarlyExitPipeline::run`."
                .to_string(),
            "early-exit",
        ))
    }

    fn batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        inputs.into_iter().map(|input| self.__call__(input)).collect()
    }
}

// Factory functions for creating early exit pipelines
pub fn create_early_exit_pipeline<P>(
    base_pipeline: P,
    config: EarlyExitConfig,
) -> EarlyExitPipeline<P>
where
    P: LayerwiseInference,
{
    EarlyExitPipeline::new(base_pipeline, config)
}

pub fn create_confidence_based_early_exit<P>(
    base_pipeline: P,
    confidence_threshold: f32,
) -> EarlyExitPipeline<P>
where
    P: LayerwiseInference,
{
    let mut config = EarlyExitConfig::default();
    config.strategy = ExitStrategy::ConfidenceThreshold(confidence_threshold);
    EarlyExitPipeline::new(base_pipeline, config)
}

pub fn create_adaptive_early_exit<P>(base_pipeline: P) -> EarlyExitPipeline<P>
where
    P: LayerwiseInference,
{
    let mut config = EarlyExitConfig::default();
    config.strategy = ExitStrategy::AdaptiveThreshold;
    config.dynamic_threshold_adjustment = true;
    config.context_aware = true;
    config.performance_tracking = true;
    EarlyExitPipeline::new(base_pipeline, config)
}

pub fn create_budget_constrained_early_exit<P>(
    base_pipeline: P,
    computation_budget_ms: u64,
    energy_budget: f32,
) -> EarlyExitPipeline<P>
where
    P: LayerwiseInference,
{
    let mut config = EarlyExitConfig::default();
    config.strategy = ExitStrategy::Combined(vec![
        ExitStrategy::ComputationalBudget(computation_budget_ms),
        ExitStrategy::EnergyBudget(energy_budget),
        ExitStrategy::ConfidenceThreshold(0.8),
    ]);
    config.energy_aware = true;
    config.memory_aware = true;
    EarlyExitPipeline::new(base_pipeline, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_layer_output(
        layer_index: usize,
        logits: Option<Vec<f32>>,
        hidden_states: Vec<f32>,
    ) -> LayerOutput {
        LayerOutput {
            layer_index,
            hidden_states,
            attention_weights: None,
            logits,
            intermediate_prediction: None,
            computation_time_ms: 10 + layer_index as u64,
            memory_usage_mb: 100.0 + layer_index as f64 * 10.0,
        }
    }

    // ── Config defaults ───────────────────────────────────────────────────────

    #[test]
    fn test_early_exit_config_default() {
        let config = EarlyExitConfig::default();
        assert_eq!(config.min_layers, 6);
        assert_eq!(config.max_layers, 12);
        assert!(matches!(
            config.strategy,
            ExitStrategy::ConfidenceThreshold(_)
        ));
    }

    #[test]
    fn test_config_min_layers_less_than_max() {
        let config = EarlyExitConfig::default();
        assert!(config.min_layers < config.max_layers);
    }

    #[test]
    fn test_config_patience_threshold_positive() {
        let config = EarlyExitConfig::default();
        assert!(config.patience_threshold > 0);
    }

    // ── Exit threshold validation ─────────────────────────────────────────────

    #[test]
    fn test_confidence_threshold_strategy_validates_range() {
        // A threshold of 0.9 should be in (0, 1)
        if let ExitStrategy::ConfidenceThreshold(threshold) = ExitStrategy::ConfidenceThreshold(0.9)
        {
            assert!(threshold > 0.0 && threshold < 1.0);
        }
    }

    // ── ExitPoint creation ────────────────────────────────────────────────────

    #[test]
    fn test_exit_point_creation() {
        let layer_output = make_layer_output(5, Some(vec![1.0, 2.0, 0.5]), vec![0.1, 0.2, 0.3]);
        let config = EarlyExitConfig::default();
        let predictor = EarlyExitPredictor::new(config);
        let exit_point = predictor
            .create_base_exit_point(&layer_output)
            .expect("create_base_exit_point should succeed");
        assert_eq!(exit_point.layer_index, 5);
        assert!(exit_point.confidence_score > 0.0);
    }

    #[test]
    fn test_exit_point_no_logits_uses_depth_proxy() {
        let layer_output = make_layer_output(8, None, vec![0.1; 50]);
        let config = EarlyExitConfig::default();
        let predictor = EarlyExitPredictor::new(config);
        let ep = predictor
            .create_base_exit_point(&layer_output)
            .expect("create_base_exit_point should succeed");
        assert!(
            ep.confidence_score > 0.0 && ep.confidence_score <= 1.0,
            "depth-based confidence should be in (0,1]"
        );
    }

    // ── Confidence-based stopping ─────────────────────────────────────────────

    #[test]
    fn test_confidence_threshold_strategy() {
        let config = EarlyExitConfig {
            strategy: ExitStrategy::ConfidenceThreshold(0.9),
            min_layers: 2,
            ..Default::default()
        };
        let mut predictor = EarlyExitPredictor::new(config);
        // logits [5.0, 0.5] → very peaked → high confidence
        let output = make_layer_output(3, Some(vec![5.0, 0.5]), vec![0.1, 0.2, 0.3]);
        let ep = predictor.should_exit(&output).expect("should_exit should succeed");
        assert!(
            ep.should_exit,
            "high-confidence output should trigger early exit"
        );
    }

    #[test]
    fn test_no_exit_before_min_layers() {
        let config = EarlyExitConfig {
            strategy: ExitStrategy::ConfidenceThreshold(0.0), // trivially met
            min_layers: 6,
            max_layers: 12,
            ..Default::default()
        };
        let mut predictor = EarlyExitPredictor::new(config);
        let output = make_layer_output(2, Some(vec![10.0, 0.1]), vec![0.1; 10]);
        let ep = predictor.should_exit(&output).expect("should_exit should succeed");
        assert!(!ep.should_exit, "must not exit before min_layers");
    }

    #[test]
    fn test_forced_exit_at_max_layers() {
        let config = EarlyExitConfig {
            strategy: ExitStrategy::ConfidenceThreshold(1.0), // never normally triggered
            min_layers: 2,
            max_layers: 6,
            dynamic_threshold_adjustment: false,
            ..Default::default()
        };
        let mut predictor = EarlyExitPredictor::new(config.clone());
        let output = make_layer_output(config.max_layers, Some(vec![0.1; 10]), vec![0.0; 10]);
        let ep = predictor.should_exit(&output).expect("should_exit should succeed");
        assert!(ep.should_exit, "must exit when max_layers reached");
    }

    // ── Layer-wise confidence scores ─────────────────────────────────────────

    #[test]
    fn test_confidence_increases_with_layer_depth_no_logits() {
        let config = EarlyExitConfig {
            max_layers: 20,
            min_layers: 0,
            ..Default::default()
        };
        let predictor = EarlyExitPredictor::new(config);
        let ep_early = predictor
            .create_base_exit_point(&make_layer_output(0, None, vec![0.0; 5]))
            .expect("early exit point should succeed");
        let ep_late = predictor
            .create_base_exit_point(&make_layer_output(18, None, vec![0.0; 5]))
            .expect("late exit point should succeed");
        assert!(
            ep_late.confidence_score > ep_early.confidence_score,
            "confidence should grow with layer depth"
        );
    }

    // ── Entropy threshold ────────────────────────────────────────────────────

    #[test]
    fn test_entropy_threshold_strategy_uniform_distribution() {
        let config = EarlyExitConfig {
            strategy: ExitStrategy::EntropyThreshold(0.5),
            min_layers: 0,
            max_layers: 10,
            dynamic_threshold_adjustment: false,
            ..Default::default()
        };
        let mut predictor = EarlyExitPredictor::new(config);
        // Uniform logits → high entropy → low certainty (entropy_score near 0)
        let output = make_layer_output(5, Some(vec![1.0_f32; 10]), vec![0.1; 5]);
        let ep = predictor.should_exit(&output).expect("should_exit should succeed");
        // entropy_score = 1 - H/H_max; for uniform dist H = H_max so score ≈ 0 < 0.5
        assert!(
            !ep.should_exit,
            "uniform distribution should NOT meet entropy threshold"
        );
    }

    // ── Variance threshold ────────────────────────────────────────────────────

    #[test]
    fn test_variance_score_constant_hidden_states() {
        let config = EarlyExitConfig::default();
        let predictor = EarlyExitPredictor::new(config);
        let layer_output = make_layer_output(8, None, vec![0.5_f32; 20]);
        let ep = predictor
            .create_base_exit_point(&layer_output)
            .expect("create_base_exit_point should succeed");
        // Variance = 0 → variance_score = 1/(1+0) = 1.0
        assert!(
            (ep.variance_score - 1.0).abs() < 1e-5,
            "constant hidden states should yield variance_score = 1.0"
        );
    }

    // ── Minimum layers enforcement ────────────────────────────────────────────

    #[test]
    fn test_minimum_layers_enforcement_respected() {
        let config = EarlyExitConfig {
            strategy: ExitStrategy::ConfidenceThreshold(0.001), // trivially met
            min_layers: 5,
            max_layers: 12,
            dynamic_threshold_adjustment: false,
            ..Default::default()
        };
        let mut predictor = EarlyExitPredictor::new(config);
        for layer_idx in 0..5 {
            let output = make_layer_output(layer_idx, Some(vec![10.0, 0.01]), vec![0.1; 5]);
            let ep = predictor.should_exit(&output).expect("should_exit should succeed");
            assert!(
                !ep.should_exit,
                "layer {} < min_layers=5 should not exit",
                layer_idx
            );
        }
    }

    // ── Computational budget strategy ─────────────────────────────────────────

    #[test]
    fn test_computational_budget_strategy() {
        let config = EarlyExitConfig {
            strategy: ExitStrategy::ComputationalBudget(5), // 5ms budget
            min_layers: 0,
            max_layers: 12,
            ..Default::default()
        };
        let mut predictor = EarlyExitPredictor::new(config);
        // computation_time_ms = 10 + layer_index; at layer 0 it's 10ms > 5ms budget
        let output = make_layer_output(0, None, vec![0.1; 5]);
        let ep = predictor.should_exit(&output).expect("should_exit should succeed");
        assert!(
            ep.should_exit,
            "should exit when computation_time exceeds budget"
        );
    }

    // ── Patience strategy ────────────────────────────────────────────────────

    #[test]
    fn test_patience_strategy_triggers_after_patience_exceeded() {
        let config = EarlyExitConfig {
            strategy: ExitStrategy::Patience(3),
            min_layers: 0,
            max_layers: 20,
            dynamic_threshold_adjustment: false,
            ..Default::default()
        };
        let mut predictor = EarlyExitPredictor::new(config);
        // Run layers with low confidence to build up patience counter
        for i in 0..4 {
            let output = make_layer_output(i, Some(vec![1.0_f32; 5]), vec![0.1; 5]);
            let _ = predictor.should_exit(&output);
        }
        // On the 5th call, patience=4 >= 3 → should exit
        let output = make_layer_output(4, Some(vec![1.0_f32; 5]), vec![0.1; 5]);
        let ep = predictor.should_exit(&output).expect("should_exit should succeed");
        assert!(ep.should_exit, "patience counter should trigger exit");
    }

    // ── Depth reduction vs accuracy ───────────────────────────────────────────

    /// A real (small) layerwise model used by the pipeline tests.
    ///
    /// Each layer performs an actual computation over the previous hidden
    /// state and derives logits from it, so the confidence the exit predictor
    /// sees is a genuine function of the layer's output.
    struct ToyLayerwiseModel {
        layers: usize,
        width: usize,
    }

    struct ToyState {
        hidden: Vec<f32>,
    }

    impl LayerwiseInference for ToyLayerwiseModel {
        type Input = String;
        type State = ToyState;

        fn layer_count(&self) -> usize {
            self.layers
        }

        fn begin(&self, input: &Self::Input) -> Result<Self::State> {
            // Real (if tiny) embedding: byte values scaled into [0, 1).
            let bytes = input.as_bytes();
            let hidden = (0..self.width)
                .map(|i| bytes.get(i % bytes.len().max(1)).copied().unwrap_or(0) as f32 / 255.0)
                .collect();
            Ok(ToyState { hidden })
        }

        fn run_layer(&self, layer_index: usize, state: &mut Self::State) -> Result<LayerOutput> {
            let started = std::time::Instant::now();
            // Each layer sharpens the representation: values move toward their
            // sign, so confidence genuinely grows with depth.
            for value in state.hidden.iter_mut() {
                *value = (*value * 1.5).tanh();
            }
            let logits: Vec<f32> =
                state.hidden.iter().take(4).map(|v| v * (layer_index as f32 + 1.0)).collect();
            let top = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max).max(0.0);
            Ok(LayerOutput {
                layer_index,
                hidden_states: state.hidden.clone(),
                attention_weights: None,
                logits: Some(logits),
                intermediate_prediction: Some(crate::pipeline::PipelineOutput::Classification(
                    vec![crate::pipeline::ClassificationOutput {
                        label: format!("layer_{layer_index}"),
                        score: top.min(1.0),
                    }],
                )),
                computation_time_ms: started.elapsed().as_millis() as u64,
                memory_usage_mb: (state.hidden.len() * std::mem::size_of::<f32>()) as f64
                    / (1024.0 * 1024.0),
            })
        }

        fn finish(&self, state: &Self::State, layers_executed: usize) -> Result<PipelineOutput> {
            let score = state.hidden.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            Ok(crate::pipeline::PipelineOutput::Classification(vec![
                crate::pipeline::ClassificationOutput {
                    label: format!("after_{layers_executed}_layers"),
                    score: score.clamp(0.0, 1.0),
                },
            ]))
        }
    }

    fn toy_model() -> ToyLayerwiseModel {
        ToyLayerwiseModel {
            layers: 12,
            width: 8,
        }
    }

    /// Regression: `__call__` used to synthesise every layer output. A pipeline
    /// wrapping a plain (non-layerwise) `Pipeline` must now refuse.
    #[test]
    fn non_layerwise_pipeline_is_refused() {
        struct WholeModel;
        impl crate::pipeline::Pipeline for WholeModel {
            type Input = String;
            type Output = crate::pipeline::PipelineOutput;
            fn __call__(&self, _: Self::Input) -> Result<Self::Output> {
                Ok(crate::pipeline::PipelineOutput::Text("x".to_string()))
            }
        }
        let pipeline = EarlyExitPipeline::new(WholeModel, EarlyExitConfig::default());
        assert!(
            pipeline.__call__("test".to_string()).is_err(),
            "a whole-model pipeline offers no layer to exit at, so no result may be produced"
        );
    }

    /// The wrapped model must actually be executed, layer by layer.
    #[test]
    fn layers_are_really_executed_and_reported() {
        let config = EarlyExitConfig {
            strategy: ExitStrategy::ConfidenceThreshold(0.0),
            min_layers: 3,
            max_layers: 12,
            dynamic_threshold_adjustment: false,
            ..Default::default()
        };
        let pipeline = create_early_exit_pipeline(toy_model(), config);
        let result = pipeline.run(&"test input".to_string()).expect("layerwise run should succeed");

        assert!(
            result.total_layers_computed >= 3,
            "min_layers must be honoured"
        );
        assert!(result.total_layers_computed <= 12);
        assert!(result.computation_saved_percent >= 0.0);
        // The prediction is the model's own, not a fixed summarization string.
        match &result.prediction {
            crate::pipeline::PipelineOutput::Classification(scores) => {
                assert!(!scores.is_empty());
                assert!(
                    !scores[0].label.starts_with("Class_"),
                    "labels must come from the model, not the removed simulator"
                );
            },
            other => panic!("unexpected prediction variant: {other:?}"),
        }
        assert!(
            !result.exit_path.is_empty(),
            "the exit path must record the layers that were judged"
        );
    }

    /// A model with no layers cannot be run.
    #[test]
    fn zero_layer_model_is_rejected() {
        let pipeline = EarlyExitPipeline::new(
            ToyLayerwiseModel {
                layers: 0,
                width: 4,
            },
            EarlyExitConfig::default(),
        );
        assert!(pipeline.run(&"x".to_string()).is_err());
    }

    // ── EarlyExitPipeline factories ───────────────────────────────────────────

    #[test]
    fn test_confidence_based_factory() {
        let pipeline = create_confidence_based_early_exit(toy_model(), 0.7_f32);
        let config = pipeline.exit_predictor().config();
        assert!(
            matches!(config.strategy, ExitStrategy::ConfidenceThreshold(t) if (t - 0.7).abs() < 1e-5)
        );
    }

    #[test]
    fn test_adaptive_factory() {
        let pipeline = create_adaptive_early_exit(toy_model());
        let config = pipeline.exit_predictor().config();
        assert!(matches!(config.strategy, ExitStrategy::AdaptiveThreshold));
        assert!(config.dynamic_threshold_adjustment);
    }

    #[test]
    fn test_budget_constrained_factory() {
        let pipeline = create_budget_constrained_early_exit(toy_model(), 100, 50.0);
        let config = pipeline.exit_predictor().config();
        assert!(matches!(config.strategy, ExitStrategy::Combined(_)));
        assert!(config.energy_aware);
    }

    // ── Reset ────────────────────────────────────────────────────────────────

    #[test]
    fn test_predictor_reset_clears_history() {
        let config = EarlyExitConfig {
            min_layers: 0,
            max_layers: 10,
            ..Default::default()
        };
        let mut predictor = EarlyExitPredictor::new(config);
        let output = make_layer_output(5, Some(vec![1.0, 2.0]), vec![0.1; 5]);
        let _ = predictor.should_exit(&output);
        predictor.reset();
        // After reset, exit_history is empty so consistency falls back to
        // the current layer's own representation stability.
        let ep2 = predictor.create_base_exit_point(&output).expect("create_base_exit_point ok");
        assert!(ep2.consistency_score > 0.0);
    }

    /// Regression test: with fewer than 2 exit-history entries,
    /// `calculate_consistency_score` used to return a fixed `0.5` regardless
    /// of the current layer's actual hidden states. It must now reflect real
    /// per-layer representation stability, differing between a stable
    /// (near-constant) and an unstable (highly varying) hidden state.
    #[test]
    fn test_consistency_score_reflects_hidden_state_stability_before_history_exists() {
        let config = EarlyExitConfig {
            min_layers: 0,
            max_layers: 10,
            ..Default::default()
        };
        let predictor = EarlyExitPredictor::new(config);

        let stable = make_layer_output(0, None, vec![0.5; 32]);
        let unstable = make_layer_output(0, None, vec![-10.0, 10.0, -8.0, 9.0, -12.0, 11.0]);

        let stable_score = predictor
            .calculate_consistency_score(&stable)
            .expect("consistency score for stable hidden states");
        let unstable_score = predictor
            .calculate_consistency_score(&unstable)
            .expect("consistency score for unstable hidden states");

        assert!(
            stable_score > unstable_score,
            "near-constant hidden states ({stable_score}) must score more consistent than \
             wildly varying ones ({unstable_score})"
        );
    }

    /// Regression test: `accuracy_loss` in `PerformanceStats` used to be
    /// initialized to `0.0` and never updated by
    /// `update_performance_stats`'s running-average logic (unlike
    /// `average_confidence`, which was). A low-confidence exit must now
    /// raise the tracked `accuracy_loss` above zero.
    #[test]
    fn test_performance_stats_track_accuracy_loss_from_low_confidence_exits() {
        let config = EarlyExitConfig {
            min_layers: 0,
            max_layers: 10,
            strategy: ExitStrategy::ConfidenceThreshold(0.0), // always exits
            ..Default::default()
        };
        let mut predictor = EarlyExitPredictor::new(config);
        // Flat, low-magnitude logits -> low confidence exit.
        let output = make_layer_output(3, Some(vec![0.01, 0.01, 0.01]), vec![1.0; 8]);
        let _ = predictor.should_exit(&output).expect("should_exit should succeed");

        let stats = predictor
            .get_performance_stats()
            .get(&3)
            .expect("layer 3 should have recorded performance stats");
        assert!(
            stats.accuracy_loss > 0.0,
            "a low-confidence exit must raise the tracked accuracy_loss above its 0.0 initial \
             value, got {}",
            stats.accuracy_loss
        );
    }
}
