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

#[derive(Debug, Clone)]
struct PerformanceStats {
    total_exits: u64,
    successful_exits: u64,
    average_confidence: f32,
    average_computation_time: f64,
    accuracy_loss: f32,
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
        if self.exit_history.len() < 2 {
            return Ok(0.5); // Not enough history
        }

        // Compare with previous layer predictions
        let recent_confidences: Vec<f32> =
            self.exit_history.iter().rev().take(3).map(|ep| ep.confidence_score).collect();

        if recent_confidences.len() < 2 {
            return Ok(0.5);
        }

        // Calculate consistency as inverse of variance in recent confidences
        let mean = recent_confidences.iter().sum::<f32>() / recent_confidences.len() as f32;
        let variance = recent_confidences.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
            / recent_confidences.len() as f32;

        Ok(1.0 / (1.0 + variance))
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

#[derive(Clone)]
pub struct EarlyExitPipeline<P> {
    base_pipeline: P,
    exit_predictor: EarlyExitPredictor,
    // Skip layer_processors for now to make it Clone
    // layer_processors: Vec<Box<dyn Fn(&LayerOutput) -> Result<LayerOutput>>>,
}

impl<P> EarlyExitPipeline<P>
where
    P: Pipeline,
{
    pub fn new(base_pipeline: P, config: EarlyExitConfig) -> Self {
        Self {
            base_pipeline,
            exit_predictor: EarlyExitPredictor::new(config),
            // layer_processors: Vec::new(),
        }
    }

    // Commented out to make the struct Clone-compatible
    // pub fn add_layer_processor<F>(&mut self, processor: F)
    // where
    //     F: Fn(&LayerOutput) -> Result<LayerOutput> + 'static,
    // {
    //     self.layer_processors.push(Box::new(processor));
    // }

    /// Get a mutable reference to the exit predictor for configuration changes
    pub fn exit_predictor_mut(&mut self) -> &mut EarlyExitPredictor {
        &mut self.exit_predictor
    }

    /// Get a reference to the exit predictor for reading configuration
    pub fn exit_predictor(&self) -> &EarlyExitPredictor {
        &self.exit_predictor
    }

    fn simulate_layer_by_layer_processing(&self, input: &P::Input) -> Result<EarlyExitResult> {
        let start_time = Instant::now();
        let mut exit_path = Vec::new();
        let mut current_layer = 0;
        let max_layers = self.exit_predictor.config.max_layers;

        // In a real implementation, this would process the model layer by layer
        // For now, we simulate the process
        while current_layer < max_layers {
            let layer_start = Instant::now();

            // Simulate layer computation
            let hidden_states = self.simulate_layer_computation(current_layer);
            let layer_output = LayerOutput {
                layer_index: current_layer,
                hidden_states: hidden_states.clone(),
                attention_weights: Some(vec![0.5; 64]), // Simulated
                logits: if current_layer >= self.exit_predictor.config.min_layers {
                    Some(self.simulate_logits(&hidden_states))
                } else {
                    None
                },
                intermediate_prediction: if current_layer >= self.exit_predictor.config.min_layers {
                    Some(self.simulate_intermediate_prediction(&hidden_states)?)
                } else {
                    None
                },
                computation_time_ms: layer_start.elapsed().as_millis() as u64,
                memory_usage_mb: 100.0 + current_layer as f64 * 10.0, // Simulated
            };

            // Check for early exit - need mutable access to predictor
            // For now, we'll create a temporary predictor to avoid the mutability issue
            let mut temp_predictor = self.exit_predictor.clone();
            let exit_point = temp_predictor.should_exit(&layer_output)?;
            exit_path.push(exit_point.clone());

            if exit_point.should_exit {
                let total_time = start_time.elapsed().as_millis() as u64;
                let computation_saved =
                    ((max_layers - current_layer - 1) as f32 / max_layers as f32) * 100.0;
                let energy_saved = computation_saved * 0.8; // Approximate

                // Extract fields before moving exit_point
                let confidence_score = exit_point.confidence_score;
                let exit_reason = exit_point.exit_reason.clone();
                let quality_score = self.estimate_quality_score(&exit_point);

                return Ok(EarlyExitResult {
                    prediction: layer_output.intermediate_prediction.unwrap_or_else(|| {
                        // Create a simple fallback prediction that matches PipelineOutput
                        PipelineOutput::Summarization(
                            "Fallback prediction due to early exit".to_string(),
                        )
                    }),
                    exit_point,
                    total_layers_computed: current_layer + 1,
                    computation_saved_percent: computation_saved,
                    energy_saved_percent: energy_saved,
                    confidence_score,
                    quality_score,
                    exit_path,
                    final_decision_reason: exit_reason,
                });
            }

            current_layer += 1;
        }

        // If we reach here, we processed all layers
        // For the final prediction, we'll create a default since we can't clone the input
        // In a real implementation, this would be handled properly
        let fallback_output =
            PipelineOutput::Summarization("Full computation completed".to_string());
        let final_prediction = fallback_output;
        let total_time = start_time.elapsed().as_millis() as u64;

        Ok(EarlyExitResult {
            prediction: match final_prediction {
                p => {
                    // Try to convert the pipeline output to PipelineOutput
                    // For now, just create a default output
                    PipelineOutput::Summarization("Full pipeline prediction completed".to_string())
                },
            },
            exit_point: ExitPoint {
                layer_index: max_layers - 1,
                confidence_score: 1.0,
                entropy_score: 1.0,
                variance_score: 1.0,
                consistency_score: 1.0,
                computation_time_ms: total_time,
                energy_consumed: 100.0,
                memory_used_mb: 100.0 + max_layers as f64 * 10.0,
                should_exit: true,
                exit_reason: "Completed all layers".to_string(),
            },
            total_layers_computed: max_layers,
            computation_saved_percent: 0.0,
            energy_saved_percent: 0.0,
            confidence_score: 1.0,
            quality_score: 1.0,
            exit_path,
            final_decision_reason: "Full computation completed".to_string(),
        })
    }

    fn simulate_layer_computation(&self, layer_index: usize) -> Vec<f32> {
        // Simulate hidden states (in reality, this would come from the actual model)
        let size = 768; // Typical hidden size
        let mut hidden_states = Vec::with_capacity(size);

        for i in 0..size {
            let value = (layer_index as f32 * 0.1 + i as f32 * 0.001).sin() * 0.5;
            hidden_states.push(value);
        }

        hidden_states
    }

    fn simulate_logits(&self, hidden_states: &[f32]) -> Vec<f32> {
        // Simulate logits based on hidden states
        let num_classes = 10;
        let mut logits = Vec::with_capacity(num_classes);

        for i in 0..num_classes {
            let logit = hidden_states[i % hidden_states.len()] * 2.0 + (i as f32 * 0.1);
            logits.push(logit);
        }

        logits
    }

    fn simulate_intermediate_prediction(&self, hidden_states: &[f32]) -> Result<PipelineOutput> {
        // Create a simple classification prediction
        let num_classes = 3;
        let mut class_scores = Vec::new();

        for i in 0..num_classes {
            let score = hidden_states[i % hidden_states.len()].abs().min(1.0);
            class_scores.push(crate::pipeline::ClassificationOutput {
                label: format!("Class_{}", i),
                score,
            });
        }

        // Sort by score descending
        class_scores
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(PipelineOutput::Classification(class_scores))
    }

    fn estimate_quality_score(&self, exit_point: &ExitPoint) -> f32 {
        // Estimate quality based on confidence, layer depth, and consistency
        let depth_factor =
            exit_point.layer_index as f32 / self.exit_predictor.config.max_layers as f32;
        let confidence_factor = exit_point.confidence_score;
        let consistency_factor = exit_point.consistency_score;

        (depth_factor * 0.3 + confidence_factor * 0.5 + consistency_factor * 0.2).min(1.0)
    }
}

impl<P> Pipeline for EarlyExitPipeline<P>
where
    P: Pipeline,
    P::Input: Clone,
{
    type Input = P::Input;
    type Output = EarlyExitResult;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        self.simulate_layer_by_layer_processing(&input)
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
    P: Pipeline,
{
    EarlyExitPipeline::new(base_pipeline, config)
}

pub fn create_confidence_based_early_exit<P>(
    base_pipeline: P,
    confidence_threshold: f32,
) -> EarlyExitPipeline<P>
where
    P: Pipeline,
{
    let mut config = EarlyExitConfig::default();
    config.strategy = ExitStrategy::ConfidenceThreshold(confidence_threshold);
    EarlyExitPipeline::new(base_pipeline, config)
}

pub fn create_adaptive_early_exit<P>(base_pipeline: P) -> EarlyExitPipeline<P>
where
    P: Pipeline,
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
    P: Pipeline,
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
    fn test_exit_point_creation() {
        let layer_output = LayerOutput {
            layer_index: 5,
            hidden_states: vec![0.1, 0.2, 0.3],
            attention_weights: None,
            logits: Some(vec![1.0, 2.0, 0.5]),
            intermediate_prediction: None,
            computation_time_ms: 10,
            memory_usage_mb: 50.0,
        };

        let config = EarlyExitConfig::default();
        let predictor = EarlyExitPredictor::new(config);
        let exit_point = predictor.create_base_exit_point(&layer_output).unwrap();

        assert_eq!(exit_point.layer_index, 5);
        assert!(exit_point.confidence_score > 0.0);
    }

    #[test]
    fn test_confidence_threshold_strategy() {
        let mut config = EarlyExitConfig::default();
        config.strategy = ExitStrategy::ConfidenceThreshold(0.9);
        config.min_layers = 2;

        let mut predictor = EarlyExitPredictor::new(config);

        let high_confidence_output = LayerOutput {
            layer_index: 3,
            hidden_states: vec![0.1, 0.2, 0.3],
            attention_weights: None,
            logits: Some(vec![5.0, 1.0, 0.5]), // High confidence
            intermediate_prediction: None,
            computation_time_ms: 10,
            memory_usage_mb: 50.0,
        };

        let exit_point = predictor.should_exit(&high_confidence_output).unwrap();
        assert!(exit_point.should_exit);
    }
}
