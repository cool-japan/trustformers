//! Adaptive (input-dependent) computation.
//!
//! Two layers:
//!
//! * this module: per-layer early-exit strategies driven by measured
//!   confidence / uncertainty / entropy, plus [`ComputationBudget`] accounting;
//! * [`dynamic`]: runtime topology modification and multi-path execution with
//!   voting.
//!
//! Every metric here is computed from the tensors the caller passes in. Where a
//! measurement can only come from the caller (a layer's wall-clock time, the
//! output of an actual layer stack) it is a parameter or a trait, never a
//! constant.

use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveComputationConfig {
    pub max_layers: usize,
    pub min_layers: usize,
    pub halt_threshold: f32,
    pub time_penalty: f32,
    pub early_exit_threshold: f32,
    pub complexity_estimation_method: ComplexityEstimationMethod,
    pub dynamic_depth_strategy: DynamicDepthStrategy,
}

impl Default for AdaptiveComputationConfig {
    fn default() -> Self {
        Self {
            max_layers: 12,
            min_layers: 2,
            halt_threshold: 0.99,
            time_penalty: 0.01,
            early_exit_threshold: 0.95,
            complexity_estimation_method: ComplexityEstimationMethod::EntropyBased,
            dynamic_depth_strategy: DynamicDepthStrategy::ConfidenceBased,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityEstimationMethod {
    EntropyBased,
    AttentionBased,
    GradientNorm,
    LearningCurve,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DynamicDepthStrategy {
    ConfidenceBased,
    UncertaintyBased,
    ResourceConstrained,
    LatencyOptimized,
    AccuracyOptimized,
}

#[derive(Debug, Clone)]
pub struct ComputationBudget {
    pub max_flops: u64,
    pub max_memory_mb: u32,
    pub max_latency_ms: u32,
    pub remaining_flops: u64,
    pub remaining_memory_mb: u32,
    pub remaining_time_ms: u32,
}

impl ComputationBudget {
    pub fn new(max_flops: u64, max_memory_mb: u32, max_latency_ms: u32) -> Self {
        Self {
            max_flops,
            max_memory_mb,
            max_latency_ms,
            remaining_flops: max_flops,
            remaining_memory_mb: max_memory_mb,
            remaining_time_ms: max_latency_ms,
        }
    }

    pub fn can_afford(&self, flops: u64, memory_mb: u32, time_ms: u32) -> bool {
        self.remaining_flops >= flops
            && self.remaining_memory_mb >= memory_mb
            && self.remaining_time_ms >= time_ms
    }

    pub fn consume(&mut self, flops: u64, memory_mb: u32, time_ms: u32) {
        self.remaining_flops = self.remaining_flops.saturating_sub(flops);
        self.remaining_memory_mb = self.remaining_memory_mb.saturating_sub(memory_mb);
        self.remaining_time_ms = self.remaining_time_ms.saturating_sub(time_ms);
    }
}

#[derive(Debug, Clone)]
pub struct LayerMetrics {
    pub layer_id: usize,
    pub flops_estimate: u64,
    pub memory_usage_mb: u32,
    pub execution_time_ms: u32,
    pub confidence_score: f32,
    pub uncertainty_score: f32,
    pub output_entropy: f32,
}

pub trait AdaptiveComputationStrategy {
    fn should_continue(
        &self,
        layer_id: usize,
        metrics: &LayerMetrics,
        budget: &ComputationBudget,
        config: &AdaptiveComputationConfig,
    ) -> bool;

    fn estimate_remaining_cost(
        &self,
        current_layer: usize,
        total_layers: usize,
        current_metrics: &LayerMetrics,
    ) -> (u64, u32, u32); // (flops, memory_mb, time_ms)

    fn adjust_computation_path(
        &self,
        input_complexity: f32,
        available_budget: &ComputationBudget,
        config: &AdaptiveComputationConfig,
    ) -> ComputationPath;
}

#[derive(Debug, Clone)]
pub struct ComputationPath {
    pub layers_to_execute: Vec<usize>,
    pub skip_patterns: Vec<LayerSkipPattern>,
    pub early_exit_points: Vec<usize>,
    pub resource_allocation: ResourceAllocation,
}

#[derive(Debug, Clone)]
pub enum LayerSkipPattern {
    Skip,
    Approximate,
    Cached,
    Pruned,
}

#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub memory_per_layer: HashMap<usize, u32>,
    pub compute_intensity: HashMap<usize, f32>,
    pub parallelism_factor: HashMap<usize, u32>,
}

pub struct ConfidenceBasedStrategy {
    confidence_history: Arc<RwLock<Vec<f32>>>,
    #[allow(dead_code)]
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
}

impl Default for ConfidenceBasedStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceBasedStrategy {
    pub fn new() -> Self {
        Self {
            confidence_history: Arc::new(RwLock::new(Vec::new())),
            performance_tracker: Arc::new(RwLock::new(PerformanceTracker::new())),
        }
    }
}

impl AdaptiveComputationStrategy for ConfidenceBasedStrategy {
    fn should_continue(
        &self,
        layer_id: usize,
        metrics: &LayerMetrics,
        budget: &ComputationBudget,
        config: &AdaptiveComputationConfig,
    ) -> bool {
        // Early exit if confidence is high enough
        if metrics.confidence_score >= config.early_exit_threshold {
            return false;
        }

        // Must continue if below minimum layers
        if layer_id < config.min_layers {
            return true;
        }

        // Stop if at maximum layers
        if layer_id >= config.max_layers {
            return false;
        }

        // Check resource constraints
        let (est_flops, est_memory, est_time) =
            self.estimate_remaining_cost(layer_id, config.max_layers, metrics);

        if !budget.can_afford(est_flops, est_memory, est_time) {
            return false;
        }

        // Adaptive halting criterion based on confidence growth rate
        let mut confidence_history =
            self.confidence_history.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        confidence_history.push(metrics.confidence_score);

        if confidence_history.len() >= 3 {
            let recent_growth = confidence_history[confidence_history.len() - 1]
                - confidence_history[confidence_history.len() - 3];

            if recent_growth < 0.01 && metrics.confidence_score > config.halt_threshold {
                return false;
            }
        }

        true
    }

    fn estimate_remaining_cost(
        &self,
        current_layer: usize,
        total_layers: usize,
        current_metrics: &LayerMetrics,
    ) -> (u64, u32, u32) {
        let remaining_layers = total_layers.saturating_sub(current_layer);

        // Estimate based on current layer metrics
        let avg_flops_per_layer = current_metrics.flops_estimate;
        let avg_memory_per_layer = current_metrics.memory_usage_mb;
        let avg_time_per_layer = current_metrics.execution_time_ms;

        (
            avg_flops_per_layer * remaining_layers as u64,
            avg_memory_per_layer * remaining_layers as u32,
            avg_time_per_layer * remaining_layers as u32,
        )
    }

    fn adjust_computation_path(
        &self,
        input_complexity: f32,
        available_budget: &ComputationBudget,
        config: &AdaptiveComputationConfig,
    ) -> ComputationPath {
        let mut layers_to_execute = Vec::new();
        let mut skip_patterns = HashMap::new();
        let mut resource_allocation = ResourceAllocation {
            memory_per_layer: HashMap::new(),
            compute_intensity: HashMap::new(),
            parallelism_factor: HashMap::new(),
        };

        // Determine layers based on input complexity
        let estimated_layers = if input_complexity < 0.3 {
            config.min_layers
        } else if input_complexity < 0.7 {
            (config.min_layers + config.max_layers) / 2
        } else {
            config.max_layers
        };

        for layer_id in 0..estimated_layers {
            layers_to_execute.push(layer_id);

            // Allocate resources based on layer position and input complexity
            let layer_importance = 1.0 - (layer_id as f32 / estimated_layers as f32);
            let complexity_factor = input_complexity * layer_importance;

            resource_allocation.memory_per_layer.insert(
                layer_id,
                (available_budget.max_memory_mb as f32 / estimated_layers as f32
                    * complexity_factor) as u32,
            );

            resource_allocation.compute_intensity.insert(layer_id, complexity_factor);
            resource_allocation.parallelism_factor.insert(layer_id, 1);

            // Determine skip patterns for less important layers
            if complexity_factor < 0.3 && layer_id > config.min_layers {
                skip_patterns.insert(layer_id, LayerSkipPattern::Approximate);
            }
        }

        ComputationPath {
            layers_to_execute,
            skip_patterns: skip_patterns.into_values().collect(),
            early_exit_points: vec![estimated_layers / 2, estimated_layers * 3 / 4],
            resource_allocation,
        }
    }
}

pub struct UncertaintyBasedStrategy {
    #[allow(dead_code)]
    uncertainty_tracker: Arc<RwLock<Vec<f32>>>,
}

impl Default for UncertaintyBasedStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl UncertaintyBasedStrategy {
    pub fn new() -> Self {
        Self {
            uncertainty_tracker: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl AdaptiveComputationStrategy for UncertaintyBasedStrategy {
    fn should_continue(
        &self,
        layer_id: usize,
        metrics: &LayerMetrics,
        budget: &ComputationBudget,
        config: &AdaptiveComputationConfig,
    ) -> bool {
        // Continue if uncertainty is still high
        if metrics.uncertainty_score > 0.1 && layer_id < config.max_layers {
            let (est_flops, est_memory, est_time) =
                self.estimate_remaining_cost(layer_id, config.max_layers, metrics);
            return budget.can_afford(est_flops, est_memory, est_time);
        }

        // Must continue if below minimum
        layer_id < config.min_layers
    }

    fn estimate_remaining_cost(
        &self,
        current_layer: usize,
        total_layers: usize,
        current_metrics: &LayerMetrics,
    ) -> (u64, u32, u32) {
        let remaining_layers = total_layers.saturating_sub(current_layer);

        // Uncertainty-based scaling: higher uncertainty means more computation needed
        let uncertainty_factor = current_metrics.uncertainty_score.max(0.1);

        (
            (current_metrics.flops_estimate as f32 * remaining_layers as f32 * uncertainty_factor)
                as u64,
            (current_metrics.memory_usage_mb as f32 * remaining_layers as f32 * uncertainty_factor)
                as u32,
            (current_metrics.execution_time_ms as f32
                * remaining_layers as f32
                * uncertainty_factor) as u32,
        )
    }

    fn adjust_computation_path(
        &self,
        input_complexity: f32,
        available_budget: &ComputationBudget,
        config: &AdaptiveComputationConfig,
    ) -> ComputationPath {
        // Similar to confidence-based but focuses on uncertainty reduction
        let estimated_layers = ((input_complexity * config.max_layers as f32) as usize)
            .max(config.min_layers)
            .min(config.max_layers);

        let layers_to_execute: Vec<usize> = (0..estimated_layers).collect();
        let mut resource_allocation = ResourceAllocation {
            memory_per_layer: HashMap::new(),
            compute_intensity: HashMap::new(),
            parallelism_factor: HashMap::new(),
        };

        // Allocate more resources to layers that typically reduce uncertainty more
        for layer_id in &layers_to_execute {
            let uncertainty_reduction_factor = 1.0 + (*layer_id as f32 / estimated_layers as f32);

            resource_allocation.memory_per_layer.insert(
                *layer_id,
                (available_budget.max_memory_mb as f32 / estimated_layers as f32
                    * uncertainty_reduction_factor) as u32,
            );

            resource_allocation
                .compute_intensity
                .insert(*layer_id, uncertainty_reduction_factor);
            resource_allocation.parallelism_factor.insert(*layer_id, 1);
        }

        ComputationPath {
            layers_to_execute,
            skip_patterns: Vec::new(),
            early_exit_points: vec![estimated_layers / 3, estimated_layers * 2 / 3],
            resource_allocation,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    layer_execution_times: HashMap<usize, Vec<u32>>,
    accuracy_by_layers: HashMap<usize, Vec<f32>>,
    resource_usage_history: Vec<(u64, u32, u32)>, // (flops, memory, time)
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            layer_execution_times: HashMap::new(),
            accuracy_by_layers: HashMap::new(),
            resource_usage_history: Vec::new(),
        }
    }

    pub fn record_layer_execution(&mut self, layer_id: usize, execution_time_ms: u32) {
        self.layer_execution_times.entry(layer_id).or_default().push(execution_time_ms);
    }

    pub fn record_accuracy(&mut self, layers_used: usize, accuracy: f32) {
        self.accuracy_by_layers.entry(layers_used).or_default().push(accuracy);
    }

    pub fn record_resource_usage(&mut self, flops: u64, memory_mb: u32, time_ms: u32) {
        self.resource_usage_history.push((flops, memory_mb, time_ms));
    }

    pub fn get_average_execution_time(&self, layer_id: usize) -> Option<f32> {
        self.layer_execution_times
            .get(&layer_id)
            .map(|times| times.iter().sum::<u32>() as f32 / times.len() as f32)
    }

    pub fn get_accuracy_trend(&self, layers_used: usize) -> Option<f32> {
        self.accuracy_by_layers.get(&layers_used).and_then(|accuracies| {
            if accuracies.is_empty() {
                None
            } else {
                Some(accuracies.iter().sum::<f32>() / accuracies.len() as f32)
            }
        })
    }
}

pub struct AdaptiveComputationManager {
    config: AdaptiveComputationConfig,
    strategy: Box<dyn AdaptiveComputationStrategy + Send + Sync>,
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    complexity_estimator: Box<dyn ComplexityEstimator + Send + Sync>,
}

impl AdaptiveComputationManager {
    pub fn new(
        config: AdaptiveComputationConfig,
        strategy: Box<dyn AdaptiveComputationStrategy + Send + Sync>,
        complexity_estimator: Box<dyn ComplexityEstimator + Send + Sync>,
    ) -> Self {
        Self {
            config,
            strategy,
            performance_tracker: Arc::new(RwLock::new(PerformanceTracker::new())),
            complexity_estimator,
        }
    }

    pub fn plan_computation(
        &self,
        input: &Tensor,
        budget: &ComputationBudget,
    ) -> Result<ComputationPath, Box<dyn std::error::Error>> {
        // Estimate input complexity
        let input_complexity = self.complexity_estimator.estimate_complexity(input)?;

        // Generate computation path
        let path = self.strategy.adjust_computation_path(input_complexity, budget, &self.config);

        Ok(path)
    }

    /// Decide whether to keep computing after a layer.
    ///
    /// `layer_duration` must be the time the layer actually took; it is what
    /// [`PerformanceTracker`] records. Pass `Duration::ZERO` only if the caller
    /// genuinely did not time the layer — the tracker then has no timing to
    /// report rather than a made-up one.
    pub fn should_continue_layer(
        &self,
        layer_id: usize,
        layer_output: &Tensor,
        budget: &ComputationBudget,
        layer_duration: Duration,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Calculate layer metrics
        let metrics = self.calculate_layer_metrics(layer_id, layer_output, layer_duration)?;

        // Use strategy to decide
        let should_continue =
            self.strategy.should_continue(layer_id, &metrics, budget, &self.config);

        // Update performance tracking
        {
            let mut tracker = self
                .performance_tracker
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            tracker.record_layer_execution(layer_id, metrics.execution_time_ms);
        }

        Ok(should_continue)
    }

    fn calculate_layer_metrics(
        &self,
        layer_id: usize,
        layer_output: &Tensor,
        layer_duration: Duration,
    ) -> Result<LayerMetrics, Box<dyn std::error::Error>> {
        // Estimate computational cost
        let flops_estimate = self.estimate_flops(layer_output)?;
        let memory_usage_mb = self.estimate_memory_usage(layer_output)?;

        // Calculate confidence and uncertainty
        let confidence_score = self.calculate_confidence(layer_output)?;
        let uncertainty_score = 1.0 - confidence_score;

        // Calculate output entropy
        let output_entropy = self.calculate_entropy(layer_output)?;

        Ok(LayerMetrics {
            layer_id,
            flops_estimate,
            memory_usage_mb,
            // Measured by the caller and threaded through; never assumed.
            execution_time_ms: u32::try_from(layer_duration.as_millis()).unwrap_or(u32::MAX),
            confidence_score,
            uncertainty_score,
            output_entropy,
        })
    }

    fn estimate_flops(&self, tensor: &Tensor) -> Result<u64, Box<dyn std::error::Error>> {
        // Rough FLOPS estimation based on tensor size
        let size: u64 = tensor.shape().iter().map(|&x| x as u64).product();
        Ok(size * 2) // Assume roughly 2 operations per element
    }

    fn estimate_memory_usage(&self, tensor: &Tensor) -> Result<u32, Box<dyn std::error::Error>> {
        let size: u64 = tensor.shape().iter().map(|&x| x as u64).product();
        Ok((size * 4 / (1024 * 1024)) as u32) // 4 bytes per f32, convert to MB
    }

    fn calculate_confidence(&self, tensor: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
        // Simple confidence calculation based on output distribution
        let max_value = tensor.max_value()?;
        let mean_value = tensor.mean()?;

        // Extract scalar values from single-element tensors
        let max_scalar = max_value.get_float(0)?;
        let mean_scalar = mean_value.get_float(0)?;

        // Higher ratio of max to mean suggests higher confidence
        let confidence = (max_scalar / (mean_scalar + 1e-8)).min(1.0);
        Ok(confidence)
    }

    fn calculate_entropy(&self, tensor: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
        // Calculate entropy of output distribution
        let softmax_output = tensor.softmax(-1)?;
        let log_probs = softmax_output.log()?;
        let entropy_tensor = softmax_output
            .mul(&log_probs)?
            .neg()?
            .sum(Some(vec![tensor.shape().len() - 1]), false)?;

        let mean_entropy = entropy_tensor.mean()?;
        Ok(mean_entropy.get_float(0)?)
    }
}

pub trait ComplexityEstimator {
    fn estimate_complexity(&self, input: &Tensor) -> Result<f32, Box<dyn std::error::Error>>;
}

pub struct EntropyBasedComplexityEstimator;

impl ComplexityEstimator for EntropyBasedComplexityEstimator {
    fn estimate_complexity(&self, input: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
        // Calculate input entropy as complexity measure
        let input_normalized = input.softmax(-1)?;
        let log_input = input_normalized.log()?;
        let entropy_tensor = input_normalized.mul(&log_input)?.neg()?.mean()?;
        let entropy = entropy_tensor.get_float(0)?;

        // Normalize to 0-1 range
        let max_entropy = (*input.shape().last().unwrap_or(&1) as f32).ln();
        Ok((entropy / max_entropy).clamp(0.0, 1.0))
    }
}

/// Runtime topology modification and multi-path execution.
pub mod dynamic;

pub use dynamic::*;

#[cfg(test)]
mod tests {
    use super::*;

    /// A path executor that scales the input, so its output really depends on
    /// which path ran.
    struct ScalingPathExecutor;

    impl PathExecutor for ScalingPathExecutor {
        fn execute(
            &self,
            input: &Tensor,
            path: &ExecutionPath,
        ) -> Result<Tensor, Box<dyn std::error::Error>> {
            let scale = match path.path_id.as_str() {
                "conservative" => 1.0,
                "aggressive" => 2.0,
                "efficient" => 0.5,
                _ => 1.5,
            };
            Ok(input.scalar_mul(scale)?)
        }
    }

    fn test_blueprint() -> ArchitectureBlueprint {
        ArchitectureBlueprint {
            layers: vec![LayerConfig {
                layer_id: 0,
                layer_type: LayerType::FeedForward,
                parameters: HashMap::new(),
                optional: false,
            }],
            connections: Vec::new(),
            metadata: ArchitectureMetadata {
                name: "test".to_string(),
                version: "1".to_string(),
                parameter_count: 4,
                memory_footprint_mb: 1,
            },
        }
    }

    /// Regression test: `execute_single_path` used to return `input.clone()`
    /// with a hardcoded `confidence: 0.85`, so the whole voting layer averaged
    /// copies of the input. Without a real executor it must now refuse.
    #[test]
    fn test_multi_path_refuses_without_an_executor() {
        let manager = DynamicArchitectureManager::new(DynamicArchitectureConfig::default());
        assert!(!manager.has_path_executor());

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).expect("from_vec failed");
        let budget = ComputationBudget::new(1_000_000, 1024, 1000);
        let blueprint = test_blueprint();
        let plan = manager
            .create_dynamic_execution_plan(&input, &blueprint, &budget)
            .expect("plan creation failed");

        let error = manager
            .execute_multi_path(&input, &plan)
            .expect_err("no path can be executed without an executor");
        assert!(
            error.to_string().contains("PathExecutor"),
            "unexpected error: {error}"
        );
    }

    /// With an executor the outputs and confidences must come from the real
    /// path computations.
    #[test]
    fn test_multi_path_uses_the_real_executor() {
        let manager = DynamicArchitectureManager::new(DynamicArchitectureConfig::default())
            .with_path_executor(Arc::new(ScalingPathExecutor));

        let input = Tensor::from_vec(vec![0.0, 0.0, 8.0, 0.0], &[1, 4]).expect("from_vec failed");
        let budget = ComputationBudget::new(1_000_000, 1024, 1000);
        let blueprint = test_blueprint();
        let plan = manager
            .create_dynamic_execution_plan(&input, &blueprint, &budget)
            .expect("plan creation failed");

        let result =
            manager.execute_multi_path(&input, &plan).expect("multi-path execution failed");

        assert!(result.paths_executed >= 1);
        for metric in &result.path_metrics {
            assert_ne!(
                metric.confidence, 0.85,
                "confidence must be measured, not the old constant"
            );
            assert!((0.0..=1.0).contains(&metric.confidence));
            assert_ne!(
                metric.resource_usage.flops, 1_000_000,
                "FLOPs must be derived from the real output"
            );
        }

        // The combined output must differ from the input: the executor scaled it.
        let combined = result.result.output.data().expect("data failed");
        let original = input.data().expect("data failed");
        assert_ne!(
            combined, original,
            "the voting layer must combine real path outputs, not copies of the input"
        );
    }

    /// Regression test: `calculate_layer_metrics` hardcoded
    /// `execution_time_ms: 10`.
    #[test]
    fn test_layer_metrics_report_the_measured_duration() {
        let manager = AdaptiveComputationManager::new(
            AdaptiveComputationConfig::default(),
            Box::new(ConfidenceBasedStrategy::default()),
            Box::new(EntropyBasedComplexityEstimator),
        );
        let output = Tensor::from_vec(vec![0.1, 0.9, 0.0, 0.0], &[1, 4]).expect("from_vec failed");
        let budget = ComputationBudget::new(1_000_000, 1024, 1000);

        let metrics = manager
            .calculate_layer_metrics(0, &output, Duration::from_millis(37))
            .expect("metrics failed");
        assert_eq!(metrics.execution_time_ms, 37);

        let zero = manager
            .calculate_layer_metrics(0, &output, Duration::ZERO)
            .expect("metrics failed");
        assert_eq!(
            zero.execution_time_ms, 0,
            "an untimed layer must report 0, not an invented 10ms"
        );

        // The public entry point threads the measured duration through.
        manager
            .should_continue_layer(0, &output, &budget, Duration::from_millis(5))
            .expect("should_continue_layer failed");
    }

    // ── ComputationBudget tests ──

    #[test]
    fn test_computation_budget() {
        let mut budget = ComputationBudget::new(1000, 100, 50);

        assert!(budget.can_afford(500, 50, 25));
        budget.consume(500, 50, 25);

        assert_eq!(budget.remaining_flops, 500);
        assert_eq!(budget.remaining_memory_mb, 50);
        assert_eq!(budget.remaining_time_ms, 25);

        assert!(!budget.can_afford(600, 60, 30));
    }

    #[test]
    fn test_budget_saturating_consume() {
        let mut budget = ComputationBudget::new(100, 10, 5);
        budget.consume(200, 20, 10);
        assert_eq!(budget.remaining_flops, 0);
        assert_eq!(budget.remaining_memory_mb, 0);
        assert_eq!(budget.remaining_time_ms, 0);
    }

    #[test]
    fn test_budget_can_afford_exact() {
        let budget = ComputationBudget::new(100, 10, 5);
        assert!(budget.can_afford(100, 10, 5));
    }

    #[test]
    fn test_budget_can_afford_fails_on_flops() {
        let budget = ComputationBudget::new(100, 10, 5);
        assert!(!budget.can_afford(101, 10, 5));
    }

    #[test]
    fn test_budget_can_afford_fails_on_memory() {
        let budget = ComputationBudget::new(100, 10, 5);
        assert!(!budget.can_afford(100, 11, 5));
    }

    #[test]
    fn test_budget_can_afford_fails_on_time() {
        let budget = ComputationBudget::new(100, 10, 5);
        assert!(!budget.can_afford(100, 10, 6));
    }

    #[test]
    fn test_budget_zero_budget() {
        let budget = ComputationBudget::new(0, 0, 0);
        assert!(budget.can_afford(0, 0, 0));
        assert!(!budget.can_afford(1, 0, 0));
    }

    // ── AdaptiveComputationConfig tests ──

    #[test]
    fn test_adaptive_config_default() {
        let config = AdaptiveComputationConfig::default();
        assert_eq!(config.max_layers, 12);
        assert_eq!(config.min_layers, 2);
        assert!((config.halt_threshold - 0.99).abs() < 1e-6);
        assert!((config.time_penalty - 0.01).abs() < 1e-6);
        assert!((config.early_exit_threshold - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_config_clone() {
        let config = AdaptiveComputationConfig::default();
        let cloned = config.clone();
        assert_eq!(config.max_layers, cloned.max_layers);
        assert_eq!(config.min_layers, cloned.min_layers);
    }

    // ── ConfidenceBasedStrategy tests ──

    #[test]
    fn test_confidence_based_strategy() {
        let strategy = ConfidenceBasedStrategy::new();
        let config = AdaptiveComputationConfig::default();
        let budget = ComputationBudget::new(10000, 1000, 100);

        let metrics = LayerMetrics {
            layer_id: 0,
            flops_estimate: 100,
            memory_usage_mb: 10,
            execution_time_ms: 5,
            confidence_score: 0.5,
            uncertainty_score: 0.5,
            output_entropy: 1.0,
        };

        assert!(strategy.should_continue(0, &metrics, &budget, &config));

        let high_confidence_metrics = LayerMetrics {
            confidence_score: 0.98,
            ..metrics
        };

        assert!(!strategy.should_continue(5, &high_confidence_metrics, &budget, &config));
    }

    #[test]
    fn test_confidence_strategy_respects_min_layers() {
        let strategy = ConfidenceBasedStrategy::new();
        let config = AdaptiveComputationConfig {
            min_layers: 5,
            ..AdaptiveComputationConfig::default()
        };
        let budget = ComputationBudget::new(10000, 1000, 100);

        let metrics = LayerMetrics {
            layer_id: 0,
            flops_estimate: 100,
            memory_usage_mb: 10,
            execution_time_ms: 5,
            confidence_score: 0.8,
            uncertainty_score: 0.2,
            output_entropy: 0.5,
        };

        // Below min_layers, must continue
        assert!(strategy.should_continue(2, &metrics, &budget, &config));
    }

    #[test]
    fn test_confidence_strategy_stops_at_max_layers() {
        let strategy = ConfidenceBasedStrategy::new();
        let config = AdaptiveComputationConfig {
            max_layers: 6,
            ..AdaptiveComputationConfig::default()
        };
        let budget = ComputationBudget::new(10000, 1000, 100);

        let metrics = LayerMetrics {
            layer_id: 6,
            flops_estimate: 100,
            memory_usage_mb: 10,
            execution_time_ms: 5,
            confidence_score: 0.5,
            uncertainty_score: 0.5,
            output_entropy: 1.0,
        };

        assert!(!strategy.should_continue(6, &metrics, &budget, &config));
    }

    #[test]
    fn test_confidence_strategy_stops_on_budget_exhaustion() {
        let strategy = ConfidenceBasedStrategy::new();
        let config = AdaptiveComputationConfig::default();
        let budget = ComputationBudget::new(10, 1, 1); // Very tight budget

        let metrics = LayerMetrics {
            layer_id: 5,
            flops_estimate: 100,
            memory_usage_mb: 10,
            execution_time_ms: 5,
            confidence_score: 0.5,
            uncertainty_score: 0.5,
            output_entropy: 1.0,
        };

        assert!(!strategy.should_continue(5, &metrics, &budget, &config));
    }

    #[test]
    fn test_confidence_strategy_estimate_remaining_cost() {
        let strategy = ConfidenceBasedStrategy::new();
        let metrics = LayerMetrics {
            layer_id: 3,
            flops_estimate: 100,
            memory_usage_mb: 10,
            execution_time_ms: 5,
            confidence_score: 0.5,
            uncertainty_score: 0.5,
            output_entropy: 1.0,
        };

        let (flops, mem, time) = strategy.estimate_remaining_cost(3, 10, &metrics);
        assert_eq!(flops, 700); // 7 remaining * 100
        assert_eq!(mem, 70); // 7 * 10
        assert_eq!(time, 35); // 7 * 5
    }

    #[test]
    fn test_confidence_strategy_adjust_computation_path_low_complexity() {
        let strategy = ConfidenceBasedStrategy::new();
        let config = AdaptiveComputationConfig {
            min_layers: 2,
            max_layers: 12,
            ..AdaptiveComputationConfig::default()
        };
        let budget = ComputationBudget::new(100000, 10000, 1000);

        let path = strategy.adjust_computation_path(0.1, &budget, &config);
        // Low complexity => min_layers
        assert_eq!(path.layers_to_execute.len(), config.min_layers);
    }

    #[test]
    fn test_confidence_strategy_adjust_computation_path_high_complexity() {
        let strategy = ConfidenceBasedStrategy::new();
        let config = AdaptiveComputationConfig {
            min_layers: 2,
            max_layers: 12,
            ..AdaptiveComputationConfig::default()
        };
        let budget = ComputationBudget::new(100000, 10000, 1000);

        let path = strategy.adjust_computation_path(0.9, &budget, &config);
        assert_eq!(path.layers_to_execute.len(), config.max_layers);
    }

    // ── UncertaintyBasedStrategy tests ──

    #[test]
    fn test_uncertainty_strategy_continues_on_high_uncertainty() {
        let strategy = UncertaintyBasedStrategy::new();
        let config = AdaptiveComputationConfig::default();
        let budget = ComputationBudget::new(100000, 10000, 1000);

        let metrics = LayerMetrics {
            layer_id: 5,
            flops_estimate: 100,
            memory_usage_mb: 10,
            execution_time_ms: 5,
            confidence_score: 0.3,
            uncertainty_score: 0.7,
            output_entropy: 1.0,
        };

        assert!(strategy.should_continue(5, &metrics, &budget, &config));
    }

    #[test]
    fn test_uncertainty_strategy_stops_on_low_uncertainty() {
        let strategy = UncertaintyBasedStrategy::new();
        let config = AdaptiveComputationConfig {
            min_layers: 2,
            ..AdaptiveComputationConfig::default()
        };
        let budget = ComputationBudget::new(100000, 10000, 1000);

        let metrics = LayerMetrics {
            layer_id: 5,
            flops_estimate: 100,
            memory_usage_mb: 10,
            execution_time_ms: 5,
            confidence_score: 0.95,
            uncertainty_score: 0.05,
            output_entropy: 0.1,
        };

        assert!(!strategy.should_continue(5, &metrics, &budget, &config));
    }

    #[test]
    fn test_uncertainty_strategy_adjust_path() {
        let strategy = UncertaintyBasedStrategy::new();
        let config = AdaptiveComputationConfig {
            min_layers: 2,
            max_layers: 10,
            ..AdaptiveComputationConfig::default()
        };
        let budget = ComputationBudget::new(100000, 10000, 1000);

        let path = strategy.adjust_computation_path(0.5, &budget, &config);
        assert!(path.layers_to_execute.len() >= config.min_layers);
        assert!(path.layers_to_execute.len() <= config.max_layers);
    }

    // ── PerformanceTracker tests ──

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new();

        tracker.record_layer_execution(0, 10);
        tracker.record_layer_execution(0, 20);
        tracker.record_accuracy(5, 0.85);
        tracker.record_accuracy(5, 0.90);

        assert_eq!(tracker.get_average_execution_time(0), Some(15.0));
        assert_eq!(tracker.get_accuracy_trend(5), Some(0.875));
    }

    #[test]
    fn test_performance_tracker_no_data() {
        let tracker = PerformanceTracker::new();
        assert_eq!(tracker.get_average_execution_time(0), None);
        assert_eq!(tracker.get_accuracy_trend(0), None);
    }

    #[test]
    fn test_performance_tracker_record_resource_usage() {
        let mut tracker = PerformanceTracker::new();
        tracker.record_resource_usage(1000, 50, 10);
        tracker.record_resource_usage(2000, 100, 20);
        assert_eq!(tracker.resource_usage_history.len(), 2);
    }

    #[test]
    fn test_performance_tracker_multiple_layers() {
        let mut tracker = PerformanceTracker::new();
        tracker.record_layer_execution(0, 10);
        tracker.record_layer_execution(1, 20);
        tracker.record_layer_execution(2, 30);

        assert_eq!(tracker.get_average_execution_time(0), Some(10.0));
        assert_eq!(tracker.get_average_execution_time(1), Some(20.0));
        assert_eq!(tracker.get_average_execution_time(2), Some(30.0));
    }

    #[test]
    fn test_performance_tracker_default() {
        let tracker = PerformanceTracker::default();
        assert!(tracker.layer_execution_times.is_empty());
        assert!(tracker.accuracy_by_layers.is_empty());
        assert!(tracker.resource_usage_history.is_empty());
    }

    // ── EntropyBasedComplexityEstimator tests ──

    #[test]
    fn test_entropy_complexity_estimator() {
        let estimator = EntropyBasedComplexityEstimator;

        let low_entropy_input = Tensor::from_vec(vec![10.0, 0.0, 0.0, 0.0, 0.0], &[1, 5])
            .expect("Tensor from_vec failed");

        let high_entropy_input = Tensor::ones(&[1, 5]).expect("Failed to create ones tensor");

        let low_complexity = estimator
            .estimate_complexity(&low_entropy_input)
            .expect("operation failed in test");
        let high_complexity = estimator
            .estimate_complexity(&high_entropy_input)
            .expect("operation failed in test");

        assert!(high_complexity > low_complexity);
        assert!((0.0..=1.0).contains(&low_complexity));
        assert!((0.0..=1.0).contains(&high_complexity));
    }

    // ── ComputationPath tests ──

    #[test]
    fn test_computation_path_structure() {
        let path = ComputationPath {
            layers_to_execute: vec![0, 1, 2, 3],
            skip_patterns: vec![LayerSkipPattern::Approximate],
            early_exit_points: vec![2],
            resource_allocation: ResourceAllocation {
                memory_per_layer: HashMap::new(),
                compute_intensity: HashMap::new(),
                parallelism_factor: HashMap::new(),
            },
        };
        assert_eq!(path.layers_to_execute.len(), 4);
        assert_eq!(path.skip_patterns.len(), 1);
        assert_eq!(path.early_exit_points, vec![2]);
    }

    // ── DynamicArchitectureConfig tests ──

    #[test]
    fn test_dynamic_architecture_config_default() {
        let config = DynamicArchitectureConfig::default();
        assert!(config.enable_dynamic_topology);
        assert_eq!(config.max_concurrent_paths, 4);
        assert!((config.layer_insertion_threshold - 0.3).abs() < 1e-6);
        assert!((config.layer_removal_threshold - 0.8).abs() < 1e-6);
        assert!((config.branching_confidence_threshold - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dynamic_architecture_config_clone() {
        let config = DynamicArchitectureConfig::default();
        let cloned = config.clone();
        assert_eq!(config.max_concurrent_paths, cloned.max_concurrent_paths);
        assert_eq!(
            config.enable_dynamic_topology,
            cloned.enable_dynamic_topology
        );
    }

    // ── LayerSkipPattern tests ──

    #[test]
    fn test_layer_skip_pattern_variants() {
        let skip = LayerSkipPattern::Skip;
        let approx = LayerSkipPattern::Approximate;
        let cached = LayerSkipPattern::Cached;
        let pruned = LayerSkipPattern::Pruned;

        // Clone and Debug
        let _skip_clone = skip.clone();
        let _debug = format!("{:?}", approx);
        let _debug2 = format!("{:?}", cached);
        let _debug3 = format!("{:?}", pruned);
    }

    // ── LayerMetrics tests ──

    #[test]
    fn test_layer_metrics_clone() {
        let metrics = LayerMetrics {
            layer_id: 0,
            flops_estimate: 1000,
            memory_usage_mb: 50,
            execution_time_ms: 10,
            confidence_score: 0.8,
            uncertainty_score: 0.2,
            output_entropy: 0.5,
        };
        let cloned = metrics.clone();
        assert_eq!(metrics.layer_id, cloned.layer_id);
        assert_eq!(metrics.flops_estimate, cloned.flops_estimate);
        assert!((metrics.confidence_score - cloned.confidence_score).abs() < 1e-6);
    }

    // ── ResourceAllocation tests ──

    #[test]
    fn test_resource_allocation_empty() {
        let alloc = ResourceAllocation {
            memory_per_layer: HashMap::new(),
            compute_intensity: HashMap::new(),
            parallelism_factor: HashMap::new(),
        };
        assert!(alloc.memory_per_layer.is_empty());
        assert!(alloc.compute_intensity.is_empty());
        assert!(alloc.parallelism_factor.is_empty());
    }

    #[test]
    fn test_resource_allocation_with_data() {
        let mut alloc = ResourceAllocation {
            memory_per_layer: HashMap::new(),
            compute_intensity: HashMap::new(),
            parallelism_factor: HashMap::new(),
        };
        alloc.memory_per_layer.insert(0, 100);
        alloc.compute_intensity.insert(0, 0.8);
        alloc.parallelism_factor.insert(0, 4);

        assert_eq!(alloc.memory_per_layer.get(&0), Some(&100));
        assert_eq!(alloc.parallelism_factor.get(&0), Some(&4));
    }

    // ── Complexity estimation method / strategy enums ──

    #[test]
    fn test_complexity_estimation_method_variants() {
        let _a = ComplexityEstimationMethod::EntropyBased;
        let _b = ComplexityEstimationMethod::AttentionBased;
        let _c = ComplexityEstimationMethod::GradientNorm;
        let _d = ComplexityEstimationMethod::LearningCurve;
        let _e = ComplexityEstimationMethod::Hybrid;
    }

    #[test]
    fn test_dynamic_depth_strategy_variants() {
        let _a = DynamicDepthStrategy::ConfidenceBased;
        let _b = DynamicDepthStrategy::UncertaintyBased;
        let _c = DynamicDepthStrategy::ResourceConstrained;
        let _d = DynamicDepthStrategy::LatencyOptimized;
        let _e = DynamicDepthStrategy::AccuracyOptimized;
    }

    #[test]
    fn test_path_selection_strategy_variants() {
        let _a = PathSelectionStrategy::ConfidenceBased;
        let _b = PathSelectionStrategy::UncertaintyBased;
        let _c = PathSelectionStrategy::EnsembleVoting;
        let _d = PathSelectionStrategy::AdaptiveRouting;
        let _e = PathSelectionStrategy::CostEffectiveness;
    }

    #[test]
    fn test_voting_mechanism_variants() {
        let _a = VotingMechanism::MajorityVote;
        let _b = VotingMechanism::WeightedAverage;
        let _c = VotingMechanism::ConfidenceWeighted;
        let _d = VotingMechanism::UncertaintyWeighted;
        let _e = VotingMechanism::ExpertMixing;
    }
}
