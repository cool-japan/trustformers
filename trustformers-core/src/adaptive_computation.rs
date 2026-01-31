#![allow(unused_variables)] // Adaptive computation implementation with reserved parameters

use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

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
        let mut confidence_history = self
            .confidence_history
            .write()
            .expect("confidence_history lock should not be poisoned");
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

    pub fn should_continue_layer(
        &self,
        layer_id: usize,
        layer_output: &Tensor,
        budget: &ComputationBudget,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Calculate layer metrics
        let metrics = self.calculate_layer_metrics(layer_id, layer_output)?;

        // Use strategy to decide
        let should_continue =
            self.strategy.should_continue(layer_id, &metrics, budget, &self.config);

        // Update performance tracking
        {
            let mut tracker = self
                .performance_tracker
                .write()
                .expect("performance_tracker lock should not be poisoned");
            tracker.record_layer_execution(layer_id, metrics.execution_time_ms);
        }

        Ok(should_continue)
    }

    fn calculate_layer_metrics(
        &self,
        layer_id: usize,
        layer_output: &Tensor,
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
            execution_time_ms: 10, // This would be measured in practice
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

// ===== DYNAMIC ARCHITECTURE SUPPORT =====

/// Dynamic architecture configuration supporting runtime topology modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicArchitectureConfig {
    pub enable_dynamic_topology: bool,
    pub max_concurrent_paths: usize,
    pub path_selection_strategy: PathSelectionStrategy,
    pub architecture_search_enabled: bool,
    pub runtime_modification_enabled: bool,
    pub voting_mechanism: VotingMechanism,
    pub layer_insertion_threshold: f32,
    pub layer_removal_threshold: f32,
    pub branching_confidence_threshold: f32,
}

impl Default for DynamicArchitectureConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_topology: true,
            max_concurrent_paths: 4,
            path_selection_strategy: PathSelectionStrategy::ConfidenceBased,
            architecture_search_enabled: false,
            runtime_modification_enabled: true,
            voting_mechanism: VotingMechanism::WeightedAverage,
            layer_insertion_threshold: 0.3,
            layer_removal_threshold: 0.8,
            branching_confidence_threshold: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathSelectionStrategy {
    ConfidenceBased,
    UncertaintyBased,
    EnsembleVoting,
    AdaptiveRouting,
    CostEffectiveness,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingMechanism {
    MajorityVote,
    WeightedAverage,
    ConfidenceWeighted,
    UncertaintyWeighted,
    ExpertMixing,
}

/// Dynamic architecture manager supporting runtime topology changes
#[derive(Debug)]
pub struct DynamicArchitectureManager {
    config: DynamicArchitectureConfig,
    #[allow(dead_code)]
    active_paths: Arc<RwLock<HashMap<String, ExecutionPath>>>,
    #[allow(dead_code)]
    architecture_cache: Arc<RwLock<HashMap<String, CachedArchitecture>>>,
    performance_history: Arc<RwLock<Vec<ArchitecturePerformance>>>,
    topology_modifier: TopologyModifier,
    path_router: PathRouter,
}

impl DynamicArchitectureManager {
    pub fn new(config: DynamicArchitectureConfig) -> Self {
        Self {
            config: config.clone(),
            active_paths: Arc::new(RwLock::new(HashMap::new())),
            architecture_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_history: Arc::new(RwLock::new(Vec::new())),
            topology_modifier: TopologyModifier::new(config.clone()),
            path_router: PathRouter::new(config.clone()),
        }
    }

    /// Create dynamic execution plan with multiple paths
    pub fn create_dynamic_execution_plan(
        &self,
        input: &Tensor,
        base_architecture: &ArchitectureBlueprint,
        constraints: &ComputationBudget,
    ) -> Result<DynamicExecutionPlan, Box<dyn std::error::Error>> {
        let input_complexity = self.analyze_input_complexity(input)?;

        // Generate multiple execution paths
        let execution_paths =
            self.generate_execution_paths(input_complexity, base_architecture, constraints)?;

        // Select optimal paths based on strategy
        let selected_paths = self.path_router.select_optimal_paths(
            &execution_paths,
            &self.config.path_selection_strategy,
            self.config.max_concurrent_paths,
        )?;

        Ok(DynamicExecutionPlan {
            paths: selected_paths,
            voting_mechanism: self.config.voting_mechanism.clone(),
            fallback_path: self.create_fallback_path(base_architecture)?,
            modification_points: self.identify_modification_points(base_architecture)?,
        })
    }

    /// Modify architecture topology at runtime
    pub fn modify_architecture_runtime(
        &self,
        current_state: &ExecutionState,
        performance_metrics: &LayerMetrics,
        architecture: &mut ArchitectureBlueprint,
    ) -> Result<Vec<TopologyModification>, Box<dyn std::error::Error>> {
        if !self.config.runtime_modification_enabled {
            return Ok(vec![]);
        }

        let mut modifications = Vec::new();

        // Dynamic layer insertion based on uncertainty
        if performance_metrics.uncertainty_score > self.config.layer_insertion_threshold {
            let insertion_point = self
                .topology_modifier
                .find_optimal_insertion_point(current_state, architecture)?;

            if let Some(point) = insertion_point {
                let new_layer =
                    self.topology_modifier.create_adaptive_layer(point, performance_metrics)?;

                modifications.push(TopologyModification::InsertLayer {
                    position: point,
                    layer_config: new_layer,
                });
            }
        }

        // Dynamic layer removal based on confidence
        if performance_metrics.confidence_score > self.config.layer_removal_threshold {
            let removal_candidates =
                self.topology_modifier.identify_redundant_layers(current_state, architecture)?;

            for candidate in removal_candidates {
                modifications.push(TopologyModification::RemoveLayer {
                    position: candidate,
                });
            }
        }

        // Dynamic branching based on confidence distribution
        if self.should_create_branch(performance_metrics) {
            let branch_config = self
                .topology_modifier
                .create_branch_configuration(current_state, performance_metrics)?;

            modifications.push(TopologyModification::CreateBranch {
                source_position: current_state.current_layer,
                branch_config,
            });
        }

        // Apply modifications to architecture
        for modification in &modifications {
            self.topology_modifier.apply_modification(architecture, modification)?;
        }

        Ok(modifications)
    }

    /// Execute multi-path computation with voting
    pub fn execute_multi_path(
        &self,
        input: &Tensor,
        execution_plan: &DynamicExecutionPlan,
    ) -> Result<MultiPathResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let mut path_results = Vec::new();
        let mut path_metrics = Vec::new();

        // Execute all paths concurrently (simplified serial execution for now)
        for (path_id, path) in execution_plan.paths.iter().enumerate() {
            let path_start = Instant::now();
            let result = self.execute_single_path(input, path)?;
            let execution_time = path_start.elapsed();

            let metrics = PathExecutionMetrics {
                path_id,
                execution_time,
                confidence: result.confidence,
                accuracy_estimate: result.accuracy_estimate,
                resource_usage: result.resource_usage.clone(),
            };

            path_results.push(result);
            path_metrics.push(metrics);
        }

        // Combine results using voting mechanism
        let final_result = self.combine_path_results(
            &path_results,
            &path_metrics,
            &execution_plan.voting_mechanism,
        )?;

        // Record performance for future optimization
        self.record_multi_path_performance(&path_metrics, &final_result);

        Ok(MultiPathResult {
            result: final_result,
            path_metrics,
            total_execution_time: start_time.elapsed(),
            paths_executed: path_results.len(),
        })
    }

    fn analyze_input_complexity(&self, input: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
        // Compute input complexity using entropy, variance, and distribution analysis
        let entropy = self.compute_entropy(input)?;
        let variance = self.compute_variance(input)?;
        let sparsity = self.compute_sparsity(input)?;

        // Combine metrics into unified complexity score
        let complexity = (entropy * 0.4 + variance * 0.3 + (1.0 - sparsity) * 0.3).clamp(0.0, 1.0);
        Ok(complexity)
    }

    fn generate_execution_paths(
        &self,
        complexity: f32,
        architecture: &ArchitectureBlueprint,
        constraints: &ComputationBudget,
    ) -> Result<Vec<ExecutionPath>, Box<dyn std::error::Error>> {
        let mut paths = Vec::new();

        // Conservative path (minimal layers, high accuracy)
        let conservative_path = ExecutionPath {
            path_id: "conservative".to_string(),
            layers: self.select_essential_layers(architecture)?,
            skip_patterns: HashMap::new(),
            resource_allocation: self.allocate_resources_conservatively(constraints)?,
            expected_accuracy: 0.9,
            expected_latency: Duration::from_millis(50),
        };
        paths.push(conservative_path);

        // Aggressive path (maximum layers, highest accuracy)
        let aggressive_path = ExecutionPath {
            path_id: "aggressive".to_string(),
            layers: architecture.layers.clone(),
            skip_patterns: HashMap::new(),
            resource_allocation: self.allocate_resources_aggressively(constraints)?,
            expected_accuracy: 0.95,
            expected_latency: Duration::from_millis(200),
        };
        paths.push(aggressive_path);

        // Adaptive path (complexity-based selection)
        let adaptive_layers = self.select_adaptive_layers(architecture, complexity)?;
        let adaptive_path = ExecutionPath {
            path_id: "adaptive".to_string(),
            layers: adaptive_layers,
            skip_patterns: self.generate_skip_patterns(complexity)?,
            resource_allocation: self.allocate_resources_adaptively(constraints, complexity)?,
            expected_accuracy: 0.85 + complexity * 0.1,
            expected_latency: Duration::from_millis((100.0 + complexity * 100.0) as u64),
        };
        paths.push(adaptive_path);

        // Efficient path (optimized for speed)
        let efficient_path = ExecutionPath {
            path_id: "efficient".to_string(),
            layers: self.select_efficient_layers(architecture)?,
            skip_patterns: self.generate_efficiency_skip_patterns()?,
            resource_allocation: self.allocate_resources_efficiently(constraints)?,
            expected_accuracy: 0.8,
            expected_latency: Duration::from_millis(30),
        };
        paths.push(efficient_path);

        Ok(paths)
    }

    fn should_create_branch(&self, metrics: &LayerMetrics) -> bool {
        metrics.uncertainty_score > self.config.branching_confidence_threshold
            && metrics.confidence_score < 0.8 // High uncertainty, moderate confidence
    }

    fn execute_single_path(
        &self,
        input: &Tensor,
        path: &ExecutionPath,
    ) -> Result<PathResult, Box<dyn std::error::Error>> {
        // Simplified path execution - in real implementation this would
        // execute the actual neural network layers according to the path
        Ok(PathResult {
            output: input.clone(), // Placeholder
            confidence: 0.85,
            accuracy_estimate: path.expected_accuracy,
            resource_usage: ResourceUsage {
                memory_mb: 100,
                flops: 1000000,
                time_ms: path.expected_latency.as_millis() as u32,
            },
        })
    }

    fn combine_path_results(
        &self,
        results: &[PathResult],
        metrics: &[PathExecutionMetrics],
        voting_mechanism: &VotingMechanism,
    ) -> Result<CombinedResult, Box<dyn std::error::Error>> {
        match voting_mechanism {
            VotingMechanism::WeightedAverage => {
                // Weight by confidence scores
                let total_confidence: f32 = metrics.iter().map(|m| m.confidence).sum();
                let mut weighted_output = Tensor::zeros_like(&results[0].output)?;

                for (result, metric) in results.iter().zip(metrics.iter()) {
                    let weight = metric.confidence / total_confidence;
                    let scaled_output = result.output.mul_scalar(weight)?;
                    weighted_output = weighted_output.add(&scaled_output)?;
                }

                Ok(CombinedResult {
                    output: weighted_output,
                    confidence: total_confidence / results.len() as f32,
                    consensus_score: self.calculate_consensus_score(metrics),
                })
            },
            VotingMechanism::MajorityVote => {
                // Find the most confident result
                let best_idx = metrics
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.confidence.partial_cmp(&b.confidence).expect("Partial comparison failed")
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                Ok(CombinedResult {
                    output: results[best_idx].output.clone(),
                    confidence: metrics[best_idx].confidence,
                    consensus_score: self.calculate_consensus_score(metrics),
                })
            },
            _ => {
                // Fallback to weighted average
                self.combine_path_results(results, metrics, &VotingMechanism::WeightedAverage)
            },
        }
    }

    fn calculate_consensus_score(&self, metrics: &[PathExecutionMetrics]) -> f32 {
        if metrics.len() < 2 {
            return 1.0;
        }

        let mean_confidence: f32 =
            metrics.iter().map(|m| m.confidence).sum::<f32>() / metrics.len() as f32;
        let variance =
            metrics.iter().map(|m| (m.confidence - mean_confidence).powi(2)).sum::<f32>()
                / metrics.len() as f32;

        // Higher consensus when variance is low
        (1.0 - variance.sqrt()).clamp(0.0, 1.0)
    }

    fn record_multi_path_performance(
        &self,
        path_metrics: &[PathExecutionMetrics],
        result: &CombinedResult,
    ) {
        let performance = ArchitecturePerformance {
            timestamp: Instant::now(),
            path_count: path_metrics.len(),
            average_confidence: path_metrics.iter().map(|m| m.confidence).sum::<f32>()
                / path_metrics.len() as f32,
            consensus_score: result.consensus_score,
            total_resource_usage: path_metrics.iter().map(|m| m.resource_usage.memory_mb).sum(),
        };

        if let Ok(mut history) = self.performance_history.write() {
            history.push(performance);
            // Keep only recent history (last 1000 entries)
            if history.len() > 1000 {
                let drain_count = history.len() - 1000;
                history.drain(0..drain_count);
            }
        }
    }

    // Helper methods for path generation
    fn select_essential_layers(
        &self,
        arch: &ArchitectureBlueprint,
    ) -> Result<Vec<LayerConfig>, Box<dyn std::error::Error>> {
        Ok(arch.layers.iter().take(arch.layers.len() / 2).cloned().collect())
    }

    fn select_adaptive_layers(
        &self,
        arch: &ArchitectureBlueprint,
        complexity: f32,
    ) -> Result<Vec<LayerConfig>, Box<dyn std::error::Error>> {
        let layer_count =
            ((arch.layers.len() as f32 * (0.5 + complexity * 0.5)) as usize).min(arch.layers.len());
        Ok(arch.layers.iter().take(layer_count).cloned().collect())
    }

    fn select_efficient_layers(
        &self,
        arch: &ArchitectureBlueprint,
    ) -> Result<Vec<LayerConfig>, Box<dyn std::error::Error>> {
        Ok(arch.layers.iter().step_by(2).cloned().collect())
    }

    fn generate_skip_patterns(
        &self,
        complexity: f32,
    ) -> Result<HashMap<usize, LayerSkipPattern>, Box<dyn std::error::Error>> {
        let mut patterns = HashMap::new();
        if complexity < 0.3 {
            // Skip every other layer for simple inputs
            for i in (1..10).step_by(2) {
                patterns.insert(i, LayerSkipPattern::Skip);
            }
        }
        Ok(patterns)
    }

    fn generate_efficiency_skip_patterns(
        &self,
    ) -> Result<HashMap<usize, LayerSkipPattern>, Box<dyn std::error::Error>> {
        let mut patterns = HashMap::new();
        // Aggressive skipping for efficiency
        for i in 2..10 {
            if i % 3 == 0 {
                patterns.insert(i, LayerSkipPattern::Approximate);
            }
        }
        Ok(patterns)
    }

    fn allocate_resources_conservatively(
        &self,
        budget: &ComputationBudget,
    ) -> Result<ResourceAllocation, Box<dyn std::error::Error>> {
        Ok(ResourceAllocation {
            memory_per_layer: (0..5).map(|i| (i, budget.max_memory_mb / 10)).collect(),
            compute_intensity: (0..5).map(|i| (i, 0.5)).collect(),
            parallelism_factor: (0..5).map(|i| (i, 1)).collect(),
        })
    }

    fn allocate_resources_aggressively(
        &self,
        budget: &ComputationBudget,
    ) -> Result<ResourceAllocation, Box<dyn std::error::Error>> {
        Ok(ResourceAllocation {
            memory_per_layer: (0..10).map(|i| (i, budget.max_memory_mb / 5)).collect(),
            compute_intensity: (0..10).map(|i| (i, 1.0)).collect(),
            parallelism_factor: (0..10).map(|i| (i, 2)).collect(),
        })
    }

    fn allocate_resources_adaptively(
        &self,
        budget: &ComputationBudget,
        complexity: f32,
    ) -> Result<ResourceAllocation, Box<dyn std::error::Error>> {
        let layer_count = (8.0 * (0.5 + complexity * 0.5)) as usize;
        Ok(ResourceAllocation {
            memory_per_layer: (0..layer_count)
                .map(|i| {
                    (
                        i,
                        (budget.max_memory_mb as f32 * (0.5 + complexity * 0.5)) as u32
                            / layer_count as u32,
                    )
                })
                .collect(),
            compute_intensity: (0..layer_count).map(|i| (i, 0.5 + complexity * 0.5)).collect(),
            parallelism_factor: (0..layer_count)
                .map(|i| (i, 1 + (complexity * 2.0) as u32))
                .collect(),
        })
    }

    fn allocate_resources_efficiently(
        &self,
        budget: &ComputationBudget,
    ) -> Result<ResourceAllocation, Box<dyn std::error::Error>> {
        Ok(ResourceAllocation {
            memory_per_layer: (0..3).map(|i| (i, budget.max_memory_mb / 20)).collect(),
            compute_intensity: (0..3).map(|i| (i, 0.3)).collect(),
            parallelism_factor: (0..3).map(|i| (i, 4)).collect(),
        })
    }

    fn create_fallback_path(
        &self,
        arch: &ArchitectureBlueprint,
    ) -> Result<ExecutionPath, Box<dyn std::error::Error>> {
        Ok(ExecutionPath {
            path_id: "fallback".to_string(),
            layers: vec![arch.layers[0].clone()], // Minimal single layer
            skip_patterns: HashMap::new(),
            resource_allocation: ResourceAllocation {
                memory_per_layer: HashMap::from([(0, 50)]),
                compute_intensity: HashMap::from([(0, 0.1)]),
                parallelism_factor: HashMap::from([(0, 1)]),
            },
            expected_accuracy: 0.6,
            expected_latency: Duration::from_millis(10),
        })
    }

    fn identify_modification_points(
        &self,
        arch: &ArchitectureBlueprint,
    ) -> Result<Vec<ModificationPoint>, Box<dyn std::error::Error>> {
        let mut points = Vec::new();

        // Add modification points between layers
        for i in 0..arch.layers.len() - 1 {
            points.push(ModificationPoint {
                position: i,
                modification_type: ModificationType::LayerInsertion,
                confidence_threshold: 0.3,
            });
        }

        // Add branching points at quarter, half, and three-quarter positions
        for &fraction in &[0.25, 0.5, 0.75] {
            let position = (arch.layers.len() as f32 * fraction) as usize;
            points.push(ModificationPoint {
                position,
                modification_type: ModificationType::BranchingPoint,
                confidence_threshold: 0.5,
            });
        }

        Ok(points)
    }

    // Tensor operation helpers
    fn compute_entropy(&self, tensor: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
        // Simplified entropy calculation
        Ok(0.5) // Placeholder
    }

    fn compute_variance(&self, tensor: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
        // Simplified variance calculation
        Ok(0.3) // Placeholder
    }

    fn compute_sparsity(&self, tensor: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
        // Simplified sparsity calculation
        Ok(0.2) // Placeholder
    }
}

// Supporting data structures for dynamic architectures

#[derive(Debug, Clone)]
pub struct ArchitectureBlueprint {
    pub layers: Vec<LayerConfig>,
    pub connections: Vec<ConnectionConfig>,
    pub metadata: ArchitectureMetadata,
}

#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub layer_id: usize,
    pub layer_type: LayerType,
    pub parameters: HashMap<String, f32>,
    pub optional: bool,
}

#[derive(Debug, Clone)]
pub enum LayerType {
    Attention,
    FeedForward,
    Normalization,
    Embedding,
    Output,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    pub from_layer: usize,
    pub to_layer: usize,
    pub connection_type: ConnectionType,
}

#[derive(Debug, Clone)]
pub enum ConnectionType {
    Sequential,
    Residual,
    Attention,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct ArchitectureMetadata {
    pub name: String,
    pub version: String,
    pub parameter_count: u64,
    pub memory_footprint_mb: u32,
}

#[derive(Debug, Clone)]
pub struct ExecutionPath {
    pub path_id: String,
    pub layers: Vec<LayerConfig>,
    pub skip_patterns: HashMap<usize, LayerSkipPattern>,
    pub resource_allocation: ResourceAllocation,
    pub expected_accuracy: f32,
    pub expected_latency: Duration,
}

#[derive(Debug)]
pub struct DynamicExecutionPlan {
    pub paths: Vec<ExecutionPath>,
    pub voting_mechanism: VotingMechanism,
    pub fallback_path: ExecutionPath,
    pub modification_points: Vec<ModificationPoint>,
}

#[derive(Debug)]
pub struct ExecutionState {
    pub current_layer: usize,
    pub intermediate_results: HashMap<usize, Tensor>,
    pub execution_metrics: Vec<LayerMetrics>,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub memory_mb: u32,
    pub flops: u64,
    pub time_ms: u32,
}

#[derive(Debug)]
pub enum TopologyModification {
    InsertLayer {
        position: usize,
        layer_config: LayerConfig,
    },
    RemoveLayer {
        position: usize,
    },
    CreateBranch {
        source_position: usize,
        branch_config: BranchConfig,
    },
    ModifyConnection {
        connection: ConnectionConfig,
    },
}

#[derive(Debug, Clone)]
pub struct BranchConfig {
    pub branch_layers: Vec<LayerConfig>,
    pub merge_strategy: MergeStrategy,
    pub condition: BranchCondition,
}

#[derive(Debug, Clone)]
pub enum MergeStrategy {
    Concatenation,
    Average,
    WeightedSum,
    Attention,
}

#[derive(Debug, Clone)]
pub enum BranchCondition {
    Always,
    ConfidenceThreshold(f32),
    UncertaintyThreshold(f32),
    Custom(String),
}

#[derive(Debug)]
pub struct ModificationPoint {
    pub position: usize,
    pub modification_type: ModificationType,
    pub confidence_threshold: f32,
}

#[derive(Debug)]
pub enum ModificationType {
    LayerInsertion,
    LayerRemoval,
    BranchingPoint,
    ConnectionModification,
}

#[derive(Debug)]
pub struct TopologyModifier {
    #[allow(dead_code)]
    config: DynamicArchitectureConfig,
}

impl TopologyModifier {
    pub fn new(config: DynamicArchitectureConfig) -> Self {
        Self { config }
    }

    pub fn find_optimal_insertion_point(
        &self,
        state: &ExecutionState,
        architecture: &ArchitectureBlueprint,
    ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        // Find the best position to insert a new layer based on current execution state
        if state.current_layer > 0 && state.current_layer < architecture.layers.len() {
            Ok(Some(state.current_layer))
        } else {
            Ok(None)
        }
    }

    pub fn create_adaptive_layer(
        &self,
        position: usize,
        metrics: &LayerMetrics,
    ) -> Result<LayerConfig, Box<dyn std::error::Error>> {
        // Create a new layer configuration based on current metrics
        let layer_type = if metrics.uncertainty_score > 0.7 {
            LayerType::Attention // Add attention for high uncertainty
        } else {
            LayerType::FeedForward // Add feedforward for general improvement
        };

        Ok(LayerConfig {
            layer_id: position * 1000, // Unique ID for inserted layers
            layer_type,
            parameters: HashMap::from([
                ("hidden_size".to_string(), 512.0),
                ("dropout".to_string(), 0.1),
            ]),
            optional: true,
        })
    }

    pub fn identify_redundant_layers(
        &self,
        state: &ExecutionState,
        architecture: &ArchitectureBlueprint,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let mut redundant = Vec::new();

        // Identify layers that can be safely removed
        for (i, layer) in architecture.layers.iter().enumerate() {
            if layer.optional
                && state.execution_metrics.get(i).is_some_and(|m| m.confidence_score > 0.9)
            {
                redundant.push(i);
            }
        }

        Ok(redundant)
    }

    pub fn create_branch_configuration(
        &self,
        state: &ExecutionState,
        metrics: &LayerMetrics,
    ) -> Result<BranchConfig, Box<dyn std::error::Error>> {
        let branch_layers = vec![LayerConfig {
            layer_id: 9000, // Branch layer ID
            layer_type: LayerType::Attention,
            parameters: HashMap::from([("heads".to_string(), 8.0)]),
            optional: true,
        }];

        Ok(BranchConfig {
            branch_layers,
            merge_strategy: MergeStrategy::WeightedSum,
            condition: BranchCondition::UncertaintyThreshold(metrics.uncertainty_score),
        })
    }

    pub fn apply_modification(
        &self,
        architecture: &mut ArchitectureBlueprint,
        modification: &TopologyModification,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match modification {
            TopologyModification::InsertLayer {
                position,
                layer_config,
            } => {
                if *position <= architecture.layers.len() {
                    architecture.layers.insert(*position, layer_config.clone());
                }
            },
            TopologyModification::RemoveLayer { position } => {
                if *position < architecture.layers.len() {
                    architecture.layers.remove(*position);
                }
            },
            TopologyModification::CreateBranch {
                source_position,
                branch_config,
            } => {
                // Add branch layers after source position
                for (i, layer) in branch_config.branch_layers.iter().enumerate() {
                    architecture.layers.insert(source_position + i + 1, layer.clone());
                }
            },
            TopologyModification::ModifyConnection { connection } => {
                // Add or modify connections
                architecture.connections.push(connection.clone());
            },
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct PathRouter {
    #[allow(dead_code)]
    config: DynamicArchitectureConfig,
}

impl PathRouter {
    pub fn new(config: DynamicArchitectureConfig) -> Self {
        Self { config }
    }

    pub fn select_optimal_paths(
        &self,
        paths: &[ExecutionPath],
        strategy: &PathSelectionStrategy,
        max_paths: usize,
    ) -> Result<Vec<ExecutionPath>, Box<dyn std::error::Error>> {
        let mut selected = match strategy {
            PathSelectionStrategy::ConfidenceBased => {
                let mut sorted_paths = paths.to_vec();
                sorted_paths.sort_by(|a, b| {
                    b.expected_accuracy
                        .partial_cmp(&a.expected_accuracy)
                        .expect("Partial comparison failed")
                });
                sorted_paths
            },
            PathSelectionStrategy::CostEffectiveness => {
                let mut sorted_paths = paths.to_vec();
                sorted_paths.sort_by(|a, b| {
                    let cost_a = a.expected_latency.as_millis() as f32 / a.expected_accuracy;
                    let cost_b = b.expected_latency.as_millis() as f32 / b.expected_accuracy;
                    cost_a.partial_cmp(&cost_b).expect("Partial comparison failed")
                });
                sorted_paths
            },
            _ => paths.to_vec(),
        };

        selected.truncate(max_paths);
        Ok(selected)
    }
}

#[derive(Debug)]
pub struct PathResult {
    pub output: Tensor,
    pub confidence: f32,
    pub accuracy_estimate: f32,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug)]
pub struct PathExecutionMetrics {
    pub path_id: usize,
    pub execution_time: Duration,
    pub confidence: f32,
    pub accuracy_estimate: f32,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug)]
pub struct CombinedResult {
    pub output: Tensor,
    pub confidence: f32,
    pub consensus_score: f32,
}

#[derive(Debug)]
pub struct MultiPathResult {
    pub result: CombinedResult,
    pub path_metrics: Vec<PathExecutionMetrics>,
    pub total_execution_time: Duration,
    pub paths_executed: usize,
}

#[derive(Debug)]
pub struct ArchitecturePerformance {
    pub timestamp: Instant,
    pub path_count: usize,
    pub average_confidence: f32,
    pub consensus_score: f32,
    pub total_resource_usage: u32,
}

#[derive(Debug)]
pub struct CachedArchitecture {
    pub blueprint: ArchitectureBlueprint,
    pub performance_history: Vec<ArchitecturePerformance>,
    pub last_used: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // Should continue with low confidence and within limits
        assert!(strategy.should_continue(0, &metrics, &budget, &config));

        let high_confidence_metrics = LayerMetrics {
            confidence_score: 0.98,
            ..metrics
        };

        // Should stop with high confidence
        assert!(!strategy.should_continue(5, &high_confidence_metrics, &budget, &config));
    }

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
    fn test_entropy_complexity_estimator() {
        let estimator = EntropyBasedComplexityEstimator;

        // Create a low-entropy input (concentrated distribution after softmax)
        let low_entropy_input = Tensor::from_vec(vec![10.0, 0.0, 0.0, 0.0, 0.0], &[1, 5])
            .expect("Tensor from_vec failed");

        // Create a high-entropy input (uniform distribution after softmax)
        let high_entropy_input = Tensor::ones(&[1, 5]).expect("Failed to create ones tensor");

        let low_complexity = estimator.estimate_complexity(&low_entropy_input).unwrap();
        let high_complexity = estimator.estimate_complexity(&high_entropy_input).unwrap();

        // Uniform distribution should have higher entropy than concentrated distribution
        assert!(high_complexity > low_complexity);

        // Test that complexity is in valid range [0, 1]
        assert!((0.0..=1.0).contains(&low_complexity));
        assert!((0.0..=1.0).contains(&high_complexity));
    }
}
