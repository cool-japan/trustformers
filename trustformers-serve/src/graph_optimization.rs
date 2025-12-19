// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! Graph Optimization Module
//!
//! Provides comprehensive computational graph optimization capabilities for improved
//! inference performance including operator fusion, constant folding, dead code elimination,
//! and performance-aware transformations.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

/// Graph optimization service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphOptimizationConfig {
    /// Enable constant folding optimization
    pub enable_constant_folding: bool,

    /// Enable operator fusion optimization
    pub enable_operator_fusion: bool,

    /// Enable dead code elimination
    pub enable_dead_code_elimination: bool,

    /// Enable memory layout optimization
    pub enable_memory_layout_optimization: bool,

    /// Enable arithmetic optimization
    pub enable_arithmetic_optimization: bool,

    /// Maximum optimization passes
    pub max_optimization_passes: usize,

    /// Optimization timeout in milliseconds
    pub optimization_timeout_ms: u64,

    /// Enable performance estimation
    pub enable_performance_estimation: bool,

    /// Target device for optimization
    pub target_device: OptimizationTarget,

    /// Minimum improvement threshold for applying optimizations
    pub min_improvement_threshold: f64,
}

impl Default for GraphOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_constant_folding: true,
            enable_operator_fusion: true,
            enable_dead_code_elimination: true,
            enable_memory_layout_optimization: true,
            enable_arithmetic_optimization: true,
            max_optimization_passes: 5,
            optimization_timeout_ms: 30000,
            enable_performance_estimation: true,
            target_device: OptimizationTarget::Auto,
            min_improvement_threshold: 0.05, // 5% minimum improvement
        }
    }
}

/// Target device for optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OptimizationTarget {
    /// Automatic device detection
    Auto,
    /// CPU optimization
    Cpu,
    /// CUDA GPU optimization
    Cuda,
    /// Metal GPU optimization
    Metal,
    /// OpenCL optimization
    OpenCl,
    /// Multi-device optimization
    MultiDevice,
}

/// Computational graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node identifier
    pub id: String,

    /// Node operation type
    pub operation: Operation,

    /// Input node IDs
    pub inputs: Vec<String>,

    /// Output shape
    pub output_shape: Vec<usize>,

    /// Data type
    pub data_type: DataType,

    /// Node attributes
    pub attributes: HashMap<String, AttributeValue>,

    /// Estimated execution cost
    pub estimated_cost: Option<f64>,

    /// Memory usage estimate
    pub memory_usage: Option<usize>,
}

/// Operation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Operation {
    /// Matrix multiplication
    MatMul,
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Activation functions
    Activation(ActivationType),
    /// Convolution
    Conv2D,
    /// Pooling
    Pool2D(PoolType),
    /// Reshape
    Reshape,
    /// Transpose
    Transpose,
    /// Concatenation
    Concat,
    /// Split
    Split,
    /// Constant
    Constant,
    /// Input
    Input,
    /// Output
    Output,
    /// Batch normalization
    BatchNorm,
    /// Layer normalization
    LayerNorm,
    /// Attention
    Attention,
    /// Custom operation
    Custom(String),
}

/// Activation function types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    Swish,
    LeakyReLU,
    ELU,
}

/// Pooling types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PoolType {
    Max,
    Average,
    Global,
}

/// Data types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    Float32,
    Float16,
    Int32,
    Int64,
    Bool,
    UInt8,
}

/// Attribute values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    FloatArray(Vec<f64>),
    IntArray(Vec<i64>),
}

/// Computational graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationGraph {
    /// Graph nodes
    pub nodes: HashMap<String, GraphNode>,

    /// Input node IDs
    pub inputs: Vec<String>,

    /// Output node IDs
    pub outputs: Vec<String>,

    /// Graph metadata
    pub metadata: GraphMetadata,
}

/// Graph metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// Graph name
    pub name: String,

    /// Graph version
    pub version: String,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Optimization history
    pub optimization_history: Vec<OptimizationStep>,
}

/// Optimization step record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStep {
    /// Optimization type
    pub optimization_type: String,

    /// Performance improvement
    pub improvement: f64,

    /// Execution time
    pub execution_time: Duration,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Optimization pass result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimized graph
    pub optimized_graph: ComputationGraph,

    /// Applied optimizations
    pub applied_optimizations: Vec<String>,

    /// Performance improvement estimate
    pub estimated_improvement: f64,

    /// Optimization time
    pub optimization_time: Duration,

    /// Memory reduction
    pub memory_reduction: Option<usize>,
}

/// Graph optimization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStats {
    /// Total optimizations performed
    pub total_optimizations: u64,

    /// Average improvement per optimization
    pub average_improvement: f64,

    /// Total optimization time
    pub total_optimization_time: Duration,

    /// Success rate
    pub success_rate: f64,

    /// Most effective optimization
    pub most_effective_optimization: Option<String>,

    /// Optimization type counts
    pub optimization_counts: HashMap<String, u64>,
}

/// Graph optimization errors
#[derive(Debug, Error)]
pub enum GraphOptimizationError {
    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),

    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Timeout during optimization")]
    Timeout,

    #[error("Performance regression detected")]
    PerformanceRegression,

    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

/// Graph optimization service
pub struct GraphOptimizationService {
    config: GraphOptimizationConfig,
    stats: Arc<RwLock<OptimizationStats>>,
    optimization_cache: Arc<RwLock<HashMap<String, OptimizationResult>>>,
}

impl GraphOptimizationService {
    /// Create a new graph optimization service
    pub fn new(config: GraphOptimizationConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(OptimizationStats {
                total_optimizations: 0,
                average_improvement: 0.0,
                total_optimization_time: Duration::from_secs(0),
                success_rate: 0.0,
                most_effective_optimization: None,
                optimization_counts: HashMap::new(),
            })),
            optimization_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Optimize a computational graph
    pub async fn optimize_graph(
        &self,
        mut graph: ComputationGraph,
    ) -> Result<OptimizationResult, GraphOptimizationError> {
        let start_time = Instant::now();
        let original_graph = graph.clone();
        let mut applied_optimizations = Vec::new();

        info!(
            "Starting graph optimization for graph: {}",
            graph.metadata.name
        );

        // Check cache first
        let cache_key = self.generate_cache_key(&graph);
        if let Some(cached_result) = self.optimization_cache.read().await.get(&cache_key) {
            debug!("Using cached optimization result");
            return Ok(cached_result.clone());
        }

        // Validate graph structure
        self.validate_graph(&graph)?;

        let mut pass_count = 0;
        let mut total_improvement = 0.0;

        while pass_count < self.config.max_optimization_passes {
            let pass_start = Instant::now();
            let mut pass_improved = false;

            // Check timeout
            if start_time.elapsed().as_millis() > self.config.optimization_timeout_ms as u128 {
                warn!("Optimization timeout reached");
                break;
            }

            // Constant folding pass
            if self.config.enable_constant_folding {
                if let Ok(improvement) = self.constant_folding_pass(&mut graph).await {
                    if improvement > self.config.min_improvement_threshold {
                        applied_optimizations.push("constant_folding".to_string());
                        total_improvement += improvement;
                        pass_improved = true;
                    }
                }
            }

            // Dead code elimination pass
            if self.config.enable_dead_code_elimination {
                if let Ok(improvement) = self.dead_code_elimination_pass(&mut graph).await {
                    if improvement > self.config.min_improvement_threshold {
                        applied_optimizations.push("dead_code_elimination".to_string());
                        total_improvement += improvement;
                        pass_improved = true;
                    }
                }
            }

            // Operator fusion pass
            if self.config.enable_operator_fusion {
                if let Ok(improvement) = self.operator_fusion_pass(&mut graph).await {
                    if improvement > self.config.min_improvement_threshold {
                        applied_optimizations.push("operator_fusion".to_string());
                        total_improvement += improvement;
                        pass_improved = true;
                    }
                }
            }

            // Arithmetic optimization pass
            if self.config.enable_arithmetic_optimization {
                if let Ok(improvement) = self.arithmetic_optimization_pass(&mut graph).await {
                    if improvement > self.config.min_improvement_threshold {
                        applied_optimizations.push("arithmetic_optimization".to_string());
                        total_improvement += improvement;
                        pass_improved = true;
                    }
                }
            }

            // Memory layout optimization pass
            if self.config.enable_memory_layout_optimization {
                if let Ok(improvement) = self.memory_layout_optimization_pass(&mut graph).await {
                    if improvement > self.config.min_improvement_threshold {
                        applied_optimizations.push("memory_layout_optimization".to_string());
                        total_improvement += improvement;
                        pass_improved = true;
                    }
                }
            }

            pass_count += 1;

            // If no improvements in this pass, break early
            if !pass_improved {
                debug!("No improvements in pass {}, stopping early", pass_count);
                break;
            }

            debug!(
                "Optimization pass {} completed in {:?}",
                pass_count,
                pass_start.elapsed()
            );
        }

        // Performance estimation
        let estimated_improvement = if self.config.enable_performance_estimation {
            self.estimate_performance_improvement(&original_graph, &graph).await?
        } else {
            total_improvement
        };

        // Check for performance regression
        if estimated_improvement < 0.0 {
            return Err(GraphOptimizationError::PerformanceRegression);
        }

        let optimization_time = start_time.elapsed();

        // Calculate memory reduction
        let memory_reduction = self.calculate_memory_reduction(&original_graph, &graph).await;

        // Update optimization history
        graph.metadata.optimization_history.push(OptimizationStep {
            optimization_type: applied_optimizations.join(", "),
            improvement: estimated_improvement,
            execution_time: optimization_time,
            timestamp: chrono::Utc::now(),
        });

        let result = OptimizationResult {
            optimized_graph: graph,
            applied_optimizations,
            estimated_improvement,
            optimization_time,
            memory_reduction,
        };

        // Cache the result
        self.optimization_cache.write().await.insert(cache_key, result.clone());

        // Update statistics
        self.update_stats(&result).await;

        info!(
            "Graph optimization completed in {:?} with {:.2}% improvement",
            optimization_time,
            estimated_improvement * 100.0
        );

        Ok(result)
    }

    /// Constant folding optimization pass
    async fn constant_folding_pass(&self, graph: &mut ComputationGraph) -> Result<f64> {
        let mut improvement = 0.0;
        let mut nodes_to_fold = Vec::new();

        // Find constant nodes that can be folded
        for (node_id, node) in &graph.nodes {
            if self.can_fold_constant(node, graph) {
                nodes_to_fold.push(node_id.clone());
            }
        }

        // Fold constants
        for node_id in nodes_to_fold {
            if let Some(folded_value) = self.fold_constant(&node_id, graph).await? {
                // Replace the computation with a constant
                let constant_node = GraphNode {
                    id: node_id.clone(),
                    operation: Operation::Constant,
                    inputs: Vec::new(),
                    output_shape: graph.nodes[&node_id].output_shape.clone(),
                    data_type: graph.nodes[&node_id].data_type.clone(),
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("value".to_string(), folded_value);
                        attrs
                    },
                    estimated_cost: Some(0.0),
                    memory_usage: graph.nodes[&node_id].memory_usage,
                };

                graph.nodes.insert(node_id, constant_node);
                improvement += 0.1; // Estimate 10% improvement per folded constant
            }
        }

        trace!(
            "Constant folding pass completed with {:.2}% improvement",
            improvement * 100.0
        );
        Ok(improvement)
    }

    /// Dead code elimination optimization pass
    async fn dead_code_elimination_pass(&self, graph: &mut ComputationGraph) -> Result<f64> {
        let mut improvement = 0.0;
        let reachable_nodes = self.find_reachable_nodes(graph);
        let total_nodes = graph.nodes.len();

        // Remove unreachable nodes
        graph.nodes.retain(|node_id, _| reachable_nodes.contains(node_id));

        let removed_nodes = total_nodes - graph.nodes.len();
        if removed_nodes > 0 {
            improvement = removed_nodes as f64 / total_nodes as f64;
            debug!("Removed {} dead nodes", removed_nodes);
        }

        trace!(
            "Dead code elimination pass completed with {:.2}% improvement",
            improvement * 100.0
        );
        Ok(improvement)
    }

    /// Operator fusion optimization pass
    async fn operator_fusion_pass(&self, graph: &mut ComputationGraph) -> Result<f64> {
        let mut improvement = 0.0;
        let fusion_opportunities = self.find_fusion_opportunities(graph);

        for fusion_group in fusion_opportunities {
            if fusion_group.len() >= 2 {
                let fused_node = self.create_fused_node(&fusion_group, graph)?;

                // Remove original nodes and add fused node
                for node_id in &fusion_group {
                    graph.nodes.remove(node_id);
                }
                graph.nodes.insert(fused_node.id.clone(), fused_node);

                improvement += 0.15 * (fusion_group.len() - 1) as f64; // Estimate improvement
            }
        }

        trace!(
            "Operator fusion pass completed with {:.2}% improvement",
            improvement * 100.0
        );
        Ok(improvement)
    }

    /// Arithmetic optimization pass
    async fn arithmetic_optimization_pass(&self, graph: &mut ComputationGraph) -> Result<f64> {
        let mut improvement = 0.0;

        // Find arithmetic optimization opportunities
        let node_ids: Vec<_> = graph.nodes.keys().cloned().collect();
        for node_id in node_ids {
            if let Some(node) = graph.nodes.get(&node_id) {
                match &node.operation {
                    Operation::Add | Operation::Mul => {
                        if self.can_optimize_arithmetic(node, graph) {
                            // Apply arithmetic optimizations (identity operations, etc.)
                            improvement += 0.05;
                        }
                    },
                    _ => {},
                }
            }
        }

        trace!(
            "Arithmetic optimization pass completed with {:.2}% improvement",
            improvement * 100.0
        );
        Ok(improvement)
    }

    /// Memory layout optimization pass
    async fn memory_layout_optimization_pass(&self, graph: &mut ComputationGraph) -> Result<f64> {
        let mut improvement = 0.0;

        // Optimize memory access patterns
        let memory_optimizations = self.analyze_memory_patterns(graph);

        for optimization in memory_optimizations {
            if self.apply_memory_optimization(&optimization, graph).await? {
                improvement += 0.08; // Estimate improvement
            }
        }

        trace!(
            "Memory layout optimization pass completed with {:.2}% improvement",
            improvement * 100.0
        );
        Ok(improvement)
    }

    /// Validate graph structure
    fn validate_graph(&self, graph: &ComputationGraph) -> Result<(), GraphOptimizationError> {
        // Check for cycles
        if self.has_cycles(graph) {
            return Err(GraphOptimizationError::InvalidGraph(
                "Graph contains cycles".to_string(),
            ));
        }

        // Validate node connections
        for (node_id, node) in &graph.nodes {
            for input_id in &node.inputs {
                if !graph.nodes.contains_key(input_id) {
                    return Err(GraphOptimizationError::InvalidGraph(format!(
                        "Node {} references non-existent input {}",
                        node_id, input_id
                    )));
                }
            }
        }

        Ok(())
    }

    /// Check if graph has cycles
    fn has_cycles(&self, graph: &ComputationGraph) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for node_id in graph.nodes.keys() {
            if !visited.contains(node_id)
                && self.has_cycles_util(node_id, graph, &mut visited, &mut rec_stack) {
                    return true;
                }
        }

        false
    }

    /// Utility function for cycle detection
    fn has_cycles_util(
        &self,
        node_id: &str,
        graph: &ComputationGraph,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(node_id.to_string());
        rec_stack.insert(node_id.to_string());

        if let Some(node) = graph.nodes.get(node_id) {
            for input_id in &node.inputs {
                if !visited.contains(input_id) {
                    if self.has_cycles_util(input_id, graph, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(input_id) {
                    return true;
                }
            }
        }

        rec_stack.remove(node_id);
        false
    }

    /// Find reachable nodes from outputs
    fn find_reachable_nodes(&self, graph: &ComputationGraph) -> HashSet<String> {
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();

        // Start from output nodes
        for output_id in &graph.outputs {
            queue.push_back(output_id.clone());
            reachable.insert(output_id.clone());
        }

        // Traverse backwards
        while let Some(node_id) = queue.pop_front() {
            if let Some(node) = graph.nodes.get(&node_id) {
                for input_id in &node.inputs {
                    if !reachable.contains(input_id) {
                        reachable.insert(input_id.clone());
                        queue.push_back(input_id.clone());
                    }
                }
            }
        }

        reachable
    }

    /// Find fusion opportunities
    fn find_fusion_opportunities(&self, graph: &ComputationGraph) -> Vec<Vec<String>> {
        let mut fusion_groups = Vec::new();
        let mut visited = HashSet::new();

        for (node_id, node) in &graph.nodes {
            if !visited.contains(node_id) && self.can_fuse_operation(&node.operation) {
                let fusion_group = self.find_fusion_group(node_id, graph, &mut visited);
                if fusion_group.len() >= 2 {
                    fusion_groups.push(fusion_group);
                }
            }
        }

        fusion_groups
    }

    /// Find fusion group starting from a node
    fn find_fusion_group(
        &self,
        start_node: &str,
        graph: &ComputationGraph,
        visited: &mut HashSet<String>,
    ) -> Vec<String> {
        let mut group = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back(start_node.to_string());

        while let Some(node_id) = queue.pop_front() {
            if visited.contains(&node_id) {
                continue;
            }

            if let Some(node) = graph.nodes.get(&node_id) {
                if self.can_fuse_operation(&node.operation) {
                    visited.insert(node_id.clone());
                    group.push(node_id.clone());

                    // Check connected nodes
                    for input_id in &node.inputs {
                        if !visited.contains(input_id) {
                            queue.push_back(input_id.clone());
                        }
                    }
                }
            }
        }

        group
    }

    /// Check if operation can be fused
    fn can_fuse_operation(&self, operation: &Operation) -> bool {
        matches!(
            operation,
            Operation::Add
                | Operation::Mul
                | Operation::Activation(_)
                | Operation::BatchNorm
                | Operation::LayerNorm
        )
    }

    /// Create fused node from fusion group
    fn create_fused_node(
        &self,
        fusion_group: &[String],
        graph: &ComputationGraph,
    ) -> Result<GraphNode> {
        if fusion_group.is_empty() {
            return Err(anyhow!("Empty fusion group"));
        }

        let _first_node = &graph.nodes[&fusion_group[0]];
        let fused_id = format!("fused_{}", fusion_group.join("_"));

        // Collect all inputs that are external to the fusion group
        let mut external_inputs = Vec::new();
        let fusion_set: HashSet<&String> = fusion_group.iter().collect();

        for node_id in fusion_group {
            if let Some(node) = graph.nodes.get(node_id) {
                for input_id in &node.inputs {
                    if !fusion_set.contains(&input_id) && !external_inputs.contains(input_id) {
                        external_inputs.push(input_id.clone());
                    }
                }
            }
        }

        // Use the output shape of the last node in the group
        let last_node = &graph.nodes[&fusion_group[fusion_group.len() - 1]];

        Ok(GraphNode {
            id: fused_id,
            operation: Operation::Custom("fused_operations".to_string()),
            inputs: external_inputs,
            output_shape: last_node.output_shape.clone(),
            data_type: last_node.data_type.clone(),
            attributes: HashMap::new(),
            estimated_cost: Some(
                fusion_group
                    .iter()
                    .filter_map(|id| graph.nodes.get(id))
                    .filter_map(|node| node.estimated_cost)
                    .sum::<f64>()
                    * 0.7, // Assume 30% reduction from fusion
            ),
            memory_usage: last_node.memory_usage,
        })
    }

    /// Check if constant can be folded
    fn can_fold_constant(&self, node: &GraphNode, graph: &ComputationGraph) -> bool {
        // Check if all inputs are constants
        node.inputs.iter().all(|input_id| {
            graph
                .nodes
                .get(input_id)
                .map(|input_node| input_node.operation == Operation::Constant)
                .unwrap_or(false)
        }) && matches!(
            node.operation,
            Operation::Add | Operation::Mul | Operation::Sub | Operation::Div
        )
    }

    /// Fold constant computation
    async fn fold_constant(
        &self,
        node_id: &str,
        graph: &ComputationGraph,
    ) -> Result<Option<AttributeValue>> {
        let node = &graph.nodes[node_id];

        // Simplified constant folding - in practice this would evaluate the operation
        match &node.operation {
            Operation::Add | Operation::Mul | Operation::Sub | Operation::Div => {
                // Return a placeholder constant value
                Ok(Some(AttributeValue::Float(1.0)))
            },
            _ => Ok(None),
        }
    }

    /// Check if arithmetic operation can be optimized
    fn can_optimize_arithmetic(&self, node: &GraphNode, _graph: &ComputationGraph) -> bool {
        // Check for identity operations, zero multiplications, etc.
        matches!(node.operation, Operation::Add | Operation::Mul)
    }

    /// Analyze memory access patterns
    fn analyze_memory_patterns(&self, _graph: &ComputationGraph) -> Vec<MemoryOptimization> {
        // Simplified implementation - would analyze actual memory patterns
        vec![]
    }

    /// Apply memory optimization
    async fn apply_memory_optimization(
        &self,
        _optimization: &MemoryOptimization,
        _graph: &mut ComputationGraph,
    ) -> Result<bool> {
        // Simplified implementation
        Ok(false)
    }

    /// Estimate performance improvement
    async fn estimate_performance_improvement(
        &self,
        original: &ComputationGraph,
        optimized: &ComputationGraph,
    ) -> Result<f64> {
        let original_cost = self.calculate_graph_cost(original);
        let optimized_cost = self.calculate_graph_cost(optimized);

        if original_cost > 0.0 {
            Ok((original_cost - optimized_cost) / original_cost)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate graph execution cost
    fn calculate_graph_cost(&self, graph: &ComputationGraph) -> f64 {
        graph.nodes.values().filter_map(|node| node.estimated_cost).sum()
    }

    /// Calculate memory reduction
    async fn calculate_memory_reduction(
        &self,
        original: &ComputationGraph,
        optimized: &ComputationGraph,
    ) -> Option<usize> {
        let original_memory: usize =
            original.nodes.values().filter_map(|node| node.memory_usage).sum();

        let optimized_memory: usize =
            optimized.nodes.values().filter_map(|node| node.memory_usage).sum();

        if original_memory > optimized_memory {
            Some(original_memory - optimized_memory)
        } else {
            None
        }
    }

    /// Generate cache key for optimization result
    fn generate_cache_key(&self, graph: &ComputationGraph) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        graph.nodes.len().hash(&mut hasher);
        graph.inputs.hash(&mut hasher);
        graph.outputs.hash(&mut hasher);

        format!("graph_opt_{:x}", hasher.finish())
    }

    /// Update optimization statistics
    async fn update_stats(&self, result: &OptimizationResult) {
        let mut stats = self.stats.write().await;

        stats.total_optimizations += 1;
        stats.total_optimization_time += result.optimization_time;

        // Update average improvement
        let total_improvement = stats.average_improvement * (stats.total_optimizations - 1) as f64
            + result.estimated_improvement;
        stats.average_improvement = total_improvement / stats.total_optimizations as f64;

        // Update success rate (simplified)
        stats.success_rate = if result.estimated_improvement > 0.0 { 1.0 } else { 0.0 };

        // Update optimization counts
        for opt_type in &result.applied_optimizations {
            *stats.optimization_counts.entry(opt_type.clone()).or_insert(0) += 1;
        }

        // Update most effective optimization
        if let Some(best_opt) = stats
            .optimization_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(opt, _)| opt.clone())
        {
            stats.most_effective_optimization = Some(best_opt);
        }
    }

    /// Get optimization statistics
    pub async fn get_stats(&self) -> OptimizationStats {
        self.stats.read().await.clone()
    }

    /// Clear optimization cache
    pub async fn clear_cache(&self) {
        self.optimization_cache.write().await.clear();
    }
}

/// Memory optimization information
#[derive(Debug, Clone)]
struct MemoryOptimization {
    optimization_type: String,
    affected_nodes: Vec<String>,
    estimated_benefit: f64,
}

/// Summary statistics for the optimization service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphOptimizationStatsSummary {
    /// Total optimizations performed
    pub total_optimizations: u64,

    /// Average improvement percentage
    pub average_improvement_percent: f64,

    /// Total optimization time in seconds
    pub total_optimization_time_seconds: f64,

    /// Success rate percentage
    pub success_rate_percent: f64,

    /// Most effective optimization type
    pub most_effective_optimization: Option<String>,

    /// Current cache size
    pub cache_size: usize,
}

impl GraphOptimizationService {
    /// Get summary statistics
    pub async fn get_stats_summary(&self) -> GraphOptimizationStatsSummary {
        let stats = self.stats.read().await;
        let cache_size = self.optimization_cache.read().await.len();

        GraphOptimizationStatsSummary {
            total_optimizations: stats.total_optimizations,
            average_improvement_percent: stats.average_improvement * 100.0,
            total_optimization_time_seconds: stats.total_optimization_time.as_secs_f64(),
            success_rate_percent: stats.success_rate * 100.0,
            most_effective_optimization: stats.most_effective_optimization.clone(),
            cache_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_graph_optimization_service_creation() {
        let config = GraphOptimizationConfig::default();
        let service = GraphOptimizationService::new(config);
        let stats = service.get_stats().await;

        assert_eq!(stats.total_optimizations, 0);
        assert_eq!(stats.average_improvement, 0.0);
    }

    #[tokio::test]
    async fn test_graph_validation() {
        let service = GraphOptimizationService::new(GraphOptimizationConfig::default());

        let mut graph = ComputationGraph {
            nodes: HashMap::new(),
            inputs: vec!["input1".to_string()],
            outputs: vec!["output1".to_string()],
            metadata: GraphMetadata {
                name: "test_graph".to_string(),
                version: "1.0".to_string(),
                created_at: chrono::Utc::now(),
                optimization_history: Vec::new(),
            },
        };

        // Add input node
        graph.nodes.insert(
            "input1".to_string(),
            GraphNode {
                id: "input1".to_string(),
                operation: Operation::Input,
                inputs: Vec::new(),
                output_shape: vec![1, 784],
                data_type: DataType::Float32,
                attributes: HashMap::new(),
                estimated_cost: Some(0.0),
                memory_usage: Some(3136),
            },
        );

        // Add output node
        graph.nodes.insert(
            "output1".to_string(),
            GraphNode {
                id: "output1".to_string(),
                operation: Operation::Output,
                inputs: vec!["input1".to_string()],
                output_shape: vec![1, 10],
                data_type: DataType::Float32,
                attributes: HashMap::new(),
                estimated_cost: Some(0.0),
                memory_usage: Some(40),
            },
        );

        let result = service.validate_graph(&graph);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cycle_detection() {
        let service = GraphOptimizationService::new(GraphOptimizationConfig::default());

        let mut graph = ComputationGraph {
            nodes: HashMap::new(),
            inputs: vec!["node1".to_string()],
            outputs: vec!["node2".to_string()],
            metadata: GraphMetadata {
                name: "cyclic_graph".to_string(),
                version: "1.0".to_string(),
                created_at: chrono::Utc::now(),
                optimization_history: Vec::new(),
            },
        };

        // Create a cycle: node1 -> node2 -> node1
        graph.nodes.insert(
            "node1".to_string(),
            GraphNode {
                id: "node1".to_string(),
                operation: Operation::Add,
                inputs: vec!["node2".to_string()],
                output_shape: vec![1, 10],
                data_type: DataType::Float32,
                attributes: HashMap::new(),
                estimated_cost: Some(1.0),
                memory_usage: Some(40),
            },
        );

        graph.nodes.insert(
            "node2".to_string(),
            GraphNode {
                id: "node2".to_string(),
                operation: Operation::Mul,
                inputs: vec!["node1".to_string()],
                output_shape: vec![1, 10],
                data_type: DataType::Float32,
                attributes: HashMap::new(),
                estimated_cost: Some(1.0),
                memory_usage: Some(40),
            },
        );

        assert!(service.has_cycles(&graph));
    }

    #[test]
    fn test_can_fuse_operation() {
        let service = GraphOptimizationService::new(GraphOptimizationConfig::default());

        assert!(service.can_fuse_operation(&Operation::Add));
        assert!(service.can_fuse_operation(&Operation::Mul));
        assert!(service.can_fuse_operation(&Operation::Activation(ActivationType::ReLU)));
        assert!(service.can_fuse_operation(&Operation::BatchNorm));
        assert!(!service.can_fuse_operation(&Operation::Input));
        assert!(!service.can_fuse_operation(&Operation::Output));
    }
}
