/*!
# Compiler Optimization Module

This module provides comprehensive compiler optimizations for TrustformeRS including:

- **JIT Compilation**: Just-in-time compilation for dynamic graphs
- **Kernel Fusion**: Automatic fusion of compatible operations
- **Graph Optimization**: Multi-pass optimization of computation graphs
- **MLIR Integration**: Integration with MLIR for advanced compiler optimizations

## Features

- Automatic operation fusion for reduced memory bandwidth
- Graph-level optimizations including constant folding and dead code elimination
- JIT compilation for dynamic computation graphs
- MLIR-based optimizations for maximum performance
- Adaptive optimization strategies based on hardware characteristics
- Comprehensive performance analysis and optimization recommendations

## Usage

```rust
use trustformers_core::compiler::{CompilerOptimizer, OptimizationLevel};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let computation_graph = ComputationGraph::new();
let mut optimizer = CompilerOptimizer::with_optimization_level(OptimizationLevel::Aggressive)?;
let optimization_result = optimizer.optimize_graph(computation_graph)?;
let compiled_model = optimizer.compile_graph(optimization_result.optimized_graph)?;
# Ok(())
# }
```
*/

pub mod analysis;
pub mod graph_optimizer;
pub mod jit_compiler;
pub mod kernel_fusion;
pub mod mlir_backend;
pub mod passes;

// Re-export key types for convenience
pub use analysis::{
    BottleneckInfo, DependencyAnalysis, GraphAnalyzer, HardwareUtilization, MemoryAnalysis,
    PerformanceAnalysis,
};
pub use jit_compiler::{
    IRInstruction, IROpcode, IntermediateRepresentation, JitBackend, JitCompiler,
};
pub use kernel_fusion::{FusionGroup, FusionPattern, FusionResult, FusionType, KernelFusion};
pub use mlir_backend::{DialectSupport, MlirBackend};
pub use passes::{
    CommonSubexpressionEliminationPass, ConstantFoldingPass, DeadCodeEliminationPass,
    MemoryLayoutOptimizationPass, OperationFusionPass, PassManager,
};

use crate::errors::invalid_input;
use crate::errors::TrustformersError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Optimization level for compiler optimizations
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimizations (debug mode)
    None,
    /// Basic optimizations with minimal compilation time
    Basic,
    /// Standard optimizations with balanced compilation time/performance
    #[default]
    Standard,
    /// Aggressive optimizations with longer compilation time
    Aggressive,
    /// Maximum optimizations (may significantly increase compilation time)
    Maximum,
}

/// Configuration for compiler optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerConfig {
    /// Optimization level to apply
    pub optimization_level: OptimizationLevel,
    /// Enable JIT compilation
    pub enable_jit: bool,
    /// Enable kernel fusion
    pub enable_fusion: bool,
    /// Enable graph optimizations
    pub enable_graph_opts: bool,
    /// Enable MLIR backend
    pub enable_mlir: bool,
    /// Target hardware characteristics
    pub target_hardware: HardwareTarget,
    /// Maximum compilation time in seconds (0 = no limit)
    pub max_compile_time: u64,
    /// Cache compiled kernels
    pub enable_cache: bool,
    /// Additional compiler flags
    pub compiler_flags: Vec<String>,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Standard,
            enable_jit: true,
            enable_fusion: true,
            enable_graph_opts: true,
            enable_mlir: false, // Experimental
            target_hardware: HardwareTarget::default(),
            max_compile_time: 300, // 5 minutes
            enable_cache: true,
            compiler_flags: Vec::new(),
        }
    }
}

/// Target hardware characteristics for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareTarget {
    /// Target device type
    pub device_type: DeviceType,
    /// Number of compute units (cores, SMs, etc.)
    pub compute_units: u32,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth: f64,
    /// Cache sizes in bytes
    pub cache_sizes: Vec<u64>,
    /// Supports specific instruction sets
    pub instruction_sets: Vec<String>,
}

impl Default for HardwareTarget {
    fn default() -> Self {
        Self {
            device_type: DeviceType::CPU,
            compute_units: 8,
            memory_bandwidth: 100.0,
            cache_sizes: vec![32768, 262144, 8388608], // L1, L2, L3
            instruction_sets: vec!["AVX2".to_string(), "FMA".to_string()],
        }
    }
}

/// Device type for optimization targeting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    CPU,
    GPU,
    TPU,
    DSP,
    FPGA,
    Custom(u32),
}

/// Compilation statistics and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationStats {
    /// Total compilation time in milliseconds
    pub compilation_time_ms: u64,
    /// Number of operations in original graph
    pub original_ops: usize,
    /// Number of operations after optimization
    pub optimized_ops: usize,
    /// Number of kernels fused
    pub fused_kernels: usize,
    /// Estimated performance improvement
    pub performance_gain: f64,
    /// Memory usage reduction
    pub memory_reduction: f64,
    /// Applied optimization passes
    pub applied_passes: Vec<String>,
}

/// Main compiler optimizer interface
pub struct CompilerOptimizer {
    config: CompilerConfig,
    graph_optimizer: graph_optimizer::GraphOptimizer,
    jit_compiler: jit_compiler::JitCompiler,
    kernel_fusion: kernel_fusion::KernelFusion,
    mlir_backend: Option<mlir_backend::MlirBackend>,
    graph_analyzer: analysis::GraphAnalyzer,
    pass_manager: passes::PassManager,
    compilation_cache: HashMap<String, Vec<u8>>,
}

impl CompilerOptimizer {
    /// Create a new compiler optimizer with the given configuration
    pub fn new(config: CompilerConfig) -> Result<Self, TrustformersError> {
        let graph_optimizer = graph_optimizer::GraphOptimizer::new(&config)?;
        let jit_compiler = jit_compiler::JitCompiler::new(&config)?;
        let kernel_fusion = kernel_fusion::KernelFusion::new(&config)?;
        let mlir_backend = if config.enable_mlir {
            Some(mlir_backend::MlirBackend::new(&config)?)
        } else {
            None
        };

        let graph_analyzer = analysis::GraphAnalyzer::new(config.target_hardware.clone());
        let pass_manager = match config.optimization_level {
            OptimizationLevel::None => passes::PassManager::new(),
            OptimizationLevel::Basic | OptimizationLevel::Standard => {
                passes::PassManager::default_pipeline()
            },
            OptimizationLevel::Aggressive | OptimizationLevel::Maximum => {
                passes::PassManager::aggressive_pipeline()
            },
        };

        Ok(Self {
            config,
            graph_optimizer,
            jit_compiler,
            kernel_fusion,
            mlir_backend,
            graph_analyzer,
            pass_manager,
            compilation_cache: HashMap::new(),
        })
    }

    /// Create a new compiler optimizer with a specific optimization level
    pub fn with_optimization_level(level: OptimizationLevel) -> Result<Self, TrustformersError> {
        let config = CompilerConfig {
            optimization_level: level,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Get the current configuration
    pub fn config(&self) -> &CompilerConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: CompilerConfig) -> Result<(), TrustformersError> {
        self.config = config;
        self.graph_optimizer.update_config(&self.config)?;
        self.jit_compiler.update_config(&self.config)?;
        self.kernel_fusion.update_config(&self.config)?;
        if let Some(ref mut mlir) = self.mlir_backend {
            mlir.update_config(&self.config)?;
        }
        Ok(())
    }

    /// Clear the compilation cache
    pub fn clear_cache(&mut self) {
        self.compilation_cache.clear();
        self.jit_compiler.clear_cache();
        if let Some(ref mut mlir) = self.mlir_backend {
            mlir.clear_cache();
        }
    }

    /// Get compilation cache statistics
    pub fn cache_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("cache_entries".to_string(), self.compilation_cache.len());
        stats.insert(
            "jit_cache_entries".to_string(),
            self.jit_compiler.cache_size(),
        );
        if let Some(ref mlir) = self.mlir_backend {
            stats.insert("mlir_cache_entries".to_string(), mlir.cache_size());
        }
        stats
    }

    /// Optimize a computation graph using all enabled optimizations
    pub fn optimize_graph(
        &mut self,
        mut graph: ComputationGraph,
    ) -> Result<OptimizationResult, TrustformersError> {
        let start_time = std::time::Instant::now();
        let original_ops = graph.nodes.len();
        let original_compute_cost = graph.total_compute_cost();
        let original_memory_cost = graph.total_memory_cost();

        // Apply optimization passes
        let pass_results = if self.config.enable_graph_opts {
            self.pass_manager.run(&mut graph)?
        } else {
            Vec::new()
        };

        // Apply kernel fusion
        let fusion_result = if self.config.enable_fusion {
            self.kernel_fusion.apply_fusion(&mut graph)?
        } else {
            kernel_fusion::FusionResult {
                fused_operations: 0,
                estimated_speedup: 1.0,
                fusion_time_ms: 0,
                applied_patterns: Vec::new(),
            }
        };

        let optimized_ops = graph.nodes.len();
        let optimized_compute_cost = graph.total_compute_cost();
        let optimized_memory_cost = graph.total_memory_cost();

        let optimization_time = start_time.elapsed();

        // Calculate improvements
        let compute_improvement = if original_compute_cost > 0.0 {
            (original_compute_cost - optimized_compute_cost) / original_compute_cost
        } else {
            0.0
        };

        let memory_improvement = if original_memory_cost > 0.0 {
            (original_memory_cost - optimized_memory_cost) / original_memory_cost
        } else {
            0.0
        };

        let applied_passes: Vec<String> = pass_results
            .iter()
            .enumerate()
            .filter(|(_, result)| result.changed)
            .map(|(i, _)| format!("pass_{}", i))
            .collect();

        Ok(OptimizationResult {
            optimized_graph: graph,
            original_operations: original_ops,
            optimized_operations: optimized_ops,
            fused_operations: fusion_result.fused_operations,
            compute_improvement,
            memory_improvement,
            estimated_speedup: fusion_result.estimated_speedup,
            optimization_time_ms: optimization_time.as_millis() as u64,
            applied_passes,
            fusion_patterns: fusion_result.applied_patterns,
        })
    }

    /// Compile an optimized graph to executable code
    pub fn compile_graph(
        &mut self,
        graph: ComputationGraph,
    ) -> Result<CompilationResult, TrustformersError> {
        if self.config.enable_jit {
            let result = self.jit_compiler.compile(graph)?;
            Ok(result)
        } else {
            // Fallback to basic compilation
            let stats = CompilationStats {
                compilation_time_ms: 0,
                original_ops: graph.nodes.len(),
                optimized_ops: graph.nodes.len(),
                fused_kernels: 0,
                performance_gain: 1.0,
                memory_reduction: 0.0,
                applied_passes: vec!["basic".to_string()],
            };

            Ok(CompilationResult {
                compiled_code: vec![0u8; 64], // Placeholder
                stats,
                metadata: HashMap::new(),
            })
        }
    }

    /// Perform comprehensive performance analysis
    pub fn analyze_performance(
        &mut self,
        graph: &ComputationGraph,
    ) -> Result<analysis::PerformanceAnalysis, TrustformersError> {
        self.graph_analyzer.analyze_performance(graph)
    }

    /// Perform memory usage analysis
    pub fn analyze_memory(
        &mut self,
        graph: &ComputationGraph,
    ) -> Result<analysis::MemoryAnalysis, TrustformersError> {
        self.graph_analyzer.analyze_memory(graph)
    }

    /// Perform dependency analysis
    pub fn analyze_dependencies(
        &mut self,
        graph: &ComputationGraph,
    ) -> Result<analysis::DependencyAnalysis, TrustformersError> {
        self.graph_analyzer.analyze_dependencies(graph)
    }

    /// Generate optimization recommendations for a graph
    pub fn recommend_optimizations(
        &mut self,
        graph: &ComputationGraph,
    ) -> Result<OptimizationRecommendations, TrustformersError> {
        let perf_analysis = self.analyze_performance(graph)?;
        let memory_analysis = self.analyze_memory(graph)?;

        let mut recommendations = Vec::new();

        // Performance-based recommendations
        for bottleneck in &perf_analysis.bottlenecks {
            if bottleneck.criticality_score > 50.0 {
                recommendations.push(OptimizationRecommendation {
                    category: RecommendationCategory::Performance,
                    priority: RecommendationPriority::High,
                    description: format!(
                        "Optimize {} operation (node {}) - {}% of total time",
                        bottleneck.operation_type, bottleneck.node_id, bottleneck.criticality_score
                    ),
                    suggested_actions: bottleneck.optimization_suggestions.clone(),
                    estimated_benefit: bottleneck.criticality_score / 100.0,
                });
            }
        }

        // Memory-based recommendations
        if memory_analysis.peak_memory_usage > 8 * 1024 * 1024 * 1024 {
            // > 8GB
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Memory,
                priority: RecommendationPriority::Medium,
                description: "High memory usage detected - consider memory optimization"
                    .to_string(),
                suggested_actions: vec![
                    "Enable gradient checkpointing".to_string(),
                    "Use mixed precision training".to_string(),
                    "Consider model parallelism".to_string(),
                ],
                estimated_benefit: 0.3,
            });
        }

        // Parallelization recommendations
        if perf_analysis.parallelizable_operations.len() > 5 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Parallelization,
                priority: RecommendationPriority::Medium,
                description: format!(
                    "Found {} parallelizable operation groups",
                    perf_analysis.parallelizable_operations.len()
                ),
                suggested_actions: vec![
                    "Enable multi-threading".to_string(),
                    "Consider GPU acceleration".to_string(),
                    "Use parallel execution backends".to_string(),
                ],
                estimated_benefit: 0.4,
            });
        }

        // Hardware utilization recommendations
        if perf_analysis.hardware_utilization.compute_utilization < 0.5 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Hardware,
                priority: RecommendationPriority::Low,
                description: "Low compute utilization detected".to_string(),
                suggested_actions: vec![
                    "Increase batch size".to_string(),
                    "Enable operation fusion".to_string(),
                    "Consider different hardware targets".to_string(),
                ],
                estimated_benefit: 0.2,
            });
        }

        // Sort by priority and benefit
        recommendations.sort_by(|a, b| match (a.priority.clone(), b.priority.clone()) {
            (RecommendationPriority::High, RecommendationPriority::High) => b
                .estimated_benefit
                .partial_cmp(&a.estimated_benefit)
                .unwrap_or(std::cmp::Ordering::Equal),
            (RecommendationPriority::High, _) => std::cmp::Ordering::Less,
            (_, RecommendationPriority::High) => std::cmp::Ordering::Greater,
            _ => b
                .estimated_benefit
                .partial_cmp(&a.estimated_benefit)
                .unwrap_or(std::cmp::Ordering::Equal),
        });

        Ok(OptimizationRecommendations {
            recommendations,
            overall_score: self.calculate_optimization_score(graph)?,
            target_hardware: self.config.target_hardware.clone(),
        })
    }

    /// Calculate an overall optimization score for the graph
    fn calculate_optimization_score(
        &mut self,
        graph: &ComputationGraph,
    ) -> Result<f64, TrustformersError> {
        let perf_analysis = self.analyze_performance(graph)?;

        // Combine various metrics into a single score (0-100)
        let utilization_score = perf_analysis.hardware_utilization.compute_utilization * 25.0;
        let balance_score = perf_analysis.load_balance_score * 25.0;
        let parallel_score = perf_analysis.hardware_utilization.parallel_efficiency * 25.0;
        let memory_score =
            (1.0 - perf_analysis.hardware_utilization.memory_utilization.min(1.0)) * 25.0;

        Ok(utilization_score + balance_score + parallel_score + memory_score)
    }

    /// Get comprehensive compiler statistics
    pub fn get_comprehensive_stats(&self) -> CompilerStatistics {
        CompilerStatistics {
            jit_stats: self.jit_compiler.get_stats().clone(),
            fusion_stats: self.kernel_fusion.get_stats().clone(),
            cache_stats: self.cache_stats(),
            config: self.config.clone(),
        }
    }
}

/// Result of compilation process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationResult {
    /// Compiled bytecode or machine code
    pub compiled_code: Vec<u8>,
    /// Compilation statistics
    pub stats: CompilationStats,
    /// Optimization metadata
    pub metadata: HashMap<String, String>,
}

/// Optimization pass result
#[derive(Debug)]
pub struct PassResult {
    /// Whether the pass made changes
    pub changed: bool,
    /// Statistics about the pass
    pub stats: HashMap<String, f64>,
    /// Pass-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Simplified computation graph representation for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationGraph {
    /// Graph nodes (operations)
    pub nodes: Vec<GraphNode>,
    /// Graph edges (data dependencies)
    pub edges: Vec<GraphEdge>,
    /// Graph metadata
    pub metadata: HashMap<String, String>,
}

/// Graph node representing an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node ID
    pub id: usize,
    /// Operation type
    pub op_type: String,
    /// Node attributes
    pub attributes: HashMap<String, String>,
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shapes
    pub output_shapes: Vec<Vec<usize>>,
    /// Estimated computation cost
    pub compute_cost: f64,
    /// Estimated memory cost
    pub memory_cost: f64,
}

/// Graph edge representing data flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub from: usize,
    /// Destination node ID
    pub to: usize,
    /// Output index from source
    pub output_idx: usize,
    /// Input index to destination
    pub input_idx: usize,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
}

impl ComputationGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.push(edge);
    }

    /// Get node by ID
    pub fn get_node(&self, id: usize) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// Get mutable node by ID
    pub fn get_node_mut(&mut self, id: usize) -> Option<&mut GraphNode> {
        self.nodes.get_mut(id)
    }

    /// Get all edges connected to a node
    pub fn get_node_edges(&self, node_id: usize) -> Vec<&GraphEdge> {
        self.edges
            .iter()
            .filter(|edge| edge.from == node_id || edge.to == node_id)
            .collect()
    }

    /// Validate the graph structure
    pub fn validate(&self) -> Result<(), TrustformersError> {
        // Check for invalid node references in edges
        for edge in &self.edges {
            if edge.from >= self.nodes.len() || edge.to >= self.nodes.len() {
                return Err(invalid_input("Edge references non-existent node"));
            }
        }

        // Check for cycles (simplified)
        if self.has_cycles() {
            return Err(invalid_input("Graph contains cycles"));
        }

        Ok(())
    }

    /// Check if the graph has cycles (simplified DFS)
    fn has_cycles(&self) -> bool {
        let mut visited = vec![false; self.nodes.len()];
        let mut rec_stack = vec![false; self.nodes.len()];

        for i in 0..self.nodes.len() {
            if !visited[i] && self.dfs_has_cycle(i, &mut visited, &mut rec_stack) {
                return true;
            }
        }
        false
    }

    fn dfs_has_cycle(&self, node: usize, visited: &mut [bool], rec_stack: &mut [bool]) -> bool {
        visited[node] = true;
        rec_stack[node] = true;

        for edge in &self.edges {
            if edge.from == node {
                let next = edge.to;
                if !visited[next] && self.dfs_has_cycle(next, visited, rec_stack) {
                    return true;
                }
                if rec_stack[next] {
                    return true;
                }
            }
        }

        rec_stack[node] = false;
        false
    }

    /// Calculate total estimated compute cost
    pub fn total_compute_cost(&self) -> f64 {
        self.nodes.iter().map(|node| node.compute_cost).sum()
    }

    /// Calculate total estimated memory cost
    pub fn total_memory_cost(&self) -> f64 {
        self.nodes.iter().map(|node| node.memory_cost).sum()
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of graph optimization process
#[derive(Debug)]
pub struct OptimizationResult {
    /// The optimized computation graph
    pub optimized_graph: ComputationGraph,
    /// Number of operations in original graph
    pub original_operations: usize,
    /// Number of operations after optimization
    pub optimized_operations: usize,
    /// Number of operations that were fused
    pub fused_operations: usize,
    /// Compute cost improvement (0.0 to 1.0)
    pub compute_improvement: f64,
    /// Memory cost improvement (0.0 to 1.0)
    pub memory_improvement: f64,
    /// Estimated overall speedup
    pub estimated_speedup: f64,
    /// Time spent on optimization in milliseconds
    pub optimization_time_ms: u64,
    /// List of applied optimization passes
    pub applied_passes: Vec<String>,
    /// List of applied fusion patterns
    pub fusion_patterns: Vec<String>,
}

/// Comprehensive compiler statistics
#[derive(Debug)]
pub struct CompilerStatistics {
    /// JIT compiler statistics
    pub jit_stats: jit_compiler::CompilationStatistics,
    /// Kernel fusion statistics
    pub fusion_stats: kernel_fusion::FusionStatistics,
    /// Cache usage statistics
    pub cache_stats: HashMap<String, usize>,
    /// Current configuration
    pub config: CompilerConfig,
}

/// Optimization recommendations for a computation graph
#[derive(Debug)]
pub struct OptimizationRecommendations {
    /// List of specific recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Overall optimization score (0-100)
    pub overall_score: f64,
    /// Target hardware configuration
    pub target_hardware: HardwareTarget,
}

/// Individual optimization recommendation
#[derive(Debug)]
pub struct OptimizationRecommendation {
    /// Category of the recommendation
    pub category: RecommendationCategory,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Human-readable description
    pub description: String,
    /// List of suggested actions
    pub suggested_actions: Vec<String>,
    /// Estimated benefit (0.0 to 1.0)
    pub estimated_benefit: f64,
}

/// Categories of optimization recommendations
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationCategory {
    Performance,
    Memory,
    Parallelization,
    Hardware,
    Compilation,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationPriority {
    High,
    Medium,
    Low,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_config_default() {
        let config = CompilerConfig::default();
        assert_eq!(config.optimization_level, OptimizationLevel::Standard);
        assert!(config.enable_jit);
        assert!(config.enable_fusion);
        assert!(config.enable_graph_opts);
    }

    #[test]
    fn test_optimization_levels() {
        assert_ne!(OptimizationLevel::None, OptimizationLevel::Maximum);
        assert_eq!(OptimizationLevel::default(), OptimizationLevel::Standard);
    }

    #[test]
    fn test_computation_graph_basic() {
        let mut graph = ComputationGraph::new();

        let node1 = GraphNode {
            id: 0,
            op_type: "MatMul".to_string(),
            attributes: HashMap::new(),
            input_shapes: vec![vec![128, 256], vec![256, 512]],
            output_shapes: vec![vec![128, 512]],
            compute_cost: 100.0,
            memory_cost: 50.0,
        };

        let node2 = GraphNode {
            id: 1,
            op_type: "ReLU".to_string(),
            attributes: HashMap::new(),
            input_shapes: vec![vec![128, 512]],
            output_shapes: vec![vec![128, 512]],
            compute_cost: 10.0,
            memory_cost: 5.0,
        };

        graph.add_node(node1);
        graph.add_node(node2);

        let edge = GraphEdge {
            from: 0,
            to: 1,
            output_idx: 0,
            input_idx: 0,
            shape: vec![128, 512],
            dtype: "f32".to_string(),
        };

        graph.add_edge(edge);

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.total_compute_cost(), 110.0);
        assert_eq!(graph.total_memory_cost(), 55.0);

        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_compiler_optimizer_creation() {
        let config = CompilerConfig::default();
        let result = CompilerOptimizer::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hardware_target_default() {
        let target = HardwareTarget::default();
        assert_eq!(target.device_type, DeviceType::CPU);
        assert_eq!(target.compute_units, 8);
        assert!(target.memory_bandwidth > 0.0);
    }
}
