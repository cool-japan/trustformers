//! Computation graph analysis tools for debugging deep learning models.
//!
//! This module provides comprehensive analysis tools for computation graphs,
//! including node analysis, dependency tracking, optimization opportunities,
//! bottleneck detection, and graph visualization capabilities.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use uuid::Uuid;

/// Represents a computation graph for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationGraph {
    /// Unique identifier for this graph
    pub id: Uuid,
    /// Map of node ID to node information
    pub nodes: HashMap<String, GraphNode>,
    /// Adjacency list representing edges (dependencies)
    pub edges: HashMap<String, Vec<String>>,
    /// Root nodes (inputs to the computation)
    pub root_nodes: HashSet<String>,
    /// Leaf nodes (outputs of the computation)
    pub leaf_nodes: HashSet<String>,
    /// Metadata about the graph
    pub metadata: GraphMetadata,
}

/// Metadata about the computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// Name of the model/graph
    pub name: String,
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// Maximum depth of the graph
    pub max_depth: usize,
    /// Memory usage estimate in bytes
    pub estimated_memory_usage: u64,
    /// FLOP count estimate
    pub estimated_flops: u64,
    /// Timestamp when graph was created
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Represents a single node in the computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique identifier for this node
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Type of operation (e.g., "MatMul", "Add", "ReLU")
    pub operation_type: OperationType,
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shapes
    pub output_shapes: Vec<Vec<usize>>,
    /// Computational complexity (FLOPs)
    pub flop_count: u64,
    /// Memory usage estimate in bytes
    pub memory_usage: u64,
    /// Execution time in microseconds (if profiled)
    pub execution_time_us: Option<u64>,
    /// Number of parameters (for parameterized operations)
    pub parameter_count: Option<u64>,
    /// Position in topological ordering
    pub topo_order: Option<usize>,
    /// Depth in the graph (distance from inputs)
    pub depth: usize,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of operations in the computation graph
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    // Arithmetic operations
    Add,
    Subtract,
    Multiply,
    Divide,
    MatMul,
    Dot,

    // Activation functions
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    Softmax,

    // Normalization
    LayerNorm,
    BatchNorm,
    RMSNorm,

    // Convolution operations
    Conv1D,
    Conv2D,
    Conv3D,
    ConvTranspose,

    // Pooling operations
    MaxPool,
    AvgPool,
    AdaptivePool,

    // Tensor operations
    Reshape,
    Transpose,
    Concat,
    Split,
    Slice,
    Gather,
    Scatter,

    // Reduction operations
    Sum,
    Mean,
    Max,
    Min,

    // Attention operations
    Attention,
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,

    // Embedding operations
    Embedding,
    PositionalEmbedding,

    // Loss functions
    CrossEntropyLoss,
    MSELoss,
    L1Loss,

    // Control flow
    If,
    While,
    Loop,

    // Custom operations
    Custom(String),
}

/// Configuration for computation graph analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalysisConfig {
    /// Whether to perform memory analysis
    pub enable_memory_analysis: bool,
    /// Whether to perform FLOP analysis
    pub enable_flop_analysis: bool,
    /// Whether to detect optimization opportunities
    pub enable_optimization_analysis: bool,
    /// Whether to perform bottleneck detection
    pub enable_bottleneck_detection: bool,
    /// Whether to analyze data flow patterns
    pub enable_dataflow_analysis: bool,
    /// Threshold for considering a node a bottleneck (microseconds)
    pub bottleneck_threshold_us: u64,
    /// Memory threshold for large operations (bytes)
    pub large_memory_threshold: u64,
}

impl Default for GraphAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_memory_analysis: true,
            enable_flop_analysis: true,
            enable_optimization_analysis: true,
            enable_bottleneck_detection: true,
            enable_dataflow_analysis: true,
            bottleneck_threshold_us: 1000,             // 1ms
            large_memory_threshold: 1024 * 1024 * 100, // 100MB
        }
    }
}

/// Main computation graph analyzer
#[derive(Debug)]
pub struct ComputationGraphAnalyzer {
    config: GraphAnalysisConfig,
    graphs: HashMap<Uuid, ComputationGraph>,
    analysis_results: HashMap<Uuid, GraphAnalysisResult>,
}

/// Comprehensive analysis result for a computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalysisResult {
    /// Graph being analyzed
    pub graph_id: Uuid,
    /// Memory analysis results
    pub memory_analysis: Option<MemoryAnalysis>,
    /// FLOP analysis results
    pub flop_analysis: Option<FlopAnalysis>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    /// Bottleneck analysis
    pub bottleneck_analysis: Option<BottleneckAnalysis>,
    /// Data flow analysis
    pub dataflow_analysis: Option<DataFlowAnalysis>,
    /// Critical path analysis
    pub critical_path: Vec<String>,
    /// Graph statistics
    pub statistics: GraphStatistics,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Memory usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    /// Total memory usage in bytes
    pub total_memory_usage: u64,
    /// Peak simultaneously-live memory usage in bytes, computed by a real
    /// liveness walk over the graph's topological order (see
    /// [`ComputationGraphAnalyzer::compute_peak_memory_usage`]): a node's
    /// output is "live" from the step it is produced until the step of its
    /// last consumer, and this is the maximum total live bytes at any one
    /// step. Never equal to `total_memory_usage` by construction (as the
    /// old placeholder was) unless every tensor really is live
    /// simultaneously.
    pub peak_memory_usage: u64,
    /// Memory usage by operation type
    pub memory_by_operation: HashMap<OperationType, u64>,
    /// Nodes with highest memory usage
    pub memory_hotspots: Vec<(String, u64)>,
    /// Memory fragmentation ratio, when measurable. This analyzer tracks
    /// only logical per-node byte counts, not a real memory
    /// allocator/placement model (address ranges, allocation order,
    /// free-list state) -- fragmentation is a property of *how* an
    /// allocator places live tensors in physical memory, which is a
    /// different question from liveness overlap (already captured by
    /// [`Self::peak_memory_usage`]) and depends on a placement policy this
    /// crate does not implement. Honestly `None` rather than a fabricated
    /// number.
    pub fragmentation_ratio: Option<f64>,
    /// Suggested memory optimizations
    pub optimization_suggestions: Vec<String>,
}

/// FLOP (Floating Point Operations) analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlopAnalysis {
    /// Total FLOP count
    pub total_flops: u64,
    /// FLOP count by operation type
    pub flops_by_operation: HashMap<OperationType, u64>,
    /// Nodes with highest FLOP count
    pub compute_hotspots: Vec<(String, u64)>,
    /// Arithmetic intensity (FLOPs per byte)
    pub arithmetic_intensity: f64,
    /// Computational complexity analysis
    pub complexity_analysis: ComplexityAnalysis,
}

/// Complexity analysis of the computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityAnalysis {
    /// Asymptotic (Big-O) time complexity, when it can honestly be
    /// derived. A single [`ComputationGraph`] is one concrete, fixed-shape
    /// instance -- it has no symbolic size parameter `n` to be asymptotic
    /// *in*, so there is nothing to honestly derive here today; always
    /// `None`, never a fabricated `"O(n)"`. Concrete costs for the actual
    /// instance are available for real via [`FlopAnalysis::total_flops`].
    pub time_complexity: Option<String>,
    /// Asymptotic (Big-O) space complexity. Same honesty caveat as
    /// [`Self::time_complexity`]; concrete space for this instance is
    /// available for real via [`MemoryAnalysis::total_memory_usage`].
    pub space_complexity: Option<String>,
    /// Real, structural parallelization-potential estimate in `[0, 1]`:
    /// `1 - (critical_path_length_in_nodes / node_count)`, i.e. the
    /// fraction of nodes that are *not* on the graph's longest
    /// dependency chain and could in principle execute alongside it. `0.0`
    /// for a pure sequential chain (every node is on the critical path),
    /// approaching `1.0` for a wide, shallow graph. Computed from the
    /// graph's real topology (see
    /// [`ComputationGraphAnalyzer::analyze_flop_usage`]) -- never the old
    /// constant `0.7`.
    pub parallelization_potential: f64,
    /// Sequential dependencies
    pub sequential_dependencies: usize,
}

/// Optimization opportunity detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Description of the opportunity
    pub description: String,
    /// Nodes involved in this optimization
    pub affected_nodes: Vec<String>,
    /// Estimated performance improvement
    pub estimated_improvement: EstimatedImprovement,
    /// Implementation difficulty (1-5)
    pub implementation_difficulty: u8,
    /// Priority level
    pub priority: OptimizationPriority,
}

/// Types of optimizations that can be applied
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Fuse multiple operations into one
    OperationFusion,
    /// Eliminate redundant computations
    RedundancyElimination,
    /// Optimize memory layout
    MemoryLayoutOptimization,
    /// Use more efficient algorithms
    AlgorithmicOptimization,
    /// Parallelize sequential operations
    Parallelization,
    /// Optimize data access patterns
    DataAccessOptimization,
    /// Reduce precision where safe
    PrecisionOptimization,
    /// Cache intermediate results
    Memoization,
    /// Optimize control flow
    ControlFlowOptimization,
}

/// Priority levels for optimizations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Estimated improvement from an optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatedImprovement {
    /// Estimated speedup (multiplicative factor)
    pub speedup_factor: f64,
    /// Estimated memory reduction in bytes
    pub memory_reduction: u64,
    /// Estimated energy savings (0.0 to 1.0)
    pub energy_savings: f64,
}

/// Bottleneck analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    /// Nodes that are bottlenecks
    pub bottleneck_nodes: Vec<String>,
    /// Critical path through the graph
    pub critical_path_nodes: Vec<String>,
    /// Total critical path time
    pub critical_path_time_us: u64,
    /// Nodes that could benefit from parallelization
    pub parallelizable_nodes: Vec<String>,
    /// Scheduling suggestions
    pub scheduling_suggestions: Vec<String>,
}

/// Data flow analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowAnalysis {
    /// Data dependencies between nodes
    pub data_dependencies: HashMap<String, Vec<String>>,
    /// Live variables at each node
    pub live_variables: HashMap<String, HashSet<String>>,
    /// Variable lifetime analysis
    pub variable_lifetimes: HashMap<String, VariableLifetime>,
    /// Memory reuse opportunities
    pub memory_reuse_opportunities: Vec<MemoryReuseOpportunity>,
}

/// Lifetime information for a variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableLifetime {
    /// Node where variable is created
    pub birth_node: String,
    /// Node where variable is last used
    pub death_node: String,
    /// All nodes that use this variable
    pub usage_nodes: Vec<String>,
    /// Memory footprint in bytes
    pub memory_footprint: u64,
}

/// Memory reuse opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReuseOpportunity {
    /// Variables that can share memory
    pub reusable_variables: Vec<String>,
    /// Memory that can be saved
    pub memory_savings: u64,
    /// Implementation complexity
    pub complexity: u8,
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Number of nodes by operation type
    pub nodes_by_type: HashMap<OperationType, usize>,
    /// Average node fan-in
    pub average_fan_in: f64,
    /// Average node fan-out
    pub average_fan_out: f64,
    /// Graph diameter (longest shortest path)
    pub diameter: usize,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Number of strongly connected components
    pub strongly_connected_components: usize,
}

impl ComputationGraphAnalyzer {
    /// Create a new computation graph analyzer
    pub fn new(config: GraphAnalysisConfig) -> Self {
        Self {
            config,
            graphs: HashMap::new(),
            analysis_results: HashMap::new(),
        }
    }

    /// Add a computation graph for analysis
    pub fn add_graph(&mut self, graph: ComputationGraph) -> Result<()> {
        let graph_id = graph.id;
        self.graphs.insert(graph_id, graph);
        Ok(())
    }

    /// Create a computation graph from operations
    pub fn create_graph(
        &mut self,
        name: String,
        operations: Vec<(String, OperationType, Vec<String>)>, // (node_id, op_type, dependencies)
    ) -> Result<Uuid> {
        let graph_id = Uuid::new_v4();
        let mut nodes = HashMap::new();
        let mut edges = HashMap::new();
        let mut root_nodes = HashSet::new();
        let mut leaf_nodes = HashSet::new();

        // Create nodes
        for (node_id, op_type, dependencies) in &operations {
            let node = GraphNode {
                id: node_id.clone(),
                name: node_id.clone(),
                operation_type: op_type.clone(),
                input_shapes: vec![],
                output_shapes: vec![],
                flop_count: self.estimate_flops(op_type, &[]),
                memory_usage: self.estimate_memory(op_type, &[]),
                execution_time_us: None,
                parameter_count: self.estimate_parameters(op_type),
                topo_order: None,
                depth: 0,
                metadata: HashMap::new(),
            };
            nodes.insert(node_id.clone(), node);

            // Track dependencies
            if dependencies.is_empty() {
                root_nodes.insert(node_id.clone());
            }
            edges.insert(node_id.clone(), dependencies.clone());
        }

        // Identify leaf nodes
        let all_dependencies: HashSet<String> = edges.values().flatten().cloned().collect();
        for node_id in nodes.keys() {
            if !all_dependencies.contains(node_id) {
                leaf_nodes.insert(node_id.clone());
            }
        }

        // Calculate depth and topological order
        self.calculate_depth_and_topo_order(&mut nodes, &edges)?;

        let metadata = GraphMetadata {
            name,
            node_count: nodes.len(),
            edge_count: edges.values().map(|deps| deps.len()).sum(),
            max_depth: nodes.values().map(|n| n.depth).max().unwrap_or(0),
            estimated_memory_usage: nodes.values().map(|n| n.memory_usage).sum(),
            estimated_flops: nodes.values().map(|n| n.flop_count).sum(),
            created_at: chrono::Utc::now(),
        };

        let graph = ComputationGraph {
            id: graph_id,
            nodes,
            edges,
            root_nodes,
            leaf_nodes,
            metadata,
        };

        self.graphs.insert(graph_id, graph);
        Ok(graph_id)
    }

    /// Analyze a computation graph
    pub fn analyze_graph(&mut self, graph_id: Uuid) -> Result<GraphAnalysisResult> {
        let graph = self
            .graphs
            .get(&graph_id)
            .ok_or_else(|| anyhow::anyhow!("Graph not found: {}", graph_id))?;

        let mut result = GraphAnalysisResult {
            graph_id,
            memory_analysis: None,
            flop_analysis: None,
            optimization_opportunities: Vec::new(),
            bottleneck_analysis: None,
            dataflow_analysis: None,
            critical_path: Vec::new(),
            statistics: self.calculate_statistics(graph)?,
            recommendations: Vec::new(),
        };

        // Perform different types of analysis based on configuration
        if self.config.enable_memory_analysis {
            result.memory_analysis = Some(self.analyze_memory_usage(graph)?);
        }

        if self.config.enable_flop_analysis {
            result.flop_analysis = Some(self.analyze_flop_usage(graph)?);
        }

        if self.config.enable_optimization_analysis {
            result.optimization_opportunities = self.detect_optimization_opportunities(graph)?;
        }

        if self.config.enable_bottleneck_detection {
            result.bottleneck_analysis = Some(self.analyze_bottlenecks(graph)?);
        }

        if self.config.enable_dataflow_analysis {
            result.dataflow_analysis = Some(self.analyze_dataflow(graph)?);
        }

        result.critical_path = self.find_critical_path(graph)?;
        result.recommendations = self.generate_recommendations(&result)?;

        self.analysis_results.insert(graph_id, result.clone());
        Ok(result)
    }

    /// Get analysis results for a graph
    pub fn get_analysis_result(&self, graph_id: Uuid) -> Option<&GraphAnalysisResult> {
        self.analysis_results.get(&graph_id)
    }

    /// Export graph analysis to DOT format for visualization
    pub fn export_to_dot(&self, graph_id: Uuid) -> Result<String> {
        let graph = self
            .graphs
            .get(&graph_id)
            .ok_or_else(|| anyhow::anyhow!("Graph not found: {}", graph_id))?;

        let mut dot = String::new();
        dot.push_str(&format!("digraph \"{}\" {{\n", graph.metadata.name));
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box, style=filled];\n\n");

        // Add nodes with styling based on operation type
        for node in graph.nodes.values() {
            let color = self.get_node_color(&node.operation_type);
            let label = format!(
                "{}\\n{}\\n{:.1} GFLOP\\n{:.1} MB",
                node.name,
                format!("{:?}", node.operation_type),
                node.flop_count as f64 / 1e9,
                node.memory_usage as f64 / (1024.0 * 1024.0)
            );

            dot.push_str(&format!(
                "  \"{}\" [label=\"{}\", fillcolor=\"{}\"];\n",
                node.id, label, color
            ));
        }

        dot.push('\n');

        // Add edges
        for (node_id, dependencies) in &graph.edges {
            for dep in dependencies {
                dot.push_str(&format!("  \"{}\" -> \"{}\";\n", dep, node_id));
            }
        }

        dot.push_str("}\n");
        Ok(dot)
    }

    // Private helper methods

    fn calculate_depth_and_topo_order(
        &self,
        nodes: &mut HashMap<String, GraphNode>,
        edges: &HashMap<String, Vec<String>>,
    ) -> Result<()> {
        // Topological sort and depth calculation
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adj_list: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize in-degrees and adjacency list
        for node_id in nodes.keys() {
            in_degree.insert(node_id.clone(), 0);
            adj_list.insert(node_id.clone(), Vec::new());
        }

        for (node_id, dependencies) in edges {
            in_degree.insert(node_id.clone(), dependencies.len());
            for dep in dependencies {
                if let Some(adj) = adj_list.get_mut(dep) {
                    adj.push(node_id.clone());
                }
            }
        }

        // Kahn's algorithm for topological sorting and depth calculation
        let mut queue = VecDeque::new();
        let mut topo_order = 0;

        // Find all nodes with no incoming edges
        for (node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back((node_id.clone(), 0)); // (node_id, depth)
            }
        }

        while let Some((node_id, depth)) = queue.pop_front() {
            // Update node
            if let Some(node) = nodes.get_mut(&node_id) {
                node.depth = depth;
                node.topo_order = Some(topo_order);
                topo_order += 1;
            }

            // Process neighbors
            if let Some(neighbors) = adj_list.get(&node_id) {
                for neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back((neighbor.clone(), depth + 1));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn estimate_flops(&self, op_type: &OperationType, shapes: &[Vec<usize>]) -> u64 {
        // Simplified FLOP estimation
        match op_type {
            OperationType::MatMul => {
                if shapes.len() >= 2 {
                    let a_shape = &shapes[0];
                    let b_shape = &shapes[1];
                    if a_shape.len() >= 2 && b_shape.len() >= 2 {
                        let m = a_shape[a_shape.len() - 2];
                        let k = a_shape[a_shape.len() - 1];
                        let n = b_shape[b_shape.len() - 1];
                        return (2 * m * k * n) as u64;
                    }
                }
                1000000 // Default estimate
            },
            OperationType::Add | OperationType::Subtract | OperationType::Multiply => {
                shapes.first().map(|s| s.iter().product::<usize>() as u64).unwrap_or(1000)
            },
            OperationType::ReLU | OperationType::Sigmoid | OperationType::Tanh => {
                shapes.first().map(|s| s.iter().product::<usize>() as u64).unwrap_or(1000)
            },
            OperationType::LayerNorm | OperationType::BatchNorm => {
                shapes.first().map(|s| (s.iter().product::<usize>() * 5) as u64).unwrap_or(5000)
            },
            _ => 1000, // Default estimate
        }
    }

    fn estimate_memory(&self, op_type: &OperationType, shapes: &[Vec<usize>]) -> u64 {
        // Simplified memory estimation (assuming float32 = 4 bytes)
        let element_size = 4u64;
        match op_type {
            OperationType::MatMul => {
                shapes
                    .iter()
                    .map(|s| s.iter().product::<usize>() as u64 * element_size)
                    .sum::<u64>()
                    .max(1024) // Minimum 1KB
            },
            _ => shapes
                .first()
                .map(|s| s.iter().product::<usize>() as u64 * element_size)
                .unwrap_or(1024),
        }
    }

    fn estimate_parameters(&self, op_type: &OperationType) -> Option<u64> {
        match op_type {
            OperationType::MatMul => Some(1000000), // Example: 1M parameters
            OperationType::Conv2D => Some(500000),
            OperationType::Embedding => Some(2000000),
            OperationType::LayerNorm => Some(1000),
            _ => None,
        }
    }

    fn analyze_memory_usage(&self, graph: &ComputationGraph) -> Result<MemoryAnalysis> {
        let total_memory_usage = graph.nodes.values().map(|n| n.memory_usage).sum();

        let mut memory_by_operation: HashMap<OperationType, u64> = HashMap::new();
        for node in graph.nodes.values() {
            *memory_by_operation.entry(node.operation_type.clone()).or_insert(0) +=
                node.memory_usage;
        }

        let mut memory_hotspots: Vec<(String, u64)> =
            graph.nodes.values().map(|n| (n.id.clone(), n.memory_usage)).collect();
        memory_hotspots.sort_by_key(|item| std::cmp::Reverse(item.1));
        memory_hotspots.truncate(10); // Top 10

        let peak_memory_usage = self.compute_peak_memory_usage(graph);
        // No real memory-allocator/placement model exists in this crate --
        // see the field's own doc comment. Honestly absent, not a
        // fabricated "10% fragmented" guess.
        let fragmentation_ratio = None;

        let optimization_suggestions = vec![
            "Consider memory pooling for frequently allocated tensors".to_string(),
            "Implement in-place operations where possible".to_string(),
            "Use gradient checkpointing for memory-intensive layers".to_string(),
        ];

        Ok(MemoryAnalysis {
            total_memory_usage,
            peak_memory_usage,
            memory_by_operation,
            memory_hotspots,
            fragmentation_ratio,
            optimization_suggestions,
        })
    }

    /// Real peak simultaneously-live memory, via a liveness walk over the
    /// graph's real topological order: at each node's execution step, its
    /// output becomes live; a dependency's output is freed the moment the
    /// *last* node (by topological position) that consumes it has
    /// executed -- except outputs in [`ComputationGraph::leaf_nodes`],
    /// which are the graph's own outputs and must stay live through the
    /// end. The result is the maximum total live bytes observed at any
    /// step; always `<= total_memory_usage` (equal only when nothing is
    /// ever freed, i.e. every tensor really is live simultaneously).
    fn compute_peak_memory_usage(&self, graph: &ComputationGraph) -> u64 {
        let mut ordered: Vec<&GraphNode> = graph.nodes.values().collect();
        ordered.sort_by_key(|n| n.topo_order.unwrap_or(usize::MAX));

        // For every node, the topological position of its LAST consumer --
        // the latest point at which its output is still needed as an
        // input.
        let mut last_use: HashMap<&str, usize> = HashMap::new();
        for node in &ordered {
            let Some(topo) = node.topo_order else {
                continue;
            };
            for dep in graph.edges.get(&node.id).into_iter().flatten() {
                last_use.entry(dep.as_str()).and_modify(|t| *t = (*t).max(topo)).or_insert(topo);
            }
        }

        let mut live: u64 = 0;
        let mut peak: u64 = 0;
        for node in &ordered {
            let Some(topo) = node.topo_order else {
                continue;
            };
            live = live.saturating_add(node.memory_usage);
            peak = peak.max(live);
            // Dedupe: an op can legitimately depend on the same upstream
            // node twice (e.g. `Multiply(x, x)`), which would otherwise
            // free `dep`'s memory once per OCCURRENCE in the edge list
            // instead of once per real tensor -- an artificial extra
            // free that could understate `live` (and therefore a LATER
            // step's peak) even though nothing changed.
            let unique_deps: HashSet<&str> =
                graph.edges.get(&node.id).into_iter().flatten().map(|s| s.as_str()).collect();
            for dep in unique_deps {
                let is_last_use = last_use.get(dep) == Some(&topo);
                if is_last_use && !graph.leaf_nodes.contains(dep) {
                    if let Some(dep_node) = graph.nodes.get(dep) {
                        live = live.saturating_sub(dep_node.memory_usage);
                    }
                }
            }
        }
        peak
    }

    fn analyze_flop_usage(&self, graph: &ComputationGraph) -> Result<FlopAnalysis> {
        let total_flops = graph.nodes.values().map(|n| n.flop_count).sum();

        let mut flops_by_operation: HashMap<OperationType, u64> = HashMap::new();
        for node in graph.nodes.values() {
            *flops_by_operation.entry(node.operation_type.clone()).or_insert(0) += node.flop_count;
        }

        let mut compute_hotspots: Vec<(String, u64)> =
            graph.nodes.values().map(|n| (n.id.clone(), n.flop_count)).collect();
        compute_hotspots.sort_by_key(|item| std::cmp::Reverse(item.1));
        compute_hotspots.truncate(10); // Top 10

        let total_memory = graph.nodes.values().map(|n| n.memory_usage).sum::<u64>();
        let arithmetic_intensity =
            if total_memory > 0 { total_flops as f64 / total_memory as f64 } else { 0.0 };

        let complexity_analysis = ComplexityAnalysis {
            // See the fields' own doc comments: a Big-O class is not
            // derivable from one concrete-shaped graph instance.
            time_complexity: None,
            space_complexity: None,
            parallelization_potential: self.compute_parallelization_potential(graph),
            sequential_dependencies: graph.metadata.max_depth,
        };

        Ok(FlopAnalysis {
            total_flops,
            flops_by_operation,
            compute_hotspots,
            arithmetic_intensity,
            complexity_analysis,
        })
    }

    /// Real, structural parallelization-potential estimate: `1 -
    /// span/work`, using UNIT per-node cost (i.e. "work" = node count,
    /// "span" = critical-path length in nodes = `max_depth + 1`) -- the
    /// classical work/span parallelism ratio from parallel-scheduling
    /// theory (Brent/Graham), consistent with
    /// [`ComplexityAnalysis::sequential_dependencies`] already being the
    /// same unweighted `max_depth`. `0.0` for a pure chain (span == work:
    /// every node is on the critical path, so nothing can run
    /// alongside it); approaches `1.0` for a wide, shallow graph where
    /// most nodes are off the critical path.
    ///
    /// Deliberately unweighted by FLOPs/bytes: those are per-node
    /// quantities that can vary by orders of magnitude between nodes, so
    /// a work-weighted version would let one disproportionately expensive
    /// node dominate the ratio and mislabel a structurally wide (many
    /// independent branches), maximally parallel graph as having "low"
    /// potential just because most of its FLOPs happen to sit on the
    /// critical path.
    fn compute_parallelization_potential(&self, graph: &ComputationGraph) -> f64 {
        let node_count = graph.nodes.len();
        if node_count == 0 {
            return 0.0;
        }
        let span = graph.metadata.max_depth + 1;
        (1.0 - span as f64 / node_count as f64).clamp(0.0, 1.0)
    }

    fn detect_optimization_opportunities(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<OptimizationOpportunity>> {
        let mut opportunities = Vec::new();

        // Look for operation fusion opportunities
        opportunities.extend(self.detect_fusion_opportunities(graph)?);

        // Look for redundant operations
        opportunities.extend(self.detect_redundancy_opportunities(graph)?);

        // Look for memory optimization opportunities
        opportunities.extend(self.detect_memory_optimizations(graph)?);

        Ok(opportunities)
    }

    fn detect_fusion_opportunities(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<OptimizationOpportunity>> {
        let mut opportunities = Vec::new();

        // Look for patterns like MatMul + Add (bias addition)
        for node in graph.nodes.values() {
            if let OperationType::Add = node.operation_type {
                let empty_deps = vec![];
                let dependencies = graph.edges.get(&node.id).unwrap_or(&empty_deps);
                for dep in dependencies {
                    if let Some(dep_node) = graph.nodes.get(dep) {
                        if let OperationType::MatMul = dep_node.operation_type {
                            opportunities.push(OptimizationOpportunity {
                                optimization_type: OptimizationType::OperationFusion,
                                description:
                                    "Fuse MatMul and Add operations into a single GEMM operation"
                                        .to_string(),
                                affected_nodes: vec![dep.clone(), node.id.clone()],
                                estimated_improvement: EstimatedImprovement {
                                    speedup_factor: 1.2,
                                    memory_reduction: 1024 * 1024, // 1MB
                                    energy_savings: 0.1,
                                },
                                implementation_difficulty: 2,
                                priority: OptimizationPriority::Medium,
                            });
                        }
                    }
                }
            }
        }

        Ok(opportunities)
    }

    /// Real common-subexpression detection: groups nodes by the exact
    /// `(operation_type, ordered dependency list)` signature they compute.
    /// Two internal nodes applying the *same operation* to the *same
    /// ordered list of upstream node ids* are, for a deterministic pure
    /// operation, computing an identical result -- all but one are fully
    /// redundant. Dependency order is kept significant (never sorted), so
    /// non-commutative operations (`Subtract`, `MatMul`, ...) are never
    /// falsely flagged as duplicates of each other with swapped operands.
    ///
    /// [`ComputationGraph::root_nodes`] (no dependencies at all) are
    /// excluded: an empty dependency list carries no proof that two roots
    /// hold the same external data -- e.g. two distinct model inputs may
    /// well share both an operation type and "no dependencies" without
    /// being remotely the same tensor.
    fn detect_redundancy_opportunities(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<OptimizationOpportunity>> {
        let empty_deps: Vec<String> = Vec::new();
        let mut signature_groups: HashMap<(&OperationType, &[String]), Vec<&str>> = HashMap::new();
        for node in graph.nodes.values() {
            let deps = graph.edges.get(&node.id).unwrap_or(&empty_deps);
            if deps.is_empty() {
                continue; // no real dependencies to prove equivalence from
            }
            signature_groups
                .entry((&node.operation_type, deps.as_slice()))
                .or_default()
                .push(node.id.as_str());
        }

        let mut opportunities = Vec::new();
        for ((op_type, deps), mut node_ids) in signature_groups {
            if node_ids.len() < 2 {
                continue;
            }
            node_ids.sort_unstable(); // deterministic report ordering

            let redundant_count = node_ids.len() - 1;
            let per_node_memory = node_ids
                .iter()
                .filter_map(|id| graph.nodes.get(*id))
                .map(|n| n.memory_usage)
                .max()
                .unwrap_or(0);

            opportunities.push(OptimizationOpportunity {
                optimization_type: OptimizationType::RedundancyElimination,
                description: format!(
                    "{} node(s) recompute the identical {} over the same {} input(s); keep one \
                     and reuse its output for the other {}",
                    node_ids.len(),
                    op_type,
                    deps.len(),
                    redundant_count,
                ),
                affected_nodes: node_ids.iter().map(|s| s.to_string()).collect(),
                estimated_improvement: EstimatedImprovement {
                    // This group's own work shrinks from `node_ids.len()`
                    // identical computations to 1 -- a real factor derived
                    // from the actual duplicate count, not an invented
                    // constant.
                    speedup_factor: node_ids.len() as f64,
                    memory_reduction: per_node_memory * redundant_count as u64,
                    energy_savings: (redundant_count as f64 / node_ids.len() as f64)
                        .clamp(0.0, 1.0),
                },
                implementation_difficulty: 2,
                priority: if redundant_count >= 3 {
                    OptimizationPriority::High
                } else {
                    OptimizationPriority::Medium
                },
            });
        }

        opportunities.sort_by(|a, b| a.affected_nodes.cmp(&b.affected_nodes));
        Ok(opportunities)
    }

    fn detect_memory_optimizations(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<OptimizationOpportunity>> {
        let mut opportunities = Vec::new();

        // Look for large memory operations
        for node in graph.nodes.values() {
            if node.memory_usage > self.config.large_memory_threshold {
                opportunities.push(OptimizationOpportunity {
                    optimization_type: OptimizationType::MemoryLayoutOptimization,
                    description: format!(
                        "Optimize memory layout for large operation: {}",
                        node.name
                    ),
                    affected_nodes: vec![node.id.clone()],
                    estimated_improvement: EstimatedImprovement {
                        speedup_factor: 1.1,
                        memory_reduction: node.memory_usage / 4, // 25% reduction
                        energy_savings: 0.05,
                    },
                    implementation_difficulty: 3,
                    priority: OptimizationPriority::Medium,
                });
            }
        }

        Ok(opportunities)
    }

    fn analyze_bottlenecks(&self, graph: &ComputationGraph) -> Result<BottleneckAnalysis> {
        let mut bottleneck_nodes = Vec::new();
        let mut parallelizable_nodes = Vec::new();

        for node in graph.nodes.values() {
            if let Some(exec_time) = node.execution_time_us {
                if exec_time > self.config.bottleneck_threshold_us {
                    bottleneck_nodes.push(node.id.clone());
                }
            }

            // Check if node can be parallelized (simplified heuristic)
            match node.operation_type {
                OperationType::MatMul | OperationType::Conv2D | OperationType::Add => {
                    parallelizable_nodes.push(node.id.clone());
                },
                _ => {},
            }
        }

        let critical_path_nodes = self.find_critical_path(graph)?;
        let critical_path_time_us = critical_path_nodes
            .iter()
            .filter_map(|id| graph.nodes.get(id))
            .filter_map(|node| node.execution_time_us)
            .sum();

        let scheduling_suggestions = vec![
            "Consider parallel execution of independent operations".to_string(),
            "Use asynchronous execution for I/O operations".to_string(),
            "Implement pipeline parallelism for sequential operations".to_string(),
        ];

        Ok(BottleneckAnalysis {
            bottleneck_nodes,
            critical_path_nodes,
            critical_path_time_us,
            parallelizable_nodes,
            scheduling_suggestions,
        })
    }

    fn analyze_dataflow(&self, graph: &ComputationGraph) -> Result<DataFlowAnalysis> {
        let mut data_dependencies = HashMap::new();
        let mut live_variables = HashMap::new();
        for (node_id, dependencies) in &graph.edges {
            data_dependencies.insert(node_id.clone(), dependencies.clone());
            live_variables.insert(node_id.clone(), dependencies.iter().cloned().collect());
        }

        // Real, deterministic variable lifetimes: each node's own OUTPUT
        // is one "variable", born when the node executes (its real
        // topological position) and alive until the topologically LAST
        // real consumer runs -- or, for a `leaf_nodes` member (the
        // graph's own published outputs), alive through the graph's end,
        // since nothing inside the graph marks when an external caller
        // is done reading it. Built from the real topo order (not
        // `graph.edges`' `HashMap` iteration order, which the previous
        // implementation used directly and which is not required to
        // reflect real execution sequence).
        let mut ordered: Vec<&GraphNode> = graph.nodes.values().collect();
        ordered.sort_by_key(|n| n.topo_order.unwrap_or(usize::MAX));

        let mut consumers_by_dep: HashMap<&str, Vec<(usize, &str)>> = HashMap::new();
        for node in &ordered {
            let Some(topo) = node.topo_order else {
                continue;
            };
            for dep in graph.edges.get(&node.id).into_iter().flatten() {
                consumers_by_dep.entry(dep.as_str()).or_default().push((topo, node.id.as_str()));
            }
        }
        let max_topo = ordered.iter().filter_map(|n| n.topo_order).max().unwrap_or(0);

        let mut variable_lifetimes = HashMap::new();
        let mut birth_death_topo: HashMap<&str, (usize, usize)> = HashMap::new();
        for node in &ordered {
            let Some(birth_topo) = node.topo_order else {
                continue;
            };
            let mut consumers = consumers_by_dep.get(node.id.as_str()).cloned().unwrap_or_default();
            consumers.sort(); // deterministic: (topo, consumer_id)

            let death_topo = if graph.leaf_nodes.contains(&node.id) {
                max_topo
            } else {
                consumers.iter().map(|&(t, _)| t).max().unwrap_or(birth_topo)
            };
            // No real consumer at `death_topo` only when the node has no
            // consumers at all (it dies right at birth); otherwise this
            // is the real id of whichever consumer's topo position IS
            // `death_topo`.
            let death_node = consumers
                .iter()
                .find(|&&(t, _)| t == death_topo)
                .map(|&(_, id)| id.to_string())
                .unwrap_or_else(|| node.id.clone());

            birth_death_topo.insert(node.id.as_str(), (birth_topo, death_topo));
            variable_lifetimes.insert(
                node.id.clone(),
                VariableLifetime {
                    birth_node: node.id.clone(),
                    death_node,
                    usage_nodes: consumers.iter().map(|&(_, id)| id.to_string()).collect(),
                    memory_footprint: node.memory_usage,
                },
            );
        }

        let memory_reuse_opportunities =
            self.find_memory_reuse_opportunities(graph, &ordered, &birth_death_topo);

        Ok(DataFlowAnalysis {
            data_dependencies,
            live_variables,
            variable_lifetimes,
            memory_reuse_opportunities,
        })
    }

    /// Real memory-reuse opportunities: pairs of internal (non-
    /// [`ComputationGraph::leaf_nodes`]) variables whose real lifetime
    /// intervals -- from [`Self::analyze_dataflow`]'s topo-order
    /// liveness computation -- do NOT overlap, so one variable's buffer
    /// could be physically reused for the other once the first is dead.
    /// `memory_savings` is the real `min(footprint_a, footprint_b)`:
    /// sizing one shared buffer to `max(a, b)` instead of allocating `a`
    /// and `b` separately saves exactly the smaller footprint.
    /// `complexity` is the real count of variables sharing the buffer
    /// (always `2` here, since this only ever proposes pairwise reuse) --
    /// not an editorial guess.
    ///
    /// Bounded to the `REUSE_CANDIDATE_LIMIT` largest-footprint
    /// variables to keep an otherwise-O(n^2) pairing tractable on large
    /// graphs; a documented performance bound, not a fabrication -- every
    /// opportunity actually reported is still real.
    fn find_memory_reuse_opportunities(
        &self,
        graph: &ComputationGraph,
        ordered: &[&GraphNode],
        birth_death_topo: &HashMap<&str, (usize, usize)>,
    ) -> Vec<MemoryReuseOpportunity> {
        const REUSE_CANDIDATE_LIMIT: usize = 200;

        let mut candidates: Vec<&GraphNode> = ordered
            .iter()
            .filter(|n| n.memory_usage > 0 && !graph.leaf_nodes.contains(&n.id))
            .copied()
            .collect();
        candidates.sort_by_key(|n| std::cmp::Reverse(n.memory_usage));
        candidates.truncate(REUSE_CANDIDATE_LIMIT);

        let mut opportunities = Vec::new();
        for (i, &a) in candidates.iter().enumerate() {
            let Some(&(a_birth, a_death)) = birth_death_topo.get(a.id.as_str()) else {
                continue;
            };
            for &b in &candidates[i + 1..] {
                let Some(&(b_birth, b_death)) = birth_death_topo.get(b.id.as_str()) else {
                    continue;
                };
                // Real non-overlap: does one variable's lifetime end
                // strictly before the other's begins? (Strict, not `<=`:
                // equality would mean one directly consumes the other at
                // that exact step, which is not a safe blind reuse.)
                let non_overlapping = a_death < b_birth || b_death < a_birth;
                if !non_overlapping {
                    continue;
                }
                let savings = a.memory_usage.min(b.memory_usage);
                if savings == 0 {
                    continue;
                }
                let mut reusable_variables = vec![a.id.clone(), b.id.clone()];
                reusable_variables.sort();
                opportunities.push(MemoryReuseOpportunity {
                    reusable_variables,
                    memory_savings: savings,
                    complexity: 2,
                });
            }
        }

        opportunities.sort_by_key(|o| std::cmp::Reverse(o.memory_savings));
        opportunities.truncate(10);
        opportunities
    }

    /// Real critical path: the longest weighted path through the
    /// dependency DAG, found by dynamic programming over the graph's real
    /// topological order (replacing the old "depth as proxy" heuristic,
    /// which only ever reported path *length*, never the actual
    /// highest-cost chain, and in fact walked every other depth level due
    /// to a double-decrement bug).
    ///
    /// Every node in the graph is weighted on the SAME scale: real
    /// profiled `execution_time_us` when *any* node has been profiled
    /// (unprofiled nodes contribute `0`, never compared against a
    /// different unit), falling back to real estimated `flop_count` for
    /// every node only when none of the graph has been profiled yet --
    /// deciding this once per graph (not per node) so a `Some(50)` µs
    /// node is never pitted against a `1_000_000`-FLOP node in the same
    /// path sum.
    fn find_critical_path(&self, graph: &ComputationGraph) -> Result<Vec<String>> {
        let mut ordered: Vec<&GraphNode> = graph.nodes.values().collect();
        ordered.sort_by_key(|n| n.topo_order.unwrap_or(usize::MAX));

        let use_time = graph.nodes.values().any(|n| n.execution_time_us.is_some());
        let weight = |node: &GraphNode| -> f64 {
            if use_time {
                node.execution_time_us.unwrap_or(0) as f64
            } else {
                node.flop_count as f64
            }
        };

        // best_cost[node] = weight of the longest path ending at `node`;
        // predecessor[node] = the dependency that achieves it, for
        // backtracking the actual path afterwards.
        let mut best_cost: HashMap<&str, f64> = HashMap::new();
        let mut predecessor: HashMap<&str, &str> = HashMap::new();

        for node in &ordered {
            if node.topo_order.is_none() {
                continue;
            }
            let mut best_dep_cost = 0.0_f64;
            let mut best_dep: Option<&str> = None;
            for dep in graph.edges.get(&node.id).into_iter().flatten() {
                if let Some(&cost) = best_cost.get(dep.as_str()) {
                    // Deterministic tie-break (lexicographically greater
                    // dep id wins) so results don't depend on HashMap
                    // iteration order.
                    let better = match best_dep {
                        None => true,
                        Some(bd) => {
                            cost > best_dep_cost || (cost == best_dep_cost && dep.as_str() > bd)
                        },
                    };
                    if better {
                        best_dep_cost = cost;
                        best_dep = Some(dep.as_str());
                    }
                }
            }
            best_cost.insert(node.id.as_str(), weight(node) + best_dep_cost);
            if let Some(dep) = best_dep {
                predecessor.insert(node.id.as_str(), dep);
            }
        }

        // The critical path ends at whichever node has the largest total
        // cost -- the real sink of the longest chain, not necessarily a
        // declared `leaf_nodes` member on a multi-output graph. Ties break
        // deterministically on node id.
        let Some((&end_node, _)) = best_cost.iter().max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(b.0))
        }) else {
            return Ok(Vec::new());
        };

        let mut path = vec![end_node.to_string()];
        let mut current = end_node;
        while let Some(&pred) = predecessor.get(current) {
            path.push(pred.to_string());
            current = pred;
        }
        path.reverse();
        Ok(path)
    }

    fn calculate_statistics(&self, graph: &ComputationGraph) -> Result<GraphStatistics> {
        let mut nodes_by_type: HashMap<OperationType, usize> = HashMap::new();
        for node in graph.nodes.values() {
            *nodes_by_type.entry(node.operation_type.clone()).or_insert(0) += 1;
        }

        let total_fan_in: usize = graph.edges.values().map(|deps| deps.len()).sum();
        let total_fan_out = total_fan_in; // In a DAG, total fan-in equals total fan-out
        let average_fan_in = total_fan_in as f64 / graph.nodes.len() as f64;
        let average_fan_out = total_fan_out as f64 / graph.nodes.len() as f64;

        Ok(GraphStatistics {
            nodes_by_type,
            average_fan_in,
            average_fan_out,
            diameter: graph.metadata.max_depth,
            clustering_coefficient: self.compute_clustering_coefficient(graph),
            strongly_connected_components: graph.nodes.len(), // Each node is its own SCC in a DAG
        })
    }

    /// Real average local clustering coefficient (Watts-Strogatz), computed
    /// on the graph's *undirected* neighbor relation: a dependency edge
    /// `dep -> node` makes `dep` and `node` neighbors regardless of
    /// direction, the conventional way to compute this statistic on a
    /// directed graph. For each node `v` with `k_v` neighbors,
    /// `C_v = (edges among v's neighbors) / (k_v * (k_v - 1) / 2)`;
    /// nodes with fewer than 2 neighbors contribute `0` (the standard
    /// convention: no pair of neighbors exists to be connected or not).
    /// The graph-level value is the mean of `C_v` over all nodes.
    ///
    /// This is *not* trivially `0.0` the way the old placeholder claimed:
    /// a "diamond"/skip-connection pattern -- a value feeding both an
    /// operation and that operation's own downstream consumer, e.g.
    /// `x -> f(x)` followed by `Add(x, f(x))` -- makes `x`'s two
    /// consumers neighbors of *each other* too (since one feeds the
    /// other), producing a real triangle and a nonzero `C_v`. This exact
    /// shape is common in transformer graphs (residual connections).
    fn compute_clustering_coefficient(&self, graph: &ComputationGraph) -> f64 {
        if graph.nodes.is_empty() {
            return 0.0;
        }

        let mut neighbors: HashMap<&str, HashSet<&str>> = HashMap::new();
        for node_id in graph.nodes.keys() {
            neighbors.entry(node_id.as_str()).or_default();
        }
        for (node_id, deps) in &graph.edges {
            for dep in deps {
                neighbors.entry(node_id.as_str()).or_default().insert(dep.as_str());
                neighbors.entry(dep.as_str()).or_default().insert(node_id.as_str());
            }
        }

        let mut coefficient_sum = 0.0;
        for neighs in neighbors.values() {
            let k = neighs.len();
            if k < 2 {
                continue; // contributes 0, per convention
            }
            let neigh_vec: Vec<&str> = neighs.iter().copied().collect();
            let mut connected_pairs = 0usize;
            for (i, &a) in neigh_vec.iter().enumerate() {
                for &b in &neigh_vec[i + 1..] {
                    if neighbors.get(a).is_some_and(|n| n.contains(b)) {
                        connected_pairs += 1;
                    }
                }
            }
            let possible_pairs = k * (k - 1) / 2;
            coefficient_sum += connected_pairs as f64 / possible_pairs as f64;
        }

        coefficient_sum / graph.nodes.len() as f64
    }

    fn generate_recommendations(&self, analysis: &GraphAnalysisResult) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        // Memory-based recommendations
        if let Some(ref memory_analysis) = analysis.memory_analysis {
            if memory_analysis.total_memory_usage > 1024 * 1024 * 1024 {
                // > 1GB
                recommendations.push(
                    "Consider using gradient checkpointing to reduce memory usage".to_string(),
                );
            }
            if let Some(ratio) = memory_analysis.fragmentation_ratio {
                if ratio > 0.2 {
                    recommendations
                        .push("Implement memory pooling to reduce fragmentation".to_string());
                }
            }
        }

        // FLOP-based recommendations
        if let Some(ref flop_analysis) = analysis.flop_analysis {
            if flop_analysis.arithmetic_intensity < 1.0 {
                recommendations
                    .push("Consider kernel fusion to improve arithmetic intensity".to_string());
            }
            if flop_analysis.complexity_analysis.parallelization_potential > 0.5 {
                recommendations.push(
                    "Explore parallelization opportunities for compute-intensive operations"
                        .to_string(),
                );
            }
        }

        // Optimization opportunities
        if analysis.optimization_opportunities.len() > 3 {
            recommendations.push(
                "Multiple optimization opportunities detected - prioritize by estimated impact"
                    .to_string(),
            );
        }

        // Bottleneck recommendations
        if let Some(ref bottleneck_analysis) = analysis.bottleneck_analysis {
            if !bottleneck_analysis.bottleneck_nodes.is_empty() {
                recommendations.push(
                    "Address bottleneck operations through optimization or parallelization"
                        .to_string(),
                );
            }
        }

        Ok(recommendations)
    }

    fn get_node_color(&self, op_type: &OperationType) -> &'static str {
        match op_type {
            OperationType::MatMul | OperationType::Dot => "lightblue",
            OperationType::Add
            | OperationType::Subtract
            | OperationType::Multiply
            | OperationType::Divide => "lightgreen",
            OperationType::ReLU
            | OperationType::Sigmoid
            | OperationType::Tanh
            | OperationType::GELU => "orange",
            OperationType::LayerNorm | OperationType::BatchNorm | OperationType::RMSNorm => {
                "yellow"
            },
            OperationType::Conv1D | OperationType::Conv2D | OperationType::Conv3D => "lightcoral",
            OperationType::Attention | OperationType::MultiHeadAttention => "purple",
            OperationType::Embedding | OperationType::PositionalEmbedding => "pink",
            _ => "lightgray",
        }
    }
}

impl fmt::Display for OperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperationType::Custom(name) => write!(f, "Custom({})", name),
            _ => write!(f, "{:?}", self),
        }
    }
}

impl Default for ComputationGraphAnalyzer {
    fn default() -> Self {
        Self::new(GraphAnalysisConfig::default())
    }
}

#[cfg(test)]
#[path = "computation_graph_tests.rs"]
mod tests;
