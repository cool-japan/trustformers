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
    /// Peak memory usage in bytes
    pub peak_memory_usage: u64,
    /// Memory usage by operation type
    pub memory_by_operation: HashMap<OperationType, u64>,
    /// Nodes with highest memory usage
    pub memory_hotspots: Vec<(String, u64)>,
    /// Memory fragmentation estimate
    pub fragmentation_ratio: f64,
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
    /// Time complexity estimate
    pub time_complexity: String,
    /// Space complexity estimate
    pub space_complexity: String,
    /// Parallelization potential (0.0 to 1.0)
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
                adj_list.get_mut(dep).unwrap().push(node_id.clone());
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
                    *in_degree.get_mut(neighbor).unwrap() -= 1;
                    if in_degree[neighbor] == 0 {
                        queue.push_back((neighbor.clone(), depth + 1));
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
        memory_hotspots.sort_by(|a, b| b.1.cmp(&a.1));
        memory_hotspots.truncate(10); // Top 10

        let peak_memory_usage = total_memory_usage; // Simplified
        let fragmentation_ratio = 0.1; // Simplified estimate

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

    fn analyze_flop_usage(&self, graph: &ComputationGraph) -> Result<FlopAnalysis> {
        let total_flops = graph.nodes.values().map(|n| n.flop_count).sum();

        let mut flops_by_operation: HashMap<OperationType, u64> = HashMap::new();
        for node in graph.nodes.values() {
            *flops_by_operation.entry(node.operation_type.clone()).or_insert(0) += node.flop_count;
        }

        let mut compute_hotspots: Vec<(String, u64)> =
            graph.nodes.values().map(|n| (n.id.clone(), n.flop_count)).collect();
        compute_hotspots.sort_by(|a, b| b.1.cmp(&a.1));
        compute_hotspots.truncate(10); // Top 10

        let total_memory = graph.nodes.values().map(|n| n.memory_usage).sum::<u64>();
        let arithmetic_intensity =
            if total_memory > 0 { total_flops as f64 / total_memory as f64 } else { 0.0 };

        let complexity_analysis = ComplexityAnalysis {
            time_complexity: "O(n)".to_string(),  // Simplified
            space_complexity: "O(n)".to_string(), // Simplified
            parallelization_potential: 0.7,       // Simplified estimate
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

    fn detect_redundancy_opportunities(
        &self,
        _graph: &ComputationGraph,
    ) -> Result<Vec<OptimizationOpportunity>> {
        // Simplified - in real implementation would detect common subexpressions
        Ok(vec![])
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
        let mut variable_lifetimes = HashMap::new();

        // Simplified dataflow analysis
        for (node_id, dependencies) in &graph.edges {
            data_dependencies.insert(node_id.clone(), dependencies.clone());
            live_variables.insert(node_id.clone(), dependencies.iter().cloned().collect());

            // Create variable lifetimes for dependencies
            for dep in dependencies {
                if !variable_lifetimes.contains_key(dep) {
                    variable_lifetimes.insert(
                        dep.clone(),
                        VariableLifetime {
                            birth_node: dep.clone(),
                            death_node: node_id.clone(),
                            usage_nodes: vec![node_id.clone()],
                            memory_footprint: graph
                                .nodes
                                .get(dep)
                                .map(|n| n.memory_usage)
                                .unwrap_or(0),
                        },
                    );
                } else {
                    let lifetime = variable_lifetimes.get_mut(dep).unwrap();
                    lifetime.death_node = node_id.clone();
                    lifetime.usage_nodes.push(node_id.clone());
                }
            }
        }

        let memory_reuse_opportunities = vec![MemoryReuseOpportunity {
            reusable_variables: vec!["var1".to_string(), "var2".to_string()],
            memory_savings: 1024 * 1024, // 1MB
            complexity: 2,
        }];

        Ok(DataFlowAnalysis {
            data_dependencies,
            live_variables,
            variable_lifetimes,
            memory_reuse_opportunities,
        })
    }

    fn find_critical_path(&self, graph: &ComputationGraph) -> Result<Vec<String>> {
        // Simplified critical path finding - uses depth as proxy
        let mut path = Vec::new();
        let mut current_depth = graph.metadata.max_depth;

        while current_depth > 0 {
            // Find a node at the current depth
            for node in graph.nodes.values() {
                if node.depth == current_depth {
                    path.push(node.id.clone());
                    current_depth -= 1;
                    break;
                }
            }
            current_depth = current_depth.saturating_sub(1);
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
            clustering_coefficient: 0.0, // Simplified - DAGs have clustering coefficient of 0
            strongly_connected_components: graph.nodes.len(), // Each node is its own SCC in a DAG
        })
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
            if memory_analysis.fragmentation_ratio > 0.2 {
                recommendations
                    .push("Implement memory pooling to reduce fragmentation".to_string());
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
mod tests {
    use super::*;

    #[test]
    fn test_computation_graph_creation() {
        let mut analyzer = ComputationGraphAnalyzer::default();

        let operations = vec![
            (
                "input".to_string(),
                OperationType::Custom("Input".to_string()),
                vec![],
            ),
            (
                "linear1".to_string(),
                OperationType::MatMul,
                vec!["input".to_string()],
            ),
            (
                "relu1".to_string(),
                OperationType::ReLU,
                vec!["linear1".to_string()],
            ),
            (
                "linear2".to_string(),
                OperationType::MatMul,
                vec!["relu1".to_string()],
            ),
            (
                "output".to_string(),
                OperationType::Custom("Output".to_string()),
                vec!["linear2".to_string()],
            ),
        ];

        let graph_id = analyzer.create_graph("test_model".to_string(), operations).unwrap();
        let analysis = analyzer.analyze_graph(graph_id).unwrap();

        assert_eq!(analysis.statistics.nodes_by_type.len(), 4); // MatMul, ReLU, Custom("Input"), Custom("Output")
        assert!(analysis.critical_path.len() > 0);
    }

    #[test]
    fn test_optimization_detection() {
        let mut analyzer = ComputationGraphAnalyzer::default();

        let operations = vec![
            (
                "input".to_string(),
                OperationType::Custom("Input".to_string()),
                vec![],
            ),
            (
                "matmul".to_string(),
                OperationType::MatMul,
                vec!["input".to_string()],
            ),
            (
                "add".to_string(),
                OperationType::Add,
                vec!["matmul".to_string()],
            ),
        ];

        let graph_id = analyzer.create_graph("fusion_test".to_string(), operations).unwrap();
        let analysis = analyzer.analyze_graph(graph_id).unwrap();

        assert!(analysis
            .optimization_opportunities
            .iter()
            .any(|op| op.optimization_type == OptimizationType::OperationFusion));
    }

    #[test]
    fn test_dot_export() {
        let mut analyzer = ComputationGraphAnalyzer::default();

        let operations = vec![
            ("a".to_string(), OperationType::MatMul, vec![]),
            ("b".to_string(), OperationType::ReLU, vec!["a".to_string()]),
        ];

        let graph_id = analyzer.create_graph("simple".to_string(), operations).unwrap();
        let dot = analyzer.export_to_dot(graph_id).unwrap();

        assert!(dot.contains("digraph"));
        assert!(dot.contains("MatMul"));
        assert!(dot.contains("ReLU"));
    }
}
