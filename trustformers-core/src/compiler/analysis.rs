#![allow(unused_variables)] // Compiler analysis with reserved parameters

/*!
# Compiler Analysis Module

This module provides comprehensive analysis capabilities for computation graphs including:

- **Performance Analysis**: Cost estimation, bottleneck detection, critical path analysis
- **Memory Analysis**: Memory usage patterns, allocation optimization, lifetime analysis
- **Dependency Analysis**: Data flow analysis, parallelization opportunities
- **Hardware Analysis**: Hardware utilization prediction, resource requirements

These analyses inform optimization decisions and provide insights into graph characteristics.
*/

use crate::compiler::{ComputationGraph, DeviceType, GraphNode, HardwareTarget};
use crate::errors::invalid_input;
use crate::errors::TrustformersError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Comprehensive performance analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    /// Total estimated execution time in milliseconds
    pub total_execution_time_ms: f64,
    /// Critical path operations
    pub critical_path: Vec<usize>,
    /// Critical path length in milliseconds
    pub critical_path_length_ms: f64,
    /// Parallelization opportunities
    pub parallelizable_operations: Vec<Vec<usize>>,
    /// Bottleneck operations
    pub bottlenecks: Vec<BottleneckInfo>,
    /// Load balancing metrics
    pub load_balance_score: f64,
    /// Hardware utilization prediction
    pub hardware_utilization: HardwareUtilization,
}

/// Bottleneck information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckInfo {
    pub node_id: usize,
    pub operation_type: String,
    pub execution_time_ms: f64,
    pub memory_usage_mb: f64,
    pub criticality_score: f64,
    pub optimization_suggestions: Vec<String>,
}

/// Hardware utilization prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareUtilization {
    pub compute_utilization: f64, // 0.0 to 1.0
    pub memory_utilization: f64,  // 0.0 to 1.0
    pub memory_bandwidth_utilization: f64,
    /// Predicted cache hit rate, when it can be predicted.
    ///
    /// Always `None` from static graph analysis: a hit rate is a property of
    /// an actual memory hierarchy (associativity, working-set size, access
    /// order at run time), and this analyser has no cache simulation model to
    /// derive one from. It is `None` rather than a plausible-looking default
    /// so a caller cannot mistake "not modelled" for "measured".
    pub cache_hit_rate_prediction: Option<f64>,
    pub parallel_efficiency: f64,
}

/// Memory analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    /// Peak memory usage in bytes
    pub peak_memory_usage: u64,
    /// Memory usage timeline
    pub memory_timeline: Vec<MemorySnapshot>,
    /// Memory allocation patterns
    pub allocation_patterns: Vec<AllocationPattern>,
    /// Memory reuse opportunities
    pub reuse_opportunities: Vec<ReuseOpportunity>,
    /// Memory fragmentation analysis, when it can be produced.
    ///
    /// Always `None` from static graph analysis: fragmentation is a property of
    /// a running allocator. It is `None` rather than a plausible-looking
    /// default so a caller cannot mistake "not analysed" for "no
    /// fragmentation".
    pub fragmentation_analysis: Option<FragmentationAnalysis>,
}

/// Memory snapshot at a point in execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub operation_id: usize,
    pub allocated_memory: u64,
    pub active_tensors: Vec<TensorInfo>,
    pub memory_pressure: f64,
}

/// Tensor information for memory analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub id: usize,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size_bytes: u64,
    pub lifetime_start: usize,
    pub lifetime_end: usize,
}

/// Memory allocation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPattern {
    pub pattern_type: AllocationType,
    pub frequency: usize,
    pub total_size: u64,
    pub optimization_potential: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationType {
    Sequential,
    Scattered,
    Temporary,
    LongLived,
    Reusable,
}

/// Memory reuse opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReuseOpportunity {
    pub tensor_id: usize,
    pub reusable_with: Vec<usize>,
    pub memory_savings: u64,
    pub implementation_complexity: ComplexityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
}

/// Memory fragmentation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationAnalysis {
    pub fragmentation_ratio: f64,
    pub largest_free_block: u64,
    pub allocation_efficiency: f64,
    pub defragmentation_potential: f64,
}

/// Dependency analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyAnalysis {
    /// Topological ordering of operations
    pub topological_order: Vec<usize>,
    /// Strongly connected components
    pub connected_components: Vec<Vec<usize>>,
    /// Data flow dependencies
    pub data_dependencies: Vec<Dependency>,
    /// Loop analysis, when the IR has loops to analyse.
    ///
    /// Always `None` for this compiler's IR, which is a DAG of tensor
    /// operations with no loop constructs.
    pub loop_analysis: Option<LoopAnalysis>,
    /// Parallelization analysis
    pub parallelization: ParallelizationAnalysis,
}

/// Data dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub from: usize,
    pub to: usize,
    pub dependency_type: DependencyType,
    pub data_size: u64,
    pub latency_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    DataFlow,
    Control,
    Memory,
    Synchronization,
}

/// Loop analysis information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopAnalysis {
    pub detected_loops: Vec<LoopInfo>,
    pub loop_carried_dependencies: Vec<Dependency>,
    pub vectorization_opportunities: Vec<VectorizationOpportunity>,
}

/// Information about detected loops
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopInfo {
    pub loop_id: usize,
    pub operations: Vec<usize>,
    pub iteration_count: Option<usize>,
    pub loop_type: LoopType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoopType {
    CountBased,
    DataDependent,
    Infinite,
    Unknown,
}

/// Vectorization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorizationOpportunity {
    pub operations: Vec<usize>,
    pub vector_width: usize,
    pub performance_gain: f64,
    pub instruction_set: String,
}

/// Parallelization analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationAnalysis {
    pub parallel_regions: Vec<ParallelRegion>,
    pub synchronization_points: Vec<usize>,
    pub load_balance_analysis: LoadBalanceAnalysis,
    pub communication_analysis: CommunicationAnalysis,
}

/// Parallel execution region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelRegion {
    pub operations: Vec<usize>,
    pub parallelism_type: ParallelismType,
    pub estimated_speedup: f64,
    pub resource_requirements: ResourceRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParallelismType {
    DataParallel,
    TaskParallel,
    Pipeline,
    Mixed,
}

/// Resource requirements for parallel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_threads: usize,
    pub optimal_threads: usize,
    pub memory_per_thread: u64,
    pub communication_bandwidth: f64,
}

/// Load balance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalanceAnalysis {
    pub balance_score: f64,
    pub work_distribution: Vec<f64>,
    pub synchronization_overhead: f64,
    pub recommendations: Vec<String>,
}

/// Communication analysis for distributed execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationAnalysis {
    pub communication_volume: u64,
    pub communication_patterns: Vec<CommunicationPattern>,
    pub network_utilization: f64,
    pub latency_sensitivity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPattern {
    pub pattern_type: CommunicationType,
    pub data_size: u64,
    pub frequency: usize,
    pub optimization_potential: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationType {
    AllToAll,
    AllReduce,
    PointToPoint,
    Broadcast,
    Gather,
    Scatter,
}

/// Main analyzer that orchestrates all analysis types
pub struct GraphAnalyzer {
    hardware_target: HardwareTarget,
    #[allow(dead_code)]
    analysis_cache: HashMap<String, AnalysisResult>,
}

/// Combined analysis result
#[derive(Debug, Clone)]
pub enum AnalysisResult {
    Performance(PerformanceAnalysis),
    Memory(MemoryAnalysis),
    Dependency(DependencyAnalysis),
}

impl GraphAnalyzer {
    /// Create a new graph analyzer
    pub fn new(hardware_target: HardwareTarget) -> Self {
        Self {
            hardware_target,
            analysis_cache: HashMap::new(),
        }
    }

    /// Perform comprehensive performance analysis
    pub fn analyze_performance(
        &mut self,
        graph: &ComputationGraph,
    ) -> Result<PerformanceAnalysis, TrustformersError> {
        // Critical path analysis
        let critical_path = self.find_critical_path(graph)?;
        let critical_path_length = self.calculate_path_length(&critical_path, graph)?;

        // Bottleneck detection
        let bottlenecks = self.detect_bottlenecks(graph)?;

        // Parallelization analysis
        let parallelizable_ops = self.find_parallelizable_operations(graph)?;

        // Load balancing
        let load_balance_score = self.calculate_load_balance_score(graph)?;

        // Hardware utilization prediction
        let hardware_utilization = self.predict_hardware_utilization(graph)?;

        let total_execution_time =
            graph.nodes.iter().map(|node| self.estimate_execution_time(node)).sum();

        Ok(PerformanceAnalysis {
            total_execution_time_ms: total_execution_time,
            critical_path,
            critical_path_length_ms: critical_path_length,
            parallelizable_operations: parallelizable_ops,
            bottlenecks,
            load_balance_score,
            hardware_utilization,
        })
    }

    /// Perform memory analysis
    pub fn analyze_memory(
        &mut self,
        graph: &ComputationGraph,
    ) -> Result<MemoryAnalysis, TrustformersError> {
        let memory_timeline = self.simulate_memory_usage(graph)?;
        let peak_memory = memory_timeline
            .iter()
            .map(|snapshot| snapshot.allocated_memory)
            .max()
            .unwrap_or(0);

        let allocation_patterns = self.analyze_allocation_patterns(graph)?;
        let reuse_opportunities = self.find_reuse_opportunities(graph)?;
        // Fragmentation needs a live allocator; `None` records that honestly.
        let fragmentation_analysis = self.analyze_fragmentation(graph).ok();

        Ok(MemoryAnalysis {
            peak_memory_usage: peak_memory,
            memory_timeline,
            allocation_patterns,
            reuse_opportunities,
            fragmentation_analysis,
        })
    }

    /// Perform dependency analysis
    pub fn analyze_dependencies(
        &mut self,
        graph: &ComputationGraph,
    ) -> Result<DependencyAnalysis, TrustformersError> {
        let topological_order = self.topological_sort(graph)?;
        let connected_components = self.find_connected_components(graph)?;
        let data_dependencies = self.analyze_data_dependencies(graph)?;
        // The IR has no loops; `None` records that rather than an empty result.
        let loop_analysis = self.analyze_loops(graph).ok();
        let parallelization = self.analyze_parallelization(graph)?;

        Ok(DependencyAnalysis {
            topological_order,
            connected_components,
            data_dependencies,
            loop_analysis,
            parallelization,
        })
    }

    /// Find critical path through the computation graph
    fn find_critical_path(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<usize>, TrustformersError> {
        let mut longest_path = HashMap::new();
        let mut predecessors = HashMap::new();

        // Initialize
        for node in &graph.nodes {
            longest_path.insert(node.id, 0.0);
        }

        // Topological sort and longest path calculation
        let topo_order = self.topological_sort(graph)?;

        for &node_id in &topo_order {
            let node_time = self.estimate_execution_time(&graph.nodes[node_id]);

            for edge in &graph.edges {
                if edge.from != node_id {
                    continue;
                }
                let new_distance = longest_path[&node_id] + node_time;
                if new_distance > longest_path[&edge.to] {
                    longest_path.insert(edge.to, new_distance);
                    predecessors.insert(edge.to, node_id);
                }
            }
        }

        // Find the end node with maximum distance
        let end_node = longest_path
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(node_id, _)| *node_id)
            .unwrap_or(0);

        // Reconstruct path
        let mut path = Vec::new();
        let mut current = end_node;

        while let Some(&predecessor) = predecessors.get(&current) {
            path.push(current);
            current = predecessor;
        }
        path.push(current);
        path.reverse();

        Ok(path)
    }

    /// Calculate the length of a path in terms of execution time
    fn calculate_path_length(
        &self,
        path: &[usize],
        graph: &ComputationGraph,
    ) -> Result<f64, TrustformersError> {
        let total_time = path
            .iter()
            .map(|&node_id| {
                if let Some(node) = graph.get_node(node_id) {
                    self.estimate_execution_time(node)
                } else {
                    0.0
                }
            })
            .sum();

        Ok(total_time)
    }

    /// Estimate execution time for a single operation
    fn estimate_execution_time(&self, node: &GraphNode) -> f64 {
        // Base time estimation based on operation type and hardware
        let base_time = match node.op_type.as_str() {
            "MatMul" => {
                // Estimate based on matrix dimensions and hardware
                let flops = node.compute_cost;
                match self.hardware_target.device_type {
                    DeviceType::GPU => flops / 10e12, // 10 TFLOPS
                    DeviceType::CPU => flops / 1e12,  // 1 TFLOP
                    _ => flops / 1e9,                 // 1 GFLOP
                }
            },
            "Conv2D" => node.compute_cost / 5e12, // 5 TFLOPS for convolution
            "Add" | "Mul" | "Sub" | "Div" => node.compute_cost / 1e13, // Very fast element-wise ops
            "ReLU" | "Sigmoid" | "Tanh" => node.compute_cost / 1e12,
            _ => node.compute_cost / 1e9, // Default estimate
        };

        // Add memory access overhead
        let memory_time = node.memory_cost / self.hardware_target.memory_bandwidth;

        (base_time + memory_time) * 1000.0 // Convert to milliseconds
    }

    /// Detect performance bottlenecks
    fn detect_bottlenecks(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<BottleneckInfo>, TrustformersError> {
        let mut bottlenecks = Vec::new();

        let total_time: f64 =
            graph.nodes.iter().map(|node| self.estimate_execution_time(node)).sum();

        for node in &graph.nodes {
            let execution_time = self.estimate_execution_time(node);
            let time_percentage = execution_time / total_time;

            // Consider nodes taking more than 10% of total time as potential bottlenecks
            if time_percentage > 0.1 {
                let memory_usage = node.memory_cost / (1024.0 * 1024.0); // Convert to MB
                let criticality_score = time_percentage * 100.0;

                let suggestions = self.generate_optimization_suggestions(node);

                bottlenecks.push(BottleneckInfo {
                    node_id: node.id,
                    operation_type: node.op_type.clone(),
                    execution_time_ms: execution_time,
                    memory_usage_mb: memory_usage,
                    criticality_score,
                    optimization_suggestions: suggestions,
                });
            }
        }

        // Sort by criticality score
        bottlenecks.sort_by(|a, b| {
            b.criticality_score
                .partial_cmp(&a.criticality_score)
                .unwrap_or(::std::cmp::Ordering::Equal)
        });

        Ok(bottlenecks)
    }

    /// Generate optimization suggestions for a node
    fn generate_optimization_suggestions(&self, node: &GraphNode) -> Vec<String> {
        let mut suggestions = Vec::new();

        match node.op_type.as_str() {
            "MatMul" => {
                suggestions.push("Consider using optimized BLAS libraries".to_string());
                suggestions.push("Try different matrix multiplication algorithms".to_string());
                suggestions
                    .push("Consider batch processing for multiple small matrices".to_string());
            },
            "Conv2D" => {
                suggestions.push("Use optimized convolution libraries (cuDNN, oneDNN)".to_string());
                suggestions
                    .push("Consider different convolution algorithms (Winograd, FFT)".to_string());
                suggestions.push("Try different data layouts (NCHW vs NHWC)".to_string());
            },
            "Attention" => {
                suggestions.push(
                    "Use FlashAttention or similar memory-efficient implementations".to_string(),
                );
                suggestions.push("Consider attention sparsity patterns".to_string());
                suggestions.push("Try different attention approximations".to_string());
            },
            _ => {
                suggestions.push("Profile the operation to understand bottlenecks".to_string());
                suggestions
                    .push("Consider operation fusion with neighboring operations".to_string());
            },
        }

        suggestions
    }

    /// Find operations that can be parallelized
    fn find_parallelizable_operations(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<Vec<usize>>, TrustformersError> {
        let mut parallel_groups = Vec::new();
        let mut visited = HashSet::new();

        // Find nodes that have no dependencies between them
        for (i, node1) in graph.nodes.iter().enumerate() {
            if visited.contains(&i) {
                continue;
            }

            let mut group = vec![i];
            visited.insert(i);

            for (j, node2) in graph.nodes.iter().enumerate() {
                if i == j || visited.contains(&j) {
                    continue;
                }
                // Check if there's a dependency path between nodes
                if self.has_dependency_path(i, j, graph) || self.has_dependency_path(j, i, graph) {
                    continue;
                }
                group.push(j);
                visited.insert(j);
            }

            if group.len() > 1 {
                parallel_groups.push(group);
            }
        }

        Ok(parallel_groups)
    }

    /// Check if there's a dependency path between two nodes
    fn has_dependency_path(&self, from: usize, to: usize, graph: &ComputationGraph) -> bool {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(from);
        visited.insert(from);

        while let Some(current) = queue.pop_front() {
            if current == to {
                return true;
            }

            for edge in &graph.edges {
                if edge.from == current && !visited.contains(&edge.to) {
                    visited.insert(edge.to);
                    queue.push_back(edge.to);
                }
            }
        }

        false
    }

    /// Calculate load balance score for the graph
    fn calculate_load_balance_score(
        &self,
        graph: &ComputationGraph,
    ) -> Result<f64, TrustformersError> {
        let execution_times: Vec<f64> =
            graph.nodes.iter().map(|node| self.estimate_execution_time(node)).collect();

        if execution_times.is_empty() {
            return Ok(1.0);
        }

        let mean_time: f64 = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
        let variance: f64 =
            execution_times.iter().map(|&time| (time - mean_time).powi(2)).sum::<f64>()
                / execution_times.len() as f64;

        let coefficient_of_variation = variance.sqrt() / mean_time.max(1e-10);

        // Load balance score: 1.0 is perfect balance, 0.0 is completely unbalanced
        Ok((1.0 / (1.0 + coefficient_of_variation)).min(1.0))
    }

    /// Typical total memory capacity for the configured target device.
    ///
    /// Not a measurement (this analyser never queries real hardware); it is
    /// a small per-device-type table of representative capacities, used
    /// consistently everywhere this analyser needs a capacity to compare a
    /// graph's estimated footprint against. Shared by
    /// `predict_hardware_utilization` and `simulate_memory_usage` so the two
    /// can never silently disagree about which device's capacity applies.
    fn available_memory_bytes(&self) -> f64 {
        match self.hardware_target.device_type {
            DeviceType::GPU => 16e9, // 16 GB typical GPU memory
            DeviceType::CPU => 64e9, // 64 GB typical system memory
            _ => 8e9,                // 8 GB default
        }
    }

    /// Predict hardware utilization
    fn predict_hardware_utilization(
        &self,
        graph: &ComputationGraph,
    ) -> Result<HardwareUtilization, TrustformersError> {
        let total_compute = graph.total_compute_cost();
        let total_memory = graph.total_memory_cost();

        // Estimate compute utilization based on operation types
        let compute_intensive_ops = graph
            .nodes
            .iter()
            .filter(|node| matches!(node.op_type.as_str(), "MatMul" | "Conv2D" | "Attention"))
            .count();

        let compute_utilization =
            (compute_intensive_ops as f64 / graph.nodes.len().max(1) as f64) * 0.8;

        // Estimate memory utilization
        let estimated_memory = total_memory;
        let memory_utilization = (estimated_memory / self.available_memory_bytes()).min(1.0);

        // Estimate memory bandwidth utilization
        let memory_bandwidth_utilization =
            (total_memory / 1e9) / self.hardware_target.memory_bandwidth;

        // No cache simulation model is available from static graph analysis;
        // see the field's doc comment on `HardwareUtilization`.
        let cache_hit_rate_prediction = None;

        // Parallel efficiency estimation
        let parallelizable_ops = self.find_parallelizable_operations(graph)?.len();
        let parallel_efficiency =
            (parallelizable_ops as f64 / graph.nodes.len().max(1) as f64) * 0.9;

        Ok(HardwareUtilization {
            compute_utilization,
            memory_utilization,
            memory_bandwidth_utilization,
            cache_hit_rate_prediction,
            parallel_efficiency,
        })
    }

    /// Simulate memory usage over time
    fn simulate_memory_usage(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<MemorySnapshot>, TrustformersError> {
        let mut snapshots = Vec::new();
        let mut active_tensors = HashMap::new();
        let mut total_memory = 0u64;

        let topo_order = self.topological_sort(graph)?;

        for &node_id in &topo_order {
            if let Some(node) = graph.get_node(node_id) {
                // Add output tensors
                for (i, shape) in node.output_shapes.iter().enumerate() {
                    let tensor_size = self.calculate_tensor_size(shape, "f32");
                    let tensor_info = TensorInfo {
                        id: node_id * 100 + i, // Simple ID scheme
                        shape: shape.clone(),
                        dtype: "f32".to_string(),
                        size_bytes: tensor_size,
                        lifetime_start: node_id,
                        lifetime_end: node_id + 10, // Estimate lifetime
                    };

                    active_tensors.insert(tensor_info.id, tensor_info);
                    total_memory += tensor_size;
                }

                // Calculate memory pressure
                // Same target-device capacity table `predict_hardware_utilization`
                // uses, rather than a hardcoded capacity independent of
                // `self.hardware_target`.
                let memory_pressure = total_memory as f64 / self.available_memory_bytes();

                let snapshot = MemorySnapshot {
                    operation_id: node_id,
                    allocated_memory: total_memory,
                    active_tensors: active_tensors.values().cloned().collect(),
                    memory_pressure,
                };

                snapshots.push(snapshot);

                // Remove expired tensors (simplified)
                active_tensors.retain(|_, tensor| tensor.lifetime_end > node_id);
                total_memory = active_tensors.values().map(|t| t.size_bytes).sum();
            }
        }

        Ok(snapshots)
    }

    /// Calculate tensor size in bytes
    fn calculate_tensor_size(&self, shape: &[usize], dtype: &str) -> u64 {
        let element_size = match dtype {
            "f32" | "i32" => 4,
            "f16" | "i16" => 2,
            "f64" | "i64" => 8,
            "i8" | "u8" => 1,
            _ => 4, // Default to 4 bytes
        };

        let elements: usize = shape.iter().product();
        (elements * element_size) as u64
    }

    /// Perform topological sort
    fn topological_sort(&self, graph: &ComputationGraph) -> Result<Vec<usize>, TrustformersError> {
        let mut in_degree = vec![0; graph.nodes.len()];
        let mut adj_list = vec![Vec::new(); graph.nodes.len()];

        // Build adjacency list and calculate in-degrees
        for edge in &graph.edges {
            if edge.from < graph.nodes.len() && edge.to < graph.nodes.len() {
                adj_list[edge.from].push(edge.to);
                in_degree[edge.to] += 1;
            }
        }

        // Kahn's algorithm
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Add nodes with no incoming edges
        for (i, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(i);
            }
        }

        while let Some(node) = queue.pop_front() {
            result.push(node);

            for &neighbor in &adj_list[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        if result.len() != graph.nodes.len() {
            return Err(invalid_input("Graph contains cycles"));
        }

        Ok(result)
    }

    /// Weakly connected components of the graph.
    ///
    /// Nodes are grouped by reachability treating edges as undirected, via
    /// union-find. Components that share no data can be scheduled or placed
    /// independently.
    fn find_connected_components(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<Vec<usize>>, TrustformersError> {
        let node_count = graph.nodes.len();
        if node_count == 0 {
            return Ok(Vec::new());
        }

        // Map node ids to dense indices; ids need not be 0..n.
        let index_of: HashMap<usize, usize> =
            graph.nodes.iter().enumerate().map(|(index, node)| (node.id, index)).collect();

        let mut parent: Vec<usize> = (0..node_count).collect();

        fn find(parent: &mut [usize], mut node: usize) -> usize {
            while parent[node] != node {
                // Path halving keeps the tree shallow.
                parent[node] = parent[parent[node]];
                node = parent[node];
            }
            node
        }

        for edge in &graph.edges {
            let (Some(&from), Some(&to)) = (index_of.get(&edge.from), index_of.get(&edge.to))
            else {
                continue;
            };
            let root_from = find(&mut parent, from);
            let root_to = find(&mut parent, to);
            if root_from != root_to {
                parent[root_to] = root_from;
            }
        }

        let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
        for index in 0..node_count {
            let root = find(&mut parent, index);
            components.entry(root).or_default().push(graph.nodes[index].id);
        }

        let mut result: Vec<Vec<usize>> = components.into_values().collect();
        for component in &mut result {
            component.sort_unstable();
        }
        // Stable order so repeated analyses of the same graph agree.
        result.sort_by(|a, b| a.first().cmp(&b.first()));
        Ok(result)
    }

    /// Data-flow dependencies, one per graph edge.
    ///
    /// `data_size` is the size of the tensor the edge carries, computed from
    /// the producer's declared output shape (4 bytes per element, f32).
    /// `latency_impact` is the producer's declared compute cost: the consumer
    /// cannot start until that work finishes.
    fn analyze_data_dependencies(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<Dependency>, TrustformersError> {
        let node_by_id: HashMap<usize, &GraphNode> =
            graph.nodes.iter().map(|node| (node.id, node)).collect();

        let mut dependencies = Vec::with_capacity(graph.edges.len());
        for edge in &graph.edges {
            let Some(producer) = node_by_id.get(&edge.from) else {
                continue;
            };

            let data_size = producer
                .output_shapes
                .get(edge.output_idx)
                .map(|shape| shape.iter().product::<usize>() as u64 * 4)
                .unwrap_or(0);

            dependencies.push(Dependency {
                from: edge.from,
                to: edge.to,
                dependency_type: DependencyType::DataFlow,
                data_size,
                latency_impact: producer.compute_cost,
            });
        }

        Ok(dependencies)
    }

    /// Loop analysis.
    ///
    /// Not implemented: the compiler IR is a DAG of tensor operations with no
    /// loop constructs, so there is nothing to detect. Returning an all-empty
    /// `LoopAnalysis` would be indistinguishable from "analysed and found
    /// none".
    fn analyze_loops(&self, _graph: &ComputationGraph) -> Result<LoopAnalysis, TrustformersError> {
        Err(TrustformersError::not_implemented(
            "loop analysis: the compiler IR has no loop constructs to analyse".to_string(),
        ))
    }

    /// Parallelization analysis derived from the graph's real structure.
    ///
    /// Independent components can run in parallel; the load balance is measured
    /// from their declared compute costs. Communication figures are left at
    /// zero with an explicit recommendation, because this analyser has no
    /// placement or network model to derive them from.
    fn analyze_parallelization(
        &self,
        graph: &ComputationGraph,
    ) -> Result<ParallelizationAnalysis, TrustformersError> {
        let components = self.find_connected_components(graph)?;
        let cost_of: HashMap<usize, f64> =
            graph.nodes.iter().map(|node| (node.id, node.compute_cost)).collect();

        let mut parallel_regions = Vec::with_capacity(components.len());
        let mut work_distribution = Vec::with_capacity(components.len());

        for component in &components {
            let work: f64 = component.iter().filter_map(|id| cost_of.get(id)).copied().sum();
            work_distribution.push(work);

            parallel_regions.push(ParallelRegion {
                operations: component.clone(),
                parallelism_type: ParallelismType::TaskParallel,
                // Independent components run concurrently; the speedup a region
                // contributes is bounded by its share of the total work.
                estimated_speedup: 1.0,
                resource_requirements: ResourceRequirements {
                    // One thread per region is the only requirement the graph
                    // itself implies; thread counts and bandwidth need a
                    // hardware model this analyser does not have.
                    min_threads: 1,
                    optimal_threads: 1,
                    memory_per_thread: component
                        .iter()
                        .filter_map(|id| graph.nodes.iter().find(|node| node.id == *id))
                        .map(|node| node.memory_cost as u64)
                        .sum(),
                    communication_bandwidth: 0.0,
                },
            });
        }

        // Balance score: 1.0 when every region carries equal work, falling to
        // 0 as one region dominates. Undefined for a single region.
        let total_work: f64 = work_distribution.iter().sum();
        let balance_score = if work_distribution.len() < 2 || total_work <= 0.0 {
            1.0
        } else {
            let mean = total_work / work_distribution.len() as f64;
            let max_deviation =
                work_distribution.iter().map(|work| (work - mean).abs()).fold(0.0f64, f64::max);
            (1.0 - max_deviation / total_work).clamp(0.0, 1.0)
        };

        let mut recommendations = Vec::new();
        if work_distribution.len() > 1 && balance_score < 0.8 {
            recommendations.push(format!(
                "work is unevenly distributed across {} independent regions (balance {:.2})",
                work_distribution.len(),
                balance_score
            ));
        }
        recommendations.push(
            "communication volume and network utilization are not modelled by this analyser"
                .to_string(),
        );

        Ok(ParallelizationAnalysis {
            parallel_regions,
            synchronization_points: Vec::new(),
            load_balance_analysis: LoadBalanceAnalysis {
                balance_score,
                work_distribution,
                // No synchronization model exists, so no overhead is claimed.
                synchronization_overhead: 0.0,
                recommendations,
            },
            communication_analysis: CommunicationAnalysis {
                communication_volume: graph
                    .edges
                    .iter()
                    .filter_map(|edge| {
                        let producer = graph.nodes.iter().find(|node| node.id == edge.from)?;
                        producer
                            .output_shapes
                            .get(edge.output_idx)
                            .map(|shape| shape.iter().product::<usize>() as u64 * 4)
                    })
                    .sum(),
                communication_patterns: Vec::new(),
                // No placement or network model: these are not estimated.
                network_utilization: 0.0,
                latency_sensitivity: 0.0,
            },
        })
    }

    /// Allocation patterns, one per distinct tensor the graph produces.
    ///
    /// A tensor consumed exactly once is `Temporary` (it can be freed straight
    /// after its consumer); one consumed several times is `LongLived`; one with
    /// no consumer is a graph output and therefore `LongLived` too.
    fn analyze_allocation_patterns(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<AllocationPattern>, TrustformersError> {
        let mut consumers: HashMap<(usize, usize), usize> = HashMap::new();
        for edge in &graph.edges {
            *consumers.entry((edge.from, edge.output_idx)).or_insert(0) += 1;
        }

        let mut temporary_total = 0u64;
        let mut temporary_count = 0usize;
        let mut long_lived_total = 0u64;
        let mut long_lived_count = 0usize;

        for node in &graph.nodes {
            for (output_idx, shape) in node.output_shapes.iter().enumerate() {
                let size = shape.iter().product::<usize>() as u64 * 4;
                match consumers.get(&(node.id, output_idx)).copied().unwrap_or(0) {
                    1 => {
                        temporary_total += size;
                        temporary_count += 1;
                    },
                    _ => {
                        long_lived_total += size;
                        long_lived_count += 1;
                    },
                }
            }
        }

        let mut patterns = Vec::new();
        if temporary_count > 0 {
            patterns.push(AllocationPattern {
                pattern_type: AllocationType::Temporary,
                frequency: temporary_count,
                total_size: temporary_total,
                // Single-use tensors are exactly the ones a reuse pass can
                // fold into their consumer's buffer.
                optimization_potential: 1.0,
            });
        }
        if long_lived_count > 0 {
            patterns.push(AllocationPattern {
                pattern_type: AllocationType::LongLived,
                frequency: long_lived_count,
                total_size: long_lived_total,
                optimization_potential: 0.0,
            });
        }

        Ok(patterns)
    }

    /// Buffer-reuse opportunities.
    ///
    /// A tensor with exactly one consumer can share that consumer's output
    /// buffer when the shapes match, saving its allocation.
    fn find_reuse_opportunities(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<ReuseOpportunity>, TrustformersError> {
        let node_by_id: HashMap<usize, &GraphNode> =
            graph.nodes.iter().map(|node| (node.id, node)).collect();

        let mut consumers: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for edge in &graph.edges {
            consumers.entry((edge.from, edge.output_idx)).or_default().push(edge.to);
        }

        let mut opportunities = Vec::new();
        for ((producer_id, output_idx), consumer_ids) in &consumers {
            if consumer_ids.len() != 1 {
                continue;
            }
            let Some(producer) = node_by_id.get(producer_id) else {
                continue;
            };
            let Some(shape) = producer.output_shapes.get(*output_idx) else {
                continue;
            };
            let Some(consumer) = node_by_id.get(&consumer_ids[0]) else {
                continue;
            };

            // In-place reuse requires the consumer to produce the same shape.
            if !consumer.output_shapes.iter().any(|candidate| candidate == shape) {
                continue;
            }

            opportunities.push(ReuseOpportunity {
                tensor_id: *producer_id,
                reusable_with: consumer_ids.clone(),
                memory_savings: shape.iter().product::<usize>() as u64 * 4,
                implementation_complexity: ComplexityLevel::Low,
            });
        }

        opportunities.sort_by_key(|opportunity| opportunity.tensor_id);
        Ok(opportunities)
    }

    /// Memory fragmentation analysis.
    ///
    /// Not implemented: fragmentation is a property of a running allocator, and
    /// this analyser sees only a static graph. Reporting a 0.1 fragmentation
    /// ratio and 90% allocation efficiency for every graph — as this used to —
    /// gave optimisation decisions a number that described nothing.
    fn analyze_fragmentation(
        &self,
        _graph: &ComputationGraph,
    ) -> Result<FragmentationAnalysis, TrustformersError> {
        Err(TrustformersError::not_implemented(
            "memory fragmentation analysis: fragmentation is a runtime allocator property and \
             cannot be derived from a static graph"
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a graph with two independent chains, so the analysis has real
    /// structure to find: 0 -> 1, and a disconnected 2.
    fn two_component_graph() -> ComputationGraph {
        let node = |id: usize, cost: f64, out: Vec<usize>| GraphNode {
            id,
            op_type: "matmul".to_string(),
            attributes: HashMap::new(),
            input_shapes: vec![],
            output_shapes: vec![out],
            compute_cost: cost,
            memory_cost: 16.0,
        };

        ComputationGraph {
            nodes: vec![
                node(0, 10.0, vec![2, 2]),
                node(1, 5.0, vec![2, 2]),
                node(2, 1.0, vec![4]),
            ],
            edges: vec![crate::compiler::GraphEdge {
                from: 0,
                to: 1,
                output_idx: 0,
                input_idx: 0,
                shape: vec![2, 2],
                dtype: "f32".to_string(),
            }],
            metadata: HashMap::new(),
        }
    }

    /// Regression test: `find_connected_components` returned `Vec::new()` for
    /// every graph, and `analyze_data_dependencies` likewise.
    #[test]
    fn test_dependency_analysis_reflects_the_real_graph() {
        let mut analyzer = GraphAnalyzer::new(HardwareTarget::default());
        let graph = two_component_graph();

        let analysis = analyzer.analyze_dependencies(&graph).expect("analysis failed");

        // Two components: {0, 1} and {2}.
        assert_eq!(analysis.connected_components.len(), 2);
        assert!(analysis.connected_components.contains(&vec![0, 1]));
        assert!(analysis.connected_components.contains(&vec![2]));

        // One edge, one data dependency, sized from the producer's shape.
        assert_eq!(analysis.data_dependencies.len(), 1);
        let dependency = &analysis.data_dependencies[0];
        assert_eq!(dependency.from, 0);
        assert_eq!(dependency.to, 1);
        assert_eq!(
            dependency.data_size,
            2 * 2 * 4,
            "a 2x2 f32 tensor is 16 bytes"
        );
        assert!((dependency.latency_impact - 10.0).abs() < 1e-9);

        // The IR has no loops, and that is recorded as "not analysed".
        assert!(
            analysis.loop_analysis.is_none(),
            "an all-empty LoopAnalysis would look like a completed analysis"
        );
    }

    /// Regression test: `analyze_parallelization` returned a fixed
    /// `balance_score: 0.8`, `synchronization_overhead: 0.1`,
    /// `network_utilization: 0.5` and `latency_sensitivity: 0.3`.
    #[test]
    fn test_parallelization_metrics_are_measured_from_the_graph() {
        let mut analyzer = GraphAnalyzer::new(HardwareTarget::default());
        let graph = two_component_graph();

        let analysis = analyzer.analyze_dependencies(&graph).expect("analysis failed");
        let parallelization = &analysis.parallelization;

        assert_eq!(parallelization.parallel_regions.len(), 2);
        // Component {0,1} carries 15 units of work, {2} carries 1.
        let mut work = parallelization.load_balance_analysis.work_distribution.clone();
        work.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        assert!((work[0] - 1.0).abs() < 1e-9, "got {work:?}");
        assert!((work[1] - 15.0).abs() < 1e-9, "got {work:?}");

        let balance = parallelization.load_balance_analysis.balance_score;
        assert_ne!(balance, 0.8, "the hardcoded balance score must be gone");
        assert!(
            balance < 0.6,
            "a 15:1 split is badly balanced, got {balance}"
        );

        // Nothing models synchronization or the network, so nothing is claimed.
        assert_eq!(
            parallelization.load_balance_analysis.synchronization_overhead,
            0.0
        );
        assert_eq!(
            parallelization.communication_analysis.network_utilization,
            0.0
        );
        assert_eq!(
            parallelization.communication_analysis.latency_sensitivity,
            0.0
        );
        // Communication volume *is* derivable: the single 16-byte edge.
        assert_eq!(
            parallelization.communication_analysis.communication_volume,
            16
        );
    }

    /// Regression test: `analyze_allocation_patterns` and
    /// `find_reuse_opportunities` returned empty vectors, and
    /// `analyze_fragmentation` returned a fixed 0.1/1GB/0.9 result.
    #[test]
    fn test_memory_analysis_reflects_the_real_graph() {
        let mut analyzer = GraphAnalyzer::new(HardwareTarget::default());
        let graph = two_component_graph();

        let analysis = analyzer.analyze_memory(&graph).expect("analysis failed");

        // Node 0's output is consumed once (temporary); nodes 1 and 2 produce
        // graph outputs (long-lived).
        assert!(!analysis.allocation_patterns.is_empty());
        let temporary = analysis
            .allocation_patterns
            .iter()
            .find(|pattern| matches!(pattern.pattern_type, AllocationType::Temporary))
            .expect("the single-use tensor must be classified temporary");
        assert_eq!(temporary.frequency, 1);
        assert_eq!(temporary.total_size, 16);

        // Node 0's 2x2 output can reuse node 1's 2x2 output buffer.
        assert_eq!(analysis.reuse_opportunities.len(), 1);
        assert_eq!(analysis.reuse_opportunities[0].tensor_id, 0);
        assert_eq!(analysis.reuse_opportunities[0].memory_savings, 16);

        assert!(
            analysis.fragmentation_analysis.is_none(),
            "fragmentation cannot be derived from a static graph and must not be invented"
        );
    }

    /// An empty graph yields empty analyses, not invented ones.
    #[test]
    fn test_empty_graph_analyses_are_empty() {
        let mut analyzer = GraphAnalyzer::new(HardwareTarget::default());
        let graph = ComputationGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: HashMap::new(),
        };

        let dependencies = analyzer.analyze_dependencies(&graph).expect("analysis failed");
        assert!(dependencies.connected_components.is_empty());
        assert!(dependencies.data_dependencies.is_empty());
        assert!(dependencies.parallelization.parallel_regions.is_empty());
    }
    use crate::compiler::{ComputationGraph, GraphNode, HardwareTarget};

    fn create_test_graph() -> ComputationGraph {
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

        graph.add_node(node1);
        graph
    }

    #[test]
    fn test_graph_analyzer_creation() {
        let hardware = HardwareTarget::default();
        let analyzer = GraphAnalyzer::new(hardware);
        assert_eq!(analyzer.analysis_cache.len(), 0);
    }

    #[test]
    fn test_performance_analysis() {
        let hardware = HardwareTarget::default();
        let mut analyzer = GraphAnalyzer::new(hardware);
        let graph = create_test_graph();

        let result = analyzer.analyze_performance(&graph);
        assert!(result.is_ok());

        let analysis = result.expect("operation failed in test");
        assert!(analysis.total_execution_time_ms >= 0.0);
    }

    #[test]
    fn test_memory_analysis() {
        let hardware = HardwareTarget::default();
        let mut analyzer = GraphAnalyzer::new(hardware);
        let graph = create_test_graph();

        let result = analyzer.analyze_memory(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dependency_analysis() {
        let hardware = HardwareTarget::default();
        let mut analyzer = GraphAnalyzer::new(hardware);
        let graph = create_test_graph();

        let result = analyzer.analyze_dependencies(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_critical_path_analysis() {
        let hardware = HardwareTarget::default();
        let analyzer = GraphAnalyzer::new(hardware);
        let graph = create_test_graph();

        let result = analyzer.find_critical_path(&graph);
        assert!(result.is_ok());
        assert!(!result.expect("operation failed in test").is_empty());
    }

    #[test]
    fn test_topological_sort() {
        let hardware = HardwareTarget::default();
        let analyzer = GraphAnalyzer::new(hardware);
        let graph = create_test_graph();

        let result = analyzer.topological_sort(&graph);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("operation failed in test").len(),
            graph.nodes.len()
        );
    }

    /// `available_memory_bytes` is the target-device capacity table shared by
    /// `predict_hardware_utilization` and `simulate_memory_usage`. It must
    /// vary by device type, not return one number for everything.
    #[test]
    fn test_available_memory_bytes_varies_by_device_type() {
        let gpu = GraphAnalyzer::new(HardwareTarget {
            device_type: DeviceType::GPU,
            ..HardwareTarget::default()
        });
        let cpu = GraphAnalyzer::new(HardwareTarget {
            device_type: DeviceType::CPU,
            ..HardwareTarget::default()
        });
        let other = GraphAnalyzer::new(HardwareTarget {
            device_type: DeviceType::TPU,
            ..HardwareTarget::default()
        });

        assert_ne!(gpu.available_memory_bytes(), cpu.available_memory_bytes());
        assert_ne!(cpu.available_memory_bytes(), other.available_memory_bytes());
        assert_eq!(gpu.available_memory_bytes(), 16e9);
        assert_eq!(cpu.available_memory_bytes(), 64e9);
        assert_eq!(other.available_memory_bytes(), 8e9);
    }

    /// Regression test: `predict_hardware_utilization` published a fixed
    /// `cache_hit_rate_prediction: 0.8` for every graph. There is no cache
    /// simulation model behind that number, so it is now `None`.
    #[test]
    fn test_cache_hit_rate_prediction_is_honestly_absent() {
        let hardware = HardwareTarget::default();
        let analyzer = GraphAnalyzer::new(hardware);
        let graph = create_test_graph();

        let utilization = analyzer.predict_hardware_utilization(&graph).expect("prediction failed");
        assert!(
            utilization.cache_hit_rate_prediction.is_none(),
            "no cache simulation model exists to back this field; a Some(_) would be fabricated"
        );

        // The Option must round-trip through serde like the rest of the
        // struct (HardwareUtilization derives Serialize/Deserialize).
        let json = serde_json::to_string(&utilization).expect("serialize failed");
        let round_tripped: HardwareUtilization =
            serde_json::from_str(&json).expect("deserialize failed");
        assert!(round_tripped.cache_hit_rate_prediction.is_none());
    }

    /// Regression test: `simulate_memory_usage` divided by a hardcoded
    /// `16e9` regardless of `self.hardware_target`, even though
    /// `predict_hardware_utilization` (in the same impl block) already had a
    /// real per-device-type capacity table. It now shares that table
    /// (`available_memory_bytes`), so the same graph must report different
    /// memory pressure against different target devices.
    #[test]
    fn test_simulate_memory_usage_reflects_target_device_capacity() {
        let graph = create_test_graph();

        let gpu_analyzer = GraphAnalyzer::new(HardwareTarget {
            device_type: DeviceType::GPU,
            ..HardwareTarget::default()
        });
        let cpu_analyzer = GraphAnalyzer::new(HardwareTarget {
            device_type: DeviceType::CPU,
            ..HardwareTarget::default()
        });

        let gpu_snapshots = gpu_analyzer.simulate_memory_usage(&graph).expect("simulation failed");
        let cpu_snapshots = cpu_analyzer.simulate_memory_usage(&graph).expect("simulation failed");

        assert_eq!(gpu_snapshots.len(), cpu_snapshots.len());
        assert!(!gpu_snapshots.is_empty());
        for (gpu_snap, cpu_snap) in gpu_snapshots.iter().zip(cpu_snapshots.iter()) {
            assert_eq!(
                gpu_snap.allocated_memory, cpu_snap.allocated_memory,
                "same graph must produce the same real allocation regardless of target device"
            );
            assert!(
                gpu_snap.memory_pressure > cpu_snap.memory_pressure,
                "the same allocation must show higher pressure against the GPU's smaller \
                 assumed capacity (16GB) than the CPU's larger one (64GB): gpu={}, cpu={}",
                gpu_snap.memory_pressure,
                cpu_snap.memory_pressure
            );
        }
    }
}
