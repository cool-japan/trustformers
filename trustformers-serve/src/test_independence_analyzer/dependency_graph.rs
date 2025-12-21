//! Dependency Graph Management and Analysis
//!
//! This module provides comprehensive dependency graph management including
//! graph construction, analysis, cycle detection, topological sorting,
//! and dependency path analysis for test independence.

use super::types::*;

use chrono::Utc;
use parking_lot::RwLock;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};
use tracing::{debug, info};

use crate::test_parallelization::TestDependency;

// Re-export types needed by other modules
pub use super::types::{DependencyEdge, GraphMetadata};

// ================================================================================================
// Dependency Graph Implementation
// ================================================================================================

/// Dependency graph for tracking test relationships
#[derive(Debug)]
pub struct DependencyGraph {
    /// Adjacency list representation (outgoing edges)
    adjacency_list: Arc<RwLock<HashMap<String, Vec<DependencyEdge>>>>,

    /// Reverse adjacency list for reverse lookups (incoming edges)
    reverse_adjacency_list: Arc<RwLock<HashMap<String, Vec<DependencyEdge>>>>,

    /// Graph metadata and analysis results
    metadata: Arc<RwLock<GraphMetadata>>,

    /// Graph analysis algorithms
    algorithms: Arc<GraphAlgorithms>,

    /// Graph validation rules
    validation_rules: Vec<GraphValidationRule>,
}

impl DependencyGraph {
    /// Create a new dependency graph
    pub fn new() -> Self {
        Self {
            adjacency_list: Arc::new(RwLock::new(HashMap::new())),
            reverse_adjacency_list: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(GraphMetadata::default())),
            algorithms: Arc::new(GraphAlgorithms::new()),
            validation_rules: vec![
                GraphValidationRule::new_no_self_loops(),
                GraphValidationRule::new_no_cycles(),
                GraphValidationRule::new_weight_validation(),
            ],
        }
    }

    /// Add a node (test) to the graph
    pub fn add_node(&self, test_id: &str) -> AnalysisResult<()> {
        let mut adj_list = self.adjacency_list.write();
        let mut rev_adj_list = self.reverse_adjacency_list.write();

        if !adj_list.contains_key(test_id) {
            adj_list.insert(test_id.to_string(), Vec::new());
            rev_adj_list.insert(test_id.to_string(), Vec::new());

            // Update metadata
            let mut metadata = self.metadata.write();
            metadata.node_count = adj_list.len();
            metadata.last_analysis = Utc::now();

            debug!(test_id = %test_id, "Added node to dependency graph");
        }

        Ok(())
    }

    /// Remove a node from the graph
    pub fn remove_node(&self, test_id: &str) -> AnalysisResult<()> {
        let mut adj_list = self.adjacency_list.write();
        let mut rev_adj_list = self.reverse_adjacency_list.write();

        // Remove all edges to/from this node
        if let Some(outgoing_edges) = adj_list.remove(test_id) {
            for edge in outgoing_edges {
                if let Some(target_edges) = rev_adj_list.get_mut(&edge.target) {
                    target_edges.retain(|e| e.target != test_id);
                }
            }
        }

        if let Some(incoming_edges) = rev_adj_list.remove(test_id) {
            for edge in incoming_edges {
                if let Some(source_edges) = adj_list.get_mut(&edge.target) {
                    source_edges.retain(|e| e.target != test_id);
                }
            }
        }

        // Remove from all other nodes
        for (_, edges) in adj_list.iter_mut() {
            edges.retain(|edge| edge.target != test_id);
        }

        for (_, edges) in rev_adj_list.iter_mut() {
            edges.retain(|edge| edge.target != test_id);
        }

        // Update metadata
        let mut metadata = self.metadata.write();
        metadata.node_count = adj_list.len();
        metadata.edge_count = adj_list.values().map(|edges| edges.len()).sum();
        metadata.last_analysis = Utc::now();

        info!(test_id = %test_id, "Removed node from dependency graph");

        Ok(())
    }

    /// Add an edge (dependency) to the graph
    pub fn add_edge(
        &self,
        source: &str,
        target: &str,
        dependency: TestDependency,
        weight: f32,
    ) -> AnalysisResult<()> {
        // Validate edge
        self.validate_edge(source, target, weight)?;

        let edge_metadata = EdgeMetadata {
            created_at: Utc::now(),
            last_validated: Utc::now(),
            confidence: 0.8, // Default confidence
            tags: Vec::new(),
            properties: HashMap::new(),
        };

        let edge = DependencyEdge {
            target: target.to_string(),
            dependency,
            weight,
            metadata: edge_metadata,
        };

        let reverse_edge = DependencyEdge {
            target: source.to_string(),
            dependency: edge.dependency.clone(),
            weight: edge.weight,
            metadata: edge.metadata.clone(),
        };

        // Add to adjacency lists
        {
            let mut adj_list = self.adjacency_list.write();
            let mut rev_adj_list = self.reverse_adjacency_list.write();

            // Ensure nodes exist
            adj_list.entry(source.to_string()).or_default();
            adj_list.entry(target.to_string()).or_default();
            rev_adj_list.entry(source.to_string()).or_default();
            rev_adj_list.entry(target.to_string()).or_default();

            // Add edges
            adj_list.get_mut(source).unwrap().push(edge);
            rev_adj_list.get_mut(target).unwrap().push(reverse_edge);
        }

        // Update metadata
        {
            let mut metadata = self.metadata.write();
            metadata.edge_count =
                self.adjacency_list.read().values().map(|edges| edges.len()).sum();
            metadata.last_analysis = Utc::now();

            // Recalculate density
            if metadata.node_count > 1 {
                let max_edges = metadata.node_count * (metadata.node_count - 1);
                metadata.density = metadata.edge_count as f32 / max_edges as f32;
            }
        }

        debug!(
            source = %source,
            target = %target,
            weight = %weight,
            "Added edge to dependency graph"
        );

        Ok(())
    }

    /// Remove an edge from the graph
    pub fn remove_edge(&self, source: &str, target: &str) -> AnalysisResult<bool> {
        let mut removed = false;

        {
            let mut adj_list = self.adjacency_list.write();
            let mut rev_adj_list = self.reverse_adjacency_list.write();

            // Remove from forward adjacency list
            if let Some(edges) = adj_list.get_mut(source) {
                let initial_len = edges.len();
                edges.retain(|edge| edge.target != target);
                removed = edges.len() < initial_len;
            }

            // Remove from reverse adjacency list
            if let Some(edges) = rev_adj_list.get_mut(target) {
                edges.retain(|edge| edge.target != source);
            }
        }

        if removed {
            // Update metadata
            let mut metadata = self.metadata.write();
            metadata.edge_count =
                self.adjacency_list.read().values().map(|edges| edges.len()).sum();
            metadata.last_analysis = Utc::now();

            debug!(
                source = %source,
                target = %target,
                "Removed edge from dependency graph"
            );
        }

        Ok(removed)
    }

    /// Get all dependencies for a test
    pub fn get_dependencies(&self, test_id: &str) -> Vec<DependencyEdge> {
        self.adjacency_list.read().get(test_id).cloned().unwrap_or_default()
    }

    /// Get all dependents (reverse dependencies) for a test
    pub fn get_dependents(&self, test_id: &str) -> Vec<DependencyEdge> {
        self.reverse_adjacency_list.read().get(test_id).cloned().unwrap_or_default()
    }

    /// Check if there's a path between two tests
    pub fn has_path(&self, source: &str, target: &str) -> bool {
        if source == target {
            return true;
        }

        let adj_list = self.adjacency_list.read();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(source.to_string());
        visited.insert(source.to_string());

        while let Some(current) = queue.pop_front() {
            if let Some(edges) = adj_list.get(&current) {
                for edge in edges {
                    if edge.target == target {
                        return true;
                    }

                    if !visited.contains(&edge.target) {
                        visited.insert(edge.target.clone());
                        queue.push_back(edge.target.clone());
                    }
                }
            }
        }

        false
    }

    /// Find shortest path between two tests
    pub fn find_shortest_path(&self, source: &str, target: &str) -> Option<Vec<String>> {
        if source == target {
            return Some(vec![source.to_string()]);
        }

        let adj_list = self.adjacency_list.read();
        let mut visited = HashMap::new();
        let mut queue = VecDeque::new();

        queue.push_back((source.to_string(), vec![source.to_string()]));
        visited.insert(source.to_string(), true);

        while let Some((current, path)) = queue.pop_front() {
            if let Some(edges) = adj_list.get(&current) {
                for edge in edges {
                    if edge.target == target {
                        let mut result_path = path.clone();
                        result_path.push(edge.target.clone());
                        return Some(result_path);
                    }

                    if !visited.contains_key(&edge.target) {
                        visited.insert(edge.target.clone(), true);
                        let mut new_path = path.clone();
                        new_path.push(edge.target.clone());
                        queue.push_back((edge.target.clone(), new_path));
                    }
                }
            }
        }

        None
    }

    /// Detect cycles in the graph
    pub fn detect_cycles(&self) -> AnalysisResult<Vec<Vec<String>>> {
        self.algorithms.detect_cycles(&self.adjacency_list.read())
    }

    /// Get strongly connected components
    pub fn get_strongly_connected_components(&self) -> AnalysisResult<Vec<Vec<String>>> {
        self.algorithms.tarjan_scc(&self.adjacency_list.read())
    }

    /// Perform topological sort
    pub fn topological_sort(&self) -> AnalysisResult<Option<Vec<String>>> {
        self.algorithms.topological_sort(&self.adjacency_list.read())
    }

    /// Calculate graph metrics
    pub fn calculate_metrics(&self) -> AnalysisResult<GraphAnalysisMetrics> {
        self.algorithms.calculate_graph_metrics(
            &self.adjacency_list.read(),
            &self.reverse_adjacency_list.read(),
        )
    }

    /// Analyze graph and update metadata
    pub fn analyze(&self) -> AnalysisResult<()> {
        let adj_list = self.adjacency_list.read();
        let rev_adj_list = self.reverse_adjacency_list.read();

        let mut metadata = self.metadata.write();

        // Update basic counts
        metadata.node_count = adj_list.len();
        metadata.edge_count = adj_list.values().map(|edges| edges.len()).sum();

        // Calculate density
        if metadata.node_count > 1 {
            let max_edges = metadata.node_count * (metadata.node_count - 1);
            metadata.density = metadata.edge_count as f32 / max_edges as f32;
        }

        // Detect cycles
        let cycles = self.algorithms.detect_cycles(&adj_list)?;
        metadata.properties.has_cycles = !cycles.is_empty();
        metadata.properties.is_dag = cycles.is_empty();

        // Get strongly connected components
        metadata.strongly_connected_components = self.algorithms.tarjan_scc(&adj_list)?;

        // Get topological order if DAG
        if metadata.properties.is_dag {
            metadata.topological_order = self.algorithms.topological_sort(&adj_list)?;
        }

        // Calculate graph properties
        let metrics = self.algorithms.calculate_graph_metrics(&adj_list, &rev_adj_list)?;
        metadata.properties.max_path_length = metrics.max_path_length;
        metadata.properties.average_degree = metrics.average_degree;
        metadata.properties.clustering_coefficient = metrics.clustering_coefficient;

        metadata.last_analysis = Utc::now();

        info!("Graph analysis completed");

        Ok(())
    }

    /// Get current graph metadata
    pub fn get_metadata(&self) -> GraphMetadata {
        (*self.metadata.read()).clone()
    }

    /// Get all nodes in the graph
    pub fn get_all_nodes(&self) -> Vec<String> {
        self.adjacency_list.read().keys().cloned().collect()
    }

    /// Get graph size (number of nodes and edges)
    pub fn get_size(&self) -> (usize, usize) {
        let adj_list = self.adjacency_list.read();
        let node_count = adj_list.len();
        let edge_count = adj_list.values().map(|edges| edges.len()).sum();
        (node_count, edge_count)
    }

    /// Clear the entire graph
    pub fn clear(&self) {
        self.adjacency_list.write().clear();
        self.reverse_adjacency_list.write().clear();
        *self.metadata.write() = GraphMetadata::default();

        info!("Dependency graph cleared");
    }

    /// Export graph to DOT format for visualization
    pub fn export_to_dot(&self, include_weights: bool) -> String {
        let adj_list = self.adjacency_list.read();
        let mut dot = String::from("digraph DependencyGraph {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Add nodes
        for node in adj_list.keys() {
            dot.push_str(&format!("  \"{}\";\n", node));
        }

        dot.push('\n');

        // Add edges
        for (source, edges) in adj_list.iter() {
            for edge in edges {
                if include_weights {
                    dot.push_str(&format!(
                        "  \"{}\" -> \"{}\" [label=\"{:.2}\"];\n",
                        source, edge.target, edge.weight
                    ));
                } else {
                    dot.push_str(&format!("  \"{}\" -> \"{}\";\n", source, edge.target));
                }
            }
        }

        dot.push_str("}\n");
        dot
    }

    // ============================================================================================
    // Private Implementation Methods
    // ============================================================================================

    /// Validate an edge before adding
    fn validate_edge(&self, source: &str, target: &str, weight: f32) -> AnalysisResult<()> {
        // Check for self-loops
        if source == target {
            return Err(AnalysisError::GraphError {
                message: "Self-loops are not allowed".to_string(),
            });
        }

        // Validate weight
        if !(0.0..=1.0).contains(&weight) {
            return Err(AnalysisError::GraphError {
                message: format!("Edge weight {} is out of range [0.0, 1.0]", weight),
            });
        }

        // Apply validation rules
        for rule in &self.validation_rules {
            rule.validate(source, target, weight)?;
        }

        Ok(())
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// Graph Algorithms Implementation
// ================================================================================================

/// Graph algorithms for dependency analysis
#[derive(Debug)]
pub struct GraphAlgorithms;

impl Default for GraphAlgorithms {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAlgorithms {
    /// Create new graph algorithms instance
    pub fn new() -> Self {
        Self
    }

    /// Detect cycles using DFS
    pub fn detect_cycles(
        &self,
        adj_list: &HashMap<String, Vec<DependencyEdge>>,
    ) -> AnalysisResult<Vec<Vec<String>>> {
        let mut white_set: HashSet<String> = adj_list.keys().cloned().collect();
        let mut gray_set = HashSet::new();
        let mut black_set = HashSet::new();
        let mut cycles = Vec::new();

        for node in adj_list.keys() {
            if white_set.contains(node) {
                let mut path = Vec::new();
                self.dfs_cycle_detection(
                    node,
                    adj_list,
                    &mut white_set,
                    &mut gray_set,
                    &mut black_set,
                    &mut path,
                    &mut cycles,
                )?;
            }
        }

        Ok(cycles)
    }

    /// DFS helper for cycle detection
    fn dfs_cycle_detection(
        &self,
        node: &str,
        adj_list: &HashMap<String, Vec<DependencyEdge>>,
        white_set: &mut HashSet<String>,
        gray_set: &mut HashSet<String>,
        black_set: &mut HashSet<String>,
        path: &mut Vec<String>,
        cycles: &mut Vec<Vec<String>>,
    ) -> AnalysisResult<()> {
        white_set.remove(node);
        gray_set.insert(node.to_string());
        path.push(node.to_string());

        if let Some(edges) = adj_list.get(node) {
            for edge in edges {
                let target = &edge.target;

                if gray_set.contains(target) {
                    // Found a cycle
                    if let Some(cycle_start) = path.iter().position(|n| n == target) {
                        let cycle = path[cycle_start..].to_vec();
                        cycles.push(cycle);
                    }
                } else if white_set.contains(target) {
                    self.dfs_cycle_detection(
                        target, adj_list, white_set, gray_set, black_set, path, cycles,
                    )?;
                }
            }
        }

        gray_set.remove(node);
        black_set.insert(node.to_string());
        path.pop();

        Ok(())
    }

    /// Tarjan's strongly connected components algorithm
    pub fn tarjan_scc(
        &self,
        adj_list: &HashMap<String, Vec<DependencyEdge>>,
    ) -> AnalysisResult<Vec<Vec<String>>> {
        let mut index = 0;
        let mut stack = Vec::new();
        let mut indices = HashMap::new();
        let mut lowlinks = HashMap::new();
        let mut on_stack = HashSet::new();
        let mut sccs = Vec::new();

        for node in adj_list.keys() {
            if !indices.contains_key(node) {
                self.tarjan_strongconnect(
                    node,
                    adj_list,
                    &mut index,
                    &mut stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut on_stack,
                    &mut sccs,
                )?;
            }
        }

        Ok(sccs)
    }

    /// Tarjan's strongconnect helper
    fn tarjan_strongconnect(
        &self,
        v: &str,
        adj_list: &HashMap<String, Vec<DependencyEdge>>,
        index: &mut usize,
        stack: &mut Vec<String>,
        indices: &mut HashMap<String, usize>,
        lowlinks: &mut HashMap<String, usize>,
        on_stack: &mut HashSet<String>,
        sccs: &mut Vec<Vec<String>>,
    ) -> AnalysisResult<()> {
        indices.insert(v.to_string(), *index);
        lowlinks.insert(v.to_string(), *index);
        *index += 1;
        stack.push(v.to_string());
        on_stack.insert(v.to_string());

        if let Some(edges) = adj_list.get(v) {
            for edge in edges {
                let w = &edge.target;

                if !indices.contains_key(w) {
                    self.tarjan_strongconnect(
                        w, adj_list, index, stack, indices, lowlinks, on_stack, sccs,
                    )?;
                    let w_lowlink = lowlinks[w];
                    let v_lowlink = lowlinks[v];
                    lowlinks.insert(v.to_string(), v_lowlink.min(w_lowlink));
                } else if on_stack.contains(w) {
                    let w_index = indices[w];
                    let v_lowlink = lowlinks[v];
                    lowlinks.insert(v.to_string(), v_lowlink.min(w_index));
                }
            }
        }

        let v_index = indices[v];
        let v_lowlink = lowlinks[v];

        if v_lowlink == v_index {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc.push(w.clone());
                if w == v {
                    break;
                }
            }
            sccs.push(scc);
        }

        Ok(())
    }

    /// Topological sort using Kahn's algorithm
    pub fn topological_sort(
        &self,
        adj_list: &HashMap<String, Vec<DependencyEdge>>,
    ) -> AnalysisResult<Option<Vec<String>>> {
        let mut in_degree = HashMap::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Calculate in-degrees
        for node in adj_list.keys() {
            in_degree.insert(node.clone(), 0);
        }

        for edges in adj_list.values() {
            for edge in edges {
                *in_degree.entry(edge.target.clone()).or_insert(0) += 1;
            }
        }

        // Find nodes with no incoming edges
        for (node, degree) in &in_degree {
            if *degree == 0 {
                queue.push_back(node.clone());
            }
        }

        // Process nodes
        while let Some(node) = queue.pop_front() {
            result.push(node.clone());

            if let Some(edges) = adj_list.get(&node) {
                for edge in edges {
                    let target_degree = in_degree.get_mut(&edge.target).unwrap();
                    *target_degree -= 1;

                    if *target_degree == 0 {
                        queue.push_back(edge.target.clone());
                    }
                }
            }
        }

        // Check if graph is DAG
        if result.len() == adj_list.len() {
            Ok(Some(result))
        } else {
            Ok(None) // Graph has cycles
        }
    }

    /// Calculate comprehensive graph metrics
    pub fn calculate_graph_metrics(
        &self,
        adj_list: &HashMap<String, Vec<DependencyEdge>>,
        _rev_adj_list: &HashMap<String, Vec<DependencyEdge>>,
    ) -> AnalysisResult<GraphAnalysisMetrics> {
        let node_count = adj_list.len();
        let edge_count: usize = adj_list.values().map(|edges| edges.len()).sum();

        let mut metrics = GraphAnalysisMetrics {
            node_count,
            edge_count,
            max_path_length: 0,
            average_degree: 0.0,
            clustering_coefficient: 0.0,
            diameter: 0,
            radius: usize::MAX,
            center_nodes: Vec::new(),
            peripheral_nodes: Vec::new(),
        };

        if node_count == 0 {
            return Ok(metrics);
        }

        // Calculate average degree
        metrics.average_degree = (2 * edge_count) as f32 / node_count as f32;

        // Calculate clustering coefficient
        metrics.clustering_coefficient = self.calculate_clustering_coefficient(adj_list)?;

        // Calculate max path length, diameter, and radius
        let (max_path, diameter, radius) = self.calculate_path_metrics(adj_list)?;
        metrics.max_path_length = max_path;
        metrics.diameter = diameter;
        metrics.radius = radius;

        // Find center and peripheral nodes
        (metrics.center_nodes, metrics.peripheral_nodes) =
            self.find_center_and_peripheral_nodes(adj_list)?;

        Ok(metrics)
    }

    /// Calculate clustering coefficient
    fn calculate_clustering_coefficient(
        &self,
        adj_list: &HashMap<String, Vec<DependencyEdge>>,
    ) -> AnalysisResult<f32> {
        let mut total_coefficient = 0.0;
        let mut node_count = 0;

        for edges in adj_list.values() {
            if edges.len() < 2 {
                continue;
            }

            let neighbors: HashSet<String> = edges.iter().map(|e| e.target.clone()).collect();
            let mut triangle_count = 0;

            for edge1 in edges {
                if let Some(neighbor_edges) = adj_list.get(&edge1.target) {
                    for edge2 in neighbor_edges {
                        if neighbors.contains(&edge2.target) {
                            triangle_count += 1;
                        }
                    }
                }
            }

            let possible_triangles = edges.len() * (edges.len() - 1);
            let coefficient = if possible_triangles > 0 {
                triangle_count as f32 / possible_triangles as f32
            } else {
                0.0
            };

            total_coefficient += coefficient;
            node_count += 1;
        }

        Ok(if node_count > 0 { total_coefficient / node_count as f32 } else { 0.0 })
    }

    /// Calculate path metrics (max path length, diameter, radius)
    fn calculate_path_metrics(
        &self,
        adj_list: &HashMap<String, Vec<DependencyEdge>>,
    ) -> AnalysisResult<(usize, usize, usize)> {
        let mut max_path_length = 0;
        let mut diameter = 0;
        let mut radius = usize::MAX;

        for start_node in adj_list.keys() {
            let distances = self.bfs_distances(start_node, adj_list);
            let max_distance = distances.values().max().copied().unwrap_or(0);

            max_path_length = max_path_length.max(max_distance);
            diameter = diameter.max(max_distance);
            radius = radius.min(max_distance);
        }

        if radius == usize::MAX {
            radius = 0;
        }

        Ok((max_path_length, diameter, radius))
    }

    /// BFS to calculate distances from a source node
    fn bfs_distances(
        &self,
        source: &str,
        adj_list: &HashMap<String, Vec<DependencyEdge>>,
    ) -> HashMap<String, usize> {
        let mut distances = HashMap::new();
        let mut queue = VecDeque::new();

        distances.insert(source.to_string(), 0);
        queue.push_back(source.to_string());

        while let Some(current) = queue.pop_front() {
            let current_distance = distances[&current];

            if let Some(edges) = adj_list.get(&current) {
                for edge in edges {
                    if !distances.contains_key(&edge.target) {
                        distances.insert(edge.target.clone(), current_distance + 1);
                        queue.push_back(edge.target.clone());
                    }
                }
            }
        }

        distances
    }

    /// Find center and peripheral nodes
    fn find_center_and_peripheral_nodes(
        &self,
        adj_list: &HashMap<String, Vec<DependencyEdge>>,
    ) -> AnalysisResult<(Vec<String>, Vec<String>)> {
        let mut eccentricities = HashMap::new();

        // Calculate eccentricity for each node
        for node in adj_list.keys() {
            let distances = self.bfs_distances(node, adj_list);
            let eccentricity = distances.values().max().copied().unwrap_or(0);
            eccentricities.insert(node.clone(), eccentricity);
        }

        let min_eccentricity = eccentricities.values().min().copied().unwrap_or(0);
        let max_eccentricity = eccentricities.values().max().copied().unwrap_or(0);

        let center_nodes: Vec<String> = eccentricities
            .iter()
            .filter(|(_, &ecc)| ecc == min_eccentricity)
            .map(|(node, _)| node.clone())
            .collect();

        let peripheral_nodes: Vec<String> = eccentricities
            .iter()
            .filter(|(_, &ecc)| ecc == max_eccentricity)
            .map(|(node, _)| node.clone())
            .collect();

        Ok((center_nodes, peripheral_nodes))
    }
}

// ================================================================================================
// Supporting Types and Implementations
// ================================================================================================

/// Graph analysis metrics
#[derive(Debug, Clone)]
pub struct GraphAnalysisMetrics {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Maximum path length in the graph
    pub max_path_length: usize,
    /// Average degree of nodes
    pub average_degree: f32,
    /// Clustering coefficient
    pub clustering_coefficient: f32,
    /// Graph diameter
    pub diameter: usize,
    /// Graph radius
    pub radius: usize,
    /// Center nodes (minimum eccentricity)
    pub center_nodes: Vec<String>,
    /// Peripheral nodes (maximum eccentricity)
    pub peripheral_nodes: Vec<String>,
}

/// Graph validation rule
pub struct GraphValidationRule {
    name: String,
    validator: Box<dyn Fn(&str, &str, f32) -> AnalysisResult<()> + Send + Sync>,
}

impl GraphValidationRule {
    /// Create a rule that prevents self-loops
    pub fn new_no_self_loops() -> Self {
        Self {
            name: "no_self_loops".to_string(),
            validator: Box::new(|source, target, _weight| {
                if source == target {
                    Err(AnalysisError::GraphError {
                        message: "Self-loops are not allowed".to_string(),
                    })
                } else {
                    Ok(())
                }
            }),
        }
    }

    /// Create a rule that prevents cycles (for DAG enforcement)
    pub fn new_no_cycles() -> Self {
        Self {
            name: "no_cycles".to_string(),
            validator: Box::new(|_source, _target, _weight| {
                // This would need access to the graph to check for cycles
                // For now, just return Ok - cycle detection is done elsewhere
                Ok(())
            }),
        }
    }

    /// Create a rule that validates edge weights
    pub fn new_weight_validation() -> Self {
        Self {
            name: "weight_validation".to_string(),
            validator: Box::new(|_source, _target, weight| {
                if !(0.0..=1.0).contains(&weight) {
                    Err(AnalysisError::GraphError {
                        message: format!(
                            "Invalid weight: {} (must be between 0.0 and 1.0)",
                            weight
                        ),
                    })
                } else {
                    Ok(())
                }
            }),
        }
    }

    /// Validate an edge
    pub fn validate(&self, source: &str, target: &str, weight: f32) -> AnalysisResult<()> {
        (self.validator)(source, target, weight)
    }
}

impl std::fmt::Debug for GraphValidationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphValidationRule").field("name", &self.name).finish()
    }
}
