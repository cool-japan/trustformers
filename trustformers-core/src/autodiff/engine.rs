//! Automatic differentiation engine.
//!
//! This module provides the main engine for managing automatic differentiation,
//! including gradient computation modes and optimization settings.

#![allow(unused_variables)] // Autodiff engine with reserved parameters

use super::graph::ComputationGraph;
use super::tape::GradientTape;
use super::variable::{GraphRef, Variable};
use crate::errors::{tensor_op_error, Result};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Gradient computation modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientMode {
    /// Forward-mode automatic differentiation
    Forward,
    /// Reverse-mode automatic differentiation (backpropagation)
    Reverse,
    /// Mixed-mode (forward for small graphs, reverse for large graphs)
    Mixed,
}

/// Configuration for the automatic differentiation engine
#[derive(Debug, Clone)]
pub struct AutodiffConfig {
    /// Gradient computation mode
    pub mode: GradientMode,
    /// Whether to enable gradient computation
    pub enabled: bool,
    /// Whether to detect anomalies in gradient computation
    pub detect_anomalies: bool,
    /// Whether to retain the computational graph after backward pass
    pub retain_graph: bool,
    /// Maximum number of operations to cache
    pub max_cache_size: usize,
    /// Whether to use graph optimization
    pub optimize_graph: bool,
    /// Whether to enable gradient checkpointing
    pub gradient_checkpointing: bool,
}

impl Default for AutodiffConfig {
    fn default() -> Self {
        Self {
            mode: GradientMode::Reverse,
            enabled: true,
            detect_anomalies: false,
            retain_graph: false,
            max_cache_size: 10000,
            optimize_graph: true,
            gradient_checkpointing: false,
        }
    }
}

/// Main automatic differentiation engine
#[derive(Debug)]
pub struct AutodiffEngine {
    /// Configuration
    config: AutodiffConfig,
    /// Current computation graph
    graph: GraphRef,
    /// Gradient tape for recording operations
    tape: Arc<Mutex<GradientTape>>,
    /// Cache for compiled operations
    #[allow(dead_code)]
    operation_cache: Arc<Mutex<HashMap<String, CompiledOperation>>>,
    /// Statistics
    stats: Arc<Mutex<AutodiffStats>>,
}

/// Compiled operation for performance optimization
#[derive(Debug, Clone)]
pub struct CompiledOperation {
    /// Operation ID
    pub id: String,
    /// Compiled function
    pub forward_fn: fn(&[&Tensor]) -> Result<Tensor>,
    /// Compiled backward function
    pub backward_fn: fn(&Tensor, &[&Tensor]) -> Result<Vec<Tensor>>,
    /// Operation metadata
    pub metadata: OperationMetadata,
}

/// Metadata for operations
#[derive(Debug, Clone)]
pub struct OperationMetadata {
    /// Operation type
    pub op_type: String,
    /// Input shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Number of parameters
    pub num_parameters: usize,
    /// Estimated FLOPS
    pub estimated_flops: usize,
}

/// Statistics for the autodiff engine
#[derive(Debug, Default, Clone)]
pub struct AutodiffStats {
    /// Number of forward passes
    pub forward_passes: u64,
    /// Number of backward passes
    pub backward_passes: u64,
    /// Total operations executed
    pub total_operations: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Total time spent in forward pass (microseconds)
    pub forward_time_us: u64,
    /// Total time spent in backward pass (microseconds)
    pub backward_time_us: u64,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
    /// Current memory usage (bytes)
    pub current_memory_usage: usize,
}

impl Default for AutodiffEngine {
    fn default() -> Self {
        Self::new(AutodiffConfig::default())
    }
}

impl AutodiffEngine {
    /// Create a new autodiff engine
    pub fn new(config: AutodiffConfig) -> Self {
        let graph = Arc::new(Mutex::new(ComputationGraph::new()));
        let tape = Arc::new(Mutex::new(GradientTape::new()));
        let operation_cache = Arc::new(Mutex::new(HashMap::new()));
        let stats = Arc::new(Mutex::new(AutodiffStats::default()));

        Self {
            config,
            graph,
            tape,
            operation_cache,
            stats,
        }
    }

    /// Enable gradient computation
    pub fn enable_grad(&mut self) {
        self.config.enabled = true;
    }

    /// Disable gradient computation
    pub fn disable_grad(&mut self) {
        self.config.enabled = false;
    }

    /// Check if gradient computation is enabled
    pub fn is_grad_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Set gradient computation mode
    pub fn set_mode(&mut self, mode: GradientMode) {
        self.config.mode = mode;
    }

    /// Get current gradient computation mode
    pub fn mode(&self) -> GradientMode {
        self.config.mode
    }

    /// Enable anomaly detection
    pub fn enable_anomaly_detection(&mut self) {
        self.config.detect_anomalies = true;
    }

    /// Disable anomaly detection
    pub fn disable_anomaly_detection(&mut self) {
        self.config.detect_anomalies = false;
    }

    /// Create a new variable
    pub fn variable(&self, tensor: Tensor, requires_grad: bool) -> Variable {
        Variable::from_graph(
            self.graph.clone(),
            {
                let mut graph = self.graph.lock().unwrap();
                graph.add_node(tensor, requires_grad, None)
            },
            requires_grad,
        )
    }

    /// Create a new variable with a name
    pub fn variable_with_name(
        &self,
        tensor: Tensor,
        requires_grad: bool,
        name: String,
    ) -> Variable {
        Variable::from_graph(
            self.graph.clone(),
            {
                let mut graph = self.graph.lock().unwrap();
                graph.add_node(tensor, requires_grad, Some(name))
            },
            requires_grad,
        )
    }

    /// Compute gradients using the current mode
    pub fn backward(&self, output: &Variable, grad_output: Option<Tensor>) -> Result<()> {
        let start_time = std::time::Instant::now();

        match self.config.mode {
            GradientMode::Forward => self.forward_mode_backward(output, grad_output),
            GradientMode::Reverse => self.reverse_mode_backward(output, grad_output),
            GradientMode::Mixed => self.mixed_mode_backward(output, grad_output),
        }?;

        // Update statistics
        let mut stats = self.stats.lock().unwrap();
        stats.backward_passes += 1;
        stats.backward_time_us += start_time.elapsed().as_micros() as u64;

        Ok(())
    }

    /// Forward-mode automatic differentiation
    fn forward_mode_backward(&self, output: &Variable, grad_output: Option<Tensor>) -> Result<()> {
        // Forward-mode AD is typically used for computing derivatives with respect to few inputs
        // This is a simplified implementation
        let mut graph = self.graph.lock().unwrap();
        graph.backward(output.node_id(), grad_output)
    }

    /// Reverse-mode automatic differentiation (standard backpropagation)
    fn reverse_mode_backward(&self, output: &Variable, grad_output: Option<Tensor>) -> Result<()> {
        let mut graph = self.graph.lock().unwrap();
        graph.backward(output.node_id(), grad_output)
    }

    /// Mixed-mode automatic differentiation
    fn mixed_mode_backward(&self, output: &Variable, grad_output: Option<Tensor>) -> Result<()> {
        // Decide between forward and reverse mode based on graph characteristics
        let graph = self.graph.lock().unwrap();
        let num_nodes = graph.num_nodes();

        // Use forward mode for small graphs, reverse mode for large graphs
        if num_nodes < 100 {
            drop(graph);
            self.forward_mode_backward(output, grad_output)
        } else {
            drop(graph);
            self.reverse_mode_backward(output, grad_output)
        }
    }

    /// Zero all gradients in the computation graph
    pub fn zero_grad(&self) {
        let mut graph = self.graph.lock().unwrap();
        graph.zero_grad();
    }

    /// Get gradient for a variable
    pub fn get_grad(&self, variable: &Variable) -> Result<Option<Tensor>> {
        let graph = self.graph.lock().unwrap();
        Ok(graph.get_gradient(variable.node_id()).cloned())
    }

    /// Clear the computation graph
    pub fn clear_graph(&self) {
        let mut graph = self.graph.lock().unwrap();
        *graph = ComputationGraph::new();

        let mut tape = self.tape.lock().unwrap();
        tape.clear();
    }

    /// Get engine statistics
    pub fn stats(&self) -> AutodiffStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = AutodiffStats::default();
    }

    /// Get the current graph
    pub fn graph(&self) -> GraphRef {
        self.graph.clone()
    }

    /// Optimize the computation graph
    pub fn optimize_graph(&self) -> Result<()> {
        if !self.config.optimize_graph {
            return Ok(());
        }

        let mut graph = self.graph.lock().unwrap();

        // Perform various graph optimizations
        self.eliminate_dead_nodes(&mut graph)?;
        self.fuse_operations(&mut graph)?;
        self.optimize_memory_layout(&mut graph)?;

        Ok(())
    }

    /// Eliminate dead nodes (nodes with no children)
    fn eliminate_dead_nodes(&self, graph: &mut ComputationGraph) -> Result<()> {
        // This is a simplified implementation
        // In practice, you would identify and remove nodes that don't contribute to the output
        Ok(())
    }

    /// Fuse operations where possible
    fn fuse_operations(&self, graph: &mut ComputationGraph) -> Result<()> {
        // This is a simplified implementation
        // In practice, you would identify patterns like Add+Mul and fuse them into FusedAddMul
        Ok(())
    }

    /// Optimize memory layout
    fn optimize_memory_layout(&self, graph: &mut ComputationGraph) -> Result<()> {
        // This is a simplified implementation
        // In practice, you would reorder operations to minimize memory usage
        Ok(())
    }

    /// Execute a function with gradient computation disabled
    pub fn no_grad<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let was_enabled = self.config.enabled;
        self.config.enabled = false;

        let result = f();

        // Restore original state
        self.config.enabled = was_enabled;

        result
    }

    /// Execute a function with gradients enabled
    pub fn with_grad<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let was_enabled = self.config.enabled;
        self.config.enabled = true;

        let result = f();

        // Restore original state
        self.config.enabled = was_enabled;

        result
    }

    /// Check for gradient anomalies
    pub fn check_anomalies(&self, variable: &Variable) -> Result<()> {
        if !self.config.detect_anomalies {
            return Ok(());
        }

        if let Some(grad) = self.get_grad(variable)? {
            let grad_values = grad.to_vec_f32()?;

            for &value in &grad_values {
                if value.is_nan() {
                    return Err(tensor_op_error(
                        "AutodiffEngine::check_anomalies",
                        "NaN detected in gradient",
                    ));
                }
                if value.is_infinite() {
                    return Err(tensor_op_error(
                        "AutodiffEngine::check_anomalies",
                        "Infinite value detected in gradient",
                    ));
                }
            }
        }

        Ok(())
    }

    /// Enable gradient checkpointing
    pub fn enable_checkpointing(&mut self) {
        self.config.gradient_checkpointing = true;
    }

    /// Disable gradient checkpointing
    pub fn disable_checkpointing(&mut self) {
        self.config.gradient_checkpointing = false;
    }

    /// Check if gradient checkpointing is enabled
    pub fn is_checkpointing_enabled(&self) -> bool {
        self.config.gradient_checkpointing
    }

    /// Export computation graph for visualization
    pub fn export_graph(&self) -> Result<String> {
        let graph = self.graph.lock().unwrap();
        let graph_export = graph.export_graph();

        // Convert to DOT format for visualization
        let mut dot = String::from("digraph G {\n");
        dot.push_str("  rankdir=TB;\n");

        for node in &graph_export.nodes {
            let node_label = if let Some(ref name) = node.name {
                name.clone()
            } else {
                format!("node_{}", node.id)
            };

            let op_label = if let Some(ref op) = node.operation {
                format!("{:?}", op)
            } else {
                "Variable".to_string()
            };

            dot.push_str(&format!(
                "  {} [label=\"{}\\n{}\\n{:?}\"];\n",
                node.id, node_label, op_label, node.shape
            ));

            for parent_id in &node.parents {
                dot.push_str(&format!("  {} -> {};\n", parent_id, node.id));
            }
        }

        dot.push_str("}\n");
        Ok(dot)
    }

    /// Get memory usage information
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        let graph = self.graph.lock().unwrap();
        let mut total_memory = 0;
        let mut num_tensors = 0;

        for node in graph.export_graph().nodes {
            total_memory += node.value.memory_usage();
            num_tensors += 1;

            if let Some(ref grad) = node.gradient {
                total_memory += grad.memory_usage();
                num_tensors += 1;
            }
        }

        Ok(MemoryInfo {
            total_memory_bytes: total_memory,
            num_tensors,
            num_nodes: graph.num_nodes(),
        })
    }
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total memory usage in bytes
    pub total_memory_bytes: usize,
    /// Number of tensors
    pub num_tensors: usize,
    /// Number of graph nodes
    pub num_nodes: usize,
}

/// Global autodiff engine instance
static GLOBAL_ENGINE: OnceLock<Arc<Mutex<AutodiffEngine>>> = OnceLock::new();

/// Initialize the global autodiff engine
pub fn init_engine(config: AutodiffConfig) {
    let _ = GLOBAL_ENGINE.set(Arc::new(Mutex::new(AutodiffEngine::new(config))));
}

/// Get the global autodiff engine
pub fn get_engine() -> Arc<Mutex<AutodiffEngine>> {
    GLOBAL_ENGINE
        .get_or_init(|| Arc::new(Mutex::new(AutodiffEngine::new(AutodiffConfig::default()))))
        .clone()
}

/// Context manager for gradient computation
pub struct GradContext {
    previous_state: bool,
}

impl GradContext {
    /// Create a new context with gradients enabled
    pub fn enable() -> Self {
        let engine = get_engine();
        let previous_state = engine.lock().unwrap().is_grad_enabled();
        engine.lock().unwrap().enable_grad();

        Self { previous_state }
    }

    /// Create a new context with gradients disabled
    pub fn disable() -> Self {
        let engine = get_engine();
        let previous_state = engine.lock().unwrap().is_grad_enabled();
        engine.lock().unwrap().disable_grad();

        Self { previous_state }
    }
}

impl Drop for GradContext {
    fn drop(&mut self) {
        let engine = get_engine();
        if self.previous_state {
            engine.lock().unwrap().enable_grad();
        } else {
            engine.lock().unwrap().disable_grad();
        }
    }
}

/// Convenience macros for gradient contexts
#[macro_export]
macro_rules! no_grad {
    ($($stmt:stmt)*) => {
        {
            let _ctx = $crate::autodiff::engine::GradContext::disable();
            $($stmt)*
        }
    };
}

#[macro_export]
macro_rules! with_grad {
    ($($stmt:stmt)*) => {
        {
            let _ctx = $crate::autodiff::engine::GradContext::enable();
            $($stmt)*
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_engine_creation() {
        let config = AutodiffConfig::default();
        let engine = AutodiffEngine::new(config);

        assert!(engine.is_grad_enabled());
        assert_eq!(engine.mode(), GradientMode::Reverse);
    }

    #[test]
    fn test_variable_creation() {
        let engine = AutodiffEngine::default();
        let tensor = Tensor::ones(&[2, 3]).unwrap();
        let var = engine.variable(tensor, true);

        assert!(var.requires_grad());
        assert_eq!(var.shape().unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_gradient_computation() {
        let engine = AutodiffEngine::default();

        let a = engine.variable(Tensor::scalar(2.0).unwrap(), true);
        let b = engine.variable(Tensor::scalar(3.0).unwrap(), true);
        let c = a.mul(&b).unwrap();

        engine.backward(&c, None).unwrap();

        let grad_a = engine.get_grad(&a).unwrap().unwrap();
        let grad_b = engine.get_grad(&b).unwrap().unwrap();

        assert_eq!(grad_a.to_scalar().unwrap(), 3.0);
        assert_eq!(grad_b.to_scalar().unwrap(), 2.0);
    }

    #[test]
    fn test_grad_context() {
        let engine = AutodiffEngine::default();
        assert!(engine.is_grad_enabled());

        {
            let _ctx = GradContext::disable();
            assert!(!get_engine().lock().unwrap().is_grad_enabled());
        }

        // Should be restored after context ends
        assert!(get_engine().lock().unwrap().is_grad_enabled());
    }

    #[test]
    fn test_engine_stats() {
        let engine = AutodiffEngine::default();
        let stats = engine.stats();

        assert_eq!(stats.forward_passes, 0);
        assert_eq!(stats.backward_passes, 0);
    }

    #[test]
    fn test_memory_info() {
        let engine = AutodiffEngine::default();
        let tensor = Tensor::ones(&[100, 100]).unwrap();
        let _var = engine.variable(tensor, true);

        let memory_info = engine.memory_info().unwrap();
        assert!(memory_info.total_memory_bytes > 0);
        assert!(memory_info.num_tensors > 0);
        assert!(memory_info.num_nodes > 0);
    }

    #[test]
    fn test_anomaly_detection() {
        let config = AutodiffConfig {
            detect_anomalies: true,
            ..Default::default()
        };
        let engine = AutodiffEngine::new(config);

        let var = engine.variable(Tensor::scalar(1.0).unwrap(), true);
        let result = engine.check_anomalies(&var);

        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_export() {
        let engine = AutodiffEngine::default();
        let a = engine.variable(Tensor::scalar(2.0).unwrap(), true);
        let b = engine.variable(Tensor::scalar(3.0).unwrap(), true);
        let _c = a.mul(&b).unwrap();

        let dot_graph = engine.export_graph().unwrap();
        assert!(dot_graph.contains("digraph G"));
        assert!(dot_graph.contains("->"));
    }
}
