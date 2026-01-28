//! Gradient tape for recording operations in automatic differentiation.
//!
//! This module provides a tape-based system for recording operations
//! during the forward pass, which can then be used to compute gradients
//! during the backward pass.

use super::graph::{NodeId, OperationType};
use crate::errors::{Result, TrustformersError};
use std::collections::VecDeque;
use std::sync::Arc;

/// Entry in the gradient tape
#[derive(Clone)]
pub struct TapeEntry {
    /// Unique identifier for this entry
    pub id: usize,
    /// Operation that was performed
    pub operation: OperationType,
    /// Input node IDs
    pub inputs: Vec<NodeId>,
    /// Output node ID
    pub output: NodeId,
    /// Gradient function for this operation
    pub grad_fn: Option<Arc<dyn super::graph::GradientFunction>>,
    /// Metadata for the operation
    pub metadata: TapeEntryMetadata,
}

impl std::fmt::Debug for TapeEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TapeEntry")
            .field("id", &self.id)
            .field("operation", &self.operation)
            .field("inputs", &self.inputs)
            .field("output", &self.output)
            .field(
                "grad_fn",
                &self.grad_fn.as_ref().map(|_| "GradientFunction"),
            )
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Metadata for a tape entry
#[derive(Debug, Clone)]
pub struct TapeEntryMetadata {
    /// Operation name
    pub name: String,
    /// Input shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Timestamp when operation was recorded
    pub timestamp: std::time::Instant,
    /// Estimated FLOPS for this operation
    pub estimated_flops: usize,
    /// Memory usage for this operation
    pub memory_usage: usize,
}

/// Gradient tape for recording operations
pub struct GradientTape {
    /// Entries in the tape
    entries: VecDeque<TapeEntry>,
    /// Next available entry ID
    next_id: usize,
    /// Maximum number of entries to keep
    max_size: usize,
    /// Whether the tape is enabled
    enabled: bool,
    /// Whether to automatically clear the tape after backward pass
    auto_clear: bool,
}

impl std::fmt::Debug for GradientTape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientTape")
            .field("entries", &self.entries)
            .field("next_id", &self.next_id)
            .field("max_size", &self.max_size)
            .field("enabled", &self.enabled)
            .field("auto_clear", &self.auto_clear)
            .finish()
    }
}

impl GradientTape {
    /// Create a new gradient tape
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            next_id: 0,
            max_size: 10000,
            enabled: true,
            auto_clear: true,
        }
    }

    /// Create a new gradient tape with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            next_id: 0,
            max_size: capacity,
            enabled: true,
            auto_clear: true,
        }
    }

    /// Record an operation on the tape
    pub fn record_operation(
        &mut self,
        operation: OperationType,
        inputs: Vec<NodeId>,
        output: NodeId,
        grad_fn: Option<Arc<dyn super::graph::GradientFunction>>,
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
    ) -> Result<usize> {
        if !self.enabled {
            return Ok(0);
        }

        let entry_id = self.next_id;
        self.next_id += 1;

        let metadata = TapeEntryMetadata {
            name: format!("{:?}", operation),
            input_shapes: input_shapes.clone(),
            output_shape: output_shape.clone(),
            timestamp: std::time::Instant::now(),
            estimated_flops: self.estimate_flops(&operation, &input_shapes, &output_shape),
            memory_usage: self.estimate_memory_usage(&input_shapes, &output_shape),
        };

        let entry = TapeEntry {
            id: entry_id,
            operation,
            inputs,
            output,
            grad_fn,
            metadata,
        };

        // Add entry to tape
        self.entries.push_back(entry);

        // Enforce maximum size
        if self.entries.len() > self.max_size {
            self.entries.pop_front();
        }

        Ok(entry_id)
    }

    /// Get an entry by ID
    pub fn get_entry(&self, id: usize) -> Option<&TapeEntry> {
        self.entries.iter().find(|entry| entry.id == id)
    }

    /// Get all entries
    pub fn entries(&self) -> impl Iterator<Item = &TapeEntry> {
        self.entries.iter()
    }

    /// Get entries in reverse order (for backward pass)
    pub fn entries_reverse(&self) -> impl Iterator<Item = &TapeEntry> {
        self.entries.iter().rev()
    }

    /// Clear the tape
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Enable the tape
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable the tape
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if the tape is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set auto-clear behavior
    pub fn set_auto_clear(&mut self, auto_clear: bool) {
        self.auto_clear = auto_clear;
    }

    /// Get auto-clear behavior
    pub fn auto_clear(&self) -> bool {
        self.auto_clear
    }

    /// Set maximum size
    pub fn set_max_size(&mut self, max_size: usize) {
        self.max_size = max_size;

        // Trim if necessary
        while self.entries.len() > max_size {
            self.entries.pop_front();
        }
    }

    /// Get maximum size
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the tape is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get tape statistics
    pub fn stats(&self) -> TapeStats {
        let mut total_flops = 0;
        let mut total_memory = 0;
        let mut operation_counts = std::collections::HashMap::new();

        for entry in &self.entries {
            total_flops += entry.metadata.estimated_flops;
            total_memory += entry.metadata.memory_usage;

            let op_name = &entry.metadata.name;
            *operation_counts.entry(op_name.clone()).or_insert(0) += 1;
        }

        TapeStats {
            num_entries: self.entries.len(),
            total_flops,
            total_memory,
            operation_counts,
        }
    }

    /// Estimate FLOPS for an operation
    fn estimate_flops(
        &self,
        operation: &OperationType,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> usize {
        match operation {
            OperationType::Add
            | OperationType::Subtract
            | OperationType::Multiply
            | OperationType::Divide => {
                // Element-wise operations
                output_shape.iter().product::<usize>()
            },
            OperationType::MatrixMultiply => {
                // Matrix multiplication: A[m,k] * B[k,n] = C[m,n] requires m*k*n operations
                if input_shapes.len() >= 2
                    && input_shapes[0].len() >= 2
                    && input_shapes[1].len() >= 2
                {
                    let m = input_shapes[0][0];
                    let k = input_shapes[0][1];
                    let n = input_shapes[1][1];
                    m * k * n
                } else {
                    0
                }
            },
            OperationType::Sigmoid | OperationType::Tanh | OperationType::ReLU => {
                // Activation functions
                output_shape.iter().product::<usize>()
            },
            OperationType::Softmax => {
                // Softmax: exp + sum + divide
                let num_elements = output_shape.iter().product::<usize>();
                num_elements * 3 // Approximate
            },
            OperationType::Sum(_) | OperationType::Mean(_) => {
                // Reduction operations
                if !input_shapes.is_empty() {
                    input_shapes[0].iter().product::<usize>()
                } else {
                    0
                }
            },
            OperationType::LayerNorm(_) => {
                // Layer normalization: mean + var + normalize
                let num_elements = output_shape.iter().product::<usize>();
                num_elements * 5 // Approximate
            },
            _ => {
                // Default estimate
                output_shape.iter().product::<usize>()
            },
        }
    }

    /// Estimate memory usage for an operation
    fn estimate_memory_usage(&self, input_shapes: &[Vec<usize>], output_shape: &[usize]) -> usize {
        let mut total_memory = 0;

        // Input tensors
        for shape in input_shapes {
            total_memory += shape.iter().product::<usize>() * 4; // 4 bytes per f32
        }

        // Output tensor
        total_memory += output_shape.iter().product::<usize>() * 4; // 4 bytes per f32

        total_memory
    }

    /// Find entries that depend on a given node
    pub fn find_dependent_entries(&self, node_id: NodeId) -> Vec<&TapeEntry> {
        self.entries.iter().filter(|entry| entry.inputs.contains(&node_id)).collect()
    }

    /// Find entries that produce a given node
    pub fn find_producer_entry(&self, node_id: NodeId) -> Option<&TapeEntry> {
        self.entries.iter().find(|entry| entry.output == node_id)
    }

    /// Helper method to process input nodes during path traversal
    fn process_entry_inputs(
        &self,
        entry: &TapeEntry,
        input_nodes: &[NodeId],
        visited: &mut std::collections::HashSet<NodeId>,
        queue: &mut std::collections::VecDeque<NodeId>,
    ) {
        for &input_node in &entry.inputs {
            if visited.contains(&input_node) || input_nodes.contains(&input_node) {
                continue;
            }
            queue.push_back(input_node);
            visited.insert(input_node);
        }
    }

    /// Get the computational path from inputs to output
    pub fn get_computational_path(
        &self,
        input_nodes: &[NodeId],
        output_node: NodeId,
    ) -> Vec<&TapeEntry> {
        let mut path = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        // Start from output and work backwards
        queue.push_back(output_node);
        visited.insert(output_node);

        while let Some(current_node) = queue.pop_front() {
            if let Some(entry) = self.find_producer_entry(current_node) {
                path.push(entry);
                self.process_entry_inputs(entry, input_nodes, &mut visited, &mut queue);
            }
        }

        // Reverse to get forward path
        path.reverse();
        path
    }

    /// Export tape to a readable format
    pub fn export_trace(&self) -> String {
        let mut trace = String::new();
        trace.push_str("Gradient Tape Trace:\n");
        trace.push_str("===================\n\n");

        for entry in &self.entries {
            trace.push_str(&format!("Entry {}: {}\n", entry.id, entry.metadata.name));
            trace.push_str(&format!("  Operation: {:?}\n", entry.operation));
            trace.push_str(&format!("  Inputs: {:?}\n", entry.inputs));
            trace.push_str(&format!("  Output: {}\n", entry.output));
            trace.push_str(&format!(
                "  Input shapes: {:?}\n",
                entry.metadata.input_shapes
            ));
            trace.push_str(&format!(
                "  Output shape: {:?}\n",
                entry.metadata.output_shape
            ));
            trace.push_str(&format!(
                "  Estimated FLOPS: {}\n",
                entry.metadata.estimated_flops
            ));
            trace.push_str(&format!(
                "  Memory usage: {} bytes\n",
                entry.metadata.memory_usage
            ));
            trace.push('\n');
        }

        let stats = self.stats();
        trace.push_str(&format!("Total entries: {}\n", stats.num_entries));
        trace.push_str(&format!("Total FLOPS: {}\n", stats.total_flops));
        trace.push_str(&format!("Total memory: {} bytes\n", stats.total_memory));
        trace.push_str("\nOperation counts:\n");
        for (op, count) in stats.operation_counts {
            trace.push_str(&format!("  {}: {}\n", op, count));
        }

        trace
    }

    /// Validate the tape for consistency
    pub fn validate(&self) -> Result<()> {
        let mut node_ids = std::collections::HashSet::new();

        for entry in &self.entries {
            // Check that all input nodes exist
            for &input_id in &entry.inputs {
                if !node_ids.contains(&input_id) {
                    return Err(TrustformersError::tensor_op_error(
                        &format!("Input node {} not found for entry {}", input_id, entry.id),
                        "GradientTape::validate",
                    ));
                }
            }

            // Add output node
            node_ids.insert(entry.output);
        }

        Ok(())
    }
}

/// Statistics for the gradient tape
#[derive(Debug, Clone)]
pub struct TapeStats {
    /// Number of entries
    pub num_entries: usize,
    /// Total estimated FLOPS
    pub total_flops: usize,
    /// Total memory usage
    pub total_memory: usize,
    /// Count of each operation type
    pub operation_counts: std::collections::HashMap<String, usize>,
}

/// Context for managing tape recording
pub struct TapeContext {
    tape: Arc<std::sync::Mutex<GradientTape>>,
    was_enabled: bool,
}

impl TapeContext {
    /// Create a new tape context
    pub fn new(tape: Arc<std::sync::Mutex<GradientTape>>) -> Self {
        let was_enabled = {
            let tape_guard = tape.lock().expect("Lock poisoned");
            tape_guard.is_enabled()
        };

        Self { tape, was_enabled }
    }

    /// Enable recording
    pub fn enable(&self) {
        let mut tape = self.tape.lock().expect("Lock poisoned");
        tape.enable();
    }

    /// Disable recording
    pub fn disable(&self) {
        let mut tape = self.tape.lock().expect("Lock poisoned");
        tape.disable();
    }

    /// Record an operation
    pub fn record(
        &self,
        operation: OperationType,
        inputs: Vec<NodeId>,
        output: NodeId,
        grad_fn: Option<Arc<dyn super::graph::GradientFunction>>,
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
    ) -> Result<usize> {
        let mut tape = self.tape.lock().expect("Lock poisoned");
        tape.record_operation(
            operation,
            inputs,
            output,
            grad_fn,
            input_shapes,
            output_shape,
        )
    }
}

impl Drop for TapeContext {
    fn drop(&mut self) {
        let mut tape = self.tape.lock().expect("Lock poisoned");
        if self.was_enabled {
            tape.enable();
        } else {
            tape.disable();
        }
    }
}

impl Default for GradientTape {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_creation() {
        let tape = GradientTape::new();
        assert!(tape.is_enabled());
        assert!(tape.is_empty());
        assert_eq!(tape.len(), 0);
    }

    #[test]
    fn test_tape_recording() {
        let mut tape = GradientTape::new();

        let entry_id = tape
            .record_operation(
                OperationType::Add,
                vec![0, 1],
                2,
                None,
                vec![vec![2, 3], vec![2, 3]],
                vec![2, 3],
            )
            .unwrap();

        assert_eq!(tape.len(), 1);
        assert_eq!(entry_id, 0);

        let entry = tape.get_entry(entry_id).unwrap();
        assert_eq!(entry.inputs, vec![0, 1]);
        assert_eq!(entry.output, 2);
    }

    #[test]
    fn test_tape_stats() {
        let mut tape = GradientTape::new();

        tape.record_operation(
            OperationType::Add,
            vec![0, 1],
            2,
            None,
            vec![vec![2, 3], vec![2, 3]],
            vec![2, 3],
        )
        .unwrap();

        tape.record_operation(
            OperationType::Multiply,
            vec![2, 3],
            4,
            None,
            vec![vec![2, 3], vec![2, 3]],
            vec![2, 3],
        )
        .unwrap();

        let stats = tape.stats();
        assert_eq!(stats.num_entries, 2);
        assert!(stats.total_flops > 0);
        assert!(stats.total_memory > 0);
        assert_eq!(stats.operation_counts.len(), 2);
    }

    #[test]
    fn test_tape_clear() {
        let mut tape = GradientTape::new();

        tape.record_operation(
            OperationType::Add,
            vec![0, 1],
            2,
            None,
            vec![vec![2, 3], vec![2, 3]],
            vec![2, 3],
        )
        .unwrap();

        assert_eq!(tape.len(), 1);

        tape.clear();
        assert_eq!(tape.len(), 0);
        assert!(tape.is_empty());
    }

    #[test]
    fn test_tape_enable_disable() {
        let mut tape = GradientTape::new();
        assert!(tape.is_enabled());

        tape.disable();
        assert!(!tape.is_enabled());

        tape.enable();
        assert!(tape.is_enabled());
    }

    #[test]
    fn test_tape_max_size() {
        let mut tape = GradientTape::with_capacity(2);
        tape.set_max_size(2);

        // Add 3 entries
        for i in 0..3 {
            tape.record_operation(
                OperationType::Add,
                vec![i, i + 1],
                i + 2,
                None,
                vec![vec![2, 3], vec![2, 3]],
                vec![2, 3],
            )
            .unwrap();
        }

        // Should only keep the last 2 entries
        assert_eq!(tape.len(), 2);
    }

    #[test]
    fn test_find_dependent_entries() {
        let mut tape = GradientTape::new();

        tape.record_operation(
            OperationType::Add,
            vec![0, 1],
            2,
            None,
            vec![vec![2, 3], vec![2, 3]],
            vec![2, 3],
        )
        .unwrap();

        tape.record_operation(
            OperationType::Multiply,
            vec![0, 2],
            3,
            None,
            vec![vec![2, 3], vec![2, 3]],
            vec![2, 3],
        )
        .unwrap();

        let dependent = tape.find_dependent_entries(0);
        assert_eq!(dependent.len(), 2);

        let dependent = tape.find_dependent_entries(1);
        assert_eq!(dependent.len(), 1);
    }

    #[test]
    fn test_find_producer_entry() {
        let mut tape = GradientTape::new();

        tape.record_operation(
            OperationType::Add,
            vec![0, 1],
            2,
            None,
            vec![vec![2, 3], vec![2, 3]],
            vec![2, 3],
        )
        .unwrap();

        let producer = tape.find_producer_entry(2);
        assert!(producer.is_some());
        assert_eq!(producer.unwrap().operation, OperationType::Add);

        let producer = tape.find_producer_entry(0);
        assert!(producer.is_none());
    }

    #[test]
    fn test_tape_export() {
        let mut tape = GradientTape::new();

        tape.record_operation(
            OperationType::Add,
            vec![0, 1],
            2,
            None,
            vec![vec![2, 3], vec![2, 3]],
            vec![2, 3],
        )
        .unwrap();

        let trace = tape.export_trace();
        assert!(trace.contains("Gradient Tape Trace"));
        assert!(trace.contains("Add"));
        assert!(trace.contains("Total entries: 1"));
    }

    #[test]
    fn test_tape_validation() {
        let mut tape = GradientTape::new();

        // This should fail validation because input node 0 doesn't exist
        tape.record_operation(
            OperationType::Add,
            vec![0, 1],
            2,
            None,
            vec![vec![2, 3], vec![2, 3]],
            vec![2, 3],
        )
        .unwrap();

        let result = tape.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_tape_context() {
        let tape = Arc::new(std::sync::Mutex::new(GradientTape::new()));
        let context = TapeContext::new(tape.clone());

        assert!(context.tape.lock().expect("Lock poisoned").is_enabled());

        context.disable();
        assert!(!context.tape.lock().expect("Lock poisoned").is_enabled());

        context.enable();
        assert!(context.tape.lock().expect("Lock poisoned").is_enabled());
    }
}
