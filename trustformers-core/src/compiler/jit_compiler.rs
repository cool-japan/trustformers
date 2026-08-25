//! JIT Compiler Module
//!
//! This module provides just-in-time compilation capabilities for dynamic computation graphs including:
//!
//! - **Dynamic Compilation**: Compile computation graphs at runtime
//! - **Code Generation**: Generate optimized machine code for target hardware
//! - **Cache Management**: Intelligent caching of compiled kernels
//! - **Runtime Optimization**: Adaptive optimization based on runtime characteristics
//! - **Multi-Backend Support**: Support for LLVM, cranelift, and custom backends

#![allow(clippy::excessive_nesting)] // Complex compiler optimization algorithms require deep nesting
#![allow(unused_variables)] // JIT compiler

use crate::compiler::{
    CompilationResult, CompilationStats, CompilerConfig, ComputationGraph, GraphNode,
};
use crate::errors::TrustformersError;
use crate::errors::{invalid_format, runtime_error, unsupported_operation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Assumed compute-cost retention when fusing two instructions, used by
/// `JitCompiler::create_fused_instruction`.
///
/// This is an unvalidated modelling assumption, not a measurement: nothing
/// in this crate profiles fused kernels against their unfused originals, so
/// there is no real per-opcode-pair savings data to differentiate this by.
/// It intentionally stays a single, clearly-labelled constant rather than a
/// per-pattern table dressed up with invented numbers — see
/// `kernel_fusion::FusionPattern::expected_speedup` for what a *real*
/// differentiated table looks like once such data exists.
const ASSUMED_FUSION_COMPUTE_RETENTION: f64 = 0.7; // i.e. an assumed 30% compute saving

/// Assumed memory-cost retention when fusing two instructions, used by
/// `JitCompiler::create_fused_instruction`. Same caveat as
/// `ASSUMED_FUSION_COMPUTE_RETENTION`: an unvalidated modelling assumption,
/// not a measurement.
const ASSUMED_FUSION_MEMORY_RETENTION: f64 = 0.8; // i.e. an assumed 20% memory saving

/// JIT compiler for dynamic compilation of computation graphs
pub struct JitCompiler {
    config: CompilerConfig,
    backend: Box<dyn JitBackend>,
    compilation_cache: Arc<Mutex<HashMap<String, CachedCompilation>>>,
    compilation_stats: CompilationStatistics,
}

impl JitCompiler {
    /// Create a new JIT compiler
    pub fn new(config: &CompilerConfig) -> Result<Self, TrustformersError> {
        let backend = Self::create_backend(config)?;

        Ok(Self {
            config: config.clone(),
            backend,
            compilation_cache: Arc::new(Mutex::new(HashMap::new())),
            compilation_stats: CompilationStatistics::new(),
        })
    }

    /// Update the configuration
    pub fn update_config(&mut self, config: &CompilerConfig) -> Result<(), TrustformersError> {
        self.config = config.clone();
        self.backend = Self::create_backend(config)?;
        Ok(())
    }

    /// Create appropriate backend based on configuration
    /// Create the backend the configuration asks for.
    ///
    /// Only the interpreter backend exists. Asking for a native JIT is an
    /// error rather than a silent downgrade: a caller who requested `llvm` and
    /// received an interpreter would attribute the interpreter's timings to a
    /// JIT.
    fn create_backend(config: &CompilerConfig) -> Result<Box<dyn JitBackend>, TrustformersError> {
        for requested in ["llvm", "cranelift"] {
            if config.compiler_flags.iter().any(|flag| flag == requested) {
                return Err(TrustformersError::not_implemented(format!(
                    "compiler flag '{}' requests a native JIT backend, which is not implemented; \
                     only the interpreter backend is available",
                    requested
                )));
            }
        }

        Ok(Box::new(InterpreterBackend::new(config)?))
    }

    /// Compile a computation graph
    pub fn compile(
        &mut self,
        graph: ComputationGraph,
    ) -> Result<CompilationResult, TrustformersError> {
        let start_time = Instant::now();

        // Generate cache key for the graph
        let cache_key = self.generate_cache_key(&graph)?;

        // Check cache first
        if self.config.enable_cache {
            if let Some(cached) = self.get_cached_compilation(&cache_key)? {
                self.compilation_stats.cache_hits += 1;
                return Ok(CompilationResult {
                    compiled_code: cached.compiled_code.clone(),
                    stats: cached.stats.clone(),
                    metadata: cached.metadata.clone(),
                });
            }
        }

        self.compilation_stats.cache_misses += 1;

        // Validate graph before compilation
        graph.validate()?;

        // Compile the graph
        let ir = self.generate_ir(&graph)?;
        let original_ir_size = ir.instructions.len();
        let original_compute_cost = self.calculate_total_compute_cost(&ir);
        let original_memory_cost = self.calculate_total_memory_cost(&ir);

        let (optimized_ir, optimization_metrics) = self.optimize_ir_with_metrics(ir)?;
        let compiled_code = self.backend.compile_ir(optimized_ir)?;

        let compilation_time = start_time.elapsed();

        let optimized_compute_cost =
            self.calculate_total_compute_cost(&optimization_metrics.optimized_ir);
        let optimized_memory_cost =
            self.calculate_total_memory_cost(&optimization_metrics.optimized_ir);

        // Calculate performance improvements
        let performance_gain = if optimized_compute_cost > 0.0 {
            original_compute_cost / optimized_compute_cost
        } else {
            1.0
        };

        let memory_reduction = if original_memory_cost > 0.0 {
            (original_memory_cost - optimized_memory_cost) / original_memory_cost
        } else {
            0.0
        };

        // Generate detailed compilation statistics
        let stats = CompilationStats {
            compilation_time_ms: compilation_time.as_millis() as u64,
            original_ops: graph.nodes.len(),
            optimized_ops: optimization_metrics.optimized_ir.instructions.len(),
            fused_kernels: optimization_metrics.fused_kernels,
            performance_gain,
            memory_reduction,
            applied_passes: optimization_metrics.applied_passes,
        };

        let metadata = HashMap::new();

        let result = CompilationResult {
            compiled_code: compiled_code.clone(),
            stats: stats.clone(),
            metadata: metadata.clone(),
        };

        // Cache the result
        if self.config.enable_cache {
            self.cache_compilation(cache_key, compiled_code, stats, metadata)?;
        }

        self.compilation_stats.compilations += 1;
        self.compilation_stats.total_compilation_time += compilation_time;

        Ok(result)
    }

    /// Generate intermediate representation from computation graph
    fn generate_ir(
        &self,
        graph: &ComputationGraph,
    ) -> Result<IntermediateRepresentation, TrustformersError> {
        let mut ir = IntermediateRepresentation::new();

        // Convert graph nodes to IR instructions
        for node in &graph.nodes {
            let instruction = self.node_to_instruction(node)?;
            ir.add_instruction(instruction);
        }

        // Add control flow information from edges
        for edge in &graph.edges {
            ir.add_dependency(edge.from, edge.to);
        }

        Ok(ir)
    }

    /// Convert a graph node to an IR instruction
    fn node_to_instruction(&self, node: &GraphNode) -> Result<IRInstruction, TrustformersError> {
        let opcode = match node.op_type.as_str() {
            "MatMul" => IROpcode::MatMul,
            "Add" => IROpcode::Add,
            "Mul" => IROpcode::Mul,
            "ReLU" => IROpcode::ReLU,
            "Sigmoid" => IROpcode::Sigmoid,
            "Tanh" => IROpcode::Tanh,
            "Softmax" => IROpcode::Softmax,
            "LayerNorm" => IROpcode::LayerNorm,
            "Attention" => IROpcode::Attention,
            "Embedding" => IROpcode::Embedding,
            "Linear" => IROpcode::Linear,
            "Conv2D" => IROpcode::Conv2D,
            "Pool2D" => IROpcode::Pool2D,
            "Reshape" => IROpcode::Reshape,
            "Transpose" => IROpcode::Transpose,
            _ => return Err(unsupported_operation("node_compilation", &node.op_type)),
        };

        Ok(IRInstruction {
            id: node.id,
            opcode,
            inputs: node.input_shapes.clone(),
            outputs: node.output_shapes.clone(),
            attributes: node.attributes.clone(),
            compute_cost: node.compute_cost,
            memory_cost: node.memory_cost,
        })
    }

    /// Optimize intermediate representation
    #[allow(dead_code)]
    fn optimize_ir(
        &self,
        mut ir: IntermediateRepresentation,
    ) -> Result<IntermediateRepresentation, TrustformersError> {
        // Apply IR-level optimizations
        ir = self.apply_constant_propagation(ir)?;
        ir = self.apply_dead_instruction_elimination(ir)?;
        ir = self.apply_instruction_scheduling(ir)?;

        Ok(ir)
    }

    /// Optimize intermediate representation with detailed metrics tracking
    fn optimize_ir_with_metrics(
        &self,
        mut ir: IntermediateRepresentation,
    ) -> Result<(IntermediateRepresentation, OptimizationMetrics), TrustformersError> {
        let mut applied_passes = Vec::new();
        let mut fused_kernels = 0;

        // Apply constant propagation
        let (ir_after_cp, cp_fused) = self.apply_constant_propagation_with_metrics(ir)?;
        ir = ir_after_cp;
        fused_kernels += cp_fused;
        applied_passes.push("constant_propagation".to_string());

        // Apply dead instruction elimination
        let (ir_after_die, die_removed) =
            self.apply_dead_instruction_elimination_with_metrics(ir)?;
        ir = ir_after_die;
        applied_passes.push(format!(
            "dead_instruction_elimination(removed: {})",
            die_removed
        ));

        // Apply instruction scheduling
        let (ir_after_sched, sched_reordered) =
            self.apply_instruction_scheduling_with_metrics(ir)?;
        ir = ir_after_sched;
        applied_passes.push(format!(
            "instruction_scheduling(reordered: {})",
            sched_reordered
        ));

        // Apply kernel fusion pass
        let (ir_after_fusion, fusion_count) = self.apply_kernel_fusion_with_metrics(ir)?;
        ir = ir_after_fusion;
        fused_kernels += fusion_count;
        applied_passes.push(format!("kernel_fusion(fused: {})", fusion_count));

        let metrics = OptimizationMetrics {
            optimized_ir: ir.clone(),
            fused_kernels,
            applied_passes,
        };

        Ok((ir, metrics))
    }

    /// Apply constant propagation optimization
    fn apply_constant_propagation(
        &self,
        mut ir: IntermediateRepresentation,
    ) -> Result<IntermediateRepresentation, TrustformersError> {
        // Simple constant propagation implementation
        let mut changed = true;
        while changed {
            changed = false;
            // Look for instructions that can be evaluated at compile time
            for instruction in &mut ir.instructions {
                if self.can_evaluate_at_compile_time(instruction) {
                    // Mark instruction as constant
                    instruction.attributes.insert("constant".to_string(), "true".to_string());
                    changed = true;
                }
            }
        }
        Ok(ir)
    }

    /// Apply dead instruction elimination
    fn apply_dead_instruction_elimination(
        &self,
        mut ir: IntermediateRepresentation,
    ) -> Result<IntermediateRepresentation, TrustformersError> {
        // Mark instructions that are used
        let mut used = vec![false; ir.instructions.len()];

        // Mark output instructions as used
        for (i, instruction) in ir.instructions.iter().enumerate() {
            if instruction.attributes.contains_key("output") {
                used[i] = true;
            }
        }

        // Propagate usage backwards through dependencies
        let mut changed = true;
        while changed {
            changed = false;
            for &(from, to) in &ir.dependencies {
                if used[to] && !used[from] {
                    used[from] = true;
                    changed = true;
                }
            }
        }

        // Remove unused instructions
        ir.instructions.retain(|instruction| used[instruction.id]);

        Ok(ir)
    }

    /// Apply instruction scheduling optimization
    fn apply_instruction_scheduling(
        &self,
        ir: IntermediateRepresentation,
    ) -> Result<IntermediateRepresentation, TrustformersError> {
        // For now, return as-is. Real implementation would reorder instructions
        // to minimize register pressure and maximize parallelism
        Ok(ir)
    }

    /// Check if an instruction can be evaluated at compile time
    fn can_evaluate_at_compile_time(&self, instruction: &IRInstruction) -> bool {
        // Simple heuristic: check if all inputs are constants
        matches!(instruction.opcode, IROpcode::Add | IROpcode::Mul)
            && instruction.attributes.get("all_inputs_constant").is_some_and(|v| v == "true")
    }

    /// Apply constant folding to arithmetic operations
    fn apply_constant_fold_arithmetic(
        &self,
        instruction: &mut IRInstruction,
    ) -> Option<(String, bool)> {
        if matches!(
            instruction.opcode,
            IROpcode::Add | IROpcode::Mul | IROpcode::Sub | IROpcode::Div
        ) {
            if let Some(constant_value) = self.evaluate_constant_instruction(instruction) {
                instruction
                    .attributes
                    .insert("folded_value".to_string(), constant_value.clone());
                return Some((constant_value, true));
            }
        }
        None
    }

    /// Generate cache key for a computation graph
    fn generate_cache_key(&self, graph: &ComputationGraph) -> Result<String, TrustformersError> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash graph structure
        graph.nodes.len().hash(&mut hasher);
        graph.edges.len().hash(&mut hasher);

        for node in &graph.nodes {
            node.op_type.hash(&mut hasher);
            node.input_shapes.hash(&mut hasher);
            node.output_shapes.hash(&mut hasher);
        }

        for edge in &graph.edges {
            edge.from.hash(&mut hasher);
            edge.to.hash(&mut hasher);
            edge.shape.hash(&mut hasher);
            edge.dtype.hash(&mut hasher);
        }

        // Include hardware target in cache key
        self.config.target_hardware.device_type.hash(&mut hasher);
        self.config.target_hardware.compute_units.hash(&mut hasher);

        Ok(format!("{:x}", hasher.finish()))
    }

    /// Get cached compilation if available
    fn get_cached_compilation(
        &self,
        cache_key: &str,
    ) -> Result<Option<CachedCompilation>, TrustformersError> {
        let cache = self
            .compilation_cache
            .lock()
            .map_err(|_| runtime_error("Failed to acquire cache lock"))?;

        Ok(cache.get(cache_key).cloned())
    }

    /// Cache a compilation result
    fn cache_compilation(
        &self,
        cache_key: String,
        compiled_code: Vec<u8>,
        stats: CompilationStats,
        metadata: HashMap<String, String>,
    ) -> Result<(), TrustformersError> {
        let mut cache = self
            .compilation_cache
            .lock()
            .map_err(|_| runtime_error("Failed to acquire cache lock"))?;

        let cached = CachedCompilation {
            compiled_code,
            stats,
            metadata,
            timestamp: std::time::SystemTime::now(),
        };

        cache.insert(cache_key, cached);
        Ok(())
    }

    /// Clear the compilation cache
    pub fn clear_cache(&mut self) {
        if let Ok(mut cache) = self.compilation_cache.lock() {
            cache.clear();
        }
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.compilation_cache.lock().map(|cache| cache.len()).unwrap_or(0)
    }

    /// Get compilation statistics
    pub fn get_stats(&self) -> &CompilationStatistics {
        &self.compilation_stats
    }

    /// Reset compilation statistics
    pub fn reset_stats(&mut self) {
        self.compilation_stats = CompilationStatistics::new();
    }

    /// Calculate total compute cost for an IR
    fn calculate_total_compute_cost(&self, ir: &IntermediateRepresentation) -> f64 {
        ir.instructions.iter().map(|inst| inst.compute_cost).sum()
    }

    /// Calculate total memory cost for an IR
    fn calculate_total_memory_cost(&self, ir: &IntermediateRepresentation) -> f64 {
        ir.instructions.iter().map(|inst| inst.memory_cost).sum()
    }

    /// Enhanced constant propagation with metrics
    fn apply_constant_propagation_with_metrics(
        &self,
        mut ir: IntermediateRepresentation,
    ) -> Result<(IntermediateRepresentation, usize), TrustformersError> {
        let mut fused_operations = 0;
        let mut changed = true;

        while changed {
            changed = false;
            let instructions_to_remove = Vec::new();

            for (i, instruction) in ir.instructions.iter_mut().enumerate() {
                if !self.can_evaluate_at_compile_time(instruction) {
                    continue;
                }

                // Mark instruction as constant and attempt to fold
                instruction.attributes.insert("constant".to_string(), "true".to_string());

                // Apply constant folding to arithmetic operations
                if let Some((_value, folded)) = self.apply_constant_fold_arithmetic(instruction) {
                    if folded {
                        fused_operations += 1;
                        changed = true;
                    }
                }
            }

            // Remove folded instructions
            for i in instructions_to_remove.into_iter().rev() {
                ir.instructions.remove(i);
            }
        }

        Ok((ir, fused_operations))
    }

    /// Enhanced dead instruction elimination with metrics
    fn apply_dead_instruction_elimination_with_metrics(
        &self,
        mut ir: IntermediateRepresentation,
    ) -> Result<(IntermediateRepresentation, usize), TrustformersError> {
        let original_count = ir.instructions.len();

        // Mark instructions that are used
        let mut used = vec![false; ir.instructions.len()];

        // Mark output instructions as used
        for (i, instruction) in ir.instructions.iter().enumerate() {
            if instruction.attributes.contains_key("output") {
                used[i] = true;
            }
        }

        // Propagate usage backwards through dependencies
        let mut changed = true;
        while changed {
            changed = false;
            for &(from, to) in &ir.dependencies {
                if to < used.len() && from < used.len() && used[to] && !used[from] {
                    used[from] = true;
                    changed = true;
                }
            }
        }

        // Remove unused instructions
        let mut instruction_id_map = HashMap::new();
        let mut new_instructions = Vec::new();
        let mut new_id = 0;

        for (old_id, instruction) in ir.instructions.into_iter().enumerate() {
            if used[old_id] {
                instruction_id_map.insert(old_id, new_id);
                new_instructions.push(IRInstruction {
                    id: new_id,
                    ..instruction
                });
                new_id += 1;
            }
        }

        ir.instructions = new_instructions;

        // Update dependencies with new IDs
        ir.dependencies = ir
            .dependencies
            .into_iter()
            .filter_map(|(from, to)| {
                if let (Some(&new_from), Some(&new_to)) =
                    (instruction_id_map.get(&from), instruction_id_map.get(&to))
                {
                    Some((new_from, new_to))
                } else {
                    None
                }
            })
            .collect();

        let removed_count = original_count - ir.instructions.len();
        Ok((ir, removed_count))
    }

    /// Enhanced instruction scheduling with metrics
    fn apply_instruction_scheduling_with_metrics(
        &self,
        mut ir: IntermediateRepresentation,
    ) -> Result<(IntermediateRepresentation, usize), TrustformersError> {
        let mut reordered_count = 0;

        // Simple scheduling based on dependency depth
        let mut instruction_depths = vec![0; ir.instructions.len()];

        // Calculate depth for each instruction
        for &(from, to) in &ir.dependencies {
            if from < instruction_depths.len() && to < instruction_depths.len() {
                instruction_depths[to] = instruction_depths[to].max(instruction_depths[from] + 1);
            }
        }

        // Sort instructions by depth (topological sort)
        let mut instruction_indices: Vec<usize> = (0..ir.instructions.len()).collect();
        instruction_indices.sort_by_key(|&i| instruction_depths[i]);

        // Check if reordering actually happened
        for (new_pos, &old_pos) in instruction_indices.iter().enumerate() {
            if new_pos != old_pos {
                reordered_count += 1;
            }
        }

        // Reorder instructions
        let mut new_instructions = Vec::new();
        for &old_index in &instruction_indices {
            if old_index < ir.instructions.len() {
                new_instructions.push(ir.instructions[old_index].clone());
            }
        }

        // Update instruction IDs to maintain order
        for (new_id, instruction) in new_instructions.iter_mut().enumerate() {
            instruction.id = new_id;
        }

        ir.instructions = new_instructions;

        Ok((ir, reordered_count))
    }

    /// Kernel fusion optimization with metrics
    fn apply_kernel_fusion_with_metrics(
        &self,
        mut ir: IntermediateRepresentation,
    ) -> Result<(IntermediateRepresentation, usize), TrustformersError> {
        let mut fused_count = 0;

        // Look for fusible patterns
        let mut i = 0;
        while i < ir.instructions.len().saturating_sub(1) {
            let can_fuse = self.can_fuse_instructions(&ir.instructions[i], &ir.instructions[i + 1]);

            if can_fuse {
                // Create fused instruction
                let fused_instruction =
                    self.create_fused_instruction(&ir.instructions[i], &ir.instructions[i + 1])?;

                // Replace the two instructions with the fused one
                ir.instructions[i] = fused_instruction;
                ir.instructions.remove(i + 1);

                // Update instruction IDs
                for j in i + 1..ir.instructions.len() {
                    ir.instructions[j].id = j;
                }

                fused_count += 1;
            } else {
                i += 1;
            }
        }

        Ok((ir, fused_count))
    }

    /// Check if two instructions can be fused
    fn can_fuse_instructions(&self, inst1: &IRInstruction, inst2: &IRInstruction) -> bool {
        // Simple fusion rules: element-wise operations can often be fused
        match (&inst1.opcode, &inst2.opcode) {
            (IROpcode::Add, IROpcode::ReLU) => true,
            (IROpcode::MatMul, IROpcode::Add) => true, // MatMul + bias
            (IROpcode::ReLU, IROpcode::Add) => true,
            (IROpcode::Add, IROpcode::Mul) => true,
            _ => false,
        }
    }

    /// Create a fused instruction from two fusible instructions
    fn create_fused_instruction(
        &self,
        inst1: &IRInstruction,
        inst2: &IRInstruction,
    ) -> Result<IRInstruction, TrustformersError> {
        let mut fused_attributes = inst1.attributes.clone();
        fused_attributes
            .extend(inst2.attributes.iter().map(|(k, v)| (format!("fused_{}", k), v.clone())));
        fused_attributes.insert(
            "fused_ops".to_string(),
            format!("{:?}+{:?}", inst1.opcode, inst2.opcode),
        );

        Ok(IRInstruction {
            id: inst1.id,
            opcode: self.get_fused_opcode(&inst1.opcode, &inst2.opcode),
            inputs: inst1.inputs.clone(),
            outputs: inst2.outputs.clone(),
            attributes: fused_attributes,
            compute_cost: inst1.compute_cost
                + inst2.compute_cost * ASSUMED_FUSION_COMPUTE_RETENTION,
            memory_cost: (inst1.memory_cost + inst2.memory_cost) * ASSUMED_FUSION_MEMORY_RETENTION,
        })
    }

    /// Get the appropriate opcode for fused operations
    fn get_fused_opcode(&self, op1: &IROpcode, op2: &IROpcode) -> IROpcode {
        match (op1, op2) {
            (IROpcode::Add, IROpcode::ReLU) => IROpcode::Custom("AddReLU".to_string()),
            (IROpcode::MatMul, IROpcode::Add) => IROpcode::Custom("MatMulBias".to_string()),
            (IROpcode::ReLU, IROpcode::Add) => IROpcode::Custom("ReLUAdd".to_string()),
            (IROpcode::Add, IROpcode::Mul) => IROpcode::Custom("AddMul".to_string()),
            _ => IROpcode::Custom(format!("{:?}_{:?}", op1, op2)),
        }
    }

    /// Evaluate a constant instruction at compile time.
    ///
    /// Real, not a placeholder: performs the actual `f64` addition/
    /// multiplication when both operands are present as literal `const_a`/
    /// `const_b` string attributes. Its scope is intentionally narrow —
    /// `Add` and `Mul` only, and only when both operands were already folded
    /// to literal attributes upstream — so it returns `None` (no crash, no
    /// guess) for every other opcode or missing/unparsable operand.
    fn evaluate_constant_instruction(&self, instruction: &IRInstruction) -> Option<String> {
        match instruction.opcode {
            IROpcode::Add
                if instruction.attributes.contains_key("const_a")
                    && instruction.attributes.contains_key("const_b") =>
            {
                // Parse and add constants
                if let (Ok(a), Ok(b)) = (
                    instruction.attributes.get("const_a")?.parse::<f64>(),
                    instruction.attributes.get("const_b")?.parse::<f64>(),
                ) {
                    return Some((a + b).to_string());
                }
            },
            IROpcode::Mul
                if instruction.attributes.contains_key("const_a")
                    && instruction.attributes.contains_key("const_b") =>
            {
                if let (Ok(a), Ok(b)) = (
                    instruction.attributes.get("const_a")?.parse::<f64>(),
                    instruction.attributes.get("const_b")?.parse::<f64>(),
                ) {
                    return Some((a * b).to_string());
                }
            },
            _ => {},
        }
        None
    }
}

/// Optimization metrics for tracking compilation improvements
#[derive(Debug, Clone)]
struct OptimizationMetrics {
    optimized_ir: IntermediateRepresentation,
    fused_kernels: usize,
    applied_passes: Vec<String>,
}

/// Cached compilation result
#[derive(Debug, Clone)]
struct CachedCompilation {
    compiled_code: Vec<u8>,
    stats: CompilationStats,
    metadata: HashMap<String, String>,
    #[allow(dead_code)]
    timestamp: std::time::SystemTime,
}

/// Compilation statistics
#[derive(Debug, Default, Clone)]
pub struct CompilationStatistics {
    pub compilations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_compilation_time: std::time::Duration,
}

impl CompilationStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    pub fn average_compilation_time(&self) -> std::time::Duration {
        if self.compilations == 0 {
            std::time::Duration::ZERO
        } else {
            self.total_compilation_time / self.compilations as u32
        }
    }
}

/// Intermediate representation for compilation
#[derive(Debug, Clone)]
pub struct IntermediateRepresentation {
    pub instructions: Vec<IRInstruction>,
    pub dependencies: Vec<(usize, usize)>,
    pub metadata: HashMap<String, String>,
}

impl IntermediateRepresentation {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            dependencies: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_instruction(&mut self, instruction: IRInstruction) {
        self.instructions.push(instruction);
    }

    pub fn add_dependency(&mut self, from: usize, to: usize) {
        self.dependencies.push((from, to));
    }
}

impl Default for IntermediateRepresentation {
    fn default() -> Self {
        Self::new()
    }
}

/// IR instruction representation
#[derive(Debug, Clone)]
pub struct IRInstruction {
    pub id: usize,
    pub opcode: IROpcode,
    pub inputs: Vec<Vec<usize>>,
    pub outputs: Vec<Vec<usize>>,
    pub attributes: HashMap<String, String>,
    pub compute_cost: f64,
    pub memory_cost: f64,
}

/// IR operation codes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IROpcode {
    // Arithmetic operations
    Add,
    Mul,
    Sub,
    Div,

    // Matrix operations
    MatMul,

    // Activation functions
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,

    // Neural network layers
    Linear,
    LayerNorm,
    Attention,
    Embedding,

    // Convolution operations
    Conv2D,
    Conv3D,
    Pool2D,
    Pool3D,

    // Shape operations
    Reshape,
    Transpose,
    Concat,
    Split,

    // Control flow
    If,
    While,
    Call,
    Return,

    // Memory operations
    Load,
    Store,
    Alloc,
    Free,

    // Custom fused operations
    Custom(String),
}

/// Trait for JIT compilation backends
pub trait JitBackend: Send + Sync {
    /// Compile IR to machine code
    fn compile_ir(&mut self, ir: IntermediateRepresentation) -> Result<Vec<u8>, TrustformersError>;

    /// Get backend name
    fn name(&self) -> &str;

    /// Get supported target architectures
    fn supported_targets(&self) -> Vec<String>;

    /// Optimize IR for this backend
    fn optimize_ir(
        &self,
        ir: IntermediateRepresentation,
    ) -> Result<IntermediateRepresentation, TrustformersError> {
        // Default implementation: no optimization
        Ok(ir)
    }
}

// The `llvm` and `cranelift` JIT backends were removed: their features pulled
// in no compiler dependency, and `compile_ir` returned the two bytes
// `0x90 0xC3` (NOP; RET) for every graph while `supported_targets()` advertised
// x86_64/aarch64/arm. Executing that as "compiled code" would run nothing at
// all. Wire `inkwell` / `cranelift-*` in and implement `JitBackend` against
// them to bring a real JIT back.

/// Interpreter backend (fallback)
pub struct InterpreterBackend {
    #[allow(dead_code)]
    config: CompilerConfig,
}

impl InterpreterBackend {
    pub fn new(config: &CompilerConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

impl JitBackend for InterpreterBackend {
    fn compile_ir(&mut self, ir: IntermediateRepresentation) -> Result<Vec<u8>, TrustformersError> {
        // Serialize IR for interpreter execution
        let serialized = serde_json::to_vec(&SerializableIR::from(ir))
            .map_err(|e| invalid_format("json", e.to_string()))?;
        Ok(serialized)
    }

    fn name(&self) -> &str {
        "Interpreter"
    }

    fn supported_targets(&self) -> Vec<String> {
        vec!["any".to_string()]
    }
}

/// Serializable version of IR for interpreter backend
#[derive(Debug, Serialize, Deserialize)]
struct SerializableIR {
    instructions: Vec<SerializableInstruction>,
    dependencies: Vec<(usize, usize)>,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SerializableInstruction {
    id: usize,
    opcode: String,
    inputs: Vec<Vec<usize>>,
    outputs: Vec<Vec<usize>>,
    attributes: HashMap<String, String>,
    compute_cost: f64,
    memory_cost: f64,
}

impl From<IntermediateRepresentation> for SerializableIR {
    fn from(ir: IntermediateRepresentation) -> Self {
        let instructions = ir
            .instructions
            .into_iter()
            .map(|inst| SerializableInstruction {
                id: inst.id,
                opcode: format!("{:?}", inst.opcode),
                inputs: inst.inputs,
                outputs: inst.outputs,
                attributes: inst.attributes,
                compute_cost: inst.compute_cost,
                memory_cost: inst.memory_cost,
            })
            .collect();

        Self {
            instructions,
            dependencies: ir.dependencies,
            metadata: ir.metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: the `llvm` and `cranelift` backends "compiled" every IR
    /// to `[0x90, 0xC3]` (NOP; RET) while advertising x86_64/aarch64/arm
    /// support. Requesting one must now fail rather than silently produce a
    /// two-byte stub — or silently fall back to the interpreter.
    #[test]
    fn test_native_jit_backends_are_refused() {
        for flag in ["llvm", "cranelift"] {
            let config = CompilerConfig {
                compiler_flags: vec![flag.to_string()],
                ..CompilerConfig::default()
            };
            let error = JitCompiler::new(&config)
                .err()
                .expect("a native JIT backend must not be silently substituted");
            assert!(
                error.to_string().contains(flag),
                "the error must name the requested backend: {error}"
            );
        }
    }

    /// Without a JIT flag the interpreter backend is selected, and it says so.
    #[test]
    fn test_interpreter_backend_is_the_default() {
        let compiler =
            JitCompiler::new(&CompilerConfig::default()).expect("interpreter is always available");
        assert_eq!(compiler.backend.name(), "Interpreter");
    }
    use crate::compiler::{CompilerConfig, ComputationGraph};

    #[test]
    fn test_jit_compiler_creation() {
        let config = CompilerConfig::default();
        let result = JitCompiler::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ir_instruction_creation() {
        let instruction = IRInstruction {
            id: 0,
            opcode: IROpcode::MatMul,
            inputs: vec![vec![128, 256], vec![256, 512]],
            outputs: vec![vec![128, 512]],
            attributes: HashMap::new(),
            compute_cost: 100.0,
            memory_cost: 50.0,
        };

        assert_eq!(instruction.opcode, IROpcode::MatMul);
        assert_eq!(instruction.inputs.len(), 2);
        assert_eq!(instruction.outputs.len(), 1);
    }

    /// Locks in the real behavior `evaluate_constant_instruction`'s doc
    /// comment now documents (it used to describe itself as a "demonstration"
    /// even though it already performed real arithmetic): computes real
    /// sums/products from literal operands, and refuses -- `None`, never a
    /// guess -- for anything outside that narrow, documented scope.
    #[test]
    fn test_evaluate_constant_instruction_computes_real_values_and_refuses_the_rest() {
        let compiler = JitCompiler::new(&CompilerConfig::default()).expect("construction failed");

        let mut attrs = HashMap::new();
        attrs.insert("const_a".to_string(), "3".to_string());
        attrs.insert("const_b".to_string(), "4".to_string());
        let add = IRInstruction {
            id: 0,
            opcode: IROpcode::Add,
            inputs: vec![],
            outputs: vec![],
            attributes: attrs,
            compute_cost: 1.0,
            memory_cost: 1.0,
        };
        assert_eq!(
            compiler.evaluate_constant_instruction(&add),
            Some("7".to_string())
        );

        let mul = IRInstruction {
            opcode: IROpcode::Mul,
            ..add.clone()
        };
        assert_eq!(
            compiler.evaluate_constant_instruction(&mul),
            Some("12".to_string())
        );

        // Different literal operands must produce a different real result,
        // not a value independent of the input.
        let mut other_attrs = HashMap::new();
        other_attrs.insert("const_a".to_string(), "10".to_string());
        other_attrs.insert("const_b".to_string(), "5".to_string());
        let other_add = IRInstruction {
            attributes: other_attrs,
            ..add.clone()
        };
        assert_eq!(
            compiler.evaluate_constant_instruction(&other_add),
            Some("15".to_string())
        );

        // No literal operands present: refuses rather than guessing.
        let unresolved = IRInstruction {
            attributes: HashMap::new(),
            ..add.clone()
        };
        assert_eq!(compiler.evaluate_constant_instruction(&unresolved), None);

        // An opcode this narrow evaluator does not implement: refuses.
        let sub = IRInstruction {
            opcode: IROpcode::Sub,
            ..add
        };
        assert_eq!(compiler.evaluate_constant_instruction(&sub), None);
    }

    /// `create_fused_instruction`'s cost formula is real (it scales with the
    /// two real input instructions' own costs); only the retention factors
    /// are an assumption, now named and documented rather than bare magic
    /// numbers. This pins the documented formula and proves the output still
    /// varies with input.
    #[test]
    fn test_create_fused_instruction_uses_documented_assumed_retention_factors() {
        let compiler = JitCompiler::new(&CompilerConfig::default()).expect("construction failed");

        let inst1 = IRInstruction {
            id: 0,
            opcode: IROpcode::Add,
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
            compute_cost: 10.0,
            memory_cost: 10.0,
        };
        let inst2 = IRInstruction {
            id: 1,
            opcode: IROpcode::ReLU,
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
            compute_cost: 20.0,
            memory_cost: 20.0,
        };

        let fused = compiler
            .create_fused_instruction(&inst1, &inst2)
            .expect("fusion must succeed for a supported opcode pair");

        assert!(
            (fused.compute_cost - (10.0 + 20.0 * ASSUMED_FUSION_COMPUTE_RETENTION)).abs() < 1e-9
        );
        assert!((fused.memory_cost - (10.0 + 20.0) * ASSUMED_FUSION_MEMORY_RETENTION).abs() < 1e-9);

        // Different input costs must produce a different fused cost: the
        // constants are calibration factors on real per-instruction data,
        // not a fixed output regardless of input.
        let inst2_heavier = IRInstruction {
            compute_cost: 200.0,
            memory_cost: 200.0,
            ..inst2
        };
        let fused_heavier = compiler
            .create_fused_instruction(&inst1, &inst2_heavier)
            .expect("fusion must succeed");
        assert_ne!(fused.compute_cost, fused_heavier.compute_cost);
        assert_ne!(fused.memory_cost, fused_heavier.memory_cost);
    }

    #[test]
    fn test_cache_key_generation() {
        let config = CompilerConfig::default();
        let compiler = JitCompiler::new(&config).expect("operation failed in test");

        let graph = ComputationGraph::new();
        let cache_key = compiler.generate_cache_key(&graph);
        assert!(cache_key.is_ok());

        let key1 = cache_key.expect("operation failed in test");
        let key2 = compiler.generate_cache_key(&graph).expect("operation failed in test");
        assert_eq!(key1, key2); // Same graph should generate same key
    }

    #[test]
    fn test_compilation_statistics() {
        let mut stats = CompilationStatistics::new();
        assert_eq!(stats.cache_hit_rate(), 0.0);

        stats.cache_hits = 3;
        stats.cache_misses = 7;
        assert_eq!(stats.cache_hit_rate(), 0.3);
    }

    #[test]
    fn test_ir_opcodes() {
        assert_ne!(IROpcode::Add, IROpcode::Mul);
        assert_eq!(IROpcode::ReLU, IROpcode::ReLU);
    }

    #[test]
    fn test_interpreter_backend() {
        let config = CompilerConfig::default();
        let backend = InterpreterBackend::new(&config);
        assert!(backend.is_ok());

        let backend = backend.expect("operation failed in test");
        assert_eq!(backend.name(), "Interpreter");
        assert!(!backend.supported_targets().is_empty());
    }
}
