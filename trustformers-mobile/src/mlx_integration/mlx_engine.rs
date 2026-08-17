//! MLX-style engine implementation (Metal + CPU, **not** Apple's MLX framework).
//!
//! See [`super::mlx_types`] for the full statement of what this module is and is not.
//! In short: an MLX-shaped graph API whose ops execute on the real Metal kernels in
//! `trustformers_core::gpu_ops::metal` (with the `metal` feature on macOS) or on CPU
//! tensor kernels otherwise. No MLX linkage, no Neural Engine dispatch.

use super::device_probe::probe_hardware;
use super::mlx_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::error::{CoreError, Result};
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

impl MlxEngine {
    /// Create a new MLX engine with the specified configuration
    pub fn new(config: MlxConfig) -> Result<Self> {
        let device_capabilities = Self::detect_device_capabilities()?;
        Self::validate_config(&config, &device_capabilities)?;
        let memory_config = config.memory_config.clone();

        Ok(Self {
            config,
            device_capabilities,
            performance_metrics: MlxPerformanceMetrics::default(),
            compiled_models: HashMap::new(),
            memory_pool: UnifiedMemoryPool::new(memory_config),
        })
    }

    /// Probe the running machine for its real capabilities.
    ///
    /// Reads `sysctlbyname` for CPU/memory facts and, with the `metal` feature on
    /// macOS, the live `MTLDevice` for GPU facts. Quantities Apple exposes no API for
    /// stay `None`.
    ///
    /// This replaces the old `detect_device_capabilities(device: &AppleSiliconDevice)`,
    /// which detected nothing: it matched the chip enum the *caller* passed in against
    /// a hardcoded table and tacked on `mlx_version: "0.15.0"` for an unlinked
    /// framework. Use [`Self::published_specs_for`] when you explicitly want the
    /// spec-sheet numbers for a named chip.
    pub fn detect_device_capabilities() -> Result<DeviceCapabilities> {
        let hw = probe_hardware()?;

        #[allow(unused_mut)]
        let mut capabilities = DeviceCapabilities {
            cpu_brand: hw.cpu_brand.clone(),
            performance_cores: hw.performance_cores,
            efficiency_cores: hw.efficiency_cores,
            logical_cores: hw.logical_cores,
            unified_memory_gb: hw.memory_gib(),
            amx_version: hw.amx_version,
            metal_device_name: None,
            apple_gpu_family: None,
            metal_max_buffer_bytes: None,
            metal_recommended_working_set_bytes: None,
            metal_unified_memory: None,
            // Not queryable through any Apple API - see the struct docs.
            gpu_cores: None,
            neural_engine_tops: None,
            memory_bandwidth_gbps: None,
        };

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            // Best-effort: a machine without a usable Metal device still has valid CPU
            // capabilities, so a Metal failure narrows the report rather than failing it.
            if let Ok(backend) = trustformers_core::gpu_ops::metal::get_metal_backend() {
                let info = backend.device_info();
                capabilities.metal_device_name = Some(info.name);
                capabilities.apple_gpu_family = info.apple_gpu_family;
                capabilities.metal_max_buffer_bytes = Some(info.max_buffer_length);
                capabilities.metal_recommended_working_set_bytes =
                    Some(info.recommended_max_working_set_size);
                capabilities.metal_unified_memory = Some(info.has_unified_memory);
            }
        }

        Ok(capabilities)
    }

    /// Apple's **published marketing specifications** for a named chip.
    ///
    /// This is a static table transcribed from Apple's product pages, not a
    /// measurement, and it says nothing about the machine this code is running on.
    /// It is kept because callers legitimately want to reason about a target device
    /// they are not currently executing on; the name makes the provenance explicit.
    pub fn published_specs_for(device: &AppleSiliconDevice) -> PublishedChipSpecs {
        let (perf_cores, eff_cores, gpu_cores, ne_tops, memory_gb, bandwidth_gbps, amx) =
            match device {
                AppleSiliconDevice::M1 => (4, 4, 7, 15.8, 8.0, 68.0, true),
                AppleSiliconDevice::M1Pro => (8, 2, 14, 15.8, 16.0, 200.0, true),
                AppleSiliconDevice::M1Max => (8, 2, 24, 15.8, 32.0, 400.0, true),
                AppleSiliconDevice::M1Ultra => (16, 4, 48, 31.6, 64.0, 800.0, true),
                AppleSiliconDevice::M2 => (4, 4, 8, 15.8, 8.0, 100.0, true),
                AppleSiliconDevice::M2Pro => (8, 4, 16, 15.8, 16.0, 200.0, true),
                AppleSiliconDevice::M2Max => (8, 4, 30, 15.8, 32.0, 400.0, true),
                AppleSiliconDevice::M2Ultra => (16, 8, 60, 31.6, 64.0, 800.0, true),
                AppleSiliconDevice::M3 => (4, 4, 10, 18.0, 8.0, 100.0, true),
                AppleSiliconDevice::M3Pro => (6, 6, 18, 18.0, 18.0, 150.0, true),
                AppleSiliconDevice::M3Max => (8, 4, 30, 18.0, 36.0, 300.0, true),
                AppleSiliconDevice::M4 => (4, 6, 10, 38.0, 16.0, 120.0, true),
                AppleSiliconDevice::M4Pro => (8, 4, 20, 38.0, 24.0, 273.0, true),
                AppleSiliconDevice::M4Max => (10, 4, 32, 38.0, 36.0, 546.0, true),
                AppleSiliconDevice::A17Pro => (2, 4, 6, 35.0, 8.0, 64.0, false),
                AppleSiliconDevice::A18 => (2, 4, 5, 35.0, 8.0, 68.0, false),
                AppleSiliconDevice::A18Pro => (2, 4, 6, 35.0, 8.0, 68.0, false),
            };

        PublishedChipSpecs {
            device: *device,
            performance_cores: perf_cores,
            efficiency_cores: eff_cores,
            gpu_cores,
            neural_engine_tops: ne_tops,
            base_configuration_memory_gb: memory_gb,
            memory_bandwidth_gbps: bandwidth_gbps,
            amx_support: amx,
        }
    }

    /// Validate a configuration against the *probed* machine.
    ///
    /// A request of `0` means "auto - use whatever the device has" and is never an
    /// error. A capability the OS does not report (`None`) cannot be validated
    /// against, so the corresponding request is accepted with a warning rather than
    /// rejected on the basis of a number nobody measured.
    fn validate_config(config: &MlxConfig, capabilities: &DeviceCapabilities) -> Result<()> {
        let requested_memory = config.memory_config.max_memory_gb;
        if requested_memory > 0.0 && requested_memory > capabilities.unified_memory_gb {
            return Err(TrustformersError::config_error(
                &format!(
                    "requested {requested_memory:.1} GB exceeds the {:.1} GB installed \
                     on this machine ({})",
                    capabilities.unified_memory_gb, capabilities.cpu_brand
                ),
                "validate_config",
            )
            .into());
        }

        let requested_perf = config.compute_units.cpu_config.performance_cores;
        if requested_perf > 0 {
            match capabilities.performance_cores {
                Some(available) if u32::from(requested_perf) > available => {
                    return Err(TrustformersError::config_error(
                        &format!(
                            "requested {requested_perf} performance cores but \
                             hw.perflevel0.logicalcpu reports {available}"
                        ),
                        "validate_config",
                    )
                    .into());
                },
                Some(_) => {},
                None => tracing::warn!(
                    requested_perf,
                    "this machine does not report performance-core counts; \
                     the request cannot be validated"
                ),
            }
        }

        let requested_eff = config.compute_units.cpu_config.efficiency_cores;
        if requested_eff > 0 {
            match capabilities.efficiency_cores {
                Some(available) if u32::from(requested_eff) > available => {
                    return Err(TrustformersError::config_error(
                        &format!(
                            "requested {requested_eff} efficiency cores but \
                             hw.perflevel1.logicalcpu reports {available}"
                        ),
                        "validate_config",
                    )
                    .into());
                },
                _ => {},
            }
        }

        // GPU core counts are not exposed by any Apple API, so a non-zero request is
        // unverifiable. Say so instead of validating against an invented number.
        let requested_gpu = config.compute_units.gpu_config.gpu_cores;
        if requested_gpu > 0 {
            tracing::warn!(
                requested_gpu,
                "GPU core counts are not queryable on Apple platforms; \
                 the request is recorded but not enforced"
            );
        }

        Ok(())
    }

    /// Compile a model for MLX execution
    pub fn compile_model(
        &mut self,
        model_id: String,
        model_graph: Vec<(MlxOperation, Vec<usize>, HashMap<String, f32>)>,
        optimization_level: OptimizationLevel,
    ) -> Result<String> {
        let compilation_start = std::time::Instant::now();

        // Create optimized graph
        let optimized_graph = self.optimize_graph(model_graph)?;

        // Calculate memory requirements
        let memory_requirements = self.calculate_memory_requirements(&optimized_graph)?;

        // Create performance profile
        let performance_profile = self.create_performance_profile(&optimized_graph)?;

        // Create compilation metadata
        let compilation_metadata = CompilationMetadata {
            compilation_time: std::time::SystemTime::now(),
            engine_backend: Self::backend_label().to_string(),
            optimization_level,
            target_device: self.config.device,
            compilation_options: HashMap::new(),
        };

        let compiled_model = CompiledMlxModel {
            model_id: model_id.clone(),
            compilation_metadata,
            optimized_graph,
            memory_requirements,
            performance_profile,
        };

        self.compiled_models.insert(model_id.clone(), compiled_model);

        // Sub-millisecond resolution: `as_millis()` truncated every fast compile to 0,
        // which read as "not measured" rather than "measured and small".
        let compilation_time = compilation_start.elapsed();
        self.performance_metrics.compilation_time_ms = compilation_time.as_secs_f64() * 1000.0;

        Ok(model_id)
    }

    /// Execute a compiled model
    pub fn execute_model(&mut self, model_id: &str, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let compiled_model = self
            .compiled_models
            .get(model_id)
            .ok_or_else(|| TrustformersError::runtime_error("Model not found".to_string()))?;

        // Clone the data we need to avoid borrow conflicts
        let optimized_graph = compiled_model.optimized_graph.clone();

        let execution_start = std::time::Instant::now();

        // Allocate memory for execution
        self.allocate_execution_memory(&optimized_graph)?;

        // Execute graph nodes in order
        let outputs = self.execute_optimized_graph(&optimized_graph, inputs)?;

        // Record the real cost of the run just performed.
        let execution_time = execution_start.elapsed();
        let executed_nodes = optimized_graph.execution_order.len();
        self.update_performance_metrics(execution_time, executed_nodes);

        Ok(outputs)
    }

    /// Build an executable dataflow graph from a caller-declared op list.
    ///
    /// # Contract
    ///
    /// `model_graph[i] = (operation, producers, parameters)`, where `producers[j]` is
    /// the **index of the node that produces** input `j` of node `i`. A producer index
    /// that is not strictly less than `i` cannot refer to an already-computed value, so
    /// it denotes a **runtime graph input**: those are numbered in order of appearance
    /// and filled from the `inputs` slice handed to `execute_model`.
    ///
    /// # What was wrong before
    ///
    /// Input tensor ids were minted fresh per node (`inputs.iter().map(|_| counter++)`)
    /// and therefore never equalled any producer's output id - the dataflow was never
    /// connected. Meanwhile the *edge* list used `inputs[j]` directly as a node index,
    /// so the canonical single-node example `vec![(MatMul, vec![0, 0], ...)]` produced
    /// a self-edge, i.e. a cycle. That only went unnoticed because dead-code
    /// elimination (whose root rule was "the op is Softmax or LayerNorm") deleted the
    /// entire graph first, after which execution fell through to
    /// `tensor_values.values().last()` and returned an arbitrary `HashMap` entry -
    /// frequently one of the *inputs* rather than the computed result.
    fn optimize_graph(
        &self,
        model_graph: Vec<(MlxOperation, Vec<usize>, HashMap<String, f32>)>,
    ) -> Result<OptimizedGraph> {
        // Pass 1: count runtime inputs so node outputs can be numbered after them.
        let mut runtime_input_count = 0usize;
        for (node_index, (_, producers, _)) in model_graph.iter().enumerate() {
            for producer in producers {
                if *producer >= node_index {
                    runtime_input_count += 1;
                }
            }
        }

        let mut nodes = Vec::with_capacity(model_graph.len());
        let mut edges = Vec::new();
        let mut next_runtime_slot = 0u64;

        // Node `i` writes tensor id `runtime_input_count + i`.
        let output_tensor_of =
            |node_index: usize| -> TensorId { (runtime_input_count + node_index) as TensorId };

        for (node_index, (operation, producers, parameters)) in model_graph.iter().enumerate() {
            let mut input_tensors = Vec::with_capacity(producers.len());
            for producer in producers {
                if *producer < node_index {
                    // Consume the producing node's output; record the dependency edge.
                    input_tensors.push(output_tensor_of(*producer));
                    edges.push(GraphEdge {
                        source: *producer,
                        destination: node_index,
                        tensor_id: output_tensor_of(*producer),
                        data_type: self.config.precision_config.default_precision,
                    });
                } else {
                    // Runtime graph input: no edge, filled by `execute_model`.
                    input_tensors.push(next_runtime_slot);
                    next_runtime_slot += 1;
                }
            }

            nodes.push(GraphNode {
                id: node_index,
                operation: *operation,
                inputs: input_tensors,
                outputs: vec![output_tensor_of(node_index)],
                parameters: parameters.clone(),
                // Assign compute unit based on operation type and build capabilities.
                compute_unit: self.assign_compute_unit(operation),
            });
        }

        // Apply graph optimizations
        if self.config.graph_optimization.operator_fusion {
            self.apply_operator_fusion(&mut nodes, &mut edges)?;
        }

        if self.config.graph_optimization.dead_code_elimination {
            self.apply_dead_code_elimination(&mut nodes, &mut edges)?;
        }

        // Create execution order
        let execution_order = self.create_execution_order(&nodes, &edges)?;

        // Create memory layout
        let memory_layout = self.create_memory_layout(&nodes, &edges)?;

        Ok(OptimizedGraph {
            nodes,
            edges,
            execution_order,
            memory_layout,
        })
    }

    /// Assign a compute unit that this build can actually dispatch to.
    ///
    /// The previous version returned `NeuralEngine` / `Hybrid` labels that no code
    /// path acted on - every op ran on the CPU regardless. Now the label is a fact:
    /// `Gpu` is only ever returned when the crate is built with the `metal` feature
    /// on macOS and the op has a Metal kernel, and `execute_node_operation` really
    /// does dispatch those to the GPU.
    fn assign_compute_unit(&self, operation: &MlxOperation) -> AssignedComputeUnit {
        if !Self::metal_available() {
            return AssignedComputeUnit::CPU;
        }
        let prefers_gpu = !matches!(
            self.config.compute_units.distribution_strategy,
            WorkloadDistributionStrategy::CpuFirst
        );
        match operation {
            MlxOperation::MatMul | MlxOperation::Attention | MlxOperation::LayerNorm
                if prefers_gpu =>
            {
                AssignedComputeUnit::Gpu
            },
            _ => AssignedComputeUnit::CPU,
        }
    }

    /// Whether this build can dispatch to Metal at all.
    pub fn metal_available() -> bool {
        cfg!(all(target_os = "macos", feature = "metal"))
    }

    /// Apply operator fusion optimization
    fn apply_operator_fusion(
        &self,
        nodes: &mut Vec<GraphNode>,
        edges: &mut Vec<GraphEdge>,
    ) -> Result<()> {
        // Simplified operator fusion: combine consecutive element-wise operations
        let mut fusion_candidates = Vec::new();

        for i in 0..nodes.len() - 1 {
            if matches!(
                nodes[i].operation,
                MlxOperation::ElementWise | MlxOperation::Activation
            ) && matches!(
                nodes[i + 1].operation,
                MlxOperation::ElementWise | MlxOperation::Activation
            ) {
                fusion_candidates.push((i, i + 1));
            }
        }

        // Apply fusion (simplified implementation)
        for (first, second) in fusion_candidates.iter().rev() {
            if *second < nodes.len() {
                // Merge parameters
                let mut merged_params = nodes[*first].parameters.clone();
                merged_params.extend(nodes[*second].parameters.clone());

                // Update first node
                nodes[*first].parameters = merged_params;
                nodes[*first].outputs = nodes[*second].outputs.clone();

                // Remove second node
                nodes.remove(*second);

                // Update edge references
                for edge in edges.iter_mut() {
                    if edge.source > *second {
                        edge.source -= 1;
                    }
                    if edge.destination > *second {
                        edge.destination -= 1;
                    } else if edge.destination == *second {
                        edge.destination = *first;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply dead code elimination
    fn apply_dead_code_elimination(
        &self,
        nodes: &mut Vec<GraphNode>,
        edges: &mut Vec<GraphEdge>,
    ) -> Result<()> {
        // Mark nodes that are reachable from outputs
        let mut reachable = vec![false; nodes.len()];

        // Mark graph outputs as reachable: a node whose products nothing else consumes.
        // (Same dataflow rule as `is_output_node`, expressed over the pre-optimisation
        // node list. The old rule marked only Softmax/LayerNorm nodes, so a graph
        // ending in a MatMul had no reachable root and every node was eliminated.)
        let node_snapshot = nodes.clone();
        for node in &*nodes {
            let is_sink = node.outputs.is_empty()
                || node.outputs.iter().any(|produced| {
                    !node_snapshot
                        .iter()
                        .any(|other| other.id != node.id && other.inputs.contains(produced))
                });
            if is_sink {
                reachable[node.id] = true;
            }
        }

        // Propagate reachability backwards
        let mut changed = true;
        while changed {
            changed = false;
            for edge in &*edges {
                if reachable[edge.destination] && !reachable[edge.source] {
                    reachable[edge.source] = true;
                    changed = true;
                }
            }
        }

        // Remove unreachable nodes
        let mut id_mapping = HashMap::new();
        let mut new_id = 0;

        for (old_id, &is_reachable) in reachable.iter().enumerate() {
            if is_reachable {
                id_mapping.insert(old_id, new_id);
                new_id += 1;
            }
        }

        // Filter nodes and update IDs
        let mut filtered_nodes = Vec::new();
        for (i, node) in nodes.iter().enumerate() {
            if reachable[i] {
                let mut updated_node = node.clone();
                updated_node.id = id_mapping.get(&i).copied().ok_or_else(|| {
                    TrustformersError::runtime_error("missing node id mapping".to_string())
                })?;
                filtered_nodes.push(updated_node);
            }
        }

        // Filter edges and update references
        let mut filtered_edges = Vec::new();
        for edge in &*edges {
            if reachable[edge.source] && reachable[edge.destination] {
                let mut updated_edge = edge.clone();
                updated_edge.source = id_mapping.get(&edge.source).copied().ok_or_else(|| {
                    TrustformersError::runtime_error("missing edge source mapping".to_string())
                })?;
                updated_edge.destination =
                    id_mapping.get(&edge.destination).copied().ok_or_else(|| {
                        TrustformersError::runtime_error(
                            "missing edge destination mapping".to_string(),
                        )
                    })?;
                filtered_edges.push(updated_edge);
            }
        }

        *nodes = filtered_nodes;
        *edges = filtered_edges;

        Ok(())
    }

    /// Check if a node is an output node
    /// A node is a graph output when nothing else consumes what it produces.
    ///
    /// The previous rule was "the operation is Softmax or LayerNorm", which made the
    /// output set depend on op *type* rather than on the dataflow - a graph ending in
    /// a MatMul had no outputs at all and fell through to the
    /// "return the last computed tensor" branch below, which read an arbitrary entry
    /// out of a `HashMap`. That returned a *nondeterministically chosen* tensor: on a
    /// two-input MatMul graph it handed back one of the inputs instead of the product.
    fn is_output_node(&self, node: &GraphNode, graph: &OptimizedGraph) -> bool {
        node.outputs.iter().any(|produced| {
            !graph
                .nodes
                .iter()
                .any(|other| other.id != node.id && other.inputs.contains(produced))
        })
    }

    /// Create execution order using topological sort
    fn create_execution_order(
        &self,
        nodes: &[GraphNode],
        edges: &[GraphEdge],
    ) -> Result<Vec<usize>> {
        let mut in_degree = vec![0; nodes.len()];
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();

        // Calculate in-degrees and build adjacency list
        for edge in edges {
            in_degree[edge.destination] += 1;
            adj_list.entry(edge.source).or_default().push(edge.destination);
        }

        // Topological sort using Kahn's algorithm
        let mut queue = std::collections::VecDeque::new();
        let mut execution_order = Vec::new();

        // Add nodes with no incoming edges
        for (i, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(i);
            }
        }

        while let Some(node_id) = queue.pop_front() {
            execution_order.push(node_id);

            if let Some(neighbors) = adj_list.get(&node_id) {
                for &neighbor in neighbors {
                    in_degree[neighbor] -= 1;
                    if in_degree[neighbor] == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if execution_order.len() != nodes.len() {
            return Err(
                TrustformersError::runtime_error("Graph contains cycles".to_string()).into(),
            );
        }

        Ok(execution_order)
    }

    /// Create memory layout for optimized execution
    fn create_memory_layout(
        &self,
        nodes: &[GraphNode],
        _edges: &[GraphEdge],
    ) -> Result<MemoryLayout> {
        let mut tensor_allocations = HashMap::new();
        let mut current_offset = 0;
        let mut alignment_requirements = HashMap::new();

        for node in nodes {
            for &tensor_id in &node.inputs {
                if let std::collections::hash_map::Entry::Vacant(e) =
                    tensor_allocations.entry(tensor_id)
                {
                    let size_bytes = self.estimate_tensor_size(&node.operation);
                    let alignment = 64; // 64-byte alignment for Apple Silicon

                    // Align offset
                    current_offset = (current_offset + alignment - 1) & !(alignment - 1);

                    e.insert(MemoryAllocation {
                        offset: current_offset,
                        size_bytes,
                        memory_type: MemoryType::SharedMemory,
                        lifetime: MemoryLifetime::Temporary,
                    });

                    alignment_requirements.insert(tensor_id, alignment);
                    current_offset += size_bytes;
                }
            }

            for &tensor_id in &node.outputs {
                if let std::collections::hash_map::Entry::Vacant(e) =
                    tensor_allocations.entry(tensor_id)
                {
                    let size_bytes = self.estimate_tensor_size(&node.operation);
                    let alignment = 64;

                    current_offset = (current_offset + alignment - 1) & !(alignment - 1);

                    e.insert(MemoryAllocation {
                        offset: current_offset,
                        size_bytes,
                        memory_type: MemoryType::SharedMemory,
                        lifetime: MemoryLifetime::Temporary,
                    });

                    alignment_requirements.insert(tensor_id, alignment);
                    current_offset += size_bytes;
                }
            }
        }

        Ok(MemoryLayout {
            tensor_allocations,
            total_memory_bytes: current_offset,
            alignment_requirements,
        })
    }

    /// Estimate tensor size based on operation type (simplified)
    fn estimate_tensor_size(&self, operation: &MlxOperation) -> usize {
        let element_size = match self.config.precision_config.default_precision {
            MlxPrecision::Float32 => 4,
            MlxPrecision::Float16 | MlxPrecision::BFloat16 => 2,
            MlxPrecision::Int8 => 1,
            MlxPrecision::Int4 => 1, // Rounded up
            _ => 4,
        };

        match operation {
            MlxOperation::MatMul => 1024 * 1024 * element_size, // 1M elements
            MlxOperation::Convolution => 512 * 512 * element_size,
            MlxOperation::Attention => 2048 * 768 * element_size,
            _ => 256 * 256 * element_size,
        }
    }

    /// Calculate memory requirements for the optimized graph
    fn calculate_memory_requirements(&self, graph: &OptimizedGraph) -> Result<MemoryRequirements> {
        let base_memory_gb =
            graph.memory_layout.total_memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0);

        Ok(MemoryRequirements {
            minimum_memory_gb: base_memory_gb,
            recommended_memory_gb: base_memory_gb * 1.5,
            peak_memory_gb: base_memory_gb * 2.0,
            fragmentation_factor: 1.2,
        })
    }

    /// Create performance profile for the compiled model
    fn create_performance_profile(
        &self,
        graph: &OptimizedGraph,
    ) -> Result<ModelPerformanceProfile> {
        let mut total_ops = 0;
        let mut estimated_latency_ms = 0.0;

        for node in &graph.nodes {
            total_ops += 1;

            // Estimate latency based on operation type and compute unit
            // Rough per-op cost model used only to order/ size the graph; it is
            // reported as an *estimate* and never as a measurement. The Neural Engine
            // rows are gone with the enum variant - nothing dispatches there.
            let op_latency = match (node.operation, node.compute_unit) {
                (MlxOperation::MatMul, AssignedComputeUnit::Gpu) => 1.2,
                (MlxOperation::MatMul, AssignedComputeUnit::CPU) => 5.0,
                (MlxOperation::Convolution, AssignedComputeUnit::Gpu) => 2.0,
                (MlxOperation::Convolution, AssignedComputeUnit::CPU) => 8.0,
                (MlxOperation::Attention, AssignedComputeUnit::Gpu) => 3.0,
                (MlxOperation::Attention, AssignedComputeUnit::CPU) => 10.0,
                (MlxOperation::LayerNorm, AssignedComputeUnit::Gpu) => 0.4,
                _ => 1.0,
            };

            estimated_latency_ms += op_latency;
        }

        let expected_throughput =
            if estimated_latency_ms > 0.0 { 1000.0 / estimated_latency_ms } else { 0.0 };

        let mut accuracy_metrics = HashMap::new();
        accuracy_metrics.insert("estimated_accuracy".to_string(), 0.95);

        Ok(ModelPerformanceProfile {
            expected_latency_ms: estimated_latency_ms,
            expected_throughput,
            power_consumption_watts: 15.0, // Estimated for Apple Silicon
            thermal_impact: 0.3,
            accuracy_metrics,
        })
    }

    /// Allocate execution memory for the optimized graph
    fn allocate_execution_memory(&mut self, graph: &OptimizedGraph) -> Result<()> {
        for (tensor_id, allocation) in &graph.memory_layout.tensor_allocations {
            self.memory_pool.allocate(*tensor_id, allocation.clone())?;
        }
        Ok(())
    }

    /// Execute the optimized graph
    fn execute_optimized_graph(
        &self,
        graph: &OptimizedGraph,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        let mut tensor_values: HashMap<TensorId, Tensor> = HashMap::new();

        // Initialize input tensors
        for (i, input) in inputs.iter().enumerate() {
            tensor_values.insert(i as TensorId, input.clone());
        }

        // Execute nodes in order
        for &node_id in &graph.execution_order {
            let node = &graph.nodes[node_id];

            // Collect input tensors
            let input_tensors: Result<Vec<Tensor>> = node
                .inputs
                .iter()
                .map(|&tensor_id| {
                    tensor_values
                        .get(&tensor_id)
                        .cloned()
                        .ok_or_else(|| {
                            TrustformersError::runtime_error("Missing input tensor".to_string())
                        })
                        .map_err(|e: TrustformersError| e.into())
                })
                .collect();

            let input_tensors = input_tensors?;

            // Execute operation
            let output_tensors = self.execute_node_operation(node, &input_tensors)?;

            // Store output tensors
            for (i, output_tensor) in output_tensors.into_iter().enumerate() {
                if let Some(&output_tensor_id) = node.outputs.get(i) {
                    tensor_values.insert(output_tensor_id, output_tensor);
                }
            }
        }

        // Collect final outputs in execution order, so the result is deterministic.
        let mut outputs = Vec::new();
        for &node_id in &graph.execution_order {
            let Some(node) = graph.nodes.get(node_id) else {
                continue;
            };
            if !self.is_output_node(node, graph) {
                continue;
            }
            for output_tensor_id in &node.outputs {
                if let Some(tensor) = tensor_values.get(output_tensor_id) {
                    outputs.push(tensor.clone());
                }
            }
        }

        if outputs.is_empty() {
            // Fall back to the last *executed* node's output - still deterministic.
            // Never `tensor_values.values().last()`: HashMap order is arbitrary.
            if let Some(tensor) = graph
                .execution_order
                .last()
                .and_then(|node_id| graph.nodes.get(*node_id))
                .and_then(|node| node.outputs.first())
                .and_then(|tensor_id| tensor_values.get(tensor_id))
            {
                outputs.push(tensor.clone());
            }
        }

        if outputs.is_empty() {
            return Err(TrustformersError::runtime_error(
                "graph execution produced no output tensors".to_string(),
            )
            .into());
        }

        Ok(outputs)
    }

    /// Execute a single node operation
    fn execute_node_operation(&self, node: &GraphNode, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        match node.operation {
            MlxOperation::MatMul => self.execute_matmul(inputs, &node.parameters),
            MlxOperation::Convolution => self.execute_convolution(inputs, &node.parameters),
            MlxOperation::Attention => self.execute_attention(inputs, &node.parameters),
            MlxOperation::LayerNorm => self.execute_layer_norm(inputs, &node.parameters),
            MlxOperation::BatchNorm => self.execute_batch_norm(inputs, &node.parameters),
            MlxOperation::Activation => self.execute_activation(inputs, &node.parameters),
            MlxOperation::Embedding => self.execute_embedding(inputs, &node.parameters),
            MlxOperation::Softmax => self.execute_softmax(inputs, &node.parameters),
            MlxOperation::Reduction => self.execute_reduction(inputs, &node.parameters),
            MlxOperation::ElementWise => self.execute_elementwise(inputs, &node.parameters),
        }
    }

    /// Real matrix multiplication: `A[m, k] @ B[k, n]`.
    ///
    /// Dispatches to the Metal GEMM in `trustformers_core::gpu_ops::metal` when this
    /// build has the `metal` feature on macOS, otherwise to the crate's CPU tensor
    /// kernels (`oxiblas`-backed). The previous body was a hand-rolled scalar triple
    /// loop labelled "Optimized matrix multiplication for Apple Silicon".
    fn execute_matmul(
        &self,
        inputs: &[Tensor],
        _parameters: &HashMap<String, f32>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(TrustformersError::runtime_error(
                "MatMul requires exactly 2 input tensors".to_string(),
            )
            .into());
        }

        let a = &inputs[0];
        let b = &inputs[1];
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(
                TrustformersError::runtime_error("MatMul requires 2D tensors".to_string()).into(),
            );
        }
        let (m, k) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);
        if k != k2 {
            return Err(TrustformersError::runtime_error(format!(
                "MatMul dimensions incompatible: [{m}, {k}] @ [{k2}, {n}]"
            ))
            .into());
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let backend = trustformers_core::gpu_ops::metal::get_metal_backend()?;
            let out = backend.matmul_f32(&a.data()?, &b.data()?, m, k, n)?;
            Ok(vec![Tensor::from_vec(out, &[m, n])?])
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            // Real CPU GEMM through the core tensor kernels, not a scalar loop.
            Ok(vec![a.matmul(b)?])
        }
    }

    /// Real direct 2-D convolution (cross-correlation), NCHW / OIHW layout.
    ///
    /// The previous body was `Ok(vec![input.clone()])` behind the comment
    /// "For simplicity, return input (real implementation would do actual
    /// convolution)" - a convolution that returned its argument unchanged.
    ///
    /// # Parameters
    ///
    /// The tensor shapes carry the geometry, so only the sampling parameters are read
    /// from `parameters`, each with an explicit default:
    ///
    /// * `stride` (default 1), `padding` (default 0, zero padding), `dilation`
    ///   (default 1).
    ///
    /// `inputs[0]` must be `[batch, in_channels, height, width]` and `inputs[1]`
    /// `[out_channels, in_channels, kernel_h, kernel_w]`; an optional `inputs[2]` is a
    /// per-output-channel bias. Anything else is a structured error, never a silent
    /// pass-through.
    fn execute_convolution(
        &self,
        inputs: &[Tensor],
        parameters: &HashMap<String, f32>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() < 2 {
            return Err(TrustformersError::runtime_error(
                "Convolution requires an input tensor and a kernel tensor".to_string(),
            )
            .into());
        }
        let input = &inputs[0];
        let kernel = &inputs[1];
        let input_shape = input.shape();
        let kernel_shape = kernel.shape();

        if input_shape.len() != 4 {
            return Err(TrustformersError::runtime_error(format!(
                "Convolution input must be 4-D [batch, in_channels, height, width], got {input_shape:?}"
            ))
            .into());
        }
        if kernel_shape.len() != 4 {
            return Err(TrustformersError::runtime_error(format!(
                "Convolution kernel must be 4-D [out_channels, in_channels, kh, kw], got {kernel_shape:?}"
            ))
            .into());
        }
        let (batch, in_channels, height, width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (out_channels, kernel_in_channels, kernel_h, kernel_w) = (
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            kernel_shape[3],
        );
        if kernel_in_channels != in_channels {
            return Err(TrustformersError::runtime_error(format!(
                "Convolution channel mismatch: input has {in_channels} channels, kernel expects \
                 {kernel_in_channels}"
            ))
            .into());
        }

        let stride = read_positive_usize(parameters, "stride", 1)?;
        let dilation = read_positive_usize(parameters, "dilation", 1)?;
        let padding = read_non_negative_usize(parameters, "padding", 0)?;

        let effective_h = dilation * (kernel_h - 1) + 1;
        let effective_w = dilation * (kernel_w - 1) + 1;
        if height + 2 * padding < effective_h || width + 2 * padding < effective_w {
            return Err(TrustformersError::runtime_error(format!(
                "Convolution kernel {effective_h}x{effective_w} (after dilation) does not fit a \
                 {height}x{width} input padded by {padding}"
            ))
            .into());
        }
        let out_h = (height + 2 * padding - effective_h) / stride + 1;
        let out_w = (width + 2 * padding - effective_w) / stride + 1;

        let bias = match inputs.get(2) {
            Some(bias_tensor) => {
                let data = bias_tensor.data()?;
                if data.len() != out_channels {
                    return Err(TrustformersError::runtime_error(format!(
                        "Convolution bias has {} elements but there are {out_channels} output channels",
                        data.len()
                    ))
                    .into());
                }
                Some(data)
            },
            None => None,
        };

        let input_data = input.data()?;
        let kernel_data = kernel.data()?;
        let mut output = vec![0.0f32; batch * out_channels * out_h * out_w];

        for b in 0..batch {
            for oc in 0..out_channels {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut acc = bias.as_ref().map(|values| values[oc]).unwrap_or(0.0);
                        for ic in 0..in_channels {
                            for ky in 0..kernel_h {
                                // Signed arithmetic so padding on the top/left edge is
                                // recognised rather than wrapping around.
                                let iy = (oy * stride + ky * dilation) as isize - padding as isize;
                                if iy < 0 || iy >= height as isize {
                                    continue;
                                }
                                for kx in 0..kernel_w {
                                    let ix =
                                        (ox * stride + kx * dilation) as isize - padding as isize;
                                    if ix < 0 || ix >= width as isize {
                                        continue;
                                    }
                                    let input_index =
                                        ((b * in_channels + ic) * height + iy as usize) * width
                                            + ix as usize;
                                    let kernel_index =
                                        ((oc * in_channels + ic) * kernel_h + ky) * kernel_w + kx;
                                    acc += input_data[input_index] * kernel_data[kernel_index];
                                }
                            }
                        }
                        let out_index = ((b * out_channels + oc) * out_h + oy) * out_w + ox;
                        output[out_index] = acc;
                    }
                }
            }
        }

        Ok(vec![Tensor::from_vec(
            output,
            &[batch, out_channels, out_h, out_w],
        )?])
    }

    /// Real multi-head scaled dot-product attention.
    ///
    /// The previous body multiplied the input by a scalar (`result[i] = input_data[i]
    /// * scale`) and called it attention.
    ///
    /// Takes `[q, k, v]`, each `[seq_len, num_heads * head_dim]`. `parameters` may set
    /// `num_heads` (default 1) and `causal` (non-zero enables the causal mask, which
    /// is the default). With the `metal` feature on macOS the computation runs on the
    /// GPU through `attention_gpu_to_gpu`; otherwise it is an exact CPU
    /// implementation of the same function.
    fn execute_attention(
        &self,
        inputs: &[Tensor],
        parameters: &HashMap<String, f32>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() < 3 {
            return Err(TrustformersError::runtime_error(format!(
                "Attention requires three input tensors (query, key, value), got {}",
                inputs.len()
            ))
            .into());
        }
        let (q, k, v) = (&inputs[0], &inputs[1], &inputs[2]);
        let shape = q.shape();
        if shape.len() != 2 {
            return Err(TrustformersError::runtime_error(format!(
                "Attention expects 2-D [seq_len, num_heads * head_dim] tensors, got {shape:?}"
            ))
            .into());
        }
        if k.shape() != shape || v.shape() != shape {
            return Err(TrustformersError::runtime_error(format!(
                "Attention requires q/k/v of equal shape, got {:?} / {:?} / {:?}",
                shape,
                k.shape(),
                v.shape()
            ))
            .into());
        }
        let (seq_len, hidden) = (shape[0], shape[1]);
        let num_heads = read_positive_usize(parameters, "num_heads", 1)?;
        if hidden % num_heads != 0 {
            return Err(TrustformersError::runtime_error(format!(
                "Attention hidden size {hidden} is not divisible by num_heads {num_heads}"
            ))
            .into());
        }
        let head_dim = hidden / num_heads;
        let causal = parameters.get("causal").copied().unwrap_or(1.0) != 0.0;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            // The Metal kernel always applies the causal mask, so only take the GPU
            // path when that is what was asked for.
            if causal {
                let backend = trustformers_core::gpu_ops::metal::get_metal_backend()?;
                let q_id = backend.create_transient_buffer(&q.data()?)?;
                let k_id = backend.create_transient_buffer(&k.data()?)?;
                let v_id = backend.create_transient_buffer(&v.data()?)?;
                let outcome = backend
                    .attention_gpu_to_gpu(&q_id, &k_id, &v_id, 1, seq_len, num_heads, head_dim)
                    .and_then(|out_id| {
                        let data = backend.download_buffer_to_vec(&out_id);
                        backend.release_buffers(&[out_id])?;
                        data
                    });
                backend.release_buffers(&[q_id, k_id, v_id])?;
                return Ok(vec![Tensor::from_vec(outcome?, &[seq_len, hidden])?]);
            }
        }

        let q_data = q.data()?;
        let k_data = k.data()?;
        let v_data = v.data()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_len * hidden];

        for head in 0..num_heads {
            let head_offset = head * head_dim;
            for row in 0..seq_len {
                let limit = if causal { row + 1 } else { seq_len };
                // Online softmax: one pass over the keys, numerically stable, no
                // seq_len-sized scratch buffer.
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;
                let mut accumulator = vec![0.0f32; head_dim];
                for col in 0..limit {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_data[row * hidden + head_offset + d]
                            * k_data[col * hidden + head_offset + d];
                    }
                    let score = dot * scale;
                    let new_max = running_max.max(score);
                    let correction = (running_max - new_max).exp();
                    let weight = (score - new_max).exp();
                    running_sum = running_sum * correction + weight;
                    for d in 0..head_dim {
                        accumulator[d] = accumulator[d] * correction
                            + weight * v_data[col * hidden + head_offset + d];
                    }
                    running_max = new_max;
                }
                let inv_sum = if running_sum > 0.0 { 1.0 / running_sum } else { 0.0 };
                for d in 0..head_dim {
                    output[row * hidden + head_offset + d] = accumulator[d] * inv_sum;
                }
            }
        }

        Ok(vec![Tensor::from_vec(output, &[seq_len, hidden])?])
    }

    /// Execute layer normalization (simplified implementation)
    fn execute_layer_norm(
        &self,
        inputs: &[Tensor],
        parameters: &HashMap<String, f32>,
    ) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(TrustformersError::runtime_error(
                "LayerNorm requires input tensor".to_string(),
            )
            .into());
        }

        let input = &inputs[0];
        let epsilon = parameters.get("epsilon").copied().unwrap_or(1e-5);
        let input_data = input.data()?;
        let shape = input.shape();

        if shape.len() < 2 {
            return Err(TrustformersError::runtime_error(
                "LayerNorm requires at least 2D input".to_string(),
            )
            .into());
        }

        let last_dim = shape[shape.len() - 1];
        let batch_size = input_data.len() / last_dim;
        let mut result = vec![0.0f32; input_data.len()];

        for b in 0..batch_size {
            let start_idx = b * last_dim;
            let end_idx = start_idx + last_dim;

            // Compute mean and variance
            let mean = input_data[start_idx..end_idx].iter().sum::<f32>() / last_dim as f32;
            let variance =
                input_data[start_idx..end_idx].iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                    / last_dim as f32;

            let inv_std = 1.0 / (variance + epsilon).sqrt();

            // Normalize
            for i in 0..last_dim {
                result[start_idx + i] = (input_data[start_idx + i] - mean) * inv_std;
            }
        }

        let result_tensor = Tensor::from_vec(result, &shape)?;
        Ok(vec![result_tensor])
    }

    /// Execute batch normalization.
    ///
    /// Computes `y = (x - mean) / sqrt(var + eps) * gamma + beta`, where the mean
    /// and variance are gathered per feature across the batch dimension (the
    /// defining behaviour of batch normalization, as opposed to layer norm which
    /// normalizes per sample across features). The input is treated as a
    /// `[batch, features]` matrix where `features` is the size of the last
    /// dimension; the remaining leading dimensions are flattened into the batch.
    fn execute_batch_norm(
        &self,
        inputs: &[Tensor],
        parameters: &HashMap<String, f32>,
    ) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(TrustformersError::runtime_error(
                "BatchNorm requires input tensor".to_string(),
            )
            .into());
        }

        let input = &inputs[0];
        let epsilon = parameters.get("epsilon").copied().unwrap_or(1e-5);
        let gamma = parameters.get("gamma").copied().unwrap_or(1.0);
        let beta = parameters.get("beta").copied().unwrap_or(0.0);
        let input_data = input.data()?;
        let shape = input.shape();

        if shape.is_empty() {
            return Err(TrustformersError::runtime_error(
                "BatchNorm requires at least 1D input".to_string(),
            )
            .into());
        }

        let num_features = shape[shape.len() - 1];
        if num_features == 0 {
            return Ok(vec![input.clone()]);
        }
        let batch_size = input_data.len() / num_features;
        let mut result = vec![0.0f32; input_data.len()];

        // Per-feature statistics across the batch dimension.
        for feature in 0..num_features {
            let mut mean = 0.0f32;
            for batch in 0..batch_size {
                mean += input_data[batch * num_features + feature];
            }
            mean /= batch_size as f32;

            let mut variance = 0.0f32;
            for batch in 0..batch_size {
                let diff = input_data[batch * num_features + feature] - mean;
                variance += diff * diff;
            }
            variance /= batch_size as f32;

            let inv_std = 1.0 / (variance + epsilon).sqrt();
            for batch in 0..batch_size {
                let idx = batch * num_features + feature;
                result[idx] = (input_data[idx] - mean) * inv_std * gamma + beta;
            }
        }

        let result_tensor = Tensor::from_vec(result, &shape)?;
        Ok(vec![result_tensor])
    }

    /// Execute activation function (simplified implementation)
    fn execute_activation(
        &self,
        inputs: &[Tensor],
        parameters: &HashMap<String, f32>,
    ) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(TrustformersError::runtime_error(
                "Activation requires input tensor".to_string(),
            )
            .into());
        }

        let input = &inputs[0];
        let activation_type = parameters.get("type").copied().unwrap_or(0.0) as i32;
        let input_data = input.data()?;
        let shape = input.shape();
        let mut result = vec![0.0f32; input_data.len()];

        match activation_type {
            0 => {
                // ReLU
                for i in 0..input_data.len() {
                    result[i] = input_data[i].max(0.0);
                }
            },
            1 => {
                // GELU
                for i in 0..input_data.len() {
                    let x = input_data[i];
                    result[i] = x * 0.5 * (1.0 + (x * 0.797_884_6).tanh());
                }
            },
            _ => {
                // Identity
                result = input_data.to_vec();
            },
        }

        let result_tensor = Tensor::from_vec(result, &shape)?;
        Ok(vec![result_tensor])
    }

    /// Execute embedding lookup.
    ///
    /// Gathers rows from the embedding table for each input token id. The first
    /// input holds the integer token ids (any shape), the second input is the
    /// `[vocab_size, embedding_dim]` embedding table. For each id the matching
    /// row is selected, producing an output of shape `indices_shape +
    /// [embedding_dim]`.
    fn execute_embedding(
        &self,
        inputs: &[Tensor],
        _parameters: &HashMap<String, f32>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() < 2 {
            return Err(TrustformersError::runtime_error(
                "Embedding requires indices and embedding table".to_string(),
            )
            .into());
        }

        let indices = &inputs[0];
        let table = &inputs[1];
        let table_shape = table.shape();
        if table_shape.len() != 2 {
            return Err(TrustformersError::runtime_error(
                "Embedding table must be a 2D [vocab_size, embedding_dim] tensor".to_string(),
            )
            .into());
        }

        let vocab_size = table_shape[0];
        let embedding_dim = table_shape[1];
        let table_data = table.data()?;
        let index_data = indices.data()?;

        let mut result = vec![0.0f32; index_data.len() * embedding_dim];
        for (token, &raw_id) in index_data.iter().enumerate() {
            let id = raw_id.round() as i64;
            if id < 0 || id as usize >= vocab_size {
                return Err(TrustformersError::runtime_error(format!(
                    "Embedding index {} out of bounds for vocabulary size {}",
                    id, vocab_size
                ))
                .into());
            }
            let row_start = id as usize * embedding_dim;
            let dst_start = token * embedding_dim;
            result[dst_start..dst_start + embedding_dim]
                .copy_from_slice(&table_data[row_start..row_start + embedding_dim]);
        }

        let mut output_shape = indices.shape();
        output_shape.push(embedding_dim);
        let result_tensor = Tensor::from_vec(result, &output_shape)?;
        Ok(vec![result_tensor])
    }

    /// Execute softmax (simplified implementation)
    fn execute_softmax(
        &self,
        inputs: &[Tensor],
        _parameters: &HashMap<String, f32>,
    ) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(TrustformersError::runtime_error(
                "Softmax requires input tensor".to_string(),
            )
            .into());
        }

        let input = &inputs[0];
        let input_data = input.data()?;
        let shape = input.shape();
        let mut result = vec![0.0f32; input_data.len()];

        if shape.is_empty() {
            return Ok(vec![input.clone()]);
        }

        let last_dim = shape[shape.len() - 1];
        let batch_size = input_data.len() / last_dim;

        for b in 0..batch_size {
            let start_idx = b * last_dim;
            let end_idx = start_idx + last_dim;

            // Find max for numerical stability
            let max_val =
                input_data[start_idx..end_idx].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Compute exponentials and sum
            let mut sum_exp = 0.0f32;
            for i in start_idx..end_idx {
                let exp_val = (input_data[i] - max_val).exp();
                result[i] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize
            for i in start_idx..end_idx {
                result[i] /= sum_exp;
            }
        }

        let result_tensor = Tensor::from_vec(result, &shape)?;
        Ok(vec![result_tensor])
    }

    /// Execute reduction operation (simplified implementation)
    fn execute_reduction(
        &self,
        inputs: &[Tensor],
        parameters: &HashMap<String, f32>,
    ) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(TrustformersError::runtime_error(
                "Reduction requires input tensor".to_string(),
            )
            .into());
        }

        let input = &inputs[0];
        let reduction_type = parameters.get("type").copied().unwrap_or(0.0) as i32;
        let input_data = input.data()?;

        let result_val = match reduction_type {
            0 => input_data.iter().sum::<f32>(), // Sum
            1 => input_data.iter().sum::<f32>() / input_data.len() as f32, // Mean
            2 => input_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)), // Max
            _ => input_data.iter().sum::<f32>(),
        };

        let result_tensor = Tensor::from_vec(vec![result_val], &[1])?;
        Ok(vec![result_tensor])
    }

    /// Execute element-wise operation (simplified implementation)
    fn execute_elementwise(
        &self,
        inputs: &[Tensor],
        parameters: &HashMap<String, f32>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() < 2 {
            return Err(TrustformersError::runtime_error(
                "ElementWise requires at least 2 input tensors".to_string(),
            )
            .into());
        }

        let a = &inputs[0];
        let b = &inputs[1];
        let op_type = parameters.get("type").copied().unwrap_or(0.0) as i32;

        if a.shape() != b.shape() {
            return Err(TrustformersError::runtime_error(
                "ElementWise inputs must have same shape".to_string(),
            )
            .into());
        }

        let a_data = a.data()?;
        let b_data = b.data()?;
        let shape = a.shape();
        let mut result = vec![0.0f32; a_data.len()];

        match op_type {
            0 => {
                // Add
                for i in 0..a_data.len() {
                    result[i] = a_data[i] + b_data[i];
                }
            },
            1 => {
                // Multiply
                for i in 0..a_data.len() {
                    result[i] = a_data[i] * b_data[i];
                }
            },
            2 => {
                // Subtract
                for i in 0..a_data.len() {
                    result[i] = a_data[i] - b_data[i];
                }
            },
            _ => {
                // Add (default)
                for i in 0..a_data.len() {
                    result[i] = a_data[i] + b_data[i];
                }
            },
        }

        let result_tensor = Tensor::from_vec(result, &shape)?;
        Ok(vec![result_tensor])
    }

    /// Record what the last execution actually cost.
    ///
    /// Every field is measured here: wall-clock time from `Instant`, node count from
    /// the executed graph, process CPU and RSS from `sysinfo`, pool occupancy from the
    /// allocator. The previous body was introduced by the comment
    /// `// Simulate MLX performance characteristics` and assigned the constants
    /// `cpu_utilization = 75.0`, `gpu_utilization = 85.0`,
    /// `neural_engine_utilization = 90.0` plus a fixed 15 W and a fixed thermal state.
    fn update_performance_metrics(&mut self, execution_time: std::time::Duration, nodes: usize) {
        let seconds = execution_time.as_secs_f64();
        self.performance_metrics.last_execution_ms = seconds * 1000.0;
        self.performance_metrics.last_execution_nodes = nodes;
        self.performance_metrics.ops_per_second =
            if seconds > 0.0 { nodes as f64 / seconds } else { 0.0 };

        let (cpu, rss) = Self::sample_process_usage();
        self.performance_metrics.cpu_utilization = cpu;
        self.performance_metrics.process_memory_gb = rss;
        self.performance_metrics.pool_memory_gb = self.memory_pool.get_total_allocated_gb();
    }

    /// Sample this process's CPU usage and resident memory.
    ///
    /// Returns `(cpu_percent, rss_gib)`; either component is `None` when the platform
    /// does not report it. CPU percentage needs two samples separated by at least
    /// `MINIMUM_CPU_UPDATE_INTERVAL`, so the first call in a process may legitimately
    /// return `None` for CPU rather than a made-up figure.
    fn sample_process_usage() -> (Option<f32>, Option<f32>) {
        use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

        let pid = Pid::from_u32(std::process::id());
        let mut system = System::new();
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_cpu().with_memory(),
        );
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_cpu().with_memory(),
        );

        match system.process(pid) {
            Some(process) => {
                let rss_gib = process.memory() as f32 / (1024.0 * 1024.0 * 1024.0);
                (Some(process.cpu_usage()), Some(rss_gib))
            },
            None => (None, None),
        }
    }

    /// `"metal"` when this build dispatches to the GPU, `"cpu"` otherwise.
    pub fn backend_label() -> &'static str {
        if Self::metal_available() {
            "metal"
        } else {
            "cpu"
        }
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> &MlxPerformanceMetrics {
        &self.performance_metrics
    }

    /// Get device capabilities
    pub fn get_device_capabilities(&self) -> &DeviceCapabilities {
        &self.device_capabilities
    }

    /// Export a performance report containing only measured or probed values.
    ///
    /// Anything this process cannot observe is printed as
    /// `not available (<why>)` rather than as a number. In particular GPU and
    /// Neural-Engine utilisation, power draw and thermal state need IOReport or
    /// `powermetrics` (root); this crate uses neither, so it does not report them.
    pub fn export_performance_report(&self) -> String {
        fn opt_u32(value: Option<u32>) -> String {
            value
                .map(|v| v.to_string())
                .unwrap_or_else(|| "not reported by sysctl".to_string())
        }
        fn opt_str(value: &Option<String>) -> String {
            value.clone().unwrap_or_else(|| "not available (metal feature off)".to_string())
        }
        fn opt_f32(value: Option<f32>, unit: &str) -> String {
            value
                .map(|v| format!("{v:.2} {unit}"))
                .unwrap_or_else(|| "not available".to_string())
        }

        let caps = &self.device_capabilities;
        let metrics = &self.performance_metrics;
        format!(
            "MLX-style Engine Report ({backend} backend)\n\
             =========================================\n\
             NOTE: this engine implements an MLX-shaped API on Metal/CPU. It does NOT\n\
             link Apple's MLX framework and never dispatches to the Neural Engine.\n\n\
             Probed hardware (sysctlbyname):\n\
             - CPU: {cpu_brand}\n\
             - Performance cores: {perf}\n\
             - Efficiency cores: {eff}\n\
             - Logical cores: {logical}\n\
             - Installed memory: {memory:.1} GiB\n\
             - AMX version: {amx}\n\n\
             Metal device:\n\
             - Name: {metal_name}\n\
             - Apple GPU family: {gpu_family}\n\
             - Max buffer length: {max_buffer}\n\
             - Unified memory: {unified}\n\n\
             Measured execution:\n\
             - Last run: {last_ms:.3} ms over {last_nodes} graph nodes\n\
             - Nodes per second: {ops:.0}\n\
             - Compilation time: {compile_ms:.3} ms\n\
             - Process CPU: {cpu}\n\
             - Process RSS: {rss}\n\
             - Memory pool allocated: {pool:.4} GiB\n\n\
             Not measured (no public API available to this process):\n\
             - GPU utilization, Neural Engine utilization, power draw, thermal state\n\
             - GPU core count, Neural Engine TOPS, memory bandwidth\n\n\
             Configuration:\n\
             - Compilation strategy: {strategy:?}\n\
             - Default precision: {precision:?}\n\
             - Memory pool strategy: {pool_strategy:?}\n\
             - Workload distribution: {distribution:?}\n\
             - Operator fusion: {fusion}\n\
             - Zero-copy operations: {zero_copy}",
            backend = Self::backend_label(),
            cpu_brand = caps.cpu_brand,
            perf = opt_u32(caps.performance_cores),
            eff = opt_u32(caps.efficiency_cores),
            logical = caps.logical_cores,
            memory = caps.unified_memory_gb,
            amx = opt_u32(caps.amx_version),
            metal_name = opt_str(&caps.metal_device_name),
            gpu_family = caps
                .apple_gpu_family
                .map(|v| format!("Apple{v}"))
                .unwrap_or_else(|| "not available".to_string()),
            max_buffer = caps
                .metal_max_buffer_bytes
                .map(|v| format!("{} MiB", v / (1024 * 1024)))
                .unwrap_or_else(|| "not available".to_string()),
            unified = caps
                .metal_unified_memory
                .map(|v| v.to_string())
                .unwrap_or_else(|| "not available".to_string()),
            last_ms = metrics.last_execution_ms,
            last_nodes = metrics.last_execution_nodes,
            ops = metrics.ops_per_second,
            compile_ms = metrics.compilation_time_ms,
            cpu = opt_f32(metrics.cpu_utilization, "%"),
            rss = opt_f32(metrics.process_memory_gb, "GiB"),
            pool = metrics.pool_memory_gb,
            strategy = self.config.compilation_strategy,
            precision = self.config.precision_config.default_precision,
            pool_strategy = self.config.memory_config.pool_strategy,
            distribution = self.config.compute_units.distribution_strategy,
            fusion = self.config.graph_optimization.operator_fusion,
            zero_copy = self.config.memory_config.zero_copy_enabled,
        )
    }
}

/// Read a strictly positive `usize` graph parameter, defaulting when absent.
fn read_positive_usize(
    parameters: &HashMap<String, f32>,
    key: &str,
    default: usize,
) -> Result<usize> {
    match parameters.get(key) {
        None => Ok(default),
        Some(value) => {
            if !value.is_finite() || *value < 1.0 || value.fract() != 0.0 {
                return Err(TrustformersError::runtime_error(format!(
                    "graph parameter '{key}' must be a positive whole number, got {value}"
                ))
                .into());
            }
            Ok(*value as usize)
        },
    }
}

/// Read a non-negative `usize` graph parameter, defaulting when absent.
fn read_non_negative_usize(
    parameters: &HashMap<String, f32>,
    key: &str,
    default: usize,
) -> Result<usize> {
    match parameters.get(key) {
        None => Ok(default),
        Some(value) => {
            if !value.is_finite() || *value < 0.0 || value.fract() != 0.0 {
                return Err(TrustformersError::runtime_error(format!(
                    "graph parameter '{key}' must be a non-negative whole number, got {value}"
                ))
                .into());
            }
            Ok(*value as usize)
        },
    }
}

impl UnifiedMemoryPool {
    fn new(config: UnifiedMemoryConfig) -> Self {
        Self {
            config,
            allocated_memory: HashMap::new(),
            total_allocated_bytes: 0,
            peak_allocated_bytes: 0,
            allocation_count: 0,
        }
    }

    fn allocate(&mut self, tensor_id: TensorId, allocation: MemoryAllocation) -> Result<()> {
        if self.allocated_memory.insert(tensor_id, allocation.clone()).is_none() {
            self.total_allocated_bytes += allocation.size_bytes;
            self.peak_allocated_bytes = self.peak_allocated_bytes.max(self.total_allocated_bytes);
            self.allocation_count += 1;
        }
        Ok(())
    }

    fn get_total_allocated_gb(&self) -> f32 {
        self.total_allocated_bytes as f32 / (1024.0 * 1024.0 * 1024.0)
    }
}

impl Default for MlxPerformanceMetrics {
    /// "Nothing measured yet": counters at zero, unmeasurable quantities at `None`.
    fn default() -> Self {
        Self {
            ops_per_second: 0.0,
            last_execution_ms: 0.0,
            last_execution_nodes: 0,
            cpu_utilization: None,
            process_memory_gb: None,
            pool_memory_gb: 0.0,
            compilation_time_ms: 0.0,
        }
    }
}

#[cfg(test)]
#[path = "mlx_engine_tests.rs"]
mod tests;
