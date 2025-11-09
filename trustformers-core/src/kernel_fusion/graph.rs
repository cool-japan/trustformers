//! Computation graph structures for kernel fusion analysis
//!
//! This module provides data structures for representing computation graphs,
//! nodes, tensor information, and related metadata used in fusion analysis.

use crate::kernel_fusion::operation_types::OperationType;
use std::collections::HashMap;

/// Computation graph representation for fusion analysis
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    pub nodes: HashMap<String, GraphNode>,
    pub edges: HashMap<String, Vec<String>>, // node_id -> dependencies
    pub execution_order: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: String,
    pub operation: OperationType,
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
    pub metadata: NodeMetadata,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub device: Device,
    pub memory_layout: MemoryLayout,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    F32,
    F16,
    BF16,
    I32,
    I8,
    U8,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    GPU(u32), // GPU device ID
    ASIC(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Blocked(Vec<usize>),
    /// Cache-optimized tiled layout for better spatial locality
    Tiled {
        tile_sizes: Vec<usize>,
    },
    /// NCHW format commonly used in computer vision
    NCHW,
    /// NHWC format for better memory coalescing on some devices
    NHWC,
    /// Packed format for quantized tensors
    Packed {
        elements_per_pack: usize,
    },
    /// Strided layout with custom strides
    Strided {
        strides: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
pub struct NodeMetadata {
    pub estimated_ops: u64,
    pub estimated_memory: usize,
    pub is_fusible: bool,
    pub fusion_priority: f64,
    pub execution_time_ns: Option<u64>,
}

impl ComputationGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            execution_order: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: GraphNode) {
        let node_id = node.id.clone();
        self.nodes.insert(node_id.clone(), node);
        self.edges.entry(node_id).or_default();
    }

    pub fn add_edge(&mut self, from: &str, to: &str) {
        self.edges.entry(to.to_string()).or_default().push(from.to_string());
    }

    pub fn get_node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    pub fn get_dependencies(&self, id: &str) -> Option<&Vec<String>> {
        self.edges.get(id)
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphNode {
    pub fn new(id: String, operation: OperationType) -> Self {
        Self {
            id,
            operation,
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: NodeMetadata::default(),
        }
    }
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self {
            estimated_ops: 0,
            estimated_memory: 0,
            is_fusible: true,
            fusion_priority: 1.0,
            execution_time_ns: None,
        }
    }
}

impl TensorInfo {
    pub fn new(shape: Vec<usize>, dtype: DataType, device: Device) -> Self {
        Self {
            shape,
            dtype,
            device,
            memory_layout: MemoryLayout::RowMajor,
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn memory_size(&self) -> usize {
        self.element_count() * self.dtype.size_bytes()
    }
}

impl DataType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 => 2,
            DataType::BF16 => 2,
            DataType::I32 => 4,
            DataType::I8 => 1,
            DataType::U8 => 1,
            DataType::Bool => 1,
        }
    }
}
