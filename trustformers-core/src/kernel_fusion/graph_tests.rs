/// Tests for kernel fusion graph types
#[cfg(test)]
mod tests {
    use crate::kernel_fusion::graph::{
        ComputationGraph, DataType, Device, GraphNode, MemoryLayout, NodeMetadata, TensorInfo,
    };
    use crate::kernel_fusion::operation_types::OperationType;

    // ---- ComputationGraph tests ----

    #[test]
    fn test_computation_graph_new_is_empty() {
        let graph = ComputationGraph::new();
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
        assert!(graph.execution_order.is_empty());
    }

    #[test]
    fn test_computation_graph_default_matches_new() {
        let a = ComputationGraph::new();
        let b = ComputationGraph::default();
        assert_eq!(a.nodes.len(), b.nodes.len());
    }

    #[test]
    fn test_computation_graph_add_node() {
        let mut graph = ComputationGraph::new();
        let node = GraphNode::new("node_0".to_string(), OperationType::ReLU);
        graph.add_node(node);
        assert_eq!(graph.nodes.len(), 1);
        assert!(graph.nodes.contains_key("node_0"));
    }

    #[test]
    fn test_computation_graph_add_multiple_nodes() {
        let mut graph = ComputationGraph::new();
        for i in 0..5 {
            let node = GraphNode::new(format!("node_{}", i), OperationType::Add);
            graph.add_node(node);
        }
        assert_eq!(graph.nodes.len(), 5);
    }

    #[test]
    fn test_computation_graph_get_node_exists() {
        let mut graph = ComputationGraph::new();
        let node = GraphNode::new("relu_1".to_string(), OperationType::ReLU);
        graph.add_node(node);
        let retrieved = graph.get_node("relu_1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.map(|n| n.id.as_str()).unwrap_or(""), "relu_1");
    }

    #[test]
    fn test_computation_graph_get_node_missing_returns_none() {
        let graph = ComputationGraph::new();
        assert!(graph.get_node("nonexistent").is_none());
    }

    #[test]
    fn test_computation_graph_add_edge() {
        let mut graph = ComputationGraph::new();
        graph.add_edge("A", "B");
        let deps = graph.get_dependencies("B");
        assert!(deps.is_some());
        let empty = vec![];
        let deps = deps.unwrap_or(&empty);
        assert!(deps.contains(&"A".to_string()));
    }

    #[test]
    fn test_computation_graph_get_dependencies_multiple() {
        let mut graph = ComputationGraph::new();
        graph.add_edge("A", "C");
        graph.add_edge("B", "C");
        let deps: Vec<String> = graph.get_dependencies("C").cloned().unwrap_or_default();
        assert!(deps.contains(&"A".to_string()));
        assert!(deps.contains(&"B".to_string()));
    }

    #[test]
    fn test_computation_graph_get_dependencies_missing_returns_none() {
        let graph = ComputationGraph::new();
        assert!(graph.get_dependencies("no_node").is_none());
    }

    // ---- GraphNode tests ----

    #[test]
    fn test_graph_node_new_empty_inputs_outputs() {
        let node = GraphNode::new("gemm_0".to_string(), OperationType::MatMul);
        assert_eq!(node.id, "gemm_0");
        assert!(node.inputs.is_empty());
        assert!(node.outputs.is_empty());
    }

    #[test]
    fn test_graph_node_operation_types() {
        let ops = vec![
            OperationType::Add,
            OperationType::Subtract,
            OperationType::Multiply,
            OperationType::GELU,
            OperationType::Softmax,
            OperationType::LayerNorm,
            OperationType::RoPE,
            OperationType::MatMul,
        ];
        for (i, op) in ops.into_iter().enumerate() {
            let node = GraphNode::new(format!("node_{}", i), op.clone());
            assert_eq!(&node.operation, &op);
        }
    }

    #[test]
    fn test_node_metadata_default_is_fusible() {
        let meta = NodeMetadata::default();
        assert!(meta.is_fusible);
        assert_eq!(meta.estimated_ops, 0);
        assert_eq!(meta.estimated_memory, 0);
        assert!(meta.execution_time_ns.is_none());
    }

    #[test]
    fn test_node_metadata_fusion_priority_default_is_one() {
        let meta = NodeMetadata::default();
        assert!((meta.fusion_priority - 1.0).abs() < 1e-10);
    }

    // ---- TensorInfo tests ----

    #[test]
    fn test_tensor_info_new_row_major() {
        let info = TensorInfo::new(vec![4, 8], DataType::F32, Device::CPU);
        assert_eq!(info.shape, vec![4, 8]);
        assert_eq!(info.dtype, DataType::F32);
        assert_eq!(info.device, Device::CPU);
        assert_eq!(info.memory_layout, MemoryLayout::RowMajor);
    }

    #[test]
    fn test_tensor_info_element_count() {
        let info = TensorInfo::new(vec![2, 3, 4], DataType::F32, Device::CPU);
        assert_eq!(info.element_count(), 24);
    }

    #[test]
    fn test_tensor_info_element_count_scalar() {
        let info = TensorInfo::new(vec![1], DataType::I8, Device::CPU);
        assert_eq!(info.element_count(), 1);
    }

    #[test]
    fn test_tensor_info_memory_size_f32() {
        let info = TensorInfo::new(vec![4, 4], DataType::F32, Device::CPU);
        assert_eq!(info.memory_size(), 64); // 16 * 4 bytes
    }

    #[test]
    fn test_tensor_info_memory_size_f16() {
        let info = TensorInfo::new(vec![8], DataType::F16, Device::CPU);
        assert_eq!(info.memory_size(), 16); // 8 * 2 bytes
    }

    #[test]
    fn test_tensor_info_memory_size_i8() {
        let info = TensorInfo::new(vec![10], DataType::I8, Device::CPU);
        assert_eq!(info.memory_size(), 10); // 10 * 1 byte
    }

    #[test]
    fn test_tensor_info_memory_size_bool() {
        let info = TensorInfo::new(vec![100], DataType::Bool, Device::CPU);
        assert_eq!(info.memory_size(), 100);
    }

    // ---- DataType tests ----

    #[test]
    fn test_data_type_size_bytes() {
        assert_eq!(DataType::F32.size_bytes(), 4);
        assert_eq!(DataType::F16.size_bytes(), 2);
        assert_eq!(DataType::BF16.size_bytes(), 2);
        assert_eq!(DataType::I32.size_bytes(), 4);
        assert_eq!(DataType::I8.size_bytes(), 1);
        assert_eq!(DataType::U8.size_bytes(), 1);
        assert_eq!(DataType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_data_type_equality() {
        assert_eq!(DataType::F32, DataType::F32);
        assert_ne!(DataType::F32, DataType::F16);
    }

    // ---- Device tests ----

    #[test]
    fn test_device_cpu_equality() {
        assert_eq!(Device::CPU, Device::CPU);
    }

    #[test]
    fn test_device_gpu_equality() {
        assert_eq!(Device::GPU(0), Device::GPU(0));
        assert_ne!(Device::GPU(0), Device::GPU(1));
    }

    #[test]
    fn test_device_asic() {
        let asic = Device::ASIC("custom_npu".to_string());
        if let Device::ASIC(name) = &asic {
            assert_eq!(name, "custom_npu");
        } else {
            panic!("Expected ASIC device");
        }
    }

    // ---- MemoryLayout tests ----

    #[test]
    fn test_memory_layout_row_major_equality() {
        assert_eq!(MemoryLayout::RowMajor, MemoryLayout::RowMajor);
        assert_ne!(MemoryLayout::RowMajor, MemoryLayout::ColumnMajor);
    }

    #[test]
    fn test_memory_layout_tiled() {
        let layout = MemoryLayout::Tiled {
            tile_sizes: vec![16, 16],
        };
        if let MemoryLayout::Tiled { tile_sizes } = layout {
            assert_eq!(tile_sizes, vec![16, 16]);
        } else {
            panic!("Expected Tiled layout");
        }
    }

    #[test]
    fn test_memory_layout_packed() {
        let layout = MemoryLayout::Packed {
            elements_per_pack: 8,
        };
        if let MemoryLayout::Packed { elements_per_pack } = layout {
            assert_eq!(elements_per_pack, 8);
        } else {
            panic!("Expected Packed layout");
        }
    }

    // ---- OperationType tests ----

    #[test]
    fn test_operation_type_custom() {
        let op = OperationType::Custom("my_fused_op".to_string());
        if let OperationType::Custom(name) = &op {
            assert_eq!(name, "my_fused_op");
        } else {
            panic!("Expected Custom operation type");
        }
    }

    #[test]
    fn test_operation_type_hash_in_hashmap() {
        use std::collections::HashMap;
        let mut map: HashMap<OperationType, usize> = HashMap::new();
        map.insert(OperationType::ReLU, 1);
        map.insert(OperationType::GELU, 2);
        map.insert(OperationType::MatMul, 3);
        assert_eq!(map.get(&OperationType::ReLU), Some(&1));
        assert_eq!(map.get(&OperationType::GELU), Some(&2));
    }
}
