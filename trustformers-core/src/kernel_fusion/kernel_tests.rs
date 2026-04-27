/// Tests for FusedKernel and KernelImplementation
#[cfg(test)]
mod tests {
    use crate::kernel_fusion::graph::{DataType, Device, TensorInfo};
    use crate::kernel_fusion::kernel::{FusedKernel, KernelImplementation};
    use crate::kernel_fusion::operation_types::{FusionPattern, OperationType};

    fn make_simple_pattern() -> FusionPattern {
        FusionPattern::ElementWiseChain(vec![OperationType::ReLU, OperationType::Add])
    }

    fn make_tensor_info(shape: Vec<usize>) -> TensorInfo {
        TensorInfo::new(shape, DataType::F32, Device::CPU)
    }

    // ---- FusedKernel tests ----

    #[test]
    fn test_fused_kernel_new_defaults() {
        let pattern = make_simple_pattern();
        let kernel = FusedKernel::new(
            "k0".to_string(),
            "relu_add".to_string(),
            pattern,
            vec!["op_0".to_string(), "op_1".to_string()],
        );
        assert_eq!(kernel.id, "k0");
        assert_eq!(kernel.name, "relu_add");
        assert_eq!(kernel.operations.len(), 2);
        assert!(kernel.inputs.is_empty());
        assert!(kernel.outputs.is_empty());
        assert!((kernel.estimated_speedup - 1.0).abs() < 1e-10);
        assert_eq!(kernel.memory_savings, 0);
    }

    #[test]
    fn test_fused_kernel_with_inputs() {
        let pattern = make_simple_pattern();
        let kernel = FusedKernel::new("k1".to_string(), "test".to_string(), pattern, vec![])
            .with_inputs(vec![make_tensor_info(vec![4, 8])]);
        assert_eq!(kernel.inputs.len(), 1);
        assert_eq!(kernel.inputs[0].shape, vec![4, 8]);
    }

    #[test]
    fn test_fused_kernel_with_outputs() {
        let pattern = make_simple_pattern();
        let kernel = FusedKernel::new("k2".to_string(), "test".to_string(), pattern, vec![])
            .with_outputs(vec![make_tensor_info(vec![2, 2])]);
        assert_eq!(kernel.outputs.len(), 1);
    }

    #[test]
    fn test_fused_kernel_with_speedup() {
        let pattern = make_simple_pattern();
        let kernel = FusedKernel::new("k3".to_string(), "test".to_string(), pattern, vec![])
            .with_speedup(3.5);
        assert!((kernel.estimated_speedup - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_fused_kernel_with_memory_savings() {
        let pattern = make_simple_pattern();
        let kernel = FusedKernel::new("k4".to_string(), "test".to_string(), pattern, vec![])
            .with_memory_savings(1024);
        assert_eq!(kernel.memory_savings, 1024);
    }

    #[test]
    fn test_fused_kernel_with_implementation_cpu() {
        let pattern = make_simple_pattern();
        let kernel = FusedKernel::new("k5".to_string(), "test".to_string(), pattern, vec![])
            .with_implementation(KernelImplementation::CPU("void main() {}".to_string()));
        assert_eq!(kernel.implementation.platform(), "CPU");
        assert_eq!(kernel.implementation.code(), "void main() {}");
    }

    #[test]
    fn test_fused_kernel_with_implementation_cuda() {
        let pattern = make_simple_pattern();
        let kernel = FusedKernel::new("k6".to_string(), "test".to_string(), pattern, vec![])
            .with_implementation(KernelImplementation::CUDA(
                "__global__ void f() {}".to_string(),
            ));
        assert_eq!(kernel.implementation.platform(), "CUDA");
    }

    #[test]
    fn test_fused_kernel_chain_builder() {
        let pattern = FusionPattern::LinearActivation {
            matmul: OperationType::MatMul,
            bias_add: true,
            activation: Some(OperationType::ReLU),
        };
        let kernel = FusedKernel::new(
            "k7".to_string(),
            "linear_relu".to_string(),
            pattern,
            vec![
                "matmul".to_string(),
                "bias_add".to_string(),
                "relu".to_string(),
            ],
        )
        .with_inputs(vec![
            make_tensor_info(vec![32, 512]),
            make_tensor_info(vec![512, 2048]),
        ])
        .with_outputs(vec![make_tensor_info(vec![32, 2048])])
        .with_speedup(2.0)
        .with_memory_savings(4096);

        assert_eq!(kernel.operations.len(), 3);
        assert_eq!(kernel.inputs.len(), 2);
        assert_eq!(kernel.outputs.len(), 1);
        assert!((kernel.estimated_speedup - 2.0).abs() < 1e-10);
        assert_eq!(kernel.memory_savings, 4096);
    }

    // ---- KernelImplementation tests ----

    #[test]
    fn test_kernel_impl_platform_names() {
        let impls = vec![
            (KernelImplementation::CUDA("".to_string()), "CUDA"),
            (KernelImplementation::ROCm("".to_string()), "ROCm"),
            (KernelImplementation::OpenCL("".to_string()), "OpenCL"),
            (KernelImplementation::CPU("".to_string()), "CPU"),
            (KernelImplementation::Vulkan("".to_string()), "Vulkan"),
            (KernelImplementation::Metal("".to_string()), "Metal"),
            (KernelImplementation::WebGPU("".to_string()), "WebGPU"),
            (KernelImplementation::SIMD("".to_string()), "SIMD"),
            (KernelImplementation::ASIC("".to_string()), "ASIC"),
        ];
        for (impl_, expected_platform) in impls {
            assert_eq!(impl_.platform(), expected_platform);
        }
    }

    #[test]
    fn test_kernel_impl_code_retrieval() {
        let code = "kernel void myKernel() {}";
        let impls = vec![
            KernelImplementation::Metal(code.to_string()),
            KernelImplementation::SIMD(code.to_string()),
        ];
        for impl_ in impls {
            assert_eq!(impl_.code(), code);
        }
    }

    #[test]
    fn test_kernel_impl_empty_code() {
        let impl_ = KernelImplementation::CPU("".to_string());
        assert_eq!(impl_.code(), "");
    }

    // ---- FusionPattern tests ----

    #[test]
    fn test_fusion_pattern_element_wise_chain() {
        let pattern = FusionPattern::ElementWiseChain(vec![
            OperationType::ReLU,
            OperationType::Multiply,
            OperationType::Add,
        ]);
        if let FusionPattern::ElementWiseChain(ops) = pattern {
            assert_eq!(ops.len(), 3);
        } else {
            panic!("Expected ElementWiseChain");
        }
    }

    #[test]
    fn test_fusion_pattern_attention_fusion() {
        let pattern = FusionPattern::AttentionFusion {
            query_key_matmul: true,
            softmax: true,
            value_matmul: true,
            dropout: false,
        };
        if let FusionPattern::AttentionFusion {
            query_key_matmul,
            softmax,
            value_matmul,
            dropout,
        } = pattern
        {
            assert!(query_key_matmul);
            assert!(softmax);
            assert!(value_matmul);
            assert!(!dropout);
        } else {
            panic!("Expected AttentionFusion");
        }
    }

    #[test]
    fn test_fusion_pattern_flash_attention() {
        let pattern = FusionPattern::FlashAttentionOptimized {
            query_key_matmul: true,
            scaled_softmax: true,
            value_matmul: true,
            causal_mask: true,
            dropout: false,
            block_size: 64,
        };
        if let FusionPattern::FlashAttentionOptimized { block_size, .. } = pattern {
            assert_eq!(block_size, 64);
        } else {
            panic!("Expected FlashAttentionOptimized");
        }
    }

    #[test]
    fn test_fusion_pattern_swiglu() {
        let pattern = FusionPattern::SwiGLU {
            gate_projection: true,
            up_projection: true,
            swish_activation: true,
            element_wise_multiply: true,
        };
        if let FusionPattern::SwiGLU {
            gate_projection,
            up_projection,
            swish_activation,
            element_wise_multiply,
        } = pattern
        {
            assert!(gate_projection && up_projection && swish_activation && element_wise_multiply);
        } else {
            panic!("Expected SwiGLU");
        }
    }

    #[test]
    fn test_fusion_pattern_rope_fusion() {
        let pattern = FusionPattern::RoPEFusion {
            apply_rope: true,
            cos_sin_cached: true,
            dimensions: 128,
        };
        if let FusionPattern::RoPEFusion { dimensions, .. } = pattern {
            assert_eq!(dimensions, 128);
        } else {
            panic!("Expected RoPEFusion");
        }
    }

    #[test]
    fn test_fused_kernel_lcg_generated_operations() {
        let mut s = 42u64;
        let mut ops = Vec::new();
        for _ in 0..5 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ops.push(format!("op_{}", s % 100));
        }
        let pattern = make_simple_pattern();
        let kernel = FusedKernel::new(
            "k_lcg".to_string(),
            "test".to_string(),
            pattern,
            ops.clone(),
        );
        assert_eq!(kernel.operations.len(), 5);
    }
}
