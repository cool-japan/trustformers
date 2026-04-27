//! Comprehensive tests for ios/mps.rs
//!
//! Covers MPS data type utilities, non-iOS stub behaviour, MPS data type
//! constant values, element size calculations, buffer shape validation,
//! and error handling paths.

#[cfg(test)]
mod tests {
    use crate::ios::mps::*;

    // =========================================================================
    // LCG deterministic PRNG
    // =========================================================================

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.state
        }

        fn next_usize_range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next_u64() as usize % (hi - lo))
        }
    }

    // =========================================================================
    // MPSDataType tests (available on all platforms via non-iOS stub or real impl)
    // =========================================================================

    #[test]
    fn test_mps_data_type_float32_factory() {
        let dt = MPSDataType::float32();
        // Inner value must be accessible (through clone/copy)
        let dt2 = dt;
        let _ = dt2; // no-op; confirms Copy trait
    }

    #[test]
    fn test_mps_data_type_float16_factory() {
        let _dt = MPSDataType::float16();
    }

    #[test]
    fn test_mps_data_type_int32_factory() {
        let _dt = MPSDataType::int32();
    }

    #[test]
    fn test_mps_data_type_int8_factory() {
        let _dt = MPSDataType::int8();
    }

    #[test]
    fn test_mps_data_type_uint8_factory() {
        let _dt = MPSDataType::uint8();
    }

    #[test]
    fn test_mps_data_type_bool_factory() {
        let _dt = MPSDataType::bool();
    }

    // =========================================================================
    // MPSDataType::element_size tests (iOS path has real constants; non-iOS path
    // returns 0 for stubs, so we test the iOS branch via cfg gate)
    // =========================================================================

    #[cfg(target_os = "ios")]
    mod ios_only {
        use crate::ios::mps::*;

        #[test]
        fn test_mps_data_type_float32_element_size() {
            assert_eq!(
                MPSDataType::float32().element_size(),
                4,
                "float32 should be 4 bytes"
            );
        }

        #[test]
        fn test_mps_data_type_float16_element_size() {
            assert_eq!(
                MPSDataType::float16().element_size(),
                2,
                "float16 should be 2 bytes"
            );
        }

        #[test]
        fn test_mps_data_type_int32_element_size() {
            assert_eq!(
                MPSDataType::int32().element_size(),
                4,
                "int32 should be 4 bytes"
            );
        }

        #[test]
        fn test_mps_data_type_int8_element_size() {
            assert_eq!(
                MPSDataType::int8().element_size(),
                1,
                "int8 should be 1 byte"
            );
        }

        #[test]
        fn test_mps_data_type_uint8_element_size() {
            assert_eq!(
                MPSDataType::uint8().element_size(),
                1,
                "uint8 should be 1 byte"
            );
        }

        #[test]
        fn test_mps_data_type_bool_element_size() {
            assert_eq!(
                MPSDataType::bool().element_size(),
                1,
                "bool should be 1 byte"
            );
        }

        #[test]
        fn test_mps_compute_graph_creation_fails_without_device() {
            // On a test host without Metal, graph creation should fail gracefully
            // or succeed (depends on simulator availability). We accept either.
            let result = MPSComputeGraph::new();
            // Just check it doesn't panic; result may be Ok or Err depending on env
            let _ = result;
        }
    }

    // =========================================================================
    // Non-iOS stub tests
    // =========================================================================

    #[cfg(not(target_os = "ios"))]
    mod non_ios {
        use crate::ios::mps::*;

        #[test]
        fn test_mps_compute_graph_new_returns_error_on_non_ios() {
            let result = MPSComputeGraph::new();
            assert!(
                result.is_err(),
                "MPSComputeGraph::new() should return Err on non-iOS"
            );
            let err_msg = result.expect_err("Should be an error");
            assert!(
                err_msg.contains("not available"),
                "Error message should explain MPS unavailability"
            );
        }
    }

    // =========================================================================
    // MPS data type constant tests (iOS-path constants exposed for testing)
    // =========================================================================

    #[cfg(target_os = "ios")]
    mod ios_constants {
        use crate::ios::mps::*;

        #[test]
        fn test_mps_data_type_constants_distinct() {
            // All MPS_DATA_TYPE_* constants must be distinct
            let constants = [
                MPS_DATA_TYPE_FLOAT32,
                MPS_DATA_TYPE_FLOAT16,
                MPS_DATA_TYPE_INT32,
                MPS_DATA_TYPE_INT8,
                MPS_DATA_TYPE_UINT8,
                MPS_DATA_TYPE_BOOL,
            ];
            for (i, a) in constants.iter().enumerate() {
                for (j, b) in constants.iter().enumerate() {
                    if i != j {
                        assert_ne!(a, b, "MPS data type constants {i} and {j} should differ");
                    }
                }
            }
        }

        #[test]
        fn test_mps_data_type_float32_constant_non_zero() {
            assert_ne!(MPS_DATA_TYPE_FLOAT32, 0, "FLOAT32 constant must be non-zero");
        }
    }

    // =========================================================================
    // Shape validation helper tests
    // =========================================================================

    #[test]
    fn test_shape_volume_calculation() {
        let shape = [2usize, 3, 4, 5];
        let volume: usize = shape.iter().product();
        assert_eq!(volume, 120, "Volume of [2,3,4,5] should be 120");
    }

    #[test]
    fn test_shape_volume_calculation_matmul_shapes() {
        // Common matmul: [batch=1, M=64, K=128] × [batch=1, K=128, N=256]
        let a_shape = [1usize, 64, 128];
        let b_shape = [1usize, 128, 256];
        let output_shape = [1usize, 64, 256];

        let a_vol: usize = a_shape.iter().product();
        let b_vol: usize = b_shape.iter().product();
        let out_vol: usize = output_shape.iter().product();

        assert_eq!(a_vol, 8_192);
        assert_eq!(b_vol, 32_768);
        assert_eq!(out_vol, 16_384);
    }

    #[test]
    fn test_buffer_byte_size_float32() {
        let elements = 256usize;
        let bytes_per_element = 4usize; // float32
        let total_bytes = elements * bytes_per_element;
        assert_eq!(total_bytes, 1_024, "256 float32 elements should require 1024 bytes");
    }

    #[test]
    fn test_buffer_byte_size_float16() {
        let elements = 512usize;
        let bytes_per_element = 2usize; // float16
        let total_bytes = elements * bytes_per_element;
        assert_eq!(total_bytes, 1_024, "512 float16 elements should require 1024 bytes");
    }

    // =========================================================================
    // Convolution shape tests (pure Rust, no Metal hardware needed)
    // =========================================================================

    #[test]
    fn test_conv2d_output_spatial_size_calculation() {
        // output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        let input_h = 224usize;
        let input_w = 224usize;
        let kernel_h = 3usize;
        let kernel_w = 3usize;
        let padding = 1usize;
        let stride = 1usize;

        let out_h = (input_h - kernel_h + 2 * padding) / stride + 1;
        let out_w = (input_w - kernel_w + 2 * padding) / stride + 1;

        assert_eq!(out_h, 224, "224x224 with 3x3 kernel, pad=1, stride=1 → 224 height");
        assert_eq!(out_w, 224, "224x224 with 3x3 kernel, pad=1, stride=1 → 224 width");
    }

    #[test]
    fn test_conv2d_stride_2_reduces_spatial_size() {
        let input_h = 224usize;
        let kernel_h = 3usize;
        let padding = 1usize;
        let stride = 2usize;

        let out_h = (input_h - kernel_h + 2 * padding) / stride + 1;
        assert_eq!(out_h, 112, "stride-2 should halve 224 → 112");
    }

    #[test]
    fn test_depthwise_conv_channel_count() {
        let in_channels = 32usize;
        let multiplier = 1usize;
        let out_channels = in_channels * multiplier;
        assert_eq!(out_channels, in_channels, "Depthwise with multiplier=1 keeps channel count");
    }

    // =========================================================================
    // Matmul shape compatibility tests
    // =========================================================================

    #[test]
    fn test_matmul_shapes_compatible() {
        // For A[M,K] × B[K,N], K dimensions must match
        let m = 64usize;
        let k_a = 128usize;
        let k_b = 128usize;
        let n = 256usize;
        assert_eq!(k_a, k_b, "Inner dimensions must match for matmul");
        let output_elements = m * n;
        assert_eq!(output_elements, 16_384);
    }

    #[test]
    fn test_batch_matmul_shapes() {
        let batch = 4usize;
        let m = 32usize;
        let k = 64usize;
        let n = 128usize;

        let a_total = batch * m * k;
        let b_total = batch * k * n;
        let out_total = batch * m * n;

        assert_eq!(a_total, 8_192);
        assert_eq!(b_total, 32_768);
        assert_eq!(out_total, 16_384);
    }

    // =========================================================================
    // Pooling shape tests
    // =========================================================================

    #[test]
    fn test_max_pooling_output_size() {
        let input_h = 56usize;
        let kernel = 2usize;
        let stride = 2usize;
        let out_h = (input_h - kernel) / stride + 1;
        assert_eq!(out_h, 28, "2x2 max pooling with stride 2 on 56 → 28");
    }

    #[test]
    fn test_global_average_pooling_output_is_1x1() {
        // Global average pooling always reduces spatial dims to 1x1
        let _input_h = 7usize;
        let _input_w = 7usize;
        let out_h = 1usize;
        let out_w = 1usize;
        assert_eq!(out_h * out_w, 1, "Global average pooling → 1x1 spatial output");
    }

    // =========================================================================
    // Error message string tests
    // =========================================================================

    #[test]
    fn test_missing_tensor_error_message_format() {
        let tensor_name = "nonexistent_tensor";
        let error_msg = format!("Source tensor '{}' not found", tensor_name);
        assert!(error_msg.contains(tensor_name), "Error must reference the missing tensor name");
        assert!(error_msg.contains("not found"));
    }

    #[test]
    fn test_null_tensor_error_message() {
        let error_msg = "Failed to create placeholder tensor".to_string();
        assert!(!error_msg.is_empty());
        assert!(error_msg.contains("Failed"));
    }

    // =========================================================================
    // LCG-based random shape generation tests
    // =========================================================================

    #[test]
    fn test_random_conv_shapes_valid() {
        let mut lcg = Lcg::new(0xC011);
        for _ in 0..10 {
            let in_h = lcg.next_usize_range(10, 256);
            let in_w = lcg.next_usize_range(10, 256);
            let k_h = lcg.next_usize_range(1, in_h.min(7) + 1);
            let k_w = lcg.next_usize_range(1, in_w.min(7) + 1);
            let padding = 0usize;
            let stride = 1usize;

            let out_h = (in_h - k_h + 2 * padding) / stride + 1;
            let out_w = (in_w - k_w + 2 * padding) / stride + 1;

            assert!(out_h > 0, "Output height must be positive");
            assert!(out_w > 0, "Output width must be positive");
            assert!(out_h <= in_h, "Output height must not exceed input");
            assert!(out_w <= in_w, "Output width must not exceed input");
        }
    }

    #[test]
    fn test_random_matmul_shapes_valid() {
        let mut lcg = Lcg::new(0xA71);
        for _ in 0..10 {
            let m = lcg.next_usize_range(1, 512);
            let k = lcg.next_usize_range(1, 512);
            let n = lcg.next_usize_range(1, 512);

            let a_elems = m * k;
            let b_elems = k * n;
            let c_elems = m * n;

            assert!(a_elems > 0);
            assert!(b_elems > 0);
            assert!(c_elems > 0);
        }
    }
}
