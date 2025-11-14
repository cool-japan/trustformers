//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::common::*;
use super::types::BufferCache;

/// Metal GPU backend for matrix multiplication and element-wise operations
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct MetalBackend {
    pub(super) device: MetalDevice,
    pub(super) command_queue: CommandQueue,
    pub(super) buffer_cache: Arc<std::sync::Mutex<BufferCache>>,
    pub(super) matmul_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) gelu_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) matmul_gelu_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) matmul_bias_gelu_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) scale_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) add_bias_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) layernorm_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) rope_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) softmax_causal_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) copy_with_offset_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) elementwise_add_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) split_qkv_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) transpose_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) reshape_to_heads_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) reshape_from_heads_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) batched_transpose_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) batched_softmax_causal_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) batched_matmul_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) batched_matmul_scaled_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) batched_scaled_matmul_softmax_causal_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) batched_scaled_matmul_softmax_gen_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) concat_seq_dim_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) flash_attention_pipeline: Arc<metal::ComputePipelineState>,
    pub(super) mps_ops: Arc<Option<MPSOperations>>,
}
