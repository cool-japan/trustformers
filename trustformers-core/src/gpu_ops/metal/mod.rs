//! Auto-generated module structure

// Common imports and utilities used by all modules
mod common;

pub mod bufferid_traits;
pub mod functions;
pub mod metalbackend_accessors;
pub mod metalbackend_attention_gpu_to_gpu_group;
pub mod metalbackend_attention_with_cache_gpu_to_gpu_group;
pub mod metalbackend_buffer_cache_size_group;
pub mod metalbackend_buffer_to_objc2_group;
pub mod metalbackend_clear_buffer_cache_group;
pub mod metalbackend_flash_attention_group;
pub mod metalbackend_gelu_f32_group;
pub mod metalbackend_initialize_mps_group;
pub mod metalbackend_layernorm_f32_group;
pub mod metalbackend_matmul_f32_group;
pub mod metalbackend_matmul_gelu_f32_group;
pub mod metalbackend_new_group;
pub mod metalbackend_remove_persistent_buffer_group;
pub mod metalbackend_rope_f32_group;
pub mod metalbackend_softmax_causal_f32_group;
pub mod metalbackend_type;
pub mod types;

// Re-export all types
pub use bufferid_traits::*;
pub use functions::*;
pub use metalbackend_accessors::*;
pub use metalbackend_attention_gpu_to_gpu_group::*;
pub use metalbackend_attention_with_cache_gpu_to_gpu_group::*;
pub use metalbackend_buffer_cache_size_group::*;
pub use metalbackend_buffer_to_objc2_group::*;
pub use metalbackend_clear_buffer_cache_group::*;
pub use metalbackend_flash_attention_group::*;
pub use metalbackend_gelu_f32_group::*;
pub use metalbackend_initialize_mps_group::*;
pub use metalbackend_layernorm_f32_group::*;
pub use metalbackend_matmul_f32_group::*;
pub use metalbackend_matmul_gelu_f32_group::*;
pub use metalbackend_new_group::*;
pub use metalbackend_remove_persistent_buffer_group::*;
pub use metalbackend_rope_f32_group::*;
pub use metalbackend_softmax_causal_f32_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use metalbackend_type::MetalBackend;
pub use metalbackend_type::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use types::BufferId;
pub use types::*;
