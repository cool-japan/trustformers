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
#[allow(unused_imports)]
pub use bufferid_traits::*;
#[allow(unused_imports)]
pub use metalbackend_accessors::*;
#[allow(unused_imports)]
pub use metalbackend_attention_gpu_to_gpu_group::*;
#[allow(unused_imports)]
pub use metalbackend_attention_with_cache_gpu_to_gpu_group::*;
#[allow(unused_imports)]
pub use metalbackend_buffer_cache_size_group::*;
#[allow(unused_imports)]
pub use metalbackend_buffer_to_objc2_group::*;
#[allow(unused_imports)]
pub use metalbackend_clear_buffer_cache_group::*;
#[allow(unused_imports)]
pub use metalbackend_flash_attention_group::*;
#[allow(unused_imports)]
pub use metalbackend_gelu_f32_group::*;
#[allow(unused_imports)]
pub use metalbackend_initialize_mps_group::*;
#[allow(unused_imports)]
pub use metalbackend_layernorm_f32_group::*;
#[allow(unused_imports)]
pub use metalbackend_matmul_f32_group::*;
#[allow(unused_imports)]
pub use metalbackend_matmul_gelu_f32_group::*;
#[allow(unused_imports)]
pub use metalbackend_new_group::*;
#[allow(unused_imports)]
pub use metalbackend_remove_persistent_buffer_group::*;
#[allow(unused_imports)]
pub use metalbackend_rope_f32_group::*;
#[allow(unused_imports)]
pub use metalbackend_softmax_causal_f32_group::*;
#[allow(unused_imports)]
pub use metalbackend_type::*;
#[allow(unused_imports)]
pub use types::*;

pub use functions::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use metalbackend_type::MetalBackend;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use types::BufferId;
