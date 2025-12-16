//! Auto-generated module structure

#[cfg(all(target_os = "macos", feature = "metal"))]
mod common;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod bufferid_traits;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod functions;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_accessors;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_attention_gpu_to_gpu_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_attention_with_cache_gpu_to_gpu_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_buffer_cache_size_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_buffer_to_objc2_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_clear_buffer_cache_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_flash_attention_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_gelu_f32_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_initialize_mps_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_layernorm_f32_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_matmul_f32_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_matmul_gelu_f32_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_new_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_remove_persistent_buffer_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_rope_f32_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_softmax_causal_f32_group;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metalbackend_type;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod types;

// Re-export all types (only on macOS with metal feature)
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use bufferid_traits::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_accessors::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_attention_gpu_to_gpu_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_attention_with_cache_gpu_to_gpu_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_buffer_cache_size_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_buffer_to_objc2_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_clear_buffer_cache_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_flash_attention_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_gelu_f32_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_initialize_mps_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_layernorm_f32_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_matmul_f32_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_matmul_gelu_f32_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_new_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_remove_persistent_buffer_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_rope_f32_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_softmax_causal_f32_group::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use metalbackend_type::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
pub use types::*;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use functions::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use metalbackend_type::MetalBackend;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use types::BufferId;
