//! Custom allocator for WASM to reduce binary size

// Use wee_alloc as the global allocator for smaller binary size
// This saves ~10KB compared to the default allocator
#[cfg(all(target_arch = "wasm32", feature = "wee-alloc"))]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// Alternative: dlmalloc for better performance with slightly larger size
#[cfg(all(target_arch = "wasm32", feature = "dlmalloc-alloc"))]
#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

// For non-WASM targets, use the default allocator
#[cfg(not(target_arch = "wasm32"))]
use std::alloc::System;

#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static ALLOC: System = System;

/// Get current allocator type for debugging
pub fn get_allocator_type() -> &'static str {
    #[cfg(all(target_arch = "wasm32", feature = "wee-alloc"))]
    return "wee_alloc";

    #[cfg(all(target_arch = "wasm32", feature = "dlmalloc-alloc"))]
    return "dlmalloc";

    #[cfg(not(target_arch = "wasm32"))]
    return "system";

    #[cfg(all(target_arch = "wasm32", not(feature = "wee-alloc"), not(feature = "dlmalloc-alloc")))]
    return "default";
}