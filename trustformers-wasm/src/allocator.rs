//! Custom allocator for WASM to reduce binary size

// Use dlmalloc as the global allocator for better performance
// Note: wee_alloc was removed due to being unmaintained (RUSTSEC-2022-0054)
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
    #[cfg(all(target_arch = "wasm32", feature = "dlmalloc-alloc"))]
    return "dlmalloc";

    #[cfg(not(target_arch = "wasm32"))]
    return "system";

    #[cfg(all(target_arch = "wasm32", not(feature = "dlmalloc-alloc")))]
    return "default";
}