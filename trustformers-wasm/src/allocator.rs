//! Custom allocator for WASM to reduce binary size
//!
//! This module was previously never `mod`-declared from [`crate`], so the
//! `dlmalloc-alloc` feature (on by default - see `Cargo.toml`) silently did
//! nothing: `#[global_allocator]` only takes effect when the item that
//! defines it is actually compiled as part of the crate. Wiring it in via
//! `pub mod allocator;` in `lib.rs` is what makes the attribute apply.
//!
//! Native targets are deliberately left alone here (no `#[global_allocator]`
//! override): installing a global allocator from a *library* crate on
//! non-wasm32 targets is surprising for downstream consumers and would
//! silently override whatever allocator the final binary chose. `dlmalloc`
//! only ever replaces the allocator on `wasm32`, and only when the
//! `dlmalloc-alloc` feature is enabled; on native, or on wasm32 without the
//! feature, Rust's own default allocator is used exactly as if this module
//! did not exist.

// Use dlmalloc as the global allocator for better performance on WASM.
// Note: wee_alloc was removed due to being unmaintained (RUSTSEC-2022-0054)
#[cfg(all(target_arch = "wasm32", feature = "dlmalloc-alloc"))]
#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

/// Get current allocator type for debugging.
///
/// Compiles (and reports "default") on every target/feature combination -
/// wasm32 with `dlmalloc-alloc`, wasm32 without it, and native - so callers
/// never need their own `cfg` gate around this function.
pub fn get_allocator_type() -> &'static str {
    #[cfg(all(target_arch = "wasm32", feature = "dlmalloc-alloc"))]
    return "dlmalloc";

    #[cfg(not(all(target_arch = "wasm32", feature = "dlmalloc-alloc")))]
    return "default";
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard: before this module was `mod`-declared in `lib.rs`,
    /// nothing here compiled as part of the crate at all, so
    /// `get_allocator_type` was unreachable from any test. Now it must
    /// report a stable, non-empty string on every target.
    #[test]
    fn test_get_allocator_type_is_non_empty() {
        assert!(!get_allocator_type().is_empty());
    }

    /// On native (this test only runs natively - wasm32 tests run under
    /// wasm-bindgen-test, not `cargo test`), no `dlmalloc-alloc` feature
    /// gate ever applies, so the reported type must always be "default"
    /// regardless of which Cargo features are enabled.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_native_allocator_is_always_default() {
        assert_eq!(get_allocator_type(), "default");
    }
}
