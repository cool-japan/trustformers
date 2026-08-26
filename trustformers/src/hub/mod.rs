//! Hugging Face Hub integration: downloading, caching, and resuming model files.
//!
//! Split into cohesive submodules: [`types`] (options/config/stats data types),
//! [`delta`] (binary delta encoding for incremental updates), [`manager`] (the
//! `DownloadManager` engine, gated on the `hub` feature), [`download`] (free
//! functions for locating and downloading Hub model files) and [`loaders`]
//! (loading configs, weights, and README-derived model cards).

pub mod delta;
pub mod download;
pub mod loaders;
pub mod manager;
#[cfg(test)]
mod tests;
pub mod types;

// Re-export all types
pub use delta::*;
pub use download::*;
pub use loaders::*;
pub use types::*;
// `manager` is entirely `#[cfg(feature = "hub")]`-gated (`DownloadManager` and
// its helpers do not exist at all without the `hub` feature), so the
// re-export must be gated too or it re-exports nothing and warns as unused.
// (Kept last, after the already alphabetically-sorted group above: rustfmt's
// import reordering does not carry a preceding line comment across a move,
// so this one is placed where reordering leaves it undisturbed.)
#[cfg(feature = "hub")]
pub use manager::*;
