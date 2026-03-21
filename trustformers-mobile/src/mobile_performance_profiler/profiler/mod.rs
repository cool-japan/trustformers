//! Main Mobile Performance Profiler
//!
//! Split into submodules for maintainability.

mod profiler_components;
mod profiler_impl;
mod profiler_types;

pub use profiler_components::*;
pub use profiler_impl::*;
pub use profiler_types::*;
