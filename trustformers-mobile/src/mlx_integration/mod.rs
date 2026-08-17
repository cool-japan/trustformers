//! MLX-style graph execution for Apple Silicon, implemented on Metal and CPU.
//!
//! **This module does not link Apple's MLX framework.** See [`mlx_types`] for the
//! full statement. The `Mlx*` type names are retained for source compatibility and
//! mean "MLX-style API", not "MLX-backed".

pub mod device_probe;
mod mlx_engine;
mod mlx_types;

pub use device_probe::{probe_hardware, ProbedHardware};
pub use mlx_engine::*;
pub use mlx_types::*;
