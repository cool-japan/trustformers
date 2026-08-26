//! Automatic parallelism strategy selection for distributed training.
//!
//! Chooses a parallelism strategy (data / tensor / sequence / expert / 3D /
//! hybrid) for a model and hardware setup, using rule-based heuristics,
//! cost/ML-based prediction, genetic search, simulated annealing and
//! multi-objective optimization.
//!
//! Split into cohesive submodules: [`config`] (selection/hardware/model
//! configuration), [`strategy_types`] (chosen strategy and its metrics),
//! [`selector`] (the `AutoParallelismSelector` engine) and [`utils`].

pub mod config;
pub mod selector;
pub mod strategy_types;
#[cfg(test)]
mod tests;
pub mod utils;

// Re-export all types
pub use config::*;
pub use selector::*;
pub use strategy_types::*;
pub use utils::*;
