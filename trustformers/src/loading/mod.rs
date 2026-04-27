//! Model weight loading utilities.
//!
//! Provides sequential and parallel weight loading for transformer models,
//! with progress reporting and statistics.

pub mod parallel_loader;

pub use parallel_loader::{
    load_model_parallel, LoadingProgress, LoadingStats, ParallelLoaderConfig,
    ParallelWeightLoader, WeightChunk,
};
