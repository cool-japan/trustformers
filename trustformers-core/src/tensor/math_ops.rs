//! Tensor mathematical operations with numerical stability enhancements.
//!
//! This module contains mathematical operations for tensors including
//! basic arithmetic, matrix operations, and advanced mathematical functions.
//! All operations include numerical stability features such as overflow/underflow
//! protection, NaN/infinity detection, and optimized algorithms.
//!
//! The implementation is organized into submodules for better maintainability:
//! - `stability`: Numerical stability utilities and constants
//! - `broadcasting`: Broadcasting utilities for tensor operations
//! - `common`: Common utilities and constants (deprecated, use stability and broadcasting)
//! - `arithmetic`: Basic arithmetic operations (add, sub, mul, div, scalar operations)
//! - `linear_algebra`: Linear algebra operations (matmul, norm, gradient clipping)
//! - `mathematical`: Mathematical functions (sin, cos, tan, sqrt, log, exp, etc.)
//! - `utilities`: Utility functions (clamp, sign, round, etc.)
//! - `statistical`: Statistical operations (mean, std, variance, sum, etc.)
//! - `advanced`: Advanced operations (layer_norm, cross_entropy, etc.)

// Import all submodules
pub mod advanced;
pub mod arithmetic;
pub mod broadcasting;
pub mod common;
pub mod linear_algebra;
pub mod mathematical;
pub mod stability;
pub mod statistical;
pub mod utilities;

// Re-export essential utilities and SIMD functions
