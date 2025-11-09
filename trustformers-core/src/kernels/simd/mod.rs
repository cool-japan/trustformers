// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized operations for performance-critical kernels
//!
//! This module provides SIMD implementations for various neural network operations
//! including layer normalization, softmax, and matrix operations, optimized for
//! different instruction sets (AVX-512, AVX2, NEON, RISC-V Vector).

pub mod cpu_features;
pub mod layer_norm;
pub mod matrix_ops;
pub mod softmax;

pub use cpu_features::CpuFeatures;
pub use layer_norm::SIMDLayerNorm;
pub use matrix_ops::SIMDMatrixOps;
pub use softmax::SIMDSoftmax;
