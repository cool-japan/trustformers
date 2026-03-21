//! # QuantumDebugConfig - Trait Implementations
//!
//! This module contains trait implementations for `QuantumDebugConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for QuantumDebugConfig {
    fn default() -> Self {
        Self {
            num_qubits: 16,
            enable_superposition_analysis: true,
            enable_entanglement_detection: true,
            enable_interference_analysis: true,
            measurement_sampling_rate: 0.1,
            enable_error_correction: true,
            enable_vqe_analysis: true,
            enable_qaoa_analysis: true,
            enable_noise_modeling: true,
            enable_hybrid_debugging: true,
            max_circuit_depth: 100,
            noise_level: 0.01,
            enable_quantum_benchmarking: true,
            enable_feature_map_analysis: true,
        }
    }
}
