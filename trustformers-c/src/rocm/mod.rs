//! ROCm backend for TrustformeRS-C
//!
//! This module provides real ROCm GPU acceleration for tensor operations and model inference
//! using AMD GPUs. It leverages HIP for device management and memory operations, and ROCblas
//! for optimized linear algebra operations.
//!
//! The module is organized into several sub-modules for better maintainability:
//! - `types`: Type definitions for ROCm devices, tensors, and configurations
//! - `manager`: Device management and context handling
//! - `operations`: Tensor operations and memory management
//! - `c_api`: C Foreign Function Interface exports

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

// Sub-modules
pub mod c_api;
pub mod manager;
pub mod operations;
pub mod types;

// Re-export public types and functions
pub use operations::RocmOperations;
pub use types::*;

// C API functions are automatically exported via #[no_mangle]
pub use c_api::*;

/// Global ROCm context manager
static ROCM_MANAGER: Lazy<Mutex<RocmManager>> = Lazy::new(|| Mutex::new(RocmManager::new()));

/// Global tensor registry for C API
static ROCM_TENSOR_REGISTRY: Lazy<Mutex<HashMap<usize, RocmTensor>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Global counter for tensor handles
static ROCM_TENSOR_HANDLE_COUNTER: Lazy<Mutex<usize>> = Lazy::new(|| {
    Mutex::new(3000) // Start at 3000 to avoid confusion with other handles
});
