//! CUDA GPU backend split modules

mod cuda_backend;
mod cuda_backend_ext;
mod cuda_dispatch;
mod cuda_types;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub use cuda_backend::*;
pub use cuda_dispatch::*;
pub use cuda_types::*;
