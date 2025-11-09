//! GPU-accelerated tensor implementation

use crate::core::tensor::WasmTensor;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

#[cfg(feature = "webgpu")]
use crate::webgpu::WebGPUBackend;
#[cfg(feature = "webgpu")]
use std::rc::Rc;

/// Enum to represent computation backend
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum ComputeBackend {
    Cpu,
    WebGpu,
}

/// GPU-accelerated tensor that can fall back to CPU
#[wasm_bindgen]
pub struct GpuTensor {
    tensor: WasmTensor,
    backend: ComputeBackend,
    #[cfg(feature = "webgpu")]
    gpu_backend: Option<Rc<WebGPUBackend>>,
}

#[wasm_bindgen]
impl GpuTensor {
    /// Create a new GPU tensor
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<GpuTensor, JsValue> {
        let tensor = WasmTensor::new(data, shape)?;

        Ok(GpuTensor {
            tensor,
            backend: ComputeBackend::Cpu,
            #[cfg(feature = "webgpu")]
            gpu_backend: None,
        })
    }

    /// Initialize WebGPU backend if available
    #[cfg(feature = "webgpu")]
    pub async fn init_webgpu(&mut self) -> Result<(), JsValue> {
        if WebGPUBackend::is_available() {
            // Get WebGPU device - this would need to be implemented in WebGPUBackend
            // For now we'll use a placeholder implementation
            web_sys::console::log_1(&"WebGPU backend initialization not yet implemented".into());
            // TODO: Implement WebGPU device creation and backend initialization
            Ok(())
        } else {
            Err(JsValue::from_str("WebGPU is not available"))
        }
    }

    /// Get the current backend
    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> ComputeBackend {
        self.backend
    }

    /// Get tensor data
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.tensor.data()
    }

    /// Get tensor shape
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<usize> {
        self.tensor.shape()
    }

    /// Matrix multiplication with automatic backend selection
    pub async fn matmul(&self, other: &GpuTensor) -> Result<GpuTensor, JsValue> {
        #[cfg(feature = "webgpu")]
        {
            if let (ComputeBackend::WebGpu, Some(_backend)) = (self.backend, &self.gpu_backend) {
                if let Some(_other_backend) = &other.gpu_backend {
                    // GPU acceleration requires mutable access to ops
                    // For now, fall back to CPU implementation
                    // TODO: Wrap backend in Rc<RefCell<>> for interior mutability
                }
            }
        }

        // Fall back to CPU
        let result_tensor = self.tensor.matmul(&other.tensor)?;
        Ok(GpuTensor {
            tensor: result_tensor,
            backend: ComputeBackend::Cpu,
            #[cfg(feature = "webgpu")]
            gpu_backend: None,
        })
    }

    /// Element-wise addition with automatic backend selection
    pub async fn add(&self, other: &GpuTensor) -> Result<GpuTensor, JsValue> {
        #[cfg(feature = "webgpu")]
        {
            if let (ComputeBackend::WebGpu, Some(backend)) = (self.backend, &self.gpu_backend) {
                if let Some(_other_backend) = &other.gpu_backend {
                    // Use GPU acceleration (simplified - currently falls back to CPU)
                    let result_tensor = backend.ops().add(&self.tensor, &other.tensor).await?;

                    return Ok(GpuTensor {
                        tensor: result_tensor,
                        backend: ComputeBackend::WebGpu,
                        gpu_backend: Some(Rc::clone(backend)),
                    });
                }
            }
        }

        // Fall back to CPU
        let result_tensor = self.tensor.add(&other.tensor)?;
        Ok(GpuTensor {
            tensor: result_tensor,
            backend: ComputeBackend::Cpu,
            #[cfg(feature = "webgpu")]
            gpu_backend: None,
        })
    }

    /// ReLU activation with automatic backend selection
    pub async fn relu(&self) -> Result<GpuTensor, JsValue> {
        #[cfg(feature = "webgpu")]
        {
            if let (ComputeBackend::WebGpu, Some(backend)) = (self.backend, &self.gpu_backend) {
                // Use GPU acceleration (simplified - currently falls back to CPU)
                let result_tensor = backend.ops().relu(&self.tensor).await?;

                return Ok(GpuTensor {
                    tensor: result_tensor,
                    backend: ComputeBackend::WebGpu,
                    gpu_backend: Some(Rc::clone(backend)),
                });
            }
        }

        // Fall back to CPU
        let result_tensor = self.tensor.relu();
        Ok(GpuTensor {
            tensor: result_tensor,
            backend: ComputeBackend::Cpu,
            #[cfg(feature = "webgpu")]
            gpu_backend: None,
        })
    }

    /// Check if WebGPU is available
    pub fn webgpu_available() -> bool {
        #[cfg(feature = "webgpu")]
        {
            WebGPUBackend::is_available()
        }

        #[cfg(not(feature = "webgpu"))]
        false
    }

    /// Get backend info as string
    pub fn backend_info(&self) -> String {
        match self.backend {
            ComputeBackend::Cpu => String::from("CPU"),
            ComputeBackend::WebGpu => String::from("WebGPU"),
        }
    }
}

/// Factory functions for creating GPU tensors
#[wasm_bindgen]
pub struct GpuTensorFactory;

#[wasm_bindgen]
impl GpuTensorFactory {
    /// Create a tensor with automatic backend selection
    pub async fn create_tensor(data: Vec<f32>, shape: Vec<usize>) -> Result<GpuTensor, JsValue> {
        let mut tensor = GpuTensor::new(data, shape)?;

        // Try to initialize WebGPU if available
        #[cfg(feature = "webgpu")]
        {
            if GpuTensor::webgpu_available() {
                match tensor.init_webgpu().await {
                    Ok(_) => {},
                    Err(_) => {
                        // Fall back to CPU silently
                    },
                }
            }
        }

        Ok(tensor)
    }

    /// Create zeros tensor
    pub async fn zeros(shape: Vec<usize>) -> Result<GpuTensor, JsValue> {
        let size = shape.iter().product();
        let data = vec![0.0f32; size];
        Self::create_tensor(data, shape).await
    }

    /// Create ones tensor
    pub async fn ones(shape: Vec<usize>) -> Result<GpuTensor, JsValue> {
        let size = shape.iter().product();
        let data = vec![1.0f32; size];
        Self::create_tensor(data, shape).await
    }
}
