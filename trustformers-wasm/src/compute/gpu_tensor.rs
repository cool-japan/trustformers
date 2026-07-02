//! GPU-accelerated tensor implementation

use crate::core::tensor::WasmTensor;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

#[cfg(feature = "webgpu")]
use crate::webgpu::types::{Gpu, GpuAdapter, GpuAdapterExt, GpuDevice, GpuExt};
#[cfg(feature = "webgpu")]
use crate::webgpu::WebGPUBackend;
#[cfg(feature = "webgpu")]
use std::cell::RefCell;
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
    gpu_backend: Option<Rc<RefCell<WebGPUBackend>>>,
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

    /// Initialize the WebGPU backend if a device can be acquired.
    ///
    /// This performs the asynchronous WebGPU handshake exactly as a browser
    /// requires it:
    ///
    /// 1. resolve `navigator.gpu` (the WebGPU entry point),
    /// 2. `requestAdapter()` to pick a physical adapter (a `Promise`),
    /// 3. `requestDevice()` to obtain a logical device and its default queue
    ///    (also a `Promise`).
    ///
    /// Both adapter and device requests are futures, so they are awaited
    /// through `wasm_bindgen_futures::JsFuture`. On success the resulting
    /// [`WebGPUBackend`] is wrapped in `Rc<RefCell<_>>` so it can be cheaply
    /// shared between derived tensors while still allowing the interior
    /// mutability that GPU compute dispatch requires. All failure paths map to
    /// a descriptive [`JsValue`] error; no `unwrap`/`expect` is used.
    #[cfg(feature = "webgpu")]
    pub async fn init_webgpu(&mut self) -> Result<(), JsValue> {
        if !WebGPUBackend::is_available() {
            return Err(JsValue::from_str("WebGPU is not available"));
        }

        // Resolve `navigator.gpu`, the root WebGPU object.
        let navigator = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window object available"))?
            .navigator();
        let gpu = js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu"))?;
        if gpu.is_undefined() || gpu.is_null() {
            return Err(JsValue::from_str("navigator.gpu is unavailable"));
        }
        let gpu: Gpu = gpu.dyn_into()?;

        // Request an adapter. `requestAdapter()` returns a Promise that resolves
        // either to a GPUAdapter or to `null` when none can be provided.
        let adapter = wasm_bindgen_futures::JsFuture::from(gpu.request_adapter()).await?;
        if adapter.is_null() || adapter.is_undefined() {
            return Err(JsValue::from_str("No suitable WebGPU adapter was found"));
        }
        let adapter: GpuAdapter = adapter.dyn_into()?;

        // Request a logical device (and its default queue). Also a Promise.
        let device = wasm_bindgen_futures::JsFuture::from(adapter.request_device()).await?;
        if device.is_null() || device.is_undefined() {
            return Err(JsValue::from_str("Failed to acquire a WebGPU device"));
        }
        let device: GpuDevice = device.dyn_into()?;

        // Build the backend (which also captures the device's queue via
        // `GpuDeviceExt::queue`) and wrap it for shared, interior-mutable access.
        let backend = WebGPUBackend::new(device)?;
        self.gpu_backend = Some(Rc::new(RefCell::new(backend)));
        self.backend = ComputeBackend::WebGpu;

        web_sys::console::log_1(&"WebGPU backend initialized".into());
        Ok(())
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
            if let (ComputeBackend::WebGpu, Some(backend)) = (self.backend, &self.gpu_backend) {
                if other.gpu_backend.is_some() {
                    // Matmul caches compute pipelines / pooled buffers and so
                    // needs mutable access to the backend. The `Rc<RefCell<_>>`
                    // wrapper now provides that interior mutability; the borrow
                    // is released before this scope ends, so no `RefCell` guard
                    // is ever held across an `.await`.
                    let result_tensor =
                        backend.borrow_mut().dispatch_matmul(&self.tensor, &other.tensor)?;
                    return Ok(GpuTensor {
                        tensor: result_tensor,
                        backend: ComputeBackend::WebGpu,
                        gpu_backend: Some(Rc::clone(backend)),
                    });
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
                if other.gpu_backend.is_some() {
                    // Dispatch through the shared backend. The `RefCell` borrow
                    // is scoped to this statement and dropped before any await,
                    // avoiding `await_holding_refcell_ref`.
                    let result_tensor =
                        backend.borrow().dispatch_add(&self.tensor, &other.tensor)?;

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
                // Dispatch through the shared backend. The `RefCell` borrow is
                // scoped to this statement and dropped before any await.
                let result_tensor = backend.borrow().dispatch_relu(&self.tensor)?;

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
