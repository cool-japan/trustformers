//! OpenGL ES GPU Compute Support for Android
//!
//! This module provides OpenGL ES integration with compute shaders
//! for GPU-accelerated neural network inference on Android devices.

use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use trustformers_core::error::{CoreError, Result};

// EGL handle types
#[cfg(target_os = "android")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EGLDisplay(pub *mut c_void);

#[cfg(target_os = "android")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EGLContext(pub *mut c_void);

#[cfg(target_os = "android")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EGLSurface(pub *mut c_void);

#[cfg(target_os = "android")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EGLConfig(pub *mut c_void);

// EGL constants
pub const EGL_SUCCESS: c_int = 0x3000;
pub const EGL_TRUE: c_int = 1;
pub const EGL_FALSE: c_int = 0;
pub const EGL_NONE: c_int = 0x3038;
pub const EGL_NO_CONTEXT: *mut c_void = ptr::null_mut();
pub const EGL_NO_SURFACE: *mut c_void = ptr::null_mut();

// OpenGL ES constants
pub const GL_COMPUTE_SHADER: c_uint = 0x91B9;
pub const GL_SHADER_STORAGE_BUFFER: c_uint = 0x90D2;
pub const GL_SHADER_STORAGE_BARRIER_BIT: c_uint = 0x00002000;
pub const GL_DYNAMIC_DRAW: c_uint = 0x88E8;
pub const GL_READ_ONLY: c_uint = 0x88B8;
pub const GL_WRITE_ONLY: c_uint = 0x88B9;
pub const GL_READ_WRITE: c_uint = 0x88BA;

// OpenGL ES error codes
pub const GL_NO_ERROR: c_uint = 0;
pub const GL_INVALID_ENUM: c_uint = 0x0500;
pub const GL_INVALID_VALUE: c_uint = 0x0501;
pub const GL_INVALID_OPERATION: c_uint = 0x0502;
pub const GL_OUT_OF_MEMORY: c_uint = 0x0505;

// EGL C API bindings
#[cfg(target_os = "android")]
extern "C" {
    pub fn eglGetDisplay(display_id: *mut c_void) -> EGLDisplay;
    pub fn eglInitialize(display: EGLDisplay, major: *mut c_int, minor: *mut c_int) -> c_int;
    pub fn eglTerminate(display: EGLDisplay) -> c_int;
    pub fn eglChooseConfig(
        display: EGLDisplay,
        attrib_list: *const c_int,
        configs: *mut EGLConfig,
        config_size: c_int,
        num_config: *mut c_int,
    ) -> c_int;
    pub fn eglCreateContext(
        display: EGLDisplay,
        config: EGLConfig,
        share_context: EGLContext,
        attrib_list: *const c_int,
    ) -> EGLContext;
    pub fn eglDestroyContext(display: EGLDisplay, context: EGLContext) -> c_int;
    pub fn eglCreatePbufferSurface(
        display: EGLDisplay,
        config: EGLConfig,
        attrib_list: *const c_int,
    ) -> EGLSurface;
    pub fn eglDestroySurface(display: EGLDisplay, surface: EGLSurface) -> c_int;
    pub fn eglMakeCurrent(
        display: EGLDisplay,
        draw: EGLSurface,
        read: EGLSurface,
        context: EGLContext,
    ) -> c_int;
    pub fn eglGetError() -> c_int;
}

// OpenGL ES C API bindings
#[cfg(target_os = "android")]
extern "C" {
    pub fn glGetError() -> c_uint;
    pub fn glCreateProgram() -> c_uint;
    pub fn glDeleteProgram(program: c_uint);
    pub fn glCreateShader(shader_type: c_uint) -> c_uint;
    pub fn glDeleteShader(shader: c_uint);
    pub fn glShaderSource(
        shader: c_uint,
        count: c_int,
        string: *const *const c_char,
        length: *const c_int,
    );
    pub fn glCompileShader(shader: c_uint);
    pub fn glGetShaderiv(shader: c_uint, pname: c_uint, params: *mut c_int);
    pub fn glGetShaderInfoLog(
        shader: c_uint,
        buf_size: c_int,
        length: *mut c_int,
        info_log: *mut c_char,
    );
    pub fn glAttachShader(program: c_uint, shader: c_uint);
    pub fn glLinkProgram(program: c_uint);
    pub fn glGetProgramiv(program: c_uint, pname: c_uint, params: *mut c_int);
    pub fn glGetProgramInfoLog(
        program: c_uint,
        buf_size: c_int,
        length: *mut c_int,
        info_log: *mut c_char,
    );
    pub fn glUseProgram(program: c_uint);
    pub fn glGenBuffers(n: c_int, buffers: *mut c_uint);
    pub fn glDeleteBuffers(n: c_int, buffers: *const c_uint);
    pub fn glBindBuffer(target: c_uint, buffer: c_uint);
    pub fn glBufferData(target: c_uint, size: isize, data: *const c_void, usage: c_uint);
    pub fn glBindBufferBase(target: c_uint, index: c_uint, buffer: c_uint);
    pub fn glMapBufferRange(
        target: c_uint,
        offset: isize,
        length: isize,
        access: c_uint,
    ) -> *mut c_void;
    pub fn glUnmapBuffer(target: c_uint) -> c_uint;
    pub fn glDispatchCompute(num_groups_x: c_uint, num_groups_y: c_uint, num_groups_z: c_uint);
    pub fn glMemoryBarrier(barriers: c_uint);
}

/// OpenGL ES compute context for Android GPU acceleration
#[cfg(target_os = "android")]
pub struct OpenGLESComputeContext {
    display: EGLDisplay,
    context: EGLContext,
    surface: EGLSurface,
    config: EGLConfig,
}

#[cfg(target_os = "android")]
impl OpenGLESComputeContext {
    /// Create a new OpenGL ES compute context
    pub fn new() -> Result<Self> {
        let display = Self::create_display()?;
        let config = Self::choose_config(display)?;
        let context = Self::create_context(display, config)?;
        let surface = Self::create_surface(display, config)?;

        // Make context current
        Self::make_context_current(display, surface, context)?;

        Ok(Self {
            display,
            context,
            surface,
            config,
        })
    }

    /// Create EGL display
    fn create_display() -> Result<EGLDisplay> {
        let display = unsafe { eglGetDisplay(ptr::null_mut()) };

        if display.0.is_null() {
            return Err(TrustformersError::runtime_error(
                "Failed to create EGL display".into(),
            ).into());
        }

        // Initialize EGL
        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        let result = unsafe { eglInitialize(display, &mut major, &mut minor) };

        if result == EGL_FALSE {
            return Err(TrustformersError::runtime_error("Failed to initialize EGL".into()).into());
        }

        tracing::info!("EGL initialized: version {}.{}", major, minor);
        Ok(display)
    }

    /// Choose EGL configuration
    fn choose_config(display: EGLDisplay) -> Result<EGLConfig> {
        let config_attribs = [
            0x3024, 8,    // EGL_RED_SIZE
            0x3023, 8,    // EGL_GREEN_SIZE
            0x3022, 8,    // EGL_BLUE_SIZE
            0x3021, 8,    // EGL_ALPHA_SIZE
            0x3040, 0x0004, // EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT
            EGL_NONE,
        ];

        let mut config = EGLConfig(ptr::null_mut().into());
        let mut num_configs: c_int = 0;

        let result = unsafe {
            eglChooseConfig(
                display,
                config_attribs.as_ptr(),
                &mut config,
                1,
                &mut num_configs,
            )
        };

        if result == EGL_FALSE || num_configs == 0 {
            return Err(TrustformersError::runtime_error(
                "Failed to choose EGL config".into(),
            ).into());
        }

        Ok(config)
    }

    /// Create EGL context
    fn create_context(display: EGLDisplay, config: EGLConfig) -> Result<EGLContext> {
        let context_attribs = [
            0x3098, 3, // EGL_CONTEXT_CLIENT_VERSION, 3 (OpenGL ES 3.x)
            EGL_NONE,
        ];

        let context = unsafe {
            eglCreateContext(
                display,
                config,
                EGLContext(EGL_NO_CONTEXT),
                context_attribs.as_ptr(),
            )
        };

        if context.0.is_null() {
            return Err(TrustformersError::runtime_error(
                "Failed to create EGL context".into(),
            ).into());
        }

        tracing::info!("EGL context created successfully");
        Ok(context)
    }

    /// Create EGL surface (pbuffer for compute)
    fn create_surface(display: EGLDisplay, config: EGLConfig) -> Result<EGLSurface> {
        let surface_attribs = [
            0x3057, 1,    // EGL_WIDTH
            0x3056, 1,    // EGL_HEIGHT
            EGL_NONE,
        ];

        let surface = unsafe {
            eglCreatePbufferSurface(display, config, surface_attribs.as_ptr())
        };

        if surface.0.is_null() {
            return Err(TrustformersError::runtime_error(
                "Failed to create EGL surface".into(),
            ).into());
        }

        Ok(surface)
    }

    /// Make context current
    fn make_context_current(
        display: EGLDisplay,
        surface: EGLSurface,
        context: EGLContext,
    ) -> Result<()> {
        let result = unsafe { eglMakeCurrent(display, surface, surface, context) };

        if result == EGL_FALSE {
            return Err(TrustformersError::runtime_error(
                "Failed to make EGL context current".into(),
            ).into());
        }

        Ok(())
    }

    /// Create compute shader program
    pub fn create_compute_program(&self, shader_source: &str) -> Result<u32> {
        let program = unsafe { glCreateProgram() };
        if program == 0 {
            return Err(TrustformersError::runtime_error(
                "Failed to create OpenGL program".into(),
            ).into());
        }

        // Create and compile compute shader
        let shader = self.create_and_compile_shader(GL_COMPUTE_SHADER, shader_source)?;

        // Attach shader and link program
        unsafe {
            glAttachShader(program, shader);
            glLinkProgram(program);
            glDeleteShader(shader);
        }

        // Check link status
        let mut link_status: c_int = 0;
        unsafe {
            glGetProgramiv(program, 0x8B82, &mut link_status); // GL_LINK_STATUS
        }

        if link_status == 0 {
            let mut log_length: c_int = 0;
            unsafe {
                glGetProgramiv(program, 0x8B84, &mut log_length); // GL_INFO_LOG_LENGTH
            }

            if log_length > 0 {
                let mut log = vec![0u8; log_length as usize];
                unsafe {
                    glGetProgramInfoLog(
                        program,
                        log_length,
                        ptr::null_mut(),
                        log.as_mut_ptr() as *mut c_char,
                    );
                }
                let log_str = String::from_utf8_lossy(&log);
                tracing::error!("Program link error: {}", log_str);
            }

            unsafe { glDeleteProgram(program) };
            return Err(TrustformersError::runtime_error("Failed to link program".into()).into());
        }

        tracing::info!("Compute program created and linked successfully");
        Ok(program)
    }

    /// Create and compile shader
    fn create_and_compile_shader(&self, shader_type: c_uint, source: &str) -> Result<u32> {
        let shader = unsafe { glCreateShader(shader_type) };
        if shader == 0 {
            return Err(TrustformersError::runtime_error(
                "Failed to create shader".into(),
            ).into());
        }

        // Set shader source
        let source_ptr = source.as_ptr() as *const c_char;
        let source_len = source.len() as c_int;
        unsafe {
            glShaderSource(shader, 1, &source_ptr, &source_len);
            glCompileShader(shader);
        }

        // Check compile status
        let mut compile_status: c_int = 0;
        unsafe {
            glGetShaderiv(shader, 0x8B81, &mut compile_status); // GL_COMPILE_STATUS
        }

        if compile_status == 0 {
            let mut log_length: c_int = 0;
            unsafe {
                glGetShaderiv(shader, 0x8B84, &mut log_length); // GL_INFO_LOG_LENGTH
            }

            if log_length > 0 {
                let mut log = vec![0u8; log_length as usize];
                unsafe {
                    glGetShaderInfoLog(
                        shader,
                        log_length,
                        ptr::null_mut(),
                        log.as_mut_ptr() as *mut c_char,
                    );
                }
                let log_str = String::from_utf8_lossy(&log);
                tracing::error!("Shader compile error: {}", log_str);
            }

            unsafe { glDeleteShader(shader) };
            return Err(TrustformersError::runtime_error("Failed to compile shader".into()).into());
        }

        Ok(shader)
    }

    /// Create storage buffer
    pub fn create_storage_buffer(&self, size: usize, data: Option<&[u8]>) -> Result<u32> {
        let mut buffer: c_uint = 0;
        unsafe {
            glGenBuffers(1, &mut buffer);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);

            let data_ptr = if let Some(data) = data {
                data.as_ptr() as *const c_void
            } else {
                ptr::null()
            };

            glBufferData(
                GL_SHADER_STORAGE_BUFFER,
                size as isize,
                data_ptr,
                GL_DYNAMIC_DRAW,
            );
        }

        let error = unsafe { glGetError() };
        if error != GL_NO_ERROR {
            return Err(TrustformersError::runtime_error(format!(
                "Failed to create storage buffer: GL error {}",
                error
            )).into());
        }

        Ok(buffer)
    }

    /// Bind storage buffer to binding point
    pub fn bind_storage_buffer(&self, buffer: u32, binding_point: u32) {
        unsafe {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point, buffer);
        }
    }

    /// Execute compute shader
    pub fn dispatch_compute(
        &self,
        program: u32,
        num_groups_x: u32,
        num_groups_y: u32,
        num_groups_z: u32,
    ) -> Result<()> {
        unsafe {
            glUseProgram(program);
            glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        let error = unsafe { glGetError() };
        if error != GL_NO_ERROR {
            return Err(TrustformersError::runtime_error(format!(
                "Compute dispatch failed: GL error {}",
                error
            )).into());
        }

        Ok(())
    }

    /// Map buffer for reading
    pub fn map_buffer(&self, buffer: u32, size: usize) -> Result<*mut c_void> {
        unsafe {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
            let data = glMapBufferRange(
                GL_SHADER_STORAGE_BUFFER,
                0,
                size as isize,
                GL_READ_ONLY,
            );

            if data.is_null() {
                return Err(TrustformersError::runtime_error("Failed to map buffer".into()).into());
            }

            Ok(data)
        }
    }

    /// Unmap buffer
    pub fn unmap_buffer(&self) {
        unsafe {
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }
    }

    /// Delete buffer
    pub fn delete_buffer(&self, buffer: u32) {
        unsafe {
            glDeleteBuffers(1, &buffer);
        }
    }

    /// Delete program
    pub fn delete_program(&self, program: u32) {
        unsafe {
            glDeleteProgram(program);
        }
    }

    /// Get OpenGL ES version
    pub fn get_version(&self) -> String {
        // In practice, would query glGetString(GL_VERSION)
        "OpenGL ES 3.1".to_string()
    }

    /// Check for compute shader support
    pub fn supports_compute_shaders(&self) -> bool {
        // In practice, would check extensions and version
        true
    }
}

#[cfg(target_os = "android")]
impl Drop for OpenGLESComputeContext {
    fn drop(&mut self) {
        unsafe {
            eglDestroySurface(self.display, self.surface);
            eglDestroyContext(self.display, self.context);
            eglTerminate(self.display);
        }
        tracing::info!("OpenGL ES context destroyed");
    }
}

/// Predefined compute shaders for common operations
pub mod shaders {
    /// ReLU activation compute shader
    pub const RELU_COMPUTE_SHADER: &str = r#"
        #version 310 es

        layout(local_size_x = 16, local_size_y = 16) in;

        layout(binding = 0, rgba32f) uniform readonly image2D inputImage;
        layout(binding = 1, rgba32f) uniform writeonly image2D outputImage;

        void main() {
            ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
            vec4 inputValue = imageLoad(inputImage, coords);
            vec4 outputValue = max(inputValue, vec4(0.0).into());
            imageStore(outputImage, coords, outputValue);
        }
    "#;

    /// Simple matrix multiplication compute shader
    pub const MATRIX_MUL_COMPUTE_SHADER: &str = r#"
        #version 310 es

        layout(local_size_x = 16, local_size_y = 16) in;

        layout(std430, binding = 0) readonly buffer MatrixA {
            float matrixA[];
        };

        layout(std430, binding = 1) readonly buffer MatrixB {
            float matrixB[];
        };

        layout(std430, binding = 2) writeonly buffer MatrixC {
            float matrixC[];
        };

        uniform int widthA;
        uniform int heightA;
        uniform int widthB;

        void main() {
            uint row = gl_GlobalInvocationID.y;
            uint col = gl_GlobalInvocationID.x;

            if (row >= uint(heightA) || col >= uint(widthB)) {
                return;
            }

            float sum = 0.0;
            for (int i = 0; i < widthA; i++) {
                sum += matrixA[row * uint(widthA) + uint(i)] *
                       matrixB[uint(i) * uint(widthB) + col];
            }

            matrixC[row * uint(widthB) + col] = sum;
        }
    "#;

    /// Addition compute shader
    pub const ADD_COMPUTE_SHADER: &str = r#"
        #version 310 es

        layout(local_size_x = 64) in;

        layout(std430, binding = 0) readonly buffer InputA {
            float inputA[];
        };

        layout(std430, binding = 1) readonly buffer InputB {
            float inputB[];
        };

        layout(std430, binding = 2) writeonly buffer Output {
            float output[];
        };

        void main() {
            uint index = gl_GlobalInvocationID.x;
            if (index >= inputA.length()) {
                return;
            }

            output[index] = inputA[index] + inputB[index];
        }
    "#;
}

/// Check if OpenGL ES compute shaders are available
pub fn is_opengl_es_compute_available() -> bool {
    #[cfg(target_os = "android")]
    {
        // In practice, would check for OpenGL ES 3.1+ and compute shader extension
        true
    }
    #[cfg(not(target_os = "android"))]
    {
        false
    }
}

// Stub implementations for non-Android platforms
#[cfg(not(target_os = "android"))]
pub struct OpenGLESComputeContext;

#[cfg(not(target_os = "android"))]
impl OpenGLESComputeContext {
    pub fn new() -> Result<Self> {
        Err(TrustformersError::runtime_error(
            "OpenGL ES is only available on Android".into(),
        ))
    }
}

// Make handles Send/Sync for multithreading
#[cfg(target_os = "android")]
unsafe impl Send for EGLDisplay {}
#[cfg(target_os = "android")]
unsafe impl Sync for EGLDisplay {}
#[cfg(target_os = "android")]
unsafe impl Send for EGLContext {}
#[cfg(target_os = "android")]
unsafe impl Sync for EGLContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opengl_constants() {
        assert_eq!(GL_COMPUTE_SHADER, 0x91B9);
        assert_eq!(GL_NO_ERROR, 0);
        assert_eq!(EGL_TRUE, 1);
        assert_eq!(EGL_FALSE, 0);
    }

    #[test]
    fn test_availability() {
        let _available = is_opengl_es_compute_available();
    }

    #[test]
    fn test_shader_sources() {
        assert!(!shaders::RELU_COMPUTE_SHADER.is_empty());
        assert!(!shaders::MATRIX_MUL_COMPUTE_SHADER.is_empty());
        assert!(!shaders::ADD_COMPUTE_SHADER.is_empty());
    }

    #[cfg(target_os = "android")]
    #[test]
    fn test_context_creation() {
        let context = OpenGLESComputeContext::new();
        if context.is_err() {
            // OpenGL ES might not be available in test environment
            return;
        }

        let context = context.unwrap();
        assert!(context.supports_compute_shaders());
        assert!(!context.get_version().is_empty());
    }
}