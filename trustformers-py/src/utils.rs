//! Python bindings for utility functions

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// Get available device (CPU, CUDA, Metal, etc.)
#[pyfunction]
pub fn get_device(_py: Python<'_>) -> PyResult<String> {
    // Check for available devices
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            return Ok("cuda".to_string());
        }
    }

    #[cfg(target_os = "macos")]
    {
        if is_metal_available() {
            return Ok("metal".to_string());
        }
    }

    Ok("cpu".to_string())
}

/// Set random seed for reproducibility
#[pyfunction]
pub fn set_seed(seed: u64) -> PyResult<()> {
    // Set seed for various components
    // In a real implementation, this would set seeds for:
    // - Rust's random number generators
    // - NumPy
    // - PyTorch (if available)
    // - CUDA (if available)

    // For now, just store it in a global variable
    std::env::set_var("TRUSTFORMERS_SEED", seed.to_string());
    Ok(())
}

/// Enable gradient computation
#[pyfunction]
pub fn enable_grad() -> PyResult<()> {
    // In a real implementation, this would enable gradient tracking
    std::env::set_var("TRUSTFORMERS_GRAD_ENABLED", "true");
    Ok(())
}

/// Disable gradient computation (no_grad context)
#[pyfunction]
pub fn no_grad() -> PyResult<()> {
    // In a real implementation, this would disable gradient tracking
    std::env::set_var("TRUSTFORMERS_GRAD_ENABLED", "false");
    Ok(())
}

/// Context manager for no_grad
#[pyclass(name = "no_grad_context")]
pub struct NoGradContext {
    prev_state: bool,
}

#[pymethods]
impl NoGradContext {
    #[new]
    pub fn new() -> Self {
        let prev_state =
            std::env::var("TRUSTFORMERS_GRAD_ENABLED").map(|v| v == "true").unwrap_or(true);
        NoGradContext { prev_state }
    }

    fn __enter__(&self) -> PyResult<()> {
        std::env::set_var("TRUSTFORMERS_GRAD_ENABLED", "false");
        Ok(())
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        std::env::set_var("TRUSTFORMERS_GRAD_ENABLED", self.prev_state.to_string());
        Ok(false) // Don't suppress exceptions
    }
}

/// Download a model from Hugging Face Hub
#[pyfunction]
#[pyo3(signature = (
    model_name,
    cache_dir = None,
    force_download = false,
    resume_download = true,
    token = None,
))]
pub fn download_model(
    py: Python<'_>,
    model_name: String,
    _cache_dir: Option<String>,
    force_download: bool,
    _resume_download: bool,
    _token: Option<String>,
) -> PyResult<Bound<'_, PyDict>> {
    // In a real implementation, this would download from HF Hub
    // For now, return mock information
    let result = PyDict::new(py);
    result.set_item("model_path", format!("/path/to/{}", model_name))?;
    result.set_item(
        "config_path",
        format!("/path/to/{}/config.json", model_name),
    )?;
    result.set_item(
        "tokenizer_path",
        format!("/path/to/{}/tokenizer.json", model_name),
    )?;
    result.set_item("cached", !force_download)?;
    Ok(result)
}

/// List available models
#[pyfunction]
pub fn list_models(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let models = vec![
        "bert-base-uncased",
        "bert-large-uncased",
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "t5-small",
        "t5-base",
        "llama-7b",
        "mistral-7b",
    ];

    let list = PyList::new(py, models)?;
    Ok(list)
}

/// Get model information
#[pyfunction]
pub fn model_info(py: Python<'_>, model_name: String) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);

    // Mock model information
    match model_name.as_str() {
        "bert-base-uncased" => {
            info.set_item("architecture", "BERT")?;
            info.set_item("hidden_size", 768)?;
            info.set_item("num_layers", 12)?;
            info.set_item("num_heads", 12)?;
            info.set_item("vocab_size", 30522)?;
            info.set_item("parameters", "110M")?;
        },
        "gpt2" => {
            info.set_item("architecture", "GPT2")?;
            info.set_item("hidden_size", 768)?;
            info.set_item("num_layers", 12)?;
            info.set_item("num_heads", 12)?;
            info.set_item("vocab_size", 50257)?;
            info.set_item("parameters", "124M")?;
        },
        "llama-7b" => {
            info.set_item("architecture", "LLaMA")?;
            info.set_item("hidden_size", 4096)?;
            info.set_item("num_layers", 32)?;
            info.set_item("num_heads", 32)?;
            info.set_item("vocab_size", 32000)?;
            info.set_item("parameters", "7B")?;
        },
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown model: {}",
                model_name
            )));
        },
    }

    Ok(info)
}

/// Memory utilities
#[pyclass(name = "MemoryTracker")]
pub struct PyMemoryTracker {
    baseline: usize,
    peak: usize,
}

#[pymethods]
impl PyMemoryTracker {
    #[new]
    pub fn new() -> Self {
        let baseline = get_memory_usage();
        PyMemoryTracker {
            baseline,
            peak: baseline,
        }
    }

    /// Get current memory usage in MB
    fn current_mb(&mut self) -> f32 {
        let current = get_memory_usage();
        self.peak = self.peak.max(current);
        current as f32 / 1_048_576.0
    }

    /// Get peak memory usage in MB
    fn peak_mb(&self) -> f32 {
        self.peak as f32 / 1_048_576.0
    }

    /// Get memory increase since baseline in MB
    fn delta_mb(&self) -> f32 {
        let current = get_memory_usage();
        (current.saturating_sub(self.baseline)) as f32 / 1_048_576.0
    }

    /// Reset baseline
    fn reset(&mut self) {
        self.baseline = get_memory_usage();
        self.peak = self.baseline;
    }

    fn __repr__(&mut self) -> String {
        format!(
            "MemoryTracker(current={:.1}MB, peak={:.1}MB, delta={:.1}MB)",
            self.current_mb(),
            self.peak_mb(),
            self.delta_mb()
        )
    }
}

/// Performance timer
#[pyclass(name = "Timer")]
pub struct PyTimer {
    name: String,
    start_time: std::time::Instant,
    laps: Vec<(String, f64)>,
}

#[pymethods]
impl PyTimer {
    #[new]
    #[pyo3(signature = (name = "Timer"))]
    pub fn new(name: &str) -> Self {
        PyTimer {
            name: name.to_string(),
            start_time: std::time::Instant::now(),
            laps: Vec::new(),
        }
    }

    /// Record a lap time
    fn lap(&mut self, label: Option<String>) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let label = label.unwrap_or_else(|| format!("Lap {}", self.laps.len() + 1));
        self.laps.push((label, elapsed));
        elapsed
    }

    /// Get total elapsed time in seconds
    fn elapsed(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get elapsed time in milliseconds
    fn elapsed_ms(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64() * 1000.0
    }

    /// Reset the timer
    fn reset(&mut self) {
        self.start_time = std::time::Instant::now();
        self.laps.clear();
    }

    /// Get all lap times
    fn get_laps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let laps = PyList::new(
            py,
            self.laps.iter().map(|(label, time)| {
                let tuple = (label.clone(), *time);
                tuple
            }),
        )?;
        Ok(laps)
    }

    fn __repr__(&self) -> String {
        format!(
            "Timer(name='{}', elapsed={:.3}s, laps={})",
            self.name,
            self.elapsed(),
            self.laps.len()
        )
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let elapsed = self.elapsed();
        println!("{}: {:.3}s", self.name, elapsed);
        Ok(false)
    }
}

/// Configuration utilities
#[pyclass(name = "Config")]
#[derive(Clone)]
pub struct PyConfig {
    data: HashMap<String, String>,
}

#[pymethods]
impl PyConfig {
    #[new]
    pub fn new() -> Self {
        PyConfig {
            data: HashMap::new(),
        }
    }

    /// Load configuration from file
    #[staticmethod]
    fn from_file(path: String) -> PyResult<Self> {
        // In a real implementation, this would load from JSON/YAML
        let mut config = PyConfig::new();
        config.set("loaded_from".to_string(), path);
        Ok(config)
    }

    /// Get configuration value
    fn get(&self, key: String, default: Option<String>) -> Option<String> {
        self.data.get(&key).cloned().or(default)
    }

    /// Set configuration value
    fn set(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }

    /// Update from dictionary
    fn update(&mut self, other: &Bound<'_, PyDict>) -> PyResult<()> {
        for (key, value) in other.iter() {
            let key: String = key.extract()?;
            let value: String = value.extract()?;
            self.data.insert(key, value);
        }
        Ok(())
    }

    /// Convert to dictionary
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in &self.data {
            dict.set_item(key, value)?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("Config(items={})", self.data.len())
    }
}

/// Logging utilities
#[pyfunction]
#[pyo3(signature = (message, level = "INFO"))]
pub fn log(message: &str, level: &str) {
    // Simple timestamp without chrono dependency
    println!("[{}] - {}", level, message);
}

/// Progress bar wrapper
#[pyclass(name = "ProgressBar")]
pub struct PyProgressBar {
    total: usize,
    current: usize,
    description: String,
}

#[pymethods]
impl PyProgressBar {
    #[new]
    #[pyo3(signature = (total, description = "Progress"))]
    pub fn new(total: usize, description: &str) -> Self {
        PyProgressBar {
            total,
            current: 0,
            description: description.to_string(),
        }
    }

    /// Update progress
    fn update(&mut self, n: usize) {
        self.current = (self.current + n).min(self.total);
        self.render();
    }

    /// Set current progress
    fn set(&mut self, n: usize) {
        self.current = n.min(self.total);
        self.render();
    }

    /// Set description
    fn set_description(&mut self, description: String) {
        self.description = description;
        self.render();
    }

    /// Close the progress bar
    fn close(&self) {
        println!(); // New line after progress bar
    }

    fn render(&self) {
        let percent = (self.current as f32 / self.total as f32 * 100.0) as usize;
        let filled = percent / 2;
        let empty = 50 - filled;

        print!(
            "\r{}: [{}{}] {}/{} ({}%)",
            self.description,
            "█".repeat(filled),
            "░".repeat(empty),
            self.current,
            self.total,
            percent
        );

        if self.current >= self.total {
            println!();
        } else {
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ProgressBar(total={}, current={}, description='{}')",
            self.total, self.current, self.description
        )
    }
}

// Helper functions

fn get_memory_usage() -> usize {
    // Mock implementation - in reality would get actual memory usage
    1024 * 1024 * 100 // 100 MB
}

#[cfg(feature = "cuda")]
fn is_cuda_available() -> bool {
    // Check if CUDA is available
    false // Mock implementation
}

#[cfg(target_os = "macos")]
fn is_metal_available() -> bool {
    // Check if Metal is available
    true // Mock implementation - Metal is usually available on macOS
}

/// Check if running in Jupyter/IPython
#[pyfunction]
pub fn is_notebook() -> bool {
    // Check for IPython/Jupyter environment
    std::env::var("JPY_PARENT_PID").is_ok()
}

/// Get version information
#[pyfunction]
pub fn version_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);
    info.set_item("trustformers", env!("CARGO_PKG_VERSION"))?;
    info.set_item("rust", env!("CARGO_PKG_RUST_VERSION"))?;
    info.set_item("python", py.version())?;

    #[cfg(feature = "cuda")]
    info.set_item("cuda", "11.8")?; // Mock version

    #[cfg(target_os = "macos")]
    info.set_item("metal", "3.0")?; // Mock version

    Ok(info)
}
