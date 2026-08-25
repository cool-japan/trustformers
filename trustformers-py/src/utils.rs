//! Python bindings for utility functions

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// Get available device (CPU, CUDA, Metal, etc.)
///
/// Returns `"cuda"` only when this crate is built with the `cuda` feature
/// (see [`is_cuda_available`] -- currently always `false`, honestly, since
/// no CUDA backend is compiled into this crate). Returns `"metal"` only
/// when built with the `metal` feature on macOS (see
/// [`is_metal_available`]); otherwise `"cpu"`.
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

/// Why `set_seed()` refuses instead of silently succeeding.
///
/// Every source of randomness this crate's Rust side uses for tensor
/// creation -- and therefore for every model's weight initialisation --
/// traces back to `trustformers_core::tensor::Tensor::randn`
/// (`trustformers-core/src/tensor/constructors.rs`), which calls
/// `scirs2_core::random::thread_rng()`. That function
/// (`scirs2-core-0.6.5/src/random/core.rs::thread_rng`) is
/// `Random::default()` wrapping `rand::rngs::ThreadRng` -- `rand`'s
/// OS-entropy-backed thread-local generator, which by design exposes no
/// public reseeding API. Verified: `scirs2-core` 0.6.5's public `random`
/// module has no `set_seed`/`reseed`/thread-local-override function for it
/// anywhere. So every `Linear`/`Embedding`/`LayerNorm` weight this crate
/// initialises (`BertModel::new`, `Gpt2Model::new`, ...) draws from a
/// generator nothing in this crate can influence.
///
/// The one seedable generator that does exist in-tree --
/// `trustformers_core::generation::core::TextGenerator::with_seed` -- is a
/// per-instance builder for autoregressive *decoding* sampling only. It has
/// nothing to do with weight initialisation, and there is no global default
/// it reads from either: an un-seeded `TextGenerator` draws a fresh seed
/// from thread entropy on every call.
///
/// Previously this function wrote a `TRUSTFORMERS_SEED` environment
/// variable and returned success. A repo-wide search found that variable
/// read nowhere in any of this workspace's 11 crates -- a silent no-op.
/// Making seeding real needs a cross-crate change: a process-global,
/// reseedable RNG that `Tensor::randn`/`randn_f16`/`randn_bf16` are changed
/// to route through instead of `thread_rng()`. That lives in
/// `trustformers-core`, outside what this crate can do by itself.
///
/// Split out from the `PyErr`-constructing wrapper below so it is
/// unit-testable with a plain `cargo test`: constructing a `PyErr`'s
/// `Display` output requires an initialized Python interpreter (this
/// crate's `cargo test` binary does not embed one), but building the
/// `&str` this function returns does not.
fn no_seed_hook_message() -> &'static str {
    "set_seed() cannot make anything in this runtime reproducible: Tensor::randn (used by \
     every Linear/Embedding/LayerNorm weight initialisation) draws from \
     scirs2_core::random::thread_rng(), which wraps rand::rngs::ThreadRng -- an OS-entropy \
     generator with no public seeding hook anywhere in scirs2-core 0.6.5. The only seedable \
     RNG in this workspace, TextGenerator::with_seed, is a per-call decoding-sampler builder \
     unrelated to weight initialisation, with no global default this function could set. \
     Previously this wrote a TRUSTFORMERS_SEED environment variable that nothing in any of \
     this workspace's crates ever read -- a silent no-op. Making seeding real requires a \
     process-global reseedable RNG in trustformers-core that Tensor::randn is changed to use \
     instead of thread_rng(); that is a cross-crate change this function cannot make."
}

fn no_seed_hook_available() -> PyErr {
    PyNotImplementedError::new_err(no_seed_hook_message())
}

/// Set random seed for reproducibility.
///
/// Always raises `NotImplementedError`: see [`no_seed_hook_available`] for
/// exactly why. This never silently succeeds without seeding anything.
#[pyfunction]
pub fn set_seed(seed: u64) -> PyResult<()> {
    let _ = seed;
    Err(no_seed_hook_available())
}

/// Enable gradient computation.
///
/// This runtime does not consult a global gradient-computation switch
/// anywhere. A real one exists --
/// `trustformers_core::autodiff::engine::{get_engine, AutodiffEngine::
/// enable_grad, AutodiffEngine::is_grad_enabled}`, backed by a genuine
/// process-global `OnceLock` -- but nothing reads `is_grad_enabled()`
/// except that engine's own (otherwise-uncalled) `no_grad`/`with_grad`
/// helper methods and its own tests: `Variable::new` and
/// `PyTensor::from_tensor_with_grad` (`trustformers-py/src/tensor.rs`) take
/// an explicit `requires_grad: bool` and never consult the engine's flag,
/// and no registered model's `forward()` builds an autodiff graph at all
/// (`trustformers-py/src/training.rs::no_training_path_available` documents
/// this independently, for the same reason `Trainer.train()` refuses).
/// Toggling that flag from here would therefore change nothing observable
/// -- it would just relocate the fiction from an unread environment
/// variable to an unread mutex-guarded field.
///
/// This function exists for PyTorch-idiom API compatibility (code that
/// unconditionally calls `enable_grad()`/`no_grad()` around inference) and
/// is a guaranteed, documented no-op: it always succeeds and never changes
/// what any computation does. Per-tensor gradient tracking, where it
/// exists at all, is controlled explicitly at tensor-construction time via
/// `Tensor(..., requires_grad=True)`, not by a global switch.
#[pyfunction]
pub fn enable_grad() -> PyResult<()> {
    Ok(())
}

/// Disable gradient computation ("no_grad" mode).
///
/// See [`enable_grad`]'s doc for the full explanation. This is the same
/// documented no-op: no global gradient-computation switch is consulted by
/// anything in this runtime, so "gradients are off" is permanently and
/// vacuously true here already, independent of this call.
#[pyfunction]
pub fn no_grad() -> PyResult<()> {
    Ok(())
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
            self.laps.iter().map(|(label, time)| (label.clone(), *time)),
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
#[pyclass(name = "Config", from_py_object)]
#[derive(Clone, Default)]
pub struct PyConfig {
    data: HashMap<String, String>,
}

#[pymethods]
impl PyConfig {
    #[new]
    pub fn new() -> Self {
        Self::default()
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
            let _ = io::stdout().flush(); // Ignore flush errors in progress display
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

#[cfg(feature = "cuda")]
fn is_cuda_available() -> bool {
    // Real reason this is unconditionally `false`, not a mock: this crate
    // has no CUDA backend. `trustformers-core` does have a real `cuda`
    // feature (gated on the `oxicuda-*` crates), but this crate's own
    // `trustformers-core` dependency line builds it with
    // `default-features = false` and no feature list at all, so that
    // backend is never compiled in here regardless of this crate's own
    // (empty, `cuda = []`) `cuda` feature. Wiring the two together is a
    // Cargo.toml change, outside this file.
    false
}

/// Whether this build can route to a Metal backend.
///
/// Requires both macOS (checked here) and this crate's `metal` cargo
/// feature being compiled in (checked below) -- `metal` is not in
/// `default`, so a plain `cargo build`/`pip install` no longer claims Metal
/// support it never compiled, which the previous "always true on macOS"
/// mock did unconditionally.
///
/// This does *not* probe for a live Metal device: that needs an FFI call
/// (e.g. via the `metal` crate's `Device::system_default`) which needs a
/// `-framework Metal` link directive in a build script this crate does not
/// have, and this crate's own `metal` feature does not forward to
/// `trustformers-core`'s real `gpu_ops::metal` backend either (both are
/// Cargo.toml changes, outside this file -- tracked as follow-ups). So
/// `true` here means "compiled with Metal requested, on macOS", which is
/// the most this function can honestly verify today.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn is_metal_available() -> bool {
    true
}

/// The `metal` feature was not compiled in: nothing in this build attempts
/// to route to a Metal backend, regardless of the host's actual hardware.
#[cfg(all(target_os = "macos", not(feature = "metal")))]
fn is_metal_available() -> bool {
    false
}

/// Check if running in Jupyter/IPython
#[pyfunction]
pub fn is_notebook() -> bool {
    // Check for IPython/Jupyter environment
    std::env::var("JPY_PARENT_PID").is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- set_seed: honest refusal, not a silent no-op ----

    #[test]
    fn set_seed_refuses_instead_of_silently_succeeding() {
        // Previously this wrote an environment variable nothing read and
        // returned `Ok(())`. There is no in-tree hook for it to wire to
        // (see `no_seed_hook_message`'s doc), so it must refuse.
        assert!(
            set_seed(42).is_err(),
            "set_seed must not silently claim success when nothing in this workspace is \
             actually seeded"
        );
    }

    /// `PyErr`'s `Display` needs an initialized Python interpreter this
    /// `cargo test` binary does not embed (see `no_seed_hook_message`'s doc
    /// for why the message is split into a plain `&str` function); this
    /// checks the message content directly instead of via the `PyErr`.
    #[test]
    fn no_seed_hook_message_names_the_real_blocker_precisely() {
        let message = no_seed_hook_message();
        assert!(
            message.contains("thread_rng"),
            "must name the actual (non-seedable) RNG entry point: {message}"
        );
        assert!(
            message.contains("TextGenerator::with_seed"),
            "must name the one seedable generator that does exist, and why it doesn't help: {message}"
        );
        assert!(
            message.contains("TRUSTFORMERS_SEED"),
            "must name what the previous silent no-op did: {message}"
        );
    }

    // ---- enable_grad / no_grad: documented no-ops, not silent ones ----

    #[test]
    fn enable_grad_and_no_grad_always_succeed() {
        assert!(enable_grad().is_ok());
        assert!(no_grad().is_ok());
    }

    #[test]
    fn enable_grad_and_no_grad_touch_no_process_environment_state() {
        // The previous implementation's only observable effect was writing
        // TRUSTFORMERS_GRAD_ENABLED, which nothing read. Confirm that side
        // effect is gone entirely rather than merely unread.
        std::env::remove_var("TRUSTFORMERS_GRAD_ENABLED");
        enable_grad().expect("documented no-op must not fail");
        no_grad().expect("documented no-op must not fail");
        assert!(
            std::env::var("TRUSTFORMERS_GRAD_ENABLED").is_err(),
            "enable_grad/no_grad must not resurrect the unread environment variable"
        );
    }

    // ---- get_device: is_metal_available / is_cuda_available honesty ----

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn is_metal_available_true_when_macos_and_metal_feature_compiled() {
        assert!(is_metal_available());
    }

    #[cfg(all(target_os = "macos", not(feature = "metal")))]
    #[test]
    fn is_metal_available_false_without_the_metal_feature() {
        // The regression this guards: a default build (no `--features
        // metal`) on macOS previously reported "metal" unconditionally.
        assert!(!is_metal_available());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn is_cuda_available_is_honestly_false() {
        assert!(!is_cuda_available());
    }
}
