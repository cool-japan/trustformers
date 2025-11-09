//! Debug mode with comprehensive logging and performance monitoring

use js_sys::Date;
use serde::{Deserialize, Serialize};
use std::string::{String, ToString};
use std::sync::Mutex;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Log levels for filtering output
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

/// Performance metrics for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub operation: String,
    pub start_time: f64,
    pub end_time: f64,
    pub duration_ms: f64,
    pub memory_before: usize,
    pub memory_after: usize,
    pub memory_delta: i64,
    pub gpu_memory_used: usize,
}

/// Debug configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DebugConfig {
    enabled: bool,
    log_level: LogLevel,
    console_output: bool,
    performance_tracking: bool,
    memory_tracking: bool,
    gpu_profiling: bool,
    max_log_entries: usize,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl DebugConfig {
    /// Create a new debug configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            enabled: false,
            log_level: LogLevel::Info,
            console_output: true,
            performance_tracking: false,
            memory_tracking: false,
            gpu_profiling: false,
            max_log_entries: 1000,
        }
    }

    /// Enable debug mode
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Disable debug mode
    pub fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Set log level
    pub fn set_log_level(mut self, level: LogLevel) -> Self {
        self.log_level = level;
        self
    }

    /// Enable/disable console output
    pub fn set_console_output(mut self, enabled: bool) -> Self {
        self.console_output = enabled;
        self
    }

    /// Enable/disable performance tracking
    pub fn set_performance_tracking(mut self, enabled: bool) -> Self {
        self.performance_tracking = enabled;
        self
    }

    /// Enable/disable memory tracking
    pub fn set_memory_tracking(mut self, enabled: bool) -> Self {
        self.memory_tracking = enabled;
        self
    }

    /// Enable/disable GPU profiling
    pub fn set_gpu_profiling(mut self, enabled: bool) -> Self {
        self.gpu_profiling = enabled;
        self
    }

    /// Set maximum number of log entries to retain
    pub fn set_max_log_entries(mut self, max: usize) -> Self {
        self.max_log_entries = max;
        self
    }

    /// Create a development configuration with detailed logging
    pub fn development() -> Self {
        Self {
            enabled: true,
            log_level: LogLevel::Debug,
            console_output: true,
            performance_tracking: true,
            memory_tracking: true,
            gpu_profiling: true,
            max_log_entries: 2000,
        }
    }

    /// Create a production configuration with minimal logging
    pub fn production() -> Self {
        Self {
            enabled: true,
            log_level: LogLevel::Warn,
            console_output: false,
            performance_tracking: false,
            memory_tracking: false,
            gpu_profiling: false,
            max_log_entries: 100,
        }
    }
}

/// Log entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: f64,
    pub level: LogLevel,
    pub message: String,
    pub category: String,
    pub source_file: Option<String>,
    pub source_line: Option<u32>,
    pub metadata: Option<serde_json::Value>,
}

/// Debug logger with performance monitoring
#[wasm_bindgen]
#[derive(Clone)]
pub struct DebugLogger {
    config: DebugConfig,
    log_entries: Vec<LogEntry>,
    performance_metrics: Vec<PerformanceMetrics>,
    active_timers: Vec<(String, f64)>,
}

#[wasm_bindgen]
impl DebugLogger {
    /// Create a new debug logger
    #[wasm_bindgen(constructor)]
    pub fn new(config: DebugConfig) -> Self {
        Self {
            config,
            log_entries: Vec::new(),
            performance_metrics: Vec::new(),
            active_timers: Vec::new(),
        }
    }

    /// Log an error message
    pub fn error(&mut self, message: &str, category: &str) {
        self.log(LogLevel::Error, message, category, None, None, None);
    }

    /// Log a warning message
    pub fn warn(&mut self, message: &str, category: &str) {
        self.log(LogLevel::Warn, message, category, None, None, None);
    }

    /// Log an info message
    pub fn info(&mut self, message: &str, category: &str) {
        self.log(LogLevel::Info, message, category, None, None, None);
    }

    /// Log a debug message
    pub fn debug(&mut self, message: &str, category: &str) {
        self.log(LogLevel::Debug, message, category, None, None, None);
    }

    /// Log a trace message
    pub fn trace(&mut self, message: &str, category: &str) {
        self.log(LogLevel::Trace, message, category, None, None, None);
    }

    /// Start a performance timer for an operation
    pub fn start_timer(&mut self, operation: &str) {
        if !self.config.enabled || !self.config.performance_tracking {
            return;
        }

        let start_time = Date::now();
        self.active_timers.push((operation.to_string(), start_time));

        if self.config.console_output {
            web_sys::console::time_with_label(&format!("⏱️ {}", operation));
        }
    }

    /// End a performance timer and record metrics
    pub fn end_timer(&mut self, operation: &str) -> Option<f64> {
        if !self.config.enabled || !self.config.performance_tracking {
            return None;
        }

        let end_time = Date::now();

        // Find and remove the timer
        if let Some(pos) = self.active_timers.iter().position(|(op, _)| op == operation) {
            let (_, start_time) = self.active_timers.remove(pos);
            let duration_ms = end_time - start_time;

            let memory_before = 0; // Would get from memory tracking
            let memory_after = crate::get_wasm_memory_usage();
            let memory_delta = memory_after as i64 - memory_before as i64;

            let metrics = PerformanceMetrics {
                operation: operation.to_string(),
                start_time,
                end_time,
                duration_ms,
                memory_before,
                memory_after,
                memory_delta,
                gpu_memory_used: 0, // Would get from GPU profiling
            };

            self.performance_metrics.push(metrics);

            if self.config.console_output {
                web_sys::console::time_end_with_label(&format!("⏱️ {}", operation));
                web_sys::console::log_1(
                    &format!("🚀 {} completed in {:.2}ms", operation, duration_ms).into(),
                );
            }

            self.trim_metrics();
            Some(duration_ms)
        } else {
            if self.config.console_output {
                web_sys::console::warn_1(
                    &format!("⚠️ Timer '{}' was not started", operation).into(),
                );
            }
            None
        }
    }

    /// Log model loading operation
    pub fn log_model_loading(&mut self, model_name: &str, size_bytes: usize, source: &str) {
        let message = format!(
            "Loading model '{}' ({:.2} MB) from {}",
            model_name,
            size_bytes as f64 / 1_048_576.0,
            source
        );
        self.info(&message, "model_loading");
    }

    /// Log inference operation
    pub fn log_inference(&mut self, model_name: &str, input_shape: &[usize], device: &str) {
        let message = format!(
            "Running inference on model '{}' with input shape {:?} on {}",
            model_name, input_shape, device
        );
        self.debug(&message, "inference");
    }

    /// Log memory usage
    pub fn log_memory_usage(&mut self, context: &str) {
        if !self.config.enabled || !self.config.memory_tracking {
            return;
        }

        let wasm_memory = crate::get_wasm_memory_usage();
        let message = format!(
            "{}: WASM memory usage: {:.2} MB",
            context,
            wasm_memory as f64 / 1_048_576.0
        );
        self.debug(&message, "memory");
    }

    /// Log GPU operation
    pub fn log_gpu_operation(&mut self, operation: &str, buffer_size: usize) {
        if !self.config.enabled || !self.config.gpu_profiling {
            return;
        }

        let message = format!(
            "GPU operation '{}' with buffer size: {:.2} MB",
            operation,
            buffer_size as f64 / 1_048_576.0
        );
        self.debug(&message, "gpu");
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> String {
        if self.performance_metrics.is_empty() {
            return "No performance metrics available".to_string();
        }

        let total_operations = self.performance_metrics.len();
        let total_time: f64 = self.performance_metrics.iter().map(|m| m.duration_ms).sum();
        let avg_time = total_time / total_operations as f64;

        let slowest = self
            .performance_metrics
            .iter()
            .max_by(|a, b| a.duration_ms.partial_cmp(&b.duration_ms).unwrap())
            .unwrap();

        format!(
            "Performance Summary:\n\
             - Total operations: {}\n\
             - Total time: {:.2}ms\n\
             - Average time: {:.2}ms\n\
             - Slowest operation: {} ({:.2}ms)",
            total_operations, total_time, avg_time, slowest.operation, slowest.duration_ms
        )
    }

    /// Export logs as formatted string
    pub fn export_logs(&self) -> String {
        let mut output = String::new();
        output.push_str("=== TrustformersWasm Debug Export ===\n\n");

        output.push_str(&format!(
            "Config: enabled={}, level={:?}, perf_tracking={}, mem_tracking={}, gpu_profiling={}\n\n",
            self.config.enabled,
            self.config.log_level,
            self.config.performance_tracking,
            self.config.memory_tracking,
            self.config.gpu_profiling
        ));

        output.push_str("=== Log Entries ===\n");
        for entry in &self.log_entries {
            output.push_str(&format!(
                "[{:.3}] {:?} [{}] {}\n",
                entry.timestamp, entry.level, entry.category, entry.message
            ));
        }

        output.push_str("\n=== Performance Metrics ===\n");
        for metric in &self.performance_metrics {
            output.push_str(&format!(
                "{}: {:.2}ms (mem: {}→{}, Δ{})\n",
                metric.operation,
                metric.duration_ms,
                metric.memory_before,
                metric.memory_after,
                metric.memory_delta
            ));
        }

        output.push_str("\n=== Summary ===\n");
        output.push_str(&self.get_performance_summary());

        output
    }

    /// Clear all logs and metrics
    pub fn clear(&mut self) {
        self.log_entries.clear();
        self.performance_metrics.clear();
        self.active_timers.clear();

        if self.config.console_output {
            web_sys::console::log_1(&"🧹 Debug logs cleared".into());
        }
    }

    /// Get number of log entries
    #[wasm_bindgen(getter)]
    pub fn log_count(&self) -> usize {
        self.log_entries.len()
    }

    /// Get number of performance metrics
    #[wasm_bindgen(getter)]
    pub fn metrics_count(&self) -> usize {
        self.performance_metrics.len()
    }

    // Private helper methods

    fn log(
        &mut self,
        level: LogLevel,
        message: &str,
        category: &str,
        source_file: Option<String>,
        source_line: Option<u32>,
        metadata: Option<serde_json::Value>,
    ) {
        if !self.config.enabled || level as u8 > self.config.log_level as u8 {
            return;
        }

        let entry = LogEntry {
            timestamp: Date::now(),
            level,
            message: message.to_string(),
            category: category.to_string(),
            source_file,
            source_line,
            metadata,
        };

        self.log_entries.push(entry);

        if self.config.console_output {
            let level_prefix = match level {
                LogLevel::Error => "❌",
                LogLevel::Warn => "⚠️",
                LogLevel::Info => "ℹ️",
                LogLevel::Debug => "🐛",
                LogLevel::Trace => "🔍",
            };

            let formatted_message = format!("{} [{}] {}", level_prefix, category, message);

            match level {
                LogLevel::Error => web_sys::console::error_1(&formatted_message.into()),
                LogLevel::Warn => web_sys::console::warn_1(&formatted_message.into()),
                LogLevel::Info => web_sys::console::info_1(&formatted_message.into()),
                LogLevel::Debug | LogLevel::Trace => {
                    web_sys::console::log_1(&formatted_message.into())
                },
            }
        }

        self.trim_logs();
    }

    fn trim_logs(&mut self) {
        if self.log_entries.len() > self.config.max_log_entries {
            let excess = self.log_entries.len() - self.config.max_log_entries;
            self.log_entries.drain(0..excess);
        }
    }

    fn trim_metrics(&mut self) {
        if self.performance_metrics.len() > self.config.max_log_entries {
            let excess = self.performance_metrics.len() - self.config.max_log_entries;
            self.performance_metrics.drain(0..excess);
        }
    }
}

/// Global debug logger instance
static GLOBAL_LOGGER: Mutex<Option<DebugLogger>> = Mutex::new(None);

/// Initialize the global debug logger
#[wasm_bindgen]
pub fn init_debug_logger(config: DebugConfig) {
    if let Ok(mut logger) = GLOBAL_LOGGER.lock() {
        *logger = Some(DebugLogger::new(config));
    }
}

/// Get the global debug logger
#[wasm_bindgen]
pub fn get_debug_logger() -> Option<DebugLogger> {
    GLOBAL_LOGGER.lock().ok().and_then(|logger| logger.clone())
}

/// Macro for easy logging with automatic category detection
#[macro_export]
macro_rules! debug_log {
    (error, $msg:expr) => {
        if let Some(mut logger) = get_debug_logger() {
            logger.error($msg, module_path!());
        }
    };
    (warn, $msg:expr) => {
        if let Some(mut logger) = get_debug_logger() {
            logger.warn($msg, module_path!());
        }
    };
    (info, $msg:expr) => {
        if let Some(mut logger) = get_debug_logger() {
            logger.info($msg, module_path!());
        }
    };
    (debug, $msg:expr) => {
        if let Some(mut logger) = get_debug_logger() {
            logger.debug($msg, module_path!());
        }
    };
    (trace, $msg:expr) => {
        if let Some(mut logger) = get_debug_logger() {
            logger.trace($msg, module_path!());
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_config() {
        let mut config = DebugConfig::new();
        assert!(!config.enabled);

        config = config.enable().set_log_level(LogLevel::Debug);
        assert!(config.enabled);
        assert_eq!(config.log_level as u8, LogLevel::Debug as u8);
    }

    #[test]
    fn test_logger_creation() {
        let config = DebugConfig::development();
        let logger = DebugLogger::new(config);
        assert_eq!(logger.log_count(), 0);
        assert_eq!(logger.metrics_count(), 0);
    }
}
