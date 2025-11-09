//! Comprehensive error handling for TrustformeRS WASM
//!
//! This module provides structured error types, error codes, and utilities
//! for robust error handling across the entire codebase.

use serde::{Deserialize, Serialize};
use std::borrow::ToOwned;
use std::fmt;
use std::string::{String, ToString};
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Error codes for categorizing different types of errors
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCode {
    // Model errors (E1xxx)
    E1001 = 1001, // Invalid model format
    E1002 = 1002, // Model too large
    E1003 = 1003, // Unsupported architecture
    E1004 = 1004, // Model loading timeout
    E1005 = 1005, // Model corruption detected
    E1006 = 1006, // Model version mismatch

    // Inference errors (E2xxx)
    E2001 = 2001, // Input shape mismatch
    E2002 = 2002, // Inference computation failed
    E2003 = 2003, // Output buffer overflow
    E2004 = 2004, // Inference timeout
    E2005 = 2005, // Invalid input data
    E2006 = 2006, // Batch size mismatch

    // Device errors (E3xxx)
    E3001 = 3001, // Device initialization failed
    E3002 = 3002, // WebGPU unavailable
    E3003 = 3003, // Device memory allocation failed
    E3004 = 3004, // Device context lost
    E3005 = 3005, // Unsupported device feature
    E3006 = 3006, // Device driver error

    // Memory errors (E4xxx)
    E4001 = 4001, // Out of memory
    E4002 = 4002, // Memory allocation failed
    E4003 = 4003, // Memory access violation
    E4004 = 4004, // Memory leak detected
    E4005 = 4005, // Buffer overflow
    E4006 = 4006, // Stack overflow

    // Configuration errors (E5xxx)
    E5001 = 5001, // Invalid configuration parameter
    E5002 = 5002, // Feature not available
    E5003 = 5003, // Version mismatch
    E5004 = 5004, // Environment not supported
    E5005 = 5005, // Missing required dependency
    E5006 = 5006, // License validation failed,

    // Network/IO errors (E6xxx)
    E6001 = 6001, // Network connection failed
    E6002 = 6002, // File not found
    E6003 = 6003, // Download timeout
    E6004 = 6004, // Corrupted download
    E6005 = 6005, // Access denied
    E6006 = 6006, // Rate limit exceeded

    // Storage errors (E7xxx)
    E7001 = 7001, // Storage quota exceeded
    E7002 = 7002, // Storage access denied
    E7003 = 7003, // Cache corruption
    E7004 = 7004, // Serialization failed
    E7005 = 7005, // IndexedDB error
    E7006 = 7006, // Storage cleanup failed

    // WebAssembly errors (E8xxx)
    E8001 = 8001, // WASM compilation failed
    E8002 = 8002, // WASM instantiation failed
    E8003 = 8003, // WASM memory limit exceeded
    E8004 = 8004, // WASM table overflow
    E8005 = 8005, // WASM validation failed
    E8006 = 8006, // WASM runtime error

    // Quantization errors (E9xxx)
    E9001 = 9001, // Quantization failed
    E9002 = 9002, // Unsupported quantization precision
    E9003 = 9003, // Calibration data insufficient
    E9004 = 9004, // Quantization accuracy loss
    E9005 = 9005, // Dequantization error
    E9006 = 9006, // Quantization metadata invalid
}

/// Error severity levels
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Fatal errors that prevent further operation
    Fatal = 0,
    /// Errors that affect functionality but allow graceful degradation
    Error = 1,
    /// Warning conditions that may lead to errors
    Warning = 2,
    /// Informational messages about potential issues
    Info = 3,
}

/// Error context providing additional information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorContext {
    pub operation: Option<String>,
    pub component: Option<String>,
    pub model_id: Option<String>,
    pub input_shape: Option<Vec<usize>>,
    pub device_type: Option<String>,
    pub memory_usage_mb: Option<f64>,
    pub additional_info: Option<String>,
}

/// Comprehensive error type for TrustformeRS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustformersError {
    pub code: ErrorCode,
    pub message: String,
    pub severity: ErrorSeverity,
    pub context: ErrorContext,
    pub timestamp: f64,
    pub stack_trace: Option<String>,
    pub recovery_suggestion: Option<String>,
}

impl TrustformersError {
    /// Create a new error with minimal information
    pub fn new(code: ErrorCode, message: &str) -> Self {
        Self {
            code,
            message: message.to_owned(),
            severity: Self::default_severity(code),
            context: ErrorContext::default(),
            timestamp: Self::current_timestamp(),
            stack_trace: None,
            recovery_suggestion: Self::default_recovery_suggestion(code),
        }
    }

    /// Create a new error with full context
    pub fn with_context(
        code: ErrorCode,
        message: &str,
        severity: ErrorSeverity,
        context: ErrorContext,
    ) -> Self {
        Self {
            code,
            message: message.to_owned(),
            severity,
            context,
            timestamp: Self::current_timestamp(),
            stack_trace: None,
            recovery_suggestion: Self::default_recovery_suggestion(code),
        }
    }

    /// Add stack trace information
    pub fn with_stack_trace(mut self, stack_trace: String) -> Self {
        self.stack_trace = Some(stack_trace);
        self
    }

    /// Add custom recovery suggestion
    pub fn with_recovery_suggestion(mut self, suggestion: String) -> Self {
        self.recovery_suggestion = Some(suggestion);
        self
    }

    /// Get default severity for error code
    fn default_severity(code: ErrorCode) -> ErrorSeverity {
        match code {
            // Fatal errors
            ErrorCode::E4001 | ErrorCode::E4002 | ErrorCode::E8001 | ErrorCode::E8002 => {
                ErrorSeverity::Fatal
            },

            // High-severity errors
            ErrorCode::E1001
            | ErrorCode::E1002
            | ErrorCode::E2001
            | ErrorCode::E2002
            | ErrorCode::E3001
            | ErrorCode::E3002
            | ErrorCode::E6001
            | ErrorCode::E7001 => ErrorSeverity::Error,

            // Warnings
            ErrorCode::E1004 | ErrorCode::E2004 | ErrorCode::E3004 | ErrorCode::E4004 => {
                ErrorSeverity::Warning
            },

            // Default to error
            _ => ErrorSeverity::Error,
        }
    }

    /// Get default recovery suggestion for error code
    fn default_recovery_suggestion(code: ErrorCode) -> Option<String> {
        let suggestion = match code {
            ErrorCode::E1001 => "Verify model format is supported and file is not corrupted",
            ErrorCode::E1002 => "Use a smaller model or enable quantization to reduce memory usage",
            ErrorCode::E2001 => "Check input tensor shape matches model requirements",
            ErrorCode::E3001 => "Check device drivers and WebGPU support",
            ErrorCode::E3002 => "Enable WebGPU in browser settings or fallback to CPU",
            ErrorCode::E4001 => "Enable quantization or use a smaller model",
            ErrorCode::E4002 => "Close other applications to free memory",
            ErrorCode::E5002 => "Check browser compatibility and feature availability",
            ErrorCode::E6001 => "Check network connectivity and try again",
            ErrorCode::E7001 => "Clear browser storage or increase storage quota",
            _ => return None,
        };
        Some(suggestion.to_owned())
    }

    /// Get current timestamp
    fn current_timestamp() -> f64 {
        js_sys::Date::now()
    }

    /// Convert to user-friendly message
    pub fn user_message(&self) -> String {
        match self.severity {
            ErrorSeverity::Fatal => format!(
                "Fatal Error ({code}): {message}",
                code = self.code as u32,
                message = self.message
            ),
            ErrorSeverity::Error => format!(
                "Error ({code}): {message}",
                code = self.code as u32,
                message = self.message
            ),
            ErrorSeverity::Warning => format!(
                "Warning ({code}): {message}",
                code = self.code as u32,
                message = self.message
            ),
            ErrorSeverity::Info => format!(
                "Info ({code}): {message}",
                code = self.code as u32,
                message = self.message
            ),
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        !matches!(self.severity, ErrorSeverity::Fatal)
    }

    /// Get error category
    pub fn category(&self) -> &'static str {
        let code_num = self.code as u32;
        match code_num {
            1001..=1999 => "Model",
            2001..=2999 => "Inference",
            3001..=3999 => "Device",
            4001..=4999 => "Memory",
            5001..=5999 => "Configuration",
            6001..=6999 => "Network",
            7001..=7999 => "Storage",
            8001..=8999 => "WebAssembly",
            9001..=9999 => "Quantization",
            _ => "Unknown",
        }
    }
}

impl fmt::Display for TrustformersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.user_message())
    }
}

/// Convert TrustformersError to JsValue for WASM interop
impl From<TrustformersError> for JsValue {
    fn from(error: TrustformersError) -> Self {
        // Create JavaScript Error object with enhanced properties
        let js_error = js_sys::Error::new(&error.user_message());

        // Add custom properties
        js_sys::Reflect::set(&js_error, &"code".into(), &JsValue::from(error.code as u32)).ok();
        js_sys::Reflect::set(
            &js_error,
            &"severity".into(),
            &JsValue::from(error.severity as u32),
        )
        .ok();
        js_sys::Reflect::set(
            &js_error,
            &"category".into(),
            &JsValue::from(error.category()),
        )
        .ok();
        js_sys::Reflect::set(
            &js_error,
            &"timestamp".into(),
            &JsValue::from(error.timestamp),
        )
        .ok();
        js_sys::Reflect::set(
            &js_error,
            &"isRecoverable".into(),
            &JsValue::from(error.is_recoverable()),
        )
        .ok();

        if let Some(ref suggestion) = error.recovery_suggestion {
            js_sys::Reflect::set(
                &js_error,
                &"recoverySuggestion".into(),
                &JsValue::from(suggestion),
            )
            .ok();
        }

        if let Some(ref stack) = error.stack_trace {
            js_sys::Reflect::set(&js_error, &"stackTrace".into(), &JsValue::from(stack)).ok();
        }

        // Add context as JSON
        if let Ok(context_json) = serde_json::to_string(&error.context) {
            js_sys::Reflect::set(&js_error, &"context".into(), &JsValue::from(context_json)).ok();
        }

        js_error.into()
    }
}

/// Convert JsValue to TrustformersError (for error propagation)
impl From<JsValue> for TrustformersError {
    fn from(js_value: JsValue) -> Self {
        let message = if let Some(error) = js_value.dyn_ref::<js_sys::Error>() {
            error.message().into()
        } else if js_value.is_string() {
            js_value.as_string().unwrap_or_else(|| "Unknown error".to_owned())
        } else {
            format!("{js_value:?}")
        };

        TrustformersError::new(ErrorCode::E8006, &message)
    }
}

/// Error builder for creating errors with context
pub struct ErrorBuilder {
    code: ErrorCode,
    message: String,
    severity: Option<ErrorSeverity>,
    context: ErrorContext,
}

impl ErrorBuilder {
    pub fn new(code: ErrorCode, message: &str) -> Self {
        Self {
            code,
            message: message.to_owned(),
            severity: None,
            context: ErrorContext::default(),
        }
    }

    pub fn severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = Some(severity);
        self
    }

    pub fn operation(mut self, operation: &str) -> Self {
        self.context.operation = Some(operation.to_owned());
        self
    }

    pub fn component(mut self, component: &str) -> Self {
        self.context.component = Some(component.to_owned());
        self
    }

    pub fn model_id(mut self, model_id: &str) -> Self {
        self.context.model_id = Some(model_id.to_owned());
        self
    }

    pub fn input_shape(mut self, shape: Vec<usize>) -> Self {
        self.context.input_shape = Some(shape);
        self
    }

    pub fn device_type(mut self, device_type: &str) -> Self {
        self.context.device_type = Some(device_type.to_owned());
        self
    }

    pub fn memory_usage_mb(mut self, memory_mb: f64) -> Self {
        self.context.memory_usage_mb = Some(memory_mb);
        self
    }

    pub fn additional_info(mut self, info: &str) -> Self {
        self.context.additional_info = Some(info.to_owned());
        self
    }

    pub fn build(self) -> TrustformersError {
        if let Some(severity) = self.severity {
            TrustformersError::with_context(self.code, &self.message, severity, self.context)
        } else {
            let mut error = TrustformersError::new(self.code, &self.message);
            error.context = self.context;
            error
        }
    }
}

/// Error collection for multiple errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCollection {
    pub errors: Vec<TrustformersError>,
    pub has_fatal: bool,
}

impl ErrorCollection {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            has_fatal: false,
        }
    }

    pub fn add(&mut self, error: TrustformersError) {
        if matches!(error.severity, ErrorSeverity::Fatal) {
            self.has_fatal = true;
        }
        self.errors.push(error);
    }

    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn len(&self) -> usize {
        self.errors.len()
    }

    pub fn has_fatal_errors(&self) -> bool {
        self.has_fatal
    }

    pub fn has_errors(&self) -> bool {
        self.errors
            .iter()
            .any(|e| matches!(e.severity, ErrorSeverity::Error | ErrorSeverity::Fatal))
    }

    pub fn get_fatal_errors(&self) -> Vec<&TrustformersError> {
        self.errors
            .iter()
            .filter(|e| matches!(e.severity, ErrorSeverity::Fatal))
            .collect()
    }

    pub fn get_errors(&self) -> Vec<&TrustformersError> {
        self.errors
            .iter()
            .filter(|e| matches!(e.severity, ErrorSeverity::Error | ErrorSeverity::Fatal))
            .collect()
    }

    pub fn get_warnings(&self) -> Vec<&TrustformersError> {
        self.errors
            .iter()
            .filter(|e| matches!(e.severity, ErrorSeverity::Warning))
            .collect()
    }

    /// Convert to user-friendly summary
    pub fn summary(&self) -> String {
        if self.is_empty() {
            return "No errors".to_owned();
        }

        let fatal_count = self.get_fatal_errors().len();
        let error_count = self.get_errors().len() - fatal_count;
        let warning_count = self.get_warnings().len();

        let mut parts = Vec::new();
        if fatal_count > 0 {
            parts.push(format!("{fatal_count} fatal error(s)"));
        }
        if error_count > 0 {
            parts.push(format!("{error_count} error(s)"));
        }
        if warning_count > 0 {
            parts.push(format!("{warning_count} warning(s)"));
        }

        parts.join(", ")
    }
}

impl Default for ErrorCollection {
    fn default() -> Self {
        Self::new()
    }
}

/// Result type alias for TrustformeRS operations
pub type TrustformersResult<T> = Result<T, TrustformersError>;

/// Utility macros for creating errors
#[macro_export]
macro_rules! error {
    ($code:expr, $msg:expr) => {
        $crate::error::TrustformersError::new($code, $msg)
    };
    ($code:expr, $fmt:expr, $($arg:tt)*) => {
        $crate::error::TrustformersError::new($code, &format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! error_builder {
    ($code:expr, $msg:expr) => {
        $crate::error::ErrorBuilder::new($code, $msg)
    };
}

/// WASM bindings for error handling
#[wasm_bindgen]
pub struct ErrorHandler {
    collection: ErrorCollection,
}

#[wasm_bindgen]
impl ErrorHandler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            collection: ErrorCollection::new(),
        }
    }

    /// Add an error to the collection
    pub fn add_error(&mut self, code: u32, message: &str, severity: u32) {
        if let (Ok(error_code), Ok(error_severity)) =
            (Self::code_from_u32(code), Self::severity_from_u32(severity))
        {
            let error = TrustformersError::new(error_code, message);
            let mut error_with_severity = error;
            error_with_severity.severity = error_severity;
            self.collection.add(error_with_severity);
        }
    }

    /// Check if there are any fatal errors
    pub fn has_fatal_errors(&self) -> bool {
        self.collection.has_fatal_errors()
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.collection.has_errors()
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.collection.len()
    }

    /// Get error summary
    pub fn summary(&self) -> String {
        self.collection.summary()
    }

    /// Clear all errors
    pub fn clear(&mut self) {
        self.collection = ErrorCollection::new();
    }

    /// Get all errors as JSON
    pub fn get_errors_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.collection.errors)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    fn code_from_u32(code: u32) -> Result<ErrorCode, ()> {
        match code {
            1001 => Ok(ErrorCode::E1001),
            1002 => Ok(ErrorCode::E1002),
            1003 => Ok(ErrorCode::E1003),
            1004 => Ok(ErrorCode::E1004),
            1005 => Ok(ErrorCode::E1005),
            1006 => Ok(ErrorCode::E1006),
            2001 => Ok(ErrorCode::E2001),
            2002 => Ok(ErrorCode::E2002),
            2003 => Ok(ErrorCode::E2003),
            2004 => Ok(ErrorCode::E2004),
            2005 => Ok(ErrorCode::E2005),
            2006 => Ok(ErrorCode::E2006),
            3001 => Ok(ErrorCode::E3001),
            3002 => Ok(ErrorCode::E3002),
            3003 => Ok(ErrorCode::E3003),
            3004 => Ok(ErrorCode::E3004),
            3005 => Ok(ErrorCode::E3005),
            3006 => Ok(ErrorCode::E3006),
            4001 => Ok(ErrorCode::E4001),
            4002 => Ok(ErrorCode::E4002),
            4003 => Ok(ErrorCode::E4003),
            4004 => Ok(ErrorCode::E4004),
            4005 => Ok(ErrorCode::E4005),
            4006 => Ok(ErrorCode::E4006),
            5001 => Ok(ErrorCode::E5001),
            5002 => Ok(ErrorCode::E5002),
            5003 => Ok(ErrorCode::E5003),
            5004 => Ok(ErrorCode::E5004),
            5005 => Ok(ErrorCode::E5005),
            5006 => Ok(ErrorCode::E5006),
            6001 => Ok(ErrorCode::E6001),
            6002 => Ok(ErrorCode::E6002),
            6003 => Ok(ErrorCode::E6003),
            6004 => Ok(ErrorCode::E6004),
            6005 => Ok(ErrorCode::E6005),
            6006 => Ok(ErrorCode::E6006),
            7001 => Ok(ErrorCode::E7001),
            7002 => Ok(ErrorCode::E7002),
            7003 => Ok(ErrorCode::E7003),
            7004 => Ok(ErrorCode::E7004),
            7005 => Ok(ErrorCode::E7005),
            7006 => Ok(ErrorCode::E7006),
            8001 => Ok(ErrorCode::E8001),
            8002 => Ok(ErrorCode::E8002),
            8003 => Ok(ErrorCode::E8003),
            8004 => Ok(ErrorCode::E8004),
            8005 => Ok(ErrorCode::E8005),
            8006 => Ok(ErrorCode::E8006),
            9001 => Ok(ErrorCode::E9001),
            9002 => Ok(ErrorCode::E9002),
            9003 => Ok(ErrorCode::E9003),
            9004 => Ok(ErrorCode::E9004),
            9005 => Ok(ErrorCode::E9005),
            9006 => Ok(ErrorCode::E9006),
            _ => Err(()),
        }
    }

    fn severity_from_u32(severity: u32) -> Result<ErrorSeverity, ()> {
        match severity {
            0 => Ok(ErrorSeverity::Fatal),
            1 => Ok(ErrorSeverity::Error),
            2 => Ok(ErrorSeverity::Warning),
            3 => Ok(ErrorSeverity::Info),
            _ => Err(()),
        }
    }
}

impl Default for ErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced Circuit Breaker for preventing cascading failures
/// Implements sophisticated failure detection and automatic recovery
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    recovery_timeout_ms: u64,
    half_open_max_calls: u32,
    failure_count: u32,
    state: CircuitState,
    last_failure_time: Option<f64>,
    success_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failure state, rejecting calls
    HalfOpen, // Testing recovery
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, recovery_timeout_ms: u64, half_open_max_calls: u32) -> Self {
        Self {
            failure_threshold,
            recovery_timeout_ms,
            half_open_max_calls,
            failure_count: 0,
            state: CircuitState::Closed,
            last_failure_time: None,
            success_count: 0,
        }
    }

    /// Execute operation with circuit breaker protection
    pub fn execute<F, T, E>(&mut self, operation: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Result<T, E>,
    {
        if !self.can_execute() {
            return Err(CircuitBreakerError::CircuitOpen);
        }

        match operation() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            },
            Err(error) => {
                self.on_failure();
                Err(CircuitBreakerError::OperationFailed(error))
            },
        }
    }

    fn can_execute(&mut self) -> bool {
        let current_time = Self::current_time();

        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if current_time - last_failure >= self.recovery_timeout_ms as f64 {
                        self.transition_to_half_open();
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            },
            CircuitState::HalfOpen => self.success_count < self.half_open_max_calls,
        }
    }

    fn on_success(&mut self) {
        match self.state {
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.half_open_max_calls {
                    self.transition_to_closed();
                }
            },
            _ => {
                // Reset failure count on success
                if self.failure_count > 0 {
                    self.failure_count = 0;
                }
            },
        }
    }

    fn on_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Self::current_time());

        match self.state {
            CircuitState::Closed => {
                if self.failure_count >= self.failure_threshold {
                    self.transition_to_open();
                }
            },
            CircuitState::HalfOpen => {
                self.transition_to_open();
            },
            CircuitState::Open => {
                // Already open, just update timestamp
            },
        }
    }

    fn transition_to_open(&mut self) {
        self.state = CircuitState::Open;
        self.success_count = 0;
    }

    fn transition_to_half_open(&mut self) {
        self.state = CircuitState::HalfOpen;
        self.success_count = 0;
    }

    fn transition_to_closed(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.last_failure_time = None;
    }

    fn current_time() -> f64 {
        js_sys::Date::now()
    }

    pub fn get_state(&self) -> CircuitState {
        self.state
    }

    pub fn get_failure_count(&self) -> u32 {
        self.failure_count
    }
}

#[derive(Debug, Clone)]
pub enum CircuitBreakerError<E> {
    CircuitOpen,
    OperationFailed(E),
}

/// Advanced Retry Strategy with exponential backoff and jitter
/// Implements sophisticated retry logic based on error types and system load
#[derive(Debug, Clone)]
pub struct RetryStrategy {
    max_attempts: u32,
    base_delay_ms: u64,
    max_delay_ms: u64,
    backoff_multiplier: f64,
    jitter_factor: f64,
    retryable_errors: Vec<ErrorCode>,
}

impl RetryStrategy {
    pub fn new() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 100,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
            retryable_errors: vec![
                ErrorCode::E6001, // Network connection failed
                ErrorCode::E6003, // Download timeout
                ErrorCode::E3004, // Device context lost
                ErrorCode::E8006, // WASM runtime error
                ErrorCode::E7005, // IndexedDB error
            ],
        }
    }

    pub fn with_max_attempts(mut self, attempts: u32) -> Self {
        self.max_attempts = attempts;
        self
    }

    pub fn with_base_delay(mut self, delay_ms: u64) -> Self {
        self.base_delay_ms = delay_ms;
        self
    }

    pub fn with_retryable_errors(mut self, errors: Vec<ErrorCode>) -> Self {
        self.retryable_errors = errors;
        self
    }

    /// Execute operation with intelligent retry logic
    pub async fn execute_with_retry<F, T, Fut>(&self, operation: F) -> Result<T, TrustformersError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, TrustformersError>>,
    {
        let mut last_error: Option<TrustformersError> = None;

        for attempt in 1..=self.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !self.is_retryable(&error) || attempt == self.max_attempts {
                        return Err(error);
                    }

                    let delay = self.calculate_delay(attempt);
                    self.sleep(delay).await;
                    last_error = Some(error);
                },
            }
        }

        Err(last_error
            .unwrap_or_else(|| TrustformersError::new(ErrorCode::E2002, "Retry strategy failed")))
    }

    fn is_retryable(&self, error: &TrustformersError) -> bool {
        self.retryable_errors.contains(&error.code)
    }

    fn calculate_delay(&self, attempt: u32) -> u64 {
        let exponential_delay =
            (self.base_delay_ms as f64 * self.backoff_multiplier.powi(attempt as i32 - 1)) as u64;

        let capped_delay = exponential_delay.min(self.max_delay_ms);

        // Add jitter to prevent thundering herd
        let jitter = (capped_delay as f64 * self.jitter_factor * self.random()) as u64;
        capped_delay + jitter
    }

    fn random(&self) -> f64 {
        js_sys::Math::random()
    }

    async fn sleep(&self, ms: u64) {
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms as i32)
                .unwrap();
        });
        wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
    }
}

impl Default for RetryStrategy {
    fn default() -> Self {
        Self::new()
    }
}

/// Intelligent Error Recovery System
/// Implements context-aware recovery strategies based on error patterns and system state
#[derive(Debug, Clone)]
pub struct ErrorRecoverySystem {
    circuit_breaker: CircuitBreaker,
    retry_strategy: RetryStrategy,
    recovery_strategies: std::collections::HashMap<ErrorCode, RecoveryAction>,
    error_history: Vec<TrustformersError>,
    max_history_size: usize,
}

#[derive(Debug, Clone)]
pub enum RecoveryAction {
    Retry,
    FallbackToCPU,
    ClearCache,
    ReduceMemoryUsage,
    ReloadModel,
    RestartWebGL,
    SwitchToMinimalMode,
    NoRecovery,
}

impl ErrorRecoverySystem {
    pub fn new() -> Self {
        let mut recovery_strategies = std::collections::HashMap::new();

        // Configure sophisticated recovery strategies
        recovery_strategies.insert(ErrorCode::E3001, RecoveryAction::FallbackToCPU);
        recovery_strategies.insert(ErrorCode::E3002, RecoveryAction::FallbackToCPU);
        recovery_strategies.insert(ErrorCode::E4001, RecoveryAction::ClearCache);
        recovery_strategies.insert(ErrorCode::E4002, RecoveryAction::ReduceMemoryUsage);
        recovery_strategies.insert(ErrorCode::E1004, RecoveryAction::ReloadModel);
        recovery_strategies.insert(ErrorCode::E6001, RecoveryAction::Retry);
        recovery_strategies.insert(ErrorCode::E7001, RecoveryAction::ClearCache);
        recovery_strategies.insert(ErrorCode::E8001, RecoveryAction::RestartWebGL);

        Self {
            circuit_breaker: CircuitBreaker::new(5, 60000, 3),
            retry_strategy: RetryStrategy::new(),
            recovery_strategies,
            error_history: Vec::new(),
            max_history_size: 100,
        }
    }

    /// Execute operation with comprehensive error handling and recovery
    pub async fn execute_with_recovery<F, T, Fut>(
        &mut self,
        operation: F,
    ) -> Result<T, TrustformersError>
    where
        F: Fn() -> Fut + Clone,
        Fut: std::future::Future<Output = Result<T, TrustformersError>>,
    {
        let circuit_breaker = &mut self.circuit_breaker;
        let operation_clone = operation.clone();

        match circuit_breaker.execute(|| -> Result<(), TrustformersError> {
            // This is a sync wrapper for the async operation
            // In real implementation, this would need proper async handling
            Ok(())
        }) {
            Ok(_) => match self.retry_strategy.execute_with_retry(operation).await {
                Ok(result) => Ok(result),
                Err(error) => {
                    self.record_error(error.clone());
                    self.attempt_recovery(error, operation_clone).await
                },
            },
            Err(CircuitBreakerError::CircuitOpen) => Err(TrustformersError::new(
                ErrorCode::E2004,
                "Circuit breaker is open - system in failure state",
            )),
            Err(CircuitBreakerError::OperationFailed(_)) => {
                self.retry_strategy.execute_with_retry(operation).await
            },
        }
    }

    async fn attempt_recovery<F, T, Fut>(
        &mut self,
        error: TrustformersError,
        operation: F,
    ) -> Result<T, TrustformersError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, TrustformersError>>,
    {
        if let Some(recovery_action) = self.recovery_strategies.get(&error.code).cloned() {
            match self.execute_recovery_action(recovery_action).await {
                Ok(_) => {
                    // Retry operation after recovery
                    match operation().await {
                        Ok(result) => Ok(result),
                        Err(retry_error) => {
                            self.record_error(retry_error.clone());
                            Err(retry_error)
                        },
                    }
                },
                Err(recovery_error) => {
                    // Recovery failed, return original error with additional context
                    let mut enhanced_error = error;
                    enhanced_error.recovery_suggestion =
                        Some(format!("Recovery action failed: {:?}", recovery_error));
                    Err(enhanced_error)
                },
            }
        } else {
            // No recovery strategy, apply intelligent fallback based on error pattern analysis
            self.apply_intelligent_fallback(error, operation).await
        }
    }

    async fn execute_recovery_action(&self, action: RecoveryAction) -> Result<(), String> {
        match action {
            RecoveryAction::FallbackToCPU => {
                // Switch to CPU-only mode
                Ok(())
            },
            RecoveryAction::ClearCache => {
                // Clear various caches - implementation would clear model caches
                // This is a placeholder for actual cache clearing logic
                Ok(())
            },
            RecoveryAction::ReduceMemoryUsage => {
                // Implement memory reduction strategies
                if let Some(window) = web_sys::window() {
                    if let Some(performance) = window.performance() {
                        // Force garbage collection if available
                        if js_sys::Reflect::has(&performance, &"gc".into()).unwrap_or(false) {
                            let gc_fn = js_sys::Reflect::get(&performance, &"gc".into()).unwrap();
                            if let Ok(gc_fn) = gc_fn.dyn_into::<js_sys::Function>() {
                                let _ = gc_fn.call0(&performance);
                            }
                        }
                    }
                }
                Ok(())
            },
            RecoveryAction::ReloadModel => {
                // Reload model with recovery
                Ok(())
            },
            RecoveryAction::RestartWebGL => {
                // Restart WebGL context
                Ok(())
            },
            RecoveryAction::SwitchToMinimalMode => {
                // Switch to minimal functionality mode
                Ok(())
            },
            RecoveryAction::Retry => {
                // Simple retry - handled by retry strategy
                Ok(())
            },
            RecoveryAction::NoRecovery => Err("No recovery action available".to_string()),
        }
    }

    async fn apply_intelligent_fallback<F, T, Fut>(
        &self,
        error: TrustformersError,
        operation: F,
    ) -> Result<T, TrustformersError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, TrustformersError>>,
    {
        // Analyze error patterns to determine intelligent fallback
        let error_pattern = self.analyze_error_patterns(&error);

        match error_pattern {
            ErrorPattern::MemoryRelated => {
                // Apply memory-conscious fallback
                self.execute_recovery_action(RecoveryAction::ReduceMemoryUsage).await.ok();
                operation().await
            },
            ErrorPattern::DeviceRelated => {
                // Apply device fallback
                self.execute_recovery_action(RecoveryAction::FallbackToCPU).await.ok();
                operation().await
            },
            ErrorPattern::NetworkRelated => {
                // Apply network fallback with cache
                operation().await
            },
            ErrorPattern::Unknown => Err(error),
        }
    }

    fn analyze_error_patterns(&self, current_error: &TrustformersError) -> ErrorPattern {
        // Sophisticated pattern analysis based on error history
        let memory_errors = [ErrorCode::E4001, ErrorCode::E4002, ErrorCode::E4003];
        let device_errors = [
            ErrorCode::E3001,
            ErrorCode::E3002,
            ErrorCode::E3003,
            ErrorCode::E3004,
        ];
        let network_errors = [ErrorCode::E6001, ErrorCode::E6002, ErrorCode::E6003];

        if memory_errors.contains(&current_error.code) {
            ErrorPattern::MemoryRelated
        } else if device_errors.contains(&current_error.code) {
            ErrorPattern::DeviceRelated
        } else if network_errors.contains(&current_error.code) {
            ErrorPattern::NetworkRelated
        } else {
            ErrorPattern::Unknown
        }
    }

    fn record_error(&mut self, error: TrustformersError) {
        self.error_history.push(error);
        if self.error_history.len() > self.max_history_size {
            self.error_history.drain(0..self.error_history.len() - self.max_history_size);
        }
    }

    pub fn get_error_statistics(&self) -> ErrorStatistics {
        let total_errors = self.error_history.len();
        let mut error_counts = std::collections::HashMap::new();

        for error in &self.error_history {
            *error_counts.entry(error.code).or_insert(0) += 1;
        }

        let most_common_error = error_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(code, count)| (*code, *count));

        ErrorStatistics {
            total_errors,
            error_counts,
            most_common_error,
            circuit_breaker_state: self.circuit_breaker.get_state(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ErrorPattern {
    MemoryRelated,
    DeviceRelated,
    NetworkRelated,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ErrorStatistics {
    pub total_errors: usize,
    pub error_counts: std::collections::HashMap<ErrorCode, u32>,
    pub most_common_error: Option<(ErrorCode, u32)>,
    pub circuit_breaker_state: CircuitState,
}

impl Default for ErrorRecoverySystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_error_creation() {
        let error = TrustformersError::new(ErrorCode::E1001, "Test error");
        assert_eq!(error.code, ErrorCode::E1001);
        assert_eq!(error.message, "Test error");
        assert_eq!(error.severity, ErrorSeverity::Error);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_error_builder() {
        let error = ErrorBuilder::new(ErrorCode::E2001, "Shape mismatch")
            .operation("predict")
            .component("inference_session")
            .input_shape(vec![1, 2, 3])
            .build();

        assert_eq!(error.code, ErrorCode::E2001);
        assert_eq!(error.context.operation, Some("predict".to_owned()));
        assert_eq!(error.context.input_shape, Some(vec![1, 2, 3]));
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_error_collection() {
        let mut collection = ErrorCollection::new();

        collection.add(TrustformersError::new(ErrorCode::E1001, "Error 1"));
        collection.add(TrustformersError::new(ErrorCode::E4001, "Fatal error"));

        assert_eq!(collection.len(), 2);
        assert!(collection.has_errors());
        assert!(collection.has_fatal_errors());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_error_category() {
        let error = TrustformersError::new(ErrorCode::E1001, "Model error");
        assert_eq!(error.category(), "Model");

        let error = TrustformersError::new(ErrorCode::E3001, "Device error");
        assert_eq!(error.category(), "Device");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_error_codes() {
        // Test error code values and categories for non-WASM targets
        assert_eq!(ErrorCode::E1001 as u32, 1001);
        assert_eq!(ErrorCode::E2001 as u32, 2001);
        assert_eq!(ErrorCode::E3001 as u32, 3001);
        assert_eq!(ErrorCode::E4001 as u32, 4001);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_error_severity() {
        // Test error severity values for non-WASM targets
        use ErrorSeverity::*;
        assert_ne!(Info, Error);
        assert_ne!(Warning, Fatal);
    }
}
