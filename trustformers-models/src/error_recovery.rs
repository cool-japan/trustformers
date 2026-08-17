//! # Comprehensive Error Recovery Framework for TrustformeRS Models
//!
//! This module provides advanced error recovery mechanisms to ensure robust operation
//! of transformer models under various failure conditions.
//!
//! ## Features
//!
//! - **Automatic Retry Strategies**: Configurable retry mechanisms with exponential backoff
//! - **Fallback Execution**: Graceful degradation to simpler model variants
//! - **State Persistence**: Save and restore model state during errors
//! - **Memory Recovery**: Intelligent memory cleanup and reallocation
//! - **Error Classification**: Smart categorization of errors for appropriate response
//! - **Circuit Breaker Pattern**: Prevent cascade failures
//! - **Checkpoint Management**: Automatic model checkpointing for recovery
//! - **Performance Monitoring**: Track recovery effectiveness
//!
//! ## Usage
//!
//! ```rust
//! use trustformers_models::error_recovery::{ErrorRecoveryManager, RecoveryConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = RecoveryConfig::default()
//!     .with_max_retries(3)
//!     .with_fallback_enabled(true);
//!
//! let mut manager = ErrorRecoveryManager::new(config);
//!
//! // Execute with automatic recovery
//! let result = manager.execute_with_recovery(|| -> anyhow::Result<i32> {
//!     // Your model operation here
//!     Ok(42)
//! })?;
//! # let _ = result;
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, warn};
use trustformers_core::traits::Model;
use uuid::Uuid;

/// Key under which a [`ModelCheckpoint`] stores the serialised parameter set.
///
/// The payload is a real safetensors buffer, so a checkpoint can be handed to
/// [`crate::weight_loading::checkpoint::Checkpoint`] — the same reader used for
/// on-disk model files — without any conversion.
pub const CHECKPOINT_STATE_KEY: &str = "model.safetensors";

/// Hooks that perform the actual recovery work.
///
/// [`ErrorRecoveryManager`] owns no model, optimizer or device, so it cannot
/// clean up memory or restore weights by itself. Register a handler to give it
/// the ability to do so; without one, every strategy that needs a handler
/// reports **failure** rather than pretending to have recovered.
///
/// Every method defaults to "this handler does not implement that action", which
/// is reported as a failed recovery — never as success.
pub trait RecoveryHandler: Send + Sync {
    /// Release cached memory (tensor caches, intermediate buffers, ...).
    fn cleanup_memory(&mut self) -> Result<()> {
        Err(anyhow!(
            "this RecoveryHandler does not implement memory cleanup"
        ))
    }

    /// Reduce resource usage (batch size, precision, concurrency) by `factor`.
    fn reduce_resources(&mut self, factor: f64) -> Result<()> {
        let _ = factor;
        Err(anyhow!(
            "this RecoveryHandler does not implement resource reduction"
        ))
    }

    /// Switch to a fallback implementation (for example CPU instead of GPU).
    fn switch_to_fallback(&mut self, implementation: &str) -> Result<()> {
        let _ = implementation;
        Err(anyhow!(
            "this RecoveryHandler does not implement fallback switching"
        ))
    }

    /// Restart a component, clearing its state.
    fn restart_component(&mut self, component: &str) -> Result<()> {
        let _ = component;
        Err(anyhow!(
            "this RecoveryHandler does not implement component restart"
        ))
    }

    /// Enter a degraded operating mode.
    fn enable_degraded_mode(&mut self, mode: &str) -> Result<()> {
        let _ = mode;
        Err(anyhow!(
            "this RecoveryHandler does not implement degraded mode"
        ))
    }

    /// Write the checkpointed state back into the live model / optimizer.
    fn restore_checkpoint(&mut self, checkpoint: &ModelCheckpoint) -> Result<()> {
        let _ = checkpoint;
        Err(anyhow!(
            "this RecoveryHandler does not implement checkpoint restore"
        ))
    }
}

/// A [`RecoveryHandler`] that restores a [`Model`]'s parameters from a checkpoint.
///
/// This is the reference implementation of checkpoint recovery: it decodes the
/// checkpoint's safetensors payload with the production checkpoint reader and
/// copies every tensor back into the live model through
/// [`Model::named_tensors_mut`].
pub struct ModelStateRestorer<'a, M: Model> {
    model: &'a mut M,
}

impl<'a, M: Model> ModelStateRestorer<'a, M> {
    /// Wrap a model so its parameters can be restored from checkpoints.
    pub fn new(model: &'a mut M) -> Self {
        Self { model }
    }
}

impl<M: Model> RecoveryHandler for ModelStateRestorer<'_, M> {
    fn restore_checkpoint(&mut self, checkpoint: &ModelCheckpoint) -> Result<()> {
        let report = checkpoint.restore_into(self.model)?;
        debug!(
            restored = report.restored_tensors,
            checkpoint = %checkpoint.checkpoint_id,
            "restored model parameters from checkpoint"
        );
        Ok(())
    }
}

/// Outcome of restoring a checkpoint into a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreReport {
    /// Number of parameter tensors written back into the model.
    pub restored_tensors: usize,
    /// Names present in the model but missing from the checkpoint.
    pub missing_from_checkpoint: Vec<String>,
    /// Names present in the checkpoint but not in the model.
    pub unused_in_model: Vec<String>,
}

/// Configuration for error recovery behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Base delay for exponential backoff (milliseconds)
    pub base_delay_ms: u64,
    /// Maximum delay between retries (milliseconds)
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Whether to enable fallback strategies
    pub enable_fallback: bool,
    /// Whether to enable automatic checkpointing
    pub enable_checkpointing: bool,
    /// Memory pressure threshold for cleanup (MB)
    pub memory_pressure_threshold_mb: f64,
    /// Circuit breaker failure threshold
    pub circuit_breaker_threshold: usize,
    /// Circuit breaker timeout (seconds)
    pub circuit_breaker_timeout_s: u64,
    /// Whether to enable performance monitoring
    pub enable_monitoring: bool,
    /// Maximum number of error history entries to keep
    pub max_error_history: usize,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            enable_fallback: true,
            enable_checkpointing: true,
            memory_pressure_threshold_mb: 1024.0,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_s: 60,
            enable_monitoring: true,
            max_error_history: 1000,
        }
    }
}

impl RecoveryConfig {
    /// Enable maximum retries
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Enable fallback strategies
    pub fn with_fallback_enabled(mut self, enabled: bool) -> Self {
        self.enable_fallback = enabled;
        self
    }

    /// Set memory pressure threshold
    pub fn with_memory_threshold(mut self, threshold_mb: f64) -> Self {
        self.memory_pressure_threshold_mb = threshold_mb;
        self
    }
}

/// Types of errors that can be recovered from
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Memory-related errors (OOM, allocation failures)
    Memory,
    /// Compute-related errors (CUDA, device failures)
    Compute,
    /// Network-related errors (distributed training)
    Network,
    /// Model-related errors (dimension mismatches, invalid states)
    Model,
    /// Data-related errors (corrupted inputs, invalid tensors)
    Data,
    /// Temporary resource unavailability
    Resource,
    /// Configuration or setup errors
    Configuration,
    /// Unknown or unclassified errors
    Unknown,
}

/// Recovery strategies for different error types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Retry with exponential backoff
    Retry {
        max_attempts: usize,
        base_delay_ms: u64,
    },
    /// Fallback to alternative implementation
    Fallback { fallback_implementation: String },
    /// Reduce resource usage and retry
    ResourceReduction { reduction_factor: f64 },
    /// Restart subsystem
    Restart { component: String },
    /// Clean memory and retry
    MemoryCleanup,
    /// Load from checkpoint
    CheckpointRestore { checkpoint_id: String },
    /// Graceful degradation
    Degrade { degraded_mode: String },
    /// No recovery possible
    NoRecovery,
}

/// Error recovery attempt information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAttempt {
    pub attempt_id: Uuid,
    pub timestamp: SystemTime,
    pub error_category: ErrorCategory,
    pub strategy: RecoveryStrategy,
    pub success: bool,
    pub duration_ms: u64,
    pub error_message: String,
    pub context: HashMap<String, String>,
}

/// Circuit breaker state
#[derive(Debug, Clone, PartialEq)]
enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker for preventing cascade failures
#[derive(Debug)]
struct CircuitBreaker {
    state: CircuitBreakerState,
    failure_count: usize,
    last_failure_time: Option<Instant>,
    failure_threshold: usize,
    timeout: Duration,
}

impl CircuitBreaker {
    fn new(failure_threshold: usize, timeout: Duration) -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            last_failure_time: None,
            failure_threshold,
            timeout,
        }
    }

    fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.timeout {
                        self.state = CircuitBreakerState::HalfOpen;
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            },
            CircuitBreakerState::HalfOpen => true,
        }
    }

    fn on_success(&mut self) {
        self.failure_count = 0;
        self.state = CircuitBreakerState::Closed;
    }

    fn on_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        if self.failure_count >= self.failure_threshold {
            self.state = CircuitBreakerState::Open;
        }
    }
}

/// Model checkpoint for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub checkpoint_id: String,
    pub timestamp: SystemTime,
    pub model_state: HashMap<String, Vec<u8>>, // Serialized tensors
    pub metadata: HashMap<String, String>,
    pub size_bytes: usize,
}

impl ModelCheckpoint {
    /// Create a new checkpoint
    pub fn new(model_state: HashMap<String, Vec<u8>>, metadata: HashMap<String, String>) -> Self {
        let size_bytes = model_state.values().map(|v| v.len()).sum();

        Self {
            checkpoint_id: Uuid::new_v4().to_string(),
            timestamp: SystemTime::now(),
            model_state,
            metadata,
            size_bytes,
        }
    }

    /// Capture a model's live parameters into a checkpoint.
    ///
    /// The parameters are serialised with the real **safetensors** format under
    /// [`CHECKPOINT_STATE_KEY`], so the payload can be written straight to disk or
    /// parsed with the production checkpoint reader.
    ///
    /// Values are stored at `f32` precision. Restoring writes them back through
    /// the target parameter's own dtype, so an `F64` parameter stays `F64` (at
    /// the checkpoint's `f32` precision) instead of silently becoming `F32`.
    ///
    /// # Errors
    ///
    /// Returns an error when the model exposes no named tensors — capturing an
    /// empty checkpoint and calling it a backup would guarantee a silent failure
    /// at restore time.
    pub fn from_model<M: Model>(model: &M, metadata: HashMap<String, String>) -> Result<Self> {
        let named = model.named_tensors();
        if named.is_empty() {
            return Err(anyhow!(
                "cannot checkpoint a model that exposes no named tensors: implement                  Model::named_tensors so the parameters can actually be saved"
            ));
        }

        // Materialise every parameter as little-endian f32 bytes.
        let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(named.len());
        for (name, tensor) in named {
            let shape = tensor.shape();
            let values =
                tensor.data().map_err(|e| anyhow!("failed to read parameter {name}: {e}"))?;
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in &values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            buffers.push((name, shape, bytes));
        }

        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = buffers
            .iter()
            .map(|(name, shape, bytes)| {
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    shape.clone(),
                    bytes.as_slice(),
                )
                .map(|view| (name.clone(), view))
                .map_err(|e| anyhow!("failed to describe parameter {name}: {e}"))
            })
            .collect::<Result<_>>()?;

        let serialized = safetensors::tensor::serialize(views, None)
            .map_err(|e| anyhow!("failed to serialise the model state: {e}"))?;

        let mut model_state = HashMap::new();
        model_state.insert(CHECKPOINT_STATE_KEY.to_string(), serialized);

        Ok(Self::new(model_state, metadata))
    }

    /// Write this checkpoint's parameters back into a live model.
    ///
    /// Tensors are matched by name against [`Model::named_tensors_mut`]; shapes
    /// must agree exactly. Returns a report describing what was written and what
    /// was left over on either side.
    pub fn restore_into<M: Model>(&self, model: &mut M) -> Result<RestoreReport> {
        let payload = self.model_state.get(CHECKPOINT_STATE_KEY).ok_or_else(|| {
            anyhow!(
                "checkpoint {} carries no `{CHECKPOINT_STATE_KEY}` payload, so there is no                  model state to restore",
                self.checkpoint_id
            )
        })?;

        let checkpoint = crate::weight_loading::checkpoint::Checkpoint::from_bytes(payload)
            .map_err(|e| anyhow!("failed to parse the checkpoint payload: {e}"))?;

        let mut targets = model.named_tensors_mut();
        if targets.is_empty() {
            return Err(anyhow!(
                "cannot restore into a model that exposes no named tensors: implement                  Model::named_tensors_mut so the weights can actually be written back"
            ));
        }

        let mut restored = 0usize;
        let mut missing = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for (name, target) in targets.iter_mut() {
            match checkpoint.get(name) {
                Some(source) => {
                    if source.shape() != target.shape() {
                        return Err(anyhow!(
                            "checkpoint tensor {name} has shape {:?} but the model expects {:?}",
                            source.shape(),
                            target.shape()
                        ));
                    }
                    // Write through the shared weight writer so the *target's*
                    // dtype survives: replacing an F64 parameter with the F32
                    // payload would silently change the model's precision.
                    let values = source
                        .data()
                        .map_err(|e| anyhow!("failed to read checkpoint tensor {name}: {e}"))?;
                    crate::model_compression::weight_ops::write_tensor(name, target, &values)
                        .map_err(|e| anyhow!("failed to restore parameter {name}: {e}"))?;
                    restored += 1;
                    seen.insert(name.clone());
                },
                None => missing.push(name.clone()),
            }
        }

        if restored == 0 {
            return Err(anyhow!(
                "checkpoint {} shares no parameter names with the model; nothing was restored",
                self.checkpoint_id
            ));
        }

        let unused = checkpoint
            .names()
            .into_iter()
            .filter(|name| !seen.contains(name))
            .collect::<Vec<_>>();

        Ok(RestoreReport {
            restored_tensors: restored,
            missing_from_checkpoint: missing,
            unused_in_model: unused,
        })
    }
}

/// Recovery performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryMetrics {
    pub total_errors: usize,
    pub successful_recoveries: usize,
    pub failed_recoveries: usize,
    pub average_recovery_time_ms: f64,
    pub recovery_rate: f64,
    pub error_frequency: f64,
    pub most_common_errors: HashMap<ErrorCategory, usize>,
    pub most_effective_strategies: HashMap<String, f64>,
}

/// Main error recovery manager
pub struct ErrorRecoveryManager {
    config: RecoveryConfig,
    error_history: VecDeque<RecoveryAttempt>,
    circuit_breakers: HashMap<String, CircuitBreaker>,
    checkpoints: HashMap<String, ModelCheckpoint>,
    /// Checkpoint ids in creation order, oldest first (used for eviction).
    checkpoint_order: VecDeque<String>,
    /// Id of the most recently created checkpoint, resolved by the `"latest"` alias.
    latest_checkpoint_id: Option<String>,
    recovery_strategies: HashMap<ErrorCategory, Vec<RecoveryStrategy>>,
    /// Caller-supplied hooks that perform the actual recovery work.
    handler: Option<Box<dyn RecoveryHandler>>,
    metrics: Arc<Mutex<RecoveryMetrics>>,
    start_time: Instant,
}

impl ErrorRecoveryManager {
    /// Maximum number of checkpoints retained; the oldest are evicted first.
    pub const MAX_CHECKPOINTS: usize = 10;

    /// Create a new error recovery manager
    pub fn new(config: RecoveryConfig) -> Self {
        let mut recovery_strategies = HashMap::new();

        // Define default recovery strategies for each error category
        recovery_strategies.insert(
            ErrorCategory::Memory,
            vec![
                RecoveryStrategy::MemoryCleanup,
                RecoveryStrategy::ResourceReduction {
                    reduction_factor: 0.5,
                },
                RecoveryStrategy::CheckpointRestore {
                    checkpoint_id: "latest".to_string(),
                },
            ],
        );

        recovery_strategies.insert(
            ErrorCategory::Compute,
            vec![
                RecoveryStrategy::Retry {
                    max_attempts: 3,
                    base_delay_ms: 1000,
                },
                RecoveryStrategy::Fallback {
                    fallback_implementation: "cpu".to_string(),
                },
                RecoveryStrategy::Restart {
                    component: "compute_engine".to_string(),
                },
            ],
        );

        recovery_strategies.insert(
            ErrorCategory::Network,
            vec![
                RecoveryStrategy::Retry {
                    max_attempts: 5,
                    base_delay_ms: 2000,
                },
                RecoveryStrategy::Fallback {
                    fallback_implementation: "local".to_string(),
                },
            ],
        );

        recovery_strategies.insert(
            ErrorCategory::Model,
            vec![
                RecoveryStrategy::CheckpointRestore {
                    checkpoint_id: "latest".to_string(),
                },
                RecoveryStrategy::Degrade {
                    degraded_mode: "simple".to_string(),
                },
                RecoveryStrategy::Restart {
                    component: "model".to_string(),
                },
            ],
        );

        recovery_strategies.insert(
            ErrorCategory::Data,
            vec![
                RecoveryStrategy::Retry {
                    max_attempts: 2,
                    base_delay_ms: 100,
                },
                RecoveryStrategy::Fallback {
                    fallback_implementation: "default_data".to_string(),
                },
            ],
        );

        recovery_strategies.insert(
            ErrorCategory::Resource,
            vec![
                RecoveryStrategy::Retry {
                    max_attempts: 3,
                    base_delay_ms: 5000,
                },
                RecoveryStrategy::ResourceReduction {
                    reduction_factor: 0.7,
                },
            ],
        );

        Self {
            config,
            error_history: VecDeque::new(),
            circuit_breakers: HashMap::new(),
            checkpoints: HashMap::new(),
            checkpoint_order: VecDeque::new(),
            latest_checkpoint_id: None,
            handler: None,
            recovery_strategies,
            metrics: Arc::new(Mutex::new(RecoveryMetrics {
                total_errors: 0,
                successful_recoveries: 0,
                failed_recoveries: 0,
                average_recovery_time_ms: 0.0,
                recovery_rate: 0.0,
                error_frequency: 0.0,
                most_common_errors: HashMap::new(),
                most_effective_strategies: HashMap::new(),
            })),
            start_time: Instant::now(),
        }
    }

    /// Execute a function with automatic error recovery
    pub fn execute_with_recovery<T, F>(&mut self, operation: F) -> Result<T>
    where
        F: Fn() -> Result<T>,
    {
        let operation_name = "default_operation";

        // Check circuit breaker
        if !self.get_or_create_circuit_breaker(operation_name).can_execute() {
            return Err(anyhow::anyhow!(
                "Circuit breaker is open for operation: {}",
                operation_name
            ));
        }

        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            let start_time = Instant::now();

            match operation() {
                Ok(result) => {
                    // Success - update circuit breaker and metrics
                    self.get_or_create_circuit_breaker(operation_name).on_success();

                    if attempt > 0 {
                        // Record successful recovery
                        self.record_successful_recovery(attempt, start_time);
                    }

                    return Ok(result);
                },
                Err(error) => {
                    last_error = Some(anyhow::anyhow!(error.to_string()));

                    // Classify error and attempt recovery
                    let error_category = self.classify_error(&error);
                    let recovery_success = self
                        .attempt_recovery(&error, error_category.clone(), attempt)
                        .unwrap_or(false);

                    if !recovery_success && attempt == self.config.max_retries {
                        // All recovery attempts failed
                        self.get_or_create_circuit_breaker(operation_name).on_failure();
                        self.record_failed_recovery(error_category, start_time, &error);
                        break;
                    }

                    // Wait before retrying (exponential backoff)
                    if attempt < self.config.max_retries {
                        let delay = self.calculate_backoff_delay(attempt);
                        std::thread::sleep(delay);
                    }
                },
            }
        }

        // Return the last error if all attempts failed
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Unknown error occurred")))
    }

    /// Classify an error into a category
    fn classify_error(&self, error: &Error) -> ErrorCategory {
        let error_string = error.to_string().to_lowercase();

        if error_string.contains("memory")
            || error_string.contains("oom")
            || error_string.contains("allocation")
        {
            ErrorCategory::Memory
        } else if error_string.contains("cuda")
            || error_string.contains("gpu")
            || error_string.contains("device")
        {
            ErrorCategory::Compute
        } else if error_string.contains("network")
            || error_string.contains("connection")
            || error_string.contains("timeout")
        {
            ErrorCategory::Network
        } else if error_string.contains("dimension")
            || error_string.contains("shape")
            || error_string.contains("tensor")
        {
            ErrorCategory::Model
        } else if error_string.contains("data")
            || error_string.contains("input")
            || error_string.contains("corrupted")
        {
            ErrorCategory::Data
        } else if error_string.contains("resource")
            || error_string.contains("unavailable")
            || error_string.contains("busy")
        {
            ErrorCategory::Resource
        } else if error_string.contains("config")
            || error_string.contains("setup")
            || error_string.contains("initialization")
        {
            ErrorCategory::Configuration
        } else {
            ErrorCategory::Unknown
        }
    }

    /// Attempt to recover from an error
    fn attempt_recovery(
        &mut self,
        error: &Error,
        category: ErrorCategory,
        _attempt: usize,
    ) -> Result<bool> {
        let strategies = self.recovery_strategies.get(&category).cloned().unwrap_or_else(|| {
            vec![RecoveryStrategy::Retry {
                max_attempts: 1,
                base_delay_ms: 1000,
            }]
        });

        let overall_start = Instant::now();
        for strategy in strategies {
            let started = Instant::now();
            let recovered = self.execute_recovery_strategy(&strategy, error, &category)?;
            if recovered {
                self.record_recovery_attempt(
                    category.clone(),
                    strategy,
                    true,
                    error,
                    started.elapsed(),
                );
                return Ok(true);
            }
        }

        self.record_recovery_attempt(
            category,
            RecoveryStrategy::NoRecovery,
            false,
            error,
            overall_start.elapsed(),
        );
        Ok(false)
    }

    /// Execute a specific recovery strategy
    fn execute_recovery_strategy(
        &mut self,
        strategy: &RecoveryStrategy,
        _error: &Error,
        _category: &ErrorCategory,
    ) -> Result<bool> {
        match strategy {
            RecoveryStrategy::Retry {
                max_attempts: _,
                base_delay_ms,
            } => {
                // Waiting is not a recovery: the retry loop in
                // `execute_with_recovery` is what may fix the problem, and it
                // records its own success. Reporting `true` here would count a
                // sleep as a successful recovery in the metrics.
                std::thread::sleep(Duration::from_millis(*base_delay_ms));
                Ok(false)
            },

            RecoveryStrategy::MemoryCleanup => {
                Ok(self.run_handler_action("memory cleanup", |handler| handler.cleanup_memory()))
            },

            RecoveryStrategy::ResourceReduction { reduction_factor } => {
                let factor = *reduction_factor;
                Ok(self.run_handler_action("resource reduction", |handler| {
                    handler.reduce_resources(factor)
                }))
            },

            RecoveryStrategy::CheckpointRestore { checkpoint_id } => {
                self.restore_from_checkpoint(checkpoint_id)
            },

            RecoveryStrategy::Fallback {
                fallback_implementation,
            } => {
                let implementation = fallback_implementation.clone();
                Ok(self.run_handler_action("fallback switch", |handler| {
                    handler.switch_to_fallback(&implementation)
                }))
            },

            RecoveryStrategy::Restart { component } => {
                let component = component.clone();
                Ok(self.run_handler_action("component restart", |handler| {
                    handler.restart_component(&component)
                }))
            },

            RecoveryStrategy::Degrade { degraded_mode } => {
                let mode = degraded_mode.clone();
                Ok(self.run_handler_action("degraded mode", |handler| {
                    handler.enable_degraded_mode(&mode)
                }))
            },

            RecoveryStrategy::NoRecovery => Ok(false),
        }
    }

    /// Run a recovery action through the registered handler.
    ///
    /// Returns `true` only when a handler is registered **and** it reported
    /// success. With no handler there is nothing that could have recovered, so
    /// the answer is `false` — never a fabricated success.
    fn run_handler_action<F>(&mut self, action: &str, run: F) -> bool
    where
        F: FnOnce(&mut dyn RecoveryHandler) -> Result<()>,
    {
        let Some(handler) = self.handler.as_deref_mut() else {
            warn!(
                action,
                "no RecoveryHandler is registered; the recovery action cannot be performed"
            );
            return false;
        };

        match run(handler) {
            Ok(()) => {
                debug!(action, "recovery action completed");
                true
            },
            Err(error) => {
                warn!(action, %error, "recovery action failed");
                false
            },
        }
    }

    /// Register the hooks that perform the actual recovery work.
    ///
    /// Without a handler the manager can still classify errors, retry and record
    /// metrics, but every strategy that needs to touch the model or the runtime
    /// reports failure.
    pub fn set_handler(&mut self, handler: Box<dyn RecoveryHandler>) {
        self.handler = Some(handler);
    }

    /// Drop the registered recovery handler.
    pub fn clear_handler(&mut self) -> Option<Box<dyn RecoveryHandler>> {
        self.handler.take()
    }

    /// Whether a recovery handler is registered.
    pub fn has_handler(&self) -> bool {
        self.handler.is_some()
    }

    /// Resolve a checkpoint id, mapping the `"latest"` alias onto the most
    /// recently created checkpoint.
    fn resolve_checkpoint_id(&self, checkpoint_id: &str) -> Option<String> {
        if checkpoint_id == "latest" {
            self.latest_checkpoint_id.clone()
        } else if self.checkpoints.contains_key(checkpoint_id) {
            Some(checkpoint_id.to_string())
        } else {
            None
        }
    }

    /// Look up a stored checkpoint (`"latest"` resolves to the newest one).
    pub fn get_checkpoint(&self, checkpoint_id: &str) -> Option<&ModelCheckpoint> {
        self.resolve_checkpoint_id(checkpoint_id)
            .and_then(|id| self.checkpoints.get(&id))
    }

    /// Restore state from a checkpoint through the registered handler.
    ///
    /// Returns `Ok(false)` when the checkpoint does not exist or when no handler
    /// is registered to write the state back: reporting a restored model that was
    /// never restored would make the recovery metrics fiction.
    fn restore_from_checkpoint(&mut self, checkpoint_id: &str) -> Result<bool> {
        let Some(resolved) = self.resolve_checkpoint_id(checkpoint_id) else {
            warn!(checkpoint_id, "checkpoint not found; cannot restore");
            return Ok(false);
        };

        let Some(checkpoint) = self.checkpoints.get(&resolved).cloned() else {
            warn!(checkpoint_id = %resolved, "checkpoint disappeared before it could be restored");
            return Ok(false);
        };

        Ok(self.run_handler_action("checkpoint restore", |handler| {
            handler.restore_checkpoint(&checkpoint)
        }))
    }

    /// Create a model checkpoint
    pub fn create_checkpoint(
        &mut self,
        model_state: HashMap<String, Vec<u8>>,
        metadata: HashMap<String, String>,
    ) -> String {
        self.store_checkpoint(ModelCheckpoint::new(model_state, metadata))
    }

    /// Capture a live model's parameters as a checkpoint.
    ///
    /// # Errors
    ///
    /// Propagates the error from [`ModelCheckpoint::from_model`] when the model
    /// exposes no named tensors.
    pub fn checkpoint_model<M: Model>(
        &mut self,
        model: &M,
        metadata: HashMap<String, String>,
    ) -> Result<String> {
        Ok(self.store_checkpoint(ModelCheckpoint::from_model(model, metadata)?))
    }

    /// Store a checkpoint, evicting the **oldest** ones once the limit is hit.
    fn store_checkpoint(&mut self, checkpoint: ModelCheckpoint) -> String {
        let checkpoint_id = checkpoint.checkpoint_id.clone();

        self.checkpoints.insert(checkpoint_id.clone(), checkpoint);
        self.checkpoint_order.push_back(checkpoint_id.clone());
        self.latest_checkpoint_id = Some(checkpoint_id.clone());

        // Evict oldest-first, in creation order, keeping the newest
        // `max_checkpoints` (the "latest" alias is a pointer, not a copy, so it
        // never occupies a slot of its own).
        while self.checkpoint_order.len() > Self::MAX_CHECKPOINTS {
            if let Some(oldest) = self.checkpoint_order.pop_front() {
                self.checkpoints.remove(&oldest);
                if self.latest_checkpoint_id.as_deref() == Some(oldest.as_str()) {
                    self.latest_checkpoint_id = None;
                }
            }
        }

        debug!(checkpoint = %checkpoint_id, "created checkpoint");
        checkpoint_id
    }

    /// Calculate exponential backoff delay
    fn calculate_backoff_delay(&self, attempt: usize) -> Duration {
        let delay_ms =
            self.config.base_delay_ms as f64 * self.config.backoff_multiplier.powi(attempt as i32);
        let delay_ms = delay_ms.min(self.config.max_delay_ms as f64) as u64;
        Duration::from_millis(delay_ms)
    }

    /// Get or create circuit breaker for an operation
    fn get_or_create_circuit_breaker(&mut self, operation: &str) -> &mut CircuitBreaker {
        self.circuit_breakers.entry(operation.to_string()).or_insert_with(|| {
            CircuitBreaker::new(
                self.config.circuit_breaker_threshold,
                Duration::from_secs(self.config.circuit_breaker_timeout_s),
            )
        })
    }

    /// Record a recovery attempt
    /// Record a recovery attempt, including how long the strategy really took.
    fn record_recovery_attempt(
        &mut self,
        category: ErrorCategory,
        strategy: RecoveryStrategy,
        success: bool,
        error: &Error,
        duration: Duration,
    ) {
        let attempt = RecoveryAttempt {
            attempt_id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            error_category: category.clone(),
            strategy: strategy.clone(),
            success,
            duration_ms: duration.as_millis() as u64,
            error_message: error.to_string(),
            context: HashMap::new(),
        };

        self.error_history.push_back(attempt);

        // Limit history size
        while self.error_history.len() > self.config.max_error_history {
            self.error_history.pop_front();
        }

        // Update metrics
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.total_errors += 1;
            if success {
                metrics.successful_recoveries += 1;
            } else {
                metrics.failed_recoveries += 1;
            }

            metrics.recovery_rate =
                metrics.successful_recoveries as f64 / metrics.total_errors as f64;

            let count = metrics.most_common_errors.entry(category).or_insert(0);
            *count += 1;
        }
    }

    /// Record successful recovery
    fn record_successful_recovery(&mut self, _attempts: usize, start_time: Instant) {
        if let Ok(mut metrics) = self.metrics.lock() {
            let duration = start_time.elapsed().as_millis() as f64;
            let total_recoveries = metrics.successful_recoveries + metrics.failed_recoveries;

            if total_recoveries > 0 {
                metrics.average_recovery_time_ms =
                    (metrics.average_recovery_time_ms * total_recoveries as f64 + duration)
                        / (total_recoveries + 1) as f64;
            } else {
                metrics.average_recovery_time_ms = duration;
            }
        }
    }

    /// Record failed recovery, timing it from when the failing attempt started.
    fn record_failed_recovery(
        &mut self,
        category: ErrorCategory,
        start_time: Instant,
        error: &Error,
    ) {
        self.record_recovery_attempt(
            category,
            RecoveryStrategy::NoRecovery,
            false,
            error,
            start_time.elapsed(),
        );
    }

    /// Get current recovery metrics
    pub fn get_metrics(&self) -> RecoveryMetrics {
        self.metrics.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Generate recovery report
    pub fn generate_recovery_report(&self) -> RecoveryReport {
        let metrics = self.get_metrics();
        let uptime = self.start_time.elapsed();

        let recent_errors: Vec<_> = self.error_history.iter().rev().take(10).cloned().collect();

        let error_trends = self.analyze_error_trends();
        let recommendations = self.generate_recommendations(&metrics, &error_trends);

        RecoveryReport {
            timestamp: SystemTime::now(),
            uptime,
            metrics,
            recent_errors,
            error_trends,
            recommendations,
            circuit_breaker_states: self.get_circuit_breaker_states(),
            checkpoint_count: self.checkpoints.len(),
        }
    }

    /// Analyze error trends
    fn analyze_error_trends(&self) -> ErrorTrends {
        let now = SystemTime::now();
        let one_hour_ago = now.checked_sub(Duration::from_secs(3600)).unwrap_or(now);

        let recent_errors: Vec<_> = self
            .error_history
            .iter()
            .filter(|attempt| attempt.timestamp >= one_hour_ago)
            .collect();

        let error_rate = recent_errors.len() as f64 / 3600.0; // errors per second
        let recovery_success_rate = if !recent_errors.is_empty() {
            recent_errors.iter().filter(|a| a.success).count() as f64 / recent_errors.len() as f64
        } else {
            1.0
        };

        let trending_up = recent_errors.len() > self.error_history.len() / 2;

        ErrorTrends {
            error_rate,
            recovery_success_rate,
            trending_up,
            most_frequent_category: self.get_most_frequent_error_category(&recent_errors),
        }
    }

    /// Get most frequent error category
    fn get_most_frequent_error_category(
        &self,
        errors: &[&RecoveryAttempt],
    ) -> Option<ErrorCategory> {
        let mut category_counts = HashMap::new();

        for error in errors {
            let count = category_counts.entry(error.error_category.clone()).or_insert(0);
            *count += 1;
        }

        category_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(category, _)| category)
    }

    /// Generate recommendations based on metrics and trends
    fn generate_recommendations(
        &self,
        metrics: &RecoveryMetrics,
        trends: &ErrorTrends,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if metrics.recovery_rate < 0.8 {
            recommendations
                .push("Recovery rate is low. Consider reviewing recovery strategies.".to_string());
        }

        if trends.error_rate > 0.1 {
            recommendations.push("High error rate detected. Investigate root causes.".to_string());
        }

        if trends.trending_up {
            recommendations
                .push("Error frequency is increasing. Monitor system closely.".to_string());
        }

        if metrics.average_recovery_time_ms > 5000.0 {
            recommendations
                .push("Recovery time is high. Optimize recovery strategies.".to_string());
        }

        if let Some(category) = &trends.most_frequent_category {
            recommendations.push(format!(
                "Most frequent error category: {:?}. Focus optimization efforts here.",
                category
            ));
        }

        if recommendations.is_empty() {
            recommendations.push("Error recovery system is operating normally.".to_string());
        }

        recommendations
    }

    /// Get circuit breaker states
    fn get_circuit_breaker_states(&self) -> HashMap<String, String> {
        self.circuit_breakers
            .iter()
            .map(|(name, breaker)| {
                let state = match breaker.state {
                    CircuitBreakerState::Closed => "CLOSED",
                    CircuitBreakerState::Open => "OPEN",
                    CircuitBreakerState::HalfOpen => "HALF_OPEN",
                };
                (name.clone(), state.to_string())
            })
            .collect()
    }
}

/// Error trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTrends {
    pub error_rate: f64,
    pub recovery_success_rate: f64,
    pub trending_up: bool,
    pub most_frequent_category: Option<ErrorCategory>,
}

/// Comprehensive recovery report
#[derive(Debug, Serialize, Deserialize)]
pub struct RecoveryReport {
    pub timestamp: SystemTime,
    pub uptime: Duration,
    pub metrics: RecoveryMetrics,
    pub recent_errors: Vec<RecoveryAttempt>,
    pub error_trends: ErrorTrends,
    pub recommendations: Vec<String>,
    pub circuit_breaker_states: HashMap<String, String>,
    pub checkpoint_count: usize,
}

/// Convenience trait for adding recovery capabilities to any operation
pub trait RecoverableOperation<T> {
    fn with_recovery(self, manager: &mut ErrorRecoveryManager) -> Result<T>;
}

impl<T, F> RecoverableOperation<T> for F
where
    F: Fn() -> Result<T>,
{
    fn with_recovery(self, manager: &mut ErrorRecoveryManager) -> Result<T> {
        manager.execute_with_recovery(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_classification() {
        let manager = ErrorRecoveryManager::new(RecoveryConfig::default());

        let memory_error = anyhow::anyhow!("Out of memory error occurred");
        assert_eq!(manager.classify_error(&memory_error), ErrorCategory::Memory);

        let cuda_error = anyhow::anyhow!("CUDA device error");
        assert_eq!(manager.classify_error(&cuda_error), ErrorCategory::Compute);
    }

    #[test]
    fn test_circuit_breaker() {
        let mut breaker = CircuitBreaker::new(2, Duration::from_secs(1));

        assert!(breaker.can_execute());

        breaker.on_failure();
        assert!(breaker.can_execute());

        breaker.on_failure();
        assert!(!breaker.can_execute()); // Should be open now

        breaker.on_success();
        assert!(breaker.can_execute()); // Should be closed again
    }

    #[test]
    fn test_backoff_calculation() {
        let config = RecoveryConfig::default();
        let manager = ErrorRecoveryManager::new(config);

        let delay0 = manager.calculate_backoff_delay(0);
        let delay1 = manager.calculate_backoff_delay(1);
        let delay2 = manager.calculate_backoff_delay(2);

        assert!(delay1 > delay0);
        assert!(delay2 > delay1);
    }

    #[test]
    fn test_recovery_config_builder() {
        let config = RecoveryConfig::default()
            .with_max_retries(5)
            .with_fallback_enabled(false)
            .with_memory_threshold(2048.0);

        assert_eq!(config.max_retries, 5);
        assert!(!config.enable_fallback);
        assert_eq!(config.memory_pressure_threshold_mb, 2048.0);
    }

    // -----------------------------------------------------------------------
    // Regression tests: recovery must do real work or report failure
    // -----------------------------------------------------------------------

    use serde::{Deserialize, Serialize};
    use std::io::Read;
    use trustformers_core::tensor::Tensor;
    use trustformers_core::traits::Config;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TinyConfig;

    impl Config for TinyConfig {
        fn architecture(&self) -> &'static str {
            "tiny"
        }
    }

    /// A minimal model that really exposes its parameters.
    struct TinyModel {
        config: TinyConfig,
        weight: Tensor,
        bias: Tensor,
    }

    impl TinyModel {
        fn new(weight: &[f32], bias: &[f32]) -> Self {
            Self {
                config: TinyConfig,
                weight: Tensor::from_slice(weight, &[2, 2]).expect("weight"),
                bias: Tensor::from_slice(bias, &[2]).expect("bias"),
            }
        }
    }

    impl Model for TinyModel {
        type Config = TinyConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Tensor) -> trustformers_core::Result<Tensor> {
            input.matmul(&self.weight)?.add(&self.bias)
        }

        fn load_pretrained(&mut self, _reader: &mut dyn Read) -> trustformers_core::Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &TinyConfig {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            6
        }

        fn named_tensors(&self) -> Vec<(String, &Tensor)> {
            vec![
                ("weight".to_string(), &self.weight),
                ("bias".to_string(), &self.bias),
            ]
        }

        fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
            vec![
                ("weight".to_string(), &mut self.weight),
                ("bias".to_string(), &mut self.bias),
            ]
        }
    }

    /// A model that never exposes its parameters (the trait default).
    struct OpaqueModel {
        config: TinyConfig,
    }

    impl Model for OpaqueModel {
        type Config = TinyConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Tensor) -> trustformers_core::Result<Tensor> {
            Ok(input)
        }

        fn load_pretrained(&mut self, _reader: &mut dyn Read) -> trustformers_core::Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &TinyConfig {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_checkpoint_round_trip_restores_real_weights() {
        let original = TinyModel::new(&[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5]);
        let checkpoint =
            ModelCheckpoint::from_model(&original, HashMap::new()).expect("checkpoint capture");

        // The payload is a real safetensors buffer.
        let payload = checkpoint
            .model_state
            .get(CHECKPOINT_STATE_KEY)
            .expect("checkpoint payload must exist");
        assert!(checkpoint.size_bytes >= 6 * 4);
        assert!(
            crate::weight_loading::checkpoint::detect_format(payload).is_ok(),
            "the checkpoint payload must be readable by the production checkpoint reader"
        );

        // A model with different weights gets the checkpointed values back.
        let mut restored = TinyModel::new(&[0.0, 0.0, 0.0, 0.0], &[0.0, 0.0]);
        let report = checkpoint.restore_into(&mut restored).expect("restore");
        assert_eq!(report.restored_tensors, 2);
        assert!(report.missing_from_checkpoint.is_empty());
        assert!(report.unused_in_model.is_empty());

        assert_eq!(
            restored.weight.data().expect("weights"),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(restored.bias.data().expect("bias"), vec![0.5, -0.5]);

        // ... and the restored model computes what the original computed.
        let input = Tensor::from_slice(&[1.0, 1.0], &[1, 2]).expect("input");
        let expected = original.forward(input.clone()).expect("forward").data().expect("data");
        let actual = restored.forward(input).expect("forward").data().expect("data");
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_checkpoint_refuses_models_without_named_tensors() {
        let opaque = OpaqueModel { config: TinyConfig };
        let error = ModelCheckpoint::from_model(&opaque, HashMap::new())
            .expect_err("a model with no named tensors cannot be checkpointed");
        assert!(error.to_string().contains("named tensors"), "{error}");

        let good = TinyModel::new(&[1.0, 2.0, 3.0, 4.0], &[0.0, 0.0]);
        let checkpoint = ModelCheckpoint::from_model(&good, HashMap::new()).expect("capture");
        let mut opaque = OpaqueModel { config: TinyConfig };
        let error = checkpoint
            .restore_into(&mut opaque)
            .expect_err("a model with no named tensors cannot be restored into");
        assert!(error.to_string().contains("named tensors"), "{error}");
    }

    #[test]
    fn test_restore_rejects_shape_mismatch() {
        let source = TinyModel::new(&[1.0, 2.0, 3.0, 4.0], &[0.0, 0.0]);
        let checkpoint = ModelCheckpoint::from_model(&source, HashMap::new()).expect("capture");

        struct WrongShape {
            config: TinyConfig,
            weight: Tensor,
        }
        impl Model for WrongShape {
            type Config = TinyConfig;
            type Input = Tensor;
            type Output = Tensor;
            fn forward(&self, input: Tensor) -> trustformers_core::Result<Tensor> {
                Ok(input)
            }
            fn load_pretrained(&mut self, _reader: &mut dyn Read) -> trustformers_core::Result<()> {
                Ok(())
            }
            fn get_config(&self) -> &TinyConfig {
                &self.config
            }
            fn num_parameters(&self) -> usize {
                3
            }
            fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
                vec![("weight".to_string(), &mut self.weight)]
            }
        }

        let mut target = WrongShape {
            config: TinyConfig,
            weight: Tensor::zeros(&[3, 1]).expect("weight"),
        };
        let error = checkpoint.restore_into(&mut target).expect_err("shape mismatch must fail");
        assert!(error.to_string().contains("shape"), "{error}");
    }

    #[test]
    fn test_checkpoint_restore_strategy_without_handler_reports_failure() {
        let mut manager = ErrorRecoveryManager::new(RecoveryConfig::default());
        let model = TinyModel::new(&[1.0, 2.0, 3.0, 4.0], &[0.0, 0.0]);
        let id = manager.checkpoint_model(&model, HashMap::new()).expect("checkpoint");

        let error = anyhow::anyhow!("out of memory");
        let recovered = manager
            .execute_recovery_strategy(
                &RecoveryStrategy::CheckpointRestore {
                    checkpoint_id: id.clone(),
                },
                &error,
                &ErrorCategory::Memory,
            )
            .expect("strategy execution");
        assert!(
            !recovered,
            "with no handler registered nothing was restored, so recovery must report failure"
        );

        // The same is true for every other handler-backed strategy.
        for strategy in [
            RecoveryStrategy::MemoryCleanup,
            RecoveryStrategy::ResourceReduction {
                reduction_factor: 0.5,
            },
            RecoveryStrategy::Fallback {
                fallback_implementation: "cpu".to_string(),
            },
            RecoveryStrategy::Restart {
                component: "device".to_string(),
            },
            RecoveryStrategy::Degrade {
                degraded_mode: "low".to_string(),
            },
        ] {
            let recovered = manager
                .execute_recovery_strategy(&strategy, &error, &ErrorCategory::Memory)
                .expect("strategy execution");
            assert!(
                !recovered,
                "{strategy:?} must not report success without a handler"
            );
        }
    }

    #[test]
    fn test_checkpoint_restore_strategy_with_handler_restores_state() {
        /// Handler that records the restored weights.
        struct RecordingHandler {
            restored: Arc<Mutex<Vec<f32>>>,
        }

        impl RecoveryHandler for RecordingHandler {
            fn restore_checkpoint(&mut self, checkpoint: &ModelCheckpoint) -> Result<()> {
                let mut model = TinyModel::new(&[0.0, 0.0, 0.0, 0.0], &[0.0, 0.0]);
                checkpoint.restore_into(&mut model)?;
                let mut slot =
                    self.restored.lock().map_err(|_| anyhow!("recording mutex poisoned"))?;
                *slot = model.weight.data()?;
                Ok(())
            }
        }

        let restored = Arc::new(Mutex::new(Vec::new()));
        let mut manager = ErrorRecoveryManager::new(RecoveryConfig::default());
        manager.set_handler(Box::new(RecordingHandler {
            restored: restored.clone(),
        }));

        let model = TinyModel::new(&[9.0, 8.0, 7.0, 6.0], &[0.0, 0.0]);
        manager.checkpoint_model(&model, HashMap::new()).expect("checkpoint");

        let recovered = manager
            .execute_recovery_strategy(
                &RecoveryStrategy::CheckpointRestore {
                    checkpoint_id: "latest".to_string(),
                },
                &anyhow::anyhow!("out of memory"),
                &ErrorCategory::Memory,
            )
            .expect("strategy execution");

        assert!(
            recovered,
            "a registered handler that succeeds means real recovery"
        );
        assert_eq!(
            *restored.lock().expect("recording mutex"),
            vec![9.0, 8.0, 7.0, 6.0]
        );
    }

    #[test]
    fn test_failing_handler_is_not_reported_as_recovery() {
        struct FailingHandler;
        impl RecoveryHandler for FailingHandler {
            fn cleanup_memory(&mut self) -> Result<()> {
                Err(anyhow!("device is wedged"))
            }
        }

        let mut manager = ErrorRecoveryManager::new(RecoveryConfig::default());
        manager.set_handler(Box::new(FailingHandler));

        let recovered = manager
            .execute_recovery_strategy(
                &RecoveryStrategy::MemoryCleanup,
                &anyhow::anyhow!("out of memory"),
                &ErrorCategory::Memory,
            )
            .expect("strategy execution");
        assert!(!recovered);
    }

    #[test]
    fn test_retry_alone_is_not_reported_as_a_recovery() {
        let mut manager = ErrorRecoveryManager::new(RecoveryConfig::default());
        let recovered = manager
            .execute_recovery_strategy(
                &RecoveryStrategy::Retry {
                    max_attempts: 3,
                    base_delay_ms: 1,
                },
                &anyhow::anyhow!("transient failure"),
                &ErrorCategory::Network,
            )
            .expect("strategy execution");
        assert!(
            !recovered,
            "waiting before a retry is not a recovery; only the retry itself can be"
        );
    }

    #[test]
    fn test_restore_preserves_the_target_dtype() {
        use trustformers_core::tensor::DType;

        let source = TinyModel::new(&[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5]);
        let checkpoint = ModelCheckpoint::from_model(&source, HashMap::new()).expect("capture");

        let mut target = TinyModel::new(&[0.0, 0.0, 0.0, 0.0], &[0.0, 0.0]);
        target.weight =
            Tensor::from_vec_with_dtype(vec![0.0; 4], &[2, 2], DType::F64).expect("f64 weights");

        checkpoint.restore_into(&mut target).expect("restore");

        assert_eq!(
            target.weight.dtype(),
            DType::F64,
            "restoring must not silently change an F64 parameter to F32"
        );
        assert_eq!(
            target.weight.data().expect("weights"),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_checkpoint_eviction_is_oldest_first() {
        let mut manager = ErrorRecoveryManager::new(RecoveryConfig::default());

        let mut ids = Vec::new();
        for i in 0..(ErrorRecoveryManager::MAX_CHECKPOINTS + 3) {
            let mut metadata = HashMap::new();
            metadata.insert("index".to_string(), i.to_string());
            ids.push(manager.create_checkpoint(HashMap::new(), metadata));
        }

        // The three oldest checkpoints are gone; every newer one survives.
        for old in ids.iter().take(3) {
            assert!(
                manager.get_checkpoint(old).is_none(),
                "the oldest checkpoints must be evicted first"
            );
        }
        for kept in ids.iter().skip(3) {
            assert!(
                manager.get_checkpoint(kept).is_some(),
                "newer checkpoints must survive eviction"
            );
        }

        // "latest" resolves to the newest checkpoint, not to a stale copy.
        let latest = manager.get_checkpoint("latest").expect("latest checkpoint");
        let newest_id = ids.last().expect("at least one checkpoint");
        assert_eq!(&latest.checkpoint_id, newest_id);
        assert_eq!(latest.metadata.get("index").map(String::as_str), Some("12"));
    }
}
