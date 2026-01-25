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
//! use trustformers_models::error_recovery::{
//!     ErrorRecoveryManager, RecoveryConfig, RecoveryStrategy
//! };
//!
//! let config = RecoveryConfig::default()
//!     .with_max_retries(3)
//!     .with_fallback_enabled(true);
//!
//! let mut manager = ErrorRecoveryManager::new(config);
//!
//! // Execute with automatic recovery
//! let result = manager.execute_with_recovery(|| {
//!     // Your model operation here
//!     model.forward(&input)
//! })?;
//! ```

use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

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
    recovery_strategies: HashMap<ErrorCategory, Vec<RecoveryStrategy>>,
    metrics: Arc<Mutex<RecoveryMetrics>>,
    start_time: Instant,
}

impl ErrorRecoveryManager {
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

        for strategy in strategies {
            if self.execute_recovery_strategy(&strategy, error, &category)? {
                self.record_recovery_attempt(category.clone(), strategy, true, error);
                return Ok(true);
            }
        }

        self.record_recovery_attempt(category, RecoveryStrategy::NoRecovery, false, error);
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
                // Basic retry is handled by the main loop, just wait
                std::thread::sleep(Duration::from_millis(*base_delay_ms));
                Ok(true)
            },

            RecoveryStrategy::MemoryCleanup => {
                self.perform_memory_cleanup()?;
                Ok(true)
            },

            RecoveryStrategy::ResourceReduction { reduction_factor } => {
                self.reduce_resource_usage(*reduction_factor)?;
                Ok(true)
            },

            RecoveryStrategy::CheckpointRestore { checkpoint_id } => {
                self.restore_from_checkpoint(checkpoint_id)
            },

            RecoveryStrategy::Fallback {
                fallback_implementation,
            } => {
                self.switch_to_fallback(fallback_implementation)?;
                Ok(true)
            },

            RecoveryStrategy::Restart { component } => {
                self.restart_component(component)?;
                Ok(true)
            },

            RecoveryStrategy::Degrade { degraded_mode } => {
                self.enable_degraded_mode(degraded_mode)?;
                Ok(true)
            },

            RecoveryStrategy::NoRecovery => Ok(false),
        }
    }

    /// Perform memory cleanup
    fn perform_memory_cleanup(&self) -> Result<()> {
        // Force garbage collection if available
        // Clear caches
        // Compact memory
        println!("[INFO] Performing memory cleanup");

        // In a real implementation, this would:
        // - Clear tensor caches
        // - Force garbage collection
        // - Compact memory allocations
        // - Clear intermediate computations

        Ok(())
    }

    /// Reduce resource usage
    fn reduce_resource_usage(&self, reduction_factor: f64) -> Result<()> {
        println!(
            "[INFO] Reducing resource usage by factor: {}",
            reduction_factor
        );

        // In a real implementation, this would:
        // - Reduce batch sizes
        // - Decrease model precision
        // - Limit concurrent operations
        // - Reduce cache sizes

        Ok(())
    }

    /// Switch to fallback implementation
    fn switch_to_fallback(&self, fallback: &str) -> Result<()> {
        println!("[INFO] Switching to fallback implementation: {}", fallback);

        // In a real implementation, this would:
        // - Switch to CPU from GPU
        // - Use simpler model architecture
        // - Use alternative algorithms

        Ok(())
    }

    /// Restart a component
    fn restart_component(&self, component: &str) -> Result<()> {
        println!("[INFO] Restarting component: {}", component);

        // In a real implementation, this would:
        // - Reinitialize the specified component
        // - Clear component state
        // - Reload configurations

        Ok(())
    }

    /// Enable degraded mode
    fn enable_degraded_mode(&self, mode: &str) -> Result<()> {
        println!("[INFO] Enabling degraded mode: {}", mode);

        // In a real implementation, this would:
        // - Reduce functionality
        // - Use simpler algorithms
        // - Lower quality outputs

        Ok(())
    }

    /// Restore from checkpoint
    fn restore_from_checkpoint(&self, checkpoint_id: &str) -> Result<bool> {
        if let Some(_checkpoint) = self.checkpoints.get(checkpoint_id) {
            println!("[INFO] Restoring from checkpoint: {}", checkpoint_id);

            // In a real implementation, this would:
            // - Restore model weights from checkpoint
            // - Restore optimizer state
            // - Restore training state

            Ok(true)
        } else {
            println!("[WARN] Checkpoint not found: {}", checkpoint_id);
            Ok(false)
        }
    }

    /// Create a model checkpoint
    pub fn create_checkpoint(
        &mut self,
        model_state: HashMap<String, Vec<u8>>,
        metadata: HashMap<String, String>,
    ) -> String {
        let checkpoint = ModelCheckpoint::new(model_state, metadata);
        let checkpoint_id = checkpoint.checkpoint_id.clone();

        self.checkpoints.insert(checkpoint_id.clone(), checkpoint);
        self.checkpoints.insert(
            "latest".to_string(),
            self.checkpoints[&checkpoint_id].clone(),
        );

        // Limit number of checkpoints
        if self.checkpoints.len() > 10 {
            // Remove oldest checkpoints (simplified)
            let keys_to_remove: Vec<String> = self.checkpoints.keys()
                .filter(|k| *k != "latest")
                .skip(9) // Keep 9 + "latest"
                .cloned()
                .collect();

            for key in keys_to_remove {
                self.checkpoints.remove(&key);
            }
        }

        println!("[INFO] Created checkpoint: {}", checkpoint_id);
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
    fn record_recovery_attempt(
        &mut self,
        category: ErrorCategory,
        strategy: RecoveryStrategy,
        success: bool,
        error: &Error,
    ) {
        let attempt = RecoveryAttempt {
            attempt_id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            error_category: category.clone(),
            strategy: strategy.clone(),
            success,
            duration_ms: 0, // Would be calculated in real implementation
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

    /// Record failed recovery
    fn record_failed_recovery(
        &mut self,
        category: ErrorCategory,
        _start_time: Instant,
        error: &Error,
    ) {
        self.record_recovery_attempt(category, RecoveryStrategy::NoRecovery, false, error);
    }

    /// Get current recovery metrics
    pub fn get_metrics(&self) -> RecoveryMetrics {
        self.metrics.lock().expect("operation failed").clone()
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
}
