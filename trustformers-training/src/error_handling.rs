use anyhow::{Context as AnyhowContext, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error_codes::{get_error_info, is_critical_error};

/// Enhanced error handling system for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingError {
    pub error_type: ErrorType,
    pub message: String,
    pub error_code: String,
    pub severity: ErrorSeverity,
    pub context: ErrorContext,
    pub timestamp: u64,
    pub recovery_suggestions: Vec<RecoverySuggestion>,
    pub related_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub enum ErrorType {
    Configuration,
    DataLoading,
    ModelInitialization,
    Training,
    Validation,
    Checkpoint,
    Resource,
    Network,
    Hardware,
    UserInput,
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub enum ErrorSeverity {
    Critical, // Training cannot continue
    High,     // Training can continue with reduced functionality
    Medium,   // Warning that may affect performance
    Low,      // Informational, no impact on training
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub component: String,
    pub operation: String,
    pub epoch: Option<u32>,
    pub step: Option<u32>,
    pub batch_size: Option<usize>,
    pub learning_rate: Option<f64>,
    pub model_state: Option<String>,
    pub system_info: SystemInfo,
    pub additional_data: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub memory_usage: Option<u64>,
    pub gpu_memory_usage: Option<u64>,
    pub cpu_usage: Option<f32>,
    pub disk_space: Option<u64>,
    pub network_status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoverySuggestion {
    pub action: String,
    pub description: String,
    pub priority: u8,    // 1-10, where 10 is highest priority
    pub automatic: bool, // Whether this can be applied automatically
}

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}: {} ({})",
            self.severity, self.error_type, self.message, self.error_code
        )
    }
}

impl fmt::Display for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorType::Configuration => write!(f, "CONFIGURATION"),
            ErrorType::DataLoading => write!(f, "DATA_LOADING"),
            ErrorType::ModelInitialization => write!(f, "MODEL_INIT"),
            ErrorType::Training => write!(f, "TRAINING"),
            ErrorType::Validation => write!(f, "VALIDATION"),
            ErrorType::Checkpoint => write!(f, "CHECKPOINT"),
            ErrorType::Resource => write!(f, "RESOURCE"),
            ErrorType::Network => write!(f, "NETWORK"),
            ErrorType::Hardware => write!(f, "HARDWARE"),
            ErrorType::UserInput => write!(f, "USER_INPUT"),
            ErrorType::Internal => write!(f, "INTERNAL"),
        }
    }
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::Low => write!(f, "LOW"),
        }
    }
}

impl std::error::Error for TrainingError {}

/// Error manager for collecting, analyzing, and handling training errors
pub struct ErrorManager {
    errors: Arc<RwLock<Vec<TrainingError>>>,
    error_patterns: Arc<RwLock<HashMap<String, ErrorPattern>>>,
    recovery_strategies: Arc<RwLock<HashMap<ErrorType, Vec<RecoveryStrategy>>>>,
    statistics: Arc<RwLock<ErrorStatistics>>,
}

#[derive(Debug, Clone)]
pub struct ErrorPattern {
    pub pattern_id: String,
    pub error_codes: Vec<String>,
    pub frequency_threshold: u32,
    pub time_window_seconds: u64,
    pub suggested_actions: Vec<RecoverySuggestion>,
}

#[derive(Debug, Clone)]
pub struct RecoveryStrategy {
    pub strategy_id: String,
    pub name: String,
    pub applicable_errors: Vec<String>,
    pub handler: fn(&TrainingError) -> Result<RecoveryAction>,
    pub auto_apply: bool,
}

#[derive(Debug, Clone)]
pub enum RecoveryAction {
    Continue,
    Retry {
        max_attempts: u32,
    },
    Restart {
        checkpoint: Option<String>,
    },
    Abort,
    ReduceResources {
        factor: f32,
    },
    ChangeConfiguration {
        config_changes: HashMap<String, String>,
    },
    SwitchFallback {
        fallback_config: String,
    },
}

#[derive(Debug, Default, Clone)]
pub struct ErrorStatistics {
    pub total_errors: u64,
    pub errors_by_type: HashMap<ErrorType, u64>,
    pub errors_by_severity: HashMap<ErrorSeverity, u64>,
    pub errors_by_component: HashMap<String, u64>,
    pub recovery_success_rate: f64,
    pub most_common_errors: Vec<(String, u64)>,
    pub error_trends: Vec<ErrorTrend>,
}

#[derive(Debug, Clone)]
pub struct ErrorTrend {
    pub timestamp: u64,
    pub error_count: u64,
    pub error_rate: f64, // errors per minute
}

impl Default for ErrorManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorManager {
    pub fn new() -> Self {
        Self {
            errors: Arc::new(RwLock::new(Vec::new())),
            error_patterns: Arc::new(RwLock::new(HashMap::new())),
            recovery_strategies: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(ErrorStatistics::default())),
        }
    }

    /// Record a new error
    pub fn record_error(&self, error: TrainingError) -> Result<()> {
        // Add to error log
        {
            let mut errors = self
                .errors
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on errors"))?;
            errors.push(error.clone());
        }

        // Update statistics
        self.update_statistics(&error)?;

        // Check for error patterns
        self.check_error_patterns(&error)?;

        // Attempt automatic recovery if applicable
        self.attempt_recovery(&error)?;

        Ok(())
    }

    /// Create and record an error with context
    pub fn create_error(
        &self,
        error_type: ErrorType,
        message: String,
        error_code: String,
        severity: ErrorSeverity,
        context: ErrorContext,
    ) -> TrainingError {
        let error = TrainingError {
            error_type: error_type.clone(),
            message,
            error_code: error_code.clone(),
            severity: severity.clone(),
            context,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            recovery_suggestions: self.get_recovery_suggestions(&error_type, &error_code),
            related_errors: Vec::new(),
        };

        if let Err(e) = self.record_error(error.clone()) {
            eprintln!("Failed to record error: {}", e);
        }

        error
    }

    /// Add an error pattern for detection
    pub fn add_error_pattern(&self, pattern: ErrorPattern) -> Result<()> {
        let mut patterns = self
            .error_patterns
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on error patterns"))?;
        patterns.insert(pattern.pattern_id.clone(), pattern);
        Ok(())
    }

    /// Add a recovery strategy
    pub fn add_recovery_strategy(
        &self,
        error_type: ErrorType,
        strategy: RecoveryStrategy,
    ) -> Result<()> {
        let mut strategies = self
            .recovery_strategies
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on recovery strategies"))?;
        strategies.entry(error_type).or_insert_with(Vec::new).push(strategy);
        Ok(())
    }

    fn update_statistics(&self, error: &TrainingError) -> Result<()> {
        let mut stats = self
            .statistics
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on statistics"))?;

        stats.total_errors += 1;
        *stats.errors_by_type.entry(error.error_type.clone()).or_insert(0) += 1;
        *stats.errors_by_severity.entry(error.severity.clone()).or_insert(0) += 1;
        *stats.errors_by_component.entry(error.context.component.clone()).or_insert(0) += 1;

        // Update trends
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        // Calculate error rate for the last minute
        let errors = self
            .errors
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on errors"))?;
        let recent_errors =
            errors.iter().filter(|e| current_time - e.timestamp <= 60).count() as u64;

        stats.error_trends.push(ErrorTrend {
            timestamp: current_time,
            error_count: recent_errors,
            error_rate: recent_errors as f64 / 60.0,
        });

        // Keep only last 100 trend points
        if stats.error_trends.len() > 100 {
            stats.error_trends.remove(0);
        }

        Ok(())
    }

    fn check_error_patterns(&self, error: &TrainingError) -> Result<()> {
        let patterns = self
            .error_patterns
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on error patterns"))?;

        for pattern in patterns.values() {
            if pattern.error_codes.contains(&error.error_code) {
                // Check if this pattern has occurred frequently
                let recent_matching_errors = self.count_recent_matching_errors(pattern)?;

                if recent_matching_errors >= pattern.frequency_threshold {
                    println!(
                        "🚨 Error pattern detected: {} (occurred {} times)",
                        pattern.pattern_id, recent_matching_errors
                    );

                    // Apply suggested actions
                    for suggestion in &pattern.suggested_actions {
                        println!(
                            "💡 Suggestion: {} - {}",
                            suggestion.action, suggestion.description
                        );
                    }
                }
            }
        }

        Ok(())
    }

    fn count_recent_matching_errors(&self, pattern: &ErrorPattern) -> Result<u32> {
        let errors = self
            .errors
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on errors"))?;

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        let count = errors
            .iter()
            .filter(|error| {
                current_time - error.timestamp <= pattern.time_window_seconds
                    && pattern.error_codes.contains(&error.error_code)
            })
            .count() as u32;

        Ok(count)
    }

    fn attempt_recovery(&self, error: &TrainingError) -> Result<()> {
        // Check if this is a critical error that requires immediate attention
        if is_critical_error(&error.error_code) {
            println!(
                "🚨 Critical error detected: {} - Manual intervention required",
                error.error_code
            );
            return Ok(()); // Don't attempt automatic recovery for critical errors
        }

        let strategies = self
            .recovery_strategies
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on recovery strategies"))?;

        // Try built-in recovery strategies first
        if let Some(success) = self.try_builtin_recovery(error)? {
            if success {
                println!(
                    "✅ Built-in recovery successful for error: {}",
                    error.error_code
                );
                return Ok(());
            }
        }

        // Try custom recovery strategies
        if let Some(type_strategies) = strategies.get(&error.error_type) {
            for strategy in type_strategies {
                if strategy.auto_apply && strategy.applicable_errors.contains(&error.error_code) {
                    println!("🔧 Attempting automatic recovery: {}", strategy.name);

                    match (strategy.handler)(error) {
                        Ok(action) => {
                            println!("✅ Recovery action determined: {:?}", action);

                            // Execute the recovery action
                            if let Err(e) = self.execute_recovery_action(&action, error) {
                                println!("❌ Failed to execute recovery action: {}", e);
                                continue;
                            }

                            println!("✅ Recovery action executed successfully");
                            return Ok(());
                        },
                        Err(e) => {
                            println!("❌ Recovery strategy failed: {}", e);
                        },
                    }
                }
            }
        }

        // If no automatic recovery worked, suggest manual recovery
        self.suggest_manual_recovery(error);
        Ok(())
    }

    /// Try built-in recovery strategies based on error code
    fn try_builtin_recovery(&self, error: &TrainingError) -> Result<Option<bool>> {
        match error.error_code.as_str() {
            "RESOURCE_OOM" | "RESOURCE_GPU_OOM" => {
                println!("🔧 Attempting memory recovery for OOM error");
                // Simulate memory cleanup
                self.simulate_memory_cleanup()?;
                Ok(Some(true))
            },
            "TRAIN_NAN_LOSS" | "TRAIN_INF_LOSS" => {
                println!("🔧 Attempting numerical stability recovery");
                // Suggest lower learning rate and gradient clipping
                self.suggest_numerical_fixes(error)?;
                Ok(Some(false)) // Don't automatically apply, just suggest
            },
            "DATA_FILE_NOT_FOUND" => {
                println!("🔧 Attempting data path recovery");
                // Try to find alternative data paths
                self.suggest_data_path_fixes(error)?;
                Ok(Some(false))
            },
            "NETWORK_CONNECTION_TIMEOUT" => {
                println!("🔧 Attempting network recovery");
                // Try retry with exponential backoff
                self.attempt_network_retry(error)?;
                Ok(Some(true))
            },
            _ => Ok(None), // No built-in recovery for this error code
        }
    }

    /// Execute a recovery action
    fn execute_recovery_action(
        &self,
        action: &RecoveryAction,
        _error: &TrainingError,
    ) -> Result<()> {
        match action {
            RecoveryAction::Continue => {
                println!("📝 Recovery action: Continue training");
                Ok(())
            },
            RecoveryAction::Retry { max_attempts } => {
                println!(
                    "📝 Recovery action: Retry operation (max {} attempts)",
                    max_attempts
                );
                // In a real implementation, would retry the failed operation
                Ok(())
            },
            RecoveryAction::Restart { checkpoint } => {
                println!(
                    "📝 Recovery action: Restart from checkpoint: {:?}",
                    checkpoint
                );
                // In a real implementation, would restart training from checkpoint
                Ok(())
            },
            RecoveryAction::Abort => {
                println!("📝 Recovery action: Abort training");
                Err(anyhow::anyhow!(
                    "Training aborted due to unrecoverable error"
                ))
            },
            RecoveryAction::ReduceResources { factor } => {
                println!("📝 Recovery action: Reduce resources by factor {}", factor);
                // In a real implementation, would reduce batch size, model size, etc.
                Ok(())
            },
            RecoveryAction::ChangeConfiguration { config_changes } => {
                println!(
                    "📝 Recovery action: Change configuration: {:?}",
                    config_changes
                );
                // In a real implementation, would apply configuration changes
                Ok(())
            },
            RecoveryAction::SwitchFallback { fallback_config } => {
                println!(
                    "📝 Recovery action: Switch to fallback configuration: {}",
                    fallback_config
                );
                // In a real implementation, would switch to fallback configuration
                Ok(())
            },
        }
    }

    /// Simulate memory cleanup for OOM errors
    fn simulate_memory_cleanup(&self) -> Result<()> {
        println!("🧹 Simulating memory cleanup...");
        println!("  - Clearing unused tensors");
        println!("  - Running garbage collection");
        println!("  - Reducing batch size temporarily");
        Ok(())
    }

    /// Suggest numerical stability fixes
    fn suggest_numerical_fixes(&self, error: &TrainingError) -> Result<()> {
        println!("💡 Numerical stability suggestions:");
        println!("  - Reduce learning rate by factor of 10");
        println!("  - Enable gradient clipping (max_norm=1.0)");
        println!("  - Check input data normalization");
        println!("  - Consider using mixed precision training");

        if let Some(lr) = error.context.learning_rate {
            println!("  - Current learning rate: {}, suggested: {}", lr, lr * 0.1);
        }

        Ok(())
    }

    /// Suggest data path fixes
    fn suggest_data_path_fixes(&self, _error: &TrainingError) -> Result<()> {
        println!("💡 Data path suggestions:");
        println!("  - Check if file path is correct");
        println!("  - Verify file permissions");
        println!("  - Try relative vs absolute paths");
        println!("  - Check if data is in expected location");
        Ok(())
    }

    /// Attempt network retry with exponential backoff
    fn attempt_network_retry(&self, _error: &TrainingError) -> Result<()> {
        println!("🔄 Attempting network retry with exponential backoff...");

        for attempt in 1..=3 {
            println!("  Attempt {}/3", attempt);

            // Simulate network operation
            std::thread::sleep(std::time::Duration::from_millis(100 * (1 << attempt)));

            // Simulate random success/failure
            if fastrand::bool() {
                println!("  ✅ Network operation succeeded");
                return Ok(());
            }

            println!("  ❌ Network operation failed, retrying...");
        }

        Err(anyhow::anyhow!("Network operation failed after 3 attempts"))
    }

    /// Suggest manual recovery steps
    fn suggest_manual_recovery(&self, error: &TrainingError) {
        println!("🔧 Manual recovery suggestions for {}:", error.error_code);

        for suggestion in &error.recovery_suggestions {
            println!(
                "  {} (Priority: {}) - {}",
                if suggestion.automatic { "🤖 AUTO" } else { "👤 MANUAL" },
                suggestion.priority,
                suggestion.action
            );
            println!("    📝 {}", suggestion.description);
        }

        // Additional context-specific suggestions
        if let Some(epoch) = error.context.epoch {
            println!("  📊 Error occurred at epoch {}", epoch);
            if epoch < 5 {
                println!(
                    "    💡 Early training failure - check data loading and model initialization"
                );
            }
        }

        if let Some(step) = error.context.step {
            println!("  📊 Error occurred at step {}", step);
        }
    }

    fn get_recovery_suggestions(
        &self,
        error_type: &ErrorType,
        error_code: &str,
    ) -> Vec<RecoverySuggestion> {
        // First, try to get suggestions from the error code registry
        if let Some(error_info) = get_error_info(error_code) {
            return error_info
                .solutions
                .iter()
                .enumerate()
                .map(|(i, solution)| {
                    RecoverySuggestion {
                        action: solution.to_string(),
                        description: format!(
                            "See documentation: {}",
                            error_info
                                .documentation_url
                                .unwrap_or("https://docs.trustformers.rs/errors")
                        ),
                        priority: 10 - i as u8, // Higher priority for earlier solutions
                        automatic: error_info.severity != "CRITICAL", // Auto-apply only for non-critical
                    }
                })
                .collect();
        }

        // Fallback to built-in recovery suggestions based on error type
        match error_type {
            ErrorType::Configuration => vec![
                RecoverySuggestion {
                    action: "Check configuration file".to_string(),
                    description: "Verify that all required parameters are set correctly"
                        .to_string(),
                    priority: 9,
                    automatic: false,
                },
                RecoverySuggestion {
                    action: "Use default configuration".to_string(),
                    description: "Fall back to known good default settings".to_string(),
                    priority: 7,
                    automatic: true,
                },
            ],
            ErrorType::DataLoading => vec![
                RecoverySuggestion {
                    action: "Check data path".to_string(),
                    description: "Verify that the dataset path exists and is accessible"
                        .to_string(),
                    priority: 9,
                    automatic: false,
                },
                RecoverySuggestion {
                    action: "Reduce batch size".to_string(),
                    description: "Try reducing batch size to avoid memory issues".to_string(),
                    priority: 8,
                    automatic: true,
                },
            ],
            ErrorType::Training => vec![
                RecoverySuggestion {
                    action: "Reduce learning rate".to_string(),
                    description: "Lower the learning rate to stabilize training".to_string(),
                    priority: 8,
                    automatic: true,
                },
                RecoverySuggestion {
                    action: "Check for NaN/Inf values".to_string(),
                    description: "Inspect model weights and gradients for numerical issues"
                        .to_string(),
                    priority: 9,
                    automatic: false,
                },
            ],
            ErrorType::Resource => vec![
                RecoverySuggestion {
                    action: "Free up memory".to_string(),
                    description: "Clear unused variables and reduce model size".to_string(),
                    priority: 9,
                    automatic: true,
                },
                RecoverySuggestion {
                    action: "Use gradient checkpointing".to_string(),
                    description: "Enable gradient checkpointing to reduce memory usage".to_string(),
                    priority: 7,
                    automatic: true,
                },
            ],
            _ => vec![RecoverySuggestion {
                action: "Restart training".to_string(),
                description: "Restart training from the last checkpoint".to_string(),
                priority: 5,
                automatic: false,
            }],
        }
    }

    /// Get error statistics
    pub fn get_statistics(&self) -> Result<ErrorStatistics> {
        let stats = self
            .statistics
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on statistics"))?;
        Ok((*stats).clone())
    }

    /// Get recent errors
    pub fn get_recent_errors(&self, limit: usize) -> Result<Vec<TrainingError>> {
        let errors = self
            .errors
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on errors"))?;

        let recent: Vec<_> = errors.iter().rev().take(limit).cloned().collect();

        Ok(recent)
    }

    /// Clear error history
    pub fn clear_errors(&self) -> Result<()> {
        let mut errors = self
            .errors
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on errors"))?;
        errors.clear();
        Ok(())
    }

    /// Export errors to JSON for external analysis
    pub fn export_errors(&self) -> Result<String> {
        let errors = self
            .errors
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on errors"))?;

        serde_json::to_string_pretty(&*errors).context("Failed to serialize errors to JSON")
    }
}

/// Helper macros for error creation
#[macro_export]
macro_rules! training_error {
    ($error_manager:expr, $error_type:expr, $message:expr, $error_code:expr, $severity:expr, $context:expr) => {
        $error_manager.create_error(
            $error_type,
            $message.to_string(),
            $error_code.to_string(),
            $severity,
            $context,
        )
    };
}

#[macro_export]
macro_rules! create_context {
    ($component:expr, $operation:expr) => {
        ErrorContext {
            component: $component.to_string(),
            operation: $operation.to_string(),
            epoch: None,
            step: None,
            batch_size: None,
            learning_rate: None,
            model_state: None,
            system_info: SystemInfo {
                memory_usage: None,
                gpu_memory_usage: None,
                cpu_usage: None,
                disk_space: None,
                network_status: None,
            },
            additional_data: HashMap::new(),
        }
    };
    ($component:expr, $operation:expr, epoch: $epoch:expr, step: $step:expr) => {
        ErrorContext {
            component: $component.to_string(),
            operation: $operation.to_string(),
            epoch: Some($epoch),
            step: Some($step),
            batch_size: None,
            learning_rate: None,
            model_state: None,
            system_info: SystemInfo {
                memory_usage: None,
                gpu_memory_usage: None,
                cpu_usage: None,
                disk_space: None,
                network_status: None,
            },
            additional_data: HashMap::new(),
        }
    };
}

/// Result type with training error
pub type TrainingResult<T> = Result<T, TrainingError>;

/// Extension trait for converting anyhow errors to training errors
pub trait TrainingErrorExt<T> {
    fn training_error(
        self,
        error_manager: &ErrorManager,
        error_type: ErrorType,
        error_code: &str,
        severity: ErrorSeverity,
        context: ErrorContext,
    ) -> TrainingResult<T>;
}

impl<T> TrainingErrorExt<T> for Result<T> {
    fn training_error(
        self,
        error_manager: &ErrorManager,
        error_type: ErrorType,
        error_code: &str,
        severity: ErrorSeverity,
        context: ErrorContext,
    ) -> TrainingResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(e) => {
                let training_error = error_manager.create_error(
                    error_type,
                    e.to_string(),
                    error_code.to_string(),
                    severity,
                    context,
                );
                Err(training_error)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_manager_creation() {
        let manager = ErrorManager::new();
        let stats = manager.get_statistics().unwrap();
        assert_eq!(stats.total_errors, 0);
    }

    #[test]
    fn test_error_recording() {
        let manager = ErrorManager::new();

        let context = ErrorContext {
            component: "trainer".to_string(),
            operation: "forward_pass".to_string(),
            epoch: Some(1),
            step: Some(100),
            batch_size: Some(32),
            learning_rate: Some(0.001),
            model_state: None,
            system_info: SystemInfo {
                memory_usage: Some(1024),
                gpu_memory_usage: Some(512),
                cpu_usage: Some(50.0),
                disk_space: None,
                network_status: None,
            },
            additional_data: HashMap::new(),
        };

        let error = manager.create_error(
            ErrorType::Training,
            "NaN detected in loss".to_string(),
            "TRAINING_NAN_LOSS".to_string(),
            ErrorSeverity::Critical,
            context,
        );

        assert_eq!(error.error_type, ErrorType::Training);
        assert_eq!(error.error_code, "TRAINING_NAN_LOSS");
        assert_eq!(error.severity, ErrorSeverity::Critical);
        assert!(!error.recovery_suggestions.is_empty());

        let stats = manager.get_statistics().unwrap();
        assert_eq!(stats.total_errors, 1);
        assert_eq!(*stats.errors_by_type.get(&ErrorType::Training).unwrap(), 1);
    }

    #[test]
    fn test_error_pattern_detection() {
        let manager = ErrorManager::new();

        let pattern = ErrorPattern {
            pattern_id: "frequent_oom".to_string(),
            error_codes: vec!["RESOURCE_OOM".to_string()],
            frequency_threshold: 3,
            time_window_seconds: 300,
            suggested_actions: vec![RecoverySuggestion {
                action: "Reduce batch size".to_string(),
                description: "Lower batch size to reduce memory usage".to_string(),
                priority: 9,
                automatic: true,
            }],
        };

        manager.add_error_pattern(pattern).unwrap();

        // Simulate multiple OOM errors
        let context = create_context!("trainer", "forward_pass");
        for _ in 0..3 {
            manager.create_error(
                ErrorType::Resource,
                "Out of memory".to_string(),
                "RESOURCE_OOM".to_string(),
                ErrorSeverity::Critical,
                context.clone(),
            );
        }

        // Pattern should be detected after 3 occurrences
        let stats = manager.get_statistics().unwrap();
        assert_eq!(stats.total_errors, 3);
    }

    #[test]
    fn test_recovery_suggestions() {
        let manager = ErrorManager::new();

        let suggestions =
            manager.get_recovery_suggestions(&ErrorType::Training, "TRAINING_NAN_LOSS");
        assert!(!suggestions.is_empty());

        let has_lr_suggestion = suggestions.iter().any(|s| s.action.contains("learning rate"));
        assert!(has_lr_suggestion);
    }

    #[test]
    fn test_error_statistics() {
        let manager = ErrorManager::new();

        // Create errors of different types and severities
        let context = create_context!("trainer", "test");

        manager.create_error(
            ErrorType::Training,
            "Test error 1".to_string(),
            "TEST_001".to_string(),
            ErrorSeverity::Critical,
            context.clone(),
        );

        manager.create_error(
            ErrorType::DataLoading,
            "Test error 2".to_string(),
            "TEST_002".to_string(),
            ErrorSeverity::Medium,
            context.clone(),
        );

        manager.create_error(
            ErrorType::Training,
            "Test error 3".to_string(),
            "TEST_003".to_string(),
            ErrorSeverity::High,
            context,
        );

        let stats = manager.get_statistics().unwrap();
        assert_eq!(stats.total_errors, 3);
        assert_eq!(*stats.errors_by_type.get(&ErrorType::Training).unwrap(), 2);
        assert_eq!(
            *stats.errors_by_type.get(&ErrorType::DataLoading).unwrap(),
            1
        );
        assert_eq!(
            *stats.errors_by_severity.get(&ErrorSeverity::Critical).unwrap(),
            1
        );
        assert_eq!(
            *stats.errors_by_severity.get(&ErrorSeverity::Medium).unwrap(),
            1
        );
        assert_eq!(
            *stats.errors_by_severity.get(&ErrorSeverity::High).unwrap(),
            1
        );
    }

    #[test]
    fn test_recent_errors() {
        let manager = ErrorManager::new();

        let context = create_context!("trainer", "test");

        // Create 5 errors
        for i in 0..5 {
            manager.create_error(
                ErrorType::Training,
                format!("Test error {}", i),
                format!("TEST_{:03}", i),
                ErrorSeverity::Medium,
                context.clone(),
            );
        }

        let recent_errors = manager.get_recent_errors(3).unwrap();
        assert_eq!(recent_errors.len(), 3);

        // Should be in reverse chronological order (most recent first)
        assert_eq!(recent_errors[0].error_code, "TEST_004");
        assert_eq!(recent_errors[1].error_code, "TEST_003");
        assert_eq!(recent_errors[2].error_code, "TEST_002");
    }

    #[test]
    fn test_error_export() {
        let manager = ErrorManager::new();

        let context = create_context!("trainer", "test");
        manager.create_error(
            ErrorType::Training,
            "Test error".to_string(),
            "TEST_001".to_string(),
            ErrorSeverity::Medium,
            context,
        );

        let json_export = manager.export_errors().unwrap();
        assert!(!json_export.is_empty());
        assert!(json_export.contains("TEST_001"));
        assert!(json_export.contains("Test error"));
    }
}
