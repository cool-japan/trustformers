use thiserror::Error;
use trustformers_core::errors::TrustformersError as CoreTrustformersError;

/// Enhanced error type for high-level trustformers operations
#[derive(Error, Debug)]
pub enum TrustformersError {
    /// Pipeline configuration or execution errors
    #[error("🔧 Pipeline Error: {message}")]
    Pipeline {
        message: String,
        pipeline_type: String,
        suggestion: Option<String>,
        recovery_actions: Vec<RecoveryAction>,
    },

    /// Model loading or inference errors
    #[error("🤖 Model Error: {message}")]
    Model {
        message: String,
        model_name: String,
        model_type: Option<String>,
        suggestion: Option<String>,
        recovery_actions: Vec<RecoveryAction>,
    },

    /// Hub and model downloading errors
    #[error("🌐 Hub Error: {message}")]
    Hub {
        message: String,
        model_id: String,
        endpoint: Option<String>,
        suggestion: Option<String>,
        recovery_actions: Vec<RecoveryAction>,
    },

    /// Auto model/tokenizer instantiation errors
    #[error("⚙️ Auto Configuration Error: {message}")]
    AutoConfig {
        message: String,
        config_type: String,
        suggestion: Option<String>,
        recovery_actions: Vec<RecoveryAction>,
    },

    /// Feature or capability not available
    #[error("❌ Feature Not Available: {message}")]
    FeatureUnavailable {
        message: String,
        feature: String,
        suggestion: Option<String>,
        alternatives: Vec<String>,
    },

    /// Resource exhaustion errors
    #[error("💾 Resource Error: {message}")]
    Resource {
        message: String,
        resource_type: String,
        current_usage: Option<String>,
        suggestion: Option<String>,
        recovery_actions: Vec<RecoveryAction>,
    },

    /// Invalid input data or parameters
    #[error("📝 Input Error: {message}")]
    InvalidInput {
        message: String,
        parameter: Option<String>,
        expected: Option<String>,
        received: Option<String>,
        suggestion: Option<String>,
    },

    /// Underlying core errors
    #[error("🔥 Core Error: {0}")]
    Core(CoreTrustformersError),

    /// IO and file system errors
    #[error("📁 IO Error: {message}")]
    Io {
        message: String,
        path: Option<String>,
        suggestion: Option<String>,
    },

    /// Network and connectivity errors
    #[error("🌐 Network Error: {message}")]
    Network {
        message: String,
        url: Option<String>,
        status_code: Option<u16>,
        suggestion: Option<String>,
        retry_recommended: bool,
    },

    /// Pipeline not found errors
    #[error("🔍 Pipeline Not Found: {message}")]
    PipelineNotFound {
        message: String,
        pipeline_name: String,
        suggestion: Option<String>,
    },
}

/// Recovery actions that can be automatically attempted
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryAction {
    /// Retry the operation with the same parameters
    Retry { max_attempts: u32, delay_ms: u64 },
    /// Use CPU instead of GPU
    FallbackToCpu,
    /// Reduce batch size
    ReduceBatchSize { factor: f32 },
    /// Use a different model
    UseAlternativeModel { model_name: String },
    /// Clear cache and retry
    ClearCache,
    /// Download model files again
    RedownloadModel,
    /// Use offline mode
    UseOfflineMode,
    /// Adjust memory settings
    ReduceMemoryUsage { reduction_factor: f32 },
    /// Use different precision
    ReducePrecision { target_precision: String },
    /// Custom recovery action
    Custom { action: String, description: String },
}

/// Result type for trustformers operations
pub type Result<T> = std::result::Result<T, TrustformersError>;

/// Error recovery context for tracking recovery attempts
#[derive(Debug, Clone)]
pub struct RecoveryContext {
    pub operation: String,
    pub attempts: u32,
    pub max_attempts: u32,
    pub last_error: Option<String>,
    pub recovery_history: Vec<RecoveryAction>,
}

impl RecoveryContext {
    pub fn new(operation: impl Into<String>, max_attempts: u32) -> Self {
        Self {
            operation: operation.into(),
            attempts: 0,
            max_attempts,
            last_error: None,
            recovery_history: Vec::new(),
        }
    }

    pub fn can_retry(&self) -> bool {
        self.attempts < self.max_attempts
    }

    pub fn record_attempt(&mut self, error: impl Into<String>) {
        self.attempts += 1;
        self.last_error = Some(error.into());
    }

    pub fn record_recovery(&mut self, action: RecoveryAction) {
        self.recovery_history.push(action);
    }
}

impl TrustformersError {
    /// Create a pipeline error with automatic suggestions
    pub fn pipeline(message: impl Into<String>, pipeline_type: impl Into<String>) -> Self {
        let msg = message.into();
        let ptype = pipeline_type.into();

        let (suggestion, recovery_actions) = Self::generate_pipeline_suggestions(&msg, &ptype);

        TrustformersError::Pipeline {
            message: msg,
            pipeline_type: ptype,
            suggestion: Some(suggestion),
            recovery_actions,
        }
    }

    /// Create a model error with automatic suggestions
    pub fn model(message: impl Into<String>, model_name: impl Into<String>) -> Self {
        let msg = message.into();
        let name = model_name.into();

        let (suggestion, recovery_actions) = Self::generate_model_suggestions(&msg, &name);

        TrustformersError::Model {
            message: msg,
            model_name: name,
            model_type: None,
            suggestion: Some(suggestion),
            recovery_actions,
        }
    }

    /// Create a hub error with automatic suggestions
    pub fn hub(message: impl Into<String>, model_id: impl Into<String>) -> Self {
        let msg = message.into();
        let id = model_id.into();

        let (suggestion, recovery_actions) = Self::generate_hub_suggestions(&msg, &id);

        TrustformersError::Hub {
            message: msg,
            model_id: id,
            endpoint: None,
            suggestion: Some(suggestion),
            recovery_actions,
        }
    }

    /// Create a feature unavailable error with alternatives
    pub fn feature_unavailable(message: impl Into<String>, feature: impl Into<String>) -> Self {
        let msg = message.into();
        let feat = feature.into();

        let (suggestion, alternatives) = Self::generate_feature_alternatives(&feat);

        TrustformersError::FeatureUnavailable {
            message: msg,
            feature: feat,
            suggestion: Some(suggestion),
            alternatives,
        }
    }

    /// Create a resource error with recovery actions
    pub fn resource(message: impl Into<String>, resource_type: impl Into<String>) -> Self {
        let msg = message.into();
        let rtype = resource_type.into();

        let (suggestion, recovery_actions) = Self::generate_resource_suggestions(&msg, &rtype);

        TrustformersError::Resource {
            message: msg,
            resource_type: rtype,
            current_usage: None,
            suggestion: Some(suggestion),
            recovery_actions,
        }
    }

    /// Create an input validation error with specific guidance
    pub fn invalid_input(
        message: impl Into<String>,
        parameter: Option<impl Into<String>>,
        expected: Option<impl Into<String>>,
        received: Option<impl Into<String>>,
    ) -> Self {
        let msg = message.into();
        let param = parameter.map(|p| p.into());
        let exp = expected.map(|e| e.into());
        let rec = received.map(|r| r.into());

        let suggestion = Self::generate_input_suggestion(&msg, &param, &exp, &rec);

        TrustformersError::InvalidInput {
            message: msg,
            parameter: param,
            expected: exp,
            received: rec,
            suggestion: Some(suggestion),
        }
    }

    /// Create a simple input validation error with just a message
    pub fn invalid_input_simple(message: impl Into<String>) -> Self {
        let msg = message.into();
        TrustformersError::InvalidInput {
            message: msg.clone(),
            parameter: None,
            expected: None,
            received: None,
            suggestion: Some("Please check the input parameters and values".to_string()),
        }
    }

    /// Create a file not found error
    pub fn file_not_found(message: impl Into<String>) -> Self {
        TrustformersError::Io {
            message: message.into(),
            path: None,
            suggestion: Some("Check that the file exists and the path is correct".to_string()),
        }
    }

    /// Create an IO error
    pub fn io_error(message: impl Into<String>) -> Self {
        TrustformersError::Io {
            message: message.into(),
            path: None,
            suggestion: Some("Check file permissions and disk space".to_string()),
        }
    }

    /// Create a new error from an error kind (bridge method for compatibility)
    pub fn new(core_error: CoreTrustformersError) -> Self {
        TrustformersError::Core(core_error)
    }

    /// Create a runtime error
    pub fn runtime_error(message: impl Into<String>) -> Self {
        TrustformersError::Pipeline {
            message: message.into(),
            pipeline_type: "runtime".to_string(),
            suggestion: Some("Check system resources and configuration".to_string()),
            recovery_actions: vec![
                RecoveryAction::Retry {
                    max_attempts: 2,
                    delay_ms: 1000,
                },
                RecoveryAction::ClearCache,
            ],
        }
    }

    /// Create a configuration error (lconfig_error alias for compatibility)
    pub fn lconfig_error(message: impl Into<String>) -> Self {
        TrustformersError::AutoConfig {
            message: message.into(),
            config_type: "configuration".to_string(),
            suggestion: Some("Check configuration parameters and values".to_string()),
            recovery_actions: vec![RecoveryAction::ClearCache],
        }
    }

    /// Attempt automatic recovery from an error
    pub async fn try_recover<T, F, Fut>(
        &self,
        context: &mut RecoveryContext,
        operation: F,
    ) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        if !context.can_retry() {
            return Err(TrustformersError::Pipeline {
                message: "Recovery attempts exhausted".to_string(),
                pipeline_type: "recovery".to_string(),
                suggestion: Some("Check logs for details".to_string()),
                recovery_actions: vec![],
            });
        }

        let recovery_actions = self.get_recovery_actions();

        for action in recovery_actions {
            if self.should_attempt_recovery(&action, context) {
                match self.execute_recovery_action(&action).await {
                    Ok(_) => {
                        context.record_recovery(action.clone());

                        // Retry the operation
                        match operation().await {
                            Ok(result) => return Ok(result),
                            Err(err) => {
                                context.record_attempt(err.to_string());
                                continue;
                            },
                        }
                    },
                    Err(_) => continue,
                }
            }
        }

        Err(TrustformersError::Pipeline {
            message: "All recovery actions failed".to_string(),
            pipeline_type: "recovery".to_string(),
            suggestion: Some("Manual intervention required".to_string()),
            recovery_actions: vec![],
        })
    }

    /// Get recovery actions for this error
    pub fn get_recovery_actions(&self) -> Vec<RecoveryAction> {
        match self {
            TrustformersError::Pipeline {
                recovery_actions, ..
            } => recovery_actions.clone(),
            TrustformersError::Model {
                recovery_actions, ..
            } => recovery_actions.clone(),
            TrustformersError::Hub {
                recovery_actions, ..
            } => recovery_actions.clone(),
            TrustformersError::AutoConfig {
                recovery_actions, ..
            } => recovery_actions.clone(),
            TrustformersError::Resource {
                recovery_actions, ..
            } => recovery_actions.clone(),
            TrustformersError::Core(core_err) => {
                // Convert core error recovery actions
                Self::convert_core_recovery_actions(core_err)
            },
            TrustformersError::Network {
                retry_recommended, ..
            } => {
                if *retry_recommended {
                    vec![RecoveryAction::Retry {
                        max_attempts: 3,
                        delay_ms: 1000,
                    }]
                } else {
                    vec![]
                }
            },
            _ => vec![],
        }
    }

    /// Check if a recovery action should be attempted
    fn should_attempt_recovery(&self, action: &RecoveryAction, context: &RecoveryContext) -> bool {
        // Don't repeat the same recovery action
        if context.recovery_history.contains(action) {
            return false;
        }

        // Check if we have attempts left for retry actions
        if let RecoveryAction::Retry { max_attempts, .. } = action {
            return context.attempts < *max_attempts;
        }

        true
    }

    /// Execute a recovery action
    async fn execute_recovery_action(&self, action: &RecoveryAction) -> Result<()> {
        match action {
            RecoveryAction::Retry { delay_ms, .. } => {
                tokio::time::sleep(std::time::Duration::from_millis(*delay_ms)).await;
                Ok(())
            },
            RecoveryAction::FallbackToCpu => {
                // Set CPU device flag (would integrate with actual device management)
                log::info!("🔄 Falling back to CPU execution");
                Ok(())
            },
            RecoveryAction::ReduceBatchSize { factor } => {
                log::info!("🔄 Reducing batch size by factor {}", factor);
                // Would integrate with actual batch size management
                Ok(())
            },
            RecoveryAction::ClearCache => {
                log::info!("🔄 Clearing cache");
                // Would integrate with actual cache clearing
                Ok(())
            },
            RecoveryAction::RedownloadModel => {
                log::info!("🔄 Attempting to redownload model");
                // Would integrate with actual model downloading
                Ok(())
            },
            RecoveryAction::UseOfflineMode => {
                log::info!("🔄 Switching to offline mode");
                // Would integrate with offline mode switching
                Ok(())
            },
            RecoveryAction::ReduceMemoryUsage { reduction_factor } => {
                log::info!("🔄 Reducing memory usage by factor {}", reduction_factor);
                // Would integrate with memory management
                Ok(())
            },
            RecoveryAction::ReducePrecision { target_precision } => {
                log::info!("🔄 Reducing precision to {}", target_precision);
                // Would integrate with precision management
                Ok(())
            },
            RecoveryAction::UseAlternativeModel { model_name } => {
                log::info!("🔄 Switching to alternative model: {}", model_name);
                // Would integrate with model switching
                Ok(())
            },
            RecoveryAction::Custom {
                action,
                description,
            } => {
                log::info!("🔄 Executing custom recovery: {} ({})", action, description);
                // Would execute custom recovery logic
                Ok(())
            },
        }
    }

    // Helper methods for generating suggestions and recovery actions

    fn generate_pipeline_suggestions(
        message: &str,
        pipeline_type: &str,
    ) -> (String, Vec<RecoveryAction>) {
        let suggestion = match pipeline_type {
            "text-generation" => "Try reducing max_length or batch_size parameters".to_string(),
            "text-classification" => {
                "Ensure input text is properly formatted and not empty".to_string()
            },
            "image-to-text" => "Verify image format is supported (JPEG, PNG, WebP)".to_string(),
            "question-answering" => "Check that both question and context are provided".to_string(),
            _ => "Review pipeline configuration and input parameters".to_string(),
        };

        let recovery_actions = vec![
            RecoveryAction::ReduceBatchSize { factor: 0.5 },
            RecoveryAction::FallbackToCpu,
            RecoveryAction::ClearCache,
        ];

        (suggestion, recovery_actions)
    }

    fn generate_model_suggestions(
        message: &str,
        model_name: &str,
    ) -> (String, Vec<RecoveryAction>) {
        let suggestion = if message.contains("not found") {
            format!(
                "Model '{}' not found. Check model name spelling or try downloading it manually.",
                model_name
            )
        } else if message.contains("memory") || message.contains("OOM") {
            "Model too large for available memory. Try using a smaller model or reducing batch size.".to_string()
        } else if message.contains("format") {
            "Model format not supported. Ensure model is in TensorFlow SavedModel or PyTorch format.".to_string()
        } else {
            "Review model configuration and ensure all required files are present.".to_string()
        };

        let recovery_actions = vec![
            RecoveryAction::RedownloadModel,
            RecoveryAction::UseAlternativeModel {
                model_name: Self::suggest_alternative_model(model_name),
            },
            RecoveryAction::ReduceMemoryUsage {
                reduction_factor: 0.7,
            },
            RecoveryAction::ReducePrecision {
                target_precision: "fp16".to_string(),
            },
        ];

        (suggestion, recovery_actions)
    }

    fn generate_hub_suggestions(message: &str, model_id: &str) -> (String, Vec<RecoveryAction>) {
        let suggestion = if message.contains("network") || message.contains("timeout") {
            "Network issue detected. Check internet connection and try again.".to_string()
        } else if message.contains("not found") || message.contains("404") {
            format!(
                "Model '{}' not found on Hugging Face Hub. Verify the model ID.",
                model_id
            )
        } else if message.contains("permission") || message.contains("403") {
            "Access denied. This may be a private model requiring authentication.".to_string()
        } else {
            "Hub connection issue. Try again or use offline mode if model is cached.".to_string()
        };

        let recovery_actions = vec![
            RecoveryAction::Retry {
                max_attempts: 3,
                delay_ms: 2000,
            },
            RecoveryAction::UseOfflineMode,
            RecoveryAction::UseAlternativeModel {
                model_name: Self::suggest_alternative_model(model_id),
            },
        ];

        (suggestion, recovery_actions)
    }

    fn generate_feature_alternatives(feature: &str) -> (String, Vec<String>) {
        let (suggestion, alternatives) = match feature {
            "vision" => (
                "Vision features require the 'vision' feature flag".to_string(),
                vec![
                    "Enable vision feature".to_string(),
                    "Use text-only pipeline".to_string(),
                ],
            ),
            "audio" => (
                "Audio features require the 'audio' feature flag".to_string(),
                vec![
                    "Enable audio feature".to_string(),
                    "Use text-only pipeline".to_string(),
                ],
            ),
            "gpu" => (
                "GPU features not available. Using CPU instead".to_string(),
                vec![
                    "Install CUDA/ROCm drivers".to_string(),
                    "Use CPU-only execution".to_string(),
                ],
            ),
            _ => (
                format!("Feature '{}' is not available in this build", feature),
                vec![
                    "Check feature flags".to_string(),
                    "Use alternative approach".to_string(),
                ],
            ),
        };

        (suggestion, alternatives)
    }

    fn generate_resource_suggestions(
        message: &str,
        resource_type: &str,
    ) -> (String, Vec<RecoveryAction>) {
        let suggestion = match resource_type {
            "memory" => {
                "Insufficient memory. Try reducing batch size or using a smaller model.".to_string()
            },
            "gpu_memory" => {
                "GPU memory exhausted. Consider using CPU or reducing model precision.".to_string()
            },
            "disk" => {
                "Insufficient disk space for model cache. Clear cache or use streaming.".to_string()
            },
            _ => format!(
                "Resource '{}' exhausted. Review usage and optimize.",
                resource_type
            ),
        };

        let recovery_actions = match resource_type {
            "memory" | "gpu_memory" => vec![
                RecoveryAction::ReduceBatchSize { factor: 0.5 },
                RecoveryAction::ReduceMemoryUsage {
                    reduction_factor: 0.6,
                },
                RecoveryAction::ReducePrecision {
                    target_precision: "fp16".to_string(),
                },
                RecoveryAction::FallbackToCpu,
            ],
            "disk" => vec![RecoveryAction::ClearCache],
            _ => vec![],
        };

        (suggestion, recovery_actions)
    }

    fn generate_input_suggestion(
        message: &str,
        parameter: &Option<String>,
        expected: &Option<String>,
        received: &Option<String>,
    ) -> String {
        if let (Some(param), Some(exp), Some(rec)) = (parameter, expected, received) {
            format!(
                "Parameter '{}' validation failed. Expected: {}, Received: {}. Please correct the input.",
                param, exp, rec
            )
        } else if let Some(param) = parameter {
            format!(
                "Parameter '{}' is invalid. Please check the documentation for valid values.",
                param
            )
        } else {
            "Input validation failed. Please review the provided parameters.".to_string()
        }
    }

    fn suggest_alternative_model(model_name: &str) -> String {
        // Simple logic to suggest alternatives based on model name patterns
        if model_name.contains("large") {
            model_name.replace("large", "base")
        } else if model_name.contains("xl") {
            model_name.replace("xl", "base")
        } else if model_name.contains("gpt2") {
            "gpt2".to_string()
        } else if model_name.contains("bert") {
            "bert-base-uncased".to_string()
        } else {
            "distilbert-base-uncased".to_string() // Fallback to a small, reliable model
        }
    }

    fn convert_core_recovery_actions(core_err: &CoreTrustformersError) -> Vec<RecoveryAction> {
        // Convert core error recovery actions to high-level recovery actions
        vec![
            RecoveryAction::FallbackToCpu,
            RecoveryAction::ReduceMemoryUsage {
                reduction_factor: 0.7,
            },
            RecoveryAction::ClearCache,
        ]
    }
}

// Convenient macros for creating errors
#[macro_export]
macro_rules! pipeline_error {
    ($msg:expr, $pipeline_type:expr) => {
        $crate::error::TrustformersError::pipeline($msg, $pipeline_type)
    };
}

#[macro_export]
macro_rules! model_error {
    ($msg:expr, $model_name:expr) => {
        $crate::error::TrustformersError::model($msg, $model_name)
    };
}

#[macro_export]
macro_rules! hub_error {
    ($msg:expr, $model_id:expr) => {
        $crate::error::TrustformersError::hub($msg, $model_id)
    };
}

// From implementations for common error types
impl From<std::io::Error> for TrustformersError {
    fn from(err: std::io::Error) -> Self {
        TrustformersError::Io {
            message: err.to_string(),
            path: None,
            suggestion: Some("Check file permissions and disk space".to_string()),
        }
    }
}

impl From<reqwest::Error> for TrustformersError {
    fn from(err: reqwest::Error) -> Self {
        let retry_recommended = err.is_timeout() || err.is_connect();

        TrustformersError::Network {
            message: err.to_string(),
            url: err.url().map(|u| u.to_string()),
            status_code: err.status().map(|s| s.as_u16()),
            suggestion: Some(if retry_recommended {
                "Network issue - retry recommended".to_string()
            } else {
                "Check network configuration and URL".to_string()
            }),
            retry_recommended,
        }
    }
}

impl From<anyhow::Error> for TrustformersError {
    fn from(err: anyhow::Error) -> Self {
        TrustformersError::Core(CoreTrustformersError::invalid_input_simple(err.to_string()))
    }
}

impl From<trustformers_core::errors::TrustformersError> for TrustformersError {
    fn from(err: trustformers_core::errors::TrustformersError) -> Self {
        TrustformersError::Core(err)
    }
}

/// Bidirectional conversion from high-level TrustformersError to core TrustformersError
impl From<TrustformersError> for trustformers_core::errors::TrustformersError {
    fn from(err: TrustformersError) -> Self {
        match err {
            TrustformersError::Core(core_err) => {
                // The Core variant already contains a CoreTrustformersError
                core_err
            },
            _ => {
                // Create a runtime error for non-core errors
                trustformers_core::errors::TrustformersError::new(
                    trustformers_core::errors::ErrorKind::RuntimeError {
                        reason: err.to_string(),
                    },
                )
                .with_operation("high_level_conversion")
                .with_suggestion("Handle specific error types in high-level error handling")
            },
        }
    }
}

/// Conversion from deprecated CoreError to high-level TrustformersError
impl From<trustformers_core::error::CoreError> for TrustformersError {
    fn from(err: trustformers_core::error::CoreError) -> Self {
        // Convert deprecated CoreError to runtime error
        TrustformersError::runtime_error(format!("Deprecated CoreError: {}", err))
    }
}

/// Bidirectional conversion from high-level TrustformersError to deprecated CoreError
impl From<TrustformersError> for trustformers_core::error::CoreError {
    fn from(err: TrustformersError) -> Self {
        use trustformers_core::error::{CoreError, ErrorCode, ErrorContext};

        match err {
            TrustformersError::InvalidInput { message, .. } => CoreError::InvalidInput(message),
            TrustformersError::Resource {
                message,
                resource_type,
                ..
            } => {
                if resource_type == "memory" {
                    CoreError::MemoryError {
                        message,
                        context: ErrorContext::new(
                            ErrorCode::E3001,
                            "memory_allocation".to_string(),
                        ),
                    }
                } else {
                    CoreError::RuntimeError(message)
                }
            },
            TrustformersError::Core(ref core_err) => {
                // The Core variant wraps a CoreTrustformersError, not a CoreError
                // We need to convert it
                CoreError::RuntimeError(core_err.to_string())
            },
            _ => CoreError::RuntimeError(err.to_string()),
        }
    }
}

impl From<serde_json::Error> for TrustformersError {
    fn from(err: serde_json::Error) -> Self {
        TrustformersError::InvalidInput {
            message: format!("JSON serialization/deserialization error: {}", err),
            parameter: None,
            expected: Some("valid JSON".to_string()),
            received: Some("invalid JSON data".to_string()),
            suggestion: Some("Check JSON format and structure".to_string()),
        }
    }
}

impl From<std::fmt::Error> for TrustformersError {
    fn from(err: std::fmt::Error) -> Self {
        TrustformersError::InvalidInput {
            message: format!("Formatting error: {}", err),
            parameter: None,
            expected: Some("valid format string".to_string()),
            received: Some("invalid format".to_string()),
            suggestion: Some("Check format string syntax".to_string()),
        }
    }
}

#[cfg(feature = "ndarray")]
impl From<ndarray::error::ShapeError> for TrustformersError {
    fn from(err: ndarray::error::ShapeError) -> Self {
        TrustformersError::InvalidInput {
            message: format!("Array shape error: {}", err),
            parameter: Some("array_shape".to_string()),
            expected: Some("compatible array shape".to_string()),
            received: Some("incompatible shape".to_string()),
            suggestion: Some("Check array dimensions and reshape if necessary".to_string()),
        }
    }
}

/// Comprehensive error recovery management system
#[derive(Debug, Clone)]
pub struct AutoRecoveryManager {
    policy: RecoveryPolicy,
    recovery_history: std::collections::HashMap<String, Vec<RecoveryAttempt>>,
    circuit_breakers: std::collections::HashMap<String, CircuitBreaker>,
}

/// Configuration for error recovery behavior
#[derive(Debug, Clone)]
pub struct RecoveryPolicy {
    /// Maximum number of recovery attempts per error type
    pub max_attempts: u32,
    /// Whether to enable automatic recovery
    pub auto_recovery_enabled: bool,
    /// Whether to learn from successful recovery strategies
    pub learning_enabled: bool,
    /// Delay between recovery attempts in milliseconds
    pub retry_delay_ms: u64,
    /// Exponential backoff factor for retries
    pub backoff_factor: f64,
    /// Maximum delay between retries in milliseconds
    pub max_delay_ms: u64,
    /// Whether to enable circuit breaker pattern
    pub circuit_breaker_enabled: bool,
    /// Number of failures before opening circuit breaker
    pub circuit_breaker_threshold: u32,
    /// Time to wait before attempting to close circuit breaker
    pub circuit_breaker_timeout_ms: u64,
}

impl Default for RecoveryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            auto_recovery_enabled: true,
            learning_enabled: true,
            retry_delay_ms: 1000,
            backoff_factor: 2.0,
            max_delay_ms: 30000,
            circuit_breaker_enabled: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_ms: 60000,
        }
    }
}

/// Circuit breaker for preventing repeated failures
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    state: CircuitBreakerState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Option<std::time::Instant>,
    threshold: u32,
    timeout: std::time::Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Preventing requests
    HalfOpen, // Testing if service recovered
}

/// Record of a recovery attempt
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    pub timestamp: std::time::SystemTime,
    pub action: RecoveryAction,
    pub success: bool,
    pub duration_ms: u64,
    pub error_type: String,
}

/// Advanced recovery strategies
#[derive(Debug, Clone, PartialEq)]
pub enum AdvancedRecoveryStrategy {
    /// Exponential backoff with jitter
    ExponentialBackoff {
        base_delay_ms: u64,
        max_delay_ms: u64,
    },
    /// Circuit breaker pattern
    CircuitBreaker {
        failure_threshold: u32,
        timeout_ms: u64,
    },
    /// Bulkhead isolation
    Bulkhead { max_concurrent_operations: u32 },
    /// Graceful degradation
    GracefulDegradation { fallback_quality: f32 },
    /// Adaptive timeout
    AdaptiveTimeout {
        base_timeout_ms: u64,
        success_factor: f32,
        failure_factor: f32,
    },
}

impl AutoRecoveryManager {
    /// Create a new recovery manager with default policy
    pub fn new() -> Self {
        Self::with_policy(RecoveryPolicy::default())
    }

    /// Create a new recovery manager with custom policy
    pub fn with_policy(policy: RecoveryPolicy) -> Self {
        Self {
            policy,
            recovery_history: std::collections::HashMap::new(),
            circuit_breakers: std::collections::HashMap::new(),
        }
    }

    /// Attempt automatic recovery for an error
    pub async fn attempt_recovery<T, F, Fut>(
        &mut self,
        operation_name: &str,
        error: &TrustformersError,
        operation: F,
    ) -> Result<T>
    where
        F: Fn() -> Fut + Clone,
        Fut: std::future::Future<Output = Result<T>>,
    {
        if !self.policy.auto_recovery_enabled {
            return Err(TrustformersError::Resource {
                message: "Auto recovery is disabled".to_string(),
                resource_type: "recovery_policy".to_string(),
                current_usage: Some(format!("Original error: {}", error)),
                suggestion: Some("Enable auto recovery or handle the error manually".to_string()),
                recovery_actions: Vec::new(),
            });
        }

        // Check circuit breaker
        if let Some(breaker) = self.circuit_breakers.get_mut(operation_name) {
            if breaker.is_open() {
                return Err(TrustformersError::Resource {
                    message: "Circuit breaker is open - operation blocked".to_string(),
                    resource_type: "circuit_breaker".to_string(),
                    current_usage: Some(format!("failures: {}", breaker.failure_count)),
                    suggestion: Some(
                        "Wait for circuit breaker to close or reset manually".to_string(),
                    ),
                    recovery_actions: vec![RecoveryAction::Retry {
                        max_attempts: 1,
                        delay_ms: self.policy.circuit_breaker_timeout_ms,
                    }],
                });
            }
        }

        let recovery_actions = error.get_recovery_actions();
        if recovery_actions.is_empty() {
            return Err(TrustformersError::Resource {
                message: "No recovery actions available for this error".to_string(),
                resource_type: "recovery_actions".to_string(),
                current_usage: Some(format!("Original error: {}", error)),
                suggestion: Some(
                    "Manually handle the error or add recovery strategies".to_string(),
                ),
                recovery_actions: Vec::new(),
            });
        }

        let mut recovery_context = RecoveryContext::new(operation_name, self.policy.max_attempts);
        let mut current_delay = self.policy.retry_delay_ms;

        for (attempt, action) in recovery_actions.iter().enumerate() {
            if !recovery_context.can_retry() {
                break;
            }

            log::info!(
                "🔄 Attempting recovery {}/{} for '{}': {:?}",
                attempt + 1,
                self.policy.max_attempts,
                operation_name,
                action
            );

            let start_time = std::time::Instant::now();

            // Execute recovery action
            if let Err(recovery_err) = self.execute_recovery_action(action).await {
                log::warn!("❌ Recovery action failed: {}", recovery_err);
                recovery_context.record_attempt(recovery_err.to_string());
                continue;
            }

            // Apply delay with exponential backoff
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(current_delay)).await;
                current_delay = std::cmp::min(
                    (current_delay as f64 * self.policy.backoff_factor) as u64,
                    self.policy.max_delay_ms,
                );
            }

            // Retry the original operation
            match operation().await {
                Ok(result) => {
                    let duration = start_time.elapsed().as_millis() as u64;

                    // Record successful recovery
                    self.record_recovery_attempt(RecoveryAttempt {
                        timestamp: std::time::SystemTime::now(),
                        action: action.clone(),
                        success: true,
                        duration_ms: duration,
                        error_type: self.error_type_name(error),
                    });

                    // Update circuit breaker on success
                    if let Some(breaker) = self.circuit_breakers.get_mut(operation_name) {
                        breaker.record_success();
                    }

                    log::info!(
                        "✅ Recovery successful for '{}' using {:?} (took {}ms)",
                        operation_name,
                        action,
                        duration
                    );

                    return Ok(result);
                },
                Err(retry_error) => {
                    let duration = start_time.elapsed().as_millis() as u64;

                    // Record failed recovery
                    self.record_recovery_attempt(RecoveryAttempt {
                        timestamp: std::time::SystemTime::now(),
                        action: action.clone(),
                        success: false,
                        duration_ms: duration,
                        error_type: self.error_type_name(error),
                    });

                    recovery_context.record_attempt(retry_error.to_string());
                    recovery_context.record_recovery(action.clone());

                    log::warn!(
                        "❌ Recovery attempt failed for '{}': {}",
                        operation_name,
                        retry_error
                    );
                },
            }
        }

        // Update circuit breaker on final failure
        self.get_or_create_circuit_breaker(operation_name).record_failure();

        // Return the original error with recovery context
        Err(TrustformersError::Core(
            CoreTrustformersError::invalid_input_simple(format!(
                "Recovery failed after {} attempts. Last error: {:?}. Recovery history: {:?}",
                recovery_context.attempts,
                recovery_context.last_error,
                recovery_context.recovery_history
            )),
        ))
    }

    /// Execute a specific recovery action
    async fn execute_recovery_action(&mut self, action: &RecoveryAction) -> Result<()> {
        match action {
            RecoveryAction::Retry {
                max_attempts: _,
                delay_ms,
            } => {
                tokio::time::sleep(std::time::Duration::from_millis(*delay_ms)).await;
                Ok(())
            },
            RecoveryAction::FallbackToCpu => {
                log::info!("🔄 Switching to CPU execution mode");
                // Implementation would set global CPU mode flag
                Ok(())
            },
            RecoveryAction::ReduceBatchSize { factor } => {
                log::info!("🔄 Reducing batch size by factor: {}", factor);
                // Implementation would adjust global batch size setting
                Ok(())
            },
            RecoveryAction::ClearCache => {
                log::info!("🔄 Clearing model cache");
                // Implementation would clear model cache
                Ok(())
            },
            RecoveryAction::ReduceMemoryUsage { reduction_factor } => {
                log::info!("🔄 Reducing memory usage by factor: {}", reduction_factor);
                // Implementation would adjust memory allocation settings
                Ok(())
            },
            RecoveryAction::ReducePrecision { target_precision } => {
                log::info!("🔄 Reducing precision to: {}", target_precision);
                // Implementation would adjust model precision settings
                Ok(())
            },
            RecoveryAction::RedownloadModel => {
                log::info!("🔄 Re-downloading model files");
                // Implementation would trigger model re-download
                Ok(())
            },
            RecoveryAction::UseOfflineMode => {
                log::info!("🔄 Switching to offline mode");
                // Implementation would enable offline mode
                Ok(())
            },
            RecoveryAction::UseAlternativeModel { model_name } => {
                log::info!("🔄 Switching to alternative model: {}", model_name);
                // Implementation would switch to alternative model
                Ok(())
            },
            RecoveryAction::Custom {
                action,
                description,
            } => {
                log::info!("🔄 Executing custom recovery: {} ({})", action, description);
                // Implementation would execute custom recovery logic
                Ok(())
            },
        }
    }

    /// Get recommended recovery strategies based on error history
    pub fn get_recommended_strategies(&self, error_type: &str) -> Vec<RecoveryAction> {
        let history = self.recovery_history.get(error_type);

        if let Some(attempts) = history {
            if self.policy.learning_enabled {
                // Return strategies that have been successful for this error type
                let successful_actions: Vec<_> = attempts
                    .iter()
                    .filter(|attempt| attempt.success)
                    .map(|attempt| attempt.action.clone())
                    .collect();

                if !successful_actions.is_empty() {
                    return successful_actions;
                }
            }
        }

        // Default strategies based on error type
        match error_type {
            "memory" | "resource" => vec![
                RecoveryAction::ReduceBatchSize { factor: 0.5 },
                RecoveryAction::ReduceMemoryUsage {
                    reduction_factor: 0.7,
                },
                RecoveryAction::FallbackToCpu,
            ],
            "network" | "hub" => vec![
                RecoveryAction::Retry {
                    max_attempts: 3,
                    delay_ms: 2000,
                },
                RecoveryAction::UseOfflineMode,
            ],
            "model" => vec![RecoveryAction::RedownloadModel, RecoveryAction::ClearCache],
            _ => vec![
                RecoveryAction::Retry {
                    max_attempts: 2,
                    delay_ms: 1000,
                },
                RecoveryAction::FallbackToCpu,
            ],
        }
    }

    /// Reset circuit breaker for an operation
    pub fn reset_circuit_breaker(&mut self, operation_name: &str) {
        if let Some(breaker) = self.circuit_breakers.get_mut(operation_name) {
            breaker.reset();
            log::info!("🔄 Circuit breaker reset for operation: {}", operation_name);
        }
    }

    /// Get recovery statistics
    pub fn get_recovery_stats(&self) -> RecoveryStats {
        let total_attempts: usize =
            self.recovery_history.values().map(|attempts| attempts.len()).sum();

        let successful_attempts: usize = self
            .recovery_history
            .values()
            .flat_map(|attempts| attempts.iter())
            .filter(|attempt| attempt.success)
            .count();

        let success_rate = if total_attempts > 0 {
            (successful_attempts as f64 / total_attempts as f64) * 100.0
        } else {
            0.0
        };

        RecoveryStats {
            total_attempts,
            successful_attempts,
            success_rate,
            error_types: self.recovery_history.len(),
            circuit_breakers_active: self
                .circuit_breakers
                .values()
                .filter(|cb| cb.state != CircuitBreakerState::Closed)
                .count(),
        }
    }

    fn record_recovery_attempt(&mut self, attempt: RecoveryAttempt) {
        let error_type = attempt.error_type.clone();
        self.recovery_history.entry(error_type).or_default().push(attempt);
    }

    fn error_type_name(&self, error: &TrustformersError) -> String {
        match error {
            TrustformersError::Pipeline { .. } => "pipeline".to_string(),
            TrustformersError::Model { .. } => "model".to_string(),
            TrustformersError::Hub { .. } => "hub".to_string(),
            TrustformersError::AutoConfig { .. } => "autoconfig".to_string(),
            TrustformersError::FeatureUnavailable { .. } => "feature".to_string(),
            TrustformersError::Resource { .. } => "resource".to_string(),
            TrustformersError::InvalidInput { .. } => "input".to_string(),
            TrustformersError::Core(_) => "core".to_string(),
            TrustformersError::Io { .. } => "io".to_string(),
            TrustformersError::Network { .. } => "network".to_string(),
            TrustformersError::PipelineNotFound { .. } => "pipeline_not_found".to_string(),
        }
    }

    fn get_or_create_circuit_breaker(&mut self, operation_name: &str) -> &mut CircuitBreaker {
        self.circuit_breakers.entry(operation_name.to_string()).or_insert_with(|| {
            CircuitBreaker::new(
                self.policy.circuit_breaker_threshold,
                std::time::Duration::from_millis(self.policy.circuit_breaker_timeout_ms),
            )
        })
    }
}

impl CircuitBreaker {
    pub fn new(threshold: u32, timeout: std::time::Duration) -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            threshold,
            timeout,
        }
    }

    pub fn is_open(&mut self) -> bool {
        match self.state {
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.timeout {
                        self.state = CircuitBreakerState::HalfOpen;
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            },
            _ => false,
        }
    }

    pub fn record_success(&mut self) {
        self.success_count += 1;
        if self.state == CircuitBreakerState::HalfOpen {
            self.state = CircuitBreakerState::Closed;
            self.failure_count = 0;
        }
    }

    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(std::time::Instant::now());

        if self.failure_count >= self.threshold {
            self.state = CircuitBreakerState::Open;
        }
    }

    pub fn reset(&mut self) {
        self.state = CircuitBreakerState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.last_failure_time = None;
    }
}

/// Statistics about recovery attempts
#[derive(Debug, Clone)]
pub struct RecoveryStats {
    pub total_attempts: usize,
    pub successful_attempts: usize,
    pub success_rate: f64,
    pub error_types: usize,
    pub circuit_breakers_active: usize,
}

impl Default for AutoRecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global recovery manager instance
static GLOBAL_RECOVERY_MANAGER: std::sync::OnceLock<
    std::sync::Arc<tokio::sync::Mutex<AutoRecoveryManager>>,
> = std::sync::OnceLock::new();

/// Get the global recovery manager instance
pub fn global_recovery_manager() -> std::sync::Arc<tokio::sync::Mutex<AutoRecoveryManager>> {
    GLOBAL_RECOVERY_MANAGER
        .get_or_init(|| std::sync::Arc::new(tokio::sync::Mutex::new(AutoRecoveryManager::new())))
        .clone()
}

/// Convenience function for automatic error recovery
pub async fn with_auto_recovery<T, F, Fut>(operation_name: &str, operation: F) -> Result<T>
where
    F: Fn() -> Fut + Clone,
    Fut: std::future::Future<Output = Result<T>>,
{
    match operation().await {
        Ok(result) => Ok(result),
        Err(error) => {
            let manager_guard = global_recovery_manager();
            let mut manager = manager_guard.lock().await;
            manager.attempt_recovery(operation_name, &error, operation).await
        },
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for TrustformersError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("TrustformersError", 3)?;
        state.serialize_field("type", &format!("{:?}", self))?;
        state.serialize_field("message", &self.to_string())?;
        state.serialize_field("recovery_actions", &self.get_recovery_actions())?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_error_creation() {
        let err = TrustformersError::pipeline("Test error", "text-generation");
        assert!(err.to_string().contains("Pipeline Error"));
        assert!(!err.get_recovery_actions().is_empty());
    }

    #[test]
    fn test_model_error_with_suggestions() {
        let err = TrustformersError::model("Model not found", "gpt2-large");
        let actions = err.get_recovery_actions();
        assert!(actions.iter().any(|a| matches!(a, RecoveryAction::UseAlternativeModel { .. })));
    }

    #[test]
    fn test_recovery_context() {
        let mut context = RecoveryContext::new("test_operation", 3);
        assert!(context.can_retry());

        context.record_attempt("First error");
        assert_eq!(context.attempts, 1);
        assert!(context.can_retry());

        context.record_recovery(RecoveryAction::FallbackToCpu);
        assert_eq!(context.recovery_history.len(), 1);
    }

    #[test]
    fn test_alternative_model_suggestion() {
        assert_eq!(
            TrustformersError::suggest_alternative_model("gpt2-large"),
            "gpt2-base"
        );
        assert_eq!(
            TrustformersError::suggest_alternative_model("bert-large-uncased"),
            "bert-base-uncased"
        );
    }
}
