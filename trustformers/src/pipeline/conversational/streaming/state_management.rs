//! Stream state management and error recovery for conversational streaming.
//!
//! This module provides comprehensive state management functionality for streaming
//! conversational AI responses, including:
//! - Stream state tracking and transitions
//! - Performance and quality monitoring
//! - Error detection and recovery strategies
//! - Resource management and cleanup
//! - Health monitoring and diagnostics
//! - Session persistence across interruptions

use super::types::*;
use crate::error::{Result, TrustformersError};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::sleep;

// ================================================================================================
// STATE MANAGEMENT TYPES
// ================================================================================================

/// Stream state manager for maintaining streaming sessions
#[derive(Debug)]
pub struct StreamStateManager {
    /// Configuration
    config: AdvancedStreamingConfig,
    /// Current state
    current_state: Arc<RwLock<StreamState>>,
    /// State history for debugging
    state_history: Arc<RwLock<VecDeque<StateTransition>>>,
    /// Error recovery manager
    error_recovery: ErrorRecoveryManager,
}

/// Current stream state
#[derive(Debug, Clone)]
pub struct StreamState {
    /// Current connection status
    pub connection: StreamConnection,
    /// Buffer state
    pub buffer: BufferState,
    /// Performance metrics
    pub performance: StreamPerformance,
    /// Quality metrics
    pub quality: StreamingQuality,
    /// Error information
    pub error_info: Option<StreamError>,
    /// Last state update
    pub last_update: Instant,
}

/// Stream performance metrics
#[derive(Debug, Clone)]
pub struct StreamPerformance {
    /// Current throughput (chunks/sec)
    pub throughput: f32,
    /// Latency metrics
    pub latency: LatencyMetrics,
    /// Resource utilization
    pub resource_usage: ResourceUsage,
    /// Network metrics
    pub network: NetworkMetrics,
}

/// Latency measurement metrics
#[derive(Debug, Clone)]
pub struct LatencyMetrics {
    /// Current latency (ms)
    pub current_ms: f64,
    /// Average latency (ms)
    pub average_ms: f64,
    /// 95th percentile latency (ms)
    pub p95_ms: f64,
    /// 99th percentile latency (ms)
    pub p99_ms: f64,
    /// Maximum latency seen (ms)
    pub max_ms: f64,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU usage percentage
    pub cpu_percent: f32,
    /// Memory usage in MB
    pub memory_mb: f64,
    /// Network bandwidth usage (Mbps)
    pub bandwidth_mbps: f32,
    /// File descriptor usage
    pub fd_count: usize,
}

/// Network performance metrics
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    /// Packets sent
    pub packets_sent: usize,
    /// Packets lost
    pub packets_lost: usize,
    /// Bandwidth utilization
    pub bandwidth_utilization: f32,
    /// Connection quality score
    pub connection_quality: f32,
}

/// State transition for tracking changes
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// Timestamp of transition
    pub timestamp: Instant,
    /// Previous state
    pub from_state: StreamConnection,
    /// New state
    pub to_state: StreamConnection,
    /// Reason for transition
    pub reason: String,
    /// Additional context
    pub context: Option<String>,
}

/// Stream error information
#[derive(Debug, Clone)]
pub struct StreamError {
    /// Error type
    pub error_type: StreamErrorType,
    /// Error message
    pub message: String,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Timestamp when error occurred
    pub timestamp: Instant,
    /// Recovery strategy
    pub recovery_strategy: Option<RecoveryStrategy>,
    /// Error context
    pub context: HashMap<String, String>,
}

/// Types of streaming errors
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StreamErrorType {
    ConnectionLost,
    BufferOverflow,
    BufferUnderflow,
    TimeoutError,
    NetworkError,
    ProcessingError,
    ConfigurationError,
    ResourceExhaustion,
    QualityDegradation,
    SecurityViolation,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Recovery strategies for different error types
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    Retry,
    Reconnect,
    BufferAdjustment,
    QualityReduction,
    Fallback,
    Restart,
    GracefulShutdown,
}

// ================================================================================================
// STREAM STATE MANAGER IMPLEMENTATION
// ================================================================================================

impl StreamStateManager {
    /// Create a new stream state manager
    pub fn new(config: AdvancedStreamingConfig) -> Self {
        Self {
            config,
            current_state: Arc::new(RwLock::new(StreamState::default())),
            state_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            error_recovery: ErrorRecoveryManager::new(),
        }
    }

    /// Update stream state
    pub async fn update_state(
        &self,
        new_connection: StreamConnection,
        reason: String,
    ) -> Result<()> {
        let mut state = self.current_state.write().await;
        let old_connection = state.connection.clone();

        state.connection = new_connection.clone();
        state.last_update = Instant::now();

        // Record state transition
        let transition = StateTransition {
            timestamp: Instant::now(),
            from_state: old_connection,
            to_state: new_connection,
            reason,
            context: None,
        };

        let mut history = self.state_history.write().await;
        history.push_back(transition);

        // Keep only recent history
        if history.len() > 100 {
            history.pop_front();
        }

        Ok(())
    }

    /// Get current stream state
    pub async fn get_current_state(&self) -> StreamState {
        self.current_state.read().await.clone()
    }

    /// Update buffer state
    pub async fn update_buffer_state(&self, buffer_state: BufferState) -> Result<()> {
        let mut state = self.current_state.write().await;
        state.buffer = buffer_state;
        state.last_update = Instant::now();
        Ok(())
    }

    /// Update performance metrics
    pub async fn update_performance(&self, performance: StreamPerformance) -> Result<()> {
        let mut state = self.current_state.write().await;
        state.performance = performance;
        state.last_update = Instant::now();
        Ok(())
    }

    /// Update quality metrics
    pub async fn update_quality(&self, quality: StreamingQuality) -> Result<()> {
        let mut state = self.current_state.write().await;
        state.quality = quality;
        state.last_update = Instant::now();
        Ok(())
    }

    /// Record error
    pub async fn record_error(&self, error: StreamError) -> Result<()> {
        let mut state = self.current_state.write().await;

        // Update connection state based on error severity
        if error.severity >= ErrorSeverity::High {
            match error.error_type {
                StreamErrorType::ConnectionLost => {
                    state.connection = StreamConnection::Disconnected;
                },
                StreamErrorType::BufferOverflow | StreamErrorType::BufferUnderflow => {
                    state.connection = StreamConnection::Buffering;
                },
                StreamErrorType::NetworkError => {
                    state.connection = StreamConnection::Reconnecting;
                },
                _ => {
                    state.connection = StreamConnection::Error(error.message.clone());
                },
            }
        }

        state.error_info = Some(error.clone());
        state.last_update = Instant::now();

        // Drop the lock before initiating error recovery to avoid deadlock
        drop(state);

        // Initiate error recovery if enabled
        if self.config.enable_error_recovery {
            self.error_recovery.handle_error(&error).await?;
        }

        Ok(())
    }

    /// Clear error state
    pub async fn clear_error(&self) -> Result<()> {
        let mut state = self.current_state.write().await;
        state.error_info = None;
        state.last_update = Instant::now();
        Ok(())
    }

    /// Get state history
    pub async fn get_state_history(&self) -> Vec<StateTransition> {
        self.state_history.read().await.iter().cloned().collect()
    }

    /// Check if state is healthy
    pub async fn is_healthy(&self) -> bool {
        let state = self.current_state.read().await;

        match state.connection {
            StreamConnection::Connected | StreamConnection::Streaming => {
                state.error_info.is_none()
                    && state.buffer.utilization < 0.9
                    && state.performance.throughput > 0.0
            },
            _ => false,
        }
    }

    /// Get detailed health status
    pub async fn get_health_status(&self) -> HealthStatus {
        let state = self.current_state.read().await;

        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        // Check connection status
        match &state.connection {
            StreamConnection::Error(msg) => {
                issues.push(format!("Connection error: {}", msg));
            },
            StreamConnection::Disconnected => {
                issues.push("Connection lost".to_string());
            },
            StreamConnection::Reconnecting => {
                warnings.push("Attempting to reconnect".to_string());
            },
            _ => {},
        }

        // Check buffer health
        if state.buffer.utilization > 0.95 {
            issues.push("Buffer nearly full".to_string());
        } else if state.buffer.utilization > 0.8 {
            warnings.push("Buffer utilization high".to_string());
        }

        // Check performance metrics
        if state.performance.throughput < 1.0 {
            warnings.push("Low throughput detected".to_string());
        }

        if state.performance.latency.current_ms > 1000.0 {
            warnings.push("High latency detected".to_string());
        }

        // Check for errors
        if let Some(ref error) = state.error_info {
            match error.severity {
                ErrorSeverity::Critical | ErrorSeverity::High => {
                    issues.push(format!("Error: {}", error.message));
                },
                ErrorSeverity::Medium => {
                    warnings.push(format!("Warning: {}", error.message));
                },
                _ => {},
            }
        }

        let overall_status = if !issues.is_empty() {
            OverallHealthStatus::Unhealthy
        } else if !warnings.is_empty() {
            OverallHealthStatus::Degraded
        } else {
            OverallHealthStatus::Healthy
        };

        HealthStatus {
            overall_status,
            connection_status: state.connection.clone(),
            buffer_utilization: state.buffer.utilization,
            throughput: state.performance.throughput,
            latency_ms: state.performance.latency.current_ms,
            issues,
            warnings,
            last_update: state.last_update,
        }
    }

    /// Reset error recovery attempts
    pub async fn reset_recovery_attempts(&self, error_type: &StreamErrorType) {
        self.error_recovery.reset_attempts(error_type).await;
    }

    /// Get recovery attempt count
    pub async fn get_recovery_attempts(&self, error_type: &StreamErrorType) -> usize {
        self.error_recovery.get_attempts(error_type).await
    }
}

// ================================================================================================
// ERROR RECOVERY MANAGER
// ================================================================================================

/// Error recovery manager for handling streaming failures
#[derive(Debug)]
pub struct ErrorRecoveryManager {
    /// Recovery strategies mapping
    strategies: HashMap<StreamErrorType, Vec<RecoveryStrategy>>,
    /// Recovery attempt tracking
    recovery_attempts: Arc<RwLock<HashMap<StreamErrorType, usize>>>,
    /// Maximum recovery attempts
    max_attempts: usize,
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager
    pub fn new() -> Self {
        let mut strategies = HashMap::new();

        strategies.insert(
            StreamErrorType::ConnectionLost,
            vec![
                RecoveryStrategy::Reconnect,
                RecoveryStrategy::BufferAdjustment,
                RecoveryStrategy::Restart,
            ],
        );

        strategies.insert(
            StreamErrorType::BufferOverflow,
            vec![
                RecoveryStrategy::BufferAdjustment,
                RecoveryStrategy::QualityReduction,
                RecoveryStrategy::Fallback,
            ],
        );

        strategies.insert(
            StreamErrorType::BufferUnderflow,
            vec![
                RecoveryStrategy::BufferAdjustment,
                RecoveryStrategy::Retry,
                RecoveryStrategy::QualityReduction,
            ],
        );

        strategies.insert(
            StreamErrorType::TimeoutError,
            vec![
                RecoveryStrategy::Retry,
                RecoveryStrategy::BufferAdjustment,
                RecoveryStrategy::Reconnect,
            ],
        );

        strategies.insert(
            StreamErrorType::NetworkError,
            vec![
                RecoveryStrategy::Retry,
                RecoveryStrategy::Reconnect,
                RecoveryStrategy::QualityReduction,
            ],
        );

        strategies.insert(
            StreamErrorType::ProcessingError,
            vec![
                RecoveryStrategy::Retry,
                RecoveryStrategy::QualityReduction,
                RecoveryStrategy::Fallback,
            ],
        );

        strategies.insert(
            StreamErrorType::QualityDegradation,
            vec![
                RecoveryStrategy::QualityReduction,
                RecoveryStrategy::BufferAdjustment,
                RecoveryStrategy::Fallback,
            ],
        );

        strategies.insert(
            StreamErrorType::ResourceExhaustion,
            vec![
                RecoveryStrategy::QualityReduction,
                RecoveryStrategy::BufferAdjustment,
                RecoveryStrategy::GracefulShutdown,
            ],
        );

        Self {
            strategies,
            recovery_attempts: Arc::new(RwLock::new(HashMap::new())),
            max_attempts: 3,
        }
    }

    /// Handle error and attempt recovery
    pub async fn handle_error(&self, error: &StreamError) -> Result<()> {
        let error_type = &error.error_type;

        // Check recovery attempts
        let mut attempts = self.recovery_attempts.write().await;
        let current_attempts = attempts.get(error_type).copied().unwrap_or(0);

        if current_attempts >= self.max_attempts {
            // Max attempts reached, give up
            return Err(TrustformersError::runtime_error(format!(
                "Max recovery attempts ({}) reached for error type: {:?}",
                self.max_attempts, error_type
            )));
        }

        // Get recovery strategies for this error type
        if let Some(strategies) = self.strategies.get(error_type) {
            if let Some(strategy) = strategies.get(current_attempts) {
                // Execute recovery strategy
                self.execute_recovery_strategy(strategy.clone(), error).await?;

                // Update attempt count
                attempts.insert(error_type.clone(), current_attempts + 1);
            }
        }

        Ok(())
    }

    /// Execute a specific recovery strategy
    async fn execute_recovery_strategy(
        &self,
        strategy: RecoveryStrategy,
        error: &StreamError,
    ) -> Result<()> {
        match strategy {
            RecoveryStrategy::Retry => {
                // Simple retry with exponential backoff
                let delay = Duration::from_millis(100 * (2_u64.pow(error.context.len() as u32)));
                sleep(delay).await;
            },
            RecoveryStrategy::Reconnect => {
                // Attempt to reconnect (placeholder implementation)
                sleep(Duration::from_millis(500)).await;
            },
            RecoveryStrategy::BufferAdjustment => {
                // Adjust buffer sizes (placeholder implementation)
                sleep(Duration::from_millis(100)).await;
            },
            RecoveryStrategy::QualityReduction => {
                // Reduce streaming quality (placeholder implementation)
                sleep(Duration::from_millis(50)).await;
            },
            RecoveryStrategy::Fallback => {
                // Switch to fallback mode (placeholder implementation)
                sleep(Duration::from_millis(200)).await;
            },
            RecoveryStrategy::Restart => {
                // Restart streaming session (placeholder implementation)
                sleep(Duration::from_millis(1000)).await;
            },
            RecoveryStrategy::GracefulShutdown => {
                // Gracefully shutdown (placeholder implementation)
                sleep(Duration::from_millis(100)).await;
            },
        }

        Ok(())
    }

    /// Reset recovery attempts for an error type
    pub async fn reset_attempts(&self, error_type: &StreamErrorType) {
        let mut attempts = self.recovery_attempts.write().await;
        attempts.remove(error_type);
    }

    /// Get current recovery attempts
    pub async fn get_attempts(&self, error_type: &StreamErrorType) -> usize {
        *self.recovery_attempts.read().await.get(error_type).unwrap_or(&0)
    }

    /// Check if recovery is possible for an error type
    pub fn can_recover(&self, error_type: &StreamErrorType) -> bool {
        self.strategies.contains_key(error_type)
    }

    /// Get available recovery strategies for an error type
    pub fn get_strategies(&self, error_type: &StreamErrorType) -> Option<&Vec<RecoveryStrategy>> {
        self.strategies.get(error_type)
    }
}

// ================================================================================================
// HEALTH MONITORING TYPES
// ================================================================================================

/// Overall health status
#[derive(Debug, Clone, PartialEq)]
pub enum OverallHealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Comprehensive health status
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub overall_status: OverallHealthStatus,
    pub connection_status: StreamConnection,
    pub buffer_utilization: f32,
    pub throughput: f32,
    pub latency_ms: f64,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
    pub last_update: Instant,
}

// ================================================================================================
// DEFAULT IMPLEMENTATIONS
// ================================================================================================

impl Default for StreamState {
    fn default() -> Self {
        Self {
            connection: StreamConnection::Connecting,
            buffer: BufferState {
                current_size: 0,
                max_size: 1000,
                utilization: 0.0,
                pending_chunks: 0,
            },
            performance: StreamPerformance::default(),
            quality: StreamingQuality::default(),
            error_info: None,
            last_update: Instant::now(),
        }
    }
}

impl Default for StreamPerformance {
    fn default() -> Self {
        Self {
            throughput: 0.0,
            latency: LatencyMetrics::default(),
            resource_usage: ResourceUsage::default(),
            network: NetworkMetrics::default(),
        }
    }
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            current_ms: 0.0,
            average_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            max_ms: 0.0,
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_mb: 0.0,
            bandwidth_mbps: 0.0,
            fd_count: 0,
        }
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            packets_sent: 0,
            packets_lost: 0,
            bandwidth_utilization: 0.0,
            connection_quality: 1.0,
        }
    }
}

impl Default for ErrorRecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// UTILITY IMPLEMENTATIONS
// ================================================================================================

impl HealthStatus {
    /// Check if the system is in a good state
    pub fn is_healthy(&self) -> bool {
        matches!(self.overall_status, OverallHealthStatus::Healthy)
    }

    /// Check if the system is degraded but operational
    pub fn is_degraded(&self) -> bool {
        matches!(self.overall_status, OverallHealthStatus::Degraded)
    }

    /// Check if the system is unhealthy
    pub fn is_unhealthy(&self) -> bool {
        matches!(self.overall_status, OverallHealthStatus::Unhealthy)
    }

    /// Get a summary of the health status
    pub fn summary(&self) -> String {
        match self.overall_status {
            OverallHealthStatus::Healthy => "System is healthy".to_string(),
            OverallHealthStatus::Degraded => {
                format!("System is degraded: {} warnings", self.warnings.len())
            },
            OverallHealthStatus::Unhealthy => {
                format!("System is unhealthy: {} issues", self.issues.len())
            },
        }
    }
}

impl StreamError {
    /// Create a new stream error
    pub fn new(error_type: StreamErrorType, message: String, severity: ErrorSeverity) -> Self {
        Self {
            error_type,
            message,
            severity,
            timestamp: Instant::now(),
            recovery_strategy: None,
            context: HashMap::new(),
        }
    }

    /// Add context to the error
    pub fn with_context(mut self, key: String, value: String) -> Self {
        self.context.insert(key, value);
        self
    }

    /// Set recovery strategy
    pub fn with_recovery_strategy(mut self, strategy: RecoveryStrategy) -> Self {
        self.recovery_strategy = Some(strategy);
        self
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        self.recovery_strategy.is_some() && !matches!(self.severity, ErrorSeverity::Critical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AdvancedStreamingConfig {
        AdvancedStreamingConfig::default()
    }

    // ---- StreamError tests ----

    #[test]
    fn test_stream_error_new_has_no_recovery_strategy() {
        let err = StreamError::new(
            StreamErrorType::TimeoutError,
            "timed out".to_string(),
            ErrorSeverity::Medium,
        );
        assert!(
            err.recovery_strategy.is_none(),
            "new error must have no recovery strategy"
        );
    }

    #[test]
    fn test_stream_error_with_recovery_strategy() {
        let err = StreamError::new(
            StreamErrorType::ConnectionLost,
            "lost".to_string(),
            ErrorSeverity::High,
        )
        .with_recovery_strategy(RecoveryStrategy::Reconnect);
        assert!(
            err.recovery_strategy.is_some(),
            "error with recovery strategy must have Some strategy"
        );
    }

    #[test]
    fn test_stream_error_is_recoverable_with_strategy() {
        let err = StreamError::new(
            StreamErrorType::NetworkError,
            "net err".to_string(),
            ErrorSeverity::Medium,
        )
        .with_recovery_strategy(RecoveryStrategy::Retry);
        assert!(
            err.is_recoverable(),
            "error with strategy and non-critical severity must be recoverable"
        );
    }

    #[test]
    fn test_stream_error_critical_not_recoverable() {
        let err = StreamError::new(
            StreamErrorType::SecurityViolation,
            "critical".to_string(),
            ErrorSeverity::Critical,
        )
        .with_recovery_strategy(RecoveryStrategy::GracefulShutdown);
        assert!(
            !err.is_recoverable(),
            "critical error must not be considered recoverable"
        );
    }

    #[test]
    fn test_stream_error_with_context_stored() {
        let err = StreamError::new(
            StreamErrorType::BufferOverflow,
            "overflow".to_string(),
            ErrorSeverity::High,
        )
        .with_context("session_id".to_string(), "abc-123".to_string());
        assert_eq!(
            err.context.get("session_id").map(String::as_str),
            Some("abc-123"),
            "context key must be stored"
        );
    }

    // ---- ErrorSeverity ordering ----

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Low < ErrorSeverity::Medium, "Low < Medium");
        assert!(ErrorSeverity::Medium < ErrorSeverity::High, "Medium < High");
        assert!(
            ErrorSeverity::High < ErrorSeverity::Critical,
            "High < Critical"
        );
    }

    // ---- ErrorRecoveryManager tests ----

    #[test]
    fn test_error_recovery_manager_new_has_no_attempts() {
        let mgr = ErrorRecoveryManager::new();
        // No async test needed; can_recover is synchronous
        assert!(
            mgr.can_recover(&StreamErrorType::ConnectionLost),
            "manager must have recovery strategies for ConnectionLost"
        );
    }

    #[test]
    fn test_error_recovery_manager_strategies_for_unknown_type() {
        let mgr = ErrorRecoveryManager::new();
        assert!(
            !mgr.can_recover(&StreamErrorType::ConfigurationError),
            "ConfigurationError should not have a recovery strategy by default"
        );
    }

    #[test]
    fn test_error_recovery_manager_get_strategies() {
        let mgr = ErrorRecoveryManager::new();
        let strategies = mgr.get_strategies(&StreamErrorType::BufferOverflow);
        assert!(
            strategies.is_some(),
            "BufferOverflow must have recovery strategies"
        );
        assert!(
            !strategies.expect("strategies must be Some").is_empty(),
            "strategies list must not be empty"
        );
    }

    // ---- HealthStatus tests ----

    #[test]
    fn test_health_status_healthy() {
        let hs = HealthStatus {
            overall_status: OverallHealthStatus::Healthy,
            connection_status: StreamConnection::Streaming,
            buffer_utilization: 0.2,
            throughput: 10.0,
            latency_ms: 50.0,
            issues: vec![],
            warnings: vec![],
            last_update: Instant::now(),
        };
        assert!(
            hs.is_healthy(),
            "OverallHealthStatus::Healthy must return true from is_healthy"
        );
        assert!(!hs.is_degraded(), "Healthy must not be degraded");
        assert!(!hs.is_unhealthy(), "Healthy must not be unhealthy");
    }

    #[test]
    fn test_health_status_degraded() {
        let hs = HealthStatus {
            overall_status: OverallHealthStatus::Degraded,
            connection_status: StreamConnection::Connected,
            buffer_utilization: 0.8,
            throughput: 3.0,
            latency_ms: 300.0,
            issues: vec![],
            warnings: vec!["High latency".to_string()],
            last_update: Instant::now(),
        };
        assert!(!hs.is_healthy(), "Degraded must not be healthy");
        assert!(
            hs.is_degraded(),
            "Degraded must return true from is_degraded"
        );
    }

    #[test]
    fn test_health_status_unhealthy() {
        let hs = HealthStatus {
            overall_status: OverallHealthStatus::Unhealthy,
            connection_status: StreamConnection::Disconnected,
            buffer_utilization: 1.0,
            throughput: 0.0,
            latency_ms: 9999.0,
            issues: vec!["Connection lost".to_string()],
            warnings: vec![],
            last_update: Instant::now(),
        };
        assert!(
            hs.is_unhealthy(),
            "Unhealthy must return true from is_unhealthy"
        );
    }

    #[test]
    fn test_health_status_summary_healthy() {
        let hs = HealthStatus {
            overall_status: OverallHealthStatus::Healthy,
            connection_status: StreamConnection::Streaming,
            buffer_utilization: 0.1,
            throughput: 20.0,
            latency_ms: 10.0,
            issues: vec![],
            warnings: vec![],
            last_update: Instant::now(),
        };
        assert!(
            hs.summary().contains("healthy"),
            "healthy summary must mention 'healthy'"
        );
    }

    // ---- StreamStateManager async tests ----

    #[tokio::test]
    async fn test_state_manager_initial_connection_is_connecting() {
        let mgr = StreamStateManager::new(default_config());
        let state = mgr.get_current_state().await;
        assert_eq!(
            state.connection,
            StreamConnection::Connecting,
            "initial connection state must be Connecting"
        );
    }

    #[tokio::test]
    async fn test_state_manager_update_state_transition() {
        let mgr = StreamStateManager::new(default_config());
        mgr.update_state(StreamConnection::Connected, "test transition".to_string())
            .await
            .expect("update_state must succeed");
        let state = mgr.get_current_state().await;
        assert_eq!(
            state.connection,
            StreamConnection::Connected,
            "state must be Connected after update"
        );
    }

    #[tokio::test]
    async fn test_state_manager_history_records_transition() {
        let mgr = StreamStateManager::new(default_config());
        mgr.update_state(StreamConnection::Streaming, "started streaming".to_string())
            .await
            .expect("update_state must succeed");
        let history = mgr.get_state_history().await;
        assert_eq!(history.len(), 1, "one transition must be recorded");
        assert_eq!(
            history[0].to_state,
            StreamConnection::Streaming,
            "transition target must be Streaming"
        );
    }

    #[tokio::test]
    async fn test_state_manager_update_buffer_state() {
        let mgr = StreamStateManager::new(default_config());
        let buf = BufferState {
            current_size: 50,
            max_size: 100,
            utilization: 0.5,
            pending_chunks: 3,
        };
        mgr.update_buffer_state(buf).await.expect("update_buffer_state must succeed");
        let state = mgr.get_current_state().await;
        assert_eq!(
            state.buffer.current_size, 50,
            "buffer current_size must be updated"
        );
    }

    #[tokio::test]
    async fn test_state_manager_clear_error() {
        let mgr = StreamStateManager::new(default_config());
        // record a low-severity error first (won't change connection state)
        let err = StreamError::new(
            StreamErrorType::ProcessingError,
            "minor error".to_string(),
            ErrorSeverity::Low,
        );
        mgr.record_error(err).await.expect("record_error must succeed");
        mgr.clear_error().await.expect("clear_error must succeed");
        let state = mgr.get_current_state().await;
        assert!(
            state.error_info.is_none(),
            "error_info must be None after clear_error"
        );
    }

    #[tokio::test]
    async fn test_state_manager_is_not_healthy_initially() {
        let mgr = StreamStateManager::new(default_config());
        // Initially Connecting with throughput 0 → not healthy
        assert!(
            !mgr.is_healthy().await,
            "initial state should not be healthy"
        );
    }
}
