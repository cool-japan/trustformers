//! Error Recovery Manager
//!
//! Manager for centralized error handling and recovery.

use super::super::types::*;
use super::*;

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex as TokioMutex, RwLock as TokioRwLock};
use tracing::{info, instrument, warn};

// Explicitly import TestProfile from profiling_pipeline to avoid ambiguity

#[derive(Debug)]
pub struct ErrorRecoveryManager {
    /// Error recovery configuration
    config: Arc<TokioRwLock<ErrorRecoveryConfig>>,
    /// Circuit breaker states
    circuit_breakers: Arc<TokioMutex<HashMap<String, CircuitBreakerState>>>,
    /// Error statistics
    error_stats: Arc<ErrorStatistics>,
    /// Recovery strategies
    recovery_strategies: Arc<RecoveryStrategies>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

/// Circuit breaker state
#[derive(Debug, Clone)]
pub struct CircuitBreakerState {
    /// Current state
    pub state: CircuitState,
    /// Failure count
    pub failure_count: usize,
    /// Last failure time
    pub last_failure: Option<SystemTime>,
    /// Next retry time (for half-open state)
    pub next_retry: Option<SystemTime>,
}

impl Default for CircuitBreakerState {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            last_failure: None,
            next_retry: None,
        }
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Error statistics
#[derive(Debug, Default)]
pub struct ErrorStatistics {
    /// Total errors encountered
    pub total_errors: AtomicU64,
    /// Total recoveries attempted
    pub total_recoveries: AtomicU64,
    /// Successful recoveries
    pub successful_recoveries: AtomicU64,
    /// Failed recoveries
    pub failed_recoveries: AtomicU64,
    /// Circuit breaker trips
    pub circuit_breaker_trips: AtomicU64,
}

/// Recovery strategies
#[derive(Debug)]
pub struct RecoveryStrategies {
    /// Strategy handlers
    pub strategies: HashMap<ErrorType, Box<dyn RecoveryStrategy + Send + Sync>>,
}

impl Default for RecoveryStrategies {
    fn default() -> Self {
        let mut strategies: HashMap<ErrorType, Box<dyn RecoveryStrategy + Send + Sync>> =
            HashMap::new();
        strategies.insert(ErrorType::Timeout, Box::new(TimeoutRecoveryStrategy));
        strategies.insert(
            ErrorType::ResourceExhaustion,
            Box::new(ResourceRecoveryStrategy),
        );
        strategies.insert(ErrorType::NetworkError, Box::new(NetworkRecoveryStrategy));
        strategies.insert(
            ErrorType::ComponentFailure,
            Box::new(ComponentRecoveryStrategy),
        );

        Self { strategies }
    }
}

/// Error types
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ErrorType {
    Timeout,
    ResourceExhaustion,
    NetworkError,
    ComponentFailure,
    ValidationError,
    UnknownError,
}

/// Recovery strategy trait
pub trait RecoveryStrategy: std::fmt::Debug + Send + Sync {
    /// Attempt to recover from an error
    fn recover<'a>(
        &'a self,
        error: &'a anyhow::Error,
        context: &'a RecoveryContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TestCharacteristics>> + Send + 'a>>;
}

/// Recovery context
#[derive(Debug, Clone)]
pub struct RecoveryContext {
    /// Component that failed
    pub component: String,
    /// Error details
    pub error_details: String,
    /// Retry attempt number
    pub retry_attempt: usize,
    /// Original request context
    pub request_context: RequestContext,
}

/// Request context for recovery
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Test data
    pub test_data: TestExecutionData,
    /// Profiling options
    pub options: ProfilingOptions,
    /// Analysis phase (if applicable)
    pub phase: Option<AnalysisPhase>,
}

/// Timeout recovery strategy
#[derive(Debug)]
pub struct TimeoutRecoveryStrategy;

impl RecoveryStrategy for TimeoutRecoveryStrategy {
    fn recover<'a>(
        &'a self,
        _error: &'a anyhow::Error,
        context: &'a RecoveryContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TestCharacteristics>> + Send + 'a>>
    {
        Box::pin(async move {
            info!(
                "Attempting timeout recovery for component: {}",
                context.component
            );

            // Implement timeout recovery logic
            // For now, return a default result with reduced confidence
            let mut characteristics = TestCharacteristics::default();
            characteristics.analysis_metadata.confidence_score = 0.5; // Reduced confidence
            characteristics
                .analysis_metadata
                .notes
                .push("Recovered from timeout".to_string());

            Ok(characteristics)
        })
    }
}

/// Resource exhaustion recovery strategy
#[derive(Debug)]
pub struct ResourceRecoveryStrategy;

impl RecoveryStrategy for ResourceRecoveryStrategy {
    fn recover<'a>(
        &'a self,
        _error: &'a anyhow::Error,
        context: &'a RecoveryContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TestCharacteristics>> + Send + 'a>>
    {
        Box::pin(async move {
            info!(
                "Attempting resource recovery for component: {}",
                context.component
            );

            // Wait a bit for resources to become available
            tokio::time::sleep(Duration::from_millis(1000)).await;

            // Return a simplified analysis result
            let mut characteristics = TestCharacteristics::default();
            characteristics.analysis_metadata.confidence_score = 0.6;
            characteristics
                .analysis_metadata
                .notes
                .push("Recovered from resource exhaustion".to_string());

            Ok(characteristics)
        })
    }
}

/// Network error recovery strategy
#[derive(Debug)]
pub struct NetworkRecoveryStrategy;

impl RecoveryStrategy for NetworkRecoveryStrategy {
    fn recover<'a>(
        &'a self,
        _error: &'a anyhow::Error,
        context: &'a RecoveryContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TestCharacteristics>> + Send + 'a>>
    {
        Box::pin(async move {
            info!(
                "Attempting network recovery for component: {}",
                context.component
            );

            // Implement exponential backoff
            let delay = Duration::from_millis(1000 * (2_u64.pow(context.retry_attempt as u32)));
            tokio::time::sleep(delay).await;

            // Return a default result
            let mut characteristics = TestCharacteristics::default();
            characteristics.analysis_metadata.confidence_score = 0.4;
            characteristics
                .analysis_metadata
                .notes
                .push("Recovered from network error".to_string());

            Ok(characteristics)
        })
    }
}

/// Component failure recovery strategy
#[derive(Debug)]
pub struct ComponentRecoveryStrategy;

impl RecoveryStrategy for ComponentRecoveryStrategy {
    fn recover<'a>(
        &'a self,
        _error: &'a anyhow::Error,
        context: &'a RecoveryContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TestCharacteristics>> + Send + 'a>>
    {
        Box::pin(async move {
            info!(
                "Attempting component recovery for component: {}",
                context.component
            );

            // Try to reinitialize the component or use a fallback
            let mut characteristics = TestCharacteristics::default();
            characteristics.analysis_metadata.confidence_score = 0.3;
            characteristics
                .analysis_metadata
                .notes
                .push("Recovered from component failure".to_string());

            Ok(characteristics)
        })
    }
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager
    pub async fn new(config: ErrorRecoveryConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(TokioRwLock::new(config)),
            circuit_breakers: Arc::new(TokioMutex::new(HashMap::new())),
            error_stats: Arc::new(ErrorStatistics::default()),
            recovery_strategies: Arc::new(RecoveryStrategies::default()),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Recover from a phase error
    #[instrument(skip(self, error, test_data, options))]
    pub async fn recover_from_phase_error(
        &self,
        phase: &AnalysisPhase,
        error: &anyhow::Error,
        test_data: &TestExecutionData,
        options: &ProfilingOptions,
    ) -> Result<PhaseResult> {
        self.error_stats.total_errors.fetch_add(1, Ordering::Relaxed);
        self.error_stats.total_recoveries.fetch_add(1, Ordering::Relaxed);

        let component_name = format!("{:?}", phase);

        // Check circuit breaker
        if self.is_circuit_breaker_open(&component_name).await {
            self.error_stats.failed_recoveries.fetch_add(1, Ordering::Relaxed);
            return Err(anyhow!(
                "Circuit breaker is open for component: {}",
                component_name
            ));
        }

        // Determine error type
        let error_type = self.classify_error(error);

        // Create recovery context
        let context = RecoveryContext {
            component: component_name.clone(),
            error_details: error.to_string(),
            retry_attempt: 1, // This could be tracked per request
            request_context: RequestContext {
                test_data: test_data.clone(),
                options: options.clone(),
                phase: Some(phase.clone()),
            },
        };

        // Attempt recovery
        let recovery_result = self.attempt_recovery(&error_type, error, &context).await;

        match recovery_result {
            Ok(characteristics) => {
                self.error_stats.successful_recoveries.fetch_add(1, Ordering::Relaxed);
                self.reset_circuit_breaker(&component_name).await;

                // Convert to appropriate phase result
                let phase_result = match phase {
                    AnalysisPhase::ResourceAnalysis => {
                        PhaseResult::ResourceAnalysis(characteristics.resource_intensity)
                    },
                    AnalysisPhase::ConcurrencyDetection => PhaseResult::ConcurrencyDetection(
                        Box::new(characteristics.concurrency_requirements),
                    ),
                    AnalysisPhase::SynchronizationAnalysis => PhaseResult::SynchronizationAnalysis(
                        characteristics.synchronization_dependencies,
                    ),
                    AnalysisPhase::PatternRecognition => {
                        PhaseResult::PatternRecognition(characteristics.performance_patterns)
                    },
                    AnalysisPhase::ProfilingPipeline => {
                        PhaseResult::ProfilingPipeline(super::orchestrator::TestProfile {
                            resource_metrics: HashMap::new(),
                        })
                    },
                    AnalysisPhase::RealTimeProfiler => {
                        PhaseResult::RealTimeProfiler(Box::new(characteristics))
                    },
                };

                Ok(phase_result)
            },
            Err(recovery_error) => {
                self.error_stats.failed_recoveries.fetch_add(1, Ordering::Relaxed);
                self.record_circuit_breaker_failure(&component_name).await;
                Err(recovery_error)
            },
        }
    }

    /// Recover from a task error
    pub async fn recover_from_task_error(
        &self,
        task: &AnalysisTask,
        error: &anyhow::Error,
    ) -> Result<TestCharacteristics> {
        self.error_stats.total_errors.fetch_add(1, Ordering::Relaxed);
        self.error_stats.total_recoveries.fetch_add(1, Ordering::Relaxed);

        let component_name = format!("task_{}", task.id);

        // Check circuit breaker
        if self.is_circuit_breaker_open(&component_name).await {
            self.error_stats.failed_recoveries.fetch_add(1, Ordering::Relaxed);
            return Err(anyhow!("Circuit breaker is open for task: {}", task.id));
        }

        // Determine error type
        let error_type = self.classify_error(error);

        // Create recovery context
        let context = RecoveryContext {
            component: component_name.clone(),
            error_details: error.to_string(),
            retry_attempt: 1,
            request_context: RequestContext {
                test_data: task.test_data.clone(),
                options: task.options.clone(),
                phase: None,
            },
        };

        // Attempt recovery
        let recovery_result = self.attempt_recovery(&error_type, error, &context).await;

        match recovery_result {
            Ok(characteristics) => {
                self.error_stats.successful_recoveries.fetch_add(1, Ordering::Relaxed);
                self.reset_circuit_breaker(&component_name).await;
                Ok(characteristics)
            },
            Err(recovery_error) => {
                self.error_stats.failed_recoveries.fetch_add(1, Ordering::Relaxed);
                self.record_circuit_breaker_failure(&component_name).await;
                Err(recovery_error)
            },
        }
    }

    /// Classify error type
    fn classify_error(&self, error: &anyhow::Error) -> ErrorType {
        let error_msg = error.to_string().to_lowercase();

        if error_msg.contains("timeout") || error_msg.contains("deadline") {
            ErrorType::Timeout
        } else if error_msg.contains("resource")
            || error_msg.contains("memory")
            || error_msg.contains("capacity")
        {
            ErrorType::ResourceExhaustion
        } else if error_msg.contains("network")
            || error_msg.contains("connection")
            || error_msg.contains("io")
        {
            ErrorType::NetworkError
        } else if error_msg.contains("component") || error_msg.contains("service") {
            ErrorType::ComponentFailure
        } else if error_msg.contains("validation") || error_msg.contains("invalid") {
            ErrorType::ValidationError
        } else {
            ErrorType::UnknownError
        }
    }

    /// Attempt recovery using appropriate strategy
    async fn attempt_recovery(
        &self,
        error_type: &ErrorType,
        error: &anyhow::Error,
        context: &RecoveryContext,
    ) -> Result<TestCharacteristics> {
        if let Some(strategy) = self.recovery_strategies.strategies.get(error_type) {
            strategy.recover(error, context).await
        } else {
            // Default recovery - return minimal characteristics
            let mut characteristics = TestCharacteristics::default();
            characteristics.analysis_metadata.confidence_score = 0.1;
            characteristics
                .analysis_metadata
                .notes
                .push(format!("Default recovery for {:?}", error_type));
            Ok(characteristics)
        }
    }

    /// Check if circuit breaker is open
    async fn is_circuit_breaker_open(&self, component: &str) -> bool {
        let circuit_breakers = self.circuit_breakers.lock().await;
        if let Some(state) = circuit_breakers.get(component) {
            matches!(state.state, CircuitState::Open)
        } else {
            false
        }
    }

    /// Record circuit breaker failure
    async fn record_circuit_breaker_failure(&self, component: &str) {
        let mut circuit_breakers = self.circuit_breakers.lock().await;
        let config = self.config.read().await;

        let state = circuit_breakers
            .entry(component.to_string())
            .or_insert_with(CircuitBreakerState::default);

        state.failure_count += 1;
        state.last_failure = Some(SystemTime::now());

        if state.failure_count >= config.circuit_breaker_threshold {
            state.state = CircuitState::Open;
            state.next_retry = Some(
                SystemTime::now()
                    + Duration::from_secs(config.circuit_breaker_reset_timeout_seconds),
            );
            self.error_stats.circuit_breaker_trips.fetch_add(1, Ordering::Relaxed);
            warn!("Circuit breaker opened for component: {}", component);
        }
    }

    /// Reset circuit breaker
    async fn reset_circuit_breaker(&self, component: &str) {
        let mut circuit_breakers = self.circuit_breakers.lock().await;
        if let Some(state) = circuit_breakers.get_mut(component) {
            state.state = CircuitState::Closed;
            state.failure_count = 0;
            state.last_failure = None;
            state.next_retry = None;
        }
    }

    /// Start monitoring task
    pub async fn start_monitoring(&self) -> Result<()> {
        // For now, this is a placeholder
        // In a full implementation, this would start a background task
        // to monitor circuit breaker states and potentially reset them
        Ok(())
    }

    /// Get error statistics
    pub async fn get_statistics(&self) -> ErrorStatistics {
        ErrorStatistics {
            total_errors: AtomicU64::new(self.error_stats.total_errors.load(Ordering::Relaxed)),
            total_recoveries: AtomicU64::new(
                self.error_stats.total_recoveries.load(Ordering::Relaxed),
            ),
            successful_recoveries: AtomicU64::new(
                self.error_stats.successful_recoveries.load(Ordering::Relaxed),
            ),
            failed_recoveries: AtomicU64::new(
                self.error_stats.failed_recoveries.load(Ordering::Relaxed),
            ),
            circuit_breaker_trips: AtomicU64::new(
                self.error_stats.circuit_breaker_trips.load(Ordering::Relaxed),
            ),
        }
    }

    /// Shutdown error recovery manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down ErrorRecoveryManager");

        self.shutdown.store(true, Ordering::Release);

        // Clear circuit breaker states
        {
            let mut circuit_breakers = self.circuit_breakers.lock().await;
            circuit_breakers.clear();
        }

        info!("ErrorRecoveryManager shutdown completed");
        Ok(())
    }
}
