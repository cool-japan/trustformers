//! Comprehensive types and data structures for streaming conversational AI responses.
//!
//! This module contains all types used in the streaming response system, including
//! configuration structs, response types, state management, metrics, error handling,
//! and quality analysis types.

use crate::core::traits::{Model, Tokenizer};
use crate::pipeline::conversational::types::*;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use trustformers_models::common_patterns::GenerativeModel;

// ================================================================================================
// CONFIGURATION TYPES
// ================================================================================================

/// Advanced streaming configuration with enhanced features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedStreamingConfig {
    /// Basic streaming configuration
    pub base_config: StreamingConfig,
    /// Enable adaptive chunking based on content
    pub adaptive_chunking: bool,
    /// Maximum chunk size in characters
    pub max_chunk_size: usize,
    /// Minimum chunk size in characters
    pub min_chunk_size: usize,
    /// Enable natural pausing at sentence boundaries
    pub natural_pausing: bool,
    /// Pause duration at punctuation marks (ms)
    pub punctuation_pause_ms: u64,
    /// Enable typing speed variation
    pub variable_typing_speed: bool,
    /// Base typing speed (characters per second)
    pub base_typing_speed: f32,
    /// Speed variation factor (0.0 to 1.0)
    pub speed_variation: f32,
    /// Enable backpressure handling
    pub enable_backpressure: bool,
    /// Maximum buffer size for backpressure
    pub max_buffer_size: usize,
    /// Timeout for chunk delivery (ms)
    pub chunk_timeout_ms: u64,
    /// Enable quality-based streaming
    pub quality_based_streaming: bool,
    /// Enable error recovery
    pub enable_error_recovery: bool,
    /// Maximum retry attempts
    pub max_retry_attempts: usize,
}

impl Default for AdvancedStreamingConfig {
    fn default() -> Self {
        Self {
            base_config: StreamingConfig::default(),
            adaptive_chunking: true,
            max_chunk_size: 50,
            min_chunk_size: 5,
            natural_pausing: true,
            punctuation_pause_ms: 150,
            variable_typing_speed: true,
            base_typing_speed: 25.0, // characters per second
            speed_variation: 0.3,
            enable_backpressure: true,
            max_buffer_size: 1000,
            chunk_timeout_ms: 5000,
            quality_based_streaming: true,
            enable_error_recovery: true,
            max_retry_attempts: 3,
        }
    }
}

// ================================================================================================
// RESPONSE AND STREAMING TYPES
// ================================================================================================

/// Extended streaming response with additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedStreamingResponse {
    /// Base streaming response
    pub base_response: StreamingResponse,
    /// Streaming state
    pub state: StreamingState,
    /// Timestamp when chunk was generated
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Estimated completion time
    pub estimated_completion: Option<chrono::DateTime<chrono::Utc>>,
    /// Current streaming metrics
    pub metrics: StreamingMetrics,
    /// Quality indicators
    pub quality: StreamingQuality,
}

/// Individual stream chunk with metadata
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Chunk content
    pub content: String,
    /// Chunk index in sequence
    pub index: usize,
    /// Type of chunk
    pub chunk_type: ChunkType,
    /// Timing information
    pub timing: ChunkTiming,
    /// Chunk metadata
    pub metadata: ChunkMetadata,
}

/// Type of chunk for different processing
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    Content,
    Sentence,
    Adaptive,
    Semantic,
    Punctuation,
    Special,
}

/// Timing information for chunks
#[derive(Debug, Clone)]
pub struct ChunkTiming {
    /// Delay before sending this chunk (ms)
    pub delay_ms: u64,
    /// Pause after sending this chunk (ms)
    pub pause_ms: u64,
    /// Variable timing factor
    pub timing_factor: f32,
}

impl Default for ChunkTiming {
    fn default() -> Self {
        Self {
            delay_ms: 50,
            pause_ms: 0,
            timing_factor: 1.0,
        }
    }
}

impl ChunkTiming {
    /// Create timing with pause
    pub fn with_pause(pause_ms: u64) -> Self {
        Self {
            delay_ms: 50,
            pause_ms,
            timing_factor: 1.0,
        }
    }

    /// Create adaptive timing based on complexity
    pub fn adaptive(complexity: f32) -> Self {
        let base_delay = 50;
        let delay_ms = (base_delay as f32 * (0.5 + complexity * 0.5)) as u64;

        Self {
            delay_ms,
            pause_ms: if complexity > 0.7 { 100 } else { 0 },
            timing_factor: 0.5 + complexity * 0.5,
        }
    }
}

/// Metadata for chunks
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    /// Complexity score
    pub complexity: f32,
    /// Importance score
    pub importance: f32,
    /// Quality indicators
    pub quality_indicators: Vec<String>,
    /// Processing hints
    pub processing_hints: Vec<String>,
}

impl Default for ChunkMetadata {
    fn default() -> Self {
        Self {
            complexity: 0.5,
            importance: 0.5,
            quality_indicators: Vec::new(),
            processing_hints: Vec::new(),
        }
    }
}

impl ChunkMetadata {
    /// Create metadata with complexity
    pub fn with_complexity(complexity: f32) -> Self {
        Self {
            complexity,
            importance: complexity,
            quality_indicators: vec!["adaptive".to_string()],
            processing_hints: Vec::new(),
        }
    }

    /// Create semantic metadata
    pub fn semantic() -> Self {
        Self {
            complexity: 0.6,
            importance: 0.7,
            quality_indicators: vec!["semantic".to_string()],
            processing_hints: vec!["maintain_context".to_string()],
        }
    }
}

// ================================================================================================
// STATE MANAGEMENT TYPES
// ================================================================================================

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

/// Buffer state for backpressure management
#[derive(Debug, Clone)]
pub struct BufferState {
    /// Current buffer size
    pub current_size: usize,
    /// Maximum buffer size
    pub max_size: usize,
    /// Buffer utilization percentage
    pub utilization: f32,
    /// Pending chunks
    pub pending_chunks: usize,
}

/// Stream connection status
#[derive(Debug, Clone, PartialEq)]
pub enum StreamConnection {
    Connecting,
    Connected,
    Streaming,
    Paused,
    Buffering,
    Reconnecting,
    Disconnected,
    Error(String),
}

/// Chunk processing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Fixed size chunks
    FixedSize(usize),
    /// Word boundary chunking
    WordBoundary,
    /// Sentence boundary chunking
    SentenceBoundary,
    /// Adaptive chunking based on content
    Adaptive,
    /// Semantic chunking
    Semantic,
}

/// Flow control state
#[derive(Debug, Clone)]
pub struct FlowState {
    /// Current flow rate (chunks per second)
    pub flow_rate: f32,
    /// Target flow rate
    pub target_rate: f32,
    /// Buffer fill level (0.0 to 1.0)
    pub buffer_fill: f32,
    /// Flow control actions taken
    pub actions_taken: Vec<FlowAction>,
    /// Last adjustment time
    pub last_adjustment: Instant,
}

impl Default for FlowState {
    fn default() -> Self {
        Self {
            flow_rate: 10.0, // Default 10 chunks per second
            target_rate: 10.0,
            buffer_fill: 0.0,
            actions_taken: Vec::new(),
            last_adjustment: Instant::now(),
        }
    }
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

// ================================================================================================
// METRICS AND PERFORMANCE TYPES
// ================================================================================================

/// Real-time streaming metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingMetrics {
    /// Current streaming speed (chunks per second)
    pub chunks_per_second: f32,
    /// Average chunk size
    pub avg_chunk_size: f32,
    /// Total chunks sent
    pub total_chunks: usize,
    /// Bytes streamed
    pub bytes_streamed: usize,
    /// Stream duration (ms)
    pub duration_ms: u64,
    /// Current buffer utilization (0.0 to 1.0)
    pub buffer_utilization: f32,
    /// Error count
    pub error_count: usize,
    /// Retry count
    pub retry_count: usize,
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self {
            chunks_per_second: 0.0,
            avg_chunk_size: 0.0,
            total_chunks: 0,
            bytes_streamed: 0,
            duration_ms: 0,
            buffer_utilization: 0.0,
            error_count: 0,
            retry_count: 0,
        }
    }
}

/// Global streaming metrics across all sessions
#[derive(Debug, Clone)]
pub struct GlobalStreamingMetrics {
    /// Total active streams
    pub active_streams: usize,
    /// Total streams created
    pub total_streams_created: usize,
    /// Average stream duration
    pub avg_stream_duration_ms: f64,
    /// Total chunks streamed
    pub total_chunks_streamed: usize,
    /// Total bytes streamed
    pub total_bytes_streamed: usize,
    /// Global error rate
    pub global_error_rate: f32,
    /// System performance metrics
    pub system_performance: SystemPerformanceMetrics,
}

impl Default for GlobalStreamingMetrics {
    fn default() -> Self {
        Self {
            active_streams: 0,
            total_streams_created: 0,
            avg_stream_duration_ms: 0.0,
            total_chunks_streamed: 0,
            total_bytes_streamed: 0,
            global_error_rate: 0.0,
            system_performance: SystemPerformanceMetrics::default(),
        }
    }
}

/// System performance metrics for streaming
#[derive(Debug, Clone)]
pub struct SystemPerformanceMetrics {
    /// CPU usage during streaming
    pub cpu_usage: f32,
    /// Memory usage
    pub memory_usage_mb: f64,
    /// Network utilization
    pub network_utilization: f32,
    /// Average latency
    pub avg_latency_ms: f64,
    /// Throughput (chunks per second)
    pub throughput: f32,
}

impl Default for SystemPerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage_mb: 0.0,
            network_utilization: 0.0,
            avg_latency_ms: 0.0,
            throughput: 0.0,
        }
    }
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

/// Backpressure metrics
#[derive(Debug, Clone, Default)]
pub struct BackpressureMetrics {
    /// Total pressure events
    pub pressure_events: usize,
    /// Time under pressure (ms)
    pub time_under_pressure_ms: u64,
    /// Flow adjustments made
    pub flow_adjustments: usize,
    /// Buffer overflows prevented
    pub overflows_prevented: usize,
    /// Quality adjustments made
    pub quality_adjustments: usize,
}

/// Statistics about streaming performance
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    pub total_chunks: usize,
    pub total_characters: usize,
    pub total_words: usize,
    pub avg_chunk_size: f32,
    pub estimated_duration_seconds: f32,
}

// ================================================================================================
// QUALITY ANALYSIS TYPES
// ================================================================================================

/// Quality indicators for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingQuality {
    /// Smoothness score (0.0 to 1.0)
    pub smoothness: f32,
    /// Naturalness score (0.0 to 1.0)
    pub naturalness: f32,
    /// Responsiveness score (0.0 to 1.0)
    pub responsiveness: f32,
    /// Coherence score (0.0 to 1.0)
    pub coherence: f32,
    /// Overall quality score (0.0 to 1.0)
    pub overall_quality: f32,
}

impl Default for StreamingQuality {
    fn default() -> Self {
        Self {
            smoothness: 1.0,
            naturalness: 1.0,
            responsiveness: 1.0,
            coherence: 1.0,
            overall_quality: 1.0,
        }
    }
}

/// Individual quality measurement
#[derive(Debug, Clone)]
pub struct QualityMeasurement {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Smoothness score (0.0 to 1.0)
    pub smoothness: f32,
    /// Naturalness score (0.0 to 1.0)
    pub naturalness: f32,
    /// Responsiveness score (0.0 to 1.0)
    pub responsiveness: f32,
    /// Coherence score (0.0 to 1.0)
    pub coherence: f32,
    /// Latency (ms)
    pub latency_ms: f64,
    /// Chunk size consistency
    pub chunk_consistency: f32,
}

/// Quality thresholds for different aspects
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Minimum smoothness threshold
    pub min_smoothness: f32,
    /// Minimum naturalness threshold
    pub min_naturalness: f32,
    /// Minimum responsiveness threshold
    pub min_responsiveness: f32,
    /// Minimum coherence threshold
    pub min_coherence: f32,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: f64,
    /// Minimum overall quality threshold
    pub min_overall_quality: f32,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_smoothness: 0.7,
            min_naturalness: 0.6,
            min_responsiveness: 0.8,
            min_coherence: 0.7,
            max_latency_ms: 200.0,
            min_overall_quality: 0.7,
        }
    }
}

/// Quality trends analysis
#[derive(Debug, Clone)]
pub struct QualityTrends {
    /// Overall quality trend
    pub overall_trend: TrendDirection,
    /// Smoothness trend
    pub smoothness_trend: TrendDirection,
    /// Naturalness trend
    pub naturalness_trend: TrendDirection,
    /// Responsiveness trend
    pub responsiveness_trend: TrendDirection,
    /// Coherence trend
    pub coherence_trend: TrendDirection,
}

impl Default for QualityTrends {
    fn default() -> Self {
        Self {
            overall_trend: TrendDirection::Stable,
            smoothness_trend: TrendDirection::Stable,
            naturalness_trend: TrendDirection::Stable,
            responsiveness_trend: TrendDirection::Stable,
            coherence_trend: TrendDirection::Stable,
        }
    }
}

// ================================================================================================
// SESSION MANAGEMENT TYPES
// ================================================================================================

/// Individual stream session
#[derive(Debug, Clone)]
pub struct StreamSession {
    /// Session ID
    pub session_id: String,
    /// Conversation ID
    pub conversation_id: String,
    /// Current state
    pub state: StreamConnection,
    /// Session metrics
    pub metrics: StreamingMetrics,
    /// Start time
    pub start_time: Instant,
    /// Last activity
    pub last_activity: Instant,
    /// Buffer state
    pub buffer_state: BufferState,
}

/// Streaming session information
#[derive(Debug, Clone)]
pub struct StreamingSession {
    pub session_id: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    pub config: StreamingConfig,
    pub state: StreamingState,
    pub stats: Option<StreamingStats>,
}

impl StreamingSession {
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            start_time: chrono::Utc::now(),
            end_time: None,
            config,
            state: StreamingState::NotStarted,
            stats: None,
        }
    }

    pub fn complete(&mut self, stats: StreamingStats) {
        self.end_time = Some(chrono::Utc::now());
        self.state = StreamingState::Completed;
        self.stats = Some(stats);
    }

    pub fn duration_ms(&self) -> Option<i64> {
        self.end_time.map(|end| (end - self.start_time).num_milliseconds())
    }
}

// ================================================================================================
// TYPING SIMULATION TYPES
// ================================================================================================

/// Typing event for natural simulation
#[derive(Debug, Clone)]
pub struct TypingEvent {
    /// Type of typing event
    pub event_type: TypingEventType,
    /// Character index in the full content
    pub char_index: usize,
    /// Content for this event
    pub content: String,
    /// Delay before this event
    pub delay: Duration,
}

/// Types of typing events
#[derive(Debug, Clone, PartialEq)]
pub enum TypingEventType {
    StartTyping,
    Pause,
    Correction,
    Hesitation,
}

/// Typing patterns analyzer for natural simulation
#[derive(Debug)]
pub struct TypingPatterns {
    /// Common typing patterns
    patterns: Vec<TypingPattern>,
}

impl Default for TypingPatterns {
    fn default() -> Self {
        Self::new()
    }
}

impl TypingPatterns {
    /// Create new typing patterns analyzer
    pub fn new() -> Self {
        Self {
            patterns: vec![
                TypingPattern::new(
                    "technical",
                    vec!["algorithm", "implementation", "function"],
                    0.8,
                ),
                TypingPattern::new("emotional", vec!["feel", "think", "believe"], 0.6),
                TypingPattern::new("question", vec!["what", "how", "why", "when", "where"], 0.7),
                TypingPattern::new("explanation", vec!["because", "therefore", "however"], 0.75),
            ],
        }
    }

    /// Analyze content for typing patterns
    pub fn analyze_content(&self, content: &str) -> TypingAnalysis {
        let content_lower = content.to_lowercase();
        let mut pattern_scores = std::collections::HashMap::new();

        for pattern in &self.patterns {
            let score = pattern.calculate_score(&content_lower);
            pattern_scores.insert(pattern.name.clone(), score);
        }

        TypingAnalysis {
            dominant_pattern: pattern_scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k.clone()),
            pattern_scores,
            complexity_score: self.calculate_complexity(&content_lower),
            naturalness_indicators: self.extract_naturalness_indicators(&content_lower),
        }
    }

    /// Calculate content complexity
    fn calculate_complexity(&self, content: &str) -> f32 {
        let word_count = content.split_whitespace().count();
        let avg_word_length = content.split_whitespace().map(|w| w.len()).sum::<usize>() as f32
            / word_count.max(1) as f32;

        let sentence_count = content.matches('.').count()
            + content.matches('!').count()
            + content.matches('?').count();

        let avg_sentence_length = word_count as f32 / sentence_count.max(1) as f32;

        // Normalize complexity factors
        let word_complexity = (avg_word_length / 8.0).min(1.0);
        let sentence_complexity = (avg_sentence_length / 20.0).min(1.0);

        (word_complexity + sentence_complexity) / 2.0
    }

    /// Extract naturalness indicators
    fn extract_naturalness_indicators(&self, content: &str) -> Vec<String> {
        let mut indicators = Vec::new();

        if content.contains("um") || content.contains("uh") || content.contains("er") {
            indicators.push("hesitation_markers".to_string());
        }

        if content.matches("...").count() > 0 {
            indicators.push("ellipsis_pauses".to_string());
        }

        if content.matches("!").count() > content.matches(".").count() {
            indicators.push("exclamatory".to_string());
        }

        if content.contains("?") {
            indicators.push("questioning".to_string());
        }

        indicators
    }
}

/// Individual typing pattern
#[derive(Debug, Clone)]
pub struct TypingPattern {
    /// Pattern name
    pub name: String,
    /// Keywords associated with this pattern
    pub keywords: Vec<String>,
    /// Base complexity score
    pub complexity: f32,
}

impl TypingPattern {
    /// Create a new typing pattern
    pub fn new(name: &str, keywords: Vec<&str>, complexity: f32) -> Self {
        Self {
            name: name.to_string(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
            complexity,
        }
    }

    /// Calculate pattern match score for content
    pub fn calculate_score(&self, content: &str) -> f32 {
        let word_count = content.split_whitespace().count();
        if word_count == 0 {
            return 0.0;
        }

        let matches = self
            .keywords
            .iter()
            .map(|keyword| content.matches(keyword).count())
            .sum::<usize>();

        (matches as f32 / word_count as f32) * self.complexity
    }
}

/// Analysis results for typing patterns
#[derive(Debug, Clone)]
pub struct TypingAnalysis {
    /// Dominant typing pattern detected
    pub dominant_pattern: Option<String>,
    /// Scores for all patterns
    pub pattern_scores: std::collections::HashMap<String, f32>,
    /// Overall complexity score
    pub complexity_score: f32,
    /// Naturalness indicators
    pub naturalness_indicators: Vec<String>,
}

// ================================================================================================
// ERROR HANDLING TYPES
// ================================================================================================

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
    pub context: std::collections::HashMap<String, String>,
}

/// Types of streaming errors
#[derive(Debug, Clone, PartialEq)]
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
// FLOW CONTROL TYPES
// ================================================================================================

/// Pressure levels for backpressure management
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum PressureLevel {
    None = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Flow control actions
#[derive(Debug, Clone)]
pub enum FlowAction {
    IncreaseRate(f32),
    DecreaseRate(f32),
    PauseFlow,
    ResumeFlow,
    BufferDrain,
    QualityAdjustment(f32),
}

// ================================================================================================
// MANAGER AND COORDINATOR TYPES (STRUCT DEFINITIONS ONLY)
// ================================================================================================

/// Central coordinator for streaming responses
#[derive(Debug)]
pub struct StreamingCoordinator {
    /// Configuration for streaming
    config: AdvancedStreamingConfig,
    /// Active streams registry
    active_streams: Arc<RwLock<std::collections::HashMap<String, StreamSession>>>,
    /// Global metrics
    global_metrics: Arc<RwLock<GlobalStreamingMetrics>>,
    /// Quality analyzer
    quality_analyzer: QualityAnalyzer,
    /// Error recovery manager
    error_recovery: ErrorRecoveryManager,
}

/// Advanced response chunker with multiple strategies
#[derive(Debug)]
pub struct ResponseChunker {
    /// Chunking strategy
    strategy: ChunkingStrategy,
    /// Configuration
    config: AdvancedStreamingConfig,
    /// Quality analyzer
    quality_analyzer: QualityAnalyzer,
}

/// Natural typing simulator for human-like response delivery
#[derive(Debug)]
pub struct TypingSimulator {
    /// Configuration
    config: AdvancedStreamingConfig,
    /// Random number generator state
    rng_state: std::sync::Mutex<fastrand::Rng>,
    /// Typing patterns analyzer
    patterns: TypingPatterns,
}

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

/// Error recovery manager for handling streaming failures
#[derive(Debug)]
pub struct ErrorRecoveryManager {
    /// Recovery strategies mapping
    strategies: std::collections::HashMap<StreamErrorType, Vec<RecoveryStrategy>>,
    /// Recovery attempt tracking
    recovery_attempts: Arc<RwLock<std::collections::HashMap<StreamErrorType, usize>>>,
    /// Maximum recovery attempts
    max_attempts: usize,
}

impl Default for ErrorRecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorRecoveryManager {
    pub fn new() -> Self {
        Self {
            strategies: std::collections::HashMap::new(),
            recovery_attempts: Arc::new(RwLock::new(std::collections::HashMap::new())),
            max_attempts: 3,
        }
    }
}

/// Backpressure controller for managing streaming flow
#[derive(Debug)]
pub struct BackpressureController {
    /// Configuration
    config: AdvancedStreamingConfig,
    /// Current pressure level
    pressure_level: Arc<RwLock<PressureLevel>>,
    /// Flow control state
    flow_state: Arc<RwLock<FlowState>>,
    /// Metrics collector
    metrics: Arc<RwLock<BackpressureMetrics>>,
}

/// Quality analyzer for streaming performance
#[derive(Debug)]
pub struct QualityAnalyzer {
    /// Quality metrics window
    metrics_window: Arc<RwLock<VecDeque<QualityMeasurement>>>,
    /// Window size
    window_size: usize,
    /// Quality thresholds
    pub thresholds: QualityThresholds,
}

impl Default for QualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl QualityAnalyzer {
    /// Create a new quality analyzer
    pub fn new() -> Self {
        Self {
            metrics_window: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            window_size: 100,
            thresholds: QualityThresholds::default(),
        }
    }

    /// Get metrics window for external access
    pub fn metrics_window(&self) -> &Arc<RwLock<VecDeque<QualityMeasurement>>> {
        &self.metrics_window
    }

    /// Get window size
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get quality thresholds
    pub fn thresholds(&self) -> &QualityThresholds {
        &self.thresholds
    }
}

/// Main streaming pipeline for conversational responses
pub struct ConversationalStreamingPipeline<M, T>
where
    M: Model + Send + Sync + GenerativeModel,
    T: Tokenizer + Send + Sync,
{
    /// Model reference
    model: Arc<M>,
    /// Tokenizer reference
    tokenizer: Arc<T>,
    /// Streaming coordinator
    coordinator: StreamingCoordinator,
    /// Response chunker
    chunker: ResponseChunker,
    /// Typing simulator
    typing_simulator: TypingSimulator,
    /// State manager
    state_manager: StreamStateManager,
    /// Backpressure controller
    backpressure_controller: BackpressureController,
}

/// Streaming manager
#[derive(Debug)]
pub struct StreamingManager {
    /// Configuration
    config: StreamingConfig,
    /// Active sessions
    active_sessions: HashMap<String, StreamingSession>,
    /// Typing simulator
    typing_simulator: Option<TypingSimulator>,
    /// Quality analyzer
    quality_analyzer: QualityAnalyzer,
}
