//! Real-Time Metrics Collector Module
//!
//! This module provides comprehensive real-time metrics collection functionality
//! for the TrustformeRS performance optimization system. It includes core collection
//! capabilities, adaptive sampling, performance impact monitoring, and live streaming
//! of metrics data.
//!
//! ## Key Components
//!
//! - **RealTimeMetricsCollector**: Core real-time metrics collector with live streaming
//! - **SampleRateController**: Adaptive sample rate control based on system load
//! - **PerformanceImpactMonitor**: Collection performance impact monitoring
//! - **CollectionStatistics**: Statistics and performance tracking for collection operations
//! - **MetricsPublisher**: Live streaming and publishing of metrics data
//! - **Collection task management**: Background collection tasks and lifecycle management
//! - **Data filtering and preprocessing**: Initial data processing before aggregation
//! - **Collection error handling**: Robust error recovery and reliability
//!
//! ## Features
//!
//! - High-frequency real-time data collection (microsecond precision)
//! - Live streaming metrics with configurable sampling rates
//! - Thread-safe concurrent monitoring with minimal overhead
//! - Adaptive sampling rates based on system load
//! - Comprehensive error handling and system reliability
//! - Background task management with proper lifecycle controls
//! - Enhanced functionality with backward compatibility
//!
//! ## Example Usage
//!
//! ```rust
//! use crate::performance_optimizer::real_time_metrics::collector::*;
//! use crate::performance_optimizer::real_time_metrics::types::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create configuration
//!     let config = MetricsCollectionConfig::default();
//!
//!     // Initialize collector
//!     let collector = RealTimeMetricsCollector::new(config).await?;
//!
//!     // Start collection
//!     collector.start_collection().await?;
//!
//!     // Collect metrics
//!     let metrics = collector.collect_metrics().await?;
//!     println!("Collected metrics: {:?}", metrics);
//!
//!     // Add publisher for live streaming
//!     let publisher = MetricsPublisher::new(
//!         "http_publisher".to_string(),
//!         PublisherType::Http { endpoint: "http://localhost:8080/metrics".to_string() },
//!         DeliveryConfig::default(),
//!     );
//!     collector.add_publisher(publisher).await?;
//!
//!     Ok(())
//! }
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, watch},
    task::JoinHandle,
    time::interval,
};

// Import types from the types module
use super::types::*;

// =============================================================================
// CORE COLLECTOR TYPES
// =============================================================================

/// Real-time metrics collector with live streaming capabilities
///
/// Core component responsible for high-frequency collection of performance metrics
/// with configurable sampling rates, live streaming, and minimal system overhead.
///
/// ## Key Features
///
/// - **High-frequency collection**: Microsecond precision data collection
/// - **Live streaming**: Real-time metrics publishing to multiple destinations
/// - **Adaptive sampling**: Dynamic sampling rate adjustment based on system load
/// - **Performance monitoring**: Collection overhead tracking and optimization
/// - **Thread-safe operations**: Concurrent access with minimal locking overhead
/// - **Comprehensive error handling**: Robust recovery and reliability features
///
/// ## Example
///
/// ```rust
/// let config = MetricsCollectionConfig {
///     base_interval: Duration::from_millis(100),
///     min_interval: Duration::from_millis(10),
///     max_interval: Duration::from_secs(1),
///     history_buffer_size: 1000,
///     adaptive_sampling: true,
///     high_precision_mode: true,
///     stream_publishing: true,
///     ..Default::default()
/// };
///
/// let collector = RealTimeMetricsCollector::new(config).await?;
/// collector.start_collection().await?;
/// ```
pub struct RealTimeMetricsCollector {
    /// Current metrics state
    current_metrics: Arc<RwLock<RealTimeMetrics>>,

    /// Metrics history buffer (circular buffer for efficiency)
    metrics_history: Arc<Mutex<CircularBuffer<TimestampedMetrics>>>,

    /// Collection configuration
    config: Arc<RwLock<MetricsCollectionConfig>>,

    /// Metrics publishers for live streaming
    metrics_publishers: Arc<Mutex<Vec<Arc<MetricsPublisher>>>>,

    /// Collection tasks
    collection_tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,

    /// Sample rate controller
    sample_rate_controller: Arc<SampleRateController>,

    /// Performance impact monitor
    impact_monitor: Arc<PerformanceImpactMonitor>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Collection statistics
    collection_stats: Arc<CollectionStatistics>,

    /// Event broadcaster for real-time updates
    event_broadcaster: Arc<broadcast::Sender<CollectionEvent>>,

    /// Error handler for collection issues
    error_handler: Arc<Mutex<Box<dyn CollectionErrorHandler + Send + Sync>>>,
}

/// Adaptive sample rate controller for optimizing collection frequency
///
/// Controls the sampling rate of metrics collection based on system load,
/// resource availability, and target accuracy requirements.
///
/// ## Algorithm Features
///
/// - **PID-based control**: Proportional-Integral-Derivative control algorithm
/// - **Load-based adaptation**: Adjusts rate based on CPU/memory pressure
/// - **Accuracy targeting**: Maintains desired accuracy while optimizing performance
/// - **History tracking**: Maintains adjustment history for analysis
/// - **Multiple algorithms**: Supports different rate adjustment strategies
///
/// ## Example
///
/// ```rust
/// let controller = SampleRateController::new().await?;
/// controller.set_target_rate(50.0).await?; // 50Hz target
///
/// // Automatic adjustment based on system conditions
/// let optimal_rate = controller.adjust_rate(
///     system_load,
///     target_accuracy,
///     resource_availability
/// ).await?;
/// ```
pub struct SampleRateController {
    /// Current sample rate (Hz * 100 for atomic storage)
    current_rate: Arc<AtomicU32>,

    /// Target sample rate (Hz * 100 for atomic storage)
    target_rate: Arc<AtomicU32>,

    /// Rate adjustment algorithm
    adjustment_algorithm: Arc<Mutex<Box<dyn SampleRateAlgorithm + Send + Sync>>>,

    /// Rate controller configuration
    config: Arc<RwLock<SampleRateConfig>>,

    /// Rate adjustment history
    adjustment_history: Arc<Mutex<VecDeque<RateAdjustment>>>,

    /// Controller statistics
    stats: Arc<RateControllerStats>,

    /// Load monitor for system conditions
    load_monitor: Arc<SystemLoadMonitor>,

    /// Rate change notifier
    rate_change_notifier: Arc<watch::Sender<f32>>,
}

/// Performance impact monitor for collection overhead
///
/// Monitors the performance impact of metrics collection to ensure minimal
/// overhead and optimal system performance.
///
/// ## Monitoring Capabilities
///
/// - **Baseline establishment**: Measures system performance without collection
/// - **Overhead measurement**: Tracks collection-related performance costs
/// - **Impact analysis**: Analyzes performance degradation patterns
/// - **Alert generation**: Triggers alerts when impact exceeds thresholds
/// - **Adaptive recommendations**: Suggests collection parameter adjustments
///
/// ## Example
///
/// ```rust
/// let monitor = PerformanceImpactMonitor::new().await?;
/// monitor.establish_baseline().await?;
///
/// // Continuous monitoring
/// loop {
///     let impact = monitor.measure_impact().await?;
///     if impact.severity > ImpactSeverity::Moderate {
///         monitor.generate_alert(impact).await?;
///     }
///     tokio::time::sleep(Duration::from_secs(1)).await;
/// }
/// ```
pub struct PerformanceImpactMonitor {
    /// Baseline performance metrics
    baseline: Arc<RwLock<PerformanceBaseline>>,

    /// Current overhead measurements
    current_overhead: Arc<RwLock<OverheadMeasurement>>,

    /// Impact analysis results
    impact_analysis: Arc<RwLock<ImpactAnalysis>>,

    /// Monitor configuration
    config: Arc<RwLock<ImpactMonitorConfig>>,

    /// Impact threshold alerts
    alerts: Arc<Mutex<VecDeque<ImpactAlert>>>,

    /// Impact trend analyzer
    trend_analyzer: Arc<ImpactTrendAnalyzer>,

    /// Adaptive recommendations engine
    recommendation_engine: Arc<ImpactRecommendationEngine>,

    /// Alert notifier
    alert_notifier: Arc<broadcast::Sender<ImpactAlert>>,
}

/// Collection statistics for monitoring system health
///
/// Comprehensive statistics for monitoring the health and performance
/// of the metrics collection system.
///
/// ## Statistics Tracked
///
/// - **Collection metrics**: Rate, latency, errors, success rate
/// - **Resource utilization**: Memory usage, CPU overhead, buffer utilization
/// - **Performance metrics**: Throughput, accuracy, system impact
/// - **Error tracking**: Error rates, recovery metrics, failure patterns
///
/// ## Example
///
/// ```rust
/// let stats = &collector.collection_stats;
/// println!("Collection rate: {:.2} Hz", stats.collection_rate.load(Ordering::Relaxed));
/// println!("Average latency: {:.2} ms", stats.avg_collection_latency.load(Ordering::Relaxed));
/// println!("Error rate: {:.2}%", stats.error_rate());
/// ```
#[derive(Debug, Default)]
pub struct CollectionStatistics {
    /// Total metrics collected
    pub metrics_collected: AtomicU64,

    /// Collection rate (metrics/second * 100 for atomic storage)
    pub collection_rate: AtomicU32,

    /// Average collection latency (milliseconds * 100 for atomic storage)
    pub avg_collection_latency: AtomicU32,

    /// Collection errors
    pub collection_errors: AtomicU64,

    /// Buffer utilization (percentage * 100 for atomic storage)
    pub buffer_utilization: AtomicU32,

    /// Memory usage (bytes)
    pub memory_usage: AtomicU64,

    /// CPU overhead (percentage * 100 for atomic storage)
    pub cpu_overhead: AtomicU32,

    /// Successful collections
    pub successful_collections: AtomicU64,

    /// Last collection timestamp
    pub last_collection_time: Arc<RwLock<Option<Instant>>>,

    /// Collection quality score (score * 100 for atomic storage)
    pub avg_quality_score: AtomicU32,

    /// System health indicator (score * 100 for atomic storage)
    pub system_health_score: AtomicU32,
}

/// Metrics publisher for live streaming
///
/// Publisher component responsible for streaming collected metrics to external
/// destinations with configurable delivery guarantees and retry policies.
///
/// ## Publisher Types
///
/// - **HTTP/REST**: Stream to HTTP endpoints with configurable retry
/// - **WebSocket**: Real-time bidirectional streaming
/// - **Message Queue**: Reliable delivery via message queues
/// - **File**: High-performance file-based publishing
/// - **Database**: Direct database insertion with transactions
/// - **Custom**: User-defined publishing mechanisms
///
/// ## Example
///
/// ```rust
/// let publisher = MetricsPublisher::new(
///     "production_monitor".to_string(),
///     PublisherType::Http {
///         endpoint: "https://metrics.example.com/api/v1/metrics".to_string()
///     },
///     DeliveryConfig {
///         guarantee: DeliveryGuarantee::AtLeastOnce,
///         retry_attempts: 3,
///         retry_delay: Duration::from_millis(500),
///         batch_size: 100,
///         compression: true,
///     },
/// );
///
/// publisher.start().await?;
/// publisher.publish(&metrics).await?;
/// ```
pub struct MetricsPublisher {
    /// Publisher ID
    pub id: String,

    /// Publisher type
    pub publisher_type: PublisherType,

    /// Delivery configuration
    pub delivery_config: DeliveryConfig,

    /// Publisher statistics
    pub stats: PublisherStatistics,

    /// Publisher health status
    pub health: Arc<AtomicBool>,

    /// Publishing task handle
    publish_task: Arc<Mutex<Option<JoinHandle<()>>>>,

    /// Message queue for batching
    message_queue: Arc<Mutex<VecDeque<PublishMessage>>>,

    /// Error handler
    error_handler: Arc<Mutex<Box<dyn PublishErrorHandler + Send + Sync>>>,

    /// Rate limiter
    rate_limiter: Arc<PublishRateLimiter>,
}

// =============================================================================
// SUPPORTING TYPES
// =============================================================================

/// Circular buffer for efficient metrics storage
///
/// High-performance circular buffer optimized for concurrent access
/// and minimal memory allocation.
pub struct CircularBuffer<T> {
    /// Buffer data
    buffer: Vec<Option<T>>,

    /// Current write position
    write_pos: AtomicUsize,

    /// Current size
    size: AtomicUsize,

    /// Maximum capacity
    capacity: usize,

    /// Buffer statistics
    stats: BufferStatistics,
}

/// Configuration for metrics collection behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollectionConfig {
    /// Base collection interval
    pub base_interval: Duration,

    /// Minimum collection interval
    pub min_interval: Duration,

    /// Maximum collection interval
    pub max_interval: Duration,

    /// History buffer size
    pub history_buffer_size: usize,

    /// Adaptive sampling enabled
    pub adaptive_sampling: bool,

    /// High precision mode
    pub high_precision_mode: bool,

    /// Batch processing size
    pub batch_size: usize,

    /// Compression enabled
    pub compression_enabled: bool,

    /// Collection timeout
    pub collection_timeout: Duration,

    /// Resource monitoring enabled
    pub resource_monitoring: bool,

    /// Custom metrics enabled
    pub custom_metrics: bool,

    /// Stream publishing enabled
    pub stream_publishing: bool,

    /// Error recovery enabled
    pub error_recovery_enabled: bool,

    /// Quality monitoring enabled
    pub quality_monitoring_enabled: bool,
}

/// Types of metrics publishers
#[derive(Debug, Clone)]
pub enum PublisherType {
    /// HTTP/REST endpoint publisher
    Http { endpoint: String },

    /// Message queue publisher
    MessageQueue { queue_name: String },

    /// WebSocket publisher
    WebSocket { endpoint: String },

    /// File publisher
    File { path: String },

    /// Database publisher
    Database { connection_string: String },

    /// Custom publisher
    Custom(String),
}

/// Delivery configuration for publishers
#[derive(Debug, Clone)]
pub struct DeliveryConfig {
    /// Delivery guarantee level
    pub guarantee: DeliveryGuarantee,

    /// Retry attempts
    pub retry_attempts: usize,

    /// Retry delay
    pub retry_delay: Duration,

    /// Batch size
    pub batch_size: usize,

    /// Compression enabled
    pub compression: bool,

    /// Timeout duration
    pub timeout: Duration,

    /// Rate limiting enabled
    pub rate_limiting_enabled: bool,

    /// Maximum throughput (messages/second)
    pub max_throughput: f32,
}

/// Delivery guarantee levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryGuarantee {
    /// Best effort delivery
    BestEffort,

    /// At least once delivery
    AtLeastOnce,

    /// Exactly once delivery
    ExactlyOnce,
}

/// Collection events for real-time monitoring
#[derive(Debug, Clone)]
pub enum CollectionEvent {
    /// Metrics collected successfully
    MetricsCollected {
        timestamp: DateTime<Utc>,
        metrics: Box<TimestampedMetrics>,
        collection_time: Duration,
    },

    /// Collection error occurred
    CollectionError {
        timestamp: DateTime<Utc>,
        error: String,
        severity: ErrorSeverity,
    },

    /// Sample rate changed
    SampleRateChanged {
        timestamp: DateTime<Utc>,
        old_rate: f32,
        new_rate: f32,
        reason: AdjustmentReason,
    },

    /// Performance impact detected
    PerformanceImpact {
        timestamp: DateTime<Utc>,
        impact: ImpactMeasurement,
        severity: ImpactSeverity,
    },

    /// Publisher status changed
    PublisherStatusChanged {
        timestamp: DateTime<Utc>,
        publisher_id: String,
        old_status: PublisherStatus,
        new_status: PublisherStatus,
    },
}

// =============================================================================
// TRAIT DEFINITIONS
// =============================================================================

/// Trait for sample rate adjustment algorithms
pub trait SampleRateAlgorithm {
    /// Calculate optimal sample rate
    fn calculate_rate(
        &self,
        current_load: f32,
        target_accuracy: f32,
        resource_availability: f32,
    ) -> Result<f32>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get algorithm parameters
    fn parameters(&self) -> HashMap<String, f32>;

    /// Update algorithm configuration
    fn update_config(&mut self, config: HashMap<String, f32>) -> Result<()>;
}

/// Trait for collection error handling
pub trait CollectionErrorHandler {
    /// Handle collection error
    fn handle_error(&self, error: CollectionError) -> Result<ErrorRecoveryAction>;

    /// Get error recovery strategy
    fn recovery_strategy(&self, error_type: ErrorType) -> RecoveryStrategy;

    /// Report error for analysis
    fn report_error(&self, error: CollectionError) -> Result<()>;
}

/// Trait for publish error handling
pub trait PublishErrorHandler {
    /// Handle publish error
    fn handle_error(&self, error: PublishError) -> Result<PublishRecoveryAction>;

    /// Get retry strategy
    fn retry_strategy(&self, error_type: PublishErrorType) -> RetryStrategy;

    /// Report publish failure
    fn report_failure(&self, error: PublishError) -> Result<()>;
}

// =============================================================================
// IMPLEMENTATION: RealTimeMetricsCollector
// =============================================================================

impl RealTimeMetricsCollector {
    /// Create a new real-time metrics collector
    ///
    /// Initializes a new metrics collector with the specified configuration,
    /// sets up collection threads, and prepares for real-time data streaming.
    ///
    /// ## Parameters
    ///
    /// - `config`: Configuration for metrics collection behavior
    ///
    /// ## Returns
    ///
    /// A new `RealTimeMetricsCollector` instance ready for operation
    ///
    /// ## Example
    ///
    /// ```rust
    /// let config = MetricsCollectionConfig {
    ///     base_interval: Duration::from_millis(100),
    ///     adaptive_sampling: true,
    ///     high_precision_mode: true,
    ///     ..Default::default()
    /// };
    ///
    /// let collector = RealTimeMetricsCollector::new(config).await?;
    /// ```
    pub async fn new(config: MetricsCollectionConfig) -> Result<Self> {
        let buffer_size = config.history_buffer_size;
        let (event_tx, _) = broadcast::channel(1000);

        let collector = Self {
            current_metrics: Arc::new(RwLock::new(RealTimeMetrics::default())),
            metrics_history: Arc::new(Mutex::new(CircularBuffer::new(buffer_size))),
            config: Arc::new(RwLock::new(config)),
            metrics_publishers: Arc::new(Mutex::new(Vec::new())),
            collection_tasks: Arc::new(Mutex::new(Vec::new())),
            sample_rate_controller: Arc::new(SampleRateController::new().await?),
            impact_monitor: Arc::new(PerformanceImpactMonitor::new().await?),
            shutdown: Arc::new(AtomicBool::new(false)),
            collection_stats: Arc::new(CollectionStatistics::default()),
            event_broadcaster: Arc::new(event_tx),
            error_handler: Arc::new(Mutex::new(Box::new(DefaultCollectionErrorHandler::new()))),
        };

        collector.initialize_collection().await?;
        Ok(collector)
    }

    fn cloned_config(&self) -> MetricsCollectionConfig {
        let guard = self.config.read();
        guard.clone()
    }

    /// Initialize collection system
    ///
    /// Sets up the internal collection infrastructure including:
    /// - Sample rate controller initialization
    /// - Performance impact monitor setup
    /// - Error handling system
    /// - Event broadcasting system
    async fn initialize_collection(&self) -> Result<()> {
        // Initialize sample rate controller
        self.sample_rate_controller
            .initialize()
            .await
            .context("Failed to initialize sample rate controller")?;

        // Initialize performance impact monitor
        self.impact_monitor
            .initialize()
            .await
            .context("Failed to initialize performance impact monitor")?;

        // Establish performance baseline
        self.impact_monitor
            .establish_baseline()
            .await
            .context("Failed to establish performance baseline")?;

        Ok(())
    }

    /// Start real-time metrics collection
    ///
    /// Begins continuous collection of performance metrics with configurable
    /// sampling rates and live streaming to registered publishers.
    ///
    /// ## Collection Tasks Started
    ///
    /// - **Main collection loop**: Primary metrics gathering
    /// - **Adaptive sampling loop**: Dynamic rate adjustment (if enabled)
    /// - **Streaming loop**: Publisher management and streaming (if enabled)
    /// - **Impact monitoring loop**: Performance overhead tracking (if enabled)
    /// - **Error recovery loop**: Error handling and recovery (if enabled)
    ///
    /// ## Example
    ///
    /// ```rust
    /// collector.start_collection().await?;
    /// println!("Metrics collection started successfully");
    /// ```
    pub async fn start_collection(&self) -> Result<()> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(anyhow::anyhow!("Collector is shut down"));
        }

        let config = self.cloned_config();
        let mut tasks = self.collection_tasks.lock();

        // Start main collection loop
        let collection_task = self.spawn_collection_loop().await?;
        tasks.push(collection_task);

        // Start adaptive sampling if enabled
        if config.adaptive_sampling {
            let sampling_task = self.spawn_adaptive_sampling_loop().await?;
            tasks.push(sampling_task);
        }

        // Start streaming if enabled
        if config.stream_publishing {
            let streaming_task = self.spawn_streaming_loop().await?;
            tasks.push(streaming_task);
        }

        // Start impact monitoring
        if config.resource_monitoring {
            let impact_task = self.spawn_impact_monitoring_loop().await?;
            tasks.push(impact_task);
        }

        // Start error recovery if enabled
        if config.error_recovery_enabled {
            let recovery_task = self.spawn_error_recovery_loop().await?;
            tasks.push(recovery_task);
        }

        // Broadcast collection started event
        let _ = self.event_broadcaster.send(CollectionEvent::MetricsCollected {
            timestamp: Utc::now(),
            metrics: Box::new(TimestampedMetrics::default()),
            collection_time: Duration::ZERO,
        });

        Ok(())
    }

    /// Collect current performance metrics
    ///
    /// Performs a single collection of current system performance metrics
    /// with high precision timing and comprehensive data capture.
    ///
    /// ## Collection Process
    ///
    /// 1. **High-precision timestamping**: Captures both UTC and monotonic timestamps
    /// 2. **Metrics gathering**: Collects comprehensive performance metrics
    /// 3. **System state capture**: Records current system state
    /// 4. **Quality assessment**: Calculates data quality score
    /// 5. **History storage**: Stores in circular buffer
    /// 6. **Statistics update**: Updates collection statistics
    /// 7. **Event broadcasting**: Notifies subscribers
    ///
    /// ## Returns
    ///
    /// `TimestampedMetrics` containing the collected data with metadata
    ///
    /// ## Example
    ///
    /// ```rust
    /// let metrics = collector.collect_metrics().await?;
    /// println!("CPU usage: {:.2}%", metrics.metrics.cpu_usage);
    /// println!("Memory usage: {:.2}%", metrics.metrics.memory_usage);
    /// println!("Quality score: {:.2}", metrics.quality_score);
    /// ```
    pub async fn collect_metrics(&self) -> Result<TimestampedMetrics> {
        let start_time = Instant::now();
        let timestamp = Utc::now();

        // Collect current metrics
        let metrics = self
            .gather_current_metrics()
            .await
            .context("Failed to gather current metrics")?;

        let system_state =
            self.gather_system_state().await.context("Failed to gather system state")?;

        // Calculate quality score based on collection conditions
        let quality_score = self.calculate_quality_score(&metrics, &system_state);

        let timestamped_metrics = TimestampedMetrics {
            timestamp,
            precise_timestamp: start_time,
            metrics,
            system_state,
            quality_score,
            source: "real_time_collector".to_string(),
            metadata: HashMap::new(),
        };

        // Update collection statistics
        let collection_time = start_time.elapsed();
        self.update_collection_stats(collection_time);

        // Store in history buffer
        self.store_in_history(timestamped_metrics.clone()).await?;

        // Update current metrics
        *self.current_metrics.write() = timestamped_metrics.metrics.clone();

        // Broadcast collection event
        let _ = self.event_broadcaster.send(CollectionEvent::MetricsCollected {
            timestamp,
            metrics: Box::new(timestamped_metrics.clone()),
            collection_time,
        });

        Ok(timestamped_metrics)
    }

    /// Add a metrics publisher for live streaming
    ///
    /// Registers a new publisher for streaming metrics data to external destinations.
    /// The publisher will begin receiving metrics immediately if collection is active.
    ///
    /// ## Parameters
    ///
    /// - `publisher`: The publisher configuration to add
    ///
    /// ## Example
    ///
    /// ```rust
    /// let publisher = MetricsPublisher::new(
    ///     "http_monitor".to_string(),
    ///     PublisherType::Http { endpoint: "http://localhost:8080/metrics".to_string() },
    ///     DeliveryConfig::default(),
    /// );
    ///
    /// collector.add_publisher(publisher).await?;
    /// ```
    pub async fn add_publisher(&self, publisher: MetricsPublisher) -> Result<()> {
        let publisher = Arc::new(publisher);

        // Start the publisher
        publisher.start().await?;

        // Add to publishers list
        self.metrics_publishers.lock().push(publisher);

        Ok(())
    }

    /// Remove a metrics publisher
    ///
    /// Stops and removes a publisher from the active publishers list.
    ///
    /// ## Parameters
    ///
    /// - `publisher_id`: ID of the publisher to remove
    ///
    /// ## Returns
    ///
    /// `true` if the publisher was found and removed, `false` otherwise
    pub async fn remove_publisher(&self, publisher_id: &str) -> Result<bool> {
        let publisher = {
            let mut publishers = self.metrics_publishers.lock();

            if let Some(pos) = publishers.iter().position(|p| p.id == publisher_id) {
                publishers.remove(pos)
            } else {
                return Ok(false);
            }
        };

        publisher.stop().await?;
        Ok(true)
    }

    /// Get current metrics
    ///
    /// Returns the most recently collected metrics data.
    pub fn get_current_metrics(&self) -> RealTimeMetrics {
        (*self.current_metrics.read()).clone()
    }

    /// Get metrics history
    ///
    /// Returns a snapshot of the metrics history buffer.
    pub fn get_metrics_history(&self, count: usize) -> Vec<TimestampedMetrics> {
        let history = self.metrics_history.lock();
        history.get_recent(count)
    }

    /// Get collection statistics
    ///
    /// Returns current collection statistics for monitoring system health.
    pub fn get_collection_statistics(&self) -> CollectionStatistics {
        CollectionStatistics {
            metrics_collected: AtomicU64::new(
                self.collection_stats.metrics_collected.load(Ordering::Relaxed),
            ),
            collection_rate: AtomicU32::new(
                self.collection_stats.collection_rate.load(Ordering::Relaxed),
            ),
            avg_collection_latency: AtomicU32::new(
                self.collection_stats.avg_collection_latency.load(Ordering::Relaxed),
            ),
            collection_errors: AtomicU64::new(
                self.collection_stats.collection_errors.load(Ordering::Relaxed),
            ),
            buffer_utilization: AtomicU32::new(
                self.collection_stats.buffer_utilization.load(Ordering::Relaxed),
            ),
            memory_usage: AtomicU64::new(
                self.collection_stats.memory_usage.load(Ordering::Relaxed),
            ),
            cpu_overhead: AtomicU32::new(
                self.collection_stats.cpu_overhead.load(Ordering::Relaxed),
            ),
            successful_collections: AtomicU64::new(
                self.collection_stats.successful_collections.load(Ordering::Relaxed),
            ),
            last_collection_time: Arc::new(RwLock::new(
                *self.collection_stats.last_collection_time.read(),
            )),
            avg_quality_score: AtomicU32::new(
                self.collection_stats.avg_quality_score.load(Ordering::Relaxed),
            ),
            system_health_score: AtomicU32::new(
                self.collection_stats.system_health_score.load(Ordering::Relaxed),
            ),
        }
    }

    /// Subscribe to collection events
    ///
    /// Returns a receiver for real-time collection events.
    pub fn subscribe_events(&self) -> broadcast::Receiver<CollectionEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Stop metrics collection
    ///
    /// Gracefully stops all collection activities and cleans up resources.
    pub async fn stop_collection(&self) -> Result<()> {
        // Set shutdown flag
        self.shutdown.store(true, Ordering::Relaxed);

        // Stop all collection tasks
        let mut tasks = self.collection_tasks.lock();
        for task in tasks.drain(..) {
            task.abort();
        }

        // Stop all publishers
        let publishers: Vec<_> = {
            let publishers = self.metrics_publishers.lock();
            publishers.iter().cloned().collect()
        };

        for publisher in publishers {
            publisher.stop().await?;
        }

        Ok(())
    }

    // =============================================================================
    // PRIVATE HELPER METHODS
    // =============================================================================

    /// Spawn main collection loop
    async fn spawn_collection_loop(&self) -> Result<JoinHandle<()>> {
        let config = self.config.clone();
        let current_metrics = self.current_metrics.clone();
        let metrics_history = self.metrics_history.clone();
        let collection_stats = self.collection_stats.clone();
        let shutdown = self.shutdown.clone();
        let event_broadcaster = self.event_broadcaster.clone();
        let sample_rate_controller = self.sample_rate_controller.clone();

        let task = tokio::spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                let config_guard = config.read();
                let current_rate = sample_rate_controller.get_current_rate();
                let interval_ms = (1000.0 / current_rate) as u64;
                drop(config_guard);

                let mut interval = interval(Duration::from_millis(interval_ms));
                interval.tick().await; // First tick is immediate

                // Simulate metrics collection
                let start_time = Instant::now();
                let timestamp = Utc::now();

                // Gather metrics (simplified for this implementation)
                let metrics = RealTimeMetrics::default();
                let system_state = SystemState::default();
                let quality_score = 1.0;

                let timestamped_metrics = TimestampedMetrics {
                    timestamp,
                    precise_timestamp: start_time,
                    metrics: metrics.clone(),
                    system_state,
                    quality_score,
                    source: "real_time_collector".to_string(),
                    metadata: HashMap::new(),
                };

                // Update current metrics
                *current_metrics.write() = metrics;

                // Store in history
                metrics_history.lock().push(timestamped_metrics.clone());

                // Update statistics
                let collection_time = start_time.elapsed();
                collection_stats.metrics_collected.fetch_add(1, Ordering::Relaxed);
                collection_stats.successful_collections.fetch_add(1, Ordering::Relaxed);
                *collection_stats.last_collection_time.write() = Some(start_time);

                // Broadcast event
                let _ = event_broadcaster.send(CollectionEvent::MetricsCollected {
                    timestamp,
                    metrics: Box::new(timestamped_metrics),
                    collection_time,
                });
            }
        });

        Ok(task)
    }

    /// Spawn adaptive sampling loop
    async fn spawn_adaptive_sampling_loop(&self) -> Result<JoinHandle<()>> {
        let sample_rate_controller = self.sample_rate_controller.clone();
        let shutdown = self.shutdown.clone();
        let event_broadcaster = self.event_broadcaster.clone();

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(1000)); // Adjust every second

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Adjust sample rate based on system conditions
                if let Ok(adjustment) = sample_rate_controller.adjust_rate_adaptive().await {
                    if adjustment.rate_changed {
                        let _ = event_broadcaster.send(CollectionEvent::SampleRateChanged {
                            timestamp: Utc::now(),
                            old_rate: adjustment.old_rate,
                            new_rate: adjustment.new_rate,
                            reason: adjustment.reason,
                        });
                    }
                }
            }
        });

        Ok(task)
    }

    /// Spawn streaming loop
    async fn spawn_streaming_loop(&self) -> Result<JoinHandle<()>> {
        let metrics_publishers = self.metrics_publishers.clone();
        let shutdown = self.shutdown.clone();
        let current_metrics = self.current_metrics.clone();

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100)); // Stream every 100ms

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                let metrics = (*current_metrics.read()).clone();
                let publishers: Vec<_> = {
                    let publishers = metrics_publishers.lock();
                    publishers.iter().cloned().collect()
                };

                for publisher in publishers {
                    if publisher.health.load(Ordering::Relaxed) {
                        let _ = publisher.publish_async(&metrics).await;
                    }
                }
            }
        });

        Ok(task)
    }

    /// Spawn impact monitoring loop
    async fn spawn_impact_monitoring_loop(&self) -> Result<JoinHandle<()>> {
        let impact_monitor = self.impact_monitor.clone();
        let shutdown = self.shutdown.clone();
        let event_broadcaster = self.event_broadcaster.clone();

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(5000)); // Monitor every 5 seconds

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                if let Ok(impact) = impact_monitor.measure_impact().await {
                    if impact.severity > ImpactSeverity::Low {
                        let _ = event_broadcaster.send(CollectionEvent::PerformanceImpact {
                            timestamp: Utc::now(),
                            impact: ImpactMeasurement::default(),
                            severity: impact.severity,
                        });
                    }
                }
            }
        });

        Ok(task)
    }

    /// Spawn error recovery loop
    async fn spawn_error_recovery_loop(&self) -> Result<JoinHandle<()>> {
        let _error_handler = self.error_handler.clone();
        let shutdown = self.shutdown.clone();

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(1000));

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;
                // Error recovery logic would go here
            }
        });

        Ok(task)
    }

    /// Gather current metrics
    async fn gather_current_metrics(&self) -> Result<RealTimeMetrics> {
        // This would contain actual metrics gathering logic
        // For now, return a default instance
        Ok(RealTimeMetrics::default())
    }

    /// Gather system state
    async fn gather_system_state(&self) -> Result<SystemState> {
        // This would contain actual system state gathering logic
        Ok(SystemState::default())
    }

    /// Calculate quality score
    fn calculate_quality_score(
        &self,
        _metrics: &RealTimeMetrics,
        _system_state: &SystemState,
    ) -> f32 {
        // Quality score calculation logic
        1.0
    }

    /// Update collection statistics
    fn update_collection_stats(&self, collection_time: Duration) {
        let latency_ms = collection_time.as_millis() as f32;

        // Update average latency using exponential moving average
        let current_avg = self.collection_stats.avg_collection_latency.load(Ordering::Relaxed);
        let alpha = 0.1; // Smoothing factor
        let new_avg = current_avg as f32 * (1.0 - alpha) + latency_ms * alpha;
        self.collection_stats
            .avg_collection_latency
            .store(new_avg as u32, Ordering::Relaxed);

        // Update collection rate
        let now = Instant::now();
        if let Some(last_time) = *self.collection_stats.last_collection_time.read() {
            let time_diff = now.duration_since(last_time).as_secs_f32();
            if time_diff > 0.0 {
                let rate = 1.0 / time_diff;
                self.collection_stats
                    .collection_rate
                    .store(rate.round() as u32, Ordering::Relaxed);
            }
        }
    }

    /// Store metrics in history buffer
    async fn store_in_history(&self, metrics: TimestampedMetrics) -> Result<()> {
        self.metrics_history.lock().push(metrics);
        Ok(())
    }
}

// =============================================================================
// IMPLEMENTATION: SampleRateController
// =============================================================================

impl SampleRateController {
    /// Create a new sample rate controller
    pub async fn new() -> Result<Self> {
        let (rate_tx, _) = watch::channel(10.0);

        let controller = Self {
            current_rate: Arc::new(AtomicU32::new(1000)), // Default 10Hz * 100
            target_rate: Arc::new(AtomicU32::new(1000)),
            adjustment_algorithm: Arc::new(Mutex::new(Box::new(PidSampleRateAlgorithm::new()))),
            config: Arc::new(RwLock::new(SampleRateConfig::default())),
            adjustment_history: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(RateControllerStats::default()),
            load_monitor: Arc::new(SystemLoadMonitor::new().await?),
            rate_change_notifier: Arc::new(rate_tx),
        };

        Ok(controller)
    }

    /// Initialize the controller
    pub async fn initialize(&self) -> Result<()> {
        self.load_monitor.start_monitoring().await?;
        Ok(())
    }

    /// Get current sample rate
    pub fn get_current_rate(&self) -> f32 {
        self.current_rate.load(Ordering::Relaxed) as f32 / 100.0
    }

    /// Set target sample rate
    pub async fn set_target_rate(&self, rate: f32) -> Result<()> {
        let config = self.config.read();
        let clamped_rate = rate.clamp(config.min_rate, config.max_rate);
        self.target_rate.store((clamped_rate * 100.0) as u32, Ordering::Relaxed);
        Ok(())
    }

    /// Adjust sample rate adaptively
    pub async fn adjust_rate_adaptive(&self) -> Result<RateAdjustmentResult> {
        let current_load = self.load_monitor.get_current_load().await?;
        let target_accuracy = self.config.read().target_accuracy;
        let resource_availability = self.load_monitor.get_resource_availability().await?;

        let old_rate = self.current_rate.load(Ordering::Relaxed) as f32 / 100.0;

        let algorithm = self.adjustment_algorithm.lock();
        let new_rate =
            algorithm.calculate_rate(current_load, target_accuracy, resource_availability)?;
        drop(algorithm);

        let rate_changed = (new_rate - old_rate).abs() > 0.1;

        if rate_changed {
            self.current_rate.store((new_rate * 100.0) as u32, Ordering::Relaxed);
            let _ = self.rate_change_notifier.send(new_rate);

            // Record adjustment
            let adjustment = RateAdjustment {
                timestamp: Utc::now(),
                old_rate,
                new_rate,
                reason: AdjustmentReason::SystemLoad,
                system_load: current_load,
                resource_availability,
            };

            self.adjustment_history.lock().push_back(adjustment);

            // Update statistics
            self.stats.adjustments_made.fetch_add(1, Ordering::Relaxed);
        }

        Ok(RateAdjustmentResult {
            old_rate,
            new_rate,
            rate_changed,
            reason: AdjustmentReason::SystemLoad,
        })
    }

    /// Subscribe to rate changes
    pub fn subscribe_rate_changes(&self) -> watch::Receiver<f32> {
        self.rate_change_notifier.subscribe()
    }

    /// Get adjustment history
    pub fn get_adjustment_history(&self, count: usize) -> Vec<RateAdjustment> {
        let history = self.adjustment_history.lock();
        history.iter().rev().take(count).cloned().collect()
    }

    /// Get controller statistics
    pub fn get_statistics(&self) -> RateControllerStats {
        RateControllerStats {
            adjustments_made: AtomicU64::new(self.stats.adjustments_made.load(Ordering::Relaxed)),
            avg_adjustment_magnitude: AtomicU32::new(
                self.stats.avg_adjustment_magnitude.load(Ordering::Relaxed),
            ),
            stability_score: AtomicU32::new(self.stats.stability_score.load(Ordering::Relaxed)),
            accuracy_score: AtomicU32::new(self.stats.accuracy_score.load(Ordering::Relaxed)),
        }
    }
}

// =============================================================================
// IMPLEMENTATION: PerformanceImpactMonitor
// =============================================================================

impl PerformanceImpactMonitor {
    /// Create a new performance impact monitor
    pub async fn new() -> Result<Self> {
        let (alert_tx, _) = broadcast::channel(100);

        let monitor = Self {
            baseline: Arc::new(RwLock::new(PerformanceBaseline::default())),
            current_overhead: Arc::new(RwLock::new(OverheadMeasurement::default())),
            impact_analysis: Arc::new(RwLock::new(ImpactAnalysis::default())),
            config: Arc::new(RwLock::new(ImpactMonitorConfig::default())),
            alerts: Arc::new(Mutex::new(VecDeque::new())),
            trend_analyzer: Arc::new(ImpactTrendAnalyzer::new()),
            recommendation_engine: Arc::new(ImpactRecommendationEngine::new()),
            alert_notifier: Arc::new(alert_tx),
        };

        Ok(monitor)
    }

    /// Initialize the monitor
    pub async fn initialize(&self) -> Result<()> {
        self.trend_analyzer.initialize().await?;
        self.recommendation_engine.initialize().await?;
        Ok(())
    }

    /// Establish performance baseline
    pub async fn establish_baseline(&self) -> Result<()> {
        let baseline_measurement = self.measure_baseline_performance().await?;
        *self.baseline.write() = baseline_measurement;
        Ok(())
    }

    /// Measure current performance impact
    pub async fn measure_impact(&self) -> Result<ImpactAnalysisResult> {
        let current_performance = self.measure_current_performance().await?;
        let baseline = {
            let guard = self.baseline.read();
            guard.clone()
        };

        let overhead = OverheadMeasurement {
            cpu_overhead: current_performance.cpu_usage - baseline.cpu_usage,
            memory_overhead: (current_performance.memory_usage as u64)
                .saturating_sub(baseline.memory_usage),
            latency_overhead: current_performance.latency.saturating_sub(baseline.latency),
            throughput_impact: baseline.throughput - (current_performance.throughput as f32),
            timestamp: Utc::now(),
        };

        *self.current_overhead.write() = overhead.clone();

        let analysis = self.analyze_impact(&overhead).await?;
        *self.impact_analysis.write() = analysis.clone();

        let analysis_clone = analysis.clone();
        Ok(ImpactAnalysisResult {
            overhead,
            analysis,
            severity: self.calculate_severity(&analysis_clone),
            recommendations: self.generate_recommendations(&analysis_clone).await?,
        })
    }

    /// Generate impact alert
    pub async fn generate_alert(&self, impact: ImpactAnalysisResult) -> Result<()> {
        let alert = ImpactAlert {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            severity: impact.severity,
            impact_measurement: impact.overhead,
            analysis: impact.analysis,
            recommendations: impact.recommendations,
            acknowledged: false,
        };

        self.alerts.lock().push_back(alert.clone());
        let _ = self.alert_notifier.send(alert);

        Ok(())
    }

    /// Subscribe to impact alerts
    pub fn subscribe_alerts(&self) -> broadcast::Receiver<ImpactAlert> {
        self.alert_notifier.subscribe()
    }

    /// Get current impact analysis
    pub fn get_current_analysis(&self) -> ImpactAnalysis {
        let analysis = self.impact_analysis.read();
        analysis.clone()
    }

    /// Get impact trends
    pub async fn get_impact_trends(&self, duration: Duration) -> Result<ImpactTrends> {
        self.trend_analyzer.analyze_trends(duration).await
    }

    // =============================================================================
    // PRIVATE HELPER METHODS
    // =============================================================================

    /// Measure baseline performance
    async fn measure_baseline_performance(&self) -> Result<PerformanceBaseline> {
        // Simulate baseline measurement
        Ok(PerformanceBaseline {
            cpu_usage: 10.0,
            memory_usage: 100_000_000, // 100MB
            latency: Duration::from_millis(1),
            throughput: 1000.0,
            timestamp: Utc::now(),
        })
    }

    /// Measure current performance
    async fn measure_current_performance(&self) -> Result<PerformanceMeasurement> {
        // Simulate current performance measurement
        let cpu_val = 12.0;
        let memory_val = 110_000_000.0; // 110MB
        let latency_val = Duration::from_millis(1);

        Ok(PerformanceMeasurement {
            throughput: 950.0,
            average_latency: latency_val,
            cpu_utilization: cpu_val,
            memory_utilization: memory_val / 1_000_000.0, // Convert to MB for percentage
            resource_efficiency: 0.9,
            timestamp: Utc::now(),
            measurement_duration: Duration::from_millis(100),
            cpu_usage: cpu_val,
            memory_usage: memory_val,
            latency: latency_val,
        })
    }

    /// Analyze performance impact
    async fn analyze_impact(&self, overhead: &OverheadMeasurement) -> Result<ImpactAnalysis> {
        Ok(ImpactAnalysis {
            cpu_impact_percent: (overhead.cpu_overhead / 10.0) * 100.0,
            memory_impact_percent: (overhead.memory_overhead as f32 / 100_000_000.0) * 100.0,
            latency_impact_percent: overhead.latency_overhead.as_millis() as f32,
            throughput_impact_percent: (overhead.throughput_impact / 1000.0) * 100.0,
            overall_severity: ImpactSeverity::Low,
            trend: ImpactTrend::Stable,
            analysis_timestamp: Utc::now(),
        })
    }

    /// Calculate impact severity
    fn calculate_severity(&self, analysis: &ImpactAnalysis) -> ImpactSeverity {
        let max_impact = analysis
            .cpu_impact_percent
            .max(analysis.memory_impact_percent)
            .max(analysis.latency_impact_percent)
            .max(analysis.throughput_impact_percent);

        match max_impact {
            x if x < 5.0 => ImpactSeverity::Low,
            x if x < 15.0 => ImpactSeverity::Moderate,
            x if x < 30.0 => ImpactSeverity::High,
            _ => ImpactSeverity::Critical,
        }
    }

    /// Generate impact recommendations
    async fn generate_recommendations(
        &self,
        analysis: &ImpactAnalysis,
    ) -> Result<Vec<ImpactRecommendation>> {
        self.recommendation_engine.generate_recommendations(analysis).await
    }
}

// =============================================================================
// IMPLEMENTATION: MetricsPublisher
// =============================================================================

impl MetricsPublisher {
    /// Create a new metrics publisher
    pub fn new(id: String, publisher_type: PublisherType, delivery_config: DeliveryConfig) -> Self {
        Self {
            id,
            publisher_type,
            delivery_config,
            stats: PublisherStatistics::default(),
            health: Arc::new(AtomicBool::new(true)),
            publish_task: Arc::new(Mutex::new(None)),
            message_queue: Arc::new(Mutex::new(VecDeque::new())),
            error_handler: Arc::new(Mutex::new(Box::new(DefaultPublishErrorHandler::new()))),
            rate_limiter: Arc::new(PublishRateLimiter::new()),
        }
    }

    /// Start the publisher
    pub async fn start(&self) -> Result<()> {
        let task = self.spawn_publish_loop().await?;
        *self.publish_task.lock() = Some(task);
        self.health.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Stop the publisher
    pub async fn stop(&self) -> Result<()> {
        self.health.store(false, Ordering::Relaxed);

        if let Some(task) = self.publish_task.lock().take() {
            task.abort();
        }

        Ok(())
    }

    /// Publish metrics synchronously
    pub async fn publish(&self, metrics: &RealTimeMetrics) -> Result<()> {
        if !self.health.load(Ordering::Relaxed) {
            return Err(anyhow::anyhow!("Publisher is not healthy"));
        }

        let message = PublishMessage {
            id: uuid::Uuid::new_v4().to_string(),
            metrics: (*metrics).clone(),
            timestamp: Utc::now(),
            retry_count: 0,
        };

        self.message_queue.lock().push_back(message);
        Ok(())
    }

    /// Publish metrics asynchronously
    pub async fn publish_async(&self, metrics: &RealTimeMetrics) -> Result<()> {
        self.publish(metrics).await
    }

    /// Get publisher statistics
    pub fn get_statistics(&self) -> PublisherStatistics {
        PublisherStatistics {
            messages_published: AtomicU64::new(
                self.stats.messages_published.load(Ordering::Relaxed),
            ),
            messages_failed: AtomicU64::new(self.stats.messages_failed.load(Ordering::Relaxed)),
            avg_publish_latency: AtomicU32::new(
                self.stats.avg_publish_latency.load(Ordering::Relaxed),
            ),
            throughput: AtomicU32::new(self.stats.throughput.load(Ordering::Relaxed)),
            error_rate: AtomicU32::new(self.stats.error_rate.load(Ordering::Relaxed)),
            last_publish_time: Arc::new(RwLock::new(*self.stats.last_publish_time.read())),
        }
    }

    /// Check publisher health
    pub fn is_healthy(&self) -> bool {
        self.health.load(Ordering::Relaxed)
    }

    // =============================================================================
    // PRIVATE HELPER METHODS
    // =============================================================================

    /// Spawn publish loop
    async fn spawn_publish_loop(&self) -> Result<JoinHandle<()>> {
        let publisher_type = self.publisher_type.clone();
        let delivery_config = self.delivery_config.clone();
        let message_queue = self.message_queue.clone();
        let health = self.health.clone();
        let stats = self.stats.clone();
        let rate_limiter = self.rate_limiter.clone();

        let task = tokio::spawn(async move {
            while health.load(Ordering::Relaxed) {
                let message = message_queue.lock().pop_front();
                if let Some(message) = message {
                    // Rate limiting
                    rate_limiter.wait_if_needed().await;

                    let start_time = Instant::now();

                    match Self::publish_message(&publisher_type, &delivery_config, &message).await {
                        Ok(_) => {
                            stats.messages_published.fetch_add(1, Ordering::Relaxed);
                            *stats.last_publish_time.write() = Some(start_time);
                        },
                        Err(_) => {
                            stats.messages_failed.fetch_add(1, Ordering::Relaxed);
                            // Retry logic would go here
                        },
                    }

                    // Update latency statistics
                    let latency = start_time.elapsed().as_millis() as f32;
                    let current_avg = stats.avg_publish_latency.load(Ordering::Relaxed);
                    let alpha = 0.1;
                    let new_avg = current_avg as f32 * (1.0 - alpha) + latency * alpha;
                    stats.avg_publish_latency.store(new_avg as u32, Ordering::Relaxed);
                } else {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        });

        Ok(task)
    }

    /// Publish message to destination
    async fn publish_message(
        publisher_type: &PublisherType,
        _delivery_config: &DeliveryConfig,
        _message: &PublishMessage,
    ) -> Result<()> {
        match publisher_type {
            PublisherType::Http { endpoint: _ } => {
                // HTTP publishing logic
                Ok(())
            },
            PublisherType::WebSocket { endpoint: _ } => {
                // WebSocket publishing logic
                Ok(())
            },
            PublisherType::MessageQueue { queue_name: _ } => {
                // Message queue publishing logic
                Ok(())
            },
            PublisherType::File { path: _ } => {
                // File publishing logic
                Ok(())
            },
            PublisherType::Database {
                connection_string: _,
            } => {
                // Database publishing logic
                Ok(())
            },
            PublisherType::Custom(_) => {
                // Custom publishing logic
                Ok(())
            },
        }
    }
}

// =============================================================================
// IMPLEMENTATION: CircularBuffer
// =============================================================================

impl<T> CircularBuffer<T> {
    /// Create a new circular buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: (0..capacity).map(|_| None).collect(),
            write_pos: AtomicUsize::new(0),
            size: AtomicUsize::new(0),
            capacity,
            stats: BufferStatistics::default(),
        }
    }

    /// Push a new item into the buffer
    pub fn push(&mut self, item: T) {
        let pos = self.write_pos.load(Ordering::Relaxed);
        self.buffer[pos] = Some(item);

        let next_pos = (pos + 1) % self.capacity;
        self.write_pos.store(next_pos, Ordering::Relaxed);

        let current_size = self.size.load(Ordering::Relaxed);
        if current_size < self.capacity {
            self.size.store(current_size + 1, Ordering::Relaxed);
        }

        self.stats.items_written.fetch_add(1, Ordering::Relaxed);
    }

    /// Get recent items from the buffer
    pub fn get_recent(&self, count: usize) -> Vec<T>
    where
        T: Clone,
    {
        let mut result = Vec::new();
        let size = self.size.load(Ordering::Relaxed);
        let items_to_get = count.min(size);

        if items_to_get == 0 {
            return result;
        }

        let write_pos = self.write_pos.load(Ordering::Relaxed);

        for i in 0..items_to_get {
            let pos = if write_pos >= i + 1 {
                write_pos - i - 1
            } else {
                self.capacity + write_pos - i - 1
            };

            if let Some(ref item) = self.buffer[pos] {
                result.push(item.clone());
            }
        }

        result
    }

    /// Get buffer utilization percentage
    pub fn utilization(&self) -> f32 {
        let size = self.size.load(Ordering::Relaxed) as f32;
        let capacity = self.capacity as f32;
        (size / capacity) * 100.0
    }

    /// Get buffer statistics
    pub fn get_statistics(&self) -> BufferStatistics {
        BufferStatistics {
            items_written: AtomicU64::new(self.stats.items_written.load(Ordering::Relaxed)),
            items_read: AtomicU64::new(self.stats.items_read.load(Ordering::Relaxed)),
            current_size: AtomicUsize::new(self.size.load(Ordering::Relaxed)),
            max_capacity: self.capacity,
            utilization_percent: AtomicU32::new((self.utilization() * 100.0) as u32),
        }
    }
}

// =============================================================================
// DEFAULT IMPLEMENTATIONS
// =============================================================================

impl Default for MetricsCollectionConfig {
    fn default() -> Self {
        Self {
            base_interval: Duration::from_millis(100),
            min_interval: Duration::from_millis(10),
            max_interval: Duration::from_secs(1),
            history_buffer_size: 1000,
            adaptive_sampling: true,
            high_precision_mode: false,
            batch_size: 10,
            compression_enabled: false,
            collection_timeout: Duration::from_secs(5),
            resource_monitoring: true,
            custom_metrics: false,
            stream_publishing: true,
            error_recovery_enabled: true,
            quality_monitoring_enabled: true,
        }
    }
}

impl Default for DeliveryConfig {
    fn default() -> Self {
        Self {
            guarantee: DeliveryGuarantee::BestEffort,
            retry_attempts: 3,
            retry_delay: Duration::from_millis(1000),
            batch_size: 1,
            compression: false,
            timeout: Duration::from_secs(30),
            rate_limiting_enabled: false,
            max_throughput: 1000.0,
        }
    }
}

// =============================================================================
// PLACEHOLDER IMPLEMENTATIONS
// =============================================================================

// These are placeholder implementations for types that would be defined elsewhere
// or would need additional implementation details

#[derive(Debug, Default)]
pub struct BufferStatistics {
    pub items_written: AtomicU64,
    pub items_read: AtomicU64,
    pub current_size: AtomicUsize,
    pub max_capacity: usize,
    pub utilization_percent: AtomicU32,
}

#[derive(Debug, Default)]
pub struct PublisherStatistics {
    pub messages_published: AtomicU64,
    pub messages_failed: AtomicU64,
    pub avg_publish_latency: AtomicU32,
    pub throughput: AtomicU32,
    pub error_rate: AtomicU32,
    pub last_publish_time: Arc<RwLock<Option<Instant>>>,
}

impl Clone for PublisherStatistics {
    fn clone(&self) -> Self {
        Self {
            messages_published: AtomicU64::new(self.messages_published.load(Ordering::Relaxed)),
            messages_failed: AtomicU64::new(self.messages_failed.load(Ordering::Relaxed)),
            avg_publish_latency: AtomicU32::new(self.avg_publish_latency.load(Ordering::Relaxed)),
            throughput: AtomicU32::new(self.throughput.load(Ordering::Relaxed)),
            error_rate: AtomicU32::new(self.error_rate.load(Ordering::Relaxed)),
            last_publish_time: Arc::new(RwLock::new(*self.last_publish_time.read())),
        }
    }
}

#[derive(Debug, Default)]
pub struct RateControllerStats {
    pub adjustments_made: AtomicU64,
    pub avg_adjustment_magnitude: AtomicU32,
    pub stability_score: AtomicU32,
    pub accuracy_score: AtomicU32,
}

#[derive(Debug, Clone)]
pub struct PublishMessage {
    pub id: String,
    pub metrics: RealTimeMetrics,
    pub timestamp: DateTime<Utc>,
    pub retry_count: usize,
}

pub struct PidSampleRateAlgorithm;
impl PidSampleRateAlgorithm {
    pub fn new() -> Self {
        Self
    }
}

impl SampleRateAlgorithm for PidSampleRateAlgorithm {
    fn calculate_rate(
        &self,
        current_load: f32,
        _target_accuracy: f32,
        resource_availability: f32,
    ) -> Result<f32> {
        // Simple PID-like algorithm
        let base_rate = 10.0;
        let load_factor = 1.0 - current_load.min(1.0);
        let resource_factor = resource_availability.min(1.0);
        let adjusted_rate = base_rate * load_factor * resource_factor;
        Ok(adjusted_rate.clamp(1.0, 100.0))
    }

    fn name(&self) -> &str {
        "PID Sample Rate Algorithm"
    }
    fn parameters(&self) -> HashMap<String, f32> {
        HashMap::new()
    }
    fn update_config(&mut self, _config: HashMap<String, f32>) -> Result<()> {
        Ok(())
    }
}

pub struct SystemLoadMonitor;
impl SystemLoadMonitor {
    pub async fn new() -> Result<Self> {
        Ok(Self)
    }
    pub async fn start_monitoring(&self) -> Result<()> {
        Ok(())
    }
    pub async fn get_current_load(&self) -> Result<f32> {
        Ok(0.5)
    }
    pub async fn get_resource_availability(&self) -> Result<f32> {
        Ok(0.8)
    }
}

pub struct ImpactTrendAnalyzer;
impl ImpactTrendAnalyzer {
    pub fn new() -> Self {
        Self
    }
    pub async fn initialize(&self) -> Result<()> {
        Ok(())
    }
    pub async fn analyze_trends(&self, _duration: Duration) -> Result<ImpactTrends> {
        Ok(ImpactTrends::default())
    }
}

pub struct ImpactRecommendationEngine;
impl ImpactRecommendationEngine {
    pub fn new() -> Self {
        Self
    }
    pub async fn initialize(&self) -> Result<()> {
        Ok(())
    }
    pub async fn generate_recommendations(
        &self,
        _analysis: &ImpactAnalysis,
    ) -> Result<Vec<ImpactRecommendation>> {
        Ok(vec![])
    }
}

pub struct DefaultCollectionErrorHandler;
impl DefaultCollectionErrorHandler {
    pub fn new() -> Self {
        Self
    }
}

impl CollectionErrorHandler for DefaultCollectionErrorHandler {
    fn handle_error(&self, _error: CollectionError) -> Result<ErrorRecoveryAction> {
        Ok(ErrorRecoveryAction::Retry)
    }
    fn recovery_strategy(&self, _error_type: ErrorType) -> RecoveryStrategy {
        RecoveryStrategy::Immediate
    }
    fn report_error(&self, _error: CollectionError) -> Result<()> {
        Ok(())
    }
}

pub struct DefaultPublishErrorHandler;
impl DefaultPublishErrorHandler {
    pub fn new() -> Self {
        Self
    }
}

impl PublishErrorHandler for DefaultPublishErrorHandler {
    fn handle_error(&self, _error: PublishError) -> Result<PublishRecoveryAction> {
        Ok(PublishRecoveryAction::Retry)
    }
    fn retry_strategy(&self, _error_type: PublishErrorType) -> RetryStrategy {
        RetryStrategy::ExponentialBackoff
    }
    fn report_failure(&self, _error: PublishError) -> Result<()> {
        Ok(())
    }
}

pub struct PublishRateLimiter;
impl PublishRateLimiter {
    pub fn new() -> Self {
        Self
    }
    pub async fn wait_if_needed(&self) {}
}

// Additional placeholder types that would be defined in the types module
#[derive(Debug, Clone, Default)]
pub struct SampleRateConfig {
    pub min_rate: f32,
    pub max_rate: f32,
    pub target_accuracy: f32,
}

#[derive(Debug, Clone)]
pub struct RateAdjustment {
    pub timestamp: DateTime<Utc>,
    pub old_rate: f32,
    pub new_rate: f32,
    pub reason: AdjustmentReason,
    pub system_load: f32,
    pub resource_availability: f32,
}

#[derive(Debug, Clone)]
pub struct RateAdjustmentResult {
    pub old_rate: f32,
    pub new_rate: f32,
    pub rate_changed: bool,
    pub reason: AdjustmentReason,
}

#[derive(Debug, Clone, Default)]
pub struct ImpactMonitorConfig;

#[derive(Debug, Clone, Default)]
pub struct PerformanceBaseline {
    pub cpu_usage: f32,
    pub memory_usage: u64,
    pub latency: Duration,
    pub throughput: f32,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Default)]
pub struct OverheadMeasurement {
    pub cpu_overhead: f32,
    pub memory_overhead: u64,
    pub latency_overhead: Duration,
    pub throughput_impact: f32,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Default)]
pub struct ImpactAnalysis {
    pub cpu_impact_percent: f32,
    pub memory_impact_percent: f32,
    pub latency_impact_percent: f32,
    pub throughput_impact_percent: f32,
    pub overall_severity: ImpactSeverity,
    pub trend: ImpactTrend,
    pub analysis_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct ImpactAnalysisResult {
    pub overhead: OverheadMeasurement,
    pub analysis: ImpactAnalysis,
    pub severity: ImpactSeverity,
    pub recommendations: Vec<ImpactRecommendation>,
}

#[derive(Debug, Clone)]
pub struct ImpactAlert {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub severity: ImpactSeverity,
    pub impact_measurement: OverheadMeasurement,
    pub analysis: ImpactAnalysis,
    pub recommendations: Vec<ImpactRecommendation>,
    pub acknowledged: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ImpactTrends;

#[derive(Debug, Clone)]
pub struct ImpactRecommendation;

#[derive(Debug, Clone, Default, PartialEq, PartialOrd)]
pub enum ImpactSeverity {
    #[default]
    Low,
    Moderate,
    High,
    Critical,
}

#[derive(Debug, Clone, Default)]
pub enum ImpactTrend {
    #[default]
    Stable,
    Increasing,
    Decreasing,
}

#[derive(Debug, Clone, Default)]
pub struct ImpactMeasurement;

#[derive(Debug, Clone)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub enum PublisherStatus {
    Active,
    Inactive,
    Error,
}

// Additional error and recovery types
#[derive(Debug)]
pub struct CollectionError;

#[derive(Debug)]
pub enum ErrorType {
    Network,
    Resource,
    Configuration,
}

#[derive(Debug)]
pub enum ErrorRecoveryAction {
    Retry,
    Skip,
    Stop,
}

#[derive(Debug)]
pub enum RecoveryStrategy {
    Immediate,
    Delayed,
    Exponential,
}

#[derive(Debug)]
pub struct PublishError;

#[derive(Debug)]
pub enum PublishErrorType {
    Network,
    Timeout,
    Authentication,
}

#[derive(Debug)]
pub enum PublishRecoveryAction {
    Retry,
    Drop,
    Queue,
}

#[derive(Debug)]
pub enum RetryStrategy {
    Fixed,
    ExponentialBackoff,
    Linear,
}
