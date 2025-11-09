use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

// Import commonly used types from core

// Import ResourceMonitorTrait from resources
use super::resources::ResourceMonitorTrait;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessRestriction {
    /// No restrictions
    None,
    /// Read only
    ReadOnly,
    /// Write only
    WriteOnly,
    /// Time limited
    TimeLimited,
    /// Count limited
    CountLimited,
    /// Size limited
    SizeLimited,
    /// Exclusive access required
    ExclusiveRequired,
    /// Permission based
    PermissionBased,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessType {
    /// Read-only access
    ReadOnly,
    /// Write-only access
    WriteOnly,
    /// Read-write access
    ReadWrite,
    /// Exclusive access
    Exclusive,
    /// Shared access
    Shared,
    /// Append access
    Append,
    /// Create access
    Create,
    /// Delete access
    Delete,
    /// Execute access
    Execute,
    /// Modify access
    Modify,
}

#[derive(Debug, Clone)]
pub struct AccessFrequency {
    pub access_count: usize,
    pub frequency_per_second: f64,
    pub last_access: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub pattern_type: String,
    pub access_sequence: Vec<String>,
    pub temporal_distribution: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct AccessPatternRequirements {
    pub required_access_type: AccessType,
    pub min_bandwidth: f64,
    pub max_latency: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct BandwidthUtilization {
    pub current_bandwidth: f64,
    pub max_bandwidth: f64,
    pub utilization_percentage: f64,
}

#[derive(Debug, Clone)]
pub struct ConnectionCharacteristics {
    pub connection_type: String,
    pub latency: std::time::Duration,
    pub bandwidth_mbps: f64,
    pub reliability: f64,
}

#[derive(Debug, Clone)]
pub struct ConnectionOverheadAnalysis {
    pub overhead_percentage: f64,
    pub connection_setup_time: std::time::Duration,
    pub teardown_time: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct DiskUsagePattern {
    pub read_write_ratio: f64,
    pub sequential_percentage: f64,
    pub random_percentage: f64,
    pub average_io_size: usize,
}

#[derive(Debug, Clone)]
pub struct FilesystemPerformanceMetrics {
    pub throughput_mbps: f64,
    pub iops: f64,
    pub average_latency: std::time::Duration,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Clone)]
pub struct IoBottleneck {
    pub bottleneck_type: String,
    pub severity: f64,
    pub affected_operations: Vec<String>,
    pub mitigation_suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IoLatencyAnalysisResults {
    pub average_latency: std::time::Duration,
    pub p50_latency: std::time::Duration,
    pub p95_latency: std::time::Duration,
    pub p99_latency: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct IoLatencyAnalyzer {
    pub analysis_enabled: bool,
    pub sample_size: usize,
    pub latency_threshold: std::time::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoMetrics {
    /// Read operations per second
    pub read_ops_per_sec: f64,
    /// Write operations per second
    pub write_ops_per_sec: f64,
    /// Read throughput (bytes/sec)
    pub read_throughput: f64,
    /// Write throughput (bytes/sec)
    pub write_throughput: f64,
    /// Average read latency
    pub avg_read_latency: Duration,
    /// Average write latency
    pub avg_write_latency: Duration,
    /// I/O queue depth
    pub queue_depth: f64,
    /// I/O utilization
    pub utilization: f64,
    /// I/O wait time
    pub wait_time: f64,
    /// Error rate
    pub error_rate: f64,
}

#[derive(Debug, Clone)]
pub struct IoMonitor {
    /// Bytes read
    pub bytes_read: u64,
    /// Bytes written
    pub bytes_written: u64,
    /// I/O operations per second
    pub iops: f64,
}

#[derive(Debug, Clone)]
pub struct IoOperation {
    /// Operation timestamp
    pub timestamp: Instant,
    /// I/O operation type
    pub operation_type: String,
    /// File or resource path
    pub path: String,
    /// Data size
    pub size: usize,
    /// Operation duration
    pub duration: Duration,
    /// Thread ID
    pub thread_id: u64,
    /// Success status
    pub success: bool,
    /// Performance metrics
    pub performance_metrics: IoMetrics,
    /// Resource contention
    pub contention: f64,
    /// Optimization opportunities
    pub optimizations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IoPattern {
    pub pattern_type: String,
    pub sequential_ratio: f64,
    pub random_ratio: f64,
    pub average_request_size: usize,
}

#[derive(Debug, Clone)]
pub struct IoPatternAnalysisResults {
    pub detected_patterns: Vec<String>,
    pub dominant_pattern: String,
    pub pattern_confidence: f64,
    pub optimization_suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IoPatternAnalyzer {
    pub analysis_window: std::time::Duration,
    pub min_samples: usize,
    pub pattern_detection_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct IoProfilingState {
    pub profiling_active: bool,
    pub operations_captured: usize,
    pub start_time: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct LatencyBounds {
    pub min_latency: std::time::Duration,
    pub max_latency: std::time::Duration,
    pub average_latency: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct LatencyCharacteristics {
    pub baseline_latency: std::time::Duration,
    pub jitter: std::time::Duration,
    pub percentile_99: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct LatencyProcessorConfig {
    pub processing_enabled: bool,
    pub sample_rate: f64,
    pub outlier_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct MtuOptimizationResults {
    pub optimal_mtu: usize,
    pub performance_gain: f64,
    pub tested_mtu_values: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct MtuOptimizer {
    pub optimization_enabled: bool,
    pub test_mtu_range: (usize, usize),
    pub current_optimal_mtu: usize,
}

#[derive(Debug, Clone)]
pub struct NetworkBandwidthAnalysis {
    pub available_bandwidth: f64,
    pub utilized_bandwidth: f64,
    pub bandwidth_efficiency: f64,
    pub bottleneck_identified: bool,
}

#[derive(Debug, Clone)]
pub struct NetworkBandwidthTester {
    pub test_duration: std::time::Duration,
    pub test_packet_size: usize,
    pub results: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct NetworkEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Event type (connect, send, receive, close)
    pub event_type: String,
    /// Remote endpoint
    pub endpoint: String,
    /// Data size
    pub data_size: usize,
    /// Event duration
    pub duration: Duration,
    /// Thread ID
    pub thread_id: u64,
    /// Success status
    pub success: bool,
    /// Network metrics
    pub metrics: NetworkMetrics,
    /// Performance impact
    pub performance_impact: f64,
    /// Reliability factors
    pub reliability_factors: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct NetworkInterface {
    pub interface_name: String,
    pub interface_type: String,
    pub bandwidth_mbps: f64,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct NetworkInterfaceComparisonResults {
    pub interfaces_analyzed: Vec<String>,
    pub optimal_interface: String,
    pub performance_comparison: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct NetworkInterfaceAnalyzer {
    pub analysis_enabled: bool,
    pub interfaces_to_analyze: Vec<String>,
    pub analysis_interval: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct NetworkLatencyAnalysis {
    pub average_latency: std::time::Duration,
    pub jitter: std::time::Duration,
    pub packet_loss_rate: f64,
}

#[derive(Debug, Clone)]
pub struct NetworkLatencyProfile {
    pub baseline_latency: std::time::Duration,
    pub latency_distribution: Vec<std::time::Duration>,
    pub outlier_threshold: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct NetworkLatencyTester {
    pub test_enabled: bool,
    pub test_endpoints: Vec<String>,
    pub test_frequency: std::time::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Bytes received per second
    pub bytes_received_per_sec: f64,
    /// Bytes sent per second
    pub bytes_sent_per_sec: f64,
    /// Packets received per second
    pub packets_received_per_sec: f64,
    /// Packets sent per second
    pub packets_sent_per_sec: f64,
    /// Network latency
    pub latency: Duration,
    /// Connection count
    pub connection_count: usize,
    /// Bandwidth utilization
    pub bandwidth_utilization: f64,
    /// Error rate
    pub error_rate: f64,
    /// Retransmission rate
    pub retransmission_rate: f64,
    /// Connection quality
    pub connection_quality: f64,
}

#[derive(Debug, Clone)]
pub struct NetworkMonitor {
    /// Bytes received
    pub bytes_received: u64,
    /// Bytes sent
    pub bytes_sent: u64,
    /// Network throughput (bytes/sec)
    pub throughput: f64,
}

#[derive(Debug, Clone)]
pub struct NetworkPattern {
    pub pattern_type: String,
    pub traffic_characteristics: HashMap<String, f64>,
    pub burst_frequency: f64,
}

#[derive(Debug, Clone)]
pub struct NetworkProfilingState {
    pub profiling_active: bool,
    pub network_events_captured: usize,
    pub profiling_start_time: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct NetworkReliability {
    pub connection_stability: f64,
    pub packet_loss_rate: f64,
    pub retransmission_rate: f64,
    pub uptime_percentage: f64,
}

#[derive(Debug, Clone)]
pub struct PacketLossCharacteristics {
    pub loss_rate: f64,
    pub loss_pattern: String,
    pub recovery_time: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct PrefetcherAnalysis {
    pub prefetch_enabled: bool,
    pub prefetch_hit_rate: f64,
    pub prefetch_accuracy: f64,
    pub performance_improvement: f64,
}

#[derive(Debug, Clone)]
pub struct ProtocolPerformanceAnalysis {
    pub protocol: String,
    pub throughput: f64,
    pub latency: std::time::Duration,
    pub efficiency: f64,
}

#[derive(Debug, Clone)]
pub struct ProtocolPerformanceMetrics {
    pub protocol_name: String,
    pub request_rate: f64,
    pub response_time: std::time::Duration,
    pub error_rate: f64,
}

#[derive(Debug, Clone)]
pub struct QueueDepthOptimizationResults {
    pub optimal_queue_depth: usize,
    pub performance_gain: f64,
    pub tested_depths: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct QueueDepthOptimizer {
    pub optimization_enabled: bool,
    pub depth_range: (usize, usize),
    pub current_optimal_depth: usize,
}

#[derive(Debug, Clone)]
pub struct RandomIoPerformance {
    pub iops: f64,
    pub average_latency: std::time::Duration,
    pub throughput_mbps: f64,
}

#[derive(Debug, Clone)]
pub struct SequentialIoPerformance {
    pub throughput_mbps: f64,
    pub average_latency: std::time::Duration,
    pub sustained_rate: f64,
}

// Trait implementations

// Struct implementations

impl IoMonitor {
    /// Create a new IoMonitor with default settings
    pub fn new() -> Self {
        Self {
            bytes_read: 0,
            bytes_written: 0,
            iops: 0.0,
        }
    }
}

impl Default for IoMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkMonitor {
    /// Create a new NetworkMonitor with default settings
    pub fn new() -> Self {
        Self {
            bytes_received: 0,
            bytes_sent: 0,
            throughput: 0.0,
        }
    }
}

impl Default for NetworkMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// Implement ResourceMonitorTrait for IoMonitor
impl ResourceMonitorTrait for IoMonitor {
    fn monitor(&self) -> String {
        format!(
            "IO Monitor: {} read, {} written, {} IOPS",
            self.bytes_read, self.bytes_written, self.iops
        )
    }
}

// Implement ResourceMonitorTrait for NetworkMonitor
impl ResourceMonitorTrait for NetworkMonitor {
    fn monitor(&self) -> String {
        format!(
            "Network Monitor: {} received, {} sent, {} bytes/sec throughput",
            self.bytes_received, self.bytes_sent, self.throughput
        )
    }
}
