//! Enhanced Distributed Training Profiling
//!
//! This module provides comprehensive profiling support for distributed training scenarios,
//! including multi-node coordination, gradient synchronization analysis, and communication
//! pattern optimization.

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Distributed training profiler
///
/// Provides advanced profiling capabilities for distributed training including:
/// - Cross-node communication analysis
/// - Gradient synchronization profiling
/// - Load balancing metrics
/// - Communication bottleneck detection
#[derive(Debug)]
pub struct DistributedProfiler {
    /// Configuration
    config: DistributedProfilerConfig,
    /// Node metadata
    nodes: Arc<RwLock<HashMap<String, NodeInfo>>>,
    /// Communication events
    comm_events: Arc<RwLock<Vec<CommunicationEvent>>>,
    /// Synchronization events
    sync_events: Arc<RwLock<Vec<SynchronizationEvent>>>,
    /// Performance snapshots per node
    node_snapshots: Arc<RwLock<HashMap<String, Vec<NodePerformanceSnapshot>>>>,
    /// Start time for profiling session
    start_time: Instant,
}

/// Configuration for distributed profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedProfilerConfig {
    /// Enable communication profiling
    pub enable_comm_profiling: bool,
    /// Enable gradient sync profiling
    pub enable_grad_sync_profiling: bool,
    /// Enable load balance profiling
    pub enable_load_balance_profiling: bool,
    /// Enable network bandwidth analysis
    pub enable_bandwidth_analysis: bool,
    /// Sampling interval (milliseconds)
    pub sampling_interval_ms: u64,
    /// Maximum events to store per category
    pub max_events_per_category: usize,
    /// Enable automatic bottleneck detection
    pub enable_bottleneck_detection: bool,
    /// Bottleneck threshold (percentage)
    pub bottleneck_threshold_pct: f64,
}

impl Default for DistributedProfilerConfig {
    fn default() -> Self {
        Self {
            enable_comm_profiling: true,
            enable_grad_sync_profiling: true,
            enable_load_balance_profiling: true,
            enable_bandwidth_analysis: true,
            sampling_interval_ms: 100,
            max_events_per_category: 10000,
            enable_bottleneck_detection: true,
            bottleneck_threshold_pct: 80.0,
        }
    }
}

/// Information about a node in the distributed cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Node ID (unique identifier)
    pub node_id: String,
    /// Rank in distributed training
    pub rank: usize,
    /// World size (total number of nodes)
    pub world_size: usize,
    /// Node hostname/IP
    pub host: String,
    /// GPU devices on this node
    pub gpu_count: usize,
    /// Node role (master, worker, etc.)
    pub role: NodeRole,
    /// Node status
    pub status: NodeStatus,
}

/// Node role in distributed training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Master/coordinator node
    Master,
    /// Worker node
    Worker,
    /// Parameter server
    ParameterServer,
    /// Hybrid role
    Hybrid,
}

/// Node status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is active and healthy
    Active,
    /// Node is idle
    Idle,
    /// Node has a warning
    Warning,
    /// Node has failed
    Failed,
    /// Node is disconnected
    Disconnected,
}

/// Communication event between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationEvent {
    /// Event ID
    pub event_id: usize,
    /// Timestamp
    pub timestamp: Duration,
    /// Source node
    pub source_node: String,
    /// Destination node
    pub dest_node: String,
    /// Communication type
    pub comm_type: CommunicationType,
    /// Data size (bytes)
    pub data_size_bytes: usize,
    /// Duration (milliseconds)
    pub duration_ms: f64,
    /// Bandwidth (MB/s)
    pub bandwidth_mbps: f64,
}

/// Type of communication between nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommunicationType {
    /// Point-to-point send
    Send,
    /// Point-to-point receive
    Recv,
    /// All-reduce operation
    AllReduce,
    /// All-gather operation
    AllGather,
    /// Broadcast operation
    Broadcast,
    /// Scatter operation
    Scatter,
    /// Reduce operation
    Reduce,
    /// Barrier synchronization
    Barrier,
}

/// Gradient synchronization event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationEvent {
    /// Event ID
    pub event_id: usize,
    /// Timestamp
    pub timestamp: Duration,
    /// Participating nodes
    pub nodes: Vec<String>,
    /// Synchronization type
    pub sync_type: SyncType,
    /// Total gradient size (bytes)
    pub gradient_size_bytes: usize,
    /// Synchronization duration (milliseconds)
    pub duration_ms: f64,
    /// Success status
    pub success: bool,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Type of gradient synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncType {
    /// Data-parallel all-reduce
    DataParallel,
    /// Model-parallel send/recv
    ModelParallel,
    /// Pipeline-parallel forward
    PipelineForward,
    /// Pipeline-parallel backward
    PipelineBackward,
    /// Hybrid parallel
    Hybrid,
}

/// Performance snapshot for a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePerformanceSnapshot {
    /// Timestamp
    pub timestamp: Duration,
    /// Node ID
    pub node_id: String,
    /// Compute utilization (0-100)
    pub compute_utilization_pct: f64,
    /// Memory utilization (0-100)
    pub memory_utilization_pct: f64,
    /// Network utilization (0-100)
    pub network_utilization_pct: f64,
    /// Throughput (samples/sec)
    pub throughput: f64,
    /// Active communication count
    pub active_communications: usize,
    /// Pending operations
    pub pending_operations: usize,
}

/// Distributed profiling report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedProfilingReport {
    /// Total profiling duration
    pub total_duration_secs: f64,
    /// Number of nodes profiled
    pub num_nodes: usize,
    /// Communication summary
    pub communication_summary: CommunicationSummary,
    /// Synchronization summary
    pub synchronization_summary: SynchronizationSummary,
    /// Load balance analysis
    pub load_balance: LoadBalanceAnalysis,
    /// Detected bottlenecks
    pub bottlenecks: Vec<Bottleneck>,
    /// Performance recommendations
    pub recommendations: Vec<String>,
}

/// Summary of communication patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationSummary {
    /// Total communication events
    pub total_events: usize,
    /// Total data transferred (bytes)
    pub total_data_bytes: usize,
    /// Average bandwidth (MB/s)
    pub avg_bandwidth_mbps: f64,
    /// Peak bandwidth (MB/s)
    pub peak_bandwidth_mbps: f64,
    /// Communication overhead (percentage of total time)
    pub overhead_pct: f64,
    /// Most common communication type
    pub most_common_type: Option<CommunicationType>,
    /// Slowest communication
    pub slowest_comm: Option<CommunicationEvent>,
}

/// Summary of synchronization operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationSummary {
    /// Total synchronization events
    pub total_syncs: usize,
    /// Successful syncs
    pub successful_syncs: usize,
    /// Failed syncs
    pub failed_syncs: usize,
    /// Average sync duration (milliseconds)
    pub avg_sync_duration_ms: f64,
    /// Maximum sync duration (milliseconds)
    pub max_sync_duration_ms: f64,
    /// Total time in synchronization (seconds)
    pub total_sync_time_secs: f64,
    /// Synchronization efficiency (0-1)
    pub sync_efficiency: f64,
}

/// Load balance analysis across nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalanceAnalysis {
    /// Load imbalance score (0-1, lower is better)
    pub imbalance_score: f64,
    /// Compute utilization per node
    pub compute_utilization: HashMap<String, f64>,
    /// Memory utilization per node
    pub memory_utilization: HashMap<String, f64>,
    /// Throughput per node
    pub throughput: HashMap<String, f64>,
    /// Straggler nodes (slowest nodes)
    pub stragglers: Vec<String>,
    /// Idle time per node (seconds)
    pub idle_time: HashMap<String, f64>,
}

/// Detected performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity (0-100)
    pub severity: f64,
    /// Affected nodes
    pub affected_nodes: Vec<String>,
    /// Description
    pub description: String,
    /// Suggested fix
    pub suggestion: String,
}

/// Type of performance bottleneck
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    /// Communication bottleneck
    Communication,
    /// Synchronization bottleneck
    Synchronization,
    /// Compute imbalance
    ComputeImbalance,
    /// Memory bottleneck
    Memory,
    /// Network congestion
    NetworkCongestion,
    /// Straggler node
    Straggler,
}

impl DistributedProfiler {
    /// Create a new distributed profiler
    ///
    /// # Arguments
    /// * `config` - Profiler configuration
    ///
    /// # Example
    /// ```rust
    /// use trustformers_debug::{DistributedProfiler, DistributedProfilerConfig};
    ///
    /// let config = DistributedProfilerConfig::default();
    /// let profiler = DistributedProfiler::new(config);
    /// ```
    pub fn new(config: DistributedProfilerConfig) -> Self {
        info!("Initializing distributed profiler");
        Self {
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            comm_events: Arc::new(RwLock::new(Vec::new())),
            sync_events: Arc::new(RwLock::new(Vec::new())),
            node_snapshots: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        }
    }

    /// Register a node in the cluster
    ///
    /// # Arguments
    /// * `node_info` - Information about the node
    pub fn register_node(&self, node_info: NodeInfo) -> Result<()> {
        debug!(
            "Registering node: {} (rank {})",
            node_info.node_id, node_info.rank
        );

        let mut nodes = self.nodes.write();
        nodes.insert(node_info.node_id.clone(), node_info);

        Ok(())
    }

    /// Record a communication event
    ///
    /// # Arguments
    /// * `event` - Communication event to record
    pub fn record_communication(&self, event: CommunicationEvent) -> Result<()> {
        if !self.config.enable_comm_profiling {
            return Ok(());
        }

        let mut events = self.comm_events.write();

        // Limit stored events
        if events.len() >= self.config.max_events_per_category {
            events.remove(0);
        }

        events.push(event);
        Ok(())
    }

    /// Record a synchronization event
    ///
    /// # Arguments
    /// * `event` - Synchronization event to record
    pub fn record_synchronization(&self, event: SynchronizationEvent) -> Result<()> {
        if !self.config.enable_grad_sync_profiling {
            return Ok(());
        }

        let mut events = self.sync_events.write();

        // Limit stored events
        if events.len() >= self.config.max_events_per_category {
            events.remove(0);
        }

        events.push(event);
        Ok(())
    }

    /// Record a performance snapshot for a node
    ///
    /// # Arguments
    /// * `snapshot` - Performance snapshot
    pub fn record_snapshot(&self, snapshot: NodePerformanceSnapshot) -> Result<()> {
        let mut snapshots = self.node_snapshots.write();

        let node_history = snapshots.entry(snapshot.node_id.clone()).or_insert_with(Vec::new);

        // Limit stored snapshots
        if node_history.len() >= self.config.max_events_per_category {
            node_history.remove(0);
        }

        node_history.push(snapshot);
        Ok(())
    }

    /// Generate a comprehensive profiling report
    ///
    /// # Returns
    /// Detailed profiling report with analysis and recommendations
    pub fn generate_report(&self) -> Result<DistributedProfilingReport> {
        info!("Generating distributed profiling report");

        let total_duration = self.start_time.elapsed().as_secs_f64();
        let nodes = self.nodes.read();
        let num_nodes = nodes.len();

        // Analyze communication patterns
        let communication_summary = self.analyze_communication()?;

        // Analyze synchronization
        let synchronization_summary = self.analyze_synchronization()?;

        // Analyze load balance
        let load_balance = self.analyze_load_balance()?;

        // Detect bottlenecks
        let bottlenecks = if self.config.enable_bottleneck_detection {
            self.detect_bottlenecks(
                &communication_summary,
                &synchronization_summary,
                &load_balance,
            )?
        } else {
            Vec::new()
        };

        // Generate recommendations
        let recommendations = self.generate_recommendations(&bottlenecks, &load_balance)?;

        Ok(DistributedProfilingReport {
            total_duration_secs: total_duration,
            num_nodes,
            communication_summary,
            synchronization_summary,
            load_balance,
            bottlenecks,
            recommendations,
        })
    }

    /// Analyze communication patterns
    fn analyze_communication(&self) -> Result<CommunicationSummary> {
        let events = self.comm_events.read();

        if events.is_empty() {
            return Ok(CommunicationSummary {
                total_events: 0,
                total_data_bytes: 0,
                avg_bandwidth_mbps: 0.0,
                peak_bandwidth_mbps: 0.0,
                overhead_pct: 0.0,
                most_common_type: None,
                slowest_comm: None,
            });
        }

        let total_events = events.len();
        let total_data_bytes: usize = events.iter().map(|e| e.data_size_bytes).sum();

        let bandwidths: Vec<f64> = events.iter().map(|e| e.bandwidth_mbps).collect();
        let avg_bandwidth_mbps = bandwidths.iter().sum::<f64>() / bandwidths.len() as f64;
        let peak_bandwidth_mbps = bandwidths.iter().fold(0.0f64, |a, &b| a.max(b));

        let total_comm_time: f64 = events.iter().map(|e| e.duration_ms).sum();
        let overhead_pct =
            (total_comm_time / 1000.0) / self.start_time.elapsed().as_secs_f64() * 100.0;

        // Find most common type
        let mut type_counts: HashMap<CommunicationType, usize> = HashMap::new();
        for event in events.iter() {
            *type_counts.entry(event.comm_type).or_insert(0) += 1;
        }
        let most_common_type =
            type_counts.iter().max_by_key(|(_, count)| *count).map(|(typ, _)| *typ);

        // Find slowest communication
        let slowest_comm = events
            .iter()
            .max_by(|a, b| a.duration_ms.partial_cmp(&b.duration_ms).unwrap())
            .cloned();

        Ok(CommunicationSummary {
            total_events,
            total_data_bytes,
            avg_bandwidth_mbps,
            peak_bandwidth_mbps,
            overhead_pct,
            most_common_type,
            slowest_comm,
        })
    }

    /// Analyze synchronization operations
    fn analyze_synchronization(&self) -> Result<SynchronizationSummary> {
        let events = self.sync_events.read();

        if events.is_empty() {
            return Ok(SynchronizationSummary {
                total_syncs: 0,
                successful_syncs: 0,
                failed_syncs: 0,
                avg_sync_duration_ms: 0.0,
                max_sync_duration_ms: 0.0,
                total_sync_time_secs: 0.0,
                sync_efficiency: 1.0,
            });
        }

        let total_syncs = events.len();
        let successful_syncs = events.iter().filter(|e| e.success).count();
        let failed_syncs = total_syncs - successful_syncs;

        let durations: Vec<f64> = events.iter().map(|e| e.duration_ms).collect();
        let avg_sync_duration_ms = durations.iter().sum::<f64>() / durations.len() as f64;
        let max_sync_duration_ms = durations.iter().fold(0.0f64, |a, &b| a.max(b));
        let total_sync_time_secs = durations.iter().sum::<f64>() / 1000.0;

        // Calculate efficiency (theoretical min time / actual time)
        let theoretical_min = events.iter()
            .map(|e| e.gradient_size_bytes as f64 / 1_000_000.0) // Convert to MB
            .sum::<f64>()
            / 10.0; // Assume 10 MB/s ideal bandwidth
        let sync_efficiency = (theoretical_min / total_sync_time_secs).min(1.0);

        Ok(SynchronizationSummary {
            total_syncs,
            successful_syncs,
            failed_syncs,
            avg_sync_duration_ms,
            max_sync_duration_ms,
            total_sync_time_secs,
            sync_efficiency,
        })
    }

    /// Analyze load balance across nodes
    fn analyze_load_balance(&self) -> Result<LoadBalanceAnalysis> {
        let snapshots = self.node_snapshots.read();

        if snapshots.is_empty() {
            return Ok(LoadBalanceAnalysis {
                imbalance_score: 0.0,
                compute_utilization: HashMap::new(),
                memory_utilization: HashMap::new(),
                throughput: HashMap::new(),
                stragglers: Vec::new(),
                idle_time: HashMap::new(),
            });
        }

        let mut compute_utilization = HashMap::new();
        let mut memory_utilization = HashMap::new();
        let mut throughput = HashMap::new();
        let mut idle_time = HashMap::new();

        // Calculate averages per node
        for (node_id, node_snapshots) in snapshots.iter() {
            if node_snapshots.is_empty() {
                continue;
            }

            let avg_compute = node_snapshots.iter().map(|s| s.compute_utilization_pct).sum::<f64>()
                / node_snapshots.len() as f64;

            let avg_memory = node_snapshots.iter().map(|s| s.memory_utilization_pct).sum::<f64>()
                / node_snapshots.len() as f64;

            let avg_throughput = node_snapshots.iter().map(|s| s.throughput).sum::<f64>()
                / node_snapshots.len() as f64;

            // Calculate idle time (when compute utilization < 10%)
            let idle_samples =
                node_snapshots.iter().filter(|s| s.compute_utilization_pct < 10.0).count();
            let idle_secs =
                idle_samples as f64 * (self.config.sampling_interval_ms as f64 / 1000.0);

            compute_utilization.insert(node_id.clone(), avg_compute);
            memory_utilization.insert(node_id.clone(), avg_memory);
            throughput.insert(node_id.clone(), avg_throughput);
            idle_time.insert(node_id.clone(), idle_secs);
        }

        // Calculate imbalance score (coefficient of variation of throughput)
        let throughput_values: Vec<f64> = throughput.values().copied().collect();
        let imbalance_score = if !throughput_values.is_empty() {
            let mean = throughput_values.iter().sum::<f64>() / throughput_values.len() as f64;
            let variance = throughput_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / throughput_values.len() as f64;
            let std_dev = variance.sqrt();
            std_dev / mean
        } else {
            0.0
        };

        // Identify stragglers (nodes with significantly lower throughput)
        let mean_throughput =
            throughput_values.iter().sum::<f64>() / throughput_values.len().max(1) as f64;
        let stragglers: Vec<String> = throughput.iter()
            .filter(|(_, &t)| t < mean_throughput * 0.7) // 30% below average
            .map(|(node_id, _)| node_id.clone())
            .collect();

        Ok(LoadBalanceAnalysis {
            imbalance_score,
            compute_utilization,
            memory_utilization,
            throughput,
            stragglers,
            idle_time,
        })
    }

    /// Detect performance bottlenecks
    fn detect_bottlenecks(
        &self,
        comm_summary: &CommunicationSummary,
        sync_summary: &SynchronizationSummary,
        load_balance: &LoadBalanceAnalysis,
    ) -> Result<Vec<Bottleneck>> {
        let mut bottlenecks = Vec::new();

        // Check for communication bottleneck
        if comm_summary.overhead_pct > self.config.bottleneck_threshold_pct {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::Communication,
                severity: comm_summary.overhead_pct,
                affected_nodes: vec!["all".to_string()],
                description: format!(
                    "Communication overhead is {:.1}%, significantly impacting performance",
                    comm_summary.overhead_pct
                ),
                suggestion: "Consider reducing communication frequency, increasing batch size, or using gradient compression".to_string(),
            });
        }

        // Check for synchronization bottleneck
        if sync_summary.sync_efficiency < 0.5 {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::Synchronization,
                severity: (1.0 - sync_summary.sync_efficiency) * 100.0,
                affected_nodes: vec!["all".to_string()],
                description: format!(
                    "Synchronization efficiency is only {:.1}%, indicating significant overhead",
                    sync_summary.sync_efficiency * 100.0
                ),
                suggestion: "Use gradient accumulation, optimize all-reduce operations, or consider hierarchical synchronization".to_string(),
            });
        }

        // Check for load imbalance
        if load_balance.imbalance_score > 0.3 {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::ComputeImbalance,
                severity: load_balance.imbalance_score * 100.0,
                affected_nodes: load_balance.stragglers.clone(),
                description: format!(
                    "High load imbalance detected (score: {:.2}), {} straggler node(s)",
                    load_balance.imbalance_score,
                    load_balance.stragglers.len()
                ),
                suggestion: "Balance data distribution, check for hardware heterogeneity, or implement dynamic load balancing".to_string(),
            });
        }

        // Check for straggler nodes
        for straggler in &load_balance.stragglers {
            if let Some(&idle_time) = load_balance.idle_time.get(straggler) {
                if idle_time > 5.0 {
                    // More than 5 seconds idle
                    bottlenecks.push(Bottleneck {
                        bottleneck_type: BottleneckType::Straggler,
                        severity: 75.0,
                        affected_nodes: vec![straggler.clone()],
                        description: format!(
                            "Node {} is a straggler with {:.1}s idle time",
                            straggler, idle_time
                        ),
                        suggestion: format!(
                            "Investigate node {} for hardware issues, resource contention, or network problems",
                            straggler
                        ),
                    });
                }
            }
        }

        Ok(bottlenecks)
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        bottlenecks: &[Bottleneck],
        load_balance: &LoadBalanceAnalysis,
    ) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        // General recommendations based on bottlenecks
        for bottleneck in bottlenecks {
            if bottleneck.severity > 50.0 {
                recommendations.push(format!(
                    "[HIGH PRIORITY] {}: {}",
                    match bottleneck.bottleneck_type {
                        BottleneckType::Communication => "Communication Bottleneck",
                        BottleneckType::Synchronization => "Synchronization Bottleneck",
                        BottleneckType::ComputeImbalance => "Load Imbalance",
                        BottleneckType::Memory => "Memory Bottleneck",
                        BottleneckType::NetworkCongestion => "Network Congestion",
                        BottleneckType::Straggler => "Straggler Node",
                    },
                    bottleneck.suggestion
                ));
            }
        }

        // Load balance recommendations
        if load_balance.imbalance_score > 0.2 {
            recommendations.push(
                "Consider implementing dynamic batch size adjustment per node based on compute capability".to_string()
            );
        }

        // Check for underutilized nodes
        let underutilized: Vec<_> = load_balance
            .compute_utilization
            .iter()
            .filter(|(_, &util)| util < 50.0)
            .collect();

        if !underutilized.is_empty() {
            recommendations.push(format!(
                "{} node(s) are underutilized (<50% compute). Consider increasing batch size or model complexity",
                underutilized.len()
            ));
        }

        // If no specific recommendations, add general ones
        if recommendations.is_empty() {
            recommendations.push(
                "Performance looks good! Continue monitoring for any degradation".to_string(),
            );
            recommendations.push(
                "Consider enabling gradient compression to reduce communication overhead"
                    .to_string(),
            );
            recommendations
                .push("Experiment with mixed-precision training for better throughput".to_string());
        }

        Ok(recommendations)
    }

    /// Export profiling data to JSON
    ///
    /// # Arguments
    /// * `path` - Output file path
    pub fn export_json(&self, path: &std::path::Path) -> Result<()> {
        let report = self.generate_report()?;
        let json =
            serde_json::to_string_pretty(&report).context("Failed to serialize report to JSON")?;
        std::fs::write(path, json).context("Failed to write JSON file")?;
        info!("Exported profiling report to {}", path.display());
        Ok(())
    }

    /// Get real-time statistics (for dashboards)
    ///
    /// # Returns
    /// Current profiling statistics
    pub fn get_realtime_stats(&self) -> Result<RealtimeStats> {
        let nodes = self.nodes.read();
        let comm_events = self.comm_events.read();
        let sync_events = self.sync_events.read();

        // Calculate recent metrics (last 10 seconds)
        let recent_cutoff = self.start_time.elapsed().saturating_sub(Duration::from_secs(10));

        let recent_comm_count = comm_events.iter().filter(|e| e.timestamp >= recent_cutoff).count();

        let recent_sync_count = sync_events.iter().filter(|e| e.timestamp >= recent_cutoff).count();

        let active_nodes = nodes.values().filter(|n| n.status == NodeStatus::Active).count();

        Ok(RealtimeStats {
            active_nodes,
            total_nodes: nodes.len(),
            recent_communications: recent_comm_count,
            recent_synchronizations: recent_sync_count,
            elapsed_time_secs: self.start_time.elapsed().as_secs_f64(),
        })
    }
}

/// Real-time statistics for dashboards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeStats {
    /// Number of active nodes
    pub active_nodes: usize,
    /// Total number of nodes
    pub total_nodes: usize,
    /// Recent communication events (last 10s)
    pub recent_communications: usize,
    /// Recent synchronization events (last 10s)
    pub recent_synchronizations: usize,
    /// Elapsed time since profiling started
    pub elapsed_time_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let config = DistributedProfilerConfig::default();
        let _profiler = DistributedProfiler::new(config);
    }

    #[test]
    fn test_node_registration() -> Result<()> {
        let config = DistributedProfilerConfig::default();
        let profiler = DistributedProfiler::new(config);

        let node = NodeInfo {
            node_id: "node-0".to_string(),
            rank: 0,
            world_size: 4,
            host: "localhost".to_string(),
            gpu_count: 1,
            role: NodeRole::Master,
            status: NodeStatus::Active,
        };

        profiler.register_node(node)?;

        let nodes = profiler.nodes.read();
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains_key("node-0"));

        Ok(())
    }

    #[test]
    fn test_communication_recording() -> Result<()> {
        let config = DistributedProfilerConfig::default();
        let profiler = DistributedProfiler::new(config);

        let event = CommunicationEvent {
            event_id: 0,
            timestamp: Duration::from_millis(100),
            source_node: "node-0".to_string(),
            dest_node: "node-1".to_string(),
            comm_type: CommunicationType::AllReduce,
            data_size_bytes: 1024 * 1024,
            duration_ms: 10.5,
            bandwidth_mbps: 95.0,
        };

        profiler.record_communication(event)?;

        let events = profiler.comm_events.read();
        assert_eq!(events.len(), 1);

        Ok(())
    }

    #[test]
    fn test_report_generation() -> Result<()> {
        let config = DistributedProfilerConfig::default();
        let profiler = DistributedProfiler::new(config);

        // Register nodes
        for i in 0..4 {
            let node = NodeInfo {
                node_id: format!("node-{}", i),
                rank: i,
                world_size: 4,
                host: "localhost".to_string(),
                gpu_count: 1,
                role: if i == 0 { NodeRole::Master } else { NodeRole::Worker },
                status: NodeStatus::Active,
            };
            profiler.register_node(node)?;
        }

        // Record some events
        for i in 0..10 {
            let event = CommunicationEvent {
                event_id: i,
                timestamp: Duration::from_millis(i as u64 * 100),
                source_node: format!("node-{}", i % 4),
                dest_node: format!("node-{}", (i + 1) % 4),
                comm_type: CommunicationType::AllReduce,
                data_size_bytes: 1024 * 1024,
                duration_ms: 10.0 + (i as f64 * 0.5),
                bandwidth_mbps: 100.0 - (i as f64 * 2.0),
            };
            profiler.record_communication(event)?;
        }

        let report = profiler.generate_report()?;

        assert_eq!(report.num_nodes, 4);
        assert_eq!(report.communication_summary.total_events, 10);
        assert!(report.communication_summary.avg_bandwidth_mbps > 0.0);

        Ok(())
    }
}
