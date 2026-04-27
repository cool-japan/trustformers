//! I/O operation monitoring and bandwidth tracking

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

/// I/O operation profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoProfile {
    pub operation_type: IoOperationType,
    pub file_path: Option<String>,
    pub bytes_transferred: usize,
    pub duration: Duration,
    pub bandwidth_mb_s: f64,
    pub queue_time: Duration,
    pub device_type: IoDeviceType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IoOperationType {
    FileRead,
    FileWrite,
    NetworkRead,
    NetworkWrite,
    DatabaseQuery,
    CacheLoad,
    CacheStore,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum IoDeviceType {
    SSD,
    HDD,
    Network,
    Memory,
    Cache,
}

/// Layer-wise latency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerLatencyProfile {
    pub layer_name: String,
    pub layer_type: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub cpu_time: Duration,
    pub gpu_time: Duration,
    pub memory_copy_time: Duration,
    pub sync_time: Duration,
    pub parameter_count: usize,
    pub flops: u64,
    pub memory_footprint_bytes: usize,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IoPerformanceSummary {
    pub total_operations: usize,
    pub total_bytes_transferred: usize,
    pub avg_bandwidth_by_device: HashMap<IoDeviceType, f64>,
    pub slowest_operations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthSample {
    pub timestamp: SystemTime,
    pub bandwidth_mb_s: f64,
    pub device_type: IoDeviceType,
}

/// I/O operation monitor
#[derive(Debug)]
pub struct IoMonitor {
    pub(crate) active_operations: HashMap<Uuid, IoOperation>,
    pub(crate) bandwidth_history: Vec<BandwidthSample>,
    pub(crate) io_queue_depth: usize,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct IoOperation {
    #[allow(dead_code)]
    pub(crate) operation_id: Uuid,
    pub(crate) start_time: Instant,
    pub(crate) operation_type: IoOperationType,
    pub(crate) bytes_expected: usize,
}

impl Default for IoMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl IoMonitor {
    pub fn new() -> Self {
        Self {
            active_operations: HashMap::new(),
            bandwidth_history: Vec::new(),
            io_queue_depth: 0,
        }
    }

    pub fn start_io_operation(
        &mut self,
        operation_type: IoOperationType,
        bytes_expected: usize,
    ) -> Uuid {
        let operation_id = Uuid::new_v4();
        let operation = IoOperation {
            operation_id,
            start_time: Instant::now(),
            operation_type,
            bytes_expected,
        };

        self.active_operations.insert(operation_id, operation);
        self.io_queue_depth += 1;
        operation_id
    }

    pub fn finish_io_operation(
        &mut self,
        operation_id: Uuid,
        bytes_transferred: usize,
    ) -> Option<IoProfile> {
        if let Some(operation) = self.active_operations.remove(&operation_id) {
            let duration = operation.start_time.elapsed();
            let bandwidth_mb_s = if duration.as_secs_f64() > 0.0 {
                bytes_transferred as f64 / (1024.0 * 1024.0) / duration.as_secs_f64()
            } else {
                0.0
            };

            self.io_queue_depth = self.io_queue_depth.saturating_sub(1);

            let device_type = match operation.operation_type {
                IoOperationType::FileRead | IoOperationType::FileWrite => IoDeviceType::SSD,
                IoOperationType::NetworkRead | IoOperationType::NetworkWrite => {
                    IoDeviceType::Network
                },
                IoOperationType::CacheLoad | IoOperationType::CacheStore => IoDeviceType::Cache,
                _ => IoDeviceType::Memory,
            };

            // Record bandwidth sample
            self.bandwidth_history.push(BandwidthSample {
                timestamp: SystemTime::now(),
                bandwidth_mb_s,
                device_type: device_type.clone(),
            });

            // Keep only recent samples
            if self.bandwidth_history.len() > 1000 {
                self.bandwidth_history.drain(0..500);
            }

            Some(IoProfile {
                operation_type: operation.operation_type,
                file_path: None, // Would be filled in practice
                bytes_transferred,
                duration,
                bandwidth_mb_s,
                queue_time: Duration::from_millis(self.io_queue_depth as u64 * 10), // Simplified
                device_type,
            })
        } else {
            None
        }
    }

    pub fn get_average_bandwidth(&self, device_type: &IoDeviceType) -> f64 {
        let samples: Vec<f64> = self
            .bandwidth_history
            .iter()
            .filter(|s| &s.device_type == device_type)
            .map(|s| s.bandwidth_mb_s)
            .collect();

        if samples.is_empty() {
            0.0
        } else {
            samples.iter().sum::<f64>() / samples.len() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_monitor_new() {
        let monitor = IoMonitor::new();
        assert_eq!(monitor.io_queue_depth, 0);
        assert!(monitor.bandwidth_history.is_empty());
    }

    #[test]
    fn test_io_monitor_start_operation() {
        let mut monitor = IoMonitor::new();
        let _id = monitor.start_io_operation(IoOperationType::FileRead, 4096);
        assert_eq!(monitor.io_queue_depth, 1);
        assert_eq!(monitor.active_operations.len(), 1);
    }

    #[test]
    fn test_io_monitor_finish_operation() {
        let mut monitor = IoMonitor::new();
        let id = monitor.start_io_operation(IoOperationType::FileWrite, 8192);
        let profile = monitor.finish_io_operation(id, 8192);
        assert!(profile.is_some());
        let p = profile.expect("profile should be Some");
        assert_eq!(p.bytes_transferred, 8192);
        assert_eq!(monitor.io_queue_depth, 0);
    }

    #[test]
    fn test_io_monitor_finish_nonexistent() {
        let mut monitor = IoMonitor::new();
        let profile = monitor.finish_io_operation(Uuid::new_v4(), 100);
        assert!(profile.is_none());
    }

    #[test]
    fn test_io_monitor_average_bandwidth_empty() {
        let monitor = IoMonitor::new();
        assert!((monitor.get_average_bandwidth(&IoDeviceType::SSD) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_io_monitor_device_type_mapping() {
        let mut monitor = IoMonitor::new();
        let id = monitor.start_io_operation(IoOperationType::NetworkRead, 1024);
        let profile = monitor.finish_io_operation(id, 1024);
        assert!(profile.is_some());
        let p = profile.expect("profile should be Some");
        assert_eq!(p.device_type, IoDeviceType::Network);
    }

    #[test]
    fn test_io_monitor_cache_device_type() {
        let mut monitor = IoMonitor::new();
        let id = monitor.start_io_operation(IoOperationType::CacheLoad, 512);
        let profile = monitor.finish_io_operation(id, 512);
        assert!(profile.is_some());
        let p = profile.expect("profile should be Some");
        assert_eq!(p.device_type, IoDeviceType::Cache);
    }
}
