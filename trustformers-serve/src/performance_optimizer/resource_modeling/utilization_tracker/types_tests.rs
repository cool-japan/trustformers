//! Tests for utilization tracker types

use super::*;
use std::time::Duration;
use chrono::Utc;

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32
    }
}

#[test]
fn test_utilization_history_new_empty() {
    let hist: UtilizationHistory<f32> = UtilizationHistory::new(100);
    assert!(hist.is_empty());
    assert_eq!(hist.len(), 0);
}

#[test]
fn test_utilization_history_add_sample() {
    let mut hist: UtilizationHistory<f32> = UtilizationHistory::new(10);
    hist.add_sample(50.0f32, Utc::now());
    assert_eq!(hist.len(), 1);
    assert!(!hist.is_empty());
}

#[test]
fn test_utilization_history_get_latest() {
    let mut hist: UtilizationHistory<f32> = UtilizationHistory::new(10);
    hist.add_sample(30.0f32, Utc::now());
    hist.add_sample(40.0f32, Utc::now());
    let latest = hist.get_latest_sample();
    assert!(latest.is_some());
    if let Some((val, _)) = latest {
        assert_eq!(val, 40.0);
    }
}

#[test]
fn test_utilization_history_max_size_enforced() {
    let mut hist: UtilizationHistory<f32> = UtilizationHistory::new(3);
    let now = Utc::now();
    hist.add_sample(1.0, now);
    hist.add_sample(2.0, now);
    hist.add_sample(3.0, now);
    hist.add_sample(4.0, now); // should evict first
    assert_eq!(hist.len(), 3);
}

#[test]
fn test_utilization_history_get_all_samples() {
    let mut hist: UtilizationHistory<f32> = UtilizationHistory::new(10);
    let now = Utc::now();
    hist.add_sample(10.0, now);
    hist.add_sample(20.0, now);
    let samples = hist.get_all_samples();
    assert_eq!(samples.len(), 2);
}

#[test]
fn test_utilization_stats_from_empty() {
    let stats = UtilizationStats::from_samples(&[]);
    assert_eq!(stats.average, 0.0);
    assert_eq!(stats.minimum, 0.0);
    assert_eq!(stats.maximum, 0.0);
}

#[test]
fn test_utilization_stats_from_samples() {
    let samples = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let stats = UtilizationStats::from_samples(&samples);
    assert!((stats.average - 30.0).abs() < 0.001);
    assert_eq!(stats.minimum, 10.0);
    assert_eq!(stats.maximum, 50.0);
    assert!(stats.std_deviation > 0.0);
}

#[test]
fn test_utilization_stats_clone() {
    let stats = UtilizationStats {
        average: 45.0,
        minimum: 10.0,
        maximum: 80.0,
        std_deviation: 15.0,
        percentile_95: 75.0,
        percentile_99: 78.0,
    };
    let c = stats.clone();
    assert_eq!(c.average, 45.0);
    assert_eq!(c.minimum, 10.0);
}

#[test]
fn test_io_monitor_config_clone() {
    let cfg = IoMonitorConfig {
        per_device_monitoring: true,
        latency_tracking: false,
        queue_depth_monitoring: true,
        device_health_monitoring: false,
        latency_threshold: Duration::from_millis(100),
        health_check_interval: Duration::from_secs(30),
    };
    let c = cfg.clone();
    assert!(c.per_device_monitoring);
    assert!(!c.latency_tracking);
    assert_eq!(c.latency_threshold, Duration::from_millis(100));
}

#[test]
fn test_cpu_monitor_config_clone() {
    let cfg = CpuMonitorConfig {
        per_core_monitoring: true,
        per_thread_monitoring: false,
        frequency_monitoring: true,
        temperature_correlation: false,
        thread_monitoring_threshold: 5.0,
        max_tracked_threads: 64,
    };
    let c = cfg.clone();
    assert!(c.per_core_monitoring);
    assert_eq!(c.max_tracked_threads, 64);
}

#[test]
fn test_memory_monitor_config_clone() {
    let cfg = MemoryMonitorConfig {
        allocation_pattern_analysis: true,
        bandwidth_monitoring: true,
        page_fault_tracking: false,
        swap_monitoring: true,
        pressure_threshold: 0.8,
        pattern_analysis_window: 100,
    };
    let c = cfg.clone();
    assert!(c.allocation_pattern_analysis);
    assert_eq!(c.pattern_analysis_window, 100);
}

#[test]
fn test_gpu_monitor_config_clone() {
    let cfg = GpuMonitorConfig {
        per_device_monitoring: true,
        memory_utilization_tracking: true,
        temperature_monitoring: true,
        power_monitoring: false,
        kernel_execution_tracking: false,
        max_tracked_kernels: 32,
    };
    let c = cfg.clone();
    assert!(c.per_device_monitoring);
    assert_eq!(c.max_tracked_kernels, 32);
}

#[test]
fn test_network_monitor_config_clone() {
    let cfg = NetworkMonitorConfig {
        per_interface_monitoring: true,
        protocol_analysis: false,
        connection_tracking: true,
        packet_loss_monitoring: false,
        monitored_protocols: vec!["tcp".to_string(), "udp".to_string()],
        max_tracked_connections: 1000,
    };
    let c = cfg.clone();
    assert!(c.per_interface_monitoring);
    assert_eq!(c.monitored_protocols.len(), 2);
}

#[test]
fn test_kernel_execution_metrics_clone() {
    let m = KernelExecutionMetrics {
        kernel_name: "matmul".to_string(),
        execution_time: Duration::from_micros(500),
        grid_size: (32, 32, 1),
        block_size: (16, 16, 1),
        shared_memory_bytes: 4096,
        timestamp: Utc::now(),
    };
    let c = m.clone();
    assert_eq!(c.kernel_name, "matmul");
    assert_eq!(c.shared_memory_bytes, 4096);
}

#[test]
fn test_allocation_pattern_clone() {
    let p = AllocationPattern {
        pattern_type: AllocationPatternType::SmallFrequent,
        allocation_size: 1024,
        frequency: 100.0,
        timestamp: Utc::now(),
    };
    let c = p.clone();
    assert_eq!(c.allocation_size, 1024);
    assert_eq!(c.frequency, 100.0);
}

#[test]
fn test_allocation_pattern_type_variants() {
    let variants = [
        AllocationPatternType::SmallFrequent,
        AllocationPatternType::LargeInfrequent,
        AllocationPatternType::SteadyGrowth,
        AllocationPatternType::Spike,
        AllocationPatternType::Leak,
    ];
    for v in &variants {
        let s = format!("{:?}", v);
        assert!(!s.is_empty());
    }
}

#[test]
fn test_memory_pressure_level_variants() {
    let variants = [
        MemoryPressureLevel::None,
        MemoryPressureLevel::Low,
        MemoryPressureLevel::Medium,
        MemoryPressureLevel::High,
        MemoryPressureLevel::Critical,
    ];
    for v in &variants {
        let s = format!("{:?}", v);
        assert!(!s.is_empty());
    }
}

#[test]
fn test_memory_usage_metrics_clone() {
    let m = MemoryUsageMetrics {
        total_usage_percent: 75.0,
        rss_usage_bytes: 1_073_741_824,
        virtual_usage_bytes: 2_147_483_648,
        available_bytes: 3_221_225_472,
        cached_bytes: 536_870_912,
        buffer_bytes: 268_435_456,
    };
    let c = m.clone();
    assert_eq!(c.total_usage_percent, 75.0);
    assert_eq!(c.rss_usage_bytes, 1_073_741_824);
}

#[test]
fn test_utilization_event_sample_collected() {
    let event = UtilizationEvent::SampleCollected {
        timestamp: Utc::now(),
        resource_type: "cpu".to_string(),
        value: 42.0,
    };
    let s = format!("{:?}", event);
    assert!(s.contains("SampleCollected"));
}

#[test]
fn test_utilization_event_threshold_exceeded() {
    let event = UtilizationEvent::ThresholdExceeded {
        timestamp: Utc::now(),
        resource_type: "memory".to_string(),
        threshold: 80.0,
        current_value: 92.0,
    };
    let s = format!("{:?}", event);
    assert!(s.contains("ThresholdExceeded"));
}

#[test]
fn test_utilization_event_anomaly_detected() {
    let event = UtilizationEvent::AnomalyDetected {
        timestamp: Utc::now(),
        resource_type: "network".to_string(),
        anomaly_score: 0.95,
        description: "Unusual traffic".to_string(),
    };
    let s = format!("{:?}", event);
    assert!(s.contains("AnomalyDetected"));
}

#[test]
fn test_monitoring_state_clone() {
    let state = MonitoringState {
        is_active: true,
        start_time: Some(Utc::now()),
        sample_count: 100,
        last_sample_time: Some(Utc::now()),
        error_count: 0,
    };
    let c = state.clone();
    assert!(c.is_active);
    assert_eq!(c.sample_count, 100);
    assert_eq!(c.error_count, 0);
}

#[test]
fn test_network_interface_statistics_clone() {
    let stats = NetworkInterfaceStatistics {
        interface_name: "eth0".to_string(),
        rx_bytes_per_sec: 1024.0,
        tx_bytes_per_sec: 512.0,
        rx_packets_per_sec: 100.0,
        tx_packets_per_sec: 50.0,
        utilization_percent: 25.0,
        error_rate: 0.001,
        drop_rate: 0.0,
        last_update: Utc::now(),
    };
    let c = stats.clone();
    assert_eq!(c.interface_name, "eth0");
    assert_eq!(c.utilization_percent, 25.0);
}

#[test]
fn test_io_device_statistics_clone() {
    let stats = IoDeviceStatistics {
        device_name: "sda".to_string(),
        read_ops_per_sec: 100.0,
        write_ops_per_sec: 50.0,
        read_bandwidth_mbps: 200.0,
        write_bandwidth_mbps: 100.0,
        average_latency: Duration::from_millis(5),
        queue_depth: 4,
        utilization_percent: 40.0,
        last_update: Utc::now(),
    };
    let c = stats.clone();
    assert_eq!(c.device_name, "sda");
    assert_eq!(c.queue_depth, 4);
}

#[test]
fn test_thread_utilization_clone() {
    let tu = ThreadUtilization {
        thread_id: 42,
        process_id: 1234,
        thread_name: "worker-1".to_string(),
        cpu_utilization: 15.0,
        memory_usage: 1048576,
        last_update: Utc::now(),
    };
    let c = tu.clone();
    assert_eq!(c.thread_id, 42);
    assert_eq!(c.memory_usage, 1048576);
}

#[test]
fn test_utilization_report_clone() {
    let empty_stats = UtilizationStats::from_samples(&[50.0]);
    let report = UtilizationReport {
        duration: Duration::from_secs(3600),
        cpu_utilization: empty_stats.clone(),
        memory_utilization: empty_stats.clone(),
        io_utilization: empty_stats.clone(),
        network_utilization: empty_stats.clone(),
        gpu_utilization: None,
        timestamp: Utc::now(),
    };
    let c = report.clone();
    assert_eq!(c.duration, Duration::from_secs(3600));
    assert!(c.gpu_utilization.is_none());
}

#[test]
fn test_retention_policy_default() {
    let rp = RetentionPolicy::default();
    let _ = rp; // just test Default works
}

#[test]
fn test_compression_config_default() {
    let cc = CompressionConfig::default();
    let _ = cc;
}
