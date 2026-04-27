#[cfg(test)]
mod tests {
    use crate::profiler::*;
    use crate::DebugConfig;
    use uuid::Uuid;
    use std::collections::HashMap;
    use std::time::Duration;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_f64(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn test_debug_config() -> DebugConfig {
        DebugConfig::default()
    }

    // Test 1: Profiler creation
    #[test]
    fn test_profiler_creation() {
        let config = test_debug_config();
        let profiler = Profiler::new(&config);
        assert!(profiler.get_events().is_empty());
        assert!(profiler.get_layer_profiles().is_empty());
        assert!(profiler.get_memory_timeline().is_empty());
    }

    // Test 2: MemoryTracker creation
    #[test]
    fn test_memory_tracker_creation() {
        let tracker = MemoryTracker::new();
        let stats = tracker.get_memory_stats();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.peak_allocated, 0);
        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.allocation_count, 0);
    }

    // Test 3: MemoryTracker allocation tracking
    #[test]
    fn test_memory_tracker_allocation() {
        let mut tracker = MemoryTracker::new();
        let alloc_id = Uuid::new_v4();
        let allocation = MemoryAllocation {
            allocation_id: alloc_id,
            size_bytes: 1024,
            allocation_type: MemoryAllocationType::Host,
            device_id: None,
            timestamp: std::time::SystemTime::now(),
            stack_trace: vec!["main".to_string()],
            freed: false,
            free_timestamp: None,
        };
        tracker.track_allocation(allocation);
        let stats = tracker.get_memory_stats();
        assert_eq!(stats.total_allocated, 1024);
        assert_eq!(stats.peak_allocated, 1024);
        assert_eq!(stats.active_allocations, 1);
    }

    // Test 4: MemoryTracker deallocation tracking
    #[test]
    fn test_memory_tracker_deallocation() {
        let mut tracker = MemoryTracker::new();
        let alloc_id = Uuid::new_v4();
        let allocation = MemoryAllocation {
            allocation_id: alloc_id,
            size_bytes: 2048,
            allocation_type: MemoryAllocationType::Device,
            device_id: Some(0),
            timestamp: std::time::SystemTime::now(),
            stack_trace: vec![],
            freed: false,
            free_timestamp: None,
        };
        tracker.track_allocation(allocation);
        tracker.track_deallocation(alloc_id);
        let stats = tracker.get_memory_stats();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.peak_allocated, 2048);
        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.deallocation_count, 1);
    }

    // Test 5: MemoryTracker peak tracking
    #[test]
    fn test_memory_tracker_peak() {
        let mut tracker = MemoryTracker::new();
        // Allocate 3 items first (peak = 3000), then deallocate some and add more
        let mut ids = Vec::new();
        for _ in 0..3 {
            let alloc_id = Uuid::new_v4();
            tracker.track_allocation(MemoryAllocation {
                allocation_id: alloc_id,
                size_bytes: 1000,
                allocation_type: MemoryAllocationType::Host,
                device_id: None,
                timestamp: std::time::SystemTime::now(),
                stack_trace: vec![],
                freed: false,
                free_timestamp: None,
            });
            ids.push(alloc_id);
        }
        // Peak should be 3000 at this point
        let stats = tracker.get_memory_stats();
        assert_eq!(stats.peak_allocated, 3000);
        // Deallocate first one
        tracker.track_deallocation(ids[0]);
        let stats = tracker.get_memory_stats();
        assert_eq!(stats.peak_allocated, 3000); // Peak unchanged
        assert_eq!(stats.active_allocations, 2);
        assert_eq!(stats.total_allocated, 2000);
    }

    // Test 6: GpuProfiler creation
    #[test]
    fn test_gpu_profiler_creation() {
        if let Ok(profiler) = GpuProfiler::new() {
            let util = profiler.get_gpu_utilization(0);
            assert!((util - 0.0).abs() < f64::EPSILON);
        }
    }

    // Test 7: GpuProfiler kernel profiling
    #[test]
    fn test_gpu_profiler_kernel() {
        if let Ok(mut profiler) = GpuProfiler::new() {
            let kernel = GpuKernelProfile {
                kernel_name: "matmul_kernel".to_string(),
                grid_size: (64, 64, 1),
                block_size: (16, 16, 1),
                shared_memory_bytes: 4096,
                registers_per_thread: 32,
                occupancy: 0.85,
                execution_time: Duration::from_micros(100),
                memory_bandwidth_gb_s: 500.0,
                compute_utilization: 0.9,
                stream_id: 0,
            };
            profiler.profile_kernel(kernel);
            let util = profiler.get_gpu_utilization(0);
            assert!((util - 0.9).abs() < f64::EPSILON);
        }
    }

    // Test 8: IoMonitor creation
    #[test]
    fn test_io_monitor_creation() {
        let monitor = IoMonitor::new();
        assert!((monitor.get_average_bandwidth(&IoDeviceType::SSD) - 0.0).abs() < f64::EPSILON);
    }

    // Test 9: IoMonitor start and finish operation
    #[test]
    fn test_io_monitor_operation() {
        let mut monitor = IoMonitor::new();
        let op_id = monitor.start_io_operation(IoOperationType::FileRead, 1024 * 1024);
        let profile = monitor.finish_io_operation(op_id, 1024 * 1024);
        assert!(profile.is_some());
        if let Some(ref p) = profile {
            assert_eq!(p.bytes_transferred, 1024 * 1024);
        }
    }

    // Test 10: IoMonitor bandwidth tracking
    #[test]
    fn test_io_monitor_bandwidth() {
        let mut monitor = IoMonitor::new();
        for _ in 0..5 {
            let op_id = monitor.start_io_operation(IoOperationType::FileRead, 1024 * 1024);
            let _profile = monitor.finish_io_operation(op_id, 1024 * 1024);
        }
        let bandwidth = monitor.get_average_bandwidth(&IoDeviceType::SSD);
        assert!(bandwidth >= 0.0);
    }

    // Test 11: Profiler timer operations
    #[test]
    fn test_profiler_timer_operations() {
        let config = test_debug_config();
        let mut profiler = Profiler::new(&config);
        profiler.start_timer("test_op");
        std::thread::sleep(Duration::from_millis(10));
        let duration = profiler.end_timer("test_op");
        assert!(duration.is_some());
        if let Some(d) = duration {
            assert!(d.as_millis() >= 10);
        }
        assert_eq!(profiler.get_events().len(), 1);
    }

    // Test 12: Profiler timer not started
    #[test]
    fn test_profiler_timer_not_started() {
        let config = test_debug_config();
        let mut profiler = Profiler::new(&config);
        let duration = profiler.end_timer("nonexistent");
        assert!(duration.is_none());
    }

    // Test 13: Profiler record layer execution
    #[test]
    fn test_profiler_record_layer_execution() {
        let config = test_debug_config();
        let mut profiler = Profiler::new(&config);
        profiler.record_layer_execution(
            "attention_0",
            "MultiHeadAttention",
            Duration::from_millis(50),
            Some(Duration::from_millis(75)),
            1024 * 1024,
            1_000_000,
        );
        assert_eq!(profiler.get_events().len(), 1);
        assert_eq!(profiler.get_layer_profiles().len(), 1);
        let profile = profiler.get_layer_profiles().get("attention_0");
        assert!(profile.is_some());
        if let Some(p) = profile {
            assert_eq!(p.call_count(), 1);
            assert_eq!(p.forward_times().len(), 1);
            assert_eq!(p.backward_times().len(), 1);
        }
    }

    // Test 14: Profiler record tensor operation
    #[test]
    fn test_profiler_record_tensor_operation() {
        let config = test_debug_config();
        let mut profiler = Profiler::new(&config);
        profiler.record_tensor_operation(
            "matmul",
            &[32, 768, 768],
            Duration::from_millis(5),
            1024 * 1024 * 10,
        );
        assert_eq!(profiler.get_events().len(), 1);
    }

    // Test 15: Profiler record model inference
    #[test]
    fn test_profiler_record_model_inference() {
        let config = test_debug_config();
        let mut profiler = Profiler::new(&config);
        profiler.record_model_inference(32, 512, Duration::from_millis(100));
        assert_eq!(profiler.get_events().len(), 1);
        if let ProfileEvent::ModelInference {
            tokens_per_second, ..
        } = &profiler.get_events()[0]
        {
            let expected_tps = (32 * 512) as f64 / 0.1;
            assert!((tokens_per_second - expected_tps).abs() < 1.0);
        }
    }

    // Test 16: Profiler record gradient computation
    #[test]
    fn test_profiler_record_gradient() {
        let config = test_debug_config();
        let mut profiler = Profiler::new(&config);
        profiler.record_gradient_computation("layer_0", 2.5, Duration::from_millis(20));
        assert_eq!(profiler.get_events().len(), 1);
    }

    // Test 17: Profiler get statistics
    #[test]
    fn test_profiler_get_statistics() {
        let config = test_debug_config();
        let mut profiler = Profiler::new(&config);
        for i in 0..10 {
            profiler.record_tensor_operation(
                "add",
                &[32, 768],
                Duration::from_millis(i + 1),
                1024,
            );
        }
        let stats = profiler.get_statistics();
        assert!(stats.contains_key("TensorOperation"));
        let tensor_stats = stats.get("TensorOperation");
        assert!(tensor_stats.is_some());
        if let Some(s) = tensor_stats {
            assert_eq!(s.count, 10);
        }
    }

    // Test 18: Profiler clear
    #[test]
    fn test_profiler_clear() {
        let config = test_debug_config();
        let mut profiler = Profiler::new(&config);
        profiler.record_tensor_operation("add", &[32], Duration::from_millis(1), 100);
        profiler.record_layer_execution(
            "layer_0",
            "Linear",
            Duration::from_millis(10),
            None,
            1024,
            100,
        );
        profiler.clear();
        assert!(profiler.get_events().is_empty());
        assert!(profiler.get_layer_profiles().is_empty());
    }

    // Test 19: Profiler memory snapshot
    #[test]
    fn test_profiler_memory_snapshot() {
        let config = test_debug_config();
        let mut profiler = Profiler::new(&config);
        profiler.take_memory_snapshot();
        profiler.take_memory_snapshot();
        assert_eq!(profiler.get_memory_timeline().len(), 2);
    }

    // Test 20: BottleneckType variants
    #[test]
    fn test_bottleneck_type_variants() {
        let types = vec![
            BottleneckType::CpuBound,
            BottleneckType::MemoryBound,
            BottleneckType::IoBound,
            BottleneckType::GpuBound,
            BottleneckType::NetworkBound,
            BottleneckType::DataLoading,
            BottleneckType::ModelComputation,
            BottleneckType::GradientComputation,
        ];
        assert_eq!(types.len(), 8);
    }

    // Test 21: BottleneckSeverity variants
    #[test]
    fn test_bottleneck_severity_variants() {
        let severities = vec![
            BottleneckSeverity::Low,
            BottleneckSeverity::Medium,
            BottleneckSeverity::High,
            BottleneckSeverity::Critical,
        ];
        assert_eq!(severities.len(), 4);
    }

    // Test 22: MemoryAllocationType variants
    #[test]
    fn test_memory_allocation_type_variants() {
        let types = vec![
            MemoryAllocationType::Host,
            MemoryAllocationType::Device,
            MemoryAllocationType::Unified,
            MemoryAllocationType::Pinned,
            MemoryAllocationType::Mapped,
        ];
        assert_eq!(types.len(), 5);
    }

    // Test 23: IoOperationType variants
    #[test]
    fn test_io_operation_type_variants() {
        let types = vec![
            IoOperationType::FileRead,
            IoOperationType::FileWrite,
            IoOperationType::NetworkRead,
            IoOperationType::NetworkWrite,
            IoOperationType::DatabaseQuery,
            IoOperationType::CacheLoad,
            IoOperationType::CacheStore,
        ];
        assert_eq!(types.len(), 7);
    }

    // Test 24: Profiler track memory allocation and deallocation
    #[test]
    fn test_profiler_memory_tracking() {
        let config = test_debug_config();
        let mut profiler = Profiler::new(&config);
        let alloc_id = profiler.track_memory_allocation(
            4096,
            MemoryAllocationType::Host,
            None,
            vec!["test_func".to_string()],
        );
        let stats_before = profiler.get_memory_stats();
        assert!(stats_before.is_some());
        if let Some(s) = stats_before {
            assert_eq!(s.total_allocated, 4096);
        }
        profiler.track_memory_deallocation(alloc_id);
        let stats_after = profiler.get_memory_stats();
        assert!(stats_after.is_some());
        if let Some(s) = stats_after {
            assert_eq!(s.total_allocated, 0);
        }
    }

    // Test 25: Profiler with LCG-generated layer data
    #[test]
    fn test_profiler_lcg_layer_data() {
        let config = test_debug_config();
        let mut profiler = Profiler::new(&config);
        let mut lcg = Lcg::new(42);
        let layer_names = ["attention", "ffn", "norm", "embedding", "output"];
        for name in &layer_names {
            let forward_ms = (lcg.next() % 200) + 1;
            let backward_ms = (lcg.next() % 300) + 1;
            let memory = (lcg.next() % (1024 * 1024)) as usize;
            let params = (lcg.next() % 10_000_000) as usize;
            profiler.record_layer_execution(
                name,
                "Linear",
                Duration::from_millis(forward_ms),
                Some(Duration::from_millis(backward_ms)),
                memory,
                params,
            );
        }
        assert_eq!(profiler.get_layer_profiles().len(), 5);
        let stats = profiler.get_statistics();
        assert!(stats.contains_key("LayerExecution"));
    }
}
