//! Tests for flame_graph_profiler module

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::flame_graph_profiler::{
        DifferentialAnalysis, FlameGraphColorScheme, FlameGraphConfig, FlameGraphDirection,
        FlameGraphExportFormat, FlameGraphNode, FlameGraphProfiler, FlameGraphSample,
        GpuKernelStats, HotFunctionInfo, MemoryUsageStats, StackFrame,
    };

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    fn make_default_config() -> FlameGraphConfig {
        FlameGraphConfig::default()
    }

    fn make_simple_stack(function_names: &[&str]) -> Vec<StackFrame> {
        function_names
            .iter()
            .map(|name| StackFrame {
                function_name: name.to_string(),
                module_name: Some("test_module".to_string()),
                file_name: None,
                line_number: None,
                address: None,
            })
            .collect()
    }

    fn make_sample(stack_names: &[&str], duration_ns: u64) -> FlameGraphSample {
        FlameGraphSample {
            stack: make_simple_stack(stack_names),
            duration_ns,
            timestamp: 1_000_000_000u64,
            thread_id: 1,
            cpu_id: Some(0),
            memory_usage: Some(1024 * 1024),
            gpu_kernel: None,
            metadata: HashMap::new(),
        }
    }

    // -------------------------------------------------------------------------
    // FlameGraphConfig
    // -------------------------------------------------------------------------

    #[test]
    fn test_default_config_fields() {
        let config = make_default_config();
        assert_eq!(config.sampling_rate, 1000);
        assert!(config.filter_noise);
        assert!(config.include_memory);
        assert!(config.include_gpu);
        assert!(!config.differential_mode);
        assert!(config.merge_similar_stacks);
    }

    #[test]
    fn test_custom_config_creation() {
        let config = FlameGraphConfig {
            sampling_rate: 500,
            min_width: 0.05,
            color_scheme: FlameGraphColorScheme::Cool,
            direction: FlameGraphDirection::BottomUp,
            title: "Custom Graph".to_string(),
            subtitle: Some("subtitle".to_string()),
            include_memory: false,
            include_gpu: false,
            differential_mode: true,
            merge_similar_stacks: false,
            filter_noise: false,
            noise_threshold: 0.5,
        };
        assert_eq!(config.sampling_rate, 500);
        assert!(config.subtitle.is_some());
        assert!(config.differential_mode);
    }

    // -------------------------------------------------------------------------
    // FlameGraphColorScheme variants
    // -------------------------------------------------------------------------

    #[test]
    fn test_color_scheme_variants_debug() {
        let schemes = [
            FlameGraphColorScheme::Hot,
            FlameGraphColorScheme::Cool,
            FlameGraphColorScheme::Java,
            FlameGraphColorScheme::Memory,
            FlameGraphColorScheme::Differential,
            FlameGraphColorScheme::Random,
        ];
        for s in &schemes {
            let dbg = format!("{:?}", s);
            assert!(!dbg.is_empty());
        }
    }

    #[test]
    fn test_color_scheme_custom_map() {
        let mut custom_colors = HashMap::new();
        custom_colors.insert("main".to_string(), "#ff0000".to_string());
        let scheme = FlameGraphColorScheme::Custom(custom_colors);
        let dbg = format!("{:?}", scheme);
        assert!(dbg.contains("Custom"));
    }

    // -------------------------------------------------------------------------
    // FlameGraphDirection
    // -------------------------------------------------------------------------

    #[test]
    fn test_direction_variants() {
        let td = FlameGraphDirection::TopDown;
        let bu = FlameGraphDirection::BottomUp;
        let dbg_td = format!("{:?}", td);
        let dbg_bu = format!("{:?}", bu);
        assert!(dbg_td.contains("TopDown"));
        assert!(dbg_bu.contains("BottomUp"));
    }

    // -------------------------------------------------------------------------
    // FlameGraphExportFormat
    // -------------------------------------------------------------------------

    #[test]
    fn test_export_format_variants_debug() {
        let formats = [
            FlameGraphExportFormat::SVG,
            FlameGraphExportFormat::InteractiveHTML,
            FlameGraphExportFormat::JSON,
            FlameGraphExportFormat::Speedscope,
            FlameGraphExportFormat::D3,
            FlameGraphExportFormat::Folded,
        ];
        for f in &formats {
            let dbg = format!("{:?}", f);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // StackFrame
    // -------------------------------------------------------------------------

    #[test]
    fn test_stack_frame_construction() {
        let frame = StackFrame {
            function_name: "my_func".to_string(),
            module_name: Some("my_module".to_string()),
            file_name: Some("src/lib.rs".to_string()),
            line_number: Some(42),
            address: Some(0xdeadbeef),
        };
        assert_eq!(frame.function_name, "my_func");
        assert_eq!(frame.line_number, Some(42));
    }

    #[test]
    fn test_stack_frame_equality() {
        let frame1 = StackFrame {
            function_name: "func_a".to_string(),
            module_name: None,
            file_name: None,
            line_number: None,
            address: None,
        };
        let frame2 = StackFrame {
            function_name: "func_a".to_string(),
            module_name: None,
            file_name: None,
            line_number: None,
            address: None,
        };
        let frame3 = StackFrame {
            function_name: "func_b".to_string(),
            module_name: None,
            file_name: None,
            line_number: None,
            address: None,
        };
        assert_eq!(frame1, frame2);
        assert_ne!(frame1, frame3);
    }

    // -------------------------------------------------------------------------
    // FlameGraphNode
    // -------------------------------------------------------------------------

    #[test]
    fn test_flame_graph_node_construction() {
        let node = FlameGraphNode {
            name: "root".to_string(),
            value: 1000,
            delta: None,
            children: HashMap::new(),
            total_value: 1000,
            self_value: 100,
            percentage: 100.0,
            color: None,
            metadata: HashMap::new(),
        };
        assert_eq!(node.name, "root");
        assert_eq!(node.value, 1000);
        assert!(node.children.is_empty());
    }

    #[test]
    fn test_flame_graph_node_with_delta() {
        let node = FlameGraphNode {
            name: "fn_a".to_string(),
            value: 500,
            delta: Some(-100),
            children: HashMap::new(),
            total_value: 500,
            self_value: 500,
            percentage: 50.0,
            color: Some("#ff0000".to_string()),
            metadata: HashMap::new(),
        };
        assert_eq!(node.delta, Some(-100));
        assert!(node.color.is_some());
    }

    // -------------------------------------------------------------------------
    // FlameGraphProfiler construction
    // -------------------------------------------------------------------------

    #[test]
    fn test_profiler_new() {
        let config = make_default_config();
        let profiler = FlameGraphProfiler::new(config);
        let dbg = format!("{:?}", profiler);
        assert!(!dbg.is_empty());
    }

    // -------------------------------------------------------------------------
    // start_sampling / stop_sampling
    // -------------------------------------------------------------------------

    #[test]
    fn test_start_sampling_succeeds() {
        let config = make_default_config();
        let mut profiler = FlameGraphProfiler::new(config);
        let result = profiler.start_sampling();
        assert!(result.is_ok(), "start_sampling should succeed");
    }

    #[test]
    fn test_stop_sampling_without_samples_fails() {
        let config = make_default_config();
        let mut profiler = FlameGraphProfiler::new(config);
        profiler.start_sampling().expect("start ok");
        // stop_sampling with no samples should fail (no samples = can't build graph)
        let result = profiler.stop_sampling();
        assert!(result.is_err(), "stop_sampling with no samples should return an error");
    }

    // -------------------------------------------------------------------------
    // add_sample / build_flame_graph
    // -------------------------------------------------------------------------

    #[test]
    fn test_add_sample_increases_count() {
        let config = make_default_config();
        let mut profiler = FlameGraphProfiler::new(config);
        profiler.start_sampling().expect("start ok");
        let sample = make_sample(&["main", "do_work", "compute"], 1000);
        profiler.add_sample(sample);
        // Add another
        let sample2 = make_sample(&["main", "do_io"], 500);
        profiler.add_sample(sample2);
        // build_flame_graph should succeed with samples
        let result = profiler.build_flame_graph();
        assert!(result.is_ok(), "build_flame_graph should succeed with samples");
    }

    #[test]
    fn test_build_flame_graph_without_samples_fails() {
        let config = make_default_config();
        let mut profiler = FlameGraphProfiler::new(config);
        let result = profiler.build_flame_graph();
        assert!(result.is_err(), "build_flame_graph should fail without samples");
    }

    // -------------------------------------------------------------------------
    // sample_gpu_kernel
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_gpu_kernel() {
        let config = make_default_config();
        let mut profiler = FlameGraphProfiler::new(config);
        profiler.start_sampling().expect("start ok");
        profiler.sample_gpu_kernel("matmul_kernel", 5000);
        // Should now have at least one sample
        let result = profiler.build_flame_graph();
        assert!(result.is_ok(), "build should succeed after GPU kernel sample");
    }

    // -------------------------------------------------------------------------
    // set_baseline for differential analysis
    // -------------------------------------------------------------------------

    #[test]
    fn test_set_baseline_captures_samples() {
        let mut config = make_default_config();
        config.differential_mode = true;
        let mut profiler = FlameGraphProfiler::new(config);
        profiler.start_sampling().expect("start ok");
        // Add some baseline samples
        for i in 0..3 {
            let sample = make_sample(&["main", "baseline_work"], 1000 + i * 100);
            profiler.add_sample(sample);
        }
        profiler.set_baseline();
        // Add current samples
        let sample = make_sample(&["main", "current_work"], 2000);
        profiler.add_sample(sample);
        let result = profiler.build_flame_graph();
        assert!(result.is_ok(), "build should succeed with differential mode");
    }

    // -------------------------------------------------------------------------
    // HotFunctionInfo / MemoryUsageStats / GpuKernelStats
    // -------------------------------------------------------------------------

    #[test]
    fn test_hot_function_info_construction() {
        let info = HotFunctionInfo {
            name: "hot_fn".to_string(),
            total_time_ns: 10000,
            self_time_ns: 5000,
            percentage: 50.0,
            call_count: 100,
        };
        assert_eq!(info.name, "hot_fn");
        assert!(info.percentage > 0.0);
    }

    #[test]
    fn test_memory_usage_stats_default() {
        let stats = MemoryUsageStats::default();
        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.peak_memory_bytes, 0);
    }

    #[test]
    fn test_gpu_kernel_stats_construction() {
        let stats = GpuKernelStats {
            total_kernel_time_ns: 50000,
            unique_kernels: 3,
            total_kernel_calls: 15,
        };
        assert_eq!(stats.unique_kernels, 3);
        assert_eq!(stats.total_kernel_calls, 15);
    }

    // -------------------------------------------------------------------------
    // DifferentialAnalysis
    // -------------------------------------------------------------------------

    #[test]
    fn test_differential_analysis_regression() {
        let diff = DifferentialAnalysis {
            baseline_samples: 100,
            current_samples: 110,
            performance_change_percent: 10.0,
            is_regression: true,
            is_improvement: false,
        };
        assert!(diff.is_regression);
        assert!(!diff.is_improvement);
    }

    #[test]
    fn test_differential_analysis_improvement() {
        let diff = DifferentialAnalysis {
            baseline_samples: 100,
            current_samples: 95,
            performance_change_percent: -15.0,
            is_regression: false,
            is_improvement: true,
        };
        assert!(diff.is_improvement);
        assert!(!diff.is_regression);
    }

    // -------------------------------------------------------------------------
    // FlameGraphSample
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_with_gpu_kernel() {
        let sample = FlameGraphSample {
            stack: vec![StackFrame {
                function_name: "GPU::matmul".to_string(),
                module_name: Some("GPU".to_string()),
                file_name: None,
                line_number: None,
                address: None,
            }],
            duration_ns: 50000,
            timestamp: 999_999_999,
            thread_id: 0,
            cpu_id: None,
            memory_usage: None,
            gpu_kernel: Some("matmul".to_string()),
            metadata: HashMap::new(),
        };
        assert!(sample.gpu_kernel.is_some());
        assert!(sample.cpu_id.is_none());
    }

    // -------------------------------------------------------------------------
    // Multiple samples building a graph
    // -------------------------------------------------------------------------

    #[test]
    fn test_flame_graph_with_varied_stacks() {
        let config = make_default_config();
        let mut profiler = FlameGraphProfiler::new(config);
        profiler.start_sampling().expect("start ok");

        let stacks: &[&[&str]] = &[
            &["main", "work_a", "compute"],
            &["main", "work_b"],
            &["main", "work_a", "io"],
            &["main", "work_c", "compute"],
            &["main", "work_a", "compute"],
        ];

        for (i, stack) in stacks.iter().enumerate() {
            profiler.add_sample(make_sample(stack, 1000 + i as u64 * 200));
        }

        let build_result = profiler.build_flame_graph();
        assert!(build_result.is_ok(), "build_flame_graph should succeed");
    }

    #[test]
    fn test_stop_sampling_with_samples_succeeds() {
        let config = make_default_config();
        let mut profiler = FlameGraphProfiler::new(config);
        profiler.start_sampling().expect("start ok");
        profiler.add_sample(make_sample(&["main"], 500));
        let result = profiler.stop_sampling();
        assert!(result.is_ok(), "stop_sampling should succeed with samples present");
    }
}
