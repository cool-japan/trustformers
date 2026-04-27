//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::errors::Result;
    use crate::performance::{LatencyMetrics, MemoryMetrics, ProfileResult};
    use std::collections::HashMap;
    use std::time::Duration;
    #[test]
    fn test_optimization_advisor() {
        let advisor = OptimizationAdvisor::new();
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: Some(MemoryMetrics {
                current_bytes: 1000 * 1024 * 1024,
                peak_bytes: 3000 * 1024 * 1024,
                allocated_bytes: 5000 * 1024 * 1024,
                reserved_bytes: 0,
                num_allocations: 1000,
                num_deallocations: 500,
                fragmentation_percent: 10.0,
            }),
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: HashMap::new(),
        };
        let report = advisor.analyze(&context).expect("operation failed in test");
        assert!(!report.suggestions.is_empty());
    }
    #[test]
    fn test_report_markdown() {
        let report = OptimizationReport {
            suggestions: vec![OptimizationSuggestion {
                id: "test1".to_string(),
                category: OptimizationCategory::Memory,
                impact: ImpactLevel::High,
                difficulty: Difficulty::Easy,
                title: "Test Optimization".to_string(),
                description: "Test description".to_string(),
                expected_improvement: PerformanceImprovement {
                    latency_reduction: Some(20.0),
                    throughput_increase: Some(30.0),
                    memory_reduction: Some(40.0),
                    other_metrics: HashMap::new(),
                },
                implementation_steps: vec!["Step 1".to_string()],
                code_examples: None,
                warnings: vec![],
                related_suggestions: vec![],
            }],
            summary: OptimizationSummary {
                total_suggestions: 1,
                suggestions_by_category: HashMap::from([(OptimizationCategory::Memory, 1)]),
                suggestions_by_impact: HashMap::from([(ImpactLevel::High, 1)]),
                potential_latency_reduction: 20.0,
                potential_memory_reduction: 40.0,
                potential_throughput_increase: 30.0,
            },
            hardware_info: HardwareInfo::default(),
        };
        let markdown = report.to_markdown();
        assert!(markdown.contains("Performance Optimization Report"));
        assert!(markdown.contains("Test Optimization"));
    }
    #[test]
    fn test_optimization_category_display() {
        assert_eq!(
            format!("{}", OptimizationCategory::Architecture),
            "Architecture"
        );
        assert_eq!(format!("{}", OptimizationCategory::Memory), "Memory");
        assert_eq!(format!("{}", OptimizationCategory::Compute), "Compute");
        assert_eq!(
            format!("{}", OptimizationCategory::Quantization),
            "Quantization"
        );
        assert_eq!(
            format!("{}", OptimizationCategory::Parallelization),
            "Parallelization"
        );
        assert_eq!(format!("{}", OptimizationCategory::Hardware), "Hardware");
        assert_eq!(
            format!("{}", OptimizationCategory::DataPipeline),
            "Data Pipeline"
        );
    }
    #[test]
    fn test_impact_level_display() {
        assert_eq!(format!("{}", ImpactLevel::Low), "Low");
        assert_eq!(format!("{}", ImpactLevel::Medium), "Medium");
        assert_eq!(format!("{}", ImpactLevel::High), "High");
        assert_eq!(format!("{}", ImpactLevel::Critical), "Critical");
    }
    #[test]
    fn test_impact_level_ordering() {
        assert!(ImpactLevel::Low < ImpactLevel::Medium);
        assert!(ImpactLevel::Medium < ImpactLevel::High);
        assert!(ImpactLevel::High < ImpactLevel::Critical);
    }
    #[test]
    fn test_difficulty_display() {
        assert_eq!(format!("{}", Difficulty::Easy), "Easy");
        assert_eq!(format!("{}", Difficulty::Medium), "Medium");
        assert_eq!(format!("{}", Difficulty::Hard), "Hard");
    }
    #[test]
    fn test_difficulty_ordering() {
        assert!(Difficulty::Easy < Difficulty::Medium);
        assert!(Difficulty::Medium < Difficulty::Hard);
    }
    #[test]
    fn test_hardware_info_default() {
        let hw = HardwareInfo::default();
        assert!(hw.cpu_cores > 0);
        assert!(hw.cpu_model.is_none());
        assert!(hw.gpu_model.is_none());
        assert!(hw.gpu_memory_mb.is_none());
        assert_eq!(hw.system_memory_mb, 8192);
        assert!(hw.simd_capabilities.is_empty());
    }
    #[test]
    fn test_optimization_advisor_default() {
        let advisor = OptimizationAdvisor::default();
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: HashMap::new(),
        };
        let report = advisor.analyze(&context).expect("analyze failed");
        assert!(report.suggestions.len() <= 10);
    }
    #[test]
    fn test_advisor_add_custom_rule() {
        struct AlwaysSuggestRule;
        impl OptimizationRule for AlwaysSuggestRule {
            fn analyze(&self, _ctx: &AnalysisContext) -> Result<Option<OptimizationSuggestion>> {
                Ok(Some(OptimizationSuggestion {
                    id: "custom_rule".to_string(),
                    category: OptimizationCategory::Hardware,
                    impact: ImpactLevel::Low,
                    difficulty: Difficulty::Easy,
                    title: "Custom suggestion".to_string(),
                    description: "Custom rule fired".to_string(),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: None,
                        throughput_increase: None,
                        memory_reduction: None,
                        other_metrics: HashMap::new(),
                    },
                    implementation_steps: vec![],
                    code_examples: None,
                    warnings: vec![],
                    related_suggestions: vec![],
                }))
            }
        }
        let mut advisor = OptimizationAdvisor::new();
        advisor.add_rule(Box::new(AlwaysSuggestRule));
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo {
                cpu_cores: 1,
                ..HardwareInfo::default()
            },
            current_config: HashMap::new(),
        };
        let report = advisor.analyze(&context).expect("analyze failed");
        assert!(report.suggestions.iter().any(|s| s.id == "custom_rule"));
    }
    #[test]
    fn test_report_high_impact_suggestions() {
        let report = OptimizationReport {
            suggestions: vec![
                OptimizationSuggestion {
                    id: "low".to_string(),
                    category: OptimizationCategory::Memory,
                    impact: ImpactLevel::Low,
                    difficulty: Difficulty::Easy,
                    title: "Low impact".to_string(),
                    description: String::new(),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: None,
                        throughput_increase: None,
                        memory_reduction: None,
                        other_metrics: HashMap::new(),
                    },
                    implementation_steps: vec![],
                    code_examples: None,
                    warnings: vec![],
                    related_suggestions: vec![],
                },
                OptimizationSuggestion {
                    id: "high".to_string(),
                    category: OptimizationCategory::Compute,
                    impact: ImpactLevel::High,
                    difficulty: Difficulty::Medium,
                    title: "High impact".to_string(),
                    description: String::new(),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: Some(50.0),
                        throughput_increase: None,
                        memory_reduction: None,
                        other_metrics: HashMap::new(),
                    },
                    implementation_steps: vec![],
                    code_examples: None,
                    warnings: vec![],
                    related_suggestions: vec![],
                },
            ],
            summary: OptimizationSummary {
                total_suggestions: 2,
                suggestions_by_category: HashMap::new(),
                suggestions_by_impact: HashMap::new(),
                potential_latency_reduction: 50.0,
                potential_memory_reduction: 0.0,
                potential_throughput_increase: 0.0,
            },
            hardware_info: HardwareInfo::default(),
        };
        let high = report.high_impact_suggestions();
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].id, "high");
    }
    #[test]
    fn test_report_easy_suggestions() {
        let report = OptimizationReport {
            suggestions: vec![
                OptimizationSuggestion {
                    id: "easy".to_string(),
                    category: OptimizationCategory::Memory,
                    impact: ImpactLevel::Low,
                    difficulty: Difficulty::Easy,
                    title: "Easy".to_string(),
                    description: String::new(),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: None,
                        throughput_increase: None,
                        memory_reduction: None,
                        other_metrics: HashMap::new(),
                    },
                    implementation_steps: vec![],
                    code_examples: None,
                    warnings: vec![],
                    related_suggestions: vec![],
                },
                OptimizationSuggestion {
                    id: "hard".to_string(),
                    category: OptimizationCategory::Compute,
                    impact: ImpactLevel::High,
                    difficulty: Difficulty::Hard,
                    title: "Hard".to_string(),
                    description: String::new(),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: None,
                        throughput_increase: None,
                        memory_reduction: None,
                        other_metrics: HashMap::new(),
                    },
                    implementation_steps: vec![],
                    code_examples: None,
                    warnings: vec![],
                    related_suggestions: vec![],
                },
            ],
            summary: OptimizationSummary {
                total_suggestions: 2,
                suggestions_by_category: HashMap::new(),
                suggestions_by_impact: HashMap::new(),
                potential_latency_reduction: 0.0,
                potential_memory_reduction: 0.0,
                potential_throughput_increase: 0.0,
            },
            hardware_info: HardwareInfo::default(),
        };
        let easy = report.easy_suggestions();
        assert_eq!(easy.len(), 1);
        assert_eq!(easy[0].id, "easy");
    }
    #[test]
    fn test_report_suggestions_by_category() {
        let report = OptimizationReport {
            suggestions: vec![
                OptimizationSuggestion {
                    id: "mem1".to_string(),
                    category: OptimizationCategory::Memory,
                    impact: ImpactLevel::Low,
                    difficulty: Difficulty::Easy,
                    title: "Mem1".to_string(),
                    description: String::new(),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: None,
                        throughput_increase: None,
                        memory_reduction: Some(10.0),
                        other_metrics: HashMap::new(),
                    },
                    implementation_steps: vec![],
                    code_examples: None,
                    warnings: vec![],
                    related_suggestions: vec![],
                },
                OptimizationSuggestion {
                    id: "comp1".to_string(),
                    category: OptimizationCategory::Compute,
                    impact: ImpactLevel::High,
                    difficulty: Difficulty::Hard,
                    title: "Comp1".to_string(),
                    description: String::new(),
                    expected_improvement: PerformanceImprovement {
                        latency_reduction: None,
                        throughput_increase: None,
                        memory_reduction: None,
                        other_metrics: HashMap::new(),
                    },
                    implementation_steps: vec![],
                    code_examples: None,
                    warnings: vec![],
                    related_suggestions: vec![],
                },
            ],
            summary: OptimizationSummary {
                total_suggestions: 2,
                suggestions_by_category: HashMap::new(),
                suggestions_by_impact: HashMap::new(),
                potential_latency_reduction: 0.0,
                potential_memory_reduction: 10.0,
                potential_throughput_increase: 0.0,
            },
            hardware_info: HardwareInfo::default(),
        };
        let mem = report.suggestions_by_category(OptimizationCategory::Memory);
        assert_eq!(mem.len(), 1);
        let compute = report.suggestions_by_category(OptimizationCategory::Compute);
        assert_eq!(compute.len(), 1);
        let arch = report.suggestions_by_category(OptimizationCategory::Architecture);
        assert!(arch.is_empty());
    }
    #[test]
    fn test_memory_optimization_rule_no_fragmentation() {
        let rule = MemoryOptimizationRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: Some(MemoryMetrics {
                current_bytes: 1000 * 1024 * 1024,
                peak_bytes: 1200 * 1024 * 1024,
                allocated_bytes: 1500 * 1024 * 1024,
                reserved_bytes: 0,
                num_allocations: 100,
                num_deallocations: 50,
                fragmentation_percent: 2.0,
            }),
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(
            result.is_none(),
            "Should not suggest when fragmentation is low"
        );
    }
    #[test]
    fn test_memory_optimization_rule_high_fragmentation() {
        let rule = MemoryOptimizationRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: Some(MemoryMetrics {
                current_bytes: 500 * 1024 * 1024,
                peak_bytes: 2000 * 1024 * 1024,
                allocated_bytes: 3000 * 1024 * 1024,
                reserved_bytes: 0,
                num_allocations: 1000,
                num_deallocations: 500,
                fragmentation_percent: 30.0,
            }),
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_some());
        let suggestion = result.expect("Expected suggestion");
        assert_eq!(suggestion.id, "memory_fragmentation");
        assert_eq!(suggestion.category, OptimizationCategory::Memory);
    }
    #[test]
    fn test_memory_optimization_rule_no_metrics() {
        let rule = MemoryOptimizationRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_parallelization_rule_few_cores() {
        let rule = ParallelizationRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo {
                cpu_cores: 2,
                ..HardwareInfo::default()
            },
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(
            result.is_none(),
            "Should not suggest parallelization for 2 cores"
        );
    }
    #[test]
    fn test_parallelization_rule_many_cores_already_enabled() {
        let rule = ParallelizationRule;
        let mut config = HashMap::new();
        config.insert("parallel_enabled".to_string(), "true".to_string());
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo {
                cpu_cores: 16,
                ..HardwareInfo::default()
            },
            current_config: config,
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none(), "Should not suggest when already enabled");
    }
    #[test]
    fn test_parallelization_rule_many_cores_not_enabled() {
        let rule = ParallelizationRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo {
                cpu_cores: 16,
                ..HardwareInfo::default()
            },
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_some());
        let suggestion = result.expect("Expected suggestion");
        assert_eq!(suggestion.id, "enable_parallelization");
    }
    #[test]
    fn test_caching_rule_inference_no_cache() {
        let rule = CachingRule;
        let mut config = HashMap::new();
        config.insert("mode".to_string(), "inference".to_string());
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: config,
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_some());
        let suggestion = result.expect("Expected suggestion");
        assert_eq!(suggestion.id, "enable_kv_cache");
    }
    #[test]
    fn test_caching_rule_inference_with_cache() {
        let rule = CachingRule;
        let mut config = HashMap::new();
        config.insert("mode".to_string(), "inference".to_string());
        config.insert("kv_cache_enabled".to_string(), "true".to_string());
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: config,
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_caching_rule_training_mode() {
        let rule = CachingRule;
        let mut config = HashMap::new();
        config.insert("mode".to_string(), "training".to_string());
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: config,
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_batching_rule_batch_1_with_gpu() {
        let rule = BatchingRule;
        let mut config = HashMap::new();
        config.insert("batch_size".to_string(), "1".to_string());
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: Some(LatencyMetrics {
                count: 100,
                mean_ms: 10.0,
                median_ms: 9.0,
                std_dev_ms: 3.0,
                min_ms: 5.0,
                max_ms: 20.0,
                p50_ms: 9.0,
                p90_ms: 15.0,
                p95_ms: 18.0,
                p99_ms: 20.0,
                p999_ms: 22.0,
                window_duration: Duration::from_secs(60),
            }),
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo {
                gpu_model: Some("NVIDIA A100".to_string()),
                ..HardwareInfo::default()
            },
            current_config: config,
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_some());
        let suggestion = result.expect("Expected suggestion");
        assert_eq!(suggestion.id, "increase_batch_size");
    }
    #[test]
    fn test_batching_rule_no_gpu() {
        let rule = BatchingRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: Some(LatencyMetrics {
                count: 100,
                mean_ms: 10.0,
                median_ms: 9.0,
                std_dev_ms: 3.0,
                min_ms: 5.0,
                max_ms: 20.0,
                p50_ms: 9.0,
                p90_ms: 15.0,
                p95_ms: 18.0,
                p99_ms: 20.0,
                p999_ms: 22.0,
                window_duration: Duration::from_secs(60),
            }),
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_mixed_precision_rule_supported_gpu() {
        let rule = MixedPrecisionRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo {
                gpu_model: Some("NVIDIA A100".to_string()),
                ..HardwareInfo::default()
            },
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_some());
        let suggestion = result.expect("Expected suggestion");
        assert_eq!(suggestion.id, "mixed_precision");
    }
    #[test]
    fn test_mixed_precision_rule_already_enabled() {
        let rule = MixedPrecisionRule;
        let mut config = HashMap::new();
        config.insert("mixed_precision".to_string(), "true".to_string());
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo {
                gpu_model: Some("NVIDIA A100".to_string()),
                ..HardwareInfo::default()
            },
            current_config: config,
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_mixed_precision_rule_unsupported_gpu() {
        let rule = MixedPrecisionRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo {
                gpu_model: Some("Old GPU".to_string()),
                ..HardwareInfo::default()
            },
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_kernel_fusion_rule_high_call_count() {
        let rule = KernelFusionRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: Some(ProfileResult {
                name: "test".to_string(),
                total_time: Duration::from_millis(50),
                call_count: 500,
                avg_time: Duration::from_micros(100),
                min_time: Duration::from_micros(50),
                max_time: Duration::from_micros(200),
                self_time: Duration::from_millis(40),
                children: vec![],
                percent_of_parent: 100.0,
            }),
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_some());
        let suggestion = result.expect("Expected suggestion");
        assert_eq!(suggestion.id, "kernel_fusion");
    }
    #[test]
    fn test_kernel_fusion_rule_low_call_count() {
        let rule = KernelFusionRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: Some(ProfileResult {
                name: "test".to_string(),
                total_time: Duration::from_millis(500),
                call_count: 10,
                avg_time: Duration::from_millis(50),
                min_time: Duration::from_millis(30),
                max_time: Duration::from_millis(80),
                self_time: Duration::from_millis(400),
                children: vec![],
                percent_of_parent: 100.0,
            }),
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_gradient_checkpointing_rule_high_memory() {
        let rule = GradientCheckpointingRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: Some(MemoryMetrics {
                current_bytes: 7 * 1024 * 1024 * 1024,
                peak_bytes: 7 * 1024 * 1024 * 1024,
                allocated_bytes: 8 * 1024 * 1024 * 1024,
                reserved_bytes: 0,
                num_allocations: 100,
                num_deallocations: 50,
                fragmentation_percent: 5.0,
            }),
            throughput_metrics: None,
            hardware_info: HardwareInfo {
                system_memory_mb: 8192,
                ..HardwareInfo::default()
            },
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_some());
        let suggestion = result.expect("Expected suggestion");
        assert_eq!(suggestion.id, "gradient_checkpointing");
    }
    #[test]
    fn test_gradient_checkpointing_rule_low_memory() {
        let rule = GradientCheckpointingRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: Some(MemoryMetrics {
                current_bytes: 1024 * 1024 * 1024,
                peak_bytes: 1024 * 1024 * 1024,
                allocated_bytes: 2 * 1024 * 1024 * 1024,
                reserved_bytes: 0,
                num_allocations: 100,
                num_deallocations: 50,
                fragmentation_percent: 5.0,
            }),
            throughput_metrics: None,
            hardware_info: HardwareInfo {
                system_memory_mb: 32768,
                ..HardwareInfo::default()
            },
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_quantization_rule_no_graph() {
        let rule = QuantizationRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_quantization_rule_already_quantized() {
        let rule = QuantizationRule;
        let mut config = HashMap::new();
        config.insert("quantization".to_string(), "true".to_string());
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: config,
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_performance_improvement_all_none() {
        let pi = PerformanceImprovement {
            latency_reduction: None,
            throughput_increase: None,
            memory_reduction: None,
            other_metrics: HashMap::new(),
        };
        assert!(pi.latency_reduction.is_none());
        assert!(pi.throughput_increase.is_none());
        assert!(pi.memory_reduction.is_none());
        assert!(pi.other_metrics.is_empty());
    }
    #[test]
    fn test_performance_improvement_with_values() {
        let mut other = HashMap::new();
        other.insert("custom".to_string(), "value".to_string());
        let pi = PerformanceImprovement {
            latency_reduction: Some(25.0),
            throughput_increase: Some(50.0),
            memory_reduction: Some(30.0),
            other_metrics: other,
        };
        assert!((pi.latency_reduction.expect("expected value") - 25.0).abs() < f32::EPSILON);
        assert!((pi.throughput_increase.expect("expected value") - 50.0).abs() < f32::EPSILON);
        assert!((pi.memory_reduction.expect("expected value") - 30.0).abs() < f32::EPSILON);
        assert_eq!(pi.other_metrics.len(), 1);
    }
    #[test]
    fn test_code_example_creation() {
        let example = CodeExample {
            language: "rust".to_string(),
            code: "let x = 1;".to_string(),
            description: "Simple example".to_string(),
        };
        assert_eq!(example.language, "rust");
        assert!(!example.code.is_empty());
    }
    #[test]
    fn test_optimization_summary_creation() {
        let summary = OptimizationSummary {
            total_suggestions: 5,
            suggestions_by_category: HashMap::from([
                (OptimizationCategory::Memory, 2),
                (OptimizationCategory::Compute, 3),
            ]),
            suggestions_by_impact: HashMap::from([(ImpactLevel::High, 3), (ImpactLevel::Low, 2)]),
            potential_latency_reduction: 35.0,
            potential_memory_reduction: 20.0,
            potential_throughput_increase: 45.0,
        };
        assert_eq!(summary.total_suggestions, 5);
        assert_eq!(
            summary.suggestions_by_category[&OptimizationCategory::Memory],
            2
        );
        assert_eq!(summary.suggestions_by_impact[&ImpactLevel::High], 3);
    }
    #[test]
    fn test_report_markdown_format_detailed() {
        let report = OptimizationReport {
            suggestions: vec![OptimizationSuggestion {
                id: "s1".to_string(),
                category: OptimizationCategory::Compute,
                impact: ImpactLevel::Critical,
                difficulty: Difficulty::Hard,
                title: "Critical Opt".to_string(),
                description: "Critical optimization needed".to_string(),
                expected_improvement: PerformanceImprovement {
                    latency_reduction: Some(60.0),
                    throughput_increase: Some(200.0),
                    memory_reduction: Some(50.0),
                    other_metrics: HashMap::new(),
                },
                implementation_steps: vec!["Step A".to_string(), "Step B".to_string()],
                code_examples: Some(vec![CodeExample {
                    language: "rust".to_string(),
                    code: "fn opt() {}".to_string(),
                    description: "Example code".to_string(),
                }]),
                warnings: vec!["Watch out".to_string()],
                related_suggestions: vec!["s2".to_string()],
            }],
            summary: OptimizationSummary {
                total_suggestions: 1,
                suggestions_by_category: HashMap::new(),
                suggestions_by_impact: HashMap::new(),
                potential_latency_reduction: 60.0,
                potential_memory_reduction: 50.0,
                potential_throughput_increase: 200.0,
            },
            hardware_info: HardwareInfo::default(),
        };
        let md = report.to_markdown();
        assert!(md.contains("Critical Opt"));
        assert!(md.contains("Step A"));
        assert!(md.contains("Step B"));
        assert!(md.contains("Example code"));
        assert!(md.contains("fn opt() {}"));
        assert!(md.contains("Watch out"));
        assert!(md.contains("60.0%"));
    }
    #[test]
    fn test_advisor_suggestions_sorted_by_impact() {
        let advisor = OptimizationAdvisor::new();
        let context = AnalysisContext {
            model_graph: None,
            profile_results: Some(ProfileResult {
                name: "test".to_string(),
                total_time: Duration::from_millis(50),
                call_count: 500,
                avg_time: Duration::from_micros(100),
                min_time: Duration::from_micros(50),
                max_time: Duration::from_micros(200),
                self_time: Duration::from_millis(40),
                children: vec![],
                percent_of_parent: 100.0,
            }),
            latency_metrics: None,
            memory_metrics: Some(MemoryMetrics {
                current_bytes: 500 * 1024 * 1024,
                peak_bytes: 2000 * 1024 * 1024,
                allocated_bytes: 3000 * 1024 * 1024,
                reserved_bytes: 0,
                num_allocations: 1000,
                num_deallocations: 500,
                fragmentation_percent: 30.0,
            }),
            throughput_metrics: None,
            hardware_info: HardwareInfo {
                cpu_cores: 16,
                ..HardwareInfo::default()
            },
            current_config: HashMap::new(),
        };
        let report = advisor.analyze(&context).expect("analyze failed");
        for i in 1..report.suggestions.len() {
            assert!(report.suggestions[i - 1].impact >= report.suggestions[i].impact);
        }
    }
    #[test]
    fn test_flash_attention_rule_no_gpu() {
        let rule = FlashAttentionRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_attention_optimization_rule_no_graph() {
        let rule = AttentionOptimizationRule;
        let context = AnalysisContext {
            model_graph: None,
            profile_results: None,
            latency_metrics: None,
            memory_metrics: None,
            throughput_metrics: None,
            hardware_info: HardwareInfo::default(),
            current_config: HashMap::new(),
        };
        let result = rule.analyze(&context).expect("analyze failed");
        assert!(result.is_none());
    }
}
