//! # Comprehensive Distributed Training Benchmarks
//!
//! This example provides a comprehensive benchmarking suite for distributed training
//! with TrustformeRS optimizers. It evaluates:
//!
//! - **Performance Scaling**: How well different optimizers scale across multiple GPUs
//! - **Compression Efficiency**: Impact of different gradient compression algorithms
//! - **Memory Optimization**: Memory usage patterns and optimization effectiveness
//! - **Fault Tolerance**: Recovery performance and training continuity
//! - **Auto-Scaling**: Dynamic scaling performance and cost efficiency
//! - **Real-world Scenarios**: Practical training scenarios with detailed analysis
//!
//! ## Benchmark Categories:
//!
//! 1. **Scaling Benchmarks**: 1-64 GPU scaling tests
//! 2. **Optimizer Comparison**: Performance across different optimizers
//! 3. **Compression Analysis**: Detailed compression algorithm evaluation
//! 4. **Memory Efficiency Tests**: Memory usage optimization benchmarks
//! 5. **Fault Tolerance Tests**: Node failure and recovery benchmarks
//! 6. **Auto-Scaling Evaluation**: Dynamic scaling performance analysis
//! 7. **End-to-End Scenarios**: Complete training pipeline benchmarks
//!
//! ## Usage:
//! ```bash
//! cargo run --example comprehensive_distributed_training_benchmarks --release
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};
use trustformers_core::{errors::Result, Tensor};
use trustformers_optim::{
    adam::Adam, advanced_distributed_features::*, averaged_adam::AveragedAdam,
    enhanced_distributed_training::*, sgd::SGD,
};

fn main() -> Result<()> {
    println!("🚀 TrustformeRS Comprehensive Distributed Training Benchmarks");
    println!("============================================================");
    println!("🔬 Running comprehensive evaluation of distributed training capabilities");
    println!();

    // Initialize benchmark suite
    let mut benchmark_suite = BenchmarkSuite::new();

    // Run all benchmark categories
    benchmark_suite.run_scaling_benchmarks()?;
    benchmark_suite.run_optimizer_comparison_benchmarks()?;
    benchmark_suite.run_compression_analysis_benchmarks()?;
    benchmark_suite.run_memory_efficiency_benchmarks()?;
    benchmark_suite.run_fault_tolerance_benchmarks()?;
    benchmark_suite.run_auto_scaling_benchmarks()?;
    benchmark_suite.run_end_to_end_scenario_benchmarks()?;

    // Generate comprehensive report
    benchmark_suite.generate_comprehensive_report()?;

    println!("\\n🎯 Comprehensive Distributed Training Benchmarks Complete!");
    println!("✨ All performance evaluations completed successfully");

    Ok(())
}

/// Comprehensive benchmark suite for distributed training
pub struct BenchmarkSuite {
    results: BenchmarkResults,
    start_time: Instant,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            results: BenchmarkResults::new(),
            start_time: Instant::now(),
        }
    }

    /// Benchmark 1: Scaling performance across different GPU counts
    pub fn run_scaling_benchmarks(&mut self) -> Result<()> {
        println!("📊 Benchmark 1: GPU Scaling Performance");
        println!("======================================");

        let gpu_counts = vec![1, 2, 4, 8, 16, 32];
        let optimizer = AveragedAdam::for_distributed_training();

        for &gpu_count in &gpu_counts {
            println!("🔍 Testing {}-GPU configuration...", gpu_count);

            let config = DistributedConfig::new()
                .with_gpus(gpu_count)
                .with_gradient_compression(CompressionType::TopK { k: 1000 });

            let scaling_result = self.benchmark_scaling_performance(config, optimizer.clone())?;
            self.results.scaling_results.push(scaling_result);
        }

        self.analyze_scaling_efficiency()?;
        println!();
        Ok(())
    }

    fn benchmark_scaling_performance(
        &self,
        config: DistributedConfig,
        optimizer: AveragedAdam,
    ) -> Result<ScalingBenchmarkResult> {
        let start_time = Instant::now();

        let mut trainer = EnhancedDistributedTrainer::new(config.clone(), optimizer)?;

        // Register model parameters (medium transformer)
        let model_params = create_benchmark_transformer_parameters(512, 12)?; // 12-layer, 512 hidden
        trainer.register_model(model_params)?;

        // Run benchmark training steps
        let benchmark_steps = 50;
        let mut step_times = Vec::new();
        let mut throughput_measurements = Vec::new();

        for step in 1..=benchmark_steps {
            let gradients = create_benchmark_gradients(step, 512)?;
            let step_start = Instant::now();
            let _result = trainer.train_step(gradients)?;
            let step_time = step_start.elapsed();

            step_times.push(step_time);

            if step % 10 == 0 {
                let stats = trainer.get_training_stats();
                throughput_measurements.push(stats.average_throughput);
            }
        }

        let total_time = start_time.elapsed();
        let avg_step_time = step_times.iter().sum::<Duration>() / step_times.len() as u32;
        let final_stats = trainer.get_training_stats();

        Ok(ScalingBenchmarkResult {
            gpu_count: config.num_gpus,
            total_time,
            avg_step_time,
            average_throughput: final_stats.average_throughput,
            gpu_utilization: final_stats.gpu_utilization.iter().sum::<f32>()
                / final_stats.gpu_utilization.len() as f32,
            memory_usage: final_stats.memory_usage.iter().sum::<f32>()
                / final_stats.memory_usage.len() as f32,
            compression_ratio: final_stats.compression_ratio,
            communication_overhead: final_stats.communication_overhead,
            scaling_efficiency: 0.0, // Will be calculated later
        })
    }

    fn analyze_scaling_efficiency(&mut self) -> Result<()> {
        if let Some(baseline) = self.results.scaling_results.first() {
            let baseline_throughput = baseline.average_throughput;

            for result in &mut self.results.scaling_results {
                let theoretical_speedup = result.gpu_count as f32;
                let actual_speedup = result.average_throughput / baseline_throughput;
                result.scaling_efficiency = actual_speedup / theoretical_speedup;

                println!(
                    "   {}-GPU: {:.1}x speedup ({:.1}% efficiency)",
                    result.gpu_count,
                    actual_speedup,
                    result.scaling_efficiency * 100.0
                );
            }
        }

        Ok(())
    }

    /// Benchmark 2: Optimizer comparison across distributed setups
    pub fn run_optimizer_comparison_benchmarks(&mut self) -> Result<()> {
        println!("⚡ Benchmark 2: Optimizer Comparison");
        println!("===================================");

        let optimizers = vec![
            ("AveragedAdam", OptimizerVariant::AveragedAdam),
            ("Adam", OptimizerVariant::Adam),
            ("SGD", OptimizerVariant::SGD),
        ];

        let gpu_count = 8; // Standard configuration for comparison
        let config = DistributedConfig::new()
            .with_gpus(gpu_count)
            .with_gradient_compression(CompressionType::Adaptive);

        for (name, optimizer_type) in optimizers {
            println!("🔍 Testing {} optimizer...", name);

            let result = self.benchmark_optimizer_performance(config.clone(), optimizer_type)?;
            self.results.optimizer_results.push(result);
        }

        self.analyze_optimizer_performance()?;
        println!();
        Ok(())
    }

    fn benchmark_optimizer_performance(
        &self,
        config: DistributedConfig,
        optimizer_type: OptimizerVariant,
    ) -> Result<OptimizerBenchmarkResult> {
        let start_time = Instant::now();

        // Create appropriate trainer based on optimizer type
        let benchmark_steps = 30;
        let mut convergence_metrics = Vec::new();
        #[allow(unused_assignments)]
        let mut memory_efficiency = 0.0;

        match optimizer_type {
            OptimizerVariant::AveragedAdam => {
                let optimizer = AveragedAdam::for_distributed_training();
                let mut trainer = EnhancedDistributedTrainer::new(config, optimizer)?;

                let model_params = create_benchmark_transformer_parameters(512, 12)?;
                trainer.register_model(model_params)?;

                for step in 1..=benchmark_steps {
                    let gradients = create_benchmark_gradients(step, 512)?;
                    trainer.train_step(gradients)?;

                    if step % 10 == 0 {
                        let stats = trainer.get_training_stats();
                        convergence_metrics.push(stats.average_throughput);
                    }
                }

                let final_stats = trainer.get_training_stats();
                memory_efficiency = final_stats.memory_usage.iter().sum::<f32>()
                    / final_stats.memory_usage.len() as f32;
            },
            OptimizerVariant::Adam => {
                let optimizer = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.01);
                let mut trainer = EnhancedDistributedTrainer::new(config, optimizer)?;

                let model_params = create_benchmark_transformer_parameters(512, 12)?;
                trainer.register_model(model_params)?;

                for step in 1..=benchmark_steps {
                    let gradients = create_benchmark_gradients(step, 512)?;
                    trainer.train_step(gradients)?;

                    if step % 10 == 0 {
                        let stats = trainer.get_training_stats();
                        convergence_metrics.push(stats.average_throughput);
                    }
                }

                let final_stats = trainer.get_training_stats();
                memory_efficiency = final_stats.memory_usage.iter().sum::<f32>()
                    / final_stats.memory_usage.len() as f32;
            },
            OptimizerVariant::SGD => {
                let optimizer = SGD::new(0.1, 0.9, 0.0001, true);
                let mut trainer = EnhancedDistributedTrainer::new(config, optimizer)?;

                let model_params = create_benchmark_transformer_parameters(512, 12)?;
                trainer.register_model(model_params)?;

                for step in 1..=benchmark_steps {
                    let gradients = create_benchmark_gradients(step, 512)?;
                    trainer.train_step(gradients)?;

                    if step % 10 == 0 {
                        let stats = trainer.get_training_stats();
                        convergence_metrics.push(stats.average_throughput);
                    }
                }

                let final_stats = trainer.get_training_stats();
                memory_efficiency = final_stats.memory_usage.iter().sum::<f32>()
                    / final_stats.memory_usage.len() as f32;
            },
        }

        let total_time = start_time.elapsed();
        let convergence_rate = if convergence_metrics.len() >= 2 {
            (convergence_metrics.last().unwrap() - convergence_metrics.first().unwrap())
                / convergence_metrics.first().unwrap()
        } else {
            0.0
        };

        Ok(OptimizerBenchmarkResult {
            optimizer_name: optimizer_type.name().to_string(),
            total_time,
            convergence_rate,
            memory_efficiency,
            final_throughput: convergence_metrics.last().copied().unwrap_or(0.0),
            stability_score: self.calculate_stability_score(&convergence_metrics),
        })
    }

    fn calculate_stability_score(&self, metrics: &[f32]) -> f32 {
        if metrics.len() < 2 {
            return 0.0;
        }

        let mean = metrics.iter().sum::<f32>() / metrics.len() as f32;
        let variance =
            metrics.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / metrics.len() as f32;

        1.0 / (1.0 + variance) // Higher score for lower variance
    }

    fn analyze_optimizer_performance(&self) -> Result<()> {
        let mut sorted_results = self.results.optimizer_results.clone();
        sorted_results.sort_by(|a, b| b.final_throughput.partial_cmp(&a.final_throughput).unwrap());

        println!("🏆 Optimizer Performance Ranking:");
        for (i, result) in sorted_results.iter().enumerate() {
            println!(
                "   {}. {}: {:.1} samples/sec, {:.1}% memory, {:.3} stability",
                i + 1,
                result.optimizer_name,
                result.final_throughput,
                result.memory_efficiency * 100.0,
                result.stability_score
            );
        }

        Ok(())
    }

    /// Benchmark 3: Compression algorithm analysis
    pub fn run_compression_analysis_benchmarks(&mut self) -> Result<()> {
        println!("🗜️  Benchmark 3: Compression Algorithm Analysis");
        println!("==============================================");

        let compression_algorithms = vec![
            ("None", CompressionType::None),
            ("TopK-1000", CompressionType::TopK { k: 1000 }),
            ("TopK-500", CompressionType::TopK { k: 500 }),
            ("PowerSGD-32", CompressionType::PowerSGD { rank: 32 }),
            (
                "Quantization-8bit",
                CompressionType::Quantization { bits: 8 },
            ),
            ("OneBitSGD", CompressionType::OneBitSGD),
            ("Adaptive", CompressionType::Adaptive),
        ];

        let gpu_count = 4;
        let optimizer = AveragedAdam::for_distributed_training();

        for (name, compression_type) in compression_algorithms {
            println!("🔍 Testing {} compression...", name);

            let config = DistributedConfig::new()
                .with_gpus(gpu_count)
                .with_gradient_compression(compression_type);

            let result = self.benchmark_compression_performance(config, optimizer.clone())?;
            self.results.compression_results.push(result);
        }

        self.analyze_compression_efficiency()?;
        println!();
        Ok(())
    }

    fn benchmark_compression_performance(
        &self,
        config: DistributedConfig,
        optimizer: AveragedAdam,
    ) -> Result<CompressionBenchmarkResult> {
        let start_time = Instant::now();

        let mut trainer = EnhancedDistributedTrainer::new(config.clone(), optimizer)?;

        let model_params = create_benchmark_transformer_parameters(768, 12)?;
        trainer.register_model(model_params)?;

        let benchmark_steps = 20;
        let mut compression_ratios = Vec::new();
        let mut communication_times = Vec::new();
        let mut throughput_measurements = Vec::new();

        for step in 1..=benchmark_steps {
            let gradients = create_benchmark_gradients(step, 768)?;
            let step_start = Instant::now();
            let result = trainer.train_step(gradients)?;
            let step_time = step_start.elapsed();

            compression_ratios.push(result.compression_ratio);
            communication_times.push(step_time);

            if step % 5 == 0 {
                let stats = trainer.get_training_stats();
                throughput_measurements.push(stats.average_throughput);
            }
        }

        let total_time = start_time.elapsed();
        let avg_compression_ratio =
            compression_ratios.iter().sum::<f32>() / compression_ratios.len() as f32;
        let avg_communication_time =
            communication_times.iter().sum::<Duration>() / communication_times.len() as u32;
        let final_throughput = throughput_measurements.last().copied().unwrap_or(0.0);

        // Calculate bandwidth savings
        let bandwidth_savings = (1.0 - avg_compression_ratio) * 100.0;

        Ok(CompressionBenchmarkResult {
            algorithm_name: self.get_compression_name(&config.compression.algorithm),
            total_time,
            avg_compression_ratio,
            avg_communication_time,
            final_throughput,
            bandwidth_savings,
            accuracy_preservation: self.calculate_accuracy_preservation(avg_compression_ratio),
        })
    }

    fn get_compression_name(&self, compression: &CompressionType) -> String {
        match compression {
            CompressionType::None => "None".to_string(),
            CompressionType::TopK { k } => format!("TopK-{}", k),
            CompressionType::RandomSparsification { ratio } => {
                format!("Random-{:.1}%", ratio * 100.0)
            },
            CompressionType::Quantization { bits } => format!("Quantization-{}bit", bits),
            CompressionType::PowerSGD { rank } => format!("PowerSGD-{}", rank),
            CompressionType::OneBitSGD => "OneBitSGD".to_string(),
            CompressionType::Adaptive => "Adaptive".to_string(),
        }
    }

    fn calculate_accuracy_preservation(&self, compression_ratio: f32) -> f32 {
        // Simplified accuracy preservation model
        if compression_ratio >= 0.9 {
            // No compression
            1.0
        } else if compression_ratio >= 0.5 {
            0.95 - (0.9 - compression_ratio) * 0.5
        } else {
            0.85 - (0.5 - compression_ratio) * 1.0
        }
    }

    fn analyze_compression_efficiency(&self) -> Result<()> {
        let mut sorted_results = self.results.compression_results.clone();
        sorted_results
            .sort_by(|a, b| b.bandwidth_savings.partial_cmp(&a.bandwidth_savings).unwrap());

        println!("🏆 Compression Efficiency Ranking:");
        for (i, result) in sorted_results.iter().enumerate() {
            println!(
                "   {}. {}: {:.1}% bandwidth savings, {:.1} throughput, {:.3} accuracy",
                i + 1,
                result.algorithm_name,
                result.bandwidth_savings,
                result.final_throughput,
                result.accuracy_preservation
            );
        }

        Ok(())
    }

    /// Benchmark 4: Memory efficiency tests
    pub fn run_memory_efficiency_benchmarks(&mut self) -> Result<()> {
        println!("💾 Benchmark 4: Memory Efficiency Analysis");
        println!("=========================================");

        let memory_configurations = vec![
            ("Baseline", MemoryOptimizationConfig::default()),
            (
                "Gradient Checkpointing",
                MemoryOptimizationConfig {
                    gradient_checkpointing: true,
                    ..Default::default()
                },
            ),
            (
                "CPU Offloading",
                MemoryOptimizationConfig {
                    cpu_offloading: true,
                    ..Default::default()
                },
            ),
            (
                "Full Optimization",
                MemoryOptimizationConfig {
                    gradient_checkpointing: true,
                    cpu_offloading: true,
                    auto_gc: true,
                    memory_threshold: 0.85,
                    ..Default::default()
                },
            ),
        ];

        let gpu_count = 4;
        let optimizer = AveragedAdam::for_distributed_training();

        for (name, memory_config) in memory_configurations {
            println!("🔍 Testing {} configuration...", name);

            let mut config = DistributedConfig::new()
                .with_gpus(gpu_count)
                .with_gradient_compression(CompressionType::TopK { k: 1000 });
            config.memory_optimization = memory_config;

            let result = self.benchmark_memory_efficiency(config, optimizer.clone())?;
            self.results.memory_results.push(result);
        }

        self.analyze_memory_optimization()?;
        println!();
        Ok(())
    }

    fn benchmark_memory_efficiency(
        &self,
        config: DistributedConfig,
        optimizer: AveragedAdam,
    ) -> Result<MemoryBenchmarkResult> {
        let start_time = Instant::now();

        let mut trainer = EnhancedDistributedTrainer::new(config.clone(), optimizer)?;

        // Use larger model to stress memory
        let model_params = create_benchmark_transformer_parameters(1024, 24)?; // 24-layer, 1024 hidden
        trainer.register_model(model_params)?;

        let benchmark_steps = 15;
        let mut memory_measurements = Vec::new();
        let mut throughput_measurements = Vec::new();

        for step in 1..=benchmark_steps {
            let gradients = create_benchmark_gradients(step, 1024)?;
            trainer.train_step(gradients)?;

            if step % 5 == 0 {
                let stats = trainer.get_training_stats();
                let avg_memory =
                    stats.memory_usage.iter().sum::<f32>() / stats.memory_usage.len() as f32;
                memory_measurements.push(avg_memory);
                throughput_measurements.push(stats.average_throughput);
            }
        }

        let total_time = start_time.elapsed();
        let peak_memory_usage = memory_measurements.iter().fold(0.0f32, |a, &b| a.max(b));
        let avg_memory_usage =
            memory_measurements.iter().sum::<f32>() / memory_measurements.len() as f32;
        let final_throughput = throughput_measurements.last().copied().unwrap_or(0.0);

        // Calculate memory efficiency score
        let memory_efficiency_score = 1.0 - avg_memory_usage; // Higher score for lower memory usage

        Ok(MemoryBenchmarkResult {
            configuration_name: self.get_memory_config_name(&config.memory_optimization),
            total_time,
            peak_memory_usage,
            avg_memory_usage,
            memory_efficiency_score,
            final_throughput,
            memory_savings: self.calculate_memory_savings(avg_memory_usage),
        })
    }

    fn get_memory_config_name(&self, config: &MemoryOptimizationConfig) -> String {
        if config.gradient_checkpointing && config.cpu_offloading {
            "Full Optimization".to_string()
        } else if config.gradient_checkpointing {
            "Gradient Checkpointing".to_string()
        } else if config.cpu_offloading {
            "CPU Offloading".to_string()
        } else {
            "Baseline".to_string()
        }
    }

    fn calculate_memory_savings(&self, current_usage: f32) -> f32 {
        let baseline_usage = 0.85; // Assume 85% baseline usage
        ((baseline_usage - current_usage) / baseline_usage * 100.0).max(0.0)
    }

    fn analyze_memory_optimization(&self) -> Result<()> {
        let mut sorted_results = self.results.memory_results.clone();
        sorted_results.sort_by(|a, b| {
            b.memory_efficiency_score.partial_cmp(&a.memory_efficiency_score).unwrap()
        });

        println!("🏆 Memory Efficiency Ranking:");
        for (i, result) in sorted_results.iter().enumerate() {
            println!(
                "   {}. {}: {:.1}% peak memory, {:.1}% savings, {:.1} throughput",
                i + 1,
                result.configuration_name,
                result.peak_memory_usage * 100.0,
                result.memory_savings,
                result.final_throughput
            );
        }

        Ok(())
    }

    /// Benchmark 5: Fault tolerance and recovery
    pub fn run_fault_tolerance_benchmarks(&mut self) -> Result<()> {
        println!("🛡️  Benchmark 5: Fault Tolerance Analysis");
        println!("=========================================");

        let fault_scenarios = vec![
            ("No Faults", FaultScenario::NoFaults),
            ("Single Node Failure", FaultScenario::SingleNodeFailure),
            (
                "Multiple Node Failures",
                FaultScenario::MultipleNodeFailures,
            ),
            ("Network Partition", FaultScenario::NetworkPartition),
        ];

        let gpu_count = 8;
        let optimizer = AveragedAdam::for_distributed_training();

        for (name, scenario) in fault_scenarios {
            println!("🔍 Testing {} scenario...", name);

            let mut config =
                DistributedConfig::new().with_gpus(gpu_count).with_fault_tolerance(true);
            config.fault_tolerance.checkpoint_frequency = 5;

            let result = self.benchmark_fault_tolerance(config, optimizer.clone(), scenario)?;
            self.results.fault_tolerance_results.push(result);
        }

        self.analyze_fault_tolerance()?;
        println!();
        Ok(())
    }

    fn benchmark_fault_tolerance(
        &self,
        config: DistributedConfig,
        optimizer: AveragedAdam,
        scenario: FaultScenario,
    ) -> Result<FaultToleranceBenchmarkResult> {
        let start_time = Instant::now();

        let mut trainer = EnhancedDistributedTrainer::new(config, optimizer)?;

        let model_params = create_benchmark_transformer_parameters(512, 12)?;
        trainer.register_model(model_params)?;

        let benchmark_steps = 25;
        let mut recovery_times = Vec::new();
        let mut throughput_before_fault = 0.0;
        let mut throughput_after_recovery = 0.0;

        for step in 1..=benchmark_steps {
            let gradients = create_benchmark_gradients(step, 512)?;
            trainer.train_step(gradients)?;

            // Simulate fault scenarios
            match (&scenario, step) {
                (FaultScenario::SingleNodeFailure, 10) => {
                    println!("   💥 Simulating single node failure...");
                    let recovery_start = Instant::now();
                    throughput_before_fault = trainer.get_training_stats().average_throughput;

                    // Simulate recovery time
                    std::thread::sleep(Duration::from_millis(100)); // Simulated recovery

                    recovery_times.push(recovery_start.elapsed());
                    println!("   ✅ Recovery completed");
                },
                (FaultScenario::MultipleNodeFailures, 15) => {
                    println!("   💥💥 Simulating multiple node failures...");
                    let recovery_start = Instant::now();
                    throughput_before_fault = trainer.get_training_stats().average_throughput;

                    // Simulate longer recovery time for multiple failures
                    std::thread::sleep(Duration::from_millis(200));

                    recovery_times.push(recovery_start.elapsed());
                    println!("   ✅ Recovery completed");
                },
                (FaultScenario::NetworkPartition, 18) => {
                    println!("   🌐 Simulating network partition...");
                    let recovery_start = Instant::now();
                    throughput_before_fault = trainer.get_training_stats().average_throughput;

                    // Simulate network recovery time
                    std::thread::sleep(Duration::from_millis(150));

                    recovery_times.push(recovery_start.elapsed());
                    println!("   ✅ Network recovered");
                },
                _ => {},
            }

            if step > 20 && throughput_after_recovery == 0.0 {
                throughput_after_recovery = trainer.get_training_stats().average_throughput;
            }
        }

        let total_time = start_time.elapsed();
        let avg_recovery_time = if recovery_times.is_empty() {
            Duration::from_secs(0)
        } else {
            recovery_times.iter().sum::<Duration>() / recovery_times.len() as u32
        };

        let recovery_efficiency = if throughput_before_fault > 0.0 {
            throughput_after_recovery / throughput_before_fault
        } else {
            1.0
        };

        Ok(FaultToleranceBenchmarkResult {
            scenario_name: scenario.name().to_string(),
            total_time,
            num_faults: recovery_times.len(),
            avg_recovery_time,
            recovery_efficiency,
            training_continuity: recovery_efficiency > 0.9,
            final_throughput: throughput_after_recovery,
        })
    }

    fn analyze_fault_tolerance(&self) -> Result<()> {
        println!("🏆 Fault Tolerance Analysis:");
        for result in &self.results.fault_tolerance_results {
            println!(
                "   {}: {} faults, {:.2}s recovery, {:.1}% efficiency, continuity: {}",
                result.scenario_name,
                result.num_faults,
                result.avg_recovery_time.as_secs_f32(),
                result.recovery_efficiency * 100.0,
                if result.training_continuity { "✅" } else { "❌" }
            );
        }

        Ok(())
    }

    /// Benchmark 6: Auto-scaling performance
    pub fn run_auto_scaling_benchmarks(&mut self) -> Result<()> {
        println!("📈 Benchmark 6: Auto-Scaling Performance");
        println!("========================================");

        let scaling_strategies = vec![
            ("Performance", ScalingStrategy::Performance),
            ("Cost-Optimized", ScalingStrategy::CostOptimized),
            ("Predictive", ScalingStrategy::Predictive),
        ];

        for (name, strategy) in scaling_strategies {
            println!("🔍 Testing {} scaling strategy...", name);

            let result = self.benchmark_auto_scaling(strategy.clone())?;
            self.results.auto_scaling_results.push(result);
        }

        self.analyze_auto_scaling_efficiency()?;
        println!();
        Ok(())
    }

    fn benchmark_auto_scaling(
        &self,
        strategy: ScalingStrategy,
    ) -> Result<AutoScalingBenchmarkResult> {
        let start_time = Instant::now();

        let auto_scaler_config = AutoScalerConfig {
            min_nodes: 2,
            max_nodes: 16,
            strategy: strategy.clone(),
            scale_up_threshold: 0.85,
            scale_down_threshold: 0.6,
            scaling_cooldown: Duration::from_secs(10), // Shorter for benchmark
            predictive_scaling: true,
            cost_priority: 0.3,
        };

        let mut auto_scaler = AutoScaler::new(auto_scaler_config);

        let benchmark_steps = 50;
        let mut scaling_events = Vec::new();
        let mut node_counts = Vec::new();
        let mut cost_metrics = Vec::new();

        for step in 1..=benchmark_steps {
            // Simulate varying workload
            let workload_intensity = 0.5 + 0.4 * (step as f32 / 10.0).sin().abs();

            let simulated_metrics = PerformanceMetrics {
                throughput: 1000.0 * workload_intensity,
                gpu_utilization: vec![workload_intensity; auto_scaler.get_current_nodes()],
                memory_usage: vec![0.7; auto_scaler.get_current_nodes()],
                communication_overhead: 0.2,
                compression_ratio: 0.1,
                bandwidth_utilization: 0.8,
                step_time: Duration::from_millis((100.0 / workload_intensity) as u64),
            };

            let decision = auto_scaler.update_and_scale(&simulated_metrics)?;

            match decision {
                ScalingDecision::ScaleUp(nodes) => {
                    scaling_events.push(("scale_up".to_string(), nodes));
                },
                ScalingDecision::ScaleDown(nodes) => {
                    scaling_events.push(("scale_down".to_string(), nodes));
                },
                ScalingDecision::NoAction => {},
            }

            node_counts.push(auto_scaler.get_current_nodes());

            // Simulate cost calculation
            let cost = auto_scaler.get_current_nodes() as f32 * 3.0; // $3 per node per hour
            cost_metrics.push(cost);
        }

        let total_time = start_time.elapsed();
        let avg_nodes = node_counts.iter().sum::<usize>() as f32 / node_counts.len() as f32;
        let total_cost = cost_metrics.iter().sum::<f32>();
        let scaling_responsiveness = scaling_events.len() as f32 / benchmark_steps as f32;

        // Calculate cost efficiency (lower cost per unit performance)
        let cost_efficiency = 1000.0 / (total_cost / benchmark_steps as f32); // samples per dollar

        Ok(AutoScalingBenchmarkResult {
            strategy_name: format!("{:?}", strategy),
            total_time,
            num_scaling_events: scaling_events.len(),
            avg_nodes,
            scaling_responsiveness,
            cost_efficiency,
            resource_utilization: 0.8, // Simplified
        })
    }

    fn analyze_auto_scaling_efficiency(&self) -> Result<()> {
        let mut sorted_results = self.results.auto_scaling_results.clone();
        sorted_results.sort_by(|a, b| b.cost_efficiency.partial_cmp(&a.cost_efficiency).unwrap());

        println!("🏆 Auto-Scaling Efficiency Ranking:");
        for (i, result) in sorted_results.iter().enumerate() {
            println!(
                "   {}. {}: {:.1} avg nodes, {} events, {:.2} cost efficiency",
                i + 1,
                result.strategy_name,
                result.avg_nodes,
                result.num_scaling_events,
                result.cost_efficiency
            );
        }

        Ok(())
    }

    /// Benchmark 7: End-to-end scenario benchmarks
    pub fn run_end_to_end_scenario_benchmarks(&mut self) -> Result<()> {
        println!("🌐 Benchmark 7: End-to-End Scenarios");
        println!("====================================");

        let scenarios = vec![
            ("Small-Scale Training", TrainingScenario::SmallScale),
            ("Large-Scale Training", TrainingScenario::LargeScale),
            (
                "Production Inference",
                TrainingScenario::ProductionInference,
            ),
            (
                "Research Experimentation",
                TrainingScenario::ResearchExperimentation,
            ),
        ];

        for (name, scenario) in scenarios {
            println!("🔍 Testing {} scenario...", name);

            let result = self.benchmark_end_to_end_scenario(scenario)?;
            self.results.end_to_end_results.push(result);
        }

        self.analyze_end_to_end_performance()?;
        println!();
        Ok(())
    }

    fn benchmark_end_to_end_scenario(
        &self,
        scenario: TrainingScenario,
    ) -> Result<EndToEndBenchmarkResult> {
        let start_time = Instant::now();

        let (config, optimizer, model_config, steps) = match scenario {
            TrainingScenario::SmallScale => {
                let config = DistributedConfig::new()
                    .with_gpus(2)
                    .with_gradient_compression(CompressionType::TopK { k: 500 });
                let optimizer = AveragedAdam::for_distributed_training();
                let model_config = (256, 6); // 6-layer, 256 hidden
                (config, optimizer, model_config, 20)
            },
            TrainingScenario::LargeScale => {
                let config = DistributedConfig::new()
                    .with_gpus(16)
                    .with_gradient_compression(CompressionType::PowerSGD { rank: 64 })
                    .with_dynamic_batching(true)
                    .with_fault_tolerance(true);
                let optimizer = AveragedAdam::for_large_scale_distributed(16);
                let model_config = (2048, 48); // 48-layer, 2048 hidden
                (config, optimizer, model_config, 30)
            },
            TrainingScenario::ProductionInference => {
                let config = DistributedConfig::new()
                    .with_gpus(4)
                    .with_gradient_compression(CompressionType::Quantization { bits: 8 });
                let optimizer = AveragedAdam::for_distributed_training();
                let model_config = (1024, 12); // 12-layer, 1024 hidden
                (config, optimizer, model_config, 15)
            },
            TrainingScenario::ResearchExperimentation => {
                let config = DistributedConfig::new()
                    .with_gpus(8)
                    .with_gradient_compression(CompressionType::Adaptive)
                    .with_dynamic_batching(true);
                let optimizer = AveragedAdam::for_distributed_training();
                let model_config = (512, 16); // 16-layer, 512 hidden
                (config, optimizer, model_config, 25)
            },
        };

        let mut trainer = EnhancedDistributedTrainer::new(config.clone(), optimizer)?;

        let model_params = create_benchmark_transformer_parameters(model_config.0, model_config.1)?;
        trainer.register_model(model_params)?;

        // Run scenario-specific training
        let mut step_times = Vec::new();
        let mut throughput_measurements = Vec::new();

        for step in 1..=steps {
            let gradients = create_benchmark_gradients(step, model_config.0)?;
            let step_start = Instant::now();
            let _result = trainer.train_step(gradients)?;
            let step_time = step_start.elapsed();

            step_times.push(step_time);

            if step % 5 == 0 {
                let stats = trainer.get_training_stats();
                throughput_measurements.push(stats.average_throughput);
            }
        }

        let total_time = start_time.elapsed();
        let final_stats = trainer.get_training_stats();

        // Calculate scenario-specific metrics
        let resource_efficiency = self.calculate_resource_efficiency(&final_stats, &config);
        let cost_effectiveness = self.calculate_cost_effectiveness(&final_stats, &config);
        let user_experience_score = self.calculate_user_experience_score(&step_times, &final_stats);

        Ok(EndToEndBenchmarkResult {
            scenario_name: scenario.name().to_string(),
            total_time,
            final_throughput: final_stats.average_throughput,
            resource_efficiency,
            cost_effectiveness,
            user_experience_score,
            overall_performance_score: (resource_efficiency
                + cost_effectiveness
                + user_experience_score)
                / 3.0,
        })
    }

    fn calculate_resource_efficiency(
        &self,
        stats: &DistributedTrainingStats,
        _config: &DistributedConfig,
    ) -> f32 {
        let avg_gpu_util =
            stats.gpu_utilization.iter().sum::<f32>() / stats.gpu_utilization.len() as f32;
        let avg_memory_util =
            stats.memory_usage.iter().sum::<f32>() / stats.memory_usage.len() as f32;
        let compression_efficiency = 1.0 - stats.compression_ratio;

        (avg_gpu_util + avg_memory_util + compression_efficiency) / 3.0
    }

    fn calculate_cost_effectiveness(
        &self,
        stats: &DistributedTrainingStats,
        config: &DistributedConfig,
    ) -> f32 {
        let cost_per_sample = (config.num_gpus as f32 * 3.0) / stats.average_throughput; // $3 per GPU hour
        1.0 / (1.0 + cost_per_sample) // Higher score for lower cost per sample
    }

    fn calculate_user_experience_score(
        &self,
        step_times: &[Duration],
        stats: &DistributedTrainingStats,
    ) -> f32 {
        let avg_step_time = step_times.iter().sum::<Duration>() / step_times.len() as u32;
        let step_time_consistency = 1.0 - self.calculate_duration_variance(step_times);
        let stability_score = if stats.bottlenecks.is_empty() { 1.0 } else { 0.7 };

        let responsiveness = 1.0 / (1.0 + avg_step_time.as_secs_f32());

        (step_time_consistency + stability_score + responsiveness) / 3.0
    }

    fn calculate_duration_variance(&self, durations: &[Duration]) -> f32 {
        if durations.len() < 2 {
            return 0.0;
        }

        let values: Vec<f32> = durations.iter().map(|d| d.as_secs_f32()).collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance.sqrt() / mean // Coefficient of variation
    }

    fn analyze_end_to_end_performance(&self) -> Result<()> {
        let mut sorted_results = self.results.end_to_end_results.clone();
        sorted_results.sort_by(|a, b| {
            b.overall_performance_score.partial_cmp(&a.overall_performance_score).unwrap()
        });

        println!("🏆 End-to-End Performance Ranking:");
        for (i, result) in sorted_results.iter().enumerate() {
            println!(
                "   {}. {}: {:.3} overall score ({:.1} throughput, {:.3} efficiency)",
                i + 1,
                result.scenario_name,
                result.overall_performance_score,
                result.final_throughput,
                result.resource_efficiency
            );
        }

        Ok(())
    }

    /// Generate comprehensive benchmark report
    pub fn generate_comprehensive_report(&self) -> Result<()> {
        println!("\\n📊 COMPREHENSIVE BENCHMARK REPORT");
        println!("=================================");

        let total_time = self.start_time.elapsed();

        println!("\\n⏱️  Benchmark Execution Summary:");
        println!(
            "   Total Runtime: {:.1} minutes",
            total_time.as_secs_f32() / 60.0
        );
        println!(
            "   Benchmarks Completed: {}",
            self.count_completed_benchmarks()
        );

        println!("\\n🔝 Top Performers by Category:");

        // Best scaling configuration
        if let Some(best_scaling) = self
            .results
            .scaling_results
            .iter()
            .max_by(|a, b| a.scaling_efficiency.partial_cmp(&b.scaling_efficiency).unwrap())
        {
            println!(
                "   📈 Best Scaling: {}-GPU ({:.1}% efficiency)",
                best_scaling.gpu_count,
                best_scaling.scaling_efficiency * 100.0
            );
        }

        // Best optimizer
        if let Some(best_optimizer) = self
            .results
            .optimizer_results
            .iter()
            .max_by(|a, b| a.final_throughput.partial_cmp(&b.final_throughput).unwrap())
        {
            println!(
                "   ⚡ Best Optimizer: {} ({:.1} samples/sec)",
                best_optimizer.optimizer_name, best_optimizer.final_throughput
            );
        }

        // Best compression
        if let Some(best_compression) = self.results.compression_results.iter().max_by(|a, b| {
            (b.bandwidth_savings * b.accuracy_preservation)
                .partial_cmp(&(a.bandwidth_savings * a.accuracy_preservation))
                .unwrap()
        }) {
            println!(
                "   🗜️  Best Compression: {} ({:.1}% savings, {:.3} accuracy)",
                best_compression.algorithm_name,
                best_compression.bandwidth_savings,
                best_compression.accuracy_preservation
            );
        }

        // Best memory optimization
        if let Some(best_memory) = self.results.memory_results.iter().max_by(|a, b| {
            a.memory_efficiency_score.partial_cmp(&b.memory_efficiency_score).unwrap()
        }) {
            println!(
                "   💾 Best Memory Config: {} ({:.1}% efficiency)",
                best_memory.configuration_name,
                best_memory.memory_efficiency_score * 100.0
            );
        }

        println!("\\n🎯 Key Insights:");
        self.generate_key_insights()?;

        println!("\\n📋 Recommendations:");
        self.generate_recommendations()?;

        println!("\\n✨ Benchmark Report Complete!");

        Ok(())
    }

    fn count_completed_benchmarks(&self) -> usize {
        let mut count = 0;
        if !self.results.scaling_results.is_empty() {
            count += 1;
        }
        if !self.results.optimizer_results.is_empty() {
            count += 1;
        }
        if !self.results.compression_results.is_empty() {
            count += 1;
        }
        if !self.results.memory_results.is_empty() {
            count += 1;
        }
        if !self.results.fault_tolerance_results.is_empty() {
            count += 1;
        }
        if !self.results.auto_scaling_results.is_empty() {
            count += 1;
        }
        if !self.results.end_to_end_results.is_empty() {
            count += 1;
        }
        count
    }

    fn generate_key_insights(&self) -> Result<()> {
        // Scaling insights
        if let Some(best_scaling) = self.results.scaling_results.last() {
            if best_scaling.scaling_efficiency > 0.8 {
                println!(
                    "   ✅ Excellent scaling efficiency achieved up to {} GPUs",
                    best_scaling.gpu_count
                );
            } else {
                println!(
                    "   ⚠️  Scaling efficiency decreases beyond {} GPUs",
                    best_scaling.gpu_count / 2
                );
            }
        }

        // Compression insights
        let avg_compression_savings = self
            .results
            .compression_results
            .iter()
            .map(|r| r.bandwidth_savings)
            .sum::<f32>()
            / self.results.compression_results.len() as f32;
        if avg_compression_savings > 70.0 {
            println!("   ✅ Compression algorithms achieve significant bandwidth savings");
        }

        // Memory insights
        let memory_optimizations_effective =
            self.results.memory_results.iter().any(|r| r.memory_savings > 20.0);
        if memory_optimizations_effective {
            println!("   ✅ Memory optimizations provide substantial memory savings");
        }

        Ok(())
    }

    fn generate_recommendations(&self) -> Result<()> {
        println!("   1. 🎯 For production workloads, use Averaged Adam with PowerSGD compression");
        println!("   2. 🔧 Enable gradient checkpointing for large models to reduce memory usage");
        println!("   3. 📈 Use predictive auto-scaling for variable workloads to optimize costs");
        println!("   4. 🛡️  Implement fault tolerance for critical training runs");
        println!(
            "   5. ⚡ Consider adaptive compression for optimal performance-bandwidth trade-offs"
        );

        Ok(())
    }
}

// Supporting types and implementations

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub scaling_results: Vec<ScalingBenchmarkResult>,
    pub optimizer_results: Vec<OptimizerBenchmarkResult>,
    pub compression_results: Vec<CompressionBenchmarkResult>,
    pub memory_results: Vec<MemoryBenchmarkResult>,
    pub fault_tolerance_results: Vec<FaultToleranceBenchmarkResult>,
    pub auto_scaling_results: Vec<AutoScalingBenchmarkResult>,
    pub end_to_end_results: Vec<EndToEndBenchmarkResult>,
}

impl BenchmarkResults {
    pub fn new() -> Self {
        Self {
            scaling_results: Vec::new(),
            optimizer_results: Vec::new(),
            compression_results: Vec::new(),
            memory_results: Vec::new(),
            fault_tolerance_results: Vec::new(),
            auto_scaling_results: Vec::new(),
            end_to_end_results: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScalingBenchmarkResult {
    pub gpu_count: usize,
    pub total_time: Duration,
    pub avg_step_time: Duration,
    pub average_throughput: f32,
    pub gpu_utilization: f32,
    pub memory_usage: f32,
    pub compression_ratio: f32,
    pub communication_overhead: f32,
    pub scaling_efficiency: f32,
}

#[derive(Debug, Clone)]
pub struct OptimizerBenchmarkResult {
    pub optimizer_name: String,
    pub total_time: Duration,
    pub convergence_rate: f32,
    pub memory_efficiency: f32,
    pub final_throughput: f32,
    pub stability_score: f32,
}

#[derive(Debug, Clone)]
pub struct CompressionBenchmarkResult {
    pub algorithm_name: String,
    pub total_time: Duration,
    pub avg_compression_ratio: f32,
    pub avg_communication_time: Duration,
    pub final_throughput: f32,
    pub bandwidth_savings: f32,
    pub accuracy_preservation: f32,
}

#[derive(Debug, Clone)]
pub struct MemoryBenchmarkResult {
    pub configuration_name: String,
    pub total_time: Duration,
    pub peak_memory_usage: f32,
    pub avg_memory_usage: f32,
    pub memory_efficiency_score: f32,
    pub final_throughput: f32,
    pub memory_savings: f32,
}

#[derive(Debug, Clone)]
pub struct FaultToleranceBenchmarkResult {
    pub scenario_name: String,
    pub total_time: Duration,
    pub num_faults: usize,
    pub avg_recovery_time: Duration,
    pub recovery_efficiency: f32,
    pub training_continuity: bool,
    pub final_throughput: f32,
}

#[derive(Debug, Clone)]
pub struct AutoScalingBenchmarkResult {
    pub strategy_name: String,
    pub total_time: Duration,
    pub num_scaling_events: usize,
    pub avg_nodes: f32,
    pub scaling_responsiveness: f32,
    pub cost_efficiency: f32,
    pub resource_utilization: f32,
}

#[derive(Debug, Clone)]
pub struct EndToEndBenchmarkResult {
    pub scenario_name: String,
    pub total_time: Duration,
    pub final_throughput: f32,
    pub resource_efficiency: f32,
    pub cost_effectiveness: f32,
    pub user_experience_score: f32,
    pub overall_performance_score: f32,
}

#[derive(Debug, Clone)]
pub enum OptimizerVariant {
    AveragedAdam,
    Adam,
    SGD,
}

impl OptimizerVariant {
    pub fn name(&self) -> &str {
        match self {
            OptimizerVariant::AveragedAdam => "AveragedAdam",
            OptimizerVariant::Adam => "Adam",
            OptimizerVariant::SGD => "SGD",
        }
    }
}

#[derive(Debug, Clone)]
pub enum FaultScenario {
    NoFaults,
    SingleNodeFailure,
    MultipleNodeFailures,
    NetworkPartition,
}

impl FaultScenario {
    pub fn name(&self) -> &str {
        match self {
            FaultScenario::NoFaults => "No Faults",
            FaultScenario::SingleNodeFailure => "Single Node Failure",
            FaultScenario::MultipleNodeFailures => "Multiple Node Failures",
            FaultScenario::NetworkPartition => "Network Partition",
        }
    }
}

#[derive(Debug, Clone)]
pub enum TrainingScenario {
    SmallScale,
    LargeScale,
    ProductionInference,
    ResearchExperimentation,
}

impl TrainingScenario {
    pub fn name(&self) -> &str {
        match self {
            TrainingScenario::SmallScale => "Small-Scale Training",
            TrainingScenario::LargeScale => "Large-Scale Training",
            TrainingScenario::ProductionInference => "Production Inference",
            TrainingScenario::ResearchExperimentation => "Research Experimentation",
        }
    }
}

// Utility functions for creating benchmark data

fn create_benchmark_transformer_parameters(
    hidden_size: usize,
    num_layers: usize,
) -> Result<HashMap<String, Tensor>> {
    let mut params = HashMap::new();

    for layer in 0..num_layers {
        // Attention weights
        params.insert(
            format!("layer.{}.attention.q_proj.weight", layer),
            Tensor::randn(&[hidden_size, hidden_size])?,
        );
        params.insert(
            format!("layer.{}.attention.k_proj.weight", layer),
            Tensor::randn(&[hidden_size, hidden_size])?,
        );
        params.insert(
            format!("layer.{}.attention.v_proj.weight", layer),
            Tensor::randn(&[hidden_size, hidden_size])?,
        );
        params.insert(
            format!("layer.{}.attention.out_proj.weight", layer),
            Tensor::randn(&[hidden_size, hidden_size])?,
        );

        // Feed-forward weights
        let ff_size = hidden_size * 4;
        params.insert(
            format!("layer.{}.mlp.fc1.weight", layer),
            Tensor::randn(&[hidden_size, ff_size])?,
        );
        params.insert(
            format!("layer.{}.mlp.fc2.weight", layer),
            Tensor::randn(&[ff_size, hidden_size])?,
        );

        // Layer norms
        params.insert(
            format!("layer.{}.ln_1.weight", layer),
            Tensor::ones(&[hidden_size])?,
        );
        params.insert(
            format!("layer.{}.ln_2.weight", layer),
            Tensor::ones(&[hidden_size])?,
        );
    }

    // Embedding and output layers
    let vocab_size = 50000;
    params.insert(
        "embedding.weight".to_string(),
        Tensor::randn(&[vocab_size, hidden_size])?,
    );
    params.insert(
        "lm_head.weight".to_string(),
        Tensor::randn(&[hidden_size, vocab_size])?,
    );

    Ok(params)
}

fn create_benchmark_gradients(step: usize, hidden_size: usize) -> Result<HashMap<String, Tensor>> {
    let mut gradients = HashMap::new();

    // Create decreasing gradient norms to simulate training progress
    let scale = 0.1 / (1.0 + step as f32 * 0.02);

    gradients.insert(
        "layer.0.attention.q_proj.weight".to_string(),
        Tensor::randn(&[hidden_size, hidden_size])?.scalar_mul(scale)?,
    );
    gradients.insert(
        "layer.0.attention.k_proj.weight".to_string(),
        Tensor::randn(&[hidden_size, hidden_size])?.scalar_mul(scale)?,
    );
    gradients.insert(
        "layer.0.mlp.fc1.weight".to_string(),
        Tensor::randn(&[hidden_size, hidden_size * 4])?.scalar_mul(scale)?,
    );
    gradients.insert(
        "embedding.weight".to_string(),
        Tensor::randn(&[50000, hidden_size])?.scalar_mul(scale * 0.1)?,
    );

    Ok(gradients)
}
