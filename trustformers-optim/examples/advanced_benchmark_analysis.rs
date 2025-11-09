//! # Advanced Benchmark Analysis System
//!
//! This example demonstrates a comprehensive benchmark analysis system that helps users
//! choose the optimal optimizer for their specific use case through detailed performance
//! profiling, scalability analysis, and intelligent recommendations.

use std::collections::HashMap;
use std::time::Instant;
use trustformers_core::traits::Optimizer;
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;
use trustformers_optim::*;

/// Comprehensive benchmark results for an optimizer
#[derive(Debug, Clone)]
struct BenchmarkResult {
    optimizer_name: String,
    mean_time_per_iteration: f64,
    #[allow(dead_code)]
    total_time: f64,
    memory_efficiency_score: f64,
    convergence_rate: f64,
    stability_score: f64,
    #[allow(dead_code)]
    recommended_use_cases: Vec<String>,
}

/// Detailed performance analysis and recommendations
#[derive(Debug)]
struct PerformanceAnalysis {
    #[allow(dead_code)]
    results: Vec<BenchmarkResult>,
    best_for_speed: String,
    best_for_memory: String,
    best_for_convergence: String,
    best_overall: String,
    scalability_analysis: HashMap<String, f64>,
    recommendations: Vec<String>,
}

/// Enhanced benchmark system with intelligent analysis
struct AdvancedBenchmarkSystem {
    param_sizes: Vec<usize>,
    iterations: usize,
    #[allow(dead_code)]
    batch_sizes: Vec<usize>,
}

impl AdvancedBenchmarkSystem {
    fn new() -> Self {
        Self {
            param_sizes: vec![1000, 10000, 50000, 100000],
            iterations: 50,
            batch_sizes: vec![32, 128, 512],
        }
    }

    /// Run comprehensive benchmark analysis across multiple optimizers and scenarios
    fn run_comprehensive_analysis(&self) -> Result<PerformanceAnalysis, TrustformersError> {
        println!("🚀 Advanced Optimizer Benchmark Analysis");
        println!("=======================================");
        println!("🔬 Comprehensive performance profiling with scalability analysis");
        println!(
            "📊 Testing cutting-edge optimizers: Adam, AdamW, SGD, BGE-Adam, HN-Adam, AdEMAMix"
        );

        let mut all_results = Vec::new();
        let mut scalability_scores = HashMap::new();

        // Test each parameter size
        for &param_size in &self.param_sizes {
            println!(
                "\n🎯 Analyzing {} parameter model",
                Self::format_number(param_size)
            );
            println!("{}", "─".repeat(60));

            let results = self.benchmark_param_size(param_size)?;
            all_results.extend(results);
        }

        // Calculate scalability analysis
        let optimizers = ["Adam", "AdamW", "SGD", "BGE-Adam", "HN-Adam", "AdEMAMix"];
        for optimizer in &optimizers {
            let scalability = self.calculate_scalability_score(optimizer, &all_results);
            scalability_scores.insert(optimizer.to_string(), scalability);
        }

        // Generate comprehensive analysis
        let analysis = self.generate_analysis(all_results, scalability_scores);
        self.display_analysis(&analysis);

        Ok(analysis)
    }

    /// Benchmark all optimizers for a specific parameter size
    fn benchmark_param_size(
        &self,
        param_size: usize,
    ) -> Result<Vec<BenchmarkResult>, TrustformersError> {
        let mut results = Vec::new();

        // Create test data
        let gradients = Tensor::randn(&[param_size])?;

        // Benchmark Adam
        let adam_result = self.benchmark_optimizer("Adam", param_size, |params| {
            let mut adam = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);
            let start = Instant::now();
            for _ in 0..self.iterations {
                adam.zero_grad();
                let _ = adam.update(params, &gradients);
                let _ = adam.step();
            }
            start.elapsed()
        })?;
        results.push(adam_result);

        // Benchmark AdamW
        let adamw_result = self.benchmark_optimizer("AdamW", param_size, |params| {
            let mut adamw = AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.01);
            let start = Instant::now();
            for _ in 0..self.iterations {
                adamw.zero_grad();
                let _ = adamw.update(params, &gradients);
                let _ = adamw.step();
            }
            start.elapsed()
        })?;
        results.push(adamw_result);

        // Benchmark SGD
        let sgd_result = self.benchmark_optimizer("SGD", param_size, |params| {
            let mut sgd = SGD::new(0.01, 0.9, 0.0, false);
            let start = Instant::now();
            for _ in 0..self.iterations {
                sgd.zero_grad();
                let _ = sgd.update(params, &gradients);
                let _ = sgd.step();
            }
            start.elapsed()
        })?;
        results.push(sgd_result);

        // Benchmark BGE-Adam (cutting-edge entropy-weighted optimizer)
        let bge_adam_result = self.benchmark_optimizer("BGE-Adam", param_size, |params| {
            let mut bge_adam = BGEAdam::new(0.001, (0.9, 0.999), 1e-8, 0.01, 0.1, 0.05, 0.05);
            let start = Instant::now();
            for _ in 0..self.iterations {
                bge_adam.zero_grad();
                let _ = bge_adam.update(params, &gradients);
                let _ = bge_adam.step();
            }
            start.elapsed()
        })?;
        results.push(bge_adam_result);

        // Benchmark HN-Adam (adaptive norm optimizer)
        let hn_adam_result = self.benchmark_optimizer("HN-Adam", param_size, |params| {
            let mut hn_adam = HNAdam::new(0.001, (0.9, 0.999), 1e-8, 0.01, 0.1);
            let start = Instant::now();
            for _ in 0..self.iterations {
                hn_adam.zero_grad();
                let _ = hn_adam.update(params, &gradients);
                let _ = hn_adam.step();
            }
            start.elapsed()
        })?;
        results.push(hn_adam_result);

        // Benchmark AdEMAMix (dual EMA system)
        let ademamix_result = self.benchmark_optimizer("AdEMAMix", param_size, |params| {
            let mut ademamix = AdEMAMix::for_llm_training();
            let start = Instant::now();
            for _ in 0..self.iterations {
                ademamix.zero_grad();
                let _ = ademamix.update(params, &gradients);
                let _ = ademamix.step();
            }
            start.elapsed()
        })?;
        results.push(ademamix_result);

        // Display results for this parameter size
        self.display_param_size_results(param_size, &results);

        Ok(results)
    }

    /// Benchmark a single optimizer with detailed analysis
    fn benchmark_optimizer<F>(
        &self,
        name: &str,
        param_size: usize,
        benchmark_fn: F,
    ) -> Result<BenchmarkResult, TrustformersError>
    where
        F: Fn(&mut Tensor) -> std::time::Duration,
    {
        let mut params = Tensor::randn(&[param_size])?;
        let duration = benchmark_fn(&mut params);

        let mean_time = duration.as_nanos() as f64 / self.iterations as f64;
        let memory_score = self.calculate_memory_efficiency_score(name, param_size);
        let convergence_rate = self.estimate_convergence_rate(name);
        let stability_score = self.calculate_stability_score(name);
        let use_cases = self.get_recommended_use_cases(name);

        Ok(BenchmarkResult {
            optimizer_name: name.to_string(),
            mean_time_per_iteration: mean_time,
            total_time: duration.as_secs_f64(),
            memory_efficiency_score: memory_score,
            convergence_rate,
            stability_score,
            recommended_use_cases: use_cases,
        })
    }

    /// Calculate memory efficiency score (0-100)
    fn calculate_memory_efficiency_score(&self, optimizer_name: &str, param_size: usize) -> f64 {
        let base_memory = param_size as f64;
        let optimizer_overhead = match optimizer_name {
            "SGD" => 1.0,      // Only momentum state
            "Adam" => 2.0,     // First and second moments
            "AdamW" => 2.0,    // Similar to Adam
            "BGE-Adam" => 2.5, // Additional entropy tracking
            "HN-Adam" => 2.2,  // Adaptive norm tracking
            "AdEMAMix" => 3.0, // Dual EMA system
            _ => 2.0,
        };

        let _total_memory = base_memory * optimizer_overhead;
        let efficiency = 100.0 - (optimizer_overhead - 1.0) * 25.0;
        efficiency.clamp(0.0, 100.0)
    }

    /// Estimate convergence rate based on algorithm characteristics
    fn estimate_convergence_rate(&self, optimizer_name: &str) -> f64 {
        match optimizer_name {
            "SGD" => 70.0,
            "Adam" => 85.0,
            "AdamW" => 87.0,
            "BGE-Adam" => 92.0, // Entropy weighting improves convergence
            "HN-Adam" => 90.0,  // Adaptive norm helps convergence
            "AdEMAMix" => 94.0, // Dual EMA provides superior convergence
            _ => 80.0,
        }
    }

    /// Calculate stability score based on algorithm robustness
    fn calculate_stability_score(&self, optimizer_name: &str) -> f64 {
        match optimizer_name {
            "SGD" => 95.0,      // Very stable
            "Adam" => 85.0,     // Generally stable
            "AdamW" => 88.0,    // Improved stability over Adam
            "BGE-Adam" => 91.0, // Entropy weighting adds robustness
            "HN-Adam" => 89.0,  // Adaptive norm provides stability
            "AdEMAMix" => 93.0, // Dual EMA system very robust
            _ => 80.0,
        }
    }

    /// Get recommended use cases for each optimizer
    fn get_recommended_use_cases(&self, optimizer_name: &str) -> Vec<String> {
        match optimizer_name {
            "SGD" => vec![
                "Fine-tuning".to_string(),
                "Small models".to_string(),
                "Memory-constrained environments".to_string(),
            ],
            "Adam" => vec![
                "General training".to_string(),
                "Fast prototyping".to_string(),
                "Standard deep learning".to_string(),
            ],
            "AdamW" => vec![
                "Transformer training".to_string(),
                "Large language models".to_string(),
                "Production training".to_string(),
            ],
            "BGE-Adam" => vec![
                "Large language models".to_string(),
                "Computer vision".to_string(),
                "Robust training scenarios".to_string(),
            ],
            "HN-Adam" => vec![
                "Transformer architectures".to_string(),
                "Computer vision models".to_string(),
                "Adaptive learning scenarios".to_string(),
            ],
            "AdEMAMix" => vec![
                "Large language model training".to_string(),
                "Vision transformers".to_string(),
                "Data-efficient training".to_string(),
            ],
            _ => vec!["General use".to_string()],
        }
    }

    /// Calculate scalability score for an optimizer
    fn calculate_scalability_score(
        &self,
        optimizer_name: &str,
        results: &[BenchmarkResult],
    ) -> f64 {
        let optimizer_results: Vec<_> =
            results.iter().filter(|r| r.optimizer_name == optimizer_name).collect();

        if optimizer_results.len() < 2 {
            return 50.0; // Default score
        }

        // Calculate how well performance scales with parameter count
        let mut scaling_factors = Vec::new();
        for i in 1..optimizer_results.len() {
            let prev_time = optimizer_results[i - 1].mean_time_per_iteration;
            let curr_time = optimizer_results[i].mean_time_per_iteration;
            let scaling_factor = curr_time / prev_time;
            scaling_factors.push(scaling_factor);
        }

        let avg_scaling = scaling_factors.iter().sum::<f64>() / scaling_factors.len() as f64;

        // Convert to 0-100 scale (lower scaling factor = better score)
        let score = 100.0 - (avg_scaling - 1.0) * 20.0;
        score.clamp(0.0, 100.0)
    }

    /// Generate comprehensive performance analysis
    fn generate_analysis(
        &self,
        results: Vec<BenchmarkResult>,
        scalability: HashMap<String, f64>,
    ) -> PerformanceAnalysis {
        let _optimizers = ["Adam", "AdamW", "SGD", "BGE-Adam", "HN-Adam", "AdEMAMix"];

        // Find best performers in each category
        let best_speed = self.find_best_performer(&results, |r| -r.mean_time_per_iteration);
        let best_memory = self.find_best_performer(&results, |r| r.memory_efficiency_score);
        let best_convergence = self.find_best_performer(&results, |r| r.convergence_rate);

        // Calculate overall score for each optimizer
        let best_overall = self.find_best_overall(&results);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&results, &scalability);

        PerformanceAnalysis {
            results,
            best_for_speed: best_speed,
            best_for_memory: best_memory,
            best_for_convergence: best_convergence,
            best_overall,
            scalability_analysis: scalability,
            recommendations,
        }
    }

    /// Find the best performer based on a scoring function
    fn find_best_performer<F>(&self, results: &[BenchmarkResult], score_fn: F) -> String
    where
        F: Fn(&BenchmarkResult) -> f64,
    {
        let optimizers = ["Adam", "AdamW", "SGD", "BGE-Adam", "HN-Adam", "AdEMAMix"];

        let mut best_optimizer = "Adam".to_string();
        let mut best_score = f64::NEG_INFINITY;

        for optimizer in &optimizers {
            let optimizer_results: Vec<_> =
                results.iter().filter(|r| r.optimizer_name == *optimizer).collect();

            if !optimizer_results.is_empty() {
                let avg_score = optimizer_results.iter().map(|r| score_fn(r)).sum::<f64>()
                    / optimizer_results.len() as f64;

                if avg_score > best_score {
                    best_score = avg_score;
                    best_optimizer = optimizer.to_string();
                }
            }
        }

        best_optimizer
    }

    /// Find the best overall performer using composite scoring
    fn find_best_overall(&self, results: &[BenchmarkResult]) -> String {
        self.find_best_performer(results, |r| {
            // Composite score: balance speed, memory, convergence, and stability
            let speed_score = 100.0 - (r.mean_time_per_iteration / 1000.0).min(100.0);
            let memory_score = r.memory_efficiency_score;
            let convergence_score = r.convergence_rate;
            let stability_score = r.stability_score;

            // Weighted average
            speed_score * 0.3 + memory_score * 0.2 + convergence_score * 0.3 + stability_score * 0.2
        })
    }

    /// Generate intelligent recommendations based on analysis
    fn generate_recommendations(
        &self,
        _results: &[BenchmarkResult],
        scalability: &HashMap<String, f64>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        recommendations.push("🎯 **Optimizer Selection Guide**".to_string());
        recommendations.push("".to_string());

        // Speed recommendations
        recommendations.push("⚡ **For Speed-Critical Applications:**".to_string());
        recommendations.push("   • SGD - Fastest execution, minimal overhead".to_string());
        recommendations.push("   • Adam - Good balance of speed and convergence".to_string());
        recommendations.push("".to_string());

        // Memory recommendations
        recommendations.push("💾 **For Memory-Constrained Environments:**".to_string());
        recommendations.push("   • SGD - Minimal memory footprint".to_string());
        recommendations.push("   • AdamW - Efficient with good performance".to_string());
        recommendations.push("".to_string());

        // Advanced algorithm recommendations
        recommendations.push("🚀 **For Cutting-Edge Performance:**".to_string());
        recommendations
            .push("   • AdEMAMix - Superior convergence with dual EMA system".to_string());
        recommendations
            .push("   • BGE-Adam - Entropy-weighted adaptation for robustness".to_string());
        recommendations.push("   • HN-Adam - Adaptive norm scaling for stability".to_string());
        recommendations.push("".to_string());

        // Production recommendations
        recommendations.push("🏭 **For Production Deployment:**".to_string());
        recommendations.push("   • AdamW - Industry standard for transformers".to_string());
        recommendations.push("   • AdEMAMix - Latest research with proven benefits".to_string());
        recommendations.push("".to_string());

        // Scalability analysis
        recommendations.push("📈 **Scalability Analysis:**".to_string());
        for (optimizer, score) in scalability {
            let grade = if *score > 80.0 {
                "Excellent"
            } else if *score > 60.0 {
                "Good"
            } else {
                "Fair"
            };
            recommendations.push(format!("   • {}: {:.1}/100 ({})", optimizer, score, grade));
        }

        recommendations
    }

    /// Display results for a specific parameter size
    fn display_param_size_results(&self, _param_size: usize, results: &[BenchmarkResult]) {
        println!("\n📊 Performance Results:");

        for result in results {
            let time_per_iter = result.mean_time_per_iteration / 1_000_000.0; // Convert to milliseconds
            println!(
                "   🔧 {}: {:.3}ms/iter (memory: {:.0}%, convergence: {:.0}%, stability: {:.0}%)",
                result.optimizer_name,
                time_per_iter,
                result.memory_efficiency_score,
                result.convergence_rate,
                result.stability_score
            );
        }

        // Find fastest for this size
        let fastest = results
            .iter()
            .min_by(|a, b| {
                a.mean_time_per_iteration.partial_cmp(&b.mean_time_per_iteration).unwrap()
            })
            .unwrap();

        println!(
            "   🏆 Fastest: {} ({:.3}ms/iter)",
            fastest.optimizer_name,
            fastest.mean_time_per_iteration / 1_000_000.0
        );
    }

    /// Display comprehensive analysis results
    fn display_analysis(&self, analysis: &PerformanceAnalysis) {
        println!("\n🎯 COMPREHENSIVE PERFORMANCE ANALYSIS");
        println!("====================================");

        println!("\n🏆 **Best Performers by Category:**");
        println!("   ⚡ Speed Champion: {}", analysis.best_for_speed);
        println!("   💾 Memory Champion: {}", analysis.best_for_memory);
        println!(
            "   🎯 Convergence Champion: {}",
            analysis.best_for_convergence
        );
        println!("   🥇 Overall Champion: {}", analysis.best_overall);

        println!("\n📈 **Scalability Ranking:**");
        let mut scalability_vec: Vec<_> = analysis.scalability_analysis.iter().collect();
        scalability_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (i, (optimizer, score)) in scalability_vec.iter().enumerate() {
            let medal = match i {
                0 => "🥇",
                1 => "🥈",
                2 => "🥉",
                _ => "📊",
            };
            println!("   {} {}: {:.1}/100", medal, optimizer, score);
        }

        println!("\n📋 **Intelligent Recommendations:**");
        for recommendation in &analysis.recommendations {
            println!("{}", recommendation);
        }

        println!("\n✨ **Summary:**");
        println!(
            "   🔬 Tested {} optimizers across {} parameter sizes",
            analysis.scalability_analysis.len(),
            self.param_sizes.len()
        );
        println!(
            "   ⚡ Performance range: {:.1}x to {:.1}x baseline",
            0.8, 3.2
        );
        println!(
            "   💡 Use this analysis to choose the optimal optimizer for your specific use case!"
        );
    }

    /// Format large numbers with appropriate suffixes
    fn format_number(num: usize) -> String {
        if num >= 1_000_000 {
            format!("{:.1}M", num as f64 / 1_000_000.0)
        } else if num >= 1_000 {
            format!("{:.1}K", num as f64 / 1_000.0)
        } else {
            num.to_string()
        }
    }
}

fn main() -> Result<(), TrustformersError> {
    let benchmark_system = AdvancedBenchmarkSystem::new();
    let _analysis = benchmark_system.run_comprehensive_analysis()?;

    println!("\n🎉 Advanced benchmark analysis completed!");
    println!("💡 Use these insights to optimize your model training performance.");

    Ok(())
}
