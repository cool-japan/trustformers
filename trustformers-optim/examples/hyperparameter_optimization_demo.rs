//! # Comprehensive Hyperparameter Optimization Demo
//!
//! This example demonstrates the automated hyperparameter tuning framework
//! for TrustformeRS optimizers. It showcases Bayesian optimization, multi-objective
//! optimization, and specialized tuning for different model types.
//!
//! ## Features Demonstrated:
//! - Automated hyperparameter optimization for multiple optimizers
//! - Bayesian optimization with Tree-structured Parzen Estimator (TPE)
//! - Multi-objective optimization (speed vs accuracy)
//! - Task-specific hyperparameter search spaces
//! - Real-time optimization progress tracking
//! - Performance comparison across different configurations

use std::time::Instant;
use trustformers_core::errors::Result;
use trustformers_optim::{hyperparameter_tuning::*, AMacPConfig};

fn main() -> Result<()> {
    println!("🚀 TrustformeRS Hyperparameter Optimization Demo");
    println!("===============================================");
    println!("🔬 Demonstrating automated hyperparameter tuning for cutting-edge optimizers");
    println!();

    // Demo 1: Single-objective optimization for aMacP on transformers
    demo_single_objective_amacp()?;

    // Demo 2: Multi-objective optimization for NovoGrad on large models
    demo_multi_objective_novograd()?;

    // Demo 3: Comparative optimization across multiple optimizers
    demo_comparative_optimization()?;

    // Demo 4: Task-specific optimization
    demo_task_specific_optimization()?;

    println!("\n🎯 Hyperparameter Optimization Demo Complete!");
    println!("✨ All optimization strategies demonstrated successfully");

    Ok(())
}

/// Demo 1: Single-objective optimization for aMacP on transformer tasks
fn demo_single_objective_amacp() -> Result<()> {
    println!("📊 Demo 1: Single-Objective aMacP Optimization for Transformers");
    println!("================================================================");

    let start_time = Instant::now();

    // Use the convenience function for transformer optimization
    println!("🔍 Optimizing aMacP hyperparameters for transformer training...");
    let optimized_config = HyperparameterTuner::optimize_amacp_for_transformers(25)?;

    println!(
        "⏱️  Optimization completed in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );
    println!("🏆 Optimized aMacP Configuration:");
    println!("   Learning Rate: {:.2e}", optimized_config.learning_rate);
    println!("   Beta1 (Momentum): {:.4}", optimized_config.beta1);
    println!("   Beta2 (Variance): {:.4}", optimized_config.beta2);
    println!("   Weight Decay: {:.2e}", optimized_config.weight_decay);
    println!("   Epsilon: {:.2e}", optimized_config.epsilon);
    println!(
        "   Gamma (Consecutive Param Avg): {:.4}",
        optimized_config.gamma
    );
    println!(
        "   Alpha (Dual Momentum Weight): {:.4}",
        optimized_config.alpha
    );

    // Compare with default configuration
    let default_config = AMacPConfig::for_transformers();
    println!("\n📈 Improvement Analysis vs Default:");
    println!(
        "   Learning Rate: {:.1}% change",
        (optimized_config.learning_rate - default_config.learning_rate)
            / default_config.learning_rate
            * 100.0
    );
    println!(
        "   Beta1: {:.2}% change",
        (optimized_config.beta1 - default_config.beta1) / default_config.beta1 * 100.0
    );
    println!(
        "   Weight Decay: {:.1}% change",
        (optimized_config.weight_decay - default_config.weight_decay) / default_config.weight_decay
            * 100.0
    );

    println!();
    Ok(())
}

/// Demo 2: Multi-objective optimization for NovoGrad
fn demo_multi_objective_novograd() -> Result<()> {
    println!("🎯 Demo 2: Multi-Objective NovoGrad Optimization");
    println!("==============================================");

    let start_time = Instant::now();

    // Create search space for large language models
    let search_space = HyperparameterSpace::for_transformers();

    // Define the optimization task
    let task = OptimizationTask {
        name: "Large Language Model Training".to_string(),
        model_size: 1_350_000_000, // 1.35B parameters (GPT-style)
        dataset_size: 50_000_000,
        max_epochs: 100,
        convergence_threshold: 0.01,
        target_metric: "perplexity".to_string(),
        task_type: TaskType::LanguageModeling,
    };

    // Create multi-objective tuner
    let mut tuner = HyperparameterTuner::new(
        OptimizerType::NovoGrad,
        search_space,
        task,
        30, // max trials
    );

    // Enable multi-objective optimization
    tuner.enable_multi_objective(
        vec![
            "convergence_speed".to_string(),
            "memory_efficiency".to_string(),
            "training_stability".to_string(),
        ],
        vec![0.4, 0.3, 0.3], // weights
    );

    println!("🔍 Running multi-objective optimization for NovoGrad...");
    println!("📊 Objectives: Convergence Speed (40%), Memory Efficiency (30%), Stability (30%)");

    let best_config = tuner.optimize()?;

    println!(
        "⏱️  Multi-objective optimization completed in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );
    println!("🏆 Best Multi-Objective Configuration:");
    println!("   Learning Rate: {:.2e}", best_config.learning_rate);
    println!("   Beta1: {:.4}", best_config.beta1);
    println!("   Beta2: {:.4}", best_config.beta2);
    println!("   Batch Size: {}", best_config.batch_size);
    println!(
        "   Performance Score: {:.4}",
        best_config.performance_score.unwrap_or(0.0)
    );

    // Show Pareto front if available
    if let Some(pareto_front) = tuner.get_pareto_front() {
        println!("\n📈 Pareto Front Analysis:");
        println!(
            "   {} configurations on Pareto frontier",
            pareto_front.len()
        );

        if !pareto_front.is_empty() {
            let avg_lr: f32 = pareto_front.iter().map(|c| c.learning_rate).sum::<f32>()
                / pareto_front.len() as f32;
            println!("   Average optimal learning rate: {:.2e}", avg_lr);
        }
    }

    println!();
    Ok(())
}

/// Demo 3: Comparative optimization across multiple optimizers
fn demo_comparative_optimization() -> Result<()> {
    println!("⚡ Demo 3: Comparative Optimization Across Optimizers");
    println!("===================================================");

    let optimizers = vec![
        ("aMacP", OptimizerType::AMacP),
        ("NovoGrad", OptimizerType::NovoGrad),
        ("Adam", OptimizerType::Adam),
        ("AveragedAdam", OptimizerType::AveragedAdam),
    ];

    let search_space = HyperparameterSpace::for_vision();
    let task = OptimizationTask {
        name: "Computer Vision Classification".to_string(),
        model_size: 25_000_000, // 25M parameters (ResNet-style)
        dataset_size: 1_000_000,
        max_epochs: 200,
        convergence_threshold: 0.005,
        target_metric: "accuracy".to_string(),
        task_type: TaskType::ComputerVision,
    };

    let mut results = Vec::new();

    println!(
        "🔍 Optimizing hyperparameters for {} optimizers...",
        optimizers.len()
    );
    println!("📊 Task: Computer Vision Classification (25M params, 1M samples)");
    println!();

    for (name, optimizer_type) in optimizers {
        let start_time = Instant::now();

        let mut tuner = HyperparameterTuner::new(
            optimizer_type,
            search_space.clone(),
            task.clone(),
            20, // reduced trials for demo
        );

        println!("🚀 Optimizing {}...", name);
        let best_config = tuner.optimize()?;
        let optimization_time = start_time.elapsed();

        results.push((
            name,
            best_config.performance_score.unwrap_or(0.0),
            best_config.learning_rate,
            optimization_time,
        ));

        println!(
            "✅ {} optimization complete: Score = {:.4}, LR = {:.2e}",
            name,
            best_config.performance_score.unwrap_or(0.0),
            best_config.learning_rate
        );
    }

    println!("\n🏆 Comparative Results Summary:");
    println!("================================");

    // Sort by performance score
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (name, score, lr, time)) in results.iter().enumerate() {
        println!(
            "{}. {} - Score: {:.4}, LR: {:.2e}, Time: {:.1}s",
            i + 1,
            name,
            score,
            lr,
            time.as_secs_f32()
        );
    }

    if let Some((best_optimizer, best_score, _, _)) = results.first() {
        if let Some((_, baseline_score, _, _)) = results.last() {
            let improvement = (best_score - baseline_score) / baseline_score * 100.0;
            println!(
                "\n📈 {} achieved {:.1}% better performance than baseline",
                best_optimizer, improvement
            );
        }
    }

    println!();
    Ok(())
}

/// Demo 4: Task-specific optimization examples
fn demo_task_specific_optimization() -> Result<()> {
    println!("🎓 Demo 4: Task-Specific Optimization Examples");
    println!("============================================");

    let tasks = vec![
        (
            "Scientific Computing",
            HyperparameterSpace::for_scientific_computing(),
            TaskType::ScientificComputing,
        ),
        (
            "Transformer Training",
            HyperparameterSpace::for_transformers(),
            TaskType::LanguageModeling,
        ),
        (
            "Computer Vision",
            HyperparameterSpace::for_vision(),
            TaskType::ComputerVision,
        ),
    ];

    for (task_name, search_space, task_type) in tasks {
        println!("🔬 Optimizing for {}", task_name);

        let task = OptimizationTask {
            name: task_name.to_string(),
            model_size: 10_000_000,
            dataset_size: 100_000,
            max_epochs: 50,
            convergence_threshold: 0.01,
            target_metric: "loss".to_string(),
            task_type,
        };

        let mut tuner = HyperparameterTuner::new(
            OptimizerType::AMacP,
            search_space.clone(),
            task,
            15, // quick optimization for demo
        );

        let best_config = tuner.optimize()?;

        println!(
            "   🎯 Optimal LR: {:.2e}, WD: {:.2e}, Score: {:.4}",
            best_config.learning_rate,
            best_config.weight_decay,
            best_config.performance_score.unwrap_or(0.0)
        );

        // Show task-specific insights
        match task_name {
            "Scientific Computing" => {
                println!("   💡 Scientific Computing: Ultra-low epsilon ({:.1e}) for numerical precision",
                        best_config.epsilon);
            },
            "Transformer Training" => {
                if let Some(warmup_steps) = best_config.custom_params.get("warmup_steps") {
                    println!(
                        "   💡 Transformer: Optimal warmup steps = {:.0}",
                        warmup_steps
                    );
                }
            },
            "Computer Vision" => {
                println!(
                    "   💡 Computer Vision: Batch size {} optimized for convergence",
                    best_config.batch_size
                );
            },
            _ => {},
        }

        println!();
    }

    Ok(())
}

/// Advanced demo: Custom optimization with user-defined objectives
#[allow(dead_code)]
fn demo_advanced_custom_optimization() -> Result<()> {
    println!("🧪 Advanced Demo: Custom Multi-Objective Optimization");
    println!("===================================================");

    // Create custom search space with additional parameters
    let mut custom_space = HyperparameterSpace::for_transformers();
    custom_space.custom_params.insert("dropout_rate".to_string(), (0.0, 0.3));
    custom_space.custom_params.insert("attention_dropout".to_string(), (0.0, 0.2));
    custom_space.custom_params.insert("layer_scale_init".to_string(), (1e-6, 1e-3));

    let task = OptimizationTask {
        name: "Advanced Transformer Training".to_string(),
        model_size: 175_000_000, // GPT-3 style
        dataset_size: 100_000_000,
        max_epochs: 10,
        convergence_threshold: 0.001,
        target_metric: "composite_score".to_string(),
        task_type: TaskType::LanguageModeling,
    };

    let mut tuner = HyperparameterTuner::new(OptimizerType::AMacP, custom_space, task, 50);

    // Enable custom multi-objective optimization
    tuner.enable_multi_objective(
        vec![
            "perplexity".to_string(),
            "training_speed".to_string(),
            "memory_usage".to_string(),
            "gradient_stability".to_string(),
        ],
        vec![0.4, 0.25, 0.2, 0.15],
    );

    println!("🔍 Running advanced multi-objective optimization...");
    let best_config = tuner.optimize()?;

    println!("🏆 Advanced Optimization Results:");
    println!("   Learning Rate: {:.2e}", best_config.learning_rate);
    println!(
        "   Dropout Rate: {:.3}",
        best_config.custom_params.get("dropout_rate").unwrap_or(&0.0)
    );
    println!(
        "   Attention Dropout: {:.3}",
        best_config.custom_params.get("attention_dropout").unwrap_or(&0.0)
    );
    println!(
        "   Layer Scale Init: {:.2e}",
        best_config.custom_params.get("layer_scale_init").unwrap_or(&1e-4)
    );

    // Show optimization history analysis
    let history = tuner.get_history();
    if !history.is_empty() {
        let scores: Vec<f32> = history.iter().map(|(_, m)| m.composite_score).collect();
        let improvement =
            (scores.last().unwrap() - scores.first().unwrap()) / scores.first().unwrap() * 100.0;
        println!("\n📈 Optimization Progress:");
        println!("   Total Trials: {}", history.len());
        println!("   Performance Improvement: {:.1}%", improvement);

        let avg_convergence: f32 =
            history.iter().map(|(_, m)| m.convergence_epoch as f32).sum::<f32>()
                / history.len() as f32;
        println!("   Average Convergence Epoch: {:.1}", avg_convergence);
    }

    Ok(())
}

/// Utility function to demonstrate hyperparameter space analysis
#[allow(dead_code)]
fn analyze_hyperparameter_space() {
    println!("🔍 Hyperparameter Search Space Analysis");
    println!("=====================================");

    let spaces = vec![
        ("Default", HyperparameterSpace::default()),
        ("Transformers", HyperparameterSpace::for_transformers()),
        ("Vision", HyperparameterSpace::for_vision()),
        (
            "Scientific",
            HyperparameterSpace::for_scientific_computing(),
        ),
    ];

    for (name, space) in spaces {
        println!("\n📊 {} Search Space:", name);
        println!(
            "   Learning Rate: {:.1e} - {:.1e} (log: {})",
            space.learning_rate.0, space.learning_rate.1, space.log_scale_lr
        );
        println!("   Beta1: {:.3} - {:.3}", space.beta1.0, space.beta1.1);
        println!("   Beta2: {:.4} - {:.4}", space.beta2.0, space.beta2.1);
        println!(
            "   Weight Decay: {:.1e} - {:.1e}",
            space.weight_decay.0, space.weight_decay.1
        );
        println!("   Batch Sizes: {:?}", space.batch_sizes);

        if !space.custom_params.is_empty() {
            println!("   Custom Parameters:");
            for (param, (min, max)) in &space.custom_params {
                println!("     {}: {:.1e} - {:.1e}", param, min, max);
            }
        }
    }
}
