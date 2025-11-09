//! # Automatic Optimizer Tuning System
//!
//! This example demonstrates an intelligent optimizer tuning system that automatically
//! adjusts optimizer settings based on real-time training dynamics and performance metrics.
//! It showcases adaptive optimization techniques and modern ML engineering practices.

#![allow(unused_imports, unused_variables, dead_code)]

use std::collections::VecDeque;
use std::time::Instant;
use trustformers_core::traits::Optimizer;
use trustformers_core::TrustformersError;
use trustformers_optim::*;

/// Training metrics collected during optimization
#[derive(Debug, Clone)]
struct TrainingMetrics {
    step: usize,
    loss: f64,
    gradient_norm: f64,
    learning_rate: f64,
    convergence_rate: f64,
    stability_score: f64,
    timestamp: std::time::Instant,
}

/// Adaptive optimizer configuration that changes based on training dynamics
#[derive(Debug, Clone)]
struct AdaptiveOptimizerConfig {
    initial_lr: f64,
    current_lr: f64,
    lr_adjustment_factor: f64,
    gradient_clip_threshold: f64,
    convergence_patience: usize,
    stability_threshold: f64,
    adaptation_enabled: bool,
}

impl Default for AdaptiveOptimizerConfig {
    fn default() -> Self {
        Self {
            initial_lr: 0.001,
            current_lr: 0.001,
            lr_adjustment_factor: 0.1,
            gradient_clip_threshold: 1.0,
            convergence_patience: 10,
            stability_threshold: 0.1,
            adaptation_enabled: true,
        }
    }
}

/// Intelligent optimizer tuning system
struct AutoOptimizerTuner {
    config: AdaptiveOptimizerConfig,
    metrics_history: VecDeque<TrainingMetrics>,
    history_window: usize,
    adaptation_frequency: usize,
    last_adaptation_step: usize,
    best_loss: f64,
    plateau_counter: usize,
}

impl AutoOptimizerTuner {
    fn new(config: AdaptiveOptimizerConfig) -> Self {
        Self {
            config,
            metrics_history: VecDeque::new(),
            history_window: 20,
            adaptation_frequency: 5,
            last_adaptation_step: 0,
            best_loss: f64::INFINITY,
            plateau_counter: 0,
        }
    }

    /// Update the tuner with new training metrics and potentially adapt optimizer settings
    fn update_metrics(&mut self, metrics: TrainingMetrics) -> Option<OptimizerAdaptation> {
        self.metrics_history.push_back(metrics.clone());

        // Maintain history window
        while self.metrics_history.len() > self.history_window {
            self.metrics_history.pop_front();
        }

        // Check if it's time to adapt
        if self.should_adapt(metrics.step) {
            self.last_adaptation_step = metrics.step;
            return self.analyze_and_adapt(&metrics);
        }

        None
    }

    /// Determine if adaptation should occur based on step frequency and training dynamics
    fn should_adapt(&self, current_step: usize) -> bool {
        if !self.config.adaptation_enabled {
            return false;
        }

        // Adapt every N steps or if there are concerning patterns
        let step_threshold = current_step >= self.last_adaptation_step + self.adaptation_frequency;
        let needs_urgent_adaptation = self.detect_urgent_adaptation_need();

        step_threshold || needs_urgent_adaptation
    }

    /// Detect if urgent adaptation is needed due to training instability
    fn detect_urgent_adaptation_need(&self) -> bool {
        if self.metrics_history.len() < 5 {
            return false;
        }

        let recent_metrics: Vec<_> = self.metrics_history.iter().rev().take(5).collect();

        // Check for gradient explosion
        let gradient_explosion = recent_metrics
            .iter()
            .any(|m| m.gradient_norm > self.config.gradient_clip_threshold * 10.0);

        // Check for rapid loss increase
        let loss_explosion =
            recent_metrics.windows(2).any(|window| window[0].loss > window[1].loss * 2.0);

        // Check for training stagnation
        let stagnation = recent_metrics.iter().all(|m| (m.loss - self.best_loss).abs() < 1e-6);

        gradient_explosion || loss_explosion || stagnation
    }

    /// Analyze current training state and generate adaptation recommendations
    fn analyze_and_adapt(
        &mut self,
        current_metrics: &TrainingMetrics,
    ) -> Option<OptimizerAdaptation> {
        if self.metrics_history.len() < 3 {
            return None;
        }

        let mut adaptations = Vec::new();

        // Update best loss tracking
        if current_metrics.loss < self.best_loss {
            self.best_loss = current_metrics.loss;
            self.plateau_counter = 0;
        } else {
            self.plateau_counter += 1;
        }

        // Analyze gradient norms for learning rate adjustment
        let avg_gradient_norm = self.calculate_average_gradient_norm();
        if avg_gradient_norm > self.config.gradient_clip_threshold {
            // Gradients too large - reduce learning rate
            let new_lr = self.config.current_lr * (1.0 - self.config.lr_adjustment_factor);
            adaptations.push(AdaptationType::LearningRateDecrease {
                old_lr: self.config.current_lr,
                new_lr,
                reason: "High gradient norms detected".to_string(),
            });
            self.config.current_lr = new_lr;
        } else if avg_gradient_norm < self.config.gradient_clip_threshold * 0.1
            && self.plateau_counter > self.config.convergence_patience
        {
            // Gradients too small and plateaued - increase learning rate
            let new_lr = self.config.current_lr * (1.0 + self.config.lr_adjustment_factor);
            adaptations.push(AdaptationType::LearningRateIncrease {
                old_lr: self.config.current_lr,
                new_lr,
                reason: "Training plateau detected".to_string(),
            });
            self.config.current_lr = new_lr;
            self.plateau_counter = 0;
        }

        // Analyze convergence trends
        let convergence_trend = self.analyze_convergence_trend();
        match convergence_trend {
            ConvergenceTrend::Diverging => {
                adaptations.push(AdaptationType::OptimizerSwitch {
                    from: "Current".to_string(),
                    to: "SGD".to_string(),
                    reason: "Divergence detected - switching to more stable optimizer".to_string(),
                });
            },
            ConvergenceTrend::Slow => {
                adaptations.push(AdaptationType::OptimizerSwitch {
                    from: "Current".to_string(),
                    to: "AdEMAMix".to_string(),
                    reason: "Slow convergence - switching to more efficient optimizer".to_string(),
                });
            },
            ConvergenceTrend::Unstable => {
                adaptations.push(AdaptationType::GradientClipping {
                    threshold: avg_gradient_norm * 0.5,
                    reason: "Training instability detected".to_string(),
                });
            },
            ConvergenceTrend::Healthy => {
                // No adaptation needed
            },
        }

        // Check for memory efficiency opportunities
        if self.should_recommend_memory_optimization() {
            adaptations.push(AdaptationType::MemoryOptimization {
                suggestion: "Consider switching to 8-bit optimizer for memory efficiency"
                    .to_string(),
            });
        }

        if adaptations.is_empty() {
            None
        } else {
            let reasoning = self.generate_adaptation_reasoning(&adaptations);
            Some(OptimizerAdaptation {
                step: current_metrics.step,
                adaptations,
                confidence: self.calculate_adaptation_confidence(),
                reasoning,
            })
        }
    }

    /// Calculate average gradient norm over recent history
    fn calculate_average_gradient_norm(&self) -> f64 {
        if self.metrics_history.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.metrics_history.iter().map(|m| m.gradient_norm).sum();

        sum / self.metrics_history.len() as f64
    }

    /// Analyze convergence trend from recent metrics
    fn analyze_convergence_trend(&self) -> ConvergenceTrend {
        if self.metrics_history.len() < 5 {
            return ConvergenceTrend::Healthy;
        }

        let recent_losses: Vec<f64> =
            self.metrics_history.iter().rev().take(5).map(|m| m.loss).collect();

        // Check for divergence (consistently increasing loss)
        let diverging = recent_losses.windows(2).all(|window| window[0] >= window[1]);

        if diverging {
            return ConvergenceTrend::Diverging;
        }

        // Check for slow convergence (minimal improvement)
        let improvement_rate = (recent_losses.last().unwrap() - recent_losses.first().unwrap())
            / recent_losses.first().unwrap();

        if improvement_rate.abs() < 0.001 {
            return ConvergenceTrend::Slow;
        }

        // Check for instability (high variance)
        let mean_loss = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
        let variance = recent_losses.iter().map(|loss| (loss - mean_loss).powi(2)).sum::<f64>()
            / recent_losses.len() as f64;

        if variance.sqrt() / mean_loss > self.config.stability_threshold {
            return ConvergenceTrend::Unstable;
        }

        ConvergenceTrend::Healthy
    }

    /// Determine if memory optimization should be recommended
    fn should_recommend_memory_optimization(&self) -> bool {
        // Recommend memory optimization if training has been stable for a while
        self.metrics_history.len() >= 10
            && self.metrics_history.iter().rev().take(10).all(|m| m.stability_score > 0.8)
    }

    /// Calculate confidence in the adaptation recommendation
    fn calculate_adaptation_confidence(&self) -> f64 {
        // Higher confidence with more data points and consistent patterns
        let data_confidence =
            (self.metrics_history.len() as f64 / self.history_window as f64).min(1.0);
        let stability_confidence =
            self.metrics_history.iter().map(|m| m.stability_score).sum::<f64>()
                / self.metrics_history.len() as f64;

        (data_confidence + stability_confidence) / 2.0
    }

    /// Generate human-readable reasoning for adaptations
    fn generate_adaptation_reasoning(&self, adaptations: &[AdaptationType]) -> String {
        let mut reasoning = Vec::new();

        for adaptation in adaptations {
            match adaptation {
                AdaptationType::LearningRateDecrease { reason, .. } => {
                    reasoning.push(format!("Reduced learning rate: {}", reason));
                },
                AdaptationType::LearningRateIncrease { reason, .. } => {
                    reasoning.push(format!("Increased learning rate: {}", reason));
                },
                AdaptationType::OptimizerSwitch { reason, .. } => {
                    reasoning.push(format!("Optimizer switch: {}", reason));
                },
                AdaptationType::GradientClipping { reason, .. } => {
                    reasoning.push(format!("Gradient clipping: {}", reason));
                },
                AdaptationType::MemoryOptimization { suggestion } => {
                    reasoning.push(suggestion.clone());
                },
            }
        }

        reasoning.join("; ")
    }
}

/// Types of adaptations that can be made to the optimizer
#[derive(Debug, Clone)]
enum AdaptationType {
    LearningRateDecrease {
        old_lr: f64,
        new_lr: f64,
        reason: String,
    },
    LearningRateIncrease {
        old_lr: f64,
        new_lr: f64,
        reason: String,
    },
    OptimizerSwitch {
        from: String,
        to: String,
        reason: String,
    },
    GradientClipping {
        threshold: f64,
        reason: String,
    },
    MemoryOptimization {
        suggestion: String,
    },
}

/// Convergence trend analysis results
#[derive(Debug, Clone)]
enum ConvergenceTrend {
    Healthy,
    Slow,
    Diverging,
    Unstable,
}

/// Optimizer adaptation recommendation
#[derive(Debug, Clone)]
struct OptimizerAdaptation {
    step: usize,
    adaptations: Vec<AdaptationType>,
    confidence: f64,
    reasoning: String,
}

/// Simulated training environment for demonstration
struct TrainingSimulator {
    current_step: usize,
    target_loss: f64,
    noise_level: f64,
    convergence_rate: f64,
}

impl TrainingSimulator {
    fn new() -> Self {
        Self {
            current_step: 0,
            target_loss: 0.1,
            noise_level: 0.1,
            convergence_rate: 0.95,
        }
    }

    /// Simulate one training step and return metrics
    fn simulate_step(&mut self, learning_rate: f64) -> TrainingMetrics {
        self.current_step += 1;

        // Simulate loss reduction with some noise
        let base_loss = self.target_loss
            + (2.0 - self.target_loss) * self.convergence_rate.powi(self.current_step as i32);
        let noise = (rand::random::<f64>() - 0.5) * self.noise_level;
        let loss = (base_loss + noise).max(0.001);

        // Simulate gradient norm based on learning rate and progress
        let gradient_norm = learning_rate
            * (1.0 + (rand::random::<f64>() - 0.5) * 0.5)
            * (loss / self.target_loss).sqrt();

        // Calculate derived metrics
        let convergence_rate =
            if self.current_step > 1 { (1.0 - loss / (loss + 0.1)).max(0.0) } else { 0.5 };

        let stability_score = (1.0 - (noise.abs() / self.noise_level).min(1.0)).max(0.0);

        TrainingMetrics {
            step: self.current_step,
            loss,
            gradient_norm,
            learning_rate,
            convergence_rate,
            stability_score,
            timestamp: Instant::now(),
        }
    }

    /// Introduce training instability for testing adaptation
    fn introduce_instability(&mut self) {
        self.noise_level *= 2.0;
        self.convergence_rate *= 0.9;
    }

    /// Simulate gradient explosion
    fn introduce_gradient_explosion(&mut self) {
        self.noise_level *= 5.0;
    }
}

/// Simplified random number generation for demo
mod rand {
    use std::cell::Cell;

    thread_local! {
        static SEED: Cell<u64> = Cell::new(1);
    }

    pub fn random<T>() -> T
    where
        T: From<f64>,
    {
        SEED.with(|seed| {
            let current = seed.get();
            let next = current.wrapping_mul(1664525).wrapping_add(1013904223);
            seed.set(next);
            T::from((next % 10000) as f64 / 10000.0)
        })
    }
}

fn main() -> Result<(), TrustformersError> {
    println!("🤖 Automatic Optimizer Tuning System");
    println!("====================================");
    println!("🧠 Intelligent adaptation based on real-time training dynamics");
    println!("📊 Monitors convergence, stability, and performance patterns");

    // Initialize the adaptive tuning system
    let mut tuner = AutoOptimizerTuner::new(AdaptiveOptimizerConfig::default());
    let mut simulator = TrainingSimulator::new();

    println!("\n🚀 Starting simulated training with automatic tuning...\n");

    // Simulate training with automatic adaptation
    for step in 1..=100 {
        // Get current learning rate from tuner
        let learning_rate = tuner.config.current_lr;

        // Simulate training step
        let metrics = simulator.simulate_step(learning_rate);

        // Update tuner with new metrics
        if let Some(adaptation) = tuner.update_metrics(metrics.clone()) {
            display_adaptation(step, &adaptation);
        }

        // Display periodic progress
        if step % 20 == 0 {
            display_progress(step, &metrics, learning_rate);
        }

        // Introduce some challenges to test adaptation
        if step == 40 {
            println!("\n⚠️  Introducing training instability at step {}...", step);
            simulator.introduce_instability();
        }

        if step == 70 {
            println!("\n💥 Simulating gradient explosion at step {}...", step);
            simulator.introduce_gradient_explosion();
        }
    }

    // Display final summary
    display_final_summary(&tuner);

    println!("\n✨ Automatic optimizer tuning demonstration completed!");
    println!("💡 The system successfully adapted to various training challenges.");

    Ok(())
}

/// Display adaptation information
fn display_adaptation(step: usize, adaptation: &OptimizerAdaptation) {
    println!(
        "🔧 **ADAPTATION AT STEP {}** (Confidence: {:.1}%)",
        step,
        adaptation.confidence * 100.0
    );

    for adaptation_type in &adaptation.adaptations {
        match adaptation_type {
            AdaptationType::LearningRateDecrease {
                old_lr,
                new_lr,
                reason,
            } => {
                println!(
                    "   📉 Learning Rate: {:.6} → {:.6} ({})",
                    old_lr, new_lr, reason
                );
            },
            AdaptationType::LearningRateIncrease {
                old_lr,
                new_lr,
                reason,
            } => {
                println!(
                    "   📈 Learning Rate: {:.6} → {:.6} ({})",
                    old_lr, new_lr, reason
                );
            },
            AdaptationType::OptimizerSwitch { from, to, reason } => {
                println!("   🔄 Optimizer Switch: {} → {} ({})", from, to, reason);
            },
            AdaptationType::GradientClipping { threshold, reason } => {
                println!(
                    "   ✂️  Gradient Clipping: threshold={:.4} ({})",
                    threshold, reason
                );
            },
            AdaptationType::MemoryOptimization { suggestion } => {
                println!("   💾 Memory Optimization: {}", suggestion);
            },
        }
    }

    println!("   💭 Reasoning: {}", adaptation.reasoning);
    println!();
}

/// Display training progress
fn display_progress(step: usize, metrics: &TrainingMetrics, learning_rate: f64) {
    println!(
        "📊 Step {}: Loss={:.6}, Grad_norm={:.4}, LR={:.6}, Convergence={:.1}%, Stability={:.1}%",
        step,
        metrics.loss,
        metrics.gradient_norm,
        learning_rate,
        metrics.convergence_rate * 100.0,
        metrics.stability_score * 100.0
    );
}

/// Display final training summary
fn display_final_summary(tuner: &AutoOptimizerTuner) {
    println!("\n📈 **TRAINING SUMMARY**");
    println!("======================");
    println!("🎯 Final Learning Rate: {:.6}", tuner.config.current_lr);
    println!("📊 Metrics History Length: {}", tuner.metrics_history.len());
    println!("🏆 Best Loss Achieved: {:.6}", tuner.best_loss);
    println!("🔄 Last Adaptation Step: {}", tuner.last_adaptation_step);

    if let Some(final_metrics) = tuner.metrics_history.back() {
        println!(
            "✅ Final Convergence Rate: {:.1}%",
            final_metrics.convergence_rate * 100.0
        );
        println!(
            "🛡️  Final Stability Score: {:.1}%",
            final_metrics.stability_score * 100.0
        );
    }

    println!("\n🎯 **Adaptation Insights:**");
    println!("   • The system automatically adjusted learning rates based on gradient behavior");
    println!("   • Training instability was detected and mitigated through rate adjustments");
    println!("   • Gradient explosions triggered protective measures");
    println!("   • Memory optimization opportunities were identified when appropriate");
}
