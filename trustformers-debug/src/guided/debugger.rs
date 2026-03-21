//! Guided debugging wizard for step-by-step model analysis
//!
//! This module provides a structured debugging approach with automated step detection,
//! progress tracking, and detailed execution guidance for comprehensive model debugging.

use crate::core::session::{DebugConfig, DebugReport, DebugSession};
use crate::interface::simple::{
    QuickAnomalySummary, QuickArchitectureSummary, QuickGradientSummary, QuickHealthSummary,
};
use crate::{MemoryProfilingReport, ProfilerReport};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Guided debugging wizard
pub struct GuidedDebugger {
    session: DebugSession,
    current_step: usize,
    steps: Vec<DebugStep>,
}

/// Debug step in guided debugging
#[derive(Debug)]
pub struct DebugStep {
    pub name: String,
    pub description: String,
    pub action: DebugAction,
    pub expected_time: std::time::Duration,
}

/// Debug action for guided debugging
#[derive(Debug)]
pub enum DebugAction {
    HealthCheck,
    GradientAnalysis,
    ArchitectureAnalysis,
    MemoryProfiling,
    PerformanceProfiling,
    AnomalyDetection,
    ComprehensiveAnalysis,
}

impl Default for GuidedDebugger {
    fn default() -> Self {
        Self::new()
    }
}

impl GuidedDebugger {
    /// Create new guided debugger with automatic step detection
    pub fn new() -> Self {
        let config = DebugConfig::default();
        let session = DebugSession::new(config);

        let steps = vec![
            DebugStep {
                name: "Health Check".to_string(),
                description: "Quick model health assessment".to_string(),
                action: DebugAction::HealthCheck,
                expected_time: std::time::Duration::from_secs(5),
            },
            DebugStep {
                name: "Gradient Analysis".to_string(),
                description: "Analyze gradient flow and stability".to_string(),
                action: DebugAction::GradientAnalysis,
                expected_time: std::time::Duration::from_secs(10),
            },
            DebugStep {
                name: "Architecture Analysis".to_string(),
                description: "Analyze model architecture and efficiency".to_string(),
                action: DebugAction::ArchitectureAnalysis,
                expected_time: std::time::Duration::from_secs(8),
            },
            DebugStep {
                name: "Memory Profiling".to_string(),
                description: "Profile memory usage and detect leaks".to_string(),
                action: DebugAction::MemoryProfiling,
                expected_time: std::time::Duration::from_secs(15),
            },
            DebugStep {
                name: "Performance Profiling".to_string(),
                description: "Analyze computational performance".to_string(),
                action: DebugAction::PerformanceProfiling,
                expected_time: std::time::Duration::from_secs(20),
            },
            DebugStep {
                name: "Anomaly Detection".to_string(),
                description: "Detect numerical anomalies and instabilities".to_string(),
                action: DebugAction::AnomalyDetection,
                expected_time: std::time::Duration::from_secs(12),
            },
        ];

        Self {
            session,
            current_step: 0,
            steps,
        }
    }

    /// Get current step
    pub fn current_step(&self) -> Option<&DebugStep> {
        self.steps.get(self.current_step)
    }

    /// Get total number of steps
    pub fn total_steps(&self) -> usize {
        self.steps.len()
    }

    /// Get progress percentage
    pub fn progress(&self) -> f64 {
        (self.current_step as f64 / self.total_steps() as f64) * 100.0
    }

    /// Execute current step
    pub async fn execute_current_step(&mut self) -> Result<StepResult> {
        if let Some(step) = self.current_step() {
            let start_time = std::time::Instant::now();

            let result = match &step.action {
                DebugAction::HealthCheck => {
                    let summary = self.session.health_checker().quick_health_check().await?;
                    StepResult::Health(summary)
                },
                DebugAction::GradientAnalysis => {
                    let analysis = self.session.gradient_debugger().quick_analysis().await?;
                    let summary =
                        crate::interface::simple::QuickGradientSummary::from_analysis(&analysis);
                    StepResult::Gradient(summary)
                },
                DebugAction::ArchitectureAnalysis => {
                    let summary = self.session.architecture_analyzer().quick_analysis().await?;
                    StepResult::Architecture(summary)
                },
                DebugAction::MemoryProfiling => {
                    if let Some(profiler) = self.session.memory_profiler_mut() {
                        let end_time = std::time::SystemTime::now();
                        let duration_secs = 60.0; // Default duration
                        let profiling_overhead_ms = 0.0; // Default overhead
                        let report = profiler
                            .generate_report(end_time, duration_secs, profiling_overhead_ms)
                            .await?;
                        StepResult::Memory(report)
                    } else {
                        StepResult::Skipped("Memory profiling not enabled".to_string())
                    }
                },
                DebugAction::PerformanceProfiling => {
                    let report = self.session.profiler().generate_report().await?;
                    StepResult::Performance(report)
                },
                DebugAction::AnomalyDetection => {
                    let summary = self.session.anomaly_detector().quick_check().await?;
                    StepResult::Anomaly(summary)
                },
                DebugAction::ComprehensiveAnalysis => {
                    let report = self.session.generate_snapshot().await?;
                    StepResult::Comprehensive(report)
                },
            };

            let _elapsed = start_time.elapsed();
            self.current_step += 1;

            Ok(result)
        } else {
            Err(anyhow::anyhow!("No more steps to execute"))
        }
    }

    /// Skip current step
    pub fn skip_current_step(&mut self) -> Result<()> {
        if self.current_step < self.total_steps() {
            self.current_step += 1;
            Ok(())
        } else {
            Err(anyhow::anyhow!("No more steps to skip"))
        }
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Check if debugging is complete
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.total_steps()
    }
}

/// Result of a debug step
#[derive(Debug, Serialize, Deserialize)]
pub enum StepResult {
    Health(QuickHealthSummary),
    Gradient(QuickGradientSummary),
    Architecture(QuickArchitectureSummary),
    Memory(MemoryProfilingReport),
    Performance(ProfilerReport),
    Anomaly(QuickAnomalySummary),
    Comprehensive(DebugReport),
    Skipped(String),
}
