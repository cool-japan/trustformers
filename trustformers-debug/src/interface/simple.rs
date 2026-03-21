//! Simplified debugging interface with one-line functions
//!
//! This module provides convenient debugging functions that require minimal setup.
//! Perfect for quick model analysis and debugging without complex configuration.

use crate::core::session::{DebugConfig, DebugReport, DebugSession};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Quick debugging levels for simplified interface
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuickDebugLevel {
    /// Light debugging - minimal overhead, basic checks
    Light,
    /// Standard debugging - balanced performance and features
    Standard,
    /// Deep debugging - comprehensive analysis, higher overhead
    Deep,
    /// Production debugging - minimal overhead for production use
    Production,
}

/// One-line debugging function with smart defaults
pub async fn quick_debug<T>(_model: &T, level: QuickDebugLevel) -> Result<SimplifiedDebugResult> {
    let config = smart_config_for_level(level);
    let mut session = DebugSession::new(config);

    // Start session
    session.start().await?;

    // Quick analysis based on level
    match level {
        QuickDebugLevel::Light => {
            // Basic health check only
            let health_summary = session.health_checker().quick_health_check().await?;
            session.stop().await?;
            Ok(SimplifiedDebugResult::Light(health_summary))
        },
        QuickDebugLevel::Standard => {
            // Standard analysis: health + gradient + architecture
            let health_summary = session.health_checker().quick_health_check().await?;
            let gradient_analysis = session.gradient_debugger().quick_analysis().await?;
            let gradient_summary = QuickGradientSummary::from_analysis(&gradient_analysis);
            let architecture_summary = session.architecture_analyzer().quick_analysis().await?;
            session.stop().await?;
            Ok(SimplifiedDebugResult::Standard {
                health: health_summary,
                gradients: gradient_summary,
                architecture: architecture_summary,
            })
        },
        QuickDebugLevel::Deep => {
            // Full analysis
            let report = session.stop().await?;
            Ok(SimplifiedDebugResult::Deep(report))
        },
        QuickDebugLevel::Production => {
            // Minimal overhead for production monitoring
            let anomaly_summary = session.anomaly_detector().quick_check().await?;
            session.stop().await?;
            Ok(SimplifiedDebugResult::Production(anomaly_summary))
        },
    }
}

/// Even simpler one-line debugging with automatic level detection
pub async fn debug<T>(model: &T) -> Result<SimplifiedDebugResult> {
    quick_debug(model, QuickDebugLevel::Standard).await
}

/// Smart configuration based on debugging level
fn smart_config_for_level(level: QuickDebugLevel) -> DebugConfig {
    match level {
        QuickDebugLevel::Light => DebugConfig {
            enable_tensor_inspection: false,
            enable_gradient_debugging: false,
            enable_model_diagnostics: false,
            enable_visualization: false,
            enable_memory_profiling: false,
            enable_computation_graph_analysis: false,
            max_tracked_tensors: 100,
            max_gradient_history: 10,
            sampling_rate: 0.1,
            ..Default::default()
        },
        QuickDebugLevel::Standard => DebugConfig {
            enable_tensor_inspection: true,
            enable_gradient_debugging: true,
            enable_model_diagnostics: true,
            enable_visualization: false,
            enable_memory_profiling: false,
            enable_computation_graph_analysis: true,
            max_tracked_tensors: 500,
            max_gradient_history: 50,
            sampling_rate: 0.5,
            ..Default::default()
        },
        QuickDebugLevel::Deep => DebugConfig::default(),
        QuickDebugLevel::Production => DebugConfig {
            enable_tensor_inspection: false,
            enable_gradient_debugging: false,
            enable_model_diagnostics: false,
            enable_visualization: false,
            enable_memory_profiling: false,
            enable_computation_graph_analysis: false,
            max_tracked_tensors: 50,
            max_gradient_history: 5,
            sampling_rate: 0.01,
            ..Default::default()
        },
    }
}

/// Simplified debug result for different levels
#[derive(Debug, Serialize, Deserialize)]
pub enum SimplifiedDebugResult {
    Light(QuickHealthSummary),
    Standard {
        health: QuickHealthSummary,
        gradients: QuickGradientSummary,
        architecture: QuickArchitectureSummary,
    },
    Deep(DebugReport),
    Production(QuickAnomalySummary),
}

impl SimplifiedDebugResult {
    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        match self {
            SimplifiedDebugResult::Light(health) => {
                format!("Health Score: {:.2}/100 ({})", health.score, health.status)
            },
            SimplifiedDebugResult::Standard {
                health,
                gradients,
                architecture,
            } => {
                format!(
                    "Health: {:.2}/100 | Gradients: {} | Architecture: {} parameters",
                    health.score, gradients.status, architecture.total_parameters
                )
            },
            SimplifiedDebugResult::Deep(report) => {
                let summary = report.summary();
                format!(
                    "Issues: {} | Critical: {} | Session: {}",
                    summary.total_issues, summary.critical_issues, summary.session_id
                )
            },
            SimplifiedDebugResult::Production(anomaly) => {
                format!("Anomalies: {} detected", anomaly.anomaly_count)
            },
        }
    }

    /// Check if there are any critical issues
    pub fn has_critical_issues(&self) -> bool {
        match self {
            SimplifiedDebugResult::Light(health) => health.score < 30.0,
            SimplifiedDebugResult::Standard { health, .. } => health.score < 30.0,
            SimplifiedDebugResult::Deep(report) => report.summary().critical_issues > 0,
            SimplifiedDebugResult::Production(anomaly) => anomaly.anomaly_count > 0,
        }
    }

    /// Get quick recommendations
    pub fn recommendations(&self) -> Vec<String> {
        match self {
            SimplifiedDebugResult::Light(health) => health.recommendations.clone(),
            SimplifiedDebugResult::Standard {
                health, gradients, ..
            } => {
                let mut recs = health.recommendations.clone();
                recs.extend(gradients.recommendations.clone());
                recs
            },
            SimplifiedDebugResult::Deep(report) => report.summary().recommendations.clone(),
            SimplifiedDebugResult::Production(anomaly) => anomaly.recommendations.clone(),
        }
    }
}

/// Quick health summary for simplified interface
#[derive(Debug, Serialize, Deserialize)]
pub struct QuickHealthSummary {
    pub score: f64,
    pub status: String,
    pub recommendations: Vec<String>,
}

/// Quick gradient summary for simplified interface
#[derive(Debug, Serialize, Deserialize)]
pub struct QuickGradientSummary {
    pub status: String,
    pub vanishing_risk: f64,
    pub exploding_risk: f64,
    pub recommendations: Vec<String>,
}

impl QuickGradientSummary {
    /// Convert from detailed gradient analysis to simple summary
    pub fn from_analysis(
        analysis: &crate::gradient_debugger::debugger::GradientQuickAnalysis,
    ) -> Self {
        use crate::gradient_debugger::types::LayerHealth;

        let status = match analysis.overall_health {
            LayerHealth::Healthy => "Healthy".to_string(),
            LayerHealth::Warning => "Warning".to_string(),
            LayerHealth::Critical => "Critical".to_string(),
            _ => "Unknown".to_string(),
        };

        // Calculate vanishing and exploding risk based on analysis
        let vanishing_risk = analysis
            .problematic_layers
            .iter()
            .filter(|layer| layer.contains("Vanishing"))
            .count() as f64
            / analysis.active_layers.max(1) as f64;

        let exploding_risk = analysis
            .problematic_layers
            .iter()
            .filter(|layer| layer.contains("Exploding"))
            .count() as f64
            / analysis.active_layers.max(1) as f64;

        let mut recommendations = Vec::new();
        if vanishing_risk > 0.1 {
            recommendations
                .push("Consider using residual connections or skip connections".to_string());
        }
        if exploding_risk > 0.1 {
            recommendations
                .push("Consider gradient clipping or learning rate reduction".to_string());
        }
        if analysis.recent_alerts_count > 0 {
            recommendations.push(format!(
                "Address {} recent gradient alerts",
                analysis.recent_alerts_count
            ));
        }
        if recommendations.is_empty() {
            recommendations.push("Gradients look stable".to_string());
        }

        Self {
            status,
            vanishing_risk,
            exploding_risk,
            recommendations,
        }
    }
}

/// Quick architecture summary for simplified interface
#[derive(Debug, Serialize, Deserialize)]
pub struct QuickArchitectureSummary {
    pub total_parameters: u64,
    pub model_size_mb: f64,
    pub efficiency_score: f64,
    pub recommendations: Vec<String>,
}

/// Quick anomaly summary for simplified interface
#[derive(Debug, Serialize, Deserialize)]
pub struct QuickAnomalySummary {
    pub anomaly_count: usize,
    pub severity_level: String,
    pub recommendations: Vec<String>,
}
