//! Health checking and diagnostics utilities

use crate::{DebugConfig, DebugSession, QuickDebugLevel, SimplifiedDebugResult};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Health check result structure
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub overall_score: f64,
    pub status: String,
    pub issues: Vec<String>,
    pub critical_issues: bool,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Debug summary structure
#[derive(Debug, Serialize, Deserialize)]
pub struct DebugSummary {
    pub config_hash: String,
    pub total_debug_runs: usize,
    pub total_issues: usize,
    pub critical_issues: usize,
    pub recommendations: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Export format options
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Html,
}

/// Debug template types
#[derive(Debug, Clone)]
pub enum DebugTemplate {
    Development,
    Production,
    Training,
    Research,
}

/// Health checking utilities
pub struct HealthChecker;

impl HealthChecker {
    /// Quick model health check with automatic issue detection
    pub async fn quick_health_check<T>(model: &T) -> Result<HealthCheckResult> {
        let result = crate::quick_debug(model, QuickDebugLevel::Light).await?;

        let health_score = match &result {
            SimplifiedDebugResult::Light(health) => health.score,
            SimplifiedDebugResult::Standard { health, .. } => health.score,
            SimplifiedDebugResult::Deep(report) => {
                let summary = report.summary();
                100.0 - (summary.critical_issues as f64 * 20.0 + summary.total_issues as f64 * 5.0)
            },
            SimplifiedDebugResult::Production(anomaly) => {
                100.0 - (anomaly.anomaly_count as f64 * 10.0)
            },
        };

        Ok(HealthCheckResult {
            overall_score: health_score,
            status: Self::score_to_status(health_score),
            issues: result.recommendations(),
            critical_issues: result.has_critical_issues(),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Convert health score to status string
    pub fn score_to_status(score: f64) -> String {
        match score {
            s if s >= 90.0 => "Excellent".to_string(),
            s if s >= 75.0 => "Good".to_string(),
            s if s >= 50.0 => "Fair".to_string(),
            s if s >= 25.0 => "Poor".to_string(),
            _ => "Critical".to_string(),
        }
    }

    /// Generate debug report summary
    pub fn generate_debug_summary(
        config: &DebugConfig,
        results: &[SimplifiedDebugResult],
    ) -> DebugSummary {
        let mut total_issues = 0;
        let mut critical_issues = 0;
        let mut all_recommendations = Vec::new();

        for result in results {
            match result {
                SimplifiedDebugResult::Light(health) => {
                    if health.score < 50.0 {
                        critical_issues += 1;
                    }
                    total_issues += 1;
                },
                SimplifiedDebugResult::Standard { health, .. } => {
                    if health.score < 50.0 {
                        critical_issues += 1;
                    }
                    total_issues += 1;
                },
                SimplifiedDebugResult::Deep(report) => {
                    let summary = report.summary();
                    total_issues += summary.total_issues;
                    critical_issues += summary.critical_issues;
                },
                SimplifiedDebugResult::Production(anomaly) => {
                    total_issues += anomaly.anomaly_count;
                    if anomaly.severity_level.to_lowercase().contains("critical")
                        || anomaly.severity_level.to_lowercase().contains("high")
                    {
                        critical_issues += 1;
                    }
                },
            }

            all_recommendations.extend(result.recommendations());
        }

        all_recommendations.dedup();

        DebugSummary {
            config_hash: Self::hash_config(config),
            total_debug_runs: results.len(),
            total_issues,
            critical_issues,
            recommendations: all_recommendations,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Export debug data in various formats
    pub async fn export_debug_data(
        session: &DebugSession,
        format: ExportFormat,
        output_path: &str,
    ) -> Result<String> {
        let report = session.generate_snapshot().await?;

        match format {
            ExportFormat::Json => {
                let json_data = serde_json::to_string_pretty(&report)?;
                tokio::fs::write(output_path, json_data).await?;
            },
            ExportFormat::Csv => {
                let csv_data = Self::report_to_csv(&report)?;
                tokio::fs::write(output_path, csv_data).await?;
            },
            ExportFormat::Html => {
                let html_data = Self::report_to_html(&report)?;
                tokio::fs::write(output_path, html_data).await?;
            },
        }

        Ok(format!("Debug data exported to {}", output_path))
    }

    /// Create a debug session template for common use cases
    pub fn create_debug_template(template_type: DebugTemplate) -> DebugConfig {
        match template_type {
            DebugTemplate::Development => DebugConfig {
                enable_tensor_inspection: true,
                enable_gradient_debugging: true,
                enable_model_diagnostics: true,
                enable_visualization: true,
                enable_memory_profiling: true,
                enable_computation_graph_analysis: true,
                max_tracked_tensors: 1000,
                max_gradient_history: 100,
                sampling_rate: 1.0,
                ..Default::default()
            },
            DebugTemplate::Production => DebugConfig {
                enable_tensor_inspection: false,
                enable_gradient_debugging: false,
                enable_model_diagnostics: false,
                enable_visualization: false,
                enable_memory_profiling: true,
                enable_computation_graph_analysis: false,
                max_tracked_tensors: 10,
                max_gradient_history: 10,
                sampling_rate: 0.1,
                ..Default::default()
            },
            DebugTemplate::Training => DebugConfig {
                enable_tensor_inspection: true,
                enable_gradient_debugging: true,
                enable_model_diagnostics: true,
                enable_visualization: false,
                enable_memory_profiling: true,
                enable_computation_graph_analysis: true,
                max_tracked_tensors: 500,
                max_gradient_history: 50,
                sampling_rate: 0.5,
                ..Default::default()
            },
            DebugTemplate::Research => DebugConfig {
                enable_tensor_inspection: true,
                enable_gradient_debugging: true,
                enable_model_diagnostics: true,
                enable_visualization: true,
                enable_memory_profiling: true,
                enable_computation_graph_analysis: true,
                max_tracked_tensors: 2000,
                max_gradient_history: 200,
                sampling_rate: 1.0,
                ..Default::default()
            },
        }
    }

    /// Hash configuration for tracking
    fn hash_config(config: &DebugConfig) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", config).hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Convert report to CSV format
    fn report_to_csv(_report: &crate::DebugReport) -> Result<String> {
        // Simple CSV conversion - in a real implementation this would be more sophisticated
        Ok(format!(
            "timestamp,score,issues\n{},{},{}",
            chrono::Utc::now().to_rfc3339(),
            0, // Placeholder - would extract from report.summary()
            0  // Placeholder - would extract from report.summary()
        ))
    }

    /// Convert report to HTML format
    fn report_to_html(report: &crate::DebugReport) -> Result<String> {
        // Simple HTML conversion - in a real implementation this would be more sophisticated
        Ok(format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>Debug Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .report {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>TrustformeRS Debug Report</h1>
    <div class="report">
        <pre>{}</pre>
    </div>
</body>
</html>
        "#,
            serde_json::to_string_pretty(report)?
        ))
    }
}
