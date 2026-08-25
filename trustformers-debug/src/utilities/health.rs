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
                Self::health_score_from_counts(summary.critical_issues, summary.total_issues)
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

    /// Convert report to CSV format.
    ///
    /// `score` and `issues` are extracted from `report.summary()` (real
    /// tensor/gradient analysis results), using the same
    /// issues-to-score formula as [`Self::quick_health_check`]'s `Deep`
    /// branch -- never the fabricated `0, 0` this used to emit
    /// unconditionally.
    fn report_to_csv(report: &crate::DebugReport) -> Result<String> {
        let summary = report.summary();
        let score = Self::health_score_from_counts(summary.critical_issues, summary.total_issues);
        Ok(format!(
            "timestamp,score,issues\n{},{:.1},{}",
            chrono::Utc::now().to_rfc3339(),
            score,
            summary.total_issues
        ))
    }

    /// Shared health-score formula: start at 100 and dock points per issue
    /// found (20 per critical issue, 5 per issue overall). Used by both
    /// [`Self::quick_health_check`] and [`Self::report_to_csv`] so the two
    /// never drift apart.
    fn health_score_from_counts(critical_issues: usize, total_issues: usize) -> f64 {
        100.0 - (critical_issues as f64 * 20.0 + total_issues as f64 * 5.0)
    }

    /// Convert report to HTML format
    fn report_to_html(report: &crate::DebugReport) -> Result<String> {
        // Deliberately a single self-contained page with inline CSS and no
        // scripts or assets: it must render from a file:// URL in an offline
        // CI artifact viewer.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_inspector::{
        AlertSeverity as TensorAlertSeverity, TensorAlert, TensorAlertType, TensorInspectionReport,
    };
    use std::collections::HashMap;
    use uuid::Uuid;

    /// Regression test for the `report_to_csv` placeholder: the old code
    /// emitted the literal string `"0,0"` for score/issues unconditionally,
    /// ignoring `report.summary()` entirely. A report with one real critical
    /// (NaN) alert must now produce a real, docked score and a real issues
    /// count -- not the old hardcoded zeros.
    #[tokio::test]
    async fn test_report_to_csv_reflects_real_summary_not_hardcoded_zeros() {
        let mut session = DebugSession::new(DebugConfig::default());
        session.start().await.expect("session should start");
        let mut report = session.stop().await.expect("session should stop");

        report.tensor_report = Some(TensorInspectionReport {
            total_tensors: 1,
            tensors_with_issues: 1,
            total_memory_usage: 0,
            alerts: vec![TensorAlert {
                id: Uuid::new_v4(),
                tensor_id: Uuid::new_v4(),
                tensor_name: "test_tensor".to_string(),
                alert_type: TensorAlertType::NaNValues,
                severity: TensorAlertSeverity::Critical,
                message: "NaN detected".to_string(),
                timestamp: chrono::Utc::now(),
            }],
            comparisons: Vec::new(),
            summary_stats: HashMap::new(),
        });

        let csv = HealthChecker::report_to_csv(&report).expect("csv conversion should succeed");
        let data_row = csv.lines().nth(1).expect("csv must have a data row after the header");
        let fields: Vec<&str> = data_row.split(',').collect();
        assert_eq!(fields.len(), 3, "expected timestamp,score,issues columns");

        let score: f64 = fields[1].parse().expect("score column must be numeric");
        let issues: usize = fields[2].parse().expect("issues column must be numeric");

        // report.summary() sees one NaN alert -> total_issues=1,
        // critical_issues=1 -> score = 100 - (1*20 + 1*5) = 75.0.
        assert_eq!(
            issues, 1,
            "issues must reflect the real NaN alert, not the old hardcoded 0"
        );
        assert_eq!(
            score, 75.0,
            "score must be computed from report.summary(), not the old hardcoded 0"
        );
    }

    #[tokio::test]
    async fn test_report_to_csv_clean_report_has_zero_issues_and_full_score() {
        let mut session = DebugSession::new(DebugConfig::default());
        session.start().await.expect("session should start");
        let report = session.stop().await.expect("session should stop");

        let csv = HealthChecker::report_to_csv(&report).expect("csv conversion should succeed");
        let data_row = csv.lines().nth(1).expect("csv must have a data row after the header");
        let fields: Vec<&str> = data_row.split(',').collect();

        let score: f64 = fields[1].parse().expect("score column must be numeric");
        let issues: usize = fields[2].parse().expect("issues column must be numeric");
        assert_eq!(issues, 0);
        assert_eq!(score, 100.0);
    }
}
