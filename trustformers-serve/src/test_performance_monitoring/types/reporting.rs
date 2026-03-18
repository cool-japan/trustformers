//! Reporting Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Import types from sibling modules
use super::config::{ReportConfig, TestPerformanceMonitoringConfig};
use super::enums::ReportFormat;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportManager {
    export_format: String,
    destination: String,
}

impl ExportManager {
    pub fn new(_config: &TestPerformanceMonitoringConfig) -> Self {
        Self {
            export_format: "json".to_string(),
            destination: "/tmp/exports".to_string(),
        }
    }

    pub fn from_report_config(config: &ReportConfig) -> Self {
        let export_format =
            config.export_formats.first().cloned().unwrap_or_else(|| "json".to_string());

        Self {
            export_format,
            destination: "/tmp/reports".to_string(),
        }
    }

    /// Export a report
    /// TODO: Implement actual report export logic
    pub async fn export_report(
        &self,
        _report: &crate::test_performance_monitoring::reporting::Report,
        _format: ReportFormat,
    ) -> Result<ExportResult, anyhow::Error> {
        Ok(ExportResult {
            export_id: format!("export_{}", chrono::Utc::now().timestamp()),
            file_path: format!("{}/report_stub", self.destination),
            format: self.export_format.clone(),
            file_size: 0,
            exported_at: chrono::Utc::now(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportRequest {
    pub template_id: String,
    pub parameters: HashMap<String, String>,
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub report_id: String,
    pub title: String,
    pub generated_at: DateTime<Utc>,
    pub record_count: usize,
    pub key_findings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateEngine {
    pub template_dir: String,
    pub cache_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateValidator {
    pub validator_id: String,
    pub validation_rules: Vec<String>,
    pub strict_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomTemplate {
    pub template_id: String,
    pub template_name: String,
    pub content: String,
    pub variables: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportStyling {
    pub theme: String,
    pub colors: HashMap<String, String>,
    pub fonts: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportParameter {
    pub param_name: String,
    pub param_type: String,
    pub default_value: Option<String>,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplateMetadata {
    pub template_id: String,
    pub version: String,
    pub author: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionLayout {
    pub layout_type: String,
    pub columns: usize,
    pub spacing: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationEngine {
    pub engine_id: String,
    pub supported_types: Vec<String>,
    pub rendering_options: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub report_id: String,
    pub generated_at: DateTime<Utc>,
    pub generator: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOption {
    pub format: String,
    pub compression: bool,
    pub encryption: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionContent {
    pub content_type: String,
    pub data: serde_json::Value,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Visualization {
    pub viz_id: String,
    pub viz_type: String,
    pub data_source: String,
    pub config: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportInsight {
    pub insight_id: String,
    pub description: String,
    pub importance: f64,
    pub action_items: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub recommendation_id: String,
    pub title: String,
    pub description: String,
    pub priority: i32,
    pub estimated_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    pub export_id: String,
    pub format: String,
    pub file_path: String,
    pub file_size: usize,
    pub exported_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthRecommendation {
    pub recommendation_id: String,
    pub title: String,
    pub description: String,
    pub priority: i32,
    pub estimated_impact: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct ReportSchedule {
    pub schedule_id: String,
    pub cron_expression: String,
    pub enabled: bool,
    pub last_run: Option<DateTime<Utc>>,
}
