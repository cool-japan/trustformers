//! Report Generation System
//!
//! This module provides comprehensive reporting capabilities for test performance monitoring,
//! including report templates, scheduling, data aggregation, and export functionality.

use super::types::*;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use tokio::sync::RwLock;
use trustformers_mobile::network_adaptation::types::TimePeriod;

/// Main reporting system for performance monitoring
#[derive(Debug)]
pub struct ReportingSystem {
    config: ReportConfig,
    template_manager: ReportTemplateManager,
    report_generator: ReportGenerator,
    report_scheduler: ReportScheduler,
    export_manager: ExportManager,
    report_storage: ReportStorage,
}

/// Report template management.
///
/// 0.2.1: dropped a `TemplateValidator` and a `custom_templates` map. The
/// validator was a struct of rule *names* derived from the config with no
/// method that could apply them -- no template was ever validated -- and the
/// custom-template map was never written to or read.
#[derive(Debug, Default)]
pub struct ReportTemplateManager {
    templates: RwLock<HashMap<String, ReportTemplate>>,
}

impl ReportTemplateManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn get_template(&self, template_id: &str) -> Result<ReportTemplate, ReportError> {
        let templates = self.templates.read().await;
        templates
            .get(template_id)
            .cloned()
            .ok_or_else(|| ReportError::TemplateNotFound {
                template_id: template_id.to_string(),
            })
    }
}

/// Report template definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplate {
    pub template_id: String,
    pub template_name: String,
    pub description: String,
    pub report_type: ReportType,
    pub data_sources: Vec<DataSource>,
    pub sections: Vec<ReportSection>,
    pub styling: ReportStyling,
    pub export_formats: Vec<ExportFormat>,
    pub parameters: Vec<ReportParameter>,
    pub metadata: ReportTemplateMetadata,
}

/// Report section configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    pub section_id: String,
    pub section_name: String,
    pub section_type: SectionType,
    pub data_query: DataQuery,
    pub visualization: VisualizationConfig,
    pub layout: SectionLayout,
    pub conditional_display: Option<ConditionalDisplay>,
}

/// Report generation engine.
///
/// 0.2.1: dropped a `DataAggregator`, `VisualizationEngine`, `TemplateEngine`
/// and `ContentProcessor`. Each was a bag of descriptive strings built from the
/// config -- aggregation type, "supported" visualisation names, a template
/// directory, transformation names -- and none of them had a single method, so
/// nothing aggregated, rendered, expanded or transformed anything. Generation
/// is stateless; what it produces is built entirely from the template it is
/// handed.
#[derive(Debug, Default)]
pub struct ReportGenerator;

impl ReportGenerator {
    pub fn new() -> Self {
        Self
    }

    pub async fn generate_from_template(
        &self,
        template: &ReportTemplate,
        parameters: HashMap<String, String>,
        time_period: TimePeriod,
    ) -> Result<GeneratedReport, ReportError> {
        let report_id = format!(
            "report-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        let parameter_summary = if parameters.is_empty() {
            "No parameters applied".to_string()
        } else {
            format!("{} parameters applied", parameters.len())
        };

        let summary = ReportSummary {
            report_id: template.template_id.clone(),
            title: template.template_name.clone(),
            generated_at: Utc::now(),
            record_count: template.sections.len(),
            key_findings: vec![parameter_summary],
        };

        let metadata = ReportMetadata {
            report_id: template.template_id.clone(),
            generated_at: Utc::now(),
            generator: "reporting_system".to_string(),
            version: "1.0".to_string(),
        };

        let export_options = template
            .export_formats
            .iter()
            .map(|format| ExportOption {
                format: format!("{:?}", format),
                compression: false,
                encryption: false,
            })
            .collect();

        Ok(GeneratedReport {
            report_id,
            template_id: template.template_id.clone(),
            title: template.template_name.clone(),
            generated_at: SystemTime::now(),
            time_period,
            sections: Vec::new(),
            summary,
            metadata,
            export_options,
        })
    }
}

/// Generated report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedReport {
    pub report_id: String,
    pub template_id: String,
    pub title: String,
    pub generated_at: SystemTime,
    pub time_period: TimePeriod,
    pub sections: Vec<GeneratedSection>,
    pub summary: ReportSummary,
    pub metadata: ReportMetadata,
    pub export_options: Vec<ExportOption>,
}

/// Generated report section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedSection {
    pub section_id: String,
    pub title: String,
    pub content: SectionContent,
    pub visualizations: Vec<Visualization>,
    pub insights: Vec<ReportInsight>,
    pub recommendations: Vec<Recommendation>,
}

/// Report scheduling system.
///
/// Holds the schedules an operator registered. Nothing here fires them: there
/// is no cron evaluator or timer in this crate, so a registered schedule is a
/// record of intent, not a running job.
///
/// 0.2.1: dropped a `SchedulerEngine` (an id, a concurrency number and the
/// string `"exponential_backoff"`, with no retry code anywhere) and a
/// `ReportNotificationManager` (a list of channel names and an empty throttle
/// map, with nothing that notifies).
#[derive(Debug, Default)]
pub struct ReportScheduler {
    scheduled_reports: RwLock<HashMap<String, ScheduledReport>>,
}

impl ReportScheduler {
    pub fn new() -> Self {
        Self::default()
    }

    /// The schedules registered so far.
    pub async fn schedules(&self) -> Vec<ScheduledReport> {
        self.scheduled_reports.read().await.values().cloned().collect()
    }

    pub async fn add_schedule(
        &self,
        scheduled_report: ScheduledReport,
    ) -> Result<String, ReportError> {
        let schedule_id = scheduled_report.schedule_id.clone();
        let mut schedules = self.scheduled_reports.write().await;
        schedules.insert(schedule_id.clone(), scheduled_report);
        Ok(schedule_id)
    }
}

/// Scheduled report configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledReport {
    pub schedule_id: String,
    pub template_id: String,
    pub schedule_name: String,
    pub cron_expression: String,
    pub recipients: Vec<String>,
    pub parameters: HashMap<String, String>,
    pub output_format: ExportFormat,
    pub delivery_method: DeliveryMethod,
    pub enabled: bool,
    pub last_execution: Option<SystemTime>,
    pub next_execution: SystemTime,
}

impl ReportingSystem {
    /// Create new reporting system
    pub fn new(config: ReportConfig) -> Self {
        Self {
            template_manager: ReportTemplateManager::new(),
            report_generator: ReportGenerator::new(),
            report_scheduler: ReportScheduler::new(),
            export_manager: ExportManager::from_report_config(&config),
            report_storage: ReportStorage::from_report_config(&config),
            config,
        }
    }

    /// The configuration this system was built with.
    pub fn config(&self) -> &ReportConfig {
        &self.config
    }

    /// The schedules registered through [`Self::schedule_report`].
    pub async fn schedules(&self) -> Vec<ScheduledReport> {
        self.report_scheduler.schedules().await
    }

    /// Generate report from template
    pub async fn generate_report(
        &self,
        template_id: &str,
        parameters: HashMap<String, String>,
        time_period: TimePeriod,
    ) -> Result<GeneratedReport, ReportError> {
        let template = self.template_manager.get_template(template_id).await?;
        self.report_generator
            .generate_from_template(&template, parameters, time_period)
            .await
    }

    /// Schedule recurring report
    pub async fn schedule_report(
        &self,
        scheduled_report: ScheduledReport,
    ) -> Result<String, ReportError> {
        self.report_scheduler.add_schedule(scheduled_report).await
    }

    /// Export report to specified format
    pub async fn export_report(
        &self,
        report_id: &str,
        _format: ExportFormat,
    ) -> Result<ExportResult, ReportError> {
        let report = self.report_storage.get_report(report_id).await.map_err(|e| {
            ReportError::DataError {
                source: "report_storage".to_string(),
                reason: format!("{}", e),
            }
        })?;

        // TODO: Convert ExportFormat to ReportFormat properly
        let report_format = ReportFormat::Json; // Using Json as default for now

        let types_result =
            self.export_manager.export_report(&report, report_format).await.map_err(|e| {
                ReportError::DataError {
                    source: "export_manager".to_string(),
                    reason: format!("{}", e),
                }
            })?;

        // Convert types::ExportResult to reporting::ExportResult
        Ok(ExportResult {
            export_path: types_result.file_path,
            format: types_result.format,
            size_bytes: types_result.file_size,
        })
    }
}

/// Report types enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    PerformanceSummary,
    TrendAnalysis,
    ComplianceReport,
    IncidentReport,
    CapacityPlanning,
    CustomReport,
    Summary,
}

/// Report data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub report_id: String,
    pub test_id: String,
    pub report_type: ReportType,
    pub content: String,
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Result of export operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    pub export_path: String,
    pub format: String,
    pub size_bytes: usize,
}

/// Export formats supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    PDF,
    HTML,
    Excel,
    CSV,
    JSON,
    PowerPoint,
}

/// Report errors
#[derive(Debug, Clone)]
pub enum ReportError {
    TemplateNotFound { template_id: String },
    GenerationFailed { reason: String },
    ExportFailed { format: String, reason: String },
    SchedulingError { reason: String },
    DataError { source: String, reason: String },
    ValidationError { field: String, reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reporting_system_creation() {
        let config = ReportConfig::default();
        let _system = ReportingSystem::new(config);

        // Basic creation test - succeeds if no panic
    }

    /// The caller's configuration must survive construction.
    #[test]
    fn system_keeps_the_configuration_it_was_given() {
        let mut config = ReportConfig::default();
        config.auto_generate_interval = Some(std::time::Duration::from_secs(4242));

        let system = ReportingSystem::new(config);

        assert_eq!(
            system.config().auto_generate_interval,
            Some(std::time::Duration::from_secs(4242))
        );
    }

    /// Exporting a report that was never stored must fail, not return a stub.
    #[tokio::test]
    async fn exporting_an_unknown_report_fails() {
        let system = ReportingSystem::new(ReportConfig::default());

        let error = system
            .export_report("no-such-report", ExportFormat::JSON)
            .await
            .expect_err("nothing has ever been stored, so there is nothing to export");
        let rendered = format!("{:?}", error);
        assert!(
            !rendered.contains("stub"),
            "an unknown report must not resolve to a stub"
        );
    }
}
