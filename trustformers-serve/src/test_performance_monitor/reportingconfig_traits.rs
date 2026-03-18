//! # ReportingConfig - Trait Implementations
//!
//! This module contains trait implementations for `ReportingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    DistributionConfig, EmailDistributionConfig, FilesystemDistributionConfig, ReportFormat,
    ReportSchedule, ReportTemplate, ReportingConfig, TemplateVariable, VariableType,
};

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            schedule: ReportSchedule {
                daily: true,
                weekly: true,
                monthly: false,
                custom_schedules: Vec::new(),
            },
            formats: vec![ReportFormat::Html, ReportFormat::Json],
            distribution: DistributionConfig {
                email: EmailDistributionConfig {
                    enabled: false,
                    recipients: Vec::new(),
                    template: "default".to_string(),
                },
                filesystem: FilesystemDistributionConfig {
                    enabled: true,
                    output_directory: "./reports".to_string(),
                    filename_pattern: "test_performance_report_{date}_{format}".to_string(),
                },
                s3: None,
            },
            templates: vec![ReportTemplate {
                name: "default".to_string(),
                content: include_str!("../../templates/default_report.html").to_string(),
                variables: vec![
                    TemplateVariable {
                        name: "summary".to_string(),
                        variable_type: VariableType::Table,
                        default_value: None,
                    },
                    TemplateVariable {
                        name: "charts".to_string(),
                        variable_type: VariableType::Chart,
                        default_value: None,
                    },
                ],
            }],
        }
    }
}
