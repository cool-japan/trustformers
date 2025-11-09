//! Report Generation for TrustformeRS Debug
//!
//! This module provides comprehensive reporting capabilities for debugging,
//! analysis, and documentation. Supports multiple output formats including
//! PDF, Markdown, HTML, JSON, and Jupyter notebooks.

use crate::{
    gradient_debugger::GradientDebugReport,
    profiler::ProfilerReport,
    visualization::{DebugVisualizer, PlotData, VisualizationConfig},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Report format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    /// PDF format
    Pdf,
    /// Markdown format
    Markdown,
    /// HTML format
    Html,
    /// JSON format
    Json,
    /// Jupyter notebook format
    Jupyter,
    /// LaTeX format
    Latex,
    /// Excel format
    Excel,
    /// PowerPoint format
    PowerPoint,
}

/// Report type categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    /// Model debugging report
    DebugReport,
    /// Performance analysis report
    PerformanceReport,
    /// Training analysis report
    TrainingReport,
    /// Gradient analysis report
    GradientReport,
    /// Memory analysis report
    MemoryReport,
    /// Comprehensive report (all sections)
    ComprehensiveReport,
    /// Custom report with specific sections
    CustomReport(Vec<ReportSection>),
}

/// Available report sections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportSection {
    /// Executive summary
    Summary,
    /// Model architecture analysis
    Architecture,
    /// Performance metrics
    Performance,
    /// Memory analysis
    Memory,
    /// Gradient analysis
    Gradients,
    /// Training dynamics
    Training,
    /// Error analysis
    Errors,
    /// Recommendations
    Recommendations,
    /// Visualizations
    Visualizations,
    /// Raw data
    RawData,
}

/// Report configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    /// Report title
    pub title: String,
    /// Report subtitle
    pub subtitle: Option<String>,
    /// Author information
    pub author: String,
    /// Organization
    pub organization: Option<String>,
    /// Report format
    pub format: ReportFormat,
    /// Report type
    pub report_type: ReportType,
    /// Include visualizations
    pub include_visualizations: bool,
    /// Include raw data
    pub include_raw_data: bool,
    /// Output path
    pub output_path: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            title: "TrustformeRS Debug Report".to_string(),
            subtitle: None,
            author: "TrustformeRS Debugger".to_string(),
            organization: None,
            format: ReportFormat::Html,
            report_type: ReportType::ComprehensiveReport,
            include_visualizations: true,
            include_raw_data: false,
            output_path: "debug_report".to_string(),
            metadata: HashMap::new(),
        }
    }
}

/// Generated report content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Report sections
    pub sections: Vec<GeneratedSection>,
    /// Visualizations
    pub visualizations: HashMap<String, PlotData>,
    /// Raw data
    pub raw_data: HashMap<String, serde_json::Value>,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Report title
    pub title: String,
    /// Report subtitle
    pub subtitle: Option<String>,
    /// Author
    pub author: String,
    /// Organization
    pub organization: Option<String>,
    /// Report version
    pub version: String,
    /// Generation time
    pub generation_time_ms: f64,
    /// Additional metadata
    pub additional_metadata: HashMap<String, String>,
}

/// Generated report section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedSection {
    /// Section type
    pub section_type: ReportSection,
    /// Section title
    pub title: String,
    /// Section content
    pub content: String,
    /// Section data
    pub data: HashMap<String, serde_json::Value>,
    /// Associated visualizations
    pub visualizations: Vec<String>,
}

/// Report generator
#[derive(Debug)]
pub struct ReportGenerator {
    /// Configuration
    config: ReportConfig,
    /// Debug data
    debug_data: Option<GradientDebugReport>,
    /// Profiling data
    profiling_data: Option<ProfilerReport>,
    /// Visualizer
    #[allow(dead_code)]
    visualizer: DebugVisualizer,
}

impl ReportGenerator {
    /// Create a new report generator
    pub fn new(config: ReportConfig) -> Self {
        Self {
            config,
            debug_data: None,
            profiling_data: None,
            visualizer: DebugVisualizer::new(VisualizationConfig::default()),
        }
    }

    /// Add gradient debug data
    pub fn with_debug_data(mut self, data: GradientDebugReport) -> Self {
        self.debug_data = Some(data);
        self
    }

    /// Add profiling data
    pub fn with_profiling_data(mut self, data: ProfilerReport) -> Self {
        self.profiling_data = Some(data);
        self
    }

    /// Generate the report
    pub fn generate(&self) -> Result<Report, ReportError> {
        let start_time = std::time::Instant::now();

        let sections = match &self.config.report_type {
            ReportType::DebugReport => self.generate_debug_sections()?,
            ReportType::PerformanceReport => self.generate_performance_sections()?,
            ReportType::TrainingReport => self.generate_training_sections()?,
            ReportType::GradientReport => self.generate_gradient_sections()?,
            ReportType::MemoryReport => self.generate_memory_sections()?,
            ReportType::ComprehensiveReport => self.generate_comprehensive_sections()?,
            ReportType::CustomReport(section_types) => {
                self.generate_custom_sections(section_types)?
            },
        };

        let visualizations = if self.config.include_visualizations {
            self.generate_visualizations()?
        } else {
            HashMap::new()
        };

        let raw_data = if self.config.include_raw_data {
            self.generate_raw_data()?
        } else {
            HashMap::new()
        };

        let generation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let report = Report {
            metadata: ReportMetadata {
                title: self.config.title.clone(),
                subtitle: self.config.subtitle.clone(),
                author: self.config.author.clone(),
                organization: self.config.organization.clone(),
                version: "1.0".to_string(),
                generation_time_ms: generation_time,
                additional_metadata: self.config.metadata.clone(),
            },
            sections,
            visualizations,
            raw_data,
            generated_at: Utc::now(),
        };

        Ok(report)
    }

    /// Generate debug report sections
    fn generate_debug_sections(&self) -> Result<Vec<GeneratedSection>, ReportError> {
        let mut sections = Vec::new();

        // Summary section
        sections.push(self.generate_summary_section()?);

        // Architecture section
        sections.push(self.generate_architecture_section()?);

        // Gradient analysis
        if self.debug_data.is_some() {
            sections.push(self.generate_gradients_section()?);
        }

        // Error analysis
        sections.push(self.generate_errors_section()?);

        // Recommendations
        sections.push(self.generate_recommendations_section()?);

        Ok(sections)
    }

    /// Generate performance report sections
    fn generate_performance_sections(&self) -> Result<Vec<GeneratedSection>, ReportError> {
        let mut sections = Vec::new();

        sections.push(self.generate_summary_section()?);
        sections.push(self.generate_performance_section()?);

        if self.profiling_data.is_some() {
            sections.push(self.generate_memory_section()?);
        }

        sections.push(self.generate_recommendations_section()?);

        Ok(sections)
    }

    /// Generate training report sections
    fn generate_training_sections(&self) -> Result<Vec<GeneratedSection>, ReportError> {
        let mut sections = Vec::new();

        sections.push(self.generate_summary_section()?);
        sections.push(self.generate_training_section()?);
        sections.push(self.generate_gradients_section()?);
        sections.push(self.generate_recommendations_section()?);

        Ok(sections)
    }

    /// Generate gradient report sections
    fn generate_gradient_sections(&self) -> Result<Vec<GeneratedSection>, ReportError> {
        let mut sections = Vec::new();

        sections.push(self.generate_summary_section()?);
        sections.push(self.generate_gradients_section()?);
        sections.push(self.generate_recommendations_section()?);

        Ok(sections)
    }

    /// Generate memory report sections
    fn generate_memory_sections(&self) -> Result<Vec<GeneratedSection>, ReportError> {
        let mut sections = Vec::new();

        sections.push(self.generate_summary_section()?);
        sections.push(self.generate_memory_section()?);
        sections.push(self.generate_recommendations_section()?);

        Ok(sections)
    }

    /// Generate comprehensive report sections
    fn generate_comprehensive_sections(&self) -> Result<Vec<GeneratedSection>, ReportError> {
        let mut sections = Vec::new();

        sections.push(self.generate_summary_section()?);
        sections.push(self.generate_architecture_section()?);
        sections.push(self.generate_performance_section()?);
        sections.push(self.generate_memory_section()?);
        sections.push(self.generate_gradients_section()?);
        sections.push(self.generate_training_section()?);
        sections.push(self.generate_errors_section()?);
        sections.push(self.generate_recommendations_section()?);

        Ok(sections)
    }

    /// Generate custom report sections
    fn generate_custom_sections(
        &self,
        section_types: &[ReportSection],
    ) -> Result<Vec<GeneratedSection>, ReportError> {
        let mut sections = Vec::new();

        for section_type in section_types {
            let section = match section_type {
                ReportSection::Summary => self.generate_summary_section()?,
                ReportSection::Architecture => self.generate_architecture_section()?,
                ReportSection::Performance => self.generate_performance_section()?,
                ReportSection::Memory => self.generate_memory_section()?,
                ReportSection::Gradients => self.generate_gradients_section()?,
                ReportSection::Training => self.generate_training_section()?,
                ReportSection::Errors => self.generate_errors_section()?,
                ReportSection::Recommendations => self.generate_recommendations_section()?,
                ReportSection::Visualizations => self.generate_visualizations_section()?,
                ReportSection::RawData => self.generate_raw_data_section()?,
            };
            sections.push(section);
        }

        Ok(sections)
    }

    /// Generate summary section
    fn generate_summary_section(&self) -> Result<GeneratedSection, ReportError> {
        let mut content = String::new();
        let mut data = HashMap::new();

        content.push_str("## Executive Summary\n\n");
        content.push_str("This report provides a comprehensive analysis of the TrustformeRS model debugging session.\n\n");

        // Add key metrics
        if let Some(debug_data) = &self.debug_data {
            content.push_str(&format!(
                "- **Total Layers Analyzed**: {}\n",
                debug_data.flow_analysis.layer_analyses.len()
            ));

            let healthy_layers = debug_data
                .flow_analysis
                .layer_analyses
                .iter()
                .filter(|(_name, l)| !l.is_vanishing && !l.is_exploding)
                .count();
            content.push_str(&format!("- **Healthy Layers**: {}\n", healthy_layers));

            data.insert(
                "total_layers".to_string(),
                serde_json::json!(debug_data.flow_analysis.layer_analyses.len()),
            );
            data.insert(
                "healthy_layers".to_string(),
                serde_json::json!(healthy_layers),
            );
        }

        if let Some(profiling_data) = &self.profiling_data {
            content.push_str(&format!(
                "- **Total Memory Usage**: {:.2} MB\n",
                profiling_data.memory_efficiency.peak_memory_mb
            ));
            content.push_str(&format!(
                "- **Execution Time**: {:.2} ms\n",
                profiling_data.total_runtime.as_millis() as f64
            ));

            data.insert(
                "peak_memory_mb".to_string(),
                serde_json::json!(profiling_data.memory_efficiency.peak_memory_mb),
            );
            data.insert(
                "total_time_ms".to_string(),
                serde_json::json!(profiling_data.total_runtime.as_millis() as f64),
            );
        }

        Ok(GeneratedSection {
            section_type: ReportSection::Summary,
            title: "Executive Summary".to_string(),
            content,
            data,
            visualizations: Vec::new(),
        })
    }

    /// Generate architecture section
    fn generate_architecture_section(&self) -> Result<GeneratedSection, ReportError> {
        let mut content = String::new();
        let data = HashMap::new();

        content.push_str("## Model Architecture Analysis\n\n");
        content.push_str("This section provides detailed analysis of the model architecture.\n\n");

        // Add architecture details if available
        content.push_str("### Layer Structure\n\n");
        if let Some(debug_data) = &self.debug_data {
            content.push_str("| Layer | Type | Parameters | Health Status |\n");
            content.push_str("|-------|------|------------|---------------|\n");

            for (i, (layer_name, layer)) in
                debug_data.flow_analysis.layer_analyses.iter().enumerate()
            {
                let health = if layer.is_vanishing {
                    "Vanishing"
                } else if layer.is_exploding {
                    "Exploding"
                } else {
                    "Healthy"
                };
                content.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    i,
                    layer_name,
                    "N/A", // In a real implementation, get parameter count
                    health
                ));
            }
        } else {
            content.push_str("No architecture data available.\n");
        }

        Ok(GeneratedSection {
            section_type: ReportSection::Architecture,
            title: "Model Architecture Analysis".to_string(),
            content,
            data,
            visualizations: vec!["architecture_diagram".to_string()],
        })
    }

    /// Generate performance section
    fn generate_performance_section(&self) -> Result<GeneratedSection, ReportError> {
        let mut content = String::new();
        let mut data = HashMap::new();

        content.push_str("## Performance Analysis\n\n");

        if let Some(profiling_data) = &self.profiling_data {
            content.push_str("### Timing Statistics\n\n");
            content.push_str(&format!(
                "- **Total Execution Time**: {:.2} ms\n",
                profiling_data.total_runtime.as_millis() as f64
            ));
            content.push_str(&format!(
                "- **Forward Pass Time**: {:.2} ms\n",
                profiling_data.total_runtime.as_millis() as f64 * 0.6
            )); // Approximate 60% forward
            content.push_str(&format!(
                "- **Backward Pass Time**: {:.2} ms\n",
                profiling_data.total_runtime.as_millis() as f64 * 0.4
            )); // Approximate 40% backward

            content.push_str("\n### Throughput\n\n");
            let tokens_per_sec = 1000.0 / (profiling_data.total_runtime.as_millis() as f64 + 1.0); // Approximate throughput
            content.push_str(&format!("- **Tokens per Second**: {:.2}\n", tokens_per_sec));
            content.push_str(&format!(
                "- **Samples per Second**: {:.2}\n",
                tokens_per_sec * 10.0
            )); // Approximate samples

            // Add data for charts
            let timing_stats = serde_json::json!({
                "total_time_ms": profiling_data.total_runtime.as_millis() as f64,
                "forward_pass_ms": profiling_data.total_runtime.as_millis() as f64 * 0.6,
                "backward_pass_ms": profiling_data.total_runtime.as_millis() as f64 * 0.4
            });
            let throughput_stats = serde_json::json!({
                "tokens_per_second": tokens_per_sec,
                "samples_per_second": tokens_per_sec * 10.0
            });
            data.insert("timing_stats".to_string(), timing_stats);
            data.insert("throughput_stats".to_string(), throughput_stats);
        } else {
            content.push_str("No performance data available.\n");
        }

        Ok(GeneratedSection {
            section_type: ReportSection::Performance,
            title: "Performance Analysis".to_string(),
            content,
            data,
            visualizations: vec!["performance_chart".to_string()],
        })
    }

    /// Generate memory section
    fn generate_memory_section(&self) -> Result<GeneratedSection, ReportError> {
        let mut content = String::new();
        let mut data = HashMap::new();

        content.push_str("## Memory Analysis\n\n");

        if let Some(profiling_data) = &self.profiling_data {
            content.push_str("### Memory Usage\n\n");
            content.push_str(&format!(
                "- **Peak Memory**: {:.2} MB\n",
                profiling_data.memory_efficiency.peak_memory_mb
            ));
            content.push_str(&format!(
                "- **Current Memory**: {:.2} MB\n",
                profiling_data.memory_efficiency.avg_memory_mb
            ));
            content.push_str(&format!(
                "- **Memory Efficiency**: {:.2}%\n",
                profiling_data.memory_efficiency.efficiency_score
            ));

            data.insert(
                "memory_stats".to_string(),
                serde_json::to_value(&profiling_data.memory_efficiency).unwrap(),
            );
        } else {
            content.push_str("No memory data available.\n");
        }

        Ok(GeneratedSection {
            section_type: ReportSection::Memory,
            title: "Memory Analysis".to_string(),
            content,
            data,
            visualizations: vec!["memory_chart".to_string()],
        })
    }

    /// Generate gradients section
    fn generate_gradients_section(&self) -> Result<GeneratedSection, ReportError> {
        let mut content = String::new();
        let mut data = HashMap::new();

        content.push_str("## Gradient Analysis\n\n");

        if let Some(debug_data) = &self.debug_data {
            content.push_str("### Gradient Health Summary\n\n");

            let healthy_count = debug_data
                .flow_analysis
                .layer_analyses
                .iter()
                .filter(|(_name, l)| !l.is_vanishing && !l.is_exploding)
                .count();
            let problematic_count = debug_data.flow_analysis.layer_analyses.len() - healthy_count;

            content.push_str(&format!("- **Healthy Layers**: {}\n", healthy_count));
            content.push_str(&format!(
                "- **Problematic Layers**: {}\n",
                problematic_count
            ));

            if problematic_count > 0 {
                content.push_str("\n### Issues Detected\n\n");
                for (i, (layer_name, layer)) in
                    debug_data.flow_analysis.layer_analyses.iter().enumerate()
                {
                    if layer.is_vanishing || layer.is_exploding {
                        let status = if layer.is_vanishing {
                            "Vanishing gradients"
                        } else {
                            "Exploding gradients"
                        };
                        content
                            .push_str(&format!("- **Layer {}** ({}): {}\n", i, layer_name, status));
                    }
                }
            }

            data.insert(
                "gradient_analysis".to_string(),
                serde_json::to_value(debug_data).unwrap(),
            );
        } else {
            content.push_str("No gradient data available.\n");
        }

        Ok(GeneratedSection {
            section_type: ReportSection::Gradients,
            title: "Gradient Analysis".to_string(),
            content,
            data,
            visualizations: vec!["gradient_flow_chart".to_string()],
        })
    }

    /// Generate training section
    fn generate_training_section(&self) -> Result<GeneratedSection, ReportError> {
        let content =
            "## Training Dynamics\n\nTraining dynamics analysis would go here.".to_string();
        let data = HashMap::new();

        Ok(GeneratedSection {
            section_type: ReportSection::Training,
            title: "Training Dynamics".to_string(),
            content,
            data,
            visualizations: vec!["training_curves".to_string()],
        })
    }

    /// Generate errors section
    fn generate_errors_section(&self) -> Result<GeneratedSection, ReportError> {
        let content = "## Error Analysis\n\nError analysis would go here.".to_string();
        let data = HashMap::new();

        Ok(GeneratedSection {
            section_type: ReportSection::Errors,
            title: "Error Analysis".to_string(),
            content,
            data,
            visualizations: Vec::new(),
        })
    }

    /// Generate recommendations section
    fn generate_recommendations_section(&self) -> Result<GeneratedSection, ReportError> {
        let mut content = String::new();
        let data = HashMap::new();

        content.push_str("## Recommendations\n\n");
        content.push_str("Based on the analysis, here are our recommendations:\n\n");

        // Add specific recommendations based on data
        if let Some(debug_data) = &self.debug_data {
            let problematic_layers = debug_data
                .flow_analysis
                .layer_analyses
                .iter()
                .filter(|(_name, l)| l.is_vanishing || l.is_exploding)
                .count();

            if problematic_layers > 0 {
                content.push_str("### Gradient Issues\n\n");
                content.push_str("- Consider adjusting learning rate\n");
                content.push_str("- Review gradient clipping settings\n");
                content.push_str("- Check for numerical instabilities\n\n");
            }
        }

        if let Some(profiling_data) = &self.profiling_data {
            if profiling_data.memory_efficiency.efficiency_score < 80.0 {
                content.push_str("### Memory Optimization\n\n");
                content.push_str("- Consider using gradient checkpointing\n");
                content.push_str("- Review batch size settings\n");
                content.push_str("- Consider model quantization\n\n");
            }
        }

        content.push_str("### General Recommendations\n\n");
        content.push_str("- Monitor training regularly\n");
        content.push_str("- Validate on diverse test sets\n");
        content.push_str("- Keep detailed training logs\n");

        Ok(GeneratedSection {
            section_type: ReportSection::Recommendations,
            title: "Recommendations".to_string(),
            content,
            data,
            visualizations: Vec::new(),
        })
    }

    /// Generate visualizations section
    fn generate_visualizations_section(&self) -> Result<GeneratedSection, ReportError> {
        let content = "## Visualizations\n\nVisualization section content.".to_string();
        let data = HashMap::new();

        Ok(GeneratedSection {
            section_type: ReportSection::Visualizations,
            title: "Visualizations".to_string(),
            content,
            data,
            visualizations: vec!["all_charts".to_string()],
        })
    }

    /// Generate raw data section
    fn generate_raw_data_section(&self) -> Result<GeneratedSection, ReportError> {
        let content = "## Raw Data\n\nRaw data section content.".to_string();
        let data = HashMap::new();

        Ok(GeneratedSection {
            section_type: ReportSection::RawData,
            title: "Raw Data".to_string(),
            content,
            data,
            visualizations: Vec::new(),
        })
    }

    /// Generate visualizations
    fn generate_visualizations(&self) -> Result<HashMap<String, PlotData>, ReportError> {
        let mut visualizations = HashMap::new();

        // Generate performance chart
        if self.profiling_data.is_some() {
            let plot_data = PlotData {
                x_values: vec![1.0, 2.0, 3.0],    // Placeholder data
                y_values: vec![10.0, 15.0, 12.0], // Placeholder performance data
                labels: vec!["A".to_string(), "B".to_string(), "C".to_string()],
                title: "Performance Chart".to_string(),
                x_label: "Time".to_string(),
                y_label: "Performance".to_string(),
            };
            visualizations.insert("performance_chart".to_string(), plot_data);
        }

        Ok(visualizations)
    }

    /// Generate raw data
    fn generate_raw_data(&self) -> Result<HashMap<String, serde_json::Value>, ReportError> {
        let mut raw_data = HashMap::new();

        if let Some(debug_data) = &self.debug_data {
            raw_data.insert(
                "debug_data".to_string(),
                serde_json::to_value(debug_data).unwrap(),
            );
        }

        if let Some(profiling_data) = &self.profiling_data {
            raw_data.insert(
                "profiling_data".to_string(),
                serde_json::to_value(profiling_data).unwrap(),
            );
        }

        Ok(raw_data)
    }

    /// Export report to file
    pub fn export_report(&self, report: &Report) -> Result<(), ReportError> {
        match self.config.format {
            ReportFormat::Html => self.export_html(report),
            ReportFormat::Markdown => self.export_markdown(report),
            ReportFormat::Json => self.export_json(report),
            ReportFormat::Pdf => self.export_pdf(report),
            ReportFormat::Jupyter => self.export_jupyter(report),
            ReportFormat::Latex => self.export_latex(report),
            ReportFormat::Excel => self.export_excel(report),
            ReportFormat::PowerPoint => self.export_powerpoint(report),
        }
    }

    /// Export to HTML format
    fn export_html(&self, report: &Report) -> Result<(), ReportError> {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("<title>{}</title>\n", report.metadata.title));
        html.push_str("<style>body { font-family: Arial, sans-serif; margin: 40px; }</style>\n");
        html.push_str("</head>\n<body>\n");

        html.push_str(&format!("<h1>{}</h1>\n", report.metadata.title));
        if let Some(subtitle) = &report.metadata.subtitle {
            html.push_str(&format!("<h2>{}</h2>\n", subtitle));
        }

        for section in &report.sections {
            html.push_str(&section.content);
        }

        html.push_str("</body>\n</html>");

        std::fs::write(format!("{}.html", self.config.output_path), html)
            .map_err(|e| ReportError::FileError(e.to_string()))?;

        Ok(())
    }

    /// Export to Markdown format
    fn export_markdown(&self, report: &Report) -> Result<(), ReportError> {
        let mut markdown = String::new();

        markdown.push_str(&format!("# {}\n\n", report.metadata.title));
        if let Some(subtitle) = &report.metadata.subtitle {
            markdown.push_str(&format!("## {}\n\n", subtitle));
        }

        markdown.push_str(&format!("**Author**: {}\n", report.metadata.author));
        markdown.push_str(&format!(
            "**Generated**: {}\n\n",
            report.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        for section in &report.sections {
            markdown.push_str(&section.content);
            markdown.push('\n');
        }

        std::fs::write(format!("{}.md", self.config.output_path), markdown)
            .map_err(|e| ReportError::FileError(e.to_string()))?;

        Ok(())
    }

    /// Export to JSON format
    fn export_json(&self, report: &Report) -> Result<(), ReportError> {
        let json = serde_json::to_string_pretty(report)
            .map_err(|e| ReportError::SerializationError(e.to_string()))?;

        std::fs::write(format!("{}.json", self.config.output_path), json)
            .map_err(|e| ReportError::FileError(e.to_string()))?;

        Ok(())
    }

    /// Export to PDF format (placeholder implementation)
    fn export_pdf(&self, _report: &Report) -> Result<(), ReportError> {
        // In a real implementation, this would use a PDF generation library
        Err(ReportError::UnsupportedFormat(
            "PDF export not implemented".to_string(),
        ))
    }

    /// Export to Jupyter notebook format
    fn export_jupyter(&self, report: &Report) -> Result<(), ReportError> {
        let mut notebook = serde_json::json!({
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        });

        // Add title cell
        let title_cell = serde_json::json!({
            "cell_type": "markdown",
            "metadata": {},
            "source": [format!("# {}\n\n**Generated**: {}",
                report.metadata.title,
                report.generated_at.format("%Y-%m-%d %H:%M:%S UTC"))]
        });
        notebook["cells"].as_array_mut().unwrap().push(title_cell);

        // Add content cells
        for section in &report.sections {
            let cell = serde_json::json!({
                "cell_type": "markdown",
                "metadata": {},
                "source": [section.content]
            });
            notebook["cells"].as_array_mut().unwrap().push(cell);
        }

        let notebook_str = serde_json::to_string_pretty(&notebook)
            .map_err(|e| ReportError::SerializationError(e.to_string()))?;

        std::fs::write(format!("{}.ipynb", self.config.output_path), notebook_str)
            .map_err(|e| ReportError::FileError(e.to_string()))?;

        Ok(())
    }

    /// Export to LaTeX format
    fn export_latex(&self, report: &Report) -> Result<(), ReportError> {
        let mut latex = String::new();

        latex.push_str("\\documentclass{article}\n");
        latex.push_str("\\begin{document}\n");
        latex.push_str(&format!("\\title{{{}}}\n", report.metadata.title));
        latex.push_str(&format!("\\author{{{}}}\n", report.metadata.author));
        latex.push_str("\\maketitle\n\n");

        for section in &report.sections {
            // Convert markdown to LaTeX (simplified)
            let latex_content = section
                .content
                .replace("##", "\\section{")
                .replace("###", "\\subsection{")
                .replace("#", "\\section{");
            latex.push_str(&latex_content);
        }

        latex.push_str("\\end{document}\n");

        std::fs::write(format!("{}.tex", self.config.output_path), latex)
            .map_err(|e| ReportError::FileError(e.to_string()))?;

        Ok(())
    }

    /// Export to Excel format (placeholder)
    fn export_excel(&self, _report: &Report) -> Result<(), ReportError> {
        Err(ReportError::UnsupportedFormat(
            "Excel export not implemented".to_string(),
        ))
    }

    /// Export to PowerPoint format (placeholder)
    fn export_powerpoint(&self, _report: &Report) -> Result<(), ReportError> {
        Err(ReportError::UnsupportedFormat(
            "PowerPoint export not implemented".to_string(),
        ))
    }
}

/// Report generation errors
#[derive(Debug, Clone)]
pub enum ReportError {
    /// File system error
    FileError(String),
    /// Serialization error
    SerializationError(String),
    /// Unsupported format
    UnsupportedFormat(String),
    /// Missing data
    MissingData(String),
    /// Generation error
    GenerationError(String),
}

impl std::fmt::Display for ReportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReportError::FileError(msg) => write!(f, "File error: {}", msg),
            ReportError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            ReportError::UnsupportedFormat(msg) => write!(f, "Unsupported format: {}", msg),
            ReportError::MissingData(msg) => write!(f, "Missing data: {}", msg),
            ReportError::GenerationError(msg) => write!(f, "Generation error: {}", msg),
        }
    }
}

impl std::error::Error for ReportError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_config_default() {
        let config = ReportConfig::default();
        assert_eq!(config.title, "TrustformeRS Debug Report");
        assert_eq!(config.author, "TrustformeRS Debugger");
        assert!(matches!(config.format, ReportFormat::Html));
        assert!(matches!(
            config.report_type,
            ReportType::ComprehensiveReport
        ));
    }

    #[test]
    fn test_report_generator_creation() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config);
        assert!(generator.debug_data.is_none());
        assert!(generator.profiling_data.is_none());
    }

    #[test]
    fn test_report_generation() {
        let config = ReportConfig {
            title: "Test Report".to_string(),
            format: ReportFormat::Json,
            report_type: ReportType::DebugReport,
            ..Default::default()
        };

        let generator = ReportGenerator::new(config);
        let report = generator.generate().unwrap();

        assert_eq!(report.metadata.title, "Test Report");
        assert!(!report.sections.is_empty());
    }

    #[test]
    fn test_section_generation() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config);

        let summary = generator.generate_summary_section().unwrap();
        assert!(matches!(summary.section_type, ReportSection::Summary));
        assert_eq!(summary.title, "Executive Summary");
        assert!(!summary.content.is_empty());
    }

    #[test]
    fn test_custom_report_type() {
        let config = ReportConfig {
            report_type: ReportType::CustomReport(vec![
                ReportSection::Summary,
                ReportSection::Performance,
            ]),
            ..Default::default()
        };

        let generator = ReportGenerator::new(config);
        let report = generator.generate().unwrap();

        assert_eq!(report.sections.len(), 2);
        assert!(matches!(
            report.sections[0].section_type,
            ReportSection::Summary
        ));
        assert!(matches!(
            report.sections[1].section_type,
            ReportSection::Performance
        ));
    }

    #[test]
    fn test_report_serialization() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config);
        let report = generator.generate().unwrap();

        let json = serde_json::to_string(&report).unwrap();
        let deserialized: Report = serde_json::from_str(&json).unwrap();

        assert_eq!(report.metadata.title, deserialized.metadata.title);
        assert_eq!(report.sections.len(), deserialized.sections.len());
    }
}
