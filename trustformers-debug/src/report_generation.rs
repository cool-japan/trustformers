//! Report Generation for TrustformeRS Debug
//!
//! This module provides comprehensive reporting capabilities for debugging,
//! analysis, and documentation. Supports multiple output formats including
//! PDF, Markdown, HTML, JSON, and Jupyter notebooks.
// reason: debug/profiling scaffolding — structs are constructed and their fields/methods
// are retained for the data model, serialization completeness, and future consumers that
// do not yet read every member. Consolidated from many item-level #[allow(dead_code)].
#![allow(dead_code)]

use crate::{
    architecture_analysis::ArchitectureAnalysisReport,
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

impl ReportFormat {
    /// Whether [`ReportGenerator::export_report`] has a real writer for this
    /// format. `Pdf` / `Excel` / `PowerPoint` are listed variants of this
    /// enum (kept for API/config-schema stability -- external configs may
    /// already reference them) but have no generation library backing them
    /// in this crate; selecting one is rejected up front by
    /// [`ReportGenerator::new`] rather than only failing after a caller has
    /// already paid for the (potentially expensive) analysis and section
    /// generation that happens before `export_report` is ever called.
    pub fn is_implemented(&self) -> bool {
        !matches!(
            self,
            ReportFormat::Pdf | ReportFormat::Excel | ReportFormat::PowerPoint
        )
    }

    /// Human-readable name used in [`ReportError::UnsupportedFormat`]
    /// messages.
    fn label(&self) -> &'static str {
        match self {
            ReportFormat::Pdf => "PDF",
            ReportFormat::Markdown => "Markdown",
            ReportFormat::Html => "HTML",
            ReportFormat::Json => "JSON",
            ReportFormat::Jupyter => "Jupyter",
            ReportFormat::Latex => "LaTeX",
            ReportFormat::Excel => "Excel",
            ReportFormat::PowerPoint => "PowerPoint",
        }
    }
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
    /// Model architecture data (parameter counts, layer shapes, ...), used by
    /// [`Self::generate_architecture_section`] to fill in real per-layer
    /// parameter counts instead of the honest-but-permanent "N/A" that is
    /// used when this is absent.
    architecture_data: Option<ArchitectureAnalysisReport>,
    /// Visualizer
    visualizer: DebugVisualizer,
}

impl ReportGenerator {
    /// Create a new report generator.
    ///
    /// Rejects `config.format` immediately (before any analysis or section
    /// generation runs) when it names a format this crate cannot write --
    /// see [`ReportFormat::is_implemented`]. The old behavior accepted any
    /// format at construction and only discovered the mismatch inside
    /// `export_report`, after a caller had already paid for the full report
    /// generation.
    pub fn new(config: ReportConfig) -> Result<Self, ReportError> {
        if !config.format.is_implemented() {
            return Err(ReportError::UnsupportedFormat(format!(
                "{} export is not implemented; choose one of Markdown, Html, Json, Jupyter, or \
                 Latex",
                config.format.label()
            )));
        }
        Ok(Self {
            config,
            debug_data: None,
            profiling_data: None,
            architecture_data: None,
            visualizer: DebugVisualizer::new(VisualizationConfig::default()),
        })
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

    /// Add model architecture data (real per-layer parameter counts, shapes,
    /// ...). Without this, [`Self::generate_architecture_section`] reports
    /// each layer's parameter count as `N/A` -- an honest absence, not a
    /// fabricated number -- rather than guessing.
    pub fn with_architecture_data(mut self, data: ArchitectureAnalysisReport) -> Self {
        self.architecture_data = Some(data);
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

        // Real per-layer parameter counts, keyed by layer name, from
        // whatever architecture data was attached via
        // `with_architecture_data`. Absent (rather than guessed) when no
        // architecture data was provided, or when a given gradient-flow
        // layer name has no matching entry there.
        let parameter_counts: HashMap<&str, usize> = self
            .architecture_data
            .as_ref()
            .map(|arch| arch.layers.iter().map(|l| (l.name.as_str(), l.parameters)).collect())
            .unwrap_or_default();

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
                let parameters = parameter_counts
                    .get(layer_name.as_str())
                    .map(|count| count.to_string())
                    .unwrap_or_else(|| "N/A".to_string());
                content.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    i, layer_name, parameters, health
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
                serde_json::to_value(&profiling_data.memory_efficiency)
                    .map_err(|e| ReportError::SerializationError(e.to_string()))?,
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
                serde_json::to_value(debug_data)
                    .map_err(|e| ReportError::SerializationError(e.to_string()))?,
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

    /// Generate visualizations.
    ///
    /// The performance chart plots `profiling_data.slowest_layers` -- a real
    /// ranked list of `(layer_name, Duration)` produced by the profiler --
    /// rather than the fixed `[1,2,3]`/`[10,15,12]` points the old
    /// implementation emitted regardless of what was actually profiled.
    /// Omitted entirely (never fabricated) when there is no profiling data,
    /// or it recorded no layer timings.
    fn generate_visualizations(&self) -> Result<HashMap<String, PlotData>, ReportError> {
        let mut visualizations = HashMap::new();

        if let Some(profiling_data) = &self.profiling_data {
            if !profiling_data.slowest_layers.is_empty() {
                let x_values: Vec<f64> =
                    (0..profiling_data.slowest_layers.len()).map(|i| i as f64).collect();
                let y_values: Vec<f64> = profiling_data
                    .slowest_layers
                    .iter()
                    .map(|(_, duration)| duration.as_secs_f64() * 1000.0)
                    .collect();
                let labels: Vec<String> =
                    profiling_data.slowest_layers.iter().map(|(name, _)| name.clone()).collect();

                let plot_data = PlotData {
                    x_values,
                    y_values,
                    labels,
                    title: "Performance Chart".to_string(),
                    x_label: "Layer Rank (slowest first)".to_string(),
                    y_label: "Duration (ms)".to_string(),
                };
                visualizations.insert("performance_chart".to_string(), plot_data);
            }
        }

        Ok(visualizations)
    }

    /// Generate raw data
    fn generate_raw_data(&self) -> Result<HashMap<String, serde_json::Value>, ReportError> {
        let mut raw_data = HashMap::new();

        if let Some(debug_data) = &self.debug_data {
            raw_data.insert(
                "debug_data".to_string(),
                serde_json::to_value(debug_data)
                    .map_err(|e| ReportError::SerializationError(e.to_string()))?,
            );
        }

        if let Some(profiling_data) = &self.profiling_data {
            raw_data.insert(
                "profiling_data".to_string(),
                serde_json::to_value(profiling_data)
                    .map_err(|e| ReportError::SerializationError(e.to_string()))?,
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

    /// PDF export: **not implemented**, returns a structured error.
    ///
    /// No PDF writer is linked into `trustformers-debug`. Use
    /// [`ReportFormat::Html`] or [`ReportFormat::LaTeX`] and convert
    /// externally.
    fn export_pdf(&self, _report: &Report) -> Result<(), ReportError> {
        Err(ReportError::UnsupportedFormat(
            "PDF export not implemented: trustformers-debug links no PDF writer. Export HTML \
             or LaTeX and convert externally."
                .to_string(),
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
        let cells = notebook["cells"].as_array_mut().ok_or_else(|| {
            ReportError::SerializationError("notebook cells should be an array".to_string())
        })?;
        cells.push(title_cell);

        // Add content cells
        for section in &report.sections {
            let cell = serde_json::json!({
                "cell_type": "markdown",
                "metadata": {},
                "source": [section.content]
            });
            cells.push(cell);
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
            latex.push_str(&markdown_headings_to_latex(&section.content));
        }

        latex.push_str("\\end{document}\n");

        std::fs::write(format!("{}.tex", self.config.output_path), latex)
            .map_err(|e| ReportError::FileError(e.to_string()))?;

        Ok(())
    }

    /// Excel export: **not implemented**, returns a structured error.
    ///
    /// (`crate::data_export` does emit a real `.xlsx` for tabular exports; a
    /// narrative `Report` has no single sheet shape to map onto.)
    fn export_excel(&self, _report: &Report) -> Result<(), ReportError> {
        Err(ReportError::UnsupportedFormat(
            "Excel export not implemented for narrative reports; see crate::data_export for \
             real .xlsx output of tabular data."
                .to_string(),
        ))
    }

    /// PowerPoint export: **not implemented**, returns a structured error.
    fn export_powerpoint(&self, _report: &Report) -> Result<(), ReportError> {
        Err(ReportError::UnsupportedFormat(
            "PowerPoint export not implemented: trustformers-debug links no OOXML presentation \
             writer."
                .to_string(),
        ))
    }
}

/// Convert the ATX-style markdown headings in `content` into LaTeX sectioning
/// commands, escaping the LaTeX specials in the rest of the text.
///
/// Handles `#`, `##` and `###` as `\section`, `\subsection` and
/// `\subsubsection`, each with a CLOSED brace.
///
/// The previous version chained
/// `.replace("##", "\\section{").replace("###", ...).replace("#", ...)`, which
/// (a) never emitted a closing `}` so every document failed to compile,
/// (b) could not reach the `###` arm at all because the `##` replacement had
/// already consumed the first two hashes, turning `### Title` into
/// `\section{\section{ Title`, and (c) rewrote `#` anywhere in the body, not
/// just at the start of a line.
fn markdown_headings_to_latex(content: &str) -> String {
    let mut out = String::with_capacity(content.len() + 32);
    for line in content.lines() {
        let trimmed = line.trim_start();
        let level = trimmed.chars().take_while(|&c| c == '#').count();
        if (1..=3).contains(&level) && trimmed.chars().nth(level) == Some(' ') {
            let command = match level {
                1 => "section",
                2 => "subsection",
                _ => "subsubsection",
            };
            let title = escape_latex(trimmed[level + 1..].trim());
            out.push_str(&format!("\\{}{{{}}}\n", command, title));
        } else {
            out.push_str(&escape_latex(line));
            out.push('\n');
        }
    }
    out
}

/// Escape the characters LaTeX treats specially so report prose cannot break
/// (or inject into) the generated document.
fn escape_latex(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\textbackslash{}"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '$' | '&' | '%' | '#' | '_' => {
                out.push('\\');
                out.push(ch);
            },
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            _ => out.push(ch),
        }
    }
    out
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
    fn markdown_headings_become_closed_latex_sections() {
        let latex = markdown_headings_to_latex("# One\n## Two\n### Three\nbody\n");
        assert!(latex.contains("\\section{One}"), "{latex}");
        assert!(latex.contains("\\subsection{Two}"), "{latex}");
        assert!(latex.contains("\\subsubsection{Three}"), "{latex}");
        // The old chained-replace produced `\section{\section{ Three` and never
        // closed a single brace.
        assert_eq!(
            latex.matches('{').count(),
            latex.matches('}').count(),
            "every brace must be closed:\n{latex}"
        );
        assert!(latex.contains("body"));
    }

    #[test]
    fn latex_specials_in_body_text_are_escaped() {
        let latex = markdown_headings_to_latex("100% of $x_1 & y#2");
        assert!(latex.contains("100\\%"), "{latex}");
        assert!(latex.contains("\\$x\\_1"), "{latex}");
        assert!(latex.contains("\\&"), "{latex}");
        assert!(latex.contains("y\\#2"), "{latex}");
        // A lone '#' inside a line must NOT be turned into a section command.
        assert!(!latex.contains("\\section"), "{latex}");
    }

    #[test]
    fn unimplemented_exports_name_what_is_missing() {
        let generator =
            ReportGenerator::new(ReportConfig::default()).expect("generator construction");
        let report = Report {
            metadata: ReportMetadata {
                title: "t".to_string(),
                subtitle: None,
                author: "a".to_string(),
                organization: None,
                version: "1".to_string(),
                generation_time_ms: 0.0,
                additional_metadata: HashMap::new(),
            },
            sections: Vec::new(),
            visualizations: HashMap::new(),
            raw_data: HashMap::new(),
            generated_at: chrono::Utc::now(),
        };
        let pdf = generator.export_pdf(&report).expect_err("pdf must be refused");
        assert!(format!("{pdf:?}").contains("PDF writer"), "{pdf:?}");
        let xls = generator.export_excel(&report).expect_err("excel must be refused");
        assert!(format!("{xls:?}").contains("data_export"), "{xls:?}");
        let ppt = generator.export_powerpoint(&report).expect_err("pptx must be refused");
        assert!(format!("{ppt:?}").contains("OOXML"), "{ppt:?}");
    }
    use crate::DebugConfig;

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
        let generator = ReportGenerator::new(config).expect("format should be implemented");
        assert!(generator.debug_data.is_none());
        assert!(generator.profiling_data.is_none());
    }

    /// Regression test: PDF used to be a silently-accepted `ReportConfig`
    /// value that only failed inside `export_report`, after a caller had
    /// already generated the full report. `ReportGenerator::new` must now
    /// reject it immediately with a clear message.
    #[test]
    fn test_new_rejects_pdf_format_immediately_instead_of_at_export_time() {
        let config = ReportConfig {
            format: ReportFormat::Pdf,
            ..Default::default()
        };

        let err = ReportGenerator::new(config)
            .expect_err("PDF must be rejected at construction, not accepted and failed later");
        let message = err.to_string();
        assert!(
            message.contains("PDF"),
            "error should name the rejected format: {message}"
        );
    }

    /// Companion: Excel and PowerPoint are the other two `ReportFormat`
    /// variants with no real writer behind them; both must be rejected the
    /// same way as PDF, not just PDF alone.
    #[test]
    fn test_new_rejects_excel_and_powerpoint_formats() {
        for format in [ReportFormat::Excel, ReportFormat::PowerPoint] {
            let config = ReportConfig {
                format,
                ..Default::default()
            };
            assert!(
                ReportGenerator::new(config).is_err(),
                "unimplemented export formats must be rejected at construction"
            );
        }
    }

    /// Companion: formats that *do* have a real writer (see `export_html`,
    /// `export_markdown`, `export_json`, `export_jupyter`, `export_latex`)
    /// must still construct successfully -- the fix must not become an
    /// overly broad rejection of every format.
    #[test]
    fn test_new_accepts_every_implemented_format() {
        for format in [
            ReportFormat::Markdown,
            ReportFormat::Html,
            ReportFormat::Json,
            ReportFormat::Jupyter,
            ReportFormat::Latex,
        ] {
            let config = ReportConfig {
                format,
                ..Default::default()
            };
            assert!(
                ReportGenerator::new(config).is_ok(),
                "implemented export formats must not be rejected"
            );
        }
    }

    #[test]
    fn test_report_generation() {
        let config = ReportConfig {
            title: "Test Report".to_string(),
            format: ReportFormat::Json,
            report_type: ReportType::DebugReport,
            ..Default::default()
        };

        let generator = ReportGenerator::new(config).expect("format should be implemented");
        let report = generator.generate().expect("operation failed in test");

        assert_eq!(report.metadata.title, "Test Report");
        assert!(!report.sections.is_empty());
    }

    #[test]
    fn test_section_generation() {
        let config = ReportConfig::default();
        let generator = ReportGenerator::new(config).expect("format should be implemented");

        let summary = generator.generate_summary_section().expect("operation failed in test");
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

        let generator = ReportGenerator::new(config).expect("format should be implemented");
        let report = generator.generate().expect("operation failed in test");

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
        let generator = ReportGenerator::new(config).expect("format should be implemented");
        let report = generator.generate().expect("operation failed in test");

        let json = serde_json::to_string(&report).expect("JSON serialization failed");
        let deserialized: Report =
            serde_json::from_str(&json).expect("JSON deserialization failed");

        assert_eq!(report.metadata.title, deserialized.metadata.title);
        assert_eq!(report.sections.len(), deserialized.sections.len());
    }

    /// Regression test: the old `generate_architecture_section` printed the
    /// literal string `"N/A"` for every layer's parameter count
    /// unconditionally, regardless of whether any architecture data was
    /// ever provided. With real architecture data attached via
    /// `with_architecture_data`, a layer that has a matching entry there
    /// must show its real parameter count.
    #[tokio::test]
    async fn test_architecture_section_uses_real_parameter_counts_not_na() {
        use crate::architecture_analysis::{
            ArchitectureAnalysisConfig, ArchitectureAnalyzer, LayerInfo, LayerType,
        };
        use crate::gradient_debugger::debugger::{FlowAnalysis, LayerFlowAnalysis};
        use crate::gradient_debugger::GradientDebugger;

        let debugger = GradientDebugger::new(DebugConfig::default());
        let mut gradient_report =
            debugger.generate_report().await.expect("gradient report should generate");
        let mut layer_analyses = HashMap::new();
        layer_analyses.insert(
            "encoder.layer0".to_string(),
            LayerFlowAnalysis {
                layer_name: "encoder.layer0".to_string(),
                is_vanishing: false,
                is_exploding: false,
                gradient_norm: 0.5,
                flow_consistency: 0.9,
            },
        );
        gradient_report.flow_analysis = FlowAnalysis { layer_analyses };

        let mut analyzer = ArchitectureAnalyzer::new(ArchitectureAnalysisConfig::default());
        analyzer.register_layer(LayerInfo {
            id: "0".to_string(),
            name: "encoder.layer0".to_string(),
            layer_type: LayerType::Linear,
            input_shape: vec![768],
            output_shape: vec![768],
            parameters: 590_592,
            trainable_parameters: 590_592,
            memory_usage: 0,
            flops: 0,
            receptive_field: None,
        });
        let architecture_report =
            analyzer.analyze().await.expect("architecture analysis should succeed");

        let generator = ReportGenerator::new(ReportConfig::default())
            .expect("Html is implemented")
            .with_debug_data(gradient_report)
            .with_architecture_data(architecture_report);

        let section = generator
            .generate_architecture_section()
            .expect("architecture section generation should succeed");

        assert!(
            section.content.contains("590592"),
            "must show the real parameter count from architecture data, not N/A: {}",
            section.content
        );
    }

    /// Companion to the above: without `with_architecture_data`, the column
    /// must still honestly say `N/A` -- this is the absence path, distinct
    /// from the bug (a permanent, unconditional `N/A` even when real data
    /// was available).
    #[tokio::test]
    async fn test_architecture_section_reports_na_without_architecture_data() {
        use crate::gradient_debugger::debugger::{FlowAnalysis, LayerFlowAnalysis};
        use crate::gradient_debugger::GradientDebugger;

        let debugger = GradientDebugger::new(DebugConfig::default());
        let mut gradient_report =
            debugger.generate_report().await.expect("gradient report should generate");
        let mut layer_analyses = HashMap::new();
        layer_analyses.insert(
            "encoder.layer0".to_string(),
            LayerFlowAnalysis {
                layer_name: "encoder.layer0".to_string(),
                is_vanishing: false,
                is_exploding: false,
                gradient_norm: 0.5,
                flow_consistency: 0.9,
            },
        );
        gradient_report.flow_analysis = FlowAnalysis { layer_analyses };

        let generator = ReportGenerator::new(ReportConfig::default())
            .expect("Html is implemented")
            .with_debug_data(gradient_report);
        let section = generator
            .generate_architecture_section()
            .expect("architecture section generation should succeed");

        assert!(section.content.contains("N/A"));
    }

    /// Regression test: the old `generate_visualizations` always emitted the
    /// fixed points `[1,2,3]`/`[10,15,12]` for the performance chart
    /// whenever any profiling data was attached, regardless of its content.
    /// The chart must instead reflect the real `slowest_layers` list.
    #[test]
    fn test_visualizations_reflect_real_slowest_layers_not_fixed_points() {
        use crate::profiler::{MemoryEfficiencyAnalysis, ProfilerReport};
        use std::time::Duration;

        let profiling_data = ProfilerReport {
            total_events: 2,
            total_runtime: Duration::from_millis(42),
            statistics: HashMap::new(),
            bottlenecks: Vec::new(),
            slowest_layers: vec![
                ("attention.0".to_string(), Duration::from_millis(30)),
                ("mlp.0".to_string(), Duration::from_millis(12)),
            ],
            memory_efficiency: MemoryEfficiencyAnalysis::default(),
            recommendations: Vec::new(),
        };

        let generator = ReportGenerator::new(ReportConfig::default())
            .expect("Html is implemented")
            .with_profiling_data(profiling_data);
        let visualizations = generator
            .generate_visualizations()
            .expect("visualization generation should succeed");

        let chart = visualizations
            .get("performance_chart")
            .expect("a performance chart should be produced from real slowest_layers data");
        assert_eq!(
            chart.y_values,
            vec![30.0, 12.0],
            "must reflect real layer durations in ms"
        );
        assert_ne!(
            chart.y_values,
            vec![10.0, 15.0, 12.0],
            "must not be the old fabricated placeholder points"
        );
        assert_eq!(
            chart.labels,
            vec!["attention.0".to_string(), "mlp.0".to_string()]
        );
    }
}
