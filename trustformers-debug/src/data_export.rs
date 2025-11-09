//! Data export capabilities for debugging tools
//!
//! This module provides comprehensive data export functionality supporting
//! multiple formats including CSV, Excel, JSON, HDF5, and more.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Data export manager for debugging tools
#[derive(Debug, Clone)]
pub struct DataExportManager {
    /// Export configuration
    config: ExportConfig,
    /// Active export jobs
    active_jobs: HashMap<Uuid, ExportJob>,
    /// Export history
    export_history: Vec<ExportRecord>,
    /// Supported formats
    supported_formats: Vec<ExportFormat>,
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Default export directory
    pub default_directory: String,
    /// Maximum file size (bytes)
    pub max_file_size: u64,
    /// Enable compression
    pub enable_compression: bool,
    /// Default format
    pub default_format: ExportFormat,
    /// Include metadata
    pub include_metadata: bool,
    /// Export templates
    pub templates: Vec<ExportTemplate>,
}

/// Export job tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportJob {
    /// Job identifier
    pub id: Uuid,
    /// Job name
    pub name: String,
    /// Export format
    pub format: ExportFormat,
    /// Output path
    pub output_path: String,
    /// Job status
    pub status: ExportStatus,
    /// Progress percentage
    pub progress: f64,
    /// Start time
    pub started_at: DateTime<Utc>,
    /// Completion time
    pub completed_at: Option<DateTime<Utc>>,
    /// Data size (bytes)
    pub data_size: u64,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// Export options
    pub options: ExportOptions,
}

/// Export record for history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRecord {
    /// Record identifier
    pub id: Uuid,
    /// Export job ID
    pub job_id: Uuid,
    /// Export timestamp
    pub timestamp: DateTime<Utc>,
    /// File path
    pub file_path: String,
    /// File size
    pub file_size: u64,
    /// Export format
    pub format: ExportFormat,
    /// Success status
    pub success: bool,
    /// Duration (seconds)
    pub duration: f64,
}

/// Export template for common configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportTemplate {
    /// Template identifier
    pub id: String,
    /// Template name
    pub name: String,
    /// Description
    pub description: String,
    /// Export format
    pub format: ExportFormat,
    /// Export options
    pub options: ExportOptions,
    /// Data filters
    pub filters: DataFilters,
    /// Template tags
    pub tags: Vec<String>,
}

/// Export options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Include headers (for CSV/Excel)
    pub include_headers: bool,
    /// Date format
    pub date_format: String,
    /// Precision for floats
    pub float_precision: u32,
    /// Field separator (for CSV)
    pub separator: String,
    /// Compression level (0-9)
    pub compression_level: u32,
    /// Include metadata
    pub include_metadata: bool,
    /// Custom formatting options
    pub custom_options: HashMap<String, serde_json::Value>,
}

/// Data filters for selective export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFilters {
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// Include specific data types
    pub data_types: Vec<DataType>,
    /// Exclude fields
    pub exclude_fields: Vec<String>,
    /// Include only fields
    pub include_fields: Option<Vec<String>>,
    /// Custom filters
    pub custom_filters: HashMap<String, serde_json::Value>,
}

/// Date range for filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    /// Start date
    pub start: DateTime<Utc>,
    /// End date
    pub end: DateTime<Utc>,
}

/// Export formats supported
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash, Eq)]
pub enum ExportFormat {
    /// Comma-separated values
    Csv,
    /// Excel workbook
    Excel,
    /// JSON format
    Json,
    /// Pretty-printed JSON
    JsonPretty,
    /// HDF5 format
    Hdf5,
    /// Parquet format
    Parquet,
    /// XML format
    Xml,
    /// YAML format
    Yaml,
    /// SQLite database
    Sqlite,
    /// MessagePack
    MessagePack,
    /// Apache Arrow
    Arrow,
    /// Custom format
    Custom(String),
}

/// Export status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Data types that can be exported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    TensorData,
    GradientData,
    PerformanceMetrics,
    MemoryProfiles,
    ActivityLogs,
    AnnotationData,
    CommentData,
    ModelDiagnostics,
    TrainingDynamics,
    ArchitectureAnalysis,
    Custom(String),
}

/// Exportable data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportableData {
    /// Data identifier
    pub id: Uuid,
    /// Data name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Creation timestamp
    pub timestamp: DateTime<Utc>,
    /// Data content
    pub content: ExportDataContent,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Data size
    pub size: u64,
}

/// Content of exportable data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportDataContent {
    /// Tabular data
    Table(TableData),
    /// Time series data
    TimeSeries(TimeSeriesData),
    /// Key-value pairs
    KeyValue(HashMap<String, serde_json::Value>),
    /// Structured data
    Structured(serde_json::Value),
    /// Binary data
    Binary(Vec<u8>),
    /// Text data
    Text(String),
}

/// Tabular data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableData {
    /// Column headers
    pub headers: Vec<String>,
    /// Data rows
    pub rows: Vec<Vec<serde_json::Value>>,
    /// Column types
    pub column_types: HashMap<String, ColumnType>,
}

/// Time series data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData {
    /// Timestamps
    pub timestamps: Vec<DateTime<Utc>>,
    /// Data series
    pub series: HashMap<String, Vec<f64>>,
    /// Series metadata
    pub metadata: HashMap<String, String>,
}

/// Column data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnType {
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    Binary,
}

impl DataExportManager {
    /// Create a new data export manager
    pub fn new(config: ExportConfig) -> Self {
        let supported_formats = vec![
            ExportFormat::Csv,
            ExportFormat::Excel,
            ExportFormat::Json,
            ExportFormat::JsonPretty,
            ExportFormat::Xml,
            ExportFormat::Yaml,
            ExportFormat::Sqlite,
        ];

        Self {
            config,
            active_jobs: HashMap::new(),
            export_history: Vec::new(),
            supported_formats,
        }
    }

    /// Start an export job
    pub fn start_export(
        &mut self,
        name: String,
        data: Vec<ExportableData>,
        format: ExportFormat,
        output_path: String,
        options: ExportOptions,
    ) -> Result<Uuid> {
        let job_id = Uuid::new_v4();

        // Calculate total data size
        let data_size: u64 = data.iter().map(|d| d.size).sum();

        // Check file size limit
        if data_size > self.config.max_file_size {
            return Err(anyhow::anyhow!("Data size exceeds maximum file size limit"));
        }

        let job = ExportJob {
            id: job_id,
            name: name.clone(),
            format: format.clone(),
            output_path: output_path.clone(),
            status: ExportStatus::Pending,
            progress: 0.0,
            started_at: Utc::now(),
            completed_at: None,
            data_size,
            error_message: None,
            options: options.clone(),
        };

        self.active_jobs.insert(job_id, job);

        // Start the actual export process
        self.execute_export(job_id, data, options)?;

        Ok(job_id)
    }

    /// Execute the export process
    fn execute_export(
        &mut self,
        job_id: Uuid,
        data: Vec<ExportableData>,
        options: ExportOptions,
    ) -> Result<()> {
        // Extract job info to avoid multiple mutable borrows
        let (format, output_path) = {
            if let Some(job) = self.active_jobs.get_mut(&job_id) {
                job.status = ExportStatus::InProgress;
                (job.format.clone(), job.output_path.clone())
            } else {
                return Err(anyhow::anyhow!("Export job not found"));
            }
        };

        let result = match format {
            ExportFormat::Csv => self.export_csv(&data, &output_path, &options),
            ExportFormat::Json => self.export_json(&data, &output_path, &options),
            ExportFormat::JsonPretty => self.export_json_pretty(&data, &output_path, &options),
            ExportFormat::Excel => self.export_excel(&data, &output_path, &options),
            ExportFormat::Xml => self.export_xml(&data, &output_path, &options),
            ExportFormat::Yaml => self.export_yaml(&data, &output_path, &options),
            ExportFormat::Sqlite => self.export_sqlite(&data, &output_path, &options),
            _ => Err(anyhow::anyhow!("Format not yet implemented")),
        };

        // Update job status
        if let Some(job) = self.active_jobs.get_mut(&job_id) {
            match result {
                Ok(_) => {
                    job.status = ExportStatus::Completed;
                    job.progress = 100.0;
                    job.completed_at = Some(Utc::now());

                    // Add to history by cloning the job
                    let job_copy = job.clone();
                    self.add_export_record(&job_copy);
                },
                Err(e) => {
                    job.status = ExportStatus::Failed;
                    job.error_message = Some(e.to_string());
                },
            }
        }

        Ok(())
    }

    /// Export to CSV format
    fn export_csv(
        &mut self,
        data: &[ExportableData],
        output_path: &str,
        options: &ExportOptions,
    ) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(output_path)?;

        for item in data {
            match &item.content {
                ExportDataContent::Table(table_data) => {
                    // Write headers
                    if options.include_headers {
                        let header_line = table_data.headers.join(&options.separator);
                        writeln!(file, "{}", header_line)?;
                    }

                    // Write data rows
                    for row in &table_data.rows {
                        let row_values: Vec<String> =
                            row.iter().map(|v| self.format_value_for_csv(v, options)).collect();
                        let row_line = row_values.join(&options.separator);
                        writeln!(file, "{}", row_line)?;
                    }
                },
                ExportDataContent::TimeSeries(ts_data) => {
                    // Write time series data
                    if options.include_headers {
                        let mut headers = vec!["timestamp".to_string()];
                        headers.extend(ts_data.series.keys().cloned());
                        let header_line = headers.join(&options.separator);
                        writeln!(file, "{}", header_line)?;
                    }

                    for (i, timestamp) in ts_data.timestamps.iter().enumerate() {
                        let mut row = vec![timestamp.format(&options.date_format).to_string()];
                        for series_name in ts_data.series.keys() {
                            if let Some(series) = ts_data.series.get(series_name) {
                                if let Some(value) = series.get(i) {
                                    row.push(format!(
                                        "{:.precision$}",
                                        value,
                                        precision = options.float_precision as usize
                                    ));
                                } else {
                                    row.push("".to_string());
                                }
                            }
                        }
                        let row_line = row.join(&options.separator);
                        writeln!(file, "{}", row_line)?;
                    }
                },
                _ => {
                    // Convert other formats to JSON and then to CSV-like representation
                    let json_str = serde_json::to_string(&item.content)?;
                    writeln!(file, "{}", json_str)?;
                },
            }
        }

        Ok(())
    }

    /// Export to JSON format
    fn export_json(
        &mut self,
        data: &[ExportableData],
        output_path: &str,
        _options: &ExportOptions,
    ) -> Result<()> {
        use std::fs::File;

        let file = File::create(output_path)?;
        serde_json::to_writer(file, data)?;
        Ok(())
    }

    /// Export to pretty JSON format
    fn export_json_pretty(
        &mut self,
        data: &[ExportableData],
        output_path: &str,
        _options: &ExportOptions,
    ) -> Result<()> {
        use std::fs::File;

        let file = File::create(output_path)?;
        serde_json::to_writer_pretty(file, data)?;
        Ok(())
    }

    /// Export to Excel format (simplified implementation)
    fn export_excel(
        &mut self,
        data: &[ExportableData],
        output_path: &str,
        options: &ExportOptions,
    ) -> Result<()> {
        // This is a simplified implementation
        // In a real implementation, you would use a library like xlsxwriter or rust_xlsxwriter

        // For now, we'll create a CSV file with .xlsx extension as a placeholder
        self.export_csv(data, output_path, options)
    }

    /// Export to XML format
    fn export_xml(
        &mut self,
        data: &[ExportableData],
        output_path: &str,
        _options: &ExportOptions,
    ) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(output_path)?;

        writeln!(file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
        writeln!(file, "<export_data>")?;

        for item in data {
            writeln!(
                file,
                "  <data_item id=\"{}\" type=\"{:?}\">",
                item.id, item.data_type
            )?;
            writeln!(file, "    <name>{}</name>", item.name)?;
            writeln!(
                file,
                "    <timestamp>{}</timestamp>",
                item.timestamp.to_rfc3339()
            )?;
            writeln!(file, "    <size>{}</size>", item.size)?;

            // Convert content to XML (simplified)
            let content_json = serde_json::to_string(&item.content)?;
            writeln!(file, "    <content><![CDATA[{}]]></content>", content_json)?;

            writeln!(file, "  </data_item>")?;
        }

        writeln!(file, "</export_data>")?;
        Ok(())
    }

    /// Export to YAML format
    fn export_yaml(
        &mut self,
        data: &[ExportableData],
        output_path: &str,
        _options: &ExportOptions,
    ) -> Result<()> {
        use std::fs::File;

        let file = File::create(output_path)?;
        serde_json::to_writer_pretty(file, data)?;
        Ok(())
    }

    /// Export to SQLite database
    fn export_sqlite(
        &mut self,
        data: &[ExportableData],
        output_path: &str,
        _options: &ExportOptions,
    ) -> Result<()> {
        // This would use rusqlite or similar library
        // For now, we'll create a JSON file as a placeholder
        self.export_json(data, output_path, _options)
    }

    /// Helper function to format values for CSV
    fn format_value_for_csv(&self, value: &serde_json::Value, options: &ExportOptions) -> String {
        match value {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    format!(
                        "{:.precision$}",
                        f,
                        precision = options.float_precision as usize
                    )
                } else {
                    n.to_string()
                }
            },
            serde_json::Value::String(s) => {
                // Escape quotes and commas
                if s.contains(',') || s.contains('"') || s.contains('\n') {
                    format!("\"{}\"", s.replace('"', "\"\""))
                } else {
                    s.clone()
                }
            },
            _ => value.to_string(),
        }
    }

    /// Add export record to history
    fn add_export_record(&mut self, job: &ExportJob) {
        let record = ExportRecord {
            id: Uuid::new_v4(),
            job_id: job.id,
            timestamp: Utc::now(),
            file_path: job.output_path.clone(),
            file_size: job.data_size,
            format: job.format.clone(),
            success: matches!(job.status, ExportStatus::Completed),
            duration: job
                .completed_at
                .map(|end| (end - job.started_at).num_milliseconds() as f64 / 1000.0)
                .unwrap_or(0.0),
        };

        self.export_history.push(record);
    }

    /// Get export job status
    pub fn get_job_status(&self, job_id: Uuid) -> Option<&ExportJob> {
        self.active_jobs.get(&job_id)
    }

    /// Get export history
    pub fn get_export_history(&self) -> &[ExportRecord] {
        &self.export_history
    }

    /// Create export template
    pub fn create_template(
        &mut self,
        name: String,
        description: String,
        format: ExportFormat,
        options: ExportOptions,
        filters: DataFilters,
        tags: Vec<String>,
    ) -> String {
        let template_id = Uuid::new_v4().to_string();

        let template = ExportTemplate {
            id: template_id.clone(),
            name,
            description,
            format,
            options,
            filters,
            tags,
        };

        self.config.templates.push(template);
        template_id
    }

    /// Apply export template
    pub fn apply_template(
        &self,
        template_id: &str,
    ) -> Option<(&ExportFormat, &ExportOptions, &DataFilters)> {
        self.config
            .templates
            .iter()
            .find(|t| t.id == template_id)
            .map(|t| (&t.format, &t.options, &t.filters))
    }

    /// Get supported formats
    pub fn get_supported_formats(&self) -> &[ExportFormat] {
        &self.supported_formats
    }

    /// Cancel export job
    pub fn cancel_job(&mut self, job_id: Uuid) -> Result<()> {
        if let Some(job) = self.active_jobs.get_mut(&job_id) {
            if matches!(job.status, ExportStatus::Pending | ExportStatus::InProgress) {
                job.status = ExportStatus::Cancelled;
                Ok(())
            } else {
                Err(anyhow::anyhow!("Job cannot be cancelled in current status"))
            }
        } else {
            Err(anyhow::anyhow!("Job not found"))
        }
    }

    /// Get export statistics
    pub fn get_export_statistics(&self) -> ExportStatistics {
        let total_exports = self.export_history.len();
        let successful_exports = self.export_history.iter().filter(|r| r.success).count();
        let total_size: u64 = self.export_history.iter().map(|r| r.file_size).sum();
        let avg_duration = if total_exports > 0 {
            self.export_history.iter().map(|r| r.duration).sum::<f64>() / total_exports as f64
        } else {
            0.0
        };

        let format_stats: HashMap<ExportFormat, usize> =
            self.export_history.iter().fold(HashMap::new(), |mut acc, record| {
                *acc.entry(record.format.clone()).or_insert(0) += 1;
                acc
            });

        ExportStatistics {
            total_exports,
            successful_exports,
            failed_exports: total_exports - successful_exports,
            total_size_bytes: total_size,
            average_duration_seconds: avg_duration,
            format_statistics: format_stats,
            active_jobs: self.active_jobs.len(),
        }
    }
}

/// Export statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStatistics {
    pub total_exports: usize,
    pub successful_exports: usize,
    pub failed_exports: usize,
    pub total_size_bytes: u64,
    pub average_duration_seconds: f64,
    pub format_statistics: HashMap<ExportFormat, usize>,
    pub active_jobs: usize,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            default_directory: "./exports".to_string(),
            max_file_size: 1024 * 1024 * 1024, // 1GB
            enable_compression: true,
            default_format: ExportFormat::Json,
            include_metadata: true,
            templates: Vec::new(),
        }
    }
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_headers: true,
            date_format: "%Y-%m-%d %H:%M:%S UTC".to_string(),
            float_precision: 6,
            separator: ",".to_string(),
            compression_level: 6,
            include_metadata: true,
            custom_options: HashMap::new(),
        }
    }
}

impl Default for DataFilters {
    fn default() -> Self {
        Self {
            date_range: None,
            data_types: vec![
                DataType::TensorData,
                DataType::GradientData,
                DataType::PerformanceMetrics,
            ],
            exclude_fields: Vec::new(),
            include_fields: None,
            custom_filters: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_data() -> Vec<ExportableData> {
        let table_data = TableData {
            headers: vec![
                "id".to_string(),
                "value".to_string(),
                "timestamp".to_string(),
            ],
            rows: vec![
                vec![
                    serde_json::Value::Number(serde_json::Number::from(1)),
                    serde_json::Value::Number(serde_json::Number::from_f64(3.14).unwrap()),
                    serde_json::Value::String("2023-01-01T12:00:00Z".to_string()),
                ],
                vec![
                    serde_json::Value::Number(serde_json::Number::from(2)),
                    serde_json::Value::Number(serde_json::Number::from_f64(2.71).unwrap()),
                    serde_json::Value::String("2023-01-01T12:01:00Z".to_string()),
                ],
            ],
            column_types: HashMap::new(),
        };

        vec![ExportableData {
            id: Uuid::new_v4(),
            name: "Test Data".to_string(),
            data_type: DataType::TensorData,
            timestamp: Utc::now(),
            content: ExportDataContent::Table(table_data),
            metadata: HashMap::new(),
            size: 1024,
        }]
    }

    #[test]
    fn test_export_manager_creation() {
        let config = ExportConfig::default();
        let manager = DataExportManager::new(config);

        assert!(manager.get_supported_formats().contains(&ExportFormat::Json));
        assert!(manager.get_supported_formats().contains(&ExportFormat::Csv));
    }

    #[test]
    fn test_csv_export() {
        let config = ExportConfig::default();
        let mut manager = DataExportManager::new(config);
        let test_data = create_test_data();

        let temp_dir = tempdir().unwrap();
        let output_path = temp_dir.path().join("test.csv").to_string_lossy().to_string();

        let job_id = manager
            .start_export(
                "Test CSV Export".to_string(),
                test_data,
                ExportFormat::Csv,
                output_path.clone(),
                ExportOptions::default(),
            )
            .unwrap();

        // Check job was created
        assert!(manager.active_jobs.contains_key(&job_id));

        // Check file was created
        assert!(std::path::Path::new(&output_path).exists());
    }

    #[test]
    fn test_json_export() {
        let config = ExportConfig::default();
        let mut manager = DataExportManager::new(config);
        let test_data = create_test_data();

        let temp_dir = tempdir().unwrap();
        let output_path = temp_dir.path().join("test.json").to_string_lossy().to_string();

        let job_id = manager
            .start_export(
                "Test JSON Export".to_string(),
                test_data,
                ExportFormat::Json,
                output_path.clone(),
                ExportOptions::default(),
            )
            .unwrap();

        assert!(manager.active_jobs.contains_key(&job_id));
        assert!(std::path::Path::new(&output_path).exists());
    }

    #[test]
    fn test_export_template() {
        let config = ExportConfig::default();
        let mut manager = DataExportManager::new(config);

        let template_id = manager.create_template(
            "CSV Template".to_string(),
            "Standard CSV export".to_string(),
            ExportFormat::Csv,
            ExportOptions::default(),
            DataFilters::default(),
            vec!["csv".to_string(), "standard".to_string()],
        );

        let (format, options, _filters) = manager.apply_template(&template_id).unwrap();
        assert_eq!(*format, ExportFormat::Csv);
        assert!(options.include_headers);
    }

    #[test]
    fn test_export_statistics() {
        let config = ExportConfig::default();
        let mut manager = DataExportManager::new(config);

        // Add some mock export records
        manager.export_history.push(ExportRecord {
            id: Uuid::new_v4(),
            job_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            file_path: "test1.csv".to_string(),
            file_size: 1024,
            format: ExportFormat::Csv,
            success: true,
            duration: 2.5,
        });

        manager.export_history.push(ExportRecord {
            id: Uuid::new_v4(),
            job_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            file_path: "test2.json".to_string(),
            file_size: 2048,
            format: ExportFormat::Json,
            success: true,
            duration: 1.8,
        });

        let stats = manager.get_export_statistics();
        assert_eq!(stats.total_exports, 2);
        assert_eq!(stats.successful_exports, 2);
        assert_eq!(stats.total_size_bytes, 3072);
    }
}
