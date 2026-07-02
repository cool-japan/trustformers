//! Data export capabilities for debugging tools
//!
//! This module provides comprehensive data export functionality supporting
//! multiple formats including CSV, Excel, JSON, HDF5, and more.

use anyhow::Result;
use chrono::{DateTime, Utc};
use oxisql_sqlite_compat::blocking::SqliteConnectionBlocking;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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
            // Binary/columnar formats that require optional external crates or services.
            // These are intentionally not implemented to keep the core dependency set
            // minimal.  Callers should use JSON or CSV as a universal fallback.
            ExportFormat::Hdf5 => Err(anyhow::anyhow!(
                "HDF5 export is not supported in this build. Use JSON or CSV instead."
            )),
            ExportFormat::Parquet => Err(anyhow::anyhow!(
                "Parquet export is not supported in this build. Use JSON or CSV instead."
            )),
            ExportFormat::MessagePack => Err(anyhow::anyhow!(
                "MessagePack export is not supported in this build. Use JSON instead."
            )),
            ExportFormat::Arrow => Err(anyhow::anyhow!(
                "Apache Arrow export is not supported in this build. Use JSON or CSV instead."
            )),
            ExportFormat::Custom(ref name) => Err(anyhow::anyhow!(
                "Custom export format '{}' is not registered. \
                 Register a handler or use one of the built-in formats.",
                name
            )),
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

    /// Export to Excel (.xlsx) format as a real Office Open XML workbook
    fn export_excel(
        &mut self,
        data: &[ExportableData],
        output_path: &str,
        options: &ExportOptions,
    ) -> Result<()> {
        use oxiarc_archive::zip::ZipWriter;

        let rows = build_xlsx_rows(data, options)?;
        let sheet_xml = build_xlsx_sheet(&rows);

        let mut buffer: Vec<u8> = Vec::new();
        {
            let mut writer = ZipWriter::new(&mut buffer);
            writer
                .add_file("[Content_Types].xml", XLSX_CONTENT_TYPES.as_bytes())
                .map_err(|e| anyhow::anyhow!("xlsx: failed to write [Content_Types].xml: {e}"))?;
            writer
                .add_file("_rels/.rels", XLSX_ROOT_RELS.as_bytes())
                .map_err(|e| anyhow::anyhow!("xlsx: failed to write _rels/.rels: {e}"))?;
            writer
                .add_file("xl/workbook.xml", XLSX_WORKBOOK.as_bytes())
                .map_err(|e| anyhow::anyhow!("xlsx: failed to write xl/workbook.xml: {e}"))?;
            writer
                .add_file("xl/_rels/workbook.xml.rels", XLSX_WORKBOOK_RELS.as_bytes())
                .map_err(|e| {
                    anyhow::anyhow!("xlsx: failed to write xl/_rels/workbook.xml.rels: {e}")
                })?;
            writer.add_file("xl/worksheets/sheet1.xml", sheet_xml.as_bytes()).map_err(|e| {
                anyhow::anyhow!("xlsx: failed to write xl/worksheets/sheet1.xml: {e}")
            })?;
            writer
                .finish()
                .map_err(|e| anyhow::anyhow!("xlsx: failed to finalize workbook package: {e}"))?;
        }

        std::fs::write(output_path, &buffer)?;
        Ok(())
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

    /// Export to a real SQLite database file.
    ///
    /// Uses the COOLJAPAN Pure-Rust `oxisql-sqlite-compat` backend (a C-free fork
    /// of Limbo) — never `libsqlite3`/`rusqlite`. Each [`ExportableData`] item is
    /// written into its own table:
    ///
    /// * [`ExportDataContent::Table`] → a table with one column per header,
    ///   with SQL column types taken from the explicit `column_types` map or
    ///   inferred from the data (`INTEGER` / `REAL` / `TEXT`).
    /// * [`ExportDataContent::TimeSeries`] → a table with a `timestamp` column
    ///   plus one `REAL` column per series.
    /// * Any other content → a single-column `content TEXT` table holding the
    ///   JSON serialization so nothing is silently dropped.
    fn export_sqlite(
        &mut self,
        data: &[ExportableData],
        output_path: &str,
        options: &ExportOptions,
    ) -> Result<()> {
        // Create the database file fresh so the export is deterministic.
        if std::path::Path::new(output_path).exists() {
            std::fs::remove_file(output_path).map_err(|e| {
                anyhow::anyhow!("failed to clear existing SQLite file '{output_path}': {e}")
            })?;
        }

        let conn = SqliteConnectionBlocking::open(output_path)
            .map_err(|e| anyhow::anyhow!("failed to open SQLite database '{output_path}': {e}"))?;

        let mut used_names: HashSet<String> = HashSet::new();
        for (index, item) in data.iter().enumerate() {
            let table_name = unique_table_name(&item.name, index, &mut used_names);
            match &item.content {
                ExportDataContent::Table(table) => {
                    write_sqlite_table(
                        &conn,
                        &table_name,
                        &table.headers,
                        &table.rows,
                        &table.column_types,
                    )?;
                },
                ExportDataContent::TimeSeries(ts) => {
                    write_sqlite_timeseries(&conn, &table_name, ts, options)?;
                },
                other => {
                    let json = serde_json::to_string(other)?;
                    conn.execute(
                        &format!("CREATE TABLE IF NOT EXISTS \"{table_name}\" (content TEXT)"),
                        &[],
                    )
                    .map_err(|e| anyhow::anyhow!("CREATE TABLE '{table_name}' failed: {e}"))?;
                    conn.execute(
                        &format!(
                            "INSERT INTO \"{table_name}\" (content) VALUES ({})",
                            quote_sql_string(&json)
                        ),
                        &[],
                    )
                    .map_err(|e| anyhow::anyhow!("INSERT into '{table_name}' failed: {e}"))?;
                },
            }
        }

        Ok(())
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

/// OOXML `[Content_Types].xml` part declaring the package content types.
const XLSX_CONTENT_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>"#;

/// OOXML `_rels/.rels` package-level relationships part.
const XLSX_ROOT_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#;

/// OOXML `xl/workbook.xml` part defining a single worksheet.
const XLSX_WORKBOOK: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>"#;

/// OOXML `xl/_rels/workbook.xml.rels` part linking the workbook to its worksheet.
const XLSX_WORKBOOK_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#;

/// A single spreadsheet cell value to emit into a worksheet.
enum XlsxCell {
    /// Inline (shared-string-free) text cell, emitted as `<c t="inlineStr">`.
    Inline(String),
    /// Numeric cell, emitted as `<c><v>..</v></c>` with the pre-formatted literal.
    Number(String),
    /// Empty placeholder cell.
    Empty,
}

/// XML-escape `&`, `<`, `>`, `"`, `'` for safe inclusion in OOXML parts.
fn xlsx_xml_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Convert a zero-based column index to an Excel column name (0 -> "A", 26 -> "AA").
fn xlsx_column_name(index: usize) -> String {
    let mut n = index + 1;
    let mut name = String::new();
    while n > 0 {
        let rem = (n - 1) % 26;
        name.insert(0, (b'A' + rem as u8) as char);
        n = (n - 1) / 26;
    }
    name
}

/// Map a JSON value to a worksheet cell, mirroring `format_value_for_csv` numeric rules.
fn xlsx_value_to_cell(value: &serde_json::Value, options: &ExportOptions) -> XlsxCell {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                XlsxCell::Number(format!(
                    "{:.precision$}",
                    f,
                    precision = options.float_precision as usize
                ))
            } else {
                XlsxCell::Number(n.to_string())
            }
        },
        serde_json::Value::String(s) => XlsxCell::Inline(s.clone()),
        _ => XlsxCell::Inline(value.to_string()),
    }
}

/// Build the worksheet rows from the export data, mirroring `export_csv` layout.
fn build_xlsx_rows(data: &[ExportableData], options: &ExportOptions) -> Result<Vec<Vec<XlsxCell>>> {
    let mut rows: Vec<Vec<XlsxCell>> = Vec::new();
    for item in data {
        match &item.content {
            ExportDataContent::Table(table_data) => {
                if options.include_headers {
                    rows.push(
                        table_data.headers.iter().map(|h| XlsxCell::Inline(h.clone())).collect(),
                    );
                }
                for row in &table_data.rows {
                    rows.push(row.iter().map(|v| xlsx_value_to_cell(v, options)).collect());
                }
            },
            ExportDataContent::TimeSeries(ts_data) => {
                if options.include_headers {
                    let mut header = vec![XlsxCell::Inline("timestamp".to_string())];
                    header.extend(ts_data.series.keys().map(|k| XlsxCell::Inline(k.clone())));
                    rows.push(header);
                }
                for (i, timestamp) in ts_data.timestamps.iter().enumerate() {
                    let mut cells = vec![XlsxCell::Inline(
                        timestamp.format(&options.date_format).to_string(),
                    )];
                    for series_name in ts_data.series.keys() {
                        if let Some(series) = ts_data.series.get(series_name) {
                            if let Some(value) = series.get(i) {
                                cells.push(XlsxCell::Number(format!(
                                    "{:.precision$}",
                                    value,
                                    precision = options.float_precision as usize
                                )));
                            } else {
                                cells.push(XlsxCell::Empty);
                            }
                        }
                    }
                    rows.push(cells);
                }
            },
            _ => {
                let json_str = serde_json::to_string(&item.content)?;
                rows.push(vec![XlsxCell::Inline(json_str)]);
            },
        }
    }
    Ok(rows)
}

/// Render worksheet rows into the `xl/worksheets/sheet1.xml` OOXML part.
fn build_xlsx_sheet(rows: &[Vec<XlsxCell>]) -> String {
    let mut sheet = String::new();
    sheet.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>");
    sheet.push_str(
        "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">",
    );
    sheet.push_str("<sheetData>");
    for (row_idx, row) in rows.iter().enumerate() {
        let row_num = row_idx + 1;
        sheet.push_str(&format!("<row r=\"{row_num}\">"));
        for (col_idx, cell) in row.iter().enumerate() {
            let cell_ref = format!("{}{row_num}", xlsx_column_name(col_idx));
            match cell {
                XlsxCell::Inline(text) => {
                    sheet.push_str(&format!(
                        "<c r=\"{cell_ref}\" t=\"inlineStr\"><is><t xml:space=\"preserve\">{}</t></is></c>",
                        xlsx_xml_escape(text)
                    ));
                },
                XlsxCell::Number(num) => {
                    sheet.push_str(&format!("<c r=\"{cell_ref}\"><v>{num}</v></c>"));
                },
                XlsxCell::Empty => {
                    sheet.push_str(&format!("<c r=\"{cell_ref}\"/>"));
                },
            }
        }
        sheet.push_str("</row>");
    }
    sheet.push_str("</sheetData></worksheet>");
    sheet
}

// ── SQLite export helpers ───────────────────────────────────────────────────────

/// Sanitize an arbitrary name into a safe, double-quotable SQL identifier.
///
/// Non-alphanumeric characters become `_`; a leading non-alphabetic character is
/// prefixed so the identifier is always valid.
fn sanitize_sql_identifier(name: &str) -> String {
    let mut sanitized: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    let needs_prefix = sanitized
        .chars()
        .next()
        .map(|c| !(c.is_ascii_alphabetic() || c == '_'))
        .unwrap_or(true);
    if needs_prefix {
        sanitized = format!("t_{sanitized}");
    }
    sanitized
}

/// Produce a collision-free table name for the export, recording it in `used`.
fn unique_table_name(name: &str, index: usize, used: &mut HashSet<String>) -> String {
    let base = sanitize_sql_identifier(name);
    if used.insert(base.clone()) {
        return base;
    }
    let candidate = format!("{base}_{index}");
    used.insert(candidate.clone());
    candidate
}

/// Quote and escape a string for use as a SQL string literal.
fn quote_sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// Convert a single JSON value into an inline SQL literal.
///
/// * `null`  → `NULL`
/// * `bool`  → `1` / `0` (SQLite has no native boolean)
/// * number  → verbatim numeric literal
/// * string  → single-quote-escaped string literal
/// * array / object → JSON text stored as a string literal
fn json_value_to_sql_literal(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "NULL".to_string(),
        serde_json::Value::Bool(b) => if *b { "1" } else { "0" }.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => quote_sql_string(s),
        other => quote_sql_string(&other.to_string()),
    }
}

/// Infer a SQL column affinity (`INTEGER` / `REAL` / `TEXT`) from the data in a
/// column when no explicit [`ColumnType`] is supplied.
fn infer_sql_column_type(rows: &[Vec<serde_json::Value>], col: usize) -> &'static str {
    let mut integers = 0usize;
    let mut floats = 0usize;
    let mut bools = 0usize;
    let mut others = 0usize;
    for row in rows {
        match row.get(col) {
            None | Some(serde_json::Value::Null) => {},
            Some(serde_json::Value::Number(n)) => {
                if n.is_i64() || n.is_u64() {
                    integers += 1;
                } else {
                    floats += 1;
                }
            },
            Some(serde_json::Value::Bool(_)) => bools += 1,
            Some(_) => others += 1,
        }
    }
    if others > 0 {
        "TEXT"
    } else if floats > 0 {
        "REAL"
    } else if integers > 0 || bools > 0 {
        "INTEGER"
    } else {
        "TEXT"
    }
}

/// Map an explicit [`ColumnType`] to its SQL affinity.
fn column_type_to_sql(column_type: &ColumnType) -> &'static str {
    match column_type {
        ColumnType::Integer | ColumnType::Boolean => "INTEGER",
        ColumnType::Float => "REAL",
        ColumnType::Binary => "BLOB",
        ColumnType::String | ColumnType::DateTime => "TEXT",
    }
}

/// Create and populate a SQLite table from [`TableData`].
fn write_sqlite_table(
    conn: &SqliteConnectionBlocking,
    table_name: &str,
    headers: &[String],
    rows: &[Vec<serde_json::Value>],
    column_types: &HashMap<String, ColumnType>,
) -> Result<()> {
    if headers.is_empty() {
        return Ok(());
    }

    let column_idents: Vec<String> = headers.iter().map(|h| sanitize_sql_identifier(h)).collect();
    let column_defs: Vec<String> = headers
        .iter()
        .enumerate()
        .map(|(idx, header)| {
            let sql_type = column_types
                .get(header)
                .map(column_type_to_sql)
                .unwrap_or_else(|| infer_sql_column_type(rows, idx));
            format!("\"{}\" {}", column_idents[idx], sql_type)
        })
        .collect();

    let create = format!(
        "CREATE TABLE IF NOT EXISTS \"{table_name}\" ({})",
        column_defs.join(", ")
    );
    conn.execute(&create, &[])
        .map_err(|e| anyhow::anyhow!("CREATE TABLE '{table_name}' failed: {e}"))?;

    let column_list =
        column_idents.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(", ");

    for row in rows {
        let values: Vec<String> = (0..headers.len())
            .map(|idx| match row.get(idx) {
                Some(value) => json_value_to_sql_literal(value),
                None => "NULL".to_string(),
            })
            .collect();
        let insert = format!(
            "INSERT INTO \"{table_name}\" ({column_list}) VALUES ({})",
            values.join(", ")
        );
        conn.execute(&insert, &[])
            .map_err(|e| anyhow::anyhow!("INSERT into '{table_name}' failed: {e}"))?;
    }

    Ok(())
}

/// Create and populate a SQLite table from [`TimeSeriesData`].
fn write_sqlite_timeseries(
    conn: &SqliteConnectionBlocking,
    table_name: &str,
    ts: &TimeSeriesData,
    options: &ExportOptions,
) -> Result<()> {
    // Deterministic series ordering.
    let mut series_names: Vec<String> = ts.series.keys().cloned().collect();
    series_names.sort();

    let mut column_defs = vec!["\"timestamp\" TEXT".to_string()];
    for name in &series_names {
        column_defs.push(format!("\"{}\" REAL", sanitize_sql_identifier(name)));
    }
    let create = format!(
        "CREATE TABLE IF NOT EXISTS \"{table_name}\" ({})",
        column_defs.join(", ")
    );
    conn.execute(&create, &[])
        .map_err(|e| anyhow::anyhow!("CREATE TABLE '{table_name}' failed: {e}"))?;

    let mut column_list = vec!["\"timestamp\"".to_string()];
    for name in &series_names {
        column_list.push(format!("\"{}\"", sanitize_sql_identifier(name)));
    }
    let column_list = column_list.join(", ");

    for (i, timestamp) in ts.timestamps.iter().enumerate() {
        let mut values = vec![quote_sql_string(
            &timestamp.format(&options.date_format).to_string(),
        )];
        for name in &series_names {
            match ts.series.get(name).and_then(|series| series.get(i)) {
                Some(value) => values.push(value.to_string()),
                None => values.push("NULL".to_string()),
            }
        }
        let insert = format!(
            "INSERT INTO \"{table_name}\" ({column_list}) VALUES ({})",
            values.join(", ")
        );
        conn.execute(&insert, &[])
            .map_err(|e| anyhow::anyhow!("INSERT into '{table_name}' failed: {e}"))?;
    }

    Ok(())
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

    // These values are test data, not approximations of mathematical constants
    #[allow(clippy::approx_constant)]
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
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(3.14).expect("operation failed in test"),
                    ),
                    serde_json::Value::String("2023-01-01T12:00:00Z".to_string()),
                ],
                vec![
                    serde_json::Value::Number(serde_json::Number::from(2)),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(2.71).expect("operation failed in test"),
                    ),
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

        let temp_dir = tempdir().expect("temp file creation failed");
        let output_path = temp_dir.path().join("test.csv").to_string_lossy().to_string();

        let job_id = manager
            .start_export(
                "Test CSV Export".to_string(),
                test_data,
                ExportFormat::Csv,
                output_path.clone(),
                ExportOptions::default(),
            )
            .expect("operation failed in test");

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

        let temp_dir = tempdir().expect("temp file creation failed");
        let output_path = temp_dir.path().join("test.json").to_string_lossy().to_string();

        let job_id = manager
            .start_export(
                "Test JSON Export".to_string(),
                test_data,
                ExportFormat::Json,
                output_path.clone(),
                ExportOptions::default(),
            )
            .expect("operation failed in test");

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

        let (format, options, _filters) =
            manager.apply_template(&template_id).expect("temp file creation failed");
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

    #[test]
    fn test_excel_export_roundtrip() {
        use oxiarc_archive::zip::ZipReader;
        use std::io::Cursor;

        let config = ExportConfig::default();
        let mut manager = DataExportManager::new(config);
        let test_data = create_test_data();

        let file_name = format!("trustformers_xlsx_test_{}.xlsx", Uuid::new_v4());
        let output_path = std::env::temp_dir().join(file_name).to_string_lossy().to_string();

        manager
            .start_export(
                "Test Excel Export".to_string(),
                test_data,
                ExportFormat::Excel,
                output_path.clone(),
                ExportOptions::default(),
            )
            .expect("excel export should succeed");

        assert!(std::path::Path::new(&output_path).exists());

        let bytes = std::fs::read(&output_path).expect("read xlsx bytes");
        let mut reader = ZipReader::new(Cursor::new(bytes)).expect("open xlsx as zip");
        let entries = reader.entries().to_vec();
        let names: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();

        assert!(
            names.iter().any(|n| n == "[Content_Types].xml"),
            "missing [Content_Types].xml; entries = {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "xl/workbook.xml"),
            "missing xl/workbook.xml; entries = {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "xl/worksheets/sheet1.xml"),
            "missing xl/worksheets/sheet1.xml; entries = {names:?}"
        );

        let sheet_entry = entries
            .iter()
            .find(|e| e.name == "xl/worksheets/sheet1.xml")
            .expect("sheet1.xml entry");
        let sheet_bytes = reader.extract(sheet_entry).expect("extract sheet1.xml");
        let sheet_xml = String::from_utf8(sheet_bytes).expect("sheet1.xml is utf8");

        assert!(sheet_xml.contains("<worksheet"));
        // Header text from create_test_data(): "id", "value", "timestamp".
        assert!(sheet_xml.contains("id"), "sheet missing header text");
        assert!(sheet_xml.contains("value"), "sheet missing header text");
        assert!(sheet_xml.contains("timestamp"), "sheet missing header text");

        let _ = std::fs::remove_file(&output_path);
    }

    #[test]
    // 3.14 is test data from create_test_data(), not an approximation of PI.
    #[allow(clippy::approx_constant)]
    fn test_sqlite_export_roundtrip() {
        let config = ExportConfig::default();
        let mut manager = DataExportManager::new(config);
        let test_data = create_test_data();

        let file_name = format!("trustformers_sqlite_test_{}.sqlite3", Uuid::new_v4());
        let output_path = std::env::temp_dir().join(file_name).to_string_lossy().to_string();

        let job_id = manager
            .start_export(
                "Test SQLite Export".to_string(),
                test_data,
                ExportFormat::Sqlite,
                output_path.clone(),
                ExportOptions::default(),
            )
            .expect("sqlite export should succeed");

        // The job must have completed (not silently failed back to JSON).
        let status = manager.get_job_status(job_id).expect("job should exist");
        assert!(
            matches!(status.status, ExportStatus::Completed),
            "export status = {:?}",
            status.status
        );
        assert!(
            std::path::Path::new(&output_path).exists(),
            "sqlite file should exist"
        );

        // Re-open the real SQLite file and read the rows back.
        let conn = SqliteConnectionBlocking::open(&output_path).expect("open sqlite file");
        let tables = conn.tables().expect("list tables");
        assert!(!tables.is_empty(), "expected at least one table");

        // create_test_data() names the item "Test Data" -> sanitized "Test_Data".
        let rows = conn
            .query(
                "SELECT \"id\", \"value\", \"timestamp\" FROM \"Test_Data\"",
                &[],
            )
            .expect("query rows back");
        assert_eq!(rows.len(), 2, "two rows should persist");

        // Collect ids to verify both rows persisted regardless of row order.
        let mut ids: Vec<i64> = rows
            .iter()
            .map(|r| r.try_get::<i64>("id").expect("id column is INTEGER"))
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2]);

        // Find the row with id == 1 and verify the float + string columns.
        let first = rows
            .iter()
            .find(|r| r.try_get::<i64>("id").map(|v| v == 1).unwrap_or(false))
            .expect("row with id=1");
        let value: f64 = first.try_get("value").expect("value column is REAL");
        assert!((value - 3.14).abs() < 1e-9, "value = {value}");
        let timestamp: String = first.try_get("timestamp").expect("timestamp column is TEXT");
        assert_eq!(timestamp, "2023-01-01T12:00:00Z");

        let _ = std::fs::remove_file(&output_path);
    }
}
