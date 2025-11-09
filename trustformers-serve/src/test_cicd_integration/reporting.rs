//! Reporting configuration and types

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Report formats
    pub formats: Vec<ReportFormat>,

    /// Report destinations
    pub destinations: Vec<ReportDestination>,

    /// Report schedule
    pub schedule: ReportSchedule,

    /// Report templates
    pub templates: Vec<ReportTemplate>,

    /// Report filters
    pub filters: Vec<ReportFilter>,
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Html,
    Pdf,
    Json,
    Xml,
    Csv,
    Xlsx,
    Custom(String),
}

/// Report destination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDestination {
    /// Destination name
    pub name: String,

    /// Destination type
    pub destination_type: ReportDestinationType,

    /// Destination configuration
    pub config: DestinationConfig,

    /// Delivery settings
    pub delivery: DeliverySettings,
}

/// Report destination types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportDestinationType {
    /// Email delivery
    Email { recipients: Vec<String> },

    /// File storage
    File { path: String },

    /// HTTP endpoint
    Http { url: String },

    /// AWS S3
    AwsS3 { bucket: String, key_prefix: String },

    /// Azure Blob Storage
    AzureBlob {
        container: String,
        blob_prefix: String,
    },

    /// Google Cloud Storage
    GoogleCloudStorage {
        bucket: String,
        object_prefix: String,
    },

    /// Slack
    Slack { webhook_url: String },

    /// Microsoft Teams
    MicrosoftTeams { webhook_url: String },

    /// Custom destination
    Custom(String),
}

/// Destination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DestinationConfig {
    /// Authentication settings
    pub auth: Option<HashMap<String, String>>,

    /// Connection settings
    pub connection: Option<HashMap<String, String>>,

    /// Custom settings
    pub custom: HashMap<String, serde_json::Value>,
}

/// Delivery settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliverySettings {
    /// Delivery method
    pub method: DeliveryMethod,

    /// Retry settings
    pub retry_attempts: usize,

    /// Delivery timeout
    pub timeout: Duration,

    /// Delivery encryption
    pub encryption: bool,
}

/// Delivery methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryMethod {
    /// Immediate delivery
    Immediate,

    /// Batch delivery
    Batch { batch_size: usize },

    /// Scheduled delivery
    Scheduled { schedule: String },

    /// Custom delivery
    Custom(String),
}

/// Report schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSchedule {
    /// Schedule enabled
    pub enabled: bool,

    /// Cron expression
    pub cron: String,

    /// Timezone
    pub timezone: String,

    /// Next execution time
    pub next_execution: Option<chrono::DateTime<chrono::Utc>>,
}

/// Report template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplate {
    /// Template name
    pub name: String,

    /// Template content
    pub content: String,

    /// Template variables
    pub variables: HashMap<String, String>,

    /// Template format
    pub format: ReportFormat,
}

/// Report filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportFilter {
    /// Filter name
    pub name: String,

    /// Filter expression
    pub expression: String,

    /// Include or exclude
    pub include: bool,
}

/// Metrics export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExportConfig {
    /// Export enabled
    pub enabled: bool,

    /// Export targets
    pub targets: Vec<super::environment::ExportTarget>,

    /// Export schedule
    pub schedule: ExportSchedule,

    /// Export transformations
    pub transformations: Vec<ExportTransformation>,
}

/// Export schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSchedule {
    /// Export interval
    pub interval: Duration,

    /// Export batch size
    pub batch_size: usize,

    /// Export timeout
    pub timeout: Duration,

    /// Export compression
    pub compression: bool,
}

/// Export transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportTransformation {
    /// Transformation name
    pub name: String,

    /// Transformation type
    pub transformation_type: TransformationType,

    /// Transformation rules
    pub rules: Vec<TransformationRule>,
}

/// Transformation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationType {
    /// Aggregation transformation
    Aggregation { function: String },

    /// Filtering transformation
    Filtering { condition: String },

    /// Mapping transformation
    Mapping { mapping: HashMap<String, String> },

    /// Normalization transformation
    Normalization { method: String },

    /// Custom transformation
    Custom(String),
}

/// Transformation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationRule {
    /// Rule pattern
    pub pattern: String,

    /// Rule replacement
    pub replacement: String,
}
