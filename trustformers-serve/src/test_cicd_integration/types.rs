//! Basic types for CI/CD integration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::test_parallelization::TestParallelizationConfig;

// Type alias for consistency
pub type ExecutionResult = crate::test_timeout_optimization::TestExecutionResult;

/// CI/CD integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CicdIntegrationConfig {
    /// Enabled features
    pub enabled_features: Vec<CicdFeature>,

    /// Environment-specific configurations
    pub environment_configs: HashMap<String, EnvironmentConfig>,

    /// Pipeline integration settings
    pub pipeline_integration: PipelineIntegrationConfig,

    /// Reporting configuration
    pub reporting_config: ReportingConfig,

    /// Metrics export configuration
    pub metrics_export: MetricsExportConfig,

    /// Optimization settings
    pub optimization_config: OptimizationConfig,

    /// Security settings
    pub security_config: SecurityConfig,

    /// Default configuration
    pub default_config: TestParallelizationConfig,
}

/// CI/CD features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CicdFeature {
    /// Environment detection
    EnvironmentDetection,

    /// Auto configuration
    AutoConfiguration,

    /// Pipeline integration
    PipelineIntegration,

    /// Performance reporting
    PerformanceReporting,

    /// Metrics export
    MetricsExport,

    /// Environment optimization
    EnvironmentOptimization,

    /// Security compliance
    SecurityCompliance,

    /// Custom feature
    Custom(String),
}

/// Environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    /// Environment name
    pub name: String,

    /// Environment type
    pub environment_type: EnvironmentType,

    /// Test parallelization configuration
    pub test_config: TestParallelizationConfig,

    /// Resource limits
    pub resource_limits: EnvironmentResourceLimits,

    /// Optimization settings
    pub optimization: EnvironmentOptimizationSettings,

    /// Security settings
    pub security: EnvironmentSecuritySettings,

    /// Monitoring configuration
    pub monitoring: EnvironmentMonitoringConfig,

    /// Environment variables
    pub environment_variables: HashMap<String, String>,

    /// Configuration overrides
    pub overrides: HashMap<String, serde_json::Value>,
}

/// Environment types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnvironmentType {
    /// Local development
    Development,

    /// Testing environment
    Testing,

    /// Staging environment
    Staging,

    /// Production environment
    Production,

    /// CI/CD pipeline
    Pipeline(PipelineType),

    /// Custom environment
    Custom(String),
}

/// Pipeline types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineType {
    /// GitHub Actions
    GitHubActions,

    /// GitLab CI
    GitLabCi,

    /// Jenkins
    Jenkins,

    /// Azure DevOps
    AzureDevOps,

    /// CircleCI
    CircleCi,

    /// Travis CI
    TravisCi,

    /// Buildkite
    Buildkite,

    /// TeamCity
    TeamCity,

    /// Custom pipeline
    Custom(String),
}

// Forward declarations for complex types (implemented in other modules)
pub use super::environment::*;
pub use super::optimization::*;
pub use super::pipeline::*;
pub use super::reporting::*;
pub use super::security::*;
