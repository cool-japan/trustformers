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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cicd_feature_debug_environment_detection() {
        let f = CicdFeature::EnvironmentDetection;
        let debug_str = format!("{:?}", f);
        assert!(debug_str.contains("EnvironmentDetection"));
    }

    #[test]
    fn test_cicd_feature_debug_auto_configuration() {
        let f = CicdFeature::AutoConfiguration;
        let debug_str = format!("{:?}", f);
        assert!(debug_str.contains("AutoConfiguration"));
    }

    #[test]
    fn test_cicd_feature_debug_pipeline_integration() {
        let f = CicdFeature::PipelineIntegration;
        let debug_str = format!("{:?}", f);
        assert!(debug_str.contains("PipelineIntegration"));
    }

    #[test]
    fn test_cicd_feature_debug_performance_reporting() {
        let f = CicdFeature::PerformanceReporting;
        let debug_str = format!("{:?}", f);
        assert!(debug_str.contains("PerformanceReporting"));
    }

    #[test]
    fn test_cicd_feature_debug_metrics_export() {
        let f = CicdFeature::MetricsExport;
        let debug_str = format!("{:?}", f);
        assert!(debug_str.contains("MetricsExport"));
    }

    #[test]
    fn test_cicd_feature_debug_environment_optimization() {
        let f = CicdFeature::EnvironmentOptimization;
        let debug_str = format!("{:?}", f);
        assert!(debug_str.contains("EnvironmentOptimization"));
    }

    #[test]
    fn test_cicd_feature_debug_security_compliance() {
        let f = CicdFeature::SecurityCompliance;
        let debug_str = format!("{:?}", f);
        assert!(debug_str.contains("SecurityCompliance"));
    }

    #[test]
    fn test_cicd_feature_custom_name() {
        let f = CicdFeature::Custom("test_coverage_enforcement".to_string());
        if let CicdFeature::Custom(name) = f {
            assert_eq!(name, "test_coverage_enforcement");
        } else {
            panic!("expected Custom variant");
        }
    }

    #[test]
    fn test_environment_type_development() {
        let et = EnvironmentType::Development;
        let debug_str = format!("{:?}", et);
        assert!(debug_str.contains("Development"));
    }

    #[test]
    fn test_environment_type_testing() {
        let et = EnvironmentType::Testing;
        let debug_str = format!("{:?}", et);
        assert!(debug_str.contains("Testing"));
    }

    #[test]
    fn test_environment_type_staging() {
        let et = EnvironmentType::Staging;
        let debug_str = format!("{:?}", et);
        assert!(debug_str.contains("Staging"));
    }

    #[test]
    fn test_environment_type_production() {
        let et = EnvironmentType::Production;
        let debug_str = format!("{:?}", et);
        assert!(debug_str.contains("Production"));
    }

    #[test]
    fn test_environment_type_github_actions_pipeline() {
        let et = EnvironmentType::Pipeline(PipelineType::GitHubActions);
        if let EnvironmentType::Pipeline(pt) = et {
            let debug_str = format!("{:?}", pt);
            assert!(debug_str.contains("GitHubActions"));
        } else {
            panic!("expected Pipeline variant");
        }
    }

    #[test]
    fn test_environment_type_custom() {
        let et = EnvironmentType::Custom("k8s_cluster".to_string());
        if let EnvironmentType::Custom(name) = et {
            assert_eq!(name, "k8s_cluster");
        } else {
            panic!("expected Custom variant");
        }
    }

    #[test]
    fn test_pipeline_type_github_actions() {
        let pt = PipelineType::GitHubActions;
        let debug_str = format!("{:?}", pt);
        assert!(debug_str.contains("GitHubActions"));
    }

    #[test]
    fn test_pipeline_type_gitlab_ci() {
        let pt = PipelineType::GitLabCi;
        let debug_str = format!("{:?}", pt);
        assert!(debug_str.contains("GitLabCi"));
    }

    #[test]
    fn test_pipeline_type_jenkins() {
        let pt = PipelineType::Jenkins;
        let debug_str = format!("{:?}", pt);
        assert!(debug_str.contains("Jenkins"));
    }

    #[test]
    fn test_pipeline_type_azure_devops() {
        let pt = PipelineType::AzureDevOps;
        let debug_str = format!("{:?}", pt);
        assert!(debug_str.contains("AzureDevOps"));
    }

    #[test]
    fn test_pipeline_type_circleci() {
        let pt = PipelineType::CircleCi;
        let debug_str = format!("{:?}", pt);
        assert!(debug_str.contains("CircleCi"));
    }

    #[test]
    fn test_pipeline_type_travis_ci() {
        let pt = PipelineType::TravisCi;
        let debug_str = format!("{:?}", pt);
        assert!(debug_str.contains("TravisCi"));
    }

    #[test]
    fn test_pipeline_type_buildkite() {
        let pt = PipelineType::Buildkite;
        let debug_str = format!("{:?}", pt);
        assert!(debug_str.contains("Buildkite"));
    }

    #[test]
    fn test_pipeline_type_teamcity() {
        let pt = PipelineType::TeamCity;
        let debug_str = format!("{:?}", pt);
        assert!(debug_str.contains("TeamCity"));
    }

    #[test]
    fn test_pipeline_type_custom() {
        let pt = PipelineType::Custom("internal_ci".to_string());
        if let PipelineType::Custom(name) = pt {
            assert_eq!(name, "internal_ci");
        } else {
            panic!("expected Custom variant");
        }
    }

    #[test]
    fn test_feature_list_all_variants() {
        let features = vec![
            CicdFeature::EnvironmentDetection,
            CicdFeature::AutoConfiguration,
            CicdFeature::PipelineIntegration,
            CicdFeature::PerformanceReporting,
            CicdFeature::MetricsExport,
            CicdFeature::EnvironmentOptimization,
            CicdFeature::SecurityCompliance,
        ];
        assert_eq!(features.len(), 7);
        for f in &features {
            let debug_str = format!("{:?}", f);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_all_pipeline_types_unique_debug() {
        let types = vec![
            format!("{:?}", PipelineType::GitHubActions),
            format!("{:?}", PipelineType::GitLabCi),
            format!("{:?}", PipelineType::Jenkins),
            format!("{:?}", PipelineType::AzureDevOps),
            format!("{:?}", PipelineType::CircleCi),
        ];
        let unique: std::collections::HashSet<_> = types.iter().collect();
        assert_eq!(types.len(), unique.len());
    }

    #[test]
    fn test_cicd_feature_clone() {
        let f = CicdFeature::EnvironmentDetection;
        let f_cloned = f.clone();
        let debug_orig = format!("{:?}", CicdFeature::EnvironmentDetection);
        let debug_clone = format!("{:?}", f_cloned);
        assert_eq!(debug_orig, debug_clone);
    }

    #[test]
    fn test_pipeline_type_clone() {
        let pt = PipelineType::GitHubActions;
        let pt_cloned = pt.clone();
        let debug_orig = format!("{:?}", PipelineType::GitHubActions);
        let debug_clone = format!("{:?}", pt_cloned);
        assert_eq!(debug_orig, debug_clone);
    }
}
