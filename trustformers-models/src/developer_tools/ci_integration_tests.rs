#[cfg(test)]
mod tests {
    use crate::developer_tools::ci_integration::*;

    fn sample_ci_config() -> CIConfig {
        CIConfig {
            project_name: "test-project".to_string(),
            provider: CIProvider::GitHubActions,
            test_matrix: TestMatrix {
                rust_versions: vec!["stable".to_string(), "nightly".to_string()],
                operating_systems: vec!["ubuntu-latest".to_string()],
                hardware_configs: vec!["cpu".to_string()],
                feature_combinations: vec![vec!["default".to_string()]],
            },
            performance_thresholds: PerformanceThresholds {
                max_latency_ms: 100.0,
                min_throughput: 50.0,
                max_memory_mb: 1024.0,
                max_regression_percent: 5.0,
            },
            notifications: NotificationConfig {
                slack_enabled: false,
                email_enabled: false,
                discord_enabled: false,
                webhooks: Vec::new(),
            },
        }
    }

    // --- CIConfig tests ---

    #[test]
    fn test_ci_config_creation() {
        let config = sample_ci_config();
        assert_eq!(config.project_name, "test-project");
    }

    #[test]
    fn test_ci_config_provider_github() {
        let config = sample_ci_config();
        assert!(matches!(config.provider, CIProvider::GitHubActions));
    }

    #[test]
    fn test_ci_config_test_matrix() {
        let config = sample_ci_config();
        assert_eq!(config.test_matrix.rust_versions.len(), 2);
        assert_eq!(config.test_matrix.operating_systems.len(), 1);
    }

    #[test]
    fn test_ci_config_performance_thresholds() {
        let config = sample_ci_config();
        assert!((config.performance_thresholds.max_latency_ms - 100.0).abs() < f64::EPSILON);
        assert!((config.performance_thresholds.min_throughput - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ci_config_clone() {
        let config = sample_ci_config();
        let cloned = config.clone();
        assert_eq!(cloned.project_name, "test-project");
    }

    // --- CIIntegration tests ---

    #[test]
    fn test_ci_integration_creation() {
        let config = sample_ci_config();
        let _integration = CIIntegration::new(config);
    }

    #[test]
    fn test_ci_integration_generate_github_actions() {
        let config = sample_ci_config();
        let integration = CIIntegration::new(config);
        let temp = std::env::temp_dir().join("ci_test_github");
        let _ = std::fs::create_dir_all(&temp);
        let result = integration.generate_ci_config(&temp);
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_ci_integration_generate_gitlab_ci() {
        let mut config = sample_ci_config();
        config.provider = CIProvider::GitLabCI;
        let integration = CIIntegration::new(config);
        let temp = std::env::temp_dir().join("ci_test_gitlab");
        let _ = std::fs::create_dir_all(&temp);
        let result = integration.generate_ci_config(&temp);
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_ci_integration_generate_jenkins() {
        let mut config = sample_ci_config();
        config.provider = CIProvider::Jenkins;
        let integration = CIIntegration::new(config);
        let temp = std::env::temp_dir().join("ci_test_jenkins");
        let _ = std::fs::create_dir_all(&temp);
        let result = integration.generate_ci_config(&temp);
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_ci_integration_generate_travis() {
        let mut config = sample_ci_config();
        config.provider = CIProvider::Travis;
        let integration = CIIntegration::new(config);
        let temp = std::env::temp_dir().join("ci_test_travis");
        let _ = std::fs::create_dir_all(&temp);
        let result = integration.generate_ci_config(&temp);
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_ci_integration_generate_circleci() {
        let mut config = sample_ci_config();
        config.provider = CIProvider::CircleCI;
        let integration = CIIntegration::new(config);
        let temp = std::env::temp_dir().join("ci_test_circleci");
        let _ = std::fs::create_dir_all(&temp);
        let result = integration.generate_ci_config(&temp);
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(&temp);
    }

    // --- TestMatrix tests ---

    #[test]
    fn test_test_matrix_creation() {
        let matrix = TestMatrix {
            rust_versions: vec!["stable".to_string()],
            operating_systems: vec!["linux".to_string(), "macos".to_string()],
            hardware_configs: vec!["cpu".to_string(), "gpu".to_string()],
            feature_combinations: vec![vec!["default".to_string()], vec!["all".to_string()]],
        };
        assert_eq!(matrix.rust_versions.len(), 1);
        assert_eq!(matrix.operating_systems.len(), 2);
        assert_eq!(matrix.hardware_configs.len(), 2);
        assert_eq!(matrix.feature_combinations.len(), 2);
    }

    #[test]
    fn test_test_matrix_clone() {
        let matrix = TestMatrix {
            rust_versions: vec!["stable".to_string()],
            operating_systems: vec!["linux".to_string()],
            hardware_configs: Vec::new(),
            feature_combinations: Vec::new(),
        };
        let cloned = matrix.clone();
        assert_eq!(cloned.rust_versions, matrix.rust_versions);
    }

    // --- NotificationConfig tests ---

    #[test]
    fn test_notification_config_all_disabled() {
        let config = NotificationConfig {
            slack_enabled: false,
            email_enabled: false,
            discord_enabled: false,
            webhooks: Vec::new(),
        };
        assert!(!config.slack_enabled);
        assert!(!config.email_enabled);
        assert!(!config.discord_enabled);
        assert!(config.webhooks.is_empty());
    }

    #[test]
    fn test_notification_config_with_webhooks() {
        let config = NotificationConfig {
            slack_enabled: true,
            email_enabled: false,
            discord_enabled: true,
            webhooks: vec![
                "https://hooks.slack.com/test".to_string(),
                "https://discord.com/api/webhooks/test".to_string(),
            ],
        };
        assert!(config.slack_enabled);
        assert_eq!(config.webhooks.len(), 2);
    }

    // --- PerformanceThresholds tests ---

    #[test]
    fn test_performance_thresholds_creation() {
        let thresholds = PerformanceThresholds {
            max_latency_ms: 200.0,
            min_throughput: 100.0,
            max_memory_mb: 2048.0,
            max_regression_percent: 10.0,
        };
        assert!((thresholds.max_latency_ms - 200.0).abs() < f64::EPSILON);
        assert!((thresholds.max_regression_percent - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_performance_thresholds_clone() {
        let thresholds = PerformanceThresholds {
            max_latency_ms: 50.0,
            min_throughput: 200.0,
            max_memory_mb: 512.0,
            max_regression_percent: 3.0,
        };
        let cloned = thresholds.clone();
        assert!((cloned.max_latency_ms - 50.0).abs() < f64::EPSILON);
    }

    // --- CIProvider tests ---

    #[test]
    fn test_ci_provider_variants() {
        let providers = vec![
            CIProvider::GitHubActions,
            CIProvider::GitLabCI,
            CIProvider::Jenkins,
            CIProvider::Travis,
            CIProvider::CircleCI,
        ];
        assert_eq!(providers.len(), 5);
    }
}
