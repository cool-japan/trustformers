//! Test Configuration Manager
//!
//! This module handles configuration management for test timeout optimization
//! across different environments (local dev, CI/CD, production testing).

use crate::test_timeout_optimization::TestTimeoutConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    time::Duration,
};
use tracing::{debug, info, warn};

/// Configuration manager for test timeout optimization
pub struct TestConfigManager {
    /// Base configuration directory
    config_dir: PathBuf,

    /// Current environment
    environment: String,

    /// Loaded configuration
    config: TestTimeoutConfig,

    /// Configuration sources (in priority order)
    sources: Vec<ConfigSource>,
}

/// Configuration source
#[derive(Debug, Clone)]
pub enum ConfigSource {
    /// Default built-in configuration
    Default,

    /// Configuration from file
    File(PathBuf),

    /// Configuration from environment variables
    Environment,

    /// Runtime configuration overrides
    Runtime(HashMap<String, String>),
}

/// Environment-specific configuration presets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentPreset {
    /// Environment name
    pub name: String,

    /// Base timeout multiplier
    pub timeout_multiplier: f32,

    /// Optimization settings
    pub optimization_settings: OptimizationSettings,

    /// Monitoring settings
    pub monitoring_settings: MonitoringSettings,

    /// Environment-specific overrides
    pub overrides: HashMap<String, String>,
}

/// Optimization settings for environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSettings {
    /// Enable adaptive timeouts
    pub adaptive_timeouts: bool,

    /// Enable early termination
    pub early_termination: bool,

    /// Enable parallel execution
    pub parallel_execution: bool,

    /// Learning rate for adaptive algorithms
    pub learning_rate: f32,

    /// Aggressiveness of optimizations (0.0 - 1.0)
    pub aggressiveness: f32,
}

/// Monitoring settings for environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringSettings {
    /// Enable performance monitoring
    pub enabled: bool,

    /// Metrics collection frequency
    pub collection_frequency: Duration,

    /// Enable detailed logging
    pub detailed_logging: bool,

    /// Performance regression threshold
    pub regression_threshold: f32,

    /// Export metrics to external systems
    pub export_metrics: bool,
}

/// Configuration validation result
#[derive(Debug)]
pub struct ValidationResult {
    /// Whether the configuration is valid
    pub is_valid: bool,

    /// Validation warnings
    pub warnings: Vec<String>,

    /// Validation errors
    pub errors: Vec<String>,

    /// Suggested fixes
    pub suggested_fixes: Vec<String>,
}

impl TestConfigManager {
    /// Create a new configuration manager
    pub fn new<P: AsRef<Path>>(config_dir: P) -> Result<Self> {
        let config_dir = config_dir.as_ref().to_path_buf();
        let environment = Self::detect_environment();

        let mut manager = Self {
            config_dir,
            environment: environment.clone(),
            config: TestTimeoutConfig::default(),
            sources: vec![ConfigSource::Default],
        };

        manager.load_configuration()?;

        info!(
            environment = %environment,
            config_dir = %manager.config_dir.display(),
            "Test configuration manager initialized"
        );

        Ok(manager)
    }

    /// Detect the current environment
    fn detect_environment() -> String {
        // Check explicit environment variable
        if let Ok(env) = env::var("TEST_ENVIRONMENT") {
            return env;
        }

        // Check CI environment indicators
        if env::var("CI").is_ok() || env::var("GITHUB_ACTIONS").is_ok() {
            return "ci".to_string();
        }

        if env::var("JENKINS_URL").is_ok() {
            return "jenkins".to_string();
        }

        // Check for development indicators
        if cfg!(debug_assertions) {
            return "development".to_string();
        }

        // Default to production
        "production".to_string()
    }

    /// Load configuration from all sources
    pub fn load_configuration(&mut self) -> Result<()> {
        debug!("Loading test timeout configuration");

        // Start with default configuration
        self.config = self.create_default_config();
        self.sources = vec![ConfigSource::Default];

        // Load environment preset if available
        if let Ok(preset_config) = self.load_environment_preset(&self.environment) {
            self.apply_environment_preset(&preset_config)?;
            self.sources.push(ConfigSource::File(
                self.config_dir.join(format!("{}.toml", self.environment)),
            ));
        }

        // Load global configuration file
        let config_file = self.config_dir.join("test_timeout.toml");
        if config_file.exists() {
            let file_config = self.load_config_file(&config_file)?;
            self.merge_configuration(file_config)?;
            self.sources.push(ConfigSource::File(config_file));
        }

        // Apply environment variable overrides
        self.apply_env_var_overrides()?;
        self.sources.push(ConfigSource::Environment);

        // Validate final configuration
        let validation = self.validate_configuration();
        if !validation.is_valid {
            return Err(anyhow::anyhow!(
                "Configuration validation failed: {:?}",
                validation.errors
            ));
        }

        for warning in validation.warnings {
            warn!("Configuration warning: {}", warning);
        }

        info!(
            environment = %self.environment,
            sources = ?self.sources.len(),
            "Test timeout configuration loaded successfully"
        );

        Ok(())
    }

    /// Create default configuration
    fn create_default_config(&self) -> TestTimeoutConfig {
        TestTimeoutConfig::default()
    }

    /// Load environment-specific preset
    fn load_environment_preset(&self, environment: &str) -> Result<EnvironmentPreset> {
        let preset_file = self.config_dir.join(format!("{}.toml", environment));

        if !preset_file.exists() {
            // Return built-in preset for known environments
            return Ok(self.create_builtin_preset(environment));
        }

        let content = fs::read_to_string(&preset_file)
            .with_context(|| format!("Failed to read preset file: {}", preset_file.display()))?;

        let preset: EnvironmentPreset = toml::from_str(&content)
            .with_context(|| format!("Failed to parse preset file: {}", preset_file.display()))?;

        Ok(preset)
    }

    /// Create built-in environment preset
    fn create_builtin_preset(&self, environment: &str) -> EnvironmentPreset {
        match environment {
            "ci" | "github" | "jenkins" => EnvironmentPreset {
                name: environment.to_string(),
                timeout_multiplier: 2.0, // Longer timeouts in CI
                optimization_settings: OptimizationSettings {
                    adaptive_timeouts: true,
                    early_termination: true,
                    parallel_execution: true,
                    learning_rate: 0.05, // Conservative learning in CI
                    aggressiveness: 0.3, // Low aggressiveness for stability
                },
                monitoring_settings: MonitoringSettings {
                    enabled: true,
                    collection_frequency: Duration::from_millis(200),
                    detailed_logging: true,
                    regression_threshold: 0.15, // 15% regression threshold
                    export_metrics: true,
                },
                overrides: HashMap::new(),
            },

            "development" | "dev" => EnvironmentPreset {
                name: environment.to_string(),
                timeout_multiplier: 0.7, // Shorter timeouts for faster feedback
                optimization_settings: OptimizationSettings {
                    adaptive_timeouts: true,
                    early_termination: true,
                    parallel_execution: true,
                    learning_rate: 0.2,  // Faster learning in dev
                    aggressiveness: 0.7, // High aggressiveness for speed
                },
                monitoring_settings: MonitoringSettings {
                    enabled: true,
                    collection_frequency: Duration::from_millis(100),
                    detailed_logging: false,
                    regression_threshold: 0.25, // More lenient in dev
                    export_metrics: false,
                },
                overrides: HashMap::new(),
            },

            "performance" | "perf" => EnvironmentPreset {
                name: environment.to_string(),
                timeout_multiplier: 5.0, // Very long timeouts for performance tests
                optimization_settings: OptimizationSettings {
                    adaptive_timeouts: false,  // Disable for consistent measurement
                    early_termination: false,  // Disable for complete execution
                    parallel_execution: false, // Sequential for accurate measurement
                    learning_rate: 0.0,
                    aggressiveness: 0.0,
                },
                monitoring_settings: MonitoringSettings {
                    enabled: true,
                    collection_frequency: Duration::from_millis(50),
                    detailed_logging: true,
                    regression_threshold: 0.05, // Very strict for performance
                    export_metrics: true,
                },
                overrides: HashMap::new(),
            },

            _ => EnvironmentPreset {
                name: environment.to_string(),
                timeout_multiplier: 1.0,
                optimization_settings: OptimizationSettings {
                    adaptive_timeouts: true,
                    early_termination: true,
                    parallel_execution: true,
                    learning_rate: 0.1,
                    aggressiveness: 0.5,
                },
                monitoring_settings: MonitoringSettings {
                    enabled: true,
                    collection_frequency: Duration::from_millis(100),
                    detailed_logging: false,
                    regression_threshold: 0.2,
                    export_metrics: false,
                },
                overrides: HashMap::new(),
            },
        }
    }

    /// Apply environment preset to configuration
    fn apply_environment_preset(&mut self, preset: &EnvironmentPreset) -> Result<()> {
        // Apply timeout multiplier
        self.config.base_timeouts.unit_tests = Duration::from_secs_f32(
            self.config.base_timeouts.unit_tests.as_secs_f32() * preset.timeout_multiplier,
        );
        self.config.base_timeouts.integration_tests = Duration::from_secs_f32(
            self.config.base_timeouts.integration_tests.as_secs_f32() * preset.timeout_multiplier,
        );
        self.config.base_timeouts.e2e_tests = Duration::from_secs_f32(
            self.config.base_timeouts.e2e_tests.as_secs_f32() * preset.timeout_multiplier,
        );
        self.config.base_timeouts.stress_tests = Duration::from_secs_f32(
            self.config.base_timeouts.stress_tests.as_secs_f32() * preset.timeout_multiplier,
        );
        self.config.base_timeouts.property_tests = Duration::from_secs_f32(
            self.config.base_timeouts.property_tests.as_secs_f32() * preset.timeout_multiplier,
        );
        self.config.base_timeouts.chaos_tests = Duration::from_secs_f32(
            self.config.base_timeouts.chaos_tests.as_secs_f32() * preset.timeout_multiplier,
        );
        self.config.base_timeouts.long_running_tests = Duration::from_secs_f32(
            self.config.base_timeouts.long_running_tests.as_secs_f32() * preset.timeout_multiplier,
        );

        // Apply optimization settings
        self.config.adaptive.enabled = preset.optimization_settings.adaptive_timeouts;
        self.config.adaptive.learning_rate = preset.optimization_settings.learning_rate;
        self.config.early_termination.enabled = preset.optimization_settings.early_termination;

        // Apply monitoring settings
        self.config.monitoring.enabled = preset.monitoring_settings.enabled;
        self.config.monitoring.collection_interval =
            preset.monitoring_settings.collection_frequency;
        self.config.monitoring.regression_threshold =
            preset.monitoring_settings.regression_threshold;

        // Update logging based on detailed_logging setting
        self.config.monitoring.timeout_logging.log_warnings =
            preset.monitoring_settings.detailed_logging;
        self.config.monitoring.timeout_logging.log_adjustments =
            preset.monitoring_settings.detailed_logging;

        debug!(
            preset_name = %preset.name,
            timeout_multiplier = preset.timeout_multiplier,
            "Applied environment preset"
        );

        Ok(())
    }

    /// Load configuration from file
    fn load_config_file(&self, path: &Path) -> Result<TestTimeoutConfig> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let config: TestTimeoutConfig = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;

        debug!(config_file = %path.display(), "Loaded configuration from file");
        Ok(config)
    }

    /// Merge configuration with current config
    fn merge_configuration(&mut self, other: TestTimeoutConfig) -> Result<()> {
        // For simplicity, we'll replace the entire config
        // In a real implementation, you'd want more sophisticated merging
        if other.enabled {
            self.config = other;
        }

        Ok(())
    }

    /// Apply environment variable overrides
    fn apply_env_var_overrides(&mut self) -> Result<()> {
        // Unit test timeout
        if let Ok(timeout_str) = env::var("TEST_TIMEOUT_UNIT") {
            if let Ok(timeout_secs) = timeout_str.parse::<u64>() {
                self.config.base_timeouts.unit_tests = Duration::from_secs(timeout_secs);
                debug!(
                    timeout_secs = timeout_secs,
                    "Override unit test timeout from env"
                );
            }
        }

        // Integration test timeout
        if let Ok(timeout_str) = env::var("TEST_TIMEOUT_INTEGRATION") {
            if let Ok(timeout_secs) = timeout_str.parse::<u64>() {
                self.config.base_timeouts.integration_tests = Duration::from_secs(timeout_secs);
                debug!(
                    timeout_secs = timeout_secs,
                    "Override integration test timeout from env"
                );
            }
        }

        // Enable/disable adaptive timeouts
        if let Ok(enabled_str) = env::var("TEST_ADAPTIVE_TIMEOUT") {
            if let Ok(enabled) = enabled_str.parse::<bool>() {
                self.config.adaptive.enabled = enabled;
                debug!(enabled = enabled, "Override adaptive timeout from env");
            }
        }

        // Enable/disable early termination
        if let Ok(enabled_str) = env::var("TEST_EARLY_TERMINATION") {
            if let Ok(enabled) = enabled_str.parse::<bool>() {
                self.config.early_termination.enabled = enabled;
                debug!(enabled = enabled, "Override early termination from env");
            }
        }

        // Learning rate
        if let Ok(rate_str) = env::var("TEST_LEARNING_RATE") {
            if let Ok(rate) = rate_str.parse::<f32>() {
                self.config.adaptive.learning_rate = rate.clamp(0.0, 1.0);
                debug!(learning_rate = rate, "Override learning rate from env");
            }
        }

        // Global timeout multiplier
        if let Ok(multiplier_str) = env::var("TEST_TIMEOUT_MULTIPLIER") {
            if let Ok(multiplier) = multiplier_str.parse::<f32>() {
                self.apply_global_timeout_multiplier(multiplier);
                debug!(
                    multiplier = multiplier,
                    "Applied global timeout multiplier from env"
                );
            }
        }

        Ok(())
    }

    /// Apply global timeout multiplier to all base timeouts
    fn apply_global_timeout_multiplier(&mut self, multiplier: f32) {
        self.config.base_timeouts.unit_tests = Duration::from_secs_f32(
            self.config.base_timeouts.unit_tests.as_secs_f32() * multiplier,
        );
        self.config.base_timeouts.integration_tests = Duration::from_secs_f32(
            self.config.base_timeouts.integration_tests.as_secs_f32() * multiplier,
        );
        self.config.base_timeouts.e2e_tests =
            Duration::from_secs_f32(self.config.base_timeouts.e2e_tests.as_secs_f32() * multiplier);
        self.config.base_timeouts.stress_tests = Duration::from_secs_f32(
            self.config.base_timeouts.stress_tests.as_secs_f32() * multiplier,
        );
        self.config.base_timeouts.property_tests = Duration::from_secs_f32(
            self.config.base_timeouts.property_tests.as_secs_f32() * multiplier,
        );
        self.config.base_timeouts.chaos_tests = Duration::from_secs_f32(
            self.config.base_timeouts.chaos_tests.as_secs_f32() * multiplier,
        );
        self.config.base_timeouts.long_running_tests = Duration::from_secs_f32(
            self.config.base_timeouts.long_running_tests.as_secs_f32() * multiplier,
        );
    }

    /// Validate the current configuration
    pub fn validate_configuration(&self) -> ValidationResult {
        let mut result = ValidationResult {
            is_valid: true,
            warnings: Vec::new(),
            errors: Vec::new(),
            suggested_fixes: Vec::new(),
        };

        // Validate timeout values
        if self.config.base_timeouts.unit_tests > Duration::from_secs(30) {
            result.warnings.push("Unit test timeout is quite long (>30s)".to_string());
            result
                .suggested_fixes
                .push("Consider reducing unit test timeout for faster feedback".to_string());
        }

        if self.config.base_timeouts.unit_tests < Duration::from_secs(1) {
            result.errors.push("Unit test timeout is too short (<1s)".to_string());
            result
                .suggested_fixes
                .push("Increase unit test timeout to at least 1 second".to_string());
            result.is_valid = false;
        }

        // Validate adaptive timeout settings
        if self.config.adaptive.enabled {
            if self.config.adaptive.learning_rate <= 0.0 || self.config.adaptive.learning_rate > 1.0
            {
                result
                    .errors
                    .push("Adaptive learning rate must be between 0.0 and 1.0".to_string());
                result
                    .suggested_fixes
                    .push("Set learning rate to a value between 0.01 and 0.5".to_string());
                result.is_valid = false;
            }

            if self.config.adaptive.min_multiplier >= self.config.adaptive.max_multiplier {
                result
                    .errors
                    .push("Adaptive min_multiplier must be less than max_multiplier".to_string());
                result
                    .suggested_fixes
                    .push("Ensure min_multiplier < max_multiplier (e.g., 0.5 < 3.0)".to_string());
                result.is_valid = false;
            }
        }

        // Validate monitoring settings
        if self.config.monitoring.enabled {
            if self.config.monitoring.collection_interval < Duration::from_millis(10) {
                result
                    .warnings
                    .push("Very frequent monitoring collection may impact performance".to_string());
                result
                    .suggested_fixes
                    .push("Consider increasing collection interval to at least 50ms".to_string());
            }

            if self.config.monitoring.regression_threshold <= 0.0 {
                result.errors.push("Regression threshold must be positive".to_string());
                result.suggested_fixes.push(
                    "Set regression threshold to a positive value (e.g., 0.1 for 10%)".to_string(),
                );
                result.is_valid = false;
            }
        }

        // Validate early termination settings
        if self.config.early_termination.enabled
            && self.config.early_termination.min_progress_rate < 0.0
        {
            result.errors.push("Minimum progress rate cannot be negative".to_string());
            result
                .suggested_fixes
                .push("Set min_progress_rate to a positive value".to_string());
            result.is_valid = false;
        }

        result
    }

    /// Get the current configuration
    pub fn get_config(&self) -> &TestTimeoutConfig {
        &self.config
    }

    /// Get the current environment
    pub fn get_environment(&self) -> &str {
        &self.environment
    }

    /// Save current configuration to file
    pub fn save_config_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let toml_content = toml::to_string_pretty(&self.config)
            .context("Failed to serialize configuration to TOML")?;

        fs::write(path.as_ref(), toml_content)
            .with_context(|| format!("Failed to write config to {}", path.as_ref().display()))?;

        info!(path = %path.as_ref().display(), "Configuration saved to file");
        Ok(())
    }

    /// Create configuration template for a specific environment
    pub fn create_environment_template(&self, environment: &str) -> EnvironmentPreset {
        self.create_builtin_preset(environment)
    }

    /// Save environment template to file
    pub fn save_environment_template<P: AsRef<Path>>(
        &self,
        environment: &str,
        path: P,
    ) -> Result<()> {
        let template = self.create_environment_template(environment);
        let toml_content = toml::to_string_pretty(&template)
            .context("Failed to serialize environment template to TOML")?;

        fs::write(path.as_ref(), toml_content)
            .with_context(|| format!("Failed to write template to {}", path.as_ref().display()))?;

        info!(
            environment = environment,
            path = %path.as_ref().display(),
            "Environment template saved to file"
        );
        Ok(())
    }

    /// Get configuration summary for diagnostics
    pub fn get_config_summary(&self) -> ConfigSummary {
        ConfigSummary {
            environment: self.environment.clone(),
            sources: self.sources.clone(),
            enabled: self.config.enabled,
            adaptive_enabled: self.config.adaptive.enabled,
            early_termination_enabled: self.config.early_termination.enabled,
            monitoring_enabled: self.config.monitoring.enabled,
            unit_test_timeout: self.config.base_timeouts.unit_tests,
            integration_test_timeout: self.config.base_timeouts.integration_tests,
            learning_rate: self.config.adaptive.learning_rate,
            regression_threshold: self.config.monitoring.regression_threshold,
        }
    }
}

/// Configuration summary for diagnostics
#[derive(Debug, Clone)]
pub struct ConfigSummary {
    pub environment: String,
    pub sources: Vec<ConfigSource>,
    pub enabled: bool,
    pub adaptive_enabled: bool,
    pub early_termination_enabled: bool,
    pub monitoring_enabled: bool,
    pub unit_test_timeout: Duration,
    pub integration_test_timeout: Duration,
    pub learning_rate: f32,
    pub regression_threshold: f32,
}

impl ConfigSummary {
    /// Print configuration summary
    pub fn print_summary(&self) {
        println!("\n=== Test Timeout Configuration Summary ===");
        println!("Environment: {}", self.environment);
        println!("Configuration sources: {} sources", self.sources.len());
        println!("Framework enabled: {}", self.enabled);
        println!("Adaptive timeouts: {}", self.adaptive_enabled);
        println!("Early termination: {}", self.early_termination_enabled);
        println!("Performance monitoring: {}", self.monitoring_enabled);
        println!("Unit test timeout: {:?}", self.unit_test_timeout);
        println!(
            "Integration test timeout: {:?}",
            self.integration_test_timeout
        );
        println!("Learning rate: {:.3}", self.learning_rate);
        println!(
            "Regression threshold: {:.1}%",
            self.regression_threshold * 100.0
        );
        println!("==========================================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    // ── LCG ──────────────────────────────────────────────────────────────────
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005_u64)
                .wrapping_add(1_442_695_040_888_963_407_u64);
            self.state
        }
        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1_u64 << 53) as f32
        }
    }

    fn make_config_dir() -> std::path::PathBuf {
        let mut tmp = env::temp_dir();
        tmp.push(format!("tcm_test_{}", uuid_u64()));
        std::fs::create_dir_all(&tmp).expect("create temp dir");
        tmp
    }

    fn uuid_u64() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(12345)
    }

    // ── TestConfigManager::new ────────────────────────────────────────────────

    #[test]
    fn test_config_manager_new_with_empty_dir_succeeds() {
        let dir = make_config_dir();
        let manager = TestConfigManager::new(&dir);
        assert!(manager.is_ok(), "TestConfigManager::new failed");
    }

    #[test]
    fn test_config_manager_get_config_returns_enabled() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let cfg = manager.get_config();
            // Default config has enabled = true
            assert!(cfg.enabled);
        }
    }

    #[test]
    fn test_config_manager_get_environment_not_empty() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let env_str = manager.get_environment();
            assert!(!env_str.is_empty());
        }
    }

    // ── validate_configuration ────────────────────────────────────────────────

    #[test]
    fn test_validate_configuration_default_is_valid() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let result = manager.validate_configuration();
            assert!(result.is_valid, "errors={}", result.errors.join(", "));
        }
    }

    #[test]
    fn test_validate_configuration_result_has_no_errors_by_default() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let result = manager.validate_configuration();
            assert!(
                result.errors.is_empty(),
                "unexpected errors: {:?}",
                result.errors
            );
        }
    }

    // ── create_environment_template ───────────────────────────────────────────

    #[test]
    fn test_create_environment_template_ci() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let preset = manager.create_environment_template("ci");
            assert_eq!(preset.name, "ci");
            assert!(
                preset.timeout_multiplier > 1.0,
                "CI should multiply timeouts"
            );
        }
    }

    #[test]
    fn test_create_environment_template_development() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let preset = manager.create_environment_template("development");
            assert_eq!(preset.name, "development");
            // Dev timeouts should be shorter (multiplier < 1.0)
            assert!(
                preset.timeout_multiplier < 1.0,
                "dev multiplier={} should be < 1.0",
                preset.timeout_multiplier
            );
        }
    }

    #[test]
    fn test_create_environment_template_performance() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let preset = manager.create_environment_template("performance");
            assert!(
                preset.timeout_multiplier > 3.0,
                "perf should have long timeouts"
            );
        }
    }

    #[test]
    fn test_create_environment_template_unknown_defaults() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let preset = manager.create_environment_template("totally_unknown_env");
            assert!((preset.timeout_multiplier - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_create_environment_template_github_same_as_ci() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let ci_preset = manager.create_environment_template("ci");
            let gh_preset = manager.create_environment_template("github");
            // Both CI variants share the same multiplier
            assert!((ci_preset.timeout_multiplier - gh_preset.timeout_multiplier).abs() < 0.01);
        }
    }

    // ── OptimizationSettings field checks ────────────────────────────────────

    #[test]
    fn test_optimization_settings_ci_preset_adaptive_enabled() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let preset = manager.create_environment_template("ci");
            assert!(preset.optimization_settings.adaptive_timeouts);
        }
    }

    #[test]
    fn test_optimization_settings_perf_preset_non_adaptive() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let preset = manager.create_environment_template("performance");
            assert!(!preset.optimization_settings.adaptive_timeouts);
            assert!(!preset.optimization_settings.parallel_execution);
        }
    }

    // ── get_config_summary ────────────────────────────────────────────────────

    #[test]
    fn test_config_summary_has_at_least_one_source() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let summary = manager.get_config_summary();
            assert!(!summary.sources.is_empty());
        }
    }

    #[test]
    fn test_config_summary_environment_matches_manager() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let env_from_manager = manager.get_environment().to_string();
            let summary = manager.get_config_summary();
            assert_eq!(summary.environment, env_from_manager);
        }
    }

    #[test]
    fn test_config_summary_print_summary_no_panic() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            // print_summary should not panic
            manager.get_config_summary().print_summary();
        }
    }

    // ── save_config_to_file ───────────────────────────────────────────────────

    #[test]
    fn test_save_config_to_file_creates_file() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let out = dir.join("saved_config.toml");
            let result = manager.save_config_to_file(&out);
            assert!(result.is_ok(), "save failed");
            assert!(out.exists(), "file should exist after save");
        }
    }

    #[test]
    fn test_save_environment_template_creates_file() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let out = dir.join("ci_template.toml");
            let result = manager.save_environment_template("ci", &out);
            assert!(result.is_ok(), "template save failed");
            assert!(out.exists());
        }
    }

    // ── ValidationResult fields ───────────────────────────────────────────────

    #[test]
    fn test_validation_result_suggested_fixes_can_be_populated() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let result = manager.validate_configuration();
            // suggested_fixes is a Vec — just confirm it's accessible
            let _ = result.suggested_fixes.len();
        }
    }

    // ── MonitoringSettings ────────────────────────────────────────────────────

    #[test]
    fn test_monitoring_settings_ci_exports_metrics() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let preset = manager.create_environment_template("ci");
            assert!(preset.monitoring_settings.export_metrics);
        }
    }

    #[test]
    fn test_monitoring_settings_dev_no_export() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            let preset = manager.create_environment_template("development");
            assert!(!preset.monitoring_settings.export_metrics);
        }
    }

    #[test]
    fn test_monitoring_settings_regression_threshold_positive() {
        let dir = make_config_dir();
        if let Ok(manager) = TestConfigManager::new(&dir) {
            for env_name in &["ci", "development", "performance"] {
                let preset = manager.create_environment_template(env_name);
                assert!(
                    preset.monitoring_settings.regression_threshold > 0.0,
                    "env={} threshold={}",
                    env_name,
                    preset.monitoring_settings.regression_threshold
                );
            }
        }
    }

    // ── LCG sanity ────────────────────────────────────────────────────────────

    #[test]
    fn test_lcg_diverse_output() {
        let mut lcg = Lcg::new(42);
        let vals: Vec<f32> = (0..20).map(|_| lcg.next_f32()).collect();
        let first = vals[0];
        let diff = vals.iter().filter(|&&v| (v - first).abs() > 1e-6).count();
        assert!(diff >= 8, "LCG appears stuck");
    }
}
