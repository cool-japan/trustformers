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
