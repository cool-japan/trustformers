//! CI/CD integration manager implementation

use anyhow::Result;
use parking_lot::RwLock;
use std::{
    collections::HashMap,
    env,
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};
use tracing::{error, info};

use crate::test_parallelization::TestParallelizationConfig;
use crate::test_performance_monitoring::types::CurrentPerformanceMetrics;

use super::types::*;

/// CI/CD integration and configuration management system
pub struct CicdIntegrationManager {
    /// Configuration
    config: Arc<RwLock<CicdIntegrationConfig>>,

    /// Environment detector
    environment_detector: Arc<EnvironmentDetector>,

    /// Configuration manager
    config_manager: Arc<ConfigurationManager>,

    /// Reporting integration
    reporting_integration: Arc<ReportingIntegration>,

    /// Metrics exporter
    metrics_exporter: Arc<MetricsExporter>,

    /// Environment optimizer
    environment_optimizer: Arc<EnvironmentOptimizer>,

    /// Background tasks
    background_tasks: Vec<tokio::task::JoinHandle<()>>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

impl CicdIntegrationManager {
    /// Create a new CI/CD integration manager
    pub fn new(config: CicdIntegrationConfig) -> Result<Self> {
        let config = Arc::new(RwLock::new(config));
        let shutdown = Arc::new(AtomicBool::new(false));

        Ok(Self {
            config: config.clone(),
            environment_detector: Arc::new(EnvironmentDetector::new()?),
            config_manager: Arc::new(ConfigurationManager::new(config.clone())?),
            reporting_integration: Arc::new(ReportingIntegration::new()?),
            metrics_exporter: Arc::new(MetricsExporter::new()?),
            environment_optimizer: Arc::new(EnvironmentOptimizer::new()?),
            background_tasks: Vec::new(),
            shutdown,
        })
    }

    /// Start the CI/CD integration manager
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting CI/CD integration manager");

        // Detect current environment
        let environment = self.environment_detector.detect_environment().await?;
        info!("Detected environment: {:?}", environment);

        // Load environment-specific configuration
        self.config_manager.load_environment_config(&environment).await?;

        // Start background tasks
        self.start_background_tasks().await?;

        info!("CI/CD integration manager started successfully");
        Ok(())
    }

    /// Stop the CI/CD integration manager
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping CI/CD integration manager");

        // Signal shutdown
        self.shutdown.store(true, std::sync::atomic::Ordering::SeqCst);

        // Wait for background tasks to complete
        for task in self.background_tasks.drain(..) {
            let _ = task.await;
        }

        info!("CI/CD integration manager stopped");
        Ok(())
    }

    /// The whole CI/CD configuration this manager was built with.
    pub fn config(&self) -> CicdIntegrationConfig {
        self.config.read().clone()
    }

    /// The environment this manager detected, or `None` before [`Self::start`].
    pub fn detected_environment(&self) -> Option<EnvironmentType> {
        self.environment_detector.detected_environment()
    }

    /// The parallelization configuration to run with.
    ///
    /// When [`Self::start`] found an `environment_configs` block for the
    /// detected environment, that block's `test_config` is what the operator
    /// asked for and is returned verbatim. Otherwise there is nothing
    /// environment-specific to apply and the fallback default is used.
    pub async fn get_optimized_config(&self) -> Result<TestParallelizationConfig> {
        if let Some(environment_config) = self.config_manager.active_environment_config() {
            return Ok(environment_config.test_config);
        }
        self.environment_optimizer.get_optimized_config().await
    }

    /// Report test results
    pub async fn report_results(&self, results: &[ExecutionResult]) -> Result<()> {
        self.reporting_integration.report_results(results).await
    }

    /// Export metrics
    pub async fn export_metrics(&self, metrics: &CurrentPerformanceMetrics) -> Result<()> {
        self.metrics_exporter.export_metrics(metrics).await
    }

    async fn start_background_tasks(&mut self) -> Result<()> {
        // Start configuration monitoring task
        let config_task = {
            let config_manager = self.config_manager.clone();
            let shutdown = self.shutdown.clone();
            tokio::spawn(async move {
                while !shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                    if let Err(e) = config_manager.monitor_configuration().await {
                        error!("Configuration monitoring error: {}", e);
                    }
                    tokio::time::sleep(Duration::from_secs(60)).await;
                }
            })
        };
        self.background_tasks.push(config_task);

        // Start metrics export task
        let metrics_task = {
            let metrics_exporter = self.metrics_exporter.clone();
            let shutdown = self.shutdown.clone();
            tokio::spawn(async move {
                while !shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                    if let Err(e) = metrics_exporter.periodic_export().await {
                        error!("Metrics export error: {}", e);
                    }
                    tokio::time::sleep(Duration::from_secs(300)).await;
                }
            })
        };
        self.background_tasks.push(metrics_task);

        Ok(())
    }
}

impl Default for CicdIntegrationConfig {
    fn default() -> Self {
        Self {
            enabled_features: vec![
                CicdFeature::EnvironmentDetection,
                CicdFeature::AutoConfiguration,
                CicdFeature::PerformanceReporting,
            ],
            environment_configs: HashMap::new(),
            pipeline_integration: Default::default(),
            reporting_config: Default::default(),
            metrics_export: Default::default(),
            optimization_config: Default::default(),
            security_config: Default::default(),
            default_config: TestParallelizationConfig::default(),
        }
    }
}

/// Detects which CI system (if any) this process is running under, from the
/// environment variables those systems set.
pub struct EnvironmentDetector {
    /// The last environment [`Self::detect_environment`] resolved. `None` until
    /// the first detection: no environment is assumed before one is observed.
    detected_environment: Arc<RwLock<Option<EnvironmentType>>>,
}

impl EnvironmentDetector {
    pub fn new() -> Result<Self> {
        Ok(Self {
            detected_environment: Arc::new(RwLock::new(None)),
        })
    }

    /// Classify the environment from the CI vendor variables present, and
    /// remember the answer so [`Self::detected_environment`] can report it.
    pub async fn detect_environment(&self) -> Result<EnvironmentType> {
        // Check for CI environment variables
        let environment = if env::var("GITHUB_ACTIONS").is_ok() {
            EnvironmentType::Pipeline(PipelineType::GitHubActions)
        } else if env::var("GITLAB_CI").is_ok() {
            EnvironmentType::Pipeline(PipelineType::GitLabCi)
        } else if env::var("JENKINS_URL").is_ok() {
            EnvironmentType::Pipeline(PipelineType::Jenkins)
        } else if env::var("CIRCLECI").is_ok() {
            EnvironmentType::Pipeline(PipelineType::CircleCi)
        } else {
            // No CI vendor announced itself.
            EnvironmentType::Development
        };

        *self.detected_environment.write() = Some(environment.clone());
        Ok(environment)
    }

    /// The environment detected by the most recent [`Self::detect_environment`]
    /// call, or `None` when detection has never run.
    pub fn detected_environment(&self) -> Option<EnvironmentType> {
        self.detected_environment.read().clone()
    }
}

/// Selects the environment-specific block of a [`CicdIntegrationConfig`].
pub struct ConfigurationManager {
    /// The whole CI/CD configuration, shared with the owning manager.
    config: Arc<RwLock<CicdIntegrationConfig>>,
    /// The environment block selected by the last
    /// [`Self::load_environment_config`], if the configuration had one for that
    /// environment.
    active: Arc<RwLock<Option<EnvironmentConfig>>>,
}

impl ConfigurationManager {
    pub fn new(config: Arc<RwLock<CicdIntegrationConfig>>) -> Result<Self> {
        Ok(Self {
            config,
            active: Arc::new(RwLock::new(None)),
        })
    }

    /// Pick the `environment_configs` entry that matches `environment` and make
    /// it the active one.
    ///
    /// 0.2.1: this used to log a line and return `Ok(())` without reading the
    /// configuration at all -- `ConfigurationManager` held its `config` handle
    /// and never touched it. Absence is still not an error: a configuration
    /// that names no block for this environment simply leaves none active, and
    /// callers fall back to their own defaults.
    pub async fn load_environment_config(&self, environment: &EnvironmentType) -> Result<()> {
        let selected = {
            let config = self.config.read();
            config
                .environment_configs
                .values()
                .find(|candidate| candidate.environment_type == *environment)
                .cloned()
        };

        match &selected {
            Some(found) => info!(
                "Loaded configuration '{}' for environment {:?}",
                found.name, environment
            ),
            None => info!("No configuration block for environment {:?}", environment),
        }
        *self.active.write() = selected;
        Ok(())
    }

    /// The environment block currently in force, if one was found.
    pub fn active_environment_config(&self) -> Option<EnvironmentConfig> {
        self.active.read().clone()
    }

    /// Re-read the shared configuration.
    ///
    /// The configuration lives behind a shared lock, so "monitoring" it means
    /// nothing more than looking again: there is no file watcher or remote
    /// config source in this crate to poll.
    pub async fn monitor_configuration(&self) -> Result<()> {
        if let Some(active) = self.active_environment_config() {
            let still_present = {
                let config = self.config.read();
                config
                    .environment_configs
                    .values()
                    .any(|candidate| candidate.environment_type == active.environment_type)
            };
            if !still_present {
                info!(
                    "Environment block '{}' was removed from the configuration",
                    active.name
                );
                *self.active.write() = None;
            }
        }
        Ok(())
    }
}

/// Reporting integration
pub struct ReportingIntegration;

impl ReportingIntegration {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    /// Accepts results and drops them.
    ///
    /// There is no reporting sink in this crate -- no CI annotation API, no
    /// artifact writer -- so nothing is published. `Ok(())` here means "nothing
    /// went wrong", not "the results were reported somewhere".
    pub async fn report_results(&self, _results: &[ExecutionResult]) -> Result<()> {
        Ok(())
    }
}

/// Metrics exporter
pub struct MetricsExporter;

impl MetricsExporter {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    /// Accepts metrics and drops them: this crate has no metrics sink wired to
    /// the exporter, so nothing leaves the process.
    pub async fn export_metrics(&self, _metrics: &CurrentPerformanceMetrics) -> Result<()> {
        Ok(())
    }

    /// The periodic tick of [`Self::export_metrics`]; likewise a no-op.
    pub async fn periodic_export(&self) -> Result<()> {
        Ok(())
    }
}

/// Environment optimizer
pub struct EnvironmentOptimizer;

impl EnvironmentOptimizer {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    /// The fallback parallelization configuration.
    ///
    /// This applies no optimization: there is no environment model here to
    /// optimize against, so it returns the crate default unchanged. The
    /// environment-specific configuration an operator supplies is applied by
    /// [`CicdIntegrationManager::get_optimized_config`] instead.
    pub async fn get_optimized_config(&self) -> Result<TestParallelizationConfig> {
        Ok(TestParallelizationConfig::default())
    }
}

// Default implementations for other types
impl Default for super::pipeline::PipelineIntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            integration_types: vec![],
            settings: Default::default(),
            rate_limiting: Default::default(),
            error_handling: Default::default(),
            hooks: Default::default(),
            artifact_management: Default::default(),
            notifications: Default::default(),
        }
    }
}

impl Default for super::pipeline::IntegrationSettings {
    fn default() -> Self {
        Self {
            connection_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(60),
            max_concurrent_connections: 10,
            retry_config: Default::default(),
            auth_config: super::environment::AuthConfig::None,
            custom_headers: HashMap::new(),
        }
    }
}

impl Default for super::pipeline::RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: super::pipeline::RateLimitStrategy::TokenBucket,
            requests_per_minute: 60,
            burst_capacity: 10,
            window_duration: Duration::from_secs(60),
        }
    }
}

impl Default for super::pipeline::ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retry_attempts: 3,
            retry_delay: Duration::from_secs(1),
            fallback_actions: vec![],
            circuit_breaker_enabled: false,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout: Duration::from_secs(60),
        }
    }
}

impl Default for super::pipeline::HookConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            pre_test_hooks: vec![],
            post_test_hooks: vec![],
            pre_optimization_hooks: vec![],
            post_optimization_hooks: vec![],
            error_hooks: vec![],
            hook_timeout: Duration::from_secs(30),
        }
    }
}

impl Default for super::pipeline::ArtifactManagementConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            storage: Default::default(),
            artifact_types: HashMap::new(),
            retention: Default::default(),
            compression: Default::default(),
            upload: Default::default(),
            encryption: Default::default(),
        }
    }
}

impl Default for super::pipeline::ArtifactStorageConfig {
    fn default() -> Self {
        Self {
            storage_type: super::pipeline::ArtifactStorageType::Local {
                path: "/tmp/artifacts".to_string(),
            },
            base_path: "/tmp/artifacts".to_string(),
            max_storage_size: None,
            quota_per_project: None,
        }
    }
}

impl Default for super::pipeline::RetentionPolicy {
    fn default() -> Self {
        Self {
            default_retention_days: 30,
            rules: vec![],
            auto_cleanup: true,
            cleanup_schedule: "0 2 * * *".to_string(), // Daily at 2 AM
        }
    }
}

impl Default for super::pipeline::CompressionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: super::pipeline::CompressionAlgorithm::Gzip,
            level: 6,
            min_file_size: 1024, // 1KB
        }
    }
}

impl Default for super::pipeline::UploadSettings {
    fn default() -> Self {
        Self {
            parallel_uploads: true,
            max_concurrent_uploads: 5,
            chunk_size: 8 * 1024 * 1024, // 8MB
            timeout: Duration::from_secs(300),
            validation: Default::default(),
        }
    }
}

impl Default for super::pipeline::UploadValidation {
    fn default() -> Self {
        Self {
            enabled: true,
            checksum_validation: true,
            size_validation: true,
            rules: vec![],
        }
    }
}

impl Default for super::pipeline::EncryptionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: super::pipeline::EncryptionAlgorithm::Aes256,
            key_management: super::pipeline::KeyManagement::Static {
                key: "default-key".to_string(),
            },
            scope: vec![],
        }
    }
}

impl Default for super::environment::NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            channels: vec![],
            templates: vec![],
            rules: vec![],
        }
    }
}

impl Default for super::environment::RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_seconds: 1.0,
            max_delay_seconds: 60.0,
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl Default for super::reporting::ReportingConfig {
    fn default() -> Self {
        Self {
            formats: vec![super::reporting::ReportFormat::Json],
            destinations: vec![],
            schedule: Default::default(),
            templates: vec![],
            filters: vec![],
        }
    }
}

impl Default for super::reporting::ReportSchedule {
    fn default() -> Self {
        Self {
            enabled: false,
            cron: "0 0 * * *".to_string(), // Daily at midnight
            timezone: "UTC".to_string(),
            next_execution: None,
        }
    }
}

impl Default for super::reporting::MetricsExportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            targets: vec![],
            schedule: Default::default(),
            transformations: vec![],
        }
    }
}

impl Default for super::reporting::ExportSchedule {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(300), // 5 minutes
            batch_size: 100,
            timeout: Duration::from_secs(60),
            compression: true,
        }
    }
}

impl Default for super::optimization::OptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: super::optimization::OptimizationStrategy::Greedy,
            targets: vec![],
            constraints: vec![],
            learning: Default::default(),
            schedule: Default::default(),
            model_persistence: Default::default(),
        }
    }
}

impl Default for super::optimization::LearningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: super::optimization::LearningAlgorithm::QLearning,
            learning_rate: 0.01,
            exploration_rate: 0.1,
            training_episodes: 1000,
            memory_size: 10000,
            batch_size: 32,
            update_frequency: 100,
        }
    }
}

impl Default for super::optimization::OptimizationSchedule {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: Duration::from_secs(3600), // 1 hour
            min_data_points: 100,
            cooldown_period: Duration::from_secs(600), // 10 minutes
            max_optimization_time: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl Default for super::optimization::ModelPersistence {
    fn default() -> Self {
        Self {
            enabled: false,
            save_path: "/tmp/models".to_string(),
            format: super::optimization::ModelFormat::Json,
            save_frequency: 100,
            keep_best_only: true,
            max_models: 10,
            compression: true,
        }
    }
}

impl Default for super::security::SecurityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            access_control: Default::default(),
            api_key_management: Default::default(),
            encryption: Default::default(),
            key_management: Default::default(),
            audit: Default::default(),
            compliance: Default::default(),
        }
    }
}

impl Default for super::security::AccessControlConfig {
    fn default() -> Self {
        Self {
            authorization_scheme: super::security::AuthorizationScheme::None,
            rbac: None,
            session_timeout: Duration::from_secs(3600), // 1 hour
            mfa_required: false,
            ip_restrictions: vec![],
        }
    }
}

impl Default for super::security::ApiKeyManagement {
    fn default() -> Self {
        Self {
            generation: Default::default(),
            rotation: Default::default(),
            validation: Default::default(),
            storage: Default::default(),
        }
    }
}

impl Default for super::security::KeyGenerationSettings {
    fn default() -> Self {
        Self {
            algorithm: super::security::KeyAlgorithm::Random,
            key_length: 256,
            entropy_source: super::security::EntropySource::SystemRandom,
            key_prefix: None,
            key_suffix: None,
        }
    }
}

impl Default for super::security::KeyRotationSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: super::security::RotationStrategy::TimeBased,
            interval: Duration::from_secs(86400 * 30), // 30 days
            grace_period: Duration::from_secs(86400),  // 1 day
            max_key_age: Duration::from_secs(86400 * 90), // 90 days
        }
    }
}

impl Default for super::security::KeyValidationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_validation: Default::default(),
            rate_limiting: false,
            max_attempts: 5,
            lockout_duration: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl Default for super::security::ValidationCaching {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl: Duration::from_secs(300), // 5 minutes
            max_size: 1000,
        }
    }
}

impl Default for super::security::KeyStorageSettings {
    fn default() -> Self {
        Self {
            storage_type: super::security::KeyStorageType::Memory,
            encryption_at_rest: true,
            backup: Default::default(),
            access_logging: true,
        }
    }
}

impl Default for super::security::BackupSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: Duration::from_secs(86400), // 1 day
            location: "/tmp/backups".to_string(),
            encryption: true,
            retention_period: Duration::from_secs(86400 * 30), // 30 days
        }
    }
}

impl Default for super::security::EncryptionConfig {
    fn default() -> Self {
        Self {
            data_at_rest: Default::default(),
            data_in_transit: Default::default(),
        }
    }
}

impl Default for super::security::DataAtRestEncryption {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: "AES-256-GCM".to_string(),
            key_derivation: Default::default(),
        }
    }
}

impl Default for super::security::KeyDerivation {
    fn default() -> Self {
        Self {
            function: super::security::DerivationFunction::Pbkdf2,
            salt_generation: Default::default(),
            iterations: 100000,
        }
    }
}

impl Default for super::security::SaltGeneration {
    fn default() -> Self {
        Self {
            length: 32,
            per_key: true,
        }
    }
}

impl Default for super::security::DataInTransitEncryption {
    fn default() -> Self {
        Self {
            tls_required: false,
            min_tls_version: "1.2".to_string(),
            certificate_management: Default::default(),
        }
    }
}

impl Default for super::security::CertificateManagement {
    fn default() -> Self {
        Self {
            source: super::security::CertificateSource::SelfSigned,
            validation: Default::default(),
            rotation: Default::default(),
        }
    }
}

impl Default for super::security::CertificateValidation {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: vec![],
            ocsp_checking: false,
            crl_checking: false,
        }
    }
}

impl Default for super::security::CertificateRotation {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: super::security::CertificateRotationStrategy::Automatic,
            renewal_threshold_days: 30,
            grace_period: Duration::from_secs(86400 * 7), // 7 days
        }
    }
}

impl Default for super::security::KeyManagementConfig {
    fn default() -> Self {
        Self {
            storage: Default::default(),
            lifecycle: Default::default(),
            access_control: Default::default(),
        }
    }
}

impl Default for super::security::KeyStorageConfig {
    fn default() -> Self {
        Self {
            backend: super::security::KeyStorageBackend::File {
                path: "/tmp/keys".to_string(),
                encryption_key: None,
            },
            backup: Default::default(),
            encryption_at_rest: true,
            access_logging: true,
        }
    }
}

impl Default for super::security::BackupConfiguration {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: super::security::BackupStrategy::Full,
            retention: Default::default(),
            archive: Default::default(),
        }
    }
}

impl Default for super::security::BackupRetention {
    fn default() -> Self {
        Self {
            daily_backups: 7,
            weekly_backups: 4,
            monthly_backups: 12,
            yearly_backups: 5,
        }
    }
}

impl Default for super::security::ArchiveSettings {
    fn default() -> Self {
        Self {
            format: super::security::ArchiveFormat::Tar,
            compression: true,
            encryption: true,
        }
    }
}

impl Default for super::security::KeyLifecycleConfig {
    fn default() -> Self {
        Self {
            generation: Default::default(),
            rotation: Default::default(),
            retirement: Default::default(),
        }
    }
}

impl Default for super::security::KeyGenerationConfig {
    fn default() -> Self {
        Self {
            algorithm: super::security::KeyGenerationAlgorithm::Aes,
            key_size: 256,
            entropy_requirements: "high".to_string(),
        }
    }
}

impl Default for super::security::KeyRotationConfig {
    fn default() -> Self {
        Self {
            schedule: super::security::RotationSchedule::Monthly,
            triggers: vec![],
            overlap_period: Duration::from_secs(86400), // 1 day
        }
    }
}

impl Default for super::security::KeyRetirementConfig {
    fn default() -> Self {
        Self {
            policy: super::security::KeyRetirementPolicy::Graceful(Duration::from_secs(86400 * 7)), // 7 days
            grace_period: Duration::from_secs(86400), // 1 day
            archive: true,
        }
    }
}

impl Default for super::security::KeyAccessControl {
    fn default() -> Self {
        Self {
            policies: vec![],
            permission_model: super::security::KeyPermissionModel::Whitelist,
            audit_access: true,
        }
    }
}

impl Default for super::security::AuditConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            targets: vec![],
            storage: Default::default(),
            retention: Default::default(),
            archive: Default::default(),
            notifications: Default::default(),
        }
    }
}

impl Default for super::security::AuditStorageConfig {
    fn default() -> Self {
        Self {
            backend: super::security::AuditStorageBackend::File {
                path: "/tmp/audit".to_string(),
            },
            format: super::security::AuditStorageFormat::Json,
            encryption: true,
            integrity_protection: true,
        }
    }
}

impl Default for super::security::AuditRetentionConfig {
    fn default() -> Self {
        Self {
            policy: super::security::AuditRetentionPolicy::TimeBased,
            retention_period: Duration::from_secs(86400 * 365), // 1 year
            archive_before_deletion: true,
        }
    }
}

impl Default for super::security::AuditArchiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            location: "/tmp/audit-archive".to_string(),
            format: super::security::ArchiveFormat::Tar,
            compression: true,
            encryption: true,
        }
    }
}

impl Default for super::security::AuditNotificationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            rules: vec![],
            alerts: Default::default(),
        }
    }
}

impl Default for super::security::AuditAlertConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            templates: vec![],
            escalation: Default::default(),
        }
    }
}

impl Default for super::security::AlertEscalationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            levels: vec![],
        }
    }
}

impl Default for super::security::ComplianceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            frameworks: vec![],
            reporting: Default::default(),
            monitoring: Default::default(),
            dashboard: Default::default(),
        }
    }
}

impl Default for super::security::ComplianceReportingConfig {
    fn default() -> Self {
        Self {
            generation: Default::default(),
            distribution: Default::default(),
            templates: vec![],
        }
    }
}

impl Default for super::security::ComplianceReportGeneration {
    fn default() -> Self {
        Self {
            enabled: false,
            schedule: "0 0 1 * *".to_string(), // Monthly on the 1st
            formats: vec!["PDF".to_string()],
            include_evidence: true,
        }
    }
}

impl Default for super::security::ComplianceReportDistribution {
    fn default() -> Self {
        Self {
            enabled: false,
            channels: vec![],
            recipients: vec![],
            encryption_required: true,
        }
    }
}

impl Default for super::security::ComplianceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            rules: vec![],
            alerts: Default::default(),
        }
    }
}

impl Default for super::security::ComplianceAlertConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            templates: vec![],
            escalation: Default::default(),
        }
    }
}

impl Default for super::security::ComplianceDashboardConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            widgets: vec![],
            access_control: Default::default(),
        }
    }
}

impl Default for super::security::DashboardAccessControl {
    fn default() -> Self {
        Self {
            authentication_required: true,
            restrictions: vec![],
            session_timeout: Duration::from_secs(3600), // 1 hour
        }
    }
}

#[cfg(test)]
mod manager_tests {
    use super::*;
    use crate::test_cicd_integration::environment::{
        EnvironmentMonitoringConfig, EnvironmentOptimizationSettings, EnvironmentResourceLimits,
        EnvironmentSecuritySettings,
    };

    fn environment_config(name: &str, environment_type: EnvironmentType) -> EnvironmentConfig {
        let mut test_config = TestParallelizationConfig::default();
        // A value no default will ever produce, so a fallback cannot pass as
        // "the operator's configuration".
        test_config.max_concurrent_tests = 4242;
        EnvironmentConfig {
            name: name.to_string(),
            environment_type,
            test_config,
            resource_limits: EnvironmentResourceLimits::default(),
            optimization: EnvironmentOptimizationSettings::default(),
            security: EnvironmentSecuritySettings::default(),
            monitoring: EnvironmentMonitoringConfig::default(),
            environment_variables: HashMap::new(),
            overrides: HashMap::new(),
        }
    }

    /// The configuration a caller passes in must survive to the accessor, not
    /// be silently replaced by a default.
    #[tokio::test]
    async fn manager_reports_the_configuration_it_was_built_with() {
        let mut config = CicdIntegrationConfig::default();
        config.environment_configs.insert(
            "testing".to_string(),
            environment_config("testing", EnvironmentType::Testing),
        );

        let manager = CicdIntegrationManager::new(config).expect("construction should succeed");

        assert!(
            manager.config().environment_configs.contains_key("testing"),
            "the manager must expose the configuration it was given"
        );
        assert!(
            manager.detected_environment().is_none(),
            "no environment may be reported before detection has run"
        );
    }

    /// Loading an environment block must actually select it, and the selected
    /// block's own test configuration must be what `get_optimized_config`
    /// returns.
    #[tokio::test]
    async fn loading_an_environment_block_selects_and_applies_it() {
        let mut config = CicdIntegrationConfig::default();
        config.environment_configs.insert(
            "testing".to_string(),
            environment_config("testing", EnvironmentType::Testing),
        );
        let manager = CicdIntegrationManager::new(config).expect("construction should succeed");

        // Nothing selected yet: the fallback default is what comes back.
        let fallback = manager
            .get_optimized_config()
            .await
            .expect("fallback config should be available");
        assert_ne!(fallback.max_concurrent_tests, 4242);

        manager
            .config_manager
            .load_environment_config(&EnvironmentType::Testing)
            .await
            .expect("loading should succeed");

        let selected = manager
            .config_manager
            .active_environment_config()
            .expect("the testing block should now be active");
        assert_eq!(selected.name, "testing");
        let applied = manager
            .get_optimized_config()
            .await
            .expect("applied config should be available");
        assert_eq!(
            applied.max_concurrent_tests, 4242,
            "the operator's environment block must be what is applied"
        );
    }

    /// An environment with no configured block leaves nothing active, rather
    /// than inventing one.
    #[tokio::test]
    async fn an_unconfigured_environment_selects_nothing() {
        let manager = CicdIntegrationManager::new(CicdIntegrationConfig::default())
            .expect("construction should succeed");

        manager
            .config_manager
            .load_environment_config(&EnvironmentType::Production)
            .await
            .expect("loading should succeed");

        assert!(
            manager.config_manager.active_environment_config().is_none(),
            "no block was configured for production, so none may be active"
        );
    }

    /// Detection must record what it found so it can be read back.
    #[tokio::test]
    async fn detection_is_remembered() {
        let detector = EnvironmentDetector::new().expect("construction should succeed");
        assert!(detector.detected_environment().is_none());

        let detected = detector.detect_environment().await.expect("detection should succeed");
        assert_eq!(
            detector.detected_environment(),
            Some(detected),
            "the detector must report the environment it just resolved"
        );
    }
}
