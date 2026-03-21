//! Configuration Manager
//!
//! Manager for centralized configuration.

use super::super::types::*;
use super::*;

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex as TokioMutex, Notify, RwLock as TokioRwLock};
use tokio::task::{spawn, JoinHandle};
use tokio::time::interval;
use tracing::{debug, info, instrument};

#[derive(Debug)]
pub struct ConfigurationManager {
    /// Main configuration
    main_config: Arc<TokioRwLock<TestCharacterizationConfig>>,
    /// Engine configuration
    engine_config: Arc<TokioRwLock<EngineConfig>>,
    /// Configuration validation rules
    validation_rules: Arc<ConfigValidationRules>,
    /// Configuration refresh task handle
    refresh_task: Arc<TokioMutex<Option<JoinHandle<()>>>>,
    /// Configuration change notifier
    change_notifier: Arc<Notify>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

/// Configuration validation rules
pub struct ConfigValidationRules {
    /// Maximum allowed values
    pub max_values: HashMap<String, f64>,
    /// Minimum allowed values
    pub min_values: HashMap<String, f64>,
    /// Required fields
    pub required_fields: Vec<String>,
    /// Validation functions
    pub custom_validators:
        Vec<Box<dyn Fn(&TestCharacterizationConfig) -> Result<()> + Send + Sync>>,
}

impl std::fmt::Debug for ConfigValidationRules {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigValidationRules")
            .field("max_values", &self.max_values)
            .field("min_values", &self.min_values)
            .field("required_fields", &self.required_fields)
            .field(
                "custom_validators",
                &format!("<{} validators>", self.custom_validators.len()),
            )
            .finish()
    }
}

impl Default for ConfigValidationRules {
    fn default() -> Self {
        let mut max_values = HashMap::new();
        max_values.insert("max_concurrent_analyses".to_string(), 64.0);
        max_values.insert("cache_ttl_seconds".to_string(), 86400.0); // 24 hours

        let mut min_values = HashMap::new();
        min_values.insert("max_concurrent_analyses".to_string(), 1.0);
        min_values.insert("cache_ttl_seconds".to_string(), 60.0); // 1 minute

        Self {
            max_values,
            min_values,
            required_fields: vec![],
            custom_validators: vec![],
        }
    }
}

impl ConfigurationManager {
    /// Create a new configuration manager
    pub async fn new(
        main_config: TestCharacterizationConfig,
        engine_config: EngineConfig,
    ) -> Result<Self> {
        Ok(Self {
            main_config: Arc::new(TokioRwLock::new(main_config)),
            engine_config: Arc::new(TokioRwLock::new(engine_config)),
            validation_rules: Arc::new(ConfigValidationRules::default()),
            refresh_task: Arc::new(TokioMutex::new(None)),
            change_notifier: Arc::new(Notify::new()),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Update configuration with validation
    #[instrument(skip(self, new_config))]
    pub async fn update_configuration(&self, new_config: TestCharacterizationConfig) -> Result<()> {
        info!("Updating configuration");

        // Validate configuration
        self.validate_configuration(&new_config).await?;

        // Update configuration
        {
            let mut config = self.main_config.write().await;
            *config = new_config;
        }

        // Notify about configuration change
        self.change_notifier.notify_waiters();

        info!("Configuration updated successfully");
        Ok(())
    }

    /// Validate configuration
    async fn validate_configuration(&self, config: &TestCharacterizationConfig) -> Result<()> {
        // Validate using validation rules
        for validator in &self.validation_rules.custom_validators {
            validator(config)?;
        }

        // Basic validation
        if config.analysis_timeout_seconds == 0 {
            return Err(anyhow!("Analysis timeout must be greater than 0"));
        }

        Ok(())
    }

    /// Get current configuration
    pub async fn get_configuration(&self) -> TestCharacterizationConfig {
        self.main_config.read().await.clone()
    }

    /// Get engine configuration
    pub async fn get_engine_configuration(&self) -> EngineConfig {
        self.engine_config.read().await.clone()
    }

    /// Start configuration refresh task
    pub async fn start_refresh_task(&self) -> Result<()> {
        let config = self.engine_config.clone();
        let shutdown = self.shutdown.clone();

        let task = spawn(async move {
            let config_guard = config.read().await;
            let mut interval = interval(Duration::from_secs(
                config_guard.config_refresh_interval_seconds,
            ));
            drop(config_guard);

            while !shutdown.load(Ordering::Acquire) {
                interval.tick().await;

                // Perform configuration refresh operations
                // This could include reloading from file, checking for updates, etc.
                debug!("Configuration refresh tick");
            }
        });

        let mut refresh_task = self.refresh_task.lock().await;
        *refresh_task = Some(task);

        Ok(())
    }

    /// Wait for configuration changes
    pub async fn wait_for_changes(&self) {
        self.change_notifier.notified().await;
    }

    /// Shutdown configuration manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down ConfigurationManager");

        self.shutdown.store(true, Ordering::Release);

        // Cancel refresh task
        let mut refresh_task = self.refresh_task.lock().await;
        if let Some(task) = refresh_task.take() {
            task.abort();
        }

        info!("ConfigurationManager shutdown completed");
        Ok(())
    }
}
