//! Experiment definitions and management

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Name of the experiment
    pub name: String,
    /// Description of what is being tested
    pub description: String,
    /// Control variant (baseline)
    pub control_variant: Variant,
    /// Treatment variants to test
    pub treatment_variants: Vec<Variant>,
    /// Percentage of traffic to include in test
    pub traffic_percentage: f64,
    /// Minimum sample size per variant
    pub min_sample_size: usize,
    /// Maximum duration in hours
    pub max_duration_hours: u64,
}

/// A variant in an A/B test
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Variant {
    /// Unique identifier for the variant
    name: String,
    /// Model or configuration identifier
    model_id: String,
    /// Optional configuration overrides
    config_overrides: Option<serde_json::Value>,
}

impl Variant {
    /// Create a new variant
    pub fn new(name: &str, model_id: &str) -> Self {
        Self {
            name: name.to_string(),
            model_id: model_id.to_string(),
            config_overrides: None,
        }
    }

    /// Create a variant with configuration overrides
    pub fn with_config(name: &str, model_id: &str, config: serde_json::Value) -> Self {
        Self {
            name: name.to_string(),
            model_id: model_id.to_string(),
            config_overrides: Some(config),
        }
    }

    /// Get variant name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get configuration overrides
    pub fn config_overrides(&self) -> Option<&serde_json::Value> {
        self.config_overrides.as_ref()
    }
}

/// An A/B test experiment
#[derive(Debug, Clone)]
pub struct Experiment {
    /// Unique experiment ID
    id: Uuid,
    /// Configuration
    config: ExperimentConfig,
    /// Current status
    status: ExperimentStatus,
    /// Start time
    start_time: Option<DateTime<Utc>>,
    /// End time
    end_time: Option<DateTime<Utc>>,
    /// Metadata
    metadata: ExperimentMetadata,
}

/// Experiment metadata
#[derive(Debug, Clone, Default)]
pub struct ExperimentMetadata {
    /// Number of requests per variant
    pub request_counts: std::collections::HashMap<String, usize>,
    /// Last update time
    pub last_updated: Option<DateTime<Utc>>,
    /// Tags for categorization
    #[allow(dead_code)]
    pub tags: Vec<String>,
    /// Owner/creator
    #[allow(dead_code)]
    pub owner: Option<String>,
}

/// Experiment status
#[derive(Debug, Clone, PartialEq)]
pub enum ExperimentStatus {
    /// Experiment is being configured
    Draft,
    /// Experiment is running
    Running,
    /// Experiment is paused
    Paused,
    /// Experiment has concluded
    Concluded,
    /// Experiment was cancelled
    Cancelled,
}

impl Experiment {
    /// Create a new experiment
    pub fn new(config: ExperimentConfig) -> Result<Self> {
        // Validate configuration
        if config.traffic_percentage <= 0.0 || config.traffic_percentage > 100.0 {
            anyhow::bail!("Traffic percentage must be between 0 and 100");
        }

        if config.treatment_variants.is_empty() {
            anyhow::bail!("At least one treatment variant is required");
        }

        if config.min_sample_size == 0 {
            anyhow::bail!("Minimum sample size must be greater than 0");
        }

        Ok(Self {
            id: Uuid::new_v4(),
            config,
            status: ExperimentStatus::Draft,
            start_time: None,
            end_time: None,
            metadata: ExperimentMetadata::default(),
        })
    }

    /// Get experiment ID
    pub fn id(&self) -> &Uuid {
        &self.id
    }

    /// Get experiment configuration
    pub fn config(&self) -> &ExperimentConfig {
        &self.config
    }

    /// Get current status
    pub fn status(&self) -> ExperimentStatus {
        self.status.clone()
    }

    /// Start the experiment
    pub fn start(&mut self) -> Result<()> {
        if self.status != ExperimentStatus::Draft {
            anyhow::bail!("Can only start experiments in Draft status");
        }

        self.status = ExperimentStatus::Running;
        self.start_time = Some(Utc::now());
        self.metadata.last_updated = Some(Utc::now());
        Ok(())
    }

    /// Pause the experiment
    pub fn pause(&mut self) -> Result<()> {
        if self.status != ExperimentStatus::Running {
            anyhow::bail!("Can only pause running experiments");
        }

        self.status = ExperimentStatus::Paused;
        self.metadata.last_updated = Some(Utc::now());
        Ok(())
    }

    /// Resume the experiment
    pub fn resume(&mut self) -> Result<()> {
        if self.status != ExperimentStatus::Paused {
            anyhow::bail!("Can only resume paused experiments");
        }

        self.status = ExperimentStatus::Running;
        self.metadata.last_updated = Some(Utc::now());
        Ok(())
    }

    /// Conclude the experiment
    pub fn conclude(&mut self) -> Result<()> {
        if self.status != ExperimentStatus::Running && self.status != ExperimentStatus::Paused {
            anyhow::bail!("Can only conclude running or paused experiments");
        }

        self.status = ExperimentStatus::Concluded;
        self.end_time = Some(Utc::now());
        self.metadata.last_updated = Some(Utc::now());
        Ok(())
    }

    /// Cancel the experiment
    pub fn cancel(&mut self) -> Result<()> {
        if self.status == ExperimentStatus::Concluded || self.status == ExperimentStatus::Cancelled
        {
            anyhow::bail!("Cannot cancel concluded or already cancelled experiments");
        }

        self.status = ExperimentStatus::Cancelled;
        self.end_time = Some(Utc::now());
        self.metadata.last_updated = Some(Utc::now());
        Ok(())
    }

    /// Check if experiment should auto-conclude
    pub fn should_auto_conclude(&self) -> bool {
        if self.status != ExperimentStatus::Running {
            return false;
        }

        // Check duration
        if let Some(start_time) = self.start_time {
            let elapsed = Utc::now() - start_time;
            if elapsed > Duration::hours(self.config.max_duration_hours as i64) {
                return true;
            }
        }

        // Check sample sizes
        let min_count = self.metadata.request_counts.values().min().copied().unwrap_or(0);
        min_count >= self.config.min_sample_size
    }

    /// Get all variants (control + treatments)
    pub fn all_variants(&self) -> Vec<&Variant> {
        let mut variants = vec![&self.config.control_variant];
        variants.extend(self.config.treatment_variants.iter());
        variants
    }

    /// Update request count for a variant
    pub fn increment_request_count(&mut self, variant_name: &str) {
        *self.metadata.request_counts.entry(variant_name.to_string()).or_insert(0) += 1;
        self.metadata.last_updated = Some(Utc::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_lifecycle() {
        let config = ExperimentConfig {
            name: "Test Experiment".to_string(),
            description: "Testing lifecycle".to_string(),
            control_variant: Variant::new("control", "model-v1"),
            treatment_variants: vec![Variant::new("treatment", "model-v2")],
            traffic_percentage: 50.0,
            min_sample_size: 100,
            max_duration_hours: 24,
        };

        let mut experiment = Experiment::new(config).unwrap();
        assert_eq!(experiment.status(), ExperimentStatus::Draft);

        // Start
        experiment.start().unwrap();
        assert_eq!(experiment.status(), ExperimentStatus::Running);
        assert!(experiment.start_time.is_some());

        // Pause
        experiment.pause().unwrap();
        assert_eq!(experiment.status(), ExperimentStatus::Paused);

        // Resume
        experiment.resume().unwrap();
        assert_eq!(experiment.status(), ExperimentStatus::Running);

        // Conclude
        experiment.conclude().unwrap();
        assert_eq!(experiment.status(), ExperimentStatus::Concluded);
        assert!(experiment.end_time.is_some());
    }

    #[test]
    fn test_variant_creation() {
        let variant = Variant::new("test", "model-123");
        assert_eq!(variant.name(), "test");
        assert_eq!(variant.model_id(), "model-123");
        assert!(variant.config_overrides().is_none());

        let config = serde_json::json!({
            "batch_size": 32,
            "temperature": 0.7
        });
        let variant_with_config = Variant::with_config("test2", "model-456", config.clone());
        assert_eq!(variant_with_config.config_overrides(), Some(&config));
    }

    #[test]
    fn test_auto_conclude() {
        let config = ExperimentConfig {
            name: "Auto Conclude Test".to_string(),
            description: "Testing auto conclusion".to_string(),
            control_variant: Variant::new("control", "model-v1"),
            treatment_variants: vec![Variant::new("treatment", "model-v2")],
            traffic_percentage: 50.0,
            min_sample_size: 2,
            max_duration_hours: 24,
        };

        let mut experiment = Experiment::new(config).unwrap();
        experiment.start().unwrap();

        // Should not auto-conclude with no samples
        assert!(!experiment.should_auto_conclude());

        // Add samples
        experiment.increment_request_count("control");
        experiment.increment_request_count("control");
        experiment.increment_request_count("treatment");
        experiment.increment_request_count("treatment");

        // Should auto-conclude when minimum samples reached
        assert!(experiment.should_auto_conclude());
    }
}
