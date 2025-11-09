//! Standardized builder patterns for TrustformeRS Core
//!
//! This module provides a consistent builder pattern implementation
//! that can be used across all modules for configuration and object construction.

#![allow(unused_variables)] // Builder pattern

use crate::errors::Result;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;

/// Standard builder trait that all builders should implement
pub trait Builder<T> {
    /// Build the final object
    fn build(self) -> Result<T>;

    /// Validate the current builder state
    fn validate(&self) -> Result<()> {
        Ok(())
    }

    /// Reset the builder to default state
    fn reset(self) -> Self
    where
        T: Default;
}

/// Configuration builder trait for objects that have configuration
pub trait ConfigBuilder<T, C>: Builder<T> {
    /// Set configuration
    fn config(self, config: C) -> Self;

    /// Get current configuration (if any)
    fn get_config(&self) -> Option<&C>;
}

/// Standard builder implementation using the builder pattern
#[derive(Debug, Clone)]
pub struct StandardBuilder<T, S = BuilderComplete> {
    data: T,
    _state: PhantomData<S>,
}

/// Builder state types for compile-time validation
#[derive(Debug, Clone)]
pub struct BuilderIncomplete;

#[derive(Debug, Clone)]
pub struct BuilderComplete;

/// Trait for objects that can be built with a standard builder
pub trait Buildable: Sized + Default {
    type Builder: Builder<Self>;

    /// Create a new builder for this type
    fn builder() -> Self::Builder;
}

/// Standard configuration trait
pub trait StandardConfig: Debug + Clone + Default + Serialize + for<'de> Deserialize<'de> {
    /// Validate the configuration
    fn validate(&self) -> Result<()> {
        Ok(())
    }

    /// Merge with another configuration (self takes precedence)
    fn merge(self, other: Self) -> Self {
        self
    }
}

impl<T> Default for StandardBuilder<T, BuilderIncomplete>
where
    T: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StandardBuilder<T, BuilderIncomplete>
where
    T: Default,
{
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            data: T::default(),
            _state: PhantomData,
        }
    }

    /// Create a builder from existing data
    pub fn from(data: T) -> Self {
        Self {
            data,
            _state: PhantomData,
        }
    }
}

impl<T> StandardBuilder<T, BuilderIncomplete>
where
    T: Clone,
{
    /// Get a mutable reference to the data
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Mark builder as complete (ready to build)
    pub fn complete(self) -> StandardBuilder<T, BuilderComplete> {
        StandardBuilder {
            data: self.data,
            _state: PhantomData,
        }
    }
}

impl<T> StandardBuilder<T, BuilderComplete>
where
    T: Clone,
{
    /// Get a reference to the data
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Get a mutable reference to the data
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T> Builder<T> for StandardBuilder<T, BuilderComplete>
where
    T: Clone + Default,
{
    fn build(self) -> Result<T> {
        self.validate()?;
        Ok(self.data)
    }

    fn reset(self) -> Self {
        Self {
            data: T::default(),
            _state: PhantomData,
        }
    }
}

/// Fluent builder macro for creating builder methods
#[macro_export]
macro_rules! builder_methods {
    (
        $builder_type:ty,
        $target_type:ty,
        {
            $(
                $method_name:ident : $field_type:ty = $field_name:ident
            ),* $(,)?
        }
    ) => {
        impl $builder_type {
            $(
                #[doc = concat!("Set ", stringify!($field_name))]
                pub fn $method_name(mut self, value: $field_type) -> Self {
                    self.data.$field_name = value;
                    self
                }
            )*
        }
    };
}

/// Type alias for validation function
pub type ValidationFn<T> = Box<dyn Fn(&T) -> Result<()> + Send + Sync>;

/// Validation builder for objects that need validation
pub struct ValidatedBuilder<T> {
    data: T,
    validators: Vec<ValidationFn<T>>,
}

impl<T> Default for ValidatedBuilder<T>
where
    T: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ValidatedBuilder<T>
where
    T: Default,
{
    /// Create a new validated builder
    pub fn new() -> Self {
        Self {
            data: T::default(),
            validators: Vec::new(),
        }
    }

    /// Add a validation function
    pub fn add_validator<F>(mut self, validator: F) -> Self
    where
        F: Fn(&T) -> Result<()> + Send + Sync + 'static,
    {
        self.validators.push(Box::new(validator));
        self
    }

    /// Get a reference to the data
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Get a mutable reference to the data
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T> Builder<T> for ValidatedBuilder<T>
where
    T: Clone,
{
    fn build(self) -> Result<T> {
        self.validate()?;
        Ok(self.data)
    }

    fn validate(&self) -> Result<()> {
        for validator in &self.validators {
            validator(&self.data)?;
        }
        Ok(())
    }

    fn reset(mut self) -> Self
    where
        T: Default,
    {
        self.data = T::default();
        self
    }
}

/// Configuration builder with validation
#[derive(Debug, Clone)]
pub struct ConfigBuilderImpl<T, C> {
    target: Option<T>,
    config: Option<C>,
    name: Option<String>,
    description: Option<String>,
    tags: Vec<String>,
}

impl<T, C> Default for ConfigBuilderImpl<T, C>
where
    C: StandardConfig,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, C> ConfigBuilderImpl<T, C>
where
    C: StandardConfig,
{
    /// Create a new config builder
    pub fn new() -> Self {
        Self {
            target: None,
            config: None,
            name: None,
            description: None,
            tags: Vec::new(),
        }
    }

    /// Set name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add tag
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags.extend(tags);
        self
    }

    /// Set target object
    pub fn target(mut self, target: T) -> Self {
        self.target = Some(target);
        self
    }
}

impl<T, C> ConfigBuilder<T, C> for ConfigBuilderImpl<T, C>
where
    C: StandardConfig,
    T: Default,
{
    fn config(mut self, config: C) -> Self {
        self.config = Some(config);
        self
    }

    fn get_config(&self) -> Option<&C> {
        self.config.as_ref()
    }
}

impl<T, C> Builder<T> for ConfigBuilderImpl<T, C>
where
    T: Default,
    C: StandardConfig,
{
    fn build(self) -> Result<T> {
        self.validate()?;
        Ok(self.target.unwrap_or_default())
    }

    fn validate(&self) -> Result<()> {
        if let Some(config) = &self.config {
            config.validate()?;
        }
        Ok(())
    }

    fn reset(self) -> Self {
        Self::new()
    }
}

/// Quick builder macro for simple cases
#[macro_export]
macro_rules! quick_builder {
    ($name:ident for $target:ty {
        $(
            $field:ident: $field_type:ty
        ),* $(,)?
    }) => {
        #[derive(Debug, Clone, Default)]
        pub struct $name {
            $(
                $field: Option<$field_type>,
            )*
        }

        impl $name {
            pub fn new() -> Self {
                Self::default()
            }

            $(
                pub fn $field(mut self, value: $field_type) -> Self {
                    self.$field = Some(value);
                    self
                }
            )*
        }

        impl Builder<$target> for $name {
            fn build(self) -> Result<$target> {
                // NOTE: This is a template implementation. Real builders should
                // implement custom logic to construct the target type from the builder fields.
                // Example implementation for a struct with the same fields:
                // Ok($target {
                //     $(
                //         $field: self.$field.ok_or_else(|| {
                //             crate::errors::TrustformersError::invalid_input {
                //                 message: format!("Missing required field: {}", stringify!($field)),
                //                 details: std::collections::HashMap::new(),
                //             }
                //         })?,
                //     )*
                // })

                // For now, return a default instance if the target implements Default
                Ok(<$target>::default())
            }

            fn reset(self) -> Self {
                Self::default()
            }
        }
    };
}

/// Builder validation error
#[derive(Debug, thiserror::Error)]
pub enum BuilderError {
    #[error("Required field missing: {field}")]
    MissingField { field: String },
    #[error("Invalid value for field {field}: {reason}")]
    InvalidValue { field: String, reason: String },
    #[error("Builder validation failed: {reason}")]
    ValidationFailed { reason: String },
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Result type for builder operations
pub type BuilderResult<T> = std::result::Result<T, BuilderError>;

/// Trait for objects that can be serialized as configuration
pub trait ConfigSerializable {
    /// Serialize to JSON string
    fn to_json(&self) -> Result<String>;

    /// Deserialize from JSON string
    fn from_json(json: &str) -> Result<Self>
    where
        Self: Sized;

    /// Save to file
    fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from file
    fn load_from_file(path: &std::path::Path) -> Result<Self>
    where
        Self: Sized,
    {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }
}

/// Default implementation for types that implement Serialize + DeserializeOwned
impl<T> ConfigSerializable for T
where
    T: Serialize + for<'de> Deserialize<'de>,
{
    fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

/// Example concrete builder implementations demonstrating best practices
///
/// Example model configuration that can be built
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub model_type: String,
    pub max_length: usize,
    pub batch_size: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl StandardConfig for ModelConfig {
    fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(crate::errors::TrustformersError::invalid_input(
                "Model name cannot be empty".to_string(),
            ));
        }
        if self.max_length == 0 {
            return Err(crate::errors::TrustformersError::invalid_input(
                "Max length must be greater than 0".to_string(),
            ));
        }
        if self.temperature < 0.0 || self.temperature > 2.0 {
            return Err(crate::errors::TrustformersError::invalid_input(
                "Temperature must be between 0.0 and 2.0".to_string(),
            ));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(crate::errors::TrustformersError::invalid_input(
                "Top-p must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Concrete builder for ModelConfig with proper validation
#[derive(Debug, Clone, Default)]
pub struct ModelConfigBuilder {
    name: Option<String>,
    model_type: Option<String>,
    max_length: Option<usize>,
    batch_size: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

impl ModelConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn model_type(mut self, model_type: impl Into<String>) -> Self {
        self.model_type = Some(model_type.into());
        self
    }

    pub fn max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
}

impl Builder<ModelConfig> for ModelConfigBuilder {
    fn build(self) -> Result<ModelConfig> {
        let config = ModelConfig {
            name: self.name.unwrap_or_default(),
            model_type: self.model_type.unwrap_or_else(|| "transformer".to_string()),
            max_length: self.max_length.unwrap_or(2048),
            batch_size: self.batch_size.unwrap_or(1),
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(1.0),
        };

        // Validate the built configuration
        config.validate()?;
        Ok(config)
    }

    fn reset(self) -> Self {
        Self::default()
    }
}

impl Buildable for ModelConfig {
    type Builder = ModelConfigBuilder;

    fn builder() -> Self::Builder {
        ModelConfigBuilder::new()
    }
}

/// Example training configuration
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub gradient_clipping: f64,
}

impl StandardConfig for TrainingConfig {
    fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 {
            return Err(crate::errors::TrustformersError::invalid_input(
                "Learning rate must be positive".to_string(),
            ));
        }
        if self.epochs == 0 {
            return Err(crate::errors::TrustformersError::invalid_input(
                "Epochs must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Example concrete training config builder without using the macro
#[derive(Debug, Clone, Default)]
pub struct TrainingConfigBuilder {
    learning_rate: Option<f64>,
    epochs: Option<usize>,
    warmup_steps: Option<usize>,
    weight_decay: Option<f64>,
    gradient_clipping: Option<f64>,
}

impl TrainingConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = Some(epochs);
        self
    }

    pub fn warmup_steps(mut self, warmup_steps: usize) -> Self {
        self.warmup_steps = Some(warmup_steps);
        self
    }

    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = Some(weight_decay);
        self
    }

    pub fn gradient_clipping(mut self, gradient_clipping: f64) -> Self {
        self.gradient_clipping = Some(gradient_clipping);
        self
    }
}

// Implement proper build method for TrainingConfigBuilder
impl Builder<TrainingConfig> for TrainingConfigBuilder {
    fn build(self) -> Result<TrainingConfig> {
        let config = TrainingConfig {
            learning_rate: self.learning_rate.unwrap_or(1e-4),
            epochs: self.epochs.unwrap_or(10),
            warmup_steps: self.warmup_steps.unwrap_or(1000),
            weight_decay: self.weight_decay.unwrap_or(0.01),
            gradient_clipping: self.gradient_clipping.unwrap_or(1.0),
        };

        config.validate()?;
        Ok(config)
    }

    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Default, PartialEq)]
    struct TestObject {
        name: String,
        value: i32,
        enabled: bool,
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    struct TestConfig {
        timeout: u64,
        retries: u32,
    }

    impl StandardConfig for TestConfig {}

    #[test]
    fn test_standard_builder() {
        let mut builder: StandardBuilder<TestObject, BuilderIncomplete> = StandardBuilder::new();
        builder.data_mut().name = "test".to_string();
        builder.data_mut().value = 42;
        builder.data_mut().enabled = true;

        let obj = builder.complete().build().unwrap();
        assert_eq!(obj.name, "test");
        assert_eq!(obj.value, 42);
        assert!(obj.enabled);
    }

    #[test]
    fn test_validated_builder() {
        let builder = ValidatedBuilder::new().add_validator(|obj: &TestObject| {
            if obj.name.is_empty() {
                return Err(anyhow::anyhow!("Name cannot be empty").into());
            }
            Ok(())
        });

        // This should fail validation
        let result = builder.build();
        assert!(result.is_err());

        // This should succeed
        let mut builder = ValidatedBuilder::new().add_validator(|obj: &TestObject| {
            if obj.name.is_empty() {
                return Err(anyhow::anyhow!("Name cannot be empty").into());
            }
            Ok(())
        });

        builder.data_mut().name = "test".to_string();
        let result = builder.build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = TestConfig {
            timeout: 5000,
            retries: 3,
        };

        let builder = ConfigBuilderImpl::new()
            .config(config)
            .name("test_config")
            .description("A test configuration")
            .tag("test")
            .target(TestObject::default());

        let result = builder.build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = TestConfig {
            timeout: 5000,
            retries: 3,
        };

        let json = config.to_json().unwrap();
        let deserialized = TestConfig::from_json(&json).unwrap();

        assert_eq!(config.timeout, deserialized.timeout);
        assert_eq!(config.retries, deserialized.retries);
    }

    // Example of using the quick_builder macro
    quick_builder!(TestObjectBuilder for TestObject {
        name: String,
        value: i32,
        enabled: bool
    });

    #[test]
    fn test_quick_builder_creation() {
        let builder = TestObjectBuilder::new().name("test".to_string()).value(42).enabled(true);

        // Note: build() would need to be implemented for the specific type
        // This just tests the builder pattern creation
        assert!(builder.name.is_some());
        assert!(builder.value.is_some());
        assert!(builder.enabled.is_some());
    }

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::builder()
            .name("test-model")
            .model_type("gpt")
            .max_length(1024)
            .batch_size(4)
            .temperature(0.7)
            .top_p(0.9)
            .build()
            .unwrap();

        assert_eq!(config.name, "test-model");
        assert_eq!(config.model_type, "gpt");
        assert_eq!(config.max_length, 1024);
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.9);
    }

    #[test]
    fn test_model_config_builder_validation() {
        // Test validation failure - invalid temperature
        let result = ModelConfig::builder()
            .name("test")
            .temperature(3.0) // Invalid: > 2.0
            .build();
        assert!(result.is_err());

        // Test validation failure - invalid top_p
        let result = ModelConfig::builder()
            .name("test")
            .top_p(1.5) // Invalid: > 1.0
            .build();
        assert!(result.is_err());

        // Test validation success
        let result = ModelConfig::builder().name("test").temperature(0.8).top_p(0.9).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_training_config_builder() {
        let config = TrainingConfigBuilder::new()
            .learning_rate(1e-3)
            .epochs(5)
            .warmup_steps(500)
            .weight_decay(0.001)
            .gradient_clipping(0.5)
            .build()
            .unwrap();

        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.epochs, 5);
        assert_eq!(config.warmup_steps, 500);
        assert_eq!(config.weight_decay, 0.001);
        assert_eq!(config.gradient_clipping, 0.5);
    }

    #[test]
    fn test_training_config_builder_defaults() {
        let config = TrainingConfigBuilder::new().build().unwrap();

        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.epochs, 10);
        assert_eq!(config.warmup_steps, 1000);
        assert_eq!(config.weight_decay, 0.01);
        assert_eq!(config.gradient_clipping, 1.0);
    }

    #[test]
    fn test_training_config_validation() {
        // Test validation failure - invalid learning rate
        let result = TrainingConfigBuilder::new()
            .learning_rate(-0.1) // Invalid: negative
            .build();
        assert!(result.is_err());

        // Test validation failure - zero epochs
        let result = TrainingConfigBuilder::new()
            .epochs(0) // Invalid: zero
            .build();
        assert!(result.is_err());
    }
}
