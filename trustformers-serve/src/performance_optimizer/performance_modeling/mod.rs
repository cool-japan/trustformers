//! Performance Modeling Module
//!
//! This module provides comprehensive machine learning-based performance modeling
//! capabilities for the TrustformeRS serving system. It includes:
//!
//! - Multiple ML model implementations (linear, polynomial, neural networks)
//! - Adaptive learning with concept drift detection and active learning
//! - Comprehensive model validation strategies
//! - Advanced feature engineering with extraction, transformation, and selection
//! - Complete training pipeline orchestration with monitoring
//! - High-performance prediction engine with caching and ensemble support
//!
//! # Architecture
//!
//! The module is organized into specialized sub-modules:
//!
//! - [`types`] - Core type definitions, traits, and configuration structures
//! - [`model_implementations`] - ML model implementations with mathematical algorithms
//! - [`adaptive_learning`] - Adaptive learning system with concept drift detection
//! - [`model_validation`] - Comprehensive validation strategies and metrics
//! - [`feature_engineering`] - Advanced feature engineering capabilities
//! - [`training_pipeline`] - Complete training pipeline orchestration
//! - [`prediction_engine`] - High-performance prediction engine with caching
//!
//! # Example Usage
//!
//! ```rust
//! use crate::performance_optimizer::performance_modeling::{
//!     PerformanceModelingManager,
//!     PredictionRequest,
//!     ModelTypeConfig,
//! };
//!
//! // Create a performance modeling manager
//! let manager = PerformanceModelingManager::new().await?;
//!
//! // Train a model
//! let model = manager.train_model(&training_data, &ModelTypeConfig::LinearRegression).await?;
//!
//! // Make predictions
//! let request = PredictionRequest { /* ... */ };
//! let prediction = manager.predict(&request).await?;
//! ```

pub mod adaptive_learning;
pub mod feature_engineering;
pub mod model_implementations;
pub mod model_validation;
pub mod prediction_engine;
pub mod training_pipeline;
pub mod types;

// Import required types from parent modules
use crate::performance_optimizer::types::PerformanceDataPoint;

// Re-export all public types for backward compatibility
pub use types::{
    DensityEstimation,
    // Statistical types
    DistributionType,
    GoodnessOfFit,
    ModelAccuracyMetrics,
    // Configuration types
    ModelConfig,
    ModelTypeConfig,
    NormalityTest,
    // Core types that actually exist in types.rs
    PerformancePrediction,
    PerformancePredictor,
    PredictionConfig,
    // Prediction types
    PredictionRequest,
    PredictionRequestBatch,
    ValidationResult,
};

pub use model_implementations::{
    ExponentialModel, LinearRegressionModel, ModelFactory, PolynomialRegressionModel,
};

pub use adaptive_learning::{AdaptiveLearningOrchestrator, ConceptDriftDetector};

pub use model_validation::{
    BootstrapValidation, ComprehensiveValidationResult, CrossValidation,
    ModelValidationOrchestrator, TimeSeriesValidation,
};

pub use feature_engineering::{
    EngineeredFeatures, FeatureEngineeringOrchestrator, FeatureExtractor, FeatureSelector,
    FeatureTransformer,
};

pub use training_pipeline::{
    DataPreparationConfig, HyperparameterTuner, HyperparameterTuningConfig,
    PipelineMonitoringConfig, TrainingMonitor, TrainingPipelineConfig,
    TrainingPipelineOrchestrator,
};

pub use prediction_engine::{PredictionEngine, UncertaintyEstimator};

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main orchestrator for the entire performance modeling system.
///
/// This struct coordinates all the specialized components to provide
/// a unified interface for performance modeling operations.
#[derive(Debug)]
pub struct PerformanceModelingManager {
    /// Model training and management
    training_pipeline: Arc<TrainingPipelineOrchestrator>,

    /// Prediction engine for serving predictions
    prediction_engine: Arc<PredictionEngine>,

    /// Feature engineering for data preprocessing
    feature_engineering: Arc<FeatureEngineeringOrchestrator>,

    /// Model validation for quality assurance
    model_validation: Arc<ModelValidationOrchestrator>,

    /// Adaptive learning for continuous improvement
    adaptive_learning: Arc<AdaptiveLearningOrchestrator>,

    /// Current active models
    active_models: Arc<RwLock<Vec<Arc<dyn PerformancePredictor>>>>,

    /// System configuration
    config: ModelConfig,
}

impl PerformanceModelingManager {
    /// Creates a new PerformanceModelingManager with default configuration.
    pub async fn new() -> Result<Self> {
        Self::with_config(ModelConfig::default()).await
    }

    /// Creates a new PerformanceModelingManager with custom configuration.
    pub async fn with_config(config: ModelConfig) -> Result<Self> {
        // Create TrainingPipelineConfig from ModelConfig
        let training_pipeline_config = TrainingPipelineConfig {
            data_preparation: DataPreparationConfig::default(),
            model_training: config.training.clone(),
            feature_engineering: config.feature_engineering.clone(),
            validation: config.validation.clone(),
            hyperparameter_tuning: HyperparameterTuningConfig::default(),
            monitoring: PipelineMonitoringConfig::default(),
        };
        let training_pipeline =
            Arc::new(TrainingPipelineOrchestrator::new(training_pipeline_config)?);

        // Convert PredictionConfig to PredictionEngineConfig
        let prediction_engine_config = prediction_engine::PredictionEngineConfig {
            enable_caching: config.prediction.enable_caching,
            cache_ttl_seconds: 300,
            max_cache_size: config.prediction.cache_size,
            enable_ensemble: true,
            ensemble_strategy: prediction_engine::EnsembleStrategy::WeightedAverage,
            enable_uncertainty: true,
            prediction_timeout: std::time::Duration::from_secs(30),
            max_batch_size: 1000,
            enable_validation: true,
        };
        let prediction_engine = Arc::new(PredictionEngine::new(prediction_engine_config));

        let feature_engineering = Arc::new(FeatureEngineeringOrchestrator::new(
            config.feature_engineering.clone(),
        ));

        let model_validation =
            Arc::new(ModelValidationOrchestrator::new(config.validation.clone()));

        let adaptive_learning = Arc::new(AdaptiveLearningOrchestrator::new(
            config.adaptive_learning.clone(),
        ));

        Ok(Self {
            training_pipeline,
            prediction_engine,
            feature_engineering,
            model_validation,
            adaptive_learning,
            active_models: Arc::new(RwLock::new(Vec::new())),
            config,
        })
    }

    /// Trains a new performance model with the given data and configuration.
    pub async fn train_model(
        &self,
        training_data: &[PerformanceDataPoint],
        model_type: &ModelTypeConfig,
    ) -> Result<Arc<dyn PerformancePredictor>> {
        // Execute the full training pipeline
        let trained_model =
            self.training_pipeline.execute_pipeline(training_data, model_type).await?;

        // Validate the model
        let validation_result =
            self.model_validation.validate_model(&trained_model, training_data).await?;

        if validation_result.overall_score < self.config.validation.minimum_accuracy {
            return Err(anyhow::anyhow!(
                "Model validation failed: accuracy {} below minimum {}",
                validation_result.overall_score,
                self.config.validation.minimum_accuracy
            ));
        }

        // TODO: Add trained model to active_models - requires refactoring to use Arc instead of Box
        // For now, return a default model
        // This is a temporary placeholder to fix compilation
        Ok(Arc::new(trained_model))
    }

    /// Makes a performance prediction using the best available model.
    pub async fn predict(&self, request: &PredictionRequest) -> Result<PerformancePrediction> {
        self.prediction_engine.predict(request).await
    }

    /// Makes batch predictions for multiple requests.
    pub async fn predict_batch(
        &self,
        requests: &PredictionRequestBatch,
    ) -> Result<Vec<PerformancePrediction>> {
        // Extract requests from batch and pass to prediction engine
        let result = self.prediction_engine.predict_batch(&requests.requests).await?;
        Ok(result.predictions)
    }

    /// Engineers features from raw performance data.
    pub async fn engineer_features(
        &self,
        data_points: &[PerformanceDataPoint],
    ) -> Result<EngineeredFeatures> {
        self.feature_engineering.engineer_features(data_points).await
    }

    /// Validates a model using comprehensive validation strategies.
    pub async fn validate_model(
        &self,
        model: &dyn PerformancePredictor,
        test_data: &[PerformanceDataPoint],
    ) -> Result<ComprehensiveValidationResult> {
        self.model_validation.validate_model(model, test_data).await
    }

    /// Updates models using adaptive learning based on new data.
    pub async fn adapt_models(&self, new_data: &[PerformanceDataPoint]) -> Result<()> {
        // Convert slice to Vec and discard learning updates
        let _ = self.adaptive_learning.process_new_data(new_data.to_vec()).await?;
        Ok(())
    }

    /// Gets accuracy metrics for all active models.
    pub async fn get_model_metrics(&self) -> Result<Vec<ModelAccuracyMetrics>> {
        let models = self.active_models.read().await;
        let mut metrics = Vec::new();

        for model in models.iter() {
            metrics.push(model.get_accuracy());
        }

        Ok(metrics)
    }

    /// Gets the current configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Updates the configuration.
    pub async fn update_config(&mut self, new_config: ModelConfig) -> Result<()> {
        self.config = new_config;

        // Recreate components with new configuration
        let training_pipeline_config = TrainingPipelineConfig {
            data_preparation: DataPreparationConfig::default(),
            model_training: self.config.training.clone(),
            feature_engineering: self.config.feature_engineering.clone(),
            validation: self.config.validation.clone(),
            hyperparameter_tuning: HyperparameterTuningConfig::default(),
            monitoring: PipelineMonitoringConfig::default(),
        };
        let training_pipeline =
            Arc::new(TrainingPipelineOrchestrator::new(training_pipeline_config)?);

        // Convert PredictionConfig to PredictionEngineConfig
        let prediction_engine_config = prediction_engine::PredictionEngineConfig {
            enable_caching: self.config.prediction.enable_caching,
            cache_ttl_seconds: 300,
            max_cache_size: self.config.prediction.cache_size,
            enable_ensemble: true,
            ensemble_strategy: prediction_engine::EnsembleStrategy::WeightedAverage,
            enable_uncertainty: true,
            prediction_timeout: std::time::Duration::from_secs(30),
            max_batch_size: 1000,
            enable_validation: true,
        };
        let prediction_engine = Arc::new(PredictionEngine::new(prediction_engine_config));

        let feature_engineering = Arc::new(FeatureEngineeringOrchestrator::new(
            self.config.feature_engineering.clone(),
        ));

        let model_validation = Arc::new(ModelValidationOrchestrator::new(
            self.config.validation.clone(),
        ));

        let adaptive_learning = Arc::new(AdaptiveLearningOrchestrator::new(
            self.config.adaptive_learning.clone(),
        ));

        self.training_pipeline = training_pipeline;
        self.prediction_engine = prediction_engine;
        self.feature_engineering = feature_engineering;
        self.model_validation = model_validation;
        self.adaptive_learning = adaptive_learning;

        Ok(())
    }
}

/// Default implementation for PerformanceModelingManager
impl Default for PerformanceModelingManager {
    fn default() -> Self {
        futures::executor::block_on(Self::new())
            .expect("Failed to create default PerformanceModelingManager")
    }
}

/// Convenience functions for common operations
impl PerformanceModelingManager {
    /// Quick prediction with default linear regression model
    pub async fn quick_predict(
        data_points: &[PerformanceDataPoint],
        request: &PredictionRequest,
    ) -> Result<PerformancePrediction> {
        let manager = Self::new().await?;

        // Train a quick linear regression model
        let model_config = ModelTypeConfig::linear_regression();
        let model = manager.train_model(data_points, &model_config).await?;

        // Make prediction
        model.predict(request)
    }

    /// Batch training of multiple model types
    pub async fn train_ensemble(
        &self,
        training_data: &[PerformanceDataPoint],
        model_types: &[ModelTypeConfig],
    ) -> Result<Vec<Arc<dyn PerformancePredictor>>> {
        let mut models = Vec::new();

        for model_type in model_types {
            match self.train_model(training_data, model_type).await {
                Ok(model) => models.push(model),
                Err(e) => {
                    eprintln!("Failed to train {:?}: {}", model_type, e);
                    continue;
                },
            }
        }

        Ok(models)
    }
}
