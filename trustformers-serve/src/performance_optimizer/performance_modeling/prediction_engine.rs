//! Prediction Engine for Performance Models
//!
//! This module provides a comprehensive prediction engine that handles
//! performance predictions, batch predictions, ensemble predictions,
//! uncertainty estimation, and prediction caching for optimal performance.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    cmp::Ordering,
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};

use super::types::*;

// =============================================================================
// PREDICTION ENGINE
// =============================================================================

/// High-performance prediction engine with caching and ensemble support
#[derive(Debug)]
pub struct PredictionEngine {
    /// Prediction configuration
    config: Arc<RwLock<PredictionEngineConfig>>,
    /// Model registry
    model_registry: Arc<RwLock<PredictionModelRegistry>>,
    /// Prediction cache
    prediction_cache: Arc<RwLock<PredictionCache>>,
    /// Ensemble coordinator
    ensemble_coordinator: Arc<EnsembleCoordinator>,
    /// Uncertainty estimator
    uncertainty_estimator: Arc<UncertaintyEstimator>,
    /// Performance tracker
    performance_tracker: Arc<Mutex<PredictionPerformanceTracker>>,
}

/// Prediction engine configuration
#[derive(Debug, Clone)]
pub struct PredictionEngineConfig {
    /// Enable prediction caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Enable ensemble predictions
    pub enable_ensemble: bool,
    /// Ensemble strategy
    pub ensemble_strategy: EnsembleStrategy,
    /// Enable uncertainty estimation
    pub enable_uncertainty: bool,
    /// Prediction timeout
    pub prediction_timeout: Duration,
    /// Batch prediction size limit
    pub max_batch_size: usize,
    /// Enable prediction validation
    pub enable_validation: bool,
}

impl Default for PredictionEngineConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_ttl_seconds: 300, // 5 minutes
            max_cache_size: 10000,
            enable_ensemble: true,
            ensemble_strategy: EnsembleStrategy::WeightedAverage,
            enable_uncertainty: true,
            prediction_timeout: Duration::from_secs(30),
            max_batch_size: 1000,
            enable_validation: true,
        }
    }
}

/// Ensemble prediction strategies
#[derive(Debug, Clone)]
pub enum EnsembleStrategy {
    /// Simple averaging
    SimpleAverage,
    /// Weighted average based on model accuracy
    WeightedAverage,
    /// Use best performing model only
    BestModel,
    /// Stacking with meta-learner
    Stacking,
    /// Voting-based ensemble
    Voting,
    /// Adaptive weighting based on input characteristics
    AdaptiveWeighting,
}

impl PredictionEngine {
    /// Create new prediction engine
    pub fn new(config: PredictionEngineConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config.clone())),
            model_registry: Arc::new(RwLock::new(PredictionModelRegistry::new())),
            prediction_cache: Arc::new(RwLock::new(PredictionCache::new(config.max_cache_size))),
            ensemble_coordinator: Arc::new(EnsembleCoordinator::new(config.ensemble_strategy)),
            uncertainty_estimator: Arc::new(UncertaintyEstimator::new()),
            performance_tracker: Arc::new(Mutex::new(PredictionPerformanceTracker::new())),
        }
    }

    /// Register a model for predictions
    pub fn register_model(
        &self,
        model_id: String,
        model: Box<dyn PerformancePredictor>,
        weight: f32,
    ) -> Result<()> {
        let mut registry = self.model_registry.write();
        registry.register_model(model_id, model, weight)
    }

    /// Make a single prediction
    pub async fn predict(&self, request: &PredictionRequest) -> Result<PerformancePrediction> {
        let start_time = std::time::Instant::now();
        let config = self.config.read();

        // Check cache first
        if config.enable_caching {
            let cache_key = self.generate_cache_key(request);
            if let Some(cached_prediction) = self.prediction_cache.write().get(&cache_key) {
                tracing::debug!("Returning cached prediction for key: {}", cache_key);

                // Update performance tracking
                {
                    let mut tracker = self.performance_tracker.lock();
                    tracker.record_cache_hit(start_time.elapsed());
                }

                return Ok(cached_prediction.prediction.clone());
            }
        }

        // Generate prediction
        let prediction = if config.enable_ensemble {
            self.predict_with_ensemble(request).await?
        } else {
            self.predict_with_single_model(request).await?
        };

        // Add uncertainty estimation if enabled
        let final_prediction = if config.enable_uncertainty {
            self.add_uncertainty_estimation(prediction, request).await?
        } else {
            prediction
        };

        // Cache the prediction
        if config.enable_caching {
            let cache_key = self.generate_cache_key(request);
            let mut cache = self.prediction_cache.write();
            cache.insert(
                cache_key,
                CachedPrediction {
                    prediction: final_prediction.clone(),
                    cached_at: Utc::now(),
                    ttl_seconds: config.cache_ttl_seconds,
                },
            );
        }

        // Update performance tracking
        {
            let mut tracker = self.performance_tracker.lock();
            tracker.record_prediction(start_time.elapsed(), final_prediction.confidence);
        }

        Ok(final_prediction)
    }

    /// Make batch predictions
    pub async fn predict_batch(
        &self,
        requests: &[PredictionRequest],
    ) -> Result<BatchPredictionResult> {
        let start_time = std::time::Instant::now();
        let config = self.config.read();

        if requests.len() > config.max_batch_size {
            return Err(anyhow!(
                "Batch size {} exceeds limit {}",
                requests.len(),
                config.max_batch_size
            ));
        }

        tracing::info!(
            "Processing batch prediction with {} requests",
            requests.len()
        );

        let mut predictions = Vec::with_capacity(requests.len());
        let mut cache_hits = 0;
        let mut cache_misses = 0;

        // Process requests in parallel (simplified - in practice would use actual parallelization)
        for request in requests {
            match self.predict(request).await {
                Ok(prediction) => {
                    predictions.push(prediction);

                    // Check if this was a cache hit
                    if config.enable_caching {
                        let cache_key = self.generate_cache_key(request);
                        if self.prediction_cache.read().contains_key(&cache_key) {
                            cache_hits += 1;
                        } else {
                            cache_misses += 1;
                        }
                    }
                },
                Err(e) => {
                    tracing::warn!("Batch prediction failed for request: {}", e);
                    // Continue with other predictions
                },
            }
        }

        // Calculate batch statistics
        let batch_statistics = self.calculate_batch_statistics(&predictions);

        // Get ensemble info if applicable
        let ensemble_info =
            if config.enable_ensemble { Some(self.get_ensemble_info()) } else { None };

        let processing_time = start_time.elapsed();

        // Update performance tracking
        {
            let mut tracker = self.performance_tracker.lock();
            tracker.record_batch_prediction(
                processing_time,
                predictions.len(),
                cache_hits,
                cache_misses,
            );
        }

        Ok(BatchPredictionResult {
            predictions,
            batch_statistics,
            processing_time,
            ensemble_info,
        })
    }

    /// Predict with ensemble of models
    async fn predict_with_ensemble(
        &self,
        request: &PredictionRequest,
    ) -> Result<PerformancePrediction> {
        let registry = self.model_registry.read();
        let models = registry.get_all_models();

        if models.is_empty() {
            return Err(anyhow!("No models registered for prediction"));
        }

        // Get predictions from all models
        let mut model_predictions = Vec::new();
        for (model_id, model_info) in models {
            match model_info.model.predict(request) {
                Ok(prediction) => {
                    model_predictions.push(WeightedPrediction {
                        prediction,
                        weight: model_info.weight,
                        model_id: model_id.clone(),
                    });
                },
                Err(e) => {
                    tracing::warn!("Model {} failed to predict: {}", model_id, e);
                    // Continue with other models
                },
            }
        }

        if model_predictions.is_empty() {
            return Err(anyhow!("All models failed to generate predictions"));
        }

        // Combine predictions using ensemble coordinator
        self.ensemble_coordinator.combine_predictions(model_predictions)
    }

    /// Predict with single best model
    async fn predict_with_single_model(
        &self,
        request: &PredictionRequest,
    ) -> Result<PerformancePrediction> {
        let registry = self.model_registry.read();
        let best_model = registry
            .get_best_model()
            .ok_or_else(|| anyhow!("No models registered for prediction"))?;

        best_model.model.predict(request)
    }

    /// Add uncertainty estimation to prediction
    async fn add_uncertainty_estimation(
        &self,
        mut prediction: PerformancePrediction,
        request: &PredictionRequest,
    ) -> Result<PerformancePrediction> {
        let uncertainty_info =
            self.uncertainty_estimator.estimate_uncertainty(&prediction, request).await?;

        // Update prediction with uncertainty information
        prediction.confidence *= uncertainty_info.confidence_adjustment;
        prediction.uncertainty_bounds = (
            prediction.uncertainty_bounds.0 - uncertainty_info.additional_uncertainty,
            prediction.uncertainty_bounds.1 + uncertainty_info.additional_uncertainty,
        );

        Ok(prediction)
    }

    /// Generate cache key for prediction request
    fn generate_cache_key(&self, request: &PredictionRequest) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash parallelism levels
        for &level in &request.parallelism_levels {
            level.hash(&mut hasher);
        }

        // Hash system state (simplified)
        request.system_state.available_cores.hash(&mut hasher);
        request.system_state.available_memory_mb.hash(&mut hasher);
        ((request.system_state.load_average * 1000.0) as u32).hash(&mut hasher);

        // Hash test characteristics (simplified)
        (request.test_characteristics.average_duration.as_millis() as u64).hash(&mut hasher);
        ((request.test_characteristics.resource_intensity.cpu_intensity * 1000.0) as u32)
            .hash(&mut hasher);

        format!("pred_{:x}", hasher.finish())
    }

    /// Calculate batch prediction statistics
    fn calculate_batch_statistics(&self, predictions: &[PerformancePrediction]) -> BatchStatistics {
        if predictions.is_empty() {
            return BatchStatistics {
                average_confidence: 0.0,
                prediction_variance: 0.0,
                optimal_parallelism: 1,
                estimated_performance_gain: 0.0,
            };
        }

        let average_confidence =
            predictions.iter().map(|p| p.confidence).sum::<f32>() / predictions.len() as f32;

        let throughputs: Vec<f64> = predictions.iter().map(|p| p.throughput).collect();

        let mean_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let prediction_variance =
            throughputs.iter().map(|&t| (t - mean_throughput).powi(2)).sum::<f64>()
                / throughputs.len() as f64;

        // Find optimal parallelism (simplified)
        let optimal_parallelism = predictions.iter()
            .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap_or(std::cmp::Ordering::Equal))
            .map(|_| 4) // Simplified
            .unwrap_or(1);

        let max_throughput = throughputs.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_throughput = throughputs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let estimated_performance_gain = if min_throughput > 0.0 {
            ((max_throughput - min_throughput) / min_throughput) as f32
        } else {
            0.0
        };

        BatchStatistics {
            average_confidence,
            prediction_variance: prediction_variance as f32,
            optimal_parallelism,
            estimated_performance_gain,
        }
    }

    /// Get ensemble information
    fn get_ensemble_info(&self) -> EnsembleInfo {
        let registry = self.model_registry.read();
        let models = registry.get_all_models();

        let mut model_weights = HashMap::new();
        let mut total_weight = 0.0f32;

        for (model_id, model_info) in models {
            model_weights.insert(model_id.clone(), model_info.weight);
            total_weight += model_info.weight;
        }

        // Normalize weights
        if total_weight > 0.0 {
            for weight in model_weights.values_mut() {
                *weight /= total_weight;
            }
        }

        let diversity_score = self.calculate_model_diversity(&model_weights);
        let consensus_level = self.calculate_consensus_level(&model_weights);

        // Convert EnsembleStrategy to EnsembleMethod
        let ensemble_method = match &self.config.read().ensemble_strategy {
            EnsembleStrategy::SimpleAverage => EnsembleMethod::SimpleAverage,
            EnsembleStrategy::WeightedAverage => EnsembleMethod::WeightedAverage,
            EnsembleStrategy::Stacking => EnsembleMethod::Stacking,
            EnsembleStrategy::BestModel => EnsembleMethod::WeightedAverage, // Map BestModel to WeightedAverage
            EnsembleStrategy::Voting => EnsembleMethod::WeightedAverage, // Map Voting to WeightedAverage
            EnsembleStrategy::AdaptiveWeighting => EnsembleMethod::WeightedAverage, // Map AdaptiveWeighting to WeightedAverage
        };

        EnsembleInfo {
            model_weights,
            ensemble_method,
            diversity_score,
            consensus_level,
        }
    }

    /// Calculate model diversity score
    fn calculate_model_diversity(&self, model_weights: &HashMap<String, f32>) -> f32 {
        if model_weights.len() < 2 {
            return 0.0;
        }

        // Simplified diversity calculation based on weight distribution
        let weights: Vec<f32> = model_weights.values().cloned().collect();
        let mean_weight = weights.iter().sum::<f32>() / weights.len() as f32;
        let variance =
            weights.iter().map(|w| (w - mean_weight).powi(2)).sum::<f32>() / weights.len() as f32;

        variance.sqrt() // Higher variance indicates more diversity
    }

    /// Calculate consensus level among models
    fn calculate_consensus_level(&self, model_weights: &HashMap<String, f32>) -> f32 {
        if model_weights.is_empty() {
            return 0.0;
        }

        // Simplified consensus calculation
        let max_weight = model_weights.values().fold(0.0f32, |a, &b| a.max(b));
        1.0 - max_weight // Lower max weight indicates higher consensus
    }

    /// Get prediction statistics
    pub fn get_prediction_statistics(&self) -> PredictionStatistics {
        let tracker = self.performance_tracker.lock();
        let cache = self.prediction_cache.read();

        PredictionStatistics {
            total_predictions: tracker.total_predictions,
            cache_hit_rate: tracker.cache_hit_rate(),
            average_prediction_time: tracker.average_prediction_time(),
            average_confidence: tracker.average_confidence(),
            cache_size: cache.size(),
            ensemble_predictions: tracker.ensemble_predictions,
            failed_predictions: tracker.failed_predictions,
        }
    }

    /// Clear prediction cache
    pub fn clear_cache(&self) {
        let mut cache = self.prediction_cache.write();
        cache.clear();
        tracing::info!("Prediction cache cleared");
    }

    /// Update model weights
    pub fn update_model_weights(&self, weight_updates: HashMap<String, f32>) -> Result<()> {
        let mut registry = self.model_registry.write();
        for (model_id, new_weight) in weight_updates {
            registry.update_model_weight(&model_id, new_weight)?;
        }
        Ok(())
    }
}

// =============================================================================
// PREDICTION MODEL REGISTRY
// =============================================================================

/// Registry for managing prediction models
#[derive(Debug)]
pub struct PredictionModelRegistry {
    /// Registered models
    models: HashMap<String, ModelInfo>,
    /// Performance tracking for model selection
    model_performance: HashMap<String, ModelPerformanceInfo>,
}

#[derive(Debug)]
pub struct ModelInfo {
    pub model: Box<dyn PerformancePredictor>,
    pub weight: f32,
    pub registered_at: DateTime<Utc>,
    pub prediction_count: u64,
}

#[derive(Debug, Clone)]
struct ModelPerformanceInfo {
    accuracy_score: f32,
    average_prediction_time: Duration,
    prediction_count: u64,
    last_updated: DateTime<Utc>,
}

impl Default for PredictionModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictionModelRegistry {
    /// Create new model registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            model_performance: HashMap::new(),
        }
    }

    /// Register a model
    pub fn register_model(
        &mut self,
        model_id: String,
        model: Box<dyn PerformancePredictor>,
        weight: f32,
    ) -> Result<()> {
        if !(0.0..=1.0).contains(&weight) {
            return Err(anyhow!("Model weight must be between 0.0 and 1.0"));
        }

        let model_info = ModelInfo {
            model,
            weight,
            registered_at: Utc::now(),
            prediction_count: 0,
        };

        self.models.insert(model_id.clone(), model_info);

        // Initialize performance tracking
        self.model_performance.insert(
            model_id,
            ModelPerformanceInfo {
                accuracy_score: 0.5, // Default
                average_prediction_time: Duration::from_millis(100),
                prediction_count: 0,
                last_updated: Utc::now(),
            },
        );

        Ok(())
    }

    /// Get all models
    pub fn get_all_models(&self) -> HashMap<String, &ModelInfo> {
        self.models.iter().map(|(k, v)| (k.clone(), v)).collect()
    }

    /// Get best performing model
    pub fn get_best_model(&self) -> Option<&ModelInfo> {
        self.models
            .iter()
            .max_by(|(id_a, info_a), (id_b, info_b)| {
                let score_a = self
                    .model_performance
                    .get(*id_a)
                    .map(|perf| perf.accuracy_score)
                    .unwrap_or(info_a.weight);
                let score_b = self
                    .model_performance
                    .get(*id_b)
                    .map(|perf| perf.accuracy_score)
                    .unwrap_or(info_b.weight);

                score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
            })
            .map(|(_, info)| info)
    }

    /// Update model weight
    pub fn update_model_weight(&mut self, model_id: &str, new_weight: f32) -> Result<()> {
        if let Some(model_info) = self.models.get_mut(model_id) {
            model_info.weight = new_weight;
            Ok(())
        } else {
            Err(anyhow!("Model {} not found", model_id))
        }
    }

    /// Update model performance
    pub fn update_model_performance(
        &mut self,
        model_id: &str,
        accuracy: f32,
        prediction_time: Duration,
    ) {
        if let Some(perf_info) = self.model_performance.get_mut(model_id) {
            perf_info.accuracy_score = accuracy;
            perf_info.average_prediction_time = prediction_time;
            perf_info.prediction_count += 1;
            perf_info.last_updated = Utc::now();
        }
    }

    /// Get model count
    pub fn model_count(&self) -> usize {
        self.models.len()
    }
}

// =============================================================================
// PREDICTION CACHE
// =============================================================================

/// High-performance prediction cache with TTL support
#[derive(Debug)]
pub struct PredictionCache {
    /// Cache entries
    cache: HashMap<String, CachedPrediction>,
    /// Access order for LRU eviction
    access_order: VecDeque<String>,
    /// Maximum cache size
    max_size: usize,
    /// Cache statistics
    stats: CacheStatistics,
}

#[derive(Debug, Clone)]
pub struct CachedPrediction {
    pub prediction: PerformancePrediction,
    pub cached_at: DateTime<Utc>,
    pub ttl_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub expired_entries: u64,
}

impl PredictionCache {
    /// Create new prediction cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: VecDeque::new(),
            max_size,
            stats: CacheStatistics {
                hits: 0,
                misses: 0,
                evictions: 0,
                expired_entries: 0,
            },
        }
    }

    /// Get cached prediction
    pub fn get(&mut self, key: &str) -> Option<CachedPrediction> {
        // Clean expired entries periodically
        if self.stats.hits % 100 == 0 {
            self.clean_expired();
        }

        if let Some(cached) = self.cache.get(key) {
            // Check if entry has expired
            let age = Utc::now() - cached.cached_at;
            if age.num_seconds() > cached.ttl_seconds as i64 {
                self.cache.remove(key);
                self.stats.expired_entries += 1;
                self.stats.misses += 1;
                return None;
            }

            // Update access order
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push_back(key.to_string());

            self.stats.hits += 1;
            Some(cached.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert prediction into cache
    pub fn insert(&mut self, key: String, prediction: CachedPrediction) {
        // Remove existing entry if present
        if self.cache.contains_key(&key) {
            if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                self.access_order.remove(pos);
            }
        } else if self.cache.len() >= self.max_size {
            // Evict least recently used entry
            if let Some(lru_key) = self.access_order.pop_front() {
                self.cache.remove(&lru_key);
                self.stats.evictions += 1;
            }
        }

        self.cache.insert(key.clone(), prediction);
        self.access_order.push_back(key);
    }

    /// Clean expired entries
    fn clean_expired(&mut self) {
        let now = Utc::now();
        let mut expired_keys = Vec::new();

        for (key, cached) in &self.cache {
            let age = now - cached.cached_at;
            if age.num_seconds() > cached.ttl_seconds as i64 {
                expired_keys.push(key.clone());
            }
        }

        for key in expired_keys {
            self.cache.remove(&key);
            if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                self.access_order.remove(pos);
            }
            self.stats.expired_entries += 1;
        }
    }

    /// Check if key exists in cache
    pub fn contains_key(&self, key: &str) -> bool {
        self.cache.contains_key(key)
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.cache.len()
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        self.stats.clone()
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f32 {
        let total = self.stats.hits + self.stats.misses;
        if total > 0 {
            self.stats.hits as f32 / total as f32
        } else {
            0.0
        }
    }
}

// =============================================================================
// ENSEMBLE COORDINATOR
// =============================================================================

/// Coordinates ensemble predictions from multiple models
#[derive(Debug)]
pub struct EnsembleCoordinator {
    /// Ensemble strategy
    strategy: EnsembleStrategy,
    /// Strategy-specific parameters
    parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct WeightedPrediction {
    pub prediction: PerformancePrediction,
    pub weight: f32,
    pub model_id: String,
}

impl EnsembleCoordinator {
    /// Create new ensemble coordinator
    pub fn new(strategy: EnsembleStrategy) -> Self {
        Self {
            strategy,
            parameters: HashMap::new(),
        }
    }

    /// Combine predictions from multiple models
    pub fn combine_predictions(
        &self,
        predictions: Vec<WeightedPrediction>,
    ) -> Result<PerformancePrediction> {
        if predictions.is_empty() {
            return Err(anyhow!("No predictions to combine"));
        }

        match &self.strategy {
            EnsembleStrategy::SimpleAverage => self.simple_average(predictions),
            EnsembleStrategy::WeightedAverage => self.weighted_average(predictions),
            EnsembleStrategy::BestModel => self.best_model(predictions),
            EnsembleStrategy::Voting => self.voting_ensemble(predictions),
            EnsembleStrategy::AdaptiveWeighting => self.adaptive_weighting(predictions),
            EnsembleStrategy::Stacking => self.stacking_ensemble(predictions),
        }
    }

    /// Simple average ensemble
    fn simple_average(
        &self,
        predictions: Vec<WeightedPrediction>,
    ) -> Result<PerformancePrediction> {
        let count = predictions.len() as f64;

        let avg_throughput =
            predictions.iter().map(|p| p.prediction.throughput).sum::<f64>() / count;

        let avg_confidence = predictions.iter().map(|p| p.prediction.confidence).sum::<f32>()
            / predictions.len() as f32;

        let avg_latency_ms =
            predictions.iter().map(|p| p.prediction.latency.as_millis() as f64).sum::<f64>()
                / count;

        // Combine feature importance (simplified)
        let mut combined_importance = HashMap::new();
        for pred in &predictions {
            for (feature, importance) in &pred.prediction.feature_importance {
                *combined_importance.entry(feature.clone()).or_insert(0.0) +=
                    importance / predictions.len() as f32;
            }
        }

        Ok(PerformancePrediction {
            throughput: avg_throughput,
            latency: Duration::from_millis(avg_latency_ms as u64),
            confidence: avg_confidence,
            uncertainty_bounds: self.calculate_combined_uncertainty_bounds(&predictions),
            model_name: "SimpleAverage".to_string(),
            feature_importance: combined_importance,
            predicted_at: Utc::now(),
        })
    }

    /// Weighted average ensemble
    fn weighted_average(
        &self,
        predictions: Vec<WeightedPrediction>,
    ) -> Result<PerformancePrediction> {
        let total_weight: f32 = predictions.iter().map(|p| p.weight).sum();

        if total_weight == 0.0 {
            return self.simple_average(predictions);
        }

        let weighted_throughput = predictions
            .iter()
            .map(|p| p.prediction.throughput * (p.weight / total_weight) as f64)
            .sum::<f64>();

        let weighted_confidence = predictions
            .iter()
            .map(|p| p.prediction.confidence * (p.weight / total_weight))
            .sum::<f32>();

        let weighted_latency_ms = predictions
            .iter()
            .map(|p| p.prediction.latency.as_millis() as f64 * (p.weight / total_weight) as f64)
            .sum::<f64>();

        // Weighted feature importance
        let mut combined_importance = HashMap::new();
        for pred in &predictions {
            let normalized_weight = pred.weight / total_weight;
            for (feature, importance) in &pred.prediction.feature_importance {
                *combined_importance.entry(feature.clone()).or_insert(0.0) +=
                    importance * normalized_weight;
            }
        }

        Ok(PerformancePrediction {
            throughput: weighted_throughput,
            latency: Duration::from_millis(weighted_latency_ms as u64),
            confidence: weighted_confidence,
            uncertainty_bounds: self.calculate_combined_uncertainty_bounds(&predictions),
            model_name: "WeightedAverage".to_string(),
            feature_importance: combined_importance,
            predicted_at: Utc::now(),
        })
    }

    /// Use best model prediction
    fn best_model(&self, predictions: Vec<WeightedPrediction>) -> Result<PerformancePrediction> {
        let best_prediction = predictions
            .into_iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| anyhow!("No predictions available"))?;

        Ok(best_prediction.prediction)
    }

    /// Voting ensemble (simplified)
    fn voting_ensemble(
        &self,
        predictions: Vec<WeightedPrediction>,
    ) -> Result<PerformancePrediction> {
        // For regression, voting is similar to averaging
        // In practice, this might involve discretizing predictions and voting
        self.weighted_average(predictions)
    }

    /// Adaptive weighting based on input characteristics
    fn adaptive_weighting(
        &self,
        mut predictions: Vec<WeightedPrediction>,
    ) -> Result<PerformancePrediction> {
        // Adjust weights based on prediction confidence and consistency
        let confidences: Vec<f32> = predictions.iter().map(|p| p.prediction.confidence).collect();
        let mean_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;

        for pred in &mut predictions {
            // Boost weight for predictions with higher confidence
            let confidence_boost = pred.prediction.confidence / mean_confidence.max(0.1);
            pred.weight *= confidence_boost;
        }

        self.weighted_average(predictions)
    }

    /// Stacking ensemble (simplified)
    fn stacking_ensemble(
        &self,
        predictions: Vec<WeightedPrediction>,
    ) -> Result<PerformancePrediction> {
        // Simplified stacking - in practice would use a meta-learner
        // For now, use adaptive weighting as a proxy
        self.adaptive_weighting(predictions)
    }

    /// Calculate combined uncertainty bounds
    fn calculate_combined_uncertainty_bounds(
        &self,
        predictions: &[WeightedPrediction],
    ) -> (f64, f64) {
        if predictions.is_empty() {
            return (0.0, 0.0);
        }

        let throughputs: Vec<f64> = predictions.iter().map(|p| p.prediction.throughput).collect();

        let min_throughput = throughputs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_throughput = throughputs.iter().fold(0.0f64, |a, &b| a.max(b));

        // Expand bounds slightly for ensemble uncertainty
        let range = max_throughput - min_throughput;
        let expansion = range * 0.1; // 10% expansion

        (min_throughput - expansion, max_throughput + expansion)
    }
}

// =============================================================================
// UNCERTAINTY ESTIMATOR
// =============================================================================

/// Estimates prediction uncertainty using various methods
#[derive(Debug)]
pub struct UncertaintyEstimator {
    /// Estimation methods
    methods: Vec<UncertaintyMethod>,
}

#[derive(Debug, Clone)]
pub enum UncertaintyMethod {
    /// Variance-based uncertainty
    Variance,
    /// Model disagreement
    ModelDisagreement,
    /// Confidence intervals
    ConfidenceIntervals,
    /// Bootstrap sampling
    Bootstrap,
}

#[derive(Debug, Clone)]
pub struct UncertaintyInfo {
    /// Confidence adjustment factor
    pub confidence_adjustment: f32,
    /// Additional uncertainty to add to bounds
    pub additional_uncertainty: f64,
    /// Uncertainty breakdown by method
    pub method_contributions: HashMap<String, f32>,
}

impl Default for UncertaintyEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl UncertaintyEstimator {
    /// Create new uncertainty estimator
    pub fn new() -> Self {
        Self {
            methods: vec![
                UncertaintyMethod::Variance,
                UncertaintyMethod::ModelDisagreement,
                UncertaintyMethod::ConfidenceIntervals,
            ],
        }
    }

    /// Estimate uncertainty for prediction
    pub async fn estimate_uncertainty(
        &self,
        prediction: &PerformancePrediction,
        _request: &PredictionRequest,
    ) -> Result<UncertaintyInfo> {
        let mut method_contributions = HashMap::new();
        let mut total_uncertainty = 0.0f32;

        for method in &self.methods {
            let uncertainty = match method {
                UncertaintyMethod::Variance => self.estimate_variance_uncertainty(prediction),
                UncertaintyMethod::ModelDisagreement => {
                    self.estimate_disagreement_uncertainty(prediction)
                },
                UncertaintyMethod::ConfidenceIntervals => {
                    self.estimate_confidence_interval_uncertainty(prediction)
                },
                UncertaintyMethod::Bootstrap => self.estimate_bootstrap_uncertainty(prediction),
            };

            let method_name = format!("{:?}", method);
            method_contributions.insert(method_name, uncertainty);
            total_uncertainty += uncertainty;
        }

        // Average uncertainty across methods
        let avg_uncertainty = total_uncertainty / self.methods.len() as f32;

        // Adjust confidence based on uncertainty
        let confidence_adjustment = (1.0 - avg_uncertainty * 0.5).max(0.1);

        // Calculate additional uncertainty for bounds
        let current_range = prediction.uncertainty_bounds.1 - prediction.uncertainty_bounds.0;
        let additional_uncertainty = current_range * avg_uncertainty as f64;

        Ok(UncertaintyInfo {
            confidence_adjustment,
            additional_uncertainty,
            method_contributions,
        })
    }

    /// Estimate variance-based uncertainty
    fn estimate_variance_uncertainty(&self, prediction: &PerformancePrediction) -> f32 {
        // Use the width of uncertainty bounds as proxy for variance
        let bound_width = prediction.uncertainty_bounds.1 - prediction.uncertainty_bounds.0;
        let relative_width = bound_width / prediction.throughput.max(1.0);
        (relative_width as f32).min(1.0)
    }

    /// Estimate model disagreement uncertainty
    fn estimate_disagreement_uncertainty(&self, prediction: &PerformancePrediction) -> f32 {
        // For single predictions, use confidence as proxy for agreement
        (1.0 - prediction.confidence).clamp(0.0, 1.0)
    }

    /// Estimate confidence interval uncertainty
    fn estimate_confidence_interval_uncertainty(&self, prediction: &PerformancePrediction) -> f32 {
        // Use inverse of confidence as uncertainty measure
        let uncertainty = 1.0 - prediction.confidence;
        uncertainty.clamp(0.0, 1.0)
    }

    /// Estimate bootstrap uncertainty
    fn estimate_bootstrap_uncertainty(&self, _prediction: &PerformancePrediction) -> f32 {
        // Simplified bootstrap uncertainty - would require actual resampling
        0.1 // Default uncertainty
    }
}

// =============================================================================
// PERFORMANCE TRACKING
// =============================================================================

/// Tracks prediction engine performance metrics
#[derive(Debug)]
pub struct PredictionPerformanceTracker {
    /// Total predictions made
    pub total_predictions: u64,
    /// Cache hits
    cache_hits: u64,
    /// Cache misses
    cache_misses: u64,
    /// Ensemble predictions
    pub ensemble_predictions: u64,
    /// Failed predictions
    pub failed_predictions: u64,
    /// Prediction times
    prediction_times: VecDeque<Duration>,
    /// Confidence scores
    confidence_scores: VecDeque<f32>,
    /// Batch prediction counts
    batch_predictions: u64,
    /// Last updated
    last_updated: DateTime<Utc>,
}

impl Default for PredictionPerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictionPerformanceTracker {
    /// Create new performance tracker
    pub fn new() -> Self {
        Self {
            total_predictions: 0,
            cache_hits: 0,
            cache_misses: 0,
            ensemble_predictions: 0,
            failed_predictions: 0,
            prediction_times: VecDeque::with_capacity(1000),
            confidence_scores: VecDeque::with_capacity(1000),
            batch_predictions: 0,
            last_updated: Utc::now(),
        }
    }

    /// Record a successful prediction
    pub fn record_prediction(&mut self, prediction_time: Duration, confidence: f32) {
        self.total_predictions += 1;

        self.prediction_times.push_back(prediction_time);
        if self.prediction_times.len() > 1000 {
            self.prediction_times.pop_front();
        }

        self.confidence_scores.push_back(confidence);
        if self.confidence_scores.len() > 1000 {
            self.confidence_scores.pop_front();
        }

        self.last_updated = Utc::now();
    }

    /// Record a cache hit
    pub fn record_cache_hit(&mut self, _lookup_time: Duration) {
        self.cache_hits += 1;
        self.last_updated = Utc::now();
    }

    /// Record a cache miss
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
        self.last_updated = Utc::now();
    }

    /// Record batch prediction
    pub fn record_batch_prediction(
        &mut self,
        _batch_time: Duration,
        batch_size: usize,
        cache_hits: usize,
        cache_misses: usize,
    ) {
        self.batch_predictions += 1;
        self.total_predictions += batch_size as u64;
        self.cache_hits += cache_hits as u64;
        self.cache_misses += cache_misses as u64;
        self.last_updated = Utc::now();
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f32 {
        let total_cache_requests = self.cache_hits + self.cache_misses;
        if total_cache_requests > 0 {
            self.cache_hits as f32 / total_cache_requests as f32
        } else {
            0.0
        }
    }

    /// Get average prediction time
    pub fn average_prediction_time(&self) -> Duration {
        if self.prediction_times.is_empty() {
            return Duration::from_secs(0);
        }

        let total_nanos: u128 = self.prediction_times.iter().map(|d| d.as_nanos()).sum();

        Duration::from_nanos((total_nanos / self.prediction_times.len() as u128) as u64)
    }

    /// Get average confidence
    pub fn average_confidence(&self) -> f32 {
        if self.confidence_scores.is_empty() {
            return 0.0;
        }

        self.confidence_scores.iter().sum::<f32>() / self.confidence_scores.len() as f32
    }
}

// =============================================================================
// STATISTICS AND REPORTING
// =============================================================================

/// Prediction engine statistics
#[derive(Debug, Clone)]
pub struct PredictionStatistics {
    /// Total predictions made
    pub total_predictions: u64,
    /// Cache hit rate
    pub cache_hit_rate: f32,
    /// Average prediction time
    pub average_prediction_time: Duration,
    /// Average confidence score
    pub average_confidence: f32,
    /// Current cache size
    pub cache_size: usize,
    /// Number of ensemble predictions
    pub ensemble_predictions: u64,
    /// Number of failed predictions
    pub failed_predictions: u64,
}
