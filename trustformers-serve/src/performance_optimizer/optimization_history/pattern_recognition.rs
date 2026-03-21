//! Pattern Recognition System
//!
//! This module provides advanced pattern recognition capabilities for optimization patterns,
//! including pattern detection algorithms, machine learning models for pattern learning,
//! and predictive capabilities for pattern occurrence. It enables identification of recurring
//! optimization patterns and proactive optimization opportunities.

use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, sync::Arc, time::Duration};

use super::types::*;
use crate::performance_optimizer::types::OptimizationEvent as OptEvent;

// =============================================================================
// PATTERN RECOGNITION SYSTEM
// =============================================================================

/// Advanced pattern recognition system for optimization patterns
///
/// Identifies recurring patterns, anomalies, and optimization opportunities
/// in historical performance data using machine learning and statistical methods.
pub struct PatternRecognitionSystem {
    /// Pattern detection algorithms
    pattern_detectors: Arc<Mutex<Vec<Box<dyn PatternDetector + Send + Sync>>>>,
    /// Recognized patterns cache
    pattern_cache: Arc<RwLock<HashMap<String, RecognizedPattern>>>,
    /// Pattern learning models
    learning_models: Arc<Mutex<Vec<Box<dyn PatternLearningModel + Send + Sync>>>>,
    /// Pattern recognition configuration
    config: Arc<RwLock<PatternRecognitionConfig>>,
}

impl PatternRecognitionSystem {
    /// Create new pattern recognition system
    pub fn new() -> Self {
        let mut system = Self {
            pattern_detectors: Arc::new(Mutex::new(Vec::new())),
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
            learning_models: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(PatternRecognitionConfig::default())),
        };

        // Initialize default detectors and models
        system.initialize_default_detectors();
        system.initialize_default_models();

        system
    }

    /// Create with custom configuration
    pub fn with_config(config: PatternRecognitionConfig) -> Self {
        let mut system = Self {
            pattern_detectors: Arc::new(Mutex::new(Vec::new())),
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
            learning_models: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(config)),
        };

        system.initialize_default_detectors();
        system.initialize_default_models();

        system
    }

    /// Recognize patterns in optimization events
    pub async fn recognize_patterns(&self, events: &[OptEvent]) -> Result<Vec<RecognizedPattern>> {
        let (min_pattern_length, enable_ml_learning) = {
            let config = self.config.read();
            (config.min_pattern_length, config.enable_ml_learning)
        };

        if events.len() < min_pattern_length {
            return Err(anyhow::anyhow!(
                "Insufficient events for pattern recognition: {} < {}",
                events.len(),
                min_pattern_length
            ));
        }

        let all_patterns = {
            let detectors = self.pattern_detectors.lock();
            let mut collected = Vec::new();

            for detector in detectors.iter() {
                if detector.is_applicable(events) {
                    match detector.detect_patterns(events) {
                        Ok(patterns) => collected.extend(patterns),
                        Err(e) => {
                            tracing::warn!("Pattern detector {} failed: {}", detector.name(), e);
                        },
                    }
                }
            }

            collected
        };

        // Filter and deduplicate patterns
        let filtered_patterns = self.filter_and_deduplicate_patterns(all_patterns)?;

        // Cache recognized patterns
        self.cache_patterns(&filtered_patterns).await;

        // Update learning models if enabled
        if enable_ml_learning {
            self.update_learning_models(&filtered_patterns).await?;
        }

        Ok(filtered_patterns)
    }

    /// Predict pattern occurrence
    pub async fn predict_pattern(
        &self,
        context: &PatternContext,
    ) -> Result<Vec<PatternPrediction>> {
        let models = self.learning_models.lock();
        let mut predictions = Vec::new();

        for model in models.iter() {
            match model.predict_pattern(context) {
                Ok(prediction) => predictions.push(prediction),
                Err(e) => {
                    tracing::warn!("Pattern prediction model {} failed: {}", model.name(), e);
                },
            }
        }

        // Filter predictions by confidence
        let config = self.config.read();
        let high_confidence_predictions: Vec<PatternPrediction> = predictions
            .into_iter()
            .filter(|p| p.confidence >= config.similarity_threshold)
            .collect();

        Ok(high_confidence_predictions)
    }

    /// Get patterns by type
    pub async fn get_patterns_by_type(&self, pattern_type: PatternType) -> Vec<RecognizedPattern> {
        let cache = self.pattern_cache.read();
        cache
            .values()
            .filter(|pattern| {
                std::mem::discriminant(&pattern.pattern_type)
                    == std::mem::discriminant(&pattern_type)
            })
            .cloned()
            .collect()
    }

    /// Get most frequent patterns
    pub async fn get_most_frequent_patterns(&self, limit: usize) -> Vec<RecognizedPattern> {
        let cache = self.pattern_cache.read();
        let mut patterns: Vec<RecognizedPattern> = cache.values().cloned().collect();
        patterns.sort_by(|a, b| {
            b.frequency.partial_cmp(&a.frequency).unwrap_or(std::cmp::Ordering::Equal)
        });
        patterns.into_iter().take(limit).collect()
    }

    /// Add pattern detector
    pub fn add_pattern_detector(&self, detector: Box<dyn PatternDetector + Send + Sync>) {
        let mut detectors = self.pattern_detectors.lock();
        detectors.push(detector);
    }

    /// Add learning model
    pub fn add_learning_model(&self, model: Box<dyn PatternLearningModel + Send + Sync>) {
        let mut models = self.learning_models.lock();
        models.push(model);
    }

    /// Update configuration
    pub fn update_config(&self, new_config: PatternRecognitionConfig) {
        let mut config = self.config.write();
        *config = new_config;
    }

    /// Clear pattern cache
    pub async fn clear_cache(&self) {
        let mut cache = self.pattern_cache.write();
        cache.clear();
    }

    /// Get pattern statistics
    pub fn get_pattern_statistics(&self) -> PatternStatistics {
        let cache = self.pattern_cache.read();

        let total_patterns = cache.len();
        let mut type_counts: HashMap<PatternType, usize> = HashMap::new();
        let mut total_effectiveness = 0.0f32;
        let mut high_confidence_count = 0;

        for pattern in cache.values() {
            *type_counts.entry(pattern.pattern_type.clone()).or_insert(0) += 1;
            total_effectiveness += pattern.effectiveness;
            if pattern.confidence >= 0.8 {
                high_confidence_count += 1;
            }
        }

        let average_effectiveness = if total_patterns > 0 {
            total_effectiveness / total_patterns as f32
        } else {
            0.0
        };

        PatternStatistics {
            total_patterns,
            pattern_type_distribution: type_counts,
            average_effectiveness,
            high_confidence_patterns: high_confidence_count,
            cache_memory_usage: cache.len() * std::mem::size_of::<RecognizedPattern>(),
        }
    }

    /// Initialize default pattern detectors
    fn initialize_default_detectors(&mut self) {
        let mut detectors = self.pattern_detectors.lock();

        detectors.push(Box::new(CyclicalPatternDetector::new()));
        detectors.push(Box::new(DegradationPatternDetector::new()));
        detectors.push(Box::new(ImprovementPatternDetector::new()));
        detectors.push(Box::new(OscillationPatternDetector::new()));
        detectors.push(Box::new(ThresholdPatternDetector::new()));
    }

    /// Initialize default learning models
    fn initialize_default_models(&mut self) {
        let mut models = self.learning_models.lock();

        models.push(Box::new(SimplePatternLearner::new()));
        models.push(Box::new(FrequencyBasedPredictor::new()));
        models.push(Box::new(ContextualPatternPredictor::new()));
    }

    /// Filter and deduplicate patterns
    fn filter_and_deduplicate_patterns(
        &self,
        patterns: Vec<RecognizedPattern>,
    ) -> Result<Vec<RecognizedPattern>> {
        let config = self.config.read();
        let mut filtered_patterns = Vec::new();

        for pattern in patterns {
            if pattern.confidence >= config.similarity_threshold {
                // Check for duplicates
                let is_duplicate = filtered_patterns.iter().any(|existing: &RecognizedPattern| {
                    self.calculate_pattern_similarity(&pattern, existing)
                        > config.similarity_threshold
                });

                if !is_duplicate {
                    filtered_patterns.push(pattern);
                }
            }
        }

        Ok(filtered_patterns)
    }

    /// Calculate similarity between two patterns
    fn calculate_pattern_similarity(
        &self,
        pattern1: &RecognizedPattern,
        pattern2: &RecognizedPattern,
    ) -> f32 {
        // Simple similarity calculation based on pattern type and characteristics
        if std::mem::discriminant(&pattern1.pattern_type)
            != std::mem::discriminant(&pattern2.pattern_type)
        {
            return 0.0;
        }

        // Calculate characteristic similarity
        let mut total_similarity = 0.0f32;
        let mut common_keys = 0;

        for (key, value1) in &pattern1.characteristics {
            if let Some(value2) = pattern2.characteristics.get(key) {
                let diff = (value1 - value2).abs();
                let max_val = value1.abs().max(value2.abs()).max(1.0);
                total_similarity += 1.0 - (diff / max_val) as f32;
                common_keys += 1;
            }
        }

        if common_keys > 0 {
            total_similarity / common_keys as f32
        } else {
            0.0
        }
    }

    /// Cache recognized patterns
    async fn cache_patterns(&self, patterns: &[RecognizedPattern]) {
        let mut cache = self.pattern_cache.write();
        let config = self.config.read();

        for pattern in patterns {
            cache.insert(pattern.id.clone(), pattern.clone());
        }

        // Maintain cache size limit
        if cache.len() > config.cache_size {
            // Remove oldest patterns
            let mut patterns_vec: Vec<_> = cache.values().cloned().collect();
            patterns_vec.sort_by_key(|p| p.first_observed);

            let to_remove = cache.len() - config.cache_size;
            for pattern in patterns_vec.iter().take(to_remove) {
                cache.remove(&pattern.id);
            }
        }
    }

    /// Update learning models with new patterns
    async fn update_learning_models(&self, patterns: &[RecognizedPattern]) -> Result<()> {
        let mut models = self.learning_models.lock();

        for model in models.iter_mut() {
            if let Err(e) = model.learn_from_patterns(patterns) {
                tracing::warn!("Failed to update learning model {}: {}", model.name(), e);
            }
        }

        Ok(())
    }
}

impl Default for PatternRecognitionSystem {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// PATTERN DETECTOR IMPLEMENTATIONS
// =============================================================================

/// Cyclical pattern detector
pub struct CyclicalPatternDetector {
    min_cycle_length: usize,
}

impl Default for CyclicalPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl CyclicalPatternDetector {
    pub fn new() -> Self {
        Self {
            min_cycle_length: 3,
        }
    }
}

impl PatternDetector for CyclicalPatternDetector {
    fn detect_patterns(&self, events: &[OptEvent]) -> Result<Vec<RecognizedPattern>> {
        if events.len() < self.min_cycle_length * 2 {
            return Ok(Vec::new());
        }

        let mut patterns = Vec::new();

        // Detect cyclical patterns by analyzing event sequences
        for cycle_length in self.min_cycle_length..=(events.len() / 2) {
            let cycles = self.extract_cycles(events, cycle_length);

            if let Some(pattern) = self.analyze_cycle_similarity(&cycles, cycle_length) {
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    fn name(&self) -> &str {
        "cyclical_pattern_detector"
    }

    fn is_applicable(&self, events: &[OptEvent]) -> bool {
        events.len() >= self.min_cycle_length * 2
    }
}

impl CyclicalPatternDetector {
    fn extract_cycles<'a>(
        &self,
        events: &'a [OptEvent],
        cycle_length: usize,
    ) -> Vec<Vec<&'a OptEvent>> {
        let mut cycles = Vec::new();
        let num_cycles = events.len() / cycle_length;

        for i in 0..num_cycles {
            let start = i * cycle_length;
            let end = start + cycle_length;
            if end <= events.len() {
                cycles.push(events[start..end].iter().collect());
            }
        }

        cycles
    }

    fn analyze_cycle_similarity(
        &self,
        cycles: &[Vec<&OptEvent>],
        cycle_length: usize,
    ) -> Option<RecognizedPattern> {
        if cycles.len() < 2 {
            return None;
        }

        // Calculate similarity between cycles
        let mut total_similarity = 0.0f32;
        let mut comparisons = 0;

        for i in 0..cycles.len() {
            for j in i + 1..cycles.len() {
                let similarity = self.calculate_cycle_similarity(&cycles[i], &cycles[j]);
                total_similarity += similarity;
                comparisons += 1;
            }
        }

        let average_similarity =
            if comparisons > 0 { total_similarity / comparisons as f32 } else { 0.0 };

        if average_similarity >= 0.7 {
            let mut characteristics = HashMap::new();
            characteristics.insert("cycle_length".to_string(), cycle_length as f64);
            characteristics.insert("similarity".to_string(), average_similarity as f64);
            characteristics.insert("num_cycles".to_string(), cycles.len() as f64);

            Some(RecognizedPattern {
                id: format!("cyclical_{}_{}", cycle_length, Utc::now().timestamp()),
                pattern_type: PatternType::Cyclical,
                description: format!(
                    "Cyclical pattern with {} events repeating {} times",
                    cycle_length,
                    cycles.len()
                ),
                frequency: cycles.len() as f32 / 100.0, // Normalized frequency
                confidence: average_similarity,
                events: cycles.iter().flatten().cloned().cloned().collect(),
                effectiveness: average_similarity * 0.8, // Effectiveness based on consistency
                first_observed: cycles
                    .first()
                    .and_then(|c| c.first())
                    .map(|e| e.timestamp)
                    .unwrap_or_else(Utc::now),
                last_observed: cycles
                    .last()
                    .and_then(|c| c.last())
                    .map(|e| e.timestamp)
                    .unwrap_or_else(Utc::now),
                characteristics,
            })
        } else {
            None
        }
    }

    fn calculate_cycle_similarity(&self, cycle1: &[&OptEvent], cycle2: &[&OptEvent]) -> f32 {
        if cycle1.len() != cycle2.len() {
            return 0.0;
        }

        let mut matching_events = 0;
        for (e1, e2) in cycle1.iter().zip(cycle2.iter()) {
            if std::mem::discriminant(&e1.event_type) == std::mem::discriminant(&e2.event_type) {
                matching_events += 1;
            }
        }

        matching_events as f32 / cycle1.len() as f32
    }
}

/// Degradation pattern detector
pub struct DegradationPatternDetector {
    min_degradation_events: usize,
    degradation_threshold: f64,
}

impl Default for DegradationPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DegradationPatternDetector {
    pub fn new() -> Self {
        Self {
            min_degradation_events: 3,
            degradation_threshold: 0.1, // 10% degradation
        }
    }
}

impl PatternDetector for DegradationPatternDetector {
    fn detect_patterns(&self, events: &[OptEvent]) -> Result<Vec<RecognizedPattern>> {
        if events.len() < self.min_degradation_events {
            return Ok(Vec::new());
        }

        let mut patterns = Vec::new();
        let mut degradation_sequences = Vec::new();
        let mut current_sequence = Vec::new();

        for event in events {
            if self.is_degradation_event(event) {
                current_sequence.push(event.clone());
            } else {
                if current_sequence.len() >= self.min_degradation_events {
                    degradation_sequences.push(current_sequence.clone());
                }
                current_sequence.clear();
            }
        }

        // Check final sequence
        if current_sequence.len() >= self.min_degradation_events {
            degradation_sequences.push(current_sequence);
        }

        for (i, sequence) in degradation_sequences.iter().enumerate() {
            if let Some(pattern) = self.create_degradation_pattern(sequence, i) {
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    fn name(&self) -> &str {
        "degradation_pattern_detector"
    }

    fn is_applicable(&self, events: &[OptEvent]) -> bool {
        events.len() >= self.min_degradation_events
    }
}

impl DegradationPatternDetector {
    fn is_degradation_event(&self, event: &OptEvent) -> bool {
        // Simplified check - in reality would analyze performance measurements
        if let (Some(before), Some(after)) = (&event.performance_before, &event.performance_after) {
            let throughput_change = (after.throughput - before.throughput) / before.throughput;
            throughput_change < -self.degradation_threshold
        } else {
            false
        }
    }

    fn create_degradation_pattern(
        &self,
        sequence: &[OptEvent],
        sequence_id: usize,
    ) -> Option<RecognizedPattern> {
        if sequence.is_empty() {
            return None;
        }

        let total_degradation = self.calculate_total_degradation(sequence);
        let confidence = (total_degradation / self.degradation_threshold).min(1.0) as f32;

        let mut characteristics = HashMap::new();
        characteristics.insert("sequence_length".to_string(), sequence.len() as f64);
        characteristics.insert("total_degradation".to_string(), total_degradation);
        characteristics.insert(
            "average_degradation".to_string(),
            total_degradation / sequence.len() as f64,
        );

        Some(RecognizedPattern {
            id: format!("degradation_{}_{}", sequence_id, Utc::now().timestamp()),
            pattern_type: PatternType::Degradation,
            description: format!(
                "Performance degradation sequence with {} events",
                sequence.len()
            ),
            frequency: 0.2, // Degradation patterns are typically infrequent
            confidence,
            events: sequence.to_vec(),
            effectiveness: 0.1, // Low effectiveness for degradation patterns
            first_observed: sequence.first().map(|e| e.timestamp).unwrap_or_else(Utc::now),
            last_observed: sequence.last().map(|e| e.timestamp).unwrap_or_else(Utc::now),
            characteristics,
        })
    }

    fn calculate_total_degradation(&self, sequence: &[OptEvent]) -> f64 {
        sequence
            .iter()
            .filter_map(|event| {
                if let (Some(before), Some(after)) =
                    (&event.performance_before, &event.performance_after)
                {
                    Some((after.throughput - before.throughput) / before.throughput)
                } else {
                    None
                }
            })
            .sum::<f64>()
            .abs()
    }
}

/// Improvement pattern detector
pub struct ImprovementPatternDetector {
    min_improvement_events: usize,
    improvement_threshold: f64,
}

impl Default for ImprovementPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ImprovementPatternDetector {
    pub fn new() -> Self {
        Self {
            min_improvement_events: 2,
            improvement_threshold: 0.05, // 5% improvement
        }
    }
}

impl PatternDetector for ImprovementPatternDetector {
    fn detect_patterns(&self, events: &[OptEvent]) -> Result<Vec<RecognizedPattern>> {
        if events.len() < self.min_improvement_events {
            return Ok(Vec::new());
        }

        let mut patterns = Vec::new();
        let mut improvement_sequences = Vec::new();
        let mut current_sequence = Vec::new();

        for event in events {
            if self.is_improvement_event(event) {
                current_sequence.push(event.clone());
            } else {
                if current_sequence.len() >= self.min_improvement_events {
                    improvement_sequences.push(current_sequence.clone());
                }
                current_sequence.clear();
            }
        }

        // Check final sequence
        if current_sequence.len() >= self.min_improvement_events {
            improvement_sequences.push(current_sequence);
        }

        for (i, sequence) in improvement_sequences.iter().enumerate() {
            if let Some(pattern) = self.create_improvement_pattern(sequence, i) {
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    fn name(&self) -> &str {
        "improvement_pattern_detector"
    }

    fn is_applicable(&self, events: &[OptEvent]) -> bool {
        events.len() >= self.min_improvement_events
    }
}

impl ImprovementPatternDetector {
    fn is_improvement_event(&self, event: &OptEvent) -> bool {
        if let (Some(before), Some(after)) = (&event.performance_before, &event.performance_after) {
            let throughput_change = (after.throughput - before.throughput) / before.throughput;
            throughput_change > self.improvement_threshold
        } else {
            false
        }
    }

    fn create_improvement_pattern(
        &self,
        sequence: &[OptEvent],
        sequence_id: usize,
    ) -> Option<RecognizedPattern> {
        if sequence.is_empty() {
            return None;
        }

        let total_improvement = self.calculate_total_improvement(sequence);
        let confidence = (total_improvement / self.improvement_threshold).min(1.0) as f32;

        let mut characteristics = HashMap::new();
        characteristics.insert("sequence_length".to_string(), sequence.len() as f64);
        characteristics.insert("total_improvement".to_string(), total_improvement);
        characteristics.insert(
            "average_improvement".to_string(),
            total_improvement / sequence.len() as f64,
        );

        Some(RecognizedPattern {
            id: format!("improvement_{}_{}", sequence_id, Utc::now().timestamp()),
            pattern_type: PatternType::Improvement,
            description: format!(
                "Performance improvement sequence with {} events",
                sequence.len()
            ),
            frequency: 0.3, // Moderate frequency for improvement patterns
            confidence,
            events: sequence.to_vec(),
            effectiveness: confidence * 0.9, // High effectiveness for improvements
            first_observed: sequence.first().map(|e| e.timestamp).unwrap_or_else(Utc::now),
            last_observed: sequence.last().map(|e| e.timestamp).unwrap_or_else(Utc::now),
            characteristics,
        })
    }

    fn calculate_total_improvement(&self, sequence: &[OptEvent]) -> f64 {
        sequence
            .iter()
            .filter_map(|event| {
                if let (Some(before), Some(after)) =
                    (&event.performance_before, &event.performance_after)
                {
                    Some((after.throughput - before.throughput) / before.throughput)
                } else {
                    None
                }
            })
            .sum()
    }
}

/// Oscillation pattern detector
pub struct OscillationPatternDetector {
    min_oscillations: usize,
    oscillation_threshold: f64,
}

impl Default for OscillationPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl OscillationPatternDetector {
    pub fn new() -> Self {
        Self {
            min_oscillations: 3,
            oscillation_threshold: 0.1, // 10% variation
        }
    }
}

impl PatternDetector for OscillationPatternDetector {
    fn detect_patterns(&self, events: &[OptEvent]) -> Result<Vec<RecognizedPattern>> {
        if events.len() < self.min_oscillations * 2 {
            return Ok(Vec::new());
        }

        let performance_values = self.extract_performance_values(events);
        if let Some(pattern) = self.detect_oscillation(&performance_values, events) {
            Ok(vec![pattern])
        } else {
            Ok(Vec::new())
        }
    }

    fn name(&self) -> &str {
        "oscillation_pattern_detector"
    }

    fn is_applicable(&self, events: &[OptEvent]) -> bool {
        events.len() >= self.min_oscillations * 2
    }
}

impl OscillationPatternDetector {
    fn extract_performance_values(&self, events: &[OptEvent]) -> Vec<f64> {
        events
            .iter()
            .filter_map(|event| event.performance_after.as_ref().map(|p| p.throughput))
            .collect()
    }

    fn detect_oscillation(&self, values: &[f64], events: &[OptEvent]) -> Option<RecognizedPattern> {
        if values.len() < 4 {
            return None;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let mut direction_changes = 0;
        let mut last_direction = None;

        for i in 1..values.len() {
            let current_direction = values[i] > values[i - 1];
            if let Some(last_dir) = last_direction {
                if current_direction != last_dir {
                    direction_changes += 1;
                }
            }
            last_direction = Some(current_direction);
        }

        let oscillation_frequency = direction_changes as f64 / (values.len() - 1) as f64;

        if oscillation_frequency >= 0.4 {
            // Calculate amplitude
            let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let amplitude = (max_val - min_val) / mean;

            if amplitude > self.oscillation_threshold {
                let confidence = (oscillation_frequency * amplitude).min(1.0) as f32;

                let mut characteristics = HashMap::new();
                characteristics.insert("oscillation_frequency".to_string(), oscillation_frequency);
                characteristics.insert("amplitude".to_string(), amplitude);
                characteristics.insert("direction_changes".to_string(), direction_changes as f64);
                characteristics.insert("mean_value".to_string(), mean);

                return Some(RecognizedPattern {
                    id: format!("oscillation_{}", Utc::now().timestamp()),
                    pattern_type: PatternType::Oscillation,
                    description: format!(
                        "Performance oscillation with {} direction changes",
                        direction_changes
                    ),
                    frequency: oscillation_frequency as f32,
                    confidence,
                    events: events.to_vec(),
                    effectiveness: 0.4, // Medium effectiveness - oscillations can indicate instability
                    first_observed: events.first().map(|e| e.timestamp).unwrap_or_else(Utc::now),
                    last_observed: events.last().map(|e| e.timestamp).unwrap_or_else(Utc::now),
                    characteristics,
                });
            }
        }

        None
    }
}

/// Threshold pattern detector
pub struct ThresholdPatternDetector {
    threshold_values: Vec<f64>,
}

impl Default for ThresholdPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ThresholdPatternDetector {
    pub fn new() -> Self {
        Self {
            threshold_values: vec![100.0, 500.0, 1000.0, 5000.0], // Common performance thresholds
        }
    }
}

impl PatternDetector for ThresholdPatternDetector {
    fn detect_patterns(&self, events: &[OptEvent]) -> Result<Vec<RecognizedPattern>> {
        let mut patterns = Vec::new();

        for &threshold in &self.threshold_values {
            if let Some(pattern) = self.detect_threshold_pattern(events, threshold) {
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    fn name(&self) -> &str {
        "threshold_pattern_detector"
    }

    fn is_applicable(&self, events: &[OptEvent]) -> bool {
        events.len() >= 3
    }
}

impl ThresholdPatternDetector {
    fn detect_threshold_pattern(
        &self,
        events: &[OptEvent],
        threshold: f64,
    ) -> Option<RecognizedPattern> {
        let threshold_events: Vec<&OptEvent> = events
            .iter()
            .filter(|event| {
                event.performance_after
                    .as_ref()
                    .map(|p| (p.throughput - threshold).abs() < threshold * 0.05) // Within 5% of threshold
                    .unwrap_or(false)
            })
            .collect();

        if threshold_events.len() >= 2 {
            let confidence = threshold_events.len() as f32 / events.len() as f32;

            let mut characteristics = HashMap::new();
            characteristics.insert("threshold_value".to_string(), threshold);
            characteristics.insert(
                "threshold_events".to_string(),
                threshold_events.len() as f64,
            );
            characteristics.insert("total_events".to_string(), events.len() as f64);

            Some(RecognizedPattern {
                id: format!("threshold_{}_{}", threshold as i64, Utc::now().timestamp()),
                pattern_type: PatternType::Threshold,
                description: format!("Performance clustering around threshold {}", threshold),
                frequency: confidence,
                confidence,
                events: threshold_events.into_iter().cloned().collect(),
                effectiveness: confidence * 0.6, // Moderate effectiveness
                first_observed: events.first().map(|e| e.timestamp).unwrap_or_else(Utc::now),
                last_observed: events.last().map(|e| e.timestamp).unwrap_or_else(Utc::now),
                characteristics,
            })
        } else {
            None
        }
    }
}

// =============================================================================
// PATTERN LEARNING MODEL IMPLEMENTATIONS
// =============================================================================

/// Simple pattern learner that tracks pattern frequencies
pub struct SimplePatternLearner {
    pattern_frequencies: HashMap<String, f32>,
    learning_rate: f32,
}

impl Default for SimplePatternLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplePatternLearner {
    pub fn new() -> Self {
        Self {
            pattern_frequencies: HashMap::new(),
            learning_rate: 0.1,
        }
    }
}

impl PatternLearningModel for SimplePatternLearner {
    fn learn_from_patterns(&mut self, patterns: &[RecognizedPattern]) -> Result<()> {
        for pattern in patterns {
            let key = format!(
                "{}_{}",
                match pattern.pattern_type {
                    PatternType::Cyclical => "cyclical",
                    PatternType::Degradation => "degradation",
                    PatternType::Improvement => "improvement",
                    PatternType::Oscillation => "oscillation",
                    PatternType::Threshold => "threshold",
                    PatternType::Custom(ref s) => s,
                },
                *pattern
                    .characteristics
                    .get("cycle_length")
                    .or(pattern.characteristics.get("sequence_length"))
                    .unwrap_or(&0.0) as i64
            );

            let current_freq = self.pattern_frequencies.get(&key).unwrap_or(&0.0);
            let updated_freq = current_freq + self.learning_rate * pattern.frequency;
            self.pattern_frequencies.insert(key, updated_freq);
        }

        Ok(())
    }

    fn predict_pattern(&self, _context: &PatternContext) -> Result<PatternPrediction> {
        // Find the most frequent pattern type
        let most_frequent = self
            .pattern_frequencies
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((pattern_key, frequency)) = most_frequent {
            let predicted_pattern = if pattern_key.starts_with("cyclical") {
                PatternType::Cyclical
            } else if pattern_key.starts_with("degradation") {
                PatternType::Degradation
            } else if pattern_key.starts_with("improvement") {
                PatternType::Improvement
            } else if pattern_key.starts_with("oscillation") {
                PatternType::Oscillation
            } else if pattern_key.starts_with("threshold") {
                PatternType::Threshold
            } else {
                PatternType::Custom("unknown".to_string())
            };

            Ok(PatternPrediction {
                predicted_pattern,
                confidence: *frequency,
                expected_occurrence: Utc::now() + chrono::Duration::hours(1), // Simple prediction
                horizon: Duration::from_secs(3600),
                recommended_actions: vec![
                    "Monitor system closely".to_string(),
                    "Prepare optimization strategies".to_string(),
                ],
            })
        } else {
            Err(anyhow::anyhow!("No pattern data available for prediction"))
        }
    }

    fn name(&self) -> &str {
        "simple_pattern_learner"
    }
}

/// Frequency-based pattern predictor
pub struct FrequencyBasedPredictor {
    pattern_intervals: HashMap<String, Vec<Duration>>,
}

impl Default for FrequencyBasedPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl FrequencyBasedPredictor {
    pub fn new() -> Self {
        Self {
            pattern_intervals: HashMap::new(),
        }
    }
}

impl PatternLearningModel for FrequencyBasedPredictor {
    fn learn_from_patterns(&mut self, patterns: &[RecognizedPattern]) -> Result<()> {
        for pattern in patterns {
            let key = format!("{:?}", pattern.pattern_type);
            let interval = pattern.last_observed.signed_duration_since(pattern.first_observed);

            if let Ok(std_duration) = interval.to_std() {
                self.pattern_intervals.entry(key).or_default().push(std_duration);
            }
        }

        Ok(())
    }

    fn predict_pattern(&self, _context: &PatternContext) -> Result<PatternPrediction> {
        // Find pattern with most predictable interval
        let mut best_prediction = None;
        let mut best_confidence = 0.0f32;

        for (pattern_type_str, intervals) in &self.pattern_intervals {
            if !intervals.is_empty() {
                let avg_interval = intervals.iter().sum::<Duration>() / intervals.len() as u32;
                let confidence = 1.0 / (intervals.len() as f32); // Simple confidence calculation

                if confidence > best_confidence {
                    best_confidence = confidence;

                    let predicted_pattern = match pattern_type_str.as_str() {
                        "Cyclical" => PatternType::Cyclical,
                        "Degradation" => PatternType::Degradation,
                        "Improvement" => PatternType::Improvement,
                        "Oscillation" => PatternType::Oscillation,
                        "Threshold" => PatternType::Threshold,
                        _ => PatternType::Custom("unknown".to_string()),
                    };

                    best_prediction = Some(PatternPrediction {
                        predicted_pattern,
                        confidence,
                        expected_occurrence: Utc::now()
                            + chrono::Duration::from_std(avg_interval).unwrap_or_default(),
                        horizon: avg_interval,
                        recommended_actions: vec![
                            format!("Expect {} pattern in {:?}", pattern_type_str, avg_interval),
                            "Prepare appropriate response".to_string(),
                        ],
                    });
                }
            }
        }

        best_prediction.ok_or_else(|| anyhow::anyhow!("No predictable patterns found"))
    }

    fn name(&self) -> &str {
        "frequency_based_predictor"
    }
}

/// Contextual pattern predictor that considers system state
pub struct ContextualPatternPredictor {
    context_patterns: HashMap<String, Vec<RecognizedPattern>>,
}

impl Default for ContextualPatternPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextualPatternPredictor {
    pub fn new() -> Self {
        Self {
            context_patterns: HashMap::new(),
        }
    }

    fn generate_context_key(&self, context: &PatternContext) -> String {
        format!(
            "cores_{}_load_{:.1}",
            context.system_state.available_cores, context.system_state.load_average
        )
    }
}

impl PatternLearningModel for ContextualPatternPredictor {
    fn learn_from_patterns(&mut self, patterns: &[RecognizedPattern]) -> Result<()> {
        // For now, just store patterns - in reality would need context information
        let generic_key = "generic".to_string();
        self.context_patterns
            .entry(generic_key)
            .or_default()
            .extend(patterns.iter().cloned());
        Ok(())
    }

    fn predict_pattern(&self, context: &PatternContext) -> Result<PatternPrediction> {
        let context_key = self.generate_context_key(context);

        // Look for patterns in similar context, fall back to generic patterns
        let relevant_patterns = self
            .context_patterns
            .get(&context_key)
            .or_else(|| self.context_patterns.get("generic"));

        if let Some(patterns) = relevant_patterns {
            if let Some(most_effective) = patterns.iter().max_by(|a, b| {
                a.effectiveness
                    .partial_cmp(&b.effectiveness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                return Ok(PatternPrediction {
                    predicted_pattern: most_effective.pattern_type.clone(),
                    confidence: most_effective.confidence * 0.8, // Reduce confidence for prediction
                    expected_occurrence: Utc::now() + chrono::Duration::minutes(30),
                    horizon: Duration::from_secs(1800),
                    recommended_actions: vec![
                        format!(
                            "Prepare for {} pattern based on system context",
                            match most_effective.pattern_type {
                                PatternType::Cyclical => "cyclical",
                                PatternType::Degradation => "degradation",
                                PatternType::Improvement => "improvement",
                                PatternType::Oscillation => "oscillation",
                                PatternType::Threshold => "threshold",
                                PatternType::Custom(ref s) => s,
                            }
                        ),
                        "Monitor key performance metrics".to_string(),
                    ],
                });
            }
        }

        Err(anyhow::anyhow!(
            "No contextual patterns available for prediction"
        ))
    }

    fn name(&self) -> &str {
        "contextual_pattern_predictor"
    }
}

// =============================================================================
// UTILITY TYPES
// =============================================================================

/// Pattern recognition statistics
#[derive(Debug, Clone)]
pub struct PatternStatistics {
    pub total_patterns: usize,
    pub pattern_type_distribution: HashMap<PatternType, usize>,
    pub average_effectiveness: f32,
    pub high_confidence_patterns: usize,
    pub cache_memory_usage: usize,
}
