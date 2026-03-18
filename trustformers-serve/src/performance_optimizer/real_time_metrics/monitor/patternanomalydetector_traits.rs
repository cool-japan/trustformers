//! # PatternAnomalyDetector - Trait Implementations
//!
//! This module contains trait implementations for `PatternAnomalyDetector`.
//!
//! ## Implemented Traits
//!
//! - `AnomalyDetectionAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::TimestampedMetrics;
use super::types::*;
use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;

use super::functions::AnomalyDetectionAlgorithm;
use super::types::{
    AnomalyAlgorithmStats, AnomalyDetectionConfig, AnomalyEvent, PatternAnomalyDetector,
    PerformanceBaseline,
};

impl AnomalyDetectionAlgorithm for PatternAnomalyDetector {
    fn detect_anomaly(
        &self,
        metrics: &TimestampedMetrics,
        _baseline: &PerformanceBaseline,
    ) -> Result<Option<AnomalyEvent>> {
        let current_pattern = self.extract_pattern(metrics);
        let mut max_similarity = 0.0f32;
        for historical_pattern in &self.patterns {
            let similarity = self.calculate_similarity(&current_pattern, historical_pattern);
            max_similarity = max_similarity.max(similarity);
        }
        if max_similarity < self.similarity_threshold && !self.patterns.is_empty() {
            let anomaly_score = 1.0 - max_similarity;
            let severity = match anomaly_score {
                s if s > 0.8 => SeverityLevel::Critical,
                s if s > 0.6 => SeverityLevel::High,
                s if s > 0.4 => SeverityLevel::Medium,
                _ => SeverityLevel::Low,
            };
            let anomaly = AnomalyEvent {
                timestamp: Utc::now(),
                anomaly_type: "pattern_deviation".to_string(),
                severity,
                description: format!(
                    "Pattern anomaly detected with similarity: {:.2}",
                    max_similarity
                ),
                affected_metrics: vec!["pattern".to_string()],
                score: anomaly_score,
                confidence: 0.75,
                expected_value: max_similarity as f64,
                actual_value: self.similarity_threshold as f64,
                deviation: (self.similarity_threshold - max_similarity) as f64,
                detection_algorithm: "pattern".to_string(),
                context: {
                    let mut ctx = HashMap::new();
                    ctx.insert("max_similarity".to_string(), max_similarity.to_string());
                    ctx.insert(
                        "patterns_count".to_string(),
                        self.patterns.len().to_string(),
                    );
                    ctx
                },
                recommendations: vec![
                    "Analyze recent system changes".to_string(),
                    "Review pattern matching parameters".to_string(),
                    "Consider expanding pattern database".to_string(),
                ],
            };
            Ok(Some(anomaly))
        } else {
            Ok(None)
        }
    }
    fn name(&self) -> &str {
        "pattern"
    }
    fn confidence(&self) -> f32 {
        0.75
    }
    fn update_parameters(&mut self, config: &AnomalyDetectionConfig) -> Result<()> {
        self.similarity_threshold = config.sensitivity;
        Ok(())
    }
    fn get_statistics(&self) -> AnomalyAlgorithmStats {
        self.stats.clone()
    }
}
