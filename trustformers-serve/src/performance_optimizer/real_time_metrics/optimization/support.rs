//! Supporting Types and Implementations
//!
//! Additional types and utility implementations for optimization system

use chrono::{DateTime, Utc};
use std::collections::HashMap;

// =============================================================================

/// Statistics for optimization engine
#[derive(Debug, Clone)]
pub struct OptimizationEngineStatistics {
    pub recommendations_generated: u64,
    pub successful_optimizations: u64,
    pub average_confidence: f32,
    pub average_impact: f32,
    pub uptime_seconds: u64,
    pub processing_latency_ms: f32,
    pub active_algorithms: usize,
}

/// Analysis result from real-time analysis algorithms
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub algorithm_name: String,
    pub timestamp: DateTime<Utc>,
    pub confidence: f32,
    pub insights: Vec<String>,
    pub recommendations: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Training example for adaptive learning
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub input_features: Vec<f32>,
    pub output_success: bool,
    pub recommendation_type: String,
    pub timestamp: DateTime<Utc>,
}
