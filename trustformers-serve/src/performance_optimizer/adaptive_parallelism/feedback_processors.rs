//! Feedback Processors for Performance Feedback System
//!
//! This module provides specialized feedback processors that handle different types
//! of performance feedback including throughput, latency, and resource utilization.
//! Each processor converts raw feedback into actionable insights and recommendations.

use anyhow::Result;
use std::collections::HashMap;

use crate::performance_optimizer::types::*;

// Note: FeedbackProcessor trait is defined in types.rs and imported above
// Re-export the trait for use by parent module
pub use crate::performance_optimizer::types::FeedbackProcessor;

// =============================================================================
// THROUGHPUT FEEDBACK PROCESSOR
// =============================================================================

/// Throughput feedback processor
pub struct ThroughputFeedbackProcessor {
    name: String,
}

impl Default for ThroughputFeedbackProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ThroughputFeedbackProcessor {
    pub fn new() -> Self {
        Self {
            name: "throughput_processor".to_string(),
        }
    }
}

impl FeedbackProcessor for ThroughputFeedbackProcessor {
    fn process_feedback(&self, feedback: &PerformanceFeedback) -> Result<ProcessedFeedback> {
        let processed_value = feedback.value;
        let confidence = 0.9; // High confidence for throughput measurements

        let recommended_action = if feedback.value < 50.0 {
            Some(RecommendedAction {
                action_type: ActionType::IncreaseParallelism,
                parameters: HashMap::new(),
                priority: 0.8,
                expected_impact: 0.3,
                reversible: true,
                estimated_duration: std::time::Duration::from_secs(5),
            })
        } else if feedback.value > 200.0 {
            Some(RecommendedAction {
                action_type: ActionType::DecreaseParallelism,
                parameters: HashMap::new(),
                priority: 0.6,
                expected_impact: 0.2,
                reversible: true,
                estimated_duration: std::time::Duration::from_secs(3),
            })
        } else {
            None
        };

        Ok(ProcessedFeedback {
            original_feedback: feedback.clone(),
            processed_value,
            processing_method: self.name.clone(),
            confidence,
            recommended_action,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_process(&self, feedback_type: &FeedbackType) -> bool {
        matches!(feedback_type, FeedbackType::Throughput)
    }
}

// =============================================================================
// LATENCY FEEDBACK PROCESSOR
// =============================================================================

/// Latency feedback processor
pub struct LatencyFeedbackProcessor {
    name: String,
}

impl Default for LatencyFeedbackProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyFeedbackProcessor {
    pub fn new() -> Self {
        Self {
            name: "latency_processor".to_string(),
        }
    }
}

impl FeedbackProcessor for LatencyFeedbackProcessor {
    fn process_feedback(&self, feedback: &PerformanceFeedback) -> Result<ProcessedFeedback> {
        // Convert latency to score (lower latency = higher score)
        let processed_value = 1000.0 / feedback.value.max(1.0);
        let confidence = 0.8;

        let recommended_action = if feedback.value > 100.0 {
            Some(RecommendedAction {
                action_type: ActionType::DecreaseParallelism,
                parameters: HashMap::new(),
                priority: 0.7,
                expected_impact: 0.25,
                reversible: true,
                estimated_duration: std::time::Duration::from_secs(3),
            })
        } else {
            None
        };

        Ok(ProcessedFeedback {
            original_feedback: feedback.clone(),
            processed_value,
            processing_method: self.name.clone(),
            confidence,
            recommended_action,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_process(&self, feedback_type: &FeedbackType) -> bool {
        matches!(feedback_type, FeedbackType::Latency)
    }
}

// =============================================================================
// RESOURCE UTILIZATION FEEDBACK PROCESSOR
// =============================================================================

/// Resource utilization feedback processor
pub struct ResourceUtilizationFeedbackProcessor {
    name: String,
}

impl Default for ResourceUtilizationFeedbackProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceUtilizationFeedbackProcessor {
    pub fn new() -> Self {
        Self {
            name: "resource_utilization_processor".to_string(),
        }
    }
}

impl FeedbackProcessor for ResourceUtilizationFeedbackProcessor {
    fn process_feedback(&self, feedback: &PerformanceFeedback) -> Result<ProcessedFeedback> {
        let processed_value = feedback.value;
        let confidence = 0.7;

        let recommended_action = if feedback.value < 0.3 {
            Some(RecommendedAction {
                action_type: ActionType::IncreaseParallelism,
                parameters: HashMap::new(),
                priority: 0.6,
                expected_impact: 0.2,
                reversible: true,
                estimated_duration: std::time::Duration::from_secs(5),
            })
        } else if feedback.value > 0.9 {
            Some(RecommendedAction {
                action_type: ActionType::DecreaseParallelism,
                parameters: HashMap::new(),
                priority: 0.8,
                expected_impact: 0.3,
                reversible: true,
                estimated_duration: std::time::Duration::from_secs(3),
            })
        } else {
            None
        };

        Ok(ProcessedFeedback {
            original_feedback: feedback.clone(),
            processed_value,
            processing_method: self.name.clone(),
            confidence,
            recommended_action,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_process(&self, feedback_type: &FeedbackType) -> bool {
        matches!(feedback_type, FeedbackType::ResourceUtilization)
    }
}
