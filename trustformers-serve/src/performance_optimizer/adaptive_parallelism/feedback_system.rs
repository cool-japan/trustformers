//! Performance Feedback System Implementation
//!
//! This module provides the PerformanceFeedbackSystem that collects, processes,
//! and aggregates performance feedback from various sources. It includes real-time
//! processing capabilities and coordination with multiple feedback processors.

use anyhow::Result;
use parking_lot::Mutex;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use super::aggregation::FeedbackAggregator;
// FeedbackProcessor trait is now imported from types (not from feedback_processors)
use crate::performance_optimizer::types::*;

// Re-export types needed by other modules
pub use crate::performance_optimizer::types::{PerformanceFeedback, PerformanceFeedbackSystem};

// =============================================================================
// PERFORMANCE FEEDBACK SYSTEM IMPLEMENTATION
// =============================================================================

impl PerformanceFeedbackSystem {
    /// Create a new performance feedback system
    pub async fn new() -> Result<Self> {
        let mut processors: Vec<Box<dyn FeedbackProcessor + Send + Sync>> = Vec::new();

        // Add default feedback processors
        processors.push(Box::new(
            super::feedback_processors::ThroughputFeedbackProcessor::new(),
        ));
        processors.push(Box::new(
            super::feedback_processors::LatencyFeedbackProcessor::new(),
        ));
        processors.push(Box::new(
            super::feedback_processors::ResourceUtilizationFeedbackProcessor::new(),
        ));

        Ok(Self {
            feedback_queue: Arc::new(Mutex::new(VecDeque::new())),
            feedback_processors: Arc::new(Mutex::new(processors)),
            feedback_aggregator: Arc::new(FeedbackAggregator::new().await?),
            real_time_feedback: Arc::new(AtomicBool::new(true)),
        })
    }

    /// Add performance feedback
    pub async fn add_feedback(&self, feedback: PerformanceFeedback) -> Result<()> {
        self.feedback_queue.lock().push_back(feedback.clone());

        // Process feedback immediately if real-time processing is enabled
        if self.real_time_feedback.load(Ordering::Relaxed) {
            self.process_feedback_item(feedback).await?;
        }

        Ok(())
    }

    /// Process a single feedback item
    async fn process_feedback_item(&self, feedback: PerformanceFeedback) -> Result<()> {
        let processors = self.feedback_processors.lock();
        let mut processed_feedback = Vec::new();

        for processor in processors.iter() {
            if processor.can_process(&feedback.feedback_type) {
                match processor.process_feedback(&feedback) {
                    Ok(processed) => processed_feedback.push(processed),
                    Err(e) => log::warn!("Feedback processor {} failed: {}", processor.name(), e),
                }
            }
        }

        if !processed_feedback.is_empty() {
            self.feedback_aggregator.aggregate_feedback(&processed_feedback).await?;
        }

        Ok(())
    }

    /// Get feedback count
    pub async fn get_feedback_count(&self) -> Result<usize> {
        Ok(self.feedback_queue.lock().len())
    }

    /// Process all queued feedback
    pub async fn process_queued_feedback(&self) -> Result<usize> {
        let mut processed_count = 0;

        while let Some(feedback) = self.feedback_queue.lock().pop_front() {
            self.process_feedback_item(feedback).await?;
            processed_count += 1;
        }

        Ok(processed_count)
    }

    /// Get recent aggregated feedback
    pub async fn get_recent_aggregated_feedback(&self) -> Result<Vec<AggregatedFeedback>> {
        self.feedback_aggregator.get_recent_aggregated_feedback().await
    }
}
