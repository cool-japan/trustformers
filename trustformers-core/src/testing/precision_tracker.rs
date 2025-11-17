//! Precision Loss Tracking for Numerical Stability Monitoring
//!
//! This module provides comprehensive tracking of precision loss during
//! numerical computations, helping identify potential instability issues
//! in transformer models and training processes.

use scirs2_core::random::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Errors that can occur during precision tracking
#[derive(Error, Debug)]
pub enum PrecisionTrackingError {
    #[error("Invalid precision parameters: {0}")]
    InvalidParameters(String),
    #[error("Tracking not initialized")]
    NotInitialized,
    #[error("Operation not found: {0}")]
    OperationNotFound(String),
    #[error("Statistics error: {0}")]
    StatisticsError(String),
}

/// Configuration for precision loss tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionConfig {
    /// Enable tracking of gradients
    pub track_gradients: bool,
    /// Enable tracking of activations
    pub track_activations: bool,
    /// Enable tracking of weights
    pub track_weights: bool,
    /// Threshold for detecting significant precision loss
    pub loss_threshold: f64,
    /// Maximum number of operations to track
    pub max_operations: usize,
    /// Sample rate for tracking (0.0 to 1.0)
    pub sample_rate: f64,
    /// Enable automatic alerts for precision issues
    pub enable_alerts: bool,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            track_gradients: true,
            track_activations: true,
            track_weights: true,
            loss_threshold: 1e-6,
            max_operations: 10000,
            sample_rate: 1.0,
            enable_alerts: true,
        }
    }
}

/// Precision data for a single operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionData {
    /// Name of the operation
    pub operation: String,
    /// Input precision (number of bits effectively used)
    pub input_precision: f64,
    /// Output precision (number of bits effectively used)
    pub output_precision: f64,
    /// Precision loss (input - output)
    pub precision_loss: f64,
    /// Relative precision loss (precision_loss / input_precision)
    pub relative_loss: f64,
    /// Timestamp of the operation
    pub timestamp: u64,
    /// Operation type (gradient, activation, weight)
    pub operation_type: OperationType,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Type of operation being tracked
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum OperationType {
    Gradient,
    Activation,
    Weight,
    MatMul,
    Attention,
    LayerNorm,
    Other,
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationType::Gradient => write!(f, "Gradient"),
            OperationType::Activation => write!(f, "Activation"),
            OperationType::Weight => write!(f, "Weight"),
            OperationType::MatMul => write!(f, "MatMul"),
            OperationType::Attention => write!(f, "Attention"),
            OperationType::LayerNorm => write!(f, "LayerNorm"),
            OperationType::Other => write!(f, "Other"),
        }
    }
}

/// Statistics for precision tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionStatistics {
    /// Total number of operations tracked
    pub total_operations: usize,
    /// Average precision loss across all operations
    pub average_precision_loss: f64,
    /// Maximum precision loss observed
    pub max_precision_loss: f64,
    /// Minimum precision loss observed
    pub min_precision_loss: f64,
    /// Standard deviation of precision loss
    pub std_precision_loss: f64,
    /// Operations with significant precision loss
    pub operations_with_loss: usize,
    /// Percentage of operations with significant loss
    pub loss_percentage: f64,
    /// Per-operation-type statistics
    pub per_type_stats: HashMap<OperationType, TypeStatistics>,
    /// Timeline of precision loss
    pub timeline: Vec<TimelinePoint>,
}

/// Statistics for a specific operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeStatistics {
    /// Number of operations of this type
    pub count: usize,
    /// Average precision loss for this type
    pub avg_loss: f64,
    /// Maximum precision loss for this type
    pub max_loss: f64,
    /// Standard deviation for this type
    pub std_loss: f64,
}

/// Point in the precision timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelinePoint {
    /// Timestamp
    pub timestamp: u64,
    /// Precision loss at this time
    pub precision_loss: f64,
    /// Operation type
    pub operation_type: OperationType,
    /// Operation name
    pub operation: String,
}

/// Main precision tracker
pub struct PrecisionTracker {
    config: PrecisionConfig,
    data: Vec<PrecisionData>,
    operation_counts: HashMap<String, usize>,
    initialized: bool,
}

impl PrecisionTracker {
    /// Create a new precision tracker with default configuration
    pub fn new() -> Self {
        Self {
            config: PrecisionConfig::default(),
            data: Vec::new(),
            operation_counts: HashMap::new(),
            initialized: true,
        }
    }

    /// Create a new precision tracker with custom configuration
    pub fn with_config(config: PrecisionConfig) -> Self {
        Self {
            config,
            data: Vec::new(),
            operation_counts: HashMap::new(),
            initialized: true,
        }
    }

    /// Track precision for a numerical operation
    pub fn track_precision(
        &mut self,
        operation: String,
        input_data: &[f64],
        output_data: &[f64],
        operation_type: OperationType,
    ) -> Result<(), PrecisionTrackingError> {
        if !self.initialized {
            return Err(PrecisionTrackingError::NotInitialized);
        }

        // Check if we should sample this operation
        let mut rng = thread_rng();
        if rng.gen::<f64>() > self.config.sample_rate {
            return Ok(());
        }

        // Check if we've exceeded max operations
        if self.data.len() >= self.config.max_operations {
            // Remove oldest entries to make room
            let remove_count = self.data.len() - self.config.max_operations + 1;
            self.data.drain(0..remove_count);
        }

        // Calculate effective precision
        let input_precision = self.calculate_effective_precision(input_data)?;
        let output_precision = self.calculate_effective_precision(output_data)?;

        // Also calculate precision loss based on relative error between input and output
        let relative_error_loss = self.calculate_relative_error_loss(input_data, output_data)?;

        // Use the maximum of the two methods to detect precision loss
        let precision_loss = (input_precision - output_precision).max(relative_error_loss);
        let relative_loss =
            if input_precision > 0.0 { precision_loss / input_precision } else { 0.0 };

        // Get current timestamp
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;

        // Update operation counts
        *self.operation_counts.entry(operation.clone()).or_insert(0) += 1;

        // Create precision data
        let precision_data = PrecisionData {
            operation: operation.clone(),
            input_precision,
            output_precision,
            precision_loss,
            relative_loss,
            timestamp,
            operation_type,
            metadata: HashMap::new(),
        };

        // Check for alerts
        if self.config.enable_alerts && precision_loss > self.config.loss_threshold {
            self.generate_alert(&precision_data);
        }

        self.data.push(precision_data);
        Ok(())
    }

    /// Track precision for tensor-like data (converting to f64)
    pub fn track_tensor_precision<T: Into<f64> + Copy>(
        &mut self,
        operation: String,
        input_data: &[T],
        output_data: &[T],
        operation_type: OperationType,
    ) -> Result<(), PrecisionTrackingError> {
        let input_f64: Vec<f64> = input_data.iter().map(|&x| x.into()).collect();
        let output_f64: Vec<f64> = output_data.iter().map(|&x| x.into()).collect();

        self.track_precision(operation, &input_f64, &output_f64, operation_type)
    }

    /// Calculate effective precision (bits of information)
    fn calculate_effective_precision(&self, data: &[f64]) -> Result<f64, PrecisionTrackingError> {
        if data.is_empty() {
            return Ok(0.0);
        }

        // Find the dynamic range of the data
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0f64, |a, b| a.max(b));
        let min_abs = data
            .iter()
            .map(|x| x.abs())
            .filter(|&x| x > 0.0)
            .fold(f64::INFINITY, |a, b| a.min(b));

        if max_abs <= 0.0 {
            return Ok(0.0);
        }

        // Calculate precision based on the dynamic range
        // For f64, we have ~15-17 decimal digits of precision
        // The effective precision depends on the magnitude of the largest value
        let max_magnitude = max_abs.log10().floor();
        let theoretical_precision = 52.0; // f64 mantissa bits

        // Adjust for the dynamic range - larger numbers lose precision in the lower bits
        let magnitude_penalty = (max_magnitude * 3.32).max(0.0); // log2(10) ≈ 3.32
        let effective_precision = (theoretical_precision - magnitude_penalty).max(0.0);

        // If we have a good dynamic range, we maintain more precision
        if min_abs > 0.0 {
            let dynamic_range_bits = (max_abs / min_abs).log2();
            // If dynamic range is reasonable, we get better precision
            if dynamic_range_bits < 40.0 {
                Ok(effective_precision.min(52.0))
            } else {
                Ok((effective_precision - (dynamic_range_bits - 40.0) * 0.5).max(0.0))
            }
        } else {
            Ok(effective_precision)
        }
    }

    /// Calculate precision loss based on relative error between input and output
    fn calculate_relative_error_loss(
        &self,
        input_data: &[f64],
        output_data: &[f64],
    ) -> Result<f64, PrecisionTrackingError> {
        if input_data.len() != output_data.len() || input_data.is_empty() {
            return Ok(0.0);
        }

        // Calculate the relative error for each element
        let mut total_error_bits = 0.0;
        let mut valid_comparisons = 0;

        for (inp, out) in input_data.iter().zip(output_data.iter()) {
            if inp.abs() > f64::EPSILON && out.abs() > f64::EPSILON {
                // Calculate precision loss based on the ratio of input to output values
                // For [1000, 2000, 3000] -> [1, 2, 3], we should detect log2(1000) ≈ 10 bits loss
                let ratio = inp.abs() / out.abs();
                if !(1.0 - f64::EPSILON..=1.0 + f64::EPSILON).contains(&ratio) {
                    // The precision loss is related to how much the magnitude changed
                    let magnitude_change = if ratio > 1.0 {
                        ratio.log2() // Lost precision when scaling down
                    } else {
                        (1.0 / ratio).log2() // Lost precision when scaling up
                    };
                    total_error_bits += magnitude_change;
                    valid_comparisons += 1;
                }
            } else if inp.abs() > f64::EPSILON && out.abs() <= f64::EPSILON {
                // Complete loss of signal
                total_error_bits += 52.0; // Maximum precision loss for f64
                valid_comparisons += 1;
            }
        }

        if valid_comparisons == 0 {
            return Ok(0.0);
        }

        // Return the average precision loss in bits
        let avg_precision_loss = total_error_bits / valid_comparisons as f64;

        // For the case of [1000, 2000, 3000] -> [1, 2, 3], this should detect
        // a significant precision loss due to the scaling factor
        Ok(avg_precision_loss.min(52.0)) // Cap at f64 precision limit
    }

    /// Generate precision statistics
    pub fn get_statistics(&self) -> Result<PrecisionStatistics, PrecisionTrackingError> {
        if self.data.is_empty() {
            return Ok(PrecisionStatistics {
                total_operations: 0,
                average_precision_loss: 0.0,
                max_precision_loss: 0.0,
                min_precision_loss: 0.0,
                std_precision_loss: 0.0,
                operations_with_loss: 0,
                loss_percentage: 0.0,
                per_type_stats: HashMap::new(),
                timeline: Vec::new(),
            });
        }

        let losses: Vec<f64> = self.data.iter().map(|d| d.precision_loss).collect();

        // Calculate basic statistics
        let total_operations = self.data.len();
        let average_precision_loss = losses.iter().sum::<f64>() / losses.len() as f64;
        let max_precision_loss = losses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_precision_loss = losses.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // Calculate standard deviation
        let variance = losses.iter().map(|x| (x - average_precision_loss).powi(2)).sum::<f64>()
            / losses.len() as f64;
        let std_precision_loss = variance.sqrt();

        // Count operations with significant loss
        let operations_with_loss = self
            .data
            .iter()
            .filter(|d| d.precision_loss > self.config.loss_threshold)
            .count();
        let loss_percentage = (operations_with_loss as f64 / total_operations as f64) * 100.0;

        // Calculate per-type statistics
        let mut per_type_stats = HashMap::new();
        for op_type in [
            OperationType::Gradient,
            OperationType::Activation,
            OperationType::Weight,
            OperationType::MatMul,
            OperationType::Attention,
            OperationType::LayerNorm,
            OperationType::Other,
        ] {
            let type_data: Vec<&PrecisionData> =
                self.data.iter().filter(|d| d.operation_type == op_type).collect();

            if !type_data.is_empty() {
                let type_losses: Vec<f64> = type_data.iter().map(|d| d.precision_loss).collect();
                let avg_loss = type_losses.iter().sum::<f64>() / type_losses.len() as f64;
                let max_loss = type_losses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let type_variance = type_losses.iter().map(|x| (x - avg_loss).powi(2)).sum::<f64>()
                    / type_losses.len() as f64;
                let std_loss = type_variance.sqrt();

                per_type_stats.insert(
                    op_type,
                    TypeStatistics {
                        count: type_data.len(),
                        avg_loss,
                        max_loss,
                        std_loss,
                    },
                );
            }
        }

        // Create timeline
        let timeline: Vec<TimelinePoint> = self
            .data
            .iter()
            .map(|d| TimelinePoint {
                timestamp: d.timestamp,
                precision_loss: d.precision_loss,
                operation_type: d.operation_type,
                operation: d.operation.clone(),
            })
            .collect();

        Ok(PrecisionStatistics {
            total_operations,
            average_precision_loss,
            max_precision_loss,
            min_precision_loss,
            std_precision_loss,
            operations_with_loss,
            loss_percentage,
            per_type_stats,
            timeline,
        })
    }

    /// Get precision data for a specific operation
    pub fn get_operation_data(&self, operation: &str) -> Vec<&PrecisionData> {
        self.data.iter().filter(|d| d.operation == operation).collect()
    }

    /// Get operations with significant precision loss
    pub fn get_problematic_operations(&self) -> Vec<&PrecisionData> {
        self.data
            .iter()
            .filter(|d| d.precision_loss > self.config.loss_threshold)
            .collect()
    }

    /// Reset tracking data
    pub fn reset(&mut self) {
        self.data.clear();
        self.operation_counts.clear();
    }

    /// Export data to JSON
    pub fn export_to_json(&self) -> Result<String, PrecisionTrackingError> {
        serde_json::to_string_pretty(&self.data)
            .map_err(|e| PrecisionTrackingError::StatisticsError(e.to_string()))
    }

    /// Generate alert for precision loss
    fn generate_alert(&self, data: &PrecisionData) {
        eprintln!(
            "⚠️  PRECISION LOSS ALERT: Operation '{}' lost {:.6} bits of precision ({:.2}% relative loss)",
            data.operation,
            data.precision_loss,
            data.relative_loss * 100.0
        );
    }

    /// Get configuration
    pub fn get_config(&self) -> &PrecisionConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PrecisionConfig) {
        self.config = config;
    }

    /// Get total number of tracked operations
    pub fn total_operations(&self) -> usize {
        self.data.len()
    }

    /// Check if there are any precision issues
    pub fn has_precision_issues(&self) -> bool {
        self.data.iter().any(|d| d.precision_loss > self.config.loss_threshold)
    }
}

impl Default for PrecisionTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience macro for tracking precision in operations
#[macro_export]
macro_rules! track_precision {
    ($tracker:expr, $op_name:expr, $op_type:expr, $input:expr, $output:expr) => {
        if let Err(e) =
            $tracker.track_tensor_precision($op_name.to_string(), $input, $output, $op_type)
        {
            eprintln!("Failed to track precision for {}: {}", $op_name, e);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_tracker_creation() {
        let tracker = PrecisionTracker::new();
        assert_eq!(tracker.total_operations(), 0);
        assert!(!tracker.has_precision_issues());
    }

    #[test]
    fn test_precision_tracking() {
        let mut tracker = PrecisionTracker::new();

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = vec![1.001, 2.001, 3.001, 4.001]; // Small precision loss

        tracker
            .track_precision(
                "test_operation".to_string(),
                &input,
                &output,
                OperationType::MatMul,
            )
            .unwrap();

        assert_eq!(tracker.total_operations(), 1);

        let stats = tracker.get_statistics().unwrap();
        assert_eq!(stats.total_operations, 1);
        assert!(stats.average_precision_loss >= 0.0);
    }

    #[test]
    fn test_precision_tracking_with_loss() {
        let mut tracker = PrecisionTracker::with_config(PrecisionConfig {
            loss_threshold: 1.0,
            enable_alerts: false,
            ..PrecisionConfig::default()
        });

        let input = vec![1000.0, 2000.0, 3000.0];
        let output = vec![1.0, 2.0, 3.0]; // Significant precision loss

        tracker
            .track_precision(
                "lossy_operation".to_string(),
                &input,
                &output,
                OperationType::Activation,
            )
            .unwrap();

        assert_eq!(tracker.total_operations(), 1);
        assert!(tracker.has_precision_issues());

        let problematic = tracker.get_problematic_operations();
        assert_eq!(problematic.len(), 1);
        assert_eq!(problematic[0].operation, "lossy_operation");
    }

    #[test]
    fn test_statistics_calculation() {
        let mut tracker = PrecisionTracker::new();

        // Add multiple operations
        for i in 0..5 {
            let input = vec![10.0 + i as f64, 20.0 + i as f64];
            let output = vec![10.0 + i as f64 + 0.1, 20.0 + i as f64 + 0.1];

            tracker
                .track_precision(
                    format!("op_{}", i),
                    &input,
                    &output,
                    OperationType::Gradient,
                )
                .unwrap();
        }

        let stats = tracker.get_statistics().unwrap();
        assert_eq!(stats.total_operations, 5);
        assert!(stats.per_type_stats.contains_key(&OperationType::Gradient));
        assert_eq!(stats.timeline.len(), 5);
    }

    #[test]
    fn test_operation_data_retrieval() {
        let mut tracker = PrecisionTracker::new();

        let input = vec![1.0, 2.0];
        let output = vec![1.0, 2.0];

        tracker
            .track_precision(
                "specific_op".to_string(),
                &input,
                &output,
                OperationType::Weight,
            )
            .unwrap();

        let op_data = tracker.get_operation_data("specific_op");
        assert_eq!(op_data.len(), 1);
        assert_eq!(op_data[0].operation, "specific_op");
        assert_eq!(op_data[0].operation_type, OperationType::Weight);
    }

    #[test]
    fn test_json_export() {
        let mut tracker = PrecisionTracker::new();

        let input = vec![1.0, 2.0];
        let output = vec![1.0, 2.0];

        tracker
            .track_precision(
                "export_test".to_string(),
                &input,
                &output,
                OperationType::Other,
            )
            .unwrap();

        let json = tracker.export_to_json().unwrap();
        assert!(json.contains("export_test"));
        assert!(json.contains("Other"));
    }

    #[test]
    fn test_config_update() {
        let mut tracker = PrecisionTracker::new();

        let new_config = PrecisionConfig {
            loss_threshold: 0.5,
            sample_rate: 0.8,
            ..PrecisionConfig::default()
        };

        tracker.update_config(new_config.clone());
        assert_eq!(tracker.get_config().loss_threshold, 0.5);
        assert_eq!(tracker.get_config().sample_rate, 0.8);
    }

    #[test]
    fn test_macro() {
        let mut tracker = PrecisionTracker::new();
        let input = [1.0f32, 2.0, 3.0];
        let output = [1.1f32, 2.1, 3.1];

        track_precision!(
            tracker,
            "macro_test",
            OperationType::MatMul,
            &input,
            &output
        );

        assert_eq!(tracker.total_operations(), 1);
    }
}
