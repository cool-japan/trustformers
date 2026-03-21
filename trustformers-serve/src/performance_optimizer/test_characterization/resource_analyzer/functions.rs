//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};

use super::super::types::{
    DataCharacteristics, ResourceIntensity, ResourceUsageDataPoint, ResourceUsageSnapshot,
};
use super::types::AlgorithmPerformanceRecord;

/// Trait for intensity calculation algorithms
pub trait IntensityCalculationAlgorithm: std::fmt::Debug {
    /// Calculate resource intensity for the given usage data
    fn calculate_intensity(
        &self,
        usage_data: &[ResourceUsageDataPoint],
    ) -> Result<ResourceIntensity>;
    /// Get algorithm name
    fn name(&self) -> &str;
    /// Get algorithm description
    fn description(&self) -> &str;
    /// Check if algorithm is suitable for the given data characteristics
    fn is_suitable_for(&self, characteristics: &DataCharacteristics) -> bool;
}
/// Trait for algorithm selection strategies
pub trait SelectionStrategy: std::fmt::Debug {
    /// Select the best algorithm for the given characteristics and performance history
    fn select_algorithm(
        &self,
        characteristics: &DataCharacteristics,
        performance_history: &HashMap<String, VecDeque<AlgorithmPerformanceRecord>>,
    ) -> Result<String>;
    /// Get strategy name
    fn name(&self) -> &str;
    /// Get strategy description
    fn description(&self) -> &str;
}
/// Trait for system resource monitoring
#[async_trait]
pub trait SystemResourceMonitor: std::fmt::Debug {
    /// Collect current system resource usage
    async fn collect_resources(&self) -> Result<ResourceUsageSnapshot>;
}
