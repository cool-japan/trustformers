//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use async_trait::async_trait;
use std::time::Duration;

use super::super::types::SystemResourceSnapshot;
use super::types::{
    CollectedData, CollectionContext, ProfilingStageType, StageExecutionContext,
    StageResourceRequirements, StageResult,
};

/// Data collector trait
pub trait DataCollector {
    /// Collect data
    fn collect(&self) -> Result<CollectedData>;
    /// Get collector name
    fn name(&self) -> &str;
    /// Check if collector is active
    fn is_active(&self) -> bool;
    /// Configure collector
    fn configure(&mut self, config: serde_json::Value) -> Result<()>;
}
/// Collection strategy trait
pub trait CollectionStrategy {
    /// Determine if data should be collected
    fn should_collect(&self, context: &CollectionContext) -> bool;
    /// Get sampling rate for current context
    fn get_sampling_rate(&self, context: &CollectionContext) -> f32;
    /// Get strategy name
    fn name(&self) -> &str;
}
/// Profiling stage trait (enhanced version)
#[async_trait]
pub trait ProfilingStage {
    /// Execute the profiling stage
    async fn execute(&self, context: &StageExecutionContext) -> Result<StageResult>;
    /// Get stage name
    fn name(&self) -> &str;
    /// Get stage type
    fn stage_type(&self) -> ProfilingStageType;
    /// Check if stage is applicable for given context
    fn is_applicable(&self, context: &StageExecutionContext) -> bool;
    /// Get stage dependencies
    fn dependencies(&self) -> Vec<ProfilingStageType>;
    /// Estimate execution time
    fn estimated_duration(&self, context: &StageExecutionContext) -> Duration;
    /// Get resource requirements
    fn resource_requirements(&self) -> StageResourceRequirements;
    /// Validate stage prerequisites
    async fn validate_prerequisites(&self, context: &StageExecutionContext) -> Result<()>;
    /// Cleanup after execution
    async fn cleanup(&self, context: &StageExecutionContext) -> Result<()>;
}
impl SystemResourceSnapshot {
    pub(crate) async fn current() -> Result<Self> {
        Ok(Self {
            snapshot_timestamp: chrono::Utc::now(),
            cpu_usage: 0.5,
            memory_usage: 1024 * 1024 * 1024,
            io_activity: 0.3,
            network_activity: 0.2,
            disk_usage: 500 * 1024 * 1024,
            network_usage: 10 * 1024 * 1024,
            io_capacity: 0.7,
        })
    }
}
