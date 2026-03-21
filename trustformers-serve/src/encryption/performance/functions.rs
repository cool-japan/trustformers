//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use super::types::{
    PerformanceConfig, HardwareAcceleration, PerformanceCachingConfig,
    ParallelProcessingConfig, MemoryManagementConfig, EncryptionAlgorithm,
};

use super::types::{AccelerationMetrics, BenchmarkRunner, CacheManager, HardwareFeatures, OperationType, PerformanceManager, PressureLevel};

/// Acceleration provider trait
pub trait AccelerationProvider {
    /// Provider name
    fn name(&self) -> &str;
    /// Check if acceleration is available
    fn is_available(&self) -> bool;
    /// Initialize acceleration
    async fn initialize(&self) -> Result<()>;
    /// Accelerated encryption
    async fn accelerated_encrypt(
        &self,
        data: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> Result<Vec<u8>>;
    /// Accelerated decryption
    async fn accelerated_decrypt(
        &self,
        data: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> Result<Vec<u8>>;
    /// Get performance metrics
    fn get_metrics(&self) -> AccelerationMetrics;
}
/// Pressure handler trait
pub trait PressureHandler {
    /// Handle memory pressure
    async fn handle_pressure(&self, level: PressureLevel) -> Result<()>;
    /// Handler priority
    fn priority(&self) -> u32;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_performance_manager_creation() {
        let config = PerformanceConfig::default();
        let performance_manager = PerformanceManager::new(config);
        assert!(performance_manager.config.enabled);
    }
    #[tokio::test]
    async fn test_cache_manager() {
        let config = PerformanceCachingConfig::default();
        let cache_manager = CacheManager::new(config);
        cache_manager.start().await.expect("async operation should succeed in test");
        let result = cache_manager
            .get_cached_result(OperationType::Encryption, b"test_data")
            .await;
        assert!(result.is_ok());
    }
    #[tokio::test]
    async fn test_hardware_features_detection() {
        let features = HardwareFeatures::detect();
        assert!(features.cpu_cores > 0);
    }
    #[tokio::test]
    async fn test_benchmark_runner() {
        let runner = BenchmarkRunner::new();
        let result = runner.run_benchmark("test_suite").await;
        assert!(result.is_ok());
    }
}
