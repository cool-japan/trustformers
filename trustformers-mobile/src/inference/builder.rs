//! Builder for constructing a `MobileInferenceEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{MobileBackend, MobileConfig, MobilePlatform};
use trustformers_core::errors::Result;

use super::engine::MobileInferenceEngine;

/// Mobile inference configuration builder
pub struct MobileInferenceBuilder {
    pub(super) config: MobileConfig,
}
impl MobileInferenceBuilder {
    /// Create new builder with default mobile configuration
    pub fn new() -> Self {
        Self {
            config: MobileConfig::default(),
        }
    }
    /// Set target platform
    pub fn platform(mut self, platform: MobilePlatform) -> Self {
        self.config.platform = platform;
        self
    }
    /// Set inference backend
    pub fn backend(mut self, backend: MobileBackend) -> Self {
        self.config.backend = backend;
        self
    }
    /// Set memory limit
    pub fn memory_limit_mb(mut self, limit: usize) -> Self {
        self.config.max_memory_mb = limit;
        self
    }
    /// Enable/disable FP16 precision
    pub fn fp16(mut self, enable: bool) -> Self {
        self.config.use_fp16 = enable;
        self
    }
    /// Set quantization scheme
    pub fn quantization(mut self, scheme: crate::MobileQuantizationScheme) -> Self {
        self.config.quantization = Some(crate::MobileQuantizationConfig {
            scheme,
            dynamic: true,
            per_channel: false,
        });
        self
    }
    /// Set thread count
    pub fn threads(mut self, count: usize) -> Self {
        self.config.num_threads = count;
        self
    }
    /// Enable/disable batching
    pub fn batching(mut self, enable: bool, max_batch_size: usize) -> Self {
        self.config.enable_batching = enable;
        self.config.max_batch_size = max_batch_size;
        self
    }
    /// Set memory optimization level
    pub fn memory_optimization(mut self, level: crate::MemoryOptimization) -> Self {
        self.config.memory_optimization = level;
        self
    }
    /// Build the inference engine
    pub fn build(self) -> Result<MobileInferenceEngine> {
        MobileInferenceEngine::new(self.config)
    }
}
impl Default for MobileInferenceBuilder {
    fn default() -> Self {
        Self::new()
    }
}
