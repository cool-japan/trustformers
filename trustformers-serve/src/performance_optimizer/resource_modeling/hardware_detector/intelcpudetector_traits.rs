//! # IntelCpuDetector - Trait Implementations
//!
//! This module contains trait implementations for `IntelCpuDetector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `CpuVendorDetector`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::CpuVendorFeatures;
use anyhow::Result;
use async_trait::async_trait;

use super::functions::CpuVendorDetector;
use super::types::IntelCpuDetector;

impl Default for IntelCpuDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CpuVendorDetector for IntelCpuDetector {
    fn vendor_name(&self) -> &str {
        "GenuineIntel"
    }
    async fn detect_vendor_features(&self) -> Result<CpuVendorFeatures> {
        Ok(CpuVendorFeatures::default())
    }
}
