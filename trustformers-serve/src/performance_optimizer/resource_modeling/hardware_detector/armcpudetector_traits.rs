//! # ArmCpuDetector - Trait Implementations
//!
//! This module contains trait implementations for `ArmCpuDetector`.
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
use super::types::ArmCpuDetector;

impl Default for ArmCpuDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CpuVendorDetector for ArmCpuDetector {
    fn vendor_name(&self) -> &str {
        "ARM"
    }
    async fn detect_vendor_features(&self) -> Result<CpuVendorFeatures> {
        Ok(CpuVendorFeatures::default())
    }
}
