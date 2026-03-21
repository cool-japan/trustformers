//! # AmdCpuDetector - Trait Implementations
//!
//! This module contains trait implementations for `AmdCpuDetector`.
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
use super::types::AmdCpuDetector;

impl Default for AmdCpuDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CpuVendorDetector for AmdCpuDetector {
    fn vendor_name(&self) -> &str {
        "AuthenticAMD"
    }
    async fn detect_vendor_features(&self) -> Result<CpuVendorFeatures> {
        Ok(CpuVendorFeatures::default())
    }
}
