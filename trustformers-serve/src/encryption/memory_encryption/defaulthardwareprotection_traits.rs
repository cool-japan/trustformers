//! # DefaultHardwareProtection - Trait Implementations
//!
//! This module contains trait implementations for `DefaultHardwareProtection`.
//!
//! ## Implemented Traits
//!
//! - `HardwareProtectionInterface`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use uuid::Uuid;

use super::types::{DefaultHardwareProtection, HardwareProtectionHandle, HardwareType, ProtectionLevel};
use super::functions::HardwareProtectionInterface;

impl HardwareProtectionInterface for DefaultHardwareProtection {
    async fn enable_protection(
        &self,
        _address: *mut u8,
        _size: usize,
        _level: ProtectionLevel,
    ) -> Result<HardwareProtectionHandle> {
        Ok(HardwareProtectionHandle {
            id: Uuid::new_v4().to_string(),
            hardware_type: HardwareType::HSM,
            protection_level: _level,
            handle_data: Vec::new(),
        })
    }
    async fn disable_protection(&self, _handle: HardwareProtectionHandle) -> Result<()> {
        Ok(())
    }
    fn is_available(&self) -> bool {
        false
    }
    fn supported_levels(&self) -> Vec<ProtectionLevel> {
        vec![ProtectionLevel::Basic]
    }
}

