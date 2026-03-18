//! # BottleneckType - Trait Implementations
//!
//! This module contains trait implementations for `BottleneckType`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::BottleneckType;

impl fmt::Display for BottleneckType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BottleneckType::Cpu => write!(f, "CPU"),
            BottleneckType::Memory => write!(f, "Memory"),
            BottleneckType::Gpu => write!(f, "GPU"),
            BottleneckType::NetworkIo => write!(f, "Network I/O"),
            BottleneckType::FilesystemIo => write!(f, "Filesystem I/O"),
            BottleneckType::Database => write!(f, "Database"),
            BottleneckType::Synchronization => write!(f, "Synchronization"),
            BottleneckType::ResourceContention => write!(f, "Resource Contention"),
        }
    }
}
