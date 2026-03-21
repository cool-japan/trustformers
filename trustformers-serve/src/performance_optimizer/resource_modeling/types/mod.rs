//! Comprehensive Types Module for Resource Modeling System
//!
//! This module contains all types extracted from the resource modeling system,
//! organized into logical categories for optimal maintainability and comprehension.
//! The resource modeling system provides comprehensive hardware detection, performance
//! profiling, thermal monitoring, and topology analysis capabilities.
//!
//! # Features
//!
//! * **Core Configuration Types**: Configuration structs for all aspects of resource modeling
//! * **Hardware Model Types**: Detailed hardware characterization and modeling types
//! * **Monitoring Types**: Real-time monitoring and tracking infrastructure
//! * **Detection Types**: Hardware detection and vendor-specific capabilities
//! * **Profiling Types**: Performance profiling and benchmarking systems
//! * **Topology Types**: System topology analysis and NUMA optimization
//! * **Utility Types**: Supporting types for resource management
//! * **Enums**: State and type enumerations for resource modeling
//! * **Error Types**: Comprehensive error handling for resource operations
//! * **Trait Definitions**: Extensible interfaces for hardware detection

// Import and re-export types from the main performance optimizer types module
pub use crate::performance_optimizer::types::{
    CacheHierarchy, CpuModel, CpuPerformanceCharacteristics, GpuDeviceModel, GpuModel,
    GpuUtilizationCharacteristics, IoModel, MemoryModel, MemoryType, NetworkInterface,
    NetworkInterfaceStatus, NetworkInterfaceType, NetworkModel, NumaTopology, StorageDevice,
    StorageDeviceType, SystemResourceModel, SystemState, TemperatureMetrics,
};

// Import ResourceModelingConfig from manager module (canonical definition)
pub use super::manager::ResourceModelingConfig;

// Module declarations
pub mod config;
pub mod detection;
pub mod enums;
pub mod error;
pub mod hardware;
pub mod monitoring;
pub mod profiling;
pub mod topology;
pub mod traits;
pub mod traits_analysis;
pub mod traits_profiling;
pub mod utility;

// Re-export all public types from submodules
pub use config::*;
pub use detection::*;
pub use enums::*;
pub use error::*;
pub use hardware::*;
pub use monitoring::*;
pub use profiling::*;
pub use topology::*;
pub use traits::*;
// traits_analysis and traits_profiling are re-exported via traits.rs
pub use utility::*;
