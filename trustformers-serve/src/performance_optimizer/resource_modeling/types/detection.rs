//! Detection types for resource modeling
//!
//! Types for hardware detection including caches, vendor-specific detectors,
//! and system capabilities.

use crate::performance_optimizer::types::{
    CacheHierarchy, GpuDeviceModel, MemoryType, NetworkInterface, StorageDevice,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// Cache for hardware detection results
///
/// Comprehensive cache for hardware detection results to minimize
/// redundant detection operations and improve system responsiveness.
#[derive(Debug, Default)]
pub struct HardwareDetectionCache {
    /// CPU frequencies (base, max)
    pub cpu_frequencies: Option<(u32, u32)>,

    /// Cache hierarchy
    pub cache_hierarchy: Option<CacheHierarchy>,

    /// Memory characteristics (type, speed, bandwidth, latency)
    pub memory_characteristics: Option<(MemoryType, u32, f32, Duration)>,

    /// Storage devices
    pub storage_devices: Option<Vec<StorageDevice>>,

    /// Network interfaces
    pub network_interfaces: Option<Vec<NetworkInterface>>,

    /// GPU devices
    pub gpu_devices: Option<Vec<GpuDeviceModel>>,
}

/// Intel hardware detector for Intel-specific optimizations
///
/// Specialized detector for Intel hardware with Intel-specific
/// performance characteristics and optimization capabilities.
#[derive(Debug, Clone, Copy)]
pub struct IntelDetector;

/// AMD hardware detector for AMD-specific optimizations
///
/// Specialized detector for AMD hardware with AMD-specific
/// performance characteristics and optimization capabilities.
#[derive(Debug, Clone, Copy)]
pub struct AmdDetector;

/// NVIDIA hardware detector for NVIDIA GPU detection
///
/// Specialized detector for NVIDIA GPUs with CUDA capabilities,
/// memory detection, and performance characterization.
#[derive(Debug, Clone, Copy)]
pub struct NvidiaDetector;

/// System capabilities assessment
///
/// Comprehensive assessment of system capabilities including
/// hardware features, performance limits, and optimization opportunities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemCapabilities {
    /// Virtualization support
    pub virtualization_support: bool,

    /// Hardware acceleration features
    pub hardware_acceleration: Vec<String>,

    /// Security features
    pub security_features: Vec<String>,

    /// Power management capabilities
    pub power_management: Vec<String>,

    /// Custom capabilities
    pub custom_capabilities: HashMap<String, bool>,
}
