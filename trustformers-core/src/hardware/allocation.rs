// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Hardware resource allocation and management components
//!
//! This module provides resource allocation strategies, load balancing, memory management,
//! and memory pressure monitoring for hardware devices.

use super::config::{AllocationStrategy, DeviceInfo, LoadBalancingStrategy, MemoryUsageStats};
use super::traits::{MemoryType, OperationParameter};
use super::HardwareResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Resource allocator for managing hardware resource assignments
#[derive(Debug, Clone)]
pub struct ResourceAllocator {
    /// Current allocation strategy
    pub strategy: AllocationStrategy,
    /// Active resource reservations
    pub reservations: HashMap<String, ResourceReservation>,
    /// Historical allocation records
    pub history: Vec<AllocationRecord>,
    /// Resource limits per device
    pub limits: HashMap<String, ResourceLimits>,
    /// Round-robin cursor over the most recently seen device list.
    round_robin_cursor: usize,
}

/// Resource reservation details
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceReservation {
    /// Target device ID
    pub device_id: String,
    /// Reserved resource amounts by type
    pub resources: HashMap<String, f64>,
    /// Reservation creation timestamp
    pub timestamp: SystemTime,
    /// Optional expiration time
    pub expiration: Option<SystemTime>,
    /// Unique reservation identifier
    pub id: String,
}

/// Allocation record for auditing and analytics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllocationRecord {
    /// Allocated device ID
    pub device_id: String,
    /// Allocation timestamp
    pub timestamp: SystemTime,
    /// Duration of allocation
    pub duration: std::time::Duration,
    /// Resources allocated
    pub resources: HashMap<String, f64>,
    /// Operation parameters
    pub operation_params: Vec<OperationParameter>,
    /// Success indicator
    pub success: bool,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

/// Resource limits configuration per device
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU utilization (0.0 - 1.0)
    pub max_cpu: f64,
    /// Maximum memory utilization (0.0 - 1.0)
    pub max_memory: f64,
    /// Maximum GPU utilization (0.0 - 1.0)
    pub max_gpu: f64,
    /// Maximum power consumption (watts)
    pub max_power: f64,
    /// Maximum bandwidth (bytes/sec)
    pub max_bandwidth: f64,
    /// Custom resource limits
    pub custom_limits: HashMap<String, f64>,
}

/// Load balancer for distributing work across devices
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    /// Active load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Device weights for weighted algorithms
    pub weights: HashMap<String, f64>,
    /// Connection counts per device
    pub connections: HashMap<String, u64>,
    /// Load history for trend analysis
    pub load_history: HashMap<String, Vec<(SystemTime, f64)>>,
    /// Adaptive thresholds for dynamic balancing
    pub adaptive_thresholds: HashMap<String, f64>,
    /// Round-robin cursor over the most recently seen device list.
    round_robin_cursor: usize,
    /// Fractional credit accumulator per device for smooth weighted
    /// round-robin (classic "current weight" scheduling algorithm).
    weighted_credits: HashMap<String, f64>,
}

/// Memory manager for device memory pools and allocation
#[derive(Debug, Clone)]
pub struct MemoryManager {
    /// Memory pools per device
    pub pools: HashMap<String, MemoryPool>,
    /// Memory usage tracking
    pub usage_tracking: HashMap<String, MemoryUsageStats>,
    /// Garbage collection schedules
    pub gc_schedule: HashMap<String, SystemTime>,
    /// Memory pressure monitor
    pub pressure_monitor: MemoryPressureMonitor,
}

/// Memory pool representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryPool {
    /// Pool identifier
    pub id: String,
    /// Associated device ID
    pub device_id: String,
    /// Total pool size in bytes
    pub total_size: usize,
    /// Currently used size in bytes
    pub used_size: usize,
    /// Available size in bytes
    pub available_size: usize,
    /// Allocated memory blocks
    pub allocated_blocks: Vec<MemoryBlock>,
    /// Free memory blocks
    pub free_blocks: Vec<MemoryBlock>,
    /// Memory fragmentation ratio (0.0 - 1.0)
    pub fragmentation_ratio: f64,
}

/// Memory block allocation unit
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryBlock {
    /// Block identifier
    pub id: String,
    /// Memory offset
    pub offset: usize,
    /// Block size in bytes
    pub size: usize,
    /// Memory type
    pub memory_type: MemoryType,
    /// Allocation timestamp
    pub allocated_at: SystemTime,
    /// Optional tags for categorization
    pub tags: Vec<String>,
}

/// Memory pressure monitor for tracking memory pressure levels
#[derive(Debug, Clone)]
pub struct MemoryPressureMonitor {
    /// Current pressure levels per device
    pub pressure_levels: HashMap<String, MemoryPressureLevel>,
    /// Historical pressure data
    pub pressure_history: HashMap<String, Vec<(SystemTime, f64)>>,
    /// Pressure thresholds configuration
    pub thresholds: HashMap<String, MemoryPressureThresholds>,
}

/// Memory pressure level indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    /// Low pressure - optimal conditions
    Low,
    /// Medium pressure - some concern
    Medium,
    /// High pressure - action recommended
    High,
    /// Critical pressure - immediate action required
    Critical,
}

/// Memory pressure threshold configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryPressureThresholds {
    /// Low pressure threshold (0.0 - 1.0)
    pub low: f64,
    /// Medium pressure threshold (0.0 - 1.0)
    pub medium: f64,
    /// High pressure threshold (0.0 - 1.0)
    pub high: f64,
    /// Critical pressure threshold (0.0 - 1.0)
    pub critical: f64,
}

impl ResourceAllocator {
    /// Create a new resource allocator
    pub fn new(strategy: AllocationStrategy) -> Self {
        Self {
            strategy,
            reservations: HashMap::new(),
            history: Vec::new(),
            limits: HashMap::new(),
            round_robin_cursor: 0,
        }
    }

    /// Allocate resources on the best available device, selected for real
    /// from `available_devices` according to `self.strategy`. Returns an
    /// error (recorded in history as a failed allocation) when no device
    /// satisfies `requirements`, rather than fabricating a device ID that
    /// corresponds to nothing.
    pub fn allocate(
        &mut self,
        requirements: &HashMap<String, f64>,
        available_devices: &[DeviceInfo],
    ) -> HardwareResult<String> {
        let selected = match self.strategy {
            AllocationStrategy::FirstAvailable => available_devices.first(),
            AllocationStrategy::BestFit => {
                self.find_best_fit_device(requirements, available_devices)
            },
            AllocationStrategy::RoundRobin => self.next_round_robin_device(available_devices),
            AllocationStrategy::LoadAware => self.find_least_loaded_device(available_devices),
            AllocationStrategy::PerformanceOptimized => {
                self.find_highest_performance_device(available_devices)
            },
            AllocationStrategy::PowerEfficient => {
                self.find_most_power_efficient_device(available_devices)
            },
        };

        let Some(device) = selected else {
            let record = AllocationRecord {
                device_id: String::new(),
                timestamp: SystemTime::now(),
                duration: std::time::Duration::from_secs(0),
                resources: requirements.clone(),
                operation_params: vec![],
                success: false,
                performance_metrics: HashMap::new(),
            };
            self.history.push(record);
            return Err(super::TrustformersError::hardware_error(
                "No available device satisfies the allocation requirements",
                "allocate",
            ));
        };

        let device_id = device.id.clone();

        // Record the allocation
        let record = AllocationRecord {
            device_id: device_id.clone(),
            timestamp: SystemTime::now(),
            duration: std::time::Duration::from_secs(0), // Will be updated on completion
            resources: requirements.clone(),
            operation_params: vec![],
            success: true,
            performance_metrics: HashMap::new(),
        };
        self.history.push(record);

        Ok(device_id)
    }

    /// Find the device with the smallest free memory that still satisfies
    /// `requirements["memory"]` (classic best-fit bin packing: minimizes
    /// wasted capacity rather than grabbing the first sufficient device).
    /// Falls back to the device with the most free memory when no memory
    /// requirement is specified.
    fn find_best_fit_device<'a>(
        &self,
        requirements: &HashMap<String, f64>,
        available_devices: &'a [DeviceInfo],
    ) -> Option<&'a DeviceInfo> {
        match requirements.get("memory").copied() {
            Some(needed) => available_devices
                .iter()
                .filter(|d| d.status.memory_usage.free as f64 >= needed)
                .min_by(|a, b| a.status.memory_usage.free.cmp(&b.status.memory_usage.free)),
            None => available_devices.iter().max_by_key(|d| d.status.memory_usage.free),
        }
    }

    /// Get next device in round-robin order, cycling through
    /// `available_devices` using a cursor carried across calls.
    fn next_round_robin_device<'a>(
        &mut self,
        available_devices: &'a [DeviceInfo],
    ) -> Option<&'a DeviceInfo> {
        if available_devices.is_empty() {
            return None;
        }
        let idx = self.round_robin_cursor % available_devices.len();
        self.round_robin_cursor = self.round_robin_cursor.wrapping_add(1);
        available_devices.get(idx)
    }

    /// Find device with the lowest current reported utilization.
    fn find_least_loaded_device<'a>(
        &self,
        available_devices: &'a [DeviceInfo],
    ) -> Option<&'a DeviceInfo> {
        available_devices
            .iter()
            .min_by(|a, b| a.status.utilization.total_cmp(&b.status.utilization))
    }

    /// Find device with the highest advertised compute unit count.
    fn find_highest_performance_device<'a>(
        &self,
        available_devices: &'a [DeviceInfo],
    ) -> Option<&'a DeviceInfo> {
        available_devices
            .iter()
            .max_by_key(|d| d.capabilities.compute_units.unwrap_or(0))
    }

    /// Find device with the lowest advertised power consumption. Devices
    /// with unknown power consumption are deprioritized (treated as
    /// infinite) rather than assumed efficient.
    fn find_most_power_efficient_device<'a>(
        &self,
        available_devices: &'a [DeviceInfo],
    ) -> Option<&'a DeviceInfo> {
        available_devices.iter().min_by(|a, b| {
            let pa = a.capabilities.power_consumption.unwrap_or(f64::INFINITY);
            let pb = b.capabilities.power_consumption.unwrap_or(f64::INFINITY);
            pa.total_cmp(&pb)
        })
    }

    /// Set resource limits for a device
    pub fn set_limits(&mut self, device_id: &str, limits: ResourceLimits) {
        self.limits.insert(device_id.to_string(), limits);
    }

    /// Get allocation history
    pub fn get_history(&self) -> &[AllocationRecord] {
        &self.history
    }
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            weights: HashMap::new(),
            connections: HashMap::new(),
            load_history: HashMap::new(),
            adaptive_thresholds: HashMap::new(),
            round_robin_cursor: 0,
            weighted_credits: HashMap::new(),
        }
    }

    /// Select next device based on load balancing strategy
    pub fn select_device(&mut self, available_devices: &[String]) -> HardwareResult<String> {
        if available_devices.is_empty() {
            return Err(super::TrustformersError::hardware_error(
                "No devices available",
                "allocate",
            ));
        }

        let selected = match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.round_robin_select(available_devices),
            LoadBalancingStrategy::LeastConnections => {
                self.least_connections_select(available_devices)
            },
            LoadBalancingStrategy::LeastUtilization => {
                self.least_utilization_select(available_devices)
            },
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.weighted_round_robin_select(available_devices)
            },
            LoadBalancingStrategy::PerformanceBased => {
                self.performance_based_select(available_devices)
            },
            LoadBalancingStrategy::Adaptive => self.adaptive_select(available_devices),
        };

        // Update connection count
        *self.connections.entry(selected.clone()).or_insert(0) += 1;

        Ok(selected)
    }

    /// Cycle through `devices` using a cursor carried across calls, so
    /// repeated calls with the same device list actually round-robin
    /// instead of always returning the first entry.
    fn round_robin_select(&mut self, devices: &[String]) -> String {
        let idx = self.round_robin_cursor % devices.len();
        self.round_robin_cursor = self.round_robin_cursor.wrapping_add(1);
        devices[idx].clone()
    }

    fn least_connections_select(&self, devices: &[String]) -> String {
        devices
            .iter()
            .min_by_key(|device| self.connections.get(*device).unwrap_or(&0))
            .cloned()
            .unwrap_or_default()
    }

    /// Pick the device with the lowest most-recent utilization sample in
    /// `self.load_history`. Devices with no recorded history are treated as
    /// unknown load (deprioritized below any device with a known, lower
    /// reading) rather than assumed idle.
    fn least_utilization_select(&self, devices: &[String]) -> String {
        devices
            .iter()
            .min_by(|a, b| self.latest_utilization(a).total_cmp(&self.latest_utilization(b)))
            .cloned()
            .unwrap_or_default()
    }

    fn latest_utilization(&self, device: &str) -> f64 {
        self.load_history
            .get(device)
            .and_then(|history| history.last())
            .map(|(_, utilization)| *utilization)
            .unwrap_or(f64::INFINITY)
    }

    /// Smooth weighted round-robin (the algorithm used by nginx/LVS):
    /// each device accrues credit equal to its configured `weights` entry
    /// every call; the device with the highest accumulated credit is
    /// selected and has `total_weight` deducted, so devices with higher
    /// weight are picked more often while every device still gets a turn.
    fn weighted_round_robin_select(&mut self, devices: &[String]) -> String {
        let total_weight: f64 =
            devices.iter().map(|d| self.weights.get(d).copied().unwrap_or(1.0)).sum();

        for device in devices {
            let weight = self.weights.get(device).copied().unwrap_or(1.0);
            *self.weighted_credits.entry(device.clone()).or_insert(0.0) += weight;
        }

        let selected = devices
            .iter()
            .max_by(|a, b| {
                let ca = self.weighted_credits.get(*a).copied().unwrap_or(0.0);
                let cb = self.weighted_credits.get(*b).copied().unwrap_or(0.0);
                ca.total_cmp(&cb)
            })
            .cloned()
            .unwrap_or_default();

        if let Some(credit) = self.weighted_credits.get_mut(&selected) {
            *credit -= total_weight.max(f64::MIN_POSITIVE);
        }

        selected
    }

    /// Pick the device with the highest configured weight, treated here as
    /// a relative performance rating (distinct from `load_history`, which
    /// `least_utilization_select` already uses).
    fn performance_based_select(&self, devices: &[String]) -> String {
        devices
            .iter()
            .max_by(|a, b| {
                let wa = self.weights.get(*a).copied().unwrap_or(0.0);
                let wb = self.weights.get(*b).copied().unwrap_or(0.0);
                wa.total_cmp(&wb)
            })
            .cloned()
            .unwrap_or_default()
    }

    /// Pick the device with the most headroom below its own configured
    /// adaptive threshold (`self.adaptive_thresholds[device] -
    /// current_utilization`), combining both fields the struct already
    /// carries. Devices without a configured threshold default to 1.0
    /// (fully open); devices without utilization history are treated as
    /// unknown load and deprioritized.
    fn adaptive_select(&self, devices: &[String]) -> String {
        devices
            .iter()
            .max_by(|a, b| self.adaptive_headroom(a).total_cmp(&self.adaptive_headroom(b)))
            .cloned()
            .unwrap_or_default()
    }

    fn adaptive_headroom(&self, device: &str) -> f64 {
        let threshold = self.adaptive_thresholds.get(device).copied().unwrap_or(1.0);
        let utilization = self.latest_utilization(device);
        if utilization.is_infinite() {
            f64::NEG_INFINITY
        } else {
            threshold - utilization
        }
    }

    /// Update device weight
    pub fn set_weight(&mut self, device_id: &str, weight: f64) {
        self.weights.insert(device_id.to_string(), weight);
    }
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            usage_tracking: HashMap::new(),
            gc_schedule: HashMap::new(),
            pressure_monitor: MemoryPressureMonitor::new(),
        }
    }

    /// Allocate memory block
    pub fn allocate_memory(
        &mut self,
        device_id: &str,
        size: usize,
        memory_type: MemoryType,
    ) -> HardwareResult<MemoryBlock> {
        let pool = self
            .pools
            .entry(device_id.to_string())
            .or_insert_with(|| MemoryPool::new(device_id));

        pool.allocate(size, memory_type)
    }

    /// Deallocate memory block
    pub fn deallocate_memory(&mut self, device_id: &str, block_id: &str) -> HardwareResult<()> {
        if let Some(pool) = self.pools.get_mut(device_id) {
            pool.deallocate(block_id)
        } else {
            Err(super::TrustformersError::hardware_error(
                "Device not found",
                "deallocate",
            ))
        }
    }

    /// Trigger garbage collection for a device
    pub fn trigger_gc(&mut self, device_id: &str) -> HardwareResult<()> {
        if let Some(pool) = self.pools.get_mut(device_id) {
            pool.garbage_collect()?;
            self.gc_schedule.insert(device_id.to_string(), SystemTime::now());
        }
        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_usage_stats(&self, device_id: &str) -> Option<&MemoryUsageStats> {
        self.usage_tracking.get(device_id)
    }
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(device_id: &str) -> Self {
        Self {
            id: format!("pool_{}", device_id),
            device_id: device_id.to_string(),
            total_size: 1024 * 1024 * 1024, // 1GB default
            used_size: 0,
            available_size: 1024 * 1024 * 1024,
            allocated_blocks: Vec::new(),
            free_blocks: Vec::new(),
            fragmentation_ratio: 0.0,
        }
    }

    /// Allocate a memory block
    pub fn allocate(
        &mut self,
        size: usize,
        memory_type: MemoryType,
    ) -> HardwareResult<MemoryBlock> {
        if self.available_size < size {
            return Err(super::TrustformersError::hardware_error(
                "Insufficient memory",
                "allocate",
            ));
        }

        let block = MemoryBlock {
            id: format!(
                "block_{}_{}",
                self.allocated_blocks.len(),
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ),
            offset: self.used_size,
            size,
            memory_type,
            allocated_at: SystemTime::now(),
            tags: vec![],
        };

        self.allocated_blocks.push(block.clone());
        self.used_size += size;
        self.available_size -= size;

        Ok(block)
    }

    /// Deallocate a memory block
    pub fn deallocate(&mut self, block_id: &str) -> HardwareResult<()> {
        if let Some(pos) = self.allocated_blocks.iter().position(|b| b.id == block_id) {
            let block = self.allocated_blocks.remove(pos);
            self.used_size -= block.size;
            self.available_size += block.size;
            self.free_blocks.push(block);
            Ok(())
        } else {
            Err(super::TrustformersError::hardware_error(
                "Block not found",
                "deallocate",
            ))
        }
    }

    /// Perform garbage collection
    pub fn garbage_collect(&mut self) -> HardwareResult<()> {
        // Coalesce free blocks and update fragmentation ratio
        self.free_blocks.sort_by_key(|b| b.offset);
        // Implementation would coalesce adjacent free blocks
        self.fragmentation_ratio = self.calculate_fragmentation();
        Ok(())
    }

    fn calculate_fragmentation(&self) -> f64 {
        if self.free_blocks.is_empty() {
            return 0.0;
        }
        // Simplified fragmentation calculation
        self.free_blocks.len() as f64 / (self.total_size / 1024) as f64
    }
}

impl MemoryPressureMonitor {
    /// Create a new memory pressure monitor
    pub fn new() -> Self {
        Self {
            pressure_levels: HashMap::new(),
            pressure_history: HashMap::new(),
            thresholds: HashMap::new(),
        }
    }

    /// Update pressure level for a device
    pub fn update_pressure(&mut self, device_id: &str, utilization: f64) {
        let default_thresholds = MemoryPressureThresholds::default();
        let thresholds = self.thresholds.get(device_id).unwrap_or(&default_thresholds);

        let level = if utilization < thresholds.low {
            MemoryPressureLevel::Low
        } else if utilization < thresholds.medium {
            MemoryPressureLevel::Medium
        } else if utilization < thresholds.high {
            MemoryPressureLevel::High
        } else {
            MemoryPressureLevel::Critical
        };

        self.pressure_levels.insert(device_id.to_string(), level);

        // Record in history
        let entry = self.pressure_history.entry(device_id.to_string()).or_default();
        entry.push((SystemTime::now(), utilization));

        // Keep only last 1000 entries
        if entry.len() > 1000 {
            entry.drain(..500);
        }
    }

    /// Get current pressure level
    pub fn get_pressure_level(&self, device_id: &str) -> Option<MemoryPressureLevel> {
        self.pressure_levels.get(device_id).copied()
    }

    /// Set pressure thresholds for a device
    pub fn set_thresholds(&mut self, device_id: &str, thresholds: MemoryPressureThresholds) {
        self.thresholds.insert(device_id.to_string(), thresholds);
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu: 0.8,
            max_memory: 0.9,
            max_gpu: 0.95,
            max_power: 300.0,
            max_bandwidth: 10_000_000_000.0, // 10 GB/s
            custom_limits: HashMap::new(),
        }
    }
}

impl Default for MemoryPressureThresholds {
    fn default() -> Self {
        Self {
            low: 0.5,
            medium: 0.7,
            high: 0.85,
            critical: 0.95,
        }
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MemoryPressureMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::traits::{DeviceStatus, MemoryUsage};
    use super::super::{DataType, HardwareCapabilities, HardwareType};
    use super::*;

    fn device(
        id: &str,
        free_memory: usize,
        utilization: f64,
        compute_units: Option<u32>,
        power: Option<f64>,
    ) -> DeviceInfo {
        DeviceInfo {
            id: id.to_string(),
            hardware_type: HardwareType::CPU,
            capabilities: HardwareCapabilities {
                data_types: vec![DataType::F32],
                max_dimensions: 4,
                memory_size: Some(free_memory),
                clock_frequency: None,
                compute_units,
                operations: vec![],
                power_consumption: power,
                thermal_design_power: None,
            },
            status: DeviceStatus {
                online: true,
                busy: false,
                error: None,
                memory_usage: MemoryUsage {
                    used: 0,
                    total: free_memory,
                    free: free_memory,
                    fragmentation: 0.0,
                },
                temperature: None,
                power_consumption: power,
                utilization,
            },
            last_seen: SystemTime::now(),
            weight: 1.0,
            priority: 0,
            tags: vec![],
        }
    }

    /// Regression test: `allocate` used to return the literal string
    /// `"device_0"` for `FirstAvailable` regardless of what devices (if
    /// any) actually existed. It must now return a real device's id, and
    /// error when there is nothing to allocate.
    #[test]
    fn test_allocate_first_available_returns_real_device_id() {
        let mut allocator = ResourceAllocator::new(AllocationStrategy::FirstAvailable);
        let devices = vec![device("real-device-7", 1024, 0.1, Some(4), Some(50.0))];
        let id = allocator
            .allocate(&HashMap::new(), &devices)
            .expect("allocation should succeed");
        assert_eq!(id, "real-device-7");
    }

    #[test]
    fn test_allocate_errors_when_no_devices_available() {
        let mut allocator = ResourceAllocator::new(AllocationStrategy::FirstAvailable);
        let result = allocator.allocate(&HashMap::new(), &[]);
        assert!(
            result.is_err(),
            "must error rather than fabricate a device id"
        );
    }

    #[test]
    fn test_allocate_load_aware_picks_least_utilized_device() {
        let mut allocator = ResourceAllocator::new(AllocationStrategy::LoadAware);
        let devices = vec![
            device("busy", 1024, 0.9, None, None),
            device("idle", 1024, 0.05, None, None),
            device("medium", 1024, 0.5, None, None),
        ];
        let id = allocator
            .allocate(&HashMap::new(), &devices)
            .expect("allocation should succeed");
        assert_eq!(id, "idle");
    }

    #[test]
    fn test_allocate_performance_optimized_picks_most_compute_units() {
        let mut allocator = ResourceAllocator::new(AllocationStrategy::PerformanceOptimized);
        let devices = vec![
            device("small", 1024, 0.0, Some(4), None),
            device("big", 1024, 0.0, Some(64), None),
        ];
        let id = allocator
            .allocate(&HashMap::new(), &devices)
            .expect("allocation should succeed");
        assert_eq!(id, "big");
    }

    #[test]
    fn test_allocate_power_efficient_picks_lowest_power() {
        let mut allocator = ResourceAllocator::new(AllocationStrategy::PowerEfficient);
        let devices = vec![
            device("hungry", 1024, 0.0, None, Some(300.0)),
            device("thrifty", 1024, 0.0, None, Some(15.0)),
        ];
        let id = allocator
            .allocate(&HashMap::new(), &devices)
            .expect("allocation should succeed");
        assert_eq!(id, "thrifty");
    }

    #[test]
    fn test_allocate_best_fit_picks_tightest_sufficient_device() {
        let mut allocator = ResourceAllocator::new(AllocationStrategy::BestFit);
        let devices = vec![
            device("huge", 1_000_000, 0.0, None, None),
            device("snug", 200, 0.0, None, None),
            device("too_small", 50, 0.0, None, None),
        ];
        let mut requirements = HashMap::new();
        requirements.insert("memory".to_string(), 100.0);
        let id = allocator.allocate(&requirements, &devices).expect("allocation should succeed");
        assert_eq!(
            id, "snug",
            "best-fit must pick the smallest device that still satisfies the requirement"
        );
    }

    /// Regression test: `round_robin_select`/`least_utilization_select`/
    /// `weighted_round_robin_select`/`performance_based_select`/
    /// `adaptive_select` used to all return `devices[0]` unconditionally.
    #[test]
    fn test_load_balancer_round_robin_actually_cycles() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        let devices = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let first = lb.select_device(&devices).expect("select should succeed");
        let second = lb.select_device(&devices).expect("select should succeed");
        let third = lb.select_device(&devices).expect("select should succeed");
        assert_ne!(
            first, second,
            "round robin must not pick the same device twice in a row"
        );
        assert_ne!(second, third);
    }

    #[test]
    fn test_load_balancer_least_utilization_uses_load_history() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::LeastUtilization);
        lb.load_history.insert("busy".to_string(), vec![(SystemTime::now(), 0.95)]);
        lb.load_history.insert("idle".to_string(), vec![(SystemTime::now(), 0.02)]);
        let devices = vec!["busy".to_string(), "idle".to_string()];
        let selected = lb.select_device(&devices).expect("select should succeed");
        assert_eq!(selected, "idle");
    }

    #[test]
    fn test_load_balancer_performance_based_uses_weights() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::PerformanceBased);
        lb.set_weight("weak", 1.0);
        lb.set_weight("strong", 10.0);
        let devices = vec!["weak".to_string(), "strong".to_string()];
        let selected = lb.select_device(&devices).expect("select should succeed");
        assert_eq!(selected, "strong");
    }

    #[test]
    fn test_load_balancer_weighted_round_robin_favors_higher_weight() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::WeightedRoundRobin);
        lb.set_weight("light", 1.0);
        lb.set_weight("heavy", 3.0);
        let devices = vec!["light".to_string(), "heavy".to_string()];

        let mut heavy_count = 0;
        for _ in 0..8 {
            if lb.select_device(&devices).expect("select should succeed") == "heavy" {
                heavy_count += 1;
            }
        }
        // With weight 3:1 over 8 selections, "heavy" should be picked
        // noticeably more than half the time.
        assert!(
            heavy_count >= 5,
            "expected heavy (weight 3) to be selected more often, got {heavy_count}/8"
        );
    }
}
