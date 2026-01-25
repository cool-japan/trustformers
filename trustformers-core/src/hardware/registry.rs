// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hardware registry for TrustformeRS
//!
//! This module provides a centralized registry for hardware backends, devices,
//! and operations. It enables dynamic discovery, registration, and management
//! of hardware components across the TrustformeRS ecosystem.

#![allow(unused_variables)] // Hardware registry with backend-specific code

use super::traits::{HardwareBackend, HardwareDevice, HardwareOperation};
use super::{HardwareCapabilities, HardwareResult, HardwareType};
use crate::errors::TrustformersError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

/// Hardware registry configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Enable automatic backend discovery
    pub auto_discovery: bool,
    /// Backend registration timeout in seconds
    pub registration_timeout: u64,
    /// Maximum number of backends per type
    pub max_backends_per_type: usize,
    /// Enable backend versioning
    pub enable_versioning: bool,
    /// Enable capability validation
    pub enable_capability_validation: bool,
    /// Registry persistence
    pub persistence: RegistryPersistence,
}

/// Registry persistence configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryPersistence {
    /// Enable persistence
    pub enabled: bool,
    /// Storage path
    pub storage_path: String,
    /// Backup interval in seconds
    pub backup_interval: u64,
    /// Retention period in days
    pub retention_period: u32,
}

/// Backend registration information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendRegistration {
    /// Backend ID
    pub id: String,
    /// Backend name
    pub name: String,
    /// Backend version
    pub version: String,
    /// Hardware type
    pub hardware_type: HardwareType,
    /// Supported operations
    pub supported_operations: Vec<String>,
    /// Registration timestamp
    pub registration_time: SystemTime,
    /// Last activity timestamp
    pub last_activity: SystemTime,
    /// Backend status
    pub status: BackendStatus,
    /// Backend metadata
    pub metadata: HashMap<String, String>,
    /// Capabilities
    pub capabilities: HardwareCapabilities,
}

/// Backend status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendStatus {
    /// Backend is active and available
    Active,
    /// Backend is inactive
    Inactive,
    /// Backend is busy
    Busy,
    /// Backend has failed
    Failed,
    /// Backend is in maintenance mode
    Maintenance,
    /// Backend is deprecated
    Deprecated,
}

/// Device registration information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceRegistration {
    /// Device ID
    pub id: String,
    /// Device name
    pub name: String,
    /// Backend ID
    pub backend_id: String,
    /// Hardware type
    pub hardware_type: HardwareType,
    /// Device capabilities
    pub capabilities: HardwareCapabilities,
    /// Registration timestamp
    pub registration_time: SystemTime,
    /// Last seen timestamp
    pub last_seen: SystemTime,
    /// Device status
    pub status: DeviceStatus,
    /// Device metadata
    pub metadata: HashMap<String, String>,
    /// Device tags
    pub tags: Vec<String>,
}

/// Device status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceStatus {
    /// Device is online and available
    Online,
    /// Device is offline
    Offline,
    /// Device is busy
    Busy,
    /// Device has failed
    Failed,
    /// Device is in maintenance mode
    Maintenance,
    /// Device is unknown
    Unknown,
}

/// Operation registration information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationRegistration {
    /// Operation ID
    pub id: String,
    /// Operation name
    pub name: String,
    /// Backend ID
    pub backend_id: String,
    /// Supported hardware types
    pub hardware_types: Vec<HardwareType>,
    /// Operation requirements
    pub requirements: OperationRequirements,
    /// Registration timestamp
    pub registration_time: SystemTime,
    /// Operation metadata
    pub metadata: HashMap<String, String>,
    /// Operation version
    pub version: String,
}

/// Operation requirements
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationRequirements {
    /// Minimum memory required (bytes)
    pub min_memory: usize,
    /// Required compute units
    pub compute_units: Option<u32>,
    /// Required data types
    pub data_types: Vec<super::DataType>,
    /// Required capabilities
    pub capabilities: Vec<String>,
    /// Performance constraints
    pub performance_constraints: PerformanceConstraints,
}

/// Performance constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PerformanceConstraints {
    /// Maximum acceptable latency (ms)
    pub max_latency: Option<f64>,
    /// Minimum required throughput (ops/sec)
    pub min_throughput: Option<f64>,
    /// Maximum power consumption (W)
    pub max_power: Option<f64>,
    /// Maximum temperature (°C)
    pub max_temperature: Option<f64>,
    /// Memory bandwidth requirements (GB/s)
    pub memory_bandwidth: Option<f64>,
}

/// Registry event types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegistryEvent {
    /// Backend registered
    BackendRegistered {
        backend_id: String,
        timestamp: SystemTime,
    },
    /// Backend unregistered
    BackendUnregistered {
        backend_id: String,
        timestamp: SystemTime,
    },
    /// Device registered
    DeviceRegistered {
        device_id: String,
        backend_id: String,
        timestamp: SystemTime,
    },
    /// Device unregistered
    DeviceUnregistered {
        device_id: String,
        timestamp: SystemTime,
    },
    /// Operation registered
    OperationRegistered {
        operation_id: String,
        backend_id: String,
        timestamp: SystemTime,
    },
    /// Operation unregistered
    OperationUnregistered {
        operation_id: String,
        timestamp: SystemTime,
    },
    /// Status changed
    StatusChanged {
        component_id: String,
        old_status: String,
        new_status: String,
        timestamp: SystemTime,
    },
}

/// Registry statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryStatistics {
    /// Total backends registered
    pub total_backends: usize,
    /// Active backends
    pub active_backends: usize,
    /// Total devices registered
    pub total_devices: usize,
    /// Online devices
    pub online_devices: usize,
    /// Total operations registered
    pub total_operations: usize,
    /// Backends by type
    pub backends_by_type: HashMap<HardwareType, usize>,
    /// Devices by type
    pub devices_by_type: HashMap<HardwareType, usize>,
    /// Operations by type
    pub operations_by_type: HashMap<HardwareType, usize>,
    /// Registry uptime
    pub uptime: std::time::Duration,
    /// Last update timestamp
    pub last_update: SystemTime,
}

/// Hardware registry implementation
pub struct HardwareRegistry {
    /// Registry configuration
    config: RegistryConfig,
    /// Registered backends
    backends: Arc<RwLock<HashMap<String, BackendRegistration>>>,
    /// Backend instances
    backend_instances: Arc<RwLock<HashMap<String, Box<dyn HardwareBackend>>>>,
    /// Registered devices
    devices: Arc<RwLock<HashMap<String, DeviceRegistration>>>,
    /// Registered operations
    operations: Arc<RwLock<HashMap<String, OperationRegistration>>>,
    /// Registry events
    events: Arc<RwLock<Vec<RegistryEvent>>>,
    /// Event listeners
    event_listeners: Arc<RwLock<Vec<Box<dyn RegistryEventListener>>>>,
    /// Registry statistics
    statistics: Arc<RwLock<RegistryStatistics>>,
    /// Creation timestamp
    creation_time: SystemTime,
}

impl std::fmt::Debug for HardwareRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HardwareRegistry")
            .field("config", &self.config)
            .field(
                "backends",
                &"<RwLock<HashMap<String, BackendRegistration>>>",
            )
            .field(
                "backend_instances",
                &"<RwLock<HashMap<String, Box<dyn HardwareBackend>>>>",
            )
            .field("devices", &"<RwLock<HashMap<String, DeviceRegistration>>>")
            .field(
                "operations",
                &"<RwLock<HashMap<String, OperationRegistration>>>",
            )
            .field("events", &"<RwLock<Vec<RegistryEvent>>>")
            .field(
                "event_listeners",
                &"<RwLock<Vec<Box<dyn RegistryEventListener>>>>",
            )
            .field("statistics", &"<RwLock<RegistryStatistics>>")
            .field("creation_time", &self.creation_time)
            .finish()
    }
}

/// Registry event listener trait
pub trait RegistryEventListener: Send + Sync {
    /// Handle registry event
    fn handle_event(&self, event: &RegistryEvent);

    /// Get listener name
    fn name(&self) -> &str;
}

/// Registry query
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RegistryQuery {
    /// Hardware type filter
    pub hardware_type: Option<HardwareType>,
    /// Backend status filter
    pub backend_status: Option<BackendStatus>,
    /// Device status filter
    pub device_status: Option<DeviceStatus>,
    /// Capability requirements
    pub capabilities: Option<Vec<String>>,
    /// Tag filters
    pub tags: Option<Vec<String>>,
    /// Metadata filters
    pub metadata: Option<HashMap<String, String>>,
    /// Operation filters
    pub operations: Option<Vec<String>>,
    /// Limit results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
}

/// Registry query result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryQueryResult {
    /// Matching backends
    pub backends: Vec<BackendRegistration>,
    /// Matching devices
    pub devices: Vec<DeviceRegistration>,
    /// Matching operations
    pub operations: Vec<OperationRegistration>,
    /// Total matching count
    pub total_count: usize,
    /// Query execution time
    pub execution_time: std::time::Duration,
}

impl Default for HardwareRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareRegistry {
    /// Create new hardware registry
    pub fn new() -> Self {
        Self {
            config: RegistryConfig::default(),
            backends: Arc::new(RwLock::new(HashMap::new())),
            backend_instances: Arc::new(RwLock::new(HashMap::new())),
            devices: Arc::new(RwLock::new(HashMap::new())),
            operations: Arc::new(RwLock::new(HashMap::new())),
            events: Arc::new(RwLock::new(Vec::new())),
            event_listeners: Arc::new(RwLock::new(Vec::new())),
            statistics: Arc::new(RwLock::new(RegistryStatistics::default())),
            creation_time: SystemTime::now(),
        }
    }

    /// Create registry with configuration
    pub fn with_config(config: RegistryConfig) -> Self {
        let mut registry = Self::new();
        registry.config = config;
        registry
    }

    /// Register hardware backend
    pub fn register_backend(&self, backend: Box<dyn HardwareBackend>) -> HardwareResult<String> {
        let backend_id = format!("{}_{}", backend.name(), backend.version());

        // Check if backend already exists
        let backends = self.backends.read().expect("Read lock poisoned");
        if backends.contains_key(&backend_id) {
            return Err(TrustformersError::invalid_config(format!(
                "Backend {} already registered",
                backend_id
            )));
        }
        drop(backends);

        // Create registration
        let registration = BackendRegistration {
            id: backend_id.clone(),
            name: backend.name().to_string(),
            version: backend.version().to_string(),
            hardware_type: self.determine_hardware_type(backend.as_ref()),
            supported_operations: backend.supported_operations().to_vec(),
            registration_time: SystemTime::now(),
            last_activity: SystemTime::now(),
            status: BackendStatus::Active,
            metadata: HashMap::new(),
            capabilities: HardwareCapabilities::default(),
        };

        // Store backend and registration
        let mut backends = self.backends.write().expect("Write lock poisoned");
        backends.insert(backend_id.clone(), registration);
        drop(backends);

        let mut backend_instances = self.backend_instances.write().expect("Write lock poisoned");
        backend_instances.insert(backend_id.clone(), backend);
        drop(backend_instances);

        // Emit event
        self.emit_event(RegistryEvent::BackendRegistered {
            backend_id: backend_id.clone(),
            timestamp: SystemTime::now(),
        });

        // Update statistics
        self.update_statistics();

        Ok(backend_id)
    }

    /// Unregister hardware backend
    pub fn unregister_backend(&self, backend_id: &str) -> HardwareResult<()> {
        let mut backends = self.backends.write().expect("Write lock poisoned");
        let mut backend_instances = self.backend_instances.write().expect("Write lock poisoned");

        if backends.remove(backend_id).is_none() {
            return Err(TrustformersError::model_error(format!(
                "Backend {} not found",
                backend_id
            )));
        }

        backend_instances.remove(backend_id);

        // Emit event
        self.emit_event(RegistryEvent::BackendUnregistered {
            backend_id: backend_id.to_string(),
            timestamp: SystemTime::now(),
        });

        // Update statistics
        self.update_statistics();

        Ok(())
    }

    /// Get backend by ID
    pub fn get_backend(&self, backend_id: &str) -> Option<BackendRegistration> {
        let backends = self.backends.read().expect("Read lock poisoned");
        backends.get(backend_id).cloned()
    }

    /// Get backend instance
    pub fn get_backend_instance(&self, backend_id: &str) -> Option<Box<dyn HardwareBackend>> {
        let backend_instances = self.backend_instances.read().expect("Read lock poisoned");
        // This is a simplified implementation
        // In practice, you'd need to implement proper cloning or use Arc<>
        None
    }

    /// List all backends
    pub fn list_backends(&self) -> Vec<BackendRegistration> {
        let backends = self.backends.read().expect("Read lock poisoned");
        backends.values().cloned().collect()
    }

    /// List backends by type
    pub fn list_backends_by_type(&self, hardware_type: HardwareType) -> Vec<BackendRegistration> {
        let backends = self.backends.read().expect("Read lock poisoned");
        backends
            .values()
            .filter(|backend| backend.hardware_type == hardware_type)
            .cloned()
            .collect()
    }

    /// Get all backend instances
    pub fn get_backends(&self) -> Vec<Box<dyn HardwareBackend>> {
        let backend_instances = self.backend_instances.read().expect("Read lock poisoned");
        // This is a simplified implementation
        // In practice, you'd need to implement proper cloning or use Arc<>
        vec![]
    }

    /// Register device
    pub fn register_device(
        &self,
        device: Box<dyn HardwareDevice>,
        backend_id: &str,
    ) -> HardwareResult<()> {
        let device_id = device.device_id().to_string();

        // Check if device already exists
        let devices = self.devices.read().expect("Read lock poisoned");
        if devices.contains_key(&device_id) {
            return Err(TrustformersError::invalid_config(format!(
                "Device {} already registered",
                device_id
            )));
        }
        drop(devices);

        // Create registration
        let registration = DeviceRegistration {
            id: device_id.clone(),
            name: device_id.clone(),
            backend_id: backend_id.to_string(),
            hardware_type: device.hardware_type(),
            capabilities: device.capabilities().clone(),
            registration_time: SystemTime::now(),
            last_seen: SystemTime::now(),
            status: DeviceStatus::Online,
            metadata: HashMap::new(),
            tags: vec![],
        };

        // Store device registration
        let mut devices = self.devices.write().expect("Write lock poisoned");
        devices.insert(device_id.clone(), registration);
        drop(devices);

        // Emit event
        self.emit_event(RegistryEvent::DeviceRegistered {
            device_id: device_id.clone(),
            backend_id: backend_id.to_string(),
            timestamp: SystemTime::now(),
        });

        // Update statistics
        self.update_statistics();

        Ok(())
    }

    /// Unregister device
    pub fn unregister_device(&self, device_id: &str) -> HardwareResult<()> {
        let mut devices = self.devices.write().expect("Write lock poisoned");

        if devices.remove(device_id).is_none() {
            return Err(TrustformersError::model_error(format!(
                "Device {} not found",
                device_id
            )));
        }

        // Emit event
        self.emit_event(RegistryEvent::DeviceUnregistered {
            device_id: device_id.to_string(),
            timestamp: SystemTime::now(),
        });

        // Update statistics
        self.update_statistics();

        Ok(())
    }

    /// Get device registration
    pub fn get_device(&self, device_id: &str) -> Option<DeviceRegistration> {
        let devices = self.devices.read().expect("Read lock poisoned");
        devices.get(device_id).cloned()
    }

    /// List all devices
    pub fn list_devices(&self) -> Vec<DeviceRegistration> {
        let devices = self.devices.read().expect("Read lock poisoned");
        devices.values().cloned().collect()
    }

    /// List devices by type
    pub fn list_devices_by_type(&self, hardware_type: HardwareType) -> Vec<DeviceRegistration> {
        let devices = self.devices.read().expect("Read lock poisoned");
        devices
            .values()
            .filter(|device| device.hardware_type == hardware_type)
            .cloned()
            .collect()
    }

    /// Register operation
    pub fn register_operation(
        &self,
        operation: Box<dyn HardwareOperation>,
        backend_id: &str,
    ) -> HardwareResult<()> {
        let operation_id = format!("{}_{}", operation.name(), backend_id);

        // Check if operation already exists
        let operations = self.operations.read().expect("Read lock poisoned");
        if operations.contains_key(&operation_id) {
            return Err(TrustformersError::invalid_config(format!(
                "Operation {} already registered",
                operation_id
            )));
        }
        drop(operations);

        // Create registration
        let registration = OperationRegistration {
            id: operation_id.clone(),
            name: operation.name().to_string(),
            backend_id: backend_id.to_string(),
            hardware_types: vec![HardwareType::CPU], // Placeholder
            requirements: OperationRequirements::default(),
            registration_time: SystemTime::now(),
            metadata: HashMap::new(),
            version: "1.0.0".to_string(),
        };

        // Store operation registration
        let mut operations = self.operations.write().expect("Write lock poisoned");
        operations.insert(operation_id.clone(), registration);
        drop(operations);

        // Emit event
        self.emit_event(RegistryEvent::OperationRegistered {
            operation_id: operation_id.clone(),
            backend_id: backend_id.to_string(),
            timestamp: SystemTime::now(),
        });

        // Update statistics
        self.update_statistics();

        Ok(())
    }

    /// Query registry
    pub fn query(&self, query: &RegistryQuery) -> RegistryQueryResult {
        let start_time = std::time::Instant::now();

        let backends = self.backends.read().expect("Read lock poisoned");
        let devices = self.devices.read().expect("Read lock poisoned");
        let operations = self.operations.read().expect("Read lock poisoned");

        // Filter backends
        let mut matching_backends: Vec<BackendRegistration> = backends
            .values()
            .filter(|backend| {
                if let Some(ref hw_type) = query.hardware_type {
                    if backend.hardware_type != *hw_type {
                        return false;
                    }
                }
                if let Some(ref status) = query.backend_status {
                    if backend.status != *status {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        // Filter devices
        let mut matching_devices: Vec<DeviceRegistration> = devices
            .values()
            .filter(|device| {
                if let Some(ref hw_type) = query.hardware_type {
                    if device.hardware_type != *hw_type {
                        return false;
                    }
                }
                if let Some(ref status) = query.device_status {
                    if device.status != *status {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        // Filter operations
        let mut matching_operations: Vec<OperationRegistration> =
            operations.values().cloned().collect();

        // Apply pagination
        let total_count =
            matching_backends.len() + matching_devices.len() + matching_operations.len();

        if let Some(limit) = query.limit {
            matching_backends.truncate(limit);
            matching_devices.truncate(limit);
            matching_operations.truncate(limit);
        }

        RegistryQueryResult {
            backends: matching_backends,
            devices: matching_devices,
            operations: matching_operations,
            total_count,
            execution_time: start_time.elapsed(),
        }
    }

    /// Add event listener
    pub fn add_event_listener(&self, listener: Box<dyn RegistryEventListener>) {
        let mut listeners = self.event_listeners.write().expect("Write lock poisoned");
        listeners.push(listener);
    }

    /// Remove event listener
    pub fn remove_event_listener(&self, listener_name: &str) {
        let mut listeners = self.event_listeners.write().expect("Write lock poisoned");
        listeners.retain(|l| l.name() != listener_name);
    }

    /// Get registry statistics
    pub fn get_statistics(&self) -> RegistryStatistics {
        let stats = self.statistics.read().expect("Read lock poisoned");
        stats.clone()
    }

    /// Update device status
    pub fn update_device_status(
        &self,
        device_id: &str,
        status: DeviceStatus,
    ) -> HardwareResult<()> {
        let mut devices = self.devices.write().expect("Write lock poisoned");

        if let Some(device) = devices.get_mut(device_id) {
            let old_status = device.status;
            device.status = status;
            device.last_seen = SystemTime::now();

            // Emit status change event
            self.emit_event(RegistryEvent::StatusChanged {
                component_id: device_id.to_string(),
                old_status: format!("{:?}", old_status),
                new_status: format!("{:?}", status),
                timestamp: SystemTime::now(),
            });

            Ok(())
        } else {
            Err(TrustformersError::model_error(format!(
                "Device {} not found",
                device_id
            )))
        }
    }

    /// Update backend status
    pub fn update_backend_status(
        &self,
        backend_id: &str,
        status: BackendStatus,
    ) -> HardwareResult<()> {
        let mut backends = self.backends.write().expect("Write lock poisoned");

        if let Some(backend) = backends.get_mut(backend_id) {
            let old_status = backend.status;
            backend.status = status;
            backend.last_activity = SystemTime::now();

            // Emit status change event
            self.emit_event(RegistryEvent::StatusChanged {
                component_id: backend_id.to_string(),
                old_status: format!("{:?}", old_status),
                new_status: format!("{:?}", status),
                timestamp: SystemTime::now(),
            });

            Ok(())
        } else {
            Err(TrustformersError::model_error(format!(
                "Backend {} not found",
                backend_id
            )))
        }
    }

    /// Get registry events
    pub fn get_events(&self, limit: Option<usize>) -> Vec<RegistryEvent> {
        let events = self.events.read().expect("Read lock poisoned");
        if let Some(limit) = limit {
            events.iter().rev().take(limit).cloned().collect()
        } else {
            events.clone()
        }
    }

    /// Clear registry events
    pub fn clear_events(&self) {
        let mut events = self.events.write().expect("Write lock poisoned");
        events.clear();
    }

    /// Emit registry event
    fn emit_event(&self, event: RegistryEvent) {
        // Store event
        let mut events = self.events.write().expect("Write lock poisoned");
        events.push(event.clone());

        // Keep only last 1000 events
        if events.len() > 1000 {
            events.drain(..500);
        }
        drop(events);

        // Notify listeners
        let listeners = self.event_listeners.read().expect("Read lock poisoned");
        for listener in listeners.iter() {
            listener.handle_event(&event);
        }
    }

    /// Update registry statistics
    fn update_statistics(&self) {
        let backends = self.backends.read().expect("Read lock poisoned");
        let devices = self.devices.read().expect("Read lock poisoned");
        let operations = self.operations.read().expect("Read lock poisoned");

        let mut stats = self.statistics.write().expect("Write lock poisoned");

        stats.total_backends = backends.len();
        stats.active_backends =
            backends.values().filter(|b| b.status == BackendStatus::Active).count();

        stats.total_devices = devices.len();
        stats.online_devices =
            devices.values().filter(|d| d.status == DeviceStatus::Online).count();

        stats.total_operations = operations.len();

        // Update type counts
        stats.backends_by_type.clear();
        for backend in backends.values() {
            *stats.backends_by_type.entry(backend.hardware_type.clone()).or_insert(0) += 1;
        }

        stats.devices_by_type.clear();
        for device in devices.values() {
            *stats.devices_by_type.entry(device.hardware_type.clone()).or_insert(0) += 1;
        }

        stats.uptime = SystemTime::now().duration_since(self.creation_time).unwrap_or_default();
        stats.last_update = SystemTime::now();
    }

    /// Determine hardware type from backend
    fn determine_hardware_type(&self, backend: &dyn HardwareBackend) -> HardwareType {
        // Simple heuristic based on backend name
        let name = backend.name().to_lowercase();
        if name.contains("cpu") {
            HardwareType::CPU
        } else if name.contains("gpu") {
            HardwareType::GPU
        } else if name.contains("asic") {
            HardwareType::ASIC
        } else if name.contains("tpu") {
            HardwareType::TPU
        } else if name.contains("fpga") {
            HardwareType::FPGA
        } else {
            HardwareType::Custom(name)
        }
    }
}

// Default implementations

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            auto_discovery: true,
            registration_timeout: 30,
            max_backends_per_type: 10,
            enable_versioning: true,
            enable_capability_validation: true,
            persistence: RegistryPersistence::default(),
        }
    }
}

impl Default for RegistryPersistence {
    fn default() -> Self {
        Self {
            enabled: false,
            storage_path: "/tmp/trustformers_registry".to_string(),
            backup_interval: 3600, // 1 hour
            retention_period: 30,  // 30 days
        }
    }
}

impl Default for OperationRequirements {
    fn default() -> Self {
        Self {
            min_memory: 0,
            compute_units: None,
            data_types: vec![super::DataType::F32],
            capabilities: vec![],
            performance_constraints: PerformanceConstraints::default(),
        }
    }
}

impl Default for RegistryStatistics {
    fn default() -> Self {
        Self {
            total_backends: 0,
            active_backends: 0,
            total_devices: 0,
            online_devices: 0,
            total_operations: 0,
            backends_by_type: HashMap::new(),
            devices_by_type: HashMap::new(),
            operations_by_type: HashMap::new(),
            uptime: std::time::Duration::from_secs(0),
            last_update: SystemTime::now(),
        }
    }
}

impl std::fmt::Display for BackendStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendStatus::Active => write!(f, "Active"),
            BackendStatus::Inactive => write!(f, "Inactive"),
            BackendStatus::Busy => write!(f, "Busy"),
            BackendStatus::Failed => write!(f, "Failed"),
            BackendStatus::Maintenance => write!(f, "Maintenance"),
            BackendStatus::Deprecated => write!(f, "Deprecated"),
        }
    }
}

impl std::fmt::Display for DeviceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceStatus::Online => write!(f, "Online"),
            DeviceStatus::Offline => write!(f, "Offline"),
            DeviceStatus::Busy => write!(f, "Busy"),
            DeviceStatus::Failed => write!(f, "Failed"),
            DeviceStatus::Maintenance => write!(f, "Maintenance"),
            DeviceStatus::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Example event listener implementation
pub struct ConsoleEventListener {
    name: String,
}

impl ConsoleEventListener {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

impl RegistryEventListener for ConsoleEventListener {
    fn handle_event(&self, event: &RegistryEvent) {
        println!("[{}] Registry event: {:?}", self.name, event);
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = HardwareRegistry::new();
        let stats = registry.get_statistics();

        assert_eq!(stats.total_backends, 0);
        assert_eq!(stats.total_devices, 0);
        assert_eq!(stats.total_operations, 0);
    }

    #[test]
    fn test_registry_config_default() {
        let config = RegistryConfig::default();

        assert!(config.auto_discovery);
        assert_eq!(config.registration_timeout, 30);
        assert_eq!(config.max_backends_per_type, 10);
        assert!(config.enable_versioning);
        assert!(config.enable_capability_validation);
    }

    #[test]
    fn test_backend_status_display() {
        assert_eq!(BackendStatus::Active.to_string(), "Active");
        assert_eq!(BackendStatus::Failed.to_string(), "Failed");
        assert_eq!(BackendStatus::Maintenance.to_string(), "Maintenance");
    }

    #[test]
    fn test_device_status_display() {
        assert_eq!(DeviceStatus::Online.to_string(), "Online");
        assert_eq!(DeviceStatus::Offline.to_string(), "Offline");
        assert_eq!(DeviceStatus::Failed.to_string(), "Failed");
    }

    #[test]
    fn test_registry_query_default() {
        let query = RegistryQuery::default();

        assert!(query.hardware_type.is_none());
        assert!(query.backend_status.is_none());
        assert!(query.device_status.is_none());
        assert!(query.limit.is_none());
    }

    #[test]
    fn test_operation_requirements_default() {
        let requirements = OperationRequirements::default();

        assert_eq!(requirements.min_memory, 0);
        assert!(requirements.compute_units.is_none());
        assert!(!requirements.data_types.is_empty());
        assert!(requirements.capabilities.is_empty());
    }

    #[test]
    fn test_performance_constraints_default() {
        let constraints = PerformanceConstraints::default();

        assert!(constraints.max_latency.is_none());
        assert!(constraints.min_throughput.is_none());
        assert!(constraints.max_power.is_none());
        assert!(constraints.max_temperature.is_none());
        assert!(constraints.memory_bandwidth.is_none());
    }

    #[test]
    fn test_console_event_listener() {
        let listener = ConsoleEventListener::new("test".to_string());
        assert_eq!(listener.name(), "test");

        let event = RegistryEvent::BackendRegistered {
            backend_id: "test_backend".to_string(),
            timestamp: SystemTime::now(),
        };

        // This would normally print to console
        listener.handle_event(&event);
    }

    #[test]
    fn test_registry_statistics_update() {
        let registry = HardwareRegistry::new();
        let initial_stats = registry.get_statistics();

        assert_eq!(initial_stats.total_backends, 0);
        assert_eq!(initial_stats.active_backends, 0);
        assert_eq!(initial_stats.total_devices, 0);
        assert_eq!(initial_stats.online_devices, 0);
    }

    #[test]
    fn test_registry_event_types() {
        let timestamp = SystemTime::now();

        let backend_event = RegistryEvent::BackendRegistered {
            backend_id: "test".to_string(),
            timestamp,
        };

        let device_event = RegistryEvent::DeviceRegistered {
            device_id: "test_device".to_string(),
            backend_id: "test_backend".to_string(),
            timestamp,
        };

        match backend_event {
            RegistryEvent::BackendRegistered { backend_id, .. } => {
                assert_eq!(backend_id, "test");
            },
            _ => panic!("Expected BackendRegistered event"),
        }

        match device_event {
            RegistryEvent::DeviceRegistered {
                device_id,
                backend_id,
                ..
            } => {
                assert_eq!(device_id, "test_device");
                assert_eq!(backend_id, "test_backend");
            },
            _ => panic!("Expected DeviceRegistered event"),
        }
    }
}
