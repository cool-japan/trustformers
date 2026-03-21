//! App State Management Types
//!
//! This module contains types for managing app lifecycle states, transitions,
//! and related context information.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// App states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AppState {
    /// App is starting up
    Launching,
    /// App is active and in foreground
    Active,
    /// App is in background but still executing
    Background,
    /// App is inactive (transitioning)
    Inactive,
    /// App is suspended (not executing)
    Suspended,
    /// App is terminating
    Terminating,
    /// Unknown state
    Unknown,
}

/// State transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StateTransition {
    LaunchToActive,
    ActiveToBackground,
    BackgroundToActive,
    ActiveToInactive,
    InactiveToActive,
    BackgroundToSuspended,
    SuspendedToBackground,
    AnyToTerminating,
}

/// State transition event
#[derive(Debug, Clone)]
pub struct StateTransitionEvent {
    pub from_state: AppState,
    pub to_state: AppState,
    pub transition: StateTransition,
    pub timestamp: Instant,
    pub reason: TransitionReason,
    pub context: TransitionContext,
}

/// Reasons for state transitions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionReason {
    UserAction,
    SystemRequest,
    LowMemory,
    LowBattery,
    ThermalPressure,
    NetworkInterruption,
    Timeout,
    Error,
    SystemSuspend,
    SystemResume,
}

/// Context information for state transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionContext {
    /// Available memory (MB)
    pub available_memory_mb: usize,
    /// Battery level (%)
    pub battery_level_percent: u8,
    /// CPU temperature (°C)
    pub cpu_temperature_celsius: f32,
    /// Network connectivity status
    pub network_connected: bool,
    /// Active background tasks count
    pub active_background_tasks: usize,
    /// Time since last user interaction (seconds)
    pub time_since_user_interaction_seconds: u64,
    /// Foreground app duration (seconds)
    pub foreground_duration_seconds: u64,
    /// Background app duration (seconds)
    pub background_duration_seconds: u64,
    /// System pressure indicators
    pub system_pressure: SystemPressureIndicators,
    /// Resource usage at transition
    pub resource_usage: ResourceUsageSnapshot,
}

/// System pressure indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPressureIndicators {
    /// Memory pressure level
    pub memory_pressure: crate::lifecycle::config::MemoryPressureLevel,
    /// Thermal state
    pub thermal_state: crate::lifecycle::config::ThermalLevel,
    /// Battery state
    pub battery_state: crate::battery::BatteryLevel,
    /// Network quality
    pub network_quality: NetworkQuality,
}

/// Network quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkQuality {
    Excellent,
    Good,
    Fair,
    Poor,
    Disconnected,
}

/// Resource usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageSnapshot {
    /// CPU usage percentage (0-100)
    pub cpu_usage_percent: f32,
    /// Memory usage in MB
    pub memory_usage_mb: usize,
    /// GPU usage percentage (0-100)
    pub gpu_usage_percent: Option<f32>,
    /// Network usage in MB/s
    pub network_usage_mbps: f32,
    /// Storage I/O in MB/s
    pub storage_io_mbps: f32,
    /// Active model count
    pub active_models: usize,
    /// Inference queue size
    pub inference_queue_size: usize,
}

/// App checkpoint for state persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppCheckpoint {
    /// App state at checkpoint
    pub app_state: AppState,
    /// Timestamp of checkpoint
    pub timestamp: u64, // Using u64 for serialization
    /// Model states
    pub model_states: Vec<ModelState>,
    /// Cache states
    pub cache_states: HashMap<String, CacheState>,
    /// User session state
    pub user_session: UserSessionState,
}

/// Model state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    /// Model identifier
    pub model_id: String,
    /// Model load state
    pub load_state: ModelLoadState,
    /// Model configuration
    pub configuration: ModelConfiguration,
    /// Performance metrics
    pub performance_metrics: ModelPerformanceMetrics,
    /// Last access timestamp
    pub last_access_timestamp: u64,
    /// Usage count
    pub usage_count: usize,
}

/// Model load states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelLoadState {
    Unloaded,
    Loading,
    Loaded,
    Error,
    Cached,
    Compressed,
    Offloaded,
}

/// Cache state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheState {
    /// Cache size in bytes
    pub size_bytes: usize,
    /// Number of cached items
    pub item_count: usize,
    /// Last access timestamp
    pub last_access_timestamp: u64,
    /// Hit rate percentage
    pub hit_rate_percent: f32,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfiguration {
    /// Model precision
    pub precision: ModelPrecision,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Backend in use
    pub backend: ModelBackend,
}

/// Model precision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelPrecision {
    FP32,
    FP16,
    INT8,
    INT4,
    Dynamic,
}

/// Model optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    UltraLowMemory,
}

/// Model backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelBackend {
    CPU,
    CoreML,
    NNAPI,
    Metal,
    Vulkan,
    Custom,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    /// Average inference time (ms)
    pub avg_inference_time_ms: f32,
    /// Memory usage (MB)
    pub memory_usage_mb: usize,
    /// Throughput (inferences/second)
    pub throughput_per_second: f32,
    /// Error rate percentage
    pub error_rate_percent: f32,
}

/// Task state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskState {
    /// Task identifier
    pub task_id: String,
    /// Task type
    pub task_type: crate::lifecycle::config::TaskType,
    /// Current status
    pub status: LifecycleTaskStatus,
    /// Priority level
    pub priority: crate::lifecycle::config::TaskPriority,
    /// Progress percentage (0-100)
    pub progress_percent: u8,
    /// Estimated completion time (seconds)
    pub estimated_completion_seconds: Option<u64>,
}

/// Task execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleTaskStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// User session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSessionState {
    /// Session start timestamp
    pub session_start_timestamp: u64,
    /// Total interaction time (seconds)
    pub total_interaction_time_seconds: u64,
    /// Recent user interactions
    pub recent_interactions: Vec<UserInteraction>,
    /// Session statistics
    pub session_stats: SessionStatistics,
}

/// User interaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInteraction {
    /// Interaction timestamp
    pub timestamp: u64,
    /// Type of interaction
    pub interaction_type: InteractionType,
    /// Interaction outcome
    pub outcome: InteractionOutcome,
    /// Duration of interaction (ms)
    pub duration_ms: u64,
}

/// Types of user interactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionType {
    Touch,
    Voice,
    Gesture,
    TextInput,
    ModelInference,
    Navigation,
    Settings,
}

/// Interaction outcomes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionOutcome {
    Success,
    Failure,
    Cancelled,
    Timeout,
    Error,
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    /// Total inferences performed
    pub total_inferences: usize,
    /// Total models loaded
    pub total_models_loaded: usize,
    /// Total background tasks executed
    pub total_background_tasks: usize,
    /// Average response time (ms)
    pub avg_response_time_ms: f32,
    /// Error count
    pub error_count: usize,
}

/// System context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemContext {
    /// Current timestamp
    pub timestamp: u64,
    /// Device information
    pub device_info: SystemDeviceInfo,
    /// Resource availability
    pub resource_availability: ResourceAvailability,
    /// Network status
    pub network_status: NetworkStatus,
    /// Power status
    pub power_status: PowerStatus,
    /// Thermal status
    pub thermal_status: ThermalStatus,
    /// Active processes
    pub active_processes: ProcessInfo,
    /// System load
    pub system_load: SystemLoad,
}

/// System device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemDeviceInfo {
    /// Device model
    pub device_model: String,
    /// OS version
    pub os_version: String,
    /// Available cores
    pub cpu_cores: usize,
    /// Total RAM (MB)
    pub total_ram_mb: usize,
    /// Available storage (MB)
    pub available_storage_mb: usize,
}

/// Resource availability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAvailability {
    /// Available CPU percentage
    pub available_cpu_percent: f32,
    /// Available memory (MB)
    pub available_memory_mb: usize,
    /// Available GPU percentage
    pub available_gpu_percent: Option<f32>,
    /// Available network bandwidth (Mbps)
    pub available_network_mbps: f32,
}

/// Network status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatus {
    /// Network connectivity
    pub connected: bool,
    /// Connection type
    pub connection_type: NetworkConnectionType,
    /// Signal strength (0-100)
    pub signal_strength_percent: u8,
    /// Network quality
    pub quality: NetworkQuality,
    /// Data usage (MB)
    pub data_usage_mb: f32,
}

/// Network connection types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkConnectionType {
    WiFi,
    Cellular,
    Ethernet,
    Bluetooth,
    Unknown,
}

/// Power status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerStatus {
    /// Battery level (%)
    pub battery_level_percent: u8,
    /// Charging status
    pub charging_status: ChargingStatus,
    /// Power save mode enabled
    pub power_save_mode: bool,
    /// Estimated battery life (minutes)
    pub estimated_battery_life_minutes: Option<u32>,
}

/// Charging status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChargingStatus {
    Charging,
    Discharging,
    Full,
    NotCharging,
    Unknown,
}

/// Thermal status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalStatus {
    /// CPU temperature (°C)
    pub cpu_temperature_celsius: f32,
    /// GPU temperature (°C)
    pub gpu_temperature_celsius: Option<f32>,
    /// Thermal state
    pub thermal_state: crate::lifecycle::config::ThermalLevel,
    /// Throttling active
    pub throttling_active: bool,
}

/// Process information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    /// Total processes
    pub total_processes: usize,
    /// Active processes
    pub active_processes: usize,
    /// Background processes
    pub background_processes: usize,
    /// Memory usage by processes (MB)
    pub process_memory_usage_mb: usize,
}

/// System load information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemLoad {
    /// 1-minute load average
    pub load_1min: f32,
    /// 5-minute load average
    pub load_5min: f32,
    /// 15-minute load average
    pub load_15min: f32,
    /// CPU utilization (%)
    pub cpu_utilization_percent: f32,
    /// Memory utilization (%)
    pub memory_utilization_percent: f32,
}

/// App state manager for lifecycle transitions
pub struct AppStateManager {
    current_state: AppState,
    previous_state: AppState,
    state_history: VecDeque<StateTransition>,
    state_listeners: Vec<Box<dyn StateListener>>,
    transition_handlers: HashMap<StateTransition, Box<dyn TransitionHandler>>,
}

/// Trait for state change listeners
pub trait StateListener: Send + Sync {
    fn on_state_change(&self, event: &StateTransitionEvent);
}

/// Trait for state transition handlers
pub trait TransitionHandler: Send + Sync {
    fn handle_transition(
        &self,
        event: &StateTransitionEvent,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

impl AppStateManager {
    /// Create new state manager
    pub fn new() -> Self {
        Self {
            current_state: AppState::Unknown,
            previous_state: AppState::Unknown,
            state_history: VecDeque::new(),
            state_listeners: Vec::new(),
            transition_handlers: HashMap::new(),
        }
    }

    /// Get current app state
    pub fn current_state(&self) -> AppState {
        self.current_state
    }

    /// Get previous app state
    pub fn previous_state(&self) -> AppState {
        self.previous_state
    }

    /// Transition to new state
    pub fn transition_to_state(
        &mut self,
        new_state: AppState,
        reason: TransitionReason,
        context: TransitionContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let transition = self.determine_transition(self.current_state, new_state)?;

        let event = StateTransitionEvent {
            from_state: self.current_state,
            to_state: new_state,
            transition,
            timestamp: Instant::now(),
            reason,
            context,
        };

        // Execute transition handler if registered
        if let Some(handler) = self.transition_handlers.get(&transition) {
            handler.handle_transition(&event)?;
        }

        // Update state
        self.previous_state = self.current_state;
        self.current_state = new_state;
        self.state_history.push_back(transition);

        // Keep history bounded
        if self.state_history.len() > 100 {
            self.state_history.pop_front();
        }

        // Notify listeners
        for listener in &self.state_listeners {
            listener.on_state_change(&event);
        }

        Ok(())
    }

    /// Register state change listener
    pub fn add_state_listener(&mut self, listener: Box<dyn StateListener>) {
        self.state_listeners.push(listener);
    }

    /// Register transition handler
    pub fn add_transition_handler(
        &mut self,
        transition: StateTransition,
        handler: Box<dyn TransitionHandler>,
    ) {
        self.transition_handlers.insert(transition, handler);
    }

    /// Get state transition history
    pub fn get_state_history(&self) -> &VecDeque<StateTransition> {
        &self.state_history
    }

    /// Determine transition type between states
    fn determine_transition(
        &self,
        from: AppState,
        to: AppState,
    ) -> Result<StateTransition, Box<dyn std::error::Error>> {
        match (from, to) {
            (AppState::Launching, AppState::Active) => Ok(StateTransition::LaunchToActive),
            (AppState::Active, AppState::Background) => Ok(StateTransition::ActiveToBackground),
            (AppState::Background, AppState::Active) => Ok(StateTransition::BackgroundToActive),
            (AppState::Active, AppState::Inactive) => Ok(StateTransition::ActiveToInactive),
            (AppState::Inactive, AppState::Active) => Ok(StateTransition::InactiveToActive),
            (AppState::Background, AppState::Suspended) => {
                Ok(StateTransition::BackgroundToSuspended)
            },
            (AppState::Suspended, AppState::Background) => {
                Ok(StateTransition::SuspendedToBackground)
            },
            (_, AppState::Terminating) => Ok(StateTransition::AnyToTerminating),
            _ => Err(format!("Invalid state transition from {:?} to {:?}", from, to).into()),
        }
    }
}

impl Default for AppStateManager {
    fn default() -> Self {
        Self::new()
    }
}
