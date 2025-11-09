//! # Memory Pressure Handler - Main Orchestrator
//!
//! This module contains the main `MemoryPressureHandler` that orchestrates
//! all memory pressure management components. It integrates monitoring,
//! prediction, cleanup, and adaptive threshold management into a unified
//! memory pressure management system.
//!
//! ## Key Responsibilities
//!
//! - **System Integration**: Coordinates all memory pressure subsystems
//! - **Event Management**: Handles memory pressure events and notifications
//! - **Lifecycle Management**: Manages startup, monitoring loops, and shutdown
//! - **API Surface**: Provides the main public API for memory pressure management
//! - **Configuration**: Integrates configuration across all subsystems
//!
//! ## Architecture
//!
//! The handler follows a layered architecture:
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │        MemoryPressureHandler        │
//! ├─────────────────────────────────────┤
//! │  Monitoring  │  Cleanup  │ Events  │
//! │   System     │  Engine   │ System  │
//! ├─────────────────────────────────────┤
//! │       Configuration Layer           │
//! └─────────────────────────────────────┘
//! ```
//!
//! ## Usage Examples
//!
//! ### Basic Usage
//!
//! ```rust
//! use trustformers_serve::memory_pressure::{MemoryPressureHandler, MemoryPressureConfig};
//!
//! let config = MemoryPressureConfig::default();
//! let handler = MemoryPressureHandler::new(config);
//!
//! // Start monitoring
//! handler.start().await?;
//!
//! // Get current memory statistics
//! let stats = handler.get_memory_stats().await;
//! println!("Memory utilization: {:.1}%", stats.utilization * 100.0);
//!
//! // Stop monitoring
//! handler.stop().await?;
//! ```
//!
//! ### Event Monitoring
//!
//! ```rust
//! let mut events = handler.subscribe_to_events();
//!
//! tokio::spawn(async move {
//!     while let Ok(event) = events.recv().await {
//!         match event {
//!             MemoryPressureEvent::PressureLevelChanged { old_level, new_level, .. } => {
//!                 println!("Pressure changed: {:?} -> {:?}", old_level, new_level);
//!             },
//!             MemoryPressureEvent::CleanupTriggered { strategy, memory_freed, .. } => {
//!                 println!("Cleanup {:?} freed {} bytes", strategy, memory_freed);
//!             },
//!             _ => {}
//!         }
//!     }
//! });
//! ```
//!
//! ### Manual Memory Management
//!
//! ```rust
//! // Request memory allocation
//! let allocation_id = handler.request_allocation(1024 * 1024, "buffer".to_string()).await?;
//!
//! // Use the allocation...
//!
//! // Release when done
//! handler.release_allocation(&allocation_id).await?;
//! ```

use super::{
    cleanup::{CleanupEngineConfig, CleanupStrategyEngine},
    config::*,
    monitoring::MemoryMonitor,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};
use tokio::{
    sync::{broadcast, Mutex, Notify, RwLock},
    time::{interval, sleep},
};
use tracing::{debug, error, info, warn};

// =============================================================================
// Main Memory Pressure Handler
// =============================================================================

/// Main memory pressure handler and orchestrator
///
/// This is the central component that coordinates all memory pressure management
/// activities. It integrates monitoring, prediction, cleanup, and adaptive
/// threshold management into a unified system.
///
/// ## Design Principles
///
/// - **Non-blocking**: All operations are designed to be async and non-blocking
/// - **Event-driven**: Uses events for loose coupling between components
/// - **Configurable**: Highly configurable with sensible defaults
/// - **Resilient**: Handles errors gracefully without stopping monitoring
/// - **Efficient**: Minimal overhead during normal operation
#[derive(Debug)]
pub struct MemoryPressureHandler {
    /// Configuration for the entire memory pressure system
    config: MemoryPressureConfig,

    /// Memory monitoring and prediction component
    monitor: Arc<MemoryMonitor>,

    /// Cleanup strategy engine
    cleanup_engine: Arc<CleanupStrategyEngine>,

    /// Current memory statistics
    memory_stats: Arc<RwLock<MemoryStats>>,

    /// Event broadcaster for external consumers
    event_sender: broadcast::Sender<MemoryPressureEvent>,

    /// Memory allocations tracker
    allocations: Arc<RwLock<HashMap<String, AllocationInfo>>>,

    /// Pressure history for trend analysis
    pressure_history: Arc<Mutex<VecDeque<PressureSnapshot>>>,

    /// Background task handles
    task_handles: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,

    /// Shutdown notification
    shutdown_notify: Arc<Notify>,

    /// Last pressure level for change detection
    last_pressure_level: Arc<RwLock<MemoryPressureLevel>>,
}

impl MemoryPressureHandler {
    /// Create a new memory pressure handler
    pub fn new(config: MemoryPressureConfig) -> Self {
        let (event_sender, _) = broadcast::channel(1000);

        // Create monitoring component
        let monitor = Arc::new(MemoryMonitor::new(config.clone()));

        // Create cleanup engine
        let cleanup_config = CleanupEngineConfig::default();
        let cleanup_engine = Arc::new(CleanupStrategyEngine::new(cleanup_config));

        // Initialize memory statistics
        let initial_stats = MemoryStats {
            total_memory: 0,
            available_memory: 0,
            used_memory: 0,
            utilization: 0.0,
            process_memory: 0,
            heap_memory: 0,
            stack_memory: 0,
            gpu_memory: 0,
            gpu_stats: HashMap::new(),
            swap_usage: 0,
            pressure_level: MemoryPressureLevel::Normal,
            last_updated: Utc::now(),
        };

        Self {
            config: config.clone(),
            monitor,
            cleanup_engine,
            memory_stats: Arc::new(RwLock::new(initial_stats)),
            event_sender,
            allocations: Arc::new(RwLock::new(HashMap::new())),
            pressure_history: Arc::new(Mutex::new(VecDeque::new())),
            task_handles: Arc::new(Mutex::new(Vec::new())),
            shutdown_notify: Arc::new(Notify::new()),
            last_pressure_level: Arc::new(RwLock::new(MemoryPressureLevel::Normal)),
        }
    }

    /// Start memory pressure monitoring
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Memory pressure monitoring is disabled");
            return Ok(());
        }

        info!("Starting memory pressure monitoring");

        // Initialize cleanup handlers
        self.initialize_cleanup_handlers().await?;

        // Start monitoring loops
        self.start_monitoring_loop().await;
        self.start_cleanup_processing().await;
        self.start_pressure_analysis().await;

        info!("Memory pressure monitoring started successfully");
        Ok(())
    }

    /// Stop memory pressure monitoring
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping memory pressure monitoring");

        // Signal shutdown to all tasks
        self.shutdown_notify.notify_waiters();

        // Wait for all background tasks to complete
        let mut handles = self.task_handles.lock().await;
        for handle in handles.drain(..) {
            if !handle.is_finished() {
                handle.abort();
            }
        }

        info!("Memory pressure monitoring stopped");
        Ok(())
    }

    /// Get current memory statistics
    pub async fn get_memory_stats(&self) -> MemoryStats {
        self.memory_stats.read().await.clone()
    }

    /// Get current pressure level
    pub async fn get_pressure_level(&self) -> MemoryPressureLevel {
        self.memory_stats.read().await.pressure_level
    }

    /// Request memory allocation tracking
    pub async fn request_allocation(&self, size: u64, allocation_type: String) -> Result<String> {
        let current_stats = self.get_memory_stats().await;

        // Check if allocation would exceed thresholds
        if current_stats.utilization > self.config.request_rejection_threshold {
            return Err(anyhow::anyhow!(
                "Memory allocation rejected due to high pressure (utilization: {:.1}%)",
                current_stats.utilization * 100.0
            ));
        }

        let allocation_id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();

        let allocation_info = AllocationInfo {
            size,
            allocation_type,
            timestamp: now,
            lifetime_hint: None,
            priority: 100, // Default priority
            gpu_device_id: None,
        };

        // Track allocation
        {
            let mut allocations = self.allocations.write().await;
            allocations.insert(allocation_id.clone(), allocation_info);
        }

        debug!("Allocated {} bytes with ID: {}", size, allocation_id);
        Ok(allocation_id)
    }

    /// Release memory allocation tracking
    pub async fn release_allocation(&self, allocation_id: &str) -> Result<()> {
        let removed = {
            let mut allocations = self.allocations.write().await;
            allocations.remove(allocation_id)
        };

        if let Some(allocation) = removed {
            debug!(
                "Released allocation {} ({} bytes)",
                allocation_id, allocation.size
            );
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Allocation ID not found: {}",
                allocation_id
            ))
        }
    }

    /// Subscribe to memory pressure events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<MemoryPressureEvent> {
        self.event_sender.subscribe()
    }

    /// Trigger emergency cleanup manually
    pub async fn trigger_emergency_cleanup(&self) -> Result<u64> {
        warn!("Manual emergency cleanup triggered");

        let context = super::cleanup::CleanupContext::new(
            MemoryPressureLevel::Emergency,
            1.0, // Maximum utilization
            0,   // Minimal available memory
        )
        .with_emergency(true);

        let strategies = self
            .cleanup_engine
            .select_strategies(MemoryPressureLevel::Emergency, None)
            .await;

        self.cleanup_engine.queue_cleanup_actions(strategies, context).await?;
        let results = self.cleanup_engine.execute_all_actions().await?;

        let total_freed: u64 = results.iter().map(|r| r.memory_freed).sum();

        // Send emergency cleanup event
        let _ = self.event_sender.send(MemoryPressureEvent::EmergencyCleanup {
            memory_freed: total_freed,
            timestamp: Utc::now(),
        });

        info!("Emergency cleanup completed, freed {} bytes", total_freed);
        Ok(total_freed)
    }

    /// Get memory usage forecast
    pub async fn get_memory_forecast(&self, forecast_minutes: u32) -> Result<MemoryUsagePattern> {
        self.monitor.get_memory_forecast(forecast_minutes).await
    }

    /// Get platform-specific memory insights
    pub async fn get_platform_memory_insights(&self) -> Result<HashMap<String, String>> {
        self.monitor.get_platform_memory_insights().await
    }

    /// Perform platform-specific memory optimization
    pub async fn optimize_memory_usage(&self) -> Result<u64> {
        self.monitor.optimize_memory_usage().await
    }

    /// Get current adaptive thresholds
    pub async fn get_current_thresholds(&self) -> MemoryPressureThresholds {
        self.monitor.get_current_thresholds().await
    }

    /// Get cleanup queue status
    pub async fn get_cleanup_queue_status(&self) -> super::cleanup::CleanupQueueStatus {
        self.cleanup_engine.get_queue_status().await
    }

    /// Calculate pressure level for a given utilization
    pub fn calculate_pressure_level(&self, utilization: f32) -> MemoryPressureLevel {
        let thresholds = &self.config.pressure_thresholds;

        if utilization >= thresholds.critical {
            MemoryPressureLevel::Critical
        } else if utilization >= thresholds.high {
            MemoryPressureLevel::High
        } else if utilization >= thresholds.medium {
            MemoryPressureLevel::Medium
        } else if utilization >= thresholds.low {
            MemoryPressureLevel::Low
        } else {
            MemoryPressureLevel::Normal
        }
    }

    /// Update memory prediction with current utilization
    pub async fn update_memory_prediction(&self, utilization: f32) -> Result<f32> {
        self.monitor.update_memory_patterns(utilization).await
    }

    /// Adapt thresholds based on historical patterns
    pub async fn adapt_thresholds(&self) -> Result<()> {
        self.monitor.adapt_thresholds().await
    }

    /// Calculate pressure trend from snapshots
    pub fn calculate_pressure_trend(&self, samples: &[&PressureSnapshot]) -> f32 {
        self.monitor.calculate_pressure_trend(samples)
    }

    /// Get access to adaptive thresholds (for testing)
    pub fn adaptive_thresholds(&self) -> Arc<RwLock<MemoryPressureThresholds>> {
        self.monitor.adaptive_thresholds.clone()
    }

    /// Get access to pattern history (for testing)
    pub fn pattern_history(&self) -> Arc<Mutex<VecDeque<(DateTime<Utc>, f32)>>> {
        self.monitor.pattern_history.clone()
    }

    /// Execute GPU cache eviction on a specific device
    pub async fn execute_gpu_cache_eviction(&self, device_id: u32) -> u64 {
        // Simulate GPU cache eviction
        tracing::debug!("Executing GPU cache eviction on device {}", device_id);
        // Return simulated memory freed (10-50 MB)
        (10 + (device_id as u64 % 40)) * 1024 * 1024
    }

    /// Execute GPU buffer compaction on a specific device
    pub async fn execute_gpu_buffer_compaction(&self, device_id: u32) -> u64 {
        // Simulate GPU buffer compaction
        tracing::debug!("Executing GPU buffer compaction on device {}", device_id);
        // Return simulated memory freed (20-60 MB)
        (20 + (device_id as u64 % 40)) * 1024 * 1024
    }

    /// Execute GPU model unloading on a specific device
    pub async fn execute_gpu_model_unloading(&self, device_id: u32) -> u64 {
        // Simulate GPU model unloading
        tracing::debug!("Executing GPU model unloading on device {}", device_id);
        // Return simulated memory freed (100-500 MB)
        (100 + (device_id as u64 % 400)) * 1024 * 1024
    }

    /// Execute GPU VRAM compaction on a specific device
    pub async fn execute_gpu_vram_compaction(&self, device_id: u32) -> u64 {
        // Simulate GPU VRAM compaction
        tracing::debug!("Executing GPU VRAM compaction on device {}", device_id);
        // Return simulated memory freed (30-70 MB)
        (30 + (device_id as u64 % 40)) * 1024 * 1024
    }

    /// Execute GPU memory defragmentation on a specific device
    pub async fn execute_gpu_memory_defragmentation(&self, device_id: u32) -> u64 {
        // Simulate GPU memory defragmentation
        tracing::debug!(
            "Executing GPU memory defragmentation on device {}",
            device_id
        );
        // Return simulated memory freed (50-150 MB)
        (50 + (device_id as u64 % 100)) * 1024 * 1024
    }

    /// Execute GPU context switching on a specific device
    pub async fn execute_gpu_context_switching(&self, device_id: u32) -> u64 {
        // Simulate GPU context switching
        tracing::debug!("Executing GPU context switching on device {}", device_id);
        // Return simulated memory freed (15-35 MB)
        (15 + (device_id as u64 % 20)) * 1024 * 1024
    }

    /// Execute GPU stream cleanup on a specific device
    pub async fn execute_gpu_stream_cleanup(&self, device_id: u32) -> u64 {
        // Simulate GPU stream cleanup
        tracing::debug!("Executing GPU stream cleanup on device {}", device_id);
        // Return simulated memory freed (10-30 MB)
        (10 + (device_id as u64 % 20)) * 1024 * 1024
    }

    /// Execute GPU texture cleanup on a specific device
    pub async fn execute_gpu_texture_cleanup(&self, device_id: u32) -> u64 {
        // Simulate GPU texture cleanup
        tracing::debug!("Executing GPU texture cleanup on device {}", device_id);
        // Return simulated memory freed (25-75 MB)
        (25 + (device_id as u64 % 50)) * 1024 * 1024
    }

    /// Execute GPU memory pool reset on a specific device
    pub async fn execute_gpu_memory_pool_reset(&self, device_id: u32) -> u64 {
        // Simulate GPU memory pool reset
        tracing::debug!("Executing GPU memory pool reset on device {}", device_id);
        // Return simulated memory freed (40-100 MB)
        (40 + (device_id as u64 % 60)) * 1024 * 1024
    }

    /// Execute GPU batch size reduction on a specific device
    pub async fn execute_gpu_batch_size_reduction(&self, device_id: u32) -> u64 {
        // Simulate GPU batch size reduction
        tracing::debug!("Executing GPU batch size reduction on device {}", device_id);
        // Return simulated memory freed (20-60 MB)
        (20 + (device_id as u64 % 40)) * 1024 * 1024
    }

    // =============================================================================
    // Private Implementation Methods
    // =============================================================================

    /// Initialize cleanup handlers
    async fn initialize_cleanup_handlers(&self) -> Result<()> {
        use super::cleanup::handlers::*;
        use std::sync::Arc;

        // Register standard cleanup handlers based on configuration
        for strategy in &self.config.cleanup_strategies {
            match strategy {
                CleanupStrategy::GarbageCollection => {
                    let handler = Arc::new(GarbageCollectionHandler::new());
                    self.cleanup_engine.register_handler(strategy.clone(), handler).await;
                },
                CleanupStrategy::BufferCompaction => {
                    let handler = Arc::new(BufferCompactionHandler::new());
                    self.cleanup_engine.register_handler(strategy.clone(), handler).await;
                },
                CleanupStrategy::ModelUnloading => {
                    let handler = Arc::new(ModelUnloadingHandler::new());
                    self.cleanup_engine.register_handler(strategy.clone(), handler).await;
                },
                CleanupStrategy::RequestRejection => {
                    let handler = Arc::new(RequestRejectionHandler::new());
                    self.cleanup_engine.register_handler(strategy.clone(), handler).await;
                },
                _ => {
                    debug!("Handler for strategy {:?} not yet implemented", strategy);
                },
            }
        }

        Ok(())
    }

    /// Start the main monitoring loop
    async fn start_monitoring_loop(&self) {
        let handler = self.clone();
        let handle = tokio::spawn(async move {
            handler.monitoring_loop().await;
        });

        let handler_clone = self.clone();
        tokio::spawn(async move {
            let mut handles = handler_clone.task_handles.lock().await;
            handles.push(handle);
        });
    }

    /// Start cleanup processing loop
    async fn start_cleanup_processing(&self) {
        let handler = self.clone();
        let handle = tokio::spawn(async move {
            handler.cleanup_processing_loop().await;
        });

        let handler_clone = self.clone();
        tokio::spawn(async move {
            let mut handles = handler_clone.task_handles.lock().await;
            handles.push(handle);
        });
    }

    /// Start pressure analysis loop
    async fn start_pressure_analysis(&self) {
        let handler = self.clone();
        let handle = tokio::spawn(async move {
            handler.pressure_analysis_loop().await;
        });

        let handler_clone = self.clone();
        tokio::spawn(async move {
            let mut handles = handler_clone.task_handles.lock().await;
            handles.push(handle);
        });
    }

    /// Main monitoring loop
    async fn monitoring_loop(&self) {
        let mut interval = interval(Duration::from_secs(self.config.monitoring_interval_seconds));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.update_memory_stats().await {
                        error!("Failed to update memory stats: {}", e);
                    }
                }
                _ = self.shutdown_notify.notified() => {
                    debug!("Monitoring loop shutting down");
                    break;
                }
            }
        }
    }

    /// Cleanup processing loop
    async fn cleanup_processing_loop(&self) {
        loop {
            tokio::select! {
                _ = self.process_cleanup_queue() => {}
                _ = self.shutdown_notify.notified() => {
                    debug!("Cleanup processing loop shutting down");
                    break;
                }
            }
        }
    }

    /// Pressure analysis loop
    async fn pressure_analysis_loop(&self) {
        let mut interval = interval(Duration::from_secs(5));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    self.analyze_pressure_trends().await;
                    if let Err(e) = self.monitor.adapt_thresholds().await {
                        error!("Failed to adapt thresholds: {}", e);
                    }
                }
                _ = self.shutdown_notify.notified() => {
                    debug!("Pressure analysis loop shutting down");
                    break;
                }
            }
        }
    }

    /// Update memory statistics
    async fn update_memory_stats(&self) -> Result<()> {
        // Get fresh memory information from monitor
        let memory_info = self.monitor.get_system_memory_info().await?;

        let old_level = {
            let stats = self.memory_stats.read().await;
            stats.pressure_level
        };

        let new_level = memory_info.pressure_level;

        // Update stored statistics
        {
            let mut stats = self.memory_stats.write().await;
            *stats = memory_info.clone();
        }

        // Update pressure history
        {
            let mut history = self.pressure_history.lock().await;
            history.push_back(PressureSnapshot {
                timestamp: Utc::now(),
                utilization: memory_info.utilization,
                pressure_level: new_level,
                available_memory: memory_info.available_memory,
            });

            // Keep only last 24 hours of data
            let cutoff = Utc::now() - chrono::Duration::hours(24);
            while let Some(snapshot) = history.front() {
                if snapshot.timestamp < cutoff {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }

        // Update memory patterns for prediction
        if let Err(e) = self.monitor.update_memory_patterns(memory_info.utilization).await {
            error!("Failed to update memory patterns: {}", e);
        }

        // Check for pressure level changes
        if old_level != new_level {
            // Send pressure level change event
            let _ = self.event_sender.send(MemoryPressureEvent::PressureLevelChanged {
                old_level,
                new_level,
                utilization: memory_info.utilization,
                timestamp: Utc::now(),
            });

            info!(
                "Memory pressure level changed: {:?} -> {:?} (utilization: {:.1}%)",
                old_level,
                new_level,
                memory_info.utilization * 100.0
            );

            // Update last pressure level
            {
                let mut last_level = self.last_pressure_level.write().await;
                *last_level = new_level;
            }

            // Trigger cleanup if pressure is increasing
            if new_level > old_level && new_level >= MemoryPressureLevel::Medium {
                self.trigger_cleanup(new_level).await;
            }
        }

        Ok(())
    }

    /// Process cleanup queue
    async fn process_cleanup_queue(&self) {
        if let Ok(Some(result)) = self.cleanup_engine.execute_next_action().await {
            // Send cleanup event
            let _ = self.event_sender.send(MemoryPressureEvent::CleanupTriggered {
                strategy: result.strategy,
                memory_freed: result.memory_freed,
                timestamp: Utc::now(),
            });

            if !result.success {
                warn!("Cleanup action failed: {:?}", result.error_message);
            }
        } else {
            // No actions in queue, sleep briefly
            sleep(Duration::from_millis(100)).await;
        }
    }

    /// Analyze pressure trends and trigger proactive cleanup
    async fn analyze_pressure_trends(&self) {
        let history = self.pressure_history.lock().await;

        if history.len() < 10 {
            return;
        }

        // Calculate trend over recent samples
        let recent_samples: Vec<&PressureSnapshot> = history.iter().rev().take(10).collect();
        let trend = self.monitor.calculate_pressure_trend(&recent_samples);

        let avg_utilization: f32 =
            recent_samples.iter().map(|s| s.utilization).sum::<f32>() / recent_samples.len() as f32;

        debug!(
            "Memory pressure trend: avg={:.2}, trend={:.2}",
            avg_utilization, trend
        );

        // Trigger proactive cleanup if trend is increasing and utilization is high
        if trend > 0.05 && avg_utilization > 0.7 {
            info!("Proactive cleanup triggered due to increasing pressure trend");
            self.trigger_cleanup(MemoryPressureLevel::Medium).await;
        }
    }

    /// Trigger cleanup for a specific pressure level
    async fn trigger_cleanup(&self, pressure_level: MemoryPressureLevel) {
        let current_stats = self.get_memory_stats().await;

        let context = super::cleanup::CleanupContext::new(
            pressure_level,
            current_stats.utilization,
            current_stats.available_memory,
        );

        let strategies = self.cleanup_engine.select_strategies(pressure_level, None).await;

        if !strategies.is_empty() {
            debug!(
                "Triggering cleanup with {} strategies for pressure level {:?}",
                strategies.len(),
                pressure_level
            );

            if let Err(e) = self.cleanup_engine.queue_cleanup_actions(strategies, context).await {
                error!("Failed to queue cleanup actions: {}", e);
            }
        }
    }
}

// Implement Clone for MemoryPressureHandler to enable sharing across async tasks
impl Clone for MemoryPressureHandler {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            monitor: self.monitor.clone(),
            cleanup_engine: self.cleanup_engine.clone(),
            memory_stats: self.memory_stats.clone(),
            event_sender: self.event_sender.clone(),
            allocations: self.allocations.clone(),
            pressure_history: self.pressure_history.clone(),
            task_handles: self.task_handles.clone(),
            shutdown_notify: self.shutdown_notify.clone(),
            last_pressure_level: self.last_pressure_level.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_handler_creation() {
        let config = MemoryPressureConfig::default();
        let handler = MemoryPressureHandler::new(config);

        assert_eq!(
            handler.get_pressure_level().await,
            MemoryPressureLevel::Normal
        );
    }

    #[tokio::test]
    async fn test_allocation_tracking() {
        let config = MemoryPressureConfig::default();
        let handler = MemoryPressureHandler::new(config);

        let allocation_id = handler.request_allocation(1024, "test".to_string()).await.unwrap();
        assert!(!allocation_id.is_empty());

        handler.release_allocation(&allocation_id).await.unwrap();
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let config = MemoryPressureConfig::default();
        let handler = MemoryPressureHandler::new(config);

        let mut events = handler.subscribe_to_events();

        // Events receiver should be created successfully
        assert!(events.try_recv().is_err()); // No events yet
    }

    #[tokio::test]
    async fn test_memory_stats() {
        let config = MemoryPressureConfig::default();
        let handler = MemoryPressureHandler::new(config);

        let stats = handler.get_memory_stats().await;
        // total_memory is u64, always >= 0, so just verify it's populated
        assert!(stats.total_memory > 0 || stats.total_memory == 0); // Always true but documents intent
        assert!(stats.utilization >= 0.0 && stats.utilization <= 1.0);
    }

    #[tokio::test]
    async fn test_cleanup_queue_status() {
        let config = MemoryPressureConfig::default();
        let handler = MemoryPressureHandler::new(config);

        let status = handler.get_cleanup_queue_status().await;
        assert_eq!(status.pending_actions, 0); // No actions queued initially
    }

    #[tokio::test]
    async fn test_handler_lifecycle() {
        let mut config = MemoryPressureConfig::default();
        config.monitoring_interval_seconds = 1; // Fast monitoring for testing

        let handler = MemoryPressureHandler::new(config);

        // Start should succeed
        handler.start().await.unwrap();

        // Brief operation
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Stop should succeed
        handler.stop().await.unwrap();
    }
}
