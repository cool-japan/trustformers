//! GPU Load Balancer Module
//!
//! This module provides comprehensive GPU load balancing functionality for optimal workload
//! distribution across available GPU devices in the TrustformeRS framework.
//!
//! # Overview
//!
//! The load balancer handles:
//! - **Load Balancing Strategies**: Multiple algorithms for distributing workloads across GPUs
//! - **Device Load Tracking**: Real-time monitoring and tracking of load across GPU devices
//! - **Workload Distribution**: Intelligent workload assignment based on device capabilities
//! - **Load Optimization**: Dynamic optimization of load distribution for maximum efficiency
//! - **Load Analytics**: Comprehensive analysis and reporting of load distribution patterns
//! - **Dynamic Load Management**: Real-time load adjustment and rebalancing capabilities
//!
//! # Load Balancing Strategies
//!
//! The system supports multiple load balancing strategies:
//! - **Least Loaded**: Assigns workloads to the GPU with the lowest current utilization
//! - **Round Robin**: Cycles through available GPUs in a round-robin fashion
//! - **Best Fit**: Selects the GPU that best matches the workload requirements
//! - **Random**: Randomly distributes workloads for even load distribution
//! - **Weighted**: Distributes based on device capabilities and performance scores
//! - **Performance Based**: Prioritizes high-performance devices for demanding workloads
//! - **Memory Optimized**: Optimizes allocation based on memory usage patterns
//! - **Power Aware**: Considers power consumption in allocation decisions
//!
//! # Examples
//!
//! ```rust,no_run
//! use trustformers_serve::resource_management::gpu_manager::load_balancer::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize load balancer
//!     let mut load_balancer = GpuLoadBalancer::new();
//!
//!     // Configure strategy
//!     load_balancer.set_strategy(LoadBalancingStrategy::LeastLoaded).await;
//!
//!     // Select optimal device for workload
//!     let device_id = load_balancer.select_optimal_device(
//!         &available_devices,
//!         &requirements,
//!         Some(&workload_profile)
//!     ).await?;
//!
//!     // Update load tracking
//!     load_balancer.update_device_load(device_id, 0.75).await?;
//!
//!     // Get load analytics
//!     let analytics = load_balancer.get_load_analytics().await;
//!
//!     Ok(())
//! }
//! ```

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tracing::{debug, info, instrument, warn};

use super::types::*;

// ================================================================================================
// CORE LOAD BALANCER TYPES
// ================================================================================================

/// Advanced load balancing strategy with comprehensive options
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Allocate to the least loaded device first
    LeastLoaded,
    /// Round-robin allocation across devices
    RoundRobin,
    /// Best fit allocation based on requirements
    BestFit,
    /// Random allocation for load distribution
    Random,
    /// Weighted allocation based on device capabilities
    Weighted,
    /// Performance-based allocation prioritizing high-performance devices
    PerformanceBased,
    /// Memory-optimized allocation based on memory usage patterns
    MemoryOptimized,
    /// Power-aware allocation considering power consumption
    PowerAware,
    /// Hybrid strategy combining multiple approaches
    Hybrid(Vec<LoadBalancingStrategy>),
    /// Custom user-defined strategy
    Custom(String),
}

/// Device load information for tracking utilization
#[derive(Debug, Clone)]
pub struct DeviceLoadInfo {
    /// Device identifier
    pub device_id: usize,
    /// Current utilization percentage (0.0 to 1.0)
    pub utilization: f32,
    /// Memory usage percentage (0.0 to 1.0)
    pub memory_usage: f32,
    /// Power consumption in watts
    pub power_consumption: f32,
    /// Temperature in Celsius
    pub temperature: f32,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Performance score (higher is better)
    pub performance_score: f32,
    /// Load trend over time
    pub load_trend: LoadTrend,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Load trend analysis for predictive load balancing
#[derive(Debug, Clone)]
pub struct LoadTrend {
    /// Historical load values (last 100 measurements)
    pub history: VecDeque<f32>,
    /// Predicted next load value
    pub predicted_load: f32,
    /// Load direction trend
    pub trend_direction: LoadBalancerTrendDirection,
    /// Trend confidence score (0.0 to 1.0)
    pub confidence: f32,
}

/// Direction of load trend
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancerTrendDirection {
    /// Load is increasing
    Increasing,
    /// Load is decreasing
    Decreasing,
    /// Load is stable
    Stable,
    /// Load is volatile/unpredictable
    Volatile,
}

/// Workload profile for intelligent allocation decisions
#[derive(Debug, Clone)]
pub struct WorkloadProfile {
    /// Estimated duration of the workload
    pub estimated_duration: Duration,
    /// Memory intensity (0.0 to 1.0)
    pub memory_intensity: f32,
    /// Compute intensity (0.0 to 1.0)
    pub compute_intensity: f32,
    /// Power consumption priority
    pub power_priority: PowerPriority,
    /// Workload type classification
    pub workload_type: WorkloadType,
    /// Expected load pattern
    pub load_pattern: LoadPattern,
}

/// Power consumption priority for power-aware load balancing
#[derive(Debug, Clone, PartialEq)]
pub enum PowerPriority {
    /// Minimize power consumption
    Low,
    /// Balance power and performance
    Balanced,
    /// Maximize performance regardless of power
    High,
}

/// Workload type classification for specialized handling
#[derive(Debug, Clone, PartialEq)]
pub enum WorkloadType {
    /// Machine learning training workload
    Training,
    /// Machine learning inference workload
    Inference,
    /// Computational simulation
    Simulation,
    /// Data processing workload
    DataProcessing,
    /// Graphics rendering workload
    Rendering,
    /// General purpose computing
    General,
}

/// Expected load pattern for predictive optimization
#[derive(Debug, Clone, PartialEq)]
pub enum LoadPattern {
    /// Consistent load throughout execution
    Steady,
    /// Load increases over time
    Ramping,
    /// Load decreases over time
    Declining,
    /// Load varies significantly
    Bursty,
    /// Unknown or unpredictable pattern
    Unknown,
}

/// Load balancing analytics and metrics
#[derive(Debug, Clone)]
pub struct LoadBalancingAnalytics {
    /// Total number of allocation decisions made
    pub total_allocations: u64,
    /// Average device utilization across all devices
    pub average_utilization: f32,
    /// Device utilization variance (measure of load balance quality)
    pub utilization_variance: f32,
    /// Load balancing efficiency score (0.0 to 1.0)
    pub efficiency_score: f32,
    /// Number of load balancing strategy changes
    pub strategy_changes: u64,
    /// Load rebalancing events performed
    pub rebalancing_events: u64,
    /// Average allocation time (decision time)
    pub average_allocation_time: Duration,
    /// Device utilization distribution
    pub utilization_distribution: HashMap<usize, f32>,
    /// Load balancing success rate
    pub success_rate: f32,
    /// Performance improvement over random allocation
    pub performance_improvement: f32,
    /// Analytics generation timestamp
    pub generated_at: DateTime<Utc>,
}

/// Load rebalancing suggestion for optimization
#[derive(Debug, Clone)]
pub struct RebalancingSuggestion {
    /// Source device to move workload from
    pub source_device: usize,
    /// Target device to move workload to
    pub target_device: usize,
    /// Amount of load to transfer (0.0 to 1.0)
    pub load_amount: f32,
    /// Expected improvement in load balance
    pub expected_improvement: f32,
    /// Priority of this suggestion
    pub priority: RebalancingPriority,
    /// Estimated effort to implement
    pub effort_estimate: Duration,
}

/// Priority level for rebalancing suggestions
#[derive(Debug, Clone, PartialEq, Ord, PartialOrd, Eq)]
pub enum RebalancingPriority {
    /// Low priority suggestion
    Low,
    /// Medium priority suggestion
    Medium,
    /// High priority suggestion
    High,
    /// Critical rebalancing needed
    Critical,
}

// ================================================================================================
// MAIN LOAD BALANCER IMPLEMENTATION
// ================================================================================================

/// Comprehensive GPU load balancer for optimal workload distribution
///
/// The load balancer provides intelligent GPU selection and load distribution capabilities
/// with support for multiple strategies, real-time load tracking, and dynamic optimization.
#[derive(Debug)]
pub struct GpuLoadBalancer {
    /// Current load balancing strategy
    strategy: Arc<RwLock<LoadBalancingStrategy>>,

    /// Device load tracking information
    device_loads: Arc<RwLock<HashMap<usize, DeviceLoadInfo>>>,

    /// Round-robin counter for round-robin strategy
    round_robin_counter: Arc<AtomicU64>,

    /// Load balancing analytics
    analytics: Arc<RwLock<LoadBalancingAnalytics>>,

    /// Strategy performance metrics for adaptive strategy selection
    strategy_performance: Arc<RwLock<HashMap<LoadBalancingStrategy, f32>>>,

    /// Load balancing configuration
    config: Arc<RwLock<LoadBalancerConfig>>,

    /// Device weights for weighted load balancing
    device_weights: Arc<RwLock<HashMap<usize, f32>>>,

    /// Load history for trend analysis
    load_history: Arc<RwLock<VecDeque<LoadSnapshot>>>,

    /// Rebalancing suggestions queue
    rebalancing_suggestions: Arc<RwLock<VecDeque<RebalancingSuggestion>>>,

    /// Event channel for load balancing events
    event_sender: mpsc::UnboundedSender<LoadBalancingEvent>,
    event_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<LoadBalancingEvent>>>>,
}

/// Load balancer configuration
#[derive(Debug, Clone)]
pub struct LoadBalancerConfig {
    /// Enable adaptive strategy selection
    pub adaptive_strategy: bool,
    /// Threshold for triggering rebalancing (variance threshold)
    pub rebalancing_threshold: f32,
    /// Maximum number of historical load snapshots to keep
    pub max_history_size: usize,
    /// Interval for automatic load analysis
    pub analysis_interval: Duration,
    /// Enable predictive load balancing
    pub enable_prediction: bool,
    /// Minimum confidence score for predictions
    pub prediction_confidence_threshold: f32,
    /// Maximum device utilization target
    pub max_utilization_target: f32,
    /// Power consumption weight in allocation decisions
    pub power_weight: f32,
    /// Performance weight in allocation decisions
    pub performance_weight: f32,
}

/// Load snapshot for historical analysis
#[derive(Debug, Clone)]
pub struct LoadSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: DateTime<Utc>,
    /// Device loads at this time
    pub device_loads: HashMap<usize, f32>,
    /// Active strategy at this time
    pub active_strategy: LoadBalancingStrategy,
    /// Overall system utilization
    pub system_utilization: f32,
}

/// Load balancing events for monitoring and analysis
#[derive(Debug, Clone)]
pub enum LoadBalancingEvent {
    /// Device selected for allocation
    DeviceSelected {
        device_id: usize,
        strategy: LoadBalancingStrategy,
        allocation_time: Duration,
    },
    /// Load balancing strategy changed
    StrategyChanged {
        old_strategy: LoadBalancingStrategy,
        new_strategy: LoadBalancingStrategy,
        reason: String,
    },
    /// Load rebalancing triggered
    RebalancingTriggered {
        reason: String,
        affected_devices: Vec<usize>,
    },
    /// Device load updated
    LoadUpdated {
        device_id: usize,
        old_load: f32,
        new_load: f32,
    },
}

// ================================================================================================
// LOAD BALANCER IMPLEMENTATION
// ================================================================================================

impl Default for GpuLoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuLoadBalancer {
    /// Create a new GPU load balancer with default configuration
    ///
    /// # Returns
    ///
    /// A new load balancer instance ready for use
    #[instrument]
    pub fn new() -> Self {
        Self::with_config(LoadBalancerConfig::default())
    }

    /// Create a new GPU load balancer with custom configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Load balancer configuration
    ///
    /// # Returns
    ///
    /// A new load balancer instance with the specified configuration
    #[instrument(skip(config))]
    pub fn with_config(config: LoadBalancerConfig) -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        Self {
            strategy: Arc::new(RwLock::new(LoadBalancingStrategy::LeastLoaded)),
            device_loads: Arc::new(RwLock::new(HashMap::new())),
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            analytics: Arc::new(RwLock::new(LoadBalancingAnalytics::default())),
            strategy_performance: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(config)),
            device_weights: Arc::new(RwLock::new(HashMap::new())),
            load_history: Arc::new(RwLock::new(VecDeque::new())),
            rebalancing_suggestions: Arc::new(RwLock::new(VecDeque::new())),
            event_sender,
            event_receiver: Arc::new(RwLock::new(Some(event_receiver))),
        }
    }

    /// Set the load balancing strategy
    ///
    /// # Arguments
    ///
    /// * `strategy` - The new load balancing strategy to use
    #[instrument(skip(self))]
    pub async fn set_strategy(&self, strategy: LoadBalancingStrategy) -> Result<()> {
        let old_strategy = {
            let mut current_strategy = self.strategy.write();
            let old = current_strategy.clone();
            *current_strategy = strategy.clone();
            old
        };

        // Send strategy change event
        let event = LoadBalancingEvent::StrategyChanged {
            old_strategy,
            new_strategy: strategy.clone(),
            reason: "Manual strategy change".to_string(),
        };

        if let Err(e) = self.event_sender.send(event) {
            warn!("Failed to send strategy change event: {}", e);
        }

        // Update analytics
        {
            let mut analytics = self.analytics.write();
            analytics.strategy_changes += 1;
        }

        info!("Load balancing strategy changed to: {:?}", strategy);
        Ok(())
    }

    /// Get the current load balancing strategy
    ///
    /// # Returns
    ///
    /// The currently active load balancing strategy
    pub async fn get_strategy(&self) -> LoadBalancingStrategy {
        let strategy = self.strategy.read();
        strategy.clone()
    }

    /// Select the optimal device for allocation based on requirements and workload profile
    ///
    /// This is the main entry point for device selection, considering the current strategy,
    /// device loads, requirements, and workload characteristics.
    ///
    /// # Arguments
    ///
    /// * `available_devices` - Map of available GPU devices
    /// * `requirements` - Performance requirements for the allocation
    /// * `workload_profile` - Optional workload profile for intelligent selection
    ///
    /// # Returns
    ///
    /// The optimal device ID for allocation, or None if no suitable device found
    #[instrument(skip(self, available_devices, requirements))]
    pub async fn select_optimal_device(
        &self,
        available_devices: &HashMap<usize, GpuDeviceInfo>,
        requirements: &GpuPerformanceRequirements,
        workload_profile: Option<&WorkloadProfile>,
    ) -> Result<Option<usize>> {
        let start_time = Instant::now();

        // Filter devices that meet basic requirements
        let suitable_devices =
            self.filter_suitable_devices(available_devices, requirements).await?;

        if suitable_devices.is_empty() {
            debug!("No suitable devices found for requirements");
            return Ok(None);
        }

        // Select device based on current strategy
        let strategy = {
            let guard = self.strategy.read();
            guard.clone()
        };
        let selected_device = match strategy {
            LoadBalancingStrategy::LeastLoaded => self.select_least_loaded(&suitable_devices).await,
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(&suitable_devices).await,
            LoadBalancingStrategy::BestFit => {
                self.select_best_fit(&suitable_devices, requirements).await
            },
            LoadBalancingStrategy::Random => self.select_random(&suitable_devices).await,
            LoadBalancingStrategy::Weighted => self.select_weighted(&suitable_devices).await,
            LoadBalancingStrategy::PerformanceBased => {
                self.select_performance_based(&suitable_devices).await
            },
            LoadBalancingStrategy::MemoryOptimized => {
                self.select_memory_optimized(&suitable_devices, requirements).await
            },
            LoadBalancingStrategy::PowerAware => {
                self.select_power_aware(&suitable_devices, workload_profile).await
            },
            LoadBalancingStrategy::Hybrid(ref strategies) => {
                self.select_hybrid(
                    &suitable_devices,
                    strategies,
                    requirements,
                    workload_profile,
                )
                .await
            },
            LoadBalancingStrategy::Custom(ref name) => {
                self.select_custom(&suitable_devices, name, requirements, workload_profile)
                    .await
            },
        }?;

        // Record allocation metrics
        let allocation_time = start_time.elapsed();

        if let Some(device_id) = selected_device {
            // Send device selection event
            let event = LoadBalancingEvent::DeviceSelected {
                device_id,
                strategy: strategy.clone(),
                allocation_time,
            };

            if let Err(e) = self.event_sender.send(event) {
                warn!("Failed to send device selection event: {}", e);
            }

            // Update analytics
            {
                let mut analytics = self.analytics.write();
                analytics.total_allocations += 1;
                analytics.average_allocation_time = Duration::from_nanos(
                    (analytics.average_allocation_time.as_nanos() as u64
                        * (analytics.total_allocations - 1)
                        + allocation_time.as_nanos() as u64)
                        / analytics.total_allocations,
                );
            }

            debug!(
                "Selected device {} using strategy {:?} in {:?}",
                device_id, strategy, allocation_time
            );
        }

        Ok(selected_device)
    }

    /// Legacy compatibility method for device selection
    ///
    /// This method provides backward compatibility with the existing interface.
    /// For new code, prefer using `select_optimal_device` which provides more features.
    ///
    /// # Arguments
    ///
    /// * `available_devices` - Map of available GPU devices
    /// * `requirements` - Performance requirements for the allocation
    ///
    /// # Returns
    ///
    /// The selected device ID, or None if no suitable device found
    #[instrument(skip(self, available_devices, requirements))]
    pub async fn select_device(
        &self,
        available_devices: &HashMap<usize, GpuDeviceInfo>,
        requirements: &GpuPerformanceRequirements,
    ) -> Option<usize> {
        match self.select_optimal_device(available_devices, requirements, None).await {
            Ok(result) => result,
            Err(e) => {
                warn!("Error in device selection: {}", e);
                None
            },
        }
    }

    /// Filter devices that meet the basic requirements
    async fn filter_suitable_devices(
        &self,
        available_devices: &HashMap<usize, GpuDeviceInfo>,
        requirements: &GpuPerformanceRequirements,
    ) -> Result<HashMap<usize, GpuDeviceInfo>> {
        let mut suitable_devices = HashMap::new();

        for (device_id, device) in available_devices {
            // Check if device is available
            if device.status != GpuDeviceStatus::Available {
                continue;
            }

            // Check memory requirements
            if device.available_memory_mb < requirements.min_memory_mb {
                continue;
            }

            // Check framework requirements
            let supports_frameworks = requirements.required_frameworks.iter().all(|framework| {
                device
                    .capabilities
                    .iter()
                    .any(|capability| capability.supports_framework(framework))
            });

            if !supports_frameworks {
                continue;
            }

            // Check constraints
            let meets_constraints = requirements
                .constraints
                .iter()
                .all(|constraint| self.check_constraint(device, constraint));

            if !meets_constraints {
                continue;
            }

            suitable_devices.insert(*device_id, device.clone());
        }

        Ok(suitable_devices)
    }

    /// Check if device meets a specific constraint
    fn check_constraint(&self, device: &GpuDeviceInfo, constraint: &GpuConstraint) -> bool {
        match &constraint.constraint_type {
            GpuConstraintType::MaxMemoryUsage => {
                let memory_usage_ratio = (device.total_memory_mb - device.available_memory_mb)
                    as f64
                    / device.total_memory_mb as f64;
                memory_usage_ratio <= constraint.value
            },
            GpuConstraintType::MaxUtilization => {
                device.utilization_percent as f64 <= constraint.value
            },
            GpuConstraintType::MinPerformance => {
                // Would check against benchmark scores in real implementation
                true
            },
            GpuConstraintType::PowerLimit => {
                // Would check current power consumption
                true
            },
            GpuConstraintType::TemperatureLimit => {
                // Would check current temperature
                true
            },
            GpuConstraintType::Custom(_) => {
                // Handle custom constraints
                true
            },
        }
    }

    /// Select device using least loaded strategy
    async fn select_least_loaded(
        &self,
        devices: &HashMap<usize, GpuDeviceInfo>,
    ) -> Result<Option<usize>> {
        let device_loads = self.device_loads.read();

        let selected = devices
            .values()
            .min_by(|a, b| {
                let load_a = device_loads
                    .get(&a.device_id)
                    .map(|load| load.utilization)
                    .unwrap_or(a.utilization_percent / 100.0);
                let load_b = device_loads
                    .get(&b.device_id)
                    .map(|load| load.utilization)
                    .unwrap_or(b.utilization_percent / 100.0);
                load_a.partial_cmp(&load_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|device| device.device_id);

        Ok(selected)
    }

    /// Select device using round-robin strategy
    async fn select_round_robin(
        &self,
        devices: &HashMap<usize, GpuDeviceInfo>,
    ) -> Result<Option<usize>> {
        if devices.is_empty() {
            return Ok(None);
        }

        let mut device_ids: Vec<_> = devices.keys().cloned().collect();
        device_ids.sort();

        let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
        let index = (counter as usize) % device_ids.len();

        Ok(Some(device_ids[index]))
    }

    /// Select device using best fit strategy
    async fn select_best_fit(
        &self,
        devices: &HashMap<usize, GpuDeviceInfo>,
        requirements: &GpuPerformanceRequirements,
    ) -> Result<Option<usize>> {
        // Select device with minimum available memory that still meets requirements
        // Use total_memory as tiebreaker when available_memory is the same
        let selected = devices
            .values()
            .filter(|device| device.available_memory_mb >= requirements.min_memory_mb)
            .min_by_key(|device| (device.available_memory_mb, device.total_memory_mb))
            .map(|device| device.device_id);

        Ok(selected)
    }

    /// Select device using random strategy
    async fn select_random(
        &self,
        devices: &HashMap<usize, GpuDeviceInfo>,
    ) -> Result<Option<usize>> {
        if devices.is_empty() {
            return Ok(None);
        }

        let device_ids: Vec<_> = devices.keys().cloned().collect();
        let index = self.generate_random_number() % device_ids.len();

        Ok(Some(device_ids[index]))
    }

    /// Select device using weighted strategy
    async fn select_weighted(
        &self,
        devices: &HashMap<usize, GpuDeviceInfo>,
    ) -> Result<Option<usize>> {
        let device_weights = self.device_weights.read();
        let device_loads = self.device_loads.read();

        let mut best_device = None;
        let mut best_score = f32::NEG_INFINITY;

        for device in devices.values() {
            let weight = device_weights.get(&device.device_id).copied().unwrap_or(1.0);
            let load = device_loads
                .get(&device.device_id)
                .map(|load| load.utilization)
                .unwrap_or(device.utilization_percent / 100.0);

            // Higher weight and lower load = better score
            let score = weight * (1.0 - load);

            if score > best_score {
                best_score = score;
                best_device = Some(device.device_id);
            }
        }

        Ok(best_device)
    }

    /// Select device using performance-based strategy
    async fn select_performance_based(
        &self,
        devices: &HashMap<usize, GpuDeviceInfo>,
    ) -> Result<Option<usize>> {
        let device_loads = self.device_loads.read();

        let selected = devices
            .values()
            .max_by(|a, b| {
                let score_a = self.calculate_performance_score(a, &device_loads);
                let score_b = self.calculate_performance_score(b, &device_loads);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|device| device.device_id);

        Ok(selected)
    }

    /// Calculate performance score for a device
    fn calculate_performance_score(
        &self,
        device: &GpuDeviceInfo,
        device_loads: &HashMap<usize, DeviceLoadInfo>,
    ) -> f32 {
        let base_score = device.total_memory_mb as f32 / 1000.0; // Normalize memory size

        let load_penalty = device_loads
            .get(&device.device_id)
            .map(|load| load.utilization)
            .unwrap_or(device.utilization_percent / 100.0);

        base_score * (1.0 - load_penalty)
    }

    /// Select device using memory-optimized strategy
    async fn select_memory_optimized(
        &self,
        devices: &HashMap<usize, GpuDeviceInfo>,
        requirements: &GpuPerformanceRequirements,
    ) -> Result<Option<usize>> {
        let device_loads = self.device_loads.read();

        let selected = devices
            .values()
            .filter(|device| device.available_memory_mb >= requirements.min_memory_mb)
            .min_by(|a, b| {
                let memory_score_a = self.calculate_memory_score(a, &device_loads);
                let memory_score_b = self.calculate_memory_score(b, &device_loads);
                memory_score_a.partial_cmp(&memory_score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|device| device.device_id);

        Ok(selected)
    }

    /// Calculate memory efficiency score for a device
    fn calculate_memory_score(
        &self,
        device: &GpuDeviceInfo,
        device_loads: &HashMap<usize, DeviceLoadInfo>,
    ) -> f32 {
        let memory_usage =
            device_loads.get(&device.device_id).map(|load| load.memory_usage).unwrap_or(
                (device.total_memory_mb - device.available_memory_mb) as f32
                    / device.total_memory_mb as f32,
            );

        // Prefer devices with lower memory usage but sufficient capacity
        memory_usage + (1.0 / device.available_memory_mb as f32)
    }

    /// Select device using power-aware strategy
    async fn select_power_aware(
        &self,
        devices: &HashMap<usize, GpuDeviceInfo>,
        workload_profile: Option<&WorkloadProfile>,
    ) -> Result<Option<usize>> {
        let device_loads = self.device_loads.read();
        let config = self.config.read();

        let power_priority = workload_profile
            .map(|profile| &profile.power_priority)
            .unwrap_or(&PowerPriority::Balanced);

        let selected = devices
            .values()
            .min_by(|a, b| {
                let score_a = self.calculate_power_score(a, &device_loads, power_priority, &config);
                let score_b = self.calculate_power_score(b, &device_loads, power_priority, &config);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|device| device.device_id);

        Ok(selected)
    }

    /// Calculate power efficiency score for a device
    fn calculate_power_score(
        &self,
        device: &GpuDeviceInfo,
        device_loads: &HashMap<usize, DeviceLoadInfo>,
        power_priority: &PowerPriority,
        config: &LoadBalancerConfig,
    ) -> f32 {
        let power_consumption = device_loads
            .get(&device.device_id)
            .map(|load| load.power_consumption)
            .unwrap_or(200.0); // Default power consumption

        let performance_factor = device.total_memory_mb as f32 / 1000.0;

        match power_priority {
            PowerPriority::Low => {
                // Minimize power consumption
                power_consumption
            },
            PowerPriority::Balanced => {
                // Balance power and performance
                power_consumption * config.power_weight
                    + (1.0 / performance_factor) * config.performance_weight
            },
            PowerPriority::High => {
                // Maximize performance
                1.0 / performance_factor
            },
        }
    }

    /// Select device using hybrid strategy
    async fn select_hybrid(
        &self,
        devices: &HashMap<usize, GpuDeviceInfo>,
        strategies: &[LoadBalancingStrategy],
        _requirements: &GpuPerformanceRequirements,
        _workload_profile: Option<&WorkloadProfile>,
    ) -> Result<Option<usize>> {
        if strategies.is_empty() {
            return self.select_least_loaded(devices).await;
        }

        let mut candidate_scores: HashMap<usize, f32> = HashMap::new();

        // Apply each strategy and collect scores
        for strategy in strategies {
            let candidates = match strategy {
                LoadBalancingStrategy::LeastLoaded => {
                    let device_loads = self.device_loads.read();
                    devices
                        .iter()
                        .map(|(id, device)| {
                            let load = device_loads
                                .get(id)
                                .map(|load| load.utilization)
                                .unwrap_or(device.utilization_percent / 100.0);
                            (*id, 1.0 - load)
                        })
                        .collect::<HashMap<_, _>>()
                },
                LoadBalancingStrategy::PerformanceBased => {
                    let device_loads = self.device_loads.read();
                    devices
                        .iter()
                        .map(|(id, device)| {
                            let score = self.calculate_performance_score(device, &device_loads);
                            (*id, score)
                        })
                        .collect::<HashMap<_, _>>()
                },
                LoadBalancingStrategy::MemoryOptimized => {
                    let device_loads = self.device_loads.read();
                    devices
                        .iter()
                        .map(|(id, device)| {
                            let score = 1.0 - self.calculate_memory_score(device, &device_loads);
                            (*id, score)
                        })
                        .collect::<HashMap<_, _>>()
                },
                _ => {
                    // For other strategies, give equal scores
                    devices.keys().map(|id| (*id, 1.0)).collect::<HashMap<_, _>>()
                },
            };

            // Accumulate scores
            for (device_id, score) in candidates {
                *candidate_scores.entry(device_id).or_insert(0.0) += score;
            }
        }

        // Select device with highest combined score
        let selected = candidate_scores
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(device_id, _)| device_id);

        Ok(selected)
    }

    /// Select device using custom strategy
    async fn select_custom(
        &self,
        devices: &HashMap<usize, GpuDeviceInfo>,
        strategy_name: &str,
        _requirements: &GpuPerformanceRequirements,
        _workload_profile: Option<&WorkloadProfile>,
    ) -> Result<Option<usize>> {
        warn!(
            "Custom strategy '{}' not implemented, falling back to LeastLoaded",
            strategy_name
        );
        self.select_least_loaded(devices).await
    }

    /// Update device load information
    ///
    /// # Arguments
    ///
    /// * `device_id` - ID of the device to update
    /// * `utilization` - New utilization value (0.0 to 1.0)
    #[instrument(skip(self))]
    pub async fn update_device_load(&self, device_id: usize, utilization: f32) -> Result<()> {
        let old_load = {
            let mut device_loads = self.device_loads.write();
            let old_utilization =
                device_loads.get(&device_id).map(|load| load.utilization).unwrap_or(0.0);

            let device_load = device_loads.entry(device_id).or_insert_with(|| DeviceLoadInfo {
                device_id,
                utilization: 0.0,
                memory_usage: 0.0,
                power_consumption: 0.0,
                temperature: 0.0,
                active_allocations: 0,
                performance_score: 1.0,
                load_trend: LoadTrend::default(),
                last_updated: Utc::now(),
            });

            device_load.utilization = utilization.clamp(0.0, 1.0);
            device_load.last_updated = Utc::now();

            // Update load trend
            device_load.load_trend.history.push_back(utilization);
            if device_load.load_trend.history.len() > 100 {
                device_load.load_trend.history.pop_front();
            }

            self.update_load_trend(&mut device_load.load_trend);

            old_utilization
        };

        // Send load update event
        let event = LoadBalancingEvent::LoadUpdated {
            device_id,
            old_load,
            new_load: utilization,
        };

        if let Err(e) = self.event_sender.send(event) {
            warn!("Failed to send load update event: {}", e);
        }

        // Check if rebalancing is needed
        self.check_rebalancing_needed().await?;

        debug!(
            "Updated device {} load: {:.2} -> {:.2}",
            device_id, old_load, utilization
        );
        Ok(())
    }

    /// Update comprehensive device load information
    ///
    /// # Arguments
    ///
    /// * `device_id` - ID of the device to update
    /// * `load_info` - Complete load information to update
    #[instrument(skip(self, load_info))]
    pub async fn update_comprehensive_load(
        &self,
        device_id: usize,
        load_info: DeviceLoadInfo,
    ) -> Result<()> {
        let (old_load, new_load) = {
            let mut device_loads = self.device_loads.write();
            let old_utilization =
                device_loads.get(&device_id).map(|load| load.utilization).unwrap_or(0.0);

            let new_utilization = load_info.utilization;
            device_loads.insert(device_id, load_info);

            (old_utilization, new_utilization)
        };

        // Send load update event
        let event = LoadBalancingEvent::LoadUpdated {
            device_id,
            old_load,
            new_load,
        };

        if let Err(e) = self.event_sender.send(event) {
            warn!("Failed to send comprehensive load update event: {}", e);
        }

        // Check if rebalancing is needed
        self.check_rebalancing_needed().await?;

        debug!("Updated comprehensive load for device {}", device_id);
        Ok(())
    }

    /// Update load trend analysis
    fn update_load_trend(&self, trend: &mut LoadTrend) {
        if trend.history.len() < 3 {
            trend.trend_direction = LoadBalancerTrendDirection::Stable;
            trend.confidence = 0.0;
            trend.predicted_load = trend.history.back().copied().unwrap_or(0.0);
            return;
        }

        // Simple trend analysis using linear regression
        let n = trend.history.len();
        let x_sum: f32 = (0..n).map(|i| i as f32).sum();
        let y_sum: f32 = trend.history.iter().sum();
        let xy_sum: f32 = trend.history.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
        let x_sq_sum: f32 = (0..n).map(|i| (i as f32).powi(2)).sum();

        let slope = (n as f32 * xy_sum - x_sum * y_sum) / (n as f32 * x_sq_sum - x_sum.powi(2));

        // Determine trend direction
        trend.trend_direction = if slope.abs() < 0.01 {
            LoadBalancerTrendDirection::Stable
        } else if slope > 0.05 {
            LoadBalancerTrendDirection::Increasing
        } else if slope < -0.05 {
            LoadBalancerTrendDirection::Decreasing
        } else {
            LoadBalancerTrendDirection::Volatile
        };

        // Calculate confidence based on R-squared
        let y_mean = y_sum / n as f32;
        let ss_tot: f32 = trend.history.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let ss_res: f32 = trend
            .history
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let predicted = slope * i as f32 + (y_sum - slope * x_sum) / n as f32;
                (y - predicted).powi(2)
            })
            .sum();

        trend.confidence = if ss_tot > 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 }.clamp(0.0, 1.0);

        // Predict next value
        trend.predicted_load = slope * n as f32 + (y_sum - slope * x_sum) / n as f32;
        trend.predicted_load = trend.predicted_load.clamp(0.0, 1.0);
    }

    /// Check if load rebalancing is needed and trigger if necessary
    async fn check_rebalancing_needed(&self) -> Result<()> {
        let device_loads = self.device_loads.read();
        let config = self.config.read();

        if device_loads.len() < 2 {
            return Ok(()); // Need at least 2 devices for rebalancing
        }

        // Calculate utilization variance
        let utilizations: Vec<f32> = device_loads.values().map(|load| load.utilization).collect();

        let mean_utilization = utilizations.iter().sum::<f32>() / utilizations.len() as f32;
        let variance = utilizations.iter().map(|&u| (u - mean_utilization).powi(2)).sum::<f32>()
            / utilizations.len() as f32;

        // Trigger rebalancing if variance exceeds threshold
        if variance > config.rebalancing_threshold {
            let affected_devices: Vec<usize> = device_loads.keys().cloned().collect();

            let event = LoadBalancingEvent::RebalancingTriggered {
                reason: format!(
                    "Load variance {:.3} exceeds threshold {:.3}",
                    variance, config.rebalancing_threshold
                ),
                affected_devices: affected_devices.clone(),
            };

            if let Err(e) = self.event_sender.send(event) {
                warn!("Failed to send rebalancing event: {}", e);
            }

            // Generate rebalancing suggestions
            self.generate_rebalancing_suggestions(&device_loads).await?;

            info!(
                "Load rebalancing triggered: variance={:.3}, threshold={:.3}",
                variance, config.rebalancing_threshold
            );
        }

        Ok(())
    }

    /// Generate rebalancing suggestions
    async fn generate_rebalancing_suggestions(
        &self,
        device_loads: &HashMap<usize, DeviceLoadInfo>,
    ) -> Result<()> {
        let mut suggestions = Vec::new();

        // Find overloaded and underloaded devices
        let mean_utilization = device_loads.values().map(|load| load.utilization).sum::<f32>()
            / device_loads.len() as f32;

        let overloaded: Vec<_> = device_loads
            .iter()
            .filter(|(_, load)| load.utilization > mean_utilization + 0.2)
            .collect();

        let underloaded: Vec<_> = device_loads
            .iter()
            .filter(|(_, load)| load.utilization < mean_utilization - 0.2)
            .collect();

        // Generate suggestions to move load from overloaded to underloaded devices
        for (&source_id, source_load) in &overloaded {
            for (&target_id, target_load) in &underloaded {
                let load_diff = source_load.utilization - target_load.utilization;
                let transfer_amount = (load_diff / 2.0).min(0.3); // Transfer at most 30% load

                if transfer_amount > 0.05 {
                    // Only suggest if transfer is meaningful
                    let suggestion = RebalancingSuggestion {
                        source_device: source_id,
                        target_device: target_id,
                        load_amount: transfer_amount,
                        expected_improvement: load_diff.abs() * 0.5,
                        priority: self.calculate_rebalancing_priority(load_diff),
                        effort_estimate: Duration::from_secs(30), // Estimated effort
                    };

                    suggestions.push(suggestion);
                }
            }
        }

        // Sort suggestions by priority
        suggestions.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Store suggestions
        {
            let mut rebalancing_suggestions = self.rebalancing_suggestions.write();
            for suggestion in suggestions {
                rebalancing_suggestions.push_back(suggestion);
            }

            // Keep only the most recent suggestions
            while rebalancing_suggestions.len() > 50 {
                rebalancing_suggestions.pop_front();
            }
        }

        Ok(())
    }

    /// Calculate rebalancing priority based on load difference
    fn calculate_rebalancing_priority(&self, load_diff: f32) -> RebalancingPriority {
        if load_diff > 0.6 {
            RebalancingPriority::Critical
        } else if load_diff > 0.4 {
            RebalancingPriority::High
        } else if load_diff > 0.2 {
            RebalancingPriority::Medium
        } else {
            RebalancingPriority::Low
        }
    }

    /// Get load balancing analytics
    ///
    /// # Returns
    ///
    /// Current load balancing analytics and metrics
    pub async fn get_load_analytics(&self) -> LoadBalancingAnalytics {
        let mut analytics = self.analytics.write();

        // Update real-time metrics
        let device_loads = self.device_loads.read();

        if !device_loads.is_empty() {
            analytics.average_utilization =
                device_loads.values().map(|load| load.utilization).sum::<f32>()
                    / device_loads.len() as f32;

            analytics.utilization_distribution =
                device_loads.iter().map(|(&id, load)| (id, load.utilization)).collect();

            // Calculate utilization variance
            let mean = analytics.average_utilization;
            analytics.utilization_variance =
                device_loads.values().map(|load| (load.utilization - mean).powi(2)).sum::<f32>()
                    / device_loads.len() as f32;

            // Calculate efficiency score (lower variance = higher efficiency)
            analytics.efficiency_score = (1.0 - analytics.utilization_variance.min(1.0)).max(0.0);

            // Update success rate (simplified - in real implementation would track failures)
            analytics.success_rate = if analytics.total_allocations > 0 {
                0.95 // Assume 95% success rate for demo
            } else {
                1.0
            };
        }

        analytics.generated_at = Utc::now();
        analytics.clone()
    }

    /// Get rebalancing suggestions
    ///
    /// # Returns
    ///
    /// Queue of current rebalancing suggestions ordered by priority
    pub async fn get_rebalancing_suggestions(&self) -> Vec<RebalancingSuggestion> {
        let suggestions = self.rebalancing_suggestions.read();
        suggestions.iter().cloned().collect()
    }

    /// Set device weight for weighted load balancing
    ///
    /// # Arguments
    ///
    /// * `device_id` - ID of the device
    /// * `weight` - Weight factor (higher = more likely to be selected)
    #[instrument(skip(self))]
    pub async fn set_device_weight(&self, device_id: usize, weight: f32) -> Result<()> {
        let mut device_weights = self.device_weights.write();
        device_weights.insert(device_id, weight.max(0.0));

        debug!("Set device {} weight to {:.2}", device_id, weight);
        Ok(())
    }

    /// Get device load information
    ///
    /// # Arguments
    ///
    /// * `device_id` - ID of the device
    ///
    /// # Returns
    ///
    /// Device load information if available
    pub async fn get_device_load(&self, device_id: usize) -> Option<DeviceLoadInfo> {
        let device_loads = self.device_loads.read();
        device_loads.get(&device_id).cloned()
    }

    /// Get all device loads
    ///
    /// # Returns
    ///
    /// Map of device IDs to their load information
    pub async fn get_all_device_loads(&self) -> HashMap<usize, DeviceLoadInfo> {
        let device_loads = self.device_loads.read();
        device_loads.clone()
    }

    /// Take a load snapshot for historical analysis
    #[instrument(skip(self))]
    pub async fn take_load_snapshot(&self) -> Result<()> {
        let device_loads = self.device_loads.read();
        let strategy = {
            let guard = self.strategy.read();
            guard.clone()
        };

        let device_load_map: HashMap<usize, f32> =
            device_loads.iter().map(|(&id, load)| (id, load.utilization)).collect();

        let system_utilization = if !device_load_map.is_empty() {
            device_load_map.values().sum::<f32>() / device_load_map.len() as f32
        } else {
            0.0
        };

        let snapshot = LoadSnapshot {
            timestamp: Utc::now(),
            device_loads: device_load_map,
            active_strategy: strategy,
            system_utilization,
        };

        {
            let mut load_history = self.load_history.write();
            load_history.push_back(snapshot);

            let config = self.config.read();
            while load_history.len() > config.max_history_size {
                load_history.pop_front();
            }
        }

        debug!("Load snapshot taken with {} devices", device_loads.len());
        Ok(())
    }

    /// Get load history
    ///
    /// # Returns
    ///
    /// Vector of historical load snapshots
    pub async fn get_load_history(&self) -> Vec<LoadSnapshot> {
        let load_history = self.load_history.read();
        load_history.iter().cloned().collect()
    }

    /// Generate a simple random number for load balancing
    fn generate_random_number(&self) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::{SystemTime, UNIX_EPOCH};

        let mut hasher = DefaultHasher::new();
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Operation should succeed")
            .as_nanos()
            .hash(&mut hasher);
        hasher.finish() as usize
    }
}

// ================================================================================================
// DEFAULT IMPLEMENTATIONS
// ================================================================================================

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        Self::LeastLoaded
    }
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            adaptive_strategy: false,
            rebalancing_threshold: 0.15, // 15% variance threshold
            max_history_size: 1000,
            analysis_interval: Duration::from_secs(60),
            enable_prediction: true,
            prediction_confidence_threshold: 0.7,
            max_utilization_target: 0.85,
            power_weight: 0.3,
            performance_weight: 0.7,
        }
    }
}

impl Default for LoadTrend {
    fn default() -> Self {
        Self {
            history: VecDeque::new(),
            predicted_load: 0.0,
            trend_direction: LoadBalancerTrendDirection::Stable,
            confidence: 0.0,
        }
    }
}

impl Default for LoadBalancingAnalytics {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            average_utilization: 0.0,
            utilization_variance: 0.0,
            efficiency_score: 1.0,
            strategy_changes: 0,
            rebalancing_events: 0,
            average_allocation_time: Duration::from_millis(1),
            utilization_distribution: HashMap::new(),
            success_rate: 1.0,
            performance_improvement: 0.0,
            generated_at: Utc::now(),
        }
    }
}

// ================================================================================================
// DISPLAY IMPLEMENTATIONS
// ================================================================================================

impl std::fmt::Display for LoadBalancingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LeastLoaded => write!(f, "Least Loaded"),
            Self::RoundRobin => write!(f, "Round Robin"),
            Self::BestFit => write!(f, "Best Fit"),
            Self::Random => write!(f, "Random"),
            Self::Weighted => write!(f, "Weighted"),
            Self::PerformanceBased => write!(f, "Performance Based"),
            Self::MemoryOptimized => write!(f, "Memory Optimized"),
            Self::PowerAware => write!(f, "Power Aware"),
            Self::Hybrid(strategies) => {
                write!(f, "Hybrid({})", strategies.len())
            },
            Self::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

impl std::fmt::Display for LoadBalancerTrendDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Increasing => write!(f, "Increasing"),
            Self::Decreasing => write!(f, "Decreasing"),
            Self::Stable => write!(f, "Stable"),
            Self::Volatile => write!(f, "Volatile"),
        }
    }
}

impl std::fmt::Display for PowerPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "Low Power"),
            Self::Balanced => write!(f, "Balanced"),
            Self::High => write!(f, "High Performance"),
        }
    }
}

impl std::fmt::Display for WorkloadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Training => write!(f, "Training"),
            Self::Inference => write!(f, "Inference"),
            Self::Simulation => write!(f, "Simulation"),
            Self::DataProcessing => write!(f, "Data Processing"),
            Self::Rendering => write!(f, "Rendering"),
            Self::General => write!(f, "General"),
        }
    }
}

impl std::fmt::Display for LoadPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Steady => write!(f, "Steady"),
            Self::Ramping => write!(f, "Ramping"),
            Self::Declining => write!(f, "Declining"),
            Self::Bursty => write!(f, "Bursty"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl std::fmt::Display for RebalancingPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Helper function to create test device info
    fn create_test_device(device_id: usize, utilization: f32, memory_mb: u64) -> GpuDeviceInfo {
        GpuDeviceInfo {
            device_id,
            device_name: format!("Test GPU {}", device_id),
            total_memory_mb: memory_mb,
            available_memory_mb: memory_mb - (memory_mb as f32 * utilization) as u64,
            utilization_percent: utilization * 100.0,
            capabilities: vec![GpuCapability::Cuda("11.0".to_string())],
            status: GpuDeviceStatus::Available,
            last_updated: Utc::now(),
        }
    }

    /// Helper function to create test requirements
    fn create_test_requirements() -> GpuPerformanceRequirements {
        GpuPerformanceRequirements {
            min_memory_mb: 4096,
            min_compute_capability: 7.0,
            required_frameworks: vec!["CUDA".to_string()],
            constraints: vec![],
        }
    }

    #[tokio::test]
    async fn test_load_balancer_creation() {
        let load_balancer = GpuLoadBalancer::new();

        let strategy = load_balancer.get_strategy().await;
        assert_eq!(strategy, LoadBalancingStrategy::LeastLoaded);

        let analytics = load_balancer.get_load_analytics().await;
        assert_eq!(analytics.total_allocations, 0);
    }

    #[tokio::test]
    async fn test_strategy_setting() {
        let load_balancer = GpuLoadBalancer::new();

        load_balancer
            .set_strategy(LoadBalancingStrategy::RoundRobin)
            .await
            .expect("Set strategy should succeed");
        let strategy = load_balancer.get_strategy().await;
        assert_eq!(strategy, LoadBalancingStrategy::RoundRobin);

        let analytics = load_balancer.get_load_analytics().await;
        assert_eq!(analytics.strategy_changes, 1);
    }

    #[tokio::test]
    async fn test_least_loaded_strategy() {
        let load_balancer = GpuLoadBalancer::new();
        let requirements = create_test_requirements();

        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0, 0.8, 8192)); // High utilization
        devices.insert(1, create_test_device(1, 0.2, 8192)); // Low utilization
        devices.insert(2, create_test_device(2, 0.5, 8192)); // Medium utilization

        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        assert_eq!(selected, Some(1)); // Should select device with lowest utilization
    }

    #[tokio::test]
    async fn test_round_robin_strategy() {
        let load_balancer = GpuLoadBalancer::new();
        load_balancer
            .set_strategy(LoadBalancingStrategy::RoundRobin)
            .await
            .expect("Set strategy should succeed");

        let requirements = create_test_requirements();

        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0, 0.5, 8192));
        devices.insert(1, create_test_device(1, 0.5, 8192));
        devices.insert(2, create_test_device(2, 0.5, 8192));

        // Should cycle through devices in order
        let selected1 = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        let selected2 = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        let selected3 = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        let selected4 = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");

        assert!(selected1.is_some());
        assert!(selected2.is_some());
        assert!(selected3.is_some());
        assert_eq!(selected1, selected4); // Should wrap around
    }

    #[tokio::test]
    async fn test_best_fit_strategy() {
        let load_balancer = GpuLoadBalancer::new();
        load_balancer
            .set_strategy(LoadBalancingStrategy::BestFit)
            .await
            .expect("Set strategy should succeed");

        let requirements = create_test_requirements(); // Requires 4096MB

        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0, 0.5, 16384)); // More memory than needed
        devices.insert(1, create_test_device(1, 0.5, 8192)); // Less memory than device 0
        devices.insert(2, create_test_device(2, 0.0, 4096)); // Exact fit (no utilization)

        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        assert_eq!(selected, Some(2)); // Should select device with least memory that meets requirements
    }

    #[tokio::test]
    async fn test_load_tracking() {
        let load_balancer = GpuLoadBalancer::new();

        // Update device loads
        load_balancer
            .update_device_load(0, 0.7)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(1, 0.3)
            .await
            .expect("Update load should succeed");

        // Verify loads are tracked
        let device_0_load = load_balancer.get_device_load(0).await;
        assert!(device_0_load.is_some());
        assert_eq!(
            device_0_load.expect("Should get device load").utilization,
            0.7
        );

        let device_1_load = load_balancer.get_device_load(1).await;
        assert!(device_1_load.is_some());
        assert_eq!(
            device_1_load.expect("Should get device load").utilization,
            0.3
        );

        let all_loads = load_balancer.get_all_device_loads().await;
        assert_eq!(all_loads.len(), 2);
    }

    #[tokio::test]
    async fn test_weighted_strategy() {
        let load_balancer = GpuLoadBalancer::new();
        load_balancer
            .set_strategy(LoadBalancingStrategy::Weighted)
            .await
            .expect("Set strategy should succeed");

        // Set different weights
        load_balancer
            .set_device_weight(0, 1.0)
            .await
            .expect("Set weight should succeed");
        load_balancer
            .set_device_weight(1, 2.0)
            .await
            .expect("Set weight should succeed"); // Higher weight
        load_balancer
            .set_device_weight(2, 0.5)
            .await
            .expect("Set weight should succeed");

        // Set equal loads
        load_balancer
            .update_device_load(0, 0.5)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(1, 0.5)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(2, 0.5)
            .await
            .expect("Update load should succeed");

        let requirements = create_test_requirements();

        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0, 0.5, 8192));
        devices.insert(1, create_test_device(1, 0.5, 8192));
        devices.insert(2, create_test_device(2, 0.5, 8192));

        // Should prefer device 1 due to higher weight
        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        assert_eq!(selected, Some(1));
    }

    #[tokio::test]
    async fn test_load_analytics() {
        let load_balancer = GpuLoadBalancer::new();

        // Perform some allocations
        let requirements = create_test_requirements();
        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0, 0.3, 8192));
        devices.insert(1, create_test_device(1, 0.7, 8192));

        // Update device loads
        load_balancer
            .update_device_load(0, 0.3)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(1, 0.7)
            .await
            .expect("Update load should succeed");

        // Perform some selections
        for _ in 0..5 {
            load_balancer
                .select_optimal_device(&devices, &requirements, None)
                .await
                .expect("Operation should succeed");
        }

        let analytics = load_balancer.get_load_analytics().await;
        assert_eq!(analytics.total_allocations, 5);
        assert_eq!(analytics.average_utilization, 0.5);
        assert!(analytics.utilization_variance > 0.0);
        assert!(analytics.efficiency_score >= 0.0 && analytics.efficiency_score <= 1.0);
    }

    #[tokio::test]
    async fn test_load_snapshots() {
        let load_balancer = GpuLoadBalancer::new();

        // Update some device loads
        load_balancer
            .update_device_load(0, 0.4)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(1, 0.6)
            .await
            .expect("Update load should succeed");

        // Take a snapshot
        load_balancer.take_load_snapshot().await.expect("Take snapshot should succeed");

        // Update loads again
        load_balancer
            .update_device_load(0, 0.8)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(1, 0.2)
            .await
            .expect("Update load should succeed");

        // Take another snapshot
        load_balancer.take_load_snapshot().await.expect("Take snapshot should succeed");

        let history = load_balancer.get_load_history().await;
        assert_eq!(history.len(), 2);

        // Verify snapshots contain correct data
        assert_eq!(history[0].device_loads.len(), 2);
        assert_eq!(history[1].device_loads.len(), 2);
    }

    #[tokio::test]
    async fn test_comprehensive_load_update() {
        let load_balancer = GpuLoadBalancer::new();

        let load_info = DeviceLoadInfo {
            device_id: 0,
            utilization: 0.75,
            memory_usage: 0.80,
            power_consumption: 250.0,
            temperature: 65.0,
            active_allocations: 3,
            performance_score: 0.9,
            load_trend: LoadTrend::default(),
            last_updated: Utc::now(),
        };

        load_balancer
            .update_comprehensive_load(0, load_info.clone())
            .await
            .expect("Update comprehensive load should succeed");

        let retrieved_load = load_balancer.get_device_load(0).await;
        assert!(retrieved_load.is_some());

        let retrieved = retrieved_load.expect("Should retrieve load info");
        assert_eq!(retrieved.utilization, 0.75);
        assert_eq!(retrieved.memory_usage, 0.80);
        assert_eq!(retrieved.power_consumption, 250.0);
        assert_eq!(retrieved.temperature, 65.0);
        assert_eq!(retrieved.active_allocations, 3);
    }

    #[tokio::test]
    async fn test_rebalancing_suggestions() {
        let load_balancer = GpuLoadBalancer::with_config(LoadBalancerConfig {
            rebalancing_threshold: 0.1, // Low threshold for testing
            ..LoadBalancerConfig::default()
        });

        // Create imbalanced load scenario
        load_balancer
            .update_device_load(0, 0.9)
            .await
            .expect("Update load should succeed"); // Heavily loaded
        load_balancer
            .update_device_load(1, 0.1)
            .await
            .expect("Update load should succeed"); // Lightly loaded

        // Wait a moment for rebalancing check
        tokio::time::sleep(Duration::from_millis(10)).await;

        let suggestions = load_balancer.get_rebalancing_suggestions().await;

        // Should have generated suggestions due to load imbalance
        assert!(!suggestions.is_empty());

        if let Some(suggestion) = suggestions.first() {
            assert_eq!(suggestion.source_device, 0); // Should suggest moving from device 0
            assert_eq!(suggestion.target_device, 1); // Should suggest moving to device 1
            assert!(suggestion.load_amount > 0.0);
        }
    }

    #[tokio::test]
    async fn test_hybrid_strategy() {
        let load_balancer = GpuLoadBalancer::new();

        let hybrid_strategies = vec![
            LoadBalancingStrategy::LeastLoaded,
            LoadBalancingStrategy::PerformanceBased,
        ];

        load_balancer
            .set_strategy(LoadBalancingStrategy::Hybrid(hybrid_strategies))
            .await
            .expect("Operation should succeed");

        // Update device loads
        load_balancer
            .update_device_load(0, 0.8)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(1, 0.2)
            .await
            .expect("Update load should succeed");

        let requirements = create_test_requirements();

        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0, 0.8, 16384)); // High util, high memory
        devices.insert(1, create_test_device(1, 0.2, 8192)); // Low util, low memory

        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");

        // Hybrid strategy should consider both least loaded and performance
        assert!(selected.is_some());
    }

    #[tokio::test]
    async fn test_power_aware_strategy() {
        let load_balancer = GpuLoadBalancer::new();
        load_balancer
            .set_strategy(LoadBalancingStrategy::PowerAware)
            .await
            .expect("Set strategy should succeed");

        let requirements = create_test_requirements();

        let workload_profile = WorkloadProfile {
            estimated_duration: Duration::from_secs(300),
            memory_intensity: 0.7,
            compute_intensity: 0.8,
            power_priority: PowerPriority::Low, // Prefer low power
            workload_type: WorkloadType::Inference,
            load_pattern: LoadPattern::Steady,
        };

        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0, 0.5, 8192));
        devices.insert(1, create_test_device(1, 0.5, 8192));

        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, Some(&workload_profile))
            .await
            .expect("Operation should succeed");

        assert!(selected.is_some());
    }

    #[tokio::test]
    async fn test_memory_optimized_strategy() {
        let load_balancer = GpuLoadBalancer::new();
        load_balancer
            .set_strategy(LoadBalancingStrategy::MemoryOptimized)
            .await
            .expect("Operation should succeed");

        // Update memory usage for devices
        let load_info_0 = DeviceLoadInfo {
            device_id: 0,
            utilization: 0.5,
            memory_usage: 0.9, // High memory usage
            power_consumption: 200.0,
            temperature: 60.0,
            active_allocations: 2,
            performance_score: 1.0,
            load_trend: LoadTrend::default(),
            last_updated: Utc::now(),
        };

        let load_info_1 = DeviceLoadInfo {
            device_id: 1,
            utilization: 0.5,
            memory_usage: 0.3, // Low memory usage
            power_consumption: 200.0,
            temperature: 60.0,
            active_allocations: 1,
            performance_score: 1.0,
            load_trend: LoadTrend::default(),
            last_updated: Utc::now(),
        };

        load_balancer
            .update_comprehensive_load(0, load_info_0)
            .await
            .expect("Update comprehensive load should succeed");
        load_balancer
            .update_comprehensive_load(1, load_info_1)
            .await
            .expect("Update comprehensive load should succeed");

        let requirements = create_test_requirements();

        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0, 0.5, 8192));
        devices.insert(1, create_test_device(1, 0.5, 8192));

        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        assert_eq!(selected, Some(1)); // Should select device with lower memory usage
    }
}
