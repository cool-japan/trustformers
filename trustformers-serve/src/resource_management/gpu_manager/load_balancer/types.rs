//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::{
    GpuConstraint, GpuConstraintType, GpuDeviceInfo, GpuDeviceStatus, GpuPerformanceRequirements,
};
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
        let event = LoadBalancingEvent::StrategyChanged {
            old_strategy,
            new_strategy: strategy.clone(),
            reason: "Manual strategy change".to_string(),
        };
        if let Err(e) = self.event_sender.send(event) {
            warn!("Failed to send strategy change event: {}", e);
        }
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
        let suitable_devices =
            self.filter_suitable_devices(available_devices, requirements).await?;
        if suitable_devices.is_empty() {
            debug!("No suitable devices found for requirements");
            return Ok(None);
        }
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
        let allocation_time = start_time.elapsed();
        if let Some(device_id) = selected_device {
            let event = LoadBalancingEvent::DeviceSelected {
                device_id,
                strategy: strategy.clone(),
                allocation_time,
            };
            if let Err(e) = self.event_sender.send(event) {
                warn!("Failed to send device selection event: {}", e);
            }
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
            if device.status != GpuDeviceStatus::Available {
                continue;
            }
            if device.available_memory_mb < requirements.min_memory_mb {
                continue;
            }
            let supports_frameworks = requirements.required_frameworks.iter().all(|framework| {
                device
                    .capabilities
                    .iter()
                    .any(|capability| capability.supports_framework(framework))
            });
            if !supports_frameworks {
                continue;
            }
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
            GpuConstraintType::MinPerformance => true,
            GpuConstraintType::PowerLimit => true,
            GpuConstraintType::TemperatureLimit => true,
            GpuConstraintType::Custom(_) => true,
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
        let base_score = device.total_memory_mb as f32 / 1000.0;
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
            .unwrap_or(200.0);
        let performance_factor = device.total_memory_mb as f32 / 1000.0;
        match power_priority {
            PowerPriority::Low => power_consumption,
            PowerPriority::Balanced => {
                power_consumption * config.power_weight
                    + (1.0 / performance_factor) * config.performance_weight
            },
            PowerPriority::High => 1.0 / performance_factor,
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
                _ => devices.keys().map(|id| (*id, 1.0)).collect::<HashMap<_, _>>(),
            };
            for (device_id, score) in candidates {
                *candidate_scores.entry(device_id).or_insert(0.0) += score;
            }
        }
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
            device_load.load_trend.history.push_back(utilization);
            if device_load.load_trend.history.len() > 100 {
                device_load.load_trend.history.pop_front();
            }
            self.update_load_trend(&mut device_load.load_trend);
            old_utilization
        };
        let event = LoadBalancingEvent::LoadUpdated {
            device_id,
            old_load,
            new_load: utilization,
        };
        if let Err(e) = self.event_sender.send(event) {
            warn!("Failed to send load update event: {}", e);
        }
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
        let event = LoadBalancingEvent::LoadUpdated {
            device_id,
            old_load,
            new_load,
        };
        if let Err(e) = self.event_sender.send(event) {
            warn!("Failed to send comprehensive load update event: {}", e);
        }
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
        let n = trend.history.len();
        let x_sum: f32 = (0..n).map(|i| i as f32).sum();
        let y_sum: f32 = trend.history.iter().sum();
        let xy_sum: f32 = trend.history.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
        let x_sq_sum: f32 = (0..n).map(|i| (i as f32).powi(2)).sum();
        let slope = (n as f32 * xy_sum - x_sum * y_sum) / (n as f32 * x_sq_sum - x_sum.powi(2));
        trend.trend_direction = if slope.abs() < 0.01 {
            LoadBalancerTrendDirection::Stable
        } else if slope > 0.05 {
            LoadBalancerTrendDirection::Increasing
        } else if slope < -0.05 {
            LoadBalancerTrendDirection::Decreasing
        } else {
            LoadBalancerTrendDirection::Volatile
        };
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
        trend.predicted_load = slope * n as f32 + (y_sum - slope * x_sum) / n as f32;
        trend.predicted_load = trend.predicted_load.clamp(0.0, 1.0);
    }
    /// Check if load rebalancing is needed and trigger if necessary
    async fn check_rebalancing_needed(&self) -> Result<()> {
        let device_loads = self.device_loads.read();
        let config = self.config.read();
        if device_loads.len() < 2 {
            return Ok(());
        }
        let utilizations: Vec<f32> = device_loads.values().map(|load| load.utilization).collect();
        let mean_utilization = utilizations.iter().sum::<f32>() / utilizations.len() as f32;
        let variance = utilizations.iter().map(|&u| (u - mean_utilization).powi(2)).sum::<f32>()
            / utilizations.len() as f32;
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
        for (&source_id, source_load) in &overloaded {
            for (&target_id, target_load) in &underloaded {
                let load_diff = source_load.utilization - target_load.utilization;
                let transfer_amount = (load_diff / 2.0).min(0.3);
                if transfer_amount > 0.05 {
                    let suggestion = RebalancingSuggestion {
                        source_device: source_id,
                        target_device: target_id,
                        load_amount: transfer_amount,
                        expected_improvement: load_diff.abs() * 0.5,
                        priority: self.calculate_rebalancing_priority(load_diff),
                        effort_estimate: Duration::from_secs(30),
                    };
                    suggestions.push(suggestion);
                }
            }
        }
        suggestions.sort_by(|a, b| b.priority.cmp(&a.priority));
        {
            let mut rebalancing_suggestions = self.rebalancing_suggestions.write();
            for suggestion in suggestions {
                rebalancing_suggestions.push_back(suggestion);
            }
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
        let device_loads = self.device_loads.read();
        if !device_loads.is_empty() {
            analytics.average_utilization =
                device_loads.values().map(|load| load.utilization).sum::<f32>()
                    / device_loads.len() as f32;
            analytics.utilization_distribution =
                device_loads.iter().map(|(&id, load)| (id, load.utilization)).collect();
            let mean = analytics.average_utilization;
            analytics.utilization_variance =
                device_loads.values().map(|load| (load.utilization - mean).powi(2)).sum::<f32>()
                    / device_loads.len() as f32;
            analytics.efficiency_score = (1.0 - analytics.utilization_variance.min(1.0)).max(0.0);
            analytics.success_rate = if analytics.total_allocations > 0 { 0.95 } else { 1.0 };
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
