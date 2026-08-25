//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::resource_management::gpu_manager::types::{
        GpuCapability, GpuDeviceInfo, GpuDeviceStatus, GpuPerformanceRequirements,
    };
    use chrono::Utc;
    use std::collections::HashMap;
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
        devices.insert(0, create_test_device(0, 0.8, 8192));
        devices.insert(1, create_test_device(1, 0.2, 8192));
        devices.insert(2, create_test_device(2, 0.5, 8192));
        // `select_least_loaded` ranks devices by the live `device_loads`
        // reading, never by `GpuDeviceInfo::utilization_percent` (a
        // discovery-time constant that nothing updates -- see the doc
        // comment on `select_least_loaded`), so a realistic test must
        // populate the same telemetry a real load balancer would.
        load_balancer
            .update_device_load(0, 0.8)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(1, 0.2)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(2, 0.5)
            .await
            .expect("Update load should succeed");
        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        assert_eq!(selected, Some(1));
    }
    /// Regression: a device with no live load reading must never be
    /// preferred over one that is genuinely measured, even at a very high
    /// load. `test_hybrid_strategy_unmonitored_device_never_wins` locks this
    /// invariant for the `Hybrid` combinator; this exercises the dedicated
    /// `LeastLoaded` strategy (the default) directly, which had no
    /// regression guard of its own before this test.
    #[tokio::test]
    async fn test_least_loaded_never_prefers_an_unmonitored_device() {
        let load_balancer = GpuLoadBalancer::new();
        // Device 1 is measured at 99% utilization -- about as bad as a real
        // device gets. Device 0 never receives an `update_device_load` call
        // at all.
        load_balancer
            .update_device_load(1, 0.99)
            .await
            .expect("Update load should succeed");
        let requirements = create_test_requirements();
        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0, 0.0, 8192));
        devices.insert(1, create_test_device(1, 0.0, 8192));
        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        assert_eq!(
            selected,
            Some(1),
            "an unmonitored device must not outscore a device measured at 99% utilization"
        );
    }
    /// Regression: when *no* candidate device carries a live load reading at
    /// all, `select_least_loaded` must not resolve the choice via `HashMap`
    /// iteration order (undisclosed, and unstable across runs of the same
    /// process). It falls back to the lowest device id -- a documented,
    /// deterministic choice -- which this test locks in across repeated
    /// calls and an insertion order that does not match id order.
    #[tokio::test]
    async fn test_least_loaded_all_unmonitored_falls_back_deterministically() {
        let load_balancer = GpuLoadBalancer::new();
        let requirements = create_test_requirements();
        let mut devices = HashMap::new();
        // Utilization is 0.0 for all three (and so plays no part in this
        // test): `create_test_device`'s `available_memory_mb` is derived
        // from it, and a high enough figure would make
        // `filter_suitable_devices` reject the device before it ever reaches
        // `select_least_loaded`, for a reason unrelated to what this test
        // exercises.
        devices.insert(5, create_test_device(5, 0.0, 8192));
        devices.insert(2, create_test_device(2, 0.0, 8192));
        devices.insert(9, create_test_device(9, 0.0, 8192));
        for _ in 0..5 {
            let selected = load_balancer
                .select_optimal_device(&devices, &requirements, None)
                .await
                .expect("Operation should succeed");
            assert_eq!(
                selected,
                Some(2),
                "with no load data at all, the fallback must be the lowest device id, every time"
            );
        }
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
        assert_eq!(selected1, selected4);
    }
    #[tokio::test]
    async fn test_best_fit_strategy() {
        let load_balancer = GpuLoadBalancer::new();
        load_balancer
            .set_strategy(LoadBalancingStrategy::BestFit)
            .await
            .expect("Set strategy should succeed");
        let requirements = create_test_requirements();
        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0, 0.5, 16384));
        devices.insert(1, create_test_device(1, 0.5, 8192));
        devices.insert(2, create_test_device(2, 0.0, 4096));
        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        assert_eq!(selected, Some(2));
    }
    #[tokio::test]
    async fn test_load_tracking() {
        let load_balancer = GpuLoadBalancer::new();
        load_balancer
            .update_device_load(0, 0.7)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(1, 0.3)
            .await
            .expect("Update load should succeed");
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
        load_balancer
            .set_device_weight(0, 1.0)
            .await
            .expect("Set weight should succeed");
        load_balancer
            .set_device_weight(1, 2.0)
            .await
            .expect("Set weight should succeed");
        load_balancer
            .set_device_weight(2, 0.5)
            .await
            .expect("Set weight should succeed");
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
        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        assert_eq!(selected, Some(1));
    }
    #[tokio::test]
    async fn test_load_analytics() {
        let load_balancer = GpuLoadBalancer::new();
        let requirements = create_test_requirements();
        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0, 0.3, 8192));
        devices.insert(1, create_test_device(1, 0.7, 8192));
        load_balancer
            .update_device_load(0, 0.3)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(1, 0.7)
            .await
            .expect("Update load should succeed");
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
        load_balancer
            .update_device_load(0, 0.4)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(1, 0.6)
            .await
            .expect("Update load should succeed");
        load_balancer.take_load_snapshot().await.expect("Take snapshot should succeed");
        load_balancer
            .update_device_load(0, 0.8)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(1, 0.2)
            .await
            .expect("Update load should succeed");
        load_balancer.take_load_snapshot().await.expect("Take snapshot should succeed");
        let history = load_balancer.get_load_history().await;
        assert_eq!(history.len(), 2);
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
            rebalancing_threshold: 0.1,
            ..LoadBalancerConfig::default()
        });
        load_balancer
            .update_device_load(0, 0.9)
            .await
            .expect("Update load should succeed");
        load_balancer
            .update_device_load(1, 0.1)
            .await
            .expect("Update load should succeed");
        tokio::time::sleep(Duration::from_millis(10)).await;
        let suggestions = load_balancer.get_rebalancing_suggestions().await;
        assert!(!suggestions.is_empty());
        if let Some(suggestion) = suggestions.first() {
            assert_eq!(suggestion.source_device, 0);
            assert_eq!(suggestion.target_device, 1);
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
        devices.insert(0, create_test_device(0, 0.8, 16384));
        devices.insert(1, create_test_device(1, 0.2, 8192));
        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        assert!(selected.is_some());
    }
    /// Regression: `select_hybrid`'s `LeastLoaded` branch used to fall back to
    /// `GpuDeviceInfo::utilization_percent` (a discovery-time constant 0.0) for
    /// a device with no live load reading, so that device scored `1.0 - 0.0 =
    /// 1.0` and won every hybrid selection over any genuinely measured device.
    /// Device 0 here is never given a load reading at all (no
    /// `update_device_load` call); device 1 is measured at 99% utilization,
    /// about as bad as a real device gets. A correct hybrid must still prefer
    /// the measured-but-heavily-loaded device over the unmonitored one.
    #[tokio::test]
    async fn test_hybrid_strategy_unmonitored_device_never_wins() {
        let load_balancer = GpuLoadBalancer::new();
        load_balancer
            .set_strategy(LoadBalancingStrategy::Hybrid(vec![
                LoadBalancingStrategy::LeastLoaded,
            ]))
            .await
            .expect("Set strategy should succeed");
        load_balancer
            .update_device_load(1, 0.99)
            .await
            .expect("Update load should succeed");
        let requirements = create_test_requirements();
        let mut devices = HashMap::new();
        // Both devices' *own records* carry utilization_percent 0.0 (the
        // discovery default) and plenty of available memory, so both pass
        // suitability filtering regardless of load; the 99% figure for
        // device 1 comes only from the live `device_loads` reading set
        // above, which is exactly the value the old fallback ignored in
        // favor of this record's own (here, identically 0.0) utilization_percent.
        devices.insert(0, create_test_device(0, 0.0, 8192));
        devices.insert(1, create_test_device(1, 0.0, 8192));
        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        assert_eq!(
            selected,
            Some(1),
            "an unmonitored device must not outscore a device measured at 99% utilization"
        );
    }
    /// Regression: `select_hybrid`'s `LeastLoaded` component used to score
    /// *every* candidate `f32::NEG_INFINITY` whenever none of them carried a
    /// load reading. Because `-infinity + finite == -infinity`, that
    /// silenced any real signal contributed by the other strategies mixed
    /// into the hybrid and left the pick to `HashMap` iteration order. Here
    /// `LeastLoaded` has nothing to say about either device (neither ever
    /// receives `update_device_load`), while `MemoryOptimized` has a
    /// genuine, real signal: device 1 has far more free memory. The real
    /// signal must decide it.
    #[tokio::test]
    async fn test_hybrid_strategy_falls_back_to_other_components_when_unmonitored() {
        let load_balancer = GpuLoadBalancer::new();
        load_balancer
            .set_strategy(LoadBalancingStrategy::Hybrid(vec![
                LoadBalancingStrategy::LeastLoaded,
                LoadBalancingStrategy::MemoryOptimized,
            ]))
            .await
            .expect("Set strategy should succeed");
        let requirements = create_test_requirements();
        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0, 0.0, 8192)); // little free memory
        devices.insert(1, create_test_device(1, 0.0, 65536)); // much more free memory
        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        assert_eq!(
            selected,
            Some(1),
            "a real MemoryOptimized signal must not be drowned out by an unmonitored \
             LeastLoaded component"
        );
    }
    /// Regression: when a hybrid's *only* component strategy is
    /// `LeastLoaded` and no device is monitored, every candidate ties at the
    /// neutral score. The tie must resolve deterministically (the lowest
    /// device id), not via `HashMap` iteration order.
    #[tokio::test]
    async fn test_hybrid_all_unmonitored_least_loaded_only_is_deterministic() {
        let load_balancer = GpuLoadBalancer::new();
        load_balancer
            .set_strategy(LoadBalancingStrategy::Hybrid(vec![
                LoadBalancingStrategy::LeastLoaded,
            ]))
            .await
            .expect("Set strategy should succeed");
        let requirements = create_test_requirements();
        let mut devices = HashMap::new();
        devices.insert(7, create_test_device(7, 0.0, 8192));
        devices.insert(3, create_test_device(3, 0.0, 8192));
        devices.insert(4, create_test_device(4, 0.0, 8192));
        for _ in 0..5 {
            let selected = load_balancer
                .select_optimal_device(&devices, &requirements, None)
                .await
                .expect("Operation should succeed");
            assert_eq!(
                selected,
                Some(3),
                "an all-tied hybrid must resolve to the lowest device id, every time"
            );
        }
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
            power_priority: PowerPriority::Low,
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
        let load_info_0 = DeviceLoadInfo {
            device_id: 0,
            utilization: 0.5,
            memory_usage: 0.9,
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
            memory_usage: 0.3,
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
        assert_eq!(selected, Some(1));
    }
}
