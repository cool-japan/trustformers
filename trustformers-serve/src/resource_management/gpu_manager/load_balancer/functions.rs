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
        let selected = load_balancer
            .select_optimal_device(&devices, &requirements, None)
            .await
            .expect("Operation should succeed");
        assert_eq!(selected, Some(1));
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
