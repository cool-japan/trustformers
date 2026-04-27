//! Comprehensive tests for manager.rs (NetworkPortManager)
//!
//! Tests for port allocation, deallocation, reservation, configuration,
//! health monitoring, and performance metrics.

#[cfg(test)]
mod tests {
    use super::NetworkPortManager;
    use crate::resource_management::port_manager::types::*;
    use crate::resource_management::types::*;
    use std::collections::HashMap;
    use std::time::Duration;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        #[allow(dead_code)]
        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    fn test_config() -> PortPoolConfig {
        PortPoolConfig {
            port_range: (10000, 10100),
            max_allocation: 50,
            allocation_timeout_secs: 300,
            enable_reservation: true,
            reserved_ranges: vec![],
        }
    }

    #[tokio::test]
    async fn test_manager_creation() {
        let config = test_config();
        let result = NetworkPortManager::new(config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_manager_invalid_range() {
        let config = PortPoolConfig {
            port_range: (9000, 8000), // Invalid: start > end
            max_allocation: 50,
            allocation_timeout_secs: 300,
            enable_reservation: true,
            reserved_ranges: vec![],
        };
        let result = NetworkPortManager::new(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_manager_zero_max_allocation() {
        let config = PortPoolConfig {
            port_range: (10000, 10100),
            max_allocation: 0,
            allocation_timeout_secs: 300,
            enable_reservation: true,
            reserved_ranges: vec![],
        };
        let result = NetworkPortManager::new(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_allocate_ports_basic() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let result = manager.allocate_ports(3, "test_001").await;
            assert!(result.is_ok());
            if let Ok(ports) = result {
                assert_eq!(ports.len(), 3);
                // All ports should be unique
                let mut unique = ports.clone();
                unique.sort();
                unique.dedup();
                assert_eq!(unique.len(), 3);
            }
        }
    }

    #[tokio::test]
    async fn test_allocate_zero_ports() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let result = manager.allocate_ports(0, "test_001").await;
            assert!(result.is_ok());
            if let Ok(ports) = result {
                assert!(ports.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_allocate_too_many_ports() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let result = manager.allocate_ports(200, "test_001").await;
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn test_deallocate_port() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            if let Ok(ports) = manager.allocate_ports(1, "test_001").await {
                let port = ports[0];
                let result = manager.deallocate_port(port).await;
                assert!(result.is_ok());
            }
        }
    }

    #[tokio::test]
    async fn test_deallocate_unallocated_port() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let result = manager.deallocate_port(10050).await;
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn test_deallocate_ports_for_test() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let _ = manager.allocate_ports(5, "test_001").await;
            let _ = manager.allocate_ports(3, "test_002").await;
            let result = manager.deallocate_ports_for_test("test_001").await;
            assert!(result.is_ok());
            // test_002 ports should still be allocated
            let count = manager.get_allocated_port_count().await;
            assert_eq!(count, 3);
        }
    }

    #[tokio::test]
    async fn test_check_availability() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let result = manager.check_availability(5).await;
            assert!(result.is_ok());
            if let Ok(available) = result {
                assert!(available);
            }
        }
    }

    #[tokio::test]
    async fn test_check_availability_insufficient() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let result = manager.check_availability(200).await;
            assert!(result.is_ok());
            if let Ok(available) = result {
                assert!(!available);
            }
        }
    }

    #[tokio::test]
    async fn test_get_statistics() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let _ = manager.allocate_ports(3, "test_001").await;
            let result = manager.get_statistics().await;
            assert!(result.is_ok());
            if let Ok(stats) = result {
                assert!(stats.total_allocated >= 3);
            }
        }
    }

    #[tokio::test]
    async fn test_get_port_allocations() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let _ = manager.allocate_ports(2, "test_001").await;
            let allocations = manager.get_port_allocations().await;
            assert_eq!(allocations.len(), 2);
            for (_, alloc) in &allocations {
                assert_eq!(alloc.test_id, "test_001");
            }
        }
    }

    #[tokio::test]
    async fn test_available_port_count() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let initial = manager.get_available_port_count().await;
            let _ = manager.allocate_ports(5, "test_001").await;
            let after = manager.get_available_port_count().await;
            assert_eq!(after, initial - 5);
        }
    }

    #[tokio::test]
    async fn test_is_port_available() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            if let Ok(ports) = manager.allocate_ports(1, "test_001").await {
                let port = ports[0];
                assert!(!manager.is_port_available(port).await);
                assert!(manager.is_port_allocated(port).await);
            }
        }
    }

    #[tokio::test]
    async fn test_get_port_allocation_details() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            if let Ok(ports) = manager.allocate_ports(1, "test_001").await {
                let port = ports[0];
                let alloc = manager.get_port_allocation(port).await;
                assert!(alloc.is_some());
                if let Some(a) = alloc {
                    assert_eq!(a.test_id, "test_001");
                }
            }
        }
    }

    #[tokio::test]
    async fn test_get_utilization() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let initial_util = manager.get_utilization().await;
            assert!((initial_util - 0.0).abs() < f32::EPSILON);
            let _ = manager.allocate_ports(10, "test_001").await;
            let util = manager.get_utilization().await;
            assert!(util > 0.0);
        }
    }

    #[tokio::test]
    async fn test_generate_allocation_report() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let _ = manager.allocate_ports(5, "test_001").await;
            let report = manager.generate_allocation_report().await;
            assert!(report.contains("Port Allocation Report"));
            assert!(report.contains("Available ports"));
            assert!(report.contains("Allocated ports"));
        }
    }

    #[tokio::test]
    async fn test_update_config() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let new_config = PortPoolConfig {
                port_range: (11000, 11100),
                max_allocation: 80,
                allocation_timeout_secs: 600,
                enable_reservation: false,
                reserved_ranges: vec![],
            };
            let result = manager.update_config(new_config).await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_health_status_retrieval() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let health = manager.get_health_status().await;
            // Health should be retrievable
            let _ = format!("{:?}", health.overall_status);
        }
    }

    #[tokio::test]
    async fn test_performance_metrics_retrieval() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let _ = manager.allocate_ports(3, "test_001").await;
            let metrics = manager.get_performance_metrics().await;
            assert!(metrics.total_allocations >= 1);
        }
    }

    #[tokio::test]
    async fn test_shutdown() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let _ = manager.allocate_ports(5, "test_001").await;
            let result = manager.shutdown().await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_concurrent_allocations() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let _ = manager.allocate_ports(5, "test_001").await;
            let _ = manager.allocate_ports(3, "test_002").await;
            let _ = manager.allocate_ports(2, "test_003").await;
            let total_allocated = manager.get_allocated_port_count().await;
            assert_eq!(total_allocated, 10);
        }
    }

    #[tokio::test]
    async fn test_allocate_deallocate_cycle() {
        if let Ok(manager) = NetworkPortManager::new(test_config()).await {
            let initial_count = manager.get_available_port_count().await;
            let _ = manager.allocate_ports(10, "test_001").await;
            let _ = manager.deallocate_ports_for_test("test_001").await;
            let final_count = manager.get_available_port_count().await;
            assert_eq!(initial_count, final_count);
        }
    }
}
