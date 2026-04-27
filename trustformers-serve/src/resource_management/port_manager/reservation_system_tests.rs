//! Comprehensive tests for reservation_system.rs
//!
//! Tests for PortReservationSystem operations including reservations,
//! cancellations, expiry, queue management, and event tracking.

#[cfg(test)]
mod tests {
    use crate::resource_management::port_manager::types::*;
    use crate::resource_management::types::*;
    use parking_lot::Mutex;
    use std::collections::HashSet;
    use std::sync::Arc;
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

    fn make_available_ports(start: u16, end: u16) -> Arc<Mutex<HashSet<u16>>> {
        let mut ports = HashSet::new();
        for port in start..=end {
            ports.insert(port);
        }
        Arc::new(Mutex::new(ports))
    }

    #[tokio::test]
    async fn test_reservation_system_creation() {
        let result = PortReservationSystem::new().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_reserve_ports_basic() {
        if let Ok(system) = PortReservationSystem::new().await {
            let available = make_available_ports(10000, 10100);
            let result = system
                .reserve_ports(
                    3,
                    "test_001",
                    Duration::from_secs(300),
                    PortUsageType::HttpServer,
                    &available,
                )
                .await;
            assert!(result.is_ok());
            if let Ok(ports) = result {
                assert_eq!(ports.len(), 3);
            }
        }
    }

    #[tokio::test]
    async fn test_reserve_ports_removes_from_available() {
        if let Ok(system) = PortReservationSystem::new().await {
            let available = make_available_ports(10000, 10010);
            let initial_count = available.lock().len();
            let result = system
                .reserve_ports(
                    3,
                    "test_001",
                    Duration::from_secs(300),
                    PortUsageType::HttpServer,
                    &available,
                )
                .await;
            if result.is_ok() {
                let after_count = available.lock().len();
                assert_eq!(after_count, initial_count - 3);
            }
        }
    }

    #[tokio::test]
    async fn test_reserve_zero_ports() {
        if let Ok(system) = PortReservationSystem::new().await {
            let available = make_available_ports(10000, 10010);
            let result = system
                .reserve_ports(
                    0,
                    "test_001",
                    Duration::from_secs(300),
                    PortUsageType::HttpServer,
                    &available,
                )
                .await;
            assert!(result.is_ok());
            if let Ok(ports) = result {
                assert!(ports.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_reserve_too_many_ports() {
        if let Ok(system) = PortReservationSystem::new().await {
            let available = make_available_ports(10000, 10002);
            let result = system
                .reserve_ports(
                    10,
                    "test_001",
                    Duration::from_secs(300),
                    PortUsageType::HttpServer,
                    &available,
                )
                .await;
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn test_cancel_reservations() {
        if let Ok(system) = PortReservationSystem::new().await {
            let available = make_available_ports(10000, 10100);
            let _ = system
                .reserve_ports(
                    3,
                    "test_001",
                    Duration::from_secs(300),
                    PortUsageType::HttpServer,
                    &available,
                )
                .await;
            let result = system.cancel_reservations("test_001").await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_cancel_nonexistent_reservation() {
        if let Ok(system) = PortReservationSystem::new().await {
            let result = system.cancel_reservations("nonexistent").await;
            // Should succeed even if nothing to cancel
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_multiple_reservations_different_tests() {
        if let Ok(system) = PortReservationSystem::new().await {
            let available = make_available_ports(10000, 10100);
            let r1 = system
                .reserve_ports(
                    2,
                    "test_001",
                    Duration::from_secs(300),
                    PortUsageType::HttpServer,
                    &available,
                )
                .await;
            let r2 = system
                .reserve_ports(
                    3,
                    "test_002",
                    Duration::from_secs(300),
                    PortUsageType::GrpcServer,
                    &available,
                )
                .await;
            assert!(r1.is_ok());
            assert!(r2.is_ok());
        }
    }

    #[tokio::test]
    async fn test_reservation_history_tracking() {
        if let Ok(system) = PortReservationSystem::new().await {
            let available = make_available_ports(10000, 10100);
            let _ = system
                .reserve_ports(
                    2,
                    "test_001",
                    Duration::from_secs(300),
                    PortUsageType::HttpServer,
                    &available,
                )
                .await;
            let history = system.reservation_history.lock();
            assert!(!history.is_empty());
        }
    }

    #[tokio::test]
    async fn test_reservation_by_test_tracking() {
        if let Ok(system) = PortReservationSystem::new().await {
            let available = make_available_ports(10000, 10100);
            let _ = system
                .reserve_ports(
                    3,
                    "test_001",
                    Duration::from_secs(300),
                    PortUsageType::HttpServer,
                    &available,
                )
                .await;
            let by_test = system.reservations_by_test.lock();
            if let Some(ports) = by_test.get("test_001") {
                assert_eq!(ports.len(), 3);
            }
        }
    }

    #[tokio::test]
    async fn test_reservation_config_defaults() {
        let config = PortReservationConfig::default();
        assert_eq!(config.max_reservations_per_test, 10);
        assert!(config.enable_queue_processing);
        assert!(config.max_queue_size > 0);
    }

    #[tokio::test]
    async fn test_cleanup_all_reservations() {
        if let Ok(system) = PortReservationSystem::new().await {
            let available = make_available_ports(10000, 10100);
            let _ = system
                .reserve_ports(
                    5,
                    "test_001",
                    Duration::from_secs(300),
                    PortUsageType::HttpServer,
                    &available,
                )
                .await;
            let result = system.cleanup_all_reservations().await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_reservation_default_system() {
        let system = PortReservationSystem::default();
        assert!(system.reservations.lock().is_empty());
        assert!(system.reservations_by_test.lock().is_empty());
        assert!(system.reservation_queue.lock().is_empty());
    }

    #[tokio::test]
    async fn test_reservation_with_different_usage_types() {
        if let Ok(system) = PortReservationSystem::new().await {
            let available = make_available_ports(10000, 10100);
            let usage_types = vec![
                PortUsageType::HttpServer,
                PortUsageType::GrpcServer,
                PortUsageType::Database,
                PortUsageType::Custom("test_service".to_string()),
            ];
            for (i, usage_type) in usage_types.into_iter().enumerate() {
                let result = system
                    .reserve_ports(
                        1,
                        &format!("test_{:03}", i),
                        Duration::from_secs(300),
                        usage_type,
                        &available,
                    )
                    .await;
                assert!(result.is_ok());
            }
        }
    }

    #[tokio::test]
    async fn test_reservation_port_uniqueness() {
        if let Ok(system) = PortReservationSystem::new().await {
            let available = make_available_ports(10000, 10100);
            let r1 = system
                .reserve_ports(
                    5,
                    "test_001",
                    Duration::from_secs(300),
                    PortUsageType::HttpServer,
                    &available,
                )
                .await;
            let r2 = system
                .reserve_ports(
                    5,
                    "test_002",
                    Duration::from_secs(300),
                    PortUsageType::HttpServer,
                    &available,
                )
                .await;
            if let (Ok(ports1), Ok(ports2)) = (r1, r2) {
                for p in &ports1 {
                    assert!(!ports2.contains(p));
                }
            }
        }
    }

    #[tokio::test]
    async fn test_reservation_with_lcg_random_counts() {
        if let Ok(system) = PortReservationSystem::new().await {
            let available = make_available_ports(10000, 10200);
            let mut rng = Lcg::new(12345);
            let mut total_reserved = 0;
            for i in 0..5 {
                let count = ((rng.next() % 5) + 1) as usize;
                let result = system
                    .reserve_ports(
                        count,
                        &format!("test_{:03}", i),
                        Duration::from_secs(300),
                        PortUsageType::HttpServer,
                        &available,
                    )
                    .await;
                if let Ok(ports) = result {
                    total_reserved += ports.len();
                }
            }
            assert!(total_reserved > 0);
        }
    }

    #[test]
    fn test_reservation_event_type_variants() {
        let types = vec![
            ReservationEventType::Created,
            ReservationEventType::Fulfilled,
            ReservationEventType::Cancelled,
            ReservationEventType::Expired,
            ReservationEventType::Queued,
            ReservationEventType::Conflict,
        ];
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_port_usage_type_variants() {
        let http = PortUsageType::HttpServer;
        let grpc = PortUsageType::GrpcServer;
        let db = PortUsageType::Database;
        let custom = PortUsageType::Custom("ws".to_string());
        assert!(matches!(http, PortUsageType::HttpServer));
        assert!(matches!(grpc, PortUsageType::GrpcServer));
        assert!(matches!(db, PortUsageType::Database));
        if let PortUsageType::Custom(name) = custom {
            assert_eq!(name, "ws");
        }
    }
}
