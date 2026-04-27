//! Tests for real-time metrics collector types

use super::types::*;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// Simple LCG for deterministic pseudo-random values
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() % 10000) as f32 / 10000.0
    }
}

#[test]
fn test_circular_buffer_new() {
    let buf: CircularBuffer<i32> = CircularBuffer::new(10);
    assert_eq!(buf.get_recent(5).len(), 0);
}

#[test]
fn test_circular_buffer_push_and_get() {
    let mut buf: CircularBuffer<i32> = CircularBuffer::new(5);
    buf.push(1);
    buf.push(2);
    buf.push(3);
    let recent = buf.get_recent(3);
    assert_eq!(recent.len(), 3);
    assert_eq!(recent[0], 3);
    assert_eq!(recent[1], 2);
    assert_eq!(recent[2], 1);
}

#[test]
fn test_circular_buffer_wrapping() {
    let mut buf: CircularBuffer<i32> = CircularBuffer::new(3);
    buf.push(1);
    buf.push(2);
    buf.push(3);
    buf.push(4);
    let recent = buf.get_recent(3);
    assert_eq!(recent.len(), 3);
    assert_eq!(recent[0], 4);
    assert_eq!(recent[1], 3);
    assert_eq!(recent[2], 2);
}

#[test]
fn test_circular_buffer_utilization_empty() {
    let buf: CircularBuffer<i32> = CircularBuffer::new(10);
    assert!((buf.utilization() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_circular_buffer_utilization_half() {
    let mut buf: CircularBuffer<i32> = CircularBuffer::new(10);
    for i in 0..5 {
        buf.push(i);
    }
    assert!((buf.utilization() - 50.0).abs() < f32::EPSILON);
}

#[test]
fn test_circular_buffer_utilization_full() {
    let mut buf: CircularBuffer<i32> = CircularBuffer::new(5);
    for i in 0..5 {
        buf.push(i);
    }
    assert!((buf.utilization() - 100.0).abs() < f32::EPSILON);
}

#[test]
fn test_circular_buffer_utilization_overflow() {
    let mut buf: CircularBuffer<i32> = CircularBuffer::new(5);
    for i in 0..10 {
        buf.push(i);
    }
    assert!((buf.utilization() - 100.0).abs() < f32::EPSILON);
}

#[test]
fn test_circular_buffer_get_recent_more_than_available() {
    let mut buf: CircularBuffer<i32> = CircularBuffer::new(10);
    buf.push(1);
    buf.push(2);
    let recent = buf.get_recent(5);
    assert_eq!(recent.len(), 2);
}

#[test]
fn test_circular_buffer_statistics() {
    let mut buf: CircularBuffer<i32> = CircularBuffer::new(10);
    buf.push(1);
    buf.push(2);
    buf.push(3);
    let stats = buf.get_statistics();
    assert_eq!(stats.items_written.load(Ordering::Relaxed), 3);
    assert_eq!(stats.current_size.load(Ordering::Relaxed), 3);
    assert_eq!(stats.max_capacity, 10);
}

#[test]
fn test_publisher_type_http() {
    let pt = PublisherType::Http {
        endpoint: "http://localhost:8080".to_string(),
    };
    if let PublisherType::Http { endpoint } = &pt {
        assert!(endpoint.starts_with("http"));
    } else {
        panic!("expected Http variant");
    }
}

#[test]
fn test_publisher_type_file() {
    let pt = PublisherType::File {
        path: "/tmp/metrics.log".to_string(),
    };
    if let PublisherType::File { path } = &pt {
        assert!(path.ends_with(".log"));
    } else {
        panic!("expected File variant");
    }
}

#[test]
fn test_publisher_type_message_queue() {
    let pt = PublisherType::MessageQueue {
        queue_name: "metrics_queue".to_string(),
    };
    if let PublisherType::MessageQueue { queue_name } = &pt {
        assert_eq!(queue_name, "metrics_queue");
    } else {
        panic!("expected MessageQueue variant");
    }
}

#[test]
fn test_impact_severity_variants() {
    let severities = [
        ImpactSeverity::Low,
        ImpactSeverity::Low,
        ImpactSeverity::Moderate,
        ImpactSeverity::High,
    ];
    assert_eq!(severities.len(), 4);
}

#[test]
fn test_publisher_status_variants() {
    let statuses = [
        PublisherStatus::Active,
        PublisherStatus::Inactive,
        PublisherStatus::Error,
    ];
    assert_eq!(statuses.len(), 3);
}

#[test]
fn test_impact_trend_variants() {
    let trends = [
        ImpactTrend::Increasing,
        ImpactTrend::Stable,
        ImpactTrend::Decreasing,
    ];
    assert_eq!(trends.len(), 3);
}

#[test]
fn test_delivery_guarantee_variants() {
    let guarantees = [
        DeliveryGuarantee::BestEffort,
        DeliveryGuarantee::AtLeastOnce,
        DeliveryGuarantee::ExactlyOnce,
    ];
    assert_eq!(guarantees.len(), 3);
}

#[test]
fn test_delivery_config_creation() {
    let config = DeliveryConfig {
        guarantee: DeliveryGuarantee::AtLeastOnce,
        retry_attempts: 3,
        retry_delay: Duration::from_millis(100),
        batch_size: 50,
        compression: true,
        timeout: Duration::from_secs(5),
        rate_limiting_enabled: false,
        max_throughput: 1000.0,
    };
    assert_eq!(config.retry_attempts, 3);
    assert_eq!(config.batch_size, 50);
    assert!(config.compression);
}

#[test]
fn test_pid_sample_rate_algorithm_new() {
    let _algo = PidSampleRateAlgorithm::new();
}

#[test]
fn test_default_collection_error_handler_new() {
    let _handler = DefaultCollectionErrorHandler::new();
}

#[test]
fn test_default_publish_error_handler_new() {
    let _handler = DefaultPublishErrorHandler::new();
}

#[test]
fn test_publish_rate_limiter_new() {
    let _limiter = PublishRateLimiter::new();
}

#[test]
fn test_impact_trend_analyzer_new() {
    let _analyzer = ImpactTrendAnalyzer::new();
}

#[test]
fn test_impact_recommendation_engine_new() {
    let _engine = ImpactRecommendationEngine::new();
}

#[test]
fn test_circular_buffer_large_capacity_lcg() {
    let mut lcg = Lcg::new(42);
    let mut buf: CircularBuffer<u64> = CircularBuffer::new(100);
    for _ in 0..200 {
        buf.push(lcg.next_u64());
    }
    let recent = buf.get_recent(100);
    assert_eq!(recent.len(), 100);
    let mut seen = std::collections::HashSet::new();
    for val in &recent {
        seen.insert(*val);
    }
    assert_eq!(seen.len(), 100);
}

#[test]
fn test_circular_buffer_single_element() {
    let mut buf: CircularBuffer<i32> = CircularBuffer::new(1);
    buf.push(42);
    let recent = buf.get_recent(1);
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0], 42);
    buf.push(99);
    let recent2 = buf.get_recent(1);
    assert_eq!(recent2[0], 99);
}

#[test]
fn test_circular_buffer_get_zero() {
    let mut buf: CircularBuffer<i32> = CircularBuffer::new(10);
    buf.push(1);
    let recent = buf.get_recent(0);
    assert!(recent.is_empty());
}

#[test]
fn test_retry_strategy_variants() {
    let strategies = [
        RetryStrategy::Fixed,
        RetryStrategy::ExponentialBackoff,
        RetryStrategy::Linear,
    ];
    assert_eq!(strategies.len(), 3);
}
