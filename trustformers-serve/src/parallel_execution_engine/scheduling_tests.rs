//! Tests for the engine's real priority queue and scheduler.
//!
//! These lock in the 0.2.1 fix: before it, `PriorityQueue` discarded every
//! pushed item and `TestScheduler` reported an empty queue no matter what had
//! been scheduled.

use super::scheduling::PriorityQueue;
use super::types::{QueueConfig, ScheduledTest};
use crate::test_parallelization::{
    IsolationRequirements, ParallelizationHints, TestParallelizationMetadata, TestResourceUsage,
};
use crate::test_timeout_optimization::{TestCategory, TestComplexityHints, TestExecutionContext};
use chrono::Utc;
use std::collections::HashMap;
use std::time::Duration;

fn metadata(test_id: &str, priority: f32) -> TestParallelizationMetadata {
    TestParallelizationMetadata {
        base_context: TestExecutionContext {
            test_name: test_id.to_string(),
            category: TestCategory::Unit,
            expected_duration: None,
            complexity_hints: TestComplexityHints::default(),
            environment: "test".to_string(),
            timeout_override: Some(Duration::from_secs(5)),
        },
        dependencies: Vec::new(),
        resource_usage: TestResourceUsage {
            test_id: test_id.to_string(),
            cpu_cores: 0.5,
            memory_mb: 8,
            gpu_devices: Vec::new(),
            network_ports: Vec::new(),
            temp_directories: Vec::new(),
            database_connections: 0,
            duration: Duration::from_millis(1),
            priority,
        },
        isolation_requirements: IsolationRequirements::default(),
        tags: Vec::new(),
        priority,
        parallelization_hints: ParallelizationHints::default(),
    }
}

pub(super) fn scheduled(test_id: &str, priority: f32) -> ScheduledTest {
    ScheduledTest {
        metadata: metadata(test_id, priority),
        priority,
        scheduled_at: Utc::now(),
        estimated_start: None,
        resource_requirements: crate::parallel_execution_engine::ResourceRequirement {
            resource_type: "mixed".to_string(),
            min_amount: 1.0,
            cpu_cores: 0.5,
            memory_mb: 8,
            gpu_devices: Vec::new(),
            network_ports: 0,
            temp_directories: 0,
            database_connections: 0,
            custom_resources: HashMap::new(),
        },
        constraints: Vec::new(),
        retry_count: 0,
        scheduling_metadata: HashMap::new(),
    }
}

#[test]
fn priority_queue_stores_what_is_pushed() {
    let mut queue: PriorityQueue<u8> = PriorityQueue::new();
    assert!(queue.is_empty());
    assert!(queue.push(7, 1.0).is_ok());
    assert_eq!(queue.len(), 1);
    assert!(!queue.is_empty());
    assert_eq!(queue.pop(), Some(7));
    assert!(queue.is_empty());
    assert_eq!(queue.pop(), None);
}

#[test]
fn priority_queue_pops_highest_priority_first() {
    let mut queue: PriorityQueue<&'static str> = PriorityQueue::new();
    assert!(queue.push("low", 0.1).is_ok());
    assert!(queue.push("high", 0.9).is_ok());
    assert!(queue.push("mid", 0.5).is_ok());
    assert_eq!(queue.pop(), Some("high"));
    assert_eq!(queue.pop(), Some("mid"));
    assert_eq!(queue.pop(), Some("low"));
}

#[test]
fn priority_queue_breaks_ties_in_push_order() {
    let mut queue: PriorityQueue<u8> = PriorityQueue::new();
    for value in [1u8, 2, 3] {
        assert!(queue.push(value, 0.5).is_ok());
    }
    assert_eq!(queue.pop(), Some(1));
    assert_eq!(queue.pop(), Some(2));
    assert_eq!(queue.pop(), Some(3));
}

#[test]
fn priority_queue_honours_disabled_priority_as_fifo() {
    let mut queue: PriorityQueue<&'static str> = PriorityQueue::with_config(QueueConfig {
        max_size: 0,
        priority_enabled: false,
    });
    assert!(queue.push("first", 0.1).is_ok());
    assert!(queue.push("second", 0.9).is_ok());
    assert_eq!(queue.pop(), Some("first"));
    assert_eq!(queue.pop(), Some("second"));
}

#[test]
fn priority_queue_rejects_pushes_beyond_max_size() {
    let mut queue: PriorityQueue<u8> = PriorityQueue::with_config(QueueConfig {
        max_size: 2,
        priority_enabled: true,
    });
    assert!(queue.push(1, 0.5).is_ok());
    assert!(queue.push(2, 0.5).is_ok());
    // The third push must fail loudly rather than silently discard the item.
    assert!(queue.push(3, 0.5).is_err());
    assert_eq!(queue.len(), 2);
}

#[test]
fn priority_queue_statistics_are_measured() {
    let mut queue: PriorityQueue<u8> = PriorityQueue::new();
    assert!(queue.push(1, 0.5).is_ok());
    assert!(queue.push(2, 0.5).is_ok());
    assert_eq!(queue.statistics().total_enqueued, 2);
    assert_eq!(queue.statistics().peak_size, 2);
    assert_eq!(queue.statistics().current_size, 2);
    assert_eq!(queue.pop(), Some(1));
    assert_eq!(queue.statistics().total_dequeued, 1);
    assert_eq!(queue.statistics().current_size, 1);
    // Wait time is measured from the real enqueue instant, so it is a genuine
    // duration rather than a stored constant.
    assert!(queue.statistics().average_wait_time <= Duration::from_secs(60));
    assert!(queue.config().priority_enabled);
}

#[tokio::test]
async fn scheduler_queues_and_returns_tests_in_priority_order() {
    let scheduler = crate::parallel_execution_engine::TestScheduler::new(
        crate::test_parallelization::SchedulingConfig::default(),
    )
    .await
    .unwrap_or_else(|e| panic!("scheduler construction failed: {e}"));

    assert!(scheduler.is_queue_empty().await);
    assert!(scheduler.schedule_test(scheduled("low", 0.2)).await.is_ok());
    assert!(scheduler.schedule_test(scheduled("high", 0.8)).await.is_ok());
    // Before 0.2.1 this assertion failed: the queue always reported empty.
    assert!(!scheduler.is_queue_empty().await);

    let first = scheduler
        .get_next_test()
        .await
        .unwrap_or_else(|e| panic!("get_next_test failed: {e}"))
        .unwrap_or_else(|| panic!("queue lost the scheduled tests"));
    assert_eq!(first.metadata.resource_usage.test_id, "high");

    // Requeue it and confirm it comes back.
    assert!(scheduler.requeue_test(first).await.is_ok());
    let names: Vec<String> = {
        let mut collected = Vec::new();
        while let Ok(Some(test)) = scheduler.get_next_test().await {
            collected.push(test.metadata.resource_usage.test_id.clone());
        }
        collected
    };
    assert_eq!(names, vec!["high".to_string(), "low".to_string()]);
    assert!(scheduler.is_queue_empty().await);
}

#[tokio::test]
async fn scheduler_records_real_history_and_decisions() {
    let scheduler = crate::parallel_execution_engine::TestScheduler::new(
        crate::test_parallelization::SchedulingConfig::default(),
    )
    .await
    .unwrap_or_else(|e| panic!("scheduler construction failed: {e}"));

    assert!(scheduler.schedule_test(scheduled("t1", 0.4)).await.is_ok());
    let _ = scheduler.get_next_test().await;

    let history = scheduler.history();
    assert_eq!(history.len(), 2, "one Queued event and one Scheduled event");
    assert!(history.iter().all(|event| event.test_id == "t1"));

    let metrics = scheduler.metrics();
    assert_eq!(metrics.decisions_made, 2);
    // The four unmeasurable fields stay structurally absent.
    assert!(metrics.scheduling_accuracy.is_none());
    assert!(metrics.queue_efficiency.is_none());

    let stats = scheduler.queue_statistics();
    assert_eq!(stats.total_enqueued, 1);
    assert_eq!(stats.total_dequeued, 1);
}
