//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant};

/// Simulate work for testing
pub(super) async fn simulate_work(duration: Duration) {
    let end_time = Instant::now() + duration;
    while Instant::now() < end_time {
        for _ in 0..1000 {
            std::hint::black_box(std::ptr::null::<u8>());
        }
        tokio::task::yield_now().await;
    }
}
/// Simulate heavy work with memory allocations
pub(super) async fn simulate_heavy_work(duration: Duration) {
    let end_time = Instant::now() + duration;
    let mut data = Vec::new();
    while Instant::now() < end_time {
        data.push(vec![0u8; 1024]);
        if data.len() > 100 {
            data.clear();
        }
        for _ in 0..10000 {
            std::hint::black_box(std::ptr::null::<u8>());
        }
        tokio::task::yield_now().await;
    }
}
/// Get current memory usage in MB (simplified implementation)
pub(super) fn get_memory_usage_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Ok(kb) = line.split_whitespace().nth(1).unwrap_or("0").parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    100.0
}
#[cfg(test)]
mod tests {

    use super::super::types::*;
    use std::time::Duration;
    #[tokio::test]
    async fn test_async_chaos_framework_creation() {
        let framework = AsyncRuntimeChaosFramework::new();
        framework.start().await.unwrap();
    }
    #[tokio::test]
    async fn test_task_cancellation_basic() {
        let framework = AsyncRuntimeChaosFramework::new();
        framework.start().await.unwrap();
        let config = TaskCancellationConfig {
            task_count: 10,
            task_duration: Duration::from_millis(100),
            cancellation_delay: Duration::from_millis(10),
            ..Default::default()
        };
        let result = framework.test_task_cancellation(config).await.unwrap();
        assert!(result.metrics.contains_key("spawned_tasks"));
        assert!(result.metrics.contains_key("cancelled_tasks"));
    }
    #[tokio::test]
    async fn test_panic_recovery_basic() {
        let framework = AsyncRuntimeChaosFramework::new();
        framework.start().await.unwrap();
        let config = PanicRecoveryConfig {
            total_tasks: 10,
            panic_task_count: 3,
            panic_type: PanicType::Immediate,
        };
        let result = framework.test_async_panic_recovery(config).await.unwrap();
        assert!(result.metrics.contains_key("actual_panics"));
        assert!(result.metrics.contains_key("panic_recoveries"));
    }
}
