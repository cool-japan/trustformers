//! Regression tests for [`ChannelHealthMonitor`].
//!
//! Until 0.2.1 the monitor was an empty struct: `get_channel_health` answered
//! `healthy: true, success_rate: 1.0` for any string at all, registration was a
//! no-op and `get_all_health_status` was always empty. Every test here asserts
//! something that answer could not produce.

use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// A channel whose health check reports whatever the test tells it to and
/// counts how many times it was asked.
#[derive(Debug)]
struct ProbeChannel {
    name: String,
    usable: AtomicBool,
    checks: AtomicUsize,
}

impl ProbeChannel {
    fn new(name: &str, usable: bool) -> Arc<Self> {
        Arc::new(Self {
            name: name.to_string(),
            usable: AtomicBool::new(usable),
            checks: AtomicUsize::new(0),
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for ProbeChannel {
    async fn send_notification(&self, _notification: &Notification) -> Result<DeliveryResult> {
        Err(anyhow!("the probe channel does not deliver"))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, _notification_type: &str) -> bool {
        true
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        self.checks.fetch_add(1, Ordering::Relaxed);
        Ok(ChannelHealth {
            healthy: self.usable.load(Ordering::Relaxed),
            last_check: Utc::now(),
            consecutive_failures: if self.usable.load(Ordering::Relaxed) { 0 } else { 3 },
            success_rate: if self.usable.load(Ordering::Relaxed) { 1.0 } else { 0.0 },
        })
    }
}

fn check_config(failure_threshold: usize, success_threshold: usize) -> HealthCheckConfig {
    HealthCheckConfig {
        enabled: true,
        interval: std::time::Duration::from_secs(60),
        timeout: std::time::Duration::from_secs(1),
        failure_threshold,
        success_threshold,
    }
}

async fn monitor() -> ChannelHealthMonitor {
    ChannelHealthMonitor::new(NotificationConfig::default())
        .await
        .expect("monitor constructs")
}

#[tokio::test]
async fn an_unregistered_channel_has_no_health_rather_than_a_healthy_default() {
    let monitor = monitor().await;
    let error = monitor
        .get_channel_health("never-registered")
        .await
        .expect_err("an unknown channel must not report itself healthy");
    assert!(error.to_string().contains("not registered"), "{error}");
}

#[tokio::test]
async fn health_comes_from_the_channels_own_check() {
    let monitor = monitor().await;
    let channel = ProbeChannel::new("probe", true);
    monitor
        .register_channel(channel.clone(), check_config(1, 1))
        .await
        .expect("registration succeeds");
    let health = monitor.get_channel_health("probe").await.expect("probe runs");
    assert!(health.healthy);
    assert_eq!(
        channel.checks.load(Ordering::Relaxed),
        1,
        "the channel must actually be asked"
    );
}

#[tokio::test]
async fn an_unusable_channel_is_reported_unhealthy() {
    let monitor = monitor().await;
    let channel = ProbeChannel::new("dead", false);
    monitor
        .register_channel(channel, check_config(1, 1))
        .await
        .expect("registration succeeds");
    let health = monitor.get_channel_health("dead").await.expect("probe runs");
    assert!(
        !health.healthy,
        "a channel that reports itself unusable is not healthy"
    );
    assert_eq!(health.consecutive_failures, 3);
}

#[tokio::test]
async fn the_failure_threshold_is_applied_as_hysteresis() {
    let monitor = monitor().await;
    let channel = ProbeChannel::new("flaky", true);
    monitor
        .register_channel(channel.clone(), check_config(2, 1))
        .await
        .expect("registration succeeds");
    assert!(monitor.check_now("flaky").await.expect("probe runs").healthy);

    channel.usable.store(false, Ordering::Relaxed);
    assert!(
        monitor.check_now("flaky").await.expect("probe runs").healthy,
        "one bad probe is below the threshold of two"
    );
    assert!(
        !monitor.check_now("flaky").await.expect("probe runs").healthy,
        "the second consecutive bad probe crosses the threshold"
    );

    channel.usable.store(true, Ordering::Relaxed);
    assert!(
        monitor.check_now("flaky").await.expect("probe runs").healthy,
        "one good probe meets the success threshold of one"
    );
}

#[tokio::test]
async fn status_map_is_empty_until_something_has_actually_been_measured() {
    let monitor = monitor().await;
    let channel = ProbeChannel::new("probe", true);
    monitor
        .register_channel(channel, check_config(1, 1))
        .await
        .expect("registration succeeds");
    assert!(
        monitor.get_all_health_status().await.is_empty(),
        "a channel awaiting its first probe has no measured status"
    );
    monitor.check_now("probe").await.expect("probe runs");
    assert!(monitor.get_all_health_status().await.contains_key("probe"));
}

#[tokio::test]
async fn unregistering_removes_the_health_record() {
    let monitor = monitor().await;
    monitor
        .register_channel(ProbeChannel::new("probe", true), check_config(1, 1))
        .await
        .expect("registration succeeds");
    monitor.check_now("probe").await.expect("probe runs");
    monitor.unregister_channel("probe").await.expect("unregistration succeeds");
    assert!(monitor.get_channel_health("probe").await.is_err());
    assert!(monitor.get_all_health_status().await.is_empty());
}

#[tokio::test]
async fn registration_refuses_a_channel_whose_health_checks_are_switched_off() {
    let monitor = monitor().await;
    let mut config = check_config(1, 1);
    config.enabled = false;
    let error = monitor
        .register_channel(ProbeChannel::new("probe", true), config)
        .await
        .expect_err("a channel with checks off cannot be monitored");
    assert!(error.to_string().contains("disabled"), "{error}");
}

#[tokio::test]
async fn starting_with_monitoring_disabled_is_an_error_not_a_silent_no_op() {
    let config = NotificationConfig {
        enable_health_monitoring: false,
        ..NotificationConfig::default()
    };
    let monitor = ChannelHealthMonitor::new(config).await.expect("monitor constructs");
    let error = monitor.start().await.expect_err("nothing will be monitored");
    assert!(error.to_string().contains("disabled"), "{error}");
}

#[tokio::test]
async fn a_probe_that_times_out_counts_against_the_channel() {
    /// A channel whose health check never returns within the timeout.
    #[derive(Debug)]
    struct StalledChannel;

    #[async_trait::async_trait]
    impl NotificationChannel for StalledChannel {
        async fn send_notification(&self, _notification: &Notification) -> Result<DeliveryResult> {
            Err(anyhow!("the stalled channel does not deliver"))
        }

        fn name(&self) -> &str {
            "stalled"
        }

        fn supports(&self, _notification_type: &str) -> bool {
            true
        }

        async fn health_check(&self) -> Result<ChannelHealth> {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            Err(anyhow!("unreachable"))
        }
    }

    let monitor = monitor().await;
    let mut config = check_config(1, 1);
    config.timeout = std::time::Duration::from_millis(20);
    monitor
        .register_channel(Arc::new(StalledChannel), config)
        .await
        .expect("registration succeeds");
    let health = monitor.check_now("stalled").await.expect("the probe itself resolves");
    assert!(
        !health.healthy,
        "a probe that never answered is not a healthy channel"
    );
    assert_eq!(health.success_rate, 0.0);
}
