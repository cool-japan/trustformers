//! # Channel Health Monitoring System
//!
//! Tracks the health of the notification channels registered with the
//! [`NotificationManager`](super::manager::NotificationManager), by running the
//! channels' own [`NotificationChannel::health_check`] probes on the interval
//! each channel was registered with.
//!
//! ## Changed in 0.2.1
//!
//! This type used to be an empty struct. `register_channel` and
//! `unregister_channel` were `Ok(())` no-ops, `get_all_health_status` returned
//! an empty map, and `get_channel_health` answered
//! `healthy: true, consecutive_failures: 0, success_rate: 1.0` for **any**
//! string, including channels that had never been registered and channels whose
//! transport does not exist. `NotificationDeliveryEngine::deliver_to_channel`
//! gates delivery on that answer, so the "health gate" admitted everything, and
//! `NotificationManager::get_channel_health` published an empty map that read as
//! "nothing is wrong".
//!
//! The monitor now holds the registered channels, calls their real health
//! checks (each already derives its answer from
//! [`ChannelCounters`](super::channels::ChannelCounters), i.e. from deliveries
//! that actually happened), and applies the registered
//! [`HealthCheckConfig`]'s failure/success thresholds as hysteresis. Asking
//! about a channel that was never registered is an error, not a clean bill of
//! health.

use super::channels::NotificationChannel;
use super::types::*;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, warn};

/// Channel health status
#[derive(Debug, Clone)]
pub struct ChannelHealth {
    /// Whether the channel is currently considered usable.
    pub healthy: bool,
    /// When the reported state was last established.
    pub last_check: DateTime<Utc>,
    /// Failed deliveries since the channel's last success.
    pub consecutive_failures: usize,
    /// Observed delivery success rate.
    pub success_rate: f32,
}

/// What the monitor knows about one registered channel.
struct Registration {
    channel: Arc<dyn NotificationChannel + Send + Sync>,
    check_config: HealthCheckConfig,
    /// Last probe result, or `None` when the channel has not been probed yet.
    last: Option<ChannelHealth>,
    /// Consecutive probes that reported the channel usable.
    consecutive_ok: usize,
    /// Consecutive probes that reported it unusable (or failed outright).
    consecutive_bad: usize,
    /// Whether the thresholds have currently latched the channel as usable.
    latched_healthy: bool,
}

impl Registration {
    fn new(
        channel: Arc<dyn NotificationChannel + Send + Sync>,
        check_config: HealthCheckConfig,
    ) -> Self {
        Self {
            channel,
            check_config,
            last: None,
            consecutive_ok: 0,
            consecutive_bad: 0,
            // A channel starts usable: nothing has failed yet. The first probe
            // replaces this with a measurement.
            latched_healthy: true,
        }
    }

    /// Folds one probe outcome into the latched state and returns what the
    /// monitor should report.
    ///
    /// `raw` is what the channel said about itself; `None` means the probe
    /// itself failed or timed out, which counts as unusable.
    fn apply(&mut self, raw: Option<ChannelHealth>) -> ChannelHealth {
        let usable = raw.as_ref().is_some_and(|health| health.healthy);
        if usable {
            self.consecutive_ok += 1;
            self.consecutive_bad = 0;
            if self.consecutive_ok >= self.check_config.success_threshold.max(1) {
                self.latched_healthy = true;
            }
        } else {
            self.consecutive_bad += 1;
            self.consecutive_ok = 0;
            if self.consecutive_bad >= self.check_config.failure_threshold.max(1) {
                self.latched_healthy = false;
            }
        }

        let reported = match raw {
            Some(health) => ChannelHealth {
                healthy: self.latched_healthy,
                ..health
            },
            None => ChannelHealth {
                healthy: self.latched_healthy,
                last_check: Utc::now(),
                consecutive_failures: self.consecutive_bad,
                // The probe did not complete, so nothing was measured about the
                // delivery rate; carry the last known figure forward rather
                // than inventing one, and 0.0 when there is no last figure.
                success_rate: self.last.as_ref().map_or(0.0, |last| last.success_rate),
            },
        };
        self.last = Some(reported.clone());
        reported
    }
}

/// Monitors the health of registered notification channels.
#[derive(Debug)]
pub struct ChannelHealthMonitor {
    config: NotificationConfig,
    registrations: Arc<RwLock<HashMap<String, Registration>>>,
    poller: Mutex<Option<JoinHandle<()>>>,
}

impl std::fmt::Debug for Registration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registration")
            .field("channel", &self.channel.name())
            .field("last", &self.last)
            .field("latched_healthy", &self.latched_healthy)
            .finish()
    }
}

impl ChannelHealthMonitor {
    /// Creates a monitor with no channels registered.
    pub async fn new(config: NotificationConfig) -> Result<Self> {
        Ok(Self {
            config,
            registrations: Arc::new(RwLock::new(HashMap::new())),
            poller: Mutex::new(None),
        })
    }

    /// Starts the background poller.
    ///
    /// The poller ticks at the shortest registered check interval (falling back
    /// to the manager's `health_check_interval`) and probes each channel whose
    /// own interval has elapsed. Calling `start` twice replaces the poller.
    pub async fn start(&self) -> Result<()> {
        if !self.config.enable_health_monitoring {
            return Err(anyhow!(
                "channel health monitoring is disabled in the notification configuration; \
                 no channel health will be observed"
            ));
        }
        let registrations = Arc::clone(&self.registrations);
        let fallback_interval = self.config.health_check_interval;
        let handle = tokio::spawn(async move {
            loop {
                let tick = {
                    let guard = registrations.read().await;
                    guard
                        .values()
                        .map(|registration| registration.check_config.interval)
                        .min()
                        .unwrap_or(fallback_interval)
                        .max(std::time::Duration::from_millis(50))
                };
                tokio::time::sleep(tick).await;
                Self::probe_due(&registrations).await;
            }
        });
        if let Some(previous) = self.poller.lock().replace(handle) {
            previous.abort();
        }
        Ok(())
    }

    /// Probes every channel whose interval has elapsed since its last check.
    async fn probe_due(registrations: &Arc<RwLock<HashMap<String, Registration>>>) {
        let due: Vec<String> = {
            let guard = registrations.read().await;
            let now = Utc::now();
            guard
                .iter()
                .filter(|(_, registration)| match &registration.last {
                    None => true,
                    Some(last) => {
                        let elapsed = now.signed_duration_since(last.last_check).to_std();
                        elapsed.is_ok_and(|elapsed| elapsed >= registration.check_config.interval)
                    },
                })
                .map(|(name, _)| name.clone())
                .collect()
        };
        for name in due {
            if let Err(error) = Self::probe_one(registrations, &name).await {
                debug!("health probe for channel {name} could not run: {error}");
            }
        }
    }

    /// Runs one channel's health check and folds the outcome into its state.
    async fn probe_one(
        registrations: &Arc<RwLock<HashMap<String, Registration>>>,
        name: &str,
    ) -> Result<ChannelHealth> {
        let (channel, timeout) = {
            let guard = registrations.read().await;
            let registration = guard.get(name).ok_or_else(|| {
                anyhow!("notification channel '{name}' is not registered for health monitoring")
            })?;
            (
                Arc::clone(&registration.channel),
                registration.check_config.timeout,
            )
        };

        let raw = match tokio::time::timeout(timeout, channel.health_check()).await {
            Ok(Ok(health)) => Some(health),
            Ok(Err(error)) => {
                warn!("health check for channel {name} failed: {error}");
                None
            },
            Err(_) => {
                warn!("health check for channel {name} timed out after {timeout:?}");
                None
            },
        };

        let mut guard = registrations.write().await;
        let registration = guard
            .get_mut(name)
            .ok_or_else(|| anyhow!("notification channel '{name}' was unregistered mid-probe"))?;
        Ok(registration.apply(raw))
    }

    /// Registers a channel so its health is tracked.
    pub async fn register_channel(
        &self,
        channel: Arc<dyn NotificationChannel + Send + Sync>,
        config: HealthCheckConfig,
    ) -> Result<()> {
        if !config.enabled {
            return Err(anyhow!(
                "health checks are disabled for channel '{}'; it cannot be registered for \
                 monitoring",
                channel.name()
            ));
        }
        let name = channel.name().to_string();
        self.registrations
            .write()
            .await
            .insert(name, Registration::new(channel, config));
        Ok(())
    }

    /// Stops tracking a channel. Unregistering a channel that was never
    /// registered is not an error -- the caller's post-condition is met either
    /// way.
    pub async fn unregister_channel(&self, name: &str) -> Result<()> {
        self.registrations.write().await.remove(name);
        Ok(())
    }

    /// Health of every registered channel that has been probed at least once.
    ///
    /// Channels awaiting their first probe are absent from the map rather than
    /// present with an invented status.
    pub async fn get_all_health_status(&self) -> HashMap<String, ChannelHealth> {
        self.registrations
            .read()
            .await
            .iter()
            .filter_map(|(name, registration)| {
                registration.last.clone().map(|health| (name.clone(), health))
            })
            .collect()
    }

    /// Health of one channel.
    ///
    /// Returns the most recent probe result, probing on demand when the channel
    /// has not been checked yet. Returns an error -- never a healthy default --
    /// for a channel that is not registered.
    pub async fn get_channel_health(&self, channel_name: &str) -> Result<ChannelHealth> {
        {
            let guard = self.registrations.read().await;
            let registration = guard.get(channel_name).ok_or_else(|| {
                anyhow!(
                    "notification channel '{channel_name}' is not registered for health \
                     monitoring, so its health is unknown"
                )
            })?;
            if let Some(health) = registration.last.clone() {
                return Ok(health);
            }
        }
        Self::probe_one(&self.registrations, channel_name).await
    }

    /// Probes one channel immediately, bypassing its interval, and returns the
    /// freshly measured state.
    pub async fn check_now(&self, channel_name: &str) -> Result<ChannelHealth> {
        Self::probe_one(&self.registrations, channel_name).await
    }

    /// Stops the poller and drops every registration.
    pub async fn shutdown(&self) -> Result<()> {
        if let Some(handle) = self.poller.lock().take() {
            handle.abort();
        }
        self.registrations.write().await.clear();
        Ok(())
    }
}

#[cfg(test)]
#[path = "health_monitor_tests.rs"]
mod health_monitor_tests;
