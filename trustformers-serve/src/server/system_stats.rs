//! Measured host and process statistics for the admin/health endpoints.
//!
//! Every value in this module comes from `sysinfo`. Where a measurement is
//! genuinely unavailable the API says so (`None` / an error) instead of
//! substituting a plausible constant.

use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use sysinfo::{Disks, Pid, ProcessesToUpdate, System};

/// A measured snapshot of host resource usage.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct HostSnapshot {
    /// Mean CPU utilization across all logical CPUs, in percent.
    pub cpu_percent: f64,
    /// System memory in use, in bytes.
    pub used_memory_bytes: u64,
    /// Total system memory, in bytes.
    pub total_memory_bytes: u64,
    /// System memory in use, in percent of total.
    pub memory_percent: f64,
    /// Resident set size of this process, in bytes.
    pub process_memory_bytes: u64,
    /// Disk utilization of the filesystem holding the working directory, in
    /// percent; `None` when no filesystem could be matched.
    pub disk_percent: Option<f64>,
}

/// Take a live measurement of host and process resource usage, off the runtime.
///
/// [`measure_host`] blocks for `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL` between
/// its two CPU samples, which would park a tokio worker; async callers must use
/// this wrapper.
///
/// This substitutes a zeroed snapshot when the underlying `spawn_blocking`
/// task itself fails (join error, not a measurement error) so existing
/// callers keep their current `HostSnapshot`-returning signature; a
/// `total_memory_bytes == 0` reading is that failure; a real host is never
/// reported that way. New callers that can represent absence directly
/// should prefer `measure_host_checked`, which returns `None` instead of
/// synthesizing a value.
pub async fn measure_host_async() -> HostSnapshot {
    measure_host_checked().await.unwrap_or_else(|| HostSnapshot {
        cpu_percent: 0.0,
        used_memory_bytes: 0,
        total_memory_bytes: 0,
        memory_percent: 0.0,
        process_memory_bytes: 0,
        disk_percent: None,
    })
}

/// Take a live measurement of host and process resource usage, off the
/// runtime, reporting `None` (rather than a fabricated zeroed reading) when
/// the underlying blocking task could not complete.
async fn measure_host_checked() -> Option<HostSnapshot> {
    match tokio::task::spawn_blocking(measure_host).await {
        Ok(snapshot) => Some(snapshot),
        Err(e) => {
            tracing::error!("host measurement task failed: {}", e);
            None
        },
    }
}

/// A [`HostSnapshot`] together with when it was taken, so a stale reading is
/// never presented as current.
#[derive(Debug, Clone, Copy)]
pub struct TimestampedSnapshot {
    pub snapshot: HostSnapshot,
    pub sampled_at: Instant,
}

/// The refresh cadence `get_stats` starts
/// [`HostSampler`] with.
pub const DEFAULT_SAMPLE_INTERVAL: Duration = Duration::from_secs(2);

/// A live, continuously-refreshed sample of host resource usage, taken by a
/// long-lived background task instead of freshly on every request.
///
/// Genuinely measuring host stats means real syscalls: two `sysinfo` CPU
/// samples `MINIMUM_CPU_UPDATE_INTERVAL` apart, plus process and disk
/// enumeration, whose latency depends on how loaded the host is right now --
/// exactly when an admin most wants `/admin/stats` to answer promptly.
/// `measure_host_async` already keeps that work off the async executor via
/// `spawn_blocking`, but doing it fresh on every request to a hot admin
/// endpoint still means every caller pays whatever that measurement costs
/// *right now*, with no upper bound; under enough host load the wait can run
/// to seconds, which is untenable for a status endpoint that other tests and
/// tools expect to answer quickly regardless of host load.
///
/// `HostSampler` instead runs one background refresh loop per server
/// instance; [`Self::current`] reads the latest completed sample without
/// waiting on `sysinfo` at all. Before the first sample lands, it honestly
/// returns `None` rather than a synthesized reading.
#[derive(Debug, Clone, Default)]
pub struct HostSampler {
    latest: Arc<RwLock<Option<TimestampedSnapshot>>>,
    started: Arc<AtomicBool>,
}

impl HostSampler {
    /// Create a sampler with no background task running yet. Call
    /// [`Self::ensure_started`] to begin refreshing it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start the background refresh loop, if it is not already running.
    ///
    /// Idempotent and cheap to call unconditionally on every request: the
    /// `swap` is a single atomic operation, and only the very first caller
    /// actually spawns the loop. This is what lets the sampler start lazily
    /// on first use rather than depending on being wired into every server
    /// construction path (production `start()` and the test-only
    /// `create_test_router()` both reach it this way, uniformly).
    pub fn ensure_started(&self, interval: Duration) {
        if self.started.swap(true, Ordering::SeqCst) {
            return;
        }
        let latest = Arc::clone(&self.latest);
        tokio::spawn(async move {
            loop {
                if let Some(snapshot) = measure_host_checked().await {
                    *latest.write() = Some(TimestampedSnapshot {
                        snapshot,
                        sampled_at: Instant::now(),
                    });
                }
                // A failed sample is left as whatever the previous state was
                // (absent, or the last real reading) rather than overwritten
                // with a fabricated value; the loop simply tries again next
                // interval.
                tokio::time::sleep(interval).await;
            }
        });
    }

    /// The latest completed sample, if any has landed yet. Never blocks.
    pub fn current(&self) -> Option<TimestampedSnapshot> {
        *self.latest.read()
    }
}

/// Take a live measurement of host and process resource usage.
///
/// CPU utilization requires two samples separated by at least
/// `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL`; this function performs both and
/// therefore **blocks the calling thread for that interval**. Async callers must
/// use [`measure_host_async`].
pub fn measure_host() -> HostSnapshot {
    let mut system = System::new();

    // Two CPU samples are required before `cpu_usage` is meaningful.
    system.refresh_cpu_usage();
    std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
    system.refresh_cpu_usage();
    system.refresh_memory();

    let cpus = system.cpus();
    let cpu_percent = if cpus.is_empty() {
        0.0
    } else {
        cpus.iter().map(|cpu| cpu.cpu_usage() as f64).sum::<f64>() / cpus.len() as f64
    };

    let total_memory_bytes = system.total_memory();
    let used_memory_bytes = system.used_memory();
    let memory_percent = if total_memory_bytes == 0 {
        0.0
    } else {
        used_memory_bytes as f64 / total_memory_bytes as f64 * 100.0
    };

    let pid = Pid::from_u32(std::process::id());
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    let process_memory_bytes = system.process(pid).map(|p| p.memory()).unwrap_or(0);

    HostSnapshot {
        cpu_percent,
        used_memory_bytes,
        total_memory_bytes,
        memory_percent,
        process_memory_bytes,
        disk_percent: disk_usage_percentage().ok(),
    }
}

/// Resident set size of the current process, in bytes.
///
/// Returns an error when the platform does not report the process, rather than
/// guessing.
pub fn process_resident_bytes() -> Result<u64> {
    let pid = Pid::from_u32(std::process::id());
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    system
        .process(pid)
        .map(|process| process.memory())
        .ok_or_else(|| anyhow!("the platform does not report process {} to sysinfo", pid))
}

/// Disk utilization, in percent, of the filesystem holding the current
/// working directory.
///
/// The filesystem is selected by longest matching mount point, which is the
/// only correct rule when mounts are nested. An error is returned when no mount
/// point matches or when the device reports a zero total.
pub fn disk_usage_percentage() -> Result<f64> {
    let current_dir = std::env::current_dir()?;
    disk_usage_percentage_for(&current_dir)
}

/// Disk utilization, in percent, of the filesystem holding `path`.
pub fn disk_usage_percentage_for(path: &Path) -> Result<f64> {
    let disks = Disks::new_with_refreshed_list();
    if disks.is_empty() {
        return Err(anyhow!("no filesystems reported by the platform"));
    }

    let mut best: Option<(usize, u64, u64)> = None;
    for disk in disks.list() {
        let mount = disk.mount_point();
        if !path.starts_with(mount) {
            continue;
        }
        let depth = mount.components().count();
        if best.map(|(best_depth, _, _)| depth > best_depth).unwrap_or(true) {
            best = Some((depth, disk.total_space(), disk.available_space()));
        }
    }

    let (_, total, available) = best.ok_or_else(|| {
        anyhow!(
            "no mounted filesystem contains {}; cannot compute disk usage",
            path.display()
        )
    })?;

    if total == 0 {
        return Err(anyhow!(
            "filesystem containing {} reports a total size of zero",
            path.display()
        ));
    }

    let used = total.saturating_sub(available);
    Ok((used as f64 / total as f64 * 100.0).clamp(0.0, 100.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: `get_disk_usage_percentage` used to return the constant 80.0.
    #[test]
    fn disk_usage_is_measured_not_constant() {
        let temp = std::env::temp_dir();
        let usage = disk_usage_percentage_for(&temp).expect("temp dir must live on a filesystem");
        assert!(
            (0.0..=100.0).contains(&usage),
            "usage must be a percentage, got {usage}"
        );

        // Cross-check against the same numbers computed independently.
        let disks = Disks::new_with_refreshed_list();
        let mut expected: Option<(usize, f64)> = None;
        for disk in disks.list() {
            let mount = disk.mount_point();
            if !temp.starts_with(mount) || disk.total_space() == 0 {
                continue;
            }
            let depth = mount.components().count();
            let value = (disk.total_space() - disk.available_space()) as f64
                / disk.total_space() as f64
                * 100.0;
            if expected.map(|(d, _)| depth > d).unwrap_or(true) {
                expected = Some((depth, value));
            }
        }
        let (_, expected) = expected.expect("cross-check must find the same filesystem");
        assert!(
            (usage - expected).abs() < 1.0,
            "reported {usage} must track the measured {expected}"
        );
    }

    #[test]
    fn unmounted_path_is_an_error() {
        // A path under no mount point at all cannot exist on a real host, so use
        // a path guaranteed not to be a prefix of any mount: the empty relative
        // path resolved against a synthetic root.
        let result = disk_usage_percentage_for(Path::new("relative/not/absolute"));
        assert!(
            result.is_err(),
            "a non-rooted path must not report a figure"
        );
    }

    #[test]
    fn host_snapshot_is_measured() {
        let snapshot = measure_host();
        assert!(
            snapshot.total_memory_bytes > 0,
            "total memory must be measured"
        );
        assert!(snapshot.used_memory_bytes <= snapshot.total_memory_bytes);
        assert!((0.0..=100.0).contains(&snapshot.memory_percent));
        assert!(snapshot.cpu_percent >= 0.0);
        assert!(
            snapshot.process_memory_bytes > 0,
            "the test process must have a measurable resident size"
        );
    }

    /// Regression: a `HostSampler` with no background loop started must
    /// never synthesize a reading -- `current()` before `ensure_started` is
    /// exactly the "no sample yet" case `/admin/stats` must render as null
    /// fields, not as a fabricated zeroed snapshot.
    #[tokio::test]
    async fn host_sampler_reports_absent_before_first_sample() {
        let sampler = HostSampler::new();
        assert!(
            sampler.current().is_none(),
            "a sampler with no background loop running must not synthesize a reading"
        );
    }

    /// Regression: once started, the sampler eventually carries a real,
    /// freshly timestamped measurement -- the whole point of moving
    /// measurement off the request path.
    #[tokio::test]
    async fn host_sampler_populates_after_starting() {
        let sampler = HostSampler::new();
        sampler.ensure_started(Duration::from_millis(50));
        // The first sample still needs sysinfo's own two-sample CPU delay
        // (`MINIMUM_CPU_UPDATE_INTERVAL`) on a background thread; poll with a
        // generous window rather than pinning the test to that constant. The
        // deadline itself is generous too (not the ~1s the measurement takes
        // on an idle host): `measure_host` does real syscalls (process and
        // disk enumeration) whose latency depends on host load, and a shared
        // CI/dev box running many concurrent builds can genuinely take tens
        // of seconds -- this loop is checking that a sample eventually
        // lands, not bounding its latency (that is precisely what the
        // sampler exists to keep off the request path; see the module doc).
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if let Some(sample) = sampler.current() {
                assert!(
                    sample.snapshot.total_memory_bytes > 0,
                    "a real sample must have measured memory"
                );
                assert!(
                    sample.sampled_at.elapsed() < Duration::from_secs(5),
                    "a just-landed sample must not already read as stale"
                );
                return;
            }
            assert!(
                Instant::now() < deadline,
                "no sample landed within 60 seconds of starting the sampler"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Regression: calling `ensure_started` more than once (the expected
    /// usage -- every `/admin/stats` request calls it) must not panic, and
    /// must not prevent a sample from landing.
    #[tokio::test]
    async fn host_sampler_ensure_started_is_idempotent() {
        let sampler = HostSampler::new();
        for _ in 0..5 {
            sampler.ensure_started(Duration::from_millis(50));
        }
        // See the generous-deadline note in `host_sampler_populates_after_starting`.
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if sampler.current().is_some() {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "no sample landed within 60 seconds of starting the sampler repeatedly"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Regression: `measure_host_async` must keep substituting a detectable
    /// zeroed snapshot only for its own join-failure case, never for a
    /// genuine measurement -- existing callers outside this module key off
    /// `total_memory_bytes == 0` to detect that failure (see
    /// `crate::custom_metrics`), so a real host must never produce it.
    #[tokio::test]
    async fn measure_host_async_is_never_zeroed_on_the_happy_path() {
        let snapshot = measure_host_async().await;
        assert!(
            snapshot.total_memory_bytes > 0,
            "a real measurement must never look like the join-failure sentinel"
        );
    }
}
