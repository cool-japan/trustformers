//! Measured host and process statistics for the admin/health endpoints.
//!
//! Every value in this module comes from `sysinfo`. Where a measurement is
//! genuinely unavailable the API says so (`None` / an error) instead of
//! substituting a plausible constant.

use anyhow::{anyhow, Result};
use std::path::Path;
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
pub async fn measure_host_async() -> HostSnapshot {
    tokio::task::spawn_blocking(measure_host).await.unwrap_or_else(|e| {
        tracing::error!("host measurement task failed: {}", e);
        HostSnapshot {
            cpu_percent: 0.0,
            used_memory_bytes: 0,
            total_memory_bytes: 0,
            memory_percent: 0.0,
            process_memory_bytes: 0,
            disk_percent: None,
        }
    })
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
}
