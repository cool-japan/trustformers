//! Real Apple-hardware probing via `sysctlbyname`.
//!
//! Everything here is measured on the running machine. There is no lookup table and
//! no chip-model guessing: if a figure cannot be obtained from the OS, it is
//! reported as `None` rather than filled in from a spec sheet.
//!
//! This replaces `detect_device_capabilities(device: &AppleSiliconDevice)`, which
//! "detected" nothing - it matched the chip enum *the caller supplied* against a
//! hardcoded table and then appended a fabricated `mlx_version: "0.15.0"` for a
//! framework this crate does not link.

use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

/// Hardware facts read from the OS at runtime.
///
/// `None` means "this machine does not expose the value"; it never means zero.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbedHardware {
    /// `machdep.cpu.brand_string`, e.g. "Apple M4 Pro".
    pub cpu_brand: String,
    /// `hw.perflevel0.logicalcpu` - performance-core count on Apple Silicon.
    pub performance_cores: Option<u32>,
    /// `hw.perflevel1.logicalcpu` - efficiency-core count on Apple Silicon.
    pub efficiency_cores: Option<u32>,
    /// `hw.logicalcpu` - total logical CPUs.
    pub logical_cores: u32,
    /// `hw.memsize` - installed RAM in bytes.
    pub memory_bytes: u64,
    /// `hw.optional.amx_version` when present: Apple Matrix coprocessor generation.
    pub amx_version: Option<u32>,
    /// `hw.cpufrequency_max` when present, in Hz. Absent on Apple Silicon.
    pub cpu_frequency_max_hz: Option<u64>,
}

impl ProbedHardware {
    /// Installed RAM in GiB.
    pub fn memory_gib(&self) -> f32 {
        self.memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Probe the running machine.
///
/// # Errors
///
/// Returns an error on non-Apple platforms, or when the mandatory sysctls
/// (`machdep.cpu.brand_string`, `hw.logicalcpu`, `hw.memsize`) are unavailable.
/// Reporting an error is the only honest option: this crate has no way to know the
/// hardware of a machine whose OS will not tell it.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn probe_hardware() -> Result<ProbedHardware> {
    let cpu_brand = sysctl_string("machdep.cpu.brand_string")
        .or_else(|| sysctl_string("hw.model"))
        .ok_or_else(|| -> CoreError {
            TrustformersError::hardware_error(
                "sysctlbyname(machdep.cpu.brand_string) is unavailable on this machine",
                "probe_hardware",
            )
            .into()
        })?;
    let logical_cores = sysctl_u64("hw.logicalcpu").ok_or_else(|| -> CoreError {
        TrustformersError::hardware_error(
            "sysctlbyname(hw.logicalcpu) is unavailable on this machine",
            "probe_hardware",
        )
        .into()
    })? as u32;
    let memory_bytes = sysctl_u64("hw.memsize").ok_or_else(|| -> CoreError {
        TrustformersError::hardware_error(
            "sysctlbyname(hw.memsize) is unavailable on this machine",
            "probe_hardware",
        )
        .into()
    })?;

    Ok(ProbedHardware {
        cpu_brand,
        performance_cores: sysctl_u64("hw.perflevel0.logicalcpu").map(|v| v as u32),
        efficiency_cores: sysctl_u64("hw.perflevel1.logicalcpu").map(|v| v as u32),
        logical_cores,
        memory_bytes,
        amx_version: sysctl_u64("hw.optional.amx_version").map(|v| v as u32),
        cpu_frequency_max_hz: sysctl_u64("hw.cpufrequency_max"),
    })
}

/// Hardware probing is Apple-only.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn probe_hardware() -> Result<ProbedHardware> {
    Err(TrustformersError::hardware_error(
        "Apple hardware probing requires macOS or iOS; this build targets another platform",
        "probe_hardware",
    )
    .into())
}

/// Read a NUL-terminated string sysctl. Returns `None` when the key is absent.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn sysctl_string(name: &str) -> Option<String> {
    let key = std::ffi::CString::new(name).ok()?;
    let mut size: libc::size_t = 0;
    // SAFETY: `key` is a valid NUL-terminated C string; passing a null value pointer
    // with a zero-initialised size asks the kernel for the required buffer size.
    let probe = unsafe {
        libc::sysctlbyname(
            key.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if probe != 0 || size == 0 {
        return None;
    }
    let mut buffer = vec![0u8; size];
    // SAFETY: `buffer` has exactly `size` bytes, which is what the probe above asked
    // the kernel to fill.
    let read = unsafe {
        libc::sysctlbyname(
            key.as_ptr(),
            buffer.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if read != 0 {
        return None;
    }
    // The kernel returns a NUL-terminated string; trim at the first NUL.
    let end = buffer.iter().position(|byte| *byte == 0).unwrap_or(buffer.len());
    String::from_utf8(buffer[..end].to_vec()).ok()
}

/// Read an integer sysctl (32- or 64-bit). Returns `None` when the key is absent.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn sysctl_u64(name: &str) -> Option<u64> {
    let key = std::ffi::CString::new(name).ok()?;
    let mut value: u64 = 0;
    let mut size: libc::size_t = std::mem::size_of::<u64>();
    // SAFETY: `value` is a live u64 and `size` describes it exactly; sysctlbyname
    // writes at most `size` bytes and updates `size` with what it wrote.
    let status = unsafe {
        libc::sysctlbyname(
            key.as_ptr(),
            &mut value as *mut u64 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        return None;
    }
    match size {
        4 => Some(u64::from(value as u32)),
        8 => Some(value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: the old `detect_device_capabilities` never touched the machine.
    /// This asserts we read real values from the running kernel.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn probe_reads_real_hardware() -> Result<()> {
        let hw = probe_hardware()?;
        assert!(
            !hw.cpu_brand.is_empty(),
            "cpu brand string must come from sysctl"
        );
        assert!(hw.logical_cores > 0, "logical core count must be positive");
        assert!(
            hw.memory_bytes >= 1024 * 1024 * 1024,
            "memsize must be at least 1 GiB, got {}",
            hw.memory_bytes
        );
        // The probe must agree with the standard library's own view of the machine.
        let std_parallelism = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(hw.logical_cores);
        assert_eq!(
            hw.logical_cores, std_parallelism,
            "sysctl hw.logicalcpu must match std::thread::available_parallelism"
        );
        if let (Some(p), Some(e)) = (hw.performance_cores, hw.efficiency_cores) {
            assert_eq!(
                p + e,
                hw.logical_cores,
                "perflevel0 + perflevel1 must account for every logical core"
            );
        }
        println!(
            "probed: {} | P={:?} E={:?} total={} | {:.1} GiB | amx={:?}",
            hw.cpu_brand,
            hw.performance_cores,
            hw.efficiency_cores,
            hw.logical_cores,
            hw.memory_gib(),
            hw.amx_version
        );
        Ok(())
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn unknown_sysctl_keys_report_absence_not_zero() {
        assert_eq!(sysctl_u64("hw.this.key.does.not.exist"), None);
        assert_eq!(sysctl_string("hw.this.key.does.not.exist"), None);
    }
}
