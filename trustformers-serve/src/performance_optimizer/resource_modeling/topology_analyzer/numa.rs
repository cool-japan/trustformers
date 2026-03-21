//! NUMA topology detection and analysis

use super::config::*;
use anyhow::Result;
use std::path::Path;

/// NUMA topology detector
pub struct NumaTopologyDetector {
    /// Configuration for NUMA detection
    pub config: NumaDetectionConfig,
}

impl NumaTopologyDetector {
    /// Create a new NUMA topology detector
    pub async fn new() -> Result<Self> {
        Ok(Self {
            config: NumaDetectionConfig::default(),
        })
    }

    /// Detect full NUMA topology
    pub async fn detect_full_topology(&self) -> Result<NumaTopologyAdvanced> {
        let domains = detect_numa_domains()?;
        let advanced_metrics = build_advanced_metrics(&domains);
        Ok(NumaTopologyAdvanced {
            domains,
            advanced_metrics,
        })
    }
}

/// Advanced NUMA topology information
pub struct NumaTopologyAdvanced {
    /// NUMA domains
    pub domains: Vec<NumaDomain>,

    /// Advanced metrics
    pub advanced_metrics: NumaAdvancedMetrics,
}

/// NUMA domain information
pub struct NumaDomain {
    /// Domain ID
    pub id: u32,

    /// CPUs in this domain
    pub cpus: Vec<u32>,

    /// Memory regions
    pub memory_regions: Vec<MemoryRegion>,
}

/// Memory region information
pub struct MemoryRegion {
    /// Start address
    pub start_address: u64,

    /// Size in bytes
    pub size: u64,

    /// Memory type
    pub memory_type: String,
}

/// Advanced NUMA metrics
pub struct NumaAdvancedMetrics {
    /// Cross-domain latencies
    pub cross_domain_latencies: Vec<Vec<f64>>,

    /// Bandwidth measurements
    pub bandwidth_measurements: Vec<Vec<f64>>,
}

// =============================================================================
// Platform-aware NUMA detection implementation
// =============================================================================

/// Detect NUMA domains using platform-specific methods with graceful fallback.
fn detect_numa_domains() -> Result<Vec<NumaDomain>> {
    // Try Linux sysfs first
    if let Some(domains) = detect_numa_domains_linux() {
        if !domains.is_empty() {
            return Ok(domains);
        }
    }

    // Try macOS sysctl
    if let Some(domains) = detect_numa_domains_macos() {
        if !domains.is_empty() {
            return Ok(domains);
        }
    }

    // Fallback: single NUMA node with all CPUs
    Ok(build_fallback_domains())
}

/// Attempt Linux NUMA detection via /sys/devices/system/node/
fn detect_numa_domains_linux() -> Option<Vec<NumaDomain>> {
    let node_base = Path::new("/sys/devices/system/node");
    if !node_base.exists() {
        return None;
    }

    let entries = std::fs::read_dir(node_base).ok()?;
    let mut domains = Vec::new();

    for entry in entries {
        let entry = entry.ok()?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if !name_str.starts_with("node") {
            continue;
        }

        let node_id_str = name_str.strip_prefix("node")?;
        let node_id: u32 = node_id_str.parse().ok()?;

        let cpus = parse_linux_node_cpus(node_base, node_id);
        let memory_regions = parse_linux_node_memory(node_base, node_id);

        domains.push(NumaDomain {
            id: node_id,
            cpus,
            memory_regions,
        });
    }

    domains.sort_by_key(|d| d.id);
    Some(domains)
}

/// Parse CPU list for a given NUMA node on Linux.
fn parse_linux_node_cpus(node_base: &Path, node_id: u32) -> Vec<u32> {
    let cpulist_path = node_base.join(format!("node{}", node_id)).join("cpulist");

    let content = match std::fs::read_to_string(&cpulist_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    parse_cpu_range_list(content.trim())
}

/// Parse a CPU range list string like "0-3,8-11" into a Vec of CPU IDs.
fn parse_cpu_range_list(s: &str) -> Vec<u32> {
    let mut cpus = Vec::new();
    if s.is_empty() {
        return cpus;
    }
    for part in s.split(',') {
        let part = part.trim();
        if let Some((start_str, end_str)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) = (
                start_str.trim().parse::<u32>(),
                end_str.trim().parse::<u32>(),
            ) {
                for cpu in start..=end {
                    cpus.push(cpu);
                }
            }
        } else if let Ok(cpu) = part.parse::<u32>() {
            cpus.push(cpu);
        }
    }
    cpus
}

/// Parse memory info for a NUMA node on Linux.
fn parse_linux_node_memory(node_base: &Path, node_id: u32) -> Vec<MemoryRegion> {
    let meminfo_path = node_base.join(format!("node{}", node_id)).join("meminfo");

    let content = match std::fs::read_to_string(&meminfo_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut total_kb: u64 = 0;
    for line in content.lines() {
        // Lines look like: "Node 0 MemTotal:       12345678 kB"
        if line.contains("MemTotal") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // The value is typically the 4th token
            if parts.len() >= 4 {
                total_kb = parts[3].parse().unwrap_or(0);
            }
        }
    }

    if total_kb > 0 {
        vec![MemoryRegion {
            start_address: 0,
            size: total_kb * 1024,
            memory_type: "DRAM".to_string(),
        }]
    } else {
        Vec::new()
    }
}

/// Attempt macOS NUMA detection via sysctl.
///
/// macOS generally presents a single NUMA node (UMA architecture),
/// but we properly detect the CPU count and memory size.
fn detect_numa_domains_macos() -> Option<Vec<NumaDomain>> {
    // Only attempt on macOS
    if cfg!(not(target_os = "macos")) {
        return None;
    }

    let cpu_count = sysctl_read_u64("hw.ncpu").unwrap_or(0);
    let mem_size = sysctl_read_u64("hw.memsize").unwrap_or(0);

    if cpu_count == 0 {
        return None;
    }

    let cpus: Vec<u32> = (0..cpu_count as u32).collect();

    let memory_regions = if mem_size > 0 {
        vec![MemoryRegion {
            start_address: 0,
            size: mem_size,
            memory_type: "DRAM".to_string(),
        }]
    } else {
        Vec::new()
    };

    Some(vec![NumaDomain {
        id: 0,
        cpus,
        memory_regions,
    }])
}

/// Read an integer value from sysctl.
fn sysctl_read_u64(key: &str) -> Option<u64> {
    let output = std::process::Command::new("sysctl").arg("-n").arg(key).output().ok()?;

    if !output.status.success() {
        return None;
    }

    let value_str = String::from_utf8(output.stdout).ok()?;
    value_str.trim().parse::<u64>().ok()
}

/// Build fallback domains when platform-specific detection fails.
fn build_fallback_domains() -> Vec<NumaDomain> {
    let cpu_count = num_cpus::get() as u32;
    let cpus: Vec<u32> = (0..cpu_count).collect();

    vec![NumaDomain {
        id: 0,
        cpus,
        memory_regions: vec![MemoryRegion {
            start_address: 0,
            size: get_total_memory_fallback(),
            memory_type: "DRAM".to_string(),
        }],
    }]
}

/// Get total system memory as a fallback using sysinfo.
fn get_total_memory_fallback() -> u64 {
    use sysinfo::System;
    let sys = System::new_all();
    sys.total_memory()
}

/// Build advanced NUMA metrics (latency/bandwidth matrices) from detected domains.
fn build_advanced_metrics(domains: &[NumaDomain]) -> NumaAdvancedMetrics {
    let n = domains.len();
    if n == 0 {
        return NumaAdvancedMetrics {
            cross_domain_latencies: Vec::new(),
            bandwidth_measurements: Vec::new(),
        };
    }

    // Build estimated latency matrix:
    //   - local access: ~100ns
    //   - remote access: ~300ns (estimated)
    let mut latencies = vec![vec![0.0_f64; n]; n];
    let mut bandwidths = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                latencies[i][j] = 100.0; // local latency estimate in ns
                bandwidths[i][j] = 50_000.0; // local bandwidth estimate in MB/s
            } else {
                latencies[i][j] = 300.0; // remote latency estimate in ns
                bandwidths[i][j] = 20_000.0; // remote bandwidth estimate in MB/s
            }
        }
    }

    // Try to read actual NUMA distances on Linux
    if let Some(distance_matrix) = read_linux_numa_distances(domains) {
        for i in 0..n.min(distance_matrix.len()) {
            for j in 0..n.min(distance_matrix[i].len()) {
                // NUMA distance 10 = local, higher = remote
                // Convert distance to estimated latency: base_latency * (distance / 10)
                let distance = distance_matrix[i][j];
                latencies[i][j] = 100.0 * (distance / 10.0);
                // Bandwidth inversely proportional to distance
                if distance > 0.0 {
                    bandwidths[i][j] = 50_000.0 * (10.0 / distance);
                }
            }
        }
    }

    NumaAdvancedMetrics {
        cross_domain_latencies: latencies,
        bandwidth_measurements: bandwidths,
    }
}

/// Read NUMA distance matrix from Linux sysfs.
fn read_linux_numa_distances(domains: &[NumaDomain]) -> Option<Vec<Vec<f64>>> {
    let mut matrix = Vec::new();

    for domain in domains {
        let distance_path = format!("/sys/devices/system/node/node{}/distance", domain.id);
        let content = std::fs::read_to_string(distance_path).ok()?;
        let row: Vec<f64> =
            content.split_whitespace().filter_map(|s| s.parse::<f64>().ok()).collect();
        if row.len() != domains.len() {
            return None;
        }
        matrix.push(row);
    }

    Some(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_numa_detector_creation() {
        let detector = NumaTopologyDetector::new().await;
        assert!(detector.is_ok());
    }

    #[tokio::test]
    async fn test_detect_full_topology_succeeds() {
        let detector = NumaTopologyDetector::new().await.expect("failed to create detector");
        let result = detector.detect_full_topology().await;
        assert!(
            result.is_ok(),
            "detect_full_topology should not fail: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_topology_has_at_least_one_domain() {
        let detector = NumaTopologyDetector::new().await.expect("failed to create detector");
        let topology = detector.detect_full_topology().await.expect("detection failed");
        assert!(
            !topology.domains.is_empty(),
            "should have at least one NUMA domain"
        );
    }

    #[tokio::test]
    async fn test_domain_has_cpus() {
        let detector = NumaTopologyDetector::new().await.expect("failed to create detector");
        let topology = detector.detect_full_topology().await.expect("detection failed");
        let domain = &topology.domains[0];
        assert!(
            !domain.cpus.is_empty(),
            "domain 0 should have at least one CPU"
        );
    }

    #[tokio::test]
    async fn test_cpu_count_matches_system() {
        let detector = NumaTopologyDetector::new().await.expect("failed to create detector");
        let topology = detector.detect_full_topology().await.expect("detection failed");
        let total_cpus: usize = topology.domains.iter().map(|d| d.cpus.len()).sum();
        // num_cpus::get() respects cgroup limits (containers), while sysfs-based
        // NUMA detection reports physical CPUs. In containerized environments
        // these can differ, so we read the hardware CPU count from the same
        // sysfs source used by NUMA detection.
        let system_cpus_logical = num_cpus::get();
        let hw_cpus = std::fs::read_to_string("/sys/devices/system/cpu/present")
            .ok()
            .map(|s| parse_cpu_range_list(s.trim()).len())
            .unwrap_or(system_cpus_logical);
        assert!(
            total_cpus >= system_cpus_logical && total_cpus <= hw_cpus,
            "detected CPU count ({}) should be between cgroup-limited ({}) and hardware ({})",
            total_cpus,
            system_cpus_logical,
            hw_cpus,
        );
    }

    #[tokio::test]
    async fn test_advanced_metrics_dimensions() {
        let detector = NumaTopologyDetector::new().await.expect("failed to create detector");
        let topology = detector.detect_full_topology().await.expect("detection failed");
        let n = topology.domains.len();
        assert_eq!(topology.advanced_metrics.cross_domain_latencies.len(), n);
        assert_eq!(topology.advanced_metrics.bandwidth_measurements.len(), n);
        for row in &topology.advanced_metrics.cross_domain_latencies {
            assert_eq!(row.len(), n);
        }
        for row in &topology.advanced_metrics.bandwidth_measurements {
            assert_eq!(row.len(), n);
        }
    }

    #[test]
    fn test_parse_cpu_range_list_simple() {
        assert_eq!(parse_cpu_range_list("0-3"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_parse_cpu_range_list_multiple() {
        assert_eq!(parse_cpu_range_list("0-1,4-5"), vec![0, 1, 4, 5]);
    }

    #[test]
    fn test_parse_cpu_range_list_single() {
        assert_eq!(parse_cpu_range_list("7"), vec![7]);
    }

    #[test]
    fn test_parse_cpu_range_list_empty() {
        assert_eq!(parse_cpu_range_list(""), Vec::<u32>::new());
    }

    #[test]
    fn test_fallback_domains() {
        let domains = build_fallback_domains();
        assert_eq!(domains.len(), 1);
        assert_eq!(domains[0].id, 0);
        assert!(!domains[0].cpus.is_empty());
    }

    #[test]
    fn test_build_advanced_metrics_empty() {
        let metrics = build_advanced_metrics(&[]);
        assert!(metrics.cross_domain_latencies.is_empty());
        assert!(metrics.bandwidth_measurements.is_empty());
    }

    #[test]
    fn test_build_advanced_metrics_single_node() {
        let domains = vec![NumaDomain {
            id: 0,
            cpus: vec![0, 1],
            memory_regions: Vec::new(),
        }];
        let metrics = build_advanced_metrics(&domains);
        assert_eq!(metrics.cross_domain_latencies.len(), 1);
        assert_eq!(metrics.cross_domain_latencies[0].len(), 1);
        // Local latency should be ~100ns
        assert!((metrics.cross_domain_latencies[0][0] - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_build_advanced_metrics_two_nodes() {
        let domains = vec![
            NumaDomain {
                id: 0,
                cpus: vec![0, 1],
                memory_regions: Vec::new(),
            },
            NumaDomain {
                id: 1,
                cpus: vec![2, 3],
                memory_regions: Vec::new(),
            },
        ];
        let metrics = build_advanced_metrics(&domains);
        assert_eq!(metrics.cross_domain_latencies.len(), 2);
        // Local latency < remote latency
        assert!(metrics.cross_domain_latencies[0][0] < metrics.cross_domain_latencies[0][1]);
    }
}
