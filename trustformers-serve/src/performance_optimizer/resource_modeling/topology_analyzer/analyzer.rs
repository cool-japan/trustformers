//! Main topology analyzer implementation

use super::config::*;
use super::numa::NumaTopologyDetector;
use super::types::*;
use anyhow::Result;

/// Main topology analyzer structure
pub struct TopologyAnalyzer {
    /// Configuration for analysis
    pub config: TopologyAnalysisConfig,

    /// Analysis cache for performance optimization
    pub cache: TopologyAnalysisCache,

    /// Enable debug mode for detailed logging
    pub debug_mode: bool,
}

impl TopologyAnalyzer {
    /// Create a new topology analyzer with default configuration
    pub async fn new() -> Result<Self> {
        Self::with_config(TopologyAnalysisConfig::default()).await
    }

    /// Create a new topology analyzer with custom configuration
    pub async fn with_config(config: TopologyAnalysisConfig) -> Result<Self> {
        Ok(Self {
            config,
            cache: TopologyAnalysisCache::default(),
            debug_mode: false,
        })
    }

    /// Enable debug mode for detailed analysis logging
    pub fn enable_debug_mode(&mut self) {
        self.debug_mode = true;
    }

    /// Analyze complete system topology
    pub async fn analyze_complete_topology(&self) -> Result<TopologyAnalysisResults> {
        let start = std::time::Instant::now();

        // Detect NUMA topology if enabled
        let numa_topology = if self.config.enable_numa_detection {
            match detect_numa_node_count().await {
                Ok(count) => Some(NumaTopologyResults { node_count: count }),
                Err(e) => {
                    if self.debug_mode {
                        log::warn!("NUMA detection failed, using default: {}", e);
                    }
                    Some(NumaTopologyResults { node_count: 1 })
                },
            }
        } else {
            None
        };

        // Detect cache hierarchy
        let cache_analysis = if self.config.enable_cache_analysis {
            detect_cache_hierarchy()
        } else {
            CacheAnalysisResults {
                cache_levels: Vec::new(),
            }
        };

        // Detect memory topology
        let memory_topology = if self.config.enable_memory_analysis {
            detect_memory_topology()
        } else {
            MemoryTopologyResults { memory_channels: 1 }
        };

        let analysis_duration = start.elapsed();

        Ok(TopologyAnalysisResults {
            numa_topology,
            cache_analysis,
            memory_topology,
            analysis_metadata: AnalysisMetadata {
                analysis_duration,
                timestamp: chrono::Utc::now(),
            },
        })
    }
}

/// Results of topology analysis
pub struct TopologyAnalysisResults {
    /// NUMA topology results
    pub numa_topology: Option<NumaTopologyResults>,

    /// Cache analysis results
    pub cache_analysis: CacheAnalysisResults,

    /// Memory topology results
    pub memory_topology: MemoryTopologyResults,

    /// Analysis metadata
    pub analysis_metadata: AnalysisMetadata,
}

/// NUMA topology analysis results
pub struct NumaTopologyResults {
    /// Number of NUMA nodes detected
    pub node_count: usize,
}

/// Cache analysis results
pub struct CacheAnalysisResults {
    /// Number of cache levels detected
    pub cache_levels: Vec<CacheLevelInfo>,
}

/// Cache level information
pub struct CacheLevelInfo {
    /// Cache level (L1, L2, L3, etc.)
    pub level: u32,

    /// Cache size in bytes
    pub size: usize,

    /// Cache type
    pub cache_type: CacheType,
}

/// Memory topology results
pub struct MemoryTopologyResults {
    /// Number of memory channels
    pub memory_channels: usize,
}

/// Analysis metadata
pub struct AnalysisMetadata {
    /// Analysis duration
    pub analysis_duration: std::time::Duration,

    /// Analysis timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// =============================================================================
// Platform-aware topology detection helpers
// =============================================================================

/// Detect the number of NUMA nodes using the full NUMA detector.
async fn detect_numa_node_count() -> Result<usize> {
    let detector = NumaTopologyDetector::new().await?;
    let topology = detector.detect_full_topology().await?;
    Ok(topology.domains.len())
}

/// Detect cache hierarchy using platform-specific methods.
fn detect_cache_hierarchy() -> CacheAnalysisResults {
    let mut cache_levels = Vec::new();

    // Try Linux sysfs cache detection
    if let Some(levels) = detect_cache_hierarchy_linux() {
        return CacheAnalysisResults {
            cache_levels: levels,
        };
    }

    // Try macOS sysctl cache detection
    if let Some(levels) = detect_cache_hierarchy_macos() {
        return CacheAnalysisResults {
            cache_levels: levels,
        };
    }

    // Fallback: provide reasonable defaults
    cache_levels.push(CacheLevelInfo {
        level: 1,
        size: 32 * 1024, // 32KB L1 data cache
        cache_type: CacheType::Data,
    });
    cache_levels.push(CacheLevelInfo {
        level: 2,
        size: 256 * 1024, // 256KB L2 cache
        cache_type: CacheType::Unified,
    });
    cache_levels.push(CacheLevelInfo {
        level: 3,
        size: 8 * 1024 * 1024, // 8MB L3 cache
        cache_type: CacheType::Unified,
    });

    CacheAnalysisResults { cache_levels }
}

/// Detect cache hierarchy from Linux sysfs.
fn detect_cache_hierarchy_linux() -> Option<Vec<CacheLevelInfo>> {
    let cpu0_cache = std::path::Path::new("/sys/devices/system/cpu/cpu0/cache");
    if !cpu0_cache.exists() {
        return None;
    }

    let mut levels = Vec::new();
    let mut seen_levels = std::collections::HashSet::new();

    for index in 0..8_u32 {
        let index_dir = cpu0_cache.join(format!("index{}", index));
        if !index_dir.exists() {
            break;
        }

        let level = read_sysfs_value::<u32>(&index_dir.join("level")).unwrap_or(0);
        let size_str = std::fs::read_to_string(index_dir.join("size")).unwrap_or_default();
        let type_str = std::fs::read_to_string(index_dir.join("type")).unwrap_or_default();

        let size = parse_cache_size(size_str.trim());
        let cache_type = match type_str.trim() {
            "Instruction" => CacheType::Instruction,
            "Data" => CacheType::Data,
            _ => CacheType::Unified,
        };

        // Only add unique (level, type) combinations
        let key = (level, cache_type.clone());
        if level > 0 && size > 0 && seen_levels.insert(key) {
            levels.push(CacheLevelInfo {
                level,
                size,
                cache_type,
            });
        }
    }

    if levels.is_empty() {
        None
    } else {
        levels.sort_by_key(|l| l.level);
        Some(levels)
    }
}

/// Parse cache size strings like "32K", "256K", "8192K", "8M".
fn parse_cache_size(s: &str) -> usize {
    let s = s.trim();
    if s.is_empty() {
        return 0;
    }

    if let Some(num_str) = s.strip_suffix('K') {
        num_str.trim().parse::<usize>().unwrap_or(0) * 1024
    } else if let Some(num_str) = s.strip_suffix('M') {
        num_str.trim().parse::<usize>().unwrap_or(0) * 1024 * 1024
    } else if let Some(num_str) = s.strip_suffix('G') {
        num_str.trim().parse::<usize>().unwrap_or(0) * 1024 * 1024 * 1024
    } else {
        s.parse::<usize>().unwrap_or(0)
    }
}

/// Read a typed value from a sysfs file.
fn read_sysfs_value<T: std::str::FromStr>(path: &std::path::Path) -> Option<T> {
    let content = std::fs::read_to_string(path).ok()?;
    content.trim().parse().ok()
}

/// Detect cache hierarchy on macOS using sysctl.
fn detect_cache_hierarchy_macos() -> Option<Vec<CacheLevelInfo>> {
    if cfg!(not(target_os = "macos")) {
        return None;
    }

    let mut levels = Vec::new();

    // L1 data cache
    if let Some(l1d_size) = sysctl_read_usize("hw.l1dcachesize") {
        levels.push(CacheLevelInfo {
            level: 1,
            size: l1d_size,
            cache_type: CacheType::Data,
        });
    }

    // L1 instruction cache
    if let Some(l1i_size) = sysctl_read_usize("hw.l1icachesize") {
        levels.push(CacheLevelInfo {
            level: 1,
            size: l1i_size,
            cache_type: CacheType::Instruction,
        });
    }

    // L2 cache
    if let Some(l2_size) = sysctl_read_usize("hw.l2cachesize") {
        levels.push(CacheLevelInfo {
            level: 2,
            size: l2_size,
            cache_type: CacheType::Unified,
        });
    }

    // L3 cache (may not exist on all Macs)
    if let Some(l3_size) = sysctl_read_usize("hw.l3cachesize") {
        if l3_size > 0 {
            levels.push(CacheLevelInfo {
                level: 3,
                size: l3_size,
                cache_type: CacheType::Unified,
            });
        }
    }

    if levels.is_empty() {
        None
    } else {
        Some(levels)
    }
}

/// Read a usize from sysctl.
fn sysctl_read_usize(key: &str) -> Option<usize> {
    let output = std::process::Command::new("sysctl").arg("-n").arg(key).output().ok()?;

    if !output.status.success() {
        return None;
    }

    let value_str = String::from_utf8(output.stdout).ok()?;
    value_str.trim().parse::<usize>().ok()
}

/// Detect memory topology (number of memory channels).
fn detect_memory_topology() -> MemoryTopologyResults {
    // Memory channel detection is difficult without specialized tools.
    // We estimate based on CPU count as a reasonable heuristic:
    //   - 1-4 CPUs: likely 1-2 channels
    //   - 8+ CPUs: likely 2-4 channels
    //   - Server-class (32+ CPUs): likely 4-8 channels
    let cpu_count = num_cpus::get_physical();
    let estimated_channels = if cpu_count >= 32 {
        6
    } else if cpu_count >= 16 {
        4
    } else if cpu_count >= 8 {
        2
    } else {
        2
    };

    MemoryTopologyResults {
        memory_channels: estimated_channels,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_topology_analyzer_creation() {
        let analyzer = TopologyAnalyzer::new().await;
        assert!(analyzer.is_ok());
    }

    #[tokio::test]
    async fn test_analyze_complete_topology_succeeds() {
        let analyzer = TopologyAnalyzer::new().await.expect("failed to create analyzer");
        let result = analyzer.analyze_complete_topology().await;
        assert!(
            result.is_ok(),
            "topology analysis should not fail: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_topology_has_numa_info() {
        let analyzer = TopologyAnalyzer::new().await.expect("failed to create analyzer");
        let results = analyzer.analyze_complete_topology().await.expect("analysis failed");
        assert!(
            results.numa_topology.is_some(),
            "NUMA topology should be detected"
        );
        let numa = results.numa_topology.as_ref().expect("numa should exist");
        assert!(numa.node_count >= 1, "should have at least 1 NUMA node");
    }

    #[tokio::test]
    async fn test_topology_has_cache_info() {
        let analyzer = TopologyAnalyzer::new().await.expect("failed to create analyzer");
        let results = analyzer.analyze_complete_topology().await.expect("analysis failed");
        assert!(
            !results.cache_analysis.cache_levels.is_empty(),
            "should detect cache levels"
        );
    }

    #[tokio::test]
    async fn test_topology_cache_levels_ordered() {
        let analyzer = TopologyAnalyzer::new().await.expect("failed to create analyzer");
        let results = analyzer.analyze_complete_topology().await.expect("analysis failed");
        let levels = &results.cache_analysis.cache_levels;
        for window in levels.windows(2) {
            assert!(
                window[0].level <= window[1].level,
                "cache levels should be ordered"
            );
        }
    }

    #[tokio::test]
    async fn test_topology_has_memory_info() {
        let analyzer = TopologyAnalyzer::new().await.expect("failed to create analyzer");
        let results = analyzer.analyze_complete_topology().await.expect("analysis failed");
        assert!(results.memory_topology.memory_channels >= 1);
    }

    #[tokio::test]
    async fn test_topology_has_metadata() {
        let analyzer = TopologyAnalyzer::new().await.expect("failed to create analyzer");
        let results = analyzer.analyze_complete_topology().await.expect("analysis failed");
        assert!(results.analysis_metadata.analysis_duration.as_nanos() > 0);
    }

    #[tokio::test]
    async fn test_topology_with_debug_mode() {
        let mut analyzer = TopologyAnalyzer::new().await.expect("failed to create analyzer");
        analyzer.enable_debug_mode();
        assert!(analyzer.debug_mode);
        let result = analyzer.analyze_complete_topology().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_topology_with_disabled_numa() {
        let config = TopologyAnalysisConfig {
            enable_numa_detection: false,
            ..Default::default()
        };
        let analyzer = TopologyAnalyzer::with_config(config).await.expect("failed to create");
        let results = analyzer.analyze_complete_topology().await.expect("analysis failed");
        assert!(results.numa_topology.is_none());
    }

    #[test]
    fn test_parse_cache_size_kilobytes() {
        assert_eq!(parse_cache_size("32K"), 32 * 1024);
        assert_eq!(parse_cache_size("256K"), 256 * 1024);
    }

    #[test]
    fn test_parse_cache_size_megabytes() {
        assert_eq!(parse_cache_size("8M"), 8 * 1024 * 1024);
    }

    #[test]
    fn test_parse_cache_size_empty() {
        assert_eq!(parse_cache_size(""), 0);
    }

    #[test]
    fn test_detect_cache_hierarchy_not_empty() {
        let result = detect_cache_hierarchy();
        assert!(
            !result.cache_levels.is_empty(),
            "should always return cache levels"
        );
    }

    #[test]
    fn test_detect_memory_topology() {
        let result = detect_memory_topology();
        assert!(result.memory_channels >= 1);
    }
}
