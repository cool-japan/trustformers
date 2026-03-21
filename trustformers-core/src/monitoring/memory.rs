// Memory profiling and tracking utilities
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Memory tracker for monitoring GPU and CPU memory usage
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    config: MemoryTrackerConfig,
    active_sessions: HashMap<String, MemorySession>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrackerConfig {
    pub track_cpu_memory: bool,
    pub track_gpu_memory: bool,
    pub snapshot_interval_ms: u64,
    pub max_snapshots: usize,
    pub track_allocations: bool,
}

impl Default for MemoryTrackerConfig {
    fn default() -> Self {
        Self {
            track_cpu_memory: true,
            track_gpu_memory: true,
            snapshot_interval_ms: 100,
            max_snapshots: 1000,
            track_allocations: false, // Expensive, disabled by default
        }
    }
}

#[derive(Debug, Clone)]
struct MemorySession {
    id: String,
    start_time: Instant,
    snapshots: Vec<MemorySnapshot>,
    peak_usage: MemoryUsage,
}

/// Memory usage snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: String,
    pub checkpoint: String,
    pub cpu_usage: MemoryUsage,
    pub gpu_usage: MemoryUsage,
    pub allocations: Vec<AllocationInfo>,
}

/// Memory usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub allocated_bytes: u64,
    pub peak_bytes: u64,
    pub available_bytes: u64,
    pub fragmentation_ratio: f64,
}

/// Information about a memory allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationInfo {
    pub address: u64,
    pub size_bytes: u64,
    pub allocation_type: AllocationType,
    pub stack_trace: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationType {
    Tensor,
    Weight,
    Activation,
    Gradient,
    Buffer,
    Other(String),
}

/// Complete memory report for a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReport {
    pub session_id: String,
    pub duration: Duration,
    pub initial_snapshot: MemorySnapshot,
    pub final_snapshot: MemorySnapshot,
    pub peak_usage: MemoryUsage,
    pub snapshots: Vec<MemorySnapshot>,
    pub allocation_summary: AllocationSummary,
    pub memory_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationSummary {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub peak_active_allocations: usize,
    pub allocation_type_breakdown: HashMap<String, u64>,
    pub largest_allocation: u64,
    pub average_allocation_size: f64,
}

impl Default for MemorySnapshot {
    fn default() -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            checkpoint: "default".to_string(),
            cpu_usage: MemoryUsage::default(),
            gpu_usage: MemoryUsage::default(),
            allocations: Vec::new(),
        }
    }
}

impl Default for MemoryUsage {
    fn default() -> Self {
        Self {
            allocated_bytes: 0,
            peak_bytes: 0,
            available_bytes: 0,
            fragmentation_ratio: 0.0,
        }
    }
}

impl Default for MemoryReport {
    fn default() -> Self {
        Self {
            session_id: String::new(),
            duration: Duration::from_secs(0),
            initial_snapshot: MemorySnapshot::default(),
            final_snapshot: MemorySnapshot::default(),
            peak_usage: MemoryUsage::default(),
            snapshots: Vec::new(),
            allocation_summary: AllocationSummary::default(),
            memory_efficiency: 0.0,
        }
    }
}

impl Default for AllocationSummary {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            peak_active_allocations: 0,
            allocation_type_breakdown: HashMap::new(),
            largest_allocation: 0,
            average_allocation_size: 0.0,
        }
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            config: MemoryTrackerConfig::default(),
            active_sessions: HashMap::new(),
        }
    }

    pub fn with_config(config: MemoryTrackerConfig) -> Self {
        Self {
            config,
            active_sessions: HashMap::new(),
        }
    }

    /// Start tracking memory for a session
    pub fn start_tracking(&mut self, session_id: &str) -> Result<()> {
        let initial_snapshot = self.take_current_snapshot("session_start")?;

        let session = MemorySession {
            id: session_id.to_string(),
            start_time: Instant::now(),
            snapshots: vec![initial_snapshot.clone()],
            peak_usage: initial_snapshot.cpu_usage.clone(),
        };

        self.active_sessions.insert(session_id.to_string(), session);
        Ok(())
    }

    /// Take a memory snapshot at a specific checkpoint
    pub fn take_snapshot(&mut self, session_id: &str, checkpoint: &str) -> Result<MemorySnapshot> {
        let snapshot = self.take_current_snapshot(checkpoint)?;

        if let Some(session) = self.active_sessions.get_mut(session_id) {
            // Update peak usage
            if snapshot.cpu_usage.allocated_bytes > session.peak_usage.allocated_bytes {
                session.peak_usage = snapshot.cpu_usage.clone();
            }

            // Add snapshot if under limit
            if session.snapshots.len() < self.config.max_snapshots {
                session.snapshots.push(snapshot.clone());
            }
        }

        Ok(snapshot)
    }

    /// End tracking and generate report
    pub fn end_tracking(&mut self, session_id: &str) -> Result<MemoryReport> {
        let session = self
            .active_sessions
            .remove(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        let final_snapshot = self.take_current_snapshot("session_end")?;
        let duration = session.start_time.elapsed();

        let initial_snapshot =
            session.snapshots.first().cloned().unwrap_or_else(MemorySnapshot::default);

        let allocation_summary = self.compute_allocation_summary(&session.snapshots);
        let memory_efficiency = self.compute_memory_efficiency(&session.snapshots);

        Ok(MemoryReport {
            session_id: session.id,
            duration,
            initial_snapshot,
            final_snapshot,
            peak_usage: session.peak_usage,
            snapshots: session.snapshots,
            allocation_summary,
            memory_efficiency,
        })
    }

    /// Clear all tracking data
    pub fn clear(&mut self) -> Result<()> {
        self.active_sessions.clear();
        Ok(())
    }

    /// Take a current memory snapshot
    fn take_current_snapshot(&self, checkpoint: &str) -> Result<MemorySnapshot> {
        let cpu_usage = if self.config.track_cpu_memory {
            self.get_cpu_memory_usage()?
        } else {
            MemoryUsage::default()
        };

        let gpu_usage = if self.config.track_gpu_memory {
            self.get_gpu_memory_usage()?
        } else {
            MemoryUsage::default()
        };

        let allocations = if self.config.track_allocations {
            self.get_current_allocations()?
        } else {
            Vec::new()
        };

        Ok(MemorySnapshot {
            timestamp: chrono::Utc::now().to_rfc3339(),
            checkpoint: checkpoint.to_string(),
            cpu_usage,
            gpu_usage,
            allocations,
        })
    }

    /// Get current CPU memory usage
    fn get_cpu_memory_usage(&self) -> Result<MemoryUsage> {
        // Simplified implementation - in practice would use system APIs
        #[cfg(target_os = "linux")]
        {
            self.get_linux_memory_info()
        }
        #[cfg(target_os = "macos")]
        {
            self.get_macos_memory_info()
        }
        #[cfg(target_os = "windows")]
        {
            self.get_windows_memory_info()
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            Ok(MemoryUsage {
                allocated_bytes: 0,
                peak_bytes: 0,
                available_bytes: 8 * 1024 * 1024 * 1024, // 8GB default
                fragmentation_ratio: 0.0,
            })
        }
    }

    #[cfg(target_os = "linux")]
    fn get_linux_memory_info(&self) -> Result<MemoryUsage> {
        // Read from /proc/meminfo
        let meminfo = std::fs::read_to_string("/proc/meminfo")?;
        let mut mem_total = 0;
        let mut mem_available = 0;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                mem_total =
                    line.split_whitespace().nth(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0)
                        * 1024; // Convert from KB to bytes
            } else if line.starts_with("MemAvailable:") {
                mem_available =
                    line.split_whitespace().nth(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0)
                        * 1024; // Convert from KB to bytes
            }
        }

        let allocated_bytes = mem_total.saturating_sub(mem_available);

        Ok(MemoryUsage {
            allocated_bytes,
            peak_bytes: allocated_bytes, // Simplified
            available_bytes: mem_available,
            fragmentation_ratio: 0.0, // Would require more detailed analysis
        })
    }

    #[cfg(target_os = "macos")]
    fn get_macos_memory_info(&self) -> Result<MemoryUsage> {
        // Simplified implementation for macOS
        Ok(MemoryUsage {
            allocated_bytes: 4 * 1024 * 1024 * 1024, // 4GB estimated
            peak_bytes: 4 * 1024 * 1024 * 1024,
            available_bytes: 4 * 1024 * 1024 * 1024,
            fragmentation_ratio: 0.1,
        })
    }

    #[cfg(target_os = "windows")]
    fn get_windows_memory_info(&self) -> Result<MemoryUsage> {
        // Simplified implementation for Windows
        Ok(MemoryUsage {
            allocated_bytes: 4 * 1024 * 1024 * 1024, // 4GB estimated
            peak_bytes: 4 * 1024 * 1024 * 1024,
            available_bytes: 4 * 1024 * 1024 * 1024,
            fragmentation_ratio: 0.1,
        })
    }

    /// Get current GPU memory usage
    fn get_gpu_memory_usage(&self) -> Result<MemoryUsage> {
        // Placeholder implementation - would integrate with CUDA/ROCm APIs
        Ok(MemoryUsage {
            allocated_bytes: 2 * 1024 * 1024 * 1024, // 2GB estimated
            peak_bytes: 2 * 1024 * 1024 * 1024,
            available_bytes: 6 * 1024 * 1024 * 1024, // 6GB available
            fragmentation_ratio: 0.05,
        })
    }

    /// Get current memory allocations
    fn get_current_allocations(&self) -> Result<Vec<AllocationInfo>> {
        // Placeholder implementation - would require custom allocator integration
        Ok(vec![
            AllocationInfo {
                address: 0x1000000,
                size_bytes: 1024 * 1024, // 1MB
                allocation_type: AllocationType::Tensor,
                stack_trace: vec!["tensor_alloc".to_string()],
            },
            AllocationInfo {
                address: 0x2000000,
                size_bytes: 512 * 1024, // 512KB
                allocation_type: AllocationType::Weight,
                stack_trace: vec!["weight_alloc".to_string()],
            },
        ])
    }

    /// Compute allocation summary from snapshots
    fn compute_allocation_summary(&self, snapshots: &[MemorySnapshot]) -> AllocationSummary {
        let mut total_allocations = 0;
        let mut type_breakdown = HashMap::new();
        let mut largest_allocation = 0;
        let mut total_size = 0;

        for snapshot in snapshots {
            total_allocations += snapshot.allocations.len();

            for allocation in &snapshot.allocations {
                total_size += allocation.size_bytes;
                largest_allocation = largest_allocation.max(allocation.size_bytes);

                let type_name = match &allocation.allocation_type {
                    AllocationType::Tensor => "Tensor",
                    AllocationType::Weight => "Weight",
                    AllocationType::Activation => "Activation",
                    AllocationType::Gradient => "Gradient",
                    AllocationType::Buffer => "Buffer",
                    AllocationType::Other(name) => name,
                }
                .to_string();

                *type_breakdown.entry(type_name).or_insert(0) += allocation.size_bytes;
            }
        }

        let average_allocation_size = if total_allocations > 0 {
            total_size as f64 / total_allocations as f64
        } else {
            0.0
        };

        AllocationSummary {
            total_allocations,
            total_deallocations: 0, // Would track in real implementation
            peak_active_allocations: total_allocations,
            allocation_type_breakdown: type_breakdown,
            largest_allocation,
            average_allocation_size,
        }
    }

    /// Compute memory efficiency metric
    fn compute_memory_efficiency(&self, snapshots: &[MemorySnapshot]) -> f64 {
        if snapshots.is_empty() {
            return 0.0;
        }

        let mut total_efficiency = 0.0;
        let mut count = 0;

        for snapshot in snapshots {
            let cpu_total = snapshot.cpu_usage.allocated_bytes + snapshot.cpu_usage.available_bytes;
            if cpu_total > 0 {
                let efficiency = snapshot.cpu_usage.allocated_bytes as f64 / cpu_total as f64;
                total_efficiency += efficiency;
                count += 1;
            }
        }

        if count > 0 {
            total_efficiency / count as f64
        } else {
            0.0
        }
    }
}

impl MemoryReport {
    /// Print a summary of the memory report
    pub fn print_summary(&self) {
        println!("Memory Report Summary");
        println!("====================");
        println!("Session Duration: {:.2}ms", self.duration.as_millis());
        println!(
            "Peak CPU Usage: {} MB",
            self.peak_usage.allocated_bytes / (1024 * 1024)
        );
        println!("Memory Efficiency: {:.2}%", self.memory_efficiency * 100.0);
        println!("Total Snapshots: {}", self.snapshots.len());

        if !self.allocation_summary.allocation_type_breakdown.is_empty() {
            println!("Allocation Breakdown:");
            for (type_name, size) in &self.allocation_summary.allocation_type_breakdown {
                println!("  {}: {} MB", type_name, size / (1024 * 1024));
            }
        }
    }

    /// Get memory growth rate
    pub fn memory_growth_rate(&self) -> f64 {
        if self.initial_snapshot.cpu_usage.allocated_bytes == 0 {
            return 0.0;
        }

        let initial = self.initial_snapshot.cpu_usage.allocated_bytes as f64;
        let final_mem = self.final_snapshot.cpu_usage.allocated_bytes as f64;

        (final_mem - initial) / initial
    }

    /// Check for memory leaks
    pub fn has_potential_memory_leak(&self, threshold: f64) -> bool {
        self.memory_growth_rate() > threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker_creation() {
        let tracker = MemoryTracker::new();
        assert!(tracker.config.track_cpu_memory);
        assert!(tracker.config.track_gpu_memory);
    }

    #[test]
    fn test_memory_tracking_session() -> Result<()> {
        let mut tracker = MemoryTracker::new();

        let session_id = "test_session";
        tracker.start_tracking(session_id)?;

        // Take a snapshot
        let snapshot = tracker.take_snapshot(session_id, "mid_point")?;
        assert_eq!(snapshot.checkpoint, "mid_point");

        // End tracking
        let report = tracker.end_tracking(session_id)?;
        assert_eq!(report.session_id, session_id);
        assert!(report.snapshots.len() >= 2); // At least start and mid_point

        Ok(())
    }

    #[test]
    fn test_memory_usage_default() {
        let usage = MemoryUsage::default();
        assert_eq!(usage.allocated_bytes, 0);
        assert_eq!(usage.peak_bytes, 0);
    }

    #[test]
    fn test_allocation_info() {
        let allocation = AllocationInfo {
            address: 0x1000,
            size_bytes: 1024,
            allocation_type: AllocationType::Tensor,
            stack_trace: vec!["test".to_string()],
        };

        assert_eq!(allocation.size_bytes, 1024);
        assert!(matches!(allocation.allocation_type, AllocationType::Tensor));
    }

    #[test]
    fn test_memory_report_growth_rate() {
        let mut report = MemoryReport::default();
        report.initial_snapshot.cpu_usage.allocated_bytes = 1000;
        report.final_snapshot.cpu_usage.allocated_bytes = 1500;

        let growth_rate = report.memory_growth_rate();
        assert!((growth_rate - 0.5).abs() < 0.001); // 50% growth
    }

    #[test]
    fn test_memory_leak_detection() {
        let mut report = MemoryReport::default();
        report.initial_snapshot.cpu_usage.allocated_bytes = 1000;
        report.final_snapshot.cpu_usage.allocated_bytes = 2000;

        assert!(report.has_potential_memory_leak(0.5)); // 100% growth > 50% threshold
        assert!(!report.has_potential_memory_leak(1.5)); // 100% growth < 150% threshold
    }

    #[test]
    fn test_memory_tracker_config() {
        let config = MemoryTrackerConfig {
            track_cpu_memory: false,
            track_gpu_memory: true,
            snapshot_interval_ms: 50,
            max_snapshots: 500,
            track_allocations: true,
        };

        let tracker = MemoryTracker::with_config(config.clone());
        assert!(!tracker.config.track_cpu_memory);
        assert!(tracker.config.track_gpu_memory);
        assert_eq!(tracker.config.max_snapshots, 500);
    }
}
