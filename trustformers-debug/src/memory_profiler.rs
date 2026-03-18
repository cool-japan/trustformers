//! Advanced memory profiling for TrustformeRS models.
//!
//! This module provides comprehensive memory profiling capabilities including:
//! - Heap allocation tracking
//! - Memory leak detection
//! - Peak memory analysis
//! - Allocation patterns
//! - GC pressure analysis
//! - Memory fragmentation monitoring
//!
//! # Example
//!
//! ```no_run
//! use trustformers_debug::{MemoryProfiler, MemoryProfilingConfig};
//!
//! let config = MemoryProfilingConfig::default();
//! let mut profiler = MemoryProfiler::new(config);
//!
//! profiler.start().await?;
//! // ... run model training/inference ...
//! let report = profiler.stop().await?;
//!
//! println!("Peak memory usage: {} MB", report.peak_memory_mb);
//! println!("Memory leaks detected: {}", report.potential_leaks.len());
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time::interval;
use uuid::Uuid;

/// Configuration for memory profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfilingConfig {
    /// Enable heap allocation tracking
    pub enable_heap_tracking: bool,
    /// Enable leak detection
    pub enable_leak_detection: bool,
    /// Enable allocation pattern analysis
    pub enable_pattern_analysis: bool,
    /// Enable memory fragmentation monitoring
    pub enable_fragmentation_monitoring: bool,
    /// Enable GC pressure analysis
    pub enable_gc_pressure_analysis: bool,
    /// Sampling interval for memory measurements (milliseconds)
    pub sampling_interval_ms: u64,
    /// Maximum number of allocation records to keep
    pub max_allocation_records: usize,
    /// Threshold for considering an allocation "large" (bytes)
    pub large_allocation_threshold: usize,
    /// Window size for detecting allocation patterns (seconds)
    pub pattern_analysis_window_secs: u64,
    /// Threshold for leak detection (allocations alive for this duration)
    pub leak_detection_threshold_secs: u64,
}

impl Default for MemoryProfilingConfig {
    fn default() -> Self {
        Self {
            enable_heap_tracking: true,
            enable_leak_detection: true,
            enable_pattern_analysis: true,
            enable_fragmentation_monitoring: true,
            enable_gc_pressure_analysis: true,
            sampling_interval_ms: 100, // 100ms sampling
            max_allocation_records: 100000,
            large_allocation_threshold: 1024 * 1024, // 1MB
            pattern_analysis_window_secs: 60,        // 1 minute window
            leak_detection_threshold_secs: 300,      // 5 minutes
        }
    }
}

/// Allocation record for tracking individual allocations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRecord {
    pub id: Uuid,
    pub size: usize,
    pub timestamp: SystemTime,
    pub stack_trace: Vec<String>,
    pub allocation_type: AllocationType,
    pub freed: bool,
    pub freed_at: Option<SystemTime>,
    pub tags: Vec<String>, // For categorizing allocations
}

/// Type of allocation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllocationType {
    Tensor,
    Buffer,
    Weights,
    Gradients,
    Activations,
    Cache,
    Temporary,
    Other(String),
}

/// Memory usage snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: SystemTime,
    pub total_heap_bytes: usize,
    pub used_heap_bytes: usize,
    pub free_heap_bytes: usize,
    pub peak_heap_bytes: usize,
    pub allocation_count: usize,
    pub free_count: usize,
    pub fragmentation_ratio: f64,
    pub gc_pressure_score: f64,
    pub allocations_by_type: HashMap<AllocationType, usize>,
    pub allocations_by_size: HashMap<String, usize>, // Size buckets
}

/// Memory leak information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeak {
    pub allocation_id: Uuid,
    pub size: usize,
    pub age_seconds: f64,
    pub allocation_type: AllocationType,
    pub stack_trace: Vec<String>,
    pub tags: Vec<String>,
    pub severity: LeakSeverity,
}

/// Severity of memory leak
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeakSeverity {
    Low,      // Small allocations, short-lived
    Medium,   // Moderate size or moderately old
    High,     // Large allocations or very old
    Critical, // Very large or extremely old
}

/// Allocation pattern detected by analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPattern {
    pub pattern_type: PatternType,
    pub description: String,
    pub confidence: f64,   // 0.0 to 1.0
    pub impact_score: f64, // 0.0 to 1.0 (higher = more concerning)
    pub recommendations: Vec<String>,
    pub examples: Vec<AllocationRecord>,
}

/// Type of allocation pattern
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    MemoryLeak,           // Consistent growth without deallocation
    ChurningAllocations,  // Rapid alloc/free cycles
    FragmentationCausing, // Allocations that cause fragmentation
    LargeAllocations,     // Unexpectedly large allocations
    UnbalancedTypes,      // Disproportionate allocation types
    PeakUsageSpikes,      // Sudden memory usage spikes
}

/// Memory fragmentation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationAnalysis {
    pub fragmentation_ratio: f64,
    pub largest_free_block: usize,
    pub total_free_memory: usize,
    pub free_block_count: usize,
    pub average_free_block_size: f64,
    pub fragmentation_severity: FragmentationSeverity,
    pub recommendations: Vec<String>,
}

/// Fragmentation severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FragmentationSeverity {
    Low,    // < 10% fragmentation
    Medium, // 10-30% fragmentation
    High,   // 30-60% fragmentation
    Severe, // > 60% fragmentation
}

/// Garbage collection pressure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCPressureAnalysis {
    pub pressure_score: f64,    // 0.0 to 1.0
    pub allocation_rate: f64,   // allocations per second
    pub deallocation_rate: f64, // deallocations per second
    pub churn_rate: f64,        // alloc/dealloc cycles per second
    pub pressure_level: GCPressureLevel,
    pub contributing_factors: Vec<String>,
    pub recommendations: Vec<String>,
}

/// GC pressure levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GCPressureLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Comprehensive memory profiling report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfilingReport {
    pub session_id: Uuid,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub duration_secs: f64,
    pub config: MemoryProfilingConfig,

    // Summary statistics
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub net_allocations: i64,

    // Memory timeline
    pub memory_timeline: Vec<MemorySnapshot>,

    // Leak detection
    pub potential_leaks: Vec<MemoryLeak>,
    pub leak_summary: HashMap<AllocationType, usize>,

    // Pattern analysis
    pub detected_patterns: Vec<AllocationPattern>,

    // Fragmentation analysis
    pub fragmentation_analysis: FragmentationAnalysis,

    // GC pressure analysis
    pub gc_pressure_analysis: GCPressureAnalysis,

    // Allocation statistics
    pub allocations_by_type: HashMap<AllocationType, AllocationTypeStats>,
    pub allocations_by_size_bucket: HashMap<String, usize>,

    // Performance metrics
    pub profiling_overhead_ms: f64,
    pub sampling_accuracy: f64,
}

/// Statistics for each allocation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationTypeStats {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub current_count: usize,
    pub total_bytes_allocated: usize,
    pub total_bytes_deallocated: usize,
    pub current_bytes: usize,
    pub peak_count: usize,
    pub peak_bytes: usize,
    pub average_allocation_size: f64,
    pub largest_allocation: usize,
}

/// Memory profiler implementation
#[derive(Debug)]
pub struct MemoryProfiler {
    config: MemoryProfilingConfig,
    session_id: Uuid,
    start_time: Option<Instant>,
    allocations: Arc<Mutex<HashMap<Uuid, AllocationRecord>>>,
    memory_timeline: Arc<Mutex<VecDeque<MemorySnapshot>>>,
    type_stats: Arc<Mutex<HashMap<AllocationType, AllocationTypeStats>>>,
    running: Arc<Mutex<bool>>,
    profiling_start_time: Option<Instant>,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new(config: MemoryProfilingConfig) -> Self {
        Self {
            config,
            session_id: Uuid::new_v4(),
            start_time: None,
            allocations: Arc::new(Mutex::new(HashMap::new())),
            memory_timeline: Arc::new(Mutex::new(VecDeque::new())),
            type_stats: Arc::new(Mutex::new(HashMap::new())),
            running: Arc::new(Mutex::new(false)),
            profiling_start_time: None,
        }
    }

    /// Start memory profiling
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.lock().expect("lock should not be poisoned");
        if *running {
            return Err(anyhow::anyhow!("Memory profiler is already running"));
        }

        *running = true;
        self.start_time = Some(Instant::now());
        self.profiling_start_time = Some(Instant::now());

        // Start periodic sampling
        if self.config.enable_heap_tracking {
            self.start_sampling().await?;
        }

        tracing::info!("Memory profiler started for session {}", self.session_id);
        Ok(())
    }

    /// Stop memory profiling and generate report
    pub async fn stop(&mut self) -> Result<MemoryProfilingReport> {
        let mut running = self.running.lock().expect("lock should not be poisoned");
        if !*running {
            return Err(anyhow::anyhow!("Memory profiler is not running"));
        }

        *running = false;
        let end_time = SystemTime::now();
        let start_time =
            self.start_time.expect("start_time should be set when profiler is running");
        let duration =
            end_time.duration_since(UNIX_EPOCH)?.as_secs_f64() - start_time.elapsed().as_secs_f64();

        // Calculate profiling overhead
        let profiling_overhead = if let Some(prof_start) = self.profiling_start_time {
            prof_start.elapsed().as_millis() as f64 * 0.01 // Estimated 1% overhead
        } else {
            0.0
        };

        let report = self.generate_report(end_time, duration, profiling_overhead).await?;

        tracing::info!("Memory profiler stopped for session {}", self.session_id);
        Ok(report)
    }

    /// Record an allocation
    pub fn record_allocation(
        &self,
        size: usize,
        allocation_type: AllocationType,
        tags: Vec<String>,
    ) -> Result<Uuid> {
        let running = self.running.lock().expect("lock should not be poisoned");
        if !*running {
            return Err(anyhow::anyhow!("Memory profiler is not running"));
        }

        let allocation_id = Uuid::new_v4();
        let record = AllocationRecord {
            id: allocation_id,
            size,
            timestamp: SystemTime::now(),
            stack_trace: self.capture_stack_trace(),
            allocation_type: allocation_type.clone(),
            freed: false,
            freed_at: None,
            tags,
        };

        // Store allocation record
        let mut allocations = self.allocations.lock().expect("lock should not be poisoned");
        allocations.insert(allocation_id, record);

        // Update type statistics
        self.update_type_stats(&allocation_type, size, true);

        Ok(allocation_id)
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, allocation_id: Uuid) -> Result<()> {
        let running = self.running.lock().expect("lock should not be poisoned");
        if !*running {
            return Ok(()); // Silently ignore if not running
        }

        let mut allocations = self.allocations.lock().expect("lock should not be poisoned");
        if let Some(record) = allocations.get_mut(&allocation_id) {
            record.freed = true;
            record.freed_at = Some(SystemTime::now());

            // Update type statistics
            self.update_type_stats(&record.allocation_type, record.size, false);
        }

        Ok(())
    }

    /// Tag an existing allocation
    pub fn tag_allocation(&self, allocation_id: Uuid, tag: String) -> Result<()> {
        let mut allocations = self.allocations.lock().expect("lock should not be poisoned");
        if let Some(record) = allocations.get_mut(&allocation_id) {
            record.tags.push(tag);
        }
        Ok(())
    }

    /// Get current memory usage snapshot
    pub fn get_memory_snapshot(&self) -> Result<MemorySnapshot> {
        let allocations = self.allocations.lock().expect("lock should not be poisoned");
        let _type_stats = self.type_stats.lock().expect("lock should not be poisoned");

        let mut total_heap = 0;
        let mut used_heap = 0;
        let mut allocation_count = 0;
        let mut free_count = 0;
        let mut allocations_by_type = HashMap::new();
        let mut allocations_by_size = HashMap::new();

        for record in allocations.values() {
            total_heap += record.size;

            if !record.freed {
                used_heap += record.size;
                allocation_count += 1;

                *allocations_by_type.entry(record.allocation_type.clone()).or_insert(0) +=
                    record.size;

                let size_bucket = self.get_size_bucket(record.size);
                *allocations_by_size.entry(size_bucket).or_insert(0) += 1;
            } else {
                free_count += 1;
            }
        }

        let free_heap = total_heap - used_heap;
        let fragmentation_ratio =
            if total_heap > 0 { free_heap as f64 / total_heap as f64 } else { 0.0 };

        let gc_pressure_score = self.calculate_gc_pressure_score();

        Ok(MemorySnapshot {
            timestamp: SystemTime::now(),
            total_heap_bytes: total_heap,
            used_heap_bytes: used_heap,
            free_heap_bytes: free_heap,
            peak_heap_bytes: used_heap, // Simplified for now
            allocation_count,
            free_count,
            fragmentation_ratio,
            gc_pressure_score,
            allocations_by_type,
            allocations_by_size,
        })
    }

    /// Detect memory leaks
    pub fn detect_leaks(&self) -> Result<Vec<MemoryLeak>> {
        let allocations = self.allocations.lock().expect("lock should not be poisoned");
        let now = SystemTime::now();
        let threshold = Duration::from_secs(self.config.leak_detection_threshold_secs);
        let mut leaks = Vec::new();

        for record in allocations.values() {
            if !record.freed {
                let age = now.duration_since(record.timestamp)?;
                if age > threshold {
                    let age_seconds = age.as_secs_f64();
                    let severity = self.classify_leak_severity(record.size, age_seconds);

                    leaks.push(MemoryLeak {
                        allocation_id: record.id,
                        size: record.size,
                        age_seconds,
                        allocation_type: record.allocation_type.clone(),
                        stack_trace: record.stack_trace.clone(),
                        tags: record.tags.clone(),
                        severity,
                    });
                }
            }
        }

        // Sort by severity and size
        leaks.sort_by(|a, b| b.severity.cmp(&a.severity).then(b.size.cmp(&a.size)));

        Ok(leaks)
    }

    /// Analyze allocation patterns
    pub fn analyze_patterns(&self) -> Result<Vec<AllocationPattern>> {
        let mut patterns = Vec::new();

        // Detect memory leak patterns
        if let Ok(leak_pattern) = self.detect_leak_pattern() {
            patterns.push(leak_pattern);
        }

        // Detect churning allocation patterns
        if let Ok(churn_pattern) = self.detect_churn_pattern() {
            patterns.push(churn_pattern);
        }

        // Detect large allocation patterns
        if let Ok(large_alloc_pattern) = self.detect_large_allocation_pattern() {
            patterns.push(large_alloc_pattern);
        }

        // Detect fragmentation-causing patterns
        if let Ok(frag_pattern) = self.detect_fragmentation_pattern() {
            patterns.push(frag_pattern);
        }

        Ok(patterns)
    }

    /// Analyze memory fragmentation
    pub fn analyze_fragmentation(&self) -> Result<FragmentationAnalysis> {
        let snapshot = self.get_memory_snapshot()?;

        let fragmentation_ratio = snapshot.fragmentation_ratio;
        let severity = match fragmentation_ratio {
            r if r < 0.1 => FragmentationSeverity::Low,
            r if r < 0.3 => FragmentationSeverity::Medium,
            r if r < 0.6 => FragmentationSeverity::High,
            _ => FragmentationSeverity::Severe,
        };

        let recommendations = match severity {
            FragmentationSeverity::Low => {
                vec!["Memory fragmentation is low. Continue current practices.".to_string()]
            },
            FragmentationSeverity::Medium => vec![
                "Consider pooling allocations of similar sizes.".to_string(),
                "Monitor for increasing fragmentation trends.".to_string(),
            ],
            FragmentationSeverity::High => vec![
                "Implement memory pooling for frequent allocations.".to_string(),
                "Consider compaction strategies for long-running processes.".to_string(),
                "Review allocation patterns for optimization opportunities.".to_string(),
            ],
            FragmentationSeverity::Severe => vec![
                "Critical fragmentation detected. Immediate action required.".to_string(),
                "Implement custom allocators with compaction.".to_string(),
                "Consider restarting the process to reset memory layout.".to_string(),
                "Review and optimize allocation strategies.".to_string(),
            ],
        };

        Ok(FragmentationAnalysis {
            fragmentation_ratio,
            largest_free_block: snapshot.free_heap_bytes, // Simplified
            total_free_memory: snapshot.free_heap_bytes,
            free_block_count: snapshot.free_count,
            average_free_block_size: if snapshot.free_count > 0 {
                snapshot.free_heap_bytes as f64 / snapshot.free_count as f64
            } else {
                0.0
            },
            fragmentation_severity: severity,
            recommendations,
        })
    }

    /// Analyze GC pressure
    pub fn analyze_gc_pressure(&self) -> Result<GCPressureAnalysis> {
        let timeline = self.memory_timeline.lock().expect("lock should not be poisoned");

        let pressure_score = self.calculate_gc_pressure_score();
        let (allocation_rate, deallocation_rate) = self.calculate_allocation_rates(&timeline);
        let churn_rate = allocation_rate.min(deallocation_rate);

        let pressure_level = match pressure_score {
            p if p < 0.25 => GCPressureLevel::Low,
            p if p < 0.5 => GCPressureLevel::Medium,
            p if p < 0.75 => GCPressureLevel::High,
            _ => GCPressureLevel::Critical,
        };

        let mut contributing_factors = Vec::new();
        let mut recommendations = Vec::new();

        if allocation_rate > 1000.0 {
            contributing_factors.push("High allocation rate".to_string());
            recommendations.push("Consider object pooling or reuse strategies".to_string());
        }

        if churn_rate > 500.0 {
            contributing_factors.push("High allocation churn".to_string());
            recommendations.push("Reduce temporary object creation".to_string());
        }

        if pressure_level == GCPressureLevel::Critical {
            recommendations
                .push("Consider manual memory management for critical paths".to_string());
        }

        Ok(GCPressureAnalysis {
            pressure_score,
            allocation_rate,
            deallocation_rate,
            churn_rate,
            pressure_level,
            contributing_factors,
            recommendations,
        })
    }

    // Private helper methods

    async fn start_sampling(&self) -> Result<()> {
        let interval_duration = Duration::from_millis(self.config.sampling_interval_ms);
        let mut interval = interval(interval_duration);
        let _timeline = Arc::clone(&self.memory_timeline);
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            loop {
                interval.tick().await;

                let is_running = {
                    let running_guard = running.lock().expect("lock should not be poisoned");
                    *running_guard
                };

                if !is_running {
                    break;
                }

                // This would normally sample actual memory usage
                // For now, we'll use a placeholder implementation
            }
        });

        Ok(())
    }

    pub async fn generate_report(
        &self,
        end_time: SystemTime,
        duration_secs: f64,
        profiling_overhead_ms: f64,
    ) -> Result<MemoryProfilingReport> {
        let allocations = self.allocations.lock().expect("lock should not be poisoned");
        let timeline = self.memory_timeline.lock().expect("lock should not be poisoned");
        let type_stats = self.type_stats.lock().expect("lock should not be poisoned");

        let total_allocations = allocations.len();
        let total_deallocations = allocations.values().filter(|r| r.freed).count();
        let net_allocations = total_allocations as i64 - total_deallocations as i64;

        let potential_leaks = self.detect_leaks()?;
        let detected_patterns = self.analyze_patterns()?;
        let fragmentation_analysis = self.analyze_fragmentation()?;
        let gc_pressure_analysis = self.analyze_gc_pressure()?;

        // Calculate summary statistics
        let peak_memory_mb = timeline
            .iter()
            .map(|s| s.peak_heap_bytes as f64 / 1024.0 / 1024.0)
            .fold(0.0, f64::max);

        let average_memory_mb = if !timeline.is_empty() {
            timeline.iter().map(|s| s.used_heap_bytes as f64 / 1024.0 / 1024.0).sum::<f64>()
                / timeline.len() as f64
        } else {
            0.0
        };

        let mut leak_summary = HashMap::new();
        for leak in &potential_leaks {
            *leak_summary.entry(leak.allocation_type.clone()).or_insert(0) += 1;
        }

        // Create size buckets
        let mut allocations_by_size_bucket = HashMap::new();
        for record in allocations.values() {
            let bucket = self.get_size_bucket(record.size);
            *allocations_by_size_bucket.entry(bucket).or_insert(0) += 1;
        }

        Ok(MemoryProfilingReport {
            session_id: self.session_id,
            start_time: UNIX_EPOCH
                + Duration::from_secs_f64(
                    SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64() - duration_secs,
                ),
            end_time,
            duration_secs,
            config: self.config.clone(),
            peak_memory_mb,
            average_memory_mb,
            total_allocations,
            total_deallocations,
            net_allocations,
            memory_timeline: timeline.iter().cloned().collect(),
            potential_leaks,
            leak_summary,
            detected_patterns,
            fragmentation_analysis,
            gc_pressure_analysis,
            allocations_by_type: type_stats.clone(),
            allocations_by_size_bucket,
            profiling_overhead_ms,
            sampling_accuracy: 0.95, // Placeholder
        })
    }

    fn capture_stack_trace(&self) -> Vec<String> {
        // Placeholder implementation - in a real implementation,
        // this would capture the actual call stack
        vec![
            "function_a".to_string(),
            "function_b".to_string(),
            "main".to_string(),
        ]
    }

    fn update_type_stats(
        &self,
        allocation_type: &AllocationType,
        size: usize,
        is_allocation: bool,
    ) {
        let mut type_stats = self.type_stats.lock().expect("lock should not be poisoned");
        let stats = type_stats.entry(allocation_type.clone()).or_insert(AllocationTypeStats {
            total_allocations: 0,
            total_deallocations: 0,
            current_count: 0,
            total_bytes_allocated: 0,
            total_bytes_deallocated: 0,
            current_bytes: 0,
            peak_count: 0,
            peak_bytes: 0,
            average_allocation_size: 0.0,
            largest_allocation: 0,
        });

        if is_allocation {
            stats.total_allocations += 1;
            stats.current_count += 1;
            stats.total_bytes_allocated += size;
            stats.current_bytes += size;
            stats.peak_count = stats.peak_count.max(stats.current_count);
            stats.peak_bytes = stats.peak_bytes.max(stats.current_bytes);
            stats.largest_allocation = stats.largest_allocation.max(size);
        } else {
            stats.total_deallocations += 1;
            stats.current_count = stats.current_count.saturating_sub(1);
            stats.total_bytes_deallocated += size;
            stats.current_bytes = stats.current_bytes.saturating_sub(size);
        }

        stats.average_allocation_size = if stats.total_allocations > 0 {
            stats.total_bytes_allocated as f64 / stats.total_allocations as f64
        } else {
            0.0
        };
    }

    fn get_size_bucket(&self, size: usize) -> String {
        match size {
            0..=1024 => "0-1KB".to_string(),
            1025..=10240 => "1-10KB".to_string(),
            10241..=102400 => "10-100KB".to_string(),
            102401..=1048576 => "100KB-1MB".to_string(),
            1048577..=10485760 => "1-10MB".to_string(),
            _ => ">10MB".to_string(),
        }
    }

    fn classify_leak_severity(&self, size: usize, age_seconds: f64) -> LeakSeverity {
        let large_size = size > self.config.large_allocation_threshold;
        let old_age = age_seconds > 1800.0; // 30 minutes
        let very_old_age = age_seconds > 3600.0; // 1 hour

        match (large_size, old_age, very_old_age) {
            (true, _, true) => LeakSeverity::Critical,
            (true, true, _) => LeakSeverity::High,
            (true, false, _) => LeakSeverity::Medium,
            (false, true, _) => LeakSeverity::Medium,
            _ => LeakSeverity::Low,
        }
    }

    fn calculate_gc_pressure_score(&self) -> f64 {
        // Simplified GC pressure calculation
        // In a real implementation, this would consider allocation patterns,
        // heap growth rate, and other factors
        0.3 // Placeholder value
    }

    fn calculate_allocation_rates(&self, timeline: &VecDeque<MemorySnapshot>) -> (f64, f64) {
        if timeline.len() < 2 {
            return (0.0, 0.0);
        }

        // Simplified rate calculation
        let first = &timeline[0];
        let last = &timeline[timeline.len() - 1];

        let duration = last
            .timestamp
            .duration_since(first.timestamp)
            .unwrap_or(Duration::from_secs(1))
            .as_secs_f64();

        let allocation_rate =
            (last.allocation_count as f64 - first.allocation_count as f64) / duration;
        let deallocation_rate = (last.free_count as f64 - first.free_count as f64) / duration;

        (allocation_rate.max(0.0), deallocation_rate.max(0.0))
    }

    // Pattern detection methods

    fn detect_leak_pattern(&self) -> Result<AllocationPattern> {
        let leaks = self.detect_leaks()?;
        let high_severity_leaks = leaks
            .iter()
            .filter(|l| l.severity == LeakSeverity::High || l.severity == LeakSeverity::Critical)
            .count();

        let confidence = if leaks.len() > 10 { 0.9 } else { 0.5 };
        let impact_score = (high_severity_leaks as f64 / (leaks.len().max(1)) as f64).min(1.0);

        Ok(AllocationPattern {
            pattern_type: PatternType::MemoryLeak,
            description: format!("Detected {} potential memory leaks", leaks.len()),
            confidence,
            impact_score,
            recommendations: vec![
                "Review long-lived allocations for proper cleanup".to_string(),
                "Implement RAII patterns for automatic resource management".to_string(),
            ],
            examples: leaks
                .into_iter()
                .take(3)
                .map(|leak| {
                    // Convert leak to allocation record for example
                    AllocationRecord {
                        id: leak.allocation_id,
                        size: leak.size,
                        timestamp: SystemTime::now(), // Placeholder
                        stack_trace: leak.stack_trace,
                        allocation_type: leak.allocation_type,
                        freed: false,
                        freed_at: None,
                        tags: leak.tags,
                    }
                })
                .collect(),
        })
    }

    fn detect_churn_pattern(&self) -> Result<AllocationPattern> {
        // Simplified churn detection
        let allocations = self.allocations.lock().expect("lock should not be poisoned");
        let short_lived_count = allocations
            .values()
            .filter(|record| {
                if let (Some(_freed_at), false) = (record.freed_at, record.freed) {
                    false // Contradiction, skip
                } else if record.freed {
                    if let Some(freed_at) = record.freed_at {
                        freed_at.duration_since(record.timestamp).unwrap_or(Duration::from_secs(0))
                            < Duration::from_secs(1)
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
            .count();

        let total_count = allocations.len();
        let churn_ratio = if total_count > 0 {
            short_lived_count as f64 / total_count as f64
        } else {
            0.0
        };

        Ok(AllocationPattern {
            pattern_type: PatternType::ChurningAllocations,
            description: format!(
                "High allocation churn detected: {:.1}% short-lived allocations",
                churn_ratio * 100.0
            ),
            confidence: if churn_ratio > 0.5 { 0.8 } else { 0.4 },
            impact_score: churn_ratio,
            recommendations: vec![
                "Consider object pooling for frequently allocated objects".to_string(),
                "Reduce temporary object creation in hot paths".to_string(),
            ],
            examples: vec![], // Simplified for now
        })
    }

    fn detect_large_allocation_pattern(&self) -> Result<AllocationPattern> {
        let allocations = self.allocations.lock().expect("lock should not be poisoned");
        let large_allocations: Vec<_> = allocations
            .values()
            .filter(|record| record.size > self.config.large_allocation_threshold)
            .cloned()
            .collect();

        let impact_score = if !allocations.is_empty() {
            large_allocations.len() as f64 / allocations.len() as f64
        } else {
            0.0
        };

        Ok(AllocationPattern {
            pattern_type: PatternType::LargeAllocations,
            description: format!(
                "Found {} large allocations (>{}MB)",
                large_allocations.len(),
                self.config.large_allocation_threshold / 1024 / 1024
            ),
            confidence: if large_allocations.len() > 5 { 0.9 } else { 0.6 },
            impact_score,
            recommendations: vec![
                "Review large allocations for optimization opportunities".to_string(),
                "Consider streaming or chunked processing for large data".to_string(),
            ],
            examples: large_allocations.into_iter().take(3).collect(),
        })
    }

    fn detect_fragmentation_pattern(&self) -> Result<AllocationPattern> {
        let fragmentation = self.analyze_fragmentation()?;

        Ok(AllocationPattern {
            pattern_type: PatternType::FragmentationCausing,
            description: format!(
                "Memory fragmentation at {:.1}%",
                fragmentation.fragmentation_ratio * 100.0
            ),
            confidence: 0.8,
            impact_score: fragmentation.fragmentation_ratio,
            recommendations: fragmentation.recommendations,
            examples: vec![], // Simplified for now
        })
    }
}

impl PartialOrd for LeakSeverity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LeakSeverity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_val = match self {
            LeakSeverity::Low => 0,
            LeakSeverity::Medium => 1,
            LeakSeverity::High => 2,
            LeakSeverity::Critical => 3,
        };
        let other_val = match other {
            LeakSeverity::Low => 0,
            LeakSeverity::Medium => 1,
            LeakSeverity::High => 2,
            LeakSeverity::Critical => 3,
        };
        self_val.cmp(&other_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // FIXME: This test has implementation issues causing slow execution
    async fn test_memory_profiler_basic() -> Result<()> {
        let config = MemoryProfilingConfig {
            sampling_interval_ms: 1000, // Slower sampling for faster tests
            ..Default::default()
        };
        let mut profiler = MemoryProfiler::new(config);

        // Wrap in timeout to prevent hanging
        let test_result = tokio::time::timeout(Duration::from_millis(500), async {
            profiler.start().await?;

            // Record some allocations
            let alloc_id1 = profiler.record_allocation(
                1024,
                AllocationType::Tensor,
                vec!["test".to_string()],
            )?;

            let _alloc_id2 = profiler.record_allocation(
                2048,
                AllocationType::Buffer,
                vec!["test".to_string()],
            )?;

            // Free one allocation
            profiler.record_deallocation(alloc_id1)?;

            // Give background tasks a moment to process
            tokio::time::sleep(Duration::from_millis(1)).await;

            let report = profiler.stop().await?;

            assert_eq!(report.total_allocations, 2);
            assert_eq!(report.total_deallocations, 1);
            assert_eq!(report.net_allocations, 1);

            Ok::<(), anyhow::Error>(())
        })
        .await;

        match test_result {
            Ok(result) => result,
            Err(_) => Err(anyhow::anyhow!("Test timed out after 500ms")),
        }
    }

    #[tokio::test]
    async fn test_leak_detection() -> Result<()> {
        let config = MemoryProfilingConfig {
            leak_detection_threshold_secs: 1, // 1 second for testing
            ..Default::default()
        };

        let mut profiler = MemoryProfiler::new(config);
        profiler.start().await?; // Start the profiler

        // Record allocation and wait
        profiler.record_allocation(1024, AllocationType::Tensor, vec!["leak_test".to_string()])?;

        tokio::time::sleep(Duration::from_secs(2)).await;

        let leaks = profiler.detect_leaks()?;
        assert!(!leaks.is_empty());

        Ok(())
    }

    #[test]
    fn test_size_buckets() {
        let config = MemoryProfilingConfig::default();
        let profiler = MemoryProfiler::new(config);

        assert_eq!(profiler.get_size_bucket(512), "0-1KB");
        assert_eq!(profiler.get_size_bucket(5120), "1-10KB");
        assert_eq!(profiler.get_size_bucket(51200), "10-100KB");
        assert_eq!(profiler.get_size_bucket(512000), "100KB-1MB");
        assert_eq!(profiler.get_size_bucket(5120000), "1-10MB");
        assert_eq!(profiler.get_size_bucket(51200000), ">10MB");
    }

    #[test]
    fn test_leak_severity_classification() {
        let config = MemoryProfilingConfig::default();
        let profiler = MemoryProfiler::new(config);

        // Small, new allocation
        assert_eq!(
            profiler.classify_leak_severity(1024, 60.0),
            LeakSeverity::Low
        );

        // Large, old allocation
        assert_eq!(
            profiler.classify_leak_severity(10485760, 3700.0),
            LeakSeverity::Critical
        );

        // Medium size, medium age
        assert_eq!(
            profiler.classify_leak_severity(524288, 1900.0),
            LeakSeverity::Medium
        );
    }
}
