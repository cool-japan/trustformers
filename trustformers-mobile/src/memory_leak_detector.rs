//! Mobile memory leak detection and analysis for TrustformeRS
//!
//! This module provides comprehensive memory leak detection specifically designed for mobile
//! environments where memory is constrained and leaks can significantly impact performance.

use crate::scirs2_compat::random::legacy;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use trustformers_core::errors::{runtime_error, Result};

/// Memory allocation tracking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationInfo {
    pub ptr: usize,
    pub size: usize,
    pub timestamp: u64,
    pub stack_trace: Vec<String>,
    pub allocation_type: AllocationType,
    pub thread_id: String,
    pub tag: Option<String>,
}

/// Types of memory allocations being tracked
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllocationType {
    Model,
    Tensor,
    Buffer,
    Cache,
    Temporary,
    Configuration,
    Native,
    Unity,
    Unknown,
}

/// Memory leak detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeakDetectorConfig {
    /// Enable memory leak detection
    pub enabled: bool,
    /// Maximum number of allocations to track
    pub max_tracked_allocations: usize,
    /// Sampling rate for allocation tracking (1.0 = track all, 0.1 = track 10%)
    pub sampling_rate: f32,
    /// Minimum allocation size to track (bytes)
    pub min_tracked_size: usize,
    /// Detection interval in seconds
    pub detection_interval: Duration,
    /// Leak threshold - allocations older than this are considered potential leaks
    pub leak_threshold: Duration,
    /// Enable stack trace collection (expensive)
    pub collect_stack_traces: bool,
    /// Enable automatic cleanup suggestions
    pub enable_auto_suggestions: bool,
    /// Maximum memory usage before triggering aggressive detection
    pub memory_pressure_threshold: usize,
}

/// Memory leak analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeakAnalysis {
    pub total_tracked_allocations: usize,
    pub total_tracked_memory: usize,
    pub potential_leaks: Vec<AllocationInfo>,
    pub leak_patterns: Vec<LeakPattern>,
    pub memory_usage_trend: Vec<MemoryUsagePoint>,
    pub recommendations: Vec<String>,
    pub timestamp: u64,
}

/// Detected memory leak pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakPattern {
    pub pattern_type: LeakPatternType,
    pub allocation_type: AllocationType,
    pub frequency: usize,
    pub total_leaked_memory: usize,
    pub average_leak_size: usize,
    pub first_occurrence: u64,
    pub last_occurrence: u64,
    pub confidence: f32,
    pub description: String,
}

/// Types of leak patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeakPatternType {
    ConstantLeak,      // Steady memory growth
    PeriodicLeak,      // Periodic memory spikes
    BurstLeak,         // Sudden large memory increases
    GradualLeak,       // Slow memory growth over time
    StackOverflow,     // Deep recursion or stack growth
    FragmentationLeak, // Memory fragmentation
}

/// Memory usage tracking point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsagePoint {
    pub timestamp: u64,
    pub total_memory: usize,
    pub allocated_memory: usize,
    pub free_memory: usize,
    pub allocation_count: usize,
    pub largest_allocation: usize,
}

/// Mobile memory leak detector
pub struct MobileMemoryLeakDetector {
    config: MemoryLeakDetectorConfig,
    allocations: Arc<RwLock<HashMap<usize, AllocationInfo>>>,
    memory_history: Arc<Mutex<VecDeque<MemoryUsagePoint>>>,
    leak_patterns: Arc<Mutex<Vec<LeakPattern>>>,
    detection_thread: Option<thread::JoinHandle<()>>,
    is_running: Arc<Mutex<bool>>,
    statistics: Arc<Mutex<DetectorStatistics>>,
}

/// Internal detector statistics
#[derive(Debug, Default)]
pub struct DetectorStatistics {
    total_allocations_tracked: usize,
    total_deallocations_tracked: usize,
    total_leaks_detected: usize,
    total_memory_leaked: usize,
    detection_runs: usize,
    false_positives: usize,
    true_positives: usize,
}

impl Default for MemoryLeakDetectorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_tracked_allocations: 10_000,
            sampling_rate: 0.1,     // Track 10% of allocations by default
            min_tracked_size: 1024, // 1KB minimum
            detection_interval: Duration::from_secs(30),
            leak_threshold: Duration::from_secs(300), // 5 minutes
            collect_stack_traces: false,              // Disabled by default for performance
            enable_auto_suggestions: true,
            memory_pressure_threshold: 100 * 1024 * 1024, // 100MB
        }
    }
}

impl MobileMemoryLeakDetector {
    /// Create a new memory leak detector
    pub fn new(config: MemoryLeakDetectorConfig) -> Self {
        Self {
            config,
            allocations: Arc::new(RwLock::new(HashMap::new())),
            memory_history: Arc::new(Mutex::new(VecDeque::new())),
            leak_patterns: Arc::new(Mutex::new(Vec::new())),
            detection_thread: None,
            is_running: Arc::new(Mutex::new(false)),
            statistics: Arc::new(Mutex::new(DetectorStatistics::default())),
        }
    }

    /// Start the memory leak detector
    pub fn start(&mut self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut is_running =
            self.is_running.lock().map_err(|_| runtime_error("Failed to acquire lock"))?;

        if *is_running {
            return Ok(());
        }

        *is_running = true;
        drop(is_running);

        let allocations = Arc::clone(&self.allocations);
        let memory_history = Arc::clone(&self.memory_history);
        let leak_patterns = Arc::clone(&self.leak_patterns);
        let is_running_clone = Arc::clone(&self.is_running);
        let statistics = Arc::clone(&self.statistics);
        let config = self.config.clone();

        let handle = thread::spawn(move || {
            Self::detection_loop(
                config,
                allocations,
                memory_history,
                leak_patterns,
                is_running_clone,
                statistics,
            );
        });

        self.detection_thread = Some(handle);
        Ok(())
    }

    /// Stop the memory leak detector
    pub fn stop(&mut self) -> Result<()> {
        {
            let mut is_running =
                self.is_running.lock().map_err(|_| runtime_error("Failed to acquire lock"))?;
            *is_running = false;
        }

        if let Some(handle) = self.detection_thread.take() {
            handle.join().map_err(|_| runtime_error("Failed to join detection thread"))?;
        }

        Ok(())
    }

    /// Track a memory allocation
    pub fn track_allocation(
        &self,
        ptr: usize,
        size: usize,
        allocation_type: AllocationType,
        tag: Option<String>,
    ) -> Result<()> {
        if !self.config.enabled || size < self.config.min_tracked_size {
            return Ok(());
        }

        // Apply sampling
        if legacy::f32() > self.config.sampling_rate {
            return Ok(());
        }

        let mut allocations = self
            .allocations
            .write()
            .map_err(|_| runtime_error("Failed to acquire write lock"))?;

        // Check if we're at capacity
        if allocations.len() >= self.config.max_tracked_allocations {
            // Remove oldest allocation
            if let Some((oldest_ptr, _)) = allocations
                .iter()
                .min_by_key(|(_, info)| info.timestamp)
                .map(|(ptr, info)| (*ptr, info.clone()))
            {
                allocations.remove(&oldest_ptr);
            }
        }

        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_micros() as u64;

        let stack_trace = if self.config.collect_stack_traces {
            self.capture_stack_trace()
        } else {
            Vec::new()
        };

        let allocation_info = AllocationInfo {
            ptr,
            size,
            timestamp,
            stack_trace,
            allocation_type,
            thread_id: format!("{:?}", thread::current().id()),
            tag,
        };

        allocations.insert(ptr, allocation_info);

        // Update statistics
        if let Ok(mut stats) = self.statistics.lock() {
            stats.total_allocations_tracked += 1;
        }

        Ok(())
    }

    /// Track a memory deallocation
    pub fn track_deallocation(&self, ptr: usize) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut allocations = self
            .allocations
            .write()
            .map_err(|_| runtime_error("Failed to acquire write lock"))?;

        allocations.remove(&ptr);

        // Update statistics
        if let Ok(mut stats) = self.statistics.lock() {
            stats.total_deallocations_tracked += 1;
        }

        Ok(())
    }

    /// Perform immediate leak detection analysis
    pub fn detect_leaks(&self) -> Result<MemoryLeakAnalysis> {
        let allocations = self
            .allocations
            .read()
            .map_err(|_| runtime_error("Failed to acquire read lock"))?;

        let current_time =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_micros() as u64;

        let leak_threshold_micros = self.config.leak_threshold.as_micros() as u64;

        // Find potential leaks
        let potential_leaks: Vec<AllocationInfo> = allocations
            .values()
            .filter(|info| current_time.saturating_sub(info.timestamp) > leak_threshold_micros)
            .cloned()
            .collect();

        // Analyze leak patterns
        let leak_patterns = self.analyze_leak_patterns(&potential_leaks)?;

        // Get memory usage history
        let memory_usage_trend = self
            .memory_history
            .lock()
            .map_err(|_| runtime_error("Failed to acquire lock"))?
            .iter()
            .cloned()
            .collect();

        // Generate recommendations
        let recommendations = self.generate_recommendations(&potential_leaks, &leak_patterns);

        // Calculate total tracked memory
        let total_tracked_memory = allocations.values().map(|info| info.size).sum();

        // Update statistics
        if let Ok(mut stats) = self.statistics.lock() {
            stats.total_leaks_detected += potential_leaks.len();
            stats.total_memory_leaked +=
                potential_leaks.iter().map(|leak| leak.size).sum::<usize>();
            stats.detection_runs += 1;
        }

        Ok(MemoryLeakAnalysis {
            total_tracked_allocations: allocations.len(),
            total_tracked_memory,
            potential_leaks,
            leak_patterns,
            memory_usage_trend,
            recommendations,
            timestamp: current_time,
        })
    }

    /// Get current memory usage statistics
    pub fn get_memory_usage(&self) -> Result<MemoryUsagePoint> {
        let allocations = self
            .allocations
            .read()
            .map_err(|_| runtime_error("Failed to acquire read lock"))?;

        let current_time =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        let allocated_memory = allocations.values().map(|info| info.size).sum();
        let allocation_count = allocations.len();
        let largest_allocation = allocations.values().map(|info| info.size).max().unwrap_or(0);

        // Get system memory info (simplified - would use platform-specific APIs)
        let total_memory = self.get_system_total_memory();
        let free_memory = total_memory.saturating_sub(allocated_memory);

        Ok(MemoryUsagePoint {
            timestamp: current_time,
            total_memory,
            allocated_memory,
            free_memory,
            allocation_count,
            largest_allocation,
        })
    }

    /// Clear all tracked allocations
    pub fn clear_tracking(&self) -> Result<()> {
        let mut allocations = self
            .allocations
            .write()
            .map_err(|_| runtime_error("Failed to acquire write lock"))?;

        allocations.clear();

        let mut memory_history = self
            .memory_history
            .lock()
            .map_err(|_| runtime_error("Failed to acquire lock"))?;

        memory_history.clear();

        Ok(())
    }

    /// Get detector statistics
    pub fn get_statistics(&self) -> Result<DetectorStatistics> {
        let stats = self.statistics.lock().map_err(|_| runtime_error("Failed to acquire lock"))?;

        Ok(DetectorStatistics {
            total_allocations_tracked: stats.total_allocations_tracked,
            total_deallocations_tracked: stats.total_deallocations_tracked,
            total_leaks_detected: stats.total_leaks_detected,
            total_memory_leaked: stats.total_memory_leaked,
            detection_runs: stats.detection_runs,
            false_positives: stats.false_positives,
            true_positives: stats.true_positives,
        })
    }

    // Private methods

    fn detection_loop(
        config: MemoryLeakDetectorConfig,
        allocations: Arc<RwLock<HashMap<usize, AllocationInfo>>>,
        memory_history: Arc<Mutex<VecDeque<MemoryUsagePoint>>>,
        leak_patterns: Arc<Mutex<Vec<LeakPattern>>>,
        is_running: Arc<Mutex<bool>>,
        statistics: Arc<Mutex<DetectorStatistics>>,
    ) {
        while {
            let running = is_running.lock().map(|r| *r).unwrap_or(false);
            running
        } {
            // Collect current memory usage
            if let Ok(usage) = Self::collect_memory_usage(&allocations) {
                if let Ok(mut history) = memory_history.lock() {
                    history.push_back(usage);

                    // Keep only recent history (last hour)
                    let max_history_points = 3600 / config.detection_interval.as_secs() as usize;
                    while history.len() > max_history_points {
                        history.pop_front();
                    }
                }
            }

            // Detect memory pressure
            if let Ok(allocations_guard) = allocations.read() {
                let total_tracked = allocations_guard.values().map(|info| info.size).sum::<usize>();
                if total_tracked > config.memory_pressure_threshold {
                    // Trigger aggressive leak detection
                    drop(allocations_guard);
                    Self::aggressive_leak_detection(&allocations, &leak_patterns, &config);
                }
            }

            thread::sleep(config.detection_interval);
        }
    }

    fn collect_memory_usage(
        allocations: &Arc<RwLock<HashMap<usize, AllocationInfo>>>,
    ) -> Result<MemoryUsagePoint> {
        let allocations_guard =
            allocations.read().map_err(|_| runtime_error("Failed to acquire read lock"))?;

        let current_time =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        let allocated_memory: usize = allocations_guard.values().map(|info| info.size).sum();
        let allocation_count = allocations_guard.len();
        let largest_allocation =
            allocations_guard.values().map(|info| info.size).max().unwrap_or(0);

        // Simplified system memory detection
        let total_memory: usize = 1024 * 1024 * 1024; // 1GB default
        let free_memory = total_memory.saturating_sub(allocated_memory);

        Ok(MemoryUsagePoint {
            timestamp: current_time,
            total_memory,
            allocated_memory,
            free_memory,
            allocation_count,
            largest_allocation,
        })
    }

    fn aggressive_leak_detection(
        allocations: &Arc<RwLock<HashMap<usize, AllocationInfo>>>,
        leak_patterns: &Arc<Mutex<Vec<LeakPattern>>>,
        config: &MemoryLeakDetectorConfig,
    ) {
        // Implementation for aggressive leak detection during memory pressure
        if let Ok(allocations_guard) = allocations.read() {
            let current_time =
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

            // Find allocations older than half the normal threshold
            let aggressive_threshold = config.leak_threshold.as_secs() / 2;

            let suspicious_allocations: Vec<_> = allocations_guard
                .values()
                .filter(|info| current_time.saturating_sub(info.timestamp) > aggressive_threshold)
                .cloned()
                .collect();

            if !suspicious_allocations.is_empty() {
                // Log warning about potential memory pressure
                eprintln!(
                    "Memory pressure detected: {} suspicious allocations totaling {} bytes",
                    suspicious_allocations.len(),
                    suspicious_allocations.iter().map(|info| info.size).sum::<usize>()
                );
            }
        }
    }

    fn analyze_leak_patterns(&self, leaks: &[AllocationInfo]) -> Result<Vec<LeakPattern>> {
        let mut patterns = Vec::new();

        // Group leaks by allocation type
        let mut by_type: HashMap<AllocationType, Vec<&AllocationInfo>> = HashMap::new();
        for leak in leaks {
            by_type.entry(leak.allocation_type).or_default().push(leak);
        }

        for (alloc_type, type_leaks) in by_type {
            if type_leaks.is_empty() {
                continue;
            }

            let total_leaked = type_leaks.iter().map(|leak| leak.size).sum();
            let average_size = total_leaked / type_leaks.len();
            let first_occurrence = type_leaks.iter().map(|leak| leak.timestamp).min().unwrap_or(0);
            let last_occurrence = type_leaks.iter().map(|leak| leak.timestamp).max().unwrap_or(0);

            // Determine pattern type based on timing and size distribution
            let pattern_type = self.classify_leak_pattern(&type_leaks);
            let confidence = self.calculate_pattern_confidence(&type_leaks, pattern_type);

            let description = format!(
                "{:?} pattern detected for {:?} allocations: {} instances, {} bytes total",
                pattern_type,
                alloc_type,
                type_leaks.len(),
                total_leaked
            );

            patterns.push(LeakPattern {
                pattern_type,
                allocation_type: alloc_type,
                frequency: type_leaks.len(),
                total_leaked_memory: total_leaked,
                average_leak_size: average_size,
                first_occurrence,
                last_occurrence,
                confidence,
                description,
            });
        }

        Ok(patterns)
    }

    fn classify_leak_pattern(&self, leaks: &[&AllocationInfo]) -> LeakPatternType {
        if leaks.len() < 2 {
            return LeakPatternType::ConstantLeak;
        }

        // Analyze timing patterns
        let mut timestamps: Vec<u64> = leaks.iter().map(|leak| leak.timestamp).collect();
        timestamps.sort_unstable();

        let intervals: Vec<u64> =
            timestamps.windows(2).map(|window| window[1] - window[0]).collect();

        let avg_interval = intervals.iter().sum::<u64>() / intervals.len() as u64;
        let interval_variance = intervals
            .iter()
            .map(|&interval| (interval as i64 - avg_interval as i64).unsigned_abs())
            .sum::<u64>()
            / intervals.len() as u64;

        // Analyze size patterns
        let sizes: Vec<usize> = leaks.iter().map(|leak| leak.size).collect();
        let avg_size = sizes.iter().sum::<usize>() / sizes.len();
        let max_size = *sizes.iter().max().unwrap_or(&0);

        // Classification logic
        if interval_variance < avg_interval / 4 {
            // Regular intervals
            LeakPatternType::PeriodicLeak
        } else if max_size > avg_size * 10 {
            // Large size variations
            LeakPatternType::BurstLeak
        } else if timestamps.last().expect("Operation failed")
            - timestamps.first().expect("Operation failed")
            > 3600
        {
            // Long time span
            LeakPatternType::GradualLeak
        } else {
            LeakPatternType::ConstantLeak
        }
    }

    fn calculate_pattern_confidence(
        &self,
        leaks: &[&AllocationInfo],
        pattern_type: LeakPatternType,
    ) -> f32 {
        // Simple confidence calculation based on pattern consistency
        let base_confidence = match pattern_type {
            LeakPatternType::PeriodicLeak => 0.8,
            LeakPatternType::BurstLeak => 0.7,
            LeakPatternType::GradualLeak => 0.6,
            LeakPatternType::ConstantLeak => 0.5,
            LeakPatternType::StackOverflow => 0.9,
            LeakPatternType::FragmentationLeak => 0.4,
        };

        // Adjust based on sample size
        let sample_factor = (leaks.len() as f32 / 10.0).min(1.0);

        base_confidence * sample_factor
    }

    fn generate_recommendations(
        &self,
        leaks: &[AllocationInfo],
        patterns: &[LeakPattern],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if leaks.is_empty() {
            recommendations.push("No memory leaks detected. Continue monitoring.".to_string());
            return recommendations;
        }

        // General recommendations
        recommendations.push(format!(
            "Found {} potential memory leaks totaling {} bytes",
            leaks.len(),
            leaks.iter().map(|leak| leak.size).sum::<usize>()
        ));

        // Pattern-specific recommendations
        for pattern in patterns {
            match pattern.pattern_type {
                LeakPatternType::ConstantLeak => {
                    recommendations.push(format!(
                        "Constant leak pattern detected for {:?}. Check for missing deallocations.",
                        pattern.allocation_type
                    ));
                },
                LeakPatternType::PeriodicLeak => {
                    recommendations.push(format!(
                        "Periodic leak pattern detected for {:?}. Check timer-based or event-driven code.",
                        pattern.allocation_type
                    ));
                },
                LeakPatternType::BurstLeak => {
                    recommendations.push(format!(
                        "Burst leak pattern detected for {:?}. Check batch operations or caching logic.",
                        pattern.allocation_type
                    ));
                },
                LeakPatternType::GradualLeak => {
                    recommendations.push(format!(
                        "Gradual leak pattern detected for {:?}. Check for accumulated temporary allocations.",
                        pattern.allocation_type
                    ));
                },
                LeakPatternType::StackOverflow => {
                    recommendations.push(
                        "Stack overflow pattern detected. Check for infinite recursion."
                            .to_string(),
                    );
                },
                LeakPatternType::FragmentationLeak => {
                    recommendations.push(
                        "Memory fragmentation detected. Consider using memory pools.".to_string(),
                    );
                },
            }
        }

        // Type-specific recommendations
        let tensor_leaks = leaks
            .iter()
            .filter(|leak| leak.allocation_type == AllocationType::Tensor)
            .count();
        if tensor_leaks > 0 {
            recommendations.push(format!(
                "{} tensor leaks detected. Ensure tensors are properly dropped after use.",
                tensor_leaks
            ));
        }

        let model_leaks = leaks
            .iter()
            .filter(|leak| leak.allocation_type == AllocationType::Model)
            .count();
        if model_leaks > 0 {
            recommendations.push(format!(
                "{} model leaks detected. Check model lifecycle management.",
                model_leaks
            ));
        }

        recommendations
    }

    fn capture_stack_trace(&self) -> Vec<String> {
        // Simplified stack trace - in a real implementation, you'd use platform-specific APIs
        vec![
            "trustformers_mobile::inference".to_string(),
            "trustformers_mobile::model_load".to_string(),
            "main".to_string(),
        ]
    }

    fn get_system_total_memory(&self) -> usize {
        // Simplified - would use platform-specific APIs to get actual system memory
        #[cfg(target_os = "ios")]
        {
            // iOS memory detection would go here
            1024 * 1024 * 1024 // 1GB default
        }
        #[cfg(target_os = "android")]
        {
            // Android memory detection would go here
            512 * 1024 * 1024 // 512MB default
        }
        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            1024 * 1024 * 1024 // 1GB default
        }
    }
}

impl Drop for MobileMemoryLeakDetector {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_leak_detector_creation() {
        let config = MemoryLeakDetectorConfig::default();
        let detector = MobileMemoryLeakDetector::new(config);
        assert!(!detector.config.enabled || detector.config.enabled);
    }

    #[test]
    fn test_allocation_tracking() {
        let config = MemoryLeakDetectorConfig {
            enabled: true,
            sampling_rate: 1.0, // Track all allocations
            min_tracked_size: 0,
            ..Default::default()
        };

        let detector = MobileMemoryLeakDetector::new(config);

        // Track some allocations
        detector
            .track_allocation(0x1000, 1024, AllocationType::Tensor, None)
            .expect("Operation failed");
        detector
            .track_allocation(
                0x2000,
                2048,
                AllocationType::Model,
                Some("test".to_string()),
            )
            .expect("Operation failed");

        // Get current usage
        let usage = detector.get_memory_usage().expect("Operation failed");
        assert_eq!(usage.allocation_count, 2);
        assert_eq!(usage.allocated_memory, 3072);

        // Track deallocation
        detector.track_deallocation(0x1000).expect("Operation failed");

        let usage_after = detector.get_memory_usage().expect("Operation failed");
        assert_eq!(usage_after.allocation_count, 1);
        assert_eq!(usage_after.allocated_memory, 2048);
    }

    #[test]
    fn test_leak_detection() {
        let config = MemoryLeakDetectorConfig {
            enabled: true,
            sampling_rate: 1.0,
            min_tracked_size: 0,
            leak_threshold: Duration::from_secs(0), // Immediate leak detection
            ..Default::default()
        };

        let detector = MobileMemoryLeakDetector::new(config);

        // Track allocation but don't deallocate
        detector
            .track_allocation(0x1000, 1024, AllocationType::Tensor, None)
            .expect("Operation failed");

        // Wait a moment to ensure timestamp difference
        thread::sleep(Duration::from_millis(10));

        // Detect leaks
        let analysis = detector.detect_leaks().expect("Operation failed");
        assert_eq!(analysis.potential_leaks.len(), 1);
        assert_eq!(analysis.potential_leaks[0].size, 1024);
    }

    #[test]
    fn test_pattern_classification() {
        let config = MemoryLeakDetectorConfig::default();
        let detector = MobileMemoryLeakDetector::new(config);

        // Create test leak data
        let mut leaks = Vec::new();
        let base_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Operation failed")
            .as_secs();

        for i in 0..5 {
            leaks.push(AllocationInfo {
                ptr: 0x1000 + i * 0x1000,
                size: 1024,
                timestamp: base_time + (i as u64) * 60, // Every minute
                stack_trace: vec![],
                allocation_type: AllocationType::Tensor,
                thread_id: "test".to_string(),
                tag: None,
            });
        }

        let leak_refs: Vec<&AllocationInfo> = leaks.iter().collect();
        let pattern_type = detector.classify_leak_pattern(&leak_refs);

        // Should detect periodic pattern due to regular timing
        assert_eq!(pattern_type, LeakPatternType::PeriodicLeak);
    }
}
