//! Memory profiling infrastructure for TrustformeRS

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Memory allocation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    /// Size of allocation in bytes
    pub size: usize,
    /// Timestamp (milliseconds since start)
    pub timestamp_ms: u64,
    /// Stack trace if available
    pub stack_trace: Option<String>,
    /// Allocation ID
    pub id: u64,
    /// Tag for categorizing allocations
    pub tag: Option<String>,
}

/// Memory snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Total allocated memory
    pub allocated_bytes: usize,
    /// Total reserved memory
    pub reserved_bytes: usize,
    /// Number of active allocations
    pub num_allocations: usize,
    /// Memory by category
    pub allocations_by_tag: HashMap<String, usize>,
    /// Largest allocations
    pub largest_allocations: Vec<(usize, String)>,
    /// Timestamp (milliseconds since start)
    pub timestamp_ms: u64,
}

impl MemorySnapshot {
    /// Get memory usage in MB
    pub fn allocated_mb(&self) -> f64 {
        self.allocated_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get reserved memory in MB
    pub fn reserved_mb(&self) -> f64 {
        self.reserved_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get fragmentation percentage
    pub fn fragmentation_percent(&self) -> f64 {
        if self.reserved_bytes > 0 {
            (1.0 - self.allocated_bytes as f64 / self.reserved_bytes as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Memory profiler for tracking allocations
pub struct MemoryProfiler {
    /// Active allocations
    allocations: Arc<Mutex<HashMap<u64, AllocationEvent>>>,
    /// Next allocation ID
    next_id: Arc<Mutex<u64>>,
    /// Whether profiling is enabled
    enabled: Arc<Mutex<bool>>,
    /// Maximum number of stack frames to capture
    #[allow(dead_code)]
    max_stack_depth: usize,
    /// Tags for current context
    context_tags: Arc<Mutex<Vec<String>>>,
    /// Start time for relative timestamps
    start_time: std::time::Instant,
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryProfiler {
    /// Create new memory profiler
    pub fn new() -> Self {
        Self {
            allocations: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(0)),
            enabled: Arc::new(Mutex::new(false)),
            max_stack_depth: 10,
            context_tags: Arc::new(Mutex::new(Vec::new())),
            start_time: std::time::Instant::now(),
        }
    }

    /// Enable profiling
    pub fn enable(&self) {
        *self.enabled.lock().expect("Lock poisoned") = true;
    }

    /// Disable profiling
    pub fn disable(&self) {
        *self.enabled.lock().expect("Lock poisoned") = false;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        *self.enabled.lock().expect("Lock poisoned")
    }

    /// Push a context tag
    pub fn push_tag(&self, tag: String) {
        self.context_tags.lock().expect("Lock poisoned").push(tag);
    }

    /// Pop a context tag
    pub fn pop_tag(&self) {
        self.context_tags.lock().expect("Lock poisoned").pop();
    }

    /// Record an allocation
    pub fn record_allocation(&self, size: usize) -> u64 {
        if !self.is_enabled() {
            return 0;
        }

        let id = {
            let mut next_id = self.next_id.lock().expect("Lock poisoned");
            let id = *next_id;
            *next_id += 1;
            id
        };

        let tag = self.context_tags.lock().expect("Lock poisoned").last().cloned();

        let event = AllocationEvent {
            size,
            timestamp_ms: self.start_time.elapsed().as_millis() as u64,
            stack_trace: self.capture_stack_trace(),
            id,
            tag,
        };

        self.allocations.lock().expect("Lock poisoned").insert(id, event);
        id
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, id: u64) {
        if !self.is_enabled() {
            return;
        }

        self.allocations.lock().expect("Lock poisoned").remove(&id);
    }

    /// Take a memory snapshot
    pub fn take_snapshot(&self) -> MemorySnapshot {
        let allocations = self.allocations.lock().expect("Lock poisoned");

        let allocated_bytes: usize = allocations.values().map(|a| a.size).sum();
        let num_allocations = allocations.len();

        // Group by tag
        let mut allocations_by_tag = HashMap::new();
        for event in allocations.values() {
            let tag = event.tag.as_ref().unwrap_or(&"untagged".to_string()).clone();
            *allocations_by_tag.entry(tag).or_insert(0) += event.size;
        }

        // Find largest allocations
        let mut largest: Vec<_> = allocations
            .values()
            .map(|a| {
                (
                    a.size,
                    a.tag.as_ref().unwrap_or(&"untagged".to_string()).clone(),
                )
            })
            .collect();
        largest.sort_by(|a, b| b.0.cmp(&a.0));
        largest.truncate(10);

        MemorySnapshot {
            allocated_bytes,
            reserved_bytes: allocated_bytes + allocated_bytes / 4, // Estimate 25% overhead
            num_allocations,
            allocations_by_tag,
            largest_allocations: largest,
            timestamp_ms: self.start_time.elapsed().as_millis() as u64,
        }
    }

    /// Clear all allocations
    pub fn clear(&self) {
        self.allocations.lock().expect("Lock poisoned").clear();
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let allocations = self.allocations.lock().expect("Lock poisoned");

        let total_size: usize = allocations.values().map(|a| a.size).sum();
        let count = allocations.len();

        let sizes: Vec<usize> = allocations.values().map(|a| a.size).collect();
        let avg_size = if count > 0 { total_size / count } else { 0 };
        let max_size = sizes.iter().max().copied().unwrap_or(0);
        let min_size = sizes.iter().min().copied().unwrap_or(0);

        MemoryStats {
            total_allocated: total_size,
            num_allocations: count,
            avg_allocation_size: avg_size,
            max_allocation_size: max_size,
            min_allocation_size: min_size,
        }
    }

    /// Capture stack trace for debugging memory allocations
    fn capture_stack_trace(&self) -> Option<String> {
        // Enhanced stack trace capture for better debugging
        #[cfg(feature = "backtrace")]
        {
            use std::backtrace::Backtrace;
            let bt = Backtrace::capture();
            if bt.status() == std::backtrace::BacktraceStatus::Captured {
                return Some(format!("{}", bt));
            }
        }

        // Fallback: capture limited call information
        #[cfg(not(feature = "backtrace"))]
        {
            // Get current thread information
            let thread = std::thread::current();
            let thread_name = thread.name().unwrap_or("unnamed");

            // Capture basic allocation context
            let context = format!(
                "Thread: {} (id: {:?})\nFunction context: memory allocation\nFile: {}\nLine: {}",
                thread_name,
                thread.id(),
                file!(),
                line!()
            );
            Some(context)
        }

        // If backtrace feature is enabled but capture failed
        #[cfg(feature = "backtrace")]
        {
            let thread = std::thread::current();
            let fallback_info = format!(
                "Backtrace capture failed\nThread: {} (id: {:?})\nTimestamp: {:?}",
                thread.name().unwrap_or("unnamed"),
                thread.id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
            );
            Some(fallback_info)
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub num_allocations: usize,
    pub avg_allocation_size: usize,
    pub max_allocation_size: usize,
    pub min_allocation_size: usize,
}

/// Memory usage tracker for specific operations
pub struct MemoryTracker {
    profiler: Arc<MemoryProfiler>,
    initial_snapshot: Option<MemorySnapshot>,
}

impl MemoryTracker {
    /// Create new memory tracker
    pub fn new(profiler: Arc<MemoryProfiler>) -> Self {
        Self {
            profiler,
            initial_snapshot: None,
        }
    }

    /// Start tracking memory for an operation
    pub fn start_tracking(&mut self, tag: &str) {
        self.profiler.push_tag(tag.to_string());
        self.initial_snapshot = Some(self.profiler.take_snapshot());
    }

    /// Stop tracking and return memory delta
    pub fn stop_tracking(&mut self) -> Option<MemoryDelta> {
        self.profiler.pop_tag();

        if let Some(initial) = self.initial_snapshot.take() {
            let final_snapshot = self.profiler.take_snapshot();

            Some(MemoryDelta {
                allocated_delta: final_snapshot.allocated_bytes as i64
                    - initial.allocated_bytes as i64,
                allocations_delta: final_snapshot.num_allocations as i64
                    - initial.num_allocations as i64,
                peak_allocated: final_snapshot.allocated_bytes.max(initial.allocated_bytes),
                duration: std::time::Duration::from_millis(
                    final_snapshot.timestamp_ms - initial.timestamp_ms,
                ),
            })
        } else {
            None
        }
    }
}

/// Memory change between two snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDelta {
    /// Change in allocated memory (can be negative)
    pub allocated_delta: i64,
    /// Change in number of allocations (can be negative)
    pub allocations_delta: i64,
    /// Peak allocated memory during the period
    pub peak_allocated: usize,
    /// Duration of the measurement
    pub duration: std::time::Duration,
}

impl MemoryDelta {
    /// Get allocated delta in MB
    pub fn allocated_delta_mb(&self) -> f64 {
        self.allocated_delta as f64 / (1024.0 * 1024.0)
    }

    /// Get peak allocated in MB
    pub fn peak_allocated_mb(&self) -> f64 {
        self.peak_allocated as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler() {
        let profiler = MemoryProfiler::new();
        profiler.enable();

        // Record some allocations
        let id1 = profiler.record_allocation(1024);
        let _id2 = profiler.record_allocation(2048);

        let stats = profiler.get_stats();
        assert_eq!(stats.num_allocations, 2);
        assert_eq!(stats.total_allocated, 3072);

        // Deallocate one
        profiler.record_deallocation(id1);

        let stats = profiler.get_stats();
        assert_eq!(stats.num_allocations, 1);
        assert_eq!(stats.total_allocated, 2048);
    }

    #[test]
    fn test_memory_snapshot() {
        let profiler = MemoryProfiler::new();
        profiler.enable();

        profiler.push_tag("tensors".to_string());
        profiler.record_allocation(1024 * 1024);
        profiler.pop_tag();

        profiler.push_tag("weights".to_string());
        profiler.record_allocation(2 * 1024 * 1024);
        profiler.pop_tag();

        let snapshot = profiler.take_snapshot();
        assert_eq!(snapshot.num_allocations, 2);
        assert_eq!(snapshot.allocated_bytes, 3 * 1024 * 1024);
        assert!(snapshot.allocations_by_tag.contains_key("tensors"));
        assert!(snapshot.allocations_by_tag.contains_key("weights"));
    }

    #[test]
    fn test_memory_tracker() {
        let profiler = Arc::new(MemoryProfiler::new());
        profiler.enable();

        let mut tracker = MemoryTracker::new(profiler.clone());

        tracker.start_tracking("test_operation");

        // Simulate some allocations
        profiler.record_allocation(1024);
        profiler.record_allocation(2048);

        let delta = tracker.stop_tracking().unwrap();
        assert_eq!(delta.allocated_delta, 3072);
        assert_eq!(delta.allocations_delta, 2);
    }
}
