//! Memory tracking and allocation analysis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use uuid::Uuid;

/// Memory allocation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    pub allocation_id: Uuid,
    pub size_bytes: usize,
    pub allocation_type: MemoryAllocationType,
    pub device_id: Option<i32>,
    pub timestamp: SystemTime,
    pub stack_trace: Vec<String>,
    pub freed: bool,
    pub free_timestamp: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryAllocationType {
    Host,
    Device,
    Unified,
    Pinned,
    Mapped,
}

/// Memory allocation tracker
#[derive(Debug)]
pub struct MemoryTracker {
    pub(crate) allocations: HashMap<Uuid, MemoryAllocation>,
    pub(crate) total_allocated: usize,
    pub(crate) peak_allocated: usize,
    pub(crate) allocation_count: usize,
    pub(crate) deallocation_count: usize,
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            total_allocated: 0,
            peak_allocated: 0,
            allocation_count: 0,
            deallocation_count: 0,
        }
    }

    pub fn track_allocation(&mut self, allocation: MemoryAllocation) {
        self.total_allocated += allocation.size_bytes;
        self.allocation_count += 1;

        if self.total_allocated > self.peak_allocated {
            self.peak_allocated = self.total_allocated;
        }

        self.allocations.insert(allocation.allocation_id, allocation);
    }

    pub fn track_deallocation(&mut self, allocation_id: Uuid) {
        if let Some(mut allocation) = self.allocations.remove(&allocation_id) {
            allocation.freed = true;
            allocation.free_timestamp = Some(SystemTime::now());
            self.total_allocated = self.total_allocated.saturating_sub(allocation.size_bytes);
            self.deallocation_count += 1;
        }
    }

    pub fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.total_allocated,
            peak_allocated: self.peak_allocated,
            active_allocations: self.allocations.len(),
            allocation_count: self.allocation_count,
            deallocation_count: self.deallocation_count,
            memory_efficiency: if self.allocation_count > 0 {
                self.deallocation_count as f64 / self.allocation_count as f64
            } else {
                1.0
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub active_allocations: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub memory_efficiency: f64,
}

/// Memory efficiency analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEfficiencyAnalysis {
    pub peak_memory_mb: f64,
    pub min_memory_mb: f64,
    pub avg_memory_mb: f64,
    pub memory_variance: f64,
    pub efficiency_score: f64,
}

impl Default for MemoryEfficiencyAnalysis {
    fn default() -> Self {
        Self {
            peak_memory_mb: 0.0,
            min_memory_mb: 0.0,
            avg_memory_mb: 0.0,
            memory_variance: 0.0,
            efficiency_score: 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker_new() {
        let tracker = MemoryTracker::new();
        let stats = tracker.get_memory_stats();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.peak_allocated, 0);
        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.allocation_count, 0);
        assert_eq!(stats.deallocation_count, 0);
        assert!((stats.memory_efficiency - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_memory_tracker_track_allocation() {
        let mut tracker = MemoryTracker::new();
        let alloc_id = Uuid::new_v4();
        let allocation = MemoryAllocation {
            allocation_id: alloc_id,
            size_bytes: 1024,
            allocation_type: MemoryAllocationType::Host,
            device_id: None,
            timestamp: SystemTime::now(),
            stack_trace: vec!["frame1".to_string()],
            freed: false,
            free_timestamp: None,
        };
        tracker.track_allocation(allocation);
        let stats = tracker.get_memory_stats();
        assert_eq!(stats.total_allocated, 1024);
        assert_eq!(stats.peak_allocated, 1024);
        assert_eq!(stats.active_allocations, 1);
        assert_eq!(stats.allocation_count, 1);
    }

    #[test]
    fn test_memory_tracker_multiple_allocations_peak() {
        let mut tracker = MemoryTracker::new();
        for size in [512, 1024, 256] {
            let allocation = MemoryAllocation {
                allocation_id: Uuid::new_v4(),
                size_bytes: size,
                allocation_type: MemoryAllocationType::Host,
                device_id: None,
                timestamp: SystemTime::now(),
                stack_trace: Vec::new(),
                freed: false,
                free_timestamp: None,
            };
            tracker.track_allocation(allocation);
        }
        let stats = tracker.get_memory_stats();
        assert_eq!(stats.total_allocated, 512 + 1024 + 256);
        assert_eq!(stats.peak_allocated, 512 + 1024 + 256);
        assert_eq!(stats.allocation_count, 3);
    }

    #[test]
    fn test_memory_tracker_deallocation() {
        let mut tracker = MemoryTracker::new();
        let alloc_id = Uuid::new_v4();
        let allocation = MemoryAllocation {
            allocation_id: alloc_id,
            size_bytes: 2048,
            allocation_type: MemoryAllocationType::Device,
            device_id: Some(0),
            timestamp: SystemTime::now(),
            stack_trace: Vec::new(),
            freed: false,
            free_timestamp: None,
        };
        tracker.track_allocation(allocation);
        tracker.track_deallocation(alloc_id);
        let stats = tracker.get_memory_stats();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.peak_allocated, 2048);
        assert_eq!(stats.deallocation_count, 1);
        assert_eq!(stats.active_allocations, 0);
    }

    #[test]
    fn test_memory_tracker_dealloc_nonexistent() {
        let mut tracker = MemoryTracker::new();
        tracker.track_deallocation(Uuid::new_v4());
        let stats = tracker.get_memory_stats();
        assert_eq!(stats.deallocation_count, 0);
        assert_eq!(stats.total_allocated, 0);
    }

    #[test]
    fn test_memory_tracker_efficiency_with_allocations() {
        let mut tracker = MemoryTracker::new();
        let ids: Vec<Uuid> = (0..4)
            .map(|_| {
                let id = Uuid::new_v4();
                tracker.track_allocation(MemoryAllocation {
                    allocation_id: id,
                    size_bytes: 100,
                    allocation_type: MemoryAllocationType::Host,
                    device_id: None,
                    timestamp: SystemTime::now(),
                    stack_trace: Vec::new(),
                    freed: false,
                    free_timestamp: None,
                });
                id
            })
            .collect();
        tracker.track_deallocation(ids[0]);
        tracker.track_deallocation(ids[1]);
        let stats = tracker.get_memory_stats();
        assert!((stats.memory_efficiency - 0.5).abs() < 1e-9);
    }
}
