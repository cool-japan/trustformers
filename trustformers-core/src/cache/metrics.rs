use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Metrics for monitoring cache performance
#[derive(Debug)]
pub struct CacheMetrics {
    // Hit/miss statistics
    hits: AtomicU64,
    misses: AtomicU64,

    // Size metrics
    total_entries: AtomicUsize,
    total_memory_bytes: AtomicUsize,

    // Eviction metrics
    evictions: AtomicU64,
    eviction_memory_freed: AtomicUsize,

    // Timing metrics (in microseconds)
    total_lookup_time_us: AtomicU64,
    total_insert_time_us: AtomicU64,

    // Performance tracking
    last_reset: parking_lot::Mutex<Instant>,
}

impl CacheMetrics {
    pub fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            total_entries: AtomicUsize::new(0),
            total_memory_bytes: AtomicUsize::new(0),
            evictions: AtomicU64::new(0),
            eviction_memory_freed: AtomicUsize::new(0),
            total_lookup_time_us: AtomicU64::new(0),
            total_insert_time_us: AtomicU64::new(0),
            last_reset: parking_lot::Mutex::new(Instant::now()),
        }
    }

    /// Record a cache hit
    pub fn record_hit(&self, lookup_time: Duration) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.total_lookup_time_us
            .fetch_add(lookup_time.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_miss(&self, lookup_time: Duration) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.total_lookup_time_us
            .fetch_add(lookup_time.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record an insertion
    pub fn record_insert(&self, size_bytes: usize, insert_time: Duration) {
        self.total_entries.fetch_add(1, Ordering::Relaxed);
        self.total_memory_bytes.fetch_add(size_bytes, Ordering::Relaxed);
        self.total_insert_time_us
            .fetch_add(insert_time.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record an eviction
    pub fn record_eviction(&self, size_bytes: usize) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
        self.eviction_memory_freed.fetch_add(size_bytes, Ordering::Relaxed);
        self.total_entries.fetch_sub(1, Ordering::Relaxed);
        self.total_memory_bytes.fetch_sub(size_bytes, Ordering::Relaxed);
    }

    /// Get hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let total = hits + self.misses.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            (hits as f64 / total as f64) * 100.0
        }
    }

    /// Get average lookup time in microseconds
    pub fn avg_lookup_time_us(&self) -> f64 {
        let total_time = self.total_lookup_time_us.load(Ordering::Relaxed);
        let total_lookups = self.hits.load(Ordering::Relaxed) + self.misses.load(Ordering::Relaxed);
        if total_lookups == 0 {
            0.0
        } else {
            total_time as f64 / total_lookups as f64
        }
    }

    /// Get average insert time in microseconds
    pub fn avg_insert_time_us(&self) -> f64 {
        let total_time = self.total_insert_time_us.load(Ordering::Relaxed);
        let total_inserts = self.total_entries.load(Ordering::Relaxed);
        if total_inserts == 0 {
            0.0
        } else {
            total_time as f64 / total_inserts as f64
        }
    }

    /// Get current memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        self.total_memory_bytes.load(Ordering::Relaxed)
    }

    /// Get current number of entries
    pub fn num_entries(&self) -> usize {
        self.total_entries.load(Ordering::Relaxed)
    }

    /// Get a snapshot of all metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        let elapsed = self.last_reset.lock().elapsed();

        MetricsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            hit_rate: self.hit_rate(),
            total_entries: self.total_entries.load(Ordering::Relaxed),
            total_memory_bytes: self.total_memory_bytes.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            eviction_memory_freed: self.eviction_memory_freed.load(Ordering::Relaxed),
            avg_lookup_time_us: self.avg_lookup_time_us(),
            avg_insert_time_us: self.avg_insert_time_us(),
            elapsed_seconds: elapsed.as_secs_f64(),
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.total_entries.store(0, Ordering::Relaxed);
        self.total_memory_bytes.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.eviction_memory_freed.store(0, Ordering::Relaxed);
        self.total_lookup_time_us.store(0, Ordering::Relaxed);
        self.total_insert_time_us.store(0, Ordering::Relaxed);
        *self.last_reset.lock() = Instant::now();
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// A point-in-time snapshot of cache metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub total_entries: usize,
    pub total_memory_bytes: usize,
    pub evictions: u64,
    pub eviction_memory_freed: usize,
    pub avg_lookup_time_us: f64,
    pub avg_insert_time_us: f64,
    pub elapsed_seconds: f64,
}

impl MetricsSnapshot {
    /// Calculate requests per second
    pub fn requests_per_second(&self) -> f64 {
        let total_requests = self.hits + self.misses;
        if self.elapsed_seconds > 0.0 {
            total_requests as f64 / self.elapsed_seconds
        } else {
            0.0
        }
    }

    /// Calculate average memory per entry
    pub fn avg_memory_per_entry(&self) -> f64 {
        if self.total_entries > 0 {
            self.total_memory_bytes as f64 / self.total_entries as f64
        } else {
            0.0
        }
    }

    /// Calculate eviction rate (evictions per second)
    pub fn eviction_rate(&self) -> f64 {
        if self.elapsed_seconds > 0.0 {
            self.evictions as f64 / self.elapsed_seconds
        } else {
            0.0
        }
    }

    /// Format metrics as a human-readable report
    pub fn format_report(&self) -> String {
        format!(
            r#"
Cache Performance Report
========================
Hit Rate: {:.2}% ({} hits, {} misses)
Total Entries: {} ({:.2} MB)
Average Memory per Entry: {:.2} KB

Performance:
- Requests/sec: {:.2}
- Avg Lookup Time: {:.2} μs
- Avg Insert Time: {:.2} μs

Evictions:
- Total: {} ({:.2}/sec)
- Memory Freed: {:.2} MB

Uptime: {:.2} seconds
"#,
            self.hit_rate,
            self.hits,
            self.misses,
            self.total_entries,
            self.total_memory_bytes as f64 / (1024.0 * 1024.0),
            self.avg_memory_per_entry() / 1024.0,
            self.requests_per_second(),
            self.avg_lookup_time_us,
            self.avg_insert_time_us,
            self.evictions,
            self.eviction_rate(),
            self.eviction_memory_freed as f64 / (1024.0 * 1024.0),
            self.elapsed_seconds,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_cache_metrics() {
        let metrics = CacheMetrics::new();

        // Record some hits
        metrics.record_hit(Duration::from_micros(100));
        metrics.record_hit(Duration::from_micros(150));

        // Record some misses
        metrics.record_miss(Duration::from_micros(50));

        // Record insertions
        metrics.record_insert(1024, Duration::from_micros(200));
        metrics.record_insert(2048, Duration::from_micros(250));

        // Check metrics
        assert!((metrics.hit_rate() - 200.0 / 3.0).abs() < 0.001);
        assert_eq!(metrics.num_entries(), 2);
        assert_eq!(metrics.memory_usage_bytes(), 3072);

        // Record eviction
        metrics.record_eviction(1024);

        assert_eq!(metrics.num_entries(), 1);
        assert_eq!(metrics.memory_usage_bytes(), 2048);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.hits, 2);
        assert_eq!(snapshot.misses, 1);
        assert_eq!(snapshot.evictions, 1);
    }

    #[test]
    fn test_metrics_snapshot_report() {
        let metrics = CacheMetrics::new();

        // Simulate some activity
        for _ in 0..100 {
            metrics.record_hit(Duration::from_micros(100));
        }
        for _ in 0..20 {
            metrics.record_miss(Duration::from_micros(150));
        }
        for i in 0..50 {
            metrics.record_insert(1024 * (i + 1), Duration::from_micros(200));
        }

        thread::sleep(Duration::from_millis(100));

        let snapshot = metrics.snapshot();
        let report = snapshot.format_report();

        // Check that report contains expected sections
        assert!(report.contains("Hit Rate:"));
        assert!(report.contains("Total Entries:"));
        assert!(report.contains("Requests/sec:"));
        assert!(report.contains("Avg Lookup Time:"));
    }
}
