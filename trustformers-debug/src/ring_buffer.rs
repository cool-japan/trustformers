//! Lock-free single-producer single-consumer (SPSC) ring buffer.
//!
//! Uses power-of-2 capacity so that modulo operations reduce to bitwise AND.
//! The implementation is based on two atomic indices (`head` for writes, `tail`
//! for reads) with `Acquire`/`Release` ordering — the same well-known pattern
//! used by LMAX Disruptor and many embedded real-time systems.
//!
//! # Concurrency model
//!
//! `LockFreeRingBuffer` is **SPSC** — exactly **one** producer and **one**
//! consumer thread at a time.  The `Send + Sync` implementations are
//! deliberately provided because the buffer is safe to move across threads;
//! it is the caller's responsibility to ensure only one thread pushes and
//! one thread pops concurrently.
//!
//! # Example
//!
//! ```
//! use trustformers_debug::ring_buffer::LockFreeRingBuffer;
//! use std::sync::Arc;
//! use std::thread;
//!
//! let buf: Arc<LockFreeRingBuffer<u64>> = Arc::new(LockFreeRingBuffer::new(16));
//! let producer = Arc::clone(&buf);
//! let consumer = Arc::clone(&buf);
//!
//! let t = thread::spawn(move || {
//!     for i in 0..8_u64 {
//!         while producer.push(i).is_err() {}
//!     }
//! });
//!
//! t.join().unwrap();
//! for i in 0..8_u64 {
//!     assert_eq!(consumer.pop(), Some(i));
//! }
//! ```

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

// ─────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────

/// Errors that can arise from ring-buffer operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RingBufferError {
    /// The buffer is full; the pushed item was not stored.
    #[error("ring buffer is full (capacity {capacity})")]
    Full {
        /// Capacity of the buffer.
        capacity: usize,
    },
    /// The requested capacity is zero.
    #[error("ring buffer capacity must be at least 1")]
    ZeroCapacity,
}

// ─────────────────────────────────────────────────────────────
// LockFreeRingBuffer
// ─────────────────────────────────────────────────────────────

/// Lock-free SPSC ring buffer with power-of-2 capacity.
///
/// Internally stores items in a fixed-size boxed slice of
/// `UnsafeCell<MaybeUninit<T>>`.  Two atomic counters (`head` for the write
/// position and `tail` for the read position) are advanced using
/// `Acquire`/`Release` ordering so that item writes are visible to the reader
/// thread before the head index update becomes visible.
///
/// # Capacity rounding
///
/// The requested capacity is rounded **up** to the next power of two so that
/// `index & mask` can replace `index % capacity`.
///
/// # Example
///
/// ```
/// use trustformers_debug::ring_buffer::LockFreeRingBuffer;
///
/// let buf = LockFreeRingBuffer::<i32>::new(6);
/// // Rounded up to the next power of 2 → 8
/// assert_eq!(buf.capacity(), 8);
///
/// assert!(buf.push(1).is_ok());
/// assert_eq!(buf.pop(), Some(1));
/// assert_eq!(buf.pop(), None);
/// ```
pub struct LockFreeRingBuffer<T: Copy + Send + 'static> {
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    capacity: usize,
    mask: usize,
    /// Write cursor — points to the next slot to fill.
    head: AtomicUsize,
    /// Read cursor — points to the next slot to drain.
    tail: AtomicUsize,
}

// SAFETY: `LockFreeRingBuffer` is safe to send across threads; it is the
// caller's responsibility to uphold the SPSC invariant (one producer, one
// consumer).
unsafe impl<T: Copy + Send + 'static> Send for LockFreeRingBuffer<T> {}
// SAFETY: Internal mutation is guarded by atomic ordering; the buffer does
// not expose mutable references to its internals.
unsafe impl<T: Copy + Send + 'static> Sync for LockFreeRingBuffer<T> {}

impl<T: Copy + Send + 'static> LockFreeRingBuffer<T> {
    /// Creates a new buffer whose actual capacity is the smallest power of 2
    /// that is `>= capacity`.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::ring_buffer::LockFreeRingBuffer;
    /// let buf = LockFreeRingBuffer::<u8>::new(10);
    /// assert_eq!(buf.capacity(), 16); // rounded up to next power of 2
    /// ```
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "LockFreeRingBuffer capacity must be at least 1");
        let actual = capacity.next_power_of_two();
        // Build the backing store as a boxed slice of uninitialised cells.
        let buffer: Box<[UnsafeCell<MaybeUninit<T>>]> =
            (0..actual).map(|_| UnsafeCell::new(MaybeUninit::uninit())).collect();
        Self {
            buffer,
            capacity: actual,
            mask: actual - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Attempts to push `item` into the buffer.
    ///
    /// Returns `Err(RingBufferError::Full { … })` if the buffer is full.
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::ring_buffer::{LockFreeRingBuffer, RingBufferError};
    ///
    /// let buf = LockFreeRingBuffer::<u8>::new(2);
    /// assert!(buf.push(10).is_ok());
    /// assert!(buf.push(20).is_ok());
    /// let err = buf.push(30).unwrap_err();
    /// assert!(matches!(err, RingBufferError::Full { .. }));
    /// ```
    pub fn push(&self, item: T) -> Result<(), RingBufferError> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if head.wrapping_sub(tail) >= self.capacity {
            return Err(RingBufferError::Full { capacity: self.capacity });
        }

        let slot = head & self.mask;
        // SAFETY: `slot` is within `[0, capacity)`.  The producer owns this
        // slot exclusively because `head - tail < capacity` guarantees the
        // consumer has not yet reached it.
        unsafe {
            (*self.buffer[slot].get()).write(item);
        }

        // Release ordering: ensures the write above is visible to the reader
        // thread before the head update.
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Attempts to pop an item from the buffer.
    ///
    /// Returns `None` if the buffer is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::ring_buffer::LockFreeRingBuffer;
    ///
    /// let buf = LockFreeRingBuffer::<u8>::new(4);
    /// assert_eq!(buf.pop(), None);
    /// buf.push(99).unwrap();
    /// assert_eq!(buf.pop(), Some(99));
    /// assert_eq!(buf.pop(), None);
    /// ```
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            return None;
        }

        let slot = tail & self.mask;
        // SAFETY: `slot` is within `[0, capacity)`.  The consumer owns this
        // slot because `tail < head` guarantees the producer has already
        // written to it.
        let item = unsafe { (*self.buffer[slot].get()).assume_init_read() };

        // Release ordering: ensures the read above completes before the tail
        // update is visible to the producer.
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(item)
    }

    /// Returns the number of items currently held in the buffer.
    ///
    /// Note: this is a point-in-time snapshot; the value may change
    /// concurrently.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    /// Returns `true` if the buffer currently holds no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the (rounded-up) capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ─────────────────────────────────────────────────────────────
// StatisticsWindow – a plain Vec-backed sliding window
// ─────────────────────────────────────────────────────────────

/// A simple sliding-window buffer that retains the most-recent `capacity`
/// values and exposes statistical helpers.
///
/// Unlike `LockFreeRingBuffer` this type is **single-threaded** and designed
/// for convenience, not throughput.
///
/// # Type bound
///
/// `T: Copy + Into<f64>` so that integer and floating-point scalars can all be
/// treated uniformly.
///
/// # Example
/// ```
/// use trustformers_debug::ring_buffer::StatisticsWindow;
/// let mut w = StatisticsWindow::new(4);
/// w.push(1u32);
/// w.push(2u32);
/// w.push(3u32);
/// assert_eq!(w.mean(), Some(2.0));
/// ```
pub struct StatisticsWindow<T: Copy + Into<f64>> {
    buf: Vec<T>,
    capacity: usize,
    /// Head position in the circular backing vec.
    head: usize,
    len: usize,
}

impl<T: Copy + Into<f64>> StatisticsWindow<T> {
    /// Create a new window that retains at most `capacity` values.
    ///
    /// Panics if `capacity == 0`.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "StatisticsWindow capacity must be >= 1");
        Self {
            buf: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            len: 0,
        }
    }

    /// Push a value; if the window is full the oldest value is evicted.
    pub fn push(&mut self, value: T) {
        if self.len < self.capacity {
            self.buf.push(value);
            self.len += 1;
        } else {
            self.buf[self.head] = value;
            self.head = (self.head + 1) % self.capacity;
        }
    }

    /// Number of values currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no values are stored.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterate over the window contents in insertion order (oldest first).
    pub fn iter_ordered(&self) -> impl Iterator<Item = T> + '_ {
        // When not yet full: plain slice from 0..len.
        // When full: ring starting at head.
        let (start, count) = if self.len < self.capacity {
            (0, self.len)
        } else {
            (self.head, self.capacity)
        };
        (0..count).map(move |i| self.buf[(start + i) % self.capacity])
    }

    /// Snapshot of all current values as a `Vec<f64>`.
    fn as_f64_vec(&self) -> Vec<f64> {
        self.iter_ordered().map(|v| v.into()).collect()
    }

    /// Mean of the current window contents.
    pub fn mean(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        let vals = self.as_f64_vec();
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }

    /// Population standard deviation of the current window.
    ///
    /// Returns `None` when fewer than 2 values are stored.
    pub fn std_dev(&self) -> Option<f64> {
        if self.len < 2 {
            return None;
        }
        let mean = self.mean()?;
        let vals = self.as_f64_vec();
        let variance = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
        Some(variance.sqrt())
    }

    /// Minimum value in the current window.
    pub fn min(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        // We compare as f64 because T may not implement Ord.
        let mut best = self.buf[0];
        let mut best_f: f64 = best.into();
        for v in self.iter_ordered() {
            let vf: f64 = v.into();
            if vf < best_f {
                best = v;
                best_f = vf;
            }
        }
        Some(best)
    }

    /// Maximum value in the current window.
    pub fn max(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let mut best = self.buf[0];
        let mut best_f: f64 = best.into();
        for v in self.iter_ordered() {
            let vf: f64 = v.into();
            if vf > best_f {
                best = v;
                best_f = vf;
            }
        }
        Some(best)
    }

    /// Approximate the `p`-th percentile (0.0–100.0) via a sorted copy.
    ///
    /// Uses nearest-rank method.
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        let mut sorted = self.as_f64_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p_clamped = p.clamp(0.0, 100.0);
        let rank = (p_clamped / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        Some(sorted[rank.min(sorted.len() - 1)])
    }

    /// Mean of the last `window` values (or all values if fewer available).
    pub fn windowed_mean(&self, window: usize) -> Option<f64> {
        if self.is_empty() || window == 0 {
            return None;
        }
        let total = self.len;
        let n = window.min(total);
        // Collect last `n` values (most-recent end of ordered sequence).
        let vals: Vec<f64> = self.iter_ordered().skip(total - n).map(|v| v.into()).collect();
        if vals.is_empty() {
            return None;
        }
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

// ─────────────────────────────────────────────────────────────
// TimestampedValue & TimestampedRingBuffer
// ─────────────────────────────────────────────────────────────

/// A value paired with a nanosecond-resolution timestamp.
#[derive(Debug, Clone, Copy)]
pub struct TimestampedValue<T: Copy> {
    pub value: T,
    /// Nanoseconds since an arbitrary epoch (caller-defined, e.g. Unix epoch).
    pub timestamp_ns: u64,
}

/// An SPSC ring buffer that stores [`TimestampedValue`] entries and exposes
/// time-range queries and throughput estimation.
///
/// Internally backed by a `StatisticsWindow<f64>` for the values and a
/// separate `Vec`-based circular buffer for the full `TimestampedValue` records.
pub struct TimestampedRingBuffer<T: Copy> {
    /// Circular backing store (index = 0 means slot 0 in the array).
    buf: Vec<TimestampedValue<T>>,
    capacity: usize,
    head: usize,
    len: usize,
}

impl<T: Copy> TimestampedRingBuffer<T> {
    /// Create a new buffer with the given capacity.
    ///
    /// Panics if `capacity == 0`.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "TimestampedRingBuffer capacity must be >= 1");
        // We can't initialise MaybeUninit here without unsafe, so we use a
        // sentinel-free approach: track length explicitly.
        // Backing store is initialised lazily via push.
        Self {
            buf: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            len: 0,
        }
    }

    /// Push a new `(value, timestamp_ns)` pair.
    ///
    /// If the buffer is full the oldest entry is evicted.
    pub fn push_now(&mut self, value: T, time_ns: u64) {
        let entry = TimestampedValue { value, timestamp_ns: time_ns };
        if self.len < self.capacity {
            self.buf.push(entry);
            self.len += 1;
        } else {
            self.buf[self.head] = entry;
            self.head = (self.head + 1) % self.capacity;
        }
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterate over all stored entries in insertion order (oldest first).
    pub fn iter_ordered(&self) -> impl Iterator<Item = TimestampedValue<T>> + '_ {
        let (start, count) = if self.len < self.capacity {
            (0, self.len)
        } else {
            (self.head, self.capacity)
        };
        (0..count).map(move |i| self.buf[(start + i) % self.capacity])
    }

    /// Estimate throughput as events per second over the entire stored window.
    ///
    /// Returns 0.0 when fewer than 2 entries are stored or the time span is
    /// zero.
    pub fn rate_per_sec(&self) -> f64 {
        if self.len < 2 {
            return 0.0;
        }
        let oldest = self.oldest_timestamp().unwrap_or(0);
        let newest = self.newest_timestamp().unwrap_or(0);
        let span_ns = newest.saturating_sub(oldest);
        if span_ns == 0 {
            return 0.0;
        }
        (self.len as f64 - 1.0) / (span_ns as f64 * 1e-9)
    }

    /// Return all values whose timestamps fall in `[start_ns, end_ns]`
    /// (inclusive on both ends).
    pub fn values_in_range(&self, start_ns: u64, end_ns: u64) -> Vec<T> {
        self.iter_ordered()
            .filter(|e| e.timestamp_ns >= start_ns && e.timestamp_ns <= end_ns)
            .map(|e| e.value)
            .collect()
    }

    /// The timestamp of the oldest retained entry.
    pub fn oldest_timestamp(&self) -> Option<u64> {
        self.iter_ordered().next().map(|e| e.timestamp_ns)
    }

    /// The timestamp of the most-recently added entry.
    pub fn newest_timestamp(&self) -> Option<u64> {
        self.iter_ordered().last().map(|e| e.timestamp_ns)
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_capacity_rounds_up_to_power_of_two() {
        let buf = LockFreeRingBuffer::<u8>::new(5);
        assert_eq!(buf.capacity(), 8);

        let buf2 = LockFreeRingBuffer::<u8>::new(8);
        assert_eq!(buf2.capacity(), 8);

        let buf3 = LockFreeRingBuffer::<u8>::new(9);
        assert_eq!(buf3.capacity(), 16);
    }

    #[test]
    fn test_push_and_pop_basic() {
        let buf = LockFreeRingBuffer::<u32>::new(4);
        assert_eq!(buf.pop(), None);
        buf.push(1).unwrap();
        buf.push(2).unwrap();
        buf.push(3).unwrap();
        assert_eq!(buf.pop(), Some(1));
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn test_full_buffer_returns_error() {
        let buf = LockFreeRingBuffer::<u32>::new(2);
        buf.push(10).unwrap();
        buf.push(20).unwrap();
        let err = buf.push(30).unwrap_err();
        assert!(matches!(err, RingBufferError::Full { capacity: 2 }));
    }

    #[test]
    fn test_len_and_is_empty() {
        let buf = LockFreeRingBuffer::<u8>::new(4);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        buf.push(1).unwrap();
        assert!(!buf.is_empty());
        assert_eq!(buf.len(), 1);
        buf.push(2).unwrap();
        assert_eq!(buf.len(), 2);
        buf.pop();
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_wrap_around() {
        let buf = LockFreeRingBuffer::<u32>::new(4);
        // Fill
        buf.push(1).unwrap();
        buf.push(2).unwrap();
        buf.push(3).unwrap();
        buf.push(4).unwrap();
        // Drain half
        assert_eq!(buf.pop(), Some(1));
        assert_eq!(buf.pop(), Some(2));
        // Push two more — exercises wrap-around
        buf.push(5).unwrap();
        buf.push(6).unwrap();
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), Some(4));
        assert_eq!(buf.pop(), Some(5));
        assert_eq!(buf.pop(), Some(6));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn test_concurrent_spsc() {
        let buf: Arc<LockFreeRingBuffer<u64>> = Arc::new(LockFreeRingBuffer::new(64));
        let producer = Arc::clone(&buf);
        let consumer = Arc::clone(&buf);

        const N: u64 = 1000;

        let producer_thread = thread::spawn(move || {
            let mut sent = 0u64;
            while sent < N {
                if producer.push(sent).is_ok() {
                    sent += 1;
                }
            }
        });

        let consumer_thread = thread::spawn(move || {
            let mut received = Vec::with_capacity(N as usize);
            while received.len() < N as usize {
                if let Some(v) = consumer.pop() {
                    received.push(v);
                }
            }
            received
        });

        producer_thread.join().unwrap();
        let received = consumer_thread.join().unwrap();

        assert_eq!(received.len(), N as usize);
        for (i, &v) in received.iter().enumerate() {
            assert_eq!(v, i as u64);
        }
    }

    #[test]
    fn test_capacity_one() {
        let buf = LockFreeRingBuffer::<u8>::new(1);
        assert_eq!(buf.capacity(), 1);
        buf.push(42).unwrap();
        assert!(buf.push(99).is_err());
        assert_eq!(buf.pop(), Some(42));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn test_f32_elements() {
        let buf = LockFreeRingBuffer::<f32>::new(8);
        buf.push(1.5_f32).unwrap();
        buf.push(2.5_f32).unwrap();
        assert!((buf.pop().unwrap() - 1.5).abs() < 1e-6);
        assert!((buf.pop().unwrap() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_concurrent_ping_pong() {
        // Producer and consumer ping-pong many small bursts
        let buf: Arc<LockFreeRingBuffer<u32>> = Arc::new(LockFreeRingBuffer::new(32));
        let p = Arc::clone(&buf);
        let c = Arc::clone(&buf);

        const ITERS: u32 = 2_000;

        let prod = thread::spawn(move || {
            for i in 0..ITERS {
                while p.push(i).is_err() {
                    thread::yield_now();
                }
            }
        });

        let cons = thread::spawn(move || {
            let mut count = 0u32;
            while count < ITERS {
                if c.pop().is_some() {
                    count += 1;
                }
            }
            count
        });

        prod.join().unwrap();
        assert_eq!(cons.join().unwrap(), ITERS);
    }

    #[test]
    fn test_multiple_wrap_arounds() {
        let buf = LockFreeRingBuffer::<u64>::new(4);
        for round in 0..10u64 {
            for i in 0..4u64 {
                buf.push(round * 4 + i).unwrap();
            }
            for i in 0..4u64 {
                assert_eq!(buf.pop(), Some(round * 4 + i));
            }
        }
    }

    // ── StatisticsWindow ────────────────────────────────────────────────────

    #[test]
    fn test_statistics_window_mean_basic() {
        let mut w = StatisticsWindow::new(8);
        w.push(1u32);
        w.push(2u32);
        w.push(3u32);
        let m = w.mean().unwrap();
        assert!((m - 2.0).abs() < 1e-9, "mean={}", m);
    }

    #[test]
    fn test_statistics_window_empty_mean_returns_none() {
        let w: StatisticsWindow<u32> = StatisticsWindow::new(4);
        assert!(w.mean().is_none());
    }

    #[test]
    fn test_statistics_window_eviction() {
        // Capacity 3: after 4 pushes oldest (1) is evicted.
        let mut w = StatisticsWindow::new(3);
        w.push(1u32);
        w.push(2u32);
        w.push(3u32);
        w.push(4u32); // evicts 1
        assert_eq!(w.len(), 3);
        let vals: Vec<u32> = w.iter_ordered().collect();
        assert_eq!(vals, vec![2, 3, 4]);
    }

    #[test]
    fn test_statistics_window_std_dev_constant() {
        let mut w = StatisticsWindow::new(5);
        for _ in 0..5 {
            w.push(7u32);
        }
        let s = w.std_dev().unwrap();
        assert!(s < 1e-9, "std of constant values should be 0, got {}", s);
    }

    #[test]
    fn test_statistics_window_std_dev_two_values() {
        let mut w = StatisticsWindow::new(4);
        w.push(0u32);
        w.push(4u32);
        // Sample std: sqrt(((0-2)^2 + (4-2)^2) / 1) = sqrt(8) ≈ 2.828
        let s = w.std_dev().unwrap();
        assert!((s - (8.0_f64).sqrt()).abs() < 1e-6, "std={}", s);
    }

    #[test]
    fn test_statistics_window_min_max() {
        let mut w = StatisticsWindow::new(8);
        w.push(5u32);
        w.push(2u32);
        w.push(9u32);
        w.push(1u32);
        assert_eq!(w.min(), Some(1u32));
        assert_eq!(w.max(), Some(9u32));
    }

    #[test]
    fn test_statistics_window_min_max_empty() {
        let w: StatisticsWindow<u32> = StatisticsWindow::new(4);
        assert!(w.min().is_none());
        assert!(w.max().is_none());
    }

    #[test]
    fn test_statistics_window_percentile_median() {
        let mut w = StatisticsWindow::new(10);
        for i in 1u32..=9 {
            w.push(i);
        }
        // Sorted: 1..9, median = 5th element (idx 4) = 5.
        let p50 = w.percentile(50.0).unwrap();
        assert!((p50 - 5.0).abs() < 1.5, "p50={}", p50);
    }

    #[test]
    fn test_statistics_window_windowed_mean_last_n() {
        let mut w = StatisticsWindow::new(10);
        for i in 1u32..=10 {
            w.push(i);
        }
        // Last 3 values: 8, 9, 10 → mean = 9.
        let wm = w.windowed_mean(3).unwrap();
        assert!((wm - 9.0).abs() < 1e-9, "windowed_mean={}", wm);
    }

    #[test]
    fn test_statistics_window_windowed_mean_larger_than_len() {
        let mut w = StatisticsWindow::new(10);
        w.push(2u32);
        w.push(4u32);
        // window=5 but only 2 values → falls back to all.
        let wm = w.windowed_mean(5).unwrap();
        assert!((wm - 3.0).abs() < 1e-9, "windowed_mean={}", wm);
    }

    #[test]
    fn test_statistics_window_windowed_mean_zero_window() {
        let mut w = StatisticsWindow::new(4);
        w.push(1u32);
        assert!(w.windowed_mean(0).is_none());
    }

    // ── TimestampedRingBuffer ───────────────────────────────────────────────

    #[test]
    fn test_timestamped_ring_buffer_basic_push_and_len() {
        let mut tb = TimestampedRingBuffer::<u32>::new(4);
        assert!(tb.is_empty());
        tb.push_now(10, 1_000_000);
        tb.push_now(20, 2_000_000);
        assert_eq!(tb.len(), 2);
    }

    #[test]
    fn test_timestamped_ring_buffer_eviction() {
        let mut tb = TimestampedRingBuffer::<u32>::new(2);
        tb.push_now(1, 100);
        tb.push_now(2, 200);
        tb.push_now(3, 300); // evicts first
        assert_eq!(tb.len(), 2);
        let vals: Vec<u32> = tb.iter_ordered().map(|e| e.value).collect();
        assert_eq!(vals, vec![2, 3]);
    }

    #[test]
    fn test_timestamped_oldest_newest() {
        let mut tb = TimestampedRingBuffer::<u32>::new(4);
        tb.push_now(0u32, 100);
        tb.push_now(1u32, 200);
        tb.push_now(2u32, 300);
        assert_eq!(tb.oldest_timestamp(), Some(100));
        assert_eq!(tb.newest_timestamp(), Some(300));
    }

    #[test]
    fn test_timestamped_oldest_newest_empty() {
        let tb: TimestampedRingBuffer<u32> = TimestampedRingBuffer::new(4);
        assert!(tb.oldest_timestamp().is_none());
        assert!(tb.newest_timestamp().is_none());
    }

    #[test]
    fn test_timestamped_rate_per_sec() {
        let mut tb = TimestampedRingBuffer::<u32>::new(4);
        // 3 events over 2 seconds → rate = 2 / 2s = 1.0 events/s.
        // (rate = (n-1) / elapsed_s)
        tb.push_now(0, 0);
        tb.push_now(1, 1_000_000_000); // 1 s
        tb.push_now(2, 2_000_000_000); // 2 s
        let rate = tb.rate_per_sec();
        assert!((rate - 1.0).abs() < 0.01, "rate={}", rate);
    }

    #[test]
    fn test_timestamped_rate_single_entry_is_zero() {
        let mut tb = TimestampedRingBuffer::<u32>::new(4);
        tb.push_now(1, 1_000_000_000);
        assert_eq!(tb.rate_per_sec(), 0.0);
    }

    #[test]
    fn test_timestamped_values_in_range() {
        let mut tb = TimestampedRingBuffer::<u32>::new(8);
        for i in 0..8u32 {
            tb.push_now(i, i as u64 * 100);
        }
        // Range 200..=500 → timestamps 200, 300, 400, 500 → values 2,3,4,5.
        let vals = tb.values_in_range(200, 500);
        assert_eq!(vals, vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_timestamped_values_in_range_empty_result() {
        let mut tb = TimestampedRingBuffer::<u32>::new(4);
        tb.push_now(1, 100);
        tb.push_now(2, 200);
        let vals = tb.values_in_range(500, 600);
        assert!(vals.is_empty());
    }
}
