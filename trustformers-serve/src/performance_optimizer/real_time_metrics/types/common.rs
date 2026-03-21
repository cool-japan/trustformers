//! Common utility types for real-time metrics system
//!
//! This module provides shared utility types used across the real-time metrics system.

use std::sync::atomic::{AtomicU32, Ordering};

/// Atomic wrapper for f32 values using u32 bit manipulation
#[derive(Debug)]
pub struct AtomicF32 {
    inner: AtomicU32,
}

impl AtomicF32 {
    pub fn new(value: f32) -> Self {
        Self {
            inner: AtomicU32::new(value.to_bits()),
        }
    }

    pub fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(self.inner.load(order))
    }

    pub fn store(&self, value: f32, order: Ordering) {
        self.inner.store(value.to_bits(), order)
    }

    pub fn swap(&self, value: f32, order: Ordering) -> f32 {
        f32::from_bits(self.inner.swap(value.to_bits(), order))
    }

    pub fn compare_exchange(
        &self,
        current: f32,
        new: f32,
        success: Ordering,
        failure: Ordering,
    ) -> Result<f32, f32> {
        match self.inner.compare_exchange(current.to_bits(), new.to_bits(), success, failure) {
            Ok(old) => Ok(f32::from_bits(old)),
            Err(old) => Err(f32::from_bits(old)),
        }
    }

    /// Atomically adds a value to the current value and returns the previous value
    pub fn fetch_add(&self, value: f32, order: Ordering) -> f32 {
        loop {
            let current = self.load(Ordering::Relaxed);
            let new = current + value;
            match self.compare_exchange(current, new, order, Ordering::Relaxed) {
                Ok(prev) => return prev,
                Err(_) => continue, // Retry if compare_exchange failed
            }
        }
    }

    /// Atomically subtracts a value from the current value and returns the previous value
    pub fn fetch_sub(&self, value: f32, order: Ordering) -> f32 {
        self.fetch_add(-value, order)
    }
}

impl Default for AtomicF32 {
    fn default() -> Self {
        Self::new(0.0)
    }
}
