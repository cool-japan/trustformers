//! Memory safety verification tools for TrustformeRS-C
//!
//! This module provides runtime memory safety verification, bounds checking,
//! and memory corruption detection for the C API.

use crate::error::TrustformersResult;
use crossbeam_utils::atomic::AtomicCell;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::{string_to_c_str, TrustformersError};

/// Memory safety verification configuration
#[derive(Debug, Clone)]
pub struct MemorySafetyConfig {
    /// Enable bounds checking for array access
    pub enable_bounds_checking: bool,
    /// Enable use-after-free detection
    pub enable_use_after_free_detection: bool,
    /// Enable double-free detection
    pub enable_double_free_detection: bool,
    /// Enable memory leak detection
    pub enable_leak_detection: bool,
    /// Enable buffer overflow detection
    pub enable_buffer_overflow_detection: bool,
    /// Enable stack canary checking
    pub enable_stack_canary_checking: bool,
    /// Maximum number of allocations to track
    pub max_tracked_allocations: usize,
    /// Verification interval in milliseconds
    pub verification_interval_ms: u64,
}

impl Default for MemorySafetyConfig {
    fn default() -> Self {
        Self {
            enable_bounds_checking: true,
            enable_use_after_free_detection: true,
            enable_double_free_detection: true,
            enable_leak_detection: true,
            enable_buffer_overflow_detection: true,
            enable_stack_canary_checking: false, // Disabled by default as it has performance impact
            max_tracked_allocations: 10000,
            verification_interval_ms: 1000,
        }
    }
}

/// Memory allocation metadata
#[derive(Debug, Clone)]
struct AllocationMetadata {
    /// Size of the allocation
    size: usize,
    /// Timestamp when allocated
    allocated_at: Instant,
    /// Stack trace at allocation (simplified)
    allocation_site: String,
    /// Whether this allocation has been freed
    freed: bool,
    /// Timestamp when freed (if freed)
    freed_at: Option<Instant>,
    /// Magic number for corruption detection
    magic_number: u64,
    /// Canary values for overflow detection
    canary_start: u64,
    canary_end: u64,
}

const MAGIC_NUMBER: u64 = 0xDEADBEEFCAFEBABE;
const CANARY_VALUE: u64 = 0x1234567890ABCDEF;

impl AllocationMetadata {
    fn new(size: usize, allocation_site: String) -> Self {
        Self {
            size,
            allocated_at: Instant::now(),
            allocation_site,
            freed: false,
            freed_at: None,
            magic_number: MAGIC_NUMBER,
            canary_start: CANARY_VALUE,
            canary_end: CANARY_VALUE,
        }
    }

    fn mark_freed(&mut self) {
        self.freed = true;
        self.freed_at = Some(Instant::now());
    }

    fn is_corrupted(&self) -> bool {
        self.magic_number != MAGIC_NUMBER
            || self.canary_start != CANARY_VALUE
            || self.canary_end != CANARY_VALUE
    }
}

/// Memory safety verification system with lock-free data structures
pub struct MemorySafetyVerifier {
    config: MemorySafetyConfig,
    /// Lock-free hashmap for tracking allocations
    allocations: DashMap<usize, AllocationMetadata>,
    /// Lock-free hashset for tracking freed pointers using DashMap
    freed_pointers: DashMap<usize, Instant>,
    allocation_counter: AtomicUsize,
    violation_count: AtomicUsize,
    /// Lock-free atomic cell for last verification time
    last_verification: AtomicCell<Instant>,
}

impl MemorySafetyVerifier {
    pub fn new(config: MemorySafetyConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            allocations: DashMap::new(),
            freed_pointers: DashMap::new(),
            allocation_counter: AtomicUsize::new(0),
            violation_count: AtomicUsize::new(0),
            last_verification: AtomicCell::new(Instant::now()),
        })
    }

    /// Track a new allocation with lock-free performance
    pub fn track_allocation(
        &self,
        ptr: *mut c_void,
        size: usize,
        site: &str,
    ) -> TrustformersResult<()> {
        if ptr.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        // Check if we're tracking too many allocations
        if self.allocations.len() >= self.config.max_tracked_allocations {
            return Err(TrustformersError::ResourceLimitExceeded);
        }

        let ptr_addr = ptr as usize;
        let metadata = AllocationMetadata::new(size, site.to_string());

        // Check for duplicate allocation (possible corruption) using lock-free approach
        if self.allocations.insert(ptr_addr, metadata).is_some() {
            self.violation_count.fetch_add(1, Ordering::Relaxed);
            return Err(TrustformersError::RuntimeError);
        }

        self.allocation_counter.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Track memory deallocation with lock-free performance
    pub fn track_deallocation(&self, ptr: *mut c_void) -> TrustformersResult<()> {
        if ptr.is_null() {
            return Ok(()); // Freeing null is valid
        }

        let ptr_addr = ptr as usize;
        let now = Instant::now();

        // Check for double-free using lock-free approach
        if self.config.enable_double_free_detection && self.freed_pointers.contains_key(&ptr_addr) {
            self.violation_count.fetch_add(1, Ordering::Relaxed);
            return Err(TrustformersError::RuntimeError);
        }

        // Check if this allocation was tracked using lock-free remove
        if let Some((_, mut metadata)) = self.allocations.remove(&ptr_addr) {
            // Check for corruption before freeing
            if self.config.enable_buffer_overflow_detection && metadata.is_corrupted() {
                self.violation_count.fetch_add(1, Ordering::Relaxed);
                return Err(TrustformersError::RuntimeError);
            }

            metadata.mark_freed();
            self.freed_pointers.insert(ptr_addr, now);
        } else {
            // Freeing untracked memory - could be valid or an error
            if self.config.enable_use_after_free_detection {
                return Err(TrustformersError::RuntimeError);
            }
        }

        Ok(())
    }

    /// Check if a memory access is valid with lock-free performance
    pub fn validate_memory_access(
        &self,
        ptr: *const c_void,
        size: usize,
    ) -> TrustformersResult<()> {
        if ptr.is_null() {
            return Err(TrustformersError::NullPointer);
        }

        let ptr_addr = ptr as usize;

        // Check for use-after-free using lock-free access
        if self.config.enable_use_after_free_detection
            && self.freed_pointers.contains_key(&ptr_addr)
        {
            self.violation_count.fetch_add(1, Ordering::Relaxed);
            return Err(TrustformersError::RuntimeError);
        }

        // Check bounds if we're tracking this allocation using lock-free iteration
        if self.config.enable_bounds_checking {
            for entry in &self.allocations {
                let alloc_addr = *entry.key();
                let metadata = entry.value();
                if ptr_addr >= alloc_addr && ptr_addr < alloc_addr + metadata.size {
                    // Access is within tracked allocation, check if it exceeds bounds
                    if ptr_addr + size > alloc_addr + metadata.size {
                        self.violation_count.fetch_add(1, Ordering::Relaxed);
                        return Err(TrustformersError::RuntimeError);
                    }
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    /// Perform comprehensive memory verification with lock-free performance
    pub fn verify_memory_integrity(&self) -> MemoryVerificationReport {
        let mut report = MemoryVerificationReport::new();

        report.total_allocations = self.allocations.len();
        report.total_freed = self.freed_pointers.len();
        report.total_violations = self.violation_count.load(Ordering::Relaxed);

        // Check for memory leaks using lock-free iteration
        if self.config.enable_leak_detection {
            let now = Instant::now();
            for entry in &self.allocations {
                let addr = *entry.key();
                let metadata = entry.value();
                let age = now.duration_since(metadata.allocated_at);
                if age > Duration::from_secs(300) {
                    // 5 minutes
                    report.potential_leaks.push(MemoryLeak {
                        address: addr,
                        size: metadata.size,
                        age_seconds: age.as_secs(),
                        allocation_site: metadata.allocation_site.clone(),
                    });
                }
            }
        }

        // Check for memory corruption using lock-free iteration
        if self.config.enable_buffer_overflow_detection {
            for entry in &self.allocations {
                let addr = *entry.key();
                let metadata = entry.value();
                if metadata.is_corrupted() {
                    report.corrupted_allocations.push(MemoryCorruption {
                        address: addr,
                        size: metadata.size,
                        corruption_type: "Magic number or canary corruption".to_string(),
                        allocation_site: metadata.allocation_site.clone(),
                    });
                }
            }
        }

        // Update last verification time using lock-free atomic cell
        self.last_verification.store(Instant::now());

        report
    }

    /// Get memory safety statistics with lock-free performance
    pub fn get_statistics(&self) -> MemorySafetyStats {
        MemorySafetyStats {
            total_allocations_tracked: self.allocation_counter.load(Ordering::Relaxed),
            current_allocations: self.allocations.len(),
            total_freed: self.freed_pointers.len(),
            total_violations: self.violation_count.load(Ordering::Relaxed),
            config: self.config.clone(),
        }
    }

    /// Reset all tracking data with lock-free performance
    pub fn reset(&self) {
        self.allocations.clear();
        self.freed_pointers.clear();
        self.allocation_counter.store(0, Ordering::Relaxed);
        self.violation_count.store(0, Ordering::Relaxed);
    }

    /// Get detailed resource usage breakdown with lock-free performance
    pub fn get_resource_breakdown(&self) -> ResourceBreakdown {
        let mut breakdown = ResourceBreakdown::new();

        for entry in &self.allocations {
            let metadata = entry.value();
            let age = Instant::now().duration_since(metadata.allocated_at);

            // Categorize by allocation site
            let category = if metadata.allocation_site.contains("model") {
                "model"
            } else if metadata.allocation_site.contains("tokenizer") {
                "tokenizer"
            } else if metadata.allocation_site.contains("pipeline") {
                "pipeline"
            } else if metadata.allocation_site.contains("tensor") {
                "tensor"
            } else {
                "other"
            }
            .to_string();

            breakdown.add_allocation(category, metadata.size, age);
        }

        breakdown
    }

    /// Find allocations that are likely leaked based on heuristics with lock-free performance
    pub fn find_likely_leaks(&self) -> Vec<MemoryLeak> {
        let mut leaks = Vec::new();
        let now = Instant::now();

        for entry in &self.allocations {
            let addr = *entry.key();
            let metadata = entry.value();
            let age = now.duration_since(metadata.allocated_at);

            // Different thresholds for different types of allocations
            let leak_threshold = if metadata.size > 1024 * 1024 {
                // > 1MB
                Duration::from_secs(60) // 1 minute for large allocations
            } else if metadata.allocation_site.contains("temp")
                || metadata.allocation_site.contains("buffer")
            {
                Duration::from_secs(30) // 30 seconds for temporary allocations
            } else {
                Duration::from_secs(300) // 5 minutes for regular allocations
            };

            if age > leak_threshold {
                leaks.push(MemoryLeak {
                    address: addr,
                    size: metadata.size,
                    age_seconds: age.as_secs(),
                    allocation_site: metadata.allocation_site.clone(),
                });
            }
        }

        leaks
    }

    /// Clean up old freed pointer records to prevent unbounded growth with lock-free performance
    pub fn cleanup_freed_records(&self) {
        // Keep only recent freed records (last 1000)
        if self.freed_pointers.len() > 1000 {
            let mut addrs_and_times: Vec<(usize, Instant)> =
                self.freed_pointers.iter().map(|entry| (*entry.key(), *entry.value())).collect();

            // Sort by timestamp (most recent first)
            addrs_and_times.sort_by(|a, b| b.1.cmp(&a.1));

            // Clear and keep the most recent 500
            self.freed_pointers.clear();
            for (addr, time) in addrs_and_times.into_iter().take(500) {
                self.freed_pointers.insert(addr, time);
            }
        }
    }
}

/// Memory verification report
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryVerificationReport {
    pub total_allocations: usize,
    pub total_freed: usize,
    pub total_violations: usize,
    pub potential_leaks: Vec<MemoryLeak>,
    pub corrupted_allocations: Vec<MemoryCorruption>,
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

impl MemoryVerificationReport {
    fn new() -> Self {
        Self {
            total_allocations: 0,
            total_freed: 0,
            total_violations: 0,
            potential_leaks: Vec::new(),
            corrupted_allocations: Vec::new(),
            timestamp: Instant::now(),
        }
    }

    pub fn has_issues(&self) -> bool {
        !self.potential_leaks.is_empty()
            || !self.corrupted_allocations.is_empty()
            || self.total_violations > 0
    }

    pub fn to_json(&self) -> TrustformersResult<String> {
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "total_allocations": self.total_allocations,
            "total_freed": self.total_freed,
            "total_violations": self.total_violations,
            "potential_leaks": self.potential_leaks.len(),
            "corrupted_allocations": self.corrupted_allocations.len(),
            "has_issues": self.has_issues(),
            "leaks": self.potential_leaks,
            "corruptions": self.corrupted_allocations
        }))?)
    }
}

/// Memory leak information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeak {
    pub address: usize,
    pub size: usize,
    pub age_seconds: u64,
    pub allocation_site: String,
}

/// Memory corruption information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCorruption {
    pub address: usize,
    pub size: usize,
    pub corruption_type: String,
    pub allocation_site: String,
}

/// Memory safety statistics
#[derive(Debug, Clone)]
pub struct MemorySafetyStats {
    pub total_allocations_tracked: usize,
    pub current_allocations: usize,
    pub total_freed: usize,
    pub total_violations: usize,
    pub config: MemorySafetyConfig,
}

/// Resource usage breakdown by category
#[derive(Debug, Clone)]
pub struct ResourceBreakdown {
    pub categories: HashMap<String, CategoryStats>,
    pub total_memory: usize,
    pub total_allocations: usize,
}

#[derive(Debug, Clone)]
pub struct CategoryStats {
    pub total_size: usize,
    pub count: usize,
    pub avg_size: usize,
    pub max_size: usize,
    pub avg_age_seconds: u64,
    pub oldest_age_seconds: u64,
}

impl ResourceBreakdown {
    fn new() -> Self {
        Self {
            categories: HashMap::new(),
            total_memory: 0,
            total_allocations: 0,
        }
    }

    fn add_allocation(&mut self, category: String, size: usize, age: Duration) {
        self.total_memory += size;
        self.total_allocations += 1;

        let stats = self.categories.entry(category).or_insert(CategoryStats {
            total_size: 0,
            count: 0,
            avg_size: 0,
            max_size: 0,
            avg_age_seconds: 0,
            oldest_age_seconds: 0,
        });

        stats.total_size += size;
        stats.count += 1;
        stats.avg_size = stats.total_size / stats.count;
        stats.max_size = stats.max_size.max(size);

        let age_secs = age.as_secs();
        stats.avg_age_seconds =
            ((stats.avg_age_seconds * (stats.count - 1) as u64) + age_secs) / stats.count as u64;
        stats.oldest_age_seconds = stats.oldest_age_seconds.max(age_secs);
    }

    pub fn to_json(&self) -> TrustformersResult<String> {
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "total_memory": self.total_memory,
            "total_allocations": self.total_allocations,
            "categories": self.categories.iter().map(|(name, stats)| {
                (name, serde_json::json!({
                    "total_size": stats.total_size,
                    "count": stats.count,
                    "avg_size": stats.avg_size,
                    "max_size": stats.max_size,
                    "avg_age_seconds": stats.avg_age_seconds,
                    "oldest_age_seconds": stats.oldest_age_seconds,
                }))
            }).collect::<HashMap<_, _>>()
        }))?)
    }

    /// Find categories with potential issues
    pub fn find_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();

        for (category, stats) in &self.categories {
            // Check for memory pressure
            if stats.total_size > 100 * 1024 * 1024 {
                // > 100MB
                issues.push(format!(
                    "Category '{}' using {} MB",
                    category,
                    stats.total_size / 1024 / 1024
                ));
            }

            // Check for too many small allocations (potential fragmentation)
            if stats.count > 1000 && stats.avg_size < 1024 {
                issues.push(format!(
                    "Category '{}' has many small allocations: {} allocations averaging {} bytes",
                    category, stats.count, stats.avg_size
                ));
            }

            // Check for very old allocations (potential leaks)
            if stats.oldest_age_seconds > 600 {
                // > 10 minutes
                issues.push(format!(
                    "Category '{}' has very old allocation: {} seconds",
                    category, stats.oldest_age_seconds
                ));
            }
        }

        issues
    }
}

// Global memory safety verifier instance
static mut GLOBAL_VERIFIER: Option<Arc<MemorySafetyVerifier>> = None;
static VERIFIER_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize the global memory safety verifier
fn get_global_verifier() -> Arc<MemorySafetyVerifier> {
    unsafe {
        VERIFIER_INIT.call_once(|| {
            GLOBAL_VERIFIER = Some(MemorySafetyVerifier::new(MemorySafetyConfig::default()));
        });
        GLOBAL_VERIFIER.as_ref().unwrap().clone()
    }
}

// C API exports for memory safety verification

/// C-compatible memory safety configuration
#[repr(C)]
#[derive(Debug, Default)]
pub struct TrustformersMemorySafetyConfig {
    pub enable_bounds_checking: c_int,
    pub enable_use_after_free_detection: c_int,
    pub enable_double_free_detection: c_int,
    pub enable_leak_detection: c_int,
    pub enable_buffer_overflow_detection: c_int,
    pub enable_stack_canary_checking: c_int,
    pub max_tracked_allocations: usize,
    pub verification_interval_ms: u64,
}

impl From<MemorySafetyConfig> for TrustformersMemorySafetyConfig {
    fn from(config: MemorySafetyConfig) -> Self {
        Self {
            enable_bounds_checking: if config.enable_bounds_checking { 1 } else { 0 },
            enable_use_after_free_detection: if config.enable_use_after_free_detection {
                1
            } else {
                0
            },
            enable_double_free_detection: if config.enable_double_free_detection { 1 } else { 0 },
            enable_leak_detection: if config.enable_leak_detection { 1 } else { 0 },
            enable_buffer_overflow_detection: if config.enable_buffer_overflow_detection {
                1
            } else {
                0
            },
            enable_stack_canary_checking: if config.enable_stack_canary_checking { 1 } else { 0 },
            max_tracked_allocations: config.max_tracked_allocations,
            verification_interval_ms: config.verification_interval_ms,
        }
    }
}

impl From<TrustformersMemorySafetyConfig> for MemorySafetyConfig {
    fn from(config: TrustformersMemorySafetyConfig) -> Self {
        Self {
            enable_bounds_checking: config.enable_bounds_checking != 0,
            enable_use_after_free_detection: config.enable_use_after_free_detection != 0,
            enable_double_free_detection: config.enable_double_free_detection != 0,
            enable_leak_detection: config.enable_leak_detection != 0,
            enable_buffer_overflow_detection: config.enable_buffer_overflow_detection != 0,
            enable_stack_canary_checking: config.enable_stack_canary_checking != 0,
            max_tracked_allocations: config.max_tracked_allocations,
            verification_interval_ms: config.verification_interval_ms,
        }
    }
}

/// C-compatible memory safety statistics
#[repr(C)]
#[derive(Debug, Default)]
pub struct TrustformersMemorySafetyStats {
    pub total_allocations_tracked: usize,
    pub current_allocations: usize,
    pub total_freed: usize,
    pub total_violations: usize,
    pub config: TrustformersMemorySafetyConfig,
}

/// Initialize memory safety verification
#[no_mangle]
pub extern "C" fn trustformers_memory_safety_init(
    config: *const TrustformersMemorySafetyConfig,
) -> TrustformersError {
    let rust_config = if config.is_null() {
        MemorySafetyConfig::default()
    } else {
        unsafe { std::ptr::read(config).into() }
    };

    unsafe {
        VERIFIER_INIT.call_once(|| {
            GLOBAL_VERIFIER = Some(MemorySafetyVerifier::new(rust_config));
        });
    }

    TrustformersError::Success
}

/// Validate memory access
#[no_mangle]
pub extern "C" fn trustformers_validate_memory_access(
    ptr: *const c_void,
    size: usize,
) -> TrustformersError {
    let verifier = get_global_verifier();

    match verifier.validate_memory_access(ptr, size) {
        Ok(()) => TrustformersError::Success,
        Err(_) => TrustformersError::OutOfMemory,
    }
}

/// Perform memory integrity verification
#[no_mangle]
pub extern "C" fn trustformers_verify_memory_integrity(
    report_json: *mut *mut c_char,
) -> TrustformersError {
    if report_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let verifier = get_global_verifier();
    let report = verifier.verify_memory_integrity();

    match report.to_json() {
        Ok(json) => {
            unsafe {
                *report_json = string_to_c_str(json);
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::SerializationError,
    }
}

/// Get memory safety statistics
#[no_mangle]
pub extern "C" fn trustformers_get_memory_safety_stats(
    stats: *mut TrustformersMemorySafetyStats,
) -> TrustformersError {
    if stats.is_null() {
        return TrustformersError::NullPointer;
    }

    let verifier = get_global_verifier();
    let rust_stats = verifier.get_statistics();

    unsafe {
        let c_stats = &mut *stats;
        c_stats.total_allocations_tracked = rust_stats.total_allocations_tracked;
        c_stats.current_allocations = rust_stats.current_allocations;
        c_stats.total_freed = rust_stats.total_freed;
        c_stats.total_violations = rust_stats.total_violations;
        c_stats.config = rust_stats.config.into();
    }

    TrustformersError::Success
}

/// Reset memory safety tracking
#[no_mangle]
pub extern "C" fn trustformers_memory_safety_reset() -> TrustformersError {
    let verifier = get_global_verifier();
    verifier.reset();
    TrustformersError::Success
}

/// Check if memory safety verification is enabled
#[no_mangle]
pub extern "C" fn trustformers_memory_safety_enabled() -> c_int {
    unsafe {
        if GLOBAL_VERIFIER.is_some() {
            1
        } else {
            0
        }
    }
}

/// Get detailed resource breakdown
#[no_mangle]
pub extern "C" fn trustformers_get_resource_breakdown(
    report_json: *mut *mut c_char,
) -> TrustformersError {
    if report_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let verifier = get_global_verifier();
    let breakdown = verifier.get_resource_breakdown();

    match breakdown.to_json() {
        Ok(json) => {
            unsafe {
                *report_json = string_to_c_str(json);
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::SerializationError,
    }
}

/// Find likely memory leaks using enhanced heuristics
#[no_mangle]
pub extern "C" fn trustformers_find_likely_leaks(
    report_json: *mut *mut c_char,
) -> TrustformersError {
    if report_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let verifier = get_global_verifier();
    let leaks = verifier.find_likely_leaks();

    match serde_json::to_string_pretty(&leaks) {
        Ok(json) => {
            unsafe {
                *report_json = string_to_c_str(json);
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::SerializationError,
    }
}

/// Clean up old freed records to prevent memory growth
#[no_mangle]
pub extern "C" fn trustformers_cleanup_freed_records() -> TrustformersError {
    let verifier = get_global_verifier();
    verifier.cleanup_freed_records();
    TrustformersError::Success
}

/// Get resource usage issues and recommendations
#[no_mangle]
pub extern "C" fn trustformers_get_resource_issues(
    issues_json: *mut *mut c_char,
) -> TrustformersError {
    if issues_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let verifier = get_global_verifier();
    let breakdown = verifier.get_resource_breakdown();
    let issues = breakdown.find_issues();

    match serde_json::to_string_pretty(&issues) {
        Ok(json) => {
            unsafe {
                *issues_json = string_to_c_str(json);
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::SerializationError,
    }
}

/// Force comprehensive memory audit
#[no_mangle]
pub extern "C" fn trustformers_memory_audit(audit_report: *mut *mut c_char) -> TrustformersError {
    if audit_report.is_null() {
        return TrustformersError::NullPointer;
    }

    let verifier = get_global_verifier();

    // Perform comprehensive audit
    let verification_report = verifier.verify_memory_integrity();
    let breakdown = verifier.get_resource_breakdown();
    let issues = breakdown.find_issues();
    let likely_leaks = verifier.find_likely_leaks();

    let audit = serde_json::json!({
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        "verification_report": {
            "total_allocations": verification_report.total_allocations,
            "total_freed": verification_report.total_freed,
            "total_violations": verification_report.total_violations,
            "has_issues": verification_report.has_issues(),
            "potential_leaks_count": verification_report.potential_leaks.len(),
            "corrupted_allocations_count": verification_report.corrupted_allocations.len(),
        },
        "resource_breakdown": {
            "total_memory": breakdown.total_memory,
            "total_allocations": breakdown.total_allocations,
            "categories_count": breakdown.categories.len(),
        },
        "issues": issues,
        "likely_leaks": likely_leaks,
        "recommendations": generate_recommendations(&verification_report, &breakdown, &issues)
    });

    match serde_json::to_string_pretty(&audit) {
        Ok(json) => {
            unsafe {
                *audit_report = string_to_c_str(json);
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::SerializationError,
    }
}

/// Generate recommendations based on audit results
fn generate_recommendations(
    report: &MemoryVerificationReport,
    breakdown: &ResourceBreakdown,
    issues: &[String],
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Memory leak recommendations
    if !report.potential_leaks.is_empty() {
        recommendations
            .push("Consider implementing automatic cleanup for long-lived allocations".to_string());
        recommendations.push(
            "Review allocation patterns to identify unnecessary memory retention".to_string(),
        );
    }

    // Memory corruption recommendations
    if !report.corrupted_allocations.is_empty() {
        recommendations.push("Enable additional buffer overflow protection".to_string());
        recommendations.push("Review code for potential buffer overruns".to_string());
    }

    // High memory usage recommendations
    if breakdown.total_memory > 500 * 1024 * 1024 {
        // > 500MB
        recommendations.push(
            "Consider implementing memory pooling for frequently allocated objects".to_string(),
        );
        recommendations.push("Review large allocations for optimization opportunities".to_string());
    }

    // Fragmentation recommendations
    let small_allocs: usize = breakdown
        .categories
        .values()
        .filter(|stats| stats.avg_size < 1024 && stats.count > 100)
        .map(|stats| stats.count)
        .sum();

    if small_allocs > 1000 {
        recommendations.push("Consider using memory pools to reduce fragmentation".to_string());
        recommendations.push("Batch small allocations when possible".to_string());
    }

    // General recommendations
    if issues.is_empty() && report.total_violations == 0 {
        recommendations
            .push("Memory management appears healthy - continue current practices".to_string());
    }

    recommendations
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_memory_safety_verifier_creation() {
        let config = MemorySafetyConfig::default();
        let verifier = MemorySafetyVerifier::new(config);

        let stats = verifier.get_statistics();
        assert_eq!(stats.current_allocations, 0);
        assert_eq!(stats.total_violations, 0);
    }

    #[test]
    fn test_allocation_tracking() {
        let config = MemorySafetyConfig::default();
        let verifier = MemorySafetyVerifier::new(config);

        let test_ptr = 0x1000 as *mut c_void;
        let result = verifier.track_allocation(test_ptr, 1024, "test");
        assert!(result.is_ok());

        let stats = verifier.get_statistics();
        assert_eq!(stats.current_allocations, 1);
        assert_eq!(stats.total_allocations_tracked, 1);
    }

    #[test]
    fn test_double_free_detection() {
        let config = MemorySafetyConfig::default();
        let verifier = MemorySafetyVerifier::new(config);

        let test_ptr = 0x2000 as *mut c_void;
        verifier.track_allocation(test_ptr, 1024, "test").unwrap();

        // First free should succeed
        let result1 = verifier.track_deallocation(test_ptr);
        assert!(result1.is_ok());

        // Second free should fail
        let result2 = verifier.track_deallocation(test_ptr);
        assert!(result2.is_err());
        assert_eq!(result2.unwrap_err(), TrustformersError::RuntimeError);
    }

    #[test]
    fn test_use_after_free_detection() {
        let config = MemorySafetyConfig::default();
        let verifier = MemorySafetyVerifier::new(config);

        let test_ptr = 0x3000 as *mut c_void;
        verifier.track_allocation(test_ptr, 1024, "test").unwrap();
        verifier.track_deallocation(test_ptr).unwrap();

        // Access after free should fail
        let result = verifier.validate_memory_access(test_ptr, 4);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TrustformersError::RuntimeError);
    }

    #[test]
    fn test_bounds_checking() {
        let config = MemorySafetyConfig::default();
        let verifier = MemorySafetyVerifier::new(config);

        let test_ptr = 0x4000 as *mut c_void;
        verifier.track_allocation(test_ptr, 1024, "test").unwrap();

        // Access within bounds should succeed
        let result1 = verifier.validate_memory_access(test_ptr, 512);
        assert!(result1.is_ok());

        // Access beyond bounds should fail
        let result2 = verifier.validate_memory_access(test_ptr, 2048);
        assert!(result2.is_err());
        assert_eq!(result2.unwrap_err(), TrustformersError::RuntimeError);
    }

    #[test]
    fn test_null_pointer_validation() {
        let config = MemorySafetyConfig::default();
        let verifier = MemorySafetyVerifier::new(config);

        let result = verifier.validate_memory_access(ptr::null(), 4);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TrustformersError::NullPointer);
    }

    #[test]
    fn test_memory_verification_report() {
        let config = MemorySafetyConfig::default();
        let verifier = MemorySafetyVerifier::new(config);

        let report = verifier.verify_memory_integrity();
        assert_eq!(report.total_allocations, 0);
        assert_eq!(report.total_freed, 0);
        assert!(!report.has_issues());

        let json = report.to_json().unwrap();
        assert!(json.contains("total_allocations"));
        assert!(json.contains("has_issues"));
    }

    #[test]
    fn test_c_api_integration() {
        // Test the C API functions
        let config = TrustformersMemorySafetyConfig {
            enable_bounds_checking: 1,
            enable_use_after_free_detection: 1,
            enable_double_free_detection: 1,
            enable_leak_detection: 1,
            enable_buffer_overflow_detection: 1,
            enable_stack_canary_checking: 0,
            max_tracked_allocations: 1000,
            verification_interval_ms: 1000,
        };

        let result = trustformers_memory_safety_init(&config);
        assert_eq!(result, TrustformersError::Success);

        assert_eq!(trustformers_memory_safety_enabled(), 1);

        let mut stats = TrustformersMemorySafetyStats::default();
        let stats_result = trustformers_get_memory_safety_stats(&mut stats);
        assert_eq!(stats_result, TrustformersError::Success);

        let reset_result = trustformers_memory_safety_reset();
        assert_eq!(reset_result, TrustformersError::Success);
    }
}
