//! Memory leak detection tests
//!
//! These tests help detect memory leaks by monitoring memory usage patterns
//! during various operations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use trustformers_core::{tensor::Tensor, layers::{Layer, Linear}};

/// Custom allocator that tracks memory usage
struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        DEALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

/// Get current memory usage (allocated - deallocated)
fn get_memory_usage() -> isize {
    let allocated = ALLOCATED.load(Ordering::Relaxed) as isize;
    let deallocated = DEALLOCATED.load(Ordering::Relaxed) as isize;
    allocated - deallocated
}

/// Reset memory tracking counters
fn reset_memory_tracking() {
    ALLOCATED.store(0, Ordering::Relaxed);
    DEALLOCATED.store(0, Ordering::Relaxed);
}

/// Run a function multiple times and check for memory leaks
fn check_memory_leak<F: Fn()>(name: &str, iterations: usize, f: F) {
    println!("Testing memory leak for: {}", name);

    // Warm up
    f();

    // Reset counters
    reset_memory_tracking();

    // Get baseline memory usage
    let baseline = get_memory_usage();

    // Run the function multiple times
    for _ in 0..iterations {
        f();
    }

    // Force drop of any lazy statics
    std::thread::yield_now();

    // Check final memory usage
    let final_usage = get_memory_usage();
    let leaked = final_usage - baseline;
    let leaked_per_iteration = leaked as f64 / iterations as f64;

    println!("  Baseline memory: {} bytes", baseline);
    println!("  Final memory: {} bytes", final_usage);
    println!("  Total leaked: {} bytes", leaked);
    println!("  Leaked per iteration: {:.2} bytes", leaked_per_iteration);

    // Allow some tolerance for allocator overhead
    const TOLERANCE_BYTES_PER_ITER: f64 = 100.0;

    if leaked_per_iteration > TOLERANCE_BYTES_PER_ITER {
        panic!(
            "Memory leak detected in {}: {:.2} bytes per iteration",
            name, leaked_per_iteration
        );
    }
}

#[test]
#[ignore] // This test requires special setup and may interfere with other tests
fn test_tensor_operations_memory_leak() {
    check_memory_leak("Tensor creation and drop", 1000, || {
        let _tensor = Tensor::new(vec![1.0; 1000], vec![10, 100]).unwrap();
    });

    check_memory_leak("Tensor arithmetic", 1000, || {
        let a = Tensor::new(vec![1.0; 100], vec![10, 10]).unwrap();
        let b = Tensor::new(vec![2.0; 100], vec![10, 10]).unwrap();
        let _c = a.add(&b).unwrap();
    });

    check_memory_leak("Tensor transpose", 1000, || {
        let tensor = Tensor::new(vec![1.0; 100], vec![10, 10]).unwrap();
        let _transposed = tensor.transpose(0, 1).unwrap();
    });
}

#[test]
#[ignore]
fn test_layer_memory_leak() {
    check_memory_leak("Linear layer forward pass", 100, || {
        let layer = Linear::new(100, 200, true);
        let input = Tensor::new(vec![1.0; 1000], vec![10, 100]).unwrap();
        let _output = layer.forward(&input).unwrap();
    });
}

/// Memory usage benchmark for different tensor sizes
#[test]
fn test_tensor_memory_usage() {
    let sizes = vec![
        (100, "100 elements"),
        (1_000, "1K elements"),
        (10_000, "10K elements"),
        (100_000, "100K elements"),
    ];

    println!("Tensor memory usage analysis:");

    for (size, label) in sizes {
        reset_memory_tracking();

        let tensor = Tensor::new(vec![0.0; size], vec![size]).unwrap();
        let allocated = ALLOCATED.load(Ordering::Relaxed);

        // Expected: size * 4 bytes (f32) + some overhead for Vec and Tensor struct
        let expected_min = size * 4;
        let overhead = allocated.saturating_sub(expected_min);

        println!("  {} ({}):", label, size);
        println!("    Allocated: {} bytes", allocated);
        println!("    Expected minimum: {} bytes", expected_min);
        println!("    Overhead: {} bytes ({:.1}%)",
            overhead,
            (overhead as f64 / expected_min as f64) * 100.0
        );

        drop(tensor);
    }
}

/// Test for cyclic references or retained memory
#[test]
fn test_no_retained_references() {
    use std::rc::Rc;
    use std::cell::RefCell;

    // Create a scenario that could potentially create cyclic references
    let initial_memory = get_memory_usage();

    {
        // Create tensors that reference each other through operations
        let a = Rc::new(RefCell::new(Tensor::new(vec![1.0; 1000], vec![100, 10]).unwrap()));
        let b = Rc::new(RefCell::new(Tensor::new(vec![2.0; 1000], vec![100, 10]).unwrap()));

        // Perform operations that might create references
        let c = a.borrow().add(&*b.borrow()).unwrap();
        let d = b.borrow().add(&*a.borrow()).unwrap();
        let _e = c.add(&d).unwrap();

        // All references should be dropped when leaving scope
    }

    // Force cleanup
    std::thread::yield_now();

    let final_memory = get_memory_usage();
    let leaked = final_memory - initial_memory;

    println!("Reference cycle test:");
    println!("  Initial memory: {} bytes", initial_memory);
    println!("  Final memory: {} bytes", final_memory);
    println!("  Difference: {} bytes", leaked);

    // Should not leak significant memory
    assert!(leaked.abs() < 10_000, "Possible reference cycle detected");
}

/// Helper to format bytes in human-readable form
fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_idx])
}