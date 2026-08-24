//! Regression tests for the baseline bounds and confidence intervals.
//!
//! Until 0.2.1 both types filled almost every field with a constant: a flat
//! ten-percent band around the mean for the CPU/memory/latency bounds, the
//! literal `(0.5, 1.0)` for efficiency, and `(2_000_000.0, 8_000_000.0)` /
//! `(200.0, 800.0)` / `(15ms, 85ms)` / `(0.0, 3.0)` for the network, I/O,
//! response-time and error-rate intervals -- the same numbers for every window
//! ever measured.

use super::*;

fn window(count: usize, mean_throughput: f64, std_dev: f64) -> WindowStatistics {
    WindowStatistics {
        count,
        mean: mean_throughput,
        std_dev,
        mean_throughput,
        throughput_std_dev: std_dev,
        mean_cpu_utilization: 40.0,
        mean_memory_utilization: 60.0,
        ..WindowStatistics::default()
    }
}

#[test]
fn variability_bounds_report_only_the_dimension_the_window_measures() {
    let bounds = VariabilityBounds::from_statistics(&window(100, 50.0, 5.0));
    // Two sigma either side of the mean.
    assert!(
        (bounds.throughput_bounds.0 - 40.0).abs() < 1e-9,
        "{bounds:?}"
    );
    assert!(
        (bounds.throughput_bounds.1 - 60.0).abs() < 1e-9,
        "{bounds:?}"
    );
    assert!(bounds.cpu_bounds.is_none(), "no CPU dispersion is recorded");
    assert!(bounds.memory_bounds.is_none());
    assert!(bounds.latency_bounds.is_none());
    assert!(bounds.efficiency_bounds.is_none());
}

#[test]
fn throughput_bounds_widen_with_the_measured_spread() {
    let tight = VariabilityBounds::from_statistics(&window(100, 50.0, 1.0));
    let loose = VariabilityBounds::from_statistics(&window(100, 50.0, 10.0));
    let tight_width = tight.throughput_bounds.1 - tight.throughput_bounds.0;
    let loose_width = loose.throughput_bounds.1 - loose.throughput_bounds.0;
    assert!(loose_width > tight_width, "{loose_width} vs {tight_width}");
}

#[test]
fn a_dimension_without_a_bound_cannot_report_a_deviation() {
    let baseline = PerformanceBaseline::from_statistics(&window(100, 50.0, 5.0));
    let mut metrics = RealTimeMetrics {
        throughput: 50.0,
        cpu_utilization: 9_000.0,
        memory_utilization: 9_000.0,
        ..RealTimeMetrics::default()
    };
    assert!(
        !baseline.check_deviation(&metrics),
        "CPU and memory have no measured bound, so they cannot be judged; \
         throughput is inside its bound"
    );
    metrics.throughput = 1_000.0;
    assert!(
        baseline.check_deviation(&metrics),
        "throughput outside its measured bound is a deviation"
    );
}

#[test]
fn confidence_intervals_report_only_the_throughput_they_measured() {
    let intervals = ConfidenceIntervals::from_statistics(&window(100, 50.0, 5.0));
    assert!((intervals.confidence_level - 95.0).abs() < 1e-6);
    // z * s / sqrt(n) = 1.96 * 5 / 10 = 0.98
    assert!((intervals.mean_lower - 49.02).abs() < 1e-9, "{intervals:?}");
    assert!((intervals.mean_upper - 50.98).abs() < 1e-9, "{intervals:?}");
    assert_eq!(
        intervals.throughput_interval,
        (intervals.mean_lower, intervals.mean_upper)
    );
    assert!(intervals.latency_interval.is_none());
    assert!(intervals.cpu_interval.is_none());
    assert!(intervals.memory_interval.is_none());
    assert!(intervals.network_interval.is_none());
    assert!(intervals.io_interval.is_none());
    assert!(intervals.response_time_interval.is_none());
    assert!(intervals.error_rate_interval.is_none());
    assert!(matches!(intervals.method, ConfidenceMethod::Normal));
}

#[test]
fn the_variance_interval_needs_at_least_two_samples() {
    let single = ConfidenceIntervals::from_statistics(&window(1, 50.0, 0.0));
    assert!(single.variance_lower.is_none());
    assert!(single.variance_upper.is_none());

    let many = ConfidenceIntervals::from_statistics(&window(101, 50.0, 5.0));
    // margin = 1.96 * 25 * sqrt(2/100) = 6.929646...
    let lower = many.variance_lower.expect("interval reported");
    let upper = many.variance_upper.expect("interval reported");
    let expected_margin = 1.96 * 25.0 * (2.0_f64 / 100.0).sqrt();
    assert!((upper - (25.0 + expected_margin)).abs() < 1e-9, "{upper}");
    assert!((lower - (25.0 - expected_margin)).abs() < 1e-9, "{lower}");
}

#[test]
fn the_default_intervals_claim_no_measurement() {
    let intervals = ConfidenceIntervals::default();
    assert_eq!(intervals.throughput_interval, (0.0, 0.0));
    assert!(intervals.variance_lower.is_none());
    assert!(intervals.network_interval.is_none());
    assert!(intervals.io_interval.is_none());
}
