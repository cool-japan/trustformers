//! Performance profiler for detailed performance analysis

#![allow(unused_variables)] // Performance profiler

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Profile result for a single operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResult {
    /// Operation name
    pub name: String,
    /// Total time spent
    pub total_time: Duration,
    /// Number of calls
    pub call_count: usize,
    /// Average time per call
    pub avg_time: Duration,
    /// Minimum time
    pub min_time: Duration,
    /// Maximum time
    pub max_time: Duration,
    /// Self time (excluding children)
    pub self_time: Duration,
    /// Child operations
    pub children: Vec<ProfileResult>,
    /// Percentage of parent time
    pub percent_of_parent: f64,
}

impl ProfileResult {
    /// Create a new profile result
    pub fn new(name: String) -> Self {
        Self {
            name,
            total_time: Duration::ZERO,
            call_count: 0,
            avg_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            self_time: Duration::ZERO,
            children: Vec::new(),
            percent_of_parent: 0.0,
        }
    }

    /// Add a timing measurement
    pub fn add_timing(&mut self, duration: Duration) {
        self.total_time += duration;
        self.call_count += 1;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
        self.avg_time = self.total_time / self.call_count as u32;
    }

    /// Calculate self time and percentages
    fn calculate_self_time(&mut self) {
        let children_time: Duration = self.children.iter().map(|c| c.total_time).sum();
        self.self_time = self.total_time.saturating_sub(children_time);

        // Calculate percentages for children
        let total_ms = self.total_time.as_secs_f64() * 1000.0;
        for child in &mut self.children {
            child.percent_of_parent = if total_ms > 0.0 {
                (child.total_time.as_secs_f64() * 1000.0 / total_ms) * 100.0
            } else {
                0.0
            };
        }
    }

    /// Record operation start (for demo compatibility)
    pub fn record_operation_start(&mut self, _name: &str) {
        // For demo purposes - in a real implementation this would start timing
    }

    /// Record operation end (for demo compatibility)
    pub fn record_operation_end(&mut self, name: &str) {
        // For demo purposes - simulate adding a timing measurement
        let duration = Duration::from_millis(5);
        self.add_timing(duration);
    }

    /// Print profile results
    pub fn print(&self, indent: usize) {
        let indent_str = " ".repeat(indent);
        println!(
            "{}{:<40} {:>8} {:>10.2}ms {:>10.2}ms {:>10.2}ms {:>6.1}%",
            indent_str,
            self.name,
            self.call_count,
            self.total_time.as_secs_f64() * 1000.0,
            self.avg_time.as_secs_f64() * 1000.0,
            self.self_time.as_secs_f64() * 1000.0,
            self.percent_of_parent,
        );

        // Print children sorted by total time
        let mut children = self.children.clone();
        children.sort_by(|a, b| b.total_time.cmp(&a.total_time));
        for child in children {
            child.print(indent + 2);
        }
    }
}

/// Performance profiler
pub struct PerformanceProfiler {
    /// Profile stack
    stack: Arc<Mutex<Vec<ProfileNode>>>,
    /// Root profile results
    roots: Arc<Mutex<HashMap<String, ProfileResult>>>,
    /// Whether profiling is enabled
    enabled: Arc<Mutex<bool>>,
}

#[derive(Debug)]
struct ProfileNode {
    name: String,
    start_time: Instant,
    result: ProfileResult,
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceProfiler {
    /// Create new profiler
    pub fn new() -> Self {
        Self {
            stack: Arc::new(Mutex::new(Vec::new())),
            roots: Arc::new(Mutex::new(HashMap::new())),
            enabled: Arc::new(Mutex::new(false)),
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

    /// Start profiling an operation
    pub fn start_operation(&self, name: &str) -> ProfileGuard {
        if !self.is_enabled() {
            return ProfileGuard {
                profiler: None,
                name: String::new(),
            };
        }

        let node = ProfileNode {
            name: name.to_string(),
            start_time: Instant::now(),
            result: ProfileResult::new(name.to_string()),
        };

        self.stack.lock().expect("Lock poisoned").push(node);

        ProfileGuard {
            profiler: Some(self.clone()),
            name: name.to_string(),
        }
    }

    /// End profiling an operation
    fn end_operation(&self, name: &str) {
        let mut stack = self.stack.lock().expect("Lock poisoned");

        if let Some(node) = stack.pop() {
            if node.name != name {
                eprintln!("Profile mismatch: expected {}, got {}", name, node.name);
                return;
            }

            let duration = node.start_time.elapsed();
            let mut result = node.result;
            result.add_timing(duration);

            if stack.is_empty() {
                // This is a root operation
                let mut roots = self.roots.lock().expect("Lock poisoned");
                roots
                    .entry(name.to_string())
                    .and_modify(|r| {
                        r.total_time += duration;
                        r.call_count += 1;
                        r.avg_time = r.total_time / r.call_count as u32;
                        r.min_time = r.min_time.min(duration);
                        r.max_time = r.max_time.max(duration);
                    })
                    .or_insert(result);
            } else {
                // Add to parent's children
                if let Some(parent) = stack.last_mut() {
                    parent.result.children.push(result);
                }
            }
        }
    }

    /// Get profile results
    pub fn get_results(&self) -> HashMap<String, ProfileResult> {
        let mut results = self.roots.lock().expect("Lock poisoned").clone();

        // Calculate self times and percentages
        for result in results.values_mut() {
            result.calculate_self_time();
        }

        results
    }

    /// Clear all profile data
    pub fn clear(&self) {
        self.stack.lock().expect("Lock poisoned").clear();
        self.roots.lock().expect("Lock poisoned").clear();
    }

    /// Print profile summary
    pub fn print_summary(&self) {
        let results = self.get_results();

        if results.is_empty() {
            println!("No profile data collected.");
            return;
        }

        println!("\n=== Performance Profile Summary ===");
        println!(
            "{:<40} {:>8} {:>10} {:>10} {:>10} {:>6}",
            "Operation", "Calls", "Total (ms)", "Avg (ms)", "Self (ms)", "%"
        );
        println!("{}", "-".repeat(84));

        // Sort by total time
        let mut sorted: Vec<_> = results.values().collect();
        sorted.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        let total_time: Duration = sorted.iter().map(|r| r.total_time).sum();
        let total_ms = total_time.as_secs_f64() * 1000.0;

        for result in sorted {
            let percent = if total_ms > 0.0 {
                (result.total_time.as_secs_f64() * 1000.0 / total_ms) * 100.0
            } else {
                0.0
            };

            println!(
                "{:<40} {:>8} {:>10.2} {:>10.2} {:>10.2} {:>6.1}",
                result.name,
                result.call_count,
                result.total_time.as_secs_f64() * 1000.0,
                result.avg_time.as_secs_f64() * 1000.0,
                result.self_time.as_secs_f64() * 1000.0,
                percent,
            );
        }

        println!("\nTotal time: {:.2}ms", total_ms);
    }

    /// Export results to flamegraph format
    pub fn export_flamegraph(&self, path: &str) -> Result<()> {
        let results = self.get_results();
        let mut file = std::fs::File::create(path)?;

        // Write flamegraph format
        for (name, result) in results {
            self.write_flamegraph_entry(&mut file, &name, &result, Vec::new())?;
        }

        Ok(())
    }

    fn write_flamegraph_entry(
        &self,
        file: &mut std::fs::File,
        name: &str,
        result: &ProfileResult,
        stack: Vec<String>,
    ) -> Result<()> {
        Self::write_flamegraph_entry_helper(file, name, result, stack)
    }

    /// Helper for recursive flamegraph entry writing
    fn write_flamegraph_entry_helper(
        file: &mut std::fs::File,
        name: &str,
        result: &ProfileResult,
        mut stack: Vec<String>,
    ) -> Result<()> {
        use std::io::Write;

        stack.push(name.to_string());
        let stack_str = stack.join(";");
        let microseconds = result.self_time.as_micros();

        if microseconds > 0 {
            writeln!(file, "{} {}", stack_str, microseconds)?;
        }

        // Write children
        for child in &result.children {
            Self::write_flamegraph_entry_helper(file, &child.name, child, stack.clone())?;
        }

        Ok(())
    }
}

impl Clone for PerformanceProfiler {
    fn clone(&self) -> Self {
        Self {
            stack: Arc::clone(&self.stack),
            roots: Arc::clone(&self.roots),
            enabled: Arc::clone(&self.enabled),
        }
    }
}

/// RAII guard for profiling
pub struct ProfileGuard {
    profiler: Option<PerformanceProfiler>,
    name: String,
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        if let Some(profiler) = &self.profiler {
            profiler.end_operation(&self.name);
        }
    }
}

/// Macro for easy profiling
#[macro_export]
macro_rules! profile {
    ($profiler:expr, $name:expr, $code:block) => {{
        let _guard = $profiler.start_operation($name);
        $code
    }};
}

lazy_static::lazy_static! {
    /// Global profiler instance
    pub static ref GLOBAL_PROFILER: PerformanceProfiler = PerformanceProfiler::new();
}

/// Profile a function with the global profiler
pub fn profile_fn<F, R>(name: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = GLOBAL_PROFILER.start_operation(name);
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_basic_profiling() {
        let profiler = PerformanceProfiler::new();
        profiler.enable();

        {
            let _guard = profiler.start_operation("test_operation");
            sleep(Duration::from_millis(10));
        }

        let results = profiler.get_results();
        assert!(results.contains_key("test_operation"));

        let result = &results["test_operation"];
        assert_eq!(result.call_count, 1);
        assert!(result.total_time >= Duration::from_millis(10));
    }

    #[test]
    fn test_nested_profiling() {
        let profiler = PerformanceProfiler::new();
        profiler.enable();

        {
            let _outer = profiler.start_operation("outer");
            sleep(Duration::from_millis(5));

            {
                let _inner = profiler.start_operation("inner");
                sleep(Duration::from_millis(10));
            }

            sleep(Duration::from_millis(5));
        }

        let results = profiler.get_results();
        let outer = &results["outer"];

        assert_eq!(outer.call_count, 1);
        assert!(outer.total_time >= Duration::from_millis(20));
        assert_eq!(outer.children.len(), 1);

        let inner = &outer.children[0];
        assert_eq!(inner.name, "inner");
        assert!(inner.total_time >= Duration::from_millis(10));
    }

    #[test]
    fn test_profile_macro() {
        let profiler = PerformanceProfiler::new();
        profiler.enable();

        profile!(profiler, "macro_test", {
            sleep(Duration::from_millis(5));
        });

        let results = profiler.get_results();
        assert!(results.contains_key("macro_test"));
    }
}
