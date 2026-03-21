//! Memory profiling reporting and HTML generation
//!
//! This module provides functionality for generating comprehensive
//! memory profiling reports in both JSON and HTML formats.

use anyhow::Result;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

use super::types::*;

/// Memory summary for quick overview
#[derive(Debug, Clone)]
pub struct MemorySummary {
    pub current_memory_mb: f64,
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub total_allocations: usize,
    pub potential_leaks: usize,
    pub active_alerts: usize,
}

impl super::profiler::MemoryProfiler {
    /// Generate a comprehensive memory report
    pub async fn generate_report(&self) -> Result<MemoryDashboardReport> {
        let metrics_history = self
            .get_metrics_history()
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock metrics history: {}", e))?
            .clone();
        let allocations = self
            .get_allocations()
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock allocations: {}", e))?
            .clone();
        let alerts = self
            .get_alerts_internal()
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock alerts: {}", e))?
            .clone();
        let patterns = self
            .get_patterns_internal()
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock patterns: {}", e))?
            .clone();

        // Calculate summary statistics
        let summary = if !metrics_history.is_empty() {
            let total_metrics: Vec<_> = metrics_history.iter().map(|m| m.total_memory_mb).collect();
            let peak_memory = total_metrics.iter().fold(0.0f64, |a, &b| a.max(b));
            let avg_memory = total_metrics.iter().sum::<f64>() / total_metrics.len() as f64;

            // Calculate efficiency score based on memory usage patterns
            let efficiency_score = self.calculate_memory_efficiency(&metrics_history);

            // Count alert types
            let mut alert_counts = HashMap::new();
            for alert in &alerts {
                let count = alert_counts.entry(alert.severity.clone()).or_insert(0);
                *count += 1;
            }

            MemoryUsageSummary {
                total_runtime_seconds: self
                    .get_start_time()
                    .map(|start| start.elapsed().as_secs_f64())
                    .unwrap_or(0.0),
                peak_memory_mb: peak_memory,
                average_memory_mb: avg_memory,
                memory_efficiency_score: efficiency_score,
                total_allocations: allocations.len() as u64,
                total_deallocations: allocations.len() as u64, // Simplified
                leaked_allocations: allocations.values().filter(|a| a.is_leaked).count() as u64,
                fragmentation_events: metrics_history
                    .iter()
                    .filter(|m| m.memory_fragmentation_ratio > 0.3)
                    .count() as u64,
                gc_pressure_events: metrics_history.iter().filter(|m| m.gc_time_ms > 200.0).count()
                    as u64,
                alert_count_by_severity: alert_counts,
            }
        } else {
            MemoryUsageSummary {
                total_runtime_seconds: 0.0,
                peak_memory_mb: 0.0,
                average_memory_mb: 0.0,
                memory_efficiency_score: 0.0,
                total_allocations: 0,
                total_deallocations: 0,
                leaked_allocations: 0,
                fragmentation_events: 0,
                gc_pressure_events: 0,
                alert_count_by_severity: HashMap::new(),
            }
        };

        let recommendations = self.generate_recommendations(&alerts, &patterns);
        let system_info = self.get_system_info();

        Ok(MemoryDashboardReport {
            timestamp: std::time::SystemTime::now(),
            summary,
            metrics_over_time: metrics_history.into_iter().collect(),
            alerts,
            patterns,
            recommendations,
            system_info,
        })
    }

    /// Save report to files
    pub async fn save_report(&self, report: &MemoryDashboardReport) -> Result<()> {
        let timestamp =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();

        let json_filename = format!(
            "{}/memory_report_{}.json",
            self.get_config().output_dir,
            timestamp
        );
        let html_filename = format!(
            "{}/memory_report_{}.html",
            self.get_config().output_dir,
            timestamp
        );

        // Save JSON report
        let json_content = serde_json::to_string_pretty(report)?;
        std::fs::write(&json_filename, json_content)?;

        // Save HTML report
        let html_content = self.generate_html_report(report)?;
        std::fs::write(&html_filename, html_content)?;

        println!(
            "Memory report saved to: {} and {}",
            json_filename, html_filename
        );
        Ok(())
    }

    /// Generate HTML report
    fn generate_html_report(&self, report: &MemoryDashboardReport) -> Result<String> {
        let mut html = String::new();

        writeln!(&mut html, "<!DOCTYPE html>")?;
        writeln!(
            &mut html,
            "<html><head><title>Memory Profiling Report</title>"
        )?;
        writeln!(&mut html, "<style>")?;
        writeln!(
            &mut html,
            "body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}"
        )?;
        writeln!(&mut html, ".container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}")?;
        writeln!(&mut html, ".summary {{ background: #e8f4fd; padding: 20px; margin: 20px 0; border-radius: 6px; border-left: 4px solid #2196F3; }}")?;
        writeln!(&mut html, ".alert {{ padding: 12px; margin: 8px 0; border-radius: 4px; border-left: 4px solid; }}")?;
        writeln!(
            &mut html,
            ".info {{ background: #e3f2fd; border-color: #2196F3; }}"
        )?;
        writeln!(
            &mut html,
            ".warning {{ background: #fff3e0; border-color: #ff9800; }}"
        )?;
        writeln!(
            &mut html,
            ".error {{ background: #ffebee; border-color: #f44336; }}"
        )?;
        writeln!(
            &mut html,
            ".critical {{ background: #fce4ec; border-color: #e91e63; }}"
        )?;
        writeln!(
            &mut html,
            "table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}"
        )?;
        writeln!(
            &mut html,
            "th, td {{ border: 1px solid #e0e0e0; padding: 12px; text-align: left; }}"
        )?;
        writeln!(
            &mut html,
            "th {{ background-color: #f8f9fa; font-weight: 600; }}"
        )?;
        writeln!(
            &mut html,
            ".metric {{ display: inline-block; margin: 10px 20px 10px 0; }}"
        )?;
        writeln!(
            &mut html,
            ".metric-label {{ font-weight: bold; color: #666; }}"
        )?;
        writeln!(
            &mut html,
            ".metric-value {{ font-size: 1.2em; color: #2196F3; }}"
        )?;
        writeln!(
            &mut html,
            "h1 {{ color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}"
        )?;
        writeln!(&mut html, "h2 {{ color: #555; margin-top: 30px; }}")?;
        writeln!(
            &mut html,
            ".recommendations {{ background: #f1f8e9; padding: 15px; border-radius: 4px; }}"
        )?;
        writeln!(&mut html, "</style></head><body>")?;

        writeln!(&mut html, "<div class='container'>")?;
        writeln!(&mut html, "<h1>🧠 Memory Profiling Report</h1>")?;
        writeln!(
            &mut html,
            "<p><strong>Generated:</strong> {:?}</p>",
            report.timestamp
        )?;
        writeln!(
            &mut html,
            "<p><strong>Runtime:</strong> {:.1} seconds</p>",
            report.summary.total_runtime_seconds
        )?;

        // Summary section with metrics
        writeln!(&mut html, "<div class='summary'>")?;
        writeln!(&mut html, "<h2>📊 Memory Usage Summary</h2>")?;
        writeln!(&mut html, "<div class='metric'><div class='metric-label'>Peak Memory:</div><div class='metric-value'>{:.2} MB</div></div>", report.summary.peak_memory_mb)?;
        writeln!(&mut html, "<div class='metric'><div class='metric-label'>Average Memory:</div><div class='metric-value'>{:.2} MB</div></div>", report.summary.average_memory_mb)?;
        writeln!(&mut html, "<div class='metric'><div class='metric-label'>Efficiency Score:</div><div class='metric-value'>{:.1}%</div></div>", report.summary.memory_efficiency_score * 100.0)?;
        writeln!(&mut html, "<div class='metric'><div class='metric-label'>Total Allocations:</div><div class='metric-value'>{}</div></div>", report.summary.total_allocations)?;
        writeln!(&mut html, "<div class='metric'><div class='metric-label'>Leaked Allocations:</div><div class='metric-value'>{}</div></div>", report.summary.leaked_allocations)?;
        writeln!(&mut html, "<div class='metric'><div class='metric-label'>Fragmentation Events:</div><div class='metric-value'>{}</div></div>", report.summary.fragmentation_events)?;
        writeln!(&mut html, "</div>")?;

        // System information
        writeln!(&mut html, "<h2>💻 System Information</h2>")?;
        writeln!(&mut html, "<table>")?;
        writeln!(&mut html, "<tr><th>Property</th><th>Value</th></tr>")?;
        writeln!(
            &mut html,
            "<tr><td>Total System Memory</td><td>{:.1} GB</td></tr>",
            report.system_info.total_system_memory_gb
        )?;
        writeln!(
            &mut html,
            "<tr><td>Available Memory</td><td>{:.1} GB</td></tr>",
            report.system_info.available_system_memory_gb
        )?;
        writeln!(
            &mut html,
            "<tr><td>CPU Count</td><td>{}</td></tr>",
            report.system_info.cpu_count
        )?;
        writeln!(
            &mut html,
            "<tr><td>Operating System</td><td>{}</td></tr>",
            report.system_info.os_info
        )?;
        writeln!(
            &mut html,
            "<tr><td>Rust Version</td><td>{}</td></tr>",
            report.system_info.rust_version
        )?;
        writeln!(&mut html, "</table>")?;

        // Alerts section
        if !report.alerts.is_empty() {
            writeln!(
                &mut html,
                "<h2>🚨 Alerts ({} total)</h2>",
                report.alerts.len()
            )?;
            for alert in &report.alerts {
                let class = match alert.severity {
                    AlertSeverity::Info => "info",
                    AlertSeverity::Warning => "warning",
                    AlertSeverity::Error => "error",
                    AlertSeverity::Critical => "critical",
                };
                writeln!(&mut html, "<div class='alert {}'>", class)?;
                writeln!(
                    &mut html,
                    "<strong>{:?}:</strong> {}",
                    alert.severity, alert.message
                )?;
                writeln!(&mut html, "</div>")?;
            }
        }

        // Memory patterns
        if !report.patterns.is_empty() {
            writeln!(&mut html, "<h2>📈 Memory Patterns</h2>")?;
            writeln!(&mut html, "<table>")?;
            writeln!(&mut html, "<tr><th>Pattern Type</th><th>Frequency</th><th>Average Size</th><th>Trend</th></tr>")?;
            for pattern in &report.patterns {
                writeln!(
                    &mut html,
                    "<tr><td>{:?}</td><td>{}</td><td>{:.2} MB</td><td>{:?}</td></tr>",
                    pattern.pattern_type, pattern.frequency, pattern.average_size, pattern.trend
                )?;
            }
            writeln!(&mut html, "</table>")?;
        }

        // Recommendations section
        if !report.recommendations.is_empty() {
            writeln!(&mut html, "<h2>💡 Recommendations</h2>")?;
            writeln!(&mut html, "<div class='recommendations'>")?;
            writeln!(&mut html, "<ul>")?;
            for rec in &report.recommendations {
                writeln!(&mut html, "<li>{}</li>", rec)?;
            }
            writeln!(&mut html, "</ul>")?;
            writeln!(&mut html, "</div>")?;
        }

        writeln!(&mut html, "</div></body></html>")?;
        Ok(html)
    }

    /// Generate recommendations based on current state
    fn generate_recommendations(
        &self,
        alerts: &[MemoryAlert],
        patterns: &[MemoryPattern],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze alerts for recommendations
        let high_memory_alerts = alerts
            .iter()
            .filter(|a| matches!(a.alert_type, MemoryAlertType::HighMemoryUsage))
            .count();

        if high_memory_alerts > 5 {
            recommendations
                .push("Consider using gradient checkpointing to reduce memory usage".to_string());
            recommendations.push("Try reducing the batch size".to_string());
            recommendations
                .push("Enable mixed precision training if not already active".to_string());
        }

        let leak_alerts = alerts
            .iter()
            .filter(|a| matches!(a.alert_type, MemoryAlertType::MemoryLeak))
            .count();

        if leak_alerts > 0 {
            recommendations
                .push("Memory leaks detected - review tensor lifecycle management".to_string());
            recommendations.push("Use explicit cleanup for large tensors".to_string());
            recommendations
                .push("Consider using weak references for circular dependencies".to_string());
        }

        // Analyze patterns for recommendations
        let large_allocation_patterns = patterns
            .iter()
            .filter(|p| matches!(p.pattern_type, PatternType::LargeAllocations))
            .count();

        if large_allocation_patterns > 10 {
            recommendations
                .push("Frequent large allocations detected - consider memory pooling".to_string());
            recommendations
                .push("Implement tensor reuse strategies for repeated operations".to_string());
        }

        let frequent_small_allocations = patterns
            .iter()
            .filter(|p| matches!(p.pattern_type, PatternType::FrequentSmallAllocations))
            .count();

        if frequent_small_allocations > 5 {
            recommendations.push(
                "Frequent small allocations detected - consider pre-allocating buffers".to_string(),
            );
            recommendations
                .push("Use object pooling for frequently allocated small objects".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push(
                "Memory usage appears normal - no specific recommendations at this time"
                    .to_string(),
            );
            recommendations
                .push("Continue monitoring for any changes in allocation patterns".to_string());
        }

        recommendations
    }

    /// Get current memory usage summary
    pub fn get_current_summary(&self) -> Option<MemorySummary> {
        let history = self.get_metrics_history().lock().ok()?;
        let alerts = self.get_alerts_internal().lock().ok()?;
        let allocations = self.get_allocations().lock().ok()?;

        if let Some(latest) = history.back() {
            let total_memory: f64 = history.iter().map(|m| m.total_memory_mb).sum();
            let avg_memory = if !history.is_empty() {
                total_memory / history.len() as f64
            } else {
                latest.total_memory_mb
            };

            Some(MemorySummary {
                current_memory_mb: latest.total_memory_mb,
                peak_memory_mb: latest.peak_memory_mb,
                average_memory_mb: avg_memory,
                total_allocations: allocations.len(),
                potential_leaks: allocations.values().filter(|a| a.is_leaked).count(),
                active_alerts: alerts.len(),
            })
        } else {
            None
        }
    }

    /// Calculate memory efficiency score (0.0 to 1.0)
    fn calculate_memory_efficiency(
        &self,
        metrics: &std::collections::VecDeque<MemoryMetrics>,
    ) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        let mut efficiency_factors = Vec::new();

        // Factor 1: Memory stability (lower variance = higher efficiency)
        let memory_values: Vec<f64> = metrics.iter().map(|m| m.total_memory_mb).collect();
        let mean_memory = memory_values.iter().sum::<f64>() / memory_values.len() as f64;
        let variance = memory_values.iter().map(|x| (x - mean_memory).powi(2)).sum::<f64>()
            / memory_values.len() as f64;
        let stability_score = 1.0 / (1.0 + variance / mean_memory.max(1.0));
        efficiency_factors.push(stability_score);

        // Factor 2: Low fragmentation
        let avg_fragmentation = metrics.iter().map(|m| m.memory_fragmentation_ratio).sum::<f64>()
            / metrics.len() as f64;
        let fragmentation_score = 1.0 - avg_fragmentation.min(1.0);
        efficiency_factors.push(fragmentation_score);

        // Factor 3: GC efficiency
        let avg_gc_time = metrics.iter().map(|m| m.gc_time_ms).sum::<f64>() / metrics.len() as f64;
        let gc_score = 1.0 / (1.0 + avg_gc_time / 100.0); // Normalize around 100ms
        efficiency_factors.push(gc_score);

        // Factor 4: Allocation efficiency
        let total_allocations: u64 = metrics.iter().map(|m| m.allocated_objects).sum();
        let total_deallocations: u64 = metrics.iter().map(|m| m.deallocated_objects).sum();
        let allocation_ratio = if total_allocations > 0 {
            total_deallocations as f64 / total_allocations as f64
        } else {
            1.0
        };
        efficiency_factors.push(allocation_ratio.min(1.0));

        // Calculate weighted average
        efficiency_factors.iter().sum::<f64>() / efficiency_factors.len() as f64
    }

    /// Get system information for the report
    fn get_system_info(&self) -> SystemInfo {
        SystemInfo {
            total_system_memory_gb: 16.0,    // Mock value
            available_system_memory_gb: 8.0, // Mock value
            cpu_count: num_cpus::get(),
            os_info: std::env::consts::OS.to_string(),
            rust_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}
