//! # ProfilerExportManager - generate_report_group Methods
//!
//! This module contains method implementations for `ProfilerExportManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, HashSet};
use super::profilerexportmanager_type::ProfilerExportManager;

impl ProfilerExportManager {
    fn generate_report(&self, data: &ProfilingData) -> Result<String> {
        let session_duration = if let Some(end) = data.session_info.end_time {
            Duration::from_secs(end.saturating_sub(data.session_info.start_time))
        } else {
            Duration::ZERO
        };
        let bottleneck_count = data.bottlenecks.len();
        let suggestion_count = data.suggestions.len();
        let metrics_count = data.metrics.len();
        let events_count = data.events.len();
        let avg_cpu_usage = if !data.metrics.is_empty() {
            data.metrics.iter().map(|m| m.cpu.usage_percent).sum::<f32>()
                / data.metrics.len() as f32
        } else {
            0.0
        };
        let avg_memory_usage = if !data.metrics.is_empty() {
            data.metrics.iter().map(|m| m.memory.heap_used_mb).sum::<f32>()
                / data.metrics.len() as f32
        } else {
            0.0
        };
        let report_html = format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformRS Mobile Performance Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px 8px 0 0; }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header .subtitle {{ margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em; }}
        .section {{ padding: 30px; border-bottom: 1px solid #eee; }}
        .section:last-child {{ border-bottom: none; }}
        .section h2 {{ color: #333; margin: 0 0 20px 0; font-size: 1.5em; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #667eea; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; margin: 0; }}
        .metric-label {{ color: #666; margin: 5px 0 0 0; text-transform: uppercase; font-size: 0.8em; letter-spacing: 1px; }}
        .bottlenecks, .suggestions {{ margin: 20px 0; }}
        .bottleneck-item, .suggestion-item {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #ffc107; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.9em; }}
        .timestamp {{ color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Performance Report</h1>
            <div class="subtitle">TrustformRS Mobile Profiler Analysis</div>
            <div class="timestamp">Generated: {}</div>
        </div>

        <div class="section">
            <h2>Session Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{:.1}s</div>
                    <div class="metric-label">Session Duration</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Events Recorded</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Metrics Snapshots</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.1}</div>
                    <div class="metric-label">Overall Health</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{:.1}%</div>
                    <div class="metric-label">Average CPU Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.1} MB</div>
                    <div class="metric-label">Average Memory Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Bottlenecks Detected</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Optimization Suggestions</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Performance Issues</h2>
            <div class="bottlenecks">
                {}
            </div>
        </div>

        <div class="section">
            <h2>Optimization Recommendations</h2>
            <div class="suggestions">
                {}
            </div>
        </div>

        <div class="footer">
            <p>Generated by TrustformRS Mobile Performance Profiler v{}</p>
        </div>
    </div>
</body>
</html>
        "#,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"), session_duration
            .as_secs_f64(), events_count, metrics_count, data.system_health
            .overall_score, avg_cpu_usage, avg_memory_usage, bottleneck_count,
            suggestion_count, if data.bottlenecks.is_empty() {
            "<div class=\"bottleneck-item\">No performance bottlenecks detected.</div>"
            .to_string() } else { data.bottlenecks.iter().take(5).map(| b |
            format!("<div class=\"bottleneck-item\"><strong>{}</strong>: {} (Severity: {:?})</div>",
            b.affected_component, b.description, b.severity)).collect::< Vec < _ >> ()
            .join("\n") }, if data.suggestions.is_empty() {
            "<div class=\"suggestion-item\">No optimization suggestions available.</div>"
            .to_string() } else { data.suggestions.iter().take(5).map(| s |
            format!("<div class=\"suggestion-item\"><strong>{}</strong>: {} (Priority: {:?})</div>",
            format!("{:?}", s.suggestion_type), s.description, s.priority)).collect::<
            Vec < _ >> ().join("\n") }, data.profiler_version
        );
        Ok(report_html)
    }
}
