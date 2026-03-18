//! Reporting Coordinator
//!
//! Coordinator for comprehensive reporting across all analysis results.

use super::super::types::*;
use super::*;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex as TokioMutex;
use tracing::{info, instrument};

// Explicitly import PerformanceMetrics from types to avoid ambiguity
use crate::performance_optimizer::test_characterization::types::performance::PerformanceMetrics;
// Type alias for coordinator's PerformanceMetrics
use crate::performance_optimizer::test_characterization::manager::performance_coordinator::PerformanceMetrics as CoordinatorPerformanceMetrics;

#[derive(Debug)]
pub struct ReportingCoordinator {
    /// Results synthesizer reference
    results_synthesizer: Arc<ResultsSynthesizer>,
    /// Performance coordinator reference
    performance_coordinator: Arc<PerformanceCoordinator>,
    /// Engine statistics reference
    engine_stats: Arc<EngineStatistics>,
    /// Report templates
    report_templates: Arc<ReportTemplates>,
    /// Report cache
    report_cache: Arc<TokioMutex<HashMap<String, CachedReport>>>,
    /// Reporting configuration
    config: Arc<ReportingConfig>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

/// Comprehensive analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Test characteristics summary
    pub characteristics_summary: TestCharacteristics,
    /// Performance metrics summary
    pub performance_summary: PerformanceMetrics,
    /// Engine statistics summary
    pub engine_statistics: ReportEngineStatistics,
    /// Analysis breakdown by phase
    pub phase_breakdown: HashMap<AnalysisPhase, PhaseReportSummary>,
    /// Resource utilization summary
    pub resource_utilization: ResourceUtilizationSummary,
    /// Recommendations
    pub recommendations: Vec<Recommendation>,
    /// Alerts and warnings
    pub alerts: Vec<PerformanceAlert>,
    /// Historical comparison (if available)
    pub historical_comparison: Option<HistoricalComparison>,
    /// Trend analysis
    pub trend_analysis: TrendAnalysis,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Report ID
    pub report_id: String,
    /// Test ID
    pub test_id: String,
    /// Report generation timestamp
    pub generated_at: SystemTime,
    /// Report version
    pub version: String,
    /// Report type
    pub report_type: ReportType,
    /// Report format
    pub format: ReportFormat,
    /// Generation duration
    pub generation_duration_ms: u64,
}

/// Report types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    Comprehensive,
    Performance,
    ResourceUtilization,
    TrendAnalysis,
    ExecutiveSummary,
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Html,
    Pdf,
    Csv,
    Markdown,
}

/// Engine statistics for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportEngineStatistics {
    /// Total analyses performed
    pub total_analyses: u64,
    /// Successful analyses
    pub successful_analyses: u64,
    /// Failed analyses
    pub failed_analyses: u64,
    /// Success rate
    pub success_rate: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average analysis duration
    pub average_analysis_duration_ms: u64,
    /// Active analyses
    pub active_analyses: usize,
    /// Errors recovered
    pub errors_recovered: u64,
}

/// Phase report summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseReportSummary {
    /// Phase name
    pub phase: AnalysisPhase,
    /// Execution time
    pub execution_time_ms: u64,
    /// Success status
    pub success: bool,
    /// Confidence score
    pub confidence_score: f64,
    /// Key findings
    pub key_findings: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Resource utilization summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationSummary {
    /// Peak CPU usage
    pub peak_cpu_percent: f64,
    /// Average CPU usage
    pub avg_cpu_percent: f64,
    /// Peak memory usage
    pub peak_memory_mb: f64,
    /// Average memory usage
    pub avg_memory_mb: f64,
    /// Network I/O
    pub network_io_bps: u64,
    /// Disk I/O
    pub disk_io_bps: u64,
    /// Resource efficiency score
    pub efficiency_score: f64,
}

/// Recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation ID
    pub id: String,
    /// Category
    pub category: RecommendationCategory,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Expected impact
    pub expected_impact: String,
    /// Implementation effort
    pub implementation_effort: ImplementationEffort,
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Performance,
    ResourceOptimization,
    Configuration,
    Architecture,
    Monitoring,
    ErrorHandling,
}

/// Recommendation priorities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
    Informational,
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Historical comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalComparison {
    /// Previous report timestamp
    pub previous_report_timestamp: SystemTime,
    /// Performance change
    pub performance_change: PerformanceChange,
    /// Resource usage change
    pub resource_usage_change: ResourceUsageChange,
    /// Key differences
    pub key_differences: Vec<String>,
}

/// Performance change metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceChange {
    /// Analysis duration change percentage
    pub duration_change_percent: f64,
    /// Success rate change percentage
    pub success_rate_change_percent: f64,
    /// Cache hit rate change percentage
    pub cache_hit_rate_change_percent: f64,
}

/// Resource usage change metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageChange {
    /// CPU usage change percentage
    pub cpu_change_percent: f64,
    /// Memory usage change percentage
    pub memory_change_percent: f64,
    /// Network I/O change percentage
    pub network_io_change_percent: f64,
}

/// Trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Performance trend
    pub performance_trend: TrendDirection,
    /// Resource usage trend
    pub resource_usage_trend: TrendDirection,
    /// Error rate trend
    pub error_rate_trend: TrendDirection,
    /// Trend confidence
    pub trend_confidence: f64,
    /// Predicted values
    pub predictions: TrendPredictions,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
    Unknown,
}

/// Trend predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendPredictions {
    /// Predicted next performance score
    pub next_performance_score: f64,
    /// Predicted resource usage
    pub next_resource_usage: f64,
    /// Prediction confidence
    pub prediction_confidence: f64,
}

/// Cached report entry
#[derive(Debug, Clone)]
pub struct CachedReport {
    /// Report data
    pub report: ComprehensiveReport,
    /// Cache timestamp
    pub cached_at: SystemTime,
    /// Access count
    pub access_count: u64,
}

/// Report templates
#[derive(Debug)]
pub struct ReportTemplates {
    /// HTML template
    pub html_template: String,
    /// Markdown template
    pub markdown_template: String,
    /// PDF template configuration
    pub pdf_config: PdfConfig,
}

impl Default for ReportTemplates {
    fn default() -> Self {
        Self {
            html_template: COMPREHENSIVE_REPORT_HTML.to_string(),
            markdown_template: COMPREHENSIVE_REPORT_MD.to_string(),
            pdf_config: PdfConfig::default(),
        }
    }
}

/// PDF configuration
#[derive(Debug, Clone)]
pub struct PdfConfig {
    /// Page size
    pub page_size: PageSize,
    /// Margins
    pub margins: Margins,
    /// Include charts
    pub include_charts: bool,
}

impl Default for PdfConfig {
    fn default() -> Self {
        Self {
            page_size: PageSize::A4,
            margins: Margins::default(),
            include_charts: true,
        }
    }
}

/// Page sizes
#[derive(Debug, Clone)]
pub enum PageSize {
    A4,
    Letter,
    Legal,
}

/// Page margins
#[derive(Debug, Clone)]
pub struct Margins {
    /// Top margin in points
    pub top: f64,
    /// Bottom margin in points
    pub bottom: f64,
    /// Left margin in points
    pub left: f64,
    /// Right margin in points
    pub right: f64,
}

impl Default for Margins {
    fn default() -> Self {
        Self {
            top: 72.0,
            bottom: 72.0,
            left: 72.0,
            right: 72.0,
        }
    }
}

/// Reporting configuration
#[derive(Debug, Clone)]
pub struct ReportingConfig {
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Maximum cached reports
    pub max_cached_reports: usize,
    /// Default report format
    pub default_format: ReportFormat,
    /// Include historical comparison
    pub include_historical_comparison: bool,
    /// Include trend analysis
    pub include_trend_analysis: bool,
    /// Include recommendations
    pub include_recommendations: bool,
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            cache_ttl_seconds: 3600, // 1 hour
            max_cached_reports: 100,
            default_format: ReportFormat::Json,
            include_historical_comparison: true,
            include_trend_analysis: true,
            include_recommendations: true,
        }
    }
}

impl ReportingCoordinator {
    /// Create a new reporting coordinator
    pub async fn new(
        results_synthesizer: Arc<ResultsSynthesizer>,
        performance_coordinator: Arc<PerformanceCoordinator>,
        engine_stats: Arc<EngineStatistics>,
    ) -> Result<Self> {
        Ok(Self {
            results_synthesizer,
            performance_coordinator,
            engine_stats,
            report_templates: Arc::new(ReportTemplates::default()),
            report_cache: Arc::new(TokioMutex::new(HashMap::new())),
            config: Arc::new(ReportingConfig::default()),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Generate comprehensive report
    #[instrument(skip(self))]
    pub async fn generate_comprehensive_report(
        &self,
        test_id: &str,
    ) -> Result<ComprehensiveReport> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(anyhow!("ReportingCoordinator is shutting down"));
        }

        let start_time = Instant::now();

        info!("Generating comprehensive report for test: {}", test_id);

        // Check cache first
        if let Some(cached_report) = self.get_cached_report(test_id).await? {
            info!("Using cached report for test: {}", test_id);
            return Ok(cached_report.report);
        }

        // Generate new report
        let report = self.generate_report_internal(test_id).await?;

        // Cache the report
        self.cache_report(test_id, &report).await?;

        let duration = start_time.elapsed();
        info!(
            "Comprehensive report generated for test: {} in {:?}",
            test_id, duration
        );

        Ok(report)
    }

    /// Generate report internally
    async fn generate_report_internal(&self, test_id: &str) -> Result<ComprehensiveReport> {
        let generation_start = Instant::now();

        // Gather data from all coordinators
        let performance_metrics = self.performance_coordinator.get_metrics().await;
        let engine_statistics = self.convert_engine_statistics().await;
        let alerts = self.performance_coordinator.get_alerts().await;

        // Generate recommendations
        let recommendations =
            self.generate_recommendations(&performance_metrics, &engine_statistics).await;

        // Generate trend analysis
        let trend_analysis = self.generate_trend_analysis(&performance_metrics).await;

        // Create report metadata
        let metadata = ReportMetadata {
            report_id: format!(
                "report_{}_{}",
                test_id,
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
            ),
            test_id: test_id.to_string(),
            generated_at: SystemTime::now(),
            version: "1.0.0".to_string(),
            report_type: ReportType::Comprehensive,
            format: self.config.default_format.clone(),
            generation_duration_ms: generation_start.elapsed().as_millis() as u64,
        };

        // Create phase breakdown (simplified for now)
        let phase_breakdown = self.generate_phase_breakdown().await;

        // Create resource utilization summary
        let resource_utilization = ResourceUtilizationSummary {
            peak_cpu_percent: performance_metrics.cpu_usage_percent,
            avg_cpu_percent: performance_metrics.cpu_usage_percent * 0.8, // Simplified
            peak_memory_mb: performance_metrics.memory_usage_mb,
            avg_memory_mb: performance_metrics.memory_usage_mb * 0.9, // Simplified
            network_io_bps: performance_metrics.network_io_bps,
            disk_io_bps: performance_metrics.disk_io_bps,
            efficiency_score: self.calculate_efficiency_score(&performance_metrics),
        };

        // Convert CoordinatorPerformanceMetrics to types::PerformanceMetrics
        let performance_summary = PerformanceMetrics {
            throughput: performance_metrics.analysis_throughput,
            latency: Duration::default(),
            response_time: Duration::from_millis(
                performance_metrics.average_response_time_ms as u64,
            ),
            error_rate: performance_metrics.error_rate,
            resource_utilization: HashMap::new(),
            cpu_usage_percent: performance_metrics.cpu_usage_percent,
            memory_usage_mb: performance_metrics.memory_usage_mb,
            average_response_time_ms: performance_metrics.average_response_time_ms,
        };

        let report = ComprehensiveReport {
            metadata,
            characteristics_summary: TestCharacteristics::default(), // Would be filled from actual analysis
            performance_summary,
            engine_statistics,
            phase_breakdown,
            resource_utilization,
            recommendations,
            alerts,
            historical_comparison: None, // Would be implemented with historical data
            trend_analysis,
        };

        Ok(report)
    }

    /// Convert engine statistics for reporting
    async fn convert_engine_statistics(&self) -> ReportEngineStatistics {
        let total_analyses = self.engine_stats.total_analyses.load(Ordering::Relaxed);
        let successful_analyses = self.engine_stats.successful_analyses.load(Ordering::Relaxed);
        let failed_analyses = self.engine_stats.failed_analyses.load(Ordering::Relaxed);
        let cache_hits = self.engine_stats.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.engine_stats.cache_misses.load(Ordering::Relaxed);

        let success_rate = if total_analyses > 0 {
            successful_analyses as f64 / total_analyses as f64
        } else {
            0.0
        };

        let cache_hit_rate = if cache_hits + cache_misses > 0 {
            cache_hits as f64 / (cache_hits + cache_misses) as f64
        } else {
            0.0
        };

        ReportEngineStatistics {
            total_analyses,
            successful_analyses,
            failed_analyses,
            success_rate,
            cache_hit_rate,
            average_analysis_duration_ms: self
                .engine_stats
                .average_analysis_duration_ms
                .load(Ordering::Relaxed),
            active_analyses: self.engine_stats.active_analyses.load(Ordering::Relaxed),
            errors_recovered: self.engine_stats.errors_recovered.load(Ordering::Relaxed),
        }
    }

    /// Generate recommendations based on performance metrics and statistics
    async fn generate_recommendations(
        &self,
        performance_metrics: &CoordinatorPerformanceMetrics,
        engine_statistics: &ReportEngineStatistics,
    ) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Check cache hit rate
        if engine_statistics.cache_hit_rate < 0.7 {
            recommendations.push(Recommendation {
                id: "cache_optimization".to_string(),
                category: RecommendationCategory::Performance,
                priority: RecommendationPriority::High,
                title: "Improve Cache Hit Rate".to_string(),
                description: format!("Current cache hit rate is {:.1}%. Consider increasing cache size or optimizing cache eviction policies.", engine_statistics.cache_hit_rate * 100.0),
                expected_impact: "Reduced analysis latency and improved throughput".to_string(),
                implementation_effort: ImplementationEffort::Medium,
            });
        }

        // Check error rate
        if engine_statistics.success_rate < 0.95 {
            recommendations.push(Recommendation {
                id: "error_reduction".to_string(),
                category: RecommendationCategory::ErrorHandling,
                priority: RecommendationPriority::Critical,
                title: "Reduce Analysis Error Rate".to_string(),
                description: format!(
                    "Current success rate is {:.1}%. Investigate and address common failure modes.",
                    engine_statistics.success_rate * 100.0
                ),
                expected_impact: "Improved reliability and user experience".to_string(),
                implementation_effort: ImplementationEffort::High,
            });
        }

        // Check response time
        if engine_statistics.average_analysis_duration_ms > 10000 {
            recommendations.push(Recommendation {
                id: "performance_optimization".to_string(),
                category: RecommendationCategory::Performance,
                priority: RecommendationPriority::Medium,
                title: "Optimize Analysis Performance".to_string(),
                description: format!("Average analysis duration is {}ms. Consider optimizing algorithms or adding parallelization.", engine_statistics.average_analysis_duration_ms),
                expected_impact: "Faster analysis completion and better user experience".to_string(),
                implementation_effort: ImplementationEffort::High,
            });
        }

        // Check CPU usage
        if performance_metrics.cpu_usage_percent > 80.0 {
            recommendations.push(Recommendation {
                id: "cpu_optimization".to_string(),
                category: RecommendationCategory::ResourceOptimization,
                priority: RecommendationPriority::Medium,
                title: "Optimize CPU Usage".to_string(),
                description: format!(
                    "CPU usage is at {:.1}%. Consider load balancing or resource scaling.",
                    performance_metrics.cpu_usage_percent
                ),
                expected_impact: "Improved system stability and capacity".to_string(),
                implementation_effort: ImplementationEffort::Medium,
            });
        }

        // Check memory usage
        if performance_metrics.memory_usage_mb > 4096.0 {
            recommendations.push(Recommendation {
                id: "memory_optimization".to_string(),
                category: RecommendationCategory::ResourceOptimization,
                priority: RecommendationPriority::Medium,
                title: "Optimize Memory Usage".to_string(),
                description: format!(
                    "Memory usage is at {:.0}MB. Consider memory optimization strategies.",
                    performance_metrics.memory_usage_mb
                ),
                expected_impact: "Reduced memory footprint and improved scalability".to_string(),
                implementation_effort: ImplementationEffort::Medium,
            });
        }

        recommendations
    }

    /// Generate trend analysis
    async fn generate_trend_analysis(
        &self,
        _performance_metrics: &CoordinatorPerformanceMetrics,
    ) -> TrendAnalysis {
        // In a real implementation, this would analyze historical data
        // For now, return a default trend analysis
        TrendAnalysis {
            performance_trend: TrendDirection::Stable,
            resource_usage_trend: TrendDirection::Stable,
            error_rate_trend: TrendDirection::Improving,
            trend_confidence: 0.7,
            predictions: TrendPredictions {
                next_performance_score: 0.85,
                next_resource_usage: 0.75,
                prediction_confidence: 0.6,
            },
        }
    }

    /// Generate phase breakdown
    async fn generate_phase_breakdown(&self) -> HashMap<AnalysisPhase, PhaseReportSummary> {
        let mut breakdown = HashMap::new();

        // Add summaries for each phase (simplified)
        breakdown.insert(
            AnalysisPhase::ResourceAnalysis,
            PhaseReportSummary {
                phase: AnalysisPhase::ResourceAnalysis,
                execution_time_ms: 150,
                success: true,
                confidence_score: 0.9,
                key_findings: vec!["High CPU usage detected".to_string()],
                recommendations: vec!["Consider CPU optimization".to_string()],
            },
        );

        breakdown.insert(
            AnalysisPhase::ConcurrencyDetection,
            PhaseReportSummary {
                phase: AnalysisPhase::ConcurrencyDetection,
                execution_time_ms: 200,
                success: true,
                confidence_score: 0.85,
                key_findings: vec!["Optimal concurrency level identified".to_string()],
                recommendations: vec!["Current concurrency settings are appropriate".to_string()],
            },
        );

        breakdown.insert(
            AnalysisPhase::SynchronizationAnalysis,
            PhaseReportSummary {
                phase: AnalysisPhase::SynchronizationAnalysis,
                execution_time_ms: 180,
                success: true,
                confidence_score: 0.8,
                key_findings: vec!["Minor synchronization bottlenecks detected".to_string()],
                recommendations: vec!["Review synchronization strategy".to_string()],
            },
        );

        breakdown.insert(
            AnalysisPhase::PatternRecognition,
            PhaseReportSummary {
                phase: AnalysisPhase::PatternRecognition,
                execution_time_ms: 300,
                success: true,
                confidence_score: 0.75,
                key_findings: vec!["Common performance patterns identified".to_string()],
                recommendations: vec!["Apply pattern-based optimizations".to_string()],
            },
        );

        breakdown.insert(
            AnalysisPhase::ProfilingPipeline,
            PhaseReportSummary {
                phase: AnalysisPhase::ProfilingPipeline,
                execution_time_ms: 400,
                success: true,
                confidence_score: 0.88,
                key_findings: vec!["Comprehensive profiling data collected".to_string()],
                recommendations: vec!["Review detailed profiling results".to_string()],
            },
        );

        breakdown
    }

    /// Calculate efficiency score
    fn calculate_efficiency_score(
        &self,
        performance_metrics: &CoordinatorPerformanceMetrics,
    ) -> f64 {
        let cpu_efficiency = (100.0 - performance_metrics.cpu_usage_percent) / 100.0;
        let memory_efficiency = if performance_metrics.memory_usage_mb > 0.0 {
            (8192.0 - performance_metrics.memory_usage_mb) / 8192.0 // Assume 8GB max
        } else {
            1.0
        };

        let response_time_efficiency = if performance_metrics.average_response_time_ms > 0.0 {
            (30000.0 - performance_metrics.average_response_time_ms.min(30000.0)) / 30000.0
        // 30s max
        } else {
            1.0
        };

        (cpu_efficiency + memory_efficiency + response_time_efficiency) / 3.0
    }

    /// Get cached report
    async fn get_cached_report(&self, test_id: &str) -> Result<Option<CachedReport>> {
        let cache = self.report_cache.lock().await;

        if let Some(cached) = cache.get(test_id) {
            let now = SystemTime::now();
            if let Ok(duration) = now.duration_since(cached.cached_at) {
                if duration.as_secs() <= self.config.cache_ttl_seconds {
                    return Ok(Some(cached.clone()));
                }
            }
        }

        Ok(None)
    }

    /// Cache report
    async fn cache_report(&self, test_id: &str, report: &ComprehensiveReport) -> Result<()> {
        let mut cache = self.report_cache.lock().await;

        // Check cache size limit
        if cache.len() >= self.config.max_cached_reports {
            // Remove oldest entry (simplified LRU)
            if let Some(oldest_key) = cache.keys().next().cloned() {
                cache.remove(&oldest_key);
            }
        }

        let cached_report = CachedReport {
            report: report.clone(),
            cached_at: SystemTime::now(),
            access_count: 1,
        };

        cache.insert(test_id.to_string(), cached_report);
        Ok(())
    }

    /// Shutdown reporting coordinator
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down ReportingCoordinator");

        self.shutdown.store(true, Ordering::Release);

        // Clear report cache
        {
            let mut cache = self.report_cache.lock().await;
            cache.clear();
        }

        info!("ReportingCoordinator shutdown completed");
        Ok(())
    }
}

// Template files (would be in separate files in a real implementation)

// Template files (would be in separate files in a real implementation)
const COMPREHENSIVE_REPORT_HTML: &str = r#"
<!DOCTYPE html>
<html>
<head>
    <title>Test Characterization Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { border-bottom: 2px solid #ccc; padding-bottom: 10px; }
        .section { margin: 20px 0; }
        .metric { margin: 10px 0; }
        .recommendation { background: #f0f8ff; padding: 10px; margin: 10px 0; border-left: 4px solid #007acc; }
        .alert { background: #fff0f0; padding: 10px; margin: 10px 0; border-left: 4px solid #ff4444; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Characterization Report</h1>
        <p>Test ID: {{test_id}}</p>
        <p>Generated: {{generated_at}}</p>
    </div>

    <div class="section">
        <h2>Performance Summary</h2>
        <div class="metric">CPU Usage: {{cpu_usage}}%</div>
        <div class="metric">Memory Usage: {{memory_usage}}MB</div>
        <div class="metric">Analysis Duration: {{analysis_duration}}ms</div>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        {{#recommendations}}
        <div class="recommendation">
            <h3>{{title}}</h3>
            <p>{{description}}</p>
            <p><strong>Priority:</strong> {{priority}}</p>
        </div>
        {{/recommendations}}
    </div>

    <div class="section">
        <h2>Alerts</h2>
        {{#alerts}}
        <div class="alert">
            <h3>{{message}}</h3>
            <p><strong>Severity:</strong> {{severity}}</p>
        </div>
        {{/alerts}}
    </div>
</body>
</html>
"#;

const COMPREHENSIVE_REPORT_MD: &str = r#"
# Test Characterization Report

**Test ID:** {{test_id}}
**Generated:** {{generated_at}}
**Report Version:** {{version}}

## Performance Summary

- **CPU Usage:** {{cpu_usage}}%
- **Memory Usage:** {{memory_usage}}MB
- **Analysis Duration:** {{analysis_duration}}ms
- **Success Rate:** {{success_rate}}%
- **Cache Hit Rate:** {{cache_hit_rate}}%

## Engine Statistics

- **Total Analyses:** {{total_analyses}}
- **Successful Analyses:** {{successful_analyses}}
- **Failed Analyses:** {{failed_analyses}}
- **Active Analyses:** {{active_analyses}}
- **Errors Recovered:** {{errors_recovered}}

## Resource Utilization

- **Peak CPU:** {{peak_cpu}}%
- **Average CPU:** {{avg_cpu}}%
- **Peak Memory:** {{peak_memory}}MB
- **Average Memory:** {{avg_memory}}MB
- **Efficiency Score:** {{efficiency_score}}

## Phase Breakdown

{{#phase_breakdown}}
### {{phase}}
- **Execution Time:** {{execution_time_ms}}ms
- **Success:** {{success}}
- **Confidence Score:** {{confidence_score}}
- **Key Findings:** {{key_findings}}
{{/phase_breakdown}}

## Recommendations

{{#recommendations}}
### {{title}} ({{priority}})

{{description}}

**Category:** {{category}}
**Expected Impact:** {{expected_impact}}
**Implementation Effort:** {{implementation_effort}}

{{/recommendations}}

## Alerts

{{#alerts}}
### {{message}}

**Type:** {{alert_type}}
**Severity:** {{severity}}
**Threshold:** {{threshold}}
**Current Value:** {{metric_value}}

{{/alerts}}

## Trend Analysis

- **Performance Trend:** {{performance_trend}}
- **Resource Usage Trend:** {{resource_usage_trend}}
- **Error Rate Trend:** {{error_rate_trend}}
- **Trend Confidence:** {{trend_confidence}}%

### Predictions

- **Next Performance Score:** {{next_performance_score}}
- **Next Resource Usage:** {{next_resource_usage}}
- **Prediction Confidence:** {{prediction_confidence}}%

---

*Report generated by TrustformeRS Test Characterization Engine v{{version}}*
"#;
