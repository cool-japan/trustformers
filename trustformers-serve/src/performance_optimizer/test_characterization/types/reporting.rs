use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// Import commonly used types from core
use super::core::{TestCharacterizationError, TestCharacterizationResult};

// Import cross-module types
use super::alerts::DashboardData;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Performance optimization
    Performance,
    /// Resource optimization
    Resource,
    /// Concurrency optimization
    Concurrency,
    /// Configuration change
    Configuration,
    /// Algorithm improvement
    Algorithm,
    /// Infrastructure upgrade
    Infrastructure,
    /// Code refactoring
    CodeRefactoring,
    /// Testing strategy
    TestingStrategy,
    /// Monitoring enhancement
    Monitoring,
    /// Security improvement
    Security,
    /// Serial execution (no concurrency)
    SerialExecution,
}

#[derive(Debug, Clone)]
pub struct ExecutiveSummary {
    pub key_findings: Vec<String>,
    pub overall_assessment: String,
    pub critical_issues: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct InsightGenerationMetrics {
    pub insights_generated: usize,
    pub average_confidence: f64,
    pub generation_time: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct InsightModel {
    pub model_type: String,
    pub accuracy: f64,
    pub training_data_size: usize,
}

impl InsightModel {
    /// Record that `data` was observed.
    ///
    /// Only `training_data_size` changes: no model is fitted here, and
    /// `accuracy` is therefore left exactly as the caller set it rather than
    /// being nudged toward a number no evaluation produced. Before 0.2.1 the
    /// method was named as an update and documented as "would retrain", which
    /// read as though the model had learned something.
    pub fn record_observed_data(&mut self, data: &[f64]) {
        self.training_data_size += data.len();
    }
}

#[derive(Debug, Clone)]
pub struct InsightType {
    pub insight_type: String,
    pub category: String,
    pub priority: u32,
}

#[derive(Debug, Clone)]
pub struct InsightsGeneratorConfig {
    pub enabled: bool,
    pub min_confidence: f64,
    pub max_insights: usize,
    pub generation_interval: std::time::Duration,
    pub recommendation_config: String,
}

impl Default for InsightsGeneratorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence: 0.7,
            max_insights: 10,
            generation_interval: std::time::Duration::from_secs(5),
            recommendation_config: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InsightsReportGenerator {
    pub insight_types: Vec<String>,
    pub confidence_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct OutputFormat {
    pub format_type: String,
    pub encoding: String,
    pub compression: bool,
}

#[derive(Debug, Clone)]
pub struct PublishingStatistics {
    pub total_published: usize,
    pub publish_success_rate: f64,
    pub average_publish_time: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct Recommendation {
    pub recommendation_id: String,
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub priority: u32,
    pub expected_impact: f64,
}

#[derive(Debug, Clone)]
pub struct RecommendationStrategy {
    pub strategy_name: String,
    pub applicable_scenarios: Vec<String>,
    pub effectiveness: f64,
}

#[derive(Debug, Clone)]
/// Produces recommendations from the strategies registered with it.
///
/// `strategies` used to be a `Vec<String>` of bare names carrying no
/// effectiveness figure, so nothing could be ranked against
/// `confidence_threshold` even in principle; `generate_recommendations`
/// returned an empty list regardless.
pub struct RecommendationSystem {
    pub strategies: Vec<RecommendationStrategy>,
    pub confidence_threshold: f64,
    pub max_recommendations: usize,
    running: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Debug, Clone)]
pub struct RecommendationValidator {
    pub validation_rules: Vec<String>,
    pub strict_mode: bool,
}

#[derive(Debug, Clone)]
pub struct ReportGeneratorConfig {
    pub output_format: String,
    pub include_visualizations: bool,
    pub max_report_size_mb: usize,
}

pub trait ReportGenerator: std::fmt::Debug + Send + Sync {
    fn generate(&self) -> String;

    /// Generate a detailed report
    fn generate_report(&self) -> TestCharacterizationResult<String> {
        Ok(self.generate())
    }
}

#[derive(Debug, Clone)]
pub struct ReportInsight {
    pub insight_type: String,
    pub description: String,
    pub confidence: f64,
    pub supporting_data: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ReportMetadata {
    pub report_id: String,
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub generated_by: String,
    pub report_version: String,
}

#[derive(Debug, Clone)]
pub struct ReportNotificationManager {
    pub notification_channels: Vec<String>,
    pub notification_templates: HashMap<String, String>,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ReportParameter {
    pub parameter_name: String,
    pub parameter_value: String,
    pub parameter_type: String,
}

#[derive(Debug, Clone)]
pub struct ReportSchedule {
    pub schedule_id: String,
    pub cron_expression: String,
    pub enabled: bool,
    pub last_run: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone)]
pub struct ReportStyling {
    pub theme: String,
    pub color_scheme: Vec<String>,
    pub font_family: String,
}

#[derive(Debug, Clone)]
pub struct ReportTemplateMetadata {
    pub template_id: String,
    pub template_name: String,
    pub template_version: String,
    pub supported_formats: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ReportingEngineConfig {
    pub engine_type: String,
    pub max_concurrent_reports: usize,
    pub cache_enabled: bool,
    pub report_generation_interval: std::time::Duration,
    pub dashboard_config: String,
}

impl Default for ReportingEngineConfig {
    fn default() -> Self {
        Self {
            engine_type: String::from("default"),
            max_concurrent_reports: 10,
            cache_enabled: true,
            report_generation_interval: std::time::Duration::from_secs(30),
            dashboard_config: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReportingMetrics {
    pub reports_generated: usize,
    pub average_generation_time: std::time::Duration,
    pub error_rate: f64,
}

#[derive(Debug, Clone)]
pub struct SubscriptionConfig {
    pub subscription_id: String,
    pub report_types: Vec<String>,
    pub delivery_schedule: String,
    pub recipients: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SubscriptionError {
    pub error_type: String,
    pub error_message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct SubscriptionFilter {
    pub filter_type: String,
    pub filter_criteria: HashMap<String, String>,
    pub include_patterns: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SubscriptionMetrics {
    pub active_subscriptions: usize,
    pub delivery_success_rate: f64,
    pub average_delivery_time: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct SubscriptionTemplateType {
    pub template_type: String,
    pub template_category: String,
    pub customizable: bool,
}

#[derive(Debug, Clone)]
pub struct SubscriptionType {
    pub subscription_type: String,
    pub frequency: String,
    pub priority: u32,
}

#[derive(Debug, Clone)]
pub struct UserPreferences {
    pub user_id: String,
    pub preferred_format: String,
    pub notification_preferences: HashMap<String, bool>,
    pub timezone: String,
}

#[derive(Debug, Clone)]
pub struct UserRole {
    pub role_name: String,
    pub permissions: Vec<String>,
    pub access_level: u32,
}

#[derive(Debug, Clone)]
pub struct Visualization {
    pub visualization_id: String,
    pub visualization_type: String,
    pub data_source: String,
    pub config: VisualizationConfig,
}

#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    pub chart_type: String,
    pub dimensions: (u32, u32),
    pub color_palette: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct VisualizationContent {
    pub content_type: String,
    pub content_data: Vec<u8>,
    pub mime_type: String,
}

#[derive(Debug, Clone)]
pub struct VisualizationOutput {
    pub output_format: String,
    pub output_data: Vec<u8>,
    pub metadata: HashMap<String, String>,
}

/// Visualization engine trait for dashboard rendering
pub trait VisualizationEngine: std::fmt::Debug + Send + Sync {
    /// Render visualization for dashboard data
    fn render(&self, data: &DashboardData) -> TestCharacterizationResult<VisualizationOutput>;

    /// Get engine name
    fn name(&self) -> &str;

    /// Get supported visualization types
    fn supported_types(&self) -> Vec<VisualizationContent>;

    /// Update visualization configuration
    fn configure(&mut self, config: HashMap<String, String>) -> TestCharacterizationResult<()>;
}

// Trait implementations

pub trait OutputFormatter: std::fmt::Debug + Send + Sync {
    fn format(&self) -> String;

    /// Format a report with the given data
    fn format_report(&self, data: &str) -> TestCharacterizationResult<String> {
        Ok(format!("{}\n{}", self.format(), data))
    }
}

impl RecommendationSystem {
    /// Create a new RecommendationSystem with default settings
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            confidence_threshold: 0.7,
            max_recommendations: 10,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Register a strategy the system may recommend.
    pub fn register_strategy(&mut self, strategy: RecommendationStrategy) {
        self.strategies.push(strategy);
    }

    /// Start the recommendation system.
    pub fn start_recommendations(&self) -> TestCharacterizationResult<()> {
        self.running.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Stop the recommendation system.
    pub fn stop_recommendations(&self) -> TestCharacterizationResult<()> {
        self.running.store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Whether the system is running right now.
    pub fn is_running(&self) -> bool {
        self.running.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Recommendations from the registered strategies that clear the
    /// confidence threshold.
    ///
    /// Before 0.2.1 this returned `Ok(Vec::new())` whatever was registered, so
    /// `strategies` and `confidence_threshold` were both inert and every caller
    /// saw an empty list it could not distinguish from "nothing to recommend".
    pub fn generate_recommendations(&self) -> TestCharacterizationResult<Vec<Recommendation>> {
        if !self.is_running() {
            return Err(TestCharacterizationError::InvalidInput {
                message: "recommendation system is not running; call start_recommendations first"
                    .to_string(),
                field: "running".to_string(),
                value: "false".to_string(),
            });
        }
        let mut recommendations: Vec<Recommendation> = self
            .strategies
            .iter()
            .filter(|strategy| strategy.effectiveness >= self.confidence_threshold)
            .map(|strategy| Recommendation {
                recommendation_id: format!("strategy:{}", strategy.strategy_name),
                recommendation_type: RecommendationType::Performance,
                description: format!(
                    "`{}` applies to {} scenario(s) with a recorded effectiveness of {:.3}",
                    strategy.strategy_name,
                    strategy.applicable_scenarios.len(),
                    strategy.effectiveness
                ),
                // Priority orders by the strategy's own recorded effectiveness;
                // the strongest strategy is priority 1.
                priority: 1,
                expected_impact: strategy.effectiveness,
            })
            .collect();
        recommendations.sort_by(|a, b| {
            b.expected_impact
                .partial_cmp(&a.expected_impact)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (index, recommendation) in recommendations.iter_mut().enumerate() {
            recommendation.priority = index as u32 + 1;
        }
        recommendations.truncate(self.max_recommendations);
        Ok(recommendations)
    }
}

impl Default for RecommendationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl InsightGenerationMetrics {
    /// Create a new InsightGenerationMetrics with default values
    pub fn new() -> Self {
        Self {
            insights_generated: 0,
            average_confidence: 0.0,
            generation_time: std::time::Duration::from_secs(0),
        }
    }

    /// Increment the number of insights generated
    pub fn increment_insights_generated(&mut self) {
        self.insights_generated += 1;
    }
}

impl Default for InsightGenerationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl InsightsReportGenerator {
    /// Create a new InsightsReportGenerator with default settings
    pub fn new() -> Self {
        Self {
            insight_types: Vec::new(),
            confidence_threshold: 0.7,
        }
    }
}

impl Default for InsightsReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ReportGenerator for InsightsReportGenerator {
    fn generate(&self) -> String {
        format!(
            "Insights Report Generator (types={}, confidence_threshold={:.2})",
            self.insight_types.len(),
            self.confidence_threshold
        )
    }
}

impl ReportingMetrics {
    /// Create a new ReportingMetrics with default values
    pub fn new() -> Self {
        Self {
            reports_generated: 0,
            average_generation_time: std::time::Duration::from_secs(0),
            error_rate: 0.0,
        }
    }

    /// Increment the number of reports generated
    pub fn increment_reports_generated(&mut self) {
        self.reports_generated += 1;
    }
}

impl Default for ReportingMetrics {
    fn default() -> Self {
        Self::new()
    }
}
