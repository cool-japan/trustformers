//! # Message Formatting and Templating System
//!
//! Provides message formatting with template engine and channel-specific formatters.

use super::types::*;
use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::{
    collections::HashMap,
    fmt::{self, Debug},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

#[derive(Debug)]

pub struct MessageFormatter {
    /// Configuration
    config: NotificationConfig,

    /// Template cache for performance
    template_cache: Arc<DashMap<String, CompiledTemplate>>,

    /// Template engine
    template_engine: Arc<TemplateEngine>,

    /// Channel formatters
    channel_formatters: Arc<DashMap<String, Arc<dyn ChannelFormatter + Send + Sync>>>,

    /// Formatting statistics
    stats: Arc<FormattingStats>,
}

/// Compiled template for efficient rendering
#[derive(Debug, Clone)]
pub struct CompiledTemplate {
    /// Template name
    pub name: String,

    /// Template content
    pub content: String,

    /// Required variables
    pub required_vars: Vec<String>,

    /// Optional variables with defaults
    pub optional_vars: HashMap<String, String>,

    /// Template metadata
    pub metadata: HashMap<String, String>,

    /// Compilation timestamp
    pub compiled_at: DateTime<Utc>,
}

/// Template engine for processing templates
#[derive(Debug)]

pub struct TemplateEngine {
    /// Available functions for templates
    functions: Arc<DashMap<String, Arc<dyn TemplateFunction + Send + Sync>>>,
}

/// Template function trait for extending template capabilities
pub trait TemplateFunction: fmt::Debug {
    /// Function name
    fn name(&self) -> &str;

    /// Execute function with arguments
    fn execute(&self, args: &[String]) -> Result<String>;

    /// Get function signature
    fn signature(&self) -> String;
}

/// Channel-specific formatter trait
pub trait ChannelFormatter: Debug {
    /// Format message for specific channel
    fn format_for_channel(&self, content: &str, channel: &str) -> Result<String>;

    /// Get supported channels
    fn supported_channels(&self) -> Vec<String>;

    /// Get maximum message length for channel
    fn max_length(&self, channel: &str) -> Option<usize>;
}

/// Formatting statistics
#[derive(Debug, Default)]
pub struct FormattingStats {
    /// Templates rendered
    pub templates_rendered: AtomicU64,

    /// Cache hits
    pub cache_hits: AtomicU64,

    /// Cache misses
    pub cache_misses: AtomicU64,

    /// Formatting errors
    pub formatting_errors: AtomicU64,

    /// Average formatting time (ms)
    pub avg_formatting_time_ms: AtomicF32,
}

impl MessageFormatter {
    pub async fn new(config: NotificationConfig) -> Result<Self> {
        let template_cache = Arc::new(DashMap::new());
        let template_engine = Arc::new(TemplateEngine::new().await?);
        let channel_formatters = Arc::new(DashMap::new());

        let formatter = Self {
            config,
            template_cache,
            template_engine,
            channel_formatters,
            stats: Arc::new(FormattingStats::default()),
        };

        // Initialize default templates and formatters
        formatter.initialize_default_templates().await?;
        formatter.initialize_channel_formatters().await?;

        Ok(formatter)
    }

    pub async fn format_notification(&self, notification: &Notification) -> Result<Notification> {
        let start_time = Instant::now();
        let mut formatted_notification = notification.clone();

        // Format using template if specified
        if let Some(template_name) = &notification.template {
            formatted_notification =
                self.apply_template(&formatted_notification, template_name).await?;
        }

        // Apply channel-specific formatting
        formatted_notification = self.apply_channel_formatting(&formatted_notification).await?;

        // Apply content enrichment
        formatted_notification = self.enrich_content(&formatted_notification).await?;

        // Update statistics
        self.stats.templates_rendered.fetch_add(1, Ordering::Relaxed);
        let formatting_time = start_time.elapsed().as_millis() as f32;
        let current_avg = self.stats.avg_formatting_time_ms.load(Ordering::Relaxed);
        let new_avg = if current_avg == 0.0 {
            formatting_time
        } else {
            (current_avg * 0.9) + (formatting_time * 0.1)
        };
        self.stats.avg_formatting_time_ms.store(new_avg, Ordering::Relaxed);

        Ok(formatted_notification)
    }

    /// Add a custom template
    pub async fn add_template(&self, template: CompiledTemplate) -> Result<()> {
        self.template_cache.insert(template.name.clone(), template);
        Ok(())
    }

    /// Remove a template
    pub async fn remove_template(&self, template_name: &str) -> Result<()> {
        self.template_cache.remove(template_name);
        Ok(())
    }

    /// Get formatting statistics
    pub fn get_stats(&self) -> FormattingStats {
        FormattingStats {
            templates_rendered: AtomicU64::new(
                self.stats.templates_rendered.load(Ordering::Relaxed),
            ),
            cache_hits: AtomicU64::new(self.stats.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.stats.cache_misses.load(Ordering::Relaxed)),
            formatting_errors: AtomicU64::new(self.stats.formatting_errors.load(Ordering::Relaxed)),
            avg_formatting_time_ms: AtomicF32::new(
                self.stats.avg_formatting_time_ms.load(Ordering::Relaxed),
            ),
        }
    }

    // Private implementation methods

    async fn apply_template(
        &self,
        notification: &Notification,
        template_name: &str,
    ) -> Result<Notification> {
        // Try to get template from cache
        let template = if let Some(template) = self.template_cache.get(template_name) {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            template.clone()
        } else {
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
            // Load template (in real implementation, this would load from storage)
            self.load_template(template_name).await?
        };

        // Render template with variables
        let rendered_content = self.render_template(&template, &notification.template_vars).await?;
        let rendered_subject = self
            .render_template_string(&notification.subject, &notification.template_vars)
            .await?;

        let mut formatted_notification = notification.clone();
        formatted_notification.content = rendered_content;
        formatted_notification.subject = rendered_subject;

        Ok(formatted_notification)
    }

    async fn apply_channel_formatting(&self, notification: &Notification) -> Result<Notification> {
        let mut formatted_notification = notification.clone();

        // Apply channel-specific formatting for each target channel
        for channel in &notification.channels {
            if let Some(formatter) = self.channel_formatters.get(channel) {
                // Format content for this channel
                let formatted_content =
                    formatter.format_for_channel(&notification.content, channel)?;

                // Check length limits
                if let Some(max_length) = formatter.max_length(channel) {
                    if formatted_content.len() > max_length {
                        let truncated = self.truncate_content(&formatted_content, max_length);
                        formatted_notification.content = truncated;
                    } else {
                        formatted_notification.content = formatted_content;
                    }
                } else {
                    formatted_notification.content = formatted_content;
                }
            }
        }

        Ok(formatted_notification)
    }

    async fn enrich_content(&self, notification: &Notification) -> Result<Notification> {
        let mut enriched_notification = notification.clone();

        // Add timestamp if not present
        if !enriched_notification.content.contains("Time:") {
            enriched_notification.content = format!(
                "{}\n\nTime: {}",
                enriched_notification.content,
                enriched_notification.created_at.format("%Y-%m-%d %H:%M:%S UTC")
            );
        }

        // Add severity indicators
        let severity_indicator = match enriched_notification.severity {
            SeverityLevel::Critical => "🚨 CRITICAL",
            SeverityLevel::High => "⚠️ HIGH",
            SeverityLevel::Medium => "📢 MEDIUM",
            SeverityLevel::Low => "💡 LOW",
            SeverityLevel::Info => "ℹ️ INFO",
            SeverityLevel::Warning => "⚠️ WARNING",
        };

        if !enriched_notification.subject.starts_with("🚨")
            && !enriched_notification.subject.starts_with("⚠️")
            && !enriched_notification.subject.starts_with("📢")
            && !enriched_notification.subject.starts_with("💡")
            && !enriched_notification.subject.starts_with("ℹ️")
        {
            enriched_notification.subject =
                format!("{} {}", severity_indicator, enriched_notification.subject);
        }

        // Add correlation info if present
        if let Some(correlation_id) = &enriched_notification.correlation_id {
            enriched_notification.content = format!(
                "{}\n\nCorrelation ID: {}",
                enriched_notification.content, correlation_id
            );
        }

        Ok(enriched_notification)
    }

    async fn render_template(
        &self,
        template: &CompiledTemplate,
        vars: &HashMap<String, String>,
    ) -> Result<String> {
        self.template_engine.render(&template.content, vars).await
    }

    async fn render_template_string(
        &self,
        template_str: &str,
        vars: &HashMap<String, String>,
    ) -> Result<String> {
        self.template_engine.render(template_str, vars).await
    }

    async fn load_template(&self, template_name: &str) -> Result<CompiledTemplate> {
        // In a real implementation, this would load from persistent storage
        // For now, return a default template based on name
        let template = match template_name {
            "default_alert" => CompiledTemplate {
                name: template_name.to_string(),
                content: "Alert: {{threshold_name}}\nCurrent Value: {{current_value}}\nThreshold: {{threshold_value}}\n\nPlease investigate this issue.".to_string(),
                required_vars: vec!["threshold_name".to_string(), "current_value".to_string(), "threshold_value".to_string()],
                optional_vars: HashMap::new(),
                metadata: HashMap::new(),
                compiled_at: Utc::now(),
            },
            "performance_alert" => CompiledTemplate {
                name: template_name.to_string(),
                content: "Performance Alert: {{metric}}\nCurrent: {{current_value}}\nThreshold: {{threshold_value}}\nImpact: {{impact}}\n\nImmediate attention required for performance optimization.".to_string(),
                required_vars: vec!["metric".to_string(), "current_value".to_string(), "threshold_value".to_string()],
                optional_vars: {
                    let mut vars = HashMap::new();
                    vars.insert("impact".to_string(), "Unknown".to_string());
                    vars
                },
                metadata: HashMap::new(),
                compiled_at: Utc::now(),
            },
            "resource_alert" => CompiledTemplate {
                name: template_name.to_string(),
                content: "Resource Alert: {{resource_type}}\nUtilization: {{utilization}}\nThreshold: {{threshold}}\n\nResource scaling or optimization may be required.".to_string(),
                required_vars: vec!["resource_type".to_string(), "utilization".to_string(), "threshold".to_string()],
                optional_vars: HashMap::new(),
                metadata: HashMap::new(),
                compiled_at: Utc::now(),
            },
            "critical_alert" => CompiledTemplate {
                name: template_name.to_string(),
                content: "🚨 CRITICAL SYSTEM ALERT 🚨\n\nSeverity: {{severity}}\nMetric: {{metric}}\nCurrent: {{current_value}}\nThreshold: {{threshold_value}}\nTime: {{alert_time}}\n\n🔥 IMMEDIATE ACTION REQUIRED 🔥\n\nThis is a critical system issue that requires immediate attention. Please escalate if not resolved within 15 minutes.".to_string(),
                required_vars: vec!["severity".to_string(), "metric".to_string(), "current_value".to_string(), "threshold_value".to_string(), "alert_time".to_string()],
                optional_vars: HashMap::new(),
                metadata: HashMap::new(),
                compiled_at: Utc::now(),
            },
            _ => {
                return Err(anyhow!("Template not found: {}", template_name));
            }
        };

        // Cache the template
        self.template_cache.insert(template_name.to_string(), template.clone());
        Ok(template)
    }

    fn truncate_content(&self, content: &str, max_length: usize) -> String {
        if content.len() <= max_length {
            return content.to_string();
        }

        let truncate_point = max_length.saturating_sub(20); // Reserve space for truncation message
        let truncated = &content[..truncate_point];
        format!("{}... [truncated]", truncated)
    }

    async fn initialize_default_templates(&self) -> Result<()> {
        // Templates are loaded on-demand via load_template method
        Ok(())
    }

    async fn initialize_channel_formatters(&self) -> Result<()> {
        // Add default channel formatters
        self.channel_formatters
            .insert("log".to_string(), Arc::new(LogChannelFormatter::new()));
        self.channel_formatters
            .insert("email".to_string(), Arc::new(EmailChannelFormatter::new()));
        self.channel_formatters
            .insert("slack".to_string(), Arc::new(SlackChannelFormatter::new()));
        self.channel_formatters.insert(
            "webhook".to_string(),
            Arc::new(WebhookChannelFormatter::new()),
        );

        Ok(())
    }
}

impl TemplateEngine {
    pub async fn new() -> Result<Self> {
        let functions = Arc::new(DashMap::new());

        let engine = Self { functions };

        // Register default template functions
        engine.register_default_functions().await?;

        Ok(engine)
    }

    pub async fn render(&self, template: &str, vars: &HashMap<String, String>) -> Result<String> {
        let mut result = template.to_string();

        // Simple variable substitution ({{variable_name}})
        for (key, value) in vars {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }

        // Process template functions ({{function_name(args)}})
        result = self.process_functions(&result).await?;

        Ok(result)
    }

    async fn register_default_functions(&self) -> Result<()> {
        self.functions.insert("now".to_string(), Arc::new(NowFunction));
        self.functions.insert(
            "format_duration".to_string(),
            Arc::new(FormatDurationFunction),
        );
        self.functions.insert("upper".to_string(), Arc::new(UpperFunction));
        self.functions.insert("lower".to_string(), Arc::new(LowerFunction));

        Ok(())
    }

    async fn process_functions(&self, template: &str) -> Result<String> {
        // Simple function processing - in production, you'd use a proper template engine
        let mut result = template.to_string();

        // Look for function calls like {{function_name(args)}}
        // This is a simplified implementation
        if result.contains("{{now()}}") {
            result = result.replace(
                "{{now()}}",
                &Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
            );
        }

        Ok(result)
    }
}

// Default template functions
#[derive(Debug)]
struct NowFunction;
impl TemplateFunction for NowFunction {
    fn name(&self) -> &str {
        "now"
    }
    fn execute(&self, _args: &[String]) -> Result<String> {
        Ok(Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string())
    }
    fn signature(&self) -> String {
        "now() -> String".to_string()
    }
}

#[derive(Debug)]
struct FormatDurationFunction;
impl TemplateFunction for FormatDurationFunction {
    fn name(&self) -> &str {
        "format_duration"
    }
    fn execute(&self, args: &[String]) -> Result<String> {
        if args.is_empty() {
            return Ok("N/A".to_string());
        }
        // Simple duration formatting
        Ok(format!("{}s", args[0]))
    }
    fn signature(&self) -> String {
        "format_duration(seconds) -> String".to_string()
    }
}

#[derive(Debug)]
struct UpperFunction;
impl TemplateFunction for UpperFunction {
    fn name(&self) -> &str {
        "upper"
    }
    fn execute(&self, args: &[String]) -> Result<String> {
        if args.is_empty() {
            return Ok(String::new());
        }
        Ok(args[0].to_uppercase())
    }
    fn signature(&self) -> String {
        "upper(text) -> String".to_string()
    }
}

#[derive(Debug)]
struct LowerFunction;
impl TemplateFunction for LowerFunction {
    fn name(&self) -> &str {
        "lower"
    }
    fn execute(&self, args: &[String]) -> Result<String> {
        if args.is_empty() {
            return Ok(String::new());
        }
        Ok(args[0].to_lowercase())
    }
    fn signature(&self) -> String {
        "lower(text) -> String".to_string()
    }
}

// Channel-specific formatters
#[derive(Debug)]
struct LogChannelFormatter;
impl LogChannelFormatter {
    fn new() -> Self {
        Self
    }
}

impl ChannelFormatter for LogChannelFormatter {
    fn format_for_channel(&self, content: &str, _channel: &str) -> Result<String> {
        // Log formatting - keep it simple and readable
        Ok(content.to_string())
    }

    fn supported_channels(&self) -> Vec<String> {
        vec!["log".to_string()]
    }

    fn max_length(&self, _channel: &str) -> Option<usize> {
        None // No length limit for logs
    }
}

#[derive(Debug)]
struct EmailChannelFormatter;
impl EmailChannelFormatter {
    fn new() -> Self {
        Self
    }
}

impl ChannelFormatter for EmailChannelFormatter {
    fn format_for_channel(&self, content: &str, _channel: &str) -> Result<String> {
        // Email formatting - add HTML structure
        let formatted = format!(
            "<html><body><pre style=\"font-family: monospace; white-space: pre-wrap;\">{}</pre></body></html>",
            content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        );
        Ok(formatted)
    }

    fn supported_channels(&self) -> Vec<String> {
        vec!["email".to_string()]
    }

    fn max_length(&self, _channel: &str) -> Option<usize> {
        Some(100_000) // 100KB limit for emails
    }
}

#[derive(Debug)]
struct SlackChannelFormatter;
impl SlackChannelFormatter {
    fn new() -> Self {
        Self
    }
}

impl ChannelFormatter for SlackChannelFormatter {
    fn format_for_channel(&self, content: &str, _channel: &str) -> Result<String> {
        // Slack formatting - convert to Slack markdown
        let mut formatted = content.to_string();

        // Convert basic formatting to Slack syntax
        formatted = formatted.replace("**", "*"); // Bold
        formatted = formatted.replace("__", "_"); // Italic

        // Add code blocks for structured content
        if formatted.contains("Current Value:") || formatted.contains("Threshold:") {
            formatted = format!("```\n{}\n```", formatted);
        }

        Ok(formatted)
    }

    fn supported_channels(&self) -> Vec<String> {
        vec!["slack".to_string()]
    }

    fn max_length(&self, _channel: &str) -> Option<usize> {
        Some(40_000) // Slack message limit
    }
}

#[derive(Debug)]
struct WebhookChannelFormatter;
impl WebhookChannelFormatter {
    fn new() -> Self {
        Self
    }
}

impl ChannelFormatter for WebhookChannelFormatter {
    fn format_for_channel(&self, content: &str, _channel: &str) -> Result<String> {
        // Webhook formatting - ensure JSON-safe content
        let formatted = content
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t");
        Ok(formatted)
    }

    fn supported_channels(&self) -> Vec<String> {
        vec!["webhook".to_string()]
    }

    fn max_length(&self, _channel: &str) -> Option<usize> {
        Some(1_000_000) // 1MB limit for webhooks
    }
}
