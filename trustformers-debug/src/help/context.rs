//! Context-aware help system for TrustformeRS debugging
//!
//! This module provides comprehensive help functionality including contextual
//! assistance, topic search, troubleshooting guides, and interactive help.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Context-aware help system
pub struct ContextHelp {
    help_database: HashMap<String, HelpEntry>,
}

/// Help entry for context help
#[derive(Debug, Clone)]
pub struct HelpEntry {
    pub topic: String,
    pub description: String,
    pub usage: String,
    pub examples: Vec<String>,
    pub related_topics: Vec<String>,
    pub troubleshooting: Vec<String>,
}

impl Default for ContextHelp {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextHelp {
    /// Create new context help system
    pub fn new() -> Self {
        let mut help_database = HashMap::new();

        // Add help entries
        help_database.insert(
            "debug_session".to_string(),
            HelpEntry {
                topic: "Debug Session".to_string(),
                description: "Main debugging session that coordinates all debugging tools"
                    .to_string(),
                usage: "let mut session = debug_session(); session.start().await?;".to_string(),
                examples: vec![
                    "Basic usage: debug_session()".to_string(),
                    "Custom config: debug_session_with_config(config)".to_string(),
                ],
                related_topics: vec!["debug_config".to_string(), "debug_report".to_string()],
                troubleshooting: vec![
                    "If start() fails, check your configuration".to_string(),
                    "Ensure async runtime is available".to_string(),
                ],
            },
        );

        help_database.insert(
            "quick_debug".to_string(),
            HelpEntry {
                topic: "Quick Debug".to_string(),
                description: "One-line debugging with smart defaults".to_string(),
                usage: "let result = debug(&model).await?;".to_string(),
                examples: vec![
                    "Basic: debug(&model).await?".to_string(),
                    "With level: quick_debug(&model, QuickDebugLevel::Light).await?".to_string(),
                ],
                related_topics: vec!["debug_levels".to_string(), "simplified_results".to_string()],
                troubleshooting: vec![
                    "Use Light level for better performance".to_string(),
                    "Check has_critical_issues() for problems".to_string(),
                ],
            },
        );

        // Add more help entries...

        Self { help_database }
    }

    /// Get help for a topic
    pub fn get_help(&self, topic: &str) -> Option<&HelpEntry> {
        self.help_database.get(topic)
    }

    /// Search help topics
    pub fn search(&self, query: &str) -> Vec<&HelpEntry> {
        self.help_database
            .values()
            .filter(|entry| {
                entry.topic.to_lowercase().contains(&query.to_lowercase())
                    || entry.description.to_lowercase().contains(&query.to_lowercase())
            })
            .collect()
    }

    /// Get all available topics
    pub fn available_topics(&self) -> Vec<String> {
        self.help_database.keys().cloned().collect()
    }

    /// Get contextual help based on current operation
    pub fn contextual_help(&self, context: &str) -> Vec<&HelpEntry> {
        match context.to_lowercase().as_str() {
            "gradient" => self.search("gradient"),
            "memory" => self.search("memory"),
            "performance" => self.search("performance"),
            "anomaly" => self.search("anomaly"),
            _ => vec![],
        }
    }
}

/// Global context help instance
static CONTEXT_HELP: OnceLock<ContextHelp> = OnceLock::new();

/// Get global context help instance
pub fn context_help() -> &'static ContextHelp {
    CONTEXT_HELP.get_or_init(ContextHelp::new)
}

/// Display help for a topic
pub fn help(topic: &str) {
    if let Some(entry) = context_help().get_help(topic) {
        println!("=== {} ===", entry.topic);
        println!("{}", entry.description);
        println!("\nUsage:");
        println!("{}", entry.usage);
        println!("\nExamples:");
        for example in &entry.examples {
            println!("  {}", example);
        }
        if !entry.related_topics.is_empty() {
            println!("\nRelated topics:");
            for topic in &entry.related_topics {
                println!("  {}", topic);
            }
        }
        if !entry.troubleshooting.is_empty() {
            println!("\nTroubleshooting:");
            for tip in &entry.troubleshooting {
                println!("  {}", tip);
            }
        }
    } else {
        println!("Help topic '{}' not found.", topic);
        println!("Available topics:");
        for topic in context_help().available_topics() {
            println!("  {}", topic);
        }
    }
}

/// Search help topics
pub fn help_search(query: &str) {
    let results = context_help().search(query);
    if results.is_empty() {
        println!("No help topics found for '{}'", query);
    } else {
        println!("Help topics matching '{}':", query);
        for entry in results {
            println!("  {} - {}", entry.topic, entry.description);
        }
    }
}
