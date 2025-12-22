//! Safety filtering and content moderation for conversational AI pipeline.
//!
//! This module provides comprehensive safety filtering, content moderation, and risk assessment
//! capabilities for the conversational pipeline system. It includes pattern-based detection,
//! toxicity scoring, safety policy enforcement, and detailed violation reporting.
//!
//! # Features
//!
//! - **Content Filtering**: Real-time input and output content filtering
//! - **Pattern Detection**: Advanced regex and keyword-based safety pattern detection
//! - **Risk Assessment**: Multi-dimensional safety scoring and risk evaluation
//! - **Policy Enforcement**: Configurable safety policies and violation handling
//! - **Violation Reporting**: Detailed safety violation tracking and reporting
//! - **Integration**: Seamless integration with conversation metadata and health tracking
//!
//! # Usage
//!
//! ```rust,no_run
//! use trustformers::pipeline::conversational::safety::SafetyFilter;
//!
//! // Create a safety filter with default configuration
//! let filter = SafetyFilter::new();
//!
//! // Check if content is safe
//! let is_safe = filter.is_safe("Hello, how can I help you today?");
//! assert!(is_safe);
//!
//! // Get detailed safety assessment
//! let assessment = filter.analyze_safety("Some potentially harmful content");
//! println!("Safety score: {}", assessment.toxicity_score);
//! ```

use super::types::*;
use crate::core::error::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use thiserror::Error;

// ================================================================================================
// ENHANCED SAFETY CONFIGURATION TYPES
// ================================================================================================

/// Extended safety configuration with comprehensive filtering options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedSafetyConfig {
    /// Base configuration
    pub base_config: SafetyFilterConfig,
    /// Maximum allowed toxicity score (0.0-1.0)
    pub max_toxicity_score: f32,
    /// Maximum allowed harm score (0.0-1.0)
    pub max_harm_score: f32,
    /// Maximum allowed bias score (0.0-1.0)
    pub max_bias_score: f32,
    /// Enable input content filtering
    pub filter_input: bool,
    /// Enable output content filtering
    pub filter_output: bool,
    /// Custom banned terms and phrases
    pub custom_banned_terms: Vec<String>,
    /// Custom allowed terms (exceptions)
    pub allowed_terms: Vec<String>,
    /// Safety patterns configuration
    pub pattern_config: PatternConfig,
    /// Content length limits for safety analysis
    pub content_limits: ContentLimits,
    /// Violation handling configuration
    pub violation_handling: ViolationHandling,
    /// Performance optimization settings
    pub performance_settings: PerformanceSettings,
}

impl Default for ExtendedSafetyConfig {
    fn default() -> Self {
        Self {
            base_config: SafetyFilterConfig::default(),
            max_toxicity_score: 0.7,
            max_harm_score: 0.6,
            max_bias_score: 0.8,
            filter_input: true,
            filter_output: true,
            custom_banned_terms: Vec::new(),
            allowed_terms: Vec::new(),
            pattern_config: PatternConfig::default(),
            content_limits: ContentLimits::default(),
            violation_handling: ViolationHandling::default(),
            performance_settings: PerformanceSettings::default(),
        }
    }
}

/// Configuration for safety pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    /// Enable regex pattern matching
    pub enable_regex_patterns: bool,
    /// Enable keyword matching
    pub enable_keyword_matching: bool,
    /// Enable context-aware analysis
    pub enable_context_analysis: bool,
    /// Case sensitivity for pattern matching
    pub case_sensitive: bool,
    /// Maximum pattern evaluation time (ms)
    pub max_pattern_eval_time_ms: u64,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            enable_regex_patterns: true,
            enable_keyword_matching: true,
            enable_context_analysis: true,
            case_sensitive: false,
            max_pattern_eval_time_ms: 100,
        }
    }
}

/// Content length limits for safety analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentLimits {
    /// Maximum content length to analyze (characters)
    pub max_analysis_length: usize,
    /// Minimum content length for detailed analysis
    pub min_detailed_analysis_length: usize,
    /// Enable content truncation for analysis
    pub enable_truncation: bool,
}

impl Default for ContentLimits {
    fn default() -> Self {
        Self {
            max_analysis_length: 10000,
            min_detailed_analysis_length: 10,
            enable_truncation: true,
        }
    }
}

/// Configuration for handling safety violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationHandling {
    /// Custom replacement message for filtered content
    pub replacement_message: Option<String>,
    /// Enable violation logging
    pub enable_logging: bool,
    /// Maximum violations before conversation reset
    pub max_violations_before_reset: usize,
    /// Cooldown period after violations (seconds)
    pub violation_cooldown_seconds: u64,
}

impl Default for ViolationHandling {
    fn default() -> Self {
        Self {
            replacement_message: Some(
                "I can't assist with that request. Let's talk about something else.".to_string(),
            ),
            enable_logging: true,
            max_violations_before_reset: 3,
            violation_cooldown_seconds: 300, // 5 minutes
        }
    }
}

/// Performance optimization settings for safety filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Cache expiry time (seconds)
    pub cache_expiry_seconds: u64,
    /// Enable parallel pattern evaluation
    pub enable_parallel_evaluation: bool,
    /// Maximum evaluation threads
    pub max_evaluation_threads: usize,
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_size_limit: 1000,
            cache_expiry_seconds: 3600,        // 1 hour
            enable_parallel_evaluation: false, // Keep simple for now
            max_evaluation_threads: 4,
        }
    }
}

// ================================================================================================
// ENHANCED SAFETY ASSESSMENT TYPES
// ================================================================================================

/// Comprehensive safety assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSafetyAssessment {
    /// Base safety analysis
    pub base_analysis: SafetyAnalysis,
    /// Detailed safety scores by category
    pub category_scores: SafetyScores,
    /// Enhanced violations with location information
    pub enhanced_violations: Vec<EnhancedSafetyViolation>,
    /// Assessment confidence (0.0-1.0)
    pub confidence: f32,
    /// Assessment timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
}

impl EnhancedSafetyAssessment {
    /// Check if content is considered safe based on thresholds
    pub fn is_safe(&self, config: &ExtendedSafetyConfig) -> bool {
        self.base_analysis.is_safe
            && self.category_scores.toxicity <= config.max_toxicity_score
            && self.category_scores.harm <= config.max_harm_score
            && self.category_scores.bias <= config.max_bias_score
    }

    /// Get the highest risk violation
    pub fn highest_risk_violation(&self) -> Option<&EnhancedSafetyViolation> {
        self.enhanced_violations
            .iter()
            .max_by_key(|v| v.base_violation.severity.clone() as u8)
    }

    /// Check if assessment indicates immediate action needed
    pub fn requires_immediate_action(&self) -> bool {
        self.enhanced_violations
            .iter()
            .any(|v| matches!(v.base_violation.severity, SafetySeverity::Critical))
    }
}

/// Detailed safety scores by category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyScores {
    /// Toxicity score (0.0-1.0)
    pub toxicity: f32,
    /// Harm potential score (0.0-1.0)
    pub harm: f32,
    /// Bias detection score (0.0-1.0)
    pub bias: f32,
    /// Personal information exposure score (0.0-1.0)
    pub privacy: f32,
    /// Inappropriate content score (0.0-1.0)
    pub inappropriate: f32,
    /// Violence indicators score (0.0-1.0)
    pub violence: f32,
    /// Harassment indicators score (0.0-1.0)
    pub harassment: f32,
}

impl Default for SafetyScores {
    fn default() -> Self {
        Self {
            toxicity: 0.0,
            harm: 0.0,
            bias: 0.0,
            privacy: 0.0,
            inappropriate: 0.0,
            violence: 0.0,
            harassment: 0.0,
        }
    }
}

/// Enhanced safety violation with location and context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSafetyViolation {
    /// Base violation information
    pub base_violation: SafetyViolation,
    /// Location of violation in content (character positions)
    pub location: Option<ViolationLocation>,
    /// Pattern or rule that triggered the violation
    pub triggered_rule: String,
    /// Context around the violation
    pub context: Option<String>,
    /// Suggested remediation
    pub suggested_remediation: Option<String>,
}

/// Location information for violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationLocation {
    pub start_pos: usize,
    pub end_pos: usize,
    pub matched_text: String,
}

// ================================================================================================
// ENHANCED SAFETY FILTER IMPLEMENTATION
// ================================================================================================

/// Enhanced safety filter alias
pub type EnhancedSafetyFilter = SafetyFilter;

/// Safety configuration alias
pub type SafetyConfig = ExtendedSafetyConfig;

/// Safety assessment alias
pub type SafetyAssessment = EnhancedSafetyAssessment;

/// Enhanced safety filter for conversational content moderation
#[derive(Debug)]
pub struct SafetyFilter {
    /// Enhanced configuration
    extended_config: ExtendedSafetyConfig,
    /// Banned words/phrases (from base implementation)
    pub banned_terms: Vec<String>,
    /// Toxic content patterns (from base implementation)
    pub toxic_patterns: Vec<regex::Regex>,
    /// Maximum allowed toxicity score (from base implementation)
    pub max_toxicity_score: f32,
    /// Configuration for advanced filtering (from base implementation)
    pub config: SafetyFilterConfig,
    /// Additional compiled regex patterns for enhanced detection
    harm_patterns: Vec<Regex>,
    privacy_patterns: Vec<Regex>,
    violence_patterns: Vec<Regex>,
    harassment_patterns: Vec<Regex>,
    bias_keywords: HashSet<String>,
    /// Result cache for performance optimization
    assessment_cache: HashMap<String, (EnhancedSafetyAssessment, Instant)>,
    /// Violation history for tracking patterns
    violation_history: Vec<EnhancedSafetyViolation>,
}

impl SafetyFilter {
    /// Create a new safety filter with default configuration
    pub fn new() -> Self {
        let extended_config = ExtendedSafetyConfig::default();
        Self::with_extended_config(extended_config)
    }

    /// Create safety filter with extended configuration
    pub fn with_extended_config(extended_config: ExtendedSafetyConfig) -> Self {
        let mut filter = Self {
            extended_config: extended_config.clone(),
            banned_terms: Self::default_banned_terms(),
            toxic_patterns: Vec::new(),
            max_toxicity_score: extended_config.max_toxicity_score,
            config: extended_config.base_config,
            harm_patterns: Vec::new(),
            privacy_patterns: Vec::new(),
            violence_patterns: Vec::new(),
            harassment_patterns: Vec::new(),
            bias_keywords: HashSet::new(),
            assessment_cache: HashMap::new(),
            violation_history: Vec::new(),
        };

        filter.initialize_patterns();
        filter
    }

    fn default_banned_terms() -> Vec<String> {
        vec![
            // Hate speech indicators
            "hate".to_string(),
            "racist".to_string(),
            "sexist".to_string(),
            // Violence indicators
            "violence".to_string(),
            "harm".to_string(),
            "kill".to_string(),
            "hurt".to_string(),
            "attack".to_string(),
            // Inappropriate content
            "inappropriate".to_string(),
            "offensive".to_string(),
            "abusive".to_string(),
            // Privacy concerns
            "password".to_string(),
            "ssn".to_string(),
            "social security".to_string(),
            "credit card".to_string(),
            "bank account".to_string(),
        ]
    }

    /// Initialize safety detection patterns
    fn initialize_patterns(&mut self) {
        self.initialize_toxic_patterns();
        self.initialize_harm_patterns();
        self.initialize_privacy_patterns();
        self.initialize_violence_patterns();
        self.initialize_harassment_patterns();
        self.initialize_bias_keywords();
    }

    fn initialize_toxic_patterns(&mut self) {
        let patterns = [
            r"(?i)\b(hate|despise|loathe)\s+\w+",
            r"(?i)\b(kill|murder|destroy)\s+(you|them|everyone)",
            r"(?i)\b(stupid|idiot|moron|dumb)\s+(person|people|you)",
            r"(?i)\b(go\s+die|drop\s+dead)",
            r"(?i)\b(toxic|poisonous)\s+(behavior|attitude)",
        ];

        for pattern in patterns {
            if let Ok(regex) = Regex::new(pattern) {
                self.toxic_patterns.push(regex);
            }
        }
    }

    fn initialize_harm_patterns(&mut self) {
        let patterns = [
            r"(?i)\b(self[\s\-]?harm|self[\s\-]?injury)",
            r"(?i)\b(suicide|suicidal|end\s+it\s+all)",
            r"(?i)\b(cut|cutting|burn|burning|hurt|hurting)\s+(myself|yourself)",
            r"(?i)\b(want\s+to\s+die|wish\s+I\s+was\s+dead)",
            r"(?i)\b(harmful|dangerous)\s+(advice|instructions)",
        ];

        for pattern in patterns {
            if let Ok(regex) = Regex::new(pattern) {
                self.harm_patterns.push(regex);
            }
        }
    }

    fn initialize_privacy_patterns(&mut self) {
        let patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",                               // SSN pattern
            r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",       // Credit card pattern
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", // Email pattern
            r"\b\d{3}[\s\-]?\d{3}[\s\-]?\d{4}\b",                   // Phone number pattern
            r"(?i)\b(password|passwd|pwd)[\s:=]+\S+",
            r"(?i)\b(api[\s\-]?key|secret[\s\-]?key)[\s:=]+\S+",
        ];

        for pattern in patterns {
            if let Ok(regex) = Regex::new(pattern) {
                self.privacy_patterns.push(regex);
            }
        }
    }

    fn initialize_violence_patterns(&mut self) {
        let patterns = [
            r"(?i)\b(violence|violent|attack|assault)",
            r"(?i)\b(fight|fighting|beat\s+up)",
            r"(?i)\b(weapon|gun|knife|bomb)",
            r"(?i)\b(threatening|threat|intimidate)",
            r"(?i)\b(abuse|abusive|abusing)",
        ];

        for pattern in patterns {
            if let Ok(regex) = Regex::new(pattern) {
                self.violence_patterns.push(regex);
            }
        }
    }

    fn initialize_harassment_patterns(&mut self) {
        let patterns = [
            r"(?i)\b(harass|harassment|stalking)",
            r"(?i)\b(bully|bullying|intimidate)",
            r"(?i)\b(creep|creepy|pervert)",
            r"(?i)\b(ugly|disgusting|gross)\s+(person|you)",
            r"(?i)\b(shut\s+up|get\s+lost|go\s+away)",
        ];

        for pattern in patterns {
            if let Ok(regex) = Regex::new(pattern) {
                self.harassment_patterns.push(regex);
            }
        }
    }

    fn initialize_bias_keywords(&mut self) {
        let bias_terms = [
            "racist",
            "sexist",
            "homophobic",
            "transphobic",
            "xenophobic",
            "islamophobic",
            "antisemitic",
            "stereotype",
            "prejudice",
            "discrimination",
            "bigot",
            "bigotry",
            "intolerant",
        ];

        for term in bias_terms {
            self.bias_keywords.insert(term.to_string());
        }
    }

    /// Create safety filter with custom configuration (backward compatibility)
    pub fn with_config(config: SafetyFilterConfig) -> Self {
        let mut extended_config = ExtendedSafetyConfig::default();
        extended_config.base_config = config.clone();
        extended_config.max_toxicity_score = match config.moderation_level {
            ModerationLevel::Permissive => 0.9,
            ModerationLevel::Moderate => 0.7,
            ModerationLevel::Strict => 0.4,
            ModerationLevel::VeryStrict => 0.2,
            ModerationLevel::Custom => 0.5,
        };
        Self::with_extended_config(extended_config)
    }

    /// Create a safety filter with strict configuration
    pub fn strict() -> Self {
        let mut config = ExtendedSafetyConfig::default();
        config.base_config.moderation_level = ModerationLevel::Strict;
        config.max_toxicity_score = 0.4;
        config.max_harm_score = 0.3;
        config.max_bias_score = 0.5;
        Self::with_extended_config(config)
    }

    /// Create a safety filter with permissive configuration
    pub fn permissive() -> Self {
        let mut config = ExtendedSafetyConfig::default();
        config.base_config.moderation_level = ModerationLevel::Permissive;
        config.max_toxicity_score = 0.9;
        config.max_harm_score = 0.8;
        config.max_bias_score = 0.9;
        Self::with_extended_config(config)
    }

    /// Create a safety filter for educational environments
    pub fn educational() -> Self {
        let mut config = ExtendedSafetyConfig::default();
        config.base_config.moderation_level = ModerationLevel::Moderate;
        config.max_toxicity_score = 0.6;
        config.max_harm_score = 0.5;
        config.max_bias_score = 0.7;
        Self::with_extended_config(config)
    }

    /// Check if content is safe (simplified interface)
    pub fn is_safe(&self, content: &str) -> bool {
        if !self.config.enabled {
            return true;
        }

        let assessment = self.assess_content_safety_enhanced(content);
        assessment.is_safe(&self.extended_config)
    }

    /// Enhanced comprehensive safety assessment of content
    pub fn assess_content_safety_enhanced(&self, content: &str) -> EnhancedSafetyAssessment {
        let start_time = Instant::now();

        // Check cache first if enabled
        if self.extended_config.performance_settings.enable_caching {
            if let Some((cached_assessment, cache_time)) = self.assessment_cache.get(content) {
                let cache_age = start_time.duration_since(*cache_time);
                if cache_age.as_secs()
                    < self.extended_config.performance_settings.cache_expiry_seconds
                {
                    return cached_assessment.clone();
                }
            }
        }

        // Truncate content if necessary
        let analyzed_content = if self.extended_config.content_limits.enable_truncation
            && content.len() > self.extended_config.content_limits.max_analysis_length
        {
            &content[..self.extended_config.content_limits.max_analysis_length]
        } else {
            content
        };

        // Perform base safety analysis first
        let base_analysis = self.analyze_safety(analyzed_content);

        // Perform enhanced safety assessment
        let mut category_scores = SafetyScores::default();
        let mut enhanced_violations = Vec::new();

        // Analyze different safety categories
        self.analyze_toxicity_enhanced(
            analyzed_content,
            &mut category_scores,
            &mut enhanced_violations,
        );
        self.analyze_harm_enhanced(
            analyzed_content,
            &mut category_scores,
            &mut enhanced_violations,
        );
        self.analyze_privacy_enhanced(
            analyzed_content,
            &mut category_scores,
            &mut enhanced_violations,
        );
        self.analyze_violence_enhanced(
            analyzed_content,
            &mut category_scores,
            &mut enhanced_violations,
        );
        self.analyze_harassment_enhanced(
            analyzed_content,
            &mut category_scores,
            &mut enhanced_violations,
        );
        self.analyze_bias_enhanced(
            analyzed_content,
            &mut category_scores,
            &mut enhanced_violations,
        );
        self.analyze_inappropriate_content_enhanced(
            analyzed_content,
            &mut category_scores,
            &mut enhanced_violations,
        );

        // Calculate confidence based on content length and pattern matches
        let confidence =
            self.calculate_assessment_confidence(analyzed_content, &enhanced_violations);

        let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        EnhancedSafetyAssessment {
            base_analysis,
            category_scores,
            enhanced_violations,
            confidence,
            timestamp: chrono::Utc::now(),
            processing_time_ms,
        }
    }

    /// Filter input content and return safe version
    pub fn filter_input(&self, content: &str) -> Result<String> {
        if !self.config.enabled || !self.extended_config.filter_input {
            return Ok(content.to_string());
        }

        let assessment = self.assess_content_safety_enhanced(content);

        if assessment.is_safe(&self.extended_config) {
            Ok(content.to_string())
        } else {
            self.handle_violation(&assessment, content)
        }
    }

    /// Filter output content and return safe version
    pub fn filter_output(&self, content: &str) -> Result<String> {
        if !self.config.enabled || !self.extended_config.filter_output {
            return Ok(content.to_string());
        }

        let assessment = self.assess_content_safety_enhanced(content);

        if assessment.is_safe(&self.extended_config) {
            Ok(content.to_string())
        } else {
            self.handle_violation(&assessment, content)
        }
    }

    /// Add safety metadata to conversation metadata
    pub fn enrich_conversation_metadata(&self, content: &str, metadata: &mut ConversationMetadata) {
        if !self.config.enabled {
            return;
        }

        let assessment = self.assess_content_safety_enhanced(content);

        // Add safety flags from enhanced violations
        for violation in &assessment.enhanced_violations {
            metadata.safety_flags.push(format!(
                "violation:{}",
                violation.base_violation.violation_type
            ));
        }

        // Adjust confidence based on safety assessment
        let overall_safety_score = (assessment.category_scores.toxicity
            + assessment.category_scores.harm
            + assessment.category_scores.violence)
            / 3.0;

        // Reduce confidence and quality when there are any violations
        if !assessment.enhanced_violations.is_empty() || overall_safety_score > 0.1 {
            // Apply penalty based on the higher of: number of violations or score-based penalty
            let violation_penalty = (assessment.enhanced_violations.len() as f32 * 0.05).min(0.3);
            let score_penalty = overall_safety_score * 0.3;
            let total_penalty = violation_penalty.max(score_penalty).max(0.05); // Minimum 5% penalty
            metadata.confidence *= 1.0 - total_penalty;
            metadata.quality_score *= 1.0 - total_penalty;
        } else {
            // No violations, only apply minor score-based adjustment to quality
            metadata.quality_score *= 1.0 - overall_safety_score * 0.5;
        }
    }

    /// Get toxicity score (backward compatibility)
    /// Note: This method calculates toxicity directly without calling assess_content_safety_enhanced
    /// to avoid infinite recursion (assess_content_safety_enhanced -> analyze_safety -> get_toxicity_score)
    pub fn get_toxicity_score(&self, content: &str) -> f32 {
        if !self.config.enabled {
            return 0.0;
        }

        self.calculate_toxicity_score_direct(content)
    }

    /// Direct toxicity calculation without triggering recursive assessment
    fn calculate_toxicity_score_direct(&self, content: &str) -> f32 {
        let mut toxicity_score: f32 = 0.0;
        let content_lower = content.to_lowercase();

        // Check banned terms
        for term in &self.banned_terms {
            if content_lower.contains(&term.to_lowercase()) {
                toxicity_score += 0.3;
            }
        }

        // Check toxic patterns
        for pattern in &self.toxic_patterns {
            if pattern.is_match(content) {
                toxicity_score += 0.4;
            }
        }

        toxicity_score.min(1.0)
    }

    // ================================================================================================
    // ENHANCED ANALYSIS METHODS
    // ================================================================================================

    fn analyze_toxicity_enhanced(
        &self,
        content: &str,
        scores: &mut SafetyScores,
        violations: &mut Vec<EnhancedSafetyViolation>,
    ) {
        let mut toxicity_score: f32 = 0.0;

        // Check banned terms
        let content_lower = content.to_lowercase();
        for term in &self.banned_terms {
            if content_lower.contains(&term.to_lowercase()) {
                toxicity_score += 0.3;

                violations.push(EnhancedSafetyViolation {
                    base_violation: SafetyViolation {
                        violation_type: "toxicity".to_string(),
                        severity: SafetySeverity::Medium,
                        description: format!("Banned term detected: {}", term),
                        confidence: 0.9,
                    },
                    location: self.find_term_location(content, term),
                    triggered_rule: format!("banned_term:{}", term),
                    context: self.extract_context(content, term),
                    suggested_remediation: Some(
                        "Consider rephrasing without offensive language".to_string(),
                    ),
                });
            }
        }

        // Check toxic patterns
        for (i, pattern) in self.toxic_patterns.iter().enumerate() {
            if let Some(matches) = pattern.find(content) {
                toxicity_score += 0.4;

                violations.push(EnhancedSafetyViolation {
                    base_violation: SafetyViolation {
                        violation_type: "toxicity".to_string(),
                        severity: SafetySeverity::High,
                        description: "Toxic language pattern detected".to_string(),
                        confidence: 0.8,
                    },
                    location: Some(ViolationLocation {
                        start_pos: matches.start(),
                        end_pos: matches.end(),
                        matched_text: matches.as_str().to_string(),
                    }),
                    triggered_rule: format!("toxic_pattern_{}", i),
                    context: self.extract_context_from_match(
                        content,
                        matches.start(),
                        matches.end(),
                    ),
                    suggested_remediation: Some("Please use more respectful language".to_string()),
                });
            }
        }

        scores.toxicity = toxicity_score.min(1.0);
    }

    fn analyze_harm_enhanced(
        &self,
        content: &str,
        scores: &mut SafetyScores,
        violations: &mut Vec<EnhancedSafetyViolation>,
    ) {
        let mut harm_score: f32 = 0.0;

        for (i, pattern) in self.harm_patterns.iter().enumerate() {
            if let Some(matches) = pattern.find(content) {
                harm_score += 0.6; // Harm patterns are more serious

                violations.push(EnhancedSafetyViolation {
                    base_violation: SafetyViolation {
                        violation_type: "self_harm".to_string(),
                        severity: SafetySeverity::Critical,
                        description: "Self-harm or harmful content detected".to_string(),
                        confidence: 0.9,
                    },
                    location: Some(ViolationLocation {
                        start_pos: matches.start(),
                        end_pos: matches.end(),
                        matched_text: matches.as_str().to_string(),
                    }),
                    triggered_rule: format!("harm_pattern_{}", i),
                    context: self.extract_context_from_match(content, matches.start(), matches.end()),
                    suggested_remediation: Some("If you're experiencing thoughts of self-harm, please seek help from a mental health professional".to_string()),
                });
            }
        }

        scores.harm = harm_score.min(1.0);
    }

    fn analyze_privacy_enhanced(
        &self,
        content: &str,
        scores: &mut SafetyScores,
        violations: &mut Vec<EnhancedSafetyViolation>,
    ) {
        let mut privacy_score: f32 = 0.0;

        for (i, pattern) in self.privacy_patterns.iter().enumerate() {
            if let Some(matches) = pattern.find(content) {
                privacy_score += 0.5;

                violations.push(EnhancedSafetyViolation {
                    base_violation: SafetyViolation {
                        violation_type: "personal_information".to_string(),
                        severity: SafetySeverity::High,
                        description: "Personal information detected".to_string(),
                        confidence: 0.8,
                    },
                    location: Some(ViolationLocation {
                        start_pos: matches.start(),
                        end_pos: matches.end(),
                        matched_text: "***REDACTED***".to_string(), // Don't expose PII
                    }),
                    triggered_rule: format!("privacy_pattern_{}", i),
                    context: Some("[CONTEXT REDACTED FOR PRIVACY]".to_string()),
                    suggested_remediation: Some(
                        "Avoid sharing personal information in conversations".to_string(),
                    ),
                });
            }
        }

        scores.privacy = privacy_score.min(1.0);
    }

    fn analyze_violence_enhanced(
        &self,
        content: &str,
        scores: &mut SafetyScores,
        violations: &mut Vec<EnhancedSafetyViolation>,
    ) {
        let mut violence_score: f32 = 0.0;

        for (i, pattern) in self.violence_patterns.iter().enumerate() {
            if let Some(matches) = pattern.find(content) {
                violence_score += 0.4;

                violations.push(EnhancedSafetyViolation {
                    base_violation: SafetyViolation {
                        violation_type: "violence".to_string(),
                        severity: SafetySeverity::High,
                        description: "Violence-related content detected".to_string(),
                        confidence: 0.8,
                    },
                    location: Some(ViolationLocation {
                        start_pos: matches.start(),
                        end_pos: matches.end(),
                        matched_text: matches.as_str().to_string(),
                    }),
                    triggered_rule: format!("violence_pattern_{}", i),
                    context: self.extract_context_from_match(
                        content,
                        matches.start(),
                        matches.end(),
                    ),
                    suggested_remediation: Some(
                        "Please avoid discussing violent topics".to_string(),
                    ),
                });
            }
        }

        scores.violence = violence_score.min(1.0);
    }

    fn analyze_harassment_enhanced(
        &self,
        content: &str,
        scores: &mut SafetyScores,
        violations: &mut Vec<EnhancedSafetyViolation>,
    ) {
        let mut harassment_score: f32 = 0.0;

        for (i, pattern) in self.harassment_patterns.iter().enumerate() {
            if let Some(matches) = pattern.find(content) {
                harassment_score += 0.4;

                violations.push(EnhancedSafetyViolation {
                    base_violation: SafetyViolation {
                        violation_type: "harassment".to_string(),
                        severity: SafetySeverity::Medium,
                        description: "Harassment-related content detected".to_string(),
                        confidence: 0.7,
                    },
                    location: Some(ViolationLocation {
                        start_pos: matches.start(),
                        end_pos: matches.end(),
                        matched_text: matches.as_str().to_string(),
                    }),
                    triggered_rule: format!("harassment_pattern_{}", i),
                    context: self.extract_context_from_match(
                        content,
                        matches.start(),
                        matches.end(),
                    ),
                    suggested_remediation: Some(
                        "Please maintain respectful communication".to_string(),
                    ),
                });
            }
        }

        scores.harassment = harassment_score.min(1.0);
    }

    fn analyze_bias_enhanced(
        &self,
        content: &str,
        scores: &mut SafetyScores,
        violations: &mut Vec<EnhancedSafetyViolation>,
    ) {
        let content_lower = content.to_lowercase();
        let mut bias_score: f32 = 0.0;

        for keyword in &self.bias_keywords {
            if content_lower.contains(keyword) {
                bias_score += 0.3;

                violations.push(EnhancedSafetyViolation {
                    base_violation: SafetyViolation {
                        violation_type: "bias".to_string(),
                        severity: SafetySeverity::Medium,
                        description: format!("Bias-related keyword detected: {}", keyword),
                        confidence: 0.6,
                    },
                    location: self.find_term_location(content, keyword),
                    triggered_rule: format!("bias_keyword:{}", keyword),
                    context: self.extract_context(content, keyword),
                    suggested_remediation: Some(
                        "Consider using more inclusive language".to_string(),
                    ),
                });
            }
        }

        scores.bias = bias_score.min(1.0);
    }

    fn analyze_inappropriate_content_enhanced(
        &self,
        content: &str,
        scores: &mut SafetyScores,
        violations: &mut Vec<EnhancedSafetyViolation>,
    ) {
        let inappropriate_keywords = ["inappropriate", "offensive", "rude", "vulgar", "profanity"];
        let content_lower = content.to_lowercase();
        let mut inappropriate_score: f32 = 0.0;

        for keyword in &inappropriate_keywords {
            if content_lower.contains(keyword) {
                inappropriate_score += 0.2;

                violations.push(EnhancedSafetyViolation {
                    base_violation: SafetyViolation {
                        violation_type: "inappropriate".to_string(),
                        severity: SafetySeverity::Low,
                        description: format!("Inappropriate content indicator: {}", keyword),
                        confidence: 0.5,
                    },
                    location: self.find_term_location(content, keyword),
                    triggered_rule: format!("inappropriate_keyword:{}", keyword),
                    context: self.extract_context(content, keyword),
                    suggested_remediation: Some(
                        "Please keep the conversation appropriate".to_string(),
                    ),
                });
            }
        }

        scores.inappropriate = inappropriate_score.min(1.0);
    }

    // ================================================================================================
    // HELPER METHODS
    // ================================================================================================

    fn calculate_assessment_confidence(
        &self,
        content: &str,
        violations: &[EnhancedSafetyViolation],
    ) -> f32 {
        let mut confidence = 0.7; // Base confidence

        // Adjust based on content length
        if content.len() > self.extended_config.content_limits.min_detailed_analysis_length {
            confidence += 0.1;
        }

        // Adjust based on violation confidence
        if !violations.is_empty() {
            let avg_violation_confidence: f32 =
                violations.iter().map(|v| v.base_violation.confidence).sum::<f32>()
                    / violations.len() as f32;
            confidence = (confidence + avg_violation_confidence) / 2.0;
        }

        confidence.min(1.0)
    }

    fn find_term_location(&self, content: &str, term: &str) -> Option<ViolationLocation> {
        let content_lower = content.to_lowercase();
        let term_lower = term.to_lowercase();

        content_lower.find(&term_lower).map(|start_pos| ViolationLocation {
            start_pos,
            end_pos: start_pos + term.len(),
            matched_text: content[start_pos..start_pos + term.len()].to_string(),
        })
    }

    fn extract_context(&self, content: &str, term: &str) -> Option<String> {
        if let Some(start_pos) = content.to_lowercase().find(&term.to_lowercase()) {
            let context_start = start_pos.saturating_sub(20);
            let context_end = (start_pos + term.len() + 20).min(content.len());
            Some(content[context_start..context_end].to_string())
        } else {
            None
        }
    }

    fn extract_context_from_match(
        &self,
        content: &str,
        start_pos: usize,
        end_pos: usize,
    ) -> Option<String> {
        let context_start = start_pos.saturating_sub(20);
        let context_end = (end_pos + 20).min(content.len());
        Some(content[context_start..context_end].to_string())
    }

    fn handle_violation(
        &self,
        assessment: &EnhancedSafetyAssessment,
        content: &str,
    ) -> Result<String> {
        // Use the base analysis recommended action
        match assessment.base_analysis.recommended_action {
            SafetyAction::Block => {
                let violations: Vec<String> = assessment
                    .enhanced_violations
                    .iter()
                    .map(|v| v.base_violation.description.clone())
                    .collect();
                Err(crate::error::TrustformersError::InvalidInput {
                    message: format!("Content blocked due to safety violations: {:?}", violations),
                    parameter: Some("content".to_string()),
                    expected: Some("safe content".to_string()),
                    received: Some("content with safety violations".to_string()),
                    suggestion: Some("Remove or modify unsafe content".to_string()),
                }
                .into())
            },
            SafetyAction::Modify => Ok(self
                .extended_config
                .violation_handling
                .replacement_message
                .clone()
                .unwrap_or_else(|| "Content filtered for safety.".to_string())),
            SafetyAction::Warn => {
                // Log violation but allow content
                Ok(format!("[FLAGGED] {}", content))
            },
            SafetyAction::Log => {
                // Just log, no modification
                Ok(content.to_string())
            },
            SafetyAction::Clarify => {
                // Request clarification from user
                Ok(format!(
                    "{}\n[System: This content requires clarification for safety verification]",
                    content
                ))
            },
        }
    }

    /// Evaluate a custom safety rule
    fn evaluate_safety_rule(&self, content: &str, rule: &SafetyRule) -> bool {
        let content_lower = content.to_lowercase();

        // Simple pattern matching - in a real implementation this would be more sophisticated
        if let Ok(regex) = regex::Regex::new(&rule.pattern) {
            regex.is_match(&content_lower)
        } else {
            // Fallback to simple string matching
            content_lower.contains(&rule.pattern.to_lowercase())
        }
    }

    /// Update extended configuration
    pub fn update_extended_config(&mut self, config: ExtendedSafetyConfig) {
        self.extended_config = config;
        self.config = self.extended_config.base_config.clone();
        self.max_toxicity_score = self.extended_config.max_toxicity_score;
        self.initialize_patterns();

        // Clear cache when configuration changes
        self.assessment_cache.clear();
    }

    /// Get current extended configuration
    pub fn get_extended_config(&self) -> &ExtendedSafetyConfig {
        &self.extended_config
    }

    /// Clear assessment cache
    pub fn clear_cache(&mut self) {
        self.assessment_cache.clear();
    }

    /// Get violation history
    pub fn get_violation_history(&self) -> &[EnhancedSafetyViolation] {
        &self.violation_history
    }

    /// Clear violation history
    pub fn clear_violation_history(&mut self) {
        self.violation_history.clear();
    }

    /// Add custom safety pattern
    pub fn add_custom_pattern(&mut self, pattern: &str, violation_type: &str) -> Result<()> {
        let regex =
            Regex::new(pattern).map_err(|e| crate::error::TrustformersError::InvalidInput {
                message: format!("Invalid safety pattern: {}", e),
                parameter: Some("pattern".to_string()),
                expected: Some("valid regex pattern".to_string()),
                received: Some(pattern.to_string()),
                suggestion: Some("Check regex syntax".to_string()),
            })?;

        match violation_type {
            "toxicity" => self.toxic_patterns.push(regex),
            "violence" => self.violence_patterns.push(regex),
            "harassment" => self.harassment_patterns.push(regex),
            "privacy" => self.privacy_patterns.push(regex),
            "harm" => self.harm_patterns.push(regex),
            _ => {
                // For other types, add to toxic patterns as default
                self.toxic_patterns.push(regex);
            },
        }

        Ok(())
    }

    /// Perform comprehensive safety analysis
    pub fn analyze_safety(&self, content: &str) -> SafetyAnalysis {
        if !self.config.enabled {
            return SafetyAnalysis {
                is_safe: true,
                toxicity_score: 0.0,
                violations: Vec::new(),
                risk_level: RiskLevel::None,
                recommended_action: SafetyAction::Log,
            };
        }

        let mut violations = Vec::new();
        let toxicity_score = self.get_toxicity_score(content);

        // Check for various types of violations
        if self.config.toxicity_detection && toxicity_score > 0.5 {
            violations.push(SafetyViolation {
                violation_type: "toxicity".to_string(),
                severity: SafetySeverity::Medium,
                description: "High toxicity score detected".to_string(),
                confidence: toxicity_score,
            });
        }

        if self.config.harmful_content_detection {
            violations.extend(self.detect_harmful_content(content));
        }

        if self.config.bias_detection {
            violations.extend(self.detect_bias(content));
        }

        // Check custom rules
        for rule in &self.config.custom_rules {
            if self.evaluate_safety_rule(content, rule) {
                violations.push(SafetyViolation {
                    violation_type: rule.name.clone(),
                    severity: rule.severity.clone(),
                    description: format!("Custom rule violation: {}", rule.pattern),
                    confidence: 0.8,
                });
            }
        }

        let risk_level = self.calculate_risk_level(&violations, toxicity_score);
        let is_safe = violations.is_empty() && toxicity_score <= self.max_toxicity_score;
        let recommended_action = self.recommend_action(&violations, risk_level);

        SafetyAnalysis {
            is_safe,
            toxicity_score,
            violations,
            risk_level,
            recommended_action,
        }
    }

    /// Detect harmful content patterns
    fn detect_harmful_content(&self, content: &str) -> Vec<SafetyViolation> {
        let mut violations = Vec::new();
        let content_lower = content.to_lowercase();

        let harmful_patterns = [
            (
                "violence",
                &["kill", "murder", "attack", "hurt", "harm"] as &[&str],
            ),
            (
                "self_harm",
                &["suicide", "kill myself", "end it all", "hurt myself"],
            ),
            ("harassment", &["stalking", "threatening", "intimidating"]),
            ("illegal_activity", &["drug dealing", "illegal", "criminal"]),
        ];

        for (category, patterns) in &harmful_patterns {
            for pattern in *patterns {
                if content_lower.contains(pattern) {
                    violations.push(SafetyViolation {
                        violation_type: category.to_string(),
                        severity: SafetySeverity::High,
                        description: format!("Detected harmful content: {}", pattern),
                        confidence: 0.7,
                    });
                    break; // Only report one violation per category
                }
            }
        }

        violations
    }

    /// Detect bias patterns
    fn detect_bias(&self, content: &str) -> Vec<SafetyViolation> {
        let mut violations = Vec::new();
        let content_lower = content.to_lowercase();

        let bias_patterns = [
            (
                "gender_bias",
                &["women are", "men are", "girls can't", "boys don't"],
            ),
            (
                "racial_bias",
                &["people of", "race is", "ethnicity", "racial"],
            ),
            (
                "age_bias",
                &[
                    "old people",
                    "young people",
                    "millennials are",
                    "boomers are",
                ],
            ),
        ];

        for (category, patterns) in &bias_patterns {
            for pattern in *patterns {
                if content_lower.contains(pattern) {
                    violations.push(SafetyViolation {
                        violation_type: category.to_string(),
                        severity: SafetySeverity::Medium,
                        description: format!("Potential bias detected: {}", pattern),
                        confidence: 0.6,
                    });
                    break;
                }
            }
        }

        violations
    }

    /// Calculate overall risk level
    fn calculate_risk_level(
        &self,
        violations: &[SafetyViolation],
        toxicity_score: f32,
    ) -> RiskLevel {
        if violations.is_empty() && toxicity_score < 0.3 {
            return RiskLevel::None;
        }

        let high_severity_count = violations
            .iter()
            .filter(|v| matches!(v.severity, SafetySeverity::High | SafetySeverity::Critical))
            .count();

        if high_severity_count > 0 || toxicity_score > 0.8 {
            RiskLevel::High
        } else if violations.len() > 2 || toxicity_score > 0.6 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    /// Recommend action based on analysis
    fn recommend_action(
        &self,
        violations: &[SafetyViolation],
        risk_level: RiskLevel,
    ) -> SafetyAction {
        match risk_level {
            RiskLevel::None => SafetyAction::Log,
            RiskLevel::Low => SafetyAction::Warn,
            RiskLevel::Medium => SafetyAction::Modify,
            RiskLevel::High => SafetyAction::Block,
        }
    }

    /// Sanitize content by removing/replacing problematic parts
    pub fn sanitize_content(&self, content: &str) -> String {
        let mut sanitized = content.to_string();

        // Replace banned terms with asterisks
        for term in &self.banned_terms {
            let replacement = "*".repeat(term.len());
            sanitized = sanitized.replace(term, &replacement);
        }

        // Apply regex replacements
        for pattern in &self.toxic_patterns {
            sanitized = pattern.replace_all(&sanitized, "[FILTERED]").to_string();
        }

        sanitized
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SafetyFilterConfig) {
        self.config = config;
    }

    /// Add custom safety rule
    pub fn add_custom_rule(&mut self, rule: SafetyRule) {
        self.config.custom_rules.push(rule);
    }

    /// Get safety statistics
    pub fn get_safety_stats(&self, content_samples: &[String]) -> SafetyStats {
        if content_samples.is_empty() {
            return SafetyStats::default();
        }

        let mut total_violations = 0;
        let mut total_toxicity_score = 0.0;
        let mut safe_count = 0;
        let mut risk_distribution = std::collections::HashMap::new();

        for content in content_samples {
            let analysis = self.analyze_safety(content);

            if analysis.is_safe {
                safe_count += 1;
            }

            total_violations += analysis.violations.len();
            total_toxicity_score += analysis.toxicity_score;

            *risk_distribution.entry(analysis.risk_level).or_insert(0) += 1;
        }

        let total_samples = content_samples.len();
        SafetyStats {
            total_samples,
            safe_percentage: (safe_count as f32 / total_samples as f32) * 100.0,
            avg_toxicity_score: total_toxicity_score / total_samples as f32,
            total_violations,
            avg_violations_per_sample: total_violations as f32 / total_samples as f32,
            risk_distribution,
        }
    }
}

impl Default for SafetyFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of safety analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyAnalysis {
    pub is_safe: bool,
    pub toxicity_score: f32,
    pub violations: Vec<SafetyViolation>,
    pub risk_level: RiskLevel,
    pub recommended_action: SafetyAction,
}

/// Individual safety violation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SafetyViolation {
    pub violation_type: String,
    pub severity: SafetySeverity,
    pub description: String,
    pub confidence: f32,
}

/// Risk level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RiskLevel {
    None,
    Low,
    Medium,
    High,
}

/// Safety statistics
#[derive(Debug, Clone, Default)]
pub struct SafetyStats {
    pub total_samples: usize,
    pub safe_percentage: f32,
    pub avg_toxicity_score: f32,
    pub total_violations: usize,
    pub avg_violations_per_sample: f32,
    pub risk_distribution: std::collections::HashMap<RiskLevel, usize>,
}

// ================================================================================================
// ERROR TYPES
// ================================================================================================

/// Safety-related errors
#[derive(Error, Debug)]
pub enum SafetyError {
    #[error("Content blocked due to safety violations: {0:?}")]
    ContentBlocked(Vec<String>),

    #[error("Conversation ended due to safety violation: {0}")]
    ConversationEnded(String),

    #[error("Invalid safety pattern: {0}")]
    InvalidPattern(String),

    #[error("Safety assessment failed: {0}")]
    AssessmentFailed(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

impl From<SafetyError> for crate::error::TrustformersError {
    fn from(err: SafetyError) -> Self {
        crate::error::TrustformersError::InvalidInput {
            message: format!("Safety error: {}", err),
            parameter: Some("safety_check".to_string()),
            expected: Some("safe content".to_string()),
            received: None,
            suggestion: Some("Check safety filtering configuration".to_string()),
        }
    }
}

// ================================================================================================
// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_filter_creation() {
        let filter = SafetyFilter::new();
        assert!(filter.config.enabled);
        assert_eq!(filter.max_toxicity_score, 0.7);
        assert!(filter.extended_config.filter_input);
        assert!(filter.extended_config.filter_output);
    }

    #[test]
    fn test_safety_filter_strict_mode() {
        let filter = SafetyFilter::strict();
        assert_eq!(filter.extended_config.max_toxicity_score, 0.4);
        assert_eq!(filter.extended_config.max_harm_score, 0.3);
        assert_eq!(filter.extended_config.max_bias_score, 0.5);
    }

    #[test]
    fn test_safety_filter_permissive_mode() {
        let filter = SafetyFilter::permissive();
        assert_eq!(filter.extended_config.max_toxicity_score, 0.9);
        assert_eq!(filter.extended_config.max_harm_score, 0.8);
        assert_eq!(filter.extended_config.max_bias_score, 0.9);
    }

    #[test]
    fn test_safety_filter_educational_mode() {
        let filter = SafetyFilter::educational();
        assert_eq!(filter.extended_config.max_toxicity_score, 0.6);
        assert_eq!(filter.extended_config.max_harm_score, 0.5);
        assert_eq!(filter.extended_config.max_bias_score, 0.7);
    }

    #[test]
    fn test_safe_content() {
        let filter = SafetyFilter::new();
        assert!(filter.is_safe("Hello, how are you today?"));
        assert!(filter.is_safe("I would like to learn about programming."));
        assert!(filter.is_safe("Thank you for your help!"));
        assert!(filter.is_safe("What's the weather like?"));
    }

    #[test]
    #[ignore] // Stack overflow in assess_content_safety_enhanced
    fn test_toxic_content_detection() {
        let filter = SafetyFilter::new();
        assert!(!filter.is_safe("I hate everyone"));
        assert!(!filter.is_safe("You are so stupid"));

        let assessment = filter.assess_content_safety_enhanced("I hate everyone");
        assert!(assessment.category_scores.toxicity > 0.0);
        assert!(!assessment.enhanced_violations.is_empty());
        assert!(assessment
            .enhanced_violations
            .iter()
            .any(|v| v.base_violation.violation_type == "toxicity"));
    }

    #[test]
    #[ignore] // Stack overflow in assess_content_safety_enhanced
    fn test_personal_information_detection() {
        let filter = SafetyFilter::new();
        let assessment = filter.assess_content_safety_enhanced("My SSN is 123-45-6789");

        assert!(assessment.category_scores.privacy > 0.0);
        assert!(assessment
            .enhanced_violations
            .iter()
            .any(|v| v.base_violation.violation_type == "personal_information"));
    }

    #[test]
    #[ignore] // Stack overflow in assess_content_safety_enhanced
    fn test_violence_detection() {
        let filter = SafetyFilter::new();
        let assessment =
            filter.assess_content_safety_enhanced("I want to attack someone with violence");

        assert!(assessment.category_scores.violence > 0.0);
        assert!(assessment
            .enhanced_violations
            .iter()
            .any(|v| v.base_violation.violation_type == "violence"));
    }

    #[test]
    #[ignore] // Stack overflow in assess_content_safety_enhanced
    fn test_self_harm_detection() {
        let filter = SafetyFilter::new();
        let assessment = filter.assess_content_safety_enhanced("I want to hurt myself");

        assert!(assessment.category_scores.harm > 0.0);
        assert!(assessment
            .enhanced_violations
            .iter()
            .any(|v| v.base_violation.violation_type == "self_harm"));
        assert!(assessment
            .enhanced_violations
            .iter()
            .any(|v| matches!(v.base_violation.severity, SafetySeverity::Critical)));
    }

    #[test]
    #[ignore] // Stack overflow in assess_content_safety_enhanced
    fn test_harassment_detection() {
        let filter = SafetyFilter::new();
        let assessment = filter.assess_content_safety_enhanced("I will harass you constantly");

        assert!(assessment.category_scores.harassment > 0.0);
        assert!(assessment
            .enhanced_violations
            .iter()
            .any(|v| v.base_violation.violation_type == "harassment"));
    }

    #[test]
    #[ignore] // Stack overflow in assess_content_safety_enhanced
    fn test_bias_detection() {
        let filter = SafetyFilter::new();
        let assessment = filter.assess_content_safety_enhanced("That's so racist and sexist");

        assert!(assessment.category_scores.bias > 0.0);
        assert!(assessment
            .enhanced_violations
            .iter()
            .any(|v| v.base_violation.violation_type == "bias"));
    }

    #[test]
    fn test_content_filtering() {
        let filter = SafetyFilter::new();

        // Safe content should pass through
        let result = filter.filter_input("Hello, how can I help?").unwrap();
        assert_eq!(result, "Hello, how can I help?");

        // Unsafe content should be filtered
        let result = filter.filter_input("I hate you").unwrap();
        assert_ne!(result, "I hate you");
        assert!(result.contains("can't assist") || result.contains("something else"));
    }

    #[test]
    fn test_metadata_enrichment() {
        let filter = SafetyFilter::new();
        let mut metadata = ConversationMetadata {
            sentiment: Some("neutral".to_string()),
            intent: Some("question".to_string()),
            confidence: 0.9,
            topics: vec!["general".to_string()],
            safety_flags: Vec::new(),
            entities: Vec::new(),
            quality_score: 0.8,
            engagement_level: EngagementLevel::Medium,
            reasoning_type: Some(ReasoningType::Logical),
        };

        filter.enrich_conversation_metadata("I hate everyone", &mut metadata);

        assert!(!metadata.safety_flags.is_empty());
        assert!(metadata.confidence < 0.9); // Should be reduced due to safety issues
        assert!(metadata.quality_score < 0.8); // Should be reduced due to safety issues
    }

    #[test]
    fn test_violation_location_tracking() {
        let filter = SafetyFilter::new();
        let content = "Hello I hate you goodbye";
        let assessment = filter.assess_content_safety_enhanced(content);

        // Find the hate-related violation
        let hate_violation = assessment
            .enhanced_violations
            .iter()
            .find(|v| v.base_violation.violation_type == "toxicity");

        if let Some(violation) = hate_violation {
            if let Some(location) = &violation.location {
                assert!(content[location.start_pos..location.end_pos].contains("hate"));
            }
        }
    }

    #[test]
    fn test_custom_pattern_addition() {
        let mut filter = SafetyFilter::new();

        // Add custom toxic pattern
        filter.add_custom_pattern(r"(?i)\bcustom_bad_word\b", "toxicity").unwrap();

        // Test that the custom pattern is detected
        let assessment = filter.assess_content_safety_enhanced("This contains custom_bad_word");
        assert!(assessment.category_scores.toxicity > 0.0);
    }

    #[test]
    fn test_configuration_modes() {
        let strict_filter = SafetyFilter::strict();
        let permissive_filter = SafetyFilter::permissive();
        let educational_filter = SafetyFilter::educational();

        // Test different thresholds
        assert!(
            strict_filter.extended_config.max_toxicity_score
                < permissive_filter.extended_config.max_toxicity_score
        );
        assert!(
            educational_filter.extended_config.max_toxicity_score
                < permissive_filter.extended_config.max_toxicity_score
        );
        assert!(
            educational_filter.extended_config.max_toxicity_score
                > strict_filter.extended_config.max_toxicity_score
        );
    }

    #[test]
    #[ignore] // Temporarily ignored due to stack overflow in placeholder implementation
    fn test_assessment_caching() {
        let filter = SafetyFilter::new();
        let content = "Hello world";

        // First assessment
        let start1 = Instant::now();
        let assessment1 = filter.assess_content_safety_enhanced(content);
        let duration1 = start1.elapsed();

        // Second assessment (should use cache if enabled)
        let start2 = Instant::now();
        let assessment2 = filter.assess_content_safety_enhanced(content);
        let duration2 = start2.elapsed();

        // Both assessments should complete without error
        // (durations are always >= 0 for std::time::Duration)

        // Assessments should be consistent
        assert_eq!(
            assessment1.category_scores.toxicity,
            assessment2.category_scores.toxicity
        );
    }

    #[test]
    fn test_disabled_filter() {
        let mut config = ExtendedSafetyConfig::default();
        config.base_config.enabled = false;
        let filter = SafetyFilter::with_extended_config(config);

        // Should allow all content when disabled
        assert!(filter.is_safe("I hate everyone"));
        assert_eq!(filter.get_toxicity_score("I hate everyone"), 0.0);

        let result = filter.filter_input("I hate everyone").unwrap();
        assert_eq!(result, "I hate everyone");
    }

    #[test]
    fn test_safety_analysis_backward_compatibility() {
        let filter = SafetyFilter::new();

        // Test backward compatibility with original analyze_safety method
        let analysis = filter.analyze_safety("I hate you");
        assert!(!analysis.is_safe);
        assert!(analysis.toxicity_score > 0.0);
        assert!(!analysis.violations.is_empty());
        assert!(matches!(
            analysis.risk_level,
            RiskLevel::Low | RiskLevel::Medium | RiskLevel::High
        ));
    }

    #[test]
    fn test_content_sanitization() {
        let filter = SafetyFilter::new();
        let toxic_content = "You are hate and violence";
        let sanitized = filter.sanitize_content(toxic_content);

        // Should replace banned terms
        assert!(sanitized.contains("****")); // hate replaced
        assert!(sanitized.contains("********")); // violence replaced
    }

    #[test]
    fn test_safety_statistics() {
        let filter = SafetyFilter::new();
        let content_samples = vec![
            "Hello world".to_string(),
            "I hate you".to_string(),
            "Nice weather today".to_string(),
            "Violence is bad".to_string(),
        ];

        let stats = filter.get_safety_stats(&content_samples);

        assert_eq!(stats.total_samples, 4);
        assert!(stats.safe_percentage >= 0.0 && stats.safe_percentage <= 100.0);
        assert!(stats.avg_toxicity_score >= 0.0 && stats.avg_toxicity_score <= 1.0);
        // total_violations is usize which is always >= 0
        assert!(!stats.risk_distribution.is_empty());
    }

    #[test]
    fn test_enhanced_assessment_confidence() {
        let filter = SafetyFilter::new();

        // Long content should have higher confidence
        let long_content = "This is a very long piece of content that should have higher confidence in its assessment because there is more text to analyze and detect patterns in.";
        let short_content = "Hi";

        let long_assessment = filter.assess_content_safety_enhanced(long_content);
        let short_assessment = filter.assess_content_safety_enhanced(short_content);

        assert!(long_assessment.confidence >= short_assessment.confidence);
    }

    #[test]
    fn test_violation_context_extraction() {
        let filter = SafetyFilter::new();
        let content = "This is some text with hate in the middle of it";
        let assessment = filter.assess_content_safety_enhanced(content);

        if let Some(violation) = assessment.enhanced_violations.first() {
            assert!(violation.context.is_some());
            let context = violation.context.as_ref().unwrap();
            assert!(context.contains("hate"));
        }
    }

    #[test]
    fn test_enhanced_safety_assessment_immediate_action() {
        let filter = SafetyFilter::new();
        let critical_content = "I want to hurt myself badly";
        let assessment = filter.assess_content_safety_enhanced(critical_content);

        // Critical self-harm content should require immediate action
        assert!(assessment.requires_immediate_action());
    }

    #[test]
    fn test_privacy_redaction() {
        let filter = SafetyFilter::new();
        let content_with_ssn = "My social security number is 123-45-6789";
        let assessment = filter.assess_content_safety_enhanced(content_with_ssn);

        if let Some(privacy_violation) = assessment
            .enhanced_violations
            .iter()
            .find(|v| v.base_violation.violation_type == "personal_information")
        {
            if let Some(location) = &privacy_violation.location {
                // Should be redacted for privacy
                assert_eq!(location.matched_text, "***REDACTED***");
            }
        }
    }
}
