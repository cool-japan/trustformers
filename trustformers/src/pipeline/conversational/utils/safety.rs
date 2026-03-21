//! Enhanced safety filtering utilities.
//!
//! This module provides comprehensive safety analysis, risk assessment,
//! and content filtering capabilities for conversational AI systems.

use serde::{Deserialize, Serialize};

/// Enhanced safety filter for conversation content
pub struct EnhancedSafetyFilter {
    /// Sensitivity level for safety checks
    sensitivity_level: f32,
    /// Custom filtering rules
    custom_rules: Vec<String>,
}

impl EnhancedSafetyFilter {
    /// Create a new enhanced safety filter
    pub fn new(sensitivity_level: f32) -> Self {
        Self {
            sensitivity_level: sensitivity_level.clamp(0.0, 1.0),
            custom_rules: Vec::new(),
        }
    }

    /// Add custom safety rule
    pub fn add_custom_rule(&mut self, rule: String) {
        self.custom_rules.push(rule);
    }

    /// Analyze content for safety issues
    pub fn analyze_content(&self, content: &str) -> SafetyAnalysis {
        let mut analysis = SafetyAnalysis {
            is_safe: true,
            risk_level: RiskLevel::Low,
            confidence: 0.9,
            issues: Vec::new(),
            filtered_content: content.to_string(),
            recommendations: Vec::new(),
        };

        // Check for various risk categories
        self.check_violence(content, &mut analysis);
        self.check_inappropriate_content(content, &mut analysis);
        self.check_personal_information(content, &mut analysis);
        self.check_hate_speech(content, &mut analysis);
        self.check_self_harm(content, &mut analysis);
        self.check_illegal_content(content, &mut analysis);
        self.check_adult_content(content, &mut analysis);
        self.check_custom_rules(content, &mut analysis);

        // Adjust based on sensitivity level
        self.apply_sensitivity_adjustments(&mut analysis);

        // Generate recommendations
        analysis.recommendations = self.generate_safety_recommendations(&analysis);

        analysis
    }

    fn check_violence(&self, content: &str, analysis: &mut SafetyAnalysis) {
        let violence_keywords = [
            "kill", "murder", "hurt", "harm", "attack", "violence", "weapon", "fight", "assault",
            "stab", "shoot", "punch", "hit", "beat", "destroy",
        ];

        let content_lower = content.to_lowercase();
        let violence_count = violence_keywords
            .iter()
            .filter(|&keyword| content_lower.contains(keyword))
            .count();

        if violence_count > 0 {
            analysis.issues.push("Violence-related content detected".to_string());
            analysis.risk_level = match violence_count {
                1..=2 => RiskLevel::Medium,
                _ => RiskLevel::High,
            };
        }
    }

    fn check_inappropriate_content(&self, content: &str, analysis: &mut SafetyAnalysis) {
        let inappropriate_keywords = [
            "inappropriate",
            "offensive",
            "rude",
            "insulting",
            "harassment",
            "discriminatory",
        ];

        let content_lower = content.to_lowercase();
        let inappropriate_count = inappropriate_keywords
            .iter()
            .filter(|&keyword| content_lower.contains(keyword))
            .count();

        if inappropriate_count > 0 {
            analysis.issues.push("Inappropriate content detected".to_string());
            if matches!(analysis.risk_level, RiskLevel::Low) {
                analysis.risk_level = RiskLevel::Medium;
            }
        }
    }

    fn check_personal_information(&self, content: &str, analysis: &mut SafetyAnalysis) {
        let pi_keywords = [
            "password",
            "ssn",
            "social security",
            "credit card",
            "bank account",
            "phone number",
            "address",
            "email",
        ];

        let content_lower = content.to_lowercase();
        let pi_count =
            pi_keywords.iter().filter(|&keyword| content_lower.contains(keyword)).count();

        if pi_count > 0 {
            analysis.issues.push("Personal information detected".to_string());
            if matches!(analysis.risk_level, RiskLevel::Low) {
                analysis.risk_level = RiskLevel::Medium;
            }
        }
    }

    fn check_hate_speech(&self, content: &str, analysis: &mut SafetyAnalysis) {
        let hate_keywords = [
            "hate",
            "racist",
            "sexist",
            "discrimination",
            "prejudice",
            "bigotry",
        ];

        let content_lower = content.to_lowercase();
        let hate_count =
            hate_keywords.iter().filter(|&keyword| content_lower.contains(keyword)).count();

        if hate_count > 0 {
            analysis.issues.push("Hate speech detected".to_string());
            analysis.risk_level = RiskLevel::High;
        }
    }

    fn check_self_harm(&self, content: &str, analysis: &mut SafetyAnalysis) {
        let self_harm_keywords = [
            "suicide",
            "self-harm",
            "cut myself",
            "kill myself",
            "end it all",
            "hurt myself",
        ];

        let content_lower = content.to_lowercase();
        let self_harm_count = self_harm_keywords
            .iter()
            .filter(|&keyword| content_lower.contains(keyword))
            .count();

        if self_harm_count > 0 {
            analysis.issues.push("Self-harm content detected".to_string());
            analysis.risk_level = RiskLevel::Critical;
        }
    }

    fn check_illegal_content(&self, content: &str, analysis: &mut SafetyAnalysis) {
        let illegal_keywords = ["illegal", "drugs", "steal", "fraud", "scam", "criminal"];

        let content_lower = content.to_lowercase();
        let illegal_count = illegal_keywords
            .iter()
            .filter(|&keyword| content_lower.contains(keyword))
            .count();

        if illegal_count > 0 {
            analysis.issues.push("Illegal content detected".to_string());
            if matches!(analysis.risk_level, RiskLevel::Low | RiskLevel::Medium) {
                analysis.risk_level = RiskLevel::High;
            }
        }
    }

    fn check_adult_content(&self, content: &str, analysis: &mut SafetyAnalysis) {
        let adult_keywords = ["sexual", "explicit", "pornographic", "adult content"];

        let content_lower = content.to_lowercase();
        let adult_count =
            adult_keywords.iter().filter(|&keyword| content_lower.contains(keyword)).count();

        if adult_count > 0 {
            analysis.issues.push("Adult content detected".to_string());
            if matches!(analysis.risk_level, RiskLevel::Low) {
                analysis.risk_level = RiskLevel::Medium;
            }
        }
    }

    fn check_custom_rules(&self, content: &str, analysis: &mut SafetyAnalysis) {
        let content_lower = content.to_lowercase();

        for rule in &self.custom_rules {
            if content_lower.contains(&rule.to_lowercase()) {
                analysis.issues.push(format!("Custom rule violation: {}", rule));
                if matches!(analysis.risk_level, RiskLevel::Low) {
                    analysis.risk_level = RiskLevel::Medium;
                }
            }
        }
    }

    fn apply_sensitivity_adjustments(&self, analysis: &mut SafetyAnalysis) {
        if self.sensitivity_level > 0.8 {
            // High sensitivity - escalate risk levels
            analysis.risk_level = match analysis.risk_level {
                RiskLevel::Low => {
                    if !analysis.issues.is_empty() {
                        RiskLevel::Medium
                    } else {
                        RiskLevel::Low
                    }
                },
                RiskLevel::Medium => RiskLevel::High,
                other => other,
            };
        } else if self.sensitivity_level < 0.3 {
            // Low sensitivity - reduce risk levels
            analysis.risk_level = match analysis.risk_level {
                RiskLevel::High => RiskLevel::Medium,
                RiskLevel::Medium => RiskLevel::Low,
                other => other,
            };
        }

        analysis.is_safe = matches!(analysis.risk_level, RiskLevel::Low);
        analysis.confidence *= self.sensitivity_level;
    }

    fn generate_safety_recommendations(&self, analysis: &SafetyAnalysis) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !analysis.is_safe {
            recommendations.push("Content review recommended".to_string());
        }

        match analysis.risk_level {
            RiskLevel::Critical => {
                recommendations.push("Immediate intervention required".to_string());
                recommendations.push("Block content and alert moderators".to_string());
            },
            RiskLevel::High => {
                recommendations.push("Content should be blocked".to_string());
                recommendations.push("Consider user education".to_string());
            },
            RiskLevel::Medium => {
                recommendations.push("Content warning may be appropriate".to_string());
                recommendations.push("Monitor user behavior".to_string());
            },
            RiskLevel::Low => {
                if !analysis.issues.is_empty() {
                    recommendations.push("Minor content adjustments suggested".to_string());
                }
            },
        }

        recommendations
    }
}

/// Result of safety analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyAnalysis {
    /// Whether the content is considered safe
    pub is_safe: bool,
    /// Risk level assessment
    pub risk_level: RiskLevel,
    /// Confidence in the analysis (0.0 to 1.0)
    pub confidence: f32,
    /// List of identified issues
    pub issues: Vec<String>,
    /// Filtered version of the content
    pub filtered_content: String,
    /// Safety recommendations
    pub recommendations: Vec<String>,
}

/// Risk level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk - content is generally safe
    Low,
    /// Medium risk - content may be problematic
    Medium,
    /// High risk - content is likely problematic
    High,
    /// Critical risk - content requires immediate attention
    Critical,
}

/// Filter result for content screening
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterResult {
    /// Whether content should be allowed
    pub allowed: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Reason for filtering decision
    pub reason: Option<String>,
    /// Suggested alternative if content is blocked
    pub alternative: Option<String>,
}
