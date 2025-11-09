//! Model Cards for TrustformeRS
//!
//! This module provides a comprehensive model card system for documenting
//! model capabilities, limitations, ethical considerations, and usage guidelines.
//! Based on the Model Cards for Model Reporting framework by Mitchell et al.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model card containing comprehensive model documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    /// Model identification and basic information
    pub model_details: ModelDetails,
    /// Intended use cases and applications
    pub intended_use: IntendedUse,
    /// Factors affecting model performance
    pub factors: Factors,
    /// Evaluation metrics and performance data
    pub metrics: Metrics,
    /// Training and evaluation data information
    pub data: DataInfo,
    /// Ethical considerations and recommendations
    pub ethical_considerations: EthicalConsiderations,
    /// Caveats and recommendations
    pub caveats_and_recommendations: CaveatsAndRecommendations,
    /// Model card metadata
    pub metadata: ModelCardMetadata,
}

/// Model identification and basic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetails {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model architecture (e.g., "transformer", "gpt", "bert")
    pub architecture: String,
    /// Model size information
    pub size: ModelSize,
    /// Brief description of the model
    pub description: String,
    /// Model authors/developers
    pub developers: Vec<String>,
    /// Organization/institution
    pub organization: Option<String>,
    /// Model license
    pub license: String,
    /// Publication date
    pub date: DateTime<Utc>,
    /// Related papers/publications
    pub papers: Vec<Publication>,
    /// Model repository/source
    pub repository: Option<String>,
    /// Contact information
    pub contact: Option<String>,
}

/// Model size information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSize {
    /// Number of parameters
    pub parameters: u64,
    /// Model size on disk (in bytes)
    pub disk_size: u64,
    /// Memory requirements (in bytes)
    pub memory_requirements: u64,
    /// Inference time (milliseconds per token)
    pub inference_time_ms: Option<f64>,
}

/// Publication information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Publication {
    /// Paper title
    pub title: String,
    /// Authors
    pub authors: Vec<String>,
    /// Publication venue
    pub venue: Option<String>,
    /// Publication year
    pub year: u32,
    /// URL or DOI
    pub url: Option<String>,
}

/// Intended use cases and applications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntendedUse {
    /// Primary use cases
    pub primary_uses: Vec<String>,
    /// Out-of-scope use cases
    pub out_of_scope_uses: Vec<String>,
    /// Target users
    pub target_users: Vec<String>,
    /// Supported languages
    pub languages: Vec<String>,
    /// Domains/fields of application
    pub domains: Vec<String>,
    /// Usage restrictions
    pub restrictions: Vec<String>,
}

/// Factors affecting model performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Factors {
    /// Relevant factors (demographic, environmental, technical)
    pub relevant_factors: Vec<Factor>,
    /// Evaluation factors
    pub evaluation_factors: Vec<String>,
}

/// Individual factor that may affect model performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Factor {
    /// Factor name
    pub name: String,
    /// Factor type (demographic, environmental, technical)
    pub factor_type: FactorType,
    /// Description of how this factor affects the model
    pub description: String,
    /// Impact level (low, medium, high)
    pub impact_level: ImpactLevel,
}

/// Types of factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorType {
    /// Demographic factors (age, gender, race, etc.)
    Demographic,
    /// Environmental factors (lighting, noise, etc.)
    Environmental,
    /// Technical factors (hardware, software, etc.)
    Technical,
    /// Data factors (data quality, distribution, etc.)
    Data,
    /// Other factors
    Other(String),
}

/// Impact level of a factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// Low impact on model performance
    Low,
    /// Medium impact on model performance
    Medium,
    /// High impact on model performance
    High,
}

/// Evaluation metrics and performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    /// Model performance metrics
    pub performance_metrics: Vec<PerformanceMetric>,
    /// Decision thresholds
    pub decision_thresholds: Vec<DecisionThreshold>,
    /// Approaches to uncertainty and variability
    pub uncertainty_approaches: Vec<String>,
}

/// Individual performance metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    /// Metric name
    pub name: String,
    /// Metric type (accuracy, f1, perplexity, etc.)
    pub metric_type: MetricType,
    /// Metric value
    pub value: f64,
    /// Standard deviation or confidence interval
    pub uncertainty: Option<f64>,
    /// Dataset used for evaluation
    pub dataset: String,
    /// Subgroup or demographic breakdown
    pub subgroup: Option<String>,
    /// Additional context or notes
    pub notes: Option<String>,
}

/// Types of performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    /// Accuracy metric
    Accuracy,
    /// Precision metric
    Precision,
    /// Recall metric
    Recall,
    /// F1 score
    F1Score,
    /// Perplexity (for language models)
    Perplexity,
    /// BLEU score (for translation/generation)
    BleuScore,
    /// ROUGE score (for summarization)
    RougeScore,
    /// Area under the curve
    Auc,
    /// Mean squared error
    Mse,
    /// Custom metric
    Custom(String),
}

/// Decision threshold information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionThreshold {
    /// Threshold name/description
    pub name: String,
    /// Threshold value
    pub value: f64,
    /// Rationale for this threshold
    pub rationale: String,
}

/// Training and evaluation data information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataInfo {
    /// Training datasets
    pub training_data: Vec<DatasetInfo>,
    /// Evaluation datasets
    pub evaluation_data: Vec<DatasetInfo>,
    /// Data preprocessing steps
    pub preprocessing: Vec<String>,
}

/// Information about a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset name
    pub name: String,
    /// Dataset description
    pub description: String,
    /// Dataset source/provider
    pub source: String,
    /// Dataset size (number of samples)
    pub size: u64,
    /// Data collection timeframe
    pub collection_timeframe: Option<String>,
    /// Data collection methodology
    pub collection_methodology: Option<String>,
    /// Known biases or limitations
    pub known_biases: Vec<String>,
    /// Data license
    pub license: Option<String>,
}

/// Ethical considerations and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalConsiderations {
    /// Sensitive use cases
    pub sensitive_use_cases: Vec<String>,
    /// Risks and harms
    pub risks_and_harms: Vec<RiskHarm>,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
    /// Fairness considerations
    pub fairness_considerations: Vec<String>,
    /// Privacy considerations
    pub privacy_considerations: Vec<String>,
    /// Human oversight requirements
    pub human_oversight: Vec<String>,
}

/// Risk or harm information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskHarm {
    /// Risk/harm description
    pub description: String,
    /// Risk level (low, medium, high)
    pub risk_level: RiskLevel,
    /// Affected groups
    pub affected_groups: Vec<String>,
    /// Likelihood of occurrence
    pub likelihood: Likelihood,
    /// Potential impact
    pub potential_impact: String,
}

/// Risk level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Critical risk
    Critical,
}

/// Likelihood of risk occurrence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Likelihood {
    /// Very unlikely
    VeryUnlikely,
    /// Unlikely
    Unlikely,
    /// Possible
    Possible,
    /// Likely
    Likely,
    /// Very likely
    VeryLikely,
}

/// Caveats and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaveatsAndRecommendations {
    /// Model limitations
    pub limitations: Vec<String>,
    /// Usage recommendations
    pub recommendations: Vec<String>,
    /// Ideal assessment approaches
    pub ideal_assessment: Vec<String>,
    /// Additional information
    pub additional_information: Option<String>,
}

/// Model card metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCardMetadata {
    /// Model card version
    pub version: String,
    /// Creation date
    pub created_date: DateTime<Utc>,
    /// Last updated date
    pub updated_date: DateTime<Utc>,
    /// Authors of the model card
    pub authors: Vec<String>,
    /// Review status
    pub review_status: ReviewStatus,
    /// Additional metadata
    pub additional_metadata: HashMap<String, String>,
}

/// Review status of the model card
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewStatus {
    /// Draft version
    Draft,
    /// Under review
    UnderReview,
    /// Approved
    Approved,
    /// Needs revision
    NeedsRevision,
}

/// Model card builder for easy construction
#[derive(Debug)]
pub struct ModelCardBuilder {
    card: ModelCard,
}

impl ModelCardBuilder {
    /// Create a new model card builder
    pub fn new(name: String, version: String, architecture: String) -> Self {
        let now = Utc::now();
        Self {
            card: ModelCard {
                model_details: ModelDetails {
                    name,
                    version,
                    architecture,
                    size: ModelSize {
                        parameters: 0,
                        disk_size: 0,
                        memory_requirements: 0,
                        inference_time_ms: None,
                    },
                    description: String::new(),
                    developers: Vec::new(),
                    organization: None,
                    license: "Unknown".to_string(),
                    date: now,
                    papers: Vec::new(),
                    repository: None,
                    contact: None,
                },
                intended_use: IntendedUse {
                    primary_uses: Vec::new(),
                    out_of_scope_uses: Vec::new(),
                    target_users: Vec::new(),
                    languages: Vec::new(),
                    domains: Vec::new(),
                    restrictions: Vec::new(),
                },
                factors: Factors {
                    relevant_factors: Vec::new(),
                    evaluation_factors: Vec::new(),
                },
                metrics: Metrics {
                    performance_metrics: Vec::new(),
                    decision_thresholds: Vec::new(),
                    uncertainty_approaches: Vec::new(),
                },
                data: DataInfo {
                    training_data: Vec::new(),
                    evaluation_data: Vec::new(),
                    preprocessing: Vec::new(),
                },
                ethical_considerations: EthicalConsiderations {
                    sensitive_use_cases: Vec::new(),
                    risks_and_harms: Vec::new(),
                    mitigation_strategies: Vec::new(),
                    fairness_considerations: Vec::new(),
                    privacy_considerations: Vec::new(),
                    human_oversight: Vec::new(),
                },
                caveats_and_recommendations: CaveatsAndRecommendations {
                    limitations: Vec::new(),
                    recommendations: Vec::new(),
                    ideal_assessment: Vec::new(),
                    additional_information: None,
                },
                metadata: ModelCardMetadata {
                    version: "1.0".to_string(),
                    created_date: now,
                    updated_date: now,
                    authors: Vec::new(),
                    review_status: ReviewStatus::Draft,
                    additional_metadata: HashMap::new(),
                },
            },
        }
    }

    /// Set model description
    pub fn description(mut self, description: String) -> Self {
        self.card.model_details.description = description;
        self
    }

    /// Add developer
    pub fn developer(mut self, developer: String) -> Self {
        self.card.model_details.developers.push(developer);
        self
    }

    /// Set organization
    pub fn organization(mut self, organization: String) -> Self {
        self.card.model_details.organization = Some(organization);
        self
    }

    /// Set license
    pub fn license(mut self, license: String) -> Self {
        self.card.model_details.license = license;
        self
    }

    /// Set model size
    pub fn size(mut self, parameters: u64, disk_size: u64, memory_requirements: u64) -> Self {
        self.card.model_details.size = ModelSize {
            parameters,
            disk_size,
            memory_requirements,
            inference_time_ms: None,
        };
        self
    }

    /// Add primary use case
    pub fn primary_use(mut self, use_case: String) -> Self {
        self.card.intended_use.primary_uses.push(use_case);
        self
    }

    /// Add out-of-scope use case
    pub fn out_of_scope_use(mut self, use_case: String) -> Self {
        self.card.intended_use.out_of_scope_uses.push(use_case);
        self
    }

    /// Add performance metric
    pub fn metric(mut self, metric: PerformanceMetric) -> Self {
        self.card.metrics.performance_metrics.push(metric);
        self
    }

    /// Add training dataset
    pub fn training_dataset(mut self, dataset: DatasetInfo) -> Self {
        self.card.data.training_data.push(dataset);
        self
    }

    /// Add ethical consideration
    pub fn risk(mut self, risk: RiskHarm) -> Self {
        self.card.ethical_considerations.risks_and_harms.push(risk);
        self
    }

    /// Add limitation
    pub fn limitation(mut self, limitation: String) -> Self {
        self.card.caveats_and_recommendations.limitations.push(limitation);
        self
    }

    /// Add recommendation
    pub fn recommendation(mut self, recommendation: String) -> Self {
        self.card.caveats_and_recommendations.recommendations.push(recommendation);
        self
    }

    /// Build the model card
    pub fn build(self) -> ModelCard {
        self.card
    }
}

impl ModelCard {
    /// Create a new model card builder
    pub fn builder(name: String, version: String, architecture: String) -> ModelCardBuilder {
        ModelCardBuilder::new(name, version, architecture)
    }

    /// Export model card to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export model card to YAML
    pub fn to_yaml(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }

    /// Import model card from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Import model card from YAML
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    /// Generate HTML representation
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!(
            "<title>Model Card: {}</title>\n",
            self.model_details.name
        ));
        html.push_str("<style>\n");
        html.push_str(include_str!("../assets/model_card.css"));
        html.push_str("</style>\n</head>\n<body>\n");

        // Header
        html.push_str(&format!(
            "<h1>Model Card: {}</h1>\n",
            self.model_details.name
        ));
        html.push_str(&format!(
            "<p class='version'>Version: {}</p>\n",
            self.model_details.version
        ));

        // Model Details
        html.push_str("<h2>Model Details</h2>\n");
        html.push_str(&format!(
            "<p><strong>Architecture:</strong> {}</p>\n",
            self.model_details.architecture
        ));
        html.push_str(&format!(
            "<p><strong>Parameters:</strong> {}</p>\n",
            self.model_details.size.parameters
        ));
        html.push_str(&format!(
            "<p><strong>Description:</strong> {}</p>\n",
            self.model_details.description
        ));
        html.push_str(&format!(
            "<p><strong>License:</strong> {}</p>\n",
            self.model_details.license
        ));

        // Intended Use
        html.push_str("<h2>Intended Use</h2>\n");
        if !self.intended_use.primary_uses.is_empty() {
            html.push_str("<h3>Primary Uses</h3>\n<ul>\n");
            for use_case in &self.intended_use.primary_uses {
                html.push_str(&format!("<li>{}</li>\n", use_case));
            }
            html.push_str("</ul>\n");
        }

        if !self.intended_use.out_of_scope_uses.is_empty() {
            html.push_str("<h3>Out-of-Scope Uses</h3>\n<ul>\n");
            for use_case in &self.intended_use.out_of_scope_uses {
                html.push_str(&format!("<li>{}</li>\n", use_case));
            }
            html.push_str("</ul>\n");
        }

        // Performance Metrics
        if !self.metrics.performance_metrics.is_empty() {
            html.push_str("<h2>Performance Metrics</h2>\n");
            html.push_str("<table>\n<tr><th>Metric</th><th>Value</th><th>Dataset</th></tr>\n");
            for metric in &self.metrics.performance_metrics {
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{:.4}</td><td>{}</td></tr>\n",
                    metric.name, metric.value, metric.dataset
                ));
            }
            html.push_str("</table>\n");
        }

        // Ethical Considerations
        if !self.ethical_considerations.risks_and_harms.is_empty() {
            html.push_str("<h2>Ethical Considerations</h2>\n");
            html.push_str("<h3>Risks and Harms</h3>\n");
            for risk in &self.ethical_considerations.risks_and_harms {
                html.push_str(&format!(
                    "<div class='risk risk-{:?}'><strong>{:?} Risk:</strong> {}</div>\n",
                    risk.risk_level, risk.risk_level, risk.description
                ));
            }
        }

        // Limitations
        if !self.caveats_and_recommendations.limitations.is_empty() {
            html.push_str("<h2>Limitations</h2>\n<ul>\n");
            for limitation in &self.caveats_and_recommendations.limitations {
                html.push_str(&format!("<li>{}</li>\n", limitation));
            }
            html.push_str("</ul>\n");
        }

        html.push_str("</body>\n</html>");
        html
    }

    /// Validate the model card for completeness
    pub fn validate(&self) -> ValidationReport {
        let mut report = ValidationReport::new();

        // Check required fields
        if self.model_details.name.is_empty() {
            report.add_error("Model name is required".to_string());
        }

        if self.model_details.description.is_empty() {
            report.add_warning("Model description should be provided".to_string());
        }

        if self.intended_use.primary_uses.is_empty() {
            report.add_warning("Primary use cases should be specified".to_string());
        }

        if self.metrics.performance_metrics.is_empty() {
            report.add_warning("Performance metrics should be provided".to_string());
        }

        if self.ethical_considerations.risks_and_harms.is_empty() {
            report.add_warning("Ethical considerations should be documented".to_string());
        }

        report
    }
}

/// Validation report for model cards
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationReport {
    fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

/// Predefined model card templates for common architectures
pub struct ModelCardTemplates;

impl ModelCardTemplates {
    /// Create a template for language models
    pub fn language_model(name: String, version: String) -> ModelCardBuilder {
        ModelCard::builder(name, version, "transformer".to_string())
            .primary_use("Text generation".to_string())
            .primary_use("Language understanding".to_string())
            .out_of_scope_use("Medical diagnosis".to_string())
            .out_of_scope_use("Legal advice".to_string())
            .limitation("May generate biased or inappropriate content".to_string())
            .limitation("Performance may vary across languages and domains".to_string())
            .recommendation("Use appropriate content filtering".to_string())
            .recommendation("Evaluate on your specific use case".to_string())
    }

    /// Create a template for vision models
    pub fn vision_model(name: String, version: String) -> ModelCardBuilder {
        ModelCard::builder(name, version, "cnn".to_string())
            .primary_use("Image classification".to_string())
            .primary_use("Computer vision tasks".to_string())
            .out_of_scope_use("Medical imaging diagnosis".to_string())
            .limitation("Performance may vary with image quality".to_string())
            .limitation("May exhibit bias across demographic groups".to_string())
            .recommendation("Validate on diverse test sets".to_string())
    }

    /// Create a template for multimodal models
    pub fn multimodal_model(name: String, version: String) -> ModelCardBuilder {
        ModelCard::builder(name, version, "multimodal_transformer".to_string())
            .primary_use("Vision-language understanding".to_string())
            .primary_use("Multimodal reasoning".to_string())
            .out_of_scope_use("Critical decision making".to_string())
            .limitation("May have alignment issues between modalities".to_string())
            .recommendation("Test on multimodal benchmarks".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_card_builder() {
        let card = ModelCard::builder(
            "TestModel".to_string(),
            "1.0".to_string(),
            "transformer".to_string(),
        )
        .description("A test model".to_string())
        .developer("Test Developer".to_string())
        .license("MIT".to_string())
        .size(1000000, 500000000, 1000000000)
        .primary_use("Testing".to_string())
        .limitation("This is just a test".to_string())
        .build();

        assert_eq!(card.model_details.name, "TestModel");
        assert_eq!(card.model_details.version, "1.0");
        assert_eq!(card.model_details.architecture, "transformer");
        assert_eq!(card.model_details.description, "A test model");
        assert_eq!(card.model_details.size.parameters, 1000000);
        assert!(card.intended_use.primary_uses.contains(&"Testing".to_string()));
        assert!(card
            .caveats_and_recommendations
            .limitations
            .contains(&"This is just a test".to_string()));
    }

    #[test]
    fn test_model_card_serialization() {
        let card = ModelCard::builder(
            "SerializationTest".to_string(),
            "1.0".to_string(),
            "transformer".to_string(),
        )
        .build();

        let json = card.to_json().unwrap();
        let deserialized_card = ModelCard::from_json(&json).unwrap();

        assert_eq!(
            card.model_details.name,
            deserialized_card.model_details.name
        );
        assert_eq!(
            card.model_details.version,
            deserialized_card.model_details.version
        );
    }

    #[test]
    fn test_model_card_validation() {
        let empty_card =
            ModelCard::builder(String::new(), "1.0".to_string(), "transformer".to_string()).build();

        let report = empty_card.validate();
        assert!(!report.is_valid());
        assert!(!report.errors.is_empty());

        let complete_card = ModelCard::builder(
            "CompleteModel".to_string(),
            "1.0".to_string(),
            "transformer".to_string(),
        )
        .description("A complete model".to_string())
        .primary_use("Testing".to_string())
        .build();

        let report = complete_card.validate();
        assert!(report.is_valid());
    }

    #[test]
    fn test_model_card_templates() {
        let lang_model =
            ModelCardTemplates::language_model("GPT-Test".to_string(), "1.0".to_string()).build();

        assert_eq!(lang_model.model_details.architecture, "transformer");
        assert!(lang_model.intended_use.primary_uses.contains(&"Text generation".to_string()));
        assert!(lang_model
            .intended_use
            .out_of_scope_uses
            .contains(&"Medical diagnosis".to_string()));

        let vision_model =
            ModelCardTemplates::vision_model("ResNet-Test".to_string(), "1.0".to_string()).build();

        assert_eq!(vision_model.model_details.architecture, "cnn");
        assert!(vision_model
            .intended_use
            .primary_uses
            .contains(&"Image classification".to_string()));
    }

    #[test]
    fn test_performance_metric() {
        let metric = PerformanceMetric {
            name: "Accuracy".to_string(),
            metric_type: MetricType::Accuracy,
            value: 0.95,
            uncertainty: Some(0.02),
            dataset: "Test Dataset".to_string(),
            subgroup: None,
            notes: None,
        };

        assert_eq!(metric.name, "Accuracy");
        assert_eq!(metric.value, 0.95);
        assert_eq!(metric.uncertainty, Some(0.02));
    }

    #[test]
    fn test_risk_harm() {
        let risk = RiskHarm {
            description: "Potential bias in outputs".to_string(),
            risk_level: RiskLevel::Medium,
            affected_groups: vec!["Minority groups".to_string()],
            likelihood: Likelihood::Possible,
            potential_impact: "Unfair treatment".to_string(),
        };

        assert_eq!(risk.description, "Potential bias in outputs");
        assert!(matches!(risk.risk_level, RiskLevel::Medium));
        assert!(matches!(risk.likelihood, Likelihood::Possible));
    }

    #[test]
    fn test_html_generation() {
        let card = ModelCard::builder(
            "HTMLTest".to_string(),
            "1.0".to_string(),
            "transformer".to_string(),
        )
        .description("A test model for HTML generation".to_string())
        .primary_use("Testing HTML output".to_string())
        .build();

        let html = card.to_html();
        assert!(html.contains("Model Card: HTMLTest"));
        assert!(html.contains("Version: 1.0"));
        assert!(html.contains("A test model for HTML generation"));
        assert!(html.contains("Testing HTML output"));
    }
}
