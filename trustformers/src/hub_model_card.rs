//! Automatic model card (README.md) generation for HuggingFace Hub.
//!
//! Generates markdown following the HuggingFace model card specification,
//! including YAML front matter metadata and structured sections.

use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::debug;

// ─── Metadata ─────────────────────────────────────────────────────────────────

/// Model card metadata stored as YAML front matter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCardMetadata {
    /// Languages the model supports, e.g. `["en", "fr"]`
    pub language: Vec<String>,
    /// SPDX license identifier, e.g. `"apache-2.0"`
    pub license: String,
    /// Library name, typically `"trustformers"`
    pub library_name: String,
    /// Arbitrary tags, e.g. `["text-generation", "causal-lm"]`
    pub tags: Vec<String>,
    /// Training datasets referenced by the model
    pub datasets: Vec<String>,
    /// Evaluation metrics reported for this model
    pub metrics: Vec<String>,
    /// Model architecture type, e.g. `"bert"` or `"gpt2"`
    pub model_type: Option<String>,
    /// HuggingFace pipeline tag, e.g. `"text-generation"`
    pub pipeline_tag: Option<String>,
}

impl Default for ModelCardMetadata {
    fn default() -> Self {
        Self {
            language: vec!["en".to_string()],
            license: "apache-2.0".to_string(),
            library_name: "trustformers".to_string(),
            tags: Vec::new(),
            datasets: Vec::new(),
            metrics: Vec::new(),
            model_type: None,
            pipeline_tag: None,
        }
    }
}

// ─── BenchmarkResult ──────────────────────────────────────────────────────────

/// A single benchmark measurement for a model
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Task type, e.g. `"text-classification"`
    pub task: String,
    /// Benchmark dataset, e.g. `"glue/sst2"`
    pub dataset: String,
    /// Metric name, e.g. `"accuracy"`
    pub metric: String,
    /// Numeric value of the metric
    pub value: f64,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(
        task: impl Into<String>,
        dataset: impl Into<String>,
        metric: impl Into<String>,
        value: f64,
    ) -> Self {
        Self {
            task: task.into(),
            dataset: dataset.into(),
            metric: metric.into(),
            value,
        }
    }
}

// ─── TrainingInfo ─────────────────────────────────────────────────────────────

/// Information about the model training process
#[derive(Debug, Clone)]
pub struct TrainingInfo {
    /// Training framework name
    pub framework: String,
    /// Total number of trainable parameters
    pub num_parameters: Option<u64>,
    /// List of training data sources
    pub training_data: Vec<String>,
    /// Optimizer name, e.g. `"AdamW"`
    pub optimizer: Option<String>,
    /// Peak learning rate
    pub learning_rate: Option<f64>,
    /// Per-device batch size
    pub batch_size: Option<usize>,
    /// Number of training epochs
    pub num_epochs: Option<usize>,
    /// Hardware used for training, e.g. `"4x A100 80GB"`
    pub hardware: Option<String>,
}

impl Default for TrainingInfo {
    fn default() -> Self {
        Self {
            framework: "TrustformeRS".to_string(),
            num_parameters: None,
            training_data: Vec::new(),
            optimizer: None,
            learning_rate: None,
            batch_size: None,
            num_epochs: None,
            hardware: None,
        }
    }
}

// ─── ModelCard ────────────────────────────────────────────────────────────────

/// Full model card content combining metadata and markdown sections
#[derive(Debug, Clone)]
pub struct ModelCard {
    /// YAML front matter metadata
    pub metadata: ModelCardMetadata,
    /// Human-readable model name
    pub model_name: String,
    /// Short description of what the model does
    pub model_description: String,
    /// List of intended use cases
    pub intended_uses: Vec<String>,
    /// Known limitations of the model
    pub limitations: Vec<String>,
    /// Training details
    pub training_info: TrainingInfo,
    /// Benchmark results
    pub benchmarks: Vec<BenchmarkResult>,
    /// BibTeX or plain-text citation
    pub citation: Option<String>,
    /// Author or organisation name
    pub author: Option<String>,
}

impl ModelCard {
    /// Create a minimal model card from a name and description
    pub fn new(model_name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            metadata: ModelCardMetadata::default(),
            model_name: model_name.into(),
            model_description: description.into(),
            intended_uses: Vec::new(),
            limitations: Vec::new(),
            training_info: TrainingInfo::default(),
            benchmarks: Vec::new(),
            citation: None,
            author: None,
        }
    }

    /// Generate the full markdown string for the model card
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // ── YAML front matter ──────────────────────────────────────────────
        md.push_str("---\n");
        if !self.metadata.language.is_empty() {
            md.push_str("language:\n");
            for lang in &self.metadata.language {
                md.push_str(&format!("- {lang}\n"));
            }
        }
        md.push_str(&format!("license: {}\n", self.metadata.license));
        md.push_str(&format!("library_name: {}\n", self.metadata.library_name));
        if !self.metadata.tags.is_empty() {
            md.push_str("tags:\n");
            for tag in &self.metadata.tags {
                md.push_str(&format!("- {tag}\n"));
            }
        }
        if !self.metadata.datasets.is_empty() {
            md.push_str("datasets:\n");
            for ds in &self.metadata.datasets {
                md.push_str(&format!("- {ds}\n"));
            }
        }
        if !self.metadata.metrics.is_empty() {
            md.push_str("metrics:\n");
            for m in &self.metadata.metrics {
                md.push_str(&format!("- {m}\n"));
            }
        }
        if let Some(ref mt) = self.metadata.model_type {
            md.push_str(&format!("model_type: {mt}\n"));
        }
        if let Some(ref pt) = self.metadata.pipeline_tag {
            md.push_str(&format!("pipeline_tag: {pt}\n"));
        }
        md.push_str("---\n\n");

        // ── Title ──────────────────────────────────────────────────────────
        md.push_str(&format!("# {}\n\n", self.model_name));

        // ── Author ─────────────────────────────────────────────────────────
        if let Some(ref author) = self.author {
            md.push_str(&format!("*Author: {author}*\n\n"));
        }

        // ── Model Description ──────────────────────────────────────────────
        md.push_str("## Model Description\n\n");
        md.push_str(&self.model_description);
        md.push_str("\n\n");

        // ── Intended Uses ──────────────────────────────────────────────────
        if !self.intended_uses.is_empty() {
            md.push_str("## Intended Uses\n\n");
            for use_case in &self.intended_uses {
                md.push_str(&format!("- {use_case}\n"));
            }
            md.push('\n');
        }

        // ── Limitations ────────────────────────────────────────────────────
        if !self.limitations.is_empty() {
            md.push_str("## Limitations\n\n");
            for lim in &self.limitations {
                md.push_str(&format!("- {lim}\n"));
            }
            md.push('\n');
        }

        // ── Training Details ───────────────────────────────────────────────
        md.push_str("## Training Details\n\n");
        md.push_str(&format!(
            "- **Framework:** {}\n",
            self.training_info.framework
        ));
        if let Some(n) = self.training_info.num_parameters {
            md.push_str(&format!("- **Parameters:** {n}\n"));
        }
        if !self.training_info.training_data.is_empty() {
            md.push_str(&format!(
                "- **Training Data:** {}\n",
                self.training_info.training_data.join(", ")
            ));
        }
        if let Some(ref opt) = self.training_info.optimizer {
            md.push_str(&format!("- **Optimizer:** {opt}\n"));
        }
        if let Some(lr) = self.training_info.learning_rate {
            md.push_str(&format!("- **Learning Rate:** {lr}\n"));
        }
        if let Some(bs) = self.training_info.batch_size {
            md.push_str(&format!("- **Batch Size:** {bs}\n"));
        }
        if let Some(ep) = self.training_info.num_epochs {
            md.push_str(&format!("- **Epochs:** {ep}\n"));
        }
        if let Some(ref hw) = self.training_info.hardware {
            md.push_str(&format!("- **Hardware:** {hw}\n"));
        }
        md.push('\n');

        // ── Benchmarks ─────────────────────────────────────────────────────
        if !self.benchmarks.is_empty() {
            md.push_str("## Evaluation Results\n\n");
            md.push_str("| Task | Dataset | Metric | Value |\n");
            md.push_str("|------|---------|--------|-------|\n");
            for b in &self.benchmarks {
                md.push_str(&format!(
                    "| {} | {} | {} | {:.4} |\n",
                    b.task, b.dataset, b.metric, b.value
                ));
            }
            md.push('\n');
        }

        // ── Citation ───────────────────────────────────────────────────────
        if let Some(ref citation) = self.citation {
            md.push_str("## Citation\n\n");
            md.push_str("```bibtex\n");
            md.push_str(citation);
            md.push_str("\n```\n\n");
        }

        // ── Footer ─────────────────────────────────────────────────────────
        md.push_str("---\n");
        md.push_str("*Generated by [TrustformeRS](https://github.com/cool-japan/trustformers)*\n");

        md
    }

    /// Parse a model card from a markdown string.
    ///
    /// Reads the YAML front matter between the first pair of `---` delimiters
    /// and extracts the `## Model Description` section body.
    pub fn from_markdown(content: &str) -> Result<Self> {
        let mut metadata = ModelCardMetadata::default();
        let mut model_name = String::new();
        let mut model_description = String::new();
        let mut author: Option<String> = None;

        // Extract YAML front matter
        if content.starts_with("---") {
            let rest = &content[3..];
            if let Some(end) = rest.find("\n---") {
                let yaml_str = &rest[..end];
                // Parse just the fields we care about
                for line in yaml_str.lines() {
                    if let Some(val) = line.strip_prefix("license: ") {
                        metadata.license = val.trim().to_string();
                    } else if let Some(val) = line.strip_prefix("library_name: ") {
                        metadata.library_name = val.trim().to_string();
                    } else if let Some(val) = line.strip_prefix("model_type: ") {
                        metadata.model_type = Some(val.trim().to_string());
                    } else if let Some(val) = line.strip_prefix("pipeline_tag: ") {
                        metadata.pipeline_tag = Some(val.trim().to_string());
                    } else if line.trim_start().starts_with("- ") {
                        // handled implicitly; full YAML parse is below
                    }
                }

                // Use serde_yaml for a richer parse of the front matter
                let parsed: serde_yaml::Value =
                    serde_yaml::from_str(yaml_str).unwrap_or(serde_yaml::Value::Null);

                if let serde_yaml::Value::Mapping(ref map) = parsed {
                    if let Some(serde_yaml::Value::Sequence(seq)) = map.get("language") {
                        metadata.language =
                            seq.iter().filter_map(|x| x.as_str().map(String::from)).collect();
                    }
                    if let Some(serde_yaml::Value::Sequence(seq)) = map.get("tags") {
                        metadata.tags =
                            seq.iter().filter_map(|x| x.as_str().map(String::from)).collect();
                    }
                    if let Some(serde_yaml::Value::Sequence(seq)) = map.get("datasets") {
                        metadata.datasets =
                            seq.iter().filter_map(|x| x.as_str().map(String::from)).collect();
                    }
                    if let Some(serde_yaml::Value::Sequence(seq)) = map.get("metrics") {
                        metadata.metrics =
                            seq.iter().filter_map(|x| x.as_str().map(String::from)).collect();
                    }
                }
            }
        }

        // Extract title (first `# ` heading)
        for line in content.lines() {
            if let Some(name) = line.strip_prefix("# ") {
                model_name = name.trim().to_string();
                break;
            }
        }

        // Extract author
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("*Author: ") {
                author = Some(rest.trim_end_matches('*').to_string());
                break;
            }
        }

        // Extract ## Model Description section
        let mut in_description = false;
        for line in content.lines() {
            if line.starts_with("## Model Description") {
                in_description = true;
                continue;
            }
            if in_description {
                if line.starts_with("## ") {
                    break;
                }
                if !model_description.is_empty() || !line.is_empty() {
                    model_description.push_str(line);
                    model_description.push('\n');
                }
            }
        }
        let model_description = model_description.trim().to_string();

        if model_name.is_empty() {
            return Err(TrustformersError::InvalidInput {
                message: "Model card must contain a top-level heading (# Model Name)".to_string(),
                parameter: Some("model_name".to_string()),
                expected: Some("A line starting with '# '".to_string()),
                received: None,
                suggestion: Some("Add '# Your Model Name' to the markdown".to_string()),
            });
        }

        debug!(
            model_name = %model_name,
            "Parsed model card from markdown"
        );

        Ok(Self {
            metadata,
            model_name,
            model_description,
            intended_uses: Vec::new(),
            limitations: Vec::new(),
            training_info: TrainingInfo::default(),
            benchmarks: Vec::new(),
            citation: None,
            author,
        })
    }

    /// Save the model card to a file as markdown
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = self.to_markdown();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| TrustformersError::Io {
                message: format!("Cannot create parent directory: {e}"),
                path: Some(parent.display().to_string()),
                suggestion: None,
            })?;
        }
        std::fs::write(path, content).map_err(|e| TrustformersError::Io {
            message: format!("Cannot write model card: {e}"),
            path: Some(path.display().to_string()),
            suggestion: None,
        })?;
        Ok(())
    }

    /// Load a model card from a markdown file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| TrustformersError::Io {
            message: format!("Cannot read model card file: {e}"),
            path: Some(path.display().to_string()),
            suggestion: Some("Ensure the file exists and is readable".to_string()),
        })?;
        Self::from_markdown(&content)
    }

    /// Add a benchmark result (fluent mutating method)
    pub fn add_benchmark(&mut self, result: BenchmarkResult) -> &mut Self {
        self.benchmarks.push(result);
        self
    }

    /// Replace the training info (builder-style consuming method)
    pub fn with_training_info(mut self, info: TrainingInfo) -> Self {
        self.training_info = info;
        self
    }

    /// Validate the model card for completeness.
    ///
    /// Returns a list of warning strings for missing or recommended fields.
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.model_name.is_empty() {
            warnings.push("model_name is empty".to_string());
        }
        if self.model_description.is_empty() {
            warnings.push("model_description is empty — add a description".to_string());
        }
        if self.metadata.language.is_empty() {
            warnings.push("metadata.language is empty — specify supported languages".to_string());
        }
        if self.metadata.license.is_empty() {
            warnings.push("metadata.license is empty — specify a license".to_string());
        }
        if self.intended_uses.is_empty() {
            warnings.push("intended_uses is empty — document intended uses".to_string());
        }
        if self.limitations.is_empty() {
            warnings.push("limitations is empty — document model limitations".to_string());
        }
        if self.benchmarks.is_empty() {
            warnings.push("benchmarks is empty — add evaluation results".to_string());
        }
        if self.metadata.pipeline_tag.is_none() {
            warnings.push("metadata.pipeline_tag is not set".to_string());
        }
        if self.author.is_none() {
            warnings.push("author is not set".to_string());
        }

        warnings
    }
}

// ─── ModelCardGenerator ───────────────────────────────────────────────────────

/// Utility for auto-generating model cards from structured metadata
pub struct ModelCardGenerator;

impl ModelCardGenerator {
    /// Generate a model card from basic model info and training details
    pub fn generate(
        model_type: &str,
        model_name: &str,
        training_info: TrainingInfo,
        pipeline_tag: Option<&str>,
    ) -> ModelCard {
        let mut metadata = ModelCardMetadata {
            model_type: Some(model_type.to_string()),
            pipeline_tag: pipeline_tag.map(String::from),
            tags: vec![model_type.to_string()],
            ..Default::default()
        };

        if let Some(pt) = pipeline_tag {
            if !metadata.tags.contains(&pt.to_string()) {
                metadata.tags.push(pt.to_string());
            }
        }

        let description = format!(
            "This is a {model_type} model trained with TrustformeRS. \
             It was trained using the {} framework.",
            training_info.framework
        );

        ModelCard {
            metadata,
            model_name: model_name.to_string(),
            model_description: description,
            intended_uses: vec![format!("This model can be used for {model_type} tasks.")],
            limitations: vec![
                "Model performance may degrade on out-of-distribution data.".to_string(),
                "The model has not been evaluated for all use cases.".to_string(),
            ],
            training_info,
            benchmarks: Vec::new(),
            citation: None,
            author: None,
        }
    }

    /// Generate a model card with pre-populated benchmark results
    pub fn generate_with_benchmarks(
        model_type: &str,
        model_name: &str,
        training_info: TrainingInfo,
        benchmarks: Vec<BenchmarkResult>,
    ) -> ModelCard {
        let mut card = Self::generate(model_type, model_name, training_info, None);
        card.benchmarks = benchmarks;

        // Auto-populate metrics in metadata from benchmarks
        card.metadata.metrics = card
            .benchmarks
            .iter()
            .map(|b| b.metric.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        card
    }
}

// ─── ModelCard additional methods ─────────────────────────────────────────────

impl ModelCard {
    /// Render only the YAML front-matter block (between the `---` delimiters).
    pub fn to_yaml_frontmatter(&self) -> String {
        let mut yaml = String::from("---\n");
        if !self.metadata.language.is_empty() {
            yaml.push_str("language:\n");
            for lang in &self.metadata.language {
                yaml.push_str(&format!("- {lang}\n"));
            }
        }
        yaml.push_str(&format!("license: {}\n", self.metadata.license));
        yaml.push_str(&format!("library_name: {}\n", self.metadata.library_name));
        if !self.metadata.tags.is_empty() {
            yaml.push_str("tags:\n");
            for tag in &self.metadata.tags {
                yaml.push_str(&format!("- {tag}\n"));
            }
        }
        if !self.metadata.datasets.is_empty() {
            yaml.push_str("datasets:\n");
            for ds in &self.metadata.datasets {
                yaml.push_str(&format!("- {ds}\n"));
            }
        }
        if !self.metadata.metrics.is_empty() {
            yaml.push_str("metrics:\n");
            for m in &self.metadata.metrics {
                yaml.push_str(&format!("- {m}\n"));
            }
        }
        if let Some(ref mt) = self.metadata.model_type {
            yaml.push_str(&format!("model_type: {mt}\n"));
        }
        if let Some(ref pt) = self.metadata.pipeline_tag {
            yaml.push_str(&format!("pipeline_tag: {pt}\n"));
        }
        if let Some(n) = self.training_info.num_parameters {
            yaml.push_str(&format!("num_parameters: {n}\n"));
        }
        yaml.push_str("---\n");
        yaml
    }
}

// ─── ModelCardError ───────────────────────────────────────────────────────────

/// Errors that can occur when building or parsing a model card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelCardError {
    /// A required field was not provided.
    MissingField(String),
    /// A field value was invalid.
    InvalidField { field: String, reason: String },
    /// The markdown could not be parsed.
    ParseError(String),
}

impl std::fmt::Display for ModelCardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelCardError::MissingField(name) => write!(f, "Missing required field: {name}"),
            ModelCardError::InvalidField { field, reason } => {
                write!(f, "Invalid field '{field}': {reason}")
            },
            ModelCardError::ParseError(msg) => write!(f, "Parse error: {msg}"),
        }
    }
}

impl std::error::Error for ModelCardError {}

// ─── ModelCardBuilder ─────────────────────────────────────────────────────────

/// Builder for constructing a [`ModelCard`] with a fluent interface.
///
/// Every setter returns `Self` for chaining.  Call [`build`][ModelCardBuilder::build]
/// when all required fields are provided.
#[derive(Debug, Clone, Default)]
pub struct ModelCardBuilder {
    model_id: Option<String>,
    architecture: Option<String>,
    parameters: Option<u64>,
    languages: Vec<String>,
    license: Option<String>,
    datasets: Vec<String>,
    metrics: Vec<(String, f32)>,
    tags: Vec<String>,
    limitations: Option<String>,
    bias_risks: Option<String>,
    description: Option<String>,
    pipeline_tag: Option<String>,
    author: Option<String>,
}

impl ModelCardBuilder {
    /// Start a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model identifier (used as the model name).
    pub fn with_model_id(mut self, id: &str) -> Self {
        self.model_id = Some(id.to_string());
        self
    }

    /// Set the model architecture (e.g. "bert", "llama").
    pub fn with_architecture(mut self, arch: &str) -> Self {
        self.architecture = Some(arch.to_string());
        self
    }

    /// Set the approximate number of trainable parameters.
    pub fn with_parameters(mut self, params: u64) -> Self {
        self.parameters = Some(params);
        self
    }

    /// Add a supported language (BCP-47 code, e.g. "en").
    /// Multiple calls accumulate languages.
    pub fn with_language(mut self, lang: &str) -> Self {
        self.languages.push(lang.to_string());
        self
    }

    /// Set the SPDX license identifier (e.g. "apache-2.0").
    pub fn with_license(mut self, license: &str) -> Self {
        self.license = Some(license.to_string());
        self
    }

    /// Add a training dataset name.
    pub fn with_dataset(mut self, dataset: &str) -> Self {
        self.datasets.push(dataset.to_string());
        self
    }

    /// Add an evaluation metric name and its numeric value.
    pub fn with_metrics(mut self, name: &str, value: f32) -> Self {
        self.metrics.push((name.to_string(), value));
        self
    }

    /// Replace the tag list.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set the limitations section text.
    pub fn with_limitations(mut self, text: &str) -> Self {
        self.limitations = Some(text.to_string());
        self
    }

    /// Set the bias & risks section text.
    pub fn with_bias_risks(mut self, text: &str) -> Self {
        self.bias_risks = Some(text.to_string());
        self
    }

    /// Set a short description of the model.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Set the HuggingFace pipeline tag.
    pub fn with_pipeline_tag(mut self, tag: &str) -> Self {
        self.pipeline_tag = Some(tag.to_string());
        self
    }

    /// Set the author name.
    pub fn with_author(mut self, author: &str) -> Self {
        self.author = Some(author.to_string());
        self
    }

    /// Build the [`ModelCard`].
    ///
    /// Returns `Err` if required fields (`model_id`) are missing.
    pub fn build(self) -> std::result::Result<ModelCard, ModelCardError> {
        let model_id = self
            .model_id
            .ok_or_else(|| ModelCardError::MissingField("model_id".to_string()))?;

        if model_id.is_empty() {
            return Err(ModelCardError::InvalidField {
                field: "model_id".to_string(),
                reason: "must not be empty".to_string(),
            });
        }

        // Resolve metadata.
        let metadata = ModelCardMetadata {
            language: if self.languages.is_empty() {
                vec!["en".to_string()]
            } else {
                self.languages
            },
            license: self.license.unwrap_or_else(|| "apache-2.0".to_string()),
            library_name: "trustformers".to_string(),
            tags: {
                let mut tags = self.tags;
                if let Some(ref arch) = self.architecture {
                    if !tags.contains(arch) {
                        tags.push(arch.clone());
                    }
                }
                tags
            },
            datasets: self.datasets,
            metrics: self.metrics.iter().map(|(n, _)| n.clone()).collect(),
            model_type: self.architecture.clone(),
            pipeline_tag: self.pipeline_tag,
        };

        let description = self.description.unwrap_or_else(|| {
            self.architecture
                .as_deref()
                .map(|a| format!("A {a} model trained with TrustformeRS."))
                .unwrap_or_else(|| "A model trained with TrustformeRS.".to_string())
        });

        // Build limitations list.
        let mut limitations = Vec::new();
        if let Some(lim_text) = self.limitations {
            limitations.push(lim_text);
        }
        if let Some(bias_text) = self.bias_risks {
            limitations.push(format!("[Bias/Risks] {bias_text}"));
        }
        if limitations.is_empty() {
            limitations
                .push("Model performance may degrade on out-of-distribution data.".to_string());
        }

        // Training info.
        let training_info = TrainingInfo {
            num_parameters: self.parameters,
            ..Default::default()
        };

        // Add metrics as benchmarks.
        let benchmarks: Vec<BenchmarkResult> = self
            .metrics
            .iter()
            .map(|(name, value)| {
                let value = *value;
                BenchmarkResult::new("evaluation", "unknown", name.as_str(), value as f64)
            })
            .collect();

        Ok(ModelCard {
            metadata,
            model_name: model_id,
            model_description: description,
            intended_uses: Vec::new(),
            limitations,
            training_info,
            benchmarks,
            citation: None,
            author: self.author,
        })
    }
}

// ─── ModelCardTemplate ────────────────────────────────────────────────────────

/// Pre-defined card templates for common model types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCardTemplate {
    /// General-purpose language model.
    General,
    /// Classification model (sequence-level label).
    Classification,
    /// Text-generation / language-generation model.
    Generation,
    /// Multilingual model supporting several languages.
    Multilingual,
}

impl ModelCardTemplate {
    /// Apply the template, returning a pre-populated [`ModelCardBuilder`].
    pub fn apply(self, model_id: &str) -> ModelCardBuilder {
        match self {
            ModelCardTemplate::General => ModelCardBuilder::new()
                .with_model_id(model_id)
                .with_language("en")
                .with_license("apache-2.0")
                .with_tags(vec![
                    "trustformers".to_string(),
                    "transformer".to_string(),
                ])
                .with_limitations(
                    "Performance may vary on out-of-distribution inputs.",
                )
                .with_bias_risks(
                    "The model may reflect biases present in its training data.",
                ),

            ModelCardTemplate::Classification => ModelCardBuilder::new()
                .with_model_id(model_id)
                .with_language("en")
                .with_license("apache-2.0")
                .with_pipeline_tag("text-classification")
                .with_tags(vec![
                    "text-classification".to_string(),
                    "trustformers".to_string(),
                ])
                .with_limitations(
                    "Classification accuracy may degrade on out-of-distribution data.",
                )
                .with_bias_risks(
                    "Classifier may exhibit label bias if training data is imbalanced.",
                ),

            ModelCardTemplate::Generation => ModelCardBuilder::new()
                .with_model_id(model_id)
                .with_language("en")
                .with_license("apache-2.0")
                .with_pipeline_tag("text-generation")
                .with_tags(vec![
                    "text-generation".to_string(),
                    "causal-lm".to_string(),
                    "trustformers".to_string(),
                ])
                .with_limitations(
                    "Generated text may be factually incorrect or harmful.",
                )
                .with_bias_risks(
                    "The model may generate biased, offensive, or misleading content.",
                ),

            ModelCardTemplate::Multilingual => ModelCardBuilder::new()
                .with_model_id(model_id)
                .with_language("en")
                .with_language("fr")
                .with_language("de")
                .with_language("es")
                .with_language("zh")
                .with_license("apache-2.0")
                .with_tags(vec![
                    "multilingual".to_string(),
                    "trustformers".to_string(),
                ])
                .with_limitations(
                    "Performance varies across languages; low-resource languages may perform worse.",
                )
                .with_bias_risks(
                    "Multilingual models can exhibit cross-lingual bias.",
                ),
        }
    }

    /// Return a string label for this template.
    pub fn label(self) -> &'static str {
        match self {
            ModelCardTemplate::General => "general",
            ModelCardTemplate::Classification => "classification",
            ModelCardTemplate::Generation => "generation",
            ModelCardTemplate::Multilingual => "multilingual",
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir()
    }

    #[test]
    fn test_model_card_new() {
        let card = ModelCard::new("MyBERT", "A BERT-based classification model");
        assert_eq!(card.model_name, "MyBERT");
        assert!(!card.model_description.is_empty());
    }

    #[test]
    fn test_model_card_metadata_default() {
        let meta = ModelCardMetadata::default();
        assert_eq!(meta.license, "apache-2.0");
        assert_eq!(meta.library_name, "trustformers");
        assert!(!meta.language.is_empty());
    }

    #[test]
    fn test_to_markdown_contains_title() {
        let card = ModelCard::new("TestModel", "Test description");
        let md = card.to_markdown();
        assert!(md.contains("# TestModel"));
    }

    #[test]
    fn test_to_markdown_contains_yaml_front_matter() {
        let card = ModelCard::new("TestModel", "desc");
        let md = card.to_markdown();
        assert!(md.starts_with("---\n"));
        assert!(md.contains("license: apache-2.0"));
        assert!(md.contains("library_name: trustformers"));
    }

    #[test]
    fn test_to_markdown_with_benchmarks() {
        let mut card = ModelCard::new("BenchModel", "desc");
        card.add_benchmark(BenchmarkResult::new(
            "text-classification",
            "glue/sst2",
            "accuracy",
            0.9234,
        ));
        let md = card.to_markdown();
        assert!(md.contains("## Evaluation Results"));
        assert!(md.contains("0.9234"));
    }

    #[test]
    fn test_from_markdown_roundtrip() {
        let mut original = ModelCard::new("RoundtripModel", "A test model for roundtrip testing");
        original.author = Some("Test Author".to_string());
        original.metadata.model_type = Some("bert".to_string());
        original.metadata.pipeline_tag = Some("text-classification".to_string());

        let md = original.to_markdown();
        let parsed = ModelCard::from_markdown(&md).unwrap();

        assert_eq!(parsed.model_name, "RoundtripModel");
        assert!(parsed.model_description.contains("roundtrip testing"));
        assert_eq!(parsed.metadata.model_type, Some("bert".to_string()));
        assert_eq!(
            parsed.metadata.pipeline_tag,
            Some("text-classification".to_string())
        );
    }

    #[test]
    fn test_from_markdown_missing_title_error() {
        let md = "No title here\nSome content";
        assert!(ModelCard::from_markdown(md).is_err());
    }

    #[test]
    fn test_save_and_load() {
        let dir = temp_dir().join("trustformers_model_card_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("README.md");

        let card = ModelCard::new("SaveLoadModel", "Testing save/load functionality");
        card.save(&path).unwrap();

        let loaded = ModelCard::load(&path).unwrap();
        assert_eq!(loaded.model_name, "SaveLoadModel");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_add_benchmark_chaining() {
        let mut card = ModelCard::new("BenchModel", "desc");
        card.add_benchmark(BenchmarkResult::new("task1", "ds1", "f1", 0.85))
            .add_benchmark(BenchmarkResult::new("task2", "ds2", "acc", 0.92));
        assert_eq!(card.benchmarks.len(), 2);
    }

    #[test]
    fn test_with_training_info() {
        let info = TrainingInfo {
            num_parameters: Some(110_000_000),
            optimizer: Some("AdamW".to_string()),
            learning_rate: Some(2e-5),
            num_epochs: Some(3),
            hardware: Some("1x A100".to_string()),
            ..Default::default()
        };
        let card = ModelCard::new("TrainModel", "desc").with_training_info(info);
        assert_eq!(card.training_info.num_parameters, Some(110_000_000));
        assert_eq!(card.training_info.optimizer, Some("AdamW".to_string()));
    }

    #[test]
    fn test_validate_warns_on_empty_card() {
        let card = ModelCard::new("", "");
        let warnings = card.validate();
        assert!(!warnings.is_empty());
        // Should warn about model_name, description, uses, limitations, benchmarks
        assert!(warnings.iter().any(|w| w.contains("model_name")));
        assert!(warnings.iter().any(|w| w.contains("model_description")));
    }

    #[test]
    fn test_validate_clean_card() {
        let mut card = ModelCard::new("CleanModel", "A clean well-documented model");
        card.author = Some("Author".to_string());
        card.intended_uses = vec!["classification".to_string()];
        card.limitations = vec!["limited training data".to_string()];
        card.metadata.pipeline_tag = Some("text-classification".to_string());
        card.add_benchmark(BenchmarkResult::new("tc", "sst2", "acc", 0.9));

        let warnings = card.validate();
        // model_name, description, language, license should all be fine
        assert!(!warnings.iter().any(|w| w.contains("model_name")));
        assert!(!warnings.iter().any(|w| w.contains("model_description")));
        assert!(!warnings.iter().any(|w| w.contains("pipeline_tag")));
    }

    #[test]
    fn test_generator_generate() {
        let info = TrainingInfo {
            num_epochs: Some(5),
            ..Default::default()
        };
        let card = ModelCardGenerator::generate("gpt2", "MyGPT", info, Some("text-generation"));
        assert_eq!(card.model_name, "MyGPT");
        assert_eq!(card.metadata.model_type, Some("gpt2".to_string()));
        assert_eq!(
            card.metadata.pipeline_tag,
            Some("text-generation".to_string())
        );
        assert!(!card.intended_uses.is_empty());
        assert!(!card.limitations.is_empty());
    }

    #[test]
    fn test_generator_with_benchmarks() {
        let benchmarks = vec![
            BenchmarkResult::new("lm", "wikitext", "perplexity", 15.3),
            BenchmarkResult::new("lm", "ptb", "perplexity", 22.1),
        ];
        let card = ModelCardGenerator::generate_with_benchmarks(
            "gpt2",
            "BenchGPT",
            TrainingInfo::default(),
            benchmarks,
        );
        assert_eq!(card.benchmarks.len(), 2);
        assert!(!card.metadata.metrics.is_empty());
    }

    #[test]
    fn test_markdown_includes_training_details() {
        let info = TrainingInfo {
            num_parameters: Some(340_000_000),
            optimizer: Some("Adam".to_string()),
            learning_rate: Some(1e-4),
            batch_size: Some(32),
            num_epochs: Some(10),
            hardware: Some("8x V100".to_string()),
            ..Default::default()
        };
        let card = ModelCard::new("DetailedModel", "desc").with_training_info(info);
        let md = card.to_markdown();
        assert!(md.contains("340000000"));
        assert!(md.contains("Adam"));
        assert!(md.contains("0.0001"));
        assert!(md.contains("8x V100"));
    }

    #[test]
    fn test_benchmark_result_new() {
        let b = BenchmarkResult::new("ner", "conll2003", "f1", 0.93);
        assert_eq!(b.task, "ner");
        assert_eq!(b.dataset, "conll2003");
        assert_eq!(b.metric, "f1");
        assert!((b.value - 0.93).abs() < 1e-9);
    }

    // ── ModelCardBuilder tests ────────────────────────────────────────────────

    #[test]
    fn test_builder_minimal() {
        let card = ModelCardBuilder::new().with_model_id("my-model").build().unwrap();
        assert_eq!(card.model_name, "my-model");
        assert_eq!(card.metadata.license, "apache-2.0");
        assert!(!card.metadata.language.is_empty());
    }

    #[test]
    fn test_builder_missing_model_id() {
        let err = ModelCardBuilder::new().build().unwrap_err();
        assert!(matches!(err, ModelCardError::MissingField(_)));
    }

    #[test]
    fn test_builder_empty_model_id() {
        let err = ModelCardBuilder::new().with_model_id("").build().unwrap_err();
        assert!(matches!(err, ModelCardError::InvalidField { .. }));
    }

    #[test]
    fn test_builder_full() {
        let card = ModelCardBuilder::new()
            .with_model_id("my-bert")
            .with_architecture("bert")
            .with_parameters(110_000_000)
            .with_language("en")
            .with_language("de")
            .with_license("mit")
            .with_dataset("glue")
            .with_metrics("accuracy", 0.94)
            .with_metrics("f1", 0.91)
            .with_tags(vec!["nlp".to_string(), "bert".to_string()])
            .with_limitations("Only handles English and German well.")
            .with_bias_risks("May reflect biases in training data.")
            .with_description("A BERT model fine-tuned for NER.")
            .with_author("COOLJAPAN")
            .build()
            .unwrap();

        assert_eq!(card.model_name, "my-bert");
        assert_eq!(card.training_info.num_parameters, Some(110_000_000));
        assert!(card.metadata.language.contains(&"en".to_string()));
        assert!(card.metadata.language.contains(&"de".to_string()));
        assert_eq!(card.metadata.license, "mit");
        assert!(card.metadata.datasets.contains(&"glue".to_string()));
        assert_eq!(card.benchmarks.len(), 2);
        assert!(card.author.as_deref() == Some("COOLJAPAN"));
        assert!(!card.limitations.is_empty());
    }

    #[test]
    fn test_builder_arch_tag_added_automatically() {
        let card = ModelCardBuilder::new()
            .with_model_id("llama-model")
            .with_architecture("llama")
            .build()
            .unwrap();
        assert!(card.metadata.tags.contains(&"llama".to_string()));
        assert_eq!(card.metadata.model_type, Some("llama".to_string()));
    }

    #[test]
    fn test_builder_default_language_fallback() {
        let card = ModelCardBuilder::new().with_model_id("no-lang-model").build().unwrap();
        // Should default to ["en"].
        assert_eq!(card.metadata.language, vec!["en"]);
    }

    #[test]
    fn test_builder_multiple_languages() {
        let card = ModelCardBuilder::new()
            .with_model_id("multi-lang")
            .with_language("en")
            .with_language("fr")
            .with_language("es")
            .build()
            .unwrap();
        assert_eq!(card.metadata.language.len(), 3);
        assert!(card.metadata.language.contains(&"fr".to_string()));
    }

    #[test]
    fn test_builder_metrics_populate_benchmarks_and_metadata() {
        let card = ModelCardBuilder::new()
            .with_model_id("bench-model")
            .with_metrics("perplexity", 12.5)
            .with_metrics("bleu", 0.45)
            .build()
            .unwrap();
        assert_eq!(card.benchmarks.len(), 2);
        assert!(card.metadata.metrics.contains(&"perplexity".to_string()));
        assert!(card.metadata.metrics.contains(&"bleu".to_string()));
    }

    // ── to_yaml_frontmatter tests ─────────────────────────────────────────────

    #[test]
    fn test_to_yaml_frontmatter_structure() {
        let card = ModelCard::new("TestModel", "desc");
        let yaml = card.to_yaml_frontmatter();
        assert!(yaml.starts_with("---\n"));
        assert!(yaml.ends_with("---\n"));
        assert!(yaml.contains("license: apache-2.0"));
        assert!(yaml.contains("library_name: trustformers"));
    }

    #[test]
    fn test_to_yaml_frontmatter_with_model_type() {
        let mut card = ModelCard::new("BertModel", "desc");
        card.metadata.model_type = Some("bert".to_string());
        card.metadata.pipeline_tag = Some("text-classification".to_string());
        let yaml = card.to_yaml_frontmatter();
        assert!(yaml.contains("model_type: bert"));
        assert!(yaml.contains("pipeline_tag: text-classification"));
    }

    #[test]
    fn test_to_yaml_frontmatter_languages() {
        let mut card = ModelCard::new("MultiLang", "desc");
        card.metadata.language = vec!["en".to_string(), "fr".to_string()];
        let yaml = card.to_yaml_frontmatter();
        assert!(yaml.contains("language:"));
        assert!(yaml.contains("- en"));
        assert!(yaml.contains("- fr"));
    }

    #[test]
    fn test_to_yaml_frontmatter_num_parameters() {
        let mut card = ModelCard::new("BigModel", "desc");
        card.training_info.num_parameters = Some(7_000_000_000);
        let yaml = card.to_yaml_frontmatter();
        assert!(yaml.contains("num_parameters: 7000000000"));
    }

    // ── ModelCardTemplate tests ───────────────────────────────────────────────

    #[test]
    fn test_template_general() {
        let card = ModelCardTemplate::General.apply("my-general-model").build().unwrap();
        assert_eq!(card.model_name, "my-general-model");
        assert!(!card.limitations.is_empty());
    }

    #[test]
    fn test_template_classification() {
        let card = ModelCardTemplate::Classification.apply("my-classifier").build().unwrap();
        assert_eq!(
            card.metadata.pipeline_tag,
            Some("text-classification".to_string())
        );
        assert!(card.metadata.tags.contains(&"text-classification".to_string()));
    }

    #[test]
    fn test_template_generation() {
        let card = ModelCardTemplate::Generation.apply("my-gpt").build().unwrap();
        assert_eq!(
            card.metadata.pipeline_tag,
            Some("text-generation".to_string())
        );
        assert!(card.metadata.tags.contains(&"causal-lm".to_string()));
    }

    #[test]
    fn test_template_multilingual() {
        let card = ModelCardTemplate::Multilingual.apply("my-multi").build().unwrap();
        assert!(card.metadata.language.len() >= 5);
        assert!(card.metadata.language.contains(&"zh".to_string()));
    }

    #[test]
    fn test_template_labels() {
        assert_eq!(ModelCardTemplate::General.label(), "general");
        assert_eq!(ModelCardTemplate::Classification.label(), "classification");
        assert_eq!(ModelCardTemplate::Generation.label(), "generation");
        assert_eq!(ModelCardTemplate::Multilingual.label(), "multilingual");
    }

    #[test]
    fn test_template_further_customisation() {
        let card = ModelCardTemplate::Generation
            .apply("base-model")
            .with_parameters(1_000_000_000)
            .with_dataset("c4")
            .with_author("COOLJAPAN")
            .build()
            .unwrap();
        assert_eq!(card.training_info.num_parameters, Some(1_000_000_000));
        assert!(card.metadata.datasets.contains(&"c4".to_string()));
        assert_eq!(card.author.as_deref(), Some("COOLJAPAN"));
    }

    #[test]
    fn test_from_markdown_roundtrip_builder() {
        let original = ModelCardBuilder::new()
            .with_model_id("RoundtripBuilt")
            .with_architecture("roberta")
            .with_language("en")
            .with_description("A RoBERTa model built with the builder.")
            .build()
            .unwrap();

        let md = original.to_markdown();
        let parsed = ModelCard::from_markdown(&md).unwrap();
        assert_eq!(parsed.model_name, "RoundtripBuilt");
        assert!(parsed.model_description.contains("RoBERTa"));
    }

    // ── ModelCardError tests ──────────────────────────────────────────────────

    #[test]
    fn test_model_card_error_display_missing_field() {
        let err = ModelCardError::MissingField("model_id".to_string());
        assert!(err.to_string().contains("model_id"));
    }

    #[test]
    fn test_model_card_error_display_invalid_field() {
        let err = ModelCardError::InvalidField {
            field: "license".to_string(),
            reason: "unknown identifier".to_string(),
        };
        assert!(err.to_string().contains("license"));
        assert!(err.to_string().contains("unknown identifier"));
    }

    #[test]
    fn test_model_card_error_display_parse_error() {
        let err = ModelCardError::ParseError("unexpected end of YAML".to_string());
        assert!(err.to_string().contains("Parse error"));
        assert!(err.to_string().contains("unexpected end"));
    }
}
