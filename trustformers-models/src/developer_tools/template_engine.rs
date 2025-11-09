//! Template Engine
//!
//! Template-based code generation system for model implementations.

use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Template engine for code generation
pub struct TemplateEngine {
    templates: HashMap<String, Template>,
    variables: HashMap<String, String>,
}

/// Template definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Template {
    /// Template name
    pub name: String,
    /// Template content with placeholders
    pub content: String,
    /// Required variables
    pub required_variables: Vec<String>,
    /// Optional variables with defaults
    pub optional_variables: HashMap<String, String>,
}

/// Template context for rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateContext {
    /// Variables to substitute in templates
    pub variables: HashMap<String, String>,
    /// Conditional blocks
    pub conditions: HashMap<String, bool>,
    /// Loop data
    pub loops: HashMap<String, Vec<HashMap<String, String>>>,
}

impl TemplateEngine {
    /// Create a new template engine
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            variables: HashMap::new(),
        }
    }

    /// Add a template to the engine
    pub fn add_template(&mut self, template: Template) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Set a global variable
    pub fn set_variable(&mut self, key: String, value: String) {
        self.variables.insert(key, value);
    }

    /// Render a template with context
    pub fn render(&self, template_name: &str, context: &TemplateContext) -> Result<String> {
        let template = self
            .templates
            .get(template_name)
            .ok_or_else(|| Error::msg(format!("Template '{}' not found", template_name)))?;

        // Validate required variables
        for required_var in &template.required_variables {
            if !context.variables.contains_key(required_var)
                && !self.variables.contains_key(required_var)
            {
                return Err(Error::msg(format!(
                    "Required variable '{}' not provided",
                    required_var
                )));
            }
        }

        let mut rendered = template.content.clone();

        // Substitute variables
        for (key, value) in &context.variables {
            rendered = rendered.replace(&format!("{{{{{}}}}}", key), value);
        }

        // Substitute global variables
        for (key, value) in &self.variables {
            rendered = rendered.replace(&format!("{{{{{}}}}}", key), value);
        }

        // Handle conditional blocks
        rendered = self.render_conditionals(rendered, &context.conditions)?;

        // Handle loops
        rendered = self.render_loops(rendered, &context.loops)?;

        Ok(rendered)
    }

    /// Render conditional blocks
    fn render_conditionals(
        &self,
        mut content: String,
        conditions: &HashMap<String, bool>,
    ) -> Result<String> {
        // Simple conditional rendering: {{#if condition}}...{{/if}}
        for (condition, enabled) in conditions {
            let start_tag = format!("{{{{#if {}}}}}", condition);
            let end_tag = "{{/if}}";

            while let Some(start_pos) = content.find(&start_tag) {
                if let Some(end_pos) = content[start_pos..].find(end_tag) {
                    let actual_end_pos = start_pos + end_pos;
                    let block_content = &content[start_pos + start_tag.len()..actual_end_pos];

                    let replacement =
                        if *enabled { block_content.to_string() } else { String::new() };

                    content.replace_range(start_pos..actual_end_pos + end_tag.len(), &replacement);
                } else {
                    return Err(Error::msg("Unclosed conditional block"));
                }
            }
        }

        Ok(content)
    }

    /// Render loop blocks
    fn render_loops(
        &self,
        mut content: String,
        loops: &HashMap<String, Vec<HashMap<String, String>>>,
    ) -> Result<String> {
        // Simple loop rendering: {{#each items}}...{{/each}}
        for (loop_name, items) in loops {
            let start_tag = format!("{{{{#each {}}}}}", loop_name);
            let end_tag = "{{/each}}";

            while let Some(start_pos) = content.find(&start_tag) {
                if let Some(end_pos) = content[start_pos..].find(end_tag) {
                    let actual_end_pos = start_pos + end_pos;
                    let template_content = &content[start_pos + start_tag.len()..actual_end_pos];

                    let mut rendered_items = Vec::new();
                    for item in items {
                        let mut item_content = template_content.to_string();
                        for (key, value) in item {
                            item_content = item_content.replace(&format!("{{{{{}}}}}", key), value);
                        }
                        rendered_items.push(item_content);
                    }

                    let replacement = rendered_items.join("");
                    content.replace_range(start_pos..actual_end_pos + end_tag.len(), &replacement);
                } else {
                    return Err(Error::msg("Unclosed loop block"));
                }
            }
        }

        Ok(content)
    }

    /// Load templates from directory
    pub fn load_templates_from_dir(&mut self, dir: &Path) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("template") {
                let content = std::fs::read_to_string(&path)?;
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| Error::msg("Invalid template filename"))?
                    .to_string();

                let template = Template {
                    name: name.clone(),
                    content: content.clone(),
                    required_variables: self.extract_required_variables(&content),
                    optional_variables: HashMap::new(),
                };

                self.add_template(template);
            }
        }

        Ok(())
    }

    /// Extract required variables from template content
    fn extract_required_variables(&self, content: &str) -> Vec<String> {
        let mut variables = Vec::new();
        let mut chars = content.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' && chars.peek() == Some(&'{') {
                chars.next(); // consume second {
                let mut var_name = String::new();

                while let Some(ch) = chars.next() {
                    if ch == '}' && chars.peek() == Some(&'}') {
                        chars.next(); // consume second }
                        if !var_name.is_empty()
                            && !var_name.starts_with('#')
                            && !var_name.starts_with('/')
                        {
                            variables.push(var_name);
                        }
                        break;
                    }
                    var_name.push(ch);
                }
            }
        }

        variables.sort();
        variables.dedup();
        variables
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Predefined templates for common patterns
pub struct ModelTemplates;

impl ModelTemplates {
    /// Get transformer model template
    pub fn transformer_model() -> Template {
        Template {
            name: "transformer_model".to_string(),
            content: r#"//! {{model_name}} Model Implementation
//!
//! {{description}}

use super::config::{{model_name}}Config;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;
use trustformers_core::layers::{Linear, LayerNorm, Embedding};
{{#if use_attention}}
use trustformers_core::layers::attention::MultiHeadAttention;
{{/if}}

/// {{model_name}} model structure
#[derive(Debug, Clone)]
pub struct {{model_name}}Model {
    config: {{model_name}}Config,
    {{#if use_embeddings}}
    embeddings: Embedding,
    {{/if}}
    {{#each layers}}
    {{name}}: {{layer_type}},
    {{/each}}
    {{#if use_output_projection}}
    output_projection: Linear,
    {{/if}}
}

impl {{model_name}}Model {
    /// Create a new {{model_name}} model
    pub fn new(config: {{model_name}}Config) -> Result<Self> {
        {{#if use_embeddings}}
        let embeddings = Embedding::new(config.vocab_size, config.hidden_size)?;
        {{/if}}

        {{#each layers}}
        let {{name}} = {{layer_type}}::new({{parameters}})?;
        {{/each}}

        {{#if use_output_projection}}
        let output_projection = Linear::new(config.hidden_size, config.vocab_size, true)?;
        {{/if}}

        Ok(Self {
            config,
            {{#if use_embeddings}}
            embeddings,
            {{/if}}
            {{#each layers}}
            {{name}},
            {{/each}}
            {{#if use_output_projection}}
            output_projection,
            {{/if}}
        })
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        {{#if use_embeddings}}
        let mut hidden_states = self.embeddings.forward(input)?;
        {{else}}
        let mut hidden_states = input.clone();
        {{/if}}

        {{#each layers}}
        hidden_states = self.{{name}}.forward(&hidden_states)?;
        {{/each}}

        {{#if use_output_projection}}
        let logits = self.output_projection.forward(&hidden_states)?;
        Ok(logits)
        {{else}}
        Ok(hidden_states)
        {{/if}}
    }

    /// Get model configuration
    pub fn config(&self) -> &{{model_name}}Config {
        &self.config
    }
}
"#
            .to_string(),
            required_variables: vec!["model_name".to_string(), "description".to_string()],
            optional_variables: HashMap::new(),
        }
    }

    /// Get configuration template
    pub fn model_config() -> Template {
        Template {
            name: "model_config".to_string(),
            content: r#"//! {{model_name}} Configuration
//!
//! Configuration parameters for {{model_name}} models.

use serde::{Deserialize, Serialize};

/// Configuration for {{model_name}} models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct {{model_name}}Config {
    {{#each parameters}}
    /// {{description}}
    pub {{name}}: {{param_type}},
    {{/each}}
}

impl Default for {{model_name}}Config {
    fn default() -> Self {
        Self {
            {{#each parameters}}
            {{name}}: {{default_value}},
            {{/each}}
        }
    }
}

impl {{model_name}}Config {
    {{#each presets}}
    /// {{description}}
    pub fn {{name}}() -> Self {
        Self {
            {{#each parameters}}
            {{name}}: {{value}},
            {{/each}}
        }
    }
    {{/each}}

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        {{#each validations}}
        if {{condition}} {
            return Err("{{error_message}}".to_string());
        }
        {{/each}}

        Ok(())
    }
}
"#
            .to_string(),
            required_variables: vec!["model_name".to_string()],
            optional_variables: HashMap::new(),
        }
    }

    /// Get test template
    pub fn model_tests() -> Template {
        Template {
            name: "model_tests".to_string(),
            content: r#"//! {{model_name}} Tests
//!
//! Comprehensive test suite for {{model_name}} model implementation.

use super::{{{model_name}}Config, {{model_name}}Model}};
use trustformers_core::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_{{model_name_lower}}_creation() {
    let config = {{model_name}}Config::default();
    let model = {{model_name}}Model::new(config).expect("Failed to create model");

    // Model should be created successfully
    assert_eq!(model.config().vocab_size, {{default_vocab_size}});
}

#[test]
fn test_{{model_name_lower}}_forward_pass() {
    let config = {{model_name}}Config::default();
    let model = {{model_name}}Model::new(config.clone()).expect("Failed to create model");

    let batch_size = 2;
    let seq_length = 10;
    let input = Tensor::zeros(&[batch_size, seq_length]);

    let output = model.forward(&input).expect("Forward pass failed");

    // Verify output shape
    match &output {
        Tensor::F32(arr) => {
            assert_eq!(arr.shape()[0], batch_size);
            assert_eq!(arr.shape()[1], seq_length);
            {{#if has_vocab_output}}
            assert_eq!(arr.shape()[2], config.vocab_size);
            {{/if}}
        }
        _ => panic!("Expected F32 tensor"),
    }
}

#[test]
fn test_{{model_name_lower}}_numerical_stability() {
    let config = {{model_name}}Config::default();
    let model = {{model_name}}Model::new(config).expect("Failed to create model");

    let input = Tensor::randn(&[4, 16]);
    let output = model.forward(&input).expect("Forward pass failed");

    // Check for NaN or infinite values
    match &output {
        Tensor::F32(arr) => {
            for &val in arr.iter() {
                assert!(val.is_finite(), "Output contains non-finite values: {}", val);
            }
        }
        _ => panic!("Expected F32 tensor"),
    }
}

{{#each additional_tests}}
#[test]
fn test_{{test_name}}() {
    {{test_body}}
}
{{/each}}
"#
            .to_string(),
            required_variables: vec![
                "model_name".to_string(),
                "model_name_lower".to_string(),
                "default_vocab_size".to_string(),
            ],
            optional_variables: HashMap::new(),
        }
    }

    /// Get module template
    pub fn model_module() -> Template {
        Template {
            name: "model_module".to_string(),
            content: r#"//! {{model_name}} Model Implementation
//!
//! {{description}}
//!
//! ## Features
//!
//! {{#each features}}
//! - {{.}}
//! {{/each}}
//!
//! ## Usage
//!
//! ```rust
//! use trustformers_models::{{model_name_lower}}::{{{model_name}}Config, {{model_name}}Model}};
//!
//! // Create model with default configuration
//! let config = {{model_name}}Config::default();
//! let model = {{model_name}}Model::new(config)?;
//!
//! // Forward pass
//! let input = Tensor::zeros(&[1, 512]);
//! let output = model.forward(&input)?;
//! ```

pub mod config;
pub mod model;
{{#if has_tasks}}
pub mod tasks;
{{/if}}
{{#if has_tests}}
pub mod tests;
{{/if}}

pub use config::{{model_name}}Config;
pub use model::{{{model_name}}Model, {{model_name}}Output};
{{#if has_tasks}}
{{#each task_heads}}
pub use tasks::{{model_name}}For{{.}};
{{/each}}
{{/if}}
"#
            .to_string(),
            required_variables: vec![
                "model_name".to_string(),
                "model_name_lower".to_string(),
                "description".to_string(),
            ],
            optional_variables: HashMap::new(),
        }
    }
}

/// Template utilities
pub struct TemplateUtils;

impl TemplateUtils {
    /// Create context for transformer model
    pub fn transformer_context(
        model_name: &str,
        description: &str,
        vocab_size: usize,
        hidden_size: usize,
    ) -> TemplateContext {
        let mut variables = HashMap::new();
        variables.insert("model_name".to_string(), model_name.to_string());
        variables.insert("description".to_string(), description.to_string());
        variables.insert("vocab_size".to_string(), vocab_size.to_string());
        variables.insert("hidden_size".to_string(), hidden_size.to_string());
        variables.insert("model_name_lower".to_string(), model_name.to_lowercase());

        let mut conditions = HashMap::new();
        conditions.insert("use_embeddings".to_string(), true);
        conditions.insert("use_attention".to_string(), true);
        conditions.insert("use_output_projection".to_string(), true);
        conditions.insert("has_vocab_output".to_string(), true);

        TemplateContext {
            variables,
            conditions,
            loops: HashMap::new(),
        }
    }

    /// Create test context
    pub fn test_context(model_name: &str, vocab_size: usize) -> TemplateContext {
        let mut variables = HashMap::new();
        variables.insert("model_name".to_string(), model_name.to_string());
        variables.insert("model_name_lower".to_string(), model_name.to_lowercase());
        variables.insert("default_vocab_size".to_string(), vocab_size.to_string());

        let mut conditions = HashMap::new();
        conditions.insert("has_vocab_output".to_string(), true);

        TemplateContext {
            variables,
            conditions,
            loops: HashMap::new(),
        }
    }
}
