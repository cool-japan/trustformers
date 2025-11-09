use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};

/// Placeholder types for advanced template processing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PlaceholderType {
    /// String placeholder
    String,
    /// Integer placeholder
    Integer,
    /// Float placeholder
    Float,
    /// Boolean placeholder
    Boolean,
    /// List of strings
    StringList,
    /// Optional placeholder (can be empty)
    Optional(Box<PlaceholderType>),
}

/// Configuration for a placeholder token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaceholderToken {
    /// Name of the placeholder
    pub name: String,
    /// Type of the placeholder
    pub placeholder_type: PlaceholderType,
    /// Default value if not provided
    pub default_value: Option<String>,
    /// Whether this placeholder is required
    pub required: bool,
    /// Validation pattern (regex)
    pub validation_pattern: Option<String>,
    /// Description for documentation
    pub description: Option<String>,
    /// Transformation function name
    pub transformation: Option<String>,
}

impl PlaceholderToken {
    pub fn new(name: String, placeholder_type: PlaceholderType) -> Self {
        Self {
            name,
            placeholder_type,
            default_value: None,
            required: true,
            validation_pattern: None,
            description: None,
            transformation: None,
        }
    }

    pub fn with_default(mut self, default: String) -> Self {
        self.default_value = Some(default);
        self.required = false;
        self
    }

    pub fn with_validation(mut self, pattern: String) -> Self {
        self.validation_pattern = Some(pattern);
        self
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_transformation(mut self, transformation: String) -> Self {
        self.transformation = Some(transformation);
        self
    }

    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }
}

/// Placeholder processing result
#[derive(Debug, Clone)]
pub struct PlaceholderValue {
    pub raw_value: String,
    pub processed_value: String,
    pub placeholder_type: PlaceholderType,
}

/// Advanced placeholder processor
pub struct PlaceholderProcessor {
    /// Available transformations
    transformations: HashMap<String, Box<dyn Fn(&str) -> String>>,
}

impl std::fmt::Debug for PlaceholderProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlaceholderProcessor")
            .field(
                "transformations",
                &format!("[{} transformations]", self.transformations.len()),
            )
            .finish()
    }
}

impl PlaceholderProcessor {
    pub fn new() -> Self {
        let mut transformations: HashMap<String, Box<dyn Fn(&str) -> String>> = HashMap::new();

        // Built-in transformations
        transformations.insert(
            "uppercase".to_string(),
            Box::new(|s: &str| s.to_uppercase()),
        );
        transformations.insert(
            "lowercase".to_string(),
            Box::new(|s: &str| s.to_lowercase()),
        );
        transformations.insert("trim".to_string(), Box::new(|s: &str| s.trim().to_string()));
        transformations.insert(
            "capitalize".to_string(),
            Box::new(|s: &str| {
                s.split_whitespace()
                    .map(|word| {
                        let mut chars = word.chars();
                        match chars.next() {
                            None => String::new(),
                            Some(first) => {
                                first.to_uppercase().collect::<String>()
                                    + &chars.as_str().to_lowercase()
                            },
                        }
                    })
                    .collect::<Vec<String>>()
                    .join(" ")
            }),
        );

        Self { transformations }
    }

    pub fn add_transformation(&mut self, name: String, func: fn(&str) -> String) {
        self.transformations.insert(name, Box::new(func));
    }

    /// Apply a transformation to a string
    fn apply_transformation(&self, transformation: &str, input: &str) -> String {
        if let Some(func) = self.transformations.get(transformation) {
            func(input)
        } else {
            input.to_string() // Unknown transformation, return as-is
        }
    }

    pub fn validate_value(&self, placeholder: &PlaceholderToken, value: &str) -> Result<()> {
        // Type validation
        match &placeholder.placeholder_type {
            PlaceholderType::Integer => {
                value.parse::<i64>().map_err(|_| {
                    TrustformersError::other(format!("Invalid integer value: {}", value))
                })?;
            },
            PlaceholderType::Float => {
                value.parse::<f64>().map_err(|_| {
                    TrustformersError::other(format!("Invalid float value: {}", value))
                })?;
            },
            PlaceholderType::Boolean => {
                value.parse::<bool>().map_err(|_| {
                    TrustformersError::other(format!("Invalid boolean value: {}", value))
                })?;
            },
            PlaceholderType::StringList => {
                // Expect comma-separated values
                if value.trim().is_empty() {
                    return Err(TrustformersError::other(
                        "String list cannot be empty".to_string(),
                    ));
                }
            },
            PlaceholderType::Optional(inner_type) => {
                if !value.is_empty() {
                    let temp_placeholder = PlaceholderToken {
                        name: placeholder.name.clone(),
                        placeholder_type: (**inner_type).clone(),
                        default_value: None,
                        required: false,
                        validation_pattern: placeholder.validation_pattern.clone(),
                        description: None,
                        transformation: None,
                    };
                    self.validate_value(&temp_placeholder, value)?;
                }
            },
            PlaceholderType::String => {
                // String is always valid
            },
        }

        // Pattern validation
        if let Some(pattern) = &placeholder.validation_pattern {
            // In a real implementation, you'd use the `regex` crate
            // For now, we'll do simple pattern matching
            if pattern == "email" && !value.contains('@') {
                return Err(TrustformersError::other("Invalid email format".to_string()));
            }
            if pattern == "url" && !value.starts_with("http") {
                return Err(TrustformersError::other("Invalid URL format".to_string()));
            }
        }

        Ok(())
    }

    pub fn process_value(
        &self,
        placeholder: &PlaceholderToken,
        value: &str,
    ) -> Result<PlaceholderValue> {
        self.validate_value(placeholder, value)?;

        let mut processed_value = value.to_string();

        // Apply transformation if specified
        if let Some(transformation) = &placeholder.transformation {
            if self.transformations.contains_key(transformation) {
                processed_value = self.apply_transformation(transformation, &processed_value);
            }
        }

        // Type-specific processing
        match &placeholder.placeholder_type {
            PlaceholderType::StringList => {
                // Format as a JSON-like array for templates
                let items: Vec<&str> = processed_value.split(',').map(|s| s.trim()).collect();
                processed_value = format!("[{}]", items.join(", "));
            },
            PlaceholderType::Boolean => {
                // Normalize boolean representation
                let bool_val = processed_value.parse::<bool>().unwrap_or(false);
                processed_value = bool_val.to_string();
            },
            _ => {},
        }

        Ok(PlaceholderValue {
            raw_value: value.to_string(),
            processed_value,
            placeholder_type: placeholder.placeholder_type.clone(),
        })
    }
}

impl Default for PlaceholderProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for special token handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokenConfig {
    /// Static special tokens that are always present
    pub static_tokens: HashMap<String, u32>,
    /// Dynamic special tokens that can be added at runtime
    pub dynamic_tokens: HashMap<String, u32>,
    /// Templates for generating special tokens
    pub templates: HashMap<String, String>,
    /// Advanced templates with placeholder tokens
    pub advanced_templates: HashMap<String, AdvancedTemplate>,
    /// Maximum number of dynamic tokens allowed
    pub max_dynamic_tokens: usize,
    /// Next available token ID for dynamic tokens
    pub next_dynamic_id: u32,
}

/// Advanced template with placeholder token support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedTemplate {
    /// Template content with placeholders
    pub content: String,
    /// Placeholder token definitions
    pub placeholders: HashMap<String, PlaceholderToken>,
    /// Template description
    pub description: Option<String>,
    /// Template category
    pub category: Option<String>,
}

impl Default for SpecialTokenConfig {
    fn default() -> Self {
        let mut static_tokens = HashMap::new();
        static_tokens.insert("[PAD]".to_string(), 0);
        static_tokens.insert("[UNK]".to_string(), 1);
        static_tokens.insert("[CLS]".to_string(), 2);
        static_tokens.insert("[SEP]".to_string(), 3);
        static_tokens.insert("[MASK]".to_string(), 4);

        let mut templates = HashMap::new();
        templates.insert(
            "user_message".to_string(),
            "<|user|>{content}<|end|>".to_string(),
        );
        templates.insert(
            "assistant_message".to_string(),
            "<|assistant|>{content}<|end|>".to_string(),
        );
        templates.insert(
            "system_message".to_string(),
            "<|system|>{content}<|end|>".to_string(),
        );
        templates.insert(
            "function_call".to_string(),
            "<|function|>{name}({args})<|end|>".to_string(),
        );
        templates.insert(
            "code_block".to_string(),
            "<|code|>{language}\n{code}\n<|end|>".to_string(),
        );

        let mut advanced_templates = HashMap::new();

        // Advanced user message template
        let mut user_placeholders = HashMap::new();
        user_placeholders.insert(
            "content".to_string(),
            PlaceholderToken::new("content".to_string(), PlaceholderType::String),
        );
        user_placeholders.insert(
            "username".to_string(),
            PlaceholderToken::new("username".to_string(), PlaceholderType::String)
                .with_default("user".to_string())
                .optional(),
        );
        user_placeholders.insert(
            "timestamp".to_string(),
            PlaceholderToken::new(
                "timestamp".to_string(),
                PlaceholderType::Optional(Box::new(PlaceholderType::String)),
            )
            .with_default("".to_string())
            .optional(),
        );

        advanced_templates.insert(
            "advanced_user_message".to_string(),
            AdvancedTemplate {
                content: "<|user:{username}|>{content}{timestamp}<|end|>".to_string(),
                placeholders: user_placeholders,
                description: Some("Advanced user message with username and timestamp".to_string()),
                category: Some("conversation".to_string()),
            },
        );

        // API call template
        let mut api_placeholders = HashMap::new();
        api_placeholders.insert(
            "endpoint".to_string(),
            PlaceholderToken::new("endpoint".to_string(), PlaceholderType::String)
                .with_validation("url".to_string()),
        );
        api_placeholders.insert(
            "method".to_string(),
            PlaceholderToken::new("method".to_string(), PlaceholderType::String)
                .with_default("GET".to_string()),
        );
        api_placeholders.insert(
            "headers".to_string(),
            PlaceholderToken::new("headers".to_string(), PlaceholderType::StringList)
                .with_default("".to_string())
                .optional(),
        );
        api_placeholders.insert(
            "timeout".to_string(),
            PlaceholderToken::new("timeout".to_string(), PlaceholderType::Integer)
                .with_default("30".to_string()),
        );

        advanced_templates.insert(
            "api_call".to_string(),
            AdvancedTemplate {
                content: "<|api:{method}|>{endpoint}|headers:{headers}|timeout:{timeout}<|end|>"
                    .to_string(),
                placeholders: api_placeholders,
                description: Some(
                    "API call template with method, endpoint, headers, and timeout".to_string(),
                ),
                category: Some("api".to_string()),
            },
        );

        // Task completion template
        let mut task_placeholders = HashMap::new();
        task_placeholders.insert(
            "task_name".to_string(),
            PlaceholderToken::new("task_name".to_string(), PlaceholderType::String)
                .with_transformation("capitalize".to_string()),
        );
        task_placeholders.insert(
            "status".to_string(),
            PlaceholderToken::new("status".to_string(), PlaceholderType::String)
                .with_transformation("uppercase".to_string()),
        );
        task_placeholders.insert(
            "completion_percentage".to_string(),
            PlaceholderToken::new(
                "completion_percentage".to_string(),
                PlaceholderType::Integer,
            )
            .with_default("0".to_string()),
        );
        task_placeholders.insert(
            "details".to_string(),
            PlaceholderToken::new(
                "details".to_string(),
                PlaceholderType::Optional(Box::new(PlaceholderType::String)),
            )
            .with_default("".to_string())
            .optional(),
        );

        advanced_templates.insert("task_completion".to_string(), AdvancedTemplate {
            content: "<|task:{task_name}|status:{status}|completion:{completion_percentage}%{details}<|end|>".to_string(),
            placeholders: task_placeholders,
            description: Some("Task completion template with status and progress".to_string()),
            category: Some("task".to_string()),
        });

        Self {
            static_tokens,
            dynamic_tokens: HashMap::new(),
            templates,
            advanced_templates,
            max_dynamic_tokens: 1000,
            next_dynamic_id: 1000, // Start dynamic tokens from ID 1000
        }
    }
}

/// Advanced special token manager with dynamic tokens and template support
#[derive(Debug)]
pub struct SpecialTokenManager {
    config: SpecialTokenConfig,
    /// Reverse mapping from token ID to token string
    id_to_token: HashMap<u32, String>,
    /// Cache for rendered templates
    template_cache: HashMap<String, String>,
    /// Placeholder processor for advanced templates
    placeholder_processor: PlaceholderProcessor,
}

impl SpecialTokenManager {
    /// Create a new special token manager
    pub fn new(config: SpecialTokenConfig) -> Self {
        let mut id_to_token = HashMap::new();

        // Build reverse mapping for static tokens
        for (token, id) in &config.static_tokens {
            id_to_token.insert(*id, token.clone());
        }

        // Build reverse mapping for dynamic tokens
        for (token, id) in &config.dynamic_tokens {
            id_to_token.insert(*id, token.clone());
        }

        Self {
            config,
            id_to_token,
            template_cache: HashMap::new(),
            placeholder_processor: PlaceholderProcessor::new(),
        }
    }

    /// Add a dynamic special token
    pub fn add_dynamic_token(&mut self, token: String) -> Result<u32> {
        if self.config.dynamic_tokens.len() >= self.config.max_dynamic_tokens {
            return Err(TrustformersError::other(
                "Maximum number of dynamic tokens reached".to_string(),
            ));
        }

        if self.config.static_tokens.contains_key(&token)
            || self.config.dynamic_tokens.contains_key(&token)
        {
            return Err(TrustformersError::other(format!(
                "Token '{}' already exists",
                token
            )));
        }

        let token_id = self.config.next_dynamic_id;
        self.config.dynamic_tokens.insert(token.clone(), token_id);
        self.id_to_token.insert(token_id, token);
        self.config.next_dynamic_id += 1;

        Ok(token_id)
    }

    /// Remove a dynamic special token
    pub fn remove_dynamic_token(&mut self, token: &str) -> Result<()> {
        if let Some(token_id) = self.config.dynamic_tokens.remove(token) {
            self.id_to_token.remove(&token_id);
            Ok(())
        } else {
            Err(TrustformersError::other(format!(
                "Dynamic token '{}' not found",
                token
            )))
        }
    }

    /// Get token ID by token string
    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.config
            .static_tokens
            .get(token)
            .or_else(|| self.config.dynamic_tokens.get(token))
            .copied()
    }

    /// Get token string by token ID
    pub fn get_token(&self, id: u32) -> Option<&String> {
        self.id_to_token.get(&id)
    }

    /// Check if a token is a special token
    pub fn is_special_token(&self, token: &str) -> bool {
        self.config.static_tokens.contains_key(token)
            || self.config.dynamic_tokens.contains_key(token)
    }

    /// Get all special tokens
    pub fn get_all_tokens(&self) -> Vec<(String, u32)> {
        let mut tokens = Vec::new();

        for (token, id) in &self.config.static_tokens {
            tokens.push((token.clone(), *id));
        }

        for (token, id) in &self.config.dynamic_tokens {
            tokens.push((token.clone(), *id));
        }

        tokens.sort_by_key(|(_, id)| *id);
        tokens
    }

    /// Add a new template
    pub fn add_template(&mut self, name: String, template: String) {
        self.config.templates.insert(name, template);
        self.template_cache.clear(); // Clear cache when templates change
    }

    /// Remove a template
    pub fn remove_template(&mut self, name: &str) -> bool {
        self.template_cache.clear(); // Clear cache when templates change
        self.config.templates.remove(name).is_some()
    }

    /// Render a template with given parameters
    pub fn render_template(
        &mut self,
        template_name: &str,
        params: &HashMap<String, String>,
    ) -> Result<String> {
        let template = self.config.templates.get(template_name).ok_or_else(|| {
            TrustformersError::other(format!("Template '{}' not found", template_name))
        })?;

        let mut result = template.clone();
        for (key, value) in params {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }

        Ok(result)
    }

    /// Render a template and return token IDs
    pub fn render_template_to_ids(
        &mut self,
        template_name: &str,
        params: &HashMap<String, String>,
    ) -> Result<Vec<u32>> {
        let rendered = self.render_template(template_name, params)?;
        Ok(self.tokenize_with_special_tokens(&rendered))
    }

    /// Tokenize text while preserving special tokens
    pub fn tokenize_with_special_tokens(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut current_pos = 0;

        while current_pos < text.len() {
            let mut found_special = false;

            // Look for special tokens starting at current position
            for (token, id) in self.get_all_tokens() {
                if text[current_pos..].starts_with(&token) {
                    tokens.push(id);
                    current_pos += token.len();
                    found_special = true;
                    break;
                }
            }

            if !found_special {
                // Extract regular text until next special token or end
                let mut end_pos = text.len();
                for (token, _) in self.get_all_tokens() {
                    if let Some(pos) = text[current_pos..].find(&token) {
                        end_pos = end_pos.min(current_pos + pos);
                    }
                }

                let regular_text = &text[current_pos..end_pos];
                if !regular_text.is_empty() {
                    // For simplicity, we'll encode regular text as character tokens
                    // In a real implementation, this would use the main tokenizer
                    for ch in regular_text.chars() {
                        // This is a placeholder - in practice, you'd use your main tokenizer
                        tokens.push(ch as u32);
                    }
                }

                current_pos = end_pos;
            }
        }

        tokens
    }

    /// Format text using a template
    pub fn format_text(&mut self, template_name: &str, content: &str) -> Result<String> {
        let mut params = HashMap::new();
        params.insert("content".to_string(), content.to_string());
        self.render_template(template_name, &params)
    }

    /// Format conversation messages
    pub fn format_conversation(&mut self, messages: &[ConversationMessage]) -> Result<String> {
        let mut result = String::new();

        for message in messages {
            let template_name = match message.role.as_str() {
                "user" => "user_message",
                "assistant" => "assistant_message",
                "system" => "system_message",
                _ => {
                    return Err(TrustformersError::other(format!(
                        "Unknown message role: {}",
                        message.role
                    )))
                },
            };

            let formatted = self.format_text(template_name, &message.content)?;
            result.push_str(&formatted);
        }

        Ok(result)
    }

    /// Create control tokens for specific tasks
    pub fn create_control_tokens(&mut self, task: &str) -> Result<Vec<String>> {
        let tokens = match task {
            "classification" => vec![
                format!("<|class_{}|>", 0),
                format!("<|class_{}|>", 1),
                "<|confidence|>".to_string(),
                "<|prediction|>".to_string(),
            ],
            "generation" => vec![
                "<|start_generation|>".to_string(),
                "<|end_generation|>".to_string(),
                "<|temperature|>".to_string(),
                "<|max_length|>".to_string(),
            ],
            "qa" => vec![
                "<|question|>".to_string(),
                "<|answer|>".to_string(),
                "<|context|>".to_string(),
                "<|confidence|>".to_string(),
            ],
            _ => return Err(TrustformersError::other(format!("Unknown task: {}", task))),
        };

        // Add these as dynamic tokens
        let mut token_ids = Vec::new();
        for token in &tokens {
            if let Ok(id) = self.add_dynamic_token(token.clone()) {
                token_ids.push(id);
            }
        }

        Ok(tokens)
    }

    /// Add an advanced template with placeholder tokens
    pub fn add_advanced_template(&mut self, name: String, template: AdvancedTemplate) {
        self.config.advanced_templates.insert(name, template);
        self.template_cache.clear(); // Clear cache when templates change
    }

    /// Remove an advanced template
    pub fn remove_advanced_template(&mut self, name: &str) -> bool {
        self.template_cache.clear(); // Clear cache when templates change
        self.config.advanced_templates.remove(name).is_some()
    }

    /// Get all advanced template names
    pub fn get_advanced_template_names(&self) -> Vec<String> {
        self.config.advanced_templates.keys().cloned().collect()
    }

    /// Get advanced template by name
    pub fn get_advanced_template(&self, name: &str) -> Option<&AdvancedTemplate> {
        self.config.advanced_templates.get(name)
    }

    /// Render an advanced template with placeholder tokens
    pub fn render_advanced_template(
        &mut self,
        template_name: &str,
        params: &HashMap<String, String>,
    ) -> Result<String> {
        let template = self.config.advanced_templates.get(template_name).ok_or_else(|| {
            TrustformersError::other(format!("Advanced template '{}' not found", template_name))
        })?;

        let mut result = template.content.clone();

        // Process each placeholder
        for (placeholder_name, placeholder_token) in &template.placeholders {
            let value = if let Some(provided_value) = params.get(placeholder_name) {
                provided_value.clone()
            } else if let Some(default_value) = &placeholder_token.default_value {
                default_value.clone()
            } else if placeholder_token.required {
                return Err(TrustformersError::other(format!(
                    "Required placeholder '{}' not provided",
                    placeholder_name
                )));
            } else {
                String::new()
            };

            // Process the placeholder value
            let processed = self.placeholder_processor.process_value(placeholder_token, &value)?;

            // Replace the placeholder in the template
            let placeholder_pattern = format!("{{{}}}", placeholder_name);
            result = result.replace(&placeholder_pattern, &processed.processed_value);
        }

        Ok(result)
    }

    /// Render an advanced template and return token IDs
    pub fn render_advanced_template_to_ids(
        &mut self,
        template_name: &str,
        params: &HashMap<String, String>,
    ) -> Result<Vec<u32>> {
        let rendered = self.render_advanced_template(template_name, params)?;
        Ok(self.tokenize_with_special_tokens(&rendered))
    }

    /// Validate placeholder values for a template
    pub fn validate_template_params(
        &self,
        template_name: &str,
        params: &HashMap<String, String>,
    ) -> Result<Vec<String>> {
        let template = self.config.advanced_templates.get(template_name).ok_or_else(|| {
            TrustformersError::other(format!("Advanced template '{}' not found", template_name))
        })?;

        let mut validation_errors = Vec::new();

        for (placeholder_name, placeholder_token) in &template.placeholders {
            if let Some(value) = params.get(placeholder_name) {
                if let Err(e) = self.placeholder_processor.validate_value(placeholder_token, value)
                {
                    validation_errors.push(format!("Placeholder '{}': {}", placeholder_name, e));
                }
            } else if placeholder_token.required {
                validation_errors.push(format!(
                    "Required placeholder '{}' not provided",
                    placeholder_name
                ));
            }
        }

        Ok(validation_errors)
    }

    /// Get placeholder documentation for a template
    pub fn get_template_documentation(
        &self,
        template_name: &str,
    ) -> Result<HashMap<String, String>> {
        let template = self.config.advanced_templates.get(template_name).ok_or_else(|| {
            TrustformersError::other(format!("Advanced template '{}' not found", template_name))
        })?;

        let mut docs = HashMap::new();

        for (placeholder_name, placeholder_token) in &template.placeholders {
            let mut doc = format!("Type: {:?}", placeholder_token.placeholder_type);

            if placeholder_token.required {
                doc.push_str(" (Required)");
            } else {
                doc.push_str(" (Optional)");
            }

            if let Some(default) = &placeholder_token.default_value {
                doc.push_str(&format!(", Default: '{}'", default));
            }

            if let Some(pattern) = &placeholder_token.validation_pattern {
                doc.push_str(&format!(", Validation: {}", pattern));
            }

            if let Some(transformation) = &placeholder_token.transformation {
                doc.push_str(&format!(", Transformation: {}", transformation));
            }

            if let Some(description) = &placeholder_token.description {
                doc.push_str(&format!(", Description: {}", description));
            }

            docs.insert(placeholder_name.clone(), doc);
        }

        Ok(docs)
    }

    /// Add a custom transformation to the placeholder processor
    pub fn add_transformation(&mut self, name: String, func: fn(&str) -> String) {
        self.placeholder_processor.add_transformation(name, func);
    }

    /// Create placeholder tokens dynamically
    pub fn create_placeholder_tokens(
        &mut self,
        placeholders: &[(&str, PlaceholderType)],
    ) -> Vec<String> {
        let mut created_tokens = Vec::new();

        for (name, _placeholder_type) in placeholders {
            let token = format!("<|placeholder:{}|>", name);
            if self.add_dynamic_token(token.clone()).is_ok() {
                created_tokens.push(token);
            }
        }

        created_tokens
    }

    /// Export configuration to JSON
    pub fn export_config(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.config)
            .map_err(|e| TrustformersError::serialization_error(e.to_string()))
    }

    /// Import configuration from JSON
    pub fn import_config(&mut self, json: &str) -> Result<()> {
        let config: SpecialTokenConfig = serde_json::from_str(json)
            .map_err(|e| TrustformersError::serialization_error(e.to_string()))?;

        self.config = config;
        self.rebuild_reverse_mapping();
        self.template_cache.clear();

        Ok(())
    }

    /// Rebuild the reverse mapping from token ID to token string
    fn rebuild_reverse_mapping(&mut self) {
        self.id_to_token.clear();

        for (token, id) in &self.config.static_tokens {
            self.id_to_token.insert(*id, token.clone());
        }

        for (token, id) in &self.config.dynamic_tokens {
            self.id_to_token.insert(*id, token.clone());
        }

        // Reset placeholder processor to default state
        self.placeholder_processor = PlaceholderProcessor::new();
    }
}

/// Represents a conversation message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

impl ConversationMessage {
    pub fn new(role: String, content: String) -> Self {
        Self { role, content }
    }

    pub fn user(content: String) -> Self {
        Self::new("user".to_string(), content)
    }

    pub fn assistant(content: String) -> Self {
        Self::new("assistant".to_string(), content)
    }

    pub fn system(content: String) -> Self {
        Self::new("system".to_string(), content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_token_manager_creation() {
        let config = SpecialTokenConfig::default();
        let manager = SpecialTokenManager::new(config);

        assert_eq!(manager.get_token_id("[PAD]"), Some(0));
        assert_eq!(manager.get_token_id("[UNK]"), Some(1));
        assert_eq!(manager.get_token(0), Some(&"[PAD]".to_string()));
    }

    #[test]
    fn test_dynamic_token_management() {
        let config = SpecialTokenConfig::default();
        let mut manager = SpecialTokenManager::new(config);

        let token_id = manager.add_dynamic_token("<|test|>".to_string()).unwrap();
        assert_eq!(manager.get_token_id("<|test|>"), Some(token_id));
        assert_eq!(manager.get_token(token_id), Some(&"<|test|>".to_string()));

        manager.remove_dynamic_token("<|test|>").unwrap();
        assert_eq!(manager.get_token_id("<|test|>"), None);
    }

    #[test]
    fn test_template_rendering() {
        let config = SpecialTokenConfig::default();
        let mut manager = SpecialTokenManager::new(config);

        let mut params = HashMap::new();
        params.insert("content".to_string(), "Hello, world!".to_string());

        let result = manager.render_template("user_message", &params).unwrap();
        assert_eq!(result, "<|user|>Hello, world!<|end|>");
    }

    #[test]
    fn test_conversation_formatting() {
        let config = SpecialTokenConfig::default();
        let mut manager = SpecialTokenManager::new(config);

        let messages = vec![
            ConversationMessage::system("You are a helpful assistant.".to_string()),
            ConversationMessage::user("What is 2+2?".to_string()),
            ConversationMessage::assistant("2+2 equals 4.".to_string()),
        ];

        let result = manager.format_conversation(&messages).unwrap();
        let expected = "<|system|>You are a helpful assistant.<|end|><|user|>What is 2+2?<|end|><|assistant|>2+2 equals 4.<|end|>";
        assert_eq!(result, expected);
    }

    #[test]
    fn test_control_token_creation() {
        let config = SpecialTokenConfig::default();
        let mut manager = SpecialTokenManager::new(config);

        let tokens = manager.create_control_tokens("classification").unwrap();
        assert!(tokens.contains(&"<|class_0|>".to_string()));
        assert!(tokens.contains(&"<|class_1|>".to_string()));
        assert!(tokens.contains(&"<|confidence|>".to_string()));

        // Verify tokens were added as dynamic tokens
        assert!(manager.is_special_token("<|class_0|>"));
        assert!(manager.is_special_token("<|confidence|>"));
    }

    #[test]
    fn test_special_token_detection() {
        let config = SpecialTokenConfig::default();
        let manager = SpecialTokenManager::new(config);

        assert!(manager.is_special_token("[PAD]"));
        assert!(manager.is_special_token("[UNK]"));
        assert!(!manager.is_special_token("regular_token"));
    }

    #[test]
    fn test_export_import_config() {
        let config = SpecialTokenConfig::default();
        let mut manager = SpecialTokenManager::new(config);

        // Add a dynamic token
        manager.add_dynamic_token("<|test|>".to_string()).unwrap();

        // Export and reimport
        let exported = manager.export_config().unwrap();
        manager.import_config(&exported).unwrap();

        // Verify the dynamic token is still there
        assert!(manager.is_special_token("<|test|>"));
    }

    #[test]
    fn test_placeholder_tokens() {
        let config = SpecialTokenConfig::default();
        let _manager = SpecialTokenManager::new(config);

        // Test basic placeholder creation
        let placeholder = PlaceholderToken::new("test".to_string(), PlaceholderType::String)
            .with_default("default_value".to_string())
            .with_validation("email".to_string())
            .with_transformation("uppercase".to_string());

        assert_eq!(placeholder.name, "test");
        assert_eq!(placeholder.placeholder_type, PlaceholderType::String);
        assert_eq!(placeholder.default_value, Some("default_value".to_string()));
        assert_eq!(placeholder.validation_pattern, Some("email".to_string()));
        assert_eq!(placeholder.transformation, Some("uppercase".to_string()));
    }

    #[test]
    fn test_placeholder_processor() {
        let processor = PlaceholderProcessor::new();

        let placeholder = PlaceholderToken::new("test".to_string(), PlaceholderType::String)
            .with_transformation("uppercase".to_string());

        let result = processor.process_value(&placeholder, "hello world").unwrap();
        assert_eq!(result.processed_value, "HELLO WORLD");
        assert_eq!(result.raw_value, "hello world");
    }

    #[test]
    fn test_placeholder_validation() {
        let processor = PlaceholderProcessor::new();

        // Test integer validation
        let int_placeholder = PlaceholderToken::new("test".to_string(), PlaceholderType::Integer);
        assert!(processor.validate_value(&int_placeholder, "123").is_ok());
        assert!(processor.validate_value(&int_placeholder, "not_a_number").is_err());

        // Test email validation
        let email_placeholder = PlaceholderToken::new("email".to_string(), PlaceholderType::String)
            .with_validation("email".to_string());
        assert!(processor.validate_value(&email_placeholder, "test@example.com").is_ok());
        assert!(processor.validate_value(&email_placeholder, "invalid_email").is_err());
    }

    #[test]
    fn test_advanced_template_rendering() {
        let config = SpecialTokenConfig::default();
        let mut manager = SpecialTokenManager::new(config);

        // Test rendering the default advanced_user_message template
        let mut params = HashMap::new();
        params.insert("content".to_string(), "Hello, world!".to_string());
        params.insert("username".to_string(), "alice".to_string());

        let result = manager.render_advanced_template("advanced_user_message", &params).unwrap();
        assert!(result.contains("alice"));
        assert!(result.contains("Hello, world!"));
    }

    #[test]
    fn test_advanced_template_with_defaults() {
        let config = SpecialTokenConfig::default();
        let mut manager = SpecialTokenManager::new(config);

        // Test rendering with only required parameters (username should default to "user")
        let mut params = HashMap::new();
        params.insert("content".to_string(), "Hello!".to_string());

        let result = manager.render_advanced_template("advanced_user_message", &params).unwrap();
        assert!(result.contains("user")); // Default username
        assert!(result.contains("Hello!"));
    }

    #[test]
    fn test_template_validation() {
        let config = SpecialTokenConfig::default();
        let manager = SpecialTokenManager::new(config);

        // Test validation with missing required parameter
        let params = HashMap::new(); // Missing required "content" parameter

        let errors = manager.validate_template_params("advanced_user_message", &params).unwrap();
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("content")));
    }

    #[test]
    fn test_template_documentation() {
        let config = SpecialTokenConfig::default();
        let manager = SpecialTokenManager::new(config);

        let docs = manager.get_template_documentation("advanced_user_message").unwrap();
        assert!(docs.contains_key("content"));
        assert!(docs.contains_key("username"));
        assert!(docs.contains_key("timestamp"));

        // Check that documentation includes type information
        let content_doc = &docs["content"];
        assert!(content_doc.contains("String"));
        assert!(content_doc.contains("Required"));
    }

    #[test]
    fn test_api_call_template() {
        let config = SpecialTokenConfig::default();
        let mut manager = SpecialTokenManager::new(config);

        let mut params = HashMap::new();
        params.insert(
            "endpoint".to_string(),
            "https://api.example.com/data".to_string(),
        );
        params.insert("method".to_string(), "POST".to_string());
        params.insert(
            "headers".to_string(),
            "Content-Type: application/json, Authorization: Bearer token".to_string(),
        );
        params.insert("timeout".to_string(), "60".to_string());

        let result = manager.render_advanced_template("api_call", &params).unwrap();
        assert!(result.contains("POST"));
        assert!(result.contains("https://api.example.com/data"));
        assert!(result.contains("60"));
    }

    #[test]
    fn test_task_completion_template() {
        let config = SpecialTokenConfig::default();
        let mut manager = SpecialTokenManager::new(config);

        let mut params = HashMap::new();
        params.insert("task_name".to_string(), "data processing".to_string());
        params.insert("status".to_string(), "completed".to_string());
        params.insert("completion_percentage".to_string(), "100".to_string());

        let result = manager.render_advanced_template("task_completion", &params).unwrap();
        assert!(result.contains("Data Processing")); // Should be capitalized
        assert!(result.contains("COMPLETED")); // Should be uppercase
        assert!(result.contains("100%"));
    }

    #[test]
    fn test_custom_transformation() {
        let config = SpecialTokenConfig::default();
        let mut manager = SpecialTokenManager::new(config);

        // Add a custom transformation
        manager.add_transformation("reverse".to_string(), |s: &str| s.chars().rev().collect());

        // Create a custom template using the transformation
        let mut placeholders = HashMap::new();
        placeholders.insert(
            "text".to_string(),
            PlaceholderToken::new("text".to_string(), PlaceholderType::String)
                .with_transformation("reverse".to_string()),
        );

        let template = AdvancedTemplate {
            content: "Reversed: {text}".to_string(),
            placeholders,
            description: Some("Test template with custom transformation".to_string()),
            category: Some("test".to_string()),
        };

        manager.add_advanced_template("reverse_test".to_string(), template);

        let mut params = HashMap::new();
        params.insert("text".to_string(), "hello".to_string());

        let result = manager.render_advanced_template("reverse_test", &params).unwrap();
        assert!(result.contains("olleh")); // "hello" reversed
    }

    #[test]
    fn test_placeholder_token_creation() {
        let config = SpecialTokenConfig::default();
        let mut manager = SpecialTokenManager::new(config);

        let placeholders = vec![
            ("user_id", PlaceholderType::Integer),
            ("email", PlaceholderType::String),
            ("is_active", PlaceholderType::Boolean),
        ];

        let created_tokens = manager.create_placeholder_tokens(&placeholders);
        assert_eq!(created_tokens.len(), 3);
        assert!(created_tokens.iter().any(|t| t.contains("user_id")));
        assert!(created_tokens.iter().any(|t| t.contains("email")));
        assert!(created_tokens.iter().any(|t| t.contains("is_active")));

        // Verify tokens were added as dynamic tokens
        for token in &created_tokens {
            assert!(manager.is_special_token(token));
        }
    }
}
