//! Template engine for code generation
//!
//! Provides a simple template system for generating language bindings
//! with placeholder substitution and basic control structures.

use super::{ast, CodeGenConfig};
use crate::error::TrustformersResult;
use anyhow::anyhow;
use std::collections::HashMap;

/// Simple template engine for code generation
pub struct TemplateEngine {
    /// Template cache
    templates: HashMap<String, Template>,
}

/// A parsed template with placeholders
#[derive(Debug, Clone)]
pub struct Template {
    /// Template content with placeholders
    pub content: String,
    /// Identified placeholders in the template
    pub placeholders: Vec<String>,
}

/// Context for template rendering
pub type TemplateContext = HashMap<String, TemplateValue>;

/// Values that can be substituted in templates
#[derive(Debug, Clone)]
pub enum TemplateValue {
    String(String),
    Number(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<TemplateValue>),
    Object(HashMap<String, TemplateValue>),
}

impl TemplateEngine {
    /// Create a new template engine
    pub fn new() -> TrustformersResult<Self> {
        Ok(Self {
            templates: HashMap::new(),
        })
    }

    /// Load a template from string content
    pub fn load_template(&mut self, name: &str, content: &str) -> TrustformersResult<()> {
        let template = Template::parse(content)?;
        self.templates.insert(name.to_string(), template);
        Ok(())
    }

    /// Get a template by name
    pub fn get_template(&self, name: &str) -> Option<&Template> {
        self.templates.get(name)
    }

    /// Render a template with the given context
    pub fn render(
        &self,
        template_name: &str,
        context: &TemplateContext,
    ) -> TrustformersResult<String> {
        let template = self
            .templates
            .get(template_name)
            .ok_or_else(|| anyhow!("Template '{}' not found", template_name))?;

        template.render(context)
    }

    /// Render template content directly without loading
    pub fn render_string(
        &self,
        content: &str,
        context: &TemplateContext,
    ) -> TrustformersResult<String> {
        let template = Template::parse(content)?;
        template.render(context)
    }

    /// Load default templates for common patterns
    pub fn load_default_templates(&mut self) -> TrustformersResult<()> {
        // Python function template
        self.load_template(
            "python_function",
            r#"def {{function_name}}({{parameters}}):
    """{{documentation}}"""
    {{#if can_fail}}
    result = _lib.{{c_name}}({{parameter_names}})
    if result != 0:
        raise TrustformersError(f"Function {{function_name}} failed with error code {result}")
    return result
    {{else}}
    return _lib.{{c_name}}({{parameter_names}})
    {{/if}}"#,
        )?;

        // Python class template
        self.load_template(
            "python_class",
            r#"class {{class_name}}(ctypes.Structure):
    """{{documentation}}"""
    {{#if is_opaque}}
    pass
    {{else}}
    _fields_ = [
        {{#each fields}}
        ('{{name}}', {{type}}),
        {{/each}}
    ]
    {{/if}}"#,
        )?;

        // C header template
        self.load_template(
            "c_header",
            r#"/* {{library_name}} C API
 * {{description}}
 * Version: {{version}}
 */

#ifndef {{header_guard}}
#define {{header_guard}}

#ifdef __cplusplus
extern "C" {
#endif

{{#each functions}}
{{return_type}} {{name}}({{parameters}});
{{/each}}

#ifdef __cplusplus
}
#endif

#endif /* {{header_guard}} */"#,
        )?;

        Ok(())
    }
}

impl Template {
    /// Parse template content and identify placeholders
    pub fn parse(content: &str) -> TrustformersResult<Self> {
        let mut placeholders = Vec::new();

        // Simple regex to find {{placeholder}} patterns
        let placeholder_regex = regex::Regex::new(r"\{\{([^}]+)\}\}")
            .expect("static regex pattern is valid");

        for captures in placeholder_regex.captures_iter(content) {
            if let Some(placeholder) = captures.get(1) {
                let placeholder_name = placeholder.as_str().trim().to_string();
                if !placeholders.contains(&placeholder_name) {
                    placeholders.push(placeholder_name);
                }
            }
        }

        Ok(Self {
            content: content.to_string(),
            placeholders,
        })
    }

    /// Render the template with the given context
    pub fn render(&self, context: &TemplateContext) -> TrustformersResult<String> {
        let mut result = self.content.clone();

        // Simple placeholder substitution
        for placeholder in &self.placeholders {
            let pattern = format!("{{{{{}}}}}", placeholder);

            if let Some(value) = context.get(placeholder) {
                let replacement = self.format_value(value)?;
                result = result.replace(&pattern, &replacement);
            } else {
                // Leave placeholder as-is if no value provided
                // In a more sophisticated implementation, this might be an error
            }
        }

        // Handle simple conditionals (very basic implementation)
        result = self.handle_conditionals(&result, context)?;

        // Handle simple loops (very basic implementation)
        result = self.handle_loops(&result, context)?;

        Ok(result)
    }

    fn format_value(&self, value: &TemplateValue) -> TrustformersResult<String> {
        match value {
            TemplateValue::String(s) => Ok(s.clone()),
            TemplateValue::Number(n) => Ok(n.to_string()),
            TemplateValue::Float(f) => Ok(f.to_string()),
            TemplateValue::Boolean(b) => Ok(b.to_string()),
            TemplateValue::List(_) => Ok("[list]".to_string()), // Simplified
            TemplateValue::Object(_) => Ok("{object}".to_string()), // Simplified
        }
    }

    fn handle_conditionals(
        &self,
        content: &str,
        context: &TemplateContext,
    ) -> TrustformersResult<String> {
        let mut result = content.to_string();

        // Very basic conditional handling: {{#if condition}}...{{/if}}
        // Use (?s) flag to enable single-line mode where . matches newlines
        let if_regex = regex::Regex::new(r"(?s)\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}")
            .expect("Regex pattern for if statements should be valid");

        while let Some(captures) = if_regex.captures(&result) {
            let condition_name = captures
                .get(1)
                .expect("Capture group 1 should exist in if regex match")
                .as_str();
            let conditional_content = captures
                .get(2)
                .expect("Capture group 2 should exist in if regex match")
                .as_str();
            let full_match = captures
                .get(0)
                .expect("Capture group 0 should exist in if regex match")
                .as_str();

            let should_include = if let Some(value) = context.get(condition_name) {
                match value {
                    TemplateValue::Boolean(b) => *b,
                    TemplateValue::String(s) => !s.is_empty(),
                    TemplateValue::Number(n) => *n != 0,
                    TemplateValue::Float(f) => *f != 0.0,
                    TemplateValue::List(l) => !l.is_empty(),
                    _ => false,
                }
            } else {
                false
            };

            let replacement =
                if should_include { conditional_content.to_string() } else { String::new() };

            result = result.replace(full_match, &replacement);
        }

        // Handle {{#if condition}}...{{else}}...{{/if}}
        // Use (?s) flag to enable single-line mode where . matches newlines
        let if_else_regex =
            regex::Regex::new(r"(?s)\{\{#if\s+(\w+)\}\}(.*?)\{\{else\}\}(.*?)\{\{/if\}\}")
                .expect("Regex pattern for if-else statements should be valid");

        while let Some(captures) = if_else_regex.captures(&result) {
            let condition_name = captures
                .get(1)
                .expect("Capture group 1 should exist in if-else regex match")
                .as_str();
            let if_content = captures
                .get(2)
                .expect("Capture group 2 should exist in if-else regex match")
                .as_str();
            let else_content = captures
                .get(3)
                .expect("Capture group 3 should exist in if-else regex match")
                .as_str();
            let full_match = captures
                .get(0)
                .expect("Capture group 0 should exist in if-else regex match")
                .as_str();

            let should_include_if = if let Some(value) = context.get(condition_name) {
                match value {
                    TemplateValue::Boolean(b) => *b,
                    TemplateValue::String(s) => !s.is_empty(),
                    TemplateValue::Number(n) => *n != 0,
                    TemplateValue::Float(f) => *f != 0.0,
                    TemplateValue::List(l) => !l.is_empty(),
                    _ => false,
                }
            } else {
                false
            };

            let replacement = if should_include_if {
                if_content.to_string()
            } else {
                else_content.to_string()
            };

            result = result.replace(full_match, &replacement);
        }

        Ok(result)
    }

    fn handle_loops(&self, content: &str, context: &TemplateContext) -> TrustformersResult<String> {
        let mut result = content.to_string();

        // Basic loop handling: {{#each list_name}}...{{/each}}
        // Use (?s) flag to enable single-line mode where . matches newlines
        let each_regex = regex::Regex::new(r"(?s)\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}")
            .expect("Regex pattern for each loops should be valid");

        while let Some(captures) = each_regex.captures(&result) {
            let list_name = captures
                .get(1)
                .expect("Capture group 1 should exist in each regex match")
                .as_str();
            let loop_content = captures
                .get(2)
                .expect("Capture group 2 should exist in each regex match")
                .as_str();
            let full_match = captures
                .get(0)
                .expect("Capture group 0 should exist in each regex match")
                .as_str();

            let replacement = if let Some(value) = context.get(list_name) {
                match value {
                    TemplateValue::List(items) => {
                        let mut loop_result = String::new();
                        for item in items {
                            // For simplicity, treat each item as a context
                            if let TemplateValue::Object(item_context) = item {
                                let item_template = Template::parse(loop_content)?;
                                let rendered_item = item_template.render(item_context)?;
                                loop_result.push_str(&rendered_item);
                            } else {
                                // For non-object items, create a simple context
                                let mut item_context = HashMap::new();
                                item_context.insert("item".to_string(), item.clone());
                                let item_template = Template::parse(loop_content)?;
                                let rendered_item = item_template.render(&item_context)?;
                                loop_result.push_str(&rendered_item);
                            }
                        }
                        loop_result
                    },
                    _ => String::new(),
                }
            } else {
                String::new()
            };

            result = result.replace(full_match, &replacement);
        }

        Ok(result)
    }
}

// Helper functions for creating template values

impl TemplateValue {
    /// Create a string template value
    pub fn string<S: Into<String>>(s: S) -> Self {
        TemplateValue::String(s.into())
    }

    /// Create a number template value
    pub fn number(n: i64) -> Self {
        TemplateValue::Number(n)
    }

    /// Create a float template value
    pub fn float(f: f64) -> Self {
        TemplateValue::Float(f)
    }

    /// Create a boolean template value
    pub fn boolean(b: bool) -> Self {
        TemplateValue::Boolean(b)
    }

    /// Create a list template value
    pub fn list(items: Vec<TemplateValue>) -> Self {
        TemplateValue::List(items)
    }

    /// Create an object template value
    pub fn object(map: HashMap<String, TemplateValue>) -> Self {
        TemplateValue::Object(map)
    }
}

/// Macro for creating template contexts more easily
#[macro_export]
macro_rules! template_context {
    ($($key:expr => $value:expr),* $(,)?) => {
        {
            let mut context = std::collections::HashMap::new();
            $(
                context.insert($key.to_string(), $value);
            )*
            context
        }
    };
}

/// Helper functions for common template patterns
pub mod helpers {
    use super::*;

    /// Create a context for a function
    pub fn function_context(func: &ast::FfiFunction) -> TemplateContext {
        let mut context = HashMap::new();

        context.insert(
            "function_name".to_string(),
            TemplateValue::string(&func.name),
        );
        context.insert("c_name".to_string(), TemplateValue::string(&func.c_name));
        context.insert(
            "can_fail".to_string(),
            TemplateValue::boolean(func.can_fail()),
        );
        context.insert(
            "documentation".to_string(),
            TemplateValue::string(func.documentation.join(" ")),
        );

        // Parameters
        let param_names: Vec<String> = func.parameters.iter().map(|p| p.name.clone()).collect();
        context.insert(
            "parameter_names".to_string(),
            TemplateValue::string(param_names.join(", ")),
        );

        // Parameter list for function signature
        let param_sigs: Vec<String> = func
            .parameters
            .iter()
            .map(|p| format!("{}: {}", p.name, p.type_info.name))
            .collect();
        context.insert(
            "parameters".to_string(),
            TemplateValue::string(param_sigs.join(", ")),
        );

        context
    }

    /// Create a context for a struct
    pub fn struct_context(struct_def: &ast::FfiStruct) -> TemplateContext {
        let mut context = HashMap::new();

        context.insert(
            "class_name".to_string(),
            TemplateValue::string(&struct_def.name),
        );
        context.insert(
            "c_name".to_string(),
            TemplateValue::string(&struct_def.c_name),
        );
        context.insert(
            "is_opaque".to_string(),
            TemplateValue::boolean(struct_def.is_opaque),
        );
        context.insert(
            "documentation".to_string(),
            TemplateValue::string(struct_def.documentation.join(" ")),
        );

        // Fields
        let fields: Vec<TemplateValue> = struct_def
            .fields
            .iter()
            .filter(|f| !f.is_private)
            .map(|f| {
                let mut field_context = HashMap::new();
                field_context.insert("name".to_string(), TemplateValue::string(&f.name));
                field_context.insert("type".to_string(), TemplateValue::string(&f.type_info.name));
                TemplateValue::object(field_context)
            })
            .collect();

        context.insert("fields".to_string(), TemplateValue::list(fields));

        context
    }

    /// Create a context for the entire interface
    pub fn interface_context(
        interface: &ast::FfiInterface,
        config: &CodeGenConfig,
    ) -> TemplateContext {
        let mut context = HashMap::new();

        context.insert(
            "library_name".to_string(),
            TemplateValue::string(&config.package_info.name),
        );
        context.insert(
            "description".to_string(),
            TemplateValue::string(&config.package_info.description),
        );
        context.insert(
            "version".to_string(),
            TemplateValue::string(&config.package_info.version),
        );
        context.insert(
            "author".to_string(),
            TemplateValue::string(&config.package_info.author),
        );

        // Header guard for C headers
        let header_guard = format!("{}_H", config.package_info.name.to_uppercase());
        context.insert(
            "header_guard".to_string(),
            TemplateValue::string(header_guard),
        );

        // Functions
        let functions: Vec<TemplateValue> = interface
            .functions
            .iter()
            .map(|f| {
                let func_ctx = function_context(f);
                TemplateValue::object(func_ctx)
            })
            .collect();
        context.insert("functions".to_string(), TemplateValue::list(functions));

        // Structs
        let structs: Vec<TemplateValue> = interface
            .structs
            .iter()
            .map(|s| {
                let struct_ctx = struct_context(s);
                TemplateValue::object(struct_ctx)
            })
            .collect();
        context.insert("structs".to_string(), TemplateValue::list(structs));

        context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_parsing() {
        let template_content = "Hello {{name}}, you are {{age}} years old!";
        let template = Template::parse(template_content).unwrap();

        assert_eq!(template.placeholders.len(), 2);
        assert!(template.placeholders.contains(&"name".to_string()));
        assert!(template.placeholders.contains(&"age".to_string()));
    }

    #[test]
    fn test_template_rendering() {
        let template_content = "Hello {{name}}, you are {{age}} years old!";
        let template = Template::parse(template_content).unwrap();

        let context = template_context! {
            "name" => TemplateValue::string("Alice"),
            "age" => TemplateValue::number(30),
        };

        let result = template.render(&context).unwrap();
        assert_eq!(result, "Hello Alice, you are 30 years old!");
    }

    #[test]
    fn test_conditional_rendering() {
        let template_content = "{{#if show_message}}Hello World!{{/if}}";
        let template = Template::parse(template_content).unwrap();

        let context_true = template_context! {
            "show_message" => TemplateValue::boolean(true),
        };

        let context_false = template_context! {
            "show_message" => TemplateValue::boolean(false),
        };

        let result_true = template.render(&context_true).unwrap();
        let result_false = template.render(&context_false).unwrap();

        assert_eq!(result_true, "Hello World!");
        assert_eq!(result_false, "");
    }

    #[test]
    fn test_loop_rendering() {
        let template_content = "{{#each items}}Item: {{item}}\n{{/each}}";
        let template = Template::parse(template_content).unwrap();

        let items = vec![
            TemplateValue::string("first"),
            TemplateValue::string("second"),
            TemplateValue::string("third"),
        ];

        let context = template_context! {
            "items" => TemplateValue::list(items),
        };

        let result = template.render(&context).unwrap();
        assert_eq!(result, "Item: first\nItem: second\nItem: third\n");
    }

    #[test]
    fn test_template_engine() {
        let mut engine = TemplateEngine::new().unwrap();

        engine.load_template("greeting", "Hello {{name}}!").unwrap();

        let context = template_context! {
            "name" => TemplateValue::string("World"),
        };

        let result = engine.render("greeting", &context).unwrap();
        assert_eq!(result, "Hello World!");
    }
}
