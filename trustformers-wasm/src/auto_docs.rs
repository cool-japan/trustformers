//! Automatic documentation generator from TypeScript definitions
//!
//! This module provides comprehensive documentation generation capabilities including:
//! - TypeScript definition parsing and analysis
//! - Multi-format documentation output (HTML, Markdown, JSON)
//! - API reference generation with examples
//! - Interactive playground documentation
//! - Theme support and customization

use js_sys::Array;
use serde::{Deserialize, Serialize};
use std::format;
use std::string::{String, ToString};
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Documentation output formats
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocFormat {
    /// HTML documentation with interactive features
    HTML,
    /// Markdown documentation
    Markdown,
    /// JSON API documentation
    JSON,
    /// OpenAPI/Swagger specification
    OpenAPI,
    /// TypeDoc compatible format
    TypeDoc,
}

/// Documentation themes
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocTheme {
    /// Default light theme
    Default,
    /// Dark theme
    Dark,
    /// Material design theme
    Material,
    /// Bootstrap theme
    Bootstrap,
    /// Minimal theme
    Minimal,
}

/// TypeScript node types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TSNodeType {
    Interface,
    Class,
    Function,
    Method,
    Property,
    Parameter,
    Enum,
    Type,
    Module,
    Namespace,
}

/// TypeScript definition node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSNode {
    pub name: String,
    pub node_type: TSNodeType,
    pub description: Option<String>,
    pub parameters: Vec<TSParameter>,
    pub return_type: Option<String>,
    pub properties: Vec<TSProperty>,
    pub examples: Vec<String>,
    pub deprecated: bool,
    pub since_version: Option<String>,
    pub file_path: String,
    pub line_number: u32,
}

/// TypeScript parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSParameter {
    pub name: String,
    pub param_type: String,
    pub optional: bool,
    pub default_value: Option<String>,
    pub description: Option<String>,
}

/// TypeScript property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSProperty {
    pub name: String,
    pub property_type: String,
    pub optional: bool,
    pub readonly: bool,
    pub description: Option<String>,
}

/// Documentation configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DocConfig {
    format: DocFormat,
    theme: DocTheme,
    include_examples: bool,
    include_source_links: bool,
    include_private: bool,
    generate_playground: bool,
    output_directory: String,
    base_url: String,
    title: String,
    version: String,
}

impl Default for DocConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl DocConfig {
    /// Create a new documentation configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            format: DocFormat::HTML,
            theme: DocTheme::Default,
            include_examples: true,
            include_source_links: true,
            include_private: false,
            generate_playground: true,
            output_directory: "./docs".to_string(),
            base_url: "/".to_string(),
            title: "TrustFormer WASM API".to_string(),
            version: "1.0.0".to_string(),
        }
    }

    /// Set documentation format
    pub fn set_format(&mut self, format: DocFormat) {
        self.format = format;
    }

    /// Set documentation theme
    pub fn set_theme(&mut self, theme: DocTheme) {
        self.theme = theme;
    }

    /// Enable/disable examples
    pub fn set_include_examples(&mut self, include: bool) {
        self.include_examples = include;
    }

    /// Enable/disable source links
    pub fn set_include_source_links(&mut self, include: bool) {
        self.include_source_links = include;
    }

    /// Enable/disable private member documentation
    pub fn set_include_private(&mut self, include: bool) {
        self.include_private = include;
    }

    /// Enable/disable playground generation
    pub fn set_generate_playground(&mut self, generate: bool) {
        self.generate_playground = generate;
    }

    /// Set output directory
    pub fn set_output_directory(&mut self, dir: String) {
        self.output_directory = dir;
    }

    /// Set base URL
    pub fn set_base_url(&mut self, url: String) {
        self.base_url = url;
    }

    /// Set documentation title
    pub fn set_title(&mut self, title: String) {
        self.title = title;
    }

    /// Set version
    pub fn set_version(&mut self, version: String) {
        self.version = version;
    }
}

/// Generated documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedDoc {
    pub title: String,
    pub version: String,
    pub generated_at: f64,
    pub format: DocFormat,
    pub content: String,
    pub assets: Vec<DocAsset>,
    pub navigation: DocNavigation,
}

/// Documentation asset (CSS, JS, images)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocAsset {
    pub filename: String,
    pub content: String,
    pub mime_type: String,
}

/// Documentation navigation structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocNavigation {
    pub sections: Vec<NavSection>,
}

/// Navigation section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavSection {
    pub title: String,
    pub items: Vec<NavItem>,
}

/// Navigation item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavItem {
    pub title: String,
    pub link: String,
    pub children: Vec<NavItem>,
}

/// Automatic documentation generator
#[wasm_bindgen]
pub struct AutoDocGenerator {
    config: DocConfig,
    ts_nodes: Vec<TSNode>,
    examples: Vec<CodeExample>,
}

/// Code example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    pub title: String,
    pub description: String,
    pub code: String,
    pub language: String,
    pub category: String,
}

#[wasm_bindgen]
impl AutoDocGenerator {
    /// Create a new documentation generator
    #[wasm_bindgen(constructor)]
    pub fn new(config: DocConfig) -> Self {
        Self {
            config,
            ts_nodes: Vec::new(),
            examples: Vec::new(),
        }
    }

    /// Parse TypeScript definitions from string
    pub fn parse_typescript(&mut self, typescript_content: &str) -> Result<(), JsValue> {
        // Simplified TypeScript parsing
        // In a real implementation, this would use a proper TypeScript parser

        let lines: Vec<&str> = typescript_content.lines().collect();
        let mut current_line = 0;

        while current_line < lines.len() {
            let line = lines[current_line].trim();

            if line.starts_with("export interface") {
                self.parse_interface(&lines, &mut current_line)?;
            } else if line.starts_with("export class") {
                self.parse_class(&lines, &mut current_line)?;
            } else if line.starts_with("export function")
                || line.starts_with("export declare function")
            {
                self.parse_function(&lines, &mut current_line)?;
            } else if line.starts_with("export enum") {
                self.parse_enum(&lines, &mut current_line)?;
            }

            current_line += 1;
        }

        Ok(())
    }

    /// Generate documentation in the configured format
    pub fn generate_documentation(&self) -> Result<String, JsValue> {
        match self.config.format {
            DocFormat::HTML => self.generate_html(),
            DocFormat::Markdown => self.generate_markdown(),
            DocFormat::JSON => self.generate_json(),
            DocFormat::OpenAPI => self.generate_openapi(),
            DocFormat::TypeDoc => self.generate_typedoc(),
        }
    }

    /// Add a code example
    pub fn add_example(&mut self, example_json: &str) -> Result<(), JsValue> {
        let example: CodeExample = serde_json::from_str(example_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid example: {e}")))?;

        self.examples.push(example);
        Ok(())
    }

    /// Generate navigation structure
    pub fn generate_navigation(&self) -> String {
        let nav = self.build_navigation();
        serde_json::to_string_pretty(&nav).unwrap_or_else(|_| "{}".to_string())
    }

    /// Generate API reference
    pub fn generate_api_reference(&self) -> String {
        let mut content = String::new();

        // Group nodes by type
        let interfaces: Vec<_> = self
            .ts_nodes
            .iter()
            .filter(|n| matches!(n.node_type, TSNodeType::Interface))
            .collect();
        let classes: Vec<_> = self
            .ts_nodes
            .iter()
            .filter(|n| matches!(n.node_type, TSNodeType::Class))
            .collect();
        let functions: Vec<_> = self
            .ts_nodes
            .iter()
            .filter(|n| matches!(n.node_type, TSNodeType::Function))
            .collect();
        let enums: Vec<_> = self
            .ts_nodes
            .iter()
            .filter(|n| matches!(n.node_type, TSNodeType::Enum))
            .collect();

        // Generate sections
        if !interfaces.is_empty() {
            content.push_str("# Interfaces\n\n");
            for interface in interfaces {
                content.push_str(&self.format_interface_doc(interface));
            }
        }

        if !classes.is_empty() {
            content.push_str("# Classes\n\n");
            for class in classes {
                content.push_str(&self.format_class_doc(class));
            }
        }

        if !functions.is_empty() {
            content.push_str("# Functions\n\n");
            for function in functions {
                content.push_str(&self.format_function_doc(function));
            }
        }

        if !enums.is_empty() {
            content.push_str("# Enumerations\n\n");
            for enum_node in enums {
                content.push_str(&self.format_enum_doc(enum_node));
            }
        }

        content
    }

    /// Generate playground documentation
    pub fn generate_playground(&self) -> String {
        if !self.config.generate_playground {
            return "Playground generation disabled".to_string();
        }

        let mut playground = String::new();

        playground.push_str(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustFormer WASM Playground</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/theme/monokai.min.css">
    <style>
        body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
        .playground { display: flex; height: 100vh; }
        .editor-panel { flex: 1; display: flex; flex-direction: column; }
        .output-panel { flex: 1; display: flex; flex-direction: column; border-left: 1px solid #ccc; }
        .toolbar { padding: 10px; background: #f5f5f5; border-bottom: 1px solid #ccc; }
        .editor { flex: 1; }
        .output { flex: 1; padding: 10px; overflow: auto; background: #1e1e1e; color: #fff; font-family: monospace; }
        button { padding: 8px 16px; margin-right: 8px; }
        select { padding: 8px; margin-right: 8px; }
    </style>
</head>
<body>
    <div class="playground">
        <div class="editor-panel">
            <div class="toolbar">
                <select id="example-select">
                    <option value="">Select an example...</option>
"#);

        // Add examples to playground
        for (i, example) in self.examples.iter().enumerate() {
            playground.push_str(&format!(
                r#"                    <option value="{}">{}</option>
"#,
                i, example.title
            ));
        }

        playground.push_str(r#"
                </select>
                <button onclick="runCode()">Run</button>
                <button onclick="clearOutput()">Clear</button>
            </div>
            <div class="editor">
                <textarea id="code-editor"></textarea>
            </div>
        </div>
        <div class="output-panel">
            <div class="toolbar">
                <strong>Output</strong>
            </div>
            <div class="output" id="output"></div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/mode/javascript/javascript.min.js"></script>
    <script>
        const examples = [
"#);

        // Add example data
        for example in &self.examples {
            playground.push_str(&format!(
                r#"            {{
                title: "{}",
                description: "{}",
                code: `{}`
            }},
"#,
                example.title,
                example.description,
                example.code.replace('`', "\\`")
            ));
        }

        playground.push_str(r#"
        ];

        const editor = CodeMirror.fromTextArea(document.getElementById('code-editor'), {
            mode: 'javascript',
            theme: 'monokai',
            lineNumbers: true,
            autoCloseBrackets: true,
            matchBrackets: true
        });

        document.getElementById('example-select').addEventListener('change', function(e) {
            const index = parseInt(e.target.value);
            if (!isNaN(index) && examples[index]) {
                editor.setValue(examples[index].code);
            }
        });

        function runCode() {
            const code = editor.getValue();
            const output = document.getElementById('output');

            try {
                // This would execute the code with the WASM module
                output.innerHTML += '> ' + code + '\n';
                output.innerHTML += 'Note: This is a demo playground. Real execution would require the WASM module.\n\n';
                output.scrollTop = output.scrollHeight;
            } catch (error) {
                output.innerHTML += 'Error: ' + error.message + '\n\n';
                output.scrollTop = output.scrollHeight;
            }
        }

        function clearOutput() {
            document.getElementById('output').innerHTML = '';
        }

        // Load first example by default
        if (examples.length > 0) {
            editor.setValue(examples[0].code);
        }
    </script>
</body>
</html>
"#);

        playground
    }

    /// Get statistics about the parsed TypeScript
    pub fn get_statistics(&self) -> String {
        let interface_count = self
            .ts_nodes
            .iter()
            .filter(|n| matches!(n.node_type, TSNodeType::Interface))
            .count();
        let class_count = self
            .ts_nodes
            .iter()
            .filter(|n| matches!(n.node_type, TSNodeType::Class))
            .count();
        let function_count = self
            .ts_nodes
            .iter()
            .filter(|n| matches!(n.node_type, TSNodeType::Function))
            .count();
        let enum_count =
            self.ts_nodes.iter().filter(|n| matches!(n.node_type, TSNodeType::Enum)).count();

        format!(
            r#"{{
  "total_nodes": {},
  "interfaces": {},
  "classes": {},
  "functions": {},
  "enums": {},
  "examples": {}
}}"#,
            self.ts_nodes.len(),
            interface_count,
            class_count,
            function_count,
            enum_count,
            self.examples.len()
        )
    }

    // Private parsing methods

    fn parse_interface(&mut self, lines: &[&str], current_line: &mut usize) -> Result<(), JsValue> {
        let line = lines[*current_line];

        // Extract interface name
        let name = self.extract_name_from_declaration(line, "interface")?;

        let mut properties = Vec::new();
        *current_line += 1;

        // Parse properties until closing brace
        while *current_line < lines.len() {
            let prop_line = lines[*current_line].trim();
            if prop_line == "}" {
                break;
            }

            if !prop_line.is_empty() && !prop_line.starts_with("//") {
                if let Ok(property) = self.parse_property(prop_line) {
                    properties.push(property);
                }
            }

            *current_line += 1;
        }

        let node = TSNode {
            name,
            node_type: TSNodeType::Interface,
            description: self
                .extract_description_comment(lines, *current_line - properties.len() - 1),
            parameters: Vec::new(),
            return_type: None,
            properties,
            examples: Vec::new(),
            deprecated: false,
            since_version: None,
            file_path: "parsed.ts".to_string(),
            line_number: *current_line as u32,
        };

        self.ts_nodes.push(node);
        Ok(())
    }

    fn parse_class(&mut self, lines: &[&str], current_line: &mut usize) -> Result<(), JsValue> {
        let line = lines[*current_line];
        let name = self.extract_name_from_declaration(line, "class")?;

        // For simplicity, treat class similar to interface
        // In a real implementation, this would parse methods, constructors, etc.
        let node = TSNode {
            name,
            node_type: TSNodeType::Class,
            description: self.extract_description_comment(lines, *current_line - 1),
            parameters: Vec::new(),
            return_type: None,
            properties: Vec::new(),
            examples: Vec::new(),
            deprecated: false,
            since_version: None,
            file_path: "parsed.ts".to_string(),
            line_number: *current_line as u32,
        };

        self.ts_nodes.push(node);
        Ok(())
    }

    fn parse_function(&mut self, lines: &[&str], current_line: &mut usize) -> Result<(), JsValue> {
        let line = lines[*current_line];
        let name = self.extract_name_from_declaration(line, "function")?;

        // Extract parameters and return type
        let (parameters, return_type) = self.parse_function_signature(line)?;

        let node = TSNode {
            name,
            node_type: TSNodeType::Function,
            description: self.extract_description_comment(lines, *current_line - 1),
            parameters,
            return_type,
            properties: Vec::new(),
            examples: Vec::new(),
            deprecated: false,
            since_version: None,
            file_path: "parsed.ts".to_string(),
            line_number: *current_line as u32,
        };

        self.ts_nodes.push(node);
        Ok(())
    }

    fn parse_enum(&mut self, lines: &[&str], current_line: &mut usize) -> Result<(), JsValue> {
        let line = lines[*current_line];
        let name = self.extract_name_from_declaration(line, "enum")?;

        let node = TSNode {
            name,
            node_type: TSNodeType::Enum,
            description: self.extract_description_comment(lines, *current_line - 1),
            parameters: Vec::new(),
            return_type: None,
            properties: Vec::new(),
            examples: Vec::new(),
            deprecated: false,
            since_version: None,
            file_path: "parsed.ts".to_string(),
            line_number: *current_line as u32,
        };

        self.ts_nodes.push(node);
        Ok(())
    }

    fn extract_name_from_declaration(&self, line: &str, keyword: &str) -> Result<String, JsValue> {
        let parts: Vec<&str> = line.split_whitespace().collect();

        for i in 0..parts.len() {
            if parts[i] == keyword && i + 1 < parts.len() {
                let name = parts[i + 1];
                // Remove any generic parameters or extends clauses
                let clean_name =
                    name.split('<').next().unwrap_or(name).split('(').next().unwrap_or(name);
                return Ok(clean_name.to_string());
            }
        }

        Err(JsValue::from_str(&format!(
            "Could not extract name from {} declaration",
            keyword
        )))
    }

    fn parse_property(&self, line: &str) -> Result<TSProperty, JsValue> {
        // Simplified property parsing
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() != 2 {
            return Err(JsValue::from_str("Invalid property syntax"));
        }

        let name_part = parts[0].trim();
        let type_part = parts[1].trim().trim_end_matches(';');

        let optional = name_part.ends_with('?');
        let readonly = name_part.starts_with("readonly ");

        let name = name_part
            .trim_start_matches("readonly ")
            .trim_end_matches('?')
            .trim()
            .to_string();

        Ok(TSProperty {
            name,
            property_type: type_part.to_string(),
            optional,
            readonly,
            description: None,
        })
    }

    fn parse_function_signature(
        &self,
        _line: &str,
    ) -> Result<(Vec<TSParameter>, Option<String>), JsValue> {
        // Simplified function signature parsing
        let parameters = Vec::new(); // Would parse actual parameters
        let return_type = Some("any".to_string()); // Would parse actual return type

        Ok((parameters, return_type))
    }

    fn extract_description_comment(&self, lines: &[&str], line_index: usize) -> Option<String> {
        if line_index == 0 {
            return None;
        }

        let comment_line = lines[line_index - 1].trim();
        if comment_line.starts_with("//") {
            Some(comment_line.trim_start_matches("//").trim().to_string())
        } else {
            None
        }
    }

    // Documentation generation methods

    fn generate_html(&self) -> Result<String, JsValue> {
        let mut html = String::new();

        html.push_str(&format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    {}
</head>
<body>
    <div class="container">
        <header>
            <h1>{}</h1>
            <p>Version: {}</p>
        </header>
        <nav>
            {}
        </nav>
        <main>
            {}
        </main>
    </div>
    {}
</body>
</html>
"#,
            self.config.title,
            self.get_html_styles(),
            self.config.title,
            self.config.version,
            self.generate_html_navigation(),
            self.generate_api_reference(),
            self.get_html_scripts()
        ));

        Ok(html)
    }

    fn generate_markdown(&self) -> Result<String, JsValue> {
        let mut md = String::new();

        md.push_str(&format!("# {title}\n\n", title = self.config.title));
        md.push_str(&format!(
            "Version: {version}\n\n",
            version = self.config.version
        ));
        md.push_str(&self.generate_api_reference());

        Ok(md)
    }

    fn generate_json(&self) -> Result<String, JsValue> {
        let doc = GeneratedDoc {
            title: self.config.title.clone(),
            version: self.config.version.clone(),
            generated_at: js_sys::Date::now(),
            format: self.config.format,
            content: self.generate_api_reference(),
            assets: Vec::new(),
            navigation: self.build_navigation(),
        };

        serde_json::to_string_pretty(&doc).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    fn generate_openapi(&self) -> Result<String, JsValue> {
        // Generate OpenAPI specification
        let mut spec = String::new();
        spec.push_str("openapi: 3.0.0\n");
        spec.push_str(&format!(
            "info:\n  title: {}\n  version: {}\n",
            self.config.title, self.config.version
        ));
        spec.push_str("paths:\n");

        // Add API endpoints based on functions
        for node in &self.ts_nodes {
            if matches!(node.node_type, TSNodeType::Function) {
                spec.push_str(&format!(
                    "  /{}:\n    post:\n      summary: {}\n",
                    node.name,
                    node.description.as_deref().unwrap_or(&node.name)
                ));
            }
        }

        Ok(spec)
    }

    fn generate_typedoc(&self) -> Result<String, JsValue> {
        // Generate TypeDoc compatible JSON
        self.generate_json()
    }

    fn build_navigation(&self) -> DocNavigation {
        let mut sections = Vec::new();

        // Group by node type
        let interface_items: Vec<_> = self
            .ts_nodes
            .iter()
            .filter(|n| matches!(n.node_type, TSNodeType::Interface))
            .map(|n| NavItem {
                title: n.name.clone(),
                link: format!("#{name}", name = n.name.to_lowercase()),
                children: Vec::new(),
            })
            .collect();

        if !interface_items.is_empty() {
            sections.push(NavSection {
                title: "Interfaces".to_string(),
                items: interface_items,
            });
        }

        let class_items: Vec<_> = self
            .ts_nodes
            .iter()
            .filter(|n| matches!(n.node_type, TSNodeType::Class))
            .map(|n| NavItem {
                title: n.name.clone(),
                link: format!("#{name}", name = n.name.to_lowercase()),
                children: Vec::new(),
            })
            .collect();

        if !class_items.is_empty() {
            sections.push(NavSection {
                title: "Classes".to_string(),
                items: class_items,
            });
        }

        DocNavigation { sections }
    }

    fn format_interface_doc(&self, interface: &TSNode) -> String {
        let mut doc = String::new();
        doc.push_str(&format!("## {name}\n\n", name = interface.name));

        if let Some(ref description) = interface.description {
            doc.push_str(&format!("{description}\n\n"));
        }

        if !interface.properties.is_empty() {
            doc.push_str("### Properties\n\n");
            for prop in &interface.properties {
                doc.push_str(&format!(
                    "- **{}**{}: `{}` {}\n",
                    prop.name,
                    if prop.optional { "?" } else { "" },
                    prop.property_type,
                    if prop.readonly { "(readonly)" } else { "" }
                ));

                if let Some(ref desc) = prop.description {
                    doc.push_str(&format!("  {desc}\n"));
                }
            }
            doc.push('\n');
        }

        doc
    }

    fn format_class_doc(&self, class: &TSNode) -> String {
        let mut doc = String::new();
        doc.push_str(&format!("## {name}\n\n", name = class.name));

        if let Some(ref description) = class.description {
            doc.push_str(&format!("{description}\n\n"));
        }

        doc
    }

    fn format_function_doc(&self, function: &TSNode) -> String {
        let mut doc = String::new();
        doc.push_str(&format!("## {name}\n\n", name = function.name));

        if let Some(ref description) = function.description {
            doc.push_str(&format!("{description}\n\n"));
        }

        if !function.parameters.is_empty() {
            doc.push_str("### Parameters\n\n");
            for param in &function.parameters {
                doc.push_str(&format!(
                    "- **{}**{}: `{}`\n",
                    param.name,
                    if param.optional { "?" } else { "" },
                    param.param_type
                ));

                if let Some(ref desc) = param.description {
                    doc.push_str(&format!("  {desc}\n"));
                }
            }
            doc.push('\n');
        }

        if let Some(ref return_type) = function.return_type {
            doc.push_str(&format!("### Returns\n\n`{return_type}`\n\n"));
        }

        doc
    }

    fn format_enum_doc(&self, enum_node: &TSNode) -> String {
        let mut doc = String::new();
        doc.push_str(&format!("## {name}\n\n", name = enum_node.name));

        if let Some(ref description) = enum_node.description {
            doc.push_str(&format!("{description}\n\n"));
        }

        doc
    }

    fn generate_html_navigation(&self) -> String {
        let nav = self.build_navigation();
        let mut html = String::new();

        html.push_str("<ul>");
        for section in &nav.sections {
            html.push_str(&format!(
                "<li><strong>{title}</strong><ul>",
                title = section.title
            ));
            for item in &section.items {
                html.push_str(&format!(
                    "<li><a href=\"{}\">{}</a></li>",
                    item.link, item.title
                ));
            }
            html.push_str("</ul></li>");
        }
        html.push_str("</ul>");

        html
    }

    fn get_html_styles(&self) -> String {
        match self.config.theme {
            DocTheme::Default => self.get_default_styles(),
            DocTheme::Dark => self.get_dark_styles(),
            DocTheme::Material => self.get_material_styles(),
            DocTheme::Bootstrap => self.get_bootstrap_styles(),
            DocTheme::Minimal => self.get_minimal_styles(),
        }
    }

    fn get_default_styles(&self) -> String {
        r#"
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #fff; }
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    header { text-align: center; margin-bottom: 2rem; }
    nav { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }
    nav ul { list-style: none; padding: 0; }
    nav li { margin: 0.5rem 0; }
    nav a { text-decoration: none; color: #0366d6; }
    h1, h2, h3 { color: #24292e; }
    code { background: #f6f8fa; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
    pre { background: #f6f8fa; padding: 1rem; border-radius: 6px; overflow-x: auto; }
</style>
"#.to_string()
    }

    fn get_dark_styles(&self) -> String {
        r#"
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #0d1117; color: #c9d1d9; }
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    header { text-align: center; margin-bottom: 2rem; }
    nav { background: #161b22; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; border: 1px solid #30363d; }
    nav ul { list-style: none; padding: 0; }
    nav li { margin: 0.5rem 0; }
    nav a { text-decoration: none; color: #58a6ff; }
    h1, h2, h3 { color: #f0f6fc; }
    code { background: #161b22; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
    pre { background: #161b22; padding: 1rem; border-radius: 6px; overflow-x: auto; border: 1px solid #30363d; }
</style>
"#.to_string()
    }

    fn get_material_styles(&self) -> String {
        r#"
<style>
    body { font-family: 'Roboto', sans-serif; margin: 0; background: #fafafa; }
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    header { text-align: center; margin-bottom: 2rem; background: #2196f3; color: white; padding: 2rem; border-radius: 4px; }
    nav { background: white; padding: 1rem; border-radius: 4px; margin-bottom: 2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    nav ul { list-style: none; padding: 0; }
    nav li { margin: 0.5rem 0; }
    nav a { text-decoration: none; color: #2196f3; }
    h1, h2, h3 { color: #212121; }
    code { background: #e8eaf6; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
    pre { background: #e8eaf6; padding: 1rem; border-radius: 4px; overflow-x: auto; }
</style>
"#.to_string()
    }

    fn get_bootstrap_styles(&self) -> String {
        r#"
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
    .container { margin-top: 2rem; }
    nav { background: #f8f9fa; padding: 1rem; border-radius: 0.375rem; margin-bottom: 2rem; }
    code { color: #e83e8c; }
</style>
"#.to_string()
    }

    fn get_minimal_styles(&self) -> String {
        r#"
<style>
    body { font-family: Georgia, serif; margin: 0; background: white; line-height: 1.6; }
    .container { max-width: 800px; margin: 0 auto; padding: 20px; }
    header { margin-bottom: 2rem; }
    nav { margin-bottom: 2rem; }
    nav ul { list-style: none; padding: 0; }
    nav a { text-decoration: none; color: #000; border-bottom: 1px dotted; }
    h1, h2, h3 { font-weight: normal; }
    code { font-family: 'Courier New', monospace; }
</style>
"#
        .to_string()
    }

    fn get_html_scripts(&self) -> String {
        r##"
<script>
    // Add interactive features
    document.addEventListener('DOMContentLoaded', function() {
        // Smooth scrolling for navigation links
        const links = document.querySelectorAll('nav a[href^="#"]');
        links.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            });
        });
    });
</script>
"##
        .to_string()
    }
}

/// Create a documentation generator with default settings
#[wasm_bindgen]
pub fn create_default_doc_generator() -> AutoDocGenerator {
    AutoDocGenerator::new(DocConfig::new())
}

/// Create a documentation generator for HTML output
#[wasm_bindgen]
pub fn create_html_doc_generator() -> AutoDocGenerator {
    let mut config = DocConfig::new();
    config.set_format(DocFormat::HTML);
    AutoDocGenerator::new(config)
}

/// Create a documentation generator for Markdown output
#[wasm_bindgen]
pub fn create_markdown_doc_generator() -> AutoDocGenerator {
    let mut config = DocConfig::new();
    config.set_format(DocFormat::Markdown);
    AutoDocGenerator::new(config)
}

/// Version and build information for the trustformers-wasm crate
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    /// Crate version
    version: String,
    /// Git commit hash (if available)
    git_hash: String,
    /// Build target
    target: String,
    /// Build profile (debug/release)
    profile: String,
    /// Build timestamp
    build_time: String,
    /// Enabled features
    features: Vec<String>,
}

#[wasm_bindgen]
impl VersionInfo {
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> String {
        self.version.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn git_hash(&self) -> String {
        self.git_hash.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        self.target.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn profile(&self) -> String {
        self.profile.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn build_time(&self) -> String {
        self.build_time.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn features(&self) -> Array {
        self.features.iter().map(|f| JsValue::from(f.as_str())).collect()
    }
}

/// Get version and build information for the trustformers-wasm crate
#[wasm_bindgen]
pub fn get_version_info() -> VersionInfo {
    VersionInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        git_hash: option_env!("GIT_HASH").unwrap_or("unknown").to_string(),
        target: std::env::consts::ARCH.to_string() + "-" + std::env::consts::OS,
        profile: if cfg!(debug_assertions) { "debug".to_string() } else { "release".to_string() },
        build_time: option_env!("BUILD_TIME").unwrap_or("unknown").to_string(),
        features: get_enabled_features(),
    }
}

/// Get list of enabled Cargo features
#[allow(clippy::vec_init_then_push)] // Needed for cfg-gated feature collection
fn get_enabled_features() -> Vec<String> {
    let mut features = Vec::new();

    #[cfg(feature = "console_panic")]
    features.push("console_panic".to_string());

    #[cfg(feature = "webgpu")]
    features.push("webgpu".to_string());

    #[cfg(feature = "web-workers")]
    features.push("web-workers".to_string());

    #[cfg(feature = "shared-memory")]
    features.push("shared-memory".to_string());

    #[cfg(feature = "kernel-fusion")]
    features.push("kernel-fusion".to_string());

    #[cfg(feature = "async-executor")]
    features.push("async-executor".to_string());

    #[cfg(feature = "indexeddb")]
    features.push("indexeddb".to_string());

    #[cfg(feature = "memory64")]
    features.push("memory64".to_string());

    #[cfg(feature = "streaming-loader")]
    features.push("streaming-loader".to_string());

    #[cfg(feature = "model-splitting")]
    features.push("model-splitting".to_string());

    #[cfg(feature = "react-components")]
    features.push("react-components".to_string());

    #[cfg(feature = "vue-components")]
    features.push("vue-components".to_string());

    #[cfg(feature = "angular-components")]
    features.push("angular-components".to_string());

    #[cfg(feature = "web-components")]
    features.push("web-components".to_string());

    #[cfg(feature = "playground")]
    features.push("playground".to_string());

    #[cfg(feature = "streaming-generation")]
    features.push("streaming-generation".to_string());

    #[cfg(feature = "mobile-optimization")]
    features.push("mobile-optimization".to_string());

    #[cfg(feature = "wee-alloc")]
    features.push("wee-alloc".to_string());

    #[cfg(feature = "dlmalloc-alloc")]
    features.push("dlmalloc-alloc".to_string());

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_config() {
        let mut config = DocConfig::new();
        assert_eq!(config.format, DocFormat::HTML);
        assert_eq!(config.theme, DocTheme::Default);

        config.set_format(DocFormat::Markdown);
        assert_eq!(config.format, DocFormat::Markdown);
    }

    #[test]
    fn test_typescript_parsing() {
        let mut generator = AutoDocGenerator::new(DocConfig::new());

        let typescript = r#"
export interface TestInterface {
  name: string;
  optional?: number;
}
        "#;

        let result = generator.parse_typescript(typescript);
        assert!(result.is_ok());
        assert_eq!(generator.ts_nodes.len(), 1);
        assert_eq!(generator.ts_nodes[0].name, "TestInterface");
    }

    #[test]
    fn test_documentation_generation() {
        let generator = AutoDocGenerator::new(DocConfig::new());
        let doc = generator.generate_documentation();
        assert!(doc.is_ok());
    }
}
