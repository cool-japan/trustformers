//! Markdown documentation generator for FFI interfaces
//!
//! Generates comprehensive human-readable Markdown documentation
//! for the C FFI API, including functions, structs, enums, and examples.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::codegen::ast::{
    FfiEnum, FfiEnumVariant, FfiField, FfiFunction, FfiInterface, FfiParameter, FfiStruct, FfiType,
    PrimitiveType,
};
use crate::codegen::templates::TemplateEngine;
use crate::codegen::{CodeGenConfig, TargetLanguage};
use crate::error::TrustformersResult;

use super::LanguageGenerator;

/// Markdown documentation generator
pub struct MarkdownDocGenerator {
    config: CodeGenConfig,
}

impl MarkdownDocGenerator {
    /// Create a new Markdown documentation generator
    pub fn new(config: &CodeGenConfig) -> TrustformersResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Generate table of contents with links to all sections
    fn generate_table_of_contents(&self, interface: &FfiInterface) -> String {
        let mut toc = String::from("## Table of Contents\n\n");

        // Main sections
        toc.push_str("- [Overview](#overview)\n");
        toc.push_str("- [Installation](#installation)\n");
        toc.push_str("- [Error Handling](#error-handling)\n");

        // Enums section
        if !interface.enums.is_empty() {
            toc.push_str("- [Enumerations](#enumerations)\n");
            for enum_type in &interface.enums {
                let anchor = enum_type.name.to_lowercase().replace('_', "-");
                toc.push_str(&format!("  - [{}](#{})\n", enum_type.name, anchor));
            }
        }

        // Structs section
        if !interface.structs.is_empty() {
            toc.push_str("- [Structures](#structures)\n");
            for struct_type in &interface.structs {
                let anchor = struct_type.name.to_lowercase().replace('_', "-");
                toc.push_str(&format!("  - [{}](#{})\n", struct_type.name, anchor));
            }
        }

        // Functions section
        if !interface.functions.is_empty() {
            toc.push_str("- [Functions](#functions)\n");

            // Group functions by category
            let categories = self.categorize_functions(&interface.functions);
            for category in categories.keys() {
                let anchor = category.to_lowercase().replace(' ', "-");
                toc.push_str(&format!("  - [{}](#{})\n", category, anchor));
            }
        }

        toc.push_str("- [Examples](#examples)\n");
        toc.push_str("- [Version Information](#version-information)\n");
        toc.push_str("- [License](#license)\n");
        toc.push('\n');

        toc
    }

    /// Group functions by category based on naming patterns
    fn categorize_functions(
        &self,
        functions: &[FfiFunction],
    ) -> HashMap<String, Vec<&FfiFunction>> {
        let mut categories: HashMap<String, Vec<&FfiFunction>> = HashMap::new();

        for func in functions {
            let category = self.infer_category(&func.name);
            categories.entry(category).or_default().push(func);
        }

        categories
    }

    /// Infer function category from name
    fn infer_category(&self, name: &str) -> String {
        let lower = name.to_lowercase();

        if lower.contains("model") && (lower.contains("load") || lower.contains("create")) {
            "Model Management".to_string()
        } else if lower.contains("tokenizer") {
            "Tokenization".to_string()
        } else if lower.contains("pipeline") {
            "Pipeline Operations".to_string()
        } else if lower.contains("tensor") {
            "Tensor Operations".to_string()
        } else if lower.contains("inference")
            || lower.contains("predict")
            || lower.contains("forward")
        {
            "Inference".to_string()
        } else if lower.contains("quantiz") {
            "Quantization".to_string()
        } else if lower.contains("device") || lower.contains("cuda") || lower.contains("gpu") {
            "Device Management".to_string()
        } else if lower.contains("error") || lower.contains("free") || lower.contains("destroy") {
            "Memory and Error Management".to_string()
        } else if lower.contains("config") || lower.contains("set") || lower.contains("get") {
            "Configuration".to_string()
        } else {
            "Miscellaneous".to_string()
        }
    }

    /// Generate documentation for enums
    fn generate_enum_docs(&self, enum_type: &FfiEnum) -> String {
        let mut doc = String::new();

        // Enum header
        let anchor = enum_type.name.to_lowercase().replace('_', "-");
        doc.push_str(&format!("### {}\n\n", enum_type.name));
        doc.push_str(&format!("<a id=\"{}\"></a>\n\n", anchor));

        // Documentation
        if !enum_type.documentation.is_empty() {
            for doc_line in &enum_type.documentation {
                doc.push_str(&format!("{}\n", doc_line));
            }
            doc.push('\n');
        }

        // Deprecation notice
        if let Some(deprecation) = &enum_type.deprecation {
            doc.push_str(&format!("> **⚠️ Deprecated**: {}\n", deprecation.message));
            if let Some(replacement) = &deprecation.replacement {
                doc.push_str(&format!("> Use `{}` instead.\n", replacement));
            }
            doc.push_str("\n");
        }

        // Enum properties
        doc.push_str("**Properties:**\n\n");
        doc.push_str(&format!(
            "- **Underlying Type**: `{:?}`\n",
            enum_type.underlying_type
        ));
        doc.push_str(&format!("- **Flags Enum**: {}\n", enum_type.is_flags));
        doc.push('\n');

        // Variants table
        doc.push_str("**Variants:**\n\n");
        doc.push_str("| Variant | Value | Description |\n");
        doc.push_str("|---------|-------|-------------|\n");

        for variant in &enum_type.variants {
            let description = if !variant.documentation.is_empty() {
                variant.documentation.join(" ")
            } else {
                String::from("-")
            };

            let variant_name = if variant.deprecation.is_some() {
                format!("~~{}~~", variant.name)
            } else {
                variant.name.clone()
            };

            doc.push_str(&format!(
                "| `{}` | `{}` | {} |\n",
                variant_name, variant.value, description
            ));
        }
        doc.push('\n');

        // Usage example
        doc.push_str("**Example:**\n\n");
        doc.push_str("```c\n");
        doc.push_str(&format!(
            "{} value = {};\n",
            enum_type.c_name, enum_type.variants[0].c_name
        ));
        doc.push_str("```\n\n");

        doc
    }

    /// Generate documentation for structs
    fn generate_struct_docs(&self, struct_type: &FfiStruct) -> String {
        let mut doc = String::new();

        // Struct header
        let anchor = struct_type.name.to_lowercase().replace('_', "-");
        doc.push_str(&format!("### {}\n\n", struct_type.name));
        doc.push_str(&format!("<a id=\"{}\"></a>\n\n", anchor));

        // Documentation
        if !struct_type.documentation.is_empty() {
            for doc_line in &struct_type.documentation {
                doc.push_str(&format!("{}\n", doc_line));
            }
            doc.push('\n');
        }

        // Deprecation notice
        if let Some(deprecation) = &struct_type.deprecation {
            doc.push_str(&format!("> **⚠️ Deprecated**: {}\n", deprecation.message));
            if let Some(replacement) = &deprecation.replacement {
                doc.push_str(&format!("> Use `{}` instead.\n", replacement));
            }
            doc.push_str("\n");
        }

        // Struct properties
        doc.push_str("**Properties:**\n\n");
        doc.push_str(&format!("- **Opaque**: {}\n", struct_type.is_opaque));
        doc.push_str(&format!("- **Packed**: {}\n", struct_type.is_packed));
        if let Some(alignment) = struct_type.alignment {
            doc.push_str(&format!("- **Alignment**: {} bytes\n", alignment));
        }
        doc.push_str(&format!(
            "- **Estimated Size**: {} bytes\n",
            struct_type.estimated_size()
        ));
        doc.push('\n');

        // Fields table (if not opaque)
        if !struct_type.is_opaque && !struct_type.fields.is_empty() {
            doc.push_str("**Fields:**\n\n");
            doc.push_str("| Field | Type | Description |\n");
            doc.push_str("|-------|------|-------------|\n");

            for field in &struct_type.fields {
                let description = if !field.documentation.is_empty() {
                    field.documentation.join(" ")
                } else {
                    String::from("-")
                };

                let field_name = if field.is_private {
                    format!("_{}_ (private)", field.name)
                } else {
                    field.name.clone()
                };

                doc.push_str(&format!(
                    "| `{}` | `{}` | {} |\n",
                    field_name,
                    self.format_type(&field.type_info),
                    description
                ));
            }
            doc.push('\n');
        } else if struct_type.is_opaque {
            doc.push_str("> This is an opaque structure. Internal fields are not accessible.\n\n");
        }

        // Platform/feature requirements
        if !struct_type.required_features.is_empty() {
            doc.push_str("**Required Features:**\n\n");
            for feature in &struct_type.required_features {
                doc.push_str(&format!("- `{}`\n", feature));
            }
            doc.push('\n');
        }

        if !struct_type.platforms.is_empty() {
            doc.push_str("**Platform Availability:**\n\n");
            for platform in &struct_type.platforms {
                doc.push_str(&format!("- {}\n", platform));
            }
            doc.push('\n');
        }

        doc
    }

    /// Generate documentation for functions
    fn generate_function_docs(&self, func: &FfiFunction) -> String {
        let mut doc = String::new();

        // Function header
        let anchor = func.name.to_lowercase().replace('_', "-");
        doc.push_str(&format!("#### {}\n\n", func.name));
        doc.push_str(&format!("<a id=\"{}\"></a>\n\n", anchor));

        // Documentation
        if !func.documentation.is_empty() {
            for doc_line in &func.documentation {
                doc.push_str(&format!("{}\n", doc_line));
            }
            doc.push('\n');
        }

        // Deprecation notice
        if let Some(deprecation) = &func.deprecation {
            doc.push_str(&format!("> **⚠️ Deprecated**: {}\n", deprecation.message));
            if let Some(replacement) = &deprecation.replacement {
                doc.push_str(&format!("> Use `{}` instead.\n", replacement));
            }
            doc.push_str("\n");
        }

        // Function signature
        doc.push_str("**Signature:**\n\n");
        doc.push_str("```c\n");
        doc.push_str(&format!(
            "{} {}(",
            self.format_type(&func.return_type),
            func.c_name
        ));

        if func.parameters.is_empty() {
            doc.push_str("void");
        } else {
            doc.push_str("\n");
            for (i, param) in func.parameters.iter().enumerate() {
                let comma = if i < func.parameters.len() - 1 { "," } else { "" };
                doc.push_str(&format!(
                    "    {} {}{}",
                    self.format_type(&param.type_info),
                    param.name,
                    comma
                ));
                doc.push('\n');
            }
        }
        doc.push_str(");\n");
        doc.push_str("```\n\n");

        // Parameters table
        if !func.parameters.is_empty() {
            doc.push_str("**Parameters:**\n\n");
            doc.push_str("| Parameter | Type | Description |\n");
            doc.push_str("|-----------|------|-------------|\n");

            for param in &func.parameters {
                let description = if !param.documentation.is_empty() {
                    param.documentation.join(" ")
                } else {
                    String::from("-")
                };

                let param_name = if param.is_optional {
                    format!("{} (optional)", param.name)
                } else {
                    param.name.clone()
                };

                doc.push_str(&format!(
                    "| `{}` | `{}` | {} |\n",
                    param_name,
                    self.format_type(&param.type_info),
                    description
                ));
            }
            doc.push('\n');
        }

        // Return value
        doc.push_str("**Returns:**\n\n");
        if func.return_type.name == "void" || func.return_type.name == "()" {
            doc.push_str("This function does not return a value.\n\n");
        } else if func.can_fail() {
            doc.push_str(&format!(
                "`{}` - Success code (`0`) on success, error code on failure. See [Error Handling](#error-handling) for details.\n\n",
                self.format_type(&func.return_type)
            ));
        } else {
            doc.push_str(&format!("`{}`\n\n", self.format_type(&func.return_type)));
        }

        // Safety notes
        if func.is_unsafe {
            doc.push_str("**Safety:**\n\n");
            doc.push_str(
                "> ⚠️ This function is marked as unsafe. Ensure all preconditions are met:\n",
            );
            doc.push_str("> - All pointers must be valid and properly aligned\n");
            doc.push_str("> - Lifetimes must be managed correctly\n");
            doc.push_str("> - Thread safety requirements must be respected\n\n");
        }

        // Platform/feature requirements
        if !func.required_features.is_empty() {
            doc.push_str("**Required Features:**\n\n");
            for feature in &func.required_features {
                doc.push_str(&format!("- `{}`\n", feature));
            }
            doc.push('\n');
        }

        if !func.platforms.is_empty() {
            doc.push_str("**Platform Availability:**\n\n");
            for platform in &func.platforms {
                doc.push_str(&format!("- {}\n", platform));
            }
            doc.push('\n');
        }

        // Example usage
        doc.push_str("**Example:**\n\n");
        doc.push_str("```c\n");
        doc.push_str(&self.generate_function_example(func));
        doc.push_str("```\n\n");

        doc.push_str("---\n\n");

        doc
    }

    /// Generate a simple usage example for a function
    fn generate_function_example(&self, func: &FfiFunction) -> String {
        let mut example = String::new();

        // Generate parameter initialization
        for param in &func.parameters {
            if param.type_info.is_pointer() && !param.type_info.is_const() {
                example.push_str(&format!(
                    "{} {};\n",
                    self.format_type(&param.type_info),
                    param.name
                ));
            } else if param.type_info.is_string() {
                example.push_str(&format!("const char* {} = \"example\";\n", param.name));
            } else {
                example.push_str(&format!(
                    "{} {} = /* value */;\n",
                    self.format_type(&param.type_info),
                    param.name
                ));
            }
        }

        if !func.parameters.is_empty() {
            example.push('\n');
        }

        // Generate function call
        if func.return_type.name != "void" && func.return_type.name != "()" {
            example.push_str(&format!(
                "{} result = ",
                self.format_type(&func.return_type)
            ));
        }

        example.push_str(&format!("{}(", func.c_name));

        if !func.parameters.is_empty() {
            let params: Vec<String> = func
                .parameters
                .iter()
                .map(|p| {
                    if p.type_info.is_pointer() && !p.type_info.is_const() {
                        format!("&{}", p.name)
                    } else {
                        p.name.clone()
                    }
                })
                .collect();
            example.push_str(&params.join(", "));
        }

        example.push_str(");\n");

        // Add error checking for functions that can fail
        if func.can_fail() {
            example.push_str("\nif (result != TRUSTFORMERS_SUCCESS) {\n");
            example.push_str("    // Handle error\n");
            example.push_str("    fprintf(stderr, \"Error: %d\\n\", result);\n");
            example.push_str("}\n");
        }

        example
    }

    /// Format a type for display in documentation
    fn format_type(&self, type_info: &FfiType) -> String {
        if type_info.is_pointer() {
            let const_qualifier = if type_info.is_const { "const " } else { "" };
            let mut_qualifier =
                if type_info.is_mutable && !type_info.is_const { "mut " } else { "" };

            let base = type_info.base_type();
            let stars = "*".repeat(type_info.pointer_level as usize);

            format!("{}{}{}{}", const_qualifier, mut_qualifier, stars, base)
        } else if let Some(array_size) = type_info.array_size {
            format!("{}[{}]", type_info.name, array_size)
        } else {
            type_info.name.clone()
        }
    }

    /// Generate error handling section
    fn generate_error_handling_section(&self, interface: &FfiInterface) -> String {
        let mut doc = String::from("## Error Handling\n\n");

        doc.push_str(
            "Most functions in this API return error codes to indicate success or failure. \n",
        );
        doc.push_str("The error codes are defined as follows:\n\n");

        // Find TrustformersError enum if it exists
        let error_enum = interface.enums.iter().find(|e| e.name.contains("Error"));

        if let Some(error_enum) = error_enum {
            doc.push_str("| Error Code | Value | Description |\n");
            doc.push_str("|------------|-------|-------------|\n");

            for variant in &error_enum.variants {
                let description = if !variant.documentation.is_empty() {
                    variant.documentation.join(" ")
                } else {
                    String::from("-")
                };

                doc.push_str(&format!(
                    "| `{}` | `{}` | {} |\n",
                    variant.name, variant.value, description
                ));
            }
            doc.push('\n');
        } else {
            doc.push_str(
                "Error codes are returned as negative integers. `0` indicates success.\n\n",
            );
        }

        doc.push_str("**Error Handling Pattern:**\n\n");
        doc.push_str("```c\n");
        doc.push_str("int result = trustformers_some_function(...);\n");
        doc.push_str("if (result != 0) {\n");
        doc.push_str("    // Handle error\n");
        doc.push_str("    const char* error_message = trustformers_get_last_error();\n");
        doc.push_str("    fprintf(stderr, \"Error: %s\\n\", error_message);\n");
        doc.push_str("    return result;\n");
        doc.push_str("}\n");
        doc.push_str("```\n\n");

        doc
    }

    /// Generate examples section
    fn generate_examples_section(&self) -> String {
        let mut doc = String::from("## Examples\n\n");

        doc.push_str("### Basic Text Generation\n\n");
        doc.push_str("```c\n");
        doc.push_str("#include <stdio.h>\n");
        doc.push_str("#include \"trustformers.h\"\n\n");
        doc.push_str("int main() {\n");
        doc.push_str("    // Load model\n");
        doc.push_str("    TrustformersModel* model;\n");
        doc.push_str("    int result = trustformers_model_from_pretrained(\n");
        doc.push_str("        \"gpt2\",\n");
        doc.push_str("        &model\n");
        doc.push_str("    );\n\n");
        doc.push_str("    if (result != 0) {\n");
        doc.push_str("        fprintf(stderr, \"Failed to load model\\n\");\n");
        doc.push_str("        return 1;\n");
        doc.push_str("    }\n\n");
        doc.push_str("    // Load tokenizer\n");
        doc.push_str("    TrustformersTokenizer* tokenizer;\n");
        doc.push_str("    result = trustformers_tokenizer_from_pretrained(\n");
        doc.push_str("        \"gpt2\",\n");
        doc.push_str("        &tokenizer\n");
        doc.push_str("    );\n\n");
        doc.push_str("    if (result != 0) {\n");
        doc.push_str("        trustformers_model_free(model);\n");
        doc.push_str("        fprintf(stderr, \"Failed to load tokenizer\\n\");\n");
        doc.push_str("        return 1;\n");
        doc.push_str("    }\n\n");
        doc.push_str("    // Generate text\n");
        doc.push_str("    const char* prompt = \"Once upon a time\";\n");
        doc.push_str("    char* output;\n");
        doc.push_str("    result = trustformers_generate_text(\n");
        doc.push_str("        model,\n");
        doc.push_str("        tokenizer,\n");
        doc.push_str("        prompt,\n");
        doc.push_str("        100, // max_length\n");
        doc.push_str("        &output\n");
        doc.push_str("    );\n\n");
        doc.push_str("    if (result == 0) {\n");
        doc.push_str("        printf(\"Generated: %s\\n\", output);\n");
        doc.push_str("        trustformers_string_free(output);\n");
        doc.push_str("    }\n\n");
        doc.push_str("    // Cleanup\n");
        doc.push_str("    trustformers_tokenizer_free(tokenizer);\n");
        doc.push_str("    trustformers_model_free(model);\n\n");
        doc.push_str("    return 0;\n");
        doc.push_str("}\n");
        doc.push_str("```\n\n");

        doc.push_str("### Using Pipelines\n\n");
        doc.push_str("```c\n");
        doc.push_str("#include \"trustformers.h\"\n\n");
        doc.push_str("int main() {\n");
        doc.push_str("    // Create pipeline\n");
        doc.push_str("    TrustformersPipeline* pipeline;\n");
        doc.push_str("    int result = trustformers_pipeline_create(\n");
        doc.push_str("        \"text-generation\",\n");
        doc.push_str("        \"gpt2\",\n");
        doc.push_str("        &pipeline\n");
        doc.push_str("    );\n\n");
        doc.push_str("    if (result != 0) {\n");
        doc.push_str("        return 1;\n");
        doc.push_str("    }\n\n");
        doc.push_str("    // Run pipeline\n");
        doc.push_str("    char* output;\n");
        doc.push_str("    result = trustformers_pipeline_run(\n");
        doc.push_str("        pipeline,\n");
        doc.push_str("        \"Hello, world!\",\n");
        doc.push_str("        &output\n");
        doc.push_str("    );\n\n");
        doc.push_str("    if (result == 0) {\n");
        doc.push_str("        printf(\"%s\\n\", output);\n");
        doc.push_str("        trustformers_string_free(output);\n");
        doc.push_str("    }\n\n");
        doc.push_str("    // Cleanup\n");
        doc.push_str("    trustformers_pipeline_free(pipeline);\n\n");
        doc.push_str("    return 0;\n");
        doc.push_str("}\n");
        doc.push_str("```\n\n");

        doc
    }
}

impl LanguageGenerator for MarkdownDocGenerator {
    fn target_language(&self) -> TargetLanguage {
        TargetLanguage::Markdown
    }

    fn file_extension(&self) -> &'static str {
        "md"
    }

    fn generate(
        &self,
        interface: &FfiInterface,
        output_dir: &Path,
        _templates: &TemplateEngine,
    ) -> TrustformersResult<()> {
        let mut doc = String::new();

        // Title and header
        doc.push_str(&format!(
            "# {} C API Documentation\n\n",
            self.config.package_info.name
        ));

        doc.push_str(&format!(
            "**Version**: {}\n\n",
            self.config.package_info.version
        ));

        // Overview
        doc.push_str("## Overview\n\n");
        doc.push_str(&format!("{}\n\n", self.config.package_info.description));
        doc.push_str(&format!(
            "This is the C API documentation for {}. ",
            self.config.package_info.name
        ));
        doc.push_str(
            "The C API provides a stable ABI for integrating with other languages and systems.\n\n",
        );

        // Metadata
        if !interface.metadata.library_name.is_empty() {
            doc.push_str(&format!(
                "**Library Name**: `{}`\n\n",
                interface.metadata.library_name
            ));
        }

        if !interface.metadata.version.is_empty() {
            doc.push_str(&format!(
                "**Library Version**: {}\n\n",
                interface.metadata.version
            ));
        }

        // Installation
        doc.push_str("## Installation\n\n");
        doc.push_str("### Building from Source\n\n");
        doc.push_str("```bash\n");
        doc.push_str("git clone ");
        doc.push_str(&self.config.package_info.repository);
        doc.push_str("\n");
        doc.push_str(&format!("cd {}\n", self.config.package_info.name));
        doc.push_str("cargo build --release\n");
        doc.push_str("```\n\n");

        doc.push_str("### Linking\n\n");
        doc.push_str("```c\n");
        doc.push_str("// Include the header\n");
        doc.push_str(&format!(
            "#include \"{}.h\"\n\n",
            self.config.package_info.name
        ));
        doc.push_str("// Link with the library\n");
        doc.push_str(&format!("// -l{}\n", self.config.package_info.name));
        doc.push_str("```\n\n");

        // Table of contents
        doc.push_str(&self.generate_table_of_contents(interface));

        // Error handling section
        doc.push_str(&self.generate_error_handling_section(interface));

        // Enumerations section
        if !interface.enums.is_empty() {
            doc.push_str("## Enumerations\n\n");
            doc.push_str("This section describes all enumeration types available in the API.\n\n");

            for enum_type in &interface.enums {
                doc.push_str(&self.generate_enum_docs(enum_type));
            }
        }

        // Structures section
        if !interface.structs.is_empty() {
            doc.push_str("## Structures\n\n");
            doc.push_str("This section describes all structure types available in the API.\n\n");

            for struct_type in &interface.structs {
                doc.push_str(&self.generate_struct_docs(struct_type));
            }
        }

        // Functions section
        if !interface.functions.is_empty() {
            doc.push_str("## Functions\n\n");
            doc.push_str("This section describes all functions available in the API.\n\n");

            // Group functions by category
            let categories = self.categorize_functions(&interface.functions);
            let mut sorted_categories: Vec<_> = categories.keys().collect();
            sorted_categories.sort();

            for category in sorted_categories {
                let funcs = &categories[category];
                let anchor = category.to_lowercase().replace(' ', "-");

                doc.push_str(&format!("### {}\n\n", category));
                doc.push_str(&format!("<a id=\"{}\"></a>\n\n", anchor));

                for func in funcs {
                    doc.push_str(&self.generate_function_docs(func));
                }
            }
        }

        // Examples section
        doc.push_str(&self.generate_examples_section());

        // Version information
        doc.push_str("## Version Information\n\n");
        doc.push_str(&format!(
            "- **Version**: {}\n",
            self.config.package_info.version
        ));
        doc.push_str(&format!(
            "- **Author**: {}\n",
            self.config.package_info.author
        ));
        doc.push_str(&format!(
            "- **Repository**: [{}]({})\n\n",
            self.config.package_info.repository, self.config.package_info.repository
        ));

        // License
        doc.push_str("## License\n\n");
        doc.push_str(&format!(
            "This project is licensed under the {} license.\n\n",
            self.config.package_info.license
        ));

        doc.push_str("---\n\n");
        doc.push_str(&format!(
            "*Generated automatically by {} documentation generator*\n",
            self.config.package_info.name
        ));

        // Write to file
        let output_path = output_dir.join("API.md");
        fs::write(&output_path, doc).map_err(|_| crate::error::TrustformersError::FileNotFound)?;

        println!("Generated Markdown documentation: {:?}", output_path);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{CodeGenConfig, PackageInfo};

    #[test]
    fn test_markdown_generator_creation() {
        let config = CodeGenConfig::default();
        let generator = MarkdownDocGenerator::new(&config);
        assert!(generator.is_ok());
    }

    #[test]
    fn test_target_language() {
        let config = CodeGenConfig::default();
        let generator = MarkdownDocGenerator::new(&config).unwrap();
        assert_eq!(generator.target_language(), TargetLanguage::Markdown);
    }

    #[test]
    fn test_file_extension() {
        let config = CodeGenConfig::default();
        let generator = MarkdownDocGenerator::new(&config).unwrap();
        assert_eq!(generator.file_extension(), "md");
    }

    #[test]
    fn test_categorize_functions() {
        let config = CodeGenConfig::default();
        let generator = MarkdownDocGenerator::new(&config).unwrap();

        let functions = vec![
            FfiFunction {
                name: "trustformers_model_load".to_string(),
                ..Default::default()
            },
            FfiFunction {
                name: "trustformers_tokenizer_encode".to_string(),
                ..Default::default()
            },
            FfiFunction {
                name: "trustformers_inference_run".to_string(),
                ..Default::default()
            },
        ];

        let categories = generator.categorize_functions(&functions);
        assert!(categories.contains_key("Model Management"));
        assert!(categories.contains_key("Tokenization"));
        assert!(categories.contains_key("Inference"));
    }

    #[test]
    fn test_format_type() {
        let config = CodeGenConfig::default();
        let generator = MarkdownDocGenerator::new(&config).unwrap();

        let pointer_type = FfiType {
            name: "c_char".to_string(),
            is_pointer: true,
            is_const: true,
            pointer_level: 1,
            ..Default::default()
        };

        let formatted = generator.format_type(&pointer_type);
        assert!(formatted.contains("const"));
        assert!(formatted.contains("*"));
    }

    #[test]
    fn test_generate_table_of_contents() {
        let config = CodeGenConfig::default();
        let generator = MarkdownDocGenerator::new(&config).unwrap();

        let mut interface = FfiInterface::default();
        interface.enums.push(FfiEnum {
            name: "TestEnum".to_string(),
            ..Default::default()
        });

        let toc = generator.generate_table_of_contents(&interface);
        assert!(toc.contains("Table of Contents"));
        assert!(toc.contains("TestEnum"));
    }
}
