//! Code generation framework for TrustformeRS language bindings
//!
//! Automatically generates language bindings (Python, Java, Go, C#, TypeScript)
//! from the Rust FFI interface definitions.

use crate::error::TrustformersResult;
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub mod ast;
pub mod generators;
pub mod parser;
pub mod templates;

use ast::*;
use generators::*;

/// Main code generator that orchestrates binding generation for all languages
pub struct CodeGenerator {
    /// Configuration for code generation
    config: CodeGenConfig,
    /// Parsed FFI interface
    interface: FfiInterface,
    /// Template engine
    templates: templates::TemplateEngine,
}

/// Configuration for code generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGenConfig {
    /// Output directory for generated bindings
    pub output_dir: PathBuf,
    /// List of languages to generate bindings for
    pub target_languages: Vec<TargetLanguage>,
    /// Package/namespace information
    pub package_info: PackageInfo,
    /// Package name for Java/Kotlin (e.g., "com.trustformers.ffi")
    #[serde(default)]
    pub package_name: Option<String>,
    /// Feature flags for conditional generation
    pub features: HashMap<String, bool>,
    /// Custom type mappings
    pub type_mappings: HashMap<String, TypeMapping>,
}

/// Supported target languages for binding generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetLanguage {
    Python,
    Java,
    Go,
    CSharp,
    TypeScript,
    JavaScript,
    Kotlin,
    Swift,
    Ruby,
    PHP,
    Sphinx,
    Markdown,
    OpenApi,
    Javadoc,
    TypeDoc,
}

/// Package/namespace information for generated bindings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    /// Package name (e.g., "trustformers")
    pub name: String,
    /// Package version
    pub version: String,
    /// Package description
    pub description: String,
    /// Author information
    pub author: String,
    /// License
    pub license: String,
    /// Repository URL
    pub repository: String,
}

/// Custom type mapping for language-specific types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeMapping {
    /// Source FFI type
    pub ffi_type: String,
    /// Target language type
    pub target_type: String,
    /// Import/using statement needed (if any)
    pub import: Option<String>,
    /// Conversion logic (from FFI to target)
    pub from_ffi: Option<String>,
    /// Conversion logic (from target to FFI)
    pub to_ffi: Option<String>,
}

impl Default for CodeGenConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("generated"),
            target_languages: vec![
                TargetLanguage::Python,
                TargetLanguage::Java,
                TargetLanguage::Go,
                TargetLanguage::TypeScript,
                TargetLanguage::CSharp,
            ],
            package_info: PackageInfo {
                name: "trustformers".to_string(),
                version: "0.1.0".to_string(),
                description: "TrustformeRS - High-performance transformer library".to_string(),
                author: "TrustformeRS Team".to_string(),
                license: "MIT".to_string(),
                repository: "https://github.com/trustformers/trustformers".to_string(),
            },
            package_name: None,
            features: HashMap::new(),
            type_mappings: HashMap::new(),
        }
    }
}

impl CodeGenerator {
    /// Create a new code generator with the given configuration
    pub fn new(config: CodeGenConfig) -> TrustformersResult<Self> {
        let templates = templates::TemplateEngine::new()?;

        Ok(Self {
            config,
            interface: FfiInterface::default(),
            templates,
        })
    }

    /// Get a reference to the configuration
    pub fn config(&self) -> &CodeGenConfig {
        &self.config
    }

    /// Parse the FFI interface from Rust source files
    pub fn parse_interface(&mut self, source_dir: &Path) -> TrustformersResult<()> {
        let parser = parser::FfiParser::new();
        self.interface = parser.parse_directory(source_dir)?;
        Ok(())
    }

    /// Generate bindings for all configured target languages
    pub fn generate_all(&self) -> TrustformersResult<()> {
        // Create output directory
        fs::create_dir_all(&self.config.output_dir)?;

        for language in &self.config.target_languages {
            self.generate_language_bindings(language)?;
        }

        Ok(())
    }

    /// Generate bindings for a specific language
    pub fn generate_language_bindings(&self, language: &TargetLanguage) -> TrustformersResult<()> {
        let generator: Box<dyn LanguageGenerator> = match language {
            TargetLanguage::Python => Box::new(PythonGenerator::new(&self.config)?),
            TargetLanguage::Java => Box::new(JavaGenerator::new(&self.config)?),
            TargetLanguage::Go => Box::new(GoGenerator::new(&self.config)?),
            TargetLanguage::CSharp => Box::new(CSharpGenerator::new(&self.config)?),
            TargetLanguage::TypeScript => Box::new(TypeScriptGenerator::new(&self.config)?),
            TargetLanguage::JavaScript => Box::new(JavaScriptGenerator::new(&self.config)?),
            TargetLanguage::Kotlin => Box::new(KotlinGenerator::new(&self.config)?),
            TargetLanguage::Swift => Box::new(SwiftGenerator::new(&self.config)?),
            TargetLanguage::Ruby => Box::new(RubyGenerator::new(&self.config)?),
            TargetLanguage::PHP => Box::new(PhpGenerator::new(&self.config)?),
            TargetLanguage::Sphinx => Box::new(SphinxDocGenerator::new(&self.config)?),
            TargetLanguage::Markdown => Box::new(MarkdownDocGenerator::new(&self.config)?),
            TargetLanguage::OpenApi => Box::new(OpenApiGenerator::new(&self.config)?),
            TargetLanguage::Javadoc => Box::new(JavadocGenerator::new(&self.config)?),
            TargetLanguage::TypeDoc => Box::new(TypeDocGenerator::new(&self.config)?),
        };

        let output_dir = self.config.output_dir.join(language.directory_name());
        fs::create_dir_all(&output_dir)?;

        generator.generate(&self.interface, &output_dir, &self.templates)?;

        println!("Generated {} bindings in {:?}", language.name(), output_dir);
        Ok(())
    }

    /// Validate the parsed interface for completeness and correctness
    pub fn validate_interface(&self) -> TrustformersResult<Vec<ValidationWarning>> {
        let mut warnings = Vec::new();

        // Check for missing documentation
        for function in &self.interface.functions {
            if function.documentation.is_empty() {
                warnings.push(ValidationWarning::MissingDocumentation {
                    item_type: "function".to_string(),
                    item_name: function.name.clone(),
                });
            }
        }

        // Check for potentially unsafe patterns
        for function in &self.interface.functions {
            if function
                .parameters
                .iter()
                .any(|p| p.type_info.is_pointer() && !p.type_info.is_const())
            {
                warnings.push(ValidationWarning::UnsafePattern {
                    function_name: function.name.clone(),
                    issue: "Non-const pointer parameter".to_string(),
                });
            }
        }

        // Check for missing error handling
        for function in &self.interface.functions {
            if !function.return_type.is_error_type() && function.can_fail() {
                warnings.push(ValidationWarning::MissingErrorHandling {
                    function_name: function.name.clone(),
                });
            }
        }

        Ok(warnings)
    }

    /// Generate API documentation in multiple formats
    pub fn generate_documentation(&self) -> TrustformersResult<()> {
        let docs_dir = self.config.output_dir.join("docs");
        fs::create_dir_all(&docs_dir)?;

        // Generate Markdown documentation
        let markdown_generator = generators::MarkdownDocGenerator::new(&self.config)?;
        markdown_generator.generate(&self.interface, &docs_dir, &self.templates)?;

        // Generate OpenAPI specification
        let openapi_generator = generators::OpenApiGenerator::new(&self.config)?;
        openapi_generator.generate(&self.interface, &docs_dir, &self.templates)?;

        // Generate language-specific documentation
        for language in &self.config.target_languages {
            let lang_docs_dir = docs_dir.join(language.directory_name());
            fs::create_dir_all(&lang_docs_dir)?;

            match language {
                TargetLanguage::Python => {
                    let sphinx_generator = generators::SphinxDocGenerator::new(&self.config)?;
                    sphinx_generator.generate(&self.interface, &lang_docs_dir, &self.templates)?;
                },
                TargetLanguage::Java => {
                    let javadoc_generator = generators::JavadocGenerator::new(&self.config)?;
                    javadoc_generator.generate(&self.interface, &lang_docs_dir, &self.templates)?;
                },
                TargetLanguage::TypeScript => {
                    let typedoc_generator = generators::TypeDocGenerator::new(&self.config)?;
                    typedoc_generator.generate(&self.interface, &lang_docs_dir, &self.templates)?;
                },
                _ => {}, // Use default markdown for other languages
            }
        }

        Ok(())
    }

    /// Update existing bindings with new changes
    pub fn update_bindings(&self, incremental: bool) -> TrustformersResult<()> {
        if incremental {
            // Only regenerate changed files
            for language in &self.config.target_languages {
                self.update_language_bindings_incremental(language)?;
            }
        } else {
            // Full regeneration
            self.generate_all()?;
        }

        Ok(())
    }

    fn update_language_bindings_incremental(
        &self,
        language: &TargetLanguage,
    ) -> TrustformersResult<()> {
        // Check timestamps of source files vs generated files
        // Only regenerate if source is newer than generated
        // This is a simplified implementation - real version would be more sophisticated

        let output_dir = self.config.output_dir.join(language.directory_name());

        if !output_dir.exists() {
            // First time generation
            return self.generate_language_bindings(language);
        }

        // For now, always regenerate (full implementation would check file timestamps)
        self.generate_language_bindings(language)
    }
}

/// Validation warnings for the FFI interface
#[derive(Debug, Clone)]
pub enum ValidationWarning {
    MissingDocumentation {
        item_type: String,
        item_name: String,
    },
    UnsafePattern {
        function_name: String,
        issue: String,
    },
    MissingErrorHandling {
        function_name: String,
    },
    TypeMismatch {
        function_name: String,
        parameter_name: String,
        expected_type: String,
        actual_type: String,
    },
    DeprecatedFunction {
        function_name: String,
        replacement: Option<String>,
    },
}

impl std::fmt::Display for ValidationWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationWarning::MissingDocumentation {
                item_type,
                item_name,
            } => {
                write!(f, "Missing documentation for {} '{}'", item_type, item_name)
            },
            ValidationWarning::UnsafePattern {
                function_name,
                issue,
            } => {
                write!(
                    f,
                    "Unsafe pattern in function '{}': {}",
                    function_name, issue
                )
            },
            ValidationWarning::MissingErrorHandling { function_name } => {
                write!(f, "Missing error handling in function '{}'", function_name)
            },
            ValidationWarning::TypeMismatch {
                function_name,
                parameter_name,
                expected_type,
                actual_type,
            } => {
                write!(
                    f,
                    "Type mismatch in function '{}' parameter '{}': expected {}, got {}",
                    function_name, parameter_name, expected_type, actual_type
                )
            },
            ValidationWarning::DeprecatedFunction {
                function_name,
                replacement,
            } => {
                if let Some(repl) = replacement {
                    write!(
                        f,
                        "Function '{}' is deprecated, use '{}' instead",
                        function_name, repl
                    )
                } else {
                    write!(f, "Function '{}' is deprecated", function_name)
                }
            },
        }
    }
}

impl TargetLanguage {
    /// Get the human-readable name of the language
    pub fn name(&self) -> &'static str {
        match self {
            TargetLanguage::Python => "Python",
            TargetLanguage::Java => "Java",
            TargetLanguage::Go => "Go",
            TargetLanguage::CSharp => "C#",
            TargetLanguage::TypeScript => "TypeScript",
            TargetLanguage::JavaScript => "JavaScript",
            TargetLanguage::Kotlin => "Kotlin",
            TargetLanguage::Swift => "Swift",
            TargetLanguage::Ruby => "Ruby",
            TargetLanguage::PHP => "PHP",
            TargetLanguage::Sphinx => "Sphinx",
            TargetLanguage::Markdown => "Markdown",
            TargetLanguage::OpenApi => "OpenAPI",
            TargetLanguage::Javadoc => "Javadoc",
            TargetLanguage::TypeDoc => "TypeDoc",
        }
    }

    /// Get the directory name for the language bindings
    pub fn directory_name(&self) -> &'static str {
        match self {
            TargetLanguage::Python => "python",
            TargetLanguage::Java => "java",
            TargetLanguage::Go => "go",
            TargetLanguage::CSharp => "csharp",
            TargetLanguage::TypeScript => "typescript",
            TargetLanguage::JavaScript => "javascript",
            TargetLanguage::Kotlin => "kotlin",
            TargetLanguage::Swift => "swift",
            TargetLanguage::Ruby => "ruby",
            TargetLanguage::PHP => "php",
            TargetLanguage::Sphinx => "sphinx",
            TargetLanguage::Markdown => "markdown",
            TargetLanguage::OpenApi => "openapi",
            TargetLanguage::Javadoc => "javadoc",
            TargetLanguage::TypeDoc => "typedoc",
        }
    }

    /// Get the file extension for source files in this language
    pub fn file_extension(&self) -> &'static str {
        match self {
            TargetLanguage::Python => "py",
            TargetLanguage::Java => "java",
            TargetLanguage::Go => "go",
            TargetLanguage::CSharp => "cs",
            TargetLanguage::TypeScript => "ts",
            TargetLanguage::JavaScript => "js",
            TargetLanguage::Kotlin => "kt",
            TargetLanguage::Swift => "swift",
            TargetLanguage::Ruby => "rb",
            TargetLanguage::PHP => "php",
            TargetLanguage::Sphinx => "rst",
            TargetLanguage::Markdown => "md",
            TargetLanguage::OpenApi => "yaml",
            TargetLanguage::Javadoc => "html",
            TargetLanguage::TypeDoc => "html",
        }
    }
}

/// CLI interface for the code generator
pub fn run_code_generator() -> TrustformersResult<()> {
    use clap::{Arg, Command};

    let matches = Command::new("trustformers-codegen")
        .version("1.0")
        .about("Generate language bindings for TrustformeRS")
        .arg(
            Arg::new("source")
                .short('s')
                .long("source")
                .value_name("DIR")
                .help("Source directory containing Rust FFI code")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("DIR")
                .help("Output directory for generated bindings")
                .default_value("generated"),
        )
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file (JSON or TOML)"),
        )
        .arg(
            Arg::new("languages")
                .short('l')
                .long("languages")
                .value_name("LANG")
                .help("Comma-separated list of target languages")
                .default_value("python,java,go,typescript,csharp"),
        )
        .arg(
            Arg::new("validate")
                .long("validate")
                .help("Validate the FFI interface without generating code")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("docs")
                .long("docs")
                .help("Generate documentation in addition to bindings")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("incremental")
                .long("incremental")
                .help("Only regenerate changed files")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    let source_dir = PathBuf::from(matches.get_one::<String>("source").unwrap());
    let output_dir = PathBuf::from(matches.get_one::<String>("output").unwrap());

    // Load configuration
    let mut config = if let Some(config_file) = matches.get_one::<String>("config") {
        let config_content = fs::read_to_string(config_file)?;
        if config_file.ends_with(".json") {
            serde_json::from_str(&config_content)?
        } else {
            toml::from_str(&config_content)?
        }
    } else {
        CodeGenConfig::default()
    };

    config.output_dir = output_dir;

    // Parse target languages
    let languages_str = matches.get_one::<String>("languages").unwrap();
    config.target_languages = languages_str
        .split(',')
        .map(|lang| match lang.trim().to_lowercase().as_str() {
            "python" => Ok(TargetLanguage::Python),
            "java" => Ok(TargetLanguage::Java),
            "go" => Ok(TargetLanguage::Go),
            "csharp" | "c#" => Ok(TargetLanguage::CSharp),
            "typescript" | "ts" => Ok(TargetLanguage::TypeScript),
            "javascript" | "js" => Ok(TargetLanguage::JavaScript),
            "kotlin" => Ok(TargetLanguage::Kotlin),
            "swift" => Ok(TargetLanguage::Swift),
            "ruby" => Ok(TargetLanguage::Ruby),
            "php" => Ok(TargetLanguage::PHP),
            "sphinx" => Ok(TargetLanguage::Sphinx),
            "markdown" | "md" => Ok(TargetLanguage::Markdown),
            "openapi" => Ok(TargetLanguage::OpenApi),
            "javadoc" => Ok(TargetLanguage::Javadoc),
            "typedoc" => Ok(TargetLanguage::TypeDoc),
            _ => Err(anyhow!("Unknown language: {}", lang).into()),
        })
        .collect::<TrustformersResult<Vec<_>>>()?;

    // Initialize code generator
    let mut generator = CodeGenerator::new(config)?;

    // Parse FFI interface
    println!("Parsing FFI interface from {:?}...", source_dir);
    generator.parse_interface(&source_dir)?;

    // Validate interface if requested
    if matches.get_flag("validate") {
        println!("Validating FFI interface...");
        let warnings = generator.validate_interface()?;

        if warnings.is_empty() {
            println!("✅ Interface validation passed with no warnings");
        } else {
            println!(
                "⚠️  Interface validation found {} warnings:",
                warnings.len()
            );
            for warning in warnings {
                println!("  - {:?}", warning);
            }
        }

        if !matches.get_flag("docs") && !matches.contains_id("languages") {
            return Ok(());
        }
    }

    // Generate bindings
    if matches.get_flag("incremental") {
        println!("Updating bindings incrementally...");
        generator.update_bindings(true)?;
    } else {
        println!("Generating language bindings...");
        generator.generate_all()?;
    }

    // Generate documentation if requested
    if matches.get_flag("docs") {
        println!("Generating documentation...");
        generator.generate_documentation()?;
    }

    println!("✅ Code generation completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_code_generator_creation() {
        let config = CodeGenConfig::default();
        let generator = CodeGenerator::new(config);
        assert!(generator.is_ok());
    }

    #[test]
    fn test_target_language_properties() {
        let python = TargetLanguage::Python;
        assert_eq!(python.name(), "Python");
        assert_eq!(python.directory_name(), "python");
        assert_eq!(python.file_extension(), "py");
    }

    #[test]
    fn test_config_serialization() {
        let config = CodeGenConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: CodeGenConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.package_info.name, deserialized.package_info.name);
    }

    #[test]
    fn test_validation_warnings() {
        let generator = CodeGenerator::new(CodeGenConfig::default()).unwrap();
        let warnings = generator.validate_interface().unwrap();
        // Empty interface should produce no warnings
        assert!(warnings.is_empty());
    }
}
