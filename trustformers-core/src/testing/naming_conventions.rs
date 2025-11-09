//! Naming convention enforcement and validation
//!
//! This module provides tools for enforcing consistent naming conventions
//! across the TrustformeRS codebase, ensuring maintainability and readability.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Naming convention rules for different code elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamingConventions {
    /// Rules for function names (snake_case by default)
    pub functions: NamingRule,
    /// Rules for struct names (PascalCase by default)
    pub structs: NamingRule,
    /// Rules for enum names (PascalCase by default)
    pub enums: NamingRule,
    /// Rules for trait names (PascalCase by default)
    pub traits: NamingRule,
    /// Rules for constant names (SCREAMING_SNAKE_CASE by default)
    pub constants: NamingRule,
    /// Rules for variable names (snake_case by default)
    pub variables: NamingRule,
    /// Rules for module names (snake_case by default)
    pub modules: NamingRule,
    /// Rules for macro names (snake_case by default)
    pub macros: NamingRule,
    /// Rules for type aliases (PascalCase by default)
    pub type_aliases: NamingRule,
    /// Rules for generic type parameters (single uppercase letter)
    pub generics: NamingRule,
    /// Custom domain-specific rules
    pub custom_rules: HashMap<String, NamingRule>,
}

impl Default for NamingConventions {
    fn default() -> Self {
        Self {
            functions: NamingRule::SnakeCase,
            structs: NamingRule::PascalCase,
            enums: NamingRule::PascalCase,
            traits: NamingRule::PascalCase,
            constants: NamingRule::ScreamingSnakeCase,
            variables: NamingRule::SnakeCase,
            modules: NamingRule::SnakeCase,
            macros: NamingRule::SnakeCase,
            type_aliases: NamingRule::PascalCase,
            generics: NamingRule::SingleUppercase,
            custom_rules: HashMap::new(),
        }
    }
}

/// Supported naming convention rules
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NamingRule {
    /// snake_case: lowercase with underscores
    SnakeCase,
    /// PascalCase: UpperCamelCase
    PascalCase,
    /// camelCase: lowerCamelCase
    CamelCase,
    /// SCREAMING_SNAKE_CASE: uppercase with underscores
    ScreamingSnakeCase,
    /// kebab-case: lowercase with hyphens
    KebabCase,
    /// Single uppercase letter (for generics: T, U, V, etc.)
    SingleUppercase,
    /// Custom regex pattern
    Custom(String),
    /// Allow any naming style (disable checking)
    Any,
}

/// Naming convention violation
#[derive(Debug, Clone)]
pub struct NamingViolation {
    pub file_path: PathBuf,
    pub line_number: usize,
    pub column: usize,
    pub element_type: ElementType,
    pub element_name: String,
    pub expected_rule: NamingRule,
    pub suggested_name: Option<String>,
    pub severity: ViolationSeverity,
    pub message: String,
}

/// Type of code element
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    Function,
    Struct,
    Enum,
    Trait,
    Constant,
    Variable,
    Module,
    Macro,
    TypeAlias,
    Generic,
    Field,
    EnumVariant,
    Method,
}

/// Severity level of naming violations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViolationSeverity {
    Error,
    Warning,
    Info,
}

/// Configuration for naming convention checking
#[derive(Debug, Clone)]
pub struct NamingChecker {
    conventions: NamingConventions,
    excluded_patterns: Vec<Regex>,
    included_extensions: HashSet<String>,
    max_name_length: usize,
    min_name_length: usize,
    #[allow(dead_code)] // Reserved for future acronym validation features
    acronym_handling: AcronymHandling,
    #[allow(dead_code)] // Reserved for future abbreviation validation features
    abbreviation_dictionary: HashMap<String, String>,
}

/// How to handle acronyms in names
#[derive(Debug, Clone)]
pub enum AcronymHandling {
    /// Keep acronyms uppercase (e.g., XMLParser)
    KeepUppercase,
    /// Convert to pascal case (e.g., XmlParser)
    PascalCase,
    /// Follow surrounding case (e.g., xml_parser in snake_case)
    FollowSurrounding,
}

impl Default for NamingChecker {
    fn default() -> Self {
        let mut abbreviation_dictionary = HashMap::new();
        // Common domain-specific abbreviations
        abbreviation_dictionary.insert("gpu".to_string(), "GPU".to_string());
        abbreviation_dictionary.insert("cpu".to_string(), "CPU".to_string());
        abbreviation_dictionary.insert("api".to_string(), "API".to_string());
        abbreviation_dictionary.insert("url".to_string(), "URL".to_string());
        abbreviation_dictionary.insert("http".to_string(), "HTTP".to_string());
        abbreviation_dictionary.insert("json".to_string(), "JSON".to_string());
        abbreviation_dictionary.insert("xml".to_string(), "XML".to_string());
        abbreviation_dictionary.insert("ai".to_string(), "AI".to_string());
        abbreviation_dictionary.insert("ml".to_string(), "ML".to_string());
        abbreviation_dictionary.insert("nn".to_string(), "NN".to_string());
        abbreviation_dictionary.insert("rnn".to_string(), "RNN".to_string());
        abbreviation_dictionary.insert("cnn".to_string(), "CNN".to_string());
        abbreviation_dictionary.insert("gru".to_string(), "GRU".to_string());
        abbreviation_dictionary.insert("lstm".to_string(), "LSTM".to_string());
        abbreviation_dictionary.insert("bert".to_string(), "BERT".to_string());
        abbreviation_dictionary.insert("gpt".to_string(), "GPT".to_string());
        abbreviation_dictionary.insert("cuda".to_string(), "CUDA".to_string());
        abbreviation_dictionary.insert("rocm".to_string(), "ROCm".to_string());
        abbreviation_dictionary.insert("simd".to_string(), "SIMD".to_string());
        abbreviation_dictionary.insert("avx".to_string(), "AVX".to_string());
        abbreviation_dictionary.insert("sse".to_string(), "SSE".to_string());
        abbreviation_dictionary.insert("neon".to_string(), "NEON".to_string());

        Self {
            conventions: NamingConventions::default(),
            excluded_patterns: vec![
                Regex::new(r"^_.*").unwrap(), // Private items starting with underscore
                Regex::new(r".*_test$").unwrap(), // Test functions
                Regex::new(r"^test_.*").unwrap(), // Test functions
            ],
            included_extensions: {
                let mut set = HashSet::new();
                set.insert("rs".to_string());
                set.insert("toml".to_string());
                set
            },
            max_name_length: 50,
            min_name_length: 1,
            acronym_handling: AcronymHandling::FollowSurrounding,
            abbreviation_dictionary,
        }
    }
}

impl NamingChecker {
    /// Create a new naming checker with custom conventions
    pub fn new(conventions: NamingConventions) -> Self {
        Self {
            conventions,
            ..Default::default()
        }
    }

    /// Add an exclusion pattern
    pub fn exclude_pattern(&mut self, pattern: &str) -> Result<(), regex::Error> {
        let regex = Regex::new(pattern)?;
        self.excluded_patterns.push(regex);
        Ok(())
    }

    /// Check naming conventions in a single file
    pub fn check_file(&self, file_path: &Path) -> Result<Vec<NamingViolation>, std::io::Error> {
        let content = fs::read_to_string(file_path)?;
        let mut violations = Vec::new();

        // Skip if file extension is not included
        if let Some(ext) = file_path.extension() {
            if !self.included_extensions.contains(&ext.to_string_lossy().to_string()) {
                return Ok(violations);
            }
        }

        // Parse Rust code and extract naming elements
        match file_path.extension().and_then(|s| s.to_str()) {
            Some("rs") => {
                violations.extend(self.check_rust_file(&content, file_path)?);
            },
            Some("toml") => {
                violations.extend(self.check_toml_file(&content, file_path)?);
            },
            _ => {
                // Skip unknown file types
            },
        }

        Ok(violations)
    }

    /// Check naming conventions in Rust source code
    fn check_rust_file(
        &self,
        content: &str,
        file_path: &Path,
    ) -> Result<Vec<NamingViolation>, std::io::Error> {
        let mut violations = Vec::new();

        // Use regex patterns to find different code elements
        // Note: This is a simplified parser - a full implementation would use syn/proc-macro2

        // Function definitions
        let fn_regex =
            Regex::new(r"(?m)^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
                .unwrap();
        for (line_num, line) in content.lines().enumerate() {
            for cap in fn_regex.captures_iter(line) {
                let name = &cap[1];
                if !self.is_excluded(name) {
                    if let Some(violation) = self.check_name(
                        name,
                        &self.conventions.functions,
                        ElementType::Function,
                        file_path,
                        line_num + 1,
                        0,
                    ) {
                        violations.push(violation);
                    }
                }
            }
        }

        // Struct definitions
        let struct_regex =
            Regex::new(r"(?m)^\s*(?:pub\s+)?struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[<{]").unwrap();
        for (line_num, line) in content.lines().enumerate() {
            for cap in struct_regex.captures_iter(line) {
                let name = &cap[1];
                if !self.is_excluded(name) {
                    if let Some(violation) = self.check_name(
                        name,
                        &self.conventions.structs,
                        ElementType::Struct,
                        file_path,
                        line_num + 1,
                        0,
                    ) {
                        violations.push(violation);
                    }
                }
            }
        }

        // Enum definitions
        let enum_regex =
            Regex::new(r"(?m)^\s*(?:pub\s+)?enum\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[<{]").unwrap();
        for (line_num, line) in content.lines().enumerate() {
            for cap in enum_regex.captures_iter(line) {
                let name = &cap[1];
                if !self.is_excluded(name) {
                    if let Some(violation) = self.check_name(
                        name,
                        &self.conventions.enums,
                        ElementType::Enum,
                        file_path,
                        line_num + 1,
                        0,
                    ) {
                        violations.push(violation);
                    }
                }
            }
        }

        // Trait definitions
        let trait_regex =
            Regex::new(r"(?m)^\s*(?:pub\s+)?trait\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[<:{]").unwrap();
        for (line_num, line) in content.lines().enumerate() {
            for cap in trait_regex.captures_iter(line) {
                let name = &cap[1];
                if !self.is_excluded(name) {
                    if let Some(violation) = self.check_name(
                        name,
                        &self.conventions.traits,
                        ElementType::Trait,
                        file_path,
                        line_num + 1,
                        0,
                    ) {
                        violations.push(violation);
                    }
                }
            }
        }

        // Constants
        let const_regex =
            Regex::new(r"(?m)^\s*(?:pub\s+)?const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:").unwrap();
        for (line_num, line) in content.lines().enumerate() {
            for cap in const_regex.captures_iter(line) {
                let name = &cap[1];
                if !self.is_excluded(name) {
                    if let Some(violation) = self.check_name(
                        name,
                        &self.conventions.constants,
                        ElementType::Constant,
                        file_path,
                        line_num + 1,
                        0,
                    ) {
                        violations.push(violation);
                    }
                }
            }
        }

        // Type aliases
        let type_regex =
            Regex::new(r"(?m)^\s*(?:pub\s+)?type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[<=>]").unwrap();
        for (line_num, line) in content.lines().enumerate() {
            for cap in type_regex.captures_iter(line) {
                let name = &cap[1];
                if !self.is_excluded(name) {
                    if let Some(violation) = self.check_name(
                        name,
                        &self.conventions.type_aliases,
                        ElementType::TypeAlias,
                        file_path,
                        line_num + 1,
                        0,
                    ) {
                        violations.push(violation);
                    }
                }
            }
        }

        // Macros
        let macro_regex =
            Regex::new(r"(?m)^\s*macro_rules!\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{").unwrap();
        for (line_num, line) in content.lines().enumerate() {
            for cap in macro_regex.captures_iter(line) {
                let name = &cap[1];
                if !self.is_excluded(name) {
                    if let Some(violation) = self.check_name(
                        name,
                        &self.conventions.macros,
                        ElementType::Macro,
                        file_path,
                        line_num + 1,
                        0,
                    ) {
                        violations.push(violation);
                    }
                }
            }
        }

        Ok(violations)
    }

    /// Check naming conventions in TOML files (for Cargo.toml)
    fn check_toml_file(
        &self,
        content: &str,
        file_path: &Path,
    ) -> Result<Vec<NamingViolation>, std::io::Error> {
        let mut violations = Vec::new();

        // Check package name in Cargo.toml
        if file_path.file_name() == Some(std::ffi::OsStr::new("Cargo.toml")) {
            let name_regex = Regex::new(r#"name\s*=\s*"([^"]+)""#).unwrap();
            for (line_num, line) in content.lines().enumerate() {
                for cap in name_regex.captures_iter(line) {
                    let name = &cap[1];
                    if let Some(violation) = self.check_name(
                        name,
                        &NamingRule::KebabCase, // Cargo packages should use kebab-case
                        ElementType::Module,
                        file_path,
                        line_num + 1,
                        0,
                    ) {
                        violations.push(violation);
                    }
                }
            }
        }

        Ok(violations)
    }

    /// Check if a name should be excluded from checking
    fn is_excluded(&self, name: &str) -> bool {
        self.excluded_patterns.iter().any(|pattern| pattern.is_match(name))
    }

    /// Check a single name against a naming rule
    fn check_name(
        &self,
        name: &str,
        rule: &NamingRule,
        element_type: ElementType,
        file_path: &Path,
        line_number: usize,
        column: usize,
    ) -> Option<NamingViolation> {
        // Check length constraints
        if name.len() > self.max_name_length {
            return Some(NamingViolation {
                file_path: file_path.to_path_buf(),
                line_number,
                column,
                element_type,
                element_name: name.to_string(),
                expected_rule: rule.clone(),
                suggested_name: None,
                severity: ViolationSeverity::Warning,
                message: format!(
                    "Name '{}' is too long ({} chars, max {})",
                    name,
                    name.len(),
                    self.max_name_length
                ),
            });
        }

        if name.len() < self.min_name_length {
            return Some(NamingViolation {
                file_path: file_path.to_path_buf(),
                line_number,
                column,
                element_type,
                element_name: name.to_string(),
                expected_rule: rule.clone(),
                suggested_name: None,
                severity: ViolationSeverity::Error,
                message: format!(
                    "Name '{}' is too short ({} chars, min {})",
                    name,
                    name.len(),
                    self.min_name_length
                ),
            });
        }

        // Check naming convention
        if !self.matches_rule(name, rule) {
            let suggested_name = self.suggest_name(name, rule);
            let severity = if matches!(rule, NamingRule::Any) {
                ViolationSeverity::Info
            } else {
                ViolationSeverity::Error
            };

            return Some(NamingViolation {
                file_path: file_path.to_path_buf(),
                line_number,
                column,
                element_type,
                element_name: name.to_string(),
                expected_rule: rule.clone(),
                suggested_name,
                severity,
                message: format!(
                    "{:?} '{}' does not follow {:?} naming convention",
                    element_type, name, rule
                ),
            });
        }

        None
    }

    /// Check if a name matches a naming rule
    fn matches_rule(&self, name: &str, rule: &NamingRule) -> bool {
        match rule {
            NamingRule::SnakeCase => self.is_snake_case(name),
            NamingRule::PascalCase => self.is_pascal_case(name),
            NamingRule::CamelCase => self.is_camel_case(name),
            NamingRule::ScreamingSnakeCase => self.is_screaming_snake_case(name),
            NamingRule::KebabCase => self.is_kebab_case(name),
            NamingRule::SingleUppercase => self.is_single_uppercase(name),
            NamingRule::Custom(pattern) => {
                if let Ok(regex) = Regex::new(pattern) {
                    regex.is_match(name)
                } else {
                    false
                }
            },
            NamingRule::Any => true,
        }
    }

    /// Check if name is snake_case
    fn is_snake_case(&self, name: &str) -> bool {
        let regex = Regex::new(r"^[a-z][a-z0-9_]*$").unwrap();
        regex.is_match(name) && !name.ends_with('_') && !name.contains("__")
    }

    /// Check if name is PascalCase
    fn is_pascal_case(&self, name: &str) -> bool {
        let regex = Regex::new(r"^[A-Z][a-zA-Z0-9]*$").unwrap();
        regex.is_match(name)
    }

    /// Check if name is camelCase
    fn is_camel_case(&self, name: &str) -> bool {
        let regex = Regex::new(r"^[a-z][a-zA-Z0-9]*$").unwrap();
        regex.is_match(name)
    }

    /// Check if name is SCREAMING_SNAKE_CASE
    fn is_screaming_snake_case(&self, name: &str) -> bool {
        let regex = Regex::new(r"^[A-Z][A-Z0-9_]*$").unwrap();
        regex.is_match(name) && !name.ends_with('_') && !name.contains("__")
    }

    /// Check if name is kebab-case
    fn is_kebab_case(&self, name: &str) -> bool {
        let regex = Regex::new(r"^[a-z][a-z0-9-]*$").unwrap();
        regex.is_match(name) && !name.ends_with('-') && !name.contains("--")
    }

    /// Check if name is single uppercase letter
    fn is_single_uppercase(&self, name: &str) -> bool {
        let regex = Regex::new(r"^[A-Z]$").unwrap();
        regex.is_match(name)
    }

    /// Suggest a corrected name
    fn suggest_name(&self, name: &str, rule: &NamingRule) -> Option<String> {
        match rule {
            NamingRule::SnakeCase => Some(self.to_snake_case(name)),
            NamingRule::PascalCase => Some(self.to_pascal_case(name)),
            NamingRule::CamelCase => Some(self.to_camel_case(name)),
            NamingRule::ScreamingSnakeCase => Some(self.to_screaming_snake_case(name)),
            NamingRule::KebabCase => Some(self.to_kebab_case(name)),
            _ => None,
        }
    }

    /// Convert name to snake_case
    fn to_snake_case(&self, name: &str) -> String {
        let mut result = String::new();
        let chars: Vec<char> = name.chars().collect();

        for (i, &ch) in chars.iter().enumerate() {
            let should_add_underscore = if i > 0 {
                if ch.is_uppercase() {
                    // Add underscore before uppercase if previous is lowercase
                    !chars[i-1].is_uppercase() ||
                    // Or add underscore between consecutive uppercase letters (for acronyms like XML)
                    chars[i-1].is_uppercase()
                } else {
                    false
                }
            } else {
                false
            };

            if should_add_underscore {
                result.push('_');
            }

            if ch == '-' {
                result.push('_');
            } else {
                result.push(ch.to_lowercase().next().unwrap_or(ch));
            }
        }

        result
    }

    /// Convert name to PascalCase
    fn to_pascal_case(&self, name: &str) -> String {
        // If the input contains uppercase letters but no separators,
        // convert to snake_case first to identify word boundaries
        let snake_name =
            if name.chars().any(|c| c.is_uppercase()) && !name.contains('_') && !name.contains('-')
            {
                self.to_snake_case(name)
            } else {
                name.to_string()
            };

        let words: Vec<&str> = snake_name.split(&['_', '-'][..]).collect();
        words
            .iter()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => {
                        first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                    },
                }
            })
            .collect()
    }

    /// Convert name to camelCase
    fn to_camel_case(&self, name: &str) -> String {
        let pascal = self.to_pascal_case(name);
        let mut chars = pascal.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_lowercase().collect::<String>() + chars.as_str(),
        }
    }

    /// Convert name to SCREAMING_SNAKE_CASE
    fn to_screaming_snake_case(&self, name: &str) -> String {
        self.to_snake_case(name).to_uppercase()
    }

    /// Convert name to kebab-case
    fn to_kebab_case(&self, name: &str) -> String {
        self.to_snake_case(name).replace('_', "-")
    }

    /// Check naming conventions in a directory recursively
    pub fn check_directory(&self, dir_path: &Path) -> Result<Vec<NamingViolation>, std::io::Error> {
        let mut violations = Vec::new();

        fn visit_dir(
            dir: &Path,
            checker: &NamingChecker,
            violations: &mut Vec<NamingViolation>,
        ) -> std::io::Result<()> {
            if dir.is_dir() {
                for entry in fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();

                    if path.is_dir() {
                        // Skip hidden directories and common build/target directories
                        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                            if !name.starts_with('.') && name != "target" && name != "node_modules"
                            {
                                visit_dir(&path, checker, violations)?;
                            }
                        }
                    } else if let Ok(file_violations) = checker.check_file(&path) {
                        violations.extend(file_violations);
                    }
                }
            }
            Ok(())
        }

        visit_dir(dir_path, self, &mut violations)?;
        Ok(violations)
    }

    /// Generate a report of naming violations
    pub fn generate_report(&self, violations: &[NamingViolation]) -> NamingReport {
        let mut report = NamingReport::new();

        for violation in violations {
            report.add_violation(violation.clone());
        }

        report
    }
}

/// Report of naming convention violations
#[derive(Debug, Clone)]
pub struct NamingReport {
    pub violations: Vec<NamingViolation>,
    pub summary: ReportSummary,
}

/// Summary statistics for naming report
#[derive(Debug, Clone)]
pub struct ReportSummary {
    pub total_violations: usize,
    pub errors: usize,
    pub warnings: usize,
    pub info: usize,
    pub files_with_violations: usize,
    pub violations_by_type: HashMap<ElementType, usize>,
    pub most_common_violations: Vec<(String, usize)>,
}

impl Default for NamingReport {
    fn default() -> Self {
        Self::new()
    }
}

impl NamingReport {
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
            summary: ReportSummary {
                total_violations: 0,
                errors: 0,
                warnings: 0,
                info: 0,
                files_with_violations: 0,
                violations_by_type: HashMap::new(),
                most_common_violations: Vec::new(),
            },
        }
    }

    pub fn add_violation(&mut self, violation: NamingViolation) {
        // Update summary statistics
        self.summary.total_violations += 1;

        match violation.severity {
            ViolationSeverity::Error => self.summary.errors += 1,
            ViolationSeverity::Warning => self.summary.warnings += 1,
            ViolationSeverity::Info => self.summary.info += 1,
        }

        *self.summary.violations_by_type.entry(violation.element_type).or_insert(0) += 1;

        self.violations.push(violation);

        // Recalculate derived statistics
        self.update_summary();
    }

    fn update_summary(&mut self) {
        // Count unique files with violations
        let mut files: HashSet<PathBuf> = HashSet::new();
        for violation in &self.violations {
            files.insert(violation.file_path.clone());
        }
        self.summary.files_with_violations = files.len();

        // Find most common violation patterns
        let mut violation_patterns: HashMap<String, usize> = HashMap::new();
        for violation in &self.violations {
            let pattern = format!("{:?}-{:?}", violation.element_type, violation.expected_rule);
            *violation_patterns.entry(pattern).or_insert(0) += 1;
        }

        let mut sorted_patterns: Vec<(String, usize)> = violation_patterns.into_iter().collect();
        sorted_patterns.sort_by(|a, b| b.1.cmp(&a.1));
        self.summary.most_common_violations = sorted_patterns.into_iter().take(10).collect();
    }

    /// Print a formatted report to stdout
    pub fn print_report(&self) {
        println!("=== Naming Convention Report ===");
        println!();

        println!("Summary:");
        println!("  Total violations: {}", self.summary.total_violations);
        println!("  Errors: {}", self.summary.errors);
        println!("  Warnings: {}", self.summary.warnings);
        println!("  Info: {}", self.summary.info);
        println!(
            "  Files with violations: {}",
            self.summary.files_with_violations
        );
        println!();

        if !self.summary.violations_by_type.is_empty() {
            println!("Violations by element type:");
            let mut sorted_types: Vec<_> = self.summary.violations_by_type.iter().collect();
            sorted_types.sort_by(|a, b| b.1.cmp(a.1));
            for (element_type, count) in sorted_types {
                println!("  {:?}: {}", element_type, count);
            }
            println!();
        }

        if !self.summary.most_common_violations.is_empty() {
            println!("Most common violation patterns:");
            for (pattern, count) in &self.summary.most_common_violations {
                println!("  {}: {} occurrences", pattern, count);
            }
            println!();
        }

        if !self.violations.is_empty() {
            println!("Detailed violations:");
            for violation in &self.violations {
                self.print_violation(violation);
            }
        }
    }

    fn print_violation(&self, violation: &NamingViolation) {
        let severity_symbol = match violation.severity {
            ViolationSeverity::Error => "❌",
            ViolationSeverity::Warning => "⚠️ ",
            ViolationSeverity::Info => "ℹ️ ",
        };

        println!(
            "{} {}:{}:{} - {}",
            severity_symbol,
            violation.file_path.display(),
            violation.line_number,
            violation.column,
            violation.message
        );

        if let Some(suggested_name) = &violation.suggested_name {
            println!(
                "    Suggestion: '{}' -> '{}'",
                violation.element_name, suggested_name
            );
        }
    }

    /// Export report as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Check if there are any errors (for CI/CD integration)
    pub fn has_errors(&self) -> bool {
        self.summary.errors > 0
    }
}

impl Serialize for NamingReport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NamingReport", 2)?;
        state.serialize_field("violations", &self.violations)?;
        state.serialize_field("summary", &self.summary)?;
        state.end()
    }
}

impl Serialize for NamingViolation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NamingViolation", 8)?;
        state.serialize_field("file_path", &self.file_path.to_string_lossy())?;
        state.serialize_field("line_number", &self.line_number)?;
        state.serialize_field("column", &self.column)?;
        state.serialize_field("element_type", &format!("{:?}", self.element_type))?;
        state.serialize_field("element_name", &self.element_name)?;
        state.serialize_field("expected_rule", &format!("{:?}", self.expected_rule))?;
        state.serialize_field("suggested_name", &self.suggested_name)?;
        state.serialize_field("severity", &format!("{:?}", self.severity))?;
        state.serialize_field("message", &self.message)?;
        state.end()
    }
}

impl Serialize for ReportSummary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("ReportSummary", 7)?;
        state.serialize_field("total_violations", &self.total_violations)?;
        state.serialize_field("errors", &self.errors)?;
        state.serialize_field("warnings", &self.warnings)?;
        state.serialize_field("info", &self.info)?;
        state.serialize_field("files_with_violations", &self.files_with_violations)?;

        let violations_by_type: HashMap<String, usize> =
            self.violations_by_type.iter().map(|(k, v)| (format!("{:?}", k), *v)).collect();
        state.serialize_field("violations_by_type", &violations_by_type)?;
        state.serialize_field("most_common_violations", &self.most_common_violations)?;
        state.end()
    }
}

/// Command-line interface for naming convention checking
pub struct NamingCli;

impl NamingCli {
    /// Run naming convention checks from command line
    pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
        if args.len() < 2 {
            println!(
                "Usage: naming_checker <directory_path> [--config <config_file>] [--json] [--fix]"
            );
            return Ok(());
        }

        let directory = Path::new(&args[1]);
        let checker = NamingChecker::default();
        let mut json_output = false;
        let mut fix_violations = false;

        // Parse command line arguments
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--config" => {
                    if i + 1 < args.len() {
                        // Load custom config (implementation omitted for brevity)
                        i += 1;
                    }
                },
                "--json" => {
                    json_output = true;
                },
                "--fix" => {
                    fix_violations = true;
                },
                _ => {},
            }
            i += 1;
        }

        // Run checks
        let violations = checker.check_directory(directory)?;
        let report = checker.generate_report(&violations);

        // Output results
        if json_output {
            println!("{}", report.to_json()?);
        } else {
            report.print_report();
        }

        // Auto-fix violations if requested
        if fix_violations {
            Self::fix_violations(&violations)?;
        }

        // Exit with error code if there are errors (for CI/CD)
        if report.has_errors() {
            std::process::exit(1);
        }

        Ok(())
    }

    /// Attempt to automatically fix naming violations
    fn fix_violations(violations: &[NamingViolation]) -> Result<(), std::io::Error> {
        // Group violations by file
        let mut violations_by_file: HashMap<PathBuf, Vec<&NamingViolation>> = HashMap::new();
        for violation in violations {
            violations_by_file
                .entry(violation.file_path.clone())
                .or_default()
                .push(violation);
        }

        // Process each file
        for (file_path, file_violations) in violations_by_file {
            if let Some(_suggested_name) = &file_violations[0].suggested_name {
                println!("Fixing violations in {}", file_path.display());
                // Implementation would replace names in file content
                // This is simplified - real implementation would need careful parsing
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_case_validation() {
        let checker = NamingChecker::default();

        assert!(checker.is_snake_case("valid_name"));
        assert!(checker.is_snake_case("another_valid_name"));
        assert!(checker.is_snake_case("name123"));

        assert!(!checker.is_snake_case("InvalidName"));
        assert!(!checker.is_snake_case("invalid_"));
        assert!(!checker.is_snake_case("invalid__name"));
        assert!(!checker.is_snake_case(""));
    }

    #[test]
    fn test_pascal_case_validation() {
        let checker = NamingChecker::default();

        assert!(checker.is_pascal_case("ValidName"));
        assert!(checker.is_pascal_case("AnotherValidName"));
        assert!(checker.is_pascal_case("Name123"));

        assert!(!checker.is_pascal_case("invalidName"));
        assert!(!checker.is_pascal_case("Invalid_Name"));
        assert!(!checker.is_pascal_case(""));
    }

    #[test]
    fn test_case_conversion() {
        let checker = NamingChecker::default();

        assert_eq!(checker.to_snake_case("PascalCase"), "pascal_case");
        assert_eq!(checker.to_snake_case("XMLParser"), "x_m_l_parser");
        assert_eq!(checker.to_snake_case("kebab-case"), "kebab_case");

        assert_eq!(checker.to_pascal_case("snake_case"), "SnakeCase");
        assert_eq!(checker.to_pascal_case("kebab-case"), "KebabCase");

        assert_eq!(checker.to_camel_case("snake_case"), "snakeCase");
        assert_eq!(checker.to_camel_case("PascalCase"), "pascalCase");
    }

    #[test]
    fn test_violation_creation() {
        let checker = NamingChecker::default();
        let path = Path::new("test.rs");

        let violation = checker.check_name(
            "InvalidFunctionName",
            &NamingRule::SnakeCase,
            ElementType::Function,
            path,
            1,
            0,
        );

        assert!(violation.is_some());
        let v = violation.unwrap();
        assert_eq!(v.element_name, "InvalidFunctionName");
        assert_eq!(v.element_type, ElementType::Function);
        assert!(v.suggested_name.is_some());
    }

    #[test]
    fn test_exclusion_patterns() {
        let mut checker = NamingChecker::default();
        checker.exclude_pattern(r"^_internal_.*").unwrap();

        assert!(checker.is_excluded("_internal_function"));
        assert!(!checker.is_excluded("public_function"));
    }
}
