//! Code Generation Pipeline
//!
//! Provides a production-quality pipeline for code generation tasks including
//! standard generation, fill-in-the-middle (FIM) completion, and instruction-to-code.

use std::collections::HashMap;
use std::fmt;

// ── IndentStyle ───────────────────────────────────────────────────────────────

/// Indentation style for generated code
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndentStyle {
    /// Use spaces with the given count per level
    Spaces(usize),
    /// Use tabs
    Tabs,
}

impl Default for IndentStyle {
    fn default() -> Self {
        IndentStyle::Spaces(4)
    }
}

impl fmt::Display for IndentStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndentStyle::Spaces(n) => write!(f, "{} spaces", n),
            IndentStyle::Tabs => write!(f, "tabs"),
        }
    }
}

// ── StopReason ────────────────────────────────────────────────────────────────

/// Reason why generation stopped
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Reached the maximum token limit
    MaxTokens,
    /// Hit a configured stop sequence
    StopSequence(String),
    /// Natural end of sequence
    EndOfSequence,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StopReason::MaxTokens => write!(f, "max_tokens"),
            StopReason::StopSequence(s) => write!(f, "stop_sequence({})", s),
            StopReason::EndOfSequence => write!(f, "end_of_sequence"),
        }
    }
}

// ── ExtractionInfo ────────────────────────────────────────────────────────────

/// Information about how code was extracted from model output
#[derive(Debug, Clone, Default)]
pub struct ExtractionInfo {
    /// Whether the output contained a markdown code fence
    pub had_markdown_fence: bool,
    /// Language tag found in the fence (e.g., "python")
    pub fence_language: Option<String>,
    /// Number of lines in the extracted code
    pub lines_extracted: usize,
}

// ── CodeGenerationConfig ──────────────────────────────────────────────────────

/// Configuration for the code generation pipeline
#[derive(Debug, Clone)]
pub struct CodeGenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// Sampling temperature — lower is more deterministic (code benefits from low temp)
    pub temperature: f32,
    /// Nucleus sampling probability
    pub top_p: f32,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Target programming language hint (e.g., "python", "rust")
    pub language: Option<String>,
    /// Sequences that signal the end of a generation (e.g., "```", "\n\n\n")
    pub stop_sequences: Vec<String>,
    /// Enable Fill-In-the-Middle mode
    pub fill_in_the_middle: bool,
    /// FIM prefix special token
    pub fim_prefix_token: String,
    /// FIM suffix special token
    pub fim_suffix_token: String,
    /// FIM middle special token
    pub fim_middle_token: String,
    /// Indentation style preference
    pub indent_style: IndentStyle,
    /// Whether to include docstrings in generated code
    pub include_docstring: bool,
}

impl Default for CodeGenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.2,
            top_p: 0.95,
            top_k: 50,
            language: None,
            stop_sequences: vec!["```".to_string(), "\n\n\n".to_string()],
            fill_in_the_middle: false,
            fim_prefix_token: "<fim_prefix>".to_string(),
            fim_suffix_token: "<fim_suffix>".to_string(),
            fim_middle_token: "<fim_middle>".to_string(),
            indent_style: IndentStyle::default(),
            include_docstring: true,
        }
    }
}

// ── CodeGenerationInput ───────────────────────────────────────────────────────

/// Input variants for the code generation pipeline
#[derive(Debug, Clone)]
pub enum CodeGenerationInput {
    /// Generate code from a plain prompt
    Prompt(String),
    /// Fill-in-the-Middle completion: given prefix and suffix, generate the middle
    FillInMiddle {
        /// Code before the cursor / insertion point
        prefix: String,
        /// Code after the cursor / insertion point
        suffix: String,
    },
    /// Convert a natural-language instruction into code
    Instruction {
        /// The task description
        task: String,
        /// Optional surrounding code context
        context: Option<String>,
    },
}

// ── CodeGenerationOutput ─────────────────────────────────────────────────────

/// Output of the code generation pipeline
#[derive(Debug, Clone)]
pub struct CodeGenerationOutput {
    /// The generated (and post-processed) code
    pub generated_code: String,
    /// Heuristically detected programming language
    pub language_detected: Option<String>,
    /// Approximate number of tokens generated
    pub num_tokens_generated: usize,
    /// Why generation stopped
    pub stop_reason: StopReason,
    /// Metadata about code extraction from the raw model output
    pub extraction_info: ExtractionInfo,
}

// ── CodeGenerationError ───────────────────────────────────────────────────────

/// Errors that can occur during code generation
#[derive(Debug, Clone)]
pub enum CodeGenerationError {
    /// Input was empty or blank
    EmptyInput,
    /// Requested language is invalid or unsupported
    InvalidLanguage(String),
    /// The generation step itself failed
    GenerationFailed(String),
}

impl fmt::Display for CodeGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodeGenerationError::EmptyInput => write!(f, "code generation input is empty"),
            CodeGenerationError::InvalidLanguage(lang) => {
                write!(f, "invalid or unsupported language: {}", lang)
            },
            CodeGenerationError::GenerationFailed(msg) => {
                write!(f, "code generation failed: {}", msg)
            },
        }
    }
}

impl std::error::Error for CodeGenerationError {}

// ── CodeGenerationPipeline ────────────────────────────────────────────────────

/// Pipeline for code generation tasks
///
/// Supports three generation modes:
/// - `Prompt`: open-ended code generation from a text prompt
/// - `FillInMiddle`: FIM-style infilling given prefix and suffix context
/// - `Instruction`: natural-language task description → code
///
/// The pipeline implements all supporting algorithms (FIM prompt construction,
/// markdown fence extraction, stop-sequence application, language detection,
/// token estimation) in pure Rust. The actual generation step uses an extractive
/// placeholder that mimics the output contract of a real LLM backend.
pub struct CodeGenerationPipeline {
    config: CodeGenerationConfig,
}

impl CodeGenerationPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: CodeGenerationConfig) -> Self {
        Self { config }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Generate code from the given input.
    pub fn generate(
        &self,
        input: CodeGenerationInput,
    ) -> Result<CodeGenerationOutput, CodeGenerationError> {
        // Validate language if provided
        if let Some(lang) = &self.config.language {
            if lang.trim().is_empty() {
                return Err(CodeGenerationError::InvalidLanguage(lang.clone()));
            }
        }

        // Build the raw prompt text from the input variant
        let raw_prompt = self.build_prompt(&input)?;

        // Simulate generation (extractive placeholder)
        let raw_output = self.simulate_generation(&raw_prompt, &input);

        // Apply stop sequences
        let (trimmed, stop_reason) =
            Self::apply_stop_sequences(&raw_output, &self.config.stop_sequences);

        // Extract code from potential markdown fences
        let (extracted_code, fence_language) = Self::extract_code_from_markdown(&trimmed);

        let had_fence =
            fence_language.is_some() || trimmed.contains("```") || extracted_code != trimmed;
        let lines_extracted = extracted_code.lines().count();

        let extraction_info = ExtractionInfo {
            had_markdown_fence: had_fence,
            fence_language,
            lines_extracted,
        };

        // Prefer config language, fall back to detection
        let language_detected =
            self.config.language.clone().or_else(|| Self::detect_language(&extracted_code));

        let num_tokens_generated = Self::count_tokens_heuristic(&extracted_code);

        Ok(CodeGenerationOutput {
            generated_code: extracted_code,
            language_detected,
            num_tokens_generated,
            stop_reason,
            extraction_info,
        })
    }

    // ── Language detection ────────────────────────────────────────────────────

    /// Heuristically detect the programming language of a code snippet.
    ///
    /// Returns `None` if no strong signal is found.
    pub fn detect_language(code: &str) -> Option<String> {
        // Count signals per language to handle ambiguous snippets
        let mut scores: HashMap<&str, usize> = HashMap::new();

        let lines: Vec<&str> = code.lines().collect();

        for line in &lines {
            let trimmed = line.trim();

            // Rust
            if trimmed.starts_with("fn ")
                || trimmed.contains(" fn ")
                || trimmed.starts_with("pub fn ")
                || trimmed.starts_with("let mut ")
                || trimmed.starts_with("let ")
                || trimmed.starts_with("use ")
                || trimmed.contains("-> Result<")
                || trimmed.contains("impl ")
                || trimmed.contains("struct ")
                || trimmed.contains("enum ")
            {
                *scores.entry("rust").or_insert(0) += 1;
            }

            // Python
            if trimmed.starts_with("def ")
                || trimmed.starts_with("class ")
                || trimmed.starts_with("import ")
                || trimmed.starts_with("from ")
                || trimmed.starts_with("    ") && code.contains("def ")
                || trimmed.starts_with("elif ")
                || trimmed.contains("self.")
                || trimmed.ends_with(':')
            {
                *scores.entry("python").or_insert(0) += 1;
            }

            // JavaScript / TypeScript
            if trimmed.starts_with("function ")
                || trimmed.contains("const ")
                || trimmed.contains("let ")
                || trimmed.contains("var ")
                || trimmed.starts_with("=>")
                || trimmed.contains("console.log")
                || trimmed.contains("document.")
            {
                *scores.entry("javascript").or_insert(0) += 1;
            }

            // Java
            if trimmed.starts_with("public class ")
                || trimmed.starts_with("private ")
                || trimmed.starts_with("protected ")
                || trimmed.contains("System.out.println")
                || trimmed.contains("public static void main")
                || trimmed.contains("@Override")
            {
                *scores.entry("java").or_insert(0) += 1;
            }

            // C / C++
            if trimmed.starts_with("#include")
                || trimmed.starts_with("#define")
                || trimmed.starts_with("#pragma")
                || trimmed.contains("->")
                || (trimmed.contains("*") && trimmed.contains("&"))
                || trimmed.contains("std::")
                || trimmed.contains("cout <<")
            {
                *scores.entry("cpp").or_insert(0) += 1;
            }

            // Go
            if trimmed.starts_with("package main")
                || trimmed.starts_with("package ")
                || trimmed.contains("func ")
                || trimmed.contains(":=")
                || trimmed.starts_with("import (")
                || trimmed.contains("fmt.Println")
            {
                *scores.entry("go").or_insert(0) += 1;
            }

            // TypeScript (augment JS score)
            if trimmed.contains(": string")
                || trimmed.contains(": number")
                || trimmed.contains(": boolean")
                || trimmed.starts_with("interface ")
                || trimmed.starts_with("type ")
            {
                *scores.entry("typescript").or_insert(0) += 1;
            }
        }

        // Return language with highest score (minimum threshold: 1)
        scores
            .into_iter()
            .max_by_key(|(_, v)| *v)
            .filter(|(_, v)| *v >= 1)
            .map(|(lang, _)| lang.to_string())
    }

    // ── Markdown fence extraction ─────────────────────────────────────────────

    /// Strip markdown code fences from model output.
    ///
    /// Returns `(extracted_code, fence_language)`.  If no fence was found the
    /// original text is returned unchanged and `fence_language` is `None`.
    pub fn extract_code_from_markdown(text: &str) -> (String, Option<String>) {
        let lines: Vec<&str> = text.lines().collect();

        // Find opening fence
        let mut fence_start: Option<usize> = None;
        let mut fence_language: Option<String> = None;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("```") {
                let lang_tag = trimmed.trim_start_matches('`').trim().to_string();
                fence_language = if lang_tag.is_empty() { None } else { Some(lang_tag) };
                fence_start = Some(i);
                break;
            }
        }

        let start_idx = match fence_start {
            None => return (text.to_string(), None),
            Some(i) => i + 1, // skip the opening fence line
        };

        // Find closing fence
        let mut fence_end: Option<usize> = None;
        for (i, line) in lines[start_idx..].iter().enumerate() {
            if line.trim() == "```" {
                fence_end = Some(start_idx + i);
                break;
            }
        }

        let end_idx = fence_end.unwrap_or(lines.len());
        let extracted = lines[start_idx..end_idx].join("\n");

        (extracted, fence_language)
    }

    // ── FIM prompt building ───────────────────────────────────────────────────

    /// Build a Fill-In-the-Middle prompt from prefix and suffix.
    pub fn build_fim_prompt(&self, prefix: &str, suffix: &str) -> String {
        format!(
            "{}{}{}{}{}",
            self.config.fim_prefix_token,
            prefix,
            self.config.fim_suffix_token,
            suffix,
            self.config.fim_middle_token,
        )
    }

    // ── Stop sequence application ─────────────────────────────────────────────

    /// Truncate `text` at the first occurrence of any stop sequence.
    ///
    /// Returns `(truncated_text, stop_reason)`.
    pub fn apply_stop_sequences(text: &str, stops: &[String]) -> (String, StopReason) {
        let mut earliest: Option<(usize, &str)> = None;

        for stop in stops {
            if stop.is_empty() {
                continue;
            }
            if let Some(pos) = text.find(stop.as_str()) {
                match earliest {
                    None => earliest = Some((pos, stop)),
                    Some((best_pos, _)) if pos < best_pos => earliest = Some((pos, stop)),
                    _ => {},
                }
            }
        }

        match earliest {
            Some((pos, stop_seq)) => (
                text[..pos].to_string(),
                StopReason::StopSequence(stop_seq.to_string()),
            ),
            None => {
                // Check natural end-of-sequence heuristic
                if text.trim_end().ends_with('\n') || text.trim_end().len() < text.len() {
                    (text.to_string(), StopReason::EndOfSequence)
                } else {
                    (text.to_string(), StopReason::MaxTokens)
                }
            },
        }
    }

    // ── Token counting ────────────────────────────────────────────────────────

    /// Estimate the number of tokens in `text` using a ~4 chars/token heuristic.
    pub fn count_tokens_heuristic(text: &str) -> usize {
        let char_count = text.chars().count();
        char_count.div_ceil(4)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Build a full prompt string from the input variant.
    fn build_prompt(&self, input: &CodeGenerationInput) -> Result<String, CodeGenerationError> {
        match input {
            CodeGenerationInput::Prompt(p) => {
                if p.trim().is_empty() {
                    return Err(CodeGenerationError::EmptyInput);
                }
                Ok(p.clone())
            },
            CodeGenerationInput::FillInMiddle { prefix, suffix } => {
                if prefix.trim().is_empty() && suffix.trim().is_empty() {
                    return Err(CodeGenerationError::EmptyInput);
                }
                Ok(self.build_fim_prompt(prefix, suffix))
            },
            CodeGenerationInput::Instruction { task, context } => {
                if task.trim().is_empty() {
                    return Err(CodeGenerationError::EmptyInput);
                }
                let lang_hint = self
                    .config
                    .language
                    .as_deref()
                    .map(|l| format!(" in {}", l))
                    .unwrap_or_default();
                let context_section = match context {
                    Some(ctx) if !ctx.is_empty() => {
                        format!("\n\nContext:\n```\n{}\n```\n", ctx)
                    },
                    _ => String::new(),
                };
                Ok(format!(
                    "Write code{} to accomplish the following task:\n{}{}\n\nCode:",
                    lang_hint, task, context_section
                ))
            },
        }
    }

    /// Simulate code generation (extractive placeholder that respects the pipeline contract).
    ///
    /// In a real deployment this would call the underlying LLM backend.
    fn simulate_generation(&self, prompt: &str, input: &CodeGenerationInput) -> String {
        let lang = self.config.language.as_deref().unwrap_or("python");
        let indent = match &self.config.indent_style {
            IndentStyle::Spaces(n) => " ".repeat(*n),
            IndentStyle::Tabs => "\t".to_string(),
        };

        match input {
            CodeGenerationInput::Prompt(p) => {
                let clean = p.trim();
                let docstring = if self.config.include_docstring {
                    match lang {
                        "python" => format!("{}\"\"\"{}\"\"\"", indent, clean),
                        "rust" => format!("{}/// {}", indent, clean),
                        "javascript" | "typescript" => {
                            format!("{}/** {} */", indent, clean)
                        },
                        _ => format!("{}// {}", indent, clean),
                    }
                } else {
                    String::new()
                };

                let body = Self::generate_stub_body(lang, &indent);
                if docstring.is_empty() {
                    body
                } else {
                    format!("{}\n{}", docstring, body)
                }
            },
            CodeGenerationInput::FillInMiddle { prefix, suffix } => {
                // Extract last meaningful line from prefix to guess intent
                let hint = prefix.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("");
                format!(
                    "{}    # completing: {}\n{}    pass\n{}",
                    indent,
                    hint,
                    indent,
                    suffix.lines().next().unwrap_or("")
                )
            },
            CodeGenerationInput::Instruction { task, .. } => {
                let docstring = if self.config.include_docstring {
                    match lang {
                        "python" => format!("{}\"\"\"{}\"\"\"", indent, task),
                        "rust" => format!("{}/// {}", indent, task),
                        _ => format!("{}// {}", indent, task),
                    }
                } else {
                    String::new()
                };
                let body = Self::generate_stub_body(lang, &indent);
                if docstring.is_empty() {
                    format!("# Task: {}\n{}", task, body)
                } else {
                    format!("# Task: {}\n{}\n{}", task, docstring, body)
                }
            },
        }
    }

    /// Build a minimal, language-appropriate stub function body.
    fn generate_stub_body(lang: &str, indent: &str) -> String {
        match lang {
            "python" => format!("def generated_function():\n{}pass\n", indent),
            "rust" => format!("fn generated_function() {{\n{}todo!()\n}}\n", indent),
            "javascript" => format!("function generatedFunction() {{\n{}// TODO\n}}\n", indent),
            "typescript" => {
                format!(
                    "function generatedFunction(): void {{\n{}// TODO\n}}\n",
                    indent
                )
            },
            "java" => format!(
                "public static void generatedMethod() {{\n{}// TODO\n}}\n",
                indent
            ),
            "go" => format!("func generatedFunction() {{\n{}// TODO\n}}\n", indent),
            "cpp" => format!("void generated_function() {{\n{}// TODO\n}}\n", indent),
            _ => format!("// generated stub\n{}// TODO\n", indent),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Configuration defaults ────────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = CodeGenerationConfig::default();
        assert_eq!(cfg.max_new_tokens, 512);
        assert!((cfg.temperature - 0.2).abs() < 1e-6);
        assert!((cfg.top_p - 0.95).abs() < 1e-6);
        assert_eq!(cfg.top_k, 50);
        assert!(cfg.language.is_none());
        assert!(!cfg.fill_in_the_middle);
        assert_eq!(cfg.fim_prefix_token, "<fim_prefix>");
        assert_eq!(cfg.fim_suffix_token, "<fim_suffix>");
        assert_eq!(cfg.fim_middle_token, "<fim_middle>");
        assert!(cfg.include_docstring);
        assert!(!cfg.stop_sequences.is_empty());
    }

    #[test]
    fn test_indent_style_display() {
        assert_eq!(IndentStyle::Spaces(4).to_string(), "4 spaces");
        assert_eq!(IndentStyle::Spaces(2).to_string(), "2 spaces");
        assert_eq!(IndentStyle::Tabs.to_string(), "tabs");
    }

    // ── FIM prompt building ───────────────────────────────────────────────────

    #[test]
    fn test_build_fim_prompt_default_tokens() {
        let pipeline = CodeGenerationPipeline::new(CodeGenerationConfig::default());
        let prompt = pipeline.build_fim_prompt("def foo():\n    ", "    return x\n");
        assert!(prompt.contains("<fim_prefix>"));
        assert!(prompt.contains("<fim_suffix>"));
        assert!(prompt.contains("<fim_middle>"));
        assert!(prompt.contains("def foo():"));
        assert!(prompt.contains("return x"));
    }

    #[test]
    fn test_build_fim_prompt_custom_tokens() {
        let mut cfg = CodeGenerationConfig::default();
        cfg.fim_prefix_token = "<PRE>".to_string();
        cfg.fim_suffix_token = "<SUF>".to_string();
        cfg.fim_middle_token = "<MID>".to_string();
        let pipeline = CodeGenerationPipeline::new(cfg);
        let prompt = pipeline.build_fim_prompt("prefix", "suffix");
        assert_eq!(prompt, "<PRE>prefix<SUF>suffix<MID>");
    }

    #[test]
    fn test_fim_prompt_ordering() {
        let pipeline = CodeGenerationPipeline::new(CodeGenerationConfig::default());
        let prompt = pipeline.build_fim_prompt("A", "B");
        let prefix_pos = prompt.find("<fim_prefix>").expect("prefix token");
        let suffix_pos = prompt.find("<fim_suffix>").expect("suffix token");
        let middle_pos = prompt.find("<fim_middle>").expect("middle token");
        assert!(prefix_pos < suffix_pos && suffix_pos < middle_pos);
    }

    // ── Language detection ────────────────────────────────────────────────────

    #[test]
    fn test_detect_language_rust() {
        let code = "fn main() {\n    let x = 42;\n    println!(\"{}\", x);\n}\n";
        let lang = CodeGenerationPipeline::detect_language(code);
        assert_eq!(lang.as_deref(), Some("rust"));
    }

    #[test]
    fn test_detect_language_python() {
        let code = "def greet(name):\n    print(f'Hello, {name}')\n\ngreet('world')\n";
        let lang = CodeGenerationPipeline::detect_language(code);
        assert_eq!(lang.as_deref(), Some("python"));
    }

    #[test]
    fn test_detect_language_javascript() {
        let code = "function add(a, b) {\n    return a + b;\n}\nconsole.log(add(1, 2));\n";
        let lang = CodeGenerationPipeline::detect_language(code);
        assert_eq!(lang.as_deref(), Some("javascript"));
    }

    #[test]
    fn test_detect_language_java() {
        let code =
            "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}\n";
        let lang = CodeGenerationPipeline::detect_language(code);
        assert_eq!(lang.as_deref(), Some("java"));
    }

    #[test]
    fn test_detect_language_cpp() {
        let code = "#include <iostream>\nint main() {\n    std::cout << \"Hello\" << std::endl;\n    return 0;\n}\n";
        let lang = CodeGenerationPipeline::detect_language(code);
        assert_eq!(lang.as_deref(), Some("cpp"));
    }

    #[test]
    fn test_detect_language_go() {
        let code = "package main\nimport \"fmt\"\nfunc main() {\n    fmt.Println(\"Hello\")\n}\n";
        let lang = CodeGenerationPipeline::detect_language(code);
        assert_eq!(lang.as_deref(), Some("go"));
    }

    // ── Markdown fence extraction ─────────────────────────────────────────────

    #[test]
    fn test_extract_code_from_markdown_with_language() {
        let text = "Here is some code:\n```python\ndef hello():\n    print('hi')\n```\n";
        let (code, lang) = CodeGenerationPipeline::extract_code_from_markdown(text);
        assert!(code.contains("def hello():"));
        assert_eq!(lang.as_deref(), Some("python"));
    }

    #[test]
    fn test_extract_code_from_markdown_no_language() {
        let text = "```\nlet x = 1;\n```";
        let (code, lang) = CodeGenerationPipeline::extract_code_from_markdown(text);
        assert!(code.contains("let x = 1;"));
        assert!(lang.is_none());
    }

    #[test]
    fn test_extract_code_no_fence_passthrough() {
        let text = "def foo():\n    pass\n";
        let (code, lang) = CodeGenerationPipeline::extract_code_from_markdown(text);
        assert_eq!(code, text);
        assert!(lang.is_none());
    }

    // ── Stop sequence application ─────────────────────────────────────────────

    #[test]
    fn test_apply_stop_sequences_triggers() {
        let text = "def foo():\n    pass\n```\nextra content";
        let stops = vec!["```".to_string()];
        let (trimmed, reason) = CodeGenerationPipeline::apply_stop_sequences(text, &stops);
        assert!(!trimmed.contains("```"));
        assert!(!trimmed.contains("extra content"));
        assert!(matches!(reason, StopReason::StopSequence(_)));
    }

    #[test]
    fn test_apply_stop_sequences_no_trigger() {
        let text = "def foo():\n    pass\n";
        let stops = vec!["```".to_string(), "\n\n\n".to_string()];
        let (trimmed, _reason) = CodeGenerationPipeline::apply_stop_sequences(text, &stops);
        assert_eq!(trimmed, text);
    }

    #[test]
    fn test_apply_stop_sequences_earliest_wins() {
        let text = "abc\n\n\nXXX```YYY";
        let stops = vec!["```".to_string(), "\n\n\n".to_string()];
        let (trimmed, reason) = CodeGenerationPipeline::apply_stop_sequences(text, &stops);
        // "\n\n\n" appears at position 3, "```" appears later
        assert!(trimmed.starts_with("abc"));
        assert!(!trimmed.contains("XXX"));
        assert!(matches!(reason, StopReason::StopSequence(s) if s == "\n\n\n"));
    }

    // ── Token counting ────────────────────────────────────────────────────────

    #[test]
    fn test_count_tokens_heuristic() {
        // 4 chars → 1 token
        assert_eq!(CodeGenerationPipeline::count_tokens_heuristic("abcd"), 1);
        // 8 chars → 2 tokens
        assert_eq!(
            CodeGenerationPipeline::count_tokens_heuristic("abcdefgh"),
            2
        );
        // Empty
        assert_eq!(CodeGenerationPipeline::count_tokens_heuristic(""), 0);
        // 5 chars → ceiling(5/4) = 2
        assert_eq!(CodeGenerationPipeline::count_tokens_heuristic("abcde"), 2);
    }

    // ── Input variants ────────────────────────────────────────────────────────

    #[test]
    fn test_generate_prompt_input() {
        let pipeline = CodeGenerationPipeline::new(CodeGenerationConfig::default());
        let result = pipeline.generate(CodeGenerationInput::Prompt(
            "Write a function to add two numbers".to_string(),
        ));
        assert!(result.is_ok());
        let out = result.expect("generation output");
        assert!(!out.generated_code.is_empty());
        assert!(out.num_tokens_generated > 0);
    }

    #[test]
    fn test_generate_fim_input() {
        let pipeline = CodeGenerationPipeline::new(CodeGenerationConfig::default());
        let result = pipeline.generate(CodeGenerationInput::FillInMiddle {
            prefix: "def add(a, b):\n    ".to_string(),
            suffix: "\n    return result\n".to_string(),
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_instruction_input() {
        let pipeline = CodeGenerationPipeline::new(CodeGenerationConfig::default());
        let result = pipeline.generate(CodeGenerationInput::Instruction {
            task: "Sort a list of integers in ascending order".to_string(),
            context: Some("import sys".to_string()),
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_empty_prompt_errors() {
        let pipeline = CodeGenerationPipeline::new(CodeGenerationConfig::default());
        let result = pipeline.generate(CodeGenerationInput::Prompt("   ".to_string()));
        assert!(matches!(result, Err(CodeGenerationError::EmptyInput)));
    }

    #[test]
    fn test_generate_with_language_hint() {
        let mut cfg = CodeGenerationConfig::default();
        cfg.language = Some("rust".to_string());
        let pipeline = CodeGenerationPipeline::new(cfg);
        let result =
            pipeline.generate(CodeGenerationInput::Prompt("compute fibonacci".to_string()));
        assert!(result.is_ok());
        let out = result.expect("output");
        assert_eq!(out.language_detected.as_deref(), Some("rust"));
    }

    #[test]
    fn test_stop_reason_display() {
        assert_eq!(StopReason::MaxTokens.to_string(), "max_tokens");
        assert_eq!(StopReason::EndOfSequence.to_string(), "end_of_sequence");
        assert_eq!(
            StopReason::StopSequence("```".to_string()).to_string(),
            "stop_sequence(```)"
        );
    }

    #[test]
    fn test_error_display() {
        assert!(CodeGenerationError::EmptyInput.to_string().contains("empty"));
        assert!(CodeGenerationError::InvalidLanguage("foo".to_string())
            .to_string()
            .contains("foo"));
        assert!(CodeGenerationError::GenerationFailed("oops".to_string())
            .to_string()
            .contains("oops"));
    }
}
