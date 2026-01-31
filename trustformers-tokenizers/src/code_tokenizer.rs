use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use trustformers_core::errors::Result;
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Supported programming languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Java,
    CSharp,
    CPlusPlus,
    C,
    Go,
    Ruby,
    PHP,
    Swift,
    Kotlin,
    Scala,
    Haskell,
    Clojure,
    SQL,
    HTML,
    CSS,
    JSON,
    XML,
    YAML,
    Markdown,
    Shell,
    PowerShell,
    R,
    Matlab,
}

impl Language {
    /// Get file extensions for the language
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Language::Rust => &["rs"],
            Language::Python => &["py", "pyx", "pyi", "pyw"],
            Language::JavaScript => &["js", "jsx", "mjs", "cjs"],
            Language::TypeScript => &["ts", "tsx", "d.ts"],
            Language::Java => &["java"],
            Language::CSharp => &["cs"],
            Language::CPlusPlus => &["cpp", "cxx", "cc", "hpp", "hxx", "hh"],
            Language::C => &["c", "h"],
            Language::Go => &["go"],
            Language::Ruby => &["rb", "rbx", "rjs", "gemspec"],
            Language::PHP => &["php", "phtml", "php3", "php4", "php5"],
            Language::Swift => &["swift"],
            Language::Kotlin => &["kt", "kts"],
            Language::Scala => &["scala", "sc"],
            Language::Haskell => &["hs", "lhs"],
            Language::Clojure => &["clj", "cljs", "cljc", "edn"],
            Language::SQL => &["sql"],
            Language::HTML => &["html", "htm", "xhtml"],
            Language::CSS => &["css", "scss", "sass", "less"],
            Language::JSON => &["json", "jsonl", "ndjson"],
            Language::XML => &["xml", "xsd", "xsl", "xslt"],
            Language::YAML => &["yaml", "yml"],
            Language::Markdown => &["md", "markdown", "mdown", "mkd"],
            Language::Shell => &["sh", "bash", "zsh", "fish"],
            Language::PowerShell => &["ps1", "psm1", "psd1"],
            Language::R => &["r", "R"],
            Language::Matlab => &["m"],
        }
    }

    /// Get keywords for the language
    pub fn keywords(&self) -> &'static [&'static str] {
        match self {
            Language::Rust => &[
                "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false",
                "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut",
                "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait",
                "true", "type", "unsafe", "use", "where", "while", "async", "await", "dyn",
            ],
            Language::Python => &[
                "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
                "continue", "def", "del", "elif", "else", "except", "finally", "for", "from",
                "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
                "raise", "return", "try", "while", "with", "yield",
            ],
            Language::JavaScript | Language::TypeScript => &[
                "break",
                "case",
                "catch",
                "class",
                "const",
                "continue",
                "debugger",
                "default",
                "delete",
                "do",
                "else",
                "export",
                "extends",
                "false",
                "finally",
                "for",
                "function",
                "if",
                "import",
                "in",
                "instanceof",
                "new",
                "null",
                "return",
                "super",
                "switch",
                "this",
                "throw",
                "true",
                "try",
                "typeof",
                "var",
                "void",
                "while",
                "with",
                "yield",
                "let",
                "static",
                "enum",
                "implements",
                "package",
                "protected",
                "interface",
                "private",
                "public",
                "async",
                "await",
            ],
            Language::Java => &[
                "abstract",
                "assert",
                "boolean",
                "break",
                "byte",
                "case",
                "catch",
                "char",
                "class",
                "const",
                "continue",
                "default",
                "do",
                "double",
                "else",
                "enum",
                "extends",
                "final",
                "finally",
                "float",
                "for",
                "goto",
                "if",
                "implements",
                "import",
                "instanceof",
                "int",
                "interface",
                "long",
                "native",
                "new",
                "package",
                "private",
                "protected",
                "public",
                "return",
                "short",
                "static",
                "strictfp",
                "super",
                "switch",
                "synchronized",
                "this",
                "throw",
                "throws",
                "transient",
                "try",
                "void",
                "volatile",
                "while",
            ],
            Language::CSharp => &[
                "abstract",
                "as",
                "base",
                "bool",
                "break",
                "byte",
                "case",
                "catch",
                "char",
                "checked",
                "class",
                "const",
                "continue",
                "decimal",
                "default",
                "delegate",
                "do",
                "double",
                "else",
                "enum",
                "event",
                "explicit",
                "extern",
                "false",
                "finally",
                "fixed",
                "float",
                "for",
                "foreach",
                "goto",
                "if",
                "implicit",
                "in",
                "int",
                "interface",
                "internal",
                "is",
                "lock",
                "long",
                "namespace",
                "new",
                "null",
                "object",
                "operator",
                "out",
                "override",
                "params",
                "private",
                "protected",
                "public",
                "readonly",
                "ref",
                "return",
                "sbyte",
                "sealed",
                "short",
                "sizeof",
                "stackalloc",
                "static",
                "string",
                "struct",
                "switch",
                "this",
                "throw",
                "true",
                "try",
                "typeof",
                "uint",
                "ulong",
                "unchecked",
                "unsafe",
                "ushort",
                "using",
                "virtual",
                "void",
                "volatile",
                "while",
            ],
            Language::Go => &[
                "break",
                "case",
                "chan",
                "const",
                "continue",
                "default",
                "defer",
                "else",
                "fallthrough",
                "for",
                "func",
                "go",
                "goto",
                "if",
                "import",
                "interface",
                "map",
                "package",
                "range",
                "return",
                "select",
                "struct",
                "switch",
                "type",
                "var",
            ],
            _ => &[], // Add more languages as needed
        }
    }

    /// Get comment patterns for the language
    pub fn comment_patterns(&self) -> CommentPatterns {
        match self {
            Language::Rust
            | Language::JavaScript
            | Language::TypeScript
            | Language::Java
            | Language::CSharp
            | Language::CPlusPlus
            | Language::Go
            | Language::Swift
            | Language::Kotlin
            | Language::Scala => CommentPatterns {
                line_comment: Some("//"),
                block_comment: Some(("/*", "*/")),
                doc_comment: Some("///"),
            },
            Language::Python | Language::Ruby | Language::Shell => CommentPatterns {
                line_comment: Some("#"),
                block_comment: None,
                doc_comment: Some("#"),
            },
            Language::C => CommentPatterns {
                line_comment: None,
                block_comment: Some(("/*", "*/")),
                doc_comment: None,
            },
            Language::HTML | Language::XML => CommentPatterns {
                line_comment: None,
                block_comment: Some(("<!--", "-->")),
                doc_comment: None,
            },
            Language::CSS => CommentPatterns {
                line_comment: None,
                block_comment: Some(("/*", "*/")),
                doc_comment: None,
            },
            Language::SQL => CommentPatterns {
                line_comment: Some("--"),
                block_comment: Some(("/*", "*/")),
                doc_comment: None,
            },
            Language::Haskell => CommentPatterns {
                line_comment: Some("--"),
                block_comment: Some(("{-", "-}")),
                doc_comment: Some("-- |"),
            },
            _ => CommentPatterns {
                line_comment: None,
                block_comment: None,
                doc_comment: None,
            },
        }
    }

    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Option<Language> {
        let ext = ext.to_lowercase();
        [
            Language::Rust,
            Language::Python,
            Language::JavaScript,
            Language::TypeScript,
            Language::Java,
            Language::CSharp,
            Language::CPlusPlus,
            Language::C,
            Language::Go,
            Language::Ruby,
            Language::PHP,
            Language::Swift,
            Language::Kotlin,
            Language::Scala,
            Language::Haskell,
            Language::Clojure,
            Language::SQL,
            Language::HTML,
            Language::CSS,
            Language::JSON,
            Language::XML,
            Language::YAML,
            Language::Markdown,
            Language::Shell,
            Language::PowerShell,
            Language::R,
            Language::Matlab,
        ]
        .into_iter()
        .find(|&lang| lang.extensions().contains(&ext.as_str()))
    }
}

/// Comment patterns for a language
#[derive(Debug, Clone)]
pub struct CommentPatterns {
    pub line_comment: Option<&'static str>,
    pub block_comment: Option<(&'static str, &'static str)>,
    pub doc_comment: Option<&'static str>,
}

/// Token types for code
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodeTokenType {
    Keyword,
    Identifier,
    Literal(LiteralType),
    Operator,
    Punctuation,
    Comment,
    Whitespace,
    String,
    Number,
    Unknown,
}

/// Types of literals
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LiteralType {
    String,
    Character,
    Integer,
    Float,
    Boolean,
    Null,
}

/// A code token with type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeToken {
    pub text: String,
    pub token_type: CodeTokenType,
    pub position: TokenPosition,
    pub language: Language,
}

/// Position information for a token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPosition {
    pub line: usize,
    pub column: usize,
    pub start_offset: usize,
    pub end_offset: usize,
}

/// Configuration for code tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTokenizerConfig {
    pub language: Option<Language>,
    pub preserve_whitespace: bool,
    pub preserve_comments: bool,
    pub include_position_info: bool,
    pub normalize_identifiers: bool,
    pub max_token_length: Option<usize>,
    pub custom_keywords: Option<HashSet<String>>,
}

impl Default for CodeTokenizerConfig {
    fn default() -> Self {
        Self {
            language: None,
            preserve_whitespace: false,
            preserve_comments: true,
            include_position_info: false,
            normalize_identifiers: false,
            max_token_length: Some(128),
            custom_keywords: None,
        }
    }
}

/// Code tokenizer implementation
pub struct CodeTokenizer {
    config: CodeTokenizerConfig,
    keywords: HashSet<String>,
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_tokens: HashMap<String, u32>,
}

impl CodeTokenizer {
    /// Create a new code tokenizer
    pub fn new(config: CodeTokenizerConfig) -> Self {
        let mut tokenizer = Self {
            config,
            keywords: HashSet::new(),
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashMap::new(),
        };

        tokenizer.initialize_vocabulary();
        tokenizer
    }

    /// Create tokenizer for a specific language
    pub fn for_language(language: Language) -> Self {
        let config = CodeTokenizerConfig {
            language: Some(language),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Initialize vocabulary with common tokens
    fn initialize_vocabulary(&mut self) {
        let mut next_id = 0u32;

        // Add special tokens
        for special in &[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]",
        ] {
            self.add_token(special, &mut next_id);
            self.special_tokens.insert(special.to_string(), next_id - 1);
        }

        // Add language keywords
        if let Some(language) = self.config.language {
            for keyword in language.keywords() {
                self.keywords.insert(keyword.to_string());
                self.add_token(keyword, &mut next_id);
            }
        }

        // Add custom keywords
        if let Some(custom_keywords) = &self.config.custom_keywords {
            let keywords_to_add: Vec<String> = custom_keywords.iter().cloned().collect();
            for keyword in keywords_to_add {
                self.keywords.insert(keyword.clone());
                self.add_token(&keyword, &mut next_id);
            }
        }

        // Add common operators and punctuation
        for op in &[
            "+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "!", "&",
            "|", "^", "~", "<<", ">>", "++", "--", "+=", "-=", "*=", "/=", "%=", "(", ")", "[",
            "]", "{", "}", ";", ",", ".", ":", "::", "->", "=>", "?",
        ] {
            self.add_token(op, &mut next_id);
        }

        // Add common literals
        for literal in &["true", "false", "null", "undefined", "nil", "None"] {
            self.add_token(literal, &mut next_id);
        }
    }

    /// Add a token to the vocabulary
    fn add_token(&mut self, token: &str, next_id: &mut u32) {
        if !self.token_to_id.contains_key(token) {
            self.token_to_id.insert(token.to_string(), *next_id);
            self.id_to_token.insert(*next_id, token.to_string());
            *next_id += 1;
        }
    }

    /// Tokenize code into structured tokens
    pub fn tokenize_code(&self, code: &str) -> Result<Vec<CodeToken>> {
        let language = self.config.language.unwrap_or(Language::JavaScript);
        let comment_patterns = language.comment_patterns();

        let mut tokens = Vec::new();
        let mut current_line = 1;
        let mut current_column = 1;
        let mut char_indices = code.char_indices().peekable();

        while let Some((start_offset, ch)) = char_indices.next() {
            let token_start_line = current_line;
            let token_start_column = current_column;

            // Update position
            if ch == '\n' {
                current_line += 1;
                current_column = 1;
            } else {
                current_column += 1;
            }

            // Skip whitespace (unless preserving)
            if ch.is_whitespace() {
                if self.config.preserve_whitespace {
                    let (text, end_offset) =
                        self.consume_whitespace(&mut char_indices, start_offset, ch);
                    tokens.push(CodeToken {
                        text,
                        token_type: CodeTokenType::Whitespace,
                        position: TokenPosition {
                            line: token_start_line,
                            column: token_start_column,
                            start_offset,
                            end_offset,
                        },
                        language,
                    });
                }
                continue;
            }

            // Handle comments
            if let Some(token) = self.try_parse_comment(
                &mut char_indices,
                start_offset,
                ch,
                &comment_patterns,
                token_start_line,
                token_start_column,
                language,
            )? {
                if self.config.preserve_comments {
                    tokens.push(token);
                }
                continue;
            }

            // Handle string literals
            if ch == '"'
                || ch == '\''
                || (ch == '`' && matches!(language, Language::JavaScript | Language::TypeScript))
            {
                let token = self.parse_string_literal(
                    &mut char_indices,
                    start_offset,
                    ch,
                    token_start_line,
                    token_start_column,
                    language,
                )?;
                tokens.push(token);
                continue;
            }

            // Handle numeric literals
            if ch.is_ascii_digit()
                || (ch == '.'
                    && char_indices.peek().map(|(_, c)| c.is_ascii_digit()).unwrap_or(false))
            {
                let token = self.parse_numeric_literal(
                    &mut char_indices,
                    start_offset,
                    ch,
                    token_start_line,
                    token_start_column,
                    language,
                )?;
                tokens.push(token);
                continue;
            }

            // Handle identifiers and keywords
            if ch.is_alphabetic() || ch == '_' || ch == '$' {
                let token = self.parse_identifier(
                    &mut char_indices,
                    start_offset,
                    ch,
                    token_start_line,
                    token_start_column,
                    language,
                )?;
                tokens.push(token);
                continue;
            }

            // Handle operators and punctuation
            let token = self.parse_operator_or_punctuation(
                &mut char_indices,
                start_offset,
                ch,
                token_start_line,
                token_start_column,
                language,
            )?;
            tokens.push(token);
        }

        Ok(tokens)
    }

    /// Consume whitespace characters
    fn consume_whitespace(
        &self,
        char_indices: &mut std::iter::Peekable<std::str::CharIndices>,
        start_offset: usize,
        first_char: char,
    ) -> (String, usize) {
        let mut text = String::new();
        text.push(first_char);
        let mut end_offset = start_offset;

        while let Some((offset, ch)) = char_indices.peek() {
            if ch.is_whitespace() {
                text.push(*ch);
                end_offset = *offset;
                char_indices.next();
            } else {
                break;
            }
        }

        (text, end_offset)
    }

    /// Try to parse a comment
    fn try_parse_comment(
        &self,
        char_indices: &mut std::iter::Peekable<std::str::CharIndices>,
        start_offset: usize,
        first_char: char,
        patterns: &CommentPatterns,
        token_start_line: usize,
        token_start_column: usize,
        language: Language,
    ) -> Result<Option<CodeToken>> {
        // Check for line comments
        if let Some(line_comment) = patterns.line_comment {
            if first_char
                == line_comment.chars().next().expect("line_comment pattern must be non-empty")
            {
                if let Some(token) = self.try_parse_line_comment(
                    char_indices,
                    start_offset,
                    line_comment,
                    token_start_line,
                    token_start_column,
                    language,
                )? {
                    return Ok(Some(token));
                }
            }
        }

        // Check for block comments
        if let Some((start_delim, end_delim)) = patterns.block_comment {
            if first_char
                == start_delim
                    .chars()
                    .next()
                    .expect("block comment start delimiter must be non-empty")
            {
                if let Some(token) = self.try_parse_block_comment(
                    char_indices,
                    start_offset,
                    start_delim,
                    end_delim,
                    token_start_line,
                    token_start_column,
                    language,
                )? {
                    return Ok(Some(token));
                }
            }
        }

        Ok(None)
    }

    /// Parse a line comment
    fn try_parse_line_comment(
        &self,
        char_indices: &mut std::iter::Peekable<std::str::CharIndices>,
        start_offset: usize,
        comment_start: &str,
        token_start_line: usize,
        token_start_column: usize,
        language: Language,
    ) -> Result<Option<CodeToken>> {
        let mut text = String::new();
        text.push_str(comment_start);

        // Skip the remaining characters of the comment start
        for _ in 1..comment_start.len() {
            if let Some((_, ch)) = char_indices.next() {
                text.push(ch);
            }
        }

        // Read until end of line
        let mut end_offset = start_offset;
        while let Some((offset, ch)) = char_indices.peek() {
            if *ch == '\n' {
                break;
            }
            text.push(*ch);
            end_offset = *offset;
            char_indices.next();
        }

        Ok(Some(CodeToken {
            text,
            token_type: CodeTokenType::Comment,
            position: TokenPosition {
                line: token_start_line,
                column: token_start_column,
                start_offset,
                end_offset,
            },
            language,
        }))
    }

    /// Parse a block comment
    fn try_parse_block_comment(
        &self,
        char_indices: &mut std::iter::Peekable<std::str::CharIndices>,
        start_offset: usize,
        start_delim: &str,
        end_delim: &str,
        token_start_line: usize,
        token_start_column: usize,
        language: Language,
    ) -> Result<Option<CodeToken>> {
        let mut text = String::new();
        text.push_str(start_delim);

        // Skip the remaining characters of the start delimiter
        for _ in 1..start_delim.len() {
            if let Some((_, ch)) = char_indices.next() {
                text.push(ch);
            }
        }

        // Read until end delimiter
        let mut end_offset = start_offset;
        let end_chars: Vec<char> = end_delim.chars().collect();
        let mut buffer = Vec::new();

        for (offset, ch) in char_indices.by_ref() {
            text.push(ch);
            end_offset = offset;
            buffer.push(ch);

            // Keep only the last few characters needed to match end delimiter
            if buffer.len() > end_chars.len() {
                buffer.remove(0);
            }

            // Check if we've found the end delimiter
            if buffer.len() == end_chars.len() && buffer == end_chars {
                break;
            }
        }

        Ok(Some(CodeToken {
            text,
            token_type: CodeTokenType::Comment,
            position: TokenPosition {
                line: token_start_line,
                column: token_start_column,
                start_offset,
                end_offset,
            },
            language,
        }))
    }

    /// Parse a string literal
    fn parse_string_literal(
        &self,
        char_indices: &mut std::iter::Peekable<std::str::CharIndices>,
        start_offset: usize,
        quote_char: char,
        token_start_line: usize,
        token_start_column: usize,
        language: Language,
    ) -> Result<CodeToken> {
        let mut text = String::new();
        text.push(quote_char);
        let mut end_offset = start_offset;
        let mut escaped = false;

        for (offset, ch) in char_indices.by_ref() {
            text.push(ch);
            end_offset = offset;

            if escaped {
                escaped = false;
                continue;
            }

            if ch == '\\' {
                escaped = true;
                continue;
            }

            if ch == quote_char {
                break;
            }
        }

        Ok(CodeToken {
            text,
            token_type: CodeTokenType::String,
            position: TokenPosition {
                line: token_start_line,
                column: token_start_column,
                start_offset,
                end_offset,
            },
            language,
        })
    }

    /// Parse a numeric literal
    fn parse_numeric_literal(
        &self,
        char_indices: &mut std::iter::Peekable<std::str::CharIndices>,
        start_offset: usize,
        first_char: char,
        token_start_line: usize,
        token_start_column: usize,
        language: Language,
    ) -> Result<CodeToken> {
        let mut text = String::new();
        text.push(first_char);
        let mut end_offset = start_offset;
        let mut has_dot = first_char == '.';

        while let Some((offset, ch)) = char_indices.peek() {
            if ch.is_ascii_digit()
                || (*ch == '.' && !has_dot)
                || (*ch == 'e' || *ch == 'E')
                || (*ch == 'x' || *ch == 'X')
                || (*ch == '_')
                || ch.is_ascii_hexdigit()
            {
                if *ch == '.' {
                    has_dot = true;
                }
                text.push(*ch);
                end_offset = *offset;
                char_indices.next();
            } else {
                break;
            }
        }

        Ok(CodeToken {
            text,
            token_type: CodeTokenType::Number,
            position: TokenPosition {
                line: token_start_line,
                column: token_start_column,
                start_offset,
                end_offset,
            },
            language,
        })
    }

    /// Parse an identifier or keyword
    fn parse_identifier(
        &self,
        char_indices: &mut std::iter::Peekable<std::str::CharIndices>,
        start_offset: usize,
        first_char: char,
        token_start_line: usize,
        token_start_column: usize,
        language: Language,
    ) -> Result<CodeToken> {
        let mut text = String::new();
        text.push(first_char);
        let mut end_offset = start_offset;

        while let Some((offset, ch)) = char_indices.peek() {
            if ch.is_alphanumeric() || *ch == '_' || *ch == '$' {
                text.push(*ch);
                end_offset = *offset;
                char_indices.next();
            } else {
                break;
            }
        }

        let token_type = if self.keywords.contains(&text) {
            CodeTokenType::Keyword
        } else {
            CodeTokenType::Identifier
        };

        Ok(CodeToken {
            text,
            token_type,
            position: TokenPosition {
                line: token_start_line,
                column: token_start_column,
                start_offset,
                end_offset,
            },
            language,
        })
    }

    /// Parse an operator or punctuation
    fn parse_operator_or_punctuation(
        &self,
        char_indices: &mut std::iter::Peekable<std::str::CharIndices>,
        start_offset: usize,
        first_char: char,
        token_start_line: usize,
        token_start_column: usize,
        language: Language,
    ) -> Result<CodeToken> {
        let mut text = String::new();
        text.push(first_char);
        let mut end_offset = start_offset;

        // Try to form multi-character operators
        let operators = [
            "==", "!=", "<=", ">=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=", "%=", "<<",
            ">>", "::", "->", "=>", "**", "//", "...", "..", ":=", "<=>",
        ];

        for op in &operators {
            if op.starts_with(first_char) && op.len() > 1 {
                let chars = op.chars().skip(1);
                let mut matched = true;
                let mut lookahead = Vec::new();

                for expected_char in chars {
                    if let Some((offset, ch)) = char_indices.peek() {
                        if *ch == expected_char {
                            lookahead.push((*offset, *ch));
                            char_indices.next();
                        } else {
                            matched = false;
                            break;
                        }
                    } else {
                        matched = false;
                        break;
                    }
                }

                if matched {
                    text = op.to_string();
                    if let Some((offset, _)) = lookahead.last() {
                        end_offset = *offset;
                    }
                    break;
                } else {
                    // Put back the consumed characters
                    for (_, _ch) in lookahead.into_iter().rev() {
                        // Note: This is a simplified approach. In a real implementation,
                        // you'd need a more sophisticated way to put back characters.
                    }
                }
            }
        }

        let token_type = match first_char {
            '(' | ')' | '[' | ']' | '{' | '}' | ';' | ',' | '.' | ':' => CodeTokenType::Punctuation,
            _ => CodeTokenType::Operator,
        };

        Ok(CodeToken {
            text,
            token_type,
            position: TokenPosition {
                line: token_start_line,
                column: token_start_column,
                start_offset,
                end_offset,
            },
            language,
        })
    }

    /// Get or create token ID
    #[allow(dead_code)]
    fn get_or_create_token_id(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            id
        } else {
            let id = self.token_to_id.len() as u32;
            self.token_to_id.insert(token.to_string(), id);
            self.id_to_token.insert(id, token.to_string());
            id
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Get token ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token from ID
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token.get(&id).cloned()
    }
}

impl Tokenizer for CodeTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let code_tokens = self.tokenize_code(text)?;
        let mut input_ids = Vec::new();

        for token in code_tokens {
            let token_text = if self.config.normalize_identifiers
                && token.token_type == CodeTokenType::Identifier
            {
                "[IDENTIFIER]".to_string()
            } else {
                token.text
            };

            if let Some(id) = self.token_to_id(&token_text) {
                input_ids.push(id);
            } else if let Some(&unk_id) = self.special_tokens.get("[UNK]") {
                input_ids.push(unk_id);
            }
        }

        let attention_mask = vec![1u8; input_ids.len()];

        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let tokens: Vec<String> = ids.iter().filter_map(|&id| self.id_to_token(id)).collect();
        Ok(tokens.join(" "))
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        let combined = format!("{}\n{}", text_a, text_b);
        self.encode(&combined)
    }

    fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.token_to_id.clone()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token.get(&id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection() {
        assert_eq!(Language::from_extension("rs"), Some(Language::Rust));
        assert_eq!(Language::from_extension("py"), Some(Language::Python));
        assert_eq!(Language::from_extension("js"), Some(Language::JavaScript));
        assert_eq!(Language::from_extension("unknown"), None);
    }

    #[test]
    fn test_rust_tokenization() {
        let tokenizer = CodeTokenizer::for_language(Language::Rust);
        let code = "fn main() { let x = 42; }";
        let tokens = tokenizer.tokenize_code(code).unwrap();

        assert!(!tokens.is_empty());

        // Check for keywords
        let fn_token = tokens.iter().find(|t| t.text == "fn").unwrap();
        assert_eq!(fn_token.token_type, CodeTokenType::Keyword);

        let let_token = tokens.iter().find(|t| t.text == "let").unwrap();
        assert_eq!(let_token.token_type, CodeTokenType::Keyword);
    }

    #[test]
    fn test_string_literal_parsing() {
        let tokenizer = CodeTokenizer::for_language(Language::JavaScript);
        let code = r#"let name = "Hello \"World\"";"#;
        let tokens = tokenizer.tokenize_code(code).unwrap();

        let string_token = tokens.iter().find(|t| t.token_type == CodeTokenType::String).unwrap();
        assert!(string_token.text.starts_with('"'));
        assert!(string_token.text.ends_with('"'));
    }

    #[test]
    fn test_comment_parsing() {
        let config = CodeTokenizerConfig {
            language: Some(Language::Rust),
            preserve_comments: true,
            ..Default::default()
        };
        let tokenizer = CodeTokenizer::new(config);
        let code = "// This is a comment\nfn main() {}";
        let tokens = tokenizer.tokenize_code(code).unwrap();

        let comment_token = tokens.iter().find(|t| t.token_type == CodeTokenType::Comment).unwrap();
        assert!(comment_token.text.starts_with("//"));
    }

    #[test]
    fn test_numeric_literals() {
        let tokenizer = CodeTokenizer::for_language(Language::Python);
        let code = "x = 42; y = 3.14; z = 0xFF;";
        let tokens = tokenizer.tokenize_code(code).unwrap();

        let numeric_tokens: Vec<_> =
            tokens.iter().filter(|t| t.token_type == CodeTokenType::Number).collect();

        assert!(numeric_tokens.len() >= 3);
    }

    #[test]
    fn test_code_tokenizer_encode() {
        let tokenizer = CodeTokenizer::for_language(Language::Python);
        let code = "def hello(): return 42";
        let result = tokenizer.encode(code).unwrap();

        assert!(!result.input_ids.is_empty());
        assert_eq!(result.input_ids.len(), result.attention_mask.len());
    }
}
