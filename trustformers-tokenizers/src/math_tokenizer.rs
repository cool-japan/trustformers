use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Types of mathematical tokens
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MathTokenType {
    /// Numbers (integers, floats, scientific notation)
    Number,
    /// Variables (single letters, subscripted/superscripted)
    Variable,
    /// Mathematical operators (+, -, *, /, ^, etc.)
    Operator,
    /// Mathematical functions (sin, cos, log, etc.)
    Function,
    /// Greek letters (α, β, γ, etc.)
    GreekLetter,
    /// Mathematical constants (π, e, ∞, etc.)
    Constant,
    /// Delimiters (parentheses, brackets, braces)
    Delimiter,
    /// LaTeX commands (\frac, \sum, \int, etc.)
    LaTeXCommand,
    /// Subscripts and superscripts
    Script,
    /// Mathematical symbols (∈, ∀, ∃, etc.)
    Symbol,
    /// Units (m, kg, s, etc.)
    Unit,
    /// Text within math (for labels, etc.)
    Text,
    /// Whitespace
    Whitespace,
    /// Unknown/other
    Unknown,
}

/// A mathematical token with position and type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathToken {
    /// The token text
    pub text: String,
    /// Token type
    pub token_type: MathTokenType,
    /// Start position in original text
    pub start: usize,
    /// End position in original text
    pub end: usize,
    /// Token ID (assigned during tokenization)
    pub id: Option<u32>,
    /// LaTeX representation if different from text
    pub latex: Option<String>,
    /// Mathematical meaning/description
    pub meaning: Option<String>,
}

impl MathToken {
    /// Create a new math token
    pub fn new(text: String, token_type: MathTokenType, start: usize, end: usize) -> Self {
        Self {
            text,
            token_type,
            start,
            end,
            id: None,
            latex: None,
            meaning: None,
        }
    }

    /// Set the token ID
    pub fn with_id(mut self, id: u32) -> Self {
        self.id = Some(id);
        self
    }

    /// Set the LaTeX representation
    pub fn with_latex(mut self, latex: String) -> Self {
        self.latex = Some(latex);
        self
    }

    /// Set the meaning
    pub fn with_meaning(mut self, meaning: String) -> Self {
        self.meaning = Some(meaning);
        self
    }
}

/// Configuration for math tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathTokenizerConfig {
    /// Whether to preserve whitespace tokens
    pub preserve_whitespace: bool,
    /// Whether to recognize LaTeX commands
    pub recognize_latex: bool,
    /// Whether to recognize scientific notation
    pub recognize_scientific_notation: bool,
    /// Whether to handle subscripts and superscripts
    pub handle_scripts: bool,
    /// Whether to recognize units
    pub recognize_units: bool,
    /// Maximum token length
    pub max_token_length: usize,
    /// Custom function names to recognize
    pub custom_functions: HashSet<String>,
    /// Custom constants to recognize
    pub custom_constants: HashMap<String, String>, // symbol -> meaning
}

impl Default for MathTokenizerConfig {
    fn default() -> Self {
        Self {
            preserve_whitespace: false,
            recognize_latex: true,
            recognize_scientific_notation: true,
            handle_scripts: true,
            recognize_units: true,
            max_token_length: 50,
            custom_functions: HashSet::new(),
            custom_constants: HashMap::new(),
        }
    }
}

/// Mathematical formula tokenizer
pub struct MathTokenizer {
    config: MathTokenizerConfig,
    /// Regular expressions for tokenization
    number_regex: Regex,
    scientific_regex: Regex,
    latex_command_regex: Regex,
    greek_letters: HashSet<String>,
    math_functions: HashSet<String>,
    math_constants: HashMap<String, String>,
    math_operators: HashSet<String>,
    math_symbols: HashMap<String, String>,
    units: HashSet<String>,
    /// Vocabulary mappings
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    next_id: u32,
}

impl MathTokenizer {
    /// Create a new math tokenizer with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(MathTokenizerConfig::default())
    }

    /// Create a new math tokenizer with custom configuration
    pub fn with_config(config: MathTokenizerConfig) -> Result<Self> {
        let number_regex = Regex::new(r"^\d+\.?\d*$").unwrap();
        let scientific_regex = Regex::new(r"^\d+\.?\d*[eE][+-]?\d+$").unwrap();
        let latex_command_regex = Regex::new(r"^\\[a-zA-Z]+$").unwrap();

        // Build Greek letters set
        let greek_letters = [
            "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ",
            "σ", "τ", "υ", "φ", "χ", "ψ", "ω", "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ", "Ι", "Κ",
            "Λ", "Μ", "Ν", "Ξ", "Ο", "Π", "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        // Build math functions set
        let mut math_functions = [
            "sin", "cos", "tan", "sec", "csc", "cot", "arcsin", "arccos", "arctan", "asin", "acos",
            "atan", "sinh", "cosh", "tanh", "sech", "csch", "coth", "log", "ln", "lg", "exp",
            "sqrt", "abs", "sgn", "min", "max", "gcd", "lcm", "floor", "ceil", "det", "tr", "rank",
            "dim", "span", "ker", "im",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<HashSet<_>>();

        // Add custom functions
        math_functions.extend(config.custom_functions.clone());

        // Build math constants
        let mut math_constants = HashMap::new();
        math_constants.insert("π".to_string(), "pi".to_string());
        math_constants.insert("e".to_string(), "euler_number".to_string());
        math_constants.insert("∞".to_string(), "infinity".to_string());
        math_constants.insert("i".to_string(), "imaginary_unit".to_string());
        math_constants.insert("φ".to_string(), "golden_ratio".to_string());
        math_constants.insert("γ".to_string(), "euler_gamma".to_string());

        // Add custom constants
        math_constants.extend(config.custom_constants.clone());

        // Build operators set
        let math_operators = [
            "+", "-", "*", "×", "·", "/", "÷", "^", "=", "≠", "≈", "≡", "<", ">", "≤", "≥", "≪",
            "≫", "±", "∓", "∝", "∼", "≅",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        // Build symbols map
        let mut math_symbols = HashMap::new();
        math_symbols.insert("∈".to_string(), "element_of".to_string());
        math_symbols.insert("∉".to_string(), "not_element_of".to_string());
        math_symbols.insert("⊂".to_string(), "subset".to_string());
        math_symbols.insert("⊃".to_string(), "superset".to_string());
        math_symbols.insert("⊆".to_string(), "subset_equal".to_string());
        math_symbols.insert("⊇".to_string(), "superset_equal".to_string());
        math_symbols.insert("∪".to_string(), "union".to_string());
        math_symbols.insert("∩".to_string(), "intersection".to_string());
        math_symbols.insert("∅".to_string(), "empty_set".to_string());
        math_symbols.insert("∀".to_string(), "for_all".to_string());
        math_symbols.insert("∃".to_string(), "exists".to_string());
        math_symbols.insert("∄".to_string(), "not_exists".to_string());
        math_symbols.insert("∇".to_string(), "nabla".to_string());
        math_symbols.insert("∂".to_string(), "partial".to_string());
        math_symbols.insert("∫".to_string(), "integral".to_string());
        math_symbols.insert("∬".to_string(), "double_integral".to_string());
        math_symbols.insert("∭".to_string(), "triple_integral".to_string());
        math_symbols.insert("∮".to_string(), "contour_integral".to_string());
        math_symbols.insert("∑".to_string(), "sum".to_string());
        math_symbols.insert("∏".to_string(), "product".to_string());
        math_symbols.insert("→".to_string(), "right_arrow".to_string());
        math_symbols.insert("←".to_string(), "left_arrow".to_string());
        math_symbols.insert("↔".to_string(), "double_arrow".to_string());
        math_symbols.insert("⇒".to_string(), "implies".to_string());
        math_symbols.insert("⇔".to_string(), "if_and_only_if".to_string());
        math_symbols.insert("√".to_string(), "square_root".to_string());

        // Build units set
        let units = [
            // Length
            "m", "cm", "mm", "km", "in", "ft", "yd", "mi", // Mass
            "g", "kg", "mg", "lb", "oz", // Time
            "s", "ms", "min", "h", "hr", "day", "yr", // Temperature
            "K", "°C", "°F", // Energy
            "J", "kJ", "cal", "kcal", "eV", "keV", "MeV", "GeV", // Power
            "W", "kW", "MW", "hp", // Frequency
            "Hz", "kHz", "MHz", "GHz", // Voltage
            "V", "mV", "kV", // Current
            "A", "mA", "μA", // Resistance
            "Ω", "kΩ", "MΩ",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let mut tokenizer = Self {
            config,
            number_regex,
            scientific_regex,
            latex_command_regex,
            greek_letters,
            math_functions,
            math_constants,
            math_operators,
            math_symbols,
            units,
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            next_id: 1, // Reserve 0 for special tokens
        };

        // Initialize vocabulary with all mathematical tokens
        tokenizer.initialize_vocabulary();
        Ok(tokenizer)
    }

    /// Initialize vocabulary with all mathematical tokens
    fn initialize_vocabulary(&mut self) {
        // Collect data to avoid borrowing issues
        let greek_letters: Vec<String> = self.greek_letters.iter().cloned().collect();
        let math_functions: Vec<String> = self.math_functions.iter().cloned().collect();
        let math_constants: Vec<String> = self.math_constants.keys().cloned().collect();
        let math_operators: Vec<String> = self.math_operators.iter().cloned().collect();
        let math_symbols: Vec<String> = self.math_symbols.keys().cloned().collect();
        let units: Vec<String> = self.units.iter().cloned().collect();

        // Add Greek letters
        for letter in &greek_letters {
            self.add_token_with_type(letter, "GreekLetter");
        }

        // Add mathematical functions
        for function in &math_functions {
            self.add_token_with_type(function, "Function");
        }

        // Add mathematical constants
        for constant in &math_constants {
            self.add_token_with_type(constant, "Constant");
        }

        // Add mathematical operators
        for operator in &math_operators {
            self.add_token_with_type(operator, "Operator");
        }

        // Add mathematical symbols
        for symbol in &math_symbols {
            self.add_token_with_type(symbol, "Symbol");
        }

        // Add units
        for unit in &units {
            self.add_token_with_type(unit, "Unit");
        }

        // Add common variables
        for var in ["x", "y", "z", "a", "b", "c", "n", "m", "k", "i", "j"] {
            self.add_token_with_type(var, "Variable");
        }

        // Add numbers 0-9
        for num in 0..10 {
            self.add_token_with_type(&num.to_string(), "Number");
        }

        // Add common punctuation and delimiters
        for punct in ["(", ")", "[", "]", "{", "}", ",", ".", "!", " "] {
            self.add_token_with_type(punct, "Punctuation");
        }
    }

    /// Add a token with type information to the vocabulary
    fn add_token_with_type(&mut self, token: &str, token_type: &str) {
        let key = format!("{}:{}", token, token_type);
        if !self.token_to_id.contains_key(&key) {
            let id = self.next_id;
            self.token_to_id.insert(key.clone(), id);
            self.id_to_token.insert(id, key);
            self.next_id += 1;
        }
    }

    /// Tokenize mathematical text into MathTokens
    pub fn tokenize_math(&mut self, text: &str) -> Result<Vec<MathToken>> {
        let mut tokens = Vec::new();
        let mut chars = text.char_indices().peekable();

        while let Some((pos, ch)) = chars.next() {
            let start_pos = pos;

            // Skip whitespace unless configured to preserve it
            if ch.is_whitespace() {
                if self.config.preserve_whitespace {
                    let mut whitespace = String::new();
                    whitespace.push(ch);

                    // Collect consecutive whitespace
                    while let Some(&(_, next_ch)) = chars.peek() {
                        if next_ch.is_whitespace() {
                            chars.next();
                            whitespace.push(next_ch);
                        } else {
                            break;
                        }
                    }

                    let end_pos = start_pos + whitespace.len();
                    tokens.push(MathToken::new(
                        whitespace,
                        MathTokenType::Whitespace,
                        start_pos,
                        end_pos,
                    ));
                }
                continue;
            }

            // Try to match different token types
            if let Some(token) = self.try_match_number(&mut chars, ch, start_pos)? {
                tokens.push(token);
            } else if let Some(token) = self.try_match_latex_command(&mut chars, ch, start_pos)? {
                tokens.push(token);
            } else if let Some(token) = self.try_match_function(&mut chars, ch, start_pos)? {
                tokens.push(token);
            } else if let Some(token) =
                self.try_match_multi_char_symbol(&mut chars, ch, start_pos)?
            {
                tokens.push(token);
            } else {
                // Single character token
                let token_text = ch.to_string();
                let token_type = self.classify_single_char(&token_text);
                let end_pos = start_pos + ch.len_utf8();

                tokens.push(MathToken::new(token_text, token_type, start_pos, end_pos));
            }
        }

        Ok(tokens)
    }

    /// Try to match a number (including scientific notation)
    fn try_match_number(
        &self,
        chars: &mut std::iter::Peekable<std::str::CharIndices>,
        first_char: char,
        start_pos: usize,
    ) -> Result<Option<MathToken>> {
        if !first_char.is_ascii_digit() && first_char != '.' {
            return Ok(None);
        }

        let mut number = String::new();
        number.push(first_char);
        let mut current_pos = start_pos + first_char.len_utf8();

        // Collect digits and decimal point
        while let Some(&(_, ch)) = chars.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                chars.next();
                number.push(ch);
                current_pos += ch.len_utf8();
            } else {
                break;
            }
        }

        // Check for scientific notation
        if let Some(&(_, ch)) = chars.peek() {
            if ch == 'e' || ch == 'E' {
                let mut temp_number = number.clone();
                temp_number.push(ch);
                chars.next();
                current_pos += ch.len_utf8();

                // Check for optional sign
                if let Some(&(_, sign_ch)) = chars.peek() {
                    if sign_ch == '+' || sign_ch == '-' {
                        chars.next();
                        temp_number.push(sign_ch);
                        current_pos += sign_ch.len_utf8();
                    }
                }

                // Must have digits after E
                let mut has_exponent_digits = false;
                while let Some(&(_, ch)) = chars.peek() {
                    if ch.is_ascii_digit() {
                        chars.next();
                        temp_number.push(ch);
                        current_pos += ch.len_utf8();
                        has_exponent_digits = true;
                    } else {
                        break;
                    }
                }

                if has_exponent_digits {
                    number = temp_number;
                }
            }
        }

        let token_type =
            if (self.config.recognize_scientific_notation && self.scientific_regex.is_match(&number))
                || self.number_regex.is_match(&number)
            {
                MathTokenType::Number
            } else {
                MathTokenType::Unknown
            };

        Ok(Some(MathToken::new(
            number,
            token_type,
            start_pos,
            current_pos,
        )))
    }

    /// Try to match a LaTeX command
    fn try_match_latex_command(
        &self,
        chars: &mut std::iter::Peekable<std::str::CharIndices>,
        first_char: char,
        start_pos: usize,
    ) -> Result<Option<MathToken>> {
        if !self.config.recognize_latex || first_char != '\\' {
            return Ok(None);
        }

        let mut command = String::new();
        command.push(first_char);
        let mut current_pos = start_pos + first_char.len_utf8();

        // Collect alphabetic characters
        while let Some(&(_, ch)) = chars.peek() {
            if ch.is_ascii_alphabetic() {
                chars.next();
                command.push(ch);
                current_pos += ch.len_utf8();
            } else {
                break;
            }
        }

        if command.len() > 1 && self.latex_command_regex.is_match(&command) {
            Ok(Some(MathToken::new(
                command,
                MathTokenType::LaTeXCommand,
                start_pos,
                current_pos,
            )))
        } else {
            Ok(None)
        }
    }

    /// Try to match a mathematical function
    fn try_match_function(
        &self,
        chars: &mut std::iter::Peekable<std::str::CharIndices>,
        first_char: char,
        start_pos: usize,
    ) -> Result<Option<MathToken>> {
        if !first_char.is_ascii_alphabetic() {
            return Ok(None);
        }

        let mut function = String::new();
        function.push(first_char);
        let mut current_pos = start_pos + first_char.len_utf8();

        // Look ahead to build potential function name
        let saved_chars = chars.clone();
        while let Some(&(_, ch)) = chars.peek() {
            if ch.is_ascii_alphabetic() {
                chars.next();
                function.push(ch);
                current_pos += ch.len_utf8();
            } else {
                break;
            }
        }

        if self.math_functions.contains(&function) {
            Ok(Some(MathToken::new(
                function,
                MathTokenType::Function,
                start_pos,
                current_pos,
            )))
        } else {
            // Restore chars iterator and try shorter match
            *chars = saved_chars;
            Ok(None)
        }
    }

    /// Try to match multi-character symbols
    fn try_match_multi_char_symbol(
        &self,
        chars: &mut std::iter::Peekable<std::str::CharIndices>,
        first_char: char,
        start_pos: usize,
    ) -> Result<Option<MathToken>> {
        // Look for 2-3 character symbols first
        let mut symbol = String::new();
        symbol.push(first_char);
        let mut current_pos = start_pos + first_char.len_utf8();

        // Try to match longer symbols first
        let saved_chars = chars.clone();
        for _ in 0..2 {
            if let Some(&(_, ch)) = chars.peek() {
                let temp_symbol = format!("{}{}", symbol, ch);
                if self.math_symbols.contains_key(&temp_symbol)
                    || self.math_operators.contains(&temp_symbol)
                {
                    chars.next();
                    symbol = temp_symbol;
                    current_pos += ch.len_utf8();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if symbol.chars().count() > 1 {
            let token_type = if self.math_symbols.contains_key(&symbol) {
                MathTokenType::Symbol
            } else if self.math_operators.contains(&symbol) {
                MathTokenType::Operator
            } else {
                MathTokenType::Unknown
            };
            Ok(Some(MathToken::new(
                symbol,
                token_type,
                start_pos,
                current_pos,
            )))
        } else {
            *chars = saved_chars;
            Ok(None)
        }
    }

    /// Classify a single character token
    fn classify_single_char(&self, ch: &str) -> MathTokenType {
        if self.greek_letters.contains(ch) {
            MathTokenType::GreekLetter
        } else if self.math_constants.contains_key(ch) {
            MathTokenType::Constant
        } else if self.math_operators.contains(ch) {
            MathTokenType::Operator
        } else if self.math_symbols.contains_key(ch) {
            MathTokenType::Symbol
        } else if self.units.contains(ch) {
            MathTokenType::Unit
        } else if matches!(ch, "(" | ")" | "[" | "]" | "{" | "}") {
            MathTokenType::Delimiter
        } else if ch.chars().all(|c| c.is_ascii_alphabetic()) {
            MathTokenType::Variable
        } else {
            MathTokenType::Unknown
        }
    }

    /// Assign IDs to tokens and build vocabulary
    fn assign_token_ids(&mut self, tokens: &mut [MathToken]) {
        for token in tokens {
            let token_key = format!("{}:{:?}", token.text, token.token_type);

            if let Some(&id) = self.token_to_id.get(&token_key) {
                token.id = Some(id);
            } else {
                let id = self.next_id;
                self.next_id += 1;

                self.token_to_id.insert(token_key.clone(), id);
                self.id_to_token.insert(id, token_key);
                token.id = Some(id);
            }
        }
    }

    /// Convert MathTokens to standard TokenizedInput
    pub fn math_tokens_to_input(&mut self, mut tokens: Vec<MathToken>) -> TokenizedInput {
        self.assign_token_ids(&mut tokens);

        let input_ids: Vec<u32> = tokens.iter().filter_map(|t| t.id).collect();

        let token_strings: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();

        let _offsets: Vec<(u32, u32)> =
            tokens.iter().map(|t| (t.start as u32, t.end as u32)).collect();

        TokenizedInput {
            input_ids,
            attention_mask: vec![1u8; token_strings.len()],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        }
    }

    /// Get mathematical analysis of tokenized text
    pub fn analyze_math(&self, tokens: &[MathToken]) -> MathAnalysis {
        let mut analysis = MathAnalysis::new();

        for token in tokens {
            analysis.total_tokens += 1;

            match token.token_type {
                MathTokenType::Number => analysis.numbers += 1,
                MathTokenType::Variable => analysis.variables += 1,
                MathTokenType::Operator => analysis.operators += 1,
                MathTokenType::Function => analysis.functions += 1,
                MathTokenType::GreekLetter => analysis.greek_letters += 1,
                MathTokenType::Constant => analysis.constants += 1,
                MathTokenType::Delimiter => analysis.delimiters += 1,
                MathTokenType::LaTeXCommand => analysis.latex_commands += 1,
                MathTokenType::Script => analysis.scripts += 1,
                MathTokenType::Symbol => analysis.symbols += 1,
                MathTokenType::Unit => analysis.units += 1,
                MathTokenType::Text => analysis.text_tokens += 1,
                MathTokenType::Whitespace => analysis.whitespace += 1,
                MathTokenType::Unknown => analysis.unknown += 1,
            }

            // Track unique tokens
            analysis.unique_tokens.insert(token.text.clone());

            // Track function names
            if token.token_type == MathTokenType::Function {
                *analysis.function_frequency.entry(token.text.clone()).or_insert(0) += 1;
            }

            // Track operators
            if token.token_type == MathTokenType::Operator {
                *analysis.operator_frequency.entry(token.text.clone()).or_insert(0) += 1;
            }
        }

        analysis.unique_token_count = analysis.unique_tokens.len();
        analysis
    }

    /// Get vocabulary statistics
    pub fn vocab_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_tokens".to_string(), self.token_to_id.len());
        stats.insert("next_id".to_string(), self.next_id as usize);

        // Count by token type
        let mut type_counts: HashMap<MathTokenType, usize> = HashMap::new();
        for key in self.token_to_id.keys() {
            if let Some(type_str) = key.split(':').nth(1) {
                if let Ok(token_type) =
                    serde_json::from_str::<MathTokenType>(&format!("\"{}\"", type_str))
                {
                    *type_counts.entry(token_type).or_insert(0) += 1;
                }
            }
        }

        for (token_type, count) in type_counts {
            stats.insert(format!("{:?}_tokens", token_type).to_lowercase(), count);
        }

        stats
    }
}

impl Default for MathTokenizer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl Tokenizer for MathTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let mut tokenizer = self.clone();
        let tokens = tokenizer.tokenize_math(text)?;
        Ok(tokenizer.math_tokens_to_input(tokens))
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let tokens: std::result::Result<Vec<String>, TrustformersError> = token_ids
            .iter()
            .map(|&id| {
                self.id_to_token
                    .get(&id)
                    .and_then(|key| key.split(':').next())
                    .map(|s| s.to_string())
                    .ok_or_else(|| TrustformersError::other(format!("Unknown token ID: {}", id)))
            })
            .collect();

        Ok(tokens?.join(" "))
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.token_to_id
            .iter()
            .map(|(key, &id)| {
                let token = key.split(':').next().unwrap_or(key).to_string();
                (token, id)
            })
            .collect()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        // Try exact match first
        self.token_to_id.get(token).copied().or_else(|| {
            // Try with different token types
            for token_type in [
                MathTokenType::Number,
                MathTokenType::Variable,
                MathTokenType::Operator,
                MathTokenType::Function,
                MathTokenType::GreekLetter,
                MathTokenType::Constant,
            ] {
                let key = format!("{}:{:?}", token, token_type);
                if let Some(&id) = self.token_to_id.get(&key) {
                    return Some(id);
                }
            }
            None
        })
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token
            .get(&id)
            .and_then(|key| key.split(':').next())
            .map(|s| s.to_string())
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        let mut tokenizer = self.clone();
        let tokens_a = tokenizer.tokenize_math(text_a)?;
        let tokens_b = tokenizer.tokenize_math(text_b)?;

        let mut combined_tokens = tokens_a;
        combined_tokens.push(MathToken {
            text: "[SEP]".to_string(),
            token_type: MathTokenType::Symbol,
            start: 0,
            end: 5,
            id: None,
            latex: None,
            meaning: None,
        });
        combined_tokens.extend(tokens_b);

        Ok(tokenizer.math_tokens_to_input(combined_tokens))
    }

    fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }
}

// Make MathTokenizer cloneable for the Tokenizer trait
impl Clone for MathTokenizer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            number_regex: Regex::new(r"^\d+\.?\d*$").unwrap(),
            scientific_regex: Regex::new(r"^\d+\.?\d*[eE][+-]?\d+$").unwrap(),
            latex_command_regex: Regex::new(r"^\\[a-zA-Z]+$").unwrap(),
            greek_letters: self.greek_letters.clone(),
            math_functions: self.math_functions.clone(),
            math_constants: self.math_constants.clone(),
            math_operators: self.math_operators.clone(),
            math_symbols: self.math_symbols.clone(),
            units: self.units.clone(),
            token_to_id: self.token_to_id.clone(),
            id_to_token: self.id_to_token.clone(),
            next_id: self.next_id,
        }
    }
}

/// Analysis results for mathematical text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathAnalysis {
    pub total_tokens: usize,
    pub unique_token_count: usize,
    pub numbers: usize,
    pub variables: usize,
    pub operators: usize,
    pub functions: usize,
    pub greek_letters: usize,
    pub constants: usize,
    pub delimiters: usize,
    pub latex_commands: usize,
    pub scripts: usize,
    pub symbols: usize,
    pub units: usize,
    pub text_tokens: usize,
    pub whitespace: usize,
    pub unknown: usize,
    pub unique_tokens: HashSet<String>,
    pub function_frequency: HashMap<String, usize>,
    pub operator_frequency: HashMap<String, usize>,
}

impl MathAnalysis {
    fn new() -> Self {
        Self {
            total_tokens: 0,
            unique_token_count: 0,
            numbers: 0,
            variables: 0,
            operators: 0,
            functions: 0,
            greek_letters: 0,
            constants: 0,
            delimiters: 0,
            latex_commands: 0,
            scripts: 0,
            symbols: 0,
            units: 0,
            text_tokens: 0,
            whitespace: 0,
            unknown: 0,
            unique_tokens: HashSet::new(),
            function_frequency: HashMap::new(),
            operator_frequency: HashMap::new(),
        }
    }

    /// Get the most common functions
    pub fn top_functions(&self, n: usize) -> Vec<(String, usize)> {
        let mut functions: Vec<(String, usize)> =
            self.function_frequency.iter().map(|(k, &v)| (k.clone(), v)).collect();
        functions.sort_by(|a, b| b.1.cmp(&a.1));
        functions.into_iter().take(n).collect()
    }

    /// Get the most common operators
    pub fn top_operators(&self, n: usize) -> Vec<(String, usize)> {
        let mut operators: Vec<(String, usize)> =
            self.operator_frequency.iter().map(|(k, &v)| (k.clone(), v)).collect();
        operators.sort_by(|a, b| b.1.cmp(&a.1));
        operators.into_iter().take(n).collect()
    }

    /// Calculate complexity score based on token diversity
    pub fn complexity_score(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }

        let _type_diversity = (self.functions + self.symbols + self.latex_commands) as f64;
        let token_diversity = self.unique_token_count as f64 / self.total_tokens as f64;

        // Combine different factors
        // Weight mathematical complexity higher
        let advanced_math_score =
            (self.functions * 3 + self.symbols * 2 + self.latex_commands * 3) as f64;
        let greek_constants_score = (self.greek_letters + self.constants) as f64;
        let operator_complexity = self.operators as f64 * 0.5;

        advanced_math_score + greek_constants_score + operator_complexity + (token_diversity * 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_tokenizer_creation() {
        let tokenizer = MathTokenizer::new().unwrap();
        assert!(tokenizer.math_functions.contains("sin"));
        assert!(tokenizer.greek_letters.contains("π"));
        assert!(tokenizer.math_operators.contains("+"));
    }

    #[test]
    fn test_number_tokenization() {
        let mut tokenizer = MathTokenizer::new().unwrap();
        let tokens = tokenizer.tokenize_math("123 3.14 2e10").unwrap();

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "123");
        assert_eq!(tokens[0].token_type, MathTokenType::Number);
        assert_eq!(tokens[1].text, "3.14");
        assert_eq!(tokens[1].token_type, MathTokenType::Number);
        assert_eq!(tokens[2].text, "2e10");
        assert_eq!(tokens[2].token_type, MathTokenType::Number);
    }

    #[test]
    fn test_function_tokenization() {
        let mut tokenizer = MathTokenizer::new().unwrap();
        let tokens = tokenizer.tokenize_math("sin(x) cos(θ) log(n)").unwrap();

        let function_tokens: Vec<&MathToken> =
            tokens.iter().filter(|t| t.token_type == MathTokenType::Function).collect();

        assert_eq!(function_tokens.len(), 3);
        assert_eq!(function_tokens[0].text, "sin");
        assert_eq!(function_tokens[1].text, "cos");
        assert_eq!(function_tokens[2].text, "log");
    }

    #[test]
    fn test_latex_commands() {
        let mut tokenizer = MathTokenizer::new().unwrap();
        let tokens = tokenizer.tokenize_math("\\frac{x}{y} \\sum_{i=1}^n").unwrap();

        let latex_tokens: Vec<&MathToken> =
            tokens.iter().filter(|t| t.token_type == MathTokenType::LaTeXCommand).collect();

        assert_eq!(latex_tokens.len(), 2);
        assert_eq!(latex_tokens[0].text, "\\frac");
        assert_eq!(latex_tokens[1].text, "\\sum");
    }

    #[test]
    fn test_greek_letters() {
        let mut tokenizer = MathTokenizer::new().unwrap();
        let tokens = tokenizer.tokenize_math("α + β = γ").unwrap();

        let greek_tokens: Vec<&MathToken> =
            tokens.iter().filter(|t| t.token_type == MathTokenType::GreekLetter).collect();

        assert_eq!(greek_tokens.len(), 3);
        assert_eq!(greek_tokens[0].text, "α");
        assert_eq!(greek_tokens[1].text, "β");
        assert_eq!(greek_tokens[2].text, "γ");
    }

    #[test]
    fn test_operators_and_symbols() {
        let mut tokenizer = MathTokenizer::new().unwrap();
        let tokens = tokenizer.tokenize_math("x ∈ ℝ, x ≥ 0").unwrap();

        let symbol_tokens: Vec<&MathToken> = tokens
            .iter()
            .filter(|t| {
                matches!(
                    t.token_type,
                    MathTokenType::Symbol | MathTokenType::Operator
                )
            })
            .collect();

        assert!(symbol_tokens.len() >= 2);
    }

    #[test]
    fn test_tokenizer_interface() {
        let tokenizer = MathTokenizer::new().unwrap();

        let encoded = tokenizer.encode("x^2 + y^2 = r^2").unwrap();
        assert!(!encoded.input_ids.is_empty());
        // Test that we can get tokens back from IDs
        let tokens: Vec<String> = encoded
            .input_ids
            .iter()
            .map(|&id| tokenizer.id_to_token(id).unwrap_or_else(|| format!("UNK_{}", id)))
            .collect();
        assert!(!tokens.is_empty());

        // Test that we can get token mappings
        let vocab = tokenizer.get_vocab();
        assert!(!vocab.is_empty());
    }

    #[test]
    fn test_math_analysis() {
        let mut tokenizer = MathTokenizer::new().unwrap();
        let tokens = tokenizer.tokenize_math("sin(x) + cos(y) = 1").unwrap();
        let analysis = tokenizer.analyze_math(&tokens);

        assert!(analysis.total_tokens > 0);
        assert!(analysis.functions >= 2); // sin, cos
        assert!(analysis.operators >= 2); // +, =
        assert!(analysis.variables >= 2); // x, y
        assert!(analysis.numbers >= 1); // 1
    }

    #[test]
    fn test_custom_config() {
        let mut config = MathTokenizerConfig::default();
        config.custom_functions.insert("myFunc".to_string());
        config.preserve_whitespace = true;

        let mut tokenizer = MathTokenizer::with_config(config).unwrap();
        let tokens = tokenizer.tokenize_math("myFunc (x)").unwrap();

        // Should have function, whitespace, delimiter, variable, delimiter
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].token_type, MathTokenType::Function);
        assert_eq!(tokens[1].token_type, MathTokenType::Whitespace);
    }

    #[test]
    fn test_scientific_notation() {
        let mut tokenizer = MathTokenizer::new().unwrap();
        let tokens = tokenizer.tokenize_math("6.022e23 1.602E-19").unwrap();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "6.022e23");
        assert_eq!(tokens[0].token_type, MathTokenType::Number);
        assert_eq!(tokens[1].text, "1.602E-19");
        assert_eq!(tokens[1].token_type, MathTokenType::Number);
    }

    #[test]
    fn test_complexity_analysis() {
        let mut tokenizer = MathTokenizer::new().unwrap();

        // Simple expression
        let simple_tokens = tokenizer.tokenize_math("x + 1").unwrap();
        let simple_analysis = tokenizer.analyze_math(&simple_tokens);

        // Complex expression
        let complex_tokens = tokenizer.tokenize_math("∫₀^∞ e^(-x²) dx = √π/2").unwrap();
        let complex_analysis = tokenizer.analyze_math(&complex_tokens);

        assert!(complex_analysis.complexity_score() > simple_analysis.complexity_score());
    }

    #[test]
    fn test_vocab_stats() {
        let mut tokenizer = MathTokenizer::new().unwrap();
        tokenizer.tokenize_math("sin(x) + cos(y)").unwrap();

        let stats = tokenizer.vocab_stats();
        assert!(stats.contains_key("total_tokens"));
        assert!(stats["total_tokens"] > 0);
    }
}
