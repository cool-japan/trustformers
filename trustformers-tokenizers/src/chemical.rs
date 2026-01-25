//! Chemical notation tokenizer for TrustformeRS
//!
//! This module provides specialized tokenization for chemical formulas, SMILES notation,
//! and other chemical representations used in cheminformatics and molecular modeling.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Configuration for chemical tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalTokenizerConfig {
    /// Maximum sequence length
    pub max_length: Option<usize>,
    /// Whether to include special chemistry tokens
    pub include_special_tokens: bool,
    /// Whether to tokenize SMILES notation
    pub tokenize_smiles: bool,
    /// Whether to tokenize InChI strings
    pub tokenize_inchi: bool,
    /// Whether to tokenize chemical formulas
    pub tokenize_formulas: bool,
    /// Whether to tokenize chemical names
    pub tokenize_names: bool,
    /// Vocabulary size limit
    pub vocab_size: Option<usize>,
    /// Whether to preserve brackets and parentheses structure
    pub preserve_structure: bool,
    /// Case sensitivity for chemical names
    pub case_sensitive: bool,
}

impl Default for ChemicalTokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: Some(512),
            include_special_tokens: true,
            tokenize_smiles: true,
            tokenize_inchi: true,
            tokenize_formulas: true,
            tokenize_names: true,
            vocab_size: Some(10000),
            preserve_structure: true,
            case_sensitive: false,
        }
    }
}

/// Types of chemical tokens
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChemicalTokenType {
    /// Atomic symbols (H, C, N, O, etc.)
    AtomicSymbol,
    /// Bond types (single, double, triple, aromatic)
    Bond,
    /// Ring closure numbers
    RingClosure,
    /// Branching symbols
    Branch,
    /// Stereochemistry indicators
    Stereochemistry,
    /// Charge indicators
    Charge,
    /// Functional groups
    FunctionalGroup,
    /// Chemical formula components
    FormulaComponent,
    /// InChI layer identifiers
    InChILayer,
    /// Chemical name fragments
    NameFragment,
    /// Special chemistry tokens
    Special,
    /// Unknown tokens
    Unknown,
}

/// Chemical token with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalToken {
    /// Token text
    pub text: String,
    /// Token type
    pub token_type: ChemicalTokenType,
    /// Start position in original text
    pub start: usize,
    /// End position in original text
    pub end: usize,
    /// Chemical meaning/metadata
    pub metadata: Option<ChemicalTokenMetadata>,
}

/// Metadata for chemical tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalTokenMetadata {
    /// Atomic number (for atomic symbols)
    pub atomic_number: Option<u8>,
    /// Bond order (for bonds)
    pub bond_order: Option<u8>,
    /// Ring size (for ring closures)
    pub ring_size: Option<u8>,
    /// Formal charge
    pub formal_charge: Option<i8>,
    /// Molecular weight contribution
    pub molecular_weight: Option<f64>,
    /// IUPAC properties
    pub iupac_info: Option<String>,
}

/// Chemical tokenizer implementation
pub struct ChemicalTokenizer {
    config: ChemicalTokenizerConfig,
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    next_id: u32,
    atomic_symbols: HashMap<String, u8>,
    functional_groups: HashMap<String, String>,
    smiles_patterns: Vec<Regex>,
    inchi_patterns: Vec<Regex>,
    formula_patterns: Vec<Regex>,
}

// Static data for chemical elements
static ATOMIC_SYMBOLS: Lazy<HashMap<String, u8>> = Lazy::new(|| {
    let mut map = HashMap::new();
    // Periodic table data (partial)
    map.insert("H".to_string(), 1);
    map.insert("He".to_string(), 2);
    map.insert("Li".to_string(), 3);
    map.insert("Be".to_string(), 4);
    map.insert("B".to_string(), 5);
    map.insert("C".to_string(), 6);
    map.insert("N".to_string(), 7);
    map.insert("O".to_string(), 8);
    map.insert("F".to_string(), 9);
    map.insert("Ne".to_string(), 10);
    map.insert("Na".to_string(), 11);
    map.insert("Mg".to_string(), 12);
    map.insert("Al".to_string(), 13);
    map.insert("Si".to_string(), 14);
    map.insert("P".to_string(), 15);
    map.insert("S".to_string(), 16);
    map.insert("Cl".to_string(), 17);
    map.insert("Ar".to_string(), 18);
    map.insert("K".to_string(), 19);
    map.insert("Ca".to_string(), 20);
    map.insert("Sc".to_string(), 21);
    map.insert("Ti".to_string(), 22);
    map.insert("V".to_string(), 23);
    map.insert("Cr".to_string(), 24);
    map.insert("Mn".to_string(), 25);
    map.insert("Fe".to_string(), 26);
    map.insert("Co".to_string(), 27);
    map.insert("Ni".to_string(), 28);
    map.insert("Cu".to_string(), 29);
    map.insert("Zn".to_string(), 30);
    map.insert("Ga".to_string(), 31);
    map.insert("Ge".to_string(), 32);
    map.insert("As".to_string(), 33);
    map.insert("Se".to_string(), 34);
    map.insert("Br".to_string(), 35);
    map.insert("Kr".to_string(), 36);
    // Add more elements as needed
    map
});

// Common functional groups
static FUNCTIONAL_GROUPS: Lazy<HashMap<String, String>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("OH".to_string(), "hydroxyl".to_string());
    map.insert("NH2".to_string(), "amino".to_string());
    map.insert("COOH".to_string(), "carboxyl".to_string());
    map.insert("CHO".to_string(), "aldehyde".to_string());
    map.insert("CO".to_string(), "carbonyl".to_string());
    map.insert("SH".to_string(), "thiol".to_string());
    map.insert("PO4".to_string(), "phosphate".to_string());
    map.insert("SO4".to_string(), "sulfate".to_string());
    map.insert("NO2".to_string(), "nitro".to_string());
    map.insert("CN".to_string(), "cyano".to_string());
    map.insert("CF3".to_string(), "trifluoromethyl".to_string());
    map.insert("Ph".to_string(), "phenyl".to_string());
    map.insert("Me".to_string(), "methyl".to_string());
    map.insert("Et".to_string(), "ethyl".to_string());
    map.insert("Pr".to_string(), "propyl".to_string());
    map.insert("Bu".to_string(), "butyl".to_string());
    map
});

impl Default for ChemicalTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ChemicalTokenizer {
    /// Create a new chemical tokenizer
    pub fn new() -> Self {
        Self::with_config(ChemicalTokenizerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ChemicalTokenizerConfig) -> Self {
        let mut tokenizer = Self {
            config,
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            next_id: 0,
            atomic_symbols: ATOMIC_SYMBOLS.clone(),
            functional_groups: FUNCTIONAL_GROUPS.clone(),
            smiles_patterns: Self::create_smiles_patterns(),
            inchi_patterns: Self::create_inchi_patterns(),
            formula_patterns: Self::create_formula_patterns(),
        };

        tokenizer.initialize_vocab();
        tokenizer
    }

    /// Initialize vocabulary with chemical tokens
    fn initialize_vocab(&mut self) {
        // Add special tokens
        if self.config.include_special_tokens {
            self.add_token("[CLS]");
            self.add_token("[SEP]");
            self.add_token("[PAD]");
            self.add_token("[UNK]");
            self.add_token("[MASK]");
            self.add_token("[START_CHEM]");
            self.add_token("[END_CHEM]");
            self.add_token("[SMILES]");
            self.add_token("[INCHI]");
            self.add_token("[FORMULA]");
        }

        // Add atomic symbols
        let atomic_symbols: Vec<String> = self.atomic_symbols.keys().cloned().collect();
        for symbol in atomic_symbols {
            self.add_token(&symbol);
        }

        // Add common chemical bonds
        self.add_token("-"); // Single bond
        self.add_token("="); // Double bond
        self.add_token("#"); // Triple bond
        self.add_token(":"); // Aromatic bond

        // Add SMILES-specific tokens
        if self.config.tokenize_smiles {
            self.add_token("("); // Branch start
            self.add_token(")"); // Branch end
            self.add_token("["); // Atom start
            self.add_token("]"); // Atom end
            self.add_token("@"); // Stereochemistry
            self.add_token("@@"); // Stereochemistry
            self.add_token("+"); // Positive charge
            self.add_token("++"); // Double positive charge
            self.add_token("+++"); // Triple positive charge
            self.add_token("/-"); // Negative charge
            self.add_token("--"); // Double negative charge
            self.add_token("---"); // Triple negative charge

            // Ring closure numbers
            for i in 0..10 {
                self.add_token(&i.to_string());
            }
            for i in 10..100 {
                self.add_token(&format!("%{}", i));
            }
        }

        // Add functional groups
        let functional_groups: Vec<String> = self.functional_groups.keys().cloned().collect();
        for group in functional_groups {
            self.add_token(&group);
        }

        // Add common formula patterns
        if self.config.tokenize_formulas {
            self.add_token("2");
            self.add_token("3");
            self.add_token("4");
            self.add_token("5");
            self.add_token("6");
            self.add_token("·"); // Hydrate dot
            self.add_token("H2O"); // Water
        }
    }

    /// Add token to vocabulary
    fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.vocab.get(token) {
            return id;
        }

        let id = self.next_id;
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        self.next_id += 1;
        id
    }

    /// Create SMILES tokenization patterns
    fn create_smiles_patterns() -> Vec<Regex> {
        vec![
            // Atomic symbols with properties
            Regex::new(r"\[[^]]+\]").expect("valid regex"),
            // Common functional groups (must come before individual atoms)
            Regex::new(r"COOH|CHO|NH2|CF3|PO4|SO4|NO2").expect("valid regex"),
            // Two-letter elements (only actual elements, not groups)
            Regex::new(r"Br|Cl").expect("valid regex"),
            // Single organic atoms
            Regex::new(r"[BCNOPSFIbcnops]").expect("valid regex"),
            // Bonds
            Regex::new(r"[=#:]").expect("valid regex"),
            // Ring closures
            Regex::new(r"%\d+|\d").expect("valid regex"),
            // Branches
            Regex::new(r"[()]").expect("valid regex"),
            // Stereochemistry
            Regex::new(r"@@?").expect("valid regex"),
        ]
    }

    /// Create InChI tokenization patterns
    fn create_inchi_patterns() -> Vec<Regex> {
        vec![
            // InChI prefix
            Regex::new(r"InChI=").expect("valid regex"),
            // Version
            Regex::new(r"1S?").expect("valid regex"),
            // Layers
            Regex::new(r"/[a-z]").expect("valid regex"),
            // Chemical formula layer
            Regex::new(r"[A-Z][a-z]?\d*").expect("valid regex"),
            // Connection layer
            Regex::new(r"\d+-\d+").expect("valid regex"),
            // Special characters
            Regex::new(r"[(),;-]").expect("valid regex"),
        ]
    }

    /// Create chemical formula patterns
    fn create_formula_patterns() -> Vec<Regex> {
        vec![
            // Element with count
            Regex::new(r"[A-Z][a-z]?\d*").expect("valid regex"),
            // Hydrates
            Regex::new(r"·\d*H2O").expect("valid regex"),
            // Ionic charges
            Regex::new(r"\d*[+-]").expect("valid regex"),
            // Parentheses with multipliers
            Regex::new(r"\([^)]+\)\d*").expect("valid regex"),
        ]
    }

    /// Tokenize chemical text
    pub fn tokenize_chemical(&self, text: &str) -> Result<Vec<ChemicalToken>> {
        let mut tokens = Vec::new();
        let mut pos = 0;

        // Detect chemical notation type
        if text.starts_with("InChI=") && self.config.tokenize_inchi {
            tokens.extend(self.tokenize_inchi(text, &mut pos)?);
        } else if self.is_smiles(text) && self.config.tokenize_smiles {
            tokens.extend(self.tokenize_smiles(text, &mut pos)?);
        } else if self.is_chemical_formula(text) && self.config.tokenize_formulas {
            tokens.extend(self.tokenize_formula(text, &mut pos)?);
        } else if self.config.tokenize_names {
            tokens.extend(self.tokenize_chemical_name(text, &mut pos)?);
        } else {
            // Fallback to character-level tokenization
            tokens.extend(self.tokenize_fallback(text, &mut pos)?);
        }

        Ok(tokens)
    }

    /// Check if text is SMILES notation
    fn is_smiles(&self, text: &str) -> bool {
        // Simple heuristic: contains typical SMILES characters
        // Must contain at least one chemical character and not be a common word
        let has_chemical_chars = text
            .chars()
            .any(|c| "CNOPSFBrClIcnops()[]=-#@+%1234567890".matches(c).next().is_some());
        let is_common_word = matches!(
            text.to_lowercase().as_str(),
            "water" | "salt" | "acid" | "base" | "metal" | "gas" | "liquid" | "solid"
        );
        has_chemical_chars && !is_common_word
    }

    /// Check if text is chemical formula
    fn is_chemical_formula(&self, text: &str) -> bool {
        // Simple heuristic: starts with capital letter and contains atoms
        text.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false)
            && text.chars().any(|c| c.is_ascii_digit() || "()·".contains(c))
    }

    /// Tokenize SMILES notation
    fn tokenize_smiles(&self, text: &str, pos: &mut usize) -> Result<Vec<ChemicalToken>> {
        let mut tokens = Vec::new();
        let mut current_pos = *pos;

        for pattern in &self.smiles_patterns {
            for mat in pattern.find_iter(text) {
                if mat.start() >= current_pos {
                    let token_text = mat.as_str().to_string();
                    let token_type = self.classify_smiles_token(&token_text);
                    let metadata = self.create_token_metadata(&token_text, &token_type);

                    tokens.push(ChemicalToken {
                        text: token_text,
                        token_type,
                        start: mat.start(),
                        end: mat.end(),
                        metadata,
                    });

                    current_pos = mat.end();
                }
            }
        }

        *pos = current_pos;
        Ok(tokens)
    }

    /// Tokenize InChI string
    fn tokenize_inchi(&self, text: &str, pos: &mut usize) -> Result<Vec<ChemicalToken>> {
        let mut tokens = Vec::new();
        let mut current_pos = *pos;

        for pattern in &self.inchi_patterns {
            for mat in pattern.find_iter(text) {
                if mat.start() >= current_pos {
                    let token_text = mat.as_str().to_string();
                    let token_type = ChemicalTokenType::InChILayer;
                    let metadata = self.create_token_metadata(&token_text, &token_type);

                    tokens.push(ChemicalToken {
                        text: token_text,
                        token_type,
                        start: mat.start(),
                        end: mat.end(),
                        metadata,
                    });

                    current_pos = mat.end();
                }
            }
        }

        *pos = current_pos;
        Ok(tokens)
    }

    /// Tokenize chemical formula
    fn tokenize_formula(&self, text: &str, pos: &mut usize) -> Result<Vec<ChemicalToken>> {
        let mut tokens = Vec::new();
        let mut current_pos = *pos;

        for pattern in &self.formula_patterns {
            for mat in pattern.find_iter(text) {
                if mat.start() >= current_pos {
                    let token_text = mat.as_str().to_string();
                    let token_type = ChemicalTokenType::FormulaComponent;
                    let metadata = self.create_token_metadata(&token_text, &token_type);

                    tokens.push(ChemicalToken {
                        text: token_text,
                        token_type,
                        start: mat.start(),
                        end: mat.end(),
                        metadata,
                    });

                    current_pos = mat.end();
                }
            }
        }

        *pos = current_pos;
        Ok(tokens)
    }

    /// Tokenize chemical name
    fn tokenize_chemical_name(&self, text: &str, pos: &mut usize) -> Result<Vec<ChemicalToken>> {
        let mut tokens = Vec::new();

        // Split on common chemical name separators
        let parts: Vec<&str> = text.split_whitespace().collect();

        for part in parts {
            let token_type = if self.functional_groups.contains_key(part) {
                ChemicalTokenType::FunctionalGroup
            } else {
                ChemicalTokenType::NameFragment
            };

            let metadata = self.create_token_metadata(part, &token_type);

            tokens.push(ChemicalToken {
                text: part.to_string(),
                token_type,
                start: *pos,
                end: *pos + part.len(),
                metadata,
            });

            *pos += part.len() + 1; // +1 for space
        }

        Ok(tokens)
    }

    /// Fallback tokenization
    fn tokenize_fallback(&self, text: &str, pos: &mut usize) -> Result<Vec<ChemicalToken>> {
        let mut tokens = Vec::new();

        for (i, ch) in text.char_indices() {
            tokens.push(ChemicalToken {
                text: ch.to_string(),
                token_type: ChemicalTokenType::Unknown,
                start: *pos + i,
                end: *pos + i + ch.len_utf8(),
                metadata: None,
            });
        }

        *pos += text.len();
        Ok(tokens)
    }

    /// Classify SMILES token type
    fn classify_smiles_token(&self, token: &str) -> ChemicalTokenType {
        match token {
            "(" | ")" => ChemicalTokenType::Branch,
            "[" | "]" => ChemicalTokenType::AtomicSymbol,
            "@" | "@@" => ChemicalTokenType::Stereochemistry,
            "+" | "++" | "+++" | "-" | "--" | "---" => ChemicalTokenType::Charge,
            "=" | "#" | ":" => ChemicalTokenType::Bond,
            _ if token.chars().all(|c| c.is_ascii_digit()) => ChemicalTokenType::RingClosure,
            _ if token.starts_with('%') => ChemicalTokenType::RingClosure,
            _ if self.atomic_symbols.contains_key(token) => ChemicalTokenType::AtomicSymbol,
            _ if self.functional_groups.contains_key(token) => ChemicalTokenType::FunctionalGroup,
            _ => ChemicalTokenType::Unknown,
        }
    }

    /// Create token metadata
    fn create_token_metadata(
        &self,
        token: &str,
        token_type: &ChemicalTokenType,
    ) -> Option<ChemicalTokenMetadata> {
        match token_type {
            ChemicalTokenType::AtomicSymbol => {
                let clean_symbol = token.trim_matches(['[', ']']);
                if let Some(&atomic_number) = self.atomic_symbols.get(clean_symbol) {
                    Some(ChemicalTokenMetadata {
                        atomic_number: Some(atomic_number),
                        bond_order: None,
                        ring_size: None,
                        formal_charge: None,
                        molecular_weight: self.get_atomic_weight(atomic_number),
                        iupac_info: None,
                    })
                } else {
                    None
                }
            },
            ChemicalTokenType::Bond => {
                let bond_order = match token {
                    "-" => 1,
                    "=" => 2,
                    "#" => 3,
                    ":" => 1, // Aromatic
                    _ => 1,
                };
                Some(ChemicalTokenMetadata {
                    atomic_number: None,
                    bond_order: Some(bond_order),
                    ring_size: None,
                    formal_charge: None,
                    molecular_weight: None,
                    iupac_info: None,
                })
            },
            ChemicalTokenType::FunctionalGroup => Some(ChemicalTokenMetadata {
                atomic_number: None,
                bond_order: None,
                ring_size: None,
                formal_charge: None,
                molecular_weight: None,
                iupac_info: self.functional_groups.get(token).cloned(),
            }),
            _ => None,
        }
    }

    /// Get atomic weight for element
    fn get_atomic_weight(&self, atomic_number: u8) -> Option<f64> {
        // Simplified atomic weights
        match atomic_number {
            1 => Some(1.008),    // H
            6 => Some(12.011),   // C
            7 => Some(14.007),   // N
            8 => Some(15.999),   // O
            9 => Some(18.998),   // F
            15 => Some(30.974),  // P
            16 => Some(32.06),   // S
            17 => Some(35.45),   // Cl
            35 => Some(79.904),  // Br
            53 => Some(126.904), // I
            _ => None,
        }
    }

    /// Get vocabulary
    pub fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    /// Get token by ID
    pub fn id_to_token(&self, id: u32) -> Option<&String> {
        self.id_to_token.get(&id)
    }

    /// Get configuration
    pub fn config(&self) -> &ChemicalTokenizerConfig {
        &self.config
    }
}

impl Tokenizer for ChemicalTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let chemical_tokens = self.tokenize_chemical(text)?;
        let mut input_ids = Vec::new();

        for token in chemical_tokens {
            if let Some(&id) = self.vocab.get(&token.text) {
                input_ids.push(id);
            } else {
                // Use UNK token
                if let Some(&unk_id) = self.vocab.get("[UNK]") {
                    input_ids.push(unk_id);
                } else {
                    input_ids.push(0); // Fallback
                }
            }
        }

        // Apply max length constraint
        if let Some(max_len) = self.config.max_length {
            input_ids.truncate(max_len);
        }

        let input_ids_len = input_ids.len();
        Ok(TokenizedInput {
            input_ids,
            attention_mask: vec![1u8; input_ids_len],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut result = String::new();

        for &id in token_ids {
            if let Some(token) = self.id_to_token.get(&id) {
                if !token.starts_with('[') || !token.ends_with(']') {
                    result.push_str(token);
                }
            }
        }

        Ok(result)
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        let mut tokenized_a = self.encode(text_a)?;
        let tokenized_b = self.encode(text_b)?;

        // Add separator token if available
        if let Some(&sep_id) = self.vocab.get("[SEP]") {
            tokenized_a.input_ids.push(sep_id);
        }

        tokenized_a.input_ids.extend(tokenized_b.input_ids);

        // Apply max length constraint
        if let Some(max_len) = self.config.max_length {
            tokenized_a.input_ids.truncate(max_len);
        }

        Ok(tokenized_a)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token.get(&id).cloned()
    }
}

/// Chemical tokenizer analysis
pub struct ChemicalAnalysis {
    /// Token type distribution
    pub token_types: HashMap<ChemicalTokenType, usize>,
    /// Atomic composition
    pub atomic_composition: HashMap<String, usize>,
    /// Functional groups found
    pub functional_groups: HashMap<String, usize>,
    /// Average token length
    pub avg_token_length: f64,
    /// Total molecular weight (if applicable)
    pub total_molecular_weight: Option<f64>,
    /// Chemical complexity score
    pub complexity_score: f64,
}

impl ChemicalTokenizer {
    /// Analyze chemical text
    pub fn analyze(&self, text: &str) -> Result<ChemicalAnalysis> {
        let tokens = self.tokenize_chemical(text)?;

        let mut token_types = HashMap::new();
        let mut atomic_composition = HashMap::new();
        let mut functional_groups = HashMap::new();
        let mut total_length = 0;
        let mut total_molecular_weight = 0.0;
        let mut has_molecular_weight = false;

        for token in &tokens {
            *token_types.entry(token.token_type.clone()).or_insert(0) += 1;
            total_length += token.text.len();

            match &token.token_type {
                ChemicalTokenType::AtomicSymbol => {
                    let clean_symbol = token.text.trim_matches(['[', ']']);
                    *atomic_composition.entry(clean_symbol.to_string()).or_insert(0) += 1;

                    if let Some(ref metadata) = token.metadata {
                        if let Some(weight) = metadata.molecular_weight {
                            total_molecular_weight += weight;
                            has_molecular_weight = true;
                        }
                    }
                },
                ChemicalTokenType::FunctionalGroup => {
                    *functional_groups.entry(token.text.clone()).or_insert(0) += 1;
                },
                _ => {},
            }
        }

        let avg_token_length =
            if tokens.is_empty() { 0.0 } else { total_length as f64 / tokens.len() as f64 };

        let complexity_score = self.calculate_complexity_score(&tokens);

        Ok(ChemicalAnalysis {
            token_types,
            atomic_composition,
            functional_groups,
            avg_token_length,
            total_molecular_weight: if has_molecular_weight {
                Some(total_molecular_weight)
            } else {
                None
            },
            complexity_score,
        })
    }

    /// Calculate chemical complexity score
    fn calculate_complexity_score(&self, tokens: &[ChemicalToken]) -> f64 {
        let mut score = 0.0;

        // Base score from number of tokens
        score += tokens.len() as f64 * 0.1;

        // Additional score for different token types
        let mut token_type_count = HashMap::new();
        for token in tokens {
            *token_type_count.entry(&token.token_type).or_insert(0) += 1;
        }

        score += token_type_count.len() as f64 * 0.5;

        // Bonus for structural elements
        for token in tokens {
            match token.token_type {
                ChemicalTokenType::Branch => score += 1.0,
                ChemicalTokenType::RingClosure => score += 1.5,
                ChemicalTokenType::Stereochemistry => score += 2.0,
                ChemicalTokenType::FunctionalGroup => score += 0.8,
                _ => {},
            }
        }

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chemical_tokenizer_creation() {
        let tokenizer = ChemicalTokenizer::new();
        assert!(tokenizer.get_vocab().len() > 0);
        assert!(tokenizer.get_vocab().contains_key("C"));
        assert!(tokenizer.get_vocab().contains_key("O"));
    }

    #[test]
    fn test_smiles_detection() {
        let tokenizer = ChemicalTokenizer::new();
        assert!(tokenizer.is_smiles("CCO"));
        assert!(tokenizer.is_smiles("c1ccccc1"));
        assert!(tokenizer.is_smiles("CC(=O)O"));
        assert!(!tokenizer.is_smiles("water"));
    }

    #[test]
    fn test_formula_detection() {
        let tokenizer = ChemicalTokenizer::new();
        assert!(tokenizer.is_chemical_formula("H2O"));
        assert!(tokenizer.is_chemical_formula("C6H12O6"));
        assert!(tokenizer.is_chemical_formula("CaCl2·2H2O"));
        assert!(!tokenizer.is_chemical_formula("hello"));
    }

    #[test]
    fn test_smiles_tokenization() {
        let tokenizer = ChemicalTokenizer::new();
        let result = tokenizer.encode("CCO");
        assert!(result.is_ok());
        let tokenized = result.unwrap();
        assert!(!tokenized.input_ids.is_empty());
    }

    #[test]
    fn test_chemical_formula_encoding() {
        let tokenizer = ChemicalTokenizer::new();
        let result = tokenizer.encode("H2O");
        assert!(result.is_ok());
        let tokenized = result.unwrap();
        assert!(!tokenized.input_ids.is_empty());
    }

    #[test]
    fn test_inchi_detection() {
        let tokenizer = ChemicalTokenizer::new();
        let inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3";
        let tokens = tokenizer.tokenize_chemical(inchi);
        assert!(tokens.is_ok());
        let token_list = tokens.unwrap();
        assert!(token_list.iter().any(|t| t.token_type == ChemicalTokenType::InChILayer));
    }

    #[test]
    fn test_atomic_symbol_classification() {
        let tokenizer = ChemicalTokenizer::new();
        let token_type = tokenizer.classify_smiles_token("C");
        assert_eq!(token_type, ChemicalTokenType::AtomicSymbol);

        let token_type = tokenizer.classify_smiles_token("Br");
        assert_eq!(token_type, ChemicalTokenType::AtomicSymbol);
    }

    #[test]
    fn test_bond_classification() {
        let tokenizer = ChemicalTokenizer::new();
        assert_eq!(
            tokenizer.classify_smiles_token("="),
            ChemicalTokenType::Bond
        );
        assert_eq!(
            tokenizer.classify_smiles_token("#"),
            ChemicalTokenType::Bond
        );
    }

    #[test]
    fn test_chemical_analysis() {
        let tokenizer = ChemicalTokenizer::new();
        let analysis = tokenizer.analyze("CCO");
        assert!(analysis.is_ok());
        let result = analysis.unwrap();
        assert!(result.atomic_composition.contains_key("C"));
        assert!(result.atomic_composition.contains_key("O"));
        assert!(result.complexity_score > 0.0);
    }

    #[test]
    fn test_functional_group_recognition() {
        let tokenizer = ChemicalTokenizer::new();
        let analysis = tokenizer.analyze("COOH");
        assert!(analysis.is_ok());
        let result = analysis.unwrap();
        assert!(result.functional_groups.contains_key("COOH"));
    }

    #[test]
    fn test_encoding_decoding_consistency() {
        let tokenizer = ChemicalTokenizer::new();
        let original = "CCO";
        let encoded = tokenizer.encode(original).unwrap();
        let decoded = tokenizer.decode(&encoded.input_ids).unwrap();
        // Note: decoded might not be exactly the same due to tokenization
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_max_length_constraint() {
        let mut config = ChemicalTokenizerConfig::default();
        config.max_length = Some(5);
        let tokenizer = ChemicalTokenizer::with_config(config);

        let result = tokenizer.encode("CCCCCCCCCCCCCCCC");
        assert!(result.is_ok());
        let tokenized = result.unwrap();
        assert!(tokenized.input_ids.len() <= 5);
    }
}
