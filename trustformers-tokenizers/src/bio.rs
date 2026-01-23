//! Biological sequence tokenizer for TrustformeRS
//!
//! This module provides specialized tokenization for biological sequences
//! including DNA, RNA, and protein sequences used in bioinformatics.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Configuration for biological sequence tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioTokenizerConfig {
    /// Maximum sequence length
    pub max_length: Option<usize>,
    /// Whether to include special bio tokens
    pub include_special_tokens: bool,
    /// Whether to tokenize DNA sequences
    pub tokenize_dna: bool,
    /// Whether to tokenize RNA sequences
    pub tokenize_rna: bool,
    /// Whether to tokenize protein sequences
    pub tokenize_proteins: bool,
    /// K-mer size for subsequence tokenization
    pub kmer_size: Option<usize>,
    /// Whether to use overlapping k-mers
    pub overlapping_kmers: bool,
    /// Whether to preserve case (for mixed case sequences)
    pub preserve_case: bool,
    /// Whether to handle ambiguous nucleotides/amino acids
    pub handle_ambiguous: bool,
    /// Whether to tokenize secondary structure annotations
    pub tokenize_structure: bool,
    /// Vocabulary size limit
    pub vocab_size: Option<usize>,
}

impl Default for BioTokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: Some(2048),
            include_special_tokens: true,
            tokenize_dna: true,
            tokenize_rna: true,
            tokenize_proteins: true,
            kmer_size: Some(3), // Codons for DNA/RNA
            overlapping_kmers: true,
            preserve_case: false,
            handle_ambiguous: true,
            tokenize_structure: false,
            vocab_size: Some(5000),
        }
    }
}

/// Types of biological tokens
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BioTokenType {
    /// DNA nucleotides (A, T, G, C)
    DNANucleotide,
    /// RNA nucleotides (A, U, G, C)
    RNANucleotide,
    /// Amino acids (20 standard + special)
    AminoAcid,
    /// K-mer sequences
    Kmer,
    /// Ambiguous nucleotides (N, R, Y, etc.)
    AmbiguousNucleotide,
    /// Ambiguous amino acids (X, B, Z)
    AmbiguousAminoAcid,
    /// Stop codons
    StopCodon,
    /// Start codons
    StartCodon,
    /// Secondary structure elements
    SecondaryStructure,
    /// Sequence modifications
    Modification,
    /// Special sequence markers
    Special,
    /// Unknown sequences
    Unknown,
}

/// Biological token with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioToken {
    /// Token text
    pub text: String,
    /// Token type
    pub token_type: BioTokenType,
    /// Start position in original sequence
    pub start: usize,
    /// End position in original sequence
    pub end: usize,
    /// Biological metadata
    pub metadata: Option<BioTokenMetadata>,
}

/// Metadata for biological tokens
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BioTokenMetadata {
    /// Molecular weight (for amino acids)
    pub molecular_weight: Option<f64>,
    /// Hydrophobicity (for amino acids)
    pub hydrophobicity: Option<f64>,
    /// Charge (for amino acids)
    pub charge: Option<i8>,
    /// GC content (for nucleotide sequences)
    pub gc_content: Option<f64>,
    /// Melting temperature (for DNA/RNA)
    pub melting_temp: Option<f64>,
    /// Codon table position
    pub codon_position: Option<u8>,
    /// Reading frame
    pub reading_frame: Option<u8>,
    /// Secondary structure type
    pub structure_type: Option<String>,
}

/// Biological sequence tokenizer
pub struct BioTokenizer {
    config: BioTokenizerConfig,
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    next_id: u32,
    amino_acids: HashMap<char, AminoAcidInfo>,
    nucleotides: HashMap<char, NucleotideInfo>,
    genetic_code: HashMap<String, char>,
    #[allow(dead_code)]
    structure_patterns: Vec<Regex>,
}

/// Amino acid information
#[derive(Debug, Clone)]
struct AminoAcidInfo {
    #[allow(dead_code)]
    name: String,
    molecular_weight: f64,
    hydrophobicity: f64,
    charge: i8,
    #[allow(dead_code)]
    single_letter: char,
    #[allow(dead_code)]
    three_letter: String,
}

/// Nucleotide information
#[derive(Debug, Clone)]
struct NucleotideInfo {
    #[allow(dead_code)]
    name: String,
    complement: char,
    #[allow(dead_code)]
    is_purine: bool,
    molecular_weight: f64,
}

// Static data for amino acids
static AMINO_ACIDS: Lazy<HashMap<char, AminoAcidInfo>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Standard amino acids with properties
    map.insert(
        'A',
        AminoAcidInfo {
            name: "Alanine".to_string(),
            molecular_weight: 89.1,
            hydrophobicity: 1.8,
            charge: 0,
            single_letter: 'A',
            three_letter: "Ala".to_string(),
        },
    );
    map.insert(
        'R',
        AminoAcidInfo {
            name: "Arginine".to_string(),
            molecular_weight: 174.2,
            hydrophobicity: -4.5,
            charge: 1,
            single_letter: 'R',
            three_letter: "Arg".to_string(),
        },
    );
    map.insert(
        'N',
        AminoAcidInfo {
            name: "Asparagine".to_string(),
            molecular_weight: 132.1,
            hydrophobicity: -3.5,
            charge: 0,
            single_letter: 'N',
            three_letter: "Asn".to_string(),
        },
    );
    map.insert(
        'D',
        AminoAcidInfo {
            name: "Aspartic acid".to_string(),
            molecular_weight: 133.1,
            hydrophobicity: -3.5,
            charge: -1,
            single_letter: 'D',
            three_letter: "Asp".to_string(),
        },
    );
    map.insert(
        'C',
        AminoAcidInfo {
            name: "Cysteine".to_string(),
            molecular_weight: 121.0,
            hydrophobicity: 2.5,
            charge: 0,
            single_letter: 'C',
            three_letter: "Cys".to_string(),
        },
    );
    map.insert(
        'E',
        AminoAcidInfo {
            name: "Glutamic acid".to_string(),
            molecular_weight: 147.1,
            hydrophobicity: -3.5,
            charge: -1,
            single_letter: 'E',
            three_letter: "Glu".to_string(),
        },
    );
    map.insert(
        'Q',
        AminoAcidInfo {
            name: "Glutamine".to_string(),
            molecular_weight: 146.1,
            hydrophobicity: -3.5,
            charge: 0,
            single_letter: 'Q',
            three_letter: "Gln".to_string(),
        },
    );
    map.insert(
        'G',
        AminoAcidInfo {
            name: "Glycine".to_string(),
            molecular_weight: 75.1,
            hydrophobicity: -0.4,
            charge: 0,
            single_letter: 'G',
            three_letter: "Gly".to_string(),
        },
    );
    map.insert(
        'H',
        AminoAcidInfo {
            name: "Histidine".to_string(),
            molecular_weight: 155.2,
            hydrophobicity: -3.2,
            charge: 0,
            single_letter: 'H',
            three_letter: "His".to_string(),
        },
    );
    map.insert(
        'I',
        AminoAcidInfo {
            name: "Isoleucine".to_string(),
            molecular_weight: 131.2,
            hydrophobicity: 4.5,
            charge: 0,
            single_letter: 'I',
            three_letter: "Ile".to_string(),
        },
    );
    map.insert(
        'L',
        AminoAcidInfo {
            name: "Leucine".to_string(),
            molecular_weight: 131.2,
            hydrophobicity: 3.8,
            charge: 0,
            single_letter: 'L',
            three_letter: "Leu".to_string(),
        },
    );
    map.insert(
        'K',
        AminoAcidInfo {
            name: "Lysine".to_string(),
            molecular_weight: 146.2,
            hydrophobicity: -3.9,
            charge: 1,
            single_letter: 'K',
            three_letter: "Lys".to_string(),
        },
    );
    map.insert(
        'M',
        AminoAcidInfo {
            name: "Methionine".to_string(),
            molecular_weight: 149.2,
            hydrophobicity: 1.9,
            charge: 0,
            single_letter: 'M',
            three_letter: "Met".to_string(),
        },
    );
    map.insert(
        'F',
        AminoAcidInfo {
            name: "Phenylalanine".to_string(),
            molecular_weight: 165.2,
            hydrophobicity: 2.8,
            charge: 0,
            single_letter: 'F',
            three_letter: "Phe".to_string(),
        },
    );
    map.insert(
        'P',
        AminoAcidInfo {
            name: "Proline".to_string(),
            molecular_weight: 115.1,
            hydrophobicity: -1.6,
            charge: 0,
            single_letter: 'P',
            three_letter: "Pro".to_string(),
        },
    );
    map.insert(
        'S',
        AminoAcidInfo {
            name: "Serine".to_string(),
            molecular_weight: 105.1,
            hydrophobicity: -0.8,
            charge: 0,
            single_letter: 'S',
            three_letter: "Ser".to_string(),
        },
    );
    map.insert(
        'T',
        AminoAcidInfo {
            name: "Threonine".to_string(),
            molecular_weight: 119.1,
            hydrophobicity: -0.7,
            charge: 0,
            single_letter: 'T',
            three_letter: "Thr".to_string(),
        },
    );
    map.insert(
        'W',
        AminoAcidInfo {
            name: "Tryptophan".to_string(),
            molecular_weight: 204.2,
            hydrophobicity: -0.9,
            charge: 0,
            single_letter: 'W',
            three_letter: "Trp".to_string(),
        },
    );
    map.insert(
        'Y',
        AminoAcidInfo {
            name: "Tyrosine".to_string(),
            molecular_weight: 181.2,
            hydrophobicity: -1.3,
            charge: 0,
            single_letter: 'Y',
            three_letter: "Tyr".to_string(),
        },
    );
    map.insert(
        'V',
        AminoAcidInfo {
            name: "Valine".to_string(),
            molecular_weight: 117.1,
            hydrophobicity: 4.2,
            charge: 0,
            single_letter: 'V',
            three_letter: "Val".to_string(),
        },
    );

    // Ambiguous amino acids
    map.insert(
        'X',
        AminoAcidInfo {
            name: "Unknown".to_string(),
            molecular_weight: 0.0,
            hydrophobicity: 0.0,
            charge: 0,
            single_letter: 'X',
            three_letter: "Xaa".to_string(),
        },
    );

    map
});

// Static data for nucleotides
static NUCLEOTIDES: Lazy<HashMap<char, NucleotideInfo>> = Lazy::new(|| {
    let mut map = HashMap::new();

    map.insert(
        'A',
        NucleotideInfo {
            name: "Adenine".to_string(),
            complement: 'T',
            is_purine: true,
            molecular_weight: 331.2,
        },
    );
    map.insert(
        'T',
        NucleotideInfo {
            name: "Thymine".to_string(),
            complement: 'A',
            is_purine: false,
            molecular_weight: 322.2,
        },
    );
    map.insert(
        'G',
        NucleotideInfo {
            name: "Guanine".to_string(),
            complement: 'C',
            is_purine: true,
            molecular_weight: 347.2,
        },
    );
    map.insert(
        'C',
        NucleotideInfo {
            name: "Cytosine".to_string(),
            complement: 'G',
            is_purine: false,
            molecular_weight: 307.2,
        },
    );
    map.insert(
        'U',
        NucleotideInfo {
            name: "Uracil".to_string(),
            complement: 'A',
            is_purine: false,
            molecular_weight: 308.2,
        },
    );

    // Ambiguous nucleotides
    map.insert(
        'N',
        NucleotideInfo {
            name: "Any nucleotide".to_string(),
            complement: 'N',
            is_purine: false,
            molecular_weight: 0.0,
        },
    );
    map.insert(
        'R',
        NucleotideInfo {
            name: "Purine".to_string(),
            complement: 'Y',
            is_purine: true,
            molecular_weight: 0.0,
        },
    );
    map.insert(
        'Y',
        NucleotideInfo {
            name: "Pyrimidine".to_string(),
            complement: 'R',
            is_purine: false,
            molecular_weight: 0.0,
        },
    );

    map
});

// Standard genetic code
static GENETIC_CODE: Lazy<HashMap<String, char>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Standard genetic code table
    map.insert("TTT".to_string(), 'F');
    map.insert("TTC".to_string(), 'F');
    map.insert("TTA".to_string(), 'L');
    map.insert("TTG".to_string(), 'L');
    map.insert("TCT".to_string(), 'S');
    map.insert("TCC".to_string(), 'S');
    map.insert("TCA".to_string(), 'S');
    map.insert("TCG".to_string(), 'S');
    map.insert("TAT".to_string(), 'Y');
    map.insert("TAC".to_string(), 'Y');
    map.insert("TAA".to_string(), '*');
    map.insert("TAG".to_string(), '*'); // Stop codons
    map.insert("TGT".to_string(), 'C');
    map.insert("TGC".to_string(), 'C');
    map.insert("TGA".to_string(), '*');
    map.insert("TGG".to_string(), 'W'); // Stop codon

    map.insert("CTT".to_string(), 'L');
    map.insert("CTC".to_string(), 'L');
    map.insert("CTA".to_string(), 'L');
    map.insert("CTG".to_string(), 'L');
    map.insert("CCT".to_string(), 'P');
    map.insert("CCC".to_string(), 'P');
    map.insert("CCA".to_string(), 'P');
    map.insert("CCG".to_string(), 'P');
    map.insert("CAT".to_string(), 'H');
    map.insert("CAC".to_string(), 'H');
    map.insert("CAA".to_string(), 'Q');
    map.insert("CAG".to_string(), 'Q');
    map.insert("CGT".to_string(), 'R');
    map.insert("CGC".to_string(), 'R');
    map.insert("CGA".to_string(), 'R');
    map.insert("CGG".to_string(), 'R');

    map.insert("ATT".to_string(), 'I');
    map.insert("ATC".to_string(), 'I');
    map.insert("ATA".to_string(), 'I');
    map.insert("ATG".to_string(), 'M'); // Start codon
    map.insert("ACT".to_string(), 'T');
    map.insert("ACC".to_string(), 'T');
    map.insert("ACA".to_string(), 'T');
    map.insert("ACG".to_string(), 'T');
    map.insert("AAT".to_string(), 'N');
    map.insert("AAC".to_string(), 'N');
    map.insert("AAA".to_string(), 'K');
    map.insert("AAG".to_string(), 'K');
    map.insert("AGT".to_string(), 'S');
    map.insert("AGC".to_string(), 'S');
    map.insert("AGA".to_string(), 'R');
    map.insert("AGG".to_string(), 'R');

    map.insert("GTT".to_string(), 'V');
    map.insert("GTC".to_string(), 'V');
    map.insert("GTA".to_string(), 'V');
    map.insert("GTG".to_string(), 'V');
    map.insert("GCT".to_string(), 'A');
    map.insert("GCC".to_string(), 'A');
    map.insert("GCA".to_string(), 'A');
    map.insert("GCG".to_string(), 'A');
    map.insert("GAT".to_string(), 'D');
    map.insert("GAC".to_string(), 'D');
    map.insert("GAA".to_string(), 'E');
    map.insert("GAG".to_string(), 'E');
    map.insert("GGT".to_string(), 'G');
    map.insert("GGC".to_string(), 'G');
    map.insert("GGA".to_string(), 'G');
    map.insert("GGG".to_string(), 'G');

    map
});

impl Default for BioTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BioTokenizer {
    /// Create a new biological tokenizer
    pub fn new() -> Self {
        Self::with_config(BioTokenizerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: BioTokenizerConfig) -> Self {
        let mut tokenizer = Self {
            config,
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            next_id: 0,
            amino_acids: AMINO_ACIDS.clone(),
            nucleotides: NUCLEOTIDES.clone(),
            genetic_code: GENETIC_CODE.clone(),
            structure_patterns: Self::create_structure_patterns(),
        };

        tokenizer.initialize_vocab();
        tokenizer
    }

    /// Initialize vocabulary with biological tokens
    fn initialize_vocab(&mut self) {
        // Add special tokens
        if self.config.include_special_tokens {
            self.add_token("[CLS]");
            self.add_token("[SEP]");
            self.add_token("[PAD]");
            self.add_token("[UNK]");
            self.add_token("[MASK]");
            self.add_token("[START_SEQ]");
            self.add_token("[END_SEQ]");
            self.add_token("[START_PROTEIN]");
            self.add_token("[END_PROTEIN]");
            self.add_token("[START_DNA]");
            self.add_token("[END_DNA]");
            self.add_token("[START_RNA]");
            self.add_token("[END_RNA]");
        }

        // Add nucleotides
        if self.config.tokenize_dna || self.config.tokenize_rna {
            let nucleotides: Vec<String> = self.nucleotides.keys().map(|c| c.to_string()).collect();
            for nucleotide in nucleotides {
                self.add_token(&nucleotide);
            }
        }

        // Add amino acids
        if self.config.tokenize_proteins {
            let amino_acids: Vec<String> = self.amino_acids.keys().map(|c| c.to_string()).collect();
            for amino_acid in amino_acids {
                self.add_token(&amino_acid);
            }
        }

        // Add k-mers if specified
        if let Some(k) = self.config.kmer_size {
            self.generate_kmers(k);
        }

        // Add stop/start codons
        self.add_token("ATG"); // Start codon
        self.add_token("TAA"); // Stop codon
        self.add_token("TAG"); // Stop codon
        self.add_token("TGA"); // Stop codon

        // Add secondary structure elements if enabled
        if self.config.tokenize_structure {
            self.add_token("H"); // Helix
            self.add_token("E"); // Beta sheet
            self.add_token("C"); // Coil/loop
            self.add_token("T"); // Turn
        }
    }

    /// Generate k-mer vocabulary
    fn generate_kmers(&mut self, k: usize) {
        if self.config.tokenize_dna || self.config.tokenize_rna {
            let nucleotides = if self.config.tokenize_rna { "AUGC" } else { "ATGC" };
            self.generate_kmer_combinations(nucleotides.chars().collect(), k, String::new());
        }

        if self.config.tokenize_proteins {
            let amino_acids: Vec<char> = self.amino_acids.keys().copied().collect();
            if k <= 3 {
                // Only generate small protein k-mers to avoid explosion
                self.generate_kmer_combinations(amino_acids, k, String::new());
            }
        }
    }

    /// Recursively generate k-mer combinations
    fn generate_kmer_combinations(&mut self, alphabet: Vec<char>, k: usize, current: String) {
        if current.len() == k {
            self.add_token(&current);
            return;
        }

        if self.vocab.len() >= self.config.vocab_size.unwrap_or(5000) {
            return; // Stop if vocabulary size limit reached
        }

        for &c in &alphabet {
            let mut next = current.clone();
            next.push(c);
            self.generate_kmer_combinations(alphabet.clone(), k, next);
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

    /// Create secondary structure patterns
    fn create_structure_patterns() -> Vec<Regex> {
        vec![
            Regex::new(r"[HEC]+").unwrap(), // Secondary structure annotations
            Regex::new(r"[αβ]+").unwrap(),  // Greek letter annotations
        ]
    }

    /// Tokenize biological sequence
    pub fn tokenize_bio(&self, sequence: &str) -> Result<Vec<BioToken>> {
        let sequence = if self.config.preserve_case {
            sequence.to_string()
        } else {
            sequence.to_uppercase()
        };

        // Detect sequence type
        let seq_type = self.detect_sequence_type(&sequence);

        match seq_type {
            SequenceType::DNA => self.tokenize_dna(&sequence),
            SequenceType::RNA => self.tokenize_rna(&sequence),
            SequenceType::Protein => self.tokenize_protein(&sequence),
            SequenceType::Structure => self.tokenize_structure(&sequence),
            SequenceType::Unknown => self.tokenize_fallback(&sequence),
        }
    }

    /// Detect sequence type
    fn detect_sequence_type(&self, sequence: &str) -> SequenceType {
        let chars: Vec<char> = sequence.chars().collect();
        let total = chars.len() as f64;

        if total == 0.0 {
            return SequenceType::Unknown;
        }

        // Count different character types
        let dna_chars = chars.iter().filter(|&&c| "ATGC".contains(c)).count() as f64 / total;
        let rna_chars = chars.iter().filter(|&&c| "AUGC".contains(c)).count() as f64 / total;
        let protein_chars =
            chars.iter().filter(|&&c| self.amino_acids.contains_key(&c)).count() as f64 / total;
        let structure_chars = chars.iter().filter(|&&c| "HEC".contains(c)).count() as f64 / total;

        // Determine most likely sequence type
        if dna_chars > 0.8 && !sequence.contains('U') {
            SequenceType::DNA
        } else if rna_chars > 0.8 && sequence.contains('U') {
            SequenceType::RNA
        } else if protein_chars > 0.8 {
            SequenceType::Protein
        } else if structure_chars > 0.5 {
            SequenceType::Structure
        } else {
            SequenceType::Unknown
        }
    }

    /// Tokenize DNA sequence
    fn tokenize_dna(&self, sequence: &str) -> Result<Vec<BioToken>> {
        let mut tokens = Vec::new();

        if let Some(k) = self.config.kmer_size {
            // K-mer tokenization
            tokens.extend(self.tokenize_kmers(sequence, k, BioTokenType::DNANucleotide)?);
        } else {
            // Single nucleotide tokenization
            for (i, c) in sequence.char_indices() {
                let token_type = if self.nucleotides.contains_key(&c) {
                    if "ATGC".contains(c) {
                        BioTokenType::DNANucleotide
                    } else {
                        BioTokenType::AmbiguousNucleotide
                    }
                } else {
                    BioTokenType::Unknown
                };

                let metadata = self.create_nucleotide_metadata(c);

                tokens.push(BioToken {
                    text: c.to_string(),
                    token_type,
                    start: i,
                    end: i + 1,
                    metadata,
                });
            }
        }

        Ok(tokens)
    }

    /// Tokenize RNA sequence
    fn tokenize_rna(&self, sequence: &str) -> Result<Vec<BioToken>> {
        let mut tokens = Vec::new();

        if let Some(k) = self.config.kmer_size {
            tokens.extend(self.tokenize_kmers(sequence, k, BioTokenType::RNANucleotide)?);
        } else {
            for (i, c) in sequence.char_indices() {
                let token_type = if self.nucleotides.contains_key(&c) {
                    if "AUGC".contains(c) {
                        BioTokenType::RNANucleotide
                    } else {
                        BioTokenType::AmbiguousNucleotide
                    }
                } else {
                    BioTokenType::Unknown
                };

                let metadata = self.create_nucleotide_metadata(c);

                tokens.push(BioToken {
                    text: c.to_string(),
                    token_type,
                    start: i,
                    end: i + 1,
                    metadata,
                });
            }
        }

        Ok(tokens)
    }

    /// Tokenize protein sequence
    fn tokenize_protein(&self, sequence: &str) -> Result<Vec<BioToken>> {
        let mut tokens = Vec::new();

        if let Some(k) = self.config.kmer_size {
            tokens.extend(self.tokenize_kmers(sequence, k, BioTokenType::AminoAcid)?);
        } else {
            for (i, c) in sequence.char_indices() {
                let token_type = if self.amino_acids.contains_key(&c) {
                    if "ACDEFGHIKLMNPQRSTVWY".contains(c) {
                        BioTokenType::AminoAcid
                    } else {
                        BioTokenType::AmbiguousAminoAcid
                    }
                } else {
                    BioTokenType::Unknown
                };

                let metadata = self.create_amino_acid_metadata(c);

                tokens.push(BioToken {
                    text: c.to_string(),
                    token_type,
                    start: i,
                    end: i + 1,
                    metadata,
                });
            }
        }

        Ok(tokens)
    }

    /// Tokenize secondary structure
    fn tokenize_structure(&self, sequence: &str) -> Result<Vec<BioToken>> {
        let mut tokens = Vec::new();

        for (i, c) in sequence.char_indices() {
            let token_type = BioTokenType::SecondaryStructure;
            let structure_type = match c {
                'H' => Some("Helix".to_string()),
                'E' => Some("Beta sheet".to_string()),
                'C' => Some("Coil".to_string()),
                'T' => Some("Turn".to_string()),
                _ => Some("Unknown".to_string()),
            };

            let metadata = BioTokenMetadata {
                structure_type,
                ..Default::default()
            };

            tokens.push(BioToken {
                text: c.to_string(),
                token_type,
                start: i,
                end: i + 1,
                metadata: Some(metadata),
            });
        }

        Ok(tokens)
    }

    /// Tokenize using k-mers
    fn tokenize_kmers(
        &self,
        sequence: &str,
        k: usize,
        _base_type: BioTokenType,
    ) -> Result<Vec<BioToken>> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = sequence.chars().collect();

        if chars.len() < k {
            return Ok(tokens);
        }

        let step = if self.config.overlapping_kmers { 1 } else { k };

        for i in (0..=chars.len() - k).step_by(step) {
            let kmer: String = chars[i..i + k].iter().collect();

            let token_type = if kmer.len() == 3 && self.genetic_code.contains_key(&kmer) {
                if self.genetic_code[&kmer] == '*' {
                    BioTokenType::StopCodon
                } else if kmer == "ATG" {
                    BioTokenType::StartCodon
                } else {
                    BioTokenType::Kmer
                }
            } else {
                BioTokenType::Kmer
            };

            let metadata = self.create_kmer_metadata(&kmer, i);

            tokens.push(BioToken {
                text: kmer,
                token_type,
                start: i,
                end: i + k,
                metadata,
            });
        }

        Ok(tokens)
    }

    /// Fallback tokenization
    fn tokenize_fallback(&self, sequence: &str) -> Result<Vec<BioToken>> {
        let mut tokens = Vec::new();

        for (i, c) in sequence.char_indices() {
            tokens.push(BioToken {
                text: c.to_string(),
                token_type: BioTokenType::Unknown,
                start: i,
                end: i + 1,
                metadata: None,
            });
        }

        Ok(tokens)
    }

    /// Create nucleotide metadata
    fn create_nucleotide_metadata(&self, nucleotide: char) -> Option<BioTokenMetadata> {
        self.nucleotides.get(&nucleotide).map(|info| BioTokenMetadata {
            molecular_weight: Some(info.molecular_weight),
            hydrophobicity: None,
            charge: None,
            gc_content: if "GC".contains(nucleotide) { Some(1.0) } else { Some(0.0) },
            melting_temp: None,
            codon_position: None,
            reading_frame: None,
            structure_type: None,
        })
    }

    /// Create amino acid metadata
    fn create_amino_acid_metadata(&self, amino_acid: char) -> Option<BioTokenMetadata> {
        self.amino_acids.get(&amino_acid).map(|info| BioTokenMetadata {
            molecular_weight: Some(info.molecular_weight),
            hydrophobicity: Some(info.hydrophobicity),
            charge: Some(info.charge),
            gc_content: None,
            melting_temp: None,
            codon_position: None,
            reading_frame: None,
            structure_type: None,
        })
    }

    /// Create k-mer metadata
    fn create_kmer_metadata(&self, kmer: &str, position: usize) -> Option<BioTokenMetadata> {
        let mut metadata = BioTokenMetadata::default();

        // Calculate GC content for DNA/RNA k-mers
        if kmer.chars().all(|c| "ATGCU".contains(c)) {
            let gc_count = kmer.chars().filter(|&c| "GC".contains(c)).count();
            metadata.gc_content = Some(gc_count as f64 / kmer.len() as f64);
        }

        // Set reading frame for codons
        if kmer.len() == 3 {
            metadata.reading_frame = Some((position % 3) as u8);
            if let Some(&amino_acid) = self.genetic_code.get(kmer) {
                if let Some(aa_info) = self.amino_acids.get(&amino_acid) {
                    metadata.molecular_weight = Some(aa_info.molecular_weight);
                    metadata.hydrophobicity = Some(aa_info.hydrophobicity);
                    metadata.charge = Some(aa_info.charge);
                }
            }
        }

        Some(metadata)
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
    pub fn config(&self) -> &BioTokenizerConfig {
        &self.config
    }

    /// Translate DNA to protein
    pub fn translate_dna(&self, dna_sequence: &str) -> Result<String> {
        let dna = dna_sequence.to_uppercase();
        let mut protein = String::new();

        for i in (0..dna.len()).step_by(3) {
            if i + 3 <= dna.len() {
                let codon = &dna[i..i + 3];
                if let Some(&amino_acid) = self.genetic_code.get(codon) {
                    if amino_acid == '*' {
                        break; // Stop at stop codon
                    }
                    protein.push(amino_acid);
                } else {
                    protein.push('X'); // Unknown amino acid
                }
            }
        }

        Ok(protein)
    }

    /// Get reverse complement of DNA sequence
    pub fn reverse_complement(&self, dna_sequence: &str) -> String {
        dna_sequence
            .chars()
            .rev()
            .map(|c| {
                if let Some(info) = self.nucleotides.get(&c.to_ascii_uppercase()) {
                    info.complement
                } else {
                    'N'
                }
            })
            .collect()
    }
}

/// Sequence type detection
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
enum SequenceType {
    DNA,
    RNA,
    Protein,
    Structure,
    Unknown,
}

impl Tokenizer for BioTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let bio_tokens = self.tokenize_bio(text)?;
        let mut input_ids = Vec::new();

        for token in bio_tokens {
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

        let input_len = input_ids.len();
        Ok(TokenizedInput {
            input_ids,
            attention_mask: vec![1; input_len],
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

/// Biological sequence analysis
pub struct BioAnalysis {
    /// Token type distribution
    pub token_types: HashMap<BioTokenType, usize>,
    /// Amino acid composition (for proteins)
    pub amino_acid_composition: HashMap<char, usize>,
    /// Nucleotide composition (for DNA/RNA)
    pub nucleotide_composition: HashMap<char, usize>,
    /// GC content (for DNA/RNA)
    pub gc_content: Option<f64>,
    /// Molecular weight (for proteins)
    pub molecular_weight: Option<f64>,
    /// Hydrophobicity (for proteins)
    pub avg_hydrophobicity: Option<f64>,
    /// Charge (for proteins)
    pub net_charge: Option<i32>,
    /// K-mer diversity
    pub kmer_diversity: f64,
    /// Sequence length
    pub sequence_length: usize,
}

impl BioTokenizer {
    /// Analyze biological sequence
    pub fn analyze(&self, sequence: &str) -> Result<BioAnalysis> {
        let tokens = self.tokenize_bio(sequence)?;

        let mut token_types = HashMap::new();
        let mut amino_acid_composition = HashMap::new();
        let mut nucleotide_composition = HashMap::new();
        let mut molecular_weight = 0.0;
        let mut total_hydrophobicity = 0.0;
        let mut net_charge = 0i32;
        let mut gc_count = 0;
        let mut nucleotide_count = 0;
        let mut protein_residue_count = 0;

        for token in &tokens {
            *token_types.entry(token.token_type.clone()).or_insert(0) += 1;

            if token.text.len() == 1 {
                let c = token.text.chars().next().unwrap();

                match token.token_type {
                    BioTokenType::AminoAcid => {
                        *amino_acid_composition.entry(c).or_insert(0) += 1;
                        if let Some(info) = self.amino_acids.get(&c) {
                            molecular_weight += info.molecular_weight;
                            total_hydrophobicity += info.hydrophobicity;
                            net_charge += info.charge as i32;
                            protein_residue_count += 1;
                        }
                    },
                    BioTokenType::DNANucleotide | BioTokenType::RNANucleotide => {
                        *nucleotide_composition.entry(c).or_insert(0) += 1;
                        if "GC".contains(c) {
                            gc_count += 1;
                        }
                        nucleotide_count += 1;
                    },
                    _ => {},
                }
            } else {
                // Handle k-mer tokens for both nucleotide and protein analysis
                if token.token_type == BioTokenType::Kmer {
                    for c in token.text.chars() {
                        // Check if it's a nucleotide sequence
                        if "ATGCU".contains(c) {
                            *nucleotide_composition.entry(c).or_insert(0) += 1;
                            if "GC".contains(c) {
                                gc_count += 1;
                            }
                            nucleotide_count += 1;
                        }
                        // Check if it's a protein sequence
                        else if self.amino_acids.contains_key(&c) {
                            *amino_acid_composition.entry(c).or_insert(0) += 1;
                            if let Some(info) = self.amino_acids.get(&c) {
                                molecular_weight += info.molecular_weight;
                                total_hydrophobicity += info.hydrophobicity;
                                net_charge += info.charge as i32;
                                protein_residue_count += 1;
                            }
                        }
                    }
                }
            }
        }

        let gc_content = if nucleotide_count > 0 {
            Some(gc_count as f64 / nucleotide_count as f64)
        } else {
            None
        };

        let avg_hydrophobicity = if protein_residue_count > 0 {
            Some(total_hydrophobicity / protein_residue_count as f64)
        } else {
            None
        };

        let molecular_weight_final =
            if protein_residue_count > 0 { Some(molecular_weight) } else { None };

        let net_charge_final = if protein_residue_count > 0 { Some(net_charge) } else { None };

        // Calculate k-mer diversity (Simpson's diversity index)
        let total_tokens = tokens.len();
        let kmer_diversity = if total_tokens > 0 {
            let mut diversity = 0.0;
            for count in token_types.values() {
                let frequency = *count as f64 / total_tokens as f64;
                diversity += frequency * frequency;
            }
            1.0 - diversity
        } else {
            0.0
        };

        Ok(BioAnalysis {
            token_types,
            amino_acid_composition,
            nucleotide_composition,
            gc_content,
            molecular_weight: molecular_weight_final,
            avg_hydrophobicity,
            net_charge: net_charge_final,
            kmer_diversity,
            sequence_length: sequence.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bio_tokenizer_creation() {
        let tokenizer = BioTokenizer::new();
        assert!(tokenizer.get_vocab().len() > 0);
        assert!(tokenizer.get_vocab().contains_key("A"));
        assert!(tokenizer.get_vocab().contains_key("T"));
    }

    #[test]
    fn test_sequence_type_detection() {
        let tokenizer = BioTokenizer::new();
        assert_eq!(
            tokenizer.detect_sequence_type("ATGCGATCG"),
            SequenceType::DNA
        );
        assert_eq!(
            tokenizer.detect_sequence_type("AUGCGAUCG"),
            SequenceType::RNA
        );
        assert_eq!(
            tokenizer.detect_sequence_type("MTKQVFTPG"),
            SequenceType::Protein
        );
    }

    #[test]
    fn test_dna_encoding() {
        let tokenizer = BioTokenizer::new();
        let result = tokenizer.encode("ATGCGATCG");
        assert!(result.is_ok());
        let tokenized = result.unwrap();
        assert!(!tokenized.input_ids.is_empty());
    }

    #[test]
    fn test_protein_encoding() {
        let tokenizer = BioTokenizer::new();
        let result = tokenizer.encode("MTKQVFTPG");
        assert!(result.is_ok());
        let tokenized = result.unwrap();
        assert!(!tokenized.input_ids.is_empty());
    }

    #[test]
    fn test_kmer_tokenization() {
        let mut config = BioTokenizerConfig::default();
        config.kmer_size = Some(3);
        let tokenizer = BioTokenizer::with_config(config);

        let tokens = tokenizer.tokenize_bio("ATGCGATCG").unwrap();
        assert!(tokens.iter().any(|t| t.text.len() == 3));
    }

    #[test]
    fn test_translation() {
        let tokenizer = BioTokenizer::new();
        let protein = tokenizer.translate_dna("ATGAAATAG").unwrap();
        assert_eq!(protein, "MK"); // ATG=M, AAA=K, TAG=stop
    }

    #[test]
    fn test_reverse_complement() {
        let tokenizer = BioTokenizer::new();
        let rc = tokenizer.reverse_complement("ATGC");
        assert_eq!(rc, "GCAT");
    }

    #[test]
    fn test_amino_acid_metadata() {
        let tokenizer = BioTokenizer::new();
        let metadata = tokenizer.create_amino_acid_metadata('A');
        assert!(metadata.is_some());
        let meta = metadata.unwrap();
        assert!(meta.molecular_weight.is_some());
        assert!(meta.hydrophobicity.is_some());
    }

    #[test]
    fn test_nucleotide_metadata() {
        let tokenizer = BioTokenizer::new();
        let metadata = tokenizer.create_nucleotide_metadata('G');
        assert!(metadata.is_some());
        let meta = metadata.unwrap();
        assert_eq!(meta.gc_content, Some(1.0));
    }

    #[test]
    fn test_bio_analysis() {
        let tokenizer = BioTokenizer::new();
        let analysis = tokenizer.analyze("ATGCGATCG");
        assert!(analysis.is_ok());
        let result = analysis.unwrap();
        assert!(result.gc_content.is_some());
        assert!(!result.nucleotide_composition.is_empty());
    }

    #[test]
    fn test_protein_analysis() {
        let tokenizer = BioTokenizer::new();
        let analysis = tokenizer.analyze("MTKQVFTPG");
        assert!(analysis.is_ok());
        let result = analysis.unwrap();
        assert!(result.molecular_weight.is_some());
        assert!(result.avg_hydrophobicity.is_some());
        assert!(!result.amino_acid_composition.is_empty());
    }

    #[test]
    fn test_stop_codon_detection() {
        let tokenizer = BioTokenizer::new();
        let tokens = tokenizer.tokenize_bio("ATGTAG").unwrap();
        assert!(tokens.iter().any(|t| t.token_type == BioTokenType::StartCodon));
        assert!(tokens.iter().any(|t| t.token_type == BioTokenType::StopCodon));
    }

    #[test]
    fn test_max_length_constraint() {
        let mut config = BioTokenizerConfig::default();
        config.max_length = Some(5);
        let tokenizer = BioTokenizer::with_config(config);

        let result = tokenizer.encode("ATGCGATCGATCGATCG");
        assert!(result.is_ok());
        let tokenized = result.unwrap();
        assert!(tokenized.input_ids.len() <= 5);
    }
}
