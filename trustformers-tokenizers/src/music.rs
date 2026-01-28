//! Music notation tokenizer for TrustformeRS
//!
//! This module provides specialized tokenization for various music notation formats
//! including ABC notation, MusicXML, MIDI, and other musical representations.

use crate::{TokenizedInput, Tokenizer};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;

/// Configuration for music tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicTokenizerConfig {
    /// Maximum sequence length
    pub max_length: Option<usize>,
    /// Whether to include special music tokens
    pub include_special_tokens: bool,
    /// Whether to tokenize ABC notation
    pub tokenize_abc: bool,
    /// Whether to tokenize MusicXML
    pub tokenize_musicxml: bool,
    /// Whether to tokenize MIDI representations
    pub tokenize_midi: bool,
    /// Whether to tokenize chord symbols
    pub tokenize_chords: bool,
    /// Whether to preserve timing information
    pub preserve_timing: bool,
    /// Whether to preserve dynamics
    pub preserve_dynamics: bool,
    /// Vocabulary size limit
    pub vocab_size: Option<usize>,
    /// Time resolution for rhythmic values
    pub time_resolution: u32,
}

impl Default for MusicTokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: Some(1024),
            include_special_tokens: true,
            tokenize_abc: true,
            tokenize_musicxml: false,
            tokenize_midi: false,
            tokenize_chords: true,
            preserve_timing: true,
            preserve_dynamics: true,
            vocab_size: Some(5000),
            time_resolution: 480, // Standard MIDI ticks per quarter note
        }
    }
}

/// Types of music tokens
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MusicTokenType {
    /// Note names (C, D, E, F, G, A, B)
    NoteName,
    /// Accidentals (sharp, flat, natural)
    Accidental,
    /// Octave indicators
    Octave,
    /// Rhythmic values (whole, half, quarter, etc.)
    Duration,
    /// Rest symbols
    Rest,
    /// Time signatures
    TimeSignature,
    /// Key signatures
    KeySignature,
    /// Tempo markings
    Tempo,
    /// Dynamic markings (pp, p, mp, mf, f, ff)
    Dynamic,
    /// Articulation marks (staccato, legato, etc.)
    Articulation,
    /// Chord symbols (C, Am, G7, etc.)
    Chord,
    /// Barlines
    Barline,
    /// Clef symbols
    Clef,
    /// Ornaments (trill, mordent, etc.)
    Ornament,
    /// Special notation symbols
    Special,
    /// Measure numbers
    Measure,
    /// Lyrics
    Lyric,
    /// Unknown tokens
    Unknown,
}

/// Music token with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicToken {
    /// Token text
    pub text: String,
    /// Token type
    pub token_type: MusicTokenType,
    /// Start position in original text
    pub start: usize,
    /// End position in original text
    pub end: usize,
    /// Musical metadata
    pub metadata: Option<MusicTokenMetadata>,
}

/// Metadata for music tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicTokenMetadata {
    /// MIDI note number (0-127)
    pub midi_note: Option<u8>,
    /// Duration in ticks
    pub duration_ticks: Option<u32>,
    /// Velocity (dynamics)
    pub velocity: Option<u8>,
    /// Pitch class (0-11)
    pub pitch_class: Option<u8>,
    /// Octave number
    pub octave: Option<i8>,
    /// Beat position
    pub beat_position: Option<f64>,
    /// Measure number
    pub measure: Option<u32>,
    /// Voice/channel number
    pub voice: Option<u8>,
}

/// Music tokenizer implementation
pub struct MusicTokenizer {
    config: MusicTokenizerConfig,
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    next_id: u32,
    note_names: HashMap<String, u8>,
    chord_patterns: Vec<Regex>,
    abc_patterns: Vec<Regex>,
    dynamics: HashMap<String, u8>,
}

// Static data for musical notes
static NOTE_NAMES: Lazy<HashMap<String, u8>> = Lazy::new(|| {
    let mut map = HashMap::new();
    // Chromatic notes with MIDI numbers (C4 = 60)
    map.insert("C".to_string(), 0);
    map.insert("C#".to_string(), 1);
    map.insert("Db".to_string(), 1);
    map.insert("D".to_string(), 2);
    map.insert("D#".to_string(), 3);
    map.insert("Eb".to_string(), 3);
    map.insert("E".to_string(), 4);
    map.insert("F".to_string(), 5);
    map.insert("F#".to_string(), 6);
    map.insert("Gb".to_string(), 6);
    map.insert("G".to_string(), 7);
    map.insert("G#".to_string(), 8);
    map.insert("Ab".to_string(), 8);
    map.insert("A".to_string(), 9);
    map.insert("A#".to_string(), 10);
    map.insert("Bb".to_string(), 10);
    map.insert("B".to_string(), 11);
    map
});

// Dynamic markings with velocity values
static DYNAMICS: Lazy<HashMap<String, u8>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("ppp".to_string(), 16);
    map.insert("pp".to_string(), 32);
    map.insert("p".to_string(), 48);
    map.insert("mp".to_string(), 64);
    map.insert("mf".to_string(), 80);
    map.insert("f".to_string(), 96);
    map.insert("ff".to_string(), 112);
    map.insert("fff".to_string(), 127);
    map
});

impl Default for MusicTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MusicTokenizer {
    /// Create a new music tokenizer
    pub fn new() -> Self {
        Self::with_config(MusicTokenizerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: MusicTokenizerConfig) -> Self {
        let mut tokenizer = Self {
            config,
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            next_id: 0,
            note_names: NOTE_NAMES.clone(),
            chord_patterns: Self::create_chord_patterns(),
            abc_patterns: Self::create_abc_patterns(),
            dynamics: DYNAMICS.clone(),
        };

        tokenizer.initialize_vocab();
        tokenizer
    }

    /// Initialize vocabulary with music tokens
    fn initialize_vocab(&mut self) {
        // Add special tokens
        if self.config.include_special_tokens {
            self.add_token("[CLS]");
            self.add_token("[SEP]");
            self.add_token("[PAD]");
            self.add_token("[UNK]");
            self.add_token("[MASK]");
            self.add_token("[START_MUSIC]");
            self.add_token("[END_MUSIC]");
            self.add_token("[BAR]");
            self.add_token("[BEAT]");
            self.add_token("[MEASURE]");
        }

        // Add note names
        let note_names: Vec<String> = self.note_names.keys().cloned().collect();
        for note in note_names {
            self.add_token(&note);
        }

        // Add accidentals
        self.add_token("#");
        self.add_token("b");
        self.add_token("♯");
        self.add_token("♭");
        self.add_token("♮"); // Natural

        // Add octave numbers
        for octave in 0..9 {
            self.add_token(&octave.to_string());
        }

        // Add duration symbols (ABC notation)
        if self.config.tokenize_abc {
            self.add_token("1"); // Whole note
            self.add_token("2"); // Half note
            self.add_token("4"); // Quarter note
            self.add_token("8"); // Eighth note
            self.add_token("16"); // Sixteenth note
            self.add_token("32"); // Thirty-second note
            self.add_token("/"); // Division symbol
            self.add_token("."); // Dotted rhythm
            self.add_token("z"); // Rest
        }

        // Add dynamics
        if self.config.preserve_dynamics {
            let dynamics: Vec<String> = self.dynamics.keys().cloned().collect();
            for dynamic in dynamics {
                self.add_token(&dynamic);
            }
        }

        // Add time signatures
        self.add_token("4/4");
        self.add_token("3/4");
        self.add_token("2/4");
        self.add_token("6/8");
        self.add_token("9/8");
        self.add_token("12/8");

        // Add key signatures
        let keys = [
            "C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb",
        ];
        for key in &keys {
            self.add_token(&format!("{}maj", key));
            self.add_token(&format!("{}min", key));
        }

        // Add clefs
        self.add_token("treble");
        self.add_token("bass");
        self.add_token("alto");
        self.add_token("tenor");

        // Add articulations
        self.add_token("staccato");
        self.add_token("legato");
        self.add_token("accent");
        self.add_token("tenuto");

        // Add ornaments
        self.add_token("trill");
        self.add_token("mordent");
        self.add_token("turn");
        self.add_token("appoggiatura");

        // Add chord symbols if enabled
        if self.config.tokenize_chords {
            let chord_types = ["", "m", "7", "M7", "m7", "dim", "aug", "sus2", "sus4"];
            for note in &["C", "D", "E", "F", "G", "A", "B"] {
                for chord_type in &chord_types {
                    self.add_token(&format!("{}{}", note, chord_type));
                }
            }
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

    /// Create chord recognition patterns
    fn create_chord_patterns() -> Vec<Regex> {
        vec![
            // Basic triads and sevenths
            Regex::new(r"[A-G][#b]?(m|maj|M|min|dim|aug|sus[24])?[67]?").expect("valid regex"),
            // Complex chord extensions
            Regex::new(r"[A-G][#b]?(add|sus|maj|min|dim|aug)\d+").expect("valid regex"),
            // Slash chords
            Regex::new(r"[A-G][#b]?[^/]*/[A-G][#b]?").expect("valid regex"),
        ]
    }

    /// Create ABC notation patterns
    fn create_abc_patterns() -> Vec<Regex> {
        vec![
            // ABC notes with accidentals and octaves
            Regex::new(r"[_=^]*[A-Ga-g][',]*").expect("valid regex"),
            // Duration modifiers
            Regex::new(r"\d*/?\.?").expect("valid regex"),
            // Rests
            Regex::new(r"z\d*/?\.?").expect("valid regex"),
            // Barlines
            Regex::new(r"\|[\|:\]]*").expect("valid regex"),
            // Chords (bracketed notes)
            Regex::new(r"\[([A-Ga-g][',]*\d*/?\.?)+\]").expect("valid regex"),
            // Slurs and ties
            Regex::new(r"[()~-]").expect("valid regex"),
        ]
    }

    /// Tokenize music notation
    pub fn tokenize_music(&self, text: &str) -> Result<Vec<MusicToken>> {
        let mut tokens = Vec::new();
        let mut pos = 0;

        // Detect music notation type
        if self.is_abc_notation(text) && self.config.tokenize_abc {
            tokens.extend(self.tokenize_abc(text, &mut pos)?);
        } else if self.is_chord_progression(text) && self.config.tokenize_chords {
            tokens.extend(self.tokenize_chords(text, &mut pos)?);
        } else if text.contains("<note>") && self.config.tokenize_musicxml {
            tokens.extend(self.tokenize_musicxml(text, &mut pos)?);
        } else {
            // Fallback tokenization
            tokens.extend(self.tokenize_fallback(text, &mut pos)?);
        }

        Ok(tokens)
    }

    /// Check if text is ABC notation
    fn is_abc_notation(&self, text: &str) -> bool {
        // ABC notation contains note letters with optional accidentals, barlines, rests
        // Accept if it has note letters and either has ABC-specific chars or is primarily notes
        let has_notes = text.chars().any(|c| "ABCDEFGabcdefg".contains(c));
        let has_abc_chars = text.chars().any(|c| "|zx'.^_=(),".contains(c));
        let note_ratio = text.chars().filter(|c| "ABCDEFGabcdefg".contains(*c)).count() as f64
            / text.len() as f64;

        has_notes && (has_abc_chars || note_ratio > 0.5)
    }

    /// Check if text is chord progression
    fn is_chord_progression(&self, text: &str) -> bool {
        self.chord_patterns.iter().any(|pattern| pattern.is_match(text))
    }

    /// Tokenize ABC notation
    fn tokenize_abc(&self, text: &str, pos: &mut usize) -> Result<Vec<MusicToken>> {
        let mut tokens = Vec::new();
        let mut current_pos = *pos;

        for pattern in &self.abc_patterns {
            for mat in pattern.find_iter(text) {
                if mat.start() >= current_pos {
                    let token_text = mat.as_str().to_string();
                    let token_type = self.classify_abc_token(&token_text);
                    let metadata = self.create_music_metadata(&token_text, &token_type);

                    tokens.push(MusicToken {
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

    /// Tokenize chord progressions
    fn tokenize_chords(&self, text: &str, pos: &mut usize) -> Result<Vec<MusicToken>> {
        let mut tokens = Vec::new();
        let mut current_pos = *pos;

        for pattern in &self.chord_patterns {
            for mat in pattern.find_iter(text) {
                if mat.start() >= current_pos {
                    let token_text = mat.as_str().to_string();
                    let token_type = MusicTokenType::Chord;
                    let metadata = self.create_chord_metadata(&token_text);

                    tokens.push(MusicToken {
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

    /// Tokenize MusicXML (simplified)
    fn tokenize_musicxml(&self, text: &str, pos: &mut usize) -> Result<Vec<MusicToken>> {
        let mut tokens = Vec::new();

        // Simple XML tag extraction
        let tag_regex = Regex::new(r"<([^>]+)>([^<]*)</[^>]+>").expect("valid regex");

        for mat in tag_regex.find_iter(text) {
            let token_text = mat.as_str().to_string();
            let token_type = MusicTokenType::Special;

            tokens.push(MusicToken {
                text: token_text,
                token_type,
                start: mat.start(),
                end: mat.end(),
                metadata: None,
            });
        }

        *pos += text.len();
        Ok(tokens)
    }

    /// Fallback tokenization
    fn tokenize_fallback(&self, text: &str, pos: &mut usize) -> Result<Vec<MusicToken>> {
        let mut tokens = Vec::new();

        for (i, ch) in text.char_indices() {
            tokens.push(MusicToken {
                text: ch.to_string(),
                token_type: MusicTokenType::Unknown,
                start: *pos + i,
                end: *pos + i + ch.len_utf8(),
                metadata: None,
            });
        }

        *pos += text.len();
        Ok(tokens)
    }

    /// Classify ABC token type
    fn classify_abc_token(&self, token: &str) -> MusicTokenType {
        if token.starts_with('z') {
            MusicTokenType::Rest
        } else if token.starts_with('|') {
            MusicTokenType::Barline
        } else if token.chars().any(|c| "ABCDEFGabcdefg".contains(c)) {
            MusicTokenType::NoteName
        } else if token.chars().all(|c| c.is_ascii_digit() || "/".contains(c))
            || token.contains('.')
        {
            MusicTokenType::Duration
        } else {
            MusicTokenType::Unknown
        }
    }

    /// Create music metadata
    fn create_music_metadata(
        &self,
        token: &str,
        token_type: &MusicTokenType,
    ) -> Option<MusicTokenMetadata> {
        match token_type {
            MusicTokenType::NoteName => self.parse_note_metadata(token),
            MusicTokenType::Duration => self.parse_duration_metadata(token),
            MusicTokenType::Dynamic => {
                if let Some(&velocity) = self.dynamics.get(token) {
                    Some(MusicTokenMetadata {
                        midi_note: None,
                        duration_ticks: None,
                        velocity: Some(velocity),
                        pitch_class: None,
                        octave: None,
                        beat_position: None,
                        measure: None,
                        voice: None,
                    })
                } else {
                    None
                }
            },
            _ => None,
        }
    }

    /// Parse note metadata from ABC notation
    fn parse_note_metadata(&self, token: &str) -> Option<MusicTokenMetadata> {
        let mut chars = token.chars().peekable();
        let mut accidental = 0i8; // -1 for flat, 0 for natural, 1 for sharp
        let mut note_char = None;
        let mut octave = 4i8; // Default octave

        // Parse accidentals
        while let Some(&ch) = chars.peek() {
            match ch {
                '_' => {
                    accidental -= 1;
                    chars.next();
                },
                '^' => {
                    accidental += 1;
                    chars.next();
                },
                '=' => {
                    accidental = 0;
                    chars.next();
                },
                _ => break,
            }
        }

        // Parse note
        if let Some(ch) = chars.next() {
            if "ABCDEFGabcdefg".contains(ch) {
                note_char = Some(ch);
                // Lowercase notes are in higher octave
                if ch.is_lowercase() {
                    octave += 1;
                }
            }
        }

        // Parse octave modifiers
        while let Some(&ch) = chars.peek() {
            match ch {
                '\'' => {
                    octave += 1;
                    chars.next();
                },
                ',' => {
                    octave -= 1;
                    chars.next();
                },
                _ => break,
            }
        }

        if let Some(note) = note_char {
            let note_upper = note.to_ascii_uppercase();
            if let Some(&base_pitch) = self.note_names.get(&note_upper.to_string()) {
                let pitch_class = (base_pitch as i8 + accidental).rem_euclid(12) as u8;
                let midi_note = (octave * 12 + pitch_class as i8) as u8;

                Some(MusicTokenMetadata {
                    midi_note: Some(midi_note),
                    duration_ticks: None,
                    velocity: None,
                    pitch_class: Some(pitch_class),
                    octave: Some(octave),
                    beat_position: None,
                    measure: None,
                    voice: None,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Parse duration metadata
    fn parse_duration_metadata(&self, token: &str) -> Option<MusicTokenMetadata> {
        // Simple duration parsing for ABC notation
        let base_duration = self.config.time_resolution; // Quarter note duration

        let duration_ticks = if token.contains('/') {
            // Fractional duration
            if let Some(slash_pos) = token.find('/') {
                let numerator = token[..slash_pos].parse::<u32>().unwrap_or(1);
                let denominator = token[slash_pos + 1..].parse::<u32>().unwrap_or(2);
                base_duration * numerator / denominator
            } else {
                base_duration / 2
            }
        } else if let Ok(multiplier) = token.parse::<u32>() {
            base_duration * multiplier
        } else {
            base_duration
        };

        Some(MusicTokenMetadata {
            midi_note: None,
            duration_ticks: Some(duration_ticks),
            velocity: None,
            pitch_class: None,
            octave: None,
            beat_position: None,
            measure: None,
            voice: None,
        })
    }

    /// Create chord metadata
    fn create_chord_metadata(&self, chord: &str) -> Option<MusicTokenMetadata> {
        // Extract root note from chord symbol
        if let Some(root_char) = chord.chars().next() {
            if "ABCDEFG".contains(root_char) {
                let root_str = root_char.to_string();
                if let Some(&pitch_class) = self.note_names.get(&root_str) {
                    return Some(MusicTokenMetadata {
                        midi_note: Some(60 + pitch_class), // C4 + pitch class
                        duration_ticks: None,
                        velocity: None,
                        pitch_class: Some(pitch_class),
                        octave: Some(4),
                        beat_position: None,
                        measure: None,
                        voice: None,
                    });
                }
            }
        }
        None
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
    pub fn config(&self) -> &MusicTokenizerConfig {
        &self.config
    }
}

impl Tokenizer for MusicTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let music_tokens = self.tokenize_music(text)?;
        let mut input_ids = Vec::new();

        for token in music_tokens {
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

        Ok(TokenizedInput {
            input_ids: input_ids.clone(),
            attention_mask: vec![1; input_ids.len()],
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

/// Music analysis results
pub struct MusicAnalysis {
    /// Token type distribution
    pub token_types: HashMap<MusicTokenType, usize>,
    /// Note distribution
    pub note_distribution: HashMap<String, usize>,
    /// Chord progression
    pub chord_progression: Vec<String>,
    /// Key signature (estimated)
    pub estimated_key: Option<String>,
    /// Time signature
    pub time_signature: Option<String>,
    /// Average tempo (if detectable)
    pub estimated_tempo: Option<f64>,
    /// Complexity score
    pub complexity_score: f64,
}

impl MusicTokenizer {
    /// Analyze musical content
    pub fn analyze(&self, text: &str) -> Result<MusicAnalysis> {
        let tokens = self.tokenize_music(text)?;

        let mut token_types = HashMap::new();
        let mut note_distribution = HashMap::new();
        let mut chord_progression = Vec::new();

        for token in &tokens {
            *token_types.entry(token.token_type.clone()).or_insert(0) += 1;

            match &token.token_type {
                MusicTokenType::NoteName => {
                    *note_distribution.entry(token.text.clone()).or_insert(0) += 1;
                },
                MusicTokenType::Chord => {
                    chord_progression.push(token.text.clone());
                },
                _ => {},
            }
        }

        let complexity_score = self.calculate_music_complexity(&tokens);
        let estimated_key = self.estimate_key(&note_distribution);

        Ok(MusicAnalysis {
            token_types,
            note_distribution,
            chord_progression,
            estimated_key,
            time_signature: None,  // Would need more sophisticated analysis
            estimated_tempo: None, // Would need timing information
            complexity_score,
        })
    }

    /// Calculate musical complexity score
    fn calculate_music_complexity(&self, tokens: &[MusicToken]) -> f64 {
        let mut score = 0.0;

        // Base score from number of tokens
        score += tokens.len() as f64 * 0.1;

        // Additional score for different token types
        let mut token_type_count = HashMap::new();
        for token in tokens {
            *token_type_count.entry(&token.token_type).or_insert(0) += 1;
        }

        score += token_type_count.len() as f64 * 0.5;

        // Bonus for musical elements
        for token in tokens {
            match token.token_type {
                MusicTokenType::Chord => score += 1.0,
                MusicTokenType::Ornament => score += 1.5,
                MusicTokenType::Dynamic => score += 0.5,
                MusicTokenType::Articulation => score += 0.3,
                _ => {},
            }
        }

        score
    }

    /// Estimate key signature from note distribution
    fn estimate_key(&self, note_distribution: &HashMap<String, usize>) -> Option<String> {
        // Simple key estimation based on most frequent notes
        if let Some((most_frequent_note, _)) =
            note_distribution.iter().max_by_key(|(_, &count)| count)
        {
            Some(format!(
                "{}maj",
                most_frequent_note.chars().next().unwrap_or('C')
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_music_tokenizer_creation() {
        let tokenizer = MusicTokenizer::new();
        assert!(tokenizer.get_vocab().len() > 0);
        assert!(tokenizer.get_vocab().contains_key("C"));
        assert!(tokenizer.get_vocab().contains_key("G"));
    }

    #[test]
    fn test_abc_detection() {
        let tokenizer = MusicTokenizer::new();
        assert!(tokenizer.is_abc_notation("CDEFGAB"));
        assert!(tokenizer.is_abc_notation("C4 D4 E4 | F4 G4 A4 B4"));
        assert!(!tokenizer.is_abc_notation("hello world"));
    }

    #[test]
    fn test_chord_detection() {
        let tokenizer = MusicTokenizer::new();
        assert!(tokenizer.is_chord_progression("C Am F G"));
        assert!(tokenizer.is_chord_progression("Cmaj7 Dm7 G7"));
        assert!(!tokenizer.is_chord_progression("hello world"));
    }

    #[test]
    fn test_abc_encoding() {
        let tokenizer = MusicTokenizer::new();
        let result = tokenizer.encode("CDEFGAB");
        assert!(result.is_ok());
        let tokenized = result.unwrap();
        assert!(!tokenized.input_ids.is_empty());
    }

    #[test]
    fn test_chord_encoding() {
        let tokenizer = MusicTokenizer::new();
        let result = tokenizer.encode("C Am F G");
        assert!(result.is_ok());
        let tokenized = result.unwrap();
        assert!(!tokenized.input_ids.is_empty());
    }

    #[test]
    fn test_note_metadata_parsing() {
        let tokenizer = MusicTokenizer::new();
        let metadata = tokenizer.parse_note_metadata("C");
        assert!(metadata.is_some());
        let meta = metadata.unwrap();
        assert_eq!(meta.pitch_class, Some(0)); // C = 0
        assert_eq!(meta.octave, Some(4));
    }

    #[test]
    fn test_accidental_parsing() {
        let tokenizer = MusicTokenizer::new();
        let metadata = tokenizer.parse_note_metadata("^C");
        assert!(metadata.is_some());
        let meta = metadata.unwrap();
        assert_eq!(meta.pitch_class, Some(1)); // C# = 1
    }

    #[test]
    fn test_duration_parsing() {
        let tokenizer = MusicTokenizer::new();
        let metadata = tokenizer.parse_duration_metadata("4");
        assert!(metadata.is_some());
        let meta = metadata.unwrap();
        assert!(meta.duration_ticks.is_some());
    }

    #[test]
    fn test_music_analysis() {
        let tokenizer = MusicTokenizer::new();
        let analysis = tokenizer.analyze("CDEFGAB");
        assert!(analysis.is_ok());
        let result = analysis.unwrap();
        assert!(!result.note_distribution.is_empty());
        assert!(result.complexity_score > 0.0);
    }

    #[test]
    fn test_chord_metadata() {
        let tokenizer = MusicTokenizer::new();
        let metadata = tokenizer.create_chord_metadata("Cmaj7");
        assert!(metadata.is_some());
        let meta = metadata.unwrap();
        assert_eq!(meta.pitch_class, Some(0)); // C = 0
    }

    #[test]
    fn test_encoding_decoding_consistency() {
        let tokenizer = MusicTokenizer::new();
        let original = "CDEFG";
        let encoded = tokenizer.encode(original).unwrap();
        let decoded = tokenizer.decode(&encoded.input_ids).unwrap();
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_max_length_constraint() {
        let mut config = MusicTokenizerConfig::default();
        config.max_length = Some(5);
        let tokenizer = MusicTokenizer::with_config(config);

        let result = tokenizer.encode("CDEFGABCDEFGAB");
        assert!(result.is_ok());
        let tokenized = result.unwrap();
        assert!(tokenized.input_ids.len() <= 5);
    }

    #[test]
    fn test_dynamic_recognition() {
        let tokenizer = MusicTokenizer::new();
        assert!(tokenizer.get_vocab().contains_key("ff"));
        assert!(tokenizer.get_vocab().contains_key("pp"));

        if let Some(&velocity) = tokenizer.dynamics.get("ff") {
            assert!(velocity > 100);
        }
    }
}
