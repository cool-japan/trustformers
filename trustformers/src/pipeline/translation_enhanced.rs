//! Enhanced translation pipeline with language detection and batch processing.
//!
//! Provides heuristic language detection via Unicode character ranges and
//! a multi-language translation pipeline supporting per-language-pair model
//! overrides and batch translation.

use std::collections::HashMap;

// ── Language ──────────────────────────────────────────────────────────────────

/// ISO 639-1 language codes with support for custom `Other` variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Language {
    English,
    French,
    German,
    Spanish,
    Italian,
    Portuguese,
    Dutch,
    Russian,
    Chinese,
    Japanese,
    Korean,
    Arabic,
    Hindi,
    Turkish,
    Polish,
    Other(String),
}

impl Language {
    /// Return the ISO 639-1 code string for this language.
    pub fn code(&self) -> &str {
        match self {
            Language::English => "en",
            Language::French => "fr",
            Language::German => "de",
            Language::Spanish => "es",
            Language::Italian => "it",
            Language::Portuguese => "pt",
            Language::Dutch => "nl",
            Language::Russian => "ru",
            Language::Chinese => "zh",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Arabic => "ar",
            Language::Hindi => "hi",
            Language::Turkish => "tr",
            Language::Polish => "pl",
            Language::Other(code) => code,
        }
    }

    /// Construct a `Language` from an ISO 639-1 code string.
    pub fn from_code(code: &str) -> Self {
        match code {
            "en" => Language::English,
            "fr" => Language::French,
            "de" => Language::German,
            "es" => Language::Spanish,
            "it" => Language::Italian,
            "pt" => Language::Portuguese,
            "nl" => Language::Dutch,
            "ru" => Language::Russian,
            "zh" => Language::Chinese,
            "ja" => Language::Japanese,
            "ko" => Language::Korean,
            "ar" => Language::Arabic,
            "hi" => Language::Hindi,
            "tr" => Language::Turkish,
            "pl" => Language::Polish,
            other => Language::Other(other.to_string()),
        }
    }

    /// Human-readable name for the language.
    pub fn name(&self) -> &str {
        match self {
            Language::English => "English",
            Language::French => "French",
            Language::German => "German",
            Language::Spanish => "Spanish",
            Language::Italian => "Italian",
            Language::Portuguese => "Portuguese",
            Language::Dutch => "Dutch",
            Language::Russian => "Russian",
            Language::Chinese => "Chinese",
            Language::Japanese => "Japanese",
            Language::Korean => "Korean",
            Language::Arabic => "Arabic",
            Language::Hindi => "Hindi",
            Language::Turkish => "Turkish",
            Language::Polish => "Polish",
            Language::Other(_) => "Unknown",
        }
    }

    /// The writing script system used by this language.
    pub fn script(&self) -> Script {
        match self {
            Language::English
            | Language::French
            | Language::German
            | Language::Spanish
            | Language::Italian
            | Language::Portuguese
            | Language::Dutch
            | Language::Turkish
            | Language::Polish => Script::Latin,
            Language::Russian => Script::Cyrillic,
            Language::Chinese => Script::Chinese,
            Language::Japanese => Script::Japanese,
            Language::Korean => Script::Korean,
            Language::Arabic => Script::Arabic,
            Language::Hindi => Script::Devanagari,
            Language::Other(_) => Script::Other,
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.name(), self.code())
    }
}

// ── Script ────────────────────────────────────────────────────────────────────

/// Writing script system.
#[derive(Debug, Clone, PartialEq)]
pub enum Script {
    Latin,
    Cyrillic,
    Chinese,
    Japanese,
    Korean,
    Arabic,
    Devanagari,
    Other,
}

// ── DetectionResult ───────────────────────────────────────────────────────────

/// Language detection result with confidence and ranked alternatives.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub language: Language,
    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Alternative (language, confidence) pairs, sorted descending by confidence.
    pub alternatives: Vec<(Language, f32)>,
}

// ── LanguageDetector ──────────────────────────────────────────────────────────

/// Heuristic language detector that uses Unicode character ranges and simple
/// word-pattern matching for Latin-script languages.
pub struct LanguageDetector;

impl LanguageDetector {
    /// Detect the language of `text` using character-level heuristics.
    ///
    /// # Detection strategy
    ///
    /// 1. **Script detection** via Unicode code-point ranges:
    ///    - Arabic (U+0600–U+06FF) → Arabic
    ///    - Devanagari (U+0900–U+097F) → Hindi
    ///    - Cyrillic (U+0400–U+04FF) → Russian
    ///    - Hangul (U+AC00–U+D7FF) → Korean
    ///    - CJK Unified (U+4E00–U+9FFF) + Hiragana/Katakana → Japanese; else Chinese
    ///
    /// 2. **Latin-script word patterns** for common function words / determiners.
    ///
    /// 3. Falls back to English with low confidence.
    pub fn detect(text: &str) -> DetectionResult {
        // Count characters in each script range
        let mut arabic_count = 0u32;
        let mut devanagari_count = 0u32;
        let mut cyrillic_count = 0u32;
        let mut hangul_count = 0u32;
        let mut cjk_count = 0u32;
        let mut hiragana_count = 0u32;
        let mut katakana_count = 0u32;
        let mut total_chars = 0u32;

        for ch in text.chars() {
            let cp = ch as u32;
            total_chars += 1;
            if (0x0600..=0x06FF).contains(&cp) {
                arabic_count += 1;
            } else if (0x0900..=0x097F).contains(&cp) {
                devanagari_count += 1;
            } else if (0x0400..=0x04FF).contains(&cp) {
                cyrillic_count += 1;
            } else if (0xAC00..=0xD7FF).contains(&cp) {
                hangul_count += 1;
            } else if (0x4E00..=0x9FFF).contains(&cp) {
                cjk_count += 1;
            } else if (0x3040..=0x309F).contains(&cp) {
                hiragana_count += 1;
            } else if (0x30A0..=0x30FF).contains(&cp) {
                katakana_count += 1;
            }
        }

        let total = total_chars.max(1) as f32;

        // Helper: fraction of text in a given script
        let frac = |n: u32| n as f32 / total;

        // Check non-Latin scripts first (high precision)
        if frac(arabic_count) > 0.05 {
            return DetectionResult {
                language: Language::Arabic,
                confidence: (frac(arabic_count) * 2.0).min(1.0),
                alternatives: vec![],
            };
        }
        if frac(devanagari_count) > 0.05 {
            return DetectionResult {
                language: Language::Hindi,
                confidence: (frac(devanagari_count) * 2.0).min(1.0),
                alternatives: vec![],
            };
        }
        if frac(cyrillic_count) > 0.05 {
            return DetectionResult {
                language: Language::Russian,
                confidence: (frac(cyrillic_count) * 2.0).min(1.0),
                alternatives: vec![],
            };
        }
        if frac(hangul_count) > 0.05 {
            return DetectionResult {
                language: Language::Korean,
                confidence: (frac(hangul_count) * 2.0).min(1.0),
                alternatives: vec![],
            };
        }
        let cjk_total = cjk_count + hiragana_count + katakana_count;
        if frac(cjk_total) > 0.05 {
            // Japanese if there is any hiragana or katakana; otherwise Chinese
            if hiragana_count > 0 || katakana_count > 0 {
                return DetectionResult {
                    language: Language::Japanese,
                    confidence: (frac(cjk_total) * 2.0).min(1.0),
                    alternatives: vec![(Language::Chinese, 0.2)],
                };
            }
            return DetectionResult {
                language: Language::Chinese,
                confidence: (frac(cjk_total) * 2.0).min(1.0),
                alternatives: vec![(Language::Japanese, 0.15)],
            };
        }

        // Latin-script heuristics: score each language by common function words
        let lower = text.to_lowercase();
        let words: Vec<&str> =
            lower.split(|c: char| !c.is_alphabetic()).filter(|s| !s.is_empty()).collect();
        let total_words = words.len().max(1) as f32;

        // (pattern_words, language, weight)
        let patterns: &[(&[&str], Language, f32)] = &[
            (
                &["the", "is", "are", "was", "were", "this", "that", "have"],
                Language::English,
                1.5,
            ),
            (
                &["le", "la", "les", "des", "une", "est", "que", "qui", "dans"],
                Language::French,
                1.5,
            ),
            (
                &[
                    "der", "die", "das", "die", "ist", "sind", "und", "ein", "eine",
                ],
                Language::German,
                1.5,
            ),
            (
                &[
                    "el", "los", "las", "una", "son", "para", "con", "que", "del",
                ],
                Language::Spanish,
                1.5,
            ),
            (
                &[
                    "il", "gli", "una", "sono", "per", "con", "che", "del", "della",
                ],
                Language::Italian,
                1.2,
            ),
            (
                &[
                    "os", "mas", "para", "com", "que", "por", "uma", "nao", "dos",
                ],
                Language::Portuguese,
                1.2,
            ),
            (
                &[
                    "de", "het", "een", "zijn", "met", "voor", "niet", "ook", "van",
                ],
                Language::Dutch,
                1.2,
            ),
            (
                &["ve", "bir", "bu", "ile", "da", "de", "bu", "ama", "olan"],
                Language::Turkish,
                1.0,
            ),
            (
                &[
                    "sie", "nie", "jest", "lub", "oraz", "jak", "ale", "tak", "ten",
                ],
                Language::Polish,
                1.0,
            ),
        ];

        let mut scored: Vec<(Language, f32)> = patterns
            .iter()
            .map(|(pattern_words, lang, weight)| {
                let hits: usize = words.iter().filter(|w| pattern_words.contains(w)).count();
                let score = (hits as f32 / total_words) * weight;
                (lang.clone(), score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_score = scored[0].1;
        let best_lang = scored[0].0.clone();

        let (language, confidence) = if best_score < 0.01 {
            // No strong signal — default to English
            (Language::English, 0.3)
        } else {
            (best_lang, (best_score * 8.0).min(0.95))
        };

        // Build alternatives list (exclude the top result)
        let alternatives: Vec<(Language, f32)> = scored
            .into_iter()
            .skip(1)
            .filter(|(_, s)| *s > 0.0)
            .take(3)
            .map(|(l, s)| (l, (s * 8.0).min(0.95)))
            .collect();

        DetectionResult {
            language,
            confidence,
            alternatives,
        }
    }
}

// ── Formality ─────────────────────────────────────────────────────────────────

/// Requested formality level for a translation.
#[derive(Debug, Clone, PartialEq)]
pub enum Formality {
    Formal,
    Informal,
    Default,
}

// ── TranslationRequest ────────────────────────────────────────────────────────

/// A single translation request.
#[derive(Debug, Clone)]
pub struct TranslationRequest {
    pub text: String,
    /// `None` means auto-detect.
    pub source_language: Option<Language>,
    pub target_language: Language,
    pub formality: Formality,
    pub preserve_formatting: bool,
}

impl TranslationRequest {
    /// Create a basic translation request targeting `target`.  Source language
    /// will be auto-detected.
    pub fn new(text: &str, target: Language) -> Self {
        Self {
            text: text.to_string(),
            source_language: None,
            target_language: target,
            formality: Formality::Default,
            preserve_formatting: false,
        }
    }

    /// Override the source language (builder style).
    pub fn with_source(mut self, source: Language) -> Self {
        self.source_language = Some(source);
        self
    }

    /// Set the desired formality level (builder style).
    pub fn with_formality(mut self, formality: Formality) -> Self {
        self.formality = formality;
        self
    }
}

// ── TranslationResult ─────────────────────────────────────────────────────────

/// The result of a translation.
#[derive(Debug, Clone)]
pub struct TranslationResult {
    pub translated_text: String,
    pub source_language: Language,
    pub target_language: Language,
    pub detection_result: Option<DetectionResult>,
    /// Identifier of the model used for translation.
    pub model_used: String,
}

// ── EnhancedTranslationPipeline ───────────────────────────────────────────────

/// Enhanced translation pipeline with language detection and per-language-pair
/// model overrides.
pub struct EnhancedTranslationPipeline {
    pub default_model: String,
    /// Per-language-pair model overrides keyed by `"<src>-<tgt>"` (e.g. `"en-fr"`).
    pub model_overrides: HashMap<String, String>,
    /// Whether to auto-detect source language when not specified.
    pub auto_detect: bool,
}

impl EnhancedTranslationPipeline {
    /// Create a pipeline with `default_model` used for all pairs without an override.
    pub fn new(default_model: &str) -> Self {
        Self {
            default_model: default_model.to_string(),
            model_overrides: HashMap::new(),
            auto_detect: true,
        }
    }

    /// Register a model override for a language pair (builder style).
    ///
    /// `lang_pair` must be in the format `"<src>-<tgt>"` (ISO 639-1 codes).
    pub fn with_model_override(mut self, lang_pair: &str, model: &str) -> Self {
        self.model_overrides.insert(lang_pair.to_string(), model.to_string());
        self
    }

    /// Return the model identifier to use for a (source, target) language pair.
    pub fn model_for_pair(&self, source: &Language, target: &Language) -> String {
        let key = format!("{}-{}", source.code(), target.code());
        self.model_overrides
            .get(&key)
            .cloned()
            .unwrap_or_else(|| self.default_model.clone())
    }

    /// Translate a single request.
    ///
    /// If `request.source_language` is `None` and `auto_detect` is `true`, the
    /// source language is detected from the text.
    ///
    /// The translation itself is a deterministic mock:
    /// words are reversed and the target language code is appended as a suffix.
    /// Real production use would call an actual translation model.
    pub fn translate(
        &self,
        request: TranslationRequest,
    ) -> Result<TranslationResult, TranslationError> {
        if request.text.trim().is_empty() {
            return Err(TranslationError::EmptyInput);
        }

        // Detect or use provided source language
        let (source_language, detection_result) = match request.source_language {
            Some(lang) => (lang, None),
            None => {
                if self.auto_detect {
                    let det = LanguageDetector::detect(&request.text);
                    let lang = det.language.clone();
                    (lang, Some(det))
                } else {
                    // Cannot proceed without a source language
                    return Err(TranslationError::DetectionFailed);
                }
            },
        };

        let model_used = self.model_for_pair(&source_language, &request.target_language);

        // Deterministic mock translation: reverse word order + append "[<tgt>]"
        let words: Vec<&str> = request.text.split_whitespace().collect();
        let reversed: Vec<&str> = words.iter().rev().cloned().collect();
        let translated_text = format!(
            "{} [{}]",
            reversed.join(" "),
            request.target_language.code()
        );

        Ok(TranslationResult {
            translated_text,
            source_language,
            target_language: request.target_language,
            detection_result,
            model_used,
        })
    }

    /// Translate a batch of `texts` into `target`, optionally with a known `source`.
    pub fn translate_batch(
        &self,
        texts: &[&str],
        target: Language,
        source: Option<Language>,
    ) -> Result<Vec<TranslationResult>, TranslationError> {
        texts
            .iter()
            .map(|text| {
                let request = TranslationRequest {
                    text: text.to_string(),
                    source_language: source.clone(),
                    target_language: target.clone(),
                    formality: Formality::Default,
                    preserve_formatting: false,
                };
                self.translate(request)
            })
            .collect()
    }

    /// Detect the language of `text`.
    pub fn detect_language(&self, text: &str) -> DetectionResult {
        LanguageDetector::detect(text)
    }

    /// Return all (source, target) language pairs for which a model override is
    /// registered.  Pairs are parsed from `"<src>-<tgt>"` keys.
    pub fn available_pairs(&self) -> Vec<(Language, Language)> {
        self.model_overrides
            .keys()
            .filter_map(|key| {
                let (src, tgt) = key.split_once('-')?;
                Some((Language::from_code(src), Language::from_code(tgt)))
            })
            .collect()
    }
}

// ── TranslationQualityEstimator ───────────────────────────────────────────────

/// Utility functions for estimating machine translation quality.
pub struct TranslationQualityEstimator;

impl TranslationQualityEstimator {
    /// Tokenise text into lowercase alphabetic words.
    fn tokenise(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphabetic())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Compute BLEU-1 (unigram precision) between `reference` and `hypothesis`.
    ///
    /// BLEU-1 = (number of hypothesis tokens appearing in reference) / |hypothesis|.
    /// Returns `0.0` for an empty hypothesis.
    pub fn compute_bleu(reference: &str, hypothesis: &str) -> f32 {
        let ref_tokens = Self::tokenise(reference);
        let hyp_tokens = Self::tokenise(hypothesis);
        if hyp_tokens.is_empty() {
            return 0.0;
        }
        let ref_counts = token_counts(&ref_tokens);
        let mut clipped = 0_usize;
        // Count each hypothesis token against the reference (clipped to ref count).
        let hyp_counts = token_counts(&hyp_tokens);
        for (tok, count) in &hyp_counts {
            let ref_count = ref_counts.get(tok).copied().unwrap_or(0);
            clipped += count.min(&ref_count);
        }
        let bp = Self::brevity_penalty(ref_tokens.len(), hyp_tokens.len());
        bp * clipped as f32 / hyp_tokens.len() as f32
    }

    /// Compute BLEU 1-4 tuple: `(bleu1, bleu2, bleu3, bleu4)`.
    ///
    /// Each component is the n-gram precision for n ∈ {1, 2, 3, 4}, individually
    /// brevity-penalised.
    pub fn compute_bleu4(reference: &str, hypothesis: &str) -> (f32, f32, f32, f32) {
        let b1 = Self::ngram_precision(reference, hypothesis, 1);
        let b2 = Self::ngram_precision(reference, hypothesis, 2);
        let b3 = Self::ngram_precision(reference, hypothesis, 3);
        let b4 = Self::ngram_precision(reference, hypothesis, 4);
        (b1, b2, b3, b4)
    }

    /// Brevity penalty: BP = min(1, exp(1 − |ref| / |hyp|)).
    ///
    /// Returns `1.0` when hypothesis is at least as long as the reference,
    /// and `0.0` when `hyp_len == 0`.
    pub fn brevity_penalty(reference_len: usize, hypothesis_len: usize) -> f32 {
        if hypothesis_len == 0 {
            return 0.0;
        }
        if hypothesis_len >= reference_len {
            return 1.0;
        }
        let ratio = reference_len as f32 / hypothesis_len as f32;
        (1.0 - ratio).exp()
    }

    /// Clipped n-gram precision between `reference` and `hypothesis`.
    ///
    /// Returns `0.0` when the hypothesis has fewer than `n` tokens.
    pub fn ngram_precision(reference: &str, hypothesis: &str, n: usize) -> f32 {
        if n == 0 {
            return 0.0;
        }
        let ref_tokens = Self::tokenise(reference);
        let hyp_tokens = Self::tokenise(hypothesis);
        if hyp_tokens.len() < n {
            return 0.0;
        }

        let ref_ngrams = extract_ngrams(&ref_tokens, n);
        let hyp_ngrams = extract_ngrams(&hyp_tokens, n);

        let ref_counts = token_counts(&ref_ngrams);
        let hyp_counts = token_counts(&hyp_ngrams);

        let mut clipped = 0_usize;
        for (gram, count) in &hyp_counts {
            let ref_count = ref_counts.get(gram).copied().unwrap_or(0);
            clipped += count.min(&ref_count);
        }

        let num_hyp_ngrams = hyp_tokens.len() - n + 1;
        let bp = Self::brevity_penalty(ref_tokens.len(), hyp_tokens.len());
        if num_hyp_ngrams == 0 {
            return 0.0;
        }
        bp * clipped as f32 / num_hyp_ngrams as f32
    }

    /// Token-level overlap F1 between `reference` and `hypothesis` (unordered).
    ///
    /// F1 = 2 × precision × recall / (precision + recall).
    /// Precision = |overlap| / |hyp|, Recall = |overlap| / |ref|.
    /// Returns `0.0` for empty inputs.
    pub fn token_overlap_f1(reference: &str, hypothesis: &str) -> f32 {
        let ref_tokens = Self::tokenise(reference);
        let hyp_tokens = Self::tokenise(hypothesis);
        if ref_tokens.is_empty() || hyp_tokens.is_empty() {
            return 0.0;
        }
        let ref_counts = token_counts(&ref_tokens);
        let hyp_counts = token_counts(&hyp_tokens);
        let mut overlap = 0_usize;
        for (tok, &h_count) in &hyp_counts {
            let r_count = ref_counts.get(tok).copied().unwrap_or(0);
            overlap += h_count.min(r_count);
        }
        let precision = overlap as f32 / hyp_tokens.len() as f32;
        let recall = overlap as f32 / ref_tokens.len() as f32;
        if precision + recall < f32::EPSILON {
            return 0.0;
        }
        2.0 * precision * recall / (precision + recall)
    }
}

// ── BackTranslationPair ───────────────────────────────────────────────────────

/// A round-trip translation pair for back-translation quality estimation.
#[derive(Debug, Clone)]
pub struct BackTranslationPair {
    /// Original source text.
    pub source: String,
    /// Translation of source into pivot/target language.
    pub translated: String,
    /// Translation of `translated` back to the source language.
    pub back_translated: String,
    /// Consistency score (BLEU-1 between `source` and `back_translated`).
    pub consistency: f32,
}

// ── BackTranslation ───────────────────────────────────────────────────────────

/// Back-translation utilities for quality estimation.
pub struct BackTranslation;

impl BackTranslation {
    /// Build a [`BackTranslationPair`] given the original source and a pivot translation.
    ///
    /// The `back_translated` field is set to `pivot` (simulating a round-trip for
    /// testing purposes), and `consistency` is computed via BLEU-1.
    ///
    /// In production, `pivot` would be passed through a second translation model.
    pub fn build_round_trip_pair(source: &str, pivot: &str) -> BackTranslationPair {
        // Simulate back-translation by reversing word order of pivot.
        let back_translated: String = pivot.split_whitespace().rev().collect::<Vec<_>>().join(" ");
        let consistency = Self::consistency_score(source, &back_translated);
        BackTranslationPair {
            source: source.to_string(),
            translated: pivot.to_string(),
            back_translated,
            consistency,
        }
    }

    /// Compute BLEU-1 between `original` and `round_trip` as a consistency score.
    pub fn consistency_score(original: &str, round_trip: &str) -> f32 {
        TranslationQualityEstimator::compute_bleu(original, round_trip)
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Count occurrences of each element in a slice of strings.
fn token_counts<T: Eq + std::hash::Hash + Clone>(tokens: &[T]) -> HashMap<T, usize> {
    let mut counts = HashMap::new();
    for tok in tokens {
        *counts.entry(tok.clone()).or_insert(0) += 1;
    }
    counts
}

/// Extract all n-gram strings from a token list.
fn extract_ngrams(tokens: &[String], n: usize) -> Vec<String> {
    if tokens.len() < n {
        return Vec::new();
    }
    tokens.windows(n).map(|w| w.join(" ")).collect()
}

// ── TranslationError ──────────────────────────────────────────────────────────

/// Errors that can occur in the enhanced translation pipeline.
#[derive(Debug)]
pub enum TranslationError {
    /// The requested language pair is not supported.
    UnsupportedLanguagePair { source: String, target: String },
    /// The input text was empty.
    EmptyInput,
    /// Automatic language detection failed.
    DetectionFailed,
}

impl std::fmt::Display for TranslationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranslationError::UnsupportedLanguagePair { source, target } => {
                write!(f, "unsupported language pair: {} → {}", source, target)
            },
            TranslationError::EmptyInput => write!(f, "input text must not be empty"),
            TranslationError::DetectionFailed => {
                write!(f, "automatic language detection failed")
            },
        }
    }
}

impl std::error::Error for TranslationError {}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Language tests ────────────────────────────────────────────────────────

    #[test]
    fn test_language_code() {
        assert_eq!(Language::English.code(), "en");
        assert_eq!(Language::French.code(), "fr");
        assert_eq!(Language::German.code(), "de");
        assert_eq!(Language::Spanish.code(), "es");
        assert_eq!(Language::Japanese.code(), "ja");
        assert_eq!(Language::Arabic.code(), "ar");
        assert_eq!(Language::Other("xx".to_string()).code(), "xx");
    }

    #[test]
    fn test_language_from_code() {
        assert_eq!(Language::from_code("en"), Language::English);
        assert_eq!(Language::from_code("fr"), Language::French);
        assert_eq!(Language::from_code("zh"), Language::Chinese);
        assert_eq!(
            Language::from_code("xyz"),
            Language::Other("xyz".to_string())
        );
    }

    #[test]
    fn test_language_name() {
        assert_eq!(Language::English.name(), "English");
        assert_eq!(Language::French.name(), "French");
        assert_eq!(Language::Japanese.name(), "Japanese");
        assert_eq!(Language::Other("xx".to_string()).name(), "Unknown");
    }

    #[test]
    fn test_language_script_latin() {
        assert_eq!(Language::English.script(), Script::Latin);
        assert_eq!(Language::French.script(), Script::Latin);
        assert_eq!(Language::German.script(), Script::Latin);
        assert_eq!(Language::Spanish.script(), Script::Latin);
        assert_eq!(Language::Polish.script(), Script::Latin);
    }

    #[test]
    fn test_language_script_cyrillic() {
        assert_eq!(Language::Russian.script(), Script::Cyrillic);
    }

    #[test]
    fn test_language_script_arabic() {
        assert_eq!(Language::Arabic.script(), Script::Arabic);
    }

    // ── LanguageDetector tests ────────────────────────────────────────────────

    #[test]
    fn test_detect_arabic_script() {
        // U+0623 is Arabic letter Alef with Hamza Above
        let arabic_text = "\u{0623}\u{0646}\u{0627} \u{0623}\u{062D}\u{0628} \u{0627}\u{0644}\u{0644}\u{063A}\u{0629}";
        let result = LanguageDetector::detect(arabic_text);
        assert_eq!(result.language, Language::Arabic);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_detect_cjk_chinese() {
        // Pure CJK characters (no hiragana/katakana) → Chinese
        let chinese_text = "\u{4E2D}\u{6587}\u{6D4B}\u{8BD5}\u{5185}\u{5BB9}";
        let result = LanguageDetector::detect(chinese_text);
        assert_eq!(result.language, Language::Chinese);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_detect_cjk_japanese() {
        // Mix of CJK + hiragana → Japanese
        let japanese_text = "\u{3053}\u{308C}\u{306F}\u{65E5}\u{672C}\u{8A9E}\u{3067}\u{3059}";
        let result = LanguageDetector::detect(japanese_text);
        assert_eq!(result.language, Language::Japanese);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_detect_latin_english() {
        let text = "the quick brown fox jumps over the lazy dog";
        let result = LanguageDetector::detect(text);
        assert_eq!(result.language, Language::English);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_detect_latin_french() {
        let text = "le chat est dans la maison et les enfants jouent";
        let result = LanguageDetector::detect(text);
        assert_eq!(result.language, Language::French);
        assert!(result.confidence > 0.3);
    }

    // ── TranslationRequest tests ──────────────────────────────────────────────

    #[test]
    fn test_translation_request_builder() {
        let req = TranslationRequest::new("Hello world", Language::French)
            .with_source(Language::English)
            .with_formality(Formality::Formal);

        assert_eq!(req.text, "Hello world");
        assert_eq!(req.source_language, Some(Language::English));
        assert_eq!(req.target_language, Language::French);
        assert_eq!(req.formality, Formality::Formal);
    }

    // ── EnhancedTranslationPipeline tests ─────────────────────────────────────

    #[test]
    fn test_pipeline_translate_basic() {
        let pipeline = EnhancedTranslationPipeline::new("default-model");
        let request =
            TranslationRequest::new("Hello world", Language::French).with_source(Language::English);
        let result = pipeline.translate(request).expect("translate");
        assert_eq!(result.source_language, Language::English);
        assert_eq!(result.target_language, Language::French);
        assert!(result.translated_text.contains("[fr]"));
        // Reversed word order mock: "world Hello [fr]"
        assert!(result.translated_text.starts_with("world Hello"));
    }

    #[test]
    fn test_pipeline_translate_batch() {
        let pipeline = EnhancedTranslationPipeline::new("default-model");
        let texts = vec!["Hello world", "Good morning", "How are you"];
        let results = pipeline
            .translate_batch(&texts, Language::German, Some(Language::English))
            .expect("batch translate");
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.target_language, Language::German);
            assert!(r.translated_text.contains("[de]"));
        }
    }

    #[test]
    fn test_pipeline_model_for_pair_override() {
        let pipeline = EnhancedTranslationPipeline::new("generic-model")
            .with_model_override("en-fr", "Helsinki-NLP/opus-mt-en-fr");
        assert_eq!(
            pipeline.model_for_pair(&Language::English, &Language::French),
            "Helsinki-NLP/opus-mt-en-fr"
        );
        // Pair without override falls back to default
        assert_eq!(
            pipeline.model_for_pair(&Language::English, &Language::German),
            "generic-model"
        );
    }

    #[test]
    fn test_pipeline_detect_language() {
        let pipeline = EnhancedTranslationPipeline::new("default-model");
        let result = pipeline.detect_language("the quick brown fox");
        assert_eq!(result.language, Language::English);
    }

    #[test]
    fn test_pipeline_available_pairs() {
        let pipeline = EnhancedTranslationPipeline::new("default-model")
            .with_model_override("en-fr", "model-a")
            .with_model_override("en-de", "model-b");
        let mut pairs = pipeline.available_pairs();
        pairs.sort_by(|a, b| a.0.code().cmp(b.0.code()).then(a.1.code().cmp(b.1.code())));
        assert_eq!(pairs.len(), 2);
        // Both pairs start with English
        assert!(pairs.iter().all(|(src, _)| *src == Language::English));
    }

    // ── TranslationError tests ────────────────────────────────────────────────

    #[test]
    fn test_translation_error_display() {
        let err_unsupported = TranslationError::UnsupportedLanguagePair {
            source: "xx".to_string(),
            target: "yy".to_string(),
        };
        let err_empty = TranslationError::EmptyInput;
        let err_detect = TranslationError::DetectionFailed;

        let msg_unsupported = format!("{}", err_unsupported);
        assert!(msg_unsupported.contains("xx"));
        assert!(msg_unsupported.contains("yy"));
        assert!(format!("{}", err_empty).contains("empty"));
        assert!(format!("{}", err_detect).contains("detection"));
    }

    // ── Auto-detect test ──────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_translate_auto_detect() {
        let pipeline = EnhancedTranslationPipeline::new("default-model");
        // No source specified → auto-detect from text
        let request = TranslationRequest::new(
            "the quick brown fox jumps over the lazy dog",
            Language::French,
        );
        let result = pipeline.translate(request).expect("translate");
        assert_eq!(result.target_language, Language::French);
        // Detection result should be populated
        assert!(result.detection_result.is_some());
    }

    #[test]
    fn test_translation_error_empty_input() {
        let pipeline = EnhancedTranslationPipeline::new("default-model");
        let request = TranslationRequest::new("", Language::French);
        let err = pipeline.translate(request).unwrap_err();
        matches!(err, TranslationError::EmptyInput);
    }

    // ── TranslationQualityEstimator::compute_bleu ─────────────────────────────

    #[test]
    fn test_compute_bleu_exact_match() {
        // Perfect match → BLEU-1 should be 1.0 (BP=1.0 because hyp len = ref len).
        let score = TranslationQualityEstimator::compute_bleu("hello world", "hello world");
        assert!((score - 1.0).abs() < 1e-5, "exact match BLEU-1={score}");
    }

    #[test]
    fn test_compute_bleu_no_overlap() {
        let score = TranslationQualityEstimator::compute_bleu("hello world", "foo bar baz");
        assert!((score).abs() < 1e-5, "no overlap → BLEU-1={score}");
    }

    #[test]
    fn test_compute_bleu_partial_overlap() {
        // reference: "the cat sat", hypothesis: "the cat ran"
        // overlap tokens: "the", "cat" → 2/3 precision, BP=1 (same length)
        let score = TranslationQualityEstimator::compute_bleu("the cat sat", "the cat ran");
        assert!(score > 0.0 && score < 1.0, "partial overlap BLEU-1={score}");
    }

    #[test]
    fn test_compute_bleu_empty_hypothesis() {
        let score = TranslationQualityEstimator::compute_bleu("hello world", "");
        assert_eq!(score, 0.0, "empty hypothesis → BLEU=0.0");
    }

    // ── TranslationQualityEstimator::brevity_penalty ─────────────────────────

    #[test]
    fn test_brevity_penalty_longer_hyp() {
        // hypothesis longer than reference → penalty = 1.0
        let bp = TranslationQualityEstimator::brevity_penalty(3, 5);
        assert!((bp - 1.0).abs() < 1e-5, "BP={bp}");
    }

    #[test]
    fn test_brevity_penalty_equal_length() {
        let bp = TranslationQualityEstimator::brevity_penalty(4, 4);
        assert!((bp - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_brevity_penalty_short_hyp() {
        // hypothesis half as long → penalty < 1.
        let bp = TranslationQualityEstimator::brevity_penalty(10, 5);
        assert!(bp < 1.0 && bp > 0.0, "BP for short hyp={bp}");
        // exp(1 - 10/5) = exp(-1) ≈ 0.3679
        assert!((bp - (-1.0_f32).exp()).abs() < 1e-4, "BP={bp}");
    }

    #[test]
    fn test_brevity_penalty_zero_hyp() {
        assert_eq!(TranslationQualityEstimator::brevity_penalty(5, 0), 0.0);
    }

    // ── TranslationQualityEstimator::ngram_precision ─────────────────────────

    #[test]
    fn test_ngram_precision_unigram_exact() {
        let p = TranslationQualityEstimator::ngram_precision(
            "the cat sat on the mat",
            "the cat sat on the mat",
            1,
        );
        assert!(
            (p - 1.0).abs() < 1e-5,
            "unigram precision for exact match={p}"
        );
    }

    #[test]
    fn test_ngram_precision_bigram_partial() {
        // Reference: "the cat sat", hypothesis: "the cat ran"
        // Bigrams in hypothesis: ["the cat", "cat ran"] → only "the cat" matches → 1/2
        let p = TranslationQualityEstimator::ngram_precision("the cat sat", "the cat ran", 2);
        assert!(p > 0.0 && p < 1.0, "bigram partial precision={p}");
    }

    #[test]
    fn test_ngram_precision_too_short() {
        // Hypothesis has 1 token, asking for 2-gram.
        let p = TranslationQualityEstimator::ngram_precision("hello world", "hello", 2);
        assert_eq!(p, 0.0, "too short for n-gram → 0.0");
    }

    #[test]
    fn test_ngram_precision_zero_n() {
        let p = TranslationQualityEstimator::ngram_precision("hello", "hello", 0);
        assert_eq!(p, 0.0);
    }

    // ── TranslationQualityEstimator::token_overlap_f1 ────────────────────────

    #[test]
    fn test_token_overlap_f1_perfect() {
        let f1 = TranslationQualityEstimator::token_overlap_f1("hello world", "hello world");
        assert!((f1 - 1.0).abs() < 1e-5, "F1 for perfect match={f1}");
    }

    #[test]
    fn test_token_overlap_f1_no_overlap() {
        let f1 = TranslationQualityEstimator::token_overlap_f1("hello world", "foo bar");
        assert_eq!(f1, 0.0, "no overlap → F1=0.0");
    }

    #[test]
    fn test_token_overlap_f1_partial() {
        let f1 = TranslationQualityEstimator::token_overlap_f1("cat dog bird", "cat dog fish");
        assert!(f1 > 0.0 && f1 < 1.0, "partial F1={f1}");
    }

    #[test]
    fn test_token_overlap_f1_empty_inputs() {
        assert_eq!(
            TranslationQualityEstimator::token_overlap_f1("", "hello"),
            0.0
        );
        assert_eq!(
            TranslationQualityEstimator::token_overlap_f1("hello", ""),
            0.0
        );
    }

    // ── BackTranslation ───────────────────────────────────────────────────────

    #[test]
    fn test_back_translation_pair_fields() {
        let pair = BackTranslation::build_round_trip_pair("hello world", "bonjour monde");
        assert_eq!(pair.source, "hello world");
        assert_eq!(pair.translated, "bonjour monde");
        assert!(
            !pair.back_translated.is_empty(),
            "back_translated should not be empty"
        );
    }

    #[test]
    fn test_back_translation_consistency_score_range() {
        let pair = BackTranslation::build_round_trip_pair("the cat sat", "le chat assis");
        assert!(
            (0.0..=1.0).contains(&pair.consistency),
            "consistency score out of range: {}",
            pair.consistency
        );
    }

    #[test]
    fn test_back_translation_perfect_round_trip() {
        // If the translated text (reversed back) matches source words well, score is high.
        // "hello world" → pivot = "world hello" → back = "hello world" → exact match
        let pair = BackTranslation::build_round_trip_pair("hello world", "world hello");
        // back_translated = "hello world" → BLEU-1 with source should be 1.0
        assert!(
            pair.consistency > 0.9,
            "near-perfect round trip consistency={}",
            pair.consistency
        );
    }

    #[test]
    fn test_consistency_score_direct() {
        let score = BackTranslation::consistency_score("hello world", "hello world");
        assert!(
            (score - 1.0).abs() < 1e-5,
            "consistency_score exact match={score}"
        );
    }

    // ── compute_bleu4 ─────────────────────────────────────────────────────────

    #[test]
    fn test_compute_bleu4_perfect_match() {
        let (b1, b2, b3, b4) = TranslationQualityEstimator::compute_bleu4(
            "the quick brown fox jumps",
            "the quick brown fox jumps",
        );
        assert!((b1 - 1.0).abs() < 1e-4, "BLEU-1={b1}");
        assert!((b2 - 1.0).abs() < 1e-4, "BLEU-2={b2}");
        assert!((b3 - 1.0).abs() < 1e-4, "BLEU-3={b3}");
        assert!((b4 - 1.0).abs() < 1e-4, "BLEU-4={b4}");
    }

    #[test]
    fn test_compute_bleu4_decreasing_with_n() {
        // For a mostly-matching but imperfect hypothesis, precision should generally
        // decrease as n increases (higher-order n-grams are harder to match).
        let (b1, b2, b3, b4) = TranslationQualityEstimator::compute_bleu4(
            "the quick brown fox",
            "the quick brown cat",
        );
        // b1 >= b2 >= b3 >= b4 is typical but not strictly guaranteed; at minimum
        // they should all be in [0, 1].
        for s in [b1, b2, b3, b4] {
            assert!((0.0..=1.0 + 1e-5).contains(&s), "BLEU-n out of range: {s}");
        }
    }
}
