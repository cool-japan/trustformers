use once_cell::sync::Lazy;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

pub trait Normalizer {
    fn normalize(&self, text: &str) -> String;
}

static WHITESPACE_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").expect("valid regex"));
static PUNCTUATION_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^\w\s]").expect("valid regex"));
static ACCENT_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]")
        .expect("valid regex")
});

pub struct NFCNormalizer;

impl Normalizer for NFCNormalizer {
    fn normalize(&self, text: &str) -> String {
        text.nfc().collect()
    }
}

pub struct NFDNormalizer;

impl Normalizer for NFDNormalizer {
    fn normalize(&self, text: &str) -> String {
        text.nfd().collect()
    }
}

pub struct LowercaseNormalizer;

impl Normalizer for LowercaseNormalizer {
    fn normalize(&self, text: &str) -> String {
        text.to_lowercase()
    }
}

pub struct ChainedNormalizer {
    normalizers: Vec<Box<dyn Normalizer>>,
}

impl ChainedNormalizer {
    pub fn new(normalizers: Vec<Box<dyn Normalizer>>) -> Self {
        Self { normalizers }
    }
}

impl Normalizer for ChainedNormalizer {
    fn normalize(&self, text: &str) -> String {
        self.normalizers.iter().fold(text.to_string(), |acc, normalizer| {
            normalizer.normalize(&acc)
        })
    }
}

pub struct WhitespaceNormalizer;

impl Normalizer for WhitespaceNormalizer {
    fn normalize(&self, text: &str) -> String {
        WHITESPACE_REGEX.replace_all(text.trim(), " ").to_string()
    }
}

pub struct AccentRemovalNormalizer;

impl Normalizer for AccentRemovalNormalizer {
    fn normalize(&self, text: &str) -> String {
        let nfd_text: String = text.nfd().collect();
        ACCENT_REGEX.replace_all(&nfd_text, "").to_string()
    }
}

pub struct PunctuationRemovalNormalizer;

impl Normalizer for PunctuationRemovalNormalizer {
    fn normalize(&self, text: &str) -> String {
        PUNCTUATION_REGEX.replace_all(text, " ").to_string()
    }
}

pub struct DigitNormalizer {
    replacement: String,
}

impl DigitNormalizer {
    pub fn new(replacement: String) -> Self {
        Self { replacement }
    }
}

impl Normalizer for DigitNormalizer {
    fn normalize(&self, text: &str) -> String {
        text.chars()
            .map(|c| if c.is_numeric() { self.replacement.clone() } else { c.to_string() })
            .collect::<String>()
    }
}

pub struct CaseNormalizer {
    uppercase: bool,
}

impl CaseNormalizer {
    pub fn uppercase() -> Self {
        Self { uppercase: true }
    }

    pub fn lowercase() -> Self {
        Self { uppercase: false }
    }
}

impl Normalizer for CaseNormalizer {
    fn normalize(&self, text: &str) -> String {
        if self.uppercase {
            text.to_uppercase()
        } else {
            text.to_lowercase()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_normalizer() {
        let normalizer = WhitespaceNormalizer;
        assert_eq!(normalizer.normalize("  hello   world  "), "hello world");
    }

    #[test]
    fn test_accent_removal() {
        let normalizer = AccentRemovalNormalizer;
        assert_eq!(normalizer.normalize("café"), "cafe");
    }

    #[test]
    fn test_chained_normalizer() {
        let normalizers: Vec<Box<dyn Normalizer>> = vec![
            Box::new(CaseNormalizer::lowercase()),
            Box::new(AccentRemovalNormalizer),
            Box::new(WhitespaceNormalizer),
        ];
        let chained = ChainedNormalizer::new(normalizers);
        assert_eq!(chained.normalize("  CAFÉ   WORLD  "), "cafe world");
    }
}
