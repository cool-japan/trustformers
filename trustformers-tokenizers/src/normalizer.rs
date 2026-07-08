use once_cell::sync::Lazy;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

pub trait Normalizer {
    fn normalize(&self, text: &str) -> String;
}

// reason: each pattern below is a compile-time constant, so `Regex::new` cannot
// fail at runtime; a `static` initializer has no fallible channel to propagate.
static WHITESPACE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\s+").expect("built-in whitespace regex must compile"));
static PUNCTUATION_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[^\w\s]").expect("built-in punctuation regex must compile"));
static ACCENT_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]")
        .expect("built-in accent regex must compile")
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

pub struct NFKCNormalizer;

impl Normalizer for NFKCNormalizer {
    fn normalize(&self, text: &str) -> String {
        text.nfkc().collect()
    }
}

pub struct NFKDNormalizer;

impl Normalizer for NFKDNormalizer {
    fn normalize(&self, text: &str) -> String {
        text.nfkd().collect()
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

    // U+00B2 SUPERSCRIPT TWO has a *compatibility* decomposition to "2" but no
    // *canonical* decomposition, so NFKC/NFKD fold it away while NFC/NFD must
    // leave it untouched. This proves the K-variants are doing genuinely
    // different work rather than being aliases of the plain variants.
    #[test]
    fn test_nfkc_normalizer_superscript_two() {
        let normalizer = NFKCNormalizer;
        assert_eq!(normalizer.normalize("\u{00b2}"), "2");
    }

    #[test]
    fn test_nfkd_normalizer_superscript_two() {
        let normalizer = NFKDNormalizer;
        assert_eq!(normalizer.normalize("\u{00b2}"), "2");
    }

    #[test]
    fn test_nfc_and_nfd_leave_superscript_two_unchanged() {
        assert_eq!(NFCNormalizer.normalize("\u{00b2}"), "\u{00b2}");
        assert_eq!(NFDNormalizer.normalize("\u{00b2}"), "\u{00b2}");
    }

    // U+FB01 LATIN SMALL LIGATURE FI is the same story: only a compatibility
    // decomposition to "fi", none canonical, so NFC/NFD keep the ligature intact.
    #[test]
    fn test_nfkc_normalizer_fi_ligature() {
        let normalizer = NFKCNormalizer;
        assert_eq!(normalizer.normalize("\u{fb01}"), "fi");
    }

    #[test]
    fn test_nfkd_normalizer_fi_ligature() {
        let normalizer = NFKDNormalizer;
        assert_eq!(normalizer.normalize("\u{fb01}"), "fi");
    }

    #[test]
    fn test_nfc_and_nfd_leave_fi_ligature_unchanged() {
        assert_eq!(NFCNormalizer.normalize("\u{fb01}"), "\u{fb01}");
        assert_eq!(NFDNormalizer.normalize("\u{fb01}"), "\u{fb01}");
    }
}
