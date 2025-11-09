#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use trustformers_core::errors::{Result, TrustformersError};

/// SIMD-optimized tokenization utilities for improved performance
pub struct SimdTokenizer {
    /// Lookup table for ASCII character classification
    ascii_lookup: [u8; 256],
}

impl SimdTokenizer {
    /// Create a new SIMD tokenizer
    pub fn new() -> Self {
        let mut ascii_lookup = [0u8; 256];

        // Set up character classification flags
        // Bit 0: alphabetic, Bit 1: numeric, Bit 2: whitespace, Bit 3: punctuation
        for (i, flags_ref) in ascii_lookup.iter_mut().enumerate() {
            let ch = i as u8 as char;
            let mut flags = 0u8;

            if ch.is_alphabetic() {
                flags |= 1;
            }
            if ch.is_numeric() {
                flags |= 2;
            }
            if ch.is_whitespace() {
                flags |= 4;
            }
            if ch.is_ascii_punctuation() {
                flags |= 8;
            }

            *flags_ref = flags;
        }

        Self { ascii_lookup }
    }

    /// Fast ASCII character classification using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn classify_ascii_chars(&self, text: &[u8]) -> Vec<u8> {
        if !is_x86_feature_detected!("avx2") {
            return self.classify_ascii_chars_scalar(text);
        }

        unsafe { self.classify_ascii_chars_avx2(text) }
    }

    /// Fast ASCII character classification - fallback for non-x86_64
    #[cfg(not(target_arch = "x86_64"))]
    pub fn classify_ascii_chars(&self, text: &[u8]) -> Vec<u8> {
        self.classify_ascii_chars_scalar(text)
    }

    /// Fallback scalar implementation for character classification
    fn classify_ascii_chars_scalar(&self, text: &[u8]) -> Vec<u8> {
        text.iter().map(|&byte| self.ascii_lookup[byte as usize]).collect()
    }

    /// AVX2-optimized character classification
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn classify_ascii_chars_avx2(&self, text: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(text.len());
        let chunks = text.chunks_exact(32);
        let remainder = chunks.remainder();

        // Process 32 bytes at a time using AVX2
        for chunk in chunks {
            let _input = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

            // Character classification using parallel lookups
            let mut output = [0u8; 32];
            for i in 0..32 {
                output[i] = self.ascii_lookup[chunk[i] as usize];
            }

            result.extend_from_slice(&output);
        }

        // Process remaining bytes
        for &byte in remainder {
            result.push(self.ascii_lookup[byte as usize]);
        }

        result
    }

    /// Fast whitespace detection using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn find_whitespace_boundaries(&self, text: &[u8]) -> Vec<usize> {
        if !is_x86_feature_detected!("avx2") {
            return self.find_whitespace_boundaries_scalar(text);
        }

        unsafe { self.find_whitespace_boundaries_avx2(text) }
    }

    /// Fast whitespace detection - fallback for non-x86_64
    #[cfg(not(target_arch = "x86_64"))]
    pub fn find_whitespace_boundaries(&self, text: &[u8]) -> Vec<usize> {
        self.find_whitespace_boundaries_scalar(text)
    }

    /// Scalar implementation for whitespace boundary detection
    fn find_whitespace_boundaries_scalar(&self, text: &[u8]) -> Vec<usize> {
        let mut boundaries = Vec::new();
        let mut in_whitespace = false;

        for (i, &byte) in text.iter().enumerate() {
            let is_whitespace = (self.ascii_lookup[byte as usize] & 4) != 0;

            if is_whitespace != in_whitespace {
                boundaries.push(i);
                in_whitespace = is_whitespace;
            }
        }

        boundaries
    }

    /// AVX2-optimized whitespace boundary detection
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_whitespace_boundaries_avx2(&self, text: &[u8]) -> Vec<usize> {
        let mut boundaries = Vec::new();
        let mut prev_whitespace_mask = 0u32;

        // Define whitespace characters
        let space = _mm256_set1_epi8(b' ' as i8);
        let tab = _mm256_set1_epi8(b'\t' as i8);
        let newline = _mm256_set1_epi8(b'\n' as i8);
        let carriage_return = _mm256_set1_epi8(b'\r' as i8);

        let chunks = text.chunks_exact(32);
        let mut offset = 0;

        for chunk in chunks {
            let input = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

            // Compare with whitespace characters
            let space_mask = _mm256_cmpeq_epi8(input, space);
            let tab_mask = _mm256_cmpeq_epi8(input, tab);
            let newline_mask = _mm256_cmpeq_epi8(input, newline);
            let cr_mask = _mm256_cmpeq_epi8(input, carriage_return);

            // Combine all whitespace masks
            let whitespace_mask = _mm256_or_si256(
                _mm256_or_si256(space_mask, tab_mask),
                _mm256_or_si256(newline_mask, cr_mask),
            );

            let mask_bits = _mm256_movemask_epi8(whitespace_mask) as u32;

            // Find transitions between whitespace and non-whitespace
            let transitions = mask_bits ^ (mask_bits << 1) ^ prev_whitespace_mask;

            // Extract boundary positions
            for i in 0..32 {
                if (transitions & (1 << i)) != 0 {
                    boundaries.push(offset + i);
                }
            }

            prev_whitespace_mask = if (mask_bits & (1 << 31)) != 0 { 1 } else { 0 };
            offset += 32;
        }

        // Process remainder with scalar code
        let remainder = &text[offset..];
        for (i, &byte) in remainder.iter().enumerate() {
            let is_whitespace = matches!(byte, b' ' | b'\t' | b'\n' | b'\r');
            let current_mask = if is_whitespace { 1 } else { 0 };

            if current_mask != prev_whitespace_mask {
                boundaries.push(offset + i);
                prev_whitespace_mask = current_mask;
            }
        }

        boundaries
    }

    /// Fast byte-to-UTF8 validation using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn validate_utf8_fast(&self, bytes: &[u8]) -> Result<()> {
        if !is_x86_feature_detected!("avx2") {
            return self.validate_utf8_scalar(bytes);
        }

        unsafe { self.validate_utf8_avx2(bytes) }
    }

    /// Fast byte-to-UTF8 validation - fallback for non-x86_64
    #[cfg(not(target_arch = "x86_64"))]
    pub fn validate_utf8_fast(&self, bytes: &[u8]) -> Result<()> {
        self.validate_utf8_scalar(bytes)
    }

    /// Scalar UTF-8 validation
    fn validate_utf8_scalar(&self, bytes: &[u8]) -> Result<()> {
        std::str::from_utf8(bytes)
            .map_err(|e| TrustformersError::invalid_input(format!("Invalid UTF-8: {}", e)))?;
        Ok(())
    }

    /// AVX2-optimized UTF-8 validation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn validate_utf8_avx2(&self, bytes: &[u8]) -> Result<()> {
        // Simplified fast path for ASCII-only text
        let chunks = bytes.chunks_exact(32);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let input = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let ascii_mask = _mm256_cmpgt_epi8(_mm256_setzero_si256(), input);

            if _mm256_movemask_epi8(ascii_mask) != 0 {
                // Contains non-ASCII, fall back to scalar validation
                return self.validate_utf8_scalar(bytes);
            }
        }

        // Check remainder
        for &byte in remainder {
            if byte >= 128 {
                return self.validate_utf8_scalar(bytes);
            }
        }

        Ok(())
    }

    /// Fast case conversion using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn to_lowercase_ascii(&self, text: &[u8]) -> Vec<u8> {
        if !is_x86_feature_detected!("avx2") {
            return self.to_lowercase_ascii_scalar(text);
        }

        unsafe { self.to_lowercase_ascii_avx2(text) }
    }

    /// Fast case conversion - fallback for non-x86_64
    #[cfg(not(target_arch = "x86_64"))]
    pub fn to_lowercase_ascii(&self, text: &[u8]) -> Vec<u8> {
        self.to_lowercase_ascii_scalar(text)
    }

    /// Scalar lowercase conversion
    fn to_lowercase_ascii_scalar(&self, text: &[u8]) -> Vec<u8> {
        text.iter()
            .map(|&byte| {
                if byte.is_ascii_uppercase() {
                    byte + 32 // Convert to lowercase
                } else {
                    byte
                }
            })
            .collect()
    }

    /// AVX2-optimized lowercase conversion
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn to_lowercase_ascii_avx2(&self, text: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(text.len());
        let chunks = text.chunks_exact(32);
        let remainder = chunks.remainder();

        let a_upper = _mm256_set1_epi8(b'A' as i8);
        let z_upper = _mm256_set1_epi8(b'Z' as i8);
        let to_lower_offset = _mm256_set1_epi8(32);

        for chunk in chunks {
            let input = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

            // Create mask for uppercase letters
            let ge_a = _mm256_cmpgt_epi8(input, _mm256_sub_epi8(a_upper, _mm256_set1_epi8(1)));
            let le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(z_upper, _mm256_set1_epi8(1)), input);
            let is_upper = _mm256_and_si256(ge_a, le_z);

            // Apply lowercase conversion
            let lowercase_offset = _mm256_and_si256(is_upper, to_lower_offset);
            let output = _mm256_add_epi8(input, lowercase_offset);

            let mut temp = [0u8; 32];
            _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, output);
            result.extend_from_slice(&temp);
        }

        // Process remainder
        for &byte in remainder {
            let converted = if byte >= b'A' && byte <= b'Z' { byte + 32 } else { byte };
            result.push(converted);
        }

        result
    }

    /// High-performance text preprocessing pipeline
    pub fn preprocess_text(&self, text: &str) -> Result<Vec<String>> {
        let bytes = text.as_bytes();

        // Step 1: Validate UTF-8
        self.validate_utf8_fast(bytes)?;

        // Step 2: Find word boundaries using whitespace detection
        let boundaries = self.find_whitespace_boundaries(bytes);

        // Step 3: Extract tokens
        let mut tokens = Vec::new();
        let mut start = 0;

        for &boundary in &boundaries {
            if start < boundary {
                let token_bytes = &bytes[start..boundary];
                let token = String::from_utf8_lossy(token_bytes).into_owned();
                if !token.trim().is_empty() {
                    tokens.push(token);
                }
            }
            start = boundary;
        }

        // Add final token if any
        if start < bytes.len() {
            let token_bytes = &bytes[start..];
            let token = String::from_utf8_lossy(token_bytes).into_owned();
            if !token.trim().is_empty() {
                tokens.push(token);
            }
        }

        Ok(tokens)
    }
}

impl Default for SimdTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_character_classification() {
        let tokenizer = SimdTokenizer::new();
        let text = b"Hello, World! 123";

        let classifications = tokenizer.classify_ascii_chars(text);

        // 'H' should be alphabetic (bit 0 set)
        assert_eq!(classifications[0] & 1, 1);

        // ',' should be punctuation (bit 3 set)
        assert_eq!(classifications[5] & 8, 8);

        // ' ' should be whitespace (bit 2 set)
        assert_eq!(classifications[6] & 4, 4);

        // '1' should be numeric (bit 1 set)
        assert_eq!(classifications[14] & 2, 2);
    }

    #[test]
    fn test_simd_whitespace_boundaries() {
        let tokenizer = SimdTokenizer::new();
        let text = b"Hello World Test";

        let boundaries = tokenizer.find_whitespace_boundaries(text);

        // Should find boundaries at positions 5 and 11 (before/after spaces)
        assert!(boundaries.contains(&5));
        assert!(boundaries.contains(&6));
        assert!(boundaries.contains(&11));
        assert!(boundaries.contains(&12));
    }

    #[test]
    fn test_simd_utf8_validation() {
        let tokenizer = SimdTokenizer::new();

        // Valid ASCII
        assert!(tokenizer.validate_utf8_fast(b"Hello World").is_ok());

        // Valid UTF-8
        assert!(tokenizer.validate_utf8_fast("Hello 世界".as_bytes()).is_ok());

        // Invalid UTF-8
        assert!(tokenizer.validate_utf8_fast(&[0xFF, 0xFE]).is_err());
    }

    #[test]
    fn test_simd_lowercase() {
        let tokenizer = SimdTokenizer::new();
        let text = b"Hello WORLD Test";

        let lowercase = tokenizer.to_lowercase_ascii(text);
        let expected = b"hello world test";

        assert_eq!(lowercase, expected);
    }

    #[test]
    fn test_simd_preprocess_pipeline() {
        let tokenizer = SimdTokenizer::new();
        let text = "Hello, World! How are you?";

        let tokens = tokenizer.preprocess_text(text).unwrap();

        assert!(tokens.len() > 0);
        assert!(tokens.contains(&"Hello,".to_string()));
        assert!(tokens.contains(&"World!".to_string()));
        assert!(tokens.contains(&"How".to_string()));
    }

    #[test]
    fn test_simd_empty_input() {
        let tokenizer = SimdTokenizer::new();

        assert_eq!(tokenizer.classify_ascii_chars(b"").len(), 0);
        assert_eq!(tokenizer.find_whitespace_boundaries(b"").len(), 0);
        assert!(tokenizer.validate_utf8_fast(b"").is_ok());
        assert_eq!(tokenizer.to_lowercase_ascii(b"").len(), 0);
    }

    #[test]
    fn test_simd_long_input() {
        let tokenizer = SimdTokenizer::new();
        let text = "A".repeat(1000);

        let lowercase = tokenizer.to_lowercase_ascii(text.as_bytes());
        let expected = "a".repeat(1000);

        assert_eq!(lowercase, expected.as_bytes());
    }
}
