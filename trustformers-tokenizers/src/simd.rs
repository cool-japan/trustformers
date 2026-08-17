#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use trustformers_core::errors::{Result, TrustformersError};

/// SIMD-optimized tokenization utilities for improved performance.
///
/// Every SIMD path here (`x86_64` AVX2, `aarch64` NEON) must produce
/// exactly the same output as [`Self`]'s scalar fallback for every input
/// byte -- the `*_parity_with_scalar` tests below check this directly,
/// including bytes `0x00..=0xFF` and every ASCII whitespace character, so
/// a SIMD path that silently disagrees with the scalar reference (e.g. by
/// missing a whitespace character the scalar path recognizes) is a real
/// correctness bug, not just a missed optimization.
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

    /// Fast ASCII character classification using SIMD (NEON)
    #[cfg(target_arch = "aarch64")]
    pub fn classify_ascii_chars(&self, text: &[u8]) -> Vec<u8> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return self.classify_ascii_chars_scalar(text);
        }

        unsafe { self.classify_ascii_chars_neon(text) }
    }

    /// Fast ASCII character classification - fallback for platforms with
    /// neither an AVX2 (`x86_64`) nor a NEON (`aarch64`) implementation.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn classify_ascii_chars(&self, text: &[u8]) -> Vec<u8> {
        self.classify_ascii_chars_scalar(text)
    }

    /// Fallback scalar implementation for character classification
    fn classify_ascii_chars_scalar(&self, text: &[u8]) -> Vec<u8> {
        text.iter().map(|&byte| self.ascii_lookup[byte as usize]).collect()
    }

    /// AVX2-optimized character classification.
    ///
    /// ASCII bytes (`< 0x80`) are classified with real vectorized range
    /// comparisons matching [`Self::ascii_lookup`]'s ASCII entries exactly
    /// (verified against the scalar table by
    /// `test_classify_avx2_parity_with_scalar` on `x86_64`). A chunk
    /// containing any non-ASCII byte falls back to the exact scalar table
    /// lookup for that chunk -- `char::is_alphabetic`/`is_numeric` on the
    /// Latin-1 codepoints those bytes map to are not expressible as a
    /// handful of contiguous-range comparisons.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn classify_ascii_chars_avx2(&self, text: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(text.len());
        let chunks = text.chunks_exact(32);
        let remainder = chunks.remainder();

        let zero = _mm256_setzero_si256();

        let upper_a = _mm256_set1_epi8((b'A' - 1) as i8);
        let upper_z = _mm256_set1_epi8((b'Z' + 1) as i8);
        let lower_a = _mm256_set1_epi8((b'a' - 1) as i8);
        let lower_z = _mm256_set1_epi8((b'z' + 1) as i8);
        let digit_0 = _mm256_set1_epi8((b'0' - 1) as i8);
        let digit_9 = _mm256_set1_epi8((b'9' + 1) as i8);
        let space = _mm256_set1_epi8(b' ' as i8);
        let tab = _mm256_set1_epi8(b'\t' as i8);
        let lf = _mm256_set1_epi8(b'\n' as i8);
        let vt = _mm256_set1_epi8(0x0Bi8);
        let ff = _mm256_set1_epi8(0x0Ci8);
        let cr = _mm256_set1_epi8(b'\r' as i8);

        let punct_lo1 = _mm256_set1_epi8((0x21 - 1) as i8);
        let punct_hi1 = _mm256_set1_epi8((0x2F + 1) as i8);
        let punct_lo2 = _mm256_set1_epi8((0x3A - 1) as i8);
        let punct_hi2 = _mm256_set1_epi8((0x40 + 1) as i8);
        let punct_lo3 = _mm256_set1_epi8((0x5B - 1) as i8);
        let punct_hi3 = _mm256_set1_epi8((0x60 + 1) as i8);
        let punct_lo4 = _mm256_set1_epi8((0x7B - 1) as i8);
        let punct_hi4 = _mm256_set1_epi8(0x7Ei8 + 1);

        for chunk in chunks {
            let input = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

            // Any byte with the high bit set reads as negative in this
            // signed-byte comparison, so `zero > input` flags it.
            if _mm256_movemask_epi8(_mm256_cmpgt_epi8(zero, input)) != 0 {
                let mut output = [0u8; 32];
                for (i, out) in output.iter_mut().enumerate() {
                    *out = self.ascii_lookup[chunk[i] as usize];
                }
                result.extend_from_slice(&output);
                continue;
            }

            let is_upper = _mm256_and_si256(
                _mm256_cmpgt_epi8(input, upper_a),
                _mm256_cmpgt_epi8(upper_z, input),
            );
            let is_lower = _mm256_and_si256(
                _mm256_cmpgt_epi8(input, lower_a),
                _mm256_cmpgt_epi8(lower_z, input),
            );
            let is_alpha = _mm256_or_si256(is_upper, is_lower);

            let is_digit = _mm256_and_si256(
                _mm256_cmpgt_epi8(input, digit_0),
                _mm256_cmpgt_epi8(digit_9, input),
            );

            let is_ws = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_cmpeq_epi8(input, space),
                    _mm256_cmpeq_epi8(input, tab),
                ),
                _mm256_or_si256(
                    _mm256_or_si256(_mm256_cmpeq_epi8(input, lf), _mm256_cmpeq_epi8(input, cr)),
                    _mm256_or_si256(_mm256_cmpeq_epi8(input, vt), _mm256_cmpeq_epi8(input, ff)),
                ),
            );

            let punct1 = _mm256_and_si256(
                _mm256_cmpgt_epi8(input, punct_lo1),
                _mm256_cmpgt_epi8(punct_hi1, input),
            );
            let punct2 = _mm256_and_si256(
                _mm256_cmpgt_epi8(input, punct_lo2),
                _mm256_cmpgt_epi8(punct_hi2, input),
            );
            let punct3 = _mm256_and_si256(
                _mm256_cmpgt_epi8(input, punct_lo3),
                _mm256_cmpgt_epi8(punct_hi3, input),
            );
            let punct4 = _mm256_and_si256(
                _mm256_cmpgt_epi8(input, punct_lo4),
                _mm256_cmpgt_epi8(punct_hi4, input),
            );
            let is_punct = _mm256_or_si256(
                _mm256_or_si256(punct1, punct2),
                _mm256_or_si256(punct3, punct4),
            );

            let alpha_bits = _mm256_and_si256(is_alpha, _mm256_set1_epi8(1));
            let digit_bits = _mm256_and_si256(is_digit, _mm256_set1_epi8(2));
            let ws_bits = _mm256_and_si256(is_ws, _mm256_set1_epi8(4));
            let punct_bits = _mm256_and_si256(is_punct, _mm256_set1_epi8(8));
            let flags = _mm256_or_si256(
                _mm256_or_si256(alpha_bits, digit_bits),
                _mm256_or_si256(ws_bits, punct_bits),
            );

            let mut output = [0u8; 32];
            _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, flags);
            result.extend_from_slice(&output);
        }

        for &byte in remainder {
            result.push(self.ascii_lookup[byte as usize]);
        }

        result
    }

    /// NEON-optimized character classification. Mirrors
    /// [`Self::classify_ascii_chars_avx2`]'s logic (real vectorized ASCII
    /// range comparisons, scalar table fallback for any chunk containing a
    /// non-ASCII byte) over 16-byte NEON registers instead of 32-byte AVX2
    /// ones.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn classify_ascii_chars_neon(&self, text: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(text.len());
        let chunks = text.chunks_exact(16);
        let remainder = chunks.remainder();

        let upper_a = vdupq_n_u8(b'A');
        let upper_z = vdupq_n_u8(b'Z');
        let lower_a = vdupq_n_u8(b'a');
        let lower_z = vdupq_n_u8(b'z');
        let digit_0 = vdupq_n_u8(b'0');
        let digit_9 = vdupq_n_u8(b'9');
        let space = vdupq_n_u8(b' ');
        let tab = vdupq_n_u8(b'\t');
        let lf = vdupq_n_u8(b'\n');
        let vt = vdupq_n_u8(0x0B);
        let ff = vdupq_n_u8(0x0C);
        let cr = vdupq_n_u8(b'\r');

        for chunk in chunks {
            let input = vld1q_u8(chunk.as_ptr());

            if vmaxvq_u8(input) >= 0x80 {
                let mut output = [0u8; 16];
                for (i, out) in output.iter_mut().enumerate() {
                    *out = self.ascii_lookup[chunk[i] as usize];
                }
                result.extend_from_slice(&output);
                continue;
            }

            let is_upper = vandq_u8(vcgeq_u8(input, upper_a), vcleq_u8(input, upper_z));
            let is_lower = vandq_u8(vcgeq_u8(input, lower_a), vcleq_u8(input, lower_z));
            let is_alpha = vorrq_u8(is_upper, is_lower);

            let is_digit = vandq_u8(vcgeq_u8(input, digit_0), vcleq_u8(input, digit_9));

            let is_ws = vorrq_u8(
                vorrq_u8(vceqq_u8(input, space), vceqq_u8(input, tab)),
                vorrq_u8(
                    vorrq_u8(vceqq_u8(input, lf), vceqq_u8(input, cr)),
                    vorrq_u8(vceqq_u8(input, vt), vceqq_u8(input, ff)),
                ),
            );

            let is_punct = Self::neon_ascii_punctuation_mask(input);

            let alpha_bits = vandq_u8(is_alpha, vdupq_n_u8(1));
            let digit_bits = vandq_u8(is_digit, vdupq_n_u8(2));
            let ws_bits = vandq_u8(is_ws, vdupq_n_u8(4));
            let punct_bits = vandq_u8(is_punct, vdupq_n_u8(8));
            let flags = vorrq_u8(
                vorrq_u8(alpha_bits, digit_bits),
                vorrq_u8(ws_bits, punct_bits),
            );

            let mut output = [0u8; 16];
            vst1q_u8(output.as_mut_ptr(), flags);
            result.extend_from_slice(&output);
        }

        for &byte in remainder {
            result.push(self.ascii_lookup[byte as usize]);
        }

        result
    }

    /// `u8::is_ascii_punctuation`'s four ranges (`0x21..=0x2F`,
    /// `0x3A..=0x40`, `0x5B..=0x60`, `0x7B..=0x7E`) as a NEON lane mask.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn neon_ascii_punctuation_mask(input: uint8x16_t) -> uint8x16_t {
        let range = |lo: u8, hi: u8| -> uint8x16_t {
            vandq_u8(
                vcgeq_u8(input, vdupq_n_u8(lo)),
                vcleq_u8(input, vdupq_n_u8(hi)),
            )
        };
        vorrq_u8(
            vorrq_u8(range(0x21, 0x2F), range(0x3A, 0x40)),
            vorrq_u8(range(0x5B, 0x60), range(0x7B, 0x7E)),
        )
    }

    /// Fast whitespace detection using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn find_whitespace_boundaries(&self, text: &[u8]) -> Vec<usize> {
        if !is_x86_feature_detected!("avx2") {
            return self.find_whitespace_boundaries_scalar(text);
        }

        unsafe { self.find_whitespace_boundaries_avx2(text) }
    }

    /// Fast whitespace detection using SIMD (NEON)
    #[cfg(target_arch = "aarch64")]
    pub fn find_whitespace_boundaries(&self, text: &[u8]) -> Vec<usize> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return self.find_whitespace_boundaries_scalar(text);
        }

        unsafe { self.find_whitespace_boundaries_neon(text) }
    }

    /// Fast whitespace detection - fallback for platforms with neither an
    /// AVX2 (`x86_64`) nor a NEON (`aarch64`) implementation.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
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

    /// AVX2-optimized whitespace boundary detection.
    ///
    /// Recognizes exactly the 8 byte values `0..=255` where
    /// [`Self::ascii_lookup`]'s bit 2 (`char::is_whitespace` applied to
    /// each byte as a Latin-1 codepoint) is set: the 6 ASCII whitespace
    /// bytes (space, tab, LF, VT, FF, CR) plus NEL (`0x85`) and NBSP
    /// (`0xA0`), which Unicode's White_Space property also covers. An
    /// earlier version of this function checked only space/tab/LF/CR
    /// (later just the 6 ASCII bytes), silently disagreeing with the
    /// scalar path -- and therefore with itself depending on which CPU ran
    /// it -- for the omitted bytes;
    /// `test_whitespace_boundaries_parity_with_scalar_for_every_byte`
    /// checks all 256 values against the scalar reference directly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_whitespace_boundaries_avx2(&self, text: &[u8]) -> Vec<usize> {
        let mut boundaries = Vec::new();
        let mut prev_whitespace_mask = 0u32;

        let space = _mm256_set1_epi8(b' ' as i8);
        let tab = _mm256_set1_epi8(b'\t' as i8);
        let newline = _mm256_set1_epi8(b'\n' as i8);
        let carriage_return = _mm256_set1_epi8(b'\r' as i8);
        let vertical_tab = _mm256_set1_epi8(0x0Bi8);
        let form_feed = _mm256_set1_epi8(0x0Ci8);
        let next_line = _mm256_set1_epi8(0x85u8 as i8);
        let no_break_space = _mm256_set1_epi8(0xA0u8 as i8);

        let chunks = text.chunks_exact(32);
        let mut offset = 0;

        for chunk in chunks {
            let input = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

            // Compare with whitespace characters. `cmpeq` compares raw
            // byte patterns, so it is correct here regardless of the
            // signed-byte interpretation `0x85`/`0xA0` would otherwise get
            // as negative values.
            let space_mask = _mm256_cmpeq_epi8(input, space);
            let tab_mask = _mm256_cmpeq_epi8(input, tab);
            let newline_mask = _mm256_cmpeq_epi8(input, newline);
            let cr_mask = _mm256_cmpeq_epi8(input, carriage_return);
            let vt_mask = _mm256_cmpeq_epi8(input, vertical_tab);
            let ff_mask = _mm256_cmpeq_epi8(input, form_feed);
            let nel_mask = _mm256_cmpeq_epi8(input, next_line);
            let nbsp_mask = _mm256_cmpeq_epi8(input, no_break_space);

            // Combine all whitespace masks
            let whitespace_mask = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_or_si256(space_mask, tab_mask),
                    _mm256_or_si256(newline_mask, cr_mask),
                ),
                _mm256_or_si256(
                    _mm256_or_si256(vt_mask, ff_mask),
                    _mm256_or_si256(nel_mask, nbsp_mask),
                ),
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
        let mut in_whitespace = prev_whitespace_mask != 0;
        for (i, &byte) in remainder.iter().enumerate() {
            let is_whitespace = matches!(
                byte,
                b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C | 0x85 | 0xA0
            );

            if is_whitespace != in_whitespace {
                boundaries.push(offset + i);
                in_whitespace = is_whitespace;
            }
        }

        boundaries
    }

    /// NEON-optimized whitespace boundary detection. Recognizes the same 8
    /// byte values as [`Self::find_whitespace_boundaries_avx2`]; see that
    /// function's docs.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn find_whitespace_boundaries_neon(&self, text: &[u8]) -> Vec<usize> {
        let mut boundaries = Vec::new();
        let mut in_whitespace = false;

        let space = vdupq_n_u8(b' ');
        let tab = vdupq_n_u8(b'\t');
        let lf = vdupq_n_u8(b'\n');
        let cr = vdupq_n_u8(b'\r');
        let vt = vdupq_n_u8(0x0B);
        let ff = vdupq_n_u8(0x0C);
        let nel = vdupq_n_u8(0x85);
        let nbsp = vdupq_n_u8(0xA0);

        let chunks = text.chunks_exact(16);
        let mut offset = 0;

        for chunk in chunks {
            let input = vld1q_u8(chunk.as_ptr());
            let ws_mask = vorrq_u8(
                vorrq_u8(
                    vorrq_u8(vceqq_u8(input, space), vceqq_u8(input, tab)),
                    vorrq_u8(vceqq_u8(input, lf), vceqq_u8(input, cr)),
                ),
                vorrq_u8(
                    vorrq_u8(vceqq_u8(input, vt), vceqq_u8(input, ff)),
                    vorrq_u8(vceqq_u8(input, nel), vceqq_u8(input, nbsp)),
                ),
            );

            let mut flags = [0u8; 16];
            vst1q_u8(flags.as_mut_ptr(), ws_mask);

            for (i, &flag) in flags.iter().enumerate() {
                let is_whitespace = flag != 0;
                if is_whitespace != in_whitespace {
                    boundaries.push(offset + i);
                    in_whitespace = is_whitespace;
                }
            }
            offset += 16;
        }

        let remainder = &text[offset..];
        for (i, &byte) in remainder.iter().enumerate() {
            let is_whitespace = matches!(
                byte,
                b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C | 0x85 | 0xA0
            );
            if is_whitespace != in_whitespace {
                boundaries.push(offset + i);
                in_whitespace = is_whitespace;
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

    /// Fast byte-to-UTF8 validation using SIMD (NEON)
    #[cfg(target_arch = "aarch64")]
    pub fn validate_utf8_fast(&self, bytes: &[u8]) -> Result<()> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return self.validate_utf8_scalar(bytes);
        }

        unsafe { self.validate_utf8_neon(bytes) }
    }

    /// Fast byte-to-UTF8 validation - fallback for platforms with neither
    /// an AVX2 (`x86_64`) nor a NEON (`aarch64`) implementation.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
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

    /// NEON-optimized UTF-8 validation. Mirrors
    /// [`Self::validate_utf8_avx2`]'s ASCII fast path.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn validate_utf8_neon(&self, bytes: &[u8]) -> Result<()> {
        let chunks = bytes.chunks_exact(16);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let input = vld1q_u8(chunk.as_ptr());
            if vmaxvq_u8(input) >= 0x80 {
                return self.validate_utf8_scalar(bytes);
            }
        }

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

    /// Fast case conversion using SIMD (NEON)
    #[cfg(target_arch = "aarch64")]
    pub fn to_lowercase_ascii(&self, text: &[u8]) -> Vec<u8> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return self.to_lowercase_ascii_scalar(text);
        }

        unsafe { self.to_lowercase_ascii_neon(text) }
    }

    /// Fast case conversion - fallback for platforms with neither an AVX2
    /// (`x86_64`) nor a NEON (`aarch64`) implementation.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
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
            let converted = if byte.is_ascii_uppercase() { byte + 32 } else { byte };
            result.push(converted);
        }

        result
    }

    /// NEON-optimized lowercase conversion. Mirrors
    /// [`Self::to_lowercase_ascii_avx2`]'s logic.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn to_lowercase_ascii_neon(&self, text: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(text.len());
        let chunks = text.chunks_exact(16);
        let remainder = chunks.remainder();

        let upper_a = vdupq_n_u8(b'A');
        let upper_z = vdupq_n_u8(b'Z');
        let to_lower_offset = vdupq_n_u8(32);

        for chunk in chunks {
            let input = vld1q_u8(chunk.as_ptr());

            let is_upper = vandq_u8(vcgeq_u8(input, upper_a), vcleq_u8(input, upper_z));
            let add_amount = vandq_u8(is_upper, to_lower_offset);
            let output = vaddq_u8(input, add_amount);

            let mut temp = [0u8; 16];
            vst1q_u8(temp.as_mut_ptr(), output);
            result.extend_from_slice(&temp);
        }

        for &byte in remainder {
            let converted = if byte.is_ascii_uppercase() { byte + 32 } else { byte };
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

        let tokens = tokenizer.preprocess_text(text).expect("Operation failed in test");

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

    /// Every whitespace byte the scalar table recognizes (`char::is_whitespace`
    /// over the ASCII range) must produce a boundary via the dispatched
    /// (potentially SIMD-accelerated) path too. Regression test for the
    /// AVX2 path's original space/tab/LF/CR-only check silently missing
    /// VT (0x0B) and FF (0x0C).
    #[test]
    fn test_whitespace_boundaries_recognizes_every_ascii_whitespace_byte() {
        let tokenizer = SimdTokenizer::new();
        for &ws in &[b' ', b'\t', b'\n', b'\r', 0x0Bu8, 0x0Cu8] {
            // Pad well past one SIMD chunk (32 bytes) on both sides so the
            // whitespace byte is exercised inside a full vector chunk, not
            // just the scalar remainder.
            let mut text = vec![b'a'; 40];
            text.push(ws);
            text.extend(vec![b'a'; 40]);

            let boundaries = tokenizer.find_whitespace_boundaries(&text);
            assert!(
                boundaries.contains(&40) && boundaries.contains(&41),
                "byte {:#04x} must be detected as a whitespace boundary",
                ws
            );
        }
    }

    /// The dispatched (potentially SIMD-accelerated) classification path
    /// must agree with the scalar reference for every possible byte value,
    /// individually and packed into full SIMD chunks (32 bytes covers both
    /// the AVX2 and NEON chunk sizes with no remainder either way).
    #[test]
    fn test_classify_parity_with_scalar_for_every_byte() {
        let tokenizer = SimdTokenizer::new();
        let all_bytes: Vec<u8> = (0..=255u8).collect();
        // 256 bytes = 8 AVX2 chunks (32B) = 16 NEON chunks (16B), no
        // remainder on either architecture.
        let dispatched = tokenizer.classify_ascii_chars(&all_bytes);
        let scalar = tokenizer.classify_ascii_chars_scalar(&all_bytes);
        assert_eq!(dispatched, scalar);

        // Also check a length that leaves a scalar remainder on both chunk
        // sizes (256 + 7).
        let mut odd_length = all_bytes.clone();
        odd_length.extend_from_slice(&all_bytes[..7]);
        assert_eq!(
            tokenizer.classify_ascii_chars(&odd_length),
            tokenizer.classify_ascii_chars_scalar(&odd_length)
        );
    }

    /// The dispatched lowercase-conversion path must agree with the
    /// scalar reference for every possible byte value.
    #[test]
    fn test_lowercase_parity_with_scalar_for_every_byte() {
        let tokenizer = SimdTokenizer::new();
        let all_bytes: Vec<u8> = (0..=255u8).collect();
        assert_eq!(
            tokenizer.to_lowercase_ascii(&all_bytes),
            tokenizer.to_lowercase_ascii_scalar(&all_bytes)
        );

        let mut odd_length = all_bytes.clone();
        odd_length.extend_from_slice(&all_bytes[..7]);
        assert_eq!(
            tokenizer.to_lowercase_ascii(&odd_length),
            tokenizer.to_lowercase_ascii_scalar(&odd_length)
        );
    }

    /// The dispatched whitespace-boundary path must agree with the scalar
    /// reference across a full mixed-content sweep of every byte value.
    #[test]
    fn test_whitespace_boundaries_parity_with_scalar_for_every_byte() {
        let tokenizer = SimdTokenizer::new();
        let all_bytes: Vec<u8> = (0..=255u8).collect();
        assert_eq!(
            tokenizer.find_whitespace_boundaries(&all_bytes),
            tokenizer.find_whitespace_boundaries_scalar(&all_bytes)
        );

        let mut odd_length = all_bytes.clone();
        odd_length.extend_from_slice(&all_bytes[..7]);
        assert_eq!(
            tokenizer.find_whitespace_boundaries(&odd_length),
            tokenizer.find_whitespace_boundaries_scalar(&odd_length)
        );
    }

    #[cfg(target_arch = "aarch64")]
    mod neon_specific {
        use super::*;

        /// Directly regression-tests the NEON classification path (bypassing
        /// runtime feature dispatch, which always selects NEON on real
        /// `aarch64` hardware anyway) against the scalar reference.
        #[test]
        fn test_classify_neon_parity_with_scalar() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let tokenizer = SimdTokenizer::new();
            let all_bytes: Vec<u8> = (0..=255u8).collect();
            let neon = unsafe { tokenizer.classify_ascii_chars_neon(&all_bytes) };
            let scalar = tokenizer.classify_ascii_chars_scalar(&all_bytes);
            assert_eq!(neon, scalar);
        }

        #[test]
        fn test_whitespace_boundaries_neon_parity_with_scalar() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let tokenizer = SimdTokenizer::new();
            let all_bytes: Vec<u8> = (0..=255u8).collect();
            let neon = unsafe { tokenizer.find_whitespace_boundaries_neon(&all_bytes) };
            let scalar = tokenizer.find_whitespace_boundaries_scalar(&all_bytes);
            assert_eq!(neon, scalar);
        }

        #[test]
        fn test_lowercase_neon_parity_with_scalar() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let tokenizer = SimdTokenizer::new();
            let all_bytes: Vec<u8> = (0..=255u8).collect();
            let neon = unsafe { tokenizer.to_lowercase_ascii_neon(&all_bytes) };
            let scalar = tokenizer.to_lowercase_ascii_scalar(&all_bytes);
            assert_eq!(neon, scalar);
        }

        #[test]
        fn test_validate_utf8_neon_agrees_with_scalar() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let tokenizer = SimdTokenizer::new();
            assert!(unsafe { tokenizer.validate_utf8_neon(b"Hello, valid ASCII text!") }.is_ok());
            assert!(unsafe { tokenizer.validate_utf8_neon(&[0xFFu8, 0xFE]) }.is_err());
            // Multi-byte UTF-8 (non-ASCII) must still validate correctly via
            // the scalar fallback the NEON path defers to.
            assert!(unsafe { tokenizer.validate_utf8_neon("Hello 世界".as_bytes()) }.is_ok());
        }
    }
}
