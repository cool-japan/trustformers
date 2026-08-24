//! Offset mapping: byte spans of tokens in the **original** input text.
//!
//! # The convention: byte offsets, not character offsets
//!
//! Every `(start, end)` this crate produces — `TokenizedInput::offset_mapping`,
//! [`crate::bpe::BPETokenizer::tokenize_with_offsets`],
//! [`crate::wordpiece::WordPieceTokenizer::tokenize_with_offsets`] — is a pair
//! of **byte** indices into the UTF-8 representation of the string that was
//! handed to the encoder, before any normalization. `&text[start..end]` is
//! therefore always a valid (never panicking) slice of that string.
//!
//! This matches the rest of the tree: HuggingFace's Rust `tokenizers` crate
//! reports byte offsets too, so [`crate::tokenizer::TokenizerImpl`] (which
//! wraps it) already behaved this way, and `trustformers-py`'s span extraction
//! slices with `text.get(start..end)` — Rust byte-range indexing.
//!
//! It does **not** match what a Python caller needs. Python's `str` is indexed
//! by code point, so `text[start:end]` in Python needs *character* offsets,
//! which is what HuggingFace's Python-facing `offset_mapping` carries. The two
//! agree for pure ASCII and diverge the moment a multi-byte character occurs
//! before the span of interest. [`byte_offsets_to_char_offsets`] performs that
//! conversion explicitly; nothing in this crate performs it silently.
//!
//! # Alignment through normalization
//!
//! Tokenizers normalize before they split (BERT strips accents and lowercases;
//! byte-level BPE applies NFC), and those transforms change byte lengths — NFD
//! turns one `é` into two characters, NFC turns two into one, `İ` lowercases to
//! two characters. Offsets must nevertheless index the *pre-normalization*
//! input, so every normalization step here returns an [`OffsetAlignment`]
//! alongside its output: a monotone map from byte ranges of the normalized
//! string back to byte ranges of its input. Alignments compose
//! ([`OffsetAlignment::rebase`]), so a whole pipeline of steps collapses into
//! one map back to the original.
//!
//! Each step is aligned *structurally* (per character, or per canonical
//! combining run) and then **verified**: the concatenation of the per-segment
//! results is compared against the real whole-string result, and if they
//! disagree the alignment degrades to a single coarse unit spanning the whole
//! input rather than reporting boundaries that are subtly wrong. A wider span
//! that genuinely contains the token's source is honest; a precise-looking span
//! pointing at the wrong characters is not.

use trustformers_core::errors::{Result, TrustformersError};
use unicode_categories::UnicodeCategories;
use unicode_normalization::char::{canonical_combining_class, compose};
use unicode_normalization::UnicodeNormalization;

/// A `(start, end)` pair of byte indices.
pub type ByteSpan = (usize, usize);

/// One contiguous piece of a normalized string together with the byte range of
/// the pre-normalization string it was produced from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AlignmentUnit {
    norm_start: usize,
    norm_end: usize,
    src_start: usize,
    src_end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AlignmentKind {
    /// The normalization left the string untouched: every byte index maps to
    /// itself. Stored as a marker rather than as one unit per character so the
    /// (overwhelmingly common) no-op case costs no allocation at all.
    Identity,
    /// Explicit units, in order, tiling the normalized string.
    Units(Vec<AlignmentUnit>),
}

/// A monotone map from byte ranges of a normalized string back to byte ranges
/// of the string it was normalized from.
///
/// The map is *not* required to be injective or length-preserving: several
/// normalized characters may come from one source character (NFD), and one
/// normalized character may come from several (NFC). A range that covers part
/// of a source character always maps to the whole of that character, which is
/// what makes `&source[start..end]` a valid slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffsetAlignment {
    kind: AlignmentKind,
    normalized_len: usize,
    source_len: usize,
}

impl Default for OffsetAlignment {
    fn default() -> Self {
        Self {
            kind: AlignmentKind::Identity,
            normalized_len: 0,
            source_len: 0,
        }
    }
}

impl OffsetAlignment {
    /// The alignment of a normalization that changed nothing.
    pub fn identity(text: &str) -> Self {
        Self {
            kind: AlignmentKind::Identity,
            normalized_len: text.len(),
            source_len: text.len(),
        }
    }

    /// The maximally coarse alignment: every normalized byte range maps to the
    /// whole source. Used only when a structural alignment failed verification.
    pub fn coarse(normalized_len: usize, source_len: usize) -> Self {
        let units = if normalized_len == 0 {
            Vec::new()
        } else {
            vec![AlignmentUnit {
                norm_start: 0,
                norm_end: normalized_len,
                src_start: 0,
                src_end: source_len,
            }]
        };
        Self {
            kind: AlignmentKind::Units(units),
            normalized_len,
            source_len,
        }
    }

    /// Byte length of the normalized string this alignment describes.
    pub fn normalized_len(&self) -> usize {
        self.normalized_len
    }

    /// Byte length of the source string this alignment maps back into.
    pub fn source_len(&self) -> usize {
        self.source_len
    }

    /// Whether this alignment is the zero-cost identity map.
    pub fn is_identity(&self) -> bool {
        matches!(self.kind, AlignmentKind::Identity)
    }

    /// Map a byte range of the normalized string to the byte range of the
    /// source it came from.
    ///
    /// The result is the union of the source ranges of every unit the query
    /// overlaps, so a range covering part of a source character reports the
    /// whole character. An empty query range maps to an empty range anchored at
    /// the corresponding source position.
    pub fn map_span(&self, start: usize, end: usize) -> ByteSpan {
        let start = start.min(self.normalized_len);
        let end = end.clamp(start, self.normalized_len);

        let units = match &self.kind {
            AlignmentKind::Identity => return (start, end),
            AlignmentKind::Units(units) => units,
        };

        let first = units.partition_point(|unit| unit.norm_end <= start);

        if start == end {
            let anchor = match units.get(first) {
                Some(unit) => unit.src_start,
                None => units.last().map(|unit| unit.src_end).unwrap_or(0),
            };
            return (anchor, anchor);
        }

        let mut src_start = usize::MAX;
        let mut src_end = 0usize;
        for unit in &units[first..] {
            if unit.norm_start >= end {
                break;
            }
            src_start = src_start.min(unit.src_start);
            src_end = src_end.max(unit.src_end);
        }

        if src_start == usize::MAX {
            // No unit overlaps the query (the normalized bytes came from
            // nothing in the source, e.g. an inserted separator). Report the
            // empty range where those bytes would sit.
            let anchor = units
                .get(first)
                .map(|unit| unit.src_start)
                .or_else(|| units.last().map(|unit| unit.src_end))
                .unwrap_or(0);
            return (anchor, anchor);
        }

        (src_start, src_end.max(src_start))
    }

    /// Compose two alignments: `self` maps `B -> A`, `inner` maps `A -> C`, and
    /// the result maps `B -> C`.
    pub fn rebase(&self, inner: &OffsetAlignment) -> OffsetAlignment {
        match &self.kind {
            AlignmentKind::Identity => OffsetAlignment {
                kind: inner.kind.clone(),
                normalized_len: self.normalized_len,
                source_len: inner.source_len,
            },
            AlignmentKind::Units(units) => {
                if inner.is_identity() {
                    return OffsetAlignment {
                        kind: self.kind.clone(),
                        normalized_len: self.normalized_len,
                        source_len: inner.source_len,
                    };
                }
                let rebased = units
                    .iter()
                    .map(|unit| {
                        let (src_start, src_end) = inner.map_span(unit.src_start, unit.src_end);
                        AlignmentUnit {
                            norm_start: unit.norm_start,
                            norm_end: unit.norm_end,
                            src_start,
                            src_end,
                        }
                    })
                    .collect();
                OffsetAlignment {
                    kind: AlignmentKind::Units(rebased),
                    normalized_len: self.normalized_len,
                    source_len: inner.source_len,
                }
            },
        }
    }
}

/// Incremental builder for an [`OffsetAlignment`] whose units are known
/// directly (used by the structural, per-character normalizers: control-code
/// stripping, CJK space padding).
#[derive(Debug, Default)]
pub struct AlignmentBuilder {
    units: Vec<AlignmentUnit>,
    normalized_len: usize,
}

impl AlignmentBuilder {
    /// A builder with no units recorded yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that the next `out_len` bytes of output were produced from
    /// `src_start..src_end` of the input.
    ///
    /// `out_len == 0` records nothing (the input range was deleted);
    /// `src_start == src_end` records an insertion, which later maps to an
    /// empty source range at that position.
    pub fn push(&mut self, out_len: usize, src_start: usize, src_end: usize) {
        if out_len == 0 {
            return;
        }
        self.units.push(AlignmentUnit {
            norm_start: self.normalized_len,
            norm_end: self.normalized_len + out_len,
            src_start,
            src_end,
        });
        self.normalized_len += out_len;
    }

    /// Advance the output cursor without recording a source range, for output
    /// bytes that were inserted from nothing (a padding space, for instance).
    pub fn skip_output(&mut self, out_len: usize) {
        self.normalized_len += out_len;
    }

    /// Finish the alignment. `source_len` is the byte length of the input.
    pub fn finish(self, source_len: usize) -> OffsetAlignment {
        OffsetAlignment {
            kind: AlignmentKind::Units(self.units),
            normalized_len: self.normalized_len,
            source_len,
        }
    }
}

/// BERT's `_run_strip_accents`: canonical decomposition, then drop every
/// non-spacing mark.
pub fn strip_accents(text: &str) -> String {
    text.nfd().filter(|ch| !ch.is_mark_nonspacing()).collect()
}

/// Byte ranges of `text`'s canonical combining runs: each run starts at a
/// *starter* (combining class 0) and absorbs the non-starters that follow it.
///
/// Canonical decomposition is per character and canonical ordering never
/// reorders across a starter, so normalizing each run independently and
/// concatenating reproduces whole-string NFD exactly.
fn canonical_runs(text: &str) -> Vec<ByteSpan> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        if index > 0 && canonical_combining_class(ch) == 0 {
            runs.push((start, index));
            start = index;
        }
    }
    runs.push((start, text.len()));
    runs
}

/// Upper bound on how many canonical runs may be merged into one NFC segment.
///
/// Composition across a starter boundary only happens for Hangul (`L + V`, then
/// `LV + T`), so two merges always suffice; the bound only exists so that a
/// pathological input cannot turn the merge loop quadratic.
const MAX_NFC_RUN_MERGES: usize = 4;

/// Byte ranges of the segments NFC may be applied to independently.
///
/// Canonical *composition* can cross a starter boundary — `L + V` and `LV + T`
/// for Hangul — so adjacent canonical runs are merged whenever the composed
/// tail of the left one combines with the head of the right one.
fn nfc_segments(text: &str) -> Vec<ByteSpan> {
    let runs = canonical_runs(text);
    let mut segments: Vec<ByteSpan> = Vec::with_capacity(runs.len());
    let mut merges_in_tail = 0usize;

    for (start, end) in runs {
        let head = text[start..end].chars().next();
        let merged = match (segments.last_mut(), head) {
            (Some(previous), Some(head)) if merges_in_tail < MAX_NFC_RUN_MERGES => {
                let tail: String = text[previous.0..previous.1].nfc().collect();
                match tail.chars().next_back() {
                    Some(tail_char) if compose(tail_char, head).is_some() => {
                        previous.1 = end;
                        true
                    },
                    _ => false,
                }
            },
            _ => false,
        };

        if merged {
            merges_in_tail += 1;
        } else {
            segments.push((start, end));
            merges_in_tail = 0;
        }
    }

    segments
}

/// Apply `transform` to each segment of `text` independently, building the
/// alignment from the per-segment results, and verify that concatenating them
/// reproduces `whole` (the real whole-string result).
///
/// On mismatch the per-segment boundaries would be wrong, so the alignment
/// degrades to one coarse unit spanning all of `text`.
fn aligned_by_segments(
    text: &str,
    segments: &[ByteSpan],
    whole: String,
    transform: impl Fn(&str) -> String,
) -> (String, OffsetAlignment) {
    let mut builder = AlignmentBuilder::new();
    let mut concatenated = String::with_capacity(whole.len());

    for &(start, end) in segments {
        let segment = &text[start..end];
        let transformed = transform(segment);
        if transformed == segment {
            // Unchanged segment: align it character by character, which is
            // strictly finer than one unit for the whole segment.
            for (index, ch) in segment.char_indices() {
                let len = ch.len_utf8();
                builder.push(len, start + index, start + index + len);
            }
        } else {
            builder.push(transformed.len(), start, end);
        }
        concatenated.push_str(&transformed);
    }

    if concatenated != whole {
        let alignment = OffsetAlignment::coarse(whole.len(), text.len());
        return (whole, alignment);
    }

    let alignment = builder.finish(text.len());
    (whole, alignment)
}

/// NFC-normalize `text`, with an alignment back into it.
pub fn aligned_nfc(text: &str) -> (String, OffsetAlignment) {
    let whole: String = text.nfc().collect();
    if whole == text {
        return (whole, OffsetAlignment::identity(text));
    }
    let segments = nfc_segments(text);
    aligned_by_segments(text, &segments, whole, |segment| segment.nfc().collect())
}

/// Strip accents (NFD, then drop non-spacing marks), with an alignment back
/// into `text`.
pub fn aligned_strip_accents(text: &str) -> (String, OffsetAlignment) {
    let whole = strip_accents(text);
    if whole == text {
        return (whole, OffsetAlignment::identity(text));
    }
    let segments = canonical_runs(text);
    aligned_by_segments(text, &segments, whole, strip_accents)
}

/// Lowercase `text` (exactly `str::to_lowercase`, final-sigma rule included),
/// with an alignment back into it.
///
/// The alignment is built per character from `char::to_lowercase`. That differs
/// from `str::to_lowercase` in exactly one place — a word-final capital sigma
/// becomes `ς` rather than `σ` — and those two characters have the same UTF-8
/// length, so the per-character *lengths* the alignment is built from stay
/// correct. The total length is verified all the same.
pub fn aligned_lowercase(text: &str) -> (String, OffsetAlignment) {
    let whole = text.to_lowercase();
    if whole == text {
        return (whole, OffsetAlignment::identity(text));
    }

    let mut builder = AlignmentBuilder::new();
    let mut total = 0usize;
    for (index, ch) in text.char_indices() {
        let out_len: usize = ch.to_lowercase().map(char::len_utf8).sum();
        builder.push(out_len, index, index + ch.len_utf8());
        total += out_len;
    }

    if total != whole.len() {
        let alignment = OffsetAlignment::coarse(whole.len(), text.len());
        return (whole, alignment);
    }

    let alignment = builder.finish(text.len());
    (whole, alignment)
}

/// Largest character boundary of `text` that is `<= index`.
pub fn floor_char_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// Smallest character boundary of `text` that is `>= index`.
pub fn ceil_char_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

/// Convert this crate's byte offsets into the Unicode **character** (code
/// point) offsets a Python caller needs to index a `str` directly, which is
/// the convention HuggingFace's Python `offset_mapping` uses.
///
/// `(0, 0)` — this crate's marker for special and padding tokens — converts to
/// `(0, 0)`, as it must.
///
/// # Errors
///
/// Returns a structured error naming the offending index when a byte offset
/// falls outside `text` or lands in the middle of a multi-byte character,
/// rather than silently reporting a plausible-looking wrong position.
pub fn byte_offsets_to_char_offsets(
    text: &str,
    offsets: &[ByteSpan],
) -> Result<Vec<(usize, usize)>> {
    let mut char_index_of_byte = vec![usize::MAX; text.len() + 1];
    for (char_index, (byte_index, _)) in text.char_indices().enumerate() {
        char_index_of_byte[byte_index] = char_index;
    }
    if let Some(last) = char_index_of_byte.last_mut() {
        *last = text.chars().count();
    }

    let lookup = |byte_index: usize| -> Result<usize> {
        let mapped = char_index_of_byte.get(byte_index).copied().unwrap_or(usize::MAX);
        if mapped == usize::MAX {
            return Err(TrustformersError::invalid_input(format!(
                "byte offset {} is not a character boundary of the {}-byte input text, so it \
                 has no character offset; offsets must come from the same string they are \
                 being converted against",
                byte_index,
                text.len()
            )));
        }
        Ok(mapped)
    };

    offsets
        .iter()
        .map(|&(start, end)| {
            if start > end {
                return Err(TrustformersError::invalid_input(format!(
                    "offset span ({}, {}) is reversed; start must not exceed end",
                    start, end
                )));
            }
            Ok((lookup(start)?, lookup(end)?))
        })
        .collect()
}

/// Inverse of [`byte_offsets_to_char_offsets`]: character (code point) offsets
/// back to this crate's byte offsets.
///
/// # Errors
///
/// Returns a structured error naming the offending index when a character
/// offset is past the end of `text`, or when a span is reversed.
pub fn char_offsets_to_byte_offsets(
    text: &str,
    offsets: &[(usize, usize)],
) -> Result<Vec<ByteSpan>> {
    let mut byte_index_of_char: Vec<usize> = text.char_indices().map(|(index, _)| index).collect();
    byte_index_of_char.push(text.len());

    let lookup = |char_index: usize| -> Result<usize> {
        byte_index_of_char.get(char_index).copied().ok_or_else(|| {
            TrustformersError::invalid_input(format!(
                "character offset {} is past the end of the input text, which has {} characters",
                char_index,
                byte_index_of_char.len().saturating_sub(1)
            ))
        })
    };

    offsets
        .iter()
        .map(|&(start, end)| {
            if start > end {
                return Err(TrustformersError::invalid_input(format!(
                    "offset span ({}, {}) is reversed; start must not exceed end",
                    start, end
                )));
            }
            Ok((lookup(start)?, lookup(end)?))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_alignment_maps_every_span_to_itself() {
        let alignment = OffsetAlignment::identity("hello");
        assert!(alignment.is_identity());
        assert_eq!(alignment.map_span(0, 5), (0, 5));
        assert_eq!(alignment.map_span(2, 3), (2, 3));
        assert_eq!(alignment.map_span(2, 2), (2, 2));
        // Out-of-range queries clamp instead of panicking.
        assert_eq!(alignment.map_span(3, 99), (3, 5));
    }

    #[test]
    fn nfc_alignment_maps_composed_char_back_to_its_decomposed_source() {
        let source = "cafe\u{301}"; // 6 bytes: "cafe" + combining acute
        let (normalized, alignment) = aligned_nfc(source);
        assert_eq!(normalized, "café");
        assert_eq!(normalized.len(), 5);
        assert!(!alignment.is_identity());

        // "caf" is untouched and aligns character by character.
        assert_eq!(alignment.map_span(0, 1), (0, 1));
        assert_eq!(alignment.map_span(1, 2), (1, 2));
        assert_eq!(alignment.map_span(2, 3), (2, 3));
        // The composed "é" (bytes 3..5 of the output) came from "e" + U+0301
        // (bytes 3..6 of the source).
        assert_eq!(alignment.map_span(3, 5), (3, 6));
        assert_eq!(&source[3..6], "e\u{301}");
    }

    #[test]
    fn nfc_alignment_composes_hangul_across_starter_boundaries() {
        // L + V + T, three starters that NFC composes into one syllable.
        let source = "\u{1100}\u{1161}\u{11A8}";
        let (normalized, alignment) = aligned_nfc(source);
        // U+AC01 (각): one syllable composed from all three jamo.
        assert_eq!(normalized, "\u{AC01}");
        // Verification must have succeeded: a coarse fallback would still be
        // correct here (one segment), so assert the mapping directly.
        assert_eq!(alignment.map_span(0, normalized.len()), (0, source.len()));
    }

    #[test]
    fn strip_accents_alignment_keeps_base_letter_pointing_at_whole_character() {
        let source = "café";
        let (normalized, alignment) = aligned_strip_accents(source);
        assert_eq!(normalized, "cafe");
        // "é" occupies bytes 3..5 of the source and becomes one byte of output.
        assert_eq!(alignment.map_span(3, 4), (3, 5));
        assert_eq!(&source[3..5], "é");
        assert_eq!(alignment.map_span(0, 1), (0, 1));
    }

    #[test]
    fn lowercase_alignment_survives_a_character_that_expands() {
        // U+0130 (İ) lowercases to two characters.
        let source = "\u{130}A";
        let (normalized, alignment) = aligned_lowercase(source);
        assert_eq!(normalized, source.to_lowercase());
        assert!(normalized.chars().count() > source.chars().count());
        // Everything the expansion produced points back at the single source
        // character it came from (bytes 0..2).
        assert_eq!(alignment.map_span(0, normalized.len() - 1), (0, 2));
    }

    #[test]
    fn lowercase_alignment_handles_the_final_sigma_rule() {
        // `str::to_lowercase` maps a word-final Σ to ς, not σ; both are two
        // bytes, so the per-character alignment stays exact.
        let source = "ΑΣ";
        let (normalized, alignment) = aligned_lowercase(source);
        assert_eq!(normalized, "ας");
        assert_eq!(alignment.map_span(2, 4), (2, 4));
    }

    #[test]
    fn alignments_compose() {
        let source = "CAFE\u{301}";
        let (stripped, first) = aligned_strip_accents(source);
        assert_eq!(stripped, "CAFE");
        let (lowered, second) = aligned_lowercase(&stripped);
        assert_eq!(lowered, "cafe");

        let combined = second.rebase(&first);
        assert_eq!(combined.source_len(), source.len());
        // "e" (byte 3 of the output) traces back to "E" + U+0301.
        assert_eq!(combined.map_span(3, 4), (3, 6));
    }

    #[test]
    fn builder_records_insertions_as_empty_source_ranges() {
        let mut builder = AlignmentBuilder::new();
        builder.push(1, 0, 1); // 'a'
        builder.skip_output(1); // an inserted space
        builder.push(3, 1, 4); // a CJK character
        let alignment = builder.finish(4);
        assert_eq!(alignment.map_span(0, 1), (0, 1));
        assert_eq!(alignment.map_span(2, 5), (1, 4));
        // The inserted byte belongs to no source range.
        assert_eq!(alignment.map_span(1, 2), (1, 1));
    }

    #[test]
    fn byte_offsets_convert_to_python_style_character_offsets() {
        // h(1) é(2) l(1) l(1) o(1) ' '(1) 世(3) 界(3) = 13 bytes, 8 characters.
        let text = "héllo 世界";
        assert_eq!(text.len(), 13);
        let byte_offsets = vec![(0, 0), (0, 1), (1, 3), (7, 10), (10, 13)];
        let char_offsets = byte_offsets_to_char_offsets(text, &byte_offsets)
            .expect("every span is on a character boundary");
        assert_eq!(char_offsets, vec![(0, 0), (0, 1), (1, 2), (6, 7), (7, 8)]);
        // The point of the conversion: Python's `text[start:end]` needs these.
        assert_eq!(&text[7..10], "世");

        // Round-trip.
        let back = char_offsets_to_byte_offsets(text, &char_offsets)
            .expect("character offsets are in range");
        assert_eq!(back, byte_offsets);
    }

    #[test]
    fn byte_offset_conversion_refuses_a_non_boundary_index() {
        let text = "héllo";
        // Byte 2 is the middle of "é".
        let error = byte_offsets_to_char_offsets(text, &[(1, 2)])
            .expect_err("a non-boundary byte offset must be an error, never a guess");
        assert!(
            error.to_string().contains("character boundary"),
            "error must explain what is wrong, got: {}",
            error
        );

        let error = byte_offsets_to_char_offsets(text, &[(0, 99)])
            .expect_err("an out-of-range byte offset must be an error");
        assert!(
            error.to_string().contains("99"),
            "error must name the index"
        );
    }

    #[test]
    fn char_offset_conversion_refuses_out_of_range_and_reversed_spans() {
        let text = "héllo";
        let error = char_offsets_to_byte_offsets(text, &[(0, 99)])
            .expect_err("an out-of-range character offset must be an error");
        assert!(error.to_string().contains("past the end"));

        let error = char_offsets_to_byte_offsets(text, &[(3, 1)])
            .expect_err("a reversed span must be an error");
        assert!(error.to_string().contains("reversed"));
    }

    #[test]
    fn coarse_alignment_still_contains_the_source() {
        let alignment = OffsetAlignment::coarse(4, 9);
        assert_eq!(alignment.map_span(0, 4), (0, 9));
        assert_eq!(alignment.map_span(1, 2), (0, 9));
        assert_eq!(alignment.map_span(2, 2), (0, 0));
    }

    #[test]
    fn char_boundary_helpers_never_split_a_character() {
        let text = "aé世";
        assert_eq!(floor_char_boundary(text, 2), 1);
        assert_eq!(ceil_char_boundary(text, 2), 3);
        assert_eq!(ceil_char_boundary(text, 99), text.len());
        assert_eq!(floor_char_boundary(text, 0), 0);
    }
}
