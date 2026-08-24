//! Tightness of the offset contract: spans must be *precise*, not merely valid.
//!
//! `tests/offset_mapping.rs` pins that every reported span is in range, lands on
//! a character boundary, and contains the text its token covers. Those are
//! validity properties, and they share a blind spot: a maximally coarse
//! alignment — every token reporting `(0, text.len())` — satisfies all of them.
//!
//! That is not a hypothetical shape. `crate::offsets` really does have such a
//! fallback: each normalization stage aligns its input structurally and then
//! verifies that concatenating the per-segment results reproduces the real
//! whole-string result; on mismatch it deliberately degrades to a single coarse
//! unit spanning the whole input (a wider span that genuinely contains the
//! token's source is honest, a precise-looking wrong one is not). The fallback
//! is the right call, but nothing until now could tell whether it was firing.
//!
//! These tests assert the properties a collapse breaks, over input chosen to
//! stress every stage: alignments pin ASCII sentinels exactly, spans advance
//! monotonically, byte-level BPE's spans tile the input with no gaps, and —
//! wherever normalization is the identity — a span is *exactly* its token's
//! surface text. Measured across the corpus below the coarse fallback never
//! fires today; these tests are what will notice if that changes.

use std::collections::HashMap;
use trustformers_tokenizers::bpe::{unicode_to_byte, BPETokenizer};
use trustformers_tokenizers::offsets::{
    aligned_lowercase, aligned_nfc, aligned_strip_accents, OffsetAlignment,
};
use trustformers_tokenizers::wordpiece::WordPieceTokenizer;

/// Input chosen to stress the normalization stages: canonical composition and
/// decomposition, combining runs, length-changing case folding, Hangul jamo
/// (the one place NFC composes *across* a starter boundary), compatibility
/// singletons, astral planes, and zero-width characters.
const ADVERSARIAL: &[(&str, &str)] = &[
    ("turkish_dotted_i", "İstanbul"),
    ("turkish_dotless_i", "Iıİi"),
    ("capital_sharp_s", "STRAẞE"),
    ("sharp_s", "Straße"),
    ("ligature_fi", "ﬁle"),
    ("zwj_family", "👨‍👩‍👧‍👦"),
    ("emoji_skin_tone", "👋🏽"),
    ("decomposed_acute", "cafe\u{301}"),
    ("precomposed_acute", "café"),
    ("stacked_marks", "a\u{300}\u{301}\u{327}b"),
    ("hangul_precomposed", "각국어"),
    ("hangul_jamo_one_syllable", "\u{1100}\u{1161}\u{11A8}"),
    (
        "hangul_jamo_three_syllables",
        "\u{1100}\u{1161}\u{11A8}\u{1102}\u{1161}\u{1103}\u{1161}",
    ),
    ("greek_final_sigma", "ΟΔΟΣ ΑΣ"),
    ("angstrom_sign", "\u{212B}"),
    ("ohm_sign", "\u{2126}"),
    ("kelvin_sign", "\u{212A}"),
    ("title_case_digraph", "\u{01C5}"),
    ("roman_numerals", "\u{2160}\u{2161}"),
    ("cherokee", "\u{13A0}\u{13A1}\u{13A2}"),
    ("arabic_harakat", "السَّلَامُ"),
    ("hebrew_points", "\u{05D0}\u{05B8}\u{05DC}\u{05B6}\u{05E3}"),
    ("devanagari", "नमस्ते"),
    ("thai", "สวัสดี"),
    ("vietnamese", "Tiếng Việt"),
    ("cjk", "世界"),
    ("fullwidth_latin", "ＨＥＬＬＯ"),
    ("circled_digits", "\u{2460}\u{2461}"),
    ("mark_over_cjk", "世\u{301}界"),
    (
        "long_mark_run",
        "e\u{301}\u{302}\u{303}\u{304}\u{305}\u{306}\u{307}",
    ),
    ("byte_order_mark", "\u{FEFF}"),
    ("no_break_space", "\u{00A0}"),
    ("zero_width_space", "\u{200B}"),
    ("georgian_mtavruli", "\u{10D0}\u{1C90}"),
    ("deseret", "\u{10400}\u{10428}"),
    ("mathematical_bold", "\u{1D400}\u{1D401}"),
    ("iota_subscript", "\u{1FB3}\u{0345}"),
    ("armenian_ligature", "\u{0587}"),
    ("lithuanian_dot_above", "\u{0049}\u{0307}\u{0300}"),
    ("everything_at_once", "Ωé世🙂İß\u{FB01}"),
];

/// `raw` between two ASCII sentinels. The sentinels are unaffected by every
/// normalization stage, so a correct alignment maps them back to themselves —
/// and a collapsed one cannot.
fn sentineled(raw: &str) -> String {
    format!("aa {raw} zz")
}

// ------------------------------------------------- the alignment engine

/// The load-bearing anti-collapse test.
///
/// If a stage's structural alignment ever fails verification, every span it
/// produces widens to the whole input. With ASCII sentinels on both ends that
/// is directly observable: the leading `aa` must still map to the first two
/// bytes and the trailing `zz` to the last two, whatever happened in between.
#[test]
fn normalization_stages_never_degrade_to_a_coarse_alignment() {
    for (name, raw) in ADVERSARIAL {
        let text = sentineled(raw);
        let stages: [(&str, (String, OffsetAlignment)); 3] = [
            ("nfc", aligned_nfc(&text)),
            ("strip_accents", aligned_strip_accents(&text)),
            ("lowercase", aligned_lowercase(&text)),
        ];

        for (stage, (normalized, alignment)) in stages {
            assert_eq!(
                alignment.source_len(),
                text.len(),
                "{name}/{stage}: alignment must describe the whole source"
            );
            assert_eq!(
                alignment.normalized_len(),
                normalized.len(),
                "{name}/{stage}: alignment must describe the whole normalized string"
            );
            assert!(
                normalized.starts_with("aa ") && normalized.ends_with(" zz"),
                "{name}/{stage}: the sentinels must survive normalization, got {normalized:?}"
            );

            assert_eq!(
                alignment.map_span(0, 2),
                (0, 2),
                "{name}/{stage}: leading sentinel widened — alignment collapsed \
                 ({text:?} -> {normalized:?})"
            );
            let normalized_len = normalized.len();
            assert_eq!(
                alignment.map_span(normalized_len - 2, normalized_len),
                (text.len() - 2, text.len()),
                "{name}/{stage}: trailing sentinel widened — alignment collapsed \
                 ({text:?} -> {normalized:?})"
            );
        }
    }
}

/// Composition must survive too: `rebase` chains the three stages the way
/// `WordPieceTokenizer` does, and the sentinels must still be pinned at the end
/// of the chain, not just at each individual step.
#[test]
fn composed_normalization_alignments_still_pin_the_sentinels() {
    for (name, raw) in ADVERSARIAL {
        let text = sentineled(raw);

        let (after_accents, accents_alignment) = aligned_strip_accents(&text);
        let (after_lower, lower_alignment) = aligned_lowercase(&after_accents);
        let composed = lower_alignment.rebase(&accents_alignment);

        assert_eq!(
            composed.source_len(),
            text.len(),
            "{name}: composed source length"
        );
        assert_eq!(
            composed.normalized_len(),
            after_lower.len(),
            "{name}: composed normalized length"
        );
        assert_eq!(
            composed.map_span(0, 2),
            (0, 2),
            "{name}: composed alignment lost the leading sentinel"
        );
        let end = after_lower.len();
        assert_eq!(
            composed.map_span(end - 2, end),
            (text.len() - 2, text.len()),
            "{name}: composed alignment lost the trailing sentinel"
        );
    }
}

// ---------------------------------------------------------------- fixtures

fn wordpiece_vocab() -> HashMap<String, u32> {
    let tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "aa", "zz", "ca", "##fe", "世", "界", "hello", "world",
    ];
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| ((*token).to_string(), index as u32))
        .collect()
}

/// A byte-level BPE vocabulary derived from the crate's own (tested) inverse
/// table, so the fixture cannot drift from the alphabet the tokenizer uses.
fn byte_level_vocab() -> HashMap<String, u32> {
    (0..=255u8)
        .map(|byte| {
            let symbol = (0u32..512)
                .filter_map(std::char::from_u32)
                .find(|candidate| unicode_to_byte(*candidate) == Some(byte));
            (symbol.into_iter().collect::<String>(), byte as u32)
        })
        .collect()
}

// ------------------------------------------------------ WordPiece tightness

/// Spans must *advance*. A collapsed alignment reports the same whole-input
/// span for every token, which this rejects on any input tokenizing to more
/// than one piece.
#[test]
fn wordpiece_spans_advance_monotonically_over_adversarial_unicode() {
    for do_lower_case in [true, false] {
        let tokenizer = WordPieceTokenizer::new(wordpiece_vocab(), do_lower_case);

        for (name, raw) in ADVERSARIAL {
            let text = sentineled(raw);
            let (tokens, offsets) = tokenizer.tokenize_with_offsets(&text);
            assert_eq!(tokens.len(), offsets.len(), "{name}: one span per token");

            for window in offsets.windows(2) {
                let (previous, next) = (window[0], window[1]);
                assert!(
                    previous.0 <= next.0 && previous.1 <= next.1,
                    "{name} (lower={do_lower_case}): spans run backwards \
                     ({previous:?} then {next:?}) in {text:?} -> {tokens:?}"
                );
                assert!(
                    previous.1 <= next.0,
                    "{name} (lower={do_lower_case}): spans overlap \
                     ({previous:?} then {next:?}) in {text:?} -> {tokens:?}"
                );
            }

            // The sentinels bracket the input, so the first and last tokens
            // must sit at the two ends — the sharpest single statement that
            // the middle did not swallow the whole string.
            let first = *offsets.first().expect("the leading sentinel always tokenizes");
            let last = *offsets.last().expect("the trailing sentinel always tokenizes");
            assert_eq!(
                first,
                (0, 2),
                "{name} (lower={do_lower_case}): leading sentinel span in {text:?} -> {tokens:?}"
            );
            assert_eq!(
                last,
                (text.len() - 2, text.len()),
                "{name} (lower={do_lower_case}): trailing sentinel span in {text:?} -> {tokens:?}"
            );
        }
    }
}

/// Where normalization preserves the surface text — a cased tokenizer over
/// input that is already NFC and accent-free — the contract is exact equality,
/// not containment: the reported span *is* the token's surface text.
///
/// Note this is stronger than "normalization is the identity". The CJK inputs
/// below still pass through the Chinese-character space padding stage, which
/// inserts spaces around every CJK codepoint; the spans must survive that and
/// still name the original character exactly.
#[test]
fn wordpiece_cased_spans_are_exactly_the_token_surface_text() {
    let mut vocab = wordpiece_vocab();
    for (index, token) in ["Hello", "World", "CAFÉ", "🙂", "##界"].iter().enumerate() {
        vocab.insert((*token).to_string(), 900 + index as u32);
    }
    let tokenizer = WordPieceTokenizer::new(vocab, false);

    for text in [
        "Hello World",
        "CAFÉ Hello",
        "世界 World",
        "🙂 Hello",
        "Hello 🙂 CAFÉ 世界",
    ] {
        let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

        for (token, &(start, end)) in tokens.iter().zip(&offsets) {
            if token == "[UNK]" {
                continue;
            }
            let surface = token.strip_prefix("##").unwrap_or(token);
            assert_eq!(
                &text[start..end],
                surface,
                "cased span ({start}, {end}) of {text:?} must be exactly {surface:?}, \
                 got {:?} (tokens {tokens:?}, offsets {offsets:?})",
                &text[start..end]
            );
        }
    }
}

/// Continuation pieces must tile their word: `##` subwords are adjacent to the
/// piece before them, with no gap and no overlap.
#[test]
fn wordpiece_continuation_spans_tile_their_word_without_gaps() {
    let tokenizer = WordPieceTokenizer::new(wordpiece_vocab(), true);
    let text = "cafe world";

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);
    assert_eq!(
        tokens,
        vec!["ca".to_string(), "##fe".to_string(), "world".to_string()]
    );

    for (index, token) in tokens.iter().enumerate() {
        if token.starts_with("##") {
            let previous = offsets[index - 1];
            assert_eq!(
                previous.1,
                offsets[index].0,
                "continuation {token:?} must start exactly where {:?} ended",
                tokens[index - 1]
            );
        }
    }
    assert_eq!(offsets, vec![(0, 2), (2, 4), (5, 10)]);
    assert_eq!(&text[0..2], "ca");
    assert_eq!(&text[2..4], "fe");
    assert_eq!(&text[5..10], "world");
}

// ------------------------------------------------------------ BPE tightness

/// Byte-level BPE loses nothing: its spans, with the duplicates that
/// sub-character byte fragments share collapsed, tile the input exactly from 0
/// to `len` with no gap and no overlap.
///
/// This is the strongest statement available for this encoder and a collapsed
/// alignment fails it immediately: one whole-input span cannot tile an input
/// that tokenizes to several distinct spans.
#[test]
fn bpe_spans_tile_the_input_exactly() {
    let tokenizer = BPETokenizer::new(byte_level_vocab(), Vec::new());

    for (name, raw) in ADVERSARIAL {
        let text = sentineled(raw);
        let (tokens, offsets) = tokenizer.tokenize_with_offsets(&text);
        assert_eq!(tokens.len(), offsets.len(), "{name}: one span per token");

        // Several tokens may share one span: a multi-byte character is several
        // byte-level tokens, and no proper sub-range of it is sliceable, so
        // each fragment honestly reports the whole character.
        let mut distinct: Vec<(usize, usize)> = Vec::with_capacity(offsets.len());
        for &span in &offsets {
            if distinct.last() != Some(&span) {
                distinct.push(span);
            }
        }

        let mut cursor = 0usize;
        for &(start, end) in &distinct {
            assert_eq!(
                start, cursor,
                "{name}: gap or overlap before span ({start}, {end}) in {text:?} \
                 -> {tokens:?} {offsets:?}"
            );
            assert!(
                end > start,
                "{name}: empty span ({start}, {end}) in {text:?} -> {tokens:?}"
            );
            cursor = end;
        }
        assert_eq!(
            cursor,
            text.len(),
            "{name}: spans must reach the end of {text:?} -> {tokens:?} {offsets:?}"
        );

        // Tiling alone is not enough: one whole-input span tiles the input
        // too. Pin the sentinels, which no collapse can survive. Without
        // merges every ASCII byte is its own token, so the four sentinel
        // bytes are the first two and last two spans.
        assert_eq!(
            (offsets[0], offsets[1]),
            ((0, 1), (1, 2)),
            "{name}: leading sentinel bytes widened — alignment collapsed \
             ({text:?} -> {tokens:?} {offsets:?})"
        );
        assert_eq!(
            (offsets[offsets.len() - 2], offsets[offsets.len() - 1]),
            (
                (text.len() - 2, text.len() - 1),
                (text.len() - 1, text.len())
            ),
            "{name}: trailing sentinel bytes widened — alignment collapsed \
             ({text:?} -> {tokens:?} {offsets:?})"
        );
    }
}

/// On input that byte-level BPE's NFC stage leaves untouched, every token's
/// bytes are exactly the bytes of its span — the exact-equality form of the
/// containment property the sibling suite asserts.
#[test]
fn bpe_spans_are_exactly_the_token_bytes_when_normalization_is_identity() {
    let tokenizer = BPETokenizer::new(byte_level_vocab(), Vec::new());

    for text in ["hello world", "世界 hello", "🙂 ok", "café already nfc"] {
        let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

        // Group the tokens that share a span and compare the group's bytes
        // against the span's bytes.
        let mut index = 0usize;
        while index < offsets.len() {
            let span = offsets[index];
            let mut group_end = index;
            while group_end < offsets.len() && offsets[group_end] == span {
                group_end += 1;
            }

            let group_bytes: Vec<u8> = tokens[index..group_end]
                .iter()
                .flat_map(|token| token.chars().filter_map(unicode_to_byte))
                .collect();
            assert_eq!(
                group_bytes,
                text.as_bytes()[span.0..span.1].to_vec(),
                "tokens {:?} must encode exactly {:?} of {text:?}",
                &tokens[index..group_end],
                &text[span.0..span.1]
            );

            index = group_end;
        }
    }
}
