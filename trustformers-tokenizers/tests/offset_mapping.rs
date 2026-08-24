//! Offset-mapping contract for the two encoders that populate it.
//!
//! `WordPieceTokenizer` and `BPETokenizer` both report, for every token, the
//! **byte** range of the caller's original (pre-normalization) string that the
//! token came from. These tests pin that contract as a round-trip property —
//! `&original[start..end]` really is the text the token covers — on ASCII,
//! accented Latin (precomposed *and* decomposed), CJK and emoji input, for
//! single sequences, sequence pairs, truncation and padding.
//!
//! See `trustformers_tokenizers::offsets` for the byte- (not character-)
//! offset convention and for `byte_offsets_to_char_offsets`, the conversion a
//! Python consumer needs; the last test here exercises it end to end.

use std::collections::HashMap;
use trustformers_core::traits::{TokenizedInput, Tokenizer};
use trustformers_tokenizers::bpe::{unicode_to_byte, BPETokenizer};
use trustformers_tokenizers::offsets::byte_offsets_to_char_offsets;
use trustformers_tokenizers::parallel::{BatchTokenizer, PaddingSide};
use trustformers_tokenizers::wordpiece::WordPieceTokenizer;

// ---------------------------------------------------------------- fixtures

/// A BERT-like vocabulary covering every fixture used below.
fn wordpiece_vocab() -> HashMap<String, u32> {
    let tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "hi", "un", "##want",
        "##ed", "run", "##ning", "世", "界", "ca", "##fe", "🙂", ",", "!", ".", "the", "quick",
        "brown", "fox",
    ];
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| ((*token).to_string(), index as u32))
        .collect()
}

fn wordpiece(do_lower_case: bool) -> WordPieceTokenizer {
    WordPieceTokenizer::new(wordpiece_vocab(), do_lower_case)
}

/// A byte-level BPE vocabulary: every single-byte symbol, so `encode` maps
/// tokens to real ids instead of falling back to the unknown token.
fn byte_level_vocab() -> HashMap<String, u32> {
    (0..=255u8)
        .map(|byte| {
            let symbol: String =
                std::char::from_u32(byte_symbol(byte)).into_iter().collect::<String>();
            (symbol, byte as u32)
        })
        .collect()
}

/// The GPT-2 `bytes_to_unicode` code point for `byte`, derived from the
/// crate's own (tested) inverse table so the fixture cannot drift from it.
fn byte_symbol(byte: u8) -> u32 {
    for code_point in 0u32..512 {
        if let Some(candidate) = std::char::from_u32(code_point) {
            if unicode_to_byte(candidate) == Some(byte) {
                return code_point;
            }
        }
    }
    unreachable!("the byte-level alphabet covers every byte")
}

fn bpe(merges: Vec<(String, String)>) -> BPETokenizer {
    BPETokenizer::new(byte_level_vocab(), merges)
}

/// The raw bytes a byte-level BPE token encodes.
fn token_bytes(token: &str) -> Vec<u8> {
    token.chars().filter_map(unicode_to_byte).collect()
}

// -------------------------------------------------------- shared assertions

/// Every span must be an ordered, in-range, character-aligned slice of `text`.
fn assert_spans_slice_the_text(text: &str, offsets: &[(usize, usize)]) {
    for &(start, end) in offsets {
        assert!(start <= end, "span ({start}, {end}) runs backwards");
        assert!(
            end <= text.len(),
            "span ({start}, {end}) is past the end of {text:?}"
        );
        assert!(
            text.is_char_boundary(start),
            "span start {start} splits a character of {text:?}"
        );
        assert!(
            text.is_char_boundary(end),
            "span end {end} splits a character of {text:?}"
        );
        // The load-bearing property: this must never panic.
        let _ = &text[start..end];
    }
}

/// Every byte-level token's bytes must occur inside the source range it
/// reports. (Only meaningful when normalization did not rewrite the text.)
fn assert_token_bytes_lie_inside_their_span(
    text: &str,
    tokens: &[String],
    offsets: &[(usize, usize)],
) {
    assert_eq!(tokens.len(), offsets.len(), "one span per token");
    for (token, &(start, end)) in tokens.iter().zip(offsets) {
        let bytes = token_bytes(token);
        if bytes.is_empty() {
            continue;
        }
        let slice = &text.as_bytes()[start..end];
        assert!(
            slice.windows(bytes.len()).any(|window| window == bytes.as_slice()),
            "token {token:?} encodes {bytes:?}, which does not occur in its reported span \
             ({start}, {end}) = {:?} of {text:?}",
            &text[start..end]
        );
    }
}

/// The tokens must, together, still encode the whole (normalized) input.
fn assert_tokens_reconstruct(text: &str, tokens: &[String]) {
    let rebuilt: Vec<u8> = tokens.iter().flat_map(|token| token_bytes(token)).collect();
    assert_eq!(
        String::from_utf8(rebuilt).as_deref(),
        Ok(text),
        "byte-level tokens must reconstruct the input"
    );
}

fn offsets_of(encoded: &TokenizedInput) -> Vec<(usize, usize)> {
    encoded
        .offset_mapping
        .clone()
        .expect("this encoder must populate offset_mapping, never leave it None")
}

// ------------------------------------------------------------- BPE: offsets

#[test]
fn bpe_offsets_round_trip_ascii() {
    // "l"+"l" is a real merge, so the pieces of "Hello" have unequal lengths
    // and a uniform split of the word would be visibly wrong.
    let tokenizer = bpe(vec![("l".to_string(), "l".to_string())]);
    let text = "Hello World";

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_spans_slice_the_text(text, &offsets);
    assert_tokens_reconstruct(text, &tokens);
    // Pure ASCII: every span is byte-exact, so the round trip is an equality.
    for (token, &(start, end)) in tokens.iter().zip(&offsets) {
        assert_eq!(
            text.as_bytes()[start..end].to_vec(),
            token_bytes(token),
            "token {token:?} must cover exactly the bytes it encodes"
        );
    }
    assert_eq!(offsets.first().map(|span| span.0), Some(0));
    assert_eq!(offsets.last().map(|span| span.1), Some(text.len()));
}

#[test]
fn bpe_offsets_round_trip_accented() {
    let tokenizer = bpe(Vec::new());
    let text = "café au lait"; // precomposed é: already NFC

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_spans_slice_the_text(text, &offsets);
    assert_tokens_reconstruct(text, &tokens);
    assert_token_bytes_lie_inside_their_span(text, &tokens, &offsets);

    // Both byte-level symbols of "é" report the whole character rather than a
    // half of it, and neither drifts past it.
    let accent_start = text.find('é').expect("fixture contains é");
    let accent_end = accent_start + 'é'.len_utf8();
    let accent_spans: Vec<(usize, usize)> = offsets
        .iter()
        .copied()
        .filter(|&(start, end)| start >= accent_start && end <= accent_end && start < end)
        .collect();
    assert_eq!(
        accent_spans,
        vec![(accent_start, accent_end), (accent_start, accent_end)]
    );
}

#[test]
fn bpe_offsets_round_trip_cjk() {
    let tokenizer = bpe(Vec::new());
    let text = "世界 hello";

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_spans_slice_the_text(text, &offsets);
    assert_tokens_reconstruct(text, &tokens);
    assert_token_bytes_lie_inside_their_span(text, &tokens, &offsets);

    // The three symbols of 世 all report exactly 世.
    let world_spans: Vec<(usize, usize)> = offsets
        .iter()
        .copied()
        .filter(|&(start, end)| end <= 3 && start < end)
        .collect();
    assert_eq!(world_spans, vec![(0, 3), (0, 3), (0, 3)]);
    assert_eq!(&text[0..3], "世");
}

#[test]
fn bpe_offsets_round_trip_emoji() {
    let tokenizer = bpe(Vec::new());
    let text = "hi 🙂 there";

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_spans_slice_the_text(text, &offsets);
    assert_tokens_reconstruct(text, &tokens);
    assert_token_bytes_lie_inside_their_span(text, &tokens, &offsets);

    let emoji_start = text.find('🙂').expect("fixture contains the emoji");
    let emoji_end = emoji_start + '🙂'.len_utf8();
    // All four bytes of the emoji report the whole emoji.
    for &(start, end) in &offsets {
        if start >= emoji_start && end <= emoji_end && start < end {
            assert_eq!((start, end), (emoji_start, emoji_end));
        }
    }
}

#[test]
fn bpe_offsets_index_the_original_text_through_nfc_normalization() {
    let tokenizer = bpe(Vec::new());
    // Decomposed: "cafe" + U+0301. NFC composes it to "café", which is one
    // byte shorter, so unaligned offsets would drift off the end.
    let text = "cafe\u{301}";
    assert_eq!(text.len(), 6);
    assert_eq!(tokenizer.normalize_text(text), "café");

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_eq!(tokens.len(), offsets.len());
    assert_spans_slice_the_text(text, &offsets);
    // Offsets index the ORIGINAL six bytes, not the five-byte normalized form.
    assert_eq!(offsets, vec![(0, 1), (1, 2), (2, 3), (3, 6), (3, 6)]);
    // The composed "é" traces back to the "e" plus its combining accent.
    assert_eq!(&text[3..6], "e\u{301}");
    // And the pieces still describe the normalized byte stream.
    assert_tokens_reconstruct("café", &tokens);
}

#[test]
fn bpe_offsets_survive_cjk_padding_and_case_folding() {
    // Both optional normalizers on at once: CJK space padding inserts bytes
    // that have no source at all, and case folding rewrites others.
    let tokenizer =
        BPETokenizer::with_options(byte_level_vocab(), Vec::new(), true, false, true, 100);
    let text = "Ab世";
    assert_eq!(tokenizer.normalize_text(text), "ab 世");

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_eq!(tokens.len(), offsets.len());
    assert_spans_slice_the_text(text, &offsets);
    // The inserted separator has no source, so it reports an empty span rather
    // than claiming a character it did not come from.
    assert!(
        offsets.iter().any(|&(start, end)| start == end),
        "the inserted padding space must report an empty span, got {offsets:?}"
    );
    // Every span still lands inside the original text, in order.
    let mut previous_end = 0usize;
    for &(start, end) in &offsets {
        assert!(
            end <= text.len(),
            "span ({start}, {end}) is past the end of {text:?}"
        );
        assert!(
            start >= previous_end,
            "spans must not run backwards: {offsets:?}"
        );
        previous_end = start;
    }
    // The three bytes of 世 all point back at 世 in the ORIGINAL string.
    let cjk_start = text.find('世').expect("fixture contains 世");
    let cjk_spans: Vec<(usize, usize)> = offsets
        .iter()
        .copied()
        .filter(|&(start, end)| start >= cjk_start && start < end)
        .collect();
    assert_eq!(cjk_spans, vec![(cjk_start, text.len()); 3]);
}

#[test]
fn bpe_tokenize_matches_tokenize_with_offsets() {
    let plain = bpe(vec![("l".to_string(), "l".to_string())]);
    let folding =
        BPETokenizer::with_options(byte_level_vocab(), Vec::new(), true, false, true, 100);
    let raw = BPETokenizer::with_options(byte_level_vocab(), Vec::new(), false, true, false, 100);

    let fixtures = [
        "",
        "Hello World",
        "cafe\u{301}",
        "café",
        "世界 hi",
        "🙂!",
        "  spaced  ",
    ];
    for tokenizer in [&plain, &folding, &raw] {
        for text in fixtures {
            let (with_offsets, offsets) = tokenizer.tokenize_with_offsets(text);
            assert_eq!(
                tokenizer.tokenize(text),
                with_offsets,
                "the offset path must emit the same tokens as the plain path for {text:?}"
            );
            assert_eq!(
                with_offsets.len(),
                offsets.len(),
                "one span per token for {text:?}"
            );
            assert_spans_slice_the_text(text, &offsets);
        }
    }
}

#[test]
fn bpe_long_pre_token_chunking_still_reports_offsets() {
    // `max_input_chars_per_word = 4` forces the chunking branch of `bpe()`.
    let tokenizer =
        BPETokenizer::with_options(byte_level_vocab(), Vec::new(), true, true, false, 4);
    let text = "abcdefghijkl";

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_eq!(tokens.len(), text.len(), "one byte-level symbol per byte");
    assert_spans_slice_the_text(text, &offsets);
    assert_tokens_reconstruct(text, &tokens);
    for (index, &(start, end)) in offsets.iter().enumerate() {
        assert_eq!(
            (start, end),
            (index, index + 1),
            "chunking must not shift the cursor"
        );
    }
}

#[test]
fn bpe_encode_populates_offset_mapping() {
    let tokenizer = bpe(Vec::new());
    let text = "héllo";

    let encoded = tokenizer.encode(text).expect("encoding must succeed");
    let offsets = offsets_of(&encoded);

    assert_eq!(
        offsets.len(),
        encoded.input_ids.len(),
        "one span per token id"
    );
    assert_spans_slice_the_text(text, &offsets);
}

#[test]
fn bpe_encode_of_empty_text_reports_an_empty_mapping() {
    let tokenizer = bpe(Vec::new());
    let encoded = tokenizer.encode("").expect("encoding must succeed");
    assert!(encoded.input_ids.is_empty());
    assert_eq!(offsets_of(&encoded), Vec::new());
}

#[test]
fn bpe_pair_offsets_index_the_documented_combined_string() {
    let tokenizer = bpe(Vec::new());
    let combined = BPETokenizer::pair_input_text("héllo", "wörld");
    assert_eq!(combined, "héllo wörld");

    let encoded = tokenizer.encode_pair("héllo", "wörld").expect("pair encoding must succeed");
    let offsets = offsets_of(&encoded);

    assert_eq!(offsets.len(), encoded.input_ids.len());
    // The documented coordinate space is the combined string, and slicing it
    // by these offsets is what a caller is supposed to do.
    assert_spans_slice_the_text(&combined, &offsets);
    let (tokens, direct) = tokenizer.tokenize_with_offsets(&combined);
    assert_eq!(
        offsets, direct,
        "pair offsets must be the combined string's offsets"
    );
    assert_token_bytes_lie_inside_their_span(&combined, &tokens, &offsets);
}

// ------------------------------------------------------ WordPiece: offsets

#[test]
fn wordpiece_offsets_round_trip_ascii() {
    let tokenizer = wordpiece(true);
    let text = "the quick brown fox";

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_eq!(tokens.len(), offsets.len());
    assert_spans_slice_the_text(text, &offsets);
    for (token, &(start, end)) in tokens.iter().zip(&offsets) {
        assert_eq!(
            &text[start..end],
            token,
            "an ASCII token must be its own source text"
        );
    }
    assert_eq!(offsets, vec![(0, 3), (4, 9), (10, 15), (16, 19)]);
}

#[test]
fn wordpiece_continuation_pieces_report_their_own_subspan() {
    let tokenizer = wordpiece(true);
    let text = "unwanted running";

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_eq!(
        tokens,
        vec!["un", "##want", "##ed", "run", "##ning"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
    );
    // Each `##` piece points at its own slice of the word, not at the whole
    // word and not at the previous piece.
    assert_eq!(offsets, vec![(0, 2), (2, 6), (6, 8), (9, 12), (12, 16)]);
    for (token, &(start, end)) in tokens.iter().zip(&offsets) {
        let surface = token.strip_prefix("##").unwrap_or(token);
        assert_eq!(&text[start..end], surface);
    }
}

#[test]
fn wordpiece_offsets_index_the_original_text_through_accent_stripping() {
    let tokenizer = wordpiece(true);
    // Lowercasing + accent stripping turns "CAFÉ" into "cafe", which is one
    // byte shorter than the input.
    let text = "CAFÉ";
    assert_eq!(text.len(), 5);

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_eq!(tokens, vec!["ca".to_string(), "##fe".to_string()]);
    assert_eq!(offsets, vec![(0, 2), (2, 5)]);
    assert_spans_slice_the_text(text, &offsets);
    // The round trip for a normalizing tokenizer: the span is the *original*
    // text, and normalizing it yields the token.
    assert_eq!(&text[2..5], "FÉ");
    assert_eq!(
        WordPieceTokenizer::strip_accents(&text[2..5]).to_lowercase(),
        "fe"
    );
}

#[test]
fn wordpiece_offsets_index_the_original_text_through_decomposed_accents() {
    let tokenizer = wordpiece(true);
    // The same word written with a combining accent: six bytes in, four out.
    let text = "CAFE\u{301}";
    assert_eq!(text.len(), 6);

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_eq!(tokens, vec!["ca".to_string(), "##fe".to_string()]);
    assert_eq!(offsets, vec![(0, 2), (2, 6)]);
    assert_spans_slice_the_text(text, &offsets);
    assert_eq!(&text[2..6], "FE\u{301}");
    assert_eq!(
        WordPieceTokenizer::strip_accents(&text[2..6]).to_lowercase(),
        "fe"
    );
}

#[test]
fn wordpiece_offsets_for_cjk_survive_the_space_padding_stage() {
    let tokenizer = wordpiece(true);
    let text = "世界";

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_eq!(tokens, vec!["世".to_string(), "界".to_string()]);
    // The padding spaces BERT inserts around CJK characters must not shift the
    // reported positions.
    assert_eq!(offsets, vec![(0, 3), (3, 6)]);
    for (token, &(start, end)) in tokens.iter().zip(&offsets) {
        assert_eq!(&text[start..end], token);
    }
}

#[test]
fn wordpiece_offsets_for_emoji() {
    let tokenizer = wordpiece(true);
    let text = "hi 🙂";

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_eq!(tokens, vec!["hi".to_string(), "🙂".to_string()]);
    assert_eq!(offsets, vec![(0, 2), (3, 7)]);
    assert_eq!(&text[3..7], "🙂");
}

#[test]
fn wordpiece_unknown_word_spans_the_whole_word() {
    let tokenizer = wordpiece(true);
    let text = "hello zzz";

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_eq!(tokens, vec!["hello".to_string(), "[UNK]".to_string()]);
    // A word no segmentation covers collapses to one unknown token, which
    // reports the whole word it stood for.
    assert_eq!(offsets, vec![(0, 5), (6, 9)]);
    assert_eq!(&text[6..9], "zzz");
}

#[test]
fn wordpiece_overlong_word_unknown_spans_the_whole_word() {
    let tokenizer = wordpiece(true);
    // `max_input_chars_per_word` is 100; a longer word takes the other
    // unknown-token path.
    let long_word = "a".repeat(150);
    let text = format!("hello {long_word}");

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(&text);

    assert_eq!(tokens, vec!["hello".to_string(), "[UNK]".to_string()]);
    assert_eq!(offsets, vec![(0, 5), (6, 156)]);
    assert_eq!(&text[6..156], long_word);
}

#[test]
fn wordpiece_span_of_a_word_containing_a_dropped_control_character_contains_it() {
    let tokenizer = wordpiece(true);
    // BERT's `clean_text` deletes NUL outright. A single `(start, end)` pair
    // cannot express a hole, so the span still contains the deleted byte —
    // documented on `tokenize_with_offsets`, and pinned here.
    let text = "he\u{0}llo";
    assert_eq!(text.len(), 6);

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_eq!(tokens, vec!["hello".to_string()]);
    assert_eq!(offsets, vec![(0, 6)]);
    assert_eq!(&text[0..6], "he\u{0}llo");
    assert_spans_slice_the_text(text, &offsets);
}

#[test]
fn wordpiece_encode_marks_special_tokens_with_empty_spans() {
    let tokenizer = wordpiece(true);
    let text = "hello world";

    let encoded = tokenizer.encode(text).expect("encoding must succeed");
    let offsets = offsets_of(&encoded);

    assert_eq!(offsets.len(), encoded.input_ids.len());
    assert_eq!(offsets, vec![(0, 0), (0, 5), (6, 11), (0, 0)]);
    let special = encoded.special_tokens_mask.expect("wordpiece populates the special mask");
    for (index, &is_special) in special.iter().enumerate() {
        if is_special == 1 {
            assert_eq!(offsets[index], (0, 0), "special tokens carry an empty span");
        } else {
            assert!(
                offsets[index].0 < offsets[index].1,
                "content tokens carry a real span"
            );
        }
    }
}

#[test]
fn wordpiece_encode_of_empty_text_is_two_specials_with_empty_spans() {
    let tokenizer = wordpiece(true);
    let encoded = tokenizer.encode("").expect("encoding must succeed");
    assert_eq!(encoded.input_ids.len(), 2);
    assert_eq!(offsets_of(&encoded), vec![(0, 0), (0, 0)]);
}

#[test]
fn wordpiece_pair_offsets_are_per_sequence() {
    let tokenizer = wordpiece(true);
    let first = "CAFÉ";
    let second = "unwanted";

    let encoded = tokenizer.encode_pair(first, second).expect("pair encoding must succeed");
    let offsets = offsets_of(&encoded);
    let types = encoded.token_type_ids.clone().expect("wordpiece populates token_type_ids");

    assert_eq!(offsets.len(), encoded.input_ids.len());
    assert_eq!(types.len(), encoded.input_ids.len());
    let special = encoded.special_tokens_mask.clone().expect("wordpiece populates the mask");

    // Segment 0 spans index `first`; segment 1 spans index `second`. There is
    // no combined coordinate space, exactly as HuggingFace reports pairs.
    for (index, &(start, end)) in offsets.iter().enumerate() {
        if special[index] == 1 {
            assert_eq!((start, end), (0, 0));
            continue;
        }
        let sequence = if types[index] == 0 { first } else { second };
        assert!(
            end <= sequence.len(),
            "span ({start}, {end}) must index its own sequence"
        );
        let _ = &sequence[start..end];
    }

    // Concretely: [CLS] ca ##fe [SEP] un ##want ##ed [SEP].
    assert_eq!(
        offsets,
        vec![
            (0, 0),
            (0, 2),
            (2, 5),
            (0, 0),
            (0, 2),
            (2, 6),
            (6, 8),
            (0, 0)
        ]
    );
    assert_eq!(types, vec![0, 0, 0, 0, 1, 1, 1, 1]);
}

#[test]
fn wordpiece_cased_tokenizer_reports_exact_surface_spans() {
    let mut vocab = wordpiece_vocab();
    vocab.insert("CAFÉ".to_string(), 900);
    let tokenizer = WordPieceTokenizer::new(vocab, false);
    let text = "CAFÉ";

    let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

    assert_eq!(tokens, vec!["CAFÉ".to_string()]);
    assert_eq!(offsets, vec![(0, 5)]);
    // A cased tokenizer performs no normalization, so the span is exactly the
    // token's surface text.
    assert_eq!(&text[0..5], "CAFÉ");
}

// ------------------------------------------- truncation / padding threading

#[test]
fn wordpiece_batch_truncation_keeps_offsets_in_lockstep() {
    let batch = BatchTokenizer::new(wordpiece(true)).with_max_length(3).with_truncation();

    let encoded = batch
        .encode_batch_padded(&["hello world"])
        .expect("batch encoding must succeed");
    let offsets = encoded.offset_mapping.clone().expect("offsets survive truncation");

    assert_eq!(encoded.input_ids[0].len(), 3);
    assert_eq!(offsets[0].len(), encoded.input_ids[0].len());
    // Truncation drops "world" but keeps [CLS]/[SEP], and drops that token's
    // span with it.
    assert_eq!(offsets[0], vec![(0, 0), (0, 5), (0, 0)]);
}

#[test]
fn wordpiece_batch_padding_appends_empty_spans_on_the_right() {
    let batch = BatchTokenizer::new(wordpiece(true)).with_padding(0);

    let encoded = batch
        .encode_batch_padded(&["hello world", "hello"])
        .expect("batch encoding must succeed");
    let offsets = encoded.offset_mapping.clone().expect("offsets survive padding");

    assert_eq!(encoded.input_ids[0].len(), encoded.input_ids[1].len());
    assert_eq!(offsets[1].len(), encoded.input_ids[1].len());
    assert_eq!(offsets[0], vec![(0, 0), (0, 5), (6, 11), (0, 0)]);
    // Padding positions carry the same empty span special tokens do.
    assert_eq!(offsets[1], vec![(0, 0), (0, 5), (0, 0), (0, 0)]);
    assert_eq!(*encoded.attention_mask[1].last().expect("padded row"), 0u8);
}

#[test]
fn wordpiece_batch_padding_prepends_empty_spans_on_the_left() {
    let batch = BatchTokenizer::new(wordpiece(true))
        .with_padding(0)
        .with_padding_side(PaddingSide::Left);

    let encoded = batch
        .encode_batch_padded(&["hello world", "hello"])
        .expect("batch encoding must succeed");
    let offsets = encoded.offset_mapping.clone().expect("offsets survive padding");

    assert_eq!(offsets[1], vec![(0, 0), (0, 0), (0, 5), (0, 0)]);
    assert_eq!(encoded.attention_mask[1][0], 0u8);
}

#[test]
fn bpe_batch_truncation_and_padding_keep_offsets_in_lockstep() {
    let batch = BatchTokenizer::new(bpe(Vec::new()))
        .with_max_length(3)
        .with_truncation()
        .with_padding(0);

    let encoded = batch
        .encode_batch_padded(&["abcdef", "ab"])
        .expect("batch encoding must succeed");
    let offsets = encoded.offset_mapping.clone().expect("offsets survive truncation + padding");

    for (row, row_offsets) in encoded.input_ids.iter().zip(&offsets) {
        assert_eq!(
            row.len(),
            row_offsets.len(),
            "offsets must stay the length of the ids"
        );
    }
    // Truncated row: the first three bytes, with their real spans.
    assert_eq!(offsets[0], vec![(0, 1), (1, 2), (2, 3)]);
    // Padded row: two real spans then an empty one.
    assert_eq!(offsets[1], vec![(0, 1), (1, 2), (0, 0)]);
}

#[test]
fn wordpiece_pair_batch_truncation_keeps_offsets_in_lockstep() {
    let batch = BatchTokenizer::new(wordpiece(true)).with_max_length(6).with_truncation();

    let encoded = batch
        .encode_pair_batch_padded(&[("hello world", "unwanted")])
        .expect("pair batch encoding must succeed");
    let offsets = encoded.offset_mapping.clone().expect("offsets survive pair truncation");

    assert_eq!(encoded.input_ids[0].len(), 6);
    assert_eq!(offsets[0].len(), encoded.input_ids[0].len());
    for &(start, end) in &offsets[0] {
        assert!(start <= end, "truncation must not corrupt a span");
    }
    // The three specials keep their empty spans through truncation.
    assert_eq!(offsets[0].iter().filter(|&&span| span == (0, 0)).count(), 3);
}

// ------------------------------------------------- the Python-facing bridge

#[test]
fn encode_offsets_convert_to_python_style_character_offsets() {
    let tokenizer = wordpiece(true);
    let text = "hi 🙂";

    let encoded = tokenizer.encode(text).expect("encoding must succeed");
    let byte_offsets = offsets_of(&encoded);
    let char_offsets = byte_offsets_to_char_offsets(text, &byte_offsets)
        .expect("every span produced by the encoder is on a character boundary");

    assert_eq!(char_offsets.len(), byte_offsets.len());
    // What a Python caller would get: `text[start:end]` over code points.
    let characters: Vec<char> = text.chars().collect();
    for (&(byte_start, byte_end), &(char_start, char_end)) in byte_offsets.iter().zip(&char_offsets)
    {
        let by_bytes = &text[byte_start..byte_end];
        let by_chars: String = characters[char_start..char_end].iter().collect();
        assert_eq!(
            by_bytes, by_chars,
            "both conventions must name the same substring"
        );
    }
    // The emoji is four bytes but one character, which is exactly the
    // divergence the conversion exists for.
    assert_eq!(byte_offsets, vec![(0, 0), (0, 2), (3, 7), (0, 0)]);
    assert_eq!(char_offsets, vec![(0, 0), (0, 2), (3, 4), (0, 0)]);
}

#[test]
fn every_bpe_and_wordpiece_encode_path_populates_offsets() {
    // A guard against a future edit quietly returning to `offset_mapping: None`
    // on any of the four encode entry points.
    let word_piece = wordpiece(true);
    let byte_level = bpe(Vec::new());

    for encoded in [
        word_piece.encode("hello world").expect("encode"),
        word_piece.encode_pair("hello", "world").expect("encode_pair"),
        byte_level.encode("hello world").expect("encode"),
        byte_level.encode_pair("hello", "world").expect("encode_pair"),
    ] {
        let offsets = offsets_of(&encoded);
        assert_eq!(
            offsets.len(),
            encoded.input_ids.len(),
            "offset_mapping must stay the same length as input_ids"
        );
    }
}
