//! Cross-checks of the hand-rolled tokenizers against the HuggingFace
//! `tokenizers` crate, which this workspace already depends on.
//!
//! Each test builds one `tokenizer.json` and one hand-rolled tokenizer from the
//! *same* vocabulary/merges, then asserts both produce identical token ids for a
//! fixed corpus. These are the tests that catch a segmentation regression: a
//! tokenizer that silently drops case, glues punctuation to words, keeps partial
//! sub-tokens, or ignores merge ranks cannot pass them.

use std::collections::HashMap;
use trustformers_core::traits::Tokenizer;
use trustformers_tokenizers::bpe::BPETokenizer;
use trustformers_tokenizers::tokenizer::TokenizerImpl;
use trustformers_tokenizers::wordpiece::WordPieceTokenizer;

fn vocab_from(tokens: &[&str]) -> HashMap<String, u32> {
    tokens
        .iter()
        .enumerate()
        .map(|(id, token)| ((*token).to_string(), id as u32))
        .collect()
}

fn vocab_json(vocab: &HashMap<String, u32>) -> String {
    let mut entries: Vec<(&String, &u32)> = vocab.iter().collect();
    entries.sort_by_key(|&(_, id)| *id);
    let body: Vec<String> = entries
        .iter()
        .map(|(token, id)| format!("{}: {}", serde_json::json!(token), id))
        .collect();
    format!("{{{}}}", body.join(", "))
}

/// Byte-level BPE: identical vocabulary and merges must give identical ids.
#[test]
fn byte_level_bpe_matches_huggingface() {
    let tokens = [
        "H",
        "e",
        "l",
        "o",
        "ll",
        "llo",
        "\u{0120}",
        "w",
        "\u{0120}w",
        "r",
        "d",
        "!",
        ",",
        "\u{0120}W",
        "W",
    ];
    let vocab = vocab_from(&tokens);
    let merges = vec![
        ("l".to_string(), "l".to_string()),
        ("ll".to_string(), "o".to_string()),
        ("\u{0120}".to_string(), "w".to_string()),
        ("\u{0120}".to_string(), "W".to_string()),
    ];

    let json = format!(
        r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {{
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            }},
            "post_processor": null,
            "decoder": {{
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            }},
            "model": {{
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": {vocab},
                "merges": ["l l", "ll o", "Ġ w", "Ġ W"]
            }}
        }}"#,
        vocab = vocab_json(&vocab)
    );

    let reference =
        TokenizerImpl::from_tokenizer_json(&json).expect("the reference tokenizer.json must parse");
    let ours = BPETokenizer::new(vocab, merges);

    for text in [
        "Hello world",
        "Hello World",
        "H e llo",
        "world!",
        "Hello, world",
    ] {
        let expected = reference.encode(text).expect("reference encoding must succeed");
        let actual = ours.encode(text).expect("encoding must succeed");
        assert_eq!(
            actual.input_ids, expected.input_ids,
            "byte-level BPE ids diverge from HuggingFace for {:?}",
            text
        );
    }
}

/// WordPiece: BERT normalization + greedy longest-match must match HuggingFace.
#[test]
fn wordpiece_matches_huggingface() {
    let tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", ",", "!", "un", "##want",
        "##ed", "run", "##ning", "cafe", "世", "界",
    ];
    let vocab = vocab_from(&tokens);

    let json = format!(
        r###"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": {{
                "type": "BertNormalizer",
                "clean_text": true,
                "handle_chinese_chars": true,
                "strip_accents": true,
                "lowercase": true
            }},
            "pre_tokenizer": {{"type": "BertPreTokenizer"}},
            "post_processor": null,
            "decoder": {{
                "type": "WordPiece",
                "prefix": "##",
                "cleanup": true
            }},
            "model": {{
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {vocab}
            }}
        }}"###,
        vocab = vocab_json(&vocab)
    );

    let reference =
        TokenizerImpl::from_tokenizer_json(&json).expect("the reference tokenizer.json must parse");
    let ours = WordPieceTokenizer::new(vocab.clone(), true);

    for text in [
        "hello, world!",
        "unwanted running",
        "CAFÉ",
        // Decomposed form: pins the accent-strip/lowercase ORDER against HF.
        "CAFE\u{301}",
        "世界",
        "unknown word",
        "HELLO   world",
    ] {
        let expected = reference.encode(text).expect("reference encoding must succeed");

        let actual_ids: Vec<u32> = ours
            .tokenize(text)
            .iter()
            .map(|token| {
                vocab
                    .get(token)
                    .copied()
                    .unwrap_or_else(|| vocab.get("[UNK]").copied().unwrap_or(1))
            })
            .collect();

        assert_eq!(
            actual_ids,
            expected.input_ids,
            "WordPiece ids diverge from HuggingFace for {:?} (ours: {:?})",
            text,
            ours.tokenize(text)
        );
    }
}

/// Unigram: the Viterbi lattice must pick the same path as HuggingFace.
///
/// Both sides run over the identical raw string (no pre-tokenizer, no
/// normalizer), so this compares the segmentation search itself — including the
/// `min_score - 10.0` unknown-character penalty both implementations use.
#[test]
fn unigram_viterbi_matches_huggingface() {
    use trustformers_tokenizers::unigram::UnigramTokenizer;

    // Deliberately laid out so greedy longest-match and greedy
    // highest-score-at-position both lose to the true Viterbi path.
    let pieces: [(&str, f64); 9] = [
        ("<unk>", 0.0),
        ("a", -1.0),
        ("ab", -1.2),
        ("b", -3.0),
        ("c", -3.0),
        ("cd", -1.0),
        ("bcd", -9.0),
        ("d", -3.0),
        ("abcd", -5.0),
    ];

    let vocab_entries: Vec<String> = pieces
        .iter()
        .map(|(token, score)| format!("[{}, {}]", serde_json::json!(token), score))
        .collect();
    let json = format!(
        r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {{
                "type": "Unigram",
                "unk_id": 0,
                "byte_fallback": false,
                "vocab": [{vocab}]
            }}
        }}"#,
        vocab = vocab_entries.join(", ")
    );

    let reference =
        TokenizerImpl::from_tokenizer_json(&json).expect("the reference tokenizer.json must parse");

    let vocab: HashMap<String, u32> = pieces
        .iter()
        .enumerate()
        .map(|(id, (token, _))| ((*token).to_string(), id as u32))
        .collect();
    let scores: HashMap<String, f32> = pieces
        .iter()
        .map(|(token, score)| ((*token).to_string(), *score as f32))
        .collect();
    let ours = UnigramTokenizer::new(vocab, scores).expect("construction must succeed");

    for text in ["abcd", "abc", "dcba", "aXb", "d"] {
        let expected = reference.encode(text).expect("reference encoding must succeed");
        let actual = ours.encode(text).expect("encoding must succeed");
        assert_eq!(
            actual.input_ids, expected.input_ids,
            "Unigram ids diverge from HuggingFace for {:?}",
            text
        );
    }

    // Pin the best path explicitly: "ab" + "cd" (-2.2) beats both the longest
    // piece "abcd" (-5.0) and the greedy "a" + "bcd" (-10.0).
    assert_eq!(
        ours.encode("abcd").expect("encoding must succeed").input_ids,
        vec![2, 5]
    );
}

/// The BPE save/load round-trip must survive a real directory on disk.
#[test]
fn wrapper_round_trip_preserves_encoding() {
    use trustformers_tokenizers::tokenizer::TokenizerWrapper;

    let dir = std::env::temp_dir().join(format!(
        "trustformers_cross_check_round_trip_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);

    let vocab = vocab_from(&["H", "e", "l", "o", "ll", "llo", "\u{0120}", "w", "r", "d"]);
    let merges = vec![
        ("l".to_string(), "l".to_string()),
        ("ll".to_string(), "o".to_string()),
    ];
    let original = TokenizerWrapper::BPE(BPETokenizer::new(vocab, merges));

    let text = "Hello world";
    let expected = original.encode(text).expect("encoding must succeed");

    original.save_pretrained(&dir).expect("saving must succeed");
    let reloaded = TokenizerWrapper::from_pretrained(&dir).expect("reloading must succeed");
    let actual = reloaded.encode(text).expect("encoding must succeed");

    assert_eq!(actual.input_ids, expected.input_ids);
    assert!(actual.input_ids.iter().any(|&id| id != 0));

    let _ = std::fs::remove_dir_all(&dir);
}
