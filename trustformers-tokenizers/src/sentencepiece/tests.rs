use super::proto::encode::{length_delimited, piece, varint_field};
use super::*;

/// Build a tokenizer directly from `(piece, score)` pairs, with normalization
/// disabled so tests drive the segmentation algorithms with exact inputs.
fn tokenizer_with_pieces(pieces: &[(&str, f32)]) -> SentencePieceTokenizer {
    let mut tokenizer =
        SentencePieceTokenizer::new().with_normalization(false).with_dummy_prefix(false);

    for (index, (text, score)) in pieces.iter().enumerate() {
        let id = index as u32;
        tokenizer.vocab.insert((*text).to_string(), id);
        tokenizer.id_to_token.insert(id, (*text).to_string());
        tokenizer.scores.insert(id, *score);
        if *text == "<unk>" {
            tokenizer.unk_token_id = Some(id);
            tokenizer.special_tokens.insert((*text).to_string(), id);
        }
    }

    tokenizer.refresh_stats();
    tokenizer
}

/// A complete in-memory SentencePiece `ModelProto`.
fn protobuf_model_bytes() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(length_delimited(1, &piece("<unk>", 0.0, 1)));
    out.extend(length_delimited(1, &piece("<s>", 0.0, 2)));
    out.extend(length_delimited(1, &piece("</s>", 0.0, 2)));
    out.extend(length_delimited(1, &piece("▁hello", -1.0, 0)));
    out.extend(length_delimited(1, &piece("▁world", -1.5, 0)));
    out.extend(length_delimited(1, &piece("<extra_id_0>", 0.0, 3)));

    let mut trainer = Vec::new();
    trainer.extend(length_delimited(1, b"corpus.txt")); // skipped field
    trainer.extend(varint_field(3, 1)); // model_type = UNIGRAM
    out.extend(length_delimited(2, &trainer));

    let mut normalizer = Vec::new();
    normalizer.extend(length_delimited(1, b"identity"));
    out.extend(length_delimited(3, &normalizer));

    out
}

fn temp_dir_for(test_name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "trustformers_sentencepiece_{}_{}",
        test_name,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir must be creatable");
    dir
}

#[test]
fn test_sentencepiece_tokenizer_creation() {
    let tokenizer = SentencePieceTokenizer::new();
    assert_eq!(tokenizer.vocab_size(), 0);
}

/// Regression: `from_pretrained` used to invent a ~142-entry T5 vocabulary.
#[test]
fn test_from_pretrained_errors_without_model_file() {
    let error = SentencePieceTokenizer::from_pretrained("t5-small")
        .expect_err("a missing .model file must be an error, never a synthetic vocabulary");
    let message = error.to_string();
    assert!(
        message.contains("spiece.model"),
        "error must name the probed paths, got: {}",
        message
    );
}

#[test]
fn test_from_pretrained_loads_text_vocab_fixture() {
    let dir = temp_dir_for("text_fixture");
    let model_path = dir.join("spiece.model");
    std::fs::write(&model_path, "<unk>\t0.0\n▁alpha\t-1.0\n▁beta\t-2.0\n")
        .expect("fixture must be writable");

    let dir_str = dir.to_str().expect("temp path must be UTF-8");
    let tokenizer =
        SentencePieceTokenizer::from_pretrained(dir_str).expect("real fixture must load");

    assert_eq!(tokenizer.token_to_id("▁alpha"), Some(1));
    assert_eq!(tokenizer.vocab_size(), 3);
    // No trace of the removed hardcoded T5 vocabulary.
    assert_eq!(tokenizer.token_to_id("<extra_id_0>"), None);
    assert_eq!(tokenizer.token_to_id("▁Hello"), None);

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_dir(&dir);
}

/// A malformed score must fail loudly: silently defaulting it to `0.0` would
/// make the corrupt piece the *most* probable one in every Viterbi path.
#[test]
fn test_text_vocab_rejects_malformed_scores() {
    let mut tokenizer = SentencePieceTokenizer::new();
    let error = tokenizer
        .load_text_vocab_from_reader("<unk>\t0.0\n▁alpha\tnot-a-number\n".as_bytes())
        .expect_err("a malformed score must be an error");
    let message = error.to_string();
    assert!(
        message.contains("not-a-number"),
        "error must quote the bad field: {}",
        message
    );

    // Nothing was committed: the tokenizer is untouched, not half-populated.
    assert_eq!(tokenizer.vocab_size(), 0);

    // A plain one-token-per-line vocabulary (no score column) still loads.
    let mut plain = SentencePieceTokenizer::new();
    plain
        .load_text_vocab_from_reader("<unk>\n▁alpha\n".as_bytes())
        .expect("a score-less vocabulary must still load");
    assert_eq!(plain.vocab_size(), 2);
    assert_eq!(plain.get_score(1), Some(0.0));
}

/// End-to-end protobuf load: pins the `ModelProto` field numbers.
#[test]
fn test_from_pretrained_loads_protobuf_model() {
    let dir = temp_dir_for("proto_fixture");
    let model_path = dir.join("spiece.model");
    std::fs::write(&model_path, protobuf_model_bytes()).expect("fixture must be writable");

    let dir_str = dir.to_str().expect("temp path must be UTF-8");
    let tokenizer =
        SentencePieceTokenizer::from_pretrained(dir_str).expect("protobuf fixture must load");

    assert_eq!(tokenizer.vocab_size(), 6);
    assert_eq!(tokenizer.unk_token_id(), Some(0));
    assert_eq!(tokenizer.bos_token_id(), Some(1));
    assert_eq!(tokenizer.eos_token_id(), Some(2));
    assert_eq!(tokenizer.model_type_string(), "Unigram");
    assert_eq!(tokenizer.get_score(4), Some(-1.5));

    let encoded = tokenizer.encode("hello world").expect("encoding must succeed");
    assert_eq!(encoded.input_ids, vec![3, 4]);
    assert_eq!(
        tokenizer.decode(&encoded.input_ids).expect("decoding must succeed"),
        "hello world"
    );

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_dir(&dir);
}

/// Regression: `tokenize_unigram` was greedy; Viterbi must find the best path.
#[test]
fn test_unigram_viterbi_beats_greedy() {
    // Greedy takes "a" (-1.0 > -1.2) at position 0 and is then forced into the
    // expensive "bcd" (-9.0). Viterbi finds "ab" + "cd" (-2.2).
    let tokenizer = tokenizer_with_pieces(&[
        ("<unk>", 0.0),
        ("a", -1.0),
        ("ab", -1.2),
        ("cd", -1.0),
        ("bcd", -9.0),
    ]);

    let pieces = tokenizer.tokenize("abcd");
    assert_eq!(pieces, vec!["ab".to_string(), "cd".to_string()]);

    let greedy = vec!["a".to_string(), "bcd".to_string()];
    assert!(
        tokenizer.segmentation_score(&pieces) > tokenizer.segmentation_score(&greedy),
        "Viterbi path {:?} must outscore the greedy path {:?}",
        pieces,
        greedy
    );
    assert!((tokenizer.segmentation_score(&pieces) - (-2.2)).abs() < 1e-5);
    assert!((tokenizer.segmentation_score(&greedy) - (-10.0)).abs() < 1e-5);
}

/// A single out-of-vocabulary character must not destroy the rest of the path.
#[test]
fn test_unigram_keeps_valid_pieces_around_unknown_characters() {
    let tokenizer = tokenizer_with_pieces(&[("<unk>", 0.0), ("ab", -1.0), ("cd", -1.0)]);

    assert_eq!(
        tokenizer.tokenize("abXcd"),
        vec!["ab".to_string(), "X".to_string(), "cd".to_string()]
    );

    let encoded = tokenizer.encode("abXcd").expect("encoding must succeed");
    assert_eq!(encoded.input_ids, vec![1, 0, 2]);
}

/// Longer pieces must win when they are actually more likely.
#[test]
fn test_unigram_prefers_high_probability_long_piece() {
    let tokenizer =
        tokenizer_with_pieces(&[("<unk>", 0.0), ("hello", -1.0), ("he", -5.0), ("llo", -5.0)]);
    assert_eq!(tokenizer.tokenize("hello"), vec!["hello".to_string()]);
}

/// Regression: byte fallback was advertised but only ever returned UNK.
#[test]
fn test_byte_fallback_maps_to_byte_pieces() {
    let tokenizer = tokenizer_with_pieces(&[("<unk>", 0.0), ("<0x48>", -20.0), ("<0x69>", -20.0)])
        .with_byte_fallback(true);

    let encoded = tokenizer.encode("Hi").expect("byte fallback must encode");
    assert_eq!(encoded.input_ids, vec![1, 2]);
    assert_ne!(
        encoded.input_ids,
        vec![0, 0],
        "byte fallback must not degenerate into UNK ids"
    );

    let decoded = tokenizer.decode(&encoded.input_ids).expect("decoding must succeed");
    assert_eq!(decoded, "Hi");
}

#[test]
fn test_byte_fallback_errors_when_byte_pieces_are_missing() {
    let tokenizer =
        tokenizer_with_pieces(&[("<unk>", 0.0), ("<0x48>", -20.0)]).with_byte_fallback(true);

    let error = tokenizer
        .encode("Z")
        .expect_err("a vocabulary without the needed byte piece must fail loudly");
    assert!(
        error.to_string().contains("<0x5A>"),
        "error must name the missing piece"
    );
}

#[test]
fn test_encode_without_unk_piece_errors() {
    let tokenizer = tokenizer_with_pieces(&[("ab", -1.0)]);
    assert!(
        tokenizer.encode("zz").is_err(),
        "a model without <unk> cannot encode out-of-vocabulary text"
    );
}

/// Regression: decode used to drop every `<extra_id_N>` sentinel unconditionally.
#[test]
fn test_decode_can_preserve_sentinels() {
    let dir = temp_dir_for("sentinels");
    let model_path = dir.join("spiece.model");
    std::fs::write(&model_path, protobuf_model_bytes()).expect("fixture must be writable");

    let dir_str = dir.to_str().expect("temp path must be UTF-8");
    let tokenizer = SentencePieceTokenizer::from_pretrained(dir_str).expect("fixture must load");

    // id 3 = "▁hello", id 5 = "<extra_id_0>" (a user-defined piece).
    assert_eq!(
        tokenizer.decode_with_options(&[3, 5], false),
        "hello<extra_id_0>"
    );
    assert_eq!(tokenizer.decode_with_options(&[3, 5], true), "hello");

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_dir(&dir);
}

/// A sentinel that the model never declared special must survive decoding.
#[test]
fn test_decode_keeps_undeclared_sentinels() {
    let tokenizer =
        tokenizer_with_pieces(&[("<unk>", 0.0), ("▁hello", -1.0), ("<extra_id_0>", -1.0)]);
    assert_eq!(
        tokenizer.decode_with_options(&[1, 2], true),
        "hello<extra_id_0>"
    );
}

/// Build a BPE-model tokenizer from `(piece, Option<score>)` pairs.
///
/// `None` deliberately leaves the piece *unscored*, which is exactly what
/// [`SentencePieceTokenizer::load_vocab_from_file`] produces for a plain
/// one-token-per-line vocabulary.
fn bpe_tokenizer_with(pieces: &[(&str, Option<f32>)]) -> SentencePieceTokenizer {
    let mut tokenizer = SentencePieceTokenizer::new()
        .with_model_type(ModelType::Bpe)
        .with_normalization(false)
        .with_dummy_prefix(false);

    for (index, (text, score)) in pieces.iter().enumerate() {
        let id = index as u32;
        tokenizer.vocab.insert((*text).to_string(), id);
        tokenizer.id_to_token.insert(id, (*text).to_string());
        if let Some(score) = score {
            tokenizer.scores.insert(id, *score);
        }
        if *text == "<unk>" {
            tokenizer.unk_token_id = Some(id);
            tokenizer.special_tokens.insert((*text).to_string(), id);
        }
    }

    tokenizer.refresh_stats();
    tokenizer
}

/// SentencePiece BPE merges the highest-scoring adjacent pair first, so the same
/// input must segment differently when the merge scores are swapped.
#[test]
fn test_bpe_model_merges_in_score_order() {
    // "ab" (-1.0) outscores "bc" (-2.0): "abc" -> ["ab", "c"].
    let ab_first = bpe_tokenizer_with(&[
        ("<unk>", Some(0.0)),
        ("a", Some(-8.0)),
        ("b", Some(-8.0)),
        ("c", Some(-8.0)),
        ("ab", Some(-1.0)),
        ("bc", Some(-2.0)),
    ]);
    assert_eq!(
        ab_first.tokenize("abc"),
        vec!["ab".to_string(), "c".to_string()]
    );

    // Swap the two scores and the segmentation flips to ["a", "bc"].
    let bc_first = bpe_tokenizer_with(&[
        ("<unk>", Some(0.0)),
        ("a", Some(-8.0)),
        ("b", Some(-8.0)),
        ("c", Some(-8.0)),
        ("ab", Some(-2.0)),
        ("bc", Some(-1.0)),
    ]);
    assert_eq!(
        bc_first.tokenize("abc"),
        vec!["a".to_string(), "bc".to_string()]
    );
}

/// Ties are broken leftmost, and exactly one pair is merged per iteration before
/// the candidates are recomputed.
#[test]
fn test_bpe_model_merges_one_pair_per_iteration_leftmost_on_ties() {
    let tokenizer =
        bpe_tokenizer_with(&[("<unk>", Some(0.0)), ("a", Some(-8.0)), ("aa", Some(-1.0))]);

    // Both ("a","a") pairs of "aaa" score -1.0; a rightmost tie-break would
    // produce ["a", "aa"] instead.
    assert_eq!(
        tokenizer.tokenize("aaa"),
        vec!["aa".to_string(), "a".to_string()],
        "the leftmost of two equally scored pairs must win"
    );

    // "aaaa" needs two iterations: merge at 0, recompute, then merge at 1.
    assert_eq!(
        tokenizer.tokenize("aaaa"),
        vec!["aa".to_string(), "aa".to_string()]
    );
}

/// Regression: an in-vocabulary piece with no recorded score used to default to
/// `0.0` — the *maximum* log probability — so it beat every genuinely scored
/// merge. It must be charged the unknown score instead.
#[test]
fn test_bpe_model_unscored_piece_does_not_outrank_scored_merges() {
    let tokenizer = bpe_tokenizer_with(&[
        ("<unk>", Some(0.0)),
        ("a", Some(-5.0)),
        ("b", Some(-5.0)),
        ("c", Some(-5.0)),
        ("ab", Some(-5.0)),
        ("bc", None), // present in the vocabulary, but unscored
    ]);

    assert!(
        tokenizer.unk_score() < -5.0,
        "unk_score must sit below every scored piece"
    );
    assert_eq!(
        tokenizer.tokenize("abc"),
        vec!["ab".to_string(), "c".to_string()],
        "an unscored piece must not be treated as the most probable merge"
    );

    // The same rule governs the public ranking helper.
    let ranked = tokenizer.get_tokens_by_score();
    let bc_score = ranked
        .iter()
        .find(|(token, _, _)| token == "bc")
        .map(|&(_, _, score)| score)
        .expect("the unscored piece must still be listed");
    assert!(
        (bc_score - tokenizer.unk_score()).abs() < 1e-6,
        "an unscored piece must be reported at unk_score, got {}",
        bc_score
    );
    assert_eq!(
        ranked.last().map(|(token, _, _)| token.as_str()),
        Some("bc"),
        "the unscored piece must rank last, not first"
    );
}

/// A plain one-token-per-line vocabulary carries no scores at all; the BPE model
/// must still segment deterministically rather than picking merges at random.
#[test]
fn test_bpe_model_handles_score_less_vocabulary() {
    let dir = temp_dir_for("plain_vocab");
    let vocab_path = dir.join("vocab.txt");
    std::fs::write(&vocab_path, "<unk>\na\nb\nc\nab\n").expect("fixture must be writable");

    let mut tokenizer = SentencePieceTokenizer::new()
        .with_model_type(ModelType::Bpe)
        .with_normalization(false)
        .with_dummy_prefix(false);
    tokenizer
        .load_vocab_from_file(vocab_path.to_str().expect("temp path must be UTF-8"))
        .expect("plain vocabulary must load");

    assert_eq!(tokenizer.vocab_size(), 5);
    assert_eq!(
        tokenizer.tokenize("abc"),
        vec!["ab".to_string(), "c".to_string()]
    );

    let _ = std::fs::remove_file(&vocab_path);
    let _ = std::fs::remove_dir(&dir);
}

/// Regression: the plain vocabulary loader ignored marker-named pieces, so a
/// file that plainly provides `<unk>` still yielded a tokenizer that refused to
/// encode out-of-vocabulary text ("model has no <unk> piece").
#[test]
fn test_plain_vocab_loader_registers_special_tokens() {
    let dir = temp_dir_for("plain_vocab_specials");
    let vocab_path = dir.join("vocab.txt");
    // Line 5 is blank: it must leave a hole, not renumber the pieces after it
    // and not create an empty-string piece.
    std::fs::write(&vocab_path, "<pad>\n<unk>\n<s>\n</s>\n\n▁hello\n")
        .expect("fixture must be writable");

    let mut tokenizer = SentencePieceTokenizer::new().with_normalization(false);
    tokenizer
        .load_vocab_from_file(vocab_path.to_str().expect("temp path must be UTF-8"))
        .expect("plain vocabulary must load");

    assert_eq!(tokenizer.pad_token_id(), Some(0));
    assert_eq!(tokenizer.unk_token_id(), Some(1));
    assert_eq!(tokenizer.bos_token_id(), Some(2));
    assert_eq!(tokenizer.eos_token_id(), Some(3));
    assert_eq!(tokenizer.token_to_id("▁hello"), Some(5));
    assert_eq!(tokenizer.token_to_id(""), None);
    assert_eq!(tokenizer.vocab_size(), 5);

    // The file's <unk> is now usable, so OOV text encodes instead of erroring.
    let encoded = tokenizer.encode("zzz").expect("the file's <unk> must be usable");
    assert_eq!(encoded.input_ids, vec![1, 1, 1]);

    let _ = std::fs::remove_file(&vocab_path);
    let _ = std::fs::remove_dir(&dir);
}

/// A vocabulary file with no pieces is an error, not a silently empty tokenizer.
#[test]
fn test_plain_vocab_loader_rejects_an_empty_file() {
    let dir = temp_dir_for("plain_vocab_empty");
    let vocab_path = dir.join("empty.txt");
    std::fs::write(&vocab_path, "\n   \n\n").expect("fixture must be writable");

    let mut tokenizer = SentencePieceTokenizer::new();
    let error = tokenizer
        .load_vocab_from_file(vocab_path.to_str().expect("temp path must be UTF-8"))
        .expect_err("a vocabulary with no pieces must be rejected");
    assert!(
        error.to_string().contains("no pieces"),
        "error must explain the file is empty, got: {}",
        error
    );
    assert_eq!(tokenizer.vocab_size(), 0);

    let _ = std::fs::remove_file(&vocab_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_enhanced_normalization() {
    let tokenizer = SentencePieceTokenizer::new().with_normalization(true).with_dummy_prefix(true);

    let normalized = tokenizer.normalize_text("Hello  world");
    assert!(normalized.starts_with(WHITESPACE_MARKER));
    assert!(!normalized.contains("  ")); // Extra spaces should be removed
    assert_eq!(normalized, "▁Hello▁world");
}

/// The dummy affix is added as a raw space *before* escaping, so switching
/// escaping off leaves a real space rather than a stray `▁`.
///
/// Regression: the affix used to be appended as a `▁` *after* escaping, which
/// injected a word-boundary marker even into models that disable escaping.
#[test]
fn test_dummy_affix_is_added_before_escaping() {
    let mut tokenizer =
        SentencePieceTokenizer::new().with_normalization(true).with_dummy_prefix(true);
    tokenizer.escape_whitespaces = false;

    assert_eq!(tokenizer.normalize_text("Hello  world"), " Hello world");
    assert!(
        !tokenizer.normalize_text("Hello  world").contains(WHITESPACE_MARKER),
        "escaping is off, so no word-boundary marker may appear"
    );
}

/// Empty and whitespace-only input take no affix (SentencePiece's
/// `!norm.empty()` guard), so they tokenize to nothing rather than to a lone
/// `▁` piece that the vocabulary would have to absorb as an unknown.
#[test]
fn test_empty_input_gets_no_dummy_affix() {
    let tokenizer = SentencePieceTokenizer::new().with_normalization(true).with_dummy_prefix(true);

    assert_eq!(tokenizer.normalize_text(""), "");
    assert_eq!(tokenizer.normalize_text("   "), "");
    assert!(tokenizer.tokenize("").is_empty());
    assert!(tokenizer.tokenize("   ").is_empty());
}

/// A literal `▁` in the input is ordinary text, not a boundary marker that
/// already did the job — the affix is still added in front of it.
#[test]
fn test_literal_marker_in_input_does_not_suppress_the_affix() {
    let tokenizer = SentencePieceTokenizer::new().with_normalization(true).with_dummy_prefix(true);

    assert_eq!(
        tokenizer.normalize_text("▁already marked"),
        "▁▁already▁marked"
    );
}

/// The overwhelmingly common configuration (escaping on, extra whitespace
/// squeezed, ordinary text) is unaffected by the affix reordering.
#[test]
fn test_default_normalization_is_unchanged_for_ordinary_text() {
    let tokenizer = SentencePieceTokenizer::new().with_normalization(true).with_dummy_prefix(true);

    for (input, expected) in [
        ("hello world", "▁hello▁world"),
        ("Hello  world", "▁Hello▁world"),
        (" leading space", "▁leading▁space"),
        ("trailing ", "▁trailing"),
        ("a", "▁a"),
        ("tab\tsep", "▁tab▁sep"),
        ("世界 mixed", "▁世界▁mixed"),
    ] {
        assert_eq!(
            tokenizer.normalize_text(input),
            expected,
            "normalization changed for {:?}",
            input
        );
    }
}

/// Regression: `treat_whitespace_as_suffix` was parsed from the model and
/// exposed by a getter, but tokenization ignored it and still emitted `▁word`
/// pieces that such a model's vocabulary does not contain.
#[test]
fn test_treat_whitespace_as_suffix_moves_the_marker() {
    let prefix_model =
        SentencePieceTokenizer::new().with_normalization(true).with_dummy_prefix(true);
    assert!(!prefix_model.treats_whitespace_as_suffix());
    assert_eq!(prefix_model.normalize_text("hello world"), "▁hello▁world");

    let suffix_model = SentencePieceTokenizer::new()
        .with_normalization(true)
        .with_dummy_prefix(true)
        .with_whitespace_as_suffix(true);
    assert!(suffix_model.treats_whitespace_as_suffix());
    assert_eq!(suffix_model.normalize_text("hello world"), "hello▁world▁");

    // The word model must re-attach the marker on the same side.
    let word_suffix = suffix_model.with_model_type(ModelType::Word);
    assert_eq!(
        word_suffix.tokenize("hello world"),
        vec!["hello▁".to_string(), "world▁".to_string()]
    );

    let word_prefix = SentencePieceTokenizer::new()
        .with_normalization(true)
        .with_dummy_prefix(true)
        .with_model_type(ModelType::Word);
    assert_eq!(
        word_prefix.tokenize("hello world"),
        vec!["▁hello".to_string(), "▁world".to_string()]
    );
}

/// A suffix-marked vocabulary must actually encode, which it cannot do if the
/// pieces are built with a leading marker.
#[test]
fn test_suffix_model_encodes_against_a_suffix_vocabulary() {
    let mut tokenizer = SentencePieceTokenizer::new()
        .with_normalization(true)
        .with_dummy_prefix(true)
        .with_whitespace_as_suffix(true);

    for (index, piece) in ["<unk>", "hello▁", "world▁"].iter().enumerate() {
        let id = index as u32;
        tokenizer.vocab.insert((*piece).to_string(), id);
        tokenizer.id_to_token.insert(id, (*piece).to_string());
        tokenizer.scores.insert(id, if id == 0 { 0.0 } else { -1.0 });
    }
    tokenizer.unk_token_id = Some(0);
    tokenizer.special_tokens.insert("<unk>".to_string(), 0);
    tokenizer.refresh_stats();

    let encoded = tokenizer.encode("hello world").expect("encoding must succeed");
    assert_eq!(encoded.input_ids, vec![1, 2]);
    assert_eq!(
        tokenizer.decode(&encoded.input_ids).expect("decoding must succeed"),
        "hello world"
    );
}

#[test]
fn test_model_types() {
    let mut tokenizer = SentencePieceTokenizer::new().with_model_type(ModelType::Char);

    // Add some character vocabulary
    for ch in 'a'..='z' {
        let token_id = tokenizer.vocab.len() as u32;
        tokenizer.vocab.insert(ch.to_string(), token_id);
        tokenizer.id_to_token.insert(token_id, ch.to_string());
    }
    tokenizer.refresh_stats();

    let tokens = tokenizer.tokenize_char("hello");
    assert_eq!(tokens.len(), 5);
    assert_eq!(tokens[0], "h");
    assert_eq!(tokens[1], "e");
}

/// Special-token detection is driven by the loaded model, not a hardcoded list.
#[test]
fn test_special_token_detection_comes_from_the_model() {
    let empty = SentencePieceTokenizer::new();
    assert!(!empty.is_special_token("<pad>"));
    assert!(!empty.is_special_token("<extra_id_0>"));

    let loaded = tokenizer_with_pieces(&[("<unk>", 0.0), ("▁hello", -1.0)]);
    assert!(loaded.is_special_token("<unk>"));
    assert!(!loaded.is_special_token("▁hello"));
}

#[test]
fn test_token_scores() {
    let mut tokenizer = SentencePieceTokenizer::new();

    // Add tokens with scores
    tokenizer.vocab.insert("test".to_string(), 0);
    tokenizer.scores.insert(0, 5.0);
    tokenizer.refresh_stats();

    assert_eq!(tokenizer.get_token_score(0), Some(5.0));
    assert_eq!(tokenizer.get_token_score(999), None);

    let sorted_tokens = tokenizer.get_tokens_by_score();
    if !sorted_tokens.is_empty() {
        assert_eq!(sorted_tokens[0].0, "test");
        assert_eq!(sorted_tokens[0].2, 5.0);
    }
}

#[test]
fn test_configuration_methods() {
    let tokenizer = SentencePieceTokenizer::new()
        .with_model_type(ModelType::Bpe)
        .with_normalization(false)
        .with_dummy_prefix(false)
        .with_byte_fallback(true);

    assert_eq!(tokenizer.model_type, ModelType::Bpe);
    assert!(!tokenizer.normalization);
    assert!(!tokenizer.add_dummy_prefix);
    assert!(tokenizer.byte_fallback);
}

#[test]
fn test_unk_score_is_below_every_known_piece() {
    let tokenizer = tokenizer_with_pieces(&[("<unk>", 0.0), ("a", -1.0), ("b", -3.5)]);
    assert!(tokenizer.unk_score() < -3.5);
}
