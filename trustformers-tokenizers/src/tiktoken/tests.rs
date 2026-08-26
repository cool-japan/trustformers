use super::*;

/// Build a rank table from `(bytes, rank)` pairs.
fn tiny_ranks(pairs: &[(&[u8], usize)]) -> RankMap {
    pairs.iter().map(|&(bytes, rank)| (bytes.to_vec(), rank)).collect()
}

/// `a b <space>` singles plus the merge `ab` (rank 3) and `ba` (rank 4).
fn table_ab_first() -> RankMap {
    tiny_ranks(&[(b"a", 0), (b"b", 1), (b" ", 2), (b"ab", 3), (b"ba", 4)])
}

/// Same pieces, but `ba` now outranks `ab`.
fn table_ba_first() -> RankMap {
    tiny_ranks(&[(b"a", 0), (b"b", 1), (b" ", 2), (b"ba", 3), (b"ab", 4)])
}

fn tokenizer_from(ranks: RankMap) -> TiktokenTokenizer {
    TiktokenTokenizer::from_encoding_spec(R50K_BASE, ranks)
        .expect("constructing from a rank table must succeed")
}

fn temp_dir_for(test_name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "trustformers_tiktoken_{}_{}",
        test_name,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir must be creatable");
    dir
}

/// Reference test: BPE must merge by rank, not emit one token per byte.
///
/// Against the old implementation (256 single-byte "ranks" and no merges) this
/// input produced one token per byte; both expectations below are hand-computed
/// from the tiktoken merge algorithm.
#[test]
fn test_byte_pair_merge_follows_rank_order() {
    let ab_first = tokenizer_from(table_ab_first());
    // "abab" -> ["ab", "ab"] (rank 3 twice), " ab" -> [" ", "ab"].
    assert_eq!(
        ab_first.encode_ordinary("abab ab").expect("encoding must succeed"),
        vec![3, 3, 2, 3]
    );

    let ba_first = tokenizer_from(table_ba_first());
    // With "ba" ranked first the same input segments as ["a", "ba", "b"].
    assert_eq!(
        ba_first.encode_ordinary("abab ab").expect("encoding must succeed"),
        vec![0, 3, 1, 2, 4]
    );

    // The degenerate byte-per-token output of the removed fake implementation.
    let per_byte = vec![0usize, 1, 0, 1, 2, 0, 1];
    assert_ne!(
        ab_first.encode_ordinary("abab ab").unwrap_or_default(),
        per_byte
    );
    assert_ne!(
        ba_first.encode_ordinary("abab ab").unwrap_or_default(),
        per_byte
    );
}

/// Straightforward O(L^2) reference: repeatedly merge the adjacent pair whose
/// concatenation has the lowest rank (leftmost on a tie), recomputing every
/// candidate from scratch each round.
///
/// `TiktokenTokenizer::byte_pair_merge` computes the same thing but repairs only
/// the two ranks adjacent to each merge, which is exactly the kind of
/// optimization that silently drifts. This reference pins it.
fn naive_byte_pair_merge(ranks: &RankMap, piece: &[u8]) -> Vec<usize> {
    let mut parts: Vec<Vec<u8>> = piece.iter().map(|&byte| vec![byte]).collect();

    loop {
        let mut best: Option<(usize, usize)> = None;

        for index in 0..parts.len().saturating_sub(1) {
            let mut merged = parts[index].clone();
            merged.extend_from_slice(&parts[index + 1]);
            if let Some(&rank) = ranks.get(&merged) {
                let improves = match best {
                    // Strictly lower only, so the leftmost of two equal ranks
                    // wins — the same tie-break the real implementation uses.
                    Some((best_rank, _)) => rank < best_rank,
                    None => true,
                };
                if improves {
                    best = Some((rank, index));
                }
            }
        }

        let Some((_, index)) = best else { break };
        let next = parts.remove(index + 1);
        parts[index].extend_from_slice(&next);
    }

    parts
        .iter()
        .map(|part| *ranks.get(part).expect("every merged part must be ranked"))
        .collect()
}

/// A rank table with **unique** ranks (so both implementations break ties the
/// same way) covering every single byte the fixtures use.
fn multi_level_ranks() -> RankMap {
    tiny_ranks(&[
        (b"a", 0),
        (b"b", 1),
        (b"c", 2),
        (b"d", 3),
        (b" ", 4),
        (b"ab", 5),
        (b"cd", 6),
        (b"bc", 7),
        (b"abc", 8),
        (b"abcd", 9),
        (b"cdab", 10),
        (b"aa", 11),
        (b"dd", 12),
        (b" a", 13),
        (b"bcd", 14),
    ])
}

/// The incremental rank repair must agree with the naive reference everywhere.
#[test]
fn test_byte_pair_merge_matches_naive_reference() {
    let ranks = multi_level_ranks();
    let tokenizer = tokenizer_from(ranks.clone());

    for piece in [
        "abcabcd",
        "aaab",
        "ddabcd",
        "cdabcd",
        "aaaa",
        "dcba",
        "abcd abcd",
        "a",
        "ba",
        "cdabcdab",
        "dddd",
        " abc",
    ] {
        let bytes = piece.as_bytes();
        let expected = naive_byte_pair_merge(&ranks, bytes);
        let actual = tokenizer.encode_piece(bytes).expect("the table is complete");
        assert_eq!(
            actual, expected,
            "incremental merge diverged from the reference for {:?}",
            piece
        );

        // ...and the result is a real segmentation of the input bytes.
        let mut rebuilt: Vec<u8> = Vec::new();
        for id in &actual {
            rebuilt.extend_from_slice(
                tokenizer.decoder.get(id).expect("every emitted id must decode"),
            );
        }
        assert_eq!(rebuilt.as_slice(), bytes);
    }
}

/// Hand-computed anchor for the merge order (see the reference walk-through in
/// the test above): "abcabcd" merges ab, ab, cd, abc, abcd -> ["abc", "abcd"].
#[test]
fn test_byte_pair_merge_hand_computed_anchor() {
    let tokenizer = tokenizer_from(multi_level_ranks());
    assert_eq!(
        tokenizer.encode_piece(b"abcabcd").expect("the table is complete"),
        vec![8, 9]
    );
}

#[test]
fn test_encode_decode_round_trip() {
    let tokenizer = tokenizer_from(table_ab_first());
    let text = "abab ab";
    let tokens = tokenizer.encode_ordinary(text).expect("encoding must succeed");
    assert_eq!(
        tokenizer.decode_tokens(&tokens).expect("decoding must succeed"),
        text
    );
}

#[test]
fn test_from_reader_and_from_file_parse_rank_files() {
    let ranks = table_ab_first();
    let mut buffer: Vec<u8> = Vec::new();
    ranks::write_ranks(&ranks, &mut buffer).expect("serialization must succeed");

    let from_reader =
        TiktokenTokenizer::from_reader(buffer.as_slice()).expect("from_reader must succeed");
    assert_eq!(from_reader.encoder(), &ranks);

    let dir = temp_dir_for("rank_file");
    let path = dir.join("r50k_base.tiktoken");
    std::fs::write(&path, &buffer).expect("rank file must be writable");

    let from_file = TiktokenTokenizer::from_file(&path).expect("from_file must succeed");
    assert_eq!(from_file.encoder(), &ranks);
    assert_eq!(
        from_file.encode_ordinary("abab").expect("encoding must succeed"),
        vec![3, 3]
    );

    // The published-encoding constructor attaches the encoding's special tokens.
    let loaded = ranks::load_ranks_from_file(&path).expect("rank file must load");
    let named = TiktokenTokenizer::from_encoding_spec(R50K_BASE, loaded)
        .expect("named encoding must build");
    assert_eq!(named.special_tokens().get("<|endoftext|>"), Some(&50256));

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

/// Regression: the no-argument constructors used to fabricate a 256-token
/// "vocabulary". They must now either load a real rank table or explain how to
/// obtain one.
#[test]
fn test_named_constructors_never_fabricate_a_vocabulary() {
    match TiktokenTokenizer::cl100k_base() {
        Ok(tokenizer) => {
            assert!(
                tokenizer.encoder().len() > 100_000,
                "cl100k_base must load the real rank table, got {} ranks",
                tokenizer.encoder().len()
            );
        },
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("cl100k_base.tiktoken"),
                "error must name the rank file, got: {}",
                message
            );
            assert!(
                message.contains(ranks::RANK_DIR_ENV),
                "error must document the override environment variable, got: {}",
                message
            );
        },
    }

    match TiktokenTokenizer::r50k_base() {
        Ok(tokenizer) => assert!(tokenizer.encoder().len() > 50_000),
        Err(error) => assert!(error.to_string().contains("r50k_base.tiktoken")),
    }
}

#[test]
fn test_unknown_encoding_name_is_rejected() {
    let error = TiktokenTokenizer::from_encoding_name("not_an_encoding")
        .expect_err("unknown encodings must error");
    assert!(error.to_string().contains("cl100k_base"));
}

/// Regression: special-token handling returned inside a `HashMap` iteration, so
/// only one special token was recognised and the result varied per process.
#[test]
fn test_all_special_tokens_are_recognised_deterministically() {
    let mut specials = HashMap::new();
    specials.insert("<|alpha|>".to_string(), 900);
    specials.insert("<|beta|>".to_string(), 901);
    let tokenizer = tokenizer_from(table_ab_first()).with_special_tokens(specials);

    let text = "ab<|alpha|>ab<|beta|>ab";
    let expected = vec![3, 900, 3, 901, 3];
    assert_eq!(
        tokenizer.encode_text(text).expect("encoding must succeed"),
        expected
    );

    // Deterministic across repeated calls (hash order must not leak through).
    for _ in 0..16 {
        assert_eq!(
            tokenizer.encode_text(text).expect("encoding must succeed"),
            expected
        );
    }
}

#[test]
fn test_special_tokens_match_leftmost_longest() {
    // "<|end" is a strict prefix of "<|endoftext|>", so both match at the same
    // position and only the longest may win.
    let mut specials = HashMap::new();
    specials.insert("<|end".to_string(), 900);
    specials.insert("<|endoftext|>".to_string(), 901);
    let tokenizer = tokenizer_from(table_ab_first()).with_special_tokens(specials);

    assert_eq!(
        tokenizer.encode_text("ab<|endoftext|>").expect("encoding must succeed"),
        vec![3, 901],
        "the longest special token at a position must win"
    );
    assert_eq!(
        tokenizer.encode_text("ab<|end").expect("encoding must succeed"),
        vec![3, 900]
    );
}

/// Regression: an allowed special token unknown to the tokenizer used to spin
/// forever because the scan made no progress.
#[test]
fn test_unknown_allowed_special_token_terminates() {
    let mut specials = HashMap::new();
    specials.insert("<|alpha|>".to_string(), 900);
    let tokenizer = tokenizer_from(table_ab_first()).with_special_tokens(specials);

    let mut allowed = HashSet::new();
    allowed.insert("<|not_in_this_tokenizer|>".to_string());

    // Must return (rather than hang) and treat the unknown marker as text; the
    // marker's characters are not in the tiny rank table, so this reports the
    // incomplete table instead of inventing ids.
    let result = tokenizer.encode_with_special_tokens("ab<|not_in_this_tokenizer|>", &allowed);
    assert!(result.is_err());

    // With only known specials allowed, encoding proceeds normally.
    let mut allowed_known = HashSet::new();
    allowed_known.insert("<|alpha|>".to_string());
    assert_eq!(
        tokenizer
            .encode_with_special_tokens("ab<|alpha|>ab", &allowed_known)
            .expect("encoding must succeed"),
        vec![3, 900, 3]
    );
}

/// Regression: the public vocab map was built with `from_utf8_lossy`, so every
/// non-UTF-8 rank entry collapsed onto U+FFFD and overwrote its neighbours.
#[test]
fn test_vocab_is_reversible_for_non_utf8_tokens() {
    let ranks = tiny_ranks(&[(b"a", 0), (&[0xff], 1), (&[0xfe], 2), (&[0xff, 0xfe], 3)]);
    let tokenizer = tokenizer_from(ranks);

    let vocab = tokenizer.get_vocab();
    // 4 rank entries + 1 special token, all distinct.
    assert_eq!(vocab.len(), 5);
    assert_eq!(vocab.values().collect::<HashSet<_>>().len(), 5);

    for (token, &id) in &vocab {
        assert_eq!(
            tokenizer.token_to_id(token),
            Some(id),
            "token {:?} must map back to id {}",
            token,
            id
        );
    }

    for id in 0..4u32 {
        let token = tokenizer.id_to_token(id).expect("every rank has a spelling");
        assert_eq!(tokenizer.token_to_id(&token), Some(id));
    }

    assert_eq!(tokenizer.vocab_size(), 5);
}

#[test]
fn test_decode_rejects_unknown_ids() {
    let tokenizer = tokenizer_from(table_ab_first());
    assert!(tokenizer.decode_tokens(&[9_999]).is_err());
}

#[test]
fn test_incomplete_rank_table_reports_an_error() {
    // No rank for the byte 'z'.
    let tokenizer = tokenizer_from(table_ab_first());
    let error = tokenizer.encode_ordinary("z").expect_err("missing ranks must error");
    assert!(error.to_string().contains("incomplete"));
}

/// The cache is keyed per pre-token, not per full input.
#[test]
fn test_cache_is_keyed_on_pre_tokens() {
    let tokenizer = tokenizer_from(table_ab_first());
    let text = "abab ab";
    let first = tokenizer.encode_ordinary(text).expect("encoding must succeed");
    let second = tokenizer.encode_ordinary(text).expect("encoding must succeed");
    assert_eq!(first, second);

    let cache = tokenizer.cache.read().expect("cache lock must not be poisoned");
    assert!(
        !cache.contains_key(text.as_bytes()),
        "the whole input must not be cached"
    );
    assert!(
        cache.keys().all(|k| k.len() <= 4),
        "only pre-tokens are cached"
    );
}

#[test]
fn test_tokenizer_trait_round_trip() {
    let tokenizer = tokenizer_from(table_ab_first());
    let encoded = tokenizer.encode("abab ab").expect("encoding must succeed");
    assert_eq!(encoded.input_ids.len(), encoded.attention_mask.len());
    assert_eq!(
        tokenizer.decode(&encoded.input_ids).expect("decoding must succeed"),
        "abab ab"
    );
}
