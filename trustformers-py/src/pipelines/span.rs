//! Pure-Rust span extraction for the token-classification and
//! question-answering pipelines.
//!
//! Real, tested extraction math: NER "simple" aggregation (real softmax +
//! argmax per token, merged into contiguous same-type entities, sliced from
//! the original text by real byte offsets) and QA span extraction (real
//! joint start/end argmax over a context range, answer text sliced from the
//! original text by real byte offsets). Free of the Python C API, so
//! unit-testable with a plain `cargo test` -- see `scoring.rs` for the same
//! split.
//!
//! # Byte offsets, not character offsets
//!
//! Every `start`/`end` in this module is a **byte** offset into `text`'s
//! UTF-8 representation -- `slice_text` is literally `text.get(start..end)`,
//! Rust's own byte-range string indexing, and that is also this crate's
//! in-tree offset convention (`trustformers_tokenizers::bpe::BPETokenizer::
//! tokenize_with_offsets` asserts the same thing against `text.as_bytes()`).
//! This is **not** the same convention HuggingFace's Python-facing pipelines
//! use: their `start`/`end` keys are Unicode *character* (codepoint) offsets,
//! chosen so a caller can index the original Python `str` directly with
//! `text[start:end]` -- Python string indexing is codepoint-based, not
//! byte-based. For any text that is pure ASCII the two conventions agree
//! (one codepoint is one byte), but they diverge the moment `text` contains a
//! multi-byte character before the span of interest (accented Latin letters,
//! CJK, emoji, ...): a byte offset fed to Python's `text[start:end]` -- or a
//! Python-style character offset fed to this module's `slice_text` -- can
//! silently return the *wrong* substring rather than erroring, because both
//! are usually still valid ranges, just of the wrong text. See the
//! `..._by_byte_offsets...` tests below for both directions of this locked
//! down with non-ASCII fixtures. Wiring this module up to a real
//! offset-producing tokenizer and reporting HuggingFace-shaped `start`/`end`
//! keys would need one more conversion step (byte offset -> codepoint
//! offset, e.g. by counting `char_indices()` up to each byte position) that
//! does not exist anywhere in this crate today; nothing here performs it
//! silently.
//!
//! Neither function is reachable from Python today. Both require a real
//! `offset_mapping`, and every tokenizer this crate wraps
//! (`WordPieceTokenizer`, `BPETokenizer`) still sets
//! `TokenizedInput::offset_mapping` to `None` unconditionally (see
//! `pipelines::span_pipeline_unavailable` for the up-to-date verification),
//! so `PyTokenClassificationPipeline`/`PyQuestionAnsweringPipeline` refuse to
//! construct rather than wrap a tokenizer that can never supply one. This
//! module is the complete extraction algorithm on the *other* side of that
//! gap -- given real (byte) offsets from any source, it produces real
//! entities/answers today -- so it is `#[allow(dead_code)]` at the two public
//! entry points (exercised by the tests below, not by any live caller)
//! rather than deleted: remove the `#[allow(dead_code)]` and wire these in
//! the moment a real offset-producing tokenizer exists in this crate (and
//! add the byte->codepoint conversion above if the wiring reports
//! HuggingFace-shaped output).

use crate::pipelines::scoring::softmax;
use trustformers_core::errors::{runtime_error, TrustformersError};
use trustformers_core::tensor::Tensor;

/// One named entity, as `aggregation_strategy='simple'` reports it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Entity {
    /// The entity's surface text, sliced from the original input string by
    /// its real byte offsets (not reconstructed from decoded tokens, which
    /// would lose whitespace/casing WordPiece normalizes away).
    pub word: String,
    /// The entity type, with any `B-`/`I-` IOB prefix stripped.
    pub entity_group: String,
    /// Mean softmax probability of the winning label across the entity's
    /// tokens.
    pub score: f32,
    /// Start **byte** offset in the original text (inclusive) -- not a
    /// character/codepoint offset; see this module's doc comment.
    pub start: usize,
    /// End **byte** offset in the original text (exclusive) -- not a
    /// character/codepoint offset; see this module's doc comment.
    pub end: usize,
}

/// One extracted answer span, as the question-answering pipeline reports it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QaAnswer {
    /// The answer's surface text, sliced from the original context string by
    /// its real byte offsets.
    pub answer: String,
    /// `softmax(start_logits)[start] * softmax(end_logits)[end]` at the
    /// chosen span -- the joint probability HuggingFace's own QA pipeline
    /// reports.
    pub score: f32,
    /// Start **byte** offset in the original text (inclusive) -- not a
    /// character/codepoint offset; see this module's doc comment.
    pub start: usize,
    /// End **byte** offset in the original text (exclusive) -- not a
    /// character/codepoint offset; see this module's doc comment.
    pub end: usize,
}

/// The entity type an IOB label names, with any `B-`/`I-` prefix stripped.
///
/// `"O"` (outside any entity) is returned unchanged, so callers can match on
/// it directly rather than needing a second special case.
fn entity_type(label: &str) -> &str {
    label.strip_prefix("B-").or_else(|| label.strip_prefix("I-")).unwrap_or(label)
}

/// The per-token winning label (index into `labels`) and its softmax score,
/// for every row of `[.., seq_len, num_labels]` logits.
///
/// # Errors
///
/// Fails when `logits` has no dimensions, when its row count does not match
/// `offsets.len()`, or when a row's softmax fails (e.g. a non-finite logit).
fn per_token_predictions(
    logits: &Tensor,
    offsets_len: usize,
    labels: &[String],
) -> Result<Vec<(String, f32)>, TrustformersError> {
    let num_labels = *logits
        .shape()
        .last()
        .ok_or_else(|| runtime_error("logits tensor has no dimensions".to_string()))?;
    if num_labels == 0 {
        return Err(runtime_error("logits tensor has an empty label axis".to_string()));
    }
    if num_labels != labels.len() {
        return Err(runtime_error(format!(
            "the classification head produced {num_labels} logits per token but {} label names \
             were given",
            labels.len()
        )));
    }
    let data = logits.to_vec_f32()?;
    if data.len() % num_labels != 0 {
        return Err(runtime_error(format!(
            "logits hold {} values, which is not a multiple of the {num_labels} labels on the \
             last axis",
            data.len()
        )));
    }
    let rows: Vec<&[f32]> = data.chunks(num_labels).collect();
    if rows.len() != offsets_len {
        return Err(runtime_error(format!(
            "logits cover {} token positions but {offsets_len} offsets were given",
            rows.len()
        )));
    }

    rows.into_iter()
        .map(|row| {
            let scores = softmax(row)?;
            let (index, &score) = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .ok_or_else(|| runtime_error("softmax produced no scores".to_string()))?;
            Ok((labels[index].clone(), score))
        })
        .collect()
}

/// Slice `text[start..end]`, without the panic a raw index would risk on a
/// byte range that does not land on a character boundary.
///
/// # Errors
///
/// Fails when `start..end` is not a valid character-boundary-aligned range
/// into `text`.
fn slice_text(text: &str, start: usize, end: usize) -> Result<String, TrustformersError> {
    text.get(start..end)
        .map(str::to_string)
        .ok_or_else(|| runtime_error(format!("offsets {start}..{end} do not slice {text:?}")))
}

/// Named-entity recognition with `aggregation_strategy='simple'`: real
/// per-token softmax + argmax, grouped into contiguous same-type entities,
/// sliced from `text` by real byte offsets (see this module's doc comment).
///
/// Matches HuggingFace's own `simple` aggregation: consecutive tokens whose
/// label names the same entity type (its `B-`/`I-` prefix ignored) merge into
/// one entity, with `word`/`start`/`end` spanning the whole group and `score`
/// the mean of the group's per-token scores. A token whose offset span is
/// empty (`start == end`, e.g. `[CLS]`/`[SEP]`/`[PAD]`) or whose label is
/// `"O"` closes any open entity without joining one.
///
/// # Errors
///
/// See [`per_token_predictions`]; also fails when `offsets.len()` does not
/// match the logits' row count, or when an offset does not land on a
/// character boundary of `text`.
#[allow(dead_code)] // see this module's doc comment: not yet reachable from a constructible pipeline
pub(crate) fn aggregate_entities_simple(
    logits: &Tensor,
    offsets: &[(usize, usize)],
    text: &str,
    labels: &[String],
) -> Result<Vec<Entity>, TrustformersError> {
    let predictions = per_token_predictions(logits, offsets.len(), labels)?;

    struct Open {
        tag: String,
        score_sum: f32,
        count: usize,
        start: usize,
        end: usize,
    }
    let mut entities = Vec::new();
    let mut open: Option<Open> = None;

    let close = |open: Option<Open>, entities: &mut Vec<Entity>| -> Result<(), TrustformersError> {
        if let Some(o) = open {
            entities.push(Entity {
                word: slice_text(text, o.start, o.end)?,
                entity_group: o.tag,
                score: o.score_sum / o.count as f32,
                start: o.start,
                end: o.end,
            });
        }
        Ok(())
    };

    for (index, (label, score)) in predictions.into_iter().enumerate() {
        let (start, end) = offsets[index];
        let tag = entity_type(&label).to_string();
        if start == end || tag == "O" {
            close(open.take(), &mut entities)?;
            continue;
        }
        match &mut open {
            Some(current) if current.tag == tag => {
                current.score_sum += score;
                current.count += 1;
                current.end = end;
            },
            _ => {
                close(open.take(), &mut entities)?;
                open = Some(Open { tag, score_sum: score, count: 1, start, end });
            },
        }
    }
    close(open, &mut entities)?;

    Ok(entities)
}

/// Extractive question answering: the real joint-probability argmax over
/// `(start, end)` pairs within `context_range` (a half-open range of token
/// indices, `end - start <= max_answer_len`), answer text sliced from `text`
/// by the winning pair's real byte offsets (see this module's doc comment).
///
/// This is HuggingFace's own QA pipeline algorithm:
/// `score(s, e) = softmax(start_logits)[s] * softmax(end_logits)[e]`,
/// maximized over `s <= e` inside the context (never the question or special
/// tokens, which is what `context_range` excludes).
///
/// # Errors
///
/// Fails when `start_logits`/`end_logits` do not cover the same number of
/// positions as `offsets`, when `context_range` is empty or out of bounds, or
/// when the winning span's offsets do not land on a character boundary of
/// `text`.
#[allow(dead_code)] // see this module's doc comment: not yet reachable from a constructible pipeline
pub(crate) fn extract_answer(
    start_logits: &Tensor,
    end_logits: &Tensor,
    offsets: &[(usize, usize)],
    context_range: std::ops::Range<usize>,
    text: &str,
    max_answer_len: usize,
) -> Result<QaAnswer, TrustformersError> {
    let start_row = start_logits.to_vec_f32()?;
    let end_row = end_logits.to_vec_f32()?;
    if start_row.len() != end_row.len() {
        return Err(runtime_error(format!(
            "start_logits and end_logits must cover the same number of positions, got {} and {}",
            start_row.len(),
            end_row.len()
        )));
    }
    if start_row.len() != offsets.len() {
        return Err(runtime_error(format!(
            "logits cover {} positions but {} offsets were given",
            start_row.len(),
            offsets.len()
        )));
    }
    if context_range.start >= context_range.end || context_range.end > start_row.len() {
        return Err(runtime_error(format!(
            "context_range {context_range:?} is empty or out of bounds for {} positions",
            start_row.len()
        )));
    }
    if max_answer_len == 0 {
        return Err(runtime_error("max_answer_len must be at least 1".to_string()));
    }

    let start_probs = softmax(&start_row)?;
    let end_probs = softmax(&end_row)?;

    let mut best: Option<(usize, usize, f32)> = None;
    for start in context_range.clone() {
        let last_end = (start + max_answer_len - 1).min(context_range.end - 1);
        let start_prob = start_probs[start];
        for (end, &end_prob) in end_probs.iter().enumerate().take(last_end + 1).skip(start) {
            let score = start_prob * end_prob;
            let is_better = best.map(|(_, _, best_score)| score > best_score).unwrap_or(true);
            if is_better {
                best = Some((start, end, score));
            }
        }
    }

    let (start_index, end_index, score) = best
        .ok_or_else(|| runtime_error("no valid answer span exists in context_range".to_string()))?;
    let (char_start, _) = offsets[start_index];
    let (_, char_end) = offsets[end_index];

    Ok(QaAnswer {
        answer: slice_text(text, char_start, char_end)?,
        score,
        start: char_start,
        end: char_end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{ArrayD, IxDyn};

    fn logits(shape: &[usize], values: Vec<f32>) -> Tensor {
        Tensor::F32(ArrayD::from_shape_vec(IxDyn(shape), values).expect("fixture shape matches"))
    }

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    // ---- aggregate_entities_simple ----

    /// "New York is nice" tokenized as [CLS] New York is nice [SEP], with
    /// New/York confidently predicted B-LOC/I-LOC and the rest O. Must merge
    /// into one LOC entity spanning "New York", sliced from the real text --
    /// not reconstructed from token strings.
    #[test]
    fn merges_a_two_token_entity_and_slices_the_real_text() {
        let text = "New York is nice";
        // offsets: [CLS] is empty, "New"=0..3, "York"=4..8, "is"=9..11, "nice"=12..16, [SEP] empty
        let offsets = vec![(0, 0), (0, 3), (4, 8), (9, 11), (12, 16), (0, 0)];
        let label_names = labels(&["O", "B-LOC", "I-LOC"]);
        // 6 tokens x 3 labels; confident predictions matching the story above.
        let data = vec![
            9.0, -9.0, -9.0, // [CLS] -> O (ignored anyway: empty offset)
            -9.0, 9.0, -9.0, // New -> B-LOC
            -9.0, -9.0, 9.0, // York -> I-LOC
            9.0, -9.0, -9.0, // is -> O
            9.0, -9.0, -9.0, // nice -> O
            9.0, -9.0, -9.0, // [SEP] -> O
        ];
        let logits = logits(&[1, 6, 3], data);

        let entities =
            aggregate_entities_simple(&logits, &offsets, text, &label_names).expect("aggregates");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_group, "LOC");
        assert_eq!(entities[0].word, "New York");
        assert_eq!(entities[0].start, 0);
        assert_eq!(entities[0].end, 8);
        assert!(entities[0].score > 0.9, "confident predictions should score high");
    }

    /// Two separate entities of different types must not merge, and the gap
    /// between them (an "O" token) must not become part of either word.
    #[test]
    fn keeps_two_entities_of_different_types_separate() {
        let text = "Tim works at Apple";
        // [CLS], Tim=0..3, works=4..9, at=10..12, Apple=13..18, [SEP]
        let offsets = vec![(0, 0), (0, 3), (4, 9), (10, 12), (13, 18), (0, 0)];
        let label_names = labels(&["O", "B-PER", "B-ORG"]);
        let data = vec![
            9.0, -9.0, -9.0, // [CLS]
            -9.0, 9.0, -9.0, // Tim -> B-PER
            9.0, -9.0, -9.0, // works -> O
            9.0, -9.0, -9.0, // at -> O
            -9.0, -9.0, 9.0, // Apple -> B-ORG
            9.0, -9.0, -9.0, // [SEP]
        ];
        let logits = logits(&[1, 6, 3], data);

        let entities =
            aggregate_entities_simple(&logits, &offsets, text, &label_names).expect("aggregates");
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].word, "Tim");
        assert_eq!(entities[0].entity_group, "PER");
        assert_eq!(entities[1].word, "Apple");
        assert_eq!(entities[1].entity_group, "ORG");
    }

    #[test]
    fn an_all_outside_sequence_produces_no_entities() {
        let text = "nothing here";
        let offsets = vec![(0, 0), (0, 7), (8, 12), (0, 0)];
        let label_names = labels(&["O", "B-MISC"]);
        let data = vec![9.0, -9.0, 9.0, -9.0, 9.0, -9.0, 9.0, -9.0];
        let logits = logits(&[1, 4, 2], data);
        let entities =
            aggregate_entities_simple(&logits, &offsets, text, &label_names).expect("aggregates");
        assert!(entities.is_empty());
    }

    /// A tampered logit (flipping one token's confident O to a confident
    /// entity label) must change the aggregated result -- proving this is a
    /// real per-input computation, not a fixed answer.
    #[test]
    fn a_tampered_logit_changes_the_aggregated_entities() {
        let text = "plain text here";
        let offsets = vec![(0, 0), (0, 5), (6, 10), (11, 15), (0, 0)];
        let label_names = labels(&["O", "B-MISC"]);
        let base = vec![
            9.0, -9.0, // [CLS]
            9.0, -9.0, // plain -> O
            9.0, -9.0, // text -> O
            9.0, -9.0, // here -> O
            9.0, -9.0, // [SEP]
        ];
        let mut tampered = base.clone();
        // Flip "text" (row index 2) from confident-O to confident-B-MISC.
        tampered[4] = -9.0;
        tampered[5] = 9.0;

        let base_entities =
            aggregate_entities_simple(&logits(&[1, 5, 2], base), &offsets, text, &label_names)
                .expect("aggregates");
        let tampered_entities = aggregate_entities_simple(
            &logits(&[1, 5, 2], tampered),
            &offsets,
            text,
            &label_names,
        )
        .expect("aggregates");

        assert!(base_entities.is_empty());
        assert_eq!(tampered_entities.len(), 1);
        assert_eq!(tampered_entities[0].word, "text");
    }

    #[test]
    fn rejects_a_row_count_offset_mismatch() {
        let logits = logits(&[1, 2, 2], vec![9.0, -9.0, 9.0, -9.0]);
        let offsets = vec![(0, 1)]; // only one offset for two logit rows
        assert!(aggregate_entities_simple(&logits, &offsets, "a", &labels(&["O", "B-X"])).is_err());
    }

    /// Locks down this module's byte-offset semantics (see the module doc
    /// comment) end-to-end through `aggregate_entities_simple`, using an
    /// accented, non-ASCII fixture. "café" is 4 Unicode scalar values but 5
    /// UTF-8 bytes (`é` is 2 bytes) -- if this module ever treated offsets as
    /// character/codepoint counts instead of byte counts (HuggingFace's own,
    /// *different* convention -- see the module doc), the entity's reported
    /// span width would be 4, and a token placed after "café" using
    /// char-count arithmetic would be sliced one byte short of its true
    /// start. Every offset here comes from `str::find`/`str::len` (both
    /// byte-based), never a hand-counted literal, so the fixture cannot be
    /// accidentally "fixed" by getting the arithmetic wrong in the same way
    /// the code under test might.
    #[test]
    fn slices_multibyte_accented_text_by_byte_offsets_not_char_counts() {
        let text = "El café está en el centro";
        let cafe_start = text.find("café").expect("fixture contains café");
        let cafe_end = cafe_start + "café".len();
        // [CLS] El café está en el centro [SEP]; only "café" is tagged.
        let offsets = vec![(0, 0), (0, 2), (cafe_start, cafe_end), (0, 0)];
        let label_names = labels(&["O", "B-MISC"]);
        let data = vec![
            9.0, -9.0, // [CLS] -> O
            9.0, -9.0, // El -> O
            -9.0, 9.0, // café -> B-MISC
            9.0, -9.0, // [SEP] -> O
        ];
        let logits = logits(&[1, 4, 2], data);

        let entities =
            aggregate_entities_simple(&logits, &offsets, text, &label_names).expect("aggregates");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].word, "café", "must recover the exact multi-byte surface text");
        assert_eq!(entities[0].start, cafe_start);
        assert_eq!(entities[0].end, cafe_end);
        // The core assertion: the span width is café's 5-byte UTF-8 length,
        // not its 4-character/codepoint count.
        assert_eq!(
            entities[0].end - entities[0].start,
            5,
            "span width must be café's BYTE length (5), not its 4-character count"
        );
        assert_ne!(
            entities[0].end - entities[0].start,
            "café".chars().count(),
            "byte length and character count must differ for this fixture, or the test proves nothing"
        );
    }

    // ---- extract_answer ----

    /// A confident start-at-2/end-at-3 span over "capital of France is
    /// Paris" must extract exactly "Paris", sliced from the real text.
    #[test]
    fn extracts_the_real_answer_span_from_confident_logits() {
        let text = "the capital of France is Paris";
        // tokens: the(0) capital(1) of(2) France(3) is(4) Paris(5)
        let offsets = vec![(0, 3), (4, 11), (12, 14), (15, 21), (22, 24), (25, 30)];
        let start_logits = logits(&[6], vec![-9.0, -9.0, -9.0, -9.0, -9.0, 9.0]);
        let end_logits = logits(&[6], vec![-9.0, -9.0, -9.0, -9.0, -9.0, 9.0]);

        let answer = extract_answer(&start_logits, &end_logits, &offsets, 0..6, text, 10)
            .expect("extracts a span");
        assert_eq!(answer.answer, "Paris");
        assert_eq!(answer.start, 25);
        assert_eq!(answer.end, 30);
        assert!(answer.score > 0.9);
    }

    /// A multi-token answer ("New York") must extract the full span, not
    /// just its first or last token.
    #[test]
    fn extracts_a_multi_token_answer_span() {
        let text = "he lives in New York City";
        // he(0) lives(1) in(2) New(3) York(4) City(5)
        let offsets = vec![(0, 2), (3, 8), (9, 11), (12, 15), (16, 20), (21, 25)];
        let start_logits = logits(&[6], vec![-9.0, -9.0, -9.0, 9.0, -9.0, -9.0]);
        let end_logits = logits(&[6], vec![-9.0, -9.0, -9.0, -9.0, 9.0, -9.0]);

        let answer = extract_answer(&start_logits, &end_logits, &offsets, 0..6, text, 10)
            .expect("extracts a span");
        assert_eq!(answer.answer, "New York");
    }

    /// The whole point of real extraction: a tampered logit must change
    /// which span comes out.
    #[test]
    fn a_tampered_logit_changes_the_extracted_answer_span() {
        let text = "red blue green yellow";
        // red(0) blue(1) green(2) yellow(3)
        let offsets = vec![(0, 3), (4, 8), (9, 14), (15, 21)];
        let start_logits = logits(&[4], vec![-9.0, 9.0, -9.0, -9.0]);
        let end_logits = logits(&[4], vec![-9.0, 9.0, -9.0, -9.0]);
        let original = extract_answer(&start_logits, &end_logits, &offsets, 0..4, text, 10)
            .expect("extracts a span");
        assert_eq!(original.answer, "blue");

        // Move the confident start/end to index 3 ("yellow") instead.
        let tampered_start = logits(&[4], vec![-9.0, -9.0, -9.0, 9.0]);
        let tampered_end = logits(&[4], vec![-9.0, -9.0, -9.0, 9.0]);
        let tampered =
            extract_answer(&tampered_start, &tampered_end, &offsets, 0..4, text, 10)
                .expect("extracts a span");
        assert_eq!(tampered.answer, "yellow");
        assert_ne!(original.answer, tampered.answer);
    }

    /// `context_range` must exclude the question: even if the question's
    /// tokens would score higher, the extracted span must stay inside the
    /// context.
    #[test]
    fn respects_the_context_range_even_when_the_question_scores_higher() {
        // "Who? " (question, indices 0..2) + "answer here" (context, 2..4).
        let text = "Who? answer here";
        // Who(0) ?(1) answer(2) here(3)
        let offsets = vec![(0, 3), (3, 4), (5, 11), (12, 16)];
        // The highest start/end score is index 0 ("Who"), outside the context.
        let start_logits = logits(&[4], vec![9.0, -9.0, -1.0, -9.0]);
        let end_logits = logits(&[4], vec![9.0, -9.0, -1.0, -9.0]);

        let answer = extract_answer(&start_logits, &end_logits, &offsets, 2..4, text, 10)
            .expect("extracts a span inside the context");
        assert_eq!(answer.answer, "answer");
        assert_eq!(answer.start, 5);
    }

    #[test]
    fn max_answer_len_bounds_the_span_width() {
        let text = "a b c d e";
        let offsets = vec![(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
        // Start confidently at 0, end confidently at 4 -- unbounded, the
        // 5-token-wide span 0..4 is by far the best score (both endpoints
        // agree with the confident logit). Every single-token diagonal
        // score (s == e) is a product of one confident and one near-zero
        // factor, all of similar tiny magnitude, so this does not rely on
        // which single token specifically wins once bounded -- only on the
        // *width* being forced down to one token.
        let start_logits = logits(&[5], vec![9.0, -9.0, -9.0, -9.0, -9.0]);
        let end_logits = logits(&[5], vec![-9.0, -9.0, -9.0, -9.0, 9.0]);

        let unbounded = extract_answer(&start_logits, &end_logits, &offsets, 0..5, text, 5)
            .expect("extracts a span");
        assert_eq!(unbounded.answer, "a b c d e", "unbounded, the wide span wins");

        // With max_answer_len=1, only single-token (start == end) spans are
        // considered at all -- the wide span is structurally unreachable,
        // regardless of which single token numerically wins the tie among
        // near-equal candidates.
        let bounded = extract_answer(&start_logits, &end_logits, &offsets, 0..5, text, 1)
            .expect("extracts a bounded span");
        assert_eq!(
            bounded.end - bounded.start,
            1,
            "max_answer_len=1 allows only a single (1-character) token, got {:?}",
            bounded
        );
        assert_ne!(
            bounded.answer, unbounded.answer,
            "the bounded search must not be able to return the wide span"
        );
    }

    #[test]
    fn rejects_an_empty_or_out_of_bounds_context_range() {
        let offsets = vec![(0, 1), (2, 3)];
        let start_logits = logits(&[2], vec![0.0, 1.0]);
        let end_logits = logits(&[2], vec![0.0, 1.0]);
        assert!(extract_answer(&start_logits, &end_logits, &offsets, 1..1, "ab", 5).is_err());
        assert!(extract_answer(&start_logits, &end_logits, &offsets, 0..5, "ab", 5).is_err());
    }

    #[test]
    fn rejects_mismatched_logits_lengths() {
        let offsets = vec![(0, 1), (2, 3), (4, 5)];
        let start_logits = logits(&[3], vec![0.0, 1.0, 2.0]);
        let end_logits = logits(&[2], vec![0.0, 1.0]);
        assert!(extract_answer(&start_logits, &end_logits, &offsets, 0..3, "abcde", 5).is_err());
    }

    /// Locks down this module's byte-offset semantics (see the module doc
    /// comment) end-to-end through `extract_answer`, using a CJK fixture
    /// mixed with ASCII. Each of "東京"'s two ideographs is 3 UTF-8 bytes, so
    /// the answer is 6 bytes but 2 characters -- and because it is mixed with
    /// single-byte ASCII words earlier in the string, a byte-width-per-token
    /// assumption of any fixed N (not just N=1) would also mis-place it.
    /// Every offset comes from `str::find`/`str::len`, never a hand-counted
    /// literal.
    #[test]
    fn extracts_a_cjk_answer_span_by_byte_offsets_not_char_counts() {
        let text = "the capital of Japan is 東京";
        let answer = "東京";
        let answer_start = text.find(answer).expect("fixture contains the answer");
        let answer_end = answer_start + answer.len();

        // tokens: the(0) capital(1) of(2) Japan(3) is(4) 東京(5)
        let the_end = "the".len();
        let capital_start = text.find("capital").expect("fixture contains capital");
        let capital_end = capital_start + "capital".len();
        let of_start = text.find("of").expect("fixture contains of");
        let of_end = of_start + "of".len();
        let japan_start = text.find("Japan").expect("fixture contains Japan");
        let japan_end = japan_start + "Japan".len();
        let is_start = text.find(" is ").expect("fixture contains is") + 1;
        let is_end = is_start + "is".len();
        let offsets = vec![
            (0, the_end),
            (capital_start, capital_end),
            (of_start, of_end),
            (japan_start, japan_end),
            (is_start, is_end),
            (answer_start, answer_end),
        ];

        let start_logits = logits(&[6], vec![-9.0, -9.0, -9.0, -9.0, -9.0, 9.0]);
        let end_logits = logits(&[6], vec![-9.0, -9.0, -9.0, -9.0, -9.0, 9.0]);

        let result = extract_answer(&start_logits, &end_logits, &offsets, 0..6, text, 10)
            .expect("extracts a span");
        assert_eq!(result.answer, "東京");
        assert_eq!(result.start, answer_start);
        assert_eq!(result.end, answer_end);
        // The core assertion: the span width is 東京's 6-byte UTF-8 length,
        // not its 2-character/codepoint count.
        assert_eq!(
            result.end - result.start,
            6,
            "span width must be 東京's BYTE length (6), not its 2-character count"
        );
        assert_ne!(
            result.end - result.start,
            answer.chars().count(),
            "byte length and character count must differ for this fixture, or the test proves nothing"
        );
    }
}
