//! WordPiece (BERT) tokenizer.
//!
//! The pre-tokenization stage is a port of BERT's `BasicTokenizer`
//! (`clean_text` -> `tokenize_chinese_chars` -> lowercase/`strip_accents` ->
//! `split_on_punc` -> whitespace split) and the sub-word stage is the standard
//! greedy longest-match-first search with `##` continuation pieces.
//!
//! There is deliberately **no** built-in vocabulary: a WordPiece tokenizer is
//! meaningless without the checkpoint's `vocab.txt`, so loading fails loudly
//! rather than substituting an invented word list.
//!
//! # Offsets
//!
//! Every stage carries a byte-span alignment back into the caller's original
//! string, so [`WordPieceTokenizer::tokenize_with_offsets`] and the
//! `offset_mapping` of [`Tokenizer::encode`]/[`Tokenizer::encode_pair`] report
//! where each piece — continuation (`##`) pieces included — came from *before*
//! accent stripping and lowercasing. See [`crate::offsets`] for the byte- (not
//! character-) offset convention and the alignment machinery.

use crate::offsets::{
    aligned_lowercase, aligned_strip_accents, AlignmentBuilder, ByteSpan, OffsetAlignment,
};
use crate::vocab::Vocab;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};
use unicode_categories::UnicodeCategories;

#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    vocab: Vocab,
    unk_token: String,
    sep_token: String,
    pad_token: String,
    cls_token: String,
    mask_token: String,
    do_lower_case: bool,
    max_input_chars_per_word: usize,
}

/// The offset every special token (and every padding token) reports.
///
/// HuggingFace's convention: a special token covers no source text, and an
/// empty span at position 0 is unambiguous because a real token starting at
/// byte 0 always has a non-empty span.
const SPECIAL_TOKEN_SPAN: ByteSpan = (0, 0);

/// One basic-tokenization word together with the byte span, in the caller's
/// original string, that every one of its characters came from.
///
/// The characters are the *normalized* ones (accent-stripped and lowercased
/// when `do_lower_case` is set) — the ones the sub-word search runs over —
/// while the spans always point back into the pre-normalization input, which
/// is what makes a `##` continuation piece able to report its own sub-span of
/// the original text.
#[derive(Debug, Clone)]
struct AlignedWord {
    /// The word as the sub-word search sees it.
    text: String,
    /// `text`'s characters, so the sub-word search does not re-collect them.
    chars: Vec<char>,
    /// One byte span in the original input per entry of `chars`.
    char_spans: Vec<ByteSpan>,
}

impl AlignedWord {
    fn new(chars: &[char], char_spans: &[ByteSpan]) -> Self {
        Self {
            text: chars.iter().collect(),
            chars: chars.to_vec(),
            char_spans: char_spans.to_vec(),
        }
    }

    /// The smallest byte span containing every one of `spans`.
    ///
    /// A union rather than `(first.0, last.1)`: normalization can map several
    /// output characters onto one source character, so neighbouring spans may
    /// repeat, and an insertion contributes an empty span.
    fn enclosing_span(spans: &[ByteSpan]) -> ByteSpan {
        let start = spans.iter().map(|&(start, _)| start).min().unwrap_or(0);
        let end = spans.iter().map(|&(_, end)| end).max().unwrap_or(start);
        (start, end.max(start))
    }

    /// The byte span of the whole word in the original input.
    fn span(&self) -> ByteSpan {
        Self::enclosing_span(&self.char_spans)
    }
}

impl WordPieceTokenizer {
    pub fn new(vocab: HashMap<String, u32>, do_lower_case: bool) -> Self {
        Self {
            vocab: Vocab::from_map(vocab),
            unk_token: "[UNK]".to_string(),
            sep_token: "[SEP]".to_string(),
            pad_token: "[PAD]".to_string(),
            cls_token: "[CLS]".to_string(),
            mask_token: "[MASK]".to_string(),
            do_lower_case,
            max_input_chars_per_word: 100,
        }
    }

    /// Load a WordPiece tokenizer for `model_name` from disk.
    ///
    /// The following locations are probed, in order:
    ///
    /// * `{model_name}/vocab.txt`
    /// * `{model_name}-vocab.txt`
    /// * `models/{model_name}/vocab.txt`
    /// * `./vocab/{model_name}.txt`
    /// * `{model_name}` itself, when it points at a file
    ///
    /// If none of them resolves, this returns an error naming every probed
    /// path. It never substitutes a synthetic vocabulary.
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let potential_paths = vec![
            format!("{}/vocab.txt", model_name),
            format!("{}-vocab.txt", model_name),
            format!("models/{}/vocab.txt", model_name),
            format!("./vocab/{}.txt", model_name),
            model_name.to_string(),
        ];

        for path in &potential_paths {
            if !std::path::Path::new(path).is_file() {
                continue;
            }
            let vocab = Self::load_vocab_from_file(path)?;
            return Ok(Self::new(vocab, model_name.contains("uncased")));
        }

        Err(TrustformersError::invalid_input(format!(
            "No WordPiece vocabulary found for '{}'. A vocab.txt from the model \
             checkpoint is required; probed paths: {}. Use \
             `WordPieceTokenizer::from_vocab_file(path, do_lower_case)` to load an \
             explicit file.",
            model_name,
            potential_paths.join(", ")
        )))
    }

    pub fn from_vocab_file(vocab_path: &str, do_lower_case: bool) -> Result<Self> {
        let vocab = Self::load_vocab_from_file(vocab_path)?;
        Ok(Self::new(vocab, do_lower_case))
    }

    /// Whether input is lowercased and accent-stripped before tokenization.
    pub fn do_lower_case(&self) -> bool {
        self.do_lower_case
    }

    /// The mask token string.
    pub fn mask_token(&self) -> &str {
        &self.mask_token
    }

    /// Load vocabulary from a file
    fn load_vocab_from_file(path: &str) -> Result<HashMap<String, u32>> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to open vocab file {}: {}", path, e))
        })?;
        let reader = BufReader::new(file);

        let mut vocab = HashMap::new();
        for (id, line) in reader.lines().enumerate() {
            let token = line
                .map_err(|e| TrustformersError::io_error(format!("Failed to read line: {}", e)))?;
            let token = token.trim().to_string();
            if !token.is_empty() {
                vocab.insert(token, id as u32);
            }
        }

        if vocab.is_empty() {
            return Err(TrustformersError::other(
                "Empty vocabulary file".to_string(),
            ));
        }

        Ok(vocab)
    }

    /// BERT `_clean_text`: drop NUL/replacement/control characters and turn every
    /// whitespace character into a plain space.
    ///
    /// Public for inspection and testing, like
    /// [`crate::bpe::BPETokenizer::pre_tokenize`]: this and the other
    /// individually named BERT stages below are projections of the aligned
    /// pipeline the encoder actually runs, so inspecting one can never show a
    /// stage the encoder does not perform.
    pub fn clean_text(text: &str) -> String {
        Self::aligned_clean_text(text).0
    }

    /// [`Self::clean_text`] plus a byte alignment back into `text`.
    ///
    /// Dropped characters contribute no alignment unit at all, which is why a
    /// word straddling a dropped control character reports a span that still
    /// contains that character: the span is the union of its pieces' sources,
    /// and there is no way to express a hole in a single `(start, end)` pair.
    /// The span therefore always *contains* the token's source text, which is
    /// the property `&text[start..end]` consumers rely on.
    fn aligned_clean_text(text: &str) -> (String, OffsetAlignment) {
        let mut cleaned = String::with_capacity(text.len());
        let mut alignment = AlignmentBuilder::new();
        for (index, ch) in text.char_indices() {
            let cp = ch as u32;
            let len = ch.len_utf8();
            if cp == 0 || cp == 0xfffd {
                continue;
            }
            if ch.is_whitespace() {
                cleaned.push(' ');
                alignment.push(1, index, index + len);
                continue;
            }
            if ch.is_control() {
                continue;
            }
            cleaned.push(ch);
            alignment.push(len, index, index + len);
        }
        (cleaned, alignment.finish(text.len()))
    }

    /// BERT `_tokenize_chinese_chars`: pad every CJK codepoint with spaces so it
    /// becomes its own token.
    pub fn tokenize_chinese_chars(text: &str) -> String {
        Self::aligned_tokenize_chinese_chars(text).0
    }

    /// [`Self::tokenize_chinese_chars`] plus a byte alignment back into `text`.
    ///
    /// The padding spaces are insertions with no source of their own; they are
    /// consumed by the whitespace split that follows and never end up inside a
    /// word.
    fn aligned_tokenize_chinese_chars(text: &str) -> (String, OffsetAlignment) {
        let mut spaced = String::with_capacity(text.len());
        let mut alignment = AlignmentBuilder::new();
        for (index, ch) in text.char_indices() {
            let len = ch.len_utf8();
            if Self::is_chinese_char(ch) {
                spaced.push(' ');
                alignment.skip_output(1);
                spaced.push(ch);
                alignment.push(len, index, index + len);
                spaced.push(' ');
                alignment.skip_output(1);
            } else {
                spaced.push(ch);
                alignment.push(len, index, index + len);
            }
        }
        (spaced, alignment.finish(text.len()))
    }

    /// The CJK ranges used by BERT's `_is_chinese_char`.
    fn is_chinese_char(ch: char) -> bool {
        let cp = ch as u32;
        (0x4E00..=0x9FFF).contains(&cp)
            || (0x3400..=0x4DBF).contains(&cp)
            || (0x20000..=0x2A6DF).contains(&cp)
            || (0x2A700..=0x2B73F).contains(&cp)
            || (0x2B740..=0x2B81F).contains(&cp)
            || (0x2B820..=0x2CEAF).contains(&cp)
            || (0xF900..=0xFAFF).contains(&cp)
            || (0x2F800..=0x2FA1F).contains(&cp)
    }

    /// BERT `_run_strip_accents`: NFD, then drop non-spacing marks.
    ///
    /// Shared with [`crate::offsets::aligned_strip_accents`], which produces
    /// the same string plus the alignment back into its input, so the offset
    /// path and the token path can never drift apart.
    pub fn strip_accents(text: &str) -> String {
        crate::offsets::strip_accents(text)
    }

    /// BERT `_is_punctuation`: the ASCII punctuation blocks plus any codepoint
    /// in a Unicode `P*` category.
    fn is_punctuation(ch: char) -> bool {
        let cp = ch as u32;
        (33..=47).contains(&cp)
            || (58..=64).contains(&cp)
            || (91..=96).contains(&cp)
            || (123..=126).contains(&cp)
            || ch.is_punctuation()
    }

    /// BERT `_run_split_on_punc`: each punctuation character becomes its own token.
    pub fn split_on_punctuation(token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        let spans: Vec<ByteSpan> = vec![(0, 0); chars.len()];
        let mut pieces = Vec::new();
        Self::split_aligned_on_punctuation(&chars, &spans, &mut pieces);
        pieces.into_iter().map(|word| word.text).collect()
    }

    /// [`Self::split_on_punctuation`] over a span-carrying word: the character
    /// spans are partitioned alongside the characters, so each piece keeps the
    /// exact source range of the characters it covers.
    fn split_aligned_on_punctuation(
        chars: &[char],
        spans: &[ByteSpan],
        output: &mut Vec<AlignedWord>,
    ) {
        let mut piece_start = 0usize;

        for (index, &ch) in chars.iter().enumerate() {
            if !Self::is_punctuation(ch) {
                continue;
            }
            if index > piece_start {
                output.push(AlignedWord::new(
                    &chars[piece_start..index],
                    &spans[piece_start..index],
                ));
            }
            output.push(AlignedWord::new(
                &chars[index..index + 1],
                &spans[index..index + 1],
            ));
            piece_start = index + 1;
        }

        if piece_start < chars.len() {
            output.push(AlignedWord::new(
                &chars[piece_start..],
                &spans[piece_start..],
            ));
        }
    }

    /// BERT `BasicTokenizer`: clean -> CJK padding -> whitespace split ->
    /// (strip accents -> lowercase) -> punctuation split.
    ///
    /// The accent/case order matches `BertNormalizer`, which strips accents
    /// *before* lowercasing. No Unicode normalization is applied in the cased
    /// path, because BERT's own tokenizer performs none — adding one here would
    /// produce ids no cased checkpoint agrees with.
    pub fn basic_tokenize(&self, text: &str) -> Vec<String> {
        self.aligned_basic_tokenize(text).into_iter().map(|word| word.text).collect()
    }

    /// [`Self::basic_tokenize`], with each word carrying the byte span in
    /// `text` of every one of its characters.
    ///
    /// This is the only implementation of the basic-tokenization pipeline;
    /// [`Self::basic_tokenize`] is a projection of it, so the tokens the offset
    /// path describes are by construction the tokens the encoder emits.
    ///
    /// Every stage contributes an alignment — control-character stripping, CJK
    /// space padding, accent stripping, lowercasing — and they are composed
    /// into a single map back to the caller's original string, so a `##`
    /// continuation piece of a lowercased, accent-stripped word still reports
    /// its own sub-span of the *pre-normalization* text.
    fn aligned_basic_tokenize(&self, text: &str) -> Vec<AlignedWord> {
        let (cleaned, clean_alignment) = Self::aligned_clean_text(text);
        let (spaced, cjk_alignment) = Self::aligned_tokenize_chinese_chars(&cleaned);
        // `spaced` -> original.
        let alignment = cjk_alignment.rebase(&clean_alignment);

        let mut words = Vec::new();
        for (word_start, word_end) in Self::whitespace_delimited_spans(&spaced) {
            let raw_word = &spaced[word_start..word_end];

            let (word_text, word_alignment) = if self.do_lower_case {
                let (stripped, strip_alignment) = aligned_strip_accents(raw_word);
                let (lowered, lower_alignment) = aligned_lowercase(&stripped);
                (lowered, lower_alignment.rebase(&strip_alignment))
            } else {
                (raw_word.to_string(), OffsetAlignment::identity(raw_word))
            };

            let mut chars = Vec::new();
            let mut char_spans = Vec::new();
            for (index, ch) in word_text.char_indices() {
                let (raw_start, raw_end) = word_alignment.map_span(index, index + ch.len_utf8());
                char_spans.push(alignment.map_span(word_start + raw_start, word_start + raw_end));
                chars.push(ch);
            }

            Self::split_aligned_on_punctuation(&chars, &char_spans, &mut words);
        }

        words
    }

    /// Byte ranges of the whitespace-delimited runs of `text`, which is exactly
    /// what `str::split_whitespace` yields, but with their positions.
    fn whitespace_delimited_spans(text: &str) -> Vec<ByteSpan> {
        let mut spans = Vec::new();
        let mut start: Option<usize> = None;

        for (index, ch) in text.char_indices() {
            if ch.is_whitespace() {
                if let Some(run_start) = start.take() {
                    spans.push((run_start, index));
                }
            } else if start.is_none() {
                start = Some(index);
            }
        }

        if let Some(run_start) = start {
            spans.push((run_start, text.len()));
        }

        spans
    }

    /// Greedy longest-match-first WordPiece segmentation.
    ///
    /// Matches HuggingFace semantics: if any position of the word fails to match
    /// a vocabulary piece, the *whole* word becomes a single unknown token — the
    /// sub-tokens collected so far are discarded.
    pub fn wordpiece_tokenize(&self, word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        self.wordpiece_piece_ranges(&chars)
            .into_iter()
            .map(|(piece, _, _)| piece)
            .collect()
    }

    /// [`Self::wordpiece_tokenize`] over a character slice, reporting for each
    /// piece the half-open character range `[start, end)` of `chars` it covers.
    ///
    /// The two fall-back paths (a word longer than
    /// `max_input_chars_per_word`, and a word no vocabulary segmentation
    /// covers) both emit one unknown token spanning the whole word, so the
    /// reported range is the whole word in those cases too.
    fn wordpiece_piece_ranges(&self, chars: &[char]) -> Vec<(String, usize, usize)> {
        if chars.len() > self.max_input_chars_per_word {
            return vec![(self.unk_token.clone(), 0, chars.len())];
        }

        let mut sub_tokens: Vec<(String, usize, usize)> = Vec::new();
        let mut candidate = String::new();
        let mut start = 0;

        while start < chars.len() {
            let mut end = chars.len();
            let mut matched: Option<(String, usize)> = None;

            while start < end {
                candidate.clear();
                if start > 0 {
                    candidate.push_str("##");
                }
                candidate.extend(chars[start..end].iter());

                if self.vocab.contains(&candidate) {
                    matched = Some((candidate.clone(), end));
                    break;
                }

                end -= 1;
            }

            match matched {
                Some((piece, next_start)) => {
                    sub_tokens.push((piece, start, next_start));
                    start = next_start;
                },
                None => return vec![(self.unk_token.clone(), 0, chars.len())],
            }
        }

        sub_tokens
    }

    /// Tokenize `text` into WordPiece string tokens (no special tokens added).
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenize_with_offsets(text).0
    }

    /// Tokenize `text`, reporting for each token the byte range of the
    /// **original** `text` it covers.
    ///
    /// The two returned vectors always have the same length: they are produced
    /// by one pass, not by two independent ones.
    ///
    /// Offsets are byte indices into `text` itself, before any normalization
    /// (see [`crate::offsets`] for the convention and for
    /// [`crate::offsets::byte_offsets_to_char_offsets`], which converts them to
    /// the character offsets a Python caller needs). `&text[start..end]` is
    /// always a valid slice and always *contains* the source of the token:
    ///
    /// * for a cased tokenizer it is exactly the token's surface text (`##`
    ///   stripped),
    /// * for `do_lower_case` it is the pre-normalization text, so
    ///   `strip_accents` + lowercase applied to it yields the token,
    /// * a character that `clean_text` deletes (NUL, `U+FFFD`, other control
    ///   characters) leaves no unit of its own, so a token spanning a deleted
    ///   character reports a range that still contains it.
    pub fn tokenize_with_offsets(&self, text: &str) -> (Vec<String>, Vec<ByteSpan>) {
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();

        for word in self.aligned_basic_tokenize(text) {
            let word_span = word.span();
            for (piece, start, end) in self.wordpiece_piece_ranges(&word.chars) {
                let span = match word.char_spans.get(start..end) {
                    Some(spans) if !spans.is_empty() => AlignedWord::enclosing_span(spans),
                    _ => word_span,
                };
                tokens.push(piece);
                offsets.push(span);
            }
        }

        (tokens, offsets)
    }

    fn is_special_token(&self, token: &str) -> bool {
        token == self.pad_token
            || token == self.cls_token
            || token == self.sep_token
            || token == self.mask_token
    }

    fn token_id(&self, token: &str) -> Result<u32> {
        match self.vocab.get_id(token) {
            Some(id) => Ok(id),
            None => self.vocab.get_id(&self.unk_token).ok_or_else(|| {
                TrustformersError::invalid_input(
                    "UNK token not found in WordPiece vocabulary".to_string(),
                )
            }),
        }
    }

    /// Decode with explicit control over special-token removal.
    pub fn decode_tokens(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| self.vocab.get_token(id))
            .filter(|token| !(skip_special_tokens && self.is_special_token(token)))
            .collect();

        let mut text = String::new();
        for token in tokens {
            if let Some(suffix) = token.strip_prefix("##") {
                text.push_str(suffix);
            } else {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&token);
            }
        }

        text
    }
}

impl Tokenizer for WordPieceTokenizer {
    /// Encode `text` as `[CLS] ... [SEP]`.
    ///
    /// `offset_mapping` is always populated: byte spans into `text` for the
    /// content tokens (see [`Self::tokenize_with_offsets`]) and `(0, 0)` for
    /// `[CLS]`/`[SEP]`, matching HuggingFace's convention for special tokens.
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let (content_tokens, content_offsets) = self.tokenize_with_offsets(text);

        let mut tokens = Vec::with_capacity(content_tokens.len() + 2);
        let mut offsets = Vec::with_capacity(content_offsets.len() + 2);

        tokens.push(self.cls_token.clone());
        offsets.push(SPECIAL_TOKEN_SPAN);
        tokens.extend(content_tokens);
        offsets.extend(content_offsets);
        tokens.push(self.sep_token.clone());
        offsets.push(SPECIAL_TOKEN_SPAN);

        let mut input_ids = Vec::with_capacity(tokens.len());
        for token in &tokens {
            input_ids.push(self.token_id(token)?);
        }

        let attention_mask = vec![1u8; input_ids.len()];
        let special_tokens_mask: Vec<u8> =
            tokens.iter().map(|t| u8::from(self.is_special_token(t))).collect();

        let input_ids_len = input_ids.len();
        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids: Some(vec![0u32; input_ids_len]),
            special_tokens_mask: Some(special_tokens_mask),
            offset_mapping: Some(offsets),
            overflowing_tokens: None,
        })
    }

    /// Encode a sequence pair as `[CLS] A [SEP] B [SEP]`.
    ///
    /// `offset_mapping` is **per sequence**, exactly as HuggingFace reports it:
    /// a token's span indexes `text` when its `token_type_ids` entry is `0` and
    /// `text2` when it is `1`. There is no combined coordinate space, because
    /// the two sequences are never concatenated. `[CLS]`/`[SEP]` are `(0, 0)`.
    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        let (first_tokens, first_offsets) = self.tokenize_with_offsets(text);
        let (second_tokens, second_offsets) = self.tokenize_with_offsets(text2);

        let mut tokens = Vec::with_capacity(first_tokens.len() + second_tokens.len() + 3);
        let mut offsets = Vec::with_capacity(first_offsets.len() + second_offsets.len() + 3);

        tokens.push(self.cls_token.clone());
        offsets.push(SPECIAL_TOKEN_SPAN);
        tokens.extend(first_tokens);
        offsets.extend(first_offsets);
        tokens.push(self.sep_token.clone());
        offsets.push(SPECIAL_TOKEN_SPAN);
        let first_seg_len = tokens.len();

        tokens.extend(second_tokens);
        offsets.extend(second_offsets);
        tokens.push(self.sep_token.clone());
        offsets.push(SPECIAL_TOKEN_SPAN);

        let mut input_ids = Vec::with_capacity(tokens.len());
        for token in &tokens {
            input_ids.push(self.token_id(token)?);
        }

        let attention_mask = vec![1u8; input_ids.len()];
        let special_tokens_mask: Vec<u8> =
            tokens.iter().map(|t| u8::from(self.is_special_token(t))).collect();

        let mut token_type_ids = vec![0u32; first_seg_len];
        token_type_ids.extend(vec![1u32; input_ids.len() - first_seg_len]);

        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids: Some(token_type_ids),
            special_tokens_mask: Some(special_tokens_mask),
            offset_mapping: Some(offsets),
            overflowing_tokens: None,
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(self.decode_tokens(ids, true))
    }

    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.get_token_to_id_map().clone()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bert_like_vocab() -> HashMap<String, u32> {
        let tokens = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", ",", "!", "un",
            "##want", "##ed", "run", "##ning", "世", "界", "cafe",
        ];
        tokens.iter().enumerate().map(|(i, t)| ((*t).to_string(), i as u32)).collect()
    }

    /// Regression: `from_pretrained` used to invent a ~100-word vocabulary.
    #[test]
    fn test_from_pretrained_errors_without_vocab_file() {
        let error = WordPieceTokenizer::from_pretrained("bert-base-uncased")
            .expect_err("a missing vocab.txt must be an error, never a synthetic vocabulary");
        let message = error.to_string();
        assert!(
            message.contains("vocab.txt"),
            "error must name the probed paths, got: {}",
            message
        );
    }

    #[test]
    fn test_from_pretrained_loads_real_vocab_file() {
        let dir = std::env::temp_dir().join(format!(
            "trustformers_wordpiece_from_pretrained_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir must be creatable");
        let vocab_path = dir.join("vocab.txt");
        std::fs::write(&vocab_path, "[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\n")
            .expect("fixture vocab must be writable");

        let model_dir = dir.to_str().expect("temp path must be UTF-8");
        let tokenizer =
            WordPieceTokenizer::from_pretrained(model_dir).expect("real vocab must load");
        assert_eq!(tokenizer.token_to_id("hello"), Some(4));
        assert_eq!(tokenizer.vocab_size(), 6);

        let _ = std::fs::remove_file(&vocab_path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// Regression: punctuation used to stay glued to the preceding word.
    #[test]
    fn test_basic_tokenize_splits_punctuation() {
        let tokenizer = WordPieceTokenizer::new(bert_like_vocab(), true);
        assert_eq!(
            tokenizer.basic_tokenize("hello, world!"),
            vec![
                "hello".to_string(),
                ",".to_string(),
                "world".to_string(),
                "!".to_string()
            ]
        );
    }

    #[test]
    fn test_basic_tokenize_pads_cjk_and_strips_accents() {
        let tokenizer = WordPieceTokenizer::new(bert_like_vocab(), true);
        assert_eq!(
            tokenizer.basic_tokenize("世界"),
            vec!["世".to_string(), "界".to_string()]
        );
        // Lowercasing + accent stripping turns "CAFÉ" into "cafe".
        assert_eq!(tokenizer.basic_tokenize("CAFÉ"), vec!["cafe".to_string()]);
        // Control characters are dropped, whitespace collapses.
        assert_eq!(
            tokenizer.basic_tokenize("hello\u{0}\tworld"),
            vec!["hello".to_string(), "world".to_string()]
        );
    }

    /// Accent stripping must run *before* lowercasing (as `BertNormalizer`
    /// does), and the cased path must not normalize the input at all.
    #[test]
    fn test_basic_tokenize_matches_bert_normalizer_order() {
        let lower = WordPieceTokenizer::new(bert_like_vocab(), true);
        // Precomposed and decomposed inputs both reduce to "cafe".
        assert_eq!(lower.basic_tokenize("CAFÉ"), vec!["cafe".to_string()]);
        assert_eq!(
            lower.basic_tokenize("CAFE\u{301}"),
            vec!["cafe".to_string()]
        );

        // Cased path: BERT applies neither accent stripping nor NFC, so the
        // decomposed form must survive verbatim.
        let cased = WordPieceTokenizer::new(bert_like_vocab(), false);
        assert_eq!(cased.basic_tokenize("CAFÉ"), vec!["CAFÉ".to_string()]);
        assert_eq!(
            cased.basic_tokenize("CAFE\u{301}"),
            vec!["CAFE\u{301}".to_string()]
        );
    }

    #[test]
    fn test_wordpiece_greedy_longest_match() {
        let tokenizer = WordPieceTokenizer::new(bert_like_vocab(), true);
        assert_eq!(
            tokenizer.wordpiece_tokenize("unwanted"),
            vec!["un".to_string(), "##want".to_string(), "##ed".to_string()]
        );
        assert_eq!(
            tokenizer.wordpiece_tokenize("running"),
            vec!["run".to_string(), "##ning".to_string()]
        );
    }

    /// Regression: a partially matching word used to keep its matched prefix.
    #[test]
    fn test_wordpiece_failure_emits_single_unk() {
        let tokenizer = WordPieceTokenizer::new(bert_like_vocab(), true);
        // "un" is in the vocabulary but "##known" is not, so BERT emits [UNK].
        assert_eq!(
            tokenizer.wordpiece_tokenize("unknown"),
            vec!["[UNK]".to_string()],
            "a failed word must collapse to a single unknown token"
        );
    }

    /// Regression: decode used to leak [CLS]/[SEP] because the replacements
    /// required surrounding spaces.
    #[test]
    fn test_decode_strips_special_tokens() {
        let tokenizer = WordPieceTokenizer::new(bert_like_vocab(), true);
        let encoded = tokenizer.encode("hello world").expect("encoding must succeed");
        let decoded = tokenizer.decode(&encoded.input_ids).expect("decoding must succeed");
        assert_eq!(decoded, "hello world");
        assert!(!decoded.contains("[CLS]"));
        assert!(!decoded.contains("[SEP]"));

        // Consecutive specials are removed too.
        let cls = tokenizer.token_to_id("[CLS]").expect("vocab has [CLS]");
        let sep = tokenizer.token_to_id("[SEP]").expect("vocab has [SEP]");
        let hello = tokenizer.token_to_id("hello").expect("vocab has hello");
        assert_eq!(
            tokenizer.decode(&[cls, cls, hello, sep, sep]).unwrap_or_default(),
            "hello"
        );

        // ...and can be kept on request.
        let kept = tokenizer.decode_tokens(&[cls, hello, sep], false);
        assert_eq!(kept, "[CLS] hello [SEP]");
    }

    #[test]
    fn test_decode_joins_continuation_pieces() {
        let tokenizer = WordPieceTokenizer::new(bert_like_vocab(), true);
        let encoded = tokenizer.encode("unwanted running").expect("encoding must succeed");
        let decoded = tokenizer.decode(&encoded.input_ids).expect("decoding must succeed");
        assert_eq!(decoded, "unwanted running");
    }

    #[test]
    fn test_special_tokens_mask_is_populated() {
        let tokenizer = WordPieceTokenizer::new(bert_like_vocab(), true);
        let encoded = tokenizer.encode("hello").expect("encoding must succeed");
        assert_eq!(encoded.special_tokens_mask, Some(vec![1, 0, 1]));
    }
}
