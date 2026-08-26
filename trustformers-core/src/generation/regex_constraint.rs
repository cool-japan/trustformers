//! Anchored regular-expression constraints for guided generation.
//!
//! Constrained decoding asks two different questions about a partially decoded
//! string, and answering either one with an unanchored `Regex::is_match` is
//! wrong:
//!
//! * **"Is this finished?"** - the *whole* generated text must match the
//!   pattern, not merely contain a match somewhere.  `\A(?:pattern)\z` is
//!   compiled once for this and evaluated by [`RegexConstraint::is_full_match`].
//! * **"Can this still become a match?"** - the standard `regex` API cannot
//!   answer this at all, because it only reports complete matches.  A prefix
//!   such as `hello ` is not a match of `hello\s+there`, yet forbidding it would
//!   make the pattern undecodable.  [`RegexConstraint::is_viable_prefix`]
//!   answers it exactly, by driving a DFA that is anchored at the start of the
//!   haystack and asking whether the automaton is still alive.
//!
//! The DFA is built with [`MatchKind::All`] so that determinization keeps every
//! branch alive after an earlier branch has already matched: for `a|ab` the
//! prefix `a` is a match *and* `ab` must remain reachable.  A state that is not
//! the dead state always has at least one live NFA thread, and every live thread
//! of a Thompson NFA can reach the match state, so "not dead" is exactly "some
//! continuation matches".

use regex::Regex;
use regex_automata::dfa::{dense, Automaton, StartKind};
use regex_automata::util::primitives::StateID;
use regex_automata::{Anchored, Input, MatchKind};

use crate::errors::{Result, TrustformersError};

/// Upper bound on the memory the prefix automaton may occupy.
///
/// Determinization is worst-case exponential, so a pathological pattern is
/// rejected at construction time rather than being allowed to exhaust memory
/// during decoding.
const PREFIX_DFA_SIZE_LIMIT: usize = 8 * (1 << 20);

/// A regular expression compiled for constrained decoding.
///
/// Both questions the decoder needs are answered exactly; nothing is guessed
/// from a hard-coded list of candidate continuations.
#[derive(Debug)]
pub struct RegexConstraint {
    pattern: String,
    full_match: Regex,
    prefix_dfa: dense::DFA<Vec<u32>>,
}

impl RegexConstraint {
    /// Compile `pattern` for constrained decoding.
    ///
    /// Fails when the pattern is not valid, or when it cannot be turned into a
    /// prefix automaton within `PREFIX_DFA_SIZE_LIMIT`.  Failing is
    /// deliberate: a constraint that cannot be enforced must not silently
    /// degrade into "accept everything".
    pub fn new(pattern: &str) -> Result<Self> {
        let full_match = Regex::new(&format!(r"\A(?:{pattern})\z")).map_err(|error| {
            TrustformersError::invalid_input(format!("invalid regex pattern `{pattern}`: {error}"))
        })?;

        let prefix_dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(StartKind::Anchored)
                    .match_kind(MatchKind::All)
                    .unicode_word_boundary(true)
                    .dfa_size_limit(Some(PREFIX_DFA_SIZE_LIMIT))
                    .determinize_size_limit(Some(PREFIX_DFA_SIZE_LIMIT)),
            )
            .build(pattern)
            .map_err(|error| {
                TrustformersError::invalid_input(format!(
                    "regex pattern `{pattern}` cannot be compiled into a prefix automaton for \
                     constrained decoding: {error}"
                ))
            })?;

        Ok(Self {
            pattern: pattern.to_string(),
            full_match,
            prefix_dfa,
        })
    }

    /// The source pattern this constraint was built from.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// `true` when `text` matches the pattern from its first to its last byte.
    pub fn is_full_match(&self, text: &str) -> bool {
        self.full_match.is_match(text)
    }

    /// `true` when some suffix can be appended to `text` to obtain a full match.
    ///
    /// Note that every full match is also a viable prefix (of itself), so a
    /// decoder may use this as its per-token admissibility test and
    /// [`RegexConstraint::is_full_match`] as its stopping test.
    ///
    /// If the automaton reaches a *quit* state - only possible for a Unicode
    /// word boundary applied to invalid UTF-8 - the answer is not decidable by
    /// this DFA and the prefix is accepted, so decoding is never blocked by an
    /// undecided case.
    pub fn is_viable_prefix(&self, text: &str) -> bool {
        let input = Input::new(text).anchored(Anchored::Yes);
        let mut state = match self.prefix_dfa.start_state_forward(&input) {
            Ok(state) => state,
            Err(_) => return true,
        };

        for &byte in text.as_bytes() {
            state = self.prefix_dfa.next_state(state, byte);
            if self.prefix_dfa.is_dead_state(state) {
                return false;
            }
            if self.prefix_dfa.is_quit_state(state) {
                return true;
            }
        }

        // A dense DFA reports matches one byte late, so the state reached after
        // a match may still be a *match* state even though determinization left
        // no live NFA thread in it: `[0-9]{3}` on `1234` lands there.  Only in
        // that case is "not dead" insufficient, and only then is a continuation
        // probe needed.
        if self.prefix_dfa.is_match_state(state) {
            return self.has_live_continuation(state);
        }

        true
    }

    /// Whether anything at all can still happen from `state`: either the text
    /// ends here as a match, or some byte leads somewhere that is not dead.
    fn has_live_continuation(&self, state: StateID) -> bool {
        if self.prefix_dfa.is_match_state(self.prefix_dfa.next_eoi_state(state)) {
            return true;
        }
        (u8::MIN..=u8::MAX).any(|byte| {
            let next = self.prefix_dfa.next_state(state, byte);
            !self.prefix_dfa.is_dead_state(next)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_match_is_anchored_at_both_ends() {
        // Regression: the validator used an unanchored `is_match`, so a
        // completely unrelated prefix satisfied a "complete" constraint.
        let constraint = RegexConstraint::new(r"\d+").expect("compile");
        assert!(constraint.is_full_match("123"));
        assert!(!constraint.is_full_match("abc123"));
        assert!(!constraint.is_full_match("123abc"));
        assert!(!constraint.is_full_match(""));
    }

    #[test]
    fn test_viable_prefix_accepts_prefixes_no_hard_coded_list_could_know() {
        // Regression: prefix viability used to be guessed by appending a fixed
        // list of strings (" ", "\\s", " world", "  world"), so a legitimate
        // prefix of any pattern that did not end in "world" was rejected.
        let constraint = RegexConstraint::new(r"hello\s+there").expect("compile");
        assert!(constraint.is_viable_prefix(""));
        assert!(constraint.is_viable_prefix("h"));
        assert!(constraint.is_viable_prefix("hello"));
        assert!(constraint.is_viable_prefix("hello "));
        assert!(constraint.is_viable_prefix("hello  the"));
        assert!(constraint.is_viable_prefix("hello there"));
        assert!(constraint.is_full_match("hello there"));

        assert!(!constraint.is_viable_prefix("help"));
        assert!(!constraint.is_viable_prefix("hello x"));
        assert!(!constraint.is_viable_prefix("hello there!"));
    }

    #[test]
    fn test_viable_prefix_handles_multibyte_text_without_panicking() {
        // Regression: the old heuristic sliced `&text[i..]` at byte offsets,
        // which panics as soon as the decoded text contains a non-ASCII
        // character.
        let constraint = RegexConstraint::new(r"\d+").expect("compile");
        assert!(!constraint.is_viable_prefix("日本語"));
        assert!(!constraint.is_full_match("日本語"));

        let unicode = RegexConstraint::new(r"日本\p{Han}+").expect("compile");
        assert!(unicode.is_viable_prefix("日"));
        assert!(unicode.is_viable_prefix("日本"));
        assert!(unicode.is_viable_prefix("日本語"));
        assert!(unicode.is_full_match("日本語"));
        assert!(!unicode.is_viable_prefix("日x"));
    }

    #[test]
    fn test_viable_prefix_keeps_longer_alternatives_alive_after_a_match() {
        // `MatchKind::All` is what makes this work: with leftmost-first
        // semantics the automaton would stop after matching `a`.
        let constraint = RegexConstraint::new("a|ab").expect("compile");
        assert!(constraint.is_viable_prefix("a"));
        assert!(constraint.is_viable_prefix("ab"));
        assert!(constraint.is_full_match("a"));
        assert!(constraint.is_full_match("ab"));
        assert!(!constraint.is_viable_prefix("abc"));
    }

    #[test]
    fn test_viable_prefix_of_a_bounded_repetition() {
        let constraint = RegexConstraint::new("[0-9]{3}").expect("compile");
        assert!(constraint.is_viable_prefix("1"));
        assert!(constraint.is_viable_prefix("12"));
        assert!(constraint.is_viable_prefix("123"));
        assert!(!constraint.is_viable_prefix("1234"));
        assert!(!constraint.is_full_match("12"));
        assert!(constraint.is_full_match("123"));
    }

    #[test]
    fn test_alternation_prefix_narrows_as_text_grows() {
        let constraint = RegexConstraint::new("(?:yes|no|maybe)").expect("compile");
        assert!(constraint.is_viable_prefix("y"));
        assert!(constraint.is_viable_prefix("m"));
        assert!(!constraint.is_viable_prefix("ye5"));
        assert!(constraint.is_full_match("maybe"));
        assert!(!constraint.is_full_match("may"));
    }

    #[test]
    fn test_viable_prefix_after_a_repeated_match_stays_exact() {
        // The DFA reports matches one byte late, so `abab` sits in a state that
        // is both a match state *and* has live threads, while `abx` sits in a
        // match-delayed state with none.  Both must be classified correctly.
        let constraint = RegexConstraint::new("(?:ab)+").expect("compile");
        assert!(constraint.is_viable_prefix("a"));
        assert!(constraint.is_viable_prefix("ab"));
        assert!(constraint.is_viable_prefix("aba"));
        assert!(constraint.is_viable_prefix("abab"));
        assert!(constraint.is_full_match("abab"));
        assert!(!constraint.is_full_match("aba"));
        assert!(!constraint.is_viable_prefix("abx"));
        assert!(!constraint.is_viable_prefix("ababx"));
    }

    #[test]
    fn test_invalid_pattern_is_rejected() {
        assert!(RegexConstraint::new("[invalid(").is_err());
    }

    #[test]
    fn test_pattern_accessor_round_trips() {
        let constraint = RegexConstraint::new(r"\w+").expect("compile");
        assert_eq!(constraint.pattern(), r"\w+");
    }

    #[test]
    fn test_word_boundary_pattern_compiles_and_matches() {
        let constraint = RegexConstraint::new(r"\bcat\b").expect("compile");
        assert!(constraint.is_full_match("cat"));
        assert!(constraint.is_viable_prefix("c"));
        assert!(!constraint.is_viable_prefix("dog"));
    }
}
