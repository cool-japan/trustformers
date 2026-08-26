/// Tests for ConstraintValidator, JsonSchemaValidator, GrammarValidator
#[cfg(test)]
mod tests {
    use crate::generation::config::GuidedGenerationConfig;
    use crate::generation::constraints::{
        ConstraintValidator, GrammarValidator, JsonSchemaValidator,
    };

    fn guided_config_empty() -> GuidedGenerationConfig {
        GuidedGenerationConfig {
            regex_pattern: None,
            grammar: None,
            json_schema: None,
            choice_list: None,
            max_violations: None,
            backtrack_on_violation: false,
            cfg: None,
        }
    }

    // ---- ConstraintValidator tests ----

    #[test]
    fn test_constraint_validator_no_constraints_allows_everything() {
        let cfg = guided_config_empty();
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");
        assert!(validator.validate_token("hello", " world", None));
        assert!(validator.validate_token("", "anything", None));
    }

    #[test]
    fn test_constraint_validator_is_complete_no_constraints_returns_true() {
        let cfg = guided_config_empty();
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");
        assert!(validator.is_complete("some text"));
        assert!(validator.is_complete(""));
    }

    #[test]
    fn test_constraint_validator_invalid_regex_returns_error() {
        let mut cfg = guided_config_empty();
        cfg.regex_pattern = Some("[invalid(".to_string());
        let result = ConstraintValidator::new(&cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_constraint_validator_valid_regex_creation() {
        let mut cfg = guided_config_empty();
        cfg.regex_pattern = Some(r"\d+".to_string());
        let result = ConstraintValidator::new(&cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_constraint_validator_regex_accepts_real_prefixes() {
        // Regression: prefix viability used to be guessed by appending a fixed
        // list of strings (" ", "\\s", " world", "  world"), so the legitimate
        // prefix "hello " of `hello\s+there` was rejected and the constraint
        // could never be satisfied.
        let mut cfg = guided_config_empty();
        cfg.regex_pattern = Some(r"hello\s+there".to_string());
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");

        assert!(validator.validate_token("", "hello", None));
        assert!(validator.validate_token("hello", " ", None));
        assert!(validator.validate_token("hello ", "the", None));
        assert!(validator.validate_token("hello ", "there", None));
        assert!(!validator.validate_token("hello ", "x", None));
        assert!(!validator.validate_token("hel", "p", None));
    }

    #[test]
    fn test_constraint_validator_regex_multibyte_token_does_not_panic() {
        // Regression: the old heuristic sliced `&text[i..]` at raw byte offsets,
        // which panics the moment a non-ASCII token is appended.
        let mut cfg = guided_config_empty();
        cfg.regex_pattern = Some(r"\d+".to_string());
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");

        assert!(!validator.validate_token("", "日本語", None));
        assert!(!validator.is_complete("日本語"));
        assert!(validator.validate_token("12", "3", None));
        assert!(validator.is_complete("123"));
    }

    #[test]
    fn test_constraint_validator_regex_is_complete_is_anchored() {
        // Regression: `is_complete` used an unanchored `is_match`, so any text
        // that merely *contained* a match counted as a finished document.
        let mut cfg = guided_config_empty();
        cfg.regex_pattern = Some(r"\d+".to_string());
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");

        assert!(validator.is_complete("42"));
        assert!(!validator.is_complete("abc42"));
        assert!(!validator.is_complete("42abc"));
    }

    #[test]
    fn test_constraint_validator_regex_filters_tokens() {
        let mut cfg = guided_config_empty();
        cfg.regex_pattern = Some("(?:yes|no)".to_string());
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");

        let tokens: Vec<(usize, f32)> = vec![(0, 1.0), (1, 0.5), (2, 0.25)];
        let tokenizer = |id: usize| match id {
            0 => "y".to_string(),
            1 => "n".to_string(),
            _ => "q".to_string(),
        };
        let kept = validator.filter_valid_tokens("", &tokens, &tokenizer);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().all(|(id, _)| *id == 0 || *id == 1));
    }

    #[test]
    fn test_constraint_validator_exposes_the_compiled_regex() {
        let mut cfg = guided_config_empty();
        cfg.regex_pattern = Some(r"\w+".to_string());
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");
        assert_eq!(validator.regex().map(|regex| regex.pattern()), Some(r"\w+"));

        let unconstrained =
            ConstraintValidator::new(&guided_config_empty()).expect("Should create validator");
        assert!(unconstrained.regex().is_none());
    }

    #[test]
    fn test_constraint_validator_choice_list_valid_prefix() {
        let mut cfg = guided_config_empty();
        cfg.choice_list = Some(vec![
            "yes".to_string(),
            "no".to_string(),
            "maybe".to_string(),
        ]);
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");
        // "y" is a prefix of "yes"
        assert!(validator.validate_token("", "y", None));
        // "n" is a prefix of "no"
        assert!(validator.validate_token("", "n", None));
    }

    #[test]
    fn test_constraint_validator_choice_list_complete_match() {
        let mut cfg = guided_config_empty();
        cfg.choice_list = Some(vec!["yes".to_string(), "no".to_string()]);
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");
        assert!(validator.is_complete("yes"));
        assert!(validator.is_complete("no"));
        assert!(!validator.is_complete("maybe"));
    }

    #[test]
    fn test_constraint_validator_choice_list_invalid_prefix() {
        let mut cfg = guided_config_empty();
        cfg.choice_list = Some(vec!["yes".to_string(), "no".to_string()]);
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");
        // "z" doesn't start any choice
        assert!(!validator.validate_token("", "z", None));
    }

    #[test]
    fn test_constraint_validator_filter_valid_tokens_empty() {
        let cfg = guided_config_empty();
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");
        let empty: Vec<(usize, f32)> = vec![];
        let result = validator.filter_valid_tokens("text", &empty, &|id| format!("tok{}", id));
        assert!(result.is_empty());
    }

    #[test]
    fn test_constraint_validator_filter_valid_tokens_no_constraints() {
        let cfg = guided_config_empty();
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");
        let tokens: Vec<(usize, f32)> = vec![(0, 1.0), (1, 0.5), (2, 0.3)];
        let result = validator.filter_valid_tokens("text", &tokens, &|id| format!("tok{}", id));
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_constraint_validator_filter_tokens_with_choice_list() {
        let mut cfg = guided_config_empty();
        cfg.choice_list = Some(vec!["yes".to_string(), "no".to_string()]);
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");
        let tokens: Vec<(usize, f32)> = vec![
            (0, 1.0), // maps to "y" - valid prefix of "yes"
            (1, 0.5), // maps to "z" - invalid
        ];
        let tokenizer = |id: usize| if id == 0 { "y".to_string() } else { "z".to_string() };
        let result = validator.filter_valid_tokens("", &tokens, &tokenizer);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
    }

    // ---- JsonSchemaValidator tests ----

    #[test]
    fn test_json_schema_validator_creation() {
        let schema = r#"{"type": "object"}"#;
        let validator = JsonSchemaValidator::new(schema);
        assert!(validator.is_ok());
    }

    #[test]
    fn test_json_schema_validator_partial_valid_open_brace() {
        let validator = JsonSchemaValidator::new("{}").expect("Should create");
        assert!(validator.validate_partial("{"));
        assert!(validator.validate_partial("{\"key\":"));
    }

    #[test]
    fn test_json_schema_validator_partial_valid_complete_object() {
        let validator = JsonSchemaValidator::new("{}").expect("Should create");
        assert!(validator.validate_partial(r#"{"key": "value"}"#));
    }

    #[test]
    fn test_json_schema_validator_partial_mismatched_braces() {
        let validator = JsonSchemaValidator::new("{}").expect("Should create");
        assert!(!validator.validate_partial("}"));
        assert!(!validator.validate_partial("]"));
    }

    #[test]
    fn test_json_schema_validator_partial_empty_string() {
        let validator = JsonSchemaValidator::new("{}").expect("Should create");
        assert!(validator.validate_partial(""));
    }

    #[test]
    fn test_json_schema_validator_partial_array() {
        let validator = JsonSchemaValidator::new("{}").expect("Should create");
        assert!(validator.validate_partial("[1, 2, 3]"));
        assert!(validator.validate_partial("["));
    }

    #[test]
    fn test_json_schema_validator_partial_string_with_braces() {
        let validator = JsonSchemaValidator::new("{}").expect("Should create");
        // Braces inside strings should not count
        assert!(validator.validate_partial(r#"{"key": "some } value"}"#));
    }

    #[test]
    fn test_json_schema_validator_complete_valid_json() {
        let validator = JsonSchemaValidator::new("{}").expect("Should create");
        assert!(validator.validate_complete(r#"{"key": "value"}"#));
        assert!(validator.validate_complete("42"));
        assert!(validator.validate_complete(r#""hello""#));
    }

    #[test]
    fn test_json_schema_validator_complete_invalid_json() {
        let validator = JsonSchemaValidator::new("{}").expect("Should create");
        assert!(!validator.validate_complete("{invalid}"));
        assert!(!validator.validate_complete(""));
        assert!(!validator.validate_complete("{"));
    }

    #[test]
    fn test_json_schema_validator_enforces_the_schema() {
        // Regression: the validator used to ignore `self.schema` entirely and
        // only check that the text was well-formed JSON.
        let validator = JsonSchemaValidator::new(r#"{"type":"object","required":["name"]}"#)
            .expect("Should create");
        assert!(validator.validate_complete(r#"{"name": "a"}"#));
        assert!(
            !validator.validate_complete("42"),
            "a number is well-formed JSON but violates the schema"
        );
        assert!(
            !validator.validate_complete(r#"{"other": 1}"#),
            "the required property is missing"
        );
    }

    #[test]
    fn test_json_schema_validator_partial_rejects_finished_violations() {
        let validator = JsonSchemaValidator::new(r#"{"type":"object"}"#).expect("Should create");
        // Still growing: cannot be judged yet.
        assert!(validator.validate_partial(r#"{"a": "#));
        // Already a complete document, and it violates the schema.
        assert!(!validator.validate_partial("42"));
    }

    #[test]
    fn test_json_schema_validator_rejects_broken_schema_text() {
        assert!(JsonSchemaValidator::new("{not json}").is_err());
    }

    // ---- GrammarValidator tests ----

    #[test]
    fn test_grammar_validator_creation_simple() {
        let grammar = "expr ::= term | term '+' expr\nterm ::= 'a' | 'b'";
        let result = GrammarValidator::new(grammar);
        assert!(result.is_ok());
    }

    #[test]
    fn test_grammar_validator_creation_empty() {
        let result = GrammarValidator::new("");
        assert!(result.is_ok());
    }

    #[test]
    fn test_grammar_validator_validate_partial_rejects_dead_ends() {
        // Regression: `validate_partial` used to `return true` unconditionally,
        // so grammar-constrained generation was a no-op.
        let validator = GrammarValidator::new("start ::= 'hello'").expect("Should create");
        assert!(validator.validate_partial("hello"));
        assert!(validator.validate_partial("hel"));
        assert!(validator.validate_partial(""));
        assert!(
            !validator.validate_partial("help"),
            "`help` can never become `hello`"
        );
        assert!(!validator.validate_partial("x"));
    }

    #[test]
    fn test_grammar_validator_validate_complete_rejects_non_sentences() {
        // Regression: `validate_complete` used to `return true` unconditionally.
        let validator = GrammarValidator::new("start ::= 'hello'").expect("Should create");
        assert!(validator.validate_complete("hello"));
        assert!(!validator.validate_complete("anything"));
        assert!(!validator.validate_complete("hell"));
        assert!(!validator.validate_complete("helloo"));
    }

    #[test]
    fn test_grammar_validator_enforces_multi_rule_grammar() {
        let grammar = "start ::= noun verb\nnoun ::= 'cat' | 'dog'\nverb ::= 'runs' | 'walks'";
        let validator = GrammarValidator::new(grammar).expect("Should create");
        assert!(validator.validate_complete("catruns"));
        assert!(validator.validate_complete("dogwalks"));
        assert!(!validator.validate_complete("catflies"));
        assert!(validator.validate_partial("cat"));
        assert!(!validator.validate_partial("bird"));
    }

    #[test]
    fn test_grammar_validator_empty_grammar_is_unconstrained() {
        let validator = GrammarValidator::new("").expect("Should create");
        assert!(validator.grammar().is_none());
        assert!(validator.validate_partial("anything at all"));
        assert!(validator.validate_complete("anything at all"));
    }

    #[test]
    fn test_grammar_validator_get_valid_next_tokens_returns_continuations() {
        let validator = GrammarValidator::new("start ::= 'a' | 'b'").expect("Should create");
        // `start` is not a viable prefix of the language, so nothing follows.
        assert_eq!(validator.get_valid_next_tokens("start").len(), 0);

        let words = GrammarValidator::new("start ::= 'foo' | 'fizz'").expect("Should create");
        let mut tokens = words.get_valid_next_tokens("f");
        tokens.sort();
        assert_eq!(tokens, vec!["izz".to_string(), "oo".to_string()]);
    }

    #[test]
    fn test_grammar_validator_rejects_undefined_rule() {
        assert!(GrammarValidator::new("start ::= undefined_rule").is_err());
    }

    #[test]
    fn test_grammar_validator_multiline_grammar() {
        let grammar = "start ::= noun verb\nnoun ::= 'cat' | 'dog'\nverb ::= 'runs' | 'walks'";
        let result = GrammarValidator::new(grammar);
        assert!(result.is_ok());
    }

    #[test]
    fn test_constraint_validator_lcg_random_tokens_no_constraint() {
        let cfg = guided_config_empty();
        let validator = ConstraintValidator::new(&cfg).expect("Should create validator");
        let mut s = 42u64;
        let mut all_valid = true;
        for _ in 0..25 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let tok_str = format!("token_{}", s % 10000);
            if !validator.validate_token("prefix", &tok_str, None) {
                all_valid = false;
            }
        }
        assert!(all_valid, "Without constraints, all tokens should be valid");
    }
}
