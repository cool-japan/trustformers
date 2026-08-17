use crate::errors::{Result, TrustformersError};
use std::collections::HashSet;

use super::config::GuidedGenerationConfig;
use super::grammar::Grammar;
use super::json_schema::JsonSchema;
use super::regex_constraint::RegexConstraint;

/// Constraint validator for guided generation.
///
/// Every configured constraint is enforced for real.  The two questions the
/// decoder asks - "may this token be appended?" ([`Self::validate_token`]) and
/// "is the text finished?" ([`Self::is_complete`]) - are answered with prefix
/// viability and full acceptance respectively, for each of the four constraint
/// kinds (regex, choice list, JSON schema, grammar).
#[derive(Debug)]
pub struct ConstraintValidator {
    regex: Option<RegexConstraint>,
    choice_list: Option<HashSet<String>>,
    json_schema: Option<JsonSchemaValidator>,
    grammar: Option<GrammarValidator>,
}

impl ConstraintValidator {
    pub fn new(config: &GuidedGenerationConfig) -> Result<Self> {
        let regex = match &config.regex_pattern {
            Some(pattern) => Some(RegexConstraint::new(pattern)?),
            None => None,
        };

        let choice_list = config
            .choice_list
            .as_ref()
            .map(|choices| choices.iter().cloned().collect::<HashSet<String>>());

        let json_schema = if let Some(schema) = &config.json_schema {
            Some(JsonSchemaValidator::new(schema)?)
        } else {
            None
        };

        let grammar = if let Some(grammar_config) = &config.grammar {
            Some(GrammarValidator::new(grammar_config)?)
        } else {
            None
        };

        Ok(Self {
            regex,
            choice_list,
            json_schema,
            grammar,
        })
    }

    /// Whether appending `new_token` to `current_text` keeps every constraint
    /// satisfiable.
    ///
    /// A token is admissible when the resulting text can *still grow into* a
    /// conforming string - not only when it already conforms - otherwise no
    /// constrained sequence longer than one token could ever be decoded.
    pub fn validate_token(
        &self,
        current_text: &str,
        new_token: &str,
        _tokenizer_fn: Option<&dyn Fn(usize) -> String>,
    ) -> bool {
        let potential_text = format!("{}{}", current_text, new_token);

        // Check regex constraint
        if let Some(regex) = &self.regex {
            if !regex.is_viable_prefix(&potential_text) {
                return false;
            }
        }

        // Check choice list constraint
        if let Some(choices) = &self.choice_list {
            if !choices.contains(&potential_text) && !self.is_valid_prefix(&potential_text, choices)
            {
                return false;
            }
        }

        // Check JSON schema constraint
        if let Some(json_validator) = &self.json_schema {
            if !json_validator.validate_partial(&potential_text) {
                return false;
            }
        }

        // Check grammar constraint
        if let Some(grammar_validator) = &self.grammar {
            if !grammar_validator.validate_partial(&potential_text) {
                return false;
            }
        }

        true
    }

    /// Whether `text` satisfies every configured constraint completely.
    ///
    /// The regex constraint is anchored at both ends here: a document that
    /// merely *contains* a match is not a conforming document.
    pub fn is_complete(&self, text: &str) -> bool {
        // Check if the current text satisfies all constraints completely
        if let Some(regex) = &self.regex {
            if !regex.is_full_match(text) {
                return false;
            }
        }

        if let Some(choices) = &self.choice_list {
            if !choices.contains(text) {
                return false;
            }
        }

        if let Some(json_validator) = &self.json_schema {
            if !json_validator.validate_complete(text) {
                return false;
            }
        }

        if let Some(grammar_validator) = &self.grammar {
            if !grammar_validator.validate_complete(text) {
                return false;
            }
        }

        true
    }

    /// The compiled regular-expression constraint, if one was configured.
    pub fn regex(&self) -> Option<&RegexConstraint> {
        self.regex.as_ref()
    }

    fn is_valid_prefix(&self, text: &str, choices: &HashSet<String>) -> bool {
        choices.iter().any(|choice| choice.starts_with(text))
    }

    pub fn filter_valid_tokens(
        &self,
        current_text: &str,
        token_logits: &[(usize, f32)],
        tokenizer_fn: &dyn Fn(usize) -> String,
    ) -> Vec<(usize, f32)> {
        token_logits
            .iter()
            .filter(|(token_id, _)| {
                let token_str = tokenizer_fn(*token_id);
                self.validate_token(current_text, &token_str, Some(tokenizer_fn))
            })
            .cloned()
            .collect()
    }
}

/// JSON Schema validator for constrained generation.
///
/// The supplied schema is compiled once and genuinely enforced - see
/// [`JsonSchema`] for the list of supported keywords.
#[derive(Debug)]
pub struct JsonSchemaValidator {
    schema: JsonSchema,
}

impl JsonSchemaValidator {
    pub fn new(schema: &str) -> Result<Self> {
        let schema = JsonSchema::parse(schema).map_err(|error| {
            TrustformersError::invalid_input(format!("invalid JSON schema: {error}"))
        })?;
        Ok(Self { schema })
    }

    /// The compiled schema.
    pub fn schema(&self) -> &JsonSchema {
        &self.schema
    }

    /// Check whether `text` can still grow into a schema-conforming document.
    ///
    /// Two things are checked: the bracket/quote structure must be consistent
    /// with *some* completion, and if `text` already parses as a whole JSON
    /// document it must satisfy the schema.  A genuinely partial document
    /// (`{"a": ` and friends) cannot be validated against value constraints
    /// yet, and is accepted so decoding can continue.
    pub fn validate_partial(&self, text: &str) -> bool {
        if !Self::structure_is_open(text) {
            return false;
        }
        match serde_json::from_str::<serde_json::Value>(text) {
            Ok(document) => self.schema.validate(&document).is_ok(),
            Err(_) => true,
        }
    }

    /// Check whether `text` is a complete document that satisfies the schema.
    pub fn validate_complete(&self, text: &str) -> bool {
        self.schema.validate_text(text).is_ok()
    }

    /// Balanced-bracket scan that tolerates unclosed structures but rejects
    /// closers that never had an opener.
    fn structure_is_open(text: &str) -> bool {
        let mut stack = Vec::new();
        let mut in_string = false;
        let mut escape_next = false;

        for ch in text.chars() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match ch {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '{' | '[' if !in_string => stack.push(ch),
                '}' if !in_string => {
                    if stack.last() == Some(&'{') {
                        stack.pop();
                    } else {
                        return false;
                    }
                },
                ']' if !in_string => {
                    if stack.last() == Some(&'[') {
                        stack.pop();
                    } else {
                        return false;
                    }
                },
                _ => {},
            }
        }

        true
    }
}

/// Grammar validator for constrained generation.
///
/// The BNF definition is compiled into a [`Grammar`] and checked with an
/// Earley recogniser, so partial validation answers the real question - "can
/// this text still become a sentence of the language?" - rather than accepting
/// everything.  A definition that contains no rules at all means "no grammar
/// constraint" and accepts any text.
#[derive(Debug)]
pub struct GrammarValidator {
    grammar: Option<Grammar>,
}

impl GrammarValidator {
    pub fn new(grammar: &str) -> Result<Self> {
        let grammar = Grammar::parse(grammar).map_err(|error| {
            TrustformersError::invalid_input(format!("invalid grammar: {error}"))
        })?;
        Ok(Self { grammar })
    }

    /// The compiled grammar, or `None` when the definition declared no rules.
    pub fn grammar(&self) -> Option<&Grammar> {
        self.grammar.as_ref()
    }

    /// `true` when `text` is a prefix of some sentence in the language.
    pub fn validate_partial(&self, text: &str) -> bool {
        match &self.grammar {
            Some(grammar) => grammar.recognize(text).viable,
            None => true,
        }
    }

    /// `true` when `text` is itself a sentence in the language.
    pub fn validate_complete(&self, text: &str) -> bool {
        match &self.grammar {
            Some(grammar) => grammar.recognize(text).complete,
            None => true,
        }
    }

    /// Terminal strings that may legally follow `prefix`.
    ///
    /// Returns an empty vector when the grammar is unconstrained or when
    /// `prefix` is already a dead end.
    pub fn get_valid_next_tokens(&self, prefix: &str) -> Vec<String> {
        match &self.grammar {
            Some(grammar) => grammar.recognize(prefix).next_terminals,
            None => Vec::new(),
        }
    }
}
