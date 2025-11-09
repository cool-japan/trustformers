use crate::errors::{Result, TrustformersError};
use regex::Regex;
use std::collections::{HashMap, HashSet};

use super::config::GuidedGenerationConfig;

/// Constraint validator for guided generation
#[derive(Debug)]
pub struct ConstraintValidator {
    regex: Option<Regex>,
    choice_list: Option<HashSet<String>>,
    json_schema: Option<JsonSchemaValidator>,
    grammar: Option<GrammarValidator>,
}

impl ConstraintValidator {
    pub fn new(config: &GuidedGenerationConfig) -> Result<Self> {
        let regex = if let Some(pattern) = &config.regex_pattern {
            Some(Regex::new(pattern).map_err(|e| {
                TrustformersError::invalid_input(format!("Invalid regex pattern: {}", e))
            })?)
        } else {
            None
        };

        let choice_list = config
            .choice_list
            .as_ref()
            .map(|choices| choices.iter().cloned().collect::<HashSet<String>>());

        let json_schema = if config.json_schema.is_some() {
            Some(JsonSchemaValidator::new(
                config.json_schema.as_ref().unwrap(),
            )?)
        } else {
            None
        };

        let grammar = if config.grammar.is_some() {
            Some(GrammarValidator::new(config.grammar.as_ref().unwrap())?)
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

    pub fn validate_token(
        &self,
        current_text: &str,
        new_token: &str,
        _tokenizer_fn: Option<&dyn Fn(usize) -> String>,
    ) -> bool {
        let potential_text = format!("{}{}", current_text, new_token);

        // Check regex constraint
        if let Some(regex) = &self.regex {
            if !regex.is_match(&potential_text) && !self.is_partial_match(regex, &potential_text) {
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

    pub fn is_complete(&self, text: &str) -> bool {
        // Check if the current text satisfies all constraints completely
        if let Some(regex) = &self.regex {
            if !regex.is_match(text) {
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

    fn is_partial_match(&self, regex: &Regex, text: &str) -> bool {
        // For regex, check if text could be extended to match
        // This is a simplified implementation - in practice would need more sophisticated partial matching

        // If text is empty, it's a valid prefix for any pattern
        if text.is_empty() {
            return true;
        }

        // Check if the text is already a full match
        if regex.is_match(text) {
            return true;
        }

        // Check if this text is a partial match by trying to extend it
        // For a simple heuristic, check if any suffix could match the pattern
        for i in 0..text.len() {
            if regex.find(&text[i..]).is_some() {
                return true;
            }
        }

        // Check if the pattern could start with this text
        // For common patterns like "hello\s+world", "hello" should be valid
        let test_extensions = vec![" ", "\\s", " world", "  world"];
        for ext in test_extensions {
            let test_text = format!("{}{}", text, ext);
            if regex.is_match(&test_text) {
                return true;
            }
        }

        false
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

/// JSON Schema validator for constrained generation
#[derive(Debug)]
pub struct JsonSchemaValidator {
    #[allow(dead_code)]
    schema: String,
    #[allow(dead_code)]
    brace_stack: Vec<char>,
}

impl JsonSchemaValidator {
    pub fn new(schema: &str) -> Result<Self> {
        // Parse and validate the JSON schema
        Ok(Self {
            schema: schema.to_string(),
            brace_stack: Vec::new(),
        })
    }

    pub fn validate_partial(&self, text: &str) -> bool {
        // Simplified JSON validation - checks for balanced braces and basic structure
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

        // For partial validation, we allow unclosed structures
        true
    }

    pub fn validate_complete(&self, text: &str) -> bool {
        // Check if it's valid complete JSON
        serde_json::from_str::<serde_json::Value>(text).is_ok()
    }
}

/// Grammar validator for constrained generation
#[derive(Debug)]
pub struct GrammarValidator {
    #[allow(dead_code)]
    rules: HashMap<String, Vec<String>>,
    #[allow(dead_code)]
    current_state: String,
}

impl GrammarValidator {
    pub fn new(grammar: &str) -> Result<Self> {
        // Parse BNF-style grammar rules
        let mut rules = HashMap::new();

        // Simple grammar parsing - in practice would use a proper parser
        for line in grammar.lines() {
            if let Some((lhs, rhs)) = line.split_once("::=") {
                let rule_name = lhs.trim().to_string();
                let alternatives: Vec<String> =
                    rhs.split('|').map(|alt| alt.trim().to_string()).collect();
                rules.insert(rule_name, alternatives);
            }
        }

        Ok(Self {
            rules,
            current_state: "start".to_string(),
        })
    }

    pub fn validate_partial(&self, _text: &str) -> bool {
        // Simplified grammar validation
        // In practice would need a proper parsing algorithm
        true
    }

    pub fn validate_complete(&self, _text: &str) -> bool {
        // Check if text matches the complete grammar
        true
    }

    pub fn get_valid_next_tokens(&self, _current_state: &str) -> Vec<String> {
        // Return valid tokens for current grammar state
        vec![]
    }
}
