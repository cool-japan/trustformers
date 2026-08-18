//! Small boolean expression language for [`super::config_management`]'s
//! `ConditionalRequirement::condition` strings.
//!
//! Configuration schemas describe "field X is required when `<condition>`"
//! rules. The condition text is free-form and user-authored (schema
//! authors, not end users, write these), so it needs a real grammar rather
//! than a hand-rolled substring search. This module implements exactly that:
//! a tokenizer, a recursive-descent parser producing an [`Expr`] AST, and an
//! evaluator that runs the AST against a JSON config object.
//!
//! # Grammar
//!
//! ```text
//! expr       := or_expr
//! or_expr    := and_expr ( "||" and_expr )*
//! and_expr   := comparison ( "&&" comparison )*
//! comparison := primary ( comp_op primary )?
//! comp_op    := "==" | "!=" | "<=" | ">=" | "<" | ">"
//! primary    := "(" expr ")" | literal | field_path
//! literal    := string | number | "true" | "false"
//! field_path := identifier ( "." identifier )*
//! ```
//!
//! `&&` binds tighter than `||`, matching every C-family language and
//! avoiding a surprise for schema authors. Comparisons are not chainable
//! (`a == b == c` is a parse error, not `(a == b) == c`) because chained
//! comparisons on arbitrary JSON values have no single obviously-correct
//! reading.
//!
//! # Value semantics
//!
//! - `==`/`!=` compare a field's actual JSON value against a literal by
//!   type-appropriate equality: string-to-string, number-to-number
//!   (compared as `f64`), bool-to-bool. A field whose actual JSON type
//!   disagrees with the literal's type is never equal (`5 == "5"` is
//!   `false`, not a type error) -- this mirrors how the old substring-based
//!   evaluator only ever matched strings, so config authors relying on that
//!   behavior are not surprised, while every other combination now has a
//!   defined answer instead of silently evaluating to `false` for reasons
//!   the caller cannot discover.
//! - `<`, `<=`, `>`, `>=` require both sides to be numbers (after resolving
//!   field paths) and produce a [`ConditionError`] otherwise -- ordering a
//!   string or bool is a schema-author mistake worth surfacing, not a
//!   silent `false`.
//! - A field path that is absent from the config evaluates to JSON `null`
//!   for `==`/`!=` purposes (so `missing_field == null` is `true`, matching
//!   the intuitive "unset" reading), but is a [`ConditionError`] under
//!   `<`/`<=`/`>`/`>=` and inside `&&`/`||` operands that are not
//!   comparisons (see below).
//! - `&&`/`||` operate on `bool`. A bare field path or literal used
//!   directly as an operand of `&&`/`||` (rather than as one side of a
//!   comparison) must itself be a JSON boolean; anything else is a
//!   [`ConditionError`], since "is this field truthy" has no single
//!   universally correct definition across strings/numbers/objects the way
//!   it does in a dynamically-typed scripting language.
//!
//! # Errors
//!
//! Every parse or evaluation failure returns a [`ConditionError`] with the
//! offending condition text and a human-readable reason, rather than the
//! old behavior of silently treating anything it did not understand as
//! `false` -- a schema author who mistypes `existing_field <> "value"`
//! learns about it immediately instead of the conditional requirement
//! simply never firing.

use std::fmt;

/// A structured error produced while parsing or evaluating a condition
/// expression. Always carries the original condition text so the caller can
/// report exactly what was wrong with which rule.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionError {
    /// The full condition string that failed to parse or evaluate.
    pub condition: String,
    /// Human-readable description of what went wrong.
    pub reason: String,
}

impl fmt::Display for ConditionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid condition `{}`: {}", self.condition, self.reason)
    }
}

impl std::error::Error for ConditionError {}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    String(String),
    Number(f64),
    True,
    False,
    Eq,    // ==
    NotEq, // !=
    Lt,    // <
    LtEq,  // <=
    Gt,    // >
    GtEq,  // >=
    AndAnd,
    OrOr,
    LParen,
    RParen,
    Dot,
}

struct Tokenizer<'a> {
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    source: &'a str,
}

impl<'a> Tokenizer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            chars: source.char_indices().peekable(),
            source,
        }
    }

    fn error(&self, reason: impl Into<String>) -> String {
        format!("{} (in `{}`)", reason.into(), self.source)
    }

    fn tokenize(mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        while let Some(&(pos, ch)) = self.chars.peek() {
            match ch {
                c if c.is_whitespace() => {
                    self.chars.next();
                },
                '(' => {
                    self.chars.next();
                    tokens.push(Token::LParen);
                },
                ')' => {
                    self.chars.next();
                    tokens.push(Token::RParen);
                },
                '.' => {
                    self.chars.next();
                    tokens.push(Token::Dot);
                },
                '=' => {
                    self.chars.next();
                    if matches!(self.chars.peek(), Some(&(_, '='))) {
                        self.chars.next();
                        tokens.push(Token::Eq);
                    } else {
                        return Err(self.error(format!(
                            "unexpected `=` at position {pos}; did you mean `==`?"
                        )));
                    }
                },
                '!' => {
                    self.chars.next();
                    if matches!(self.chars.peek(), Some(&(_, '='))) {
                        self.chars.next();
                        tokens.push(Token::NotEq);
                    } else {
                        return Err(self.error(format!(
                            "unexpected `!` at position {pos}; did you mean `!=`?"
                        )));
                    }
                },
                '<' => {
                    self.chars.next();
                    if matches!(self.chars.peek(), Some(&(_, '='))) {
                        self.chars.next();
                        tokens.push(Token::LtEq);
                    } else {
                        tokens.push(Token::Lt);
                    }
                },
                '>' => {
                    self.chars.next();
                    if matches!(self.chars.peek(), Some(&(_, '='))) {
                        self.chars.next();
                        tokens.push(Token::GtEq);
                    } else {
                        tokens.push(Token::Gt);
                    }
                },
                '&' => {
                    self.chars.next();
                    if matches!(self.chars.peek(), Some(&(_, '&'))) {
                        self.chars.next();
                        tokens.push(Token::AndAnd);
                    } else {
                        return Err(self.error(format!(
                            "unexpected `&` at position {pos}; did you mean `&&`?"
                        )));
                    }
                },
                '|' => {
                    self.chars.next();
                    if matches!(self.chars.peek(), Some(&(_, '|'))) {
                        self.chars.next();
                        tokens.push(Token::OrOr);
                    } else {
                        return Err(self.error(format!(
                            "unexpected `|` at position {pos}; did you mean `||`?"
                        )));
                    }
                },
                '"' | '\'' => {
                    let quote = ch;
                    self.chars.next();
                    let mut value = String::new();
                    let mut closed = false;
                    for (_, c) in self.chars.by_ref() {
                        if c == quote {
                            closed = true;
                            break;
                        }
                        value.push(c);
                    }
                    if !closed {
                        return Err(self.error(format!(
                            "unterminated string literal starting at position {pos}"
                        )));
                    }
                    tokens.push(Token::String(value));
                },
                c if c.is_ascii_digit() || (c == '-' && self.looks_like_number_start()) => {
                    let mut number = String::new();
                    if c == '-' {
                        number.push(c);
                        self.chars.next();
                    }
                    while let Some(&(_, c)) = self.chars.peek() {
                        if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' {
                            number.push(c);
                            self.chars.next();
                        } else {
                            break;
                        }
                    }
                    let value = number
                        .parse::<f64>()
                        .map_err(|_| self.error(format!("`{number}` is not a valid number")))?;
                    tokens.push(Token::Number(value));
                },
                c if c.is_alphabetic() || c == '_' => {
                    let mut ident = String::new();
                    while let Some(&(_, c)) = self.chars.peek() {
                        if c.is_alphanumeric() || c == '_' || c == '-' {
                            ident.push(c);
                            self.chars.next();
                        } else {
                            break;
                        }
                    }
                    match ident.as_str() {
                        "true" => tokens.push(Token::True),
                        "false" => tokens.push(Token::False),
                        _ => tokens.push(Token::Ident(ident)),
                    }
                },
                other => {
                    return Err(
                        self.error(format!("unexpected character `{other}` at position {pos}"))
                    );
                },
            }
        }
        Ok(tokens)
    }

    /// A `-` starts a numeric literal only when immediately followed by a
    /// digit -- this keeps `field-name` (a hyphenated identifier, common in
    /// config field names) from being mis-tokenized as subtraction, which
    /// this grammar does not otherwise support.
    fn looks_like_number_start(&self) -> bool {
        let mut lookahead = self.chars.clone();
        lookahead.next(); // consume the '-' itself
        matches!(lookahead.peek(), Some((_, c)) if c.is_ascii_digit())
    }
}

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Literal {
    String(String),
    Number(f64),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
enum CompareOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

#[derive(Debug, Clone, PartialEq)]
enum Operand {
    Field(Vec<String>),
    Literal(Literal),
}

#[derive(Debug, Clone, PartialEq)]
enum Expr {
    /// A bare operand used directly as a boolean (must resolve to a JSON
    /// bool at evaluation time).
    Bare(Operand),
    Compare(Operand, CompareOp, Operand),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

// ---------------------------------------------------------------------------
// Parser (recursive descent)
// ---------------------------------------------------------------------------

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    source: &'a str,
}

impl<'a> Parser<'a> {
    fn new(tokens: Vec<Token>, source: &'a str) -> Self {
        Self {
            tokens,
            pos: 0,
            source,
        }
    }

    fn error(&self, reason: impl Into<String>) -> String {
        format!("{} (in `{}`)", reason.into(), self.source)
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        tok
    }

    fn parse(mut self) -> Result<Expr, String> {
        let expr = self.parse_or()?;
        if self.pos != self.tokens.len() {
            return Err(self.error(format!(
                "unexpected trailing tokens after a complete expression (starting at token {})",
                self.pos
            )));
        }
        Ok(expr)
    }

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Some(Token::OrOr)) {
            self.advance();
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_comparison()?;
        while matches!(self.peek(), Some(Token::AndAnd)) {
            self.advance();
            let right = self.parse_comparison()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr, String> {
        // A parenthesized sub-expression is a full boolean expression, not
        // just an operand, so it is handled before falling through to
        // "operand [comp_op operand]".
        if matches!(self.peek(), Some(Token::LParen)) {
            self.advance();
            let inner = self.parse_or()?;
            match self.advance() {
                Some(Token::RParen) => return Ok(inner),
                _ => {
                    return Err(self.error("expected closing `)`"));
                },
            }
        }

        let left = self.parse_operand()?;
        let op = match self.peek() {
            Some(Token::Eq) => Some(CompareOp::Eq),
            Some(Token::NotEq) => Some(CompareOp::NotEq),
            Some(Token::Lt) => Some(CompareOp::Lt),
            Some(Token::LtEq) => Some(CompareOp::LtEq),
            Some(Token::Gt) => Some(CompareOp::Gt),
            Some(Token::GtEq) => Some(CompareOp::GtEq),
            _ => None,
        };
        match op {
            Some(op) => {
                self.advance();
                let right = self.parse_operand()?;
                Ok(Expr::Compare(left, op, right))
            },
            None => Ok(Expr::Bare(left)),
        }
    }

    fn parse_operand(&mut self) -> Result<Operand, String> {
        match self.advance() {
            Some(Token::String(s)) => Ok(Operand::Literal(Literal::String(s))),
            Some(Token::Number(n)) => Ok(Operand::Literal(Literal::Number(n))),
            Some(Token::True) => Ok(Operand::Literal(Literal::Bool(true))),
            Some(Token::False) => Ok(Operand::Literal(Literal::Bool(false))),
            Some(Token::Ident(first)) => {
                let mut path = vec![first];
                while matches!(self.peek(), Some(Token::Dot)) {
                    self.advance();
                    match self.advance() {
                        Some(Token::Ident(part)) => path.push(part),
                        _ => {
                            return Err(
                                self.error("expected a field name after `.` in a field path")
                            );
                        },
                    }
                }
                Ok(Operand::Field(path))
            },
            Some(other) => Err(self.error(format!(
                "expected a field name, string, number, or `true`/`false`, found {other:?}"
            ))),
            None => Err(self.error("expected an operand but the expression ended")),
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Parse `condition` and evaluate it against `config`.
///
/// # Errors
///
/// Returns a [`ConditionError`] for any syntax the grammar documented on
/// this module does not recognize, or for a runtime type mismatch (e.g.
/// ordering a string, or using a non-boolean field as a bare `&&`/`||`
/// operand) -- never silently returns `false` for input it could not make
/// sense of.
pub fn evaluate(
    condition: &str,
    config: &serde_json::Map<String, serde_json::Value>,
) -> Result<bool, ConditionError> {
    let tokens = Tokenizer::new(condition).tokenize().map_err(|reason| ConditionError {
        condition: condition.to_string(),
        reason,
    })?;
    if tokens.is_empty() {
        return Err(ConditionError {
            condition: condition.to_string(),
            reason: "condition is empty".to_string(),
        });
    }
    let ast = Parser::new(tokens, condition).parse().map_err(|reason| ConditionError {
        condition: condition.to_string(),
        reason,
    })?;
    eval_expr(&ast, config).map_err(|reason| ConditionError {
        condition: condition.to_string(),
        reason,
    })
}

fn resolve_field<'a>(
    path: &[String],
    config: &'a serde_json::Map<String, serde_json::Value>,
) -> Option<&'a serde_json::Value> {
    let mut parts = path.iter();
    let first = parts.next()?;
    let mut current = config.get(first)?;
    for part in parts {
        current = current.as_object()?.get(part)?;
    }
    Some(current)
}

fn resolve_operand<'a>(
    operand: &'a Operand,
    config: &'a serde_json::Map<String, serde_json::Value>,
) -> ResolvedValue<'a> {
    match operand {
        Operand::Literal(Literal::String(s)) => {
            ResolvedValue::Owned(serde_json::Value::String(s.clone()))
        },
        Operand::Literal(Literal::Number(n)) => ResolvedValue::Owned(
            serde_json::Number::from_f64(*n)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
        ),
        Operand::Literal(Literal::Bool(b)) => ResolvedValue::Owned(serde_json::Value::Bool(*b)),
        Operand::Field(path) => match resolve_field(path, config) {
            Some(value) => ResolvedValue::Borrowed(value),
            // An absent field reads as JSON null, which lets `field == null`
            // express "field is unset" -- see the module doc comment.
            None => ResolvedValue::Owned(serde_json::Value::Null),
        },
    }
}

/// Either a reference into the config (for a resolved field path) or an
/// owned value (for a literal, or a field path that resolved to "absent").
/// Avoids cloning every field lookup while still letting literals produce a
/// value without borrowing from anything.
enum ResolvedValue<'a> {
    Borrowed(&'a serde_json::Value),
    Owned(serde_json::Value),
}

impl ResolvedValue<'_> {
    fn as_value(&self) -> &serde_json::Value {
        match self {
            ResolvedValue::Borrowed(v) => v,
            ResolvedValue::Owned(v) => v,
        }
    }

    fn describe(operand: &Operand) -> String {
        match operand {
            Operand::Field(path) => format!("field `{}`", path.join(".")),
            Operand::Literal(Literal::String(s)) => format!("string literal \"{s}\""),
            Operand::Literal(Literal::Number(n)) => format!("number literal {n}"),
            Operand::Literal(Literal::Bool(b)) => format!("boolean literal {b}"),
        }
    }
}

fn eval_expr(
    expr: &Expr,
    config: &serde_json::Map<String, serde_json::Value>,
) -> Result<bool, String> {
    match expr {
        Expr::Bare(operand) => {
            let resolved = resolve_operand(operand, config);
            resolved.as_value().as_bool().ok_or_else(|| {
                format!(
                    "{} is used directly as a boolean but is not `true`/`false` (got {})",
                    ResolvedValue::describe(operand),
                    resolved.as_value()
                )
            })
        },
        Expr::Compare(left, op, right) => eval_compare(left, op, right, config),
        Expr::And(left, right) => Ok(eval_expr(left, config)? && eval_expr(right, config)?),
        Expr::Or(left, right) => Ok(eval_expr(left, config)? || eval_expr(right, config)?),
    }
}

fn eval_compare(
    left: &Operand,
    op: &CompareOp,
    right: &Operand,
    config: &serde_json::Map<String, serde_json::Value>,
) -> Result<bool, String> {
    let left_value = resolve_operand(left, config);
    let right_value = resolve_operand(right, config);
    let (l, r) = (left_value.as_value(), right_value.as_value());

    match op {
        CompareOp::Eq => Ok(json_values_equal(l, r)),
        CompareOp::NotEq => Ok(!json_values_equal(l, r)),
        CompareOp::Lt | CompareOp::LtEq | CompareOp::Gt | CompareOp::GtEq => {
            let ln = l.as_f64().ok_or_else(|| {
                format!(
                    "{} is not a number, so it cannot be used with a numeric ordering \
                     operator (got {l})",
                    ResolvedValue::describe(left)
                )
            })?;
            let rn = r.as_f64().ok_or_else(|| {
                format!(
                    "{} is not a number, so it cannot be used with a numeric ordering \
                     operator (got {r})",
                    ResolvedValue::describe(right)
                )
            })?;
            Ok(match op {
                CompareOp::Lt => ln < rn,
                CompareOp::LtEq => ln <= rn,
                CompareOp::Gt => ln > rn,
                CompareOp::GtEq => ln >= rn,
                CompareOp::Eq | CompareOp::NotEq => unreachable!("handled above"),
            })
        },
    }
}

/// Type-appropriate equality: values must share a JSON type to be equal
/// (numbers compare as `f64`; strings and bools compare directly; `null`
/// equals only `null`). A type mismatch is `false`, not an error -- see the
/// module doc comment for why this differs from the ordering operators.
fn json_values_equal(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::Null, serde_json::Value::Null) => true,
        (serde_json::Value::Bool(a), serde_json::Value::Bool(b)) => a == b,
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => a.as_f64() == b.as_f64(),
        (serde_json::Value::String(a), serde_json::Value::String(b)) => a == b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config_from(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        match value {
            serde_json::Value::Object(map) => map,
            _ => panic!("test config must be a JSON object"),
        }
    }

    // -------------------------------------------------------------------
    // Basic equality (the one case the old substring-based evaluator
    // supported) must still work.
    // -------------------------------------------------------------------

    #[test]
    fn test_simple_string_equality_true() {
        let config = config_from(json!({ "model_type": "bert" }));
        assert_eq!(evaluate("model_type == \"bert\"", &config), Ok(true));
    }

    #[test]
    fn test_simple_string_equality_false() {
        let config = config_from(json!({ "model_type": "gpt2" }));
        assert_eq!(evaluate("model_type == \"bert\"", &config), Ok(false));
    }

    #[test]
    fn test_single_quotes_supported() {
        let config = config_from(json!({ "model_type": "bert" }));
        assert_eq!(evaluate("model_type == 'bert'", &config), Ok(true));
    }

    // -------------------------------------------------------------------
    // Operators the old evaluator did NOT support: !=, <, <=, >, >=
    // -------------------------------------------------------------------

    #[test]
    fn test_not_equal() {
        let config = config_from(json!({ "mode": "fast" }));
        assert_eq!(evaluate("mode != \"slow\"", &config), Ok(true));
        assert_eq!(evaluate("mode != \"fast\"", &config), Ok(false));
    }

    #[test]
    fn test_less_than() {
        let config = config_from(json!({ "batch_size": 8 }));
        assert_eq!(evaluate("batch_size < 16", &config), Ok(true));
        assert_eq!(evaluate("batch_size < 4", &config), Ok(false));
    }

    #[test]
    fn test_less_than_or_equal() {
        let config = config_from(json!({ "batch_size": 16 }));
        assert_eq!(evaluate("batch_size <= 16", &config), Ok(true));
        assert_eq!(evaluate("batch_size <= 15", &config), Ok(false));
    }

    #[test]
    fn test_greater_than() {
        let config = config_from(json!({ "learning_rate": 0.01 }));
        assert_eq!(evaluate("learning_rate > 0.001", &config), Ok(true));
        assert_eq!(evaluate("learning_rate > 1.0", &config), Ok(false));
    }

    #[test]
    fn test_greater_than_or_equal() {
        let config = config_from(json!({ "num_epochs": 10 }));
        assert_eq!(evaluate("num_epochs >= 10", &config), Ok(true));
        assert_eq!(evaluate("num_epochs >= 11", &config), Ok(false));
    }

    #[test]
    fn test_negative_number_literal() {
        let config = config_from(json!({ "offset": -5 }));
        assert_eq!(evaluate("offset == -5", &config), Ok(true));
        assert_eq!(evaluate("offset > -10", &config), Ok(true));
    }

    #[test]
    fn test_boolean_literal_comparison() {
        let config = config_from(json!({ "enabled": true }));
        assert_eq!(evaluate("enabled == true", &config), Ok(true));
        assert_eq!(evaluate("enabled == false", &config), Ok(false));
    }

    #[test]
    fn test_bare_boolean_field() {
        let config = config_from(json!({ "use_auth_token": true }));
        assert_eq!(evaluate("use_auth_token", &config), Ok(true));
        let config2 = config_from(json!({ "use_auth_token": false }));
        assert_eq!(evaluate("use_auth_token", &config2), Ok(false));
    }

    // -------------------------------------------------------------------
    // Logical operators: &&, ||, and precedence (&& binds tighter than ||)
    // -------------------------------------------------------------------

    #[test]
    fn test_and_both_true() {
        let config = config_from(json!({ "a": 1, "b": 2 }));
        assert_eq!(evaluate("a == 1 && b == 2", &config), Ok(true));
    }

    #[test]
    fn test_and_one_false() {
        let config = config_from(json!({ "a": 1, "b": 3 }));
        assert_eq!(evaluate("a == 1 && b == 2", &config), Ok(false));
    }

    #[test]
    fn test_or_one_true() {
        let config = config_from(json!({ "a": 1, "b": 3 }));
        assert_eq!(evaluate("a == 99 || b == 3", &config), Ok(true));
    }

    #[test]
    fn test_or_both_false() {
        let config = config_from(json!({ "a": 1, "b": 3 }));
        assert_eq!(evaluate("a == 99 || b == 100", &config), Ok(false));
    }

    #[test]
    fn test_precedence_and_binds_tighter_than_or() {
        // `a || (b && c)`, NOT `(a || b) && c`.
        // a=false, b=true, c=false:
        //   a || (b && c) = false || (true && false) = false || false = false
        //   (a || b) && c = (false || true) && false = true && false = false
        // Both give false here, so pick values where the two readings differ:
        // a=true, b=false, c=false:
        //   a || (b && c) = true || (false && false) = true || false = true
        //   (a || b) && c = (true || false) && false = true && false = false
        let config = config_from(json!({ "a": true, "b": false, "c": false }));
        assert_eq!(evaluate("a || b && c", &config), Ok(true));
    }

    #[test]
    fn test_parentheses_override_precedence() {
        // (a || b) && c with a=true, b=false, c=false must be false, proving
        // parens actually change the grouping vs. the un-parenthesized case
        // above (which evaluates the same inputs to true).
        let config = config_from(json!({ "a": true, "b": false, "c": false }));
        assert_eq!(evaluate("(a || b) && c", &config), Ok(false));
    }

    #[test]
    fn test_nested_parentheses() {
        let config = config_from(json!({ "a": 1, "b": 2, "c": 3, "d": 4 }));
        assert_eq!(
            evaluate("(a == 1 && b == 2) || (c == 99 && d == 4)", &config),
            Ok(true)
        );
    }

    #[test]
    fn test_multiple_and_chained() {
        let config = config_from(json!({ "a": 1, "b": 2, "c": 3 }));
        assert_eq!(evaluate("a == 1 && b == 2 && c == 3", &config), Ok(true));
        assert_eq!(evaluate("a == 1 && b == 2 && c == 99", &config), Ok(false));
    }

    // -------------------------------------------------------------------
    // Dotted field paths (nested objects)
    // -------------------------------------------------------------------

    #[test]
    fn test_nested_field_path() {
        let config = config_from(json!({ "model": { "type": "bert" } }));
        assert_eq!(evaluate("model.type == \"bert\"", &config), Ok(true));
    }

    #[test]
    fn test_deeply_nested_field_path() {
        let config = config_from(json!({ "a": { "b": { "c": 42 } } }));
        assert_eq!(evaluate("a.b.c == 42", &config), Ok(true));
    }

    // -------------------------------------------------------------------
    // Missing fields resolve to null for equality, error for ordering
    // -------------------------------------------------------------------

    #[test]
    fn test_missing_field_equals_null_literal_via_comparison_is_false_against_string() {
        let config = config_from(json!({ "present": "x" }));
        // missing field compares as JSON null, so equality against a string
        // literal is false (type mismatch), not an error.
        assert_eq!(evaluate("missing == \"x\"", &config), Ok(false));
    }

    #[test]
    fn test_missing_field_not_equal_is_true() {
        let config = config_from(json!({ "present": "x" }));
        assert_eq!(evaluate("missing != \"x\"", &config), Ok(true));
    }

    #[test]
    fn test_missing_field_ordering_is_error() {
        let config = config_from(json!({ "present": 1 }));
        let result = evaluate("missing > 5", &config);
        assert!(
            result.is_err(),
            "ordering against a missing field must error, not silently be false"
        );
    }

    // -------------------------------------------------------------------
    // Structured errors instead of silent `false`
    // -------------------------------------------------------------------

    #[test]
    fn test_unparseable_condition_is_error_not_false() {
        let config = config_from(json!({ "a": 1 }));
        // Old evaluator: no "==" substring -> silently `false`.
        // New evaluator: `<>` is not a real operator -> structured error.
        let result = evaluate("a <> 1", &config);
        assert!(
            result.is_err(),
            "an unparseable condition must be a structured error"
        );
    }

    #[test]
    fn test_empty_condition_is_error() {
        let config = config_from(json!({}));
        assert!(evaluate("", &config).is_err());
    }

    #[test]
    fn test_dangling_operator_is_error() {
        let config = config_from(json!({ "a": 1 }));
        assert!(evaluate("a ==", &config).is_err());
    }

    #[test]
    fn test_unterminated_string_is_error() {
        let config = config_from(json!({ "a": 1 }));
        assert!(evaluate("a == \"unterminated", &config).is_err());
    }

    #[test]
    fn test_unmatched_paren_is_error() {
        let config = config_from(json!({ "a": 1 }));
        assert!(evaluate("(a == 1", &config).is_err());
    }

    #[test]
    fn test_trailing_garbage_is_error() {
        let config = config_from(json!({ "a": 1 }));
        assert!(evaluate("a == 1 )", &config).is_err());
    }

    #[test]
    fn test_ordering_a_string_is_error() {
        let config = config_from(json!({ "name": "bert" }));
        let result = evaluate("name > \"aardvark\"", &config);
        assert!(
            result.is_err(),
            "ordering a string must be a structured error"
        );
    }

    #[test]
    fn test_bare_non_boolean_field_is_error() {
        let config = config_from(json!({ "count": 5 }));
        let result = evaluate("count", &config);
        assert!(
            result.is_err(),
            "a bare non-boolean field used as a whole condition must error"
        );
    }

    #[test]
    fn test_error_message_includes_condition_text() {
        let config = config_from(json!({}));
        let err = evaluate("a <> b", &config).expect_err("must fail");
        assert_eq!(err.condition, "a <> b");
        assert!(!err.reason.is_empty());
    }

    // -------------------------------------------------------------------
    // Whitespace / formatting tolerance
    // -------------------------------------------------------------------

    #[test]
    fn test_tolerates_extra_whitespace() {
        let config = config_from(json!({ "a": 1 }));
        assert_eq!(evaluate("   a   ==   1   ", &config), Ok(true));
    }

    #[test]
    fn test_tolerates_no_whitespace() {
        let config = config_from(json!({ "a": 1 }));
        assert_eq!(evaluate("a==1", &config), Ok(true));
    }

    #[test]
    fn test_hyphenated_field_name() {
        let config = config_from(json!({ "field-name": "value" }));
        assert_eq!(evaluate("field-name == \"value\"", &config), Ok(true));
    }
}
