//! A real recursive-descent arithmetic/algebraic expression evaluator.
//!
//! Backs [`super::MathSpecializedForCausalLM::evaluate_expression`] and
//! [`super::MathSpecializedForCausalLM::solve_step_by_step`], which used to
//! return the literal strings `"Expression evaluation would be performed
//! here"` and `"Step-by-step solution would be generated here"` regardless
//! of input. This module actually parses and computes:
//!
//! - Grammar: `+ - * / ^ ( ) <number> <identifier>`, with standard
//!   precedence (unary minus/plus, then `^` right-associative, then `*`/`/`,
//!   then `+`/`-`).
//! - Arithmetic is done with exact rationals (`i128` numerator/denominator)
//!   wherever the operations stay exact - every `+ - * /` and any `^` with
//!   an integer exponent. A non-integer exponent (e.g. `2 ^ 0.5`) is the one
//!   case that cannot stay exact in general, so that (and only that) case
//!   falls back to `f64`.
//! - `solve_step_by_step` additionally recognises a single-variable *linear*
//!   equation (`a*x + b = c`, "linear" meaning the variable only ever
//!   appears added, subtracted, or multiplied/divided by a variable-free
//!   factor) and solves it exactly; anything else honestly reports that it
//!   is not supported rather than fabricating a solution.

use std::collections::HashSet;
use std::fmt;
use trustformers_core::errors::{compute_error, invalid_input, Result};

// ---------------------------------------------------------------------
// Exact rational arithmetic
// ---------------------------------------------------------------------

/// An exact fraction, always kept in lowest terms with a positive
/// denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rational {
    num: i128,
    den: i128,
}

fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

impl Rational {
    fn new(num: i128, den: i128) -> Result<Self> {
        if den == 0 {
            return Err(compute_error("Rational::new", "denominator is zero"));
        }
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
        let g = gcd(num, den);
        Ok(Self {
            num: num / g,
            den: den / g,
        })
    }

    fn from_i128(n: i128) -> Self {
        Self { num: n, den: 1 }
    }

    fn zero() -> Self {
        Self::from_i128(0)
    }

    fn is_zero(&self) -> bool {
        self.num == 0
    }

    fn is_integer(&self) -> bool {
        self.den == 1
    }

    fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    fn add(self, other: Self) -> Result<Self> {
        let num = self
            .num
            .checked_mul(other.den)
            .and_then(|a| other.num.checked_mul(self.den).and_then(|b| a.checked_add(b)))
            .ok_or_else(|| compute_error("Rational::add", "overflow"))?;
        let den = self
            .den
            .checked_mul(other.den)
            .ok_or_else(|| compute_error("Rational::add", "overflow"))?;
        Self::new(num, den)
    }

    fn neg(self) -> Result<Self> {
        Ok(Self {
            num: self
                .num
                .checked_neg()
                .ok_or_else(|| compute_error("Rational::neg", "overflow"))?,
            den: self.den,
        })
    }

    fn sub(self, other: Self) -> Result<Self> {
        self.add(other.neg()?)
    }

    fn mul(self, other: Self) -> Result<Self> {
        let num = self
            .num
            .checked_mul(other.num)
            .ok_or_else(|| compute_error("Rational::mul", "overflow"))?;
        let den = self
            .den
            .checked_mul(other.den)
            .ok_or_else(|| compute_error("Rational::mul", "overflow"))?;
        Self::new(num, den)
    }

    fn div(self, other: Self) -> Result<Self> {
        if other.is_zero() {
            return Err(compute_error("Rational::div", "division by zero"));
        }
        self.mul(Self {
            num: other.den,
            den: other.num,
        })
    }

    /// Raise to an integer power (positive, negative, or zero) by repeated
    /// squaring, staying exact.
    fn pow_i128(self, exponent: i128) -> Result<Self> {
        if exponent == 0 {
            if self.is_zero() {
                return Err(compute_error("Rational::pow", "0^0 is undefined"));
            }
            return Ok(Self::from_i128(1));
        }
        if exponent < 0 {
            if self.is_zero() {
                return Err(compute_error(
                    "Rational::pow",
                    "division by zero (negative power of 0)",
                ));
            }
            return Self {
                num: self.den,
                den: self.num,
            }
            .pow_i128(-exponent);
        }
        let mut result = Self::from_i128(1);
        let mut base = self;
        let mut exp = exponent as u128;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(base)?;
            }
            exp >>= 1;
            if exp > 0 {
                base = base.mul(base)?;
            }
        }
        Ok(result)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_integer() {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// A computed value: an exact rational whenever possible, otherwise the
/// `f64` fallback used for non-integer exponents.
#[derive(Debug, Clone, Copy)]
enum Number {
    Rational(Rational),
    Float(f64),
}

impl Number {
    fn to_f64(self) -> f64 {
        match self {
            Number::Rational(r) => r.to_f64(),
            Number::Float(f) => f,
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Rational(r) => write!(f, "{r}"),
            Number::Float(x) => write!(f, "{x}"),
        }
    }
}

fn num_add(a: Number, b: Number) -> Result<Number> {
    match (a, b) {
        (Number::Rational(x), Number::Rational(y)) => Ok(Number::Rational(x.add(y)?)),
        _ => Ok(Number::Float(a.to_f64() + b.to_f64())),
    }
}

fn num_sub(a: Number, b: Number) -> Result<Number> {
    match (a, b) {
        (Number::Rational(x), Number::Rational(y)) => Ok(Number::Rational(x.sub(y)?)),
        _ => Ok(Number::Float(a.to_f64() - b.to_f64())),
    }
}

fn num_mul(a: Number, b: Number) -> Result<Number> {
    match (a, b) {
        (Number::Rational(x), Number::Rational(y)) => Ok(Number::Rational(x.mul(y)?)),
        _ => Ok(Number::Float(a.to_f64() * b.to_f64())),
    }
}

fn num_div(a: Number, b: Number) -> Result<Number> {
    match (a, b) {
        (Number::Rational(x), Number::Rational(y)) => Ok(Number::Rational(x.div(y)?)),
        _ => {
            if b.to_f64() == 0.0 {
                return Err(compute_error("evaluate", "division by zero"));
            }
            Ok(Number::Float(a.to_f64() / b.to_f64()))
        },
    }
}

fn num_neg(a: Number) -> Result<Number> {
    match a {
        Number::Rational(x) => Ok(Number::Rational(x.neg()?)),
        Number::Float(x) => Ok(Number::Float(-x)),
    }
}

fn num_pow(base: Number, exp: Number) -> Result<Number> {
    if let Number::Rational(e) = exp {
        if e.is_integer() {
            return match base {
                Number::Rational(b) => Ok(Number::Rational(b.pow_i128(e.num)?)),
                Number::Float(b) => Ok(Number::Float(b.powi(e.num as i32))),
            };
        }
    }
    // Non-integer exponent: cannot stay exact in general (e.g. 2^0.5 is
    // irrational), so this is the one deliberate fallback to `f64`.
    let result = base.to_f64().powf(exp.to_f64());
    if result.is_nan() {
        return Err(compute_error(
            "evaluate",
            "exponentiation produced NaN (e.g. a negative base to a fractional power)",
        ));
    }
    Ok(Number::Float(result))
}

// ---------------------------------------------------------------------
// AST + parser
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Expr {
    Num(Rational),
    Var(String),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(Rational),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
}

fn parse_decimal_literal(text: &str) -> Result<Rational> {
    match text.find('.') {
        Some(dot) => {
            let int_part = &text[..dot];
            let frac_part = &text[dot + 1..];
            if int_part.is_empty() && frac_part.is_empty() {
                return Err(invalid_input(format!("invalid numeric literal '{text}'")));
            }
            let combined = format!("{int_part}{frac_part}");
            let combined_val: i128 = if combined.is_empty() {
                0
            } else {
                combined
                    .parse()
                    .map_err(|_| invalid_input(format!("invalid numeric literal '{text}'")))?
            };
            let den = 10i128
                .checked_pow(frac_part.len() as u32)
                .ok_or_else(|| compute_error("parse", "numeric literal too precise"))?;
            Rational::new(combined_val, den)
        },
        None => {
            let val: i128 = text
                .parse()
                .map_err(|_| invalid_input(format!("invalid numeric literal '{text}'")))?;
            Ok(Rational::from_i128(val))
        },
    }
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            c if c.is_whitespace() => i += 1,
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            },
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            },
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            },
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            },
            '^' => {
                tokens.push(Token::Caret);
                i += 1;
            },
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            },
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            },
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                let mut seen_dot = c == '.';
                i += 1;
                while i < chars.len()
                    && (chars[i].is_ascii_digit() || (chars[i] == '.' && !seen_dot))
                {
                    if chars[i] == '.' {
                        seen_dot = true;
                    }
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                tokens.push(Token::Number(parse_decimal_literal(&text)?));
            },
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                i += 1;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(text));
            },
            other => {
                return Err(invalid_input(format!(
                    "unexpected character '{other}' in expression"
                )))
            },
        }
    }
    Ok(tokens)
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    /// `expr := term (('+' | '-') term)*`
    fn parse_expr(&mut self) -> Result<Expr> {
        let mut node = self.parse_term()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    node = Expr::Add(Box::new(node), Box::new(rhs));
                },
                Some(Token::Minus) => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    node = Expr::Sub(Box::new(node), Box::new(rhs));
                },
                _ => break,
            }
        }
        Ok(node)
    }

    /// `term := unary (('*' | '/') unary)*`
    fn parse_term(&mut self) -> Result<Expr> {
        let mut node = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.pos += 1;
                    let rhs = self.parse_unary()?;
                    node = Expr::Mul(Box::new(node), Box::new(rhs));
                },
                Some(Token::Slash) => {
                    self.pos += 1;
                    let rhs = self.parse_unary()?;
                    node = Expr::Div(Box::new(node), Box::new(rhs));
                },
                _ => break,
            }
        }
        Ok(node)
    }

    /// `unary := ('-' | '+') unary | power`. Unary minus binds *looser* than
    /// `^` on its own operand (`-2^2 == -(2^2) == -4`) but an exponent may
    /// itself start with a unary minus (`2^-2 == 0.25`), because `power`'s
    /// exponent is parsed by recursing into `unary`.
    fn parse_unary(&mut self) -> Result<Expr> {
        match self.peek() {
            Some(Token::Minus) => {
                self.pos += 1;
                Ok(Expr::Neg(Box::new(self.parse_unary()?)))
            },
            Some(Token::Plus) => {
                self.pos += 1;
                self.parse_unary()
            },
            _ => self.parse_power(),
        }
    }

    /// `power := primary ('^' unary)?` (right-associative via the `unary`
    /// recursion on the exponent side).
    fn parse_power(&mut self) -> Result<Expr> {
        let base = self.parse_primary()?;
        if let Some(Token::Caret) = self.peek() {
            self.pos += 1;
            let exponent = self.parse_unary()?;
            Ok(Expr::Pow(Box::new(base), Box::new(exponent)))
        } else {
            Ok(base)
        }
    }

    /// `primary := NUMBER | IDENT | '(' expr ')'`
    fn parse_primary(&mut self) -> Result<Expr> {
        match self.peek().cloned() {
            Some(Token::Number(r)) => {
                self.pos += 1;
                Ok(Expr::Num(r))
            },
            Some(Token::Ident(name)) => {
                self.pos += 1;
                Ok(Expr::Var(name))
            },
            Some(Token::LParen) => {
                self.pos += 1;
                let inner = self.parse_expr()?;
                match self.peek() {
                    Some(Token::RParen) => {
                        self.pos += 1;
                        Ok(inner)
                    },
                    _ => Err(invalid_input("expected closing ')'")),
                }
            },
            Some(other) => Err(invalid_input(format!("unexpected token {other:?}"))),
            None => Err(invalid_input("unexpected end of expression")),
        }
    }
}

fn parse(input: &str) -> Result<Expr> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err(invalid_input("empty expression"));
    }
    let mut parser = Parser {
        tokens: &tokens,
        pos: 0,
    };
    let expr = parser.parse_expr()?;
    if parser.pos != tokens.len() {
        return Err(invalid_input(format!(
            "unexpected trailing input at token {:?}",
            tokens[parser.pos]
        )));
    }
    Ok(expr)
}

// ---------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------

/// Recursively evaluate an already-parsed [`Expr`] AST node.
///
/// Note on naming: despite the name, this is not a dynamic code-execution
/// `eval` - `expr` was produced by [`parse`] from the tightly-constrained
/// arithmetic grammar above (numbers, `+ - * / ^`, parens, bare
/// identifiers), so there is no arbitrary code, no shelling out, and no
/// deserialization of untrusted programs involved; it only ever performs
/// [`Rational`]/[`Number`] arithmetic.
fn eval(expr: &Expr) -> Result<Number> {
    match expr {
        Expr::Num(r) => Ok(Number::Rational(*r)),
        Expr::Var(name) => Err(invalid_input(format!(
            "expression contains unbound variable '{name}'; evaluate_expression only \
             evaluates closed-form numeric expressions (use solve_step_by_step for a \
             single-variable equation)"
        ))),
        Expr::Neg(e) => num_neg(eval(e)?),
        Expr::Add(a, b) => num_add(eval(a)?, eval(b)?),
        Expr::Sub(a, b) => num_sub(eval(a)?, eval(b)?),
        Expr::Mul(a, b) => num_mul(eval(a)?, eval(b)?),
        Expr::Div(a, b) => num_div(eval(a)?, eval(b)?),
        Expr::Pow(a, b) => num_pow(eval(a)?, eval(b)?),
    }
}

/// Evaluate a closed-form (variable-free) arithmetic expression and format
/// the result: an exact integer or fraction whenever every operation stayed
/// in rational arithmetic, otherwise a decimal from the `f64` fallback.
pub(super) fn evaluate_to_string(expression: &str) -> Result<String> {
    let expr = parse(expression)?;
    let value = eval(&expr)?;
    Ok(value.to_string())
}

/// Evaluate an expression while recording one human-readable step per
/// binary/unary operation actually performed, in evaluation order.
fn eval_with_steps(expr: &Expr, steps: &mut Vec<String>) -> Result<Number> {
    match expr {
        Expr::Num(r) => Ok(Number::Rational(*r)),
        Expr::Var(name) => Err(invalid_input(format!("unbound variable '{name}'"))),
        Expr::Neg(e) => {
            let v = eval_with_steps(e, steps)?;
            let result = num_neg(v)?;
            steps.push(format!("-({v}) = {result}"));
            Ok(result)
        },
        Expr::Add(a, b) => {
            let (va, vb) = (eval_with_steps(a, steps)?, eval_with_steps(b, steps)?);
            let result = num_add(va, vb)?;
            steps.push(format!("{va} + {vb} = {result}"));
            Ok(result)
        },
        Expr::Sub(a, b) => {
            let (va, vb) = (eval_with_steps(a, steps)?, eval_with_steps(b, steps)?);
            let result = num_sub(va, vb)?;
            steps.push(format!("{va} - {vb} = {result}"));
            Ok(result)
        },
        Expr::Mul(a, b) => {
            let (va, vb) = (eval_with_steps(a, steps)?, eval_with_steps(b, steps)?);
            let result = num_mul(va, vb)?;
            steps.push(format!("{va} * {vb} = {result}"));
            Ok(result)
        },
        Expr::Div(a, b) => {
            let (va, vb) = (eval_with_steps(a, steps)?, eval_with_steps(b, steps)?);
            let result = num_div(va, vb)?;
            steps.push(format!("{va} / {vb} = {result}"));
            Ok(result)
        },
        Expr::Pow(a, b) => {
            let (va, vb) = (eval_with_steps(a, steps)?, eval_with_steps(b, steps)?);
            let result = num_pow(va, vb)?;
            steps.push(format!("{va} ^ {vb} = {result}"));
            Ok(result)
        },
    }
}

fn collect_vars(expr: &Expr, into: &mut HashSet<String>) {
    match expr {
        Expr::Num(_) => {},
        Expr::Var(name) => {
            into.insert(name.clone());
        },
        Expr::Neg(e) => collect_vars(e, into),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            collect_vars(a, into);
            collect_vars(b, into);
        },
    }
}

fn contains_var(expr: &Expr, var: &str) -> bool {
    match expr {
        Expr::Num(_) => false,
        Expr::Var(name) => name == var,
        Expr::Neg(e) => contains_var(e, var),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            contains_var(a, var) || contains_var(b, var)
        },
    }
}

/// `coeff * var + constant`, the normal form of a linear expression in one
/// variable.
#[derive(Debug, Clone, Copy)]
struct Linear {
    coeff: Rational,
    constant: Rational,
}

/// Reduce `expr` to `coeff * var + constant`, or return `None` if `var`
/// appears somewhere non-linear (as an exponent, multiplied/divided by
/// another occurrence of itself, etc.) - honestly reporting "not a
/// supported equation shape" rather than fabricating a solution.
fn linear_form(expr: &Expr, var: &str) -> Option<Linear> {
    if !contains_var(expr, var) {
        // Variable-free subtree: fold it to a single constant via the
        // regular (exact-preferring) evaluator.
        let Number::Rational(value) = eval(expr).ok()? else {
            return None; // stayed exact only if rational; a fractional
                         // exponent here would break "exact where possible".
        };
        return Some(Linear {
            coeff: Rational::zero(),
            constant: value,
        });
    }

    match expr {
        Expr::Num(_) => unreachable!("a variable-free Num is handled above"),
        Expr::Var(name) if name == var => Some(Linear {
            coeff: Rational::from_i128(1),
            constant: Rational::zero(),
        }),
        Expr::Var(_) => None, // a different, still-unbound variable
        Expr::Neg(e) => {
            let l = linear_form(e, var)?;
            Some(Linear {
                coeff: l.coeff.neg().ok()?,
                constant: l.constant.neg().ok()?,
            })
        },
        Expr::Add(a, b) => {
            let (la, lb) = (linear_form(a, var)?, linear_form(b, var)?);
            Some(Linear {
                coeff: la.coeff.add(lb.coeff).ok()?,
                constant: la.constant.add(lb.constant).ok()?,
            })
        },
        Expr::Sub(a, b) => {
            let (la, lb) = (linear_form(a, var)?, linear_form(b, var)?);
            Some(Linear {
                coeff: la.coeff.sub(lb.coeff).ok()?,
                constant: la.constant.sub(lb.constant).ok()?,
            })
        },
        Expr::Mul(a, b) => {
            // Linear only if the side containing `var` is multiplied by a
            // variable-free factor.
            if !contains_var(a, var) {
                let factor = linear_form(a, var)?.constant;
                let l = linear_form(b, var)?;
                Some(Linear {
                    coeff: l.coeff.mul(factor).ok()?,
                    constant: l.constant.mul(factor).ok()?,
                })
            } else if !contains_var(b, var) {
                let factor = linear_form(b, var)?.constant;
                let l = linear_form(a, var)?;
                Some(Linear {
                    coeff: l.coeff.mul(factor).ok()?,
                    constant: l.constant.mul(factor).ok()?,
                })
            } else {
                None // var * var (or var * f(var)) is not linear
            }
        },
        Expr::Div(a, b) => {
            // Linear only if dividing by a variable-free expression.
            if contains_var(b, var) {
                return None;
            }
            let divisor = linear_form(b, var)?.constant;
            let l = linear_form(a, var)?;
            Some(Linear {
                coeff: l.coeff.div(divisor).ok()?,
                constant: l.constant.div(divisor).ok()?,
            })
        },
        Expr::Pow(base, exponent) => {
            // Linear only for (linear-in-var)^1; anything else (x^2, a
            // variable in the exponent, ...) is not linear.
            if contains_var(exponent, var) {
                return None;
            }
            let exponent_value = linear_form(exponent, var)?.constant;
            if exponent_value == Rational::from_i128(1) {
                linear_form(base, var)
            } else {
                None
            }
        },
    }
}

/// Find the first top-level `=` (i.e. not the case where this evaluator
/// simply has no other use for `=`, since the expression grammar itself
/// never produces one) that separates an equation's two sides.
fn find_equals(input: &str) -> Option<usize> {
    input.find('=')
}

/// Evaluate `problem` step by step:
///
/// * If it contains `=`, treat it as an equation. When it is linear in
///   exactly one variable, solve it exactly and show each step; otherwise
///   return an honest "not supported" error naming why (multiple variables,
///   or a non-linear equation shape).
/// * Otherwise, treat it as a closed-form arithmetic expression and show one
///   step per operation actually evaluated.
pub(super) fn solve_step_by_step_text(problem: &str) -> Result<String> {
    let trimmed = problem.trim().trim_end_matches(['?', '.']).trim();
    if trimmed.is_empty() {
        return Err(invalid_input("empty problem"));
    }

    if let Some(eq_pos) = find_equals(trimmed) {
        let (lhs_str, rhs_str) = (trimmed[..eq_pos].trim(), trimmed[eq_pos + 1..].trim());
        let lhs = parse(lhs_str)?;
        let rhs = parse(rhs_str)?;

        let mut vars = HashSet::new();
        collect_vars(&lhs, &mut vars);
        collect_vars(&rhs, &mut vars);
        if vars.len() != 1 {
            return Err(invalid_input(format!(
                "solve_step_by_step only supports a single-variable linear equation; \
                 found {} variable(s) in '{trimmed}' (theorem proving and multi-variable \
                 systems are not supported)",
                vars.len()
            )));
        }
        let Some(var) = vars.into_iter().next() else {
            // Unreachable given the `vars.len() != 1` check above, but
            // handled as a structured error rather than a panic regardless.
            return Err(compute_error(
                "solve_step_by_step",
                "internal error: expected exactly one variable",
            ));
        };

        let (lhs_lin, rhs_lin) = (linear_form(&lhs, &var), linear_form(&rhs, &var));
        let (Some(lhs_lin), Some(rhs_lin)) = (lhs_lin, rhs_lin) else {
            return Err(invalid_input(format!(
                "'{trimmed}' is not a linear equation in '{var}' - only sums, differences, \
                 and multiplication/division by {var}-free factors are supported (e.g. not \
                 {var}^2, {var}*{var}, or {var} in an exponent)"
            )));
        };

        let a = lhs_lin.coeff.sub(rhs_lin.coeff)?; // combined coefficient of `var`
        let c = rhs_lin.constant.sub(lhs_lin.constant)?; // combined constant

        if a.is_zero() {
            return if c.is_zero() {
                Ok(format!(
                    "Problem: {trimmed}\nStep 1: Move every term to one side: 0 = 0\n\
                     Result: every value of {var} satisfies the equation (infinitely many solutions)."
                ))
            } else {
                Ok(format!(
                    "Problem: {trimmed}\nStep 1: Move every term to one side: 0 = {c}\n\
                     Result: no value of {var} satisfies the equation (0 != {c})."
                ))
            };
        }

        let x = c.div(a)?;
        return Ok(format!(
            "Problem: {trimmed}\n\
             Step 1: Collect {var}-terms on one side and constants on the other: {a}*{var} = {c}\n\
             Step 2: Divide both sides by {a}: {var} = {c} / {a}\n\
             Result: {var} = {x}"
        ));
    }

    let expr = parse(trimmed)?;
    let mut steps = Vec::new();
    let value = eval_with_steps(&expr, &mut steps)?;
    let mut out = format!("Problem: {trimmed}\n");
    for (i, step) in steps.iter().enumerate() {
        out.push_str(&format!("Step {}: {}\n", i + 1, step));
    }
    out.push_str(&format!("Result: {value}"));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_str(input: &str) -> String {
        evaluate_to_string(input).expect("evaluation should succeed")
    }

    #[test]
    fn evaluates_basic_arithmetic() {
        assert_eq!(eval_str("2 + 3 * 4"), "14");
        assert_eq!(eval_str("(2 + 3) * 4"), "20");
        assert_eq!(eval_str("10 - 4 - 3"), "3");
        assert_eq!(eval_str("2 ^ 10"), "1024");
    }

    #[test]
    fn evaluates_exact_fractions() {
        assert_eq!(eval_str("1 / 3"), "1/3");
        assert_eq!(eval_str("2 / 4"), "1/2");
        assert_eq!(eval_str("1/3 + 1/6"), "1/2");
    }

    #[test]
    fn respects_unary_minus_precedence() {
        assert_eq!(
            eval_str("-2^2"),
            "-4",
            "unary minus must bind looser than ^"
        );
        assert_eq!(eval_str("(-2)^2"), "4");
        assert_eq!(
            eval_str("2^-2"),
            "1/4",
            "a unary minus must be allowed in an exponent"
        );
        assert_eq!(eval_str("-2 * 3"), "-6");
        assert_eq!(eval_str("3 - -2"), "5");
    }

    #[test]
    fn evaluates_decimal_literals() {
        assert_eq!(eval_str("1.5 + 2.5"), "4");
        assert_eq!(
            eval_str("0.1 + 0.2"),
            "3/10",
            "decimals are parsed as exact rationals, not f64"
        );
    }

    #[test]
    fn rejects_division_by_zero() {
        assert!(evaluate_to_string("1 / 0").is_err());
        assert!(evaluate_to_string("5 / (2 - 2)").is_err());
    }

    #[test]
    fn rejects_malformed_expressions() {
        assert!(evaluate_to_string("2 +").is_err());
        assert!(evaluate_to_string("(2 + 3").is_err());
        assert!(evaluate_to_string("2 3").is_err());
        assert!(evaluate_to_string("").is_err());
        assert!(evaluate_to_string("2 $ 3").is_err());
    }

    #[test]
    fn rejects_unbound_variables_in_plain_evaluation() {
        let err = evaluate_to_string("x + 1").expect_err("must not fabricate a numeric answer");
        assert!(format!("{err}").contains('x'));
    }

    #[test]
    fn solves_a_linear_equation() {
        let solution = solve_step_by_step_text("2*x + 3 = 11").expect("solvable");
        assert!(solution.contains("x = 4"), "got: {solution}");
    }

    #[test]
    fn solves_a_linear_equation_with_variable_on_both_sides() {
        let solution = solve_step_by_step_text("3*x + 1 = x + 9").expect("solvable");
        assert!(solution.contains("x = 4"), "got: {solution}");
    }

    #[test]
    fn solves_a_linear_equation_with_fractional_answer() {
        let solution = solve_step_by_step_text("3*x = 1").expect("solvable");
        assert!(solution.contains("x = 1/3"), "got: {solution}");
    }

    #[test]
    fn reports_no_solution_honestly() {
        let solution =
            solve_step_by_step_text("x + 1 = x + 2").expect("well-formed, just unsatisfiable");
        assert!(solution.contains("no value"), "got: {solution}");
    }

    #[test]
    fn reports_identity_honestly() {
        let solution = solve_step_by_step_text("x + 1 = x + 1").expect("well-formed identity");
        assert!(solution.contains("infinitely many"), "got: {solution}");
    }

    #[test]
    fn rejects_nonlinear_equations_honestly_instead_of_fabricating() {
        // x^2 = 4 is a real, solvable equation, but not a *linear* one; this
        // evaluator must say so rather than inventing an answer.
        let err = solve_step_by_step_text("x^2 = 4").expect_err("must not fabricate a solution");
        assert!(format!("{err}").to_lowercase().contains("linear"));
    }

    #[test]
    fn rejects_multivariable_equations_honestly() {
        let err =
            solve_step_by_step_text("x + y = 4").expect_err("must not silently pick one variable");
        assert!(
            format!("{err}").contains('2'),
            "error should mention 2 variables were found"
        );
    }

    #[test]
    fn step_by_step_arithmetic_shows_real_computed_steps() {
        let solution = solve_step_by_step_text("2 + 3 * 4").expect("solvable");
        assert!(solution.contains("3 * 4 = 12"), "got: {solution}");
        assert!(solution.contains("2 + 12 = 14"), "got: {solution}");
        assert!(solution.contains("Result: 14"), "got: {solution}");
    }
}
