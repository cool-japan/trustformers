//! BNF grammar parsing and Earley recognition for grammar-constrained
//! generation.
//!
//! The grammar syntax is the usual `name ::= alternative | alternative` BNF
//! form.  Each alternative is a whitespace-separated sequence of symbols:
//!
//! * `'literal'` or `"literal"` — a terminal, matched character by character;
//! * anything else — a reference to another rule (a non-terminal).
//!
//! Lines that are blank or start with `#` are ignored.  The left-hand side of
//! the first rule is the start symbol unless a rule called `start` or `root`
//! exists.
//!
//! Recognition uses an Earley parser, which handles the full class of
//! context-free grammars (left recursion, ambiguity and nullable rules
//! included).  Two questions are answered:
//!
//! * [`ParseOutcome::viable`] — is the text a **prefix** of some string in the
//!   language?  This is what constrained decoding needs mid-generation.
//! * [`ParseOutcome::complete`] — is the text itself in the language?

use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failures encountered while parsing a BNF grammar definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarError {
    /// A quoted terminal was never closed.
    UnterminatedTerminal { line: usize, text: String },
    /// A rule referenced a non-terminal that is never defined.
    UndefinedNonTerminal { name: String },
    /// A rule had an empty left-hand side.
    MissingRuleName { line: usize },
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GrammarError::UnterminatedTerminal { line, text } => {
                write!(f, "unterminated quoted terminal on line {line}: {text}")
            },
            GrammarError::UndefinedNonTerminal { name } => {
                write!(f, "grammar references undefined rule `{name}`")
            },
            GrammarError::MissingRuleName { line } => {
                write!(f, "rule on line {line} has no name before `::=`")
            },
        }
    }
}

impl std::error::Error for GrammarError {}

// ---------------------------------------------------------------------------
// Symbols and productions
// ---------------------------------------------------------------------------

/// One element of a grammar alternative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarSymbol {
    /// A literal string that must appear verbatim in the input.
    Terminal(String),
    /// A reference to another rule.
    NonTerminal(String),
}

#[derive(Debug, Clone)]
struct Production {
    lhs: String,
    symbols: Vec<GrammarSymbol>,
}

// ---------------------------------------------------------------------------
// Grammar
// ---------------------------------------------------------------------------

/// A parsed BNF grammar with an Earley recogniser.
#[derive(Debug, Clone)]
pub struct Grammar {
    productions: Vec<Production>,
    by_name: HashMap<String, Vec<usize>>,
    start: String,
}

/// What the recogniser learned about one input string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseOutcome {
    /// The input is a prefix of at least one string in the language.
    pub viable: bool,
    /// The input itself is in the language.
    pub complete: bool,
    /// Terminal continuations that could legally follow the input.
    pub next_terminals: Vec<String>,
}

/// One Earley item: `production`, dot position, and the input offset the item
/// was created at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Item {
    production: usize,
    dot: usize,
    origin: usize,
}

impl Grammar {
    /// Parse a BNF grammar definition.
    ///
    /// Returns `Ok(None)` when the text defines no rules at all, which means
    /// "no grammar constraint" rather than "a grammar that rejects
    /// everything".
    pub fn parse(text: &str) -> Result<Option<Self>, GrammarError> {
        let mut productions: Vec<Production> = Vec::new();
        let mut by_name: HashMap<String, Vec<usize>> = HashMap::new();
        let mut order: Vec<String> = Vec::new();

        for (line_number, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((lhs, rhs)) = line.split_once("::=") else {
                continue;
            };
            let name = lhs.trim().to_string();
            if name.is_empty() {
                return Err(GrammarError::MissingRuleName {
                    line: line_number + 1,
                });
            }
            if !by_name.contains_key(&name) {
                order.push(name.clone());
            }

            for alternative in split_alternatives(rhs, line_number + 1)? {
                let symbols = parse_symbols(&alternative, line_number + 1)?;
                let index = productions.len();
                productions.push(Production {
                    lhs: name.clone(),
                    symbols,
                });
                by_name.entry(name.clone()).or_default().push(index);
            }
        }

        if productions.is_empty() {
            return Ok(None);
        }

        for production in &productions {
            for symbol in &production.symbols {
                if let GrammarSymbol::NonTerminal(name) = symbol {
                    if !by_name.contains_key(name) {
                        return Err(GrammarError::UndefinedNonTerminal { name: name.clone() });
                    }
                }
            }
        }

        let start = if by_name.contains_key("start") {
            "start".to_string()
        } else if by_name.contains_key("root") {
            "root".to_string()
        } else {
            order.first().cloned().unwrap_or_else(|| "start".to_string())
        };

        Ok(Some(Self {
            productions,
            by_name,
            start,
        }))
    }

    /// The start symbol used by the recogniser.
    pub fn start_symbol(&self) -> &str {
        &self.start
    }

    /// Number of productions (alternatives), not rules.
    pub fn production_count(&self) -> usize {
        self.productions.len()
    }

    /// The alternatives declared for `name`, if any.
    pub fn alternatives(&self, name: &str) -> Vec<Vec<GrammarSymbol>> {
        self.by_name
            .get(name)
            .map(|indices| {
                indices.iter().map(|&index| self.productions[index].symbols.clone()).collect()
            })
            .unwrap_or_default()
    }

    /// Run the Earley recogniser over `input`.
    pub fn recognize(&self, input: &str) -> ParseOutcome {
        let chars: Vec<char> = input.chars().collect();
        let n = chars.len();

        let mut charts: Vec<Vec<Item>> = vec![Vec::new(); n + 1];
        let mut seen: Vec<HashSet<Item>> = vec![HashSet::new(); n + 1];

        if let Some(indices) = self.by_name.get(&self.start) {
            for &production in indices {
                push_item(
                    &mut charts,
                    &mut seen,
                    0,
                    Item {
                        production,
                        dot: 0,
                        origin: 0,
                    },
                );
            }
        }

        let mut partial_alive = false;
        let mut next_terminals: Vec<String> = Vec::new();

        for position in 0..=n {
            // Predict / complete / epsilon-scan until nothing new appears.
            loop {
                let before = charts[position].len();
                let mut additions: Vec<Item> = Vec::new();

                for item in &charts[position] {
                    match self.next_symbol(item) {
                        None => {
                            // Completion: advance every item waiting on this rule.
                            let lhs = self.productions[item.production].lhs.as_str();
                            for candidate in &charts[item.origin] {
                                let waiting_on_lhs = matches!(
                                    self.next_symbol(candidate),
                                    Some(GrammarSymbol::NonTerminal(name)) if name == lhs
                                );
                                if waiting_on_lhs {
                                    additions.push(Item {
                                        dot: candidate.dot + 1,
                                        ..*candidate
                                    });
                                }
                            }
                        },
                        Some(GrammarSymbol::NonTerminal(name)) => {
                            if let Some(indices) = self.by_name.get(name) {
                                for &production in indices {
                                    additions.push(Item {
                                        production,
                                        dot: 0,
                                        origin: position,
                                    });
                                }
                            }
                        },
                        Some(GrammarSymbol::Terminal(text)) if text.is_empty() => {
                            additions.push(Item {
                                dot: item.dot + 1,
                                ..*item
                            });
                        },
                        Some(GrammarSymbol::Terminal(_)) => {},
                    }
                }

                for item in additions {
                    push_item(&mut charts, &mut seen, position, item);
                }
                if charts[position].len() == before {
                    break;
                }
            }

            // Scan: consume terminals starting at this position.
            let mut scanned: Vec<(usize, Item)> = Vec::new();
            for item in &charts[position] {
                let Some(GrammarSymbol::Terminal(text)) = self.next_symbol(item) else {
                    continue;
                };
                if text.is_empty() {
                    continue;
                }
                let terminal: Vec<char> = text.chars().collect();
                let available = n - position;

                if terminal.len() <= available {
                    if chars[position..position + terminal.len()] == terminal[..] {
                        scanned.push((
                            position + terminal.len(),
                            Item {
                                dot: item.dot + 1,
                                ..*item
                            },
                        ));
                    }
                } else if terminal[..available] == chars[position..n] {
                    // The input ends inside this terminal: still a viable prefix.
                    partial_alive = true;
                    let suffix: String = terminal[available..].iter().collect();
                    if !suffix.is_empty() && !next_terminals.contains(&suffix) {
                        next_terminals.push(suffix);
                    }
                }
            }
            for (target, item) in scanned {
                push_item(&mut charts, &mut seen, target, item);
            }
        }

        let complete = charts[n].iter().any(|item| {
            let production = &self.productions[item.production];
            item.origin == 0 && item.dot == production.symbols.len() && production.lhs == self.start
        });

        ParseOutcome {
            viable: !charts[n].is_empty() || partial_alive,
            complete,
            next_terminals,
        }
    }

    fn next_symbol(&self, item: &Item) -> Option<&GrammarSymbol> {
        self.productions[item.production].symbols.get(item.dot)
    }
}

fn push_item(charts: &mut [Vec<Item>], seen: &mut [HashSet<Item>], position: usize, item: Item) {
    if position >= charts.len() {
        return;
    }
    if seen[position].insert(item) {
        charts[position].push(item);
    }
}

/// Split the right-hand side of a rule on `|`, ignoring separators inside
/// quoted terminals.
fn split_alternatives(rhs: &str, line: usize) -> Result<Vec<String>, GrammarError> {
    let mut alternatives = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;

    for ch in rhs.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match quote {
            Some(open) => {
                current.push(ch);
                if ch == '\\' {
                    escaped = true;
                } else if ch == open {
                    quote = None;
                }
            },
            None => {
                if ch == '\'' || ch == '"' {
                    quote = Some(ch);
                    current.push(ch);
                } else if ch == '|' {
                    alternatives.push(current.trim().to_string());
                    current = String::new();
                } else {
                    current.push(ch);
                }
            },
        }
    }

    if quote.is_some() {
        return Err(GrammarError::UnterminatedTerminal {
            line,
            text: rhs.trim().to_string(),
        });
    }

    alternatives.push(current.trim().to_string());
    Ok(alternatives)
}

/// Tokenise one alternative into terminals and non-terminals.
fn parse_symbols(alternative: &str, line: usize) -> Result<Vec<GrammarSymbol>, GrammarError> {
    let mut symbols = Vec::new();
    let chars: Vec<char> = alternative.chars().collect();
    let mut index = 0;

    while index < chars.len() {
        let ch = chars[index];
        if ch.is_whitespace() {
            index += 1;
            continue;
        }

        if ch == '\'' || ch == '"' {
            let quote = ch;
            index += 1;
            let mut literal = String::new();
            let mut closed = false;
            while index < chars.len() {
                if chars[index] == '\\' && index + 1 < chars.len() {
                    literal.push(chars[index + 1]);
                    index += 2;
                    continue;
                }
                if chars[index] == quote {
                    closed = true;
                    index += 1;
                    break;
                }
                literal.push(chars[index]);
                index += 1;
            }
            if !closed {
                return Err(GrammarError::UnterminatedTerminal {
                    line,
                    text: alternative.to_string(),
                });
            }
            symbols.push(GrammarSymbol::Terminal(literal));
        } else {
            let mut name = String::new();
            while index < chars.len()
                && !chars[index].is_whitespace()
                && chars[index] != '\''
                && chars[index] != '"'
            {
                name.push(chars[index]);
                index += 1;
            }
            symbols.push(GrammarSymbol::NonTerminal(name));
        }
    }

    Ok(symbols)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grammar(text: &str) -> Grammar {
        Grammar::parse(text).expect("grammar parses").expect("grammar has rules")
    }

    #[test]
    fn test_empty_grammar_is_unconstrained() {
        assert!(Grammar::parse("").expect("parse").is_none());
        assert!(Grammar::parse("   \n# comment\n").expect("parse").is_none());
    }

    #[test]
    fn test_single_terminal_rule() {
        let g = grammar("start ::= 'hello'");
        assert!(g.recognize("hello").complete);
        assert!(g.recognize("hello").viable);

        let partial = g.recognize("hel");
        assert!(partial.viable, "a prefix of the only word is viable");
        assert!(!partial.complete);
        assert_eq!(partial.next_terminals, vec!["lo".to_string()]);

        let wrong = g.recognize("help");
        assert!(!wrong.viable, "`help` cannot become `hello`");
        assert!(!wrong.complete);
    }

    #[test]
    fn test_alternatives() {
        let g = grammar("start ::= 'cat' | 'dog'");
        assert!(g.recognize("cat").complete);
        assert!(g.recognize("dog").complete);
        assert!(!g.recognize("cow").viable);
        assert!(g.recognize("d").viable);
        assert!(!g.recognize("catdog").viable);
    }

    #[test]
    fn test_concatenated_nonterminals() {
        let g = grammar("start ::= noun verb\nnoun ::= 'cat' | 'dog'\nverb ::= 'runs'");
        assert!(g.recognize("catruns").complete);
        assert!(g.recognize("dogruns").complete);
        assert!(g.recognize("cat").viable);
        assert!(!g.recognize("cat").complete);
        assert!(!g.recognize("catwalks").viable);
    }

    #[test]
    fn test_left_recursive_grammar_terminates() {
        // Left recursion is exactly what a naive LL parser cannot do.
        let g = grammar("list ::= list ',' item | item\nitem ::= 'a' | 'b'");
        assert!(g.recognize("a").complete);
        assert!(g.recognize("a,b").complete);
        assert!(g.recognize("a,b,a,b").complete);
        assert!(!g.recognize("a,").complete);
        assert!(g.recognize("a,").viable);
        assert!(!g.recognize(",a").viable);
    }

    #[test]
    fn test_nullable_rule() {
        let g = grammar("start ::= 'a' maybe\nmaybe ::= 'b' |");
        assert!(g.recognize("a").complete);
        assert!(g.recognize("ab").complete);
        assert!(!g.recognize("ac").viable);
    }

    #[test]
    fn test_balanced_parentheses_is_context_free() {
        let g = grammar("s ::= '(' s ')' s |");
        assert!(g.recognize("").complete);
        assert!(g.recognize("()").complete);
        assert!(g.recognize("(())()").complete);
        assert!(!g.recognize("(()").complete);
        assert!(g.recognize("(()").viable);
        assert!(!g.recognize(")(").viable);
    }

    #[test]
    fn test_start_symbol_selection() {
        assert_eq!(grammar("a ::= 'x'\nb ::= 'y'").start_symbol(), "a");
        assert_eq!(grammar("a ::= 'x'\nstart ::= 'y'").start_symbol(), "start");
        assert_eq!(grammar("a ::= 'x'\nroot ::= 'y'").start_symbol(), "root");
    }

    #[test]
    fn test_undefined_nonterminal_is_an_error() {
        let err = Grammar::parse("start ::= missing").expect_err("should fail");
        assert_eq!(
            err,
            GrammarError::UndefinedNonTerminal {
                name: "missing".to_string()
            }
        );
    }

    #[test]
    fn test_unterminated_terminal_is_an_error() {
        assert!(matches!(
            Grammar::parse("start ::= 'oops"),
            Err(GrammarError::UnterminatedTerminal { .. })
        ));
    }

    #[test]
    fn test_pipe_inside_a_terminal_is_not_a_separator() {
        let g = grammar("start ::= 'a|b'");
        assert!(g.recognize("a|b").complete);
        assert_eq!(g.production_count(), 1);
    }

    #[test]
    fn test_next_terminals_reports_continuations() {
        let g = grammar("start ::= 'foo' | 'fizz'");
        let outcome = g.recognize("f");
        let mut continuations = outcome.next_terminals;
        continuations.sort();
        assert_eq!(continuations, vec!["izz".to_string(), "oo".to_string()]);
    }

    #[test]
    fn test_alternatives_accessor() {
        let g = grammar("start ::= 'a' | 'b'");
        let alternatives = g.alternatives("start");
        assert_eq!(alternatives.len(), 2);
        assert_eq!(alternatives[0], vec![GrammarSymbol::Terminal("a".into())]);
        assert!(g.alternatives("nope").is_empty());
    }

    #[test]
    fn test_escaped_quote_in_terminal() {
        let g = grammar(r"start ::= 'it\'s'");
        assert!(g.recognize("it's").complete);
    }
}
