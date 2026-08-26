//! Interactive Debugger for TrustformeRS
//!
//! Provides step-through execution, breakpoints, variable inspection,
//! call stack visualization, and time-travel debugging capabilities.
// reason: debug/profiling scaffolding — structs are constructed and their fields/methods
// are retained for the data model, serialization completeness, and future consumers that
// do not yet read every member. Consolidated from many item-level #[allow(dead_code)].
#![allow(dead_code)]

use anyhow::Result;
use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use uuid::Uuid;

use crate::DebugConfig;

/// Interactive debugger for step-through debugging and inspection
#[derive(Debug)]
pub struct InteractiveDebugger {
    config: DebugConfig,
    state: Arc<RwLock<DebuggerState>>,
    breakpoints: Arc<RwLock<HashMap<String, Breakpoint>>>,
    execution_history: Arc<Mutex<VecDeque<ExecutionSnapshot>>>,
    current_step: Arc<Mutex<usize>>,
    max_history_size: usize,
}

/// Current state of the debugger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebuggerState {
    pub is_running: bool,
    pub is_paused: bool,
    pub current_location: Option<DebugLocation>,
    pub call_stack: Vec<StackFrame>,
    pub variables: IndexMap<String, VariableValue>,
    pub step_mode: StepMode,
    pub session_start: DateTime<Utc>,
}

/// Debugging location identifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugLocation {
    pub module: String,
    pub function: String,
    pub line: Option<u32>,
    pub instruction: Option<String>,
    pub context: Option<String>,
}

/// Call stack frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    pub id: Uuid,
    pub location: DebugLocation,
    pub locals: IndexMap<String, VariableValue>,
    pub timestamp: DateTime<Utc>,
    pub depth: usize,
}

/// Variable value with type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableValue {
    pub name: String,
    pub value: String,
    pub type_name: String,
    pub size_bytes: Option<usize>,
    pub shape: Option<Vec<usize>>,
    pub is_tensor: bool,
    pub metadata: HashMap<String, String>,
}

/// Execution step mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepMode {
    /// Step into function calls
    StepInto,
    /// Step over function calls
    StepOver,
    /// Step out of current function
    StepOut,
    /// Continue until next breakpoint
    Continue,
    /// Single instruction step
    SingleStep,
}

/// Breakpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakpoint {
    pub id: Uuid,
    pub location: DebugLocation,
    pub condition: Option<String>,
    pub hit_count: usize,
    pub enabled: bool,
    pub temporary: bool,
    pub log_message: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Snapshot of execution state for time-travel debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSnapshot {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub step_number: usize,
    pub location: DebugLocation,
    pub call_stack: Vec<StackFrame>,
    pub variables: IndexMap<String, VariableValue>,
    pub memory_usage: Option<usize>,
    pub performance_metrics: HashMap<String, f64>,
}

/// Debugger command for external control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DebuggerCommand {
    Start,
    Pause,
    Resume,
    Step(StepMode),
    SetBreakpoint(DebugLocation, Option<String>),
    RemoveBreakpoint(Uuid),
    InspectVariable(String),
    EvaluateExpression(String),
    ShowCallStack,
    ShowHistory,
    JumpToStep(usize),
    Reset,
    Exit,
}

/// Response from debugger operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DebuggerResponse {
    Started,
    Paused(DebugLocation),
    Resumed,
    Stepped(ExecutionSnapshot),
    BreakpointHit(Breakpoint, ExecutionSnapshot),
    VariableInspected(VariableValue),
    ExpressionEvaluated(String),
    CallStackShown(Vec<StackFrame>),
    HistoryShown(Vec<ExecutionSnapshot>),
    JumpedToStep(ExecutionSnapshot),
    Reset,
    Error(String),
}

impl InteractiveDebugger {
    /// Create a new interactive debugger
    pub fn new(config: &DebugConfig) -> Self {
        Self {
            config: config.clone(),
            state: Arc::new(RwLock::new(DebuggerState {
                is_running: false,
                is_paused: false,
                current_location: None,
                call_stack: Vec::new(),
                variables: IndexMap::new(),
                step_mode: StepMode::Continue,
                session_start: Utc::now(),
            })),
            breakpoints: Arc::new(RwLock::new(HashMap::new())),
            execution_history: Arc::new(Mutex::new(VecDeque::new())),
            current_step: Arc::new(Mutex::new(0)),
            max_history_size: config.max_gradient_history, // Reuse config value
        }
    }

    /// Start the debugger
    pub async fn start(&mut self) -> Result<()> {
        let mut state = self.state.write();
        state.is_running = true;
        state.session_start = Utc::now();
        tracing::info!("Interactive debugger started");
        Ok(())
    }

    /// Process a debugger command
    pub async fn process_command(&self, command: DebuggerCommand) -> Result<DebuggerResponse> {
        match command {
            DebuggerCommand::Start => {
                let mut state = self.state.write();
                state.is_running = true;
                Ok(DebuggerResponse::Started)
            },

            DebuggerCommand::Pause => {
                let mut state = self.state.write();
                state.is_paused = true;
                if let Some(location) = &state.current_location {
                    Ok(DebuggerResponse::Paused(location.clone()))
                } else {
                    Ok(DebuggerResponse::Paused(DebugLocation {
                        module: "unknown".to_string(),
                        function: "unknown".to_string(),
                        line: None,
                        instruction: None,
                        context: None,
                    }))
                }
            },

            DebuggerCommand::Resume => {
                let mut state = self.state.write();
                state.is_paused = false;
                state.step_mode = StepMode::Continue;
                Ok(DebuggerResponse::Resumed)
            },

            DebuggerCommand::Step(mode) => self.execute_step(mode).await,

            DebuggerCommand::SetBreakpoint(location, condition) => {
                self.set_breakpoint(location, condition).await
            },

            DebuggerCommand::RemoveBreakpoint(id) => self.remove_breakpoint(id).await,

            DebuggerCommand::InspectVariable(name) => self.inspect_variable(&name).await,

            DebuggerCommand::EvaluateExpression(expr) => self.evaluate_expression(&expr).await,

            DebuggerCommand::ShowCallStack => {
                let state = self.state.read();
                Ok(DebuggerResponse::CallStackShown(state.call_stack.clone()))
            },

            DebuggerCommand::ShowHistory => {
                let history = self.execution_history.lock();
                Ok(DebuggerResponse::HistoryShown(
                    history.iter().cloned().collect(),
                ))
            },

            DebuggerCommand::JumpToStep(step_num) => self.jump_to_step(step_num).await,

            DebuggerCommand::Reset => self.reset().await,

            DebuggerCommand::Exit => {
                let mut state = self.state.write();
                state.is_running = false;
                Ok(DebuggerResponse::Reset)
            },
        }
    }

    /// Execute a debugging step
    async fn execute_step(&self, mode: StepMode) -> Result<DebuggerResponse> {
        let mut state = self.state.write();
        state.step_mode = mode;
        state.is_paused = true;

        // Create execution snapshot
        let snapshot = ExecutionSnapshot {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            step_number: {
                let mut step = self.current_step.lock();
                *step += 1;
                *step
            },
            location: state.current_location.clone().unwrap_or_else(|| DebugLocation {
                module: "runtime".to_string(),
                function: "step".to_string(),
                line: None,
                instruction: Some(format!("Step {:?}", mode)),
                context: None,
            }),
            call_stack: state.call_stack.clone(),
            variables: state.variables.clone(),
            memory_usage: None,
            performance_metrics: HashMap::new(),
        };

        // Add to history
        {
            let mut history = self.execution_history.lock();
            history.push_back(snapshot.clone());
            if history.len() > self.max_history_size {
                history.pop_front();
            }
        }

        Ok(DebuggerResponse::Stepped(snapshot))
    }

    /// Set a breakpoint
    async fn set_breakpoint(
        &self,
        location: DebugLocation,
        condition: Option<String>,
    ) -> Result<DebuggerResponse> {
        let breakpoint = Breakpoint {
            id: Uuid::new_v4(),
            location,
            condition,
            hit_count: 0,
            enabled: true,
            temporary: false,
            log_message: None,
            created_at: Utc::now(),
        };

        self.breakpoints.write().insert(breakpoint.id.to_string(), breakpoint.clone());

        tracing::info!(
            "Breakpoint set at {}::{}",
            breakpoint.location.module,
            breakpoint.location.function
        );
        Ok(DebuggerResponse::Started) // Use Started as generic success
    }

    /// Remove a breakpoint
    async fn remove_breakpoint(&self, id: Uuid) -> Result<DebuggerResponse> {
        if self.breakpoints.write().remove(&id.to_string()).is_some() {
            tracing::info!("Breakpoint {} removed", id);
        }
        Ok(DebuggerResponse::Started)
    }

    /// Inspect a variable
    async fn inspect_variable(&self, name: &str) -> Result<DebuggerResponse> {
        let state = self.state.read();
        if let Some(var) = state.variables.get(name) {
            Ok(DebuggerResponse::VariableInspected(var.clone()))
        } else {
            Ok(DebuggerResponse::Error(format!(
                "Variable '{}' not found",
                name
            )))
        }
    }

    /// Evaluate a numeric watch expression against the current debugger
    /// state (`state.variables`).
    ///
    /// Supports `+ - * / ( )`, numeric literals, and identifiers that
    /// resolve to a currently-tracked variable whose `value` string parses
    /// as `f64`. This is intentionally a minimal calculator, not a general
    /// interpreter -- but every result it returns is a real evaluation of
    /// the given expression against real debugger state. An expression this
    /// evaluator cannot parse, or one that references an unknown variable
    /// or a non-numeric one, returns [`DebuggerResponse::Error`] rather
    /// than a fabricated "not implemented" success -- so a client UI never
    /// displays a made-up value as if it were the real evaluated result.
    async fn evaluate_expression(&self, expr: &str) -> Result<DebuggerResponse> {
        let variables = self.state.read().variables.clone();
        match ExpressionEvaluator::new(expr, &variables).evaluate() {
            Ok(value) => Ok(DebuggerResponse::ExpressionEvaluated(format_eval_result(
                value,
            ))),
            Err(message) => Ok(DebuggerResponse::Error(message)),
        }
    }

    /// Jump to a specific step in history (time-travel debugging)
    async fn jump_to_step(&self, step_num: usize) -> Result<DebuggerResponse> {
        let history = self.execution_history.lock();
        if let Some(snapshot) = history.iter().find(|s| s.step_number == step_num) {
            let snapshot = snapshot.clone();
            drop(history);

            // Restore state from snapshot
            {
                let mut state = self.state.write();
                state.current_location = Some(snapshot.location.clone());
                state.call_stack = snapshot.call_stack.clone();
                state.variables = snapshot.variables.clone();
                state.is_paused = true;
            }

            *self.current_step.lock() = step_num;
            Ok(DebuggerResponse::JumpedToStep(snapshot))
        } else {
            Ok(DebuggerResponse::Error(format!(
                "Step {} not found in history",
                step_num
            )))
        }
    }

    /// Reset the debugger state
    pub async fn reset(&self) -> Result<DebuggerResponse> {
        {
            let mut state = self.state.write();
            *state = DebuggerState {
                is_running: false,
                is_paused: false,
                current_location: None,
                call_stack: Vec::new(),
                variables: IndexMap::new(),
                step_mode: StepMode::Continue,
                session_start: Utc::now(),
            };
        }

        self.breakpoints.write().clear();
        self.execution_history.lock().clear();
        *self.current_step.lock() = 0;

        Ok(DebuggerResponse::Reset)
    }

    /// Add a variable to the current scope
    pub async fn add_variable(&self, name: String, value: String, type_name: String) -> Result<()> {
        let var = VariableValue {
            name: name.clone(),
            value,
            type_name,
            size_bytes: None,
            shape: None,
            is_tensor: false,
            metadata: HashMap::new(),
        };

        self.state.write().variables.insert(name, var);
        Ok(())
    }

    /// Update current execution location
    pub async fn update_location(&self, location: DebugLocation) -> Result<()> {
        let mut state = self.state.write();
        state.current_location = Some(location.clone());

        // Check if we hit a breakpoint
        let breakpoints = self.breakpoints.read();
        for (_, breakpoint) in breakpoints.iter() {
            if breakpoint.enabled
                && breakpoint.location.module == location.module
                && breakpoint.location.function == location.function
            {
                state.is_paused = true;
                tracing::info!(
                    "Breakpoint hit at {}::{}",
                    location.module,
                    location.function
                );
                break;
            }
        }

        Ok(())
    }

    /// Push a new frame onto the call stack
    pub async fn push_frame(&self, location: DebugLocation) -> Result<()> {
        let frame = StackFrame {
            id: Uuid::new_v4(),
            location,
            locals: IndexMap::new(),
            timestamp: Utc::now(),
            depth: self.state.read().call_stack.len(),
        };

        self.state.write().call_stack.push(frame);
        Ok(())
    }

    /// Pop a frame from the call stack
    pub async fn pop_frame(&self) -> Result<Option<StackFrame>> {
        Ok(self.state.write().call_stack.pop())
    }

    /// Generate debugger report
    pub async fn generate_report(&self) -> Result<InteractiveDebuggerReport> {
        let state = self.state.read();
        let breakpoints = self.breakpoints.read();
        let history = self.execution_history.lock();

        Ok(InteractiveDebuggerReport {
            session_duration: Utc::now() - state.session_start,
            total_steps: *self.current_step.lock(),
            total_breakpoints: breakpoints.len(),
            breakpoint_hits: breakpoints.values().map(|b| b.hit_count).sum(),
            max_call_stack_depth: state.call_stack.len(),
            variables_tracked: state.variables.len(),
            history_entries: history.len(),
            current_state: state.clone(),
        })
    }

    /// Check if debugger is currently paused
    pub fn is_paused(&self) -> bool {
        self.state.read().is_paused
    }

    /// Get current step number
    pub fn current_step(&self) -> usize {
        *self.current_step.lock()
    }

    /// Get active breakpoints
    pub fn get_breakpoints(&self) -> Vec<Breakpoint> {
        self.breakpoints.read().values().cloned().collect()
    }
}

/// Report generated by the interactive debugger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveDebuggerReport {
    pub session_duration: chrono::Duration,
    pub total_steps: usize,
    pub total_breakpoints: usize,
    pub breakpoint_hits: usize,
    pub max_call_stack_depth: usize,
    pub variables_tracked: usize,
    pub history_entries: usize,
    pub current_state: DebuggerState,
}

impl Default for DebuggerState {
    fn default() -> Self {
        Self {
            is_running: false,
            is_paused: false,
            current_location: None,
            call_stack: Vec::new(),
            variables: IndexMap::new(),
            step_mode: StepMode::Continue,
            session_start: Utc::now(),
        }
    }
}

/// Formats a numeric evaluation result the way a debugger watch expression
/// display should: integral values print without a trailing `.0` (`"3"`,
/// not `"3.0"`), everything else prints with full `f64` precision.
fn format_eval_result(value: f64) -> String {
    if value.fract() == 0.0 && value.is_finite() && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// A minimal recursive-descent parser/evaluator for numeric watch
/// expressions over [`InteractiveDebugger`]'s tracked variables.
///
/// Grammar (standard precedence, left-associative `+ - * /`, right-assoc
/// unary `-`):
/// ```text
/// expr   := term (('+' | '-') term)*
/// term   := unary (('*' | '/') unary)*
/// unary  := '-' unary | atom
/// atom   := NUMBER | IDENT | '(' expr ')'
/// ```
struct ExpressionEvaluator<'a> {
    tokens: Vec<Token>,
    pos: usize,
    variables: &'a IndexMap<String, VariableValue>,
    source: &'a str,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
}

impl<'a> ExpressionEvaluator<'a> {
    fn new(expr: &'a str, variables: &'a IndexMap<String, VariableValue>) -> Self {
        Self {
            tokens: Vec::new(),
            pos: 0,
            variables,
            source: expr,
        }
    }

    /// Parse and evaluate the expression, returning either the numeric
    /// result or a human-readable error describing exactly what went wrong
    /// (unknown variable, non-numeric variable, malformed syntax, division
    /// by zero, or trailing unparsed input).
    fn evaluate(mut self) -> Result<f64, String> {
        self.tokenize()?;
        if self.tokens.is_empty() {
            return Err("empty expression".to_string());
        }
        let value = self.parse_expr()?;
        if self.pos != self.tokens.len() {
            return Err(format!(
                "unexpected trailing input in expression {:?} at token {}",
                self.source, self.pos
            ));
        }
        Ok(value)
    }

    fn tokenize(&mut self) -> Result<(), String> {
        let chars: Vec<char> = self.source.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            match c {
                ' ' | '\t' | '\n' | '\r' => {
                    i += 1;
                },
                '+' => {
                    self.tokens.push(Token::Plus);
                    i += 1;
                },
                '-' => {
                    self.tokens.push(Token::Minus);
                    i += 1;
                },
                '*' => {
                    self.tokens.push(Token::Star);
                    i += 1;
                },
                '/' => {
                    self.tokens.push(Token::Slash);
                    i += 1;
                },
                '(' => {
                    self.tokens.push(Token::LParen);
                    i += 1;
                },
                ')' => {
                    self.tokens.push(Token::RParen);
                    i += 1;
                },
                c if c.is_ascii_digit() || c == '.' => {
                    let start = i;
                    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                        i += 1;
                    }
                    let text: String = chars[start..i].iter().collect();
                    let value = text.parse::<f64>().map_err(|_| {
                        format!("invalid number literal {text:?} in {:?}", self.source)
                    })?;
                    self.tokens.push(Token::Number(value));
                },
                c if c.is_alphabetic() || c == '_' => {
                    let start = i;
                    while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                        i += 1;
                    }
                    let text: String = chars[start..i].iter().collect();
                    self.tokens.push(Token::Ident(text));
                },
                other => {
                    return Err(format!(
                        "unsupported character {other:?} in expression {:?}",
                        self.source
                    ));
                },
            }
        }
        Ok(())
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.pos).cloned();
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn parse_expr(&mut self) -> Result<f64, String> {
        let mut value = self.parse_term()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.advance();
                    value += self.parse_term()?;
                },
                Some(Token::Minus) => {
                    self.advance();
                    value -= self.parse_term()?;
                },
                _ => break,
            }
        }
        Ok(value)
    }

    fn parse_term(&mut self) -> Result<f64, String> {
        let mut value = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.advance();
                    value *= self.parse_unary()?;
                },
                Some(Token::Slash) => {
                    self.advance();
                    let divisor = self.parse_unary()?;
                    if divisor == 0.0 {
                        return Err(format!("division by zero in expression {:?}", self.source));
                    }
                    value /= divisor;
                },
                _ => break,
            }
        }
        Ok(value)
    }

    fn parse_unary(&mut self) -> Result<f64, String> {
        if matches!(self.peek(), Some(Token::Minus)) {
            self.advance();
            return Ok(-self.parse_unary()?);
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<f64, String> {
        match self.advance() {
            Some(Token::Number(n)) => Ok(n),
            Some(Token::Ident(name)) => {
                let var = self.variables.get(&name).ok_or_else(|| {
                    format!(
                        "unknown variable {name:?} referenced in expression {:?}",
                        self.source
                    )
                })?;
                var.value.trim().parse::<f64>().map_err(|_| {
                    format!(
                        "variable {name:?} has non-numeric value {:?} and cannot be used in an \
                         arithmetic expression",
                        var.value
                    )
                })
            },
            Some(Token::LParen) => {
                let value = self.parse_expr()?;
                match self.advance() {
                    Some(Token::RParen) => Ok(value),
                    _ => Err(format!(
                        "missing closing ')' in expression {:?}",
                        self.source
                    )),
                }
            },
            other => Err(format!(
                "unexpected token {other:?} in expression {:?}",
                self.source
            )),
        }
    }
}

#[cfg(test)]
mod expression_evaluator_tests {
    use super::*;

    fn var(name: &str, value: &str) -> VariableValue {
        VariableValue {
            name: name.to_string(),
            value: value.to_string(),
            type_name: "f64".to_string(),
            size_bytes: None,
            shape: None,
            is_tensor: false,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_numeric_literal() {
        let vars = IndexMap::new();
        let result = ExpressionEvaluator::new("42", &vars).evaluate();
        assert_eq!(result, Ok(42.0));
    }

    #[test]
    fn test_basic_arithmetic_precedence() {
        let vars = IndexMap::new();
        // 2 + 3 * 4 = 14, not 20 -- confirms real operator precedence.
        assert_eq!(
            ExpressionEvaluator::new("2 + 3 * 4", &vars).evaluate(),
            Ok(14.0)
        );
    }

    #[test]
    fn test_parentheses_override_precedence() {
        let vars = IndexMap::new();
        assert_eq!(
            ExpressionEvaluator::new("(2 + 3) * 4", &vars).evaluate(),
            Ok(20.0)
        );
    }

    #[test]
    fn test_unary_minus() {
        let vars = IndexMap::new();
        assert_eq!(
            ExpressionEvaluator::new("-5 + 3", &vars).evaluate(),
            Ok(-2.0)
        );
    }

    #[test]
    fn test_variable_lookup_resolves_real_value() {
        let mut vars = IndexMap::new();
        vars.insert("loss".to_string(), var("loss", "0.485"));
        let result = ExpressionEvaluator::new("loss", &vars).evaluate();
        assert_eq!(result, Ok(0.485));
    }

    #[test]
    fn test_variable_in_arithmetic_expression() {
        let mut vars = IndexMap::new();
        vars.insert("grad_norm".to_string(), var("grad_norm", "2.5"));
        let result = ExpressionEvaluator::new("grad_norm * 2", &vars).evaluate();
        assert_eq!(result, Ok(5.0));
    }

    #[test]
    fn test_unknown_variable_is_a_real_error_not_a_placeholder() {
        let vars = IndexMap::new();
        let result = ExpressionEvaluator::new("undefined_var", &vars).evaluate();
        assert!(result.is_err());
        assert!(result.expect_err("should be an error").contains("unknown variable"));
    }

    #[test]
    fn test_non_numeric_variable_is_a_real_error() {
        let mut vars = IndexMap::new();
        vars.insert("model_name".to_string(), var("model_name", "gpt2"));
        let result = ExpressionEvaluator::new("model_name", &vars).evaluate();
        assert!(result.is_err());
        assert!(result.expect_err("should be an error").contains("non-numeric"));
    }

    #[test]
    fn test_division_by_zero_is_a_real_error() {
        let vars = IndexMap::new();
        let result = ExpressionEvaluator::new("1 / 0", &vars).evaluate();
        assert!(result.is_err());
        assert!(result.expect_err("should be an error").contains("division by zero"));
    }

    #[test]
    fn test_malformed_expression_is_a_real_error() {
        let vars = IndexMap::new();
        assert!(ExpressionEvaluator::new("(1 + 2", &vars).evaluate().is_err());
        assert!(ExpressionEvaluator::new("1 +", &vars).evaluate().is_err());
        assert!(ExpressionEvaluator::new("1 2", &vars).evaluate().is_err());
        assert!(ExpressionEvaluator::new("1 $ 2", &vars).evaluate().is_err());
    }

    #[test]
    fn test_format_eval_result_integral_vs_fractional() {
        assert_eq!(format_eval_result(3.0), "3");
        assert_eq!(format_eval_result(3.5), "3.5");
        assert_eq!(format_eval_result(-2.0), "-2");
    }
}

#[cfg(test)]
mod evaluate_expression_integration_tests {
    use super::*;

    fn make_debug_config() -> DebugConfig {
        DebugConfig::default()
    }

    #[tokio::test]
    async fn test_process_command_returns_error_for_unknown_variable_not_fake_success() {
        let debugger = InteractiveDebugger::new(&make_debug_config());
        let response = debugger
            .process_command(DebuggerCommand::EvaluateExpression(
                "nonexistent".to_string(),
            ))
            .await
            .expect("process_command should not itself error");

        // The old implementation returned `ExpressionEvaluated("Expression
        // evaluation not implemented")` here -- a client would have
        // displayed that placeholder text as the evaluated value.
        match response {
            DebuggerResponse::Error(message) => {
                assert!(message.contains("unknown variable"));
            },
            other => panic!("expected DebuggerResponse::Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_process_command_evaluates_real_arithmetic() {
        let debugger = InteractiveDebugger::new(&make_debug_config());
        let response = debugger
            .process_command(DebuggerCommand::EvaluateExpression(
                "(2 + 3) * 4".to_string(),
            ))
            .await
            .expect("process_command should not error");

        match response {
            DebuggerResponse::ExpressionEvaluated(value) => assert_eq!(value, "20"),
            other => panic!("expected DebuggerResponse::ExpressionEvaluated, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_process_command_evaluates_expression_over_real_tracked_variable() {
        let debugger = InteractiveDebugger::new(&make_debug_config());
        {
            let mut state = debugger.state.write();
            state.variables.insert(
                "batch_size".to_string(),
                VariableValue {
                    name: "batch_size".to_string(),
                    value: "32".to_string(),
                    type_name: "usize".to_string(),
                    size_bytes: None,
                    shape: None,
                    is_tensor: false,
                    metadata: HashMap::new(),
                },
            );
        }

        let response = debugger
            .process_command(DebuggerCommand::EvaluateExpression(
                "batch_size * 2".to_string(),
            ))
            .await
            .expect("process_command should not error");

        match response {
            DebuggerResponse::ExpressionEvaluated(value) => assert_eq!(value, "64"),
            other => panic!("expected DebuggerResponse::ExpressionEvaluated, got {other:?}"),
        }
    }
}
