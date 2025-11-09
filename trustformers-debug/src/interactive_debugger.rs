//! Interactive Debugger for TrustformeRS
//!
//! Provides step-through execution, breakpoints, variable inspection,
//! call stack visualization, and time-travel debugging capabilities.

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
    #[allow(dead_code)]
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

    /// Evaluate an expression
    async fn evaluate_expression(&self, _expr: &str) -> Result<DebuggerResponse> {
        // Simplified expression evaluation - in a real implementation,
        // this would parse and evaluate the expression
        Ok(DebuggerResponse::ExpressionEvaluated(
            "Expression evaluation not implemented".to_string(),
        ))
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
