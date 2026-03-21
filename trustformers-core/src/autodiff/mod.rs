//! Automatic differentiation framework for gradient computation.
//!
//! This module provides a comprehensive automatic differentiation (autodiff) system
//! that can compute gradients automatically for any computation graph. It supports
//! both forward-mode and reverse-mode automatic differentiation.

pub mod debugger;
pub mod engine;
pub mod gradient_checker;
pub mod graph;
pub mod operations;
pub mod tape;
pub mod variable;

pub use debugger::{
    AnalysisResult, DebuggerConfig, GradientFlowStats, GraphDebugger, GraphIssue,
    GraphOutputFormat, IssueSeverity, IssueType, MemoryStats, NodeDebugInfo, TraversalInfo,
};
pub use engine::{AutodiffEngine, GradientMode};
pub use gradient_checker::{
    check_and_report_gradients, check_gradients, ElementError, GradientCheckConfig,
    GradientCheckResult, GradientCheckStats, GradientChecker,
};
pub use graph::{ComputationGraph, GraphNode, NodeId, OperationType};
pub use operations::{grad_fn, AutodiffOp, OpType};
pub use tape::{GradientTape, TapeEntry};
pub use variable::{Variable, VariableRef};
