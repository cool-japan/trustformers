//! IDE integration tools for TrustformeRS debugging
//!
//! Provides interfaces for IDE plugins and development environment integration

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

use crate::{DebugConfig, InteractiveDebugger};

/// IDE plugin interface
#[derive(Debug)]
pub struct IDEPlugin {
    pub plugin_id: Uuid,
    pub name: String,
    pub version: String,
    pub supported_ides: Vec<SupportedIDE>,
    pub capabilities: IDECapabilities,
    pub debugger: Option<InteractiveDebugger>,
    pub config: DebugConfig,
}

/// Supported IDE types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupportedIDE {
    VSCode,
    IntelliJ,
    Vim,
    Emacs,
    Sublime,
    Atom,
    Jupyter,
    JupyterLab,
    Custom(String),
}

/// IDE plugin capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IDECapabilities {
    pub syntax_highlighting: bool,
    pub code_completion: bool,
    pub inline_debugging: bool,
    pub tensor_visualization: bool,
    pub real_time_metrics: bool,
    pub breakpoint_management: bool,
    pub call_stack_navigation: bool,
    pub variable_inspection: bool,
    pub performance_profiling: bool,
    pub error_annotations: bool,
    pub jupyter_widgets: bool,
    pub interactive_plots: bool,
    pub notebook_integration: bool,
    pub kernel_communication: bool,
}

/// IDE message protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IDEMessage {
    // Debug session management
    StartDebugSession {
        session_id: Uuid,
    },
    StopDebugSession {
        session_id: Uuid,
    },

    // Breakpoint management
    SetBreakpoint {
        file: PathBuf,
        line: u32,
        condition: Option<String>,
    },
    RemoveBreakpoint {
        file: PathBuf,
        line: u32,
    },
    ToggleBreakpoint {
        file: PathBuf,
        line: u32,
    },

    // Debug control
    StepInto,
    StepOver,
    StepOut,
    Continue,
    Pause,

    // Variable inspection
    InspectVariable {
        variable_name: String,
    },
    EvaluateExpression {
        expression: String,
    },

    // Visualization requests
    ShowTensorVisualization {
        tensor_name: String,
    },
    ShowGradientFlow {
        layer_name: String,
    },
    ShowLossLandscape,
    ShowPerformanceMetrics,

    // Code navigation
    GotoDefinition {
        symbol: String,
    },
    FindReferences {
        symbol: String,
    },
    ShowCallStack,

    // Error handling
    ShowError {
        message: String,
        file: Option<PathBuf>,
        line: Option<u32>,
    },
    ShowWarning {
        message: String,
        file: Option<PathBuf>,
        line: Option<u32>,
    },

    // Status updates
    UpdateStatus {
        status: String,
    },
    UpdateProgress {
        progress: f64,
        message: String,
    },

    // Jupyter-specific messages
    CreateWidget {
        widget_type: String,
        widget_id: String,
        options: HashMap<String, String>,
    },
    UpdateWidget {
        widget_id: String,
        data: HashMap<String, String>,
    },
    RemoveWidget {
        widget_id: String,
    },
    ExecuteCell {
        cell_content: String,
    },
    InsertCell {
        content: String,
        cell_type: String,
    },
    ShowNotebook {
        notebook_content: String,
    },
    KernelRestart,
    KernelInterrupt,
    DisplayData {
        data: HashMap<String, String>,
        metadata: HashMap<String, String>,
    },
    StreamOutput {
        stream_type: String,
        text: String,
    },
}

/// IDE response messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IDEResponse {
    // Session responses
    SessionStarted {
        session_id: Uuid,
    },
    SessionStopped {
        session_id: Uuid,
    },
    SessionError {
        error: String,
    },

    // Debug responses
    BreakpointSet {
        file: PathBuf,
        line: u32,
    },
    BreakpointRemoved {
        file: PathBuf,
        line: u32,
    },
    BreakpointHit {
        file: PathBuf,
        line: u32,
    },

    // Execution responses
    ExecutionPaused {
        location: DebugLocation,
    },
    ExecutionResumed,
    ExecutionStepped {
        location: DebugLocation,
    },

    // Variable responses
    VariableValue {
        name: String,
        value: String,
        type_name: String,
    },
    ExpressionResult {
        expression: String,
        result: String,
    },

    // Visualization responses
    VisualizationData {
        data_type: String,
        data: Vec<u8>,
    },
    VisualizationPath {
        path: PathBuf,
    },

    // Navigation responses
    DefinitionLocation {
        file: PathBuf,
        line: u32,
        column: u32,
    },
    ReferenceLocations {
        locations: Vec<SourceLocation>,
    },
    CallStackData {
        frames: Vec<CallStackFrame>,
    },

    // General responses
    Success {
        message: String,
    },
    Error {
        error: String,
    },
    Warning {
        warning: String,
    },

    // Status responses
    StatusUpdate {
        status: String,
    },
    ProgressUpdate {
        progress: f64,
        message: String,
    },
}

/// Debug location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugLocation {
    pub file: PathBuf,
    pub line: u32,
    pub column: u32,
    pub function: String,
    pub module: String,
}

/// Source location for navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: PathBuf,
    pub line: u32,
    pub column: u32,
    pub context: String,
}

/// Call stack frame information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallStackFrame {
    pub function: String,
    pub file: PathBuf,
    pub line: u32,
    pub variables: HashMap<String, String>,
}

/// IDE plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IDEPluginConfig {
    pub enable_syntax_highlighting: bool,
    pub enable_code_completion: bool,
    pub enable_inline_debugging: bool,
    pub enable_tensor_visualization: bool,
    pub enable_real_time_metrics: bool,
    pub auto_open_debugger: bool,
    pub visualization_format: String,
    pub debug_port: u16,
    pub max_variable_display_length: usize,
    pub refresh_interval_ms: u64,
    pub workspace_root: PathBuf,
    pub log_level: String,
}

impl Default for IDEPluginConfig {
    fn default() -> Self {
        Self {
            enable_syntax_highlighting: true,
            enable_code_completion: true,
            enable_inline_debugging: true,
            enable_tensor_visualization: true,
            enable_real_time_metrics: true,
            auto_open_debugger: false,
            visualization_format: "png".to_string(),
            debug_port: 8899,
            max_variable_display_length: 1000,
            refresh_interval_ms: 1000,
            workspace_root: PathBuf::from("."),
            log_level: "info".to_string(),
        }
    }
}

impl IDEPlugin {
    /// Create a new IDE plugin
    pub fn new(name: String, version: String, config: DebugConfig) -> Self {
        Self {
            plugin_id: Uuid::new_v4(),
            name,
            version,
            supported_ides: vec![SupportedIDE::VSCode, SupportedIDE::IntelliJ],
            capabilities: IDECapabilities {
                syntax_highlighting: true,
                code_completion: true,
                inline_debugging: true,
                tensor_visualization: true,
                real_time_metrics: true,
                breakpoint_management: true,
                call_stack_navigation: true,
                variable_inspection: true,
                performance_profiling: true,
                error_annotations: true,
                jupyter_widgets: true,
                interactive_plots: true,
                notebook_integration: true,
                kernel_communication: true,
            },
            debugger: None,
            config,
        }
    }

    /// Initialize the plugin with a debugger
    pub async fn initialize(&mut self, debugger: InteractiveDebugger) -> Result<()> {
        self.debugger = Some(debugger);
        tracing::info!("IDE plugin '{}' initialized", self.name);
        Ok(())
    }

    /// Handle IDE messages
    pub async fn handle_message(&mut self, message: IDEMessage) -> Result<IDEResponse> {
        match message {
            IDEMessage::StartDebugSession { session_id } => {
                if let Some(ref mut debugger) = self.debugger {
                    debugger.start().await?;
                    Ok(IDEResponse::SessionStarted { session_id })
                } else {
                    Ok(IDEResponse::SessionError {
                        error: "Debugger not initialized".to_string(),
                    })
                }
            },

            IDEMessage::StopDebugSession { session_id } => {
                if let Some(ref mut debugger) = self.debugger {
                    debugger.reset().await?;
                    Ok(IDEResponse::SessionStopped { session_id })
                } else {
                    Ok(IDEResponse::SessionError {
                        error: "Debugger not initialized".to_string(),
                    })
                }
            },

            IDEMessage::SetBreakpoint {
                file,
                line,
                condition,
            } => {
                if let Some(ref debugger) = self.debugger {
                    let location = crate::interactive_debugger::DebugLocation {
                        module: file.file_stem().unwrap_or_default().to_string_lossy().to_string(),
                        function: "unknown".to_string(),
                        line: Some(line),
                        instruction: None,
                        context: Some(file.to_string_lossy().to_string()),
                    };

                    debugger
                        .process_command(
                            crate::interactive_debugger::DebuggerCommand::SetBreakpoint(
                                location, condition,
                            ),
                        )
                        .await?;

                    Ok(IDEResponse::BreakpointSet { file, line })
                } else {
                    Ok(IDEResponse::Error {
                        error: "Debugger not initialized".to_string(),
                    })
                }
            },

            IDEMessage::RemoveBreakpoint { file, line } => {
                // For simplicity, we'll just return success
                // In a real implementation, we'd need to track breakpoint IDs
                Ok(IDEResponse::BreakpointRemoved { file, line })
            },

            IDEMessage::StepInto => {
                if let Some(ref debugger) = self.debugger {
                    let _response = debugger
                        .process_command(crate::interactive_debugger::DebuggerCommand::Step(
                            crate::interactive_debugger::StepMode::StepInto,
                        ))
                        .await?;

                    Ok(IDEResponse::ExecutionStepped {
                        location: DebugLocation {
                            file: PathBuf::from("unknown"),
                            line: 0,
                            column: 0,
                            function: "unknown".to_string(),
                            module: "unknown".to_string(),
                        },
                    })
                } else {
                    Ok(IDEResponse::Error {
                        error: "Debugger not initialized".to_string(),
                    })
                }
            },

            IDEMessage::StepOver => {
                if let Some(ref debugger) = self.debugger {
                    debugger
                        .process_command(crate::interactive_debugger::DebuggerCommand::Step(
                            crate::interactive_debugger::StepMode::StepOver,
                        ))
                        .await?;

                    Ok(IDEResponse::ExecutionStepped {
                        location: DebugLocation {
                            file: PathBuf::from("unknown"),
                            line: 0,
                            column: 0,
                            function: "unknown".to_string(),
                            module: "unknown".to_string(),
                        },
                    })
                } else {
                    Ok(IDEResponse::Error {
                        error: "Debugger not initialized".to_string(),
                    })
                }
            },

            IDEMessage::Continue => {
                if let Some(ref debugger) = self.debugger {
                    debugger
                        .process_command(crate::interactive_debugger::DebuggerCommand::Resume)
                        .await?;

                    Ok(IDEResponse::ExecutionResumed)
                } else {
                    Ok(IDEResponse::Error {
                        error: "Debugger not initialized".to_string(),
                    })
                }
            },

            IDEMessage::Pause => {
                if let Some(ref debugger) = self.debugger {
                    debugger
                        .process_command(crate::interactive_debugger::DebuggerCommand::Pause)
                        .await?;

                    Ok(IDEResponse::ExecutionPaused {
                        location: DebugLocation {
                            file: PathBuf::from("unknown"),
                            line: 0,
                            column: 0,
                            function: "unknown".to_string(),
                            module: "unknown".to_string(),
                        },
                    })
                } else {
                    Ok(IDEResponse::Error {
                        error: "Debugger not initialized".to_string(),
                    })
                }
            },

            IDEMessage::InspectVariable { variable_name } => {
                if let Some(ref debugger) = self.debugger {
                    let response = debugger
                        .process_command(
                            crate::interactive_debugger::DebuggerCommand::InspectVariable(
                                variable_name.clone(),
                            ),
                        )
                        .await?;

                    match response {
                        crate::interactive_debugger::DebuggerResponse::VariableInspected(var) => {
                            Ok(IDEResponse::VariableValue {
                                name: var.name,
                                value: var.value,
                                type_name: var.type_name,
                            })
                        },
                        _ => Ok(IDEResponse::Error {
                            error: format!("Variable '{}' not found", variable_name),
                        }),
                    }
                } else {
                    Ok(IDEResponse::Error {
                        error: "Debugger not initialized".to_string(),
                    })
                }
            },

            IDEMessage::EvaluateExpression { expression } => {
                if let Some(ref debugger) = self.debugger {
                    let response = debugger
                        .process_command(
                            crate::interactive_debugger::DebuggerCommand::EvaluateExpression(
                                expression.clone(),
                            ),
                        )
                        .await?;

                    match response {
                        crate::interactive_debugger::DebuggerResponse::ExpressionEvaluated(
                            result,
                        ) => Ok(IDEResponse::ExpressionResult { expression, result }),
                        _ => Ok(IDEResponse::Error {
                            error: "Expression evaluation failed".to_string(),
                        }),
                    }
                } else {
                    Ok(IDEResponse::Error {
                        error: "Debugger not initialized".to_string(),
                    })
                }
            },

            IDEMessage::ShowCallStack => {
                if let Some(ref debugger) = self.debugger {
                    let response = debugger
                        .process_command(
                            crate::interactive_debugger::DebuggerCommand::ShowCallStack,
                        )
                        .await?;

                    match response {
                        crate::interactive_debugger::DebuggerResponse::CallStackShown(frames) => {
                            let call_stack_frames: Vec<CallStackFrame> = frames
                                .into_iter()
                                .map(|frame| CallStackFrame {
                                    function: frame.location.function,
                                    file: PathBuf::from(frame.location.context.unwrap_or_default()),
                                    line: frame.location.line.unwrap_or(0),
                                    variables: frame
                                        .locals
                                        .into_iter()
                                        .map(|(k, v)| (k, v.value))
                                        .collect(),
                                })
                                .collect();

                            Ok(IDEResponse::CallStackData {
                                frames: call_stack_frames,
                            })
                        },
                        _ => Ok(IDEResponse::Error {
                            error: "Failed to get call stack".to_string(),
                        }),
                    }
                } else {
                    Ok(IDEResponse::Error {
                        error: "Debugger not initialized".to_string(),
                    })
                }
            },

            IDEMessage::ShowTensorVisualization { tensor_name } => {
                // This would integrate with the visualization system
                Ok(IDEResponse::Success {
                    message: format!("Tensor visualization for '{}' requested", tensor_name),
                })
            },

            IDEMessage::ShowGradientFlow { layer_name } => Ok(IDEResponse::Success {
                message: format!("Gradient flow visualization for '{}' requested", layer_name),
            }),

            IDEMessage::ShowLossLandscape => Ok(IDEResponse::Success {
                message: "Loss landscape visualization requested".to_string(),
            }),

            IDEMessage::ShowPerformanceMetrics => Ok(IDEResponse::Success {
                message: "Performance metrics visualization requested".to_string(),
            }),

            IDEMessage::GotoDefinition { symbol: _ } => {
                // Simplified implementation
                Ok(IDEResponse::DefinitionLocation {
                    file: PathBuf::from("src/lib.rs"),
                    line: 1,
                    column: 1,
                })
            },

            IDEMessage::FindReferences { symbol } => {
                // Simplified implementation
                Ok(IDEResponse::ReferenceLocations {
                    locations: vec![SourceLocation {
                        file: PathBuf::from("src/lib.rs"),
                        line: 1,
                        column: 1,
                        context: format!("Reference to {}", symbol),
                    }],
                })
            },

            IDEMessage::ShowError {
                message,
                file,
                line,
            } => {
                tracing::error!("IDE Error: {} at {:?}:{:?}", message, file, line);
                Ok(IDEResponse::Success {
                    message: "Error displayed".to_string(),
                })
            },

            IDEMessage::ShowWarning {
                message,
                file,
                line,
            } => {
                tracing::warn!("IDE Warning: {} at {:?}:{:?}", message, file, line);
                Ok(IDEResponse::Success {
                    message: "Warning displayed".to_string(),
                })
            },

            IDEMessage::UpdateStatus { status } => Ok(IDEResponse::StatusUpdate { status }),

            IDEMessage::UpdateProgress { progress, message } => {
                Ok(IDEResponse::ProgressUpdate { progress, message })
            },

            _ => Ok(IDEResponse::Error {
                error: "Unsupported message type".to_string(),
            }),
        }
    }

    /// Generate plugin manifest for different IDEs
    pub fn generate_manifest(&self, ide: &SupportedIDE) -> Result<String> {
        match ide {
            SupportedIDE::VSCode => self.generate_vscode_manifest(),
            SupportedIDE::IntelliJ => self.generate_intellij_manifest(),
            SupportedIDE::Vim => self.generate_vim_manifest(),
            SupportedIDE::Emacs => self.generate_emacs_manifest(),
            _ => Ok("Generic plugin manifest".to_string()),
        }
    }

    fn generate_vscode_manifest(&self) -> Result<String> {
        let manifest = serde_json::json!({
            "name": "trustformers-debug",
            "displayName": "TrustformeRS Debug",
            "description": "Advanced debugging tools for TrustformeRS models",
            "version": self.version,
            "engines": {
                "vscode": "^1.60.0"
            },
            "categories": ["Debuggers", "Machine Learning"],
            "main": "./out/extension.js",
            "contributes": {
                "debuggers": [{
                    "type": "trustformers",
                    "label": "TrustformeRS Debug",
                    "program": "./out/debugAdapter.js",
                    "runtime": "node",
                    "configurationAttributes": {
                        "launch": {
                            "required": ["program"],
                            "properties": {
                                "program": {
                                    "type": "string",
                                    "description": "Path to TrustformeRS program"
                                },
                                "args": {
                                    "type": "array",
                                    "description": "Command line arguments"
                                },
                                "cwd": {
                                    "type": "string",
                                    "description": "Working directory"
                                }
                            }
                        }
                    }
                }],
                "commands": [{
                    "command": "trustformers.showTensorVisualization",
                    "title": "Show Tensor Visualization",
                    "category": "TrustformeRS"
                }, {
                    "command": "trustformers.showGradientFlow",
                    "title": "Show Gradient Flow",
                    "category": "TrustformeRS"
                }, {
                    "command": "trustformers.showLossLandscape",
                    "title": "Show Loss Landscape",
                    "category": "TrustformeRS"
                }]
            },
            "scripts": {
                "vscode:prepublish": "npm run compile",
                "compile": "tsc -p ./",
                "watch": "tsc -watch -p ./"
            }
        });

        Ok(serde_json::to_string_pretty(&manifest)?)
    }

    fn generate_intellij_manifest(&self) -> Result<String> {
        let manifest = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<idea-plugin>
    <id>com.trustformers.debug</id>
    <name>TrustformeRS Debug</name>
    <version>{}</version>
    <vendor>TrustformeRS Team</vendor>

    <description><![CDATA[
        Advanced debugging tools for TrustformeRS models including:
        - Interactive debugging
        - Tensor visualization
        - Gradient flow analysis
        - Loss landscape visualization
    ]]></description>

    <depends>com.intellij.modules.platform</depends>
    <depends>com.intellij.modules.lang</depends>

    <extensions defaultExtensionNs="com.intellij">
        <debugger.positionManagerFactory implementation="com.trustformers.debug.TrustformersPositionManagerFactory"/>
        <xdebugger.breakpointType implementation="com.trustformers.debug.TrustformersLineBreakpointType"/>
        <programRunner implementation="com.trustformers.debug.TrustformersDebugRunner"/>
        <configurationType implementation="com.trustformers.debug.TrustformersConfigurationType"/>
    </extensions>

    <actions>
        <action id="TrustformeRS.ShowTensorVisualization" class="com.trustformers.debug.actions.ShowTensorVisualizationAction" text="Show Tensor Visualization"/>
        <action id="TrustformeRS.ShowGradientFlow" class="com.trustformers.debug.actions.ShowGradientFlowAction" text="Show Gradient Flow"/>
        <action id="TrustformeRS.ShowLossLandscape" class="com.trustformers.debug.actions.ShowLossLandscapeAction" text="Show Loss Landscape"/>
    </actions>
</idea-plugin>"#,
            self.version
        );

        Ok(manifest)
    }

    fn generate_vim_manifest(&self) -> Result<String> {
        let manifest = format!(
            r#"" TrustformeRS Debug Plugin for Vim
" Version: {}
" Description: Advanced debugging tools for TrustformeRS models

if exists('g:loaded_trustformers_debug')
    finish
endif
let g:loaded_trustformers_debug = 1

" Plugin configuration
let g:trustformers_debug_port = get(g:, 'trustformers_debug_port', 8899)
let g:trustformers_debug_auto_open = get(g:, 'trustformers_debug_auto_open', 0)
let g:trustformers_debug_visualization_format = get(g:, 'trustformers_debug_visualization_format', 'png')

" Commands
command! TrustformersStartDebug call trustformers#debug#start()
command! TrustformersStopDebug call trustformers#debug#stop()
command! TrustformersStepInto call trustformers#debug#step_into()
command! TrustformersStepOver call trustformers#debug#step_over()
command! TrustformersShowTensorVisualization call trustformers#debug#show_tensor_visualization()
command! TrustformersShowGradientFlow call trustformers#debug#show_gradient_flow()
command! TrustformersShowLossLandscape call trustformers#debug#show_loss_landscape()

" Key mappings
nnoremap <leader>tds :TrustformersStartDebug<CR>
nnoremap <leader>tdt :TrustformersStopDebug<CR>
nnoremap <leader>tdi :TrustformersStepInto<CR>
nnoremap <leader>tdo :TrustformersStepOver<CR>
nnoremap <leader>tdv :TrustformersShowTensorVisualization<CR>
nnoremap <leader>tdg :TrustformersShowGradientFlow<CR>
nnoremap <leader>tdl :TrustformersShowLossLandscape<CR>

" Autocommands
augroup TrustformersDebug
    autocmd!
    autocmd FileType rust call trustformers#debug#setup_buffer()
augroup END"#,
            self.version
        );

        Ok(manifest)
    }

    fn generate_emacs_manifest(&self) -> Result<String> {
        let manifest = format!(
            r#";;; trustformers-debug.el --- Advanced debugging tools for TrustformeRS models  -*- lexical-binding: t; -*-

;; Copyright (C) 2024 TrustformeRS Team

;; Author: TrustformeRS Team
;; Version: {}
;; Package-Requires: ((emacs "26.1"))
;; Keywords: debugging, machine-learning, rust
;; URL: https://github.com/cool-japan/trustformers

;;; Commentary:

;; This package provides advanced debugging tools for TrustformeRS models including:
;; - Interactive debugging
;; - Tensor visualization
;; - Gradient flow analysis
;; - Loss landscape visualization

;;; Code:

(require 'json)
(require 'url)

(defgroup trustformers-debug nil
  "Advanced debugging tools for TrustformeRS models."
  :group 'debugging
  :prefix "trustformers-debug-")

(defcustom trustformers-debug-port 8899
  "Port for TrustformeRS debug server."
  :type 'integer
  :group 'trustformers-debug)

(defcustom trustformers-debug-auto-open nil
  "Whether to automatically open debug session."
  :type 'boolean
  :group 'trustformers-debug)

(defvar trustformers-debug-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "C-c C-d s") #'trustformers-debug-start)
    (define-key map (kbd "C-c C-d t") #'trustformers-debug-stop)
    (define-key map (kbd "C-c C-d i") #'trustformers-debug-step-into)
    (define-key map (kbd "C-c C-d o") #'trustformers-debug-step-over)
    (define-key map (kbd "C-c C-d v") #'trustformers-debug-show-tensor-visualization)
    (define-key map (kbd "C-c C-d g") #'trustformers-debug-show-gradient-flow)
    (define-key map (kbd "C-c C-d l") #'trustformers-debug-show-loss-landscape)
    map)
  "Keymap for TrustformeRS debug mode.")

(define-minor-mode trustformers-debug-mode
  "Minor mode for TrustformeRS debugging."
  :lighter " TF-Debug"
  :keymap trustformers-debug-mode-map)

(defun trustformers-debug-start ()
  "Start TrustformeRS debug session."
  (interactive)
  (message "Starting TrustformeRS debug session..."))

(defun trustformers-debug-stop ()
  "Stop TrustformeRS debug session."
  (interactive)
  (message "Stopping TrustformeRS debug session..."))

(defun trustformers-debug-step-into ()
  "Step into current function."
  (interactive)
  (message "Stepping into..."))

(defun trustformers-debug-step-over ()
  "Step over current line."
  (interactive)
  (message "Stepping over..."))

(defun trustformers-debug-show-tensor-visualization ()
  "Show tensor visualization."
  (interactive)
  (message "Showing tensor visualization..."))

(defun trustformers-debug-show-gradient-flow ()
  "Show gradient flow analysis."
  (interactive)
  (message "Showing gradient flow analysis..."))

(defun trustformers-debug-show-loss-landscape ()
  "Show loss landscape visualization."
  (interactive)
  (message "Showing loss landscape visualization..."))

(provide 'trustformers-debug)
;;; trustformers-debug.el ends here"#,
            self.version
        );

        Ok(manifest)
    }

    /// Get supported IDE types
    pub fn get_supported_ides(&self) -> &[SupportedIDE] {
        &self.supported_ides
    }

    /// Get plugin capabilities
    pub fn get_capabilities(&self) -> &IDECapabilities {
        &self.capabilities
    }

    /// Check if plugin is initialized
    pub fn is_initialized(&self) -> bool {
        self.debugger.is_some()
    }
}

/// IDE plugin manager
#[derive(Debug)]
pub struct IDEPluginManager {
    plugins: HashMap<Uuid, IDEPlugin>,
    active_sessions: HashMap<Uuid, Uuid>, // session_id -> plugin_id
}

impl IDEPluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            active_sessions: HashMap::new(),
        }
    }

    /// Register a new plugin
    pub fn register_plugin(&mut self, plugin: IDEPlugin) -> Uuid {
        let plugin_id = plugin.plugin_id;
        self.plugins.insert(plugin_id, plugin);
        plugin_id
    }

    /// Get a plugin by ID
    pub fn get_plugin(&self, plugin_id: &Uuid) -> Option<&IDEPlugin> {
        self.plugins.get(plugin_id)
    }

    /// Get a mutable plugin by ID
    pub fn get_plugin_mut(&mut self, plugin_id: &Uuid) -> Option<&mut IDEPlugin> {
        self.plugins.get_mut(plugin_id)
    }

    /// Start a debug session
    pub fn start_session(&mut self, plugin_id: Uuid) -> Uuid {
        let session_id = Uuid::new_v4();
        self.active_sessions.insert(session_id, plugin_id);
        session_id
    }

    /// Stop a debug session
    pub fn stop_session(&mut self, session_id: &Uuid) -> Option<Uuid> {
        self.active_sessions.remove(session_id)
    }

    /// Get all registered plugins
    pub fn get_all_plugins(&self) -> Vec<&IDEPlugin> {
        self.plugins.values().collect()
    }

    /// Get active sessions
    pub fn get_active_sessions(&self) -> Vec<Uuid> {
        self.active_sessions.keys().cloned().collect()
    }
}

impl Default for IDEPluginManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Jupyter widget types for debugging visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JupyterWidgetType {
    /// Interactive plot widget using plotly
    PlotWidget,
    /// Tensor visualization widget
    TensorWidget,
    /// Training metrics dashboard
    MetricsDashboard,
    /// Gradient flow visualization
    GradientFlowWidget,
    /// Model architecture diagram
    ArchitectureWidget,
    /// Performance profiler widget
    ProfilerWidget,
    /// Memory usage widget
    MemoryWidget,
    /// Debug console widget
    DebugConsole,
    /// Progress bar widget
    ProgressBar,
    /// Custom widget with specified type
    Custom(String),
}

/// Jupyter widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterWidgetConfig {
    pub widget_id: String,
    pub widget_type: JupyterWidgetType,
    pub title: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub resizable: bool,
    pub auto_update: bool,
    pub update_interval: Option<u32>, // milliseconds
    pub options: HashMap<String, String>,
}

/// Jupyter widget data for updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterWidgetData {
    pub plot_data: Option<crate::visualization::PlotData>,
    pub plot_3d_data: Option<crate::visualization::Plot3DData>,
    pub text_data: Option<String>,
    pub html_data: Option<String>,
    pub json_data: Option<String>,
    pub binary_data: Option<Vec<u8>>,
    pub metadata: HashMap<String, String>,
}

/// Jupyter widget manager for creating and managing debug widgets
#[derive(Debug)]
pub struct JupyterWidgetManager {
    widgets: HashMap<String, JupyterWidgetConfig>,
    kernel_connection: Option<String>,
    comm_targets: HashMap<String, String>,
}

impl JupyterWidgetManager {
    /// Create a new Jupyter widget manager
    pub fn new() -> Self {
        Self {
            widgets: HashMap::new(),
            kernel_connection: None,
            comm_targets: HashMap::new(),
        }
    }

    /// Connect to Jupyter kernel
    pub fn connect_to_kernel(&mut self, connection_info: String) -> Result<()> {
        self.kernel_connection = Some(connection_info);

        // Register comm targets for different widget types
        self.register_comm_target("trustformers_plot_widget".to_string());
        self.register_comm_target("trustformers_tensor_widget".to_string());
        self.register_comm_target("trustformers_metrics_widget".to_string());
        self.register_comm_target("trustformers_gradient_widget".to_string());
        self.register_comm_target("trustformers_debug_console".to_string());

        Ok(())
    }

    /// Register a comm target for widget communication
    fn register_comm_target(&mut self, target_name: String) {
        let comm_id = Uuid::new_v4().to_string();
        self.comm_targets.insert(target_name, comm_id);
    }

    /// Create a new Jupyter widget
    pub fn create_widget(&mut self, config: JupyterWidgetConfig) -> Result<String> {
        let widget_id = config.widget_id.clone();

        // Generate widget HTML/JavaScript based on type
        let widget_content = self.generate_widget_content(&config)?;

        // Store widget configuration
        self.widgets.insert(widget_id.clone(), config);

        // Send widget creation message to kernel
        self.send_widget_creation_message(&widget_id, &widget_content)?;

        Ok(widget_id)
    }

    /// Update an existing widget with new data
    pub fn update_widget(&mut self, widget_id: &str, data: JupyterWidgetData) -> Result<()> {
        let widget = self
            .widgets
            .get(widget_id)
            .ok_or_else(|| anyhow::anyhow!("Widget with id '{}' not found", widget_id))?;

        // Generate update message based on widget type and data
        let update_message = self.generate_update_message(widget, &data)?;

        // Send update message to kernel
        self.send_widget_update_message(widget_id, &update_message)?;

        Ok(())
    }

    /// Remove a widget
    pub fn remove_widget(&mut self, widget_id: &str) -> Result<()> {
        if self.widgets.remove(widget_id).is_some() {
            self.send_widget_removal_message(widget_id)?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Widget with id '{}' not found", widget_id))
        }
    }

    /// Create a plot widget for tensor visualization
    pub fn create_plot_widget(
        &mut self,
        title: &str,
        plot_data: &crate::visualization::PlotData,
    ) -> Result<String> {
        let widget_id = format!("plot_{}", Uuid::new_v4().to_string().replace("-", ""));

        let config = JupyterWidgetConfig {
            widget_id: widget_id.clone(),
            widget_type: JupyterWidgetType::PlotWidget,
            title: title.to_string(),
            width: Some(800),
            height: Some(600),
            resizable: true,
            auto_update: false,
            update_interval: None,
            options: HashMap::new(),
        };

        self.create_widget(config)?;

        let data = JupyterWidgetData {
            plot_data: Some(plot_data.clone()),
            plot_3d_data: None,
            text_data: None,
            html_data: None,
            json_data: None,
            binary_data: None,
            metadata: HashMap::new(),
        };

        self.update_widget(&widget_id, data)?;

        Ok(widget_id)
    }

    /// Create a metrics dashboard widget
    pub fn create_metrics_dashboard(&mut self, title: &str) -> Result<String> {
        let widget_id = format!("metrics_{}", Uuid::new_v4().to_string().replace("-", ""));

        let config = JupyterWidgetConfig {
            widget_id: widget_id.clone(),
            widget_type: JupyterWidgetType::MetricsDashboard,
            title: title.to_string(),
            width: Some(1000),
            height: Some(400),
            resizable: true,
            auto_update: true,
            update_interval: Some(1000), // 1 second
            options: HashMap::new(),
        };

        self.create_widget(config)?;

        Ok(widget_id)
    }

    /// Create a gradient flow widget
    pub fn create_gradient_flow_widget(&mut self, title: &str) -> Result<String> {
        let widget_id = format!("gradient_{}", Uuid::new_v4().to_string().replace("-", ""));

        let config = JupyterWidgetConfig {
            widget_id: widget_id.clone(),
            widget_type: JupyterWidgetType::GradientFlowWidget,
            title: title.to_string(),
            width: Some(800),
            height: Some(500),
            resizable: true,
            auto_update: true,
            update_interval: Some(500), // 0.5 seconds
            options: HashMap::new(),
        };

        self.create_widget(config)?;

        Ok(widget_id)
    }

    /// Create a debug console widget
    pub fn create_debug_console(&mut self, title: &str) -> Result<String> {
        let widget_id = format!("console_{}", Uuid::new_v4().to_string().replace("-", ""));

        let config = JupyterWidgetConfig {
            widget_id: widget_id.clone(),
            widget_type: JupyterWidgetType::DebugConsole,
            title: title.to_string(),
            width: Some(600),
            height: Some(300),
            resizable: true,
            auto_update: false,
            update_interval: None,
            options: HashMap::new(),
        };

        self.create_widget(config)?;

        Ok(widget_id)
    }

    /// Create a comprehensive debugging dashboard
    pub fn create_debug_dashboard(&mut self) -> Result<Vec<String>> {
        let mut widget_ids = Vec::new();

        // Create multiple widgets for comprehensive debugging
        widget_ids.push(self.create_metrics_dashboard("Training Metrics")?);
        widget_ids.push(self.create_gradient_flow_widget("Gradient Flow")?);
        widget_ids.push(self.create_debug_console("Debug Console")?);

        Ok(widget_ids)
    }

    /// Generate widget content based on widget type
    fn generate_widget_content(&self, config: &JupyterWidgetConfig) -> Result<String> {
        match &config.widget_type {
            JupyterWidgetType::PlotWidget => self.generate_plot_widget_content(config),
            JupyterWidgetType::TensorWidget => self.generate_tensor_widget_content(config),
            JupyterWidgetType::MetricsDashboard => self.generate_metrics_dashboard_content(config),
            JupyterWidgetType::GradientFlowWidget => self.generate_gradient_widget_content(config),
            JupyterWidgetType::ArchitectureWidget => {
                self.generate_architecture_widget_content(config)
            },
            JupyterWidgetType::ProfilerWidget => self.generate_profiler_widget_content(config),
            JupyterWidgetType::MemoryWidget => self.generate_memory_widget_content(config),
            JupyterWidgetType::DebugConsole => self.generate_debug_console_content(config),
            JupyterWidgetType::ProgressBar => self.generate_progress_bar_content(config),
            JupyterWidgetType::Custom(widget_type) => {
                self.generate_custom_widget_content(config, widget_type)
            },
        }
    }

    /// Generate plot widget HTML/JavaScript
    fn generate_plot_widget_content(&self, config: &JupyterWidgetConfig) -> Result<String> {
        let content = format!(
            r#"
<div id="{widget_id}" style="width: {width}px; height: {height}px; border: 1px solid #ccc; border-radius: 4px;">
    <div id="{widget_id}_plot" style="width: 100%; height: 100%;"></div>
</div>

<script>
requirejs(['https://cdn.plot.ly/plotly-2.26.0.min.js'], function(Plotly) {{
    window.trustformers_widgets = window.trustformers_widgets || {{}};
    window.trustformers_widgets['{widget_id}'] = {{
        element: document.getElementById('{widget_id}_plot'),
        update: function(data) {{
            if (data.plot_data) {{
                const trace = {{
                    x: data.plot_data.x_values,
                    y: data.plot_data.y_values,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Data',
                    line: {{ color: '#007bff', width: 2 }},
                    marker: {{ size: 6 }}
                }};

                const layout = {{
                    title: data.plot_data.title,
                    xaxis: {{ title: data.plot_data.x_label }},
                    yaxis: {{ title: data.plot_data.y_label }},
                    margin: {{ t: 50, l: 60, r: 30, b: 60 }},
                    showlegend: false
                }};

                Plotly.newPlot(this.element, [trace], layout, {{ responsive: true }});
            }}
        }}
    }};
}});
</script>
"#,
            widget_id = config.widget_id,
            width = config.width.unwrap_or(800),
            height = config.height.unwrap_or(600)
        );

        Ok(content)
    }

    /// Generate tensor widget content
    fn generate_tensor_widget_content(&self, config: &JupyterWidgetConfig) -> Result<String> {
        let content = format!(
            r#"
<div id="{widget_id}" class="trustformers-tensor-widget" style="width: {width}px; height: {height}px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f9f9f9;">
    <h3>{title}</h3>
    <div id="{widget_id}_content" style="height: calc(100% - 40px); overflow-y: auto;">
        <div id="{widget_id}_stats"></div>
        <div id="{widget_id}_visualization"></div>
    </div>
</div>

<script>
window.trustformers_widgets = window.trustformers_widgets || {{}};
window.trustformers_widgets['{widget_id}'] = {{
    update: function(data) {{
        const statsDiv = document.getElementById('{widget_id}_stats');
        const vizDiv = document.getElementById('{widget_id}_visualization');

        if (data.text_data) {{
            statsDiv.innerHTML = '<pre>' + data.text_data + '</pre>';
        }}

        if (data.html_data) {{
            vizDiv.innerHTML = data.html_data;
        }}
    }}
}};
</script>
"#,
            widget_id = config.widget_id,
            title = config.title,
            width = config.width.unwrap_or(600),
            height = config.height.unwrap_or(400)
        );

        Ok(content)
    }

    /// Generate metrics dashboard content
    fn generate_metrics_dashboard_content(&self, config: &JupyterWidgetConfig) -> Result<String> {
        let content = format!(
            r#"
<div id="{widget_id}" class="trustformers-metrics-dashboard" style="width: {width}px; height: {height}px; border: 1px solid #ccc; border-radius: 4px; background: white;">
    <div style="background: #f8f9fa; padding: 10px; border-bottom: 1px solid #dee2e6;">
        <h4 style="margin: 0;">{title}</h4>
    </div>
    <div id="{widget_id}_metrics" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; padding: 10px; height: calc(100% - 60px); overflow-y: auto;">
        <!-- Metrics will be populated here -->
    </div>
</div>

<script>
window.trustformers_widgets = window.trustformers_widgets || {{}};
window.trustformers_widgets['{widget_id}'] = {{
    update: function(data) {{
        const metricsDiv = document.getElementById('{widget_id}_metrics');

        if (data.json_data) {{
            const metrics = JSON.parse(data.json_data);
            let html = '';

            for (const [key, value] of Object.entries(metrics)) {{
                html += `
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; text-align: center;">
                        <div style="font-size: 14px; color: #6c757d; margin-bottom: 4px;">${{key}}</div>
                        <div style="font-size: 20px; font-weight: bold; color: #495057;">${{value}}</div>
                    </div>
                `;
            }}

            metricsDiv.innerHTML = html;
        }}
    }}
}};

// Auto-update if enabled
{auto_update_script}
</script>
"#,
            widget_id = config.widget_id,
            title = config.title,
            width = config.width.unwrap_or(1000),
            height = config.height.unwrap_or(400),
            auto_update_script = if config.auto_update {
                format!(
                    r#"
setInterval(function() {{
    // Request updated metrics from kernel
    if (window.trustformers_kernel_comm) {{
        window.trustformers_kernel_comm.send({{
            widget_id: '{widget_id}',
            action: 'request_update'
        }});
    }}
}}, {});
"#,
                    config.update_interval.unwrap_or(1000),
                    widget_id = config.widget_id
                )
            } else {
                "".to_string()
            }
        );

        Ok(content)
    }

    /// Generate gradient flow widget content
    fn generate_gradient_widget_content(&self, config: &JupyterWidgetConfig) -> Result<String> {
        // Similar implementation to metrics dashboard but specialized for gradient visualization
        let content = format!(
            r#"
<div id="{widget_id}" class="trustformers-gradient-widget" style="width: {width}px; height: {height}px; border: 1px solid #ccc; border-radius: 4px;">
    <div style="background: #f8f9fa; padding: 10px; border-bottom: 1px solid #dee2e6;">
        <h4 style="margin: 0;">{title}</h4>
    </div>
    <div id="{widget_id}_gradient_plot" style="width: 100%; height: calc(100% - 60px);"></div>
</div>

<script>
requirejs(['https://cdn.plot.ly/plotly-2.26.0.min.js'], function(Plotly) {{
    window.trustformers_widgets = window.trustformers_widgets || {{}};
    window.trustformers_widgets['{widget_id}'] = {{
        element: document.getElementById('{widget_id}_gradient_plot'),
        update: function(data) {{
            if (data.plot_data) {{
                const trace = {{
                    x: data.plot_data.x_values,
                    y: data.plot_data.y_values,
                    type: 'bar',
                    marker: {{ color: '#dc3545' }},
                    name: 'Gradient Norm'
                }};

                const layout = {{
                    title: 'Gradient Flow by Layer',
                    xaxis: {{ title: 'Layer' }},
                    yaxis: {{ title: 'Gradient Norm', type: 'log' }},
                    margin: {{ t: 50, l: 60, r: 30, b: 60 }}
                }};

                Plotly.newPlot(this.element, [trace], layout, {{ responsive: true }});
            }}
        }}
    }};
}});
</script>
"#,
            widget_id = config.widget_id,
            title = config.title,
            width = config.width.unwrap_or(800),
            height = config.height.unwrap_or(500)
        );

        Ok(content)
    }

    /// Generate architecture widget content
    fn generate_architecture_widget_content(&self, config: &JupyterWidgetConfig) -> Result<String> {
        // Placeholder implementation for architecture visualization
        Ok(format!(
            "<div id='{}'>Architecture Widget - {}</div>",
            config.widget_id, config.title
        ))
    }

    /// Generate profiler widget content
    fn generate_profiler_widget_content(&self, config: &JupyterWidgetConfig) -> Result<String> {
        // Placeholder implementation for profiler visualization
        Ok(format!(
            "<div id='{}'>Profiler Widget - {}</div>",
            config.widget_id, config.title
        ))
    }

    /// Generate memory widget content
    fn generate_memory_widget_content(&self, config: &JupyterWidgetConfig) -> Result<String> {
        // Placeholder implementation for memory usage visualization
        Ok(format!(
            "<div id='{}'>Memory Widget - {}</div>",
            config.widget_id, config.title
        ))
    }

    /// Generate debug console content
    fn generate_debug_console_content(&self, config: &JupyterWidgetConfig) -> Result<String> {
        let content = format!(
            r#"
<div id="{widget_id}" class="trustformers-debug-console" style="width: {width}px; height: {height}px; border: 1px solid #ccc; border-radius: 4px; background: #1e1e1e; color: #d4d4d4; font-family: 'Courier New', monospace;">
    <div style="background: #2d2d30; padding: 8px; border-bottom: 1px solid #3e3e42; color: #cccccc; font-size: 12px;">
        {title}
    </div>
    <div id="{widget_id}_output" style="height: calc(100% - 90px); overflow-y: auto; padding: 10px; font-size: 12px; line-height: 1.4;">
        <!-- Console output will appear here -->
    </div>
    <div style="border-top: 1px solid #3e3e42; padding: 5px;">
        <input id="{widget_id}_input" type="text" placeholder="Enter debug command..."
               style="width: 100%; background: #3c3c3c; border: 1px solid #5a5a5a; color: #d4d4d4; padding: 5px; border-radius: 2px; font-family: inherit; font-size: 12px;">
    </div>
</div>

<script>
window.trustformers_widgets = window.trustformers_widgets || {{}};
window.trustformers_widgets['{widget_id}'] = {{
    outputDiv: document.getElementById('{widget_id}_output'),
    inputEl: document.getElementById('{widget_id}_input'),

    init: function() {{
        this.inputEl.addEventListener('keypress', (e) => {{
            if (e.key === 'Enter') {{
                const command = this.inputEl.value;
                this.addOutput('> ' + command, 'input');
                this.inputEl.value = '';

                // Send command to kernel
                if (window.trustformers_kernel_comm) {{
                    window.trustformers_kernel_comm.send({{
                        widget_id: '{widget_id}',
                        action: 'execute_command',
                        command: command
                    }});
                }}
            }}
        }});
    }},

    addOutput: function(text, type = 'output') {{
        const div = document.createElement('div');
        div.style.marginBottom = '2px';
        div.style.color = type === 'input' ? '#569cd6' :
                          type === 'error' ? '#f14c4c' : '#d4d4d4';
        div.textContent = text;
        this.outputDiv.appendChild(div);
        this.outputDiv.scrollTop = this.outputDiv.scrollHeight;
    }},

    update: function(data) {{
        if (data.text_data) {{
            this.addOutput(data.text_data, data.metadata.type || 'output');
        }}
    }}
}};

window.trustformers_widgets['{widget_id}'].init();
</script>
"#,
            widget_id = config.widget_id,
            title = config.title,
            width = config.width.unwrap_or(600),
            height = config.height.unwrap_or(300)
        );

        Ok(content)
    }

    /// Generate progress bar content
    fn generate_progress_bar_content(&self, config: &JupyterWidgetConfig) -> Result<String> {
        // Placeholder implementation for progress bar
        Ok(format!(
            "<div id='{}'>Progress Bar - {}</div>",
            config.widget_id, config.title
        ))
    }

    /// Generate custom widget content
    fn generate_custom_widget_content(
        &self,
        config: &JupyterWidgetConfig,
        widget_type: &str,
    ) -> Result<String> {
        // Placeholder implementation for custom widgets
        Ok(format!(
            "<div id='{}'>Custom Widget ({}) - {}</div>",
            config.widget_id, widget_type, config.title
        ))
    }

    /// Generate update message for widget
    fn generate_update_message(
        &self,
        config: &JupyterWidgetConfig,
        data: &JupyterWidgetData,
    ) -> Result<String> {
        let message = serde_json::json!({
            "widget_id": config.widget_id,
            "data": {
                "plot_data": data.plot_data,
                "plot_3d_data": data.plot_3d_data,
                "text_data": data.text_data,
                "html_data": data.html_data,
                "json_data": data.json_data,
                "metadata": data.metadata
            }
        });

        Ok(message.to_string())
    }

    /// Send widget creation message to Jupyter kernel
    fn send_widget_creation_message(&self, widget_id: &str, content: &str) -> Result<()> {
        // In a real implementation, this would send the message via Jupyter's comm protocol
        tracing::info!("Creating Jupyter widget: {}", widget_id);
        tracing::debug!("Widget content: {}", content);
        Ok(())
    }

    /// Send widget update message to Jupyter kernel
    fn send_widget_update_message(&self, widget_id: &str, message: &str) -> Result<()> {
        // In a real implementation, this would send the message via Jupyter's comm protocol
        tracing::debug!("Updating Jupyter widget {}: {}", widget_id, message);
        Ok(())
    }

    /// Send widget removal message to Jupyter kernel
    fn send_widget_removal_message(&self, widget_id: &str) -> Result<()> {
        // In a real implementation, this would send the message via Jupyter's comm protocol
        tracing::info!("Removing Jupyter widget: {}", widget_id);
        Ok(())
    }

    /// Get all active widgets
    pub fn get_active_widgets(&self) -> Vec<&JupyterWidgetConfig> {
        self.widgets.values().collect()
    }

    /// Check if kernel is connected
    pub fn is_kernel_connected(&self) -> bool {
        self.kernel_connection.is_some()
    }

    /// Get widget by ID
    pub fn get_widget(&self, widget_id: &str) -> Option<&JupyterWidgetConfig> {
        self.widgets.get(widget_id)
    }
}

impl Default for JupyterWidgetManager {
    fn default() -> Self {
        Self::new()
    }
}
