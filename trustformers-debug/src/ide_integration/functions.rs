//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::DebugConfig;
    use std::path::PathBuf;
    use uuid::Uuid;
    #[test]
    fn test_supported_ide_variants() {
        let ides = [
            SupportedIDE::VSCode,
            SupportedIDE::IntelliJ,
            SupportedIDE::Vim,
            SupportedIDE::Emacs,
            SupportedIDE::Sublime,
            SupportedIDE::Atom,
            SupportedIDE::Jupyter,
            SupportedIDE::JupyterLab,
            SupportedIDE::Custom("custom_ide".to_string()),
        ];
        assert_eq!(ides.len(), 9);
    }
    #[test]
    fn test_ide_capabilities_creation() {
        let caps = IDECapabilities {
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
            jupyter_widgets: false,
            interactive_plots: false,
            notebook_integration: false,
            kernel_communication: false,
        };
        assert!(caps.syntax_highlighting);
        assert!(!caps.jupyter_widgets);
    }
    #[test]
    fn test_ide_message_set_breakpoint() {
        let msg = IDEMessage::SetBreakpoint {
            file: PathBuf::from("/src/main.rs"),
            line: 42,
            condition: Some("x > 10".to_string()),
        };
        match msg {
            IDEMessage::SetBreakpoint {
                file,
                line,
                condition,
            } => {
                assert_eq!(file, PathBuf::from("/src/main.rs"));
                assert_eq!(line, 42);
                assert!(condition.is_some());
            },
            _ => panic!("Expected SetBreakpoint"),
        }
    }
    #[test]
    fn test_ide_message_remove_breakpoint() {
        let msg = IDEMessage::RemoveBreakpoint {
            file: PathBuf::from("/src/lib.rs"),
            line: 10,
        };
        match msg {
            IDEMessage::RemoveBreakpoint { file, line } => {
                assert_eq!(file, PathBuf::from("/src/lib.rs"));
                assert_eq!(line, 10);
            },
            _ => panic!("Expected RemoveBreakpoint"),
        }
    }
    #[test]
    fn test_ide_message_control_variants() {
        let messages = [
            IDEMessage::StepInto,
            IDEMessage::StepOver,
            IDEMessage::StepOut,
            IDEMessage::Continue,
            IDEMessage::Pause,
        ];
        assert_eq!(messages.len(), 5);
    }
    #[test]
    fn test_ide_message_debug_session() {
        let session_id = Uuid::new_v4();
        let start_msg = IDEMessage::StartDebugSession { session_id };
        let stop_msg = IDEMessage::StopDebugSession { session_id };
        match start_msg {
            IDEMessage::StartDebugSession { session_id: s } => assert_eq!(s, session_id),
            _ => panic!("Expected StartDebugSession"),
        }
        match stop_msg {
            IDEMessage::StopDebugSession { session_id: s } => assert_eq!(s, session_id),
            _ => panic!("Expected StopDebugSession"),
        }
    }
    #[test]
    fn test_ide_message_inspect_variable() {
        let msg = IDEMessage::InspectVariable {
            variable_name: "tensor_weights".to_string(),
        };
        match msg {
            IDEMessage::InspectVariable { variable_name } => {
                assert_eq!(variable_name, "tensor_weights");
            },
            _ => panic!("Expected InspectVariable"),
        }
    }
    #[test]
    fn test_ide_message_evaluate_expression() {
        let msg = IDEMessage::EvaluateExpression {
            expression: "model.parameters().count()".to_string(),
        };
        match msg {
            IDEMessage::EvaluateExpression { expression } => {
                assert!(expression.contains("parameters"));
            },
            _ => panic!("Expected EvaluateExpression"),
        }
    }
    #[test]
    fn test_jupyter_widget_manager_new() {
        let manager = JupyterWidgetManager::new();
        assert!(manager.get_active_widgets().is_empty());
        assert!(!manager.is_kernel_connected());
    }
    #[test]
    fn test_jupyter_widget_manager_default() {
        let manager = JupyterWidgetManager::default();
        assert!(!manager.is_kernel_connected());
    }
    #[test]
    fn test_jupyter_widget_manager_get_missing_widget() {
        let manager = JupyterWidgetManager::new();
        assert!(manager.get_widget("nonexistent").is_none());
    }
    #[test]
    fn test_ide_plugin_creation() {
        let plugin = IDEPlugin {
            plugin_id: Uuid::new_v4(),
            name: "TrustformeRS Debug".to_string(),
            version: "1.0.0".to_string(),
            supported_ides: vec![SupportedIDE::VSCode, SupportedIDE::IntelliJ],
            capabilities: IDECapabilities {
                syntax_highlighting: true,
                code_completion: true,
                inline_debugging: true,
                tensor_visualization: true,
                real_time_metrics: false,
                breakpoint_management: true,
                call_stack_navigation: true,
                variable_inspection: true,
                performance_profiling: false,
                error_annotations: true,
                jupyter_widgets: false,
                interactive_plots: false,
                notebook_integration: false,
                kernel_communication: false,
            },
            debugger: None,
            config: DebugConfig::default(),
        };
        assert_eq!(plugin.name, "TrustformeRS Debug");
        assert_eq!(plugin.supported_ides.len(), 2);
        assert!(plugin.capabilities.syntax_highlighting);
        assert!(plugin.debugger.is_none());
    }
    #[test]
    fn test_ide_capabilities_all_enabled() {
        let caps = IDECapabilities {
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
        };
        assert!(caps.jupyter_widgets);
        assert!(caps.kernel_communication);
    }
    #[test]
    fn test_ide_message_toggle_breakpoint() {
        let msg = IDEMessage::ToggleBreakpoint {
            file: PathBuf::from("/test.rs"),
            line: 5,
        };
        match msg {
            IDEMessage::ToggleBreakpoint { file, line } => {
                assert_eq!(line, 5);
                assert_eq!(file.to_str().expect("path should be valid"), "/test.rs");
            },
            _ => panic!("Expected ToggleBreakpoint"),
        }
    }
    #[test]
    fn test_ide_message_show_tensor_visualization() {
        let msg = IDEMessage::ShowTensorVisualization {
            tensor_name: "attention_weights".to_string(),
        };
        match msg {
            IDEMessage::ShowTensorVisualization { tensor_name } => {
                assert_eq!(tensor_name, "attention_weights");
            },
            _ => panic!("Expected ShowTensorVisualization"),
        }
    }
    #[test]
    fn test_custom_ide_variant() {
        let ide = SupportedIDE::Custom("neovim".to_string());
        match ide {
            SupportedIDE::Custom(name) => assert_eq!(name, "neovim"),
            _ => panic!("Expected Custom variant"),
        }
    }
    #[test]
    fn test_ide_capabilities_partial() {
        let caps = IDECapabilities {
            syntax_highlighting: true,
            code_completion: false,
            inline_debugging: false,
            tensor_visualization: false,
            real_time_metrics: false,
            breakpoint_management: false,
            call_stack_navigation: false,
            variable_inspection: false,
            performance_profiling: false,
            error_annotations: false,
            jupyter_widgets: false,
            interactive_plots: false,
            notebook_integration: false,
            kernel_communication: false,
        };
        assert!(caps.syntax_highlighting);
        assert!(!caps.code_completion);
    }
    #[test]
    fn test_ide_message_show_gradient_flow() {
        let msg = IDEMessage::ShowGradientFlow {
            layer_name: "encoder.layer.0".to_string(),
        };
        match msg {
            IDEMessage::ShowGradientFlow { layer_name } => {
                assert_eq!(layer_name, "encoder.layer.0");
            },
            _ => panic!("Expected ShowGradientFlow"),
        }
    }
    #[test]
    fn test_jupyter_widget_manager_kernel_not_connected() {
        let manager = JupyterWidgetManager::new();
        assert!(!manager.is_kernel_connected());
        assert!(manager.get_active_widgets().is_empty());
    }
    #[test]
    fn test_ide_plugin_no_debugger() {
        let plugin = IDEPlugin {
            plugin_id: Uuid::new_v4(),
            name: "test".to_string(),
            version: "0.1.0".to_string(),
            supported_ides: vec![SupportedIDE::VSCode],
            capabilities: IDECapabilities {
                syntax_highlighting: false,
                code_completion: false,
                inline_debugging: false,
                tensor_visualization: false,
                real_time_metrics: false,
                breakpoint_management: false,
                call_stack_navigation: false,
                variable_inspection: false,
                performance_profiling: false,
                error_annotations: false,
                jupyter_widgets: false,
                interactive_plots: false,
                notebook_integration: false,
                kernel_communication: false,
            },
            debugger: None,
            config: DebugConfig::default(),
        };
        assert!(plugin.debugger.is_none());
        assert_eq!(plugin.version, "0.1.0");
    }
    #[test]
    fn test_supported_ide_clone() {
        let ide = SupportedIDE::VSCode;
        let cloned = ide.clone();
        assert!(matches!(cloned, SupportedIDE::VSCode));
    }
    #[test]
    fn test_ide_capabilities_clone() {
        let caps = IDECapabilities {
            syntax_highlighting: true,
            code_completion: true,
            inline_debugging: false,
            tensor_visualization: false,
            real_time_metrics: false,
            breakpoint_management: false,
            call_stack_navigation: false,
            variable_inspection: false,
            performance_profiling: false,
            error_annotations: false,
            jupyter_widgets: false,
            interactive_plots: false,
            notebook_integration: false,
            kernel_communication: false,
        };
        let cloned = caps.clone();
        assert!(cloned.syntax_highlighting);
        assert!(cloned.code_completion);
    }
    #[test]
    fn test_ide_message_set_breakpoint_no_condition() {
        let msg = IDEMessage::SetBreakpoint {
            file: PathBuf::from("/test.rs"),
            line: 1,
            condition: None,
        };
        match msg {
            IDEMessage::SetBreakpoint { condition, .. } => {
                assert!(condition.is_none());
            },
            _ => panic!("Expected SetBreakpoint"),
        }
    }
    #[test]
    fn test_multiple_supported_ides() {
        let ides = [
            SupportedIDE::VSCode,
            SupportedIDE::IntelliJ,
            SupportedIDE::Jupyter,
            SupportedIDE::JupyterLab,
        ];
        assert_eq!(ides.len(), 4);
        assert!(matches!(ides[0], SupportedIDE::VSCode));
        assert!(matches!(ides[2], SupportedIDE::Jupyter));
    }
    #[test]
    fn test_ide_plugin_multiple_ides() {
        let plugin = IDEPlugin {
            plugin_id: Uuid::new_v4(),
            name: "multi".to_string(),
            version: "2.0.0".to_string(),
            supported_ides: vec![
                SupportedIDE::VSCode,
                SupportedIDE::IntelliJ,
                SupportedIDE::Vim,
                SupportedIDE::Emacs,
            ],
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
                jupyter_widgets: false,
                interactive_plots: false,
                notebook_integration: false,
                kernel_communication: false,
            },
            debugger: None,
            config: DebugConfig::default(),
        };
        assert_eq!(plugin.supported_ides.len(), 4);
    }
}
