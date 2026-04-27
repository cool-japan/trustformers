#[cfg(test)]
mod tests {
    use crate::ide_integration::*;
    use crate::DebugConfig;
    use uuid::Uuid;
    use std::collections::HashMap;
    use std::path::PathBuf;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn _next_f64(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    // Test 1: SupportedIDE variants
    #[test]
    fn test_supported_ide_variants() {
        let ides = vec![
            SupportedIDE::VSCode,
            SupportedIDE::IntelliJ,
            SupportedIDE::Vim,
            SupportedIDE::Emacs,
            SupportedIDE::Sublime,
            SupportedIDE::Atom,
            SupportedIDE::Jupyter,
            SupportedIDE::JupyterLab,
            SupportedIDE::Custom("MyIDE".to_string()),
        ];
        assert_eq!(ides.len(), 9);
        if let SupportedIDE::Custom(ref name) = ides[8] {
            assert_eq!(name, "MyIDE");
        }
    }

    // Test 2: IDECapabilities construction
    #[test]
    fn test_ide_capabilities_construction() {
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

    // Test 3: IDEPluginConfig default
    #[test]
    fn test_ide_plugin_config_default() {
        let config = IDEPluginConfig::default();
        assert!(config.enable_syntax_highlighting);
        assert!(config.enable_code_completion);
        assert!(config.enable_inline_debugging);
        assert!(!config.auto_open_debugger);
        assert_eq!(config.visualization_format, "png");
        assert_eq!(config.debug_port, 8899);
        assert_eq!(config.max_variable_display_length, 1000);
        assert_eq!(config.refresh_interval_ms, 1000);
        assert_eq!(config.log_level, "info");
    }

    // Test 4: DebugLocation construction
    #[test]
    fn test_debug_location_construction() {
        let location = DebugLocation {
            file: PathBuf::from("/src/model.rs"),
            line: 42,
            column: 10,
            function: "forward".to_string(),
            module: "model".to_string(),
        };
        assert_eq!(location.line, 42);
        assert_eq!(location.column, 10);
        assert_eq!(location.function, "forward");
    }

    // Test 5: SourceLocation construction
    #[test]
    fn test_source_location_construction() {
        let location = SourceLocation {
            file: PathBuf::from("/src/layers/attention.rs"),
            line: 100,
            column: 5,
            context: "pub fn attention_forward(...)".to_string(),
        };
        assert_eq!(location.line, 100);
        assert!(!location.context.is_empty());
    }

    // Test 6: CallStackFrame construction
    #[test]
    fn test_call_stack_frame_construction() {
        let mut variables = HashMap::new();
        variables.insert("batch_size".to_string(), "32".to_string());
        variables.insert("hidden_dim".to_string(), "768".to_string());

        let frame = CallStackFrame {
            function: "forward".to_string(),
            file: PathBuf::from("/src/model.rs"),
            line: 42,
            variables,
        };
        assert_eq!(frame.function, "forward");
        assert_eq!(frame.variables.len(), 2);
    }

    // Test 7: IDEMessage StartDebugSession
    #[test]
    fn test_ide_message_start_debug_session() {
        let session_id = Uuid::new_v4();
        let msg = IDEMessage::StartDebugSession { session_id };
        if let IDEMessage::StartDebugSession { session_id: sid } = msg {
            assert_eq!(sid, session_id);
        }
    }

    // Test 8: IDEMessage SetBreakpoint
    #[test]
    fn test_ide_message_set_breakpoint() {
        let msg = IDEMessage::SetBreakpoint {
            file: PathBuf::from("/src/model.rs"),
            line: 42,
            condition: Some("batch_size > 64".to_string()),
        };
        if let IDEMessage::SetBreakpoint {
            file,
            line,
            condition,
        } = msg
        {
            assert_eq!(line, 42);
            assert!(condition.is_some());
            assert_eq!(file, PathBuf::from("/src/model.rs"));
        }
    }

    // Test 9: IDEMessage debug control variants
    #[test]
    fn test_ide_message_debug_control() {
        let messages: Vec<IDEMessage> = vec![
            IDEMessage::StepInto,
            IDEMessage::StepOver,
            IDEMessage::StepOut,
            IDEMessage::Continue,
            IDEMessage::Pause,
        ];
        assert_eq!(messages.len(), 5);
    }

    // Test 10: IDEMessage InspectVariable
    #[test]
    fn test_ide_message_inspect_variable() {
        let msg = IDEMessage::InspectVariable {
            variable_name: "hidden_states".to_string(),
        };
        if let IDEMessage::InspectVariable { variable_name } = msg {
            assert_eq!(variable_name, "hidden_states");
        }
    }

    // Test 11: IDEResponse variants
    #[test]
    fn test_ide_response_variants() {
        let session_id = Uuid::new_v4();
        let responses: Vec<IDEResponse> = vec![
            IDEResponse::SessionStarted { session_id },
            IDEResponse::SessionStopped { session_id },
            IDEResponse::ExecutionResumed,
            IDEResponse::Success {
                message: "OK".to_string(),
            },
            IDEResponse::Error {
                error: "Failed".to_string(),
            },
        ];
        assert_eq!(responses.len(), 5);
    }

    // Test 12: IDEResponse VariableValue
    #[test]
    fn test_ide_response_variable_value() {
        let resp = IDEResponse::VariableValue {
            name: "tensor_x".to_string(),
            value: "[1.0, 2.0, 3.0]".to_string(),
            type_name: "Tensor<f32>".to_string(),
        };
        if let IDEResponse::VariableValue {
            name,
            value,
            type_name,
        } = resp
        {
            assert_eq!(name, "tensor_x");
            assert!(!value.is_empty());
            assert!(type_name.contains("Tensor"));
        }
    }

    // Test 13: IDEResponse BreakpointHit
    #[test]
    fn test_ide_response_breakpoint_hit() {
        let resp = IDEResponse::BreakpointHit {
            file: PathBuf::from("/src/layers/norm.rs"),
            line: 55,
        };
        if let IDEResponse::BreakpointHit { file, line } = resp {
            assert_eq!(line, 55);
            assert_eq!(file, PathBuf::from("/src/layers/norm.rs"));
        }
    }

    // Test 14: IDEResponse ExecutionPaused
    #[test]
    fn test_ide_response_execution_paused() {
        let location = DebugLocation {
            file: PathBuf::from("/src/model.rs"),
            line: 100,
            column: 1,
            function: "backward".to_string(),
            module: "training".to_string(),
        };
        let resp = IDEResponse::ExecutionPaused { location };
        if let IDEResponse::ExecutionPaused { location: loc } = resp {
            assert_eq!(loc.function, "backward");
        }
    }

    // Test 15: IDEMessage Jupyter messages
    #[test]
    fn test_ide_message_jupyter() {
        let mut options = HashMap::new();
        options.insert("width".to_string(), "800".to_string());
        options.insert("height".to_string(), "600".to_string());

        let msg = IDEMessage::CreateWidget {
            widget_type: "plot".to_string(),
            widget_id: "widget_1".to_string(),
            options,
        };
        if let IDEMessage::CreateWidget {
            widget_type,
            widget_id,
            options: opts,
        } = msg
        {
            assert_eq!(widget_type, "plot");
            assert_eq!(widget_id, "widget_1");
            assert_eq!(opts.len(), 2);
        }
    }

    // Test 16: IDEMessage visualization requests
    #[test]
    fn test_ide_message_visualizations() {
        let messages: Vec<IDEMessage> = vec![
            IDEMessage::ShowTensorVisualization {
                tensor_name: "attention_weights".to_string(),
            },
            IDEMessage::ShowGradientFlow {
                layer_name: "layer_0".to_string(),
            },
            IDEMessage::ShowLossLandscape,
            IDEMessage::ShowPerformanceMetrics,
        ];
        assert_eq!(messages.len(), 4);
    }

    // Test 17: IDEPlugin creation
    #[test]
    fn test_ide_plugin_creation() {
        let config = DebugConfig::default();
        let plugin = IDEPlugin::new("TestPlugin".to_string(), "1.0.0".to_string(), config);
        assert_eq!(plugin.name, "TestPlugin");
        assert_eq!(plugin.version, "1.0.0");
        assert!(plugin.debugger.is_none());
        assert_eq!(plugin.supported_ides.len(), 2);
        assert!(plugin.capabilities.syntax_highlighting);
    }

    // Test 18: IDEPluginConfig with custom values
    #[test]
    fn test_ide_plugin_config_custom() {
        let config = IDEPluginConfig {
            enable_syntax_highlighting: false,
            enable_code_completion: false,
            enable_inline_debugging: true,
            enable_tensor_visualization: true,
            enable_real_time_metrics: false,
            auto_open_debugger: true,
            visualization_format: "svg".to_string(),
            debug_port: 9999,
            max_variable_display_length: 500,
            refresh_interval_ms: 2000,
            workspace_root: PathBuf::from("/home/user/project"),
            log_level: "debug".to_string(),
        };
        assert!(!config.enable_syntax_highlighting);
        assert!(config.auto_open_debugger);
        assert_eq!(config.debug_port, 9999);
        assert_eq!(config.visualization_format, "svg");
    }

    // Test 19: IDEResponse CallStackData
    #[test]
    fn test_ide_response_call_stack() {
        let frames = vec![
            CallStackFrame {
                function: "forward".to_string(),
                file: PathBuf::from("/src/model.rs"),
                line: 42,
                variables: HashMap::new(),
            },
            CallStackFrame {
                function: "attention".to_string(),
                file: PathBuf::from("/src/attention.rs"),
                line: 100,
                variables: HashMap::new(),
            },
        ];
        let resp = IDEResponse::CallStackData { frames };
        if let IDEResponse::CallStackData { frames: f } = resp {
            assert_eq!(f.len(), 2);
        }
    }

    // Test 20: IDEMessage code navigation
    #[test]
    fn test_ide_message_navigation() {
        let msgs: Vec<IDEMessage> = vec![
            IDEMessage::GotoDefinition {
                symbol: "LayerNorm".to_string(),
            },
            IDEMessage::FindReferences {
                symbol: "forward".to_string(),
            },
            IDEMessage::ShowCallStack,
        ];
        assert_eq!(msgs.len(), 3);
    }

    // Test 21: IDEResponse DefinitionLocation
    #[test]
    fn test_ide_response_definition_location() {
        let resp = IDEResponse::DefinitionLocation {
            file: PathBuf::from("/src/layers/mod.rs"),
            line: 15,
            column: 8,
        };
        if let IDEResponse::DefinitionLocation { file, line, column } = resp {
            assert_eq!(line, 15);
            assert_eq!(column, 8);
            assert!(file.to_string_lossy().contains("layers"));
        }
    }

    // Test 22: IDEMessage error handling messages
    #[test]
    fn test_ide_message_error_handling() {
        let msg = IDEMessage::ShowError {
            message: "Shape mismatch in attention layer".to_string(),
            file: Some(PathBuf::from("/src/attention.rs")),
            line: Some(42),
        };
        if let IDEMessage::ShowError {
            message,
            file,
            line,
        } = msg
        {
            assert!(!message.is_empty());
            assert!(file.is_some());
            assert_eq!(line, Some(42));
        }
    }

    // Test 23: IDEMessage status updates
    #[test]
    fn test_ide_message_status_updates() {
        let msg = IDEMessage::UpdateProgress {
            progress: 0.75,
            message: "Training epoch 3/4".to_string(),
        };
        if let IDEMessage::UpdateProgress { progress, message } = msg {
            assert!((progress - 0.75).abs() < f64::EPSILON);
            assert!(!message.is_empty());
        }
    }

    // Test 24: Multiple CallStackFrames with LCG
    #[test]
    fn test_call_stack_frames_with_lcg() {
        let mut lcg = Lcg::new(42);
        let function_names = ["forward", "attention", "linear", "norm", "dropout"];
        let frames: Vec<CallStackFrame> = function_names
            .iter()
            .map(|name| {
                let line = (lcg.next() % 1000) as u32;
                CallStackFrame {
                    function: name.to_string(),
                    file: PathBuf::from(format!("/src/{}.rs", name)),
                    line,
                    variables: HashMap::new(),
                }
            })
            .collect();
        assert_eq!(frames.len(), 5);
        for frame in &frames {
            assert!(frame.line < 1000);
        }
    }

    // Test 25: IDEResponse ReferenceLocations
    #[test]
    fn test_ide_response_reference_locations() {
        let locations = vec![
            SourceLocation {
                file: PathBuf::from("/src/model.rs"),
                line: 10,
                column: 5,
                context: "use crate::layers::LayerNorm".to_string(),
            },
            SourceLocation {
                file: PathBuf::from("/src/training.rs"),
                line: 55,
                column: 12,
                context: "let norm = LayerNorm::new(768)".to_string(),
            },
        ];
        let resp = IDEResponse::ReferenceLocations { locations };
        if let IDEResponse::ReferenceLocations { locations: locs } = resp {
            assert_eq!(locs.len(), 2);
        }
    }
}
