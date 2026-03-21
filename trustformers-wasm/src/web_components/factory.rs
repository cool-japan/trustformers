use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

use super::components::batch_processor::generate_batch_processor_component;
use super::components::debug_console::generate_debug_console_component;
use super::components::inference_engine::generate_inference_engine_component;
use super::components::model_loader::generate_model_loader_component;
use super::components::model_registry::generate_model_registry_component;
use super::components::performance_monitor::generate_performance_monitor_component;
use super::components::quantization_control::generate_quantization_control_component;
use super::components::tensor_viz::generate_tensor_viz_component;
use super::config::{ComponentType, WebComponentConfig, WebComponentFactory};

#[wasm_bindgen]
impl WebComponentFactory {
    #[wasm_bindgen(constructor)]
    pub fn new(config: WebComponentConfig) -> Result<WebComponentFactory, JsValue> {
        let window = web_sys::window().ok_or("No global window object")?;
        let document = window.document().ok_or("No document object")?;

        Ok(WebComponentFactory {
            config,
            document,
            registered_components: Vec::new(),
        })
    }

    pub fn register_component(&mut self, component_type: ComponentType) -> Result<String, JsValue> {
        let tag_name = match component_type {
            ComponentType::InferenceEngine => "trustformers-inference-engine",
            ComponentType::ModelLoader => "trustformers-model-loader",
            ComponentType::TensorVisualization => "trustformers-tensor-viz",
            ComponentType::PerformanceMonitor => "trustformers-performance-monitor",
            ComponentType::BatchProcessor => "trustformers-batch-processor",
            ComponentType::ModelRegistry => "trustformers-model-registry",
            ComponentType::QuantizationControl => "trustformers-quantization-control",
            ComponentType::DebugConsole => "trustformers-debug-console",
        };

        let component_definition = self.generate_component_definition(component_type)?;

        // Register the custom element
        self.register_custom_element(tag_name, &component_definition)?;

        self.registered_components.push(tag_name.to_string());
        Ok(tag_name.to_string())
    }

    fn generate_component_definition(
        &self,
        component_type: ComponentType,
    ) -> Result<String, JsValue> {
        let (template, styles, script) = match component_type {
            ComponentType::InferenceEngine => generate_inference_engine_component(),
            ComponentType::ModelLoader => generate_model_loader_component(),
            ComponentType::TensorVisualization => generate_tensor_viz_component(),
            ComponentType::PerformanceMonitor => generate_performance_monitor_component(),
            ComponentType::BatchProcessor => generate_batch_processor_component(),
            ComponentType::ModelRegistry => generate_model_registry_component(),
            ComponentType::QuantizationControl => generate_quantization_control_component(),
            ComponentType::DebugConsole => generate_debug_console_component(),
        };

        Ok(format!(
            r#"
class extends HTMLElement {{
    constructor() {{
        super();
        this.config = {};
        this.state = {{}};
        this.setupComponent();
    }}

    setupComponent() {{
        {}

        const template = document.createElement('template');
        template.innerHTML = `
            <style>
                {}
                {}
            </style>
            {}
        `;

        {}

        this.appendChild(template.content.cloneNode(true));
        this.attachEventListeners();
        this.initializeState();
    }}

    {}

    connectedCallback() {{
        this.render();
        this.dispatchEvent(new CustomEvent('trustformers:component-connected', {{
            bubbles: true,
            detail: {{ componentType: '{}' }}
        }}));
    }}

    disconnectedCallback() {{
        this.cleanup();
        this.dispatchEvent(new CustomEvent('trustformers:component-disconnected', {{
            bubbles: true,
            detail: {{ componentType: '{}' }}
        }}));
    }}

    attributeChangedCallback(name, oldValue, newValue) {{
        this.handleAttributeChange(name, oldValue, newValue);
        this.render();
    }}

    static get observedAttributes() {{
        return ['model-id', 'model-url', 'theme', 'debug', 'auto-resize'];
    }}

    render() {{
        // Component-specific rendering logic
        this.updateComponentState();
    }}

    cleanup() {{
        // Component cleanup
        this.removeAllEventListeners();
    }}
}}
"#,
            serde_json::to_string(&self.config).unwrap_or_default(),
            if self.config.enable_shadow_dom() {
                "this.attachShadow({ mode: 'open' });"
            } else {
                ""
            },
            self.get_base_styles(),
            styles,
            template,
            if self.config.enable_shadow_dom() {
                "this.shadowRoot.appendChild(template.content.cloneNode(true));"
            } else {
                ""
            },
            script,
            match component_type {
                ComponentType::InferenceEngine => "InferenceEngine",
                ComponentType::ModelLoader => "ModelLoader",
                ComponentType::TensorVisualization => "TensorVisualization",
                ComponentType::PerformanceMonitor => "PerformanceMonitor",
                ComponentType::BatchProcessor => "BatchProcessor",
                ComponentType::ModelRegistry => "ModelRegistry",
                ComponentType::QuantizationControl => "QuantizationControl",
                ComponentType::DebugConsole => "DebugConsole",
            },
            match component_type {
                ComponentType::InferenceEngine => "InferenceEngine",
                ComponentType::ModelLoader => "ModelLoader",
                ComponentType::TensorVisualization => "TensorVisualization",
                ComponentType::PerformanceMonitor => "PerformanceMonitor",
                ComponentType::BatchProcessor => "BatchProcessor",
                ComponentType::ModelRegistry => "ModelRegistry",
                ComponentType::QuantizationControl => "QuantizationControl",
                ComponentType::DebugConsole => "DebugConsole",
            }
        ))
    }

    fn get_base_styles(&self) -> String {
        let base_styles = r#"
            :host {
                display: block;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                --primary-color: #007acc;
                --secondary-color: #f3f3f3;
                --text-color: #333;
                --border-color: #ddd;
                --error-color: #e74c3c;
                --success-color: #27ae60;
                --warning-color: #f39c12;
                --border-radius: 4px;
                --box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            :host(.dark) {
                --primary-color: #0078d4;
                --secondary-color: #2d2d2d;
                --text-color: #ffffff;
                --border-color: #555;
                background-color: #1a1a1a;
                color: var(--text-color);
            }

            .container {
                padding: 1rem;
                border: 1px solid var(--border-color);
                border-radius: var(--border-radius);
                background: var(--secondary-color);
                box-shadow: var(--box-shadow);
            }

            .header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid var(--border-color);
            }

            .title {
                font-size: 1.2rem;
                font-weight: 600;
                color: var(--text-color);
            }

            .status {
                padding: 0.25rem 0.5rem;
                border-radius: var(--border-radius);
                font-size: 0.875rem;
                font-weight: 500;
            }

            .status.idle { background: #f8f9fa; color: #6c757d; }
            .status.loading { background: #fff3cd; color: #856404; }
            .status.ready { background: #d1ecf1; color: #0c5460; }
            .status.processing { background: #ffeaa7; color: #6c5ce7; }
            .status.complete { background: #d4edda; color: #155724; }
            .status.error { background: #f8d7da; color: #721c24; }

            button {
                background: var(--primary-color);
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: var(--border-radius);
                cursor: pointer;
                font-size: 0.875rem;
                transition: opacity 0.2s;
            }

            button:hover {
                opacity: 0.9;
            }

            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            .grid {
                display: grid;
                gap: 1rem;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            }

            .card {
                background: white;
                border: 1px solid var(--border-color);
                border-radius: var(--border-radius);
                padding: 1rem;
                box-shadow: var(--box-shadow);
            }

            .progress-bar {
                width: 100%;
                height: 4px;
                background: var(--border-color);
                border-radius: 2px;
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                background: var(--primary-color);
                transition: width 0.3s ease;
            }

            .metric {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem 0;
                border-bottom: 1px solid var(--border-color);
            }

            .metric:last-child {
                border-bottom: none;
            }

            .metric-label {
                font-weight: 500;
                color: var(--text-color);
            }

            .metric-value {
                font-family: monospace;
                color: var(--primary-color);
            }
        "#;

        format!(
            "{base_styles}\n{custom}",
            custom = self.config.custom_styles()
        )
    }

    fn register_custom_element(
        &self,
        tag_name: &str,
        class_definition: &str,
    ) -> Result<(), JsValue> {
        let _window = web_sys::window().ok_or("No global window object")?;

        // Use eval to define and register the custom element class
        let script = format!(
            r#"
            if (!customElements.get('{}')) {{
                const elementClass = {};
                customElements.define('{}', elementClass);
            }}
            "#,
            tag_name, class_definition, tag_name
        );

        js_sys::eval(&script)?;
        Ok(())
    }

    pub fn create_component(
        &self,
        component_type: ComponentType,
    ) -> Result<web_sys::Element, JsValue> {
        let tag_name = match component_type {
            ComponentType::InferenceEngine => "trustformers-inference-engine",
            ComponentType::ModelLoader => "trustformers-model-loader",
            ComponentType::TensorVisualization => "trustformers-tensor-viz",
            ComponentType::PerformanceMonitor => "trustformers-performance-monitor",
            ComponentType::BatchProcessor => "trustformers-batch-processor",
            ComponentType::ModelRegistry => "trustformers-model-registry",
            ComponentType::QuantizationControl => "trustformers-quantization-control",
            ComponentType::DebugConsole => "trustformers-debug-console",
        };

        self.document.create_element(tag_name)
    }

    pub fn get_registered_components(&self) -> Vec<String> {
        self.registered_components.clone()
    }

    pub fn is_component_registered(&self, component_type: ComponentType) -> bool {
        let tag_name = match component_type {
            ComponentType::InferenceEngine => "trustformers-inference-engine",
            ComponentType::ModelLoader => "trustformers-model-loader",
            ComponentType::TensorVisualization => "trustformers-tensor-viz",
            ComponentType::PerformanceMonitor => "trustformers-performance-monitor",
            ComponentType::BatchProcessor => "trustformers-batch-processor",
            ComponentType::ModelRegistry => "trustformers-model-registry",
            ComponentType::QuantizationControl => "trustformers-quantization-control",
            ComponentType::DebugConsole => "trustformers-debug-console",
        };

        self.registered_components.contains(&tag_name.to_string())
    }
}
