use std::string::String;
use wasm_bindgen::prelude::*;

use super::config::{WebComponentConfig, WebComponentFactory};

#[wasm_bindgen]
pub fn is_web_components_supported() -> bool {
    let window = match web_sys::window() {
        Some(w) => w,
        None => return false,
    };

    // Check for custom elements support
    let custom_elements =
        js_sys::Reflect::get(&window, &"customElements".into()).unwrap_or_default();
    !custom_elements.is_undefined()
}

#[wasm_bindgen]
pub fn generate_web_components_package(config: WebComponentConfig) -> Result<String, JsValue> {
    let _factory = WebComponentFactory::new(config)?;

    // Generate a complete package with all components
    let package = r#"
// TrustformeRS Web Components Package
// Generated automatically - do not modify

import {{ TrustformersWasm }} from './trustformers-wasm.js';

// Initialize TrustformeRS
const trustformers = new TrustformersWasm();

// Web Components registration
const componentFactory = new WebComponentFactory({{
    enableShadowDom: true,
    theme: 'default',
    customStyles: '',
    enableDebug: false,
    autoResize: true,
    eventDelegation: true
}});

// Register all components
const components = [
    'InferenceEngine',
    'ModelLoader',
    'TensorVisualization',
    'PerformanceMonitor',
    'BatchProcessor',
    'ModelRegistry',
    'QuantizationControl',
    'DebugConsole'
];

components.forEach(component => {{
    componentFactory.registerComponent(component);
}});

// Export for use in applications
export {{ componentFactory, trustformers }};

// Auto-initialize when loaded
if (typeof window !== 'undefined') {{
    window.TrustformersWebComponents = {{ componentFactory, trustformers }};

    // Dispatch ready event
    window.dispatchEvent(new CustomEvent('trustformers:web-components-ready', {{
        detail: {{ components }}
    }}));
}}
"#
    .to_string();

    Ok(package)
}

#[wasm_bindgen]
pub fn create_web_component_html_template() -> String {
    r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS Web Components Demo</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 2rem;
            background: #f8f9fa;
        }

        .demo-container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .component-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin-top: 2rem;
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 2rem;
        }

        .component-section {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .component-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #007acc;
        }
    </style>
</head>
<body>
    <div class="demo-container">
        <h1>TrustformeRS Web Components Demo</h1>

        <div class="component-grid">
            <div class="component-section">
                <div class="component-title">Inference Engine</div>
                <trustformers-inference-engine
                    model-id="demo-model"
                    theme="default">
                </trustformers-inference-engine>
            </div>

            <div class="component-section">
                <div class="component-title">Model Loader</div>
                <trustformers-model-loader theme="default"></trustformers-model-loader>
            </div>

            <div class="component-section">
                <div class="component-title">Tensor Visualization</div>
                <trustformers-tensor-viz theme="default"></trustformers-tensor-viz>
            </div>

            <div class="component-section">
                <div class="component-title">Performance Monitor</div>
                <trustformers-performance-monitor theme="default"></trustformers-performance-monitor>
            </div>

            <div class="component-section">
                <div class="component-title">Batch Processor</div>
                <trustformers-batch-processor theme="default"></trustformers-batch-processor>
            </div>

            <div class="component-section">
                <div class="component-title">Model Registry</div>
                <trustformers-model-registry theme="default"></trustformers-model-registry>
            </div>

            <div class="component-section">
                <div class="component-title">Quantization Control</div>
                <trustformers-quantization-control theme="default"></trustformers-quantization-control>
            </div>

            <div class="component-section">
                <div class="component-title">Debug Console</div>
                <trustformers-debug-console theme="default"></trustformers-debug-console>
            </div>
        </div>
    </div>

    <script type="module">
        import { componentFactory } from './trustformers-web-components.js';

        // Listen for component events
        document.addEventListener('trustformers:component-connected', (e) => {
            console.log('Component connected:', e.detail.componentType);
        });

        document.addEventListener('trustformers:component-disconnected', (e) => {
            console.log('Component disconnected:', e.detail.componentType);
        });

        // Wait for components to be ready
        window.addEventListener('trustformers:web-components-ready', (e) => {
            console.log('TrustformeRS Web Components ready:', e.detail.components);
        });
    </script>
</body>
</html>
"#.to_string()
}
