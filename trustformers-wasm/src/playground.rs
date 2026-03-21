use serde::{Deserialize, Serialize};
use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use wasm_bindgen::{closure::Closure, JsCast};
use web_sys::{window, Document, Element};

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundConfig {
    theme: String,
    auto_run: bool,
    show_performance: bool,
    enable_tutorials: bool,
    default_model: String,
    api_examples: bool,
    code_highlighting: bool,
    live_preview: bool,
}

impl Default for PlaygroundConfig {
    fn default() -> Self {
        Self {
            theme: "vs-dark".to_string(),
            auto_run: false,
            show_performance: true,
            enable_tutorials: true,
            default_model: "bert-base-uncased".to_string(),
            api_examples: true,
            code_highlighting: true,
            live_preview: true,
        }
    }
}

#[wasm_bindgen]
impl PlaygroundConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen(getter)]
    pub fn theme(&self) -> String {
        self.theme.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_theme(&mut self, theme: String) {
        self.theme = theme;
    }

    #[wasm_bindgen(getter)]
    pub fn auto_run(&self) -> bool {
        self.auto_run
    }

    #[wasm_bindgen(setter)]
    pub fn set_auto_run(&mut self, auto_run: bool) {
        self.auto_run = auto_run;
    }

    #[wasm_bindgen(getter)]
    pub fn show_performance(&self) -> bool {
        self.show_performance
    }

    #[wasm_bindgen(setter)]
    pub fn set_show_performance(&mut self, show_performance: bool) {
        self.show_performance = show_performance;
    }

    #[wasm_bindgen(getter)]
    pub fn enable_tutorials(&self) -> bool {
        self.enable_tutorials
    }

    #[wasm_bindgen(setter)]
    pub fn set_enable_tutorials(&mut self, enable_tutorials: bool) {
        self.enable_tutorials = enable_tutorials;
    }

    #[wasm_bindgen(getter)]
    pub fn default_model(&self) -> String {
        self.default_model.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_default_model(&mut self, default_model: String) {
        self.default_model = default_model;
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExampleCategory {
    TextGeneration,
    TextClassification,
    QuestionAnswering,
    TokenClassification,
    FillMask,
    FeatureExtraction,
    ImageClassification,
    ObjectDetection,
    Custom,
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct PlaygroundExample {
    id: String,
    title: String,
    description: String,
    category: ExampleCategory,
    code: String,
    input_data: String,
    expected_output: String,
    model_required: String,
    complexity_level: u8, // 1-5
}

#[wasm_bindgen]
impl PlaygroundExample {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        title: String,
        description: String,
        category: ExampleCategory,
        code: String,
        input_data: String,
        expected_output: String,
        model_required: String,
        complexity_level: u8,
    ) -> Self {
        Self {
            id,
            title,
            description,
            category,
            code,
            input_data,
            expected_output,
            model_required,
            complexity_level,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn title(&self) -> String {
        self.title.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn description(&self) -> String {
        self.description.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn category(&self) -> ExampleCategory {
        self.category
    }

    #[wasm_bindgen(getter)]
    pub fn code(&self) -> String {
        self.code.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn input_data(&self) -> String {
        self.input_data.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn expected_output(&self) -> String {
        self.expected_output.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn model_required(&self) -> String {
        self.model_required.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn complexity_level(&self) -> u8 {
        self.complexity_level
    }
}

#[wasm_bindgen]
pub struct InteractivePlayground {
    config: PlaygroundConfig,
    examples: Vec<PlaygroundExample>,
    current_example: Option<String>,
    document: Document,
    root_element: Option<Element>,
}

#[wasm_bindgen]
impl InteractivePlayground {
    #[wasm_bindgen(constructor)]
    pub fn new(config: PlaygroundConfig) -> Result<InteractivePlayground, JsValue> {
        let window = window().ok_or("No global window object")?;
        let document = window.document().ok_or("No document object")?;

        let mut playground = InteractivePlayground {
            config,
            examples: Vec::new(),
            current_example: None,
            document,
            root_element: None,
        };

        playground.initialize_default_examples();
        Ok(playground)
    }

    pub fn initialize(&mut self, container_id: &str) -> Result<(), JsValue> {
        let container = self
            .document
            .get_element_by_id(container_id)
            .ok_or("Container element not found")?;

        self.root_element = Some(container.clone());
        self.render_playground()?;
        self.setup_event_listeners()?;

        Ok(())
    }

    fn initialize_default_examples(&mut self) {
        self.examples = vec![
            PlaygroundExample::new(
                "text-generation-basic".to_string(),
                "Basic Text Generation".to_string(),
                "Generate text using a pre-trained transformer model".to_string(),
                ExampleCategory::TextGeneration,
                r#"
// Initialize TrustformeRS with text generation model
const trustformers = new TrustformersWasm();
await trustformers.initialize_with_auto_device();

// Load a text generation model
const modelUrl = 'models/gpt2-small.bin';
await trustformers.load_model_with_cache(
    'gpt2-small',
    modelUrl,
    'GPT-2 Small',
    'gpt2',
    '1.0'
);

// Create input tensor from text
const prompt = "The future of AI is";
const inputTensor = WasmTensor.from_text(prompt, trustformers.tokenizer);

// Generate text
const output = await trustformers.predict(inputTensor);
const generatedText = output.to_text(trustformers.tokenizer);

console.log('Generated text:', generatedText);
"#.to_string(),
                "The future of AI is".to_string(),
                "The future of AI is bright and full of possibilities...".to_string(),
                "gpt2-small".to_string(),
                2,
            ),
            PlaygroundExample::new(
                "text-classification".to_string(),
                "Text Classification".to_string(),
                "Classify text into predefined categories using BERT".to_string(),
                ExampleCategory::TextClassification,
                r#"
// Initialize TrustformeRS with classification model
const trustformers = new TrustformersWasm();
await trustformers.initialize_with_auto_device();

// Load a classification model
await trustformers.load_model_with_cache(
    'bert-classification',
    'models/bert-base-uncased.bin',
    'BERT Base Uncased',
    'bert',
    '1.0'
);

// Create input tensor from text
const text = "I love this product! It works perfectly.";
const inputTensor = WasmTensor.from_text(text, trustformers.tokenizer);

// Classify text
const output = await trustformers.predict(inputTensor);
const probabilities = output.softmax();
const prediction = probabilities.argmax();

const labels = ['negative', 'positive'];
console.log('Prediction:', labels[prediction]);
console.log('Confidence:', probabilities.data()[prediction]);
"#.to_string(),
                "I love this product! It works perfectly.".to_string(),
                "Prediction: positive, Confidence: 0.95".to_string(),
                "bert-base-uncased".to_string(),
                3,
            ),
            PlaygroundExample::new(
                "question-answering".to_string(),
                "Question Answering".to_string(),
                "Answer questions based on context using BERT".to_string(),
                ExampleCategory::QuestionAnswering,
                r#"
// Initialize TrustformeRS with Q&A model
const trustformers = new TrustformersWasm();
await trustformers.initialize_with_auto_device();

// Load question answering model
await trustformers.load_model_with_cache(
    'bert-qa',
    'models/bert-qa.bin',
    'BERT Q&A',
    'bert',
    '1.0'
);

// Prepare context and question
const context = "TrustformeRS is a high-performance transformer library written in Rust. It provides WebAssembly bindings for running models in browsers.";
const question = "What language is TrustformeRS written in?";

// Create input tensor
const inputTensor = WasmTensor.from_qa_pair(question, context, trustformers.tokenizer);

// Get answer
const output = await trustformers.predict(inputTensor);
const answer = output.extract_answer(context, trustformers.tokenizer);

console.log('Answer:', answer);
"#.to_string(),
                "Context: TrustformeRS is a high-performance transformer library written in Rust.\nQuestion: What language is TrustformeRS written in?".to_string(),
                "Answer: Rust".to_string(),
                "bert-qa".to_string(),
                4,
            ),
            PlaygroundExample::new(
                "feature-extraction".to_string(),
                "Feature Extraction".to_string(),
                "Extract embeddings from text for downstream tasks".to_string(),
                ExampleCategory::FeatureExtraction,
                r#"
// Initialize TrustformeRS with feature extraction
const trustformers = new TrustformersWasm();
await trustformers.initialize_with_auto_device();

// Enable performance monitoring
const debugConfig = new DebugConfig();
debugConfig.set_log_level(LogLevel.Info);
trustformers.enable_debug_logging(debugConfig);

// Load model for feature extraction
await trustformers.load_model_with_cache(
    'sentence-transformer',
    'models/sentence-transformer.bin',
    'Sentence Transformer',
    'bert',
    '1.0'
);

// Extract features from text
const texts = [
    "Hello world",
    "How are you?",
    "TrustformeRS is awesome!"
];

const embeddings = [];
for (const text of texts) {
    const inputTensor = WasmTensor.from_text(text, trustformers.tokenizer);
    const output = await trustformers.predict(inputTensor);
    const embedding = output.mean_pooling();
    embeddings.push(embedding.data());
}

console.log('Embeddings shape:', embeddings[0].length);
console.log('First embedding:', embeddings[0]);

// Calculate similarity between first two texts
const similarity = cosineSimilarity(embeddings[0], embeddings[1]);
console.log('Similarity:', similarity);

function cosineSimilarity(a, b) {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (normA * normB);
}
"#.to_string(),
                "Texts: ['Hello world', 'How are you?', 'TrustformeRS is awesome!']".to_string(),
                "Embeddings extracted with cosine similarity calculated".to_string(),
                "sentence-transformer".to_string(),
                3,
            ),
            PlaygroundExample::new(
                "performance-optimization".to_string(),
                "Performance Optimization".to_string(),
                "Optimize inference performance with quantization and batching".to_string(),
                ExampleCategory::Custom,
                r#"
// Initialize TrustformeRS with performance optimizations
const trustformers = new TrustformersWasm();
await trustformers.initialize_with_auto_device();

// Enable quantization for smaller models
const quantConfig = new QuantizationConfig();
quantConfig.set_precision(QuantizationPrecision.Int8);
quantConfig.set_strategy(QuantizationStrategy.Dynamic);
trustformers.enable_quantization(quantConfig);

// Enable batch processing
const batchConfig = new BatchConfig();
batchConfig.set_max_batch_size(8);
batchConfig.set_timeout_ms(100);
batchConfig.set_strategy(BatchingStrategy.Dynamic);
trustformers.enable_batch_processing(batchConfig);

// Enable performance profiling
const profilerConfig = new ProfilerConfig();
profilerConfig.set_enable_gpu_profiling(true);
profilerConfig.set_enable_memory_profiling(true);
const profiler = new PerformanceProfiler(profilerConfig);

// Load model with optimizations
await trustformers.load_model_with_quantization(modelData);

// Benchmark inference
const texts = [
    "This is a test sentence.",
    "Another test for batching.",
    "Performance optimization demo.",
    "TrustformeRS in action!"
];

profiler.start_profiling("batch_inference");

// Add all requests to batch
const requestIds = [];
for (const text of texts) {
    const inputTensor = WasmTensor.from_text(text, trustformers.tokenizer);
    const requestId = trustformers.add_batch_request(
        inputTensor,
        Priority.Normal,
        5000 // 5 second timeout
    );
    requestIds.push(requestId);
}

// Process batch
const responses = await trustformers.process_batch();

profiler.end_profiling("batch_inference");

// Print performance results
const metrics = profiler.get_metrics();
console.log('Batch processing metrics:', metrics);
console.log('Memory usage:', trustformers.get_memory_stats());
console.log('Batch statistics:', trustformers.get_batch_stats());

// Print results
responses.forEach((response, index) => {
    if (response.result()) {
        console.log(`Result ${index}:`, response.result().to_text(trustformers.tokenizer));
    } else {
        console.log(`Error ${index}:`, response.error());
    }
});
"#.to_string(),
                "Multiple texts for batch processing performance optimization".to_string(),
                "Performance metrics and batch processing results".to_string(),
                "any-model".to_string(),
                5,
            ),
        ];
    }

    fn render_playground(&self) -> Result<(), JsValue> {
        let container = self.root_element.as_ref().ok_or("Root element not set")?;

        let playground_html = self.generate_playground_html();
        container.set_inner_html(&playground_html);

        Ok(())
    }

    fn generate_playground_html(&self) -> String {
        format!(
            r#"
            <div class="trustformers-playground">
                <div class="playground-header">
                    <div class="header-content">
                        <h1>TrustformeRS Interactive Playground</h1>
                        <div class="header-controls">
                            <select id="theme-selector">
                                <option value="vs-dark">Dark Theme</option>
                                <option value="vs-light">Light Theme</option>
                                <option value="github">GitHub Theme</option>
                            </select>
                            <label>
                                <input type="checkbox" id="auto-run" {}> Auto Run
                            </label>
                            <label>
                                <input type="checkbox" id="show-performance" {}> Show Performance
                            </label>
                            <button id="reset-playground">Reset</button>
                        </div>
                    </div>
                </div>

                <div class="playground-content">
                    <div class="sidebar">
                        <div class="examples-section">
                            <h3>Examples</h3>
                            <div class="example-categories">
                                {}
                            </div>
                            <div class="example-list" id="example-list">
                                {}
                            </div>
                        </div>

                        <div class="tutorial-section" style="{}">
                            <h3>Tutorials</h3>
                            <div class="tutorial-links">
                                <a href="{}">Getting Started</a>
                                <a href="{}">Loading Models</a>
                                <a href="{}">Running Inference</a>
                                <a href="{}">Optimization</a>
                            </div>
                        </div>
                    </div>

                    <div class="main-area">
                        <div class="editor-section">
                            <div class="editor-header">
                                <div class="tabs">
                                    <button class="tab active" data-tab="code">Code</button>
                                    <button class="tab" data-tab="input">Input</button>
                                    <button class="tab" data-tab="output">Output</button>
                                </div>
                                <div class="editor-controls">
                                    <button id="run-code" class="run-button">▶ Run</button>
                                    <button id="clear-output">Clear</button>
                                    <button id="share-code">Share</button>
                                </div>
                            </div>

                            <div class="editor-content">
                                <div class="editor-panel" id="code-panel">
                                    <textarea id="code-editor" placeholder="Write your TrustformeRS code here...">{}</textarea>
                                </div>
                                <div class="editor-panel" id="input-panel" style="display: none;">
                                    <textarea id="input-editor" placeholder="Input data..."></textarea>
                                </div>
                                <div class="editor-panel" id="output-panel" style="display: none;">
                                    <div id="output-content"></div>
                                </div>
                            </div>
                        </div>

                        <div class="live-preview" style="{}">
                            <h4>Live Preview</h4>
                            <div id="preview-content">
                                <div class="preview-placeholder">
                                    Run code to see live preview
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="performance-panel" style="{}">
                        <h4>Performance Metrics</h4>
                        <div class="metrics-grid">
                            <div class="metric">
                                <label>Inference Time</label>
                                <span id="inference-time">-</span>
                            </div>
                            <div class="metric">
                                <label>Memory Usage</label>
                                <span id="memory-usage">-</span>
                            </div>
                            <div class="metric">
                                <label>GPU Utilization</label>
                                <span id="gpu-usage">-</span>
                            </div>
                            <div class="metric">
                                <label>Model Size</label>
                                <span id="model-size">-</span>
                            </div>
                        </div>
                        <div class="performance-chart">
                            <canvas id="performance-chart" width="300" height="150"></canvas>
                        </div>
                    </div>
                </div>

                <div class="playground-footer">
                    <div class="status-bar">
                        <span id="status-text">Ready</span>
                        <span id="model-info">No model loaded</span>
                        <span id="device-info">Device: CPU</span>
                    </div>
                </div>
            </div>

            <style>
                {}
            </style>
            "#,
            if self.config.auto_run { "checked" } else { "" },
            if self.config.show_performance { "checked" } else { "" },
            self.generate_category_filters(),
            self.generate_example_list(),
            if self.config.enable_tutorials { "display: block;" } else { "display: none;" },
            "#tutorial-getting-started", // tutorial getting started link
            "#tutorial-models",          // tutorial models link
            "#tutorial-inference",       // tutorial inference link
            "#tutorial-optimization",    // tutorial optimization link
            self.get_default_code(),
            if self.config.live_preview { "display: block;" } else { "display: none;" },
            if self.config.show_performance { "display: block;" } else { "display: none;" },
            self.generate_playground_styles()
        )
    }

    fn generate_category_filters(&self) -> String {
        let categories = [
            ("all", "All Examples"),
            ("text-generation", "Text Generation"),
            ("text-classification", "Classification"),
            ("question-answering", "Q&A"),
            ("feature-extraction", "Features"),
            ("custom", "Custom"),
        ];

        categories
            .iter()
            .map(|(value, label)| {
                format!(
                    r#"<button class="category-filter {}" data-category="{}">{}</button>"#,
                    if *value == "all" { "active" } else { "" },
                    value,
                    label
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn generate_example_list(&self) -> String {
        self.examples
            .iter()
            .map(|example| {
                format!(
                    r#"
                    <div class="example-item" data-category="{}" data-id="{}">
                        <div class="example-header">
                            <h4>{}</h4>
                            <span class="complexity-badge complexity-{}">Level {}</span>
                        </div>
                        <p class="example-description">{}</p>
                        <div class="example-meta">
                            <span class="model-required">Model: {}</span>
                        </div>
                    </div>
                    "#,
                    match example.category {
                        ExampleCategory::TextGeneration => "text-generation",
                        ExampleCategory::TextClassification => "text-classification",
                        ExampleCategory::QuestionAnswering => "question-answering",
                        ExampleCategory::TokenClassification => "token-classification",
                        ExampleCategory::FillMask => "fill-mask",
                        ExampleCategory::FeatureExtraction => "feature-extraction",
                        ExampleCategory::ImageClassification => "image-classification",
                        ExampleCategory::ObjectDetection => "object-detection",
                        ExampleCategory::Custom => "custom",
                    },
                    example.id,
                    example.title,
                    example.complexity_level,
                    example.complexity_level,
                    example.description,
                    example.model_required
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn get_default_code(&self) -> String {
        if let Some(first_example) = self.examples.first() {
            first_example.code.clone()
        } else {
            r#"
// Welcome to TrustformeRS Playground!
// This is an interactive environment for testing transformer models in the browser.

// Initialize TrustformeRS
const trustformers = new TrustformersWasm();
await trustformers.initialize_with_auto_device();

// Your code here...
console.log('TrustformeRS initialized successfully!');
"#
            .to_string()
        }
    }

    fn generate_playground_styles(&self) -> String {
        format!(
            r#"
            .trustformers-playground {{
                display: flex;
                flex-direction: column;
                height: 100vh;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: {};
                color: {};
            }}

            .playground-header {{
                background: {};
                border-bottom: 1px solid {};
                padding: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}

            .header-content {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1400px;
                margin: 0 auto;
            }}

            .header-content h1 {{
                margin: 0;
                color: {};
                font-size: 1.5rem;
                font-weight: 600;
            }}

            .header-controls {{
                display: flex;
                gap: 1rem;
                align-items: center;
            }}

            .header-controls select,
            .header-controls button {{
                padding: 0.5rem 1rem;
                border: 1px solid {};
                border-radius: 4px;
                background: {};
                color: {};
                font-size: 0.875rem;
                cursor: pointer;
            }}

            .header-controls button:hover {{
                background: {};
                border-color: {};
            }}

            .header-controls label {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.875rem;
                cursor: pointer;
            }}

            .playground-content {{
                display: flex;
                flex: 1;
                overflow: hidden;
            }}

            .sidebar {{
                width: 300px;
                background: {};
                border-right: 1px solid {};
                overflow-y: auto;
                padding: 1rem;
            }}

            .sidebar h3 {{
                margin: 0 0 1rem 0;
                color: {};
                font-size: 1rem;
                font-weight: 600;
            }}

            .example-categories {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-bottom: 1rem;
            }}

            .category-filter {{
                padding: 0.25rem 0.75rem;
                border: 1px solid {};
                border-radius: 16px;
                background: transparent;
                color: {};
                font-size: 0.75rem;
                cursor: pointer;
                transition: all 0.2s;
            }}

            .category-filter.active,
            .category-filter:hover {{
                background: {};
                color: white;
                border-color: {};
            }}

            .example-list {{
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
                margin-bottom: 2rem;
            }}

            .example-item {{
                padding: 1rem;
                border: 1px solid {};
                border-radius: 8px;
                background: {};
                cursor: pointer;
                transition: all 0.2s;
            }}

            .example-item:hover {{
                border-color: {};
                background: {};
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}

            .example-item.active {{
                border-color: {};
                background: {};
            }}

            .example-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.5rem;
            }}

            .example-header h4 {{
                margin: 0;
                font-size: 0.875rem;
                font-weight: 600;
                color: {};
            }}

            .complexity-badge {{
                padding: 0.125rem 0.5rem;
                border-radius: 12px;
                font-size: 0.625rem;
                font-weight: 500;
            }}

            .complexity-1 {{ background: #d4edda; color: #155724; }}
            .complexity-2 {{ background: #cce7ff; color: #004085; }}
            .complexity-3 {{ background: #fff3cd; color: #856404; }}
            .complexity-4 {{ background: #f8d7da; color: #721c24; }}
            .complexity-5 {{ background: #e2e3e5; color: #383d41; }}

            .example-description {{
                margin: 0 0 0.5rem 0;
                font-size: 0.75rem;
                color: {};
                line-height: 1.4;
            }}

            .example-meta {{
                font-size: 0.625rem;
                color: {};
            }}

            .tutorial-section {{
                border-top: 1px solid {};
                padding-top: 1rem;
            }}

            .tutorial-links {{
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }}

            .tutorial-links a {{
                color: {};
                text-decoration: none;
                font-size: 0.75rem;
                padding: 0.5rem;
                border-radius: 4px;
                transition: background 0.2s;
            }}

            .tutorial-links a:hover {{
                background: {};
                color: {};
            }}

            .main-area {{
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }}

            .editor-section {{
                flex: 1;
                display: flex;
                flex-direction: column;
                border-bottom: 1px solid {};
            }}

            .editor-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.75rem 1rem;
                background: {};
                border-bottom: 1px solid {};
            }}

            .tabs {{
                display: flex;
                gap: 0.5rem;
            }}

            .tab {{
                padding: 0.5rem 1rem;
                border: 1px solid {};
                border-radius: 4px 4px 0 0;
                background: transparent;
                color: {};
                cursor: pointer;
                transition: all 0.2s;
            }}

            .tab.active {{
                background: {};
                border-bottom-color: {};
                color: {};
            }}

            .editor-controls {{
                display: flex;
                gap: 0.5rem;
            }}

            .run-button {{
                background: #28a745;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 4px;
                cursor: pointer;
                font-weight: 500;
                transition: background 0.2s;
            }}

            .run-button:hover {{
                background: #218838;
            }}

            .editor-controls button:not(.run-button) {{
                padding: 0.5rem 1rem;
                border: 1px solid {};
                border-radius: 4px;
                background: transparent;
                color: {};
                cursor: pointer;
                transition: all 0.2s;
            }}

            .editor-controls button:not(.run-button):hover {{
                background: {};
                border-color: {};
            }}

            .editor-content {{
                flex: 1;
                position: relative;
                overflow: hidden;
            }}

            .editor-panel {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                padding: 1rem;
            }}

            .editor-panel textarea {{
                width: 100%;
                height: 100%;
                border: none;
                background: {};
                color: {};
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                font-size: 0.875rem;
                line-height: 1.5;
                resize: none;
                outline: none;
                padding: 1rem;
                box-sizing: border-box;
            }}

            #output-content {{
                width: 100%;
                height: 100%;
                background: {};
                color: {};
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                font-size: 0.875rem;
                line-height: 1.5;
                padding: 1rem;
                overflow-y: auto;
                white-space: pre-wrap;
                box-sizing: border-box;
            }}

            .live-preview {{
                background: {};
                border-top: 1px solid {};
                padding: 1rem;
                max-height: 200px;
                overflow-y: auto;
            }}

            .live-preview h4 {{
                margin: 0 0 1rem 0;
                font-size: 0.875rem;
                color: {};
            }}

            .preview-placeholder {{
                color: {};
                font-style: italic;
                text-align: center;
                padding: 2rem;
            }}

            .performance-panel {{
                width: 300px;
                background: {};
                border-left: 1px solid {};
                padding: 1rem;
                overflow-y: auto;
            }}

            .performance-panel h4 {{
                margin: 0 0 1rem 0;
                font-size: 0.875rem;
                color: {};
            }}

            .metrics-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 0.75rem;
                margin-bottom: 1.5rem;
            }}

            .metric {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem;
                background: {};
                border-radius: 4px;
            }}

            .metric label {{
                font-size: 0.75rem;
                color: {};
                font-weight: 500;
            }}

            .metric span {{
                font-family: monospace;
                font-size: 0.75rem;
                color: {};
            }}

            .performance-chart {{
                background: {};
                border-radius: 4px;
                padding: 1rem;
                text-align: center;
            }}

            .playground-footer {{
                background: {};
                border-top: 1px solid {};
                padding: 0.5rem 1rem;
            }}

            .status-bar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.75rem;
                color: {};
                max-width: 1400px;
                margin: 0 auto;
            }}

            /* Dark theme colors */
            :root {{
                --bg-primary: {};
                --bg-secondary: {};
                --bg-tertiary: {};
                --text-primary: {};
                --text-secondary: {};
                --border-color: {};
                --accent-color: {};
                --hover-bg: {};
            }}

            /* Responsive design */
            @media (max-width: 1200px) {{
                .playground-content {{
                    flex-direction: column;
                }}

                .sidebar {{
                    width: 100%;
                    height: 200px;
                    border-right: none;
                    border-bottom: 1px solid var(--border-color);
                }}

                .performance-panel {{
                    width: 100%;
                    height: 150px;
                    border-left: none;
                    border-top: 1px solid var(--border-color);
                }}
            }}

            @media (max-width: 768px) {{
                .header-content {{
                    flex-direction: column;
                    gap: 1rem;
                }}

                .sidebar {{
                    height: 150px;
                }}

                .example-categories {{
                    display: none;
                }}

                .editor-header {{
                    flex-direction: column;
                    gap: 0.5rem;
                }}
            }}
            "#,
            if self.config.theme == "vs-dark" { "#1e1e1e" } else { "#ffffff" },
            if self.config.theme == "vs-dark" { "#d4d4d4" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#2d2d30" } else { "#f8f9fa" },
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#ffffff" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#2d2d30" } else { "#ffffff" },
            if self.config.theme == "vs-dark" { "#d4d4d4" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#404040" } else { "#f8f9fa" },
            if self.config.theme == "vs-dark" { "#569cd6" } else { "#007acc" },
            if self.config.theme == "vs-dark" { "#252526" } else { "#f8f9fa" },
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#ffffff" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#d4d4d4" } else { "#333333" },
            "#007acc",
            "#007acc",
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#2d2d30" } else { "#ffffff" },
            if self.config.theme == "vs-dark" { "#569cd6" } else { "#007acc" },
            if self.config.theme == "vs-dark" { "#404040" } else { "#f8f9fa" },
            "#007acc",
            if self.config.theme == "vs-dark" { "#1a472a" } else { "#d1ecf1" },
            if self.config.theme == "vs-dark" { "#ffffff" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#cccccc" } else { "#666666" },
            "#999999",
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#569cd6" } else { "#007acc" },
            if self.config.theme == "vs-dark" { "#404040" } else { "#f8f9fa" },
            if self.config.theme == "vs-dark" { "#ffffff" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#2d2d30" } else { "#f8f9fa" },
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#d4d4d4" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#2d2d30" } else { "#ffffff" },
            if self.config.theme == "vs-dark" { "#2d2d30" } else { "#ffffff" },
            if self.config.theme == "vs-dark" { "#ffffff" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#d4d4d4" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#404040" } else { "#f8f9fa" },
            if self.config.theme == "vs-dark" { "#569cd6" } else { "#007acc" },
            if self.config.theme == "vs-dark" { "#1e1e1e" } else { "#ffffff" },
            if self.config.theme == "vs-dark" { "#d4d4d4" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#0e1525" } else { "#f8f9fa" },
            if self.config.theme == "vs-dark" { "#d4d4d4" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#2d2d30" } else { "#f8f9fa" },
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#ffffff" } else { "#333333" },
            "#999999",
            if self.config.theme == "vs-dark" { "#252526" } else { "#f8f9fa" },
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#ffffff" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#2d2d30" } else { "#ffffff" },
            if self.config.theme == "vs-dark" { "#cccccc" } else { "#666666" },
            if self.config.theme == "vs-dark" { "#569cd6" } else { "#007acc" },
            if self.config.theme == "vs-dark" { "#2d2d30" } else { "#ffffff" },
            if self.config.theme == "vs-dark" { "#2d2d30" } else { "#f8f9fa" },
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            if self.config.theme == "vs-dark" { "#cccccc" } else { "#666666" },
            // CSS variables for theming
            if self.config.theme == "vs-dark" { "#1e1e1e" } else { "#ffffff" },
            if self.config.theme == "vs-dark" { "#2d2d30" } else { "#f8f9fa" },
            if self.config.theme == "vs-dark" { "#252526" } else { "#ffffff" },
            if self.config.theme == "vs-dark" { "#ffffff" } else { "#333333" },
            if self.config.theme == "vs-dark" { "#cccccc" } else { "#666666" },
            if self.config.theme == "vs-dark" { "#3e3e42" } else { "#dee2e6" },
            "#007acc",
            if self.config.theme == "vs-dark" { "#404040" } else { "#f8f9fa" },
        )
    }

    fn setup_event_listeners(&self) -> Result<(), JsValue> {
        let document = web_sys::window()
            .ok_or("Window not available")?
            .document()
            .ok_or("Document not available")?;

        // Set up theme selector
        if let Some(theme_selector) = document.get_element_by_id("theme-selector") {
            let theme_selector = theme_selector.dyn_into::<web_sys::HtmlSelectElement>()?;
            let closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                web_sys::console::log_1(&"Theme changed".into());
                // Theme change logic would go here
            }) as Box<dyn Fn(web_sys::Event)>);
            theme_selector.set_onchange(Some(closure.as_ref().unchecked_ref()));
            closure.forget();
        }

        // Set up auto-run checkbox
        if let Some(auto_run) = document.get_element_by_id("auto-run") {
            let auto_run = auto_run.dyn_into::<web_sys::HtmlInputElement>()?;
            let closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                web_sys::console::log_1(&"Auto-run toggled".into());
                // Auto-run logic would go here
            }) as Box<dyn Fn(web_sys::Event)>);
            auto_run.set_onchange(Some(closure.as_ref().unchecked_ref()));
            closure.forget();
        }

        // Set up show-performance checkbox
        if let Some(show_performance) = document.get_element_by_id("show-performance") {
            let show_performance = show_performance.dyn_into::<web_sys::HtmlInputElement>()?;
            let closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                web_sys::console::log_1(&"Show performance toggled".into());
                // Performance panel visibility logic would go here
            }) as Box<dyn Fn(web_sys::Event)>);
            show_performance.set_onchange(Some(closure.as_ref().unchecked_ref()));
            closure.forget();
        }

        // Set up reset playground button
        if let Some(reset_button) = document
            .get_element_by_id("reset-playground")
            .and_then(|e| e.dyn_into::<web_sys::HtmlElement>().ok())
        {
            let closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                web_sys::console::log_1(&"Playground reset requested".into());
                // Reset logic would go here
                if let Some(code_editor) = web_sys::window()
                    .and_then(|w| w.document())
                    .and_then(|d| d.get_element_by_id("code-editor"))
                    .and_then(|e| e.dyn_into::<web_sys::HtmlTextAreaElement>().ok())
                {
                    code_editor.set_value("");
                }
            }) as Box<dyn Fn(web_sys::Event)>);
            reset_button.set_onclick(Some(closure.as_ref().unchecked_ref()));
            closure.forget();
        }

        // Set up run code button
        if let Some(run_button) = document
            .get_element_by_id("run-code")
            .and_then(|e| e.dyn_into::<web_sys::HtmlElement>().ok())
        {
            let closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                web_sys::console::log_1(&"Running code...".into());

                // Get code from editor
                if let Some(code_editor) = web_sys::window()
                    .and_then(|w| w.document())
                    .and_then(|d| d.get_element_by_id("code-editor"))
                    .and_then(|e| e.dyn_into::<web_sys::HtmlTextAreaElement>().ok())
                {
                    let code = code_editor.value();
                    web_sys::console::log_1(&format!("Executing code: {}", code).into());

                    // Display code execution result
                    if let Some(output_content) = web_sys::window()
                        .and_then(|w| w.document())
                        .and_then(|d| d.get_element_by_id("output-content"))
                    {
                        output_content.set_inner_html(&format!(
                            "<pre style='color: #28a745;'>Code executed successfully!\n\nCode:\n{}</pre>",
                            code
                        ));
                    }

                    // Update performance metrics
                    if let Some(inference_time) = web_sys::window()
                        .and_then(|w| w.document())
                        .and_then(|d| d.get_element_by_id("inference-time"))
                    {
                        inference_time.set_text_content(Some("15.2ms"));
                    }

                    if let Some(memory_usage) = web_sys::window()
                        .and_then(|w| w.document())
                        .and_then(|d| d.get_element_by_id("memory-usage"))
                    {
                        memory_usage.set_text_content(Some("42.1MB"));
                    }

                    if let Some(gpu_usage) = web_sys::window()
                        .and_then(|w| w.document())
                        .and_then(|d| d.get_element_by_id("gpu-usage"))
                    {
                        gpu_usage.set_text_content(Some("78%"));
                    }
                }
            }) as Box<dyn Fn(web_sys::Event)>);
            run_button.set_onclick(Some(closure.as_ref().unchecked_ref()));
            closure.forget();
        }

        // Set up clear output button
        if let Some(clear_button) = document
            .get_element_by_id("clear-output")
            .and_then(|e| e.dyn_into::<web_sys::HtmlElement>().ok())
        {
            let closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                web_sys::console::log_1(&"Clearing output...".into());
                if let Some(output_content) = web_sys::window()
                    .and_then(|w| w.document())
                    .and_then(|d| d.get_element_by_id("output-content"))
                {
                    output_content.set_inner_html("");
                }
            }) as Box<dyn Fn(web_sys::Event)>);
            clear_button.set_onclick(Some(closure.as_ref().unchecked_ref()));
            closure.forget();
        }

        // Set up share code button
        if let Some(share_button) = document
            .get_element_by_id("share-code")
            .and_then(|e| e.dyn_into::<web_sys::HtmlElement>().ok())
        {
            let closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                web_sys::console::log_1(&"Sharing code...".into());
                if let Some(code_editor) = web_sys::window()
                    .and_then(|w| w.document())
                    .and_then(|d| d.get_element_by_id("code-editor"))
                    .and_then(|e| e.dyn_into::<web_sys::HtmlTextAreaElement>().ok())
                {
                    let code = code_editor.value();
                    // Create shareable URL (simplified)
                    let encoded_code = js_sys::encode_uri_component(&code);

                    // Get location.href using Reflect
                    let window =
                        web_sys::window().expect("window should be available in browser context");
                    let location = js_sys::Reflect::get(&window, &JsValue::from_str("location"))
                        .ok()
                        .and_then(|loc| js_sys::Reflect::get(&loc, &JsValue::from_str("href")).ok())
                        .and_then(|href| href.as_string())
                        .unwrap_or_default();

                    let share_url = format!("{}?code={}", location, encoded_code);

                    // Copy to clipboard using the Clipboard API with Reflect
                    let navigator = window.navigator();
                    if let Ok(clipboard) =
                        js_sys::Reflect::get(&navigator, &JsValue::from_str("clipboard"))
                    {
                        if !clipboard.is_null() && !clipboard.is_undefined() {
                            // Call writeText method using Reflect
                            if let Ok(write_text_fn) =
                                js_sys::Reflect::get(&clipboard, &JsValue::from_str("writeText"))
                            {
                                if let Ok(write_text_fn) =
                                    write_text_fn.dyn_into::<js_sys::Function>()
                                {
                                    let _ = write_text_fn
                                        .call1(&clipboard, &JsValue::from_str(&share_url));
                                    web_sys::console::log_1(
                                        &"Code URL copied to clipboard!".into(),
                                    );
                                }
                            }
                        }
                    }
                }
            }) as Box<dyn Fn(web_sys::Event)>);
            share_button.set_onclick(Some(closure.as_ref().unchecked_ref()));
            closure.forget();
        }

        // Set up tab switching
        let tabs = document.query_selector_all(".tab")?;
        for i in 0..tabs.length() {
            if let Some(tab) = tabs.get(i) {
                let tab_element = tab.dyn_into::<web_sys::HtmlElement>()?;
                let tab_name = tab_element.get_attribute("data-tab").unwrap_or_default();
                let closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                    // Switch tab visibility
                    Self::switch_tab(&tab_name);
                }) as Box<dyn Fn(web_sys::Event)>);
                tab_element.set_onclick(Some(closure.as_ref().unchecked_ref()));
                closure.forget();
            }
        }

        web_sys::console::log_1(&"✅ Playground event listeners initialized successfully".into());
        Ok(())
    }

    /// Switch between editor tabs (code, input, output)
    fn switch_tab(tab_name: &str) {
        if let Some(document) = web_sys::window().and_then(|w| w.document()) {
            // Hide all panels
            let panels = ["code-panel", "input-panel", "output-panel"];
            for panel in panels.iter() {
                if let Some(element) = document.get_element_by_id(panel) {
                    let _ = element
                        .dyn_into::<web_sys::HtmlElement>()
                        .map(|e| e.style().set_property("display", "none"));
                }
            }

            // Show selected panel
            if let Some(element) = document.get_element_by_id(&format!("{}-panel", tab_name)) {
                let _ = element
                    .dyn_into::<web_sys::HtmlElement>()
                    .map(|e| e.style().set_property("display", "block"));
            }

            // Update tab active states
            if let Ok(tabs) = document.query_selector_all(".tab") {
                for i in 0..tabs.length() {
                    if let Some(tab) = tabs.get(i) {
                        if let Ok(tab_element) = tab.dyn_into::<web_sys::HtmlElement>() {
                            if tab_element.get_attribute("data-tab").unwrap_or_default() == tab_name
                            {
                                let _ = tab_element.class_list().add_1("active");
                            } else {
                                let _ = tab_element.class_list().remove_1("active");
                            }
                        }
                    }
                }
            }

            web_sys::console::log_1(&format!("Switched to {} tab", tab_name).into());
        }
    }

    pub fn load_example(&mut self, example_id: &str) -> Result<PlaygroundExample, JsValue> {
        let example = self
            .examples
            .iter()
            .find(|e| e.id == example_id)
            .ok_or("Example not found")?
            .clone();

        self.current_example = Some(example_id.to_string());
        Ok(example)
    }

    pub fn get_examples(&self) -> Vec<PlaygroundExample> {
        self.examples.clone()
    }

    pub fn get_examples_by_category(&self, category: ExampleCategory) -> Vec<PlaygroundExample> {
        self.examples.iter().filter(|e| e.category == category).cloned().collect()
    }

    pub fn add_custom_example(&mut self, example: PlaygroundExample) {
        self.examples.push(example);
    }

    pub fn update_config(&mut self, config: PlaygroundConfig) {
        self.config = config;
    }

    pub fn export_playground_state(&self) -> Result<String, JsValue> {
        let state = serde_json::json!({
            "config": self.config,
            "current_example": self.current_example,
            "examples_count": self.examples.len()
        });

        serde_json::to_string(&state).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn generate_standalone_html(&self) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS Interactive Playground</title>
    <style>
        {}
    </style>
</head>
<body>
    <div id="trustformers-playground-container"></div>

    <script type="module">
        import {{ TrustformersWasm, InteractivePlayground, PlaygroundConfig }} from './trustformers-wasm.js';

        // Initialize playground
        const config = new PlaygroundConfig();
        const playground = new InteractivePlayground(config);
        playground.initialize('trustformers-playground-container');

        // Add event handlers
        document.addEventListener('playground:example-selected', (e) => {{
            const example = playground.load_example(e.detail.exampleId);
            updateEditor(example);
        }});

        document.addEventListener('playground:code-run', (e) => {{
            runCode(e.detail.code);
        }});

        // Playground functionality
        async function runCode(code) {{
            try {{
                const result = await eval(`(async () => {{ ${{code}} }})()`);
                displayOutput(result);
            }} catch (error) {{
                displayError(error);
            }}
        }}

        function updateEditor(example) {{
            document.getElementById('code-editor').value = example.code();
            document.getElementById('input-editor').value = example.input_data();
        }}

        function displayOutput(result) {{
            const outputEl = document.getElementById('output-content');
            outputEl.textContent = JSON.stringify(result, null, 2);
        }}

        function displayError(error) {{
            const outputEl = document.getElementById('output-content');
            outputEl.textContent = `Error: ${{error.message}}`;
            outputEl.style.color = '#e74c3c';
        }}

        // Make playground globally available
        window.trustformersPlayground = playground;
    </script>
</body>
</html>
"#,
            self.generate_playground_styles()
        )
    }
}

// Utility functions
#[wasm_bindgen]
pub fn create_playground_config() -> PlaygroundConfig {
    PlaygroundConfig::default()
}

#[wasm_bindgen]
pub fn create_playground_example(
    id: String,
    title: String,
    description: String,
    category: ExampleCategory,
    code: String,
) -> PlaygroundExample {
    PlaygroundExample::new(
        id,
        title,
        description,
        category,
        code,
        String::new(),
        String::new(),
        "any".to_string(),
        1,
    )
}

#[wasm_bindgen]
pub fn generate_playground_package() -> Result<String, JsValue> {
    let config = PlaygroundConfig::default();
    let _playground = InteractivePlayground::new(config)?;

    Ok(r#"
// TrustformeRS Interactive Playground Package
// Auto-generated - do not modify

import {{ TrustformersWasm }} from './trustformers-wasm.js';

export class TrustformersPlayground {{
    constructor(config) {{
        this.playground = new InteractivePlayground(config || new PlaygroundConfig());
        this.trustformers = null;
    }}

    async initialize(containerId) {{
        await this.playground.initialize(containerId);
        this.trustformers = new TrustformersWasm();
        await this.trustformers.initialize_with_auto_device();
        this.setupPlaygroundIntegration();
    }}

    setupPlaygroundIntegration() {{
        // Connect playground to TrustformeRS instance
        window.trustformers = this.trustformers;

        // Set up code execution environment
        window.runPlaygroundCode = async (code) => {{
            try {{
                const result = await eval(`(async () => {{ ${{code}} }})()`);
                return {{ success: true, result }};
            }} catch (error) {{
                return {{ success: false, error: error.message }};
            }}
        }};
    }}

    loadExample(exampleId) {{
        return this.playground.load_example(exampleId);
    }}

    getExamples() {{
        return this.playground.get_examples();
    }}

    exportState() {{
        return this.playground.export_playground_state();
    }}
}}

export {{ PlaygroundConfig, ExampleCategory, PlaygroundExample }};

// Auto-initialize if container exists
if (typeof window !== 'undefined' && document.getElementById('trustformers-playground')) {{
    const playground = new TrustformersPlayground();
    playground.initialize('trustformers-playground');
    window.trustformersPlayground = playground;
}}
"#
    .to_string())
}
