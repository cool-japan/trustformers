//! Web-based Interactive Demo
#![allow(unused_variables)]
//!
//! This example creates a web interface for TrustformeRS, allowing users to
//! interact with models through a browser. Perfect for demonstrations and
//! quick testing without CLI knowledge.

use axum::{
    extract::{Query, State},
    response::{Html, Json, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

#[derive(Clone)]
struct AppState {
    // In a real implementation, these would be actual pipelines
    #[allow(dead_code)]
    active_models: Arc<Mutex<HashMap<String, String>>>,
    session_data: Arc<Mutex<HashMap<String, SessionData>>>,
}

#[derive(Clone, Serialize, Deserialize)]
struct SessionData {
    current_task: String,
    current_model: String,
    history: Vec<InferenceResult>,
}

#[derive(Clone, Serialize, Deserialize)]
struct InferenceResult {
    timestamp: u64,
    task: String,
    input: String,
    output: serde_json::Value,
    latency_ms: u64,
}

#[derive(Deserialize)]
struct InferenceRequest {
    task: String,
    model: Option<String>,
    input: String,
    parameters: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct InferenceResponse {
    success: bool,
    result: Option<serde_json::Value>,
    error: Option<String>,
    latency_ms: u64,
    model_info: ModelInfo,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    task: String,
    parameters: u64,
    memory_mb: u64,
}

#[derive(Serialize)]
struct ModelListResponse {
    tasks: HashMap<String, Vec<ModelOption>>,
}

#[derive(Serialize)]
struct ModelOption {
    name: String,
    description: String,
    size: String,
    speed: String,
    accuracy: String,
}

#[derive(Deserialize)]
struct SessionQuery {
    session_id: Option<String>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            active_models: Arc::new(Mutex::new(HashMap::new())),
            session_data: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

// Route handlers
async fn serve_index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn serve_app_js() -> Response {
    Response::builder()
        .header("content-type", "application/javascript")
        .body(APP_JS.into())
        .unwrap()
}

async fn serve_style_css() -> Response {
    Response::builder()
        .header("content-type", "text/css")
        .body(STYLE_CSS.into())
        .unwrap()
}

async fn get_models() -> Json<ModelListResponse> {
    let mut tasks = HashMap::new();

    // Text Classification models
    tasks.insert(
        "text-classification".to_string(),
        vec![
            ModelOption {
                name: "distilbert-base-uncased-finetuned-sst-2-english".to_string(),
                description: "Fast sentiment analysis (positive/negative)".to_string(),
                size: "Small".to_string(),
                speed: "Fast".to_string(),
                accuracy: "High".to_string(),
            },
            ModelOption {
                name: "cardiffnlp/twitter-roberta-base-sentiment-latest".to_string(),
                description: "Advanced sentiment analysis for social media".to_string(),
                size: "Medium".to_string(),
                speed: "Medium".to_string(),
                accuracy: "Very High".to_string(),
            },
        ],
    );

    // Text Generation models
    tasks.insert(
        "text-generation".to_string(),
        vec![
            ModelOption {
                name: "gpt2".to_string(),
                description: "General purpose text generation".to_string(),
                size: "Medium".to_string(),
                speed: "Fast".to_string(),
                accuracy: "High".to_string(),
            },
            ModelOption {
                name: "gpt2-medium".to_string(),
                description: "Higher quality text generation".to_string(),
                size: "Large".to_string(),
                speed: "Medium".to_string(),
                accuracy: "Very High".to_string(),
            },
        ],
    );

    // Question Answering models
    tasks.insert(
        "question-answering".to_string(),
        vec![
            ModelOption {
                name: "distilbert-base-cased-distilled-squad".to_string(),
                description: "Fast question answering".to_string(),
                size: "Small".to_string(),
                speed: "Fast".to_string(),
                accuracy: "High".to_string(),
            },
            ModelOption {
                name: "deepset/roberta-base-squad2".to_string(),
                description: "Advanced question answering with unanswerable detection".to_string(),
                size: "Medium".to_string(),
                speed: "Medium".to_string(),
                accuracy: "Very High".to_string(),
            },
        ],
    );

    // Summarization models
    tasks.insert(
        "summarization".to_string(),
        vec![
            ModelOption {
                name: "facebook/bart-large-cnn".to_string(),
                description: "News article summarization".to_string(),
                size: "Large".to_string(),
                speed: "Medium".to_string(),
                accuracy: "Very High".to_string(),
            },
            ModelOption {
                name: "t5-small".to_string(),
                description: "General purpose summarization".to_string(),
                size: "Small".to_string(),
                speed: "Fast".to_string(),
                accuracy: "High".to_string(),
            },
        ],
    );

    Json(ModelListResponse { tasks })
}

async fn run_inference(
    State(state): State<AppState>,
    Json(request): Json<InferenceRequest>,
) -> Json<InferenceResponse> {
    let start_time = std::time::Instant::now();

    let model_name = request.model.unwrap_or_else(|| match request.task.as_str() {
        "text-classification" => "distilbert-base-uncased-finetuned-sst-2-english".to_string(),
        "text-generation" => "gpt2".to_string(),
        "question-answering" => "distilbert-base-cased-distilled-squad".to_string(),
        "summarization" => "facebook/bart-large-cnn".to_string(),
        _ => "distilbert-base-uncased".to_string(),
    });

    // Simulate model loading and inference
    let result = match request.task.as_str() {
        "text-classification" => simulate_classification(&request.input),
        "text-generation" => simulate_generation(&request.input, request.parameters),
        "question-answering" => simulate_qa(&request.input),
        "summarization" => simulate_summarization(&request.input),
        _ => Err("Unsupported task".to_string()),
    };

    let latency = start_time.elapsed().as_millis() as u64;

    match result {
        Ok(output) => Json(InferenceResponse {
            success: true,
            result: Some(output),
            error: None,
            latency_ms: latency + 50, // Add simulated processing time
            model_info: ModelInfo {
                name: model_name,
                task: request.task,
                parameters: 110_000_000, // Simulated parameter count
                memory_mb: 1200,
            },
        }),
        Err(error) => Json(InferenceResponse {
            success: false,
            result: None,
            error: Some(error),
            latency_ms: latency,
            model_info: ModelInfo {
                name: model_name,
                task: request.task,
                parameters: 0,
                memory_mb: 0,
            },
        }),
    }
}

async fn get_session(
    State(state): State<AppState>,
    Query(params): Query<SessionQuery>,
) -> Json<Option<SessionData>> {
    let session_id = params.session_id.unwrap_or_else(|| "default".to_string());
    let sessions = state.session_data.lock().await;
    Json(sessions.get(&session_id).cloned())
}

async fn save_session(
    State(state): State<AppState>,
    Query(params): Query<SessionQuery>,
    Json(session): Json<SessionData>,
) -> Json<bool> {
    let session_id = params.session_id.unwrap_or_else(|| "default".to_string());
    let mut sessions = state.session_data.lock().await;
    sessions.insert(session_id, session);
    Json(true)
}

// Simulation functions for demo purposes
fn simulate_classification(text: &str) -> std::result::Result<serde_json::Value, String> {
    // Simple sentiment analysis simulation
    let positive_words = [
        "good",
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "love",
        "fantastic",
    ];
    let negative_words = ["bad", "terrible", "awful", "hate", "horrible", "worst"];

    let text_lower = text.to_lowercase();
    let positive_count = positive_words.iter().filter(|&&word| text_lower.contains(word)).count();
    let negative_count = negative_words.iter().filter(|&&word| text_lower.contains(word)).count();

    let (label, score) = if positive_count > negative_count {
        ("POSITIVE", 0.85 + (positive_count as f64 * 0.1).min(0.15))
    } else if negative_count > positive_count {
        ("NEGATIVE", 0.85 + (negative_count as f64 * 0.1).min(0.15))
    } else {
        ("NEUTRAL", 0.55)
    };

    Ok(serde_json::json!({
        "label": label,
        "score": score,
        "all_scores": [
            {"label": "POSITIVE", "score": if label == "POSITIVE" { score } else { 1.0 - score }},
            {"label": "NEGATIVE", "score": if label == "NEGATIVE" { score } else { 1.0 - score }},
        ]
    }))
}

fn simulate_generation(
    prompt: &str,
    parameters: Option<serde_json::Value>,
) -> std::result::Result<serde_json::Value, String> {
    let max_length = parameters
        .as_ref()
        .and_then(|p| p.get("max_length"))
        .and_then(|v| v.as_u64())
        .unwrap_or(50) as usize;

    let continuations = vec![
        "is an incredible advancement in machine learning technology.",
        "represents the future of artificial intelligence and natural language processing.",
        "demonstrates the power of modern deep learning architectures.",
        "shows how transformer models can understand and generate human-like text.",
        "illustrates the remarkable capabilities of neural networks in language tasks.",
    ];

    let continuation = continuations[prompt.len() % continuations.len()];
    let generated_text = format!("{} {}", prompt, continuation);

    // Truncate to max_length words
    let words: Vec<&str> = generated_text.split_whitespace().collect();
    let truncated = if words.len() > max_length {
        words[..max_length].join(" ") + "..."
    } else {
        generated_text
    };

    Ok(serde_json::json!({
        "generated_text": truncated,
        "prompt": prompt,
        "parameters": {
            "max_length": max_length,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0
        }
    }))
}

fn simulate_qa(input: &str) -> std::result::Result<serde_json::Value, String> {
    // Parse context and question (expecting format: "context: ... question: ...")
    if let Some(context_start) = input.find("context:") {
        if let Some(question_start) = input.find("question:") {
            let context = input[context_start + 8..question_start].trim();
            let question = input[question_start + 9..].trim();

            // Simple keyword matching for demo
            let answer = if question.to_lowercase().contains("what") {
                if context.contains("TrustformeRS") {
                    "TrustformeRS"
                } else {
                    "machine learning"
                }
            } else if question.to_lowercase().contains("when") {
                "2024"
            } else if question.to_lowercase().contains("where") {
                "in the cloud"
            } else {
                "Yes"
            };

            return Ok(serde_json::json!({
                "answer": answer,
                "confidence": 0.89,
                "start": context.find(answer).unwrap_or(0),
                "end": context.find(answer).unwrap_or(0) + answer.len(),
                "context": context,
                "question": question
            }));
        }
    }

    Err("Please format as 'context: [text] question: [question]'".to_string())
}

fn simulate_summarization(text: &str) -> std::result::Result<serde_json::Value, String> {
    if text.len() < 50 {
        return Err("Text too short to summarize (minimum 50 characters)".to_string());
    }

    // Extract first and last sentences for simple summary
    let sentences: Vec<&str> = text.split('.').filter(|s| !s.trim().is_empty()).collect();

    let summary = if sentences.len() >= 2 {
        format!(
            "{}. {}",
            sentences[0].trim(),
            sentences.last().unwrap().trim()
        )
    } else {
        format!("Summary: {}", &text[..text.len().min(100)])
    };

    Ok(serde_json::json!({
        "summary_text": summary,
        "original_length": text.len(),
        "summary_length": summary.len(),
        "compression_ratio": summary.len() as f64 / text.len() as f64
    }))
}

#[tokio::main]
async fn main() {
    println!("🚀 Starting TrustformeRS Web Demo...");

    let state = AppState::default();

    let app = Router::new()
        .route("/", get(serve_index))
        .route("/app.js", get(serve_app_js))
        .route("/style.css", get(serve_style_css))
        .route("/api/models", get(get_models))
        .route("/api/inference", post(run_inference))
        .route("/api/session", get(get_session))
        .route("/api/session", post(save_session))
        .layer(ServiceBuilder::new().layer(CorsLayer::permissive()))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("Failed to bind to port 3000");

    println!("🌐 Web demo running at: http://localhost:3000");
    println!("📱 Open this URL in your browser to try TrustformeRS!");

    axum::serve(listener, app).await.expect("Failed to start web server");
}

// Embedded HTML, CSS, and JavaScript
const INDEX_HTML: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS Interactive Demo</title>
    <link rel="stylesheet" href="/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 TrustformeRS Interactive Demo</h1>
            <p>Explore the power of Rust-based machine learning in your browser!</p>
        </header>

        <main>
            <div class="task-selector">
                <h2>Select a Task</h2>
                <div class="task-buttons">
                    <button class="task-btn" data-task="text-classification">📊 Text Classification</button>
                    <button class="task-btn" data-task="text-generation">✍️ Text Generation</button>
                    <button class="task-btn" data-task="question-answering">❓ Question Answering</button>
                    <button class="task-btn" data-task="summarization">📄 Summarization</button>
                </div>
            </div>

            <div class="model-selector" style="display: none;">
                <h3>Choose a Model</h3>
                <select id="model-select">
                    <option value="">Loading models...</option>
                </select>
                <div class="model-info" id="model-info"></div>
            </div>

            <div class="input-section" style="display: none;">
                <h3>Input</h3>
                <div id="input-container"></div>
                <button id="run-inference" class="primary-btn">🔄 Run Inference</button>
            </div>

            <div class="results-section" style="display: none;">
                <h3>Results</h3>
                <div id="results-container"></div>
                <div class="performance-metrics" id="performance-metrics"></div>
            </div>

            <div class="examples-section">
                <h3>💡 Example Inputs</h3>
                <div id="examples-container"></div>
            </div>
        </main>

        <footer>
            <div class="session-controls">
                <button id="save-session">💾 Save Session</button>
                <button id="load-session">📂 Load Session</button>
                <button id="clear-history">🗑️ Clear History</button>
            </div>

            <div class="history-section">
                <h3>📈 Inference History</h3>
                <div id="history-container"></div>
            </div>
        </footer>
    </div>

    <script src="/app.js"></script>
</body>
</html>
"#;

const STYLE_CSS: &str = r#"
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

header {
    text-align: center;
    margin-bottom: 40px;
    color: white;
}

header h1 {
    font-size: 2.5em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

header p {
    font-size: 1.2em;
    opacity: 0.9;
}

main {
    background: white;
    border-radius: 12px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    margin-bottom: 30px;
}

.task-selector h2 {
    margin-bottom: 20px;
    color: #4a5568;
}

.task-buttons {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 15px;
    margin-bottom: 30px;
}

.task-btn {
    padding: 20px;
    border: 2px solid #e2e8f0;
    background: white;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 1.1em;
    font-weight: 500;
}

.task-btn:hover {
    border-color: #667eea;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
}

.task-btn.active {
    background: #667eea;
    color: white;
    border-color: #667eea;
}

.model-selector, .input-section, .results-section {
    margin-top: 30px;
    padding-top: 30px;
    border-top: 1px solid #e2e8f0;
}

#model-select {
    width: 100%;
    padding: 12px;
    border: 2px solid #e2e8f0;
    border-radius: 6px;
    font-size: 1em;
    margin-bottom: 15px;
}

.model-info {
    background: #f7fafc;
    padding: 15px;
    border-radius: 6px;
    border-left: 4px solid #667eea;
}

textarea, input[type="text"] {
    width: 100%;
    padding: 12px;
    border: 2px solid #e2e8f0;
    border-radius: 6px;
    font-size: 1em;
    margin-bottom: 15px;
    resize: vertical;
}

.primary-btn {
    background: #667eea;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 6px;
    font-size: 1.1em;
    cursor: pointer;
    transition: background 0.3s ease;
}

.primary-btn:hover {
    background: #5a67d8;
}

.primary-btn:disabled {
    background: #a0aec0;
    cursor: not-allowed;
}

.results-container {
    background: #f7fafc;
    padding: 20px;
    border-radius: 6px;
    margin-bottom: 15px;
}

.performance-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin-top: 15px;
}

.metric {
    background: white;
    padding: 15px;
    border-radius: 6px;
    text-align: center;
    border: 1px solid #e2e8f0;
}

.metric-value {
    font-size: 1.5em;
    font-weight: bold;
    color: #667eea;
}

.metric-label {
    font-size: 0.9em;
    color: #718096;
    margin-top: 5px;
}

.examples-section {
    margin-top: 30px;
    padding-top: 30px;
    border-top: 1px solid #e2e8f0;
}

.example-item {
    background: #f7fafc;
    padding: 15px;
    border-radius: 6px;
    margin-bottom: 10px;
    cursor: pointer;
    transition: background 0.3s ease;
}

.example-item:hover {
    background: #edf2f7;
}

footer {
    background: white;
    border-radius: 12px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.session-controls {
    display: flex;
    gap: 15px;
    margin-bottom: 30px;
    flex-wrap: wrap;
}

.session-controls button {
    padding: 10px 20px;
    border: 2px solid #667eea;
    background: white;
    color: #667eea;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.session-controls button:hover {
    background: #667eea;
    color: white;
}

.history-item {
    background: #f7fafc;
    padding: 15px;
    border-radius: 6px;
    margin-bottom: 10px;
    border-left: 4px solid #667eea;
}

.history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.history-timestamp {
    font-size: 0.9em;
    color: #718096;
}

.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.error {
    background: #fed7d7;
    color: #c53030;
    padding: 15px;
    border-radius: 6px;
    margin: 15px 0;
}

.success {
    background: #c6f6d5;
    color: #22543d;
    padding: 15px;
    border-radius: 6px;
    margin: 15px 0;
}

@media (max-width: 768px) {
    .container {
        padding: 10px;
    }

    header h1 {
        font-size: 2em;
    }

    .task-buttons {
        grid-template-columns: 1fr;
    }

    .session-controls {
        flex-direction: column;
    }
}
"#;

const APP_JS: &str = r#"
class TrustformersDemoApp {
    constructor() {
        this.currentTask = null;
        this.currentModel = null;
        this.models = {};
        this.history = [];

        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadModels();
        this.updateExamples();
    }

    setupEventListeners() {
        // Task selection
        document.querySelectorAll('.task-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.selectTask(e.target.dataset.task);
            });
        });

        // Model selection
        document.getElementById('model-select').addEventListener('change', (e) => {
            this.selectModel(e.target.value);
        });

        // Run inference
        document.getElementById('run-inference').addEventListener('click', () => {
            this.runInference();
        });

        // Session controls
        document.getElementById('save-session').addEventListener('click', () => {
            this.saveSession();
        });

        document.getElementById('load-session').addEventListener('click', () => {
            this.loadSession();
        });

        document.getElementById('clear-history').addEventListener('click', () => {
            this.clearHistory();
        });
    }

    async loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            this.models = data.tasks;
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    selectTask(task) {
        this.currentTask = task;

        // Update UI
        document.querySelectorAll('.task-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-task="${task}"]`).classList.add('active');

        // Show model selector
        this.updateModelSelector();
        document.querySelector('.model-selector').style.display = 'block';

        // Update examples
        this.updateExamples();
    }

    updateModelSelector() {
        const select = document.getElementById('model-select');
        const taskModels = this.models[this.currentTask] || [];

        select.innerHTML = '<option value="">Select a model...</option>';

        taskModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.name;
            select.appendChild(option);
        });

        // Auto-select first model
        if (taskModels.length > 0) {
            select.value = taskModels[0].name;
            this.selectModel(taskModels[0].name);
        }
    }

    selectModel(modelName) {
        this.currentModel = modelName;

        const model = this.models[this.currentTask]?.find(m => m.name === modelName);
        if (model) {
            document.getElementById('model-info').innerHTML = `
                <strong>${model.name}</strong><br>
                ${model.description}<br>
                <small>Size: ${model.size} | Speed: ${model.speed} | Accuracy: ${model.accuracy}</small>
            `;
        }

        // Show input section
        this.updateInputSection();
        document.querySelector('.input-section').style.display = 'block';
    }

    updateInputSection() {
        const container = document.getElementById('input-container');

        switch (this.currentTask) {
            case 'text-classification':
                container.innerHTML = `
                    <textarea id="text-input" placeholder="Enter text to classify..." rows="3"></textarea>
                `;
                break;

            case 'text-generation':
                container.innerHTML = `
                    <textarea id="text-input" placeholder="Enter prompt for text generation..." rows="3"></textarea>
                    <div style="display: flex; gap: 15px; margin-top: 10px;">
                        <label>Max Length: <input type="number" id="max-length" value="50" min="10" max="200"></label>
                        <label>Temperature: <input type="number" id="temperature" value="1.0" min="0.1" max="2.0" step="0.1"></label>
                    </div>
                `;
                break;

            case 'question-answering':
                container.innerHTML = `
                    <textarea id="context-input" placeholder="Enter context text..." rows="4"></textarea>
                    <input type="text" id="question-input" placeholder="Enter your question...">
                `;
                break;

            case 'summarization':
                container.innerHTML = `
                    <textarea id="text-input" placeholder="Enter text to summarize..." rows="5"></textarea>
                `;
                break;
        }
    }

    async runInference() {
        const btn = document.getElementById('run-inference');
        btn.disabled = true;
        btn.innerHTML = '<span class="loading"></span> Processing...';

        try {
            const input = this.getInputData();
            if (!input) {
                throw new Error('Please provide input text');
            }

            const request = {
                task: this.currentTask,
                model: this.currentModel,
                input: input,
                parameters: this.getParameters()
            };

            const response = await fetch('/api/inference', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request)
            });

            const result = await response.json();
            this.displayResults(result);

            if (result.success) {
                this.addToHistory({
                    timestamp: Date.now(),
                    task: this.currentTask,
                    input: input,
                    output: result.result,
                    latency_ms: result.latency_ms
                });
            }

        } catch (error) {
            this.displayError(error.message);
        } finally {
            btn.disabled = false;
            btn.innerHTML = '🔄 Run Inference';
        }
    }

    getInputData() {
        switch (this.currentTask) {
            case 'text-classification':
            case 'text-generation':
            case 'summarization':
                return document.getElementById('text-input')?.value.trim();

            case 'question-answering':
                const context = document.getElementById('context-input')?.value.trim();
                const question = document.getElementById('question-input')?.value.trim();
                if (!context || !question) return null;
                return `context: ${context} question: ${question}`;

            default:
                return null;
        }
    }

    getParameters() {
        const params = {};

        if (this.currentTask === 'text-generation') {
            const maxLength = document.getElementById('max-length')?.value;
            const temperature = document.getElementById('temperature')?.value;

            if (maxLength) params.max_length = parseInt(maxLength);
            if (temperature) params.temperature = parseFloat(temperature);
        }

        return Object.keys(params).length > 0 ? params : null;
    }

    displayResults(result) {
        const container = document.getElementById('results-container');

        if (!result.success) {
            container.innerHTML = `<div class="error">Error: ${result.error}</div>`;
            document.querySelector('.results-section').style.display = 'block';
            return;
        }

        let html = '';

        switch (this.currentTask) {
            case 'text-classification':
                html = this.formatClassificationResult(result.result);
                break;
            case 'text-generation':
                html = this.formatGenerationResult(result.result);
                break;
            case 'question-answering':
                html = this.formatQAResult(result.result);
                break;
            case 'summarization':
                html = this.formatSummarizationResult(result.result);
                break;
        }

        container.innerHTML = html;

        // Update performance metrics
        this.updatePerformanceMetrics(result);

        document.querySelector('.results-section').style.display = 'block';
    }

    formatClassificationResult(result) {
        const mainScore = (result.score * 100).toFixed(1);

        let html = `
            <div class="success">
                <strong>Classification:</strong> ${result.label} (${mainScore}% confidence)
            </div>
        `;

        if (result.all_scores) {
            html += '<h4>All Scores:</h4>';
            result.all_scores.forEach(score => {
                const percentage = (score.score * 100).toFixed(1);
                const barWidth = percentage;
                html += `
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>${score.label}</span>
                            <span>${percentage}%</span>
                        </div>
                        <div style="background: #e2e8f0; height: 8px; border-radius: 4px; margin-top: 5px;">
                            <div style="background: #667eea; height: 100%; width: ${barWidth}%; border-radius: 4px;"></div>
                        </div>
                    </div>
                `;
            });
        }

        return html;
    }

    formatGenerationResult(result) {
        return `
            <div class="success">
                <strong>Generated Text:</strong><br>
                <p style="margin-top: 10px; font-style: italic;">"${result.generated_text}"</p>
            </div>
            <div style="margin-top: 15px; font-size: 0.9em; color: #718096;">
                <strong>Parameters:</strong> Max Length: ${result.parameters.max_length},
                Temperature: ${result.parameters.temperature}
            </div>
        `;
    }

    formatQAResult(result) {
        return `
            <div class="success">
                <strong>Answer:</strong> ${result.answer}<br>
                <strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%
            </div>
            <div style="margin-top: 15px;">
                <strong>Context:</strong><br>
                <p style="background: #f7fafc; padding: 10px; border-radius: 4px; margin-top: 5px;">
                    ${result.context}
                </p>
            </div>
        `;
    }

    formatSummarizationResult(result) {
        const compressionPercentage = ((1 - result.compression_ratio) * 100).toFixed(1);

        return `
            <div class="success">
                <strong>Summary:</strong><br>
                <p style="margin-top: 10px; font-style: italic;">"${result.summary_text}"</p>
            </div>
            <div style="margin-top: 15px; font-size: 0.9em; color: #718096;">
                <strong>Compression:</strong> ${result.original_length} → ${result.summary_length} characters
                (${compressionPercentage}% reduction)
            </div>
        `;
    }

    updatePerformanceMetrics(result) {
        const container = document.getElementById('performance-metrics');

        container.innerHTML = `
            <div class="metric">
                <div class="metric-value">${result.latency_ms}ms</div>
                <div class="metric-label">Latency</div>
            </div>
            <div class="metric">
                <div class="metric-value">${(result.model_info.parameters / 1e6).toFixed(0)}M</div>
                <div class="metric-label">Parameters</div>
            </div>
            <div class="metric">
                <div class="metric-value">${result.model_info.memory_mb}MB</div>
                <div class="metric-label">Memory</div>
            </div>
            <div class="metric">
                <div class="metric-value">${(1000 / result.latency_ms).toFixed(1)}</div>
                <div class="metric-label">Inferences/sec</div>
            </div>
        `;
    }

    displayError(message) {
        const container = document.getElementById('results-container');
        container.innerHTML = `<div class="error">Error: ${message}</div>`;
        document.querySelector('.results-section').style.display = 'block';
    }

    updateExamples() {
        const container = document.getElementById('examples-container');

        const examples = {
            'text-classification': [
                'I love this new machine learning library!',
                'This software is terrible and slow.',
                'The weather is nice today.',
                'TrustformeRS makes ML development so much easier!'
            ],
            'text-generation': [
                'The future of artificial intelligence',
                'Once upon a time in a distant galaxy',
                'Climate change is a serious issue that',
                'The benefits of renewable energy include'
            ],
            'question-answering': [
                'context: TrustformeRS is a high-performance machine learning library written in Rust. It provides state-of-the-art transformer models with excellent performance and memory efficiency. question: What is TrustformeRS?',
                'context: Machine learning models require careful tuning of hyperparameters to achieve optimal performance. question: What do ML models need for optimal performance?'
            ],
            'summarization': [
                'Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.',
                'Climate change refers to long-term shifts and alterations in global or regional climate patterns. Since the mid-20th century, climate change has been primarily attributed to the increased levels of atmospheric carbon dioxide produced by the use of fossil fuels. The consequences of climate change include rising sea levels, extreme weather events, and changes in precipitation patterns that affect agriculture and water resources worldwide.'
            ]
        };

        const taskExamples = examples[this.currentTask] || [];

        if (taskExamples.length === 0) {
            container.innerHTML = '<p>No examples available for this task.</p>';
            return;
        }

        container.innerHTML = taskExamples.map(example => `
            <div class="example-item" onclick="app.useExample('${example.replace(/'/g, "\\'")}')">
                ${example.length > 100 ? example.substring(0, 100) + '...' : example}
            </div>
        `).join('');
    }

    useExample(example) {
        switch (this.currentTask) {
            case 'text-classification':
            case 'text-generation':
            case 'summarization':
                document.getElementById('text-input').value = example;
                break;

            case 'question-answering':
                const parts = example.split(' question: ');
                if (parts.length === 2) {
                    document.getElementById('context-input').value = parts[0].replace('context: ', '');
                    document.getElementById('question-input').value = parts[1];
                }
                break;
        }
    }

    addToHistory(item) {
        this.history.unshift(item);
        if (this.history.length > 10) {
            this.history = this.history.slice(0, 10);
        }
        this.updateHistoryDisplay();
    }

    updateHistoryDisplay() {
        const container = document.getElementById('history-container');

        if (this.history.length === 0) {
            container.innerHTML = '<p>No inference history yet. Try running some examples!</p>';
            return;
        }

        container.innerHTML = this.history.map(item => `
            <div class="history-item">
                <div class="history-header">
                    <strong>${item.task}</strong>
                    <span class="history-timestamp">${new Date(item.timestamp).toLocaleTimeString()}</span>
                </div>
                <div><strong>Input:</strong> ${item.input.substring(0, 100)}${item.input.length > 100 ? '...' : ''}</div>
                <div><strong>Latency:</strong> ${item.latency_ms}ms</div>
            </div>
        `).join('');
    }

    async saveSession() {
        try {
            const sessionData = {
                current_task: this.currentTask,
                current_model: this.currentModel,
                history: this.history
            };

            await fetch('/api/session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(sessionData)
            });

            alert('Session saved successfully!');
        } catch (error) {
            alert('Failed to save session: ' + error.message);
        }
    }

    async loadSession() {
        try {
            const response = await fetch('/api/session');
            const sessionData = await response.json();

            if (sessionData) {
                this.currentTask = sessionData.current_task;
                this.currentModel = sessionData.current_model;
                this.history = sessionData.history || [];

                // Update UI
                if (this.currentTask) {
                    this.selectTask(this.currentTask);
                    if (this.currentModel) {
                        document.getElementById('model-select').value = this.currentModel;
                        this.selectModel(this.currentModel);
                    }
                }

                this.updateHistoryDisplay();
                alert('Session loaded successfully!');
            } else {
                alert('No saved session found.');
            }
        } catch (error) {
            alert('Failed to load session: ' + error.message);
        }
    }

    clearHistory() {
        this.history = [];
        this.updateHistoryDisplay();
    }
}

// Initialize the app when the page loads
const app = new TrustformersDemoApp();
"#;
