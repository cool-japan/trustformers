//! React component bindings for TrustFormer WASM
//!
//! This module provides React component wrappers and hooks for easy integration
//! of TrustFormer WASM functionality into React applications.

#![allow(dead_code)]

use js_sys::Object;
use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// React hook state for model loading
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ModelLoadingState {
    is_loading: bool,
    progress: f64,
    error: Option<String>,
    model_loaded: bool,
}

/// React hook state for inference
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct InferenceState {
    is_inferring: bool,
    result: Option<String>,
    error: Option<String>,
    inference_time_ms: f64,
}

/// Configuration for React components
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ReactConfig {
    auto_load_model: bool,
    show_progress: bool,
    enable_streaming: bool,
    debug_mode: bool,
    model_url: String,
    fallback_message: String,
}

/// React component factory for TrustFormer
#[wasm_bindgen]
pub struct ReactComponentFactory {
    config: ReactConfig,
    component_registry: Vec<ComponentDefinition>,
}

/// Component definition for React integration
#[derive(Debug, Clone)]
struct ComponentDefinition {
    name: String,
    props_schema: String,
    component_type: ComponentType,
    render_function: String,
}

/// Types of React components
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComponentType {
    /// Text generation component
    TextGenerator,
    /// Chat interface component
    ChatInterface,
    /// Model loading component
    ModelLoader,
    /// Inference progress component
    InferenceProgress,
    /// Error boundary component
    ErrorBoundary,
    /// Settings panel component
    SettingsPanel,
}

/// React hook manager
#[wasm_bindgen]
pub struct ReactHookManager {
    hooks: Vec<HookDefinition>,
    state_managers: Vec<StateManager>,
}

/// Hook definition
#[derive(Debug, Clone)]
struct HookDefinition {
    name: String,
    hook_type: HookType,
    dependencies: Vec<String>,
    implementation: String,
}

/// Types of React hooks
#[derive(Debug, Clone, Copy, PartialEq)]
enum HookType {
    ModelLoader,
    Inference,
    StreamingGeneration,
    ModelState,
    ErrorHandling,
}

/// State manager for React hooks
#[derive(Debug, Clone)]
struct StateManager {
    id: String,
    state_type: String,
    initial_state: String,
    reducers: Vec<String>,
}

#[wasm_bindgen]
impl ReactConfig {
    /// Create new React configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> ReactConfig {
        ReactConfig {
            auto_load_model: true,
            show_progress: true,
            enable_streaming: false,
            debug_mode: false,
            model_url: String::new(),
            fallback_message: "Loading AI model...".to_string(),
        }
    }

    /// Set auto-load model option
    pub fn set_auto_load_model(&mut self, auto_load: bool) {
        self.auto_load_model = auto_load;
    }

    /// Set show progress option
    pub fn set_show_progress(&mut self, show_progress: bool) {
        self.show_progress = show_progress;
    }

    /// Set streaming enabled
    pub fn set_enable_streaming(&mut self, enable_streaming: bool) {
        self.enable_streaming = enable_streaming;
    }

    /// Set debug mode
    pub fn set_debug_mode(&mut self, debug_mode: bool) {
        self.debug_mode = debug_mode;
    }

    /// Set model URL
    pub fn set_model_url(&mut self, url: String) {
        self.model_url = url;
    }

    /// Set fallback message
    pub fn set_fallback_message(&mut self, message: String) {
        self.fallback_message = message;
    }

    #[wasm_bindgen(getter)]
    pub fn auto_load_model(&self) -> bool {
        self.auto_load_model
    }

    #[wasm_bindgen(getter)]
    pub fn show_progress(&self) -> bool {
        self.show_progress
    }

    #[wasm_bindgen(getter)]
    pub fn enable_streaming(&self) -> bool {
        self.enable_streaming
    }

    #[wasm_bindgen(getter)]
    pub fn debug_mode(&self) -> bool {
        self.debug_mode
    }

    #[wasm_bindgen(getter)]
    pub fn model_url(&self) -> String {
        self.model_url.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn fallback_message(&self) -> String {
        self.fallback_message.clone()
    }
}

impl Default for ReactConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl ModelLoadingState {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ModelLoadingState {
        ModelLoadingState {
            is_loading: false,
            progress: 0.0,
            error: None,
            model_loaded: false,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn is_loading(&self) -> bool {
        self.is_loading
    }

    #[wasm_bindgen(getter)]
    pub fn progress(&self) -> f64 {
        self.progress
    }

    #[wasm_bindgen(getter)]
    pub fn model_loaded(&self) -> bool {
        self.model_loaded
    }

    #[wasm_bindgen(getter)]
    pub fn error(&self) -> Option<String> {
        self.error.clone()
    }

    pub fn set_loading(&mut self, is_loading: bool) {
        self.is_loading = is_loading;
    }

    pub fn set_progress(&mut self, progress: f64) {
        self.progress = progress;
    }

    pub fn set_model_loaded(&mut self, loaded: bool) {
        self.model_loaded = loaded;
    }

    pub fn set_error(&mut self, error: Option<String>) {
        self.error = error;
    }
}

impl Default for ModelLoadingState {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl InferenceState {
    #[wasm_bindgen(constructor)]
    pub fn new() -> InferenceState {
        InferenceState {
            is_inferring: false,
            result: None,
            error: None,
            inference_time_ms: 0.0,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn is_inferring(&self) -> bool {
        self.is_inferring
    }

    #[wasm_bindgen(getter)]
    pub fn result(&self) -> Option<String> {
        self.result.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn error(&self) -> Option<String> {
        self.error.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn inference_time_ms(&self) -> f64 {
        self.inference_time_ms
    }

    pub fn set_inferring(&mut self, is_inferring: bool) {
        self.is_inferring = is_inferring;
    }

    pub fn set_result(&mut self, result: Option<String>) {
        self.result = result;
    }

    pub fn set_error(&mut self, error: Option<String>) {
        self.error = error;
    }

    pub fn set_inference_time(&mut self, time_ms: f64) {
        self.inference_time_ms = time_ms;
    }
}

impl Default for InferenceState {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl ReactComponentFactory {
    /// Create new React component factory
    #[wasm_bindgen(constructor)]
    pub fn new(config: ReactConfig) -> ReactComponentFactory {
        ReactComponentFactory {
            config,
            component_registry: Vec::new(),
        }
    }

    /// Generate Text Generator React component
    pub fn generate_text_generator_component(&self) -> String {
        format!(
            r#"
import React, {{ useState, useEffect, useCallback }} from 'react';
import {{ useTrustFormerModel, useTrustFormerInference }} from './hooks';

const TrustFormerTextGenerator = ({{
  modelUrl = '{}',
  placeholder = 'Enter your prompt...',
  maxLength = 100,
  temperature = 0.7,
  onGenerate,
  showProgress = {},
  style = {{}},
  className = ''
}}) => {{
  const [prompt, setPrompt] = useState('');
  const [generations, setGenerations] = useState([]);

  const {{ modelState, loadModel }} = useTrustFormerModel({{
    modelUrl,
    autoLoad: {}
  }});

  const {{ inferenceState, generateText }} = useTrustFormerInference();

  useEffect(() => {{
    if (modelUrl && !modelState.model_loaded && !modelState.is_loading) {{
      loadModel();
    }}
  }}, [modelUrl, modelState.model_loaded, modelState.is_loading, loadModel]);

  const handleGenerate = useCallback(async () => {{
    if (!prompt.trim() || !modelState.model_loaded) return;

    try {{
      const result = await generateText(prompt, {{
        maxLength,
        temperature
      }});

      const newGeneration = {{
        id: Date.now(),
        prompt,
        result: result.text,
        timestamp: new Date().toISOString(),
        inferenceTime: result.inferenceTime
      }};

      setGenerations(prev => [newGeneration, ...prev]);
      setPrompt('');

      if (onGenerate) {{
        onGenerate(newGeneration);
      }}
    }} catch (error) {{
      console.error('Generation failed:', error);
    }}
  }}, [prompt, modelState.model_loaded, generateText, maxLength, temperature, onGenerate]);

  const handleKeyPress = useCallback((e) => {{
    if (e.key === 'Enter' && e.ctrlKey) {{
      handleGenerate();
    }}
  }}, [handleGenerate]);

  if (modelState.error) {{
    return (
      <div className={{`trustformer-error ${{className}}`}} style={{style}}>
        <p>Error loading model: {{modelState.error}}</p>
        <button onClick={{loadModel}}>Retry</button>
      </div>
    );
  }}

  return (
    <div className={{`trustformer-text-generator ${{className}}`}} style={{style}}>
      {{showProgress && modelState.is_loading && (
        <div className="trustformer-loading">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{{{ width: `${{modelState.progress}}%` }}}}
            />
          </div>
          <p>Loading model... {{Math.round(modelState.progress)}}%</p>
        </div>
      )}}

      {{modelState.model_loaded && (
        <>
          <div className="input-section">
            <textarea
              value={{prompt}}
              onChange={{(e) => setPrompt(e.target.value)}}
              onKeyPress={{handleKeyPress}}
              placeholder={{placeholder}}
              disabled={{inferenceState.is_inferring}}
              rows={{3}}
            />
            <button
              onClick={{handleGenerate}}
              disabled={{!prompt.trim() || inferenceState.is_inferring}}
            >
              {{inferenceState.is_inferring ? 'Generating...' : 'Generate'}}
            </button>
          </div>

          <div className="generations">
            {{generations.map((gen) => (
              <div key={{gen.id}} className="generation-item">
                <div className="prompt">
                  <strong>Prompt:</strong> {{gen.prompt}}
                </div>
                <div className="result">
                  <strong>Result:</strong> {{gen.result}}
                </div>
                <div className="meta">
                  <small>
                    Generated in {{gen.inferenceTime}}ms at {{new Date(gen.timestamp).toLocaleTimeString()}}
                  </small>
                </div>
              </div>
            ))}}
          </div>
        </>
      )}}
    </div>
  );
}};

export default TrustFormerTextGenerator;
"#,
            self.config.model_url, self.config.show_progress, self.config.auto_load_model
        )
    }

    /// Generate Chat Interface React component
    pub fn generate_chat_interface_component(&self) -> String {
        format!(
            r#"
import React, {{ useState, useEffect, useCallback, useRef }} from 'react';
import {{ useTrustFormerModel, useTrustFormerInference }} from './hooks';

const TrustFormerChatInterface = ({{
  modelUrl = '{}',
  placeholder = 'Type your message...',
  systemPrompt = 'You are a helpful AI assistant.',
  maxTokens = 150,
  temperature = 0.7,
  onMessage,
  style = {{}},
  className = ''
}}) => {{
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const {{ modelState, loadModel }} = useTrustFormerModel({{
    modelUrl,
    autoLoad: {}
  }});

  const {{ inferenceState, generateText }} = useTrustFormerInference();

  const scrollToBottom = () => {{
    messagesEndRef.current?.scrollIntoView({{ behavior: 'smooth' }});
  }};

  useEffect(() => {{
    scrollToBottom();
  }}, [messages]);

  useEffect(() => {{
    if (modelUrl && !modelState.model_loaded && !modelState.is_loading) {{
      loadModel();
    }}
  }}, [modelUrl, modelState.model_loaded, modelState.is_loading, loadModel]);

  const handleSendMessage = useCallback(async () => {{
    if (!input.trim() || !modelState.model_loaded || inferenceState.is_inferring) return;

    const userMessage = {{
      id: Date.now(),
      type: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString()
    }};

    setMessages(prev => [...prev, userMessage]);
    setInput('');

    // Prepare conversation context
    const conversationHistory = messages.map(msg =>
      `${{msg.type === 'user' ? 'User' : 'Assistant'}}: ${{msg.content}}`
    ).join('\\n');

    const prompt = `${{systemPrompt}}\\n\\n${{conversationHistory}}\\nUser: ${{userMessage.content}}\\nAssistant:`;

    try {{
      const result = await generateText(prompt, {{
        maxLength: maxTokens,
        temperature
      }});

      const assistantMessage = {{
        id: Date.now() + 1,
        type: 'assistant',
        content: result.text.trim(),
        timestamp: new Date().toISOString(),
        inferenceTime: result.inferenceTime
      }};

      setMessages(prev => [...prev, assistantMessage]);

      if (onMessage) {{
        onMessage({{ user: userMessage, assistant: assistantMessage }});
      }}
    }} catch (error) {{
      console.error('Chat inference failed:', error);
      const errorMessage = {{
        id: Date.now() + 1,
        type: 'error',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString()
      }};
      setMessages(prev => [...prev, errorMessage]);
    }}
  }}, [input, modelState.model_loaded, inferenceState.is_inferring, messages, generateText, systemPrompt, maxTokens, temperature, onMessage]);

  const handleKeyPress = useCallback((e) => {{
    if (e.key === 'Enter' && !e.shiftKey) {{
      e.preventDefault();
      handleSendMessage();
    }}
  }}, [handleSendMessage]);

  const clearChat = useCallback(() => {{
    setMessages([]);
  }}, []);

  if (modelState.error) {{
    return (
      <div className={{`trustformer-chat-error ${{className}}`}} style={{style}}>
        <p>Error loading model: {{modelState.error}}</p>
        <button onClick={{loadModel}}>Retry</button>
      </div>
    );
  }}

  return (
    <div className={{`trustformer-chat-interface ${{className}}`}} style={{style}}>
      {{modelState.is_loading && (
        <div className="chat-loading">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{{{ width: `${{modelState.progress}}%` }}}}
            />
          </div>
          <p>{}... {{Math.round(modelState.progress)}}%</p>
        </div>
      )}}

      {{modelState.model_loaded && (
        <>
          <div className="chat-header">
            <h3>AI Assistant</h3>
            <button onClick={{clearChat}} disabled={{messages.length === 0}}>
              Clear Chat
            </button>
          </div>

          <div className="chat-messages">
            {{messages.map((message) => (
              <div key={{message.id}} className={{`message message-${{message.type}}`}}>
                <div className="message-content">
                  {{message.content}}
                </div>
                <div className="message-meta">
                  <small>
                    {{new Date(message.timestamp).toLocaleTimeString()}}
                    {{message.inferenceTime && ` • ${{message.inferenceTime}}ms`}}
                  </small>
                </div>
              </div>
            ))}}
            {{inferenceState.is_inferring && (
              <div className="message message-assistant typing">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}}
            <div ref={{messagesEndRef}} />
          </div>

          <div className="chat-input">
            <textarea
              value={{input}}
              onChange={{(e) => setInput(e.target.value)}}
              onKeyPress={{handleKeyPress}}
              placeholder={{placeholder}}
              disabled={{inferenceState.is_inferring}}
              rows={{2}}
            />
            <button
              onClick={{handleSendMessage}}
              disabled={{!input.trim() || inferenceState.is_inferring}}
            >
              Send
            </button>
          </div>
        </>
      )}}
    </div>
  );
}};

export default TrustFormerChatInterface;
"#,
            self.config.model_url, self.config.auto_load_model, self.config.fallback_message
        )
    }

    /// Generate React hooks
    pub fn generate_react_hooks(&self) -> String {
        format!(
            r#"
import {{ useState, useEffect, useCallback, useRef }} from 'react';

// Import the WASM module
let wasmModule = null;
const initWasm = async () => {{
  if (!wasmModule) {{
    wasmModule = await import('trustformers-wasm');
    await wasmModule.default();
  }}
  return wasmModule;
}};

// Hook for managing TrustFormer model loading
export const useTrustFormerModel = ({{
  modelUrl,
  autoLoad = true,
  onLoadComplete,
  onLoadError
}} = {{}}) => {{
  const [modelState, setModelState] = useState({{
    is_loading: false,
    progress: 0,
    error: null,
    model_loaded: false
  }});

  const sessionRef = useRef(null);

  const loadModel = useCallback(async () => {{
    if (!modelUrl) {{
      setModelState(prev => ({{
        ...prev,
        error: 'No model URL provided'
      }}));
      return;
    }}

    setModelState(prev => ({{
      ...prev,
      is_loading: true,
      error: null,
      progress: 0
    }}));

    try {{
      const wasm = await initWasm();

      // Create inference session
      sessionRef.current = new wasm.InferenceSession('transformer');

      // Initialize with auto device selection
      await sessionRef.current.initialize_with_auto_device();

      // Enable debug logging if configured
      if ({}) {{
        const debugConfig = new wasm.DebugConfig();
        debugConfig.set_log_level(wasm.LogLevel.Info);
        sessionRef.current.enable_debug_logging(debugConfig);
      }}

      // Load model from URL or cache
      await sessionRef.current.load_model_with_cache(
        'model_' + modelUrl.split('/').pop(),
        modelUrl,
        'TrustFormer Model',
        'transformer',
        '1.0.0'
      );

      setModelState(prev => ({{
        ...prev,
        is_loading: false,
        model_loaded: true,
        progress: 100
      }}));

      if (onLoadComplete) {{
        onLoadComplete(sessionRef.current);
      }}

    }} catch (error) {{
      console.error('Model loading failed:', error);
      const errorMessage = error.message || 'Failed to load model';

      setModelState(prev => ({{
        ...prev,
        is_loading: false,
        error: errorMessage
      }}));

      if (onLoadError) {{
        onLoadError(error);
      }}
    }}
  }}, [modelUrl, onLoadComplete, onLoadError]);

  useEffect(() => {{
    if (autoLoad && modelUrl && !modelState.model_loaded && !modelState.is_loading) {{
      loadModel();
    }}
  }}, [autoLoad, modelUrl, modelState.model_loaded, modelState.is_loading, loadModel]);

  return {{
    modelState,
    loadModel,
    session: sessionRef.current
  }};
}};

// Hook for managing inference
export const useTrustFormerInference = () => {{
  const [inferenceState, setInferenceState] = useState({{
    is_inferring: false,
    result: null,
    error: null,
    inference_time_ms: 0
  }});

  const generateText = useCallback(async (prompt, options = {{}}) => {{
    setInferenceState(prev => ({{
      ...prev,
      is_inferring: true,
      error: null
    }}));

    const startTime = performance.now();

    try {{
      const wasm = await initWasm();

      // Create a simple tensor for the prompt (this is simplified)
      // In a real implementation, you'd tokenize the prompt properly
      const inputTensor = new wasm.WasmTensor([prompt.length], new Float32Array(prompt.length));

      // Perform inference (simplified)
      const result = inputTensor; // In reality, this would be actual model inference

      const endTime = performance.now();
      const inferenceTime = endTime - startTime;

      // Simulate text generation result
      const generatedText = `Generated response for: "${{prompt}}"`;

      const inferenceResult = {{
        text: generatedText,
        inferenceTime: Math.round(inferenceTime),
        tokenCount: generatedText.split(' ').length
      }};

      setInferenceState(prev => ({{
        ...prev,
        is_inferring: false,
        result: inferenceResult,
        inference_time_ms: inferenceTime
      }}));

      return inferenceResult;

    }} catch (error) {{
      console.error('Inference failed:', error);
      const errorMessage = error.message || 'Inference failed';

      setInferenceState(prev => ({{
        ...prev,
        is_inferring: false,
        error: errorMessage
      }}));

      throw error;
    }}
  }}, []);

  return {{
    inferenceState,
    generateText
  }};
}};

// Hook for streaming generation
export const useTrustFormerStreaming = () => {{
  const [streamState, setStreamState] = useState({{
    is_streaming: false,
    partial_result: '',
    complete_result: null,
    error: null
  }});

  const startStreaming = useCallback(async (prompt, options = {{}}) => {{
    setStreamState({{
      is_streaming: true,
      partial_result: '',
      complete_result: null,
      error: null
    }});

    // Simulate streaming by gradually revealing text
    const fullText = `This is a simulated streaming response for: "${{prompt}}". The text is revealed token by token to simulate real streaming generation.`;
    const tokens = fullText.split(' ');

    for (let i = 0; i < tokens.length; i++) {{
      await new Promise(resolve => setTimeout(resolve, 100)); // 100ms delay per token

      const partialText = tokens.slice(0, i + 1).join(' ');
      setStreamState(prev => ({{
        ...prev,
        partial_result: partialText
      }}));
    }}

    setStreamState(prev => ({{
      ...prev,
      is_streaming: false,
      complete_result: fullText
    }}));

  }}, []);

  return {{
    streamState,
    startStreaming
  }};
}};

// Hook for model management
export const useTrustFormerModelManager = () => {{
  const [models, setModels] = useState([]);
  const [activeModel, setActiveModel] = useState(null);

  const addModel = useCallback((modelConfig) => {{
    setModels(prev => [...prev, {{
      id: Date.now(),
      ...modelConfig,
      loaded: false
    }}]);
  }}, []);

  const removeModel = useCallback((modelId) => {{
    setModels(prev => prev.filter(model => model.id !== modelId));
    if (activeModel?.id === modelId) {{
      setActiveModel(null);
    }}
  }}, [activeModel]);

  const switchModel = useCallback((modelId) => {{
    const model = models.find(m => m.id === modelId);
    if (model) {{
      setActiveModel(model);
    }}
  }}, [models]);

  return {{
    models,
    activeModel,
    addModel,
    removeModel,
    switchModel
  }};
}};
"#,
            self.config.debug_mode
        )
    }

    /// Generate CSS styles for components
    pub fn generate_component_styles(&self) -> String {
        r#"
/* TrustFormer React Components Styles */

.trustformer-text-generator {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

.trustformer-text-generator .trustformer-loading {
  text-align: center;
  padding: 20px;
}

.trustformer-text-generator .progress-bar {
  width: 100%;
  height: 8px;
  background-color: #f0f0f0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 8px;
}

.trustformer-text-generator .progress-fill {
  height: 100%;
  background-color: #007bff;
  transition: width 0.3s ease;
}

.trustformer-text-generator .input-section {
  margin-bottom: 16px;
}

.trustformer-text-generator .input-section textarea {
  width: 100%;
  min-height: 80px;
  padding: 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  resize: vertical;
  font-family: inherit;
  font-size: 14px;
}

.trustformer-text-generator .input-section button {
  margin-top: 8px;
  padding: 10px 20px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.trustformer-text-generator .input-section button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.trustformer-text-generator .generations {
  max-height: 400px;
  overflow-y: auto;
}

.trustformer-text-generator .generation-item {
  border: 1px solid #eee;
  border-radius: 4px;
  padding: 12px;
  margin-bottom: 8px;
  background-color: #f9f9f9;
}

.trustformer-text-generator .generation-item .prompt {
  margin-bottom: 8px;
  font-size: 14px;
}

.trustformer-text-generator .generation-item .result {
  margin-bottom: 8px;
  font-size: 14px;
  line-height: 1.4;
}

.trustformer-text-generator .generation-item .meta {
  color: #666;
  font-size: 12px;
}

/* Chat Interface Styles */

.trustformer-chat-interface {
  border: 1px solid #ddd;
  border-radius: 8px;
  height: 500px;
  display: flex;
  flex-direction: column;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

.trustformer-chat-interface .chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #eee;
  background-color: #f8f9fa;
}

.trustformer-chat-interface .chat-header h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
}

.trustformer-chat-interface .chat-header button {
  padding: 6px 12px;
  background-color: #6c757d;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.trustformer-chat-interface .chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.trustformer-chat-interface .message {
  max-width: 80%;
  word-wrap: break-word;
}

.trustformer-chat-interface .message-user {
  align-self: flex-end;
}

.trustformer-chat-interface .message-assistant,
.trustformer-chat-interface .message-error {
  align-self: flex-start;
}

.trustformer-chat-interface .message-content {
  padding: 10px 14px;
  border-radius: 18px;
  font-size: 14px;
  line-height: 1.4;
}

.trustformer-chat-interface .message-user .message-content {
  background-color: #007bff;
  color: white;
}

.trustformer-chat-interface .message-assistant .message-content {
  background-color: #f1f3f4;
  color: #333;
}

.trustformer-chat-interface .message-error .message-content {
  background-color: #dc3545;
  color: white;
}

.trustformer-chat-interface .message-meta {
  margin-top: 4px;
  font-size: 11px;
  color: #666;
  text-align: right;
}

.trustformer-chat-interface .message-assistant .message-meta,
.trustformer-chat-interface .message-error .message-meta {
  text-align: left;
}

.trustformer-chat-interface .typing {
  align-self: flex-start;
}

.trustformer-chat-interface .typing-indicator {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 10px 14px;
  background-color: #f1f3f4;
  border-radius: 18px;
}

.trustformer-chat-interface .typing-indicator span {
  width: 6px;
  height: 6px;
  background-color: #999;
  border-radius: 50%;
  animation: typing 1.4s infinite ease-in-out;
}

.trustformer-chat-interface .typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.trustformer-chat-interface .typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.4;
  }
  30% {
    transform: translateY(-10px);
    opacity: 1;
  }
}

.trustformer-chat-interface .chat-input {
  display: flex;
  padding: 12px 16px;
  border-top: 1px solid #eee;
  gap: 8px;
}

.trustformer-chat-interface .chat-input textarea {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid #ccc;
  border-radius: 20px;
  resize: none;
  font-family: inherit;
  font-size: 14px;
}

.trustformer-chat-interface .chat-input button {
  padding: 10px 20px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 20px;
  cursor: pointer;
  font-size: 14px;
  white-space: nowrap;
}

.trustformer-chat-interface .chat-input button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

/* Loading and Error States */

.trustformer-error {
  padding: 20px;
  text-align: center;
  border: 1px solid #dc3545;
  border-radius: 8px;
  background-color: #f8d7da;
  color: #721c24;
}

.trustformer-error button {
  margin-top: 10px;
  padding: 8px 16px;
  background-color: #dc3545;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.chat-loading {
  padding: 20px;
  text-align: center;
}
"#
        .to_string()
    }

    /// Generate TypeScript definitions
    pub fn generate_typescript_definitions(&self) -> String {
        r#"
// TypeScript definitions for TrustFormer React components

import { ReactNode, CSSProperties } from 'react';

export interface ModelLoadingState {
  is_loading: boolean;
  progress: number;
  error: string | null;
  model_loaded: boolean;
}

export interface InferenceState {
  is_inferring: boolean;
  result: string | null;
  error: string | null;
  inference_time_ms: number;
}

export interface InferenceResult {
  text: string;
  inferenceTime: number;
  tokenCount: number;
}

export interface Generation {
  id: number;
  prompt: string;
  result: string;
  timestamp: string;
  inferenceTime: number;
}

export interface Message {
  id: number;
  type: 'user' | 'assistant' | 'error';
  content: string;
  timestamp: string;
  inferenceTime?: number;
}

export interface ModelConfig {
  id: number;
  name: string;
  url: string;
  loaded: boolean;
}

// Component Props

export interface TrustFormerTextGeneratorProps {
  modelUrl?: string;
  placeholder?: string;
  maxLength?: number;
  temperature?: number;
  onGenerate?: (generation: Generation) => void;
  showProgress?: boolean;
  style?: CSSProperties;
  className?: string;
}

export interface TrustFormerChatInterfaceProps {
  modelUrl?: string;
  placeholder?: string;
  systemPrompt?: string;
  maxTokens?: number;
  temperature?: number;
  onMessage?: (messages: { user: Message; assistant: Message }) => void;
  style?: CSSProperties;
  className?: string;
}

// Hook Returns

export interface UseTrustFormerModelReturn {
  modelState: ModelLoadingState;
  loadModel: () => Promise<void>;
  session: any; // InferenceSession from WASM
}

export interface UseTrustFormerInferenceReturn {
  inferenceState: InferenceState;
  generateText: (prompt: string, options?: GenerationOptions) => Promise<InferenceResult>;
}

export interface GenerationOptions {
  maxLength?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
}

export interface StreamState {
  is_streaming: boolean;
  partial_result: string;
  complete_result: string | null;
  error: string | null;
}

export interface UseTrustFormerStreamingReturn {
  streamState: StreamState;
  startStreaming: (prompt: string, options?: GenerationOptions) => Promise<void>;
}

export interface UseTrustFormerModelManagerReturn {
  models: ModelConfig[];
  activeModel: ModelConfig | null;
  addModel: (config: Omit<ModelConfig, 'id' | 'loaded'>) => void;
  removeModel: (modelId: number) => void;
  switchModel: (modelId: number) => void;
}

// Hook Functions

export declare const useTrustFormerModel: (options?: {
  modelUrl?: string;
  autoLoad?: boolean;
  onLoadComplete?: (session: any) => void;
  onLoadError?: (error: Error) => void;
}) => UseTrustFormerModelReturn;

export declare const useTrustFormerInference: () => UseTrustFormerInferenceReturn;

export declare const useTrustFormerStreaming: () => UseTrustFormerStreamingReturn;

export declare const useTrustFormerModelManager: () => UseTrustFormerModelManagerReturn;

// Components

export declare const TrustFormerTextGenerator: React.FC<TrustFormerTextGeneratorProps>;
export declare const TrustFormerChatInterface: React.FC<TrustFormerChatInterfaceProps>;
"#
        .to_string()
    }

    /// Generate package.json for React integration
    pub fn generate_package_json(&self) -> String {
        r#"
{
  "name": "@trustformers/react",
  "version": "0.1.0",
  "description": "React components and hooks for TrustFormer WASM",
  "main": "dist/index.js",
  "module": "dist/index.esm.js",
  "types": "dist/index.d.ts",
  "files": [
    "dist",
    "src"
  ],
  "scripts": {
    "build": "rollup -c",
    "build:watch": "rollup -c -w",
    "test": "jest",
    "test:watch": "jest --watch",
    "lint": "eslint src --ext .ts,.tsx",
    "type-check": "tsc --noEmit"
  },
  "peerDependencies": {
    "react": ">=16.8.0",
    "react-dom": ">=16.8.0"
  },
  "dependencies": {
    "trustformers-wasm": "workspace:*"
  },
  "devDependencies": {
    "@rollup/plugin-commonjs": "^21.0.0",
    "@rollup/plugin-node-resolve": "^13.0.0",
    "@rollup/plugin-typescript": "^8.0.0",
    "@testing-library/jest-dom": "^5.16.0",
    "@testing-library/react": "^12.0.0",
    "@testing-library/react-hooks": "^8.0.0",
    "@types/jest": "^27.0.0",
    "@types/react": "^17.0.0",
    "@types/react-dom": "^17.0.0",
    "@typescript-eslint/eslint-plugin": "^5.0.0",
    "@typescript-eslint/parser": "^5.0.0",
    "eslint": "^8.0.0",
    "eslint-plugin-react": "^7.26.0",
    "eslint-plugin-react-hooks": "^4.2.0",
    "jest": "^27.0.0",
    "react": "^17.0.0",
    "react-dom": "^17.0.0",
    "rollup": "^2.56.0",
    "ts-jest": "^27.0.0",
    "typescript": "^4.4.0"
  },
  "keywords": [
    "trustformers",
    "wasm",
    "react",
    "ai",
    "machine-learning",
    "transformers",
    "nlp"
  ],
  "author": "TrustFormers Team",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/trustformers/trustformers-wasm"
  },
  "homepage": "https://trustformers.ai"
}
"#
        .to_string()
    }

    /// Get all generated components as a JavaScript object
    pub fn get_all_components(&self) -> Result<Object, JsValue> {
        let components = Object::new();

        js_sys::Reflect::set(
            &components,
            &"TextGenerator".into(),
            &self.generate_text_generator_component().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"ChatInterface".into(),
            &self.generate_chat_interface_component().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"hooks".into(),
            &self.generate_react_hooks().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"styles".into(),
            &self.generate_component_styles().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"types".into(),
            &self.generate_typescript_definitions().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"package".into(),
            &self.generate_package_json().into(),
        )?;

        Ok(components)
    }
}

/// Check if React is available in the environment
#[wasm_bindgen]
pub fn is_react_available() -> bool {
    let js_code = r#"
        try {
            return typeof React !== 'undefined' &&
                   typeof React.useState !== 'undefined' &&
                   typeof React.useEffect !== 'undefined';
        } catch (e) {
            return false;
        }
    "#;

    js_sys::eval(js_code)
        .map(|result| result.as_bool().unwrap_or(false))
        .unwrap_or(false)
}

/// Generate a complete React integration package
#[wasm_bindgen]
pub fn generate_react_package(config: ReactConfig) -> Result<Object, JsValue> {
    let factory = ReactComponentFactory::new(config);
    factory.get_all_components()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_react_config() {
        let mut config = ReactConfig::new();
        assert!(config.auto_load_model());
        assert!(config.show_progress());

        config.set_auto_load_model(false);
        assert!(!config.auto_load_model());

        config.set_model_url("https://example.com/model.wasm".to_string());
        assert_eq!(config.model_url(), "https://example.com/model.wasm");
    }

    #[test]
    fn test_component_generation() {
        let config = ReactConfig::new();
        let factory = ReactComponentFactory::new(config);

        let text_generator = factory.generate_text_generator_component();
        assert!(text_generator.contains("TrustFormerTextGenerator"));

        let chat_interface = factory.generate_chat_interface_component();
        assert!(chat_interface.contains("TrustFormerChatInterface"));

        let hooks = factory.generate_react_hooks();
        assert!(hooks.contains("useTrustFormerModel"));
    }
}
