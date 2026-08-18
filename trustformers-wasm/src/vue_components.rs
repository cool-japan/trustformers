//! Vue.js component bindings for TrustFormer WASM
//!
//! This module provides Vue.js component wrappers and composables for easy integration
//! of TrustFormer WASM functionality into Vue applications.

#![allow(dead_code)]
use js_sys::Object;
use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Vue composable state for model loading
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct VueModelState {
    is_loading: bool,
    progress: f64,
    error: Option<String>,
    model_loaded: bool,
}

/// Vue composable state for inference
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct VueInferenceState {
    is_inferring: bool,
    result: Option<String>,
    error: Option<String>,
    inference_time_ms: f64,
}

/// Configuration for Vue components
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct VueConfig {
    auto_load_model: bool,
    show_progress: bool,
    enable_streaming: bool,
    debug_mode: bool,
    model_url: String,
    fallback_message: String,
}

/// Vue component factory for TrustFormer
#[wasm_bindgen]
pub struct VueComponentFactory {
    config: VueConfig,
    component_registry: Vec<VueComponentDefinition>,
}

/// Component definition for Vue integration
#[derive(Debug, Clone)]
struct VueComponentDefinition {
    name: String,
    props_schema: String,
    component_type: VueComponentType,
    template: String,
}

/// Types of Vue components
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VueComponentType {
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

/// Vue composable manager
#[wasm_bindgen]
pub struct VueComposableManager {
    composables: Vec<ComposableDefinition>,
    state_managers: Vec<VueStateManager>,
}

/// Composable definition
#[derive(Debug, Clone)]
struct ComposableDefinition {
    name: String,
    composable_type: ComposableType,
    dependencies: Vec<String>,
    implementation: String,
}

/// Types of Vue composables
#[derive(Debug, Clone, Copy, PartialEq)]
enum ComposableType {
    ModelLoader,
    Inference,
    StreamingGeneration,
    ModelState,
    ErrorHandling,
}

/// State manager for Vue composables
#[derive(Debug, Clone)]
struct VueStateManager {
    id: String,
    state_type: String,
    initial_state: String,
    mutations: Vec<String>,
}

#[wasm_bindgen]
impl VueConfig {
    /// Create new Vue configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> VueConfig {
        VueConfig {
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

impl Default for VueConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl VueModelState {
    #[wasm_bindgen(constructor)]
    pub fn new() -> VueModelState {
        VueModelState {
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

impl Default for VueModelState {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl VueInferenceState {
    #[wasm_bindgen(constructor)]
    pub fn new() -> VueInferenceState {
        VueInferenceState {
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

impl Default for VueInferenceState {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl VueComponentFactory {
    /// Create new Vue component factory
    #[wasm_bindgen(constructor)]
    pub fn new(config: VueConfig) -> VueComponentFactory {
        VueComponentFactory {
            config,
            component_registry: Vec::new(),
        }
    }

    /// Generate Text Generator Vue component
    pub fn generate_text_generator_component(&self) -> String {
        format!(
            r#"
<template>
  <div :class="['trustformer-text-generator', props.className]" :style="props.style">
    <!-- Loading state -->
    <div v-if="modelState.isLoading && showProgress" class="trustformer-loading">
      <div class="progress-bar">
        <div
          class="progress-fill"
          :style="{{ width: `${{modelState.progress}}%` }}"
        ></div>
      </div>
      <p>Loading model... {{{{ Math.round(modelState.progress) }}}}%</p>
    </div>

    <!-- Error state -->
    <div v-if="modelState.error" class="trustformer-error">
      <p>Error loading model: {{{{ modelState.error }}}}</p>
      <button @click="loadModel">Retry</button>
    </div>

    <!-- Main interface -->
    <div v-if="modelState.modelLoaded">
      <div class="input-section">
        <textarea
          v-model="prompt"
          @keypress="handleKeyPress"
          :placeholder="props.placeholder"
          :disabled="inferenceState.isInferring"
          rows="3"
        ></textarea>
        <button
          @click="handleGenerate"
          :disabled="!prompt.trim() || inferenceState.isInferring"
        >
          {{{{ inferenceState.isInferring ? 'Generating...' : 'Generate' }}}}
        </button>
      </div>

      <div class="generations">
        <div
          v-for="generation in generations"
          :key="generation.id"
          class="generation-item"
        >
          <div class="prompt">
            <strong>Prompt:</strong> {{{{ generation.prompt }}}}
          </div>
          <div class="result">
            <strong>Result:</strong> {{{{ generation.result }}}}
          </div>
          <div class="meta">
            <small>
              Generated in {{{{ generation.inferenceTime }}}}ms at {{{{ new Date(generation.timestamp).toLocaleTimeString() }}}}
            </small>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {{ ref, computed, watch, onMounted }} from 'vue'
import {{ useTrustFormerModel, useTrustFormerInference }} from './composables'

// Props interface
interface Props {{
  modelUrl?: string
  vocabUrl?: string
  placeholder?: string
  maxLength?: number
  temperature?: number
  showProgress?: boolean
  style?: Record<string, any>
  className?: string
}}

// Events interface
interface Emits {{
  (e: 'generate', generation: any): void
}}

// Component setup
const props = withDefaults(defineProps<Props>(), {{
  modelUrl: '{}',
  vocabUrl: '',
  placeholder: 'Enter your prompt...',
  maxLength: 100,
  temperature: 0.7,
  showProgress: {},
  style: () => ({{}}),
  className: ''
}})

const emit = defineEmits<Emits>()

// Reactive state
const prompt = ref('')
const generations = ref<any[]>([])

// Composables
const {{ modelState, loadModel, pipeline }} = useTrustFormerModel({{
  modelUrl: computed(() => props.modelUrl),
  vocabUrl: computed(() => props.vocabUrl),
  autoLoad: {}
}})

const {{ inferenceState, generateText }} = useTrustFormerInference(pipeline)

// Load model on mount if needed
onMounted(() => {{
  if (props.modelUrl && !modelState.value.modelLoaded && !modelState.value.isLoading) {{
    loadModel()
  }}
}})

// Watch for model URL changes
watch(() => props.modelUrl, (newUrl) => {{
  if (newUrl && !modelState.value.modelLoaded && !modelState.value.isLoading) {{
    loadModel()
  }}
}})

// Handle text generation
const handleGenerate = async () => {{
  if (!prompt.value.trim() || !modelState.value.modelLoaded) return

  try {{
    const result = await generateText(prompt.value, {{
      maxLength: props.maxLength,
      temperature: props.temperature
    }})

    const newGeneration = {{
      id: Date.now(),
      prompt: prompt.value,
      result: result.text,
      timestamp: new Date().toISOString(),
      inferenceTime: result.inferenceTime
    }}

    generations.value.unshift(newGeneration)
    prompt.value = ''

    emit('generate', newGeneration)
  }} catch (error) {{
    console.error('Generation failed:', error)
  }}
}}

// Handle keyboard shortcuts
const handleKeyPress = (e: KeyboardEvent) => {{
  if (e.key === 'Enter' && e.ctrlKey) {{
    handleGenerate()
  }}
}}
</script>

<style scoped>
/* Component-specific styles here */
</style>
"#,
            self.config.model_url, self.config.show_progress, self.config.auto_load_model
        )
    }

    /// Generate Chat Interface Vue component
    pub fn generate_chat_interface_component(&self) -> String {
        format!(
            r#"
<template>
  <div :class="['trustformer-chat-interface', props.className]" :style="props.style">
    <!-- Loading state -->
    <div v-if="modelState.isLoading" class="chat-loading">
      <div class="progress-bar">
        <div
          class="progress-fill"
          :style="{{ width: `${{modelState.progress}}%` }}"
        ></div>
      </div>
      <p>{}... {{{{ Math.round(modelState.progress) }}}}%</p>
    </div>

    <!-- Error state -->
    <div v-if="modelState.error" class="trustformer-chat-error">
      <p>Error loading model: {{{{ modelState.error }}}}</p>
      <button @click="loadModel">Retry</button>
    </div>

    <!-- Chat interface -->
    <template v-if="modelState.modelLoaded">
      <div class="chat-header">
        <h3>AI Assistant</h3>
        <button @click="clearChat" :disabled="messages.length === 0">
          Clear Chat
        </button>
      </div>

      <div class="chat-messages" ref="messagesContainer">
        <div
          v-for="message in messages"
          :key="message.id"
          :class="['message', `message-${{message.type}}`]"
        >
          <div class="message-content">
            {{{{ message.content }}}}
          </div>
          <div class="message-meta">
            <small>
              {{{{ new Date(message.timestamp).toLocaleTimeString() }}}}
              <span v-if="message.inferenceTime"> • {{{{ message.inferenceTime }}}}ms</span>
            </small>
          </div>
        </div>

        <!-- Typing indicator -->
        <div v-if="inferenceState.isInferring" class="message message-assistant typing">
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>

      <div class="chat-input">
        <textarea
          v-model="input"
          @keypress="handleKeyPress"
          :placeholder="props.placeholder"
          :disabled="inferenceState.isInferring"
          rows="2"
        ></textarea>
        <button
          @click="handleSendMessage"
          :disabled="!input.trim() || inferenceState.isInferring"
        >
          Send
        </button>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import {{ ref, computed, watch, onMounted, nextTick }} from 'vue'
import {{ useTrustFormerModel, useTrustFormerInference }} from './composables'

// Props interface
interface Props {{
  modelUrl?: string
  vocabUrl?: string
  placeholder?: string
  systemPrompt?: string
  maxTokens?: number
  temperature?: number
  style?: Record<string, any>
  className?: string
}}

// Events interface
interface Emits {{
  (e: 'message', payload: {{ user: any, assistant: any }}): void
}}

// Component setup
const props = withDefaults(defineProps<Props>(), {{
  modelUrl: '{}',
  vocabUrl: '',
  placeholder: 'Type your message...',
  systemPrompt: 'You are a helpful AI assistant.',
  maxTokens: 150,
  temperature: 0.7,
  style: () => ({{}}),
  className: ''
}})

const emit = defineEmits<Emits>()

// Reactive state
const messages = ref<any[]>([])
const input = ref('')
const messagesContainer = ref<HTMLElement>()

// Composables
const {{ modelState, loadModel, pipeline }} = useTrustFormerModel({{
  modelUrl: computed(() => props.modelUrl),
  vocabUrl: computed(() => props.vocabUrl),
  autoLoad: {}
}})

const {{ inferenceState, generateText }} = useTrustFormerInference(pipeline)

// Load model on mount if needed
onMounted(() => {{
  if (props.modelUrl && !modelState.value.modelLoaded && !modelState.value.isLoading) {{
    loadModel()
  }}
}})

// Watch for model URL changes
watch(() => props.modelUrl, (newUrl) => {{
  if (newUrl && !modelState.value.modelLoaded && !modelState.value.isLoading) {{
    loadModel()
  }}
}})

// Auto-scroll to bottom when messages change
watch(messages, () => {{
  nextTick(() => {{
    scrollToBottom()
  }})
}}, {{ deep: true }})

// Scroll to bottom of messages
const scrollToBottom = () => {{
  if (messagesContainer.value) {{
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }}
}}

// Handle sending message
const handleSendMessage = async () => {{
  if (!input.value.trim() || !modelState.value.modelLoaded || inferenceState.value.isInferring) return

  const userMessage = {{
    id: Date.now(),
    type: 'user',
    content: input.value.trim(),
    timestamp: new Date().toISOString()
  }}

  messages.value.push(userMessage)
  input.value = ''

  // Prepare conversation context
  const conversationHistory = messages.value.map(msg =>
    `${{msg.type === 'user' ? 'User' : 'Assistant'}}: ${{msg.content}}`
  ).join('\\n')

  const prompt = `${{props.systemPrompt}}\\n\\n${{conversationHistory}}\\nUser: ${{userMessage.content}}\\nAssistant:`

  try {{
    const result = await generateText(prompt, {{
      maxLength: props.maxTokens,
      temperature: props.temperature
    }})

    const assistantMessage = {{
      id: Date.now() + 1,
      type: 'assistant',
      content: result.text.trim(),
      timestamp: new Date().toISOString(),
      inferenceTime: result.inferenceTime
    }}

    messages.value.push(assistantMessage)

    emit('message', {{ user: userMessage, assistant: assistantMessage }})
  }} catch (error) {{
    console.error('Chat inference failed:', error)
    const errorMessage = {{
      id: Date.now() + 1,
      type: 'error',
      content: 'Sorry, I encountered an error. Please try again.',
      timestamp: new Date().toISOString()
    }}
    messages.value.push(errorMessage)
  }}
}}

// Handle keyboard shortcuts
const handleKeyPress = (e: KeyboardEvent) => {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault()
    handleSendMessage()
  }}
}}

// Clear chat messages
const clearChat = () => {{
  messages.value = []
}}
</script>

<style scoped>
/* Component-specific styles here */
</style>
"#,
            self.config.fallback_message, self.config.model_url, self.config.auto_load_model
        )
    }

    /// Generate Vue composables
    pub fn generate_vue_composables(&self) -> String {
        format!(
            r#"
// Vue composables for TrustFormer WASM integration

import {{ ref, computed, watch, onMounted, onUnmounted, Ref, ComputedRef }} from 'vue'

// Import the WASM module
let wasmModule: any = null
const initWasm = async () => {{
  if (!wasmModule) {{
    wasmModule = await import('trustformers-wasm')
    await wasmModule.default()
  }}
  return wasmModule
}}

// Interfaces
export interface ModelState {{
  isLoading: boolean
  progress: number
  error: string | null
  modelLoaded: boolean
}}

export interface InferenceState {{
  isInferring: boolean
  result: string | null
  error: string | null
  inferenceTimeMs: number
}}

export interface InferenceResult {{
  text: string
  inferenceTime: number
  tokenCount: number
}}

export interface GenerationOptions {{
  maxLength?: number
  temperature?: number
  topP?: number
  topK?: number
}}

export interface StreamState {{
  isStreaming: boolean
  partialResult: string
  completeResult: string | null
  error: string | null
}}

export interface ModelConfig {{
  id: number
  name: string
  url: string
  loaded: boolean
}}

// Maps a friendly model-type string to the real `wasm.ModelArchitecture`
// enum variant (mirrors `resolve_model_architecture` in `lib.rs`).
const MODEL_ARCHITECTURES: Record<string, string> = {{
  bert: 'Bert',
  gpt2: 'GPT2',
  gpt: 'GPT2',
  t5: 'T5',
  llama: 'Llama',
  mistral: 'Mistral'
}}

// Composable for model + tokenizer loading and management. When `vocabUrl`
// is supplied, this also builds a real `TextGenerationPipeline`
// (`WasmModel` + `WasmTokenizer`, both loaded with real data) so
// `useTrustFormerInference`/`useTrustFormerStreaming` below can drive
// genuine generation instead of fabricating output. Per project policy, a
// tokenizer is never built with a fabricated vocabulary - without
// `vocabUrl` there is simply no `pipeline`.
export function useTrustFormerModel(options: {{
  modelUrl?: ComputedRef<string> | Ref<string> | string
  vocabUrl?: ComputedRef<string> | Ref<string> | string
  modelType?: string
  autoLoad?: boolean
  onLoadComplete?: (result: {{ session: any; pipeline: any }}) => void
  onLoadError?: (error: Error) => void
}} = {{}}) {{
  const modelState = ref<ModelState>({{
    isLoading: false,
    progress: 0,
    error: null,
    modelLoaded: false
  }})

  const session = ref<any>(null)
  const pipeline = ref<any>(null)

  // Normalize modelUrl/vocabUrl to computed refs
  const normalizedModelUrl = computed(() => {{
    if (typeof options.modelUrl === 'string') {{
      return options.modelUrl
    }} else if (options.modelUrl) {{
      return options.modelUrl.value
    }}
    return ''
  }})

  const normalizedVocabUrl = computed(() => {{
    if (typeof options.vocabUrl === 'string') {{
      return options.vocabUrl
    }} else if (options.vocabUrl) {{
      return options.vocabUrl.value
    }}
    return ''
  }})

  const modelType = options.modelType ?? 'gpt2'

  const loadModel = async () => {{
    if (!normalizedModelUrl.value) {{
      modelState.value.error = 'No model URL provided'
      return
    }}

    modelState.value = {{
      isLoading: true,
      progress: 0,
      error: null,
      modelLoaded: false
    }}

    try {{
      const wasm = await initWasm()

      // Create the higher-level tensor-in/tensor-out inference session.
      session.value = new wasm.InferenceSession(modelType)
      await session.value.initialize_with_auto_device()

      // Enable debug logging if configured
      if ({}) {{
        const debugConfig = new wasm.DebugConfig()
        debugConfig.set_log_level(wasm.LogLevel.Info)
        session.value.enable_debug_logging(debugConfig)
      }}

      // Load model from URL or cache
      await session.value.load_model_with_cache(
        'model_' + normalizedModelUrl.value.split('/').pop(),
        normalizedModelUrl.value,
        'TrustFormer Model',
        modelType,
        '1.0.0'
      )

      // Build a real text-generation pipeline for the composables below.
      if (normalizedVocabUrl.value) {{
        const modelResponse = await fetch(normalizedModelUrl.value)
        if (!modelResponse.ok) {{
          throw new Error(`Failed to fetch model weights: HTTP ${{modelResponse.status}}`)
        }}
        const modelBytes = new Uint8Array(await modelResponse.arrayBuffer())

        const architectureName = MODEL_ARCHITECTURES[modelType.toLowerCase()] ?? 'GPT2'
        const config = new wasm.ModelConfig(wasm.ModelArchitecture[architectureName])
        const model = new wasm.WasmModel(config)
        await model.load_weights(modelBytes)

        const vocabResponse = await fetch(normalizedVocabUrl.value)
        if (!vocabResponse.ok) {{
          throw new Error(`Failed to fetch vocabulary: HTTP ${{vocabResponse.status}}`)
        }}
        const vocab = await vocabResponse.json()
        const tokenizer = new wasm.WasmTokenizer(wasm.TokenizerType.BPE)
        tokenizer.load_vocab(vocab)

        pipeline.value = new wasm.TextGenerationPipeline(model, tokenizer)
      }}

      modelState.value = {{
        isLoading: false,
        progress: 100,
        error: null,
        modelLoaded: true
      }}

      if (options.onLoadComplete) {{
        options.onLoadComplete({{ session: session.value, pipeline: pipeline.value }})
      }}

    }} catch (error: any) {{
      console.error('Model loading failed:', error)
      const errorMessage = error.message || 'Failed to load model'

      modelState.value = {{
        isLoading: false,
        progress: 0,
        error: errorMessage,
        modelLoaded: false
      }}

      if (options.onLoadError) {{
        options.onLoadError(error)
      }}
    }}
  }}

  // Auto-load on mount if configured
  onMounted(() => {{
    if (options.autoLoad !== false && normalizedModelUrl.value && !modelState.value.modelLoaded) {{
      loadModel()
    }}
  }})

  // Watch for URL changes
  watch(normalizedModelUrl, (newUrl) => {{
    if (newUrl && !modelState.value.modelLoaded && !modelState.value.isLoading) {{
      loadModel()
    }}
  }})

  return {{
    modelState: computed(() => modelState.value),
    loadModel,
    session: computed(() => session.value),
    pipeline: computed(() => pipeline.value)
  }}
}}

// Composable for real (non-streaming) text generation, driven by a
// `TextGenerationPipeline` built by `useTrustFormerModel` (pass its
// `pipeline` ref/value in here). Runs the pipeline's actual autoregressive
// `encode` -> `next_token` -> `decode` loop instead of fabricating a canned
// response string.
export function useTrustFormerInference(pipeline: Ref<any> | ComputedRef<any> | any) {{
  const inferenceState = ref<InferenceState>({{
    isInferring: false,
    result: null,
    error: null,
    inferenceTimeMs: 0
  }})

  const resolvePipeline = () => (pipeline && 'value' in pipeline ? pipeline.value : pipeline)

  const generateText = async (prompt: string, options: GenerationOptions = {{}}): Promise<InferenceResult> => {{
    const activePipeline = resolvePipeline()
    if (!activePipeline) {{
      const error = new Error(
        'useTrustFormerInference: no pipeline available - call useTrustFormerModel with a vocabUrl first'
      )
      inferenceState.value = {{ ...inferenceState.value, error: error.message }}
      throw error
    }}

    const maxLength = options.maxLength ?? 100

    inferenceState.value = {{
      isInferring: true,
      result: null,
      error: null,
      inferenceTimeMs: 0
    }}

    const startTime = performance.now()

    try {{
      // Real generation: encode the prompt, then repeatedly call the
      // pipeline's real `next_token` (the same autoregressive step
      // `StreamingGenerator` drives internally) until `maxLength` tokens
      // have been produced.
      let contextIds = Array.from(activePipeline.encode(prompt, true)) as number[]
      const generatedIds: number[] = []

      for (let i = 0; i < maxLength; i++) {{
        const nextId = activePipeline.next_token(Uint32Array.from(contextIds))
        generatedIds.push(nextId)
        contextIds = [...contextIds, nextId]
      }}

      const generatedText = activePipeline.decode(Uint32Array.from(generatedIds), true)
      const endTime = performance.now()
      const inferenceTime = endTime - startTime

      const inferenceResult: InferenceResult = {{
        text: generatedText,
        inferenceTime: Math.round(inferenceTime),
        tokenCount: generatedIds.length
      }}

      inferenceState.value = {{
        isInferring: false,
        result: inferenceResult.text,
        error: null,
        inferenceTimeMs: inferenceTime
      }}

      return inferenceResult

    }} catch (error: any) {{
      console.error('Inference failed:', error)
      const errorMessage = error.message || 'Inference failed'

      inferenceState.value = {{
        isInferring: false,
        result: null,
        error: errorMessage,
        inferenceTimeMs: 0
      }}

      throw error
    }}
  }}

  return {{
    inferenceState: computed(() => inferenceState.value),
    generateText
  }}
}}

// Composable for real streaming generation, driven by the real
// `StreamingGenerator` (see `streaming_generation.rs`) attached to a
// `TextGenerationPipeline` built by `useTrustFormerModel` (pass its
// `pipeline` ref/value in here). This used to fake streaming entirely with
// a hardcoded canned string revealed word-by-word via `setTimeout`. Every
// token below instead comes from `StreamingGenerator.start_streaming`'s
// real autoregressive loop, reported through its
// `on_token`/`on_complete`/`on_error` callbacks.
export function useTrustFormerStreaming(pipeline: Ref<any> | ComputedRef<any> | any) {{
  const streamState = ref<StreamState>({{
    isStreaming: false,
    partialResult: '',
    completeResult: null,
    error: null
  }})

  const generatorRef = ref<any>(null)
  const resolvePipeline = () => (pipeline && 'value' in pipeline ? pipeline.value : pipeline)

  const startStreaming = async (prompt: string, options: GenerationOptions = {{}}) => {{
    const activePipeline = resolvePipeline()
    if (!activePipeline) {{
      const error = new Error(
        'useTrustFormerStreaming: no pipeline available - call useTrustFormerModel with a vocabUrl first'
      )
      streamState.value = {{ ...streamState.value, error: error.message }}
      throw error
    }}

    streamState.value = {{
      isStreaming: true,
      partialResult: '',
      completeResult: null,
      error: null
    }}

    try {{
      const wasm = await initWasm()

      const config = new wasm.StreamingConfig()
      config.set_max_tokens(options.maxLength ?? 100)
      config.set_temperature(options.temperature ?? 0.7)
      if (options.topP !== undefined) config.set_top_p(options.topP)
      if (options.topK !== undefined) config.set_top_k(options.topK)

      const generator = new wasm.StreamingGenerator(config)
      generator.set_pipeline(activePipeline)
      generatorRef.value = generator

      let accumulated = ''

      generator.on_token((token: any) => {{
        accumulated += token.token
        streamState.value.partialResult = accumulated
      }})

      generator.on_complete(() => {{
        streamState.value.isStreaming = false
        streamState.value.completeResult = accumulated
      }})

      generator.on_error((error: any) => {{
        streamState.value.isStreaming = false
        streamState.value.error = String(error)
      }})

      await generator.start_streaming(prompt)

    }} catch (error: any) {{
      streamState.value.isStreaming = false
      streamState.value.error = error.message || 'Streaming failed'
    }}
  }}

  const stopStreaming = () => {{
    if (generatorRef.value) {{
      generatorRef.value.stop_streaming()
    }}
    streamState.value.isStreaming = false
  }}

  return {{
    streamState: computed(() => streamState.value),
    startStreaming,
    stopStreaming
  }}
}}

// Composable for model management
export function useTrustFormerModelManager() {{
  const models = ref<ModelConfig[]>([])
  const activeModel = ref<ModelConfig | null>(null)

  const addModel = (modelConfig: Omit<ModelConfig, 'id' | 'loaded'>) => {{
    const newModel: ModelConfig = {{
      id: Date.now(),
      ...modelConfig,
      loaded: false
    }}
    models.value.push(newModel)
    return newModel
  }}

  const removeModel = (modelId: number) => {{
    const index = models.value.findIndex(model => model.id === modelId)
    if (index !== -1) {{
      models.value.splice(index, 1)
      if (activeModel.value?.id === modelId) {{
        activeModel.value = null
      }}
    }}
  }}

  const switchModel = (modelId: number) => {{
    const model = models.value.find(m => m.id === modelId)
    if (model) {{
      activeModel.value = model
    }}
  }}

  const markModelLoaded = (modelId: number, loaded: boolean = true) => {{
    const model = models.value.find(m => m.id === modelId)
    if (model) {{
      model.loaded = loaded
    }}
  }}

  return {{
    models: computed(() => models.value),
    activeModel: computed(() => activeModel.value),
    addModel,
    removeModel,
    switchModel,
    markModelLoaded
  }}
}}

// Composable for performance monitoring
export function useTrustFormerPerformance() {{
  const metrics = ref({{
    modelLoadTime: 0,
    averageInferenceTime: 0,
    totalInferences: 0,
    memoryUsage: 0,
    gpuUsage: 0
  }})

  const startTimer = () => {{
    return performance.now()
  }}

  const endTimer = (startTime: number) => {{
    return performance.now() - startTime
  }}

  const recordInference = (inferenceTime: number) => {{
    metrics.value.totalInferences++
    metrics.value.averageInferenceTime =
      (metrics.value.averageInferenceTime * (metrics.value.totalInferences - 1) + inferenceTime) /
      metrics.value.totalInferences
  }}

  const resetMetrics = () => {{
    metrics.value = {{
      modelLoadTime: 0,
      averageInferenceTime: 0,
      totalInferences: 0,
      memoryUsage: 0,
      gpuUsage: 0
    }}
  }}

  return {{
    metrics: computed(() => metrics.value),
    startTimer,
    endTimer,
    recordInference,
    resetMetrics
  }}
}}
"#,
            self.config.debug_mode
        )
    }

    /// Generate CSS styles for Vue components
    pub fn generate_vue_component_styles(&self) -> String {
        r#"
/* TrustFormer Vue Components Styles */

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
  background-color: #42b883;
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
  background-color: #42b883;
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
  background-color: #42b883;
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
  background-color: #42b883;
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

.trustformer-chat-error {
  padding: 20px;
  text-align: center;
  border: 1px solid #dc3545;
  border-radius: 8px;
  background-color: #f8d7da;
  color: #721c24;
}

.trustformer-chat-error button {
  margin-top: 10px;
  padding: 8px 16px;
  background-color: #dc3545;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
"#
        .to_string()
    }

    /// Generate TypeScript definitions for Vue
    pub fn generate_vue_typescript_definitions(&self) -> String {
        r#"
// TypeScript definitions for TrustFormer Vue components

import { ComputedRef, Ref } from 'vue'

export interface ModelState {
  isLoading: boolean
  progress: number
  error: string | null
  modelLoaded: boolean
}

export interface InferenceState {
  isInferring: boolean
  result: string | null
  error: string | null
  inferenceTimeMs: number
}

export interface InferenceResult {
  text: string
  inferenceTime: number
  tokenCount: number
}

export interface Generation {
  id: number
  prompt: string
  result: string
  timestamp: string
  inferenceTime: number
}

export interface Message {
  id: number
  type: 'user' | 'assistant' | 'error'
  content: string
  timestamp: string
  inferenceTime?: number
}

export interface ModelConfig {
  id: number
  name: string
  url: string
  loaded: boolean
}

export interface GenerationOptions {
  maxLength?: number
  temperature?: number
  topP?: number
  topK?: number
}

export interface StreamState {
  isStreaming: boolean
  partialResult: string
  completeResult: string | null
  error: string | null
}

// Component Props

export interface TrustFormerTextGeneratorProps {
  modelUrl?: string
  placeholder?: string
  maxLength?: number
  temperature?: number
  showProgress?: boolean
  style?: Record<string, any>
  className?: string
}

export interface TrustFormerChatInterfaceProps {
  modelUrl?: string
  placeholder?: string
  systemPrompt?: string
  maxTokens?: number
  temperature?: number
  style?: Record<string, any>
  className?: string
}

// Composable Returns

export interface UseTrustFormerModelReturn {
  modelState: ComputedRef<ModelState>
  loadModel: () => Promise<void>
  session: ComputedRef<any>
  pipeline: ComputedRef<any>
}

export interface UseTrustFormerInferenceReturn {
  inferenceState: ComputedRef<InferenceState>
  generateText: (prompt: string, options?: GenerationOptions) => Promise<InferenceResult>
}

export interface UseTrustFormerStreamingReturn {
  streamState: ComputedRef<StreamState>
  startStreaming: (prompt: string, options?: GenerationOptions) => Promise<void>
  stopStreaming: () => void
}

export interface UseTrustFormerModelManagerReturn {
  models: ComputedRef<ModelConfig[]>
  activeModel: ComputedRef<ModelConfig | null>
  addModel: (config: Omit<ModelConfig, 'id' | 'loaded'>) => ModelConfig
  removeModel: (modelId: number) => void
  switchModel: (modelId: number) => void
  markModelLoaded: (modelId: number, loaded?: boolean) => void
}

export interface PerformanceMetrics {
  modelLoadTime: number
  averageInferenceTime: number
  totalInferences: number
  memoryUsage: number
  gpuUsage: number
}

export interface UseTrustFormerPerformanceReturn {
  metrics: ComputedRef<PerformanceMetrics>
  startTimer: () => number
  endTimer: (startTime: number) => number
  recordInference: (inferenceTime: number) => void
  resetMetrics: () => void
}

// Composable Functions

export declare function useTrustFormerModel(options?: {
  modelUrl?: ComputedRef<string> | Ref<string> | string
  vocabUrl?: ComputedRef<string> | Ref<string> | string
  modelType?: string
  autoLoad?: boolean
  onLoadComplete?: (result: { session: any; pipeline: any }) => void
  onLoadError?: (error: Error) => void
}): UseTrustFormerModelReturn

export declare function useTrustFormerInference(pipeline: ComputedRef<any> | Ref<any> | any): UseTrustFormerInferenceReturn

export declare function useTrustFormerStreaming(pipeline: ComputedRef<any> | Ref<any> | any): UseTrustFormerStreamingReturn

export declare function useTrustFormerModelManager(): UseTrustFormerModelManagerReturn

export declare function useTrustFormerPerformance(): UseTrustFormerPerformanceReturn

// Plugin Installation

export interface TrustFormerVueOptions {
  defaultModelUrl?: string
  autoLoad?: boolean
  debugMode?: boolean
}

declare module '@vue/runtime-core' {
  interface ComponentCustomProperties {
    $trustformer: {
      modelState: ModelState
      loadModel: () => Promise<void>
      generateText: (prompt: string, options?: GenerationOptions) => Promise<InferenceResult>
    }
  }
}
"#
        .to_string()
    }

    /// Generate package.json for Vue integration
    pub fn generate_vue_package_json(&self) -> String {
        r#"
{
  "name": "@trustformers/vue",
  "version": "0.1.0",
  "description": "Vue.js components and composables for TrustFormer WASM",
  "main": "dist/index.js",
  "module": "dist/index.esm.js",
  "types": "dist/index.d.ts",
  "files": [
    "dist",
    "src"
  ],
  "scripts": {
    "build": "vite build",
    "build:watch": "vite build --watch",
    "test": "vitest",
    "test:watch": "vitest --watch",
    "lint": "eslint src --ext .ts,.vue",
    "type-check": "vue-tsc --noEmit"
  },
  "peerDependencies": {
    "vue": ">=3.0.0"
  },
  "dependencies": {
    "trustformers-wasm": "workspace:*"
  },
  "devDependencies": {
    "@types/node": "^18.0.0",
    "@typescript-eslint/eslint-plugin": "^5.0.0",
    "@typescript-eslint/parser": "^5.0.0",
    "@vitejs/plugin-vue": "^4.0.0",
    "@vue/test-utils": "^2.3.0",
    "eslint": "^8.0.0",
    "eslint-plugin-vue": "^9.0.0",
    "jsdom": "^21.0.0",
    "typescript": "^4.9.0",
    "vite": "^4.0.0",
    "vitest": "^0.28.0",
    "vue": "^3.2.0",
    "vue-tsc": "^1.0.0"
  },
  "keywords": [
    "trustformers",
    "wasm",
    "vue",
    "ai",
    "machine-learning",
    "transformers",
    "nlp",
    "composition-api"
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

    /// Generate Vue plugin for global installation
    pub fn generate_vue_plugin(&self) -> String {
        r#"
// Vue plugin for global TrustFormer integration

import { App, Plugin } from 'vue'
import { useTrustFormerModel, useTrustFormerInference } from './composables'

export interface TrustFormerVueOptions {
  defaultModelUrl?: string
  defaultVocabUrl?: string
  autoLoad?: boolean
  debugMode?: boolean
}

export const TrustFormerPlugin: Plugin = {
  install(app: App, options: TrustFormerVueOptions = {}) {
    // Global properties
    app.config.globalProperties.$trustformer = {
      modelState: null,
      loadModel: () => Promise.resolve(),
      generateText: () => Promise.reject('Not initialized')
    }

    // Provide global composables if needed
    if (options.defaultModelUrl) {
      const { modelState, loadModel, pipeline } = useTrustFormerModel({
        modelUrl: options.defaultModelUrl,
        vocabUrl: options.defaultVocabUrl,
        autoLoad: options.autoLoad ?? true
      })

      const { generateText } = useTrustFormerInference(pipeline)

      app.config.globalProperties.$trustformer = {
        modelState,
        loadModel,
        generateText
      }
    }

    // Global components registration (optional)
    // app.component('TrustFormerTextGenerator', TrustFormerTextGenerator)
    // app.component('TrustFormerChatInterface', TrustFormerChatInterface)
  }
}

export default TrustFormerPlugin
"#
        .to_string()
    }

    /// Get all generated Vue components as a JavaScript object
    pub fn get_all_vue_components(&self) -> Result<Object, JsValue> {
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
            &"composables".into(),
            &self.generate_vue_composables().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"styles".into(),
            &self.generate_vue_component_styles().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"types".into(),
            &self.generate_vue_typescript_definitions().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"package".into(),
            &self.generate_vue_package_json().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"plugin".into(),
            &self.generate_vue_plugin().into(),
        )?;

        Ok(components)
    }
}

/// Check if Vue is available in the environment
#[wasm_bindgen]
pub fn is_vue_available() -> bool {
    let js_code = r#"
        try {
            return typeof Vue !== 'undefined' &&
                   typeof Vue.ref !== 'undefined' &&
                   typeof Vue.computed !== 'undefined';
        } catch (e) {
            return false;
        }
    "#;

    js_sys::eval(js_code)
        .map(|result| result.as_bool().unwrap_or(false))
        .unwrap_or(false)
}

/// Generate a complete Vue integration package
#[wasm_bindgen]
pub fn generate_vue_package(config: VueConfig) -> Result<Object, JsValue> {
    let factory = VueComponentFactory::new(config);
    factory.get_all_vue_components()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vue_config() {
        let mut config = VueConfig::new();
        assert!(config.auto_load_model());
        assert!(config.show_progress());

        config.set_auto_load_model(false);
        assert!(!config.auto_load_model());

        config.set_model_url("https://example.com/model.wasm".to_string());
        assert_eq!(config.model_url(), "https://example.com/model.wasm");
    }

    #[test]
    fn test_vue_component_generation() {
        let config = VueConfig::new();
        let factory = VueComponentFactory::new(config);

        let text_generator = factory.generate_text_generator_component();
        assert!(text_generator.contains("template"));
        assert!(text_generator.contains("script setup"));

        let chat_interface = factory.generate_chat_interface_component();
        assert!(chat_interface.contains("trustformer-chat-interface"));

        let composables = factory.generate_vue_composables();
        assert!(composables.contains("useTrustFormerModel"));
    }

    /// Regression guard: `useTrustFormerInference`/`useTrustFormerStreaming`
    /// used to fabricate their output entirely (a hardcoded response string
    /// and a canned "streaming response" revealed via `setTimeout`) rather
    /// than calling any real WASM generation API. The generated composables
    /// must now call the real `TextGenerationPipeline`/`StreamingGenerator`
    /// exports and must not contain either fabricated phrase.
    #[test]
    fn test_vue_composables_call_real_generation_api_not_fabricated_text() {
        let config = VueConfig::new();
        let factory = VueComponentFactory::new(config);
        let composables = factory.generate_vue_composables();

        assert!(!composables.contains("Generated response for"));
        assert!(!composables.contains("simulated streaming response"));
        assert!(!composables.contains("In reality, this would be actual model inference"));

        assert!(composables.contains("activePipeline.encode"));
        assert!(composables.contains("activePipeline.next_token"));
        assert!(composables.contains("activePipeline.decode"));
        assert!(composables.contains("wasm.StreamingGenerator"));
        assert!(composables.contains("generator.set_pipeline"));
        assert!(composables.contains("start_streaming"));
        assert!(composables.contains("on_token"));
        assert!(composables.contains("on_complete"));
    }
}
