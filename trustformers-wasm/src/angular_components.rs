//! Angular service and directive bindings for TrustFormer WASM
//!
//! This module provides Angular services, directives, and components for easy integration
//! of TrustFormer WASM functionality into Angular applications.

#![allow(dead_code)]

use js_sys::Object;
use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Angular service state for model loading
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct AngularModelState {
    is_loading: bool,
    progress: f64,
    error: Option<String>,
    model_loaded: bool,
}

/// Angular service state for inference
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct AngularInferenceState {
    is_inferring: bool,
    result: Option<String>,
    error: Option<String>,
    inference_time_ms: f64,
}

/// Configuration for Angular services
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct AngularConfig {
    auto_load_model: bool,
    show_progress: bool,
    enable_streaming: bool,
    debug_mode: bool,
    model_url: String,
    fallback_message: String,
}

/// Angular service factory for TrustFormer
#[wasm_bindgen]
pub struct AngularServiceFactory {
    config: AngularConfig,
    service_registry: Vec<ServiceDefinition>,
}

/// Service definition for Angular integration
#[derive(Debug, Clone)]
struct ServiceDefinition {
    name: String,
    service_type: AngularServiceType,
    dependencies: Vec<String>,
    implementation: String,
}

/// Types of Angular services
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AngularServiceType {
    /// Model management service
    ModelService,
    /// Inference service
    InferenceService,
    /// Streaming service
    StreamingService,
    /// Error handling service
    ErrorService,
    /// Performance monitoring service
    PerformanceService,
}

/// Types of Angular components
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AngularComponentType {
    /// Text generation component
    TextGenerator,
    /// Chat interface component
    ChatInterface,
    /// Model loading component
    ModelLoader,
    /// Inference progress component
    InferenceProgress,
    /// Error display component
    ErrorDisplay,
    /// Settings panel component
    SettingsPanel,
}

/// Angular directive manager
#[wasm_bindgen]
pub struct AngularDirectiveManager {
    directives: Vec<DirectiveDefinition>,
}

/// Directive definition
#[derive(Debug, Clone)]
struct DirectiveDefinition {
    name: String,
    directive_type: DirectiveType,
    selector: String,
    implementation: String,
}

/// Types of Angular directives
#[derive(Debug, Clone, Copy, PartialEq)]
enum DirectiveType {
    ModelLoader,
    InferenceProcessor,
    StreamingHandler,
    ErrorHandler,
    PerformanceMonitor,
}

#[wasm_bindgen]
impl AngularConfig {
    /// Create new Angular configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> AngularConfig {
        AngularConfig {
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

impl Default for AngularConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl AngularModelState {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AngularModelState {
        AngularModelState {
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

impl Default for AngularModelState {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl AngularInferenceState {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AngularInferenceState {
        AngularInferenceState {
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

impl Default for AngularInferenceState {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl AngularServiceFactory {
    /// Create new Angular service factory
    #[wasm_bindgen(constructor)]
    pub fn new(config: AngularConfig) -> AngularServiceFactory {
        AngularServiceFactory {
            config,
            service_registry: Vec::new(),
        }
    }

    /// Generate TrustFormer Model Service
    pub fn generate_model_service(&self) -> String {
        format!(
            r#"
import {{ Injectable }} from '@angular/core';
import {{ BehaviorSubject, Observable, throwError }} from 'rxjs';
import {{ catchError, tap }} from 'rxjs/operators';

export interface ModelState {{
  isLoading: boolean;
  progress: number;
  error: string | null;
  modelLoaded: boolean;
}}

export interface ModelConfig {{
  modelUrl: string;
  autoLoad: boolean;
  debugMode: boolean;
}}

@Injectable({{
  providedIn: 'root'
}})
export class TrustFormerModelService {{
  private wasmModule: any = null;
  private session: any = null;

  private modelStateSubject = new BehaviorSubject<ModelState>({{
    isLoading: false,
    progress: 0,
    error: null,
    modelLoaded: false
  }});

  public modelState$ = this.modelStateSubject.asObservable();

  private config: ModelConfig = {{
    modelUrl: '{}',
    autoLoad: {},
    debugMode: {}
  }};

  constructor() {{
    if (this.config.autoLoad && this.config.modelUrl) {{
      this.loadModel(this.config.modelUrl);
    }}
  }}

  async loadModel(modelUrl?: string): Promise<void> {{
    const url = modelUrl || this.config.modelUrl;

    if (!url) {{
      this.updateModelState({{ error: 'No model URL provided' }});
      return;
    }}

    this.updateModelState({{
      isLoading: true,
      progress: 0,
      error: null
    }});

    try {{
      // Initialize WASM module
      if (!this.wasmModule) {{
        this.wasmModule = await import('trustformers-wasm');
        await this.wasmModule.default();
      }}

      // Create inference session
      this.session = new this.wasmModule.InferenceSession('transformer');

      // Initialize with auto device selection
      await this.session.initialize_with_auto_device();

      // Enable debug logging if configured
      if (this.config.debugMode) {{
        const debugConfig = new this.wasmModule.DebugConfig();
        debugConfig.set_log_level(this.wasmModule.LogLevel.Info);
        this.session.enable_debug_logging(debugConfig);
      }}

      // Load model from URL or cache
      await this.session.load_model_with_cache(
        'model_' + url.split('/').pop(),
        url,
        'TrustFormer Model',
        'transformer',
        '1.0.0'
      );

      this.updateModelState({{
        isLoading: false,
        progress: 100,
        modelLoaded: true
      }});

    }} catch (error: any) {{
      console.error('Model loading failed:', error);
      this.updateModelState({{
        isLoading: false,
        error: error.message || 'Failed to load model'
      }});
      throw error;
    }}
  }}

  getSession(): any {{
    return this.session;
  }}

  getCurrentState(): ModelState {{
    return this.modelStateSubject.value;
  }}

  updateConfig(config: Partial<ModelConfig>): void {{
    this.config = {{ ...this.config, ...config }};
  }}

  private updateModelState(update: Partial<ModelState>): void {{
    const currentState = this.modelStateSubject.value;
    this.modelStateSubject.next({{ ...currentState, ...update }});
  }}
}}
"#,
            self.config.model_url, self.config.auto_load_model, self.config.debug_mode
        )
    }

    /// Generate TrustFormer Inference Service
    pub fn generate_inference_service(&self) -> String {
        r#"
import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, throwError } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';
import { TrustFormerModelService } from './model.service';

export interface InferenceState {
  isInferring: boolean;
  result: string | null;
  error: string | null;
  inferenceTimeMs: number;
}

export interface InferenceResult {
  text: string;
  inferenceTime: number;
  tokenCount: number;
}

export interface GenerationOptions {
  maxLength?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
}

@Injectable({
  providedIn: 'root'
})
export class TrustFormerInferenceService {
  private inferenceStateSubject = new BehaviorSubject<InferenceState>({
    isInferring: false,
    result: null,
    error: null,
    inferenceTimeMs: 0
  });

  public inferenceState$ = this.inferenceStateSubject.asObservable();

  constructor(private modelService: TrustFormerModelService) {}

  async generateText(prompt: string, options: GenerationOptions = {}): Promise<InferenceResult> {
    if (!this.modelService.getCurrentState().modelLoaded) {
      throw new Error('Model not loaded. Please load a model first.');
    }

    this.updateInferenceState({
      isInferring: true,
      error: null
    });

    const startTime = performance.now();

    try {
      const session = this.modelService.getSession();
      if (!session) {
        throw new Error('No active session');
      }

      // Create a simple tensor for the prompt (this is simplified)
      // In a real implementation, you'd tokenize the prompt properly
      const wasmModule = await import('trustformers-wasm');
      const inputTensor = new wasmModule.WasmTensor([prompt.length], new Float32Array(prompt.length));

      // Perform inference (simplified)
      const result = inputTensor; // In reality, this would be actual model inference

      const endTime = performance.now();
      const inferenceTime = endTime - startTime;

      // Simulate text generation result
      const generatedText = `Generated response for: "${prompt}"`;

      const inferenceResult: InferenceResult = {
        text: generatedText,
        inferenceTime: Math.round(inferenceTime),
        tokenCount: generatedText.split(' ').length
      };

      this.updateInferenceState({
        isInferring: false,
        result: inferenceResult.text,
        inferenceTimeMs: inferenceTime
      });

      return inferenceResult;

    } catch (error: any) {
      console.error('Inference failed:', error);
      this.updateInferenceState({
        isInferring: false,
        error: error.message || 'Inference failed'
      });
      throw error;
    }
  }

  getCurrentState(): InferenceState {
    return this.inferenceStateSubject.value;
  }

  private updateInferenceState(update: Partial<InferenceState>): void {
    const currentState = this.inferenceStateSubject.value;
    this.inferenceStateSubject.next({ ...currentState, ...update });
  }
}
"#.to_string()
    }

    /// Generate TrustFormer Streaming Service
    pub fn generate_streaming_service(&self) -> String {
        r#"
import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, Subject } from 'rxjs';
import { TrustFormerInferenceService, GenerationOptions } from './inference.service';

export interface StreamState {
  isStreaming: boolean;
  partialResult: string;
  completeResult: string | null;
  error: string | null;
}

export interface StreamToken {
  token: string;
  index: number;
  isComplete: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class TrustFormerStreamingService {
  private streamStateSubject = new BehaviorSubject<StreamState>({
    isStreaming: false,
    partialResult: '',
    completeResult: null,
    error: null
  });

  public streamState$ = this.streamStateSubject.asObservable();

  private tokenSubject = new Subject<StreamToken>();
  public tokens$ = this.tokenSubject.asObservable();

  constructor(private inferenceService: TrustFormerInferenceService) {}

  async startStreaming(prompt: string, options: GenerationOptions = {}): Promise<void> {
    this.updateStreamState({
      isStreaming: true,
      partialResult: '',
      completeResult: null,
      error: null
    });

    try {
      // Simulate streaming by gradually revealing text
      const fullText = `This is a simulated streaming response for: "${prompt}". The text is revealed token by token to simulate real streaming generation.`;
      const tokens = fullText.split(' ');

      for (let i = 0; i < tokens.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 100)); // 100ms delay per token

        const token = tokens[i];
        const partialText = tokens.slice(0, i + 1).join(' ');
        const isComplete = i === tokens.length - 1;

        // Emit token
        this.tokenSubject.next({
          token,
          index: i,
          isComplete
        });

        // Update partial result
        this.updateStreamState({
          partialResult: partialText
        });

        if (isComplete) {
          this.updateStreamState({
            isStreaming: false,
            completeResult: fullText
          });
        }
      }

    } catch (error: any) {
      this.updateStreamState({
        isStreaming: false,
        error: error.message || 'Streaming failed'
      });
    }
  }

  stopStreaming(): void {
    this.updateStreamState({
      isStreaming: false
    });
  }

  getCurrentState(): StreamState {
    return this.streamStateSubject.value;
  }

  private updateStreamState(update: Partial<StreamState>): void {
    const currentState = this.streamStateSubject.value;
    this.streamStateSubject.next({ ...currentState, ...update });
  }
}
"#.to_string()
    }

    /// Generate Text Generator Angular component
    pub fn generate_text_generator_component(&self) -> String {
        format!(
            r#"
import {{ Component, Input, Output, EventEmitter, OnInit, OnDestroy }} from '@angular/core';
import {{ Subscription }} from 'rxjs';
import {{ TrustFormerModelService, ModelState }} from '../services/model.service';
import {{ TrustFormerInferenceService, InferenceState, InferenceResult }} from '../services/inference.service';

export interface Generation {{
  id: number;
  prompt: string;
  result: string;
  timestamp: string;
  inferenceTime: number;
}}

@Component({{
  selector: 'trustformer-text-generator',
  template: `
    <div class="trustformer-text-generator" [ngClass]="className" [ngStyle]="style">
      <!-- Loading state -->
      <div *ngIf="modelState.isLoading && showProgress" class="trustformer-loading">
        <div class="progress-bar">
          <div class="progress-fill" [style.width.%]="modelState.progress"></div>
        </div>
        <p>Loading model... {{{{ modelState.progress | number:'1.0-0' }}}}%</p>
      </div>

      <!-- Error state -->
      <div *ngIf="modelState.error" class="trustformer-error">
        <p>Error loading model: {{{{ modelState.error }}}}</p>
        <button (click)="loadModel()">Retry</button>
      </div>

      <!-- Main interface -->
      <div *ngIf="modelState.modelLoaded">
        <div class="input-section">
          <textarea
            [(ngModel)]="prompt"
            (keydown)="onKeyDown($event)"
            [placeholder]="placeholder"
            [disabled]="inferenceState.isInferring"
            rows="3"
          ></textarea>
          <button
            (click)="handleGenerate()"
            [disabled]="!prompt.trim() || inferenceState.isInferring"
          >
            {{{{ inferenceState.isInferring ? 'Generating...' : 'Generate' }}}}
          </button>
        </div>

        <div class="generations">
          <div *ngFor="let generation of generations" class="generation-item">
            <div class="prompt">
              <strong>Prompt:</strong> {{{{ generation.prompt }}}}
            </div>
            <div class="result">
              <strong>Result:</strong> {{{{ generation.result }}}}
            </div>
            <div class="meta">
              <small>
                Generated in {{{{ generation.inferenceTime }}}}ms at {{{{ generation.timestamp | date:'short' }}}}
              </small>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    /* Component styles will be added here */
  `]
}})
export class TrustFormerTextGeneratorComponent implements OnInit, OnDestroy {{
  @Input() modelUrl: string = '{}';
  @Input() placeholder: string = 'Enter your prompt...';
  @Input() maxLength: number = 100;
  @Input() temperature: number = 0.7;
  @Input() showProgress: boolean = {};
  @Input() style: any = {{}};
  @Input() className: string = '';

  @Output() generate = new EventEmitter<Generation>();

  prompt = '';
  generations: Generation[] = [];

  modelState: ModelState = {{
    isLoading: false,
    progress: 0,
    error: null,
    modelLoaded: false
  }};

  inferenceState: InferenceState = {{
    isInferring: false,
    result: null,
    error: null,
    inferenceTimeMs: 0
  }};

  private subscriptions: Subscription[] = [];

  constructor(
    private modelService: TrustFormerModelService,
    private inferenceService: TrustFormerInferenceService
  ) {{}}

  ngOnInit(): void {{
    // Subscribe to model state
    this.subscriptions.push(
      this.modelService.modelState$.subscribe(state => {{
        this.modelState = state;
      }})
    );

    // Subscribe to inference state
    this.subscriptions.push(
      this.inferenceService.inferenceState$.subscribe(state => {{
        this.inferenceState = state;
      }})
    );

    // Load model if URL is provided and auto-load is enabled
    if (this.modelUrl && {}) {{
      this.loadModel();
    }}
  }}

  ngOnDestroy(): void {{
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }}

  loadModel(): void {{
    this.modelService.loadModel(this.modelUrl);
  }}

  async handleGenerate(): Promise<void> {{
    if (!this.prompt.trim() || !this.modelState.modelLoaded) {{
      return;
    }}

    try {{
      const result = await this.inferenceService.generateText(this.prompt, {{
        maxLength: this.maxLength,
        temperature: this.temperature
      }});

      const newGeneration: Generation = {{
        id: Date.now(),
        prompt: this.prompt,
        result: result.text,
        timestamp: new Date().toISOString(),
        inferenceTime: result.inferenceTime
      }};

      this.generations.unshift(newGeneration);
      this.prompt = '';

      this.generate.emit(newGeneration);
    }} catch (error) {{
      console.error('Generation failed:', error);
    }}
  }}

  onKeyDown(event: KeyboardEvent): void {{
    if (event.key === 'Enter' && event.ctrlKey) {{
      this.handleGenerate();
    }}
  }}
}}
"#,
            self.config.model_url, self.config.show_progress, self.config.auto_load_model
        )
    }

    /// Generate Chat Interface Angular component
    pub fn generate_chat_interface_component(&self) -> String {
        format!(
            r#"
import {{ Component, Input, Output, EventEmitter, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewChecked }} from '@angular/core';
import {{ Subscription }} from 'rxjs';
import {{ TrustFormerModelService, ModelState }} from '../services/model.service';
import {{ TrustFormerInferenceService, InferenceState }} from '../services/inference.service';

export interface Message {{
  id: number;
  type: 'user' | 'assistant' | 'error';
  content: string;
  timestamp: string;
  inferenceTime?: number;
}}

@Component({{
  selector: 'trustformer-chat-interface',
  template: `
    <div class="trustformer-chat-interface" [ngClass]="className" [ngStyle]="style">
      <!-- Loading state -->
      <div *ngIf="modelState.isLoading" class="chat-loading">
        <div class="progress-bar">
          <div class="progress-fill" [style.width.%]="modelState.progress"></div>
        </div>
        <p>{}... {{{{ modelState.progress | number:'1.0-0' }}}}%</p>
      </div>

      <!-- Error state -->
      <div *ngIf="modelState.error" class="trustformer-chat-error">
        <p>Error loading model: {{{{ modelState.error }}}}</p>
        <button (click)="loadModel()">Retry</button>
      </div>

      <!-- Chat interface -->
      <ng-container *ngIf="modelState.modelLoaded">
        <div class="chat-header">
          <h3>AI Assistant</h3>
          <button (click)="clearChat()" [disabled]="messages.length === 0">
            Clear Chat
          </button>
        </div>

        <div class="chat-messages" #messagesContainer>
          <div *ngFor="let message of messages"
               [ngClass]="['message', 'message-' + message.type]">
            <div class="message-content">
              {{{{ message.content }}}}
            </div>
            <div class="message-meta">
              <small>
                {{{{ message.timestamp | date:'short' }}}}
                <span *ngIf="message.inferenceTime"> • {{{{ message.inferenceTime }}}}ms</span>
              </small>
            </div>
          </div>

          <!-- Typing indicator -->
          <div *ngIf="inferenceState.isInferring" class="message message-assistant typing">
            <div class="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>

        <div class="chat-input">
          <textarea
            [(ngModel)]="input"
            (keydown)="onKeyDown($event)"
            [placeholder]="placeholder"
            [disabled]="inferenceState.isInferring"
            rows="2"
          ></textarea>
          <button
            (click)="handleSendMessage()"
            [disabled]="!input.trim() || inferenceState.isInferring"
          >
            Send
          </button>
        </div>
      </ng-container>
    </div>
  `,
  styles: [`
    /* Component styles will be added here */
  `]
}})
export class TrustFormerChatInterfaceComponent implements OnInit, OnDestroy, AfterViewChecked {{
  @Input() modelUrl: string = '{}';
  @Input() placeholder: string = 'Type your message...';
  @Input() systemPrompt: string = 'You are a helpful AI assistant.';
  @Input() maxTokens: number = 150;
  @Input() temperature: number = 0.7;
  @Input() style: any = {{}};
  @Input() className: string = '';

  @Output() message = new EventEmitter<{{ user: Message, assistant: Message }}>();

  @ViewChild('messagesContainer') messagesContainer!: ElementRef;

  messages: Message[] = [];
  input = '';

  modelState: ModelState = {{
    isLoading: false,
    progress: 0,
    error: null,
    modelLoaded: false
  }};

  inferenceState: InferenceState = {{
    isInferring: false,
    result: null,
    error: null,
    inferenceTimeMs: 0
  }};

  private subscriptions: Subscription[] = [];
  private shouldScrollToBottom = false;

  constructor(
    private modelService: TrustFormerModelService,
    private inferenceService: TrustFormerInferenceService
  ) {{}}

  ngOnInit(): void {{
    // Subscribe to model state
    this.subscriptions.push(
      this.modelService.modelState$.subscribe(state => {{
        this.modelState = state;
      }})
    );

    // Subscribe to inference state
    this.subscriptions.push(
      this.inferenceService.inferenceState$.subscribe(state => {{
        this.inferenceState = state;
      }})
    );

    // Load model if URL is provided and auto-load is enabled
    if (this.modelUrl && {}) {{
      this.loadModel();
    }}
  }}

  ngAfterViewChecked(): void {{
    if (this.shouldScrollToBottom) {{
      this.scrollToBottom();
      this.shouldScrollToBottom = false;
    }}
  }}

  ngOnDestroy(): void {{
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }}

  loadModel(): void {{
    this.modelService.loadModel(this.modelUrl);
  }}

  async handleSendMessage(): Promise<void> {{
    if (!this.input.trim() || !this.modelState.modelLoaded || this.inferenceState.isInferring) {{
      return;
    }}

    const userMessage: Message = {{
      id: Date.now(),
      type: 'user',
      content: this.input.trim(),
      timestamp: new Date().toISOString()
    }};

    this.messages.push(userMessage);
    this.input = '';
    this.shouldScrollToBottom = true;

    // Prepare conversation context
    const conversationHistory = this.messages.map(msg =>
      `${{msg.type === 'user' ? 'User' : 'Assistant'}}: ${{msg.content}}`
    ).join('\\n');

    const prompt = `${{this.systemPrompt}}\\n\\n${{conversationHistory}}\\nUser: ${{userMessage.content}}\\nAssistant:`;

    try {{
      const result = await this.inferenceService.generateText(prompt, {{
        maxLength: this.maxTokens,
        temperature: this.temperature
      }});

      const assistantMessage: Message = {{
        id: Date.now() + 1,
        type: 'assistant',
        content: result.text.trim(),
        timestamp: new Date().toISOString(),
        inferenceTime: result.inferenceTime
      }};

      this.messages.push(assistantMessage);
      this.shouldScrollToBottom = true;

      this.message.emit({{ user: userMessage, assistant: assistantMessage }});
    }} catch (error) {{
      console.error('Chat inference failed:', error);
      const errorMessage: Message = {{
        id: Date.now() + 1,
        type: 'error',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString()
      }};
      this.messages.push(errorMessage);
      this.shouldScrollToBottom = true;
    }}
  }}

  onKeyDown(event: KeyboardEvent): void {{
    if (event.key === 'Enter' && !event.shiftKey) {{
      event.preventDefault();
      this.handleSendMessage();
    }}
  }}

  clearChat(): void {{
    this.messages = [];
  }}

  private scrollToBottom(): void {{
    if (this.messagesContainer) {{
      const element = this.messagesContainer.nativeElement;
      element.scrollTop = element.scrollHeight;
    }}
  }}
}}
"#,
            self.config.fallback_message, self.config.model_url, self.config.auto_load_model
        )
    }

    /// Generate Angular module definition
    pub fn generate_angular_module(&self) -> String {
        r#"
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

// Services
import { TrustFormerModelService } from './services/model.service';
import { TrustFormerInferenceService } from './services/inference.service';
import { TrustFormerStreamingService } from './services/streaming.service';

// Components
import { TrustFormerTextGeneratorComponent } from './components/text-generator.component';
import { TrustFormerChatInterfaceComponent } from './components/chat-interface.component';

// Directives
import { TrustFormerModelLoaderDirective } from './directives/model-loader.directive';
import { TrustFormerInferenceDirective } from './directives/inference.directive';

@NgModule({
  declarations: [
    TrustFormerTextGeneratorComponent,
    TrustFormerChatInterfaceComponent,
    TrustFormerModelLoaderDirective,
    TrustFormerInferenceDirective
  ],
  imports: [
    CommonModule,
    FormsModule
  ],
  providers: [
    TrustFormerModelService,
    TrustFormerInferenceService,
    TrustFormerStreamingService
  ],
  exports: [
    TrustFormerTextGeneratorComponent,
    TrustFormerChatInterfaceComponent,
    TrustFormerModelLoaderDirective,
    TrustFormerInferenceDirective
  ]
})
export class TrustFormerModule { }
"#
        .to_string()
    }

    /// Generate Angular directives
    pub fn generate_angular_directives(&self) -> String {
        r#"
// Model Loader Directive
import { Directive, Input, OnInit, OnDestroy, Output, EventEmitter } from '@angular/core';
import { Subscription } from 'rxjs';
import { TrustFormerModelService } from '../services/model.service';

@Directive({
  selector: '[trustformerModelLoader]'
})
export class TrustFormerModelLoaderDirective implements OnInit, OnDestroy {
  @Input('trustformerModelLoader') modelUrl!: string;
  @Input() autoLoad: boolean = true;
  @Output() modelLoaded = new EventEmitter<void>();
  @Output() modelError = new EventEmitter<string>();

  private subscription?: Subscription;

  constructor(private modelService: TrustFormerModelService) {}

  ngOnInit(): void {
    this.subscription = this.modelService.modelState$.subscribe(state => {
      if (state.modelLoaded) {
        this.modelLoaded.emit();
      }
      if (state.error) {
        this.modelError.emit(state.error);
      }
    });

    if (this.autoLoad && this.modelUrl) {
      this.modelService.loadModel(this.modelUrl);
    }
  }

  ngOnDestroy(): void {
    this.subscription?.unsubscribe();
  }
}

// Inference Directive
import { Directive, Input, OnInit, HostListener, Output, EventEmitter } from '@angular/core';
import { TrustFormerInferenceService } from '../services/inference.service';

@Directive({
  selector: '[trustformerInference]'
})
export class TrustFormerInferenceDirective implements OnInit {
  @Input('trustformerInference') prompt!: string;
  @Input() triggerOn: string = 'click';
  @Output() inferenceResult = new EventEmitter<any>();
  @Output() inferenceError = new EventEmitter<string>();

  constructor(private inferenceService: TrustFormerInferenceService) {}

  ngOnInit(): void {}

  @HostListener('click', ['$event'])
  async onClick(event: Event): Promise<void> {
    if (this.triggerOn === 'click') {
      await this.performInference();
    }
  }

  @HostListener('keydown.enter', ['$event'])
  async onEnter(event: KeyboardEvent): Promise<void> {
    if (this.triggerOn === 'enter') {
      await this.performInference();
    }
  }

  private async performInference(): Promise<void> {
    if (!this.prompt) return;

    try {
      const result = await this.inferenceService.generateText(this.prompt);
      this.inferenceResult.emit(result);
    } catch (error: any) {
      this.inferenceError.emit(error.message);
    }
  }
}
"#
        .to_string()
    }

    /// Generate TypeScript definitions for Angular
    pub fn generate_angular_typescript_definitions(&self) -> String {
        r#"
// TypeScript definitions for TrustFormer Angular integration

import { Observable } from 'rxjs';
import { EventEmitter, OnInit, OnDestroy } from '@angular/core';

export interface ModelState {
  isLoading: boolean;
  progress: number;
  error: string | null;
  modelLoaded: boolean;
}

export interface InferenceState {
  isInferring: boolean;
  result: string | null;
  error: string | null;
  inferenceTimeMs: number;
}

export interface InferenceResult {
  text: string;
  inferenceTime: number;
  tokenCount: number;
}

export interface GenerationOptions {
  maxLength?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
}

export interface StreamState {
  isStreaming: boolean;
  partialResult: string;
  completeResult: string | null;
  error: string | null;
}

export interface StreamToken {
  token: string;
  index: number;
  isComplete: boolean;
}

export interface Message {
  id: number;
  type: 'user' | 'assistant' | 'error';
  content: string;
  timestamp: string;
  inferenceTime?: number;
}

export interface Generation {
  id: number;
  prompt: string;
  result: string;
  timestamp: string;
  inferenceTime: number;
}

export interface ModelConfig {
  modelUrl: string;
  autoLoad: boolean;
  debugMode: boolean;
}

// Service Interfaces

export interface ITrustFormerModelService {
  modelState$: Observable<ModelState>;
  loadModel(modelUrl?: string): Promise<void>;
  getSession(): any;
  getCurrentState(): ModelState;
  updateConfig(config: Partial<ModelConfig>): void;
}

export interface ITrustFormerInferenceService {
  inferenceState$: Observable<InferenceState>;
  generateText(prompt: string, options?: GenerationOptions): Promise<InferenceResult>;
  getCurrentState(): InferenceState;
}

export interface ITrustFormerStreamingService {
  streamState$: Observable<StreamState>;
  tokens$: Observable<StreamToken>;
  startStreaming(prompt: string, options?: GenerationOptions): Promise<void>;
  stopStreaming(): void;
  getCurrentState(): StreamState;
}

// Component Interfaces

export interface ITrustFormerTextGeneratorComponent extends OnInit, OnDestroy {
  modelUrl: string;
  placeholder: string;
  maxLength: number;
  temperature: number;
  showProgress: boolean;
  style: any;
  className: string;
  generate: EventEmitter<Generation>;
}

export interface ITrustFormerChatInterfaceComponent extends OnInit, OnDestroy {
  modelUrl: string;
  placeholder: string;
  systemPrompt: string;
  maxTokens: number;
  temperature: number;
  style: any;
  className: string;
  message: EventEmitter<{ user: Message; assistant: Message }>;
}

// Directive Interfaces

export interface ITrustFormerModelLoaderDirective extends OnInit, OnDestroy {
  modelUrl: string;
  autoLoad: boolean;
  modelLoaded: EventEmitter<void>;
  modelError: EventEmitter<string>;
}

export interface ITrustFormerInferenceDirective extends OnInit {
  prompt: string;
  triggerOn: string;
  inferenceResult: EventEmitter<any>;
  inferenceError: EventEmitter<string>;
}
"#
        .to_string()
    }

    /// Generate package.json for Angular integration
    pub fn generate_angular_package_json(&self) -> String {
        r#"
{
  "name": "@trustformers/angular",
  "version": "0.1.0",
  "description": "Angular services, components, and directives for TrustFormer WASM",
  "main": "dist/index.js",
  "module": "dist/esm2022/index.mjs",
  "types": "dist/index.d.ts",
  "files": [
    "dist",
    "src"
  ],
  "scripts": {
    "build": "ng-packagr -p ng-package.json",
    "build:watch": "ng-packagr -p ng-package.json --watch",
    "test": "ng test",
    "test:watch": "ng test --watch",
    "lint": "ng lint",
    "type-check": "tsc --noEmit"
  },
  "peerDependencies": {
    "@angular/core": ">=13.0.0",
    "@angular/common": ">=13.0.0",
    "@angular/forms": ">=13.0.0",
    "rxjs": ">=7.0.0",
    "typescript": ">=4.7.0"
  },
  "dependencies": {
    "trustformers-wasm": "workspace:*"
  },
  "devDependencies": {
    "@angular-devkit/build-angular": "^15.0.0",
    "@angular/cli": "^15.0.0",
    "@angular/compiler": "^15.0.0",
    "@angular/compiler-cli": "^15.0.0",
    "@types/jasmine": "^4.3.0",
    "@types/node": "^18.0.0",
    "jasmine-core": "^4.5.0",
    "karma": "^6.4.0",
    "karma-chrome-headless": "^3.1.0",
    "karma-coverage": "^2.2.0",
    "karma-jasmine": "^5.1.0",
    "karma-jasmine-html-reporter": "^2.0.0",
    "ng-packagr": "^15.0.0",
    "typescript": "^4.9.0"
  },
  "keywords": [
    "trustformers",
    "wasm",
    "angular",
    "ai",
    "machine-learning",
    "transformers",
    "nlp",
    "rxjs"
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

    /// Generate CSS styles for Angular components
    pub fn generate_angular_component_styles(&self) -> String {
        r#"
/* TrustFormer Angular Components Styles */

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
  background-color: #dd1b16; /* Angular red */
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
  background-color: #dd1b16;
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
  background-color: #dd1b16;
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
  background-color: #dd1b16;
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

    /// Get all generated Angular components as a JavaScript object
    pub fn get_all_angular_components(&self) -> Result<Object, JsValue> {
        let components = Object::new();

        js_sys::Reflect::set(
            &components,
            &"ModelService".into(),
            &self.generate_model_service().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"InferenceService".into(),
            &self.generate_inference_service().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"StreamingService".into(),
            &self.generate_streaming_service().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"TextGeneratorComponent".into(),
            &self.generate_text_generator_component().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"ChatInterfaceComponent".into(),
            &self.generate_chat_interface_component().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"Module".into(),
            &self.generate_angular_module().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"Directives".into(),
            &self.generate_angular_directives().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"types".into(),
            &self.generate_angular_typescript_definitions().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"package".into(),
            &self.generate_angular_package_json().into(),
        )?;

        js_sys::Reflect::set(
            &components,
            &"styles".into(),
            &self.generate_angular_component_styles().into(),
        )?;

        Ok(components)
    }
}

/// Check if Angular is available in the environment
#[wasm_bindgen]
pub fn is_angular_available() -> bool {
    let js_code = r#"
        try {
            return typeof ng !== 'undefined' &&
                   typeof ng.core !== 'undefined' &&
                   typeof ng.core.Component !== 'undefined';
        } catch (e) {
            return false;
        }
    "#;

    js_sys::eval(js_code)
        .map(|result| result.as_bool().unwrap_or(false))
        .unwrap_or(false)
}

/// Generate a complete Angular integration package
#[wasm_bindgen]
pub fn generate_angular_package(config: AngularConfig) -> Result<Object, JsValue> {
    let factory = AngularServiceFactory::new(config);
    factory.get_all_angular_components()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angular_config() {
        let mut config = AngularConfig::new();
        assert!(config.auto_load_model());
        assert!(config.show_progress());

        config.set_auto_load_model(false);
        assert!(!config.auto_load_model());

        config.set_model_url("https://example.com/model.wasm".to_string());
        assert_eq!(config.model_url(), "https://example.com/model.wasm");
    }

    #[test]
    fn test_angular_service_generation() {
        let config = AngularConfig::new();
        let factory = AngularServiceFactory::new(config);

        let model_service = factory.generate_model_service();
        assert!(model_service.contains("@Injectable"));
        assert!(model_service.contains("TrustFormerModelService"));

        let inference_service = factory.generate_inference_service();
        assert!(inference_service.contains("TrustFormerInferenceService"));

        let text_generator = factory.generate_text_generator_component();
        assert!(text_generator.contains("@Component"));
    }
}
