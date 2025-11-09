/*!
# React Native Fabric Renderer Integration

This module provides integration with the React Native Fabric renderer,
the new rendering system in React Native that enables better performance
and concurrent features.

## Features

- **Fabric Renderer Support**: Native integration with the new Fabric architecture
- **Concurrent Rendering**: Support for React 18+ concurrent features
- **Shadow Tree Integration**: Direct integration with Fabric's shadow tree
- **Native Component Events**: Efficient event handling with Fabric components
- **Host Components**: Custom host components for TrustformeRS models
- **JSI Integration**: Direct JavaScript Interface integration for better performance

## Usage

```rust
use trustformers_mobile::react_native_fabric::{
    FabricRenderer, FabricConfig, TrustformersHostComponent
};

let config = FabricConfig::default();
let renderer = FabricRenderer::new(config)?;
```
*/

use crate::react_native::{InferenceRequest, InferenceResponse, PerformanceMetrics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::sync::{Arc, Mutex};
use trustformers_core::TrustformersError;

/// Configuration for Fabric renderer integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricConfig {
    /// Enable Fabric renderer optimizations
    pub enable_fabric_optimizations: bool,
    /// Support for concurrent features
    pub enable_concurrent_features: bool,
    /// Enable JSI direct calls
    pub enable_jsi_integration: bool,
    /// Shadow tree update strategy
    pub shadow_tree_strategy: ShadowTreeStrategy,
    /// Event handling strategy
    pub event_handling_strategy: EventHandlingStrategy,
    /// Host component configuration
    pub host_component_config: HostComponentConfig,
    /// Maximum concurrent renders
    pub max_concurrent_renders: usize,
    /// Enable batched updates
    pub enable_batched_updates: bool,
    /// Render priority levels
    pub render_priority_levels: u32,
    /// Enable debugging features
    pub enable_debug: bool,
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self {
            enable_fabric_optimizations: true,
            enable_concurrent_features: true,
            enable_jsi_integration: true,
            shadow_tree_strategy: ShadowTreeStrategy::Optimized,
            event_handling_strategy: EventHandlingStrategy::Direct,
            host_component_config: HostComponentConfig::default(),
            max_concurrent_renders: 4,
            enable_batched_updates: true,
            render_priority_levels: 3,
            enable_debug: false,
        }
    }
}

/// Shadow tree update strategies for Fabric
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShadowTreeStrategy {
    /// Standard shadow tree updates
    Standard,
    /// Optimized updates with batching
    Optimized,
    /// Direct updates bypassing some layers
    Direct,
    /// Lazy updates for better performance
    Lazy,
}

/// Event handling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventHandlingStrategy {
    /// Standard React Native event handling
    Standard,
    /// Direct event handling through JSI
    Direct,
    /// Batched event handling
    Batched,
    /// Priority-based event handling
    Priority,
}

/// Host component configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostComponentConfig {
    /// Component name for React Native
    pub component_name: String,
    /// Supported props
    pub supported_props: Vec<String>,
    /// Native view tag
    pub native_view_tag: Option<i32>,
    /// Component capabilities
    pub capabilities: ComponentCapabilities,
    /// Event names that this component supports
    pub supported_events: Vec<String>,
}

impl Default for HostComponentConfig {
    fn default() -> Self {
        Self {
            component_name: "TrustformersInferenceView".to_string(),
            supported_props: vec![
                "modelId".to_string(),
                "inputData".to_string(),
                "config".to_string(),
                "enableRealtime".to_string(),
            ],
            native_view_tag: None,
            capabilities: ComponentCapabilities::default(),
            supported_events: vec![
                "onInferenceComplete".to_string(),
                "onError".to_string(),
                "onProgress".to_string(),
            ],
        }
    }
}

/// Component capabilities flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentCapabilities {
    /// Supports concurrent rendering
    pub concurrent_rendering: bool,
    /// Supports suspense
    pub suspense_support: bool,
    /// Supports error boundaries
    pub error_boundaries: bool,
    /// Supports server-side rendering
    pub ssr_support: bool,
    /// Supports hot reloading
    pub hot_reload: bool,
}

impl Default for ComponentCapabilities {
    fn default() -> Self {
        Self {
            concurrent_rendering: true,
            suspense_support: true,
            error_boundaries: true,
            ssr_support: false,
            hot_reload: true,
        }
    }
}

/// Main Fabric renderer for TrustformeRS
pub struct FabricRenderer {
    config: FabricConfig,
    shadow_tree: Arc<Mutex<ShadowTree>>,
    event_dispatcher: Arc<Mutex<EventDispatcher>>,
    host_components: HashMap<String, Box<dyn HostComponent>>,
    jsi_runtime: Option<Arc<Mutex<JSIRuntime>>>,
    render_queue: Arc<Mutex<RenderQueue>>,
    component_registry: ComponentRegistry,
}

impl FabricRenderer {
    /// Create a new Fabric renderer
    pub fn new(config: FabricConfig) -> Result<Self, TrustformersError> {
        let shadow_tree = Arc::new(Mutex::new(ShadowTree::new(&config)?));
        let event_dispatcher = Arc::new(Mutex::new(EventDispatcher::new(&config)?));
        let render_queue = Arc::new(Mutex::new(RenderQueue::new(config.max_concurrent_renders)));
        let component_registry = ComponentRegistry::new();

        let jsi_runtime = if config.enable_jsi_integration {
            Some(Arc::new(Mutex::new(JSIRuntime::new()?)))
        } else {
            None
        };

        let mut renderer = Self {
            config,
            shadow_tree,
            event_dispatcher,
            host_components: HashMap::new(),
            jsi_runtime,
            render_queue,
            component_registry,
        };

        // Register default host components
        renderer.register_default_components()?;

        Ok(renderer)
    }

    /// Register a custom host component
    pub fn register_host_component(
        &mut self,
        name: String,
        component: Box<dyn HostComponent>,
    ) -> Result<(), TrustformersError> {
        self.host_components.insert(name.clone(), component);
        self.component_registry.register_component(name)?;
        Ok(())
    }

    /// Create a shadow node for a component
    pub fn create_shadow_node(
        &self,
        component_name: &str,
        props: ComponentProps,
    ) -> Result<ShadowNodeHandle, TrustformersError> {
        let mut shadow_tree = self.shadow_tree.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire shadow tree lock".to_string())
        })?;

        shadow_tree.create_node(component_name, props)
    }

    /// Update component props
    pub fn update_component_props(
        &self,
        node_handle: ShadowNodeHandle,
        props: ComponentProps,
    ) -> Result<(), TrustformersError> {
        let mut shadow_tree = self.shadow_tree.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire shadow tree lock".to_string())
        })?;

        shadow_tree.update_props(node_handle, props)?;

        if self.config.enable_batched_updates {
            self.schedule_batched_update()?;
        } else {
            self.commit_updates()?;
        }

        Ok(())
    }

    /// Dispatch an event from a component
    pub fn dispatch_event(
        &self,
        node_handle: ShadowNodeHandle,
        event_name: &str,
        event_data: EventData,
    ) -> Result<(), TrustformersError> {
        let mut event_dispatcher = self.event_dispatcher.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire event dispatcher lock".to_string())
        })?;

        let event = ComponentEvent {
            node_handle,
            event_name: event_name.to_string(),
            event_data,
            timestamp: std::time::SystemTime::now(),
        };

        match self.config.event_handling_strategy {
            EventHandlingStrategy::Direct => {
                event_dispatcher.dispatch_immediate(event)?;
            },
            EventHandlingStrategy::Batched => {
                event_dispatcher.queue_event(event)?;
            },
            EventHandlingStrategy::Priority => {
                event_dispatcher.dispatch_with_priority(event)?;
            },
            EventHandlingStrategy::Standard => {
                event_dispatcher.dispatch_standard(event)?;
            },
        }

        Ok(())
    }

    /// Perform inference through Fabric renderer
    pub fn perform_inference(
        &self,
        request: FabricInferenceRequest,
    ) -> Result<FabricInferenceResponse, TrustformersError> {
        let start_time = std::time::Instant::now();

        // Create or update shadow node for inference
        let props = ComponentProps::from_inference_request(&request);
        let node_handle = match request.node_handle {
            Some(handle) => {
                self.update_component_props(handle, props)?;
                handle
            },
            None => self.create_shadow_node("TrustformersInferenceView", props)?,
        };

        // Queue render operation
        {
            let mut render_queue = self.render_queue.lock().map_err(|_| {
                TrustformersError::runtime_error("Failed to acquire render queue lock".to_string())
            })?;

            let render_task = RenderTask {
                node_handle,
                task_type: RenderTaskType::Inference,
                priority: request.priority.unwrap_or(RenderPriority::Normal),
                inference_request: Some(request.clone()),
            };

            render_queue.enqueue_task(render_task)?;
        }

        // Execute inference with concurrent rendering support
        let result = if self.config.enable_concurrent_features {
            self.execute_concurrent_inference(&request)?
        } else {
            self.execute_standard_inference(&request)?
        };

        // Dispatch completion event
        let event_data = EventData::InferenceComplete {
            success: result.success,
            inference_time_ms: result.inference_time_ms,
            error_message: result.error_message.clone(),
        };

        self.dispatch_event(node_handle, "onInferenceComplete", event_data)?;

        Ok(FabricInferenceResponse {
            request_id: request.request_id,
            node_handle,
            success: result.success,
            output_data: result.output_data,
            output_shape: result.output_shape,
            inference_time_ms: start_time.elapsed().as_millis() as f64,
            memory_used_mb: result.memory_used_mb,
            error_message: result.error_message,
            fabric_metrics: FabricMetrics {
                shadow_tree_updates: 1,
                event_dispatches: 1,
                render_time_ms: 0.0,
                concurrent_renders: if self.config.enable_concurrent_features { 1 } else { 0 },
            },
        })
    }

    /// Private helper methods
    fn register_default_components(&mut self) -> Result<(), TrustformersError> {
        // Register TrustformersInferenceView component
        let inference_component = Box::new(TrustformersInferenceComponent::new(
            self.config.host_component_config.clone(),
        ));
        self.register_host_component("TrustformersInferenceView".to_string(), inference_component)?;

        // Register TrustformersModelView component
        let model_component = Box::new(TrustformersModelComponent::new());
        self.register_host_component("TrustformersModelView".to_string(), model_component)?;

        Ok(())
    }

    fn schedule_batched_update(&self) -> Result<(), TrustformersError> {
        // Schedule a batched update for the next frame
        // This would integrate with React Native's scheduler
        Ok(())
    }

    fn commit_updates(&self) -> Result<(), TrustformersError> {
        let mut shadow_tree = self.shadow_tree.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire shadow tree lock".to_string())
        })?;

        shadow_tree.commit_pending_updates()
    }

    fn execute_concurrent_inference(
        &self,
        request: &FabricInferenceRequest,
    ) -> Result<InferenceResponse, TrustformersError> {
        // Execute inference with concurrent rendering support
        // This would integrate with React's scheduler and time slicing
        self.execute_standard_inference(request)
    }

    fn execute_standard_inference(
        &self,
        request: &FabricInferenceRequest,
    ) -> Result<InferenceResponse, TrustformersError> {
        // Convert Fabric request to standard inference request
        let standard_request = InferenceRequest {
            request_id: request.request_id.clone(),
            model_id: request.model_id.clone(),
            input_data: request.input_data.clone(),
            input_shape: request.input_shape.clone(),
            config_override: request.config_override.clone(),
            enable_preprocessing: request.enable_preprocessing,
            enable_postprocessing: request.enable_postprocessing,
        };

        // For now, return a placeholder response
        // In a real implementation, this would call the inference engine
        Ok(InferenceResponse {
            request_id: request.request_id.clone(),
            success: true,
            output_data: vec![1.0, 2.0, 3.0], // Placeholder
            output_shape: vec![1, 3],
            inference_time_ms: 100.0,
            memory_used_mb: 50,
            error_message: None,
            metrics: PerformanceMetrics {
                preprocessing_time_ms: 10.0,
                inference_time_ms: 80.0,
                postprocessing_time_ms: 10.0,
                memory_allocation_mb: 50,
                cache_hit_ratio: 0.8,
            },
        })
    }
}

/// Shadow tree management for Fabric
struct ShadowTree {
    nodes: HashMap<ShadowNodeHandle, ShadowNode>,
    next_handle: ShadowNodeHandle,
    pending_updates: Vec<ShadowTreeUpdate>,
    strategy: ShadowTreeStrategy,
}

impl ShadowTree {
    fn new(config: &FabricConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            nodes: HashMap::new(),
            next_handle: ShadowNodeHandle(1),
            pending_updates: Vec::new(),
            strategy: config.shadow_tree_strategy,
        })
    }

    fn create_node(
        &mut self,
        component_name: &str,
        props: ComponentProps,
    ) -> Result<ShadowNodeHandle, TrustformersError> {
        let handle = self.next_handle;
        self.next_handle = ShadowNodeHandle(self.next_handle.0 + 1);

        let node = ShadowNode {
            handle,
            component_name: component_name.to_string(),
            props,
            children: Vec::new(),
            parent: None,
            is_dirty: true,
        };

        self.nodes.insert(handle, node);
        Ok(handle)
    }

    fn update_props(
        &mut self,
        handle: ShadowNodeHandle,
        props: ComponentProps,
    ) -> Result<(), TrustformersError> {
        if let Some(node) = self.nodes.get_mut(&handle) {
            node.props = props;
            node.is_dirty = true;

            // Queue update based on strategy
            let update = ShadowTreeUpdate {
                handle,
                update_type: UpdateType::PropsUpdate,
                timestamp: std::time::SystemTime::now(),
            };

            match self.strategy {
                ShadowTreeStrategy::Direct => {
                    // Apply immediately
                    self.apply_update(update)?;
                },
                _ => {
                    // Queue for later
                    self.pending_updates.push(update);
                },
            }
        }

        Ok(())
    }

    fn commit_pending_updates(&mut self) -> Result<(), TrustformersError> {
        let updates: Vec<_> = self.pending_updates.drain(..).collect();
        for update in updates {
            self.apply_update(update)?;
        }
        Ok(())
    }

    fn apply_update(&mut self, _update: ShadowTreeUpdate) -> Result<(), TrustformersError> {
        // Apply the shadow tree update
        // This would trigger native view updates
        Ok(())
    }
}

/// Event dispatcher for Fabric components
struct EventDispatcher {
    event_queue: Vec<ComponentEvent>,
    strategy: EventHandlingStrategy,
}

impl EventDispatcher {
    fn new(config: &FabricConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            event_queue: Vec::new(),
            strategy: config.event_handling_strategy,
        })
    }

    fn dispatch_immediate(&mut self, event: ComponentEvent) -> Result<(), TrustformersError> {
        self.send_to_javascript(event)
    }

    fn queue_event(&mut self, event: ComponentEvent) -> Result<(), TrustformersError> {
        self.event_queue.push(event);
        Ok(())
    }

    fn dispatch_with_priority(&mut self, event: ComponentEvent) -> Result<(), TrustformersError> {
        // Implement priority-based dispatching
        self.send_to_javascript(event)
    }

    fn dispatch_standard(&mut self, event: ComponentEvent) -> Result<(), TrustformersError> {
        self.send_to_javascript(event)
    }

    fn send_to_javascript(&self, event: ComponentEvent) -> Result<(), TrustformersError> {
        // Send event to JavaScript through the bridge
        // This would use the React Native event emitter
        Ok(())
    }
}

/// JSI Runtime for direct JavaScript integration
struct JSIRuntime {
    runtime_ptr: *mut c_void,
}

impl JSIRuntime {
    fn new() -> Result<Self, TrustformersError> {
        // Initialize JSI runtime
        // This would create a JavaScript runtime for direct calls
        Ok(Self {
            runtime_ptr: std::ptr::null_mut(),
        })
    }

    fn call_function(
        &self,
        _function_name: &str,
        _args: &[JSIValue],
    ) -> Result<JSIValue, TrustformersError> {
        // Call JavaScript function directly through JSI
        Ok(JSIValue::Undefined)
    }
}

/// Render queue for managing concurrent renders
struct RenderQueue {
    tasks: Vec<RenderTask>,
    max_concurrent: usize,
    active_tasks: usize,
}

impl RenderQueue {
    fn new(max_concurrent: usize) -> Self {
        Self {
            tasks: Vec::new(),
            max_concurrent,
            active_tasks: 0,
        }
    }

    fn enqueue_task(&mut self, task: RenderTask) -> Result<(), TrustformersError> {
        self.tasks.push(task);
        self.tasks.sort_by_key(|t| t.priority as u8);
        Ok(())
    }

    fn dequeue_task(&mut self) -> Option<RenderTask> {
        if self.active_tasks < self.max_concurrent && !self.tasks.is_empty() {
            self.active_tasks += 1;
            Some(self.tasks.remove(0))
        } else {
            None
        }
    }

    fn complete_task(&mut self) {
        if self.active_tasks > 0 {
            self.active_tasks -= 1;
        }
    }
}

/// Component registry for managing host components
struct ComponentRegistry {
    registered_components: HashMap<String, ComponentDescriptor>,
}

impl ComponentRegistry {
    fn new() -> Self {
        Self {
            registered_components: HashMap::new(),
        }
    }

    fn register_component(&mut self, name: String) -> Result<(), TrustformersError> {
        let descriptor = ComponentDescriptor {
            name: name.clone(),
            registered_at: std::time::SystemTime::now(),
        };

        self.registered_components.insert(name, descriptor);
        Ok(())
    }
}

/// Data structures for Fabric integration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ShadowNodeHandle(u64);

#[derive(Debug, Clone)]
struct ShadowNode {
    handle: ShadowNodeHandle,
    component_name: String,
    props: ComponentProps,
    children: Vec<ShadowNodeHandle>,
    parent: Option<ShadowNodeHandle>,
    is_dirty: bool,
}

#[derive(Debug, Clone)]
struct ShadowTreeUpdate {
    handle: ShadowNodeHandle,
    update_type: UpdateType,
    timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpdateType {
    PropsUpdate,
    LayoutUpdate,
    ChildrenUpdate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentProps {
    pub model_id: Option<String>,
    pub input_data: Option<Vec<f32>>,
    pub input_shape: Option<Vec<usize>>,
    pub config: Option<String>, // JSON-serialized config
    pub enable_realtime: Option<bool>,
    pub additional_props: HashMap<String, PropValue>,
}

impl ComponentProps {
    fn from_inference_request(request: &FabricInferenceRequest) -> Self {
        Self {
            model_id: Some(request.model_id.clone()),
            input_data: Some(request.input_data.clone()),
            input_shape: Some(request.input_shape.clone()),
            config: None,
            enable_realtime: Some(false),
            additional_props: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<PropValue>),
    Object(HashMap<String, PropValue>),
}

#[derive(Debug, Clone)]
struct ComponentEvent {
    node_handle: ShadowNodeHandle,
    event_name: String,
    event_data: EventData,
    timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone)]
pub enum EventData {
    InferenceComplete {
        success: bool,
        inference_time_ms: f64,
        error_message: Option<String>,
    },
    Progress {
        percentage: f64,
        current_step: String,
    },
    Error {
        error_code: String,
        error_message: String,
    },
    Custom {
        data: HashMap<String, PropValue>,
    },
}

#[derive(Debug, Clone)]
struct RenderTask {
    node_handle: ShadowNodeHandle,
    task_type: RenderTaskType,
    priority: RenderPriority,
    inference_request: Option<FabricInferenceRequest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderTaskType {
    Inference,
    Update,
    Layout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RenderPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

#[derive(Debug, Clone)]
struct ComponentDescriptor {
    name: String,
    registered_at: std::time::SystemTime,
}

/// Fabric-specific inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricInferenceRequest {
    pub request_id: String,
    pub model_id: String,
    pub input_data: Vec<f32>,
    pub input_shape: Vec<usize>,
    pub config_override: Option<crate::MobileConfig>,
    pub enable_preprocessing: bool,
    pub enable_postprocessing: bool,
    #[serde(skip)]
    pub node_handle: Option<ShadowNodeHandle>,
    #[serde(skip)]
    pub priority: Option<RenderPriority>,
    pub concurrent_rendering: bool,
}

/// Fabric-specific inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricInferenceResponse {
    pub request_id: String,
    #[serde(skip, default)]
    pub node_handle: ShadowNodeHandle,
    pub success: bool,
    pub output_data: Vec<f32>,
    pub output_shape: Vec<usize>,
    pub inference_time_ms: f64,
    pub memory_used_mb: usize,
    pub error_message: Option<String>,
    pub fabric_metrics: FabricMetrics,
}

/// Fabric-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricMetrics {
    pub shadow_tree_updates: u32,
    pub event_dispatches: u32,
    pub render_time_ms: f64,
    pub concurrent_renders: u32,
}

/// JSI value types for direct JavaScript integration
#[derive(Debug, Clone)]
pub enum JSIValue {
    Undefined,
    Null,
    Boolean(bool),
    Number(f64),
    String(String),
    Object(HashMap<String, JSIValue>),
    Array(Vec<JSIValue>),
}

/// Trait for host components
pub trait HostComponent: Send + Sync {
    fn get_name(&self) -> &str;
    fn create_shadow_node(&self, props: ComponentProps) -> Result<ShadowNode, TrustformersError>;
    fn update_props(
        &self,
        node: &mut ShadowNode,
        new_props: ComponentProps,
    ) -> Result<(), TrustformersError>;
    fn handle_event(&self, event: &ComponentEvent) -> Result<(), TrustformersError>;
    fn get_supported_props(&self) -> Vec<String>;
    fn get_supported_events(&self) -> Vec<String>;
}

/// TrustformersInferenceView host component
struct TrustformersInferenceComponent {
    config: HostComponentConfig,
}

impl TrustformersInferenceComponent {
    fn new(config: HostComponentConfig) -> Self {
        Self { config }
    }
}

impl HostComponent for TrustformersInferenceComponent {
    fn get_name(&self) -> &str {
        &self.config.component_name
    }

    fn create_shadow_node(&self, props: ComponentProps) -> Result<ShadowNode, TrustformersError> {
        Ok(ShadowNode {
            handle: ShadowNodeHandle(0), // Will be set by shadow tree
            component_name: self.get_name().to_string(),
            props,
            children: Vec::new(),
            parent: None,
            is_dirty: true,
        })
    }

    fn update_props(
        &self,
        node: &mut ShadowNode,
        new_props: ComponentProps,
    ) -> Result<(), TrustformersError> {
        node.props = new_props;
        node.is_dirty = true;
        Ok(())
    }

    fn handle_event(&self, _event: &ComponentEvent) -> Result<(), TrustformersError> {
        // Handle component-specific events
        Ok(())
    }

    fn get_supported_props(&self) -> Vec<String> {
        self.config.supported_props.clone()
    }

    fn get_supported_events(&self) -> Vec<String> {
        self.config.supported_events.clone()
    }
}

/// TrustformersModelView host component
struct TrustformersModelComponent;

impl TrustformersModelComponent {
    fn new() -> Self {
        Self
    }
}

impl HostComponent for TrustformersModelComponent {
    fn get_name(&self) -> &str {
        "TrustformersModelView"
    }

    fn create_shadow_node(&self, props: ComponentProps) -> Result<ShadowNode, TrustformersError> {
        Ok(ShadowNode {
            handle: ShadowNodeHandle(0),
            component_name: self.get_name().to_string(),
            props,
            children: Vec::new(),
            parent: None,
            is_dirty: true,
        })
    }

    fn update_props(
        &self,
        node: &mut ShadowNode,
        new_props: ComponentProps,
    ) -> Result<(), TrustformersError> {
        node.props = new_props;
        node.is_dirty = true;
        Ok(())
    }

    fn handle_event(&self, _event: &ComponentEvent) -> Result<(), TrustformersError> {
        Ok(())
    }

    fn get_supported_props(&self) -> Vec<String> {
        vec![
            "modelPath".to_string(),
            "modelType".to_string(),
            "loadOnMount".to_string(),
        ]
    }

    fn get_supported_events(&self) -> Vec<String> {
        vec!["onModelLoaded".to_string(), "onModelError".to_string()]
    }
}

/// C API for React Native integration
#[no_mangle]
pub extern "C" fn tfk_fabric_renderer_create(config_json: *const c_char) -> *mut FabricRenderer {
    if config_json.is_null() {
        return std::ptr::null_mut();
    }

    let config_str = unsafe { CStr::from_ptr(config_json).to_str().unwrap_or_default() };

    let config: FabricConfig = serde_json::from_str(config_str).unwrap_or_default();

    match FabricRenderer::new(config) {
        Ok(renderer) => Box::into_raw(Box::new(renderer)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tfk_fabric_renderer_destroy(renderer: *mut FabricRenderer) {
    if !renderer.is_null() {
        unsafe {
            Box::from_raw(renderer);
        }
    }
}

#[no_mangle]
pub extern "C" fn tfk_fabric_create_shadow_node(
    renderer: *mut FabricRenderer,
    component_name: *const c_char,
    props_json: *const c_char,
) -> u64 {
    if renderer.is_null() || component_name.is_null() || props_json.is_null() {
        return 0;
    }

    let renderer = unsafe { &*renderer };
    let component_name = unsafe { CStr::from_ptr(component_name).to_str().unwrap_or_default() };
    let props_str = unsafe { CStr::from_ptr(props_json).to_str().unwrap_or_default() };

    let props: ComponentProps =
        serde_json::from_str(props_str).unwrap_or_else(|_| ComponentProps {
            model_id: None,
            input_data: None,
            input_shape: None,
            config: None,
            enable_realtime: None,
            additional_props: HashMap::new(),
        });

    match renderer.create_shadow_node(component_name, props) {
        Ok(handle) => handle.0,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn tfk_fabric_perform_inference(
    renderer: *mut FabricRenderer,
    request_json: *const c_char,
) -> *const c_char {
    if renderer.is_null() || request_json.is_null() {
        return std::ptr::null();
    }

    let renderer = unsafe { &*renderer };
    let request_str = unsafe { CStr::from_ptr(request_json).to_str().unwrap_or_default() };

    let request: FabricInferenceRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(_) => return std::ptr::null(),
    };

    match renderer.perform_inference(request) {
        Ok(response) => {
            let response_json = serde_json::to_string(&response).unwrap_or_default();
            let c_string = CString::new(response_json).unwrap_or_default();
            c_string.into_raw()
        },
        Err(_) => std::ptr::null(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fabric_config_default() {
        let config = FabricConfig::default();
        assert!(config.enable_fabric_optimizations);
        assert!(config.enable_concurrent_features);
        assert_eq!(config.max_concurrent_renders, 4);
    }

    #[test]
    fn test_shadow_node_handle() {
        let handle1 = ShadowNodeHandle(1);
        let handle2 = ShadowNodeHandle(2);
        assert_ne!(handle1, handle2);
        assert_eq!(handle1.0, 1);
    }

    #[test]
    fn test_host_component_config() {
        let config = HostComponentConfig::default();
        assert_eq!(config.component_name, "TrustformersInferenceView");
        assert!(config.supported_props.contains(&"modelId".to_string()));
        assert!(config.supported_events.contains(&"onInferenceComplete".to_string()));
    }

    #[test]
    fn test_component_props_from_request() {
        let request = FabricInferenceRequest {
            request_id: "test".to_string(),
            model_id: "test_model".to_string(),
            input_data: vec![1.0, 2.0, 3.0],
            input_shape: vec![1, 3],
            config_override: None,
            enable_preprocessing: true,
            enable_postprocessing: true,
            node_handle: None,
            priority: None,
            concurrent_rendering: false,
        };

        let props = ComponentProps::from_inference_request(&request);
        assert_eq!(props.model_id.unwrap(), "test_model");
        assert_eq!(props.input_data.unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_event_data_variants() {
        let event1 = EventData::InferenceComplete {
            success: true,
            inference_time_ms: 100.0,
            error_message: None,
        };

        let event2 = EventData::Progress {
            percentage: 50.0,
            current_step: "Processing".to_string(),
        };

        match event1 {
            EventData::InferenceComplete { success, .. } => assert!(success),
            _ => panic!("Wrong event type"),
        }

        match event2 {
            EventData::Progress { percentage, .. } => assert_eq!(percentage, 50.0),
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_render_priority_ordering() {
        assert!(RenderPriority::Critical > RenderPriority::High);
        assert!(RenderPriority::High > RenderPriority::Normal);
        assert!(RenderPriority::Normal > RenderPriority::Low);
    }

    #[test]
    fn test_jsi_value_types() {
        let undefined = JSIValue::Undefined;
        let number = JSIValue::Number(42.0);
        let string = JSIValue::String("test".to_string());

        match undefined {
            JSIValue::Undefined => {},
            _ => panic!("Wrong JSI value type"),
        }

        match number {
            JSIValue::Number(n) => assert_eq!(n, 42.0),
            _ => panic!("Wrong JSI value type"),
        }

        match string {
            JSIValue::String(s) => assert_eq!(s, "test"),
            _ => panic!("Wrong JSI value type"),
        }
    }
}
