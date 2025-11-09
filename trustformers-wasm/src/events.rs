//! Event system for lifecycle hooks and notifications
//!
//! This module provides a comprehensive event system for TrustformeRS WASM,
//! allowing users to hook into various lifecycle events like model loading,
//! inference operations, errors, and performance metrics.

use js_sys::{Function, Object};
use serde::{Deserialize, Serialize};
use std::borrow::ToOwned;
use std::boxed::Box;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::string::{String, ToString};
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Event types that can be emitted by the system
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EventType {
    // Model lifecycle events
    ModelLoadStart = 1000,
    ModelLoadProgress = 1001,
    ModelLoadComplete = 1002,
    ModelLoadError = 1003,
    ModelUnload = 1004,

    // Inference events
    InferenceStart = 2000,
    InferenceProgress = 2001,
    InferenceComplete = 2002,
    InferenceError = 2003,
    InferenceBatch = 2004,

    // Device events
    DeviceInitialized = 3000,
    DeviceChanged = 3001,
    DeviceError = 3002,
    MemoryWarning = 3003,
    PerformanceAlert = 3004,

    // Storage events
    CacheHit = 4000,
    CacheMiss = 4001,
    CacheEviction = 4002,
    StorageQuotaWarning = 4003,
    StorageError = 4004,

    // Configuration events
    ConfigurationChanged = 5000,
    FeatureToggled = 5001,
    QuantizationEnabled = 5002,
    BatchingEnabled = 5003,

    // Error events
    ErrorOccurred = 6000,
    RecoveryAttempt = 6001,
    RecoverySuccess = 6002,
    RecoveryFailed = 6003,

    // Performance events
    PerformanceMetric = 7000,
    BottleneckDetected = 7001,
    OptimizationSuggestion = 7002,
    BenchmarkComplete = 7003,

    // System events
    InitializationComplete = 8000,
    ShutdownStart = 8001,
    ShutdownComplete = 8002,
    ResourceCleanup = 8003,
}

/// Priority levels for events
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Event data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventData {
    pub event_type: EventType,
    pub priority: EventPriority,
    pub timestamp: f64,
    pub source: String,
    pub data: BTreeMap<String, String>,
    pub error: Option<String>,
    pub duration_ms: Option<f64>,
}

impl EventData {
    pub fn new(event_type: EventType, source: &str) -> Self {
        Self {
            event_type,
            priority: EventPriority::Normal,
            timestamp: js_sys::Date::now(),
            source: source.to_owned(),
            data: BTreeMap::new(),
            error: None,
            duration_ms: None,
        }
    }

    pub fn with_priority(mut self, priority: EventPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_data(mut self, key: &str, value: &str) -> Self {
        self.data.insert(key.to_owned(), value.to_owned());
        self
    }

    pub fn with_error(mut self, error: &str) -> Self {
        self.error = Some(error.to_owned());
        self
    }

    pub fn with_duration(mut self, duration_ms: f64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// Event listener trait for handling events
pub trait EventListener {
    fn handle_event(&mut self, event: &EventData);
}

/// JavaScript callback wrapper
struct JsEventListener {
    callback: Function,
}

impl JsEventListener {
    fn new(callback: Function) -> Self {
        Self { callback }
    }
}

impl EventListener for JsEventListener {
    fn handle_event(&mut self, event: &EventData) {
        // Convert event to JavaScript object
        if let Ok(js_event) = self.event_to_js_object(event) {
            let _ = self.callback.call1(&JsValue::NULL, &js_event);
        }
    }
}

impl JsEventListener {
    fn event_to_js_object(&self, event: &EventData) -> Result<Object, JsValue> {
        let obj = Object::new();

        js_sys::Reflect::set(
            &obj,
            &"type".into(),
            &JsValue::from(event.event_type as u32),
        )?;
        js_sys::Reflect::set(
            &obj,
            &"priority".into(),
            &JsValue::from(event.priority as u32),
        )?;
        js_sys::Reflect::set(&obj, &"timestamp".into(), &JsValue::from(event.timestamp))?;
        js_sys::Reflect::set(&obj, &"source".into(), &JsValue::from(&event.source))?;

        // Convert data map to JavaScript object
        let data_obj = Object::new();
        for (key, value) in &event.data {
            js_sys::Reflect::set(&data_obj, &JsValue::from(key), &JsValue::from(value))?;
        }
        js_sys::Reflect::set(&obj, &"data".into(), &data_obj)?;

        if let Some(ref error) = event.error {
            js_sys::Reflect::set(&obj, &"error".into(), &JsValue::from(error))?;
        }

        if let Some(duration) = event.duration_ms {
            js_sys::Reflect::set(&obj, &"duration".into(), &JsValue::from(duration))?;
        }

        Ok(obj)
    }
}

/// Event emitter for publishing events
pub struct EventEmitter {
    listeners: BTreeMap<EventType, Vec<Box<dyn EventListener>>>,
    global_listeners: Vec<Box<dyn EventListener>>,
    event_history: Vec<EventData>,
    max_history: usize,
    enabled: bool,
}

impl EventEmitter {
    pub fn new() -> Self {
        Self {
            listeners: BTreeMap::new(),
            global_listeners: Vec::new(),
            event_history: Vec::new(),
            max_history: 1000,
            enabled: true,
        }
    }

    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history = max_history;
        self
    }

    /// Add an event listener for a specific event type
    pub fn add_listener(&mut self, event_type: EventType, listener: Box<dyn EventListener>) {
        self.listeners.entry(event_type).or_default().push(listener);
    }

    /// Add a global event listener that receives all events
    pub fn add_global_listener(&mut self, listener: Box<dyn EventListener>) {
        self.global_listeners.push(listener);
    }

    /// Emit an event to all registered listeners
    pub fn emit(&mut self, event: EventData) {
        if !self.enabled {
            return;
        }

        // Store in history
        self.event_history.push(event.clone());
        if self.event_history.len() > self.max_history {
            self.event_history.remove(0);
        }

        // Notify specific listeners
        if let Some(listeners) = self.listeners.get_mut(&event.event_type) {
            for listener in listeners {
                listener.handle_event(&event);
            }
        }

        // Notify global listeners
        for listener in &mut self.global_listeners {
            listener.handle_event(&event);
        }
    }

    /// Remove all listeners for a specific event type
    pub fn remove_listeners(&mut self, event_type: EventType) {
        self.listeners.remove(&event_type);
    }

    /// Clear all listeners
    pub fn clear_listeners(&mut self) {
        self.listeners.clear();
        self.global_listeners.clear();
    }

    /// Get event history
    pub fn get_history(&self) -> &[EventData] {
        &self.event_history
    }

    /// Get events by type from history
    pub fn get_events_by_type(&self, event_type: EventType) -> Vec<&EventData> {
        self.event_history.iter().filter(|e| e.event_type == event_type).collect()
    }

    /// Get events by priority from history
    pub fn get_events_by_priority(&self, priority: EventPriority) -> Vec<&EventData> {
        self.event_history.iter().filter(|e| e.priority >= priority).collect()
    }

    /// Clear event history
    pub fn clear_history(&mut self) {
        self.event_history.clear();
    }

    /// Enable/disable event emission
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if event emission is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for EventEmitter {
    fn default() -> Self {
        Self::new()
    }
}

/// Global event manager with WASM bindings
#[wasm_bindgen]
pub struct EventManager {
    emitter: Rc<RefCell<EventEmitter>>,
}

#[wasm_bindgen]
impl EventManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            emitter: Rc::new(RefCell::new(EventEmitter::new())),
        }
    }

    /// Add a JavaScript function as an event listener
    pub fn add_listener(&mut self, event_type: u32, callback: &Function) -> Result<(), JsValue> {
        if let Ok(event_type) = Self::event_type_from_u32(event_type) {
            let listener = Box::new(JsEventListener::new(callback.clone()));
            self.emitter.borrow_mut().add_listener(event_type, listener);
            Ok(())
        } else {
            Err(JsValue::from_str("Invalid event type"))
        }
    }

    /// Add a global JavaScript function listener
    pub fn add_global_listener(&mut self, callback: &Function) {
        let listener = Box::new(JsEventListener::new(callback.clone()));
        self.emitter.borrow_mut().add_global_listener(listener);
    }

    /// Emit an event from JavaScript
    pub fn emit_event(
        &mut self,
        event_type: u32,
        source: &str,
        data: Option<Object>,
        priority: Option<u32>,
    ) -> Result<(), JsValue> {
        let event_type = Self::event_type_from_u32(event_type)?;
        let priority = if let Some(p) = priority {
            Self::priority_from_u32(p)?
        } else {
            EventPriority::Normal
        };

        let mut event = EventData::new(event_type, source).with_priority(priority);

        // Extract data from JavaScript object
        if let Some(data_obj) = data {
            let keys = Object::keys(&data_obj);
            for i in 0..keys.length() {
                if let (Some(key), Some(value)) = (
                    keys.get(i).as_string(),
                    js_sys::Reflect::get(&data_obj, &keys.get(i)).ok().and_then(|v| v.as_string()),
                ) {
                    event = event.with_data(&key, &value);
                }
            }
        }

        self.emitter.borrow_mut().emit(event);
        Ok(())
    }

    /// Remove listeners for a specific event type
    pub fn remove_listeners(&mut self, event_type: u32) -> Result<(), JsValue> {
        let event_type = Self::event_type_from_u32(event_type)?;
        self.emitter.borrow_mut().remove_listeners(event_type);
        Ok(())
    }

    /// Clear all listeners
    pub fn clear_listeners(&mut self) {
        self.emitter.borrow_mut().clear_listeners();
    }

    /// Get event history as JSON
    pub fn get_history_json(&self) -> Result<String, JsValue> {
        let binding = self.emitter.borrow();
        let history = binding.get_history();
        serde_json::to_string(history).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get events by type as JSON
    pub fn get_events_by_type_json(&self, event_type: u32) -> Result<String, JsValue> {
        let event_type = Self::event_type_from_u32(event_type)?;
        let binding = self.emitter.borrow();
        let events = binding.get_events_by_type(event_type);
        serde_json::to_string(&events).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Clear event history
    pub fn clear_history(&mut self) {
        self.emitter.borrow_mut().clear_history();
    }

    /// Enable/disable event emission
    pub fn set_enabled(&mut self, enabled: bool) {
        self.emitter.borrow_mut().set_enabled(enabled);
    }

    /// Check if event emission is enabled
    pub fn is_enabled(&self) -> bool {
        self.emitter.borrow().is_enabled()
    }

    /// Get event statistics
    pub fn get_statistics(&self) -> Object {
        let emitter = self.emitter.borrow();
        let history = emitter.get_history();

        let obj = Object::new();
        let _ = js_sys::Reflect::set(&obj, &"total_events".into(), &JsValue::from(history.len()));

        // Count by type
        let mut type_counts = BTreeMap::new();
        for event in history {
            *type_counts.entry(event.event_type as u32).or_insert(0) += 1;
        }

        let type_obj = Object::new();
        for (event_type, count) in type_counts {
            let _ =
                js_sys::Reflect::set(&type_obj, &JsValue::from(event_type), &JsValue::from(count));
        }
        let _ = js_sys::Reflect::set(&obj, &"by_type".into(), &type_obj);

        // Count by priority
        let mut priority_counts = [0; 4]; // Low, Normal, High, Critical
        for event in history {
            priority_counts[event.priority as usize] += 1;
        }

        let priority_obj = Object::new();
        let _ = js_sys::Reflect::set(
            &priority_obj,
            &"low".into(),
            &JsValue::from(priority_counts[0]),
        );
        let _ = js_sys::Reflect::set(
            &priority_obj,
            &"normal".into(),
            &JsValue::from(priority_counts[1]),
        );
        let _ = js_sys::Reflect::set(
            &priority_obj,
            &"high".into(),
            &JsValue::from(priority_counts[2]),
        );
        let _ = js_sys::Reflect::set(
            &priority_obj,
            &"critical".into(),
            &JsValue::from(priority_counts[3]),
        );
        let _ = js_sys::Reflect::set(&obj, &"by_priority".into(), &priority_obj);

        obj
    }

    fn event_type_from_u32(event_type: u32) -> Result<EventType, JsValue> {
        match event_type {
            1000 => Ok(EventType::ModelLoadStart),
            1001 => Ok(EventType::ModelLoadProgress),
            1002 => Ok(EventType::ModelLoadComplete),
            1003 => Ok(EventType::ModelLoadError),
            1004 => Ok(EventType::ModelUnload),
            2000 => Ok(EventType::InferenceStart),
            2001 => Ok(EventType::InferenceProgress),
            2002 => Ok(EventType::InferenceComplete),
            2003 => Ok(EventType::InferenceError),
            2004 => Ok(EventType::InferenceBatch),
            3000 => Ok(EventType::DeviceInitialized),
            3001 => Ok(EventType::DeviceChanged),
            3002 => Ok(EventType::DeviceError),
            3003 => Ok(EventType::MemoryWarning),
            3004 => Ok(EventType::PerformanceAlert),
            4000 => Ok(EventType::CacheHit),
            4001 => Ok(EventType::CacheMiss),
            4002 => Ok(EventType::CacheEviction),
            4003 => Ok(EventType::StorageQuotaWarning),
            4004 => Ok(EventType::StorageError),
            5000 => Ok(EventType::ConfigurationChanged),
            5001 => Ok(EventType::FeatureToggled),
            5002 => Ok(EventType::QuantizationEnabled),
            5003 => Ok(EventType::BatchingEnabled),
            6000 => Ok(EventType::ErrorOccurred),
            6001 => Ok(EventType::RecoveryAttempt),
            6002 => Ok(EventType::RecoverySuccess),
            6003 => Ok(EventType::RecoveryFailed),
            7000 => Ok(EventType::PerformanceMetric),
            7001 => Ok(EventType::BottleneckDetected),
            7002 => Ok(EventType::OptimizationSuggestion),
            7003 => Ok(EventType::BenchmarkComplete),
            8000 => Ok(EventType::InitializationComplete),
            8001 => Ok(EventType::ShutdownStart),
            8002 => Ok(EventType::ShutdownComplete),
            8003 => Ok(EventType::ResourceCleanup),
            _ => Err(JsValue::from_str("Invalid event type")),
        }
    }

    fn priority_from_u32(priority: u32) -> Result<EventPriority, JsValue> {
        match priority {
            0 => Ok(EventPriority::Low),
            1 => Ok(EventPriority::Normal),
            2 => Ok(EventPriority::High),
            3 => Ok(EventPriority::Critical),
            _ => Err(JsValue::from_str("Invalid priority")),
        }
    }
}

impl Default for EventManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper trait for components to emit events
pub trait EventEmittable {
    fn get_event_emitter(&mut self) -> &mut EventEmitter;

    fn emit_event(&mut self, event: EventData) {
        self.get_event_emitter().emit(event);
    }

    fn emit_model_load_start(&mut self, model_id: &str, size_mb: f64) {
        let event = EventData::new(EventType::ModelLoadStart, "inference_session")
            .with_data("model_id", model_id)
            .with_data("size_mb", &format!("{size_mb:.2}"));
        self.emit_event(event);
    }

    fn emit_model_load_complete(&mut self, model_id: &str, duration_ms: f64) {
        let event = EventData::new(EventType::ModelLoadComplete, "inference_session")
            .with_data("model_id", model_id)
            .with_duration(duration_ms);
        self.emit_event(event);
    }

    fn emit_inference_start(&mut self, input_shape: &[usize]) {
        let shape_str = format!("{input_shape:?}");
        let event = EventData::new(EventType::InferenceStart, "inference_session")
            .with_data("input_shape", &shape_str);
        self.emit_event(event);
    }

    fn emit_inference_complete(&mut self, duration_ms: f64, output_size: usize) {
        let event = EventData::new(EventType::InferenceComplete, "inference_session")
            .with_duration(duration_ms)
            .with_data("output_size", &output_size.to_string());
        self.emit_event(event);
    }

    fn emit_error(&mut self, error_msg: &str, operation: &str) {
        let event = EventData::new(EventType::ErrorOccurred, "inference_session")
            .with_priority(EventPriority::High)
            .with_error(error_msg)
            .with_data("operation", operation);
        self.emit_event(event);
    }

    fn emit_memory_warning(&mut self, current_mb: f64, limit_mb: f64) {
        let event = EventData::new(EventType::MemoryWarning, "system")
            .with_priority(EventPriority::High)
            .with_data("current_mb", &format!("{current_mb:.2}"))
            .with_data("limit_mb", &format!("{limit_mb:.2}"));
        self.emit_event(event);
    }
}

/// Utility functions for creating common events
impl EventData {
    pub fn model_load_start(model_id: &str, size_mb: f64) -> Self {
        Self::new(EventType::ModelLoadStart, "inference_session")
            .with_data("model_id", model_id)
            .with_data("size_mb", &format!("{size_mb:.2}"))
    }

    pub fn model_load_complete(model_id: &str, duration_ms: f64) -> Self {
        Self::new(EventType::ModelLoadComplete, "inference_session")
            .with_data("model_id", model_id)
            .with_duration(duration_ms)
    }

    pub fn inference_start(input_shape: &[usize]) -> Self {
        let shape_str = format!("{input_shape:?}");
        Self::new(EventType::InferenceStart, "inference_session")
            .with_data("input_shape", &shape_str)
    }

    pub fn inference_complete(duration_ms: f64, output_size: usize) -> Self {
        Self::new(EventType::InferenceComplete, "inference_session")
            .with_duration(duration_ms)
            .with_data("output_size", &output_size.to_string())
    }

    pub fn error_occurred(error_msg: &str, operation: &str) -> Self {
        Self::new(EventType::ErrorOccurred, "system")
            .with_priority(EventPriority::High)
            .with_error(error_msg)
            .with_data("operation", operation)
    }

    pub fn memory_warning(current_mb: f64, limit_mb: f64) -> Self {
        Self::new(EventType::MemoryWarning, "system")
            .with_priority(EventPriority::High)
            .with_data("current_mb", &format!("{current_mb:.2}"))
            .with_data("limit_mb", &format!("{:.2}", limit_mb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_event_creation() {
        let event = EventData::new(EventType::ModelLoadStart, "test")
            .with_data("key", "value")
            .with_priority(EventPriority::High);

        assert_eq!(event.event_type, EventType::ModelLoadStart);
        assert_eq!(event.priority, EventPriority::High);
        assert_eq!(event.source, "test");
        assert_eq!(event.data.get("key"), Some(&"value".to_owned()));
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_event_emitter() {
        let mut emitter = EventEmitter::new();
        let event = EventData::new(EventType::ModelLoadStart, "test");

        emitter.emit(event);

        let history = emitter.get_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].event_type, EventType::ModelLoadStart);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_event_filtering() {
        let mut emitter = EventEmitter::new();

        emitter.emit(EventData::new(EventType::ModelLoadStart, "test"));
        emitter.emit(EventData::new(EventType::InferenceStart, "test"));
        emitter.emit(EventData::new(EventType::ModelLoadStart, "test"));

        let model_events = emitter.get_events_by_type(EventType::ModelLoadStart);
        assert_eq!(model_events.len(), 2);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_event_types() {
        // Test event type values for non-WASM targets
        use EventType::*;
        assert_ne!(ModelLoadStart, ModelLoadComplete);
        assert_ne!(InferenceStart, InferenceComplete);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_event_priorities() {
        // Test event priority values for non-WASM targets
        use EventPriority::*;
        assert_ne!(Low, High);
        assert_ne!(Normal, Critical);
    }
}
