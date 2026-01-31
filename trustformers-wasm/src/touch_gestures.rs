#![allow(dead_code)]

use core::cell::RefCell;
use js_sys::{Function, Object, WeakMap};
use serde::{Deserialize, Serialize};
use std::boxed::Box;
use std::collections::BTreeMap;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{window, Element, HtmlElement, TouchEvent, TouchList};

// Global registry for gesture recognizers
thread_local! {
    static GESTURE_REGISTRY: RefCell<WeakMap> = RefCell::new(WeakMap::new());
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GestureType {
    Tap,
    DoubleTap,
    LongPress,
    Swipe,
    Pinch,
    Rotate,
    Pan,
    TwoFingerTap,
    ThreeFingerTap,
    Unknown,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SwipeDirection {
    Up,
    Down,
    Left,
    Right,
    Unknown,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TouchPoint {
    pub x: f64,
    pub y: f64,
    pub id: i32,
    pub timestamp: f64,
}

#[wasm_bindgen]
impl TouchPoint {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64, id: i32, timestamp: f64) -> Self {
        Self {
            x,
            y,
            id,
            timestamp,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f64 {
        self.x
    }

    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f64 {
        self.y
    }

    #[wasm_bindgen(getter)]
    pub fn id(&self) -> i32 {
        self.id
    }

    #[wasm_bindgen(getter)]
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureEvent {
    gesture_type: GestureType,
    start_point: TouchPoint,
    end_point: Option<TouchPoint>,
    swipe_direction: Option<SwipeDirection>,
    distance: f64,
    duration: f64,
    scale: f64,
    rotation: f64,
    velocity: f64,
    touch_count: u32,
}

#[wasm_bindgen]
impl GestureEvent {
    #[wasm_bindgen(getter)]
    pub fn gesture_type(&self) -> GestureType {
        self.gesture_type
    }

    #[wasm_bindgen(getter)]
    pub fn start_point(&self) -> TouchPoint {
        self.start_point.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn end_point(&self) -> Option<TouchPoint> {
        self.end_point.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn swipe_direction(&self) -> Option<SwipeDirection> {
        self.swipe_direction
    }

    #[wasm_bindgen(getter)]
    pub fn distance(&self) -> f64 {
        self.distance
    }

    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f64 {
        self.duration
    }

    #[wasm_bindgen(getter)]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    #[wasm_bindgen(getter)]
    pub fn rotation(&self) -> f64 {
        self.rotation
    }

    #[wasm_bindgen(getter)]
    pub fn velocity(&self) -> f64 {
        self.velocity
    }

    #[wasm_bindgen(getter)]
    pub fn touch_count(&self) -> u32 {
        self.touch_count
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureConfig {
    pub tap_timeout: f64,
    pub double_tap_interval: f64,
    pub long_press_duration: f64,
    pub swipe_min_distance: f64,
    pub swipe_max_time: f64,
    pub pinch_threshold: f64,
    pub rotation_threshold: f64,
    pub pan_threshold: f64,
    pub velocity_threshold: f64,
    pub enable_prevent_default: bool,
}

impl Default for GestureConfig {
    fn default() -> Self {
        Self {
            tap_timeout: 300.0,           // 300ms for tap detection
            double_tap_interval: 400.0,   // 400ms between taps for double tap
            long_press_duration: 500.0,   // 500ms for long press
            swipe_min_distance: 50.0,     // 50px minimum swipe distance
            swipe_max_time: 300.0,        // 300ms maximum swipe time
            pinch_threshold: 10.0,        // 10px threshold for pinch detection
            rotation_threshold: 5.0,      // 5 degrees threshold for rotation
            pan_threshold: 10.0,          // 10px threshold for pan detection
            velocity_threshold: 0.1,      // 0.1 px/ms minimum velocity
            enable_prevent_default: true, // Prevent default touch behaviors
        }
    }
}

#[wasm_bindgen]
impl GestureConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen(getter)]
    pub fn tap_timeout(&self) -> f64 {
        self.tap_timeout
    }

    #[wasm_bindgen(setter)]
    pub fn set_tap_timeout(&mut self, value: f64) {
        self.tap_timeout = value;
    }

    #[wasm_bindgen(getter)]
    pub fn double_tap_interval(&self) -> f64 {
        self.double_tap_interval
    }

    #[wasm_bindgen(setter)]
    pub fn set_double_tap_interval(&mut self, value: f64) {
        self.double_tap_interval = value;
    }

    #[wasm_bindgen(getter)]
    pub fn long_press_duration(&self) -> f64 {
        self.long_press_duration
    }

    #[wasm_bindgen(setter)]
    pub fn set_long_press_duration(&mut self, value: f64) {
        self.long_press_duration = value;
    }

    #[wasm_bindgen(getter)]
    pub fn swipe_min_distance(&self) -> f64 {
        self.swipe_min_distance
    }

    #[wasm_bindgen(setter)]
    pub fn set_swipe_min_distance(&mut self, value: f64) {
        self.swipe_min_distance = value;
    }

    #[wasm_bindgen(getter)]
    pub fn enable_prevent_default(&self) -> bool {
        self.enable_prevent_default
    }

    #[wasm_bindgen(setter)]
    pub fn set_enable_prevent_default(&mut self, value: bool) {
        self.enable_prevent_default = value;
    }
}

#[wasm_bindgen]
pub struct TouchGestureRecognizer {
    config: GestureConfig,
    target_element: Option<Element>,
    active_touches: Vec<TouchPoint>,
    gesture_start_time: f64,
    last_tap_time: f64,
    tap_count: u32,
    initial_distance: f64,
    initial_angle: f64,
    callbacks: Object,
    long_press_timer: Option<i32>,
    event_handlers: BTreeMap<String, Closure<dyn FnMut(TouchEvent)>>,
    touch_history: Vec<TouchPoint>,
    is_attached: bool,
}

#[wasm_bindgen]
impl TouchGestureRecognizer {
    #[wasm_bindgen(constructor)]
    pub fn new(config: GestureConfig) -> Self {
        Self {
            config,
            target_element: None,
            active_touches: Vec::new(),
            gesture_start_time: 0.0,
            last_tap_time: 0.0,
            tap_count: 0,
            initial_distance: 0.0,
            initial_angle: 0.0,
            callbacks: Object::new(),
            long_press_timer: None,
            event_handlers: BTreeMap::new(),
            touch_history: Vec::new(),
            is_attached: false,
        }
    }

    pub fn attach_to_element(&mut self, element: &Element) -> Result<(), JsValue> {
        if self.is_attached {
            self.detach_from_element()?;
        }

        self.target_element = Some(element.clone());

        // Register this recognizer in the global registry
        GESTURE_REGISTRY.with(|registry| {
            let reg = registry.borrow();
            reg.set(
                &element.clone().into(),
                &JsValue::from(self as *const _ as u32),
            );
        });

        // Store element reference for creating event handlers that capture it
        let element_ref = element.clone();

        // Create touch start handler
        let touchstart = {
            let element_clone = element_ref.clone();
            Closure::wrap(Box::new(move |event: TouchEvent| {
                if let Some(recognizer) = get_recognizer_from_element(&element_clone) {
                    recognizer.handle_touch_start(&event);
                }
            }) as Box<dyn FnMut(TouchEvent)>)
        };

        // Create touch move handler
        let touchmove = {
            let element_clone = element_ref.clone();
            Closure::wrap(Box::new(move |event: TouchEvent| {
                if let Some(recognizer) = get_recognizer_from_element(&element_clone) {
                    recognizer.handle_touch_move(&event);
                }
            }) as Box<dyn FnMut(TouchEvent)>)
        };

        // Create touch end handler
        let touchend = {
            let element_clone = element_ref.clone();
            Closure::wrap(Box::new(move |event: TouchEvent| {
                if let Some(recognizer) = get_recognizer_from_element(&element_clone) {
                    recognizer.handle_touch_end(&event);
                }
            }) as Box<dyn FnMut(TouchEvent)>)
        };

        // Create touch cancel handler
        let touchcancel = {
            let element_clone = element_ref.clone();
            Closure::wrap(Box::new(move |event: TouchEvent| {
                if let Some(recognizer) = get_recognizer_from_element(&element_clone) {
                    recognizer.handle_touch_cancel(&event);
                }
            }) as Box<dyn FnMut(TouchEvent)>)
        };

        // Add event listeners
        element
            .add_event_listener_with_callback("touchstart", touchstart.as_ref().unchecked_ref())?;
        element
            .add_event_listener_with_callback("touchmove", touchmove.as_ref().unchecked_ref())?;
        element.add_event_listener_with_callback("touchend", touchend.as_ref().unchecked_ref())?;
        element.add_event_listener_with_callback(
            "touchcancel",
            touchcancel.as_ref().unchecked_ref(),
        )?;

        // Store the closures to prevent them from being dropped
        self.event_handlers.insert("touchstart".to_string(), touchstart);
        self.event_handlers.insert("touchmove".to_string(), touchmove);
        self.event_handlers.insert("touchend".to_string(), touchend);
        self.event_handlers.insert("touchcancel".to_string(), touchcancel);

        self.is_attached = true;
        Ok(())
    }

    pub fn detach_from_element(&mut self) -> Result<(), JsValue> {
        if let Some(element) = &self.target_element {
            // Remove event listeners
            for (event_type, handler) in &self.event_handlers {
                element.remove_event_listener_with_callback(
                    event_type,
                    handler.as_ref().unchecked_ref(),
                )?;
            }

            // Remove from registry
            GESTURE_REGISTRY.with(|registry| {
                let reg = registry.borrow();
                reg.delete(&element.clone().into());
            });
        }

        self.event_handlers.clear();
        self.target_element = None;
        self.is_attached = false;
        Ok(())
    }

    pub fn set_gesture_callback(&mut self, gesture_type: GestureType, callback: &Function) {
        let gesture_name = match gesture_type {
            GestureType::Tap => "tap",
            GestureType::DoubleTap => "doubletap",
            GestureType::LongPress => "longpress",
            GestureType::Swipe => "swipe",
            GestureType::Pinch => "pinch",
            GestureType::Rotate => "rotate",
            GestureType::Pan => "pan",
            GestureType::TwoFingerTap => "twofingertap",
            GestureType::ThreeFingerTap => "threefingertap",
            GestureType::Unknown => "unknown",
        };

        js_sys::Reflect::set(&self.callbacks, &JsValue::from_str(gesture_name), callback)
            .expect("Failed to set gesture callback in callbacks object");
    }

    fn handle_touch_start(&mut self, event: &TouchEvent) {
        if self.config.enable_prevent_default {
            event.prevent_default();
        }

        let current_time = get_current_time();
        let touches = event.touches();

        self.active_touches.clear();
        self.gesture_start_time = current_time;

        // Record all active touches
        for i in 0..touches.length() {
            if let Some(touch) = touches.item(i) {
                let touch_point = TouchPoint::new(
                    touch.client_x() as f64,
                    touch.client_y() as f64,
                    touch.identifier(),
                    current_time,
                );
                self.active_touches.push(touch_point.clone());

                // Add to history for velocity calculations
                self.touch_history.push(touch_point);
            }
        }

        // Handle multi-touch gestures
        if self.active_touches.len() >= 2 {
            self.initial_distance = self.calculate_distance_between_touches(0, 1);
            self.initial_angle = self.calculate_angle_between_touches(0, 1);
        }

        // Set up long press timer for single touch
        if self.active_touches.len() == 1 {
            self.setup_long_press_timer();
        }
    }

    fn handle_touch_move(&mut self, event: &TouchEvent) {
        if self.config.enable_prevent_default {
            event.prevent_default();
        }

        let current_time = get_current_time();
        let touches = event.touches();

        // Cancel long press if touch moves significantly
        if let Some(_timer_id) = self.long_press_timer {
            if self.active_touches.len() == 1 && touches.length() == 1 {
                if let Some(touch) = touches.item(0) {
                    let start_touch = &self.active_touches[0];
                    let distance = self.calculate_distance(
                        start_touch.x,
                        start_touch.y,
                        touch.client_x() as f64,
                        touch.client_y() as f64,
                    );

                    if distance > self.config.pan_threshold {
                        self.cancel_long_press_timer();
                        self.detect_pan_gesture(current_time);
                    }
                }
            }
        }

        // Detect pinch and rotation gestures
        if self.active_touches.len() >= 2 && touches.length() >= 2 {
            self.detect_pinch_and_rotation_gestures(current_time, &touches);
        }
    }

    fn handle_touch_end(&mut self, event: &TouchEvent) {
        if self.config.enable_prevent_default {
            event.prevent_default();
        }

        let current_time = get_current_time();
        let duration = current_time - self.gesture_start_time;

        self.cancel_long_press_timer();

        // Handle different gestures based on touch count and duration
        match self.active_touches.len() {
            1 => self.handle_single_touch_end(current_time, duration),
            2 => self.handle_two_touch_end(current_time, duration),
            3 => self.handle_three_touch_end(current_time, duration),
            _ => {},
        }

        self.active_touches.clear();
    }

    fn handle_touch_cancel(&mut self, _event: &TouchEvent) {
        self.cancel_long_press_timer();
        self.active_touches.clear();
    }

    fn handle_single_touch_end(&mut self, current_time: f64, duration: f64) {
        if duration < self.config.tap_timeout {
            // Check for double tap
            if current_time - self.last_tap_time < self.config.double_tap_interval {
                self.tap_count += 1;
                if self.tap_count == 2 {
                    self.emit_gesture_event(GestureType::DoubleTap, current_time, duration);
                    self.tap_count = 0;
                    return;
                }
            } else {
                self.tap_count = 1;
            }

            self.last_tap_time = current_time;

            // Emit single tap after a delay to check for double tap
            let window = window().expect("window should be available in browser context");
            let timeout_callback = Closure::wrap(Box::new(move || {
                // If no second tap occurred, emit single tap
                // This would need to be implemented with proper closure handling
            }) as Box<dyn FnMut()>);

            window
                .set_timeout_with_callback_and_timeout_and_arguments_0(
                    timeout_callback.as_ref().unchecked_ref(),
                    self.config.double_tap_interval as i32,
                )
                .expect("set_timeout should succeed with valid callback");
            timeout_callback.forget();
        } else {
            // Check for swipe gesture
            self.detect_swipe_gesture(current_time, duration);
        }
    }

    fn handle_two_touch_end(&mut self, current_time: f64, duration: f64) {
        if duration < self.config.tap_timeout {
            self.emit_gesture_event(GestureType::TwoFingerTap, current_time, duration);
        }
    }

    fn handle_three_touch_end(&mut self, current_time: f64, duration: f64) {
        if duration < self.config.tap_timeout {
            self.emit_gesture_event(GestureType::ThreeFingerTap, current_time, duration);
        }
    }

    fn detect_swipe_gesture(&mut self, current_time: f64, duration: f64) {
        if self.active_touches.is_empty() {
            return;
        }

        let start_touch = &self.active_touches[0];
        let end_x = start_touch.x; // This should be updated with final position
        let end_y = start_touch.y; // This should be updated with final position

        let distance = self.calculate_distance(start_touch.x, start_touch.y, end_x, end_y);

        if distance >= self.config.swipe_min_distance && duration <= self.config.swipe_max_time {
            let direction =
                self.calculate_swipe_direction(start_touch.x, start_touch.y, end_x, end_y);
            let velocity = distance / duration;

            if velocity >= self.config.velocity_threshold {
                self.emit_swipe_event(direction, distance, velocity, current_time, duration);
            }
        }
    }

    fn detect_pan_gesture(&mut self, current_time: f64) {
        let duration = current_time - self.gesture_start_time;
        self.emit_gesture_event(GestureType::Pan, current_time, duration);
    }

    fn detect_pinch_and_rotation_gestures(&mut self, current_time: f64, touches: &TouchList) {
        if touches.length() < 2 {
            return;
        }

        let touch1 = touches.item(0).expect("touch list has at least 2 items after length check");
        let touch2 = touches.item(1).expect("touch list has at least 2 items after length check");

        let current_distance = self.calculate_distance(
            touch1.client_x() as f64,
            touch1.client_y() as f64,
            touch2.client_x() as f64,
            touch2.client_y() as f64,
        );

        let current_angle = self.calculate_angle(
            touch1.client_x() as f64,
            touch1.client_y() as f64,
            touch2.client_x() as f64,
            touch2.client_y() as f64,
        );

        // Detect pinch gesture
        let distance_change = (current_distance - self.initial_distance).abs();
        if distance_change > self.config.pinch_threshold {
            let scale = current_distance / self.initial_distance;
            let duration = current_time - self.gesture_start_time;
            self.emit_pinch_event(scale, duration, current_time);
        }

        // Detect rotation gesture
        let angle_change = (current_angle - self.initial_angle).abs();
        if angle_change > self.config.rotation_threshold {
            let rotation = current_angle - self.initial_angle;
            let duration = current_time - self.gesture_start_time;
            self.emit_rotation_event(rotation, duration, current_time);
        }
    }

    fn setup_long_press_timer(&mut self) {
        let window = window().unwrap();
        let timeout_callback = Closure::wrap(Box::new(move || {
            // Emit long press event
            // This would need proper implementation with closure handling
        }) as Box<dyn FnMut()>);

        let timer_id = window
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                timeout_callback.as_ref().unchecked_ref(),
                self.config.long_press_duration as i32,
            )
            .unwrap();

        self.long_press_timer = Some(timer_id);
        timeout_callback.forget();
    }

    fn cancel_long_press_timer(&mut self) {
        if let Some(timer_id) = self.long_press_timer.take() {
            let window = window().expect("window should be available in browser context");
            window.clear_timeout_with_handle(timer_id);
        }
    }

    fn calculate_distance(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        let dx = x2 - x1;
        let dy = y2 - y1;
        (dx * dx + dy * dy).sqrt()
    }

    fn calculate_distance_between_touches(&self, index1: usize, index2: usize) -> f64 {
        if index1 < self.active_touches.len() && index2 < self.active_touches.len() {
            let touch1 = &self.active_touches[index1];
            let touch2 = &self.active_touches[index2];
            self.calculate_distance(touch1.x, touch1.y, touch2.x, touch2.y)
        } else {
            0.0
        }
    }

    fn calculate_angle(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        let dx = x2 - x1;
        let dy = y2 - y1;
        dy.atan2(dx) * 180.0 / std::f64::consts::PI
    }

    fn calculate_angle_between_touches(&self, index1: usize, index2: usize) -> f64 {
        if index1 < self.active_touches.len() && index2 < self.active_touches.len() {
            let touch1 = &self.active_touches[index1];
            let touch2 = &self.active_touches[index2];
            self.calculate_angle(touch1.x, touch1.y, touch2.x, touch2.y)
        } else {
            0.0
        }
    }

    fn calculate_swipe_direction(
        &self,
        start_x: f64,
        start_y: f64,
        end_x: f64,
        end_y: f64,
    ) -> SwipeDirection {
        let dx = end_x - start_x;
        let dy = end_y - start_y;

        if dx.abs() > dy.abs() {
            if dx > 0.0 {
                SwipeDirection::Right
            } else {
                SwipeDirection::Left
            }
        } else if dy > 0.0 {
            SwipeDirection::Down
        } else {
            SwipeDirection::Up
        }
    }

    fn emit_gesture_event(&self, gesture_type: GestureType, _timestamp: f64, duration: f64) {
        if self.active_touches.is_empty() {
            return;
        }

        let start_point = self.active_touches[0].clone();
        let gesture_event = GestureEvent {
            gesture_type,
            start_point,
            end_point: None,
            swipe_direction: None,
            distance: 0.0,
            duration,
            scale: 1.0,
            rotation: 0.0,
            velocity: 0.0,
            touch_count: self.active_touches.len() as u32,
        };

        self.call_gesture_callback(gesture_type, &gesture_event);
    }

    fn emit_swipe_event(
        &self,
        direction: SwipeDirection,
        distance: f64,
        velocity: f64,
        _timestamp: f64,
        duration: f64,
    ) {
        if self.active_touches.is_empty() {
            return;
        }

        let start_point = self.active_touches[0].clone();
        let gesture_event = GestureEvent {
            gesture_type: GestureType::Swipe,
            start_point,
            end_point: None,
            swipe_direction: Some(direction),
            distance,
            duration,
            scale: 1.0,
            rotation: 0.0,
            velocity,
            touch_count: 1,
        };

        self.call_gesture_callback(GestureType::Swipe, &gesture_event);
    }

    fn emit_pinch_event(&self, scale: f64, duration: f64, _timestamp: f64) {
        if self.active_touches.len() < 2 {
            return;
        }

        let start_point = self.active_touches[0].clone();
        let gesture_event = GestureEvent {
            gesture_type: GestureType::Pinch,
            start_point,
            end_point: None,
            swipe_direction: None,
            distance: 0.0,
            duration,
            scale,
            rotation: 0.0,
            velocity: 0.0,
            touch_count: self.active_touches.len() as u32,
        };

        self.call_gesture_callback(GestureType::Pinch, &gesture_event);
    }

    fn emit_rotation_event(&self, rotation: f64, duration: f64, _timestamp: f64) {
        if self.active_touches.len() < 2 {
            return;
        }

        let start_point = self.active_touches[0].clone();
        let gesture_event = GestureEvent {
            gesture_type: GestureType::Rotate,
            start_point,
            end_point: None,
            swipe_direction: None,
            distance: 0.0,
            duration,
            scale: 1.0,
            rotation,
            velocity: 0.0,
            touch_count: self.active_touches.len() as u32,
        };

        self.call_gesture_callback(GestureType::Rotate, &gesture_event);
    }

    fn call_gesture_callback(&self, gesture_type: GestureType, event: &GestureEvent) {
        let gesture_name = match gesture_type {
            GestureType::Tap => "tap",
            GestureType::DoubleTap => "doubletap",
            GestureType::LongPress => "longpress",
            GestureType::Swipe => "swipe",
            GestureType::Pinch => "pinch",
            GestureType::Rotate => "rotate",
            GestureType::Pan => "pan",
            GestureType::TwoFingerTap => "twofingertap",
            GestureType::ThreeFingerTap => "threefingertap",
            GestureType::Unknown => "unknown",
        };

        if let Ok(callback) =
            js_sys::Reflect::get(&self.callbacks, &JsValue::from_str(gesture_name))
        {
            if let Ok(function) = callback.dyn_into::<Function>() {
                // Convert event to JsValue for callback
                let event_js = serde_wasm_bindgen::to_value(event).unwrap_or(JsValue::UNDEFINED);
                let _ = function.call1(&JsValue::NULL, &event_js);
            }
        }
    }

    pub fn get_active_touch_count(&self) -> u32 {
        self.active_touches.len() as u32
    }

    pub fn is_gesture_active(&self) -> bool {
        !self.active_touches.is_empty()
    }

    // Enhanced methods for better mobile interaction
    pub fn get_touch_velocity(&self) -> f64 {
        if self.touch_history.len() < 2 {
            return 0.0;
        }

        let recent = &self.touch_history[self.touch_history.len() - 1];
        let previous = &self.touch_history[self.touch_history.len() - 2];

        let distance = self.calculate_distance(previous.x, previous.y, recent.x, recent.y);
        let time_diff = recent.timestamp - previous.timestamp;

        if time_diff > 0.0 {
            distance / time_diff
        } else {
            0.0
        }
    }

    pub fn get_touch_acceleration(&self) -> f64 {
        if self.touch_history.len() < 3 {
            return 0.0;
        }

        let len = self.touch_history.len();
        let recent = &self.touch_history[len - 1];
        let middle = &self.touch_history[len - 2];
        let previous = &self.touch_history[len - 3];

        let v1 = self.calculate_distance(previous.x, previous.y, middle.x, middle.y)
            / (middle.timestamp - previous.timestamp);
        let v2 = self.calculate_distance(middle.x, middle.y, recent.x, recent.y)
            / (recent.timestamp - middle.timestamp);

        let time_diff = recent.timestamp - previous.timestamp;
        if time_diff > 0.0 {
            (v2 - v1) / time_diff
        } else {
            0.0
        }
    }

    pub fn reset_gesture_state(&mut self) {
        self.active_touches.clear();
        self.touch_history.clear();
        self.cancel_long_press_timer();
        self.tap_count = 0;
        self.gesture_start_time = 0.0;
        self.initial_distance = 0.0;
        self.initial_angle = 0.0;
    }

    pub fn get_gesture_duration(&self) -> f64 {
        if self.gesture_start_time > 0.0 {
            get_current_time() - self.gesture_start_time
        } else {
            0.0
        }
    }

    // Method to get current touch pressure (if supported)
    pub fn get_average_touch_pressure(&self) -> f64 {
        if self.active_touches.is_empty() {
            return 0.0;
        }

        // Note: TouchPoint would need to be extended with pressure data
        // For now, return 1.0 as default pressure
        1.0
    }

    // Method to check if gesture is likely accidental (e.g., palm rejection)
    pub fn is_likely_accidental_touch(&self) -> bool {
        if self.active_touches.len() > 5 {
            return true; // Too many touches, likely palm
        }

        if let Some(first_touch) = self.active_touches.first() {
            let current_time = get_current_time();
            let duration = current_time - first_touch.timestamp;

            // Very quick touches might be accidental
            if duration < 50.0 && self.get_touch_velocity() > 5.0 {
                return true;
            }
        }

        false
    }
}

// Utility functions
fn get_current_time() -> f64 {
    if let Some(window) = window() {
        if let Some(performance) = window.performance() {
            performance.now()
        } else {
            js_sys::Date::now()
        }
    } else {
        0.0
    }
}

// Get recognizer from element using the registry
fn get_recognizer_from_element(_element: &Element) -> Option<&'static mut TouchGestureRecognizer> {
    GESTURE_REGISTRY.with(|registry| {
        let reg = registry.borrow();
        let recognizer_ptr = reg.get(&_element.clone().into());
        if let Some(ptr_val) = recognizer_ptr.as_f64() {
            // SAFETY: This is unsafe but necessary for the pattern.
            // In production, you'd want a safer approach using Rc<RefCell<>> or similar.
            unsafe {
                let ptr = ptr_val as u32 as *mut TouchGestureRecognizer;
                ptr.as_mut()
            }
        } else {
            None
        }
    })
}

// Enhanced touch history tracking
fn add_touch_to_history(recognizer: &mut TouchGestureRecognizer, touch: TouchPoint) {
    recognizer.touch_history.push(touch);

    // Keep only recent history (last 50 touches)
    if recognizer.touch_history.len() > 50 {
        recognizer.touch_history.remove(0);
    }
}

// Factory function for easy creation
#[wasm_bindgen]
pub fn create_touch_gesture_recognizer() -> TouchGestureRecognizer {
    let config = GestureConfig::default();
    TouchGestureRecognizer::new(config)
}

#[wasm_bindgen]
pub fn create_touch_gesture_recognizer_with_config(
    config: GestureConfig,
) -> TouchGestureRecognizer {
    TouchGestureRecognizer::new(config)
}

// Quick setup function for common use cases
#[wasm_bindgen]
pub fn setup_basic_gestures_on_element(
    element: &Element,
) -> Result<TouchGestureRecognizer, JsValue> {
    let config = GestureConfig::default();
    let mut recognizer = TouchGestureRecognizer::new(config);
    recognizer.attach_to_element(element)?;
    Ok(recognizer)
}

// Gesture utilities
#[wasm_bindgen]
pub fn is_touch_device() -> bool {
    if let Some(window) = window() {
        let navigator = window.navigator();
        let max_touch_points = js_sys::Reflect::get(&navigator, &"maxTouchPoints".into())
            .unwrap_or_default()
            .as_f64()
            .unwrap_or(0.0);

        max_touch_points > 0.0
            || js_sys::Reflect::has(&window, &"ontouchstart".into()).unwrap_or(false)
    } else {
        false
    }
}

#[wasm_bindgen]
pub fn get_max_touch_points() -> u32 {
    if let Some(window) = window() {
        let navigator = window.navigator();
        js_sys::Reflect::get(&navigator, &"maxTouchPoints".into())
            .unwrap_or_default()
            .as_f64()
            .unwrap_or(1.0) as u32
    } else {
        1
    }
}

#[wasm_bindgen]
pub fn enable_touch_action_none(element: &Element) {
    if let Ok(html_element) = element.clone().dyn_into::<HtmlElement>() {
        let style = html_element.style();
        let _ = style.set_property("touch-action", "none");
    }
}

#[wasm_bindgen]
pub fn disable_context_menu(element: &Element) {
    let contextmenu_handler = Closure::wrap(Box::new(move |event: web_sys::Event| {
        event.prevent_default();
    }) as Box<dyn FnMut(web_sys::Event)>);

    let _ = element.add_event_listener_with_callback(
        "contextmenu",
        contextmenu_handler.as_ref().unchecked_ref(),
    );

    contextmenu_handler.forget();
}
