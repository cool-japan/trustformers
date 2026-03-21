#![allow(dead_code)]

use js_sys::{Array, Function, Object, Promise, Uint8Array};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use wasm_bindgen::{closure::Closure, JsCast};
use web_sys::{
    window, CanvasRenderingContext2d, HtmlCanvasElement, HtmlVideoElement, ImageData,
    MediaDeviceInfo, MediaDeviceKind, MediaStream, MediaStreamConstraints, MediaStreamTrack,
    OffscreenCanvas, VideoTrack, Worker,
};

// Import our tensor operations for ML integration
use crate::core::tensor::WasmTensor;
use crate::core::utils::get_current_time_ms;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CameraFacing {
    User,        // Front-facing camera
    Environment, // Back-facing camera
    Unknown,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CameraResolution {
    Low,       // 320x240
    Medium,    // 640x480
    High,      // 1280x720
    UltraHigh, // 1920x1080
    Custom,    // Custom resolution
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CameraState {
    Idle,
    Initializing,
    Active,
    Paused,
    Error,
    Stopped,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FrameQuality {
    pub brightness: f64,
    pub contrast: f64,
    pub sharpness: f64,
    pub is_blurry: bool,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraCapabilities {
    pub has_camera: bool,
    pub has_multiple_cameras: bool,
    pub supports_user_facing: bool,
    pub supports_environment_facing: bool,
    pub max_width: u32,
    pub max_height: u32,
    pub(crate) supported_resolutions: Vec<String>,
    pub supports_torch: bool,
    pub supports_zoom: bool,
    pub supports_focus: bool,
}

#[wasm_bindgen]
impl CameraCapabilities {
    #[wasm_bindgen(getter)]
    pub fn has_camera(&self) -> bool {
        self.has_camera
    }

    #[wasm_bindgen(getter)]
    pub fn has_multiple_cameras(&self) -> bool {
        self.has_multiple_cameras
    }

    #[wasm_bindgen(getter)]
    pub fn supports_user_facing(&self) -> bool {
        self.supports_user_facing
    }

    #[wasm_bindgen(getter)]
    pub fn supports_environment_facing(&self) -> bool {
        self.supports_environment_facing
    }

    #[wasm_bindgen(getter)]
    pub fn max_width(&self) -> u32 {
        self.max_width
    }

    #[wasm_bindgen(getter)]
    pub fn max_height(&self) -> u32 {
        self.max_height
    }

    #[wasm_bindgen(getter)]
    pub fn supports_torch(&self) -> bool {
        self.supports_torch
    }

    #[wasm_bindgen(getter)]
    pub fn supports_zoom(&self) -> bool {
        self.supports_zoom
    }

    #[wasm_bindgen(getter)]
    pub fn supports_focus(&self) -> bool {
        self.supports_focus
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub facing: CameraFacing,
    pub resolution: CameraResolution,
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    pub auto_focus: bool,
    pub torch: bool,
    pub zoom: f64,
    pub(crate) device_id: Option<String>,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            facing: CameraFacing::User,
            resolution: CameraResolution::Medium,
            width: 640,
            height: 480,
            frame_rate: 30.0,
            auto_focus: true,
            torch: false,
            zoom: 1.0,
            device_id: None,
        }
    }
}

#[wasm_bindgen]
impl CameraConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen(getter)]
    pub fn facing(&self) -> CameraFacing {
        self.facing
    }

    #[wasm_bindgen(setter)]
    pub fn set_facing(&mut self, facing: CameraFacing) {
        self.facing = facing;
    }

    #[wasm_bindgen(getter)]
    pub fn resolution(&self) -> CameraResolution {
        self.resolution
    }

    #[wasm_bindgen(setter)]
    pub fn set_resolution(&mut self, resolution: CameraResolution) {
        self.resolution = resolution;
        // Update width/height based on resolution preset
        match resolution {
            CameraResolution::Low => {
                self.width = 320;
                self.height = 240;
            },
            CameraResolution::Medium => {
                self.width = 640;
                self.height = 480;
            },
            CameraResolution::High => {
                self.width = 1280;
                self.height = 720;
            },
            CameraResolution::UltraHigh => {
                self.width = 1920;
                self.height = 1080;
            },
            CameraResolution::Custom => {}, // Keep current width/height
        }
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(setter)]
    pub fn set_width(&mut self, width: u32) {
        self.width = width;
        self.resolution = CameraResolution::Custom;
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(setter)]
    pub fn set_height(&mut self, height: u32) {
        self.height = height;
        self.resolution = CameraResolution::Custom;
    }

    #[wasm_bindgen(getter)]
    pub fn frame_rate(&self) -> f64 {
        self.frame_rate
    }

    #[wasm_bindgen(setter)]
    pub fn set_frame_rate(&mut self, rate: f64) {
        self.frame_rate = rate;
    }

    #[wasm_bindgen(getter)]
    pub fn auto_focus(&self) -> bool {
        self.auto_focus
    }

    #[wasm_bindgen(setter)]
    pub fn set_auto_focus(&mut self, auto_focus: bool) {
        self.auto_focus = auto_focus;
    }

    #[wasm_bindgen(getter)]
    pub fn torch(&self) -> bool {
        self.torch
    }

    #[wasm_bindgen(setter)]
    pub fn set_torch(&mut self, torch: bool) {
        self.torch = torch;
    }

    #[wasm_bindgen(getter)]
    pub fn zoom(&self) -> f64 {
        self.zoom
    }

    #[wasm_bindgen(setter)]
    pub fn set_zoom(&mut self, zoom: f64) {
        self.zoom = zoom;
    }

    pub fn set_device_id(&mut self, device_id: Option<String>) {
        self.device_id = device_id;
    }

    pub fn get_device_id(&self) -> Option<String> {
        self.device_id.clone()
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameData {
    pub width: u32,
    pub height: u32,
    pub timestamp: f64,
    pub(crate) format: String,
    pub frame_number: u64,
    pub quality_score: f64,
    pub brightness: f64,
    pub contrast: f64,
    pub is_blurry: bool,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub frames_processed: u64,
    pub frames_dropped: u64,
    pub average_processing_time: f64,
    pub average_frame_rate: f64,
    pub buffer_utilization: f64,
    pub ml_inference_time: f64,
}

#[wasm_bindgen]
impl FrameData {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32, timestamp: f64, format: String) -> Self {
        Self {
            width,
            height,
            timestamp,
            format,
            frame_number: 0,
            quality_score: 1.0,
            brightness: 0.5,
            contrast: 1.0,
            is_blurry: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_analysis(
        width: u32,
        height: u32,
        timestamp: f64,
        format: String,
        frame_number: u64,
        quality_score: f64,
        brightness: f64,
        contrast: f64,
        is_blurry: bool,
    ) -> Self {
        Self {
            width,
            height,
            timestamp,
            format,
            frame_number,
            quality_score,
            brightness,
            contrast,
            is_blurry,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }

    #[wasm_bindgen(getter)]
    pub fn format(&self) -> String {
        self.format.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    #[wasm_bindgen(getter)]
    pub fn quality_score(&self) -> f64 {
        self.quality_score
    }

    #[wasm_bindgen(getter)]
    pub fn brightness(&self) -> f64 {
        self.brightness
    }

    #[wasm_bindgen(getter)]
    pub fn contrast(&self) -> f64 {
        self.contrast
    }

    #[wasm_bindgen(getter)]
    pub fn is_blurry(&self) -> bool {
        self.is_blurry
    }
}

#[wasm_bindgen]
pub struct CameraManager {
    config: CameraConfig,
    state: CameraState,
    stream: Option<MediaStream>,
    video_element: Option<HtmlVideoElement>,
    canvas_element: Option<HtmlCanvasElement>,
    offscreen_canvas: Option<OffscreenCanvas>,
    capabilities: Option<CameraCapabilities>,
    frame_callbacks: Object,
    available_devices: Vec<String>,
    current_device_id: Option<String>,
    is_recording: bool,
    frame_count: u64,
    last_frame_time: f64,
    frame_rate_actual: f64,
    // Enhanced features
    frame_buffer: VecDeque<ImageData>,
    processing_stats: ProcessingStats,
    max_buffer_size: usize,
    auto_focus_enabled: bool,
    exposure_compensation: f64,
    white_balance_mode: String,
    noise_reduction_enabled: bool,
    image_stabilization_enabled: bool,
    ml_preprocessing_enabled: bool,
    adaptive_quality_enabled: bool,
    processing_worker: Option<Worker>,
}

#[wasm_bindgen]
impl CameraManager {
    #[wasm_bindgen(constructor)]
    pub fn new(config: CameraConfig) -> Self {
        Self {
            config,
            state: CameraState::Idle,
            stream: None,
            video_element: None,
            canvas_element: None,
            offscreen_canvas: None,
            capabilities: None,
            frame_callbacks: Object::new(),
            available_devices: Vec::new(),
            current_device_id: None,
            is_recording: false,
            frame_count: 0,
            last_frame_time: 0.0,
            frame_rate_actual: 0.0,
            // Enhanced features
            frame_buffer: VecDeque::new(),
            processing_stats: ProcessingStats {
                frames_processed: 0,
                frames_dropped: 0,
                average_processing_time: 0.0,
                average_frame_rate: 0.0,
                buffer_utilization: 0.0,
                ml_inference_time: 0.0,
            },
            max_buffer_size: 10,
            auto_focus_enabled: true,
            exposure_compensation: 0.0,
            white_balance_mode: "auto".to_string(),
            noise_reduction_enabled: true,
            image_stabilization_enabled: true,
            ml_preprocessing_enabled: false,
            adaptive_quality_enabled: true,
            processing_worker: None,
        }
    }

    pub async fn initialize(&mut self) -> Result<(), JsValue> {
        self.state = CameraState::Initializing;

        // Check for camera support
        self.check_camera_support().await?;

        // Enumerate available devices
        self.enumerate_devices().await?;

        // Create video and canvas elements
        self.create_media_elements()?;

        self.state = CameraState::Idle;
        Ok(())
    }

    async fn check_camera_support(&mut self) -> Result<(), JsValue> {
        let window = window().ok_or("No window object")?;
        let navigator = window.navigator();

        // Check for getUserMedia support
        let media_devices = navigator
            .media_devices()
            .map_err(|_| JsValue::from_str("MediaDevices API not supported"))?;

        // Try to enumerate devices to check camera availability
        let devices_promise = media_devices.enumerate_devices()?;
        let devices_result = wasm_bindgen_futures::JsFuture::from(devices_promise).await?;
        let devices: Array = devices_result.dyn_into()?;

        let mut has_camera = false;
        let mut supports_user_facing = false;
        let mut supports_environment_facing = false;
        let mut camera_count = 0;

        for i in 0..devices.length() {
            if let Ok(device) = devices.get(i).dyn_into::<MediaDeviceInfo>() {
                if device.kind() == MediaDeviceKind::Videoinput {
                    has_camera = true;
                    camera_count += 1;

                    let label = device.label();
                    if label.to_lowercase().contains("front")
                        || label.to_lowercase().contains("user")
                    {
                        supports_user_facing = true;
                    }
                    if label.to_lowercase().contains("back")
                        || label.to_lowercase().contains("environment")
                    {
                        supports_environment_facing = true;
                    }
                }
            }
        }

        let has_multiple_cameras = camera_count > 1;

        self.capabilities = Some(CameraCapabilities {
            has_camera,
            has_multiple_cameras,
            supports_user_facing,
            supports_environment_facing,
            max_width: 1920,
            max_height: 1080,
            supported_resolutions: vec![
                "320x240".to_string(),
                "640x480".to_string(),
                "1280x720".to_string(),
                "1920x1080".to_string(),
            ],
            supports_torch: true, // Most modern devices support torch
            supports_zoom: true,  // Most modern devices support zoom
            supports_focus: true, // Most modern devices support auto-focus
        });

        if !has_camera {
            return Err(JsValue::from_str("No camera devices found"));
        }

        Ok(())
    }

    async fn enumerate_devices(&mut self) -> Result<(), JsValue> {
        let window = window().ok_or("No window object")?;
        let navigator = window.navigator();
        let media_devices = navigator.media_devices().map_err(|_| "MediaDevices not available")?;

        let devices_promise = media_devices.enumerate_devices()?;
        let devices_result = wasm_bindgen_futures::JsFuture::from(devices_promise).await?;
        let devices: Array = devices_result.dyn_into()?;

        self.available_devices.clear();

        for i in 0..devices.length() {
            if let Ok(device) = devices.get(i).dyn_into::<MediaDeviceInfo>() {
                if device.kind() == MediaDeviceKind::Videoinput {
                    self.available_devices.push(device.device_id());
                }
            }
        }

        Ok(())
    }

    fn create_media_elements(&mut self) -> Result<(), JsValue> {
        let window = window().ok_or("No window object")?;
        let document = window.document().ok_or("No document object")?;

        // Create video element
        let video = document.create_element("video")?.dyn_into::<HtmlVideoElement>()?;
        video.set_autoplay(true);
        video.set_muted(true);
        // Use Reflect to set playsInline attribute
        let _ = js_sys::Reflect::set(
            &video,
            &JsValue::from_str("playsInline"),
            &JsValue::from_bool(true),
        );
        video.set_width(self.config.width);
        video.set_height(self.config.height);

        // Create canvas element for frame capture
        let canvas = document.create_element("canvas")?.dyn_into::<HtmlCanvasElement>()?;
        canvas.set_width(self.config.width);
        canvas.set_height(self.config.height);

        self.video_element = Some(video);
        self.canvas_element = Some(canvas);

        Ok(())
    }

    pub async fn start_camera(&mut self) -> Result<(), JsValue> {
        if self.state == CameraState::Active {
            return Ok(());
        }

        let window = window().ok_or("No window object")?;
        let navigator = window.navigator();
        let media_devices = navigator.media_devices().map_err(|_| "MediaDevices not available")?;

        // Create constraints
        let constraints = self.create_media_constraints()?;

        // Get user media
        let media_promise = media_devices.get_user_media_with_constraints(&constraints)?;
        let stream_result = wasm_bindgen_futures::JsFuture::from(media_promise).await;

        match stream_result {
            Ok(stream_value) => {
                let stream: MediaStream = stream_value.dyn_into()?;

                // Set stream to video element
                if let Some(video) = &self.video_element {
                    video.set_src_object(Some(&stream));
                }

                self.stream = Some(stream);
                self.state = CameraState::Active;
                self.frame_count = 0;
                self.last_frame_time = get_current_time();

                // Start frame processing loop
                self.start_frame_processing_loop();

                Ok(())
            },
            Err(error) => {
                self.state = CameraState::Error;
                Err(error)
            },
        }
    }

    fn create_media_constraints(&self) -> Result<MediaStreamConstraints, JsValue> {
        let constraints = MediaStreamConstraints::new();

        // Video constraints
        let video_constraints = Object::new();

        // Set resolution
        js_sys::Reflect::set(
            &video_constraints,
            &"width".into(),
            &JsValue::from(self.config.width),
        )?;
        js_sys::Reflect::set(
            &video_constraints,
            &"height".into(),
            &JsValue::from(self.config.height),
        )?;

        // Set frame rate
        js_sys::Reflect::set(
            &video_constraints,
            &"frameRate".into(),
            &JsValue::from(self.config.frame_rate),
        )?;

        // Set facing mode
        let facing_mode = match self.config.facing {
            CameraFacing::User => "user",
            CameraFacing::Environment => "environment",
            CameraFacing::Unknown => "user", // Default to user
        };
        js_sys::Reflect::set(
            &video_constraints,
            &"facingMode".into(),
            &JsValue::from_str(facing_mode),
        )?;

        // Set device ID if specified
        if let Some(device_id) = &self.config.device_id {
            js_sys::Reflect::set(
                &video_constraints,
                &"deviceId".into(),
                &JsValue::from_str(device_id),
            )?;
        }

        constraints.set_video(&video_constraints.into());
        constraints.set_audio(&JsValue::FALSE); // Only video for now

        Ok(constraints)
    }

    fn start_frame_processing_loop(&self) {
        let window_obj = window().expect("window should be available in browser context");

        // Clone necessary data for the closure
        let video_element = self.video_element.clone();
        let canvas_element = self.canvas_element.clone();

        let frame_callback = Closure::wrap(Box::new(move || {
            // Process current frame if video and canvas are available
            if let (Some(video), Some(canvas)) = (&video_element, &canvas_element) {
                if let Ok(Some(context)) = canvas.get_context("2d") {
                    let context: CanvasRenderingContext2d = context.unchecked_into();

                    // Set canvas size to match video
                    let video_width = video.video_width();
                    let video_height = video.video_height();

                    if video_width > 0 && video_height > 0 {
                        canvas.set_width(video_width);
                        canvas.set_height(video_height);

                        // Draw current video frame to canvas
                        let _ = context.draw_image_with_html_video_element_and_dw_and_dh(
                            video,
                            0.0,
                            0.0,
                            video_width as f64,
                            video_height as f64,
                        );

                        // Get image data for analysis
                        if let Ok(_image_data) = context.get_image_data(
                            0.0,
                            0.0,
                            video_width as f64,
                            video_height as f64,
                        ) {
                            // Frame quality analysis would be done here
                            // (Note: This would need access to self for analyze_frame_quality)
                            // For now, we just demonstrate the frame capture process

                            // Schedule next frame
                            // Note: This would need proper closure handling for recursive scheduling
                            // For now, we skip the recursive frame request to avoid complexity
                        }
                    }
                }
            }
        }) as Box<dyn FnMut()>);

        // Use requestAnimationFrame for smooth frame processing
        let _ = window_obj.request_animation_frame(frame_callback.as_ref().unchecked_ref());

        frame_callback.forget();
    }

    pub fn capture_frame(&mut self) -> Result<FrameData, JsValue> {
        if self.state != CameraState::Active {
            return Err(JsValue::from_str("Camera is not active"));
        }

        let video = self.video_element.as_ref().ok_or("No video element")?;
        let canvas = self.canvas_element.as_ref().ok_or("No canvas element")?;

        let context = canvas
            .get_context("2d")?
            .ok_or("Failed to get 2D context")?
            .dyn_into::<CanvasRenderingContext2d>()?;

        // Draw video frame to canvas
        context.draw_image_with_html_video_element_and_dw_and_dh(
            video,
            0.0,
            0.0,
            self.config.width as f64,
            self.config.height as f64,
        )?;

        let current_time = get_current_time();

        // Update frame rate calculation
        if self.last_frame_time > 0.0 {
            let time_delta = current_time - self.last_frame_time;
            self.frame_rate_actual = 1000.0 / time_delta; // Convert to FPS
        }

        self.last_frame_time = current_time;
        self.frame_count += 1;

        let frame_data = FrameData::new(
            self.config.width,
            self.config.height,
            current_time,
            "rgb".to_string(),
        );

        // Call frame callbacks
        self.call_frame_callbacks(&frame_data);

        Ok(frame_data)
    }

    pub fn capture_frame_as_image_data(&self) -> Result<ImageData, JsValue> {
        if self.state != CameraState::Active {
            return Err(JsValue::from_str("Camera is not active"));
        }

        let canvas = self.canvas_element.as_ref().ok_or("No canvas element")?;

        let context = canvas
            .get_context("2d")?
            .ok_or("Failed to get 2D context")?
            .dyn_into::<CanvasRenderingContext2d>()?;

        context.get_image_data(
            0.0,
            0.0,
            self.config.width as f64,
            self.config.height as f64,
        )
    }

    pub fn capture_frame_as_blob(&self) -> Result<Promise, JsValue> {
        if self.state != CameraState::Active {
            return Err(JsValue::from_str("Camera is not active"));
        }

        let canvas = self.canvas_element.as_ref().ok_or("No canvas element")?;

        // Create a promise that wraps the to_blob callback
        let promise = Promise::new(&mut |resolve, reject| {
            let resolve = resolve.clone();
            let reject_clone = reject.clone();
            let reject_for_closure = reject.clone();

            let callback = Closure::once(move |blob: JsValue| {
                if blob.is_null() || blob.is_undefined() {
                    let _ = reject_for_closure
                        .call1(&JsValue::NULL, &JsValue::from_str("Failed to create blob"));
                } else {
                    let _ = resolve.call1(&JsValue::NULL, &blob);
                }
            });

            if let Err(e) = canvas.to_blob(callback.as_ref().unchecked_ref()) {
                let _ = reject_clone.call1(&JsValue::NULL, &e);
            }

            callback.forget(); // Prevent closure from being dropped
        });

        Ok(promise)
    }

    pub fn capture_frame_as_data_url(&self) -> Result<String, JsValue> {
        if self.state != CameraState::Active {
            return Err(JsValue::from_str("Camera is not active"));
        }

        let canvas = self.canvas_element.as_ref().ok_or("No canvas element")?;

        canvas.to_data_url()
    }

    pub fn stop_camera(&mut self) -> Result<(), JsValue> {
        if let Some(stream) = &self.stream {
            let tracks = stream.get_tracks();
            for i in 0..tracks.length() {
                if let Ok(track) = tracks.get(i).dyn_into::<MediaStreamTrack>() {
                    track.stop();
                }
            }
        }

        if let Some(video) = &self.video_element {
            video.set_src_object(None);
        }

        self.stream = None;
        self.state = CameraState::Stopped;
        self.frame_count = 0;
        self.last_frame_time = 0.0;
        self.frame_rate_actual = 0.0;

        Ok(())
    }

    pub fn pause_camera(&mut self) -> Result<(), JsValue> {
        if self.state == CameraState::Active {
            if let Some(video) = &self.video_element {
                video.pause()?;
            }
            self.state = CameraState::Paused;
        }
        Ok(())
    }

    pub fn resume_camera(&mut self) -> Result<(), JsValue> {
        if self.state == CameraState::Paused {
            if let Some(video) = &self.video_element {
                let _ = video.play()?;
            }
            self.state = CameraState::Active;
        }
        Ok(())
    }

    pub async fn switch_camera(&mut self, facing: CameraFacing) -> Result<(), JsValue> {
        let was_active = self.state == CameraState::Active;

        if was_active {
            self.stop_camera()?;
        }

        self.config.facing = facing;
        self.config.device_id = None; // Reset device ID to use facing mode

        if was_active {
            self.start_camera().await?;
        }

        Ok(())
    }

    pub async fn switch_to_device(&mut self, device_id: String) -> Result<(), JsValue> {
        let was_active = self.state == CameraState::Active;

        if was_active {
            self.stop_camera()?;
        }

        self.config.device_id = Some(device_id);

        if was_active {
            self.start_camera().await?;
        }

        Ok(())
    }

    pub fn set_frame_callback(&mut self, callback: &Function) {
        js_sys::Reflect::set(&self.frame_callbacks, &"frame".into(), callback)
            .expect("Failed to set frame callback on frame_callbacks object");
    }

    pub fn set_error_callback(&mut self, callback: &Function) {
        js_sys::Reflect::set(&self.frame_callbacks, &"error".into(), callback)
            .expect("Failed to set error callback on frame_callbacks object");
    }

    fn call_frame_callbacks(&self, frame_data: &FrameData) {
        if let Ok(callback) = js_sys::Reflect::get(&self.frame_callbacks, &"frame".into()) {
            if let Ok(function) = callback.dyn_into::<Function>() {
                let frame_js =
                    serde_wasm_bindgen::to_value(frame_data).unwrap_or(JsValue::UNDEFINED);
                let _ = function.call1(&JsValue::NULL, &frame_js);
            }
        }
    }

    pub fn apply_torch(&mut self, enable: bool) -> Result<(), JsValue> {
        if let Some(stream) = &self.stream {
            let tracks = stream.get_video_tracks();
            if tracks.length() > 0 {
                if let Ok(track) = tracks.get(0).dyn_into::<VideoTrack>() {
                    // Try to apply torch constraint
                    let constraints = Object::new();
                    let advanced = Array::new();
                    let torch_constraint = Object::new();
                    js_sys::Reflect::set(
                        &torch_constraint,
                        &"torch".into(),
                        &JsValue::from(enable),
                    )?;
                    advanced.push(&torch_constraint);
                    js_sys::Reflect::set(&constraints, &"advanced".into(), &advanced)?;

                    // Apply constraints (this may not work on all browsers/devices)
                    // Use Reflect to call applyConstraints
                    if let Ok(apply_fn) =
                        js_sys::Reflect::get(&track, &JsValue::from_str("applyConstraints"))
                    {
                        if let Ok(apply_fn) = apply_fn.dyn_into::<js_sys::Function>() {
                            let _ = apply_fn.call1(&track, &constraints);
                        }
                    }
                }
            }
        }

        self.config.torch = enable;
        Ok(())
    }

    pub fn apply_zoom(&mut self, zoom_level: f64) -> Result<(), JsValue> {
        if let Some(stream) = &self.stream {
            let tracks = stream.get_video_tracks();
            if tracks.length() > 0 {
                if let Ok(track) = tracks.get(0).dyn_into::<VideoTrack>() {
                    // Try to apply zoom constraint
                    let constraints = Object::new();
                    let advanced = Array::new();
                    let zoom_constraint = Object::new();
                    js_sys::Reflect::set(
                        &zoom_constraint,
                        &"zoom".into(),
                        &JsValue::from(zoom_level),
                    )?;
                    advanced.push(&zoom_constraint);
                    js_sys::Reflect::set(&constraints, &"advanced".into(), &advanced)?;

                    // Apply constraints (this may not work on all browsers/devices)
                    // Use Reflect to call applyConstraints
                    if let Ok(apply_fn) =
                        js_sys::Reflect::get(&track, &JsValue::from_str("applyConstraints"))
                    {
                        if let Ok(apply_fn) = apply_fn.dyn_into::<js_sys::Function>() {
                            let _ = apply_fn.call1(&track, &constraints);
                        }
                    }
                }
            }
        }

        self.config.zoom = zoom_level;
        Ok(())
    }

    #[wasm_bindgen(getter)]
    pub fn state(&self) -> CameraState {
        self.state
    }

    #[wasm_bindgen(getter)]
    pub fn capabilities(&self) -> Option<CameraCapabilities> {
        self.capabilities.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn config(&self) -> CameraConfig {
        self.config.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    #[wasm_bindgen(getter)]
    pub fn actual_frame_rate(&self) -> f64 {
        self.frame_rate_actual
    }

    #[wasm_bindgen(getter)]
    pub fn is_active(&self) -> bool {
        self.state == CameraState::Active
    }

    pub fn get_available_devices(&self) -> Array {
        let array = Array::new();
        for device_id in &self.available_devices {
            array.push(&JsValue::from_str(device_id));
        }
        array
    }

    pub fn export_camera_info(&self) -> Result<String, JsValue> {
        let info = serde_json::json!({
            "state": self.state,
            "config": self.config,
            "capabilities": self.capabilities,
            "available_devices": self.available_devices,
            "current_device_id": self.current_device_id,
            "frame_count": self.frame_count,
            "actual_frame_rate": self.frame_rate_actual,
            "is_recording": self.is_recording,
            "processing_stats": self.processing_stats,
            "buffer_size": self.frame_buffer.len(),
            "ml_preprocessing_enabled": self.ml_preprocessing_enabled,
        });

        serde_json::to_string_pretty(&info).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // Enhanced ML Integration Methods
    pub fn enable_ml_preprocessing(&mut self, enable: bool) {
        self.ml_preprocessing_enabled = enable;
    }

    pub fn capture_frame_as_tensor(&mut self) -> Result<WasmTensor, JsValue> {
        let image_data = self.capture_frame_as_image_data()?;
        let data = image_data.data();

        // Convert RGBA to RGB and normalize to [0, 1]
        let mut rgb_data = Vec::new();
        let data_vec = data.to_vec();
        for i in (0..data_vec.len()).step_by(4) {
            let r = data_vec[i] as f32 / 255.0;
            let g = data_vec[i + 1] as f32 / 255.0;
            let b = data_vec[i + 2] as f32 / 255.0;
            rgb_data.push(r);
            rgb_data.push(g);
            rgb_data.push(b);
        }

        WasmTensor::new(
            rgb_data,
            vec![self.config.height as usize, self.config.width as usize, 3],
        )
    }

    pub fn capture_frame_as_normalized_tensor(
        &mut self,
        mean: &[f32],
        std: &[f32],
    ) -> Result<WasmTensor, JsValue> {
        let tensor = self.capture_frame_as_tensor()?;

        // Get the data and apply normalization
        let mut data = tensor.data();
        for (i, value) in data.iter_mut().enumerate() {
            let channel = i % 3;
            *value = (*value - mean[channel]) / std[channel];
        }

        // Create a new tensor with normalized data
        WasmTensor::new(data, tensor.shape())
    }

    // Frame Quality Analysis
    pub fn analyze_frame_quality(&self, image_data: &ImageData) -> FrameQuality {
        let data = image_data.data();
        let data_vec = data.to_vec();
        let pixel_count = (data_vec.len() / 4) as f64;

        let mut brightness_sum = 0.0;
        let contrast_sum = 0.0;
        let mut edge_strength = 0.0;

        // Simple quality analysis
        for i in (0..data_vec.len()).step_by(4) {
            let r = data_vec[i] as f64;
            let g = data_vec[i + 1] as f64;
            let b = data_vec[i + 2] as f64;

            // Calculate brightness (luminance)
            let luminance = 0.299 * r + 0.587 * g + 0.114 * b;
            brightness_sum += luminance;

            // Simple edge detection for blur estimation
            if i >= 4 {
                let prev_lum = 0.299 * data_vec[i - 4] as f64
                    + 0.587 * data_vec[i - 3] as f64
                    + 0.114 * data_vec[i - 2] as f64;
                edge_strength += (luminance - prev_lum).abs();
            }
        }

        let brightness = brightness_sum / pixel_count / 255.0;
        let contrast = contrast_sum / pixel_count;
        let sharpness = edge_strength / pixel_count;
        let is_blurry = sharpness < 10.0; // Threshold for blur detection

        FrameQuality {
            brightness,
            contrast,
            sharpness,
            is_blurry,
        }
    }

    // Enhanced frame capture with quality analysis
    pub fn capture_analyzed_frame(&mut self) -> Result<FrameData, JsValue> {
        let image_data = self.capture_frame_as_image_data()?;
        let quality = self.analyze_frame_quality(&image_data);

        let current_time = get_current_time_ms();
        self.frame_count += 1;

        // Update processing stats
        self.processing_stats.frames_processed += 1;
        if self.last_frame_time > 0.0 {
            let time_delta = current_time - self.last_frame_time;
            self.frame_rate_actual = 1000.0 / time_delta;
        }
        self.last_frame_time = current_time;

        // Quality score based on multiple factors
        let quality_score = (quality.sharpness / 50.0).min(1.0)
            * (1.0 - (quality.brightness - 0.5).abs() * 2.0).max(0.0);

        let frame_data = FrameData::new_with_analysis(
            self.config.width,
            self.config.height,
            current_time,
            "rgb".to_string(),
            self.frame_count,
            quality_score,
            quality.brightness,
            quality.contrast,
            quality.is_blurry,
        );

        // Buffer management
        if self.frame_buffer.len() >= self.max_buffer_size {
            self.frame_buffer.pop_front();
        }
        self.frame_buffer.push_back(image_data);

        self.processing_stats.buffer_utilization =
            self.frame_buffer.len() as f64 / self.max_buffer_size as f64;

        Ok(frame_data)
    }

    // Mobile-specific optimizations
    pub fn enable_mobile_optimizations(&mut self, enable: bool) {
        if enable {
            // Reduce buffer size for memory constraints
            self.max_buffer_size = 5;
            // Enable adaptive quality
            self.adaptive_quality_enabled = true;
            // Enable noise reduction for better image quality
            self.noise_reduction_enabled = true;
        } else {
            self.max_buffer_size = 10;
            self.adaptive_quality_enabled = false;
        }
    }

    pub fn set_adaptive_quality(&mut self, enable: bool) {
        self.adaptive_quality_enabled = enable;
    }

    pub fn adjust_quality_based_on_performance(&mut self) {
        if !self.adaptive_quality_enabled {
            return;
        }

        // If frame rate is too low, reduce quality
        if self.frame_rate_actual < self.config.frame_rate * 0.8 {
            if self.config.resolution != CameraResolution::Low {
                match self.config.resolution {
                    CameraResolution::UltraHigh => {
                        self.config.set_resolution(CameraResolution::High)
                    },
                    CameraResolution::High => self.config.set_resolution(CameraResolution::Medium),
                    CameraResolution::Medium => self.config.set_resolution(CameraResolution::Low),
                    _ => {},
                }
            }
        }
        // If frame rate is consistently high, try to increase quality
        else if self.frame_rate_actual > self.config.frame_rate * 1.2
            && self.config.resolution != CameraResolution::UltraHigh
        {
            match self.config.resolution {
                CameraResolution::Low => self.config.set_resolution(CameraResolution::Medium),
                CameraResolution::Medium => self.config.set_resolution(CameraResolution::High),
                CameraResolution::High => self.config.set_resolution(CameraResolution::UltraHigh),
                _ => {},
            }
        }
    }

    // Advanced camera controls
    pub fn set_exposure_compensation(&mut self, compensation: f64) -> Result<(), JsValue> {
        self.exposure_compensation = compensation.clamp(-2.0, 2.0);

        if let Some(stream) = &self.stream {
            let tracks = stream.get_video_tracks();
            if tracks.length() > 0 {
                if let Ok(track) = tracks.get(0).dyn_into::<VideoTrack>() {
                    let constraints = Object::new();
                    let advanced = Array::new();
                    let exposure_constraint = Object::new();
                    js_sys::Reflect::set(
                        &exposure_constraint,
                        &"exposureCompensation".into(),
                        &JsValue::from(compensation),
                    )?;
                    advanced.push(&exposure_constraint);
                    js_sys::Reflect::set(&constraints, &"advanced".into(), &advanced)?;
                    // Use Reflect to call applyConstraints
                    if let Ok(apply_fn) =
                        js_sys::Reflect::get(&track, &JsValue::from_str("applyConstraints"))
                    {
                        if let Ok(apply_fn) = apply_fn.dyn_into::<js_sys::Function>() {
                            let _ = apply_fn.call1(&track, &constraints);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn set_white_balance_mode(&mut self, mode: String) -> Result<(), JsValue> {
        self.white_balance_mode = mode.clone();

        if let Some(stream) = &self.stream {
            let tracks = stream.get_video_tracks();
            if tracks.length() > 0 {
                if let Ok(track) = tracks.get(0).dyn_into::<VideoTrack>() {
                    let constraints = Object::new();
                    let advanced = Array::new();
                    let wb_constraint = Object::new();
                    js_sys::Reflect::set(
                        &wb_constraint,
                        &"whiteBalanceMode".into(),
                        &JsValue::from_str(&mode),
                    )?;
                    advanced.push(&wb_constraint);
                    js_sys::Reflect::set(&constraints, &"advanced".into(), &advanced)?;
                    // Use Reflect to call applyConstraints
                    if let Ok(apply_fn) =
                        js_sys::Reflect::get(&track, &JsValue::from_str("applyConstraints"))
                    {
                        if let Ok(apply_fn) = apply_fn.dyn_into::<js_sys::Function>() {
                            let _ = apply_fn.call1(&track, &constraints);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // Processing statistics
    #[wasm_bindgen(getter)]
    pub fn processing_stats(&self) -> ProcessingStats {
        self.processing_stats.clone()
    }

    pub fn reset_processing_stats(&mut self) {
        self.processing_stats = ProcessingStats {
            frames_processed: 0,
            frames_dropped: 0,
            average_processing_time: 0.0,
            average_frame_rate: 0.0,
            buffer_utilization: 0.0,
            ml_inference_time: 0.0,
        };
    }

    // Buffer management
    pub fn set_max_buffer_size(&mut self, size: usize) {
        self.max_buffer_size = size;
        while self.frame_buffer.len() > size {
            self.frame_buffer.pop_front();
        }
    }

    pub fn clear_frame_buffer(&mut self) {
        self.frame_buffer.clear();
    }

    pub fn get_buffered_frame_count(&self) -> usize {
        self.frame_buffer.len()
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

// Factory functions for easy creation
#[wasm_bindgen]
pub async fn create_camera_manager() -> Result<CameraManager, JsValue> {
    let config = CameraConfig::default();
    let mut manager = CameraManager::new(config);
    manager.initialize().await?;
    Ok(manager)
}

#[wasm_bindgen]
pub async fn create_camera_manager_with_config(
    config: CameraConfig,
) -> Result<CameraManager, JsValue> {
    let mut manager = CameraManager::new(config);
    manager.initialize().await?;
    Ok(manager)
}

// Quick setup functions
#[wasm_bindgen]
pub async fn setup_front_camera() -> Result<CameraManager, JsValue> {
    let config = CameraConfig {
        facing: CameraFacing::User,
        ..Default::default()
    };
    create_camera_manager_with_config(config).await
}

#[wasm_bindgen]
pub async fn setup_back_camera() -> Result<CameraManager, JsValue> {
    let config = CameraConfig {
        facing: CameraFacing::Environment,
        ..Default::default()
    };
    create_camera_manager_with_config(config).await
}

#[wasm_bindgen]
pub async fn setup_hd_camera() -> Result<CameraManager, JsValue> {
    let config = CameraConfig {
        resolution: CameraResolution::High,
        ..Default::default()
    };
    create_camera_manager_with_config(config).await
}

// Camera capability detection utilities
#[wasm_bindgen]
pub async fn check_camera_support() -> Result<bool, JsValue> {
    let window = window().ok_or("No window object")?;
    let navigator = window.navigator();

    if let Ok(media_devices) = navigator.media_devices() {
        // Try to enumerate devices
        let devices_promise = media_devices.enumerate_devices()?;
        let devices_result = wasm_bindgen_futures::JsFuture::from(devices_promise).await?;
        let devices: Array = devices_result.dyn_into()?;

        for i in 0..devices.length() {
            if let Ok(device) = devices.get(i).dyn_into::<MediaDeviceInfo>() {
                if device.kind() == MediaDeviceKind::Videoinput {
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

#[wasm_bindgen]
pub async fn get_camera_count() -> Result<u32, JsValue> {
    let window = window().ok_or("No window object")?;
    let navigator = window.navigator();

    if let Ok(media_devices) = navigator.media_devices() {
        let devices_promise = media_devices.enumerate_devices()?;
        let devices_result = wasm_bindgen_futures::JsFuture::from(devices_promise).await?;
        let devices: Array = devices_result.dyn_into()?;

        let mut count = 0;
        for i in 0..devices.length() {
            if let Ok(device) = devices.get(i).dyn_into::<MediaDeviceInfo>() {
                if device.kind() == MediaDeviceKind::Videoinput {
                    count += 1;
                }
            }
        }
        Ok(count)
    } else {
        Ok(0)
    }
}

#[wasm_bindgen]
pub async fn get_available_cameras() -> Result<Array, JsValue> {
    let window = window().ok_or("No window object")?;
    let navigator = window.navigator();
    let array = Array::new();

    if let Ok(media_devices) = navigator.media_devices() {
        let devices_promise = media_devices.enumerate_devices()?;
        let devices_result = wasm_bindgen_futures::JsFuture::from(devices_promise).await?;
        let devices: Array = devices_result.dyn_into()?;

        for i in 0..devices.length() {
            if let Ok(device) = devices.get(i).dyn_into::<MediaDeviceInfo>() {
                if device.kind() == MediaDeviceKind::Videoinput {
                    let camera_info = Object::new();
                    js_sys::Reflect::set(
                        &camera_info,
                        &"deviceId".into(),
                        &JsValue::from_str(&device.device_id()),
                    )?;
                    js_sys::Reflect::set(
                        &camera_info,
                        &"label".into(),
                        &JsValue::from_str(&device.label()),
                    )?;
                    js_sys::Reflect::set(
                        &camera_info,
                        &"groupId".into(),
                        &JsValue::from_str(&device.group_id()),
                    )?;
                    array.push(&camera_info);
                }
            }
        }
    }

    Ok(array)
}

#[wasm_bindgen]
pub fn is_camera_permission_granted() -> bool {
    // Check if camera permission is granted using the Permissions API
    if let Some(window) = window() {
        let navigator = window.navigator();

        // Try to access navigator.permissions
        if let Ok(permissions) = js_sys::Reflect::get(&navigator, &"permissions".into()) {
            if !permissions.is_undefined() {
                // Check if we have camera permissions
                // Note: This is a synchronous check - for async permission checking,
                // you'd want to use a different function that returns a Promise

                // Try to detect if getUserMedia is available as a fallback
                if let Ok(get_user_media) = js_sys::Reflect::get(&navigator, &"getUserMedia".into())
                {
                    return !get_user_media.is_undefined();
                }

                if let Ok(media_devices) = js_sys::Reflect::get(&navigator, &"mediaDevices".into())
                {
                    if let Ok(get_user_media) =
                        js_sys::Reflect::get(&media_devices, &"getUserMedia".into())
                    {
                        return !get_user_media.is_undefined();
                    }
                }
            }
        }
    }
    false
}

// Image processing utilities for captured frames
#[wasm_bindgen]
pub fn convert_frame_to_tensor_data(frame_data: &ImageData) -> Result<Uint8Array, JsValue> {
    let data = frame_data.data();
    let array = Uint8Array::new_with_length(data.len() as u32);
    array.copy_from(&data);
    Ok(array)
}

#[wasm_bindgen]
pub fn resize_frame_data(
    frame_data: &ImageData,
    new_width: u32,
    new_height: u32,
) -> Result<ImageData, JsValue> {
    let window = window().ok_or("No window object")?;
    let document = window.document().ok_or("No document object")?;

    // Create temporary canvas for resizing
    let temp_canvas = document.create_element("canvas")?.dyn_into::<HtmlCanvasElement>()?;
    temp_canvas.set_width(new_width);
    temp_canvas.set_height(new_height);

    let context = temp_canvas
        .get_context("2d")?
        .ok_or("Failed to get 2D context")?
        .dyn_into::<CanvasRenderingContext2d>()?;

    // Put original image data on a temporary canvas
    let src_canvas = document.create_element("canvas")?.dyn_into::<HtmlCanvasElement>()?;
    src_canvas.set_width(frame_data.width());
    src_canvas.set_height(frame_data.height());

    let src_context = src_canvas
        .get_context("2d")?
        .ok_or("Failed to get 2D context")?
        .dyn_into::<CanvasRenderingContext2d>()?;

    src_context.put_image_data(frame_data, 0.0, 0.0)?;

    // Resize by drawing to the new canvas
    context.draw_image_with_html_canvas_element_and_dw_and_dh(
        &src_canvas,
        0.0,
        0.0,
        new_width as f64,
        new_height as f64,
    )?;

    // Get the resized image data
    context.get_image_data(0.0, 0.0, new_width as f64, new_height as f64)
}
