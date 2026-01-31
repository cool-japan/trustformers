use js_sys::{Array, Promise};
use serde::{Deserialize, Serialize};
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use web_sys::window;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DeviceClass {
    HighEnd,
    MidRange,
    LowEnd,
    Unknown,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NetworkType {
    Wifi,
    FourG,
    ThreeG,
    TwoG,
    Slow2G,
    Unknown,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BatteryStatus {
    High,     // > 75%
    Medium,   // 25-75%
    Low,      // 10-25%
    Critical, // < 10%
    Unknown,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelSize {
    Tiny,       // < 50MB
    Small,      // 50-200MB
    Medium,     // 200-500MB
    Large,      // 500MB-1GB
    ExtraLarge, // > 1GB
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileCapabilities {
    device_class: DeviceClass,
    memory_gb: f64,
    core_count: u32,
    max_texture_size: u32,
    supports_webgpu: bool,
    supports_webgl2: bool,
    supports_webassembly_simd: bool,
    screen_width: u32,
    screen_height: u32,
    device_pixel_ratio: f64,
    is_mobile: bool,
    is_tablet: bool,
}

#[wasm_bindgen]
impl MobileCapabilities {
    #[wasm_bindgen(getter)]
    pub fn device_class(&self) -> DeviceClass {
        self.device_class
    }

    #[wasm_bindgen(getter)]
    pub fn memory_gb(&self) -> f64 {
        self.memory_gb
    }

    #[wasm_bindgen(getter)]
    pub fn core_count(&self) -> u32 {
        self.core_count
    }

    #[wasm_bindgen(getter)]
    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    #[wasm_bindgen(getter)]
    pub fn supports_webgpu(&self) -> bool {
        self.supports_webgpu
    }

    #[wasm_bindgen(getter)]
    pub fn supports_webgl2(&self) -> bool {
        self.supports_webgl2
    }

    #[wasm_bindgen(getter)]
    pub fn supports_webassembly_simd(&self) -> bool {
        self.supports_webassembly_simd
    }

    #[wasm_bindgen(getter)]
    pub fn is_mobile(&self) -> bool {
        self.is_mobile
    }

    #[wasm_bindgen(getter)]
    pub fn is_tablet(&self) -> bool {
        self.is_tablet
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatus {
    network_type: NetworkType,
    effective_type: String,
    downlink_mbps: f64,
    rtt_ms: u32,
    save_data: bool,
    online: bool,
}

#[wasm_bindgen]
impl NetworkStatus {
    #[wasm_bindgen(getter)]
    pub fn network_type(&self) -> NetworkType {
        self.network_type
    }

    #[wasm_bindgen(getter)]
    pub fn effective_type(&self) -> String {
        self.effective_type.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn downlink_mbps(&self) -> f64 {
        self.downlink_mbps
    }

    #[wasm_bindgen(getter)]
    pub fn rtt_ms(&self) -> u32 {
        self.rtt_ms
    }

    #[wasm_bindgen(getter)]
    pub fn save_data(&self) -> bool {
        self.save_data
    }

    #[wasm_bindgen(getter)]
    pub fn online(&self) -> bool {
        self.online
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryInfo {
    status: BatteryStatus,
    level: f64, // 0.0 - 1.0
    charging: bool,
    charging_time: f64,    // seconds until full (if charging)
    discharging_time: f64, // seconds until empty (if discharging)
}

#[wasm_bindgen]
impl BatteryInfo {
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> BatteryStatus {
        self.status
    }

    #[wasm_bindgen(getter)]
    pub fn level(&self) -> f64 {
        self.level
    }

    #[wasm_bindgen(getter)]
    pub fn charging(&self) -> bool {
        self.charging
    }

    #[wasm_bindgen(getter)]
    pub fn charging_time(&self) -> f64 {
        self.charging_time
    }

    #[wasm_bindgen(getter)]
    pub fn discharging_time(&self) -> f64 {
        self.discharging_time
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveModelConfig {
    max_model_size: ModelSize,
    prefer_quantization: bool,
    enable_chunked_loading: bool,
    cache_models: bool,
    use_service_worker: bool,
    background_prefetch: bool,
    battery_aware: bool,
    network_aware: bool,
}

impl Default for AdaptiveModelConfig {
    fn default() -> Self {
        Self {
            max_model_size: ModelSize::Medium,
            prefer_quantization: true,
            enable_chunked_loading: true,
            cache_models: true,
            use_service_worker: true,
            background_prefetch: false,
            battery_aware: true,
            network_aware: true,
        }
    }
}

#[wasm_bindgen]
impl AdaptiveModelConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen(getter)]
    pub fn max_model_size(&self) -> ModelSize {
        self.max_model_size
    }

    #[wasm_bindgen(setter)]
    pub fn set_max_model_size(&mut self, size: ModelSize) {
        self.max_model_size = size;
    }

    #[wasm_bindgen(getter)]
    pub fn prefer_quantization(&self) -> bool {
        self.prefer_quantization
    }

    #[wasm_bindgen(setter)]
    pub fn set_prefer_quantization(&mut self, value: bool) {
        self.prefer_quantization = value;
    }

    #[wasm_bindgen(getter)]
    pub fn enable_chunked_loading(&self) -> bool {
        self.enable_chunked_loading
    }

    #[wasm_bindgen(setter)]
    pub fn set_enable_chunked_loading(&mut self, value: bool) {
        self.enable_chunked_loading = value;
    }

    #[wasm_bindgen(getter)]
    pub fn cache_models(&self) -> bool {
        self.cache_models
    }

    #[wasm_bindgen(setter)]
    pub fn set_cache_models(&mut self, value: bool) {
        self.cache_models = value;
    }

    #[wasm_bindgen(getter)]
    pub fn battery_aware(&self) -> bool {
        self.battery_aware
    }

    #[wasm_bindgen(setter)]
    pub fn set_battery_aware(&mut self, value: bool) {
        self.battery_aware = value;
    }

    #[wasm_bindgen(getter)]
    pub fn network_aware(&self) -> bool {
        self.network_aware
    }

    #[wasm_bindgen(setter)]
    pub fn set_network_aware(&mut self, value: bool) {
        self.network_aware = value;
    }
}

#[wasm_bindgen]
pub struct MobileOptimizer {
    config: AdaptiveModelConfig,
    capabilities: Option<MobileCapabilities>,
    network_status: Option<NetworkStatus>,
    battery_info: Option<BatteryInfo>,
    model_recommendations: Vec<String>,
}

#[wasm_bindgen]
impl MobileOptimizer {
    #[wasm_bindgen(constructor)]
    pub fn new(config: AdaptiveModelConfig) -> Self {
        Self {
            config,
            capabilities: None,
            network_status: None,
            battery_info: None,
            model_recommendations: Vec::new(),
        }
    }

    pub async fn initialize(&mut self) -> Result<(), JsValue> {
        self.detect_capabilities().await?;
        self.update_network_status().await?;
        self.update_battery_info().await?;
        self.generate_recommendations();
        Ok(())
    }

    async fn detect_capabilities(&mut self) -> Result<(), JsValue> {
        let window = window().ok_or("No window object")?;
        let navigator = window.navigator();

        // Detect device memory
        let device_memory =
            js_sys::Reflect::get(&navigator, &"deviceMemory".into()).unwrap_or_default();
        let memory_gb = if device_memory.is_undefined() {
            4.0 // Default assumption
        } else {
            device_memory.as_f64().unwrap_or(4.0)
        };

        // Detect hardware concurrency (CPU cores)
        let core_count = navigator.hardware_concurrency() as u32;

        // Detect screen dimensions
        let screen = window.screen().map_err(|_| "No screen object")?;
        let screen_width = screen.width().unwrap_or(1920) as u32;
        let screen_height = screen.height().unwrap_or(1080) as u32;
        let device_pixel_ratio = window.device_pixel_ratio();

        // Detect device type
        let user_agent = navigator.user_agent().unwrap_or_default();
        let is_mobile = user_agent.contains("Mobile") || user_agent.contains("Android");
        let is_tablet = user_agent.contains("Tablet") || user_agent.contains("iPad");

        // Detect WebGPU support
        let supports_webgpu = js_sys::Reflect::has(&navigator, &"gpu".into()).unwrap_or(false);

        // Detect WebGL2 support
        let supports_webgl2 = self.detect_webgl2_support();

        // Detect WebAssembly SIMD support
        let supports_webassembly_simd = self.detect_wasm_simd_support();

        // Estimate max texture size
        let max_texture_size = if supports_webgl2 { 16384 } else { 4096 };

        // Classify device
        let device_class = self.classify_device(memory_gb, core_count, supports_webgpu, is_mobile);

        self.capabilities = Some(MobileCapabilities {
            device_class,
            memory_gb,
            core_count,
            max_texture_size,
            supports_webgpu,
            supports_webgl2,
            supports_webassembly_simd,
            screen_width,
            screen_height,
            device_pixel_ratio,
            is_mobile,
            is_tablet,
        });

        Ok(())
    }

    fn classify_device(
        &self,
        memory_gb: f64,
        core_count: u32,
        supports_webgpu: bool,
        is_mobile: bool,
    ) -> DeviceClass {
        if is_mobile {
            if memory_gb >= 6.0 && core_count >= 8 && supports_webgpu {
                DeviceClass::HighEnd
            } else if memory_gb >= 3.0 && core_count >= 4 {
                DeviceClass::MidRange
            } else {
                DeviceClass::LowEnd
            }
        } else if memory_gb >= 8.0 && core_count >= 8 {
            DeviceClass::HighEnd
        } else if memory_gb >= 4.0 && core_count >= 4 {
            DeviceClass::MidRange
        } else {
            DeviceClass::LowEnd
        }
    }

    fn detect_webgl2_support(&self) -> bool {
        let window = match window() {
            Some(w) => w,
            None => return false,
        };
        let document = match window.document() {
            Some(d) => d,
            None => return false,
        };
        let canvas = match document.create_element("canvas") {
            Ok(c) => c,
            Err(_) => return false,
        };
        let html_canvas = match canvas.dyn_into::<web_sys::HtmlCanvasElement>() {
            Ok(c) => c,
            Err(_) => return false,
        };
        match html_canvas.get_context("webgl2") {
            Ok(Some(_)) => true,
            _ => false,
        }
    }

    fn detect_wasm_simd_support(&self) -> bool {
        // WebAssembly SIMD detection would require feature detection
        // For now, assume it's supported on modern browsers
        true
    }

    async fn update_network_status(&mut self) -> Result<(), JsValue> {
        let window = window().ok_or("No window object")?;
        let navigator = window.navigator();

        // Try to get NetworkInformation API
        let connection = js_sys::Reflect::get(&navigator, &"connection".into())
            .or_else(|_| js_sys::Reflect::get(&navigator, &"mozConnection".into()))
            .or_else(|_| js_sys::Reflect::get(&navigator, &"webkitConnection".into()));

        let network_status = if let Ok(conn) = connection {
            if !conn.is_undefined() {
                let effective_type = js_sys::Reflect::get(&conn, &"effectiveType".into())
                    .unwrap_or_default()
                    .as_string()
                    .unwrap_or_default();

                let downlink = js_sys::Reflect::get(&conn, &"downlink".into())
                    .unwrap_or_default()
                    .as_f64()
                    .unwrap_or(10.0);

                let rtt = js_sys::Reflect::get(&conn, &"rtt".into())
                    .unwrap_or_default()
                    .as_f64()
                    .unwrap_or(50.0) as u32;

                let save_data = js_sys::Reflect::get(&conn, &"saveData".into())
                    .unwrap_or_default()
                    .as_bool()
                    .unwrap_or(false);

                let network_type = match effective_type.as_str() {
                    "slow-2g" => NetworkType::Slow2G,
                    "2g" => NetworkType::TwoG,
                    "3g" => NetworkType::ThreeG,
                    "4g" => NetworkType::FourG,
                    _ => {
                        if downlink > 10.0 {
                            NetworkType::Wifi
                        } else {
                            NetworkType::Unknown
                        }
                    },
                };

                NetworkStatus {
                    network_type,
                    effective_type,
                    downlink_mbps: downlink,
                    rtt_ms: rtt,
                    save_data,
                    online: navigator.on_line(),
                }
            } else {
                // Fallback for browsers without NetworkInformation API
                NetworkStatus {
                    network_type: NetworkType::Unknown,
                    effective_type: "unknown".to_string(),
                    downlink_mbps: 10.0,
                    rtt_ms: 50,
                    save_data: false,
                    online: navigator.on_line(),
                }
            }
        } else {
            NetworkStatus {
                network_type: NetworkType::Unknown,
                effective_type: "unknown".to_string(),
                downlink_mbps: 10.0,
                rtt_ms: 50,
                save_data: false,
                online: navigator.on_line(),
            }
        };

        self.network_status = Some(network_status);
        Ok(())
    }

    async fn update_battery_info(&mut self) -> Result<(), JsValue> {
        let window = window().ok_or("No window object")?;
        let navigator = window.navigator();

        // Try to get Battery API
        let get_battery = js_sys::Reflect::get(&navigator, &"getBattery".into());

        let battery_info = if let Ok(get_battery_fn) = get_battery {
            if !get_battery_fn.is_undefined() {
                // Call getBattery() and await the promise
                let battery_promise = js_sys::Function::from(get_battery_fn).call0(&navigator)?;
                let battery_promise: Promise = battery_promise.dyn_into()?;

                match wasm_bindgen_futures::JsFuture::from(battery_promise).await {
                    Ok(battery) => {
                        let level = js_sys::Reflect::get(&battery, &"level".into())
                            .unwrap_or_default()
                            .as_f64()
                            .unwrap_or(1.0);

                        let charging = js_sys::Reflect::get(&battery, &"charging".into())
                            .unwrap_or_default()
                            .as_bool()
                            .unwrap_or(false);

                        let charging_time = js_sys::Reflect::get(&battery, &"chargingTime".into())
                            .unwrap_or_default()
                            .as_f64()
                            .unwrap_or(f64::INFINITY);

                        let discharging_time =
                            js_sys::Reflect::get(&battery, &"dischargingTime".into())
                                .unwrap_or_default()
                                .as_f64()
                                .unwrap_or(f64::INFINITY);

                        let status = if level > 0.75 {
                            BatteryStatus::High
                        } else if level > 0.25 {
                            BatteryStatus::Medium
                        } else if level > 0.10 {
                            BatteryStatus::Low
                        } else {
                            BatteryStatus::Critical
                        };

                        BatteryInfo {
                            status,
                            level,
                            charging,
                            charging_time,
                            discharging_time,
                        }
                    },
                    Err(_) => {
                        // Fallback when Battery API is not available
                        BatteryInfo {
                            status: BatteryStatus::Unknown,
                            level: 1.0,
                            charging: false,
                            charging_time: f64::INFINITY,
                            discharging_time: f64::INFINITY,
                        }
                    },
                }
            } else {
                BatteryInfo {
                    status: BatteryStatus::Unknown,
                    level: 1.0,
                    charging: false,
                    charging_time: f64::INFINITY,
                    discharging_time: f64::INFINITY,
                }
            }
        } else {
            BatteryInfo {
                status: BatteryStatus::Unknown,
                level: 1.0,
                charging: false,
                charging_time: f64::INFINITY,
                discharging_time: f64::INFINITY,
            }
        };

        self.battery_info = Some(battery_info);
        Ok(())
    }

    fn generate_recommendations(&mut self) {
        let mut recommendations = Vec::new();

        if let Some(capabilities) = &self.capabilities {
            // Device-based recommendations
            match capabilities.device_class {
                DeviceClass::HighEnd => {
                    recommendations
                        .push("High-end device detected: Can handle large models".to_string());
                    recommendations
                        .push("Consider using full-precision models for best accuracy".to_string());
                    if capabilities.supports_webgpu {
                        recommendations
                            .push("WebGPU available: Enable GPU acceleration".to_string());
                    }
                },
                DeviceClass::MidRange => {
                    recommendations
                        .push("Mid-range device: Recommend medium-sized models".to_string());
                    recommendations.push("Enable quantization for better performance".to_string());
                    recommendations.push("Use chunked loading for large models".to_string());
                },
                DeviceClass::LowEnd => {
                    recommendations
                        .push("Low-end device: Use small or quantized models only".to_string());
                    recommendations
                        .push("Enable aggressive caching to reduce load times".to_string());
                    recommendations.push("Consider progressive model loading".to_string());
                },
                DeviceClass::Unknown => {
                    recommendations
                        .push("Device capabilities unknown: Use conservative settings".to_string());
                },
            }

            if capabilities.is_mobile {
                recommendations
                    .push("Mobile device: Enable battery and network awareness".to_string());
                recommendations
                    .push("Consider reducing model size for better user experience".to_string());
            }
        }

        if let Some(network) = &self.network_status {
            // Network-based recommendations
            match network.network_type {
                NetworkType::Wifi => {
                    recommendations
                        .push("WiFi connection: Full models can be downloaded".to_string());
                },
                NetworkType::FourG => {
                    recommendations
                        .push("4G connection: Moderate model sizes recommended".to_string());
                },
                NetworkType::ThreeG | NetworkType::TwoG | NetworkType::Slow2G => {
                    recommendations
                        .push("Slow connection: Use only cached or tiny models".to_string());
                    recommendations.push("Enable progressive loading and compression".to_string());
                },
                NetworkType::Unknown => {
                    recommendations.push(
                        "Network speed unknown: Use conservative loading strategy".to_string(),
                    );
                },
            }

            if network.save_data {
                recommendations.push("Data saver mode enabled: Minimize downloads".to_string());
            }

            if !network.online {
                recommendations.push("Offline mode: Use only cached models".to_string());
            }
        }

        if let Some(battery) = &self.battery_info {
            // Battery-based recommendations
            match battery.status {
                BatteryStatus::Critical | BatteryStatus::Low => {
                    recommendations
                        .push("Low battery: Reduce CPU/GPU intensive operations".to_string());
                    recommendations.push("Use quantized models to reduce computation".to_string());
                },
                BatteryStatus::Medium => {
                    recommendations
                        .push("Medium battery: Balance performance and efficiency".to_string());
                },
                BatteryStatus::High => {
                    recommendations
                        .push("High battery: Full performance mode available".to_string());
                },
                BatteryStatus::Unknown => {
                    recommendations
                        .push("Battery status unknown: Use moderate power settings".to_string());
                },
            }

            if battery.charging {
                recommendations
                    .push("Device charging: Can use higher performance settings".to_string());
            }
        }

        self.model_recommendations = recommendations;
    }

    pub fn get_recommended_model_size(&self) -> ModelSize {
        if let Some(capabilities) = &self.capabilities {
            if let Some(network) = &self.network_status {
                if let Some(battery) = &self.battery_info {
                    // Combine all factors to determine optimal model size
                    let base_size = match capabilities.device_class {
                        DeviceClass::HighEnd => ModelSize::Large,
                        DeviceClass::MidRange => ModelSize::Medium,
                        DeviceClass::LowEnd => ModelSize::Small,
                        DeviceClass::Unknown => ModelSize::Small,
                    };

                    // Adjust based on network
                    let network_adjusted = match network.network_type {
                        NetworkType::Wifi => base_size,
                        NetworkType::FourG => match base_size {
                            ModelSize::Large => ModelSize::Medium,
                            ModelSize::ExtraLarge => ModelSize::Large,
                            other => other,
                        },
                        NetworkType::ThreeG | NetworkType::TwoG | NetworkType::Slow2G => {
                            ModelSize::Tiny
                        },
                        NetworkType::Unknown => ModelSize::Small,
                    };

                    // Adjust based on battery
                    match battery.status {
                        BatteryStatus::Critical | BatteryStatus::Low => ModelSize::Tiny,
                        BatteryStatus::Medium => match network_adjusted {
                            ModelSize::Large => ModelSize::Medium,
                            ModelSize::ExtraLarge => ModelSize::Large,
                            other => other,
                        },
                        _ => network_adjusted,
                    }
                } else {
                    ModelSize::Small
                }
            } else {
                ModelSize::Small
            }
        } else {
            ModelSize::Small
        }
    }

    pub fn should_use_quantization(&self) -> bool {
        if !self.config.prefer_quantization {
            return false;
        }

        if let Some(capabilities) = &self.capabilities {
            match capabilities.device_class {
                DeviceClass::LowEnd => true,
                DeviceClass::MidRange => {
                    // Use quantization on mid-range mobile devices
                    capabilities.is_mobile
                },
                DeviceClass::HighEnd => {
                    // Use quantization only if battery is low or network is slow
                    if let Some(battery) = &self.battery_info {
                        if matches!(battery.status, BatteryStatus::Low | BatteryStatus::Critical) {
                            return true;
                        }
                    }

                    if let Some(network) = &self.network_status {
                        matches!(
                            network.network_type,
                            NetworkType::ThreeG | NetworkType::TwoG | NetworkType::Slow2G
                        )
                    } else {
                        false
                    }
                },
                DeviceClass::Unknown => true,
            }
        } else {
            true
        }
    }

    pub fn should_use_chunked_loading(&self) -> bool {
        if !self.config.enable_chunked_loading {
            return false;
        }

        if let Some(network) = &self.network_status {
            match network.network_type {
                NetworkType::Wifi => false,
                NetworkType::FourG => true,
                NetworkType::ThreeG | NetworkType::TwoG | NetworkType::Slow2G => true,
                NetworkType::Unknown => true,
            }
        } else {
            true
        }
    }

    pub fn get_recommended_chunk_size_kb(&self) -> u32 {
        if let Some(network) = &self.network_status {
            match network.network_type {
                NetworkType::Wifi => 1024,  // 1MB chunks
                NetworkType::FourG => 512,  // 512KB chunks
                NetworkType::ThreeG => 256, // 256KB chunks
                NetworkType::TwoG => 128,   // 128KB chunks
                NetworkType::Slow2G => 64,  // 64KB chunks
                NetworkType::Unknown => 256,
            }
        } else {
            256 // Default safe chunk size
        }
    }

    pub fn should_enable_background_prefetch(&self) -> bool {
        if !self.config.background_prefetch {
            return false;
        }

        // Only enable background prefetch on good conditions
        let good_network = if let Some(network) = &self.network_status {
            matches!(network.network_type, NetworkType::Wifi | NetworkType::FourG)
                && !network.save_data
        } else {
            false
        };

        let good_battery = if let Some(battery) = &self.battery_info {
            matches!(battery.status, BatteryStatus::High | BatteryStatus::Medium)
                || battery.charging
        } else {
            false
        };

        good_network && good_battery
    }

    #[wasm_bindgen(getter)]
    pub fn capabilities(&self) -> Option<MobileCapabilities> {
        self.capabilities.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn network_status(&self) -> Option<NetworkStatus> {
        self.network_status.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn battery_info(&self) -> Option<BatteryInfo> {
        self.battery_info.clone()
    }

    pub fn get_recommendations(&self) -> Array {
        let array = Array::new();
        for recommendation in &self.model_recommendations {
            array.push(&JsValue::from_str(recommendation));
        }
        array
    }

    pub fn export_profile(&self) -> Result<String, JsValue> {
        let profile = serde_json::json!({
            "capabilities": self.capabilities,
            "network_status": self.network_status,
            "battery_info": self.battery_info,
            "recommendations": self.model_recommendations,
            "recommended_model_size": self.get_recommended_model_size(),
            "use_quantization": self.should_use_quantization(),
            "use_chunked_loading": self.should_use_chunked_loading(),
            "chunk_size_kb": self.get_recommended_chunk_size_kb(),
            "enable_background_prefetch": self.should_enable_background_prefetch(),
        });

        serde_json::to_string_pretty(&profile).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// Utility functions
#[wasm_bindgen]
pub fn is_mobile_device() -> bool {
    if let Some(window) = window() {
        let navigator = window.navigator();
        let user_agent = navigator.user_agent().unwrap_or_default();
        user_agent.contains("Mobile") || user_agent.contains("Android")
    } else {
        false
    }
}

#[wasm_bindgen]
pub fn is_tablet_device() -> bool {
    if let Some(window) = window() {
        let navigator = window.navigator();
        let user_agent = navigator.user_agent().unwrap_or_default();
        user_agent.contains("Tablet") || user_agent.contains("iPad")
    } else {
        false
    }
}

#[wasm_bindgen]
pub fn get_device_memory_gb() -> f64 {
    if let Some(window) = window() {
        let navigator = window.navigator();
        let device_memory =
            js_sys::Reflect::get(&navigator, &"deviceMemory".into()).unwrap_or_default();
        device_memory.as_f64().unwrap_or(4.0)
    } else {
        4.0
    }
}

#[wasm_bindgen]
pub fn is_low_data_mode() -> bool {
    if let Some(window) = window() {
        let navigator = window.navigator();
        let connection = js_sys::Reflect::get(&navigator, &"connection".into())
            .or_else(|_| js_sys::Reflect::get(&navigator, &"mozConnection".into()))
            .or_else(|_| js_sys::Reflect::get(&navigator, &"webkitConnection".into()));

        if let Ok(conn) = connection {
            if !conn.is_undefined() {
                return js_sys::Reflect::get(&conn, &"saveData".into())
                    .unwrap_or_default()
                    .as_bool()
                    .unwrap_or(false);
            }
        }
    }
    false
}

#[wasm_bindgen]
pub async fn create_mobile_optimizer() -> Result<MobileOptimizer, JsValue> {
    let config = AdaptiveModelConfig::default();
    let mut optimizer = MobileOptimizer::new(config);
    optimizer.initialize().await?;
    Ok(optimizer)
}

#[wasm_bindgen]
pub fn get_optimal_model_for_device() -> Result<String, JsValue> {
    let config = AdaptiveModelConfig::default();
    let _optimizer = MobileOptimizer::new(config);

    // Quick synchronous assessment based on user agent
    let is_mobile = is_mobile_device();
    let device_memory = get_device_memory_gb();
    let is_low_data = is_low_data_mode();

    let recommendation = if is_mobile {
        if device_memory < 2.0 || is_low_data {
            "tiny-quantized"
        } else if device_memory < 4.0 {
            "small-quantized"
        } else if device_memory < 6.0 {
            "medium-quantized"
        } else {
            "medium"
        }
    } else if device_memory < 4.0 {
        "small"
    } else if device_memory < 8.0 {
        "medium"
    } else {
        "large"
    };

    Ok(recommendation.to_string())
}
