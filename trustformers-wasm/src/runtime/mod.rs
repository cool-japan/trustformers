// Runtime Module - Edge computing and deployment
//
// This module provides runtime capabilities for deploying TrustformeRS
// models in edge computing environments including CDNs, edge functions,
// and distributed inference systems.

#[cfg(feature = "web-workers")]
pub mod edge_runtime;

#[cfg(feature = "web-workers")]
pub mod geo_distribution;

#[cfg(feature = "web-workers")]
pub mod edge_caching;

#[cfg(feature = "web-workers")]
pub mod service_worker;

// Re-export main types for convenience
#[cfg(feature = "web-workers")]
pub use edge_runtime::{EdgeCapabilities, EdgeInferenceConfig, EdgeRuntime, EdgeRuntimeDetector};

#[cfg(feature = "web-workers")]
pub use geo_distribution::{
    create_geo_distribution_manager, estimate_network_latency, get_distance_between_points,
    EdgeLocation, GeoDistributionManager, GeoRegion, RoutingDecision, RoutingWeights, UserLocation,
};

#[cfg(feature = "web-workers")]
pub use edge_caching::{
    create_edge_cache_manager, create_edge_computing_cache_config,
    create_memory_efficient_cache_config, create_performance_cache_config, estimate_cache_overhead,
    CacheConfig, CacheEntry, CacheEntryType, CacheStatistics, ConsistencyLevel, EdgeCacheManager,
    EvictionPolicy, ReplicationStrategy,
};

#[cfg(feature = "web-workers")]
pub use service_worker::{
    CachePriority, CacheStats, InferenceConfig, PWAInstaller, ServiceWorkerManager,
    ServiceWorkerMessage, ServiceWorkerResponse,
};

/// Runtime module initialization
pub fn initialize() -> Result<(), RuntimeError> {
    web_sys::console::log_1(&"Initializing TrustformeRS WASM runtime module".into());

    #[cfg(feature = "web-workers")]
    {
        // Module initialize functions not yet implemented
        // edge_runtime::initialize()?;
        // geo_distribution::initialize()?;
        // edge_caching::initialize()?;
        // service_worker::initialize()?;
        web_sys::console::log_1(&"Edge runtime subsystems initialized".into());
    }

    web_sys::console::log_1(&"TrustformeRS WASM runtime module initialized successfully".into());
    Ok(())
}

/// Runtime module error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    EdgeRuntimeError(String),
    GeoDistributionError(String),
    CachingError(String),
    ServiceWorkerError(String),
    NetworkError(String),
    ConfigurationError(String),
    InitializationError(String),
}

impl core::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RuntimeError::EdgeRuntimeError(msg) => write!(f, "Edge runtime error: {}", msg),
            RuntimeError::GeoDistributionError(msg) => write!(f, "Geo distribution error: {}", msg),
            RuntimeError::CachingError(msg) => write!(f, "Caching error: {}", msg),
            RuntimeError::ServiceWorkerError(msg) => write!(f, "Service worker error: {}", msg),
            RuntimeError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            RuntimeError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            RuntimeError::InitializationError(msg) => write!(f, "Initialization error: {}", msg),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Runtime module configuration
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub enable_edge_caching: bool,
    pub enable_geo_distribution: bool,
    pub cache_size_mb: u32,
    pub max_edge_locations: u32,
    pub failover_enabled: bool,
    pub compression_enabled: bool,
    pub analytics_enabled: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            enable_edge_caching: true,
            enable_geo_distribution: true,
            cache_size_mb: 100,
            max_edge_locations: 50,
            failover_enabled: true,
            compression_enabled: true,
            analytics_enabled: true,
        }
    }
}

/// Runtime environment detection
#[derive(Debug, Clone)]
pub struct RuntimeEnvironment {
    pub platform: RuntimePlatform,
    pub capabilities: RuntimeCapabilities,
    pub constraints: RuntimeConstraints,
    pub location: Option<GeographicLocation>,
}

impl RuntimeEnvironment {
    /// Detect the current runtime environment
    pub async fn detect() -> Self {
        let platform = Self::detect_platform();
        let capabilities = Self::detect_capabilities().await;
        let constraints = Self::detect_constraints();
        let location = Self::detect_location().await;

        Self {
            platform,
            capabilities,
            constraints,
            location,
        }
    }

    fn detect_platform() -> RuntimePlatform {
        let user_agent = web_sys::window()
            .and_then(|w| w.navigator().user_agent().ok())
            .unwrap_or_default();

        if user_agent.contains("Cloudflare-Workers") {
            RuntimePlatform::CloudflareWorkers
        } else if user_agent.contains("Deno") {
            RuntimePlatform::DenoDeploy
        } else if user_agent.contains("Vercel") {
            RuntimePlatform::VercelEdge
        } else if user_agent.contains("AWS") {
            RuntimePlatform::AWSLambdaEdge
        } else if user_agent.contains("Fastly") {
            RuntimePlatform::FastlyComputeEdge
        } else if user_agent.contains("Chrome")
            || user_agent.contains("Firefox")
            || user_agent.contains("Safari")
        {
            RuntimePlatform::Browser
        } else {
            RuntimePlatform::Unknown
        }
    }

    async fn detect_capabilities() -> RuntimeCapabilities {
        RuntimeCapabilities {
            has_web_workers: Self::has_web_workers(),
            has_service_workers: Self::has_service_workers(),
            has_indexeddb: Self::has_indexeddb(),
            has_cache_api: Self::has_cache_api(),
            has_fetch_api: Self::has_fetch_api(),
            has_websockets: Self::has_websockets(),
            has_webrtc: Self::has_webrtc(),
            max_memory_mb: Self::get_max_memory(),
            max_execution_time_ms: Self::get_max_execution_time(),
        }
    }

    fn detect_constraints() -> RuntimeConstraints {
        RuntimeConstraints {
            memory_limit_mb: None, // Will be detected based on platform
            execution_time_limit_ms: None,
            network_bandwidth_limit: None,
            storage_limit_mb: None,
            concurrent_requests_limit: None,
        }
    }

    async fn detect_location() -> Option<GeographicLocation> {
        // Try to get location from various sources
        if let Some(_geolocation) = web_sys::window().and_then(|w| w.navigator().geolocation().ok())
        {
            // Browser geolocation API (requires user permission)
            None // For now, return None
        } else {
            // Try CloudFlare headers, Vercel headers, etc.
            None
        }
    }

    // Helper methods for capability detection
    fn has_web_workers() -> bool {
        js_sys::Reflect::has(&js_sys::global(), &"Worker".into()).unwrap_or(false)
    }

    fn has_service_workers() -> bool {
        web_sys::window().map(|w| w.navigator().service_worker()).is_some()
    }

    fn has_indexeddb() -> bool {
        js_sys::Reflect::has(&js_sys::global(), &"indexedDB".into()).unwrap_or(false)
    }

    fn has_cache_api() -> bool {
        js_sys::Reflect::has(&js_sys::global(), &"caches".into()).unwrap_or(false)
    }

    fn has_fetch_api() -> bool {
        js_sys::Reflect::has(&js_sys::global(), &"fetch".into()).unwrap_or(false)
    }

    fn has_websockets() -> bool {
        js_sys::Reflect::has(&js_sys::global(), &"WebSocket".into()).unwrap_or(false)
    }

    fn has_webrtc() -> bool {
        js_sys::Reflect::has(&js_sys::global(), &"RTCPeerConnection".into()).unwrap_or(false)
    }

    fn get_max_memory() -> Option<u32> {
        // Try to detect available memory
        if let Some(navigator) = web_sys::window().map(|w| w.navigator()) {
            // Chrome has deviceMemory API
            js_sys::Reflect::get(&navigator, &"deviceMemory".into())
                .ok()
                .and_then(|v| v.as_f64())
                .map(|mb| (mb * 1024.0) as u32)
        } else {
            None
        }
    }

    fn get_max_execution_time() -> Option<u32> {
        // Platform-specific execution time limits
        None // Will be set based on detected platform
    }
}

/// Runtime platform types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePlatform {
    Browser,
    CloudflareWorkers,
    DenoDeploy,
    VercelEdge,
    AWSLambdaEdge,
    FastlyComputeEdge,
    NetlifyEdge,
    Unknown,
}

impl core::fmt::Display for RuntimePlatform {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            RuntimePlatform::Browser => "Browser",
            RuntimePlatform::CloudflareWorkers => "Cloudflare Workers",
            RuntimePlatform::DenoDeploy => "Deno Deploy",
            RuntimePlatform::VercelEdge => "Vercel Edge",
            RuntimePlatform::AWSLambdaEdge => "AWS Lambda@Edge",
            RuntimePlatform::FastlyComputeEdge => "Fastly Compute@Edge",
            RuntimePlatform::NetlifyEdge => "Netlify Edge",
            RuntimePlatform::Unknown => "Unknown",
        };
        write!(f, "{}", name)
    }
}

/// Runtime capabilities
#[derive(Debug, Clone)]
pub struct RuntimeCapabilities {
    pub has_web_workers: bool,
    pub has_service_workers: bool,
    pub has_indexeddb: bool,
    pub has_cache_api: bool,
    pub has_fetch_api: bool,
    pub has_websockets: bool,
    pub has_webrtc: bool,
    pub max_memory_mb: Option<u32>,
    pub max_execution_time_ms: Option<u32>,
}

/// Runtime constraints
#[derive(Debug, Clone)]
pub struct RuntimeConstraints {
    pub memory_limit_mb: Option<u32>,
    pub execution_time_limit_ms: Option<u32>,
    pub network_bandwidth_limit: Option<u32>,
    pub storage_limit_mb: Option<u32>,
    pub concurrent_requests_limit: Option<u32>,
}

/// Geographic location information
#[derive(Debug, Clone)]
pub struct GeographicLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub country: Option<String>,
    pub region: Option<String>,
    pub city: Option<String>,
    pub timezone: Option<String>,
}

/// Runtime manager for coordinating edge deployment
pub struct RuntimeManager {
    config: RuntimeConfig,
    environment: RuntimeEnvironment,
    #[cfg(feature = "web-workers")]
    cache_manager: Option<edge_caching::EdgeCacheManager>,
    #[cfg(feature = "web-workers")]
    geo_manager: Option<geo_distribution::GeoDistributionManager>,
}

impl RuntimeManager {
    /// Create a new runtime manager
    pub async fn new(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        let environment = RuntimeEnvironment::detect().await;

        let mut manager = Self {
            config,
            environment,
            #[cfg(feature = "web-workers")]
            cache_manager: None,
            #[cfg(feature = "web-workers")]
            geo_manager: None,
        };

        manager.initialize_subsystems().await?;
        Ok(manager)
    }

    async fn initialize_subsystems(&mut self) -> Result<(), RuntimeError> {
        #[cfg(feature = "web-workers")]
        {
            if self.config.enable_edge_caching {
                let cache_config = edge_caching::CacheConfig {
                    max_size_bytes: (self.config.cache_size_mb as usize) * 1024 * 1024,
                    enable_compression: self.config.compression_enabled,
                    ..Default::default()
                };
                // Use Europe as default region; can be updated later based on user location
                let default_region = geo_distribution::GeoRegion::Europe;
                self.cache_manager = Some(edge_caching::EdgeCacheManager::new(
                    cache_config,
                    default_region,
                ));
                web_sys::console::log_1(&"Edge cache manager initialized".into());
            }

            if self.config.enable_geo_distribution {
                self.geo_manager = Some(geo_distribution::GeoDistributionManager::new());
                web_sys::console::log_1(&"Geo distribution manager initialized".into());
            }
        }

        Ok(())
    }

    /// Get runtime environment information
    pub fn environment(&self) -> &RuntimeEnvironment {
        &self.environment
    }

    /// Get runtime configuration
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Check if feature is supported in current runtime
    pub fn is_feature_supported(&self, feature: RuntimeFeature) -> bool {
        match feature {
            RuntimeFeature::WebWorkers => self.environment.capabilities.has_web_workers,
            RuntimeFeature::ServiceWorkers => self.environment.capabilities.has_service_workers,
            RuntimeFeature::IndexedDB => self.environment.capabilities.has_indexeddb,
            RuntimeFeature::CacheAPI => self.environment.capabilities.has_cache_api,
            RuntimeFeature::WebSockets => self.environment.capabilities.has_websockets,
            RuntimeFeature::WebRTC => self.environment.capabilities.has_webrtc,
        }
    }
}

/// Runtime features enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeFeature {
    WebWorkers,
    ServiceWorkers,
    IndexedDB,
    CacheAPI,
    WebSockets,
    WebRTC,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_default() {
        let config = RuntimeConfig::default();
        assert!(config.enable_edge_caching);
        assert!(config.enable_geo_distribution);
        assert_eq!(config.cache_size_mb, 100);
        assert!(config.failover_enabled);
    }

    #[test]
    fn test_runtime_platform_display() {
        assert_eq!(format!("{}", RuntimePlatform::Browser), "Browser");
        assert_eq!(
            format!("{}", RuntimePlatform::CloudflareWorkers),
            "Cloudflare Workers"
        );
        assert_eq!(format!("{}", RuntimePlatform::VercelEdge), "Vercel Edge");
    }
}
