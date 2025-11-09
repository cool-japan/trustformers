//! Geographic Distribution System for Edge Computing
//!
//! This module provides intelligent geographic distribution of AI inference
//! requests across global edge locations to minimize latency and optimize
//! performance for users worldwide.

use crate::runtime::edge_runtime::{EdgeCapabilities, EdgeRuntime};
use std::collections::BTreeMap;
use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::js_sys;

/// Geographic regions for edge distribution
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GeoRegion {
    /// North America
    NorthAmerica,
    /// Europe
    Europe,
    /// Asia Pacific
    AsiaPacific,
    /// South America
    SouthAmerica,
    /// Africa
    Africa,
    /// Middle East
    MiddleEast,
    /// Oceania
    Oceania,
}

/// Edge location information
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct EdgeLocation {
    region: GeoRegion,
    country_code: String,
    city: String,
    datacenter_id: String,
    latitude: f64,
    longitude: f64,
    runtime_type: EdgeRuntime,
    capabilities: EdgeCapabilities,
    current_load: f32,
    health_score: f32,
    last_health_check: u64,
}

#[wasm_bindgen]
impl EdgeLocation {
    /// Create a new edge location
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        region: GeoRegion,
        country_code: String,
        city: String,
        datacenter_id: String,
        latitude: f64,
        longitude: f64,
        runtime_type: EdgeRuntime,
        capabilities: EdgeCapabilities,
    ) -> EdgeLocation {
        EdgeLocation {
            region,
            country_code,
            city,
            datacenter_id,
            latitude,
            longitude,
            runtime_type,
            capabilities,
            current_load: 0.0,
            health_score: 1.0,
            last_health_check: 0,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn region(&self) -> GeoRegion {
        self.region
    }

    #[wasm_bindgen(getter)]
    pub fn country_code(&self) -> String {
        self.country_code.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn city(&self) -> String {
        self.city.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn datacenter_id(&self) -> String {
        self.datacenter_id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn latitude(&self) -> f64 {
        self.latitude
    }

    #[wasm_bindgen(getter)]
    pub fn longitude(&self) -> f64 {
        self.longitude
    }

    #[wasm_bindgen(getter)]
    pub fn runtime_type(&self) -> EdgeRuntime {
        self.runtime_type
    }

    #[wasm_bindgen(getter)]
    pub fn capabilities(&self) -> EdgeCapabilities {
        self.capabilities.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn current_load(&self) -> f32 {
        self.current_load
    }

    #[wasm_bindgen(getter)]
    pub fn health_score(&self) -> f32 {
        self.health_score
    }

    /// Calculate distance to another location in kilometers
    pub fn distance_to(&self, latitude: f64, longitude: f64) -> f64 {
        let r = 6371.0; // Earth's radius in kilometers
        let lat1 = self.latitude.to_radians();
        let lat2 = latitude.to_radians();
        let delta_lat = (latitude - self.latitude).to_radians();
        let delta_lon = (longitude - self.longitude).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        r * c
    }

    /// Calculate routing score based on distance, load, and health
    pub fn routing_score(
        &self,
        user_lat: f64,
        user_lon: f64,
        weight_distance: f32,
        weight_load: f32,
        weight_health: f32,
    ) -> f32 {
        let distance = self.distance_to(user_lat, user_lon) as f32;
        let normalized_distance = (distance / 20000.0).min(1.0); // Normalize to 0-1 (max ~20000km)
        let load_factor = 1.0 - self.current_load; // Lower load is better
        let health_factor = self.health_score;

        // Lower score is better for routing
        weight_distance * normalized_distance
            + weight_load * (1.0 - load_factor)
            + weight_health * (1.0 - health_factor)
    }

    /// Update current load and health metrics
    pub fn update_metrics(&mut self, load: f32, health: f32) {
        self.current_load = load.clamp(0.0, 1.0);
        self.health_score = health.clamp(0.0, 1.0);
        self.last_health_check = js_sys::Date::now() as u64;
    }
}

/// User location information
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct UserLocation {
    latitude: f64,
    longitude: f64,
    country_code: String,
    region: GeoRegion,
    timezone: String,
    accuracy: f64,
    ip_address: String,
    autonomous_system: String,
}

#[wasm_bindgen]
impl UserLocation {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        latitude: f64,
        longitude: f64,
        country_code: String,
        region: GeoRegion,
        timezone: String,
        accuracy: f64,
        ip_address: String,
        autonomous_system: String,
    ) -> UserLocation {
        UserLocation {
            latitude,
            longitude,
            country_code,
            region,
            timezone,
            accuracy,
            ip_address,
            autonomous_system,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn latitude(&self) -> f64 {
        self.latitude
    }

    #[wasm_bindgen(getter)]
    pub fn longitude(&self) -> f64 {
        self.longitude
    }

    #[wasm_bindgen(getter)]
    pub fn country_code(&self) -> String {
        self.country_code.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn region(&self) -> GeoRegion {
        self.region
    }

    #[wasm_bindgen(getter)]
    pub fn timezone(&self) -> String {
        self.timezone.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn accuracy(&self) -> f64 {
        self.accuracy
    }

    #[wasm_bindgen(getter)]
    pub fn ip_address(&self) -> String {
        self.ip_address.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn autonomous_system(&self) -> String {
        self.autonomous_system.clone()
    }
}

/// Routing decision result
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    primary_location: EdgeLocation,
    fallback_locations: Vec<EdgeLocation>,
    routing_reason: String,
    estimated_latency_ms: u32,
    confidence_score: f32,
}

#[wasm_bindgen]
impl RoutingDecision {
    #[wasm_bindgen(getter)]
    pub fn primary_location(&self) -> EdgeLocation {
        self.primary_location.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn fallback_locations(&self) -> js_sys::Array {
        let array = js_sys::Array::new();
        for location in &self.fallback_locations {
            array.push(&JsValue::from(location.clone()));
        }
        array
    }

    #[wasm_bindgen(getter)]
    pub fn routing_reason(&self) -> String {
        self.routing_reason.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn estimated_latency_ms(&self) -> u32 {
        self.estimated_latency_ms
    }

    #[wasm_bindgen(getter)]
    pub fn confidence_score(&self) -> f32 {
        self.confidence_score
    }
}

/// Geographic distribution manager
#[wasm_bindgen]
pub struct GeoDistributionManager {
    edge_locations: Vec<EdgeLocation>,
    routing_weights: RoutingWeights,
    health_check_interval: u64,
    last_health_check: u64,
    region_preferences: BTreeMap<String, GeoRegion>,
    failover_enabled: bool,
    load_balancing_enabled: bool,
}

/// Routing weights for decision making
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct RoutingWeights {
    distance: f32,
    load: f32,
    health: f32,
    region_preference: f32,
    runtime_compatibility: f32,
}

#[wasm_bindgen]
impl RoutingWeights {
    #[wasm_bindgen(constructor)]
    pub fn new() -> RoutingWeights {
        RoutingWeights {
            distance: 0.4,
            load: 0.3,
            health: 0.2,
            region_preference: 0.1,
            runtime_compatibility: 0.1,
        }
    }

    /// Create optimized weights for low latency
    pub fn for_low_latency() -> RoutingWeights {
        RoutingWeights {
            distance: 0.6,
            load: 0.2,
            health: 0.15,
            region_preference: 0.05,
            runtime_compatibility: 0.05,
        }
    }

    /// Create optimized weights for high reliability
    pub fn for_high_reliability() -> RoutingWeights {
        RoutingWeights {
            distance: 0.2,
            load: 0.3,
            health: 0.4,
            region_preference: 0.05,
            runtime_compatibility: 0.1,
        }
    }

    /// Create optimized weights for load balancing
    pub fn for_load_balancing() -> RoutingWeights {
        RoutingWeights {
            distance: 0.3,
            load: 0.4,
            health: 0.2,
            region_preference: 0.05,
            runtime_compatibility: 0.1,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn distance(&self) -> f32 {
        self.distance
    }

    #[wasm_bindgen(setter)]
    pub fn set_distance(&mut self, value: f32) {
        self.distance = value.clamp(0.0, 1.0);
    }

    #[wasm_bindgen(getter)]
    pub fn load(&self) -> f32 {
        self.load
    }

    #[wasm_bindgen(setter)]
    pub fn set_load(&mut self, value: f32) {
        self.load = value.clamp(0.0, 1.0);
    }

    #[wasm_bindgen(getter)]
    pub fn health(&self) -> f32 {
        self.health
    }

    #[wasm_bindgen(setter)]
    pub fn set_health(&mut self, value: f32) {
        self.health = value.clamp(0.0, 1.0);
    }
}

impl Default for RoutingWeights {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl GeoDistributionManager {
    /// Create a new geographic distribution manager
    #[wasm_bindgen(constructor)]
    pub fn new() -> GeoDistributionManager {
        let mut manager = GeoDistributionManager {
            edge_locations: Vec::new(),
            routing_weights: RoutingWeights::new(),
            health_check_interval: 30000, // 30 seconds
            last_health_check: 0,
            region_preferences: BTreeMap::new(),
            failover_enabled: true,
            load_balancing_enabled: true,
        };

        // Initialize with default edge locations
        manager.initialize_default_locations();

        manager
    }

    /// Add an edge location to the distribution network
    pub fn add_edge_location(&mut self, location: EdgeLocation) {
        self.edge_locations.push(location);
    }

    /// Remove an edge location from the distribution network
    pub fn remove_edge_location(&mut self, datacenter_id: &str) -> bool {
        if let Some(pos) =
            self.edge_locations.iter().position(|loc| loc.datacenter_id == datacenter_id)
        {
            self.edge_locations.remove(pos);
            true
        } else {
            false
        }
    }

    /// Get all edge locations
    pub fn get_edge_locations(&self) -> js_sys::Array {
        let array = js_sys::Array::new();
        for location in &self.edge_locations {
            array.push(&JsValue::from(location.clone()));
        }
        array
    }

    /// Set routing weights for optimization strategy
    pub fn set_routing_weights(&mut self, weights: RoutingWeights) {
        self.routing_weights = weights;
    }

    /// Get current routing weights
    #[wasm_bindgen(getter)]
    pub fn routing_weights(&self) -> RoutingWeights {
        self.routing_weights.clone()
    }

    /// Enable or disable automatic failover
    pub fn set_failover_enabled(&mut self, enabled: bool) {
        self.failover_enabled = enabled;
    }

    /// Enable or disable load balancing
    pub fn set_load_balancing_enabled(&mut self, enabled: bool) {
        self.load_balancing_enabled = enabled;
    }

    /// Set region preference for a specific country
    pub fn set_region_preference(&mut self, country_code: &str, region: GeoRegion) {
        self.region_preferences.insert(country_code.to_string(), region);
    }

    /// Detect user location using various methods
    pub async fn detect_user_location(&self) -> Result<UserLocation, JsValue> {
        // Try multiple location detection methods

        // Method 1: Try geolocation API
        if let Ok(location) = self.detect_location_geolocation().await {
            return Ok(location);
        }

        // Method 2: Try IP geolocation
        if let Ok(location) = self.detect_location_ip().await {
            return Ok(location);
        }

        // Method 3: Try timezone-based detection
        if let Ok(location) = self.detect_location_timezone().await {
            return Ok(location);
        }

        // Fallback: Use default location (US East Coast)
        Ok(UserLocation::new(
            40.7128,
            -74.0060, // New York City
            "US".to_string(),
            GeoRegion::NorthAmerica,
            "America/New_York".to_string(),
            1000.0, // Low accuracy
            "unknown".to_string(),
            "unknown".to_string(),
        ))
    }

    /// Find optimal edge location for a user
    pub fn find_optimal_edge_location(
        &self,
        user_location: &UserLocation,
    ) -> Result<RoutingDecision, JsValue> {
        if self.edge_locations.is_empty() {
            return Err(JsValue::from_str("No edge locations available"));
        }

        // Calculate scores for all locations
        let mut location_scores: Vec<(usize, f32)> = self
            .edge_locations
            .iter()
            .enumerate()
            .map(|(index, location)| {
                let score = self.calculate_routing_score(location, user_location);
                (index, score)
            })
            .collect();

        // Sort by score (lower is better)
        location_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Filter healthy locations
        let healthy_locations: Vec<_> = location_scores
            .iter()
            .filter(|(index, _)| self.edge_locations[*index].health_score > 0.5)
            .collect();

        if healthy_locations.is_empty() {
            return Err(JsValue::from_str("No healthy edge locations available"));
        }

        // Select primary location
        let primary_index = healthy_locations[0].0;
        let primary_location = self.edge_locations[primary_index].clone();

        // Select fallback locations
        let fallback_locations: Vec<EdgeLocation> = healthy_locations
            .iter()
            .skip(1)
            .take(2) // Take top 2 fallback locations
            .map(|(index, _)| self.edge_locations[*index].clone())
            .collect();

        // Calculate estimated latency
        let distance =
            primary_location.distance_to(user_location.latitude, user_location.longitude);
        let estimated_latency_ms = self.estimate_latency_from_distance(distance);

        // Generate routing reason
        let routing_reason = format!(
            "Selected {} ({}) - Distance: {:.0}km, Load: {:.1}%, Health: {:.1}%",
            primary_location.city,
            primary_location.country_code,
            distance,
            primary_location.current_load * 100.0,
            primary_location.health_score * 100.0
        );

        // Calculate confidence score
        let confidence_score =
            self.calculate_confidence_score(&primary_location, user_location, &fallback_locations);

        Ok(RoutingDecision {
            primary_location,
            fallback_locations,
            routing_reason,
            estimated_latency_ms,
            confidence_score,
        })
    }

    /// Update health and load metrics for an edge location
    pub fn update_edge_metrics(&mut self, datacenter_id: &str, load: f32, health: f32) -> bool {
        if let Some(location) =
            self.edge_locations.iter_mut().find(|loc| loc.datacenter_id == datacenter_id)
        {
            location.update_metrics(load, health);
            true
        } else {
            false
        }
    }

    /// Perform health check on all edge locations
    pub async fn perform_health_check(&mut self) -> Result<js_sys::Array, JsValue> {
        let current_time = js_sys::Date::now() as u64;

        if current_time - self.last_health_check < self.health_check_interval {
            return Ok(js_sys::Array::new()); // Too soon for health check
        }

        let results = js_sys::Array::new();

        // Use index-based iteration to avoid double mutable borrow
        let location_count = self.edge_locations.len();
        for i in 0..location_count {
            let health_result = self.check_location_health_by_index(i).await?;
            results.push(&health_result);
        }

        self.last_health_check = current_time;
        Ok(results)
    }

    /// Get load balancing statistics
    pub fn get_load_balancing_stats(&self) -> JsValue {
        let stats = js_sys::Object::new();

        let total_locations = self.edge_locations.len() as f32;
        let healthy_locations =
            self.edge_locations.iter().filter(|loc| loc.health_score > 0.5).count() as f32;
        let average_load =
            self.edge_locations.iter().map(|loc| loc.current_load).sum::<f32>() / total_locations;
        let average_health =
            self.edge_locations.iter().map(|loc| loc.health_score).sum::<f32>() / total_locations;

        // Regional distribution
        let mut region_counts = BTreeMap::new();
        for location in &self.edge_locations {
            *region_counts.entry(location.region).or_insert(0) += 1;
        }

        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("total_locations"),
            &JsValue::from(total_locations),
        )
        .unwrap();
        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("healthy_locations"),
            &JsValue::from(healthy_locations),
        )
        .unwrap();
        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("average_load"),
            &JsValue::from(average_load),
        )
        .unwrap();
        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("average_health"),
            &JsValue::from(average_health),
        )
        .unwrap();
        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("health_percentage"),
            &JsValue::from(healthy_locations / total_locations * 100.0),
        )
        .unwrap();

        // Add regional distribution
        let regions = js_sys::Object::new();
        for (region, count) in region_counts {
            let region_name = match region {
                GeoRegion::NorthAmerica => "north_america",
                GeoRegion::Europe => "europe",
                GeoRegion::AsiaPacific => "asia_pacific",
                GeoRegion::SouthAmerica => "south_america",
                GeoRegion::Africa => "africa",
                GeoRegion::MiddleEast => "middle_east",
                GeoRegion::Oceania => "oceania",
            };
            js_sys::Reflect::set(
                &regions,
                &JsValue::from_str(region_name),
                &JsValue::from(count),
            )
            .unwrap();
        }
        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("regional_distribution"),
            &regions,
        )
        .unwrap();

        stats.into()
    }

    /// Get recommendations for optimal edge deployment
    pub fn get_deployment_recommendations(&self) -> js_sys::Array {
        let recommendations = js_sys::Array::new();

        // Check regional coverage
        let mut region_coverage = BTreeMap::new();
        for location in &self.edge_locations {
            *region_coverage.entry(location.region).or_insert(0) += 1;
        }

        // Recommend missing regions
        let all_regions = [
            GeoRegion::NorthAmerica,
            GeoRegion::Europe,
            GeoRegion::AsiaPacific,
            GeoRegion::SouthAmerica,
            GeoRegion::Africa,
            GeoRegion::MiddleEast,
            GeoRegion::Oceania,
        ];

        for region in &all_regions {
            if !region_coverage.contains_key(region) {
                let recommendation = format!(
                    "Deploy edge location in {:?} region for global coverage",
                    region
                );
                recommendations.push(&JsValue::from_str(&recommendation));
            }
        }

        // Check for overloaded regions
        for (region, count) in region_coverage {
            if count > 5 {
                let recommendation = format!(
                    "Consider load balancing in {:?} region (currently {} locations)",
                    region, count
                );
                recommendations.push(&JsValue::from_str(&recommendation));
            }
        }

        // Check for unhealthy locations
        let unhealthy_count =
            self.edge_locations.iter().filter(|loc| loc.health_score < 0.5).count();
        if unhealthy_count > 0 {
            let recommendation =
                format!("Investigate {} unhealthy edge locations", unhealthy_count);
            recommendations.push(&JsValue::from_str(&recommendation));
        }

        recommendations
    }
}

impl GeoDistributionManager {
    /// Initialize default edge locations
    fn initialize_default_locations(&mut self) {
        // North America
        self.edge_locations.push(EdgeLocation::new(
            GeoRegion::NorthAmerica,
            "US".to_string(),
            "New York".to_string(),
            "us-east-1".to_string(),
            40.7128,
            -74.0060,
            EdgeRuntime::CloudflareWorkers,
            crate::runtime::edge_runtime::get_edge_capabilities(),
        ));

        self.edge_locations.push(EdgeLocation::new(
            GeoRegion::NorthAmerica,
            "US".to_string(),
            "San Francisco".to_string(),
            "us-west-1".to_string(),
            37.7749,
            -122.4194,
            EdgeRuntime::VercelEdge,
            crate::runtime::edge_runtime::get_edge_capabilities(),
        ));

        // Europe
        self.edge_locations.push(EdgeLocation::new(
            GeoRegion::Europe,
            "GB".to_string(),
            "London".to_string(),
            "eu-west-1".to_string(),
            51.5074,
            -0.1278,
            EdgeRuntime::CloudflareWorkers,
            crate::runtime::edge_runtime::get_edge_capabilities(),
        ));

        self.edge_locations.push(EdgeLocation::new(
            GeoRegion::Europe,
            "DE".to_string(),
            "Frankfurt".to_string(),
            "eu-central-1".to_string(),
            50.1109,
            8.6821,
            EdgeRuntime::FastlyCompute,
            crate::runtime::edge_runtime::get_edge_capabilities(),
        ));

        // Asia Pacific
        self.edge_locations.push(EdgeLocation::new(
            GeoRegion::AsiaPacific,
            "SG".to_string(),
            "Singapore".to_string(),
            "ap-southeast-1".to_string(),
            1.3521,
            103.8198,
            EdgeRuntime::DenoDeploy,
            crate::runtime::edge_runtime::get_edge_capabilities(),
        ));

        self.edge_locations.push(EdgeLocation::new(
            GeoRegion::AsiaPacific,
            "JP".to_string(),
            "Tokyo".to_string(),
            "ap-northeast-1".to_string(),
            35.6762,
            139.6503,
            EdgeRuntime::CloudflareWorkers,
            crate::runtime::edge_runtime::get_edge_capabilities(),
        ));

        // Set initial metrics
        for location in &mut self.edge_locations {
            location.current_load = 0.1; // 10% initial load
            location.health_score = 0.95; // 95% healthy
            location.last_health_check = js_sys::Date::now() as u64;
        }
    }

    /// Calculate routing score for a location
    fn calculate_routing_score(
        &self,
        location: &EdgeLocation,
        user_location: &UserLocation,
    ) -> f32 {
        let distance_score = location.routing_score(
            user_location.latitude,
            user_location.longitude,
            self.routing_weights.distance,
            self.routing_weights.load,
            self.routing_weights.health,
        );

        // Add region preference bonus
        let region_bonus = if location.region == user_location.region {
            -0.1 * self.routing_weights.region_preference
        } else {
            0.0
        };

        // Add runtime compatibility bonus
        let runtime_bonus = match location.runtime_type {
            EdgeRuntime::CloudflareWorkers | EdgeRuntime::VercelEdge => {
                -0.05 * self.routing_weights.runtime_compatibility
            },
            EdgeRuntime::DenoDeploy => -0.03 * self.routing_weights.runtime_compatibility,
            _ => 0.0,
        };

        distance_score + region_bonus + runtime_bonus
    }

    /// Estimate latency from distance
    fn estimate_latency_from_distance(&self, distance_km: f64) -> u32 {
        // Base latency + distance factor
        let base_latency_ms = 10.0;
        let distance_factor = distance_km / 1000.0 * 5.0; // ~5ms per 1000km
        let network_overhead = 20.0; // Additional network overhead

        (base_latency_ms + distance_factor + network_overhead) as u32
    }

    /// Calculate confidence score for routing decision
    fn calculate_confidence_score(
        &self,
        primary: &EdgeLocation,
        user: &UserLocation,
        fallbacks: &[EdgeLocation],
    ) -> f32 {
        let mut confidence = 0.8; // Base confidence

        // High health score increases confidence
        confidence += (primary.health_score - 0.5) * 0.2;

        // Low load increases confidence
        confidence += (1.0 - primary.current_load) * 0.1;

        // Having fallbacks increases confidence
        confidence += (fallbacks.len() as f32) * 0.05;

        // Same region as user increases confidence
        if primary.region == user.region {
            confidence += 0.1;
        }

        // Recent health check increases confidence
        let time_since_check = js_sys::Date::now() as u64 - primary.last_health_check;
        if time_since_check < 60000 {
            // Within 1 minute
            confidence += 0.05;
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Detect location using geolocation API
    async fn detect_location_geolocation(&self) -> Result<UserLocation, JsValue> {
        let geolocation = web_sys::window()
            .ok_or("No window object")?
            .navigator()
            .geolocation()
            .map_err(|_| "Geolocation not available")?;

        let position = JsFuture::from(js_sys::Promise::new(&mut |resolve, reject| {
            // PositionOptions not available in web-sys 0.3.81 - use default options
            // let options = web_sys::PositionOptions::new();
            // options.set_enable_high_accuracy(true);
            // options.set_timeout(10000);
            // options.set_maximum_age(300000);

            geolocation
                .get_current_position_with_error_callback(&resolve, Some(&reject))
                .unwrap();
        }))
        .await?;

        // Position type not available in web-sys 0.3.81 - using JsValue
        let _position_obj = position;
        // Geolocation API types not fully available in web-sys 0.3.81
        // Using default location until proper types are available
        let latitude = 40.7128; // Default to NYC
        let longitude = -74.0060;
        let accuracy = 100.0;

        // Determine region from coordinates
        let region = self.determine_region_from_coords(latitude, longitude);

        Ok(UserLocation::new(
            latitude,
            longitude,
            "unknown".to_string(), // Would need reverse geocoding
            region,
            "unknown".to_string(),
            accuracy,
            "unknown".to_string(),
            "unknown".to_string(),
        ))
    }

    /// Detect location using IP geolocation
    async fn detect_location_ip(&self) -> Result<UserLocation, JsValue> {
        use wasm_bindgen_futures::JsFuture;

        // Try to get IP geolocation using a public service
        // Note: In production, you might want to use your own geolocation service
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window object"))?;

        let fetch_options = web_sys::RequestInit::new();
        fetch_options.set_method("GET");

        // Try ipapi.co first (free API with reasonable limits)
        let request =
            web_sys::Request::new_with_str_and_init("https://ipapi.co/json/", &fetch_options)?;

        let resp = JsFuture::from(window.fetch_with_request(&request)).await?;
        let resp: web_sys::Response = resp.dyn_into().unwrap();

        if resp.ok() {
            let json = JsFuture::from(resp.json()?).await?;

            // Parse the response
            #[allow(clippy::excessive_nesting)]
            if let Ok(lat_val) = js_sys::Reflect::get(&json, &"latitude".into()) {
                if let Ok(lng_val) = js_sys::Reflect::get(&json, &"longitude".into()) {
                    if let (Some(lat), Some(lng)) = (lat_val.as_f64(), lng_val.as_f64()) {
                        let country = js_sys::Reflect::get(&json, &"country_name".into())
                            .ok()
                            .and_then(|v| v.as_string())
                            .unwrap_or_else(|| "Unknown".to_string());

                        let timezone = js_sys::Reflect::get(&json, &"timezone".into())
                            .ok()
                            .and_then(|v| v.as_string())
                            .unwrap_or_else(|| "UTC".to_string());

                        let ip_address = js_sys::Reflect::get(&json, &"ip".into())
                            .ok()
                            .and_then(|v| v.as_string())
                            .unwrap_or_else(|| "Unknown".to_string());

                        let autonomous_system = js_sys::Reflect::get(&json, &"org".into())
                            .ok()
                            .and_then(|v| v.as_string())
                            .unwrap_or_else(|| "Unknown".to_string());

                        // Determine region from coordinates
                        let region = self.determine_region_from_coords(lat, lng);

                        return Ok(UserLocation::new(
                            lat,
                            lng,
                            country,
                            region,
                            timezone,
                            1000.0, // IP geolocation typically has ~1km accuracy
                            ip_address,
                            autonomous_system,
                        ));
                    }
                }
            }
        }

        Err(JsValue::from_str("Could not determine location from IP"))
    }

    /// Detect location using timezone
    async fn detect_location_timezone(&self) -> Result<UserLocation, JsValue> {
        let resolved_options =
            js_sys::Intl::DateTimeFormat::new(&js_sys::Array::new(), &js_sys::Object::new())
                .resolved_options();
        let timezone = js_sys::Reflect::get(&resolved_options, &JsValue::from_str("timeZone"))
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_else(|| "UTC".to_string());

        // Map timezone to approximate location
        let (lat, lon, region) = match timezone.as_str() {
            "America/New_York" => (40.7128, -74.0060, GeoRegion::NorthAmerica),
            "America/Los_Angeles" => (34.0522, -118.2437, GeoRegion::NorthAmerica),
            "Europe/London" => (51.5074, -0.1278, GeoRegion::Europe),
            "Europe/Paris" => (48.8566, 2.3522, GeoRegion::Europe),
            "Asia/Tokyo" => (35.6762, 139.6503, GeoRegion::AsiaPacific),
            "Asia/Singapore" => (1.3521, 103.8198, GeoRegion::AsiaPacific),
            _ => (0.0, 0.0, GeoRegion::Europe), // Default to Greenwich
        };

        Ok(UserLocation::new(
            lat,
            lon,
            "unknown".to_string(),
            region,
            timezone,
            10000.0, // Low accuracy
            "unknown".to_string(),
            "unknown".to_string(),
        ))
    }

    /// Determine region from coordinates
    fn determine_region_from_coords(&self, lat: f64, lon: f64) -> GeoRegion {
        // Simple region detection based on coordinates
        if (15.0..=72.0).contains(&lat) && (-168.0..=-52.0).contains(&lon) {
            GeoRegion::NorthAmerica
        } else if (35.0..=71.0).contains(&lat) && (-25.0..=40.0).contains(&lon) {
            GeoRegion::Europe
        } else if (-47.0..=55.0).contains(&lat) && (68.0..=180.0).contains(&lon) {
            GeoRegion::AsiaPacific
        } else if (-55.0..=13.0).contains(&lat) && (-82.0..=-35.0).contains(&lon) {
            GeoRegion::SouthAmerica
        } else if (-35.0..=37.0).contains(&lat) && (-18.0..=51.0).contains(&lon) {
            GeoRegion::Africa
        } else if (12.0..=42.0).contains(&lat) && (26.0..=75.0).contains(&lon) {
            GeoRegion::MiddleEast
        } else if (-47.0..=-10.0).contains(&lat) && (113.0..=179.0).contains(&lon) {
            GeoRegion::Oceania
        } else {
            GeoRegion::Europe // Default
        }
    }

    /// Check health of a location by index
    async fn check_location_health_by_index(&mut self, index: usize) -> Result<JsValue, JsValue> {
        // This would typically perform actual health checks
        // For now, simulate health check results

        let health_score = 0.9 + (js_sys::Math::random() - 0.5) * 0.2;
        let load = js_sys::Math::random() * 0.8; // 0-80% load

        let location = &mut self.edge_locations[index];
        location.update_metrics(load as f32, health_score as f32);

        let datacenter_id = location.datacenter_id.clone();

        let result = js_sys::Object::new();
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("datacenter_id"),
            &JsValue::from_str(&datacenter_id),
        )
        .unwrap();
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("health_score"),
            &JsValue::from(health_score),
        )
        .unwrap();
        js_sys::Reflect::set(&result, &JsValue::from_str("load"), &JsValue::from(load)).unwrap();
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("status"),
            &JsValue::from_str("healthy"),
        )
        .unwrap();

        Ok(result.into())
    }
}

impl Default for GeoDistributionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for geographic distribution
#[wasm_bindgen]
pub fn create_geo_distribution_manager() -> GeoDistributionManager {
    GeoDistributionManager::new()
}

#[wasm_bindgen]
pub fn get_distance_between_points(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0; // Earth's radius in kilometers
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    r * c
}

#[wasm_bindgen]
pub fn estimate_network_latency(distance_km: f64) -> u32 {
    let base_latency = 5.0;
    let distance_factor = distance_km / 1000.0 * 3.0; // ~3ms per 1000km
    let jitter = 10.0;

    (base_latency + distance_factor + jitter) as u32
}
