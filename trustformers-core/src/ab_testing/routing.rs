//! Traffic routing and splitting strategies

use super::experiment::ExperimentStatus;
use super::{Experiment, Variant};
use anyhow::Result;
use scirs2_core::random::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Traffic routing strategy
#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    /// Random assignment based on hash
    HashBased,
    /// Round-robin assignment
    RoundRobin,
    /// User segment based routing
    SegmentBased(Vec<UserSegment>),
    /// Weighted random assignment
    WeightedRandom(HashMap<String, f64>),
    /// Sticky sessions (user always gets same variant)
    Sticky,
}

/// User segment for targeted routing
#[derive(Debug, Clone)]
pub struct UserSegment {
    /// Segment name
    pub name: String,
    /// Condition to match users
    pub condition: SegmentCondition,
    /// Variant to assign to this segment
    pub variant_name: String,
}

/// Conditions for segment matching
#[derive(Debug, Clone)]
pub enum SegmentCondition {
    /// User ID matches pattern
    UserIdPattern(String),
    /// User has specific attribute
    HasAttribute(String, String),
    /// User is in specific geographic region
    GeoRegion(String),
    /// User is using specific platform
    Platform(Platform),
    /// Custom condition function
    Custom(String), // In practice, this would be a function pointer
}

/// Platform types
#[derive(Debug, Clone, PartialEq)]
pub enum Platform {
    Ios,
    Android,
    Web,
    Desktop,
}

/// Traffic splitter implementation
pub struct TrafficSplitter {
    /// Default routing strategy
    default_strategy: RoutingStrategy,
    /// Cache for sticky sessions
    sticky_cache: parking_lot::RwLock<HashMap<String, String>>,
    /// Round-robin counters
    round_robin_counters: parking_lot::RwLock<HashMap<String, usize>>,
}

impl Default for TrafficSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl TrafficSplitter {
    /// Create a new traffic splitter
    pub fn new() -> Self {
        Self {
            default_strategy: RoutingStrategy::HashBased,
            sticky_cache: parking_lot::RwLock::new(HashMap::new()),
            round_robin_counters: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Create with specific strategy
    pub fn with_strategy(strategy: RoutingStrategy) -> Self {
        Self {
            default_strategy: strategy,
            sticky_cache: parking_lot::RwLock::new(HashMap::new()),
            round_robin_counters: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Route a user to a variant
    pub fn route(&self, experiment: &Experiment, user_id: &str) -> Result<Variant> {
        // Check if experiment is running
        if experiment.status() != ExperimentStatus::Running {
            return Ok(experiment.config().control_variant.clone());
        }

        // Route based on strategy (traffic percentage is handled within the strategy)
        match &self.default_strategy {
            RoutingStrategy::HashBased => self.route_hash_based(experiment, user_id),
            RoutingStrategy::RoundRobin => self.route_round_robin(experiment),
            RoutingStrategy::SegmentBased(segments) => {
                self.route_segment_based(experiment, user_id, segments)
            },
            RoutingStrategy::WeightedRandom(weights) => {
                self.route_weighted_random(experiment, weights)
            },
            RoutingStrategy::Sticky => self.route_sticky(experiment, user_id),
        }
    }

    /// Check if user should be included in experiment
    #[allow(dead_code)]
    fn should_include_in_experiment(&self, experiment: &Experiment, user_id: &str) -> bool {
        let hash = self.hash_user_id(user_id, &experiment.id().to_string());
        let threshold = (experiment.config().traffic_percentage / 100.0 * u64::MAX as f64) as u64;
        hash < threshold
    }

    /// Hash-based routing
    fn route_hash_based(&self, experiment: &Experiment, user_id: &str) -> Result<Variant> {
        let hash = self.hash_user_id(user_id, &experiment.id().to_string());
        let control_variant = &experiment.config().control_variant;
        let treatment_variants = &experiment.config().treatment_variants;

        if treatment_variants.is_empty() {
            return Ok(control_variant.clone());
        }

        // Use traffic percentage to decide between control and treatment
        let traffic_percentage = experiment.config().traffic_percentage;
        let threshold = (traffic_percentage / 100.0 * u64::MAX as f64) as u64;

        if hash < threshold {
            // User goes to treatment - pick randomly among treatment variants
            let treatment_index = (hash as usize) % treatment_variants.len();
            Ok(treatment_variants[treatment_index].clone())
        } else {
            // User goes to control
            Ok(control_variant.clone())
        }
    }

    /// Round-robin routing
    fn route_round_robin(&self, experiment: &Experiment) -> Result<Variant> {
        let variants = experiment.all_variants();
        let mut counters = self.round_robin_counters.write();
        let counter = counters.entry(experiment.id().to_string()).or_insert(0);
        let index = *counter % variants.len();
        *counter += 1;
        Ok(variants[index].clone())
    }

    /// Segment-based routing
    fn route_segment_based(
        &self,
        experiment: &Experiment,
        user_id: &str,
        segments: &[UserSegment],
    ) -> Result<Variant> {
        // Check if user matches any segment
        for segment in segments {
            if !self.matches_segment(user_id, &segment.condition) {
                continue;
            }

            // Find the variant by name
            for variant in experiment.all_variants() {
                if variant.name() == segment.variant_name {
                    return Ok(variant.clone());
                }
            }
        }

        // Fall back to hash-based routing
        self.route_hash_based(experiment, user_id)
    }

    /// Weighted random routing
    fn route_weighted_random(
        &self,
        experiment: &Experiment,
        weights: &HashMap<String, f64>,
    ) -> Result<Variant> {
        let variants = experiment.all_variants();
        let total_weight: f64 = weights.values().sum();

        if total_weight == 0.0 {
            // Fall back to equal weights
            return self.route_hash_based(experiment, &uuid::Uuid::new_v4().to_string());
        }

        let mut rng = thread_rng();
        let random_value: f64 = rng.gen_range(0.0..total_weight);
        let mut cumulative_weight = 0.0;

        for variant in variants {
            let weight = weights.get(variant.name()).unwrap_or(&1.0);
            cumulative_weight += weight;
            if random_value < cumulative_weight {
                return Ok(variant.clone());
            }
        }

        // Fallback to control
        Ok(experiment.config().control_variant.clone())
    }

    /// Sticky session routing
    fn route_sticky(&self, experiment: &Experiment, user_id: &str) -> Result<Variant> {
        let cache_key = format!("{}:{}", experiment.id(), user_id);

        // Check cache
        if let Some(cached_variant) = self.get_cached_variant(experiment, &cache_key) {
            return Ok(cached_variant);
        }

        // Not in cache, route and store
        let variant = self.route_hash_based(experiment, user_id)?;
        {
            let mut cache = self.sticky_cache.write();
            cache.insert(cache_key, variant.name().to_string());
        }

        Ok(variant)
    }

    /// Get cached variant if it exists
    fn get_cached_variant(&self, experiment: &Experiment, cache_key: &str) -> Option<Variant> {
        let cache = self.sticky_cache.read();
        let variant_name = cache.get(cache_key)?;

        for variant in experiment.all_variants() {
            if variant.name() == variant_name {
                return Some(variant.clone());
            }
        }
        None
    }

    /// Check if user matches segment condition
    fn matches_segment(&self, user_id: &str, condition: &SegmentCondition) -> bool {
        match condition {
            SegmentCondition::UserIdPattern(pattern) => {
                // Simple pattern matching (in practice, use regex)
                user_id.contains(pattern)
            },
            SegmentCondition::HasAttribute(_, _) => {
                // Would require user attribute lookup
                false
            },
            SegmentCondition::GeoRegion(_) => {
                // Would require geo lookup
                false
            },
            SegmentCondition::Platform(_) => {
                // Would require platform detection
                false
            },
            SegmentCondition::Custom(_) => {
                // Would execute custom function
                false
            },
        }
    }

    /// Hash user ID with experiment ID for consistent assignment
    fn hash_user_id(&self, user_id: &str, experiment_id: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        user_id.hash(&mut hasher);
        experiment_id.hash(&mut hasher);
        hasher.finish()
    }

    /// Clear sticky cache for an experiment
    pub fn clear_sticky_cache(&self, experiment_id: &str) {
        let mut cache = self.sticky_cache.write();
        cache.retain(|k, _| !k.starts_with(&format!("{}:", experiment_id)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ab_testing::ExperimentConfig;

    fn create_test_experiment() -> Experiment {
        let config = ExperimentConfig {
            name: "Test".to_string(),
            description: "Test".to_string(),
            control_variant: Variant::new("control", "v1"),
            treatment_variants: vec![Variant::new("treatment", "v2")],
            traffic_percentage: 100.0,
            min_sample_size: 100,
            max_duration_hours: 24,
        };
        let mut exp = Experiment::new(config).unwrap();
        exp.start().unwrap();
        exp
    }

    #[test]
    fn test_hash_based_routing() {
        let splitter = TrafficSplitter::new();
        let experiment = create_test_experiment();

        // Same user should always get same variant
        let user_id = "test-user-123";
        let variant1 = splitter.route(&experiment, user_id).unwrap();
        let variant2 = splitter.route(&experiment, user_id).unwrap();
        assert_eq!(variant1, variant2);
    }

    #[test]
    fn test_round_robin_routing() {
        let splitter = TrafficSplitter::with_strategy(RoutingStrategy::RoundRobin);
        let experiment = create_test_experiment();

        let mut control_count = 0;
        let mut treatment_count = 0;

        // Should alternate between variants
        for _ in 0..10 {
            let variant = splitter.route(&experiment, "any-user").unwrap();
            match variant.name() {
                "control" => control_count += 1,
                "treatment" => treatment_count += 1,
                name => panic!("Unexpected variant name: {}", name),
            }
        }

        assert_eq!(control_count, 5);
        assert_eq!(treatment_count, 5);
    }

    #[test]
    fn test_sticky_routing() {
        let splitter = TrafficSplitter::with_strategy(RoutingStrategy::Sticky);
        let experiment = create_test_experiment();

        let user_id = "sticky-user";
        let first_variant = splitter.route(&experiment, user_id).unwrap();

        // Multiple calls should return same variant
        for _ in 0..10 {
            let variant = splitter.route(&experiment, user_id).unwrap();
            assert_eq!(variant, first_variant);
        }
    }

    #[test]
    fn test_traffic_percentage() {
        let config = ExperimentConfig {
            name: "Test".to_string(),
            description: "Test".to_string(),
            control_variant: Variant::new("control", "v1"),
            treatment_variants: vec![Variant::new("treatment", "v2")],
            traffic_percentage: 10.0, // Only 10% of users
            min_sample_size: 100,
            max_duration_hours: 24,
        };
        let mut exp = Experiment::new(config).unwrap();
        exp.start().unwrap();

        let splitter = TrafficSplitter::new();
        let mut included_count = 0;

        // Test with many users
        for i in 0..1000 {
            let user_id = format!("user-{}", i);
            let variant = splitter.route(&exp, &user_id).unwrap();

            // If included in experiment, might get treatment
            if variant.name() != "control" || splitter.should_include_in_experiment(&exp, &user_id)
            {
                included_count += 1;
            }
        }

        // Should be roughly 10% (allow some variance)
        let inclusion_rate = included_count as f64 / 1000.0;
        assert!((inclusion_rate - 0.1).abs() < 0.05);
    }
}
