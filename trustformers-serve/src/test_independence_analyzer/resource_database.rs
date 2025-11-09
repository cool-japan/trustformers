//! Resource Usage Database Module
//!
//! This module provides comprehensive resource usage tracking and management
//! for the test independence analysis system. It maintains detailed records
//! of resource consumption patterns, allocation events, and performance
//! characteristics to enable intelligent conflict detection and optimization.

use crate::test_independence_analyzer::types::*;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};
use tracing::{debug, info};

/// Comprehensive resource usage database with advanced tracking capabilities
#[derive(Debug)]
pub struct ResourceUsageDatabase {
    /// Resource usage records indexed by test ID
    usage_records: RwLock<HashMap<String, Vec<ResourceUsageRecord>>>,

    /// Resource type definitions and specifications
    resource_types: RwLock<HashMap<String, ResourceTypeDefinition>>,

    /// Historical allocation events for analysis
    allocation_history: RwLock<Vec<ResourceAllocationEvent>>,

    /// Database performance and usage statistics
    statistics: RwLock<DatabaseStatistics>,

    /// Configuration for database behavior
    config: RwLock<DatabaseConfig>,

    /// Resource usage aggregations for optimization
    aggregations: RwLock<ResourceUsageAggregations>,

    /// Resource performance baselines
    performance_baselines: RwLock<HashMap<String, PerformanceBaseline>>,
}

/// Database configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Maximum records to keep per test
    pub max_records_per_test: usize,

    /// History retention period
    pub retention_period: Duration,

    /// Enable automatic cleanup of old records
    pub auto_cleanup: bool,

    /// Cleanup interval
    pub cleanup_interval: Duration,

    /// Enable performance tracking
    pub performance_tracking: bool,

    /// Enable detailed logging
    pub detailed_logging: bool,

    /// Maximum allocation events to keep
    pub max_allocation_events: usize,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            max_records_per_test: 1000,
            retention_period: Duration::from_secs(7 * 24 * 3600), // 7 days
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(3600), // 1 hour
            performance_tracking: true,
            detailed_logging: false,
            max_allocation_events: 10000,
        }
    }
}

/// Resource usage record with comprehensive tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageRecord {
    /// Unique record identifier
    pub id: String,

    /// Test identifier that used the resource
    pub test_id: String,

    /// Resource type identifier
    pub resource_type: String,

    /// Specific resource instance identifier
    pub resource_id: String,

    /// Usage start timestamp
    pub start_time: DateTime<Utc>,

    /// Usage duration
    pub duration: Duration,

    /// Resource usage amount (normalized 0.0-1.0)
    pub usage_amount: f64,

    /// Usage efficiency score (0.0-1.0)
    pub efficiency: f32,

    /// Number of concurrent users during this usage
    pub concurrent_users: usize,

    /// Peak usage during the session
    pub peak_usage: f64,

    /// Average usage during the session
    pub average_usage: f64,

    /// Usage variance/stability
    pub usage_variance: f64,

    /// Associated performance metrics
    pub performance_metrics: UsagePerformanceMetrics,

    /// Usage tags for categorization
    pub tags: Vec<String>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Performance metrics during resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePerformanceMetrics {
    /// Throughput achieved during usage
    pub throughput: f64,

    /// Average latency experienced
    pub average_latency: Duration,

    /// Peak latency experienced
    pub peak_latency: Duration,

    /// Error rate during usage
    pub error_rate: f32,

    /// Resource contention experienced
    pub contention_score: f32,

    /// Quality of service score
    pub qos_score: f32,
}

impl Default for UsagePerformanceMetrics {
    fn default() -> Self {
        Self {
            throughput: 0.0,
            average_latency: Duration::from_millis(0),
            peak_latency: Duration::from_millis(0),
            error_rate: 0.0,
            contention_score: 0.0,
            qos_score: 1.0,
        }
    }
}

/// Resource type definition with comprehensive specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTypeDefinition {
    /// Resource type identifier
    pub type_id: String,

    /// Human-readable name
    pub name: String,

    /// Detailed description
    pub description: String,

    /// Resource category
    pub category: ResourceCategory,

    /// Maximum concurrent users supported
    pub max_concurrent_users: Option<usize>,

    /// Resource sharing capabilities
    pub sharing_capabilities: ResourceSharingSpec,

    /// Conflict detection and resolution rules
    pub conflict_rules: Vec<ConflictRule>,

    /// Performance characteristics and limits
    pub performance_characteristics: PerformanceCharacteristics,

    /// Resource scaling properties
    pub scaling_properties: ResourceScalingProperties,

    /// Monitoring and alerting configuration
    pub monitoring_config: ResourceMonitoringConfig,
}

/// Resource categories for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceCategory {
    /// Computational resources (CPU, GPU)
    Computational,

    /// Memory resources (RAM, cache)
    Memory,

    /// Storage resources (disk, database)
    Storage,

    /// Network resources (ports, bandwidth)
    Network,

    /// System resources (processes, handles)
    System,

    /// Custom resource category
    Custom(String),
}

/// Resource scaling properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceScalingProperties {
    /// Can the resource be scaled horizontally
    pub horizontal_scaling: bool,

    /// Can the resource be scaled vertically
    pub vertical_scaling: bool,

    /// Scaling overhead factor
    pub scaling_overhead: f32,

    /// Minimum scale units
    pub min_scale_units: usize,

    /// Maximum scale units
    pub max_scale_units: Option<usize>,

    /// Scaling response time
    pub scaling_response_time: Duration,
}

/// Resource monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoringConfig {
    /// Enable usage monitoring
    pub monitor_usage: bool,

    /// Enable performance monitoring
    pub monitor_performance: bool,

    /// Monitoring sampling interval
    pub sampling_interval: Duration,

    /// Usage threshold for alerts
    pub usage_threshold: f32,

    /// Performance threshold for alerts
    pub performance_threshold: f32,

    /// Enable predictive analytics
    pub predictive_analytics: bool,
}

/// Resource allocation event with comprehensive tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocationEvent {
    /// Event unique identifier
    pub event_id: String,

    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Type of allocation event
    pub event_type: AllocationEventType,

    /// Test that triggered the event
    pub test_id: String,

    /// Resource type involved
    pub resource_type: String,

    /// Specific resource instance
    pub resource_id: String,

    /// Allocation request details
    pub allocation_details: AllocationDetails,

    /// Event outcome
    pub outcome: AllocationOutcome,

    /// Event duration (for allocation/deallocation)
    pub event_duration: Option<Duration>,

    /// Associated metrics at time of event
    pub event_metrics: AllocationEventMetrics,
}

/// Types of resource allocation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationEventType {
    /// Resource allocation requested
    AllocationRequested,

    /// Resource successfully allocated
    Allocated,

    /// Resource deallocated/released
    Deallocated,

    /// Allocation request failed
    AllocationFailed,

    /// Resource reallocation/migration
    Reallocated,

    /// Resource contention detected
    ContentionDetected,

    /// Resource conflict identified
    ConflictDetected,

    /// Resource performance degradation
    PerformanceDegradation,

    /// Resource threshold exceeded
    ThresholdExceeded,
}

/// Allocation request outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationOutcome {
    /// Request was successful
    Success,

    /// Request was denied due to lack of resources
    Denied { reason: String },

    /// Request was queued for later processing
    Queued { estimated_wait: Duration },

    /// Request failed due to error
    Failed { error: String },

    /// Request was partially fulfilled
    Partial { fulfilled_amount: f64 },
}

/// Allocation details with comprehensive information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationDetails {
    /// Amount of resource requested/allocated
    pub requested_amount: f64,

    /// Amount actually allocated
    pub allocated_amount: f64,

    /// Expected usage duration
    pub expected_duration: Option<Duration>,

    /// Actual usage duration
    pub actual_duration: Option<Duration>,

    /// Allocation priority level
    pub priority: f32,

    /// Quality of service requirements
    pub qos_requirements: QosRequirements,

    /// Allocation preferences
    pub preferences: AllocationPreferences,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Quality of service requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QosRequirements {
    /// Minimum acceptable performance level
    pub min_performance: f32,

    /// Maximum acceptable latency
    pub max_latency: Duration,

    /// Required availability level
    pub required_availability: f32,

    /// Maximum acceptable error rate
    pub max_error_rate: f32,
}

/// Allocation preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPreferences {
    /// Preferred resource instances
    pub preferred_instances: Vec<String>,

    /// Resource locality preferences
    pub locality_preferences: Vec<LocalityPreference>,

    /// Sharing tolerance level
    pub sharing_tolerance: SharingTolerance,

    /// Performance vs cost preference
    pub performance_cost_preference: f32, // 0.0 = cost-optimized, 1.0 = performance-optimized
}

/// Locality preferences for resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalityPreference {
    /// Locality type (e.g., "datacenter", "rack", "node")
    pub locality_type: String,

    /// Preferred locality value
    pub preferred_value: String,

    /// Preference strength (0.0-1.0)
    pub strength: f32,
}

/// Sharing tolerance levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SharingTolerance {
    /// Exclusive access required
    Exclusive,

    /// Limited sharing acceptable
    Limited { max_concurrent: usize },

    /// Moderate sharing acceptable
    Moderate,

    /// High sharing tolerance
    High,

    /// Any sharing level acceptable
    Any,
}

/// Metrics captured at allocation event time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEventMetrics {
    /// System resource utilization at event time
    pub system_utilization: f32,

    /// Resource pool status
    pub pool_status: ResourcePoolStatus,

    /// Current allocation queue length
    pub queue_length: usize,

    /// Average allocation time recently
    pub recent_average_allocation_time: Duration,

    /// Number of active allocations
    pub active_allocations: usize,
}

/// Resource pool status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePoolStatus {
    /// Total resources in pool
    pub total_capacity: f64,

    /// Currently allocated resources
    pub allocated_capacity: f64,

    /// Available resources
    pub available_capacity: f64,

    /// Reserved resources
    pub reserved_capacity: f64,

    /// Pool utilization percentage
    pub utilization_percentage: f32,
}

/// Resource usage aggregations for analysis and optimization
#[derive(Debug, Default)]
pub struct ResourceUsageAggregations {
    /// Usage by resource type
    pub usage_by_type: HashMap<String, ResourceTypeUsageStats>,

    /// Usage by test
    pub usage_by_test: HashMap<String, TestResourceUsageStats>,

    /// Usage patterns over time
    pub temporal_patterns: TemporalUsagePatterns,

    /// Resource efficiency metrics
    pub efficiency_metrics: ResourceEfficiencyMetrics,
}

/// Usage statistics for a resource type
#[derive(Debug, Clone)]
pub struct ResourceTypeUsageStats {
    /// Total usage time
    pub total_usage_time: Duration,

    /// Average usage per session
    pub average_usage_per_session: f64,

    /// Peak concurrent usage
    pub peak_concurrent_usage: usize,

    /// Usage efficiency distribution
    pub efficiency_distribution: EfficiencyDistribution,

    /// Common usage patterns
    pub usage_patterns: Vec<UsagePattern>,
}

/// Usage statistics for a specific test
#[derive(Debug, Clone)]
pub struct TestResourceUsageStats {
    /// Resources used by this test
    pub resources_used: Vec<String>,

    /// Total resource cost
    pub total_resource_cost: f64,

    /// Average efficiency across resources
    pub average_efficiency: f32,

    /// Resource usage stability
    pub usage_stability: f32,

    /// Most constraining resource
    pub bottleneck_resource: Option<String>,
}

/// Temporal usage patterns
#[derive(Debug, Default, Clone)]
pub struct TemporalUsagePatterns {
    /// Hourly usage distribution
    pub hourly_distribution: [f64; 24],

    /// Daily usage trends
    pub daily_trends: HashMap<String, f64>, // day_of_week -> average_usage

    /// Seasonal patterns (if applicable)
    pub seasonal_patterns: HashMap<String, f64>,

    /// Peak usage periods
    pub peak_periods: Vec<PeakUsagePeriod>,
}

/// Peak usage period information
#[derive(Debug, Clone)]
pub struct PeakUsagePeriod {
    /// Start time of peak period
    pub start_time: DateTime<Utc>,

    /// Duration of peak period
    pub duration: Duration,

    /// Peak usage level
    pub peak_level: f64,

    /// Resources affected
    pub affected_resources: Vec<String>,
}

/// Resource efficiency metrics
#[derive(Debug, Default, Clone)]
pub struct ResourceEfficiencyMetrics {
    /// Overall efficiency score
    pub overall_efficiency: f32,

    /// Efficiency by resource type
    pub efficiency_by_type: HashMap<String, f32>,

    /// Waste indicators
    pub waste_indicators: WasteIndicators,

    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Waste indicators for resource usage
#[derive(Debug, Clone, Default)]
pub struct WasteIndicators {
    /// Unused allocated resources percentage
    pub unused_allocation_percentage: f32,

    /// Over-provisioning factor
    pub over_provisioning_factor: f32,

    /// Resource fragmentation score
    pub fragmentation_score: f32,

    /// Idle time percentage
    pub idle_time_percentage: f32,
}

/// Optimization opportunity identification
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    /// Opportunity type
    pub opportunity_type: OptimizationType,

    /// Description of the opportunity
    pub description: String,

    /// Potential savings
    pub potential_savings: f64,

    /// Implementation difficulty
    pub implementation_difficulty: OptimizationDifficulty,

    /// Affected resources
    pub affected_resources: Vec<String>,
}

/// Types of optimization opportunities
#[derive(Debug, Clone)]
pub enum OptimizationType {
    /// Resource consolidation opportunity
    Consolidation,

    /// Resource sharing improvement
    SharingImprovement,

    /// Allocation timing optimization
    TimingOptimization,

    /// Resource type substitution
    TypeSubstitution,

    /// Capacity adjustment
    CapacityAdjustment,

    /// Custom optimization type
    Custom(String),
}

/// Optimization implementation difficulty
#[derive(Debug, Clone)]
pub enum OptimizationDifficulty {
    /// Easy (configuration change)
    Easy,

    /// Medium (minor code changes)
    Medium,

    /// Hard (significant refactoring)
    Hard,

    /// Very Hard (major architectural change)
    VeryHard,
}

/// Efficiency distribution statistics
#[derive(Debug, Clone)]
pub struct EfficiencyDistribution {
    /// Percentile values (P10, P25, P50, P75, P90)
    pub percentiles: [f32; 5],

    /// Mean efficiency
    pub mean: f32,

    /// Standard deviation
    pub std_dev: f32,

    /// Efficiency score categories
    pub categories: EfficiencyCategories,
}

/// Efficiency score categories
#[derive(Debug, Clone)]
pub struct EfficiencyCategories {
    /// Percentage with high efficiency (>0.8)
    pub high_efficiency: f32,

    /// Percentage with medium efficiency (0.5-0.8)
    pub medium_efficiency: f32,

    /// Percentage with low efficiency (<0.5)
    pub low_efficiency: f32,
}

/// Common usage patterns
#[derive(Debug, Clone)]
pub struct UsagePattern {
    /// Pattern name/identifier
    pub pattern_name: String,

    /// Pattern description
    pub description: String,

    /// Frequency of this pattern
    pub frequency: f32,

    /// Pattern characteristics
    pub characteristics: PatternCharacteristics,
}

/// Usage pattern characteristics
#[derive(Debug, Clone)]
pub struct PatternCharacteristics {
    /// Typical duration
    pub typical_duration: Duration,

    /// Usage intensity
    pub usage_intensity: f32,

    /// Resource requirements
    pub resource_requirements: HashMap<String, f64>,

    /// Concurrency level
    pub concurrency_level: f32,
}

/// Performance baseline for resource types
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    /// Resource type this baseline applies to
    pub resource_type: String,

    /// Baseline establishment date
    pub established_at: DateTime<Utc>,

    /// Baseline metrics
    pub baseline_metrics: BaselineMetrics,

    /// Baseline confidence level
    pub confidence_level: f32,

    /// Number of samples used
    pub sample_count: usize,
}

/// Baseline performance metrics
#[derive(Debug, Clone)]
pub struct BaselineMetrics {
    /// Baseline throughput
    pub throughput: f64,

    /// Baseline latency
    pub latency: Duration,

    /// Baseline efficiency
    pub efficiency: f32,

    /// Baseline error rate
    pub error_rate: f32,

    /// Baseline resource utilization
    pub resource_utilization: f32,
}

impl ResourceUsageDatabase {
    /// Create a new resource usage database with default configuration
    pub fn new() -> Self {
        Self::with_config(DatabaseConfig::default())
    }

    /// Create a new resource usage database with custom configuration
    pub fn with_config(config: DatabaseConfig) -> Self {
        Self {
            usage_records: RwLock::new(HashMap::new()),
            resource_types: RwLock::new(HashMap::new()),
            allocation_history: RwLock::new(Vec::new()),
            statistics: RwLock::new(DatabaseStatistics::default()),
            config: RwLock::new(config),
            aggregations: RwLock::new(ResourceUsageAggregations::default()),
            performance_baselines: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new resource type with the database
    pub fn register_resource_type(
        &self,
        resource_type: ResourceTypeDefinition,
    ) -> AnalysisResult<()> {
        let mut types = self.resource_types.write();

        if types.contains_key(&resource_type.type_id) {
            return Err(AnalysisError::ResourceTypeAlreadyExists {
                type_id: resource_type.type_id,
            });
        }

        types.insert(resource_type.type_id.clone(), resource_type);

        if self.config.read().detailed_logging {
            debug!("Registered new resource type: {}", types.len());
        }

        Ok(())
    }

    /// Record resource usage for a test
    pub fn record_usage(&self, record: ResourceUsageRecord) -> AnalysisResult<()> {
        let config = self.config.read();
        let test_id = record.test_id.clone();

        // Validate the record
        self.validate_usage_record(&record)?;

        // Add to usage records
        {
            let mut records = self.usage_records.write();
            let test_records = records.entry(test_id.clone()).or_default();

            // Enforce maximum records per test
            if test_records.len() >= config.max_records_per_test {
                test_records.remove(0); // Remove oldest record
            }

            test_records.push(record);
        }

        // Update statistics
        self.update_usage_statistics(&test_id)?;

        if config.detailed_logging {
            debug!("Recorded resource usage for test: {}", test_id);
        }

        Ok(())
    }

    /// Record an allocation event
    pub fn record_allocation_event(&self, event: ResourceAllocationEvent) -> AnalysisResult<()> {
        let config = self.config.read();

        // Validate the event
        self.validate_allocation_event(&event)?;

        // Add to allocation history
        {
            let mut history = self.allocation_history.write();

            // Enforce maximum events limit
            if history.len() >= config.max_allocation_events {
                history.remove(0); // Remove oldest event
            }

            history.push(event.clone());
        }

        // Update aggregations
        self.update_allocation_aggregations(&event)?;

        if config.detailed_logging {
            debug!("Recorded allocation event: {:?}", event.event_type);
        }

        Ok(())
    }

    /// Get usage records for a specific test
    pub fn get_test_usage_records(&self, test_id: &str) -> Vec<ResourceUsageRecord> {
        self.usage_records.read().get(test_id).cloned().unwrap_or_default()
    }

    /// Get usage records for a specific resource type
    pub fn get_resource_type_usage(&self, resource_type: &str) -> Vec<ResourceUsageRecord> {
        let records = self.usage_records.read();
        let mut result = Vec::new();

        for test_records in records.values() {
            for record in test_records {
                if record.resource_type == resource_type {
                    result.push(record.clone());
                }
            }
        }

        result
    }

    /// Get allocation events for a specific test
    pub fn get_test_allocation_events(&self, test_id: &str) -> Vec<ResourceAllocationEvent> {
        self.allocation_history
            .read()
            .iter()
            .filter(|event| event.test_id == test_id)
            .cloned()
            .collect()
    }

    /// Generate usage statistics report
    pub fn generate_usage_report(&self) -> UsageReport {
        let records = self.usage_records.read();
        let history = self.allocation_history.read();
        let aggregations = self.aggregations.read();

        let total_records = records.values().map(|v| v.len()).sum::<usize>();
        let total_events = history.len();

        let resource_type_counts = self.calculate_resource_type_distribution(&records);
        let test_usage_summary = self.calculate_test_usage_summary(&records);

        UsageReport {
            generated_at: Utc::now(),
            total_usage_records: total_records,
            total_allocation_events: total_events,
            resource_type_distribution: resource_type_counts,
            test_usage_summary,
            efficiency_overview: aggregations.efficiency_metrics.clone(),
            temporal_patterns: aggregations.temporal_patterns.clone(),
        }
    }

    /// Validate a usage record
    fn validate_usage_record(&self, record: &ResourceUsageRecord) -> AnalysisResult<()> {
        if record.test_id.is_empty() {
            return Err(AnalysisError::InvalidUsageRecord {
                reason: "Test ID cannot be empty".to_string(),
            });
        }

        if record.resource_type.is_empty() {
            return Err(AnalysisError::InvalidUsageRecord {
                reason: "Resource type cannot be empty".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&record.usage_amount) {
            return Err(AnalysisError::InvalidUsageRecord {
                reason: "Usage amount must be between 0.0 and 1.0".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&record.efficiency) {
            return Err(AnalysisError::InvalidUsageRecord {
                reason: "Efficiency must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(())
    }

    /// Validate an allocation event
    fn validate_allocation_event(&self, event: &ResourceAllocationEvent) -> AnalysisResult<()> {
        if event.test_id.is_empty() {
            return Err(AnalysisError::InvalidAllocationEvent {
                reason: "Test ID cannot be empty".to_string(),
            });
        }

        if event.resource_type.is_empty() {
            return Err(AnalysisError::InvalidAllocationEvent {
                reason: "Resource type cannot be empty".to_string(),
            });
        }

        Ok(())
    }

    /// Update usage statistics after adding a record
    fn update_usage_statistics(&self, _test_id: &str) -> AnalysisResult<()> {
        // Implementation would update aggregated statistics
        Ok(())
    }

    /// Update aggregations after an allocation event
    fn update_allocation_aggregations(
        &self,
        _event: &ResourceAllocationEvent,
    ) -> AnalysisResult<()> {
        // Implementation would update aggregated data
        Ok(())
    }

    /// Calculate resource type distribution
    fn calculate_resource_type_distribution(
        &self,
        records: &HashMap<String, Vec<ResourceUsageRecord>>,
    ) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for test_records in records.values() {
            for record in test_records {
                *distribution.entry(record.resource_type.clone()).or_insert(0) += 1;
            }
        }

        distribution
    }

    /// Calculate test usage summary
    fn calculate_test_usage_summary(
        &self,
        records: &HashMap<String, Vec<ResourceUsageRecord>>,
    ) -> HashMap<String, TestUsageSummary> {
        let mut summary = HashMap::new();

        for (test_id, test_records) in records {
            let total_usage_time: Duration = test_records.iter().map(|r| r.duration).sum();
            let average_efficiency =
                test_records.iter().map(|r| r.efficiency).sum::<f32>() / test_records.len() as f32;
            let unique_resources = test_records
                .iter()
                .map(|r| &r.resource_type)
                .collect::<std::collections::HashSet<_>>()
                .len();

            summary.insert(
                test_id.clone(),
                TestUsageSummary {
                    total_usage_time,
                    average_efficiency,
                    unique_resources_used: unique_resources,
                    total_records: test_records.len(),
                },
            );
        }

        summary
    }

    /// Perform database cleanup
    pub fn cleanup(&self) -> AnalysisResult<CleanupResult> {
        let config = self.config.read();
        let cutoff_time = Utc::now()
            - chrono::Duration::from_std(config.retention_period)
                .map_err(|e| AnalysisError::TimeConversionError { source: e.into() })?;

        let mut records_cleaned = 0;
        let events_cleaned;

        // Clean up old usage records
        {
            let mut records = self.usage_records.write();
            for test_records in records.values_mut() {
                let original_len = test_records.len();
                test_records.retain(|record| record.start_time > cutoff_time);
                records_cleaned += original_len - test_records.len();
            }
        }

        // Clean up old allocation events
        {
            let mut history = self.allocation_history.write();
            let original_len = history.len();
            history.retain(|event| event.timestamp > cutoff_time);
            events_cleaned = original_len - history.len();
        }

        info!(
            "Database cleanup completed: {} records, {} events cleaned",
            records_cleaned, events_cleaned
        );

        Ok(CleanupResult {
            records_cleaned,
            events_cleaned,
        })
    }

    /// Get database statistics
    pub fn get_statistics(&self) -> DatabaseStatistics {
        (*self.statistics.read()).clone()
    }
}

/// Database usage report
#[derive(Debug, Clone)]
pub struct UsageReport {
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,

    /// Total usage records in database
    pub total_usage_records: usize,

    /// Total allocation events recorded
    pub total_allocation_events: usize,

    /// Distribution of records by resource type
    pub resource_type_distribution: HashMap<String, usize>,

    /// Usage summary by test
    pub test_usage_summary: HashMap<String, TestUsageSummary>,

    /// Overall efficiency metrics
    pub efficiency_overview: ResourceEfficiencyMetrics,

    /// Temporal usage patterns
    pub temporal_patterns: TemporalUsagePatterns,
}

/// Summary of usage for a specific test
#[derive(Debug, Clone)]
pub struct TestUsageSummary {
    /// Total time resources were used
    pub total_usage_time: Duration,

    /// Average efficiency across all resource usage
    pub average_efficiency: f32,

    /// Number of unique resource types used
    pub unique_resources_used: usize,

    /// Total number of usage records
    pub total_records: usize,
}

/// Result of cleanup operation
#[derive(Debug, Clone)]
pub struct CleanupResult {
    /// Number of usage records cleaned
    pub records_cleaned: usize,

    /// Number of allocation events cleaned
    pub events_cleaned: usize,
}

impl Default for ResourceUsageDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_database_creation() {
        let db = ResourceUsageDatabase::new();
        let stats = db.get_statistics();
        assert_eq!(stats.total_resources, 0);
    }

    #[test]
    fn test_resource_type_registration() {
        let db = ResourceUsageDatabase::new();

        let resource_type = ResourceTypeDefinition {
            type_id: "test_cpu".to_string(),
            name: "Test CPU".to_string(),
            description: "CPU resource for testing".to_string(),
            category: ResourceCategory::Computational,
            max_concurrent_users: Some(4),
            sharing_capabilities: ResourceSharingSpec {
                shareable: true,
                max_sharing_level: Some(4),
                sharing_overhead: 0.1,
                constraints: vec![],
            },
            conflict_rules: vec![],
            performance_characteristics: PerformanceCharacteristics {
                throughput: ThroughputCharacteristics {
                    base_throughput: 100.0,
                    peak_throughput: 1000.0,
                    unit: "ops/sec".to_string(),
                },
                latency: LatencyCharacteristics {
                    base_latency: Duration::from_millis(10),
                    load_latency: Duration::from_millis(50),
                    variance: Duration::from_millis(5),
                },
                scalability: ScalabilityCharacteristics {
                    max_scale_factor: 8.0,
                    scale_efficiency: 0.9,
                    bottlenecks: vec![],
                },
            },
            scaling_properties: ResourceScalingProperties {
                horizontal_scaling: true,
                vertical_scaling: false,
                scaling_overhead: 0.05,
                min_scale_units: 1,
                max_scale_units: Some(16),
                scaling_response_time: Duration::from_secs(30),
            },
            monitoring_config: ResourceMonitoringConfig {
                monitor_usage: true,
                monitor_performance: true,
                sampling_interval: Duration::from_secs(10),
                usage_threshold: 0.8,
                performance_threshold: 0.9,
                predictive_analytics: false,
            },
        };

        assert!(db.register_resource_type(resource_type).is_ok());
    }

    #[test]
    fn test_usage_record_validation() {
        let db = ResourceUsageDatabase::new();

        // Valid record
        let valid_record = ResourceUsageRecord {
            id: "record_1".to_string(),
            test_id: "test_1".to_string(),
            resource_type: "cpu".to_string(),
            resource_id: "cpu_0".to_string(),
            start_time: Utc::now(),
            duration: Duration::from_secs(10),
            usage_amount: 0.5,
            efficiency: 0.8,
            concurrent_users: 1,
            peak_usage: 0.6,
            average_usage: 0.5,
            usage_variance: 0.1,
            performance_metrics: UsagePerformanceMetrics::default(),
            tags: vec![],
            metadata: HashMap::new(),
        };

        assert!(db.record_usage(valid_record).is_ok());

        // Invalid record (empty test_id)
        let invalid_record = ResourceUsageRecord {
            id: "record_2".to_string(),
            test_id: "".to_string(),
            resource_type: "cpu".to_string(),
            resource_id: "cpu_0".to_string(),
            start_time: Utc::now(),
            duration: Duration::from_secs(10),
            usage_amount: 0.5,
            efficiency: 0.8,
            concurrent_users: 1,
            peak_usage: 0.6,
            average_usage: 0.5,
            usage_variance: 0.1,
            performance_metrics: UsagePerformanceMetrics::default(),
            tags: vec![],
            metadata: HashMap::new(),
        };

        assert!(db.record_usage(invalid_record).is_err());
    }

    #[test]
    fn test_usage_report_generation() {
        let db = ResourceUsageDatabase::new();

        // Add some test data
        let record = ResourceUsageRecord {
            id: "record_1".to_string(),
            test_id: "test_1".to_string(),
            resource_type: "cpu".to_string(),
            resource_id: "cpu_0".to_string(),
            start_time: Utc::now(),
            duration: Duration::from_secs(10),
            usage_amount: 0.5,
            efficiency: 0.8,
            concurrent_users: 1,
            peak_usage: 0.6,
            average_usage: 0.5,
            usage_variance: 0.1,
            performance_metrics: UsagePerformanceMetrics::default(),
            tags: vec![],
            metadata: HashMap::new(),
        };

        db.record_usage(record).unwrap();

        let report = db.generate_usage_report();
        assert_eq!(report.total_usage_records, 1);
        assert!(report.resource_type_distribution.contains_key("cpu"));
    }
}
