//! Custom resource type management for extensible resource handling.

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, info};

/// Custom resource manager for handling extensible resource types
pub struct CustomResourceManager {
    /// Custom resource types
    resource_types: Arc<Mutex<HashMap<String, CustomResourceType>>>,
    /// Active custom resources
    active_resources: Arc<Mutex<HashMap<String, CustomResource>>>,
    /// Resource providers
    providers: Arc<Mutex<HashMap<String, Box<dyn CustomResourceProvider + Send + Sync>>>>,
    /// Usage statistics
    usage_stats: Arc<Mutex<CustomResourceStatistics>>,
}

/// Custom resource type definition
#[derive(Debug, Clone)]
pub struct CustomResourceType {
    /// Resource type name
    pub type_name: String,
    /// Resource description
    pub description: String,
    /// Resource schema
    pub schema: CustomResourceSchema,
    /// Default configuration
    pub default_config: HashMap<String, String>,
    /// Resource lifecycle hooks
    pub lifecycle_hooks: CustomResourceLifecycleHooks,
}

/// Custom resource schema
#[derive(Debug, Clone)]
pub struct CustomResourceSchema {
    /// Required properties
    pub required_properties: Vec<String>,
    /// Optional properties
    pub optional_properties: Vec<String>,
    /// Property types
    pub property_types: HashMap<String, CustomPropertyType>,
    /// Validation rules
    pub validation_rules: Vec<ValidationRule>,
}

/// Custom property types
#[derive(Debug, Clone)]
pub enum CustomPropertyType {
    String,
    Integer,
    Float,
    Boolean,
    Array(Box<CustomPropertyType>),
    Object(HashMap<String, CustomPropertyType>),
}

/// Validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Property name
    pub property: String,
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Rule parameters
    pub parameters: HashMap<String, String>,
}

/// Validation rule types
#[derive(Debug, Clone)]
pub enum ValidationRuleType {
    Range(f64, f64),
    MinLength(usize),
    MaxLength(usize),
    Pattern(String),
    Custom(String),
}

/// Resource lifecycle hooks
#[derive(Debug, Clone)]
pub struct CustomResourceLifecycleHooks {
    /// Pre-allocation hook
    pub pre_allocation: Option<String>,
    /// Post-allocation hook
    pub post_allocation: Option<String>,
    /// Pre-deallocation hook
    pub pre_deallocation: Option<String>,
    /// Post-deallocation hook
    pub post_deallocation: Option<String>,
    /// Health check hook
    pub health_check: Option<String>,
}

/// Custom resource instance
#[derive(Debug, Clone)]
pub struct CustomResource {
    /// Resource ID
    pub resource_id: String,
    /// Resource type
    pub resource_type: String,
    /// Test ID using this resource
    pub test_id: String,
    /// Resource properties
    pub properties: HashMap<String, CustomPropertyValue>,
    /// Resource state
    pub state: CustomResourceState,
    /// Allocation timestamp
    pub allocated_at: DateTime<Utc>,
    /// Last accessed timestamp
    pub last_accessed: DateTime<Utc>,
    /// Resource metadata
    pub metadata: HashMap<String, String>,
}

/// Custom property value
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CustomPropertyValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<CustomPropertyValue>),
    Object(HashMap<String, CustomPropertyValue>),
}

/// Custom resource state
#[derive(Debug, Clone)]
pub enum CustomResourceState {
    /// Resource is available
    Available,
    /// Resource is allocated
    Allocated,
    /// Resource is in use
    InUse,
    /// Resource has error
    Error(String),
    /// Resource is being cleaned up
    Cleanup,
    /// Resource is in maintenance
    Maintenance,
}

/// Custom resource provider trait
pub trait CustomResourceProvider: std::fmt::Debug {
    /// Allocate a custom resource
    fn allocate(
        &self,
        resource_type: &str,
        properties: &HashMap<String, CustomPropertyValue>,
    ) -> Result<CustomResource>;

    /// Deallocate a custom resource
    fn deallocate(&self, resource: &CustomResource) -> Result<()>;

    /// Check resource health
    fn check_health(&self, resource: &CustomResource) -> Result<bool>;

    /// Update resource properties
    fn update_properties(
        &self,
        resource: &mut CustomResource,
        properties: HashMap<String, CustomPropertyValue>,
    ) -> Result<()>;

    /// Get provider name
    fn name(&self) -> &str;

    /// Get supported resource types
    fn supported_types(&self) -> Vec<String>;
}

/// Custom resource statistics
#[derive(Debug, Default, Clone)]
pub struct CustomResourceStatistics {
    /// Total resources allocated
    pub total_allocated: u64,
    /// Currently active resources
    pub currently_active: usize,
    /// Resources by type
    pub by_type: HashMap<String, usize>,
    /// Allocation failures
    pub allocation_failures: u64,
    /// Average allocation time
    pub average_allocation_time: std::time::Duration,
}

impl CustomResourceManager {
    /// Create new custom resource manager
    pub async fn new() -> Result<Self> {
        Ok(Self {
            resource_types: Arc::new(Mutex::new(HashMap::new())),
            active_resources: Arc::new(Mutex::new(HashMap::new())),
            providers: Arc::new(Mutex::new(HashMap::new())),
            usage_stats: Arc::new(Mutex::new(CustomResourceStatistics::default())),
        })
    }

    /// Register a custom resource type
    pub async fn register_resource_type(&self, resource_type: CustomResourceType) -> Result<()> {
        info!(
            "Registering custom resource type: {}",
            resource_type.type_name
        );

        let mut resource_types = self.resource_types.lock();
        resource_types.insert(resource_type.type_name.clone(), resource_type);

        Ok(())
    }

    /// Register a custom resource provider
    pub async fn register_provider(
        &self,
        provider: Box<dyn CustomResourceProvider + Send + Sync>,
    ) -> Result<()> {
        info!("Registering custom resource provider: {}", provider.name());

        let mut providers = self.providers.lock();
        providers.insert(provider.name().to_string(), provider);

        Ok(())
    }

    /// Allocate a custom resource
    pub async fn allocate_resource(
        &self,
        resource_type: &str,
        properties: HashMap<String, CustomPropertyValue>,
        test_id: &str,
    ) -> Result<String> {
        info!(
            "Allocating custom resource of type '{}' for test: {}",
            resource_type, test_id
        );

        // Validate resource type exists
        let resource_types = self.resource_types.lock();
        let type_def = resource_types
            .get(resource_type)
            .ok_or_else(|| anyhow::anyhow!("Unknown resource type: {}", resource_type))?;

        // Validate properties
        self.validate_properties(&type_def.schema, &properties)?;
        drop(resource_types);

        // Find provider for this resource type
        let providers = self.providers.lock();
        let provider = providers
            .values()
            .find(|p| p.supported_types().contains(&resource_type.to_string()))
            .ok_or_else(|| {
                anyhow::anyhow!("No provider found for resource type: {}", resource_type)
            })?;

        // Allocate the resource
        let mut resource = provider.allocate(resource_type, &properties)?;
        resource.test_id = test_id.to_string();

        let resource_id = resource.resource_id.clone();

        // Track the allocation
        let mut active_resources = self.active_resources.lock();
        active_resources.insert(resource_id.clone(), resource);

        // Update statistics
        // usage_stats is already Arc<Mutex<CustomResourceStatistics>>, no need for double .usage_stats
        let mut stats = self.usage_stats.lock();
        stats.total_allocated += 1;
        stats.currently_active += 1;
        *stats.by_type.entry(resource_type.to_string()).or_insert(0) += 1;

        info!("Successfully allocated custom resource: {}", resource_id);
        Ok(resource_id)
    }

    /// Deallocate a custom resource
    pub async fn deallocate_resource(&self, resource_id: &str) -> Result<()> {
        debug!("Deallocating custom resource: {}", resource_id);

        let mut active_resources = self.active_resources.lock();
        if let Some(resource) = active_resources.remove(resource_id) {
            // Find provider and deallocate
            let providers = self.providers.lock();
            if let Some(provider) = providers
                .values()
                .find(|p| p.supported_types().contains(&resource.resource_type))
            {
                provider.deallocate(&resource)?;
            }

            // Update statistics
            let mut stats = self.usage_stats.lock();
            stats.currently_active = stats.currently_active.saturating_sub(1);
            if let Some(count) = stats.by_type.get_mut(&resource.resource_type) {
                *count = count.saturating_sub(1);
            }

            info!("Successfully deallocated custom resource: {}", resource_id);
        } else {
            debug!(
                "Custom resource {} not found or already deallocated",
                resource_id
            );
        }

        Ok(())
    }

    /// Deallocate resources for a test
    pub async fn deallocate_resources_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating custom resources for test: {}", test_id);

        let mut active_resources = self.active_resources.lock();
        let resources_to_remove: Vec<String> = active_resources
            .iter()
            .filter(|(_, resource)| resource.test_id == test_id)
            .map(|(resource_id, _)| resource_id.clone())
            .collect();

        let providers = self.providers.lock();
        for resource_id in &resources_to_remove {
            if let Some(resource) = active_resources.remove(resource_id) {
                // Find provider and deallocate
                if let Some(provider) = providers
                    .values()
                    .find(|p| p.supported_types().contains(&resource.resource_type))
                {
                    let _ = provider.deallocate(&resource);
                }
            }
        }

        // Update statistics
        let mut stats = self.usage_stats.lock();
        stats.currently_active = stats.currently_active.saturating_sub(resources_to_remove.len());

        info!(
            "Released {} custom resources for test: {}",
            resources_to_remove.len(),
            test_id
        );
        Ok(())
    }

    /// Validate resource properties against schema
    fn validate_properties(
        &self,
        schema: &CustomResourceSchema,
        properties: &HashMap<String, CustomPropertyValue>,
    ) -> Result<()> {
        // Check required properties
        for required_prop in &schema.required_properties {
            if !properties.contains_key(required_prop) {
                return Err(anyhow::anyhow!(
                    "Missing required property: {}",
                    required_prop
                ));
            }
        }

        // Validate property types
        for (prop_name, prop_value) in properties {
            if let Some(expected_type) = schema.property_types.get(prop_name) {
                if !self.validate_property_type(prop_value, expected_type) {
                    return Err(anyhow::anyhow!("Property '{}' has invalid type", prop_name));
                }
            }
        }

        Ok(())
    }

    /// Validate property type
    fn validate_property_type(
        &self,
        value: &CustomPropertyValue,
        expected_type: &CustomPropertyType,
    ) -> bool {
        match (value, expected_type) {
            (CustomPropertyValue::String(_), CustomPropertyType::String) => true,
            (CustomPropertyValue::Integer(_), CustomPropertyType::Integer) => true,
            (CustomPropertyValue::Float(_), CustomPropertyType::Float) => true,
            (CustomPropertyValue::Boolean(_), CustomPropertyType::Boolean) => true,
            (CustomPropertyValue::Array(arr), CustomPropertyType::Array(elem_type)) => {
                arr.iter().all(|elem| self.validate_property_type(elem, elem_type))
            },
            _ => false,
        }
    }

    /// Get active resources
    pub async fn get_active_resources(&self) -> Result<Vec<CustomResource>> {
        let active_resources = self.active_resources.lock();
        Ok(active_resources.values().cloned().collect())
    }

    /// Get resources for a specific test
    pub async fn get_resources_for_test(&self, test_id: &str) -> Result<Vec<CustomResource>> {
        let active_resources = self.active_resources.lock();
        Ok(active_resources
            .values()
            .filter(|resource| resource.test_id == test_id)
            .cloned()
            .collect())
    }

    /// Get usage statistics
    pub async fn get_statistics(&self) -> Result<CustomResourceStatistics> {
        let stats = self.usage_stats.lock();
        // MutexGuard doesn't implement Clone, dereference to clone the inner value
        Ok((*stats).clone())
    }

    /// Health check for custom resources
    pub async fn health_check_resources(&self) -> Result<HashMap<String, bool>> {
        let mut health_results = HashMap::new();
        let active_resources = self.active_resources.lock();
        let providers = self.providers.lock();

        for (resource_id, resource) in active_resources.iter() {
            if let Some(provider) = providers
                .values()
                .find(|p| p.supported_types().contains(&resource.resource_type))
            {
                let is_healthy = provider.check_health(resource).unwrap_or(false);
                health_results.insert(resource_id.clone(), is_healthy);
            }
        }

        Ok(health_results)
    }
}
