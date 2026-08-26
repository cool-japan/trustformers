//! Dashboard Management System
//!
//! This module provides dashboard services for test performance monitoring,
//! including widget management, real-time updates, and user interface components.

use super::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::sync::{broadcast, RwLock};

/// Main dashboard management system.
///
/// 0.2.1: this used to also hold a `LayoutEngine` and a
/// `RwLock<HashMap<String, UserPreferences>>`. The layout engine stored a copy
/// of the config and had no other method; the preferences map was never written
/// to and never read. Both are gone -- a manager that appears to lay dashboards
/// out and honour per-user preferences, but does neither, is a claim about
/// behaviour that does not exist.
#[derive(Debug)]
pub struct DashboardManager {
    config: DashboardConfig,
    dashboard_store: RwLock<HashMap<String, Dashboard>>,
    widget_manager: WidgetManager,
    real_time_updater: RealTimeUpdater,
}

/// Dashboard definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    pub dashboard_id: String,
    pub dashboard_name: String,
    pub description: String,
    pub layout: DashboardLayout,
    pub widgets: Vec<Widget>,
    pub filters: Vec<DashboardFilter>,
    pub refresh_interval: Duration,
    pub permissions: DashboardPermissions,
    pub created_at: SystemTime,
    pub last_modified: SystemTime,
    pub owner: String,
    pub shared_with: Vec<String>,
}

/// Widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    pub widget_id: String,
    pub widget_name: String,
    pub widget_type: WidgetType,
    pub data_source: DataSource,
    pub visualization: VisualizationConfig,
    pub position: WidgetPosition,
    pub size: WidgetSize,
    pub configuration: WidgetConfiguration,
    pub refresh_rate: Duration,
    pub filters: Vec<WidgetFilter>,
}

/// Widget types available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    Chart,
    Metric,
    Table,
    Gauge,
    Heatmap,
    Timeline,
    Alert,
    Text,
    Custom { widget_class: String },
}

/// Widget management system.
///
/// 0.2.1: dropped a `WidgetFactory` (a list of widget-type name strings that
/// nothing consulted, and which manufactured nothing) and a `WidgetUpdater`
/// (an interval nothing ticked).
#[derive(Debug)]
pub struct WidgetManager {
    widget_registry: RwLock<HashMap<String, WidgetDefinition>>,
    widget_cache: RwLock<HashMap<String, WidgetData>>,
}

impl WidgetManager {
    fn new(config: &DashboardConfig) -> Self {
        let mut registry = HashMap::new();
        for widget_position in &config.layout.widgets {
            registry.insert(
                widget_position.widget_id.clone(),
                WidgetDefinition {
                    widget_id: widget_position.widget_id.clone(),
                    widget_type: "custom".to_string(),
                    configuration: WidgetConfiguration {
                        widget_type: "custom".to_string(),
                        data_source: DataSource::default(),
                        refresh_rate: config.refresh_interval,
                        filters: Vec::new(),
                    },
                },
            );
        }

        Self {
            widget_registry: RwLock::new(registry),
            widget_cache: RwLock::new(HashMap::new()),
        }
    }

    async fn update_widget_data(
        &self,
        widget_id: &str,
        data: WidgetData,
    ) -> Result<(), DashboardError> {
        let registry = self.widget_registry.read().await;
        if !registry.contains_key(widget_id) {
            return Err(DashboardError::WidgetError {
                widget_id: widget_id.to_string(),
                reason: "Widget not registered".to_string(),
            });
        }
        drop(registry);

        let mut cache = self.widget_cache.write().await;
        cache.insert(widget_id.to_string(), data);
        Ok(())
    }
}

/// Real-time dashboard updates.
///
/// 0.2.1: dropped a per-dashboard `subscriptions` map and an `UpdateScheduler`.
/// [`DashboardManager::subscribe_to_updates`] hands out receivers on the single
/// broadcast channel below and always did; the map was never populated, and the
/// scheduler only held a config copy.
#[derive(Debug)]
pub struct RealTimeUpdater {
    update_sender: broadcast::Sender<DashboardUpdate>,
}

/// Dashboard update event
#[derive(Debug, Clone)]
pub struct DashboardUpdate {
    pub dashboard_id: String,
    pub widget_id: String,
    pub update_type: UpdateType,
    pub data: UpdateData,
    pub timestamp: SystemTime,
}

impl DashboardManager {
    /// Create new dashboard manager
    pub fn new(config: DashboardConfig) -> Self {
        let (update_sender, _) = broadcast::channel(1000);

        Self {
            config: config.clone(),
            dashboard_store: RwLock::new(HashMap::new()),
            widget_manager: WidgetManager::new(&config),
            real_time_updater: RealTimeUpdater { update_sender },
        }
    }

    /// The configuration this manager was built with.
    pub fn config(&self) -> &DashboardConfig {
        &self.config
    }

    /// Create new dashboard
    pub async fn create_dashboard(&self, dashboard: Dashboard) -> Result<String, DashboardError> {
        let dashboard_id = dashboard.dashboard_id.clone();
        let mut store = self.dashboard_store.write().await;
        store.insert(dashboard_id.clone(), dashboard);
        Ok(dashboard_id)
    }

    /// Get dashboard
    pub async fn get_dashboard(&self, dashboard_id: &str) -> Result<Dashboard, DashboardError> {
        let store = self.dashboard_store.read().await;
        store
            .get(dashboard_id)
            .cloned()
            .ok_or_else(|| DashboardError::DashboardNotFound {
                dashboard_id: dashboard_id.to_string(),
            })
    }

    /// Update widget data
    pub async fn update_widget_data(
        &self,
        widget_id: &str,
        data: WidgetData,
    ) -> Result<(), DashboardError> {
        self.widget_manager.update_widget_data(widget_id, data).await
    }

    /// Subscribe to dashboard updates
    pub async fn subscribe_to_updates(
        &self,
        _dashboard_id: &str,
    ) -> broadcast::Receiver<DashboardUpdate> {
        self.real_time_updater.update_sender.subscribe()
    }
}

/// Dashboard errors
#[derive(Debug, Clone)]
pub enum DashboardError {
    DashboardNotFound { dashboard_id: String },
    WidgetError { widget_id: String, reason: String },
    LayoutError { reason: String },
    PermissionDenied { user: String, operation: String },
    ConfigurationError { parameter: String, reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_manager_creation() {
        let config = DashboardConfig::default();
        let _manager = DashboardManager::new(config);

        // Basic creation test - succeeds if no panic
    }

    /// The caller's configuration must survive construction rather than being
    /// replaced by a default.
    #[test]
    fn manager_keeps_the_configuration_it_was_given() {
        let mut config = DashboardConfig::default();
        config.refresh_interval = Duration::from_millis(4242);

        let manager = DashboardManager::new(config);

        assert_eq!(
            manager.config().refresh_interval,
            Duration::from_millis(4242)
        );
    }
}
