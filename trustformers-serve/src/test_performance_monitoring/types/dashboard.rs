//! Dashboard Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// Import types from sibling modules
use super::config::{DashboardConfig, SubscriptionConfig, WidgetConfiguration};
use super::enums::{FilterOperator, FilterValue};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayoutEngine {
    config: DashboardConfig,
}

impl LayoutEngine {
    pub fn new(config: &DashboardConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DashboardLayout {
    pub grid_columns: u32,
    pub grid_rows: u32,
    pub widgets: Vec<WidgetPosition>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DashboardFilter {
    pub field: String,
    pub operator: FilterOperator,
    pub value: FilterValue,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WidgetPosition {
    pub widget_id: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WidgetSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WidgetFilter {
    pub field: String,
    pub value: FilterValue,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WidgetFactory {
    pub widget_types: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WidgetDefinition {
    pub widget_id: String,
    pub widget_type: String,
    pub configuration: WidgetConfiguration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WidgetData {
    pub widget_id: String,
    pub data: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WidgetUpdater {
    pub update_interval: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DashboardPermissions {
    pub owner: String,
    pub viewers: Vec<String>,
    pub editors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DashboardSubscription {
    pub subscription_id: String,
    pub dashboard_id: String,
    pub user_id: String,
    pub notification_config: SubscriptionConfig,
}
