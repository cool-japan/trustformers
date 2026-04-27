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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_layout_default() {
        let dl = DashboardLayout::default();
        assert_eq!(dl.grid_columns, 0);
        assert_eq!(dl.grid_rows, 0);
        assert!(dl.widgets.is_empty());
    }

    #[test]
    fn test_dashboard_layout_with_grid() {
        let dl = DashboardLayout {
            grid_columns: 12,
            grid_rows: 8,
            widgets: Vec::new(),
        };
        assert_eq!(dl.grid_columns, 12);
        assert_eq!(dl.grid_rows, 8);
    }

    #[test]
    fn test_dashboard_layout_with_widgets() {
        let w1 = WidgetPosition {
            widget_id: "w1".to_string(),
            x: 0,
            y: 0,
            width: 6,
            height: 4,
        };
        let w2 = WidgetPosition {
            widget_id: "w2".to_string(),
            x: 6,
            y: 0,
            width: 6,
            height: 4,
        };
        let dl = DashboardLayout {
            grid_columns: 12,
            grid_rows: 8,
            widgets: vec![w1, w2],
        };
        assert_eq!(dl.widgets.len(), 2);
        assert_eq!(dl.widgets[0].widget_id, "w1");
        assert_eq!(dl.widgets[1].x, 6);
    }

    #[test]
    fn test_widget_position_default() {
        let wp = WidgetPosition::default();
        assert!(wp.widget_id.is_empty());
        assert_eq!(wp.x, 0);
        assert_eq!(wp.y, 0);
        assert_eq!(wp.width, 0);
        assert_eq!(wp.height, 0);
    }

    #[test]
    fn test_widget_position_placement() {
        let wp = WidgetPosition {
            widget_id: "chart-1".to_string(),
            x: 3,
            y: 2,
            width: 4,
            height: 3,
        };
        assert_eq!(wp.x + wp.width, 7);
        assert_eq!(wp.y + wp.height, 5);
    }

    #[test]
    fn test_widget_size_construction() {
        let ws = WidgetSize {
            width: 800,
            height: 600,
        };
        assert_eq!(ws.width, 800);
        assert_eq!(ws.height, 600);
    }

    #[test]
    fn test_widget_size_aspect_ratio() {
        let ws = WidgetSize {
            width: 1920,
            height: 1080,
        };
        let ratio = ws.width as f64 / ws.height as f64;
        assert!((ratio - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_widget_factory_construction() {
        let wf = WidgetFactory {
            widget_types: vec![
                "line_chart".to_string(),
                "bar_chart".to_string(),
                "gauge".to_string(),
            ],
        };
        assert_eq!(wf.widget_types.len(), 3);
        assert!(wf.widget_types.contains(&"gauge".to_string()));
    }

    #[test]
    fn test_widget_factory_empty() {
        let wf = WidgetFactory {
            widget_types: Vec::new(),
        };
        assert!(wf.widget_types.is_empty());
    }

    #[test]
    fn test_widget_data_construction() {
        let now = chrono::Utc::now();
        let wd = WidgetData {
            widget_id: "wd-001".to_string(),
            data: serde_json::json!({"value": 42, "label": "CPU"}),
            timestamp: now,
        };
        assert_eq!(wd.widget_id, "wd-001");
        assert!(!wd.data.is_null());
    }

    #[test]
    fn test_widget_data_null_data() {
        let now = chrono::Utc::now();
        let wd = WidgetData {
            widget_id: "wd-002".to_string(),
            data: serde_json::Value::Null,
            timestamp: now,
        };
        assert!(wd.data.is_null());
    }

    #[test]
    fn test_widget_updater_construction() {
        let wu = WidgetUpdater {
            update_interval: std::time::Duration::from_secs(30),
        };
        assert_eq!(wu.update_interval, std::time::Duration::from_secs(30));
    }

    #[test]
    fn test_widget_updater_fast_interval() {
        let wu = WidgetUpdater {
            update_interval: std::time::Duration::from_secs(1),
        };
        assert!(wu.update_interval < std::time::Duration::from_secs(5));
    }

    #[test]
    fn test_dashboard_permissions_construction() {
        let dp = DashboardPermissions {
            owner: "alice".to_string(),
            viewers: vec!["bob".to_string(), "carol".to_string()],
            editors: vec!["dave".to_string()],
        };
        assert_eq!(dp.owner, "alice");
        assert_eq!(dp.viewers.len(), 2);
        assert_eq!(dp.editors.len(), 1);
    }

    #[test]
    fn test_dashboard_permissions_no_viewers() {
        let dp = DashboardPermissions {
            owner: "admin".to_string(),
            viewers: Vec::new(),
            editors: Vec::new(),
        };
        assert!(dp.viewers.is_empty());
        assert!(dp.editors.is_empty());
    }

    #[test]
    fn test_dashboard_subscription_construction() {
        let config = super::super::config::SubscriptionConfig::default();
        let ds = DashboardSubscription {
            subscription_id: "sub-001".to_string(),
            dashboard_id: "dash-001".to_string(),
            user_id: "user-42".to_string(),
            notification_config: config,
        };
        assert_eq!(ds.subscription_id, "sub-001");
        assert_eq!(ds.dashboard_id, "dash-001");
        assert_eq!(ds.user_id, "user-42");
    }

    #[test]
    fn test_widget_data_array_data() {
        let now = chrono::Utc::now();
        let wd = WidgetData {
            widget_id: "wd-003".to_string(),
            data: serde_json::json!([1, 2, 3, 4, 5]),
            timestamp: now,
        };
        if let serde_json::Value::Array(arr) = &wd.data {
            assert_eq!(arr.len(), 5);
        } else {
            panic!("expected array data");
        }
    }

    #[test]
    fn test_dashboard_layout_covers_full_grid() {
        let widgets: Vec<WidgetPosition> = (0..4)
            .map(|i| WidgetPosition {
                widget_id: format!("w-{}", i),
                x: (i % 2) as u32 * 6,
                y: (i / 2) as u32 * 4,
                width: 6,
                height: 4,
            })
            .collect();
        let dl = DashboardLayout {
            grid_columns: 12,
            grid_rows: 8,
            widgets,
        };
        assert_eq!(dl.widgets.len(), 4);
        for (i, w) in dl.widgets.iter().enumerate() {
            assert_eq!(w.widget_id, format!("w-{}", i));
        }
    }

    #[test]
    fn test_dashboard_filter_default() {
        let df = DashboardFilter::default();
        assert!(df.field.is_empty());
    }

    #[test]
    fn test_dashboard_filter_with_field() {
        let df = DashboardFilter {
            field: "status".to_string(),
            operator: super::super::enums::FilterOperator::Equals,
            value: super::super::enums::FilterValue::String("active".to_string()),
        };
        assert_eq!(df.field, "status");
    }

    #[test]
    fn test_dashboard_permissions_owner_not_in_viewers() {
        let dp = DashboardPermissions {
            owner: "owner_user".to_string(),
            viewers: vec!["viewer1".to_string(), "viewer2".to_string()],
            editors: vec!["editor1".to_string()],
        };
        assert!(!dp.viewers.contains(&dp.owner));
    }
}
