//! # DashboardConfig - Trait Implementations
//!
//! This module contains trait implementations for `DashboardConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{ChartConfig, ChartType, DashboardConfig, DashboardTheme, DataSource};

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            host: "127.0.0.1".to_string(),
            port: 8081,
            refresh_interval: Duration::from_secs(5),
            theme: DashboardTheme::Auto,
            charts: vec![
                ChartConfig {
                    title: "Test Execution Times".to_string(),
                    chart_type: ChartType::LineChart,
                    data_source: DataSource::ExecutionTimes,
                    time_window: Duration::from_secs(3600),
                    update_interval: Duration::from_secs(30),
                },
                ChartConfig {
                    title: "Success Rate".to_string(),
                    chart_type: ChartType::AreaChart,
                    data_source: DataSource::SuccessRates,
                    time_window: Duration::from_secs(3600),
                    update_interval: Duration::from_secs(30),
                },
                ChartConfig {
                    title: "Timeout Events".to_string(),
                    chart_type: ChartType::BarChart,
                    data_source: DataSource::TimeoutEvents,
                    time_window: Duration::from_secs(3600),
                    update_interval: Duration::from_secs(60),
                },
            ],
        }
    }
}
