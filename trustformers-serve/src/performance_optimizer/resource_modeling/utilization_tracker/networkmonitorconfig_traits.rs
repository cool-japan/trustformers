//! # NetworkMonitorConfig - Trait Implementations
//!
//! This module contains trait implementations for `NetworkMonitorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::NetworkMonitorConfig;

impl Default for NetworkMonitorConfig {
    fn default() -> Self {
        Self {
            per_interface_monitoring: true,
            protocol_analysis: true,
            connection_tracking: true,
            packet_loss_monitoring: true,
            monitored_protocols: vec!["TCP".to_string(), "UDP".to_string(), "ICMP".to_string()],
            max_tracked_connections: 1000,
        }
    }
}
