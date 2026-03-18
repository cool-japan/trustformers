//! # DefaultSystemResourceMonitor - Trait Implementations
//!
//! This module contains trait implementations for `DefaultSystemResourceMonitor`.
//!
//! ## Implemented Traits
//!
//! - `SystemResourceMonitor`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::ResourceUsageSnapshot;
use anyhow::Result;
use async_trait::async_trait;
use std::time::Instant;

use super::functions::SystemResourceMonitor;
use super::types::DefaultSystemResourceMonitor;

#[async_trait]
impl SystemResourceMonitor for DefaultSystemResourceMonitor {
    async fn collect_resources(&self) -> Result<ResourceUsageSnapshot> {
        let network_rx = self.get_network_rx_rate().await.unwrap_or(0.0);
        let network_tx = self.get_network_tx_rate().await.unwrap_or(0.0);
        let gpu_usage_val = self.get_gpu_usage().await.unwrap_or(0.0);
        Ok(ResourceUsageSnapshot {
            timestamp: Instant::now(),
            cpu_usage: self.get_cpu_usage().await.unwrap_or(0.0),
            memory_usage: self.get_memory_usage().await.unwrap_or(0),
            available_memory: self.get_available_memory().await.unwrap_or(0),
            io_read_rate: self.get_io_read_rate().await.unwrap_or(0.0),
            io_write_rate: self.get_io_write_rate().await.unwrap_or(0.0),
            network_in_rate: network_rx,
            network_out_rate: network_tx,
            network_rx_rate: network_rx,
            network_tx_rate: network_tx,
            gpu_utilization: gpu_usage_val,
            gpu_usage: gpu_usage_val,
            gpu_memory_usage: 0,
            disk_usage: self.get_disk_usage().await.unwrap_or(0.0),
            load_average: [0.15, 0.12, 0.10],
            process_count: 150,
            thread_count: 800,
            memory_pressure: 0.25,
            io_wait: 0.05,
        })
    }
}
