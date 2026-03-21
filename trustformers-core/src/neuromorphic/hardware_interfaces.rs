//! Hardware interfaces for neuromorphic platforms

use crate::neuromorphic::{spiking_networks::SpikingNeuralNetwork, NeuromorphicConfig};
use anyhow::Result;

/// Trait for neuromorphic hardware interfaces
pub trait NeuromorphicHardware: std::fmt::Debug {
    fn deploy_network(&mut self, name: &str, network: &SpikingNeuralNetwork) -> Result<()>;
    fn configure_platform(&mut self, config: &NeuromorphicConfig) -> Result<()>;
}
