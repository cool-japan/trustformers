//! Neuromorphic Computing Research Infrastructure
//!
//! This module provides experimental support for neuromorphic computing architectures
//! including spiking neural networks, event-driven processing, and neuromorphic hardware.

pub mod encoding;
pub mod event_processing;
pub mod hardware_interfaces;
pub mod neuron_models;
pub mod spiking_networks;
pub mod synaptic_plasticity;

pub use encoding::*;
pub use hardware_interfaces::*;
pub use spiking_networks::*;

use crate::tensor::Tensor;
use anyhow::Result;
use std::collections::{HashMap, VecDeque};

/// Neuromorphic hardware platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuromorphicPlatform {
    Simulation,
    Loihi,       // Intel Loihi
    TrueNorth,   // IBM TrueNorth
    SpiNNaker,   // SpiNNaker
    BrainScaleS, // BrainScaleS
    Akida,       // BrainChip Akida
    Darwin,      // Cambricon Darwin
}

/// Spike event representation
#[derive(Debug, Clone, Copy)]
pub struct SpikeEvent {
    pub neuron_id: usize,
    pub timestamp: f64,
    pub weight: f32,
}

/// Neuromorphic network configuration
#[derive(Debug, Clone)]
pub struct NeuromorphicConfig {
    pub platform: NeuromorphicPlatform,
    pub time_step: f64,
    pub simulation_time: f64,
    pub num_cores: usize,
    pub neurons_per_core: usize,
    pub synapses_per_core: usize,
    pub enable_plasticity: bool,
    pub power_optimization: bool,
}

/// Event buffer for spike processing
#[derive(Debug, Clone)]
pub struct EventBuffer {
    events: VecDeque<SpikeEvent>,
    max_size: usize,
    current_time: f64,
}

/// Neuromorphic processor simulation
#[derive(Debug)]
pub struct NeuromorphicProcessor {
    config: NeuromorphicConfig,
    networks: HashMap<String, SpikingNeuralNetwork>,
    event_buffer: EventBuffer,
    hardware_interface: Option<Box<dyn NeuromorphicHardware>>,
    energy_tracker: EnergyTracker,
}

/// Energy consumption tracking
#[derive(Debug, Clone)]
pub struct EnergyTracker {
    pub spike_energy: f64,
    pub synaptic_energy: f64,
    pub leakage_energy: f64,
    pub total_spikes: usize,
    pub total_synaptic_operations: usize,
}

impl SpikeEvent {
    /// Create a new spike event
    pub fn new(neuron_id: usize, timestamp: f64, weight: f32) -> Self {
        Self {
            neuron_id,
            timestamp,
            weight,
        }
    }

    /// Check if spike is active at given time
    pub fn is_active_at(&self, time: f64, tolerance: f64) -> bool {
        (self.timestamp - time).abs() < tolerance
    }
}

impl EventBuffer {
    /// Create a new event buffer
    pub fn new(max_size: usize) -> Self {
        Self {
            events: VecDeque::new(),
            max_size,
            current_time: 0.0,
        }
    }

    /// Add spike event
    pub fn add_event(&mut self, event: SpikeEvent) {
        if self.events.len() >= self.max_size {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// Get events in time window
    pub fn get_events_in_window(&self, start_time: f64, end_time: f64) -> Vec<SpikeEvent> {
        self.events
            .iter()
            .filter(|event| event.timestamp >= start_time && event.timestamp <= end_time)
            .cloned()
            .collect()
    }

    /// Clear old events
    pub fn clear_old_events(&mut self, cutoff_time: f64) {
        self.events.retain(|event| event.timestamp >= cutoff_time);
    }

    /// Update current time
    pub fn update_time(&mut self, new_time: f64) {
        self.current_time = new_time;
        self.clear_old_events(new_time - 10.0); // Keep last 10 time units
    }

    /// Get event count
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl NeuromorphicProcessor {
    /// Create a new neuromorphic processor
    pub fn new(config: NeuromorphicConfig) -> Self {
        Self {
            config: config.clone(),
            networks: HashMap::new(),
            event_buffer: EventBuffer::new(10000),
            hardware_interface: None,
            energy_tracker: EnergyTracker::new(),
        }
    }

    /// Create a processor with specific platform
    pub fn with_platform(platform: NeuromorphicPlatform) -> Self {
        let config = NeuromorphicConfig {
            platform,
            time_step: 1.0,
            simulation_time: 1000.0,
            num_cores: match platform {
                NeuromorphicPlatform::Loihi => 128,
                NeuromorphicPlatform::TrueNorth => 4096,
                NeuromorphicPlatform::SpiNNaker => 18,
                _ => 1,
            },
            neurons_per_core: match platform {
                NeuromorphicPlatform::Loihi => 1024,
                NeuromorphicPlatform::TrueNorth => 256,
                NeuromorphicPlatform::SpiNNaker => 17424,
                _ => 1000,
            },
            synapses_per_core: match platform {
                NeuromorphicPlatform::Loihi => 1024 * 1024,
                NeuromorphicPlatform::TrueNorth => 256 * 256,
                NeuromorphicPlatform::SpiNNaker => 17424 * 1000,
                _ => 1000000,
            },
            enable_plasticity: true,
            power_optimization: true,
        };
        Self::new(config)
    }

    /// Add a spiking neural network
    pub fn add_network(&mut self, name: String, network: SpikingNeuralNetwork) {
        self.networks.insert(name, network);
    }

    /// Process input spikes
    pub fn process_spikes(&mut self, input_spikes: &[SpikeEvent]) -> Result<Vec<SpikeEvent>> {
        // Add input spikes to event buffer
        for &spike in input_spikes {
            self.event_buffer.add_event(spike);
            self.energy_tracker.record_spike();
        }

        let mut output_spikes = Vec::new();

        // Process each network
        for network in self.networks.values_mut() {
            let network_output = network.process_time_step(input_spikes, self.config.time_step)?;
            output_spikes.extend(network_output);
        }

        // Update time
        self.event_buffer
            .update_time(self.event_buffer.current_time + self.config.time_step);

        Ok(output_spikes)
    }

    /// Convert tensor input to spike trains
    pub fn tensor_to_spikes(
        &self,
        input: &Tensor,
        encoding: SpikeEncoding,
    ) -> Result<Vec<SpikeEvent>> {
        let spike_encoder = SpikeEncoder::new(encoding);
        spike_encoder.encode(input, self.config.time_step)
    }

    /// Convert spike trains to tensor output
    pub fn spikes_to_tensor(&self, spikes: &[SpikeEvent], output_size: usize) -> Result<Tensor> {
        let spike_decoder = SpikeDecoder::new(SpikeDecoding::RateCode);
        spike_decoder.decode(spikes, output_size, self.config.simulation_time)
    }

    /// Run simulation for specified time
    pub fn simulate(
        &mut self,
        duration: f64,
        input_pattern: &[SpikeEvent],
    ) -> Result<Vec<SpikeEvent>> {
        let mut all_output_spikes = Vec::new();
        let steps = (duration / self.config.time_step) as usize;

        for step in 0..steps {
            let step_time = step as f64 * self.config.time_step;

            // Get input spikes for this time step
            let step_input: Vec<SpikeEvent> = input_pattern
                .iter()
                .filter(|spike| {
                    let time_diff = (spike.timestamp - step_time).abs();
                    time_diff < self.config.time_step / 2.0
                })
                .cloned()
                .collect();

            // Process time step
            let output_spikes = self.process_spikes(&step_input)?;
            all_output_spikes.extend(output_spikes);
        }

        Ok(all_output_spikes)
    }

    /// Get energy consumption statistics
    pub fn get_energy_stats(&self) -> &EnergyTracker {
        &self.energy_tracker
    }

    /// Optimize for low power consumption
    pub fn optimize_power(&mut self) {
        if self.config.power_optimization {
            // Implement power optimization strategies
            for network in self.networks.values_mut() {
                network.enable_power_gating();
                network.adjust_firing_thresholds(1.1); // Increase thresholds to reduce firing
            }
        }
    }

    /// Set hardware interface
    pub fn set_hardware_interface(&mut self, interface: Box<dyn NeuromorphicHardware>) {
        self.hardware_interface = Some(interface);
    }

    /// Deploy to hardware if available
    pub fn deploy_to_hardware(&mut self) -> Result<()> {
        if let Some(ref mut hardware) = self.hardware_interface {
            for (name, network) in &self.networks {
                hardware.deploy_network(name, network)?;
            }
            hardware.configure_platform(&self.config)?;
        }
        Ok(())
    }
}

impl Default for EnergyTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl EnergyTracker {
    /// Create a new energy tracker
    pub fn new() -> Self {
        Self {
            spike_energy: 0.0,
            synaptic_energy: 0.0,
            leakage_energy: 0.0,
            total_spikes: 0,
            total_synaptic_operations: 0,
        }
    }

    /// Record a spike event
    pub fn record_spike(&mut self) {
        self.total_spikes += 1;
        self.spike_energy += 1e-12; // 1 pJ per spike (typical neuromorphic)
    }

    /// Record synaptic operation
    pub fn record_synaptic_op(&mut self) {
        self.total_synaptic_operations += 1;
        self.synaptic_energy += 0.1e-12; // 0.1 pJ per synaptic operation
    }

    /// Update leakage energy
    pub fn update_leakage(&mut self, time_step: f64, num_neurons: usize) {
        // Leakage power for neuromorphic chips is very low
        self.leakage_energy += time_step * num_neurons as f64 * 1e-15; // 1 fJ per neuron per time step
    }

    /// Get total energy consumption
    pub fn total_energy(&self) -> f64 {
        self.spike_energy + self.synaptic_energy + self.leakage_energy
    }

    /// Get energy efficiency (operations per joule)
    pub fn energy_efficiency(&self) -> f64 {
        let total_ops = self.total_spikes + self.total_synaptic_operations;
        if self.total_energy() > 0.0 {
            total_ops as f64 / self.total_energy()
        } else {
            0.0
        }
    }
}

impl Default for NeuromorphicConfig {
    fn default() -> Self {
        Self {
            platform: NeuromorphicPlatform::Simulation,
            time_step: 1.0,
            simulation_time: 1000.0,
            num_cores: 1,
            neurons_per_core: 1000,
            synapses_per_core: 1000000,
            enable_plasticity: true,
            power_optimization: false,
        }
    }
}

/// Convert classical neural network to spiking neural network
pub fn convert_to_spiking(
    classical_weights: &Tensor,
    conversion_method: ConversionMethod,
) -> Result<SpikingNeuralNetwork> {
    match conversion_method {
        ConversionMethod::RateCoding => convert_rate_coding(classical_weights),
        ConversionMethod::TemporalCoding => convert_temporal_coding(classical_weights),
        ConversionMethod::Hybrid => convert_hybrid_coding(classical_weights),
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ConversionMethod {
    RateCoding,
    TemporalCoding,
    Hybrid,
}

fn convert_rate_coding(weights: &Tensor) -> Result<SpikingNeuralNetwork> {
    // Simplified conversion for demonstration
    let weight_data = weights.data()?;
    let num_neurons = weights.shape()[0];

    let mut network = SpikingNeuralNetwork::new(num_neurons);

    // Convert weights to synaptic connections
    for (i, &weight) in weight_data.iter().enumerate() {
        let source = i % num_neurons;
        let target = (i + 1) % num_neurons;
        network.add_synapse(source, target, weight, 1.0)?;
    }

    Ok(network)
}

fn convert_temporal_coding(weights: &Tensor) -> Result<SpikingNeuralNetwork> {
    // Temporal coding conversion (placeholder)
    convert_rate_coding(weights)
}

fn convert_hybrid_coding(weights: &Tensor) -> Result<SpikingNeuralNetwork> {
    // Hybrid coding conversion (placeholder)
    convert_rate_coding(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_event_creation() {
        let spike = SpikeEvent::new(5, 10.5, 0.8);
        assert_eq!(spike.neuron_id, 5);
        assert_eq!(spike.timestamp, 10.5);
        assert_eq!(spike.weight, 0.8);
    }

    #[test]
    fn test_spike_event_activity() {
        let spike = SpikeEvent::new(0, 10.0, 1.0);
        assert!(spike.is_active_at(10.0, 0.1));
        assert!(spike.is_active_at(10.05, 0.1));
        assert!(!spike.is_active_at(10.2, 0.1));
    }

    #[test]
    fn test_event_buffer() {
        let mut buffer = EventBuffer::new(3);
        assert!(buffer.is_empty());

        buffer.add_event(SpikeEvent::new(0, 1.0, 1.0));
        buffer.add_event(SpikeEvent::new(1, 2.0, 1.0));
        buffer.add_event(SpikeEvent::new(2, 3.0, 1.0));
        assert_eq!(buffer.len(), 3);

        // Adding one more should remove the first
        buffer.add_event(SpikeEvent::new(3, 4.0, 1.0));
        assert_eq!(buffer.len(), 3);

        let events = buffer.get_events_in_window(2.0, 4.0);
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_neuromorphic_processor_creation() {
        let processor = NeuromorphicProcessor::with_platform(NeuromorphicPlatform::Loihi);
        assert_eq!(processor.config.platform, NeuromorphicPlatform::Loihi);
        assert_eq!(processor.config.num_cores, 128);
        assert_eq!(processor.config.neurons_per_core, 1024);
    }

    #[test]
    fn test_energy_tracker() {
        let mut tracker = EnergyTracker::new();
        assert_eq!(tracker.total_spikes, 0);
        assert_eq!(tracker.total_energy(), 0.0);

        tracker.record_spike();
        assert_eq!(tracker.total_spikes, 1);
        assert!(tracker.total_energy() > 0.0);

        tracker.record_synaptic_op();
        assert_eq!(tracker.total_synaptic_operations, 1);

        let efficiency = tracker.energy_efficiency();
        assert!(efficiency > 0.0);
    }

    #[test]
    fn test_platform_configurations() {
        let platforms = [
            NeuromorphicPlatform::Loihi,
            NeuromorphicPlatform::TrueNorth,
            NeuromorphicPlatform::SpiNNaker,
            NeuromorphicPlatform::Simulation,
        ];

        for platform in &platforms {
            let processor = NeuromorphicProcessor::with_platform(*platform);
            assert_eq!(processor.config.platform, *platform);
            assert!(processor.config.neurons_per_core > 0);
            assert!(processor.config.synapses_per_core > 0);
        }
    }

    #[test]
    fn test_neuromorphic_config_default() {
        let config = NeuromorphicConfig::default();
        assert_eq!(config.platform, NeuromorphicPlatform::Simulation);
        assert_eq!(config.time_step, 1.0);
        assert!(config.enable_plasticity);
        assert!(!config.power_optimization);
    }

    #[test]
    fn test_conversion_methods() {
        let weights =
            Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]).expect("Tensor from_vec failed");

        let methods = [
            ConversionMethod::RateCoding,
            ConversionMethod::TemporalCoding,
            ConversionMethod::Hybrid,
        ];

        for method in &methods {
            let result = convert_to_spiking(&weights, *method);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_event_buffer_time_window() {
        let mut buffer = EventBuffer::new(10);

        buffer.add_event(SpikeEvent::new(0, 1.0, 1.0));
        buffer.add_event(SpikeEvent::new(1, 5.0, 1.0));
        buffer.add_event(SpikeEvent::new(2, 10.0, 1.0));

        let events_early = buffer.get_events_in_window(0.0, 3.0);
        assert_eq!(events_early.len(), 1);

        let events_mid = buffer.get_events_in_window(3.0, 7.0);
        assert_eq!(events_mid.len(), 1);

        let events_all = buffer.get_events_in_window(0.0, 15.0);
        assert_eq!(events_all.len(), 3);
    }

    #[test]
    fn test_energy_leakage_update() {
        let mut tracker = EnergyTracker::new();
        let initial_energy = tracker.total_energy();

        tracker.update_leakage(1.0, 1000);
        assert!(tracker.total_energy() > initial_energy);
        assert!(tracker.leakage_energy > 0.0);
    }
}
