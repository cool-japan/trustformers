//! Spike encoding and decoding for neuromorphic computing

#![allow(unused_variables)] // Neuromorphic encoding

use crate::neuromorphic::SpikeEvent;
use crate::tensor::Tensor;
use anyhow::Result;
use scirs2_core::random::*;

/// Spike encoding schemes
#[derive(Debug, Clone, Copy)]
pub enum SpikeEncoding {
    RateCode,
    TemporalCode,
    PopulationCode,
    RankOrderCode,
    DeltaCode,
}

/// Spike decoding schemes
#[derive(Debug, Clone, Copy)]
pub enum SpikeDecoding {
    RateCode,
    FirstSpike,
    PopulationVector,
    TemporalPattern,
}

/// Spike encoder
#[derive(Debug, Clone)]
pub struct SpikeEncoder {
    encoding: SpikeEncoding,
    time_window: f64,
    max_frequency: f64,
}

/// Spike decoder
#[derive(Debug, Clone)]
pub struct SpikeDecoder {
    decoding: SpikeDecoding,
}

impl SpikeEncoder {
    pub fn new(encoding: SpikeEncoding) -> Self {
        Self {
            encoding,
            time_window: 100.0,
            max_frequency: 100.0,
        }
    }

    pub fn encode(&self, input: &Tensor, time_step: f64) -> Result<Vec<SpikeEvent>> {
        let data = input.data()?;
        match self.encoding {
            SpikeEncoding::RateCode => self.rate_encode(&data, time_step),
            SpikeEncoding::TemporalCode => self.temporal_encode(&data, time_step),
            SpikeEncoding::PopulationCode => self.population_encode(&data, time_step),
            _ => self.rate_encode(&data, time_step), // Default to rate coding
        }
    }

    fn rate_encode(&self, data: &[f32], time_step: f64) -> Result<Vec<SpikeEvent>> {
        let mut spikes = Vec::new();
        let steps = (self.time_window / time_step) as usize;
        let mut rng = thread_rng();

        for (neuron_id, &value) in data.iter().enumerate() {
            let frequency = value.abs() * (self.max_frequency as f32);
            let spike_probability = frequency * (time_step as f32) / 1000.0;

            for step in 0..steps {
                if rng.random::<f64>() < spike_probability as f64 {
                    let timestamp = step as f64 * time_step;
                    spikes.push(SpikeEvent::new(neuron_id, timestamp, value));
                }
            }
        }

        Ok(spikes)
    }

    fn temporal_encode(&self, data: &[f32], time_step: f64) -> Result<Vec<SpikeEvent>> {
        let mut spikes = Vec::new();

        for (neuron_id, &value) in data.iter().enumerate() {
            // Temporal coding: higher values spike earlier
            let normalized_value = (value + 1.0) / 2.0; // Normalize to [0,1]
            let spike_time = (1.0 - normalized_value) as f64 * self.time_window;
            spikes.push(SpikeEvent::new(neuron_id, spike_time, 1.0));
        }

        Ok(spikes)
    }

    fn population_encode(&self, data: &[f32], time_step: f64) -> Result<Vec<SpikeEvent>> {
        let mut spikes = Vec::new();
        let population_size = 10; // Use 10 neurons per input value
        let mut rng = thread_rng();

        for (input_id, &value) in data.iter().enumerate() {
            let normalized_value = (value + 1.0) / 2.0; // Normalize to [0,1]

            for pop_neuron in 0..population_size {
                let center = pop_neuron as f64 / population_size as f64;
                let sigma = 0.1;
                let activation =
                    (-0.5 * ((normalized_value as f64 - center) / sigma).powi(2)).exp();

                if activation > 0.5 && rng.random::<f64>() < activation {
                    let neuron_id = input_id * population_size + pop_neuron;
                    spikes.push(SpikeEvent::new(neuron_id, 0.0, activation as f32));
                }
            }
        }

        Ok(spikes)
    }
}

impl SpikeDecoder {
    pub fn new(decoding: SpikeDecoding) -> Self {
        Self { decoding }
    }

    pub fn decode(
        &self,
        spikes: &[SpikeEvent],
        output_size: usize,
        time_window: f64,
    ) -> Result<Tensor> {
        match self.decoding {
            SpikeDecoding::RateCode => self.rate_decode(spikes, output_size, time_window),
            SpikeDecoding::FirstSpike => self.first_spike_decode(spikes, output_size, time_window),
            SpikeDecoding::PopulationVector => {
                self.population_decode(spikes, output_size, time_window)
            },
            _ => self.rate_decode(spikes, output_size, time_window),
        }
    }

    fn rate_decode(
        &self,
        spikes: &[SpikeEvent],
        output_size: usize,
        time_window: f64,
    ) -> Result<Tensor> {
        let mut counts = vec![0.0f32; output_size];

        for spike in spikes {
            if spike.neuron_id < output_size {
                counts[spike.neuron_id] += 1.0;
            }
        }

        // Convert to rates (spikes per second)
        for count in &mut counts {
            *count /= (time_window / 1000.0) as f32;
        }

        Ok(Tensor::from_vec(counts, &[output_size])?)
    }

    fn first_spike_decode(
        &self,
        spikes: &[SpikeEvent],
        output_size: usize,
        time_window: f64,
    ) -> Result<Tensor> {
        let mut first_spikes = vec![time_window as f32; output_size];

        for spike in spikes {
            if spike.neuron_id < output_size
                && spike.timestamp < first_spikes[spike.neuron_id] as f64
            {
                first_spikes[spike.neuron_id] = spike.timestamp as f32;
            }
        }

        // Convert to activation (earlier spike = higher activation)
        for time in &mut first_spikes {
            *time = 1.0 - (*time / time_window as f32);
        }

        Ok(Tensor::from_vec(first_spikes, &[output_size])?)
    }

    fn population_decode(
        &self,
        spikes: &[SpikeEvent],
        output_size: usize,
        time_window: f64,
    ) -> Result<Tensor> {
        // Simplified population decoding
        self.rate_decode(spikes, output_size, time_window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_encoder_creation() {
        let encoder = SpikeEncoder::new(SpikeEncoding::RateCode);
        assert!(matches!(encoder.encoding, SpikeEncoding::RateCode));
        assert_eq!(encoder.time_window, 100.0);
        assert_eq!(encoder.max_frequency, 100.0);
    }

    #[test]
    fn test_rate_encoding() {
        let encoder = SpikeEncoder::new(SpikeEncoding::RateCode);
        let input = Tensor::from_vec(vec![0.5, 1.0, 0.0], &[3]).expect("Tensor from_vec failed");

        let spikes = encoder.encode(&input, 1.0).expect("Encoding failed");
        assert!(!spikes.is_empty());

        // Check that spikes are generated for non-zero inputs
        let has_neuron_0 = spikes.iter().any(|s| s.neuron_id == 0);
        let has_neuron_1 = spikes.iter().any(|s| s.neuron_id == 1);

        assert!(has_neuron_0 || has_neuron_1); // At least one should have spikes
    }

    #[test]
    fn test_temporal_encoding() {
        let encoder = SpikeEncoder::new(SpikeEncoding::TemporalCode);
        let input = Tensor::from_vec(vec![1.0, 0.0, -1.0], &[3]).expect("Tensor from_vec failed");

        let spikes = encoder.encode(&input, 1.0).expect("Encoding failed");
        assert_eq!(spikes.len(), 3); // One spike per input

        // Higher values should spike earlier
        let spike_0 = spikes.iter().find(|s| s.neuron_id == 0).unwrap();
        let spike_2 = spikes.iter().find(|s| s.neuron_id == 2).unwrap();
        assert!(spike_0.timestamp < spike_2.timestamp);
    }

    #[test]
    fn test_population_encoding() {
        let encoder = SpikeEncoder::new(SpikeEncoding::PopulationCode);
        let input = Tensor::from_vec(vec![0.5], &[1]).expect("Tensor from_vec failed");

        let spikes = encoder.encode(&input, 1.0).expect("Encoding failed");
        assert!(!spikes.is_empty());
    }

    #[test]
    fn test_spike_decoder_creation() {
        let decoder = SpikeDecoder::new(SpikeDecoding::RateCode);
        assert!(matches!(decoder.decoding, SpikeDecoding::RateCode));
    }

    #[test]
    fn test_rate_decoding() {
        let decoder = SpikeDecoder::new(SpikeDecoding::RateCode);
        let spikes = vec![
            SpikeEvent::new(0, 1.0, 1.0),
            SpikeEvent::new(0, 2.0, 1.0),
            SpikeEvent::new(1, 3.0, 1.0),
        ];

        let result = decoder.decode(&spikes, 3, 100.0).expect("Decoding failed");
        let data = result.data().unwrap();

        assert_eq!(data.len(), 3);
        assert!(data[0] > data[1]); // Neuron 0 had more spikes
        assert_eq!(data[2], 0.0); // Neuron 2 had no spikes
    }

    #[test]
    fn test_first_spike_decoding() {
        let decoder = SpikeDecoder::new(SpikeDecoding::FirstSpike);
        let spikes = vec![
            SpikeEvent::new(0, 10.0, 1.0),
            SpikeEvent::new(1, 5.0, 1.0),
            SpikeEvent::new(0, 20.0, 1.0), // Later spike, should be ignored
        ];

        let result = decoder.decode(&spikes, 3, 100.0).expect("Decoding failed");
        let data = result.data().unwrap();

        assert_eq!(data.len(), 3);
        assert!(data[1] > data[0]); // Neuron 1 spiked earlier, higher activation
        assert_eq!(data[2], 0.0); // Neuron 2 never spiked
    }

    #[test]
    fn test_encoding_decoding_roundtrip() {
        let encoder = SpikeEncoder::new(SpikeEncoding::RateCode);
        let decoder = SpikeDecoder::new(SpikeDecoding::RateCode);

        let input = Tensor::from_vec(vec![0.0, 0.5, 1.0], &[3]).expect("Tensor from_vec failed");

        // Run multiple trials to account for randomness in spike encoding
        let mut monotonic_count = 0;
        let trials = 10;

        for _ in 0..trials {
            let spikes = encoder.encode(&input, 1.0).expect("Encoding failed");
            let output = decoder.decode(&spikes, 3, 100.0).expect("Decoding failed");

            assert_eq!(output.shape(), &[3]);

            let output_data = output.data().unwrap();

            // Check if the relationship is monotonic (higher input -> higher output rate)
            if output_data[1] > 0.0 && output_data[2] > 0.0 {
                if output_data[2] >= output_data[1] && output_data[1] >= output_data[0] {
                    monotonic_count += 1;
                }
            } else if output_data[1] == 0.0 && output_data[2] > 0.0 {
                // When middle value is 0, just check that highest input gives highest output
                if output_data[2] >= output_data[0] {
                    monotonic_count += 1;
                }
            } else {
                monotonic_count += 1; // Count as success when outputs are zero
            }
        }

        // Due to probabilistic nature, expect monotonic relationship in most trials
        // Allow for some variance due to random spike generation
        assert!(
            monotonic_count >= trials / 2,
            "Monotonic relationship should hold in at least half the trials, got {}/{}",
            monotonic_count,
            trials
        );
    }
}
