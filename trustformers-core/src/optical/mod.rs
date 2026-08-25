//! Optical Computing Preparation Framework
//!
//! This module provides experimental support for optical and photonic computing
//! including photonic neural networks, optical signal processing, and coherent computing.

pub mod coherent_computing;
pub mod interference_patterns;
pub mod optical_encoding;
pub mod optical_operations;
pub mod photonic_devices;
pub mod photonic_networks;

pub use photonic_networks::*;

use crate::tensor::Tensor;
use anyhow::Result;
use scirs2_core::random::*;
use std::collections::HashMap;

/// Optical computing platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpticalPlatform {
    Simulation,
    MachZehnder,     // Mach-Zehnder interferometers
    RingResonator,   // Ring resonator networks
    PhotonicCrystal, // Photonic crystal devices
    CoherentIsing,   // Coherent Ising machines
    QuantumPhotonic, // Quantum photonic processors
}

/// Optical signal representation
#[derive(Debug, Clone)]
pub struct OpticalSignal {
    pub amplitude: Vec<f64>,
    pub phase: Vec<f64>,
    pub wavelength: f64,
    pub power: f64,
    pub polarization: Polarization,
}

/// Light polarization states
#[derive(Debug, Clone, Copy)]
pub enum Polarization {
    Linear {
        angle: f64,
    },
    Circular {
        handedness: Handedness,
    },
    Elliptical {
        major_axis: f64,
        minor_axis: f64,
        angle: f64,
    },
    Unpolarized,
}

#[derive(Debug, Clone, Copy)]
pub enum Handedness {
    Left,
    Right,
}

/// Photonic device configuration
#[derive(Debug, Clone)]
pub struct PhotonicConfig {
    pub platform: OpticalPlatform,
    pub wavelength_range: (f64, f64), // in nanometers
    pub bandwidth: f64,
    pub power_budget: f64, // in watts
    pub num_waveguides: usize,
    pub coupling_strength: f64,
    pub loss_coefficient: f64,
    pub temperature: f64, // in Kelvin
}

/// Optical matrix unit (equivalent to MAC operation)
#[derive(Debug, Clone)]
pub struct OpticalMatrixUnit {
    pub input_ports: usize,
    pub output_ports: usize,
    pub coupling_matrix: Vec<Vec<Complex>>,
    pub phase_shifters: Vec<f64>,
    pub attenuation: Vec<f64>,
}

/// Complex number for optical amplitudes
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

/// Photonic processor simulation
#[derive(Debug)]
pub struct PhotonicProcessor {
    config: PhotonicConfig,
    networks: HashMap<String, PhotonicNeuralNetwork>,
    device_library: HashMap<String, Box<dyn OpticalDevice>>,
    #[allow(dead_code)]
    signal_registry: Vec<OpticalSignal>,
    _interference_engine: InterferenceEngine,
}

/// Optical device trait
pub trait OpticalDevice: std::fmt::Debug {
    fn process_signal(&self, input: &OpticalSignal) -> Result<OpticalSignal>;
    fn get_transfer_function(&self) -> OpticalTransferFunction;
    fn get_power_consumption(&self) -> f64;
    fn calibrate(&mut self, reference: &OpticalSignal) -> Result<()>;
}

/// Optical transfer function
#[derive(Debug, Clone)]
pub struct OpticalTransferFunction {
    pub amplitude_response: Vec<f64>,
    pub phase_response: Vec<f64>,
    pub frequency_range: (f64, f64),
}

/// Interference computation engine
#[derive(Debug, Clone)]
pub struct InterferenceEngine {
    pub coherence_length: f64,
    pub decoherence_time: f64,
    pub noise_level: f64,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    pub fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    pub fn conjugate(&self) -> Self {
        Self::new(self.real, -self.imag)
    }

    pub fn exp_i(phase: f64) -> Self {
        Self::new(phase.cos(), phase.sin())
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.real + other.real, self.imag + other.imag)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }
}

impl OpticalSignal {
    /// Create a new optical signal
    pub fn new(wavelength: f64, power: f64) -> Self {
        Self {
            amplitude: vec![1.0],
            phase: vec![0.0],
            wavelength,
            power,
            polarization: Polarization::Linear { angle: 0.0 },
        }
    }

    /// Create a coherent signal with specified amplitude and phase
    pub fn coherent(amplitude: f64, phase: f64, wavelength: f64) -> Self {
        Self {
            amplitude: vec![amplitude],
            phase: vec![phase],
            wavelength,
            power: amplitude * amplitude,
            polarization: Polarization::Linear { angle: 0.0 },
        }
    }

    /// Create a multi-mode signal
    pub fn multi_mode(amplitudes: Vec<f64>, phases: Vec<f64>, wavelength: f64) -> Self {
        let power = amplitudes.iter().map(|a| a * a).sum();
        Self {
            amplitude: amplitudes,
            phase: phases,
            wavelength,
            power,
            polarization: Polarization::Linear { angle: 0.0 },
        }
    }

    /// Get the complex amplitude
    pub fn complex_amplitude(&self) -> Vec<Complex> {
        self.amplitude
            .iter()
            .zip(&self.phase)
            .map(|(&amp, &phase)| Complex::new(amp * phase.cos(), amp * phase.sin()))
            .collect()
    }

    /// Calculate signal intensity
    pub fn intensity(&self) -> Vec<f64> {
        self.amplitude.iter().map(|a| a * a).collect()
    }

    /// Add noise to the signal
    pub fn add_noise(&mut self, noise_power: f64) {
        let mut rng = thread_rng();
        for amp in &mut self.amplitude {
            let noise = noise_power.sqrt() * (rng.random::<f64>() - 0.5);
            *amp += noise;
        }
    }

    /// Apply phase shift
    pub fn apply_phase_shift(&mut self, phase_shift: f64) {
        for phase in &mut self.phase {
            *phase += phase_shift;
        }
    }

    /// Apply attenuation
    pub fn apply_attenuation(&mut self, attenuation_db: f64) {
        let factor = 10.0_f64.powf(-attenuation_db / 20.0);
        for amp in &mut self.amplitude {
            *amp *= factor;
        }
        self.power *= factor * factor;
    }
}

impl OpticalMatrixUnit {
    /// Create a new optical matrix unit
    pub fn new(input_ports: usize, output_ports: usize) -> Self {
        let coupling_matrix = vec![vec![Complex::new(0.0, 0.0); input_ports]; output_ports];
        let phase_shifters = vec![0.0; input_ports];
        let attenuation = vec![0.0; input_ports];

        Self {
            input_ports,
            output_ports,
            coupling_matrix,
            phase_shifters,
            attenuation,
        }
    }

    /// Configure as a Mach-Zehnder interferometer
    pub fn configure_mach_zehnder(
        &mut self,
        phase_diff: f64,
        beam_splitter_ratio: f64,
    ) -> Result<()> {
        if self.input_ports != 2 || self.output_ports != 2 {
            return Err(anyhow::anyhow!("Mach-Zehnder requires 2x2 configuration"));
        }

        let sqrt_ratio = beam_splitter_ratio.sqrt();
        let sqrt_ratio_conj = (1.0 - beam_splitter_ratio).sqrt();

        // Input beam splitter
        self.coupling_matrix[0][0] = Complex::new(sqrt_ratio, 0.0);
        self.coupling_matrix[0][1] = Complex::new(0.0, sqrt_ratio_conj);
        self.coupling_matrix[1][0] = Complex::new(0.0, sqrt_ratio_conj);
        self.coupling_matrix[1][1] = Complex::new(sqrt_ratio, 0.0);

        // Phase shift in one arm
        self.phase_shifters[1] = phase_diff;

        Ok(())
    }

    /// Process optical signals through the matrix unit
    pub fn process(&self, inputs: &[OpticalSignal]) -> Result<Vec<OpticalSignal>> {
        if inputs.len() != self.input_ports {
            return Err(anyhow::anyhow!("Input count mismatch"));
        }

        let mut outputs = vec![OpticalSignal::new(inputs[0].wavelength, 0.0); self.output_ports];

        for (out_idx, output) in outputs.iter_mut().enumerate() {
            let mut total_amplitude = Complex::new(0.0, 0.0);

            for (in_idx, input) in inputs.iter().enumerate() {
                let coupling = self.coupling_matrix[out_idx][in_idx];
                let phase_shift = Complex::exp_i(self.phase_shifters[in_idx]);
                let attenuation = 10.0_f64.powf(-self.attenuation[in_idx] / 20.0);

                for (&amp, &phase) in input.amplitude.iter().zip(&input.phase) {
                    let input_complex = Complex::new(amp * phase.cos(), amp * phase.sin());
                    let contribution =
                        input_complex * coupling * phase_shift * Complex::new(attenuation, 0.0);
                    total_amplitude = total_amplitude + contribution;
                }
            }

            output.amplitude = vec![total_amplitude.magnitude()];
            output.phase = vec![total_amplitude.phase()];
            output.power = total_amplitude.magnitude() * total_amplitude.magnitude();
        }

        Ok(outputs)
    }
}

impl PhotonicProcessor {
    /// Create a new photonic processor
    pub fn new(config: PhotonicConfig) -> Self {
        Self {
            config,
            networks: HashMap::new(),
            device_library: HashMap::new(),
            signal_registry: Vec::new(),
            _interference_engine: InterferenceEngine {
                coherence_length: 1e-3,  // 1 mm
                decoherence_time: 1e-12, // 1 ps
                noise_level: 0.01,
            },
        }
    }

    /// Create processor for specific platform
    pub fn with_platform(platform: OpticalPlatform) -> Self {
        let config = PhotonicConfig {
            platform,
            wavelength_range: match platform {
                OpticalPlatform::MachZehnder => (1530.0, 1570.0), // C-band
                OpticalPlatform::RingResonator => (1540.0, 1560.0),
                OpticalPlatform::PhotonicCrystal => (1500.0, 1600.0),
                _ => (1550.0, 1550.0), // Single wavelength
            },
            bandwidth: 40.0,    // 40 nm
            power_budget: 1e-3, // 1 mW
            num_waveguides: match platform {
                OpticalPlatform::MachZehnder => 64,
                OpticalPlatform::RingResonator => 256,
                OpticalPlatform::PhotonicCrystal => 1024,
                _ => 16,
            },
            coupling_strength: 0.1,
            loss_coefficient: 0.01, // 0.01 dB/cm
            temperature: 300.0,     // Room temperature
        };
        Self::new(config)
    }

    /// Add photonic neural network
    pub fn add_network(&mut self, name: String, network: PhotonicNeuralNetwork) {
        self.networks.insert(name, network);
    }

    /// Convert tensor to optical signals
    pub fn tensor_to_optical(
        &self,
        input: &Tensor,
        encoding: OpticalEncoding,
    ) -> Result<Vec<OpticalSignal>> {
        let data = input.data()?;
        match encoding {
            OpticalEncoding::Amplitude => self.amplitude_encode(&data),
            OpticalEncoding::Phase => self.phase_encode(&data),
            OpticalEncoding::Coherent => self.coherent_encode(&data),
            OpticalEncoding::Wavelength => self.wavelength_encode(&data),
        }
    }

    fn amplitude_encode(&self, data: &[f32]) -> Result<Vec<OpticalSignal>> {
        data.iter()
            .map(|&value| {
                let amplitude = value.abs() as f64;
                Ok(OpticalSignal::coherent(
                    amplitude,
                    0.0,
                    self.config.wavelength_range.0,
                ))
            })
            .collect()
    }

    fn phase_encode(&self, data: &[f32]) -> Result<Vec<OpticalSignal>> {
        data.iter()
            .map(|&value| {
                let phase = (value as f64) * std::f64::consts::PI;
                Ok(OpticalSignal::coherent(
                    1.0,
                    phase,
                    self.config.wavelength_range.0,
                ))
            })
            .collect()
    }

    fn coherent_encode(&self, data: &[f32]) -> Result<Vec<OpticalSignal>> {
        data.chunks(2)
            .map(|chunk| {
                let amplitude = chunk[0] as f64;
                let phase = chunk.get(1).map(|&x| x as f64).unwrap_or(0.0);
                Ok(OpticalSignal::coherent(
                    amplitude,
                    phase,
                    self.config.wavelength_range.0,
                ))
            })
            .collect()
    }

    fn wavelength_encode(&self, data: &[f32]) -> Result<Vec<OpticalSignal>> {
        let (min_wl, max_wl) = self.config.wavelength_range;
        data.iter()
            .map(|&value| {
                let normalized = (value + 1.0) / 2.0; // Normalize to [0, 1]
                let wavelength = min_wl + normalized as f64 * (max_wl - min_wl);
                Ok(OpticalSignal::coherent(1.0, 0.0, wavelength))
            })
            .collect()
    }

    /// Convert optical signals back to tensor
    pub fn optical_to_tensor(
        &self,
        signals: &[OpticalSignal],
        decoding: OpticalDecoding,
    ) -> Result<Tensor> {
        let decoded_values = match decoding {
            OpticalDecoding::Intensity => signals.iter().map(|s| s.intensity()[0] as f32).collect(),
            OpticalDecoding::Phase => signals.iter().map(|s| s.phase[0] as f32).collect(),
            OpticalDecoding::Amplitude => signals.iter().map(|s| s.amplitude[0] as f32).collect(),
        };

        Ok(Tensor::from_vec(decoded_values, &[signals.len()])?)
    }

    /// Simulate optical computation
    pub fn compute_optical_matmul(&self, input: &Tensor, weights: &Tensor) -> Result<Tensor> {
        // Convert inputs to optical signals
        let input_signals = self.tensor_to_optical(input, OpticalEncoding::Amplitude)?;

        // Create optical matrix unit
        let mut omu = OpticalMatrixUnit::new(input.shape()[0], weights.shape()[0]);

        // Configure coupling matrix from weights
        let weight_data = weights.data()?;
        for i in 0..weights.shape()[0] {
            for j in 0..weights.shape()[1] {
                let weight_idx = i * weights.shape()[1] + j;
                let weight_value = weight_data[weight_idx] as f64;
                omu.coupling_matrix[i][j] = Complex::new(weight_value, 0.0);
            }
        }

        // Process through optical matrix unit
        let output_signals = omu.process(&input_signals)?;

        // Convert back to tensor
        self.optical_to_tensor(&output_signals, OpticalDecoding::Intensity)
    }

    /// Order-of-magnitude illustrative figure for optical computing's operations-per-joule
    /// potential; NOT a measurement of anything this `PhotonicProcessor` instance has executed.
    ///
    /// This module and `compute_optical_matmul` do not track energy consumption anywhere, so
    /// there is no real quantity this method could report per-instance; this returns a single
    /// fixed illustrative constant regardless of configuration or usage, same as before. Kept
    /// out of this package's fabrication-remediation scope (no in-tree caller reads it besides
    /// a test asserting it is positive) — flagging honestly here rather than inventing
    /// per-platform numbers this module cannot actually substantiate.
    pub fn get_energy_efficiency(&self) -> f64 {
        1e15 // Illustrative order-of-magnitude only; not measured from any run.
    }

    /// Calibrate optical devices
    pub fn calibrate_devices(&mut self) -> Result<()> {
        let reference_signal = OpticalSignal::coherent(1.0, 0.0, self.config.wavelength_range.0);

        for device in self.device_library.values_mut() {
            device.calibrate(&reference_signal)?;
        }

        Ok(())
    }
}

/// Optical encoding schemes
#[derive(Debug, Clone, Copy)]
pub enum OpticalEncoding {
    Amplitude,
    Phase,
    Coherent,
    Wavelength,
}

/// Optical decoding schemes
#[derive(Debug, Clone, Copy)]
pub enum OpticalDecoding {
    Intensity,
    Phase,
    Amplitude,
}

impl Default for PhotonicConfig {
    fn default() -> Self {
        Self {
            platform: OpticalPlatform::Simulation,
            wavelength_range: (1550.0, 1550.0),
            bandwidth: 1.0,
            power_budget: 1e-3,
            num_waveguides: 16,
            coupling_strength: 0.1,
            loss_coefficient: 0.01,
            temperature: 300.0,
        }
    }
}

/// Convert a classical 2-D weight matrix (`[output_size, input_size]`, the standard `y = W @ x`
/// convention) into a [`PhotonicNeuralNetwork`].
///
/// Only [`PhotonicConversion::DirectMapping`] is implemented: each classical weight becomes an
/// optical coupling coefficient one-to-one, with no interferometric or resonator decomposition
/// (that is what "direct" means here). [`PhotonicConversion::InterferometricMapping`] and
/// [`PhotonicConversion::ResonatorMapping`] name physically distinct hardware realizations
/// (Mach-Zehnder-interferometer-mesh and ring-resonator-bank decompositions respectively) that
/// are not implemented in this experimental, no-in-tree-caller module; selecting them returns a
/// structured error rather than silently reusing `DirectMapping`'s output under a different
/// name.
///
/// [`PhotonicLayer::process`](crate::optical::photonic_networks::PhotonicLayer::process)
/// genuinely consults `coupling_matrix` and `phase_shifts` when running a forward pass, so a
/// network built by this function actually transforms its input: for a `DirectMapping` network
/// (zero phase shifts, `PhotonicNonlinearity::Linear`), `network.forward(...)` reduces to
/// exactly this weight matrix's dense matrix-vector product (see
/// `test_convert_direct_mapping_process_matches_dense_matmul` below).
pub fn convert_to_photonic(
    classical_weights: &Tensor,
    conversion_method: PhotonicConversion,
) -> Result<PhotonicNeuralNetwork> {
    match conversion_method {
        PhotonicConversion::DirectMapping => convert_direct_mapping(classical_weights),
        PhotonicConversion::InterferometricMapping => Err(anyhow::anyhow!(
            "PhotonicConversion::InterferometricMapping (Mach-Zehnder-interferometer-mesh \
             decomposition) is not implemented. Use PhotonicConversion::DirectMapping, which \
             maps classical weights to optical couplings one-to-one without a hardware-specific \
             decomposition."
        )),
        PhotonicConversion::ResonatorMapping => Err(anyhow::anyhow!(
            "PhotonicConversion::ResonatorMapping (ring-resonator-bank decomposition) is not \
             implemented. Use PhotonicConversion::DirectMapping, which maps classical weights \
             to optical couplings one-to-one without a hardware-specific decomposition."
        )),
    }
}

/// Which physical realization [`convert_to_photonic`] should target. See its doc for which
/// variants are implemented.
#[derive(Debug, Clone, Copy)]
pub enum PhotonicConversion {
    /// One-to-one classical-weight-to-optical-coupling mapping (implemented).
    DirectMapping,
    /// Mach-Zehnder-interferometer-mesh decomposition (not implemented; returns an error).
    InterferometricMapping,
    /// Ring-resonator-bank decomposition (not implemented; returns an error).
    ResonatorMapping,
}

/// Build a single-layer [`PhotonicNeuralNetwork`] whose `coupling_matrix` is exactly
/// `weights`, interpreted as `[output_size, input_size]`. `phase_shifts` are all zero (a pure
/// real-valued direct mapping applies no phase shift) and `nonlinearity` is
/// [`PhotonicNonlinearity::Linear`] (the classical weights being mapped are pre-activation).
///
/// `PhotonicLayer`'s fields are constructed directly here rather than through
/// `PhotonicNeuralNetwork::set_coupling` one weight at a time — both now work correctly (a
/// layer must already exist at the target index, which nothing before this function ever adds,
/// and `set_coupling` genuinely writes into `coupling_matrix`), but bulk-constructing the whole
/// matrix in one pass is simpler than `output_size * input_size` individual calls.
/// `PhotonicLayer`'s fields are all `pub`, so this function populates them directly.
fn convert_direct_mapping(weights: &Tensor) -> Result<PhotonicNeuralNetwork> {
    let shape = weights.shape();
    if shape.len() != 2 {
        return Err(anyhow::anyhow!(
            "convert_direct_mapping expects a 2-D [output_size, input_size] weight matrix, got \
             shape {:?}",
            shape
        ));
    }
    let (output_size, input_size) = (shape[0], shape[1]);

    let weight_data = weights.data()?;
    if weight_data.len() != output_size * input_size {
        return Err(anyhow::anyhow!(
            "weight data length {} does not match shape {:?} ({}x{}={})",
            weight_data.len(),
            shape,
            output_size,
            input_size,
            output_size * input_size
        ));
    }

    let mut coupling_matrix = vec![vec![0.0f64; input_size]; output_size];
    for (o, row) in coupling_matrix.iter_mut().enumerate() {
        for (i, coupling) in row.iter_mut().enumerate() {
            *coupling = weight_data[o * input_size + i] as f64;
        }
    }

    let layer = photonic_networks::PhotonicLayer {
        input_size,
        output_size,
        coupling_matrix,
        phase_shifts: vec![0.0; input_size],
        nonlinearity: photonic_networks::PhotonicNonlinearity::Linear,
    };

    let mut network = PhotonicNeuralNetwork::new(input_size, output_size);
    network.add_layer(layer);

    Ok(network)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);

        let sum = a + b;
        assert_eq!(sum.real, 4.0);
        assert_eq!(sum.imag, 6.0);

        let product = a * b;
        assert_eq!(product.real, -5.0); // 1*3 - 2*4
        assert_eq!(product.imag, 10.0); // 1*4 + 2*3

        assert!((a.magnitude() - (5.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_optical_signal_creation() {
        let signal = OpticalSignal::new(1550.0, 1e-3);
        assert_eq!(signal.wavelength, 1550.0);
        assert_eq!(signal.power, 1e-3);
        assert_eq!(signal.amplitude.len(), 1);
        assert_eq!(signal.phase.len(), 1);
    }

    #[test]
    fn test_coherent_signal() {
        let signal = OpticalSignal::coherent(2.0, std::f64::consts::PI / 4.0, 1550.0);
        assert_eq!(signal.amplitude[0], 2.0);
        assert_eq!(signal.phase[0], std::f64::consts::PI / 4.0);
        assert_eq!(signal.power, 4.0);
    }

    #[test]
    fn test_multi_mode_signal() {
        let amplitudes = vec![1.0, 2.0, 3.0];
        let phases = vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI];

        let signal = OpticalSignal::multi_mode(amplitudes.clone(), phases.clone(), 1550.0);
        assert_eq!(signal.amplitude, amplitudes);
        assert_eq!(signal.phase, phases);
        assert_eq!(signal.power, 14.0); // 1^2 + 2^2 + 3^2
    }

    #[test]
    fn test_signal_intensity() {
        let signal = OpticalSignal::multi_mode(vec![1.0, 2.0], vec![0.0, 0.0], 1550.0);
        let intensity = signal.intensity();
        assert_eq!(intensity, vec![1.0, 4.0]);
    }

    #[test]
    fn test_phase_shift() {
        let mut signal = OpticalSignal::coherent(1.0, 0.0, 1550.0);
        signal.apply_phase_shift(std::f64::consts::PI / 2.0);
        assert!((signal.phase[0] - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_attenuation() {
        let mut signal = OpticalSignal::coherent(1.0, 0.0, 1550.0);
        let original_power = signal.power;

        signal.apply_attenuation(3.0); // 3 dB attenuation
        assert!(signal.power < original_power);
        assert!((signal.power / original_power - 0.5).abs() < 0.01); // ~3dB ≈ factor of 2
    }

    #[test]
    fn test_optical_matrix_unit() {
        let omu = OpticalMatrixUnit::new(2, 2);
        assert_eq!(omu.input_ports, 2);
        assert_eq!(omu.output_ports, 2);
        assert_eq!(omu.coupling_matrix.len(), 2);
        assert_eq!(omu.coupling_matrix[0].len(), 2);
    }

    #[test]
    fn test_mach_zehnder_configuration() {
        let mut omu = OpticalMatrixUnit::new(2, 2);
        let result = omu.configure_mach_zehnder(std::f64::consts::PI, 0.5);
        assert!(result.is_ok());
        assert_eq!(omu.phase_shifters[1], std::f64::consts::PI);
    }

    #[test]
    fn test_photonic_processor_creation() {
        let processor = PhotonicProcessor::with_platform(OpticalPlatform::MachZehnder);
        assert_eq!(processor.config.platform, OpticalPlatform::MachZehnder);
        assert_eq!(processor.config.num_waveguides, 64);
    }

    #[test]
    fn test_platform_configurations() {
        let platforms = [
            OpticalPlatform::MachZehnder,
            OpticalPlatform::RingResonator,
            OpticalPlatform::PhotonicCrystal,
            OpticalPlatform::Simulation,
        ];

        for &platform in &platforms {
            let processor = PhotonicProcessor::with_platform(platform);
            assert_eq!(processor.config.platform, platform);
            assert!(processor.config.num_waveguides > 0);
            assert!(processor.config.power_budget > 0.0);
        }
    }

    #[test]
    fn test_optical_encoding() {
        let processor = PhotonicProcessor::with_platform(OpticalPlatform::Simulation);
        let input = Tensor::from_vec(vec![0.5, 1.0, -0.5], &[3]).expect("Tensor from_vec failed");

        let amplitude_encoded = processor.tensor_to_optical(&input, OpticalEncoding::Amplitude);
        assert!(amplitude_encoded.is_ok());

        let phase_encoded = processor.tensor_to_optical(&input, OpticalEncoding::Phase);
        assert!(phase_encoded.is_ok());

        let signals = amplitude_encoded.expect("operation failed in test");
        assert_eq!(signals.len(), 3);
        assert_eq!(signals[0].amplitude[0], 0.5);
        assert_eq!(signals[1].amplitude[0], 1.0);
        assert_eq!(signals[2].amplitude[0], 0.5); // abs(-0.5)
    }

    #[test]
    fn test_optical_decoding() {
        let processor = PhotonicProcessor::with_platform(OpticalPlatform::Simulation);
        let signals = vec![
            OpticalSignal::coherent(1.0, 0.0, 1550.0),
            OpticalSignal::coherent(2.0, std::f64::consts::PI / 2.0, 1550.0),
            OpticalSignal::coherent(0.5, std::f64::consts::PI, 1550.0),
        ];

        let intensity_decoded = processor.optical_to_tensor(&signals, OpticalDecoding::Intensity);
        assert!(intensity_decoded.is_ok());

        let decoded = intensity_decoded.expect("operation failed in test");
        let data = decoded.data().expect("operation failed in test");
        assert_eq!(data.len(), 3);
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 4.0);
        assert_eq!(data[2], 0.25);
    }

    #[test]
    fn test_complex_amplitude() {
        let signal = OpticalSignal::multi_mode(
            vec![1.0, 2.0],
            vec![0.0, std::f64::consts::PI / 2.0],
            1550.0,
        );

        let complex_amps = signal.complex_amplitude();
        assert_eq!(complex_amps.len(), 2);
        assert!((complex_amps[0].real - 1.0).abs() < 1e-10);
        assert!(complex_amps[0].imag.abs() < 1e-10);
        assert!(complex_amps[1].real.abs() < 1e-10);
        assert!((complex_amps[1].imag - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_energy_efficiency() {
        let processor = PhotonicProcessor::with_platform(OpticalPlatform::Simulation);
        let efficiency = processor.get_energy_efficiency();
        assert!(efficiency > 0.0);
    }

    /// `DirectMapping` must genuinely map every classical weight into the resulting network's
    /// coupling matrix, in the same [output_size, input_size] layout as the input.
    #[test]
    fn test_convert_direct_mapping_populates_real_coupling_matrix() {
        // 2x3 matrix: [output_size=2, input_size=3]
        let weights =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("tensor");

        let network = convert_to_photonic(&weights, PhotonicConversion::DirectMapping)
            .expect("DirectMapping must succeed for a well-formed 2-D weight matrix");

        assert_eq!(network.num_inputs, 3);
        assert_eq!(network.num_outputs, 2);
        assert_eq!(
            network.layers.len(),
            1,
            "DirectMapping produces exactly one layer"
        );

        let layer = &network.layers[0];
        assert_eq!(
            layer.coupling_matrix,
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
        );
        assert!(matches!(layer.nonlinearity, PhotonicNonlinearity::Linear));
    }

    /// A non-2-D weight tensor must produce a structured error, not a panic or a silently
    /// wrong mapping.
    #[test]
    fn test_convert_direct_mapping_rejects_non_2d_weights() {
        let weights = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2, 1]).expect("tensor");
        let result = convert_to_photonic(&weights, PhotonicConversion::DirectMapping);
        assert!(
            result.is_err(),
            "a rank-3 weight tensor must be rejected, not silently used"
        );
    }

    /// End-to-end: a network built by `convert_to_photonic(..., DirectMapping)` and then run
    /// through `PhotonicLayer::process` (via `PhotonicNeuralNetwork::forward`) must produce
    /// exactly the source weight matrix's dense matrix-vector product — proving `process`
    /// genuinely consumes the `coupling_matrix` `convert_direct_mapping` builds, not a network
    /// that carries correct data nobody reads. The two output rows differ (`[1,2,3]` vs
    /// `[4,5,6]`) and `x` is chosen so both dot products are non-zero and unequal, so a bug that
    /// dropped the coupling matrix or mixed up rows could not accidentally pass.
    #[test]
    fn test_convert_direct_mapping_process_matches_dense_matmul() {
        // weights: [output_size=2, input_size=3] = [[1,2,3],[4,5,6]]
        let weights =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("tensor");
        let network = convert_to_photonic(&weights, PhotonicConversion::DirectMapping)
            .expect("DirectMapping must succeed");

        // x = [1.0, -2.0, -1.0]: dense matmul gives y0 = 1-4-3 = -6, y1 = 4-10-6 = -12 — two
        // distinct, both-negative values, so this also proves sign fidelity (not just
        // magnitude) survives the complex-amplitude round trip.
        let x = [1.0f64, -2.0, -1.0];
        let inputs: Vec<OpticalSignal> =
            x.iter().map(|&v| OpticalSignal::coherent(v, 0.0, network.wavelength)).collect();

        let outputs = network.forward(&inputs).expect("forward must succeed");
        assert_eq!(outputs.len(), 2);

        let expected = [-6.0f64, -12.0];
        for (o, &expected_value) in expected.iter().enumerate() {
            // Zero phase shifts mean the imaginary part is exactly zero, so
            // amplitude * cos(phase) recovers the signed dense-matmul value exactly (magnitude
            // alone would lose the sign).
            let recovered = outputs[o].amplitude[0] * outputs[o].phase[0].cos();
            assert!(
                (recovered - expected_value).abs() < 1e-9,
                "output {o}: expected {expected_value}, recovered {recovered} from {:?}",
                outputs[o]
            );
        }
    }

    /// The two unimplemented conversion methods must refuse with a structured error, never
    /// silently fall back to DirectMapping's output.
    #[test]
    fn test_unimplemented_conversion_methods_return_structured_errors() {
        let weights = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor");

        let interferometric =
            convert_to_photonic(&weights, PhotonicConversion::InterferometricMapping);
        assert!(interferometric.is_err());
        assert!(interferometric
            .unwrap_err()
            .to_string()
            .to_lowercase()
            .contains("not implemented"));

        let resonator = convert_to_photonic(&weights, PhotonicConversion::ResonatorMapping);
        assert!(resonator.is_err());
        assert!(resonator.unwrap_err().to_string().to_lowercase().contains("not implemented"));
    }
}
