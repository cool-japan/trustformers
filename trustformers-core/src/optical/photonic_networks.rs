//! Photonic neural networks implementation

use crate::optical::{Complex, OpticalSignal};
use anyhow::Result;

/// Photonic neural network
#[derive(Debug, Clone)]
pub struct PhotonicNeuralNetwork {
    pub layers: Vec<PhotonicLayer>,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub wavelength: f64,
}

/// Photonic layer implementation
#[derive(Debug, Clone)]
pub struct PhotonicLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub coupling_matrix: Vec<Vec<f64>>,
    pub phase_shifts: Vec<f64>,
    pub nonlinearity: PhotonicNonlinearity,
}

/// Photonic nonlinearity types
#[derive(Debug, Clone, Copy)]
pub enum PhotonicNonlinearity {
    Saturable,
    Kerr,
    ElectroOptic,
    Linear,
}

impl PhotonicNonlinearity {
    /// Apply this nonlinearity to one output channel's raw linear-combination result (the
    /// complex value produced by combining every input through the layer's coupling matrix and
    /// phase shifters), returning `(amplitude, phase)` ready for [`OpticalSignal::coherent`].
    ///
    /// Only [`PhotonicNonlinearity::Linear`] is implemented: since it applies no nonlinear
    /// transform (the classical-weight direct mapping this module documents is purely linear),
    /// the result is exactly `raw`'s polar form (`raw.magnitude()`, `raw.phase()`).
    /// `Saturable`/`Kerr`/`ElectroOptic` each name a physically distinct nonlinear response
    /// (saturable-absorber transmission, Kerr self-phase-modulation, electro-optic phase
    /// modulation) that needs a device parameter -- a saturation intensity, a Kerr coefficient,
    /// an applied field -- which `PhotonicLayer` has no field to carry. Applying any of them
    /// here would mean inventing an arbitrary constant with no physical grounding, so selecting
    /// one returns a structured error instead of a fabricated response.
    fn apply(&self, raw: Complex) -> Result<(f64, f64)> {
        match self {
            PhotonicNonlinearity::Linear => Ok((raw.magnitude(), raw.phase())),
            other => Err(anyhow::anyhow!(
                "PhotonicLayer::process: PhotonicNonlinearity::{other:?} is not implemented -- \
                 it needs a physical device parameter (saturation intensity / Kerr coefficient / \
                 applied field) that PhotonicLayer has no field to carry, and inventing one would \
                 be a fabricated result. Use PhotonicNonlinearity::Linear, the only implemented \
                 variant."
            )),
        }
    }
}

impl PhotonicNeuralNetwork {
    pub fn new(num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            layers: Vec::new(),
            num_inputs,
            num_outputs,
            wavelength: 1550.0,
        }
    }

    pub fn add_layer(&mut self, layer: PhotonicLayer) {
        self.layers.push(layer);
    }

    /// Set a single entry of layer `layer`'s coupling matrix: the coupling coefficient routing
    /// input channel `input_channel` into output neuron `output_neuron`
    /// (`coupling_matrix[output_neuron][input_channel]`). Returns a structured error -- never
    /// silently drops the value -- when any index is out of bounds for the target layer's
    /// actual shape. [`PhotonicLayer::process`] reads exactly this matrix, so a call that
    /// returns `Ok(())` genuinely changes the layer's forward transform.
    ///
    /// `PhotonicLayer`'s fields are all `pub`; `phase_shifts`, and bulk `coupling_matrix`
    /// replacement, can be set directly (e.g. `network.layers[layer].phase_shifts[i] = value`)
    /// without a dedicated setter -- this method exists for the common single-weight case.
    pub fn set_coupling(
        &mut self,
        layer: usize,
        output_neuron: usize,
        input_channel: usize,
        coupling: f64,
    ) -> Result<()> {
        let num_layers = self.layers.len();
        let target_layer = self.layers.get_mut(layer).ok_or_else(|| {
            anyhow::anyhow!(
                "set_coupling: layer index {layer} out of bounds ({num_layers} layer(s) exist)"
            )
        })?;

        let num_outputs = target_layer.coupling_matrix.len();
        let row = target_layer.coupling_matrix.get_mut(output_neuron).ok_or_else(|| {
            anyhow::anyhow!(
                "set_coupling: output_neuron index {output_neuron} out of bounds (layer {layer} \
                 has {num_outputs} output neuron(s))"
            )
        })?;

        let num_inputs = row.len();
        let entry = row.get_mut(input_channel).ok_or_else(|| {
            anyhow::anyhow!(
                "set_coupling: input_channel index {input_channel} out of bounds (layer \
                 {layer}'s coupling matrix row has {num_inputs} entries)"
            )
        })?;

        *entry = coupling;
        Ok(())
    }

    pub fn forward(&self, inputs: &[OpticalSignal]) -> Result<Vec<OpticalSignal>> {
        let mut current_signals = inputs.to_vec();

        for layer in &self.layers {
            current_signals = layer.process(&current_signals)?;
        }

        Ok(current_signals)
    }
}

impl PhotonicLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            coupling_matrix: vec![vec![0.0; input_size]; output_size],
            phase_shifts: vec![0.0; input_size],
            nonlinearity: PhotonicNonlinearity::Linear,
        }
    }

    /// Forward pass: applies this layer's coupling matrix and phase shifters as a
    /// complex-amplitude linear transform, then this layer's nonlinearity.
    ///
    /// Every input channel's (possibly multi-mode) amplitude/phase components are first summed
    /// into a single complex amplitude (the same convention
    /// [`OpticalMatrixUnit::process`](crate::optical::OpticalMatrixUnit::process) uses), each
    /// channel's complex amplitude is rotated by its phase shifter
    /// (`exp(i * phase_shifts[input_channel])`), and every output neuron combines all channels
    /// through its row of `coupling_matrix` (real-valued coupling coefficients):
    ///
    /// ```text
    /// raw[o] = Σ_i coupling_matrix[o][i] * exp(i·phase_shifts[i]) * input_complex[i]
    /// output[o] = nonlinearity(raw[o])
    /// ```
    ///
    /// This is the direct-mapping semantics [`crate::optical::convert_to_photonic`] documents:
    /// a network built by
    /// [`PhotonicConversion::DirectMapping`](crate::optical::PhotonicConversion::DirectMapping)
    /// (real couplings equal to the source classical weights, zero phase shifts,
    /// `PhotonicNonlinearity::Linear`) reduces this to exactly the source weight matrix's dense
    /// matrix-vector product.
    ///
    /// Returns a structured error, rather than panicking or silently substituting zeros, when
    /// `inputs.len()` doesn't match `input_size`, or when `coupling_matrix`/`phase_shifts` don't
    /// have the shape `input_size`/`output_size` promise (this can happen if they were mutated
    /// directly through their `pub` fields with the wrong dimensions rather than through
    /// [`PhotonicNeuralNetwork::set_coupling`]).
    pub fn process(&self, inputs: &[OpticalSignal]) -> Result<Vec<OpticalSignal>> {
        if inputs.len() != self.input_size {
            return Err(anyhow::anyhow!(
                "PhotonicLayer::process: expected {} input signal(s) (input_size), got {}",
                self.input_size,
                inputs.len()
            ));
        }
        if self.coupling_matrix.len() != self.output_size {
            return Err(anyhow::anyhow!(
                "PhotonicLayer::process: coupling_matrix has {} row(s), expected output_size {}",
                self.coupling_matrix.len(),
                self.output_size
            ));
        }
        for (o, row) in self.coupling_matrix.iter().enumerate() {
            if row.len() != self.input_size {
                return Err(anyhow::anyhow!(
                    "PhotonicLayer::process: coupling_matrix row {o} has {} entries, expected \
                     input_size {}",
                    row.len(),
                    self.input_size
                ));
            }
        }
        if self.phase_shifts.len() != self.input_size {
            return Err(anyhow::anyhow!(
                "PhotonicLayer::process: phase_shifts has {} entries, expected input_size {}",
                self.phase_shifts.len(),
                self.input_size
            ));
        }

        // Every input channel's complex amplitude: its own (possibly multi-mode)
        // amplitude/phase components summed into one complex number, the same convention
        // `OpticalMatrixUnit::process` uses.
        let input_complex: Vec<Complex> = inputs
            .iter()
            .map(|signal| {
                signal.amplitude.iter().zip(&signal.phase).fold(
                    Complex::new(0.0, 0.0),
                    |acc, (&amplitude, &phase)| {
                        acc + Complex::new(amplitude * phase.cos(), amplitude * phase.sin())
                    },
                )
            })
            .collect();

        let wavelength = inputs.first().map(|signal| signal.wavelength).unwrap_or(1550.0);

        let mut outputs = Vec::with_capacity(self.output_size);
        for row in &self.coupling_matrix {
            let mut raw = Complex::new(0.0, 0.0);
            for (i, &coupling) in row.iter().enumerate() {
                let phase_factor = Complex::exp_i(self.phase_shifts[i]);
                raw = raw + Complex::new(coupling, 0.0) * phase_factor * input_complex[i];
            }
            let (amplitude, phase) = self.nonlinearity.apply(raw)?;
            outputs.push(OpticalSignal::coherent(amplitude, phase, wavelength));
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(amplitude: f64, phase: f64) -> OpticalSignal {
        OpticalSignal::coherent(amplitude, phase, 1550.0)
    }

    // -- PhotonicLayer::process: identity coupling --

    #[test]
    fn test_process_identity_coupling_returns_input_unchanged() {
        let mut layer = PhotonicLayer::new(2, 2);
        layer.coupling_matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // phase_shifts already zero from `new`.

        let inputs = vec![signal(3.0, 0.0), signal(5.0, 0.0)];
        let outputs = layer.process(&inputs).expect("identity-coupling process must succeed");

        assert_eq!(outputs.len(), 2);
        assert!(
            (outputs[0].amplitude[0] - 3.0).abs() < 1e-9,
            "got {:?}",
            outputs[0]
        );
        assert!(outputs[0].phase[0].abs() < 1e-9);
        assert!(
            (outputs[1].amplitude[0] - 5.0).abs() < 1e-9,
            "got {:?}",
            outputs[1]
        );
        assert!(outputs[1].phase[0].abs() < 1e-9);
    }

    // -- PhotonicLayer::process: known 2x2 coupling with a nonzero phase shift, hand-computed --

    #[test]
    fn test_process_known_2x2_coupling_matches_hand_computed_output() {
        let mut layer = PhotonicLayer::new(2, 2);
        layer.coupling_matrix = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        layer.phase_shifts = vec![0.0, std::f64::consts::FRAC_PI_2];

        let inputs = vec![signal(2.0, 0.0), signal(3.0, 0.0)];
        let outputs = layer.process(&inputs).expect("process must succeed");

        // By hand: input_complex = [2+0i, 3+0i]; phase_factor = [1, i].
        // raw0 = 1.0*1*(2+0i) + 0.5*i*(3+0i) = 2 + 1.5i -> magnitude 2.5, phase atan2(1.5, 2.0)
        // raw1 = 0.5*1*(2+0i) + 1.0*i*(3+0i) = 1 + 3.0i -> magnitude sqrt(10), phase atan2(3.0, 1.0)
        assert!(
            (outputs[0].amplitude[0] - 2.5).abs() < 1e-9,
            "got {:?}",
            outputs[0]
        );
        assert!((outputs[0].phase[0] - 1.5f64.atan2(2.0)).abs() < 1e-9);

        assert!(
            (outputs[1].amplitude[0] - 10.0f64.sqrt()).abs() < 1e-9,
            "got {:?}",
            outputs[1]
        );
        assert!((outputs[1].phase[0] - 3.0f64.atan2(1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_process_rejects_wrong_input_count() {
        let layer = PhotonicLayer::new(2, 2);
        let result = layer.process(&[signal(1.0, 0.0)]);
        assert!(
            result.is_err(),
            "one input signal for an input_size=2 layer must error"
        );
    }

    #[test]
    fn test_process_rejects_malformed_coupling_matrix_shape() {
        let mut layer = PhotonicLayer::new(2, 2);
        layer.coupling_matrix = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]; // rows too wide
        let result = layer.process(&[signal(1.0, 0.0), signal(1.0, 0.0)]);
        assert!(
            result.is_err(),
            "a coupling_matrix row with the wrong width must error, not panic"
        );
    }

    #[test]
    fn test_process_rejects_malformed_phase_shifts_length() {
        let mut layer = PhotonicLayer::new(2, 2);
        layer.phase_shifts = vec![0.0]; // should have 2 entries (input_size)
        let result = layer.process(&[signal(1.0, 0.0), signal(1.0, 0.0)]);
        assert!(
            result.is_err(),
            "a mismatched phase_shifts length must error, not panic"
        );
    }

    #[test]
    fn test_process_non_linear_nonlinearity_returns_structured_error() {
        let mut layer = PhotonicLayer::new(1, 1);
        layer.coupling_matrix = vec![vec![1.0]];
        layer.nonlinearity = PhotonicNonlinearity::Kerr;

        let result = layer.process(&[signal(1.0, 0.0)]);
        assert!(result.is_err());
        let message = result.expect_err("checked above").to_string().to_lowercase();
        assert!(
            message.contains("not implemented"),
            "expected a 'not implemented' style refusal, got: {message}"
        );
    }

    // -- PhotonicNeuralNetwork::set_coupling --

    #[test]
    fn test_set_coupling_out_of_bounds_layer_returns_error() {
        let mut network = PhotonicNeuralNetwork::new(2, 2);
        assert!(
            network.set_coupling(0, 0, 0, 1.0).is_err(),
            "no layers exist yet"
        );
    }

    #[test]
    fn test_set_coupling_out_of_bounds_indices_return_errors() {
        let mut network = PhotonicNeuralNetwork::new(2, 2);
        network.add_layer(PhotonicLayer::new(2, 2));

        assert!(
            network.set_coupling(0, 5, 0, 1.0).is_err(),
            "output_neuron 5 is out of bounds"
        );
        assert!(
            network.set_coupling(0, 0, 5, 1.0).is_err(),
            "input_channel 5 is out of bounds"
        );
    }

    #[test]
    fn test_set_coupling_round_trip_is_read_by_process() {
        let mut network = PhotonicNeuralNetwork::new(2, 2);
        network.add_layer(PhotonicLayer::new(2, 2));

        network
            .set_coupling(0, 0, 0, 2.0)
            .expect("set_coupling must succeed for a valid index");
        network
            .set_coupling(0, 0, 1, 3.0)
            .expect("set_coupling must succeed for a valid index");
        network
            .set_coupling(0, 1, 0, 4.0)
            .expect("set_coupling must succeed for a valid index");
        network
            .set_coupling(0, 1, 1, 5.0)
            .expect("set_coupling must succeed for a valid index");

        // The value is genuinely stored, not dropped.
        assert_eq!(
            network.layers[0].coupling_matrix,
            vec![vec![2.0, 3.0], vec![4.0, 5.0]]
        );

        // ...and `process` genuinely reads it back: dense matmul [[2,3],[4,5]] @ [1,1] = [5,9].
        let outputs = network
            .forward(&[signal(1.0, 0.0), signal(1.0, 0.0)])
            .expect("forward must succeed with a fully-populated layer");
        assert!(
            (outputs[0].amplitude[0] - 5.0).abs() < 1e-9,
            "got {:?}",
            outputs[0]
        );
        assert!(
            (outputs[1].amplitude[0] - 9.0).abs() < 1e-9,
            "got {:?}",
            outputs[1]
        );
    }
}
