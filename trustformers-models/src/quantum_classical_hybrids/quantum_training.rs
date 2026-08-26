//! Hybrid quantum-classical training loop.
//!
//! The manager owns the parameters it trains: a classical parameter vector and
//! the rotation angles of a [`VariationalCircuit`]. Each strategy applies a
//! real gradient-descent step to those vectors, and the reported fidelity is
//! the exactly computed state overlap `|⟨ψ(θ_before)|ψ(θ_after)⟩|²` between the
//! circuits before and after the update — not a configuration constant.

use trustformers_core::errors::{Result, TrustformersError};

use super::{
    config::{HybridTrainingStrategy, QuantumClassicalConfig},
    statevector::VariationalCircuit,
};

/// Quantum training manager
#[derive(Debug)]
pub struct QuantumTrainingManager {
    /// Configuration
    pub config: QuantumClassicalConfig,
    /// Training strategy
    pub training_strategy: HybridTrainingStrategy,
    /// Classical learning rate
    pub classical_lr: f64,
    /// Quantum learning rate
    pub quantum_lr: f64,
    /// Classical parameters updated by the trainer
    pub classical_parameters: Vec<f32>,
    /// Quantum circuit rotation angles updated by the trainer
    pub quantum_parameters: Vec<f64>,
    /// Circuit used to measure the effect of a quantum update
    pub circuit: VariationalCircuit,
    /// Training metrics
    pub training_metrics: QuantumTrainingMetrics,
    /// Current epoch
    pub current_epoch: usize,
    /// Training history
    pub training_history: Vec<QuantumTrainingMetrics>,
}

/// Quantum training metrics
#[derive(Debug, Clone)]
pub struct QuantumTrainingMetrics {
    /// Classical loss
    pub classical_loss: f64,
    /// Quantum loss
    pub quantum_loss: f64,
    /// Total loss
    pub total_loss: f64,
    /// Quantum fidelity
    pub quantum_fidelity: f64,
    /// Classical accuracy
    pub classical_accuracy: f64,
    /// Quantum advantage metric
    pub quantum_advantage: f64,
    /// Training time
    pub training_time: f64,
}

impl QuantumTrainingManager {
    /// Create a new quantum training manager with zero-initialised classical
    /// parameters and uniformly initialised circuit angles.
    pub fn new(config: &QuantumClassicalConfig) -> Result<Self> {
        let circuit = VariationalCircuit::new(config.num_qubits, config.ansatz_layers())?;
        let quantum_parameters = vec![0.1; circuit.parameter_count()];
        let classical_parameters = vec![0.0f32; config.d_model];

        Ok(Self {
            config: config.clone(),
            training_strategy: config.hybrid_training_strategy.clone(),
            classical_lr: config.classical_learning_rate,
            quantum_lr: config.quantum_learning_rate,
            classical_parameters,
            quantum_parameters,
            circuit,
            training_metrics: QuantumTrainingMetrics::default(),
            current_epoch: 0,
            training_history: Vec::new(),
        })
    }

    /// Replace the classical parameter vector the trainer updates.
    pub fn set_classical_parameters(&mut self, parameters: Vec<f32>) {
        self.classical_parameters = parameters;
    }

    /// Replace the quantum parameter vector the trainer updates.
    pub fn set_quantum_parameters(&mut self, parameters: Vec<f64>) -> Result<()> {
        if parameters.len() != self.circuit.parameter_count() {
            return Err(TrustformersError::invalid_input(format!(
                "expected {} circuit parameters, got {}",
                self.circuit.parameter_count(),
                parameters.len()
            )));
        }
        self.quantum_parameters = parameters;
        Ok(())
    }

    /// Train one epoch
    pub fn train_epoch(
        &mut self,
        classical_gradients: &[f32],
        quantum_gradients: &[f64],
    ) -> Result<QuantumTrainingMetrics> {
        let start_time = std::time::Instant::now();

        match self.training_strategy {
            HybridTrainingStrategy::Sequential => {
                self.train_sequential(classical_gradients, quantum_gradients)?;
            },
            HybridTrainingStrategy::Alternating => {
                self.train_alternating(classical_gradients, quantum_gradients)?;
            },
            HybridTrainingStrategy::Joint => {
                self.train_joint(classical_gradients, quantum_gradients)?;
            },
            HybridTrainingStrategy::Adaptive => {
                self.train_adaptive(classical_gradients, quantum_gradients)?;
            },
        }

        self.training_metrics.total_loss =
            self.training_metrics.classical_loss + self.training_metrics.quantum_loss;
        self.training_metrics.quantum_advantage = self.config.get_quantum_advantage_factor();
        self.training_metrics.training_time = start_time.elapsed().as_secs_f64();

        self.training_history.push(self.training_metrics.clone());
        self.current_epoch += 1;

        Ok(self.training_metrics.clone())
    }

    /// Sequential training strategy
    fn train_sequential(
        &mut self,
        classical_gradients: &[f32],
        quantum_gradients: &[f64],
    ) -> Result<()> {
        self.update_classical_parameters(classical_gradients)?;
        self.update_quantum_parameters(quantum_gradients)?;
        Ok(())
    }

    /// Alternating training strategy
    fn train_alternating(
        &mut self,
        classical_gradients: &[f32],
        quantum_gradients: &[f64],
    ) -> Result<()> {
        if self.current_epoch.is_multiple_of(2) {
            self.update_classical_parameters(classical_gradients)?;
        } else {
            self.update_quantum_parameters(quantum_gradients)?;
        }
        Ok(())
    }

    /// Joint training strategy
    fn train_joint(
        &mut self,
        classical_gradients: &[f32],
        quantum_gradients: &[f64],
    ) -> Result<()> {
        self.update_classical_parameters(classical_gradients)?;
        self.update_quantum_parameters(quantum_gradients)?;
        Ok(())
    }

    /// Adaptive training strategy: update whichever half has the larger
    /// gradient norm.
    fn train_adaptive(
        &mut self,
        classical_gradients: &[f32],
        quantum_gradients: &[f64],
    ) -> Result<()> {
        let classical_grad_norm =
            classical_gradients.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt() as f64;
        let quantum_grad_norm = quantum_gradients.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();

        if classical_grad_norm > quantum_grad_norm {
            self.update_classical_parameters(classical_gradients)?;
        } else {
            self.update_quantum_parameters(quantum_gradients)?;
        }

        Ok(())
    }

    /// Apply a gradient-descent step to the classical parameters.
    ///
    /// Returns an error when the gradient length does not match the parameter
    /// vector, rather than silently recording a metric and changing nothing.
    pub fn update_classical_parameters(&mut self, gradients: &[f32]) -> Result<()> {
        if gradients.len() != self.classical_parameters.len() {
            return Err(TrustformersError::invalid_input(format!(
                "expected {} classical gradients, got {}",
                self.classical_parameters.len(),
                gradients.len()
            )));
        }

        let lr = self.classical_lr as f32;
        for (parameter, gradient) in self.classical_parameters.iter_mut().zip(gradients) {
            *parameter -= lr * gradient;
        }

        self.training_metrics.classical_loss =
            gradients.iter().map(|&x| x.powi(2)).sum::<f32>() as f64;

        Ok(())
    }

    /// Apply a gradient-descent step to the circuit angles and measure the
    /// fidelity between the states before and after the update.
    pub fn update_quantum_parameters(&mut self, gradients: &[f64]) -> Result<()> {
        if gradients.len() != self.quantum_parameters.len() {
            return Err(TrustformersError::invalid_input(format!(
                "expected {} quantum gradients, got {}",
                self.quantum_parameters.len(),
                gradients.len()
            )));
        }

        let encoding = vec![0.0f64; self.circuit.num_qubits()];
        let before = self.circuit.run(&encoding, &self.quantum_parameters)?;

        for (parameter, gradient) in self.quantum_parameters.iter_mut().zip(gradients) {
            *parameter -= self.quantum_lr * gradient;
        }

        let after = self.circuit.run(&encoding, &self.quantum_parameters)?;

        self.training_metrics.quantum_loss = gradients.iter().map(|&x| x.powi(2)).sum::<f64>();
        self.training_metrics.quantum_fidelity = before.fidelity(&after)?;

        Ok(())
    }

    /// Get training statistics
    pub fn get_training_stats(&self) -> QuantumTrainingStats {
        let epochs = self.training_history.len().max(1) as f64;

        QuantumTrainingStats {
            total_epochs: self.current_epoch,
            avg_classical_loss: self.training_history.iter().map(|m| m.classical_loss).sum::<f64>()
                / epochs,
            avg_quantum_loss: self.training_history.iter().map(|m| m.quantum_loss).sum::<f64>()
                / epochs,
            avg_quantum_fidelity: self
                .training_history
                .iter()
                .map(|m| m.quantum_fidelity)
                .sum::<f64>()
                / epochs,
            training_strategy: self.training_strategy.clone(),
            convergence_rate: self.compute_convergence_rate(),
        }
    }

    /// Relative decrease of the total loss over the recorded history.
    fn compute_convergence_rate(&self) -> f64 {
        if self.training_history.len() < 2 {
            return 0.0;
        }

        let first_loss = self.training_history[0].total_loss;
        let Some(last_entry) = self.training_history.last() else {
            return 0.0;
        };
        let last_loss = last_entry.total_loss;

        if first_loss > 0.0 {
            (first_loss - last_loss) / first_loss
        } else {
            0.0
        }
    }

    /// Reset training state (parameters are left untouched).
    pub fn reset(&mut self) {
        self.current_epoch = 0;
        self.training_history.clear();
        self.training_metrics = QuantumTrainingMetrics::default();
    }
}

/// Quantum training statistics
#[derive(Debug, Clone)]
pub struct QuantumTrainingStats {
    /// Total epochs trained
    pub total_epochs: usize,
    /// Average classical loss
    pub avg_classical_loss: f64,
    /// Average quantum loss
    pub avg_quantum_loss: f64,
    /// Average quantum fidelity
    pub avg_quantum_fidelity: f64,
    /// Training strategy used
    pub training_strategy: HybridTrainingStrategy,
    /// Convergence rate
    pub convergence_rate: f64,
}

impl Default for QuantumTrainingMetrics {
    fn default() -> Self {
        Self {
            classical_loss: 0.0,
            quantum_loss: 0.0,
            total_loss: 0.0,
            quantum_fidelity: 1.0,
            classical_accuracy: 0.0,
            quantum_advantage: 0.0,
            training_time: 0.0,
        }
    }
}
