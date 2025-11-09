use serde::{Deserialize, Serialize};
use trustformers_core::{
    errors::{invalid_config, Result},
    quantum::{MeasurementBasis, QuantumAnsatz, QuantumBackend, QuantumEncoding, RotationAxis},
    traits::Config,
};

/// Quantum-classical hybrid architecture type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantumHybridArchitecture {
    /// Quantum transformer with classical attention
    QuantumTransformer,
    /// Quantum graph neural network
    QuantumGraphNeuralNetwork,
    /// Quantum convolutional neural network
    QuantumConvolutionalNN,
    /// Quantum recurrent neural network
    QuantumRecurrentNN,
    /// Hybrid quantum-classical attention
    QuantumAttention,
    /// Quantum embedding layers
    QuantumEmbedding,
    /// Variational quantum circuit
    VariationalQuantumCircuit,
    /// Quantum approximate optimization algorithm
    QuantumApproximateOptimization,
}

/// Quantum circuit ansatz configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantumAnsatzConfig {
    /// Hardware-efficient ansatz
    HardwareEfficient { layers: usize },
    /// Alternating ansatz
    Alternating { layers: usize },
    /// Real amplitudes ansatz
    RealAmplitudes { layers: usize },
    /// Efficient SU(2) ansatz
    EfficientSU2 { layers: usize },
    /// Custom ansatz
    Custom { gates: Vec<String>, layers: usize },
}

/// Quantum measurement strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantumMeasurementStrategy {
    /// Expectation value measurement
    Expectation,
    /// Probability distribution measurement
    ProbabilityDistribution,
    /// Sampling-based measurement
    Sampling { shots: usize },
    /// Tomography-based measurement
    Tomography,
}

/// Quantum error mitigation technique
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantumErrorMitigation {
    /// No error mitigation
    None,
    /// Zero-noise extrapolation
    ZeroNoiseExtrapolation,
    /// Readout error mitigation
    ReadoutErrorMitigation,
    /// Symmetry verification
    SymmetryVerification,
    /// Clifford data regression
    CliffordDataRegression,
}

/// Quantum-classical hybrid model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumClassicalConfig {
    /// Architecture type
    pub architecture: QuantumHybridArchitecture,
    /// Model dimension
    pub d_model: usize,
    /// Number of classical layers
    pub n_classical_layers: usize,
    /// Number of quantum layers
    pub n_quantum_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Number of qubits
    pub num_qubits: usize,
    /// Quantum backend
    pub quantum_backend: QuantumBackend,
    /// Quantum encoding method
    pub quantum_encoding: QuantumEncoding,
    /// Quantum ansatz configuration
    pub quantum_ansatz: QuantumAnsatzConfig,
    /// Measurement basis
    pub measurement_basis: MeasurementBasis,
    /// Measurement strategy
    pub measurement_strategy: QuantumMeasurementStrategy,
    /// Error mitigation technique
    pub error_mitigation: QuantumErrorMitigation,
    /// Number of shots for sampling
    pub num_shots: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Quantum learning rate
    pub quantum_learning_rate: f64,
    /// Classical learning rate
    pub classical_learning_rate: f64,
    /// Quantum noise variance
    pub quantum_noise_variance: f64,
    /// Whether to use quantum gradients
    pub use_quantum_gradients: bool,
    /// Parameter shift rule stepsize
    pub parameter_shift_stepsize: f64,
    /// Quantum optimization tolerance
    pub quantum_optimization_tolerance: f64,
    /// Maximum quantum optimization iterations
    pub max_quantum_iterations: usize,
    /// Hybrid training strategy
    pub hybrid_training_strategy: HybridTrainingStrategy,
    /// Quantum device connectivity
    pub quantum_connectivity: QuantumConnectivity,
    /// Whether to use quantum acceleration
    pub use_quantum_acceleration: bool,
    /// Quantum entanglement depth
    pub entanglement_depth: usize,
    /// Whether to use bias
    pub use_bias: bool,
    /// Initializer range
    pub initializer_range: f32,
    /// Model type identifier
    pub model_type: String,
}

/// Hybrid training strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HybridTrainingStrategy {
    /// Sequential training (classical first, then quantum)
    Sequential,
    /// Alternating training (classical and quantum interleaved)
    Alternating,
    /// Joint training (classical and quantum simultaneous)
    Joint,
    /// Adaptive training (dynamically adjust based on performance)
    Adaptive,
}

/// Quantum connectivity pattern
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantumConnectivity {
    /// Fully connected qubits
    FullyConnected,
    /// Linear chain connectivity
    Linear,
    /// Circular connectivity
    Circular,
    /// 2D grid connectivity
    Grid2D { rows: usize, cols: usize },
    /// Custom connectivity
    Custom { edges: Vec<(usize, usize)> },
}

impl Default for QuantumClassicalConfig {
    fn default() -> Self {
        Self {
            architecture: QuantumHybridArchitecture::QuantumTransformer,
            d_model: 512,
            n_classical_layers: 6,
            n_quantum_layers: 2,
            vocab_size: 50000,
            max_position_embeddings: 2048,
            num_qubits: 8,
            quantum_backend: QuantumBackend::Simulator,
            quantum_encoding: QuantumEncoding::Angle,
            quantum_ansatz: QuantumAnsatzConfig::HardwareEfficient { layers: 2 },
            measurement_basis: MeasurementBasis::Computational,
            measurement_strategy: QuantumMeasurementStrategy::Expectation,
            error_mitigation: QuantumErrorMitigation::None,
            num_shots: 1024,
            circuit_depth: 10,
            quantum_learning_rate: 0.01,
            classical_learning_rate: 0.001,
            quantum_noise_variance: 0.01,
            use_quantum_gradients: true,
            parameter_shift_stepsize: 0.5,
            quantum_optimization_tolerance: 1e-6,
            max_quantum_iterations: 1000,
            hybrid_training_strategy: HybridTrainingStrategy::Alternating,
            quantum_connectivity: QuantumConnectivity::FullyConnected,
            use_quantum_acceleration: true,
            entanglement_depth: 2,
            use_bias: true,
            initializer_range: 0.02,
            model_type: "quantum_classical_hybrid".to_string(),
        }
    }
}

impl QuantumClassicalConfig {
    /// Create a quantum transformer configuration
    pub fn quantum_transformer() -> Self {
        Self {
            architecture: QuantumHybridArchitecture::QuantumTransformer,
            n_classical_layers: 8,
            n_quantum_layers: 2,
            num_qubits: 12,
            quantum_ansatz: QuantumAnsatzConfig::HardwareEfficient { layers: 3 },
            ..Default::default()
        }
    }

    /// Create a quantum GNN configuration
    pub fn quantum_gnn() -> Self {
        Self {
            architecture: QuantumHybridArchitecture::QuantumGraphNeuralNetwork,
            n_classical_layers: 4,
            n_quantum_layers: 3,
            num_qubits: 16,
            quantum_encoding: QuantumEncoding::Amplitude,
            quantum_ansatz: QuantumAnsatzConfig::Alternating { layers: 2 },
            ..Default::default()
        }
    }

    /// Create a quantum CNN configuration
    pub fn quantum_cnn() -> Self {
        Self {
            architecture: QuantumHybridArchitecture::QuantumConvolutionalNN,
            n_classical_layers: 6,
            n_quantum_layers: 4,
            num_qubits: 10,
            quantum_encoding: QuantumEncoding::Basis,
            quantum_ansatz: QuantumAnsatzConfig::RealAmplitudes { layers: 2 },
            ..Default::default()
        }
    }

    /// Create a quantum RNN configuration
    pub fn quantum_rnn() -> Self {
        Self {
            architecture: QuantumHybridArchitecture::QuantumRecurrentNN,
            n_classical_layers: 3,
            n_quantum_layers: 2,
            num_qubits: 8,
            quantum_encoding: QuantumEncoding::Angle,
            quantum_ansatz: QuantumAnsatzConfig::EfficientSU2 { layers: 2 },
            hybrid_training_strategy: HybridTrainingStrategy::Sequential,
            ..Default::default()
        }
    }

    /// Create a variational quantum circuit configuration
    pub fn variational_quantum_circuit() -> Self {
        Self {
            architecture: QuantumHybridArchitecture::VariationalQuantumCircuit,
            n_classical_layers: 2,
            n_quantum_layers: 5,
            num_qubits: 20,
            quantum_ansatz: QuantumAnsatzConfig::Custom {
                gates: vec!["RX".to_string(), "RY".to_string(), "CNOT".to_string()],
                layers: 4,
            },
            measurement_strategy: QuantumMeasurementStrategy::Sampling { shots: 2048 },
            ..Default::default()
        }
    }

    /// Create a quantum approximate optimization algorithm configuration
    pub fn quantum_approximate_optimization() -> Self {
        Self {
            architecture: QuantumHybridArchitecture::QuantumApproximateOptimization,
            n_classical_layers: 1,
            n_quantum_layers: 10,
            num_qubits: 16,
            quantum_ansatz: QuantumAnsatzConfig::Alternating { layers: 8 },
            hybrid_training_strategy: HybridTrainingStrategy::Adaptive,
            quantum_optimization_tolerance: 1e-8,
            max_quantum_iterations: 5000,
            ..Default::default()
        }
    }

    /// Create configuration from pretrained model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            "quantum-transformer" => Some(Self::quantum_transformer()),
            "quantum-gnn" => Some(Self::quantum_gnn()),
            "quantum-cnn" => Some(Self::quantum_cnn()),
            "quantum-rnn" => Some(Self::quantum_rnn()),
            "variational-quantum-circuit" => Some(Self::variational_quantum_circuit()),
            "quantum-approximate-optimization" => Some(Self::quantum_approximate_optimization()),
            _ => None,
        }
    }

    /// Get effective quantum dimension
    pub fn get_quantum_dimension(&self) -> usize {
        2_usize.pow(self.num_qubits as u32)
    }

    /// Get total number of layers
    pub fn get_total_layers(&self) -> usize {
        self.n_classical_layers + self.n_quantum_layers
    }

    /// Get quantum circuit parameters count
    pub fn get_quantum_parameters_count(&self) -> usize {
        match &self.quantum_ansatz {
            QuantumAnsatzConfig::HardwareEfficient { layers } => {
                self.num_qubits * layers + (self.num_qubits - 1) * layers
            },
            QuantumAnsatzConfig::Alternating { layers } => self.num_qubits * layers * 2,
            QuantumAnsatzConfig::RealAmplitudes { layers } => self.num_qubits * layers,
            QuantumAnsatzConfig::EfficientSU2 { layers } => self.num_qubits * layers * 3,
            QuantumAnsatzConfig::Custom { gates, layers } => gates.len() * layers * self.num_qubits,
        }
    }

    /// Get quantum advantage factor estimate
    pub fn get_quantum_advantage_factor(&self) -> f64 {
        // Simple heuristic based on qubit count and circuit depth
        let quantum_volume = self.num_qubits * self.circuit_depth;
        let classical_volume = self.n_classical_layers * self.d_model;

        if quantum_volume > classical_volume {
            (quantum_volume as f64 / classical_volume as f64).log2()
        } else {
            1.0
        }
    }
}

impl Config for QuantumClassicalConfig {
    fn architecture(&self) -> &'static str {
        match self.architecture {
            QuantumHybridArchitecture::QuantumTransformer => "quantum_transformer",
            QuantumHybridArchitecture::QuantumGraphNeuralNetwork => "quantum_gnn",
            QuantumHybridArchitecture::QuantumConvolutionalNN => "quantum_cnn",
            QuantumHybridArchitecture::QuantumRecurrentNN => "quantum_rnn",
            QuantumHybridArchitecture::QuantumAttention => "quantum_attention",
            QuantumHybridArchitecture::QuantumEmbedding => "quantum_embedding",
            QuantumHybridArchitecture::VariationalQuantumCircuit => "variational_quantum_circuit",
            QuantumHybridArchitecture::QuantumApproximateOptimization => {
                "quantum_approximate_optimization"
            },
        }
    }

    fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            return Err(invalid_config("d_model", "d_model must be greater than 0"));
        }
        if self.n_classical_layers == 0 && self.n_quantum_layers == 0 {
            return Err(invalid_config(
                "layers",
                "At least one classical or quantum layer must be specified",
            ));
        }
        if self.vocab_size == 0 {
            return Err(invalid_config(
                "vocab_size",
                "vocab_size must be greater than 0",
            ));
        }
        if self.num_qubits == 0 {
            return Err(invalid_config(
                "num_qubits",
                "num_qubits must be greater than 0",
            ));
        }
        if self.num_qubits > 30 {
            return Err(invalid_config(
                "num_qubits",
                "num_qubits is limited to 30 for simulation efficiency",
            ));
        }
        if self.circuit_depth == 0 {
            return Err(invalid_config(
                "circuit_depth",
                "circuit_depth must be greater than 0",
            ));
        }
        if self.quantum_learning_rate <= 0.0 {
            return Err(invalid_config(
                "quantum_learning_rate",
                "quantum_learning_rate must be greater than 0",
            ));
        }
        if self.classical_learning_rate <= 0.0 {
            return Err(invalid_config(
                "classical_learning_rate",
                "classical_learning_rate must be greater than 0",
            ));
        }
        if self.parameter_shift_stepsize <= 0.0 {
            return Err(invalid_config(
                "parameter_shift_stepsize",
                "parameter_shift_stepsize must be greater than 0",
            ));
        }
        if self.quantum_optimization_tolerance <= 0.0 {
            return Err(invalid_config(
                "quantum_optimization_tolerance",
                "quantum_optimization_tolerance must be greater than 0",
            ));
        }
        if self.max_quantum_iterations == 0 {
            return Err(invalid_config(
                "max_quantum_iterations",
                "max_quantum_iterations must be greater than 0",
            ));
        }
        if self.entanglement_depth == 0 {
            return Err(invalid_config(
                "entanglement_depth",
                "entanglement_depth must be greater than 0",
            ));
        }
        Ok(())
    }
}

/// Convert QuantumAnsatzConfig to QuantumAnsatz
impl From<QuantumAnsatzConfig> for QuantumAnsatz {
    fn from(config: QuantumAnsatzConfig) -> Self {
        match config {
            QuantumAnsatzConfig::HardwareEfficient { layers: _ } => QuantumAnsatz::Hardware,
            QuantumAnsatzConfig::Alternating { layers } => QuantumAnsatz::Efficient { layers },
            QuantumAnsatzConfig::RealAmplitudes { layers } => {
                QuantumAnsatz::RealAmplitudes { layers }
            },
            QuantumAnsatzConfig::EfficientSU2 { layers } => QuantumAnsatz::Efficient { layers },
            QuantumAnsatzConfig::Custom { gates, layers: _ } => {
                // Convert gates to rotation axes - simplified mapping
                let rotation_blocks = gates
                    .iter()
                    .map(|gate| {
                        match gate.as_str() {
                            "RX" => RotationAxis::X,
                            "RY" => RotationAxis::Y,
                            "RZ" => RotationAxis::Z,
                            _ => RotationAxis::Y, // Default fallback
                        }
                    })
                    .collect();
                QuantumAnsatz::TwoLocal {
                    rotation_blocks,
                    entanglement: "linear".to_string(),
                }
            },
        }
    }
}
