use serde::{Deserialize, Serialize};
use trustformers_core::{
    errors::{invalid_config, Result},
    traits::Config,
};

/// Biologically-inspired network architecture type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BiologicalArchitecture {
    /// Spiking Neural Networks with temporal dynamics
    SpikingNeuralNetwork,
    /// Hopfield Networks for associative memory
    HopfieldNetwork,
    /// Liquid Time-Constant Networks with adaptive dynamics
    LiquidTimeConstant,
    /// Neural Turing Machines with external memory
    NeuralTuringMachine,
    /// Reservoir Computing with echo state networks
    ReservoirComputing,
    /// Capsule Networks with part-whole relationships
    CapsuleNetwork,
    /// Dendritic computation with multi-compartment neurons
    DendriticComputation,
    /// Biological memory models with synaptic plasticity
    BiologicalMemory,
}

/// Neuron model type for spiking networks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NeuronModel {
    /// Leaky Integrate-and-Fire
    LeakyIntegrateAndFire,
    /// Izhikevich model
    Izhikevich,
    /// Hodgkin-Huxley model
    HodgkinHuxley,
    /// Adaptive Exponential Integrate-and-Fire
    AdaptiveExponentialIF,
    /// Spike Response Model
    SpikeResponseModel,
}

/// Synaptic plasticity type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PlasticityType {
    /// Spike-Timing Dependent Plasticity
    STDP,
    /// Hebbian plasticity
    Hebbian,
    /// Anti-Hebbian plasticity
    AntiHebbian,
    /// Homeostatic plasticity
    Homeostatic,
    /// Metaplasticity
    Metaplasticity,
}

/// Memory type for biological memory models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryType {
    /// Short-term memory
    ShortTerm,
    /// Long-term memory
    LongTerm,
    /// Working memory
    Working,
    /// Episodic memory
    Episodic,
    /// Semantic memory
    Semantic,
}

/// Biologically-inspired model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalConfig {
    /// Architecture type
    pub architecture: BiologicalArchitecture,
    /// Model dimension
    pub d_model: usize,
    /// Number of layers
    pub n_layer: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Neuron model type (for spiking networks)
    pub neuron_model: NeuronModel,
    /// Plasticity type
    pub plasticity_type: PlasticityType,
    /// Memory type
    pub memory_type: MemoryType,
    /// Number of neurons per layer
    pub neurons_per_layer: usize,
    /// Time step size for temporal dynamics
    pub dt: f32,
    /// Membrane time constant
    pub tau_mem: f32,
    /// Synaptic time constant
    pub tau_syn: f32,
    /// Threshold voltage for spiking
    pub v_threshold: f32,
    /// Reset voltage
    pub v_reset: f32,
    /// Refractory period
    pub refractory_period: f32,
    /// Learning rate for plasticity
    pub learning_rate: f32,
    /// STDP time window
    pub stdp_window: f32,
    /// Homeostatic target rate
    pub target_rate: f32,
    /// Noise variance
    pub noise_variance: f32,
    /// Memory capacity (for Hopfield networks)
    pub memory_capacity: usize,
    /// Reservoir size (for reservoir computing)
    pub reservoir_size: usize,
    /// Spectral radius (for reservoir computing)
    pub spectral_radius: f32,
    /// Input scaling
    pub input_scaling: f32,
    /// Leak rate (for LTC networks)
    pub leak_rate: f32,
    /// Number of capsules (for capsule networks)
    pub num_capsules: usize,
    /// Capsule dimension
    pub capsule_dim: usize,
    /// Routing iterations
    pub routing_iterations: usize,
    /// Number of dendritic compartments
    pub num_compartments: usize,
    /// Dendritic delay
    pub dendritic_delay: f32,
    /// Whether to use bias
    pub use_bias: bool,
    /// Initializer range
    pub initializer_range: f32,
    /// Model type identifier
    pub model_type: String,
}

impl Default for BiologicalConfig {
    fn default() -> Self {
        Self {
            architecture: BiologicalArchitecture::SpikingNeuralNetwork,
            d_model: 768,
            n_layer: 12,
            vocab_size: 50000,
            max_position_embeddings: 2048,
            neuron_model: NeuronModel::LeakyIntegrateAndFire,
            plasticity_type: PlasticityType::STDP,
            memory_type: MemoryType::ShortTerm,
            neurons_per_layer: 1000,
            dt: 0.001,
            tau_mem: 0.02,
            tau_syn: 0.005,
            v_threshold: 1.0,
            v_reset: 0.0,
            refractory_period: 0.002,
            learning_rate: 0.001,
            stdp_window: 0.02,
            target_rate: 10.0,
            noise_variance: 0.01,
            memory_capacity: 1000,
            reservoir_size: 1000,
            spectral_radius: 0.9,
            input_scaling: 1.0,
            leak_rate: 0.1,
            num_capsules: 10,
            capsule_dim: 16,
            routing_iterations: 3,
            num_compartments: 5,
            dendritic_delay: 0.001,
            use_bias: true,
            initializer_range: 0.02,
            model_type: "biological".to_string(),
        }
    }
}

impl BiologicalConfig {
    /// Create a spiking neural network configuration
    pub fn spiking_neural_network() -> Self {
        Self {
            architecture: BiologicalArchitecture::SpikingNeuralNetwork,
            neuron_model: NeuronModel::LeakyIntegrateAndFire,
            plasticity_type: PlasticityType::STDP,
            ..Default::default()
        }
    }

    /// Create a Hopfield network configuration
    pub fn hopfield_network() -> Self {
        Self {
            architecture: BiologicalArchitecture::HopfieldNetwork,
            memory_capacity: 1000,
            plasticity_type: PlasticityType::Hebbian,
            ..Default::default()
        }
    }

    /// Create a liquid time-constant network configuration
    pub fn liquid_time_constant() -> Self {
        Self {
            architecture: BiologicalArchitecture::LiquidTimeConstant,
            leak_rate: 0.1,
            reservoir_size: 1000,
            ..Default::default()
        }
    }

    /// Create a neural Turing machine configuration
    pub fn neural_turing_machine() -> Self {
        Self {
            architecture: BiologicalArchitecture::NeuralTuringMachine,
            memory_capacity: 128,
            memory_type: MemoryType::Working,
            ..Default::default()
        }
    }

    /// Create a reservoir computing configuration
    pub fn reservoir_computing() -> Self {
        Self {
            architecture: BiologicalArchitecture::ReservoirComputing,
            reservoir_size: 1000,
            spectral_radius: 0.9,
            input_scaling: 1.0,
            ..Default::default()
        }
    }

    /// Create a capsule network configuration
    pub fn capsule_network() -> Self {
        Self {
            architecture: BiologicalArchitecture::CapsuleNetwork,
            num_capsules: 10,
            capsule_dim: 16,
            routing_iterations: 3,
            ..Default::default()
        }
    }

    /// Create a dendritic computation configuration
    pub fn dendritic_computation() -> Self {
        Self {
            architecture: BiologicalArchitecture::DendriticComputation,
            num_compartments: 5,
            dendritic_delay: 0.001,
            ..Default::default()
        }
    }

    /// Create a biological memory configuration
    pub fn biological_memory() -> Self {
        Self {
            architecture: BiologicalArchitecture::BiologicalMemory,
            memory_type: MemoryType::Episodic,
            plasticity_type: PlasticityType::Metaplasticity,
            ..Default::default()
        }
    }

    /// Create configuration from pretrained model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            "spiking-neural-network" => Some(Self::spiking_neural_network()),
            "hopfield-network" => Some(Self::hopfield_network()),
            "liquid-time-constant" => Some(Self::liquid_time_constant()),
            "neural-turing-machine" => Some(Self::neural_turing_machine()),
            "reservoir-computing" => Some(Self::reservoir_computing()),
            "capsule-network" => Some(Self::capsule_network()),
            "dendritic-computation" => Some(Self::dendritic_computation()),
            "biological-memory" => Some(Self::biological_memory()),
            _ => None,
        }
    }

    /// Get the effective time constant based on neuron model
    pub fn get_effective_tau(&self) -> f32 {
        match self.neuron_model {
            NeuronModel::LeakyIntegrateAndFire => self.tau_mem,
            NeuronModel::Izhikevich => self.tau_mem * 0.5,
            NeuronModel::HodgkinHuxley => self.tau_mem * 2.0,
            NeuronModel::AdaptiveExponentialIF => self.tau_mem * 1.5,
            NeuronModel::SpikeResponseModel => self.tau_syn,
        }
    }

    /// Get the plasticity learning window
    pub fn get_plasticity_window(&self) -> f32 {
        match self.plasticity_type {
            PlasticityType::STDP => self.stdp_window,
            PlasticityType::Hebbian => self.stdp_window * 2.0,
            PlasticityType::AntiHebbian => self.stdp_window * 2.0,
            PlasticityType::Homeostatic => self.stdp_window * 10.0,
            PlasticityType::Metaplasticity => self.stdp_window * 5.0,
        }
    }
}

impl Config for BiologicalConfig {
    fn architecture(&self) -> &'static str {
        match self.architecture {
            BiologicalArchitecture::SpikingNeuralNetwork => "spiking_neural_network",
            BiologicalArchitecture::HopfieldNetwork => "hopfield_network",
            BiologicalArchitecture::LiquidTimeConstant => "liquid_time_constant",
            BiologicalArchitecture::NeuralTuringMachine => "neural_turing_machine",
            BiologicalArchitecture::ReservoirComputing => "reservoir_computing",
            BiologicalArchitecture::CapsuleNetwork => "capsule_network",
            BiologicalArchitecture::DendriticComputation => "dendritic_computation",
            BiologicalArchitecture::BiologicalMemory => "biological_memory",
        }
    }

    fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            return Err(invalid_config(
                "config_field",
                "d_model must be greater than 0",
            ));
        }
        if self.n_layer == 0 {
            return Err(invalid_config(
                "config_field",
                "n_layer must be greater than 0",
            ));
        }
        if self.vocab_size == 0 {
            return Err(invalid_config(
                "config_field",
                "vocab_size must be greater than 0",
            ));
        }
        if self.neurons_per_layer == 0 {
            return Err(invalid_config(
                "config_field",
                "neurons_per_layer must be greater than 0",
            ));
        }
        if self.dt <= 0.0 {
            return Err(invalid_config("config_field", "dt must be greater than 0"));
        }
        if self.tau_mem <= 0.0 {
            return Err(invalid_config(
                "config_field",
                "tau_mem must be greater than 0",
            ));
        }
        if self.tau_syn <= 0.0 {
            return Err(invalid_config(
                "config_field",
                "tau_syn must be greater than 0",
            ));
        }
        if self.v_threshold <= self.v_reset {
            return Err(invalid_config(
                "config_field",
                "v_threshold must be greater than v_reset",
            ));
        }
        if self.refractory_period < 0.0 {
            return Err(invalid_config(
                "config_field",
                "refractory_period must be non-negative",
            ));
        }
        if self.learning_rate <= 0.0 {
            return Err(invalid_config(
                "config_field",
                "learning_rate must be greater than 0",
            ));
        }
        if self.spectral_radius <= 0.0 || self.spectral_radius >= 1.0 {
            return Err(invalid_config(
                "config_field",
                "spectral_radius must be between 0 and 1",
            ));
        }
        if self.leak_rate <= 0.0 || self.leak_rate > 1.0 {
            return Err(invalid_config(
                "config_field",
                "leak_rate must be between 0 and 1",
            ));
        }
        if self.num_capsules == 0 {
            return Err(invalid_config(
                "config_field",
                "num_capsules must be greater than 0",
            ));
        }
        if self.capsule_dim == 0 {
            return Err(invalid_config(
                "config_field",
                "capsule_dim must be greater than 0",
            ));
        }
        if self.routing_iterations == 0 {
            return Err(invalid_config(
                "config_field",
                "routing_iterations must be greater than 0",
            ));
        }
        if self.num_compartments == 0 {
            return Err(invalid_config(
                "config_field",
                "num_compartments must be greater than 0",
            ));
        }
        Ok(())
    }
}
