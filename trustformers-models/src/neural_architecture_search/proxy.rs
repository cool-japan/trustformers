//! Real train-and-evaluate loop for candidate architectures.
//!
//! A NAS run is only meaningful if its fitness comes from *measuring* candidates.
//! This module builds a small real network from each candidate architecture,
//! trains it with stochastic gradient descent on a synthetic task, and measures
//! its accuracy on held-out data and its inference latency with a wall clock.
//!
//! The proxy task is deliberately non-linear (`sign(x·a) == sign(x·b)`), so
//! depth, width and the activation function all change the achievable accuracy —
//! a linear model saturates around chance level, a two-layer network does not.
//!
//! Evaluation is deterministic: the RNG is seeded from the architecture itself,
//! so the same architecture always yields the same measurement.

use std::collections::HashMap;
use std::time::Instant;

use trustformers_core::errors::{Result, TrustformersError};

use super::Architecture;

/// What one evaluation of a candidate architecture measured.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredPerformance {
    /// Accuracy on held-out data, in `[0, 1]` — measured, not modelled.
    pub accuracy: f32,
    /// Final training loss.
    pub train_loss: f32,
    /// Number of parameters of the network that was actually trained.
    pub trained_parameters: usize,
    /// Measured wall-clock time of one inference pass over the evaluation set.
    pub inference_seconds: f64,
    /// Extra metrics the evaluator can supply for `OptimizationObjective::Custom`.
    pub custom_metrics: HashMap<String, f32>,
}

/// Evaluates candidate architectures.
///
/// Implement this to plug a real training pipeline into the search; the default
/// [`ProxyTaskEvaluator`] trains a small network on a synthetic task so that the
/// search has real measurements to work with out of the box.
pub trait ArchitectureEvaluator: Send {
    /// Train and evaluate `architecture`, returning measured performance.
    fn evaluate(&mut self, architecture: &Architecture) -> Result<MeasuredPerformance>;

    /// Human-readable description of what the numbers were measured on.
    fn description(&self) -> String {
        "custom architecture evaluator".to_string()
    }
}

/// Configuration of the built-in proxy task.
#[derive(Debug, Clone)]
pub struct ProxyTaskConfig {
    /// Input dimensionality of the synthetic task.
    pub input_dim: usize,
    /// Number of training samples.
    pub train_samples: usize,
    /// Number of held-out evaluation samples.
    pub eval_samples: usize,
    /// Gradient steps to run.
    pub train_steps: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Learning rate for SGD.
    pub learning_rate: f32,
    /// Largest hidden width the proxy will build (candidate widths are scaled
    /// into `[2, max_width]` so a search over 4096-wide transformers stays cheap).
    pub max_width: usize,
    /// Largest depth the proxy will build.
    pub max_depth: usize,
    /// Seed mixed into every evaluation.
    pub seed: u64,
}

impl Default for ProxyTaskConfig {
    fn default() -> Self {
        Self {
            input_dim: 8,
            train_samples: 256,
            eval_samples: 128,
            train_steps: 500,
            batch_size: 32,
            learning_rate: 0.3,
            max_width: 32,
            max_depth: 4,
            seed: 0x5EED,
        }
    }
}

/// Deterministic xorshift RNG — no external dependency, fully reproducible.
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Uniform in `[-1, 1)`.
    fn next_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    fn next_index(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u64() % bound as u64) as usize
        }
    }
}

/// Activation functions the proxy network can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Activation {
    Relu,
    Gelu,
    Tanh,
    Silu,
}

impl Activation {
    fn from_name(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "gelu" | "gelu_new" => Activation::Gelu,
            "tanh" => Activation::Tanh,
            "silu" | "swish" => Activation::Silu,
            _ => Activation::Relu,
        }
    }

    fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Relu => x.max(0.0),
            Activation::Gelu => {
                // tanh approximation of GELU
                let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            },
            Activation::Tanh => x.tanh(),
            Activation::Silu => x / (1.0 + (-x).exp()),
        }
    }

    /// Derivative with respect to the pre-activation, given the pre-activation.
    fn derivative(self, x: f32) -> f32 {
        match self {
            Activation::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            },
            Activation::Gelu => {
                let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
                let tanh = inner.tanh();
                let d_inner = 0.797_884_6 * (1.0 + 3.0 * 0.044_715 * x * x);
                0.5 * (1.0 + tanh) + 0.5 * x * (1.0 - tanh * tanh) * d_inner
            },
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            },
            Activation::Silu => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 + x * (1.0 - sigmoid))
            },
        }
    }
}

/// A dense layer of the proxy network.
struct DenseLayer {
    /// Row-major `[inputs, outputs]`.
    weight: Vec<f32>,
    bias: Vec<f32>,
    inputs: usize,
    outputs: usize,
}

impl DenseLayer {
    fn new(inputs: usize, outputs: usize, rng: &mut Xorshift) -> Self {
        // He-style initialisation keeps deep proxies trainable.
        let scale = (2.0 / inputs as f32).sqrt();
        let weight = (0..inputs * outputs).map(|_| rng.next_signed() * scale).collect();
        Self {
            weight,
            bias: vec![0.0; outputs],
            inputs,
            outputs,
        }
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        output.copy_from_slice(&self.bias);
        for i in 0..self.inputs {
            let x = input[i];
            if x == 0.0 {
                continue;
            }
            let row = i * self.outputs;
            for o in 0..self.outputs {
                output[o] += x * self.weight[row + o];
            }
        }
    }

    fn parameters(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

/// A small multi-layer perceptron built from a candidate architecture.
struct ProxyNetwork {
    layers: Vec<DenseLayer>,
    activation: Activation,
}

impl ProxyNetwork {
    fn new(
        input_dim: usize,
        width: usize,
        depth: usize,
        activation: Activation,
        rng: &mut Xorshift,
    ) -> Self {
        let mut layers = Vec::with_capacity(depth + 1);
        let mut previous = input_dim;
        for _ in 0..depth {
            layers.push(DenseLayer::new(previous, width, rng));
            previous = width;
        }
        // Two-class output head.
        layers.push(DenseLayer::new(previous, 2, rng));
        Self { layers, activation }
    }

    fn parameters(&self) -> usize {
        self.layers.iter().map(DenseLayer::parameters).sum()
    }

    /// Forward pass returning the pre-activations and activations of every layer.
    fn forward(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut pre_activations = Vec::with_capacity(self.layers.len());
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_vec());

        for (index, layer) in self.layers.iter().enumerate() {
            let mut z = vec![0.0; layer.outputs];
            let previous = activations.last().unwrap_or(&activations[0]).clone();
            layer.forward(&previous, &mut z);

            let a = if index + 1 == self.layers.len() {
                z.clone()
            } else {
                z.iter().map(|value| self.activation.apply(*value)).collect()
            };

            pre_activations.push(z);
            activations.push(a);
        }

        (pre_activations, activations)
    }

    /// Class prediction for one sample.
    fn predict(&self, input: &[f32]) -> usize {
        let (_, activations) = self.forward(input);
        let logits = activations.last().cloned().unwrap_or_default();
        if logits.len() < 2 || logits[1] > logits[0] {
            1
        } else {
            0
        }
    }

    /// One SGD step over a mini-batch; returns the mean cross-entropy loss.
    fn train_batch(&mut self, batch: &[(Vec<f32>, usize)], learning_rate: f32) -> f32 {
        let mut weight_gradients: Vec<Vec<f32>> =
            self.layers.iter().map(|layer| vec![0.0; layer.weight.len()]).collect();
        let mut bias_gradients: Vec<Vec<f32>> =
            self.layers.iter().map(|layer| vec![0.0; layer.bias.len()]).collect();
        let mut total_loss = 0.0f32;

        for (input, label) in batch {
            let (pre_activations, activations) = self.forward(input);
            let logits = match activations.last() {
                Some(logits) => logits.clone(),
                None => continue,
            };

            // Softmax cross-entropy.
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probabilities: Vec<f32> =
                exps.iter().map(|e| if sum > 0.0 { e / sum } else { 0.0 }).collect();
            let target = (*label).min(probabilities.len().saturating_sub(1));
            total_loss -= probabilities.get(target).copied().unwrap_or(1e-12).max(1e-12).ln();

            // dL/dlogits
            let mut delta: Vec<f32> = probabilities.clone();
            if let Some(value) = delta.get_mut(target) {
                *value -= 1.0;
            }

            // Backward through the layers.
            for index in (0..self.layers.len()).rev() {
                let layer = &self.layers[index];
                let input_activation = &activations[index];

                for i in 0..layer.inputs {
                    let x = input_activation[i];
                    if x == 0.0 {
                        continue;
                    }
                    let row = i * layer.outputs;
                    for o in 0..layer.outputs {
                        weight_gradients[index][row + o] += x * delta[o];
                    }
                }
                for o in 0..layer.outputs {
                    bias_gradients[index][o] += delta[o];
                }

                if index == 0 {
                    break;
                }

                // Propagate to the previous layer.
                let mut previous_delta = vec![0.0f32; layer.inputs];
                for i in 0..layer.inputs {
                    let row = i * layer.outputs;
                    let mut sum = 0.0;
                    for o in 0..layer.outputs {
                        sum += layer.weight[row + o] * delta[o];
                    }
                    previous_delta[i] =
                        sum * self.activation.derivative(pre_activations[index - 1][i]);
                }
                delta = previous_delta;
            }
        }

        let scale = learning_rate / batch.len().max(1) as f32;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            for (weight, gradient) in layer.weight.iter_mut().zip(weight_gradients[index].iter()) {
                *weight -= scale * gradient;
            }
            for (bias, gradient) in layer.bias.iter_mut().zip(bias_gradients[index].iter()) {
                *bias -= scale * gradient;
            }
        }

        total_loss / batch.len().max(1) as f32
    }
}

/// The default evaluator: builds, trains and measures a real small network.
pub struct ProxyTaskEvaluator {
    config: ProxyTaskConfig,
    train_set: Vec<(Vec<f32>, usize)>,
    eval_set: Vec<(Vec<f32>, usize)>,
}

impl ProxyTaskEvaluator {
    /// Create an evaluator with the default proxy task.
    pub fn new() -> Self {
        Self::with_config(ProxyTaskConfig::default())
    }

    /// Create an evaluator with a custom proxy task.
    pub fn with_config(config: ProxyTaskConfig) -> Self {
        let (train_set, eval_set) = Self::build_dataset(&config);
        Self {
            config,
            train_set,
            eval_set,
        }
    }

    /// Build the synthetic dataset once; every candidate sees the same data.
    fn build_dataset(config: &ProxyTaskConfig) -> (Vec<(Vec<f32>, usize)>, Vec<(Vec<f32>, usize)>) {
        let mut rng = Xorshift::new(config.seed);

        // Two fixed random projections define a non-linear (XOR-like) boundary.
        let a: Vec<f32> = (0..config.input_dim).map(|_| rng.next_signed()).collect();
        let b: Vec<f32> = (0..config.input_dim).map(|_| rng.next_signed()).collect();

        let sample = |rng: &mut Xorshift| {
            let x: Vec<f32> = (0..config.input_dim).map(|_| rng.next_signed()).collect();
            let projection_a: f32 = x.iter().zip(a.iter()).map(|(x, a)| x * a).sum();
            let projection_b: f32 = x.iter().zip(b.iter()).map(|(x, b)| x * b).sum();
            let label = usize::from((projection_a > 0.0) == (projection_b > 0.0));
            (x, label)
        };

        let train = (0..config.train_samples).map(|_| sample(&mut rng)).collect();
        let eval = (0..config.eval_samples).map(|_| sample(&mut rng)).collect();
        (train, eval)
    }

    /// Map a candidate architecture onto the proxy network's shape.
    fn proxy_shape(&self, architecture: &Architecture) -> (usize, usize, Activation) {
        let hidden = architecture.dimensions.get("hidden_size").copied().unwrap_or(64).max(1);
        // `num_layers <= 0` builds a purely linear model, which is a meaningful
        // (and measurably weaker) candidate on this non-linear task.
        let layers = architecture.dimensions.get("num_layers").copied().unwrap_or(2).max(0);

        // Scale the candidate's width logarithmically into the proxy's budget so
        // that "bigger" stays bigger without training a 4096-wide network.
        let width =
            (((hidden as f32).log2().max(1.0) * 2.0) as usize).clamp(2, self.config.max_width);
        let depth = (layers.max(0) as usize).min(self.config.max_depth);

        let activation = architecture
            .choices
            .get("activation")
            .map(|name| Activation::from_name(name))
            .unwrap_or(Activation::Relu);

        (width, depth, activation)
    }

    /// Deterministic seed derived from the architecture.
    fn seed_for(&self, architecture: &Architecture) -> u64 {
        let mut hash = self.config.seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut dimensions: Vec<(&String, &i32)> = architecture.dimensions.iter().collect();
        dimensions.sort();
        for (name, value) in dimensions {
            for byte in name.as_bytes() {
                hash = hash.rotate_left(5) ^ u64::from(*byte);
            }
            hash = hash.rotate_left(7) ^ (*value as u64);
        }
        let mut choices: Vec<(&String, &String)> = architecture.choices.iter().collect();
        choices.sort();
        for (name, value) in choices {
            for byte in name.as_bytes().iter().chain(value.as_bytes()) {
                hash = hash.rotate_left(3) ^ u64::from(*byte);
            }
        }
        hash | 1
    }
}

impl Default for ProxyTaskEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchitectureEvaluator for ProxyTaskEvaluator {
    fn evaluate(&mut self, architecture: &Architecture) -> Result<MeasuredPerformance> {
        if self.train_set.is_empty() || self.eval_set.is_empty() {
            return Err(TrustformersError::invalid_config(
                "the proxy task has no data to train or evaluate on".to_string(),
            ));
        }

        let (width, depth, activation) = self.proxy_shape(architecture);
        let mut rng = Xorshift::new(self.seed_for(architecture));
        let mut network =
            ProxyNetwork::new(self.config.input_dim, width, depth, activation, &mut rng);

        // --- Train on the training split -----------------------------------
        let mut train_loss = f32::NAN;
        for _ in 0..self.config.train_steps {
            let batch: Vec<(Vec<f32>, usize)> = (0..self.config.batch_size)
                .map(|_| {
                    let index = rng.next_index(self.train_set.len());
                    self.train_set[index].clone()
                })
                .collect();
            train_loss = network.train_batch(&batch, self.config.learning_rate);
            if !train_loss.is_finite() {
                return Err(TrustformersError::invalid_operation(format!(
                    "the proxy network diverged while training architecture {}",
                    architecture.metadata.id
                )));
            }
        }

        // --- Measure accuracy on the held-out split ------------------------
        let start = Instant::now();
        let mut correct = 0usize;
        for (input, label) in &self.eval_set {
            if network.predict(input) == *label {
                correct += 1;
            }
        }
        let inference_seconds = start.elapsed().as_secs_f64();

        Ok(MeasuredPerformance {
            accuracy: correct as f32 / self.eval_set.len() as f32,
            train_loss,
            trained_parameters: network.parameters(),
            inference_seconds,
            custom_metrics: HashMap::new(),
        })
    }

    fn description(&self) -> String {
        format!(
            "proxy task: {}-dimensional non-linear binary classification, {} train / {} held-out \
             samples, {} SGD steps",
            self.config.input_dim,
            self.config.train_samples,
            self.config.eval_samples,
            self.config.train_steps
        )
    }
}

#[cfg(test)]
#[path = "proxy_tests.rs"]
mod tests;
