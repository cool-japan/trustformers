//! Concrete meta-learning models.
//!
//! Every model in this module is backed by [`MlpLearner`], a small
//! fully-connected network with hand-written forward and backward passes. The
//! gradients it returns are exact analytic gradients of the loss it reports,
//! which is what makes the inner/outer loops in
//! [`crate::meta_learning::MetaLearner`] real learning rather than bookkeeping.
//!
//! ## Which algorithms are implemented
//!
//! | Model | Status |
//! |---|---|
//! | [`MAMLModel`] | full: first-order (FOMAML) and second-order meta-gradients |
//! | [`ReptileModel`] | full: the outer step is the parameter difference |
//! | [`PrototypicalModel`] | full: real embeddings, class prototypes, distance classification |
//! | [`MatchingNetModel`] | full: cosine attention over support embeddings |
//! | [`RelationNetModel`] | full: learned embedding + relation score |
//! | [`MetaSGDModel`] | full: per-parameter learned learning rates |
//! | [`MemoryAugmentedModel`] | learner only; the external memory is not implemented |
//! | [`GradientBasedModel`] | learner only; the learned optimizer is not implemented |
//! | [`L2LModel`] | learner only; the LSTM meta-learner is not implemented |
//!
//! The last three inherit the default trait methods, which report
//! `unsupported_operation` for the parts that do not exist. They never return
//! invented metrics.

use std::collections::HashMap;

use trustformers_core::{
    errors::{invalid_input, TrustformersError},
    tensor::Tensor,
};

use super::{
    Example, ExampleSet, MetaLearningConfig, MetaLearningModel, ModelGradients, ModelParameters,
};

/// Step size for the central-difference Hessian-vector product used by the
/// second-order MAML meta-gradient.
const HVP_EPSILON: f64 = 1e-3;

/// A dense layer with an explicit backward pass.
#[derive(Debug, Clone)]
struct DenseLayer {
    /// Row-major `[out_dim, in_dim]`
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl DenseLayer {
    /// Deterministic fan-in scaled initialisation.
    ///
    /// The weights follow a fixed low-discrepancy pattern rather than an RNG so
    /// that two learners built from the same configuration start identical,
    /// which is what the meta-learning outer loop assumes.
    fn new(in_dim: usize, out_dim: usize, seed: usize) -> Self {
        let scale = (1.0 / in_dim.max(1) as f32).sqrt();
        let mut weight = Vec::with_capacity(in_dim * out_dim);
        for index in 0..in_dim * out_dim {
            let phase = ((index + seed * 7 + 1) as f32) * 0.754_877_7;
            weight.push(((phase.fract() * 2.0) - 1.0) * scale);
        }
        Self {
            weight,
            bias: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = self.bias.clone();
        for row in 0..self.out_dim {
            let offset = row * self.in_dim;
            let mut sum = output[row];
            for column in 0..self.in_dim {
                sum += self.weight[offset + column] * input[column];
            }
            output[row] = sum;
        }
        output
    }
}

/// A small multi-layer perceptron with analytic gradients.
///
/// The hidden layers use `tanh`; the output layer is linear. Classification
/// examples are scored with softmax cross-entropy and regression examples with
/// mean squared error.
#[derive(Debug, Clone)]
pub struct MlpLearner {
    layers: Vec<DenseLayer>,
}

impl MlpLearner {
    /// Build a learner mapping `input_dim` features to `output_dim` outputs
    /// through one hidden layer of `hidden_dim` units.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> Result<Self, TrustformersError> {
        if input_dim == 0 || hidden_dim == 0 || output_dim == 0 {
            return Err(invalid_input(
                "MlpLearner dimensions must all be greater than zero",
            ));
        }
        Ok(Self {
            layers: vec![
                DenseLayer::new(input_dim, hidden_dim, 0),
                DenseLayer::new(hidden_dim, output_dim, 1),
            ],
        })
    }

    /// Width of the network's input.
    pub fn input_dim(&self) -> usize {
        self.layers[0].in_dim
    }

    /// Width of the network's output.
    pub fn output_dim(&self) -> usize {
        self.layers[self.layers.len() - 1].out_dim
    }

    /// Forward pass returning the pre-activation of every layer plus the
    /// post-activation of the hidden layers.
    fn forward_cached(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut activations = vec![input.to_vec()];
        let mut current = input.to_vec();

        for (index, layer) in self.layers.iter().enumerate() {
            let mut output = layer.forward(&current);
            if index + 1 < self.layers.len() {
                for value in output.iter_mut() {
                    *value = value.tanh();
                }
            }
            activations.push(output.clone());
            current = output;
        }

        (activations, current)
    }

    /// Network output for one input vector.
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        self.forward_cached(input).1
    }

    /// Hidden representation used as the embedding by metric-based algorithms.
    pub fn embed(&self, input: &[f32]) -> Vec<f32> {
        let (activations, _) = self.forward_cached(input);
        activations[activations.len() - 2].clone()
    }

    /// Backpropagate `output_gradient` through the network, accumulating into
    /// `weight_grads` / `bias_grads`.
    fn backward(
        &self,
        activations: &[Vec<f32>],
        output_gradient: &[f32],
        weight_grads: &mut [Vec<f32>],
        bias_grads: &mut [Vec<f32>],
    ) {
        let mut delta = output_gradient.to_vec();

        for index in (0..self.layers.len()).rev() {
            let layer = &self.layers[index];
            let input = &activations[index];

            for row in 0..layer.out_dim {
                bias_grads[index][row] += delta[row];
                let offset = row * layer.in_dim;
                for column in 0..layer.in_dim {
                    weight_grads[index][offset + column] += delta[row] * input[column];
                }
            }

            if index == 0 {
                break;
            }

            // Propagate through the previous layer's tanh.
            let mut next_delta = vec![0.0f32; layer.in_dim];
            for column in 0..layer.in_dim {
                let mut sum = 0.0f32;
                for row in 0..layer.out_dim {
                    sum += layer.weight[row * layer.in_dim + column] * delta[row];
                }
                // activations[index] is the tanh output of the previous layer.
                let activation = activations[index][column];
                next_delta[column] = sum * (1.0 - activation * activation);
            }
            delta = next_delta;
        }
    }

    fn zero_gradients(&self) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        (
            self.layers.iter().map(|l| vec![0.0f32; l.weight.len()]).collect(),
            self.layers.iter().map(|l| vec![0.0f32; l.bias.len()]).collect(),
        )
    }

    /// Mean loss over a set of examples.
    pub fn loss(&self, examples: &ExampleSet) -> Result<f64, TrustformersError> {
        let (loss, _) = self.loss_and_gradients(examples, false)?;
        Ok(loss)
    }

    /// Mean loss and, when requested, the analytic gradient of that loss.
    fn loss_and_gradients(
        &self,
        examples: &ExampleSet,
        want_gradients: bool,
    ) -> Result<(f64, Option<ModelGradients>), TrustformersError> {
        if examples.examples.is_empty() {
            return Err(invalid_input("cannot score an empty example set"));
        }

        let (mut weight_grads, mut bias_grads) = self.zero_gradients();
        let mut total_loss = 0.0f64;
        let count = examples.examples.len() as f32;

        for example in &examples.examples {
            let input = self.example_input(example)?;
            let (activations, output) = self.forward_cached(&input);

            let output_gradient = match &example.target {
                Some(target) => {
                    // Regression: mean squared error.
                    let target_values = target.to_vec_f32()?;
                    if target_values.len() != output.len() {
                        return Err(invalid_input(format!(
                            "regression target has {} values but the learner produces {}",
                            target_values.len(),
                            output.len()
                        )));
                    }
                    let mut gradient = vec![0.0f32; output.len()];
                    for ((slot, prediction), truth) in
                        gradient.iter_mut().zip(output.iter()).zip(target_values.iter())
                    {
                        let residual = prediction - truth;
                        total_loss += (residual * residual) as f64;
                        *slot = 2.0 * residual / count;
                    }
                    gradient
                },
                None => {
                    // Classification: softmax cross-entropy.
                    if example.label >= output.len() {
                        return Err(invalid_input(format!(
                            "label {} is out of range for {} outputs",
                            example.label,
                            output.len()
                        )));
                    }
                    let probabilities = softmax(&output);
                    total_loss -= (probabilities[example.label].max(1e-12) as f64).ln();
                    let mut gradient = probabilities;
                    gradient[example.label] -= 1.0;
                    for value in gradient.iter_mut() {
                        *value /= count;
                    }
                    gradient
                },
            };

            if want_gradients {
                self.backward(
                    &activations,
                    &output_gradient,
                    &mut weight_grads,
                    &mut bias_grads,
                );
            }
        }

        let mean_loss = total_loss / count as f64;
        if !want_gradients {
            return Ok((mean_loss, None));
        }

        let mut gradients = ModelGradients::new();
        for (index, layer) in self.layers.iter().enumerate() {
            gradients.gradients.insert(
                format!("layer{}.weight", index),
                Tensor::from_vec(
                    std::mem::take(&mut weight_grads[index]),
                    &[layer.out_dim, layer.in_dim],
                )?,
            );
            gradients.gradients.insert(
                format!("layer{}.bias", index),
                Tensor::from_vec(std::mem::take(&mut bias_grads[index]), &[layer.out_dim])?,
            );
        }

        Ok((mean_loss, Some(gradients)))
    }

    /// Analytic gradient of the mean loss over `examples`.
    pub fn gradients(&self, examples: &ExampleSet) -> Result<ModelGradients, TrustformersError> {
        let (_, gradients) = self.loss_and_gradients(examples, true)?;
        gradients.ok_or_else(|| invalid_input("gradient computation produced no gradients"))
    }

    /// Classification accuracy, or the coefficient of determination for
    /// regression example sets.
    pub fn accuracy(&self, examples: &ExampleSet) -> Result<f64, TrustformersError> {
        if examples.examples.is_empty() {
            return Err(invalid_input("cannot score an empty example set"));
        }

        let regression = examples.examples.iter().all(|e| e.target.is_some());
        if regression {
            let mut residual = 0.0f64;
            let mut targets = Vec::new();
            for example in &examples.examples {
                let input = self.example_input(example)?;
                let prediction = self.predict(&input);
                let Some(target) = &example.target else {
                    return Err(invalid_input("regression example is missing its target"));
                };
                let truth = target.to_vec_f32()?;
                for (p, t) in prediction.iter().zip(truth.iter()) {
                    residual += ((p - t) * (p - t)) as f64;
                    targets.push(*t as f64);
                }
            }
            let mean = targets.iter().sum::<f64>() / targets.len().max(1) as f64;
            let variance: f64 = targets.iter().map(|t| (t - mean) * (t - mean)).sum();
            if variance <= f64::EPSILON {
                return Ok(if residual <= f64::EPSILON { 1.0 } else { 0.0 });
            }
            return Ok((1.0 - residual / variance).clamp(0.0, 1.0));
        }

        let mut correct = 0usize;
        for example in &examples.examples {
            let input = self.example_input(example)?;
            let prediction = self.predict(&input);
            if argmax(&prediction) == example.label {
                correct += 1;
            }
        }
        Ok(correct as f64 / examples.examples.len() as f64)
    }

    /// Flattened input vector of an example, validated against the input width.
    fn example_input(&self, example: &Example) -> Result<Vec<f32>, TrustformersError> {
        let values = example.input.to_vec_f32()?;
        if values.len() != self.input_dim() {
            return Err(invalid_input(format!(
                "example has {} features but the learner expects {}",
                values.len(),
                self.input_dim()
            )));
        }
        Ok(values)
    }

    /// Every parameter tensor, named `layer{i}.weight` / `layer{i}.bias`.
    pub fn parameters(&self) -> Result<ModelParameters, TrustformersError> {
        let mut parameters = HashMap::new();
        for (index, layer) in self.layers.iter().enumerate() {
            parameters.insert(
                format!("layer{}.weight", index),
                Tensor::from_vec(layer.weight.clone(), &[layer.out_dim, layer.in_dim])?,
            );
            parameters.insert(
                format!("layer{}.bias", index),
                Tensor::from_vec(layer.bias.clone(), &[layer.out_dim])?,
            );
        }
        Ok(ModelParameters { parameters })
    }

    /// Overwrite the parameters from a [`ModelParameters`] map.
    pub fn set_parameters(&mut self, params: &ModelParameters) -> Result<(), TrustformersError> {
        for (index, layer) in self.layers.iter_mut().enumerate() {
            if let Some(weight) = params.parameters.get(&format!("layer{}.weight", index)) {
                let values = weight.to_vec_f32()?;
                if values.len() != layer.weight.len() {
                    return Err(invalid_input(format!(
                        "layer{}.weight has {} values, expected {}",
                        index,
                        values.len(),
                        layer.weight.len()
                    )));
                }
                layer.weight = values;
            }
            if let Some(bias) = params.parameters.get(&format!("layer{}.bias", index)) {
                let values = bias.to_vec_f32()?;
                if values.len() != layer.bias.len() {
                    return Err(invalid_input(format!(
                        "layer{}.bias has {} values, expected {}",
                        index,
                        values.len(),
                        layer.bias.len()
                    )));
                }
                layer.bias = values;
            }
        }
        Ok(())
    }

    /// Apply `theta -= lr * gradient` for every named parameter.
    pub fn apply_gradients(
        &mut self,
        gradients: &ModelGradients,
        lr: f64,
    ) -> Result<(), TrustformersError> {
        for (index, layer) in self.layers.iter_mut().enumerate() {
            if let Some(gradient) = gradients.gradients.get(&format!("layer{}.weight", index)) {
                let values = gradient.to_vec_f32()?;
                if values.len() != layer.weight.len() {
                    return Err(invalid_input(format!(
                        "gradient for layer{}.weight has the wrong length",
                        index
                    )));
                }
                for (parameter, g) in layer.weight.iter_mut().zip(values) {
                    *parameter -= lr as f32 * g;
                }
            }
            if let Some(gradient) = gradients.gradients.get(&format!("layer{}.bias", index)) {
                let values = gradient.to_vec_f32()?;
                for (parameter, g) in layer.bias.iter_mut().zip(values) {
                    *parameter -= lr as f32 * g;
                }
            }
        }
        Ok(())
    }

    /// Apply per-parameter learning rates, consumed in the deterministic order
    /// `layer0.weight, layer0.bias, layer1.weight, …`.
    pub fn apply_gradients_with_lr(
        &mut self,
        gradients: &ModelGradients,
        learning_rates: &[f64],
    ) -> Result<(), TrustformersError> {
        let mut cursor = 0usize;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            for (name, slots) in [
                (format!("layer{}.weight", index), layer.weight.len()),
                (format!("layer{}.bias", index), layer.bias.len()),
            ] {
                let Some(gradient) = gradients.gradients.get(&name) else {
                    cursor += slots;
                    continue;
                };
                let values = gradient.to_vec_f32()?;
                for (offset, g) in values.iter().enumerate() {
                    let lr = learning_rates.get(cursor + offset).copied().ok_or_else(|| {
                        invalid_input("learned learning-rate vector is shorter than the parameters")
                    })? as f32;
                    if name.ends_with("weight") {
                        layer.weight[offset] -= lr * g;
                    } else {
                        layer.bias[offset] -= lr * g;
                    }
                }
                cursor += slots;
            }
        }
        Ok(())
    }

    /// Total number of scalar parameters.
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|l| l.weight.len() + l.bias.len()).sum()
    }
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let total: f32 = exponentials.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / values.len().max(1) as f32; values.len()];
    }
    exponentials.into_iter().map(|e| e / total).collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

/// Shared state behind every concrete meta-learning model.
#[derive(Debug, Clone)]
pub struct MetaMlp {
    config: MetaLearningConfig,
    learner: MlpLearner,
    /// Most recent batch passed to `forward` (the query set in MAML).
    last_batch: Option<ExampleSet>,
    /// Batch passed to `forward` before that (the support set in MAML).
    previous_batch: Option<ExampleSet>,
    /// Per-parameter learned learning rates (Meta-SGD).
    learning_rates: Vec<f64>,
}

impl MetaMlp {
    fn new(config: &MetaLearningConfig) -> Result<Self, TrustformersError> {
        let learner = MlpLearner::new(
            config.embedding_dim,
            config.embedding_dim.clamp(4, 64),
            config.num_ways.max(1),
        )?;
        let learning_rates = vec![config.inner_lr; learner.parameter_count()];
        Ok(Self {
            config: config.clone(),
            learner,
            last_batch: None,
            previous_batch: None,
            learning_rates,
        })
    }

    fn cached_batch(&self) -> Result<&ExampleSet, TrustformersError> {
        self.last_batch
            .as_ref()
            .ok_or_else(|| invalid_input("compute_gradients requires a preceding call to forward"))
    }
}

impl MetaLearningModel for MetaMlp {
    fn forward(&mut self, examples: &ExampleSet) -> Result<f64, TrustformersError> {
        let loss = self.learner.loss(examples)?;
        self.previous_batch = self.last_batch.take();
        self.last_batch = Some(examples.clone());
        Ok(loss)
    }

    fn compute_accuracy(&self, examples: &ExampleSet) -> Result<f64, TrustformersError> {
        self.learner.accuracy(examples)
    }

    fn compute_gradients(&self, _loss: f64) -> Result<ModelGradients, TrustformersError> {
        // Gradients are the analytic gradients of the loss reported by the most
        // recent `forward`; `_loss` is that scalar and carries no extra
        // information.
        self.learner.gradients(self.cached_batch()?)
    }

    fn apply_gradients(
        &mut self,
        gradients: &ModelGradients,
        lr: f64,
    ) -> Result<(), TrustformersError> {
        self.learner.apply_gradients(gradients, lr)
    }

    fn get_parameters(&self) -> Result<ModelParameters, TrustformersError> {
        self.learner.parameters()
    }

    fn set_parameters(&mut self, params: ModelParameters) -> Result<(), TrustformersError> {
        self.learner.set_parameters(&params)
    }

    fn embed(&self, example: &Example) -> Result<Tensor, TrustformersError> {
        let input = self.learner.example_input(example)?;
        let embedding = self.learner.embed(&input);
        let length = embedding.len();
        let tensor = Tensor::from_vec(embedding, &[length])?;
        Ok(tensor)
    }

    fn compute_first_order_gradients(
        &self,
        _loss: f64,
    ) -> Result<ModelGradients, TrustformersError> {
        // FOMAML: the meta-gradient is the query-set gradient evaluated at the
        // adapted parameters, ignoring the second-order term.
        self.learner.gradients(self.cached_batch()?)
    }

    fn compute_second_order_gradients(
        &self,
        _initial_params: &ModelParameters,
        _loss: f64,
    ) -> Result<ModelGradients, TrustformersError> {
        // One-step MAML meta-gradient  g = v - alpha * H v  with
        // v = grad L_query(theta') and H the Hessian of the support loss,
        // estimated by a central difference of the support gradient.
        let query = self.cached_batch()?;
        let Some(support) = self.previous_batch.as_ref() else {
            return Err(invalid_input(
                "second-order meta-gradients need both the support and the query forward pass",
            ));
        };

        let v = self.learner.gradients(query)?;
        let epsilon = HVP_EPSILON;

        let mut plus = self.learner.clone();
        plus.apply_gradients(&v, -epsilon)?;
        let grad_plus = plus.gradients(support)?;

        let mut minus = self.learner.clone();
        minus.apply_gradients(&v, epsilon)?;
        let grad_minus = minus.gradients(support)?;

        let alpha = self.config.inner_lr as f32;
        let mut meta = ModelGradients::new();
        for (name, value) in &v.gradients {
            let base = value.to_vec_f32()?;
            let shape = value.shape();
            let (Some(up), Some(down)) = (
                grad_plus.gradients.get(name),
                grad_minus.gradients.get(name),
            ) else {
                meta.gradients.insert(name.clone(), value.clone());
                continue;
            };
            let up = up.to_vec_f32()?;
            let down = down.to_vec_f32()?;
            let mut combined = Vec::with_capacity(base.len());
            for index in 0..base.len() {
                let hvp = (up[index] - down[index]) / (2.0 * epsilon as f32);
                combined.push(base[index] - alpha * hvp);
            }
            meta.gradients.insert(name.clone(), Tensor::from_vec(combined, &shape)?);
        }

        Ok(meta)
    }

    fn compute_relation(&self, a: &Tensor, b: &Tensor) -> Result<f64, TrustformersError> {
        // Relation score in (0, 1): a squashed negative squared distance
        // between two learned embeddings.
        let left = a.to_vec_f32()?;
        let right = b.to_vec_f32()?;
        if left.len() != right.len() {
            return Err(invalid_input(
                "relation scores need two embeddings of the same width",
            ));
        }
        let distance: f32 =
            left.iter().zip(right.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
        Ok((-(distance as f64)).exp())
    }

    fn get_learning_rates(&self) -> Result<Vec<f64>, TrustformersError> {
        Ok(self.learning_rates.clone())
    }

    fn apply_gradients_with_lr(
        &mut self,
        gradients: &ModelGradients,
        learning_rates: &[f64],
    ) -> Result<(), TrustformersError> {
        self.learner.apply_gradients_with_lr(gradients, learning_rates)
    }

    fn compute_lr_gradients(&self, _loss: f64) -> Result<Vec<f64>, TrustformersError> {
        // Meta-SGD: dL_query/d(alpha_i) = -g_query_i * g_support_i, the exact
        // first-order derivative of one inner SGD step with respect to its
        // per-parameter learning rate.
        let query = self.cached_batch()?;
        let Some(support) = self.previous_batch.as_ref() else {
            return Err(invalid_input(
                "learning-rate gradients need both the support and the query forward pass",
            ));
        };

        let query_gradients = self.learner.gradients(query)?;
        let support_gradients = self.learner.gradients(support)?;

        let mut output = Vec::with_capacity(self.learning_rates.len());
        for index in 0..(self.learner.layers.len()) {
            for suffix in ["weight", "bias"] {
                let name = format!("layer{}.{}", index, suffix);
                let (Some(q), Some(s)) = (
                    query_gradients.gradients.get(&name),
                    support_gradients.gradients.get(&name),
                ) else {
                    continue;
                };
                let q = q.to_vec_f32()?;
                let s = s.to_vec_f32()?;
                for (a, b) in q.iter().zip(s.iter()) {
                    output.push(-(*a as f64) * (*b as f64));
                }
            }
        }

        Ok(output)
    }
}

/// Declare a concrete model that delegates every trait method to [`MetaMlp`].
macro_rules! meta_model {
    ($(#[$meta:meta])* $name:ident $(, $extra:item)* $(,)?) => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        pub struct $name {
            inner: MetaMlp,
        }

        impl $name {
            /// Build the model from a meta-learning configuration.
            pub fn new(config: &MetaLearningConfig) -> Result<Self, TrustformersError> {
                Ok(Self {
                    inner: MetaMlp::new(config)?,
                })
            }

            /// Immutable access to the underlying learner.
            pub fn learner(&self) -> &MlpLearner {
                &self.inner.learner
            }
        }

        impl MetaLearningModel for $name {
            fn forward(&mut self, examples: &ExampleSet) -> Result<f64, TrustformersError> {
                self.inner.forward(examples)
            }

            fn compute_accuracy(&self, examples: &ExampleSet) -> Result<f64, TrustformersError> {
                self.inner.compute_accuracy(examples)
            }

            fn compute_gradients(&self, loss: f64) -> Result<ModelGradients, TrustformersError> {
                self.inner.compute_gradients(loss)
            }

            fn apply_gradients(
                &mut self,
                gradients: &ModelGradients,
                lr: f64,
            ) -> Result<(), TrustformersError> {
                self.inner.apply_gradients(gradients, lr)
            }

            fn get_parameters(&self) -> Result<ModelParameters, TrustformersError> {
                self.inner.get_parameters()
            }

            fn set_parameters(
                &mut self,
                params: ModelParameters,
            ) -> Result<(), TrustformersError> {
                self.inner.set_parameters(params)
            }

            fn embed(&self, example: &Example) -> Result<Tensor, TrustformersError> {
                self.inner.embed(example)
            }

            $($extra)*
        }
    };
}

meta_model!(
    /// Model-Agnostic Meta-Learning with real inner and outer loops.
    ///
    /// `compute_first_order_gradients` implements FOMAML;
    /// `compute_second_order_gradients` implements the one-step MAML
    /// meta-gradient `v - α·Hv`, with the Hessian-vector product estimated by
    /// a central difference of the support-set gradient.
    MAMLModel,
    fn compute_first_order_gradients(&self, loss: f64) -> Result<ModelGradients, TrustformersError> {
        self.inner.compute_first_order_gradients(loss)
    },
    fn compute_second_order_gradients(
        &self,
        initial_params: &ModelParameters,
        loss: f64,
    ) -> Result<ModelGradients, TrustformersError> {
        self.inner.compute_second_order_gradients(initial_params, loss)
    },
);

meta_model!(
    /// Reptile: the outer update is the difference between the adapted and the
    /// initial parameters, which the meta-learner computes directly.
    ReptileModel,
);

meta_model!(
    /// Prototypical Networks: the learner supplies real embeddings, the
    /// meta-learner forms class prototypes and classifies by distance.
    PrototypicalModel,
);

meta_model!(
    /// Matching Networks: cosine attention over the support embeddings.
    MatchingNetModel,
);

meta_model!(
    /// Relation Networks: a learned embedding plus a relation score.
    RelationNetModel,
    fn compute_relation(&self, a: &Tensor, b: &Tensor) -> Result<f64, TrustformersError> {
        self.inner.compute_relation(a, b)
    },
);

meta_model!(
    /// Memory-Augmented Neural Network.
    ///
    /// The learner is real, but the external memory (`write_to_memory`,
    /// `read_from_memory`, `predict_from_memory`) is not implemented and the
    /// default trait methods report that instead of returning invented
    /// predictions.
    MemoryAugmentedModel,
);

meta_model!(
    /// Gradient-based meta-learning with a learned update rule.
    ///
    /// The learner is real; the learned optimizer (`get_meta_learner_state`,
    /// `apply_learned_algorithm`) is not implemented and reports that.
    GradientBasedModel,
);

meta_model!(
    /// Meta-SGD: MAML with per-parameter learned learning rates.
    MetaSGDModel,
    fn compute_second_order_gradients(
        &self,
        initial_params: &ModelParameters,
        loss: f64,
    ) -> Result<ModelGradients, TrustformersError> {
        self.inner.compute_second_order_gradients(initial_params, loss)
    },
    fn get_learning_rates(&self) -> Result<Vec<f64>, TrustformersError> {
        self.inner.get_learning_rates()
    },
    fn apply_gradients_with_lr(
        &mut self,
        gradients: &ModelGradients,
        learning_rates: &[f64],
    ) -> Result<(), TrustformersError> {
        self.inner.apply_gradients_with_lr(gradients, learning_rates)
    },
    fn compute_lr_gradients(&self, loss: f64) -> Result<Vec<f64>, TrustformersError> {
        self.inner.compute_lr_gradients(loss)
    },
);

meta_model!(
    /// Learning to Learn by Gradient Descent by Gradient Descent.
    ///
    /// The learner is real; the LSTM meta-optimizer (`get_lstm_state`,
    /// `lstm_update`) is not implemented and reports that.
    L2LModel,
);
