/*!
# Meta-Learning Module

This module provides comprehensive meta-learning capabilities for transformer models,
enabling rapid adaptation to new tasks with minimal training data.

## Features

- **Model-Agnostic Meta-Learning (MAML)**: Learn initialization for rapid adaptation
- **Reptile**: First-order approximation to MAML
- **Prototypical Networks**: Learning metric embeddings for few-shot classification
- **Matching Networks**: End-to-end differentiable nearest neighbor
- **Relation Networks**: Learning to compare representations
- **Memory-Augmented Networks**: External memory for meta-learning
- **Gradient-Based Meta-Learning**: Various gradient-based approaches

## Usage

```rust
use trustformers_models::meta_learning::{
    MetaLearner, MetaLearningConfig, MetaAlgorithm, TaskBatch
};

let config = MetaLearningConfig {
    algorithm: MetaAlgorithm::MAML,
    inner_lr: 0.01,
    meta_lr: 0.001,
    inner_steps: 5,
    ..Default::default()
};

let mut meta_learner = MetaLearner::new(config)?;
```
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{
    errors::{invalid_input, unsupported_operation, TrustformersError},
    tensor::Tensor,
};

/// Configuration for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig {
    /// Meta-learning algorithm to use
    pub algorithm: MetaAlgorithm,
    /// Inner loop learning rate (task-specific adaptation)
    pub inner_lr: f64,
    /// Meta learning rate (across-task learning)
    pub meta_lr: f64,
    /// Number of inner loop gradient steps
    pub inner_steps: usize,
    /// Number of support examples per task
    pub support_size: usize,
    /// Number of query examples per task
    pub query_size: usize,
    /// Number of ways (classes) per task
    pub num_ways: usize,
    /// Number of shots (examples per class) per task
    pub num_shots: usize,
    /// Whether to use first-order approximation
    pub first_order: bool,
    /// Temperature for softmax in prototypical networks
    pub temperature: f64,
    /// Dimension of learned embeddings
    pub embedding_dim: usize,
    /// Whether to normalize embeddings
    pub normalize_embeddings: bool,
    /// Memory size for memory-augmented networks
    pub memory_size: usize,
    /// Memory key dimension
    pub memory_key_dim: usize,
    /// Memory value dimension
    pub memory_value_dim: usize,
    /// Number of meta-training tasks per batch
    pub meta_batch_size: usize,
    /// Whether to use task-specific parameters
    pub task_specific_params: bool,
    /// L2 regularization for inner loop
    pub inner_l2_reg: f64,
    /// Gradient clipping threshold
    pub grad_clip_norm: f64,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            algorithm: MetaAlgorithm::MAML,
            inner_lr: 0.01,
            meta_lr: 0.001,
            inner_steps: 5,
            support_size: 5,
            query_size: 15,
            num_ways: 5,
            num_shots: 1,
            first_order: false,
            temperature: 1.0,
            embedding_dim: 512,
            normalize_embeddings: true,
            memory_size: 128,
            memory_key_dim: 64,
            memory_value_dim: 256,
            meta_batch_size: 32,
            task_specific_params: false,
            inner_l2_reg: 0.0001,
            grad_clip_norm: 10.0,
        }
    }
}

/// Meta-learning algorithms supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetaAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML,
    /// Reptile (first-order MAML)
    Reptile,
    /// Prototypical Networks
    ProtoNet,
    /// Matching Networks
    MatchingNet,
    /// Relation Networks
    RelationNet,
    /// Memory-Augmented Neural Networks
    MANN,
    /// Gradient-Based Meta-Learning
    GBML,
    /// Meta-SGD (learn learning rates)
    MetaSGD,
    /// Learning to Learn by Gradient Descent
    L2L,
}

/// Meta-learning trainer
pub struct MetaLearner {
    config: MetaLearningConfig,
    model: Box<dyn MetaLearningModel>,
    optimizer: Box<dyn MetaOptimizer>,
    task_sampler: TaskSampler,
    meta_statistics: MetaStatistics,
    episode_history: Vec<EpisodeResult>,
    current_episode: usize,
}

impl MetaLearner {
    /// Create a new meta-learner
    pub fn new(config: MetaLearningConfig) -> Result<Self, TrustformersError> {
        let model = Self::create_model(&config)?;
        let optimizer = Self::create_optimizer(&config)?;
        let task_sampler = TaskSampler::new(&config)?;

        Ok(Self {
            config,
            model,
            optimizer,
            task_sampler,
            meta_statistics: MetaStatistics::new(),
            episode_history: Vec::new(),
            current_episode: 0,
        })
    }

    /// Create appropriate model based on algorithm
    fn create_model(
        config: &MetaLearningConfig,
    ) -> Result<Box<dyn MetaLearningModel>, TrustformersError> {
        match config.algorithm {
            MetaAlgorithm::MAML => Ok(Box::new(MAMLModel::new(config)?)),
            MetaAlgorithm::Reptile => Ok(Box::new(ReptileModel::new(config)?)),
            MetaAlgorithm::ProtoNet => Ok(Box::new(PrototypicalModel::new(config)?)),
            MetaAlgorithm::MatchingNet => Ok(Box::new(MatchingNetModel::new(config)?)),
            MetaAlgorithm::RelationNet => Ok(Box::new(RelationNetModel::new(config)?)),
            MetaAlgorithm::MANN => Ok(Box::new(MemoryAugmentedModel::new(config)?)),
            MetaAlgorithm::GBML => Ok(Box::new(GradientBasedModel::new(config)?)),
            MetaAlgorithm::MetaSGD => Ok(Box::new(MetaSGDModel::new(config)?)),
            MetaAlgorithm::L2L => Ok(Box::new(L2LModel::new(config)?)),
        }
    }

    /// Create appropriate optimizer
    fn create_optimizer(
        config: &MetaLearningConfig,
    ) -> Result<Box<dyn MetaOptimizer>, TrustformersError> {
        match config.algorithm {
            MetaAlgorithm::MAML | MetaAlgorithm::Reptile | MetaAlgorithm::GBML => {
                Ok(Box::new(SGDMetaOptimizer::new(config.meta_lr)?))
            },
            MetaAlgorithm::MetaSGD => Ok(Box::new(LearnedLROptimizer::new(config.meta_lr)?)),
            _ => Ok(Box::new(AdamMetaOptimizer::new(config.meta_lr)?)),
        }
    }

    /// Train on a single meta-learning episode
    pub fn train_episode(
        &mut self,
        task_batch: TaskBatch,
    ) -> Result<EpisodeResult, TrustformersError> {
        let start_time = std::time::Instant::now();
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let num_tasks = task_batch.tasks.len();

        for task in &task_batch.tasks {
            let task_result = self.train_single_task(task)?;
            total_loss += task_result.query_loss;
            total_accuracy += task_result.query_accuracy;
        }

        // Update meta-parameters
        self.optimizer.step(&mut *self.model)?;

        let episode_result = EpisodeResult {
            episode: self.current_episode,
            meta_loss: total_loss / num_tasks as f64,
            meta_accuracy: total_accuracy / num_tasks as f64,
            num_tasks,
            episode_time: start_time.elapsed(),
            algorithm: self.config.algorithm,
        };

        self.episode_history.push(episode_result.clone());
        self.meta_statistics.update(&episode_result);
        self.current_episode += 1;

        Ok(episode_result)
    }

    /// Train on a single task within an episode
    fn train_single_task(&mut self, task: &Task) -> Result<TaskResult, TrustformersError> {
        match self.config.algorithm {
            MetaAlgorithm::MAML => self.train_maml_task(task),
            MetaAlgorithm::Reptile => self.train_reptile_task(task),
            MetaAlgorithm::ProtoNet => self.train_prototypical_task(task),
            MetaAlgorithm::MatchingNet => self.train_matching_task(task),
            MetaAlgorithm::RelationNet => self.train_relation_task(task),
            MetaAlgorithm::MANN => self.train_memory_task(task),
            MetaAlgorithm::GBML => self.train_gradient_based_task(task),
            MetaAlgorithm::MetaSGD => self.train_meta_sgd_task(task),
            MetaAlgorithm::L2L => self.train_l2l_task(task),
        }
    }

    /// MAML training for a single task
    fn train_maml_task(&mut self, task: &Task) -> Result<TaskResult, TrustformersError> {
        // Save initial parameters
        let initial_params = self.model.get_parameters()?;

        // Inner loop: adapt to support set
        for _ in 0..self.config.inner_steps {
            let support_loss = self.model.forward(&task.support_set)?;
            let gradients = self.model.compute_gradients(support_loss)?;

            // Apply inner loop update
            self.model.apply_gradients(&gradients, self.config.inner_lr)?;
        }

        // Compute loss on query set with adapted parameters
        let query_loss = self.model.forward(&task.query_set)?;
        let query_accuracy = self.model.compute_accuracy(&task.query_set)?;

        // Compute meta-gradients (through the inner loop)
        let meta_gradients = if self.config.first_order {
            // First-order approximation (Reptile-style)
            self.model.compute_first_order_gradients(query_loss)?
        } else {
            // Full second-order gradients
            self.model.compute_second_order_gradients(&initial_params, query_loss)?
        };

        // Store meta-gradients for meta-update
        self.optimizer.accumulate_gradients(meta_gradients)?;

        // Restore initial parameters for next task
        self.model.set_parameters(initial_params)?;

        Ok(TaskResult {
            support_loss: 0.0, // We don't track support loss in MAML
            query_loss,
            query_accuracy,
            adaptation_time: std::time::Duration::from_millis(0),
        })
    }

    /// Reptile training for a single task
    fn train_reptile_task(&mut self, task: &Task) -> Result<TaskResult, TrustformersError> {
        let initial_params = self.model.get_parameters()?;

        // Inner loop on support set
        for _ in 0..self.config.inner_steps {
            let support_loss = self.model.forward(&task.support_set)?;
            let gradients = self.model.compute_gradients(support_loss)?;
            self.model.apply_gradients(&gradients, self.config.inner_lr)?;
        }

        let adapted_params = self.model.get_parameters()?;
        let query_loss = self.model.forward(&task.query_set)?;
        let query_accuracy = self.model.compute_accuracy(&task.query_set)?;

        // Reptile meta-gradient: direction from initial to adapted parameters
        let meta_gradients = self.compute_param_difference(&initial_params, &adapted_params)?;
        self.optimizer.accumulate_gradients(meta_gradients)?;

        // Restore parameters
        self.model.set_parameters(initial_params)?;

        Ok(TaskResult {
            support_loss: 0.0,
            query_loss,
            query_accuracy,
            adaptation_time: std::time::Duration::from_millis(0),
        })
    }

    /// Prototypical Networks training
    fn train_prototypical_task(&mut self, task: &Task) -> Result<TaskResult, TrustformersError> {
        // Compute prototypes from support set
        let prototypes = self.compute_prototypes(&task.support_set)?;

        // Classify query examples based on distance to prototypes
        let query_loss = self.compute_prototypical_loss(&task.query_set, &prototypes)?;
        let query_accuracy = self.compute_prototypical_accuracy(&task.query_set, &prototypes)?;

        // Standard gradient computation
        let gradients = self.model.compute_gradients(query_loss)?;
        self.optimizer.accumulate_gradients(gradients)?;

        Ok(TaskResult {
            support_loss: 0.0,
            query_loss,
            query_accuracy,
            adaptation_time: std::time::Duration::from_millis(0),
        })
    }

    /// Matching Networks training
    fn train_matching_task(&mut self, task: &Task) -> Result<TaskResult, TrustformersError> {
        // Compute attention weights between query and support examples
        let attention_weights =
            self.compute_attention_weights(&task.query_set, &task.support_set)?;

        // Weighted combination of support labels
        let predictions =
            self.compute_matching_predictions(&attention_weights, &task.support_set)?;

        let query_loss = self.compute_matching_loss(&predictions, &task.query_set)?;
        let query_accuracy = self.compute_matching_accuracy(&predictions, &task.query_set)?;

        let gradients = self.model.compute_gradients(query_loss)?;
        self.optimizer.accumulate_gradients(gradients)?;

        Ok(TaskResult {
            support_loss: 0.0,
            query_loss,
            query_accuracy,
            adaptation_time: std::time::Duration::from_millis(0),
        })
    }

    /// Relation Networks training
    fn train_relation_task(&mut self, task: &Task) -> Result<TaskResult, TrustformersError> {
        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        // For each query example, compute relation scores with all support examples
        for query_example in &task.query_set.examples {
            let query_embedding = self.model.embed(query_example)?;
            let mut relation_scores = Vec::new();

            for support_example in &task.support_set.examples {
                let support_embedding = self.model.embed(support_example)?;
                let relation_score =
                    self.model.compute_relation(&query_embedding, &support_embedding)?;
                relation_scores.push(relation_score);
            }

            // Compute loss and accuracy for this query example
            let loss =
                self.compute_relation_loss(&relation_scores, query_example, &task.support_set)?;
            total_loss += loss;

            if self.is_correct_prediction(&relation_scores, query_example, &task.support_set)? {
                correct_predictions += 1;
            }
            total_predictions += 1;
        }

        let query_loss = total_loss / total_predictions as f64;
        let query_accuracy = correct_predictions as f64 / total_predictions as f64;

        let gradients = self.model.compute_gradients(query_loss)?;
        self.optimizer.accumulate_gradients(gradients)?;

        Ok(TaskResult {
            support_loss: 0.0,
            query_loss,
            query_accuracy,
            adaptation_time: std::time::Duration::from_millis(0),
        })
    }

    /// Memory-Augmented Networks training
    fn train_memory_task(&mut self, task: &Task) -> Result<TaskResult, TrustformersError> {
        // Write support examples to memory
        for example in &task.support_set.examples {
            self.model.write_to_memory(example)?;
        }

        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let total_predictions = task.query_set.examples.len();

        // For each query example, read from memory and predict
        for query_example in &task.query_set.examples {
            let memory_output = self.model.read_from_memory(query_example)?;
            let prediction = self.model.predict_from_memory(&memory_output)?;

            let loss = self.compute_memory_loss(&prediction, query_example)?;
            total_loss += loss;

            if self.is_memory_prediction_correct(&prediction, query_example)? {
                correct_predictions += 1;
            }
        }

        let query_loss = total_loss / total_predictions as f64;
        let query_accuracy = correct_predictions as f64 / total_predictions as f64;

        let gradients = self.model.compute_gradients(query_loss)?;
        self.optimizer.accumulate_gradients(gradients)?;

        // Clear memory for next task
        self.model.clear_memory()?;

        Ok(TaskResult {
            support_loss: 0.0,
            query_loss,
            query_accuracy,
            adaptation_time: std::time::Duration::from_millis(0),
        })
    }

    /// Gradient-Based Meta-Learning training
    fn train_gradient_based_task(&mut self, task: &Task) -> Result<TaskResult, TrustformersError> {
        // Learn the learning algorithm itself
        let meta_learner_state = self.model.get_meta_learner_state()?;

        // Apply learned learning algorithm to support set
        let adapted_params =
            self.model.apply_learned_algorithm(&task.support_set, &meta_learner_state)?;

        // Evaluate on query set
        let query_loss = self.model.evaluate_with_params(&task.query_set, &adapted_params)?;
        let query_accuracy =
            self.model.compute_accuracy_with_params(&task.query_set, &adapted_params)?;

        // Update meta-learner
        let gradients = self.model.compute_meta_learner_gradients(query_loss)?;
        self.optimizer.accumulate_gradients(gradients)?;

        Ok(TaskResult {
            support_loss: 0.0,
            query_loss,
            query_accuracy,
            adaptation_time: std::time::Duration::from_millis(0),
        })
    }

    /// Meta-SGD training (learns learning rates)
    fn train_meta_sgd_task(&mut self, task: &Task) -> Result<TaskResult, TrustformersError> {
        let initial_params = self.model.get_parameters()?;
        let learning_rates = self.model.get_learning_rates()?;

        // Inner loop with learned learning rates
        for _ in 0..self.config.inner_steps {
            let support_loss = self.model.forward(&task.support_set)?;
            let gradients = self.model.compute_gradients(support_loss)?;

            // Apply gradients with learned learning rates
            self.model.apply_gradients_with_lr(&gradients, &learning_rates)?;
        }

        let query_loss = self.model.forward(&task.query_set)?;
        let query_accuracy = self.model.compute_accuracy(&task.query_set)?;

        // Update both parameters and learning rates
        let param_gradients =
            self.model.compute_second_order_gradients(&initial_params, query_loss)?;
        let lr_gradients = self.model.compute_lr_gradients(query_loss)?;

        self.optimizer.accumulate_param_gradients(param_gradients)?;
        self.optimizer.accumulate_lr_gradients(lr_gradients)?;

        // Restore parameters
        self.model.set_parameters(initial_params)?;

        Ok(TaskResult {
            support_loss: 0.0,
            query_loss,
            query_accuracy,
            adaptation_time: std::time::Duration::from_millis(0),
        })
    }

    /// Learning to Learn by Gradient Descent training
    fn train_l2l_task(&mut self, task: &Task) -> Result<TaskResult, TrustformersError> {
        // Use LSTM meta-learner to generate updates
        let mut lstm_state = self.model.get_lstm_state()?;
        let initial_params = self.model.get_parameters()?;

        // Inner loop using LSTM meta-learner
        for step in 0..self.config.inner_steps {
            let support_loss = self.model.forward(&task.support_set)?;
            let gradients = self.model.compute_gradients(support_loss)?;

            // LSTM generates parameter updates
            let (updates, new_lstm_state) =
                self.model.lstm_update(&gradients, &lstm_state, step)?;
            lstm_state = new_lstm_state;

            // Apply LSTM-generated updates
            self.model.apply_lstm_updates(&updates)?;
        }

        let query_loss = self.model.forward(&task.query_set)?;
        let query_accuracy = self.model.compute_accuracy(&task.query_set)?;

        // Update LSTM meta-learner
        let lstm_gradients = self.model.compute_lstm_gradients(query_loss)?;
        self.optimizer.accumulate_gradients(lstm_gradients)?;

        // Restore parameters
        self.model.set_parameters(initial_params)?;

        Ok(TaskResult {
            support_loss: 0.0,
            query_loss,
            query_accuracy,
            adaptation_time: std::time::Duration::from_millis(0),
        })
    }

    /// Helper methods for specific computations
    fn compute_param_difference(
        &self,
        params1: &ModelParameters,
        params2: &ModelParameters,
    ) -> Result<ModelGradients, TrustformersError> {
        // Compute difference between parameter sets
        let mut gradients = ModelGradients::new();

        for (name, param1) in &params1.parameters {
            if let Some(param2) = params2.parameters.get(name) {
                let diff = param2.sub(param1)?; // Reptile direction
                gradients.gradients.insert(name.clone(), diff);
            }
        }

        Ok(gradients)
    }

    fn compute_prototypes(
        &self,
        support_set: &ExampleSet,
    ) -> Result<Vec<Tensor>, TrustformersError> {
        let mut prototypes = Vec::new();
        let num_classes = self.config.num_ways;

        for class_id in 0..num_classes {
            let mut class_embeddings = Vec::new();

            // Collect embeddings for this class
            for example in &support_set.examples {
                if example.label == class_id {
                    let embedding = self.model.embed(example)?;
                    class_embeddings.push(embedding);
                }
            }

            // Compute prototype as mean of class embeddings
            if !class_embeddings.is_empty() {
                let prototype = self.compute_mean_embedding(&class_embeddings)?;
                prototypes.push(prototype);
            }
        }

        Ok(prototypes)
    }

    fn compute_mean_embedding(&self, embeddings: &[Tensor]) -> Result<Tensor, TrustformersError> {
        if embeddings.is_empty() {
            return Err(invalid_input("Empty embeddings list"));
        }

        let mut sum = embeddings[0].clone();
        for embedding in &embeddings[1..] {
            sum = sum.add(embedding)?;
        }

        sum.scalar_div(embeddings.len() as f32)
    }

    fn compute_prototypical_loss(
        &self,
        query_set: &ExampleSet,
        prototypes: &[Tensor],
    ) -> Result<f64, TrustformersError> {
        let mut total_loss = 0.0;

        for example in &query_set.examples {
            let query_embedding = self.model.embed(example)?;
            let distances = self.compute_distances(&query_embedding, prototypes)?;
            let log_probs = self.compute_log_softmax(&distances, self.config.temperature)?;

            // Negative log-likelihood loss
            total_loss -= log_probs[example.label];
        }

        Ok(total_loss / query_set.examples.len() as f64)
    }

    fn compute_prototypical_accuracy(
        &self,
        query_set: &ExampleSet,
        prototypes: &[Tensor],
    ) -> Result<f64, TrustformersError> {
        let mut correct = 0;

        for example in &query_set.examples {
            let query_embedding = self.model.embed(example)?;
            let distances = self.compute_distances(&query_embedding, prototypes)?;

            // Predict class with minimum distance
            let predicted_class = distances
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("operation failed"))
                .map(|(i, _)| i)
                .unwrap_or(0);

            if predicted_class == example.label {
                correct += 1;
            }
        }

        Ok(correct as f64 / query_set.examples.len() as f64)
    }

    fn compute_distances(
        &self,
        query: &Tensor,
        prototypes: &[Tensor],
    ) -> Result<Vec<f64>, TrustformersError> {
        let mut distances = Vec::new();

        for prototype in prototypes {
            let diff = query.sub(prototype)?;
            let distance = diff.norm()? as f64;
            distances.push(distance);
        }

        Ok(distances)
    }

    fn compute_log_softmax(
        &self,
        distances: &[f64],
        temperature: f64,
    ) -> Result<Vec<f64>, TrustformersError> {
        // Convert distances to negative log probabilities
        let neg_distances: Vec<f64> = distances.iter().map(|d| -d / temperature).collect();

        // Compute log softmax
        let max_val = neg_distances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f64 = neg_distances.iter().map(|x| (x - max_val).exp()).sum();
        let log_sum = max_val + exp_sum.ln();

        Ok(neg_distances.iter().map(|x| x - log_sum).collect())
    }

    // Additional helper methods would be implemented here...
    fn compute_attention_weights(
        &self,
        _query_set: &ExampleSet,
        _support_set: &ExampleSet,
    ) -> Result<Vec<Vec<f64>>, TrustformersError> {
        // Placeholder implementation
        Ok(vec![vec![1.0]])
    }

    fn compute_matching_predictions(
        &self,
        _weights: &[Vec<f64>],
        _support_set: &ExampleSet,
    ) -> Result<Vec<Vec<f64>>, TrustformersError> {
        Ok(vec![vec![1.0]])
    }

    fn compute_matching_loss(
        &self,
        _predictions: &[Vec<f64>],
        _query_set: &ExampleSet,
    ) -> Result<f64, TrustformersError> {
        Ok(1.0)
    }

    fn compute_matching_accuracy(
        &self,
        _predictions: &[Vec<f64>],
        _query_set: &ExampleSet,
    ) -> Result<f64, TrustformersError> {
        Ok(0.8)
    }

    fn compute_relation_loss(
        &self,
        _scores: &[f64],
        _example: &Example,
        _support_set: &ExampleSet,
    ) -> Result<f64, TrustformersError> {
        Ok(1.0)
    }

    fn is_correct_prediction(
        &self,
        _scores: &[f64],
        _example: &Example,
        _support_set: &ExampleSet,
    ) -> Result<bool, TrustformersError> {
        Ok(true)
    }

    fn compute_memory_loss(
        &self,
        _prediction: &MemoryPrediction,
        _example: &Example,
    ) -> Result<f64, TrustformersError> {
        Ok(1.0)
    }

    fn is_memory_prediction_correct(
        &self,
        _prediction: &MemoryPrediction,
        _example: &Example,
    ) -> Result<bool, TrustformersError> {
        Ok(true)
    }

    /// Evaluate the meta-learner on new tasks
    pub fn evaluate(
        &mut self,
        task_batch: TaskBatch,
    ) -> Result<EvaluationResult, TrustformersError> {
        let mut total_accuracy = 0.0;
        let mut task_results = Vec::new();

        for task in &task_batch.tasks {
            let task_result = self.evaluate_single_task(task)?;
            total_accuracy += task_result.query_accuracy;
            task_results.push(task_result);
        }

        Ok(EvaluationResult {
            average_accuracy: total_accuracy / task_batch.tasks.len() as f64,
            task_results,
            num_tasks: task_batch.tasks.len(),
        })
    }

    fn evaluate_single_task(&mut self, task: &Task) -> Result<TaskResult, TrustformersError> {
        // Similar to training but without gradient updates
        match self.config.algorithm {
            MetaAlgorithm::MAML | MetaAlgorithm::Reptile => {
                let initial_params = self.model.get_parameters()?;

                // Adapt to support set
                for _ in 0..self.config.inner_steps {
                    let support_loss = self.model.forward(&task.support_set)?;
                    let gradients = self.model.compute_gradients(support_loss)?;
                    self.model.apply_gradients(&gradients, self.config.inner_lr)?;
                }

                // Evaluate on query set
                let query_loss = self.model.forward(&task.query_set)?;
                let query_accuracy = self.model.compute_accuracy(&task.query_set)?;

                // Restore parameters
                self.model.set_parameters(initial_params)?;

                Ok(TaskResult {
                    support_loss: 0.0,
                    query_loss,
                    query_accuracy,
                    adaptation_time: std::time::Duration::from_millis(0),
                })
            },
            MetaAlgorithm::ProtoNet => self.train_prototypical_task(task),
            _ => {
                // For other algorithms, use the same evaluation as training
                // but without accumulating gradients
                self.train_single_task(task)
            },
        }
    }

    /// Get meta-learning statistics
    pub fn get_statistics(&self) -> &MetaStatistics {
        &self.meta_statistics
    }

    /// Get episode history
    pub fn get_episode_history(&self) -> &[EpisodeResult] {
        &self.episode_history
    }

    /// Sample a new task batch
    pub fn sample_task_batch(&mut self) -> Result<TaskBatch, TrustformersError> {
        self.task_sampler.sample_batch(self.config.meta_batch_size)
    }
}

/// Core data structures for meta-learning
#[derive(Debug, Clone)]
pub struct Task {
    pub task_id: String,
    pub support_set: ExampleSet,
    pub query_set: ExampleSet,
    pub task_type: TaskType,
}

#[derive(Debug, Clone)]
pub struct TaskBatch {
    pub tasks: Vec<Task>,
    pub batch_id: String,
}

#[derive(Debug, Clone)]
pub struct ExampleSet {
    pub examples: Vec<Example>,
    pub num_classes: usize,
}

#[derive(Debug, Clone)]
pub struct Example {
    pub input: Tensor,
    pub label: usize,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Classification,
    Regression,
    Generation,
    SequenceLabeling,
}

/// Results and statistics
#[derive(Debug, Clone)]
pub struct EpisodeResult {
    pub episode: usize,
    pub meta_loss: f64,
    pub meta_accuracy: f64,
    pub num_tasks: usize,
    pub episode_time: std::time::Duration,
    pub algorithm: MetaAlgorithm,
}

#[derive(Debug, Clone)]
pub struct TaskResult {
    pub support_loss: f64,
    pub query_loss: f64,
    pub query_accuracy: f64,
    pub adaptation_time: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub average_accuracy: f64,
    pub task_results: Vec<TaskResult>,
    pub num_tasks: usize,
}

#[derive(Debug)]
pub struct MetaStatistics {
    pub total_episodes: usize,
    pub average_accuracy: f64,
    pub best_accuracy: f64,
    pub recent_accuracies: std::collections::VecDeque<f64>,
    pub convergence_rate: f64,
}

impl Default for MetaStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaStatistics {
    pub fn new() -> Self {
        Self {
            total_episodes: 0,
            average_accuracy: 0.0,
            best_accuracy: 0.0,
            recent_accuracies: std::collections::VecDeque::with_capacity(100),
            convergence_rate: 0.0,
        }
    }

    pub fn update(&mut self, episode_result: &EpisodeResult) {
        self.total_episodes += 1;

        // Update running average
        let alpha = 0.01; // Exponential moving average factor
        self.average_accuracy =
            alpha * episode_result.meta_accuracy + (1.0 - alpha) * self.average_accuracy;

        // Update best accuracy
        if episode_result.meta_accuracy > self.best_accuracy {
            self.best_accuracy = episode_result.meta_accuracy;
        }

        // Track recent accuracies for convergence analysis
        self.recent_accuracies.push_back(episode_result.meta_accuracy);
        if self.recent_accuracies.len() > 100 {
            self.recent_accuracies.pop_front();
        }

        // Estimate convergence rate
        if self.recent_accuracies.len() > 10 {
            let recent_mean =
                self.recent_accuracies.iter().sum::<f64>() / self.recent_accuracies.len() as f64;
            let older_mean = self.recent_accuracies.iter().take(50).sum::<f64>()
                / (50.0f64).min(self.recent_accuracies.len() as f64);
            self.convergence_rate = (recent_mean - older_mean).abs();
        }
    }
}

/// Trait definitions for model components
pub trait MetaLearningModel: Send + Sync {
    fn forward(&mut self, examples: &ExampleSet) -> Result<f64, TrustformersError>;
    fn compute_accuracy(&self, examples: &ExampleSet) -> Result<f64, TrustformersError>;
    fn compute_gradients(&self, loss: f64) -> Result<ModelGradients, TrustformersError>;
    fn apply_gradients(
        &mut self,
        gradients: &ModelGradients,
        lr: f64,
    ) -> Result<(), TrustformersError>;
    fn get_parameters(&self) -> Result<ModelParameters, TrustformersError>;
    fn set_parameters(&mut self, params: ModelParameters) -> Result<(), TrustformersError>;
    fn embed(&self, example: &Example) -> Result<Tensor, TrustformersError>;

    // Additional methods for specific algorithms
    fn compute_second_order_gradients(
        &self,
        _initial_params: &ModelParameters,
        _loss: f64,
    ) -> Result<ModelGradients, TrustformersError> {
        Err(unsupported_operation(
            "compute_second_order_gradients",
            "meta_learning",
        ))
    }

    fn compute_first_order_gradients(
        &self,
        _loss: f64,
    ) -> Result<ModelGradients, TrustformersError> {
        Err(unsupported_operation(
            "compute_first_order_gradients",
            "meta_learning",
        ))
    }

    fn compute_relation(&self, _emb1: &Tensor, _emb2: &Tensor) -> Result<f64, TrustformersError> {
        Err(unsupported_operation("compute_relation", "meta_learning"))
    }

    fn write_to_memory(&mut self, _example: &Example) -> Result<(), TrustformersError> {
        Err(unsupported_operation("write_to_memory", "meta_learning"))
    }

    fn read_from_memory(&self, _example: &Example) -> Result<MemoryOutput, TrustformersError> {
        Err(unsupported_operation("read_from_memory", "meta_learning"))
    }

    fn predict_from_memory(
        &self,
        _memory_output: &MemoryOutput,
    ) -> Result<MemoryPrediction, TrustformersError> {
        Err(unsupported_operation(
            "predict_from_memory",
            "meta_learning",
        ))
    }

    fn clear_memory(&mut self) -> Result<(), TrustformersError> {
        Ok(())
    }

    fn get_learning_rates(&self) -> Result<Vec<f64>, TrustformersError> {
        Err(unsupported_operation("get_learning_rates", "meta_learning"))
    }

    fn apply_gradients_with_lr(
        &mut self,
        _gradients: &ModelGradients,
        _learning_rates: &[f64],
    ) -> Result<(), TrustformersError> {
        Err(unsupported_operation(
            "apply_gradients_with_lr",
            "meta_learning",
        ))
    }

    fn compute_lr_gradients(&self, _loss: f64) -> Result<Vec<f64>, TrustformersError> {
        Err(unsupported_operation(
            "compute_lr_gradients",
            "meta_learning",
        ))
    }

    fn get_meta_learner_state(&self) -> Result<MetaLearnerState, TrustformersError> {
        Err(unsupported_operation(
            "get_meta_learner_state",
            "meta_learning",
        ))
    }

    fn apply_learned_algorithm(
        &self,
        _support_set: &ExampleSet,
        _state: &MetaLearnerState,
    ) -> Result<ModelParameters, TrustformersError> {
        Err(unsupported_operation(
            "apply_learned_algorithm",
            "meta_learning",
        ))
    }

    fn evaluate_with_params(
        &self,
        _examples: &ExampleSet,
        _params: &ModelParameters,
    ) -> Result<f64, TrustformersError> {
        Err(unsupported_operation(
            "evaluate_with_params",
            "meta_learning",
        ))
    }

    fn compute_accuracy_with_params(
        &self,
        _examples: &ExampleSet,
        _params: &ModelParameters,
    ) -> Result<f64, TrustformersError> {
        Err(unsupported_operation(
            "compute_accuracy_with_params",
            "meta_learning",
        ))
    }

    fn compute_meta_learner_gradients(
        &self,
        _loss: f64,
    ) -> Result<ModelGradients, TrustformersError> {
        Err(unsupported_operation(
            "compute_meta_learner_gradients",
            "meta_learning",
        ))
    }

    fn get_lstm_state(&self) -> Result<LSTMState, TrustformersError> {
        Err(unsupported_operation("get_lstm_state", "meta_learning"))
    }

    fn lstm_update(
        &self,
        _gradients: &ModelGradients,
        _state: &LSTMState,
        _step: usize,
    ) -> Result<(ModelUpdates, LSTMState), TrustformersError> {
        Err(unsupported_operation("lstm_update", "meta_learning"))
    }

    fn apply_lstm_updates(&mut self, _updates: &ModelUpdates) -> Result<(), TrustformersError> {
        Err(unsupported_operation("apply_lstm_updates", "meta_learning"))
    }

    fn compute_lstm_gradients(&self, _loss: f64) -> Result<ModelGradients, TrustformersError> {
        Err(unsupported_operation(
            "compute_lstm_gradients",
            "meta_learning",
        ))
    }
}

pub trait MetaOptimizer: Send + Sync {
    fn step(&mut self, model: &mut dyn MetaLearningModel) -> Result<(), TrustformersError>;
    fn accumulate_gradients(&mut self, gradients: ModelGradients) -> Result<(), TrustformersError>;
    fn accumulate_param_gradients(
        &mut self,
        _gradients: ModelGradients,
    ) -> Result<(), TrustformersError> {
        self.accumulate_gradients(_gradients)
    }
    fn accumulate_lr_gradients(
        &mut self,
        _lr_gradients: Vec<f64>,
    ) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn reset(&mut self) -> Result<(), TrustformersError>;
}

/// Supporting data structures
#[derive(Debug, Clone)]
pub struct ModelParameters {
    pub parameters: HashMap<String, Tensor>,
}

#[derive(Debug, Clone)]
pub struct ModelGradients {
    pub gradients: HashMap<String, Tensor>,
}

impl Default for ModelGradients {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelGradients {
    pub fn new() -> Self {
        Self {
            gradients: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryOutput {
    pub content: Tensor,
    pub attention_weights: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MemoryPrediction {
    pub logits: Tensor,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct MetaLearnerState {
    pub hidden_state: Tensor,
    pub cell_state: Tensor,
}

#[derive(Debug, Clone)]
pub struct LSTMState {
    pub hidden: Tensor,
    pub cell: Tensor,
}

#[derive(Debug, Clone)]
pub struct ModelUpdates {
    pub updates: HashMap<String, Tensor>,
}

/// Task sampling for meta-learning
pub struct TaskSampler {
    config: MetaLearningConfig,
    #[allow(dead_code)]
    task_distributions: Vec<TaskDistribution>,
    current_task_id: usize,
}

impl TaskSampler {
    pub fn new(config: &MetaLearningConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.clone(),
            task_distributions: Vec::new(),
            current_task_id: 0,
        })
    }

    pub fn sample_batch(&mut self, batch_size: usize) -> Result<TaskBatch, TrustformersError> {
        let mut tasks = Vec::new();

        for _ in 0..batch_size {
            let task = self.sample_single_task()?;
            tasks.push(task);
        }

        Ok(TaskBatch {
            tasks,
            batch_id: format!("batch_{}", self.current_task_id),
        })
    }

    fn sample_single_task(&mut self) -> Result<Task, TrustformersError> {
        // For now, create a simple synthetic task
        let support_set = self.create_example_set(self.config.support_size)?;
        let query_set = self.create_example_set(self.config.query_size)?;

        self.current_task_id += 1;

        Ok(Task {
            task_id: format!("task_{}", self.current_task_id),
            support_set,
            query_set,
            task_type: TaskType::Classification,
        })
    }

    fn create_example_set(&self, size: usize) -> Result<ExampleSet, TrustformersError> {
        let mut examples = Vec::new();

        for i in 0..size {
            let input = Tensor::randn(&[self.config.embedding_dim])?;
            let label = i % self.config.num_ways; // Cycle through classes

            examples.push(Example {
                input,
                label,
                metadata: HashMap::new(),
            });
        }

        Ok(ExampleSet {
            examples,
            num_classes: self.config.num_ways,
        })
    }
}

#[derive(Debug)]
pub struct TaskDistribution {
    pub name: String,
    pub sampling_weight: f64,
}

/// Concrete implementations of meta-learning models would go here
/// For brevity, I'll include basic stubs
pub struct MAMLModel {
    #[allow(dead_code)]
    config: MetaLearningConfig,
}

impl MAMLModel {
    pub fn new(config: &MetaLearningConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

impl MetaLearningModel for MAMLModel {
    fn forward(&mut self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.5) // Placeholder
    }

    fn compute_accuracy(&self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.8) // Placeholder
    }

    fn compute_gradients(&self, _loss: f64) -> Result<ModelGradients, TrustformersError> {
        Ok(ModelGradients::new())
    }

    fn apply_gradients(
        &mut self,
        _gradients: &ModelGradients,
        _lr: f64,
    ) -> Result<(), TrustformersError> {
        Ok(())
    }

    fn get_parameters(&self) -> Result<ModelParameters, TrustformersError> {
        Ok(ModelParameters {
            parameters: HashMap::new(),
        })
    }

    fn set_parameters(&mut self, _params: ModelParameters) -> Result<(), TrustformersError> {
        Ok(())
    }

    fn embed(&self, example: &Example) -> Result<Tensor, TrustformersError> {
        Ok(example.input.clone())
    }

    fn compute_second_order_gradients(
        &self,
        _initial_params: &ModelParameters,
        _loss: f64,
    ) -> Result<ModelGradients, TrustformersError> {
        Ok(ModelGradients::new())
    }

    fn compute_first_order_gradients(
        &self,
        _loss: f64,
    ) -> Result<ModelGradients, TrustformersError> {
        Ok(ModelGradients::new())
    }
}

// Similar stub implementations for other models...
pub struct ReptileModel {
    #[allow(dead_code)]
    config: MetaLearningConfig,
}

impl ReptileModel {
    pub fn new(config: &MetaLearningConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

impl MetaLearningModel for ReptileModel {
    fn forward(&mut self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.5)
    }
    fn compute_accuracy(&self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.8)
    }
    fn compute_gradients(&self, _loss: f64) -> Result<ModelGradients, TrustformersError> {
        Ok(ModelGradients::new())
    }
    fn apply_gradients(
        &mut self,
        _gradients: &ModelGradients,
        _lr: f64,
    ) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn get_parameters(&self) -> Result<ModelParameters, TrustformersError> {
        Ok(ModelParameters {
            parameters: HashMap::new(),
        })
    }
    fn set_parameters(&mut self, _params: ModelParameters) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn embed(&self, example: &Example) -> Result<Tensor, TrustformersError> {
        Ok(example.input.clone())
    }
}

pub struct PrototypicalModel {
    #[allow(dead_code)]
    config: MetaLearningConfig,
}
impl PrototypicalModel {
    pub fn new(config: &MetaLearningConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.clone(),
        })
    }
}
impl MetaLearningModel for PrototypicalModel {
    fn forward(&mut self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.5)
    }
    fn compute_accuracy(&self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.8)
    }
    fn compute_gradients(&self, _loss: f64) -> Result<ModelGradients, TrustformersError> {
        Ok(ModelGradients::new())
    }
    fn apply_gradients(
        &mut self,
        _gradients: &ModelGradients,
        _lr: f64,
    ) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn get_parameters(&self) -> Result<ModelParameters, TrustformersError> {
        Ok(ModelParameters {
            parameters: HashMap::new(),
        })
    }
    fn set_parameters(&mut self, _params: ModelParameters) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn embed(&self, example: &Example) -> Result<Tensor, TrustformersError> {
        Ok(example.input.clone())
    }
}

pub struct MatchingNetModel {
    #[allow(dead_code)]
    config: MetaLearningConfig,
}
impl MatchingNetModel {
    pub fn new(config: &MetaLearningConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.clone(),
        })
    }
}
impl MetaLearningModel for MatchingNetModel {
    fn forward(&mut self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.5)
    }
    fn compute_accuracy(&self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.8)
    }
    fn compute_gradients(&self, _loss: f64) -> Result<ModelGradients, TrustformersError> {
        Ok(ModelGradients::new())
    }
    fn apply_gradients(
        &mut self,
        _gradients: &ModelGradients,
        _lr: f64,
    ) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn get_parameters(&self) -> Result<ModelParameters, TrustformersError> {
        Ok(ModelParameters {
            parameters: HashMap::new(),
        })
    }
    fn set_parameters(&mut self, _params: ModelParameters) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn embed(&self, example: &Example) -> Result<Tensor, TrustformersError> {
        Ok(example.input.clone())
    }
}

pub struct RelationNetModel {
    #[allow(dead_code)]
    config: MetaLearningConfig,
}
impl RelationNetModel {
    pub fn new(config: &MetaLearningConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.clone(),
        })
    }
}
impl MetaLearningModel for RelationNetModel {
    fn forward(&mut self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.5)
    }
    fn compute_accuracy(&self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.8)
    }
    fn compute_gradients(&self, _loss: f64) -> Result<ModelGradients, TrustformersError> {
        Ok(ModelGradients::new())
    }
    fn apply_gradients(
        &mut self,
        _gradients: &ModelGradients,
        _lr: f64,
    ) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn get_parameters(&self) -> Result<ModelParameters, TrustformersError> {
        Ok(ModelParameters {
            parameters: HashMap::new(),
        })
    }
    fn set_parameters(&mut self, _params: ModelParameters) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn embed(&self, example: &Example) -> Result<Tensor, TrustformersError> {
        Ok(example.input.clone())
    }
    fn compute_relation(&self, _emb1: &Tensor, _emb2: &Tensor) -> Result<f64, TrustformersError> {
        Ok(0.5)
    }
}

pub struct MemoryAugmentedModel {
    #[allow(dead_code)]
    config: MetaLearningConfig,
}
impl MemoryAugmentedModel {
    pub fn new(config: &MetaLearningConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.clone(),
        })
    }
}
impl MetaLearningModel for MemoryAugmentedModel {
    fn forward(&mut self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.5)
    }
    fn compute_accuracy(&self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.8)
    }
    fn compute_gradients(&self, _loss: f64) -> Result<ModelGradients, TrustformersError> {
        Ok(ModelGradients::new())
    }
    fn apply_gradients(
        &mut self,
        _gradients: &ModelGradients,
        _lr: f64,
    ) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn get_parameters(&self) -> Result<ModelParameters, TrustformersError> {
        Ok(ModelParameters {
            parameters: HashMap::new(),
        })
    }
    fn set_parameters(&mut self, _params: ModelParameters) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn embed(&self, example: &Example) -> Result<Tensor, TrustformersError> {
        Ok(example.input.clone())
    }
    fn write_to_memory(&mut self, _example: &Example) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn read_from_memory(&self, _example: &Example) -> Result<MemoryOutput, TrustformersError> {
        Ok(MemoryOutput {
            content: Tensor::zeros(&[64])?,
            attention_weights: vec![1.0],
        })
    }
    fn predict_from_memory(
        &self,
        _memory_output: &MemoryOutput,
    ) -> Result<MemoryPrediction, TrustformersError> {
        Ok(MemoryPrediction {
            logits: Tensor::zeros(&[5])?,
            confidence: 0.8,
        })
    }
}

pub struct GradientBasedModel {
    #[allow(dead_code)]
    config: MetaLearningConfig,
}
impl GradientBasedModel {
    pub fn new(config: &MetaLearningConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.clone(),
        })
    }
}
impl MetaLearningModel for GradientBasedModel {
    fn forward(&mut self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.5)
    }
    fn compute_accuracy(&self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.8)
    }
    fn compute_gradients(&self, _loss: f64) -> Result<ModelGradients, TrustformersError> {
        Ok(ModelGradients::new())
    }
    fn apply_gradients(
        &mut self,
        _gradients: &ModelGradients,
        _lr: f64,
    ) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn get_parameters(&self) -> Result<ModelParameters, TrustformersError> {
        Ok(ModelParameters {
            parameters: HashMap::new(),
        })
    }
    fn set_parameters(&mut self, _params: ModelParameters) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn embed(&self, example: &Example) -> Result<Tensor, TrustformersError> {
        Ok(example.input.clone())
    }
}

pub struct MetaSGDModel {
    #[allow(dead_code)]
    config: MetaLearningConfig,
}
impl MetaSGDModel {
    pub fn new(config: &MetaLearningConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.clone(),
        })
    }
}
impl MetaLearningModel for MetaSGDModel {
    fn forward(&mut self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.5)
    }
    fn compute_accuracy(&self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.8)
    }
    fn compute_gradients(&self, _loss: f64) -> Result<ModelGradients, TrustformersError> {
        Ok(ModelGradients::new())
    }
    fn apply_gradients(
        &mut self,
        _gradients: &ModelGradients,
        _lr: f64,
    ) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn get_parameters(&self) -> Result<ModelParameters, TrustformersError> {
        Ok(ModelParameters {
            parameters: HashMap::new(),
        })
    }
    fn set_parameters(&mut self, _params: ModelParameters) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn embed(&self, example: &Example) -> Result<Tensor, TrustformersError> {
        Ok(example.input.clone())
    }
}

pub struct L2LModel {
    #[allow(dead_code)]
    config: MetaLearningConfig,
}
impl L2LModel {
    pub fn new(config: &MetaLearningConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.clone(),
        })
    }
}
impl MetaLearningModel for L2LModel {
    fn forward(&mut self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.5)
    }
    fn compute_accuracy(&self, _examples: &ExampleSet) -> Result<f64, TrustformersError> {
        Ok(0.8)
    }
    fn compute_gradients(&self, _loss: f64) -> Result<ModelGradients, TrustformersError> {
        Ok(ModelGradients::new())
    }
    fn apply_gradients(
        &mut self,
        _gradients: &ModelGradients,
        _lr: f64,
    ) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn get_parameters(&self) -> Result<ModelParameters, TrustformersError> {
        Ok(ModelParameters {
            parameters: HashMap::new(),
        })
    }
    fn set_parameters(&mut self, _params: ModelParameters) -> Result<(), TrustformersError> {
        Ok(())
    }
    fn embed(&self, example: &Example) -> Result<Tensor, TrustformersError> {
        Ok(example.input.clone())
    }
}

/// Optimizer implementations
pub struct SGDMetaOptimizer {
    learning_rate: f64,
    accumulated_gradients: Option<ModelGradients>,
}

impl SGDMetaOptimizer {
    pub fn new(learning_rate: f64) -> Result<Self, TrustformersError> {
        Ok(Self {
            learning_rate,
            accumulated_gradients: None,
        })
    }
}

impl MetaOptimizer for SGDMetaOptimizer {
    fn step(&mut self, model: &mut dyn MetaLearningModel) -> Result<(), TrustformersError> {
        if let Some(gradients) = &self.accumulated_gradients {
            model.apply_gradients(gradients, self.learning_rate)?;
            self.accumulated_gradients = None;
        }
        Ok(())
    }

    fn accumulate_gradients(&mut self, gradients: ModelGradients) -> Result<(), TrustformersError> {
        self.accumulated_gradients = Some(gradients);
        Ok(())
    }

    fn reset(&mut self) -> Result<(), TrustformersError> {
        self.accumulated_gradients = None;
        Ok(())
    }
}

pub struct AdamMetaOptimizer {
    learning_rate: f64,
    accumulated_gradients: Option<ModelGradients>,
}

impl AdamMetaOptimizer {
    pub fn new(learning_rate: f64) -> Result<Self, TrustformersError> {
        Ok(Self {
            learning_rate,
            accumulated_gradients: None,
        })
    }
}

impl MetaOptimizer for AdamMetaOptimizer {
    fn step(&mut self, model: &mut dyn MetaLearningModel) -> Result<(), TrustformersError> {
        if let Some(gradients) = &self.accumulated_gradients {
            model.apply_gradients(gradients, self.learning_rate)?;
            self.accumulated_gradients = None;
        }
        Ok(())
    }

    fn accumulate_gradients(&mut self, gradients: ModelGradients) -> Result<(), TrustformersError> {
        self.accumulated_gradients = Some(gradients);
        Ok(())
    }

    fn reset(&mut self) -> Result<(), TrustformersError> {
        self.accumulated_gradients = None;
        Ok(())
    }
}

pub struct LearnedLROptimizer {
    learning_rate: f64,
    accumulated_gradients: Option<ModelGradients>,
    accumulated_lr_gradients: Option<Vec<f64>>,
}

impl LearnedLROptimizer {
    pub fn new(learning_rate: f64) -> Result<Self, TrustformersError> {
        Ok(Self {
            learning_rate,
            accumulated_gradients: None,
            accumulated_lr_gradients: None,
        })
    }
}

impl MetaOptimizer for LearnedLROptimizer {
    fn step(&mut self, model: &mut dyn MetaLearningModel) -> Result<(), TrustformersError> {
        if let Some(gradients) = &self.accumulated_gradients {
            model.apply_gradients(gradients, self.learning_rate)?;
            self.accumulated_gradients = None;
        }
        Ok(())
    }

    fn accumulate_gradients(&mut self, gradients: ModelGradients) -> Result<(), TrustformersError> {
        self.accumulated_gradients = Some(gradients);
        Ok(())
    }

    fn accumulate_lr_gradients(&mut self, lr_gradients: Vec<f64>) -> Result<(), TrustformersError> {
        self.accumulated_lr_gradients = Some(lr_gradients);
        Ok(())
    }

    fn reset(&mut self) -> Result<(), TrustformersError> {
        self.accumulated_gradients = None;
        self.accumulated_lr_gradients = None;
        Ok(())
    }
}

/// Utility functions
pub mod utils {
    use super::*;

    /// Create a few-shot classification task configuration
    pub fn create_few_shot_config(
        num_ways: usize,
        num_shots: usize,
        query_size: usize,
    ) -> MetaLearningConfig {
        MetaLearningConfig {
            num_ways,
            num_shots,
            support_size: num_ways * num_shots,
            query_size,
            ..Default::default()
        }
    }

    /// Create MAML configuration with sensible defaults
    pub fn create_maml_config() -> MetaLearningConfig {
        MetaLearningConfig {
            algorithm: MetaAlgorithm::MAML,
            inner_lr: 0.01,
            meta_lr: 0.001,
            inner_steps: 5,
            first_order: false,
            ..Default::default()
        }
    }

    /// Create Reptile configuration (first-order MAML)
    pub fn create_reptile_config() -> MetaLearningConfig {
        MetaLearningConfig {
            algorithm: MetaAlgorithm::Reptile,
            inner_lr: 0.01,
            meta_lr: 0.001,
            inner_steps: 10,
            first_order: true,
            ..Default::default()
        }
    }

    /// Create Prototypical Networks configuration
    pub fn create_protonet_config() -> MetaLearningConfig {
        MetaLearningConfig {
            algorithm: MetaAlgorithm::ProtoNet,
            temperature: 1.0,
            normalize_embeddings: true,
            embedding_dim: 512,
            ..Default::default()
        }
    }

    /// Calculate meta-learning performance metrics
    pub fn calculate_performance_metrics(episode_results: &[EpisodeResult]) -> PerformanceMetrics {
        if episode_results.is_empty() {
            return PerformanceMetrics::default();
        }

        let accuracies: Vec<f64> = episode_results.iter().map(|r| r.meta_accuracy).collect();
        let mean_accuracy = accuracies.iter().sum::<f64>() / accuracies.len() as f64;

        let variance = accuracies.iter().map(|acc| (acc - mean_accuracy).powi(2)).sum::<f64>()
            / accuracies.len() as f64;
        let std_dev = variance.sqrt();

        let max_accuracy = accuracies.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_accuracy = accuracies.iter().fold(1.0f64, |a, &b| a.min(b));

        PerformanceMetrics {
            mean_accuracy,
            std_dev,
            max_accuracy,
            min_accuracy,
            num_episodes: episode_results.len(),
        }
    }

    /// Estimate convergence based on recent performance
    pub fn estimate_convergence(
        episode_results: &[EpisodeResult],
        window_size: usize,
    ) -> ConvergenceMetrics {
        if episode_results.len() < window_size * 2 {
            return ConvergenceMetrics::default();
        }

        let recent_window = &episode_results[episode_results.len() - window_size..];
        let older_window = &episode_results
            [episode_results.len() - window_size * 2..episode_results.len() - window_size];

        let recent_mean =
            recent_window.iter().map(|r| r.meta_accuracy).sum::<f64>() / window_size as f64;
        let older_mean =
            older_window.iter().map(|r| r.meta_accuracy).sum::<f64>() / window_size as f64;

        let improvement_rate = recent_mean - older_mean;
        let has_converged = improvement_rate.abs() < 0.001;

        ConvergenceMetrics {
            improvement_rate,
            has_converged,
            recent_mean,
            older_mean,
        }
    }
}

#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    pub mean_accuracy: f64,
    pub std_dev: f64,
    pub max_accuracy: f64,
    pub min_accuracy: f64,
    pub num_episodes: usize,
}

#[derive(Debug, Default)]
pub struct ConvergenceMetrics {
    pub improvement_rate: f64,
    pub has_converged: bool,
    pub recent_mean: f64,
    pub older_mean: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_learning_config_default() {
        let config = MetaLearningConfig::default();
        assert_eq!(config.algorithm, MetaAlgorithm::MAML);
        assert_eq!(config.num_ways, 5);
        assert_eq!(config.num_shots, 1);
    }

    #[test]
    fn test_meta_learner_creation() {
        let config = MetaLearningConfig::default();
        let result = MetaLearner::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_task_sampler() {
        let config = MetaLearningConfig::default();
        let mut sampler = TaskSampler::new(&config).expect("operation failed");
        let task_batch = sampler.sample_batch(4).expect("operation failed");
        assert_eq!(task_batch.tasks.len(), 4);
    }

    #[test]
    fn test_meta_statistics() {
        let mut stats = MetaStatistics::new();
        let episode_result = EpisodeResult {
            episode: 0,
            meta_loss: 0.5,
            meta_accuracy: 0.8,
            num_tasks: 10,
            episode_time: std::time::Duration::from_millis(100),
            algorithm: MetaAlgorithm::MAML,
        };

        stats.update(&episode_result);
        assert!(stats.total_episodes > 0);
        assert!(stats.best_accuracy > 0.0);
    }

    #[test]
    fn test_utils_few_shot_config() {
        let config = utils::create_few_shot_config(5, 1, 15);
        assert_eq!(config.num_ways, 5);
        assert_eq!(config.num_shots, 1);
        assert_eq!(config.support_size, 5);
        assert_eq!(config.query_size, 15);
    }

    #[test]
    fn test_meta_algorithms() {
        assert_ne!(MetaAlgorithm::MAML, MetaAlgorithm::Reptile);
        assert_eq!(MetaAlgorithm::ProtoNet as u8, 2);
    }

    #[test]
    fn test_performance_metrics_calculation() {
        let episode_results = vec![
            EpisodeResult {
                episode: 0,
                meta_loss: 0.5,
                meta_accuracy: 0.8,
                num_tasks: 10,
                episode_time: std::time::Duration::from_millis(100),
                algorithm: MetaAlgorithm::MAML,
            },
            EpisodeResult {
                episode: 1,
                meta_loss: 0.4,
                meta_accuracy: 0.85,
                num_tasks: 10,
                episode_time: std::time::Duration::from_millis(100),
                algorithm: MetaAlgorithm::MAML,
            },
        ];

        let metrics = utils::calculate_performance_metrics(&episode_results);
        assert!(metrics.mean_accuracy > 0.8);
        assert!(metrics.std_dev >= 0.0);
        assert_eq!(metrics.num_episodes, 2);
    }
}
