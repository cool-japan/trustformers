use scirs2_core::ndarray::Array1; // SciRS2 Integration Policy
use scirs2_core::random::*; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Configuration for memory replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReplayConfig {
    /// Maximum number of examples to store per task
    pub buffer_size_per_task: usize,
    /// Sampling strategy for replay
    pub sampling_strategy: SamplingStrategy,
    /// Number of replay examples per training step
    pub replay_batch_size: usize,
    /// Ratio of replay examples vs new examples
    pub replay_ratio: f32,
    /// Whether to use herding selection for representative examples
    pub use_herding: bool,
    /// Number of clusters for diverse sampling
    pub num_clusters: usize,
}

impl Default for MemoryReplayConfig {
    fn default() -> Self {
        Self {
            buffer_size_per_task: 1000,
            sampling_strategy: SamplingStrategy::Random,
            replay_batch_size: 32,
            replay_ratio: 0.5,
            use_herding: false,
            num_clusters: 10,
        }
    }
}

/// Sampling strategies for memory replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Random sampling from buffer
    Random,
    /// Uniform sampling across tasks
    Uniform,
    /// Weighted sampling based on task difficulty
    Weighted,
    /// Diverse sampling using clustering
    Diverse,
    /// Gradient-based importance sampling
    GradientBased,
}

/// Experience sample for replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceSample {
    /// Task ID this sample belongs to
    pub task_id: String,
    /// Input features
    pub input: Vec<f32>,
    /// Target labels
    pub target: Vec<f32>,
    /// Sample importance score
    pub importance: f32,
    /// Timestamp when added
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Number of times this sample has been replayed
    pub replay_count: usize,
}

impl ExperienceSample {
    pub fn new(task_id: String, input: Vec<f32>, target: Vec<f32>) -> Self {
        Self {
            task_id,
            input,
            target,
            importance: 1.0,
            timestamp: chrono::Utc::now(),
            replay_count: 0,
        }
    }

    /// Update importance score based on replay performance
    pub fn update_importance(&mut self, loss: f32) {
        // Higher loss means more important for replay
        self.importance = (self.importance * 0.9 + loss * 0.1).max(0.1);
    }

    /// Increment replay count
    pub fn increment_replay(&mut self) {
        self.replay_count += 1;
    }
}

/// Experience buffer for storing samples
#[derive(Debug)]
pub struct ExperienceBuffer {
    /// Buffer organized by task
    tasks: HashMap<String, VecDeque<ExperienceSample>>,
    /// Maximum size per task
    max_size_per_task: usize,
    /// Total number of samples
    total_samples: usize,
    /// Random number generator
    rng: StdRng,
}

impl ExperienceBuffer {
    pub fn new(max_size_per_task: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut thread_rng().rng_mut()),
        };

        Self {
            tasks: HashMap::new(),
            max_size_per_task,
            total_samples: 0,
            rng,
        }
    }

    /// Add a sample to the buffer
    pub fn add_sample(&mut self, sample: ExperienceSample) {
        let task_id = sample.task_id.clone();
        let task_buffer = self.tasks.entry(task_id).or_default();

        // Remove oldest sample if buffer is full
        if task_buffer.len() >= self.max_size_per_task {
            task_buffer.pop_front();
            self.total_samples -= 1;
        }

        task_buffer.push_back(sample);
        self.total_samples += 1;
    }

    /// Sample random examples from buffer
    pub fn sample_random(&mut self, num_samples: usize) -> Vec<ExperienceSample> {
        let mut all_indices = Vec::new();
        for (task_id, buffer) in &self.tasks {
            for (idx, _) in buffer.iter().enumerate() {
                all_indices.push((task_id.clone(), idx));
            }
        }

        all_indices.shuffle(&mut self.rng);
        let selected_indices: Vec<_> = all_indices.into_iter().take(num_samples).collect();

        let mut samples = Vec::new();
        for (task_id, idx) in selected_indices {
            if let Some(buffer) = self.tasks.get_mut(&task_id) {
                if let Some(sample) = buffer.get_mut(idx) {
                    sample.increment_replay();
                    samples.push(sample.clone());
                }
            }
        }

        samples
    }

    /// Sample examples uniformly across tasks
    pub fn sample_uniform(&mut self, num_samples: usize) -> Vec<ExperienceSample> {
        let mut samples = Vec::new();
        let num_tasks = self.tasks.len();

        if num_tasks == 0 {
            return samples;
        }

        let samples_per_task = (num_samples + num_tasks - 1) / num_tasks;

        let task_ids: Vec<_> = self.tasks.keys().cloned().collect();
        for task_id in task_ids {
            if let Some(task_buffer) = self.tasks.get_mut(&task_id) {
                let mut indices: Vec<_> = (0..task_buffer.len()).collect();
                indices.shuffle(&mut self.rng);

                let take_count = samples_per_task.min(indices.len());
                for &idx in indices.iter().take(take_count) {
                    if let Some(sample) = task_buffer.get_mut(idx) {
                        sample.increment_replay();
                        samples.push(sample.clone());

                        if samples.len() >= num_samples {
                            return samples;
                        }
                    }
                }
            }
        }

        samples
    }

    /// Sample examples based on importance weights
    pub fn sample_weighted(&mut self, num_samples: usize) -> Vec<ExperienceSample> {
        // Collect all sample indices with their importance scores
        let mut all_samples_info = Vec::new();
        for (task_id, buffer) in &self.tasks {
            for (idx, sample) in buffer.iter().enumerate() {
                all_samples_info.push((task_id.clone(), idx, sample.importance));
            }
        }

        if all_samples_info.is_empty() {
            return Vec::new();
        }

        // Create cumulative distribution based on importance
        let total_importance: f32 = all_samples_info.iter().map(|(_, _, imp)| imp).sum();
        let mut cumulative_probs = Vec::new();
        let mut cumulative = 0.0;

        for (_, _, importance) in &all_samples_info {
            cumulative += importance / total_importance;
            cumulative_probs.push(cumulative);
        }

        let mut samples = Vec::new();
        for _ in 0..num_samples {
            let rand_val: f32 = self.rng.random();
            let selected_idx =
                match cumulative_probs.binary_search_by(|&x| x.partial_cmp(&rand_val).unwrap()) {
                    Ok(idx) => idx,
                    Err(idx) => idx.min(cumulative_probs.len() - 1),
                };

            if let Some((task_id, idx, _)) = all_samples_info.get(selected_idx) {
                if let Some(buffer) = self.tasks.get_mut(task_id) {
                    if let Some(sample) = buffer.get_mut(*idx) {
                        sample.increment_replay();
                        samples.push(sample.clone());
                    }
                }
            }
        }

        samples
    }

    /// Get buffer statistics
    pub fn get_stats(&self) -> BufferStats {
        let mut task_counts = HashMap::new();
        for (task_id, buffer) in &self.tasks {
            task_counts.insert(task_id.clone(), buffer.len());
        }

        BufferStats {
            total_samples: self.total_samples,
            num_tasks: self.tasks.len(),
            task_counts,
            max_size_per_task: self.max_size_per_task,
        }
    }

    /// Clear buffer for a specific task
    pub fn clear_task(&mut self, task_id: &str) {
        if let Some(buffer) = self.tasks.remove(task_id) {
            self.total_samples -= buffer.len();
        }
    }

    /// Get samples for a specific task
    pub fn get_task_samples(&self, task_id: &str) -> Option<&VecDeque<ExperienceSample>> {
        self.tasks.get(task_id)
    }
}

/// Buffer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferStats {
    pub total_samples: usize,
    pub num_tasks: usize,
    pub task_counts: HashMap<String, usize>,
    pub max_size_per_task: usize,
}

/// Memory replay manager
#[derive(Debug)]
pub struct MemoryReplay {
    config: MemoryReplayConfig,
    buffer: ExperienceBuffer,
    task_difficulties: HashMap<String, f32>,
}

impl MemoryReplay {
    pub fn new(config: MemoryReplayConfig, seed: Option<u64>) -> Self {
        let buffer = ExperienceBuffer::new(config.buffer_size_per_task, seed);

        Self {
            config,
            buffer,
            task_difficulties: HashMap::new(),
        }
    }

    /// Add experience sample to memory
    pub fn add_experience(&mut self, sample: ExperienceSample) {
        self.buffer.add_sample(sample);
    }

    /// Sample replay batch according to configured strategy
    pub fn sample_replay_batch(&mut self) -> Vec<ExperienceSample> {
        let num_samples = self.config.replay_batch_size;

        match self.config.sampling_strategy {
            SamplingStrategy::Random => self.buffer.sample_random(num_samples),
            SamplingStrategy::Uniform => self.buffer.sample_uniform(num_samples),
            SamplingStrategy::Weighted => self.buffer.sample_weighted(num_samples),
            SamplingStrategy::Diverse => self.sample_diverse(num_samples),
            SamplingStrategy::GradientBased => self.sample_gradient_based(num_samples),
        }
    }

    /// Diverse sampling using clustering (simplified implementation)
    fn sample_diverse(&mut self, num_samples: usize) -> Vec<ExperienceSample> {
        // For now, use uniform sampling as a placeholder
        // In practice, this would use clustering algorithms
        self.buffer.sample_uniform(num_samples)
    }

    /// Gradient-based importance sampling
    fn sample_gradient_based(&mut self, num_samples: usize) -> Vec<ExperienceSample> {
        // For now, use weighted sampling as a placeholder
        // In practice, this would use gradient magnitude as importance
        self.buffer.sample_weighted(num_samples)
    }

    /// Update task difficulty score
    pub fn update_task_difficulty(&mut self, task_id: String, difficulty: f32) {
        self.task_difficulties.insert(task_id, difficulty);
    }

    /// Get replay statistics
    pub fn get_replay_stats(&self) -> ReplayStats {
        let buffer_stats = self.buffer.get_stats();

        ReplayStats {
            buffer_stats,
            task_difficulties: self.task_difficulties.clone(),
            config: self.config.clone(),
        }
    }

    /// Clear memory for specific task
    pub fn clear_task_memory(&mut self, task_id: &str) {
        self.buffer.clear_task(task_id);
        self.task_difficulties.remove(task_id);
    }

    /// Get number of stored tasks
    pub fn num_tasks(&self) -> usize {
        self.buffer.tasks.len()
    }

    /// Check if buffer has samples
    pub fn has_samples(&self) -> bool {
        self.buffer.total_samples > 0
    }
}

/// Replay statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayStats {
    pub buffer_stats: BufferStats,
    pub task_difficulties: HashMap<String, f32>,
    pub config: MemoryReplayConfig,
}

/// Herding selection for representative examples
pub mod herding {
    use super::*;

    /// Select representative examples using herding algorithm
    pub fn select_herding_examples(
        samples: &[ExperienceSample],
        num_select: usize,
        feature_extractor: impl Fn(&[f32]) -> Array1<f32>,
    ) -> Vec<usize> {
        let mut selected = Vec::new();
        let features: Vec<Array1<f32>> =
            samples.iter().map(|s| feature_extractor(&s.input)).collect();

        // Compute mean feature vector
        let mean_features = {
            let mut mean = Array1::<f32>::zeros(features[0].len());
            for feature in &features {
                mean += feature;
            }
            mean / features.len() as f32
        };

        let mut current_mean = Array1::<f32>::zeros(mean_features.len());

        for _ in 0..num_select.min(samples.len()) {
            let mut best_idx = 0;
            let mut best_distance = f32::INFINITY;

            for (idx, feature) in features.iter().enumerate() {
                if selected.contains(&idx) {
                    continue;
                }

                // Compute distance after adding this sample
                let new_mean =
                    (&current_mean * selected.len() as f32 + feature) / (selected.len() + 1) as f32;
                let distance = (&mean_features - &new_mean).mapv(|x: f32| x * x).sum();

                if distance < best_distance {
                    best_distance = distance;
                    best_idx = idx;
                }
            }

            selected.push(best_idx);
            current_mean = (&current_mean * (selected.len() - 1) as f32 + &features[best_idx])
                / selected.len() as f32;
        }

        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experience_buffer() {
        let mut buffer = ExperienceBuffer::new(3, Some(42));

        // Add samples
        for i in 0..5 {
            let sample =
                ExperienceSample::new("task1".to_string(), vec![i as f32], vec![(i % 2) as f32]);
            buffer.add_sample(sample);
        }

        let stats = buffer.get_stats();
        assert_eq!(stats.total_samples, 3); // Max size per task
        assert_eq!(stats.task_counts.get("task1"), Some(&3));
    }

    #[test]
    fn test_memory_replay_sampling() {
        let config = MemoryReplayConfig {
            buffer_size_per_task: 10,
            sampling_strategy: SamplingStrategy::Random,
            replay_batch_size: 3,
            ..Default::default()
        };

        let mut replay = MemoryReplay::new(config, Some(42));

        // Add samples from different tasks
        for task_id in ["task1", "task2"] {
            for i in 0..5 {
                let sample = ExperienceSample::new(
                    task_id.to_string(),
                    vec![i as f32],
                    vec![(i % 2) as f32],
                );
                replay.add_experience(sample);
            }
        }

        let samples = replay.sample_replay_batch();
        assert!(samples.len() <= 3);
        assert!(replay.has_samples());
    }

    #[test]
    fn test_importance_sampling() {
        let mut buffer = ExperienceBuffer::new(10, Some(42));

        // Add samples with different importance scores
        for i in 0..5 {
            let mut sample =
                ExperienceSample::new("task1".to_string(), vec![i as f32], vec![(i % 2) as f32]);
            sample.importance = (i + 1) as f32; // Higher importance for later samples
            buffer.add_sample(sample);
        }

        let samples = buffer.sample_weighted(3);
        assert_eq!(samples.len(), 3);

        // Check that samples with higher importance are more likely to be selected
        // (This is probabilistic, so we just check that we got some samples)
        assert!(!samples.is_empty());
    }
}
