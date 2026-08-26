use anyhow::{anyhow, Result};
use scirs2_core::ndarray::Array1; // SciRS2 Integration Policy
use scirs2_core::random::*; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Squared Euclidean distance between two feature vectors.
///
/// Vectors of unequal length are compared as if the shorter one were zero-padded, so a buffer
/// holding heterogeneous inputs still yields a well-defined metric.
fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().max(b.len());
    let mut total = 0.0f32;
    for i in 0..len {
        let x = a.get(i).copied().unwrap_or(0.0);
        let y = b.get(i).copied().unwrap_or(0.0);
        let d = x - y;
        total += d * d;
    }
    total
}

/// Component-wise mean of a set of feature vectors, zero-padded to the longest.
fn mean_vector(vectors: &[Vec<f32>]) -> Vec<f32> {
    let len = vectors.iter().map(|v| v.len()).max().unwrap_or(0);
    if vectors.is_empty() || len == 0 {
        return vec![0.0; len];
    }
    let mut acc = vec![0.0f32; len];
    for vector in vectors {
        for (i, slot) in acc.iter_mut().enumerate() {
            *slot += vector.get(i).copied().unwrap_or(0.0);
        }
    }
    let n = vectors.len() as f32;
    for slot in acc.iter_mut() {
        *slot /= n;
    }
    acc
}

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
    /// Gradient norm ‖∇L(x)‖ most recently measured for this sample, when one was recorded.
    ///
    /// This is the quantity [`SamplingStrategy::GradientBased`] builds its importance
    /// distribution from. `None` means "never measured" — deliberately distinct from a
    /// measured `0.0`, so the sampler can refuse to invent an importance it was never given.
    #[serde(default)]
    pub gradient_norm: Option<f32>,
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
            gradient_norm: None,
        }
    }

    /// Record the gradient norm measured for this sample during a backward pass.
    ///
    /// # Errors
    ///
    /// `norm` is negative or not finite — neither is a valid magnitude.
    pub fn set_gradient_norm(&mut self, norm: f32) -> Result<()> {
        if !norm.is_finite() || norm < 0.0 {
            return Err(anyhow!(
                "gradient norm must be finite and non-negative, got {norm}"
            ));
        }
        self.gradient_norm = Some(norm);
        Ok(())
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

        let samples_per_task = num_samples.div_ceil(num_tasks);

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
            let selected_idx = match cumulative_probs.binary_search_by(|&x| {
                x.partial_cmp(&rand_val).unwrap_or(std::cmp::Ordering::Equal)
            }) {
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

    /// Stable `(task_id, index)` keys for every buffered sample.
    ///
    /// Sorted by task id so that selection algorithms built on top of it are reproducible —
    /// `self.tasks` is a `HashMap` and its iteration order changes between runs.
    pub fn sample_keys(&self) -> Vec<(String, usize)> {
        let mut task_ids: Vec<&String> = self.tasks.keys().collect();
        task_ids.sort();
        let mut keys = Vec::with_capacity(self.total_samples);
        for task_id in task_ids {
            if let Some(buffer) = self.tasks.get(task_id) {
                for idx in 0..buffer.len() {
                    keys.push((task_id.clone(), idx));
                }
            }
        }
        keys
    }

    /// Borrow one buffered sample by its [`ExperienceBuffer::sample_keys`] key.
    pub fn sample_at(&self, task_id: &str, index: usize) -> Option<&ExperienceSample> {
        self.tasks.get(task_id).and_then(|buffer| buffer.get(index))
    }

    /// Clone out the samples named by `keys`, incrementing each one's replay counter.
    ///
    /// Keys that no longer resolve are skipped rather than producing a placeholder sample.
    pub fn take_by_keys(
        &mut self,
        keys: impl Iterator<Item = (String, usize)>,
    ) -> Vec<ExperienceSample> {
        let mut samples = Vec::new();
        for (task_id, index) in keys {
            if let Some(buffer) = self.tasks.get_mut(&task_id) {
                if let Some(sample) = buffer.get_mut(index) {
                    sample.increment_replay();
                    samples.push(sample.clone());
                }
            }
        }
        samples
    }

    /// Attach a measured gradient norm to one buffered sample.
    ///
    /// # Errors
    ///
    /// `task_id` is unknown, `index` is out of range for that task, or `norm` is negative or
    /// not finite.
    pub fn record_gradient_norm(&mut self, task_id: &str, index: usize, norm: f32) -> Result<()> {
        let buffer = self
            .tasks
            .get_mut(task_id)
            .ok_or_else(|| anyhow!("no replay buffer for task '{task_id}'"))?;
        let len = buffer.len();
        let sample = buffer.get_mut(index).ok_or_else(|| {
            anyhow!("sample index {index} is out of range for task '{task_id}' ({len} samples)")
        })?;
        sample.set_gradient_norm(norm)
    }

    /// Draw the next uniform variate in `[0, 1)` from the buffer's seeded generator.
    ///
    /// Exposed so selection algorithms living on [`MemoryReplay`] stay on the same
    /// reproducible stream as the buffer's own samplers.
    pub fn next_uniform(&mut self) -> f32 {
        self.rng.random()
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

    /// Sample a replay batch with the configured [`SamplingStrategy`].
    ///
    /// Each strategy runs its own algorithm — `Diverse` and `GradientBased` used to silently
    /// delegate to uniform and weighted sampling respectively, so selecting them changed
    /// nothing about the batch.
    ///
    /// # Errors
    ///
    /// Only the strategies that can be *undefined* for the current buffer state fail:
    /// [`SamplingStrategy::GradientBased`] returns an error when no sample carries a recorded
    /// gradient norm (see [`MemoryReplay::record_gradient_norm`]), because an importance
    /// distribution proportional to a quantity that was never measured does not exist. The
    /// other strategies return an empty batch for an empty buffer, as before.
    pub fn sample_replay_batch(&mut self) -> Result<Vec<ExperienceSample>> {
        let num_samples = self.config.replay_batch_size;

        match self.config.sampling_strategy {
            SamplingStrategy::Random => Ok(self.buffer.sample_random(num_samples)),
            SamplingStrategy::Uniform => Ok(self.buffer.sample_uniform(num_samples)),
            SamplingStrategy::Weighted => Ok(self.buffer.sample_weighted(num_samples)),
            SamplingStrategy::Diverse => Ok(self.sample_diverse(num_samples)),
            SamplingStrategy::GradientBased => self.sample_gradient_based(num_samples),
        }
    }

    /// Record the gradient norm ‖∇L(xᵢ)‖ measured for one buffered sample.
    ///
    /// [`SamplingStrategy::GradientBased`] draws its distribution from exactly these values,
    /// so a training loop that wants gradient-based replay must call this after each backward
    /// pass over a replayed (or newly buffered) example.
    ///
    /// # Errors
    ///
    /// The task is unknown, `index` is past the end of that task's buffer, or `norm` is
    /// negative or not finite.
    pub fn record_gradient_norm(&mut self, task_id: &str, index: usize, norm: f32) -> Result<()> {
        self.buffer.record_gradient_norm(task_id, index, norm)
    }

    /// Diverse replay selection by farthest-point clustering over the stored inputs.
    ///
    /// The algorithm is k-center greedy (farthest-point traversal), the standard 2-approximation
    /// for the k-center objective:
    ///
    /// 1. `k = min(config.num_clusters, |buffer|)` centers are chosen greedily — the first is
    ///    the sample farthest from the pool centroid, and each subsequent center is the sample
    ///    whose distance to the nearest existing center is largest.
    /// 2. Every sample is assigned to its nearest center, forming `k` clusters.
    /// 3. The batch is drawn round-robin across clusters, taking from each the still-unpicked
    ///    sample closest to that cluster's center.
    ///
    /// Round-robin over maximally-separated clusters is what makes the batch cover the input
    /// space rather than over-represent whichever region happens to hold the most samples.
    ///
    /// Distances are squared Euclidean, with shorter feature vectors zero-padded to the longer
    /// length so heterogeneous inputs are still comparable.
    fn sample_diverse(&mut self, num_samples: usize) -> Vec<ExperienceSample> {
        if num_samples == 0 {
            return Vec::new();
        }
        let pool = self.buffer.sample_keys();
        if pool.is_empty() {
            return Vec::new();
        }

        let inputs: Vec<Vec<f32>> = pool
            .iter()
            .map(|(task_id, idx)| {
                self.buffer
                    .sample_at(task_id, *idx)
                    .map(|s| s.input.clone())
                    .unwrap_or_default()
            })
            .collect();

        let k = self.config.num_clusters.clamp(1, pool.len());

        // ── 1. k-center greedy ───────────────────────────────────────────────
        // Seed deterministically with the point farthest from the centroid, so the same
        // buffer always yields the same batch.
        let centroid = mean_vector(&inputs);
        let mut centers: Vec<usize> = Vec::with_capacity(k);
        let seed = (0..inputs.len())
            .max_by(|&a, &b| {
                squared_distance(&inputs[a], &centroid)
                    .partial_cmp(&squared_distance(&inputs[b], &centroid))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        centers.push(seed);

        // Distance from each point to its nearest chosen center.
        let mut nearest: Vec<f32> =
            inputs.iter().map(|v| squared_distance(v, &inputs[seed])).collect();
        let mut assignment: Vec<usize> = vec![0; inputs.len()];

        while centers.len() < k {
            let next = (0..inputs.len())
                .max_by(|&a, &b| {
                    nearest[a].partial_cmp(&nearest[b]).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            if nearest[next] <= 0.0 {
                // Every remaining point coincides with an existing center: more clusters
                // would be empty, so stop early rather than emit duplicates.
                break;
            }
            let center_slot = centers.len();
            centers.push(next);
            for i in 0..inputs.len() {
                let d = squared_distance(&inputs[i], &inputs[next]);
                if d < nearest[i] {
                    nearest[i] = d;
                    assignment[i] = center_slot;
                }
            }
        }

        // ── 2. cluster membership, each ordered by distance to its center ─────
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); centers.len()];
        for (i, &slot) in assignment.iter().enumerate() {
            clusters[slot.min(centers.len() - 1)].push(i);
        }
        for (slot, members) in clusters.iter_mut().enumerate() {
            let center = &inputs[centers[slot]];
            members.sort_by(|&a, &b| {
                squared_distance(&inputs[a], center)
                    .partial_cmp(&squared_distance(&inputs[b], center))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
        }

        // ── 3. round-robin draw ──────────────────────────────────────────────
        let mut chosen: Vec<usize> = Vec::with_capacity(num_samples.min(pool.len()));
        let mut cursors = vec![0usize; clusters.len()];
        while chosen.len() < num_samples.min(pool.len()) {
            let mut progressed = false;
            for (slot, members) in clusters.iter().enumerate() {
                if chosen.len() >= num_samples.min(pool.len()) {
                    break;
                }
                if let Some(&point) = members.get(cursors[slot]) {
                    cursors[slot] += 1;
                    chosen.push(point);
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }

        self.buffer.take_by_keys(chosen.into_iter().map(|i| pool[i].clone()))
    }

    /// GraNd-style gradient-norm importance sampling.
    ///
    /// Samples are drawn **without replacement** with probability proportional to the gradient
    /// norm recorded by [`MemoryReplay::record_gradient_norm`], which is the textbook
    /// importance distribution for variance-reduced replay: the examples the model is still
    /// getting wrong are the ones worth replaying.
    ///
    /// Samples with no recorded gradient norm are not eligible — they have no measured
    /// importance, so including them would mean inventing one.
    ///
    /// # Errors
    ///
    /// No buffered sample carries a recorded gradient norm, or every recorded norm is zero
    /// (a degenerate distribution). Both mean the caller has not fed gradient magnitudes back
    /// into the buffer, which is a usage error rather than an empty batch.
    fn sample_gradient_based(&mut self, num_samples: usize) -> Result<Vec<ExperienceSample>> {
        let pool = self.buffer.sample_keys();
        if pool.is_empty() {
            return Ok(Vec::new());
        }

        let mut eligible: Vec<(usize, f32)> = Vec::new();
        let mut any_recorded = false;
        for (i, (task_id, idx)) in pool.iter().enumerate() {
            if let Some(sample) = self.buffer.sample_at(task_id, *idx) {
                if let Some(norm) = sample.gradient_norm {
                    any_recorded = true;
                    if norm > 0.0 {
                        eligible.push((i, norm));
                    }
                }
            }
        }

        if !any_recorded {
            return Err(anyhow!(
                "SamplingStrategy::GradientBased needs per-sample gradient norms, but none of \
                 the {} buffered samples has one; call MemoryReplay::record_gradient_norm \
                 after the backward pass (or choose SamplingStrategy::Weighted, which uses the \
                 `importance` field instead)",
                pool.len()
            ));
        }
        if eligible.is_empty() {
            return Err(anyhow!(
                "SamplingStrategy::GradientBased: every recorded gradient norm is zero, so the \
                 importance distribution is degenerate"
            ));
        }

        // Sequential draw without replacement: pick proportionally, then remove the winner and
        // renormalise. `num_samples` larger than the eligible set simply returns the whole set.
        let take = num_samples.min(eligible.len());
        let mut chosen = Vec::with_capacity(take);
        for _ in 0..take {
            let total: f32 = eligible.iter().map(|(_, w)| *w).sum();
            if total <= 0.0 {
                break;
            }
            let target: f32 = self.buffer.next_uniform() * total;
            let mut acc = 0.0;
            let mut winner = eligible.len() - 1;
            for (slot, (_, weight)) in eligible.iter().enumerate() {
                acc += *weight;
                if acc >= target {
                    winner = slot;
                    break;
                }
            }
            let (point, _) = eligible.remove(winner);
            chosen.push(pool[point].clone());
        }

        Ok(self.buffer.take_by_keys(chosen.into_iter()))
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

        let samples = replay.sample_replay_batch().expect("replay sampling failed");
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

    #[test]
    fn test_memory_replay_config_default() {
        let config = MemoryReplayConfig::default();
        assert_eq!(config.buffer_size_per_task, 1000);
        assert_eq!(config.replay_batch_size, 32);
        assert_eq!(config.replay_ratio, 0.5);
        assert!(!config.use_herding);
        assert_eq!(config.num_clusters, 10);
    }

    #[test]
    fn test_sampling_strategy_variants() {
        let strategies = [
            SamplingStrategy::Random,
            SamplingStrategy::Uniform,
            SamplingStrategy::Weighted,
            SamplingStrategy::Diverse,
            SamplingStrategy::GradientBased,
        ];
        assert_eq!(strategies.len(), 5);
    }

    #[test]
    fn test_experience_sample_creation() {
        let sample = ExperienceSample::new("task_1".to_string(), vec![1.0, 2.0], vec![0.0]);
        assert_eq!(sample.task_id, "task_1");
        assert_eq!(sample.input.len(), 2);
        assert_eq!(sample.target.len(), 1);
        assert_eq!(sample.importance, 1.0);
        assert_eq!(sample.replay_count, 0);
    }

    #[test]
    fn test_experience_sample_update_importance() {
        let mut sample = ExperienceSample::new("task_1".to_string(), vec![1.0], vec![0.0]);
        sample.update_importance(5.0);
        assert!(sample.importance > 0.1);
        assert!(sample.importance < 5.0);
    }

    #[test]
    fn test_experience_sample_increment_replay() {
        let mut sample = ExperienceSample::new("task_1".to_string(), vec![1.0], vec![0.0]);
        assert_eq!(sample.replay_count, 0);
        sample.increment_replay();
        assert_eq!(sample.replay_count, 1);
        sample.increment_replay();
        assert_eq!(sample.replay_count, 2);
    }

    #[test]
    fn test_experience_buffer_empty() {
        let buffer = ExperienceBuffer::new(10, Some(42));
        let stats = buffer.get_stats();
        assert_eq!(stats.total_samples, 0);
        assert!(stats.task_counts.is_empty());
    }

    #[test]
    fn test_experience_buffer_multiple_tasks() {
        let mut buffer = ExperienceBuffer::new(5, Some(42));
        for i in 0..3 {
            let sample = ExperienceSample::new("task_a".to_string(), vec![i as f32], vec![0.0]);
            buffer.add_sample(sample);
        }
        for i in 0..4 {
            let sample = ExperienceSample::new("task_b".to_string(), vec![i as f32], vec![1.0]);
            buffer.add_sample(sample);
        }
        let stats = buffer.get_stats();
        assert_eq!(stats.task_counts.get("task_a"), Some(&3));
        assert_eq!(stats.task_counts.get("task_b"), Some(&4));
    }

    #[test]
    fn test_experience_buffer_overflow() {
        let mut buffer = ExperienceBuffer::new(2, Some(42));
        for i in 0..5 {
            let sample = ExperienceSample::new("task1".to_string(), vec![i as f32], vec![0.0]);
            buffer.add_sample(sample);
        }
        let stats = buffer.get_stats();
        assert_eq!(stats.total_samples, 2); // Only last 2 kept
    }

    #[test]
    fn test_memory_replay_creation() {
        let config = MemoryReplayConfig::default();
        let replay = MemoryReplay::new(config, Some(42));
        assert!(!replay.has_samples());
    }

    #[test]
    fn test_memory_replay_has_samples() {
        let config = MemoryReplayConfig::default();
        let mut replay = MemoryReplay::new(config, Some(42));
        assert!(!replay.has_samples());
        let sample = ExperienceSample::new("task1".to_string(), vec![1.0], vec![0.0]);
        replay.add_experience(sample);
        assert!(replay.has_samples());
    }

    #[test]
    fn test_memory_replay_uniform_sampling() {
        let config = MemoryReplayConfig {
            buffer_size_per_task: 10,
            sampling_strategy: SamplingStrategy::Uniform,
            replay_batch_size: 4,
            ..Default::default()
        };
        let mut replay = MemoryReplay::new(config, Some(42));
        for task_id in ["task1", "task2", "task3"] {
            for i in 0..5 {
                let sample = ExperienceSample::new(
                    task_id.to_string(),
                    vec![i as f32],
                    vec![(i % 2) as f32],
                );
                replay.add_experience(sample);
            }
        }
        let samples = replay.sample_replay_batch().expect("replay sampling failed");
        assert!(samples.len() <= 4);
    }

    #[test]
    fn test_importance_update_minimum() {
        let mut sample = ExperienceSample::new("task1".to_string(), vec![1.0], vec![0.0]);
        // Even with 0 loss, importance should stay above minimum
        sample.update_importance(0.0);
        assert!(sample.importance >= 0.1);
    }

    #[test]
    fn test_importance_update_repeated() {
        let mut sample = ExperienceSample::new("task1".to_string(), vec![1.0], vec![0.0]);
        for _ in 0..10 {
            sample.update_importance(2.0);
        }
        // After many updates with high loss, importance should be substantial
        assert!(sample.importance > 0.1);
    }

    #[test]
    fn test_weighted_sampling_returns_samples() {
        let mut buffer = ExperienceBuffer::new(20, Some(42));
        for i in 0..10 {
            let mut sample = ExperienceSample::new("task1".to_string(), vec![i as f32], vec![0.0]);
            sample.importance = (i + 1) as f32;
            buffer.add_sample(sample);
        }
        let samples = buffer.sample_weighted(5);
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_buffer_stats_task_counts() {
        let mut buffer = ExperienceBuffer::new(100, Some(42));
        for i in 0..10 {
            let task = if i < 5 { "task_a" } else { "task_b" };
            let sample = ExperienceSample::new(task.to_string(), vec![i as f32], vec![0.0]);
            buffer.add_sample(sample);
        }
        let stats = buffer.get_stats();
        assert_eq!(stats.total_samples, 10);
        assert_eq!(stats.task_counts.len(), 2);
    }

    #[test]
    fn test_replay_batch_size_larger_than_buffer() {
        let config = MemoryReplayConfig {
            buffer_size_per_task: 3,
            sampling_strategy: SamplingStrategy::Random,
            replay_batch_size: 100, // Much larger than available
            ..Default::default()
        };
        let mut replay = MemoryReplay::new(config, Some(42));
        for i in 0..3 {
            let sample = ExperienceSample::new("task1".to_string(), vec![i as f32], vec![0.0]);
            replay.add_experience(sample);
        }
        let samples = replay.sample_replay_batch().expect("replay sampling failed");
        assert!(samples.len() <= 3);
    }

    // ── Diverse sampling really clusters ─────────────────────────────────────

    /// Six samples of one task in three well-separated 1-D clusters:
    /// `{0.0, 0.1, 0.2}`, `{10.0, 10.1}`, `{20.0}`.
    fn clustered_replay(strategy: SamplingStrategy, batch: usize) -> MemoryReplay {
        let config = MemoryReplayConfig {
            buffer_size_per_task: 16,
            sampling_strategy: strategy,
            replay_batch_size: batch,
            num_clusters: 3,
            ..Default::default()
        };
        let mut replay = MemoryReplay::new(config, Some(42));
        for x in [0.0f32, 0.1, 0.2, 10.0, 10.1, 20.0] {
            replay.add_experience(ExperienceSample::new("t".to_string(), vec![x], vec![0.0]));
        }
        replay
    }

    /// Cluster id of a 1-D input under the fixture's `{~0, ~10, ~20}` layout.
    fn cluster_of(x: f32) -> usize {
        if x < 5.0 {
            0
        } else if x < 15.0 {
            1
        } else {
            2
        }
    }

    #[test]
    fn test_diverse_sampling_covers_every_cluster() {
        // Regression: `sample_diverse` delegated to `sample_uniform`, which shuffles within a
        // task and therefore has no reason to touch all three clusters.
        let mut replay = clustered_replay(SamplingStrategy::Diverse, 3);
        let samples = replay.sample_replay_batch().expect("diverse sampling failed");
        assert_eq!(samples.len(), 3, "expected a full batch of 3");

        let mut clusters: Vec<usize> = samples.iter().map(|s| cluster_of(s.input[0])).collect();
        clusters.sort_unstable();
        clusters.dedup();
        assert_eq!(
            clusters.len(),
            3,
            "farthest-point selection must pick one sample per cluster, got inputs {:?}",
            samples.iter().map(|s| s.input[0]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_diverse_sampling_is_deterministic_for_a_fixed_buffer() {
        // k-center greedy is seeded from the centroid, not the RNG, so the same buffer must
        // always produce the same batch — uniform sampling could not promise this.
        let first: Vec<f32> = clustered_replay(SamplingStrategy::Diverse, 3)
            .sample_replay_batch()
            .expect("diverse sampling failed")
            .iter()
            .map(|s| s.input[0])
            .collect();
        let second: Vec<f32> = clustered_replay(SamplingStrategy::Diverse, 3)
            .sample_replay_batch()
            .expect("diverse sampling failed")
            .iter()
            .map(|s| s.input[0])
            .collect();
        assert_eq!(first, second);
    }

    #[test]
    fn test_diverse_sampling_differs_from_uniform_on_clustered_data() {
        let diverse: Vec<f32> = clustered_replay(SamplingStrategy::Diverse, 3)
            .sample_replay_batch()
            .expect("diverse sampling failed")
            .iter()
            .map(|s| s.input[0])
            .collect();
        let uniform: Vec<f32> = clustered_replay(SamplingStrategy::Uniform, 3)
            .sample_replay_batch()
            .expect("uniform sampling failed")
            .iter()
            .map(|s| s.input[0])
            .collect();
        // Under the old placeholder these two calls were literally the same code path.
        assert_ne!(
            diverse, uniform,
            "diverse and uniform sampling must not be the same algorithm"
        );
    }

    // ── Gradient-based sampling uses recorded gradient norms ─────────────────

    #[test]
    fn test_gradient_based_errors_without_recorded_norms() {
        // Regression: this used to silently return `sample_weighted`'s result.
        let mut replay = clustered_replay(SamplingStrategy::GradientBased, 2);
        let err = replay
            .sample_replay_batch()
            .expect_err("gradient-based replay without gradient norms must be an error");
        let message = err.to_string();
        assert!(
            message.contains("record_gradient_norm"),
            "the error should name the missing step, got: {message}"
        );
    }

    #[test]
    fn test_gradient_based_follows_gradient_norm_not_importance() {
        let config = MemoryReplayConfig {
            buffer_size_per_task: 8,
            sampling_strategy: SamplingStrategy::GradientBased,
            replay_batch_size: 1,
            ..Default::default()
        };
        let mut replay = MemoryReplay::new(config, Some(7));
        for x in [0.0f32, 1.0, 2.0] {
            let mut sample = ExperienceSample::new("t".to_string(), vec![x], vec![0.0]);
            // Make `importance` point the opposite way: the weighted sampler would strongly
            // prefer the first two samples.
            sample.importance = if x < 2.0 { 100.0 } else { 0.01 };
            replay.add_experience(sample);
        }
        // Only the third sample has any gradient signal.
        replay.record_gradient_norm("t", 0, 0.0).expect("record 0");
        replay.record_gradient_norm("t", 1, 0.0).expect("record 1");
        replay.record_gradient_norm("t", 2, 5.0).expect("record 2");

        let samples = replay.sample_replay_batch().expect("gradient sampling failed");
        assert_eq!(samples.len(), 1);
        assert_eq!(
            samples[0].input[0], 2.0,
            "the only sample with a non-zero gradient norm must be drawn, \
             regardless of the `importance` field"
        );
    }

    #[test]
    fn test_gradient_based_errors_when_all_norms_are_zero() {
        let config = MemoryReplayConfig {
            sampling_strategy: SamplingStrategy::GradientBased,
            replay_batch_size: 1,
            ..Default::default()
        };
        let mut replay = MemoryReplay::new(config, Some(1));
        replay.add_experience(ExperienceSample::new("t".to_string(), vec![0.0], vec![0.0]));
        replay.record_gradient_norm("t", 0, 0.0).expect("record");
        assert!(
            replay.sample_replay_batch().is_err(),
            "an all-zero gradient distribution is degenerate and must be reported"
        );
    }

    #[test]
    fn test_record_gradient_norm_validates_its_arguments() {
        let mut replay = clustered_replay(SamplingStrategy::GradientBased, 1);
        assert!(replay.record_gradient_norm("missing", 0, 1.0).is_err());
        assert!(replay.record_gradient_norm("t", 999, 1.0).is_err());
        assert!(replay.record_gradient_norm("t", 0, -1.0).is_err());
        assert!(replay.record_gradient_norm("t", 0, f32::NAN).is_err());
        assert!(replay.record_gradient_norm("t", 0, 2.5).is_ok());
    }

    #[test]
    fn test_gradient_based_draws_without_replacement() {
        let config = MemoryReplayConfig {
            sampling_strategy: SamplingStrategy::GradientBased,
            replay_batch_size: 3,
            ..Default::default()
        };
        let mut replay = MemoryReplay::new(config, Some(11));
        for x in [0.0f32, 1.0, 2.0] {
            replay.add_experience(ExperienceSample::new("t".to_string(), vec![x], vec![0.0]));
        }
        for idx in 0..3 {
            replay.record_gradient_norm("t", idx, 1.0 + idx as f32).expect("record");
        }
        let samples = replay.sample_replay_batch().expect("gradient sampling failed");
        assert_eq!(samples.len(), 3);
        let mut inputs: Vec<f32> = samples.iter().map(|s| s.input[0]).collect();
        inputs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(
            inputs,
            vec![0.0, 1.0, 2.0],
            "sampling without replacement must return each eligible sample once"
        );
    }

    #[test]
    fn test_squared_distance_zero_pads_shorter_vectors() {
        // 3² + 4² = 25 whether or not the second vector spells out its trailing zero.
        assert!((squared_distance(&[3.0, 4.0], &[0.0]) - 25.0).abs() < 1e-6);
        assert!((squared_distance(&[3.0, 4.0], &[0.0, 0.0]) - 25.0).abs() < 1e-6);
    }
}
