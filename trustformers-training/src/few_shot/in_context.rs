use anyhow::Result;
use scirs2_core::ndarray::Array1; // SciRS2 Integration Policy
use scirs2_core::random::*; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for in-context learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InContextConfig {
    /// Maximum context length in tokens
    pub max_context_length: usize,
    /// Strategy for example selection
    pub selection_strategy: SelectionStrategy,
    /// Whether to use example ordering optimization
    pub optimize_ordering: bool,
    /// Temperature for similarity computations
    pub temperature: f32,
    /// Number of demonstrations to use
    pub num_demonstrations: usize,
    /// Whether to include task instructions
    pub include_instructions: bool,
}

impl Default for InContextConfig {
    fn default() -> Self {
        Self {
            max_context_length: 2048,
            selection_strategy: SelectionStrategy::Similarity,
            optimize_ordering: true,
            temperature: 1.0,
            num_demonstrations: 5,
            include_instructions: true,
        }
    }
}

/// Strategies for selecting in-context examples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Random selection
    Random,
    /// Select most similar examples
    Similarity,
    /// Select diverse examples
    Diverse,
    /// Select based on uncertainty
    Uncertainty,
    /// Learned selection policy
    Learned,
}

/// In-context learning example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ICLExample {
    /// Input text or tokens
    pub input: String,
    /// Output text or tokens
    pub output: String,
    /// Optional explanation
    pub explanation: Option<String>,
    /// Embedding representation
    pub embedding: Option<Array1<f32>>,
    /// Confidence score
    pub confidence: f32,
}

impl ICLExample {
    pub fn new(input: String, output: String) -> Self {
        Self {
            input,
            output,
            explanation: None,
            embedding: None,
            confidence: 1.0,
        }
    }

    /// Compute similarity to another example
    pub fn similarity(&self, other: &ICLExample) -> f32 {
        if let (Some(emb1), Some(emb2)) = (&self.embedding, &other.embedding) {
            // Cosine similarity
            let dot_product = emb1.dot(emb2);
            let norm1 = emb1.dot(emb1).sqrt();
            let norm2 = emb2.dot(emb2).sqrt();
            dot_product / (norm1 * norm2 + 1e-8)
        } else {
            0.0
        }
    }
}

/// In-context learning context manager
#[derive(Debug)]
pub struct ICLContext {
    examples: Vec<ICLExample>,
    max_length: usize,
    current_length: usize,
}

impl ICLContext {
    pub fn new(max_length: usize) -> Self {
        Self {
            examples: Vec::new(),
            max_length,
            current_length: 0,
        }
    }

    /// Add example to context
    pub fn add_example(&mut self, example: ICLExample) -> Result<()> {
        let example_length = example.input.len() + example.output.len();
        if self.current_length + example_length > self.max_length {
            return Err(anyhow::anyhow!("Context length exceeded"));
        }

        self.current_length += example_length;
        self.examples.push(example);
        Ok(())
    }

    /// Format context for model input
    pub fn format_context(&self, query: &str, include_instruction: bool) -> String {
        let mut context = String::new();

        if include_instruction {
            context.push_str("Given the following examples, complete the task:\n\n");
        }

        for (i, example) in self.examples.iter().enumerate() {
            context.push_str(&format!("Example {}:\n", i + 1));
            context.push_str(&format!("Input: {}\n", example.input));
            context.push_str(&format!("Output: {}\n", example.output));
            if let Some(explanation) = &example.explanation {
                context.push_str(&format!("Explanation: {}\n", explanation));
            }
            context.push('\n');
        }

        context.push_str("Now complete:\n");
        context.push_str(&format!("Input: {}\n", query));
        context.push_str("Output: ");

        context
    }

    /// Get total context length
    pub fn get_length(&self) -> usize {
        self.current_length
    }

    /// Clear context
    pub fn clear(&mut self) {
        self.examples.clear();
        self.current_length = 0;
    }
}

/// In-context learner
pub struct InContextLearner {
    config: InContextConfig,
    example_bank: HashMap<String, Vec<ICLExample>>,
    embedder: Option<Box<dyn Fn(&str) -> Array1<f32>>>,
}

impl InContextLearner {
    pub fn new(config: InContextConfig) -> Self {
        Self {
            config,
            example_bank: HashMap::new(),
            embedder: None,
        }
    }

    /// Set embedding function
    pub fn set_embedder<F>(&mut self, embedder: F)
    where
        F: Fn(&str) -> Array1<f32> + 'static,
    {
        self.embedder = Some(Box::new(embedder));
    }

    /// Add examples for a task
    pub fn add_task_examples(&mut self, task_id: String, examples: Vec<ICLExample>) {
        self.example_bank.insert(task_id, examples);
    }

    /// Select examples for in-context learning
    pub fn select_examples(
        &self,
        task_id: &str,
        query: &str,
        num_examples: usize,
    ) -> Result<Vec<ICLExample>> {
        let examples = self
            .example_bank
            .get(task_id)
            .ok_or_else(|| anyhow::anyhow!("Task not found: {}", task_id))?;

        let selected = match self.config.selection_strategy {
            SelectionStrategy::Random => self.select_random(examples, num_examples),
            SelectionStrategy::Similarity => self.select_similar(examples, query, num_examples)?,
            SelectionStrategy::Diverse => self.select_diverse(examples, num_examples),
            SelectionStrategy::Uncertainty => self.select_uncertain(examples, num_examples),
            SelectionStrategy::Learned => self.select_learned(examples, query, num_examples)?,
        };

        Ok(selected)
    }

    /// Random selection
    fn select_random(&self, examples: &[ICLExample], num: usize) -> Vec<ICLExample> {
        use rand::seq::SliceRandom;
        let mut rng = thread_rng();
        let mut indices: Vec<_> = (0..examples.len()).collect();
        indices.shuffle(&mut rng);

        indices
            .into_iter()
            .take(num.min(examples.len()))
            .map(|i| examples[i].clone())
            .collect()
    }

    /// Similarity-based selection
    fn select_similar(
        &self,
        examples: &[ICLExample],
        query: &str,
        num: usize,
    ) -> Result<Vec<ICLExample>> {
        if self.embedder.is_none() {
            return Ok(self.select_random(examples, num));
        }

        let embedder = self.embedder.as_ref().unwrap();
        let query_embedding = embedder(query);

        let mut similarities: Vec<(usize, f32)> = examples
            .iter()
            .enumerate()
            .filter_map(|(i, ex)| {
                if let Some(emb) = &ex.embedding {
                    let sim = self.cosine_similarity(&query_embedding, emb);
                    Some((i, sim))
                } else {
                    None
                }
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(similarities.into_iter().take(num).map(|(i, _)| examples[i].clone()).collect())
    }

    /// Diverse selection using greedy algorithm
    fn select_diverse(&self, examples: &[ICLExample], num: usize) -> Vec<ICLExample> {
        if examples.is_empty() || num == 0 {
            return Vec::new();
        }

        let mut selected = vec![0]; // Start with first example
        let mut remaining: Vec<usize> = (1..examples.len()).collect();

        while selected.len() < num && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_min_sim = f32::NEG_INFINITY;

            for (i, &candidate_idx) in remaining.iter().enumerate() {
                let candidate = &examples[candidate_idx];

                // Find minimum similarity to already selected examples
                let min_sim = selected
                    .iter()
                    .map(|&sel_idx| examples[sel_idx].similarity(candidate))
                    .fold(f32::INFINITY, f32::min);

                if min_sim > best_min_sim {
                    best_min_sim = min_sim;
                    best_idx = i;
                }
            }

            selected.push(remaining[best_idx]);
            remaining.remove(best_idx);
        }

        selected.into_iter().map(|i| examples[i].clone()).collect()
    }

    /// Uncertainty-based selection
    fn select_uncertain(&self, examples: &[ICLExample], num: usize) -> Vec<ICLExample> {
        let mut sorted_examples = examples.to_vec();
        sorted_examples.sort_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap());

        sorted_examples.into_iter().take(num).collect()
    }

    /// Learned selection (placeholder)
    fn select_learned(
        &self,
        examples: &[ICLExample],
        _query: &str,
        num: usize,
    ) -> Result<Vec<ICLExample>> {
        // In practice, this would use a learned policy
        Ok(self.select_random(examples, num))
    }

    /// Compute cosine similarity
    fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();
        dot_product / (norm_a * norm_b + 1e-8)
    }

    /// Create formatted context for a query
    pub fn create_context(&self, task_id: &str, query: &str) -> Result<String> {
        let examples = self.select_examples(task_id, query, self.config.num_demonstrations)?;

        let mut context = ICLContext::new(self.config.max_context_length);

        for example in examples {
            context.add_example(example)?;
        }

        Ok(context.format_context(query, self.config.include_instructions))
    }

    /// Optimize example ordering
    pub fn optimize_ordering(&self, examples: &mut [ICLExample]) {
        if !self.config.optimize_ordering {
            return;
        }

        // Simple heuristic: order by confidence
        examples.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_icl_example() {
        let mut ex1 = ICLExample::new("What is 2+2?".to_string(), "4".to_string());
        let mut ex2 = ICLExample::new("What is 3+3?".to_string(), "6".to_string());

        ex1.embedding = Some(Array1::from_vec(vec![1.0, 0.0]));
        ex2.embedding = Some(Array1::from_vec(vec![0.0, 1.0]));

        let similarity = ex1.similarity(&ex2);
        assert_abs_diff_eq!(similarity, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_icl_context() {
        let mut context = ICLContext::new(1000);

        let ex1 = ICLExample::new("What is 2+2?".to_string(), "4".to_string());
        let ex2 = ICLExample::new("What is 3+3?".to_string(), "6".to_string());

        context.add_example(ex1).unwrap();
        context.add_example(ex2).unwrap();

        let formatted = context.format_context("What is 4+4?", true);
        assert!(formatted.contains("Example 1:"));
        assert!(formatted.contains("What is 2+2?"));
        assert!(formatted.contains("What is 4+4?"));
    }

    #[test]
    fn test_in_context_learner() {
        let config = InContextConfig::default();
        let mut learner = InContextLearner::new(config);

        let examples = vec![
            ICLExample::new("What is 2+2?".to_string(), "4".to_string()),
            ICLExample::new("What is 3+3?".to_string(), "6".to_string()),
            ICLExample::new("What is 4+4?".to_string(), "8".to_string()),
        ];

        learner.add_task_examples("math".to_string(), examples);

        let selected = learner.select_examples("math", "What is 5+5?", 2).unwrap();
        assert_eq!(selected.len(), 2);

        let context = learner.create_context("math", "What is 5+5?").unwrap();
        assert!(context.contains("Input: What is 5+5?"));
    }
}
