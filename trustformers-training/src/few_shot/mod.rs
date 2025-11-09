pub mod cross_task;
pub mod in_context;
pub mod meta_learning;
pub mod prompt_tuning;
pub mod task_adaptation;

pub use cross_task::{CrossTaskGeneralizer, GeneralizationConfig, TaskEmbedding};
pub use in_context::{ICLExample, InContextConfig, InContextLearner};
pub use meta_learning::{
    MAMLConfig, MAMLTrainer, MetaLearningAlgorithm, ReptileConfig, ReptileTrainer, TaskBatch,
};
pub use prompt_tuning::{PromptConfig, PromptTuner, SoftPrompt};
pub use task_adaptation::{AdaptationConfig, TaskAdapter, TaskDescriptor};

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Configuration for few-shot and zero-shot learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotConfig {
    /// Number of examples per class (K in K-shot)
    pub k_shot: usize,
    /// Method for few-shot learning
    pub method: FewShotMethod,
    /// In-context learning configuration
    pub in_context: Option<InContextConfig>,
    /// Prompt tuning configuration
    pub prompt_tuning: Option<PromptConfig>,
    /// Meta-learning configuration
    pub meta_learning: Option<MetaLearningConfig>,
    /// Task adaptation configuration
    pub task_adaptation: Option<AdaptationConfig>,
    /// Whether to use cross-task generalization
    pub enable_cross_task: bool,
}

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            k_shot: 5,
            method: FewShotMethod::InContext,
            in_context: Some(InContextConfig::default()),
            prompt_tuning: None,
            meta_learning: None,
            task_adaptation: None,
            enable_cross_task: false,
        }
    }
}

/// Methods for few-shot learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FewShotMethod {
    /// In-context learning (like GPT-3)
    InContext,
    /// Prompt tuning with soft prompts
    PromptTuning,
    /// Meta-learning (MAML, Reptile)
    MetaLearning,
    /// Task-specific adaptation
    TaskAdaptation,
    /// Combined approach
    Combined(Vec<FewShotMethod>),
}

/// Meta-learning configuration wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaLearningConfig {
    MAML(MAMLConfig),
    Reptile(ReptileConfig),
}

/// Few-shot learning example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotExample {
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    pub task_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Support set for few-shot learning
#[derive(Debug, Clone)]
pub struct SupportSet {
    pub examples: Vec<FewShotExample>,
    pub task_id: String,
    pub k_shot: usize,
    pub num_classes: Option<usize>,
}

impl SupportSet {
    pub fn new(task_id: String, k_shot: usize) -> Self {
        Self {
            examples: Vec::new(),
            task_id,
            k_shot,
            num_classes: None,
        }
    }

    pub fn add_example(&mut self, example: FewShotExample) -> Result<()> {
        if self.examples.len() >= self.k_shot * self.num_classes.unwrap_or(usize::MAX) {
            return Err(anyhow::anyhow!("Support set is full"));
        }
        self.examples.push(example);
        Ok(())
    }

    pub fn is_complete(&self) -> bool {
        if let Some(num_classes) = self.num_classes {
            self.examples.len() == self.k_shot * num_classes
        } else {
            false
        }
    }
}

/// Query set for evaluation
#[derive(Debug, Clone)]
pub struct QuerySet {
    pub examples: Vec<FewShotExample>,
    pub task_id: String,
}

/// Few-shot learning manager
pub struct FewShotLearningManager {
    config: FewShotConfig,
    support_sets: std::collections::HashMap<String, SupportSet>,
    query_sets: std::collections::HashMap<String, QuerySet>,
}

impl FewShotLearningManager {
    pub fn new(config: FewShotConfig) -> Self {
        Self {
            config,
            support_sets: std::collections::HashMap::new(),
            query_sets: std::collections::HashMap::new(),
        }
    }

    pub fn create_support_set(&mut self, task_id: String, num_classes: usize) -> Result<()> {
        let mut support_set = SupportSet::new(task_id.clone(), self.config.k_shot);
        support_set.num_classes = Some(num_classes);
        self.support_sets.insert(task_id, support_set);
        Ok(())
    }

    pub fn add_support_example(&mut self, task_id: &str, example: FewShotExample) -> Result<()> {
        let support_set = self
            .support_sets
            .get_mut(task_id)
            .ok_or_else(|| anyhow::anyhow!("Support set not found for task: {}", task_id))?;
        support_set.add_example(example)?;
        Ok(())
    }

    pub fn create_query_set(&mut self, task_id: String) -> Result<()> {
        let query_set = QuerySet {
            examples: Vec::new(),
            task_id: task_id.clone(),
        };
        self.query_sets.insert(task_id, query_set);
        Ok(())
    }

    pub fn add_query_example(&mut self, task_id: &str, example: FewShotExample) -> Result<()> {
        let query_set = self
            .query_sets
            .get_mut(task_id)
            .ok_or_else(|| anyhow::anyhow!("Query set not found for task: {}", task_id))?;
        query_set.examples.push(example);
        Ok(())
    }

    pub fn get_support_set(&self, task_id: &str) -> Option<&SupportSet> {
        self.support_sets.get(task_id)
    }

    pub fn get_query_set(&self, task_id: &str) -> Option<&QuerySet> {
        self.query_sets.get(task_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_support_set() {
        let mut support_set = SupportSet::new("task1".to_string(), 5);
        support_set.num_classes = Some(2);

        for i in 0..10 {
            let example = FewShotExample {
                input: vec![i as f32],
                output: vec![(i % 2) as f32],
                task_id: Some("task1".to_string()),
                metadata: None,
            };
            support_set.add_example(example).unwrap();
        }

        assert!(support_set.is_complete());
        assert_eq!(support_set.examples.len(), 10);
    }

    #[test]
    fn test_few_shot_manager() {
        let config = FewShotConfig::default();
        let mut manager = FewShotLearningManager::new(config);

        manager.create_support_set("task1".to_string(), 2).unwrap();
        manager.create_query_set("task1".to_string()).unwrap();

        let example = FewShotExample {
            input: vec![1.0, 2.0],
            output: vec![0.0],
            task_id: Some("task1".to_string()),
            metadata: None,
        };

        manager.add_support_example("task1", example.clone()).unwrap();
        manager.add_query_example("task1", example).unwrap();

        assert!(manager.get_support_set("task1").is_some());
        assert!(manager.get_query_set("task1").is_some());
    }
}
