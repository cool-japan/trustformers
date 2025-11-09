pub mod catastrophic_prevention;
pub mod ewc;
pub mod memory_replay;
pub mod progressive_networks;
pub mod task_boundary;

pub use catastrophic_prevention::{CatastrophicPreventionStrategy, RegularizationMethod};
pub use ewc::{EWCConfig, EWCTrainer, FisherInformation};
pub use memory_replay::{ExperienceBuffer, MemoryReplay, MemoryReplayConfig};
pub use progressive_networks::{ProgressiveConfig, ProgressiveNetwork, TaskModule};
pub use task_boundary::{BoundaryDetectionConfig, TaskBoundaryDetector, TaskTransition};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for continual learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualLearningConfig {
    /// Method for preventing catastrophic forgetting
    pub prevention_method: CatastrophicPreventionStrategy,
    /// Task boundary detection configuration
    pub boundary_detection: BoundaryDetectionConfig,
    /// Memory replay configuration
    pub memory_replay: Option<MemoryReplayConfig>,
    /// EWC configuration
    pub ewc: Option<EWCConfig>,
    /// Progressive networks configuration
    pub progressive: Option<ProgressiveConfig>,
    /// Maximum number of tasks to remember
    pub max_tasks: usize,
    /// Whether to use online or offline learning
    pub online_learning: bool,
}

impl Default for ContinualLearningConfig {
    fn default() -> Self {
        Self {
            prevention_method: CatastrophicPreventionStrategy::EWC,
            boundary_detection: BoundaryDetectionConfig::default(),
            memory_replay: None,
            ewc: Some(EWCConfig::default()),
            progressive: None,
            max_tasks: 10,
            online_learning: true,
        }
    }
}

/// Task information for continual learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    pub task_id: String,
    pub name: String,
    pub description: Option<String>,
    pub data_size: usize,
    pub num_classes: Option<usize>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Continual learning manager
pub struct ContinualLearningManager {
    config: ContinualLearningConfig,
    tasks: Vec<TaskInfo>,
    current_task: Option<String>,
    task_transitions: Vec<TaskTransition>,
    #[allow(dead_code)]
    prevention_strategies: HashMap<String, Box<dyn RegularizationMethod>>,
}

impl ContinualLearningManager {
    pub fn new(config: ContinualLearningConfig) -> Self {
        Self {
            config,
            tasks: Vec::new(),
            current_task: None,
            task_transitions: Vec::new(),
            prevention_strategies: HashMap::new(),
        }
    }

    pub fn add_task(&mut self, task: TaskInfo) -> anyhow::Result<()> {
        if self.tasks.len() >= self.config.max_tasks {
            return Err(anyhow::anyhow!("Maximum number of tasks reached"));
        }

        self.tasks.push(task);
        Ok(())
    }

    pub fn set_current_task(&mut self, task_id: String) -> anyhow::Result<()> {
        if !self.tasks.iter().any(|t| t.task_id == task_id) {
            return Err(anyhow::anyhow!("Task not found: {}", task_id));
        }

        if let Some(prev_task) = &self.current_task {
            let transition = TaskTransition {
                from_task: prev_task.clone(),
                to_task: task_id.clone(),
                timestamp: chrono::Utc::now(),
                boundary_score: 1.0, // This would be computed by boundary detector
            };
            self.task_transitions.push(transition);
        }

        self.current_task = Some(task_id);
        Ok(())
    }

    pub fn get_current_task(&self) -> Option<&TaskInfo> {
        self.current_task
            .as_ref()
            .and_then(|id| self.tasks.iter().find(|t| &t.task_id == id))
    }

    pub fn get_task_count(&self) -> usize {
        self.tasks.len()
    }

    pub fn get_task_transitions(&self) -> &[TaskTransition] {
        &self.task_transitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continual_learning_manager() {
        let config = ContinualLearningConfig::default();
        let mut manager = ContinualLearningManager::new(config);

        let task1 = TaskInfo {
            task_id: "task1".to_string(),
            name: "Classification Task 1".to_string(),
            description: Some("First classification task".to_string()),
            data_size: 1000,
            num_classes: Some(10),
            created_at: chrono::Utc::now(),
        };

        manager.add_task(task1).unwrap();
        assert_eq!(manager.get_task_count(), 1);

        manager.set_current_task("task1".to_string()).unwrap();
        assert!(manager.get_current_task().is_some());
    }

    #[test]
    fn test_max_tasks_limit() {
        let mut config = ContinualLearningConfig::default();
        config.max_tasks = 2;
        let mut manager = ContinualLearningManager::new(config);

        for i in 0..3 {
            let task = TaskInfo {
                task_id: format!("task{}", i),
                name: format!("Task {}", i),
                description: None,
                data_size: 100,
                num_classes: Some(5),
                created_at: chrono::Utc::now(),
            };

            if i < 2 {
                assert!(manager.add_task(task).is_ok());
            } else {
                assert!(manager.add_task(task).is_err());
            }
        }
    }
}
