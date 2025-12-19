// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! Task Scheduling and Resource Assignment
//!
//! This module handles intelligent task scheduling and processor assignment
//! for optimal performance and resource utilization.

use anyhow::Result;
use std::collections::VecDeque;

use super::{
    config::{LoadBalancerConfig, LoadBalancingStrategy},
    types::{ComputeTask, ProcessorResource, ProcessorType, TaskPriority},
};

/// Task scheduler interface
pub trait TaskScheduler {
    /// Assign a processor for a given task
    fn assign_processor(
        &self,
        task: &ComputeTask,
        resources: &[ProcessorResource],
    ) -> Option<(ProcessorType, usize)>;

    /// Get task priority score
    fn get_priority_score(&self, task: &ComputeTask) -> f32;

    /// Check if task is suitable for GPU execution
    fn is_gpu_suitable(&self, task: &ComputeTask) -> bool;

    /// Calculate processor efficiency for task
    fn calculate_processor_efficiency(
        &self,
        task: &ComputeTask,
        resource: &ProcessorResource,
    ) -> f32;
}

/// Default task scheduler implementation
pub struct DefaultTaskScheduler {
    config: LoadBalancerConfig,
    cpu_counter: usize,
    gpu_counter: usize,
}

impl DefaultTaskScheduler {
    /// Create a new task scheduler
    pub fn new(config: LoadBalancerConfig) -> Self {
        Self {
            config,
            cpu_counter: 0,
            gpu_counter: 0,
        }
    }

    /// Update round-robin counters
    pub fn update_counters(&mut self, assigned_type: ProcessorType) {
        match assigned_type {
            ProcessorType::CPU => {
                self.cpu_counter = (self.cpu_counter + 1) % self.config.cpu_pool_size
            },
            ProcessorType::GPU => self.gpu_counter += 1, // GPU counter can grow unbounded
        }
    }

    /// Get best CPU resource
    fn get_best_cpu(&self, resources: &[ProcessorResource], task: &ComputeTask) -> Option<usize> {
        resources
            .iter()
            .enumerate()
            .filter(|(_, r)| r.processor_type == ProcessorType::CPU && r.is_available())
            .max_by(|(_, a), (_, b)| {
                let score_a = self.calculate_processor_efficiency(task, a);
                let score_b = self.calculate_processor_efficiency(task, b);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
    }

    /// Get best GPU resource
    fn get_best_gpu(&self, resources: &[ProcessorResource], task: &ComputeTask) -> Option<usize> {
        resources
            .iter()
            .enumerate()
            .filter(|(_, r)| r.processor_type == ProcessorType::GPU && r.is_available())
            .max_by(|(_, a), (_, b)| {
                let score_a = self.calculate_processor_efficiency(task, a);
                let score_b = self.calculate_processor_efficiency(task, b);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
    }

    /// Apply round-robin assignment
    fn assign_round_robin(
        &mut self,
        resources: &[ProcessorResource],
    ) -> Option<(ProcessorType, usize)> {
        // Alternate between CPU and GPU
        if self.cpu_counter % 2 == 0 {
            if let Some(cpu_idx) = resources
                .iter()
                .enumerate()
                .find(|(_, r)| r.processor_type == ProcessorType::CPU && r.is_available())
                .map(|(idx, _)| idx)
            {
                self.cpu_counter += 1;
                return Some((ProcessorType::CPU, cpu_idx));
            }
        }

        // Try GPU
        if let Some(gpu_idx) = resources
            .iter()
            .enumerate()
            .find(|(_, r)| r.processor_type == ProcessorType::GPU && r.is_available())
            .map(|(idx, _)| idx)
        {
            self.gpu_counter += 1;
            return Some((ProcessorType::GPU, gpu_idx));
        }

        // Fallback to any available CPU
        resources
            .iter()
            .enumerate()
            .find(|(_, r)| r.processor_type == ProcessorType::CPU && r.is_available())
            .map(|(idx, _)| (ProcessorType::CPU, idx))
    }

    /// Apply least loaded assignment
    fn assign_least_loaded(
        &self,
        resources: &[ProcessorResource],
    ) -> Option<(ProcessorType, usize)> {
        resources
            .iter()
            .enumerate()
            .filter(|(_, r)| r.is_available())
            .min_by(|(_, a), (_, b)| {
                a.utilization.partial_cmp(&b.utilization).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, r)| (r.processor_type, idx))
    }
}

impl TaskScheduler for DefaultTaskScheduler {
    fn assign_processor(
        &self,
        task: &ComputeTask,
        resources: &[ProcessorResource],
    ) -> Option<(ProcessorType, usize)> {
        // Check preferred processor first
        if let Some(preferred) = task.preferred_processor {
            let candidates: Vec<_> = resources
                .iter()
                .enumerate()
                .filter(|(_, r)| r.processor_type == preferred && r.is_available())
                .collect();

            if !candidates.is_empty() {
                let best = candidates.into_iter().max_by(|(_, a), (_, b)| {
                    let score_a = self.calculate_processor_efficiency(task, a);
                    let score_b = self.calculate_processor_efficiency(task, b);
                    score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
                });

                if let Some((idx, _)) = best {
                    return Some((preferred, idx));
                }
            }
        }

        // Apply strategy-based assignment
        match self.config.strategy {
            LoadBalancingStrategy::RoundRobin => {
                // Note: This modifies state, but we're treating it as immutable for simplicity
                // In real implementation, this would need mutable access
                if task.id.len() % 2 == 0 {
                    if let Some(cpu_idx) = self.get_best_cpu(resources, task) {
                        return Some((ProcessorType::CPU, cpu_idx));
                    }
                }
                if let Some(gpu_idx) = self.get_best_gpu(resources, task) {
                    return Some((ProcessorType::GPU, gpu_idx));
                }
                self.get_best_cpu(resources, task).map(|idx| (ProcessorType::CPU, idx))
            },

            LoadBalancingStrategy::LeastLoaded => resources
                .iter()
                .enumerate()
                .filter(|(_, r)| r.is_available())
                .min_by(|(_, a), (_, b)| {
                    a.utilization.partial_cmp(&b.utilization).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, r)| (r.processor_type, idx)),

            LoadBalancingStrategy::Adaptive => {
                // Intelligent assignment based on task characteristics
                if self.is_gpu_suitable(task) {
                    if let Some(gpu_idx) = self.get_best_gpu(resources, task) {
                        return Some((ProcessorType::GPU, gpu_idx));
                    }
                }

                // Fallback to best CPU
                self.get_best_cpu(resources, task).map(|idx| (ProcessorType::CPU, idx))
            },

            LoadBalancingStrategy::PerformanceOptimized => {
                // Choose processor with highest efficiency for this task
                resources
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.is_available())
                    .max_by(|(_, a), (_, b)| {
                        let score_a = self.calculate_processor_efficiency(task, a);
                        let score_b = self.calculate_processor_efficiency(task, b);
                        score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, r)| (r.processor_type, idx))
            },

            LoadBalancingStrategy::PowerEfficient => {
                // Prefer processor with best power efficiency
                resources
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.is_available())
                    .max_by(|(_, a), (_, b)| {
                        a.power_efficiency_rating
                            .partial_cmp(&b.power_efficiency_rating)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, r)| (r.processor_type, idx))
            },

            LoadBalancingStrategy::MemoryOptimized => {
                // Prefer processor with most available memory
                resources
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.is_available() && r.available_memory >= task.memory_required)
                    .max_by(|(_, a), (_, b)| a.available_memory.cmp(&b.available_memory))
                    .map(|(idx, r)| (r.processor_type, idx))
            },

            LoadBalancingStrategy::LatencyOptimized => {
                // Prefer processor with lowest latency
                resources
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.is_available())
                    .min_by(|(_, a), (_, b)| {
                        a.performance
                            .latency_per_op
                            .partial_cmp(&b.performance.latency_per_op)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, r)| (r.processor_type, idx))
            },

            LoadBalancingStrategy::ThroughputOptimized => {
                // Prefer processor with highest throughput
                resources
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.is_available())
                    .max_by(|(_, a), (_, b)| {
                        a.performance
                            .ops_per_second
                            .partial_cmp(&b.performance.ops_per_second)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, r)| (r.processor_type, idx))
            },

            LoadBalancingStrategy::NumaAware => {
                // NUMA-aware assignment (simplified)
                if self.config.enable_numa_awareness {
                    // Prefer local NUMA nodes
                    resources
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| r.is_available())
                        .max_by(|(_, a), (_, b)| {
                            let score_a = self.calculate_processor_efficiency(task, a);
                            let score_b = self.calculate_processor_efficiency(task, b);
                            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, r)| (r.processor_type, idx))
                } else {
                    self.assign_least_loaded(resources)
                }
            },
        }
    }

    fn get_priority_score(&self, task: &ComputeTask) -> f32 {
        let base_score = match task.priority {
            TaskPriority::Critical => 4.0,
            TaskPriority::High => 3.0,
            TaskPriority::Normal => 2.0,
            TaskPriority::Low => 1.0,
        };

        // Adjust for deadline urgency
        let deadline_factor = if task.deadline.is_some() {
            1.5 // Boost priority for tasks with deadlines
        } else {
            1.0
        };

        // Adjust for task size
        let size_factor = if task.compute_operations > 100000 {
            1.2 // Boost priority for large tasks
        } else {
            1.0
        };

        base_score * deadline_factor * size_factor
    }

    fn is_gpu_suitable(&self, task: &ComputeTask) -> bool {
        task.is_gpu_suitable() && task.compute_operations >= self.config.min_gpu_task_size
    }

    fn calculate_processor_efficiency(
        &self,
        task: &ComputeTask,
        resource: &ProcessorResource,
    ) -> f32 {
        let base_efficiency = resource.efficiency_score();

        // Task-specific adjustments
        let task_fit =
            if resource.processor_type == ProcessorType::GPU && self.is_gpu_suitable(task) {
                1.5 // GPU bonus for suitable tasks
            } else if resource.processor_type == ProcessorType::CPU {
                1.0
            } else {
                0.7 // GPU penalty for unsuitable tasks
            };

        // Memory availability factor
        let memory_factor = if resource.available_memory >= task.memory_required {
            1.0
        } else {
            0.3 // Heavy penalty for insufficient memory
        };

        // Performance characteristics factor
        let perf_factor = match task.parallelizability {
            p if p > 0.7 && resource.processor_type == ProcessorType::GPU => 1.3,
            p if p < 0.3 && resource.processor_type == ProcessorType::CPU => 1.2,
            _ => 1.0,
        };

        base_efficiency * task_fit * memory_factor * perf_factor
    }
}

/// Task queue manager
pub struct TaskQueueManager {
    priority_queues: std::collections::BTreeMap<TaskPriority, VecDeque<ComputeTask>>,
    max_capacity: usize,
    total_tasks: usize,
}

impl TaskQueueManager {
    /// Create a new task queue manager
    pub fn new(max_capacity: usize) -> Self {
        Self {
            priority_queues: std::collections::BTreeMap::new(),
            max_capacity,
            total_tasks: 0,
        }
    }

    /// Add task to queue
    pub fn enqueue(&mut self, task: ComputeTask) -> Result<()> {
        if self.total_tasks >= self.max_capacity {
            return Err(anyhow::anyhow!("Task queue is full"));
        }

        let priority = task.priority.clone();
        self.priority_queues
            .entry(priority)
            .or_default()
            .push_back(task);

        self.total_tasks += 1;
        Ok(())
    }

    /// Get next task from queue (highest priority first)
    pub fn dequeue(&mut self) -> Option<ComputeTask> {
        // Iterate through priorities in reverse order (highest first)
        for (_, queue) in self.priority_queues.iter_mut().rev() {
            if let Some(task) = queue.pop_front() {
                self.total_tasks -= 1;
                return Some(task);
            }
        }
        None
    }

    /// Get queue size
    pub fn size(&self) -> usize {
        self.total_tasks
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.total_tasks == 0
    }

    /// Check if queue is full
    pub fn is_full(&self) -> bool {
        self.total_tasks >= self.max_capacity
    }
}
