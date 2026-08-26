//! The engine's priority queue and test scheduler.
//!
//! Split out of `types.rs` in 0.2.1 to keep every file in this module under the
//! 2000-line limit.

use super::types::{
    DependencyTracker, QueueConfig, QueueStatistics, ScheduledTest, SchedulerMetrics,
    SchedulingConfig, SchedulingEvent, SchedulingEventDetails, SchedulingEventType,
};
use anyhow::Result;
use chrono::Utc;
use log::debug;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};

/// One entry in a [`PriorityQueue`].
#[derive(Debug)]
struct QueuedItem<T> {
    /// The queued value.
    item: T,
    /// Priority it was pushed at; higher pops first.
    priority: f32,
    /// When it entered the queue, used to measure real wait times.
    enqueued_at: Instant,
    /// Monotonic push counter, used only to break priority ties FIFO.
    sequence: u64,
}
/// Priority queue for scheduled work: highest priority pops first, ties break
/// in push order.
///
/// 0.2.1: this used to carry three `_`-prefixed fields that nothing read or
/// wrote, so every item pushed at it was silently discarded and the queue was
/// permanently, invisibly empty. It is now a real queue, and every number in
/// its [`QueueStatistics`] is measured from this queue's own traffic --
/// `average_wait_time` from the enqueue instants of the items actually
/// dequeued, `throughput` from those dequeues over the queue's lifetime.
/// Nothing is estimated.
#[derive(Debug)]
pub struct PriorityQueue<T> {
    /// Items currently queued.
    items: VecDeque<QueuedItem<T>>,
    /// Statistics measured from this queue's own traffic.
    stats: QueueStatistics,
    /// Queue configuration; `max_size == 0` means unbounded.
    config: QueueConfig,
    /// Next sequence number handed to a pushed item.
    next_sequence: u64,
    /// Instant of the first push, the origin for throughput.
    first_push: Option<Instant>,
    /// Summed wait time of every item dequeued so far.
    total_wait: Duration,
}
impl<T> PriorityQueue<T> {
    /// Build an unbounded queue.
    pub fn new() -> Self {
        Self::with_config(QueueConfig::default())
    }

    /// Build a queue honouring `config`. A `max_size` of 0 means unbounded.
    pub fn with_config(config: QueueConfig) -> Self {
        Self {
            items: VecDeque::new(),
            stats: QueueStatistics::default(),
            config,
            next_sequence: 0,
            first_push: None,
            total_wait: Duration::ZERO,
        }
    }

    /// Enqueue `item` at `priority`.
    ///
    /// # Errors
    ///
    /// Returns an error when the queue is already at its configured
    /// `max_size`; the item is *not* stored, so the caller learns about the
    /// rejection instead of losing the work silently.
    pub fn push(&mut self, item: T, priority: f32) -> Result<()> {
        if self.config.max_size > 0 && self.items.len() >= self.config.max_size {
            return Err(anyhow::anyhow!(
                "priority queue is full ({} items, max_size {})",
                self.items.len(),
                self.config.max_size
            ));
        }
        let now = Instant::now();
        self.first_push.get_or_insert(now);
        self.items.push_back(QueuedItem {
            item,
            priority,
            enqueued_at: now,
            sequence: self.next_sequence,
        });
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.stats.total_enqueued += 1;
        self.stats.current_size = self.items.len();
        self.stats.peak_size = self.stats.peak_size.max(self.items.len());
        Ok(())
    }

    /// Remove and return the highest-priority item, or `None` when empty.
    ///
    /// When `config.priority_enabled` is false the queue degrades to plain
    /// FIFO, which is what the flag actually means -- it is read here rather
    /// than merely stored.
    pub fn pop(&mut self) -> Option<T> {
        let index = if self.config.priority_enabled {
            let mut best: Option<usize> = None;
            for (index, candidate) in self.items.iter().enumerate() {
                let better = match best {
                    None => true,
                    Some(current) => {
                        let incumbent = &self.items[current];
                        candidate.priority > incumbent.priority
                            || (candidate.priority == incumbent.priority
                                && candidate.sequence < incumbent.sequence)
                    },
                };
                if better {
                    best = Some(index);
                }
            }
            best?
        } else {
            if self.items.is_empty() {
                return None;
            }
            0
        };
        let entry = self.items.remove(index)?;
        let waited = entry.enqueued_at.elapsed();
        self.total_wait += waited;
        self.stats.total_dequeued += 1;
        self.stats.current_size = self.items.len();
        self.stats.average_wait_time = self
            .total_wait
            .checked_div(u32::try_from(self.stats.total_dequeued).unwrap_or(u32::MAX))
            .unwrap_or(Duration::ZERO);
        if let Some(first_push) = self.first_push {
            let lifetime = first_push.elapsed().as_secs_f64();
            if lifetime > 0.0 {
                self.stats.throughput = self.stats.total_dequeued as f64 / lifetime;
            }
        }
        Some(entry.item)
    }

    /// Number of items currently queued.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the queue currently holds no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Statistics measured from this queue's traffic.
    pub fn statistics(&self) -> &QueueStatistics {
        &self.stats
    }

    /// The configuration this queue was built with.
    pub fn config(&self) -> &QueueConfig {
        &self.config
    }
}
/// Test scheduler for managing test execution order and timing.
///
/// 0.2.1: every method here was a no-op -- `schedule_test` discarded its
/// argument and returned `Ok(())`, `get_next_test` always returned `Ok(None)`
/// and `is_queue_empty` always returned `true`. The consequence was that
/// [`ParallelExecutionEngine::execute_parallel`](super::engine::ParallelExecutionEngine::execute_parallel) silently dropped every test
/// handed to it and reported success over an empty result set. The queue is
/// real now: tests are stored, ordered by the priority the engine computed for
/// them, and handed back one at a time.
pub struct TestScheduler {
    /// Scheduling configuration.
    config: Arc<RwLock<SchedulingConfig>>,
    /// Test queue, ordered by [`ScheduledTest::priority`].
    test_queue: Arc<Mutex<PriorityQueue<ScheduledTest>>>,
    /// Scheduling history, one entry per real queue transition.
    scheduling_history: Arc<Mutex<Vec<SchedulingEvent>>>,
    /// Scheduler performance metrics measured from real decisions.
    metrics: Arc<Mutex<SchedulerMetrics>>,
    /// Dependency tracker.
    dependency_tracker: Arc<DependencyTracker>,
}
impl TestScheduler {
    /// Build a scheduler from the caller's scheduling configuration.
    ///
    /// # Errors
    ///
    /// Infallible today; the `Result` matches the other component
    /// constructors so a future validating scheduler can reject bad configs.
    pub async fn new(config: crate::test_parallelization::SchedulingConfig) -> Result<Self> {
        // Carry across only the field with an exact counterpart, exactly as
        // `monitoring_config_from` does for the resource-monitoring config;
        // the rest keep this module's own defaults rather than being invented
        // from unrelated values.
        let local_config = SchedulingConfig {
            strategy: config.strategy.clone(),
            ..SchedulingConfig::default()
        };
        let queue_config = QueueConfig {
            max_size: local_config.queue_management.max_queue_size,
            priority_enabled: true,
        };
        Ok(Self {
            config: Arc::new(RwLock::new(local_config)),
            test_queue: Arc::new(Mutex::new(PriorityQueue::with_config(queue_config))),
            scheduling_history: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(SchedulerMetrics::default())),
            dependency_tracker: Arc::new(DependencyTracker::default()),
        })
    }

    /// The scheduling configuration in force.
    pub fn config(&self) -> SchedulingConfig {
        self.config.read().clone()
    }

    /// Dependency tracker shared with the engine.
    pub fn dependency_tracker(&self) -> &Arc<DependencyTracker> {
        &self.dependency_tracker
    }

    /// Every queue transition recorded so far, oldest first.
    pub fn history(&self) -> Vec<SchedulingEvent> {
        self.scheduling_history.lock().clone()
    }

    /// Metrics measured from this scheduler's own decisions.
    pub fn metrics(&self) -> SchedulerMetrics {
        self.metrics.lock().clone()
    }

    /// Statistics measured by the underlying queue.
    pub fn queue_statistics(&self) -> QueueStatistics {
        self.test_queue.lock().statistics().clone()
    }

    /// Record one real queue transition.
    fn record_event(
        &self,
        event_type: SchedulingEventType,
        test: &ScheduledTest,
        queue_position: Option<usize>,
        wait_time: Option<Duration>,
    ) {
        self.scheduling_history.lock().push(SchedulingEvent {
            timestamp: Utc::now(),
            event_type,
            test_id: test.metadata.resource_usage.test_id.clone(),
            details: SchedulingEventDetails {
                priority: test.priority,
                queue_position,
                // Nothing is allocated or assigned at queue-transition time,
                // so these stay absent rather than carrying a made-up id.
                resource_allocation: None,
                worker_id: None,
                wait_time,
                additional_details: HashMap::new(),
            },
            metadata: HashMap::new(),
        });
    }

    /// Charge one scheduling decision against the metrics, timed for real.
    fn record_decision(&self, elapsed: Duration) {
        let mut metrics = self.metrics.lock();
        let previous = metrics.decisions_made;
        metrics.decisions_made = previous.saturating_add(1);
        let total = metrics
            .average_decision_time
            .saturating_mul(u32::try_from(previous).unwrap_or(u32::MAX))
            + elapsed;
        metrics.average_decision_time = total
            .checked_div(u32::try_from(metrics.decisions_made).unwrap_or(u32::MAX))
            .unwrap_or(elapsed);
    }

    /// Queue `test` at the priority the engine computed for it.
    ///
    /// # Errors
    ///
    /// Returns an error when the queue is at its configured `max_queue_size`;
    /// the test is rejected loudly rather than dropped silently.
    pub async fn schedule_test(&self, test: ScheduledTest) -> Result<()> {
        let started = Instant::now();
        let priority = test.priority;
        let position = {
            let mut queue = self.test_queue.lock();
            queue.push(test.clone(), priority)?;
            queue.len()
        };
        self.record_event(SchedulingEventType::Queued, &test, Some(position), None);
        self.record_decision(started.elapsed());
        debug!(
            "Queued test {} at priority {priority} (queue depth {position})",
            test.metadata.resource_usage.test_id
        );
        Ok(())
    }

    /// Remove and return the highest-priority queued test, if any.
    ///
    /// # Errors
    ///
    /// Infallible today; the `Result` is kept so a future dependency-aware
    /// scheduler can report an unsatisfiable ordering.
    pub async fn get_next_test(&self) -> Result<Option<ScheduledTest>> {
        let started = Instant::now();
        let next = self.test_queue.lock().pop();
        let Some(test) = next else {
            return Ok(None);
        };
        let waited = (Utc::now() - test.scheduled_at).to_std().ok();
        self.record_event(SchedulingEventType::Scheduled, &test, None, waited);
        self.record_decision(started.elapsed());
        Ok(Some(test))
    }

    /// Put a previously dequeued test back in the queue.
    ///
    /// # Errors
    ///
    /// Returns an error when the queue is at its configured `max_queue_size`.
    pub async fn requeue_test(&self, test: ScheduledTest) -> Result<()> {
        let started = Instant::now();
        let priority = test.priority;
        let position = {
            let mut queue = self.test_queue.lock();
            queue.push(test.clone(), priority)?;
            queue.len()
        };
        self.record_event(
            SchedulingEventType::Rescheduled,
            &test,
            Some(position),
            None,
        );
        self.record_decision(started.elapsed());
        Ok(())
    }

    /// Whether no test is currently queued.
    pub async fn is_queue_empty(&self) -> bool {
        self.test_queue.lock().is_empty()
    }
}
