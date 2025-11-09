//! Thread pool management for parallel inference with work-stealing scheduler

use crate::error::{TrustformersError, TrustformersResult};
use crossbeam_deque::{Injector, Steal, Stealer, Worker as DequeWorker};
use crossbeam_utils::Backoff;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

/// Configuration for the thread pool
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads
    pub num_threads: usize,
    /// Maximum number of tasks in the queue
    pub max_queue_size: usize,
    /// Stack size for worker threads in bytes
    pub stack_size: Option<usize>,
    /// Thread priority (0-100, where 100 is highest)
    pub priority: u8,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            max_queue_size: 1000,
            stack_size: None,
            priority: 50,
        }
    }
}

/// Task that can be executed by the thread pool
pub type Task = Box<dyn FnOnce() + Send + 'static>;

/// Work-stealing thread pool for parallel inference
pub struct ThreadPool {
    workers: Vec<Worker>,
    global_queue: Arc<Injector<Task>>,
    stealers: Vec<Stealer<Task>>,
    shutdown: Arc<AtomicBool>,
    thread_count: Arc<AtomicUsize>,
}

struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl ThreadPool {
    /// Create a new thread pool with the given configuration
    pub fn new(config: ThreadPoolConfig) -> TrustformersResult<Self> {
        let global_queue = Arc::new(Injector::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_count = Arc::new(AtomicUsize::new(config.num_threads));

        let mut workers = Vec::with_capacity(config.num_threads);
        let mut stealers = Vec::with_capacity(config.num_threads);

        // Create local worker queues
        let local_queues: Vec<DequeWorker<Task>> =
            (0..config.num_threads).map(|_| DequeWorker::new_fifo()).collect();

        // Extract stealers for work-stealing
        for worker_queue in &local_queues {
            stealers.push(worker_queue.stealer());
        }

        for (id, local_queue) in local_queues.into_iter().enumerate() {
            let global_queue = Arc::clone(&global_queue);
            let shutdown_flag = Arc::clone(&shutdown);
            let thread_count_ref = Arc::clone(&thread_count);
            let stealers_clone = stealers.clone();

            let mut builder = thread::Builder::new().name(format!("trustformers-worker-{}", id));

            if let Some(stack_size) = config.stack_size {
                builder = builder.stack_size(stack_size);
            }

            let thread = builder
                .spawn(move || {
                    let backoff = Backoff::new();

                    while !shutdown_flag.load(Ordering::Relaxed) {
                        // Try to find a task in the following order:
                        // 1. Local queue (LIFO for better cache locality)
                        // 2. Global queue (FIFO)
                        // 3. Steal from other workers (FIFO)

                        if let Some(task) = local_queue.pop() {
                            task();
                            backoff.reset();
                            continue;
                        }

                        match global_queue.steal_batch_and_pop(&local_queue) {
                            Steal::Success(task) => {
                                task();
                                backoff.reset();
                                continue;
                            },
                            Steal::Empty => {},
                            Steal::Retry => {},
                        }

                        // Try to steal from other workers
                        let mut found_work = false;
                        for (stealer_id, stealer) in stealers_clone.iter().enumerate() {
                            if stealer_id == id {
                                continue; // Don't steal from self
                            }

                            match stealer.steal_batch_and_pop(&local_queue) {
                                Steal::Success(task) => {
                                    task();
                                    found_work = true;
                                    backoff.reset();
                                    break;
                                },
                                Steal::Empty => {},
                                Steal::Retry => {},
                            }
                        }

                        if !found_work {
                            backoff.snooze();
                        }
                    }

                    thread_count_ref.fetch_sub(1, Ordering::Relaxed);
                })
                .map_err(|_| TrustformersError::RuntimeError)?;

            workers.push(Worker {
                id,
                thread: Some(thread),
            });
        }

        Ok(ThreadPool {
            workers,
            global_queue,
            stealers,
            shutdown,
            thread_count,
        })
    }

    /// Execute a task on the thread pool
    pub fn execute<F>(&self, f: F) -> TrustformersResult<()>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(TrustformersError::RuntimeError);
        }

        let task = Box::new(f);
        self.global_queue.push(task);

        Ok(())
    }

    /// Get the number of worker threads
    pub fn thread_count(&self) -> usize {
        self.workers.len()
    }

    /// Get the number of active worker threads
    pub fn active_thread_count(&self) -> usize {
        self.thread_count.load(Ordering::Relaxed)
    }

    /// Check if the thread pool is shutting down
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for all workers to finish
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

/// Global thread pool instance
static mut GLOBAL_THREAD_POOL: Option<Arc<ThreadPool>> = None;
static INIT_THREAD_POOL: std::sync::Once = std::sync::Once::new();

/// Initialize the global thread pool
pub fn init_global_thread_pool(config: ThreadPoolConfig) -> TrustformersResult<()> {
    INIT_THREAD_POOL.call_once(|| {
        let pool = ThreadPool::new(config).expect("Failed to create thread pool");
        unsafe {
            GLOBAL_THREAD_POOL = Some(Arc::new(pool));
        }
    });
    Ok(())
}

/// Get a reference to the global thread pool
pub fn global_thread_pool() -> Option<Arc<ThreadPool>> {
    unsafe { GLOBAL_THREAD_POOL.clone() }
}

/// Execute a task on the global thread pool
pub fn execute_global<F>(f: F) -> TrustformersResult<()>
where
    F: FnOnce() + Send + 'static,
{
    match global_thread_pool() {
        Some(pool) => pool.execute(f),
        None => {
            // Fallback: execute immediately
            f();
            Ok(())
        },
    }
}

/// C API functions for thread pool management
use std::os::raw::{c_int, c_uint};

/// Initialize the global thread pool with the specified number of threads
#[no_mangle]
pub extern "C" fn trustformers_init_thread_pool(num_threads: c_uint) -> c_int {
    let config = ThreadPoolConfig {
        num_threads: num_threads as usize,
        ..Default::default()
    };

    match init_global_thread_pool(config) {
        Ok(()) => TrustformersError::Success as c_int,
        Err(e) => e as c_int,
    }
}

/// Get the number of worker threads in the global thread pool
#[no_mangle]
pub extern "C" fn trustformers_get_thread_count() -> c_uint {
    match global_thread_pool() {
        Some(pool) => pool.thread_count() as c_uint,
        None => 0,
    }
}

/// Check if the global thread pool is available
#[no_mangle]
pub extern "C" fn trustformers_thread_pool_available() -> c_int {
    match global_thread_pool() {
        Some(_) => 1,
        None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn test_thread_pool_creation() {
        let config = ThreadPoolConfig {
            num_threads: 2,
            ..Default::default()
        };

        let pool = ThreadPool::new(config).unwrap();
        assert_eq!(pool.thread_count(), 2);
        assert!(!pool.is_shutdown());
    }

    #[test]
    fn test_task_execution() {
        let config = ThreadPoolConfig {
            num_threads: 2,
            ..Default::default()
        };

        let pool = ThreadPool::new(config).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..10 {
            let counter = Arc::clone(&counter);
            pool.execute(move || {
                counter.fetch_add(1, Ordering::Relaxed);
            })
            .unwrap();
        }

        // Wait a bit for tasks to complete
        thread::sleep(Duration::from_millis(100));

        assert_eq!(counter.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_global_thread_pool() {
        let config = ThreadPoolConfig {
            num_threads: 1,
            ..Default::default()
        };

        init_global_thread_pool(config).unwrap();

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        execute_global(move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();

        thread::sleep(Duration::from_millis(50));

        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
}
