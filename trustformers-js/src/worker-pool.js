/* global Worker, AbortController */

/**
 * TrustformeRS Web Worker Pool Manager
 *
 * Provides efficient parallel processing across multiple Web Workers
 * with automatic load balancing, task queuing, and resource management.
 *
 * Features:
 * - Automatic worker pool sizing based on CPU cores
 * - Task queue with priority support
 * - Load balancing across workers
 * - Automatic retry on worker failure
 * - Progress tracking and cancellation
 * - Memory-efficient worker reuse
 *
 * @module worker-pool
 */

/**
 * Worker pool configuration
 * @typedef {Object} WorkerPoolConfig
 * @property {number} [maxWorkers] - Maximum number of workers (default: navigator.hardwareConcurrency)
 * @property {number} [minWorkers=1] - Minimum number of workers
 * @property {number} [idleTimeout=60000] - Time before idle workers are terminated (ms)
 * @property {number} [taskTimeout=300000] - Maximum time for a task (ms)
 * @property {number} [maxRetries=3] - Maximum retries for failed tasks
 * @property {string} [workerScript] - Path to worker script
 * @property {boolean} [autoScale=true] - Automatically scale worker count
 */

/**
 * Task priority levels
 * @enum {number}
 */
export const TaskPriority = {
  LOW: 0,
  NORMAL: 1,
  HIGH: 2,
  CRITICAL: 3
};

/**
 * Task status
 * @enum {string}
 */
export const TaskStatus = {
  PENDING: 'pending',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled'
};

/**
 * Worker state
 * @typedef {Object} WorkerState
 * @property {Worker} worker - The Web Worker instance
 * @property {boolean} busy - Whether the worker is currently processing
 * @property {Task|null} currentTask - Currently executing task
 * @property {number} tasksCompleted - Number of completed tasks
 * @property {number} lastUsed - Timestamp of last use
 * @property {Map<string, Function>} pendingCallbacks - Callbacks for pending operations
 */

/**
 * Task descriptor
 * @typedef {Object} Task
 * @property {string} id - Unique task ID
 * @property {string} command - Command to execute
 * @property {Object} params - Command parameters
 * @property {number} priority - Task priority
 * @property {string} status - Current status
 * @property {number} retries - Number of retries attempted
 * @property {Function} resolve - Promise resolve callback
 * @property {Function} reject - Promise reject callback
 * @property {number} createdAt - Task creation timestamp
 * @property {number} [startedAt] - Task start timestamp
 * @property {number} [completedAt] - Task completion timestamp
 * @property {AbortController} abortController - For task cancellation
 */

/**
 * Web Worker Pool Manager
 */
export class WorkerPool {
  /**
   * Create a new worker pool
   * @param {WorkerPoolConfig} config - Pool configuration
   */
  constructor(config = {}) {
    this.config = {
      maxWorkers: typeof navigator !== 'undefined' ? (navigator.hardwareConcurrency || 4) : 4,
      minWorkers: 1,
      idleTimeout: 60000,
      taskTimeout: 300000,
      maxRetries: 3,
      workerScript: './worker.js',
      autoScale: true,
      ...config
    };

    /** @type {WorkerState[]} */
    this.workers = [];

    /** @type {Task[]} */
    this.taskQueue = [];

    /** @type {Map<string, Task>} */
    this.activeTasks = new Map();

    /** @type {Map<string, Task>} */
    this.completedTasks = new Map();

    this.nextTaskId = 0;
    this.initialized = false;
    this.shuttingDown = false;

    // Statistics
    this.stats = {
      tasksSubmitted: 0,
      tasksCompleted: 0,
      tasksFailed: 0,
      tasksCancelled: 0,
      totalExecutionTime: 0,
      averageExecutionTime: 0
    };

    // Event listeners
    this.eventListeners = new Map();
  }

  /**
   * Initialize the worker pool
   * @returns {Promise<void>}
   */
  async initialize() {
    if (this.initialized) return;

    // Create minimum number of workers
    for (let i = 0; i < this.config.minWorkers; i++) {
      await this.createWorker();
    }

    this.initialized = true;
    this.emit('initialized', { workerCount: this.workers.length });

    // Start idle worker cleanup if enabled
    if (this.config.idleTimeout > 0) {
      this.startIdleWorkerCleanup();
    }
  }

  /**
   * Create a new worker
   * @returns {Promise<WorkerState>}
   */
  async createWorker() {
    if (this.workers.length >= this.config.maxWorkers) {
      throw new Error(`Maximum worker count (${this.config.maxWorkers}) reached`);
    }

    const worker = new Worker(this.config.workerScript, { type: 'module' });

    const workerState = {
      worker,
      busy: false,
      currentTask: null,
      tasksCompleted: 0,
      lastUsed: Date.now(),
      pendingCallbacks: new Map()
    };

    // Set up message handler
    worker.onmessage = (event) => this.handleWorkerMessage(workerState, event);
    worker.onerror = (error) => this.handleWorkerError(workerState, error);

    this.workers.push(workerState);
    this.emit('workerCreated', { workerCount: this.workers.length });

    // Initialize the worker
    await this.sendToWorker(workerState, 'initialize', {});

    return workerState;
  }

  /**
   * Handle messages from workers
   * @param {WorkerState} workerState - Worker state
   * @param {MessageEvent} event - Message event
   */
  handleWorkerMessage(workerState, event) {
    const { id, type, ...data } = event.data;

    if (type === 'ready') {
      this.emit('workerReady', { worker: workerState });
      return;
    }

    if (type === 'error') {
      console.error('Worker error:', data.error);
      this.emit('workerError', { worker: workerState, error: data.error });
      return;
    }

    // Handle callback for specific request
    if (id && workerState.pendingCallbacks.has(id)) {
      const callback = workerState.pendingCallbacks.get(id);
      workerState.pendingCallbacks.delete(id);
      callback(data);
    }
  }

  /**
   * Handle worker errors
   * @param {WorkerState} workerState - Worker state
   * @param {ErrorEvent} error - Error event
   */
  handleWorkerError(workerState, error) {
    console.error('Worker error:', error);

    // Mark current task as failed
    if (workerState.currentTask) {
      this.handleTaskFailure(workerState.currentTask, error);
    }

    // Remove failed worker
    this.removeWorker(workerState);

    // Create replacement worker if needed
    if (this.workers.length < this.config.minWorkers && !this.shuttingDown) {
      this.createWorker().catch(console.error);
    }
  }

  /**
   * Send a message to a worker
   * @param {WorkerState} workerState - Worker state
   * @param {string} command - Command to execute
   * @param {Object} params - Command parameters
   * @returns {Promise<any>}
   */
  sendToWorker(workerState, command, params) {
    return new Promise((resolve, reject) => {
      const id = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      workerState.pendingCallbacks.set(id, (data) => {
        if (data.success) {
          resolve(data);
        } else {
          reject(new Error(data.error || 'Worker operation failed'));
        }
      });

      workerState.worker.postMessage({ id, command, ...params });

      // Set timeout
      setTimeout(() => {
        if (workerState.pendingCallbacks.has(id)) {
          workerState.pendingCallbacks.delete(id);
          reject(new Error('Worker operation timeout'));
        }
      }, this.config.taskTimeout);
    });
  }

  /**
   * Submit a task to the pool
   * @param {string} command - Command to execute
   * @param {Object} params - Command parameters
   * @param {number} [priority=TaskPriority.NORMAL] - Task priority
   * @returns {Promise<any>}
   */
  async submit(command, params = {}, priority = TaskPriority.NORMAL) {
    if (!this.initialized) {
      await this.initialize();
    }

    if (this.shuttingDown) {
      throw new Error('Worker pool is shutting down');
    }

    return new Promise((resolve, reject) => {
      const task = {
        id: `task_${this.nextTaskId++}`,
        command,
        params,
        priority,
        status: TaskStatus.PENDING,
        retries: 0,
        resolve,
        reject,
        createdAt: Date.now(),
        abortController: new AbortController()
      };

      this.taskQueue.push(task);
      this.stats.tasksSubmitted++;

      // Sort by priority
      this.taskQueue.sort((a, b) => b.priority - a.priority);

      this.emit('taskSubmitted', { task });

      // Try to execute immediately
      this.processTasks().catch(console.error);
    });
  }

  /**
   * Submit multiple tasks in batch
   * @param {Array<{command: string, params: Object, priority?: number}>} tasks - Tasks to submit
   * @returns {Promise<any[]>}
   */
  async submitBatch(tasks) {
    return Promise.all(
      tasks.map(({ command, params, priority }) =>
        this.submit(command, params, priority)
      )
    );
  }

  /**
   * Process queued tasks
   */
  async processTasks() {
    while (this.taskQueue.length > 0 && !this.shuttingDown) {
      // Find available worker
      let workerState = this.workers.find(w => !w.busy);

      // Create new worker if needed and allowed
      if (!workerState && this.config.autoScale && this.workers.length < this.config.maxWorkers) {
        try {
          workerState = await this.createWorker();
        } catch {
          // Max workers reached, wait for available worker
          break;
        }
      }

      if (!workerState) {
        // No available workers
        break;
      }

      // Get next task
      const task = this.taskQueue.shift();
      if (!task) break;

      // Execute task
      await this.executeTask(workerState, task);
    }
  }

  /**
   * Execute a task on a worker
   * @param {WorkerState} workerState - Worker to use
   * @param {Task} task - Task to execute
   */
  async executeTask(workerState, task) {
    workerState.busy = true;
    workerState.currentTask = task;
    task.status = TaskStatus.RUNNING;
    task.startedAt = Date.now();

    this.activeTasks.set(task.id, task);
    this.emit('taskStarted', { task, worker: workerState });

    try {
      // Send task to worker
      const result = await Promise.race([
        this.sendToWorker(workerState, task.command, task.params),
        new Promise((_, reject) => {
          task.abortController.signal.addEventListener('abort', () => {
            reject(new Error('Task cancelled'));
          });
        })
      ]);

      // Task completed successfully
      this.handleTaskSuccess(workerState, task, result);
    } catch (error) {
      // Task failed
      if (error.message === 'Task cancelled') {
        this.handleTaskCancellation(task);
      } else {
        this.handleTaskFailure(task, error);
      }
    } finally {
      workerState.busy = false;
      workerState.currentTask = null;
      workerState.lastUsed = Date.now();
      workerState.tasksCompleted++;

      // Process next task
      this.processTasks().catch(console.error);
    }
  }

  /**
   * Handle successful task completion
   * @param {WorkerState} workerState - Worker state
   * @param {Task} task - Completed task
   * @param {any} result - Task result
   */
  handleTaskSuccess(workerState, task, result) {
    task.status = TaskStatus.COMPLETED;
    task.completedAt = Date.now();

    const executionTime = task.completedAt - task.startedAt;
    this.stats.tasksCompleted++;
    this.stats.totalExecutionTime += executionTime;
    this.stats.averageExecutionTime = this.stats.totalExecutionTime / this.stats.tasksCompleted;

    this.activeTasks.delete(task.id);
    this.completedTasks.set(task.id, task);

    this.emit('taskCompleted', { task, result, executionTime });
    task.resolve(result);
  }

  /**
   * Handle task failure
   * @param {Task} task - Failed task
   * @param {Error} error - Error
   */
  handleTaskFailure(task, error) {
    task.retries++;

    // Retry if under limit
    if (task.retries < this.config.maxRetries) {
      task.status = TaskStatus.PENDING;
      this.taskQueue.unshift(task); // Add to front for immediate retry
      this.emit('taskRetry', { task, error, retryCount: task.retries });
    } else {
      // Max retries exceeded
      task.status = TaskStatus.FAILED;
      task.completedAt = Date.now();

      this.stats.tasksFailed++;
      this.activeTasks.delete(task.id);

      this.emit('taskFailed', { task, error });
      task.reject(error);
    }
  }

  /**
   * Handle task cancellation
   * @param {Task} task - Cancelled task
   */
  handleTaskCancellation(task) {
    task.status = TaskStatus.CANCELLED;
    task.completedAt = Date.now();

    this.stats.tasksCancelled++;
    this.activeTasks.delete(task.id);

    this.emit('taskCancelled', { task });
    task.reject(new Error('Task cancelled'));
  }

  /**
   * Cancel a task
   * @param {string} taskId - Task ID to cancel
   * @returns {boolean} - Whether task was cancelled
   */
  cancelTask(taskId) {
    // Check active tasks
    const activeTask = this.activeTasks.get(taskId);
    if (activeTask) {
      activeTask.abortController.abort();
      return true;
    }

    // Check queued tasks
    const queueIndex = this.taskQueue.findIndex(t => t.id === taskId);
    if (queueIndex >= 0) {
      const [task] = this.taskQueue.splice(queueIndex, 1);
      this.handleTaskCancellation(task);
      return true;
    }

    return false;
  }

  /**
   * Remove a worker from the pool
   * @param {WorkerState} workerState - Worker to remove
   */
  removeWorker(workerState) {
    const index = this.workers.indexOf(workerState);
    if (index >= 0) {
      workerState.worker.terminate();
      this.workers.splice(index, 1);
      this.emit('workerRemoved', { workerCount: this.workers.length });
    }
  }

  /**
   * Start idle worker cleanup
   */
  startIdleWorkerCleanup() {
    setInterval(() => {
      const now = Date.now();
      const idleWorkers = this.workers.filter(
        w => !w.busy &&
             now - w.lastUsed > this.config.idleTimeout &&
             this.workers.length > this.config.minWorkers
      );

      idleWorkers.forEach(w => this.removeWorker(w));
    }, this.config.idleTimeout / 2);
  }

  /**
   * Get pool statistics
   * @returns {Object} Statistics
   */
  getStats() {
    return {
      ...this.stats,
      workers: {
        total: this.workers.length,
        busy: this.workers.filter(w => w.busy).length,
        idle: this.workers.filter(w => !w.busy).length
      },
      tasks: {
        queued: this.taskQueue.length,
        active: this.activeTasks.size,
        completed: this.completedTasks.size
      }
    };
  }

  /**
   * Add event listener
   * @param {string} event - Event name
   * @param {Function} listener - Event listener
   */
  on(event, listener) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event).push(listener);
  }

  /**
   * Remove event listener
   * @param {string} event - Event name
   * @param {Function} listener - Event listener
   */
  off(event, listener) {
    if (this.eventListeners.has(event)) {
      const listeners = this.eventListeners.get(event);
      const index = listeners.indexOf(listener);
      if (index >= 0) {
        listeners.splice(index, 1);
      }
    }
  }

  /**
   * Emit event
   * @param {string} event - Event name
   * @param {any} data - Event data
   */
  emit(event, data) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error(`Error in event listener for ${event}:`, error);
        }
      });
    }
  }

  /**
   * Shutdown the worker pool
   * @param {boolean} [force=false] - Force immediate shutdown
   * @returns {Promise<void>}
   */
  async shutdown(force = false) {
    this.shuttingDown = true;
    this.emit('shuttingDown', { force });

    if (!force) {
      // Wait for active tasks to complete
      while (this.activeTasks.size > 0) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    } else {
      // Cancel all tasks
      for (const task of this.activeTasks.values()) {
        this.cancelTask(task.id);
      }
      this.taskQueue.forEach(task => this.handleTaskCancellation(task));
      this.taskQueue = [];
    }

    // Terminate all workers
    this.workers.forEach(w => w.worker.terminate());
    this.workers = [];

    this.initialized = false;
    this.emit('shutdown');
  }
}

/**
 * Parallel processing utilities
 */
export class ParallelProcessor {
  /**
   * Create a parallel processor
   * @param {WorkerPoolConfig} config - Pool configuration
   */
  constructor(config = {}) {
    this.pool = new WorkerPool(config);
  }

  /**
   * Initialize the processor
   * @returns {Promise<void>}
   */
  async initialize() {
    await this.pool.initialize();
  }

  /**
   * Map function over array in parallel
   * @param {Array} items - Items to process
   * @param {Function} fn - Function to apply
   * @param {Object} [options] - Processing options
   * @returns {Promise<Array>} Results
   */
  async map(items, fn, options = {}) {
    const {
      chunkSize = Math.ceil(items.length / this.pool.config.maxWorkers),
      priority = TaskPriority.NORMAL
    } = options;

    // Split items into chunks
    const chunks = [];
    for (let i = 0; i < items.length; i += chunkSize) {
      chunks.push(items.slice(i, i + chunkSize));
    }

    // Process chunks in parallel
    const results = await this.pool.submitBatch(
      chunks.map((chunk, index) => ({
        command: 'map',
        params: { chunk, fn: fn.toString(), index },
        priority
      }))
    );

    // Flatten results
    return results.flatMap(r => r.result);
  }

  /**
   * Reduce array in parallel
   * @param {Array} _items - Items to reduce
   * @param {Function} _fn - Reduction function
   * @param {any} _initialValue - Initial value
   * @returns {Promise<any>} Result
   */
  async reduce(_items, _fn, _initialValue) {
    // TODO: Implement parallel reduce
    throw new Error('Not implemented yet');
  }

  /**
   * Filter array in parallel
   * @param {Array} _items - Items to filter
   * @param {Function} _predicate - Filter predicate
   * @returns {Promise<Array>} Filtered items
   */
  async filter(_items, _predicate) {
    // TODO: Implement parallel filter
    throw new Error('Not implemented yet');
  }

  /**
   * Shutdown the processor
   * @returns {Promise<void>}
   */
  async shutdown() {
    await this.pool.shutdown();
  }
}

/**
 * Create a worker pool
 * @param {WorkerPoolConfig} config - Pool configuration
 * @returns {WorkerPool}
 */
export function createWorkerPool(config) {
  return new WorkerPool(config);
}

/**
 * Create a parallel processor
 * @param {WorkerPoolConfig} config - Pool configuration
 * @returns {ParallelProcessor}
 */
export function createParallelProcessor(config) {
  return new ParallelProcessor(config);
}

export default {
  WorkerPool,
  ParallelProcessor,
  createWorkerPool,
  createParallelProcessor,
  TaskPriority,
  TaskStatus
};
