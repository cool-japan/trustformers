//! # Retry Scheduler for Failed Notifications
//!
//! Schedules and manages retry attempts for failed notification deliveries.

use super::delivery::{RetryItem, RetryScheduler, RetryStats};
use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use std::{collections::VecDeque, sync::Arc};

impl RetryScheduler {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            retry_queue: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(RetryStats::default()),
        })
    }

    pub async fn schedule_retry(&self, retry_item: RetryItem) -> Result<()> {
        let mut queue = self.retry_queue.lock();
        queue.push_back(retry_item);
        Ok(())
    }

    pub async fn get_due_retries(&self) -> Result<Vec<RetryItem>> {
        let mut queue = self.retry_queue.lock();
        let now = Utc::now();
        let mut due_retries = Vec::new();

        // Extract due retries
        let mut remaining = VecDeque::new();
        while let Some(item) = queue.pop_front() {
            if item.retry_time <= now {
                due_retries.push(item);
            } else {
                remaining.push_back(item);
            }
        }

        // Put back non-due items
        *queue = remaining;

        Ok(due_retries)
    }
}
