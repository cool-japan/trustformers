//! Database *slot* allocation for parallel test execution.
//!
//! ## What this module is, and what it deliberately is not
//!
//! This is a counting semaphore over a finite pool of named slots, plus the
//! endpoint URL an operator configured. It hands a test permission to use up to
//! N database connections and tracks who holds what; it **never opens a
//! connection, never speaks a database protocol, and never verifies that the
//! configured endpoint exists**. This crate has no database client at all.
//!
//! 0.2.1: the types here used to say otherwise. A `DatabaseConnection` carried a
//! `connection_url` synthesised as `postgresql://localhost:5432/test_{test_id}`
//! -- a host, port, database name and driver nobody had configured and nothing
//! ever dialled -- and stamped `DatabaseType::PostgreSQL` on every slot
//! regardless of the configured URL. Everything is now named for what it does:
//! slots, an endpoint carried through from configuration, and a database type
//! *parsed* from that endpoint's scheme.

use anyhow::Result;
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::{debug, info, warn};

use super::types::{DatabasePoolConfig, DatabaseUsageStatistics};

/// Allocator handing out a finite pool of database slots to tests.
///
/// See the module documentation: this reserves *permission* to use a database
/// connection, it does not establish one.
pub struct DatabaseSlotAllocator {
    /// Configuration
    config: Arc<Mutex<DatabasePoolConfig>>,
    /// Slots free to be handed out
    available_slots: Arc<Mutex<Vec<String>>>,
    /// Slots currently held by a test, keyed by slot id
    allocated_slots: Arc<Mutex<HashMap<String, DatabaseSlot>>>,
    /// Usage statistics
    usage_stats: Arc<Mutex<DatabaseUsageStatistics>>,
}

/// One reserved database slot.
///
/// A slot is a reservation, not a connection: no socket is opened for it and
/// [`Self::endpoint_url`] is simply the endpoint the operator configured, echoed
/// back so the holder knows where it is expected to connect *itself*.
#[derive(Debug, Clone)]
pub struct DatabaseSlot {
    /// Slot identifier, unique within the pool
    pub slot_id: String,
    /// Test that holds the slot
    pub test_id: String,
    /// The configured endpoint this slot grants use of. Never dialled here.
    pub endpoint_url: String,
    /// Allocated timestamp
    pub allocated_at: chrono::DateTime<chrono::Utc>,
    /// Database kind, parsed from the endpoint's URL scheme
    pub database_type: DatabaseType,
}

/// Database kinds recognised from a configured endpoint's URL scheme.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatabaseType {
    /// PostgreSQL (`postgres://`, `postgresql://`)
    PostgreSQL,
    /// MySQL (`mysql://`)
    MySQL,
    /// SQLite (`sqlite:`)
    SQLite,
    /// MongoDB (`mongodb://`)
    MongoDB,
    /// Redis (`redis://`, `rediss://`)
    Redis,
    /// Any other scheme, carried verbatim rather than guessed at.
    Custom(String),
}

impl DatabaseType {
    /// Classify `url` by its scheme.
    ///
    /// A URL with no recognisable scheme becomes
    /// [`DatabaseType::Custom`] holding the raw string, because guessing a
    /// driver from an unparseable endpoint would be a fabrication.
    pub fn from_endpoint_url(url: &str) -> Self {
        let scheme = match url.split_once(':') {
            Some((scheme, _)) => scheme.to_ascii_lowercase(),
            None => return Self::Custom(url.to_string()),
        };
        match scheme.as_str() {
            "postgres" | "postgresql" => Self::PostgreSQL,
            "mysql" | "mariadb" => Self::MySQL,
            "sqlite" => Self::SQLite,
            "mongodb" | "mongodb+srv" => Self::MongoDB,
            "redis" | "rediss" => Self::Redis,
            _ => Self::Custom(scheme),
        }
    }
}

impl DatabaseSlotAllocator {
    /// Create a slot allocator sized by `config.max_connections`.
    ///
    /// # Errors
    ///
    /// Infallible today; the `Result` matches the other pool constructors in
    /// this module so a future validating allocator can reject bad configs.
    pub async fn new(config: DatabasePoolConfig) -> Result<Self> {
        let mut available_slots = Vec::new();

        // Name the slots in the pool. These are identifiers, not handles.
        for i in 0..config.max_connections {
            available_slots.push(format!("db_slot_{}", i));
        }

        info!(
            "Initialized database slot allocator with {} slots for endpoint {}",
            available_slots.len(),
            config.database_url
        );

        Ok(Self {
            config: Arc::new(Mutex::new(config)),
            available_slots: Arc::new(Mutex::new(available_slots)),
            allocated_slots: Arc::new(Mutex::new(HashMap::new())),
            usage_stats: Arc::new(Mutex::new(DatabaseUsageStatistics::default())),
        })
    }

    /// The endpoint every slot in this pool refers to.
    pub fn endpoint_url(&self) -> String {
        self.config.lock().database_url.clone()
    }

    /// Reserve `count` slots for `test_id`, returning their slot ids.
    ///
    /// No connection is opened; the caller is expected to connect to
    /// [`Self::endpoint_url`] itself, within the budget this reservation grants.
    ///
    /// # Errors
    ///
    /// Returns an error when fewer than `count` slots are free.
    pub async fn allocate_slots(&self, count: usize, test_id: &str) -> Result<Vec<String>> {
        if count == 0 {
            return Ok(vec![]);
        }

        let endpoint_url = self.config.lock().database_url.clone();
        let database_type = DatabaseType::from_endpoint_url(&endpoint_url);
        let mut available_slots = self.available_slots.lock();
        let mut allocated_slots = self.allocated_slots.lock();
        let mut usage_stats = self.usage_stats.lock();

        if available_slots.len() < count {
            return Err(anyhow::anyhow!(
                "Insufficient available database slots: requested {}, available {}",
                count,
                available_slots.len()
            ));
        }

        let mut allocated = Vec::new();
        let now = chrono::Utc::now();

        for _ in 0..count {
            let Some(slot_id) = available_slots.pop() else {
                // Unreachable given the capacity check above, but roll back
                // rather than leave a partial reservation behind.
                for held in &allocated {
                    if let Some(slot) = allocated_slots.remove(held) {
                        available_slots.push(slot.slot_id);
                    }
                }
                return Err(anyhow::anyhow!(
                    "database slot pool drained while allocating for test {test_id}"
                ));
            };
            allocated_slots.insert(
                slot_id.clone(),
                DatabaseSlot {
                    slot_id: slot_id.clone(),
                    test_id: test_id.to_string(),
                    endpoint_url: endpoint_url.clone(),
                    allocated_at: now,
                    database_type: database_type.clone(),
                },
            );
            allocated.push(slot_id);
        }

        usage_stats.total_allocated += count as u64;
        usage_stats.currently_active = allocated_slots.len();
        usage_stats.peak_usage = usage_stats.peak_usage.max(allocated_slots.len());

        info!(
            "Allocated {} database slots for test {}: {:?}",
            allocated.len(),
            test_id,
            allocated
        );

        Ok(allocated)
    }

    /// Release one slot back to the pool.
    ///
    /// # Errors
    ///
    /// Returns an error when `slot_id` is not currently held.
    pub async fn release_slot(&self, slot_id: &str) -> Result<()> {
        let mut available_slots = self.available_slots.lock();
        let mut allocated_slots = self.allocated_slots.lock();
        let mut usage_stats = self.usage_stats.lock();

        let Some(slot) = allocated_slots.remove(slot_id) else {
            warn!("Attempted to release database slot {slot_id} that was not allocated");
            return Err(anyhow::anyhow!("database slot {slot_id} was not allocated"));
        };
        available_slots.push(slot.slot_id.clone());
        usage_stats.currently_active = allocated_slots.len();

        // Fold this slot's real held-time into the running mean. Sub-second
        // holds are measured in milliseconds rather than truncated to zero.
        let held = chrono::Utc::now().signed_duration_since(slot.allocated_at);
        let held = Duration::from_millis(held.num_milliseconds().max(0) as u64);
        usage_stats.released_count += 1;
        usage_stats.total_held_time += held;
        usage_stats.average_lifetime = usage_stats
            .total_held_time
            .checked_div(u32::try_from(usage_stats.released_count).unwrap_or(u32::MAX))
            .unwrap_or(held);

        info!(
            "Released database slot {} held by test {}",
            slot_id, slot.test_id
        );
        Ok(())
    }

    /// Release every slot held by `test_id`.
    ///
    /// # Errors
    ///
    /// Infallible today; the `Result` matches the sibling pool managers so the
    /// caller can treat all of them uniformly.
    pub async fn release_slots_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Releasing database slots for test: {}", test_id);

        let mut available_slots = self.available_slots.lock();
        let mut allocated_slots = self.allocated_slots.lock();
        let mut usage_stats = self.usage_stats.lock();

        let now = chrono::Utc::now();
        let mut released = Vec::new();
        let mut held_total = Duration::ZERO;

        allocated_slots.retain(|slot_id, slot| {
            if slot.test_id == test_id {
                available_slots.push(slot.slot_id.clone());
                let held = now.signed_duration_since(slot.allocated_at);
                held_total += Duration::from_millis(held.num_milliseconds().max(0) as u64);
                released.push(slot_id.clone());
                false
            } else {
                true
            }
        });

        usage_stats.currently_active = allocated_slots.len();
        if !released.is_empty() {
            usage_stats.released_count += released.len() as u64;
            usage_stats.total_held_time += held_total;
            usage_stats.average_lifetime = usage_stats
                .total_held_time
                .checked_div(u32::try_from(usage_stats.released_count).unwrap_or(u32::MAX))
                .unwrap_or(Duration::ZERO);
            info!(
                "Released {} database slots for test {}: {:?}",
                released.len(),
                test_id,
                released
            );
        }

        Ok(())
    }

    /// Whether `count` slots are free right now.
    ///
    /// # Errors
    ///
    /// Infallible today; the `Result` matches the sibling pool managers.
    pub async fn check_availability(&self, count: usize) -> Result<bool> {
        let available_slots = self.available_slots.lock();
        Ok(available_slots.len() >= count)
    }

    /// Get current database slot usage statistics
    ///
    /// # Errors
    ///
    /// Infallible today; the `Result` matches the sibling pool managers.
    pub async fn get_statistics(&self) -> Result<DatabaseUsageStatistics> {
        let stats = self.usage_stats.lock();
        Ok(stats.clone())
    }

    /// Number of slots free right now.
    pub async fn get_available_slot_count(&self) -> usize {
        let available_slots = self.available_slots.lock();
        available_slots.len()
    }

    /// Number of slots currently held.
    pub async fn get_active_slot_count(&self) -> usize {
        let allocated_slots = self.allocated_slots.lock();
        allocated_slots.len()
    }

    /// Fraction of the pool currently held, in `0.0..=1.0`.
    pub async fn get_utilization(&self) -> f32 {
        let max_slots = self.config.lock().max_connections;
        let active_count = self.get_active_slot_count().await;

        if max_slots == 0 {
            0.0
        } else {
            active_count as f32 / max_slots as f32
        }
    }

    /// Render a report over this allocator's real, measured numbers.
    ///
    /// 0.2.1: the previous report ended with a "Query throughput: 0.00
    /// queries/sec" line. Nothing here observes a query -- this allocator never
    /// touches a database -- so the line is gone rather than reporting an
    /// unmeasured value as a measured zero.
    pub async fn generate_slot_report(&self) -> String {
        let stats = self.get_statistics().await.unwrap_or_default();
        let available_count = self.get_available_slot_count().await;
        let active_count = self.get_active_slot_count().await;
        let utilization = self.get_utilization().await;

        format!(
            "Database Slot Report (reservations only; no connection is opened):\n\
             - Configured endpoint: {}\n\
             - Available slots: {}\n\
             - Held slots: {}\n\
             - Total slots allocated: {}\n\
             - Peak slots held: {}\n\
             - Current utilization: {:.1}%\n\
             - Mean slot hold time: {}ms",
            self.endpoint_url(),
            available_count,
            active_count,
            stats.total_allocated,
            stats.peak_usage,
            utilization * 100.0,
            stats.average_lifetime.as_millis()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_management::types::DatabasePoolConfig;

    #[tokio::test]
    async fn test_database_slot_allocator_creation() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseSlotAllocator::new(config).await;
        assert!(mgr.is_ok());
    }

    #[tokio::test]
    async fn test_initial_available_slots() {
        let config = DatabasePoolConfig::default();
        let max = config.max_connections;
        let mgr = DatabaseSlotAllocator::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        assert_eq!(mgr.get_available_slot_count().await, max);
    }

    #[tokio::test]
    async fn test_allocate_slots() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseSlotAllocator::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        let conns = mgr.allocate_slots(3, "test-001").await;
        assert!(conns.is_ok());
        assert_eq!(conns.unwrap_or_default().len(), 3);
    }

    #[tokio::test]
    async fn test_allocate_zero_slots() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseSlotAllocator::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        let conns = mgr.allocate_slots(0, "test-zero").await;
        assert!(conns.is_ok());
        assert!(conns.unwrap_or_default().is_empty());
    }

    #[tokio::test]
    async fn test_active_count_after_allocation() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseSlotAllocator::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        mgr.allocate_slots(4, "test-active").await.unwrap_or_default();
        assert_eq!(mgr.get_active_slot_count().await, 4);
    }

    #[tokio::test]
    async fn test_release_slot() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseSlotAllocator::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        let conns = mgr.allocate_slots(1, "test-dealloc").await.unwrap_or_default();
        if let Some(conn_id) = conns.first() {
            let result = mgr.release_slot(conn_id).await;
            assert!(result.is_ok());
            assert_eq!(mgr.get_active_slot_count().await, 0);
        }
    }

    #[tokio::test]
    async fn test_release_nonexistent_slot() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseSlotAllocator::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        let result = mgr.release_slot("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_release_slots_for_test() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseSlotAllocator::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        mgr.allocate_slots(3, "my-test").await.unwrap_or_default();
        let result = mgr.release_slots_for_test("my-test").await;
        assert!(result.is_ok());
        assert_eq!(mgr.get_active_slot_count().await, 0);
    }

    #[tokio::test]
    async fn test_check_availability() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseSlotAllocator::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        assert!(mgr.check_availability(5).await.unwrap_or(false));
    }

    #[tokio::test]
    async fn test_utilization_zero_initially() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseSlotAllocator::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        assert_eq!(mgr.get_utilization().await, 0.0);
    }

    #[tokio::test]
    async fn test_utilization_after_allocation() {
        let mut config = DatabasePoolConfig::default();
        config.max_connections = 10;
        let mgr = DatabaseSlotAllocator::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        mgr.allocate_slots(5, "test-util").await.unwrap_or_default();
        assert_eq!(mgr.get_utilization().await, 0.5);
    }

    #[tokio::test]
    async fn test_generate_slot_report() {
        let config = DatabasePoolConfig::default();
        let mgr = DatabaseSlotAllocator::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        let report = mgr.generate_slot_report().await;
        assert!(report.contains("Database Slot Report"));
    }

    #[test]
    fn test_database_type_variants() {
        assert_eq!(format!("{:?}", DatabaseType::PostgreSQL), "PostgreSQL");
        assert_eq!(format!("{:?}", DatabaseType::MySQL), "MySQL");
        assert_eq!(format!("{:?}", DatabaseType::SQLite), "SQLite");
        assert_eq!(format!("{:?}", DatabaseType::MongoDB), "MongoDB");
        assert_eq!(format!("{:?}", DatabaseType::Redis), "Redis");
        let custom = DatabaseType::Custom("cassandra".to_string());
        match custom {
            DatabaseType::Custom(name) => assert_eq!(name, "cassandra"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_database_type_parsed_from_endpoint_scheme() {
        assert_eq!(
            DatabaseType::from_endpoint_url("postgresql://db:5432/app"),
            DatabaseType::PostgreSQL
        );
        assert_eq!(
            DatabaseType::from_endpoint_url("postgres://db/app"),
            DatabaseType::PostgreSQL
        );
        assert_eq!(
            DatabaseType::from_endpoint_url("mysql://db/app"),
            DatabaseType::MySQL
        );
        assert_eq!(
            DatabaseType::from_endpoint_url("sqlite::memory:"),
            DatabaseType::SQLite
        );
        assert_eq!(
            DatabaseType::from_endpoint_url("redis://cache:6379"),
            DatabaseType::Redis
        );
        assert_eq!(
            DatabaseType::from_endpoint_url("mongodb://m:27017"),
            DatabaseType::MongoDB
        );
        // An unknown scheme is carried verbatim rather than guessed at.
        assert_eq!(
            DatabaseType::from_endpoint_url("clickhouse://c:9000"),
            DatabaseType::Custom("clickhouse".to_string())
        );
        // No scheme at all keeps the whole string.
        assert_eq!(
            DatabaseType::from_endpoint_url("not-a-url"),
            DatabaseType::Custom("not-a-url".to_string())
        );
    }

    #[tokio::test]
    async fn test_slot_carries_the_configured_endpoint_not_an_invented_one() {
        let mut config = DatabasePoolConfig::default();
        config.database_url = "mysql://configured-host:3306/fixture".to_string();
        let allocator = DatabaseSlotAllocator::new(config)
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        assert_eq!(
            allocator.endpoint_url(),
            "mysql://configured-host:3306/fixture"
        );
        let slots = allocator.allocate_slots(1, "endpoint-test").await.unwrap_or_default();
        assert_eq!(slots.len(), 1);
        // Before 0.2.1 every slot claimed postgresql://localhost:5432/test_<id>
        // no matter what was configured.
        let report = allocator.generate_slot_report().await;
        assert!(report.contains("mysql://configured-host:3306/fixture"));
        assert!(!report.to_lowercase().contains("localhost:5432"));
    }

    #[tokio::test]
    async fn test_report_omits_unmeasured_query_throughput() {
        let allocator = DatabaseSlotAllocator::new(DatabasePoolConfig::default())
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        let report = allocator.generate_slot_report().await;
        assert!(!report.contains("queries/sec"));
        assert!(report.contains("no connection is opened"));
    }

    #[tokio::test]
    async fn test_hold_time_statistics_are_measured() {
        let allocator = DatabaseSlotAllocator::new(DatabasePoolConfig::default())
            .await
            .unwrap_or_else(|_| panic!("creation failed"));
        let slots = allocator.allocate_slots(2, "held").await.unwrap_or_default();
        assert_eq!(
            allocator.get_statistics().await.unwrap_or_default().total_allocated,
            2
        );
        assert_eq!(
            allocator.get_statistics().await.unwrap_or_default().peak_usage,
            2
        );
        if let Some(slot_id) = slots.first() {
            assert!(allocator.release_slot(slot_id).await.is_ok());
        }
        let stats = allocator.get_statistics().await.unwrap_or_default();
        assert_eq!(stats.released_count, 1);
        assert_eq!(stats.currently_active, 1);
        assert!(allocator.release_slots_for_test("held").await.is_ok());
        let stats = allocator.get_statistics().await.unwrap_or_default();
        assert_eq!(stats.released_count, 2);
        assert_eq!(stats.currently_active, 0);
        assert_eq!(stats.average_lifetime, stats.total_held_time / 2);
    }
}
