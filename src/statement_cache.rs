//! Statement caching for improved performance
//!
//! This module provides client-side statement caching to avoid repeated
//! parsing of SQL statements on the Oracle server. When a statement is
//! executed, its cursor ID and metadata are cached. Subsequent executions
//! of the same SQL text can reuse the cached cursor, skipping the parse phase.
//!
//! # Known limitation: server-side cursor cleanup
//!
//! When a cursor completes or a cached statement is evicted, we reset the
//! cursor_id to 0 locally but do not send a cursor-close message to the
//! server. Python-oracledb piggybacks close-cursor requests on subsequent
//! messages to free server resources. For long-running connections with many
//! distinct SQL statements, this could lead to server-side cursor accumulation.
//! Oracle will eventually reclaim these, but explicit cleanup would be better.

use indexmap::IndexMap;
use std::time::Instant;

use crate::statement::Statement;

/// Wrapper for a cached statement with usage tracking
#[derive(Debug)]
struct CachedStatement {
    /// The cached statement with cursor_id and metadata
    statement: Statement,
    /// Whether the statement is currently in use
    in_use: bool,
    /// When this statement was last used
    last_used: Instant,
}

impl CachedStatement {
    fn new(statement: Statement) -> Self {
        Self {
            statement,
            in_use: false,
            last_used: Instant::now(),
        }
    }

    fn touch(&mut self) {
        self.last_used = Instant::now();
    }
}

/// Client-side statement cache using LRU eviction
///
/// The cache stores prepared statements keyed by their SQL text.
/// When a statement is retrieved from cache, its cursor ID is preserved,
/// allowing Oracle to skip parsing and use the cached server cursor.
///
/// # Example
///
/// ```ignore
/// // Statement caching is automatic when enabled via config
/// let mut config = Config::new("localhost", 1521, "FREEPDB1", "user", "pass");
/// config.set_stmtcachesize(20);  // Enable with 20 statement cache
///
/// let conn = Connection::connect_with_config(config).await?;
///
/// // First call: parses SQL, gets cursor_id from Oracle
/// conn.query("SELECT * FROM users WHERE id = :1", &[Value::Integer(1)]).await?;
///
/// // Second call: reuses cached cursor, no re-parsing!
/// conn.query("SELECT * FROM users WHERE id = :1", &[Value::Integer(2)]).await?;
/// ```
#[derive(Debug)]
pub struct StatementCache {
    /// The cache using IndexMap for O(1) lookup + LRU ordering
    cache: IndexMap<String, CachedStatement>,
    /// Maximum number of statements to cache
    max_size: usize,
    /// Server-side cursor ids that were orphaned locally (cursor reset or LRU
    /// eviction) and still need a close-cursors message sent to the server.
    /// Drained by the connection as a piggyback on the next request; without
    /// this a long-lived (pooled) connection leaks cursors until ORA-01000.
    cursors_to_close: Vec<u16>,
}

impl StatementCache {
    /// Create a new statement cache with the given maximum size
    ///
    /// A size of 0 effectively disables caching.
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: IndexMap::with_capacity(max_size),
            max_size,
            cursors_to_close: Vec::new(),
        }
    }

    /// Server-side cursor ids awaiting a close-cursors message.
    pub fn cursors_to_close(&self) -> &[u16] {
        &self.cursors_to_close
    }

    /// Clear the pending close-cursors queue (call after the close has been
    /// sent to the server).
    pub fn clear_cursors_to_close(&mut self) {
        self.cursors_to_close.clear();
    }

    /// Queue a server-side cursor id for a close-cursors message on the next
    /// request. Used by the connection to close the cursor a query actually
    /// used (which, on the cache-reuse path, is never written back into the
    /// cached statement). Ignores 0 (no cursor).
    pub fn queue_cursor_to_close(&mut self, cursor_id: u16) {
        if cursor_id != 0 {
            self.cursors_to_close.push(cursor_id);
        }
    }

    /// Get a statement from the cache, if available
    ///
    /// Returns a clone of the cached statement with preserved cursor_id and metadata.
    /// If the cached statement is already in use, returns a fresh statement.
    /// Updates LRU ordering on hit.
    pub fn get(&mut self, sql: &str) -> Option<Statement> {
        if self.max_size == 0 {
            return None;
        }

        // Check if we have this SQL cached
        if let Some(cached) = self.cache.get_mut(sql) {
            cached.touch();

            if cached.in_use {
                // Statement is in use - return a fresh statement
                // The caller will get a new cursor from Oracle
                tracing::trace!(sql = sql, "Statement cache hit but in use, returning fresh");
                return None;
            }

            // Mark as in use and return a clone for reuse
            cached.in_use = true;
            tracing::trace!(
                sql = sql,
                cursor_id = cached.statement.cursor_id(),
                "Statement cache hit"
            );
            return Some(cached.statement.clone_for_reuse());
        }

        tracing::trace!(sql = sql, "Statement cache miss");
        None
    }

    /// Store a statement in the cache
    ///
    /// DDL statements are never cached. If the cache is full, the least
    /// recently used statement is evicted and its cursor ID is queued for closing.
    pub fn put(&mut self, sql: String, statement: Statement) {
        if self.max_size == 0 {
            return;
        }

        // Never cache DDL statements (CREATE, ALTER, DROP, etc.)
        if statement.is_ddl() {
            tracing::trace!(sql = sql, "Not caching DDL statement");
            return;
        }

        // Don't cache statements without a cursor_id (not yet executed)
        if statement.cursor_id() == 0 {
            tracing::trace!(sql = sql, "Not caching statement without cursor_id");
            return;
        }

        // Check if already cached (update it)
        if let Some(cached) = self.cache.get_mut(&sql) {
            cached.statement = statement;
            cached.in_use = false;
            cached.touch();
            tracing::trace!(sql = sql, "Updated existing cache entry");
            return;
        }

        // Evict LRU entry if cache is full
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        tracing::trace!(
            sql = sql,
            cursor_id = statement.cursor_id(),
            "Adding statement to cache"
        );
        self.cache.insert(sql, CachedStatement::new(statement));
    }

    /// Return a statement to the cache after use
    ///
    /// This marks the statement as no longer in use so it can be reused.
    pub fn return_statement(&mut self, sql: &str) {
        if let Some(cached) = self.cache.get_mut(sql) {
            cached.in_use = false;
            tracing::trace!(sql = sql, "Statement returned to cache");
        }
    }

    /// Mark a cursor as closed in the cache
    ///
    /// Resets cursor_id to 0 so the next execution gets a fresh cursor from
    /// Oracle. This prevents data corruption from reusing stale cursor IDs.
    ///
    /// Following python-oracledb's clear_cursor design pattern.
    ///
    /// Resets the cached cursor id to 0 so the next execution re-parses and
    /// gets a fresh cursor. This does not free the cursor on the server; the
    /// connection queues the cursor it actually used via
    /// [`queue_cursor_to_close`](Self::queue_cursor_to_close).
    pub fn mark_cursor_closed(&mut self, sql: &str) {
        if let Some(cached) = self.cache.get_mut(sql) {
            if cached.statement.cursor_id() != 0 {
                cached.statement.set_cursor_id(0);
                cached.statement.set_executed(false);
                tracing::trace!(sql = sql, "Cursor reset to 0");
            }
        }
    }

    /// Clear all cached statements
    ///
    /// This should be called when the session changes (e.g., DRCP session switch).
    /// Pending close-cursors ids are dropped too: they belong to the old
    /// session, so closing them against a new session could target an unrelated
    /// cursor.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.cursors_to_close.clear();
        tracing::debug!("Statement cache cleared");
    }

    /// Get the current number of cached statements
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get the maximum cache size
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Evict the least recently used entry
    fn evict_lru(&mut self) {
        // Find the LRU entry (first entry that's not in use)
        let lru_key = self
            .cache
            .iter()
            .filter(|(_, cached)| !cached.in_use)
            .min_by_key(|(_, cached)| cached.last_used)
            .map(|(key, _)| key.clone());

        if let Some(key) = lru_key {
            if let Some(cached) = self.cache.swap_remove(&key) {
                let cursor_id = cached.statement.cursor_id();
                tracing::trace!(sql = key, cursor_id, "Evicted LRU statement from cache");
                // The evicted statement's server cursor is still open; queue it
                // for closing so eviction does not leak cursors.
                if cursor_id != 0 {
                    self.cursors_to_close.push(cursor_id);
                }
            }
        } else {
            // All statements are in use - this is rare but possible
            tracing::warn!("Statement cache full and all statements in use");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_statement(sql: &str, cursor_id: u16) -> Statement {
        let mut stmt = Statement::new(sql);
        stmt.set_cursor_id(cursor_id);
        stmt.set_executed(true);
        stmt
    }

    #[test]
    fn test_cache_basic() {
        let mut cache = StatementCache::new(5);

        // Add a statement
        let stmt = make_test_statement("SELECT 1 FROM DUAL", 100);
        cache.put("SELECT 1 FROM DUAL".to_string(), stmt);

        assert_eq!(cache.len(), 1);

        // Retrieve it
        let cached = cache.get("SELECT 1 FROM DUAL").expect("Should be cached");
        assert_eq!(cached.cursor_id(), 100);

        // Return it
        cache.return_statement("SELECT 1 FROM DUAL");
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = StatementCache::new(5);
        assert!(cache.get("SELECT 1 FROM DUAL").is_none());
    }

    #[test]
    fn test_cache_disabled() {
        let mut cache = StatementCache::new(0);

        let stmt = make_test_statement("SELECT 1 FROM DUAL", 100);
        cache.put("SELECT 1 FROM DUAL".to_string(), stmt);

        assert_eq!(cache.len(), 0);
        assert!(cache.get("SELECT 1 FROM DUAL").is_none());
    }

    #[test]
    fn test_ddl_not_cached() {
        let mut cache = StatementCache::new(5);

        let mut stmt = Statement::new("CREATE TABLE test (id NUMBER)");
        stmt.set_cursor_id(100);
        cache.put("CREATE TABLE test (id NUMBER)".to_string(), stmt);

        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_no_cursor_not_cached() {
        let mut cache = StatementCache::new(5);

        // Statement without cursor_id should not be cached
        let stmt = Statement::new("SELECT 1 FROM DUAL");
        cache.put("SELECT 1 FROM DUAL".to_string(), stmt);

        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = StatementCache::new(3);

        // Add 3 statements
        cache.put(
            "SELECT 1 FROM DUAL".to_string(),
            make_test_statement("SELECT 1 FROM DUAL", 1),
        );
        cache.put(
            "SELECT 2 FROM DUAL".to_string(),
            make_test_statement("SELECT 2 FROM DUAL", 2),
        );
        cache.put(
            "SELECT 3 FROM DUAL".to_string(),
            make_test_statement("SELECT 3 FROM DUAL", 3),
        );

        assert_eq!(cache.len(), 3);

        // Access the first one to make it recently used
        cache.get("SELECT 1 FROM DUAL");
        cache.return_statement("SELECT 1 FROM DUAL");

        // Add a 4th - should evict "SELECT 2" (LRU)
        cache.put(
            "SELECT 4 FROM DUAL".to_string(),
            make_test_statement("SELECT 4 FROM DUAL", 4),
        );

        assert_eq!(cache.len(), 3);
        assert!(cache.get("SELECT 2 FROM DUAL").is_none()); // Evicted
        assert!(cache.get("SELECT 1 FROM DUAL").is_some()); // Still there
    }

    #[test]
    fn test_in_use_not_returned() {
        let mut cache = StatementCache::new(5);

        cache.put(
            "SELECT 1 FROM DUAL".to_string(),
            make_test_statement("SELECT 1 FROM DUAL", 100),
        );

        // Get the statement (marks it in use)
        let _ = cache.get("SELECT 1 FROM DUAL");

        // Try to get it again - should return None because it's in use
        assert!(cache.get("SELECT 1 FROM DUAL").is_none());

        // Return it
        cache.return_statement("SELECT 1 FROM DUAL");

        // Now we can get it again
        assert!(cache.get("SELECT 1 FROM DUAL").is_some());
    }

    #[test]
    fn test_clear() {
        let mut cache = StatementCache::new(5);

        cache.put(
            "SELECT 1 FROM DUAL".to_string(),
            make_test_statement("SELECT 1 FROM DUAL", 1),
        );
        cache.put(
            "SELECT 2 FROM DUAL".to_string(),
            make_test_statement("SELECT 2 FROM DUAL", 2),
        );

        assert_eq!(cache.len(), 2);

        cache.clear();

        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_update_existing() {
        let mut cache = StatementCache::new(5);

        cache.put(
            "SELECT 1 FROM DUAL".to_string(),
            make_test_statement("SELECT 1 FROM DUAL", 100),
        );

        // Update with new cursor_id
        cache.put(
            "SELECT 1 FROM DUAL".to_string(),
            make_test_statement("SELECT 1 FROM DUAL", 200),
        );

        assert_eq!(cache.len(), 1);

        let cached = cache.get("SELECT 1 FROM DUAL").unwrap();
        assert_eq!(cached.cursor_id(), 200);
    }

    #[test]
    fn test_mark_cursor_closed_resets_without_queuing() {
        let mut cache = StatementCache::new(5);
        cache.put(
            "SELECT 1 FROM DUAL".to_string(),
            make_test_statement("SELECT 1 FROM DUAL", 42),
        );

        // mark_cursor_closed only resets the cached cursor locally; queuing the
        // server close is the connection's job (it knows the real cursor used).
        cache.mark_cursor_closed("SELECT 1 FROM DUAL");
        assert_eq!(cache.get("SELECT 1 FROM DUAL").unwrap().cursor_id(), 0);
        assert!(cache.cursors_to_close().is_empty());
    }

    #[test]
    fn test_queue_and_drain_cursors_to_close() {
        let mut cache = StatementCache::new(5);
        assert!(cache.cursors_to_close().is_empty());

        cache.queue_cursor_to_close(42);
        cache.queue_cursor_to_close(0); // 0 = no cursor, ignored
        cache.queue_cursor_to_close(100);
        assert_eq!(cache.cursors_to_close(), &[42, 100]);

        // Draining clears the queue (called once the close is on the wire).
        cache.clear_cursors_to_close();
        assert!(cache.cursors_to_close().is_empty());
    }

    #[test]
    fn test_lru_eviction_queues_evicted_cursor() {
        let mut cache = StatementCache::new(1);
        cache.put(
            "SELECT 1 FROM DUAL".to_string(),
            make_test_statement("SELECT 1 FROM DUAL", 7),
        );
        // Inserting a second statement evicts the first; its server cursor
        // (id 7) is still open and must be queued for closing.
        cache.put(
            "SELECT 2 FROM DUAL".to_string(),
            make_test_statement("SELECT 2 FROM DUAL", 8),
        );

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.cursors_to_close(), &[7]);
    }

    #[test]
    fn test_clear_drops_pending_closes() {
        let mut cache = StatementCache::new(5);
        cache.queue_cursor_to_close(5);
        assert_eq!(cache.cursors_to_close(), &[5]);

        // A session change abandons the old cursors; queued ids must be dropped
        // so we never close a cursor id belonging to a different session.
        cache.clear();
        assert!(cache.cursors_to_close().is_empty());
    }
}
