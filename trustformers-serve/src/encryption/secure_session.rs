//! Secure Inference Session Management
//!
//! Provides bearer-token based session authentication with policy enforcement
//! for rate limiting, model access control, and request quotas.

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// Error Types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by session management operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionError {
    /// No session exists with the given ID.
    SessionNotFound(String),
    /// The provided token does not match the session token.
    InvalidToken,
    /// The session lifetime has elapsed.
    SessionExpired,
    /// The request rate exceeds the configured limit.
    RateLimitExceeded,
    /// The requested model is not in the session's allowlist.
    ModelNotAllowed(String),
    /// The session has exhausted its maximum request count.
    RequestLimitExceeded,
    /// The requested token count exceeds the per-request limit.
    TokenLimitExceeded,
    /// The hex string has an invalid format or length.
    InvalidHexString(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionNotFound(id) => write!(f, "Session not found: '{}'", id),
            Self::InvalidToken => write!(f, "Invalid session token"),
            Self::SessionExpired => write!(f, "Session has expired"),
            Self::RateLimitExceeded => write!(f, "Rate limit exceeded"),
            Self::ModelNotAllowed(model) => write!(f, "Model '{}' is not allowed for this session", model),
            Self::RequestLimitExceeded => write!(f, "Request limit for this session has been reached"),
            Self::TokenLimitExceeded => write!(f, "Token count exceeds the per-request maximum"),
            Self::InvalidHexString(s) => write!(f, "Invalid hex string: '{}'", s),
        }
    }
}

impl std::error::Error for SessionError {}

// ─────────────────────────────────────────────────────────────────────────────
// FNV-1a helpers for deterministic token generation
// ─────────────────────────────────────────────────────────────────────────────

const FNV_OFFSET_BASIS_64: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME_64: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64-bit hash over a byte slice.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS_64;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME_64);
    }
    hash
}

/// Expand a 64-bit seed into 32 deterministic bytes using iterated FNV-1a.
fn expand_seed_to_token(seed: u64) -> [u8; 32] {
    let mut token = [0u8; 32];
    let mut state = seed;
    // Four rounds of 8 bytes each, mixing in round index.
    for chunk_start in (0..32usize).step_by(8) {
        state = fnv1a_64(&state.to_le_bytes());
        state ^= fnv1a_64(&(chunk_start as u64).to_le_bytes());
        let bytes = state.to_le_bytes();
        token[chunk_start..chunk_start + 8].copy_from_slice(&bytes);
    }
    token
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionToken
// ─────────────────────────────────────────────────────────────────────────────

/// A 32-byte bearer token for session authentication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionToken {
    /// Raw token bytes — deterministic, derived from the seed via FNV-1a.
    pub token: [u8; 32],
}

impl SessionToken {
    /// Generate a deterministic session token from a 64-bit seed.
    pub fn generate(seed: u64) -> Self {
        Self { token: expand_seed_to_token(seed) }
    }

    /// Encode the token as a lowercase hexadecimal string (64 characters).
    pub fn to_hex(&self) -> String {
        self.token.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Decode a hexadecimal string back into a [`SessionToken`].
    ///
    /// # Errors
    /// Returns [`SessionError::InvalidHexString`] if the string is not exactly
    /// 64 hex characters.
    pub fn from_hex(s: &str) -> Result<Self, SessionError> {
        if s.len() != 64 {
            return Err(SessionError::InvalidHexString(s.to_string()));
        }

        let mut token = [0u8; 32];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hi = hex_nibble(chunk[0])
                .ok_or_else(|| SessionError::InvalidHexString(s.to_string()))?;
            let lo = hex_nibble(chunk[1])
                .ok_or_else(|| SessionError::InvalidHexString(s.to_string()))?;
            token[i] = (hi << 4) | lo;
        }
        Ok(Self { token })
    }

    /// Constant-time comparison of this token against a provided byte array.
    ///
    /// Always compares all 32 bytes to prevent timing side-channels.
    pub fn verify(&self, provided: &[u8; 32]) -> bool {
        // XOR all bytes together; if all match the accumulator stays 0.
        let mut diff: u8 = 0;
        for (a, b) in self.token.iter().zip(provided.iter()) {
            diff |= a ^ b;
        }
        diff == 0
    }
}

/// Decode a single ASCII hex character to its nibble value, or `None` if invalid.
fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Policy constraints applied to every request within a session.
#[derive(Debug, Clone)]
pub struct SessionPolicy {
    /// Maximum total requests allowed in this session.
    pub max_requests: u64,
    /// Maximum number of tokens that may be generated in a single request.
    pub max_tokens_per_request: usize,
    /// Maximum request rate in requests per second.
    pub rate_limit_rps: f32,
    /// Models this session may access; empty means all models are allowed.
    pub allowed_models: Vec<String>,
    /// Session lifetime in milliseconds.
    pub session_timeout_ms: u64,
    /// IP addresses allowed to use this session; empty means all IPs allowed.
    pub ip_allowlist: Vec<String>,
}

impl Default for SessionPolicy {
    fn default() -> Self {
        Self {
            max_requests: 10_000,
            max_tokens_per_request: 4096,
            rate_limit_rps: 100.0,
            allowed_models: vec![],
            session_timeout_ms: 3_600_000, // 1 hour
            ip_allowlist: vec![],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionState
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime state of an active inference session.
pub struct SessionState {
    /// Human-readable identifier for this session.
    pub session_id: String,
    /// Bearer token used to authenticate requests.
    pub token: SessionToken,
    /// Policy constraints attached to this session.
    pub policy: SessionPolicy,
    /// Moment at which the session was created.
    pub created_at: Instant,
    /// Number of requests processed so far.
    pub request_count: u64,
    /// Cumulative number of tokens generated across all requests.
    pub total_tokens_generated: u64,
    /// Timestamp of the most recent request, if any.
    pub last_request_time: Option<Instant>,
}

impl SessionState {
    /// Return `true` if the session lifetime has elapsed.
    pub fn is_expired(&self) -> bool {
        let elapsed_ms = self.created_at.elapsed().as_millis() as u64;
        elapsed_ms >= self.policy.session_timeout_ms
    }

    /// Return the number of requests that may still be processed.
    pub fn requests_remaining(&self) -> u64 {
        self.policy.max_requests.saturating_sub(self.request_count)
    }

    /// Return `true` if sufficient time has passed since the last request
    /// (i.e., the configured rate limit is not being exceeded).
    ///
    /// If no prior request has been recorded, the check always passes.
    pub fn check_rate_limit(&self) -> bool {
        match self.last_request_time {
            None => true,
            Some(last) => {
                // Minimum interval between requests in seconds.
                let min_interval_secs = if self.policy.rate_limit_rps > 0.0 {
                    1.0 / self.policy.rate_limit_rps
                } else {
                    f32::MAX
                };
                let elapsed_secs = last.elapsed().as_secs_f32();
                elapsed_secs >= min_interval_secs
            },
        }
    }

    /// Return `true` if `model_id` is permitted under this session's policy.
    ///
    /// An empty `allowed_models` list means all models are permitted.
    pub fn model_allowed(&self, model_id: &str) -> bool {
        if self.policy.allowed_models.is_empty() {
            return true;
        }
        self.policy.allowed_models.iter().any(|m| m == model_id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SecureSessionManager
// ─────────────────────────────────────────────────────────────────────────────

/// Manages the lifecycle and authentication of secure inference sessions.
pub struct SecureSessionManager {
    sessions: HashMap<String, SessionState>,
}

impl SecureSessionManager {
    /// Create an empty session manager.
    pub fn new() -> Self {
        Self { sessions: HashMap::new() }
    }

    /// Create a new session and return its bearer token.
    ///
    /// # Arguments
    /// * `session_id` — unique identifier for the new session
    /// * `policy` — access control policy for this session
    /// * `seed` — 64-bit seed for deterministic token generation
    pub fn create_session(
        &mut self,
        session_id: String,
        policy: SessionPolicy,
        seed: u64,
    ) -> SessionToken {
        let token = SessionToken::generate(seed);
        let state = SessionState {
            session_id: session_id.clone(),
            token: token.clone(),
            policy,
            created_at: Instant::now(),
            request_count: 0,
            total_tokens_generated: 0,
            last_request_time: None,
        };
        self.sessions.insert(session_id, state);
        token
    }

    /// Verify that the provided token bytes match the session's stored token.
    ///
    /// # Errors
    /// - [`SessionError::SessionNotFound`] — no session with that ID
    /// - [`SessionError::InvalidToken`] — token mismatch
    pub fn authenticate(
        &self,
        session_id: &str,
        token_bytes: &[u8; 32],
    ) -> Result<(), SessionError> {
        let state = self
            .sessions
            .get(session_id)
            .ok_or_else(|| SessionError::SessionNotFound(session_id.to_string()))?;

        if !state.token.verify(token_bytes) {
            return Err(SessionError::InvalidToken);
        }
        Ok(())
    }

    /// Perform all policy checks required before processing a request.
    ///
    /// Does **not** mutate state (that is done by [`record_request`]).
    ///
    /// # Errors
    /// Checks are applied in this order:
    /// 1. [`SessionError::SessionNotFound`]
    /// 2. [`SessionError::SessionExpired`]
    /// 3. [`SessionError::RateLimitExceeded`]
    /// 4. [`SessionError::ModelNotAllowed`]
    /// 5. [`SessionError::TokenLimitExceeded`]
    /// 6. [`SessionError::RequestLimitExceeded`]
    pub fn check_request(
        &self,
        session_id: &str,
        model_id: &str,
        max_tokens: usize,
    ) -> Result<(), SessionError> {
        let state = self
            .sessions
            .get(session_id)
            .ok_or_else(|| SessionError::SessionNotFound(session_id.to_string()))?;

        if state.is_expired() {
            return Err(SessionError::SessionExpired);
        }

        if !state.check_rate_limit() {
            return Err(SessionError::RateLimitExceeded);
        }

        if !state.model_allowed(model_id) {
            return Err(SessionError::ModelNotAllowed(model_id.to_string()));
        }

        if max_tokens > state.policy.max_tokens_per_request {
            return Err(SessionError::TokenLimitExceeded);
        }

        if state.requests_remaining() == 0 {
            return Err(SessionError::RequestLimitExceeded);
        }

        Ok(())
    }

    /// Record a completed request and update session counters.
    ///
    /// # Errors
    /// - [`SessionError::SessionNotFound`] — no session with that ID
    pub fn record_request(
        &mut self,
        session_id: &str,
        tokens_generated: usize,
    ) -> Result<(), SessionError> {
        let state = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| SessionError::SessionNotFound(session_id.to_string()))?;

        state.request_count += 1;
        state.total_tokens_generated += tokens_generated as u64;
        state.last_request_time = Some(Instant::now());
        Ok(())
    }

    /// Revoke a session, returning `true` if it existed.
    pub fn revoke_session(&mut self, session_id: &str) -> bool {
        self.sessions.remove(session_id).is_some()
    }

    /// Return the number of currently active (non-expired) sessions.
    pub fn active_session_count(&self) -> usize {
        self.sessions.values().filter(|s| !s.is_expired()).count()
    }
}

impl Default for SecureSessionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_manager_with_session(session_id: &str) -> (SecureSessionManager, SessionToken) {
        let mut mgr = SecureSessionManager::new();
        let token = mgr.create_session(
            session_id.to_string(),
            SessionPolicy::default(),
            42,
        );
        (mgr, token)
    }

    // ── 1. Session creation returns a non-zero token ──────────────────────────
    #[test]
    fn test_session_creation() {
        let (mgr, token) = default_manager_with_session("sess-001");
        assert_eq!(mgr.active_session_count(), 1);
        // Token must not be all zeros.
        assert_ne!(token.token, [0u8; 32]);
    }

    // ── 2. Token hex round-trip ───────────────────────────────────────────────
    #[test]
    fn test_token_hex_roundtrip() {
        let token = SessionToken::generate(0xDEAD_BEEF_CAFE_1234);
        let hex = token.to_hex();
        assert_eq!(hex.len(), 64);
        let recovered = SessionToken::from_hex(&hex).expect("valid hex");
        assert_eq!(token, recovered);
    }

    // ── 3. Token hex from_hex rejects invalid input ───────────────────────────
    #[test]
    fn test_token_from_hex_invalid() {
        // Too short.
        assert!(SessionToken::from_hex("deadbeef").is_err());
        // Correct length but invalid character.
        let bad = "x".repeat(64);
        assert!(SessionToken::from_hex(&bad).is_err());
    }

    // ── 4. Token verification — correct and incorrect ─────────────────────────
    #[test]
    fn test_token_verification() {
        let token = SessionToken::generate(99);
        assert!(token.verify(&token.token));

        let mut wrong = token.token;
        wrong[0] ^= 0xFF;
        assert!(!token.verify(&wrong));
    }

    // ── 5. Session expiry check with a very short timeout ─────────────────────
    #[test]
    fn test_session_expiry() {
        let short_policy = SessionPolicy {
            session_timeout_ms: 0, // expires immediately
            ..Default::default()
        };
        let mut mgr = SecureSessionManager::new();
        mgr.create_session("expire-sess".to_string(), short_policy, 1);

        // With a 0 ms timeout the session should already be expired.
        let result = mgr.check_request("expire-sess", "model-a", 100);
        assert!(matches!(result, Err(SessionError::SessionExpired)));
    }

    // ── 6. Rate limiting rejects a second immediate request ───────────────────
    #[test]
    fn test_rate_limit_enforcement() {
        // Set an extremely low rate: 0.001 rps → min interval ≈ 1000 s.
        let policy = SessionPolicy {
            rate_limit_rps: 0.001,
            ..Default::default()
        };
        let mut mgr = SecureSessionManager::new();
        mgr.create_session("rate-sess".to_string(), policy, 2);

        // First request succeeds and records a timestamp.
        mgr.check_request("rate-sess", "any-model", 10)
            .expect("first request should succeed");
        mgr.record_request("rate-sess", 10).expect("record should succeed");

        // Second immediate request must be rate-limited.
        let result = mgr.check_request("rate-sess", "any-model", 10);
        assert!(matches!(result, Err(SessionError::RateLimitExceeded)));
    }

    // ── 7. Model allowlist enforcement ────────────────────────────────────────
    #[test]
    fn test_model_allowlist() {
        let policy = SessionPolicy {
            allowed_models: vec!["gpt-trust-7b".to_string()],
            ..Default::default()
        };
        let mut mgr = SecureSessionManager::new();
        mgr.create_session("model-sess".to_string(), policy, 3);

        // Allowed model — should pass.
        assert!(mgr.check_request("model-sess", "gpt-trust-7b", 100).is_ok());

        // Disallowed model — should fail.
        let result = mgr.check_request("model-sess", "not-allowed-model", 100);
        assert!(matches!(result, Err(SessionError::ModelNotAllowed(_))));
    }

    // ── 8. Request limit exhaustion ───────────────────────────────────────────
    #[test]
    fn test_request_limit() {
        // Use a very high rate limit so rate-limiting does not interfere.
        let policy = SessionPolicy {
            max_requests: 2,
            rate_limit_rps: f32::MAX,
            ..Default::default()
        };
        let mut mgr = SecureSessionManager::new();
        mgr.create_session("limit-sess".to_string(), policy, 4);

        // Consume all allowed requests.
        for i in 0..2u64 {
            mgr.check_request("limit-sess", "model", 10)
                .unwrap_or_else(|e| panic!("request {} should succeed: {}", i, e));
            mgr.record_request("limit-sess", 10).expect("record");
        }

        // Next request must be rejected.
        let result = mgr.check_request("limit-sess", "model", 10);
        assert!(matches!(result, Err(SessionError::RequestLimitExceeded)));
    }

    // ── 9. Session revocation ─────────────────────────────────────────────────
    #[test]
    fn test_session_revocation() {
        let (mut mgr, _) = default_manager_with_session("revoke-sess");
        assert_eq!(mgr.active_session_count(), 1);

        let removed = mgr.revoke_session("revoke-sess");
        assert!(removed);
        assert_eq!(mgr.active_session_count(), 0);

        // Revoking again returns false.
        assert!(!mgr.revoke_session("revoke-sess"));
    }

    // ── 10. active_session_count excludes expired sessions ────────────────────
    #[test]
    fn test_active_session_count() {
        let mut mgr = SecureSessionManager::new();

        // One normal session.
        mgr.create_session("live-1".to_string(), SessionPolicy::default(), 10);

        // One immediately-expired session.
        let dead_policy = SessionPolicy {
            session_timeout_ms: 0,
            ..Default::default()
        };
        mgr.create_session("dead-1".to_string(), dead_policy, 11);

        // Only the live session counts.
        assert_eq!(mgr.active_session_count(), 1);
    }

    // ── 11. check_request pipeline — all checks in sequence ──────────────────
    #[test]
    fn test_check_request_full_pipeline() {
        let policy = SessionPolicy {
            allowed_models: vec!["trust-model".to_string()],
            max_tokens_per_request: 512,
            max_requests: 100,
            rate_limit_rps: 1000.0, // effectively unlimited for this test
            session_timeout_ms: 3_600_000,
            ip_allowlist: vec![],
        };
        let mut mgr = SecureSessionManager::new();
        mgr.create_session("pipe-sess".to_string(), policy, 99);

        // Correct usage — should pass.
        assert!(mgr.check_request("pipe-sess", "trust-model", 512).is_ok());

        // Token limit exceeded.
        let result = mgr.check_request("pipe-sess", "trust-model", 513);
        assert!(matches!(result, Err(SessionError::TokenLimitExceeded)));

        // Session not found.
        let result = mgr.check_request("nonexistent", "trust-model", 10);
        assert!(matches!(result, Err(SessionError::SessionNotFound(_))));
    }

    // ── 12. authenticate succeeds with correct token, fails with wrong ────────
    #[test]
    fn test_authentication() {
        let (mgr, token) = default_manager_with_session("auth-sess");

        // Correct token.
        assert!(mgr.authenticate("auth-sess", &token.token).is_ok());

        // Wrong token.
        let mut bad = token.token;
        bad[15] ^= 0x01;
        let result = mgr.authenticate("auth-sess", &bad);
        assert!(matches!(result, Err(SessionError::InvalidToken)));

        // Non-existent session.
        let result = mgr.authenticate("ghost-sess", &token.token);
        assert!(matches!(result, Err(SessionError::SessionNotFound(_))));
    }

    // ── 13. record_request increments counters correctly ─────────────────────
    #[test]
    fn test_record_request_increments_counters() {
        let (mut mgr, _) = default_manager_with_session("counter-sess");

        mgr.record_request("counter-sess", 100).expect("first record");
        mgr.record_request("counter-sess", 200).expect("second record");

        let state = mgr.sessions.get("counter-sess").expect("session must exist");
        assert_eq!(state.request_count, 2);
        assert_eq!(state.total_tokens_generated, 300);
        assert!(state.last_request_time.is_some());
    }

    // ── 14. record_request fails for non-existent session ────────────────────
    #[test]
    fn test_record_request_session_not_found() {
        let mut mgr = SecureSessionManager::new();
        let result = mgr.record_request("ghost", 50);
        assert!(matches!(result, Err(SessionError::SessionNotFound(_))));
    }

    // ── 15. Token generation is deterministic ─────────────────────────────────
    #[test]
    fn test_token_generation_is_deterministic() {
        let t1 = SessionToken::generate(12345);
        let t2 = SessionToken::generate(12345);
        assert_eq!(t1, t2);

        let t3 = SessionToken::generate(12346);
        assert_ne!(t1, t3);
    }

    // ── 16. Session expiry via elapsed Duration (not instant) ─────────────────
    #[test]
    fn test_session_not_expired_when_fresh() {
        // 1-hour timeout: a freshly created session must NOT be expired.
        let policy = SessionPolicy {
            session_timeout_ms: 3_600_000,
            ..Default::default()
        };
        let mut mgr = SecureSessionManager::new();
        mgr.create_session("fresh-sess".to_string(), policy, 77);

        let state = mgr.sessions.get("fresh-sess").expect("session must exist");
        // Elapsed time is tiny (microseconds); far below 1 hour.
        assert!(!state.is_expired());
    }

    // ── 17. requests_remaining saturates at zero ──────────────────────────────
    #[test]
    fn test_requests_remaining_saturates() {
        let policy = SessionPolicy { max_requests: 1, ..Default::default() };
        let mut mgr = SecureSessionManager::new();
        mgr.create_session("sat-sess".to_string(), policy, 88);
        mgr.record_request("sat-sess", 0).expect("record");
        mgr.record_request("sat-sess", 0).expect("record over limit stored ok");

        let state = mgr.sessions.get("sat-sess").expect("session exists");
        // request_count is 2, max_requests is 1 → saturates to 0.
        assert_eq!(state.requests_remaining(), 0);
    }

    // ── 18. Default manager has zero sessions ─────────────────────────────────
    #[test]
    fn test_default_manager_is_empty() {
        let mgr = SecureSessionManager::default();
        assert_eq!(mgr.active_session_count(), 0);
    }

    // ── 19. Multiple sessions can coexist ─────────────────────────────────────
    #[test]
    fn test_multiple_sessions_coexist() {
        let mut mgr = SecureSessionManager::new();
        mgr.create_session("sess-a".to_string(), SessionPolicy::default(), 100);
        mgr.create_session("sess-b".to_string(), SessionPolicy::default(), 101);
        mgr.create_session("sess-c".to_string(), SessionPolicy::default(), 102);
        assert_eq!(mgr.active_session_count(), 3);
    }

    // ── 20. Different seeds yield different tokens ─────────────────────────────
    #[test]
    fn test_different_seeds_yield_different_tokens() {
        let mut mgr = SecureSessionManager::new();
        let t1 = mgr.create_session("s1".to_string(), SessionPolicy::default(), 1000);
        let t2 = mgr.create_session("s2".to_string(), SessionPolicy::default(), 1001);
        assert_ne!(t1.token, t2.token);
    }

    // ── 21. Same seed always yields the same token ────────────────────────────
    #[test]
    fn test_same_seed_yields_same_token() {
        let t1 = SessionToken::generate(42_424_242);
        let t2 = SessionToken::generate(42_424_242);
        assert_eq!(t1.token, t2.token);
    }

    // ── 22. Token hex string is exactly 64 lowercase hex chars ───────────────
    #[test]
    fn test_token_hex_length_and_charset() {
        let token = SessionToken::generate(0x0102_0304_0506_0708);
        let hex = token.to_hex();
        assert_eq!(hex.len(), 64, "hex must be 64 characters");
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "hex must be lowercase");
    }

    // ── 23. Revoking one session leaves others intact ─────────────────────────
    #[test]
    fn test_revoke_one_of_many() {
        let mut mgr = SecureSessionManager::new();
        mgr.create_session("keep-1".to_string(), SessionPolicy::default(), 200);
        mgr.create_session("remove".to_string(), SessionPolicy::default(), 201);
        mgr.create_session("keep-2".to_string(), SessionPolicy::default(), 202);

        let removed = mgr.revoke_session("remove");
        assert!(removed, "revoke should return true for existing session");
        assert_eq!(mgr.active_session_count(), 2);

        // Remaining sessions are still valid.
        assert!(mgr.check_request("keep-1", "any-model", 10).is_ok());
        assert!(mgr.check_request("keep-2", "any-model", 10).is_ok());
    }

    // ── 24. zero rate_limit_rps blocks all requests after the first ───────────
    #[test]
    fn test_zero_rate_limit_rps_blocks() {
        let policy = SessionPolicy {
            rate_limit_rps: 0.0,  // 0 rps → min_interval = f32::MAX
            ..Default::default()
        };
        let mut mgr = SecureSessionManager::new();
        mgr.create_session("zero-rate".to_string(), policy, 300);

        // First request is fine (no prior timestamp).
        mgr.check_request("zero-rate", "any-model", 10)
            .expect("first request must pass with no prior timestamp");
        mgr.record_request("zero-rate", 10).expect("record");

        // Second immediate request must fail.
        let result = mgr.check_request("zero-rate", "any-model", 10);
        assert!(matches!(result, Err(SessionError::RateLimitExceeded)));
    }

    // ── 25. ip_allowlist field exists and can be set ──────────────────────────
    #[test]
    fn test_ip_allowlist_field() {
        let policy = SessionPolicy {
            ip_allowlist: vec!["192.168.1.1".to_string(), "10.0.0.1".to_string()],
            ..Default::default()
        };
        assert_eq!(policy.ip_allowlist.len(), 2);
    }

    // ── 26. model_allowed with empty list permits anything ────────────────────
    #[test]
    fn test_model_allowed_empty_list_permits_all() {
        let policy = SessionPolicy { allowed_models: vec![], ..Default::default() };
        let mut mgr = SecureSessionManager::new();
        mgr.create_session("open-sess".to_string(), policy, 400);

        // Any model identifier must be accepted.
        assert!(mgr.check_request("open-sess", "arbitrary-model-xyz", 10).is_ok());
        assert!(mgr.check_request("open-sess", "another-model", 10).is_ok());
    }

    // ── 27. token.verify uses constant-time comparison ────────────────────────
    //     Flipping every individual bit must make verify() return false.
    #[test]
    fn test_token_verify_all_bits() {
        let token = SessionToken::generate(9_999_999);
        for bit_pos in 0..256u32 {
            let byte_idx = (bit_pos / 8) as usize;
            let bit_mask = 1u8 << (bit_pos % 8);
            let mut tampered = token.token;
            tampered[byte_idx] ^= bit_mask;
            assert!(!token.verify(&tampered),
                "single bit flip at position {} must fail verify", bit_pos);
        }
    }

    // ── 28. requests_remaining equals max when no requests recorded ───────────
    #[test]
    fn test_requests_remaining_initial_equals_max() {
        let max = 500u64;
        let policy = SessionPolicy { max_requests: max, ..Default::default() };
        let mut mgr = SecureSessionManager::new();
        mgr.create_session("fresh".to_string(), policy, 500);
        let state = mgr.sessions.get("fresh").expect("must exist");
        assert_eq!(state.requests_remaining(), max);
    }

    // ── 29. SessionError Display messages contain key identifiers ─────────────
    #[test]
    fn test_session_error_display_messages() {
        let sid = "my-session-42".to_string();
        let not_found = SessionError::SessionNotFound(sid.clone());
        assert!(not_found.to_string().contains(&sid));

        let model_err = SessionError::ModelNotAllowed("gpt-bad".to_string());
        assert!(model_err.to_string().contains("gpt-bad"));

        let hex_err = SessionError::InvalidHexString("zzzz".to_string());
        assert!(hex_err.to_string().contains("zzzz"));

        // Simple variants must have non-empty display.
        assert!(!SessionError::InvalidToken.to_string().is_empty());
        assert!(!SessionError::SessionExpired.to_string().is_empty());
        assert!(!SessionError::RateLimitExceeded.to_string().is_empty());
        assert!(!SessionError::RequestLimitExceeded.to_string().is_empty());
        assert!(!SessionError::TokenLimitExceeded.to_string().is_empty());
    }

    // ── 30. from_hex rejects strings that are longer than 64 chars ───────────
    #[test]
    fn test_from_hex_rejects_too_long() {
        let too_long = "ab".repeat(33); // 66 characters
        let result = SessionToken::from_hex(&too_long);
        assert!(result.is_err(), "string longer than 64 chars must be rejected");
    }

    // ── 31. total_tokens_generated accumulates across requests ────────────────
    #[test]
    fn test_total_tokens_accumulated() {
        let (mut mgr, _) = default_manager_with_session("acc-sess");
        mgr.record_request("acc-sess", 100).expect("record 1");
        mgr.record_request("acc-sess", 250).expect("record 2");
        mgr.record_request("acc-sess", 50).expect("record 3");

        let state = mgr.sessions.get("acc-sess").expect("must exist");
        assert_eq!(state.total_tokens_generated, 400);
        assert_eq!(state.request_count, 3);
    }

    // ── 32. SessionPolicy clone produces an independent copy ──────────────────
    #[test]
    fn test_session_policy_clone_is_independent() {
        let mut original = SessionPolicy::default();
        let mut cloned = original.clone();
        cloned.max_requests = 99;
        original.max_requests = 1;
        assert_ne!(original.max_requests, cloned.max_requests);
    }
}
