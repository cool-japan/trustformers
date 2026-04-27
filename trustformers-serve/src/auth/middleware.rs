//! Standalone auth middleware types for API key and JWT management.
//!
//! These types are self-contained and do not depend on the axum/chrono/jsonwebtoken
//! stack used by the rest of the auth module.

use std::collections::HashMap;

// ── ApiKeyInfo ────────────────────────────────────────────────────────────────

/// A lightweight API key record that stores an FNV-1a hash of the raw key
/// rather than the key itself.
#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    /// Unique identifier for this key (client-visible).
    pub key_id: String,
    /// FNV-1a hex digest of the raw key (never stored in plain text).
    pub key_hash: String,
    /// Wall-clock time when the key was created.
    pub created_at: std::time::SystemTime,
    /// Optional expiry; `None` means the key never expires.
    pub expires_at: Option<std::time::SystemTime>,
    /// Permission scopes granted to this key (e.g. `"inference:read"`).
    pub scopes: Vec<String>,
    /// Optional requests-per-minute rate limit.
    pub rate_limit: Option<u32>,
}

impl ApiKeyInfo {
    /// Create a new `ApiKeyInfo` by hashing `raw_key` with FNV-1a.
    pub fn new(key_id: &str, raw_key: &str) -> Self {
        Self {
            key_id: key_id.to_string(),
            key_hash: Self::hash_key(raw_key),
            created_at: std::time::SystemTime::now(),
            expires_at: None,
            scopes: Vec::new(),
            rate_limit: None,
        }
    }

    /// FNV-1a 64-bit hash of `raw_key`, returned as a 16-character lowercase hex string.
    pub fn hash_key(raw_key: &str) -> String {
        let mut hash: u64 = 14_695_981_039_346_656_037;
        for byte in raw_key.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
        format!("{:016x}", hash)
    }

    /// Returns `true` if the key has an expiry that is in the past.
    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            None => false,
            Some(exp) => std::time::SystemTime::now() > exp,
        }
    }

    /// Returns `true` if `scope` is listed in this key's scopes.
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope)
    }
}

// ── JwtClaims ─────────────────────────────────────────────────────────────────

/// Minimal JWT claims for model-server authorization (no external dependencies).
#[derive(Debug, Clone)]
pub struct JwtClaims {
    /// Subject — typically the user or service account ID.
    pub sub: String,
    /// Issuer — the authority that signed this token.
    pub iss: String,
    /// Issued-at Unix timestamp (seconds).
    pub iat: u64,
    /// Expiration Unix timestamp (seconds).
    pub exp: u64,
    /// Permission scopes carried by this token.
    pub scopes: Vec<String>,
    /// Model IDs this token is authorized to access.
    pub model_access: Vec<String>,
}

impl JwtClaims {
    /// Returns `true` if `current_time_secs >= exp`.
    pub fn is_expired(&self, current_time_secs: u64) -> bool {
        current_time_secs >= self.exp
    }

    /// Returns `true` if `model_id` appears in `model_access`.
    pub fn has_model_access(&self, model_id: &str) -> bool {
        self.model_access.iter().any(|m| m == model_id)
    }

    /// Serialize the claims to a JSON string (manual, no serde dependency).
    pub fn to_payload_json(&self) -> String {
        let scopes_json = self
            .scopes
            .iter()
            .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(",");
        let models_json = self
            .model_access
            .iter()
            .map(|m| format!("\"{}\"", m.replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(",");

        format!(
            r#"{{"sub":"{sub}","iss":"{iss}","iat":{iat},"exp":{exp},"scopes":[{scopes}],"model_access":[{models}]}}"#,
            sub = self.sub.replace('"', "\\\""),
            iss = self.iss.replace('"', "\\\""),
            iat = self.iat,
            exp = self.exp,
            scopes = scopes_json,
            models = models_json,
        )
    }
}

// ── AuthConfig ────────────────────────────────────────────────────────────────

/// Configuration for the auth middleware.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// Whether API key authentication is enabled.
    pub allow_api_keys: bool,
    /// Whether JWT authentication is enabled.
    pub allow_jwt: bool,
    /// Whether to reject requests that did not arrive over HTTPS.
    pub require_https: bool,
    /// Expected issuer claim in JWTs.
    pub jwt_issuer: String,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            allow_api_keys: true,
            allow_jwt: true,
            require_https: false,
            jwt_issuer: "trustformers-serve".to_string(),
        }
    }
}

// ── AuthStats ─────────────────────────────────────────────────────────────────

/// Running authentication statistics.
#[derive(Debug, Clone, Default)]
pub struct AuthStats {
    pub total_requests: u64,
    pub auth_successes: u64,
    pub auth_failures: u64,
    pub expired_token_count: u64,
}

// ── AuthError ─────────────────────────────────────────────────────────────────

/// Errors returned by authentication operations.
#[derive(Debug)]
pub enum AuthError {
    /// No credentials were supplied with the request.
    MissingCredentials,
    /// The API key was not found or its hash does not match.
    InvalidApiKey,
    /// The credential is valid but has expired.
    ExpiredToken,
    /// The credential does not carry the required scope.
    InsufficientScope(String),
    /// The JWT could not be validated.
    InvalidJwt(String),
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::MissingCredentials => write!(f, "no credentials provided"),
            AuthError::InvalidApiKey => write!(f, "invalid API key"),
            AuthError::ExpiredToken => write!(f, "credential has expired"),
            AuthError::InsufficientScope(scope) => {
                write!(f, "insufficient scope: required '{}'", scope)
            }
            AuthError::InvalidJwt(msg) => write!(f, "invalid JWT: {}", msg),
        }
    }
}

impl std::error::Error for AuthError {}

// ── AuthMiddleware ────────────────────────────────────────────────────────────

/// Main authentication middleware.
///
/// Stores registered API keys indexed by their **hash** so that the raw key is
/// never retained in memory beyond the registration call.
pub struct AuthMiddleware {
    /// Registered keys, keyed by their FNV-1a hash.
    pub valid_api_keys: HashMap<String, ApiKeyInfo>,
    pub config: AuthConfig,
    pub auth_stats: AuthStats,
}

impl AuthMiddleware {
    /// Create a new middleware with the given configuration.
    pub fn new(config: AuthConfig) -> Self {
        Self {
            valid_api_keys: HashMap::new(),
            config,
            auth_stats: AuthStats::default(),
        }
    }

    /// Register an API key.  The key is stored indexed by its hash.
    pub fn register_api_key(&mut self, key: ApiKeyInfo) {
        self.valid_api_keys.insert(key.key_hash.clone(), key);
    }

    /// Revoke an API key by its `key_id`.
    ///
    /// Returns `true` if the key was found and removed, `false` otherwise.
    pub fn revoke_api_key(&mut self, key_id: &str) -> bool {
        let hash_to_remove = self
            .valid_api_keys
            .iter()
            .find(|(_, v)| v.key_id == key_id)
            .map(|(k, _)| k.clone());

        match hash_to_remove {
            Some(h) => {
                self.valid_api_keys.remove(&h);
                true
            }
            None => false,
        }
    }

    /// Authenticate using a raw API key string.
    ///
    /// Hashes the supplied key and looks it up in the registered keys.
    /// Updates `auth_stats` on every call.
    pub fn authenticate_api_key(&mut self, raw_key: &str) -> Result<&ApiKeyInfo, AuthError> {
        self.auth_stats.total_requests += 1;

        let hash = ApiKeyInfo::hash_key(raw_key);
        if !self.valid_api_keys.contains_key(&hash) {
            self.auth_stats.auth_failures += 1;
            return Err(AuthError::InvalidApiKey);
        }

        // Check expiry before returning a reference.
        let expired = self
            .valid_api_keys
            .get(&hash)
            .map(|k| k.is_expired())
            .unwrap_or(true);

        if expired {
            self.auth_stats.auth_failures += 1;
            self.auth_stats.expired_token_count += 1;
            return Err(AuthError::ExpiredToken);
        }

        self.auth_stats.auth_successes += 1;
        // Safety: we confirmed the key exists above.
        Ok(self.valid_api_keys.get(&hash).expect("key must exist after contains_key check"))
    }

    /// Authenticate using pre-parsed JWT claims.
    ///
    /// Validates expiry and issuer.  Updates `auth_stats` on every call.
    pub fn authenticate_jwt_claims(
        &mut self,
        claims: &JwtClaims,
        current_time: u64,
    ) -> Result<(), AuthError> {
        self.auth_stats.total_requests += 1;

        if claims.is_expired(current_time) {
            self.auth_stats.auth_failures += 1;
            self.auth_stats.expired_token_count += 1;
            return Err(AuthError::ExpiredToken);
        }

        if claims.iss != self.config.jwt_issuer {
            self.auth_stats.auth_failures += 1;
            return Err(AuthError::InvalidJwt(format!(
                "issuer mismatch: expected '{}', got '{}'",
                self.config.jwt_issuer, claims.iss
            )));
        }

        self.auth_stats.auth_successes += 1;
        Ok(())
    }

    /// Return a reference to the current authentication statistics.
    pub fn stats(&self) -> &AuthStats {
        &self.auth_stats
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    fn make_claims(exp: u64, iss: &str) -> JwtClaims {
        JwtClaims {
            sub: "user-42".to_string(),
            iss: iss.to_string(),
            iat: 0,
            exp,
            scopes: vec!["inference:read".to_string()],
            model_access: vec!["llama3".to_string(), "phi3".to_string()],
        }
    }

    // 1. ApiKeyInfo::new creates correct key_id
    #[test]
    fn test_api_key_info_new_key_id() {
        let key = ApiKeyInfo::new("key-001", "supersecret");
        assert_eq!(key.key_id, "key-001");
    }

    // 2. hash_key is deterministic
    #[test]
    fn test_hash_key_deterministic() {
        let h1 = ApiKeyInfo::hash_key("my-secret-key");
        let h2 = ApiKeyInfo::hash_key("my-secret-key");
        assert_eq!(h1, h2);
    }

    // 3. hash_key produces 16-char hex string
    #[test]
    fn test_hash_key_length() {
        let h = ApiKeyInfo::hash_key("test");
        assert_eq!(h.len(), 16, "FNV-1a 64-bit hash should be 16 hex chars");
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // 4. hash_key differs for different inputs
    #[test]
    fn test_hash_key_differs_for_different_inputs() {
        let h1 = ApiKeyInfo::hash_key("key-a");
        let h2 = ApiKeyInfo::hash_key("key-b");
        assert_ne!(h1, h2);
    }

    // 5. is_expired returns false for key with no expires_at
    #[test]
    fn test_api_key_not_expired_no_expiry() {
        let key = ApiKeyInfo::new("k", "raw");
        assert!(!key.is_expired());
    }

    // 6. is_expired returns false for future expiry
    #[test]
    fn test_api_key_not_expired_future() {
        let mut key = ApiKeyInfo::new("k", "raw");
        key.expires_at = Some(SystemTime::now() + Duration::from_secs(3600));
        assert!(!key.is_expired());
    }

    // 7. is_expired returns true for past expiry
    #[test]
    fn test_api_key_expired_past() {
        let mut key = ApiKeyInfo::new("k", "raw");
        // Subtract 1 second — already in the past.
        key.expires_at = Some(SystemTime::now() - Duration::from_secs(1));
        assert!(key.is_expired());
    }

    // 8. has_scope returns true for matching scope
    #[test]
    fn test_api_key_has_scope_true() {
        let mut key = ApiKeyInfo::new("k", "raw");
        key.scopes = vec!["inference:read".to_string(), "admin:write".to_string()];
        assert!(key.has_scope("inference:read"));
    }

    // 9. has_scope returns false for missing scope
    #[test]
    fn test_api_key_has_scope_false() {
        let key = ApiKeyInfo::new("k", "raw");
        assert!(!key.has_scope("admin:write"));
    }

    // 10. JwtClaims::is_expired returns true when current_time >= exp
    #[test]
    fn test_jwt_claims_is_expired_at_boundary() {
        let claims = make_claims(1000, "trustformers-serve");
        assert!(claims.is_expired(1000));
        assert!(claims.is_expired(1001));
    }

    // 11. JwtClaims::is_expired returns false when current_time < exp
    #[test]
    fn test_jwt_claims_not_expired() {
        let claims = make_claims(9999, "trustformers-serve");
        assert!(!claims.is_expired(9998));
    }

    // 12. JwtClaims::has_model_access returns true for matching model
    #[test]
    fn test_jwt_has_model_access_true() {
        let claims = make_claims(9999, "trustformers-serve");
        assert!(claims.has_model_access("llama3"));
    }

    // 13. JwtClaims::has_model_access returns false for missing model
    #[test]
    fn test_jwt_has_model_access_false() {
        let claims = make_claims(9999, "trustformers-serve");
        assert!(!claims.has_model_access("gpt-4"));
    }

    // 14. JwtClaims::to_payload_json contains "sub" field
    #[test]
    fn test_jwt_payload_json_has_sub() {
        let claims = make_claims(9999, "trustformers-serve");
        let json = claims.to_payload_json();
        assert!(json.contains("\"sub\""), "JSON must contain sub field");
        assert!(json.contains("user-42"));
    }

    // 15. JwtClaims::to_payload_json contains "exp" field
    #[test]
    fn test_jwt_payload_json_has_exp() {
        let claims = make_claims(42000, "trustformers-serve");
        let json = claims.to_payload_json();
        assert!(json.contains("\"exp\""));
        assert!(json.contains("42000"));
    }

    // 16. AuthMiddleware::new initializes with empty keys
    #[test]
    fn test_auth_middleware_new_empty() {
        let mw = AuthMiddleware::new(AuthConfig::default());
        assert!(mw.valid_api_keys.is_empty());
        assert_eq!(mw.auth_stats.total_requests, 0);
    }

    // 17. register_api_key adds key to valid_api_keys
    #[test]
    fn test_register_api_key_adds_entry() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let key = ApiKeyInfo::new("key-001", "mysecret");
        mw.register_api_key(key);
        assert_eq!(mw.valid_api_keys.len(), 1);
    }

    // 18. revoke_api_key returns true for existing key
    #[test]
    fn test_revoke_api_key_existing() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        mw.register_api_key(ApiKeyInfo::new("key-001", "mysecret"));
        let removed = mw.revoke_api_key("key-001");
        assert!(removed);
        assert!(mw.valid_api_keys.is_empty());
    }

    // 19. revoke_api_key returns false for non-existent key
    #[test]
    fn test_revoke_api_key_non_existent() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        assert!(!mw.revoke_api_key("does-not-exist"));
    }

    // 20. authenticate_api_key succeeds with valid registered key
    #[test]
    fn test_authenticate_api_key_success() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        mw.register_api_key(ApiKeyInfo::new("k1", "correct-raw-key"));
        let result = mw.authenticate_api_key("correct-raw-key");
        assert!(result.is_ok());
        assert_eq!(result.expect("should be ok").key_id, "k1");
    }

    // 21. authenticate_api_key returns InvalidApiKey for unknown key
    #[test]
    fn test_authenticate_api_key_invalid() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let result = mw.authenticate_api_key("unknown-key");
        assert!(matches!(result, Err(AuthError::InvalidApiKey)));
    }

    // 22. authenticate_api_key updates auth_stats.total_requests
    #[test]
    fn test_authenticate_api_key_updates_stats() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let _ = mw.authenticate_api_key("any-key");
        let _ = mw.authenticate_api_key("another-key");
        assert_eq!(mw.auth_stats.total_requests, 2);
    }

    // 23. authenticate_jwt_claims succeeds for non-expired claims with matching issuer
    #[test]
    fn test_authenticate_jwt_claims_success() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let claims = make_claims(9999, "trustformers-serve");
        let result = mw.authenticate_jwt_claims(&claims, 1000);
        assert!(result.is_ok());
        assert_eq!(mw.auth_stats.auth_successes, 1);
    }

    // 24. authenticate_jwt_claims returns ExpiredToken for expired claims
    #[test]
    fn test_authenticate_jwt_claims_expired() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let claims = make_claims(500, "trustformers-serve");
        let result = mw.authenticate_jwt_claims(&claims, 1000); // current_time > exp
        assert!(matches!(result, Err(AuthError::ExpiredToken)));
        assert_eq!(mw.auth_stats.expired_token_count, 1);
    }

    // 25. AuthStats default has all zeros
    #[test]
    fn test_auth_stats_default_zeros() {
        let stats = AuthStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.auth_successes, 0);
        assert_eq!(stats.auth_failures, 0);
        assert_eq!(stats.expired_token_count, 0);
    }

    // 26. AuthError Display messages are non-empty
    #[test]
    fn test_auth_error_display_non_empty() {
        let errors: Vec<AuthError> = vec![
            AuthError::MissingCredentials,
            AuthError::InvalidApiKey,
            AuthError::ExpiredToken,
            AuthError::InsufficientScope("admin".to_string()),
            AuthError::InvalidJwt("bad sig".to_string()),
        ];
        for e in errors {
            assert!(!e.to_string().is_empty(), "Display must be non-empty for {:?}", e);
        }
    }

    // 27. authenticate_jwt_claims returns error for mismatched issuer
    #[test]
    fn test_authenticate_jwt_claims_issuer_mismatch() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let claims = make_claims(9999, "other-issuer");
        let result = mw.authenticate_jwt_claims(&claims, 1000);
        assert!(matches!(result, Err(AuthError::InvalidJwt(_))));
        assert_eq!(mw.auth_stats.auth_failures, 1);
    }
}
