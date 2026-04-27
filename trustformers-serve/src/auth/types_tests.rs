//! Comprehensive tests for auth/types.rs

#[cfg(test)]
mod tests {
    use crate::auth::types::{
        ApiKey, ApiKeyRateLimit, AuthConfig, AuthError, AuthService, Claims, OAuth2Config,
        OAuth2State, User,
    };
    use jsonwebtoken::Algorithm;
    use std::collections::HashMap;

    fn make_auth_config() -> AuthConfig {
        AuthConfig {
            secret_key: "test-secret-key-at-least-32-chars!!".to_string(),
            issuer: "trustformers-serve".to_string(),
            audience: "trustformers-api".to_string(),
            algorithm: Algorithm::HS256,
            leeway: 0,
            oauth2_providers: HashMap::new(),
            oauth2_state_max_age: 600,
        }
    }

    fn make_auth_service() -> AuthService {
        AuthService::new(make_auth_config())
    }

    // --- ApiKey tests ---

    #[test]
    fn test_api_key_new_creates_active_key() {
        let key = ApiKey::new(
            "test-key".to_string(),
            "user-001".to_string(),
            vec!["inference:read".to_string()],
        );
        assert!(key.is_active);
        assert!(!key.id.is_empty());
        assert!(key.key.starts_with("tf_"));
        assert_eq!(key.usage_count, 0);
        assert!(key.expires_at.is_none());
    }

    #[test]
    fn test_api_key_has_scope_match() {
        let key = ApiKey::new(
            "k".to_string(),
            "u".to_string(),
            vec!["inference:read".to_string(), "admin:write".to_string()],
        );
        assert!(key.has_scope("inference:read"));
        assert!(key.has_scope("admin:write"));
        assert!(!key.has_scope("inference:write"));
    }

    #[test]
    fn test_api_key_has_any_scope() {
        let key = ApiKey::new(
            "k".to_string(),
            "u".to_string(),
            vec!["inference:read".to_string()],
        );
        assert!(key.has_any_scope(&["inference:read", "admin:write"]));
        assert!(!key.has_any_scope(&["admin:read", "admin:write"]));
    }

    #[test]
    fn test_api_key_not_expired_without_expiry() {
        let key = ApiKey::new("k".to_string(), "u".to_string(), vec![]);
        assert!(!key.is_expired());
    }

    #[test]
    fn test_api_key_expired_with_past_expiry() {
        let past = chrono::Utc::now() - chrono::Duration::seconds(3600);
        let key = ApiKey::new("k".to_string(), "u".to_string(), vec![]).with_expiration(past);
        assert!(key.is_expired());
    }

    #[test]
    fn test_api_key_not_expired_with_future_expiry() {
        let future = chrono::Utc::now() + chrono::Duration::seconds(3600);
        let key = ApiKey::new("k".to_string(), "u".to_string(), vec![]).with_expiration(future);
        assert!(!key.is_expired());
    }

    #[test]
    fn test_api_key_is_valid_active_not_expired() {
        let key = ApiKey::new("k".to_string(), "u".to_string(), vec![]);
        assert!(key.is_valid());
    }

    #[test]
    fn test_api_key_is_invalid_when_inactive() {
        let mut key = ApiKey::new("k".to_string(), "u".to_string(), vec![]);
        key.is_active = false;
        assert!(!key.is_valid());
    }

    #[test]
    fn test_api_key_update_last_used_increments_count() {
        let mut key = ApiKey::new("k".to_string(), "u".to_string(), vec![]);
        assert_eq!(key.usage_count, 0);
        key.update_last_used();
        assert_eq!(key.usage_count, 1);
        key.update_last_used();
        assert_eq!(key.usage_count, 2);
        assert!(key.last_used_at.is_some());
    }

    #[test]
    fn test_api_key_ip_allowed_no_whitelist() {
        let key = ApiKey::new("k".to_string(), "u".to_string(), vec![]);
        assert!(key.is_ip_allowed("192.168.1.1"));
        assert!(key.is_ip_allowed("10.0.0.1"));
    }

    #[test]
    fn test_api_key_ip_allowed_with_whitelist() {
        let key = ApiKey::new("k".to_string(), "u".to_string(), vec![])
            .with_ip_whitelist(vec!["192.168.1.1".to_string()]);
        assert!(key.is_ip_allowed("192.168.1.1"));
        assert!(!key.is_ip_allowed("10.0.0.1"));
    }

    #[test]
    fn test_api_key_endpoint_allowed_no_restriction() {
        let key = ApiKey::new("k".to_string(), "u".to_string(), vec![]);
        assert!(key.is_endpoint_allowed("/api/v1/inference"));
    }

    #[test]
    fn test_api_key_endpoint_allowed_with_prefix_match() {
        let key = ApiKey::new("k".to_string(), "u".to_string(), vec![])
            .with_allowed_endpoints(vec!["/api/v1/".to_string()]);
        assert!(key.is_endpoint_allowed("/api/v1/inference"));
        assert!(!key.is_endpoint_allowed("/admin/users"));
    }

    #[test]
    fn test_api_key_with_rate_limit() {
        let rate_limit = ApiKeyRateLimit {
            requests_per_minute: 60,
            requests_per_hour: 3600,
            requests_per_day: 86400,
        };
        let key = ApiKey::new("k".to_string(), "u".to_string(), vec![]).with_rate_limit(rate_limit);
        let rl = key.rate_limit.as_ref().expect("rate limit should be set");
        assert_eq!(rl.requests_per_minute, 60);
        assert_eq!(rl.requests_per_hour, 3600);
    }

    // --- Claims tests ---

    #[test]
    fn test_claims_new_creates_valid_claims() {
        let claims = Claims::new(
            "user-123".to_string(),
            "issuer".to_string(),
            "audience".to_string(),
            vec!["read".to_string()],
            3600,
        );
        assert_eq!(claims.sub, "user-123");
        assert_eq!(claims.iss, "issuer");
        assert_eq!(claims.aud, "audience");
        assert!(!claims.is_expired());
    }

    #[test]
    fn test_claims_has_scope() {
        let claims = Claims::new(
            "u".to_string(),
            "i".to_string(),
            "a".to_string(),
            vec!["inference:read".to_string(), "admin:write".to_string()],
            3600,
        );
        assert!(claims.has_scope("inference:read"));
        assert!(!claims.has_scope("inference:write"));
    }

    #[test]
    fn test_claims_has_any_scope() {
        let claims = Claims::new(
            "u".to_string(),
            "i".to_string(),
            "a".to_string(),
            vec!["inference:read".to_string()],
            3600,
        );
        assert!(claims.has_any_scope(&["inference:read", "admin:write"]));
        assert!(!claims.has_any_scope(&["admin:read", "admin:write"]));
    }

    #[test]
    fn test_claims_not_expired_with_future_exp() {
        let claims = Claims::new(
            "u".to_string(),
            "i".to_string(),
            "a".to_string(),
            vec![],
            3600,
        );
        assert!(!claims.is_expired());
    }

    // --- User tests ---

    #[test]
    fn test_user_new_creates_active_user() {
        let user = User::new("alice".to_string(), "secret123".to_string());
        assert_eq!(user.username, "alice");
        assert!(user.is_active);
        assert!(!user.id.is_empty());
        assert!(user.roles.contains(&"user".to_string()));
    }

    #[test]
    fn test_user_password_verification_correct() {
        let user = User::new("bob".to_string(), "mypassword".to_string());
        assert!(user.verify_password("mypassword"));
    }

    #[test]
    fn test_user_password_verification_wrong() {
        let user = User::new("bob".to_string(), "mypassword".to_string());
        assert!(!user.verify_password("wrongpassword"));
    }

    #[test]
    fn test_user_update_last_login() {
        let mut user = User::new("carol".to_string(), "pass".to_string());
        assert!(user.last_login_at.is_none());
        user.update_last_login();
        assert!(user.last_login_at.is_some());
    }

    // --- AuthService tests ---

    #[test]
    fn test_auth_service_authenticate_admin_user() {
        let svc = make_auth_service();
        let result = svc.authenticate_user("admin", "admin123");
        assert!(result.is_ok());
        let user = result.expect("auth should succeed");
        assert_eq!(user.username, "admin");
    }

    #[test]
    fn test_auth_service_authenticate_test_user() {
        let svc = make_auth_service();
        let result = svc.authenticate_user("testuser", "password123");
        assert!(result.is_ok());
    }

    #[test]
    fn test_auth_service_wrong_password_fails() {
        let svc = make_auth_service();
        let result = svc.authenticate_user("admin", "wrongpass");
        assert!(result.is_err());
        assert!(matches!(result, Err(AuthError::InvalidCredentials)));
    }

    #[test]
    fn test_auth_service_unknown_user_fails() {
        let svc = make_auth_service();
        let result = svc.authenticate_user("nobody", "pass");
        assert!(result.is_err());
    }

    #[test]
    fn test_auth_service_create_and_verify_token() {
        let svc = make_auth_service();
        let claims = Claims::new(
            "user-001".to_string(),
            "trustformers-serve".to_string(),
            "trustformers-api".to_string(),
            vec!["inference:read".to_string()],
            3600,
        );
        let token = svc.create_token(&claims).expect("token creation should succeed");
        assert!(!token.is_empty());
        let verified = svc.verify_token(&token).expect("token verification should succeed");
        assert_eq!(verified.sub, "user-001");
    }

    #[test]
    fn test_auth_service_verify_invalid_token_fails() {
        let svc = make_auth_service();
        let result = svc.verify_token("not.a.valid.jwt.token");
        assert!(result.is_err());
    }

    #[test]
    fn test_auth_service_extract_bearer_token() {
        let svc = make_auth_service();
        let header = "Bearer mytoken123";
        let token = svc.extract_token_from_header(header).expect("should extract token");
        assert_eq!(token, "mytoken123");
    }

    #[test]
    fn test_auth_service_extract_token_invalid_format_fails() {
        let svc = make_auth_service();
        let result = svc.extract_token_from_header("Basic abc123");
        assert!(result.is_err());
        assert!(matches!(result, Err(AuthError::InvalidHeaderFormat)));
    }

    #[test]
    fn test_auth_service_create_api_key() {
        let svc = make_auth_service();
        let key = svc.create_api_key(
            "my-api-key".to_string(),
            "user-001".to_string(),
            vec!["inference:read".to_string()],
        );
        assert!(key.is_active);
        assert_eq!(key.name, "my-api-key");
        assert_eq!(key.user_id, "user-001");
    }

    #[test]
    fn test_auth_service_revoke_api_key() {
        let svc = make_auth_service();
        let key = svc.create_api_key("test-key".to_string(), "u".to_string(), vec![]);
        svc.revoke_api_key(&key.key).expect("revoke should succeed");
        // Verify the key is deactivated
        let keys = svc.list_api_keys("u");
        let found_key = keys.iter().find(|k| k.id == key.id);
        assert!(found_key.is_some());
        let found = found_key.expect("key should exist");
        assert!(!found.is_active);
    }

    #[test]
    fn test_auth_service_list_api_keys_by_user() {
        let svc = make_auth_service();
        svc.create_api_key("key-1".to_string(), "user-A".to_string(), vec![]);
        svc.create_api_key("key-2".to_string(), "user-A".to_string(), vec![]);
        svc.create_api_key("key-3".to_string(), "user-B".to_string(), vec![]);
        let user_a_keys = svc.list_api_keys("user-A");
        assert_eq!(user_a_keys.len(), 2);
        let user_b_keys = svc.list_api_keys("user-B");
        assert_eq!(user_b_keys.len(), 1);
    }

    #[test]
    fn test_oauth2_state_new() {
        let state = OAuth2State::new(
            "https://example.com/callback".to_string(),
            vec!["openid".to_string()],
        );
        assert!(!state.state.is_empty());
        assert_eq!(state.redirect_uri, "https://example.com/callback");
        assert_eq!(state.scopes, vec!["openid".to_string()]);
    }

    #[test]
    fn test_oauth2_state_not_expired_freshly_created() {
        let state = OAuth2State::new("https://cb.example.com/".to_string(), vec![]);
        assert!(!state.is_expired(600));
    }

    #[test]
    fn test_oauth2_config_google() {
        let config = OAuth2Config::google(
            "client-id".to_string(),
            "client-secret".to_string(),
            "https://example.com/callback".to_string(),
        );
        assert_eq!(config.provider_name, "google");
        assert!(config.auth_url.contains("accounts.google.com"));
        assert!(!config.default_scopes.is_empty());
    }

    #[test]
    fn test_oauth2_config_github() {
        let config = OAuth2Config::github(
            "gh-client-id".to_string(),
            "gh-secret".to_string(),
            "https://example.com/cb".to_string(),
        );
        assert_eq!(config.provider_name, "github");
        assert!(config.auth_url.contains("github.com"));
    }

    #[test]
    fn test_oauth2_config_azure() {
        let config = OAuth2Config::azure(
            "az-client-id".to_string(),
            "az-secret".to_string(),
            "https://example.com/cb".to_string(),
            "tenant-xyz".to_string(),
        );
        assert_eq!(config.provider_name, "azure");
        assert!(config.auth_url.contains("tenant-xyz"));
        assert!(config.token_url.contains("tenant-xyz"));
    }

    #[test]
    fn test_auth_service_create_new_user() {
        let svc = make_auth_service();
        let result = svc.create_user("newuser".to_string(), "pass123".to_string());
        assert!(result.is_ok());
        let user_id = result.expect("user creation should succeed");
        assert!(!user_id.is_empty());
    }

    #[test]
    fn test_auth_service_duplicate_user_fails() {
        let svc = make_auth_service();
        svc.create_user("dup-user".to_string(), "pass".to_string())
            .expect("first create should succeed");
        let result = svc.create_user("dup-user".to_string(), "pass2".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_api_key_extract_from_apikey_header() {
        let svc = make_auth_service();
        let header = "ApiKey tf_abc123";
        let key = svc.extract_api_key_from_header(header).expect("should extract");
        assert_eq!(key, "tf_abc123");
    }

    #[test]
    fn test_api_key_extract_from_bearer_header() {
        let svc = make_auth_service();
        let header = "Bearer tf_xyz789";
        let key = svc.extract_api_key_from_header(header).expect("should extract");
        assert_eq!(key, "tf_xyz789");
    }
}
