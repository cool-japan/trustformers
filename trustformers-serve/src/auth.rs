use crate::audit::{AuditEventType, AuditLogger};
use axum::{
    extract::{Query, Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::Response,
};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::time::SystemTime;
use thiserror::Error;
use uuid::Uuid;

/// Hash a password using SHA-256 with salt
fn hash_password(password: &str) -> String {
    let salt = "trustformers_salt_2024"; // In production, use a random salt per user
    let mut hasher = Sha256::new();
    hasher.update(format!("{}{}", password, salt));
    format!("{:x}", hasher.finalize())
}

/// Verify a password against its hash
fn verify_password(password: &str, hash: &str) -> bool {
    hash_password(password) == hash
}

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("Missing authorization header")]
    MissingAuthHeader,

    #[error("Invalid authorization header format")]
    InvalidHeaderFormat,

    #[error("Invalid token: {0}")]
    InvalidToken(#[from] jsonwebtoken::errors::Error),

    #[error("Token expired")]
    TokenExpired,

    #[error("Insufficient permissions")]
    InsufficientPermissions,

    #[error("Invalid credentials")]
    InvalidCredentials,

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("API key expired")]
    ApiKeyExpired,

    #[error("API key revoked")]
    ApiKeyRevoked,

    #[error("OAuth2 authorization error: {0}")]
    OAuth2AuthorizationError(String),

    #[error("OAuth2 token exchange error: {0}")]
    OAuth2TokenError(String),

    #[error("OAuth2 provider error: {0}")]
    OAuth2ProviderError(String),

    #[error("Invalid OAuth2 state parameter")]
    InvalidOAuth2State,

    #[error("OAuth2 scope denied")]
    OAuth2ScopeDenied,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,         // Subject (user ID)
    pub exp: usize,          // Expiration time
    pub iat: usize,          // Issued at
    pub iss: String,         // Issuer
    pub aud: String,         // Audience
    pub scopes: Vec<String>, // Permissions/scopes
}

impl Claims {
    pub fn new(
        user_id: String,
        issuer: String,
        audience: String,
        scopes: Vec<String>,
        expires_in_seconds: usize,
    ) -> Self {
        let now = chrono::Utc::now().timestamp() as usize;
        Self {
            sub: user_id,
            exp: now + expires_in_seconds,
            iat: now,
            iss: issuer,
            aud: audience,
            scopes,
        }
    }

    pub fn has_scope(&self, required_scope: &str) -> bool {
        self.scopes.iter().any(|scope| scope == required_scope)
    }

    pub fn has_any_scope(&self, required_scopes: &[&str]) -> bool {
        required_scopes.iter().any(|scope| self.has_scope(scope))
    }

    pub fn is_expired(&self) -> bool {
        let now = chrono::Utc::now().timestamp() as usize;
        self.exp < now
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,
    pub key: String,
    pub name: String,
    pub user_id: String,
    pub scopes: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    pub last_used_at: Option<chrono::DateTime<chrono::Utc>>,
    pub is_active: bool,
    pub rate_limit: Option<ApiKeyRateLimit>,
    pub usage_count: u64,
    pub ip_whitelist: Option<Vec<String>>,
    pub allowed_endpoints: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyRateLimit {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub requests_per_day: u32,
}

impl ApiKey {
    pub fn new(name: String, user_id: String, scopes: Vec<String>) -> Self {
        let id = Uuid::new_v4().to_string();
        let key = format!("tf_{}", Uuid::new_v4().simple());

        Self {
            id,
            key,
            name,
            user_id,
            scopes,
            created_at: chrono::Utc::now(),
            expires_at: None,
            last_used_at: None,
            is_active: true,
            rate_limit: None,
            usage_count: 0,
            ip_whitelist: None,
            allowed_endpoints: None,
        }
    }

    pub fn with_expiration(mut self, expires_at: chrono::DateTime<chrono::Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    pub fn with_rate_limit(mut self, rate_limit: ApiKeyRateLimit) -> Self {
        self.rate_limit = Some(rate_limit);
        self
    }

    pub fn with_ip_whitelist(mut self, ip_whitelist: Vec<String>) -> Self {
        self.ip_whitelist = Some(ip_whitelist);
        self
    }

    pub fn with_allowed_endpoints(mut self, allowed_endpoints: Vec<String>) -> Self {
        self.allowed_endpoints = Some(allowed_endpoints);
        self
    }

    pub fn has_scope(&self, required_scope: &str) -> bool {
        self.scopes.iter().any(|scope| scope == required_scope)
    }

    pub fn has_any_scope(&self, required_scopes: &[&str]) -> bool {
        required_scopes.iter().any(|scope| self.has_scope(scope))
    }

    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(expires_at) => chrono::Utc::now() > expires_at,
            None => false,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.is_active && !self.is_expired()
    }

    pub fn update_last_used(&mut self) {
        self.last_used_at = Some(chrono::Utc::now());
        self.usage_count += 1;
    }

    pub fn is_ip_allowed(&self, ip: &str) -> bool {
        match &self.ip_whitelist {
            Some(whitelist) => whitelist.contains(&ip.to_string()),
            None => true, // No whitelist means all IPs are allowed
        }
    }

    pub fn is_endpoint_allowed(&self, endpoint: &str) -> bool {
        match &self.allowed_endpoints {
            Some(endpoints) => endpoints.iter().any(|e| endpoint.starts_with(e)),
            None => true, // No restrictions means all endpoints are allowed
        }
    }
}

/// OAuth2 configuration for a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Config {
    /// OAuth2 provider name
    pub provider_name: String,

    /// Client ID
    pub client_id: String,

    /// Client secret
    pub client_secret: String,

    /// Authorization URL
    pub auth_url: String,

    /// Token URL
    pub token_url: String,

    /// User info URL
    pub user_info_url: String,

    /// Redirect URI
    pub redirect_uri: String,

    /// Default scopes
    pub default_scopes: Vec<String>,

    /// Additional parameters for authorization
    pub additional_params: HashMap<String, String>,
}

impl OAuth2Config {
    /// Create Google OAuth2 configuration
    pub fn google(client_id: String, client_secret: String, redirect_uri: String) -> Self {
        Self {
            provider_name: "google".to_string(),
            client_id,
            client_secret,
            auth_url: "https://accounts.google.com/o/oauth2/v2/auth".to_string(),
            token_url: "https://oauth2.googleapis.com/token".to_string(),
            user_info_url: "https://www.googleapis.com/oauth2/v2/userinfo".to_string(),
            redirect_uri,
            default_scopes: vec![
                "openid".to_string(),
                "profile".to_string(),
                "email".to_string(),
            ],
            additional_params: HashMap::new(),
        }
    }

    /// Create GitHub OAuth2 configuration
    pub fn github(client_id: String, client_secret: String, redirect_uri: String) -> Self {
        Self {
            provider_name: "github".to_string(),
            client_id,
            client_secret,
            auth_url: "https://github.com/login/oauth/authorize".to_string(),
            token_url: "https://github.com/login/oauth/access_token".to_string(),
            user_info_url: "https://api.github.com/user".to_string(),
            redirect_uri,
            default_scopes: vec!["user".to_string(), "user:email".to_string()],
            additional_params: HashMap::new(),
        }
    }

    /// Create Microsoft Azure OAuth2 configuration
    pub fn azure(
        client_id: String,
        client_secret: String,
        redirect_uri: String,
        tenant_id: String,
    ) -> Self {
        Self {
            provider_name: "azure".to_string(),
            client_id,
            client_secret,
            auth_url: format!(
                "https://login.microsoftonline.com/{}/oauth2/v2.0/authorize",
                tenant_id
            ),
            token_url: format!(
                "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
                tenant_id
            ),
            user_info_url: "https://graph.microsoft.com/v1.0/me".to_string(),
            redirect_uri,
            default_scopes: vec![
                "openid".to_string(),
                "profile".to_string(),
                "email".to_string(),
            ],
            additional_params: HashMap::new(),
        }
    }

    /// Create custom OAuth2 configuration
    pub fn custom(
        provider_name: String,
        client_id: String,
        client_secret: String,
        auth_url: String,
        token_url: String,
        user_info_url: String,
        redirect_uri: String,
        default_scopes: Vec<String>,
    ) -> Self {
        Self {
            provider_name,
            client_id,
            client_secret,
            auth_url,
            token_url,
            user_info_url,
            redirect_uri,
            default_scopes,
            additional_params: HashMap::new(),
        }
    }
}

/// OAuth2 access token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Token {
    /// Access token
    pub access_token: String,

    /// Token type (usually "Bearer")
    pub token_type: String,

    /// Expires in seconds
    pub expires_in: Option<u64>,

    /// Refresh token
    pub refresh_token: Option<String>,

    /// Scopes granted
    pub scope: Option<String>,

    /// Additional token data
    pub additional_data: HashMap<String, serde_json::Value>,

    /// When the token was issued
    pub issued_at: chrono::DateTime<chrono::Utc>,
}

impl OAuth2Token {
    pub fn new(access_token: String, token_type: String) -> Self {
        Self {
            access_token,
            token_type,
            expires_in: None,
            refresh_token: None,
            scope: None,
            additional_data: HashMap::new(),
            issued_at: chrono::Utc::now(),
        }
    }

    pub fn with_expiration(mut self, expires_in: u64) -> Self {
        self.expires_in = Some(expires_in);
        self
    }

    pub fn with_refresh_token(mut self, refresh_token: String) -> Self {
        self.refresh_token = Some(refresh_token);
        self
    }

    pub fn with_scope(mut self, scope: String) -> Self {
        self.scope = Some(scope);
        self
    }

    pub fn is_expired(&self) -> bool {
        match self.expires_in {
            Some(expires_in) => {
                let expires_at = self.issued_at + chrono::Duration::seconds(expires_in as i64);
                chrono::Utc::now() > expires_at
            },
            None => false,
        }
    }
}

/// OAuth2 user information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2UserInfo {
    /// User ID from the provider
    pub id: String,

    /// User email
    pub email: Option<String>,

    /// User name
    pub name: Option<String>,

    /// User display name
    pub display_name: Option<String>,

    /// User avatar URL
    pub avatar_url: Option<String>,

    /// Provider name
    pub provider: String,

    /// Raw user data from provider
    pub raw_data: serde_json::Value,
}

/// User account for authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// Unique user ID
    pub id: String,

    /// Username for authentication
    pub username: String,

    /// Hashed password
    pub password_hash: String,

    /// User email
    pub email: Option<String>,

    /// Full name
    pub full_name: Option<String>,

    /// User roles/permissions
    pub roles: Vec<String>,

    /// Account creation time
    pub created_at: SystemTime,

    /// Last login time
    pub last_login_at: Option<SystemTime>,

    /// Whether the account is active
    pub is_active: bool,
}

impl User {
    /// Create a new user with hashed password
    pub fn new(username: String, password: String) -> Self {
        let id = Uuid::new_v4().to_string();
        let password_hash = hash_password(&password);

        Self {
            id,
            username,
            password_hash,
            email: None,
            full_name: None,
            roles: vec!["user".to_string()],
            created_at: SystemTime::now(),
            last_login_at: None,
            is_active: true,
        }
    }

    /// Verify password against stored hash
    pub fn verify_password(&self, password: &str) -> bool {
        verify_password(password, &self.password_hash)
    }

    /// Update last login time
    pub fn update_last_login(&mut self) {
        self.last_login_at = Some(SystemTime::now());
    }
}

/// OAuth2 authorization state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2State {
    /// State parameter (CSRF protection)
    pub state: String,

    /// Redirect URI after authentication
    pub redirect_uri: String,

    /// Requested scopes
    pub scopes: Vec<String>,

    /// Additional parameters
    pub additional_params: HashMap<String, String>,

    /// State creation time
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl OAuth2State {
    pub fn new(redirect_uri: String, scopes: Vec<String>) -> Self {
        Self {
            state: Uuid::new_v4().to_string(),
            redirect_uri,
            scopes,
            additional_params: HashMap::new(),
            created_at: chrono::Utc::now(),
        }
    }

    pub fn is_expired(&self, max_age_seconds: i64) -> bool {
        let expires_at = self.created_at + chrono::Duration::seconds(max_age_seconds);
        chrono::Utc::now() > expires_at
    }
}

#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub secret_key: String,
    pub issuer: String,
    pub audience: String,
    pub algorithm: Algorithm,
    pub leeway: i64, // Clock skew tolerance in seconds

    /// OAuth2 provider configurations
    pub oauth2_providers: HashMap<String, OAuth2Config>,

    /// OAuth2 state max age in seconds
    pub oauth2_state_max_age: i64,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            secret_key: "your-secret-key".to_string(),
            issuer: "trustformers-serve".to_string(),
            audience: "trustformers-api".to_string(),
            algorithm: Algorithm::HS256,
            leeway: 60, // 1 minute
            oauth2_providers: HashMap::new(),
            oauth2_state_max_age: 600, // 10 minutes
        }
    }
}

#[derive(Clone)]
pub struct AuthService {
    config: AuthConfig,
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    validation: Validation,
    api_keys: std::sync::Arc<std::sync::RwLock<HashMap<String, ApiKey>>>,
    oauth2_states: std::sync::Arc<std::sync::RwLock<HashMap<String, OAuth2State>>>,
    users: std::sync::Arc<std::sync::RwLock<HashMap<String, User>>>,
    audit_logger: Option<AuditLogger>,
}

impl AuthService {
    pub fn new(config: AuthConfig) -> Self {
        let encoding_key = EncodingKey::from_secret(config.secret_key.as_ref());
        let decoding_key = DecodingKey::from_secret(config.secret_key.as_ref());

        let mut validation = Validation::new(config.algorithm);
        validation.set_issuer(&[&config.issuer]);
        validation.set_audience(&[&config.audience]);
        validation.leeway = config.leeway as u64;

        let mut users = HashMap::new();

        // Create default admin user for testing
        let mut admin_user = User::new("admin".to_string(), "admin123".to_string());
        admin_user.roles = vec!["admin".to_string(), "user".to_string()];
        users.insert(admin_user.username.clone(), admin_user);

        // Create test user
        let test_user = User::new("testuser".to_string(), "password123".to_string());
        users.insert(test_user.username.clone(), test_user);

        Self {
            config,
            encoding_key,
            decoding_key,
            validation,
            api_keys: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            oauth2_states: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
            users: std::sync::Arc::new(std::sync::RwLock::new(users)),
            audit_logger: None,
        }
    }

    pub fn with_audit_logger(mut self, audit_logger: AuditLogger) -> Self {
        self.audit_logger = Some(audit_logger);
        self
    }

    /// Authenticate a user with username and password
    pub fn authenticate_user(&self, username: &str, password: &str) -> Result<User, AuthError> {
        let users = self.users.read().unwrap();

        if let Some(user) = users.get(username) {
            if !user.is_active {
                return Err(AuthError::InvalidCredentials);
            }

            if user.verify_password(password) {
                let mut authenticated_user = user.clone();
                authenticated_user.update_last_login();

                // Update the user in the store with new last login time
                drop(users);
                let mut users_write = self.users.write().unwrap();
                if let Some(stored_user) = users_write.get_mut(username) {
                    stored_user.last_login_at = authenticated_user.last_login_at;
                }

                return Ok(authenticated_user);
            }
        }

        Err(AuthError::InvalidCredentials)
    }

    /// Create a new user account
    pub fn create_user(&self, username: String, password: String) -> Result<String, AuthError> {
        let mut users = self.users.write().unwrap();

        if users.contains_key(&username) {
            return Err(AuthError::InvalidCredentials); // User already exists
        }

        let user = User::new(username.clone(), password);
        let user_id = user.id.clone();
        users.insert(username, user);

        Ok(user_id)
    }

    pub fn create_token(&self, claims: &Claims) -> Result<String, AuthError> {
        let header = Header::new(self.config.algorithm);
        encode(&header, claims, &self.encoding_key).map_err(AuthError::InvalidToken)
    }

    pub fn verify_token(&self, token: &str) -> Result<Claims, AuthError> {
        let token_data = decode::<Claims>(token, &self.decoding_key, &self.validation)
            .map_err(AuthError::InvalidToken)?;

        let claims = token_data.claims;

        if claims.is_expired() {
            return Err(AuthError::TokenExpired);
        }

        Ok(claims)
    }

    pub fn extract_token_from_header<'a>(
        &self,
        auth_header: &'a str,
    ) -> Result<&'a str, AuthError> {
        if !auth_header.starts_with("Bearer ") {
            return Err(AuthError::InvalidHeaderFormat);
        }

        Ok(&auth_header[7..])
    }

    pub fn extract_api_key_from_header<'a>(
        &self,
        auth_header: &'a str,
    ) -> Result<&'a str, AuthError> {
        if auth_header.starts_with("Bearer ") {
            Ok(&auth_header[7..])
        } else if auth_header.starts_with("ApiKey ") {
            Ok(&auth_header[7..])
        } else {
            Ok(auth_header)
        }
    }

    pub fn create_api_key(&self, name: String, user_id: String, scopes: Vec<String>) -> ApiKey {
        let api_key = ApiKey::new(name, user_id.clone(), scopes);

        if let Ok(mut keys) = self.api_keys.write() {
            keys.insert(api_key.key.clone(), api_key.clone());
        }

        // Log audit event
        if let Some(audit_logger) = &self.audit_logger {
            let mut details = HashMap::new();
            details.insert("api_key_name".to_string(), api_key.name.clone());
            details.insert("scopes".to_string(), api_key.scopes.join(","));

            tokio::spawn({
                let audit_logger = audit_logger.clone();
                let api_key_id = api_key.id.clone();
                let user_id = user_id.clone();
                async move {
                    let _ = audit_logger
                        .log_api_key_event(
                            AuditEventType::ApiKeyCreated,
                            api_key_id,
                            user_id,
                            details,
                        )
                        .await;
                }
            });
        }

        api_key
    }

    pub fn revoke_api_key(&self, key: &str) -> Result<(), AuthError> {
        if let Ok(mut keys) = self.api_keys.write() {
            if let Some(api_key) = keys.get_mut(key) {
                api_key.is_active = false;

                // Log audit event
                if let Some(audit_logger) = &self.audit_logger {
                    let mut details = HashMap::new();
                    details.insert("api_key_name".to_string(), api_key.name.clone());

                    tokio::spawn({
                        let audit_logger = audit_logger.clone();
                        let api_key_id = api_key.id.clone();
                        let user_id = api_key.user_id.clone();
                        async move {
                            let _ = audit_logger
                                .log_api_key_event(
                                    AuditEventType::ApiKeyRevoked,
                                    api_key_id,
                                    user_id,
                                    details,
                                )
                                .await;
                        }
                    });
                }

                Ok(())
            } else {
                Err(AuthError::InvalidApiKey)
            }
        } else {
            Err(AuthError::InvalidApiKey)
        }
    }

    pub fn verify_api_key(&self, key: &str) -> Result<ApiKey, AuthError> {
        if let Ok(mut keys) = self.api_keys.write() {
            if let Some(api_key) = keys.get_mut(key) {
                if !api_key.is_valid() {
                    if api_key.is_expired() {
                        // Log expired API key usage attempt
                        if let Some(audit_logger) = &self.audit_logger {
                            let mut details = HashMap::new();
                            details.insert("api_key_name".to_string(), api_key.name.clone());

                            tokio::spawn({
                                let audit_logger = audit_logger.clone();
                                let api_key_id = api_key.id.clone();
                                let user_id = api_key.user_id.clone();
                                async move {
                                    let _ = audit_logger
                                        .log_api_key_event(
                                            AuditEventType::ApiKeyExpired,
                                            api_key_id,
                                            user_id,
                                            details,
                                        )
                                        .await;
                                }
                            });
                        }
                        return Err(AuthError::ApiKeyExpired);
                    } else {
                        return Err(AuthError::ApiKeyRevoked);
                    }
                }

                api_key.update_last_used();

                // Log successful API key usage
                if let Some(audit_logger) = &self.audit_logger {
                    let mut details = HashMap::new();
                    details.insert("api_key_name".to_string(), api_key.name.clone());

                    tokio::spawn({
                        let audit_logger = audit_logger.clone();
                        let api_key_id = api_key.id.clone();
                        let user_id = api_key.user_id.clone();
                        async move {
                            let _ = audit_logger
                                .log_api_key_event(
                                    AuditEventType::ApiKeyUsed,
                                    api_key_id,
                                    user_id,
                                    details,
                                )
                                .await;
                        }
                    });
                }

                Ok(api_key.clone())
            } else {
                Err(AuthError::InvalidApiKey)
            }
        } else {
            Err(AuthError::InvalidApiKey)
        }
    }

    pub fn list_api_keys(&self, user_id: &str) -> Vec<ApiKey> {
        if let Ok(keys) = self.api_keys.read() {
            keys.values().filter(|api_key| api_key.user_id == user_id).cloned().collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_api_key(&self, key: &str) -> Option<ApiKey> {
        if let Ok(keys) = self.api_keys.read() {
            keys.get(key).cloned()
        } else {
            None
        }
    }

    /// OAuth2 methods

    /// Add OAuth2 provider configuration
    pub fn add_oauth2_provider(&mut self, provider_name: String, config: OAuth2Config) {
        self.config.oauth2_providers.insert(provider_name, config);
    }

    /// Generate OAuth2 authorization URL
    pub fn generate_oauth2_auth_url(
        &self,
        provider_name: &str,
        redirect_uri: String,
        scopes: Option<Vec<String>>,
        additional_params: Option<HashMap<String, String>>,
    ) -> Result<String, AuthError> {
        let provider_config = self.config.oauth2_providers.get(provider_name).ok_or_else(|| {
            AuthError::OAuth2ProviderError(format!("Provider '{}' not found", provider_name))
        })?;

        let scopes = scopes.unwrap_or_else(|| provider_config.default_scopes.clone());
        let state = OAuth2State::new(redirect_uri, scopes.clone());

        // Store state for later validation
        if let Ok(mut states) = self.oauth2_states.write() {
            states.insert(state.state.clone(), state.clone());
        }

        // Build authorization URL
        let mut url = url::Url::parse(&provider_config.auth_url)
            .map_err(|e| AuthError::OAuth2AuthorizationError(format!("Invalid auth URL: {}", e)))?;

        url.query_pairs_mut()
            .append_pair("client_id", &provider_config.client_id)
            .append_pair("redirect_uri", &provider_config.redirect_uri)
            .append_pair("response_type", "code")
            .append_pair("state", &state.state)
            .append_pair("scope", &scopes.join(" "));

        // Add additional parameters
        if let Some(params) = additional_params {
            for (key, value) in params {
                url.query_pairs_mut().append_pair(&key, &value);
            }
        }

        // Add provider-specific parameters
        for (key, value) in &provider_config.additional_params {
            url.query_pairs_mut().append_pair(key, value);
        }

        Ok(url.to_string())
    }

    /// Exchange authorization code for access token
    pub async fn exchange_oauth2_code(
        &self,
        provider_name: &str,
        code: String,
        state: String,
    ) -> Result<(OAuth2Token, OAuth2UserInfo), AuthError> {
        let provider_config = self.config.oauth2_providers.get(provider_name).ok_or_else(|| {
            AuthError::OAuth2ProviderError(format!("Provider '{}' not found", provider_name))
        })?;

        // Validate state
        let _oauth2_state = self.validate_oauth2_state(&state)?;

        // Exchange code for token
        let token_response = self.request_oauth2_token(provider_config, code).await?;
        let oauth2_token = self.parse_token_response(token_response).await?;

        // Get user info
        let user_info = self.get_oauth2_user_info(provider_config, &oauth2_token).await?;

        // Clean up state
        if let Ok(mut states) = self.oauth2_states.write() {
            states.remove(&state);
        }

        Ok((oauth2_token, user_info))
    }

    /// Refresh OAuth2 token
    pub async fn refresh_oauth2_token(
        &self,
        provider_name: &str,
        refresh_token: String,
    ) -> Result<OAuth2Token, AuthError> {
        let provider_config = self.config.oauth2_providers.get(provider_name).ok_or_else(|| {
            AuthError::OAuth2ProviderError(format!("Provider '{}' not found", provider_name))
        })?;

        let client = reqwest::Client::new();
        let mut params = HashMap::new();
        params.insert("grant_type", "refresh_token");
        params.insert("refresh_token", &refresh_token);
        params.insert("client_id", &provider_config.client_id);
        params.insert("client_secret", &provider_config.client_secret);

        let response = client
            .post(&provider_config.token_url)
            .form(&params)
            .send()
            .await
            .map_err(|e| AuthError::OAuth2TokenError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AuthError::OAuth2TokenError(format!(
                "Token refresh failed: {}",
                error_text
            )));
        }

        let token_response = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| AuthError::OAuth2TokenError(format!("Invalid response: {}", e)))?;

        self.parse_token_response(token_response).await
    }

    /// Validate OAuth2 state parameter
    fn validate_oauth2_state(&self, state: &str) -> Result<OAuth2State, AuthError> {
        let oauth2_state = {
            let states = self.oauth2_states.read().unwrap();
            states.get(state).cloned()
        };

        match oauth2_state {
            Some(oauth2_state) => {
                if oauth2_state.is_expired(self.config.oauth2_state_max_age) {
                    // Clean up expired state
                    if let Ok(mut states) = self.oauth2_states.write() {
                        states.remove(state);
                    }
                    Err(AuthError::InvalidOAuth2State)
                } else {
                    Ok(oauth2_state)
                }
            },
            None => Err(AuthError::InvalidOAuth2State),
        }
    }

    /// Request access token from provider
    async fn request_oauth2_token(
        &self,
        provider_config: &OAuth2Config,
        code: String,
    ) -> Result<serde_json::Value, AuthError> {
        let client = reqwest::Client::new();
        let mut params = HashMap::new();
        params.insert("grant_type", "authorization_code");
        params.insert("code", &code);
        params.insert("client_id", &provider_config.client_id);
        params.insert("client_secret", &provider_config.client_secret);
        params.insert("redirect_uri", &provider_config.redirect_uri);

        let response = client
            .post(&provider_config.token_url)
            .form(&params)
            .send()
            .await
            .map_err(|e| AuthError::OAuth2TokenError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AuthError::OAuth2TokenError(format!(
                "Token exchange failed: {}",
                error_text
            )));
        }

        response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| AuthError::OAuth2TokenError(format!("Invalid response: {}", e)))
    }

    /// Parse token response into OAuth2Token
    async fn parse_token_response(
        &self,
        response: serde_json::Value,
    ) -> Result<OAuth2Token, AuthError> {
        let access_token = response
            .get("access_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AuthError::OAuth2TokenError("Missing access_token".to_string()))?
            .to_string();

        let token_type = response
            .get("token_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Bearer")
            .to_string();

        let mut token = OAuth2Token::new(access_token, token_type);

        if let Some(expires_in) = response.get("expires_in").and_then(|v| v.as_u64()) {
            token = token.with_expiration(expires_in);
        }

        if let Some(refresh_token) = response.get("refresh_token").and_then(|v| v.as_str()) {
            token = token.with_refresh_token(refresh_token.to_string());
        }

        if let Some(scope) = response.get("scope").and_then(|v| v.as_str()) {
            token = token.with_scope(scope.to_string());
        }

        // Store additional data
        if let serde_json::Value::Object(map) = response {
            for (key, value) in map {
                if ![
                    "access_token",
                    "token_type",
                    "expires_in",
                    "refresh_token",
                    "scope",
                ]
                .contains(&key.as_str())
                {
                    token.additional_data.insert(key, value);
                }
            }
        }

        Ok(token)
    }

    /// Get user information from OAuth2 provider
    async fn get_oauth2_user_info(
        &self,
        provider_config: &OAuth2Config,
        token: &OAuth2Token,
    ) -> Result<OAuth2UserInfo, AuthError> {
        let client = reqwest::Client::new();
        let response = client
            .get(&provider_config.user_info_url)
            .bearer_auth(&token.access_token)
            .send()
            .await
            .map_err(|e| {
                AuthError::OAuth2ProviderError(format!("User info request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AuthError::OAuth2ProviderError(format!(
                "User info failed: {}",
                error_text
            )));
        }

        let user_data = response.json::<serde_json::Value>().await.map_err(|e| {
            AuthError::OAuth2ProviderError(format!("Invalid user info response: {}", e))
        })?;

        // Parse user info based on provider
        let user_info = match provider_config.provider_name.as_str() {
            "google" => self.parse_google_user_info(user_data),
            "github" => self.parse_github_user_info(user_data),
            "azure" => self.parse_azure_user_info(user_data),
            _ => self.parse_generic_user_info(user_data, &provider_config.provider_name),
        }?;

        Ok(user_info)
    }

    /// Parse Google user info
    fn parse_google_user_info(&self, data: serde_json::Value) -> Result<OAuth2UserInfo, AuthError> {
        let id = data
            .get("id")
            .or_else(|| data.get("sub"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| AuthError::OAuth2ProviderError("Missing user ID".to_string()))?
            .to_string();

        let email = data.get("email").and_then(|v| v.as_str()).map(|s| s.to_string());
        let name = data.get("name").and_then(|v| v.as_str()).map(|s| s.to_string());
        let display_name = data.get("given_name").and_then(|v| v.as_str()).map(|s| s.to_string());
        let avatar_url = data.get("picture").and_then(|v| v.as_str()).map(|s| s.to_string());

        Ok(OAuth2UserInfo {
            id,
            email,
            name,
            display_name,
            avatar_url,
            provider: "google".to_string(),
            raw_data: data,
        })
    }

    /// Parse GitHub user info
    fn parse_github_user_info(&self, data: serde_json::Value) -> Result<OAuth2UserInfo, AuthError> {
        let id = data
            .get("id")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AuthError::OAuth2ProviderError("Missing user ID".to_string()))?
            .to_string();

        let email = data.get("email").and_then(|v| v.as_str()).map(|s| s.to_string());
        let name = data.get("name").and_then(|v| v.as_str()).map(|s| s.to_string());
        let display_name = data.get("login").and_then(|v| v.as_str()).map(|s| s.to_string());
        let avatar_url = data.get("avatar_url").and_then(|v| v.as_str()).map(|s| s.to_string());

        Ok(OAuth2UserInfo {
            id,
            email,
            name,
            display_name,
            avatar_url,
            provider: "github".to_string(),
            raw_data: data,
        })
    }

    /// Parse Azure user info
    fn parse_azure_user_info(&self, data: serde_json::Value) -> Result<OAuth2UserInfo, AuthError> {
        let id = data
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AuthError::OAuth2ProviderError("Missing user ID".to_string()))?
            .to_string();

        let email = data
            .get("mail")
            .or_else(|| data.get("userPrincipalName"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let name = data.get("displayName").and_then(|v| v.as_str()).map(|s| s.to_string());
        let display_name = data.get("givenName").and_then(|v| v.as_str()).map(|s| s.to_string());
        let avatar_url = None; // Azure doesn't provide avatar URL by default

        Ok(OAuth2UserInfo {
            id,
            email,
            name,
            display_name,
            avatar_url,
            provider: "azure".to_string(),
            raw_data: data,
        })
    }

    /// Parse generic user info
    fn parse_generic_user_info(
        &self,
        data: serde_json::Value,
        provider: &str,
    ) -> Result<OAuth2UserInfo, AuthError> {
        let id = data
            .get("id")
            .or_else(|| data.get("sub"))
            .or_else(|| data.get("user_id"))
            .and_then(|v| {
                v.as_str().or_else(|| {
                    v.as_u64().map(|n| Box::leak(n.to_string().into_boxed_str()) as &str)
                })
            })
            .ok_or_else(|| AuthError::OAuth2ProviderError("Missing user ID".to_string()))?
            .to_string();

        let email = data.get("email").and_then(|v| v.as_str()).map(|s| s.to_string());
        let name = data.get("name").and_then(|v| v.as_str()).map(|s| s.to_string());
        let display_name = data
            .get("display_name")
            .or_else(|| data.get("username"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let avatar_url = data
            .get("avatar_url")
            .or_else(|| data.get("picture"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(OAuth2UserInfo {
            id,
            email,
            name,
            display_name,
            avatar_url,
            provider: provider.to_string(),
            raw_data: data,
        })
    }

    /// Create JWT token from OAuth2 user info
    pub fn create_token_from_oauth2_user(
        &self,
        user_info: &OAuth2UserInfo,
        scopes: Vec<String>,
    ) -> Result<String, AuthError> {
        let claims = Claims::new(
            format!("{}:{}", user_info.provider, user_info.id),
            self.config.issuer.clone(),
            self.config.audience.clone(),
            scopes,
            3600, // 1 hour default
        );

        self.create_token(&claims)
    }

    /// Clean up expired OAuth2 states
    pub fn cleanup_expired_oauth2_states(&self) {
        if let Ok(mut states) = self.oauth2_states.write() {
            let max_age = self.config.oauth2_state_max_age;
            states.retain(|_, state| !state.is_expired(max_age));
        }
    }
}

#[derive(Clone)]
pub struct AuthMiddleware {
    auth_service: AuthService,
    required_scopes: HashSet<String>,
    skip_paths: HashSet<String>,
}

impl AuthMiddleware {
    pub fn new(auth_service: AuthService) -> Self {
        Self {
            auth_service,
            required_scopes: HashSet::new(),
            skip_paths: HashSet::from([
                "/health".to_string(),
                "/metrics".to_string(),
                "/".to_string(),
            ]),
        }
    }

    pub fn with_required_scopes(mut self, scopes: Vec<String>) -> Self {
        self.required_scopes = scopes.into_iter().collect();
        self
    }

    pub fn with_skip_paths(mut self, paths: Vec<String>) -> Self {
        self.skip_paths = paths.into_iter().collect();
        self
    }

    pub async fn middleware(
        State(auth): State<AuthMiddleware>,
        mut request: Request,
        next: Next,
    ) -> Result<Response, StatusCode> {
        // Skip authentication for certain paths
        if auth.skip_paths.contains(request.uri().path()) {
            return Ok(next.run(request).await);
        }

        // Extract authorization header
        let auth_header = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .ok_or(StatusCode::UNAUTHORIZED)?;

        // Try JWT authentication first
        if auth_header.starts_with("Bearer ") {
            let token = auth
                .auth_service
                .extract_token_from_header(auth_header)
                .map_err(|_| StatusCode::UNAUTHORIZED)?;

            match auth.auth_service.verify_token(token) {
                Ok(claims) => {
                    // Check required scopes for JWT
                    if !auth.required_scopes.is_empty() {
                        let required_scopes: Vec<&str> =
                            auth.required_scopes.iter().map(|s| s.as_str()).collect();
                        if !claims.has_any_scope(&required_scopes) {
                            return Err(StatusCode::FORBIDDEN);
                        }
                    }

                    // Add claims to request extensions for use by handlers
                    request.extensions_mut().insert(claims);
                    return Ok(next.run(request).await);
                },
                Err(_) => {
                    // If JWT verification fails, try API key authentication
                },
            }
        }

        // Try API key authentication
        let api_key = auth
            .auth_service
            .extract_api_key_from_header(auth_header)
            .map_err(|_| StatusCode::UNAUTHORIZED)?;

        let api_key_data = auth
            .auth_service
            .verify_api_key(api_key)
            .map_err(|_| StatusCode::UNAUTHORIZED)?;

        // Check IP whitelist
        if let Some(client_ip) = request
            .headers()
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split(',').next())
            .map(|s| s.trim())
            .or_else(|| request.headers().get("x-real-ip").and_then(|v| v.to_str().ok()))
        {
            if !api_key_data.is_ip_allowed(client_ip) {
                return Err(StatusCode::FORBIDDEN);
            }
        }

        // Check endpoint restrictions
        if !api_key_data.is_endpoint_allowed(request.uri().path()) {
            return Err(StatusCode::FORBIDDEN);
        }

        // Check required scopes for API key
        if !auth.required_scopes.is_empty() {
            let required_scopes: Vec<&str> =
                auth.required_scopes.iter().map(|s| s.as_str()).collect();
            if !api_key_data.has_any_scope(&required_scopes) {
                return Err(StatusCode::FORBIDDEN);
            }
        }

        // Create claims-like data for API key
        let claims = Claims::new(
            api_key_data.user_id.clone(),
            "trustformers-serve".to_string(),
            "trustformers-api".to_string(),
            api_key_data.scopes.clone(),
            3600, // Default 1 hour
        );

        // Add both API key data and claims to request extensions
        request.extensions_mut().insert(api_key_data);
        request.extensions_mut().insert(claims);

        Ok(next.run(request).await)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenRequest {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateApiKeyRequest {
    pub name: String,
    pub scopes: Vec<String>,
    pub expires_in_days: Option<u32>,
    pub rate_limit: Option<ApiKeyRateLimit>,
    pub ip_whitelist: Option<Vec<String>>,
    pub allowed_endpoints: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateApiKeyResponse {
    pub id: String,
    pub key: String,
    pub name: String,
    pub scopes: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListApiKeysResponse {
    pub api_keys: Vec<ApiKeyInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiKeyInfo {
    pub id: String,
    pub name: String,
    pub scopes: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    pub last_used_at: Option<chrono::DateTime<chrono::Utc>>,
    pub is_active: bool,
    pub usage_count: u64,
    pub ip_whitelist: Option<Vec<String>>,
    pub allowed_endpoints: Option<Vec<String>>,
}

impl From<ApiKey> for ApiKeyInfo {
    fn from(api_key: ApiKey) -> Self {
        Self {
            id: api_key.id,
            name: api_key.name,
            scopes: api_key.scopes,
            created_at: api_key.created_at,
            expires_at: api_key.expires_at,
            last_used_at: api_key.last_used_at,
            is_active: api_key.is_active,
            usage_count: api_key.usage_count,
            ip_whitelist: api_key.ip_whitelist,
            allowed_endpoints: api_key.allowed_endpoints,
        }
    }
}

pub async fn login_handler(
    State(auth_service): State<AuthService>,
    axum::Json(request): axum::Json<TokenRequest>,
) -> Result<axum::Json<TokenResponse>, StatusCode> {
    // Validate input
    if request.username.is_empty() || request.password.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Authenticate user
    let user = match auth_service.authenticate_user(&request.username, &request.password) {
        Ok(user) => user,
        Err(AuthError::InvalidCredentials) => {
            return Err(StatusCode::UNAUTHORIZED);
        },
        Err(_) => {
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        },
    };

    // Create JWT token with user-specific permissions
    let expires_in = 3600; // 1 hour
    let mut scopes = vec!["inference:read".to_string(), "inference:write".to_string()];

    // Add admin scopes if user has admin role
    if user.roles.contains(&"admin".to_string()) {
        scopes.extend(vec![
            "admin:read".to_string(),
            "admin:write".to_string(),
            "users:manage".to_string(),
        ]);
    }

    let claims = Claims::new(
        user.username,
        "trustformers-serve".to_string(),
        "trustformers-api".to_string(),
        scopes,
        expires_in,
    );

    let token = auth_service
        .create_token(&claims)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let response = TokenResponse {
        access_token: token,
        token_type: "Bearer".to_string(),
        expires_in,
    };

    Ok(axum::Json(response))
}

pub async fn create_api_key_handler(
    State(auth_service): State<AuthService>,
    axum::Extension(claims): axum::Extension<Claims>,
    axum::Json(request): axum::Json<CreateApiKeyRequest>,
) -> Result<axum::Json<CreateApiKeyResponse>, StatusCode> {
    // Validate request
    if request.name.is_empty() || request.scopes.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Create API key
    let mut api_key = auth_service.create_api_key(request.name, claims.sub.clone(), request.scopes);

    // Set expiration if provided
    if let Some(expires_in_days) = request.expires_in_days {
        let expires_at = chrono::Utc::now() + chrono::Duration::days(expires_in_days as i64);
        api_key.expires_at = Some(expires_at);
    }

    // Set rate limit if provided
    if let Some(rate_limit) = request.rate_limit {
        api_key.rate_limit = Some(rate_limit);
    }

    // Set IP whitelist if provided
    if let Some(ip_whitelist) = request.ip_whitelist {
        api_key.ip_whitelist = Some(ip_whitelist);
    }

    // Set allowed endpoints if provided
    if let Some(allowed_endpoints) = request.allowed_endpoints {
        api_key.allowed_endpoints = Some(allowed_endpoints);
    }

    let response = CreateApiKeyResponse {
        id: api_key.id.clone(),
        key: api_key.key.clone(), // Only returned once during creation
        name: api_key.name.clone(),
        scopes: api_key.scopes.clone(),
        created_at: api_key.created_at,
        expires_at: api_key.expires_at,
    };

    Ok(axum::Json(response))
}

pub async fn list_api_keys_handler(
    State(auth_service): State<AuthService>,
    axum::Extension(claims): axum::Extension<Claims>,
) -> Result<axum::Json<ListApiKeysResponse>, StatusCode> {
    let api_keys = auth_service.list_api_keys(&claims.sub);
    let api_key_infos: Vec<ApiKeyInfo> = api_keys.into_iter().map(|k| k.into()).collect();

    let response = ListApiKeysResponse {
        api_keys: api_key_infos,
    };

    Ok(axum::Json(response))
}

pub async fn revoke_api_key_handler(
    State(auth_service): State<AuthService>,
    axum::Extension(claims): axum::Extension<Claims>,
    axum::extract::Path(key_id): axum::extract::Path<String>,
) -> Result<StatusCode, StatusCode> {
    // First verify the user owns this API key
    let api_keys = auth_service.list_api_keys(&claims.sub);
    let api_key = api_keys.iter().find(|k| k.id == key_id).ok_or(StatusCode::NOT_FOUND)?;

    // Revoke the API key
    auth_service
        .revoke_api_key(&api_key.key)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(StatusCode::NO_CONTENT)
}

/// OAuth2 request and response types

#[derive(Debug, Deserialize)]
pub struct OAuth2AuthRequest {
    pub provider: String,
    pub redirect_uri: Option<String>,
    pub scopes: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct OAuth2AuthResponse {
    pub auth_url: String,
    pub state: String,
}

#[derive(Debug, Deserialize)]
pub struct OAuth2CallbackQuery {
    pub code: Option<String>,
    pub state: Option<String>,
    pub error: Option<String>,
    pub error_description: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OAuth2TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: usize,
    pub user_info: OAuth2UserInfo,
}

/// OAuth2 HTTP handlers

/// Initiate OAuth2 authorization
pub async fn oauth2_authorize_handler(
    State(auth_service): State<AuthService>,
    Query(request): Query<OAuth2AuthRequest>,
) -> Result<axum::Json<OAuth2AuthResponse>, StatusCode> {
    let redirect_uri = request.redirect_uri.unwrap_or_else(|| {
        // Default redirect URI - should be configured properly in production
        "http://localhost:8080/auth/oauth2/callback".to_string()
    });

    let auth_url = auth_service
        .generate_oauth2_auth_url(&request.provider, redirect_uri, request.scopes, None)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Extract state from URL for response
    let url = url::Url::parse(&auth_url).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let state = url
        .query_pairs()
        .find(|(key, _)| key == "state")
        .map(|(_, value)| value.to_string())
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    let response = OAuth2AuthResponse { auth_url, state };

    Ok(axum::Json(response))
}

/// Handle OAuth2 callback
pub async fn oauth2_callback_handler(
    State(auth_service): State<AuthService>,
    axum::extract::Path(provider): axum::extract::Path<String>,
    Query(query): Query<OAuth2CallbackQuery>,
) -> Result<axum::Json<OAuth2TokenResponse>, StatusCode> {
    // Check for OAuth2 error
    if let Some(error) = query.error {
        tracing::error!(
            "OAuth2 error: {} - {}",
            error,
            query.error_description.unwrap_or_default()
        );
        return Err(StatusCode::BAD_REQUEST);
    }

    let code = query.code.ok_or(StatusCode::BAD_REQUEST)?;
    let state = query.state.ok_or(StatusCode::BAD_REQUEST)?;

    // Exchange code for token and get user info
    let (_oauth2_token, user_info) =
        auth_service.exchange_oauth2_code(&provider, code, state).await.map_err(|e| {
            tracing::error!("OAuth2 token exchange failed: {}", e);
            StatusCode::UNAUTHORIZED
        })?;

    // Create JWT token for the user
    let jwt_token = auth_service
        .create_token_from_oauth2_user(
            &user_info,
            vec!["inference:read".to_string(), "inference:write".to_string()],
        )
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let response = OAuth2TokenResponse {
        access_token: jwt_token,
        token_type: "Bearer".to_string(),
        expires_in: 3600,
        user_info,
    };

    Ok(axum::Json(response))
}

/// OAuth2 token refresh handler
pub async fn oauth2_refresh_handler(
    State(auth_service): State<AuthService>,
    axum::extract::Path(provider): axum::extract::Path<String>,
    axum::Json(request): axum::Json<OAuth2RefreshRequest>,
) -> Result<axum::Json<OAuth2RefreshResponse>, StatusCode> {
    let refreshed_token = auth_service
        .refresh_oauth2_token(&provider, request.refresh_token)
        .await
        .map_err(|_| StatusCode::UNAUTHORIZED)?;

    let response = OAuth2RefreshResponse {
        access_token: refreshed_token.access_token,
        token_type: refreshed_token.token_type,
        expires_in: refreshed_token.expires_in,
        refresh_token: refreshed_token.refresh_token,
    };

    Ok(axum::Json(response))
}

/// List available OAuth2 providers
pub async fn oauth2_providers_handler(
    State(auth_service): State<AuthService>,
) -> Result<axum::Json<OAuth2ProvidersResponse>, StatusCode> {
    let providers: Vec<OAuth2ProviderInfo> = auth_service
        .config
        .oauth2_providers
        .iter()
        .map(|(name, config)| OAuth2ProviderInfo {
            name: name.clone(),
            display_name: config.provider_name.clone(),
            scopes: config.default_scopes.clone(),
        })
        .collect();

    let response = OAuth2ProvidersResponse { providers };
    Ok(axum::Json(response))
}

#[derive(Debug, Deserialize)]
pub struct OAuth2RefreshRequest {
    pub refresh_token: String,
}

#[derive(Debug, Serialize)]
pub struct OAuth2RefreshResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub refresh_token: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OAuth2ProvidersResponse {
    pub providers: Vec<OAuth2ProviderInfo>,
}

#[derive(Debug, Serialize)]
pub struct OAuth2ProviderInfo {
    pub name: String,
    pub display_name: String,
    pub scopes: Vec<String>,
}
