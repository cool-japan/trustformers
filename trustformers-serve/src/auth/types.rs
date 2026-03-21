//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::{hash_password, verify_password};
use crate::audit::{AuditEventType, AuditLogger};
use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::Response,
};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::SystemTime;
use thiserror::Error;
use uuid::Uuid;

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
            None => true,
        }
    }
    pub fn is_endpoint_allowed(&self, endpoint: &str) -> bool {
        match &self.allowed_endpoints {
            Some(endpoints) => endpoints.iter().any(|e| endpoint.starts_with(e)),
            None => true,
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct ListApiKeysResponse {
    pub api_keys: Vec<ApiKeyInfo>,
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
#[derive(Debug, Serialize)]
pub struct OAuth2RefreshResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub refresh_token: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct OAuth2ProviderInfo {
    pub name: String,
    pub display_name: String,
    pub scopes: Vec<String>,
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
#[derive(Debug, Deserialize)]
pub struct OAuth2RefreshRequest {
    pub refresh_token: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub exp: usize,
    pub iat: usize,
    pub iss: String,
    pub aud: String,
    pub scopes: Vec<String>,
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
pub struct ApiKeyRateLimit {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub requests_per_day: u32,
}
#[derive(Debug, Serialize)]
pub struct OAuth2ProvidersResponse {
    pub providers: Vec<OAuth2ProviderInfo>,
}
#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub secret_key: String,
    pub issuer: String,
    pub audience: String,
    pub algorithm: Algorithm,
    pub leeway: i64,
    /// OAuth2 provider configurations
    pub oauth2_providers: HashMap<String, OAuth2Config>,
    /// OAuth2 state max age in seconds
    pub oauth2_state_max_age: i64,
}
/// OAuth2 request and response types
#[derive(Debug, Deserialize)]
pub struct OAuth2AuthRequest {
    pub provider: String,
    pub redirect_uri: Option<String>,
    pub scopes: Option<Vec<String>>,
}
#[derive(Debug, Serialize)]
pub struct OAuth2TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: usize,
    pub user_info: OAuth2UserInfo,
}
#[derive(Clone)]
pub struct AuthService {
    pub(crate) config: AuthConfig,
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
        let mut admin_user = User::new("admin".to_string(), "admin123".to_string());
        admin_user.roles = vec!["admin".to_string(), "user".to_string()];
        users.insert(admin_user.username.clone(), admin_user);
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
        let users = self.users.read().expect("users lock should not be poisoned");
        if let Some(user) = users.get(username) {
            if !user.is_active {
                return Err(AuthError::InvalidCredentials);
            }
            if user.verify_password(password) {
                let mut authenticated_user = user.clone();
                authenticated_user.update_last_login();
                drop(users);
                let mut users_write =
                    self.users.write().expect("users lock should not be poisoned");
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
        let mut users = self.users.write().expect("users lock should not be poisoned");
        if users.contains_key(&username) {
            return Err(AuthError::InvalidCredentials);
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
        if let Ok(mut states) = self.oauth2_states.write() {
            states.insert(state.state.clone(), state.clone());
        }
        let mut url = url::Url::parse(&provider_config.auth_url)
            .map_err(|e| AuthError::OAuth2AuthorizationError(format!("Invalid auth URL: {}", e)))?;
        url.query_pairs_mut()
            .append_pair("client_id", &provider_config.client_id)
            .append_pair("redirect_uri", &provider_config.redirect_uri)
            .append_pair("response_type", "code")
            .append_pair("state", &state.state)
            .append_pair("scope", &scopes.join(" "));
        if let Some(params) = additional_params {
            for (key, value) in params {
                url.query_pairs_mut().append_pair(&key, &value);
            }
        }
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
        let _oauth2_state = self.validate_oauth2_state(&state)?;
        let token_response = self.request_oauth2_token(provider_config, code).await?;
        let oauth2_token = self.parse_token_response(token_response).await?;
        let user_info = self.get_oauth2_user_info(provider_config, &oauth2_token).await?;
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
            let states =
                self.oauth2_states.read().expect("oauth2_states lock should not be poisoned");
            states.get(state).cloned()
        };
        match oauth2_state {
            Some(oauth2_state) => {
                if oauth2_state.is_expired(self.config.oauth2_state_max_age) {
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
        let avatar_url = None;
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
            3600,
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
#[derive(Debug, Serialize)]
pub struct OAuth2AuthResponse {
    pub auth_url: String,
    pub state: String,
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
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateApiKeyResponse {
    pub id: String,
    pub key: String,
    pub name: String,
    pub scopes: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
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
        if auth.skip_paths.contains(request.uri().path()) {
            return Ok(next.run(request).await);
        }
        let auth_header = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .ok_or(StatusCode::UNAUTHORIZED)?;
        if auth_header.starts_with("Bearer ") {
            let token = auth
                .auth_service
                .extract_token_from_header(auth_header)
                .map_err(|_| StatusCode::UNAUTHORIZED)?;
            match auth.auth_service.verify_token(token) {
                Ok(claims) => {
                    if !auth.required_scopes.is_empty() {
                        let required_scopes: Vec<&str> =
                            auth.required_scopes.iter().map(|s| s.as_str()).collect();
                        if !claims.has_any_scope(&required_scopes) {
                            return Err(StatusCode::FORBIDDEN);
                        }
                    }
                    request.extensions_mut().insert(claims);
                    return Ok(next.run(request).await);
                },
                Err(_) => {},
            }
        }
        let api_key = auth
            .auth_service
            .extract_api_key_from_header(auth_header)
            .map_err(|_| StatusCode::UNAUTHORIZED)?;
        let api_key_data = auth
            .auth_service
            .verify_api_key(api_key)
            .map_err(|_| StatusCode::UNAUTHORIZED)?;
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
        if !api_key_data.is_endpoint_allowed(request.uri().path()) {
            return Err(StatusCode::FORBIDDEN);
        }
        if !auth.required_scopes.is_empty() {
            let required_scopes: Vec<&str> =
                auth.required_scopes.iter().map(|s| s.as_str()).collect();
            if !api_key_data.has_any_scope(&required_scopes) {
                return Err(StatusCode::FORBIDDEN);
            }
        }
        let claims = Claims::new(
            api_key_data.user_id.clone(),
            "trustformers-serve".to_string(),
            "trustformers-api".to_string(),
            api_key_data.scopes.clone(),
            3600,
        );
        request.extensions_mut().insert(api_key_data);
        request.extensions_mut().insert(claims);
        Ok(next.run(request).await)
    }
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
#[derive(Debug, Deserialize)]
pub struct OAuth2CallbackQuery {
    pub code: Option<String>,
    pub state: Option<String>,
    pub error: Option<String>,
    pub error_description: Option<String>,
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
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: usize,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenRequest {
    pub username: String,
    pub password: String,
}
