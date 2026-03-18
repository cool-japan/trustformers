//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use axum::{
    extract::{Query, State},
    http::StatusCode,
};
use sha2::{Digest, Sha256};

use super::types::{
    ApiKeyInfo, AuthError, AuthService, Claims, CreateApiKeyRequest, CreateApiKeyResponse,
    ListApiKeysResponse, OAuth2AuthRequest, OAuth2AuthResponse, OAuth2CallbackQuery,
    OAuth2ProviderInfo, OAuth2ProvidersResponse, OAuth2RefreshRequest, OAuth2RefreshResponse,
    OAuth2TokenResponse, TokenRequest, TokenResponse,
};

/// Hash a password using SHA-256 with salt
pub(crate) fn hash_password(password: &str) -> String {
    let salt = "trustformers_salt_2024";
    let mut hasher = Sha256::new();
    hasher.update(format!("{}{}", password, salt));
    format!("{:x}", hasher.finalize())
}
/// Verify a password against its hash
pub(crate) fn verify_password(password: &str, hash: &str) -> bool {
    hash_password(password) == hash
}
pub async fn login_handler(
    State(auth_service): State<AuthService>,
    axum::Json(request): axum::Json<TokenRequest>,
) -> Result<axum::Json<TokenResponse>, StatusCode> {
    if request.username.is_empty() || request.password.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    let user = match auth_service.authenticate_user(&request.username, &request.password) {
        Ok(user) => user,
        Err(AuthError::InvalidCredentials) => {
            return Err(StatusCode::UNAUTHORIZED);
        },
        Err(_) => {
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        },
    };
    let expires_in = 3600;
    let mut scopes = vec!["inference:read".to_string(), "inference:write".to_string()];
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
    if request.name.is_empty() || request.scopes.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    let mut api_key = auth_service.create_api_key(request.name, claims.sub.clone(), request.scopes);
    if let Some(expires_in_days) = request.expires_in_days {
        let expires_at = chrono::Utc::now() + chrono::Duration::days(expires_in_days as i64);
        api_key.expires_at = Some(expires_at);
    }
    if let Some(rate_limit) = request.rate_limit {
        api_key.rate_limit = Some(rate_limit);
    }
    if let Some(ip_whitelist) = request.ip_whitelist {
        api_key.ip_whitelist = Some(ip_whitelist);
    }
    if let Some(allowed_endpoints) = request.allowed_endpoints {
        api_key.allowed_endpoints = Some(allowed_endpoints);
    }
    let response = CreateApiKeyResponse {
        id: api_key.id.clone(),
        key: api_key.key.clone(),
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
    let api_keys = auth_service.list_api_keys(&claims.sub);
    let api_key = api_keys.iter().find(|k| k.id == key_id).ok_or(StatusCode::NOT_FOUND)?;
    auth_service
        .revoke_api_key(&api_key.key)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(StatusCode::NO_CONTENT)
}
/// OAuth2 HTTP handlers
/// Initiate OAuth2 authorization
pub async fn oauth2_authorize_handler(
    State(auth_service): State<AuthService>,
    Query(request): Query<OAuth2AuthRequest>,
) -> Result<axum::Json<OAuth2AuthResponse>, StatusCode> {
    let redirect_uri = request
        .redirect_uri
        .unwrap_or_else(|| "http://localhost:8080/auth/oauth2/callback".to_string());
    let auth_url = auth_service
        .generate_oauth2_auth_url(&request.provider, redirect_uri, request.scopes, None)
        .map_err(|_| StatusCode::BAD_REQUEST)?;
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
    let (_oauth2_token, user_info) =
        auth_service.exchange_oauth2_code(&provider, code, state).await.map_err(|e| {
            tracing::error!("OAuth2 token exchange failed: {}", e);
            StatusCode::UNAUTHORIZED
        })?;
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
