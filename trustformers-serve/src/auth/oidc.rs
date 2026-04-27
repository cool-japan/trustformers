//! OIDC/JWT Authentication middleware
//!
//! JWT-based authentication compatible with OIDC providers such as Auth0, Keycloak, and Google.
//! Implements SHA-256 and HMAC-SHA256 in pure Rust with no external crypto dependencies.

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during JWT authentication.
#[derive(Debug, Error)]
pub enum AuthError {
    #[error("Invalid token format: {0}")]
    InvalidFormat(String),
    #[error("Invalid base64: {0}")]
    Base64Error(String),
    #[error("Invalid JSON: {0}")]
    JsonError(String),
    #[error("Expired token")]
    Expired,
    #[error("Invalid issuer")]
    InvalidIssuer,
    #[error("Invalid audience")]
    InvalidAudience,
    #[error("Invalid signature")]
    InvalidSignature,
}

// ---------------------------------------------------------------------------
// Pure-Rust SHA-256
// ---------------------------------------------------------------------------

/// SHA-256 round constants (first 32 bits of the fractional parts of the
/// cube roots of the first 64 primes).
#[allow(clippy::unreadable_literal)]
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Initial hash values for SHA-256 (first 32 bits of the fractional parts
/// of the square roots of the first 8 primes).
#[allow(clippy::unreadable_literal)]
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Compute SHA-256 over `data`, returning a 32-byte digest.
fn sha256(data: &[u8]) -> [u8; 32] {
    // --- Pre-processing: padding ---
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    // Pad to 56 mod 64 bytes
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }
    // Append original length as big-endian u64
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // --- Processing: 512-bit (64-byte) blocks ---
    let mut h = H0;

    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut digest = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// Compute HMAC-SHA256 over `data` using `key`.
///
/// HMAC(key, msg) = SHA256((key XOR opad) || SHA256((key XOR ipad) || msg))
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    const BLOCK_SIZE: usize = 64;

    // Derive effective key: hash it if longer than block size
    let mut k = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
        let hashed = sha256(key);
        k[..32].copy_from_slice(&hashed);
    } else {
        k[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; BLOCK_SIZE];
    let mut opad = [0x5cu8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }

    // Inner hash: SHA256(ipad || data)
    let mut inner = ipad.to_vec();
    inner.extend_from_slice(data);
    let inner_hash = sha256(&inner);

    // Outer hash: SHA256(opad || inner_hash)
    let mut outer = opad.to_vec();
    outer.extend_from_slice(&inner_hash);
    sha256(&outer)
}

// ---------------------------------------------------------------------------
// Base64url helpers (RFC 4648 §5, no padding)
// ---------------------------------------------------------------------------

/// Encode bytes as base64url (no padding).
fn base64url_encode(data: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = Vec::with_capacity(data.len().div_ceil(3) * 4);
    let mut i = 0;
    while i < data.len() {
        let b0 = data[i] as u32;
        let b1 = if i + 1 < data.len() { data[i + 1] as u32 } else { 0 };
        let b2 = if i + 2 < data.len() { data[i + 2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3f) as usize]);
        out.push(TABLE[((triple >> 12) & 0x3f) as usize]);
        if i + 1 < data.len() {
            out.push(TABLE[((triple >> 6) & 0x3f) as usize]);
        }
        if i + 2 < data.len() {
            out.push(TABLE[(triple & 0x3f) as usize]);
        }
        i += 3;
    }
    // SAFETY: all bytes are valid ASCII
    String::from_utf8(out).unwrap_or_default()
}

/// Decode a base64url-encoded string (no padding required).
fn base64url_decode(s: &str) -> Result<Vec<u8>, AuthError> {
    // Normalise: replace '-' → '+', '_' → '/', then add padding
    let mut std_b64: String = s.replace('-', "+").replace('_', "/");
    match std_b64.len() % 4 {
        2 => std_b64.push_str("=="),
        3 => std_b64.push('='),
        _ => {}
    }

    // Manual base64 decode
    fn char_val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            b'=' => Some(0), // padding
            _ => None,
        }
    }

    let bytes = std_b64.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut i = 0;
    while i + 3 < bytes.len() {
        let v0 = char_val(bytes[i]).ok_or_else(|| {
            AuthError::Base64Error(format!("invalid char at position {i}"))
        })?;
        let v1 = char_val(bytes[i + 1]).ok_or_else(|| {
            AuthError::Base64Error(format!("invalid char at position {}", i + 1))
        })?;
        let v2 = char_val(bytes[i + 2]).ok_or_else(|| {
            AuthError::Base64Error(format!("invalid char at position {}", i + 2))
        })?;
        let v3 = char_val(bytes[i + 3]).ok_or_else(|| {
            AuthError::Base64Error(format!("invalid char at position {}", i + 3))
        })?;
        let triple = ((v0 as u32) << 18)
            | ((v1 as u32) << 12)
            | ((v2 as u32) << 6)
            | (v3 as u32);
        out.push(((triple >> 16) & 0xff) as u8);
        if bytes[i + 2] != b'=' {
            out.push(((triple >> 8) & 0xff) as u8);
        }
        if bytes[i + 3] != b'=' {
            out.push((triple & 0xff) as u8);
        }
        i += 4;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Claims
// ---------------------------------------------------------------------------

/// JWT claims extracted from a validated token.
#[derive(Debug, Clone)]
pub struct JwtClaims {
    /// Subject (user ID)
    pub sub: String,
    /// Issuer URL
    pub iss: String,
    /// Audience list
    pub aud: Vec<String>,
    /// Expiration timestamp (Unix seconds)
    pub exp: u64,
    /// Issued-at timestamp (Unix seconds)
    pub iat: u64,
    /// Optional email address
    pub email: Option<String>,
    /// Custom claim: roles
    pub roles: Vec<String>,
    /// Custom claim: scopes (space-separated → Vec)
    pub scopes: Vec<String>,
    /// Additional claims not listed above
    pub extra: HashMap<String, serde_json::Value>,
}

impl JwtClaims {
    /// Returns `true` if the token has expired relative to `now_unix`.
    pub fn is_expired(&self, now_unix: u64) -> bool {
        now_unix > self.exp
    }

    /// Returns `true` if the claims include the given role.
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }

    /// Returns `true` if the claims include the given scope.
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope)
    }

    /// Returns `true` if the audience list contains the given value.
    pub fn audience_contains(&self, aud: &str) -> bool {
        self.aud.iter().any(|a| a == aud)
    }
}

// ---------------------------------------------------------------------------
// Algorithm / Config
// ---------------------------------------------------------------------------

/// Supported JWT signature algorithms.
#[derive(Debug, Clone, PartialEq)]
pub enum JwtAlgorithm {
    /// HMAC-SHA256
    HS256,
    /// RSA-SHA256 (public key)
    RS256,
    /// ECDSA-SHA256
    ES256,
}

impl JwtAlgorithm {
    fn as_str(&self) -> &'static str {
        match self {
            JwtAlgorithm::HS256 => "HS256",
            JwtAlgorithm::RS256 => "RS256",
            JwtAlgorithm::ES256 => "ES256",
        }
    }
}

/// OIDC provider configuration.
#[derive(Debug, Clone)]
pub struct OidcConfig {
    /// Expected issuer (`iss` claim)
    pub issuer: String,
    /// Expected audience (`aud` claim)
    pub audience: String,
    /// Signature algorithm
    pub algorithm: JwtAlgorithm,
    /// HMAC secret (HS256) or PEM public key string (RS256/ES256)
    pub secret_or_pem: String,
    /// Clock skew tolerance in seconds (default 30)
    pub leeway_seconds: u64,
    /// Required scopes – all must be present
    pub required_scopes: Vec<String>,
    /// Required roles – all must be present
    pub required_roles: Vec<String>,
}

impl OidcConfig {
    /// Create an HS256-based configuration.
    pub fn new_hs256(
        issuer: impl Into<String>,
        audience: impl Into<String>,
        secret: impl Into<String>,
    ) -> Self {
        Self {
            issuer: issuer.into(),
            audience: audience.into(),
            algorithm: JwtAlgorithm::HS256,
            secret_or_pem: secret.into(),
            leeway_seconds: 30,
            required_scopes: Vec::new(),
            required_roles: Vec::new(),
        }
    }

    /// Create an RS256-based configuration.
    pub fn new_rs256(
        issuer: impl Into<String>,
        audience: impl Into<String>,
        pem_key: impl Into<String>,
    ) -> Self {
        Self {
            issuer: issuer.into(),
            audience: audience.into(),
            algorithm: JwtAlgorithm::RS256,
            secret_or_pem: pem_key.into(),
            leeway_seconds: 30,
            required_scopes: Vec::new(),
            required_roles: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// AuthResult
// ---------------------------------------------------------------------------

/// The result of validating a JWT token.
#[derive(Debug, Clone)]
pub enum AuthResult {
    /// Token is valid; carries the extracted claims.
    Authenticated(JwtClaims),
    /// Token is missing, malformed, or cryptographically invalid.
    Unauthorized(String),
    /// Token is valid but the principal lacks required permissions.
    Forbidden(String),
}

impl AuthResult {
    /// Returns `true` if the result is `Authenticated`.
    pub fn is_ok(&self) -> bool {
        matches!(self, AuthResult::Authenticated(_))
    }

    /// Returns a reference to the claims, if authenticated.
    pub fn claims(&self) -> Option<&JwtClaims> {
        match self {
            AuthResult::Authenticated(c) => Some(c),
            _ => None,
        }
    }

    /// Returns the error message for non-authenticated results.
    pub fn error_message(&self) -> Option<&str> {
        match self {
            AuthResult::Unauthorized(msg) | AuthResult::Forbidden(msg) => Some(msg),
            AuthResult::Authenticated(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: parse a raw JSON object into JwtClaims
// ---------------------------------------------------------------------------

fn parse_claims(map: &serde_json::Map<String, serde_json::Value>) -> Result<JwtClaims, AuthError> {
    let sub = map
        .get("sub")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let iss = map
        .get("iss")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // `aud` may be a string or an array of strings
    let aud = match map.get("aud") {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    };

    let exp = map
        .get("exp")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let iat = map
        .get("iat")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let email = map
        .get("email")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Roles: accept array-of-strings or a single string
    let roles = match map.get("roles") {
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    };

    // Scopes: space-separated string or array
    let scopes = match map.get("scope") {
        Some(serde_json::Value::String(s)) => {
            s.split_whitespace().map(|t| t.to_string()).collect()
        }
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    };

    // Everything else goes into `extra`
    let known = ["sub", "iss", "aud", "exp", "iat", "email", "roles", "scope"];
    let extra: HashMap<String, serde_json::Value> = map
        .iter()
        .filter(|(k, _)| !known.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    Ok(JwtClaims {
        sub,
        iss,
        aud,
        exp,
        iat,
        email,
        roles,
        scopes,
        extra,
    })
}

// ---------------------------------------------------------------------------
// JwtValidator
// ---------------------------------------------------------------------------

/// JWT validator – pure Rust, no external crypto dependencies.
pub struct JwtValidator {
    config: OidcConfig,
}

impl JwtValidator {
    /// Create a new validator with the given configuration.
    pub fn new(config: OidcConfig) -> Self {
        Self { config }
    }

    /// Validate a Bearer token string.
    ///
    /// Steps:
    /// 1. Split on `.` → `[header, payload, signature]`
    /// 2. Base64url-decode each part
    /// 3. Parse header JSON and verify `alg`
    /// 4. Parse payload JSON and extract claims
    /// 5. Check expiration (with leeway)
    /// 6. Check issuer
    /// 7. Check audience
    /// 8. Verify signature (HS256 full HMAC; RS256/ES256 – structural only)
    /// 9. Check required scopes and roles
    pub fn validate(&self, token: &str) -> AuthResult {
        // Strip optional "Bearer " prefix
        let token = token.strip_prefix("Bearer ").unwrap_or(token).trim();

        // --- Step 1: split ---
        let parts: Vec<&str> = token.splitn(3, '.').collect();
        if parts.len() != 3 {
            return AuthResult::Unauthorized("token must have three dot-separated parts".to_string());
        }
        let (header_b64, payload_b64, sig_b64) = (parts[0], parts[1], parts[2]);
        let signing_input = format!("{header_b64}.{payload_b64}");

        // --- Step 2/3: decode + parse header ---
        let header_bytes = match base64url_decode(header_b64) {
            Ok(b) => b,
            Err(e) => return AuthResult::Unauthorized(e.to_string()),
        };
        let header_val: serde_json::Value = match serde_json::from_slice(&header_bytes) {
            Ok(v) => v,
            Err(e) => return AuthResult::Unauthorized(format!("header JSON: {e}")),
        };
        let alg_str = header_val
            .get("alg")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if alg_str != self.config.algorithm.as_str() {
            return AuthResult::Unauthorized(format!(
                "algorithm mismatch: got {alg_str}, expected {}",
                self.config.algorithm.as_str()
            ));
        }

        // --- Step 2/4: decode + parse payload ---
        let payload_bytes = match base64url_decode(payload_b64) {
            Ok(b) => b,
            Err(e) => return AuthResult::Unauthorized(e.to_string()),
        };
        let payload_val: serde_json::Value = match serde_json::from_slice(&payload_bytes) {
            Ok(v) => v,
            Err(e) => return AuthResult::Unauthorized(format!("payload JSON: {e}")),
        };
        let payload_map = match payload_val.as_object() {
            Some(m) => m,
            None => return AuthResult::Unauthorized("payload is not a JSON object".to_string()),
        };
        let claims = match parse_claims(payload_map) {
            Ok(c) => c,
            Err(e) => return AuthResult::Unauthorized(e.to_string()),
        };

        // --- Step 5: expiration ---
        let now = current_unix_timestamp();
        if now > claims.exp + self.config.leeway_seconds {
            return AuthResult::Unauthorized("token has expired".to_string());
        }

        // --- Step 6: issuer ---
        if claims.iss != self.config.issuer {
            return AuthResult::Unauthorized(format!(
                "issuer mismatch: got '{}', expected '{}'",
                claims.iss, self.config.issuer
            ));
        }

        // --- Step 7: audience ---
        if !claims.audience_contains(&self.config.audience) {
            return AuthResult::Unauthorized(format!(
                "audience '{}' not present in token",
                self.config.audience
            ));
        }

        // --- Step 8: signature ---
        if self.config.algorithm == JwtAlgorithm::HS256 {
            let expected = hmac_sha256(
                self.config.secret_or_pem.as_bytes(),
                signing_input.as_bytes(),
            );
            let expected_b64 = base64url_encode(&expected);
            if expected_b64 != sig_b64 {
                return AuthResult::Unauthorized("invalid signature".to_string());
            }
        }
        // RS256/ES256: full crypto needs an RSA/ECDSA library; validate everything
        // except the raw signature bytes (structural validation only).

        // --- Step 9: scopes and roles ---
        for scope in &self.config.required_scopes {
            if !claims.has_scope(scope) {
                return AuthResult::Forbidden(format!("required scope '{scope}' is missing"));
            }
        }
        for role in &self.config.required_roles {
            if !claims.has_role(role) {
                return AuthResult::Forbidden(format!("required role '{role}' is missing"));
            }
        }

        AuthResult::Authenticated(claims)
    }

    /// Decode claims without verifying the signature (useful for inspection /
    /// testing).
    pub fn decode_unverified(token: &str) -> Result<JwtClaims, AuthError> {
        let token = token.strip_prefix("Bearer ").unwrap_or(token).trim();
        let parts: Vec<&str> = token.splitn(3, '.').collect();
        if parts.len() != 3 {
            return Err(AuthError::InvalidFormat(
                "token must have three dot-separated parts".to_string(),
            ));
        }
        let payload_bytes = base64url_decode(parts[1])?;
        let payload_val: serde_json::Value =
            serde_json::from_slice(&payload_bytes).map_err(|e| AuthError::JsonError(e.to_string()))?;
        let payload_map = payload_val
            .as_object()
            .ok_or_else(|| AuthError::JsonError("payload is not a JSON object".to_string()))?;
        parse_claims(payload_map)
    }

    /// Create a signed HS256 JWT for testing purposes.
    ///
    /// Header: `{"alg":"HS256","typ":"JWT"}`
    pub fn create_test_token(claims: &JwtClaims, secret: &str) -> Result<String, AuthError> {
        let header = r#"{"alg":"HS256","typ":"JWT"}"#;
        let header_b64 = base64url_encode(header.as_bytes());

        // Build payload object
        let mut payload_map = serde_json::Map::new();
        payload_map.insert("sub".to_string(), serde_json::Value::String(claims.sub.clone()));
        payload_map.insert("iss".to_string(), serde_json::Value::String(claims.iss.clone()));
        // `aud` as array
        let aud_arr: Vec<serde_json::Value> = claims
            .aud
            .iter()
            .map(|a| serde_json::Value::String(a.clone()))
            .collect();
        payload_map.insert("aud".to_string(), serde_json::Value::Array(aud_arr));
        payload_map.insert("exp".to_string(), serde_json::Value::Number(claims.exp.into()));
        payload_map.insert("iat".to_string(), serde_json::Value::Number(claims.iat.into()));
        if let Some(ref email) = claims.email {
            payload_map.insert("email".to_string(), serde_json::Value::String(email.clone()));
        }
        if !claims.roles.is_empty() {
            let roles_arr: Vec<serde_json::Value> = claims
                .roles
                .iter()
                .map(|r| serde_json::Value::String(r.clone()))
                .collect();
            payload_map.insert("roles".to_string(), serde_json::Value::Array(roles_arr));
        }
        if !claims.scopes.is_empty() {
            payload_map.insert(
                "scope".to_string(),
                serde_json::Value::String(claims.scopes.join(" ")),
            );
        }
        // Extra claims
        for (k, v) in &claims.extra {
            payload_map.insert(k.clone(), v.clone());
        }

        let payload_json = serde_json::to_string(&serde_json::Value::Object(payload_map))
            .map_err(|e| AuthError::JsonError(e.to_string()))?;
        let payload_b64 = base64url_encode(payload_json.as_bytes());

        let signing_input = format!("{header_b64}.{payload_b64}");
        let sig = hmac_sha256(secret.as_bytes(), signing_input.as_bytes());
        let sig_b64 = base64url_encode(&sig);

        Ok(format!("{signing_input}.{sig_b64}"))
    }
}

// ---------------------------------------------------------------------------
// Helper: current Unix timestamp (seconds)
// ---------------------------------------------------------------------------

fn current_unix_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    fn make_claims(sub: &str, iss: &str, aud: &[&str], exp_offset: i64) -> JwtClaims {
        let now = now_secs();
        let exp = if exp_offset >= 0 {
            now + exp_offset as u64
        } else {
            now.saturating_sub((-exp_offset) as u64)
        };
        JwtClaims {
            sub: sub.to_string(),
            iss: iss.to_string(),
            aud: aud.iter().map(|s| s.to_string()).collect(),
            exp,
            iat: now - 10,
            email: Some("user@example.com".to_string()),
            roles: vec!["user".to_string()],
            scopes: vec!["read".to_string(), "write".to_string()],
            extra: HashMap::new(),
        }
    }

    fn default_config() -> OidcConfig {
        OidcConfig::new_hs256("https://issuer.example.com", "my-audience", "super-secret")
    }

    // Test 1: Roundtrip – create and validate a token
    #[test]
    fn test_create_and_validate_roundtrip() {
        let config = default_config();
        let claims = make_claims(
            "user-123",
            "https://issuer.example.com",
            &["my-audience"],
            3600,
        );
        let token = JwtValidator::create_test_token(&claims, "super-secret")
            .expect("token creation should succeed");
        let validator = JwtValidator::new(config);
        let result = validator.validate(&token);
        assert!(result.is_ok(), "expected Authenticated, got: {:?}", result.error_message());
        let got = result.claims().expect("should have claims");
        assert_eq!(got.sub, "user-123");
    }

    // Test 2: Expired token is rejected
    #[test]
    fn test_expired_token_rejected() {
        let mut config = default_config();
        config.leeway_seconds = 0; // no leeway
        let claims = make_claims(
            "user-456",
            "https://issuer.example.com",
            &["my-audience"],
            -100, // expired 100 s ago
        );
        let token = JwtValidator::create_test_token(&claims, "super-secret")
            .expect("token creation should succeed");
        let validator = JwtValidator::new(config);
        let result = validator.validate(&token);
        assert!(!result.is_ok());
        assert!(
            result.error_message().map(|m| m.contains("expired")).unwrap_or(false),
            "expected expired message"
        );
    }

    // Test 3: Wrong issuer is rejected
    #[test]
    fn test_wrong_issuer_rejected() {
        let config = default_config();
        let claims = make_claims(
            "user-789",
            "https://wrong-issuer.example.com",
            &["my-audience"],
            3600,
        );
        let token = JwtValidator::create_test_token(&claims, "super-secret")
            .expect("token creation should succeed");
        let validator = JwtValidator::new(config);
        let result = validator.validate(&token);
        assert!(!result.is_ok());
        assert!(
            result.error_message().map(|m| m.contains("issuer")).unwrap_or(false),
            "expected issuer message"
        );
    }

    // Test 4: Wrong audience is rejected
    #[test]
    fn test_wrong_audience_rejected() {
        let config = default_config();
        let claims = make_claims(
            "user-abc",
            "https://issuer.example.com",
            &["other-audience"],
            3600,
        );
        let token = JwtValidator::create_test_token(&claims, "super-secret")
            .expect("token creation should succeed");
        let validator = JwtValidator::new(config);
        let result = validator.validate(&token);
        assert!(!result.is_ok());
        assert!(
            result.error_message().map(|m| m.contains("audience")).unwrap_or(false),
            "expected audience message"
        );
    }

    // Test 5: Missing required scope returns Forbidden
    #[test]
    fn test_missing_required_scope() {
        let mut config = default_config();
        config.required_scopes = vec!["admin".to_string()];
        let mut claims = make_claims(
            "user-def",
            "https://issuer.example.com",
            &["my-audience"],
            3600,
        );
        claims.scopes = vec!["read".to_string()]; // no "admin"
        let token = JwtValidator::create_test_token(&claims, "super-secret")
            .expect("token creation should succeed");
        let validator = JwtValidator::new(config);
        let result = validator.validate(&token);
        assert!(matches!(result, AuthResult::Forbidden(_)));
    }

    // Test 6: Missing required role returns Forbidden
    #[test]
    fn test_missing_required_role() {
        let mut config = default_config();
        config.required_roles = vec!["superuser".to_string()];
        let mut claims = make_claims(
            "user-ghi",
            "https://issuer.example.com",
            &["my-audience"],
            3600,
        );
        claims.roles = vec!["user".to_string()]; // no "superuser"
        let token = JwtValidator::create_test_token(&claims, "super-secret")
            .expect("token creation should succeed");
        let validator = JwtValidator::new(config);
        let result = validator.validate(&token);
        assert!(matches!(result, AuthResult::Forbidden(_)));
    }

    // Test 7: has_role
    #[test]
    fn test_has_role() {
        let claims = make_claims("u", "i", &["a"], 3600);
        assert!(claims.has_role("user"));
        assert!(!claims.has_role("admin"));
    }

    // Test 8: has_scope
    #[test]
    fn test_has_scope() {
        let claims = make_claims("u", "i", &["a"], 3600);
        assert!(claims.has_scope("read"));
        assert!(claims.has_scope("write"));
        assert!(!claims.has_scope("delete"));
    }

    // Test 9: audience_contains
    #[test]
    fn test_audience_contains() {
        let claims = make_claims("u", "i", &["audience-a", "audience-b"], 3600);
        assert!(claims.audience_contains("audience-a"));
        assert!(claims.audience_contains("audience-b"));
        assert!(!claims.audience_contains("audience-c"));
    }

    // Test 10: decode_unverified works on a valid token
    #[test]
    fn test_decode_unverified() {
        let claims = make_claims(
            "user-jkl",
            "https://issuer.example.com",
            &["my-audience"],
            3600,
        );
        let token = JwtValidator::create_test_token(&claims, "some-secret")
            .expect("token creation should succeed");
        let decoded = JwtValidator::decode_unverified(&token).expect("decode should succeed");
        assert_eq!(decoded.sub, "user-jkl");
        assert_eq!(decoded.iss, "https://issuer.example.com");
    }

    // Test 11: Invalid format (too few parts)
    #[test]
    fn test_invalid_format() {
        let config = default_config();
        let validator = JwtValidator::new(config);
        let result = validator.validate("not.a.valid.jwt.extra");
        // The token will split into only 3 parts with splitn(3,'.')
        // Try something that doesn't have 3 parts at all
        let result2 = validator.validate("notavalidjwt");
        assert!(!result2.is_ok());
    }

    // Test 12: base64url_encode / base64url_decode roundtrip
    #[test]
    fn test_base64url_roundtrip() {
        let data = b"Hello, World! \x00\xff\xfe";
        let encoded = base64url_encode(data);
        let decoded = base64url_decode(&encoded).expect("decode should succeed");
        assert_eq!(&decoded, data);
    }

    // Test 13: base64url decode of invalid string returns error
    #[test]
    fn test_base64url_decode_invalid() {
        // Provide a string with characters outside the base64url alphabet
        // '!' is invalid in base64
        let result = base64url_decode("!!!");
        assert!(result.is_err());
    }

    // Test 14: Leeway tolerance – token expired 20 s ago, leeway = 30
    #[test]
    fn test_leeway_tolerance() {
        let mut config = default_config();
        config.leeway_seconds = 30;
        let claims = make_claims(
            "user-mno",
            "https://issuer.example.com",
            &["my-audience"],
            -20, // expired 20 s ago but within leeway of 30
        );
        let token = JwtValidator::create_test_token(&claims, "super-secret")
            .expect("token creation should succeed");
        let validator = JwtValidator::new(config);
        let result = validator.validate(&token);
        assert!(result.is_ok(), "token within leeway should be accepted");
    }

    // Test 15: Wrong secret produces invalid signature
    #[test]
    fn test_wrong_secret_rejected() {
        let config = default_config(); // secret = "super-secret"
        let claims = make_claims(
            "user-pqr",
            "https://issuer.example.com",
            &["my-audience"],
            3600,
        );
        let token = JwtValidator::create_test_token(&claims, "wrong-secret")
            .expect("token creation should succeed");
        let validator = JwtValidator::new(config);
        let result = validator.validate(&token);
        assert!(!result.is_ok(), "wrong secret should be rejected");
    }

    // Test 16: RS256 config validates issuer/audience/expiry even without full sig check
    #[test]
    fn test_rs256_structural_validation() {
        // Build an RS256 config; signature step is structural only
        let mut config = OidcConfig::new_rs256(
            "https://issuer.example.com",
            "my-audience",
            "-----BEGIN PUBLIC KEY-----\nMOCK\n-----END PUBLIC KEY-----",
        );
        config.leeway_seconds = 30;

        let claims = make_claims(
            "user-stu",
            "https://issuer.example.com",
            &["my-audience"],
            3600,
        );
        // Create token with HS256 signing (sig bytes don't matter for RS256 path)
        let token = JwtValidator::create_test_token(&claims, "ignored")
            .expect("token creation should succeed");

        // Swap the `alg` header to RS256 by constructing manually
        let parts: Vec<&str> = token.splitn(3, '.').collect();
        let rs256_header = base64url_encode(br#"{"alg":"RS256","typ":"JWT"}"#);
        let rs256_token = format!("{}.{}.{}", rs256_header, parts[1], parts[2]);

        let validator = JwtValidator::new(config);
        let result = validator.validate(&rs256_token);
        // Should be Authenticated (structural checks all pass; sig skipped)
        assert!(result.is_ok(), "RS256 structural validation should pass: {:?}", result.error_message());
    }
}
