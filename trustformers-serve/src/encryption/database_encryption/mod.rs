//! Auto-generated module structure

pub mod connectionauthentication_traits;
pub mod authenticationcredentials_traits;
pub mod tlsconfig_traits;
pub mod connectionencryptionconfig_traits;
pub mod queryencryptionconfig_traits;
pub mod types;
pub mod functions;

// Re-export all types
pub use connectionauthentication_traits::*;
pub use authenticationcredentials_traits::*;
pub use tlsconfig_traits::*;
pub use connectionencryptionconfig_traits::*;
pub use queryencryptionconfig_traits::*;
pub use types::*;
pub use functions::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};
    use std::collections::HashMap;

    // ── DatabaseEncryptionScope ───────────────────────────────────────────

    #[test]
    fn test_db_encryption_scope_column_level_debug() {
        let s = DatabaseEncryptionScope::ColumnLevel;
        assert!(format!("{:?}", s).contains("ColumnLevel"));
    }

    #[test]
    fn test_db_encryption_scope_table_level_debug() {
        let s = DatabaseEncryptionScope::TableLevel;
        assert!(format!("{:?}", s).contains("TableLevel"));
    }

    #[test]
    fn test_db_encryption_scope_all_variants_distinct() {
        let variants = [
            DatabaseEncryptionScope::ColumnLevel,
            DatabaseEncryptionScope::TableLevel,
            DatabaseEncryptionScope::DatabaseLevel,
            DatabaseEncryptionScope::TransparentDataEncryption,
            DatabaseEncryptionScope::FieldLevel,
            DatabaseEncryptionScope::QueryLevel,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // ── TDEKeyStatus ──────────────────────────────────────────────────────

    #[test]
    fn test_tde_key_status_active_debug() {
        let s = TDEKeyStatus::Active;
        assert!(format!("{:?}", s).contains("Active"));
    }

    #[test]
    fn test_tde_key_status_rotating_debug() {
        let s = TDEKeyStatus::Rotating;
        assert!(format!("{:?}", s).contains("Rotating"));
    }

    #[test]
    fn test_tde_key_status_deprecated_debug() {
        let s = TDEKeyStatus::Deprecated;
        assert!(format!("{:?}", s).contains("Deprecated"));
    }

    #[test]
    fn test_tde_key_status_revoked_debug() {
        let s = TDEKeyStatus::Revoked;
        assert!(format!("{:?}", s).contains("Revoked"));
    }

    #[test]
    fn test_tde_key_status_equality() {
        assert_eq!(TDEKeyStatus::Active, TDEKeyStatus::Active);
        assert_ne!(TDEKeyStatus::Active, TDEKeyStatus::Revoked);
        assert_ne!(TDEKeyStatus::Rotating, TDEKeyStatus::Deprecated);
    }

    // ── TLSVersion ────────────────────────────────────────────────────────

    #[test]
    fn test_tls_version_12_debug() {
        let v = TLSVersion::TLS12;
        assert!(format!("{:?}", v).contains("TLS12"));
    }

    #[test]
    fn test_tls_version_13_debug() {
        let v = TLSVersion::TLS13;
        assert!(format!("{:?}", v).contains("TLS13"));
    }

    // ── QueryParsingMode ──────────────────────────────────────────────────

    #[test]
    fn test_query_parsing_mode_full_debug() {
        let m = QueryParsingMode::Full;
        assert!(format!("{:?}", m).contains("Full"));
    }

    #[test]
    fn test_query_parsing_mode_parameter_only_debug() {
        let m = QueryParsingMode::ParameterOnly;
        assert!(format!("{:?}", m).contains("ParameterOnly"));
    }

    #[test]
    fn test_query_parsing_mode_pattern_based_debug() {
        let m = QueryParsingMode::PatternBased;
        assert!(format!("{:?}", m).contains("PatternBased"));
    }

    // ── TableEncryptionStatus ─────────────────────────────────────────────

    #[test]
    fn test_table_encryption_status_all_variants() {
        let variants = [
            TableEncryptionStatus::NotEncrypted,
            TableEncryptionStatus::Encrypting,
            TableEncryptionStatus::Encrypted,
            TableEncryptionStatus::Decrypting,
            TableEncryptionStatus::Failed,
        ];
        for v in &variants {
            assert!(!format!("{:?}", v).is_empty());
        }
    }

    #[test]
    fn test_table_encryption_status_equality() {
        assert_eq!(TableEncryptionStatus::Encrypted, TableEncryptionStatus::Encrypted);
        assert_ne!(TableEncryptionStatus::Encrypted, TableEncryptionStatus::Failed);
    }

    // ── AuthenticationMethod ──────────────────────────────────────────────

    #[test]
    fn test_authentication_method_username_password_debug() {
        let m = AuthenticationMethod::UsernamePassword;
        assert!(format!("{:?}", m).contains("UsernamePassword"));
    }

    #[test]
    fn test_authentication_method_certificate_debug() {
        let m = AuthenticationMethod::Certificate;
        assert!(format!("{:?}", m).contains("Certificate"));
    }

    #[test]
    fn test_authentication_method_token_debug() {
        let m = AuthenticationMethod::Token;
        assert!(format!("{:?}", m).contains("Token"));
    }

    #[test]
    fn test_authentication_method_kerberos_debug() {
        let m = AuthenticationMethod::Kerberos;
        assert!(format!("{:?}", m).contains("Kerberos"));
    }

    // ── AuthenticationCredentials ─────────────────────────────────────────

    #[test]
    fn test_authentication_credentials_with_username() {
        let creds = AuthenticationCredentials {
            username: Some("db_user".to_string()),
            password: Some(b"hashed_pw".to_vec()),
            certificate_path: None,
            token: None,
        };
        assert_eq!(creds.username.as_deref(), Some("db_user"));
        assert!(creds.certificate_path.is_none());
        assert!(creds.token.is_none());
    }

    #[test]
    fn test_authentication_credentials_with_token() {
        let creds = AuthenticationCredentials {
            username: None,
            password: None,
            certificate_path: None,
            token: Some("bearer_abc123".to_string()),
        };
        assert!(creds.username.is_none());
        assert_eq!(creds.token.as_deref(), Some("bearer_abc123"));
    }

    // ── TLSConfig ─────────────────────────────────────────────────────────

    #[test]
    fn test_tls_config_fields() {
        let cfg = TLSConfig {
            version: TLSVersion::TLS13,
            cipher_suites: vec![
                "TLS_AES_256_GCM_SHA384".to_string(),
                "TLS_CHACHA20_POLY1305_SHA256".to_string(),
            ],
            certificate_validation: true,
            client_certificate: Some("/etc/ssl/client.crt".to_string()),
        };
        assert!(cfg.certificate_validation);
        assert_eq!(cfg.cipher_suites.len(), 2);
        assert!(cfg.client_certificate.is_some());
    }

    // ── ParameterEncryptionCache ──────────────────────────────────────────

    #[test]
    fn test_parameter_encryption_cache_fields() {
        let cache = ParameterEncryptionCache {
            parameter_hash: "abc123".to_string(),
            encrypted_value: vec![0xDE, 0xAD, 0xBE, 0xEF],
            key_id: "key_001".to_string(),
            cached_at: SystemTime::now(),
        };
        assert_eq!(cache.parameter_hash, "abc123");
        assert_eq!(cache.encrypted_value.len(), 4);
        assert_eq!(cache.key_id, "key_001");
    }

    // ── PatternMatch ──────────────────────────────────────────────────────

    #[test]
    fn test_pattern_match_fields() {
        let pm = PatternMatch {
            match_text: "***13***".to_string(),
            position: 42,
            length: 13,
            confidence: 0.95_f64,
        };
        assert_eq!(pm.position, 42);
        assert_eq!(pm.length, 13);
        assert!((pm.confidence - 0.95).abs() < 1e-9);
    }

    // ── ConnectionAuthentication ──────────────────────────────────────────

    #[test]
    fn test_connection_authentication_fields() {
        let auth = ConnectionAuthentication {
            method: AuthenticationMethod::UsernamePassword,
            credentials: AuthenticationCredentials {
                username: Some("admin".to_string()),
                password: None,
                certificate_path: None,
                token: None,
            },
        };
        match auth.method {
            AuthenticationMethod::UsernamePassword => {}
            _ => panic!("Wrong method"),
        }
        assert_eq!(auth.credentials.username.as_deref(), Some("admin"));
    }

    // ── QueryEncryptionConfig ─────────────────────────────────────────────

    #[test]
    fn test_query_encryption_config_fields() {
        let cfg = QueryEncryptionConfig {
            enabled: true,
            encrypt_parameters: true,
            encrypt_results: false,
            parsing_mode: QueryParsingMode::Full,
        };
        assert!(cfg.enabled);
        assert!(cfg.encrypt_parameters);
        assert!(!cfg.encrypt_results);
    }
}
