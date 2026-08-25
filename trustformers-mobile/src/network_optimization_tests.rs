//! Tests for the `network_optimization` module.

use super::*;

#[test]
fn test_network_optimization_config_default() {
    let config = NetworkOptimizationConfig::default();
    assert!(config.enable_resumable_downloads);
    assert!(config.enable_bandwidth_awareness);
    assert!(!config.enable_p2p_sharing); // Should be disabled by default
    assert!(config.enable_edge_servers);
}

#[test]
fn test_network_optimization_config_validation() {
    let mut config = NetworkOptimizationConfig::default();
    assert!(config.validate().is_ok());

    config.download_optimization.max_concurrent_downloads = 0;
    assert!(config.validate().is_err());

    config.download_optimization.max_concurrent_downloads = 15;
    assert!(config.validate().is_err());
}

#[test]
fn test_download_priority_ordering() {
    assert!(DownloadPriority::Critical > DownloadPriority::High);
    assert!(DownloadPriority::High > DownloadPriority::Normal);
    assert!(DownloadPriority::Normal > DownloadPriority::Low);
}

#[tokio::test]
async fn test_network_optimization_manager_creation() {
    let config = NetworkOptimizationConfig::default();
    let result = NetworkOptimizationManager::new(config);
    assert!(result.is_ok());
}

#[test]
fn test_bandwidth_thresholds() {
    let thresholds = BandwidthThresholds {
        low_bandwidth_kbps: 100.0,
        medium_bandwidth_kbps: 1000.0,
        high_bandwidth_kbps: 10000.0,
        ultra_high_bandwidth_kbps: 100000.0,
    };

    assert!(thresholds.ultra_high_bandwidth_kbps > thresholds.high_bandwidth_kbps);
    assert!(thresholds.high_bandwidth_kbps > thresholds.medium_bandwidth_kbps);
    assert!(thresholds.medium_bandwidth_kbps > thresholds.low_bandwidth_kbps);
}

// --- Regression tests for the removed wifi_only/charging_only fabrications ---

#[test]
fn test_is_wifi_connected_and_is_device_charging_report_unverifiable_honestly() {
    // Regression guard for the previous fabricated constants (`true`
    // for WiFi, `false` for charging): both must now say "cannot
    // verify" rather than assert a connectivity/charging state nothing
    // ever measured.
    let manager = NetworkOptimizationManager::new(NetworkOptimizationConfig::default())
        .expect("manager creation should succeed");
    assert_eq!(manager.is_wifi_connected(), None);
    assert_eq!(manager.is_device_charging(), None);
}

fn no_op_constraints() -> DownloadConstraints {
    DownloadConstraints {
        wifi_only: false,
        charging_only: false,
        max_bandwidth_kbps: None,
        time_windows: Vec::new(),
    }
}

#[tokio::test]
async fn test_check_download_constraints_wifi_only_fails_closed_when_unverifiable() {
    let manager = NetworkOptimizationManager::new(NetworkOptimizationConfig::default())
        .expect("manager creation should succeed");
    let constraints = DownloadConstraints {
        wifi_only: true,
        ..no_op_constraints()
    };

    let result = manager.check_download_constraints(&constraints).await;
    assert!(
        result.is_err(),
        "a wifi_only download must refuse with a structured error when WiFi connectivity \
         cannot be verified, not silently proceed on a possibly-metered connection \
         (previous behavior: `is_wifi_connected` == `true` unconditionally)"
    );
}

#[tokio::test]
async fn test_check_download_constraints_charging_only_fails_closed_when_unverifiable() {
    let manager = NetworkOptimizationManager::new(NetworkOptimizationConfig::default())
        .expect("manager creation should succeed");
    let constraints = DownloadConstraints {
        charging_only: true,
        ..no_op_constraints()
    };

    let result = manager.check_download_constraints(&constraints).await;
    assert!(
        result.is_err(),
        "a charging_only download must refuse with a structured error naming the \
         unverifiable check, distinguishable from an ordinary 'not charging right now' \
         Ok(false) -- previously both collapsed into the same permanent Ok(false) with no \
         way to tell them apart"
    );
}

#[tokio::test]
async fn test_check_download_constraints_passes_when_neither_constraint_requested() {
    let manager = NetworkOptimizationManager::new(NetworkOptimizationConfig::default())
        .expect("manager creation should succeed");

    let result = manager.check_download_constraints(&no_op_constraints()).await;
    assert!(
        result.expect("a download with no wifi/charging constraint must not error"),
        "no wifi_only/charging_only constraint was requested, so the unverifiable WiFi/\
         charging checks must never be consulted"
    );
}

/// Regression test for the previous `get_current_time_info` hardcoded
/// `hour: 12, day_of_week: 1` regardless of the real clock, which made
/// `time_windows`-restricted downloads (the live
/// `check_download_constraints` path, via `is_time_in_window`) gated
/// by whether "noon on Monday" happened to fall in the configured
/// window rather than by the actual time. Compares against an
/// independent `chrono::Local::now()` read bracketing the call, the
/// same technique used in `device_info.rs`'s live-sensor test, to
/// tolerate the rare hour/day rollover exactly at the boundary rather
/// than asserting byte-for-byte equality with a stale single read.
#[tokio::test]
async fn test_get_current_time_info_reflects_the_real_clock_not_hardcoded_noon_monday() {
    use chrono::{Datelike, Timelike};

    let manager = NetworkOptimizationManager::new(NetworkOptimizationConfig::default())
        .expect("manager creation should succeed");

    let before = chrono::Local::now();
    let observed = manager.get_current_time_info();
    let after = chrono::Local::now();

    let before_hour = before.hour() as usize;
    let after_hour = after.hour() as usize;
    let before_day = before.weekday().num_days_from_sunday() as usize;
    let after_day = after.weekday().num_days_from_sunday() as usize;

    assert!(
        observed.hour == before_hour || observed.hour == after_hour,
        "hour {} must match a real clock read bracketing the call (saw {before_hour} or \
         {after_hour}), not the previous hardcoded 12",
        observed.hour
    );
    assert!(
        observed.day_of_week == before_day || observed.day_of_week == after_day,
        "day_of_week {} must match a real clock read bracketing the call (saw {before_day} \
         or {after_day}), not the previous hardcoded 1",
        observed.day_of_week
    );
}

#[test]
fn test_add_shared_model_computes_real_hash_and_size_not_placeholders() {
    use std::io::Write;

    let config = NetworkOptimizationConfig::default();
    let mut manager = P2PManager::new(&config.p2p_config);

    let dir = std::env::temp_dir();
    let path_a = dir.join(format!(
        "trustformers_p2p_test_a_{}.bin",
        std::process::id()
    ));
    let path_b = dir.join(format!(
        "trustformers_p2p_test_b_{}.bin",
        std::process::id()
    ));

    let content_a = b"trustformers p2p shared model bytes, sample A";
    let content_b =
        b"a completely different set of shared model bytes for sample B, and longer too";

    std::fs::File::create(&path_a)
        .and_then(|mut f| f.write_all(content_a))
        .expect("write test file A");
    std::fs::File::create(&path_b)
        .and_then(|mut f| f.write_all(content_b))
        .expect("write test file B");

    manager
        .add_shared_model("model-a", &path_a)
        .expect("add_shared_model A must succeed");
    manager
        .add_shared_model("model-b", &path_b)
        .expect("add_shared_model B must succeed");

    let (hash_a, size_a) = {
        let m = manager.shared_models.get("model-a").expect("model-a must be recorded");
        (m.model_hash.clone(), m.size_bytes)
    };
    let (hash_b, size_b) = {
        let m = manager.shared_models.get("model-b").expect("model-b must be recorded");
        (m.model_hash.clone(), m.size_bytes)
    };

    // Real, content-derived size -- not the old hardcoded 1 MiB for every model.
    assert_eq!(size_a, content_a.len());
    assert_eq!(size_b, content_b.len());
    assert_ne!(
        size_a,
        1024 * 1024,
        "must not be the old hardcoded placeholder size"
    );

    // Real, content-derived hash -- not the old constant string, and different
    // content must not collide onto the same hash.
    assert_ne!(hash_a, "placeholder_hash");
    assert_ne!(hash_b, "placeholder_hash");
    assert_ne!(
        hash_a, hash_b,
        "different file contents must not hash identically"
    );

    // Hashing the same content again must reproduce the identical hash.
    manager
        .add_shared_model("model-a-again", &path_a)
        .expect("re-adding the same content must succeed");
    let hash_a_again = manager
        .shared_models
        .get("model-a-again")
        .expect("model-a-again must be recorded")
        .model_hash
        .clone();
    assert_eq!(
        hash_a, hash_a_again,
        "hashing identical content twice must produce the same hash"
    );

    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);
}

#[test]
fn test_add_shared_model_errors_on_missing_file_instead_of_fabricating() {
    let config = NetworkOptimizationConfig::default();
    let mut manager = P2PManager::new(&config.p2p_config);

    let missing_path = std::env::temp_dir().join(format!(
        "trustformers_p2p_test_missing_{}_{}.bin",
        std::process::id(),
        "does_not_exist"
    ));
    let _ = std::fs::remove_file(&missing_path); // ensure it really is absent

    let result = manager.add_shared_model("missing-model", &missing_path);
    assert!(
        result.is_err(),
        "sharing a model whose file does not exist must return a structured error, not a \
         fabricated hash/size"
    );
    assert!(
        !manager.shared_models.contains_key("missing-model"),
        "a failed share must not leave a fabricated record behind"
    );
}

// --- Regression tests: exact-digest P2P hash coverage --------------
//
// The pre-existing tests above (`test_add_shared_model_computes_real_
// hash_and_size_not_placeholders`, `test_add_shared_model_errors_on_
// missing_file_instead_of_fabricating`) prove `add_shared_model` is
// not a placeholder (real hash, real size, reproducible, errors on a
// missing file) but never assert an exact expected digest and never
// go through the public `enable_p2p_sharing` entry point. These three
// close both gaps: an exact hardcoded SHA-256 hex (cross-validated
// independently against `shasum -a 256` and `openssl dgst -sha256`,
// not merely recomputed with the same `sha2` crate the implementation
// uses), reached through `NetworkOptimizationManager::enable_p2p_
// sharing` end to end (including its config gate and its `Mutex`-
// guarded `P2PManager`), plus a chunk-boundary payload exercising the
// 65536-byte streaming loop beyond a single read.

#[tokio::test]
async fn test_enable_p2p_sharing_produces_exact_sha256_and_size_for_known_bytes() {
    // SHA-256("abc") is the standard FIPS 180-4 example digest --
    // independently cross-checked here against both `shasum -a 256`
    // and `openssl dgst -sha256` (not just recomputed with the same
    // `sha2` crate the implementation under test uses, which would be
    // tautological).
    const KNOWN_CONTENT: &[u8] = b"abc";
    const EXPECTED_SHA256_HEX: &str =
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const EXPECTED_SIZE_BYTES: usize = 3;

    let mut config = NetworkOptimizationConfig::default();
    config.enable_p2p_sharing = true;
    let manager = NetworkOptimizationManager::new(config).expect("manager creation should succeed");

    let path = std::env::temp_dir().join(format!(
        "trustformers_p2p_exact_hash_test_{}.bin",
        std::process::id()
    ));
    std::fs::write(&path, KNOWN_CONTENT).expect("write known-bytes test file");

    manager
        .enable_p2p_sharing("exact-hash-model", &path)
        .await
        .expect("enable_p2p_sharing must succeed through the real public API");

    let (hash, size) = {
        let p2p = manager.p2p_manager.lock().unwrap_or_else(|p| p.into_inner());
        let recorded = p2p
            .shared_models
            .get("exact-hash-model")
            .expect("enable_p2p_sharing must have recorded the model");
        (recorded.model_hash.clone(), recorded.size_bytes)
    };

    assert_eq!(
        hash, EXPECTED_SHA256_HEX,
        "must match the independently-computed SHA-256 digest exactly, not merely differ \
         from a placeholder"
    );
    assert_eq!(size, EXPECTED_SIZE_BYTES);

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn test_enable_p2p_sharing_with_flag_disabled_errors_and_records_nothing() {
    // `NetworkOptimizationConfig::default()` disables P2P sharing
    // (see `test_network_optimization_config_default` above); this
    // locks that the public `enable_p2p_sharing` gate actually
    // refuses to run `add_shared_model` at all in that case, rather
    // than silently succeeding with nothing to hash.
    let manager = NetworkOptimizationManager::new(NetworkOptimizationConfig::default())
        .expect("manager creation should succeed");

    let path = std::env::temp_dir().join(format!(
        "trustformers_p2p_disabled_test_{}.bin",
        std::process::id()
    ));
    std::fs::write(&path, b"abc").expect("write test file");

    let result = manager.enable_p2p_sharing("should-not-be-shared", &path).await;
    assert!(
        result.is_err(),
        "enable_p2p_sharing must refuse when the config flag is disabled"
    );

    let recorded = {
        let p2p = manager.p2p_manager.lock().unwrap_or_else(|p| p.into_inner());
        p2p.shared_models.contains_key("should-not-be-shared")
    };
    assert!(!recorded, "a refused share must not record anything");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_add_shared_model_exact_sha256_across_chunk_boundary() {
    // `add_shared_model` streams the file in 65536-byte chunks
    // (`network_optimization.rs`'s own `chunk = [0u8; 65536]`); this
    // payload is larger than one chunk so the loop's chunk-boundary
    // accumulation is actually exercised, not just a single-read
    // path. Digest cross-validated independently via both Python's
    // `hashlib.sha256` and `shasum -a 256` against the identical
    // `(i % 256) as u8` generation used here.
    const PAYLOAD_LEN: usize = 70_000;
    const EXPECTED_SHA256_HEX: &str =
        "0c6c96cc20d3f906e54f1f1296e8878c1ac39262fb587cd56235c3aa9103d837";

    let payload: Vec<u8> = (0..PAYLOAD_LEN).map(|i| (i % 256) as u8).collect();

    let config = NetworkOptimizationConfig::default();
    let mut manager = P2PManager::new(&config.p2p_config);

    let path = std::env::temp_dir().join(format!(
        "trustformers_p2p_chunk_boundary_test_{}.bin",
        std::process::id()
    ));
    std::fs::write(&path, &payload).expect("write chunk-boundary test file");

    manager
        .add_shared_model("chunk-boundary-model", &path)
        .expect("add_shared_model must succeed");

    let recorded = manager
        .shared_models
        .get("chunk-boundary-model")
        .expect("model must be recorded");

    assert_eq!(recorded.size_bytes, PAYLOAD_LEN);
    assert_eq!(recorded.model_hash, EXPECTED_SHA256_HEX);

    let _ = std::fs::remove_file(&path);
}
