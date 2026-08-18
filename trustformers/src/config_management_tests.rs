//! Unit tests for [`super`] -- split out of `config_management.rs` to keep
//! it under the workspace's 2000-line-per-file policy (see the `#[path =
//! "config_management_tests.rs"] mod tests;` declaration at the bottom of
//! that file; the same convention is used by
//! `pipeline/adaptive_inference.rs` and `auto/optimizers/mod.rs`).

use super::*;

#[test]
fn test_configuration_manager_creation() {
    let manager = ConfigurationManager::new();
    assert!(manager.schema_registry.contains_key("training"));
    assert!(manager.schema_registry.contains_key("model"));
    assert!(manager.schema_registry.contains_key("conversational"));
}

#[test]
fn test_validation_success() {
    let manager = ConfigurationManager::new();
    let config = serde_json::json!({
        "num_epochs": 5,
        "batch_size": 32,
        "learning_rate": 2e-5
    });

    let result = manager.validate_config("training", &config);
    assert!(result.is_valid);
    assert!(result.errors.is_empty());
}

#[test]
fn test_validation_missing_required_field() {
    let manager = ConfigurationManager::new();
    let config = serde_json::json!({
        "num_epochs": 5,
        "batch_size": 32
        // missing learning_rate
    });

    let result = manager.validate_config("training", &config);
    assert!(!result.is_valid);
    assert_eq!(result.errors.len(), 1);
    assert!(matches!(
        result.errors[0].error_type,
        ValidationErrorType::MissingRequiredField
    ));
}

#[test]
fn test_validation_type_mismatch() {
    let manager = ConfigurationManager::new();
    let config = serde_json::json!({
        "num_epochs": "not_a_number",
        "batch_size": 32,
        "learning_rate": 2e-5
    });

    let result = manager.validate_config("training", &config);
    assert!(!result.is_valid);
    assert!(result
        .errors
        .iter()
        .any(|e| matches!(e.error_type, ValidationErrorType::TypeMismatch)));
}

#[test]
fn test_migration() {
    let manager = ConfigurationManager::new();
    let old_config = serde_json::json!({
        "num_epochs": 5,
        "batch_size": 32,
        "learning_rate": 2e-5
    });

    let migrated = manager
        .migrate_config("training", &old_config, "1.0.0", "2.0.0")
        .expect("operation failed in test");

    assert!(migrated.get("gradient_accumulation_steps").is_some());
    assert!(migrated.get("warmup_steps").is_some());
}

// -------------------------------------------------------------------
// Multi-hop migration paths: regression coverage for the bug where
// `migrate_config` found a multi-hop path only when the migrations
// `Vec` happened to already be ordered to match the path being walked.
// A real graph search (`find_migration_path`, BFS) must find a valid
// path regardless of registration order.
// -------------------------------------------------------------------

fn passthrough_migration(from: &str, to: &str, marker: &'static str) -> Migration {
    Migration::new(from, to, "test migration", move |config| {
        let mut new_config = config.clone();
        if let serde_json::Value::Object(map) = &mut new_config {
            map.insert(marker.to_string(), serde_json::Value::Bool(true));
        }
        Ok(new_config)
    })
}

#[test]
fn test_multi_hop_migration_applies_every_migration_in_path_order() {
    // 1.0.0 -> 2.0.0 -> 3.0.0, with only the two direct one-hop
    // migrations registered -- no direct 1.0.0 -> 3.0.0 migration
    // exists. The old single-pass code could resolve this only when
    // `migrations` happened to already be in the right order; this
    // must succeed unconditionally.
    let mut manager = ConfigurationManager::new();
    manager.register_migration(
        "multihop".to_string(),
        passthrough_migration("1.0.0", "2.0.0", "step_one_applied"),
    );
    manager.register_migration(
        "multihop".to_string(),
        passthrough_migration("2.0.0", "3.0.0", "step_two_applied"),
    );

    let config = serde_json::json!({});
    let migrated = manager
        .migrate_config("multihop", &config, "1.0.0", "3.0.0")
        .expect("a valid two-hop path must be found");

    assert_eq!(
        migrated.get("step_one_applied"),
        Some(&serde_json::Value::Bool(true)),
        "the first hop's migration must have run"
    );
    assert_eq!(
        migrated.get("step_two_applied"),
        Some(&serde_json::Value::Bool(true)),
        "the second hop's migration must have run"
    );
}

#[test]
fn test_multi_hop_migration_succeeds_regardless_of_registration_order() {
    // Same two-hop chain as above, but registered in *reverse* order
    // (the second hop registered before the first). Regression: the
    // old `is_version_on_path`/linear-pass code depended on `migrations`
    // already being ordered to match the path, so this exact
    // registration order would previously fail to find a path that
    // plainly exists.
    let mut manager = ConfigurationManager::new();
    manager.register_migration(
        "reverse_order".to_string(),
        passthrough_migration("2.0.0", "3.0.0", "step_two_applied"),
    );
    manager.register_migration(
        "reverse_order".to_string(),
        passthrough_migration("1.0.0", "2.0.0", "step_one_applied"),
    );

    let config = serde_json::json!({});
    let migrated = manager.migrate_config("reverse_order", &config, "1.0.0", "3.0.0").expect(
        "a valid two-hop path must be found regardless of the order migrations were \
             registered in",
    );

    assert_eq!(
        migrated.get("step_one_applied"),
        Some(&serde_json::Value::Bool(true))
    );
    assert_eq!(
        migrated.get("step_two_applied"),
        Some(&serde_json::Value::Bool(true))
    );
}

#[test]
fn test_three_hop_migration_chain() {
    // 1.0.0 -> 2.0.0 -> 3.0.0 -> 4.0.0: verifies the search is not
    // hardcoded to two hops.
    let mut manager = ConfigurationManager::new();
    manager.register_migration(
        "threehop".to_string(),
        passthrough_migration("3.0.0", "4.0.0", "hop_three"),
    );
    manager.register_migration(
        "threehop".to_string(),
        passthrough_migration("1.0.0", "2.0.0", "hop_one"),
    );
    manager.register_migration(
        "threehop".to_string(),
        passthrough_migration("2.0.0", "3.0.0", "hop_two"),
    );

    let config = serde_json::json!({});
    let migrated = manager
        .migrate_config("threehop", &config, "1.0.0", "4.0.0")
        .expect("a valid three-hop path must be found");

    for marker in ["hop_one", "hop_two", "hop_three"] {
        assert_eq!(
            migrated.get(marker),
            Some(&serde_json::Value::Bool(true)),
            "migration `{marker}` must have been applied"
        );
    }
}

#[test]
fn test_migration_prefers_direct_path_over_longer_detour() {
    // Both a direct 1.0.0 -> 3.0.0 migration and a longer detour
    // (1.0.0 -> 2.0.0 -> 3.0.0) are registered. BFS finds the
    // shortest path, so only the direct migration's marker must be
    // present -- proving the detour's migrations were not also
    // (redundantly, or wrongly-ordered) applied.
    let mut manager = ConfigurationManager::new();
    manager.register_migration(
        "shortest".to_string(),
        passthrough_migration("1.0.0", "2.0.0", "detour_hop_one"),
    );
    manager.register_migration(
        "shortest".to_string(),
        passthrough_migration("2.0.0", "3.0.0", "detour_hop_two"),
    );
    manager.register_migration(
        "shortest".to_string(),
        passthrough_migration("1.0.0", "3.0.0", "direct_hop"),
    );

    let config = serde_json::json!({});
    let migrated = manager
        .migrate_config("shortest", &config, "1.0.0", "3.0.0")
        .expect("a path must be found");

    assert_eq!(
        migrated.get("direct_hop"),
        Some(&serde_json::Value::Bool(true)),
        "the shortest (direct) path must be used"
    );
    assert_eq!(
        migrated.get("detour_hop_one"),
        None,
        "the longer detour must not be taken when a direct path exists"
    );
}

#[test]
fn test_migration_no_path_is_structured_error_not_silent_passthrough() {
    // 1.0.0 -> 2.0.0 is registered, but nothing reaches 9.9.9: this
    // must be a real error, not a silent no-op that returns the
    // original config unmigrated.
    let mut manager = ConfigurationManager::new();
    manager.register_migration(
        "nopath".to_string(),
        passthrough_migration("1.0.0", "2.0.0", "applied"),
    );

    let config = serde_json::json!({ "original": true });
    let result = manager.migrate_config("nopath", &config, "1.0.0", "9.9.9");
    assert!(
        result.is_err(),
        "an unreachable target version must be a structured error"
    );
}

#[test]
fn test_migration_same_version_is_identity_without_registered_migrations() {
    // from_version == to_version must succeed trivially (nothing to
    // migrate), even though no migration whose `from_version` equals
    // `to_version` exists in the registry.
    let mut manager = ConfigurationManager::new();
    manager.register_migration(
        "identity".to_string(),
        passthrough_migration("1.0.0", "2.0.0", "applied"),
    );

    let config = serde_json::json!({ "unchanged": true });
    let migrated = manager
        .migrate_config("identity", &config, "1.0.0", "1.0.0")
        .expect("migrating a version to itself must succeed");
    assert_eq!(
        migrated, config,
        "identity migration must not alter the config"
    );
}

#[test]
fn test_migration_graph_with_cycle_does_not_infinite_loop() {
    // A migration graph containing a cycle (2.0.0 -> 1.0.0, back to the
    // start) must not cause the BFS to loop forever; it must still find
    // the real forward path.
    let mut manager = ConfigurationManager::new();
    manager.register_migration(
        "cyclic".to_string(),
        passthrough_migration("1.0.0", "2.0.0", "forward"),
    );
    manager.register_migration(
        "cyclic".to_string(),
        passthrough_migration("2.0.0", "1.0.0", "backward"),
    );
    manager.register_migration(
        "cyclic".to_string(),
        passthrough_migration("2.0.0", "3.0.0", "onward"),
    );

    let config = serde_json::json!({});
    let migrated = manager
        .migrate_config("cyclic", &config, "1.0.0", "3.0.0")
        .expect("a cycle elsewhere in the graph must not prevent finding the real path");
    assert_eq!(
        migrated.get("forward"),
        Some(&serde_json::Value::Bool(true))
    );
    assert_eq!(migrated.get("onward"), Some(&serde_json::Value::Bool(true)));
}

#[test]
fn test_template_generation() {
    let manager = ConfigurationManager::new();
    let template = manager.generate_template("training").expect("temp file creation failed");

    assert!(template.get("num_epochs").is_some());
    assert!(template.get("batch_size").is_some());
    assert!(template.get("learning_rate").is_some());
}

#[test]
fn test_config_comparison() {
    let manager = ConfigurationManager::new();

    let config1 = serde_json::json!({
        "num_epochs": 5,
        "batch_size": 32
    });

    let config2 = serde_json::json!({
        "num_epochs": 10,
        "learning_rate": 2e-5
    });

    let comparison = manager.compare_configs(&config1, &config2);

    assert_eq!(comparison.modified_fields.len(), 1); // num_epochs changed
    assert_eq!(comparison.added_fields.len(), 1); // learning_rate added
    assert_eq!(comparison.removed_fields.len(), 1); // batch_size removed
}

#[test]
fn test_preset_creation() {
    let manager = ConfigurationManager::new();

    let config = manager
        .create_from_preset(
            "training",
            "fast_development",
            Some(HashMap::from([(
                "batch_size".to_string(),
                serde_json::Value::Number(serde_json::Number::from(16)),
            )])),
        )
        .expect("operation failed in test");

    assert_eq!(
        config.get("num_epochs").expect("expected value not found"),
        &serde_json::Value::Number(serde_json::Number::from(3))
    );
    assert_eq!(
        config.get("batch_size").expect("expected value not found"),
        &serde_json::Value::Number(serde_json::Number::from(16))
    ); // overridden
}

#[test]
fn test_recommendations() {
    let manager = ConfigurationManager::new();

    let config = serde_json::json!({
        "batch_size": 8,
        "learning_rate": 1e-2 // Very high learning rate
    });

    let context = RecommendationContext {
        hardware_info: HashMap::from([(
            "gpu_memory_gb".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(16.0).expect("operation failed in test"),
            ),
        )]),
        use_case: "production".to_string(),
        performance_requirements: PerformanceRequirements {
            max_latency_ms: None,
            min_throughput: None,
            memory_budget_gb: None,
            power_budget_watts: None,
        },
        constraints: vec![],
    };

    let recommendations = manager.get_recommendations("training", &config, &context);

    assert!(!recommendations.is_empty());
    assert!(recommendations.iter().any(|r| r.field == "batch_size"));
    assert!(recommendations.iter().any(|r| r.field == "learning_rate"));
}

#[test]
fn test_unknown_config_type() {
    let manager = ConfigurationManager::new();
    let config = serde_json::json!({"test": "value"});

    let result = manager.validate_config("unknown_type", &config);
    assert!(!result.is_valid);
    assert!(matches!(
        result.errors[0].error_type,
        ValidationErrorType::UnknownConfigType
    ));
}

#[test]
fn test_constraint_validation() {
    let manager = ConfigurationManager::new();
    let config = serde_json::json!({
        "num_epochs": -1, // Violates minimum value constraint
        "batch_size": 32,
        "learning_rate": 2e-5
    });

    let result = manager.validate_config("training", &config);
    assert!(!result.is_valid);
    assert!(result
        .errors
        .iter()
        .any(|e| matches!(e.error_type, ValidationErrorType::ConstraintViolation)));
}

// -------------------------------------------------------------------
// Conditional requirements driven by the real expression evaluator
// (`crate::config_condition`). Regression coverage for the bug where
// `evaluate_condition` only understood `==` on string fields and
// silently returned `false` for everything else, so conditional
// requirements built on `!=`/`<`/`&&`/etc. could never fire.
// -------------------------------------------------------------------

fn schema_with_condition(condition: &str) -> ConfigSchema {
    ConfigSchema {
        name: "conditional-test".to_string(),
        version: "1.0.0".to_string(),
        description: "schema exercising conditional_requirements".to_string(),
        fields: HashMap::new(),
        required_fields: HashSet::new(),
        conditional_requirements: vec![ConditionalRequirement {
            condition: condition.to_string(),
            required_fields: vec!["api_key".to_string()],
        }],
    }
}

#[test]
fn test_conditional_requirement_fires_on_not_equal() {
    // `!=` was entirely unsupported by the old evaluator.
    let validator = ConfigValidator::new();
    let schema = schema_with_condition("environment != \"local\"");
    let config = serde_json::json!({ "environment": "production" });
    let result = validator.validate(&config, &schema);
    assert!(
        !result.is_valid,
        "missing api_key must be flagged when environment != local"
    );
    assert!(result.errors.iter().any(|e| matches!(
        e.error_type,
        ValidationErrorType::ConditionalRequirementNotMet
    )));
}

#[test]
fn test_conditional_requirement_does_not_fire_when_condition_false() {
    let validator = ConfigValidator::new();
    let schema = schema_with_condition("environment != \"local\"");
    let config = serde_json::json!({ "environment": "local" });
    let result = validator.validate(&config, &schema);
    assert!(
        result.is_valid,
        "condition is false, so the missing api_key must not be flagged"
    );
}

#[test]
fn test_conditional_requirement_fires_on_numeric_comparison() {
    // `<`/`>`/etc. were entirely unsupported by the old evaluator.
    let validator = ConfigValidator::new();
    let schema = schema_with_condition("batch_size > 64");
    let config = serde_json::json!({ "batch_size": 128 });
    let result = validator.validate(&config, &schema);
    assert!(!result.is_valid);
}

#[test]
fn test_conditional_requirement_fires_on_and_with_precedence() {
    // Exercises `&&` binding tighter than `||`, end to end through
    // `ConfigValidator::validate`, not just the standalone evaluator.
    let validator = ConfigValidator::new();
    let schema = schema_with_condition("region == \"eu\" || tier == \"pro\" && strict == true");
    let config = serde_json::json!({ "region": "us", "tier": "pro", "strict": true });
    let result = validator.validate(&config, &schema);
    assert!(
        !result.is_valid,
        "`tier == pro && strict == true` must fire even though `region == eu` is false"
    );
}

#[test]
fn test_conditional_requirement_satisfied_field_present_is_valid() {
    let validator = ConfigValidator::new();
    let schema = schema_with_condition("environment != \"local\"");
    let config = serde_json::json!({ "environment": "production", "api_key": "secret" });
    let result = validator.validate(&config, &schema);
    assert!(
        result.is_valid,
        "api_key is present, so the condition being true is fine"
    );
}

#[test]
fn test_malformed_condition_is_structured_error_not_silent_pass() {
    // Regression test for the exact bug: the old evaluator would treat
    // any operator it didn't recognize as "condition not met" and
    // silently return `false`, so a schema author's typo (`<>` instead
    // of `!=`) would validate cleanly forever. The new evaluator must
    // surface this as a validation error instead.
    let validator = ConfigValidator::new();
    let schema = schema_with_condition("environment <> \"local\"");
    let config = serde_json::json!({ "environment": "production" });
    let result = validator.validate(&config, &schema);
    assert!(
        !result.is_valid,
        "a malformed condition must fail validation, not pass silently"
    );
    assert!(
        result
            .errors
            .iter()
            .any(|e| matches!(e.error_type, ValidationErrorType::InvalidCondition)),
        "the malformed condition must be reported as InvalidCondition"
    );
}

#[test]
fn test_malformed_condition_error_message_names_the_condition() {
    let validator = ConfigValidator::new();
    let schema = schema_with_condition("a <> b");
    let config = serde_json::json!({});
    let result = validator.validate(&config, &schema);
    let invalid = result
        .errors
        .iter()
        .find(|e| matches!(e.error_type, ValidationErrorType::InvalidCondition))
        .expect("must report InvalidCondition");
    assert!(
        invalid.message.contains("a <> b"),
        "error message should include the offending condition text: {}",
        invalid.message
    );
}
