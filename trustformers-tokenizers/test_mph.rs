use std::collections::HashMap;

// Mock trustformers_core for testing
mod trustformers_core {
    pub mod error {
        use std::fmt;

        #[derive(Debug)]
        pub enum CoreError {
            TokenizerError(String),
        }

        impl fmt::Display for CoreError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    TrustformersError::TokenizerError(msg) => write!(f, "Tokenizer error: {}", msg),
                }
            }
        }

        impl std::error::Error for CoreError {}

        pub type Result<T> = std::result::Result<T, CoreError>;
    }
}

// Include the minimal perfect hash module directly
mod minimal_perfect_hash;

use minimal_perfect_hash::*;

fn main() {
    println!("Testing Minimal Perfect Hash implementation...");

    // Test 1: Basic creation and lookup
    let keys = vec![
        "hello".to_string(),
        "world".to_string(),
        "test".to_string(),
        "minimal".to_string(),
        "perfect".to_string(),
        "hash".to_string(),
    ];

    let config = MinimalPerfectHashConfig::default();

    match MinimalPerfectHash::new(&keys, config) {
        Ok(mph) => {
            println!("✓ Successfully created minimal perfect hash");
            println!("  Hash table size: {}", mph.len());

            // Test lookups
            let mut success_count = 0;
            for key in &keys {
                if mph.contains(key) {
                    success_count += 1;
                    println!("  ✓ Found key: {}", key);
                } else {
                    println!("  ✗ Missing key: {}", key);
                }
            }

            println!("  Found {}/{} keys", success_count, keys.len());

            // Test non-existent keys
            let non_existent = vec!["missing", "nonexistent", "absent"];
            for key in &non_existent {
                if !mph.contains(key) {
                    println!("  ✓ Correctly rejected non-existent key: {}", key);
                } else {
                    println!("  ✗ Incorrectly found non-existent key: {}", key);
                }
            }

            // Test memory usage
            let usage = mph.memory_usage();
            println!("  Memory usage: {}", usage);
        }
        Err(e) => {
            println!("✗ Failed to create minimal perfect hash: {:?}", e);
        }
    }

    // Test 2: Vocabulary implementation
    println!("\nTesting MinimalPerfectHashVocab...");

    let tokens = vec![
        "the".to_string(),
        "quick".to_string(),
        "brown".to_string(),
        "fox".to_string(),
        "jumps".to_string(),
        "over".to_string(),
        "lazy".to_string(),
        "dog".to_string(),
    ];

    match MinimalPerfectHashVocab::new(tokens.clone()) {
        Ok(vocab) => {
            println!("✓ Successfully created vocabulary");
            println!("  Vocabulary size: {}", vocab.size());

            // Test token-to-ID mapping
            for (expected_id, token) in tokens.iter().enumerate() {
                match vocab.get_id(token) {
                    Some(id) => {
                        if id == expected_id as u32 {
                            println!("  ✓ Token '{}' -> ID {}", token, id);
                        } else {
                            println!("  ✗ Token '{}' -> ID {} (expected {})", token, id, expected_id);
                        }
                    }
                    None => {
                        println!("  ✗ Token '{}' not found", token);
                    }
                }
            }

            // Test ID-to-token mapping
            for (id, expected_token) in tokens.iter().enumerate() {
                match vocab.get_token(id as u32) {
                    Some(token) => {
                        if token == expected_token {
                            println!("  ✓ ID {} -> Token '{}'", id, token);
                        } else {
                            println!("  ✗ ID {} -> Token '{}' (expected '{}')", id, token, expected_token);
                        }
                    }
                    None => {
                        println!("  ✗ ID {} not found", id);
                    }
                }
            }

            // Test efficiency comparison
            let comparison = vocab.efficiency_comparison();
            println!("  Efficiency: {}", comparison);
        }
        Err(e) => {
            println!("✗ Failed to create vocabulary: {:?}", e);
        }
    }

    println!("\nMinimal Perfect Hash testing completed!");
}