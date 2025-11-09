#!/bin/bash

# Security audit script for TrustformeRS WASM
# This script performs security checks on dependencies and build outputs

set -e

echo "🔒 TrustformeRS WASM Security Audit"
echo "=================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for cargo-audit
if command_exists cargo-audit; then
    echo "📋 Running dependency vulnerability scan..."
    # Try to run from workspace root if possible, fallback to local
    if [ -f "../Cargo.lock" ]; then
        cd .. && cargo audit --quiet --color never || echo "⚠️  Audit failed or found issues"
        cd trustformers-wasm
    else
        cargo audit --quiet --color never || echo "⚠️  Audit failed or found issues"
    fi
else
    echo "⚠️  cargo-audit not installed. Install with: cargo install cargo-audit"
fi

# Check for known security patterns in source code
echo "🔍 Scanning for security patterns..."

# Check for unsafe code blocks
unsafe_count=$(find src -name "*.rs" -exec grep -l "unsafe" {} \; 2>/dev/null | wc -l || echo 0)
if [ "$unsafe_count" -gt 0 ]; then
    echo "⚠️  Found $unsafe_count files with unsafe code blocks"
    find src -name "*.rs" -exec grep -l "unsafe" {} \; 2>/dev/null | head -5
else
    echo "✅ No unsafe code blocks found"
fi

# Check for potential secret leaks (simple patterns)
echo "🔐 Checking for potential secrets..."
secret_patterns=("password" "secret" "token" "key" "api_key" "private_key")
for pattern in "${secret_patterns[@]}"; do
    matches=$(find src -name "*.rs" -exec grep -i "$pattern" {} \; 2>/dev/null | wc -l || echo 0)
    if [ "$matches" -gt 0 ]; then
        echo "⚠️  Found $matches potential secret references for '$pattern'"
    fi
done

# Check for debug/println statements that might leak info
debug_count=$(find src -name "*.rs" -exec grep -E "(println!|dbg!|eprintln!)" {} \; 2>/dev/null | wc -l || echo 0)
if [ "$debug_count" -gt 0 ]; then
    echo "⚠️  Found $debug_count debug print statements (review for info leaks)"
else
    echo "✅ No debug print statements found"
fi

# Check WASM binary size for potential bloat
echo "🗜️  Checking WASM binary size..."
if [ -f "pkg/trustformers_wasm_bg.wasm" ]; then
    size=$(stat -f%z "pkg/trustformers_wasm_bg.wasm" 2>/dev/null || stat -c%s "pkg/trustformers_wasm_bg.wasm" 2>/dev/null || echo "unknown")
    size_mb=$((size / 1024 / 1024))
    if [ "$size_mb" -gt 5 ]; then
        echo "⚠️  WASM binary is ${size_mb}MB (consider size optimization)"
    else
        echo "✅ WASM binary size acceptable: ${size_mb}MB"
    fi
else
    echo "📝 WASM binary not found (run 'wasm-pack build' first)"
fi

# Check for permissions in package.json if exists
if [ -f "pkg/package.json" ]; then
    echo "📦 Checking package.json configuration..."
    if grep -q '"private": false' pkg/package.json 2>/dev/null; then
        echo "⚠️  Package is public - ensure no sensitive data in pkg/"
    fi
fi

# Dependency license check
echo "📋 Checking dependency licenses..."
if command_exists cargo-license; then
    cargo license --color never 2>/dev/null | grep -E "(GPL|AGPL|LGPL)" && echo "⚠️  Found copyleft licenses" || echo "✅ No restrictive licenses found"
else
    echo "💡 Install cargo-license for license checking: cargo install cargo-license"
fi

echo ""
echo "✅ Security audit complete"
echo "💡 Consider running this script regularly and before releases"