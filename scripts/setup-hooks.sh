#!/usr/bin/env bash
# Setup git hooks for TrustformeRS code style automation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Setting up TrustformeRS git hooks..."

# Create hooks directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/.git/hooks"

# Create pre-commit hook
cat > "${PROJECT_ROOT}/.git/hooks/pre-commit" << 'EOF'
#!/usr/bin/env bash
# TrustformeRS pre-commit hook

set -euo pipefail

echo "Running pre-commit checks..."

# Check if pre-commit is installed
if command -v pre-commit &> /dev/null; then
    pre-commit run
else
    echo "Warning: pre-commit not installed. Running basic checks..."
    
    # Run rustfmt
    if command -v rustfmt &> /dev/null; then
        echo "Checking code formatting..."
        cargo fmt --all -- --check || {
            echo "Error: Code formatting issues found. Run 'cargo fmt' to fix."
            exit 1
        }
    fi
    
    # Run clippy
    if command -v cargo-clippy &> /dev/null; then
        echo "Running clippy..."
        cargo clippy --all-targets --all-features -- -D warnings || {
            echo "Error: Clippy warnings found."
            exit 1
        }
    fi
fi

echo "Pre-commit checks passed!"
EOF

# Create commit-msg hook
cat > "${PROJECT_ROOT}/.git/hooks/commit-msg" << 'EOF'
#!/usr/bin/env bash
# TrustformeRS commit message hook

set -euo pipefail

COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# Check commit message format
# Expected format: <type>(<scope>): <subject>
# Example: feat(core): add new tensor operations

PATTERN='^(feat|fix|docs|style|refactor|perf|test|chore|build|ci|revert)(\([a-z\-]+\))?: .{1,50}'

if ! echo "$COMMIT_MSG" | grep -qE "$PATTERN"; then
    echo "Error: Invalid commit message format!"
    echo ""
    echo "Expected format: <type>(<scope>): <subject>"
    echo ""
    echo "Types:"
    echo "  feat     - A new feature"
    echo "  fix      - A bug fix"
    echo "  docs     - Documentation changes"
    echo "  style    - Code style changes (formatting, etc)"
    echo "  refactor - Code refactoring"
    echo "  perf     - Performance improvements"
    echo "  test     - Test additions or modifications"
    echo "  chore    - Maintenance tasks"
    echo "  build    - Build system changes"
    echo "  ci       - CI/CD changes"
    echo "  revert   - Revert a previous commit"
    echo ""
    echo "Scopes (optional):"
    echo "  core, models, optim, data, tokenizers, training, py, mobile, wasm, etc."
    echo ""
    echo "Example: feat(models): add LLaMA 3 model implementation"
    echo ""
    exit 1
fi

# Check commit message length
SUBJECT_LINE=$(echo "$COMMIT_MSG" | head -n1)
if [ ${#SUBJECT_LINE} -gt 72 ]; then
    echo "Error: Commit subject line is too long (${#SUBJECT_LINE} > 72 characters)"
    exit 1
fi

echo "Commit message format is valid!"
EOF

# Create pre-push hook
cat > "${PROJECT_ROOT}/.git/hooks/pre-push" << 'EOF'
#!/usr/bin/env bash
# TrustformeRS pre-push hook

set -euo pipefail

echo "Running pre-push checks..."

# Run tests
echo "Running tests..."
if command -v cargo-nextest &> /dev/null; then
    cargo nextest run --no-fail-fast || {
        echo "Error: Tests failed. Fix failing tests before pushing."
        exit 1
    }
else
    cargo test || {
        echo "Error: Tests failed. Fix failing tests before pushing."
        exit 1
    }
fi

# Run security audit
if command -v cargo-audit &> /dev/null; then
    echo "Running security audit..."
    cargo audit || {
        echo "Warning: Security vulnerabilities found. Consider fixing before pushing."
        # Don't fail on audit issues, just warn
    }
fi

echo "Pre-push checks passed!"
EOF

# Make hooks executable
chmod +x "${PROJECT_ROOT}/.git/hooks/pre-commit"
chmod +x "${PROJECT_ROOT}/.git/hooks/commit-msg"
chmod +x "${PROJECT_ROOT}/.git/hooks/pre-push"

echo "Git hooks installed successfully!"

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo ""
    echo "Note: pre-commit is not installed. For full functionality, install it with:"
    echo "  pip install pre-commit"
    echo "  pre-commit install"
fi

echo ""
echo "Hooks installed:"
echo "  - pre-commit: Runs formatting and linting checks"
echo "  - commit-msg: Validates commit message format"
echo "  - pre-push: Runs tests and security audit"
echo ""
echo "To bypass hooks (not recommended), use --no-verify flag"