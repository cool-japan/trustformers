#!/bin/bash

# NPM publishing script for trustformers-wasm
# Usage: ./publish-npm.sh [--dry-run] [package-dir]

set -e

DRY_RUN=false
PACKAGE_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            PACKAGE_DIR="$1"
            shift
            ;;
    esac
done

echo "📦 Publishing TrustformeRS WASM to NPM..."

# Validate NPM token
if [ -z "$NPM_TOKEN" ]; then
    echo "❌ Error: NPM_TOKEN environment variable not set"
    exit 1
fi

# Configure NPM authentication
echo "🔐 Configuring NPM authentication..."
echo "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > ~/.npmrc

# Function to publish a package
publish_package() {
    local pkg_dir="$1"
    local pkg_name="$2"
    
    echo ""
    echo "📦 Publishing $pkg_name from $pkg_dir..."
    
    if [ ! -d "$pkg_dir" ]; then
        echo "⚠️ Directory $pkg_dir not found, skipping..."
        return 0
    fi
    
    cd "$pkg_dir"
    
    # Verify package.json exists
    if [ ! -f "package.json" ]; then
        echo "❌ No package.json found in $pkg_dir"
        cd ..
        return 1
    fi
    
    # Extract version from package.json
    PKG_VERSION=$(grep -o '"version": *"[^"]*"' package.json | grep -o '[0-9][^"]*')
    
    echo "  📋 Package: $pkg_name"
    echo "  🏷️ Version: $PKG_VERSION"
    
    # Check if version already exists
    if npm view "$pkg_name@$PKG_VERSION" version &>/dev/null; then
        echo "  ⚠️ Version $PKG_VERSION already exists on NPM, skipping..."
        cd ..
        return 0
    fi
    
    # Validate package
    echo "  🔍 Validating package..."
    npm pack --dry-run > package-validation.log 2>&1
    
    # Check for WASM files
    if ! grep -q "\.wasm$" package-validation.log; then
        echo "  ⚠️ Warning: No WASM files found in package"
    fi
    
    # Check for TypeScript definitions
    if ! grep -q "\.d\.ts$" package-validation.log; then
        echo "  ⚠️ Warning: No TypeScript definitions found in package"
    fi
    
    # Estimate package size
    PACK_SIZE=$(npm pack --dry-run 2>&1 | grep "tarball" | grep -o "[0-9.]*[kMG]*B" | tail -1)
    echo "  📏 Estimated package size: $PACK_SIZE"
    
    # Size check
    PACK_SIZE_BYTES=$(npm pack --dry-run 2>&1 | grep "bytes" | grep -o "[0-9]*" | tail -1)
    if [ -n "$PACK_SIZE_BYTES" ] && [ "$PACK_SIZE_BYTES" -gt 10485760 ]; then # 10MB
        echo "  ⚠️ Warning: Package size ($(numfmt --to=iec $PACK_SIZE_BYTES)) exceeds 10MB"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "  🧪 DRY RUN: Would publish $pkg_name@$PKG_VERSION"
        echo "  📋 Package contents:"
        npm pack --dry-run | head -20
        if [ "$(npm pack --dry-run | wc -l)" -gt 20 ]; then
            echo "  ... and $(( $(npm pack --dry-run | wc -l) - 20 )) more files"
        fi
    else
        echo "  🚀 Publishing to NPM..."
        
        # Publish with appropriate tag
        if [[ "$PKG_VERSION" =~ alpha|beta|rc ]]; then
            echo "  🏷️ Publishing as pre-release (beta tag)"
            npm publish --tag beta --access public
        else
            echo "  🏷️ Publishing as stable release (latest tag)"
            npm publish --access public
        fi
        
        echo "  ✅ Successfully published $pkg_name@$PKG_VERSION"
        
        # Verify publication
        sleep 2
        if npm view "$pkg_name@$PKG_VERSION" version &>/dev/null; then
            echo "  ✅ Publication verified on NPM registry"
        else
            echo "  ⚠️ Warning: Could not verify publication (may take a few minutes to propagate)"
        fi
    fi
    
    # Clean up
    rm -f package-validation.log
    cd ..
}

# Determine which packages to publish
if [ -n "$PACKAGE_DIR" ]; then
    # Publish specific package
    PKG_NAME=$(basename "$PACKAGE_DIR")
    publish_package "$PACKAGE_DIR" "@trustformers/wasm-$PKG_NAME"
else
    # Publish all package variants
    echo "🔍 Discovering packages to publish..."
    
    # Main packages
    for PKG_DIR in pkg-*; do
        if [ -d "$PKG_DIR" ]; then
            TARGET_NAME=$(echo "$PKG_DIR" | sed 's/pkg-//')
            
            # Determine package name based on target
            case "$TARGET_NAME" in
                "web-minimal")
                    PKG_NAME="@trustformers/wasm-web"
                    ;;
                "web-full")
                    PKG_NAME="@trustformers/wasm-web-full"
                    ;;
                "bundler-minimal")
                    PKG_NAME="@trustformers/wasm-bundler"
                    ;;
                "bundler-full")
                    PKG_NAME="@trustformers/wasm-bundler-full"
                    ;;
                "nodejs-minimal")
                    PKG_NAME="@trustformers/wasm-node"
                    ;;
                "nodejs-full")
                    PKG_NAME="@trustformers/wasm-node-full"
                    ;;
                *)
                    PKG_NAME="@trustformers/wasm-$TARGET_NAME"
                    ;;
            esac
            
            publish_package "$PKG_DIR" "$PKG_NAME"
        fi
    done
    
    # Publish main umbrella package if it exists
    if [ -f "package.json" ]; then
        echo ""
        echo "📦 Publishing main umbrella package..."
        
        # Update package.json to include references to sub-packages
        cat > package-umbrella.json << EOF
{
  "name": "@trustformers/wasm",
  "version": "$(grep -o '"version": *"[^"]*"' package.json | grep -o '[0-9][^"]*')",
  "description": "WebAssembly bindings for TrustformeRS transformer library - umbrella package",
  "main": "index.js",
  "types": "index.d.ts",
  "files": [
    "index.js",
    "index.d.ts",
    "README.md"
  ],
  "scripts": {
    "postinstall": "node install-target.js"
  },
  "dependencies": {},
  "peerDependencies": {
    "@trustformers/wasm-web": "$(grep -o '"version": *"[^"]*"' package.json | grep -o '[0-9][^"]*')",
    "@trustformers/wasm-bundler": "$(grep -o '"version": *"[^"]*"' package.json | grep -o '[0-9][^"]*')",
    "@trustformers/wasm-node": "$(grep -o '"version": *"[^"]*"' package.json | grep -o '[0-9][^"]*')"
  },
  "keywords": [
    "webassembly",
    "wasm",
    "transformers",
    "machine-learning",
    "nlp",
    "ai",
    "neural-networks"
  ],
  "author": "TrustformeRS Team",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/your-org/trustformers.git",
    "directory": "trustformers-wasm"
  },
  "bugs": {
    "url": "https://github.com/your-org/trustformers/issues"
  },
  "homepage": "https://trustformers.dev"
}
EOF
        
        # Create umbrella package files
        cat > index.js << 'EOF'
// TrustformeRS WASM Umbrella Package
// This package helps automatically install the correct target-specific package

console.log('🚀 TrustformeRS WASM: Auto-detecting environment...');

// Environment detection
function detectEnvironment() {
    if (typeof window !== 'undefined' && typeof document !== 'undefined') {
        // Browser environment
        if (typeof module !== 'undefined' && module.exports) {
            return 'bundler'; // Browser with bundler
        } else {
            return 'web'; // Direct browser usage
        }
    } else if (typeof global !== 'undefined' && typeof process !== 'undefined') {
        return 'node'; // Node.js environment
    } else if (typeof importScripts !== 'undefined') {
        return 'worker'; // Web Worker
    } else {
        return 'unknown';
    }
}

const env = detectEnvironment();

console.log(`📍 Detected environment: ${env}`);
console.log('💡 Install the appropriate target-specific package:');

switch (env) {
    case 'web':
        console.log('   npm install @trustformers/wasm-web');
        break;
    case 'bundler':
        console.log('   npm install @trustformers/wasm-bundler');
        break;
    case 'node':
        console.log('   npm install @trustformers/wasm-node');
        break;
    default:
        console.log('   See https://docs.trustformers.dev/wasm for installation guide');
}

module.exports = {
    detectEnvironment,
    recommendedPackage: `@trustformers/wasm-${env}`
};
EOF
        
        cat > index.d.ts << 'EOF'
// TrustformeRS WASM Umbrella Package TypeScript Definitions

export function detectEnvironment(): 'web' | 'bundler' | 'node' | 'worker' | 'unknown';
export const recommendedPackage: string;
EOF
        
        cat > install-target.js << 'EOF'
#!/usr/bin/env node

const { detectEnvironment } = require('./index.js');

const env = detectEnvironment();
const packageMap = {
    'web': '@trustformers/wasm-web',
    'bundler': '@trustformers/wasm-bundler', 
    'node': '@trustformers/wasm-node',
    'worker': '@trustformers/wasm-web'
};

const targetPackage = packageMap[env];

if (targetPackage) {
    console.log(`🎯 Recommended package for ${env}: ${targetPackage}`);
    console.log('📦 Install with: npm install ' + targetPackage);
} else {
    console.log('❓ Could not detect environment. See https://docs.trustformers.dev/wasm');
}
EOF
        
        # Use umbrella package.json
        mv package.json package-original.json
        mv package-umbrella.json package.json
        
        if [ "$DRY_RUN" = true ]; then
            echo "  🧪 DRY RUN: Would publish umbrella package @trustformers/wasm"
        else
            echo "  🚀 Publishing umbrella package..."
            npm publish --access public
            echo "  ✅ Successfully published umbrella package @trustformers/wasm"
        fi
        
        # Restore original package.json
        mv package-original.json package.json
        rm -f index.js index.d.ts install-target.js
    fi
fi

# Final summary
echo ""
echo "📊 NPM Publishing Summary:"
echo "  🏷️ Mode: $([ "$DRY_RUN" = true ] && echo "DRY RUN" || echo "LIVE PUBLISH")"
echo "  📦 Registry: https://www.npmjs.com/"
echo "  🔍 Search packages: https://www.npmjs.com/search?q=%40trustformers%2Fwasm"

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "✅ Publishing complete!"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Verify packages at: https://www.npmjs.com/org/trustformers"
    echo "  2. Test installation: npm install @trustformers/wasm"
    echo "  3. Update documentation with new version numbers"
    echo "  4. Announce release on social media/blog"
else
    echo ""
    echo "🧪 Dry run complete! Remove --dry-run flag to publish for real."
fi

# Cleanup
rm -f ~/.npmrc

echo ""
echo "🎉 NPM publishing process finished!"