#!/bin/bash
# TrustformeRS-C Universal Binary Builder for macOS
# This script builds universal binaries for both x86_64 and ARM64 architectures

set -e

echo "Building TrustformeRS-C Universal Binary..."

# Configuration
CRATE_NAME="trustformers_c"
TARGET_DIR="target/universal"
TARGETS=("x86_64-apple-darwin" "aarch64-apple-darwin")

# Clean previous builds
echo "Cleaning previous builds..."
cargo clean

# Create target directory
mkdir -p "$TARGET_DIR"

# Build for each target
for target in "${TARGETS[@]}"; do
    echo "Building for $target..."
    cargo build --release --target "$target" --features "universal-binary"
    
    # Check if build succeeded
    if [ ! -f "target/$target/release/lib$CRATE_NAME.dylib" ]; then
        echo "Error: Build failed for $target"
        exit 1
    fi
done

# Create universal binary
echo "Creating universal binary..."
lipo -create \
    "target/x86_64-apple-darwin/release/lib$CRATE_NAME.dylib" \
    "target/aarch64-apple-darwin/release/lib$CRATE_NAME.dylib" \
    -output "$TARGET_DIR/lib$CRATE_NAME.dylib"

# Create universal static library
echo "Creating universal static library..."
lipo -create \
    "target/x86_64-apple-darwin/release/lib$CRATE_NAME.a" \
    "target/aarch64-apple-darwin/release/lib$CRATE_NAME.a" \
    -output "$TARGET_DIR/lib$CRATE_NAME.a"

# Copy header file from one of the targets
echo "Copying header file..."
cp "target/include/trustformers-c/trustformers.h" "$TARGET_DIR/"

# Generate Info.plist for the framework
echo "Generating framework structure..."
FRAMEWORK_DIR="$TARGET_DIR/TrustformeRS.framework"
mkdir -p "$FRAMEWORK_DIR/Versions/A"
mkdir -p "$FRAMEWORK_DIR/Versions/A/Headers"
mkdir -p "$FRAMEWORK_DIR/Versions/A/Resources"

# Copy files to framework
cp "$TARGET_DIR/lib$CRATE_NAME.dylib" "$FRAMEWORK_DIR/Versions/A/TrustformeRS"
cp "$TARGET_DIR/trustformers.h" "$FRAMEWORK_DIR/Versions/A/Headers/"

# Create symbolic links
cd "$FRAMEWORK_DIR"
ln -sf "Versions/A/TrustformeRS" "TrustformeRS"
ln -sf "Versions/A/Headers" "Headers"
ln -sf "Versions/A/Resources" "Resources"
ln -sf "A" "Versions/Current"
cd - > /dev/null

# Create Info.plist
cat > "$FRAMEWORK_DIR/Versions/A/Resources/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleExecutable</key>
    <string>TrustformeRS</string>
    <key>CFBundleIdentifier</key>
    <string>com.cooljapan.trustformers</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>TrustformeRS</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>$(cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "trustformers-c") | .version')</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleVersion</key>
    <string>$(cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "trustformers-c") | .version')</string>
    <key>NSPrincipalClass</key>
    <string></string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>MacOSX</string>
    </array>
    <key>LSMinimumSystemVersion</key>
    <string>10.12</string>
</dict>
</plist>
EOF

# Verify the universal binary
echo "Verifying universal binary..."
file "$TARGET_DIR/lib$CRATE_NAME.dylib"
lipo -info "$TARGET_DIR/lib$CRATE_NAME.dylib"

echo "Universal binary build completed!"
echo "Dynamic library: $TARGET_DIR/lib$CRATE_NAME.dylib"
echo "Static library: $TARGET_DIR/lib$CRATE_NAME.a"
echo "Framework: $FRAMEWORK_DIR"
echo "Header: $TARGET_DIR/trustformers.h"

# Optional: Run tests on both architectures
if [ "$1" == "--test" ]; then
    echo "Running tests..."
    for target in "${TARGETS[@]}"; do
        echo "Testing $target..."
        cargo test --release --target "$target"
    done
fi