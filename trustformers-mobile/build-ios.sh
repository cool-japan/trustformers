#!/bin/bash

# TrustformersKit iOS Build Script
# Builds the Rust library for iOS deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building TrustformersKit for iOS...${NC}"

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}Error: Cargo.toml not found. Please run this script from the trustformers-mobile directory.${NC}"
    exit 1
fi

# Install cargo-lipo if not already installed
if ! command -v cargo-lipo &> /dev/null; then
    echo -e "${YELLOW}Installing cargo-lipo...${NC}"
    cargo install cargo-lipo
fi

# Add iOS targets if not already added
echo -e "${YELLOW}Adding iOS targets...${NC}"
rustup target add aarch64-apple-ios      # 64-bit ARM (iPhone 5S+, iPad Air+)
rustup target add x86_64-apple-ios        # 64-bit x86 (simulator on Intel Macs)
rustup target add aarch64-apple-ios-sim   # ARM64 simulator (M1 Macs)

# Create output directories
mkdir -p target/universal/release
mkdir -p ios-framework/TrustformersKit/Frameworks

# Build for release
echo -e "${GREEN}Building release library...${NC}"

# Build for iOS device (ARM64)
echo "Building for iOS devices (arm64)..."
cargo build --release --target aarch64-apple-ios

# Build for iOS simulator (x86_64)
echo "Building for iOS simulator (x86_64)..."
cargo build --release --target x86_64-apple-ios

# Build for iOS simulator (ARM64 - M1 Macs)
echo "Building for iOS simulator (arm64-sim)..."
cargo build --release --target aarch64-apple-ios-sim

# Create universal library using lipo
echo -e "${GREEN}Creating universal library...${NC}"

# Create device library
lipo -create \
    target/aarch64-apple-ios/release/libtrustformers_mobile.a \
    -output target/universal/release/libtrustformers_mobile_device.a

# Create simulator library (both architectures)
lipo -create \
    target/x86_64-apple-ios/release/libtrustformers_mobile.a \
    target/aarch64-apple-ios-sim/release/libtrustformers_mobile.a \
    -output target/universal/release/libtrustformers_mobile_sim.a

# Create XCFramework
echo -e "${GREEN}Creating XCFramework...${NC}"

xcodebuild -create-xcframework \
    -library target/universal/release/libtrustformers_mobile_device.a \
    -headers ios-framework/TrustformersKit/Headers \
    -library target/universal/release/libtrustformers_mobile_sim.a \
    -headers ios-framework/TrustformersKit/Headers \
    -output ios-framework/TrustformersKit.xcframework

# Generate Swift bindings
echo -e "${GREEN}Generating Swift bindings...${NC}"

# Create bridging header
cat > ios-framework/TrustformersKit/Sources/TrustformersKit-Bridging-Header.h <<EOF
//
//  TrustformersKit-Bridging-Header.h
//  TrustformersKit
//
//  Bridging header for Swift
//

#ifndef TrustformersKit_Bridging_Header_h
#define TrustformersKit_Bridging_Header_h

#import "../Headers/TrustformersKit-C.h"

#endif /* TrustformersKit_Bridging_Header_h */
EOF

# Create module map for the static library
mkdir -p ios-framework/TrustformersKit.xcframework/modules
cat > ios-framework/TrustformersKit.xcframework/modules/module.modulemap <<EOF
module TrustformersKitCore {
    header "../Headers/TrustformersKit-C.h"
    export *
}
EOF

# Build size report
echo -e "${GREEN}Build complete! Library sizes:${NC}"
ls -lh target/universal/release/libtrustformers_mobile*.a

# Verify architectures
echo -e "${GREEN}Architectures in universal libraries:${NC}"
echo "Device library:"
lipo -info target/universal/release/libtrustformers_mobile_device.a
echo "Simulator library:"
lipo -info target/universal/release/libtrustformers_mobile_sim.a

echo -e "${GREEN}✅ iOS framework build complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. The XCFramework is located at: ios-framework/TrustformersKit.xcframework"
echo "2. Drag this framework into your Xcode project"
echo "3. Make sure to embed & sign the framework in your app target"
echo "4. Import TrustformersKit in your Swift files"