#!/bin/bash

# TrustformeRS Android Build Script
# Builds the Rust library for Android deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building TrustformeRS for Android...${NC}"

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}Error: Cargo.toml not found. Please run this script from the trustformers-mobile directory.${NC}"
    exit 1
fi

# Check for required tools
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: cargo not found. Please install Rust.${NC}"
    exit 1
fi

# Check for Android NDK
if [ -z "$ANDROID_NDK_HOME" ] && [ -z "$NDK_HOME" ]; then
    echo -e "${RED}Error: Android NDK not found. Please set ANDROID_NDK_HOME or NDK_HOME environment variable.${NC}"
    exit 1
fi

NDK_HOME="${ANDROID_NDK_HOME:-$NDK_HOME}"
echo -e "${GREEN}Using Android NDK at: $NDK_HOME${NC}"

# Install cargo-ndk if not already installed
if ! command -v cargo-ndk &> /dev/null; then
    echo -e "${YELLOW}Installing cargo-ndk...${NC}"
    cargo install cargo-ndk
fi

# Add Android targets if not already added
echo -e "${YELLOW}Adding Android targets...${NC}"
rustup target add aarch64-linux-android     # 64-bit ARM
rustup target add armv7-linux-androideabi   # 32-bit ARM
rustup target add i686-linux-android        # 32-bit x86
rustup target add x86_64-linux-android      # 64-bit x86

# Set up environment variables for Android
export ANDROID_NDK_HOME="$NDK_HOME"
export CC_aarch64_linux_android="$NDK_HOME/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin/aarch64-linux-android30-clang"
export CC_armv7_linux_androideabi="$NDK_HOME/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin/armv7a-linux-androideabi30-clang"
export CC_i686_linux_android="$NDK_HOME/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin/i686-linux-android30-clang"
export CC_x86_64_linux_android="$NDK_HOME/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin/x86_64-linux-android30-clang"

export AR_aarch64_linux_android="$NDK_HOME/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin/llvm-ar"
export AR_armv7_linux_androideabi="$NDK_HOME/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin/llvm-ar"
export AR_i686_linux_android="$NDK_HOME/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin/llvm-ar"
export AR_x86_64_linux_android="$NDK_HOME/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin/llvm-ar"

# Create output directories
mkdir -p android-lib/libs/arm64-v8a
mkdir -p android-lib/libs/armeabi-v7a
mkdir -p android-lib/libs/x86
mkdir -p android-lib/libs/x86_64

# Build for each Android architecture
echo -e "${GREEN}Building for Android architectures...${NC}"

# Build for ARM64
echo "Building for Android ARM64..."
cargo ndk -t arm64-v8a -p 30 -- build --release
cp target/aarch64-linux-android/release/libtrustformers_mobile.so android-lib/libs/arm64-v8a/

# Build for ARMv7
echo "Building for Android ARMv7..."
cargo ndk -t armeabi-v7a -p 30 -- build --release
cp target/armv7-linux-androideabi/release/libtrustformers_mobile.so android-lib/libs/armeabi-v7a/

# Build for x86
echo "Building for Android x86..."
cargo ndk -t x86 -p 30 -- build --release
cp target/i686-linux-android/release/libtrustformers_mobile.so android-lib/libs/x86/

# Build for x86_64
echo "Building for Android x86_64..."
cargo ndk -t x86_64 -p 30 -- build --release
cp target/x86_64-linux-android/release/libtrustformers_mobile.so android-lib/libs/x86_64/

# Generate JNI headers (optional, for reference)
echo -e "${GREEN}Generating JNI headers...${NC}"
cd android-lib/src/main/java
javac -h ../jni com/trustformers/TrustformersEngine.java || true
cd ../../../..

# Create AAR package
echo -e "${GREEN}Building Android library (AAR)...${NC}"
cd android-lib

# Check if gradle wrapper exists
if [ -f "gradlew" ]; then
    ./gradlew clean assembleRelease
elif command -v gradle &> /dev/null; then
    gradle clean assembleRelease
else
    echo -e "${YELLOW}Warning: Gradle not found. Please build the AAR manually.${NC}"
fi

cd ..

# Library size report
echo -e "${GREEN}Build complete! Library sizes:${NC}"
ls -lh android-lib/libs/*/libtrustformers_mobile.so

echo -e "${GREEN}✅ Android library build complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. The native libraries are in: android-lib/libs/"
echo "2. If Gradle is available, the AAR is at: android-lib/build/outputs/aar/"
echo "3. Add the AAR to your Android project dependencies"
echo "4. Import com.trustformers.TrustformersEngine in your code"