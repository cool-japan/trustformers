cmake_minimum_required(VERSION 3.18.1)
project("trustformers_mobile")

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find the prebuilt Rust library
set(RUST_LIB_DIR ${CMAKE_SOURCE_DIR}/../../../../target)

# Add the Rust static library for each ABI
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    set(RUST_TARGET "aarch64-linux-android")
elseif(${ANDROID_ABI} STREQUAL "armeabi-v7a")
    set(RUST_TARGET "armv7-linux-androideabi")
elseif(${ANDROID_ABI} STREQUAL "x86")
    set(RUST_TARGET "i686-linux-android")
elseif(${ANDROID_ABI} STREQUAL "x86_64")
    set(RUST_TARGET "x86_64-linux-android")
else()
    message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
endif()

# Path to the Rust library
set(RUST_LIB_PATH ${RUST_LIB_DIR}/${RUST_TARGET}/release/libtrustformers_mobile.a)

# Check if Rust library exists
if(NOT EXISTS ${RUST_LIB_PATH})
    message(FATAL_ERROR "Rust library not found at: ${RUST_LIB_PATH}")
endif()

# Import the Rust library
add_library(trustformers_rust STATIC IMPORTED)
set_target_properties(trustformers_rust PROPERTIES
    IMPORTED_LOCATION ${RUST_LIB_PATH}
)

# Create the JNI wrapper library
add_library(trustformers_mobile SHARED
    trustformers_jni.cpp
)

# Find required libraries
find_library(log-lib log)
find_library(android-lib android)

# Link libraries
target_link_libraries(trustformers_mobile
    trustformers_rust
    ${log-lib}
    ${android-lib}
)

# Include directories
target_include_directories(trustformers_mobile PRIVATE
    ${CMAKE_SOURCE_DIR}
)