# frozen_string_literal: true

require "mkmf"

# Configuration for building TrustFormeRS Ruby C extension

# Set extension name
extension_name = "trustformers_native"

# Platform-specific configuration
case RUBY_PLATFORM
when /darwin/
  # macOS configuration
  CONFIG["CC"] = "clang" unless CONFIG["CC"]
  
  # Add macOS-specific flags
  $CFLAGS << " -std=c99 -fPIC"
  $LDFLAGS << " -Wl,-undefined,dynamic_lookup"
  
  # Library naming for macOS
  lib_name = "libtrustformers_c.dylib"
  
when /linux/
  # Linux configuration
  CONFIG["CC"] = "gcc" unless CONFIG["CC"]
  
  # Add Linux-specific flags
  $CFLAGS << " -std=c99 -fPIC"
  $LDFLAGS << " -Wl,--no-undefined"
  
  # Library naming for Linux
  lib_name = "libtrustformers_c.so"
  
when /mswin|mingw|cygwin/
  # Windows configuration
  $CFLAGS << " -std=c99"
  
  # Library naming for Windows
  lib_name = "trustformers_c.dll"
  
else
  puts "Warning: Unsupported platform #{RUBY_PLATFORM}"
  lib_name = "libtrustformers_c.so"
end

# Find TrustFormeRS library
def find_trustformers_library
  # Search paths for the TrustFormeRS library
  search_paths = [
    # Relative to gem root
    "../../target/release",
    "../../target/debug",
    
    # System paths
    "/usr/local/lib",
    "/usr/lib",
    "/opt/homebrew/lib",
    
    # Environment variable
    ENV["TRUSTFORMERS_LIB_DIR"]
  ].compact

  search_paths.each do |path|
    next unless Dir.exist?(path)
    
    # Check for different library extensions
    %w[.so .dylib .dll].each do |ext|
      lib_file = File.join(path, "libtrustformers_c#{ext}")
      if File.exist?(lib_file)
        puts "Found TrustFormeRS library at: #{lib_file}"
        return path
      end
    end
    
    # Also check without lib prefix (Windows)
    %w[.dll .so .dylib].each do |ext|
      lib_file = File.join(path, "trustformers_c#{ext}")
      if File.exist?(lib_file)
        puts "Found TrustFormeRS library at: #{lib_file}"
        return path
      end
    end
  end
  
  nil
end

# Find header files
def find_trustformers_headers
  # Search paths for TrustFormeRS headers
  search_paths = [
    # Relative to gem root
    "../../trustformers-c/include",
    "../../include",
    
    # System paths
    "/usr/local/include",
    "/usr/include",
    "/opt/homebrew/include",
    
    # Environment variable
    ENV["TRUSTFORMERS_INCLUDE_DIR"]
  ].compact

  search_paths.each do |path|
    next unless Dir.exist?(path)
    
    header_file = File.join(path, "trustformers.h")
    if File.exist?(header_file)
      puts "Found TrustFormeRS headers at: #{path}"
      return path
    end
  end
  
  nil
end

# Locate TrustFormeRS library and headers
lib_dir = find_trustformers_library
include_dir = find_trustformers_headers

unless lib_dir
  puts <<~ERROR
    ERROR: Could not find TrustFormeRS library.
    
    Please ensure that TrustFormeRS is built and available in one of these locations:
    - ../../target/release/ (relative to gem)
    - /usr/local/lib
    - /usr/lib
    - Set TRUSTFORMERS_LIB_DIR environment variable
    
    To build TrustFormeRS:
    1. Navigate to the TrustFormeRS root directory
    2. Run: cargo build --release --package trustformers-c
  ERROR
  exit 1
end

unless include_dir
  puts <<~ERROR
    ERROR: Could not find TrustFormeRS header files.
    
    Please ensure that TrustFormeRS headers are available in one of these locations:
    - ../../trustformers-c/include/ (relative to gem)
    - /usr/local/include
    - /usr/include
    - Set TRUSTFORMERS_INCLUDE_DIR environment variable
  ERROR
  exit 1
end

# Add library and include directories
dir_config("trustformers", include_dir, lib_dir)

# Link against TrustFormeRS library
unless have_library("trustformers_c")
  puts "ERROR: Could not link against TrustFormeRS library"
  exit 1
end

# Check for required functions
required_functions = %w[
  trustformers_version
  trustformers_init
  trustformers_free
  model_load
  model_generate
  tokenizer_load
  tokenizer_encode
  tokenizer_decode
  pipeline_create
  pipeline_predict
]

missing_functions = []
required_functions.each do |func|
  unless have_func(func, "trustformers.h")
    missing_functions << func
  end
end

unless missing_functions.empty?
  puts "ERROR: Missing required functions: #{missing_functions.join(', ')}"
  puts "Please ensure you have a compatible version of TrustFormeRS"
  exit 1
end

# Compiler and linker flags
$CFLAGS << " -Wall -Wextra -Wno-unused-parameter"
$CFLAGS << " -O2" unless CONFIG["debugflags"]

# Debug build configuration
if ENV["DEBUG"] == "1"
  $CFLAGS << " -g -DDEBUG"
  puts "Building in debug mode"
end

# Thread safety
$CFLAGS << " -pthread"
$LDFLAGS << " -pthread"

# Platform-specific optimizations
case RUBY_PLATFORM
when /darwin/
  # Enable Apple-specific optimizations
  $CFLAGS << " -march=native" if ENV["NATIVE_OPT"] == "1"
  
when /linux/
  # Enable Linux-specific optimizations
  $CFLAGS << " -march=native" if ENV["NATIVE_OPT"] == "1"
  
when /mswin|mingw|cygwin/
  # Windows-specific configuration
  $LDFLAGS << " -static-libgcc"
end

# Check Ruby version compatibility
ruby_version = RUBY_VERSION.split('.').map(&:to_i)
if ruby_version[0] < 3 || (ruby_version[0] == 2 && ruby_version[1] < 7)
  puts "WARNING: Ruby #{RUBY_VERSION} may not be fully supported. Recommended: Ruby 3.0+"
end

# Create configuration header
config_h = <<~CONFIG_H
  #ifndef TRUSTFORMERS_RUBY_CONFIG_H
  #define TRUSTFORMERS_RUBY_CONFIG_H
  
  /* Platform detection */
  #ifdef __APPLE__
  #define TRUSTFORMERS_PLATFORM_DARWIN 1
  #elif defined(__linux__)
  #define TRUSTFORMERS_PLATFORM_LINUX 1
  #elif defined(_WIN32)
  #define TRUSTFORMERS_PLATFORM_WINDOWS 1
  #endif
  
  /* Ruby version compatibility */
  #define RUBY_VERSION_MAJOR #{ruby_version[0]}
  #define RUBY_VERSION_MINOR #{ruby_version[1]}
  #define RUBY_VERSION_PATCH #{ruby_version[2]}
  
  /* Build configuration */
  #ifdef DEBUG
  #define TRUSTFORMERS_DEBUG 1
  #endif
  
  #endif /* TRUSTFORMERS_RUBY_CONFIG_H */
CONFIG_H

File.write("config.h", config_h)

# Output build information
puts "TrustFormeRS Ruby Extension Build Configuration:"
puts "  Ruby Version: #{RUBY_VERSION}"
puts "  Platform: #{RUBY_PLATFORM}"
puts "  Library Directory: #{lib_dir}"
puts "  Include Directory: #{include_dir}"
puts "  Compiler: #{CONFIG['CC']}"
puts "  CFLAGS: #{$CFLAGS}"
puts "  LDFLAGS: #{$LDFLAGS}"

# Create Makefile
create_makefile(extension_name)

puts "\nBuild configuration complete. Run 'make' to build the extension."