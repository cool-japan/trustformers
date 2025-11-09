/*
 * TrustFormeRS Ruby C Extension
 * 
 * This file provides a minimal C extension for TrustFormeRS Ruby bindings.
 * The main functionality is implemented via FFI in the Ruby layer,
 * but this extension provides some utilities and fallback functions.
 */

#include <ruby.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_TRUSTFORMERS_H
#include <trustformers.h>
#endif

// Module and class definitions
static VALUE mTrustFormeRS;
static VALUE cNative;

/*
 * Get TrustFormeRS version
 * This is a simple wrapper that can be used if FFI is not available
 */
static VALUE
native_version(VALUE self)
{
#ifdef HAVE_TRUSTFORMERS_H
    const char* version = trustformers_version();
    return rb_str_new_cstr(version);
#else
    return rb_str_new_cstr("0.1.0-ruby");
#endif
}

/*
 * Check if GPU is available
 */
static VALUE
native_gpu_available(VALUE self)
{
#ifdef HAVE_TRUSTFORMERS_H
    return trustformers_is_gpu_available() ? Qtrue : Qfalse;
#else
    return Qfalse;
#endif
}

/*
 * Check if CUDA is available
 */
static VALUE
native_cuda_available(VALUE self)
{
#ifdef HAVE_TRUSTFORMERS_H
    return trustformers_is_cuda_available() ? Qtrue : Qfalse;
#else
    return Qfalse;
#endif
}

/*
 * Get platform information
 */
static VALUE
native_platform_info(VALUE self)
{
    VALUE hash = rb_hash_new();
    
    // Platform detection
#ifdef TRUSTFORMERS_PLATFORM_DARWIN
    rb_hash_aset(hash, rb_str_new_cstr("platform"), rb_str_new_cstr("darwin"));
#elif defined(TRUSTFORMERS_PLATFORM_LINUX)
    rb_hash_aset(hash, rb_str_new_cstr("platform"), rb_str_new_cstr("linux"));
#elif defined(TRUSTFORMERS_PLATFORM_WINDOWS)
    rb_hash_aset(hash, rb_str_new_cstr("platform"), rb_str_new_cstr("windows"));
#else
    rb_hash_aset(hash, rb_str_new_cstr("platform"), rb_str_new_cstr("unknown"));
#endif

    // Ruby version
    VALUE ruby_version = rb_hash_new();
    rb_hash_aset(ruby_version, rb_str_new_cstr("major"), INT2NUM(RUBY_VERSION_MAJOR));
    rb_hash_aset(ruby_version, rb_str_new_cstr("minor"), INT2NUM(RUBY_VERSION_MINOR));
    rb_hash_aset(ruby_version, rb_str_new_cstr("patch"), INT2NUM(RUBY_VERSION_PATCH));
    rb_hash_aset(hash, rb_str_new_cstr("ruby_version"), ruby_version);
    
    // Build configuration
#ifdef TRUSTFORMERS_DEBUG
    rb_hash_aset(hash, rb_str_new_cstr("debug_build"), Qtrue);
#else
    rb_hash_aset(hash, rb_str_new_cstr("debug_build"), Qfalse);
#endif

    // Extension availability
#ifdef HAVE_TRUSTFORMERS_H
    rb_hash_aset(hash, rb_str_new_cstr("native_library"), Qtrue);
#else
    rb_hash_aset(hash, rb_str_new_cstr("native_library"), Qfalse);
#endif

    return hash;
}

/*
 * Utility function to validate UTF-8 strings
 */
static VALUE
native_validate_utf8(VALUE self, VALUE str)
{
    Check_Type(str, T_STRING);
    
    const char* data = RSTRING_PTR(str);
    long len = RSTRING_LEN(str);
    
    // Simple UTF-8 validation
    for (long i = 0; i < len; i++) {
        unsigned char byte = (unsigned char)data[i];
        
        if (byte < 0x80) {
            // ASCII character
            continue;
        } else if ((byte >> 5) == 0x06) {
            // 110xxxxx - 2 byte sequence
            if (i + 1 >= len || (data[i + 1] & 0xC0) != 0x80) {
                return Qfalse;
            }
            i++;
        } else if ((byte >> 4) == 0x0E) {
            // 1110xxxx - 3 byte sequence
            if (i + 2 >= len || 
                (data[i + 1] & 0xC0) != 0x80 || 
                (data[i + 2] & 0xC0) != 0x80) {
                return Qfalse;
            }
            i += 2;
        } else if ((byte >> 3) == 0x1E) {
            // 11110xxx - 4 byte sequence
            if (i + 3 >= len || 
                (data[i + 1] & 0xC0) != 0x80 || 
                (data[i + 2] & 0xC0) != 0x80 || 
                (data[i + 3] & 0xC0) != 0x80) {
                return Qfalse;
            }
            i += 3;
        } else {
            // Invalid UTF-8 sequence
            return Qfalse;
        }
    }
    
    return Qtrue;
}

/*
 * Memory usage statistics (Ruby-specific)
 */
static VALUE
native_memory_stats(VALUE self)
{
    VALUE hash = rb_hash_new();
    
    // Get Ruby GC stats
    VALUE gc_stat = rb_gc_stat_new();
    rb_hash_aset(hash, rb_str_new_cstr("gc_stats"), gc_stat);
    
    // Process memory info (platform-specific)
#ifdef TRUSTFORMERS_PLATFORM_LINUX
    FILE* status = fopen("/proc/self/status", "r");
    if (status) {
        char line[256];
        while (fgets(line, sizeof(line), status)) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                int memory_kb;
                if (sscanf(line, "VmRSS: %d kB", &memory_kb) == 1) {
                    rb_hash_aset(hash, rb_str_new_cstr("rss_kb"), INT2NUM(memory_kb));
                }
                break;
            }
        }
        fclose(status);
    }
#endif

    return hash;
}

/*
 * Benchmark helper for performance testing
 */
static VALUE
native_benchmark_noop(VALUE self, VALUE iterations)
{
    Check_Type(iterations, T_FIXNUM);
    
    long iter = NUM2LONG(iterations);
    volatile long result = 0;
    
    for (long i = 0; i < iter; i++) {
        result += i;
    }
    
    return LONG2NUM(result);
}

/*
 * Initialize the native extension
 */
void
Init_trustformers_native(void)
{
    // Define TrustFormeRS module
    mTrustFormeRS = rb_define_module("TrustFormeRS");
    
    // Define Native class under TrustFormeRS module
    cNative = rb_define_class_under(mTrustFormeRS, "NativeExtension", rb_cObject);
    
    // Module methods
    rb_define_singleton_method(cNative, "version", native_version, 0);
    rb_define_singleton_method(cNative, "gpu_available?", native_gpu_available, 0);
    rb_define_singleton_method(cNative, "cuda_available?", native_cuda_available, 0);
    rb_define_singleton_method(cNative, "platform_info", native_platform_info, 0);
    rb_define_singleton_method(cNative, "validate_utf8", native_validate_utf8, 1);
    rb_define_singleton_method(cNative, "memory_stats", native_memory_stats, 0);
    rb_define_singleton_method(cNative, "benchmark_noop", native_benchmark_noop, 1);
    
    // Constants
    rb_define_const(cNative, "VERSION", rb_str_new_cstr("0.1.0"));
    
#ifdef HAVE_TRUSTFORMERS_H
    rb_define_const(cNative, "NATIVE_LIBRARY_AVAILABLE", Qtrue);
#else
    rb_define_const(cNative, "NATIVE_LIBRARY_AVAILABLE", Qfalse);
#endif

#ifdef TRUSTFORMERS_DEBUG
    rb_define_const(cNative, "DEBUG_BUILD", Qtrue);
#else
    rb_define_const(cNative, "DEBUG_BUILD", Qfalse);
#endif
}