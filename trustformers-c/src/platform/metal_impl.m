//
//  metal_impl.m
//  TrustformeRS Metal Implementation
//
//  This file contains the Objective-C/Metal implementation for GPU acceleration
//  on Apple Silicon devices using Metal Performance Shaders.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <math.h>

// C interface for Metal operations
#ifdef __cplusplus
extern "C" {
#endif

// Device enumeration and management
int metal_get_device_count(void) {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    int count = (int)[devices count];
    return count;
}

void *metal_create_device(int device_id) {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    if (device_id < 0 || device_id >= (int)[devices count]) {
        return NULL;
    }
    
    id<MTLDevice> device = [devices objectAtIndex:(NSUInteger)device_id];
    return (__bridge_retained void *)device;
}

void metal_release_device(void *device) {
    if (device != NULL) {
        // Transfer ownership back to ARC
        id<MTLDevice> mtlDevice __attribute__((unused)) = (__bridge_transfer id<MTLDevice>)device;
        // ARC will handle cleanup automatically
    }
}

char *metal_get_device_name(void *device) {
    if (device == NULL) return NULL;
    
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    NSString *name = [mtlDevice name];
    
    const char *cString = [name UTF8String];
    size_t length = strlen(cString) + 1;
    char *result = (char *)malloc(length);
    strcpy(result, cString);
    
    return result;
}

uint64_t metal_get_max_buffer_length(void *device) {
    if (device == NULL) return 0;
    
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    return [mtlDevice maxBufferLength];
}

uint64_t metal_get_max_threadgroup_memory_length(void *device) {
    if (device == NULL) return 0;
    
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    return [mtlDevice maxThreadgroupMemoryLength];
}

uint64_t metal_get_max_threads_per_threadgroup(void *device) {
    if (device == NULL) return 0;
    
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    MTLSize maxThreads = [mtlDevice maxThreadsPerThreadgroup];
    return maxThreads.width * maxThreads.height * maxThreads.depth;
}

int metal_has_unified_memory(void *device) {
    if (device == NULL) return 0;
    
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    return [mtlDevice hasUnifiedMemory] ? 1 : 0;
}

int metal_supports_family_apple7(void *device) {
    if (device == NULL) return 0;
    
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    if (@available(macOS 11.0, iOS 14.0, *)) {
        return [mtlDevice supportsFamily:MTLGPUFamilyApple7] ? 1 : 0;
    }
    return 0;
}

int metal_supports_family_apple8(void *device) {
    if (device == NULL) return 0;
    
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    if (@available(macOS 12.0, iOS 15.0, *)) {
        return [mtlDevice supportsFamily:MTLGPUFamilyApple8] ? 1 : 0;
    }
    return 0;
}

int metal_supports_family_apple9(void *device) {
    if (device == NULL) return 0;
    
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    if (@available(macOS 13.0, iOS 16.0, *)) {
        return [mtlDevice supportsFamily:MTLGPUFamilyApple9] ? 1 : 0;
    }
    return 0;
}

uint64_t metal_get_memory_size(void *device) {
    if (device == NULL) return 0;
    
    // For Apple Silicon, we can estimate based on system memory
    // since it uses unified memory architecture
    uint64_t physicalMemory = [[NSProcessInfo processInfo] physicalMemory];
    
    // Assume 80% of system memory is available for Metal operations
    return (uint64_t)(physicalMemory * 0.8);
}

// Command queue management
void *metal_create_command_queue(void *device) {
    if (device == NULL) return NULL;
    
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLCommandQueue> commandQueue = [mtlDevice newCommandQueue];
    
    return (__bridge_retained void *)commandQueue;
}

void metal_release_command_queue(void *queue) {
    if (queue != NULL) {
        id<MTLCommandQueue> commandQueue __attribute__((unused)) = (__bridge_transfer id<MTLCommandQueue>)queue;
        // ARC will handle cleanup automatically
    }
}

// Buffer management
void *metal_create_buffer(void *device, uint64_t size) {
    if (device == NULL || size == 0) return NULL;
    
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLBuffer> buffer = [mtlDevice newBufferWithLength:size 
                                                 options:MTLResourceStorageModeShared];
    
    return (__bridge_retained void *)buffer;
}

void metal_release_buffer(void *buffer) {
    if (buffer != NULL) {
        id<MTLBuffer> mtlBuffer __attribute__((unused)) = (__bridge_transfer id<MTLBuffer>)buffer;
        // ARC will handle cleanup automatically
    }
}

void *metal_get_buffer_contents(void *buffer) {
    if (buffer == NULL) return NULL;
    
    id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
    return [mtlBuffer contents];
}

void metal_did_modify_range(void *buffer, uint64_t offset, uint64_t length) {
    if (buffer == NULL) return;
    
    id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
    [mtlBuffer didModifyRange:NSMakeRange(offset, length)];
}

// High-level operations
int metal_matrix_multiply(void *queue, void *a, void *b, void *c, 
                         uint64_t m, uint64_t n, uint64_t k, int data_type __attribute__((unused))) {
    if (queue == NULL || a == NULL || b == NULL || c == NULL) return -1;
    
    @autoreleasepool {
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        id<MTLBuffer> bufferA = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> bufferB = (__bridge id<MTLBuffer>)b;
        id<MTLBuffer> bufferC = (__bridge id<MTLBuffer>)c;
        
        // Create matrix descriptors
        MPSMatrixDescriptor *descriptorA = [MPSMatrixDescriptor 
            matrixDescriptorWithRows:m columns:k 
            rowBytes:k * sizeof(float) dataType:MPSDataTypeFloat32];
        
        MPSMatrixDescriptor *descriptorB = [MPSMatrixDescriptor 
            matrixDescriptorWithRows:k columns:n 
            rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];
        
        MPSMatrixDescriptor *descriptorC = [MPSMatrixDescriptor 
            matrixDescriptorWithRows:m columns:n 
            rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];
        
        // Create matrix objects
        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descriptorA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descriptorB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descriptorC];
        
        // Create matrix multiplication kernel
        MPSMatrixMultiplication *matMul = [[MPSMatrixMultiplication alloc] 
            initWithDevice:[commandQueue device] transposeLeft:NO transposeRight:NO 
            resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];
        
        // Create command buffer and encode operation
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [matMul encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
        
        // Commit and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Check for errors
        if (commandBuffer.error != nil) {
            NSLog(@"Metal matrix multiplication error: %@", commandBuffer.error.localizedDescription);
            return -1;
        }
        
        return 0;
    }
}

int metal_tensor_add(void *queue __attribute__((unused)), void *a, void *b, void *result, 
                    uint64_t elements, int data_type __attribute__((unused))) {
    if (queue == NULL || a == NULL || b == NULL || result == NULL) return -1;
    
    @autoreleasepool {
        id<MTLBuffer> bufferA = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> bufferB = (__bridge id<MTLBuffer>)b;
        id<MTLBuffer> bufferResult = (__bridge id<MTLBuffer>)result;
        
        // For element-wise addition, use simple CPU implementation with shared memory
        // In a production implementation, this would use Metal compute shaders
        float *input_a = (float *)[bufferA contents];
        float *input_b = (float *)[bufferB contents];
        float *output = (float *)[bufferResult contents];
        
        // Perform element-wise addition
        for (uint64_t i = 0; i < elements; i++) {
            output[i] = input_a[i] + input_b[i];
        }
        
        // Mark buffer as modified
        [bufferResult didModifyRange:NSMakeRange(0, elements * sizeof(float))];
        
        return 0;
    }
}

int metal_apply_activation(void *queue __attribute__((unused)), void *input, void *output, 
                          uint64_t elements, int activation_type, int data_type __attribute__((unused))) {
    if (queue == NULL || input == NULL || output == NULL) return -1;
    
    @autoreleasepool {
        id<MTLBuffer> bufferInput = (__bridge id<MTLBuffer>)input;
        id<MTLBuffer> bufferOutput = (__bridge id<MTLBuffer>)output;
        
        // Apply activation function based on type using modern MPS APIs
        switch (activation_type) {
            case 0: { // ReLU
                float *input_ptr = (float *)[bufferInput contents];
                float *output_ptr = (float *)[bufferOutput contents];
                
                // Simple CPU implementation for ReLU (would normally use GPU compute shader)
                for (uint64_t i = 0; i < elements; i++) {
                    output_ptr[i] = fmaxf(0.0f, input_ptr[i]);
                }
                break;
            }
            case 1: { // GELU (approximated with tanh)
                // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                float *input_ptr = (float *)[bufferInput contents];
                float *output_ptr = (float *)[bufferOutput contents];
                
                for (uint64_t i = 0; i < elements; i++) {
                    float x = input_ptr[i];
                    float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x);
                    output_ptr[i] = 0.5f * x * (1.0f + tanhf(inner));
                }
                break;
            }
            case 2: { // Tanh
                float *input_ptr = (float *)[bufferInput contents];
                float *output_ptr = (float *)[bufferOutput contents];
                
                for (uint64_t i = 0; i < elements; i++) {
                    output_ptr[i] = tanhf(input_ptr[i]);
                }
                break;
            }
            case 3: { // Sigmoid
                float *input_ptr = (float *)[bufferInput contents];
                float *output_ptr = (float *)[bufferOutput contents];
                
                for (uint64_t i = 0; i < elements; i++) {
                    output_ptr[i] = 1.0f / (1.0f + expf(-input_ptr[i]));
                }
                break;
            }
            default:
                return -1; // Unsupported activation type
        }
        
        // Mark buffer as modified
        [bufferOutput didModifyRange:NSMakeRange(0, elements * sizeof(float))];
        
        // For this simplified implementation, we don't need to commit a command buffer
        // since we're using CPU operations with shared memory
        
        return 0;
    }
}

// Utility functions
void metal_free_string(char *string) {
    if (string != NULL) {
        free(string);
    }
}

#ifdef __cplusplus
} // extern "C"
#endif