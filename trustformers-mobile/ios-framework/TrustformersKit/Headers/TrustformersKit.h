//
//  TrustformersKit.h
//  TrustformersKit
//
//  iOS Framework for TrustformeRS Mobile Deployment
//  Copyright (c) 2024 TrustformeRS. All rights reserved.
//

#import <Foundation/Foundation.h>

//! Project version number for TrustformersKit
FOUNDATION_EXPORT double TrustformersKitVersionNumber;

//! Project version string for TrustformersKit
FOUNDATION_EXPORT const unsigned char TrustformersKitVersionString[];

// Core types and enums
typedef NS_ENUM(NSUInteger, TFKModelBackend) {
    TFKModelBackendCPU = 0,
    TFKModelBackendCoreML = 1,
    TFKModelBackendGPU = 2,
    TFKModelBackendCustom = 3
};

typedef NS_ENUM(NSUInteger, TFKMemoryOptimization) {
    TFKMemoryOptimizationMinimal = 0,
    TFKMemoryOptimizationBalanced = 1,
    TFKMemoryOptimizationMaximum = 2
};

typedef NS_ENUM(NSUInteger, TFKQuantizationScheme) {
    TFKQuantizationSchemeInt8 = 0,
    TFKQuantizationSchemeInt4 = 1,
    TFKQuantizationSchemeFP16 = 2,
    TFKQuantizationSchemeDynamic = 3
};

typedef NS_ENUM(NSUInteger, TFKThermalState) {
    TFKThermalStateNominal = 0,
    TFKThermalStateFair = 1,
    TFKThermalStateSerious = 2,
    TFKThermalStateCritical = 3
};

// Forward declarations
@class TFKInferenceEngine;
@class TFKModel;
@class TFKTensor;
@class TFKModelConfig;
@class TFKDeviceInfo;
@class TFKPerformanceStats;
@class TFKInferenceResult;

NS_ASSUME_NONNULL_BEGIN

#pragma mark - TFKModelConfig

/**
 * Configuration for TrustformersKit models
 */
@interface TFKModelConfig : NSObject

@property (nonatomic, assign) TFKModelBackend backend;
@property (nonatomic, assign) TFKMemoryOptimization memoryOptimization;
@property (nonatomic, assign) NSUInteger maxMemoryMB;
@property (nonatomic, assign) BOOL useFP16;
@property (nonatomic, assign) BOOL enableQuantization;
@property (nonatomic, assign) TFKQuantizationScheme quantizationScheme;
@property (nonatomic, assign) NSUInteger numThreads;
@property (nonatomic, assign) BOOL enableBatching;
@property (nonatomic, assign) NSUInteger maxBatchSize;

/**
 * Create default configuration
 */
+ (instancetype)defaultConfig;

/**
 * Create optimized configuration for current device
 */
+ (instancetype)optimizedConfig;

/**
 * Create ultra-low memory configuration
 */
+ (instancetype)ultraLowMemoryConfig;

/**
 * Validate configuration
 */
- (BOOL)validate:(NSError * _Nullable __autoreleasing *)error;

/**
 * Estimate memory usage for model with given size
 */
- (NSUInteger)estimateMemoryUsageForModelSizeMB:(NSUInteger)modelSizeMB;

@end

#pragma mark - TFKTensor

/**
 * Tensor representation for model inputs and outputs
 */
@interface TFKTensor : NSObject

@property (nonatomic, readonly) NSArray<NSNumber *> *shape;
@property (nonatomic, readonly) NSUInteger elementCount;
@property (nonatomic, readonly) NSUInteger byteSize;

/**
 * Create tensor from float array
 */
+ (nullable instancetype)tensorWithFloats:(float *)data shape:(NSArray<NSNumber *> *)shape;

/**
 * Create tensor from NSData
 */
+ (nullable instancetype)tensorWithData:(NSData *)data shape:(NSArray<NSNumber *> *)shape;

/**
 * Create tensor from MLMultiArray (Core ML interop)
 */
#if TARGET_OS_IOS
+ (nullable instancetype)tensorWithMLMultiArray:(MLMultiArray *)multiArray;
#endif

/**
 * Get float data
 */
- (nullable float *)floatData;

/**
 * Get data as NSData
 */
- (NSData *)dataRepresentation;

/**
 * Convert to MLMultiArray (Core ML interop)
 */
#if TARGET_OS_IOS
- (nullable MLMultiArray *)toMLMultiArray;
#endif

@end

#pragma mark - TFKInferenceResult

/**
 * Result of model inference
 */
@interface TFKInferenceResult : NSObject

@property (nonatomic, readonly) TFKTensor *output;
@property (nonatomic, readonly) NSTimeInterval inferenceTime;
@property (nonatomic, readonly) NSUInteger memoryUsed;
@property (nonatomic, readonly) BOOL success;
@property (nonatomic, readonly, nullable) NSError *error;

@end

#pragma mark - TFKModel

/**
 * Represents a TrustformersKit model
 */
@interface TFKModel : NSObject

@property (nonatomic, readonly) NSString *modelPath;
@property (nonatomic, readonly) TFKModelConfig *config;
@property (nonatomic, readonly) BOOL isLoaded;

/**
 * Load model from file path
 */
- (instancetype)initWithPath:(NSString *)path config:(TFKModelConfig *)config;

/**
 * Load model from bundle resource
 */
- (instancetype)initWithBundleResource:(NSString *)resourceName
                                  type:(NSString *)type
                                config:(TFKModelConfig *)config;

/**
 * Load model from Core ML model (if using Core ML backend)
 */
#if TARGET_OS_IOS
- (instancetype)initWithCoreMLModel:(MLModel *)coreMLModel
                             config:(TFKModelConfig *)config;
#endif

/**
 * Perform inference
 */
- (TFKInferenceResult *)inferenceWithInput:(TFKTensor *)input;

/**
 * Perform batch inference
 */
- (NSArray<TFKInferenceResult *> *)batchInferenceWithInputs:(NSArray<TFKTensor *> *)inputs;

/**
 * Preload model into memory
 */
- (BOOL)preload:(NSError * _Nullable __autoreleasing *)error;

/**
 * Unload model from memory
 */
- (void)unload;

@end

#pragma mark - TFKInferenceEngine

/**
 * Main inference engine for TrustformersKit
 */
@interface TFKInferenceEngine : NSObject

@property (nonatomic, readonly) TFKPerformanceStats *performanceStats;
@property (nonatomic, readonly) TFKDeviceInfo *deviceInfo;

/**
 * Shared instance
 */
+ (instancetype)sharedEngine;

/**
 * Initialize with configuration
 */
- (instancetype)initWithConfig:(TFKModelConfig *)config;

/**
 * Load model
 */
- (nullable TFKModel *)loadModelAtPath:(NSString *)path
                                config:(TFKModelConfig *)config
                                 error:(NSError * _Nullable __autoreleasing *)error;

/**
 * Perform inference with model
 */
- (TFKInferenceResult *)performInference:(TFKModel *)model
                                   input:(TFKTensor *)input;

/**
 * Set thermal throttling
 */
- (void)setThermalThrottling:(BOOL)enabled;

/**
 * Set memory pressure handler
 */
- (void)setMemoryPressureHandler:(void (^)(TFKMemoryOptimization))handler;

/**
 * Optimize for battery life
 */
- (void)optimizeForBatteryLife:(BOOL)enabled;

@end

#pragma mark - TFKDeviceInfo

/**
 * Device information and capabilities
 */
@interface TFKDeviceInfo : NSObject

@property (nonatomic, readonly) NSString *deviceModel;
@property (nonatomic, readonly) NSString *systemVersion;
@property (nonatomic, readonly) NSUInteger totalMemoryMB;
@property (nonatomic, readonly) NSUInteger availableMemoryMB;
@property (nonatomic, readonly) NSUInteger cpuCores;
@property (nonatomic, readonly) BOOL hasGPU;
@property (nonatomic, readonly) BOOL hasCoreML;
@property (nonatomic, readonly) BOOL hasNeuralEngine;
@property (nonatomic, readonly) TFKThermalState thermalState;
@property (nonatomic, readonly) float batteryLevel;
@property (nonatomic, readonly) BOOL isCharging;

/**
 * Get current device info
 */
+ (instancetype)currentDevice;

/**
 * Check if feature is supported
 */
- (BOOL)supportsFeature:(NSString *)feature;

/**
 * Get recommended configuration for device
 */
- (TFKModelConfig *)recommendedConfig;

@end

#pragma mark - TFKPerformanceStats

/**
 * Performance statistics for inference
 */
@interface TFKPerformanceStats : NSObject

@property (nonatomic, readonly) NSUInteger totalInferences;
@property (nonatomic, readonly) NSTimeInterval averageInferenceTime;
@property (nonatomic, readonly) NSTimeInterval minInferenceTime;
@property (nonatomic, readonly) NSTimeInterval maxInferenceTime;
@property (nonatomic, readonly) NSUInteger currentMemoryUsageMB;
@property (nonatomic, readonly) NSUInteger peakMemoryUsageMB;
@property (nonatomic, readonly) float averageCPUUsage;
@property (nonatomic, readonly) float averageGPUUsage;

/**
 * Reset statistics
 */
- (void)reset;

/**
 * Get performance summary
 */
- (NSString *)performanceSummary;

@end

#pragma mark - TFKLogger

/**
 * Logging configuration
 */
@interface TFKLogger : NSObject

typedef NS_ENUM(NSUInteger, TFKLogLevel) {
    TFKLogLevelVerbose = 0,
    TFKLogLevelDebug = 1,
    TFKLogLevelInfo = 2,
    TFKLogLevelWarning = 3,
    TFKLogLevelError = 4
};

/**
 * Set log level
 */
+ (void)setLogLevel:(TFKLogLevel)level;

/**
 * Set custom log handler
 */
+ (void)setLogHandler:(void (^)(TFKLogLevel level, NSString *message))handler;

@end

#pragma mark - Error Domain

/**
 * Error domain for TrustformersKit
 */
FOUNDATION_EXPORT NSErrorDomain const TFKErrorDomain;

typedef NS_ENUM(NSInteger, TFKErrorCode) {
    TFKErrorCodeUnknown = -1,
    TFKErrorCodeModelNotFound = 1000,
    TFKErrorCodeModelLoadFailed = 1001,
    TFKErrorCodeInvalidConfiguration = 1002,
    TFKErrorCodeInferenceFailed = 1003,
    TFKErrorCodeOutOfMemory = 1004,
    TFKErrorCodeUnsupportedBackend = 1005,
    TFKErrorCodeTensorShapeMismatch = 1006,
    TFKErrorCodeQuantizationFailed = 1007,
    TFKErrorCodeThermalThrottling = 1008
};

NS_ASSUME_NONNULL_END