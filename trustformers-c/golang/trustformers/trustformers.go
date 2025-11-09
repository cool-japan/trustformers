// Package trustformers provides Go bindings for TrustformeRS-C library
package trustformers

/*
#cgo CFLAGS: -I../../
#cgo LDFLAGS: -L../../target/release -ltrustformers_c

#include <stdlib.h>
#include <string.h>

// Forward declarations for TrustformeRS C API
typedef enum {
    TRUSTFORMERS_SUCCESS = 0,
    TRUSTFORMERS_NULL_POINTER = 1,
    TRUSTFORMERS_INVALID_PARAMETER = 2,
    TRUSTFORMERS_RUNTIME_ERROR = 3,
    TRUSTFORMERS_SERIALIZATION_ERROR = 4,
    TRUSTFORMERS_MEMORY_ERROR = 5,
    TRUSTFORMERS_IO_ERROR = 6,
    TRUSTFORMERS_NOT_IMPLEMENTED = 7,
    TRUSTFORMERS_UNKNOWN_ERROR = 8,
} TrustformersError;

typedef struct {
    unsigned long long total_memory_bytes;
    unsigned long long peak_memory_bytes;
    unsigned long long allocated_models;
    unsigned long long allocated_tokenizers;
    unsigned long long allocated_pipelines;
    unsigned long long allocated_tensors;
} TrustformersMemoryUsage;

typedef struct {
    char* version;
    char* features;
    char* build_date;
    char* target;
} TrustformersBuildInfo;

typedef struct {
    TrustformersMemoryUsage basic;
    double fragmentation_ratio;
    unsigned long long avg_allocation_size;
    char* type_usage_json;
    int pressure_level;
    double allocation_rate;
    double deallocation_rate;
} TrustformersAdvancedMemoryUsage;

typedef struct {
    unsigned long long total_operations;
    double avg_operation_time_ms;
    double min_operation_time_ms;
    double max_operation_time_ms;
    double cache_hit_rate;
    double performance_score;
    int num_optimization_hints;
    char* optimization_hints_json;
} TrustformersPerformanceMetrics;

typedef struct {
    int enable_tracking;
    int enable_caching;
    int cache_size_mb;
    int num_threads;
    int enable_simd;
    int optimize_batch_size;
    int memory_optimization_level;
} TrustformersOptimizationConfig;

// Core API functions
extern TrustformersError trustformers_init();
extern TrustformersError trustformers_cleanup();
extern const char* trustformers_version();
extern TrustformersError trustformers_build_info(TrustformersBuildInfo* info);
extern void trustformers_free_string(char* ptr);
extern int trustformers_has_feature(const char* feature);
extern TrustformersError trustformers_set_log_level(int level);

// Memory management functions
extern TrustformersError trustformers_get_memory_usage(TrustformersMemoryUsage* usage);
extern TrustformersError trustformers_get_advanced_memory_usage(TrustformersAdvancedMemoryUsage* usage);
extern TrustformersError trustformers_memory_cleanup();
extern TrustformersError trustformers_set_memory_limits(unsigned long long max_memory_mb, unsigned long long warning_threshold_mb);
extern TrustformersError trustformers_check_memory_leaks(char** leak_report);
extern void trustformers_advanced_memory_usage_free(TrustformersAdvancedMemoryUsage* usage);

// Performance functions
extern TrustformersError trustformers_get_performance_metrics(TrustformersPerformanceMetrics* metrics);
extern TrustformersError trustformers_apply_optimizations(const TrustformersOptimizationConfig* config);
extern TrustformersError trustformers_start_profiling();
extern TrustformersError trustformers_stop_profiling(char** report);
extern void trustformers_performance_metrics_free(TrustformersPerformanceMetrics* metrics);

// Model functions (to be expanded)
extern void* trustformers_load_model_from_hub(const char* model_name, TrustformersError* error);
extern void* trustformers_load_model_from_path(const char* model_path, TrustformersError* error);
extern TrustformersError trustformers_model_free(void* model);

// Tokenizer functions (to be expanded)
extern void* trustformers_load_tokenizer_from_hub(const char* model_name, TrustformersError* error);
extern void* trustformers_load_tokenizer_from_path(const char* tokenizer_path, TrustformersError* error);
extern TrustformersError trustformers_tokenizer_free(void* tokenizer);

// Pipeline functions (to be expanded)
extern void* trustformers_create_text_generation_pipeline(void* model, void* tokenizer, TrustformersError* error);
extern void* trustformers_create_text_classification_pipeline(void* model, void* tokenizer, TrustformersError* error);
extern TrustformersError trustformers_pipeline_free(void* pipeline);
*/
import "C"
import (
	"encoding/json"
	"errors"
	"runtime"
	"unsafe"
)

// Error types
var (
	ErrNullPointer        = errors.New("null pointer")
	ErrInvalidParameter   = errors.New("invalid parameter")
	ErrRuntimeError       = errors.New("runtime error")
	ErrSerializationError = errors.New("serialization error")
	ErrMemoryError        = errors.New("memory error")
	ErrIOError            = errors.New("I/O error")
	ErrNotImplemented     = errors.New("not implemented")
	ErrUnknownError       = errors.New("unknown error")
)

// LogLevel represents logging levels
type LogLevel int

const (
	LogOff LogLevel = iota
	LogError
	LogWarn
	LogInfo
	LogDebug
	LogTrace
)

// MemoryUsage contains memory usage statistics
type MemoryUsage struct {
	TotalMemoryBytes     uint64 `json:"total_memory_bytes"`
	PeakMemoryBytes      uint64 `json:"peak_memory_bytes"`
	AllocatedModels      uint64 `json:"allocated_models"`
	AllocatedTokenizers  uint64 `json:"allocated_tokenizers"`
	AllocatedPipelines   uint64 `json:"allocated_pipelines"`
	AllocatedTensors     uint64 `json:"allocated_tensors"`
}

// AdvancedMemoryUsage contains advanced memory statistics
type AdvancedMemoryUsage struct {
	Basic               MemoryUsage     `json:"basic"`
	FragmentationRatio  float64         `json:"fragmentation_ratio"`
	AvgAllocationSize   uint64          `json:"avg_allocation_size"`
	TypeUsage           map[string]uint64 `json:"type_usage"`
	PressureLevel       int             `json:"pressure_level"`
	AllocationRate      float64         `json:"allocation_rate"`
	DeallocationRate    float64         `json:"deallocation_rate"`
}

// BuildInfo contains build information
type BuildInfo struct {
	Version   string `json:"version"`
	Features  string `json:"features"`
	BuildDate string `json:"build_date"`
	Target    string `json:"target"`
}

// PerformanceMetrics contains performance statistics
type PerformanceMetrics struct {
	TotalOperations        uint64                   `json:"total_operations"`
	AvgOperationTimeMs     float64                  `json:"avg_operation_time_ms"`
	MinOperationTimeMs     float64                  `json:"min_operation_time_ms"`
	MaxOperationTimeMs     float64                  `json:"max_operation_time_ms"`
	CacheHitRate           float64                  `json:"cache_hit_rate"`
	PerformanceScore       float64                  `json:"performance_score"`
	NumOptimizationHints   int                      `json:"num_optimization_hints"`
	OptimizationHints      []OptimizationHint       `json:"optimization_hints"`
}

// OptimizationHint represents a performance optimization suggestion
type OptimizationHint struct {
	Type                     string  `json:"type"`
	Description              string  `json:"description"`
	PotentialImprovement     float64 `json:"potential_improvement"`
	ImplementationDifficulty int     `json:"implementation_difficulty"`
}

// OptimizationConfig contains optimization settings
type OptimizationConfig struct {
	EnableTracking           bool `json:"enable_tracking"`
	EnableCaching            bool `json:"enable_caching"`
	CacheSizeMB              int  `json:"cache_size_mb"`
	NumThreads               int  `json:"num_threads"`
	EnableSIMD               bool `json:"enable_simd"`
	OptimizeBatchSize        bool `json:"optimize_batch_size"`
	MemoryOptimizationLevel  int  `json:"memory_optimization_level"`
}

// TrustformeRS represents the main library interface
type TrustformeRS struct {
	initialized bool
}

// NewTrustformeRS creates a new TrustformeRS instance
func NewTrustformeRS() (*TrustformeRS, error) {
	tf := &TrustformeRS{}
	if err := tf.Init(); err != nil {
		return nil, err
	}
	runtime.SetFinalizer(tf, (*TrustformeRS).finalize)
	return tf, nil
}

// Init initializes the TrustformeRS library
func (tf *TrustformeRS) Init() error {
	if tf.initialized {
		return nil
	}

	err := C.trustformers_init()
	if err := checkError(err); err != nil {
		return err
	}

	tf.initialized = true
	return nil
}

// Cleanup cleans up the TrustformeRS library
func (tf *TrustformeRS) Cleanup() error {
	if !tf.initialized {
		return nil
	}

	err := C.trustformers_cleanup()
	if err := checkError(err); err != nil {
		return err
	}

	tf.initialized = false
	runtime.SetFinalizer(tf, nil)
	return nil
}

// finalize is called by the finalizer
func (tf *TrustformeRS) finalize() {
	if tf.initialized {
		tf.Cleanup()
	}
}

// Version returns the library version
func (tf *TrustformeRS) Version() string {
	cVersion := C.trustformers_version()
	if cVersion == nil {
		return "unknown"
	}
	return C.GoString(cVersion)
}

// BuildInfo returns build information
func (tf *TrustformeRS) BuildInfo() (BuildInfo, error) {
	var cBuildInfo C.TrustformersBuildInfo
	err := C.trustformers_build_info(&cBuildInfo)
	if err := checkError(err); err != nil {
		return BuildInfo{}, err
	}

	buildInfo := BuildInfo{
		Version:   cStringToString(cBuildInfo.version),
		Features:  cStringToString(cBuildInfo.features),
		BuildDate: cStringToString(cBuildInfo.build_date),
		Target:    cStringToString(cBuildInfo.target),
	}

	// Free allocated strings
	freeCString(cBuildInfo.version)
	freeCString(cBuildInfo.features)
	freeCString(cBuildInfo.build_date)
	freeCString(cBuildInfo.target)

	return buildInfo, nil
}

// HasFeature checks if a feature is available
func (tf *TrustformeRS) HasFeature(feature string) bool {
	cFeature := C.CString(feature)
	defer C.free(unsafe.Pointer(cFeature))
	
	result := C.trustformers_has_feature(cFeature)
	return int(result) != 0
}

// SetLogLevel sets the logging level
func (tf *TrustformeRS) SetLogLevel(level LogLevel) error {
	err := C.trustformers_set_log_level(C.int(level))
	return checkError(err)
}

// GetMemoryUsage returns current memory usage statistics
func (tf *TrustformeRS) GetMemoryUsage() (MemoryUsage, error) {
	var cUsage C.TrustformersMemoryUsage
	err := C.trustformers_get_memory_usage(&cUsage)
	if err := checkError(err); err != nil {
		return MemoryUsage{}, err
	}

	return MemoryUsage{
		TotalMemoryBytes:    uint64(cUsage.total_memory_bytes),
		PeakMemoryBytes:     uint64(cUsage.peak_memory_bytes),
		AllocatedModels:     uint64(cUsage.allocated_models),
		AllocatedTokenizers: uint64(cUsage.allocated_tokenizers),
		AllocatedPipelines:  uint64(cUsage.allocated_pipelines),
		AllocatedTensors:    uint64(cUsage.allocated_tensors),
	}, nil
}

// GetAdvancedMemoryUsage returns advanced memory usage statistics
func (tf *TrustformeRS) GetAdvancedMemoryUsage() (AdvancedMemoryUsage, error) {
	var cUsage C.TrustformersAdvancedMemoryUsage
	err := C.trustformers_get_advanced_memory_usage(&cUsage)
	if err := checkError(err); err != nil {
		return AdvancedMemoryUsage{}, err
	}
	defer C.trustformers_advanced_memory_usage_free(&cUsage)

	// Parse type usage JSON
	typeUsage := make(map[string]uint64)
	if cUsage.type_usage_json != nil {
		typeUsageJSON := C.GoString(cUsage.type_usage_json)
		json.Unmarshal([]byte(typeUsageJSON), &typeUsage)
	}

	return AdvancedMemoryUsage{
		Basic: MemoryUsage{
			TotalMemoryBytes:    uint64(cUsage.basic.total_memory_bytes),
			PeakMemoryBytes:     uint64(cUsage.basic.peak_memory_bytes),
			AllocatedModels:     uint64(cUsage.basic.allocated_models),
			AllocatedTokenizers: uint64(cUsage.basic.allocated_tokenizers),
			AllocatedPipelines:  uint64(cUsage.basic.allocated_pipelines),
			AllocatedTensors:    uint64(cUsage.basic.allocated_tensors),
		},
		FragmentationRatio: float64(cUsage.fragmentation_ratio),
		AvgAllocationSize:  uint64(cUsage.avg_allocation_size),
		TypeUsage:          typeUsage,
		PressureLevel:      int(cUsage.pressure_level),
		AllocationRate:     float64(cUsage.allocation_rate),
		DeallocationRate:   float64(cUsage.deallocation_rate),
	}, nil
}

// MemoryCleanup forces memory cleanup
func (tf *TrustformeRS) MemoryCleanup() error {
	err := C.trustformers_memory_cleanup()
	return checkError(err)
}

// SetMemoryLimits sets memory limits and thresholds
func (tf *TrustformeRS) SetMemoryLimits(maxMemoryMB, warningThresholdMB uint64) error {
	err := C.trustformers_set_memory_limits(C.ulonglong(maxMemoryMB), C.ulonglong(warningThresholdMB))
	return checkError(err)
}

// CheckMemoryLeaks checks for memory leaks and returns a report
func (tf *TrustformeRS) CheckMemoryLeaks() (map[string]interface{}, error) {
	var cReport *C.char
	err := C.trustformers_check_memory_leaks(&cReport)
	if err := checkError(err); err != nil {
		return nil, err
	}
	defer freeCString(cReport)

	if cReport == nil {
		return make(map[string]interface{}), nil
	}

	reportJSON := C.GoString(cReport)
	var report map[string]interface{}
	if err := json.Unmarshal([]byte(reportJSON), &report); err != nil {
		return nil, err
	}

	return report, nil
}

// GetPerformanceMetrics returns performance metrics
func (tf *TrustformeRS) GetPerformanceMetrics() (PerformanceMetrics, error) {
	var cMetrics C.TrustformersPerformanceMetrics
	err := C.trustformers_get_performance_metrics(&cMetrics)
	if err := checkError(err); err != nil {
		return PerformanceMetrics{}, err
	}
	defer C.trustformers_performance_metrics_free(&cMetrics)

	// Parse optimization hints JSON
	var hints []OptimizationHint
	if cMetrics.optimization_hints_json != nil {
		hintsJSON := C.GoString(cMetrics.optimization_hints_json)
		json.Unmarshal([]byte(hintsJSON), &hints)
	}

	return PerformanceMetrics{
		TotalOperations:      uint64(cMetrics.total_operations),
		AvgOperationTimeMs:   float64(cMetrics.avg_operation_time_ms),
		MinOperationTimeMs:   float64(cMetrics.min_operation_time_ms),
		MaxOperationTimeMs:   float64(cMetrics.max_operation_time_ms),
		CacheHitRate:         float64(cMetrics.cache_hit_rate),
		PerformanceScore:     float64(cMetrics.performance_score),
		NumOptimizationHints: int(cMetrics.num_optimization_hints),
		OptimizationHints:    hints,
	}, nil
}

// ApplyOptimizations applies performance optimizations
func (tf *TrustformeRS) ApplyOptimizations(config OptimizationConfig) error {
	cConfig := C.TrustformersOptimizationConfig{
		enable_tracking:           boolToInt(config.EnableTracking),
		enable_caching:            boolToInt(config.EnableCaching),
		cache_size_mb:             C.int(config.CacheSizeMB),
		num_threads:               C.int(config.NumThreads),
		enable_simd:               boolToInt(config.EnableSIMD),
		optimize_batch_size:       boolToInt(config.OptimizeBatchSize),
		memory_optimization_level: C.int(config.MemoryOptimizationLevel),
	}

	err := C.trustformers_apply_optimizations(&cConfig)
	return checkError(err)
}

// StartProfiling starts a performance profiling session
func (tf *TrustformeRS) StartProfiling() error {
	err := C.trustformers_start_profiling()
	return checkError(err)
}

// StopProfiling stops profiling and returns a report
func (tf *TrustformeRS) StopProfiling() (map[string]interface{}, error) {
	var cReport *C.char
	err := C.trustformers_stop_profiling(&cReport)
	if err := checkError(err); err != nil {
		return nil, err
	}
	defer freeCString(cReport)

	if cReport == nil {
		return make(map[string]interface{}), nil
	}

	reportJSON := C.GoString(cReport)
	var report map[string]interface{}
	if err := json.Unmarshal([]byte(reportJSON), &report); err != nil {
		return nil, err
	}

	return report, nil
}

// Utility functions

func checkError(code C.TrustformersError) error {
	switch code {
	case C.TRUSTFORMERS_SUCCESS:
		return nil
	case C.TRUSTFORMERS_NULL_POINTER:
		return ErrNullPointer
	case C.TRUSTFORMERS_INVALID_PARAMETER:
		return ErrInvalidParameter
	case C.TRUSTFORMERS_RUNTIME_ERROR:
		return ErrRuntimeError
	case C.TRUSTFORMERS_SERIALIZATION_ERROR:
		return ErrSerializationError
	case C.TRUSTFORMERS_MEMORY_ERROR:
		return ErrMemoryError
	case C.TRUSTFORMERS_IO_ERROR:
		return ErrIOError
	case C.TRUSTFORMERS_NOT_IMPLEMENTED:
		return ErrNotImplemented
	default:
		return ErrUnknownError
	}
}

func cStringToString(cStr *C.char) string {
	if cStr == nil {
		return ""
	}
	return C.GoString(cStr)
}

func freeCString(cStr *C.char) {
	if cStr != nil {
		C.trustformers_free_string(cStr)
	}
}

func boolToInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// DefaultOptimizationConfig returns a default optimization configuration
func DefaultOptimizationConfig() OptimizationConfig {
	return OptimizationConfig{
		EnableTracking:          true,
		EnableCaching:           true,
		CacheSizeMB:             256,
		NumThreads:              0, // Auto-detect
		EnableSIMD:              true,
		OptimizeBatchSize:       true,
		MemoryOptimizationLevel: 2,
	}
}