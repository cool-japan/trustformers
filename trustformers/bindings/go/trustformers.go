// Package trustformers provides Go bindings for the TrustformeRS machine learning library
package trustformers

/*
#cgo LDFLAGS: -ltrust_transformers_c
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

// Core types
typedef struct TrustformersModel TrustformersModel;
typedef struct TrustformersTokenizer TrustformersTokenizer;
typedef struct TrustformersPipeline TrustformersPipeline;

// Error handling
typedef enum {
    TRUSTFORMERS_SUCCESS = 0,
    TRUSTFORMERS_ERROR_INVALID_PARAMETER = 1,
    TRUSTFORMERS_ERROR_MODEL_LOAD_FAILED = 2,
    TRUSTFORMERS_ERROR_INFERENCE_FAILED = 3,
    TRUSTFORMERS_ERROR_OUT_OF_MEMORY = 4,
    TRUSTFORMERS_ERROR_CUDA_ERROR = 5,
    TRUSTFORMERS_ERROR_UNKNOWN = 255
} TrustformersErrorCode;

// Model functions
TrustformersModel* trustformers_load_model(const char* model_path, TrustformersErrorCode* error);
void trustformers_free_model(TrustformersModel* model);
char* trustformers_model_generate(TrustformersModel* model, const char* input, TrustformersErrorCode* error);

// Tokenizer functions
TrustformersTokenizer* trustformers_load_tokenizer(const char* tokenizer_path, TrustformersErrorCode* error);
void trustformers_free_tokenizer(TrustformersTokenizer* tokenizer);
char* trustformers_tokenizer_encode(TrustformersTokenizer* tokenizer, const char* text, TrustformersErrorCode* error);
char* trustformers_tokenizer_decode(TrustformersTokenizer* tokenizer, const char* tokens, TrustformersErrorCode* error);

// Pipeline functions
TrustformersPipeline* trustformers_create_pipeline(const char* task, const char* model_name, TrustformersErrorCode* error);
void trustformers_free_pipeline(TrustformersPipeline* pipeline);
char* trustformers_pipeline_predict(TrustformersPipeline* pipeline, const char* input, TrustformersErrorCode* error);

// Utility functions
char* trustformers_get_last_error();
void trustformers_free_string(char* str);
bool trustformers_is_cuda_available();
int trustformers_get_cuda_device_count();

*/
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

// Error represents a TrustformeRS error
type Error struct {
	Code    ErrorCode
	Message string
}

func (e Error) Error() string {
	return e.Message
}

// ErrorCode represents different types of errors
type ErrorCode int

const (
	Success             ErrorCode = C.TRUSTFORMERS_SUCCESS
	InvalidParameter    ErrorCode = C.TRUSTFORMERS_ERROR_INVALID_PARAMETER
	ModelLoadFailed     ErrorCode = C.TRUSTFORMERS_ERROR_MODEL_LOAD_FAILED
	InferenceFailed     ErrorCode = C.TRUSTFORMERS_ERROR_INFERENCE_FAILED
	OutOfMemory         ErrorCode = C.TRUSTFORMERS_ERROR_OUT_OF_MEMORY
	CudaError          ErrorCode = C.TRUSTFORMERS_ERROR_CUDA_ERROR
	Unknown            ErrorCode = C.TRUSTFORMERS_ERROR_UNKNOWN
)

// Model represents a loaded transformer model
type Model struct {
	ptr *C.TrustformersModel
}

// NewModel loads a model from the specified path
func NewModel(modelPath string) (*Model, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	
	var errorCode C.TrustformersErrorCode
	ptr := C.trustformers_load_model(cPath, &errorCode)
	
	if errorCode != C.TRUSTFORMERS_SUCCESS {
		return nil, &Error{
			Code:    ErrorCode(errorCode),
			Message: getLastError(),
		}
	}
	
	if ptr == nil {
		return nil, &Error{
			Code:    ModelLoadFailed,
			Message: "Failed to load model",
		}
	}
	
	model := &Model{ptr: ptr}
	runtime.SetFinalizer(model, (*Model).Close)
	return model, nil
}

// Generate generates text using the model
func (m *Model) Generate(input string) (string, error) {
	if m.ptr == nil {
		return "", &Error{Code: InvalidParameter, Message: "Model is closed"}
	}
	
	cInput := C.CString(input)
	defer C.free(unsafe.Pointer(cInput))
	
	var errorCode C.TrustformersErrorCode
	cResult := C.trustformers_model_generate(m.ptr, cInput, &errorCode)
	
	if errorCode != C.TRUSTFORMERS_SUCCESS {
		return "", &Error{
			Code:    ErrorCode(errorCode),
			Message: getLastError(),
		}
	}
	
	if cResult == nil {
		return "", &Error{Code: InferenceFailed, Message: "Generation failed"}
	}
	
	result := C.GoString(cResult)
	C.trustformers_free_string(cResult)
	return result, nil
}

// Close frees the model resources
func (m *Model) Close() {
	if m.ptr != nil {
		C.trustformers_free_model(m.ptr)
		m.ptr = nil
		runtime.SetFinalizer(m, nil)
	}
}

// Tokenizer represents a tokenizer for text processing
type Tokenizer struct {
	ptr *C.TrustformersTokenizer
}

// NewTokenizer loads a tokenizer from the specified path
func NewTokenizer(tokenizerPath string) (*Tokenizer, error) {
	cPath := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(cPath))
	
	var errorCode C.TrustformersErrorCode
	ptr := C.trustformers_load_tokenizer(cPath, &errorCode)
	
	if errorCode != C.TRUSTFORMERS_SUCCESS {
		return nil, &Error{
			Code:    ErrorCode(errorCode),
			Message: getLastError(),
		}
	}
	
	if ptr == nil {
		return nil, &Error{
			Code:    ModelLoadFailed,
			Message: "Failed to load tokenizer",
		}
	}
	
	tokenizer := &Tokenizer{ptr: ptr}
	runtime.SetFinalizer(tokenizer, (*Tokenizer).Close)
	return tokenizer, nil
}

// Encode converts text to tokens
func (t *Tokenizer) Encode(text string) (string, error) {
	if t.ptr == nil {
		return "", &Error{Code: InvalidParameter, Message: "Tokenizer is closed"}
	}
	
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	
	var errorCode C.TrustformersErrorCode
	cResult := C.trustformers_tokenizer_encode(t.ptr, cText, &errorCode)
	
	if errorCode != C.TRUSTFORMERS_SUCCESS {
		return "", &Error{
			Code:    ErrorCode(errorCode),
			Message: getLastError(),
		}
	}
	
	if cResult == nil {
		return "", &Error{Code: InferenceFailed, Message: "Encoding failed"}
	}
	
	result := C.GoString(cResult)
	C.trustformers_free_string(cResult)
	return result, nil
}

// Decode converts tokens to text
func (t *Tokenizer) Decode(tokens string) (string, error) {
	if t.ptr == nil {
		return "", &Error{Code: InvalidParameter, Message: "Tokenizer is closed"}
	}
	
	cTokens := C.CString(tokens)
	defer C.free(unsafe.Pointer(cTokens))
	
	var errorCode C.TrustformersErrorCode
	cResult := C.trustformers_tokenizer_decode(t.ptr, cTokens, &errorCode)
	
	if errorCode != C.TRUSTFORMERS_SUCCESS {
		return "", &Error{
			Code:    ErrorCode(errorCode),
			Message: getLastError(),
		}
	}
	
	if cResult == nil {
		return "", &Error{Code: InferenceFailed, Message: "Decoding failed"}
	}
	
	result := C.GoString(cResult)
	C.trustformers_free_string(cResult)
	return result, nil
}

// Close frees the tokenizer resources
func (t *Tokenizer) Close() {
	if t.ptr != nil {
		C.trustformers_free_tokenizer(t.ptr)
		t.ptr = nil
		runtime.SetFinalizer(t, nil)
	}
}

// Pipeline represents a high-level ML pipeline
type Pipeline struct {
	ptr *C.TrustformersPipeline
}

// NewPipeline creates a new pipeline for the specified task
func NewPipeline(task, modelName string) (*Pipeline, error) {
	cTask := C.CString(task)
	defer C.free(unsafe.Pointer(cTask))
	
	cModelName := C.CString(modelName)
	defer C.free(unsafe.Pointer(cModelName))
	
	var errorCode C.TrustformersErrorCode
	ptr := C.trustformers_create_pipeline(cTask, cModelName, &errorCode)
	
	if errorCode != C.TRUSTFORMERS_SUCCESS {
		return nil, &Error{
			Code:    ErrorCode(errorCode),
			Message: getLastError(),
		}
	}
	
	if ptr == nil {
		return nil, &Error{
			Code:    ModelLoadFailed,
			Message: "Failed to create pipeline",
		}
	}
	
	pipeline := &Pipeline{ptr: ptr}
	runtime.SetFinalizer(pipeline, (*Pipeline).Close)
	return pipeline, nil
}

// Predict makes a prediction using the pipeline
func (p *Pipeline) Predict(input string) (string, error) {
	if p.ptr == nil {
		return "", &Error{Code: InvalidParameter, Message: "Pipeline is closed"}
	}
	
	cInput := C.CString(input)
	defer C.free(unsafe.Pointer(cInput))
	
	var errorCode C.TrustformersErrorCode
	cResult := C.trustformers_pipeline_predict(p.ptr, cInput, &errorCode)
	
	if errorCode != C.TRUSTFORMERS_SUCCESS {
		return "", &Error{
			Code:    ErrorCode(errorCode),
			Message: getLastError(),
		}
	}
	
	if cResult == nil {
		return "", &Error{Code: InferenceFailed, Message: "Prediction failed"}
	}
	
	result := C.GoString(cResult)
	C.trustformers_free_string(cResult)
	return result, nil
}

// Close frees the pipeline resources
func (p *Pipeline) Close() {
	if p.ptr != nil {
		C.trustformers_free_pipeline(p.ptr)
		p.ptr = nil
		runtime.SetFinalizer(p, nil)
	}
}

// Utility functions

// IsCudaAvailable checks if CUDA is available
func IsCudaAvailable() bool {
	return bool(C.trustformers_is_cuda_available())
}

// GetCudaDeviceCount returns the number of available CUDA devices
func GetCudaDeviceCount() int {
	return int(C.trustformers_get_cuda_device_count())
}

// getLastError retrieves the last error message from the C library
func getLastError() string {
	cError := C.trustformers_get_last_error()
	if cError == nil {
		return "Unknown error"
	}
	
	errorMsg := C.GoString(cError)
	C.trustformers_free_string(cError)
	return errorMsg
}

// Device information
type DeviceInfo struct {
	ID          int
	Name        string
	TotalMemory uint64
	FreeMemory  uint64
}

// GetDeviceInfo returns information about available devices
func GetDeviceInfo() []DeviceInfo {
	var devices []DeviceInfo
	
	if IsCudaAvailable() {
		deviceCount := GetCudaDeviceCount()
		for i := 0; i < deviceCount; i++ {
			devices = append(devices, DeviceInfo{
				ID:          i,
				Name:        "CUDA Device " + string(rune(i)),
				TotalMemory: 0, // Would need additional C functions to get this
				FreeMemory:  0, // Would need additional C functions to get this
			})
		}
	}
	
	return devices
}

// GenerationConfig represents configuration for text generation
type GenerationConfig struct {
	MaxNewTokens   int
	Temperature    float32
	TopP          float32
	TopK          int
	DoSample      bool
	NumBeams      int
	NoRepeatNgramSize int
}

// DefaultGenerationConfig returns default generation configuration
func DefaultGenerationConfig() GenerationConfig {
	return GenerationConfig{
		MaxNewTokens:      50,
		Temperature:       1.0,
		TopP:             1.0,
		TopK:             50,
		DoSample:         true,
		NumBeams:         1,
		NoRepeatNgramSize: 0,
	}
}

// HighLevelAPI provides convenient wrapper functions

// GenerateText is a convenience function for text generation
func GenerateText(modelPath, input string) (string, error) {
	model, err := NewModel(modelPath)
	if err != nil {
		return "", err
	}
	defer model.Close()
	
	return model.Generate(input)
}

// ClassifyText is a convenience function for text classification
func ClassifyText(modelName, input string) (string, error) {
	pipeline, err := NewPipeline("text-classification", modelName)
	if err != nil {
		return "", err
	}
	defer pipeline.Close()
	
	return pipeline.Predict(input)
}

// AnswerQuestion is a convenience function for question answering
func AnswerQuestion(modelName, context, question string) (string, error) {
	pipeline, err := NewPipeline("question-answering", modelName)
	if err != nil {
		return "", err
	}
	defer pipeline.Close()
	
	input := "context: " + context + " question: " + question
	return pipeline.Predict(input)
}

// TokenizeText is a convenience function for tokenization
func TokenizeText(tokenizerPath, text string) (string, error) {
	tokenizer, err := NewTokenizer(tokenizerPath)
	if err != nil {
		return "", err
	}
	defer tokenizer.Close()
	
	return tokenizer.Encode(text)
}