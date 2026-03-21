package trustformers

/*
#include <stdlib.h>

// Model-related functions
extern void* trustformers_load_model_from_hub(const char* model_name, TrustformersError* error);
extern void* trustformers_load_model_from_path(const char* model_path, TrustformersError* error);
extern TrustformersError trustformers_model_free(void* model);
extern TrustformersError trustformers_model_get_info(void* model, char** info_json);
extern TrustformersError trustformers_model_set_quantization(void* model, int quantization_bits);
extern TrustformersError trustformers_model_validate(void* model, int* is_valid);
extern TrustformersError trustformers_model_get_metadata(void* model, char** metadata_json);
*/
import "C"
import (
	"encoding/json"
	"runtime"
	"unsafe"
)

// Model represents a loaded transformer model
type Model struct {
	handle unsafe.Pointer
	tf     *TrustformeRS
}

// ModelInfo contains model information
type ModelInfo struct {
	Name         string            `json:"name"`
	Type         string            `json:"type"`
	Architecture string            `json:"architecture"`
	Parameters   int64             `json:"parameters"`
	Precision    string            `json:"precision"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// LoadModelFromHub loads a model from Hugging Face Hub
func (tf *TrustformeRS) LoadModelFromHub(modelName string) (*Model, error) {
	if !tf.initialized {
		return nil, errors.New("TrustformeRS not initialized")
	}

	cModelName := C.CString(modelName)
	defer C.free(unsafe.Pointer(cModelName))

	var cError C.TrustformersError
	handle := C.trustformers_load_model_from_hub(cModelName, &cError)
	if err := checkError(cError); err != nil {
		return nil, err
	}

	if handle == nil {
		return nil, ErrRuntimeError
	}

	model := &Model{
		handle: handle,
		tf:     tf,
	}

	runtime.SetFinalizer(model, (*Model).finalize)
	return model, nil
}

// LoadModelFromPath loads a model from a local path
func (tf *TrustformeRS) LoadModelFromPath(modelPath string) (*Model, error) {
	if !tf.initialized {
		return nil, errors.New("TrustformeRS not initialized")
	}

	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	var cError C.TrustformersError
	handle := C.trustformers_load_model_from_path(cModelPath, &cError)
	if err := checkError(cError); err != nil {
		return nil, err
	}

	if handle == nil {
		return nil, ErrRuntimeError
	}

	model := &Model{
		handle: handle,
		tf:     tf,
	}

	runtime.SetFinalizer(model, (*Model).finalize)
	return model, nil
}

// Free releases the model resources
func (m *Model) Free() error {
	if m.handle == nil {
		return nil
	}

	err := C.trustformers_model_free(m.handle)
	if err := checkError(err); err != nil {
		return err
	}

	m.handle = nil
	runtime.SetFinalizer(m, nil)
	return nil
}

// finalize is called by the finalizer
func (m *Model) finalize() {
	if m.handle != nil {
		m.Free()
	}
}

// GetInfo returns model information
func (m *Model) GetInfo() (ModelInfo, error) {
	if m.handle == nil {
		return ModelInfo{}, errors.New("model not loaded")
	}

	var cInfoJSON *C.char
	err := C.trustformers_model_get_info(m.handle, &cInfoJSON)
	if err := checkError(err); err != nil {
		return ModelInfo{}, err
	}
	defer freeCString(cInfoJSON)

	if cInfoJSON == nil {
		return ModelInfo{}, errors.New("failed to get model info")
	}

	infoJSON := C.GoString(cInfoJSON)
	var info ModelInfo
	if err := json.Unmarshal([]byte(infoJSON), &info); err != nil {
		return ModelInfo{}, err
	}

	return info, nil
}

// SetQuantization sets the quantization level for the model
func (m *Model) SetQuantization(bits int) error {
	if m.handle == nil {
		return errors.New("model not loaded")
	}

	err := C.trustformers_model_set_quantization(m.handle, C.int(bits))
	return checkError(err)
}

// Validate checks if the model is valid
func (m *Model) Validate() (bool, error) {
	if m.handle == nil {
		return false, errors.New("model not loaded")
	}

	var cIsValid C.int
	err := C.trustformers_model_validate(m.handle, &cIsValid)
	if err := checkError(err); err != nil {
		return false, err
	}

	return int(cIsValid) != 0, nil
}

// GetMetadata returns model metadata
func (m *Model) GetMetadata() (map[string]interface{}, error) {
	if m.handle == nil {
		return nil, errors.New("model not loaded")
	}

	var cMetadataJSON *C.char
	err := C.trustformers_model_get_metadata(m.handle, &cMetadataJSON)
	if err := checkError(err); err != nil {
		return nil, err
	}
	defer freeCString(cMetadataJSON)

	if cMetadataJSON == nil {
		return make(map[string]interface{}), nil
	}

	metadataJSON := C.GoString(cMetadataJSON)
	var metadata map[string]interface{}
	if err := json.Unmarshal([]byte(metadataJSON), &metadata); err != nil {
		return nil, err
	}

	return metadata, nil
}

// IsLoaded checks if the model is loaded
func (m *Model) IsLoaded() bool {
	return m.handle != nil
}