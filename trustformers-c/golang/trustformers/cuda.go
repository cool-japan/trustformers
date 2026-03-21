package trustformers

// +build cuda

/*
#cgo CFLAGS: -I../../
#cgo LDFLAGS: -L../../target/release -ltrustformers_c

#include <stdlib.h>

// CUDA-related functions
extern TrustformersError trustformers_cuda_init();
extern int trustformers_cuda_get_device_count();
extern TrustformersError trustformers_cuda_get_device_info(int device_id, char** info_json);
extern TrustformersError trustformers_cuda_set_device(int device_id);
extern int trustformers_cuda_get_current_device();
extern TrustformersError trustformers_cuda_allocate_tensor(const unsigned long* shape, int rank, int dtype, int device_id, unsigned long* tensor_handle);
extern TrustformersError trustformers_cuda_free_tensor(unsigned long tensor_handle);
extern TrustformersError trustformers_cuda_matrix_multiply(unsigned long a_handle, unsigned long b_handle, unsigned long c_handle, unsigned long m, unsigned long n, unsigned long k);
extern TrustformersError trustformers_cuda_copy_host_to_device(const float* host_data, unsigned long tensor_handle, unsigned long size);
extern TrustformersError trustformers_cuda_copy_device_to_host(unsigned long tensor_handle, float* host_data, unsigned long size);
extern int trustformers_cuda_is_available();
extern TrustformersError trustformers_cuda_get_memory_info(int device_id, unsigned long long* free_memory, unsigned long long* total_memory);
extern TrustformersError trustformers_cuda_synchronize();
*/
import "C"
import (
	"encoding/json"
	"runtime"
	"unsafe"
)

// CudaTensorDataType represents CUDA tensor data types
type CudaTensorDataType int

const (
	CudaFloat32 CudaTensorDataType = 0
	CudaFloat16 CudaTensorDataType = 1
	CudaInt32   CudaTensorDataType = 2
	CudaInt8    CudaTensorDataType = 3
	CudaUInt8   CudaTensorDataType = 4
)

// CudaDeviceInfo contains CUDA device information
type CudaDeviceInfo struct {
	DeviceID                    int     `json:"device_id"`
	Name                        string  `json:"name"`
	ComputeCapability           string  `json:"compute_capability"`
	TotalMemoryMB               uint64  `json:"total_memory_mb"`
	FreeMemoryMB                uint64  `json:"free_memory_mb"`
	MultiprocessorCount         int     `json:"multiprocessor_count"`
	MaxThreadsPerBlock          int     `json:"max_threads_per_block"`
	WarpSize                    int     `json:"warp_size"`
	MaxGridSize                 [3]int  `json:"max_grid_size"`
	MaxBlockSize                [3]int  `json:"max_block_size"`
}

// CudaMemoryInfo contains CUDA memory information
type CudaMemoryInfo struct {
	FreeMemoryBytes  uint64 `json:"free_memory_bytes"`
	TotalMemoryBytes uint64 `json:"total_memory_bytes"`
	UsedMemoryBytes  uint64 `json:"used_memory_bytes"`
}

// CudaTensor represents a tensor allocated on CUDA device
type CudaTensor struct {
	handle    uint64
	shape     []int
	dtype     CudaTensorDataType
	deviceID  int
	sizeBytes uint64
}

// CudaBackend provides CUDA GPU acceleration functionality
type CudaBackend struct {
	initialized bool
	deviceCount int
	currentDevice int
}

// NewCudaBackend creates a new CUDA backend instance
func NewCudaBackend() (*CudaBackend, error) {
	backend := &CudaBackend{
		currentDevice: -1,
	}
	
	if err := backend.Init(); err != nil {
		return nil, err
	}
	
	return backend, nil
}

// Init initializes the CUDA backend
func (cb *CudaBackend) Init() error {
	if cb.initialized {
		return nil
	}

	err := C.trustformers_cuda_init()
	if err := checkError(err); err != nil {
		return err
	}

	cb.deviceCount = int(C.trustformers_cuda_get_device_count())
	if cb.deviceCount <= 0 {
		return errors.New("no CUDA devices found")
	}

	// Set first device as current
	if err := cb.SetDevice(0); err != nil {
		return err
	}

	cb.initialized = true
	return nil
}

// IsAvailable checks if CUDA is available on the system
func (cb *CudaBackend) IsAvailable() bool {
	result := C.trustformers_cuda_is_available()
	return int(result) != 0
}

// GetDeviceCount returns the number of available CUDA devices
func (cb *CudaBackend) GetDeviceCount() int {
	return cb.deviceCount
}

// GetDeviceInfo returns information about a specific CUDA device
func (cb *CudaBackend) GetDeviceInfo(deviceID int) (CudaDeviceInfo, error) {
	if deviceID < 0 || deviceID >= cb.deviceCount {
		return CudaDeviceInfo{}, errors.New("invalid device ID")
	}

	var cInfoJSON *C.char
	err := C.trustformers_cuda_get_device_info(C.int(deviceID), &cInfoJSON)
	if err := checkError(err); err != nil {
		return CudaDeviceInfo{}, err
	}
	defer freeCString(cInfoJSON)

	if cInfoJSON == nil {
		return CudaDeviceInfo{}, errors.New("failed to get device info")
	}

	infoJSON := C.GoString(cInfoJSON)
	var info CudaDeviceInfo
	if err := json.Unmarshal([]byte(infoJSON), &info); err != nil {
		return CudaDeviceInfo{}, err
	}

	return info, nil
}

// SetDevice sets the current CUDA device
func (cb *CudaBackend) SetDevice(deviceID int) error {
	if deviceID < 0 || deviceID >= cb.deviceCount {
		return errors.New("invalid device ID")
	}

	err := C.trustformers_cuda_set_device(C.int(deviceID))
	if err := checkError(err); err != nil {
		return err
	}

	cb.currentDevice = deviceID
	return nil
}

// GetCurrentDevice returns the current CUDA device ID
func (cb *CudaBackend) GetCurrentDevice() int {
	return cb.currentDevice
}

// GetMemoryInfo returns memory information for a specific device
func (cb *CudaBackend) GetMemoryInfo(deviceID int) (CudaMemoryInfo, error) {
	if deviceID < 0 || deviceID >= cb.deviceCount {
		return CudaMemoryInfo{}, errors.New("invalid device ID")
	}

	var freeMemory, totalMemory C.ulonglong
	err := C.trustformers_cuda_get_memory_info(C.int(deviceID), &freeMemory, &totalMemory)
	if err := checkError(err); err != nil {
		return CudaMemoryInfo{}, err
	}

	free := uint64(freeMemory)
	total := uint64(totalMemory)
	used := total - free

	return CudaMemoryInfo{
		FreeMemoryBytes:  free,
		TotalMemoryBytes: total,
		UsedMemoryBytes:  used,
	}, nil
}

// AllocateTensor allocates a tensor on CUDA device
func (cb *CudaBackend) AllocateTensor(shape []int, dtype CudaTensorDataType, deviceID int) (*CudaTensor, error) {
	if !cb.initialized {
		return nil, errors.New("CUDA backend not initialized")
	}

	if deviceID < 0 || deviceID >= cb.deviceCount {
		return nil, errors.New("invalid device ID")
	}

	if len(shape) == 0 {
		return nil, errors.New("tensor shape cannot be empty")
	}

	// Convert shape to C array
	cShape := make([]C.ulong, len(shape))
	totalElements := 1
	for i, dim := range shape {
		if dim <= 0 {
			return nil, errors.New("tensor dimensions must be positive")
		}
		cShape[i] = C.ulong(dim)
		totalElements *= dim
	}

	var tensorHandle C.ulong
	err := C.trustformers_cuda_allocate_tensor(
		(*C.ulong)(&cShape[0]),
		C.int(len(shape)),
		C.int(dtype),
		C.int(deviceID),
		&tensorHandle,
	)
	if err := checkError(err); err != nil {
		return nil, err
	}

	// Calculate size in bytes
	var elementSize int
	switch dtype {
	case CudaFloat32, CudaInt32:
		elementSize = 4
	case CudaFloat16:
		elementSize = 2
	case CudaInt8, CudaUInt8:
		elementSize = 1
	}

	tensor := &CudaTensor{
		handle:    uint64(tensorHandle),
		shape:     make([]int, len(shape)),
		dtype:     dtype,
		deviceID:  deviceID,
		sizeBytes: uint64(totalElements * elementSize),
	}
	copy(tensor.shape, shape)

	runtime.SetFinalizer(tensor, (*CudaTensor).finalize)
	return tensor, nil
}

// Free releases the CUDA tensor memory
func (ct *CudaTensor) Free() error {
	if ct.handle == 0 {
		return nil
	}

	err := C.trustformers_cuda_free_tensor(C.ulong(ct.handle))
	if err := checkError(err); err != nil {
		return err
	}

	ct.handle = 0
	runtime.SetFinalizer(ct, nil)
	return nil
}

// finalize is called by the finalizer
func (ct *CudaTensor) finalize() {
	if ct.handle != 0 {
		ct.Free()
	}
}

// GetShape returns the tensor shape
func (ct *CudaTensor) GetShape() []int {
	shape := make([]int, len(ct.shape))
	copy(shape, ct.shape)
	return shape
}

// GetDataType returns the tensor data type
func (ct *CudaTensor) GetDataType() CudaTensorDataType {
	return ct.dtype
}

// GetDeviceID returns the device ID where the tensor is allocated
func (ct *CudaTensor) GetDeviceID() int {
	return ct.deviceID
}

// GetSizeBytes returns the tensor size in bytes
func (ct *CudaTensor) GetSizeBytes() uint64 {
	return ct.sizeBytes
}

// CopyFromHost copies data from host memory to CUDA device
func (ct *CudaTensor) CopyFromHost(hostData []float32) error {
	if ct.handle == 0 {
		return errors.New("tensor not allocated")
	}

	if ct.dtype != CudaFloat32 {
		return errors.New("data type mismatch: expected float32")
	}

	expectedSize := ct.sizeBytes / 4 // 4 bytes per float32
	if uint64(len(hostData)) != expectedSize {
		return errors.New("data size mismatch")
	}

	err := C.trustformers_cuda_copy_host_to_device(
		(*C.float)(&hostData[0]),
		C.ulong(ct.handle),
		C.ulong(len(hostData)),
	)
	return checkError(err)
}

// CopyToHost copies data from CUDA device to host memory
func (ct *CudaTensor) CopyToHost(hostData []float32) error {
	if ct.handle == 0 {
		return errors.New("tensor not allocated")
	}

	if ct.dtype != CudaFloat32 {
		return errors.New("data type mismatch: expected float32")
	}

	expectedSize := ct.sizeBytes / 4 // 4 bytes per float32
	if uint64(len(hostData)) != expectedSize {
		return errors.New("data size mismatch")
	}

	err := C.trustformers_cuda_copy_device_to_host(
		C.ulong(ct.handle),
		(*C.float)(&hostData[0]),
		C.ulong(len(hostData)),
	)
	return checkError(err)
}

// MatrixMultiply performs matrix multiplication on CUDA: C = A * B
func (cb *CudaBackend) MatrixMultiply(a, b, c *CudaTensor, m, n, k int) error {
	if !cb.initialized {
		return errors.New("CUDA backend not initialized")
	}

	if a.handle == 0 || b.handle == 0 || c.handle == 0 {
		return errors.New("invalid tensor handles")
	}

	// Validate matrix dimensions
	if len(a.shape) != 2 || len(b.shape) != 2 || len(c.shape) != 2 {
		return errors.New("matrices must be 2D tensors")
	}

	if a.shape[0] != m || a.shape[1] != k {
		return errors.New("matrix A dimension mismatch")
	}

	if b.shape[0] != k || b.shape[1] != n {
		return errors.New("matrix B dimension mismatch")
	}

	if c.shape[0] != m || c.shape[1] != n {
		return errors.New("matrix C dimension mismatch")
	}

	err := C.trustformers_cuda_matrix_multiply(
		C.ulong(a.handle),
		C.ulong(b.handle),
		C.ulong(c.handle),
		C.ulong(m),
		C.ulong(n),
		C.ulong(k),
	)
	return checkError(err)
}

// Synchronize waits for all CUDA operations to complete
func (cb *CudaBackend) Synchronize() error {
	err := C.trustformers_cuda_synchronize()
	return checkError(err)
}

// IsInitialized returns whether the CUDA backend is initialized
func (cb *CudaBackend) IsInitialized() bool {
	return cb.initialized
}

// Utility functions for checking CUDA availability
func IsCudaAvailable() bool {
	result := C.trustformers_cuda_is_available()
	return int(result) != 0
}

func GetCudaDeviceCount() int {
	if !IsCudaAvailable() {
		return 0
	}
	return int(C.trustformers_cuda_get_device_count())
}