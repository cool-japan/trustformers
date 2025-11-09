// +build cuda

package main

import (
	"fmt"
	"log"
	"time"

	"github.com/trustformers/trustformers-c/golang/trustformers"
)

func main() {
	// Check if CUDA is available
	if !trustformers.IsCudaAvailable() {
		log.Fatal("CUDA is not available on this system")
	}

	fmt.Printf("CUDA devices available: %d\n", trustformers.GetCudaDeviceCount())

	// Initialize TrustformeRS
	tf, err := trustformers.NewTrustformeRS()
	if err != nil {
		log.Fatalf("Failed to initialize TrustformeRS: %v", err)
	}
	defer tf.Cleanup()

	// Initialize CUDA backend
	cuda, err := trustformers.NewCudaBackend()
	if err != nil {
		log.Fatalf("Failed to initialize CUDA backend: %v", err)
	}

	fmt.Printf("CUDA backend initialized successfully\n")
	fmt.Printf("Number of CUDA devices: %d\n", cuda.GetDeviceCount())
	fmt.Printf("Current device: %d\n", cuda.GetCurrentDevice())

	// Get device information
	for deviceID := 0; deviceID < cuda.GetDeviceCount(); deviceID++ {
		deviceInfo, err := cuda.GetDeviceInfo(deviceID)
		if err != nil {
			log.Printf("Failed to get device %d info: %v", deviceID, err)
			continue
		}

		fmt.Printf("\nDevice %d Information:\n", deviceID)
		fmt.Printf("  Name: %s\n", deviceInfo.Name)
		fmt.Printf("  Compute Capability: %s\n", deviceInfo.ComputeCapability)
		fmt.Printf("  Total Memory: %.2f GB\n", float64(deviceInfo.TotalMemoryMB)/1024)
		fmt.Printf("  Free Memory: %.2f GB\n", float64(deviceInfo.FreeMemoryMB)/1024)
		fmt.Printf("  Multiprocessors: %d\n", deviceInfo.MultiprocessorCount)
		fmt.Printf("  Max Threads per Block: %d\n", deviceInfo.MaxThreadsPerBlock)
		fmt.Printf("  Warp Size: %d\n", deviceInfo.WarpSize)

		// Get memory info
		memInfo, err := cuda.GetMemoryInfo(deviceID)
		if err != nil {
			log.Printf("Failed to get memory info for device %d: %v", deviceID, err)
		} else {
			fmt.Printf("  Memory Usage:\n")
			fmt.Printf("    Total: %.2f GB\n", float64(memInfo.TotalMemoryBytes)/(1024*1024*1024))
			fmt.Printf("    Free: %.2f GB\n", float64(memInfo.FreeMemoryBytes)/(1024*1024*1024))
			fmt.Printf("    Used: %.2f GB\n", float64(memInfo.UsedMemoryBytes)/(1024*1024*1024))
		}
	}

	fmt.Println("\n" + "="*60)
	fmt.Println("CUDA Tensor Operations Demo")
	fmt.Println("="*60)

	// Matrix multiplication example
	demonstrateMatrixMultiplication(cuda)

	// Tensor allocation example
	demonstrateTensorOperations(cuda)

	fmt.Println("\nCUDA acceleration demo completed!")
}

func demonstrateMatrixMultiplication(cuda *trustformers.CudaBackend) {
	fmt.Println("\nMatrix Multiplication Demo:")
	
	// Matrix dimensions
	m, n, k := 1024, 1024, 1024
	
	fmt.Printf("Computing C = A * B where A is %dx%d and B is %dx%d\n", m, k, k, n)

	// Allocate matrices on GPU
	matrixA, err := cuda.AllocateTensor([]int{m, k}, trustformers.CudaFloat32, 0)
	if err != nil {
		log.Fatalf("Failed to allocate matrix A: %v", err)
	}
	defer matrixA.Free()

	matrixB, err := cuda.AllocateTensor([]int{k, n}, trustformers.CudaFloat32, 0)
	if err != nil {
		log.Fatalf("Failed to allocate matrix B: %v", err)
	}
	defer matrixB.Free()

	matrixC, err := cuda.AllocateTensor([]int{m, n}, trustformers.CudaFloat32, 0)
	if err != nil {
		log.Fatalf("Failed to allocate matrix C: %v", err)
	}
	defer matrixC.Free()

	fmt.Printf("Matrix A shape: %v, size: %.2f MB\n", matrixA.GetShape(), float64(matrixA.GetSizeBytes())/(1024*1024))
	fmt.Printf("Matrix B shape: %v, size: %.2f MB\n", matrixB.GetShape(), float64(matrixB.GetSizeBytes())/(1024*1024))
	fmt.Printf("Matrix C shape: %v, size: %.2f MB\n", matrixC.GetShape(), float64(matrixC.GetSizeBytes())/(1024*1024))

	// Initialize matrices with test data
	dataA := make([]float32, m*k)
	dataB := make([]float32, k*n)
	
	// Fill with simple test values
	for i := 0; i < m*k; i++ {
		dataA[i] = float32(i%100) * 0.01 // Values between 0 and 0.99
	}
	for i := 0; i < k*n; i++ {
		dataB[i] = float32(i%100) * 0.01 // Values between 0 and 0.99
	}

	// Copy data to GPU
	fmt.Println("Copying data to GPU...")
	start := time.Now()
	
	if err := matrixA.CopyFromHost(dataA); err != nil {
		log.Fatalf("Failed to copy matrix A to GPU: %v", err)
	}
	
	if err := matrixB.CopyFromHost(dataB); err != nil {
		log.Fatalf("Failed to copy matrix B to GPU: %v", err)
	}
	
	copyTime := time.Since(start)
	fmt.Printf("Data copy time: %v\n", copyTime)

	// Perform matrix multiplication
	fmt.Println("Performing matrix multiplication on GPU...")
	start = time.Now()
	
	if err := cuda.MatrixMultiply(matrixA, matrixB, matrixC, m, n, k); err != nil {
		log.Fatalf("Failed to perform matrix multiplication: %v", err)
	}
	
	// Synchronize to ensure completion
	if err := cuda.Synchronize(); err != nil {
		log.Fatalf("Failed to synchronize CUDA operations: %v", err)
	}
	
	computeTime := time.Since(start)
	fmt.Printf("Matrix multiplication time: %v\n", computeTime)

	// Copy result back to host
	fmt.Println("Copying result back to host...")
	start = time.Now()
	
	resultData := make([]float32, m*n)
	if err := matrixC.CopyToHost(resultData); err != nil {
		log.Fatalf("Failed to copy result from GPU: %v", err)
	}
	
	copyBackTime := time.Since(start)
	fmt.Printf("Result copy time: %v\n", copyBackTime)

	// Verify a few elements of the result
	fmt.Println("Sample result values:")
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			idx := i*n + j
			fmt.Printf("  C[%d,%d] = %.4f", i, j, resultData[idx])
		}
		fmt.Println()
	}

	// Calculate performance metrics
	totalOps := float64(m) * float64(n) * float64(k) * 2 // 2 ops per element (multiply + add)
	gflops := totalOps / computeTime.Seconds() / 1e9
	
	fmt.Printf("\nPerformance Metrics:\n")
	fmt.Printf("  Total Operations: %.2e\n", totalOps)
	fmt.Printf("  Compute Time: %v\n", computeTime)
	fmt.Printf("  Performance: %.2f GFLOPS\n", gflops)
	fmt.Printf("  Total Time (including data transfer): %v\n", copyTime + computeTime + copyBackTime)
}

func demonstrateTensorOperations(cuda *trustformers.CudaBackend) {
	fmt.Println("\nTensor Operations Demo:")

	// Create tensors with different data types
	tensorF32, err := cuda.AllocateTensor([]int{256, 256}, trustformers.CudaFloat32, 0)
	if err != nil {
		log.Fatalf("Failed to allocate float32 tensor: %v", err)
	}
	defer tensorF32.Free()

	tensorF16, err := cuda.AllocateTensor([]int{512, 512}, trustformers.CudaFloat16, 0)
	if err != nil {
		log.Fatalf("Failed to allocate float16 tensor: %v", err)
	}
	defer tensorF16.Free()

	tensorInt8, err := cuda.AllocateTensor([]int{1024, 1024}, trustformers.CudaInt8, 0)
	if err != nil {
		log.Fatalf("Failed to allocate int8 tensor: %v", err)
	}
	defer tensorInt8.Free()

	fmt.Printf("Float32 tensor: shape %v, size %.2f MB, device %d\n", 
		tensorF32.GetShape(), 
		float64(tensorF32.GetSizeBytes())/(1024*1024),
		tensorF32.GetDeviceID())

	fmt.Printf("Float16 tensor: shape %v, size %.2f MB, device %d\n", 
		tensorF16.GetShape(), 
		float64(tensorF16.GetSizeBytes())/(1024*1024),
		tensorF16.GetDeviceID())

	fmt.Printf("Int8 tensor: shape %v, size %.2f MB, device %d\n", 
		tensorInt8.GetShape(), 
		float64(tensorInt8.GetSizeBytes())/(1024*1024),
		tensorInt8.GetDeviceID())

	// Test data transfer for float32 tensor
	fmt.Println("\nTesting data transfer for float32 tensor...")
	
	hostData := make([]float32, 256*256)
	for i := range hostData {
		hostData[i] = float32(i % 1000) / 1000.0
	}

	start := time.Now()
	if err := tensorF32.CopyFromHost(hostData); err != nil {
		log.Fatalf("Failed to copy data to GPU: %v", err)
	}
	copyToGPUTime := time.Since(start)

	resultData := make([]float32, 256*256)
	start = time.Now()
	if err := tensorF32.CopyToHost(resultData); err != nil {
		log.Fatalf("Failed to copy data from GPU: %v", err)
	}
	copyFromGPUTime := time.Since(start)

	fmt.Printf("Copy to GPU time: %v\n", copyToGPUTime)
	fmt.Printf("Copy from GPU time: %v\n", copyFromGPUTime)

	// Verify data integrity
	matches := 0
	for i := range hostData {
		if hostData[i] == resultData[i] {
			matches++
		}
	}
	
	fmt.Printf("Data integrity check: %d/%d elements match (%.2f%%)\n", 
		matches, len(hostData), float64(matches)/float64(len(hostData))*100)

	if matches == len(hostData) {
		fmt.Println("✓ Data transfer successful - all elements match!")
	} else {
		fmt.Println("✗ Data transfer error - some elements don't match")
	}
}