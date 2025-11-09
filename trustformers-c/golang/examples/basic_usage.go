package main

import (
	"fmt"
	"log"

	"github.com/trustformers/trustformers-c/golang/trustformers"
)

func main() {
	// Initialize TrustformeRS
	tf, err := trustformers.NewTrustformeRS()
	if err != nil {
		log.Fatalf("Failed to initialize TrustformeRS: %v", err)
	}
	defer tf.Cleanup()

	// Display library information
	fmt.Printf("TrustformeRS Version: %s\n", tf.Version())

	buildInfo, err := tf.BuildInfo()
	if err != nil {
		log.Printf("Failed to get build info: %v", err)
	} else {
		fmt.Printf("Build Info: %+v\n", buildInfo)
	}

	// Check available features
	features := []string{"tokenizers", "models", "pipelines", "gpu", "onnx"}
	fmt.Println("\nAvailable Features:")
	for _, feature := range features {
		if tf.HasFeature(feature) {
			fmt.Printf("  ✓ %s\n", feature)
		} else {
			fmt.Printf("  ✗ %s\n", feature)
		}
	}

	// Display memory usage
	memUsage, err := tf.GetMemoryUsage()
	if err != nil {
		log.Printf("Failed to get memory usage: %v", err)
	} else {
		fmt.Printf("\nMemory Usage:\n")
		fmt.Printf("  Total: %d bytes\n", memUsage.TotalMemoryBytes)
		fmt.Printf("  Peak: %d bytes\n", memUsage.PeakMemoryBytes)
		fmt.Printf("  Models: %d\n", memUsage.AllocatedModels)
		fmt.Printf("  Tokenizers: %d\n", memUsage.AllocatedTokenizers)
		fmt.Printf("  Pipelines: %d\n", memUsage.AllocatedPipelines)
	}

	// Set optimization configuration
	config := trustformers.OptimizationConfig{
		EnableTracking:          true,
		EnableCaching:           true,
		CacheSizeMB:             256,
		NumThreads:              4,
		EnableSIMD:              true,
		OptimizeBatchSize:       true,
		MemoryOptimizationLevel: 2,
	}

	if err := tf.ApplyOptimizations(config); err != nil {
		log.Printf("Failed to apply optimizations: %v", err)
	} else {
		fmt.Println("\nOptimizations applied successfully")
	}

	// Get performance metrics
	metrics, err := tf.GetPerformanceMetrics()
	if err != nil {
		log.Printf("Failed to get performance metrics: %v", err)
	} else {
		fmt.Printf("\nPerformance Metrics:\n")
		fmt.Printf("  Total Operations: %d\n", metrics.TotalOperations)
		fmt.Printf("  Avg Operation Time: %.2f ms\n", metrics.AvgOperationTimeMs)
		fmt.Printf("  Cache Hit Rate: %.2f%%\n", metrics.CacheHitRate*100)
		fmt.Printf("  Performance Score: %.2f\n", metrics.PerformanceScore)
		
		if len(metrics.OptimizationHints) > 0 {
			fmt.Printf("  Optimization Hints: %d available\n", len(metrics.OptimizationHints))
		}
	}

	fmt.Println("\nBasic TrustformeRS setup completed successfully!")
}