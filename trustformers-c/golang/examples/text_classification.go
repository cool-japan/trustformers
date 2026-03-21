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

	fmt.Println("Loading classification model and tokenizer...")

	// Load model for classification (replace with actual classification model)
	model, err := tf.LoadModelFromHub("distilbert-base-uncased-finetuned-sst-2-english")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Free()

	// Load tokenizer
	tokenizer, err := tf.LoadTokenizerFromHub("distilbert-base-uncased-finetuned-sst-2-english")
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	defer tokenizer.Free()

	// Create text classification pipeline
	pipeline, err := tf.CreateTextClassificationPipeline(model, tokenizer)
	if err != nil {
		log.Fatalf("Failed to create classification pipeline: %v", err)
	}
	defer pipeline.Free()

	// Get pipeline info
	pipelineInfo, err := pipeline.GetInfo()
	if err != nil {
		log.Printf("Failed to get pipeline info: %v", err)
	} else {
		fmt.Printf("Pipeline Type: %s\n", pipelineInfo.Type)
		fmt.Printf("Capabilities: %v\n", pipelineInfo.Capabilities)
	}

	fmt.Println("\nClassifying individual texts...")

	// Test texts for sentiment analysis
	testTexts := []string{
		"I love this movie! It's absolutely fantastic.",
		"This is the worst film I've ever seen.",
		"The movie was okay, nothing special.",
		"Brilliant acting and amazing story!",
		"I fell asleep halfway through.",
	}

	// Classify each text individually
	for i, text := range testTexts {
		fmt.Printf("\nText %d: %s\n", i+1, text)
		
		results, err := pipeline.ClassifyText(text)
		if err != nil {
			log.Printf("Failed to classify text: %v", err)
			continue
		}

		fmt.Println("Classification results:")
		for _, result := range results {
			fmt.Printf("  Label: %s, Score: %.4f\n", result.Label, result.Score)
		}
	}

	fmt.Println("\n" + "="*50)
	fmt.Println("Batch classification...")

	// Classify all texts in a batch
	batchResults, err := pipeline.ClassifyTextBatch(testTexts)
	if err != nil {
		log.Fatalf("Failed to classify text batch: %v", err)
	}

	fmt.Println("\nBatch classification results:")
	for i, results := range batchResults {
		fmt.Printf("\nText %d: %s\n", i+1, testTexts[i])
		for _, result := range results {
			fmt.Printf("  %s: %.4f\n", result.Label, result.Score)
		}
	}

	// Memory usage after processing
	fmt.Println("\n" + "="*50)
	fmt.Println("Memory usage after classification:")
	
	memUsage, err := tf.GetMemoryUsage()
	if err != nil {
		log.Printf("Failed to get memory usage: %v", err)
	} else {
		fmt.Printf("Total memory: %d bytes\n", memUsage.TotalMemoryBytes)
		fmt.Printf("Peak memory: %d bytes\n", memUsage.PeakMemoryBytes)
	}

	// Advanced memory statistics
	advancedMemUsage, err := tf.GetAdvancedMemoryUsage()
	if err != nil {
		log.Printf("Failed to get advanced memory usage: %v", err)
	} else {
		fmt.Printf("Memory pressure level: %d\n", advancedMemUsage.PressureLevel)
		fmt.Printf("Fragmentation ratio: %.2f\n", advancedMemUsage.FragmentationRatio)
		fmt.Printf("Average allocation size: %d bytes\n", advancedMemUsage.AvgAllocationSize)
		
		if len(advancedMemUsage.TypeUsage) > 0 {
			fmt.Println("Memory usage by type:")
			for typeName, usage := range advancedMemUsage.TypeUsage {
				fmt.Printf("  %s: %d bytes\n", typeName, usage)
			}
		}
	}

	// Performance metrics
	metrics, err := tf.GetPerformanceMetrics()
	if err != nil {
		log.Printf("Failed to get performance metrics: %v", err)
	} else {
		fmt.Printf("\nPerformance Summary:\n")
		fmt.Printf("Total operations: %d\n", metrics.TotalOperations)
		fmt.Printf("Average operation time: %.2f ms\n", metrics.AvgOperationTimeMs)
		fmt.Printf("Performance score: %.2f/100\n", metrics.PerformanceScore)
		
		if len(metrics.OptimizationHints) > 0 {
			fmt.Println("\nOptimization hints:")
			for _, hint := range metrics.OptimizationHints {
				fmt.Printf("  - %s: %s (%.1f%% improvement)\n", 
					hint.Type, hint.Description, hint.PotentialImprovement)
			}
		}
	}

	fmt.Println("\nText classification example completed!")
}