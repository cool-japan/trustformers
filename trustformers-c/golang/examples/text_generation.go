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

	fmt.Println("Loading model and tokenizer...")

	// Load model from Hugging Face Hub (replace with actual model name)
	model, err := tf.LoadModelFromHub("gpt2")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Free()

	// Load tokenizer
	tokenizer, err := tf.LoadTokenizerFromHub("gpt2")
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	defer tokenizer.Free()

	// Get model and tokenizer info
	modelInfo, err := model.GetInfo()
	if err != nil {
		log.Printf("Failed to get model info: %v", err)
	} else {
		fmt.Printf("Model Info: %s (%s)\n", modelInfo.Name, modelInfo.Architecture)
		fmt.Printf("Parameters: %d\n", modelInfo.Parameters)
	}

	tokenizerInfo, err := tokenizer.GetInfo()
	if err != nil {
		log.Printf("Failed to get tokenizer info: %v", err)
	} else {
		fmt.Printf("Tokenizer: %s (vocab size: %d)\n", tokenizerInfo.Type, tokenizerInfo.VocabSize)
	}

	// Create text generation pipeline
	pipeline, err := tf.CreateTextGenerationPipeline(model, tokenizer)
	if err != nil {
		log.Fatalf("Failed to create pipeline: %v", err)
	}
	defer pipeline.Free()

	fmt.Println("\nGenerating text...")

	// Simple text generation
	prompt := "The future of artificial intelligence is"
	generatedText, err := pipeline.GenerateText(prompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}

	fmt.Printf("Prompt: %s\n", prompt)
	fmt.Printf("Generated: %s\n\n", generatedText)

	// Text generation with custom options
	options := trustformers.GenerationOptions{
		MaxLength:         100,
		Temperature:       0.8,
		TopK:              50,
		TopP:              0.9,
		RepetitionPenalty: 1.1,
		DoSample:          true,
	}

	fmt.Println("Generating text with custom options...")
	customGeneratedText, err := pipeline.GenerateTextWithOptions(prompt, options)
	if err != nil {
		log.Fatalf("Failed to generate text with options: %v", err)
	}

	fmt.Printf("Prompt: %s\n", prompt)
	fmt.Printf("Generated (custom): %s\n\n", customGeneratedText)

	// Test tokenizer directly
	fmt.Println("Testing tokenizer...")
	testText := "Hello, world! This is a test."
	tokens, err := tokenizer.Encode(testText)
	if err != nil {
		log.Printf("Failed to encode text: %v", err)
	} else {
		fmt.Printf("Original: %s\n", testText)
		fmt.Printf("Tokens: %v\n", tokens)

		decodedText, err := tokenizer.Decode(tokens)
		if err != nil {
			log.Printf("Failed to decode tokens: %v", err)
		} else {
			fmt.Printf("Decoded: %s\n", decodedText)
		}
	}

	// Batch encoding example
	texts := []string{
		"First sentence to encode.",
		"Second sentence for batch processing.",
		"Third and final sentence.",
	}

	fmt.Println("\nBatch encoding example...")
	tokenBatches, err := tokenizer.EncodeBatch(texts)
	if err != nil {
		log.Printf("Failed to encode batch: %v", err)
	} else {
		for i, tokens := range tokenBatches {
			fmt.Printf("Text %d: %s -> %v\n", i+1, texts[i], tokens)
		}

		// Decode the batch
		decodedTexts, err := tokenizer.DecodeBatch(tokenBatches)
		if err != nil {
			log.Printf("Failed to decode batch: %v", err)
		} else {
			fmt.Println("Decoded batch:")
			for i, text := range decodedTexts {
				fmt.Printf("  %d: %s\n", i+1, text)
			}
		}
	}

	// Get pipeline performance stats
	stats, err := pipeline.GetPerformanceStats()
	if err != nil {
		log.Printf("Failed to get pipeline stats: %v", err)
	} else {
		fmt.Printf("\nPipeline Performance Stats: %+v\n", stats)
	}

	fmt.Println("\nText generation example completed!")
}