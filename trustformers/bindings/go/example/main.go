package main

import (
	"fmt"
	"log"
	"os"

	trustformers "github.com/trustformers/trustformers-go"
)

func main() {
	// Example 1: Text Generation
	fmt.Println("=== Text Generation Example ===")
	if err := textGenerationExample(); err != nil {
		log.Printf("Text generation error: %v", err)
	}

	// Example 2: Text Classification
	fmt.Println("\n=== Text Classification Example ===")
	if err := textClassificationExample(); err != nil {
		log.Printf("Text classification error: %v", err)
	}

	// Example 3: Question Answering
	fmt.Println("\n=== Question Answering Example ===")
	if err := questionAnsweringExample(); err != nil {
		log.Printf("Question answering error: %v", err)
	}

	// Example 4: Tokenization
	fmt.Println("\n=== Tokenization Example ===")
	if err := tokenizationExample(); err != nil {
		log.Printf("Tokenization error: %v", err)
	}

	// Example 5: Device Information
	fmt.Println("\n=== Device Information ===")
	deviceInfoExample()

	// Example 6: Convenience Functions
	fmt.Println("\n=== Convenience Functions Example ===")
	convenienceFunctionsExample()
}

func textGenerationExample() error {
	// Load a model for text generation
	modelPath := getModelPath("gpt2")
	model, err := trustformers.NewModel(modelPath)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}
	defer model.Close()

	// Generate text
	input := "The future of artificial intelligence is"
	result, err := model.Generate(input)
	if err != nil {
		return fmt.Errorf("generation failed: %w", err)
	}

	fmt.Printf("Input: %s\n", input)
	fmt.Printf("Generated: %s\n", result)
	return nil
}

func textClassificationExample() error {
	// Create a classification pipeline
	pipeline, err := trustformers.NewPipeline("text-classification", "sentiment-analysis")
	if err != nil {
		return fmt.Errorf("failed to create pipeline: %w", err)
	}
	defer pipeline.Close()

	// Classify text sentiment
	texts := []string{
		"I love this movie!",
		"This is terrible.",
		"The weather is nice today.",
	}

	for _, text := range texts {
		result, err := pipeline.Predict(text)
		if err != nil {
			log.Printf("Classification failed for '%s': %v", text, err)
			continue
		}
		fmt.Printf("Text: %s -> Classification: %s\n", text, result)
	}

	return nil
}

func questionAnsweringExample() error {
	// Create a question answering pipeline
	pipeline, err := trustformers.NewPipeline("question-answering", "distilbert-base-cased-distilled-squad")
	if err != nil {
		return fmt.Errorf("failed to create QA pipeline: %w", err)
	}
	defer pipeline.Close()

	// Answer questions
	context := "The TrustformeRS library is a high-performance machine learning framework written in Rust. It provides efficient implementations of transformer models and supports various hardware accelerations including CUDA and ROCm."
	questions := []string{
		"What is TrustformeRS?",
		"What language is it written in?",
		"What hardware accelerations does it support?",
	}

	for _, question := range questions {
		input := fmt.Sprintf("context: %s question: %s", context, question)
		answer, err := pipeline.Predict(input)
		if err != nil {
			log.Printf("QA failed for '%s': %v", question, err)
			continue
		}
		fmt.Printf("Q: %s\nA: %s\n\n", question, answer)
	}

	return nil
}

func tokenizationExample() error {
	// Load a tokenizer
	tokenizerPath := getTokenizerPath("gpt2")
	tokenizer, err := trustformers.NewTokenizer(tokenizerPath)
	if err != nil {
		return fmt.Errorf("failed to load tokenizer: %w", err)
	}
	defer tokenizer.Close()

	// Tokenize text
	text := "Hello, world! This is a test sentence."
	tokens, err := tokenizer.Encode(text)
	if err != nil {
		return fmt.Errorf("encoding failed: %w", err)
	}

	fmt.Printf("Original text: %s\n", text)
	fmt.Printf("Tokens: %s\n", tokens)

	// Decode tokens back to text
	decoded, err := tokenizer.Decode(tokens)
	if err != nil {
		return fmt.Errorf("decoding failed: %w", err)
	}

	fmt.Printf("Decoded text: %s\n", decoded)
	return nil
}

func deviceInfoExample() {
	// Check CUDA availability
	if trustformers.IsCudaAvailable() {
		fmt.Printf("CUDA is available\n")
		deviceCount := trustformers.GetCudaDeviceCount()
		fmt.Printf("Number of CUDA devices: %d\n", deviceCount)

		// Get device information
		devices := trustformers.GetDeviceInfo()
		for _, device := range devices {
			fmt.Printf("Device %d: %s\n", device.ID, device.Name)
		}
	} else {
		fmt.Println("CUDA is not available")
	}
}

func convenienceFunctionsExample() {
	// Example using convenience functions
	fmt.Println("Using convenience functions...")

	// Text generation
	modelPath := getModelPath("gpt2")
	if result, err := trustformers.GenerateText(modelPath, "The sky is"); err == nil {
		fmt.Printf("Generated: %s\n", result)
	} else {
		log.Printf("Generation error: %v", err)
	}

	// Text classification
	if result, err := trustformers.ClassifyText("sentiment-analysis", "I'm so happy today!"); err == nil {
		fmt.Printf("Sentiment: %s\n", result)
	} else {
		log.Printf("Classification error: %v", err)
	}

	// Question answering
	context := "Go is a programming language developed by Google."
	question := "Who developed Go?"
	if result, err := trustformers.AnswerQuestion("qa-model", context, question); err == nil {
		fmt.Printf("Answer: %s\n", result)
	} else {
		log.Printf("QA error: %v", err)
	}

	// Tokenization
	tokenizerPath := getTokenizerPath("gpt2")
	if result, err := trustformers.TokenizeText(tokenizerPath, "Hello Go!"); err == nil {
		fmt.Printf("Tokens: %s\n", result)
	} else {
		log.Printf("Tokenization error: %v", err)
	}
}

// Helper functions to get model and tokenizer paths
func getModelPath(modelName string) string {
	if path := os.Getenv("TRUSTFORMERS_MODEL_PATH"); path != "" {
		return path + "/" + modelName
	}
	return "./models/" + modelName
}

func getTokenizerPath(tokenizerName string) string {
	if path := os.Getenv("TRUSTFORMERS_TOKENIZER_PATH"); path != "" {
		return path + "/" + tokenizerName
	}
	return "./tokenizers/" + tokenizerName
}

// Error handling example
func handleError(err error) {
	if trustformersErr, ok := err.(*trustformers.Error); ok {
		switch trustformersErr.Code {
		case trustformers.ModelLoadFailed:
			fmt.Printf("Model loading failed: %s\n", trustformersErr.Message)
		case trustformers.InferenceFailed:
			fmt.Printf("Inference failed: %s\n", trustformersErr.Message)
		case trustformers.OutOfMemory:
			fmt.Printf("Out of memory: %s\n", trustformersErr.Message)
		case trustformers.CudaError:
			fmt.Printf("CUDA error: %s\n", trustformersErr.Message)
		default:
			fmt.Printf("Error: %s\n", trustformersErr.Message)
		}
	} else {
		fmt.Printf("Unknown error: %v\n", err)
	}
}