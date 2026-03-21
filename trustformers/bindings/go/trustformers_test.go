package trustformers

import (
	"os"
	"strings"
	"testing"
)

// Mock C library for testing
// In a real implementation, these would be provided by the C library

func TestModel(t *testing.T) {
	// Skip if C library is not available
	if !isLibraryAvailable() {
		t.Skip("TrustformeRS C library not available")
	}

	t.Run("NewModel", func(t *testing.T) {
		// This would fail in test environment without actual model files
		// In real tests, you'd use test fixtures
		_, err := NewModel("nonexistent/model")
		if err == nil {
			t.Error("Expected error for nonexistent model")
		}

		// Check error type
		if trustformersErr, ok := err.(*Error); ok {
			if trustformersErr.Code != ModelLoadFailed {
				t.Errorf("Expected ModelLoadFailed, got %v", trustformersErr.Code)
			}
		} else {
			t.Error("Expected TrustformeRS error type")
		}
	})

	t.Run("ModelClose", func(t *testing.T) {
		// Test that Close() can be called multiple times safely
		model := &Model{ptr: nil} // Mock closed model
		model.Close() // Should not panic
		model.Close() // Should not panic
	})

	t.Run("ModelGenerate", func(t *testing.T) {
		model := &Model{ptr: nil} // Mock closed model
		_, err := model.Generate("test input")
		if err == nil {
			t.Error("Expected error for closed model")
		}

		if trustformersErr, ok := err.(*Error); ok {
			if trustformersErr.Code != InvalidParameter {
				t.Errorf("Expected InvalidParameter, got %v", trustformersErr.Code)
			}
		}
	})
}

func TestTokenizer(t *testing.T) {
	if !isLibraryAvailable() {
		t.Skip("TrustformeRS C library not available")
	}

	t.Run("NewTokenizer", func(t *testing.T) {
		_, err := NewTokenizer("nonexistent/tokenizer")
		if err == nil {
			t.Error("Expected error for nonexistent tokenizer")
		}
	})

	t.Run("TokenizerClose", func(t *testing.T) {
		tokenizer := &Tokenizer{ptr: nil}
		tokenizer.Close() // Should not panic
		tokenizer.Close() // Should not panic
	})

	t.Run("TokenizerEncode", func(t *testing.T) {
		tokenizer := &Tokenizer{ptr: nil}
		_, err := tokenizer.Encode("test text")
		if err == nil {
			t.Error("Expected error for closed tokenizer")
		}
	})

	t.Run("TokenizerDecode", func(t *testing.T) {
		tokenizer := &Tokenizer{ptr: nil}
		_, err := tokenizer.Decode("test tokens")
		if err == nil {
			t.Error("Expected error for closed tokenizer")
		}
	})
}

func TestPipeline(t *testing.T) {
	if !isLibraryAvailable() {
		t.Skip("TrustformeRS C library not available")
	}

	t.Run("NewPipeline", func(t *testing.T) {
		_, err := NewPipeline("text-classification", "nonexistent-model")
		if err == nil {
			t.Error("Expected error for nonexistent model")
		}
	})

	t.Run("PipelineClose", func(t *testing.T) {
		pipeline := &Pipeline{ptr: nil}
		pipeline.Close() // Should not panic
		pipeline.Close() // Should not panic
	})

	t.Run("PipelinePredict", func(t *testing.T) {
		pipeline := &Pipeline{ptr: nil}
		_, err := pipeline.Predict("test input")
		if err == nil {
			t.Error("Expected error for closed pipeline")
		}
	})
}

func TestErrorTypes(t *testing.T) {
	t.Run("ErrorCodes", func(t *testing.T) {
		// Test error code values
		if Success != 0 {
			t.Errorf("Expected Success to be 0, got %d", Success)
		}

		if InvalidParameter == Success {
			t.Error("InvalidParameter should not equal Success")
		}
	})

	t.Run("ErrorMessage", func(t *testing.T) {
		err := &Error{
			Code:    ModelLoadFailed,
			Message: "Test error message",
		}

		if err.Error() != "Test error message" {
			t.Errorf("Expected 'Test error message', got '%s'", err.Error())
		}
	})
}

func TestConvenienceFunctions(t *testing.T) {
	if !isLibraryAvailable() {
		t.Skip("TrustformeRS C library not available")
	}

	t.Run("GenerateText", func(t *testing.T) {
		_, err := GenerateText("nonexistent/model", "test input")
		if err == nil {
			t.Error("Expected error for nonexistent model")
		}
	})

	t.Run("ClassifyText", func(t *testing.T) {
		_, err := ClassifyText("nonexistent-model", "test input")
		if err == nil {
			t.Error("Expected error for nonexistent model")
		}
	})

	t.Run("AnswerQuestion", func(t *testing.T) {
		_, err := AnswerQuestion("nonexistent-model", "context", "question")
		if err == nil {
			t.Error("Expected error for nonexistent model")
		}
	})

	t.Run("TokenizeText", func(t *testing.T) {
		_, err := TokenizeText("nonexistent/tokenizer", "test text")
		if err == nil {
			t.Error("Expected error for nonexistent tokenizer")
		}
	})
}

func TestGenerationConfig(t *testing.T) {
	t.Run("DefaultConfig", func(t *testing.T) {
		config := DefaultGenerationConfig()

		if config.MaxNewTokens <= 0 {
			t.Error("MaxNewTokens should be positive")
		}

		if config.Temperature <= 0 {
			t.Error("Temperature should be positive")
		}

		if config.TopP <= 0 || config.TopP > 1 {
			t.Error("TopP should be between 0 and 1")
		}

		if config.TopK <= 0 {
			t.Error("TopK should be positive")
		}
	})
}

func TestDeviceInfo(t *testing.T) {
	t.Run("GetDeviceInfo", func(t *testing.T) {
		devices := GetDeviceInfo()
		// Should not panic, but may return empty slice if no CUDA
		_ = devices
	})

	t.Run("CudaFunctions", func(t *testing.T) {
		// These should not panic
		available := IsCudaAvailable()
		count := GetCudaDeviceCount()

		if available && count <= 0 {
			t.Error("If CUDA is available, device count should be positive")
		}

		if !available && count > 0 {
			t.Error("If CUDA is not available, device count should be zero")
		}
	})
}

// Benchmark tests
func BenchmarkModelGenerate(b *testing.B) {
	if !isLibraryAvailable() {
		b.Skip("TrustformeRS C library not available")
	}

	// This would need a real model for benchmarking
	b.Skip("Requires real model for benchmarking")
}

func BenchmarkTokenizerEncode(b *testing.B) {
	if !isLibraryAvailable() {
		b.Skip("TrustformeRS C library not available")
	}

	// This would need a real tokenizer for benchmarking
	b.Skip("Requires real tokenizer for benchmarking")
}

// Helper functions for testing

func isLibraryAvailable() bool {
	// Check if the C library is available
	// In a real implementation, this might try to load the library
	// or check for specific environment variables
	return os.Getenv("TRUSTFORMERS_TEST_WITH_LIBRARY") == "1"
}

// Integration tests (require real library and models)
func TestIntegration(t *testing.T) {
	if !isLibraryAvailable() || os.Getenv("TRUSTFORMERS_INTEGRATION_TEST") != "1" {
		t.Skip("Integration tests require TRUSTFORMERS_INTEGRATION_TEST=1 and real library")
	}

	modelPath := os.Getenv("TRUSTFORMERS_TEST_MODEL_PATH")
	if modelPath == "" {
		t.Skip("Integration tests require TRUSTFORMERS_TEST_MODEL_PATH")
	}

	t.Run("RealModelGeneration", func(t *testing.T) {
		model, err := NewModel(modelPath)
		if err != nil {
			t.Fatalf("Failed to load model: %v", err)
		}
		defer model.Close()

		result, err := model.Generate("The sky is")
		if err != nil {
			t.Fatalf("Generation failed: %v", err)
		}

		if len(result) == 0 {
			t.Error("Generated text should not be empty")
		}

		// Basic sanity check
		if !strings.Contains(result, "The sky is") {
			t.Error("Generated text should contain the input prompt")
		}
	})

	t.Run("RealTokenization", func(t *testing.T) {
		tokenizerPath := os.Getenv("TRUSTFORMERS_TEST_TOKENIZER_PATH")
		if tokenizerPath == "" {
			t.Skip("Integration test requires TRUSTFORMERS_TEST_TOKENIZER_PATH")
		}

		tokenizer, err := NewTokenizer(tokenizerPath)
		if err != nil {
			t.Fatalf("Failed to load tokenizer: %v", err)
		}
		defer tokenizer.Close()

		text := "Hello, world!"
		tokens, err := tokenizer.Encode(text)
		if err != nil {
			t.Fatalf("Encoding failed: %v", err)
		}

		if len(tokens) == 0 {
			t.Error("Tokens should not be empty")
		}

		decoded, err := tokenizer.Decode(tokens)
		if err != nil {
			t.Fatalf("Decoding failed: %v", err)
		}

		// The decoded text might not be exactly the same due to tokenization
		// but should not be empty
		if len(decoded) == 0 {
			t.Error("Decoded text should not be empty")
		}
	})
}

// Example tests that serve as documentation
func ExampleModel_Generate() {
	model, err := NewModel("./models/gpt2")
	if err != nil {
		panic(err)
	}
	defer model.Close()

	result, err := model.Generate("The future of AI is")
	if err != nil {
		panic(err)
	}

	println("Generated:", result)
}

func ExamplePipeline_Predict() {
	pipeline, err := NewPipeline("text-classification", "sentiment-analysis")
	if err != nil {
		panic(err)
	}
	defer pipeline.Close()

	result, err := pipeline.Predict("I love this product!")
	if err != nil {
		panic(err)
	}

	println("Sentiment:", result)
}

func ExampleGenerateText() {
	result, err := GenerateText("./models/gpt2", "Once upon a time")
	if err != nil {
		panic(err)
	}

	println("Story:", result)
}