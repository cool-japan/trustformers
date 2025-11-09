// +build serving

package main

import (
	"fmt"
	"log"
	"time"

	"github.com/trustformers/trustformers-c/golang/trustformers"
)

func main() {
	// Initialize TrustformeRS
	tf, err := trustformers.NewTrustformeRS()
	if err != nil {
		log.Fatalf("Failed to initialize TrustformeRS: %v", err)
	}
	defer tf.Cleanup()

	fmt.Println("TrustformeRS HTTP Server Demo")
	fmt.Println("==========================================")

	// Create HTTP server with custom configuration
	config := trustformers.HttpServerConfig{
		Host:             "127.0.0.1",
		Port:             8080,
		MaxConnections:   200,
		RequestTimeoutMs: 30000,
		EnableCors:       true,
		CorsOrigins:      []string{"*"},
		EnableMetrics:    true,
		MetricsEndpoint:  "/metrics",
		HealthEndpoint:   "/health",
		ApiPrefix:        "/api/v1",
		EnableSSL:        false,
	}

	server, err := trustformers.NewHttpServerWithConfig(config)
	if err != nil {
		log.Fatalf("Failed to create HTTP server: %v", err)
	}
	defer server.Destroy()

	fmt.Printf("HTTP server created with ID: %s\n", server.GetServerID())
	fmt.Printf("Server address: %s\n", server.GetAddress())
	fmt.Printf("Health endpoint: %s\n", server.GetHealthEndpoint())
	fmt.Printf("Metrics endpoint: %s\n", server.GetMetricsEndpoint())

	// Add text generation endpoints
	fmt.Println("\nAdding model endpoints...")
	
	err = server.AddTextGenerationEndpoint(
		"gpt2-text-generator",
		"/models/gpt2",
		"/generate/gpt2",
		trustformers.WithMaxBatchSize(16),
		trustformers.WithTimeout(30000),
		trustformers.WithStreaming(true),
		trustformers.WithRateLimit(100, 10), // 100 requests per minute, burst of 10
	)
	if err != nil {
		log.Fatalf("Failed to add text generation endpoint: %v", err)
	}
	fmt.Println("✓ Added GPT-2 text generation endpoint: /api/v1/generate/gpt2")

	err = server.AddTextGenerationEndpoint(
		"llama-text-generator", 
		"/models/llama-7b",
		"/generate/llama",
		trustformers.WithMaxBatchSize(8),
		trustformers.WithTimeout(45000),
		trustformers.WithStreaming(true),
	)
	if err != nil {
		log.Fatalf("Failed to add LLaMA endpoint: %v", err)
	}
	fmt.Println("✓ Added LLaMA text generation endpoint: /api/v1/generate/llama")

	// Add text classification endpoints
	err = server.AddTextClassificationEndpoint(
		"sentiment-classifier",
		"/models/distilbert-sentiment",
		"/classify/sentiment",
		trustformers.WithMaxBatchSize(64),
		trustformers.WithTimeout(15000),
		trustformers.WithRateLimit(200, 20),
	)
	if err != nil {
		log.Fatalf("Failed to add sentiment classification endpoint: %v", err)
	}
	fmt.Println("✓ Added sentiment classification endpoint: /api/v1/classify/sentiment")

	err = server.AddTextClassificationEndpoint(
		"topic-classifier",
		"/models/bert-topic",
		"/classify/topic",
		trustformers.WithMaxBatchSize(32),
		trustformers.WithTimeout(20000),
	)
	if err != nil {
		log.Fatalf("Failed to add topic classification endpoint: %v", err)
	}
	fmt.Println("✓ Added topic classification endpoint: /api/v1/classify/topic")

	// Add a custom endpoint using the generic method
	customEndpoint := trustformers.ModelEndpoint{
		Name:            "custom-qa-model",
		ModelPath:       "/models/bert-qa",
		EndpointPath:    "/qa/bert",
		MaxBatchSize:    16,
		TimeoutMs:       25000,
		EnableStreaming: false,
		RateLimit: &trustformers.RateLimit{
			RequestsPerMinute: 50,
			BurstLimit:        5,
		},
	}
	
	err = server.AddModelEndpoint(customEndpoint)
	if err != nil {
		log.Fatalf("Failed to add custom QA endpoint: %v", err)
	}
	fmt.Println("✓ Added custom QA endpoint: /api/v1/qa/bert")

	fmt.Println("\nStarting HTTP server...")
	
	// Start the server
	err = server.Start()
	if err != nil {
		log.Fatalf("Failed to start HTTP server: %v", err)
	}
	
	fmt.Printf("🚀 HTTP server started successfully on %s\n", server.GetAddress())
	fmt.Println("\nAvailable endpoints:")
	fmt.Println("  Health Check:    GET  " + server.GetHealthEndpoint())
	fmt.Println("  Metrics:         GET  " + server.GetMetricsEndpoint())
	fmt.Println("  GPT-2 Generate:  POST http://127.0.0.1:8080/api/v1/generate/gpt2")
	fmt.Println("  LLaMA Generate:  POST http://127.0.0.1:8080/api/v1/generate/llama")
	fmt.Println("  Sentiment:       POST http://127.0.0.1:8080/api/v1/classify/sentiment")
	fmt.Println("  Topic Class:     POST http://127.0.0.1:8080/api/v1/classify/topic")
	fmt.Println("  Question Answer: POST http://127.0.0.1:8080/api/v1/qa/bert")

	fmt.Println("\nExample requests:")
	fmt.Println("Text Generation:")
	fmt.Println(`  curl -X POST http://127.0.0.1:8080/api/v1/generate/gpt2 \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The future of AI is", "max_length": 100, "temperature": 0.8}'`)

	fmt.Println("\nText Classification:")
	fmt.Println(`  curl -X POST http://127.0.0.1:8080/api/v1/classify/sentiment \
    -H "Content-Type: application/json" \
    -d '{"text": "I love this product!"}'`)

	// Simulate server operation and monitoring
	fmt.Println("\nServer is running. Monitoring metrics...")
	
	for i := 0; i < 10; i++ {
		time.Sleep(2 * time.Second)
		
		// Get current metrics
		metrics, err := server.GetMetrics()
		if err != nil {
			log.Printf("Failed to get metrics: %v", err)
			continue
		}

		fmt.Printf("\n[%s] Server Status:\n", time.Now().Format("15:04:05"))
		fmt.Printf("  Running: %v\n", server.IsRunning())
		fmt.Printf("  Total Requests: %d\n", metrics.TotalRequests)
		fmt.Printf("  Successful: %d\n", metrics.SuccessfulRequests)
		fmt.Printf("  Failed: %d\n", metrics.FailedRequests)
		fmt.Printf("  Avg Response Time: %.2f ms\n", metrics.AverageResponseTimeMs)
		fmt.Printf("  Current Connections: %d\n", metrics.CurrentConnections)
		fmt.Printf("  Peak Connections: %d\n", metrics.PeakConnections)
		fmt.Printf("  Bytes Sent: %d\n", metrics.BytesSent)
		fmt.Printf("  Bytes Received: %d\n", metrics.BytesReceived)
		
		if len(metrics.ModelRequests) > 0 {
			fmt.Println("  Model Requests:")
			for model, requests := range metrics.ModelRequests {
				fmt.Printf("    %s: %d\n", model, requests)
			}
		}
	}

	fmt.Println("\nDemonstrating server management...")
	
	// Stop the server
	fmt.Println("Stopping server...")
	err = server.Stop()
	if err != nil {
		log.Printf("Failed to stop server: %v", err)
	} else {
		fmt.Println("✓ Server stopped successfully")
	}
	
	// Wait a moment
	time.Sleep(1 * time.Second)
	
	// Restart the server
	fmt.Println("Restarting server...")
	err = server.Start()
	if err != nil {
		log.Printf("Failed to restart server: %v", err)
	} else {
		fmt.Printf("✓ Server restarted on %s\n", server.GetAddress())
	}
	
	// Wait a bit more
	time.Sleep(3 * time.Second)
	
	// Final metrics check
	fmt.Println("\nFinal metrics check:")
	finalMetrics, err := server.GetMetrics()
	if err != nil {
		log.Printf("Failed to get final metrics: %v", err)
	} else {
		fmt.Printf("  Final total requests: %d\n", finalMetrics.TotalRequests)
		fmt.Printf("  Final success rate: %.2f%%\n", 
			float64(finalMetrics.SuccessfulRequests)/float64(finalMetrics.TotalRequests)*100)
	}

	fmt.Println("\nStopping server for cleanup...")
	err = server.Stop()
	if err != nil {
		log.Printf("Failed to stop server: %v", err)
	}

	fmt.Println("\nHTTP server demo completed!")
	fmt.Println("\nTo test the server with real requests, you can:")
	fmt.Println("1. Start the server")
	fmt.Println("2. Use curl or a REST client to send requests to the endpoints")
	fmt.Println("3. Monitor the metrics endpoint for performance data")
	fmt.Println("4. Check the health endpoint for server status")
}