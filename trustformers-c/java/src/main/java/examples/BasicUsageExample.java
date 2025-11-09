package examples;

import com.trustformers.*;
import com.trustformers.Pipeline.GenerationOptions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Basic usage example demonstrating TrustformeRS Java bindings.
 * Shows initialization, model loading, and various NLP tasks.
 */
public class BasicUsageExample {
    
    private static final Logger logger = LoggerFactory.getLogger(BasicUsageExample.class);
    
    public static void main(String[] args) {
        try {
            // Initialize TrustformeRS
            System.out.println("Initializing TrustformeRS...");
            TrustformeRS trustformers = new TrustformeRS();
            
            // Display version and build information
            System.out.println("TrustformeRS Version: " + trustformers.getVersion());
            TrustformeRS.BuildInfo buildInfo = trustformers.getBuildInfo();
            System.out.println("Build Info: " + buildInfo);
            
            // Check available features
            System.out.println("GPU Support: " + trustformers.hasFeature("gpu"));
            System.out.println("CUDA Support: " + trustformers.hasFeature("cuda"));
            System.out.println("SIMD Support: " + trustformers.hasFeature("simd"));
            
            // Configure optimization
            TrustformeRS.OptimizationConfig config = TrustformeRS.OptimizationConfig.defaultConfig();
            config.enableSIMD = true;
            config.cacheSizeMB = 512;
            trustformers.applyOptimizations(config);
            System.out.println("Applied performance optimizations");
            
            // Display memory usage
            TrustformeRS.MemoryUsage memUsage = trustformers.getMemoryUsage();
            System.out.println("Initial Memory Usage: " + memUsage);
            
            // Text Generation Example
            demonstrateTextGeneration(trustformers);
            
            // Text Classification Example
            demonstrateTextClassification(trustformers);
            
            // Question Answering Example
            demonstrateQuestionAnswering(trustformers);
            
            // Conversational Example
            demonstrateConversation(trustformers);
            
            // Performance monitoring
            demonstratePerformanceMonitoring(trustformers);
            
            // Memory monitoring
            demonstrateMemoryMonitoring(trustformers);
            
            // Cleanup
            trustformers.close();
            System.out.println("TrustformeRS cleanup completed");
            
        } catch (Exception e) {
            logger.error("Error in BasicUsageExample", e);
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void demonstrateTextGeneration(TrustformeRS trustformers) {
        System.out.println("\n" + "=".repeat(50));
        System.out.println("TEXT GENERATION EXAMPLE");
        System.out.println("=".repeat(50));
        
        try {
            // Load model and tokenizer for text generation
            System.out.println("Loading GPT-2 model and tokenizer...");
            Model model = trustformers.loadModelFromHub("gpt2");
            Tokenizer tokenizer = trustformers.loadTokenizerFromHub("gpt2");
            
            // Display model information
            Model.ModelInfo modelInfo = model.getInfo();
            System.out.println("Model Info: " + modelInfo);
            
            // Display tokenizer information
            Tokenizer.TokenizerInfo tokenizerInfo = tokenizer.getInfo();
            System.out.println("Tokenizer Info: " + tokenizerInfo);
            
            // Create text generation pipeline
            Pipeline pipeline = Pipeline.createTextGeneration(trustformers, model, tokenizer);
            
            // Basic text generation
            String prompt = "The future of artificial intelligence is";
            System.out.println("\nPrompt: " + prompt);
            String generated = pipeline.generateText(prompt);
            System.out.println("Generated: " + generated);
            
            // Advanced text generation with options
            GenerationOptions options = new GenerationOptions()
                .maxLength(100)
                .temperature(0.8)
                .topK(50)
                .doSample(true);
            
            String advancedGenerated = pipeline.generateText(prompt, options);
            System.out.println("\nAdvanced Generated: " + advancedGenerated);
            
            // Cleanup
            pipeline.close();
            tokenizer.close();
            model.close();
            
        } catch (TrustformersException e) {
            System.err.println("Text generation failed: " + e.getMessage());
        }
    }
    
    private static void demonstrateTextClassification(TrustformeRS trustformers) {
        System.out.println("\n" + "=".repeat(50));
        System.out.println("TEXT CLASSIFICATION EXAMPLE");
        System.out.println("=".repeat(50));
        
        try {
            // Load classification model
            System.out.println("Loading DistilBERT classification model...");
            Model model = trustformers.loadModelFromHub("distilbert-base-uncased-finetuned-sst-2-english");
            Tokenizer tokenizer = trustformers.loadTokenizerFromHub("distilbert-base-uncased-finetuned-sst-2-english");
            
            // Create classification pipeline
            Pipeline pipeline = Pipeline.createTextClassification(trustformers, model, tokenizer);
            
            // Single text classification
            String text = "I love this product! It's amazing!";
            System.out.println("Text: " + text);
            Pipeline.ClassificationResult[] results = pipeline.classifyText(text);
            
            System.out.println("Classification Results:");
            for (Pipeline.ClassificationResult result : results) {
                System.out.println("  " + result);
            }
            
            // Batch classification
            String[] texts = {
                "This movie is terrible.",
                "I'm so happy with this purchase!",
                "The weather is okay today.",
                "This is the worst service ever!"
            };
            
            System.out.println("\nBatch Classification:");
            Pipeline.ClassificationResult[][] batchResults = pipeline.classifyTextBatch(texts);
            
            for (int i = 0; i < texts.length; i++) {
                System.out.println("Text: " + texts[i]);
                for (Pipeline.ClassificationResult result : batchResults[i]) {
                    System.out.println("  " + result);
                }
            }
            
            // Cleanup
            pipeline.close();
            tokenizer.close();
            model.close();
            
        } catch (TrustformersException e) {
            System.err.println("Text classification failed: " + e.getMessage());
        }
    }
    
    private static void demonstrateQuestionAnswering(TrustformeRS trustformers) {
        System.out.println("\n" + "=".repeat(50));
        System.out.println("QUESTION ANSWERING EXAMPLE");
        System.out.println("=".repeat(50));
        
        try {
            // Load QA model
            System.out.println("Loading BERT QA model...");
            Model model = trustformers.loadModelFromHub("bert-large-uncased-whole-word-masking-finetuned-squad");
            Tokenizer tokenizer = trustformers.loadTokenizerFromHub("bert-large-uncased-whole-word-masking-finetuned-squad");
            
            // Create QA pipeline
            Pipeline pipeline = Pipeline.createQuestionAnswering(trustformers, model, tokenizer);
            
            // Question answering
            String context = "TrustformeRS is a high-performance transformer library written in Rust. " +
                           "It provides fast inference and training for various NLP tasks including " +
                           "text generation, classification, and question answering. The library " +
                           "supports GPU acceleration through CUDA and can be used from multiple " +
                           "programming languages including Java, Python, and Go.";
            
            String question = "What programming language is TrustformeRS written in?";
            
            System.out.println("Context: " + context);
            System.out.println("Question: " + question);
            
            Pipeline.AnswerResult answer = pipeline.answerQuestion(context, question);
            System.out.println("Answer: " + answer);
            
            // Multiple questions
            String[] questions = {
                "What tasks does TrustformeRS support?",
                "Does TrustformeRS support GPU acceleration?",
                "What languages can use TrustformeRS?"
            };
            
            System.out.println("\nMultiple Questions:");
            for (String q : questions) {
                System.out.println("Q: " + q);
                Pipeline.AnswerResult ans = pipeline.answerQuestion(context, q);
                System.out.println("A: " + ans.answer + " (score: " + String.format("%.3f", ans.score) + ")");
            }
            
            // Cleanup
            pipeline.close();
            tokenizer.close();
            model.close();
            
        } catch (TrustformersException e) {
            System.err.println("Question answering failed: " + e.getMessage());
        }
    }
    
    private static void demonstrateConversation(TrustformeRS trustformers) {
        System.out.println("\n" + "=".repeat(50));
        System.out.println("CONVERSATIONAL EXAMPLE");
        System.out.println("=".repeat(50));
        
        try {
            // Load conversational model
            System.out.println("Loading DialoGPT model...");
            Model model = trustformers.loadModelFromHub("microsoft/DialoGPT-medium");
            Tokenizer tokenizer = trustformers.loadTokenizerFromHub("microsoft/DialoGPT-medium");
            
            // Create conversational pipeline
            Pipeline pipeline = Pipeline.createConversational(trustformers, model, tokenizer);
            
            // Simulate a conversation
            String[] userInputs = {
                "Hello! How are you today?",
                "What's your favorite programming language?",
                "Can you help me with machine learning?",
                "Thank you for your help!"
            };
            
            System.out.println("Starting conversation:");
            for (String userInput : userInputs) {
                System.out.println("User: " + userInput);
                String botResponse = pipeline.addConversationTurn(userInput);
                System.out.println("Bot: " + botResponse);
            }
            
            // Get conversation history
            System.out.println("\nConversation History:");
            Pipeline.ConversationTurn[] history = pipeline.getConversationHistory();
            for (Pipeline.ConversationTurn turn : history) {
                System.out.println("  " + turn);
            }
            
            // Clear conversation and start fresh
            pipeline.clearConversation();
            System.out.println("\nConversation cleared");
            
            // Cleanup
            pipeline.close();
            tokenizer.close();
            model.close();
            
        } catch (TrustformersException e) {
            System.err.println("Conversation failed: " + e.getMessage());
        }
    }
    
    private static void demonstratePerformanceMonitoring(TrustformeRS trustformers) {
        System.out.println("\n" + "=".repeat(50));
        System.out.println("PERFORMANCE MONITORING EXAMPLE");
        System.out.println("=".repeat(50));
        
        try {
            // Start profiling
            trustformers.startProfiling();
            System.out.println("Started performance profiling");
            
            // Simulate some work
            Thread.sleep(1000);
            
            // Stop profiling and get report
            System.out.println("Stopping profiling and generating report...");
            System.out.println("Performance Report: " + trustformers.stopProfiling());
            
        } catch (Exception e) {
            System.err.println("Performance monitoring failed: " + e.getMessage());
        }
    }
    
    private static void demonstrateMemoryMonitoring(TrustformeRS trustformers) {
        System.out.println("\n" + "=".repeat(50));
        System.out.println("MEMORY MONITORING EXAMPLE");
        System.out.println("=".repeat(50));
        
        try {
            // Get current memory usage
            TrustformeRS.MemoryUsage memUsage = trustformers.getMemoryUsage();
            System.out.println("Current Memory Usage: " + memUsage);
            
            // Get advanced memory statistics
            System.out.println("Advanced Memory Usage: " + trustformers.getAdvancedMemoryUsage());
            
            // Check for memory leaks
            System.out.println("Memory Leak Report: " + trustformers.checkMemoryLeaks());
            
            // Set memory limits
            trustformers.setMemoryLimits(2048, 1536); // 2GB max, warning at 1.5GB
            System.out.println("Set memory limits: 2GB max, warning at 1.5GB");
            
            // Force memory cleanup
            trustformers.memoryCleanup();
            System.out.println("Performed memory cleanup");
            
            // Check memory usage after cleanup
            TrustformeRS.MemoryUsage afterCleanup = trustformers.getMemoryUsage();
            System.out.println("Memory Usage After Cleanup: " + afterCleanup);
            
        } catch (Exception e) {
            System.err.println("Memory monitoring failed: " + e.getMessage());
        }
    }
}