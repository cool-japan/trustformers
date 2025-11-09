package examples;

import ai.trustformers.*;
import java.util.Map;
import java.util.HashMap;
import java.util.List;

/**
 * Comprehensive examples demonstrating TrustFormeRS Java API usage.
 * 
 * This class shows how to use Models, Tokenizers, and Pipelines
 * for various machine learning tasks.
 */
public class TrustFormeRSExample {
    
    public static void main(String[] args) {
        try {
            // Print system information
            System.out.println("=== TrustFormeRS System Information ===");
            TrustFormeRS.printSystemInfo();
            System.out.println();
            
            // Example 1: Basic Model Usage
            basicModelExample();
            
            // Example 2: Tokenizer Usage
            tokenizerExample();
            
            // Example 3: Text Generation Pipeline
            textGenerationExample();
            
            // Example 4: Text Classification Pipeline
            textClassificationExample();
            
            // Example 5: Question Answering Pipeline
            questionAnsweringExample();
            
            // Example 6: Summarization Pipeline
            summarizationExample();
            
            // Example 7: Batch Processing
            batchProcessingExample();
            
            // Example 8: Hub Integration
            hubIntegrationExample();
            
        } catch (Exception e) {
            System.err.println("Error in examples: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Example 1: Basic Model Usage
     */
    private static void basicModelExample() {
        System.out.println("=== Example 1: Basic Model Usage ===");
        
        try (Model model = Model.fromHub("gpt2")) {
            // Get model information
            System.out.println("Model Info: " + model.getModelInfo());
            
            // Generate text
            String prompt = "The future of artificial intelligence is";
            String generated = model.generateText(prompt);
            System.out.println("Generated: " + generated);
            
            // Get embeddings
            float[] embeddings = model.getEmbeddings("Hello, world!");
            System.out.println("Embedding dimension: " + embeddings.length);
            
        } catch (TrustFormeRSException e) {
            System.err.println("Model example failed: " + e.getUserFriendlyMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Example 2: Tokenizer Usage
     */
    private static void tokenizerExample() {
        System.out.println("=== Example 2: Tokenizer Usage ===");
        
        try (Tokenizer tokenizer = Tokenizer.fromHub("gpt2")) {
            String text = "Hello, world! How are you today?";
            
            // Encode text
            int[] tokens = tokenizer.encode(text);
            System.out.println("Original text: " + text);
            System.out.println("Token count: " + tokens.length);
            
            // Decode tokens
            String decoded = tokenizer.decode(tokens);
            System.out.println("Decoded text: " + decoded);
            
            // Tokenize to strings
            String[] tokenStrings = tokenizer.tokenize(text);
            System.out.println("Tokens: " + String.join(" | ", tokenStrings));
            
            // Vocabulary size
            System.out.println("Vocabulary size: " + tokenizer.getVocabSize());
            
        } catch (TrustFormeRSException e) {
            System.err.println("Tokenizer example failed: " + e.getUserFriendlyMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Example 3: Text Generation Pipeline
     */
    private static void textGenerationExample() {
        System.out.println("=== Example 3: Text Generation Pipeline ===");
        
        try (Pipeline pipeline = Pipeline.fromHub(Pipeline.TEXT_GENERATION, "gpt2")) {
            String prompt = "Once upon a time, in a land far away,";
            
            // Generate with custom parameters
            String generated = pipeline.generateText(prompt, 50, 0.8);
            System.out.println("Prompt: " + prompt);
            System.out.println("Generated: " + generated);
            
            // Alternative using process method
            Map<String, Object> config = new HashMap<>();
            config.put("max_length", 30);
            config.put("temperature", 0.5);
            config.put("top_p", 0.9);
            
            Map<String, Object> result = pipeline.process(prompt, config);
            System.out.println("Alternative generation: " + result.get("generated_text"));
            
        } catch (TrustFormeRSException e) {
            System.err.println("Text generation example failed: " + e.getUserFriendlyMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Example 4: Text Classification Pipeline
     */
    private static void textClassificationExample() {
        System.out.println("=== Example 4: Text Classification Pipeline ===");
        
        try (Pipeline pipeline = Pipeline.fromHub(Pipeline.TEXT_CLASSIFICATION, "distilbert-base-uncased-finetuned-sst-2-english")) {
            String text = "I love this new product! It's amazing.";
            
            // Classify text
            List<Map<String, Object>> results = pipeline.classifyText(text);
            System.out.println("Text: " + text);
            System.out.println("Classifications:");
            
            for (Map<String, Object> result : results) {
                String label = (String) result.get("label");
                Double score = (Double) result.get("score");
                System.out.printf("  %s: %.4f%n", label, score);
            }
            
        } catch (TrustFormeRSException e) {
            System.err.println("Text classification example failed: " + e.getUserFriendlyMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Example 5: Question Answering Pipeline
     */
    private static void questionAnsweringExample() {
        System.out.println("=== Example 5: Question Answering Pipeline ===");
        
        try (Pipeline pipeline = Pipeline.fromHub(Pipeline.QUESTION_ANSWERING, "distilbert-base-cased-distilled-squad")) {
            String question = "What is the capital of France?";
            String context = "France is a country in Western Europe. Its capital and largest city is Paris, " +
                           "which is located in the north-central part of the country.";
            
            // Answer question
            Map<String, Object> result = pipeline.answerQuestion(question, context);
            
            System.out.println("Question: " + question);
            System.out.println("Context: " + context);
            System.out.println("Answer: " + result.get("answer"));
            System.out.println("Score: " + result.get("score"));
            System.out.println("Start: " + result.get("start"));
            System.out.println("End: " + result.get("end"));
            
        } catch (TrustFormeRSException e) {
            System.err.println("Question answering example failed: " + e.getUserFriendlyMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Example 6: Summarization Pipeline
     */
    private static void summarizationExample() {
        System.out.println("=== Example 6: Summarization Pipeline ===");
        
        try (Pipeline pipeline = Pipeline.fromHub(Pipeline.SUMMARIZATION, "t5-small")) {
            String longText = "Artificial intelligence (AI) is intelligence demonstrated by machines, " +
                            "in contrast to the natural intelligence displayed by humans and animals. " +
                            "Leading AI textbooks define the field as the study of intelligent agents: " +
                            "any device that perceives its environment and takes actions that maximize " +
                            "its chance of successfully achieving its goals. Colloquially, the term " +
                            "artificial intelligence is often used to describe machines that mimic " +
                            "cognitive functions that humans associate with the human mind, such as " +
                            "learning and problem solving.";
            
            // Summarize text
            String summary = pipeline.summarizeText(longText, 50, 10);
            
            System.out.println("Original text (" + longText.length() + " chars): " + longText);
            System.out.println("Summary: " + summary);
            
        } catch (TrustFormeRSException e) {
            System.err.println("Summarization example failed: " + e.getUserFriendlyMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Example 7: Batch Processing
     */
    private static void batchProcessingExample() {
        System.out.println("=== Example 7: Batch Processing ===");
        
        try (Pipeline pipeline = Pipeline.fromHub(Pipeline.TEXT_CLASSIFICATION, "distilbert-base-uncased-finetuned-sst-2-english")) {
            String[] texts = {
                "I love this product!",
                "This is terrible.",
                "It's okay, nothing special.",
                "Amazing quality and great service!",
                "I'm not sure about this."
            };
            
            // Process batch
            List<Map<String, Object>> results = pipeline.processBatch(texts);
            
            System.out.println("Batch classification results:");
            for (int i = 0; i < texts.length; i++) {
                System.out.println("Text " + (i + 1) + ": " + texts[i]);
                Map<String, Object> result = results.get(i);
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> classifications = (List<Map<String, Object>>) result.get("classifications");
                
                for (Map<String, Object> classification : classifications) {
                    String label = (String) classification.get("label");
                    Double score = (Double) classification.get("score");
                    System.out.printf("  %s: %.4f%n", label, score);
                }
                System.out.println();
            }
            
        } catch (TrustFormeRSException e) {
            System.err.println("Batch processing example failed: " + e.getUserFriendlyMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Example 8: Hub Integration
     */
    private static void hubIntegrationExample() {
        System.out.println("=== Example 8: Hub Integration ===");
        
        try {
            // Load different models from Hub
            String[] modelNames = {"gpt2", "bert-base-uncased", "t5-small"};
            
            for (String modelName : modelNames) {
                try (Model model = Model.fromHub(modelName)) {
                    System.out.println("Loaded model: " + modelName);
                    System.out.println("Model info: " + model.getModelInfo());
                    
                    // Test basic functionality
                    if (modelName.startsWith("gpt")) {
                        String generated = model.generateText("Hello", 10, 0.7);
                        System.out.println("Sample generation: " + generated);
                    } else {
                        float[] embeddings = model.getEmbeddings("test");
                        System.out.println("Embedding dimension: " + embeddings.length);
                    }
                    
                    System.out.println();
                } catch (TrustFormeRSException e) {
                    System.err.println("Failed to load " + modelName + ": " + e.getMessage());
                }
            }
            
        } catch (Exception e) {
            System.err.println("Hub integration example failed: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Utility method to print device information
     */
    private static void printDeviceInfo() {
        System.out.println("=== Device Information ===");
        
        Map<Integer, String> devices = TrustFormeRS.getAllDeviceInfo();
        if (devices.isEmpty()) {
            System.out.println("No devices found");
        } else {
            for (Map.Entry<Integer, String> entry : devices.entrySet()) {
                System.out.println("Device " + entry.getKey() + ": " + entry.getValue());
            }
        }
        
        System.out.println("CUDA Available: " + TrustFormeRS.isCudaAvailable());
        System.out.println("Metal Available: " + TrustFormeRS.isMetalAvailable());
        System.out.println();
    }
}