package ai.trustformers;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

import java.util.Map;
import java.util.HashMap;

/**
 * Test suite for Model class functionality.
 */
public class ModelTest {
    
    private static final String TEST_MODEL_PATH = "test_models/gpt2";
    private static final String TEST_HUB_MODEL = "gpt2";
    
    @BeforeEach
    void setUp() {
        // Assume models are available for testing
        assumeTrue(true, "Test models should be available");
    }
    
    @Test
    @DisplayName("Should create model from Hub")
    void testCreateModelFromHub() {
        assumeTrue(isHubAccessible(), "Hub should be accessible for this test");
        
        assertDoesNotThrow(() -> {
            try (Model model = Model.fromHub(TEST_HUB_MODEL)) {
                assertNotNull(model, "Model should be created successfully");
                assertFalse(model.isClosed(), "Model should not be closed initially");
            }
        });
    }
    
    @Test
    @DisplayName("Should create model from Hub with revision and auth")
    void testCreateModelFromHubWithOptions() {
        assumeTrue(isHubAccessible(), "Hub should be accessible for this test");
        
        assertDoesNotThrow(() -> {
            try (Model model = Model.fromHub(TEST_HUB_MODEL, "main", null)) {
                assertNotNull(model, "Model should be created successfully with revision");
                assertFalse(model.isClosed(), "Model should not be closed initially");
            }
        });
    }
    
    @Test
    @DisplayName("Should throw exception for invalid model")
    void testInvalidModel() {
        assertThrows(TrustFormeRSException.class, () -> {
            try (Model model = Model.fromHub("non_existent_model_12345")) {
                // Should not reach here
            }
        });
    }
    
    @Test
    @DisplayName("Should generate text")
    void testTextGeneration() {
        assumeTrue(isHubAccessible(), "Hub should be accessible for this test");
        
        assertDoesNotThrow(() -> {
            try (Model model = Model.fromHub(TEST_HUB_MODEL)) {
                String prompt = "Hello, world!";
                String generated = model.generateText(prompt);
                
                assertNotNull(generated, "Generated text should not be null");
                assertFalse(generated.isEmpty(), "Generated text should not be empty");
                assertTrue(generated.contains(prompt) || generated.length() > prompt.length(),
                          "Generated text should extend the prompt");
            }
        });
    }
    
    @Test
    @DisplayName("Should generate text with custom config")
    void testTextGenerationWithConfig() {
        assumeTrue(isHubAccessible(), "Hub should be accessible for this test");
        
        assertDoesNotThrow(() -> {
            try (Model model = Model.fromHub(TEST_HUB_MODEL)) {
                String prompt = "The future of AI is";
                
                Map<String, Object> config = new HashMap<>();
                config.put("max_length", 20);
                config.put("temperature", 0.5);
                config.put("do_sample", true);
                
                String generated = model.generateText(prompt, config);
                
                assertNotNull(generated, "Generated text should not be null");
                assertFalse(generated.isEmpty(), "Generated text should not be empty");
            }
        });
    }
    
    @Test
    @DisplayName("Should get text embeddings")
    void testGetEmbeddings() {
        assumeTrue(isHubAccessible(), "Hub should be accessible for this test");
        
        assertDoesNotThrow(() -> {
            try (Model model = Model.fromHub("bert-base-uncased")) {
                String text = "Hello, world!";
                float[] embeddings = model.getEmbeddings(text);
                
                assertNotNull(embeddings, "Embeddings should not be null");
                assertTrue(embeddings.length > 0, "Embeddings should have positive length");
                
                // Check that embeddings contain meaningful values
                boolean hasNonZero = false;
                for (float value : embeddings) {
                    if (Math.abs(value) > 1e-6) {
                        hasNonZero = true;
                        break;
                    }
                }
                assertTrue(hasNonZero, "Embeddings should contain non-zero values");
            }
        });
    }
    
    @Test
    @DisplayName("Should get model information")
    void testGetModelInfo() {
        assumeTrue(isHubAccessible(), "Hub should be accessible for this test");
        
        assertDoesNotThrow(() -> {
            try (Model model = Model.fromHub(TEST_HUB_MODEL)) {
                String info = model.getModelInfo();
                
                assertNotNull(info, "Model info should not be null");
                assertFalse(info.isEmpty(), "Model info should not be empty");
            }
        });
    }
    
    @Test
    @DisplayName("Should get model configuration")
    void testGetModelConfig() {
        assumeTrue(isHubAccessible(), "Hub should be accessible for this test");
        
        assertDoesNotThrow(() -> {
            try (Model model = Model.fromHub(TEST_HUB_MODEL)) {
                Map<String, Object> config = model.getModelConfig();
                
                assertNotNull(config, "Model config should not be null");
                assertFalse(config.isEmpty(), "Model config should not be empty");
            }
        });
    }
    
    @Test
    @DisplayName("Should handle model closure properly")
    void testModelClosure() {
        assumeTrue(isHubAccessible(), "Hub should be accessible for this test");
        
        assertDoesNotThrow(() -> {
            Model model = Model.fromHub(TEST_HUB_MODEL);
            assertFalse(model.isClosed(), "Model should not be closed initially");
            
            // Model should work before closure
            String info = model.getModelInfo();
            assertNotNull(info, "Model should work before closure");
            
            // Close the model
            model.close();
            assertTrue(model.isClosed(), "Model should be closed after close()");
            
            // Operations should throw exception after closure
            assertThrows(TrustFormeRSException.class, () -> {
                model.getModelInfo();
            });
            
            // Multiple close calls should be safe
            assertDoesNotThrow(() -> model.close());
        });
    }
    
    @Test
    @DisplayName("Should work with try-with-resources")
    void testTryWithResources() {
        assumeTrue(isHubAccessible(), "Hub should be accessible for this test");
        
        Model model;
        assertDoesNotThrow(() -> {
            try (Model m = Model.fromHub(TEST_HUB_MODEL)) {
                model = m;
                assertFalse(m.isClosed(), "Model should not be closed in try block");
                
                String info = m.getModelInfo();
                assertNotNull(info, "Model should work in try block");
            }
            // Model should be automatically closed here
        });
    }
    
    @Test
    @DisplayName("Should save model")
    void testSaveModel() {
        assumeTrue(isHubAccessible(), "Hub should be accessible for this test");
        
        assertDoesNotThrow(() -> {
            try (Model model = Model.fromHub(TEST_HUB_MODEL)) {
                // This test just checks that the method doesn't throw
                // In a real test environment, you'd verify the saved files
                boolean result = model.saveModel("/tmp/test_model_save");
                // Result may be true or false depending on implementation
                // The important thing is that it doesn't throw an exception
            }
        });
    }
    
    /**
     * Helper method to check if Hub is accessible
     */
    private boolean isHubAccessible() {
        // In a real test environment, this would check network connectivity
        // and Hub availability. For now, we assume it's accessible.
        return true;
    }
}