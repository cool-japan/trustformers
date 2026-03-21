package com.trustformers;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for TrustformeRS Java bindings.
 * 
 * Note: These tests require the TrustformeRS native library to be available.
 * Run 'cargo build --release' in the parent directory before running tests.
 */
public class TrustformersTest {
    
    private static final Logger logger = LoggerFactory.getLogger(TrustformersTest.class);
    
    private TrustformeRS trustformers;
    
    @BeforeEach
    void setUp(TestInfo testInfo) {
        logger.info("Starting test: {}", testInfo.getDisplayName());
        
        try {
            trustformers = new TrustformeRS();
            logger.info("TrustformeRS initialized successfully for test");
        } catch (UnsatisfiedLinkError e) {
            logger.warn("Native library not available, skipping test: {}", e.getMessage());
            org.junit.jupiter.api.Assumptions.assumeTrue(false, "Native library not available");
        } catch (TrustformersException e) {
            logger.error("Failed to initialize TrustformeRS", e);
            fail("Failed to initialize TrustformeRS: " + e.getMessage());
        }
    }
    
    @AfterEach
    void tearDown(TestInfo testInfo) {
        if (trustformers != null) {
            trustformers.close();
            logger.info("TrustformeRS cleaned up after test: {}", testInfo.getDisplayName());
        }
    }
    
    @Test
    void testInitialization() {
        assertNotNull(trustformers);
        assertTrue(trustformers.isInitialized());
        logger.info("✓ Initialization test passed");
    }
    
    @Test
    void testVersion() {
        String version = trustformers.getVersion();
        assertNotNull(version);
        assertFalse(version.trim().isEmpty());
        logger.info("✓ Version test passed: {}", version);
    }
    
    @Test
    void testBuildInfo() throws TrustformersException {
        TrustformeRS.BuildInfo buildInfo = trustformers.getBuildInfo();
        assertNotNull(buildInfo);
        assertNotNull(buildInfo.version);
        assertNotNull(buildInfo.features);
        assertNotNull(buildInfo.buildDate);
        assertNotNull(buildInfo.target);
        logger.info("✓ Build info test passed: {}", buildInfo);
    }
    
    @Test
    void testFeatureCheck() {
        // Test common features
        boolean gpuSupport = trustformers.hasFeature("gpu");
        boolean cudaSupport = trustformers.hasFeature("cuda");
        boolean simdSupport = trustformers.hasFeature("simd");
        
        // These should not throw exceptions
        assertNotNull(gpuSupport);
        assertNotNull(cudaSupport);
        assertNotNull(simdSupport);
        
        logger.info("✓ Feature check test passed - GPU: {}, CUDA: {}, SIMD: {}", 
                   gpuSupport, cudaSupport, simdSupport);
    }
    
    @Test
    void testLogLevel() throws TrustformersException {
        // Test setting different log levels
        trustformers.setLogLevel(TrustformeRS.LogLevel.DEBUG);
        trustformers.setLogLevel(TrustformeRS.LogLevel.INFO);
        trustformers.setLogLevel(TrustformeRS.LogLevel.WARN);
        trustformers.setLogLevel(TrustformeRS.LogLevel.ERROR);
        trustformers.setLogLevel(TrustformeRS.LogLevel.OFF);
        
        logger.info("✓ Log level test passed");
    }
    
    @Test
    void testMemoryUsage() throws TrustformersException {
        TrustformeRS.MemoryUsage memUsage = trustformers.getMemoryUsage();
        assertNotNull(memUsage);
        assertTrue(memUsage.totalMemoryBytes >= 0);
        assertTrue(memUsage.peakMemoryBytes >= 0);
        assertTrue(memUsage.allocatedModels >= 0);
        assertTrue(memUsage.allocatedTokenizers >= 0);
        assertTrue(memUsage.allocatedPipelines >= 0);
        assertTrue(memUsage.allocatedTensors >= 0);
        
        logger.info("✓ Memory usage test passed: {}", memUsage);
    }
    
    @Test
    void testAdvancedMemoryUsage() throws TrustformersException {
        var advancedUsage = trustformers.getAdvancedMemoryUsage();
        assertNotNull(advancedUsage);
        
        logger.info("✓ Advanced memory usage test passed");
    }
    
    @Test
    void testMemoryOperations() throws TrustformersException {
        // Test memory cleanup
        trustformers.memoryCleanup();
        
        // Test memory limits
        trustformers.setMemoryLimits(1024, 768); // 1GB max, warning at 768MB
        
        // Test memory leak check
        var leakReport = trustformers.checkMemoryLeaks();
        assertNotNull(leakReport);
        
        logger.info("✓ Memory operations test passed");
    }
    
    @Test
    void testOptimizations() throws TrustformersException {
        TrustformeRS.OptimizationConfig config = TrustformeRS.OptimizationConfig.defaultConfig();
        config.enableSIMD = true;
        config.enableCaching = true;
        config.cacheSizeMB = 256;
        config.numThreads = 2;
        
        // This should not throw an exception
        trustformers.applyOptimizations(config);
        
        logger.info("✓ Optimizations test passed");
    }
    
    @Test
    void testProfiling() throws TrustformersException {
        // Start profiling
        trustformers.startProfiling();
        
        // Simulate some work
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        // Stop profiling
        var report = trustformers.stopProfiling();
        assertNotNull(report);
        
        logger.info("✓ Profiling test passed");
    }
    
    @Test
    void testExceptionHandling() {
        // Test invalid feature name
        assertDoesNotThrow(() -> {
            boolean result = trustformers.hasFeature("invalid_feature_name_that_does_not_exist");
            assertFalse(result); // Should return false, not throw
        });
        
        // Test invalid log level (this should handle gracefully)
        assertDoesNotThrow(() -> {
            try {
                // We can't directly test invalid enum values, but we can test the method
                trustformers.setLogLevel(TrustformeRS.LogLevel.INFO);
            } catch (TrustformersException e) {
                // This is expected for some invalid parameters
                assertTrue(e.getErrorCode() != TrustformersException.ErrorCode.SUCCESS);
            }
        });
        
        logger.info("✓ Exception handling test passed");
    }
    
    @Test
    void testResourceManagement() throws TrustformersException {
        // Test multiple initialization and cleanup cycles
        TrustformeRS tf1 = new TrustformeRS();
        assertTrue(tf1.isInitialized());
        tf1.close();
        assertFalse(tf1.isInitialized());
        
        TrustformeRS tf2 = new TrustformeRS();
        assertTrue(tf2.isInitialized());
        tf2.close();
        assertFalse(tf2.isInitialized());
        
        logger.info("✓ Resource management test passed");
    }
    
    @Test
    void testConcurrentAccess() throws InterruptedException {
        // Test that multiple threads can safely access the same TrustformeRS instance
        final int numThreads = 4;
        final int operationsPerThread = 10;
        
        Thread[] threads = new Thread[numThreads];
        final boolean[] results = new boolean[numThreads];
        
        for (int i = 0; i < numThreads; i++) {
            final int threadIndex = i;
            threads[i] = new Thread(() -> {
                try {
                    for (int j = 0; j < operationsPerThread; j++) {
                        String version = trustformers.getVersion();
                        assertNotNull(version);
                        
                        boolean hasGpu = trustformers.hasFeature("gpu");
                        // Just ensure it doesn't throw
                        
                        TrustformeRS.MemoryUsage usage = trustformers.getMemoryUsage();
                        assertNotNull(usage);
                    }
                    results[threadIndex] = true;
                } catch (Exception e) {
                    logger.error("Thread {} failed", threadIndex, e);
                    results[threadIndex] = false;
                }
            });
        }
        
        // Start all threads
        for (Thread thread : threads) {
            thread.start();
        }
        
        // Wait for all threads to complete
        for (Thread thread : threads) {
            thread.join(5000); // 5 second timeout
        }
        
        // Check that all threads succeeded
        for (int i = 0; i < numThreads; i++) {
            assertTrue(results[i], "Thread " + i + " failed");
        }
        
        logger.info("✓ Concurrent access test passed");
    }
}