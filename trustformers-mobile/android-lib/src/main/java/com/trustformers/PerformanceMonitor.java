package com.trustformers;

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Performance monitoring for TrustformeRS Android
 * 
 * This class tracks inference performance metrics including
 * timing, memory usage, and throughput.
 */
public class PerformanceMonitor {
    private static final int MAX_SAMPLES = 1000;
    
    private final ConcurrentLinkedQueue<InferenceSample> samples;
    private final AtomicInteger totalInferences;
    private final AtomicLong totalInferenceTimeMs;
    private final AtomicLong totalDataProcessed;
    private long startTimeMs;
    
    /**
     * Single inference sample
     */
    private static class InferenceSample {
        final long timestampMs;
        final long inferenceTimeMs;
        final int inputSizeBytes;
        final int outputSizeBytes;
        final long memoryUsedBytes;
        
        InferenceSample(long timestampMs, long inferenceTimeMs, 
                       int inputSizeBytes, int outputSizeBytes,
                       long memoryUsedBytes) {
            this.timestampMs = timestampMs;
            this.inferenceTimeMs = inferenceTimeMs;
            this.inputSizeBytes = inputSizeBytes;
            this.outputSizeBytes = outputSizeBytes;
            this.memoryUsedBytes = memoryUsedBytes;
        }
    }
    
    public PerformanceMonitor() {
        this.samples = new ConcurrentLinkedQueue<>();
        this.totalInferences = new AtomicInteger(0);
        this.totalInferenceTimeMs = new AtomicLong(0);
        this.totalDataProcessed = new AtomicLong(0);
        this.startTimeMs = System.currentTimeMillis();
    }
    
    /**
     * Record an inference
     * @param inferenceTimeMs Time taken for inference in milliseconds
     * @param inputSizeBytes Size of input data in bytes
     */
    public void recordInference(long inferenceTimeMs, int inputSizeBytes) {
        recordInference(inferenceTimeMs, inputSizeBytes, 0, 0);
    }
    
    /**
     * Record an inference with detailed metrics
     * @param inferenceTimeMs Time taken for inference in milliseconds
     * @param inputSizeBytes Size of input data in bytes
     * @param outputSizeBytes Size of output data in bytes
     * @param memoryUsedBytes Memory used during inference
     */
    public void recordInference(long inferenceTimeMs, int inputSizeBytes, 
                               int outputSizeBytes, long memoryUsedBytes) {
        long timestamp = System.currentTimeMillis();
        
        InferenceSample sample = new InferenceSample(
            timestamp,
            inferenceTimeMs,
            inputSizeBytes,
            outputSizeBytes,
            memoryUsedBytes
        );
        
        samples.offer(sample);
        
        // Maintain max samples
        while (samples.size() > MAX_SAMPLES) {
            samples.poll();
        }
        
        totalInferences.incrementAndGet();
        totalInferenceTimeMs.addAndGet(inferenceTimeMs);
        totalDataProcessed.addAndGet(inputSizeBytes);
    }
    
    /**
     * Record a batch inference
     * @param inferenceTimeMs Time taken for batch inference in milliseconds
     * @param batchSize Number of inputs in the batch
     * @param totalInputSizeBytes Total size of batch input data in bytes
     */
    public void recordBatchInference(long inferenceTimeMs, int batchSize, int totalInputSizeBytes) {
        long timestamp = System.currentTimeMillis();
        
        // Record one sample representing the entire batch
        InferenceSample sample = new InferenceSample(
            timestamp,
            inferenceTimeMs,
            totalInputSizeBytes,
            0, // output size not available yet
            0  // memory usage not available yet
        );
        
        samples.offer(sample);
        
        // Maintain max samples
        while (samples.size() > MAX_SAMPLES) {
            samples.poll();
        }
        
        // Update totals - count each item in batch as an inference
        totalInferences.addAndGet(batchSize);
        totalInferenceTimeMs.addAndGet(inferenceTimeMs);
        totalDataProcessed.addAndGet(totalInputSizeBytes);
    }
    
    /**
     * Get performance statistics
     * @return Current performance statistics
     */
    @NonNull
    public PerformanceStats getStats() {
        List<InferenceSample> sampleList = new ArrayList<>(samples);
        
        if (sampleList.isEmpty()) {
            return new PerformanceStats();
        }
        
        // Calculate statistics
        long totalTime = 0;
        long minTime = Long.MAX_VALUE;
        long maxTime = 0;
        long totalMemory = 0;
        
        List<Long> times = new ArrayList<>();
        
        for (InferenceSample sample : sampleList) {
            totalTime += sample.inferenceTimeMs;
            minTime = Math.min(minTime, sample.inferenceTimeMs);
            maxTime = Math.max(maxTime, sample.inferenceTimeMs);
            totalMemory += sample.memoryUsedBytes;
            times.add(sample.inferenceTimeMs);
        }
        
        double avgTime = (double) totalTime / sampleList.size();
        
        // Calculate percentiles
        Collections.sort(times);
        long p50 = getPercentile(times, 50);
        long p90 = getPercentile(times, 90);
        long p95 = getPercentile(times, 95);
        long p99 = getPercentile(times, 99);
        
        // Calculate throughput
        long elapsedTimeMs = System.currentTimeMillis() - startTimeMs;
        double inferencePerSecond = 0;
        if (elapsedTimeMs > 0) {
            inferencePerSecond = (totalInferences.get() * 1000.0) / elapsedTimeMs;
        }
        
        double mbPerSecond = 0;
        if (elapsedTimeMs > 0) {
            mbPerSecond = (totalDataProcessed.get() / (1024.0 * 1024.0)) / 
                         (elapsedTimeMs / 1000.0);
        }
        
        return new PerformanceStats(
            totalInferences.get(),
            avgTime,
            minTime,
            maxTime,
            p50,
            p90,
            p95,
            p99,
            inferencePerSecond,
            mbPerSecond,
            totalMemory / Math.max(1, sampleList.size())
        );
    }
    
    /**
     * Reset all statistics
     */
    public void reset() {
        samples.clear();
        totalInferences.set(0);
        totalInferenceTimeMs.set(0);
        totalDataProcessed.set(0);
        startTimeMs = System.currentTimeMillis();
    }
    
    private long getPercentile(List<Long> sortedValues, int percentile) {
        if (sortedValues.isEmpty()) {
            return 0;
        }
        
        int index = (int) Math.ceil((percentile / 100.0) * sortedValues.size()) - 1;
        index = Math.max(0, Math.min(index, sortedValues.size() - 1));
        return sortedValues.get(index);
    }
    
    /**
     * Performance statistics
     */
    public static class PerformanceStats {
        private final int totalInferences;
        private final double avgInferenceTimeMs;
        private final long minInferenceTimeMs;
        private final long maxInferenceTimeMs;
        private final long p50InferenceTimeMs;
        private final long p90InferenceTimeMs;
        private final long p95InferenceTimeMs;
        private final long p99InferenceTimeMs;
        private final double inferencesPerSecond;
        private final double mbPerSecond;
        private final long avgMemoryBytes;
        
        PerformanceStats() {
            this(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }
        
        PerformanceStats(int totalInferences, double avgInferenceTimeMs,
                        long minInferenceTimeMs, long maxInferenceTimeMs,
                        long p50InferenceTimeMs, long p90InferenceTimeMs,
                        long p95InferenceTimeMs, long p99InferenceTimeMs,
                        double inferencesPerSecond, double mbPerSecond,
                        long avgMemoryBytes) {
            this.totalInferences = totalInferences;
            this.avgInferenceTimeMs = avgInferenceTimeMs;
            this.minInferenceTimeMs = minInferenceTimeMs;
            this.maxInferenceTimeMs = maxInferenceTimeMs;
            this.p50InferenceTimeMs = p50InferenceTimeMs;
            this.p90InferenceTimeMs = p90InferenceTimeMs;
            this.p95InferenceTimeMs = p95InferenceTimeMs;
            this.p99InferenceTimeMs = p99InferenceTimeMs;
            this.inferencesPerSecond = inferencesPerSecond;
            this.mbPerSecond = mbPerSecond;
            this.avgMemoryBytes = avgMemoryBytes;
        }
        
        public int getTotalInferences() {
            return totalInferences;
        }
        
        public double getAvgInferenceTimeMs() {
            return avgInferenceTimeMs;
        }
        
        public long getMinInferenceTimeMs() {
            return minInferenceTimeMs;
        }
        
        public long getMaxInferenceTimeMs() {
            return maxInferenceTimeMs;
        }
        
        public long getP50InferenceTimeMs() {
            return p50InferenceTimeMs;
        }
        
        public long getP90InferenceTimeMs() {
            return p90InferenceTimeMs;
        }
        
        public long getP95InferenceTimeMs() {
            return p95InferenceTimeMs;
        }
        
        public long getP99InferenceTimeMs() {
            return p99InferenceTimeMs;
        }
        
        public double getInferencesPerSecond() {
            return inferencesPerSecond;
        }
        
        public double getMbPerSecond() {
            return mbPerSecond;
        }
        
        public long getAvgMemoryMB() {
            return avgMemoryBytes / (1024 * 1024);
        }
        
        /**
         * Get summary string
         * @return Human-readable summary
         */
        @NonNull
        public String getSummary() {
            StringBuilder sb = new StringBuilder();
            sb.append("Performance Statistics:\n");
            sb.append(String.format("  Total inferences: %d\n", totalInferences));
            sb.append(String.format("  Average time: %.2f ms\n", avgInferenceTimeMs));
            sb.append(String.format("  Min/Max: %d/%d ms\n", minInferenceTimeMs, maxInferenceTimeMs));
            sb.append(String.format("  Percentiles (50/90/95/99): %d/%d/%d/%d ms\n",
                p50InferenceTimeMs, p90InferenceTimeMs, p95InferenceTimeMs, p99InferenceTimeMs));
            sb.append(String.format("  Throughput: %.2f inferences/sec\n", inferencesPerSecond));
            sb.append(String.format("  Data rate: %.2f MB/sec\n", mbPerSecond));
            sb.append(String.format("  Avg memory: %d MB\n", getAvgMemoryMB()));
            return sb.toString();
        }
        
        @Override
        public String toString() {
            return getSummary();
        }
    }
}