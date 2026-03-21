/**
 * TrustformeRS Image Classification Examples
 * 
 * Demonstrates image classification capabilities using TrustformeRS JavaScript bindings.
 * Shows various image processing scenarios including file upload, base64 images,
 * batch processing, and multimodal inference.
 */

import { TrustformersJS, ImageClassifier, MultiModalProcessor } from '../src/index.js';

/**
 * Basic image classification with file input
 */
export async function basicImageClassification() {
    console.log('\n=== Basic Image Classification ===');
    
    try {
        // Initialize image classifier
        const classifier = new ImageClassifier({
            model: 'resnet50',
            device: 'auto',
            precision: 'fp16'
        });
        
        // Load pre-trained model
        await classifier.loadModel();
        console.log('✓ Image classification model loaded');
        
        // Example with base64 image data (1x1 red pixel for demo)
        const redPixelBase64 = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
        
        // Classify image
        const results = await classifier.classify({
            image: redPixelBase64,
            topK: 5,
            threshold: 0.1
        });
        
        console.log('Image classification results:');
        results.predictions.forEach((pred, index) => {
            console.log(`  ${index + 1}. ${pred.label}: ${(pred.confidence * 100).toFixed(2)}%`);
        });
        
        // Performance metrics
        console.log(`\nPerformance:`);
        console.log(`  Inference time: ${results.inferenceTime}ms`);
        console.log(`  Preprocessing time: ${results.preprocessingTime}ms`);
        console.log(`  Total time: ${results.totalTime}ms`);
        
        return results;
        
    } catch (error) {
        console.error('Error in basic image classification:', error);
        throw error;
    }
}

/**
 * Batch image classification
 */
export async function batchImageClassification() {
    console.log('\n=== Batch Image Classification ===');
    
    try {
        const classifier = new ImageClassifier({
            model: 'efficientnet_b0',
            device: 'auto',
            batchSize: 4
        });
        
        await classifier.loadModel();
        
        // Create sample images (different colored pixels)
        const images = [
            'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', // Red
            'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQIHWNgwAcAABQAAY1h0Y4AAAAASUVORK5CYII=', // Green
            'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQIHWMgAQwAAAMAARIAMEEAAAAASUVORK5CYII=', // Blue
            'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADklEQVR42mP8/5+BgQEACQgB/xf3kX8AAAAASUVORK5CYII='  // White
        ];
        
        // Classify batch of images
        const results = await classifier.classifyBatch({
            images: images,
            topK: 3,
            threshold: 0.05
        });
        
        console.log(`Processed ${results.length} images in batch:`);
        results.forEach((result, index) => {
            console.log(`\nImage ${index + 1}:`);
            result.predictions.forEach((pred, predIndex) => {
                console.log(`  ${predIndex + 1}. ${pred.label}: ${(pred.confidence * 100).toFixed(2)}%`);
            });
        });
        
        // Batch performance metrics
        const totalTime = results.reduce((sum, r) => sum + r.totalTime, 0);
        const avgTime = totalTime / results.length;
        console.log(`\nBatch Performance:`);
        console.log(`  Total batch time: ${totalTime}ms`);
        console.log(`  Average per image: ${avgTime.toFixed(2)}ms`);
        console.log(`  Throughput: ${(1000 / avgTime).toFixed(2)} images/sec`);
        
        return results;
        
    } catch (error) {
        console.error('Error in batch image classification:', error);
        throw error;
    }
}

/**
 * Advanced image classification with preprocessing options
 */
export async function advancedImageClassification() {
    console.log('\n=== Advanced Image Classification with Preprocessing ===');
    
    try {
        const classifier = new ImageClassifier({
            model: 'vit_base_patch16_224',
            device: 'auto',
            preprocessing: {
                resize: [224, 224],
                normalize: {
                    mean: [0.485, 0.456, 0.406],
                    std: [0.229, 0.224, 0.225]
                },
                centerCrop: true,
                augmentation: {
                    horizontalFlip: 0.5,
                    randomRotation: 15,
                    colorJitter: {
                        brightness: 0.1,
                        contrast: 0.1,
                        saturation: 0.1,
                        hue: 0.05
                    }
                }
            }
        });
        
        await classifier.loadModel();
        console.log('✓ Vision Transformer model loaded with preprocessing');
        
        // Create a more complex test image (checkered pattern)
        const canvas = createTestCanvas(256, 256, 'checkered');
        const imageData = canvas.toDataURL('image/png');
        
        // Classify with detailed preprocessing
        const results = await classifier.classify({
            image: imageData,
            topK: 10,
            threshold: 0.01,
            returnFeatures: true,
            returnAttention: true,
            preprocessingDetails: true
        });
        
        console.log('Advanced classification results:');
        results.predictions.forEach((pred, index) => {
            console.log(`  ${index + 1}. ${pred.label}: ${(pred.confidence * 100).toFixed(2)}%`);
        });
        
        // Feature vector information
        if (results.features) {
            console.log(`\nFeature Vector:`);
            console.log(`  Dimensions: ${results.features.shape}`);
            console.log(`  L2 Norm: ${results.features.l2Norm.toFixed(4)}`);
        }
        
        // Attention maps (for Vision Transformers)
        if (results.attention) {
            console.log(`\nAttention Maps:`);
            console.log(`  Number of heads: ${results.attention.numHeads}`);
            console.log(`  Attention shape: ${results.attention.shape}`);
            console.log(`  Max attention: ${results.attention.maxValue.toFixed(4)}`);
        }
        
        // Preprocessing details
        if (results.preprocessingDetails) {
            console.log(`\nPreprocessing Applied:`);
            results.preprocessingDetails.forEach(step => {
                console.log(`  ${step.operation}: ${step.parameters}`);
            });
        }
        
        return results;
        
    } catch (error) {
        console.error('Error in advanced image classification:', error);
        throw error;
    }
}

/**
 * Multimodal classification (image + text)
 */
export async function multimodalImageTextClassification() {
    console.log('\n=== Multimodal Image + Text Classification ===');
    
    try {
        const processor = new MultiModalProcessor({
            model: 'clip_vit_b32',
            device: 'auto',
            modalities: ['image', 'text']
        });
        
        await processor.loadModel();
        console.log('✓ CLIP multimodal model loaded');
        
        // Test image
        const testImage = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
        
        // Text descriptions to match against
        const textCandidates = [
            'a red colored object',
            'a blue colored object', 
            'a green colored object',
            'a white colored object',
            'a black colored object',
            'a colorful pattern',
            'a geometric shape',
            'a natural landscape',
            'an animal',
            'a vehicle'
        ];
        
        // Perform multimodal matching
        const results = await processor.matchImageText({
            image: testImage,
            texts: textCandidates,
            temperature: 0.07
        });
        
        console.log('Image-Text matching results:');
        results.matches.forEach((match, index) => {
            console.log(`  ${index + 1}. "${match.text}": ${(match.similarity * 100).toFixed(2)}%`);
        });
        
        // Zero-shot classification
        const zeroShotResults = await processor.zeroShotClassify({
            image: testImage,
            labels: ['red object', 'blue object', 'green object', 'abstract art', 'photograph'],
            template: 'this is a photo of {}'
        });
        
        console.log('\nZero-shot classification:');
        zeroShotResults.predictions.forEach((pred, index) => {
            console.log(`  ${index + 1}. ${pred.label}: ${(pred.confidence * 100).toFixed(2)}%`);
        });
        
        return { matches: results, zeroShot: zeroShotResults };
        
    } catch (error) {
        console.error('Error in multimodal classification:', error);
        throw error;
    }
}

/**
 * Real-time image classification from camera/video
 */
export async function realTimeImageClassification() {
    console.log('\n=== Real-time Image Classification Setup ===');
    
    try {
        const classifier = new ImageClassifier({
            model: 'mobilenet_v3_large',
            device: 'gpu',
            optimizedForSpeed: true,
            precision: 'fp16'
        });
        
        await classifier.loadModel();
        console.log('✓ MobileNet model loaded for real-time inference');
        
        // Simulate real-time processing setup
        const streamProcessor = classifier.createStreamProcessor({
            targetFPS: 30,
            skipFrames: 2,
            bufferSize: 3,
            confidenceThreshold: 0.3
        });
        
        console.log('Real-time stream processor configured:');
        console.log(`  Target FPS: ${streamProcessor.targetFPS}`);
        console.log(`  Frame skipping: ${streamProcessor.skipFrames}`);
        console.log(`  Buffer size: ${streamProcessor.bufferSize}`);
        
        // Simulate processing a few frames
        const simulatedFrames = generateSimulatedFrames(5);
        
        for (const [index, frame] of simulatedFrames.entries()) {
            console.log(`\nProcessing frame ${index + 1}:`);
            
            const result = await streamProcessor.processFrame(frame);
            
            if (result.predictions.length > 0) {
                const topPred = result.predictions[0];
                console.log(`  Top prediction: ${topPred.label} (${(topPred.confidence * 100).toFixed(1)}%)`);
                console.log(`  Processing time: ${result.processingTime}ms`);
                console.log(`  FPS: ${(1000 / result.processingTime).toFixed(1)}`);
            } else {
                console.log('  No confident predictions');
            }
        }
        
        // Stream statistics
        const stats = streamProcessor.getStatistics();
        console.log('\nStream Processing Statistics:');
        console.log(`  Frames processed: ${stats.framesProcessed}`);
        console.log(`  Frames dropped: ${stats.framesDropped}`);
        console.log(`  Average FPS: ${stats.averageFPS.toFixed(2)}`);
        console.log(`  Average latency: ${stats.averageLatency.toFixed(2)}ms`);
        
        return streamProcessor;
        
    } catch (error) {
        console.error('Error in real-time image classification:', error);
        throw error;
    }
}

/**
 * Custom model image classification
 */
export async function customModelImageClassification() {
    console.log('\n=== Custom Model Image Classification ===');
    
    try {
        // Load a custom trained model
        const classifier = new ImageClassifier({
            modelPath: './models/custom_food_classifier.trustformers',
            configPath: './models/custom_food_classifier_config.json',
            device: 'auto',
            customLabels: [
                'pizza', 'burger', 'sushi', 'pasta', 'salad',
                'cake', 'ice cream', 'fruit', 'soup', 'sandwich'
            ]
        });
        
        await classifier.loadModel();
        console.log('✓ Custom food classification model loaded');
        
        // Test with sample food image
        const foodImage = createTestCanvas(224, 224, 'food_pattern');
        const imageData = foodImage.toDataURL('image/jpeg', 0.95);
        
        const results = await classifier.classify({
            image: imageData,
            topK: 5,
            threshold: 0.1,
            returnGradCam: true // For explainability
        });
        
        console.log('Custom model classification results:');
        results.predictions.forEach((pred, index) => {
            console.log(`  ${index + 1}. ${pred.label}: ${(pred.confidence * 100).toFixed(2)}%`);
        });
        
        // GradCAM visualization (if available)
        if (results.gradCam) {
            console.log('\nGradCAM Heatmap generated:');
            console.log(`  Heatmap size: ${results.gradCam.width}x${results.gradCam.height}`);
            console.log(`  Max activation: ${results.gradCam.maxActivation.toFixed(4)}`);
            // In a real implementation, this would be visualized
        }
        
        return results;
        
    } catch (error) {
        console.error('Error in custom model classification:', error);
        // Fallback to standard model if custom model not available
        console.log('Falling back to standard ImageNet model...');
        return await basicImageClassification();
    }
}

/**
 * Performance benchmarking for image classification
 */
export async function benchmarkImageClassification() {
    console.log('\n=== Image Classification Performance Benchmark ===');
    
    const models = [
        { name: 'ResNet-50', model: 'resnet50', expectedFPS: 100 },
        { name: 'EfficientNet-B0', model: 'efficientnet_b0', expectedFPS: 120 },
        { name: 'MobileNet-V3', model: 'mobilenet_v3_large', expectedFPS: 200 },
        { name: 'Vision Transformer', model: 'vit_base_patch16_224', expectedFPS: 80 }
    ];
    
    const benchmarkResults = [];
    const testImage = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    
    for (const modelConfig of models) {
        console.log(`\nBenchmarking ${modelConfig.name}...`);
        
        try {
            const classifier = new ImageClassifier({
                model: modelConfig.model,
                device: 'auto',
                precision: 'fp16'
            });
            
            const loadStart = performance.now();
            await classifier.loadModel();
            const loadTime = performance.now() - loadStart;
            
            // Warmup runs
            for (let i = 0; i < 3; i++) {
                await classifier.classify({ image: testImage, topK: 1 });
            }
            
            // Benchmark runs
            const numRuns = 10;
            const times = [];
            
            for (let i = 0; i < numRuns; i++) {
                const start = performance.now();
                await classifier.classify({ image: testImage, topK: 5 });
                times.push(performance.now() - start);
            }
            
            const avgTime = times.reduce((a, b) => a + b) / times.length;
            const fps = 1000 / avgTime;
            const minTime = Math.min(...times);
            const maxTime = Math.max(...times);
            
            const result = {
                model: modelConfig.name,
                loadTime: loadTime.toFixed(2),
                avgInferenceTime: avgTime.toFixed(2),
                fps: fps.toFixed(2),
                minTime: minTime.toFixed(2),
                maxTime: maxTime.toFixed(2),
                expectedFPS: modelConfig.expectedFPS,
                performanceRatio: (fps / modelConfig.expectedFPS).toFixed(2)
            };
            
            benchmarkResults.push(result);
            
            console.log(`  Load time: ${result.loadTime}ms`);
            console.log(`  Avg inference: ${result.avgInferenceTime}ms`);
            console.log(`  FPS: ${result.fps} (expected: ${result.expectedFPS})`);
            console.log(`  Performance ratio: ${result.performanceRatio}x`);
            
        } catch (error) {
            console.error(`  Failed to benchmark ${modelConfig.name}:`, error.message);
            benchmarkResults.push({
                model: modelConfig.name,
                error: error.message
            });
        }
    }
    
    // Summary
    console.log('\n=== Benchmark Summary ===');
    console.log('Model                | Load Time | Avg Time | FPS   | Ratio');
    console.log('---------------------|-----------|----------|-------|-------');
    benchmarkResults.forEach(result => {
        if (result.error) {
            console.log(`${result.model.padEnd(20)} | ERROR: ${result.error}`);
        } else {
            console.log(
                `${result.model.padEnd(20)} | ` +
                `${result.loadTime.padEnd(9)} | ` +
                `${result.avgInferenceTime.padEnd(8)} | ` +
                `${result.fps.padEnd(5)} | ` +
                `${result.performanceRatio}x`
            );
        }
    });
    
    return benchmarkResults;
}

// Helper functions

function createTestCanvas(width, height, pattern = 'solid') {
    // In browser environment, this would create an actual canvas
    // For Node.js, this is a mock implementation
    return {
        width,
        height,
        pattern,
        toDataURL: (format = 'image/png', quality = 1.0) => {
            // Return different base64 data based on pattern
            switch (pattern) {
                case 'checkered':
                    return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
                case 'food_pattern':
                    return 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=';
                default:
                    return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
            }
        }
    };
}

function generateSimulatedFrames(count) {
    const frames = [];
    for (let i = 0; i < count; i++) {
        // Generate different test patterns
        const patterns = ['solid', 'checkered', 'gradient', 'noise', 'pattern'];
        const pattern = patterns[i % patterns.length];
        const canvas = createTestCanvas(224, 224, pattern);
        frames.push(canvas.toDataURL('image/png'));
    }
    return frames;
}

// Export main demo function
export async function runAllImageClassificationExamples() {
    console.log('🖼️  TrustformeRS Image Classification Examples');
    console.log('='.repeat(50));
    
    try {
        // Run all examples
        await basicImageClassification();
        await batchImageClassification();
        await advancedImageClassification();
        await multimodalImageTextClassification();
        await realTimeImageClassification();
        await customModelImageClassification();
        
        // Performance benchmark
        const benchmarkResults = await benchmarkImageClassification();
        
        console.log('\n✅ All image classification examples completed successfully!');
        console.log('📊 Check the benchmark results above for performance analysis.');
        
        return {
            status: 'success',
            benchmarks: benchmarkResults,
            timestamp: new Date().toISOString()
        };
        
    } catch (error) {
        console.error('\n❌ Error running image classification examples:', error);
        return {
            status: 'error',
            error: error.message,
            timestamp: new Date().toISOString()
        };
    }
}

// Auto-run if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    runAllImageClassificationExamples()
        .then(result => {
            console.log('\nFinal result:', result);
            process.exit(result.status === 'success' ? 0 : 1);
        })
        .catch(error => {
            console.error('Fatal error:', error);
            process.exit(1);
        });
}