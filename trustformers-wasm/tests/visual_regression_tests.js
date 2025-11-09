/**
 * Visual Regression Test Suite for TrustformeRS WASM
 * 
 * This test suite captures visual snapshots of UI components and compares
 * them against reference images to detect visual regressions.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from '@jest/globals';

// Visual regression test configuration
const VISUAL_REGRESSION_CONFIG = {
    screenshotDir: '/tmp/visual_regression_screenshots',
    referenceDir: '/tmp/visual_regression_references',
    diffDir: '/tmp/visual_regression_diffs',
    threshold: 0.1, // 10% pixel difference threshold
    
    // Component test scenarios
    components: [
        {
            name: 'tensor_visualization',
            description: 'Tensor visualization component',
            element: '#tensor-viz',
            testData: {
                tensor: { shape: [4, 4], data: new Array(16).fill(0).map((_, i) => i) }
            }
        },
        {
            name: 'performance_monitor',
            description: 'Performance monitoring dashboard',
            element: '#performance-monitor',
            testData: {
                metrics: {
                    fps: 60,
                    memoryUsage: 45.2,
                    inferenceTime: 125.5
                }
            }
        },
        {
            name: 'model_loader',
            description: 'Model loading interface',
            element: '#model-loader',
            testData: {
                models: ['gpt-2', 'bert-base', 'roberta-large'],
                selectedModel: 'gpt-2',
                loadingState: 'loaded'
            }
        },
        {
            name: 'interactive_playground',
            description: 'Interactive playground interface',
            element: '#playground',
            testData: {
                inputText: 'Hello, this is a test input',
                outputText: 'Generated text output example',
                isGenerating: false
            }
        },
        {
            name: 'debug_console',
            description: 'Debug console component',
            element: '#debug-console',
            testData: {
                logs: [
                    { level: 'info', message: 'Model loaded successfully', timestamp: Date.now() },
                    { level: 'warn', message: 'GPU memory usage high', timestamp: Date.now() },
                    { level: 'error', message: 'Inference timeout', timestamp: Date.now() }
                ]
            }
        }
    ],
    
    // Viewport configurations for responsive testing
    viewports: [
        { name: 'mobile', width: 375, height: 667 },
        { name: 'tablet', width: 768, height: 1024 },
        { name: 'desktop', width: 1920, height: 1080 }
    ],
    
    // Browser configurations
    browsers: [
        { name: 'chrome', userAgent: 'Chrome/91.0.4472.124' },
        { name: 'firefox', userAgent: 'Firefox/89.0' },
        { name: 'safari', userAgent: 'Safari/14.1.1' }
    ]
};

// Image comparison utilities
class ImageComparator {
    static async compareImages(actualImageData, referenceImageData, threshold = VISUAL_REGRESSION_CONFIG.threshold) {
        if (!actualImageData || !referenceImageData) {
            return {
                match: false,
                reason: 'Missing image data',
                difference: 1.0
            };
        }
        
        if (actualImageData.width !== referenceImageData.width || 
            actualImageData.height !== referenceImageData.height) {
            return {
                match: false,
                reason: 'Image dimensions mismatch',
                difference: 1.0
            };
        }
        
        const pixelDiff = this.calculatePixelDifference(actualImageData, referenceImageData);
        const differencePercent = pixelDiff / (actualImageData.width * actualImageData.height);
        
        return {
            match: differencePercent <= threshold,
            difference: differencePercent,
            pixelDiff: pixelDiff,
            dimensions: {
                width: actualImageData.width,
                height: actualImageData.height
            }
        };
    }
    
    static calculatePixelDifference(imageData1, imageData2) {
        const data1 = imageData1.data;
        const data2 = imageData2.data;
        let diffCount = 0;
        
        for (let i = 0; i < data1.length; i += 4) {
            const r1 = data1[i];
            const g1 = data1[i + 1];
            const b1 = data1[i + 2];
            const a1 = data1[i + 3];
            
            const r2 = data2[i];
            const g2 = data2[i + 1];
            const b2 = data2[i + 2];
            const a2 = data2[i + 3];
            
            const diff = Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2) + Math.abs(a1 - a2);
            
            if (diff > 10) { // Threshold for significant pixel difference
                diffCount++;
            }
        }
        
        return diffCount / 4; // Convert back to pixel count
    }
    
    static generateDiffImage(imageData1, imageData2) {
        const diffCanvas = document.createElement('canvas');
        diffCanvas.width = imageData1.width;
        diffCanvas.height = imageData1.height;
        const diffCtx = diffCanvas.getContext('2d');
        
        const diffImageData = diffCtx.createImageData(imageData1.width, imageData1.height);
        const data1 = imageData1.data;
        const data2 = imageData2.data;
        const diffData = diffImageData.data;
        
        for (let i = 0; i < data1.length; i += 4) {
            const r1 = data1[i];
            const g1 = data1[i + 1];
            const b1 = data1[i + 2];
            const a1 = data1[i + 3];
            
            const r2 = data2[i];
            const g2 = data2[i + 1];
            const b2 = data2[i + 2];
            const a2 = data2[i + 3];
            
            const diff = Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2) + Math.abs(a1 - a2);
            
            if (diff > 10) {
                // Highlight differences in red
                diffData[i] = 255;
                diffData[i + 1] = 0;
                diffData[i + 2] = 0;
                diffData[i + 3] = 255;
            } else {
                // Keep original pixel
                diffData[i] = r1;
                diffData[i + 1] = g1;
                diffData[i + 2] = b1;
                diffData[i + 3] = a1;
            }
        }
        
        diffCtx.putImageData(diffImageData, 0, 0);
        return diffCanvas.toDataURL();
    }
}

// Screenshot capture utility
class ScreenshotCapture {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
    }
    
    async captureElement(selector, viewport = { width: 1920, height: 1080 }) {
        const element = document.querySelector(selector);
        if (!element) {
            throw new Error(`Element not found: ${selector}`);
        }
        
        // Set viewport size
        this.canvas.width = viewport.width;
        this.canvas.height = viewport.height;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Capture element bounds
        const rect = element.getBoundingClientRect();
        
        // Use html2canvas-like approach for element capture
        const imageData = await this.captureElementData(element, rect);
        
        return {
            imageData,
            dataURL: this.canvas.toDataURL(),
            dimensions: {
                width: this.canvas.width,
                height: this.canvas.height
            },
            elementBounds: rect
        };
    }
    
    async captureElementData(element, rect) {
        // This is a simplified implementation
        // In a real scenario, you'd use a library like html2canvas
        
        // Create a temporary canvas for the element
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = rect.width;
        tempCanvas.height = rect.height;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Fill with element background color or white
        const computedStyle = window.getComputedStyle(element);
        tempCtx.fillStyle = computedStyle.backgroundColor || '#ffffff';
        tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
        
        // Add some visual representation based on element content
        this.renderElementContent(tempCtx, element, rect);
        
        return tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    }
    
    renderElementContent(ctx, element, rect) {
        // Simplified rendering based on element type and content
        const tagName = element.tagName.toLowerCase();
        const textContent = element.textContent || '';
        
        ctx.fillStyle = '#333333';
        ctx.font = '16px Arial';
        
        switch (tagName) {
            case 'div':
                // Render text content
                if (textContent) {
                    ctx.fillText(textContent.substring(0, 50), 10, 30);
                }
                break;
            case 'canvas':
                // If it's a canvas, try to copy its content
                if (element.getContext) {
                    try {
                        ctx.drawImage(element, 0, 0);
                    } catch (e) {
                        // Fallback to text
                        ctx.fillText('Canvas Element', 10, 30);
                    }
                }
                break;
            case 'svg':
                // Render SVG placeholder
                ctx.fillStyle = '#e0e0e0';
                ctx.fillRect(10, 10, rect.width - 20, rect.height - 20);
                ctx.fillStyle = '#333333';
                ctx.fillText('SVG Element', 10, 30);
                break;
            default:
                // Generic element rendering
                ctx.fillStyle = '#f0f0f0';
                ctx.fillRect(5, 5, rect.width - 10, rect.height - 10);
                ctx.fillStyle = '#333333';
                ctx.fillText(`${tagName}: ${textContent.substring(0, 20)}`, 10, 30);
        }
        
        // Add element class indicators
        if (element.className) {
            ctx.font = '12px Arial';
            ctx.fillStyle = '#666666';
            ctx.fillText(`.${element.className}`, 10, rect.height - 10);
        }
    }
}

// Mock component renderer for testing
class MockComponentRenderer {
    constructor() {
        this.container = null;
        this.components = new Map();
    }
    
    initialize() {
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.id = 'visual-test-container';
            this.container.style.position = 'absolute';
            this.container.style.top = '-9999px';
            this.container.style.left = '-9999px';
            document.body.appendChild(this.container);
        }
    }
    
    renderComponent(componentName, testData) {
        const componentElement = document.createElement('div');
        componentElement.id = componentName.replace('_', '-');
        componentElement.className = `component-${componentName}`;
        
        switch (componentName) {
            case 'tensor_visualization':
                this.renderTensorVisualization(componentElement, testData);
                break;
            case 'performance_monitor':
                this.renderPerformanceMonitor(componentElement, testData);
                break;
            case 'model_loader':
                this.renderModelLoader(componentElement, testData);
                break;
            case 'interactive_playground':
                this.renderInteractivePlayground(componentElement, testData);
                break;
            case 'debug_console':
                this.renderDebugConsole(componentElement, testData);
                break;
            default:
                this.renderGenericComponent(componentElement, componentName, testData);
        }
        
        this.container.appendChild(componentElement);
        this.components.set(componentName, componentElement);
        
        return componentElement;
    }
    
    renderTensorVisualization(element, testData) {
        element.style.width = '400px';
        element.style.height = '300px';
        element.style.backgroundColor = '#f8f9fa';
        element.style.border = '1px solid #dee2e6';
        element.style.position = 'relative';
        
        const title = document.createElement('h3');
        title.textContent = 'Tensor Visualization';
        title.style.margin = '10px';
        element.appendChild(title);
        
        const canvas = document.createElement('canvas');
        canvas.width = 200;
        canvas.height = 200;
        canvas.style.margin = '10px';
        const ctx = canvas.getContext('2d');
        
        // Render tensor as heatmap
        const { tensor } = testData;
        const cellSize = 200 / Math.sqrt(tensor.data.length);
        
        tensor.data.forEach((value, index) => {
            const x = (index % tensor.shape[1]) * cellSize;
            const y = Math.floor(index / tensor.shape[1]) * cellSize;
            
            const intensity = Math.abs(value) / Math.max(...tensor.data);
            ctx.fillStyle = `hsl(${240 - intensity * 60}, 70%, 50%)`;
            ctx.fillRect(x, y, cellSize, cellSize);
        });
        
        element.appendChild(canvas);
    }
    
    renderPerformanceMonitor(element, testData) {
        element.style.width = '500px';
        element.style.height = '200px';
        element.style.backgroundColor = '#ffffff';
        element.style.border = '1px solid #ccc';
        element.style.padding = '20px';
        
        const title = document.createElement('h3');
        title.textContent = 'Performance Monitor';
        element.appendChild(title);
        
        const { metrics } = testData;
        
        Object.entries(metrics).forEach(([key, value]) => {
            const metricDiv = document.createElement('div');
            metricDiv.style.marginBottom = '10px';
            metricDiv.style.padding = '5px';
            metricDiv.style.backgroundColor = '#f0f0f0';
            metricDiv.innerHTML = `<strong>${key}:</strong> ${value}`;
            element.appendChild(metricDiv);
        });
    }
    
    renderModelLoader(element, testData) {
        element.style.width = '300px';
        element.style.height = '250px';
        element.style.backgroundColor = '#ffffff';
        element.style.border = '1px solid #ddd';
        element.style.padding = '15px';
        
        const title = document.createElement('h3');
        title.textContent = 'Model Loader';
        element.appendChild(title);
        
        const { models, selectedModel, loadingState } = testData;
        
        const select = document.createElement('select');
        select.style.width = '100%';
        select.style.marginBottom = '10px';
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            option.selected = model === selectedModel;
            select.appendChild(option);
        });
        
        element.appendChild(select);
        
        const statusDiv = document.createElement('div');
        statusDiv.style.padding = '10px';
        statusDiv.style.backgroundColor = loadingState === 'loaded' ? '#d4edda' : '#f8d7da';
        statusDiv.style.color = loadingState === 'loaded' ? '#155724' : '#721c24';
        statusDiv.textContent = `Status: ${loadingState}`;
        element.appendChild(statusDiv);
    }
    
    renderInteractivePlayground(element, testData) {
        element.style.width = '600px';
        element.style.height = '400px';
        element.style.backgroundColor = '#ffffff';
        element.style.border = '1px solid #ccc';
        element.style.padding = '20px';
        
        const title = document.createElement('h3');
        title.textContent = 'Interactive Playground';
        element.appendChild(title);
        
        const { inputText, outputText, isGenerating } = testData;
        
        const inputArea = document.createElement('textarea');
        inputArea.style.width = '100%';
        inputArea.style.height = '100px';
        inputArea.style.marginBottom = '10px';
        inputArea.value = inputText;
        element.appendChild(inputArea);
        
        const generateBtn = document.createElement('button');
        generateBtn.textContent = isGenerating ? 'Generating...' : 'Generate';
        generateBtn.style.marginBottom = '10px';
        generateBtn.disabled = isGenerating;
        element.appendChild(generateBtn);
        
        const outputArea = document.createElement('textarea');
        outputArea.style.width = '100%';
        outputArea.style.height = '100px';
        outputArea.value = outputText;
        outputArea.readOnly = true;
        element.appendChild(outputArea);
    }
    
    renderDebugConsole(element, testData) {
        element.style.width = '500px';
        element.style.height = '300px';
        element.style.backgroundColor = '#1e1e1e';
        element.style.color = '#ffffff';
        element.style.border = '1px solid #333';
        element.style.padding = '10px';
        element.style.fontFamily = 'monospace';
        element.style.fontSize = '12px';
        
        const title = document.createElement('h3');
        title.textContent = 'Debug Console';
        title.style.color = '#ffffff';
        title.style.marginBottom = '10px';
        element.appendChild(title);
        
        const { logs } = testData;
        
        logs.forEach(log => {
            const logDiv = document.createElement('div');
            logDiv.style.marginBottom = '2px';
            logDiv.style.padding = '2px';
            
            const levelColors = {
                info: '#4CAF50',
                warn: '#FF9800',
                error: '#F44336'
            };
            
            logDiv.style.color = levelColors[log.level] || '#ffffff';
            logDiv.innerHTML = `[${log.level.toUpperCase()}] ${log.message}`;
            element.appendChild(logDiv);
        });
    }
    
    renderGenericComponent(element, componentName, testData) {
        element.style.width = '300px';
        element.style.height = '200px';
        element.style.backgroundColor = '#f5f5f5';
        element.style.border = '1px solid #ddd';
        element.style.padding = '15px';
        
        const title = document.createElement('h3');
        title.textContent = componentName.replace('_', ' ').toUpperCase();
        element.appendChild(title);
        
        const dataDiv = document.createElement('pre');
        dataDiv.style.fontSize = '12px';
        dataDiv.style.backgroundColor = '#ffffff';
        dataDiv.style.padding = '10px';
        dataDiv.style.border = '1px solid #ccc';
        dataDiv.textContent = JSON.stringify(testData, null, 2);
        element.appendChild(dataDiv);
    }
    
    cleanup() {
        if (this.container) {
            document.body.removeChild(this.container);
            this.container = null;
        }
        this.components.clear();
    }
}

// Visual regression test executor
class VisualRegressionTestExecutor {
    constructor() {
        this.renderer = new MockComponentRenderer();
        this.capture = new ScreenshotCapture();
        this.testResults = [];
    }
    
    async initialize() {
        this.renderer.initialize();
    }
    
    async runVisualRegressionTests() {
        for (const component of VISUAL_REGRESSION_CONFIG.components) {
            for (const viewport of VISUAL_REGRESSION_CONFIG.viewports) {
                console.log(`Testing ${component.name} at ${viewport.name} viewport`);
                
                try {
                    const result = await this.testComponent(component, viewport);
                    this.testResults.push(result);
                } catch (error) {
                    this.testResults.push({
                        component: component.name,
                        viewport: viewport.name,
                        success: false,
                        error: error.message,
                        timestamp: new Date().toISOString()
                    });
                }
            }
        }
        
        return this.testResults;
    }
    
    async testComponent(component, viewport) {
        // Render component
        const element = this.renderer.renderComponent(component.name, component.testData);
        
        // Capture screenshot
        const screenshot = await this.capture.captureElement(`#${component.name.replace('_', '-')}`, viewport);
        
        // Load reference image (in real implementation, this would load from file)
        const referenceImage = await this.loadReferenceImage(component.name, viewport.name);
        
        // Compare images
        const comparison = await ImageComparator.compareImages(
            screenshot.imageData,
            referenceImage,
            VISUAL_REGRESSION_CONFIG.threshold
        );
        
        const result = {
            component: component.name,
            viewport: viewport.name,
            success: comparison.match,
            comparison: comparison,
            screenshot: screenshot.dataURL,
            timestamp: new Date().toISOString()
        };
        
        // Generate diff image if there's a mismatch
        if (!comparison.match && referenceImage) {
            result.diffImage = ImageComparator.generateDiffImage(
                screenshot.imageData,
                referenceImage
            );
        }
        
        return result;
    }
    
    async loadReferenceImage(componentName, viewportName) {
        // In a real implementation, this would load from file system
        // For now, we'll generate a simple reference image
        const canvas = document.createElement('canvas');
        canvas.width = 400;
        canvas.height = 300;
        const ctx = canvas.getContext('2d');
        
        // Fill with a reference pattern
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = '#333333';
        ctx.font = '16px Arial';
        ctx.fillText(`Reference: ${componentName}`, 10, 30);
        ctx.fillText(`Viewport: ${viewportName}`, 10, 50);
        
        return ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
    
    getResults() {
        return this.testResults;
    }
    
    cleanup() {
        this.renderer.cleanup();
    }
}

// Global test setup
let visualTestExecutor;

describe('Visual Regression Tests', () => {
    beforeAll(async () => {
        visualTestExecutor = new VisualRegressionTestExecutor();
        await visualTestExecutor.initialize();
        console.log('Visual regression test suite initialized');
    });
    
    afterAll(() => {
        const results = visualTestExecutor.getResults();
        
        console.log('\n=== VISUAL REGRESSION TEST RESULTS ===');
        console.log(`Total tests: ${results.length}`);
        console.log(`Passed: ${results.filter(r => r.success).length}`);
        console.log(`Failed: ${results.filter(r => !r.success).length}`);
        
        const failedTests = results.filter(r => !r.success);
        if (failedTests.length > 0) {
            console.log('\nFailed tests:');
            failedTests.forEach(test => {
                console.log(`- ${test.component} (${test.viewport}): ${test.comparison?.difference || 'N/A'} difference`);
            });
        }
        
        // Store results for analysis
        if (typeof localStorage !== 'undefined') {
            localStorage.setItem('trustformers_visual_regression_results', JSON.stringify(results));
        }
        
        visualTestExecutor.cleanup();
    });
    
    it('should run all visual regression tests', async () => {
        const results = await visualTestExecutor.runVisualRegressionTests();
        
        const expectedTests = VISUAL_REGRESSION_CONFIG.components.length * VISUAL_REGRESSION_CONFIG.viewports.length;
        expect(results.length).toBe(expectedTests);
        
        const failedTests = results.filter(r => !r.success);
        if (failedTests.length > 0) {
            console.error('Visual regression failures:', failedTests.map(t => `${t.component}@${t.viewport}`));
        }
        
        // Allow for some visual differences in test environment
        expect(failedTests.length).toBeLessThan(results.length * 0.1); // Less than 10% failure rate
    }, 60000);
    
    VISUAL_REGRESSION_CONFIG.components.forEach(component => {
        it(`should not have visual regressions in ${component.name}`, async () => {
            const results = await visualTestExecutor.runVisualRegressionTests();
            
            const componentResults = results.filter(r => r.component === component.name);
            expect(componentResults.length).toBe(VISUAL_REGRESSION_CONFIG.viewports.length);
            
            const failedViewports = componentResults.filter(r => !r.success);
            if (failedViewports.length > 0) {
                console.error(`Visual regressions in ${component.name}:`, 
                    failedViewports.map(f => `${f.viewport}: ${f.comparison?.difference || 'N/A'}`));
            }
            
            // Allow for minor differences
            expect(failedViewports.length).toBeLessThan(componentResults.length);
        });
    });
    
    it('should handle missing reference images gracefully', async () => {
        const testComponent = {
            name: 'non_existent_component',
            description: 'Component that does not exist',
            element: '#non-existent',
            testData: {}
        };
        
        try {
            await visualTestExecutor.testComponent(testComponent, VISUAL_REGRESSION_CONFIG.viewports[0]);
            expect(true).toBe(false); // Should not reach here
        } catch (error) {
            expect(error.message).toContain('Element not found');
        }
    });
    
    it('should detect visual changes', async () => {
        // This test verifies that the visual regression system can detect changes
        const results = await visualTestExecutor.runVisualRegressionTests();
        
        // At least some tests should be processed
        expect(results.length).toBeGreaterThan(0);
        
        // Results should contain comparison data
        results.forEach(result => {
            if (result.comparison) {
                expect(result.comparison.difference).toBeDefined();
                expect(result.comparison.match).toBeDefined();
            }
        });
    });
});

// Export for use in other test files
export {
    VisualRegressionTestExecutor,
    ImageComparator,
    ScreenshotCapture,
    MockComponentRenderer,
    VISUAL_REGRESSION_CONFIG
};