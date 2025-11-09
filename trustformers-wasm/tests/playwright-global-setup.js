/**
 * Playwright global setup
 * Runs once before all tests across all browsers
 */

import { chromium } from '@playwright/test';
import fs from 'fs/promises';
import path from 'path';

async function globalSetup() {
  console.log('🚀 Starting global setup for cross-browser tests...');
  
  // Ensure test directories exist
  const directories = [
    'test-results',
    'playwright-report',
    'coverage',
    'screenshots',
    'videos'
  ];
  
  for (const dir of directories) {
    try {
      await fs.mkdir(dir, { recursive: true });
    } catch (error) {
      console.warn(`Could not create directory ${dir}:`, error.message);
    }
  }
  
  // Create test manifest
  const testManifest = {
    timestamp: new Date().toISOString(),
    environment: {
      node_version: process.version,
      platform: process.platform,
      arch: process.arch,
      ci: !!process.env.CI,
      github_actions: !!process.env.GITHUB_ACTIONS,
    },
    browsers: [
      'chromium',
      'firefox', 
      'webkit',
      'edge'
    ],
    test_types: [
      'compatibility',
      'performance',
      'webgpu',
      'error_handling'
    ]
  };
  
  try {
    await fs.writeFile(
      'test-manifest.json', 
      JSON.stringify(testManifest, null, 2)
    );
    console.log('📋 Test manifest created');
  } catch (error) {
    console.warn('Could not create test manifest:', error.message);
  }
  
  // Pre-compile WASM if needed
  try {
    // Check if WASM file exists
    const wasmFiles = await fs.readdir('.')
      .then(files => files.filter(f => f.endsWith('.wasm')))
      .catch(() => []);
    
    if (wasmFiles.length === 0) {
      console.log('⚠️  No WASM files found - tests will use mock WASM module');
    } else {
      console.log(`✅ Found WASM files: ${wasmFiles.join(', ')}`);
    }
  } catch (error) {
    console.warn('Could not check for WASM files:', error.message);
  }
  
  // Check browser capabilities
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  
  try {
    // Check WebAssembly support
    const wasmSupport = await page.evaluate(() => {
      return typeof WebAssembly !== 'undefined';
    });
    
    // Check WebGPU support  
    const webgpuSupport = await page.evaluate(async () => {
      if (!navigator.gpu) return false;
      try {
        const adapter = await navigator.gpu.requestAdapter();
        return !!adapter;
      } catch {
        return false;
      }
    });
    
    // Check SharedArrayBuffer support
    const sabSupport = await page.evaluate(() => {
      return typeof SharedArrayBuffer !== 'undefined';
    });
    
    const capabilities = {
      webassembly: wasmSupport,
      webgpu: webgpuSupport,
      sharedarraybuffer: sabSupport,
      userAgent: await page.evaluate(() => navigator.userAgent),
      platform: await page.evaluate(() => navigator.platform),
      hardwareConcurrency: await page.evaluate(() => navigator.hardwareConcurrency),
      deviceMemory: await page.evaluate(() => navigator.deviceMemory || 'unknown')
    };
    
    await fs.writeFile(
      'browser-capabilities.json',
      JSON.stringify(capabilities, null, 2)
    );
    
    console.log('🔍 Browser capabilities detected:');
    console.log(`  WebAssembly: ${capabilities.webassembly ? '✅' : '❌'}`);
    console.log(`  WebGPU: ${capabilities.webgpu ? '✅' : '❌'}`);
    console.log(`  SharedArrayBuffer: ${capabilities.sharedarraybuffer ? '✅' : '❌'}`);
    
  } finally {
    await context.close();
    await browser.close();
  }
  
  // Set global test environment variables
  process.env.TEST_START_TIME = Date.now().toString();
  process.env.TEST_SESSION_ID = Math.random().toString(36).substring(7);
  
  console.log('✅ Global setup completed');
}

export default globalSetup;