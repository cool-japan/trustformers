/**
 * Playwright configuration for cross-browser testing
 * @type {import('@playwright/test').PlaywrightTestConfig}
 */

import { defineConfig, devices } from '@playwright/test';
import path from 'path';

export default defineConfig({
  // Test directory
  testDir: './tests',
  
  // Test files pattern
  testMatch: '**/*.playwright.test.js',
  
  // Global setup
  globalSetup: './tests/playwright-global-setup.js',
  
  // Maximum time one test can run for
  timeout: 60 * 1000,
  
  // Test timeout expectation
  expect: {
    timeout: 10 * 1000,
  },
  
  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,
  
  // Retry on CI only
  retries: process.env.CI ? 2 : 0,
  
  // Opt out of parallel tests on CI
  workers: process.env.CI ? 1 : undefined,
  
  // Reporter to use
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results.json' }],
    ['junit', { outputFile: 'junit-results.xml' }],
    process.env.CI ? ['github'] : ['list']
  ],
  
  // Shared settings for all the projects below
  use: {
    // Base URL to use in actions like `await page.goto('/')`
    baseURL: 'http://localhost:8080',
    
    // Collect trace when retrying the failed test
    trace: 'on-first-retry',
    
    // Record video on failure
    video: 'retain-on-failure',
    
    // Take screenshot on failure
    screenshot: 'only-on-failure',
  },

  // Configure projects for major browsers
  projects: [
    {
      name: 'chromium',
      use: { 
        ...devices['Desktop Chrome'],
        // Enable experimental web platform features
        launchOptions: {
          args: [
            '--enable-experimental-web-platform-features',
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-vulkan=native',
            '--enable-gpu-rasterization',
            '--enable-oop-rasterization',
            '--disable-web-security', // For testing only
          ]
        }
      },
    },
    
    {
      name: 'firefox',
      use: { 
        ...devices['Desktop Firefox'],
        launchOptions: {
          firefoxUserPrefs: {
            // Enable WebGPU in Firefox (experimental)
            'dom.webgpu.enabled': true,
            'gfx.webgpu.force-enabled': true,
            // Enable SharedArrayBuffer
            'javascript.options.shared_memory': true,
            // Enable SIMD
            'javascript.options.wasm_simd': true,
          }
        }
      },
    },
    
    {
      name: 'webkit',
      use: { 
        ...devices['Desktop Safari'],
        launchOptions: {
          // Safari-specific WebKit options
        }
      },
    },
    
    {
      name: 'edge',
      use: { 
        ...devices['Desktop Edge'],
        channel: 'msedge',
        launchOptions: {
          args: [
            '--enable-experimental-web-platform-features',
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
          ]
        }
      },
    },

    // Mobile browsers
    {
      name: 'Mobile Chrome',
      use: { 
        ...devices['Pixel 5'],
        launchOptions: {
          args: [
            '--enable-experimental-web-platform-features',
            '--disable-web-security',
          ]
        }
      },
    },
    
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },

    // Specific WebGPU testing browsers
    {
      name: 'chrome-webgpu',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-experimental-web-platform-features',
            '--use-vulkan=native',
            '--enable-features=Vulkan,VulkanFromANGLE',
            '--disable-vulkan-fallback-to-gl-for-testing',
          ]
        }
      },
      testMatch: '**/*webgpu*.test.js',
    },

    // Performance testing browser
    {
      name: 'chrome-performance',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            '--enable-experimental-web-platform-features',
            '--js-flags=--max-old-space-size=4096',
            '--memory-pressure-off',
            '--max_old_space_size=4096',
          ]
        }
      },
      testMatch: '**/*performance*.test.js',
    },
  ],

  // Run your local dev server before starting the tests
  webServer: {
    command: 'npm run serve',
    port: 8080,
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },

  // Global test configuration
  globalTeardown: './tests/playwright-global-teardown.js',

  // Test metadata
  metadata: {
    testType: 'cross-browser',
    framework: 'trustformers-wasm',
    environment: process.env.NODE_ENV || 'test',
  },
});