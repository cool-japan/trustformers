#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');
const { gzipSync, brotliCompressSync } = require('zlib');

class BundleSizeAnalyzer {
  constructor() {
    this.distDir = path.resolve(__dirname, '..', 'dist');
    this.limits = {
      'trustformers.esm.min.js': 500 * 1024, // 500KB
      'trustformers.umd.min.js': 600 * 1024, // 600KB
      'trustformers.cjs.min.js': 550 * 1024, // 550KB
      'trustformers.iife.min.js': 580 * 1024, // 580KB
      'modules/tensor.min.js': 100 * 1024, // 100KB
      'modules/models.min.js': 150 * 1024, // 150KB
      'modules/pipeline.min.js': 120 * 1024, // 120KB
      'modules/utils.min.js': 80 * 1024 // 80KB
    };
  }

  async analyze() {
    console.log('📊 Analyzing bundle sizes...\n');

    const results = [];
    let allPassed = true;

    try {
      for (const [filename, limit] of Object.entries(this.limits)) {
        const filePath = path.join(this.distDir, filename);
        
        try {
          const content = await fs.readFile(filePath);
          const originalSize = content.length;
          const gzippedSize = gzipSync(content).length;
          const brotliSize = brotliCompressSync(content).length;
          
          const gzippedPercent = ((gzippedSize / originalSize) * 100).toFixed(1);
          const brotliPercent = ((brotliSize / originalSize) * 100).toFixed(1);
          const limitPassed = gzippedSize <= limit;
          
          if (!limitPassed) allPassed = false;
          
          results.push({
            filename,
            originalSize,
            gzippedSize,
            brotliSize,
            limit,
            limitPassed,
            gzippedPercent,
            brotliPercent
          });
          
          console.log(`📄 ${filename}`);
          console.log(`   Original: ${this.formatBytes(originalSize)}`);
          console.log(`   Gzipped:  ${this.formatBytes(gzippedSize)} (${gzippedPercent}% compression)`);
          console.log(`   Brotli:   ${this.formatBytes(brotliSize)} (${brotliPercent}% compression)`);
          console.log(`   Limit:    ${this.formatBytes(limit)}`);
          console.log(`   Status:   ${limitPassed ? '✅ PASS' : '❌ FAIL'}`);
          console.log('');
          
        } catch (error) {
          console.log(`⚠️  ${filename}: File not found`);
          results.push({
            filename,
            error: 'File not found'
          });
        }
      }

      // Summary
      console.log('📋 Summary');
      console.log('==========');
      
      const totalOriginal = results.reduce((sum, r) => sum + (r.originalSize || 0), 0);
      const totalGzipped = results.reduce((sum, r) => sum + (r.gzippedSize || 0), 0);
      const totalBrotli = results.reduce((sum, r) => sum + (r.brotliSize || 0), 0);
      
      console.log(`Total Original Size: ${this.formatBytes(totalOriginal)}`);
      console.log(`Total Gzipped Size:  ${this.formatBytes(totalGzipped)}`);
      console.log(`Total Brotli Size:   ${this.formatBytes(totalBrotli)}`);
      console.log(`Overall Compression: ${((totalGzipped / totalOriginal) * 100).toFixed(1)}% (gzip)`);
      console.log(`Overall Compression: ${((totalBrotli / totalOriginal) * 100).toFixed(1)}% (brotli)`);
      
      const passed = results.filter(r => r.limitPassed).length;
      const total = results.filter(r => !r.error).length;
      
      console.log(`\nSize Checks: ${passed}/${total} passed`);
      
      if (allPassed) {
        console.log('\n🎉 All bundle size checks passed!');
      } else {
        console.log('\n❌ Some bundle size checks failed!');
        console.log('\nRecommendations:');
        console.log('1. Review and optimize large dependencies');
        console.log('2. Enable tree-shaking in your bundler');
        console.log('3. Use dynamic imports for non-critical code');
        console.log('4. Consider splitting large modules');
      }

      // Save results
      await this.saveResults(results);
      
      return allPassed;
      
    } catch (error) {
      console.error('❌ Bundle size analysis failed:', error.message);
      return false;
    }
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  async saveResults(results) {
    const reportPath = path.join(this.distDir, 'bundle-size-report.json');
    const report = {
      timestamp: new Date().toISOString(),
      results,
      summary: {
        totalFiles: results.filter(r => !r.error).length,
        passedChecks: results.filter(r => r.limitPassed).length,
        totalOriginalSize: results.reduce((sum, r) => sum + (r.originalSize || 0), 0),
        totalGzippedSize: results.reduce((sum, r) => sum + (r.gzippedSize || 0), 0),
        totalBrotliSize: results.reduce((sum, r) => sum + (r.brotliSize || 0), 0)
      }
    };
    
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    console.log(`📊 Bundle size report saved to: ${reportPath}`);
  }
}

// Run analysis if called directly
if (require.main === module) {
  const analyzer = new BundleSizeAnalyzer();
  analyzer.analyze().then(passed => {
    process.exit(passed ? 0 : 1);
  });
}

module.exports = BundleSizeAnalyzer;