#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');
const { gzipSync, brotliCompressSync } = require('zlib');

class TrustformersBuildSystem {
  constructor() {
    this.rootDir = path.resolve(__dirname, '..');
    this.distDir = path.join(this.rootDir, 'dist');
    this.srcDir = path.join(this.rootDir, 'src');
    this.pkgDir = path.join(this.rootDir, 'pkg');
    this.buildStats = {
      startTime: Date.now(),
      files: [],
      totalSize: 0,
      gzippedSize: 0,
      brotliSize: 0
    };
  }

  async initialize() {
    console.log('🚀 TrustformeRS Build System v2.0');
    console.log('=====================================');
    
    // Clean dist directory
    await this.cleanDist();
    
    // Ensure pkg directory exists
    await this.ensurePkgDirectory();
    
    // Build WASM module first
    await this.buildWasm();
    
    console.log('✅ Initialization complete\n');
  }

  async cleanDist() {
    console.log('🧹 Cleaning dist directory...');
    try {
      await fs.rm(this.distDir, { recursive: true, force: true });
      await fs.mkdir(this.distDir, { recursive: true });
      await fs.mkdir(path.join(this.distDir, 'modules'), { recursive: true });
      console.log('✅ Dist directory cleaned');
    } catch (error) {
      console.error('❌ Error cleaning dist directory:', error.message);
      throw error;
    }
  }

  async ensurePkgDirectory() {
    console.log('📦 Ensuring WASM package directory...');
    try {
      await fs.access(this.pkgDir);
      console.log('✅ Package directory exists');
    } catch {
      console.log('⚠️  Package directory not found, will be created during WASM build');
    }
  }

  async buildWasm() {
    console.log('🔧 Building WASM module...');
    try {
      const wasmDir = path.resolve(this.rootDir, '..', 'trustformers-wasm');
      process.chdir(wasmDir);
      
      execSync('wasm-pack build --target web --out-dir ../trustformers-js/pkg --release', {
        stdio: 'inherit'
      });
      
      process.chdir(this.rootDir);
      console.log('✅ WASM module built successfully');
      
      // Copy WASM files to dist
      await this.copyWasmFiles();
    } catch (error) {
      console.error('❌ WASM build failed:', error.message);
      throw error;
    }
  }

  async copyWasmFiles() {
    console.log('📋 Copying WASM files to dist...');
    try {
      const wasmFiles = ['trustformers_wasm.js', 'trustformers_wasm_bg.wasm', 'trustformers_wasm.d.ts'];
      
      for (const file of wasmFiles) {
        const srcPath = path.join(this.pkgDir, file);
        const destPath = path.join(this.distDir, file);
        await fs.copyFile(srcPath, destPath);
      }
      
      console.log('✅ WASM files copied');
    } catch (error) {
      console.error('❌ Error copying WASM files:', error.message);
      throw error;
    }
  }

  async buildRollup() {
    console.log('📦 Building with Rollup (tree-shaking optimized)...');
    try {
      process.env.NODE_ENV = 'production';
      execSync('npx rollup -c rollup.config.js', { stdio: 'inherit' });
      console.log('✅ Rollup build complete');
    } catch (error) {
      console.error('❌ Rollup build failed:', error.message);
      throw error;
    }
  }

  async buildWebpack() {
    console.log('📦 Building with Webpack (compatibility builds)...');
    try {
      process.env.NODE_ENV = 'production';
      execSync('npx webpack --mode production', { stdio: 'inherit' });
      console.log('✅ Webpack build complete');
    } catch (error) {
      console.error('❌ Webpack build failed:', error.message);
      throw error;
    }
  }

  async generateTypeDefinitions() {
    console.log('🔤 Generating TypeScript definitions...');
    try {
      // Copy main type definitions
      const typesSource = path.join(this.srcDir, 'index.d.ts');
      const typesDest = path.join(this.distDir, 'index.d.ts');
      await fs.copyFile(typesSource, typesDest);

      // Generate modular type definitions
      const moduleDirs = ['tensor', 'models', 'pipeline', 'utils'];
      for (const moduleDir of moduleDirs) {
        const moduleTypesPath = path.join(this.srcDir, moduleDir, 'index.d.ts');
        try {
          await fs.access(moduleTypesPath);
          const destPath = path.join(this.distDir, 'modules', `${moduleDir}.d.ts`);
          await fs.copyFile(moduleTypesPath, destPath);
        } catch {
          // Type file doesn't exist, skip
        }
      }

      console.log('✅ TypeScript definitions generated');
    } catch (error) {
      console.error('❌ Error generating type definitions:', error.message);
      throw error;
    }
  }

  async generatePackageFiles() {
    console.log('📝 Generating package.json variants...');
    try {
      const mainPackage = JSON.parse(await fs.readFile(path.join(this.rootDir, 'package.json'), 'utf8'));
      
      // ESM package.json
      const esmPackage = {
        ...mainPackage,
        type: 'module',
        main: './trustformers.esm.min.js',
        module: './trustformers.esm.min.js',
        exports: {
          '.': {
            import: './trustformers.esm.min.js',
            require: './trustformers.cjs.min.js',
            types: './index.d.ts'
          },
          './modules/*': {
            import: './modules/*.min.js',
            types: './modules/*.d.ts'
          }
        }
      };
      
      await fs.writeFile(
        path.join(this.distDir, 'package.json'),
        JSON.stringify(esmPackage, null, 2)
      );

      console.log('✅ Package files generated');
    } catch (error) {
      console.error('❌ Error generating package files:', error.message);
      throw error;
    }
  }

  async createCDNManifest() {
    console.log('🌐 Creating CDN manifest...');
    try {
      const manifest = {
        name: 'TrustformeRS',
        version: require('../package.json').version,
        files: {
          'trustformers.umd.min.js': {
            format: 'umd',
            global: 'TrustformeRS',
            description: 'Complete UMD bundle for CDN usage'
          },
          'trustformers.esm.min.js': {
            format: 'esm',
            description: 'ES module for modern bundlers'
          },
          'trustformers.iife.min.js': {
            format: 'iife',
            global: 'TrustformeRS',
            description: 'IIFE bundle for direct browser usage'
          },
          'trustformers_wasm_bg.wasm': {
            format: 'wasm',
            description: 'WebAssembly module'
          }
        },
        integrity: {},
        cdn: {
          jsdelivr: `https://cdn.jsdelivr.net/npm/trustformers@${require('../package.json').version}/dist/`,
          unpkg: `https://unpkg.com/trustformers@${require('../package.json').version}/dist/`
        }
      };

      await fs.writeFile(
        path.join(this.distDir, 'cdn-manifest.json'),
        JSON.stringify(manifest, null, 2)
      );

      console.log('✅ CDN manifest created');
    } catch (error) {
      console.error('❌ Error creating CDN manifest:', error.message);
      throw error;
    }
  }

  async calculateFileStats() {
    console.log('📊 Calculating build statistics...');
    try {
      const files = await fs.readdir(this.distDir, { recursive: true });
      
      for (const file of files) {
        const filePath = path.join(this.distDir, file);
        const stats = await fs.stat(filePath);
        
        if (stats.isFile()) {
          const content = await fs.readFile(filePath);
          const gzipped = gzipSync(content);
          const brotli = brotliCompressSync(content);
          
          this.buildStats.files.push({
            name: file,
            size: stats.size,
            gzippedSize: gzipped.length,
            brotliSize: brotli.length
          });
          
          this.buildStats.totalSize += stats.size;
          this.buildStats.gzippedSize += gzipped.length;
          this.buildStats.brotliSize += brotli.length;
        }
      }

      console.log('✅ Build statistics calculated');
    } catch (error) {
      console.error('❌ Error calculating statistics:', error.message);
      throw error;
    }
  }

  async generateBuildReport() {
    console.log('📋 Generating build report...');
    try {
      const endTime = Date.now();
      const buildTime = ((endTime - this.buildStats.startTime) / 1000).toFixed(2);
      
      const report = {
        buildTime: `${buildTime}s`,
        timestamp: new Date().toISOString(),
        version: require('../package.json').version,
        stats: this.buildStats,
        summary: {
          totalFiles: this.buildStats.files.length,
          totalSize: this.formatBytes(this.buildStats.totalSize),
          gzippedSize: this.formatBytes(this.buildStats.gzippedSize),
          brotliSize: this.formatBytes(this.buildStats.brotliSize),
          compressionRatio: {
            gzip: ((1 - this.buildStats.gzippedSize / this.buildStats.totalSize) * 100).toFixed(1) + '%',
            brotli: ((1 - this.buildStats.brotliSize / this.buildStats.totalSize) * 100).toFixed(1) + '%'
          }
        }
      };

      await fs.writeFile(
        path.join(this.distDir, 'build-report.json'),
        JSON.stringify(report, null, 2)
      );

      // Print summary
      console.log('\n📊 Build Summary');
      console.log('================');
      console.log(`Build Time: ${report.buildTime}`);
      console.log(`Total Files: ${report.summary.totalFiles}`);
      console.log(`Total Size: ${report.summary.totalSize}`);
      console.log(`Gzipped: ${report.summary.gzippedSize} (${report.summary.compressionRatio.gzip} compression)`);
      console.log(`Brotli: ${report.summary.brotliSize} (${report.summary.compressionRatio.brotli} compression)`);

      console.log('\n📄 Individual Files:');
      this.buildStats.files
        .sort((a, b) => b.size - a.size)
        .slice(0, 10)
        .forEach(file => {
          console.log(`  ${file.name}: ${this.formatBytes(file.size)} (${this.formatBytes(file.gzippedSize)} gzipped)`);
        });

      console.log('✅ Build report generated');
    } catch (error) {
      console.error('❌ Error generating build report:', error.message);
      throw error;
    }
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  async build() {
    try {
      await this.initialize();
      await this.buildRollup();
      await this.buildWebpack();
      await this.generateTypeDefinitions();
      await this.generatePackageFiles();
      await this.createCDNManifest();
      await this.calculateFileStats();
      await this.generateBuildReport();
      
      console.log('\n🎉 Build completed successfully!');
      console.log(`📁 Output directory: ${this.distDir}`);
    } catch (error) {
      console.error('\n💥 Build failed:', error.message);
      process.exit(1);
    }
  }
}

// Run build if called directly
if (require.main === module) {
  const builder = new TrustformersBuildSystem();
  builder.build();
}

module.exports = TrustformersBuildSystem;