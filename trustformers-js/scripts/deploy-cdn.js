#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

class CDNDeployment {
  constructor() {
    this.distDir = path.resolve(__dirname, '..', 'dist');
    this.packageInfo = require('../package.json');
    this.cdnConfig = {
      jsdelivr: {
        baseUrl: 'https://cdn.jsdelivr.net/npm/trustformers@latest/dist/',
        purgeUrl: 'https://purge.jsdelivr.net/npm/trustformers@',
        enabled: true
      },
      unpkg: {
        baseUrl: 'https://unpkg.com/trustformers@latest/dist/',
        enabled: true
      },
      custom: {
        baseUrl: process.env.CUSTOM_CDN_URL || '',
        apiKey: process.env.CDN_API_KEY || '',
        enabled: !!process.env.CUSTOM_CDN_URL
      }
    };
  }

  async deploy() {
    console.log('🚀 Starting CDN deployment...');
    console.log(`📦 Package: ${this.packageInfo.name}@${this.packageInfo.version}`);
    
    try {
      // Generate integrity hashes
      await this.generateIntegrityHashes();
      
      // Update CDN manifest
      await this.updateCDNManifest();
      
      // Purge CDN caches
      await this.purgeCDNCaches();
      
      // Generate CDN examples
      await this.generateCDNExamples();
      
      // Update documentation
      await this.updateDocumentation();
      
      console.log('✅ CDN deployment completed successfully!');
      
    } catch (error) {
      console.error('❌ CDN deployment failed:', error.message);
      process.exit(1);
    }
  }

  async generateIntegrityHashes() {
    console.log('🔐 Generating integrity hashes...');
    
    const files = await fs.readdir(this.distDir);
    const hashes = {};
    
    for (const file of files) {
      if (file.endsWith('.js') || file.endsWith('.wasm')) {
        const filePath = path.join(this.distDir, file);
        const content = await fs.readFile(filePath);
        const hash = crypto.createHash('sha384').update(content).digest('base64');
        hashes[file] = `sha384-${hash}`;
        console.log(`  ${file}: ${hashes[file]}`);
      }
    }
    
    // Save hashes to manifest
    const hashesPath = path.join(this.distDir, 'integrity.json');
    await fs.writeFile(hashesPath, JSON.stringify(hashes, null, 2));
    
    console.log('✅ Integrity hashes generated');
  }

  async updateCDNManifest() {
    console.log('📄 Updating CDN manifest...');
    
    const manifestPath = path.join(this.distDir, 'cdn-manifest.json');
    const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
    
    // Update version
    manifest.version = this.packageInfo.version;
    
    // Update file sizes and integrity
    const integrityHashes = JSON.parse(
      await fs.readFile(path.join(this.distDir, 'integrity.json'), 'utf8')
    );
    
    for (const [filename, fileInfo] of Object.entries(manifest.files)) {
      const filePath = path.join(this.distDir, filename);
      try {
        const stats = await fs.stat(filePath);
        fileInfo.size = stats.size;
        fileInfo.integrity = integrityHashes[filename] || '';
        fileInfo.lastModified = stats.mtime.toISOString();
      } catch {
        console.warn(`⚠️  File not found: ${filename}`);
      }
    }
    
    // Update CDN URLs with version
    manifest.cdn = {
      jsdelivr: `https://cdn.jsdelivr.net/npm/trustformers@${this.packageInfo.version}/dist/`,
      unpkg: `https://unpkg.com/trustformers@${this.packageInfo.version}/dist/`,
      ...this.cdnConfig.custom.enabled && {
        custom: this.cdnConfig.custom.baseUrl
      }
    };
    
    await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2));
    console.log('✅ CDN manifest updated');
  }

  async purgeCDNCaches() {
    console.log('🔄 Purging CDN caches...');
    
    const purgeTasks = [];
    
    // Purge jsDelivr
    if (this.cdnConfig.jsdelivr.enabled) {
      purgeTasks.push(this.purgeJsDelivr());
    }
    
    // Purge custom CDN
    if (this.cdnConfig.custom.enabled && this.cdnConfig.custom.apiKey) {
      purgeTasks.push(this.purgeCustomCDN());
    }
    
    const results = await Promise.allSettled(purgeTasks);
    
    results.forEach((result, index) => {
      const cdnName = index === 0 ? 'jsDelivr' : 'Custom CDN';
      if (result.status === 'fulfilled') {
        console.log(`  ✅ ${cdnName} cache purged`);
      } else {
        console.warn(`  ⚠️  ${cdnName} cache purge failed:`, result.reason.message);
      }
    });
  }

  async purgeJsDelivr() {
    const purgeUrl = `${this.cdnConfig.jsdelivr.purgeUrl}${this.packageInfo.version}`;
    
    const response = await fetch(purgeUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok) {
      throw new Error(`jsDelivr purge failed: ${response.statusText}`);
    }
    
    return response.json();
  }

  async purgeCustomCDN() {
    // Implementation depends on your CDN provider
    // This is a generic example
    const response = await fetch(`${this.cdnConfig.custom.baseUrl}/purge`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.cdnConfig.custom.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        package: this.packageInfo.name,
        version: this.packageInfo.version
      })
    });
    
    if (!response.ok) {
      throw new Error(`Custom CDN purge failed: ${response.statusText}`);
    }
    
    return response.json();
  }

  async generateCDNExamples() {
    console.log('📝 Generating CDN usage examples...');
    
    const examples = {
      umd: {
        html: this.generateUMDExample(),
        description: 'Direct browser usage with UMD bundle'
      },
      esm: {
        html: this.generateESMExample(),
        description: 'Modern ES modules usage'
      },
      iife: {
        html: this.generateIIFEExample(),
        description: 'Simple script tag usage'
      }
    };
    
    const examplesDir = path.join(this.distDir, 'examples');
    await fs.mkdir(examplesDir, { recursive: true });
    
    for (const [type, example] of Object.entries(examples)) {
      const filename = `cdn-${type}-example.html`;
      const filepath = path.join(examplesDir, filename);
      await fs.writeFile(filepath, example.html);
      console.log(`  📄 Generated: ${filename}`);
    }
    
    console.log('✅ CDN examples generated');
  }

  generateUMDExample() {
    const version = this.packageInfo.version;
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS CDN UMD Example</title>
</head>
<body>
    <h1>TrustformeRS UMD Example</h1>
    <div id="output"></div>
    
    <script src="https://cdn.jsdelivr.net/npm/trustformers@${version}/dist/trustformers.umd.min.js"></script>
    <script>
        async function main() {
            try {
                await TrustformeRS.init();
                
                // Create a tensor
                const tensor = TrustformeRS.tensor([1, 2, 3, 4], [2, 2]);
                console.log('Tensor shape:', tensor.shape);
                
                // Load a model (example)
                // const model = await TrustformeRS.loadModel('path/to/model');
                
                document.getElementById('output').innerHTML = 
                    '<p>✅ TrustformeRS initialized successfully!</p>' +
                    '<p>Tensor shape: [' + tensor.shape.join(', ') + ']</p>';
                    
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('output').innerHTML = 
                    '<p>❌ Error: ' + error.message + '</p>';
            }
        }
        
        main();
    </script>
</body>
</html>`;
  }

  generateESMExample() {
    const version = this.packageInfo.version;
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS CDN ESM Example</title>
</head>
<body>
    <h1>TrustformeRS ES Modules Example</h1>
    <div id="output"></div>
    
    <script type="module">
        import * as TrustformeRS from 'https://cdn.jsdelivr.net/npm/trustformers@${version}/dist/trustformers.esm.min.js';
        
        async function main() {
            try {
                await TrustformeRS.init();
                
                // Create a tensor
                const tensor = TrustformeRS.tensor([1, 2, 3, 4], [2, 2]);
                console.log('Tensor shape:', tensor.shape);
                
                // Demonstrate tree-shaking with modular imports
                const { tensor: createTensor, add } = TrustformeRS;
                const a = createTensor([1, 2], [2, 1]);
                const b = createTensor([3, 4], [2, 1]);
                const result = add(a, b);
                
                document.getElementById('output').innerHTML = 
                    '<p>✅ TrustformeRS ESM initialized successfully!</p>' +
                    '<p>Tensor shape: [' + tensor.shape.join(', ') + ']</p>' +
                    '<p>Addition result shape: [' + result.shape.join(', ') + ']</p>';
                    
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('output').innerHTML = 
                    '<p>❌ Error: ' + error.message + '</p>';
            }
        }
        
        main();
    </script>
</body>
</html>`;
  }

  generateIIFEExample() {
    const version = this.packageInfo.version;
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS CDN IIFE Example</title>
</head>
<body>
    <h1>TrustformeRS IIFE Example</h1>
    <div id="output"></div>
    
    <script src="https://cdn.jsdelivr.net/npm/trustformers@${version}/dist/trustformers.iife.min.js"></script>
    <script>
        // TrustformeRS is available as a global variable
        TrustformeRS.init().then(() => {
            // Create a tensor
            const tensor = TrustformeRS.tensor([1, 2, 3, 4], [2, 2]);
            console.log('Tensor created:', tensor);
            
            document.getElementById('output').innerHTML = 
                '<p>✅ TrustformeRS IIFE initialized successfully!</p>' +
                '<p>Tensor shape: [' + tensor.shape.join(', ') + ']</p>';
                
        }).catch(error => {
            console.error('Error:', error);
            document.getElementById('output').innerHTML = 
                '<p>❌ Error: ' + error.message + '</p>';
        });
    </script>
</body>
</html>`;
  }

  async updateDocumentation() {
    console.log('📚 Updating documentation...');
    
    const cdnDocsPath = path.join(this.distDir, 'CDN_USAGE.md');
    const cdnDocs = this.generateCDNDocumentation();
    
    await fs.writeFile(cdnDocsPath, cdnDocs);
    console.log('✅ CDN documentation updated');
  }

  generateCDNDocumentation() {
    const version = this.packageInfo.version;
    return `# TrustformeRS CDN Usage

## Quick Start

### Via jsDelivr (Recommended)

\`\`\`html
<!-- UMD Bundle -->
<script src="https://cdn.jsdelivr.net/npm/trustformers@${version}/dist/trustformers.umd.min.js"></script>

<!-- ES Module -->
<script type="module">
  import * as TrustformeRS from 'https://cdn.jsdelivr.net/npm/trustformers@${version}/dist/trustformers.esm.min.js';
</script>
\`\`\`

### Via unpkg

\`\`\`html
<!-- UMD Bundle -->
<script src="https://unpkg.com/trustformers@${version}/dist/trustformers.umd.min.js"></script>

<!-- ES Module -->
<script type="module">
  import * as TrustformeRS from 'https://unpkg.com/trustformers@${version}/dist/trustformers.esm.min.js';
</script>
\`\`\`

## Available Builds

| Format | File | Size | Use Case |
|--------|------|------|----------|
| UMD | \`trustformers.umd.min.js\` | ~500KB | Direct browser usage, legacy support |
| ESM | \`trustformers.esm.min.js\` | ~450KB | Modern bundlers, tree-shaking |
| IIFE | \`trustformers.iife.min.js\` | ~480KB | Simple script tag usage |
| CommonJS | \`trustformers.cjs.min.js\` | ~460KB | Node.js environments |

## Modular Imports (Tree-shaking)

\`\`\`javascript
// Import only what you need
import { tensor, add, multiply } from 'https://cdn.jsdelivr.net/npm/trustformers@${version}/dist/modules/tensor.min.js';
import { loadModel } from 'https://cdn.jsdelivr.net/npm/trustformers@${version}/dist/modules/models.min.js';
\`\`\`

## Security

All CDN files include SHA-384 integrity hashes for security:

\`\`\`html
<script 
  src="https://cdn.jsdelivr.net/npm/trustformers@${version}/dist/trustformers.umd.min.js"
  integrity="sha384-..."
  crossorigin="anonymous">
</script>
\`\`\`

## Examples

See the \`examples/\` directory for complete usage examples:
- \`cdn-umd-example.html\` - UMD bundle usage
- \`cdn-esm-example.html\` - ES modules usage  
- \`cdn-iife-example.html\` - IIFE bundle usage

## Version Pinning

For production use, always pin to a specific version:

\`\`\`html
<!-- Pin to exact version -->
<script src="https://cdn.jsdelivr.net/npm/trustformers@${version}/dist/trustformers.umd.min.js"></script>

<!-- Pin to major version (auto-updates) -->
<script src="https://cdn.jsdelivr.net/npm/trustformers@^${version}/dist/trustformers.umd.min.js"></script>
\`\`\`

## Performance Tips

1. Use \`preload\` for critical resources:
   \`\`\`html
   <link rel="preload" href="https://cdn.jsdelivr.net/npm/trustformers@${version}/dist/trustformers.umd.min.js" as="script">
   \`\`\`

2. Enable compression at your server level
3. Use modular imports to reduce bundle size
4. Consider using a Web Worker for heavy computations

For more information, visit our [documentation](https://github.com/yourusername/trustformers).
`;
  }
}

// Run deployment if called directly
if (require.main === module) {
  const deployment = new CDNDeployment();
  deployment.deploy();
}

module.exports = CDNDeployment;