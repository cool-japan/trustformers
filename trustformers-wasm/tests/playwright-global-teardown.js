/**
 * Playwright global teardown
 * Runs once after all tests across all browsers
 */

import fs from 'fs/promises';
import path from 'path';

async function globalTeardown() {
  console.log('🧹 Starting global teardown...');
  
  try {
    // Calculate test session duration
    const startTime = process.env.TEST_START_TIME;
    const endTime = Date.now();
    const duration = startTime ? endTime - parseInt(startTime) : 0;
    
    // Collect test results summary
    const testSummary = {
      sessionId: process.env.TEST_SESSION_ID,
      startTime: startTime ? new Date(parseInt(startTime)).toISOString() : null,
      endTime: new Date(endTime).toISOString(),
      duration: `${Math.round(duration / 1000)}s`,
      environment: {
        node_version: process.version,
        platform: process.platform,
        arch: process.arch,
        ci: !!process.env.CI,
      }
    };
    
    // Try to read test results
    try {
      const resultsFile = await fs.readFile('test-results.json', 'utf8');
      const results = JSON.parse(resultsFile);
      
      testSummary.results = {
        total: results.suites?.reduce((acc, suite) => acc + (suite.specs?.length || 0), 0) || 0,
        passed: 0,
        failed: 0,
        skipped: 0,
      };
      
      // Count test results
      if (results.suites) {
        for (const suite of results.suites) {
          if (suite.specs) {
            for (const spec of suite.specs) {
              if (spec.tests) {
                for (const test of spec.tests) {
                  if (test.status === 'passed') testSummary.results.passed++;
                  else if (test.status === 'failed') testSummary.results.failed++;
                  else if (test.status === 'skipped') testSummary.results.skipped++;
                }
              }
            }
          }
        }
      }
      
      console.log('📊 Test Results Summary:');
      console.log(`  Total: ${testSummary.results.total}`);
      console.log(`  Passed: ${testSummary.results.passed} ✅`);
      console.log(`  Failed: ${testSummary.results.failed} ${testSummary.results.failed > 0 ? '❌' : ''}`);
      console.log(`  Skipped: ${testSummary.results.skipped} ⏭️`);
      console.log(`  Duration: ${testSummary.duration}`);
      
    } catch (error) {
      console.warn('Could not read test results:', error.message);
      testSummary.results = { error: 'Could not read test results' };
    }
    
    // Write test summary
    await fs.writeFile(
      'test-summary.json',
      JSON.stringify(testSummary, null, 2)
    );
    
    // Cleanup old test artifacts (keep last 5 runs)
    await cleanupOldArtifacts();
    
    // Generate coverage report if available
    await generateCoverageReport();
    
    // Create markdown report for GitHub Actions
    if (process.env.GITHUB_ACTIONS) {
      await createGitHubReport(testSummary);
    }
    
  } catch (error) {
    console.error('Error during global teardown:', error);
  }
  
  console.log('✅ Global teardown completed');
}

async function cleanupOldArtifacts() {
  try {
    const directories = ['screenshots', 'videos', 'test-results'];
    
    for (const dir of directories) {
      try {
        const files = await fs.readdir(dir);
        if (files.length > 50) { // Keep only last 50 files
          const fileStats = await Promise.all(
            files.map(async (file) => {
              const stat = await fs.stat(path.join(dir, file));
              return { file, mtime: stat.mtime };
            })
          );
          
          fileStats.sort((a, b) => b.mtime - a.mtime);
          const filesToDelete = fileStats.slice(50);
          
          for (const { file } of filesToDelete) {
            await fs.unlink(path.join(dir, file));
          }
          
          if (filesToDelete.length > 0) {
            console.log(`🗑️  Cleaned up ${filesToDelete.length} old files from ${dir}`);
          }
        }
      } catch (error) {
        // Directory might not exist, ignore
      }
    }
  } catch (error) {
    console.warn('Could not cleanup old artifacts:', error.message);
  }
}

async function generateCoverageReport() {
  try {
    // Check if coverage data exists
    const coverageFiles = await fs.readdir('coverage').catch(() => []);
    
    if (coverageFiles.length > 0) {
      console.log('📈 Coverage report available in coverage/ directory');
      
      // Try to read coverage summary
      try {
        const coverageSummary = await fs.readFile('coverage/coverage-summary.json', 'utf8');
        const summary = JSON.parse(coverageSummary);
        
        console.log('📊 Coverage Summary:');
        if (summary.total) {
          console.log(`  Lines: ${summary.total.lines.pct}%`);
          console.log(`  Functions: ${summary.total.functions.pct}%`);
          console.log(`  Branches: ${summary.total.branches.pct}%`);
          console.log(`  Statements: ${summary.total.statements.pct}%`);
        }
      } catch (error) {
        console.log('  (Coverage summary not available)');
      }
    }
  } catch (error) {
    console.warn('Could not generate coverage report:', error.message);
  }
}

async function createGitHubReport(testSummary) {
  try {
    const report = `# 🧪 Cross-Browser Test Results

## Summary
- **Total Tests:** ${testSummary.results?.total || 'Unknown'}
- **Passed:** ${testSummary.results?.passed || 0} ✅
- **Failed:** ${testSummary.results?.failed || 0} ${(testSummary.results?.failed || 0) > 0 ? '❌' : ''}
- **Skipped:** ${testSummary.results?.skipped || 0} ⏭️
- **Duration:** ${testSummary.duration}

## Environment
- **Platform:** ${testSummary.environment.platform}
- **Architecture:** ${testSummary.environment.arch}
- **Node.js:** ${testSummary.environment.node_version}
- **CI:** ${testSummary.environment.ci ? 'Yes' : 'No'}

## Test Session
- **Session ID:** ${testSummary.sessionId}
- **Start Time:** ${testSummary.startTime}
- **End Time:** ${testSummary.endTime}

---
*Generated by TrustformeRS WASM Cross-Browser Test Suite*
`;

    await fs.writeFile('github-test-report.md', report);
    
    // Set GitHub Actions output
    if (process.env.GITHUB_OUTPUT) {
      const output = `test_summary<<EOF
${report}
EOF
`;
      await fs.appendFile(process.env.GITHUB_OUTPUT, output);
    }
    
    console.log('📝 GitHub Actions report created');
    
  } catch (error) {
    console.warn('Could not create GitHub report:', error.message);
  }
}

export default globalTeardown;