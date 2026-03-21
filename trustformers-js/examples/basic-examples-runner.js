/**
 * Basic Examples Runner
 * 
 * This script runs all the basic TrustformeRS examples in sequence.
 * It demonstrates the core capabilities of the library across different tasks.
 */

import { runTextClassification } from './text-classification-basic.js';
import { runQuestionAnswering } from './question-answering-basic.js';
import { runTextGeneration } from './text-generation-basic.js';
import { runSentimentAnalysis } from './sentiment-analysis-basic.js';

async function runAllBasicExamples() {
  console.log('🚀 TrustformeRS Basic Examples Runner\n');
  console.log('This demo showcases the core capabilities of TrustformeRS JavaScript API\n');
  console.log('═'.repeat(80));
  
  const examples = [
    {
      name: 'Text Classification',
      description: 'Classify text into predefined categories',
      runner: runTextClassification,
      emoji: '📊'
    },
    {
      name: 'Question Answering',
      description: 'Extract answers from context using natural language questions',
      runner: runQuestionAnswering,
      emoji: '❓'
    },
    {
      name: 'Text Generation',
      description: 'Generate text continuations from prompts',
      runner: runTextGeneration,
      emoji: '✍️'
    },
    {
      name: 'Sentiment Analysis',
      description: 'Analyze emotional tone and sentiment in text',
      runner: runSentimentAnalysis,
      emoji: '😊'
    }
  ];
  
  let successCount = 0;
  let totalTime = 0;
  const overallTimer = Date.now();
  
  for (let i = 0; i < examples.length; i++) {
    const example = examples[i];
    
    console.log(`\n${example.emoji} Example ${i + 1}/${examples.length}: ${example.name}`);
    console.log(`Description: ${example.description}`);
    console.log('─'.repeat(60));
    
    try {
      const startTime = Date.now();
      await example.runner();
      const exampleTime = Date.now() - startTime;
      totalTime += exampleTime;
      successCount++;
      
      console.log(`✅ ${example.name} completed successfully in ${exampleTime}ms`);
      
      if (i < examples.length - 1) {
        console.log('\n⏳ Preparing next example...');
        await new Promise(resolve => setTimeout(resolve, 1000)); // Brief pause
      }
      
    } catch (error) {
      console.error(`❌ ${example.name} failed:`, error.message);
    }
    
    console.log('═'.repeat(80));
  }
  
  // Summary
  const overallTime = Date.now() - overallTimer;
  console.log('\n🎯 Examples Summary:');
  console.log(`✅ Successful: ${successCount}/${examples.length}`);
  console.log(`⏱️  Total execution time: ${overallTime}ms`);
  console.log(`📊 Average time per example: ${Math.round(totalTime / successCount)}ms`);
  
  if (successCount === examples.length) {
    console.log('\n🎉 All examples completed successfully!');
    console.log('\n💡 Next Steps:');
    console.log('• Try modifying the examples with your own data');
    console.log('• Explore the advanced examples for more complex use cases');
    console.log('• Check the API documentation for additional features');
    console.log('• Experiment with different model configurations');
  } else {
    console.log(`\n⚠️  ${examples.length - successCount} example(s) failed. Check the logs above for details.`);
  }
  
  console.log('\n📚 Learn More:');
  console.log('• Documentation: ../docs/api-reference.md');
  console.log('• Migration Guide: ../docs/migration-guide.md');
  console.log('• Advanced Examples: ./enhanced-tensor-operations.js');
  console.log('• Performance Demo: ./performance-optimization-demo.html');
  
  console.log('\n🔗 TrustformeRS Resources:');
  console.log('• GitHub: https://github.com/your-org/trustformers');
  console.log('• Documentation: https://trustformers.dev');
  console.log('• Community: https://discord.gg/trustformers');
  
  console.log('\nThank you for exploring TrustformeRS! 🙏');
}

// Menu-driven interface for selective example running
async function runInteractiveMenu() {
  console.log('🎛️  TrustformeRS Interactive Examples Menu\n');
  
  console.log('Available Examples:');
  console.log('1. 📊 Text Classification');
  console.log('2. ❓ Question Answering');
  console.log('3. ✍️  Text Generation');
  console.log('4. 😊 Sentiment Analysis');
  console.log('5. 🚀 Run All Examples');
  console.log('6. ❌ Exit\n');
  
  // Note: In a real implementation, you would use readline or similar
  // For this example, we'll just run all examples
  console.log('Running all examples (interactive menu would require readline in real implementation)...\n');
  await runAllBasicExamples();
}

// Check command line arguments for options
const args = process.argv.slice(2);

if (args.includes('--help') || args.includes('-h')) {
  console.log('TrustformeRS Basic Examples Runner\n');
  console.log('Usage: node basic-examples-runner.js [options]\n');
  console.log('Options:');
  console.log('  --help, -h          Show this help message');
  console.log('  --interactive, -i   Run in interactive mode');
  console.log('  --classification    Run only text classification example');
  console.log('  --qa               Run only question answering example');
  console.log('  --generation       Run only text generation example');
  console.log('  --sentiment        Run only sentiment analysis example');
  console.log('  (no args)          Run all examples');
  
} else if (args.includes('--interactive') || args.includes('-i')) {
  runInteractiveMenu();
  
} else if (args.includes('--classification')) {
  runTextClassification();
  
} else if (args.includes('--qa')) {
  runQuestionAnswering();
  
} else if (args.includes('--generation')) {
  runTextGeneration();
  
} else if (args.includes('--sentiment')) {
  runSentimentAnalysis();
  
} else {
  // Default: run all examples
  runAllBasicExamples();
}

export { 
  runAllBasicExamples, 
  runInteractiveMenu 
};