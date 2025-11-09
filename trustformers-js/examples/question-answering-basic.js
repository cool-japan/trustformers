/**
 * Basic Question Answering Example
 * 
 * This example demonstrates how to use TrustformeRS for question answering tasks.
 * It shows how to extract answers from a given context using BERT-based models.
 */

import { 
  initialize, 
  createModel, 
  createTokenizer,
  Pipeline,
  utils 
} from '../src/index.js';

async function runQuestionAnswering() {
  try {
    console.log('🤖 Starting Question Answering Example\n');
    
    // Initialize TrustformeRS
    console.log('Initializing TrustformeRS...');
    await initialize({
      wasmPath: '../pkg/trustformers_wasm_bg.wasm'
    });
    console.log('✅ TrustformeRS initialized\n');
    
    // Create BERT model for question answering
    console.log('Creating BERT model for Q&A...');
    const model = createModel('bert_base');
    const tokenizer = createTokenizer('wordpiece');
    
    // Load vocabulary with Q&A relevant terms
    const vocab = {
      '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
      'what': 5, 'where': 6, 'when': 7, 'who': 8, 'why': 9, 'how': 10,
      'is': 11, 'are': 12, 'was': 13, 'were': 14, 'will': 15, 'can': 16,
      'the': 17, 'a': 18, 'an': 19, 'in': 20, 'on': 21, 'at': 22, 'by': 23,
      'for': 24, 'with': 25, 'of': 26, 'to': 27, 'from': 28, 'about': 29,
      'paris': 30, 'france': 31, 'capital': 32, 'city': 33, 'country': 34,
      'europe': 35, 'located': 36, 'known': 37, 'famous': 38, 'tower': 39,
      'eiffel': 40, 'louvre': 41, 'museum': 42, 'seine': 43, 'river': 44,
      'population': 45, 'million': 46, 'language': 47, 'french': 48,
      'einstein': 49, 'albert': 50, 'physicist': 51, 'theory': 52,
      'relativity': 53, 'born': 54, 'germany': 55, 'nobel': 56, 'prize': 57,
      'science': 58, 'mathematics': 59, 'formula': 60, 'energy': 61,
      'mass': 62, 'light': 63, 'speed': 64, 'famous': 65, 'equation': 66,
      'published': 67, 'work': 68, 'research': 69, 'university': 70
    };
    
    tokenizer.load_vocab(vocab);
    console.log(`✅ Model and tokenizer ready (vocab size: ${tokenizer.vocab_size})\n`);
    
    // Create Q&A pipeline
    const qaSystem = Pipeline.questionAnswering(model, tokenizer);
    
    // Sample contexts and questions
    const qaExamples = [
      {
        context: "Paris is the capital and largest city of France. It is located in the north-central part of the country along the river Seine. Paris is known for its museums, architecture, and cultural landmarks including the Eiffel Tower and the Louvre Museum. The city has a population of over 2 million people.",
        questions: [
          "What is the capital of France?",
          "Where is Paris located?",
          "What is Paris known for?",
          "What is the population of Paris?"
        ]
      },
      {
        context: "Albert Einstein was a theoretical physicist born in Germany in 1879. He is best known for his theory of relativity and the famous equation E=mc². Einstein won the Nobel Prize in Physics in 1921 for his work on the photoelectric effect. His research revolutionized our understanding of space, time, and energy.",
        questions: [
          "Who was Albert Einstein?",
          "When was Einstein born?",
          "What is Einstein's famous equation?",
          "Why did Einstein win the Nobel Prize?"
        ]
      }
    ];
    
    console.log('❓ Running Question Answering Examples:\n');
    
    // Process each context and its questions
    for (let i = 0; i < qaExamples.length; i++) {
      const example = qaExamples[i];
      console.log(`📖 Context ${i + 1}:`);
      console.log(`"${example.context}"\n`);
      
      // Answer each question for this context
      for (let j = 0; j < example.questions.length; j++) {
        const question = example.questions[j];
        console.log(`   ❓ Question: "${question}"`);
        
        try {
          const timer = utils.timer(`qa_${i}_${j}`);
          const answer = await qaSystem.answer(question, example.context);
          const elapsed = timer.elapsed();
          
          console.log(`   💡 Answer: "${answer.answer}"`);
          console.log(`   🎯 Confidence: ${(answer.score * 100).toFixed(1)}%`);
          console.log(`   📍 Position: ${answer.start}-${answer.end}`);
          console.log(`   ⏱️  Time: ${elapsed.toFixed(2)}ms\n`);
          
          answer.free();
        } catch (error) {
          console.log(`   ❌ Failed to answer: ${error.message}\n`);
        }
      }
      
      console.log('─'.repeat(60) + '\n');
    }
    
    // Interactive Q&A demonstration
    console.log('🔄 Interactive Q&A Demonstration:\n');
    
    const interactiveContext = "The TrustformeRS library is a comprehensive machine learning framework written in Rust. It provides high-performance tensor operations, neural network implementations, and support for transformer architectures. The library includes JavaScript bindings for web deployment, Python bindings for data science workflows, and mobile deployment capabilities for iOS and Android.";
    
    const interactiveQuestions = [
      "What is TrustformeRS?",
      "What language is TrustformeRS written in?",
      "What platforms does TrustformeRS support?",
      "What architectures does it support?"
    ];
    
    console.log(`📖 Context: "${interactiveContext}"\n`);
    
    const batchTimer = utils.timer('interactive_qa');
    let totalQuestions = 0;
    
    for (const question of interactiveQuestions) {
      try {
        console.log(`❓ "${question}"`);
        const answer = await qaSystem.answer(question, interactiveContext);
        console.log(`💡 "${answer.answer}" (${(answer.score * 100).toFixed(1)}%)\n`);
        answer.free();
        totalQuestions++;
      } catch (error) {
        console.log(`❌ Error: ${error.message}\n`);
      }
    }
    
    const totalTime = batchTimer.elapsed();
    console.log(`📊 Answered ${totalQuestions} questions in ${totalTime.toFixed(2)}ms`);
    console.log(`⚡ Average time per question: ${(totalTime / totalQuestions).toFixed(2)}ms\n`);
    
    console.log('💡 Q&A Tips:');
    console.log('• Provide clear, specific questions for better results');
    console.log('• Include relevant context that contains the answer');
    console.log('• Higher confidence scores indicate more reliable answers');
    console.log('• Check the position range to see where the answer was found');
    
    // Cleanup
    console.log('\n🧹 Cleaning up resources...');
    qaSystem.free();
    tokenizer.free();
    model.free();
    
    console.log('✅ Question Answering Example completed successfully!');
    
  } catch (error) {
    console.error('❌ Error in question answering example:', error);
    console.error('Stack trace:', error.stack);
  }
}

// Check if running directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runQuestionAnswering();
}

export { runQuestionAnswering };