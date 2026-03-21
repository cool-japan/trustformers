/**
 * Basic Sentiment Analysis Example
 * 
 * This example demonstrates sentiment analysis using TrustformeRS.
 * It shows how to analyze the emotional tone and sentiment of text.
 */

import { 
  initialize, 
  createModel, 
  createTokenizer,
  Pipeline,
  utils 
} from '../src/index.js';

async function runSentimentAnalysis() {
  try {
    console.log('😊 Starting Sentiment Analysis Example\n');
    
    // Initialize TrustformeRS
    console.log('Initializing TrustformeRS...');
    await initialize({
      wasmPath: '../pkg/trustformers_wasm_bg.wasm'
    });
    console.log('✅ TrustformeRS initialized\n');
    
    // Create BERT model for sentiment analysis
    console.log('Creating BERT model for sentiment analysis...');
    const model = createModel('bert_base');
    const tokenizer = createTokenizer('wordpiece');
    
    // Load sentiment-focused vocabulary
    const vocab = {
      '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
      // Positive sentiment words
      'love': 5, 'amazing': 6, 'excellent': 7, 'fantastic': 8, 'wonderful': 9,
      'great': 10, 'perfect': 11, 'awesome': 12, 'brilliant': 13, 'outstanding': 14,
      'incredible': 15, 'superb': 16, 'magnificent': 17, 'marvelous': 18, 'splendid': 19,
      'delighted': 20, 'thrilled': 21, 'excited': 22, 'happy': 23, 'joyful': 24,
      'pleased': 25, 'satisfied': 26, 'impressed': 27, 'beautiful': 28, 'charming': 29,
      
      // Negative sentiment words
      'hate': 30, 'terrible': 31, 'awful': 32, 'horrible': 33, 'disgusting': 34,
      'worst': 35, 'bad': 36, 'poor': 37, 'disappointing': 38, 'frustrating': 39,
      'annoying': 40, 'irritating': 41, 'unpleasant': 42, 'dreadful': 43, 'appalling': 44,
      'sad': 45, 'angry': 46, 'upset': 47, 'depressed': 48, 'miserable': 49,
      'unhappy': 50, 'dissatisfied': 51, 'disappointed': 52, 'disgusted': 53, 'furious': 54,
      
      // Neutral/common words
      'okay': 55, 'fine': 56, 'average': 57, 'normal': 58, 'standard': 59,
      'the': 60, 'is': 61, 'was': 62, 'this': 63, 'that': 64, 'it': 65,
      'i': 66, 'you': 67, 'we': 68, 'they': 69, 'he': 70, 'she': 71,
      'product': 72, 'service': 73, 'experience': 74, 'quality': 75, 'price': 76,
      'customer': 77, 'support': 78, 'delivery': 79, 'staff': 80, 'team': 81,
      'very': 82, 'really': 83, 'quite': 84, 'somewhat': 85, 'extremely': 86,
      'highly': 87, 'completely': 88, 'totally': 89, 'absolutely': 90, 'definitely': 91
    };
    
    tokenizer.load_vocab(vocab);
    console.log(`✅ Model and tokenizer ready (vocab size: ${tokenizer.vocab_size})\n`);
    
    // Create sentiment analysis pipeline
    const sentimentAnalyzer = Pipeline.textClassification(
      model, 
      tokenizer,
      ['positive', 'negative', 'neutral']
    );
    
    // Test texts with various sentiments
    const testTexts = [
      // Clearly positive
      "I absolutely love this product! It's amazing and works perfectly.",
      "Excellent quality and fantastic customer service. Highly recommended!",
      "This is the best purchase I've ever made. Absolutely thrilled!",
      "Outstanding experience! The team was wonderful and very helpful.",
      
      // Clearly negative  
      "This is terrible! Worst experience ever. Completely disappointed.",
      "Awful quality and horrible customer support. Total waste of money.",
      "I hate this product. It's frustrating and doesn't work at all.",
      "Disgusting service! The staff was rude and unprofessional.",
      
      // Neutral/mixed
      "It's okay, nothing special but does the job.",
      "Average quality for the price. Not bad, not great either.",
      "The product is fine, though delivery was a bit slow.",
      "Standard service, met my basic expectations.",
      
      // Complex sentiments
      "The product quality is excellent, but the price is too high.",
      "Great features, but the user interface is quite confusing.",
      "I love the design, but disappointed with the battery life.",
      "Fantastic idea, poorly executed. Could be much better."
    ];
    
    console.log('🔍 Analyzing Sentiment in Test Texts:\n');
    
    // Analyze each text individually
    for (let i = 0; i < testTexts.length; i++) {
      const text = testTexts[i];
      console.log(`${i + 1}. "${text}"`);
      
      try {
        const timer = utils.timer(`sentiment_${i}`);
        const result = await sentimentAnalyzer.classify(text);
        const elapsed = timer.elapsed();
        
        // Format sentiment with emoji
        const sentimentEmoji = getSentimentEmoji(result.label);
        const confidence = (result.score * 100).toFixed(1);
        
        console.log(`   ${sentimentEmoji} Sentiment: ${result.label.toUpperCase()}`);
        console.log(`   📊 Confidence: ${confidence}%`);
        console.log(`   ⏱️  Time: ${elapsed.toFixed(2)}ms`);
        
        // Show all scores for detailed analysis
        const allScores = Array.from(result.all_scores);
        console.log(`   📈 Scores: Pos: ${(allScores[0] * 100).toFixed(1)}%, Neg: ${(allScores[1] * 100).toFixed(1)}%, Neu: ${(allScores[2] * 100).toFixed(1)}%\n`);
        
        result.free();
      } catch (error) {
        console.log(`   ❌ Analysis failed: ${error.message}\n`);
      }
    }
    
    // Batch sentiment analysis
    console.log('⚡ Batch Sentiment Analysis:\n');
    
    const batchTexts = [
      "Love it!",
      "Hate it!",
      "It's okay.",
      "Absolutely fantastic!",
      "Completely terrible!"
    ];
    
    try {
      const batchTimer = utils.timer('batch_sentiment');
      const batchResults = await sentimentAnalyzer.classify_batch(batchTexts);
      const batchElapsed = batchTimer.elapsed();
      
      console.log(`✅ Analyzed ${batchTexts.length} texts in ${batchElapsed.toFixed(2)}ms`);
      console.log(`📊 Average time per text: ${(batchElapsed / batchTexts.length).toFixed(2)}ms\n`);
      
      batchResults.forEach((result, i) => {
        const emoji = getSentimentEmoji(result.label);
        console.log(`${i + 1}. "${batchTexts[i]}" → ${emoji} ${result.label.toUpperCase()} (${(result.score * 100).toFixed(1)}%)`);
        result.free();
      });
      
    } catch (error) {
      console.log(`❌ Batch analysis failed: ${error.message}`);
    }
    
    // Sentiment distribution analysis
    console.log('\n📈 Sentiment Distribution Analysis:\n');
    
    const reviewTexts = [
      "Excellent product, highly recommend!",
      "Good quality but expensive.",
      "Average experience, nothing special.",
      "Poor quality, would not buy again.",
      "Amazing service and fast delivery!",
      "Okay product, met expectations.",
      "Terrible customer support experience.",
      "Love the design and functionality!",
      "Disappointing quality for the price.",
      "Fantastic value for money!"
    ];
    
    let positiveCount = 0;
    let negativeCount = 0;
    let neutralCount = 0;
    let totalConfidence = 0;
    
    console.log('Analyzing product review sentiments...\n');
    
    for (let i = 0; i < reviewTexts.length; i++) {
      try {
        const result = await sentimentAnalyzer.classify(reviewTexts[i]);
        const emoji = getSentimentEmoji(result.label);
        
        console.log(`${i + 1}. ${emoji} ${result.label.toUpperCase()} (${(result.score * 100).toFixed(1)}%)`);
        
        // Count sentiments
        if (result.label === 'positive') positiveCount++;
        else if (result.label === 'negative') negativeCount++;
        else neutralCount++;
        
        totalConfidence += result.score;
        result.free();
      } catch (error) {
        console.log(`${i + 1}. ❌ Analysis failed`);
      }
    }
    
    console.log('\n📊 Summary Statistics:');
    console.log(`😊 Positive: ${positiveCount} (${(positiveCount / reviewTexts.length * 100).toFixed(1)}%)`);
    console.log(`😞 Negative: ${negativeCount} (${(negativeCount / reviewTexts.length * 100).toFixed(1)}%)`);
    console.log(`😐 Neutral: ${neutralCount} (${(neutralCount / reviewTexts.length * 100).toFixed(1)}%)`);
    console.log(`🎯 Average Confidence: ${(totalConfidence / reviewTexts.length * 100).toFixed(1)}%\n`);
    
    console.log('💡 Sentiment Analysis Tips:');
    console.log('• Higher confidence indicates more certain predictions');
    console.log('• Mixed sentiments may show lower confidence scores');
    console.log('• Context and sarcasm can affect accuracy');
    console.log('• Batch processing is more efficient for large datasets');
    console.log('• Consider fine-tuning models for domain-specific analysis');
    
    // Cleanup
    console.log('\n🧹 Cleaning up resources...');
    sentimentAnalyzer.free();
    tokenizer.free();
    model.free();
    
    console.log('✅ Sentiment Analysis Example completed successfully!');
    
  } catch (error) {
    console.error('❌ Error in sentiment analysis example:', error);
    console.error('Stack trace:', error.stack);
  }
}

// Helper function to get emoji for sentiment
function getSentimentEmoji(sentiment) {
  switch (sentiment.toLowerCase()) {
    case 'positive': return '😊';
    case 'negative': return '😞';
    case 'neutral': return '😐';
    default: return '🤔';
  }
}

// Check if running directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runSentimentAnalysis();
}

export { runSentimentAnalysis };