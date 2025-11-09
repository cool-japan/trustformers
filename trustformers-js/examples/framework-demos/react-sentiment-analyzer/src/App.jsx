import React, { useState, useEffect, useCallback } from 'react';
import { Heart, ThumbsUp, ThumbsDown, AlertCircle, Loader2, BarChart3 } from 'lucide-react';
import { TrustformersReact } from '@trustformers/js/frameworks';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [sentiment, setSentiment] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [model, setModel] = useState(null);

  // Initialize TrustformeRS model
  useEffect(() => {
    const initModel = async () => {
      try {
        const pipeline = await TrustformersReact.createPipeline('sentiment-analysis');
        setModel(pipeline);
      } catch (err) {
        console.error('Failed to initialize model:', err);
        setError('Failed to initialize sentiment analysis model');
      }
    };

    initModel();
  }, []);

  // Analyze sentiment with debouncing
  const analyzeSentiment = useCallback(async (inputText) => {
    if (!model || !inputText.trim()) {
      setSentiment(null);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await model.predict(inputText);
      const sentimentResult = {
        label: result.label,
        score: result.score,
        confidence: Math.round(result.score * 100),
        text: inputText,
        timestamp: new Date().toISOString()
      };
      
      setSentiment(sentimentResult);
      
      // Add to history
      setHistory(prev => [sentimentResult, ...prev.slice(0, 9)]);
      
    } catch (err) {
      console.error('Sentiment analysis failed:', err);
      setError('Failed to analyze sentiment. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [model]);

  // Debounced text change handler
  useEffect(() => {
    const timer = setTimeout(() => {
      if (text.trim()) {
        analyzeSentiment(text);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [text, analyzeSentiment]);

  const getSentimentIcon = (label) => {
    switch (label?.toLowerCase()) {
      case 'positive':
        return <ThumbsUp className="w-6 h-6 text-green-500" />;
      case 'negative':
        return <ThumbsDown className="w-6 h-6 text-red-500" />;
      case 'neutral':
        return <Heart className="w-6 h-6 text-gray-500" />;
      default:
        return <AlertCircle className="w-6 h-6 text-gray-400" />;
    }
  };

  const getSentimentColor = (label) => {
    switch (label?.toLowerCase()) {
      case 'positive':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'negative':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'neutral':
        return 'text-gray-600 bg-gray-50 border-gray-200';
      default:
        return 'text-gray-400 bg-gray-50 border-gray-200';
    }
  };

  const clearHistory = () => {
    setHistory([]);
    setText('');
    setSentiment(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            TrustformeRS Sentiment Analyzer
          </h1>
          <p className="text-gray-600">
            Real-time sentiment analysis powered by TrustformeRS and React
          </p>
        </div>

        {/* Main Analysis Panel */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="mb-4">
            <label htmlFor="text-input" className="block text-sm font-medium text-gray-700 mb-2">
              Enter text to analyze:
            </label>
            <textarea
              id="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type something to analyze its sentiment... For example: 'I love this new feature!' or 'This is really disappointing.'"
              className="w-full h-32 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-gray-700"
              disabled={!model}
            />
          </div>

          {/* Loading State */}
          {loading && (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="w-6 h-6 animate-spin text-blue-500 mr-2" />
              <span className="text-gray-600">Analyzing sentiment...</span>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                <span className="text-red-700">{error}</span>
              </div>
            </div>
          )}

          {/* Sentiment Result */}
          {sentiment && !loading && (
            <div className={`border rounded-lg p-4 ${getSentimentColor(sentiment.label)}`}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  {getSentimentIcon(sentiment.label)}
                  <span className="ml-2 font-semibold text-lg capitalize">
                    {sentiment.label}
                  </span>
                </div>
                <div className="text-sm font-medium">
                  {sentiment.confidence}% confident
                </div>
              </div>
              
              {/* Confidence Bar */}
              <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${
                    sentiment.label?.toLowerCase() === 'positive' ? 'bg-green-500' :
                    sentiment.label?.toLowerCase() === 'negative' ? 'bg-red-500' : 'bg-gray-500'
                  }`}
                  style={{ width: `${sentiment.confidence}%` }}
                ></div>
              </div>
              
              <p className="text-sm opacity-75 mt-2">
                Raw score: {sentiment.score?.toFixed(4)}
              </p>
            </div>
          )}

          {/* Model Status */}
          <div className="mt-4 text-sm text-gray-500">
            Status: {model ? 'Model loaded ✓' : 'Loading model...'}
          </div>
        </div>

        {/* History Panel */}
        {history.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-800 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2" />
                Analysis History
              </h2>
              <button
                onClick={clearHistory}
                className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
              >
                Clear History
              </button>
            </div>
            
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {history.map((item, index) => (
                <div
                  key={index}
                  className="border border-gray-200 rounded-lg p-3 hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center">
                      {getSentimentIcon(item.label)}
                      <span className={`ml-2 font-medium capitalize ${
                        item.label?.toLowerCase() === 'positive' ? 'text-green-600' :
                        item.label?.toLowerCase() === 'negative' ? 'text-red-600' : 'text-gray-600'
                      }`}>
                        {item.label}
                      </span>
                    </div>
                    <span className="text-sm text-gray-500">
                      {item.confidence}%
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 truncate" title={item.text}>
                    {item.text}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    {new Date(item.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500 text-sm">
          <p>Powered by TrustformeRS • Built with React • Real-time ML Inference</p>
        </div>
      </div>
    </div>
  );
}

export default App;