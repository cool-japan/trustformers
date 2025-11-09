# TrustformeRS React Sentiment Analyzer

A comprehensive React application demonstrating real-time sentiment analysis using TrustformeRS machine learning capabilities.

## Features

- 🎯 **Real-time Sentiment Analysis**: Analyze text sentiment as you type
- 📊 **Visual Feedback**: Color-coded results with confidence indicators
- 📈 **Analysis History**: Track previous analyses with timestamps
- 🎨 **Modern UI**: Beautiful, responsive interface with animations
- ⚡ **Performance Optimized**: Debounced input and efficient rendering
- 📱 **Mobile Responsive**: Works great on all device sizes
- ♿ **Accessible**: Screen reader friendly with proper ARIA labels

## Screenshots

```
┌─────────────────────────────────────────────────┐
│  TrustformeRS Sentiment Analyzer                │
│  Real-time sentiment analysis powered by AI     │
├─────────────────────────────────────────────────┤
│                                                 │
│  Enter text to analyze:                         │
│  ┌─────────────────────────────────────────────┐ │
│  │ I love this new feature! It works great... │ │
│  │                                             │ │
│  └─────────────────────────────────────────────┘ │
│                                                 │
│  👍 Positive               92% confident        │
│  ████████████████████░░                         │
│  Raw score: 0.9234                              │
│                                                 │
│  📊 Analysis History                            │
│  ┌─────────────────────────────────────────────┐ │
│  │ 👍 Positive (92%) - "I love this new..."   │ │
│  │ 👎 Negative (87%) - "This is disappointing" │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Technology Stack

- **React 18**: Latest React with Hooks and Concurrent Features
- **Vite**: Fast build tool with HMR and optimized bundling
- **TrustformeRS**: Machine learning inference engine
- **Lucide React**: Beautiful, customizable icons
- **Tailwind CSS**: Utility-first CSS framework (via CDN)
- **Modern JavaScript**: ES2020+ features with full browser support

## Prerequisites

- Node.js 16.0.0 or higher
- npm 7.0.0 or higher
- Modern browser with WebAssembly support

## Installation

1. **Clone and Navigate**
   ```bash
   cd trustformers-js/examples/framework-demos/react-sentiment-analyzer
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Start Development Server**
   ```bash
   npm run dev
   ```

4. **Open Browser**
   ```
   http://localhost:3000
   ```

## Available Scripts

```bash
# Start development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build locally
npm run preview

# Run linting
npm run lint

# Type checking (if TypeScript)
npm run type-check
```

## Project Structure

```
react-sentiment-analyzer/
├── public/
│   ├── trustformers-logo.svg
│   └── manifest.json
├── src/
│   ├── components/           # Reusable React components
│   ├── hooks/               # Custom React hooks
│   ├── utils/               # Utility functions
│   ├── App.jsx              # Main application component
│   ├── App.css              # Component-specific styles
│   ├── main.jsx             # Application entry point
│   └── index.css            # Global styles
├── package.json
├── vite.config.js           # Vite configuration
└── README.md
```

## Key Components

### App.jsx
The main application component handling:
- State management for text input and analysis results
- TrustformeRS model initialization
- Real-time sentiment analysis with debouncing
- Analysis history tracking
- Error handling and loading states

### TrustformeRS Integration
```javascript
import { TrustformersReact } from '@trustformers/js/frameworks';

// Initialize the model
const pipeline = await TrustformersReact.createPipeline('sentiment-analysis');

// Perform analysis
const result = await pipeline.predict(text);
```

## Configuration

### Vite Configuration
- **Fast Refresh**: Enabled for instant component updates
- **Code Splitting**: Automatic vendor and app chunk separation
- **Asset Optimization**: Inline small assets, optimize large ones
- **WebAssembly Support**: Native WASM loading for TrustformeRS

### Performance Optimizations
- **Debounced Input**: 500ms delay to prevent excessive API calls
- **Memoized Components**: Prevent unnecessary re-renders
- **Lazy Loading**: Components loaded on demand
- **Bundle Splitting**: Separate chunks for better caching

## Browser Support

| Browser | Version | Status |
|---------|---------|--------|
| Chrome  | 80+     | ✅ Full Support |
| Firefox | 80+     | ✅ Full Support |
| Safari  | 14+     | ✅ Full Support |
| Edge    | 80+     | ✅ Full Support |

## API Reference

### TrustformersReact Methods

```javascript
// Initialize the React integration
await TrustformersReact.initialize(options);

// Create a sentiment analysis pipeline
const pipeline = await TrustformersReact.createPipeline('sentiment-analysis');

// Analyze text sentiment
const result = await pipeline.predict(text);
// Returns: { label: 'positive', score: 0.92, confidence: 92 }
```

### Sentiment Result Object

```typescript
interface SentimentResult {
  label: 'positive' | 'negative' | 'neutral';
  score: number;        // Raw confidence score (0-1)
  confidence: number;   // Percentage confidence (0-100)
  text: string;         // Original analyzed text
  timestamp: string;    // ISO timestamp of analysis
}
```

## Customization

### Styling
The app uses Tailwind CSS classes that can be customized:

```css
/* Custom color scheme */
.sentiment-positive { @apply bg-emerald-50 border-emerald-200 text-emerald-700; }
.sentiment-negative { @apply bg-rose-50 border-rose-200 text-rose-700; }
.sentiment-neutral { @apply bg-slate-50 border-slate-200 text-slate-700; }
```

### Model Configuration
```javascript
const pipeline = await TrustformersReact.createPipeline('sentiment-analysis', {
  modelPath: '/models/sentiment-v2/',
  threshold: 0.7,
  maxLength: 512
});
```

## Troubleshooting

### Common Issues

**1. WebAssembly not supported**
```
Error: WebAssembly is not supported in this browser
```
Solution: Update to a modern browser version

**2. Model loading failed**
```
Error: Failed to initialize sentiment analysis model
```
Solution: Check network connection and model file accessibility

**3. Build optimization warnings**
```
Warning: Chunk size exceeds recommended limit
```
Solution: Enable chunk splitting in vite.config.js

### Development Tips

1. **Hot Reload Issues**: Clear browser cache and restart dev server
2. **Performance**: Use React DevTools Profiler to identify bottlenecks
3. **Memory Leaks**: Ensure proper cleanup of event listeners and timers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This demo is part of the TrustformeRS project and follows the same license terms.

## Related Examples

- [Vue.js Document Analyzer](../vue-document-analyzer/)
- [Node.js API Server](../node-api-server/)
- [Electron AI Assistant](../electron-ai-assistant/)
- [Mobile PWA](../mobile-pwa/)

## Support

For questions or issues:
- 📖 [TrustformeRS Documentation](https://docs.trustformers.dev)
- 💬 [Community Discussions](https://github.com/trustformers/discussions)
- 🐛 [Report Issues](https://github.com/trustformers/issues)