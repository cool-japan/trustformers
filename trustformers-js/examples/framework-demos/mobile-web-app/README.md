# TrustformeRS Mobile Web App Demo

A comprehensive Progressive Web App (PWA) demonstrating advanced AI capabilities powered by TrustformeRS machine learning technology. This mobile-first application showcases real-time text analysis, conversational AI, computer vision, and voice recognition in a native-like mobile experience.

## 🚀 Features

### 🤖 AI Chat Assistant
- **Real-time Conversation**: Engage with an AI assistant powered by state-of-the-art language models
- **Context Awareness**: Maintains conversation history for more meaningful interactions
- **Personality Modes**: Choose from professional, friendly, or casual AI personalities
- **Quick Actions**: Preset conversation starters for common tasks

### 📊 Text Analysis
- **Sentiment Analysis**: Real-time sentiment detection with confidence scores
- **Keyword Extraction**: Automatic identification of key terms and topics
- **Text Statistics**: Word count, sentence analysis, and reading time estimation
- **Visual Results**: Interactive charts and confidence meters

### 🎤 Voice Assistant
- **Speech Recognition**: Convert speech to text using Web Speech API
- **Voice Commands**: Natural language voice interaction with the AI
- **Text-to-Speech**: AI responses can be spoken aloud
- **Audio Visualization**: Real-time audio wave visualization

### 📷 Computer Vision
- **Camera Integration**: Real-time camera access with capture functionality
- **Image Analysis**: Object detection, text extraction, and scene description
- **File Upload**: Analyze images from device storage
- **Mobile Optimized**: Optimized for mobile camera interfaces

### 📱 Progressive Web App
- **Offline Functionality**: Service worker enables offline usage
- **App Installation**: Install as a native-like app on mobile and desktop
- **Background Sync**: Automatic synchronization when connection is restored
- **Push Notifications**: Real-time notifications support
- **App Shortcuts**: Quick access to key features from home screen

### 🎨 Mobile Experience
- **Touch Optimized**: Designed for touch interaction with proper target sizes
- **Haptic Feedback**: Tactile feedback for button presses and interactions
- **Responsive Design**: Adapts to all screen sizes and orientations
- **Safe Area Support**: Proper handling of device safe areas and notches
- **Gesture Support**: Touch gestures and long-press interactions

## 🛠 Technology Stack

- **Frontend**: Vanilla JavaScript ES6+, HTML5, CSS3
- **AI Engine**: TrustformeRS JavaScript SDK
- **PWA**: Service Worker, Web App Manifest, Background Sync
- **APIs**: Web Speech API, Camera API, Geolocation API, Web Share API
- **Storage**: LocalStorage, IndexedDB for offline data
- **Styling**: CSS Grid, Flexbox, CSS Custom Properties (Variables)

## 📋 Requirements

### Browser Support
- **Chrome**: 80+
- **Firefox**: 75+
- **Safari**: 13+
- **Edge**: 80+
- **iOS Safari**: 13+
- **Android Chrome**: 80+

### Device Features
- **Camera**: For image analysis (optional)
- **Microphone**: For voice recognition (optional)
- **HTTPS**: Required for camera/microphone access and service worker
- **JavaScript**: Must be enabled

## 🚀 Getting Started

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/trustformers/trustformers-js.git
   cd trustformers-js/examples/framework-demos/mobile-web-app
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm start
   ```

4. **Open in browser**:
   Navigate to `http://localhost:3000`

### HTTPS Development (Required for PWA features)

```bash
npm run serve:https
```
Navigate to `https://localhost:3001` (accept the self-signed certificate)

### Production Build

```bash
npm run build
```

### Deployment

```bash
npm run deploy
```

## 📱 Installation as PWA

### Desktop
1. Open the app in Chrome/Edge
2. Look for the "Install" button in the address bar
3. Click "Install" to add to your desktop

### Mobile (Android)
1. Open the app in Chrome
2. Tap the menu (⋮) button
3. Select "Add to Home screen"
4. Tap "Add" to install

### Mobile (iOS)
1. Open the app in Safari
2. Tap the Share button (□↗)
3. Select "Add to Home Screen"
4. Tap "Add" to install

## 🎯 Usage Guide

### Chat Interface
1. **Start Conversation**: Type a message in the chat input
2. **Quick Actions**: Use preset buttons for common tasks
3. **Voice Input**: Switch to voice tab for speech input
4. **Settings**: Customize AI personality and response length

### Text Analyzer
1. **Input Text**: Paste or type text to analyze
2. **Run Analysis**: Click "Analyze Text" button
3. **View Results**: See sentiment, keywords, and statistics
4. **Clear Results**: Use "Clear" to reset the analyzer

### Camera Analysis
1. **Start Camera**: Tap "Start Camera" to access device camera
2. **Capture Photo**: Take a photo for analysis
3. **Upload Image**: Alternatively, upload an image file
4. **View Analysis**: See detected objects, extracted text, and scene description

### Voice Assistant
1. **Start Listening**: Tap "Tap to Speak" button
2. **Speak Clearly**: Voice will be converted to text
3. **AI Response**: Get intelligent responses to your voice input
4. **Text-to-Speech**: AI can read responses aloud

## ⚙️ Configuration

### Settings Panel
Access via the settings button in the bottom bar:

- **Theme**: Auto, Light, or Dark mode
- **AI Settings**: Response length and personality
- **Mobile Features**: Haptic feedback, auto-save, offline mode
- **Data Management**: Clear data, export conversations

### Environment Variables
Create a `.env` file for configuration:

```env
TRUSTFORMERS_API_KEY=your_api_key_here
TRUSTFORMERS_MODEL_URL=https://models.trustformers.dev
OFFLINE_MODE=false
DEBUG_MODE=true
```

## 🔧 Development

### File Structure
```
mobile-web-app/
├── index.html          # Main HTML file
├── manifest.json       # PWA manifest
├── sw.js              # Service worker
├── css/
│   └── styles.css     # Main stylesheet
├── js/
│   └── main.js        # Application logic
├── icons/             # PWA icons
├── screenshots/       # App screenshots
└── package.json       # Dependencies and scripts
```

### Key Components

#### App Container (`TrustformersMobileApp`)
- Main application class handling initialization
- Tab navigation and state management
- Device feature integration
- Settings and data persistence

#### Service Worker (`sw.js`)
- Caching strategies for offline functionality
- Background sync for conversation data
- Push notification handling
- App update management

#### Manifest (`manifest.json`)
- PWA configuration and metadata
- App icons and theme colors
- Shortcuts and file handlers
- Platform-specific settings

### API Integration

The app integrates with TrustformeRS through the JavaScript SDK:

```javascript
// Initialize TrustformeRS
const trustformers = await TrustformersMobile.initialize({
  models: ['sentiment-analysis', 'text-generation'],
  optimization: { quantization: true, webgl: true }
});

// Generate response
const response = await trustformers.generateResponse(message);

// Analyze text
const analysis = await trustformers.analyzeText(text, {
  sentiment: true,
  keywords: true
});

// Analyze image
const imageAnalysis = await trustformers.analyzeImage(imageData, {
  objectDetection: true,
  textExtraction: true
});
```

## 🧪 Testing

### Automated Testing

```bash
# Run all tests
npm test

# Lighthouse audit
npm run test:lighthouse

# PWA compliance test
npm run test:pwa

# Security audit
npm run audit

# Bundle size check
npm run analyze
```

### Manual Testing Checklist

#### Core Functionality
- [ ] Chat interface responds to messages
- [ ] Text analyzer processes input correctly
- [ ] Camera captures and analyzes images
- [ ] Voice recognition converts speech to text
- [ ] Settings persist across sessions

#### PWA Features
- [ ] App installs as PWA
- [ ] Service worker caches resources
- [ ] Offline functionality works
- [ ] Push notifications display
- [ ] App shortcuts function

#### Mobile Features
- [ ] Touch gestures work properly
- [ ] Haptic feedback triggers
- [ ] Safe area spacing correct
- [ ] Orientation changes handled
- [ ] Camera permissions requested

#### Cross-Browser
- [ ] Chrome (desktop/mobile)
- [ ] Safari (desktop/mobile)  
- [ ] Firefox (desktop/mobile)
- [ ] Edge (desktop)

## 🔒 Security & Privacy

### Data Protection
- **Local Storage**: Conversations stored locally on device
- **No Server**: Default configuration doesn't send data to external servers
- **Encryption**: Sensitive data encrypted in storage
- **Permissions**: Explicit permission requests for camera/microphone

### Content Security Policy
```
default-src 'self'; 
script-src 'self' 'unsafe-inline'; 
style-src 'self' 'unsafe-inline'; 
img-src 'self' data: https:; 
connect-src 'self' https:
```

### Privacy Features
- **Data Export**: Users can export their data
- **Data Deletion**: Clear all data functionality
- **Offline Mode**: Fully functional without network
- **No Tracking**: No analytics or tracking by default

## 🚀 Performance

### Optimization Techniques
- **Code Splitting**: Dynamic imports for features
- **Lazy Loading**: Images and components loaded on demand  
- **Service Worker Caching**: Aggressive caching for offline use
- **Asset Minification**: CSS and JavaScript minification
- **WebP Images**: Modern image formats for smaller sizes

### Performance Metrics
- **First Contentful Paint**: < 1.5s
- **Largest Contentful Paint**: < 2.5s
- **Time to Interactive**: < 3.5s
- **Cumulative Layout Shift**: < 0.1
- **App Bundle Size**: < 200KB gzipped

### Mobile Optimizations
- **Touch Target Size**: Minimum 44px touch targets
- **Viewport Optimization**: Proper viewport meta tag
- **Font Loading**: Optimized web font loading
- **Image Optimization**: Responsive images with srcset
- **Network Adaptation**: Adaptive loading based on connection

## 🐛 Troubleshooting

### Common Issues

#### App Won't Install
- **Solution**: Ensure HTTPS connection and manifest.json is valid
- **Check**: Browser dev tools for PWA installability

#### Camera Not Working  
- **Solution**: Check browser permissions and HTTPS requirement
- **Check**: Device camera availability and browser support

#### Voice Recognition Failed
- **Solution**: Ensure microphone permissions and quiet environment
- **Check**: Browser speech recognition support

#### Offline Features Not Working
- **Solution**: Verify service worker registration and HTTPS
- **Check**: Browser dev tools Application tab

#### Performance Issues
- **Solution**: Clear browser cache and check network connection
- **Check**: Browser dev tools Performance tab

### Debug Mode

Enable debug mode in settings or URL parameter:
```
?debug=true
```

This provides:
- Detailed console logging
- Performance metrics display
- Error tracking information
- Service worker debug info

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and test thoroughly
4. Commit with descriptive messages: `git commit -m 'Add new feature'`
5. Push to your fork: `git push origin feature/new-feature`
6. Submit a pull request

### Development Guidelines

- **Code Style**: Use Prettier and ESLint configurations
- **Testing**: Add tests for new features
- **Documentation**: Update README for significant changes
- **Performance**: Ensure changes don't negatively impact performance
- **Accessibility**: Maintain WCAG 2.1 AA compliance

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../../LICENSE) file for details.

## 🙏 Acknowledgments

- **TrustformeRS Team**: For the amazing machine learning framework
- **Web APIs**: For enabling modern web capabilities
- **PWA Community**: For Progressive Web App standards and best practices
- **Open Source Contributors**: For the libraries and tools that made this possible

## 📞 Support

- **Documentation**: [https://docs.trustformers.dev](https://docs.trustformers.dev)
- **Issues**: [GitHub Issues](https://github.com/trustformers/trustformers-js/issues)
- **Discussions**: [GitHub Discussions](https://github.com/trustformers/trustformers-js/discussions)
- **Email**: [support@trustformers.dev](mailto:support@trustformers.dev)

---

**Built with ❤️ by the TrustformeRS team**

*Experience the future of AI-powered mobile applications with TrustformeRS!*