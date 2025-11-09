/**
 * TrustformeRS Mobile Web App
 * Advanced mobile AI assistant with PWA features
 */

import { TrustformersMobile } from '@trustformers/js/mobile';

class TrustformersMobileApp {
  constructor() {
    this.isInitialized = false;
    this.currentTab = 'chat';
    this.isOnline = navigator.onLine;
    this.conversations = this.loadData('conversations', []);
    this.settings = this.loadData('settings', this.getDefaultSettings());
    this.trustformers = null;
    this.mediaStream = null;
    this.recognition = null;
    this.synthesis = null;
    this.installPrompt = null;
    
    // Touch and gesture handling
    this.touchStart = { x: 0, y: 0, time: 0 };
    this.touchThreshold = 10;
    this.longPressTimeout = null;
    
    // Performance monitoring
    this.performanceMetrics = {
      appStart: performance.now(),
      aiLoadTime: 0,
      responseTime: 0,
      userInteractions: 0
    };
    
    this.init();
  }

  getDefaultSettings() {
    return {
      theme: 'auto',
      responseLength: 'medium',
      aiPersonality: 'friendly',
      hapticFeedback: true,
      autoSave: true,
      offlineMode: false,
      notifications: true,
      voiceEnabled: true,
      cameraEnabled: true
    };
  }

  async init() {
    try {
      console.log('🚀 Initializing TrustformeRS Mobile App...');
      
      // Initialize UI
      this.initializeUI();
      this.setupEventListeners();
      this.setupTouchGestures();
      this.setupPWAFeatures();
      
      // Initialize TrustformeRS
      await this.initializeTrustformers();
      
      // Setup device features
      await this.initializeDeviceFeatures();
      
      // Apply settings
      this.applySettings();
      
      // Hide loading screen
      this.hideLoadingScreen();
      
      this.isInitialized = true;
      this.logMetric('appLoadTime', performance.now() - this.performanceMetrics.appStart);
      
      console.log('✅ TrustformeRS Mobile App initialized successfully');
      
    } catch (error) {
      console.error('❌ Failed to initialize app:', error);
      this.showError('Failed to initialize app. Please refresh and try again.');
    }
  }

  initializeUI() {
    // Setup tab navigation
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
      button.addEventListener('click', (e) => {
        const tabId = e.target.closest('.tab-button').dataset.tab;
        this.switchTab(tabId);
        this.hapticFeedback('light');
      });
    });
    
    // Setup chat functionality
    this.setupChatInterface();
    
    // Setup analyzer functionality
    this.setupAnalyzerInterface();
    
    // Setup camera functionality
    this.setupCameraInterface();
    
    // Setup voice functionality
    this.setupVoiceInterface();
    
    // Setup settings panel
    this.setupSettingsInterface();
    
    // Monitor online/offline status
    this.setupNetworkMonitoring();
    
    // Auto-resize text areas
    this.setupAutoResize();
  }

  async initializeTrustformers() {
    const startTime = performance.now();
    
    try {
      console.log('🤖 Loading TrustformeRS models...');
      
      this.trustformers = await TrustformersMobile.initialize({
        models: ['sentiment-analysis', 'text-classification', 'text-generation'],
        optimization: {
          quantization: true,
          webgl: true,
          webgpu: this.hasWebGPU(),
          simd: this.hasSIMD()
        },
        offline: this.settings.offlineMode,
        debug: this.isDevelopment()
      });
      
      this.logMetric('aiLoadTime', performance.now() - startTime);
      this.updateAIStatus('loaded');
      
      console.log('✅ TrustformeRS models loaded successfully');
      
    } catch (error) {
      console.error('❌ Failed to load TrustformeRS:', error);
      this.updateAIStatus('error');
      throw error;
    }
  }

  async initializeDeviceFeatures() {
    // Initialize speech recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = false;
      this.recognition.interimResults = true;
      this.recognition.lang = navigator.language || 'en-US';
    }
    
    // Initialize speech synthesis
    if ('speechSynthesis' in window) {
      this.synthesis = window.speechSynthesis;
    }
    
    // Request permissions
    await this.requestPermissions();
  }

  async requestPermissions() {
    const permissions = ['camera', 'microphone', 'notifications'];
    
    for (const permission of permissions) {
      try {
        if (navigator.permissions) {
          const result = await navigator.permissions.query({ name: permission });
          console.log(`📋 Permission ${permission}:`, result.state);
        }
      } catch (error) {
        console.warn(`⚠️ Could not query ${permission} permission:`, error);
      }
    }
  }

  setupEventListeners() {
    // Bottom bar buttons
    document.getElementById('settings-button').addEventListener('click', () => {
      this.openSettings();
      this.hapticFeedback('medium');
    });
    
    document.getElementById('clear-chat').addEventListener('click', () => {
      this.clearChat();
      this.hapticFeedback('medium');
    });
    
    document.getElementById('share-button').addEventListener('click', () => {
      this.shareContent();
      this.hapticFeedback('medium');
    });
    
    // PWA install prompt
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      this.installPrompt = e;
      this.showInstallBanner();
    });
    
    document.getElementById('install-button').addEventListener('click', () => {
      this.installApp();
    });
    
    document.getElementById('dismiss-install').addEventListener('click', () => {
      this.hideInstallBanner();
    });
    
    // Keyboard events
    document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));
    
    // Visibility change
    document.addEventListener('visibilitychange', this.handleVisibilityChange.bind(this));
    
    // App state changes
    window.addEventListener('beforeunload', this.saveAppState.bind(this));
  }

  setupTouchGestures() {
    const app = document.getElementById('app');
    
    // Touch event handlers
    app.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: true });
    app.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
    app.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: true });
    
    // Long press detection
    app.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  setupPWAFeatures() {
    // Service Worker registration is handled in HTML
    
    // Handle app installation
    window.addEventListener('appinstalled', () => {
      console.log('📱 PWA was installed');
      this.hideInstallBanner();
      this.showToast('App installed successfully!', 'success');
    });
    
    // Handle updates
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        this.showToast('App updated! Refresh for new features.', 'info');
      });
    }
  }

  setupChatInterface() {
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const attachmentButton = document.getElementById('attachment-button');
    
    chatInput.addEventListener('input', () => {
      this.updateSendButton();
      this.adjustTextareaHeight(chatInput);
    });
    
    chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });
    
    sendButton.addEventListener('click', () => {
      this.sendMessage();
      this.hapticFeedback('medium');
    });
    
    attachmentButton.addEventListener('click', () => {
      this.showAttachmentOptions();
      this.hapticFeedback('light');
    });
    
    // Quick actions
    document.addEventListener('click', (e) => {
      if (e.target.classList.contains('quick-action')) {
        const action = e.target.dataset.action;
        chatInput.value = action;
        this.sendMessage();
        this.hapticFeedback('light');
      }
    });
  }

  setupAnalyzerInterface() {
    const analyzerText = document.getElementById('analyzer-text');
    const analyzeButton = document.getElementById('analyze-button');
    const clearButton = document.getElementById('clear-analyzer');
    
    analyzerText.addEventListener('input', () => {
      analyzeButton.disabled = !analyzerText.value.trim();
    });
    
    analyzeButton.addEventListener('click', () => {
      this.analyzeText();
      this.hapticFeedback('medium');
    });
    
    clearButton.addEventListener('click', () => {
      this.clearAnalyzer();
      this.hapticFeedback('light');
    });
  }

  setupCameraInterface() {
    const startCamera = document.getElementById('start-camera');
    const capturePhoto = document.getElementById('capture-photo');
    const uploadImage = document.getElementById('upload-image');
    const imageUpload = document.getElementById('image-upload');
    
    startCamera.addEventListener('click', () => {
      this.startCamera();
      this.hapticFeedback('medium');
    });
    
    capturePhoto.addEventListener('click', () => {
      this.capturePhoto();
      this.hapticFeedback('heavy');
    });
    
    uploadImage.addEventListener('click', () => {
      imageUpload.click();
      this.hapticFeedback('light');
    });
    
    imageUpload.addEventListener('change', (e) => {
      this.handleImageUpload(e.target.files[0]);
    });
    
    // Camera placeholder click
    document.getElementById('camera-placeholder').addEventListener('click', () => {
      this.startCamera();
    });
  }

  setupVoiceInterface() {
    const startListening = document.getElementById('start-listening');
    const stopListening = document.getElementById('stop-listening');
    const speakResponse = document.getElementById('speak-response');
    const copyResponse = document.getElementById('copy-response');
    
    startListening.addEventListener('click', () => {
      this.startListening();
      this.hapticFeedback('medium');
    });
    
    stopListening.addEventListener('click', () => {
      this.stopListening();
      this.hapticFeedback('medium');
    });
    
    speakResponse.addEventListener('click', () => {
      this.speakResponse();
      this.hapticFeedback('light');
    });
    
    copyResponse.addEventListener('click', () => {
      this.copyResponse();
      this.hapticFeedback('light');
    });
  }

  setupSettingsInterface() {
    const closeSettings = document.getElementById('close-settings');
    const themeSelect = document.getElementById('theme-select');
    const responseLengthSelect = document.getElementById('response-length');
    const personalitySelect = document.getElementById('ai-personality');
    const hapticToggle = document.getElementById('haptic-feedback');
    const autoSaveToggle = document.getElementById('auto-save');
    const offlineModeToggle = document.getElementById('offline-mode');
    const clearDataButton = document.getElementById('clear-data');
    const exportDataButton = document.getElementById('export-data');
    
    closeSettings.addEventListener('click', () => {
      this.closeSettings();
    });
    
    themeSelect.addEventListener('change', (e) => {
      this.updateSetting('theme', e.target.value);
    });
    
    responseLengthSelect.addEventListener('change', (e) => {
      this.updateSetting('responseLength', e.target.value);
    });
    
    personalitySelect.addEventListener('change', (e) => {
      this.updateSetting('aiPersonality', e.target.value);
    });
    
    hapticToggle.addEventListener('change', (e) => {
      this.updateSetting('hapticFeedback', e.target.checked);
    });
    
    autoSaveToggle.addEventListener('change', (e) => {
      this.updateSetting('autoSave', e.target.checked);
    });
    
    offlineModeToggle.addEventListener('change', (e) => {
      this.updateSetting('offlineMode', e.target.checked);
    });
    
    clearDataButton.addEventListener('click', () => {
      this.clearAllData();
    });
    
    exportDataButton.addEventListener('click', () => {
      this.exportData();
    });
  }

  setupNetworkMonitoring() {
    const updateOnlineStatus = () => {
      this.isOnline = navigator.onLine;
      this.updateConnectionStatus();
      
      if (!this.isOnline) {
        this.showOfflineBanner();
      } else {
        this.hideOfflineBanner();
      }
    };
    
    window.addEventListener('online', updateOnlineStatus);
    window.addEventListener('offline', updateOnlineStatus);
    
    updateOnlineStatus();
  }

  setupAutoResize() {
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
      textarea.addEventListener('input', () => {
        this.adjustTextareaHeight(textarea);
      });
    });
  }

  // Chat functionality
  async sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    try {
      // Add user message to chat
      this.addMessageToChat(message, 'user');
      input.value = '';
      this.updateSendButton();
      this.adjustTextareaHeight(input);
      
      // Show typing indicator
      this.showTypingIndicator();
      
      // Get AI response
      const startTime = performance.now();
      const response = await this.getAIResponse(message);
      this.logMetric('responseTime', performance.now() - startTime);
      
      // Hide typing indicator
      this.hideTypingIndicator();
      
      // Add AI response to chat
      this.addMessageToChat(response, 'assistant');
      
      // Save conversation
      if (this.settings.autoSave) {
        this.saveConversation();
      }
      
      // Update metrics
      this.performanceMetrics.userInteractions++;
      
    } catch (error) {
      console.error('❌ Failed to send message:', error);
      this.hideTypingIndicator();
      this.addMessageToChat('Sorry, I encountered an error. Please try again.', 'assistant', true);
    }
  }

  async getAIResponse(message) {
    if (!this.trustformers) {
      throw new Error('TrustformeRS not initialized');
    }
    
    const options = {
      personality: this.settings.aiPersonality,
      length: this.settings.responseLength,
      context: this.getConversationContext()
    };
    
    return await this.trustformers.generateResponse(message, options);
  }

  addMessageToChat(message, sender, isError = false) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    
    messageDiv.className = `${sender}-bubble`;
    if (isError) messageDiv.classList.add('error');
    
    const bubbleContent = document.createElement('div');
    bubbleContent.className = 'bubble-content';
    bubbleContent.textContent = message;
    
    messageDiv.appendChild(bubbleContent);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Add to conversation history
    this.conversations.push({
      message,
      sender,
      timestamp: new Date().toISOString(),
      isError
    });
  }

  showTypingIndicator() {
    document.getElementById('typing-indicator').classList.remove('hidden');
  }

  hideTypingIndicator() {
    document.getElementById('typing-indicator').classList.add('hidden');
  }

  updateSendButton() {
    const input = document.getElementById('chat-input');
    const button = document.getElementById('send-button');
    button.disabled = !input.value.trim();
  }

  clearChat() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML = `
      <div class="welcome-message">
        <div class="assistant-bubble">
          <div class="bubble-content">
            <p>👋 Welcome to TrustformeRS Mobile! I'm your AI assistant powered by advanced machine learning.</p>
            <p>Try asking me anything or use one of these quick actions:</p>
            <div class="quick-actions">
              <button class="quick-action" data-action="How are you feeling today?">😊 Mood Check</button>
              <button class="quick-action" data-action="Analyze this text for sentiment">📊 Text Analysis</button>
              <button class="quick-action" data-action="Help me write something">✍️ Writing Help</button>
            </div>
          </div>
        </div>
      </div>
    `;
    this.conversations = [];
    this.saveData('conversations', []);
  }

  // Analyzer functionality
  async analyzeText() {
    const textarea = document.getElementById('analyzer-text');
    const text = textarea.value.trim();
    
    if (!text) return;
    
    try {
      const results = document.getElementById('analysis-results');
      results.classList.add('hidden');
      
      // Show loading state
      const analyzeButton = document.getElementById('analyze-button');
      const originalText = analyzeButton.innerHTML;
      analyzeButton.innerHTML = '<span class="btn-icon">⏳</span>Analyzing...';
      analyzeButton.disabled = true;
      
      // Perform analysis
      const analysis = await this.trustformers.analyzeText(text, {
        sentiment: true,
        keywords: true,
        statistics: true
      });
      
      // Update UI with results
      this.displayAnalysisResults(analysis);
      results.classList.remove('hidden');
      
      // Restore button
      analyzeButton.innerHTML = originalText;
      analyzeButton.disabled = false;
      
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      this.showError('Analysis failed. Please try again.');
      
      // Restore button
      const analyzeButton = document.getElementById('analyze-button');
      analyzeButton.innerHTML = '<span class="btn-icon">🔍</span>Analyze Text';
      analyzeButton.disabled = false;
    }
  }

  displayAnalysisResults(analysis) {
    // Update sentiment
    const sentimentResult = document.getElementById('sentiment-result');
    const sentimentLabel = sentimentResult.querySelector('.sentiment-label');
    const sentimentConfidence = sentimentResult.querySelector('.sentiment-confidence');
    const sentimentBar = document.getElementById('sentiment-bar');
    
    sentimentLabel.textContent = analysis.sentiment.label;
    sentimentLabel.className = `sentiment-label ${analysis.sentiment.label.toLowerCase()}`;
    sentimentConfidence.textContent = `${Math.round(analysis.sentiment.confidence * 100)}%`;
    
    sentimentBar.className = `confidence-fill ${analysis.sentiment.label.toLowerCase()}`;
    sentimentBar.style.width = `${analysis.sentiment.confidence * 100}%`;
    
    // Update keywords
    const keywordsDisplay = document.getElementById('keywords-result');
    keywordsDisplay.innerHTML = '';
    
    analysis.keywords.slice(0, 8).forEach(keyword => {
      const tag = document.createElement('span');
      tag.className = 'keyword-tag';
      tag.innerHTML = `
        ${keyword.word}
        <span class="keyword-frequency">${keyword.frequency}</span>
      `;
      keywordsDisplay.appendChild(tag);
    });
    
    // Update statistics
    document.getElementById('word-count').textContent = analysis.statistics.wordCount;
    document.getElementById('sentence-count').textContent = analysis.statistics.sentenceCount;
    document.getElementById('reading-time').textContent = `${Math.ceil(analysis.statistics.wordCount / 200)} min`;
  }

  clearAnalyzer() {
    document.getElementById('analyzer-text').value = '';
    document.getElementById('analysis-results').classList.add('hidden');
    document.getElementById('analyze-button').disabled = true;
  }

  // Camera functionality
  async startCamera() {
    try {
      const video = document.getElementById('camera-video');
      const placeholder = document.getElementById('camera-placeholder');
      const startButton = document.getElementById('start-camera');
      const captureButton = document.getElementById('capture-photo');
      
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Back camera on mobile
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      });
      
      video.srcObject = this.mediaStream;
      placeholder.classList.add('hidden');
      video.classList.remove('hidden');
      startButton.classList.add('hidden');
      captureButton.classList.remove('hidden');
      
      console.log('📷 Camera started successfully');
      
    } catch (error) {
      console.error('❌ Failed to start camera:', error);
      this.showError('Could not access camera. Please check permissions.');
    }
  }

  async capturePhoto() {
    try {
      const video = document.getElementById('camera-video');
      const canvas = document.getElementById('camera-canvas');
      const ctx = canvas.getContext('2d');
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      await this.analyzeImage(imageData);
      
      console.log('📸 Photo captured and analyzed');
      
    } catch (error) {
      console.error('❌ Failed to capture photo:', error);
      this.showError('Failed to capture photo. Please try again.');
    }
  }

  async handleImageUpload(file) {
    if (!file) return;
    
    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        await this.analyzeImage(e.target.result);
      };
      reader.readAsDataURL(file);
      
    } catch (error) {
      console.error('❌ Failed to process image:', error);
      this.showError('Failed to process image. Please try again.');
    }
  }

  async analyzeImage(imageData) {
    try {
      const analysis = await this.trustformers.analyzeImage(imageData, {
        objectDetection: true,
        textExtraction: true,
        sceneDescription: true
      });
      
      this.displayImageAnalysis(analysis);
      
    } catch (error) {
      console.error('❌ Image analysis failed:', error);
      this.showError('Image analysis failed. Please try again.');
    }
  }

  displayImageAnalysis(analysis) {
    const resultsContainer = document.getElementById('image-results');
    const analysisContainer = document.getElementById('image-analysis');
    
    resultsContainer.innerHTML = `
      <div class="analysis-section">
        <h4>🔍 Objects Detected</h4>
        <div class="object-tags">
          ${analysis.objects.map(obj => 
            `<span class="object-tag">${obj.label} (${Math.round(obj.confidence * 100)}%)</span>`
          ).join('')}
        </div>
      </div>
      
      <div class="analysis-section">
        <h4>📝 Extracted Text</h4>
        <p class="extracted-text">${analysis.text || 'No text detected'}</p>
      </div>
      
      <div class="analysis-section">
        <h4>🖼️ Scene Description</h4>
        <p class="scene-description">${analysis.description}</p>
      </div>
    `;
    
    analysisContainer.classList.remove('hidden');
  }

  // Voice functionality
  async startListening() {
    if (!this.recognition) {
      this.showError('Speech recognition not supported on this device.');
      return;
    }
    
    try {
      const startButton = document.getElementById('start-listening');
      const stopButton = document.getElementById('stop-listening');
      const status = document.getElementById('voice-status');
      const animation = document.getElementById('voice-animation');
      
      startButton.classList.add('hidden');
      stopButton.classList.remove('hidden');
      status.querySelector('.status-text').textContent = 'Listening...';
      animation.classList.add('listening');
      
      this.recognition.start();
      
      this.recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript;
        this.displayTranscript(transcript, event.results[event.results.length - 1].isFinal);
      };
      
      this.recognition.onend = () => {
        this.stopListening();
      };
      
      this.recognition.onerror = (error) => {
        console.error('❌ Speech recognition error:', error);
        this.stopListening();
        this.showError('Speech recognition failed. Please try again.');
      };
      
    } catch (error) {
      console.error('❌ Failed to start listening:', error);
      this.showError('Failed to start voice recognition.');
    }
  }

  stopListening() {
    if (this.recognition) {
      this.recognition.stop();
    }
    
    const startButton = document.getElementById('start-listening');
    const stopButton = document.getElementById('stop-listening');
    const status = document.getElementById('voice-status');
    const animation = document.getElementById('voice-animation');
    
    startButton.classList.remove('hidden');
    stopButton.classList.add('hidden');
    status.querySelector('.status-text').textContent = 'Ready to listen';
    animation.classList.remove('listening');
  }

  displayTranscript(text, isFinal) {
    const transcriptContainer = document.getElementById('voice-transcript');
    const transcriptText = document.getElementById('transcript-text');
    
    transcriptText.textContent = text;
    transcriptContainer.classList.remove('hidden');
    
    if (isFinal) {
      this.processVoiceInput(text);
    }
  }

  async processVoiceInput(text) {
    try {
      const response = await this.getAIResponse(text);
      
      const responseContainer = document.getElementById('voice-response');
      const responseText = document.getElementById('response-text');
      
      responseText.textContent = response;
      responseContainer.classList.remove('hidden');
      
      // Auto-speak response if enabled
      if (this.settings.voiceEnabled) {
        this.speakText(response);
      }
      
    } catch (error) {
      console.error('❌ Voice processing failed:', error);
      this.showError('Failed to process voice input.');
    }
  }

  speakResponse() {
    const responseText = document.getElementById('response-text').textContent;
    this.speakText(responseText);
  }

  speakText(text) {
    if (this.synthesis && text) {
      this.synthesis.cancel(); // Stop any current speech
      
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 0.8;
      
      this.synthesis.speak(utterance);
    }
  }

  copyResponse() {
    const responseText = document.getElementById('response-text').textContent;
    
    if (navigator.clipboard) {
      navigator.clipboard.writeText(responseText).then(() => {
        this.showToast('Response copied to clipboard', 'success');
      });
    } else {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = responseText;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      this.showToast('Response copied to clipboard', 'success');
    }
  }

  // Settings functionality
  openSettings() {
    const panel = document.getElementById('settings-panel');
    panel.classList.add('open');
    
    // Update settings form with current values
    document.getElementById('theme-select').value = this.settings.theme;
    document.getElementById('response-length').value = this.settings.responseLength;
    document.getElementById('ai-personality').value = this.settings.aiPersonality;
    document.getElementById('haptic-feedback').checked = this.settings.hapticFeedback;
    document.getElementById('auto-save').checked = this.settings.autoSave;
    document.getElementById('offline-mode').checked = this.settings.offlineMode;
  }

  closeSettings() {
    const panel = document.getElementById('settings-panel');
    panel.classList.remove('open');
  }

  updateSetting(key, value) {
    this.settings[key] = value;
    this.saveData('settings', this.settings);
    this.applySettings();
  }

  applySettings() {
    // Apply theme
    const theme = this.settings.theme;
    if (theme === 'auto') {
      document.body.classList.toggle('dark', window.matchMedia('(prefers-color-scheme: dark)').matches);
    } else {
      document.body.classList.toggle('dark', theme === 'dark');
    }
  }

  clearAllData() {
    if (confirm('Are you sure you want to clear all data? This cannot be undone.')) {
      localStorage.clear();
      this.conversations = [];
      this.settings = this.getDefaultSettings();
      this.clearChat();
      this.showToast('All data cleared', 'info');
      this.hapticFeedback('heavy');
    }
  }

  exportData() {
    const data = {
      conversations: this.conversations,
      settings: this.settings,
      exportDate: new Date().toISOString(),
      version: '1.0.0'
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `trustformers-mobile-data-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    URL.revokeObjectURL(url);
    this.showToast('Data exported successfully', 'success');
  }

  // Utility functions
  switchTab(tabId) {
    // Update active tab button
    document.querySelectorAll('.tab-button').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    
    // Update active tab content
    document.querySelectorAll('.tab-content').forEach(content => {
      content.classList.toggle('active', content.id === `${tabId}-tab`);
    });
    
    this.currentTab = tabId;
    
    // Tab-specific logic
    if (tabId === 'camera' && this.mediaStream) {
      // Reset camera when switching away and back
      this.stopCamera();
    }
  }

  stopCamera() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
      
      const video = document.getElementById('camera-video');
      const placeholder = document.getElementById('camera-placeholder');
      const startButton = document.getElementById('start-camera');
      const captureButton = document.getElementById('capture-photo');
      
      video.classList.add('hidden');
      placeholder.classList.remove('hidden');
      startButton.classList.remove('hidden');
      captureButton.classList.add('hidden');
    }
  }

  adjustTextareaHeight(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
  }

  hapticFeedback(type = 'light') {
    if (this.settings.hapticFeedback && 'vibrate' in navigator) {
      const patterns = {
        light: [10],
        medium: [20],
        heavy: [30, 10, 30]
      };
      navigator.vibrate(patterns[type] || patterns.light);
    }
  }

  showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    // Style toast
    Object.assign(toast.style, {
      position: 'fixed',
      bottom: '100px',
      left: '50%',
      transform: 'translateX(-50%)',
      background: type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#3b82f6',
      color: 'white',
      padding: '12px 20px',
      borderRadius: '12px',
      fontSize: '14px',
      fontWeight: '500',
      zIndex: '10000',
      animation: 'fadeInUp 0.3s ease, fadeOut 0.3s ease 2.7s forwards'
    });
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 3000);
  }

  showError(message) {
    this.showToast(message, 'error');
  }

  hideLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    const app = document.getElementById('app');
    
    loadingScreen.style.opacity = '0';
    
    setTimeout(() => {
      loadingScreen.classList.add('hidden');
      app.classList.remove('hidden');
    }, 500);
  }

  showInstallBanner() {
    document.getElementById('install-banner').classList.remove('hidden');
  }

  hideInstallBanner() {
    document.getElementById('install-banner').classList.add('hidden');
  }

  async installApp() {
    if (this.installPrompt) {
      const result = await this.installPrompt.prompt();
      console.log('📱 Install prompt result:', result);
      this.installPrompt = null;
    }
  }

  showOfflineBanner() {
    document.getElementById('offline-banner').classList.remove('hidden');
  }

  hideOfflineBanner() {
    document.getElementById('offline-banner').classList.add('hidden');
  }

  updateConnectionStatus() {
    const statusDot = document.querySelector('.status-dot');
    statusDot.classList.toggle('offline', !this.isOnline);
  }

  updateAIStatus(status) {
    const aiIcon = document.querySelector('.ai-icon');
    const indicators = {
      loading: '🔄',
      loaded: '🧠',
      error: '❌'
    };
    aiIcon.textContent = indicators[status] || '🧠';
  }

  async shareContent() {
    const shareData = {
      title: 'TrustformeRS Mobile AI Assistant',
      text: 'Check out this amazing AI assistant powered by TrustformeRS!',
      url: window.location.href
    };
    
    if (navigator.share) {
      try {
        await navigator.share(shareData);
      } catch (error) {
        console.log('❌ Share failed:', error);
      }
    } else {
      // Fallback - copy to clipboard
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(shareData.url);
        this.showToast('Link copied to clipboard', 'success');
      }
    }
  }

  // Touch gesture handlers
  handleTouchStart(e) {
    const touch = e.touches[0];
    this.touchStart = {
      x: touch.clientX,
      y: touch.clientY,
      time: Date.now()
    };
    
    // Start long press detection
    this.longPressTimeout = setTimeout(() => {
      this.handleLongPress(e);
    }, 500);
  }

  handleTouchMove(e) {
    if (this.longPressTimeout) {
      const touch = e.touches[0];
      const deltaX = Math.abs(touch.clientX - this.touchStart.x);
      const deltaY = Math.abs(touch.clientY - this.touchStart.y);
      
      // Cancel long press if moved too much
      if (deltaX > this.touchThreshold || deltaY > this.touchThreshold) {
        clearTimeout(this.longPressTimeout);
        this.longPressTimeout = null;
      }
    }
  }

  handleTouchEnd(e) {
    if (this.longPressTimeout) {
      clearTimeout(this.longPressTimeout);
      this.longPressTimeout = null;
    }
  }

  handleLongPress(e) {
    // Handle long press gestures
    this.hapticFeedback('medium');
    console.log('🤏 Long press detected');
  }

  handleKeyboardShortcuts(e) {
    // Handle keyboard shortcuts
    if (e.ctrlKey || e.metaKey) {
      switch (e.key) {
        case 'k':
          e.preventDefault();
          this.switchTab('chat');
          break;
        case 'l':
          e.preventDefault();
          this.clearChat();
          break;
        case ',':
          e.preventDefault();
          this.openSettings();
          break;
      }
    }
  }

  handleVisibilityChange() {
    if (document.hidden) {
      // App became hidden
      this.saveAppState();
    } else {
      // App became visible
      this.updateConnectionStatus();
    }
  }

  saveAppState() {
    // Save current state before app closes
    this.saveData('conversations', this.conversations);
    this.saveData('settings', this.settings);
    this.saveData('currentTab', this.currentTab);
    this.saveData('performanceMetrics', this.performanceMetrics);
  }

  getConversationContext() {
    // Get last few messages for context
    return this.conversations
      .filter(msg => !msg.isError)
      .slice(-6)
      .map(msg => ({ role: msg.sender === 'user' ? 'user' : 'assistant', content: msg.message }));
  }

  saveConversation() {
    this.saveData('conversations', this.conversations);
  }

  // Data persistence
  saveData(key, data) {
    try {
      localStorage.setItem(`trustformers_mobile_${key}`, JSON.stringify(data));
    } catch (error) {
      console.error('❌ Failed to save data:', error);
    }
  }

  loadData(key, defaultValue = null) {
    try {
      const data = localStorage.getItem(`trustformers_mobile_${key}`);
      return data ? JSON.parse(data) : defaultValue;
    } catch (error) {
      console.error('❌ Failed to load data:', error);
      return defaultValue;
    }
  }

  // Performance and feature detection
  hasWebGPU() {
    return 'gpu' in navigator;
  }

  hasSIMD() {
    return typeof WebAssembly.SIMD !== 'undefined';
  }

  isDevelopment() {
    return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  }

  logMetric(metric, value) {
    this.performanceMetrics[metric] = value;
    console.log(`📊 ${metric}:`, value);
  }

  // Show attachment options
  showAttachmentOptions() {
    const options = [
      { text: '📷 Take Photo', action: () => this.switchTab('camera') },
      { text: '🖼️ Upload Image', action: () => document.getElementById('image-upload').click() },
      { text: '🎤 Voice Message', action: () => this.switchTab('voice') }
    ];
    
    // Create simple options menu
    const menu = document.createElement('div');
    menu.className = 'attachment-menu';
    menu.innerHTML = options.map(opt => 
      `<button class="attachment-option">${opt.text}</button>`
    ).join('');
    
    // Position and show menu
    Object.assign(menu.style, {
      position: 'fixed',
      bottom: '140px',
      left: '16px',
      right: '16px',
      background: 'var(--bg-primary)',
      border: '1px solid var(--border-color)',
      borderRadius: '12px',
      padding: '8px',
      zIndex: '1000',
      animation: 'fadeInUp 0.3s ease'
    });
    
    document.body.appendChild(menu);
    
    // Handle option clicks
    menu.addEventListener('click', (e) => {
      const index = Array.from(menu.children).indexOf(e.target);
      if (index >= 0) {
        options[index].action();
        document.body.removeChild(menu);
      }
    });
    
    // Remove menu after delay
    setTimeout(() => {
      if (menu.parentNode) {
        document.body.removeChild(menu);
      }
    }, 5000);
  }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.trustformersApp = new TrustformersMobileApp();
});

// Export for testing
export default TrustformersMobileApp;