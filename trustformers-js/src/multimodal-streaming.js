/**
 * Multi-Modal Streaming Framework
 *
 * Real-time multi-modal AI processing with:
 * - Text streaming with token-level updates
 * - Image streaming with progressive generation
 * - Audio streaming with real-time transcription
 * - Video streaming with frame-level processing
 * - Cross-modal interactions (text-to-image, image-to-text, etc.)
 * - Synchronized multi-modal outputs
 * - Adaptive quality streaming
 * - Buffer management and backpressure
 */

/**
 * Base Stream Handler
 * Abstract base class for modal-specific streaming
 */
export class BaseStreamHandler {
  constructor(config = {}) {
    this.config = config;
    this.bufferSize = config.bufferSize || 1024;
    this.buffer = [];
    this.isStreaming = false;
    this.subscribers = new Set();
    this.statistics = {
      itemsProcessed: 0,
      bytesProcessed: 0,
      startTime: null,
      endTime: null,
      errors: 0,
    };
  }

  /**
   * Subscribe to stream updates
   */
  subscribe(callback) {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }

  /**
   * Emit data to all subscribers
   */
  emit(data) {
    for (const callback of this.subscribers) {
      try {
        callback(data);
      } catch (error) {
        console.error('Stream subscriber error:', error);
        this.statistics.errors++;
      }
    }
  }

  /**
   * Start streaming
   */
  async start() {
    if (this.isStreaming) {
      throw new Error('Stream already active');
    }

    this.isStreaming = true;
    this.statistics.startTime = Date.now();
    this.buffer = [];
  }

  /**
   * Stop streaming
   */
  async stop() {
    this.isStreaming = false;
    this.statistics.endTime = Date.now();
  }

  /**
   * Check buffer capacity
   */
  hasCapacity() {
    return this.buffer.length < this.bufferSize;
  }

  getStatistics() {
    const duration = this.statistics.endTime
      ? this.statistics.endTime - this.statistics.startTime
      : Date.now() - (this.statistics.startTime || Date.now());

    return {
      ...this.statistics,
      duration,
      throughput: this.statistics.itemsProcessed / (duration / 1000),
    };
  }

  /**
   * Process a single chunk of data (to be overridden by subclasses)
   */
  async process(chunk) {
    throw new Error('process() must be implemented by subclass');
  }
}

/**
 * Text Stream Handler
 * Handles streaming text generation token by token
 */
export class TextStreamHandler extends BaseStreamHandler {
  constructor(model, tokenizer, config = {}) {
    super(config);
    this.model = model;
    this.tokenizer = tokenizer;
    this.chunkSize = config.chunkSize || 1; // tokens per chunk
    this.maxLength = config.maxLength || 1000;
    this.temperature = config.temperature || 1.0;
    this.topP = config.topP || 0.9;
    this.topK = config.topK || 50;
  }

  /**
   * Process a text chunk
   */
  async process(chunk) {
    this.statistics.itemsProcessed++;
    this.statistics.bytesProcessed += chunk.text ? chunk.text.length : 0;

    return {
      modality: 'text',
      data: chunk.text || chunk,
      timestamp: Date.now(),
      processed: true,
    };
  }

  /**
   * Stream text generation
   */
  async *streamGenerate(prompt, config = {}) {
    await this.start();

    const {
      maxLength = this.maxLength,
      temperature = this.temperature,
      topP = this.topP,
      topK = this.topK,
    } = config;

    let generatedText = prompt;
    let tokenIds = await this.tokenizer.encode(prompt);

    try {
      for (let i = 0; i < maxLength && this.isStreaming; i++) {
        // Generate next token
        const nextTokenId = await this.generateNextToken(tokenIds, { temperature, topP, topK });

        // Check for end of sequence
        if (this.isEndToken(nextTokenId)) {
          break;
        }

        tokenIds.push(nextTokenId);
        const newToken = await this.tokenizer.decode([nextTokenId]);
        generatedText += newToken;

        // Emit chunk
        const chunk = {
          token: newToken,
          tokenId: nextTokenId,
          position: i,
          fullText: generatedText,
        };

        this.emit(chunk);
        yield chunk;

        this.statistics.itemsProcessed++;
        this.statistics.bytesProcessed += newToken.length;

        // Backpressure handling
        if (!this.hasCapacity()) {
          await this.waitForCapacity();
        }
      }
    } finally {
      await this.stop();
    }

    return generatedText;
  }

  async generateNextToken(tokenIds, config) {
    // Simulated token generation
    const logits = await this.model.forward(tokenIds);
    return this.sampleToken(logits, config);
  }

  sampleToken(logits, config) {
    const { temperature, topP, topK } = config;

    // Apply temperature
    const scaledLogits = logits.map(l => l / temperature);

    // Apply top-k filtering
    const topKLogits = this.filterTopK(scaledLogits, topK);

    // Apply top-p (nucleus) filtering
    const topPLogits = this.filterTopP(topKLogits, topP);

    // Sample from distribution
    const probs = this.softmax(topPLogits);
    return this.sampleFromDistribution(probs);
  }

  filterTopK(logits, k) {
    const indexed = logits.map((val, idx) => ({ val, idx }));
    indexed.sort((a, b) => b.val - a.val);

    const result = new Float32Array(logits.length).fill(-Infinity);
    for (let i = 0; i < Math.min(k, indexed.length); i++) {
      result[indexed[i].idx] = indexed[i].val;
    }

    return result;
  }

  filterTopP(logits, p) {
    const indexed = logits.map((val, idx) => ({ val, idx }));
    indexed.sort((a, b) => b.val - a.val);

    const probs = this.softmax(logits);
    let cumProb = 0;
    const result = new Float32Array(logits.length).fill(-Infinity);

    for (const item of indexed) {
      if (cumProb >= p) break;
      result[item.idx] = item.val;
      cumProb += probs[item.idx];
    }

    return result;
  }

  softmax(logits) {
    const maxLogit = Math.max(...logits.filter(l => isFinite(l)));
    const expLogits = logits.map(l => (isFinite(l) ? Math.exp(l - maxLogit) : 0));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(e => e / sumExp);
  }

  sampleFromDistribution(probs) {
    const rand = Math.random();
    let cumProb = 0;

    for (let i = 0; i < probs.length; i++) {
      cumProb += probs[i];
      if (rand < cumProb) {
        return i;
      }
    }

    return probs.length - 1;
  }

  isEndToken(tokenId) {
    // Check for end-of-sequence tokens
    const eosTokens = [0, 1, 2]; // Common EOS token IDs
    return eosTokens.includes(tokenId);
  }

  async waitForCapacity() {
    // Wait for buffer to have capacity
    return new Promise(resolve => {
      const checkCapacity = () => {
        if (this.hasCapacity()) {
          resolve();
        } else {
          setTimeout(checkCapacity, 10);
        }
      };
      checkCapacity();
    });
  }
}

/**
 * Image Stream Handler
 * Handles progressive image generation and streaming
 */
export class ImageStreamHandler extends BaseStreamHandler {
  constructor(model, config = {}) {
    super(config);
    this.model = model;
    this.progressiveSteps = config.progressiveSteps || 10;
    this.qualityLevels = config.qualityLevels || ['low', 'medium', 'high'];
    this.adaptiveQuality = config.adaptiveQuality !== false;
  }

  /**
   * Process an image chunk
   */
  async process(chunk) {
    this.statistics.itemsProcessed++;
    this.statistics.bytesProcessed += chunk.image ? chunk.image.byteLength || 0 : 0;

    return {
      modality: 'image',
      data: chunk.image || chunk,
      timestamp: Date.now(),
      quality: chunk.quality || 'medium',
      processed: true,
    };
  }

  /**
   * Stream image generation with progressive refinement
   */
  async *streamGenerate(prompt, config = {}) {
    await this.start();

    const {
      width = 512,
      height = 512,
      steps = this.progressiveSteps,
      seed = Math.random(),
    } = config;

    try {
      // Generate image progressively
      let currentImage = this.initializeImage(width, height);

      for (let step = 0; step < steps && this.isStreaming; step++) {
        // Denoise/refine image
        currentImage = await this.refineImage(currentImage, prompt, step, steps, seed);

        // Determine quality level based on step
        const qualityLevel = this.determineQuality(step, steps);

        // Encode image at appropriate quality
        const encodedImage = this.encodeImage(currentImage, qualityLevel);

        const chunk = {
          image: encodedImage,
          step: step + 1,
          totalSteps: steps,
          quality: qualityLevel,
          progress: ((step + 1) / steps) * 100,
        };

        this.emit(chunk);
        yield chunk;

        this.statistics.itemsProcessed++;
        this.statistics.bytesProcessed += encodedImage.byteLength || 0;

        // Backpressure handling
        if (!this.hasCapacity()) {
          await this.waitForCapacity();
        }
      }
    } finally {
      await this.stop();
    }
  }

  initializeImage(width, height) {
    // Initialize with noise
    return new Float32Array(width * height * 3).map(() => Math.random());
  }

  async refineImage(currentImage, prompt, step, totalSteps, seed) {
    // Simulated diffusion/refinement step
    const noiseLevel = 1 - step / totalSteps;

    return currentImage.map(pixel => {
      const noise = (Math.random() - 0.5) * noiseLevel;
      return Math.max(0, Math.min(1, pixel + noise * 0.1));
    });
  }

  determineQuality(step, totalSteps) {
    if (!this.adaptiveQuality) {
      return 'high';
    }

    const progress = step / totalSteps;

    if (progress < 0.3) return 'low';
    if (progress < 0.7) return 'medium';
    return 'high';
  }

  encodeImage(imageData, quality) {
    // Simulated image encoding
    // In real implementation, would use canvas.toBlob() or similar
    const compressionFactor = quality === 'low' ? 0.3 : quality === 'medium' ? 0.6 : 1.0;
    const dataSize = Math.floor(imageData.length * compressionFactor);

    return {
      data: imageData.slice(0, dataSize),
      width: Math.sqrt(imageData.length / 3),
      height: Math.sqrt(imageData.length / 3),
      format: 'rgba',
      quality,
      byteLength: dataSize * 4,
    };
  }

  async waitForCapacity() {
    return new Promise(resolve => {
      const checkCapacity = () => {
        if (this.hasCapacity()) {
          resolve();
        } else {
          setTimeout(checkCapacity, 10);
        }
      };
      checkCapacity();
    });
  }
}

/**
 * Audio Stream Handler
 * Handles real-time audio transcription and generation
 */
export class AudioStreamHandler extends BaseStreamHandler {
  constructor(model, config = {}) {
    super(config);
    this.model = model;
    this.sampleRate = config.sampleRate || 16000;
    this.chunkDuration = config.chunkDuration || 1.0; // seconds
    this.chunkSize = Math.floor(this.sampleRate * this.chunkDuration);
  }

  /**
   * Process an audio chunk
   */
  async process(chunk) {
    this.statistics.itemsProcessed++;
    this.statistics.bytesProcessed += chunk.audio ? chunk.audio.byteLength || chunk.audio.length * 2 : 0;

    return {
      modality: 'audio',
      data: chunk.audio || chunk,
      timestamp: Date.now(),
      sampleRate: this.sampleRate,
      processed: true,
    };
  }

  /**
   * Stream audio transcription
   */
  async *streamTranscribe(audioStream, config = {}) {
    await this.start();

    const { language = 'auto', enableTimestamps = true } = config;

    let audioBuffer = [];
    let currentTime = 0;

    try {
      for await (const audioChunk of audioStream) {
        if (!this.isStreaming) break;

        audioBuffer.push(...audioChunk);

        // Process when we have enough data
        while (audioBuffer.length >= this.chunkSize) {
          const chunk = audioBuffer.slice(0, this.chunkSize);
          audioBuffer = audioBuffer.slice(this.chunkSize);

          const transcription = await this.transcribeChunk(chunk, language);

          const result = {
            text: transcription,
            timestamp: enableTimestamps ? currentTime : null,
            duration: this.chunkDuration,
            confidence: Math.random() * 0.3 + 0.7, // Simulated confidence
          };

          this.emit(result);
          yield result;

          currentTime += this.chunkDuration;
          this.statistics.itemsProcessed++;
          this.statistics.bytesProcessed += chunk.length * 2; // 16-bit audio
        }

        // Backpressure handling
        if (!this.hasCapacity()) {
          await this.waitForCapacity();
        }
      }

      // Process remaining audio
      if (audioBuffer.length > 0) {
        const transcription = await this.transcribeChunk(audioBuffer, language);
        const result = {
          text: transcription,
          timestamp: currentTime,
          duration: audioBuffer.length / this.sampleRate,
          confidence: Math.random() * 0.3 + 0.7,
        };

        this.emit(result);
        yield result;
      }
    } finally {
      await this.stop();
    }
  }

  async transcribeChunk(audioData, language) {
    // Simulated transcription
    const words = ['hello', 'world', 'how', 'are', 'you', 'today', 'the', 'quick', 'brown', 'fox'];
    const numWords = Math.floor(Math.random() * 5) + 1;

    const transcription = [];
    for (let i = 0; i < numWords; i++) {
      transcription.push(words[Math.floor(Math.random() * words.length)]);
    }

    return transcription.join(' ');
  }

  /**
   * Stream audio generation
   */
  async *streamGenerate(text, config = {}) {
    await this.start();

    const { voice = 'default', speed = 1.0 } = config;

    try {
      // Generate audio in chunks
      const duration = text.length * 0.1; // Rough estimate
      const numChunks = Math.ceil(duration / this.chunkDuration);

      for (let i = 0; i < numChunks && this.isStreaming; i++) {
        const audioChunk = this.generateAudioChunk(text, i, numChunks, voice, speed);

        const result = {
          audio: audioChunk,
          timestamp: i * this.chunkDuration,
          duration: this.chunkDuration,
          format: 'pcm_f32le',
          sampleRate: this.sampleRate,
        };

        this.emit(result);
        yield result;

        this.statistics.itemsProcessed++;
        this.statistics.bytesProcessed += audioChunk.length * 4;

        if (!this.hasCapacity()) {
          await this.waitForCapacity();
        }
      }
    } finally {
      await this.stop();
    }
  }

  generateAudioChunk(text, chunkIndex, totalChunks, voice, speed) {
    // Simulated audio generation (sine wave)
    const audioData = new Float32Array(this.chunkSize);

    for (let i = 0; i < this.chunkSize; i++) {
      const t = (chunkIndex * this.chunkSize + i) / this.sampleRate;
      const frequency = 440 + (Math.random() - 0.5) * 100; // Varying pitch
      audioData[i] = Math.sin(2 * Math.PI * frequency * t) * 0.3;
    }

    return audioData;
  }

  async waitForCapacity() {
    return new Promise(resolve => {
      const checkCapacity = () => {
        if (this.hasCapacity()) {
          resolve();
        } else {
          setTimeout(checkCapacity, 10);
        }
      };
      checkCapacity();
    });
  }
}

/**
 * Multi-Modal Stream Coordinator
 * Synchronizes multiple modal streams
 */
export class MultiModalStreamCoordinator {
  constructor(config = {}) {
    this.config = config;
    this.streams = new Map();
    this.syncQueue = [];
    this.isCoordinating = false;
  }

  /**
   * Register a stream
   */
  registerStream(modalityType, stream) {
    this.streams.set(modalityType, {
      stream,
      latestData: null,
      timestamp: null,
    });
  }

  /**
   * Register a modality handler (alias for registerStream)
   */
  registerModality(modalityType, handler) {
    return this.registerStream(modalityType, handler);
  }

  /**
   * Add a chunk of data to be processed
   */
  async addChunk(chunk) {
    const modality = chunk.modality || 'unknown';
    const streamInfo = this.streams.get(modality);

    if (streamInfo) {
      // Process through the handler if available
      if (streamInfo.stream && streamInfo.stream.process) {
        const processed = await streamInfo.stream.process(chunk);
        this.handleStreamData(modality, processed);
        return processed;
      } else {
        // Just store the raw chunk
        this.handleStreamData(modality, chunk);
        return chunk;
      }
    } else {
      // Store in sync queue for unregistered modality
      this.syncQueue.push({ modality, chunk, timestamp: Date.now() });
      return chunk;
    }
  }

  /**
   * Start coordinated streaming
   */
  async *streamMultiModal(config = {}) {
    this.isCoordinating = true;

    const {
      syncMode = 'loose', // 'strict', 'loose', 'independent'
      maxDelay = 100, // ms
    } = config;

    try {
      // Subscribe to all streams
      const unsubscribers = [];
      for (const [modality, streamInfo] of this.streams) {
        const unsubscribe = streamInfo.stream.subscribe(data => {
          this.handleStreamData(modality, data);
        });
        unsubscribers.push(unsubscribe);
      }

      // Emit coordinated updates
      while (this.isCoordinating) {
        const coordinatedData = await this.getCoordinatedData(syncMode, maxDelay);

        if (coordinatedData) {
          yield coordinatedData;
        }

        await this.sleep(10);
      }

      // Cleanup
      unsubscribers.forEach(unsub => unsub());
    } finally {
      this.isCoordinating = false;
    }
  }

  handleStreamData(modality, data) {
    const streamInfo = this.streams.get(modality);
    if (streamInfo) {
      streamInfo.latestData = data;
      streamInfo.timestamp = Date.now();

      this.syncQueue.push({
        modality,
        data,
        timestamp: streamInfo.timestamp,
      });
    }
  }

  async getCoordinatedData(syncMode, maxDelay) {
    if (this.syncQueue.length === 0) {
      return null;
    }

    if (syncMode === 'independent') {
      // Return immediately without synchronization
      return this.syncQueue.shift();
    }

    if (syncMode === 'loose') {
      // Wait for at least one update from each modality
      const modalitiesWithData = new Set(this.syncQueue.map(item => item.modality));

      if (modalitiesWithData.size === this.streams.size) {
        return this.combineLatestData();
      }
    }

    if (syncMode === 'strict') {
      // Wait for synchronized updates within maxDelay
      const now = Date.now();
      const oldestTimestamp = Math.min(...this.syncQueue.map(item => item.timestamp));

      if (now - oldestTimestamp > maxDelay) {
        return this.combineLatestData();
      }
    }

    return null;
  }

  combineLatestData() {
    const combined = {};

    for (const [modality, streamInfo] of this.streams) {
      combined[modality] = streamInfo.latestData;
    }

    // Clear processed items from queue
    this.syncQueue = [];

    return {
      timestamp: Date.now(),
      data: combined,
    };
  }

  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  stop() {
    this.isCoordinating = false;

    // Stop all streams
    for (const [, streamInfo] of this.streams) {
      if (streamInfo.stream.stop) {
        streamInfo.stream.stop();
      }
    }
  }
}

/**
 * Create multi-modal streaming system
 */
export function createMultiModalStreaming(models, config = {}) {
  const coordinator = new MultiModalStreamCoordinator(config);

  // Create stream handlers for each modality
  if (models.text) {
    const textStream = new TextStreamHandler(models.text.model, models.text.tokenizer, config.text);
    coordinator.registerStream('text', textStream);
  }

  if (models.image) {
    const imageStream = new ImageStreamHandler(models.image.model, config.image);
    coordinator.registerStream('image', imageStream);
  }

  if (models.audio) {
    const audioStream = new AudioStreamHandler(models.audio.model, config.audio);
    coordinator.registerStream('audio', audioStream);
  }

  return coordinator;
}

// All components already exported via 'export class' and 'export function' declarations above
