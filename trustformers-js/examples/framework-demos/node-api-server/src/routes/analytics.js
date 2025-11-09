/**
 * Analytics API Routes
 * Provides comprehensive text analysis endpoints using TrustformeRS
 */

import express from 'express';
import { body, query, validationResult } from 'express-validator';
import rateLimit from 'express-rate-limit';
import multer from 'multer';

import logger from '../utils/logger.js';
import {
  analyzeSentiment,
  classifyText,
  generateText,
  answerQuestion,
  summarizeText,
} from '../utils/trustformers.js';

const router = express.Router();

// Configure multer for file uploads
const upload = multer({
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
    files: 1,
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['text/plain', 'application/json', 'text/csv'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only text files are allowed.'));
    }
  },
});

// Rate limiting for expensive operations
const heavyOperationLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 20, // Limit each IP to 20 requests per windowMs for heavy operations
  message: {
    error: 'Too many analysis requests. Please try again later.',
    retryAfter: '15 minutes',
  },
});

// Validation middleware
const validateText = [
  body('text')
    .isString()
    .isLength({ min: 1, max: 10000 })
    .withMessage('Text must be a string between 1 and 10,000 characters'),
];

const validateBatchTexts = [
  body('texts')
    .isArray({ min: 1, max: 100 })
    .withMessage('Texts must be an array with 1-100 items'),
  body('texts.*')
    .isString()
    .isLength({ min: 1, max: 5000 })
    .withMessage('Each text must be a string between 1 and 5,000 characters'),
];

const validateClassification = [
  ...validateText,
  body('labels')
    .optional()
    .isArray({ min: 1, max: 50 })
    .withMessage('Labels must be an array with 1-50 items'),
  body('labels.*')
    .optional()
    .isString()
    .isLength({ min: 1, max: 100 })
    .withMessage('Each label must be a string between 1 and 100 characters'),
];

const validateGeneration = [
  body('prompt')
    .isString()
    .isLength({ min: 1, max: 1000 })
    .withMessage('Prompt must be a string between 1 and 1,000 characters'),
  body('maxLength')
    .optional()
    .isInt({ min: 10, max: 500 })
    .withMessage('Max length must be between 10 and 500'),
  body('temperature')
    .optional()
    .isFloat({ min: 0.1, max: 2.0 })
    .withMessage('Temperature must be between 0.1 and 2.0'),
];

const validateQuestionAnswering = [
  body('question')
    .isString()
    .isLength({ min: 1, max: 500 })
    .withMessage('Question must be a string between 1 and 500 characters'),
  body('context')
    .isString()
    .isLength({ min: 1, max: 5000 })
    .withMessage('Context must be a string between 1 and 5,000 characters'),
];

const validateSummarization = [
  body('text')
    .isString()
    .isLength({ min: 50, max: 10000 })
    .withMessage('Text must be a string between 50 and 10,000 characters'),
  body('maxLength')
    .optional()
    .isInt({ min: 20, max: 300 })
    .withMessage('Max length must be between 20 and 300'),
  body('minLength')
    .optional()
    .isInt({ min: 10, max: 150 })
    .withMessage('Min length must be between 10 and 150'),
];

// Error handling middleware
const handleValidationErrors = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logger.validation(errors.array(), {
      endpoint: req.path,
      method: req.method,
      requestId: req.requestId,
    });
    
    return res.status(400).json({
      success: false,
      error: 'Validation failed',
      details: errors.array(),
      timestamp: new Date().toISOString(),
    });
  }
  next();
};

/**
 * @swagger
 * /analytics/sentiment:
 *   post:
 *     summary: Analyze text sentiment
 *     description: Analyzes the sentiment of the provided text using machine learning
 *     tags: [Analytics]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - text
 *             properties:
 *               text:
 *                 type: string
 *                 minLength: 1
 *                 maxLength: 10000
 *                 description: Text to analyze
 *                 example: "I love this new feature! It works amazingly well."
 *               returnAllScores:
 *                 type: boolean
 *                 description: Return confidence scores for all labels
 *                 default: false
 *     responses:
 *       200:
 *         description: Successful sentiment analysis
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 result:
 *                   type: object
 *                   properties:
 *                     label:
 *                       type: string
 *                       enum: [POSITIVE, NEGATIVE, NEUTRAL]
 *                     score:
 *                       type: number
 *                       minimum: 0
 *                       maximum: 1
 *                 metadata:
 *                   type: object
 *                   properties:
 *                     duration:
 *                       type: string
 *                     textLength:
 *                       type: integer
 *                     model:
 *                       type: string
 *       400:
 *         description: Invalid input
 *       429:
 *         description: Rate limit exceeded
 *       500:
 *         description: Internal server error
 */
router.post('/sentiment', validateText, handleValidationErrors, async (req, res) => {
  try {
    const { text, returnAllScores = false } = req.body;
    const requestId = req.requestId;
    
    logger.info('Processing sentiment analysis request', {
      requestId,
      textLength: text.length,
      returnAllScores,
    });
    
    const result = await analyzeSentiment(text, { returnAllScores });
    
    res.json({
      success: true,
      data: result.result,
      metadata: {
        ...result.metadata,
        requestId,
        endpoint: 'sentiment',
        timestamp: new Date().toISOString(),
      },
    });
    
  } catch (error) {
    logger.errorWithContext(error, {
      endpoint: 'sentiment',
      requestId: req.requestId,
      textLength: req.body.text?.length,
    });
    
    res.status(500).json({
      success: false,
      error: 'Sentiment analysis failed',
      message: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
      requestId: req.requestId,
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * @swagger
 * /analytics/classify:
 *   post:
 *     summary: Classify text into categories
 *     description: Classifies text into provided categories or performs zero-shot classification
 *     tags: [Analytics]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - text
 *             properties:
 *               text:
 *                 type: string
 *                 description: Text to classify
 *               labels:
 *                 type: array
 *                 items:
 *                   type: string
 *                 description: Possible classification labels
 *                 example: ["technology", "business", "sports", "politics"]
 *     responses:
 *       200:
 *         description: Successful text classification
 */
router.post('/classify', validateClassification, handleValidationErrors, async (req, res) => {
  try {
    const { text, labels = [], ...options } = req.body;
    const requestId = req.requestId;
    
    logger.info('Processing text classification request', {
      requestId,
      textLength: text.length,
      labelsCount: labels.length,
    });
    
    const result = await classifyText(text, labels, options);
    
    res.json({
      success: true,
      data: result.result,
      metadata: {
        ...result.metadata,
        requestId,
        endpoint: 'classify',
        timestamp: new Date().toISOString(),
      },
    });
    
  } catch (error) {
    logger.errorWithContext(error, {
      endpoint: 'classify',
      requestId: req.requestId,
      textLength: req.body.text?.length,
    });
    
    res.status(500).json({
      success: false,
      error: 'Text classification failed',
      message: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
      requestId: req.requestId,
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * @swagger
 * /analytics/generate:
 *   post:
 *     summary: Generate text from prompt
 *     description: Generates text continuation based on the provided prompt
 *     tags: [Analytics]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - prompt
 *             properties:
 *               prompt:
 *                 type: string
 *                 description: Text prompt for generation
 *               maxLength:
 *                 type: integer
 *                 minimum: 10
 *                 maximum: 500
 *                 default: 100
 *               temperature:
 *                 type: number
 *                 minimum: 0.1
 *                 maximum: 2.0
 *                 default: 0.7
 */
router.post('/generate', heavyOperationLimiter, validateGeneration, handleValidationErrors, async (req, res) => {
  try {
    const { prompt, ...options } = req.body;
    const requestId = req.requestId;
    
    logger.info('Processing text generation request', {
      requestId,
      promptLength: prompt.length,
      options,
    });
    
    const result = await generateText(prompt, options);
    
    res.json({
      success: true,
      data: result.result,
      metadata: {
        ...result.metadata,
        requestId,
        endpoint: 'generate',
        timestamp: new Date().toISOString(),
      },
    });
    
  } catch (error) {
    logger.errorWithContext(error, {
      endpoint: 'generate',
      requestId: req.requestId,
      promptLength: req.body.prompt?.length,
    });
    
    res.status(500).json({
      success: false,
      error: 'Text generation failed',
      message: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
      requestId: req.requestId,
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * @swagger
 * /analytics/question-answer:
 *   post:
 *     summary: Answer questions based on context
 *     description: Answers questions using the provided context
 *     tags: [Analytics]
 */
router.post('/question-answer', validateQuestionAnswering, handleValidationErrors, async (req, res) => {
  try {
    const { question, context, ...options } = req.body;
    const requestId = req.requestId;
    
    logger.info('Processing question answering request', {
      requestId,
      questionLength: question.length,
      contextLength: context.length,
    });
    
    const result = await answerQuestion(question, context, options);
    
    res.json({
      success: true,
      data: result.result,
      metadata: {
        ...result.metadata,
        requestId,
        endpoint: 'question-answer',
        timestamp: new Date().toISOString(),
      },
    });
    
  } catch (error) {
    logger.errorWithContext(error, {
      endpoint: 'question-answer',
      requestId: req.requestId,
      questionLength: req.body.question?.length,
      contextLength: req.body.context?.length,
    });
    
    res.status(500).json({
      success: false,
      error: 'Question answering failed',
      message: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
      requestId: req.requestId,
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * @swagger
 * /analytics/summarize:
 *   post:
 *     summary: Summarize text
 *     description: Generates a concise summary of the provided text
 *     tags: [Analytics]
 */
router.post('/summarize', heavyOperationLimiter, validateSummarization, handleValidationErrors, async (req, res) => {
  try {
    const { text, ...options } = req.body;
    const requestId = req.requestId;
    
    logger.info('Processing text summarization request', {
      requestId,
      textLength: text.length,
      options,
    });
    
    const result = await summarizeText(text, options);
    
    res.json({
      success: true,
      data: result.result,
      metadata: {
        ...result.metadata,
        requestId,
        endpoint: 'summarize',
        timestamp: new Date().toISOString(),
      },
    });
    
  } catch (error) {
    logger.errorWithContext(error, {
      endpoint: 'summarize',
      requestId: req.requestId,
      textLength: req.body.text?.length,
    });
    
    res.status(500).json({
      success: false,
      error: 'Text summarization failed',
      message: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
      requestId: req.requestId,
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * @swagger
 * /analytics/batch:
 *   post:
 *     summary: Batch process multiple texts
 *     description: Processes multiple texts with the same operation for efficiency
 *     tags: [Analytics]
 */
router.post('/batch', heavyOperationLimiter, validateBatchTexts, handleValidationErrors, async (req, res) => {
  try {
    const { texts, operation = 'sentiment', ...options } = req.body;
    const requestId = req.requestId;
    
    logger.info('Processing batch analysis request', {
      requestId,
      textsCount: texts.length,
      operation,
      totalLength: texts.reduce((sum, text) => sum + text.length, 0),
    });
    
    const startTime = Date.now();
    const results = [];
    
    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];
      let result;
      
      switch (operation) {
        case 'sentiment':
          result = await analyzeSentiment(text, options);
          break;
        case 'classify':
          result = await classifyText(text, options.labels || [], options);
          break;
        case 'summarize':
          result = await summarizeText(text, options);
          break;
        default:
          throw new Error(`Unsupported batch operation: ${operation}`);
      }
      
      results.push({
        index: i,
        text: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
        result: result.result,
      });
    }
    
    const duration = Date.now() - startTime;
    
    res.json({
      success: true,
      data: {
        results,
        summary: {
          total: texts.length,
          successful: results.length,
          failed: texts.length - results.length,
        },
      },
      metadata: {
        requestId,
        endpoint: 'batch',
        operation,
        duration: `${duration}ms`,
        textsCount: texts.length,
        timestamp: new Date().toISOString(),
      },
    });
    
  } catch (error) {
    logger.errorWithContext(error, {
      endpoint: 'batch',
      requestId: req.requestId,
      textsCount: req.body.texts?.length,
      operation: req.body.operation,
    });
    
    res.status(500).json({
      success: false,
      error: 'Batch processing failed',
      message: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
      requestId: req.requestId,
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * @swagger
 * /analytics/analyze-file:
 *   post:
 *     summary: Analyze uploaded file
 *     description: Analyzes text content from uploaded file
 *     tags: [Analytics]
 */
router.post('/analyze-file', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No file uploaded',
        timestamp: new Date().toISOString(),
      });
    }
    
    const text = req.file.buffer.toString('utf-8');
    const { operation = 'sentiment', ...options } = req.body;
    const requestId = req.requestId;
    
    logger.info('Processing file analysis request', {
      requestId,
      filename: req.file.originalname,
      fileSize: req.file.size,
      textLength: text.length,
      operation,
    });
    
    let result;
    switch (operation) {
      case 'sentiment':
        result = await analyzeSentiment(text, options);
        break;
      case 'classify':
        result = await classifyText(text, JSON.parse(options.labels || '[]'), options);
        break;
      case 'summarize':
        result = await summarizeText(text, options);
        break;
      default:
        throw new Error(`Unsupported file analysis operation: ${operation}`);
    }
    
    res.json({
      success: true,
      data: result.result,
      metadata: {
        ...result.metadata,
        requestId,
        endpoint: 'analyze-file',
        filename: req.file.originalname,
        fileSize: req.file.size,
        operation,
        timestamp: new Date().toISOString(),
      },
    });
    
  } catch (error) {
    logger.errorWithContext(error, {
      endpoint: 'analyze-file',
      requestId: req.requestId,
      filename: req.file?.originalname,
      fileSize: req.file?.size,
    });
    
    res.status(500).json({
      success: false,
      error: 'File analysis failed',
      message: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
      requestId: req.requestId,
      timestamp: new Date().toISOString(),
    });
  }
});

export default router;