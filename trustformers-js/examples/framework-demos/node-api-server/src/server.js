#!/usr/bin/env node

/**
 * TrustformeRS Node.js API Server
 * Demonstrates comprehensive machine learning API endpoints using TrustformeRS
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import swaggerUi from 'swagger-ui-express';
import swaggerJsdoc from 'swagger-jsdoc';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

import logger from './utils/logger.js';
import { initializeTrustformers } from './utils/trustformers.js';
import { errorHandler, notFoundHandler } from './middleware/errorHandling.js';
import { requestLogger } from './middleware/logging.js';

// Import route modules
import analyticsRoutes from './routes/analytics.js';
import modelsRoutes from './routes/models.js';
import pipelinesRoutes from './routes/pipelines.js';
import healthRoutes from './routes/health.js';

// Configuration
dotenv.config();
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const PORT = process.env.PORT || 3001;
const NODE_ENV = process.env.NODE_ENV || 'development';
const API_VERSION = process.env.API_VERSION || 'v1';

// Create Express app
const app = express();

// Trust proxy if running behind reverse proxy
if (process.env.TRUST_PROXY) {
  app.set('trust proxy', process.env.TRUST_PROXY);
}

// Swagger/OpenAPI configuration
const swaggerOptions = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'TrustformeRS API Server',
      version: '1.0.0',
      description: 'Comprehensive machine learning API powered by TrustformeRS',
      license: {
        name: 'MIT',
        url: 'https://opensource.org/licenses/MIT',
      },
      contact: {
        name: 'TrustformeRS Team',
        url: 'https://trustformers.dev',
      },
    },
    servers: [
      {
        url: `http://localhost:${PORT}/api/${API_VERSION}`,
        description: 'Development server',
      },
      {
        url: `https://api.trustformers.dev/${API_VERSION}`,
        description: 'Production server',
      },
    ],
    components: {
      securitySchemes: {
        BearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT',
        },
        ApiKeyAuth: {
          type: 'apiKey',
          in: 'header',
          name: 'X-API-Key',
        },
      },
    },
  },
  apis: ['./src/routes/*.js', './src/models/*.js'], // paths to files containing OpenAPI definitions
};

const specs = swaggerJsdoc(swaggerOptions);

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'", "https:"],
      scriptSrc: ["'self'", "https:"],
      imgSrc: ["'self'", "data:", "https:"],
    },
  },
  crossOriginEmbedderPolicy: false, // Required for SharedArrayBuffer in ML workloads
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: NODE_ENV === 'production' ? 100 : 1000, // Limit each IP to 100 requests per windowMs in production
  message: {
    error: 'Too many requests from this IP, please try again later.',
    retryAfter: '15 minutes',
  },
  standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
  legacyHeaders: false, // Disable the `X-RateLimit-*` headers
});

app.use('/api/', limiter);

// CORS configuration
const corsOptions = {
  origin: NODE_ENV === 'production' 
    ? ['https://trustformers.dev', 'https://api.trustformers.dev']
    : true, // Allow all origins in development
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key'],
  credentials: true,
  maxAge: 86400, // 24 hours
};

app.use(cors(corsOptions));

// Compression
app.use(compression({
  level: 6,
  threshold: 1024, // Only compress responses larger than 1KB
}));

// Body parsing middleware
app.use(express.json({ 
  limit: '10mb',
  type: ['application/json', 'text/plain'],
}));
app.use(express.urlencoded({ 
  extended: true, 
  limit: '10mb',
}));

// Request logging
app.use(requestLogger);

// API documentation
app.use('/docs', swaggerUi.serve, swaggerUi.setup(specs, {
  explorer: true,
  customCss: '.swagger-ui .topbar { display: none }',
  customSiteTitle: 'TrustformeRS API Documentation',
}));

// Serve static files (for demo frontend)
app.use('/static', express.static(join(__dirname, '../public')));

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    name: 'TrustformeRS API Server',
    version: '1.0.0',
    description: 'Machine learning API powered by TrustformeRS',
    documentation: '/docs',
    health: '/health',
    endpoints: {
      analytics: `/api/${API_VERSION}/analytics`,
      models: `/api/${API_VERSION}/models`,
      pipelines: `/api/${API_VERSION}/pipelines`,
    },
    features: [
      'Sentiment Analysis',
      'Text Classification',
      'Text Generation',
      'Question Answering',
      'Token Classification',
      'Document Analysis',
      'Batch Processing',
      'Model Management',
    ],
  });
});

// API routes
app.use('/health', healthRoutes);
app.use(`/api/${API_VERSION}/analytics`, analyticsRoutes);
app.use(`/api/${API_VERSION}/models`, modelsRoutes);
app.use(`/api/${API_VERSION}/pipelines`, pipelinesRoutes);

// Error handling middleware
app.use(notFoundHandler);
app.use(errorHandler);

// Global error handlers
process.on('uncaughtException', (error) => {
  logger.error('Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Graceful shutdown
const gracefulShutdown = (signal) => {
  logger.info(`Received ${signal}. Starting graceful shutdown...`);
  
  server.close(() => {
    logger.info('HTTP server closed.');
    
    // Close database connections, cleanup resources, etc.
    process.exit(0);
  });
  
  // Force close after 10 seconds
  setTimeout(() => {
    logger.error('Could not close connections in time, forcefully shutting down');
    process.exit(1);
  }, 10000);
};

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Initialize TrustformeRS and start server
async function startServer() {
  try {
    logger.info('Starting TrustformeRS API Server...');
    
    // Initialize TrustformeRS
    logger.info('Initializing TrustformeRS...');
    await initializeTrustformers();
    logger.info('TrustformeRS initialized successfully');
    
    // Start HTTP server
    const server = app.listen(PORT, () => {
      logger.info(`🚀 Server running on port ${PORT}`);
      logger.info(`📚 API Documentation: http://localhost:${PORT}/docs`);
      logger.info(`🏥 Health Check: http://localhost:${PORT}/health`);
      logger.info(`🔗 API Endpoints: http://localhost:${PORT}/api/${API_VERSION}`);
      
      if (NODE_ENV === 'development') {
        logger.info('🛠️  Development mode - CORS and rate limiting relaxed');
      }
    });
    
    // Increase server timeout for ML operations
    server.timeout = 5 * 60 * 1000; // 5 minutes
    server.keepAliveTimeout = 65000;
    server.headersTimeout = 66000;
    
    // Export server for testing
    if (NODE_ENV === 'test') {
      export { app, server };
    }
    
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Start the server if this file is run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  startServer();
}

export default app;