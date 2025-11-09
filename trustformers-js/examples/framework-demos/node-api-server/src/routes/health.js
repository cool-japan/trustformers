/**
 * Health Check Routes
 * Provides system health monitoring endpoints
 */

import express from 'express';
import logger from '../utils/logger.js';
import { healthCheck, getSystemInfo } from '../utils/trustformers.js';

const router = express.Router();

/**
 * @swagger
 * /health:
 *   get:
 *     summary: Basic health check
 *     description: Returns basic server health status
 *     tags: [Health]
 *     responses:
 *       200:
 *         description: Server is healthy
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   enum: [healthy]
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *                 uptime:
 *                   type: number
 *       503:
 *         description: Server is unhealthy
 */
router.get('/', async (req, res) => {
  try {
    const health = await healthCheck();
    const statusCode = health.status === 'healthy' ? 200 : 503;
    
    logger.health('api_server', health.status, {
      checks: health.checks,
      uptime: health.uptime,
    });
    
    res.status(statusCode).json({
      status: health.status,
      timestamp: health.timestamp,
      uptime: health.uptime,
      version: process.env.npm_package_version || '1.0.0',
    });
    
  } catch (error) {
    logger.errorWithContext(error, {
      endpoint: 'health_check',
      requestId: req.requestId,
    });
    
    res.status(503).json({
      status: 'unhealthy',
      error: 'Health check failed',
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * @swagger
 * /health/detailed:
 *   get:
 *     summary: Detailed health check
 *     description: Returns comprehensive system health information
 *     tags: [Health]
 *     responses:
 *       200:
 *         description: Detailed health information
 */
router.get('/detailed', async (req, res) => {
  try {
    const [health, systemInfo] = await Promise.all([
      healthCheck(),
      Promise.resolve(getSystemInfo()),
    ]);
    
    const statusCode = health.status === 'healthy' ? 200 : 503;
    
    res.status(statusCode).json({
      status: health.status,
      checks: health.checks,
      system: systemInfo,
      timestamp: health.timestamp,
      uptime: health.uptime,
      version: process.env.npm_package_version || '1.0.0',
      environment: process.env.NODE_ENV || 'development',
    });
    
  } catch (error) {
    logger.errorWithContext(error, {
      endpoint: 'health_detailed',
      requestId: req.requestId,
    });
    
    res.status(503).json({
      status: 'unhealthy',
      error: 'Detailed health check failed',
      timestamp: new Date().toISOString(),
    });
  }
});

export default router;