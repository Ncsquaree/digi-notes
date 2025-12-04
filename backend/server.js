require('dotenv').config();
const path = require('path');
const http = require('http');
const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const { v4: uuidv4 } = require('uuid');

const logger = require('./src/utils/logger');
const requestLogger = require('./src/middleware/requestLogger');
const errorHandler = require('./src/middleware/errorHandler');
const routes = require('./src/routes');
const db = require('./src/config/database');
const redisClient = require('./src/config/redis');
const { swaggerSpec } = require('./src/config/swagger');
const swaggerUi = require('swagger-ui-express');

const PORT = process.env.PORT || 5000;

const app = express();
app.set('trust proxy', true);

// Middleware
app.use(helmet());
const corsOrigins = (process.env.CORS_ORIGIN || '').split(',').map(s => s.trim()).filter(Boolean);
app.use(cors({ origin: corsOrigins.length ? corsOrigins : true }));
app.use(express.json({ limit: process.env.VALIDATION_MAX_BODY_SIZE || '10mb' }));
app.use(express.urlencoded({ extended: true, limit: process.env.VALIDATION_MAX_BODY_SIZE || '10mb' }));

const limiter = rateLimit({
  windowMs: Number(process.env.RATE_LIMIT_WINDOW_MS || 15 * 60 * 1000),
  max: Number(process.env.RATE_LIMIT_MAX_REQUESTS || 100)
});
app.use(limiter);

// Request logger (adds req.id)
app.use((req, res, next) => {
  req.id = req.headers['x-request-id'] || uuidv4();
  next();
});
app.use(requestLogger);

// Health endpoints
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.get('/ready', async (req, res) => {
  const services = {};
  try {
    await db.testConnection();
    services.database = 'ok';
  } catch (e) {
    services.database = `error: ${e.message}`;
  }

  try {
    const r = await redisClient.testConnection();
    services.redis = r === 'PONG' ? 'ok' : String(r);
  } catch (e) {
    services.redis = `error: ${e.message}`;
  }

  // Determine readiness: database is mandatory; Redis is optional unless explicitly required
  const redisRequired = process.env.REDIS_REQUIRED_FOR_READY === 'true';
  const dbReady = services.database === 'ok';
  const redisReady = services.redis === 'ok';
  const ready = dbReady && (redisRequired ? redisReady : true);

  res.status(ready ? 200 : 503).json({ status: ready ? 'ready' : 'not ready', services });
});

// Mount API routes
app.use('/api', routes);

// Swagger UI and JSON spec
try {
  app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec));
  app.get('/api-docs.json', (req, res) => res.json(swaggerSpec));
} catch (e) {
  logger.warn('Failed to mount Swagger UI', e.message || e);
}

// 404 handler for unknown routes
app.use((req, res, next) => {
  res.status(404).json({ success: false, error: { message: 'Route not found', requestId: req.id } });
});

// Error handler
app.use(errorHandler);

const server = http.createServer(app);

server.listen(PORT, () => {
  logger.info(`Server started on port ${PORT}`);
});

// Graceful shutdown
const shutdown = async (signal) => {
  logger.info(`Received ${signal}. Shutting down gracefully.`);
  try {
    // Close server and wait for existing connections to finish (with timeout)
    const closePromise = new Promise((resolve) => {
      let settled = false;
      try {
        server.close((err) => {
          if (settled) return;
          settled = true;
          if (err) {
            logger.error('Error while closing HTTP server', err);
            return resolve({ ok: false, error: err });
          }
          logger.info('Stopped accepting new connections');
          return resolve({ ok: true });
        });
      } catch (err) {
        if (!settled) {
          settled = true;
          logger.error('Exception while calling server.close()', err);
          return resolve({ ok: false, error: err });
        }
      }

      // Force close after timeout
      setTimeout(() => {
        if (!settled) {
          settled = true;
          logger.warn('Forcing shutdown after timeout while closing server');
          return resolve({ ok: false, timeout: true });
        }
      }, Number(process.env.SHUTDOWN_TIMEOUT_MS || 15000));
    });

    await closePromise;

    // Now close DB pool
    try {
      await db.pool.end();
      logger.info('Database pool closed');
    } catch (e) {
      logger.error('Error closing DB pool', e);
    }

    // Disconnect redis
    try {
      await redisClient.disconnect();
      logger.info('Redis client disconnected');
    } catch (e) {
      logger.warn('Redis disconnect error', e.message || e);
    }

    logger.info('Graceful shutdown complete');
    process.exit(0);
  } catch (err) {
    logger.error('Error during shutdown procedure', err);
    process.exit(1);
  }
};

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));

process.on('uncaughtException', (err) => {
  logger.error('Uncaught exception', err);
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  logger.error('Unhandled rejection', reason);
  process.exit(1);
});

module.exports = app;
