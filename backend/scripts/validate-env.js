const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const { Client } = require('pg');
const redis = require('redis');
const { S3Client, HeadBucketCommand } = require('@aws-sdk/client-s3');
const fsPromises = require('fs').promises;

dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

// CLI args
const args = process.argv.slice(2 || 0);
const STRICT = args.includes('--strict') || args.includes('-s');

const required = {
  server: ['NODE_ENV', 'PORT'],
  database: ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD'],
  jwt: ['JWT_SECRET', 'JWT_EXPIRES_IN'],
  aws: ['AWS_REGION', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_S3_BUCKET'],
  ai: ['AI_SERVICE_URL']
};

const errors = [];

function checkPresence(keys, category) {
  keys.forEach((k) => {
    if (!process.env[k]) {
      errors.push(`${category}: Missing ${k}`);
    }
  });
}

Object.entries(required).forEach(([cat, keys]) => checkPresence(keys, cat));

// Basic format checks
const port = Number(process.env.PORT || process.env.DB_PORT);
if (isNaN(port) || port < 1 || port > 65535) {
  errors.push('PORT/DB_PORT must be a valid integer between 1 and 65535');
}

if (process.env.JWT_SECRET && process.env.JWT_SECRET.length < 32) {
  errors.push('JWT_SECRET must be at least 32 characters');
}
if (!process.env.JWT_REFRESH_SECRET || process.env.JWT_REFRESH_SECRET.length < 32) {
  errors.push('JWT_REFRESH_SECRET must be present and at least 32 characters');
}

const urlPattern = /^https?:\/\//;
if (process.env.AI_SERVICE_URL && !urlPattern.test(process.env.AI_SERVICE_URL)) {
  errors.push('AI_SERVICE_URL must be a valid URL starting with http:// or https://');
}

// Validate numeric configs
const rateWindow = Number(process.env.RATE_LIMIT_WINDOW_MS || 15 * 60 * 1000);
const rateMax = Number(process.env.RATE_LIMIT_MAX_REQUESTS || 100);
if (isNaN(rateWindow) || rateWindow <= 0) errors.push('RATE_LIMIT_WINDOW_MS must be a positive integer');
if (isNaN(rateMax) || rateMax <= 0 || rateMax > 10000) errors.push('RATE_LIMIT_MAX_REQUESTS must be a positive integer less than 10000');

const corsOrigins = (process.env.CORS_ORIGIN || '').split(',').map(s => s.trim()).filter(Boolean);
// CORS is flexible, but ensure variables are not obviously malformed
if (process.env.CORS_ORIGIN && corsOrigins.length === 0) console.warn('CORS_ORIGIN appears set but no valid origins parsed');

(async () => {
  if (errors.length > 0) {
    console.error('Environment validation failed:');
    errors.forEach((e) => console.error(' -', e));
    process.exit(1);
  }

  // Test DB connection
  const client = new Client({
    host: process.env.DB_HOST,
    port: Number(process.env.DB_PORT || 5432),
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME
  });
  try {
    await client.connect();
    await client.query('SELECT 1');
    console.log('Postgres: OK');
  } catch (err) {
    console.error('Postgres connection failed:', err.message);
    process.exit(1);
  } finally {
    try { await client.end(); } catch (e) {}
  }

  // Test Redis if configured
  if (process.env.REDIS_HOST) {
    try {
      const rclient = redis.createClient({
        socket: { host: process.env.REDIS_HOST, port: Number(process.env.REDIS_PORT || 6379) },
        password: process.env.REDIS_PASSWORD || undefined
      });
      rclient.on('error', () => {});
      await rclient.connect();
      const ping = await rclient.ping();
      console.log('Redis:', ping === 'PONG' ? 'OK' : ping);
      await rclient.disconnect();
    } catch (err) {
      console.warn('Redis connection warning:', err.message);
      if (STRICT) process.exit(1);
    }
  }

  // Check log directory writability if LOG_FILE_PATH configured
  if (process.env.LOG_FILE_PATH) {
    const logDir = path.dirname(process.env.LOG_FILE_PATH);
    try {
      await fsPromises.access(logDir);
      console.log('Log directory exists:', logDir);
    } catch (err) {
      try {
        await fsPromises.mkdir(logDir, { recursive: true });
        console.log('Created log directory:', logDir);
      } catch (mkdirErr) {
        console.error('Log directory not writable or cannot be created:', mkdirErr.message);
        if (STRICT) process.exit(1);
      }
    }
  }

  // Validate S3 bucket exists via headBucket
  if (process.env.AWS_S3_BUCKET) {
    try {
      const s3 = new S3Client({ region: process.env.AWS_REGION });
      await s3.send(new HeadBucketCommand({ Bucket: process.env.AWS_S3_BUCKET }));
      console.log('S3 bucket accessible:', process.env.AWS_S3_BUCKET);
    } catch (err) {
      console.warn('S3 headBucket warning:', err.message || err);
      if (STRICT) process.exit(1);
    }
  }

  // Optional: probe AI service /health
  if (process.env.AI_SERVICE_URL) {
    try {
      const resp = await require('axios').get(`${process.env.AI_SERVICE_URL.replace(/\/$/, '')}/health`, { timeout: 3000 });
      if (resp.status === 200) console.log('AI service: OK');
      else console.warn('AI service health returned non-200:', resp.status);
    } catch (err) {
      console.warn('AI service health probe failed:', err.message || err);
      if (STRICT) process.exit(1);
    }
  }

  console.log('All validations passed');
  process.exit(0);
})();
