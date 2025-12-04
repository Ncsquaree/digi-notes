const redis = require('redis');
const logger = require('../utils/logger');

const host = process.env.REDIS_HOST || 'redis';
const port = Number(process.env.REDIS_PORT || 6379);
const password = process.env.REDIS_PASSWORD || undefined;

const client = redis.createClient({
  socket: { host, port, reconnectStrategy: retries => Math.min(retries * 50, 2000) },
  password: password || undefined
});

let connected = false;

client.on('connect', () => {
  connected = true;
  logger.info('Redis: connected');
});

client.on('error', (err) => {
  connected = false;
  logger.warn('Redis error', err.message);
});

client.on('end', () => {
  connected = false;
  logger.info('Redis: connection closed');
});

async function init() {
  try {
    await client.connect();
  } catch (e) {
    logger.warn('Redis connection failed at init: ' + e.message);
  }
}

async function testConnection() {
  if (!connected) {
    // attempt a connect but don't throw irrecoverably
    await init();
  }
  if (!connected) throw new Error('Redis not connected');
  return client.ping();
}

async function disconnect() {
  try {
    if (connected) await client.disconnect();
  } catch (e) {
    logger.warn('Error during Redis disconnect: ' + e.message);
  }
}

// initialize in background
init();

module.exports = { client, testConnection, disconnect };
