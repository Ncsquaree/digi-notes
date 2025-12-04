const { createLogger, format, transports } = require('winston');
const DailyRotateFile = require('winston-daily-rotate-file');
const path = require('path');

const LOG_LEVEL = process.env.LOG_LEVEL || 'info';
const LOG_FILE_PATH = process.env.LOG_FILE_PATH || path.join(__dirname, '..', '..', 'logs');
const LOG_MAX_SIZE = process.env.LOG_MAX_SIZE || '10m';
const LOG_MAX_FILES = process.env.LOG_MAX_FILES || '7d';
const NODE_ENV = process.env.NODE_ENV || 'development';

const baseFormat = format.combine(
  format.timestamp(),
  format.errors({ stack: true })
);

const consoleFormat = NODE_ENV === 'development'
  ? format.combine(baseFormat, format.colorize(), format.printf(({ timestamp, level, message, stack }) => `${timestamp} ${level}: ${stack || message}`))
  : format.combine(baseFormat, format.printf(({ timestamp, level, message, stack }) => JSON.stringify({ timestamp, level, message, stack })));

const logger = createLogger({
  level: LOG_LEVEL,
  format: format.json(),
  transports: [
    new transports.Console({ format: consoleFormat }),
    new DailyRotateFile({ filename: path.join(LOG_FILE_PATH, 'error-%DATE%.log'), level: 'error', datePattern: 'YYYY-MM-DD', maxSize: LOG_MAX_SIZE, maxFiles: LOG_MAX_FILES }),
    new DailyRotateFile({ filename: path.join(LOG_FILE_PATH, 'combined-%DATE%.log'), datePattern: 'YYYY-MM-DD', maxSize: LOG_MAX_SIZE, maxFiles: LOG_MAX_FILES })
  ],
  exceptionHandlers: [
    new DailyRotateFile({ filename: path.join(LOG_FILE_PATH, 'exceptions-%DATE%.log'), datePattern: 'YYYY-MM-DD', maxSize: LOG_MAX_SIZE, maxFiles: LOG_MAX_FILES })
  ],
  rejectionHandlers: [
    new DailyRotateFile({ filename: path.join(LOG_FILE_PATH, 'rejections-%DATE%.log'), datePattern: 'YYYY-MM-DD', maxSize: LOG_MAX_SIZE, maxFiles: LOG_MAX_FILES })
  ]
});

function logRequest(req, res, duration) {
  const entry = {
    requestId: req.id,
    method: req.method,
    url: req.originalUrl || req.url,
    statusCode: res.statusCode,
    durationMs: duration,
    ip: req.ip
  };
  if (res.statusCode >= 500) logger.error('HTTP request', entry);
  else if (res.statusCode >= 400) logger.warn('HTTP request', entry);
  else logger.info('HTTP request', entry);
}

function logError(err, context = {}) {
  logger.error(err.message || 'Error', { stack: err.stack, ...context });
}

function logDatabaseQuery(query, duration) {
  if (NODE_ENV === 'development') logger.debug('DB query', { query, duration });
}

module.exports = logger;
module.exports.logRequest = logRequest;
module.exports.logError = logError;
module.exports.logDatabaseQuery = logDatabaseQuery;
