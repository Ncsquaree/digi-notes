const logger = require('../utils/logger');
const { v4: uuidv4 } = require('uuid');

function sanitizeBody(body) {
  try {
    const copy = JSON.parse(JSON.stringify(body || {}));
    const sensitive = ['password', 'confirmPassword', 'token', 'accessToken', 'refreshToken', 'secret'];
    Object.keys(copy).forEach((k) => {
      if (sensitive.includes(k) || sensitive.some(s => k.toLowerCase().includes(s.toLowerCase()))) {
        copy[k] = '[REDACTED]';
      }
    });
    return copy;
  } catch (e) {
    return '[unserializable]';
  }
}

module.exports = (req, res, next) => {
  const start = Date.now();
  // prefer incoming X-Request-ID header when present for cross-service tracing
  const requestId = req.headers['x-request-id'] || req.id || uuidv4();
  req.id = requestId;

  const logStart = {
    requestId,
    method: req.method,
    url: req.originalUrl || req.url,
    ip: req.ip,
    ua: req.headers['user-agent']
  };

  // Include sanitized body for write operations in non-production only
  const writeMethods = ['POST', 'PUT', 'PATCH'];
  if (process.env.NODE_ENV !== 'production' && writeMethods.includes(req.method)) {
    logStart.body = sanitizeBody(req.body);
  }

  logger.info('Incoming request', logStart);
  // expose request id to clients for end-to-end tracing
  try {
    res.setHeader && res.setHeader('X-Request-ID', requestId);
  } catch (e) {
    // ignore header set errors
  }
  res.on('finish', () => {
    const duration = Date.now() - start;
    const entry = {
      requestId,
      method: req.method,
      url: req.originalUrl || req.url,
      statusCode: res.statusCode,
      durationMs: duration,
      contentLength: res.getHeader('content-length') || 0
    };
    if (process.env.NODE_ENV !== 'production' && writeMethods.includes(req.method)) {
      entry.body = sanitizeBody(req.body);
    }
    if (res.statusCode >= 500) logger.error('Request completed', entry);
    else if (res.statusCode >= 400) logger.warn('Request completed', entry);
    else logger.info('Request completed', entry);
  });

  next();
};
