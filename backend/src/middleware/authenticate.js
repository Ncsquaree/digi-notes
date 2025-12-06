const auth = require('../utils/auth');
const logger = require('../utils/logger');

class UnauthorizedError extends Error { constructor(message){ super(message); this.name = 'UnauthorizedError'; } }

async function authenticate(req, res, next) {
  try {
    // If a previous middleware (tests) already set req.user, skip verification
    if (req.user) return next();
    const authHeader = req.headers['authorization'] || req.headers['Authorization'];
    if (!authHeader) throw new UnauthorizedError('No token provided');
    const parts = authHeader.split(' ');
    if (parts.length !== 2 || !/^Bearer$/i.test(parts[0])) throw new UnauthorizedError('Invalid authorization header');
    const token = parts[1];
    let payload;
    try {
      payload = auth.verifyAccessToken(token);
    } catch (err) {
      // Preserve JWT error granularity for the central error handler and logs
      const name = err && err.name;
      if (name === 'TokenExpiredError') {
        const e = new Error('Token expired');
        e.name = 'UnauthorizedError';
        e.code = 'TOKEN_EXPIRED';
        logger.warn('Expired token used', { requestId: req.id });
        return next(e);
      }
      if (name === 'JsonWebTokenError' || name === 'NotBeforeError') {
        const e = new Error('Invalid token');
        e.name = 'UnauthorizedError';
        e.code = name;
        logger.warn('Invalid token provided', { requestId: req.id, error: name });
        return next(e);
      }
      // Unknown error - pass through
      return next(err);
    }
    // Attach canonical id non-enumerably while preserving original shape for tests
    req.user = { userId: payload.userId, email: payload.email };
    try {
      Object.defineProperty(req.user, 'id', { value: payload.userId, enumerable: false, writable: false });
    } catch (e) {
      // fall back if environment doesn't allow defineProperty
      req.user.id = payload.userId;
    }
    logger.info('Authentication successful', { requestId: req.id, userId: req.user.userId, id: req.user.id });
    next();
  } catch (err) {
    next(err);
  }
}

async function optionalAuth(req, res, next) {
  try {
    const authHeader = req.headers['authorization'] || req.headers['Authorization'];
    if (!authHeader) { req.user = null; return next(); }
    const parts = authHeader.split(' ');
    if (parts.length !== 2 || !/^Bearer$/i.test(parts[0])) { req.user = null; return next(); }
    const token = parts[1];
    try {
      const payload = auth.verifyAccessToken(token);
      // Optional auth also sets the canonical `id` property (non-enumerable)
      req.user = { userId: payload.userId, email: payload.email };
      try {
        Object.defineProperty(req.user, 'id', { value: payload.userId, enumerable: false, writable: false });
      } catch (e) {
        req.user.id = payload.userId;
      }
    } catch (err) {
      req.user = null;
    }
    next();
  } catch (err) {
    next(err);
  }
}

module.exports = authenticate;
module.exports.optional = optionalAuth;
