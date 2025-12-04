const logger = require('../utils/logger');
const {
  AppError,
  ValidationError,
} = require('../utils/errors');

function mapStatus(err) {
  if (err && typeof err.statusCode === 'number') return err.statusCode;
  if (err instanceof AppError) return err.statusCode || 500;
  if (err && err.name === 'ValidationError') return 400;
  if (err && (err.name === 'UnauthorizedError' || err.name === 'JsonWebTokenError' || err.name === 'TokenExpiredError')) return 401;
  if (err && err.name === 'ForbiddenError') return 403;
  if (err && err.name === 'NotFoundError') return 404;
  return 500;
}

module.exports = (err, req, res, next) => {
  const status = mapStatus(err);
  const requestId = req && req.id;
  const safeMessage = process.env.NODE_ENV === 'production' && status === 500 ? 'Internal server error' : (err && err.message) || 'Error';

  // Prepare validation detail if present
  let validationErrors = null;
  if (err && Array.isArray(err.errors) && err.errors.length > 0) {
    validationErrors = err.errors;
  } else if (err && typeof err.array === 'function') {
    try {
      const arr = err.array();
      validationErrors = arr.map((e) => ({ field: e.param, message: e.msg, value: e.value }));
    } catch (e) {
      // ignore
    }
  }

  logger.error('Unhandled error', {
    message: err && err.message,
    stack: err && err.stack,
    requestId,
    method: req && req.method,
    url: req && req.originalUrl,
    errorClass: err && err.constructor && err.constructor.name,
    errorCode: err && err.code,
    validationErrors,
  });

  const payload = {
    success: false,
    error: {
      message: safeMessage,
      code: (err && (err.code || err.name)) || 'ERROR',
      requestId,
    },
  };

  if (validationErrors) {
    payload.error.validationErrors = validationErrors;
    // If the top-level message is generic, prefer the first validation message
    if (!payload.error.message || payload.error.message === 'Validation failed') {
      payload.error.message = (validationErrors[0] && validationErrors[0].message) || payload.error.message;
    }
  }

  if (process.env.NODE_ENV !== 'production') {
    payload.error.details = { stack: err && err.stack, status };
  }

  res.status(status).json(payload);
};
