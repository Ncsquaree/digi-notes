/**
 * Validation middleware using express-validator.
 * Exports a `validate()` factory that runs validation chains and throws
 * a ValidationError (from utils/errors) when validation fails.
 */
'use strict';

const { body, param, query, validationResult } = require('express-validator');
const { ValidationError } = require('../utils/errors');

/**
 * Runs the provided validation chains and throws ValidationError with
 * formatted field errors when validation fails.
 * @param {Array} validations - array of express-validator chains
 */
/**
 * Validation runner middleware.
 * Note: we call `next(err)` for async safety. If your project relies on
 * `express-async-errors` or Express 5's async error handling, throwing here
 * would also work — but using `next(err)` is compatible with Express 4.
 */
function validate(validations) {
  return async (req, res, next) => {
    try {
      await Promise.all((validations || []).map((v) => v.run(req)));
      const result = validationResult(req);
      if (result.isEmpty()) return next();
      const formatted = result.array().map((e) => ({ field: e.param, message: e.msg, value: e.value }));
      // Derive a helpful top-level message from the first field error so
      // consumers that only read `error.message` get useful feedback.
      const topMessage = (formatted[0] && formatted[0].message) ? String(formatted[0].message) : 'Validation failed';
      // Pass error into express error pipeline
      return next(new ValidationError(topMessage, formatted));
    } catch (err) {
      return next(err);
    }
  };
}

/**
 * Sanitizer helper to trim and escape strings to mitigate XSS.
 * Returns an array of sanitizers for use in validation chains.
 */
function sanitizeInput(field) {
  const doEscape = process.env.SANITIZE_HTML === 'true';
  let chain = body(field).trim();
  if (doEscape) chain = chain.escape();
  return [chain];
}

// Common validation helper factories -------------------------------------------------
function isUUID(field) {
  // By default validate path/param UUIDs
  return param(field).isUUID(4).withMessage(`${field} must be a valid UUID`);
}

function isEmail(field) {
  return body(field).exists().withMessage(`${field} is required`).isEmail().withMessage('Invalid email').normalizeEmail();
}

function isString(field, min = 1, max = 1024) {
  const doEscape = process.env.SANITIZE_HTML === 'true';
  let chain = body(field)
    .exists()
    .withMessage(`${field} is required`)
    .isString()
    .withMessage(`${field} must be a string`)
    .isLength({ min, max })
    .withMessage(`${field} must be between ${min} and ${max} characters`)
    .trim();
  if (doEscape) chain = chain.escape();
  return chain;
}

function isOptionalString(field, max = 1024) {
  const doEscape = process.env.SANITIZE_HTML === 'true';
  let chain = body(field)
    .optional()
    .isString()
    .withMessage(`${field} must be a string`)
    .isLength({ max })
    .withMessage(`${field} must be at most ${max} characters`)
    .trim();
  if (doEscape) chain = chain.escape();
  return chain;
}

/**
 * Returns middleware which rejects unknown body fields when
 * VALIDATION_STRICT_MODE is enabled. Provide an array of allowed field names.
 */
function rejectUnknownBody(allowedFields = []) {
  return (req, res, next) => {
    if (process.env.VALIDATION_STRICT_MODE !== 'true') return next();
    const bodyKeys = req.body && typeof req.body === 'object' ? Object.keys(req.body) : [];
    const unknown = bodyKeys.filter((k) => !allowedFields.includes(k));
    if (unknown.length > 0) {
      const errors = unknown.map((f) => ({ field: f, message: 'Unknown field', value: req.body[f] }));
      return next(new ValidationError('Unknown fields present in request', errors));
    }
    return next();
  };
}

function isInteger(field, min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER) {
  return body(field).optional().isInt({ min, max }).toInt().withMessage(`${field} must be an integer`);
}

function isEnum(field, values = []) {
  return body(field).optional().isIn(values).withMessage(`${field} must be one of: ${values.join(', ')}`);
}

function isDate(field) {
  return body(field).optional().isISO8601().toDate().withMessage(`${field} must be a valid ISO8601 date`);
}

function isJSON(field) {
  return body(field).optional().custom((value) => {
    if (typeof value === 'object') return true;
    try {
      JSON.parse(value);
      return true;
    } catch (e) {
      throw new Error(`${field} must be valid JSON`);
    }
  });
}

module.exports = {
  validate,
  sanitizeInput,
  // helpers
  isUUID,
  isEmail,
  isString,
  isOptionalString,
  isInteger,
  isEnum,
  isDate,
  isJSON,
  rejectUnknownBody,
  body,
  param,
  query,
};
