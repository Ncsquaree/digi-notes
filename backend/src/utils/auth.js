const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { v4: uuidv4 } = require('uuid');
const crypto = require('crypto');
const logger = require('./logger');

const JWT_SECRET = process.env.JWT_SECRET;
// Note: expiry values are read at call time to allow tests to mutate env vars
const JWT_REFRESH_SECRET = process.env.JWT_REFRESH_SECRET;
const JWT_ISSUER = process.env.JWT_ISSUER || undefined;
const JWT_AUDIENCE = process.env.JWT_AUDIENCE || undefined;
const NODE_ENV = process.env.NODE_ENV || 'development';

function ensureSecret(name, value) {
  if (!value) {
    if (NODE_ENV === 'production') {
      throw new Error(`${name} must be set in production and be at least 32 characters`);
    }
    logger.warn(`${name} is not set. Using an insecure fallback secret for development only.`);
    return;
  }
  if (String(value).length < 32) {
    if (NODE_ENV === 'production') {
      throw new Error(`${name} must be at least 32 characters in production`);
    }
    logger.warn(`${name} appears to be weak (less than 32 chars). Use a strong random secret.`);
  }
}

ensureSecret('JWT_SECRET', JWT_SECRET);
ensureSecret('JWT_REFRESH_SECRET', JWT_REFRESH_SECRET);

function hashPassword(password) {
  if (!password) throw Object.assign(new Error('Password is required'), { name: 'ValidationError' });
  const salt = bcrypt.genSaltSync(10);
  return bcrypt.hashSync(password, salt);
}

async function comparePassword(password, hash) {
  return bcrypt.compare(password, hash);
}

function parseExpiryToSeconds(exp) {
  // accepts formats like '15m','1h','30d','3600s'
  if (!exp) return undefined;
  const m = String(exp).match(/^(\d+)([smhd])?$/);
  if (!m) return undefined;
  const val = Number(m[1]);
  const unit = m[2] || 's';
  switch (unit) {
    case 's': return val;
    case 'm': return val * 60;
    case 'h': return val * 60 * 60;
    case 'd': return val * 60 * 60 * 24;
    default: return val;
  }
}

function generateAccessToken(user) {
  const payload = { userId: user.id, email: user.email };
  const expiresIn = process.env.JWT_EXPIRES_IN || '15m';
  const opts = { expiresIn };
  if (JWT_ISSUER) opts.issuer = JWT_ISSUER;
  if (JWT_AUDIENCE) opts.audience = JWT_AUDIENCE;
  if (!JWT_SECRET && NODE_ENV === 'production') throw new Error('JWT_SECRET is not configured');
  const secret = JWT_SECRET || 'dev-secret';
  return jwt.sign(payload, secret, opts);
}

function generateRefreshToken(user) {
  const payload = { userId: user.id, tokenId: uuidv4() };
  const expiresIn = process.env.JWT_REFRESH_EXPIRES_IN || '30d';
  const opts = { expiresIn };
  if (JWT_ISSUER) opts.issuer = JWT_ISSUER;
  if (JWT_AUDIENCE) opts.audience = JWT_AUDIENCE;
  if (!JWT_REFRESH_SECRET && NODE_ENV === 'production') throw new Error('JWT_REFRESH_SECRET is not configured');
  const secret = JWT_REFRESH_SECRET || 'dev-refresh-secret';
  return jwt.sign(payload, secret, opts);
}

function generateTokenPair(user) {
  const accessToken = generateAccessToken(user);
  const refreshToken = generateRefreshToken(user);
  return { accessToken, refreshToken };
}

function verifyAccessToken(token) {
  try {
    if (!JWT_SECRET && NODE_ENV === 'production') throw new Error('JWT_SECRET is not configured');
    const secret = JWT_SECRET || 'dev-secret';
    return jwt.verify(token, secret, { issuer: JWT_ISSUER || undefined, audience: JWT_AUDIENCE || undefined });
  } catch (err) {
    // Normalize error names to integrate with error handler
    err.name = err.name || 'JsonWebTokenError';
    throw err;
  }
}

function verifyRefreshToken(token) {
  try {
    if (!JWT_REFRESH_SECRET && NODE_ENV === 'production') throw new Error('JWT_REFRESH_SECRET is not configured');
    const secret = JWT_REFRESH_SECRET || 'dev-refresh-secret';
    return jwt.verify(token, secret, { issuer: JWT_ISSUER || undefined, audience: JWT_AUDIENCE || undefined });
  } catch (err) {
    err.name = err.name || 'JsonWebTokenError';
    throw err;
  }
}

function hashToken(token) {
  return crypto.createHash('sha256').update(token).digest('hex');
}

module.exports = {
  hashPassword,
  comparePassword,
  generateAccessToken,
  generateRefreshToken,
  generateTokenPair,
  verifyAccessToken,
  verifyRefreshToken,
  hashToken,
  parseExpiryToSeconds
};
