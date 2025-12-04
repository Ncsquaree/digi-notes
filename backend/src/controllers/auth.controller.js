const db = require('../config/database');
const auth = require('../utils/auth');
const logger = require('../utils/logger');
const { ValidationError, ConflictError, UnauthorizedError, NotFoundError, InternalError } = require('../utils/errors');

function nowPlusSeconds(sec) {
  return new Date(Date.now() + (sec * 1000));
}

function getExpiryDateFromString(expStr) {
  const seconds = auth.parseExpiryToSeconds(expStr);
  if (!seconds) return null;
  return nowPlusSeconds(seconds);
}

exports.register = async (req, res, next) => {
  try {
    const { email, password, firstName, lastName } = req.body || {};

    const { pool } = db;
    const existing = await pool.query('SELECT id FROM users WHERE email = $1 LIMIT 1', [email]);
    if (existing.rowCount > 0) {
      throw new ConflictError('Email already registered');
    }

    const password_hash = auth.hashPassword(password);

    const insertUser = await pool.query(
      `INSERT INTO users (email, password_hash, first_name, last_name, is_active, email_verified) VALUES ($1,$2,$3,$4, true, false) RETURNING id, email, first_name, last_name, created_at`,
      [email, password_hash, firstName || null, lastName || null]
    );
    const user = insertUser.rows[0];

    const { accessToken, refreshToken } = auth.generateTokenPair({ id: user.id, email: user.email });
    const tokenHash = auth.hashToken(refreshToken);
    const expiresAt = getExpiryDateFromString(process.env.JWT_REFRESH_EXPIRES_IN || '30d');
    if (!expiresAt) {
      const msg = 'Invalid JWT_REFRESH_EXPIRES_IN configuration; refresh tokens will not expire';
      if (process.env.NODE_ENV === 'production') {
        throw new InternalError(msg);
      } else {
        logger.warn(msg);
      }
    }

    await pool.query('INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES ($1,$2,$3)', [user.id, tokenHash, expiresAt]);

    logger.info('User registered', { userId: user.id, email: user.email });

    res.status(201).json({ success: true, data: { user: { id: user.id, email: user.email, firstName: user.first_name, lastName: user.last_name }, accessToken, refreshToken } });
  } catch (err) {
    next(err);
  }
};

exports.login = async (req, res, next) => {
  try {
    const { email, password } = req.body || {};
    const { pool } = db;
    const userRes = await pool.query('SELECT id, email, password_hash, is_active FROM users WHERE email = $1 LIMIT 1', [email]);
    if (userRes.rowCount === 0) {
      throw new UnauthorizedError('Invalid credentials');
    }
    const user = userRes.rows[0];
    if (!user.is_active) {
      throw new UnauthorizedError('Invalid credentials');
    }
    const valid = await auth.comparePassword(password, user.password_hash);
    if (!valid) {
      throw new UnauthorizedError('Invalid credentials');
    }

    await pool.query('UPDATE users SET last_login = NOW() WHERE id = $1', [user.id]);

    const { accessToken, refreshToken } = auth.generateTokenPair({ id: user.id, email: user.email });
    const tokenHash = auth.hashToken(refreshToken);
    const expiresAt = getExpiryDateFromString(process.env.JWT_REFRESH_EXPIRES_IN || '30d');
    if (!expiresAt) {
      const msg = 'Invalid JWT_REFRESH_EXPIRES_IN configuration; refresh tokens will not expire';
      if (process.env.NODE_ENV === 'production') {
        throw new InternalError(msg);
      } else {
        logger.warn(msg);
      }
    }
    await pool.query('INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES ($1,$2,$3)', [user.id, tokenHash, expiresAt]);

    logger.info('User login', { userId: user.id, ip: req.ip });
    res.json({ success: true, data: { user: { id: user.id, email: user.email }, accessToken, refreshToken } });
  } catch (err) {
    next(err);
  }
};

exports.refreshToken = async (req, res, next) => {
  try {
    const { refreshToken } = req.body || {};

    const decoded = auth.verifyRefreshToken(refreshToken);
    const tokenHash = auth.hashToken(refreshToken);
    const { pool } = db;
    const tokenRow = await pool.query('SELECT id, user_id, revoked, expires_at FROM refresh_tokens WHERE token_hash = $1 LIMIT 1', [tokenHash]);
    if (tokenRow.rowCount === 0) { throw new UnauthorizedError('Invalid or expired refresh token'); }
    const tokenRec = tokenRow.rows[0];
    if (tokenRec.revoked) { throw new UnauthorizedError('Invalid or expired refresh token'); }
    if (tokenRec.expires_at && new Date(tokenRec.expires_at) <= new Date()) { throw new UnauthorizedError('Invalid or expired refresh token'); }

    // Optionally rotate refresh token
    const userRes = await pool.query('SELECT id, email FROM users WHERE id = $1 LIMIT 1', [tokenRec.user_id]);
    if (userRes.rowCount === 0) { throw new NotFoundError('User not found'); }
    const user = userRes.rows[0];

    // Rotate: revoke old and issue new
    await pool.query('UPDATE refresh_tokens SET revoked = true WHERE id = $1', [tokenRec.id]);
    const { accessToken, refreshToken: newRefresh } = auth.generateTokenPair({ id: user.id, email: user.email });
    const newHash = auth.hashToken(newRefresh);
    const expiresAt = getExpiryDateFromString(process.env.JWT_REFRESH_EXPIRES_IN || '30d');
    if (!expiresAt) {
      const msg = 'Invalid JWT_REFRESH_EXPIRES_IN configuration; refresh tokens will not expire';
      if (process.env.NODE_ENV === 'production') {
        throw new InternalError(msg);
      } else {
        logger.warn(msg);
      }
    }
    await pool.query('INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES ($1,$2,$3)', [user.id, newHash, expiresAt]);

    logger.info('Refresh token rotated', { userId: user.id });
    res.json({ success: true, data: { accessToken, refreshToken: newRefresh } });
  } catch (err) {
    next(err);
  }
};

exports.logout = async (req, res, next) => {
  try {
    const { refreshToken } = req.body || {};
    const tokenHash = auth.hashToken(refreshToken);
    const { pool } = db;
    const userId = req.user && req.user.userId;
    if (!userId) { throw new UnauthorizedError('Unauthenticated'); }
    const result = await pool.query('UPDATE refresh_tokens SET revoked = true WHERE token_hash = $1 AND user_id = $2', [tokenHash, userId]);
    if (result.rowCount === 0) {
      logger.warn('Logout attempted but no matching refresh token found for user', { userId, tokenHash });
    } else {
      logger.info('User logout', { userId });
    }
    res.json({ success: true, message: 'Logged out successfully' });
  } catch (err) {
    next(err);
  }
};

exports.getCurrentUser = async (req, res, next) => {
  try {
    const userId = req.user && req.user.userId;
    if (!userId) { throw new UnauthorizedError('Unauthenticated'); }
    const { pool } = db;
    const userRes = await pool.query('SELECT id, email, first_name, last_name, created_at, last_login, email_verified FROM users WHERE id = $1 LIMIT 1', [userId]);
    if (userRes.rowCount === 0) { throw new NotFoundError('User not found'); }
    const u = userRes.rows[0];
    res.json({ success: true, data: { user: { id: u.id, email: u.email, firstName: u.first_name, lastName: u.last_name, createdAt: u.created_at, lastLogin: u.last_login, emailVerified: u.email_verified } } });
  } catch (err) {
    next(err);
  }
};
