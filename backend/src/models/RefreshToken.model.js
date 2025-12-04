const db = require('../config/database');

class RefreshToken {
  static async findById(tokenId) {
    const res = await db.pool.query('SELECT * FROM refresh_tokens WHERE id = $1 LIMIT 1', [tokenId]);
    return res.rowCount ? res.rows[0] : null;
  }

  static async findByTokenHash(tokenHash) {
    const res = await db.pool.query('SELECT * FROM refresh_tokens WHERE token_hash = $1 AND revoked = false AND expires_at > NOW() LIMIT 1', [tokenHash]);
    return res.rowCount ? res.rows[0] : null;
  }

  static async findByUserId(userId, { includeRevoked = false } = {}) {
    const q = includeRevoked ? 'SELECT * FROM refresh_tokens WHERE user_id = $1 ORDER BY created_at DESC' : 'SELECT * FROM refresh_tokens WHERE user_id = $1 AND revoked = false ORDER BY created_at DESC';
    const res = await db.pool.query(q, [userId]);
    return res.rows;
  }

  static async create({ user_id, token_hash, expires_at }) {
    const res = await db.pool.query('INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES ($1,$2,$3) RETURNING *', [user_id, token_hash, expires_at]);
    return res.rows[0];
  }

  static async revoke(tokenId) {
    await db.pool.query('UPDATE refresh_tokens SET revoked = true WHERE id = $1', [tokenId]);
    return true;
  }

  static async revokeByTokenHash(tokenHash, userId) {
    await db.pool.query('UPDATE refresh_tokens SET revoked = true WHERE token_hash = $1 AND user_id = $2', [tokenHash, userId]);
    return true;
  }

  static async revokeAllForUser(userId) {
    await db.pool.query('UPDATE refresh_tokens SET revoked = true WHERE user_id = $1', [userId]);
    return true;
  }

  static async deleteExpired() {
    await db.pool.query('DELETE FROM refresh_tokens WHERE expires_at < NOW()');
    return true;
  }

  static async countByUserId(userId, { includeRevoked = false } = {}) {
    const q = includeRevoked ? 'SELECT COUNT(*)::int as cnt FROM refresh_tokens WHERE user_id = $1' : 'SELECT COUNT(*)::int as cnt FROM refresh_tokens WHERE user_id = $1 AND revoked = false';
    const res = await db.pool.query(q, [userId]);
    return res.rows[0].cnt;
  }
}

module.exports = RefreshToken;
