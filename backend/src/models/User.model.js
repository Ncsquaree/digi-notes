const db = require('../config/database');
const logger = require('../utils/logger');

/**
 * User model repository
 */
class User {
  /**
   * Find user by id
   * @param {string} userId
   */
  static async findById(userId) {
    const res = await db.pool.query('SELECT * FROM users WHERE id = $1 LIMIT 1', [userId]);
    return res.rowCount ? res.rows[0] : null;
  }

  /**
   * Find user by email
   * @param {string} email
   */
  static async findByEmail(email) {
    const res = await db.pool.query('SELECT * FROM users WHERE email = $1 LIMIT 1', [email]);
    return res.rowCount ? res.rows[0] : null;
  }

  /**
   * Create user
   */
  static async create({ email, password_hash, first_name = null, last_name = null }) {
    const res = await db.pool.query(
      `INSERT INTO users (email, password_hash, first_name, last_name, is_active, email_verified) VALUES ($1,$2,$3,$4,true,false) RETURNING *`,
      [email, password_hash, first_name, last_name]
    );
    return res.rows[0];
  }

  /**
   * Update user with dynamic fields
   */
  static async update(userId, fields = {}) {
    const keys = Object.keys(fields);
    if (!keys.length) return this.findById(userId);

    // Allowed columns for update
    const allowed = ['first_name', 'last_name', 'is_active', 'email_verified', 'password_hash'];
    const validKeys = keys.filter(k => allowed.includes(k));
    if (!validKeys.length) {
      logger.warn('User.update called with no updatable fields', { userId, attempted: keys });
      return this.findById(userId);
    }

    const sets = validKeys.map((k, i) => `${k} = $${i + 2}`);
    const values = [userId, ...validKeys.map((k) => fields[k])];
    const sql = `UPDATE users SET ${sets.join(', ')} WHERE id = $1 RETURNING *`;
    const res = await db.pool.query(sql, values);
    return res.rowCount ? res.rows[0] : null;
  }

  static async delete(userId) {
    await db.pool.query('DELETE FROM users WHERE id = $1', [userId]);
    return true;
  }

  static async updateLastLogin(userId) {
    await db.pool.query('UPDATE users SET last_login = NOW() WHERE id = $1', [userId]);
    return true;
  }

  static async list({ limit = 50, offset = 0, filters = {} } = {}) {
    const clauses = [];
    const values = [];
    let idx = 1;
    if (typeof filters.is_active !== 'undefined') {
      clauses.push(`is_active = $${idx++}`);
      values.push(filters.is_active);
    }
    if (typeof filters.email_verified !== 'undefined') {
      clauses.push(`email_verified = $${idx++}`);
      values.push(filters.email_verified);
    }
    const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : '';
    const sql = `SELECT * FROM users ${where} ORDER BY created_at DESC LIMIT $${idx++} OFFSET $${idx}`;
    values.push(limit, offset);
    const res = await db.pool.query(sql, values);
    return res.rows;
  }
}

module.exports = User;
