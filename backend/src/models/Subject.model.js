const db = require('../config/database');

class Subject {
  static async findById(subjectId) {
    const res = await db.pool.query('SELECT * FROM subjects WHERE id = $1 LIMIT 1', [subjectId]);
    return res.rowCount ? res.rows[0] : null;
  }

  static async findByUserId(userId, { limit = 50, offset = 0 } = {}) {
    const res = await db.pool.query('SELECT * FROM subjects WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3', [userId, limit, offset]);
    return res.rows;
  }

  static async findByUserAndName(userId, name) {
    const res = await db.pool.query('SELECT * FROM subjects WHERE user_id = $1 AND name = $2 LIMIT 1', [userId, name]);
    return res.rowCount ? res.rows[0] : null;
  }

  static async create({ user_id, name, description = null, color = null, icon = null }) {
    const res = await db.pool.query(
      `INSERT INTO subjects (user_id, name, description, color, icon) VALUES ($1,$2,$3,$4,$5) RETURNING *`,
      [user_id, name, description, color, icon]
    );
    return res.rows[0];
  }

  static async update(subjectId, fields = {}) {
    const keys = Object.keys(fields);
    if (!keys.length) return this.findById(subjectId);

    const allowed = ['name', 'description', 'color', 'icon'];
    const validKeys = keys.filter(k => allowed.includes(k));
    if (!validKeys.length) return this.findById(subjectId);

    const sets = validKeys.map((k, i) => `${k} = $${i + 2}`);
    const values = [subjectId, ...validKeys.map((k) => fields[k])];
    const sql = `UPDATE subjects SET ${sets.join(', ')} WHERE id = $1 RETURNING *`;
    const res = await db.pool.query(sql, values);
    return res.rowCount ? res.rows[0] : null;
  }

  static async delete(subjectId) {
    await db.pool.query('DELETE FROM subjects WHERE id = $1', [subjectId]);
    return true;
  }

  static async countByUserId(userId) {
    const res = await db.pool.query('SELECT COUNT(*)::int as cnt FROM subjects WHERE user_id = $1', [userId]);
    return res.rows[0].cnt;
  }
}

module.exports = Subject;
