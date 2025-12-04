const db = require('../config/database');

class Chapter {
  static async findById(chapterId) {
    const res = await db.pool.query('SELECT * FROM chapters WHERE id = $1 LIMIT 1', [chapterId]);
    return res.rowCount ? res.rows[0] : null;
  }

  static async findBySubjectId(subjectId, { limit = 100, offset = 0 } = {}) {
    const res = await db.pool.query('SELECT * FROM chapters WHERE subject_id = $1 ORDER BY order_index ASC, created_at ASC LIMIT $2 OFFSET $3', [subjectId, limit, offset]);
    return res.rows;
  }

  static async create({ subject_id, name, description = null, order_index = 0 }) {
    const res = await db.pool.query(
      `INSERT INTO chapters (subject_id, name, description, order_index) VALUES ($1,$2,$3,$4) RETURNING *`,
      [subject_id, name, description, order_index]
    );
    return res.rows[0];
  }

  static async update(chapterId, fields = {}) {
    const keys = Object.keys(fields);
    if (!keys.length) return this.findById(chapterId);

    const allowed = ['name', 'description', 'order_index'];
    const validKeys = keys.filter(k => allowed.includes(k));
    if (!validKeys.length) return this.findById(chapterId);

    const sets = validKeys.map((k, i) => `${k} = $${i + 2}`);
    const values = [chapterId, ...validKeys.map((k) => fields[k])];
    const sql = `UPDATE chapters SET ${sets.join(', ')} WHERE id = $1 RETURNING *`;
    const res = await db.pool.query(sql, values);
    return res.rowCount ? res.rows[0] : null;
  }

  static async delete(chapterId) {
    await db.pool.query('DELETE FROM chapters WHERE id = $1', [chapterId]);
    return true;
  }

  static async reorder(chapterId, newOrderIndex) {
    // Simple implementation: update the chapter's order_index
    const res = await db.pool.query('UPDATE chapters SET order_index = $1 WHERE id = $2 RETURNING *', [newOrderIndex, chapterId]);
    return res.rowCount ? res.rows[0] : null;
  }

  static async countBySubjectId(subjectId) {
    const res = await db.pool.query('SELECT COUNT(*)::int as cnt FROM chapters WHERE subject_id = $1', [subjectId]);
    return res.rows[0].cnt;
  }
}

module.exports = Chapter;
