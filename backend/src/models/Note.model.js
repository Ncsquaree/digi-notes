const db = require('../config/database');

class Note {
  static async findById(noteId) {
    const res = await db.pool.query('SELECT * FROM notes WHERE id = $1 LIMIT 1', [noteId]);
    return res.rowCount ? res.rows[0] : null;
  }

  static async findByUserId(userId, { limit = 50, offset = 0, filters = {} } = {}) {
    const clauses = ['user_id = $1'];
    const values = [userId];
    let idx = 2;
    if (filters.subject_id) { clauses.push(`subject_id = $${idx++}`); values.push(filters.subject_id); }
    if (filters.chapter_id) { clauses.push(`chapter_id = $${idx++}`); values.push(filters.chapter_id); }
    if (filters.processing_status) { clauses.push(`processing_status = $${idx++}`); values.push(filters.processing_status); }
    const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : '';
    const sql = `SELECT * FROM notes ${where} ORDER BY created_at DESC LIMIT $${idx++} OFFSET $${idx}`;
    values.push(limit, offset);
    const res = await db.pool.query(sql, values);
    return res.rows;
  }

  static async create({ user_id, subject_id = null, chapter_id = null, title, original_image_url, processing_status = 'pending' }) {
    const res = await db.pool.query(
      `INSERT INTO notes (user_id, subject_id, chapter_id, title, original_image_url, processing_status) VALUES ($1,$2,$3,$4,$5,$6) RETURNING *`,
      [user_id, subject_id, chapter_id, title, original_image_url, processing_status]
    );
    return res.rows[0];
  }

  static async update(noteId, fields = {}) {
    const keys = Object.keys(fields);
    if (!keys.length) return this.findById(noteId);

    // Whitelist allowed updatable columns to prevent arbitrary SQL injection
    // NOTE: If you add fields here (e.g. summaries), ensure the `notes` table has
    // the corresponding columns via a DB migration. If the DB doesn't include
    // them yet, add the migration before writing these fields to avoid SQL errors.
    const allowed = ['title', 'original_image_url', 'parsed_content', 'processing_status', 'error_message', 'subject_id', 'chapter_id', 'summary_brief', 'summary_detailed', 'processing_task_id'];
    const validKeys = keys.filter(k => allowed.includes(k));
    if (!validKeys.length) return this.findById(noteId);

    const sets = validKeys.map((k, i) => `${k} = $${i + 2}`);
    const values = [noteId, ...validKeys.map((k) => fields[k])];
    const sql = `UPDATE notes SET ${sets.join(', ')} WHERE id = $1 RETURNING *`;
    const res = await db.pool.query(sql, values);
    return res.rowCount ? res.rows[0] : null;
  }

  static async delete(noteId) {
    await db.pool.query('DELETE FROM notes WHERE id = $1', [noteId]);
    return true;
  }

  static async updateProcessingStatus(noteId, status, errorMessage = null) {
    const res = await db.pool.query('UPDATE notes SET processing_status = $1, error_message = $2 WHERE id = $3 RETURNING *', [status, errorMessage, noteId]);
    return res.rowCount ? res.rows[0] : null;
  }

  static async findByProcessingStatus(status, { limit = 100 } = {}) {
    const res = await db.pool.query('SELECT * FROM notes WHERE processing_status = $1 ORDER BY created_at ASC LIMIT $2', [status, limit]);
    return res.rows;
  }

  /**
   * Count notes for a user.
   * Supported filters: { processing_status, subject_id, chapter_id }
   */
  static async countByUserId(userId, filters = {}) {
    const clauses = ['user_id = $1'];
    const values = [userId];
    let idx = 2;
    if (typeof filters.subject_id !== 'undefined') { clauses.push(`subject_id = $${idx++}`); values.push(filters.subject_id); }
    if (typeof filters.chapter_id !== 'undefined') { clauses.push(`chapter_id = $${idx++}`); values.push(filters.chapter_id); }
    if (typeof filters.processing_status !== 'undefined') { clauses.push(`processing_status = $${idx++}`); values.push(filters.processing_status); }
    const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : '';
    const sql = `SELECT COUNT(*)::int as cnt FROM notes ${where}`;
    const res = await db.pool.query(sql, values);
    return res.rows[0].cnt;
  }
}

module.exports = Note;
