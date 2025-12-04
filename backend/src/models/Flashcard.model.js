const db = require('../config/database');

class Flashcard {
  static async findById(flashcardId) {
    const res = await db.pool.query('SELECT * FROM flashcards WHERE id = $1 LIMIT 1', [flashcardId]);
    return res.rowCount ? res.rows[0] : null;
  }

  static async findByUserId(userId, { limit = 50, offset = 0, filters = {} } = {}) {
    const clauses = ['user_id = $1'];
    const values = [userId];
    let idx = 2;
    if (filters.note_id) { clauses.push(`note_id = $${idx++}`); values.push(filters.note_id); }
    if (filters.due_today) { clauses.push(`next_review_date <= CURRENT_DATE`); }
    const where = `WHERE ${clauses.join(' AND ')}`;
    const sql = `SELECT * FROM flashcards ${where} ORDER BY next_review_date ASC LIMIT $${idx++} OFFSET $${idx}`;
    values.push(limit, offset);
    const res = await db.pool.query(sql, values);
    return res.rows;
  }

  static async findByNoteId(noteId) {
    const res = await db.pool.query('SELECT * FROM flashcards WHERE note_id = $1 ORDER BY created_at ASC', [noteId]);
    return res.rows;
  }

  static async findDueForReview(userId, { limit = 20 } = {}) {
    const res = await db.pool.query('SELECT * FROM flashcards WHERE user_id = $1 AND next_review_date <= CURRENT_DATE ORDER BY next_review_date ASC LIMIT $2', [userId, limit]);
    return res.rows;
  }

  static async create({ note_id, user_id, question, answer, difficulty = 0, interval = 0, repetitions = 0, next_review_date = null }) {
    const nr = next_review_date || new Date().toISOString().slice(0,10);
    const res = await db.pool.query(
      `INSERT INTO flashcards (note_id, user_id, question, answer, difficulty, interval, repetitions, next_review_date) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING *`,
      [note_id, user_id, question, answer, difficulty, interval, repetitions, nr]
    );
    return res.rows[0];
  }

  static async update(flashcardId, fields = {}) {
    const keys = Object.keys(fields);
    if (!keys.length) return this.findById(flashcardId);

    const allowed = ['question', 'answer', 'difficulty', 'interval', 'repetitions', 'next_review_date', 'note_id'];
    const validKeys = keys.filter(k => allowed.includes(k));
    if (!validKeys.length) return this.findById(flashcardId);

    const sets = validKeys.map((k, i) => `${k} = $${i + 2}`);
    const values = [flashcardId, ...validKeys.map((k) => fields[k])];
    const sql = `UPDATE flashcards SET ${sets.join(', ')} WHERE id = $1 RETURNING *`;
    const res = await db.pool.query(sql, values);
    return res.rowCount ? res.rows[0] : null;
  }

  static async delete(flashcardId) {
    await db.pool.query('DELETE FROM flashcards WHERE id = $1', [flashcardId]);
    return true;
  }

  static async bulkCreate(flashcardsArray = []) {
    if (!flashcardsArray.length) return [];
    const cols = ['note_id','user_id','question','answer','difficulty','interval','repetitions','next_review_date'];
    const values = [];
    const placeholders = flashcardsArray.map((f, i) => {
      const idx = i * cols.length;
      values.push(f.note_id, f.user_id, f.question, f.answer, f.difficulty || 0, f.interval || 0, f.repetitions || 0, f.next_review_date || new Date().toISOString().slice(0,10));
      return `($${idx+1},$${idx+2},$${idx+3},$${idx+4},$${idx+5},$${idx+6},$${idx+7},$${idx+8})`;
    }).join(',');
    const sql = `INSERT INTO flashcards (${cols.join(',')}) VALUES ${placeholders} RETURNING *`;
    const res = await db.pool.query(sql, values);
    return res.rows;
  }

  static async countByUserId(userId, { includeRevoked = false } = {}) {
    const res = await db.pool.query('SELECT COUNT(*)::int as cnt FROM flashcards WHERE user_id = $1', [userId]);
    return res.rows[0].cnt;
  }
}

module.exports = Flashcard;
