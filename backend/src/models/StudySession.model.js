const db = require('../config/database');

class StudySession {
  static async findById(sessionId) {
    const res = await db.pool.query('SELECT * FROM study_sessions WHERE id = $1 LIMIT 1', [sessionId]);
    return res.rowCount ? res.rows[0] : null;
  }

  static async findByUserId(userId, { limit = 50, offset = 0, startDate = null, endDate = null } = {}) {
    const clauses = ['user_id = $1'];
    const values = [userId];
    let idx = 2;
    if (startDate) { clauses.push(`reviewed_at >= $${idx++}`); values.push(startDate); }
    if (endDate) { clauses.push(`reviewed_at <= $${idx++}`); values.push(endDate); }
    const where = `WHERE ${clauses.join(' AND ')}`;
    const sql = `SELECT * FROM study_sessions ${where} ORDER BY reviewed_at DESC LIMIT $${idx++} OFFSET $${idx}`;
    values.push(limit, offset);
    const res = await db.pool.query(sql, values);
    return res.rows;
  }

  static async findByFlashcardId(flashcardId, { limit = 10 } = {}) {
    const res = await db.pool.query('SELECT * FROM study_sessions WHERE flashcard_id = $1 ORDER BY reviewed_at DESC LIMIT $2', [flashcardId, limit]);
    return res.rows;
  }

  static async create({ user_id, flashcard_id, quality, time_spent_seconds = null }) {
    const res = await db.pool.query(
      `INSERT INTO study_sessions (user_id, flashcard_id, quality, time_spent_seconds) VALUES ($1,$2,$3,$4) RETURNING *`,
      [user_id, flashcard_id, quality, time_spent_seconds]
    );
    return res.rows[0];
  }

  static async getStatsByUserId(userId, { startDate = null, endDate = null } = {}) {
    const clauses = ['s.user_id = $1'];
    const values = [userId];
    let idx = 2;
    if (startDate) { clauses.push(`s.reviewed_at >= $${idx++}`); values.push(startDate); }
    if (endDate) { clauses.push(`s.reviewed_at <= $${idx++}`); values.push(endDate); }
    const where = `WHERE ${clauses.join(' AND ')}`;
    const sql = `SELECT COUNT(*)::int AS total_sessions, AVG(s.quality)::numeric AS avg_quality, COALESCE(SUM(s.time_spent_seconds),0)::int AS total_time FROM study_sessions s ${where}`;
    const res = await db.pool.query(sql, values);
    return res.rows[0];
  }

  static async getStatsByFlashcardId(flashcardId) {
    const res = await db.pool.query('SELECT COUNT(*)::int AS review_count, AVG(quality)::numeric AS avg_quality FROM study_sessions WHERE flashcard_id = $1', [flashcardId]);
    return res.rows[0];
  }
}

module.exports = StudySession;
