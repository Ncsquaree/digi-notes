const db = require('../config/database');
const { Flashcard, Note, StudySession } = require('../models');
const { NotFoundError, ForbiddenError, InternalError } = require('../utils/errors');
const logger = require('../utils/logger');

class FlashcardService {
  /**
   * Get a flashcard and ensure ownership
   * @param {string} flashcardId
   * @param {string} userId
   */
  static async getFlashcardById(flashcardId, userId) {
    const f = await Flashcard.findById(flashcardId);
    if (!f) throw new NotFoundError('Flashcard not found');
    if (f.user_id !== userId) throw new ForbiddenError('Access denied');
    return f;
  }

  /**
   * Count flashcards for a user
   * @param {string} userId
   */
  static async countFlashcardsByUser(userId) {
    return Flashcard.countByUserId(userId);
  }

  /**
   * List flashcards for a user
   * @param {string} userId
   * @param {object} opts
   */
  static async getFlashcardsByUserId(userId, opts = {}) {
    return Flashcard.findByUserId(userId, opts);
  }

  /**
   * List flashcards for a note (ensures note ownership)
   * @param {string} noteId
   * @param {string} userId
   */
  static async getFlashcardsByNoteId(noteId, userId) {
    const note = await Note.findById(noteId);
    if (!note) throw new NotFoundError('Note not found');
    if (note.user_id !== userId) throw new ForbiddenError('Access denied');
    return Flashcard.findByNoteId(noteId);
  }

  /**
   * Get due flashcards for review
   * @param {string} userId
   * @param {object} opts
   */
  static async getDueFlashcards(userId, opts = {}) {
    return Flashcard.findDueForReview(userId, opts);
  }

  /**
   * Create a flashcard (validates note ownership)
   * @param {string} noteId
   * @param {string} userId
   * @param {object} payload
   */
  static async createFlashcard(noteId, userId, payload = {}) {
    const note = await Note.findById(noteId);
    if (!note) throw new NotFoundError('Note not found');
    if (note.user_id !== userId) throw new ForbiddenError('Access denied');
    const f = await Flashcard.create({ note_id: noteId, user_id: userId, question: payload.question, answer: payload.answer, difficulty: payload.difficulty || 0 });
    logger.info('Flashcard created', { flashcardId: f.id, noteId, userId });
    return f;
  }

  /**
   * Bulk create flashcards produced by AI for a note
   * @param {string} noteId
   * @param {string} userId
   * @param {Array<object>} flashcardsArray
   */
  static async createFlashcardsFromAI(noteId, userId, flashcardsArray = []) {
    const note = await Note.findById(noteId);
    if (!note) throw new NotFoundError('Note not found');
    if (note.user_id !== userId) throw new ForbiddenError('Access denied');
    const prepared = flashcardsArray.map(f => ({ note_id: noteId, user_id: userId, question: f.question, answer: f.answer, difficulty: f.difficulty || 0 }));
    return Flashcard.bulkCreate(prepared);
  }

  /**
   * Update a flashcard (ensures ownership)
   * @param {string} flashcardId
   * @param {string} userId
   * @param {object} fields
   */
  static async updateFlashcard(flashcardId, userId, fields = {}) {
    const f = await Flashcard.findById(flashcardId);
    if (!f) throw new NotFoundError('Flashcard not found');
    if (f.user_id !== userId) throw new ForbiddenError('Access denied');
    return Flashcard.update(flashcardId, fields);
  }

  /**
   * Delete a flashcard (ensures ownership)
   * @param {string} flashcardId
   * @param {string} userId
   */
  static async deleteFlashcard(flashcardId, userId) {
    const f = await Flashcard.findById(flashcardId);
    if (!f) throw new NotFoundError('Flashcard not found');
    if (f.user_id !== userId) throw new ForbiddenError('Access denied');
    await Flashcard.delete(flashcardId);
    logger.info('Flashcard deleted', { flashcardId, userId });
    return true;
  }

  /**
   * Review a flashcard using the SM-2 algorithm and record a StudySession.
   * This updates the flashcard's interval/repetitions/ef and creates a study session in a single transaction.
   * @param {string} flashcardId
   * @param {string} userId
   * @param {number} quality - 0..5
   * @param {number|null} time_spent_seconds
   */
  static async reviewFlashcard(flashcardId, userId, quality, time_spent_seconds = null) {
    const client = await db.pool.connect();
    try {
      await client.query('BEGIN');
      const res = await client.query('SELECT * FROM flashcards WHERE id = $1 FOR UPDATE', [flashcardId]);
      if (!res.rowCount) throw new NotFoundError('Flashcard not found');
      const card = res.rows[0];
      if (card.user_id !== userId) throw new ForbiddenError('Access denied');

      // SM-2 algorithm
      let { ef = 2.5, repetitions = 0, interval = 0 } = card;
      ef = Number(ef) || 2.5;
      repetitions = Number(repetitions) || 0;
      interval = Number(interval) || 0;

      if (quality < 3) {
        repetitions = 0;
        interval = 1;
      } else {
        repetitions += 1;
        if (repetitions === 1) interval = 1;
        else if (repetitions === 2) interval = 6;
        else interval = Math.round(interval * ef) || 1;
      }

      // Update EF
      const q = quality;
      const newEf = Math.max(1.3, ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)));

      // compute next review date
      const nextDate = new Date();
      nextDate.setDate(nextDate.getDate() + interval);
      const nextReviewDate = nextDate.toISOString().slice(0, 10);

      const updateRes = await client.query(
        `UPDATE flashcards SET repetitions = $1, interval = $2, difficulty = $3, next_review_date = $4, easiness_factor = $5 WHERE id = $6 RETURNING *`,
        [repetitions, interval, card.difficulty || 0, nextReviewDate, newEf, flashcardId]
      );

      const updatedCard = updateRes.rows[0];

      const session = await client.query(
        `INSERT INTO study_sessions (user_id, flashcard_id, quality, time_spent_seconds) VALUES ($1,$2,$3,$4) RETURNING *`,
        [userId, flashcardId, quality, time_spent_seconds]
      );

      await client.query('COMMIT');
      return { flashcard: updatedCard, session: session.rows[0] };
    } catch (err) {
      await client.query('ROLLBACK');
      logger.logError(err, { flashcardId, userId });
      if (err.isOperational) throw err;
      throw new InternalError('Failed to review flashcard');
    } finally {
      client.release();
    }
  }

  /**
   * Return aggregated study statistics for a user by delegating to StudySession model
   * @param {string} userId
   * @param {object} opts
   */
  static async getStudyStats(userId, opts = {}) {
    try {
      if (!userId) throw new Error('userId required');
      // StudySession.getStatsByUserId should return aggregated fields
      const stats = await StudySession.getStatsByUserId(userId, opts);
      // normalize shape expected by controllers
      return {
        // StudySession.getStatsByUserId returns aliases: total_sessions, avg_quality, total_time
        total_reviews: stats.total_sessions ?? stats.total_reviews ?? stats.totalReviews ?? 0,
        avg_quality: stats.avg_quality ?? stats.avgQuality ?? 0,
        total_study_time_seconds: stats.total_time ?? stats.total_study_time_seconds ?? stats.totalStudyTimeSeconds ?? 0,
      };
    } catch (err) {
      logger.logError(err, { userId, fn: 'getStudyStats' });
      throw new InternalError('Failed to compute study stats');
    }
  }
}

module.exports = FlashcardService;
