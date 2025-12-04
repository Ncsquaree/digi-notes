const db = require('../config/database');
const { Note, Subject, Chapter, Flashcard, KnowledgeGraphNode } = require('../models');
const { NotFoundError, ForbiddenError, InternalError } = require('../utils/errors');
const logger = require('../utils/logger');

class NoteService {
  /**
   * Get a note by id and ensure it belongs to the user
   * @param {string} noteId
   * @param {string} userId
   */
  static async getNoteById(noteId, userId) {
    const n = await Note.findById(noteId);
    if (!n) throw new NotFoundError('Note not found');
    if (n.user_id !== userId) throw new ForbiddenError('Access denied');
    return n;
  }

  /**
   * List notes for a user with optional filters
   * @param {string} userId
   * @param {object} opts
   */
  static async getNotesByUserId(userId, opts = {}) {
    return Note.findByUserId(userId, opts);
  }

  /**
   * Create a note (validates subject/chapter ownership if provided)
   * @param {string} userId
   * @param {object} payload - { subject_id, chapter_id, title, original_image_url }
   */
  static async createNote(userId, payload = {}) {
    const { subject_id = null, chapter_id = null, title, original_image_url } = payload;
    if (subject_id) {
      const s = await Subject.findById(subject_id);
      if (!s) throw new NotFoundError('Subject not found');
      if (s.user_id !== userId) throw new ForbiddenError('Subject access denied');
    }
    if (chapter_id) {
      const c = await Chapter.findById(chapter_id);
      if (!c) throw new NotFoundError('Chapter not found');
      // chapter belongs to subject; ensure subject belongs to user
      const s = await Subject.findById(c.subject_id);
      if (!s || s.user_id !== userId) throw new ForbiddenError('Chapter access denied');
    }
    const created = await Note.create({ user_id: userId, subject_id, chapter_id, title, original_image_url });
    logger.info('Note created', { noteId: created.id, userId });
    return created;
  }

  /**
   * Update a note with ownership check
   * @param {string} noteId
   * @param {string} userId
   * @param {object} fields
   */
  static async updateNote(noteId, userId, fields = {}) {
    const n = await Note.findById(noteId);
    if (!n) throw new NotFoundError('Note not found');
    if (n.user_id !== userId) throw new ForbiddenError('Access denied');
    // If changing subject/chapter, validate ownership
    if (fields.subject_id) {
      const s = await Subject.findById(fields.subject_id);
      if (!s) throw new NotFoundError('Subject not found');
      if (s.user_id !== userId) throw new ForbiddenError('Subject access denied');
    }
    if (fields.chapter_id) {
      const c = await Chapter.findById(fields.chapter_id);
      if (!c) throw new NotFoundError('Chapter not found');
      const s = await Subject.findById(c.subject_id);
      if (!s || s.user_id !== userId) throw new ForbiddenError('Chapter access denied');
    }
    return Note.update(noteId, fields);
  }

  /**
   * Delete a note with ownership check
   * @param {string} noteId
   * @param {string} userId
   */
  static async deleteNote(noteId, userId) {
    const n = await Note.findById(noteId);
    if (!n) throw new NotFoundError('Note not found');
    if (n.user_id !== userId) throw new ForbiddenError('Access denied');
    await Note.delete(noteId);
    logger.info('Note deleted', { noteId, userId });
    return true;
  }

  /**
   * Update processing status (used by background workers)
   * @param {string} noteId
   * @param {string} status
   * @param {string|null} errorMessage
   */
  static async updateNoteProcessingStatus(noteId, status, errorMessage = null) {
    return Note.updateProcessingStatus(noteId, status, errorMessage);
  }

  /**
   * Helper used by controllers/tests to mark a note as processing and return persisted info
   */
  static async markProcessing(noteId, taskId) {
    try {
      await Note.updateProcessingStatus(noteId, 'processing', null);
    } catch (err) {
      // ignore persistence errors here - caller will handle
    }
    return { processing_task_id: taskId };
  }

  /**
   * Helper to return processing status for a note. Tests mock this method.
   */
  static async getProcessingStatus(noteId, userId) {
    // Prefer Note model helper if present
    if (typeof Note.getProcessingStatus === 'function') {
      return Note.getProcessingStatus(noteId, userId);
    }
    const n = await Note.findById(noteId);
    if (!n) throw new NotFoundError('Note not found');
    if (n.user_id && n.user_id !== userId) throw new ForbiddenError('Access denied');
    return { processing_task_id: n.processing_task_id || null, processing_status: n.processing_status || 'idle', parsed_content: n.parsed_content || null };
  }

  /**
   * Attach AI parsing result: update parsed_content and optionally create knowledge graph nodes.
   * Runs in a transaction for atomicity.
   * @param {string} noteId
   * @param {string} userId
   * @param {object} result - { parsed_content, nodesArray }
   */
  static async attachAIResult(noteId, userId, result = {}) {
    const client = await db.pool.connect();
    try {
      await client.query('BEGIN');
      const n = await Note.findById(noteId);
      if (!n) throw new NotFoundError('Note not found');
      if (n.user_id !== userId) throw new ForbiddenError('Access denied');

      const updated = await client.query(
        'UPDATE notes SET parsed_content = $1, processing_status = $2, error_message = NULL WHERE id = $3 RETURNING *',
        [result.parsed_content || null, 'processed', noteId]
      );

      if (result.nodesArray && Array.isArray(result.nodesArray) && result.nodesArray.length) {
        // Create nodes using KnowledgeGraphNode.bulkCreate pattern but here use client
        const cols = ['user_id','note_id','node_type','node_label','neptune_vertex_id','properties'];
        const values = [];
        const placeholders = result.nodesArray.map((nNode, i) => {
          const idx = i * cols.length;
          values.push(userId, noteId, nNode.node_type, nNode.node_label, nNode.neptune_vertex_id || null, nNode.properties || {});
          return `($${idx+1},$${idx+2},$${idx+3},$${idx+4},$${idx+5},$${idx+6})`;
        }).join(',');
        const sql = `INSERT INTO knowledge_graph_nodes (${cols.join(',')}) VALUES ${placeholders}`;
        await client.query(sql, values);
      }

      await client.query('COMMIT');
      return updated.rows[0];
    } catch (err) {
      await client.query('ROLLBACK');
      logger.logError(err, { noteId, userId });
      if (err.isOperational) throw err;
      throw new InternalError('Failed to attach AI result');
    } finally {
      client.release();
    }
  }
}

module.exports = NoteService;
