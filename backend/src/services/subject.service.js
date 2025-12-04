const { Subject } = require('../models');
const { NotFoundError, ForbiddenError, ConflictError } = require('../utils/errors');
const logger = require('../utils/logger');

class SubjectService {
  static async getSubjectById(subjectId, userId) {
    const s = await Subject.findById(subjectId);
    if (!s) throw new NotFoundError('Subject not found');
    if (s.user_id !== userId) throw new ForbiddenError('Access denied');
    return s;
  }

  static async getSubjectsByUserId(userId, opts = {}) {
    return Subject.findByUserId(userId, opts);
  }

  static async createSubject(userId, { name, description, color, icon }) {
    const exists = await Subject.findByUserAndName(userId, name);
    if (exists) throw new ConflictError('Subject name already exists');
    const created = await Subject.create({ user_id: userId, name, description, color, icon });
    logger.info('Subject created', { subjectId: created.id, userId });
    return created;
  }

  static async updateSubject(subjectId, userId, fields) {
    const s = await Subject.findById(subjectId);
    if (!s) throw new NotFoundError('Subject not found');
    if (s.user_id !== userId) throw new ForbiddenError('Access denied');
    if (fields.name) {
      const dup = await Subject.findByUserAndName(userId, fields.name);
      if (dup && dup.id !== subjectId) throw new ConflictError('Subject name already exists');
    }
    return Subject.update(subjectId, fields);
  }

  static async deleteSubject(subjectId, userId) {
    const s = await Subject.findById(subjectId);
    if (!s) throw new NotFoundError('Subject not found');
    if (s.user_id !== userId) throw new ForbiddenError('Access denied');
    await Subject.delete(subjectId);
    logger.info('Subject deleted', { subjectId, userId });
    return true;
  }

  static async getSubjectWithStats(subjectId, userId) {
    const s = await this.getSubjectById(subjectId, userId);
    // lightweight stats
    const chapterCount = await require('../models/Chapter.model').countBySubjectId(subjectId);
    const noteCountRes = await require('../models/Note.model').countByUserId(userId, { subject_id: subjectId });
    return { subject: s, stats: { chapterCount, noteCount: noteCountRes } };
  }
}

module.exports = SubjectService;
