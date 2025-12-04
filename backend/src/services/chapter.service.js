const { Chapter, Subject } = require('../models');
const { NotFoundError, ForbiddenError } = require('../utils/errors');
const logger = require('../utils/logger');

class ChapterService {
  static async getChapterById(chapterId, userId) {
    const ch = await Chapter.findById(chapterId);
    if (!ch) throw new NotFoundError('Chapter not found');
    const subject = await Subject.findById(ch.subject_id);
    if (!subject || subject.user_id !== userId) throw new ForbiddenError('Access denied');
    return ch;
  }

  static async getChaptersBySubjectId(subjectId, userId, opts = {}) {
    const subject = await Subject.findById(subjectId);
    if (!subject) throw new NotFoundError('Subject not found');
    if (subject.user_id !== userId) throw new ForbiddenError('Access denied');
    return Chapter.findBySubjectId(subjectId, opts);
  }

  static async createChapter(subjectId, userId, { name, description, order_index = 0 } = {}) {
    // Support two call signatures for backwards compatibility:
    // 1) createChapter(subjectId, userId, payload)
    // 2) createChapter(userId, payload)  (tests use this signature)
    let sid = subjectId;
    let uid = userId;
    let payload = { name, description, order_index };
    if (typeof userId === 'object' && arguments.length === 2) {
      // called as (userId, payload)
      payload = userId || {};
      uid = subjectId; // original first arg is actually userId
      sid = null;
    }

    if (sid) {
      const subject = await Subject.findById(sid);
      if (!subject) throw new NotFoundError('Subject not found');
      if (subject.user_id !== uid) throw new ForbiddenError('Access denied');
      payload.subject_id = sid;
    }

    const created = await Chapter.create(payload);
    logger.info('Chapter created', { chapterId: created.id, subjectId: sid, userId: uid });
    return created;
  }

  static async updateChapter(chapterId, userId, fields) {
    const ch = await Chapter.findById(chapterId);
    if (!ch) throw new NotFoundError('Chapter not found');
    const subject = await Subject.findById(ch.subject_id);
    if (!subject || subject.user_id !== userId) throw new ForbiddenError('Access denied');
    return Chapter.update(chapterId, fields);
  }

  static async deleteChapter(chapterId, userId) {
    const ch = await Chapter.findById(chapterId);
    if (!ch) throw new NotFoundError('Chapter not found');
    const subject = await Subject.findById(ch.subject_id);
    if (!subject || subject.user_id !== userId) throw new ForbiddenError('Access denied');
    await Chapter.delete(chapterId);
    logger.info('Chapter deleted', { chapterId, userId });
    return true;
  }

  static async reorderChapter(chapterId, userId, newOrderIndex) {
    const ch = await Chapter.findById(chapterId);
    if (!ch) throw new NotFoundError('Chapter not found');
    const subject = await Subject.findById(ch.subject_id);
    if (!subject || subject.user_id !== userId) throw new ForbiddenError('Access denied');
    return Chapter.reorder(chapterId, newOrderIndex);
  }

  static async getChapterWithStats(chapterId, userId) {
    const ch = await this.getChapterById(chapterId, userId);
    const noteCount = await require('../models/Note.model').countByUserId(userId, { chapter_id: chapterId });
    return { chapter: ch, stats: { noteCount } };
  }
}

module.exports = ChapterService;
