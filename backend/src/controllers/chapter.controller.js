const ChapterService = require('../services/chapter.service');
const NoteService = require('../services/note.service');

module.exports = {
  async listChapters(req, res, next) {
    try {
      const userId = req.user.id;
      const subjectId = req.query.subjectId || null;
      const chapters = await ChapterService.getChaptersBySubjectId(subjectId, userId, { page: req.query.page, limit: req.query.limit });
      return res.json({ success: true, data: chapters });
    } catch (err) {
      next(err);
    }
  },

  async getChapter(req, res, next) {
    try {
      const userId = req.user.id;
      const chapter = await ChapterService.getChapterById(req.params.id, userId);
      return res.json({ success: true, data: chapter });
    } catch (err) {
      next(err);
    }
  },

  async createChapter(req, res, next) {
    try {
      const userId = req.user.id;
      const subjectId = req.body.subjectId || req.body.subject_id;
      const payload = { name: req.body.name, description: req.body.description, order_index: req.body.orderIndex ?? req.body.order_index };
      const created = await ChapterService.createChapter(subjectId, userId, payload);
      return res.status(201).json({ success: true, data: created });
    } catch (err) {
      next(err);
    }
  },

  async updateChapter(req, res, next) {
    try {
      const userId = req.user.id;
      const updated = await ChapterService.updateChapter(req.params.id, userId, req.body);
      return res.json({ success: true, data: updated });
    } catch (err) {
      next(err);
    }
  },

  async deleteChapter(req, res, next) {
    try {
      const userId = req.user.id;
      await ChapterService.deleteChapter(req.params.id, userId);
      return res.status(204).end();
    } catch (err) {
      next(err);
    }
  },

  async reorderChapter(req, res, next) {
    try {
      const userId = req.user.id;
      const updated = await ChapterService.reorderChapter(req.params.id, userId, req.body.orderIndex);
      return res.json({ success: true, data: updated });
    } catch (err) {
      next(err);
    }
  },

  async getChapterNotes(req, res, next) {
    try {
      const userId = req.user.id;
      const chapterId = req.params.id;
      const notes = await NoteService.getNotesByUserId(userId, { filters: { chapter_id: chapterId }, limit: req.query.limit, offset: req.query.offset });
      return res.json({ success: true, notes });
    } catch (err) {
      next(err);
    }
  }
};
