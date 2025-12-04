const SubjectService = require('../services/subject.service');
const ChapterService = require('../services/chapter.service');
const NoteService = require('../services/note.service');

module.exports = {
  async listSubjects(req, res, next) {
    try {
      const userId = req.user.id;
      const subjects = await SubjectService.getSubjectsByUserId(userId, { page: req.query.page, limit: req.query.limit });
      return res.json({ success: true, data: subjects });
    } catch (err) {
      next(err);
    }
  },

  async getSubject(req, res, next) {
    try {
      const userId = req.user.id;
      const subject = await SubjectService.getSubjectById(req.params.id, userId);
      return res.json({ success: true, data: subject });
    } catch (err) {
      next(err);
    }
  },

  async createSubject(req, res, next) {
    try {
      const userId = req.user.id;
      const payload = { name: req.body.name, description: req.body.description, color: req.body.color, icon: req.body.icon };
      const created = await SubjectService.createSubject(userId, payload);
      return res.status(201).json({ success: true, data: created });
    } catch (err) {
      next(err);
    }
  },

  async updateSubject(req, res, next) {
    try {
      const userId = req.user.id;
      const updated = await SubjectService.updateSubject(req.params.id, userId, req.body);
      return res.json({ success: true, data: updated });
    } catch (err) {
      next(err);
    }
  },

  async deleteSubject(req, res, next) {
    try {
      const userId = req.user.id;
      await SubjectService.deleteSubject(req.params.id, userId);
      return res.status(204).end();
    } catch (err) {
      next(err);
    }
  },

  async getSubjectChapters(req, res, next) {
    try {
      const userId = req.user.id;
      const subjectId = req.params.id;
      const chapters = await ChapterService.getChaptersBySubjectId(subjectId, userId, { page: req.query.page, limit: req.query.limit });
      return res.json({ success: true, data: chapters });
    } catch (err) {
      next(err);
    }
  },

  async getSubjectNotes(req, res, next) {
    try {
      const userId = req.user.id;
      const subjectId = req.params.id;
      const notes = await NoteService.getNotesByUserId(userId, { filters: { subject_id: subjectId }, limit: req.query.limit, offset: req.query.offset });
      return res.json({ success: true, notes });
    } catch (err) {
      next(err);
    }
  }
};
