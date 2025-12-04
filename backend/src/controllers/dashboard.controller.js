const SubjectService = require('../services/subject.service');
const NoteService = require('../services/note.service');
const FlashcardService = require('../services/flashcard.service');
const logger = require('../utils/logger');

module.exports = {
  async getDashboardStats(req, res, next) {
    try {
      const userId = req.user.id;
      const recentSubjects = await SubjectService.getSubjectsByUserId(userId, { limit: 5 });
      const notes = await NoteService.getNotesByUserId(userId, { limit: 1, offset: 0 });
      const totalNotes = await require('../models').Note.countByUserId(userId).catch(() => null);
      const totalSubjects = await require('../models').Subject.countByUserId(userId).catch(() => null);
      const totalFlashcards = await FlashcardService.countFlashcardsByUser(userId);
      const dueFlashcards = await FlashcardService.getDueFlashcards(userId, { limit: 10000 });

      const stats = {
        total_subjects: totalSubjects ?? (Array.isArray(recentSubjects) ? recentSubjects.length : 0),
        total_notes: totalNotes ?? 0,
        total_flashcards: totalFlashcards ?? 0,
        due_flashcards: Array.isArray(dueFlashcards) ? dueFlashcards.length : (dueFlashcards || 0),
        recent_subjects: Array.isArray(recentSubjects) ? recentSubjects : []
      };

      logger.info('dashboard_stats_fetched', { userId });
      return res.json({ success: true, stats });
    } catch (err) {
      next(err);
    }
  }
};
