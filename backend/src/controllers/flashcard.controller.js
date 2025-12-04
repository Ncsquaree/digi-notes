const FlashcardService = require('../services/flashcard.service');
const logger = require('../utils/logger');
const { ValidationError, NotFoundError, ForbiddenError } = require('../utils/errors');

module.exports = {
  async listFlashcards(req, res, next) {
    try {
      const userId = (req.user && (req.user.id || req.user.userId));
      const noteId = req.query.noteId;
      const page = parseInt(req.query.page || '1', 10);
      const pageSize = Math.min(parseInt(req.query.pageSize || req.query.pageSize || '10', 10), 100);
      const limit = pageSize;
      const offset = (page - 1) * limit;

      let result;
      if (noteId) {
        result = await FlashcardService.getFlashcardsByNoteId(noteId, userId, { limit, offset });
      } else {
        result = await FlashcardService.getFlashcardsByUserId(userId, { limit, offset });
      }

      let flashcards = result.rows || result;
      let pagination = result.pagination || { page, pageSize: limit, total: (Array.isArray(flashcards) ? flashcards.length : 0) };

      // Normalize pagination keys to the shape tests expect (pageSize)
      if (pagination.pageSize === undefined && pagination.limit !== undefined) {
        pagination.pageSize = pagination.limit;
        delete pagination.limit;
      }

      logger.info('flashcards_listed', { userId, count: Array.isArray(flashcards) ? flashcards.length : 0 });
      return res.json({ success: true, flashcards, pagination });
    } catch (err) {
      next(err);
    }
  },

  async getDueFlashcards(req, res, next) {
    try {
      const userId = (req.user && (req.user.id || req.user.userId));
      const limit = Math.min(parseInt(req.query.limit || '20', 10), 100);
      const flashcards = await FlashcardService.getDueFlashcards(userId, { limit });
      return res.json({ success: true, flashcards });
    } catch (err) {
      next(err);
    }
  },

  async getFlashcard(req, res, next) {
    try {
      const userId = (req.user && (req.user.id || req.user.userId));
      const flashcardId = req.params.id;
      const flashcard = await FlashcardService.getFlashcardById(flashcardId, userId);
      if (!flashcard) throw new NotFoundError('Flashcard not found');
      if (flashcard.user_id && flashcard.user_id !== userId) throw new ForbiddenError('Access denied');
      return res.json({ success: true, flashcard });
    } catch (err) {
      next(err);
    }
  },

  async createFlashcard(req, res, next) {
    try {
      const userId = (req.user && (req.user.id || req.user.userId));
      const { noteId, question, answer, difficulty } = req.body;
      const flashcard = await FlashcardService.createFlashcard(noteId, userId, { question, answer, difficulty });
      return res.status(201).json({ success: true, flashcard });
    } catch (err) {
      next(err);
    }
  },

  async updateFlashcard(req, res, next) {
    try {
      const userId = (req.user && (req.user.id || req.user.userId));
      const flashcardId = req.params.id;
      const allowed = {};
      if (req.body.question) allowed.question = req.body.question;
      if (req.body.answer) allowed.answer = req.body.answer;
      const flashcard = await FlashcardService.updateFlashcard(flashcardId, userId, allowed);
      return res.json({ success: true, flashcard });
    } catch (err) {
      next(err);
    }
  },

  async deleteFlashcard(req, res, next) {
    try {
      const userId = (req.user && (req.user.id || req.user.userId));
      const flashcardId = req.params.id;
      await FlashcardService.deleteFlashcard(flashcardId, userId);
      return res.json({ success: true });
    } catch (err) {
      next(err);
    }
  },

  async reviewFlashcard(req, res, next) {
    try {
      const userId = req.user.id;
      const flashcardId = req.params.id;
      const quality = parseInt(req.body.quality, 10);
      const timeSpentSeconds = req.body.timeSpentSeconds ? parseInt(req.body.timeSpentSeconds, 10) : null;
      if (Number.isNaN(quality) || quality < 0 || quality > 5) {
        throw new ValidationError('quality must be an integer between 0 and 5');
      }
      // Support service returning either { flashcard, session } or flashcard directly
      const rv = await FlashcardService.reviewFlashcard(flashcardId, userId, quality, timeSpentSeconds);
      let flashcard, session;
      if (rv && rv.flashcard) {
        flashcard = rv.flashcard;
        session = rv.session;
      } else {
        flashcard = rv;
        session = null;
      }
      logger.info('flashcard_reviewed', { flashcardId, userId, quality, interval: flashcard.interval });
      const payload = { success: true, flashcard, metadata: { next_review_date: flashcard.next_review_date, interval: flashcard.interval } };
      if (session) payload.session = session;
      return res.json(payload);
    } catch (err) {
      next(err);
    }
  },

  async getFlashcardStats(req, res, next) {
    try {
      const userId = req.user.id;
      const startDate = req.query.startDate;
      const endDate = req.query.endDate;
      const total_flashcards = await FlashcardService.countFlashcardsByUser(userId);
      const due = await FlashcardService.getDueFlashcards(userId, { limit: 10000 });
      const stats = await FlashcardService.getStudyStats(userId, { startDate, endDate });
      const total = (total_flashcards && (total_flashcards.total || total_flashcards)) || 0;
      const payload = {
        total,
        due_count: Array.isArray(due) ? due.length : (due || 0),
        reviews: stats.reviews || stats.total_reviews || 0,
      };
      return res.json({ success: true, stats: payload });
    } catch (err) {
      next(err);
    }
  },
};
