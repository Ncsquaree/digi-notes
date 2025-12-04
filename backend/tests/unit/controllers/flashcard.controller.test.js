jest.mock('../../../src/services/flashcard.service', () => require('../../helpers/mockServices').mockFlashcardService);
const flashcardController = require('../../../src/controllers/flashcard.controller');
const FlashcardService = require('../../../src/services/flashcard.service');
const { NotFoundError, ForbiddenError, ValidationError } = require('../../../src/utils/errors');

describe('flashcard.controller', () => {
  afterEach(() => jest.clearAllMocks());

  const user = { id: 'user-1' };
  const resFactory = () => ({ status: jest.fn().mockReturnThis(), json: jest.fn() });
  const next = jest.fn();

  describe('createFlashcard', () => {
    it('creates flashcard and returns 201', async () => {
      const req = { user, body: { noteId: 'note-1', question: 'Q?', answer: 'A' } };
      const res = resFactory();
      FlashcardService.createFlashcard.mockResolvedValueOnce({ id: 'fc-1', question: 'Q?' });

      await flashcardController.createFlashcard(req, res, next);
      expect(res.status).toHaveBeenCalledWith(201);
      expect(res.json).toHaveBeenCalledWith({ success: true, flashcard: { id: 'fc-1', question: 'Q?' } });
    });

    it('propagates NotFoundError when note missing', async () => {
      const req = { user, body: { noteId: 'missing', question: 'Q?', answer: 'A' } };
      const res = resFactory();
      FlashcardService.createFlashcard.mockRejectedValueOnce(new NotFoundError('note not found'));

      await flashcardController.createFlashcard(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(NotFoundError));
    });
  });

  describe('listFlashcards', () => {
    it('returns list with pagination', async () => {
      const req = { user, query: { page: '1', pageSize: '10' } };
      const res = resFactory();
      FlashcardService.getFlashcardsByUserId.mockResolvedValueOnce({ rows: [{ id: 'f1' }], pagination: { page: 1, pageSize: 10, total: 1 } });

      await flashcardController.listFlashcards(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, flashcards: [{ id: 'f1' }], pagination: { page: 1, pageSize: 10, total: 1 } });
    });
  });

  describe('reviewFlashcard', () => {
    it('processes valid quality and returns metadata', async () => {
      const req = { params: { id: 'f1' }, body: { quality: 4 }, user };
      const res = resFactory();
      FlashcardService.reviewFlashcard.mockResolvedValueOnce({ id: 'f1', interval: 10, next_review_date: '2025-12-10' });

      await flashcardController.reviewFlashcard(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, flashcard: expect.objectContaining({ id: 'f1' }), metadata: expect.objectContaining({ next_review_date: '2025-12-10' }) });
    });

    it('returns ValidationError for invalid quality', async () => {
      const req = { params: { id: 'f1' }, body: { quality: 7 }, user };
      const res = resFactory();
      FlashcardService.reviewFlashcard.mockRejectedValueOnce(new ValidationError('quality range'));

      await flashcardController.reviewFlashcard(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(ValidationError));
    });
  });
  
  describe('getDueFlashcards', () => {
    it('returns due flashcards metadata', async () => {
      const req = { user, query: {} };
      const res = resFactory();
      FlashcardService.getDueFlashcards.mockResolvedValueOnce([{ id: 'f1' }, { id: 'f2' }]);

      await flashcardController.getDueFlashcards(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, flashcards: [{ id: 'f1' }, { id: 'f2' }] });
    });
  });

  describe('getFlashcard', () => {
    it('returns flashcard when found and owned', async () => {
      const req = { params: { id: 'f1' }, user };
      const res = resFactory();
      FlashcardService.getFlashcardById.mockResolvedValueOnce({ id: 'f1', user_id: 'user-1' });

      await flashcardController.getFlashcard(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, flashcard: expect.objectContaining({ id: 'f1' }) });
    });

    it('propagates NotFoundError when not found', async () => {
      const req = { params: { id: 'missing' }, user };
      const res = resFactory();
      FlashcardService.getFlashcardById.mockResolvedValueOnce(null);

      await flashcardController.getFlashcard(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(NotFoundError));
    });

    it('propagates ForbiddenError when ownership mismatch', async () => {
      const req = { params: { id: 'f1' }, user };
      const res = resFactory();
      FlashcardService.getFlashcardById.mockResolvedValueOnce({ id: 'f1', user_id: 'other' });

      await flashcardController.getFlashcard(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(ForbiddenError));
    });
  });

  describe('updateFlashcard', () => {
    it('updates flashcard successfully', async () => {
      const req = { params: { id: 'f1' }, body: { front: 'Q' }, user };
      const res = resFactory();
      FlashcardService.updateFlashcard.mockResolvedValueOnce({ id: 'f1', front: 'Q' });

      await flashcardController.updateFlashcard(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, flashcard: expect.objectContaining({ id: 'f1' }) });
    });

    it('propagates NotFoundError', async () => {
      const req = { params: { id: 'missing' }, body: {}, user };
      const res = resFactory();
      FlashcardService.updateFlashcard.mockRejectedValueOnce(new NotFoundError('none'));

      await flashcardController.updateFlashcard(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(NotFoundError));
    });
  });

  describe('deleteFlashcard', () => {
    it('deletes flashcard and returns success', async () => {
      const req = { params: { id: 'f1' }, user };
      const res = resFactory();
      FlashcardService.deleteFlashcard.mockResolvedValueOnce(true);

      await flashcardController.deleteFlashcard(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true });
    });
  });

  describe('getFlashcardStats', () => {
    it('returns aggregated stats with date filters', async () => {
      const req = { user, query: { startDate: '2025-01-01', endDate: '2025-12-31' } };
      const res = resFactory();
      FlashcardService.countFlashcardsByUser.mockResolvedValueOnce({ total: 10 });
      FlashcardService.getDueFlashcards.mockResolvedValueOnce([{ id: 'f1' }]);
      FlashcardService.getStudyStats.mockResolvedValueOnce({ total: 5, reviews: 20, avg_quality: 4.2, study_time: 3600 });

      await flashcardController.getFlashcardStats(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, stats: expect.objectContaining({ total: 10, due_count: 1, reviews: 20 }) });
    });
  });
});
