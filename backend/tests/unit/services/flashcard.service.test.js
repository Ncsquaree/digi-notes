jest.mock('../../../src/config/database', () => require('../../helpers/mockDatabase'));
jest.mock('../../../src/models', () => ({ Flashcard: require('../../helpers/mockModels').mockFlashcard, Note: require('../../helpers/mockModels').mockNote, StudySession: require('../../helpers/mockModels').mockStudySession }));

const FlashcardService = require('../../../src/services/flashcard.service');
const { NotFoundError, ForbiddenError } = require('../../../src/utils/errors');
const { mockPool } = require('../../helpers/mockDatabase');

describe('FlashcardService', () => {
  afterEach(() => jest.clearAllMocks());

  describe('getFlashcardById', () => {
    it('returns flashcard when owned', async () => {
      const f = await FlashcardService.getFlashcardById('fc-1', 'user-1');
      expect(f.id).toBe('fc-1');
    });

    it('throws NotFoundError when missing', async () => {
      const models = require('../../../src/models');
      models.Flashcard.findById.mockResolvedValueOnce(null);
      await expect(FlashcardService.getFlashcardById('x', 'user-1')).rejects.toBeInstanceOf(NotFoundError);
    });

    it('throws ForbiddenError when not owner', async () => {
      const models = require('../../../src/models');
      models.Flashcard.findById.mockResolvedValueOnce({ id: 'fc-1', user_id: 'other' });
      await expect(FlashcardService.getFlashcardById('fc-1', 'user-1')).rejects.toBeInstanceOf(ForbiddenError);
    });
  });

  describe('reviewFlashcard', () => {
    it('throws NotFoundError when not found in DB select', async () => {
      const db = require('../../../src/config/database');
      db.pool.connect.mockResolvedValueOnce({ query: jest.fn().mockResolvedValueOnce({ rowCount: 0, rows: [] }), release: jest.fn() });
      await expect(FlashcardService.reviewFlashcard('x', 'user-1', 4)).rejects.toBeInstanceOf(NotFoundError);
    });

    it('throws ForbiddenError when card not owned', async () => {
      const client = { query: jest.fn().mockResolvedValueOnce({ rowCount: 1, rows: [{ id: 'fc-1', user_id: 'other', ef: 2.5, repetitions: 0, interval: 0, difficulty: 1 }] }), release: jest.fn() };
      const db = require('../../../src/config/database');
      db.pool.connect.mockResolvedValueOnce(client);
      await expect(FlashcardService.reviewFlashcard('fc-1', 'user-1', 4)).rejects.toBeInstanceOf(ForbiddenError);
    });

    it('processes review and returns updated card and session', async () => {
      const selectRes = { rowCount: 1, rows: [{ id: 'fc-1', user_id: 'user-1', ef: 2.5, repetitions: 1, interval: 1, difficulty: 1 }] };
      const updateRes = { rows: [{ id: 'fc-1', repetitions: 2, interval: 6, easiness_factor: 2.6, next_review_date: '2025-12-10' }] };
      const sessionRes = { rows: [{ id: 's1', user_id: 'user-1', flashcard_id: 'fc-1', quality: 4, time_spent_seconds: null }] };
      const client = { query: jest.fn(async (sql, params) => {
        if (sql && sql.toLowerCase().includes('select * from flashcards')) return selectRes;
        if (sql && sql.toLowerCase().includes('update flashcards')) return updateRes;
        if (sql && sql.toLowerCase().includes('insert into study_sessions')) return sessionRes;
        if (sql && sql.toLowerCase().includes('begin')) return { rows: [] };
        if (sql && sql.toLowerCase().includes('commit')) return { rows: [] };
        return { rowCount: 0, rows: [] };
      }), release: jest.fn() };
      const db = require('../../../src/config/database');
      db.pool.connect.mockResolvedValueOnce(client);

      const result = await FlashcardService.reviewFlashcard('fc-1', 'user-1', 4);
      expect(result.flashcard).toBeDefined();
      expect(result.session).toBeDefined();
    });
  });
});
