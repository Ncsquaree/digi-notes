jest.mock('axios');
jest.mock('../../../src/services/flashcard.service', () => require('../../helpers/mockServices').mockFlashcardService);
const axios = require('axios');
const FlashcardService = require('../../../src/services/flashcard.service');
const aiController = require('../../../src/controllers/ai.controller');
const { ValidationError } = require('../../../src/utils/errors');

describe('ai.controller', () => {
  afterEach(() => jest.clearAllMocks());

  describe('generateFlashcards', () => {
    it('returns flashcards from AI service without persisting when noteId absent', async () => {
      const req = { user: { id: 'user-1' }, body: { content: 'Some content', count: 2 } };
      const res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
      const next = jest.fn();
      axios.post.mockResolvedValueOnce({ data: { flashcards: [{ question: 'Q1', answer: 'A1' }, { question: 'Q2', answer: 'A2' }] } });

      await aiController.generateFlashcards(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, flashcards: expect.any(Array), metadata: expect.any(Object) });
    });

    it('persists flashcards when noteId provided', async () => {
      const req = { user: { id: 'user-1' }, body: { content: 'C', count: 2, noteId: 'n1' } };
      const res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
      const next = jest.fn();
      axios.post.mockResolvedValueOnce({ data: { flashcards: [{ question: 'Q1', answer: 'A1' }, { question: 'Q2', answer: 'A2' }] } });
      FlashcardService.createFlashcardsFromAI.mockResolvedValueOnce({ inserted: 2 });

      await aiController.generateFlashcards(req, res, next);
      expect(FlashcardService.createFlashcardsFromAI).toHaveBeenCalled();
      expect(res.json).toHaveBeenCalledWith({ success: true, flashcards: expect.any(Array), metadata: expect.objectContaining({ persisted: true, persisted_count: 2 }) });
    });

    it('returns ValidationError when content missing', async () => {
      const req = { user: { id: 'user-1' }, body: { count: 2 } };
      const res = { json: jest.fn() };
      const next = jest.fn();

      await aiController.generateFlashcards(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(ValidationError));
    });

    it('maps axios timeout to 504', async () => {
      const req = { user: { id: 'user-1' }, body: { content: 'C' } };
      const res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
      const next = jest.fn();
      const err = new Error('timeout');
      err.code = 'ECONNABORTED';
      axios.post.mockRejectedValueOnce(err);

      await aiController.generateFlashcards(req, res, next);
      expect(res.status).toHaveBeenCalledWith(504);
      expect(res.json).toHaveBeenCalledWith(expect.objectContaining({ success: false }));
    });

    it('maps AI internal error to 502/500 payload', async () => {
      const req = { user: { id: 'user-1' }, body: { content: 'C' } };
      const res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
      const next = jest.fn();
      const err = { response: { status: 500, data: { error: 'AI failed' } } };
      axios.post.mockRejectedValueOnce(err);

      await aiController.generateFlashcards(req, res, next);
      expect(res.status).toHaveBeenCalledWith(502);
      expect(res.json).toHaveBeenCalledWith(expect.objectContaining({ success: false }));
    });
  });

  describe('generateQuiz', () => {
    it('returns quiz questions from AI', async () => {
      const req = { user: { id: 'u1' }, body: { content: 'C', count: 3 } };
      const res = { json: jest.fn() };
      const next = jest.fn();
      axios.post.mockResolvedValueOnce({ data: { quiz: { questions: [{ q: 'Q?' }] } } });

      await aiController.generateQuiz(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, quiz: expect.objectContaining({ questions: expect.any(Array) }) });
    });

    it('returns ValidationError for invalid request', async () => {
      const req = { user: { id: 'u1' }, body: {} };
      const res = { json: jest.fn() };
      const next = jest.fn();

      await aiController.generateQuiz(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(ValidationError));
    });
  });

  describe('generateMindmap', () => {
    it('returns mindmap from AI', async () => {
      const req = { user: { id: 'u1' }, body: { content: 'C' } };
      const res = { json: jest.fn() };
      const next = jest.fn();
      axios.post.mockResolvedValueOnce({ data: { mindmap: { nodes: [], edges: [] } } });

      await aiController.generateMindmap(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, mindmap: expect.objectContaining({ nodes: expect.any(Array) }) });
    });

    it('maps axios errors to next for unexpected shapes', async () => {
      const req = { user: { id: 'u1' }, body: { content: 'C' } };
      const res = { json: jest.fn() };
      const next = jest.fn();
      axios.post.mockRejectedValueOnce(new Error('boom'));

      await aiController.generateMindmap(req, res, next);
      expect(next).toHaveBeenCalled();
    });
  });
});
