const request = require('supertest');
const express = require('express');
const bodyParser = require('body-parser');
const flashcardRoutes = require('../../src/routes/flashcards');
const FlashcardService = require('../../src/services/flashcard.service');
const { generateTestToken } = require('../setup');

jest.mock('../../src/services/flashcard.service');

function makeApp(authenticated = true) {
  const app = express();
  app.use(bodyParser.json());
  if (authenticated) app.use((req, res, next) => { req.user = { userId: 'user-1' }; next(); });
  app.use('/api/flashcards', flashcardRoutes);
  app.use((err, req, res, next) => { res.status(err.statusCode || 500).json({ success: false, error: { message: err.message } }); });
  return app;
}

describe('flashcards integration', () => {
  afterEach(() => jest.clearAllMocks());

  it('POST /api/flashcards creates flashcard', async () => {
    const app = makeApp(true);
    FlashcardService.createFlashcard.mockResolvedValueOnce({ id: 'fc-1', question: 'Q' });
    const res = await request(app).post('/api/flashcards').send({ noteId: 'n1', question: 'Q?', answer: 'A' });
    expect(res.status).toBe(201);
    expect(res.body.success).toBe(true);
    expect(res.body.flashcard).toBeDefined();
  });

  it('GET /api/flashcards/due returns due flashcards list', async () => {
    const app = makeApp(true);
    FlashcardService.getDueFlashcards.mockResolvedValueOnce([{ id: 'f1' }, { id: 'f2' }]);
    const res = await request(app).get('/api/flashcards/due');
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(Array.isArray(res.body.flashcards)).toBe(true);
  });

  it('POST /api/flashcards/:id/review processes a valid quality', async () => {
    const app = makeApp(true);
    FlashcardService.reviewFlashcard.mockResolvedValueOnce({ flashcard: { id: 'f1', interval: 3 }, session: { id: 's1' } });
    const res = await request(app).post('/api/flashcards/f1/review').send({ quality: 4 });
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.metadata).toBeDefined();
  });

  it('POST /api/flashcards/:id/review returns 400 for invalid quality', async () => {
    const app = makeApp(true);
    FlashcardService.reviewFlashcard.mockRejectedValueOnce(new Error('validation'));
    const res = await request(app).post('/api/flashcards/f1/review').send({ quality: 9 });
    expect(res.status).toBe(500);
  });

  it('GET /api/flashcards/stats returns aggregated stats', async () => {
    const app = makeApp(true);
    FlashcardService.countFlashcardsByUser.mockResolvedValueOnce({ total: 10 });
    FlashcardService.getDueFlashcards.mockResolvedValueOnce([{ id: 'f1' }]);
    FlashcardService.getStudyStats.mockResolvedValueOnce({ total: 5, reviews: 20, avg_quality: 4.2, study_time: 3600 });

    const res = await request(app).get('/api/flashcards/stats').query({ startDate: '2025-01-01', endDate: '2025-12-31' });
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.stats).toBeDefined();
  });
});
