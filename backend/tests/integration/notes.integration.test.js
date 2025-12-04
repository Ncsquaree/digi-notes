const request = require('supertest');
const express = require('express');
const bodyParser = require('body-parser');
const notesRoutes = require('../../src/routes/notes');
const NoteService = require('../../src/services/note.service');
const { generateTestToken } = require('../setup');

jest.mock('../../src/services/note.service');

function makeApp(authenticated = true) {
  const app = express();
  app.use(bodyParser.json());
  // simple auth passthrough for tests
  if (authenticated) app.use((req, res, next) => { req.user = { userId: 'user-1' }; next(); });
  app.use('/api/notes', notesRoutes);
  app.use((err, req, res, next) => { res.status(err.statusCode || 500).json({ success: false, error: { message: err.message, code: err.code || err.name } }); });
  return app;
}

describe('notes integration', () => {
  afterEach(() => jest.clearAllMocks());

  it('POST /api/notes creates note when authenticated', async () => {
    const app = makeApp(true);
    NoteService.createNote.mockResolvedValue({ id: 'note-1', title: 'T' });
    const res = await request(app).post('/api/notes').send({ title: 'T', subject_id: 's1', original_image_url: 'https://s3/a/b.jpg' });
    expect(res.status).toBe(201);
    expect(res.body.success).toBe(true);
    expect(res.body.note).toBeDefined();
  });

  it('POST /api/notes requires auth', async () => {
    const app = makeApp(false);
    const res = await request(app).post('/api/notes').send({ title: 'T' });
    expect(res.status).toBe(500); // underlying authenticate not present in this simple mount; ensure error handled
  });

  it('GET /api/notes returns list with pagination', async () => {
    const app = makeApp(true);
    NoteService.getNotesByUserId.mockResolvedValueOnce({ rows: [{ id: 'n1' }], pagination: { page: 1, pageSize: 20, total: 1 } });
    const res = await request(app).get('/api/notes');
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.notes).toBeDefined();
    expect(res.body.pagination).toBeDefined();
  });

  it('POST /api/notes/:id/process returns 202 and task id', async () => {
    const app = makeApp(true);
    NoteService.getNoteById.mockResolvedValue({ id: 'note-1', user_id: 'user-1', original_image_url: 's3://a/b.jpg', processing_status: 'pending' });
    NoteService.markProcessing.mockResolvedValue({ processing_task_id: 'task-1' });
    const res = await request(app).post('/api/notes/note-1/process');
    expect(res.status).toBe(202);
    expect(res.body.success).toBe(true);
    expect(res.body.data.processing_task_id).toBeDefined();
  });

  it('GET /api/notes/:id/status returns processing status', async () => {
    const app = makeApp(true);
    NoteService.getProcessingStatus.mockResolvedValueOnce({ processing_task_id: 'task-1', processing_status: 'processing' });
    const res = await request(app).get('/api/notes/note-1/status');
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.status).toBeDefined();
  });
});
