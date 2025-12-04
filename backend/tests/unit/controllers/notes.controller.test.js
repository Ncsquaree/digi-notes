jest.mock('../../../src/services/note.service', () => require('../../helpers/mockServices').mockNoteService);
jest.mock('axios');
const NoteService = require('../../../src/services/note.service');
const axios = require('axios');
const notesController = require('../../../src/controllers/notes.controller');
const { NotFoundError, ForbiddenError, ValidationError, ConflictError } = require('../../../src/utils/errors');

describe('notes.controller', () => {
  afterEach(() => jest.clearAllMocks());

  const user = { id: 'user-1' };
  const resFactory = () => ({ status: jest.fn().mockReturnThis(), json: jest.fn() });
  const next = jest.fn();

  describe('listNotes', () => {
    it('returns notes list with pagination defaults', async () => {
      const req = { user, query: {} };
      const res = resFactory();
      NoteService.getNotesByUserId.mockResolvedValueOnce({ rows: [{ id: 'note-1' }], pagination: { page: 1, pageSize: 20, total: 1 } });

      await notesController.listNotes(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, notes: [{ id: 'note-1' }], pagination: { page: 1, pageSize: 20, total: 1 } });
    });
  });

  describe('getNote', () => {
    it('returns note when found and belongs to user', async () => {
      const req = { params: { id: 'n1' }, user };
      const res = resFactory();
      NoteService.getNoteById.mockResolvedValueOnce({ id: 'n1', user_id: 'user-1', title: 'T' });

      await notesController.getNote(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, note: expect.objectContaining({ id: 'n1' }) });
    });

    it('calls next with NotFoundError when not found', async () => {
      const req = { params: { id: 'missing' }, user };
      const res = resFactory();
      NoteService.getNoteById.mockResolvedValueOnce(null);

      await notesController.getNote(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(NotFoundError));
    });

    it('calls next with ForbiddenError when ownership mismatch', async () => {
      const req = { params: { id: 'n1' }, user };
      const res = resFactory();
      NoteService.getNoteById.mockResolvedValueOnce({ id: 'n1', user_id: 'other' });

      await notesController.getNote(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(ForbiddenError));
    });
  });

  describe('createNote', () => {
    it('creates note and returns 201', async () => {
      const body = { title: 'New', subject_id: 's1', original_image_url: 'https://s3.amazonaws.com/b/key.jpg' };
      const req = { body, user };
      const res = resFactory();
      NoteService.createNote.mockResolvedValueOnce({ id: 'n2', user_id: 'user-1', ...body });

      await notesController.createNote(req, res, next);
      expect(res.status).toHaveBeenCalledWith(201);
      expect(res.json).toHaveBeenCalledWith({ success: true, note: expect.objectContaining({ id: 'n2' }) });
    });

    it('propagates NotFoundError from service', async () => {
      const body = { title: 'New', subject_id: 'missing' };
      const req = { body, user };
      const res = resFactory();
      NoteService.createNote.mockRejectedValueOnce(new NotFoundError('subject missing'));

      await notesController.createNote(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(NotFoundError));
    });
  });

  describe('updateNote', () => {
    it('updates partial fields successfully', async () => {
      const req = { params: { id: 'n1' }, body: { title: 'Updated' }, user };
      const res = resFactory();
      NoteService.updateNote.mockResolvedValueOnce({ id: 'n1', user_id: 'user-1', title: 'Updated' });

      await notesController.updateNote(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, note: expect.objectContaining({ title: 'Updated' }) });
    });

    it('propagates NotFoundError', async () => {
      const req = { params: { id: 'missing' }, body: {}, user };
      const res = resFactory();
      NoteService.updateNote.mockRejectedValueOnce(new NotFoundError('no note'));

      await notesController.updateNote(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(NotFoundError));
    });
  });

  describe('deleteNote', () => {
    it('deletes note successfully', async () => {
      const req = { params: { id: 'n1' }, user };
      const res = resFactory();
      NoteService.deleteNote.mockResolvedValueOnce(true);

      await notesController.deleteNote(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true });
    });

    it('propagates ForbiddenError', async () => {
      const req = { params: { id: 'n1' }, user };
      const res = resFactory();
      NoteService.deleteNote.mockRejectedValueOnce(new ForbiddenError('nope'));

      await notesController.deleteNote(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(ForbiddenError));
    });
  });

  describe('processNote', () => {
    it('returns validation error when image url missing', async () => {
      const req = { params: { id: 'n1' }, user };
      const res = resFactory();
      NoteService.getNoteById.mockResolvedValueOnce({ id: 'n1', user_id: 'user-1', original_image_url: null });

      await notesController.processNote(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(ValidationError));
    });

    it('returns conflict when already processing', async () => {
      const req = { params: { id: 'n1' }, user };
      const res = resFactory();
      NoteService.getNoteById.mockResolvedValueOnce({ id: 'n1', user_id: 'user-1', processing_status: 'processing', original_image_url: 'u' });

      await notesController.processNote(req, res, next);
      expect(next).toHaveBeenCalledWith(expect.any(ConflictError));
    });

    it('initiates AI processing (fire-and-forget) and returns task id', async () => {
      const req = { params: { id: 'n1' }, user };
      const res = resFactory();
      NoteService.getNoteById.mockResolvedValueOnce({ id: 'n1', user_id: 'user-1', processing_status: 'idle', original_image_url: 'https://s3/key?x=1' });
      NoteService.markProcessing.mockResolvedValueOnce({ processing_task_id: 'task-1' });
      axios.post.mockResolvedValueOnce({ data: { taskId: 'task-1' } });

      await notesController.processNote(req, res, next);
      expect(res.status).toHaveBeenCalledWith(202);
      expect(res.json).toHaveBeenCalledWith(expect.objectContaining({ success: true, data: expect.objectContaining({ processing_task_id: 'task-1' }) }));
      // axios.post invoked in background orchestration
      expect(axios.post).toHaveBeenCalled();
    });
  });

  describe('getProcessingStatus', () => {
    it('returns DB status merged with AI progress when available', async () => {
      const req = { params: { id: 'n1' }, user };
      const res = resFactory();
      NoteService.getProcessingStatus.mockResolvedValueOnce({ processing_task_id: 'task-1', processing_status: 'completed', parsed_content: 'ok' });

      await notesController.getProcessingStatus(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, status: expect.objectContaining({ processing_status: 'completed' }) });
    });

    it('falls back to DB result when AI times out', async () => {
      const req = { params: { id: 'n1' }, user };
      const res = resFactory();
      NoteService.getProcessingStatus.mockResolvedValueOnce({ processing_task_id: 'task-1', processing_status: 'processing' });

      await notesController.getProcessingStatus(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, status: expect.objectContaining({ processing_status: 'processing' }) });
    });
  });
});
