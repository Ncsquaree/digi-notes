jest.mock('../../../src/models', () => ({
  Note: require('../../helpers/mockModels').mockNote,
  Subject: require('../../helpers/mockModels').mockSubject,
  Chapter: require('../../helpers/mockModels').mockChapter,
  KnowledgeGraphNode: {},
}));
jest.mock('../../../src/config/database', () => require('../../helpers/mockDatabase'));

const NoteService = require('../../../src/services/note.service');
const { NotFoundError, ForbiddenError } = require('../../../src/utils/errors');
const { mockPool } = require('../../helpers/mockDatabase');
const { mockNote } = require('../../helpers/mockModels');

describe('NoteService', () => {
  afterEach(() => jest.clearAllMocks());

  describe('getNoteById', () => {
    it('returns note when owned by user', async () => {
      const n = await NoteService.getNoteById('note-1', 'user-1');
      expect(n.id).toBe('note-1');
    });

    it('throws NotFoundError when missing', async () => {
      const models = require('../../../src/models');
      models.Note.findById.mockResolvedValueOnce(null);
      await expect(NoteService.getNoteById('missing', 'user-1')).rejects.toBeInstanceOf(NotFoundError);
    });

    it('throws ForbiddenError when not owner', async () => {
      const models = require('../../../src/models');
      models.Note.findById.mockResolvedValueOnce({ id: 'n1', user_id: 'other' });
      await expect(NoteService.getNoteById('n1', 'user-1')).rejects.toBeInstanceOf(ForbiddenError);
    });
  });

  describe('attachAIResult', () => {
    it('attaches parsed content and returns updated note', async () => {
      const models = require('../../../src/models');
      models.Note.findById.mockResolvedValueOnce({ id: 'note-1', user_id: 'user-1' });
      // mock pool.connect client to return update result
      const db = require('../../../src/config/database');
      db.mockPool = db.mockPool || mockPool;
      db.pool = {
        connect: jest.fn().mockResolvedValue({
          query: jest.fn(async (sql, params) => {
            if (sql && sql.toLowerCase().includes('update notes')) return { rows: [{ id: 'note-1', parsed_content: 'parsed', processing_status: 'processed' }] };
            return { rowCount: 0, rows: [] };
          }),
          release: jest.fn(),
        }),
      };

      const result = await NoteService.attachAIResult('note-1', 'user-1', { parsed_content: 'parsed', nodesArray: [] });
      expect(result.parsed_content).toBe('parsed');
    });

    it('throws NotFoundError when note missing', async () => {
      const models = require('../../../src/models');
      models.Note.findById.mockResolvedValueOnce(null);
      await expect(NoteService.attachAIResult('x', 'user-1', {})).rejects.toBeInstanceOf(NotFoundError);
    });
  });
});
