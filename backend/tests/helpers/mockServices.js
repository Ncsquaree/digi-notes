// Centralized service mocks for tests
const { NotFoundError, ForbiddenError } = require('../../src/utils/errors');

const makeMock = (seed = {}) => ({ ...seed });

const mockUserService = {
  getUserById: jest.fn(async (id) => ({ id, email: `${id}@example.com`, name: 'Test User' })),
  getUserByEmail: jest.fn(async (email) => ({ id: 'user-1', email })),
  createUser: jest.fn(async (payload) => ({ id: 'user-1', ...payload })),
  updateUser: jest.fn(async (id, fields) => ({ id, ...fields })),
  deleteUser: jest.fn(async (id) => true),
  listUsers: jest.fn(async () => []),
  updateLastLogin: jest.fn(async (id) => true),
};

const mockNoteService = {
  getNoteById: jest.fn(async (noteId, userId) => ({ id: noteId, user_id: userId, title: 'Note', original_image_url: 's3://bucket/key.jpg', processing_status: 'pending' })),
  getNotesByUserId: jest.fn(async (userId, opts) => ([])),
  createNote: jest.fn(async (userId, payload) => ({ id: 'note-1', user_id: userId, ...payload })),
  updateNote: jest.fn(async (noteId, userId, fields) => ({ id: noteId, ...fields })),
  deleteNote: jest.fn(async (noteId, userId) => true),
  updateNoteProcessingStatus: jest.fn(async (noteId, status, errMsg = null) => ({ id: noteId, processing_status: status, error_message: errMsg })),
  markProcessing: jest.fn(async (noteId, taskId) => ({ processing_task_id: taskId })),
  getProcessingStatus: jest.fn(async (noteId, userId) => ({ processing_task_id: 'task-1', processing_status: 'processing' })),
  attachAIResult: jest.fn(async (noteId, userId, result) => ({ id: noteId, parsed_content: result.parsed_content })),
};

const mockSubjectService = {
  getSubjectById: jest.fn(async (id) => ({ id })),
  getSubjectsByUserId: jest.fn(async (userId) => []),
  createSubject: jest.fn(async (userId, payload) => ({ id: 'sub-1', ...payload })),
  updateSubject: jest.fn(async (id, userId, fields) => ({ id, ...fields })),
  deleteSubject: jest.fn(async (id, userId) => true),
  getSubjectWithStats: jest.fn(async () => ({})),
};

const mockChapterService = {
  getChapterById: jest.fn(async (id) => ({ id })),
  getChaptersBySubjectId: jest.fn(async (subjectId) => []),
  createChapter: jest.fn(async (userId, payload) => ({ id: 'ch-1', ...payload })),
  updateChapter: jest.fn(async (id, userId, fields) => ({ id, ...fields })),
  deleteChapter: jest.fn(async (id, userId) => true),
};

const mockFlashcardService = {
  getFlashcardById: jest.fn(async (id, userId) => ({ id, user_id: userId, question: 'Q', answer: 'A' })),
  countFlashcardsByUser: jest.fn(async (userId) => 0),
  getFlashcardsByUserId: jest.fn(async (userId, opts) => []),
  getFlashcardsByNoteId: jest.fn(async (noteId, userId) => []),
  getDueFlashcards: jest.fn(async (userId, opts) => []),
  createFlashcard: jest.fn(async (noteId, userId, payload) => ({ id: 'fc-1', note_id: noteId, user_id: userId, ...payload })),
  createFlashcardsFromAI: jest.fn(async (noteId, userId, arr) => arr.map((f, i) => ({ id: `fc-${i}`, note_id: noteId, user_id: userId, ...f }))),
  updateFlashcard: jest.fn(async (id, userId, fields) => ({ id, ...fields })),
  deleteFlashcard: jest.fn(async (id, userId) => true),
  reviewFlashcard: jest.fn(async (id, userId, quality, time_spent_seconds = null) => ({ flashcard: { id, interval: 1, repetitions: 1, easiness_factor: 2.5 }, session: { id: 's1' } })),
  getStudyStats: jest.fn(async (userId, opts) => ({ total: 0, due: 0, reviews: 0, avg_quality: 0, study_time: 0 })),
};

const mockS3Service = {
  generatePresignedUploadUrl: jest.fn(async (key, ct) => ({ uploadUrl: `https://s3/${key}`, key })),
  generatePresignedDownloadUrl: jest.fn(async (key) => ({ downloadUrl: `https://s3/${key}`, key })),
  checkFileExists: jest.fn(async (key) => true),
  deleteFile: jest.fn(async (key) => true),
  validateFileUpload: jest.fn((file) => ({ valid: true })),
  generateUniqueKey: jest.fn((userId, filename) => `notes/${userId}/${filename}`),
};

module.exports = {
  mockUserService,
  mockNoteService,
  mockSubjectService,
  mockChapterService,
  mockFlashcardService,
  mockS3Service,
};
