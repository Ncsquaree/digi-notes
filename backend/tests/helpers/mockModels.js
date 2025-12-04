// Centralized model mocks

const mockUser = {
  findById: jest.fn(async (id) => ({ id, email: `${id}@example.com`, name: 'Test User' })),
  findByEmail: jest.fn(async (email) => ({ id: 'user-1', email })),
  create: jest.fn(async (payload) => ({ id: 'user-1', ...payload })),
  update: jest.fn(async (id, fields) => ({ id, ...fields })),
  delete: jest.fn(async (id) => true),
  list: jest.fn(async () => []),
  updateLastLogin: jest.fn(async (id) => true),
};

const mockNote = {
  findById: jest.fn(async (id) => ({ id, user_id: 'user-1', title: 'Note', original_image_url: 's3://bucket/key.jpg', processing_status: 'pending' })),
  findByUserId: jest.fn(async (userId, opts) => []),
  create: jest.fn(async (payload) => ({ id: 'note-1', ...payload })),
  update: jest.fn(async (id, fields) => ({ id, ...fields })),
  delete: jest.fn(async (id) => true),
  updateProcessingStatus: jest.fn(async (id, status, errMsg) => ({ id, processing_status: status, error_message: errMsg })),
};

const mockSubject = {
  findById: jest.fn(async (id) => ({ id })),
  findByUserId: jest.fn(async (userId) => []),
  findByUserAndName: jest.fn(async () => null),
  create: jest.fn(async (payload) => ({ id: 's-1', ...payload })),
  update: jest.fn(async (id, fields) => ({ id, ...fields })),
  delete: jest.fn(async (id) => true),
};

const mockChapter = {
  findById: jest.fn(async (id) => ({ id })),
  findBySubjectId: jest.fn(async (sid) => []),
  countBySubjectId: jest.fn(async (sid) => 0),
  create: jest.fn(async (payload) => ({ id: 'c-1', ...payload })),
  update: jest.fn(async (id, fields) => ({ id, ...fields })),
  delete: jest.fn(async (id) => true),
};

const mockFlashcard = {
  findById: jest.fn(async (id) => ({ id, question: 'Q', answer: 'A' })),
  findByUserId: jest.fn(async (userId, opts) => []),
  findByNoteId: jest.fn(async (noteId) => []),
  findDueForReview: jest.fn(async (userId, opts) => []),
  countByUserId: jest.fn(async (userId) => 0),
  create: jest.fn(async (payload) => ({ id: 'fc-1', ...payload })),
  bulkCreate: jest.fn(async (arr) => arr.map((f, i) => ({ id: `fc-${i}`, ...f }))),
  update: jest.fn(async (id, fields) => ({ id, ...fields })),
  delete: jest.fn(async (id) => true),
};

const mockStudySession = {
  create: jest.fn(async (payload) => ({ id: 'ss-1', ...payload })),
  findByUserId: jest.fn(async (userId) => []),
  getStats: jest.fn(async (userId, opts) => ({ total: 0 })),
};

const mockKnowledgeGraphNode = {
  create: jest.fn(async (payload) => ({ id: 'n-1', ...payload })),
  bulkCreate: jest.fn(async (arr) => arr.map((n, i) => ({ id: `n-${i}`, ...n }))),
  findByNoteId: jest.fn(async (noteId) => []),
};

const mockRefreshToken = {
  create: jest.fn(async (payload) => ({ id: 'rt-1', ...payload })),
  findByTokenHash: jest.fn(async (hash) => null),
  revoke: jest.fn(async (hash) => true),
};

module.exports = {
  mockUser,
  mockNote,
  mockSubject,
  mockChapter,
  mockFlashcard,
  mockStudySession,
  mockKnowledgeGraphNode,
  mockRefreshToken,
};
