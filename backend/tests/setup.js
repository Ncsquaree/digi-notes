// Global test setup
// Load .env.test if present so env vars in that file take precedence
const path = require('path');
const dotenv = require('dotenv');
dotenv.config({ path: path.join(__dirname, '..', '.env.test') });
process.env.NODE_ENV = process.env.NODE_ENV || 'test';

// default test env vars (can be overridden by .env.test)
process.env.JWT_SECRET = process.env.JWT_SECRET || 'test-jwt-secret-min-32-chars-long-for-testing';
process.env.JWT_REFRESH_SECRET = process.env.JWT_REFRESH_SECRET || 'test-refresh-secret-min-32-chars-long-for-testing';
process.env.DATABASE_URL = process.env.DATABASE_URL || 'postgresql://test:test@localhost:5432/digi_notes_test';
process.env.REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379/1';
process.env.AWS_S3_BUCKET = process.env.AWS_S3_BUCKET || 'test-bucket';
process.env.AWS_REGION = process.env.AWS_REGION || 'us-east-1';
process.env.AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';
process.env.AI_SERVICE_TIMEOUT = process.env.AI_SERVICE_TIMEOUT || '5000';
process.env.LOG_LEVEL = process.env.LOG_LEVEL || 'error';
process.env.SANITIZE_HTML = process.env.SANITIZE_HTML || 'false';
process.env.VALIDATION_STRICT_MODE = process.env.VALIDATION_STRICT_MODE || 'false';

// silence winston logger during tests
jest.mock('../src/utils/logger', () => {
  const noop = () => {};
  const logger = {
    info: noop,
    warn: noop,
    error: noop,
    logError: noop,
    logRequest: noop,
    debug: noop,
  };
  return logger;
});

// export test helpers
const jwt = require('jsonwebtoken');

module.exports.generateTestToken = function generateTestToken(userId = 'test-user', email = 'test@example.com', opts = {}) {
  const payload = { userId, email };
  return jwt.sign(payload, process.env.JWT_SECRET, { expiresIn: process.env.JWT_EXPIRES_IN || '15m', ...opts });
};

module.exports.mockUser = function mockUser(overrides = {}) {
  return Object.assign({ id: 'user-1', email: 'u@example.com', name: 'Test User', is_active: true }, overrides);
};

module.exports.mockNote = function mockNote(overrides = {}) {
  return Object.assign({ id: 'note-1', user_id: 'user-1', title: 'Test Note', original_image_url: 's3://test-bucket/path/to/file.jpg', processing_status: 'pending' }, overrides);
};

module.exports.mockFlashcard = function mockFlashcard(overrides = {}) {
  return Object.assign({ id: 'fc-1', note_id: 'note-1', user_id: 'user-1', question: 'Q?', answer: 'A' }, overrides);
};
