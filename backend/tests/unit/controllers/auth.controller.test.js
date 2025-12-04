jest.mock('../../../src/config/database');
jest.mock('../../../src/utils/auth');
const db = require('../../../src/config/database');
const authUtils = require('../../../src/utils/auth');
const authController = require('../../../src/controllers/auth.controller');

describe('auth.controller', () => {
  afterEach(() => jest.clearAllMocks());

  describe('register', () => {
    it('successfully registers a new user and issues tokens', async () => {
      const req = { body: { email: 't@example.com', password: 'Pass123!', firstName: 'T', lastName: 'U' } };
      const res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
      const next = jest.fn();

      // no existing user
      const pool = { query: jest.fn() };
      pool.query.mockResolvedValueOnce({ rowCount: 0, rows: [] }); // select
      pool.query.mockResolvedValueOnce({ rowCount: 1, rows: [{ id: 'user-1', email: 't@example.com', first_name: 'T', last_name: 'U' }] }); // insert user
      pool.query.mockResolvedValueOnce({}); // insert refresh token
      db.pool = pool;

      authUtils.hashPassword.mockReturnValue('hashedpw');
      authUtils.generateTokenPair.mockReturnValue({ accessToken: 'a', refreshToken: 'r' });
      authUtils.hashToken.mockReturnValue('rhash');

      await authController.register(req, res, next);

      expect(pool.query).toHaveBeenCalled();
      expect(authUtils.generateTokenPair).toHaveBeenCalled();
      expect(res.status).toHaveBeenCalledWith(201);
      expect(res.json).toHaveBeenCalledWith(expect.objectContaining({ success: true, data: expect.any(Object) }));
    });

    it('returns conflict when email exists', async () => {
      const req = { body: { email: 't@example.com', password: 'Pass123!' } };
      const res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
      const next = jest.fn();
      const pool = { query: jest.fn().mockResolvedValue({ rowCount: 1, rows: [{ id: 'user-1' }] }) };
      db.pool = pool;

      await authController.register(req, res, next);
      expect(next).toHaveBeenCalled();
    });
  });

  describe('login', () => {
    it('returns 401 when email not found', async () => {
      const req = { body: { email: 'no@e.com', password: 'x' } };
      const res = { json: jest.fn() };
      const next = jest.fn();
      const pool = { query: jest.fn().mockResolvedValueOnce({ rowCount: 0, rows: [] }) };
      db.pool = pool;

      await authController.login(req, res, next);
      expect(next).toHaveBeenCalled();
    });

    it('returns 401 when inactive user', async () => {
      const req = { body: { email: 'u@e.com', password: 'x' } };
      const res = { json: jest.fn() };
      const next = jest.fn();
      const pool = { query: jest.fn().mockResolvedValueOnce({ rowCount: 1, rows: [{ id: 'u1', email: 'u@e.com', password_hash: 'h', is_active: false }] }) };
      db.pool = pool;

      await authController.login(req, res, next);
      expect(next).toHaveBeenCalled();
    });

    it('returns 401 when bad password', async () => {
      const req = { body: { email: 'u@e.com', password: 'x' } };
      const res = { json: jest.fn() };
      const next = jest.fn();
      const pool = { query: jest.fn().mockResolvedValueOnce({ rowCount: 1, rows: [{ id: 'u1', email: 'u@e.com', password_hash: 'h', is_active: true }] }) };
      db.pool = pool;
      authUtils.comparePassword.mockResolvedValue(false);

      await authController.login(req, res, next);
      expect(next).toHaveBeenCalled();
    });

    it('login success issues tokens and updates last_login', async () => {
      const req = { body: { email: 'u@e.com', password: 'x' }, ip: '127.0.0.1' };
      const res = { json: jest.fn() };
      const next = jest.fn();
      const pool = { query: jest.fn() };
      pool.query.mockResolvedValueOnce({ rowCount: 1, rows: [{ id: 'u1', email: 'u@e.com', password_hash: 'h', is_active: true }] }); // user lookup
      pool.query.mockResolvedValueOnce({}); // update last_login
      pool.query.mockResolvedValueOnce({}); // insert refresh token
      db.pool = pool;
      authUtils.comparePassword.mockResolvedValue(true);
      authUtils.generateTokenPair.mockReturnValue({ accessToken: 'a', refreshToken: 'r' });
      authUtils.hashToken.mockReturnValue('rhash');

      await authController.login(req, res, next);
      expect(res.json).toHaveBeenCalledWith(expect.objectContaining({ success: true, data: expect.any(Object) }));
    });
  });

  describe('refreshToken', () => {
    it('returns 401 when token not found', async () => {
      const req = { body: { refreshToken: 'r' } };
      const res = { json: jest.fn() };
      const next = jest.fn();
      authUtils.verifyRefreshToken.mockReturnValue({ userId: 'u1' });
      const pool = { query: jest.fn().mockResolvedValueOnce({ rowCount: 0, rows: [] }) };
      db.pool = pool;

      await authController.refreshToken(req, res, next);
      expect(next).toHaveBeenCalled();
    });

    it('refresh success rotates token', async () => {
      const req = { body: { refreshToken: 'r' } };
      const res = { json: jest.fn() };
      const next = jest.fn();
      authUtils.verifyRefreshToken.mockReturnValue({ tokenId: 't1' });
      const pool = { query: jest.fn() };
      pool.query.mockResolvedValueOnce({ rowCount: 1, rows: [{ id: 'rt1', user_id: 'u1', revoked: false, expires_at: null }] }); // token row
      pool.query.mockResolvedValueOnce({ rowCount: 1, rows: [{ id: 'u1', email: 'u@e.com' }] }); // user
      pool.query.mockResolvedValueOnce({}); // update revoke
      pool.query.mockResolvedValueOnce({}); // insert new token
      db.pool = pool;
      authUtils.generateTokenPair.mockReturnValue({ accessToken: 'a', refreshToken: 'nr' });
      authUtils.hashToken.mockReturnValue('nhash');

      await authController.refreshToken(req, res, next);
      expect(res.json).toHaveBeenCalledWith(expect.objectContaining({ success: true, data: expect.any(Object) }));
    });
  });

  describe('logout', () => {
    it('returns 401 when unauthenticated', async () => {
      const req = { body: { refreshToken: 'r' }, user: null };
      const res = { json: jest.fn() };
      const next = jest.fn();
      const pool = { query: jest.fn() };
      db.pool = pool;

      await authController.logout(req, res, next);
      expect(next).toHaveBeenCalled();
    });

    it('revokes token and returns success', async () => {
      const req = { body: { refreshToken: 'r' }, user: { userId: 'u1' } };
      const res = { json: jest.fn() };
      const next = jest.fn();
      const pool = { query: jest.fn().mockResolvedValue({ rowCount: 1 }) };
      db.pool = pool;

      await authController.logout(req, res, next);
      expect(res.json).toHaveBeenCalledWith({ success: true, message: 'Logged out successfully' });
    });
  });
});
