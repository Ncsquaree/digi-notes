const authMiddleware = require('../../../src/middleware/authenticate');
const authUtils = require('../../../src/utils/auth');

jest.mock('../../../src/utils/auth');

const authenticate = authMiddleware.default || authMiddleware || authMiddleware.authenticate || authMiddleware;
const optionalAuth = authMiddleware.optional;

describe('authenticate middleware', () => {
  afterEach(() => jest.clearAllMocks());

  it('calls next with UnauthorizedError when header missing', async () => {
    const req = { headers: {} };
    const res = {};
    const next = jest.fn();

    await authenticate(req, res, next);
    expect(next).toHaveBeenCalled();
    const err = next.mock.calls[0][0];
    expect(err).toBeDefined();
    expect(err.name).toBe('UnauthorizedError');
  });

  it('calls next with UnauthorizedError when header malformed', async () => {
    const req = { headers: { authorization: 'Basic abc' } };
    const res = {};
    const next = jest.fn();

    await authenticate(req, res, next);
    const err = next.mock.calls[0][0];
    expect(err.name).toBe('UnauthorizedError');
  });

  it('sets req.user and calls next on valid token', async () => {
    const token = 't';
    const req = { headers: { authorization: `Bearer ${token}` }, id: 'rid' };
    const res = {};
    const next = jest.fn();
    authUtils.verifyAccessToken.mockReturnValue({ userId: 'u1', email: 'u@example.com' });

    await authenticate(req, res, next);
    expect(req.user).toEqual({ userId: 'u1', email: 'u@example.com' });
    expect(next).toHaveBeenCalledWith();
  });

  it('maps TokenExpiredError to UnauthorizedError with TOKEN_EXPIRED code', async () => {
    const token = 't';
    const req = { headers: { authorization: `Bearer ${token}` }, id: 'rid' };
    const res = {};
    const next = jest.fn();
    const err = new Error('jwt expired');
    err.name = 'TokenExpiredError';
    authUtils.verifyAccessToken.mockImplementation(() => { throw err; });

    await authenticate(req, res, next);
    const calledErr = next.mock.calls[0][0];
    expect(calledErr).toBeDefined();
    expect(calledErr.name).toBe('UnauthorizedError');
    expect(calledErr.code).toBe('TOKEN_EXPIRED');
  });

  it('maps JsonWebTokenError to UnauthorizedError with code', async () => {
    const token = 't';
    const req = { headers: { authorization: `Bearer ${token}` }, id: 'rid' };
    const res = {};
    const next = jest.fn();
    const err = new Error('invalid token');
    err.name = 'JsonWebTokenError';
    authUtils.verifyAccessToken.mockImplementation(() => { throw err; });

    await authenticate(req, res, next);
    const calledErr = next.mock.calls[0][0];
    expect(calledErr.name).toBe('UnauthorizedError');
    expect(calledErr.code).toBe('JsonWebTokenError');
  });
});

describe('optionalAuth middleware', () => {
  afterEach(() => jest.clearAllMocks());

  it('optionalAuth sets req.user = null when no header', async () => {
    const req = { headers: {} };
    const res = {};
    const next = jest.fn();

    await optionalAuth(req, res, next);
    expect(req.user).toBeNull();
    expect(next).toHaveBeenCalled();
  });

  it('optionalAuth sets req.user when valid token', async () => {
    const token = 't';
    const req = { headers: { authorization: `Bearer ${token}` } };
    const res = {};
    const next = jest.fn();
    authUtils.verifyAccessToken.mockReturnValue({ userId: 'u1', email: 'u@example.com' });

    await optionalAuth(req, res, next);
    expect(req.user).toEqual({ userId: 'u1', email: 'u@example.com' });
    expect(next).toHaveBeenCalled();
  });

  it('optionalAuth swallows invalid token and sets req.user null', async () => {
    const token = 't';
    const req = { headers: { authorization: `Bearer ${token}` } };
    const res = {};
    const next = jest.fn();
    const err = new Error('bad');
    err.name = 'JsonWebTokenError';
    authUtils.verifyAccessToken.mockImplementation(() => { throw err; });

    await optionalAuth(req, res, next);
    expect(req.user).toBeNull();
    expect(next).toHaveBeenCalled();
  });
});
