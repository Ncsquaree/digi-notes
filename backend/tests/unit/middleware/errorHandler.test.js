const errorHandler = require('../../../src/middleware/errorHandler');
const { ValidationError, NotFoundError, UnauthorizedError, ForbiddenError } = require('../../../src/utils/errors');

describe('errorHandler middleware', () => {
  let req, res, next;
  beforeEach(() => {
    req = { id: 'rid', method: 'GET', originalUrl: '/test' };
    res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
    next = jest.fn();
    process.env.NODE_ENV = 'test';
  });

  it('formats ValidationError with validationErrors array and 400 status', () => {
    const err = new ValidationError('Bad', [{ field: 'a', message: 'err', value: 'x' }]);
    errorHandler(err, req, res, next);
    expect(res.status).toHaveBeenCalledWith(400);
    const payload = res.json.mock.calls[0][0];
    expect(payload.success).toBe(false);
    expect(payload.error.validationErrors).toBeDefined();
    expect(payload.error.message).toBe('Bad');
  });

  it('maps UnauthorizedError to 401', () => {
    const err = new UnauthorizedError('no token');
    errorHandler(err, req, res, next);
    expect(res.status).toHaveBeenCalledWith(401);
    const payload = res.json.mock.calls[0][0];
    expect(payload.error.code).toBe(err.code || err.name);
  });

  it('maps NotFoundError to 404', () => {
    const err = new NotFoundError('missing');
    errorHandler(err, req, res, next);
    expect(res.status).toHaveBeenCalledWith(404);
  });

  it('hides details in production for 500 errors', () => {
    process.env.NODE_ENV = 'production';
    const err = new Error('boom');
    errorHandler(err, req, res, next);
    expect(res.status).toHaveBeenCalledWith(500);
    const payload = res.json.mock.calls[0][0];
    expect(payload.error.message).toBe('Internal server error');
    expect(payload.error.details).toBeUndefined();
  });

  it('includes details in non-production', () => {
    process.env.NODE_ENV = 'development';
    const err = new Error('boom2');
    errorHandler(err, req, res, next);
    expect(res.status).toHaveBeenCalledWith(500);
    const payload = res.json.mock.calls[0][0];
    expect(payload.error.details).toBeDefined();
    expect(payload.error.details.status).toBe(500);
  });
});
