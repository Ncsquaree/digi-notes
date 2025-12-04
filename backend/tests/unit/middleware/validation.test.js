const { validate, isUUID, isEmail, isString, isOptionalString, rejectUnknownBody } = require('../../../src/middleware/validation');
const { ValidationError } = require('../../../src/utils/errors');

describe('validation middleware', () => {
  afterEach(() => { jest.clearAllMocks(); process.env.VALIDATION_STRICT_MODE = 'false'; });

  it('passes when no errors', async () => {
    const req = { body: {} };
    const res = {};
    const next = jest.fn();
    const chain = [];
    await validate(chain)(req, res, next);
    expect(next).toHaveBeenCalled();
  });

  it('returns ValidationError with multiple field errors', async () => {
    const req = { params: { id: 'not-uuid' }, body: { email: 'bad', title: 'a' } };
    const res = {};
    const next = jest.fn();
    const mw = validate([isUUID('id'), isEmail('email'), isString('title', 3, 100)]);

    await mw(req, res, next);
    expect(next).toHaveBeenCalled();
    const err = next.mock.calls[0][0];
    expect(err).toBeInstanceOf(ValidationError);
    expect(Array.isArray(err.errors)).toBe(true);
    expect(err.errors.find(e => e.field === 'id')).toBeDefined();
    expect(err.errors.find(e => e.field === 'email')).toBeDefined();
  });

  it('allows optional string when omitted', async () => {
    const req = { body: {} };
    const res = {};
    const next = jest.fn();
    const mw = validate([isOptionalString('note')]);

    await mw(req, res, next);
    expect(next).toHaveBeenCalledWith();
  });

  it('rejectUnknownBody rejects unknown fields when strict', async () => {
    process.env.VALIDATION_STRICT_MODE = 'true';
    const req = { body: { allowed: 'x', extra: 'y' } };
    const res = {};
    const next = jest.fn();
    const middleware = rejectUnknownBody(['allowed']);
    await middleware(req, res, next);
    expect(next).toHaveBeenCalled();
    const err = next.mock.calls[0][0];
    expect(err).toBeInstanceOf(ValidationError);
    expect(err.errors[0].field).toBe('extra');
  });
});
