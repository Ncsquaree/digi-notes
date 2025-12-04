const errors = require('../../../src/utils/errors');

describe('errors utils', () => {
  it('AppError sets properties', () => {
    const e = new errors.AppError('msg', 418, 'CODE');
    expect(e.statusCode).toBe(418);
    expect(e.code).toBe('CODE');
    expect(e.isOperational).toBe(true);
  });

  it('ValidationError default', () => {
    const e = new errors.ValidationError();
    expect(e.statusCode).toBe(400);
  });
});
