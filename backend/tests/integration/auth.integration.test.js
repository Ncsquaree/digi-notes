const request = require('supertest');
const express = require('express');
const bodyParser = require('body-parser');
const authRoutes = require('../../src/routes/auth');
const db = require('../../src/config/database');
const { generateTestToken } = require('../setup');

jest.mock('../../src/config/database');

function makeApp() {
  const app = express();
  app.use(bodyParser.json());
  app.use('/api/auth', authRoutes);
  // central error handler as in app
  app.use((err, req, res, next) => { res.status(err.statusCode || 500).json({ success: false, error: { message: err.message } }); });
  return app;
}

describe('auth integration', () => {
  beforeEach(() => jest.clearAllMocks());

  it('POST /register succeeds and returns tokens', async () => {
    const app = makeApp();
    // register flow: first query checks existing email -> none, second returns created user
    db.pool = { query: jest.fn()
      .mockResolvedValueOnce({ rowCount: 0, rows: [] }) // existing check
      .mockResolvedValueOnce({ rowCount: 1, rows: [{ id: 'u1', email: 'u@example.com', first_name: 'A', last_name: 'B' }] }) // insert user
      .mockResolvedValueOnce({}) // insert refresh token
    };

    const res = await request(app).post('/api/auth/register').send({ email: 'u@example.com', password: 'Password123!', firstName: 'A', lastName: 'B' });
    expect(res.status).toBe(201);
    expect(res.body.success).toBe(true);
    expect(res.body.data.accessToken).toBeDefined();
    expect(res.body.data.refreshToken).toBeDefined();
  });

  it('POST /login returns 200 and tokens on success', async () => {
    const app = makeApp();
    db.pool = { query: jest.fn()
      .mockResolvedValueOnce({ rowCount: 1, rows: [{ id: 'u1', email: 'u@example.com', password_hash: '$2a$10$hash', is_active: true }] }) // select user
      .mockResolvedValueOnce({}) // update last_login
      .mockResolvedValueOnce({}) // insert refresh token
    };

    // mock comparePassword to accept
    const authUtils = require('../../src/utils/auth');
    jest.spyOn(authUtils, 'comparePassword').mockResolvedValue(true);

    const res = await request(app).post('/api/auth/login').send({ email: 'u@example.com', password: 'Password123!' });
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.data.accessToken).toBeDefined();
  });

  it('GET /me requires auth and returns current user', async () => {
    const app = makeApp();
    const token = generateTestToken('u1', 'u@example.com');
    db.pool = { query: jest.fn().mockResolvedValueOnce({ rowCount: 1, rows: [{ id: 'u1', email: 'u@example.com', first_name: 'A', last_name: 'B', created_at: new Date(), last_login: null, email_verified: false }] }) };
    const res = await request(app).get('/api/auth/me').set('Authorization', `Bearer ${token}`);
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.data.user.id).toBe('u1');
  });

  it('POST /refresh-token returns 200 on valid refresh token', async () => {
    const app = makeApp();
    // simulate verifyRefreshToken by generating a real refresh token
    const authUtils = require('../../src/utils/auth');
    const user = { id: 'u1', email: 'u@example.com' };
    const refresh = authUtils.generateRefreshToken(user);
    // token lookup
    db.pool = { query: jest.fn()
      .mockResolvedValueOnce({ rowCount: 1, rows: [{ id: 'rt1', user_id: 'u1', revoked: false, expires_at: null }] }) // token row
      .mockResolvedValueOnce({ rowCount: 1, rows: [{ id: 'u1', email: 'u@example.com' }] }) // user
      .mockResolvedValueOnce({}) // revoke
      .mockResolvedValueOnce({}) // insert new token
    };

    const res = await request(app).post('/api/auth/refresh-token').send({ refreshToken: refresh });
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.data.accessToken).toBeDefined();
  });

  it('POST /logout requires auth and revokes token', async () => {
    const app = makeApp();
    const token = generateTestToken('u1', 'u@example.com');
    db.pool = { query: jest.fn().mockResolvedValueOnce({ rowCount: 1 }) };
    const res = await request(app).post('/api/auth/logout').set('Authorization', `Bearer ${token}`).send({ refreshToken: 'r' });
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
  });
});
