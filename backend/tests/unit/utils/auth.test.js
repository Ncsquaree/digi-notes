const auth = require('../../../src/utils/auth');

describe('utils/auth', () => {
  afterEach(() => jest.clearAllMocks());

  it('hash and compare password', async () => {
    const pw = 'secret123';
    const hash = await auth.hashPassword(pw);
    const ok = await auth.comparePassword(pw, hash);
    expect(ok).toBe(true);
  });

  it('parse expiry strings correctly', () => {
    expect(auth.parseExpiryToSeconds('15m')).toBe(900);
    expect(auth.parseExpiryToSeconds('1h')).toBe(3600);
    expect(auth.parseExpiryToSeconds('30d')).toBe(30 * 24 * 3600);
    expect(auth.parseExpiryToSeconds('3600s')).toBe(3600);
  });

  it('generate and verify token pair', () => {
    const user = { id: 'u1', email: 'a@b.com' };
    const pair = auth.generateTokenPair(user);
    expect(pair.accessToken).toBeDefined();
    expect(pair.refreshToken).toBeDefined();

    const payload = auth.verifyAccessToken(pair.accessToken);
    expect(payload.userId).toBe('u1');
  });

  it('verifyAccessToken throws for expired token', async () => {
    const original = process.env.JWT_EXPIRES_IN;
    process.env.JWT_EXPIRES_IN = '1s';
    const token = auth.generateAccessToken({ id: 'u2', email: 'a@b.com' });
    // wait for expiry
    await new Promise((r) => setTimeout(r, 1100));
    expect(() => auth.verifyAccessToken(token)).toThrow();
    process.env.JWT_EXPIRES_IN = original;
  });
  
  it('generate and verify refresh token and hashToken deterministic', () => {
    const user = { id: 'u9', email: 'r@e.com' };
    const refresh = auth.generateRefreshToken(user);
    const payload = auth.verifyRefreshToken(refresh);
    expect(payload.userId).toBe('u9');
    expect(payload.tokenId).toBeDefined();

    const hash1 = auth.hashToken('sometoken');
    const hash2 = auth.hashToken('sometoken');
    expect(hash1).toBe(hash2);
    expect(typeof hash1).toBe('string');
  });
});
