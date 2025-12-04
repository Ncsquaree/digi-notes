jest.mock('../../../src/models', () => ({ User: require('../../helpers/mockModels').mockUser }));
const UserService = require('../../../src/services/user.service');
const { NotFoundError, ConflictError } = require('../../../src/utils/errors');

describe('UserService', () => {
  afterEach(() => jest.clearAllMocks());

  it('getUserById returns user when exists', async () => {
    const u = await UserService.getUserById('user-1');
    expect(u.id).toBe('user-1');
  });

  it('throws NotFoundError when missing', async () => {
    const models = require('../../../src/models');
    models.User.findById.mockResolvedValueOnce(null);
    await expect(UserService.getUserById('missing')).rejects.toBeInstanceOf(NotFoundError);
  });

  it('createUser throws ConflictError when email exists', async () => {
    const models = require('../../../src/models');
    models.User.findByEmail.mockResolvedValueOnce({ id: 'user-1' });
    await expect(UserService.createUser({ email: 'x@x.com', password: 'p' })).rejects.toBeInstanceOf(ConflictError);
  });
});
