jest.mock('../../../src/models', () => ({ Subject: require('../../helpers/mockModels').mockSubject }));
const SubjectService = require('../../../src/services/subject.service');
const { NotFoundError, ForbiddenError } = require('../../../src/utils/errors');

describe('SubjectService', () => {
  afterEach(() => jest.clearAllMocks());

  it('getSubjectById returns subject', async () => {
    const s = await SubjectService.getSubjectById('s1');
    expect(s.id).toBe('s1');
  });

  it('createSubject returns created subject', async () => {
    const s = await SubjectService.createSubject('user-1', { name: 'Math' });
    expect(s).toBeDefined();
  });
});
