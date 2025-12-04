jest.mock('../../../src/models', () => ({ Chapter: require('../../helpers/mockModels').mockChapter, Subject: require('../../helpers/mockModels').mockSubject }));
const ChapterService = require('../../../src/services/chapter.service');
const { NotFoundError, ForbiddenError } = require('../../../src/utils/errors');

describe('ChapterService', () => {
  afterEach(() => jest.clearAllMocks());

  it('getChapterById returns chapter', async () => {
    const c = await ChapterService.getChapterById('c1');
    expect(c.id).toBe('c1');
  });

  it('createChapter returns created chapter', async () => {
    const c = await ChapterService.createChapter('user-1', { title: 'Intro' });
    expect(c).toBeDefined();
  });
});
