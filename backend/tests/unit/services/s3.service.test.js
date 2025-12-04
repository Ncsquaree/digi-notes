jest.mock('../../../src/models');
const S3Service = require('../../../src/services/s3.service');

describe('S3Service', () => {
  afterEach(() => jest.clearAllMocks());

  it('generatePresignedUploadUrl returns url and key', async () => {
    const r = await S3Service.generatePresignedUploadUrl('notes/u1/file.png', 'image/png');
    expect(r.uploadUrl).toBeDefined();
    expect(r.key).toBeDefined();
  });

  it('validateFileUpload returns valid for known good file', () => {
    const info = S3Service.validateFileUpload({ key: 'a.png', contentType: 'image/png', size: 1024 });
    expect(info.valid).toBe(true);
  });
});
