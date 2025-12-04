const S3Service = require('../services/s3.service');
const logger = require('../utils/logger');
const { ValidationError, InternalError } = require('../utils/errors');

module.exports = {
  async getPresignedUploadUrl(req, res, next) {
    try {
      const userId = req.user.id;
      const { filename, contentType } = req.body;
      if (!filename) throw new ValidationError('filename is required');
      if (!contentType) throw new ValidationError('contentType is required');

      const key = await S3Service.generateUniqueKey(userId, filename);
      const { uploadUrl, expiresIn } = await S3Service.generatePresignedUploadUrl(key, contentType);

      logger.info('s3_presign_generated', { userId, key });
      return res.json({ success: true, uploadUrl, key, expiresIn });
    } catch (err) {
      logger.logError(err, { fn: 'getPresignedUploadUrl' });
      next(err);
    }
  }
};
